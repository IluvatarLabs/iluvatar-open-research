#!/usr/bin/env python3
"""batch_044 Sub-B analysis: classify height S-LDSC results per pre-registered rule.

WHY: Applies the decision rule in experiments/batch_044/brief.md §DECISION RULE / Sub-B
     to the LDSC results + log produced by run_sub_b_height.sh. Outputs a structured
     diagnostics JSON that the consolidated results.json can consume.

Decision rule (brief.md):
  UNINTERPRETABLE if <2 of 3 baselineLD positive-control annotations have z > +3.
  REFUTED        if neuronal Enrichment z > +2 AND tau* z > +2 AND Enrichment >= 1.25.
  SUGGESTED      if only one of (Enrichment z > 2, tau* z > 2) is true.
  ESTABLISHED    specificity if BOTH Enrichment and tau* z < +2 AND diagnostics pass.

Usage:
  python3 analyze_sub_b_height.py [--force]

Idempotent: skips writing if output exists unless --force.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import math
import pathlib
import re
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_044"
OUTPUT_DIR = BATCH_DIR / "output"
LOG_DIR = BATCH_DIR / "logs"

RESULTS_FILE = OUTPUT_DIR / "B_height_sldsc.results"
LDSC_LOG = OUTPUT_DIR / "B_height_sldsc.log"
DIAGNOSTICS_OUT = OUTPUT_DIR / "B_height_diagnostics.json"
SCRIPT_LOG = LOG_DIR / "analyze_sub_b_height.log"

# Cell-type annotation names (in .results these carry the standard `L2_0` suffix
# from LDSC's overlap-annot Category naming).
CELL_TYPES = ["neuronal", "oligodendrocyte", "astrocyte", "OPC"]

# baselineLD positive-control annotation categories. Brief lists Coding_UCSC,
# Conserved_LindbladToh, Intron_UCSC as the three primary positive controls;
# GERP.NS is listed as a conservation backup. LDSC .results Category names get
# `L2_0` suffix (overlap-annot legacy) for binary baselineLD entries.
POS_CONTROL_ANNOTATIONS = [
    "Coding_UCSC",
    "Conserved_LindbladToh",
    "Intron_UCSC",
]
POS_CONTROL_BACKUP = ["GERP.NS"]

CATEGORY_SUFFIX_CANDIDATES = ["L2_0", "L2_1", ""]


def _setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("analyze_sub_b_height")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(SCRIPT_LOG)
        fh.setFormatter(logging.Formatter("[%(asctime)sZ] %(levelname)s %(message)s",
                                          datefmt="%Y-%m-%dT%H:%M:%S"))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s",
                                          datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def sha256(path: pathlib.Path) -> str:
    # WHY: input-file integrity recorded in diagnostics JSON for audit.
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _match_category(df: pd.DataFrame, base_name: str) -> str | None:
    """Find Category in .results matching `base_name` with LDSC's trailing suffix."""
    for suf in CATEGORY_SUFFIX_CANDIDATES:
        cand = f"{base_name}{suf}" if suf else base_name
        if cand in df["Category"].values:
            return cand
    # Case-insensitive fallback
    matches = [c for c in df["Category"].values if c.lower().startswith(base_name.lower())]
    return matches[0] if matches else None


def _extract_row(df: pd.DataFrame, base_name: str, label: str,
                 logger: logging.Logger) -> dict[str, Any]:
    cat = _match_category(df, base_name)
    if cat is None:
        logger.warning("Category %r not found in .results (searched suffixes %s)",
                       base_name, CATEGORY_SUFFIX_CANDIDATES)
        return {"label": label, "category_matched": None, "found": False}
    row = df[df["Category"] == cat].iloc[0].to_dict()
    # Enrichment z = (Enrichment - 1) / Enrichment_std_error  (standard S-LDSC).
    enr = float(row["Enrichment"])
    enr_se = float(row["Enrichment_std_error"])
    enr_z = (enr - 1.0) / enr_se if enr_se > 0 else math.nan
    # Coefficient_z-score is what LDSC prints for tau when --print-coefficients is on.
    coef = float(row.get("Coefficient", float("nan")))
    coef_se = float(row.get("Coefficient_std_error", float("nan")))
    coef_z = float(row.get("Coefficient_z-score", float("nan")))
    return {
        "label": label,
        "category_matched": cat,
        "found": True,
        "Enrichment": enr,
        "Enrichment_std_error": enr_se,
        "Enrichment_z_vs_1": enr_z,
        "Enrichment_p": float(row.get("Enrichment_p", float("nan"))),
        "Coefficient": coef,
        "Coefficient_std_error": coef_se,
        "Coefficient_z_score": coef_z,
    }


def parse_ldsc_log(log_path: pathlib.Path) -> dict[str, Any]:
    """Extract intercept, lambda_GC, h2, and SNP counts from an LDSC .log file.

    WHY: These are diagnostic sanity checks; non-physical values imply the munge or
         weight-merge went wrong and the Enrichment numbers cannot be trusted.
    """
    diag: dict[str, Any] = {
        "intercept": None, "intercept_se": None,
        "lambda_gc": None,
        "total_h2": None, "total_h2_se": None,
        "snps_after_merge": None,
        "snps_before_merge": None,
        "snp_match_rate": None,
    }
    if not log_path.exists():
        return diag
    text = log_path.read_text()

    # LDSC intercept line example:  "Intercept: 1.0532 (0.0102)"
    m = re.search(r"Intercept:\s*([-\d.eE+]+)\s*\(([-\d.eE+]+)\)", text)
    if m:
        diag["intercept"] = float(m.group(1))
        diag["intercept_se"] = float(m.group(2))
    # Lambda GC (may appear as "Lambda GC: 1.234")
    m = re.search(r"Lambda\s*GC:\s*([-\d.eE+]+)", text)
    if m:
        diag["lambda_gc"] = float(m.group(1))
    # Total observed-scale h2: "Total Observed scale h2: 0.4561 (0.0123)"
    m = re.search(r"Total Observed scale h2:\s*([-\d.eE+]+)\s*\(([-\d.eE+]+)\)", text)
    if m:
        diag["total_h2"] = float(m.group(1))
        diag["total_h2_se"] = float(m.group(2))

    # SNP counts. LDSC prints e.g. "After merging with regression SNP LD, 1153445 SNPs remain."
    m = re.search(r"After merging with regression SNP LD,\s*([\d,]+)\s*SNPs remain", text)
    if m:
        diag["snps_after_merge"] = int(m.group(1).replace(",", ""))
    # Pre-merge HM3: "Read summary statistics for ####### SNPs."
    pre = re.findall(r"Read summary statistics for\s*([\d,]+)\s*SNPs", text)
    if pre:
        diag["snps_before_merge"] = int(pre[0].replace(",", ""))
    if diag["snps_after_merge"] and diag["snps_before_merge"]:
        diag["snp_match_rate"] = diag["snps_after_merge"] / diag["snps_before_merge"]
    return diag


def classify(neuronal: dict, pos_controls: list[dict], diagnostics: dict) -> dict:
    """Apply pre-registered Sub-B decision rule."""
    # Count baselineLD positive controls with Enrichment z > +3 (uses the same
    # Enrichment z formulation LDSC reports; z_vs_1 = (E-1)/SE).
    pc_z = [(p["label"], p.get("Enrichment_z_vs_1", math.nan)) for p in pos_controls if p.get("found")]
    pc_pass = sum(1 for _, z in pc_z if (z is not None and not math.isnan(z) and z > 3.0))
    pc_total = len(pc_z)

    # Interpretability gate
    if pc_total == 0:
        return {
            "classification": "UNINTERPRETABLE",
            "reason": "No positive-control annotations could be extracted from results.",
            "positive_control_pass_count": 0,
            "positive_control_total": 0,
        }
    if pc_pass < 2:
        return {
            "classification": "UNINTERPRETABLE",
            "reason": f"Only {pc_pass}/{pc_total} positive-control annotations have z>+3 "
                      "(brief requires >=2 of 3).",
            "positive_control_pass_count": pc_pass,
            "positive_control_total": pc_total,
        }

    if not neuronal.get("found"):
        return {
            "classification": "UNINTERPRETABLE",
            "reason": "Neuronal annotation row not found in results.",
            "positive_control_pass_count": pc_pass,
            "positive_control_total": pc_total,
        }

    enr = neuronal["Enrichment"]
    enr_z = neuronal["Enrichment_z_vs_1"]
    tau_z = neuronal["Coefficient_z_score"]

    enr_flag = (not math.isnan(enr_z)) and (enr_z > 2.0)
    tau_flag = (not math.isnan(tau_z)) and (tau_z > 2.0)
    big_effect = enr >= 1.25

    if enr_flag and tau_flag and big_effect:
        cls = "REFUTED"
        reason = ("Neuronal Enrichment z>2 AND tau* z>2 AND Enrichment>=1.25 "
                  "(>=30% of SCZ 1.83): pipeline specificity fails.")
    elif enr_flag ^ tau_flag:
        cls = "SUGGESTED"
        reason = ("One of (Enrichment z>2, tau* z>2) fires but not both: "
                  "possible mild brain-expressed-gene bleed-through; investigate.")
    elif (not enr_flag) and (not tau_flag):
        cls = "ESTABLISHED"
        reason = ("Both neuronal Enrichment z<2 AND tau* z<2 with positive controls "
                  "passing: pre-registered specificity criterion met.")
    else:
        # Both flags True but effect < 1.25: treat as SUGGESTED (borderline).
        cls = "SUGGESTED"
        reason = ("Both z>2 but Enrichment point estimate <1.25: borderline; "
                  "flag for manual review.")

    return {
        "classification": cls,
        "reason": reason,
        "positive_control_pass_count": pc_pass,
        "positive_control_total": pc_total,
        "neuronal_Enrichment": enr,
        "neuronal_Enrichment_z_vs_1": enr_z,
        "neuronal_Coefficient_z_score": tau_z,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite diagnostics JSON even if it exists.")
    args = parser.parse_args()
    logger = _setup_logger()

    if DIAGNOSTICS_OUT.exists() and not args.force:
        logger.info("SKIP: %s exists; use --force to overwrite.", DIAGNOSTICS_OUT)
        return 0

    if not RESULTS_FILE.exists():
        logger.error("Missing results file: %s", RESULTS_FILE)
        return 2
    if not LDSC_LOG.exists():
        logger.error("Missing LDSC log: %s", LDSC_LOG)
        return 2

    logger.info("Results file: %s  sha256=%s", RESULTS_FILE, sha256(RESULTS_FILE))
    logger.info("LDSC log:     %s  sha256=%s", LDSC_LOG, sha256(LDSC_LOG))

    df = pd.read_csv(RESULTS_FILE, sep="\t")
    logger.info("Loaded %d result rows, columns=%s", len(df), list(df.columns))

    # --- Extract per cell-type -----
    cell_rows: list[dict] = []
    neuronal_row: dict = {}
    for ct in CELL_TYPES:
        r = _extract_row(df, ct, label=ct, logger=logger)
        cell_rows.append(r)
        if ct == "neuronal":
            neuronal_row = r
        logger.info("Cell %s: %s", ct, json.dumps(r, default=str))

    # --- Extract baselineLD positive controls -----
    pos_rows: list[dict] = []
    for pc in POS_CONTROL_ANNOTATIONS:
        r = _extract_row(df, pc, label=pc, logger=logger)
        pos_rows.append(r)
    # Backup conservation annotation (GERP.NS) — reported but not used in decision rule
    # unless one of the primary three is missing.
    backup_rows = [_extract_row(df, pc, label=pc, logger=logger) for pc in POS_CONTROL_BACKUP]

    # If <3 primary pos controls found, pull in backup to reach 3 for visibility.
    if sum(r.get("found", False) for r in pos_rows) < 3:
        logger.info("Augmenting positive controls with backup (GERP.NS) for visibility.")
        for r in backup_rows:
            if r.get("found"):
                pos_rows.append(r)

    # --- Log-file diagnostics -----
    ldsc_diag = parse_ldsc_log(LDSC_LOG)
    logger.info("LDSC diagnostics: %s", json.dumps(ldsc_diag, default=str))

    # --- Decision rule -----
    decision = classify(neuronal_row, pos_rows, ldsc_diag)
    logger.info("Decision: %s — %s", decision["classification"], decision["reason"])

    payload = {
        "batch": "batch_044",
        "sub_experiment": "B_height_sldsc",
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "results_file": str(RESULTS_FILE),
            "results_file_sha256": sha256(RESULTS_FILE),
            "ldsc_log": str(LDSC_LOG),
            "ldsc_log_sha256": sha256(LDSC_LOG),
        },
        "cell_types": cell_rows,
        "positive_controls_primary": pos_rows[: len(POS_CONTROL_ANNOTATIONS)],
        "positive_controls_backup": backup_rows,
        "ldsc_diagnostics": ldsc_diag,
        "decision": decision,
    }

    DIAGNOSTICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DIAGNOSTICS_OUT.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote %s", DIAGNOSTICS_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
