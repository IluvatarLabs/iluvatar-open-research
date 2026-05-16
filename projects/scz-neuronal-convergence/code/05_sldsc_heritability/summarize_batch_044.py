#!/usr/bin/env python3
"""Consolidate batch_044 Sub-B + Sub-C outputs into experiments/batch_044/output/results.json.

WHY: parent iteration doc + challenge.md consumers expect a single results.json per
     design.yaml deliverables. This script aggregates the per-sub-experiment JSONs
     without re-running anything; missing inputs are reported as null with a warning
     rather than synthesized.

Scope note (per parent-directive 2026-04-22): Sub-C tests 3 gene lists
(PGC3_Prioritised_EDT1, PGC3_Prioritised_SynGO, SCHEMA_exome_wide_significant).
The MAGMA top-1% list was dropped because MAGMA binary is not on PATH and
synthesizing gene scores would violate Rule 0.

Sub-A output (A_eas_diagnostics.json) is deferred to batch_045 per brief v3.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import pathlib
import sys

PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_044"
OUTPUT_DIR = BATCH_DIR / "output"
LOG_DIR = BATCH_DIR / "logs"

SUB_B_DIAG = OUTPUT_DIR / "B_height_diagnostics.json"
SUB_C_LOEUF_FISHER = OUTPUT_DIR / "C_loeuf_fisher.json"
SUB_C_LOEUF_LOGISTIC = OUTPUT_DIR / "C_loeuf_logistic.json"
SUB_C_HAR = OUTPUT_DIR / "C_har_fisher.json"
SUB_C_PERM = OUTPUT_DIR / "C_permutation_length_stratified.json"
SUB_C_SUMMARY = OUTPUT_DIR / "C_summary.json"
RESULTS_OUT = OUTPUT_DIR / "results.json"
SCRIPT_LOG = LOG_DIR / "summarize_batch_044.log"

# Canonical gene-list names (kept in sync with run_sub_c_constraint_har.py).
LIST_PGC3_EDT1 = "PGC3_Prioritised_EDT1"
LIST_PGC3_SYNGO = "PGC3_Prioritised_SynGO"
LIST_SCHEMA = "SCHEMA_exome_wide_significant"
SUB_C_GENE_LISTS = [LIST_PGC3_EDT1, LIST_PGC3_SYNGO, LIST_SCHEMA]


def _setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("summarize_batch_044")
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


def sha256(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path, logger: logging.Logger) -> dict | None:
    if not path.exists():
        logger.warning("Missing: %s", path)
        return None
    with path.open() as fh:
        data = json.load(fh)
    logger.info("Loaded %s (sha256=%s)", path, sha256(path))
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-generate results.json even if it exists.")
    args = parser.parse_args()

    logger = _setup_logger()
    if RESULTS_OUT.exists() and not args.force:
        logger.info("SKIP: %s exists; use --force.", RESULTS_OUT)
        return 0

    sub_b = load_json(SUB_B_DIAG, logger)
    sub_c_fisher = load_json(SUB_C_LOEUF_FISHER, logger)
    sub_c_logit = load_json(SUB_C_LOEUF_LOGISTIC, logger)
    sub_c_har = load_json(SUB_C_HAR, logger)
    sub_c_perm = load_json(SUB_C_PERM, logger)
    sub_c_summary = load_json(SUB_C_SUMMARY, logger)

    # Extract a compact Sub-C highlight block so downstream (challenge/review)
    # can read top-line outputs without re-parsing five JSONs.
    sub_c_highlights: dict = {}
    if sub_c_summary:
        sub_c_highlights = {
            "n_gene_lists_tested": (sub_c_summary.get("scope") or {}).get("n_gene_lists"),
            "gene_lists": (sub_c_summary.get("scope") or {}).get("gene_lists"),
            "gene_list_sizes": sub_c_summary.get("gene_list_sizes"),
            "schema_positive_control": sub_c_summary.get("schema_positive_control"),
            "decision": sub_c_summary.get("decision"),
            "loeuf_logistic_primary_list": sub_c_summary.get("loeuf_logistic_primary_list"),
        }

    results = {
        "batch": "batch_044",
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "scope_note": ("Sub-A (trans-ancestry EAS S-LDSC) deferred to batch_045 "
                       "per brief.md v3. This results.json covers Sub-B (height "
                       "specificity) and Sub-C (constraint/HAR) only. Sub-C tests "
                       "3 gene lists (MAGMA top-1% dropped: binary unavailable, "
                       "Rule 0 blocks synthesis)."),
        "sub_c_expected_gene_lists": SUB_C_GENE_LISTS,
        "sub_experiments": {
            "B_height_sldsc": {
                "classification": ((sub_b or {}).get("decision") or {}).get("classification"),
                "reason": ((sub_b or {}).get("decision") or {}).get("reason"),
                "diagnostics": sub_b,
            },
            "C_constraint_har": {
                "classification": ((sub_c_summary or {}).get("decision") or {}).get("classification"),
                "reason": ((sub_c_summary or {}).get("decision") or {}).get("reason"),
                "highlights": sub_c_highlights,
                "summary": sub_c_summary,
                "loeuf_fisher": sub_c_fisher,
                "loeuf_logistic": sub_c_logit,
                "har_fisher": sub_c_har,
                "length_stratified_permutation": sub_c_perm,
            },
        },
        "missing_artifacts": [
            str(p) for p in [SUB_B_DIAG, SUB_C_LOEUF_FISHER, SUB_C_LOEUF_LOGISTIC,
                             SUB_C_HAR, SUB_C_PERM, SUB_C_SUMMARY]
            if not p.exists()
        ],
    }

    with RESULTS_OUT.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Wrote %s", RESULTS_OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
