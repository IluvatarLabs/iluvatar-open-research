#!/usr/bin/env python3
"""
batch_053_A — F121/F124-synaptic Construct Robustness Re-Audit (v2).

This script audits whether the SynGO_EDT1 constraint signature (pLI OR ≈ 20.91)
reported for F121/F124-synaptic is a property of the 14-gene hand-curated anchor
only (`SynGO_EDT1_batch047`) or also holds on the broader arithmetic intersection
of ST12 rows with `SynGO.GeneSetMemb == 'yes'`.

Four gene lists are tested against 3 constraint metrics each:
    1. SynGO_EDT1_batch047  (14 hand-curated genes — original F121/F124 construct)
    2. SynGO_EDT1_ST12      (ST12 "SynGO yes" column, ~56 genes; arithmetic)
    3. SynGO_EDT1_intersection_14  (batch047 ∩ ST12)
    4. ST12_minus_hand      (ST12 − batch047; stress-test for hand-list dominance)

Constraint metrics: pLI >= 0.9, LOEUF <= 0.35, missense_z top decile.
Null model: length-stratified permutation within log1p(cds_length) deciles
(5000 perms, RNG_SEED=20260423) — the SAME code path batch_048 used to
produce the original F121/F124-synaptic OR=20.9 result.

HARD REPRODUCIBILITY GATE: `SynGO_EDT1_batch047` × `pLI >= 0.9` OR MUST fall
within 20.91 ± 5 (i.e. [15.91, 25.91]). Outside this window implies pipeline
mismatch and the script writes `challenge.md` and aborts BEFORE any
interpretation is written.

Why import-exactly from batch_048 (not re-implement):
    batch_048's `load_gnomad`, `load_pgc3_edt1`, `fisher_enrichment`,
    `length_perm_test`, and the `SYNOGO_EDT1_BATCH047` constant produced the
    original F121/F124 numbers. Re-implementing any of these risks exactly the
    construct-invalidity bug that iter_052 caught (a "reframe" accidentally
    changed the operational definition of the gene set). Per the
    `audit-import-exact-classifier` skill, the correct behaviour is to import
    verbatim and only add NEW logic (extra gene lists, missense_z metric,
    reproduction-gate output).

Usage:
    python3 experiments/batch_053_A/scripts/run_batch_053_A.py [--n-perm 5000]
    python3 experiments/batch_053_A/scripts/run_batch_053_A.py --smoke   # n_perm=100

Outputs (in experiments/batch_053_A/output/):
    gene_lists_used.json     — gene list membership + SHA256 (written BEFORE tests)
    reproduction_check.json  — batch_047 pLI OR reproduction gate (pass/fail)
    results.json             — full 4 × 3 test results
    challenge.md             — written ONLY if reproduction gate fails
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_053_A"
OUTPUT_DIR = BATCH_DIR / "output"
LOG_DIR = BATCH_DIR / "logs"

# Import verbatim from batch_048 scripts — do NOT re-implement.
# WHY: these functions produced F121/F124-synaptic numbers; importing guarantees
# reproduction of the original construct. (Cardinal rule #1 + audit skill.)
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "batch_048" / "scripts"))
from run_batch_048 import (  # noqa: E402  (deliberate sys.path insert)
    PLI_THRESHOLD,
    LOEUF_THRESHOLD,
    RNG_SEED,
    SYNOGO_EDT1_BATCH047,
    _or_from_masks,
    fisher_enrichment,
    length_perm_test,
    load_gnomad,
    load_pgc3_edt1,
)

# ----------------------------------------------------------------------------
# Pre-registered constants (all values have a cited source)
# ----------------------------------------------------------------------------
# Reproduction target — from experiments/batch_048/output/A_edt1_decomposition.json:
#   gene_list=SynGO_EDT1_batch047, constraint_metric="pLI >= 0.9", or=20.9050...
# Brief §DECISION RULE allows ±5 around 20.91.
REPRODUCTION_TARGET_OR = 20.91
REPRODUCTION_TOLERANCE = 5.0

# Default perm count (brief §MEASUREMENT; same as batch_048).
DEFAULT_N_PERM = 5000

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def _setup_logger(log_file: pathlib.Path) -> logging.Logger:
    """Configure INFO-level file + stdout logger."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_053_A")
    logger.setLevel(logging.INFO)
    # Clear any stale handlers (pytest / repeated runs).
    logger.handlers = []
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    for h in (logging.FileHandler(log_file, mode="w"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def _sha256_of_sorted(genes: list[str]) -> str:
    """SHA256 of sorted, newline-joined gene symbols — deterministic membership ID."""
    payload = "\n".join(sorted(set(genes))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _python_env_info() -> dict[str, str]:
    import scipy
    import statsmodels
    return {
        "python_version": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
    }


# ----------------------------------------------------------------------------
# Gene list construction — all four lists built per brief §WHAT.
# ----------------------------------------------------------------------------
def build_gene_lists(logger: logging.Logger,
                     edt1: dict[str, set[str]]) -> dict[str, list[str]]:
    """Build the 4 audit gene lists.

    WHY each list:
        - SynGO_EDT1_batch047: original hand-curated F121/F124 anchor; reproduces OR=20.91.
        - SynGO_EDT1_ST12:     arithmetic construct (ST12 rows with SynGO=yes); tests
                               whether broader column-based intersection carries signal.
        - SynGO_EDT1_intersection_14: hand list ∩ ST12 — documents which hand genes
                               are actually annotated in the ST12 SynGO column.
        - ST12_minus_hand:     ST12 − hand; tests whether signal survives without
                               the 14 canonical hand-picked genes (H053A3).
    """
    hand = list(SYNOGO_EDT1_BATCH047)
    st12 = sorted(edt1["EDT1_SynGO"])
    intersection = sorted(set(hand) & set(st12))
    diff = sorted(set(st12) - set(hand))

    gene_lists = {
        "SynGO_EDT1_batch047": sorted(set(hand)),
        "SynGO_EDT1_ST12": st12,
        "SynGO_EDT1_intersection_14": intersection,
        "ST12_minus_hand": diff,
    }

    for name, genes in gene_lists.items():
        logger.info("GENE_LIST %s: n=%d (sha256=%s)",
                    name, len(genes), _sha256_of_sorted(genes)[:16])

    return gene_lists


# ----------------------------------------------------------------------------
# Metric runners — exposes pLI, LOEUF, and missense_z-top-decile.
# ----------------------------------------------------------------------------
def _missense_z_top_decile_mask(gnomad: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Mask of genes in the TOP decile of mis.z_score (higher = more constrained).

    WHY top decile (not bottom): Karczewski 2020 — high missense_z indicates fewer
    observed than expected missense variants (i.e. missense-constrained). batch_047
    used a different (median-|z|) direction for syn_z; here we follow the brief's
    explicit 'missense_z top decile' spec, which corresponds to the 90th percentile
    of mis.z_score. This threshold is computed from the gnomAD background AFTER
    MHC exclusion to match the permutation universe.
    """
    bg = gnomad[~gnomad["mhc_indicator"]]
    vals = pd.to_numeric(bg["mis.z_score"], errors="coerce")
    # Top decile = >= 90th percentile of non-NaN values.
    threshold = float(np.nanpercentile(vals.to_numpy(), 90))
    # NaN → False so it is not considered "top decile".
    mask = (vals.fillna(-np.inf) >= threshold).to_numpy(dtype=bool)
    return mask, threshold


def _run_one_cell(logger: logging.Logger,
                  gene_list_name: str,
                  genes_in_list: set[str],
                  metric_name: str,
                  metric_mask: np.ndarray,
                  bg_genes_arr: np.ndarray,
                  bg_genes_set: set[str],
                  lengths: np.ndarray,
                  n_perm: int) -> dict[str, Any]:
    """Run Fisher + length-stratified permutation for a single (gene_list, metric) cell.

    Uses batch_048's `fisher_enrichment` and `length_perm_test` verbatim.
    The RNG is re-seeded per cell with RNG_SEED (matches batch_048 convention;
    any given list/metric is deterministic given this seed).
    """
    target_set = set(bg_genes_arr[metric_mask])
    # Intersect list with bg FIRST — matches batch_048 convention (see run_sub_a line
    # 407: `syngo_edt1 = set(SYNOGO_EDT1_BATCH047) & bg_genes_set`). Without this,
    # fisher_enrichment inflates `b` by counting list-genes absent from the gnomAD
    # universe as "in-list-but-not-target", depressing OR. (Caught at smoke-test:
    # full 14-gene list produced OR=8.36; bg-intersected 11-gene produced OR=20.91.)
    genes_in_bg = genes_in_list & bg_genes_set
    list_mask = np.array([g in genes_in_bg for g in bg_genes_arr], dtype=bool)

    n_in_list = int(list_mask.sum())  # genes in BOTH the list AND the bg universe
    if n_in_list < 3:
        logger.warning("CELL [%s × %s]: n_in_bg=%d < 3, skipping inferential test",
                       gene_list_name, metric_name, n_in_list)
        return {
            "gene_list": gene_list_name,
            "constraint_metric": metric_name,
            "or": None, "ci_low": None, "ci_high": None,
            "p_fisher": None, "emp_p": None,
            "n_in_list": n_in_list,
            "n_pli_hit": None,
            "n_bg": int(len(bg_genes_arr)),
            "classification": "UNINTERPRETABLE",
            "note": "n_in_bg < 3",
        }

    fisher = fisher_enrichment(genes_in_bg, bg_genes_set, target_set)
    perm = length_perm_test(
        list_mask, metric_mask, lengths, n_perm,
        np.random.default_rng(RNG_SEED),
    )

    n_pli_hit = int(fisher["a"])  # list ∩ target — generic "metric hit" count
    or_val = float(fisher["or"])
    emp_p = float(perm["emp_p"])

    logger.info(
        "CELL [%s × %s]: n_in_bg=%d n_hit=%d OR=%s p=%.4g emp_p=%.4g "
        "(null_mean=%.3f null_std=%.3f)",
        gene_list_name, metric_name,
        n_in_list, n_pli_hit,
        ("inf" if np.isinf(or_val) else f"{or_val:.3f}"),
        fisher["p"], emp_p,
        perm["null_mean"], perm["null_std"],
    )

    return {
        "gene_list": gene_list_name,
        "constraint_metric": metric_name,
        "or": or_val,
        "ci_low": float(fisher["ci_low"]),
        "ci_high": float(fisher["ci_high"]),
        "p_fisher": float(fisher["p"]),
        "emp_p": emp_p,
        "n_in_list": n_in_list,
        "n_pli_hit": n_pli_hit,
        "n_bg": int(fisher["n_bg"]),
        "null_mean_or": float(perm["null_mean"]),
        "null_std_or": float(perm["null_std"]),
        "n_perm_completed": int(perm["n_perm"]),
    }


# ----------------------------------------------------------------------------
# Classification rule — brief §DECISION RULE
# ----------------------------------------------------------------------------
def classify_cell(row: dict[str, Any]) -> str:
    """Map (gene_list, metric, OR, emp_p) → classification label.

    Classifications come from brief H053A2 (primary list robustness) and H053A3
    (hand-list-sole-carrier). Applied per-cell so the downstream manuscript can
    cross-tabulate primary (ST12) × secondary (ST12_minus_hand).

        ROBUST              : OR ≥ 10 AND emp_p < 0.01
        DILUTED             : 3 ≤ OR < 10 AND emp_p < 0.05
        HAND-LIST-SPECIFIC  : OR < 3 OR emp_p > 0.05 (on the ST12 / diff lists)
        UNINTERPRETABLE     : missing OR/emp_p or n_in_bg < 3

    For the reproduction cell (batch047 × pLI), the ROBUST/DILUTED mapping still
    applies; the separate reproduction gate is additional (see main()).

    Design note: the string `HAND-LIST-SPECIFIC` is used for ALL lists (not just
    ST12) as the generic "null/small effect" label, consistent with the spec's
    classification field (brief §MEASUREMENT + preflight schema). Downstream
    interpretation respects the per-list meaning.
    """
    or_val = row.get("or")
    emp_p = row.get("emp_p")
    if or_val is None or emp_p is None or row.get("n_in_list", 0) < 3:
        return "UNINTERPRETABLE"

    # Infinite OR with emp_p < 0.01 is a ROBUST positive (all list genes hit target).
    if np.isinf(or_val):
        return "ROBUST" if emp_p < 0.01 else "UNINTERPRETABLE"

    if or_val >= 10 and emp_p < 0.01:
        return "ROBUST"
    if or_val >= 3 and emp_p < 0.05:
        return "DILUTED"
    # Fallback: small OR or non-significant permutation.
    return "HAND-LIST-SPECIFIC"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="batch_053_A constraint robustness re-audit",
    )
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM,
                        help="Permutation count (default 5000; batch_048 convention).")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: n_perm=100 override for schema verification.")
    args = parser.parse_args()

    n_perm = 100 if args.smoke else args.n_perm

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(LOG_DIR / "run.log")
    env = _python_env_info()
    logger.info("=" * 70)
    logger.info("batch_053_A — constraint robustness re-audit")
    logger.info("cmd: %s", " ".join(sys.argv))
    logger.info("n_perm=%d RNG_SEED=%d", n_perm, RNG_SEED)
    logger.info("env: %s", env)
    logger.info("reproduction target: OR=%.2f ± %.1f (batch_048 F121/F124)",
                REPRODUCTION_TARGET_OR, REPRODUCTION_TOLERANCE)

    # ---------- 1. Load gnomAD + PGC3 ST12 (via batch_048 exact functions) ----------
    gnomad = load_gnomad(logger)
    edt1 = load_pgc3_edt1(logger)

    # ---------- 2. Build gene lists (ALL before any inferential test) ----------
    gene_lists = build_gene_lists(logger, edt1)

    # ---------- 3. Write gene_lists_used.json BEFORE tests ----------
    gene_lists_meta = {
        "batch": "batch_053_A",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "n_perm": n_perm,
        "env": env,
        "gene_lists": {
            name: {
                "n": len(genes),
                "members": genes,
                "sha256_sorted": _sha256_of_sorted(genes),
            }
            for name, genes in gene_lists.items()
        },
    }
    gene_lists_path = OUTPUT_DIR / "gene_lists_used.json"
    with gene_lists_path.open("w") as fh:
        json.dump(gene_lists_meta, fh, indent=2, default=str)
    logger.info("Wrote %s", gene_lists_path)

    # ---------- 4. Prepare background arrays (MHC-excluded, matches batch_048) ----------
    bg = gnomad[~gnomad["mhc_indicator"]].copy()
    bg_genes_arr = bg["gene"].astype(str).to_numpy()
    bg_genes_set = set(bg_genes_arr.tolist())
    lengths = bg["gene_length"].fillna(0).to_numpy()
    pli_mask = bg["pli_ge_09"].astype(bool).to_numpy()
    loeuf_mask = bg["loeuf_lt_035"].astype(bool).to_numpy()
    misz_mask, misz_threshold = _missense_z_top_decile_mask(gnomad)

    logger.info("BG: n=%d, pLI_ge_09=%d, LOEUF_lt_035=%d, mis.z_top_decile=%d (thresh=%.3f)",
                len(bg_genes_arr), int(pli_mask.sum()), int(loeuf_mask.sum()),
                int(misz_mask.sum()), misz_threshold)

    metrics = [
        (f"pLI >= {PLI_THRESHOLD}", pli_mask),
        (f"LOEUF <= {LOEUF_THRESHOLD}", loeuf_mask),
        ("missense_z top decile", misz_mask),
    ]

    # ---------- 5. Reproduction gate — run batch047 × pLI FIRST ----------
    logger.info("-" * 70)
    logger.info("REPRODUCTION GATE: SynGO_EDT1_batch047 × pLI >= 0.9")
    repro_cell = _run_one_cell(
        logger,
        "SynGO_EDT1_batch047",
        set(gene_lists["SynGO_EDT1_batch047"]),
        f"pLI >= {PLI_THRESHOLD}",
        pli_mask, bg_genes_arr, bg_genes_set, lengths, n_perm,
    )
    observed_or = repro_cell["or"]
    repro_passes = (
        observed_or is not None
        and not np.isinf(observed_or)
        and abs(float(observed_or) - REPRODUCTION_TARGET_OR) <= REPRODUCTION_TOLERANCE
    )
    repro_result = {
        "batch": "batch_053_A",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "target_pLI_OR": REPRODUCTION_TARGET_OR,
        "tolerance": REPRODUCTION_TOLERANCE,
        "observed_pLI_OR": (None if observed_or is None else float(observed_or)),
        "passes": bool(repro_passes),
        "cell_detail": repro_cell,
    }
    repro_path = OUTPUT_DIR / "reproduction_check.json"
    with repro_path.open("w") as fh:
        json.dump(repro_result, fh, indent=2, default=str)
    logger.info("Wrote %s (passes=%s, observed_OR=%s)",
                repro_path, repro_passes, observed_or)

    if not repro_passes:
        # STOP — do not run the rest of the audit.
        # WHY: brief §UNINTERPRETABLE — if reproduction fails the entire downstream
        # audit is uninterpretable because the null / metric path itself is suspect.
        msg = (
            f"REPRODUCTION GATE FAILED: observed OR={observed_or} is NOT within "
            f"[{REPRODUCTION_TARGET_OR - REPRODUCTION_TOLERANCE:.2f}, "
            f"{REPRODUCTION_TARGET_OR + REPRODUCTION_TOLERANCE:.2f}] of target "
            f"{REPRODUCTION_TARGET_OR}. The pipeline disagrees with batch_048; "
            "downstream audit classifications would be uninterpretable. "
            "Investigate gnomAD loader dedup, PGC3 ST12 loader, or permutation code path."
        )
        logger.error(msg)
        challenge_path = BATCH_DIR / "challenge.md"
        with challenge_path.open("w") as fh:
            fh.write(f"# batch_053_A — Reproduction Gate Failure\n\n{msg}\n\n"
                     f"- observed OR: {observed_or}\n"
                     f"- target: {REPRODUCTION_TARGET_OR} ± {REPRODUCTION_TOLERANCE}\n"
                     f"- cell detail: see `output/reproduction_check.json`\n")
        logger.error("Wrote %s — STOPPING before results.json.", challenge_path)
        return 2

    # ---------- 6. Run the full 4 × 3 grid ----------
    logger.info("-" * 70)
    logger.info("Running full 4 (gene lists) × 3 (metrics) = 12 cells")
    all_rows: list[dict[str, Any]] = []
    # We already ran batch047 × pLI for the reproduction gate; reuse it.
    repro_cell["classification"] = classify_cell(repro_cell)
    all_rows.append(repro_cell)

    for list_name, list_genes in gene_lists.items():
        list_set = set(list_genes)
        for metric_name, metric_mask in metrics:
            if (list_name == "SynGO_EDT1_batch047"
                    and metric_name == f"pLI >= {PLI_THRESHOLD}"):
                # Already computed for the reproduction gate; skip duplicate.
                continue
            row = _run_one_cell(
                logger, list_name, list_set,
                metric_name, metric_mask,
                bg_genes_arr, bg_genes_set, lengths, n_perm,
            )
            row["classification"] = classify_cell(row)
            all_rows.append(row)

    # Sort rows for deterministic output (by list name, then metric).
    list_order = list(gene_lists.keys())
    metric_order = [m[0] for m in metrics]
    all_rows.sort(key=lambda r: (list_order.index(r["gene_list"]),
                                 metric_order.index(r["constraint_metric"])))

    # ---------- 7. Write results.json ----------
    results = {
        "batch": "batch_053_A",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "rng_seed": RNG_SEED,
        "n_perm": n_perm,
        "env": env,
        "reproduction_check": {
            "target_pLI_OR": REPRODUCTION_TARGET_OR,
            "tolerance": REPRODUCTION_TOLERANCE,
            "observed_pLI_OR": float(observed_or),
            "passes": bool(repro_passes),
        },
        "gene_lists": {name: genes for name, genes in gene_lists.items()},
        "missense_z_top_decile_threshold": misz_threshold,
        "results": all_rows,
    }
    results_path = OUTPUT_DIR / "results.json"
    with results_path.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Wrote %s", results_path)

    # ---------- 8. Console summary ----------
    logger.info("=" * 70)
    logger.info("SUMMARY (sorted by gene_list × metric):")
    for r in all_rows:
        or_str = ("inf" if r["or"] is None or np.isinf(r["or"]) else f"{r['or']:.2f}")
        emp_str = "N/A" if r["emp_p"] is None else f"{r['emp_p']:.4g}"
        logger.info(
            "  [%-28s × %-22s] OR=%-7s emp_p=%-10s n=%-3d hit=%-4s → %s",
            r["gene_list"], r["constraint_metric"], or_str, emp_str,
            r["n_in_list"],
            (str(r["n_pli_hit"]) if r["n_pli_hit"] is not None else "N/A"),
            r["classification"],
        )

    logger.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
