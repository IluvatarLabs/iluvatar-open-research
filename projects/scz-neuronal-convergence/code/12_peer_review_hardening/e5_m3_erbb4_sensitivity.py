#!/usr/bin/env python3
"""batch_061 E5 -- M3 ERBB4-excluded OR sensitivity (DESCRIPTIVE, CONDITIONAL).

Implements brief_v2.md section E5 EXACTLY.

Purpose: Report M3 myelination module OR after removing ERBB4. ERBB4 is
one of ~5 M3 genes overlapping PoPS-top-1000. NRG1-ERBB4 is a neuronal
(not myelination-specific) signaling pathway, so its presence in M3
overlap may inflate the OR and mischaracterize the module signal.

Conditional gate: Execute ONLY if E3 formal results show M3 PoPS OR >= 3.0.
If E3 shows M3 is a training artifact (OR < 3.0 under PoPS), E5 is moot.

Steps:
  1. Load PoPS predictions -> PoPS-top-1000.
  2. Define M3 module genes (same as e8_g4_module.py).
  3. Check E3 results (conditional gate): M3 PoPS OR >= 3.0?
  4. If gate passes: Fisher's exact test for M3-with-ERBB4 and M3-without-ERBB4.
  5. Report OR, CI, p-value for both conditions.

WHY this experiment: F060_07 showed M3 myelination OR=5.8 (PoPS-top-1000)
with 5 overlap genes. ERBB4 is a known neuronal receptor (NRG1-ERBB4
signaling) that appears in the myelination module because oligodendrocyte
precursor cells express ERBB4. If removing ERBB4 substantially drops the
OR, the M3 signal may be driven by neuronal spillover rather than genuine
myelination enrichment. NOV-060-05.

This is DESCRIPTIVE -- NO formal decision rule. With ~4 overlap genes from
~27 module genes, the CI is too wide for formal classification.

Source: Gandal et al. 2018 (Science) DOI:10.1126/science.aat8127.

Output: experiments/batch_061/output/e5/results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Imports from batch_060/_common via importlib.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")

_B060_COMMON_PATH = PROJECT_ROOT / "experiments" / "batch_060" / "scripts" / "_common.py"
_spec060 = _ilu.spec_from_file_location("batch060_common_e5", str(_B060_COMMON_PATH))
_b060 = _ilu.module_from_spec(_spec060)  # type: ignore[arg-type]
assert _spec060 is not None and _spec060.loader is not None
_spec060.loader.exec_module(_b060)

# Re-bind names for clarity.
BATCH054_P05_PREDS = _b060.BATCH054_P05_PREDS
GENE_ANNOT = _b060.GENE_ANNOT
atomic_write_json = _b060.atomic_write_json
load_preds = _b060.load_preds
setup_logger = _b060.setup_logger
symbols_to_ensgids = _b060.symbols_to_ensgids
B060_SEED_MASTER = _b060.B060_SEED_MASTER

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_061"
OUTPUT_DIR = BATCH_DIR / "output" / "e5"
LOGS_DIR = BATCH_DIR / "logs"

# E3 results file (for conditional gate).
E3_RESULTS = BATCH_DIR / "output" / "e3" / "results.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_K = 1000  # brief_v2: top-1000 PoPS genes
SEED = B060_SEED_MASTER

# Conditional gate threshold (brief_v2 E5): M3 PoPS OR >= 3.0.
# WHY 3.0: brief_v2 specifies "Execute E5 ONLY if E3 formal results show
# M3 PoPS OR remains >= 3.0." Below 3.0, the M3 signal is already weak
# enough that removing a single gene is uninformative.
M3_POPS_OR_GATE = 3.0

# M3 myelination module (same as batch_060 E8, e8_g4_module.py).
# Sourced from Gandal et al. 2018 (Science) DOI:10.1126/science.aat8127.
GANDAL_M3_MYELINATION_HUBS = [
    "MBP", "PLP1", "MOG", "MAG", "MOBP", "CLDN11",
    "CNP", "OLIG1", "OLIG2", "SOX10", "NKX2-2", "MYRF",
    "UGT8", "FA2H", "GALC", "ASPA", "NAA",
    "ERBB3", "ERBB4", "NRG1",
    "ENPP2", "TF", "TPPP",
    "ERMN", "PLLP", "LPAR1",
    "ST18", "QKI", "BCAS1",
]

# ERBB4 symbol for exclusion.
ERBB4_SYMBOL = "ERBB4"


# =============================================================================
# Fisher's exact test (same as E3)
# =============================================================================
def fisher_exact_overlap(
    top_ensg: set[str],
    module_ensg: set[str],
    universe_ensg: set[str],
) -> dict:
    """Fisher's exact test for overlap of top-K genes with module markers.

    WHY Fisher's exact: same rationale as E3 -- exact test, handles small
    cell counts, directly gives OR.

    Contingency table:
                    In module    Not in module
    Top-K:              a              b
    Not top-K:          c              d
    """
    top_in_u = top_ensg & universe_ensg
    mod_in_u = module_ensg & universe_ensg

    a = len(top_in_u & mod_in_u)
    b = len(top_in_u - mod_in_u)
    c = len(mod_in_u - top_in_u)
    d = len(universe_ensg - top_in_u - mod_in_u)

    table = np.array([[a, b], [c, d]])
    or_val, p_val = stats.fisher_exact(table, alternative="greater")

    # 95% CI for log(OR) via Woolf's method with 0.5 correction.
    a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = np.log(a_c * d_c / (b_c * c_c))
    se_log_or = np.sqrt(1 / a_c + 1 / b_c + 1 / c_c + 1 / d_c)
    ci_lo = float(np.exp(log_or - 1.96 * se_log_or))
    ci_hi = float(np.exp(log_or + 1.96 * se_log_or))

    overlap_genes = sorted(top_in_u & mod_in_u)

    return {
        "OR": round(float(or_val), 4) if np.isfinite(or_val) else "Inf",
        "CI_lo": round(ci_lo, 4),
        "CI_hi": round(ci_hi, 4),
        "p_fisher": float(p_val),
        "a_overlap": int(a),
        "b_top_only": int(b),
        "c_module_only": int(c),
        "d_neither": int(d),
        "n_module_in_universe": len(mod_in_u),
        "n_top_in_universe": len(top_in_u),
        "n_universe": len(universe_ensg),
        "overlap_ensgids": overlap_genes,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("e5_m3_erbb4_sensitivity", LOGS_DIR / "e5_m3_erbb4_sensitivity.log")
    logger.info("=== E5: M3 ERBB4-Excluded OR Sensitivity (DESCRIPTIVE) ===")
    t0 = time.time()

    results: dict = {
        "experiment": "E5_m3_erbb4_sensitivity",
        "brief": "brief_v2.md section E5",
        "analysis_type": "DESCRIPTIVE -- no formal decision rule",
        "conditional_on": "E3 M3 PoPS OR >= 3.0",
        "seed": SEED,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ------------------------------------------------------------------
    # Step 1: Load PoPS predictions.
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading PoPS predictions...")
    if not BATCH054_P05_PREDS.exists():
        results["verdict"] = "BLOCKED"
        results["blockers"] = [f"PoPS predictions missing: {BATCH054_P05_PREDS}"]
        logger.error("BLOCKED: %s", results["blockers"])
        atomic_write_json(results, OUTPUT_DIR / "results.json")
        return
    preds = load_preds(BATCH054_P05_PREDS)
    preds_sorted = preds.sort_values("PoPS_Score", ascending=False).reset_index(drop=True)
    n_pops = len(preds_sorted)
    pops_top_k = min(TOP_K, n_pops)
    pops_top_ensg = set(preds_sorted["ENSGID"].iloc[:pops_top_k].tolist())
    pops_all_ensg = set(preds_sorted["ENSGID"].tolist())
    logger.info("PoPS: %d total, top-%d loaded", n_pops, pops_top_k)

    # ------------------------------------------------------------------
    # Step 2: Map M3 module genes.
    # ------------------------------------------------------------------
    logger.info("Step 2: Mapping M3 myelination module genes...")
    m3_symbols = set(GANDAL_M3_MYELINATION_HUBS)
    m3_ensg, m3_sym_map = symbols_to_ensgids(m3_symbols)
    unmapped_m3 = sorted(m3_symbols - set(m3_sym_map.keys()))
    logger.info(
        "M3: %d symbols -> %d ENSGIDs. Unmapped: %s",
        len(m3_symbols), len(m3_ensg), unmapped_m3 if unmapped_m3 else "none",
    )

    # Get ERBB4 ENSGID.
    erbb4_ensg = m3_sym_map.get(ERBB4_SYMBOL)
    if erbb4_ensg is None:
        logger.warning("ERBB4 not found in gene annotation. Cannot run leave-out.")
        results["verdict"] = "BLOCKED"
        results["blockers"] = ["ERBB4 not mapped to ENSGID"]
        results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["elapsed_seconds"] = round(time.time() - t0, 1)
        atomic_write_json(results, OUTPUT_DIR / "results.json")
        return
    logger.info("ERBB4 ENSGID: %s", erbb4_ensg)

    results["m3_mapping"] = {
        "n_symbols": len(m3_symbols),
        "n_mapped": len(m3_ensg),
        "unmapped": unmapped_m3,
        "erbb4_ensgid": erbb4_ensg,
    }

    # M3 without ERBB4.
    m3_no_erbb4 = m3_ensg - {erbb4_ensg}
    logger.info(
        "M3 without ERBB4: %d genes (was %d)", len(m3_no_erbb4), len(m3_ensg),
    )

    # ------------------------------------------------------------------
    # Step 3: Conditional gate -- check E3 results.
    # WHY conditional: brief_v2 E5 says "Execute E5 ONLY if E3 formal
    # results show M3 PoPS OR remains >= 3.0." If M3 is already weak
    # in PoPS ranking, removing a single gene is uninformative.
    # ------------------------------------------------------------------
    logger.info("Step 3: Checking E3 conditional gate...")
    gate_passed = False
    e3_m3_pops_or = None

    if E3_RESULTS.exists():
        try:
            with open(E3_RESULTS, "r") as fh:
                e3_data = json.load(fh)
            # Extract M3 PoPS OR from E3 results.
            fisher_pops = e3_data.get("fisher_pops", {})
            m3_pops_data = fisher_pops.get("M3_myelination", {})
            e3_m3_pops_or = m3_pops_data.get("OR")
            if e3_m3_pops_or is not None:
                e3_m3_pops_or_f = float(e3_m3_pops_or) if isinstance(
                    e3_m3_pops_or, (int, float)
                ) else float("inf")
                gate_passed = e3_m3_pops_or_f >= M3_POPS_OR_GATE
                logger.info(
                    "E3 M3 PoPS OR = %.2f. Gate threshold = %.1f. Gate %s.",
                    e3_m3_pops_or_f, M3_POPS_OR_GATE,
                    "PASSED" if gate_passed else "FAILED",
                )
            else:
                logger.warning("E3 results exist but M3 PoPS OR not found.")
        except Exception as exc:
            logger.warning("Failed to read E3 results: %s", exc)
    else:
        logger.warning("E3 results not found at %s. Gate check skipped.", E3_RESULTS)

    results["conditional_gate"] = {
        "e3_results_found": E3_RESULTS.exists(),
        "e3_m3_pops_or": e3_m3_pops_or,
        "gate_threshold": M3_POPS_OR_GATE,
        "gate_passed": gate_passed,
    }

    if not gate_passed:
        skip_reason = (
            f"E5 SKIPPED -- M3 PoPS OR "
            f"{'= ' + str(e3_m3_pops_or) if e3_m3_pops_or is not None else 'not available'}"
            f" < {M3_POPS_OR_GATE} per E3, E5 conditional gate not met"
        )
        if not E3_RESULTS.exists():
            skip_reason = (
                "E5 SKIPPED -- E3 results not found. Run E3 first, then re-run E5."
            )
        results["verdict"] = "SKIPPED"
        results["verdict_reason"] = skip_reason
        logger.info(skip_reason)
        results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["elapsed_seconds"] = round(time.time() - t0, 1)
        atomic_write_json(results, OUTPUT_DIR / "results.json")
        return

    # ------------------------------------------------------------------
    # Step 4: Fisher's exact test for M3 with and without ERBB4.
    # Universe = all genes with PoPS predictions (same as E3/E8).
    # ------------------------------------------------------------------
    logger.info("Step 4: Fisher's exact test for M3 +/- ERBB4...")
    universe = pops_all_ensg

    # Load gene annotation for symbol lookup.
    annot_df = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME"]]
    ensg2sym = dict(zip(annot_df["ENSGID"], annot_df["NAME"]))

    # M3 with ERBB4.
    fisher_with = fisher_exact_overlap(pops_top_ensg, m3_ensg, universe)
    logger.info(
        "M3 with ERBB4: OR=%.2f [%.2f, %.2f], p=%.4g, overlap=%d",
        fisher_with["OR"] if isinstance(fisher_with["OR"], float) else float("inf"),
        fisher_with["CI_lo"], fisher_with["CI_hi"],
        fisher_with["p_fisher"], fisher_with["a_overlap"],
    )

    # M3 without ERBB4.
    fisher_without = fisher_exact_overlap(pops_top_ensg, m3_no_erbb4, universe)
    logger.info(
        "M3 without ERBB4: OR=%.2f [%.2f, %.2f], p=%.4g, overlap=%d",
        fisher_without["OR"] if isinstance(fisher_without["OR"], float) else float("inf"),
        fisher_without["CI_lo"], fisher_without["CI_hi"],
        fisher_without["p_fisher"], fisher_without["a_overlap"],
    )

    # Annotate overlap genes with symbols.
    with_overlap_syms = sorted([ensg2sym.get(e, e) for e in fisher_with["overlap_ensgids"]])
    without_overlap_syms = sorted([ensg2sym.get(e, e) for e in fisher_without["overlap_ensgids"]])

    results["fisher_m3_with_erbb4"] = fisher_with
    results["fisher_m3_with_erbb4"]["overlap_symbols"] = with_overlap_syms
    results["fisher_m3_without_erbb4"] = fisher_without
    results["fisher_m3_without_erbb4"]["overlap_symbols"] = without_overlap_syms

    # ------------------------------------------------------------------
    # Step 5: OR change summary.
    # ------------------------------------------------------------------
    logger.info("Step 5: OR change summary...")
    or_with = fisher_with["OR"]
    or_without = fisher_without["OR"]
    or_with_f = float(or_with) if isinstance(or_with, (int, float)) else float("inf")
    or_without_f = float(or_without) if isinstance(or_without, (int, float)) else float("inf")

    if np.isfinite(or_with_f) and or_with_f != 0:
        or_change_pct = round(100 * (or_without_f - or_with_f) / or_with_f, 1)
    else:
        or_change_pct = "N/A"

    # Check if ERBB4 is in the PoPS-top-1000 overlap.
    erbb4_in_overlap = erbb4_ensg in (set(fisher_with["overlap_ensgids"]))

    results["or_change_summary"] = {
        "OR_with_erbb4": or_with,
        "OR_without_erbb4": or_without,
        "CI_with": [fisher_with["CI_lo"], fisher_with["CI_hi"]],
        "CI_without": [fisher_without["CI_lo"], fisher_without["CI_hi"]],
        "overlap_with": fisher_with["a_overlap"],
        "overlap_without": fisher_without["a_overlap"],
        "or_change_pct": or_change_pct,
        "erbb4_in_pops_top1000": erbb4_in_overlap,
        "erbb4_ensgid": erbb4_ensg,
    }

    logger.info(
        "OR: %.2f (with) -> %.2f (without), change=%s%%",
        or_with_f if np.isfinite(or_with_f) else float("inf"),
        or_without_f if np.isfinite(or_without_f) else float("inf"),
        or_change_pct,
    )

    # ------------------------------------------------------------------
    # Step 6: Descriptive conclusion.
    # ------------------------------------------------------------------
    results["descriptive_note"] = (
        f"M3 OR decreased from {or_with} to {or_without} "
        f"(95% CI [{fisher_without['CI_lo']}, {fisher_without['CI_hi']}]) "
        f"upon ERBB4 removal. With {fisher_without['a_overlap']} overlap genes "
        f"from {fisher_without['n_module_in_universe']} module genes, the CI is "
        f"too wide for formal classification. ERBB4 was "
        f"{'in' if erbb4_in_overlap else 'NOT in'} PoPS-top-1000."
    )

    results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["elapsed_seconds"] = round(time.time() - t0, 1)

    atomic_write_json(results, OUTPUT_DIR / "results.json")
    logger.info(
        "E5 complete. Elapsed: %.1fs. This is DESCRIPTIVE -- no formal verdict.",
        results["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
