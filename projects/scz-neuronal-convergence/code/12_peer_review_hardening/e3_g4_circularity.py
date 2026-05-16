#!/usr/bin/env python3
"""batch_061 E3 -- MAGMA-Z-top-1000 vs PoPS-top-1000 module enrichment (circularity test).

Implements brief_v2.md section E3 EXACTLY.

Purpose: Test whether G4 module enrichment (F060_07) is a genuine genomic
signal or a PoPS training artifact. PoPS incorporates brain expression
features during training, so any module overlap could be partly circular.
MAGMA-Z is derived purely from GWAS summary statistics and is therefore
independent of PoPS training features.

Steps:
  1. Load PoPS predictions -> PoPS-top-1000 (by PoPS_Score, descending).
  2. Load MAGMA Z-scores -> MAGMA-Z-top-1000 (by |ZSTAT|, descending).
  3. Compute overlap between the two top-1000 lists.
  4. For each Gandal module (M1/M2/M3): Fisher's exact test for enrichment
     in PoPS-top-1000 AND in MAGMA-Z-top-1000.
  5. Compute OR ratio (MAGMA / PoPS) per module.
  6. BH correction across 3 modules within each ranking method.
  7. Decision rule per brief_v2.

WHY this experiment: F060_07 showed M1 neuronal module OR=49.7 (q < 1e-10)
in PoPS-top-1000. But PoPS uses brain expression features (which correlate
with M1 membership) during training. MAGMA-Z provides an orthogonal ranking
that does not use expression features. If M1 enrichment persists with
MAGMA-Z ranking, it is a genuine genomic signal (possibly amplified by PoPS
training). NOV-060-02.

Source: Gandal et al. 2018 (Science) DOI:10.1126/science.aat8127.

Output: experiments/batch_061/output/e3/results.json
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
# WHY importlib: all _common.py files share the same filename; sys.path
# would shadow. importlib loads under distinct module names.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")

_B060_COMMON_PATH = PROJECT_ROOT / "experiments" / "batch_060" / "scripts" / "_common.py"
_spec060 = _ilu.spec_from_file_location("batch060_common", str(_B060_COMMON_PATH))
_b060 = _ilu.module_from_spec(_spec060)  # type: ignore[arg-type]
assert _spec060 is not None and _spec060.loader is not None
_spec060.loader.exec_module(_b060)

# Re-bind names for clarity and auditor traceability.
BATCH054_P05_PREDS = _b060.BATCH054_P05_PREDS
GENE_ANNOT = _b060.GENE_ANNOT
MAGMA_SCZ_GENES_OUT = _b060.MAGMA_SCZ_GENES_OUT
atomic_write_json = _b060.atomic_write_json
bh_fdr = _b060.bh_fdr
load_preds = _b060.load_preds
setup_logger = _b060.setup_logger
symbols_to_ensgids = _b060.symbols_to_ensgids
B060_SEED_MASTER = _b060.B060_SEED_MASTER

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_061"
OUTPUT_DIR = BATCH_DIR / "output" / "e3"
LOGS_DIR = BATCH_DIR / "logs"

# MAGMA Z-score file (SCZ, ENSGID-keyed).
# WHY this file: batch_053_B output uses ENSGIDs directly, no Entrez mapping
# needed. The ZSTAT column is the gene-level Z-score from MAGMA.
MAGMA_SCZ_ENSGID = (
    PROJECT_ROOT / "experiments" / "batch_053_B" / "output"
    / "PGC3_EUR_gene_ENSGID.genes.out"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_K = 1000  # brief_v2 E3: top-1000 for both rankings
SEED = B060_SEED_MASTER

# Gandal 2018 transcriptomic module hub genes.
# WHY these gene lists: Identical to batch_060 E8 (e8_g4_module.py).
# Sourced from Gandal et al. 2018 (Science) DOI:10.1126/science.aat8127,
# Table S4 hub genes. See e8_g4_module.py for full provenance documentation.
GANDAL_M1_NEURONAL_HUBS = [
    "SYT1", "SNAP25", "SYN1", "SYP", "GAD1", "GAD2", "SLC17A7",
    "NRGN", "CAMK2A", "GRIA1", "GRIA2", "GRIN1", "GRIN2A", "GRIN2B",
    "DLG4", "HOMER1", "SHANK2", "SYNGAP1", "NRXN1", "NRXN3",
    "NLGN1", "CACNA1A", "CACNA1B", "SCN1A", "SCN2A", "SCN8A",
    "KCNAB2", "KCND2", "SLC12A5", "GABRA1", "GABRB2", "GABRG2",
    "RAB3A", "STX1A", "STXBP1", "UNC13A", "RIMS1", "BSN",
    "PCLO", "CPLX1", "CPLX2", "NSF", "VAMP2", "SYT4",
    "NEFL", "NEFM", "NEFH", "TUBB3", "MAP2", "MAPT",
]

GANDAL_M2_ASTROGLIA_HUBS = [
    "GFAP", "AQP4", "SLC1A2", "SLC1A3", "GJA1", "ALDH1L1",
    "S100B", "SOX9", "NFIA", "NFIB", "CLU",
    "CSF1R", "CX3CR1", "TREM2", "TYROBP", "C1QA", "C1QB", "C1QC",
    "AIF1", "CD68", "ITGAM", "P2RY12", "TMEM119",
    "HEXB", "CTSS", "C3", "FCER1G", "IRF8",
    "TLR2", "CD14", "MRC1", "MSR1",
    "IL1B", "TNF", "CCL2", "CXCL10",
    "SERPINA3", "VIM", "HSPB1",
]

GANDAL_M3_MYELINATION_HUBS = [
    "MBP", "PLP1", "MOG", "MAG", "MOBP", "CLDN11",
    "CNP", "OLIG1", "OLIG2", "SOX10", "NKX2-2", "MYRF",
    "UGT8", "FA2H", "GALC", "ASPA", "NAA",
    "ERBB3", "ERBB4", "NRG1",
    "ENPP2", "TF", "TPPP",
    "ERMN", "PLLP", "LPAR1",
    "ST18", "QKI", "BCAS1",
]

MODULES_RAW = {
    "M1_neuronal": GANDAL_M1_NEURONAL_HUBS,
    "M2_astroglia": GANDAL_M2_ASTROGLIA_HUBS,
    "M3_myelination": GANDAL_M3_MYELINATION_HUBS,
}


# =============================================================================
# Fisher's exact test (reused from e8_g4_module.py pattern)
# =============================================================================
def fisher_exact_overlap(
    top_ensg: set[str],
    module_ensg: set[str],
    universe_ensg: set[str],
) -> dict:
    """Fisher's exact test for overlap of top-K genes with module markers.

    WHY Fisher's exact: brief_v2 section E3 specifies Fisher's exact test.
    It is exact (no asymptotics), handles small cell counts, and directly
    gives the odds ratio.

    Contingency table:
                    In module    Not in module
    Top-K:              a              b
    Not top-K:          c              d

    Returns dict with OR, CI, p-value, cell counts.
    """
    top_in_u = top_ensg & universe_ensg
    mod_in_u = module_ensg & universe_ensg

    a = len(top_in_u & mod_in_u)
    b = len(top_in_u - mod_in_u)
    c = len(mod_in_u - top_in_u)
    d = len(universe_ensg - top_in_u - mod_in_u)

    table = np.array([[a, b], [c, d]])
    # One-sided test for enrichment (greater).
    or_val, p_val = stats.fisher_exact(table, alternative="greater")

    # 95% CI for log(OR) using Woolf's method with 0.5 continuity correction.
    # WHY Woolf's: standard approach for Fisher OR CI; 0.5 correction avoids
    # log(0) when any cell is zero.
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
    logger = setup_logger("e3_g4_circularity", LOGS_DIR / "e3_g4_circularity.log")
    logger.info("=== E3: MAGMA-Z-top-1000 vs PoPS-top-1000 module enrichment ===")
    t0 = time.time()

    results: dict = {
        "experiment": "E3_g4_circularity",
        "brief": "brief_v2.md section E3",
        "seed": SEED,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ------------------------------------------------------------------
    # Step 1: Load PoPS predictions -> PoPS-top-1000.
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
    logger.info("PoPS predictions: %d genes total", n_pops)

    pops_top_k = min(TOP_K, n_pops)
    pops_top_ensg = set(preds_sorted["ENSGID"].iloc[:pops_top_k].tolist())
    pops_all_ensg = set(preds_sorted["ENSGID"].tolist())
    logger.info(
        "PoPS top-%d: min score=%.4f",
        pops_top_k, preds_sorted["PoPS_Score"].iloc[pops_top_k - 1],
    )

    # ------------------------------------------------------------------
    # Step 2: Load MAGMA Z-scores -> MAGMA-Z-top-1000.
    # WHY |ZSTAT|: brief_v2 E3 specifies "top-1000 by absolute MAGMA Z."
    # Genes with large positive or negative Z are both biologically
    # interesting (positive = enriched for risk variants; negative =
    # depleted, suggesting protective or constrained loci).
    # ------------------------------------------------------------------
    logger.info("Step 2: Loading MAGMA Z-scores...")
    if not MAGMA_SCZ_ENSGID.exists():
        results["verdict"] = "BLOCKED"
        results["blockers"] = [f"MAGMA Z-scores missing: {MAGMA_SCZ_ENSGID}"]
        logger.error("BLOCKED: %s", results["blockers"])
        atomic_write_json(results, OUTPUT_DIR / "results.json")
        return
    magma = pd.read_csv(MAGMA_SCZ_ENSGID, sep=r"\s+")
    magma = magma.rename(columns={"GENE": "ENSGID", "ZSTAT": "MAGMA_Z"})
    magma["abs_MAGMA_Z"] = magma["MAGMA_Z"].abs()
    magma_sorted = magma.sort_values("abs_MAGMA_Z", ascending=False).reset_index(drop=True)
    n_magma = len(magma_sorted)
    logger.info("MAGMA Z-scores: %d genes total", n_magma)

    magma_top_k = min(TOP_K, n_magma)
    magma_top_ensg = set(magma_sorted["ENSGID"].iloc[:magma_top_k].tolist())
    magma_all_ensg = set(magma_sorted["ENSGID"].tolist())
    logger.info(
        "MAGMA top-%d: min |Z|=%.4f",
        magma_top_k, magma_sorted["abs_MAGMA_Z"].iloc[magma_top_k - 1],
    )

    # ------------------------------------------------------------------
    # Step 3: Compute overlap between the two top-1000 lists.
    # WHY report overlap: brief_v2 E3 pre-computation gate requires
    # overlap in [10%, 80%] for interpretability. If overlap is too high
    # (>80%) the tests are redundant; if too low (<10%) the comparison
    # is not meaningful.
    # ------------------------------------------------------------------
    logger.info("Step 3: Overlap between PoPS-top-1000 and MAGMA-Z-top-1000...")
    overlap_ensg = pops_top_ensg & magma_top_ensg
    overlap_count = len(overlap_ensg)
    overlap_pct = round(100 * overlap_count / TOP_K, 1)
    logger.info(
        "Overlap: %d/%d = %.1f%%", overlap_count, TOP_K, overlap_pct,
    )

    # Load gene annotation for symbol lookup.
    annot_df = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME"]]
    ensg2sym = dict(zip(annot_df["ENSGID"], annot_df["NAME"]))

    overlap_genes_named = sorted(
        [ensg2sym.get(e, e) for e in overlap_ensg]
    )

    results["overlap"] = {
        "overlap_count": overlap_count,
        "overlap_pct": overlap_pct,
        "n_pops_top": pops_top_k,
        "n_magma_top": magma_top_k,
        "interpretable": 10.0 <= overlap_pct <= 80.0,
        "overlap_genes": overlap_genes_named[:100],  # Cap for readability.
    }

    # Check interpretability gate (brief_v2 E3 UNINTERPRETABLE).
    if overlap_pct < 10.0 or overlap_pct > 80.0:
        results["verdict"] = "UNINTERPRETABLE"
        results["verdict_reason"] = (
            f"Overlap {overlap_pct}% outside [10%, 80%] range. "
            "Test is not interpretable per brief_v2 E3."
        )
        logger.error(results["verdict_reason"])
        results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["elapsed_seconds"] = round(time.time() - t0, 1)
        atomic_write_json(results, OUTPUT_DIR / "results.json")
        return

    # ------------------------------------------------------------------
    # Step 4: Map Gandal modules to ENSGIDs and run Fisher's exact tests.
    # ------------------------------------------------------------------
    logger.info("Step 4: Module mapping and Fisher's exact tests...")

    # Map module symbols to ENSGIDs.
    module_ensg: dict[str, set[str]] = {}
    module_mapping_info: dict[str, dict] = {}
    for mod_name, symbols in MODULES_RAW.items():
        ensg_set, sym_map = symbols_to_ensgids(set(symbols))
        module_ensg[mod_name] = ensg_set
        unmapped = sorted(set(symbols) - set(sym_map.keys()))
        module_mapping_info[mod_name] = {
            "n_symbols": len(symbols),
            "n_mapped_ensg": len(ensg_set),
            "mapping_rate": round(len(ensg_set) / max(len(symbols), 1), 3),
            "unmapped_symbols": unmapped,
        }
        logger.info(
            "  %s: %d symbols -> %d ENSGIDs (%.1f%% mapped). Unmapped: %s",
            mod_name, len(symbols), len(ensg_set),
            100 * len(ensg_set) / max(len(symbols), 1),
            unmapped if unmapped else "none",
        )
    results["module_mapping"] = module_mapping_info

    # Universe = all genes with PoPS predictions (same as E8).
    # WHY: Fisher test compares top-K vs rest within the space of all
    # scoreable genes. Using only the intersection would inflate OR.
    universe_pops = pops_all_ensg

    # For MAGMA, use all genes with MAGMA Z-scores.
    universe_magma = magma_all_ensg

    results["universe_info"] = {
        "n_pops_universe": len(universe_pops),
        "n_magma_universe": len(universe_magma),
    }

    # Fisher tests for each module x each ranking method.
    fisher_pops: dict[str, dict] = {}
    fisher_magma: dict[str, dict] = {}

    for mod_name, ensg_set in module_ensg.items():
        # PoPS-top-1000 enrichment.
        fisher_pops[mod_name] = fisher_exact_overlap(
            pops_top_ensg, ensg_set, universe_pops,
        )
        # MAGMA-Z-top-1000 enrichment.
        fisher_magma[mod_name] = fisher_exact_overlap(
            magma_top_ensg, ensg_set, universe_magma,
        )
        logger.info(
            "  %s PoPS: OR=%.2f [%.2f, %.2f], p=%.4g, overlap=%d",
            mod_name,
            fisher_pops[mod_name]["OR"] if isinstance(fisher_pops[mod_name]["OR"], float) else float("inf"),
            fisher_pops[mod_name]["CI_lo"], fisher_pops[mod_name]["CI_hi"],
            fisher_pops[mod_name]["p_fisher"],
            fisher_pops[mod_name]["a_overlap"],
        )
        logger.info(
            "  %s MAGMA: OR=%.2f [%.2f, %.2f], p=%.4g, overlap=%d",
            mod_name,
            fisher_magma[mod_name]["OR"] if isinstance(fisher_magma[mod_name]["OR"], float) else float("inf"),
            fisher_magma[mod_name]["CI_lo"], fisher_magma[mod_name]["CI_hi"],
            fisher_magma[mod_name]["p_fisher"],
            fisher_magma[mod_name]["a_overlap"],
        )

    # ------------------------------------------------------------------
    # Step 5: BH correction across 3 modules within each method.
    # WHY BH across 3 (not 6): brief_v2 E3 specifies BH correction
    # "within each ranking method." PoPS and MAGMA are separate analyses
    # with separate hypotheses, so each gets its own BH family.
    # ------------------------------------------------------------------
    logger.info("Step 5: BH-FDR correction...")
    module_names = list(module_ensg.keys())

    # PoPS BH.
    p_pops = [fisher_pops[m]["p_fisher"] for m in module_names]
    q_pops = bh_fdr(p_pops)
    for i, m in enumerate(module_names):
        fisher_pops[m]["q_BH"] = round(float(q_pops[i]), 6)

    # MAGMA BH.
    p_magma = [fisher_magma[m]["p_fisher"] for m in module_names]
    q_magma = bh_fdr(p_magma)
    for i, m in enumerate(module_names):
        fisher_magma[m]["q_BH"] = round(float(q_magma[i]), 6)

    results["fisher_pops"] = fisher_pops
    results["fisher_magma"] = fisher_magma

    # ------------------------------------------------------------------
    # Step 6: OR ratio (MAGMA / PoPS) per module.
    # WHY OR ratio: brief_v2 E3 primary metric. OR_ratio < 0.15 means
    # PoPS amplifies by > 6x. OR_ratio near 1.0 means no circularity.
    # ------------------------------------------------------------------
    logger.info("Step 6: OR ratio (MAGMA / PoPS)...")
    or_ratios: dict[str, dict] = {}
    for m in module_names:
        or_pops = fisher_pops[m]["OR"]
        or_magma = fisher_magma[m]["OR"]
        # Handle Inf values.
        or_pops_f = float(or_pops) if isinstance(or_pops, (int, float)) else float("inf")
        or_magma_f = float(or_magma) if isinstance(or_magma, (int, float)) else float("inf")

        if or_pops_f == 0.0 or not np.isfinite(or_pops_f):
            ratio = "Inf" if or_magma_f > 0 else "undefined"
        else:
            ratio = round(or_magma_f / or_pops_f, 4)

        or_ratios[m] = {
            "OR_pops": or_pops,
            "OR_magma": or_magma,
            "ratio_magma_over_pops": ratio,
        }
        logger.info(
            "  %s: MAGMA OR=%.2f / PoPS OR=%.2f = ratio %.4f",
            m,
            or_magma_f if np.isfinite(or_magma_f) else float("inf"),
            or_pops_f if np.isfinite(or_pops_f) else float("inf"),
            float(ratio) if isinstance(ratio, (int, float)) else float("nan"),
        )
    results["or_ratios"] = or_ratios

    # ------------------------------------------------------------------
    # Step 7: Annotate overlap genes with symbols.
    # ------------------------------------------------------------------
    logger.info("Step 7: Annotating overlap genes...")
    overlap_detail: dict[str, dict] = {}
    for m in module_names:
        pops_overlap = fisher_pops[m]["overlap_ensgids"]
        magma_overlap = fisher_magma[m]["overlap_ensgids"]
        overlap_detail[m] = {
            "pops_overlap_genes": sorted([ensg2sym.get(e, e) for e in pops_overlap]),
            "magma_overlap_genes": sorted([ensg2sym.get(e, e) for e in magma_overlap]),
            "n_pops_overlap": len(pops_overlap),
            "n_magma_overlap": len(magma_overlap),
        }
    results["overlap_detail"] = overlap_detail

    # ------------------------------------------------------------------
    # Step 8: Decision rule (brief_v2 E3).
    # M1 OR(MAGMA) >= 2.0, p < 0.05 -> GENUINE_SIGNAL_AMPLIFIED_BY_TRAINING
    # M1 OR(MAGMA) < 1.5 -> TRAINING_ARTIFACT
    # M1 OR(MAGMA) in [1.5, 2.0] -> AMBIGUOUS
    # M3: EXPLORATORY (expected overlap ~2-3 genes, insufficient power).
    # ------------------------------------------------------------------
    logger.info("Step 8: Decision rule...")
    m1_magma_or = fisher_magma["M1_neuronal"]["OR"]
    m1_magma_or_f = float(m1_magma_or) if isinstance(m1_magma_or, (int, float)) else float("inf")
    m1_magma_p = fisher_magma["M1_neuronal"]["p_fisher"]

    if m1_magma_or_f >= 2.0 and m1_magma_p < 0.05:
        verdict = "GENUINE_SIGNAL_AMPLIFIED_BY_TRAINING"
        reason = (
            f"M1 neuronal OR(MAGMA)={m1_magma_or_f:.2f} >= 2.0 with "
            f"p={m1_magma_p:.4g} < 0.05. Polygenic risk genuinely concentrates "
            f"in neuronal/synaptic genes. PoPS amplifies this signal by "
            f"incorporating brain expression features."
        )
    elif m1_magma_or_f < 1.5:
        verdict = "TRAINING_ARTIFACT"
        reason = (
            f"M1 neuronal OR(MAGMA)={m1_magma_or_f:.2f} < 1.5. Module "
            f"enrichment in PoPS-top-1000 is driven by PoPS training features, "
            f"not by underlying GWAS signal."
        )
    else:
        verdict = "AMBIGUOUS"
        reason = (
            f"M1 neuronal OR(MAGMA)={m1_magma_or_f:.2f} in [1.5, 2.0] range. "
            f"Some genuine signal may exist but the effect is modest and "
            f"potentially confounded by PoPS training."
        )

    # M3 is exploratory per brief_v2.
    m3_note = (
        f"M3 myelination: EXPLORATORY. OR(MAGMA)="
        f"{fisher_magma['M3_myelination']['OR']}, "
        f"overlap={fisher_magma['M3_myelination']['a_overlap']} genes. "
        f"Expected overlap ~2-3 genes, insufficient power for formal test."
    )

    results["verdict"] = verdict
    results["verdict_reason"] = reason
    results["m3_exploratory_note"] = m3_note

    results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["elapsed_seconds"] = round(time.time() - t0, 1)

    atomic_write_json(results, OUTPUT_DIR / "results.json")
    logger.info(
        "E3 complete. Verdict: %s. Reason: %s. Elapsed: %.1fs",
        verdict, reason, results["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
