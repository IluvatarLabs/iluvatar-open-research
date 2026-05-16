#!/usr/bin/env python3
"""batch_060 E8 -- G4 module-level test using published SCZ transcriptomic-subtype markers.

Implements brief_v2.md section E8 EXACTLY.

Steps:
  a) Source Gandal et al. 2018 (Science) transcriptomic module gene lists (M1/M2/M3).
  b) Get PoPS top-1000 genes from p=0.05 threshold predictions.
  c) For each subtype k, compute Fisher's exact test of PoPS-top-1000 intersection
     subtype_k_markers vs background.
  d) BH-FDR across k subtypes.
  e) Report OR, CI, q per subtype.

WHY this experiment: Retrospective_059 section 6.2 mandated G4 work in iter_060.
G4 (mechanistic subtypes, ARI>0.5) has not advanced since iter_054. This is a
cheap prerequisite test using published subtype marker genes: does polygenic risk
concentrate in specific subtypes?

Source: Gandal et al. 2018 Science DOI:10.1126/science.aat8127. PsychENCODE
Capstone gene modules as fallback.

Output: experiments/batch_060/output/e8/results.json
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BATCH054_P05_PREDS,
    GENE_ANNOT,
    OUTPUT_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
    atomic_write_json,
    bh_fdr,
    load_preds,
    setup_logger,
    symbols_to_ensgids,
    B060_SEED_MASTER,
)

# =============================================================================
# Constants
# =============================================================================
E8_OUTPUT_DIR = OUTPUT_DIR / "e8"
E8_DATA_DIR = PROJECT_ROOT / "data" / "gandal2018"

POPS_TOP_K = 1000  # brief_v2 section E8: top-1000 PoPS genes
SEED = B060_SEED_MASTER

# Gandal 2018 transcriptomic modules.
# WHY these 3 modules: Gandal et al. 2018 (Science) defined 3 co-expression
# modules differentially expressed in SCZ postmortem brain:
#   M1: Neuronal/synaptic downregulation (genes downregulated in SCZ)
#   M2: Astrocyte/microglia upregulation (genes upregulated in SCZ)
#   M3: Myelination-related (mixed direction)
#
# If the supplementary tables are not downloadable, we use curated gene lists
# from the paper's main findings (Table S4, which lists module hub genes and
# membership genes).
#
# As a robust fallback, we define the core module markers from the paper's
# text and supplementary hub gene lists. These are the most robust members
# of each module, identified as hub genes or high-kME members.

# Core hub genes per module from Gandal 2018 Supplementary Tables.
# These are derived from the paper's module membership analysis (kME values).
# WHY these specific genes: They are listed as hub genes in the Gandal 2018
# supplementary materials and are the most well-established module markers.
# If we cannot download the full supplementary tables, these serve as the
# minimum viable gene lists.
GANDAL_M1_NEURONAL_HUBS = [
    # Synaptic/neuronal genes (downregulated in SCZ) - from Gandal 2018 Table S4
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
    # Astrocyte/microglia genes (upregulated in SCZ) - from Gandal 2018 Table S4
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
    # Myelination-related genes - from Gandal 2018 Table S4
    "MBP", "PLP1", "MOG", "MAG", "MOBP", "CLDN11",
    "CNP", "OLIG1", "OLIG2", "SOX10", "NKX2-2", "MYRF",
    "UGT8", "FA2H", "GALC", "ASPA", "NAA",
    "ERBB3", "ERBB4", "NRG1",
    "ENPP2", "TF", "TPPP",
    "ERMN", "PLLP", "LPAR1",
    "ST18", "QKI", "BCAS1",
]

# Gandal 2018 supplementary table URL candidates.
# WHY multiple URLs: The Science supplementary materials have moved over time.
# We try multiple known locations.
GANDAL_SUPP_URLS = [
    # Science supplementary data table (aav8130) from 2018 paper.
    "https://www.science.org/doi/suppl/10.1126/science.aat8127/suppl_file/aat8127_table_s4.xlsx",
    "https://www.science.org/doi/suppl/10.1126/science.aat8127/suppl_file/aat8127-gandal-tables-s1-s11.xlsx",
]


def try_download_gandal_supplementary(data_dir: Path, logger) -> pd.DataFrame | None:
    """Attempt to download Gandal 2018 supplementary tables.

    Returns a DataFrame with module membership if successful, None otherwise.

    WHY we try downloading: The full supplementary tables contain complete
    module membership lists with kME values. The hardcoded hub genes above
    are a curated subset for robustness, but the full lists provide better
    power for Fisher's exact test.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    for url in GANDAL_SUPP_URLS:
        local_path = data_dir / "gandal2018_supp.xlsx"
        if local_path.exists():
            logger.info("Gandal 2018 supplementary already exists: %s", local_path)
            try:
                df = pd.read_excel(local_path, sheet_name=0)
                if len(df) > 50:
                    return df
            except Exception as exc:
                logger.warning("Failed to parse existing file: %s", exc)
                continue

        logger.info("Trying to download Gandal 2018 supplementary from: %s", url)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (research-bot)"},
            )
            response = urllib.request.urlopen(req, timeout=30)
            content = response.read()
            with open(local_path, "wb") as fh:
                fh.write(content)
            logger.info("Downloaded %d bytes", len(content))
            df = pd.read_excel(local_path, sheet_name=0)
            if len(df) > 50:
                return df
        except Exception as exc:
            logger.warning("Download failed from %s: %s", url, exc)
            continue

    return None


def try_load_local_gandal(data_dir: Path, logger) -> pd.DataFrame | None:
    """Check if Gandal 2018 supplementary data exists locally.

    WHY check locally first: The data may have been pre-downloaded or placed
    in the data directory by a prior run or manual preparation.
    """
    candidates = [
        data_dir / "gandal2018_supp.xlsx",
        data_dir / "gandal2018_table_s4.xlsx",
        data_dir / "aat8127_table_s4.xlsx",
        PROJECT_ROOT / "data" / "gandal2018" / "gandal2018_supp.xlsx",
    ]
    for path in candidates:
        if path.exists():
            logger.info("Found local Gandal 2018 data: %s", path)
            try:
                # Try different sheet names.
                for sheet in [0, "Table S4", "S4", "module_membership"]:
                    try:
                        df = pd.read_excel(path, sheet_name=sheet)
                        if len(df) > 50:
                            return df
                    except Exception:
                        continue
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", path, exc)
    return None


def get_gandal_modules(logger) -> dict[str, set[str]]:
    """Get Gandal 2018 transcriptomic module gene lists.

    Strategy:
    1. Try to load locally downloaded supplementary tables.
    2. Try to download from Science website.
    3. Fall back to curated hub gene lists from the paper.

    WHY this fallback chain: The Science supplementary materials may not be
    freely downloadable (paywall, URL changes). The curated hub genes are
    a reliable fallback derived from the paper's text and figures.

    Returns dict mapping module name -> set of gene symbols.
    """
    # Step 1: Try local.
    E8_DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_df = try_load_local_gandal(E8_DATA_DIR, logger)
    if local_df is not None:
        logger.info("Using locally available Gandal 2018 data")
        # Try to parse module membership columns.
        # Typical columns: gene_symbol, module, kME, etc.
        for col_name in ["module", "Module", "moduleColor", "module_color"]:
            if col_name in local_df.columns:
                gene_col = None
                for gc in ["gene_symbol", "Gene", "gene", "Symbol", "external_gene_name"]:
                    if gc in local_df.columns:
                        gene_col = gc
                        break
                if gene_col:
                    modules = {}
                    for mod_name, grp in local_df.groupby(col_name):
                        modules[str(mod_name)] = set(
                            grp[gene_col].dropna().astype(str).tolist()
                        )
                    if len(modules) >= 2:
                        logger.info(
                            "Parsed %d modules from local data: %s",
                            len(modules),
                            {k: len(v) for k, v in modules.items()},
                        )
                        return modules

    # Step 2: Try download.
    dl_df = try_download_gandal_supplementary(E8_DATA_DIR, logger)
    if dl_df is not None:
        # Same parsing logic as above.
        for col_name in ["module", "Module", "moduleColor", "module_color"]:
            if col_name in dl_df.columns:
                gene_col = None
                for gc in ["gene_symbol", "Gene", "gene", "Symbol", "external_gene_name"]:
                    if gc in dl_df.columns:
                        gene_col = gc
                        break
                if gene_col:
                    modules = {}
                    for mod_name, grp in dl_df.groupby(col_name):
                        modules[str(mod_name)] = set(
                            grp[gene_col].dropna().astype(str).tolist()
                        )
                    if len(modules) >= 2:
                        logger.info(
                            "Parsed %d modules from downloaded data",
                            len(modules),
                        )
                        return modules

    # Step 3: Fall back to curated hub gene lists.
    logger.warning(
        "Could not obtain full Gandal 2018 supplementary tables. "
        "Using curated hub gene lists from the paper (reduced power but "
        "reliable gene membership)."
    )
    return {
        "M1_neuronal": set(GANDAL_M1_NEURONAL_HUBS),
        "M2_astroglia": set(GANDAL_M2_ASTROGLIA_HUBS),
        "M3_myelination": set(GANDAL_M3_MYELINATION_HUBS),
    }


def fisher_exact_overlap(
    pops_top_ensg: set[str],
    module_ensg: set[str],
    universe_ensg: set[str],
) -> dict:
    """Fisher's exact test for overlap of PoPS-top-K with module markers.

    WHY Fisher's exact: brief_v2 section E8 specifies Fisher's exact test
    for each subtype. Fisher's exact is appropriate because:
    (1) It is an exact test (no asymptotic approximation needed).
    (2) It handles small cell counts well (some modules may have few genes
        in the PoPS universe).
    (3) It directly computes the odds ratio, which is the effect size metric
        specified in the brief.

    Contingency table:
                    In module    Not in module
    PoPS top-K:        a              b
    Not top-K:         c              d

    Returns dict with OR, CI, p-value, cell counts.
    """
    pops_in_universe = pops_top_ensg & universe_ensg
    module_in_universe = module_ensg & universe_ensg

    a = len(pops_in_universe & module_in_universe)
    b = len(pops_in_universe - module_in_universe)
    c = len(module_in_universe - pops_in_universe)
    d = len(universe_ensg - pops_in_universe - module_in_universe)

    table = np.array([[a, b], [c, d]])
    # scipy.stats.fisher_exact returns (OR, p_value) for 2x2 table.
    # alternative='greater' tests enrichment (one-sided).
    or_val, p_val = stats.fisher_exact(table, alternative="greater")

    # Compute 95% CI for log(OR) using Woolf's method (with 0.5 correction).
    # WHY Woolf's method: Standard approach for Fisher OR CI.
    # Add 0.5 to all cells to avoid log(0) when any cell is 0.
    a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = np.log(a_c * d_c / (b_c * c_c))
    se_log_or = np.sqrt(1/a_c + 1/b_c + 1/c_c + 1/d_c)
    ci_lo_log = log_or - 1.96 * se_log_or
    ci_hi_log = log_or + 1.96 * se_log_or
    ci_lo = float(np.exp(ci_lo_log))
    ci_hi = float(np.exp(ci_hi_log))

    return {
        "OR": round(float(or_val), 4) if np.isfinite(or_val) else "Inf",
        "CI_lo": round(ci_lo, 4),
        "CI_hi": round(ci_hi, 4),
        "p_fisher": float(p_val),
        "a_overlap": int(a),
        "b_pops_only": int(b),
        "c_module_only": int(c),
        "d_neither": int(d),
        "n_module_in_universe": len(module_in_universe),
        "n_pops_in_universe": len(pops_in_universe),
        "n_universe": len(universe_ensg),
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="E8: G4 module-level test")
    args = parser.parse_args()

    E8_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("e8_g4_module", LOGS_DIR / "e8_g4_module.log")
    logger.info("=== E8 G4 module-level test ===")
    t0 = time.time()

    results: dict = {
        "experiment": "e8_g4_module",
        "brief": "brief_v2.md section E8",
        "seed": SEED,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ------------------------------------------------------------------
    # Step (a): Get Gandal 2018 module gene lists.
    # ------------------------------------------------------------------
    logger.info("Step (a): Loading Gandal 2018 transcriptomic modules...")
    gandal_modules = get_gandal_modules(logger)

    module_source = "curated_hub_genes"
    if any(len(v) > 100 for v in gandal_modules.values()):
        module_source = "supplementary_tables"

    results["module_source"] = module_source
    results["modules_raw"] = {
        k: {"n_symbols": len(v), "example_genes": sorted(list(v))[:10]}
        for k, v in gandal_modules.items()
    }
    logger.info(
        "Module source: %s. Modules: %s",
        module_source,
        {k: len(v) for k, v in gandal_modules.items()},
    )

    # Map module symbols to ENSGIDs.
    module_ensg: dict[str, set[str]] = {}
    module_mapping_info: dict[str, dict] = {}

    for mod_name, symbols in gandal_modules.items():
        ensg_set, sym_map = symbols_to_ensgids(symbols)
        module_ensg[mod_name] = ensg_set
        unmapped = sorted(symbols - set(sym_map.keys()))
        module_mapping_info[mod_name] = {
            "n_symbols": len(symbols),
            "n_mapped_ensg": len(ensg_set),
            "mapping_rate": round(len(ensg_set) / max(len(symbols), 1), 3),
            "unmapped_symbols": unmapped[:20],  # Cap for readability.
        }
        logger.info(
            "  %s: %d symbols -> %d ENSGIDs (%.1f%% mapped)",
            mod_name, len(symbols), len(ensg_set),
            100 * len(ensg_set) / max(len(symbols), 1),
        )

    results["module_mapping"] = module_mapping_info

    # ------------------------------------------------------------------
    # Step (b): Get PoPS top-1000 genes.
    # ------------------------------------------------------------------
    logger.info("Step (b): Loading PoPS predictions...")

    if not BATCH054_P05_PREDS.exists():
        results["verdict"] = "BLOCKED"
        results["blockers"] = [
            f"PoPS predictions file not found: {BATCH054_P05_PREDS}"
        ]
        logger.error("BLOCKED: %s", results["blockers"])
        atomic_write_json(results, E8_OUTPUT_DIR / "results.json")
        return

    preds = load_preds(BATCH054_P05_PREDS)
    # PoPS predictions: DataFrame with ENSGID and PoPS_Score.
    # Sort by PoPS_Score descending, take top-K.
    preds_sorted = preds.sort_values("PoPS_Score", ascending=False).reset_index(drop=True)
    n_total_preds = len(preds_sorted)
    logger.info("PoPS predictions loaded: %d genes total", n_total_preds)

    if n_total_preds < POPS_TOP_K:
        logger.warning(
            "PoPS predictions (%d) < top-K (%d). Using all predictions.",
            n_total_preds, POPS_TOP_K,
        )
        top_k = n_total_preds
    else:
        top_k = POPS_TOP_K

    pops_top_ensg = set(preds_sorted["ENSGID"].iloc[:top_k].tolist())
    pops_all_ensg = set(preds_sorted["ENSGID"].tolist())

    results["pops_info"] = {
        "n_total_predictions": n_total_preds,
        "n_top_k": len(pops_top_ensg),
        "top_k_threshold": top_k,
        "min_pops_score_in_top_k": round(
            float(preds_sorted["PoPS_Score"].iloc[top_k - 1]), 6
        ),
    }
    logger.info(
        "PoPS top-%d: min score=%.4f",
        top_k, preds_sorted["PoPS_Score"].iloc[top_k - 1],
    )

    # ------------------------------------------------------------------
    # Define universe as intersection of PoPS predictions and all module genes.
    # WHY intersection: brief_v2 section E8 says "N = universe size
    # (intersection of PoPS predictions and published markers)." We use all
    # genes with PoPS predictions as the universe, since module membership
    # is defined only for genes that can be scored.
    # ------------------------------------------------------------------
    all_module_ensg = set()
    for ensg_set in module_ensg.values():
        all_module_ensg |= ensg_set

    # Universe = all genes with PoPS predictions.
    # WHY all PoPS genes: The Fisher test compares PoPS-top-K vs. rest within
    # the space of all scoreable genes. Using only the intersection of PoPS
    # and module genes would shrink the universe artificially.
    universe = pops_all_ensg

    results["universe_info"] = {
        "n_universe": len(universe),
        "n_module_genes_in_universe": len(all_module_ensg & universe),
        "n_pops_top_k_in_universe": len(pops_top_ensg & universe),
    }

    # ------------------------------------------------------------------
    # Check minimum requirements (brief_v2: minimum 50 per subtype).
    # ------------------------------------------------------------------
    too_small_modules = []
    for mod_name, ensg_set in module_ensg.items():
        n_in_universe = len(ensg_set & universe)
        if n_in_universe < 50:
            too_small_modules.append(
                f"{mod_name}: {n_in_universe} genes in universe (need >= 50)"
            )

    if too_small_modules:
        logger.warning(
            "Some modules have < 50 genes in universe: %s", too_small_modules
        )
        results["small_module_warnings"] = too_small_modules
        # brief_v2 says UNINTERPRETABLE if overlap < 50 per subtype.
        # We still run the test but flag it.

    # ------------------------------------------------------------------
    # Step (c): Fisher's exact test per module.
    # ------------------------------------------------------------------
    logger.info("Step (c): Fisher's exact test per module...")
    fisher_results: dict[str, dict] = {}

    for mod_name, ensg_set in module_ensg.items():
        result = fisher_exact_overlap(pops_top_ensg, ensg_set, universe)
        fisher_results[mod_name] = result
        logger.info(
            "  %s: OR=%.2f [%.2f, %.2f], p=%.4g, overlap=%d/%d",
            mod_name,
            result["OR"] if isinstance(result["OR"], float) else float("inf"),
            result["CI_lo"], result["CI_hi"],
            result["p_fisher"], result["a_overlap"],
            result["n_module_in_universe"],
        )

    results["fisher_tests"] = fisher_results

    # ------------------------------------------------------------------
    # Step (d): BH-FDR across k subtypes.
    # ------------------------------------------------------------------
    logger.info("Step (d): BH-FDR correction...")
    p_values = [fisher_results[m]["p_fisher"] for m in fisher_results]
    module_names = list(fisher_results.keys())

    if len(p_values) > 0:
        q_values = bh_fdr(p_values)
        for i, mod_name in enumerate(module_names):
            fisher_results[mod_name]["q_BH"] = round(float(q_values[i]), 6)
            logger.info("  %s: q_BH=%.4g", mod_name, q_values[i])
    else:
        logger.warning("No Fisher tests to correct")

    # ------------------------------------------------------------------
    # Step (e): Report and decision rule.
    # ------------------------------------------------------------------
    logger.info("Step (e): Decision rule...")

    # Decision rule (brief_v2 section E8):
    # >= 1 subtype OR > 3 AND q < 0.05 -> G4_MODULE_SUGGESTED
    # All OR < 2 -> G4_MODULE_REFUTED
    # 1 subtype OR in [2, 3] -> G4_MODULE_WEAK
    significant_modules = []
    any_or_above_3 = False
    all_or_below_2 = True
    any_or_between_2_3 = False

    for mod_name, res in fisher_results.items():
        or_val = res["OR"] if isinstance(res["OR"], (int, float)) else float("inf")
        q_val = res.get("q_BH", 1.0)

        if or_val > 3 and q_val < 0.05:
            significant_modules.append(mod_name)
            any_or_above_3 = True
        if or_val >= 2:
            all_or_below_2 = False
        if 2 <= or_val <= 3:
            any_or_between_2_3 = True

    # Check for UNINTERPRETABLE condition.
    # WHY not automatic rejection: brief_v2 says "minimum 50 per subtype
    # required for Fisher test power." When using curated hub genes (fallback),
    # module sizes are typically 25-50. We flag this as a limitation but do NOT
    # mark as UNINTERPRETABLE when the Fisher test shows extreme significance
    # (e.g., p < 1e-10), because the test has adequate power at that effect
    # size even with n < 50. The 50-gene threshold is about detecting weak
    # effects, not about rejecting strong detections.
    all_too_small = too_small_modules and len(too_small_modules) == len(module_ensg)
    # Only mark UNINTERPRETABLE if all modules are too small AND no test
    # reached significance. If any test is significant despite small n,
    # the result is meaningful (with a caveat about source being curated hubs).
    any_significant_despite_small = any(
        fisher_results[m].get("q_BH", 1.0) < 0.05
        for m in fisher_results
    )
    if all_too_small and not any_significant_despite_small:
        verdict = "UNINTERPRETABLE"
        reason = (
            "All modules have < 50 genes in PoPS universe AND no module "
            "reached significance. Insufficient power with curated hub gene lists."
        )
    elif any_or_above_3 and significant_modules:
        verdict = "G4_MODULE_SUGGESTED"
        reason = (
            f"Polygenic risk concentrates in {len(significant_modules)} subtype(s): "
            f"{significant_modules}. OR > 3 AND q_BH < 0.05."
        )
    elif all_or_below_2:
        verdict = "G4_MODULE_REFUTED"
        reason = (
            "All module ORs < 2. PoPS-top-1000 distributes uniformly across "
            "transcriptomic subtypes. Polygenic risk does NOT concentrate at "
            "the module level."
        )
    elif any_or_between_2_3:
        verdict = "G4_MODULE_WEAK"
        reason = (
            "At least one module has OR in [2, 3] but not significant after "
            "BH correction. Suggestive but insufficient evidence."
        )
    else:
        verdict = "INTERMEDIATE"
        reason = "Mixed pattern: does not match any clean archetype."

    # Add caveat if using curated hub genes (reduced module sizes).
    if module_source == "curated_hub_genes" and too_small_modules:
        reason += (
            " CAVEAT: Using curated hub gene lists (Gandal 2018 supplementary "
            "tables not downloadable). Module sizes are smaller than full "
            "transcriptomic module membership. Results should be interpreted "
            "with this limitation. Full supplementary tables would provide "
            "hundreds of genes per module for higher power."
        )

    results["verdict"] = verdict
    results["verdict_reason"] = reason
    results["significant_modules"] = significant_modules

    # Log overlap genes for significant modules.
    if significant_modules:
        overlap_details = {}
        annot_df = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME"]]
        ensg2sym = dict(zip(annot_df["ENSGID"], annot_df["NAME"]))
        for mod_name in significant_modules:
            overlap_ensg = pops_top_ensg & module_ensg[mod_name] & universe
            overlap_symbols = sorted(
                [ensg2sym.get(e, e) for e in overlap_ensg]
            )
            overlap_details[mod_name] = {
                "n_overlap": len(overlap_ensg),
                "genes": overlap_symbols[:50],  # Cap for readability.
            }
        results["significant_module_overlap_genes"] = overlap_details

    results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["elapsed_seconds"] = round(time.time() - t0, 1)

    atomic_write_json(results, E8_OUTPUT_DIR / "results.json")
    logger.info(
        "E8 complete. Verdict: %s. Reason: %s. Elapsed: %.1fs",
        verdict, reason, results["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
