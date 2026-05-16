#!/usr/bin/env python3
"""batch_061 E4 -- Leave-CLU-CR1-Out AD complement sensitivity (DESCRIPTIVE).

Implements brief_v2.md section E4 EXACTLY.

Purpose: Quantify how much of the AD complement enrichment signal (F060_06)
is driven by CLU and CR1. Both are established AD GWAS hits (Lambert 2013,
Jansen 2019). This is a DESCRIPTIVE analysis with NO formal decision rule.

Steps:
  1. Load AD MAGMA Z-scores (Entrez-keyed, mapped to ENSGID).
  2. Define complement gene set (MHC-excluded, same as batch_060 E6).
  3. Map gene symbols to ENSGID via gene annotation.
  4. Load gnomAD constraint metrics for covariates.
  5. Run OLS competitive beta for AD under 4 conditions:
     a) Full complement set (reproduces F060_06 baseline)
     b) Without CLU (ENSG00000120885)
     c) Without CR1 (ENSG00000203710)
     d) Without both CLU and CR1
  6. Report DFBETAS for each gene in each condition.

WHY this experiment: F060_06 showed AD complement beta=0.692 with
DFBETAS_max=1.14. CLU and CR1 are the most plausible drivers because
they are established AD GWAS hits AND complement pathway members. But
with n=23 genes, MDE is ~0.60 sigma, so we CANNOT formally distinguish
"CLU/CR1-driven" from "genuine pathway enrichment." We report point
estimates and CIs as descriptive evidence.

Source: F060_06, NOV-060-03, D060_10.

Output: experiments/batch_061/output/e4/results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

# ---------------------------------------------------------------------------
# Imports from batch_060/_common via importlib.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")

_B060_COMMON_PATH = PROJECT_ROOT / "experiments" / "batch_060" / "scripts" / "_common.py"
_spec060 = _ilu.spec_from_file_location("batch060_common_e4", str(_B060_COMMON_PATH))
_b060 = _ilu.module_from_spec(_spec060)  # type: ignore[arg-type]
assert _spec060 is not None and _spec060.loader is not None
_spec060.loader.exec_module(_b060)

# Re-bind names for clarity.
GENE_ANNOT = _b060.GENE_ANNOT
GNOMAD_TSV = _b060.GNOMAD_TSV
BATCH055B_WORK = _b060.BATCH055B_WORK
atomic_write_json = _b060.atomic_write_json
bh_fdr = _b060.bh_fdr
load_gene_annot = _b060.load_gene_annot
load_gnomad_per_brief_v2 = _b060.load_gnomad_per_brief_v2
load_magma_disorder = _b060.load_magma_disorder
build_sub_a_frame = _b060.build_sub_a_frame
setup_logger = _b060.setup_logger
B060_SEED_MASTER = _b060.B060_SEED_MASTER

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_061"
OUTPUT_DIR = BATCH_DIR / "output" / "e4"
LOGS_DIR = BATCH_DIR / "logs"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = B060_SEED_MASTER

# OLS covariates: same as Sub-A v2.1 (brief_v2 shared design).
PRIMARY_COVS = [
    "log10_gene_length", "lof_pLI",
    "log10_exp_lof_plus1", "log10_NSNPS_plus1",
]

# Complement gene set: identical to batch_060 E6 (e4e5e6_env_axis_battery.py).
# Full set of 28 complement pathway genes.
COMPLEMENT_FULL = [
    "C1QA", "C1QB", "C1QC", "C1R", "C1S",
    "C2", "C3", "C4A", "C4B", "C5", "C6", "C7",
    "C8A", "C8B", "C8G", "C9",
    "CFB", "CFD", "CFP", "CFH", "CFI",
    "CR1", "CR2", "CD46", "CD55", "CD59",
    "SERPING1", "CLU",
]

# MHC-region genes to exclude (chr6:25-34Mb): C4A, C4B, C2, CFB.
# WHY these 4: brief_v2 section E6 specifies them; they are within the MHC
# region and their MAGMA Z-scores are confounded by the extended LD structure.
MHC_EXCLUDE = {"C4A", "C4B", "C2", "CFB"}

# Expected chromosome locations for complement genes with short names
# (aliasing-prone per brief_v2). Source: NCBI Gene database (GRCh37).
COMPLEMENT_EXPECTED_CHR = {
    "C3": "19",
    "C5": "9",
    "C6": "5",
    "C7": "5",
    "C8A": "1",
    "C8B": "1",
    "C8G": "9",
    "C9": "5",
}

# CLU and CR1 ENSGIDs (per brief_v2 E4).
CLU_ENSG = "ENSG00000120885"
CR1_ENSG = "ENSG00000203710"


# =============================================================================
# Gene symbol verification (same logic as batch_060 E6)
# =============================================================================
def verify_complement_genes(
    symbols: list[str],
    annot: pd.DataFrame,
    logger,
) -> tuple[set[str], dict[str, str], list[str]]:
    """Map complement gene symbols to ENSGIDs with chromosome verification.

    WHY chromosome verification: short complement gene names (C3, C5, etc.)
    are prone to aliasing with non-complement genes. We verify chromosome
    against known locations per brief_v2 E6 specification.

    Returns:
      ensg_set: set of mapped ENSGIDs
      sym_to_ensg: dict mapping symbol -> ENSGID
      unmapped: list of unmapped symbols
    """
    name_dedup = annot.drop_duplicates(subset="NAME", keep="first").set_index("NAME")
    sym_to_ensg: dict[str, str] = {}
    unmapped: list[str] = []

    for s in symbols:
        ensg = name_dedup["ENSGID"].get(s)
        if ensg is None:
            unmapped.append(s)
            continue
        # Chromosome verification for aliasing-prone names.
        if s in COMPLEMENT_EXPECTED_CHR:
            chr_val = str(name_dedup["CHR"].get(s, "?"))
            expected = COMPLEMENT_EXPECTED_CHR[s]
            if chr_val != expected:
                logger.warning(
                    "%s: chromosome mismatch -- expected chr%s, got chr%s "
                    "(ENSGID=%s). Possible alias collision. DROPPING.",
                    s, expected, chr_val, ensg,
                )
                unmapped.append(s)
                continue
        sym_to_ensg[s] = ensg

    ensg_set = set(sym_to_ensg.values())
    logger.info(
        "Complement: mapped %d/%d symbols. Unmapped: %s",
        len(ensg_set), len(symbols), unmapped if unmapped else "none",
    )
    return ensg_set, sym_to_ensg, unmapped


# =============================================================================
# OLS competitive regression for AD
# =============================================================================
def run_ols_competitive(
    frame: pd.DataFrame,
    gene_set_ensg: set[str],
    covs: list[str],
    indicator_col: str = "in_set",
) -> dict:
    """Run OLS competitive regression: MAGMA_Z ~ in_set + covariates.

    WHY OLS (not robust): This is a descriptive analysis. OLS gives the
    standard point estimate and SE. Huber/Tukey would give slightly different
    estimates but the brief specifies OLS for the primary report.

    Returns: dict with beta, se, CI, p-value, R-squared.
    """
    frame = frame.copy()
    frame[indicator_col] = frame["ENSGID"].isin(gene_set_ensg).astype(int)
    n_in_set = int(frame[indicator_col].sum())

    if n_in_set < 3:
        return {
            "status": "failed",
            "reason": f"Too few genes in set: {n_in_set}",
            "n_in_set": n_in_set,
        }

    X = frame[[indicator_col] + covs].to_numpy(dtype=float)
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")

    try:
        model = sm.OLS(y, Xc).fit()
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}

    # Indicator is column 1 (after constant at column 0).
    beta = float(model.params[1])
    se = float(model.bse[1])
    ci_lo = float(model.conf_int(alpha=0.05)[1][0])
    ci_hi = float(model.conf_int(alpha=0.05)[1][1])
    p_two = float(model.pvalues[1])
    # One-sided p for enrichment (positive beta).
    p_one = p_two / 2.0 if beta > 0 else 1.0 - p_two / 2.0

    return {
        "status": "ok",
        "beta": round(beta, 4),
        "se": round(se, 4),
        "CI_lo": round(ci_lo, 4),
        "CI_hi": round(ci_hi, 4),
        "p_two_sided": float(p_two),
        "p_one_sided": float(p_one),
        "r_squared": round(float(model.rsquared), 6),
        "n_universe": len(frame),
        "n_in_set": n_in_set,
    }


# =============================================================================
# DFBETAS computation
# =============================================================================
def compute_per_gene_dfbetas(
    frame: pd.DataFrame,
    gene_set_ensg: set[str],
    covs: list[str],
    ensg2sym: dict[str, str],
    indicator_col: str = "in_set",
) -> dict:
    """Compute DFBETAS for each gene in the gene set.

    WHY DFBETAS: brief_v2 E4 specifies "DFBETAS per gene." DFBETAS
    quantifies how much each observation shifts the indicator coefficient.
    A large DFBETAS for CLU or CR1 would indicate those genes are driving
    the overall beta.

    Returns: dict mapping gene symbol -> DFBETAS value, plus summary stats.
    """
    frame = frame.copy()
    frame[indicator_col] = frame["ENSGID"].isin(gene_set_ensg).astype(int)

    X = frame[[indicator_col] + covs].to_numpy(dtype=float)
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")

    try:
        ols_fit = sm.OLS(y, Xc).fit()
        infl = OLSInfluence(ols_fit)
        dfbetas = np.asarray(infl.dfbetas)
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}

    if dfbetas.shape[1] <= 1:
        return {"status": "failed", "reason": f"dfbetas shape: {dfbetas.shape}"}

    # Indicator coefficient is column 1 (after constant).
    b_col = dfbetas[:, 1]

    # Extract DFBETAS for each gene in the set.
    per_gene: dict[str, float] = {}
    for idx_pos, (_, row) in enumerate(frame.iterrows()):
        ensg = row["ENSGID"]
        if ensg in gene_set_ensg:
            sym = ensg2sym.get(ensg, ensg)
            per_gene[sym] = round(float(b_col[idx_pos]), 4)

    max_abs = float(np.nanmax(np.abs(b_col))) if b_col.size else float("nan")
    argmax_idx = int(np.nanargmax(np.abs(b_col))) if b_col.size else -1
    argmax_ensg = frame.iloc[argmax_idx]["ENSGID"] if argmax_idx >= 0 else "unknown"
    argmax_sym = ensg2sym.get(argmax_ensg, argmax_ensg)

    return {
        "status": "ok",
        "per_gene_dfbetas": per_gene,
        "max_abs_dfbetas": round(max_abs, 4),
        "max_abs_gene": argmax_sym,
        "max_abs_ensgid": argmax_ensg,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("e4_complement_sensitivity", LOGS_DIR / "e4_complement_sensitivity.log")
    logger.info("=== E4: Leave-CLU-CR1-Out AD Complement Sensitivity ===")
    t0 = time.time()

    results: dict = {
        "experiment": "E4_complement_sensitivity",
        "brief": "brief_v2.md section E4",
        "analysis_type": "DESCRIPTIVE -- no formal decision rule",
        "seed": SEED,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ------------------------------------------------------------------
    # Step 1: Load gene annotation + gnomAD.
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading annotations and gnomAD...")
    annot = load_gene_annot()
    gnomad = load_gnomad_per_brief_v2()
    ensg2sym = dict(zip(annot["ENSGID"], annot["NAME"]))
    logger.info("gene_annot: %d genes. gnomAD: %d genes.", len(annot), len(gnomad))

    # ------------------------------------------------------------------
    # Step 2: Map complement gene set (MHC-excluded).
    # ------------------------------------------------------------------
    logger.info("Step 2: Mapping complement genes (MHC-excluded)...")
    complement_symbols = [s for s in COMPLEMENT_FULL if s not in MHC_EXCLUDE]
    full_ensg, full_sym_map, full_unmapped = verify_complement_genes(
        complement_symbols, annot, logger,
    )

    results["gene_set"] = {
        "name": "Complement_MHC_excluded",
        "source": (
            "Kim et al. 2021 Nat Neurosci, same as batch_060 E6. "
            "MHC genes excluded: C4A, C4B, C2, CFB."
        ),
        "n_requested": len(complement_symbols),
        "n_mapped": len(full_ensg),
        "unmapped": full_unmapped,
        "symbol_to_ensgid": full_sym_map,
    }

    # Verify CLU and CR1 are in the mapped set.
    clu_mapped = CLU_ENSG in full_ensg
    cr1_mapped = CR1_ENSG in full_ensg
    logger.info("CLU (%s) mapped: %s. CR1 (%s) mapped: %s.",
                CLU_ENSG, clu_mapped, CR1_ENSG, cr1_mapped)
    results["clu_mapped"] = clu_mapped
    results["cr1_mapped"] = cr1_mapped

    if not clu_mapped or not cr1_mapped:
        results["verdict"] = "BLOCKED"
        results["blockers"] = [
            f"CLU mapped={clu_mapped}, CR1 mapped={cr1_mapped}. "
            "Cannot run leave-out analysis without both mapped."
        ]
        logger.error("BLOCKED: %s", results["blockers"])
        results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["elapsed_seconds"] = round(time.time() - t0, 1)
        atomic_write_json(results, OUTPUT_DIR / "results.json")
        return

    # ------------------------------------------------------------------
    # Step 3: Build AD regression frame.
    # WHY build_sub_a_frame: Cardinal Rule 1. This function assembles the
    # standard MAGMA-Z regression frame with gnomAD covariates.
    # ------------------------------------------------------------------
    logger.info("Step 3: Building AD regression frame...")
    try:
        frame = build_sub_a_frame("alzheimers", gnomad, annot, full_ensg)
    except Exception as exc:
        results["verdict"] = "BLOCKED"
        results["blockers"] = [f"build_sub_a_frame failed: {exc}"]
        logger.exception("BLOCKED: build_sub_a_frame failed")
        results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["elapsed_seconds"] = round(time.time() - t0, 1)
        atomic_write_json(results, OUTPUT_DIR / "results.json")
        return

    logger.info("AD frame: %d genes, %d in complement set",
                len(frame), int(frame["in_set"].sum()))

    # ------------------------------------------------------------------
    # Step 4: Run OLS under 4 conditions.
    # ------------------------------------------------------------------
    logger.info("Step 4: Running OLS under 4 conditions...")

    # Define gene set variants.
    conditions = {
        "full": full_ensg,
        "without_CLU": full_ensg - {CLU_ENSG},
        "without_CR1": full_ensg - {CR1_ENSG},
        "without_CLU_CR1": full_ensg - {CLU_ENSG, CR1_ENSG},
    }

    ols_results: dict[str, dict] = {}
    for cond_name, gene_set in conditions.items():
        ols_res = run_ols_competitive(frame, gene_set, PRIMARY_COVS)
        ols_results[cond_name] = ols_res
        if ols_res["status"] == "ok":
            logger.info(
                "  %s: beta=%.4f [%.4f, %.4f], p_one=%.4g, n_in_set=%d",
                cond_name, ols_res["beta"], ols_res["CI_lo"], ols_res["CI_hi"],
                ols_res["p_one_sided"], ols_res["n_in_set"],
            )
        else:
            logger.warning("  %s: %s", cond_name, ols_res.get("reason"))

    results["ols_conditions"] = ols_results

    # ------------------------------------------------------------------
    # Step 5: DFBETAS per gene for each condition.
    # WHY per-condition DFBETAS: brief_v2 E4 specifies "DFBETAS per gene."
    # Removing CLU/CR1 changes the DFBETAS of remaining genes (because the
    # OLS fit changes). We report DFBETAS under each condition for
    # transparency.
    # ------------------------------------------------------------------
    logger.info("Step 5: Computing DFBETAS per gene for each condition...")
    dfbetas_results: dict[str, dict] = {}
    for cond_name, gene_set in conditions.items():
        dfb = compute_per_gene_dfbetas(
            frame, gene_set, PRIMARY_COVS, ensg2sym,
        )
        dfbetas_results[cond_name] = dfb
        if dfb["status"] == "ok":
            logger.info(
                "  %s: max_abs_dfbetas=%.4f (%s), n_genes=%d",
                cond_name, dfb["max_abs_dfbetas"], dfb["max_abs_gene"],
                len(dfb["per_gene_dfbetas"]),
            )
        else:
            logger.warning("  %s DFBETAS failed: %s", cond_name, dfb.get("reason"))

    results["dfbetas_conditions"] = dfbetas_results

    # ------------------------------------------------------------------
    # Step 6: Compute beta change summary.
    # WHY: This is the primary descriptive output. We quantify how much
    # beta drops when CLU/CR1 are removed, with CIs.
    # ------------------------------------------------------------------
    logger.info("Step 6: Beta change summary...")
    full_res = ols_results["full"]
    summary: dict[str, dict] = {}

    if full_res["status"] == "ok":
        full_beta = full_res["beta"]
        for cond_name in ["without_CLU", "without_CR1", "without_CLU_CR1"]:
            cond_res = ols_results[cond_name]
            if cond_res["status"] == "ok":
                delta = round(cond_res["beta"] - full_beta, 4)
                pct_change = round(100 * delta / full_beta, 1) if full_beta != 0 else "N/A"
                summary[cond_name] = {
                    "beta_full": full_beta,
                    "beta_reduced": cond_res["beta"],
                    "delta_beta": delta,
                    "pct_change": pct_change,
                    "CI_reduced": [cond_res["CI_lo"], cond_res["CI_hi"]],
                    "n_genes_removed": len(full_ensg) - len(conditions[cond_name]),
                    "n_remaining": cond_res["n_in_set"],
                }
                logger.info(
                    "  %s: beta %.4f -> %.4f (delta=%.4f, %.1f%%)",
                    cond_name, full_beta, cond_res["beta"], delta,
                    pct_change if isinstance(pct_change, (int, float)) else 0,
                )

    results["beta_change_summary"] = summary

    # ------------------------------------------------------------------
    # Step 7: Descriptive conclusion (NO formal decision rule).
    # ------------------------------------------------------------------
    results["descriptive_note"] = (
        "This is a DESCRIPTIVE analysis. With n=21-23 complement genes, "
        "the MDE is approximately 0.60 sigma. The analysis CANNOT formally "
        "distinguish 'CLU/CR1-driven signal' from 'genuine pathway enrichment.' "
        "Point estimates and CIs are reported for interpretation alongside "
        "domain knowledge about CLU and CR1 as established AD GWAS hits."
    )

    results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["elapsed_seconds"] = round(time.time() - t0, 1)

    atomic_write_json(results, OUTPUT_DIR / "results.json")
    logger.info(
        "E4 complete. Elapsed: %.1fs. This is DESCRIPTIVE -- no formal verdict.",
        results["elapsed_seconds"],
    )


if __name__ == "__main__":
    main()
