#!/usr/bin/env python3
"""batch_060 E4/E5/E6 — Environmental-axis gene-set batteries.

Implements brief_v2.md sections E4, E5, E6 EXACTLY.

Three gene sets tested against 8 disorders using the Sub-A v2.1 battery
(batch_058 framework):
  E4: IEG rPRGs (Tyssowski 2018, n=19)
  E5: GR targets (Reddy 2009, n=22)
  E6: Complement cascade (Kim 2021 null replication)
      - PRIMARY: MHC-excluded (n=24)
      - SENSITIVITY: full complement (n=28)

Per gene set:
  1. Gene symbol verification via gene_annot_jun10.txt (NAME -> ENSGID).
     Chromosome location verification for complement short names.
     Abort if >3 missing per set (UNINTERPRETABLE threshold from brief).
  2. Sub-A v2.1 battery per disorder (8 disorders):
     - OLS beta + 95% CI
     - HuberT beta + 95% CI
     - TukeyBiweight proxy
     - Rank-MAGMA OLS (sign concordance)
     - DFBETAS (max, plus specific DFBETAS for DUSP1)
  3. BH-FDR across 8 OLS one-sided p-values (one family per gene set).
  4. LOEUF sensitivity: for disorders reaching q < 0.05, rerun with
     lof_oe_ci_upper instead of lof_pLI.
  5. Size-matched random gene set control: for q < 0.05, 10,000 random
     gene sets matched on gene-length decile, OLS pipeline, empirical
     percentile of observed beta.
  6. Reproduction gate: SCZ B3 beta_OLS must be in [+2.5, +3.5].
  7. Decision rules per brief_v2 sections E4/E5/E6.

Output:
  experiments/batch_060/output/e4/results.json
  experiments/batch_060/output/e5/results.json
  experiments/batch_060/output/e6/results.json

Cardinal rules:
  - Rule 0: No fabrication. Unmapped genes logged, never silently dropped.
  - Rule 1: Import Sub-A v2.1 battery from batch_058, not reimplemented.
  - Rule 5: Every decision documented with WHY.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

# ---------------------------------------------------------------------------
# Imports from batch_060/_common (which re-exports batch_059/058/057/056).
# WHY importlib instead of sys.path: all _common.py files share the same
# filename; sys.path would shadow this script's _common imports. importlib
# loads under distinct module names.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")

_B060_COMMON_PATH = PROJECT_ROOT / "experiments" / "batch_060" / "scripts" / "_common.py"
_spec060 = _ilu.spec_from_file_location("batch060_common", str(_B060_COMMON_PATH))
_b060 = _ilu.module_from_spec(_spec060)  # type: ignore[arg-type]
assert _spec060 is not None and _spec060.loader is not None
_spec060.loader.exec_module(_b060)

# Also load batch_058 sub_a_robust_battery for its fit_ols, fit_huber, etc.
# WHY: Cardinal Rule 1. These functions are parameterized by indicator column
# and are reusable for arbitrary gene-set indicators.
_B058_SUBA_PATH = (
    PROJECT_ROOT / "experiments" / "batch_058" / "scripts" / "sub_a_robust_battery.py"
)
_spec058suba = _ilu.spec_from_file_location("batch058_sub_a", str(_B058_SUBA_PATH))
_b058_suba = _ilu.module_from_spec(_spec058suba)  # type: ignore[arg-type]
assert _spec058suba is not None and _spec058suba.loader is not None
_spec058suba.loader.exec_module(_b058_suba)

# ---------------------------------------------------------------------------
# Re-bind names for clarity and auditor traceability.
# ---------------------------------------------------------------------------
# From batch_060 _common (transitive):
BATCH_DIR = _b060.BATCH_DIR
OUTPUT_DIR = _b060.OUTPUT_DIR
LOGS_DIR = _b060.LOGS_DIR
GENE_ANNOT = _b060.GENE_ANNOT
GNOMAD_TSV = _b060.GNOMAD_TSV
MAGMA_GENELOC = _b060.MAGMA_GENELOC
DISORDERS = _b060.DISORDERS
PSYCHIATRIC = _b060.PSYCHIATRIC
LOEUF_DISORDERS = _b060.LOEUF_DISORDERS
BH_Q = _b060.BH_Q
DFBETAS_CUTOFF = _b060.DFBETAS_CUTOFF
REPRO_R1_SUB_A_LO = _b060.REPRO_R1_SUB_A_LO
REPRO_R1_SUB_A_HI = _b060.REPRO_R1_SUB_A_HI
B3_GENES = _b060.B3_GENES
B060_SEED_MASTER = _b060.B060_SEED_MASTER

load_gene_annot = _b060.load_gene_annot
load_gnomad_per_brief_v2 = _b060.load_gnomad_per_brief_v2
build_sub_a_frame = _b060.build_sub_a_frame
bh_fdr = _b060.bh_fdr
atomic_write_json = _b060.atomic_write_json
sha256_file = _b060.sha256_file
setup_logger = _b060.setup_logger
symbols_to_ensgids = _b060.symbols_to_ensgids
rank_gaussianize = _b060.rank_gaussianize
fit_tukey_biweight = _b060.fit_tukey_biweight
# compute_dfbetas_cooks is NOT imported — replaced by compute_influence_combined
# which merges DFBETAS + Cook's D + per-gene DFBETAS into a single
# OLSInfluence pass to halve wall time.

# From batch_058 sub_a_robust_battery:
# WHY import these functions: they implement the exact OLS, Huber, rank-MAGMA
# fits specified by Sub-A v2.1 (brief_v2.md section Sub-A). Reimplementing
# them would violate Cardinal Rule 1.
fit_ols = _b058_suba.fit_ols
fit_huber = _b058_suba.fit_huber
fit_rank_magma_ols = _b058_suba.fit_rank_magma_ols
run_loeuf_sensitivity = _b058_suba.run_loeuf_sensitivity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 20260424  # design.yaml master seed

# Covariates: same as Sub-A v2.1 (brief_v2).
PRIMARY_COVS = [
    "log10_gene_length", "lof_pLI",
    "log10_exp_lof_plus1", "log10_NSNPS_plus1",
]
LOEUF_COVS = [
    "log10_gene_length", "lof_oe_ci_upper",
    "log10_exp_lof_plus1", "log10_NSNPS_plus1",
]

MIN_N_UNIVERSE = 15000  # minimum gene universe size for interpretable result

# Size-matched null parameters (brief_v2 section "Size-Matched Random Gene
# Set Control").
SIZE_MATCHED_N_DRAWS = 10000
SIZE_MATCHED_N_DECILES = 10

# Reproduction gate (brief_v2 section "Shared Design Elements").
REPRO_GATE_LO = 2.5
REPRO_GATE_HI = 3.5

# UNINTERPRETABLE threshold: >3 missing genes per set (brief_v2).
MAX_MISSING_GENES = 3

# ---------------------------------------------------------------------------
# Gene sets (brief_v2 section "Gene Lists")
# ---------------------------------------------------------------------------
IEG_RPRGS = [
    "FOS", "FOSB", "JUNB", "EGR1", "EGR2", "EGR3", "EGR4",
    "NR4A1", "NR4A2", "NR4A3", "ARC", "NPAS4", "BTG2",
    "DUSP1", "DUSP5", "GADD45B", "GADD45G", "ATF3", "PPP1R15A",
]

GR_TARGETS = [
    "FKBP5", "TSC22D3", "SGK1", "PER1", "DUSP1", "KLF15",
    "ZBTB16", "NFKBIA", "CDKN1A", "TXNIP", "DDIT4", "MT2A",
    "IL1R2", "VIPR1", "ANGPTL4", "GLUL", "PDK4", "ERRFI1",
    "SCNN1A", "CEBPD", "KLF9", "TIPARP",
]

# Full complement set (n=28), including MHC-region genes.
COMPLEMENT_FULL = [
    "C1QA", "C1QB", "C1QC", "C1R", "C1S",
    "C2", "C3", "C4A", "C4B", "C5", "C6", "C7",
    "C8A", "C8B", "C8G", "C9",
    "CFB", "CFD", "CFP", "CFH", "CFI",
    "CR1", "CR2", "CD46", "CD55", "CD59",
    "SERPING1", "CLU",
]

# MHC-region genes to exclude for PRIMARY analysis (chr6:25-34Mb).
# WHY these 4: brief_v2 section E6 specifies C4A, C4B, C2, CFB as MHC genes.
MHC_EXCLUDE = {"C4A", "C4B", "C2", "CFB"}

# Expected chromosome locations for complement genes with short ambiguous names.
# WHY: brief_v2 section "Gene Symbol Verification" warns C3/C5/C6/C7/C8A/C8B/
# C8G/C9 are aliasing-prone. We verify by chromosome.
# Source: NCBI Gene database (GRCh37 coordinates, matching gene_annot_jun10.txt).
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


# ===========================================================================
# Gene symbol verification
# ===========================================================================
def verify_gene_set(
    symbols: list[str],
    annot: pd.DataFrame,
    logger: logging.Logger,
    set_name: str,
    expected_chr: dict[str, str] | None = None,
) -> tuple[set[str], dict[str, str], list[str], list[str]]:
    """Map gene symbols to ENSGIDs via gene_annot NAME column.

    Returns:
      ensg_set: set of mapped ENSGIDs
      sym_to_ensg: dict mapping symbol -> ENSGID
      unmapped: list of symbols not found in gene_annot
      multi_mapped: list of symbols with >1 ENSGID in gene_annot

    WHY gene_annot (not MAGMA geneloc): gene_annot_jun10.txt is the canonical
    ENSGID<->symbol bridge used by all downstream loaders (build_sub_a_frame,
    load_magma_disorder, etc.). Using MAGMA geneloc would give Entrez IDs that
    still need mapping to ENSGID via gene_annot. Going directly to gene_annot
    avoids a double-mapping step.

    WHY chromosome verification for complement genes: gene names like C3, C5,
    C6, C7, C9 are short and prone to aliasing with non-complement genes.
    brief_v2 mandates chromosome location verification. We cross-check the
    CHR column in gene_annot against known complement gene chromosomes.
    """
    # Build name -> ENSGID lookup. Use drop_duplicates(keep="first") to match
    # the convention in symbols_to_ensgids (batch_057/_common.py).
    name_dedup = annot.drop_duplicates(subset="NAME", keep="first").set_index("NAME")

    # Also check for multi-mapped symbols (same NAME -> multiple ENSGIDs).
    name_counts = annot.groupby("NAME")["ENSGID"].nunique()
    multi_syms = set(name_counts[name_counts > 1].index)

    sym_to_ensg: dict[str, str] = {}
    unmapped: list[str] = []
    multi_mapped: list[str] = []
    chr_mismatches: list[str] = []

    for s in symbols:
        if s in multi_syms:
            multi_mapped.append(s)
            # Still use the first mapping (consistent with symbols_to_ensgids).
            # Log the multi-mapping for audit.
        ensg = name_dedup["ENSGID"].get(s)
        if ensg is None:
            unmapped.append(s)
            continue
        # Chromosome verification for complement genes.
        if expected_chr and s in expected_chr:
            chr_val = str(name_dedup["CHR"].get(s, "?"))
            expected = expected_chr[s]
            if chr_val != expected:
                chr_mismatches.append(
                    f"{s}: expected chr{expected}, got chr{chr_val}"
                )
                logger.warning(
                    "%s: %s chromosome mismatch — expected chr%s, "
                    "got chr%s (ENSGID=%s). Possible alias collision.",
                    set_name, s, expected, chr_val, ensg,
                )
                # Do NOT map this gene — it may be an alias for a different gene.
                unmapped.append(s)
                continue
        sym_to_ensg[s] = ensg

    ensg_set = set(sym_to_ensg.values())

    logger.info(
        "%s: mapped %d/%d symbols to ENSGIDs. Unmapped: %s. "
        "Multi-mapped: %s. Chr mismatches: %s.",
        set_name, len(ensg_set), len(symbols),
        unmapped if unmapped else "none",
        multi_mapped if multi_mapped else "none",
        chr_mismatches if chr_mismatches else "none",
    )
    return ensg_set, sym_to_ensg, unmapped, multi_mapped


# ===========================================================================
# Combined DFBETAS + Cook's D + per-gene DFBETAS (single OLSInfluence call)
# ===========================================================================
def compute_influence_combined(
    frame: pd.DataFrame,
    covs: list[str],
    indicator_col: str,
    target_ensgid: str | None = None,
) -> tuple[dict, float | None]:
    """Compute DFBETAS/Cook's D AND specific-gene DFBETAS in ONE OLSInfluence pass.

    WHY combined: OLSInfluence is O(N^2) for N~16,500 and takes ~60s per call.
    compute_dfbetas_cooks and compute_dfbetas_for_gene would each do a full
    pass (2x60s per disorder). Merging them into a single call halves the time.

    Returns:
      (influence_dict, dfbetas_target)
      - influence_dict: same schema as compute_dfbetas_cooks from batch_058
      - dfbetas_target: DFBETAS value for target_ensgid (or None if not found)
    """
    X = frame[[indicator_col] + covs].to_numpy(dtype=float)
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    try:
        ols_fit = sm.OLS(y, Xc).fit()
        infl = OLSInfluence(ols_fit)
        dfbetas = np.asarray(infl.dfbetas)
        cooks = np.asarray(infl.cooks_distance[0])
    except Exception as exc:
        return (
            {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"},
            None,
        )
    if dfbetas.shape[1] <= 1:
        return (
            {"status": "failed",
             "reason": f"dfbetas has too few cols: {dfbetas.shape}"},
            None,
        )
    # Indicator coefficient is column 1 (after constant at column 0).
    b_col = dfbetas[:, 1]
    max_abs = float(np.nanmax(np.abs(b_col))) if b_col.size else float("nan")
    argmax_abs = int(np.nanargmax(np.abs(b_col))) if b_col.size else -1
    cooks_max = float(np.nanmax(cooks)) if cooks.size else float("nan")
    cooks_mean = float(np.nanmean(cooks)) if cooks.size else float("nan")
    cooks_argmax = int(np.nanargmax(cooks)) if cooks.size else -1

    influence_dict = {
        "status": "ok",
        "max_abs_dfbetas_b3": max_abs,
        "argmax_dfbetas_b3_idx": argmax_abs,
        "cooks_max": cooks_max,
        "cooks_mean": cooks_mean,
        "cooks_argmax_idx": cooks_argmax,
    }

    # Per-gene DFBETAS for target_ensgid (e.g. DUSP1).
    dfbetas_target = None
    if target_ensgid is not None and target_ensgid in frame["ENSGID"].values:
        idx = frame.index[frame["ENSGID"] == target_ensgid].tolist()
        if idx:
            pos = frame.index.get_loc(idx[0])
            dfbetas_target = float(b_col[pos])

    return influence_dict, dfbetas_target


# ===========================================================================
# Run one disorder through the full Sub-A v2.1 battery
# ===========================================================================
def run_disorder_battery(
    disorder: str,
    gnomad: pd.DataFrame,
    annot: pd.DataFrame,
    gene_set_ensg: set[str],
    dusp1_ensgid: str | None,
    logger: logging.Logger,
    indicator_col: str = "in_set",
) -> dict:
    """Run full Sub-A v2.1 diagnostic battery for one disorder and gene set.

    WHY not directly call batch_058's run_disorder: that function hardcodes
    smoke_frame_size logic and references B3-specific variables. We replicate
    the same battery pipeline here but parameterized for arbitrary gene sets.
    The individual fit functions (fit_ols, fit_huber, fit_rank_magma_ols,
    fit_tukey_biweight) are imported from batch_058 to avoid reimplementation
    (Cardinal Rule 1). Influence diagnostics use compute_influence_combined
    (this script) which merges DFBETAS + Cook's D + per-gene DFBETAS into
    a single OLSInfluence pass.
    """
    try:
        frame = build_sub_a_frame(
            disorder, gnomad, annot, gene_set_ensg,
            gene_set_col=indicator_col,
        )
    except Exception as exc:
        logger.exception("build_sub_a_frame failed for %s", disorder)
        return {"status": "failed", "reason": str(exc)}

    n = len(frame)
    n_in_set = int(frame[indicator_col].sum())
    logger.info("%s: n=%d in_set=%d", disorder, n, n_in_set)

    if n < MIN_N_UNIVERSE:
        return {"status": "failed", "reason": f"n={n} < {MIN_N_UNIVERSE}"}
    # WHY n_in_set >= 5 (not 10 like B3): small gene sets (19-28 genes) may
    # lose some members to the inner join with gnomAD/NSNPS. We use 5 as the
    # floor for regression interpretability (at least 5 genes needed for
    # stable indicator coefficient).
    if n_in_set < 5:
        return {"status": "failed", "reason": f"in_set={n_in_set} < 5"}

    ols = fit_ols(frame, indicator_col, PRIMARY_COVS)
    huber = fit_huber(frame, indicator_col, PRIMARY_COVS)
    tukey = fit_tukey_biweight(frame, PRIMARY_COVS, indicator_col)
    rank_ols = fit_rank_magma_ols(frame, indicator_col, PRIMARY_COVS)

    # Combined DFBETAS + Cook's D + per-gene DFBETAS in a single
    # OLSInfluence pass (WHY: OLSInfluence is O(N^2); merging avoids
    # a redundant ~60s computation per disorder).
    infl, dfbetas_dusp1 = compute_influence_combined(
        frame, PRIMARY_COVS, indicator_col, dusp1_ensgid,
    )

    return {
        "status": "ok",
        "n_gene_universe": n,
        "n_set_in_universe": n_in_set,
        "ols": ols,
        "huber": huber,
        "tukey": tukey,
        "rank_magma_ols": rank_ols,
        "influence": infl,
        "dfbetas_dusp1": dfbetas_dusp1,
    }


# ===========================================================================
# LOEUF sensitivity for a single disorder + gene set
# ===========================================================================
def run_loeuf_one(
    disorder: str,
    gnomad: pd.DataFrame,
    annot: pd.DataFrame,
    gene_set_ensg: set[str],
    logger: logging.Logger,
    indicator_col: str = "in_set",
) -> dict:
    """OLS + Huber + Tukey with lof_pLI swapped to lof_oe_ci_upper.

    WHY: brief_v2 section E4 MEASUREMENT: "LOEUF sensitivity. For any disorder
    reaching q < 0.05, rerun with LOEUF covariates." This tests whether the
    enrichment signal is robust to constraint metric choice.
    """
    try:
        frame = build_sub_a_frame(
            disorder, gnomad, annot, gene_set_ensg,
            gene_set_col=indicator_col,
        )
    except Exception as exc:
        logger.exception("LOEUF build failed for %s", disorder)
        return {"status": "failed", "reason": str(exc)}
    frame = frame.dropna(subset=["lof_oe_ci_upper"]).copy()
    n_in_set = int(frame[indicator_col].sum()) if len(frame) else 0
    if frame.empty or n_in_set < 5:
        return {"status": "skipped",
                "reason": f"n={len(frame)} in_set={n_in_set}"}
    ols = fit_ols(frame, indicator_col, LOEUF_COVS)
    huber = fit_huber(frame, indicator_col, LOEUF_COVS)
    tukey = fit_tukey_biweight(frame, LOEUF_COVS, indicator_col)
    return {
        "status": "ok",
        "ols": ols, "huber": huber, "tukey": tukey,
        "covariates": LOEUF_COVS,
    }


# ===========================================================================
# Size-matched random gene set control
# ===========================================================================
def size_matched_null(
    frame: pd.DataFrame,
    observed_beta: float,
    gene_set_size: int,
    gene_set_ensg: set[str],
    indicator_col: str,
    logger: logging.Logger,
    seed: int = SEED,
) -> dict:
    """Draw 10,000 random gene sets matched on gene-length decile distribution.

    WHY length-decile matching: brief_v2 section "Size-Matched Random Gene Set
    Control" mandates matching on the gene-length decile distribution of the
    test gene set. This ensures the null distribution controls for the known
    confound between gene length and MAGMA-Z.

    Algorithm:
      1. Compute gene-length deciles for all genes in the regression frame.
      2. Count how many test-set genes fall in each decile.
      3. For each of 10,000 draws, sample the same number of genes from each
         decile (sampling without replacement within each decile).
      4. Run OLS on each draw, record beta.
      5. Report empirical percentile of observed beta.

    Returns dict with null_betas_mean, null_betas_sd, empirical_percentile,
    n_draws, n_draws_failed.
    """
    rng = np.random.default_rng(seed)

    # Decile assignment.
    frame = frame.copy()
    frame["length_decile"] = pd.qcut(
        frame["log10_gene_length"], q=SIZE_MATCHED_N_DECILES,
        labels=False, duplicates="drop",
    )

    # Count test-set genes per decile.
    in_set_mask = frame[indicator_col] == 1
    decile_counts = (
        frame.loc[in_set_mask, "length_decile"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    # Available genes per decile (excluding test set to avoid self-sampling).
    out_set_mask = ~in_set_mask
    decile_pools: dict[int, np.ndarray] = {}
    for d_val in sorted(decile_counts.keys()):
        pool_idx = frame.index[out_set_mask & (frame["length_decile"] == d_val)].to_numpy()
        decile_pools[d_val] = pool_idx
        if len(pool_idx) < decile_counts[d_val]:
            logger.warning(
                "Decile %d: need %d genes but only %d available in pool. "
                "Will sample with replacement for this decile.",
                d_val, decile_counts[d_val], len(pool_idx),
            )

    # Pre-compute regression components for speed.
    # WHY pre-compute: running 10,000 full build_sub_a_frame calls would be
    # prohibitively slow. Instead we pre-compute the design matrix (constant +
    # covariates) and only change the indicator column per draw. We use
    # np.linalg.lstsq directly (not sm.OLS) to avoid the overhead of
    # statsmodels result objects (~1ms vs ~10ms per fit at N=16k).
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
    cov_cols = PRIMARY_COVS
    X_cov = frame[cov_cols].to_numpy(dtype=float)
    n = len(frame)
    # Pre-build the constant + covariates portion of the design matrix.
    # Layout: [const, indicator, cov1, cov2, ...]. indicator at col 1.
    X_base = np.column_stack([np.ones(n), np.zeros(n), X_cov])

    # Pre-compute positional index mapping from frame.index to array row.
    # WHY: frame.index.get_loc is slow inside a tight loop. Building a dict
    # once is O(N); looking up O(1) per gene per draw.
    idx_to_pos = {idx: i for i, idx in enumerate(frame.index)}

    null_betas: list[float] = []
    n_failed = 0

    for i in range(SIZE_MATCHED_N_DRAWS):
        # Draw gene set matched on decile distribution.
        draw_idx: list[int] = []
        for d_val, count in decile_counts.items():
            pool = decile_pools[d_val]
            if len(pool) == 0:
                continue
            replace = len(pool) < count
            sampled = rng.choice(pool, size=count, replace=replace)
            draw_idx.extend(sampled.tolist())

        if len(draw_idx) == 0:
            n_failed += 1
            continue

        # Build indicator column (col 1) in X_base.
        X_base[:, 1] = 0.0
        for idx in draw_idx:
            X_base[idx_to_pos[idx], 1] = 1.0

        # OLS via numpy lstsq (much faster than sm.OLS for 10k iterations).
        try:
            params, _, _, _ = np.linalg.lstsq(X_base, y, rcond=None)
            beta = float(params[1])  # indicator coefficient
            if np.isfinite(beta):
                null_betas.append(beta)
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

    if not null_betas:
        return {"status": "failed", "reason": "all draws failed"}

    null_arr = np.array(null_betas)
    # Empirical percentile: fraction of null betas <= observed beta.
    emp_pct = float(np.mean(null_arr <= observed_beta))

    return {
        "status": "ok",
        "observed_beta": float(observed_beta),
        "null_mean": float(np.mean(null_arr)),
        "null_sd": float(np.std(null_arr)),
        "null_median": float(np.median(null_arr)),
        "empirical_percentile": emp_pct,
        "robust": bool(emp_pct > 0.95),
        "n_draws": len(null_betas),
        "n_draws_failed": n_failed,
        "seed": seed,
    }


# ===========================================================================
# Reproduction gate: SCZ B3 beta_OLS
# ===========================================================================
def run_reproduction_gate(
    gnomad: pd.DataFrame,
    annot: pd.DataFrame,
    logger: logging.Logger,
) -> dict:
    """Verify SCZ B3 beta_OLS is in [+2.5, +3.5] (iter_058 anchor 3.24).

    WHY: brief_v2 section "Shared Design Elements" mandates this reproduction
    gate. If it fails, all results are UNINTERPRETABLE because the regression
    frame may have drifted.
    """
    # Map B3 symbols to ENSGIDs (same as batch_058 sub_a_robust_battery.py).
    annot_by_name = annot.drop_duplicates(subset="NAME", keep="first").set_index("NAME")
    b3_ensg: dict[str, str] = {}
    for s in B3_GENES:
        ensg = annot_by_name["ENSGID"].get(s)
        if ensg is not None:
            b3_ensg[s] = ensg
    b3_ensg_set = set(b3_ensg.values())
    logger.info("Reproduction gate: B3 mapped %d/%d", len(b3_ensg_set), len(B3_GENES))

    frame = build_sub_a_frame("scz", gnomad, annot, b3_ensg_set)
    ols = fit_ols(frame, "in_set", PRIMARY_COVS)
    beta = float(ols["beta_1"]) if ols.get("status") == "ok" else None

    return {
        "b3_n_mapped": len(b3_ensg_set),
        "b3_n_total": len(B3_GENES),
        "scz_b3_beta_ols": beta,
        "target_lo": REPRO_GATE_LO,
        "target_hi": REPRO_GATE_HI,
        "pass": bool(
            beta is not None
            and REPRO_GATE_LO <= beta <= REPRO_GATE_HI
        ),
    }


# ===========================================================================
# Decision rules (per brief_v2 sections E4/E5/E6)
# ===========================================================================
def classify_e4(per_disorder: dict[str, dict], q_by_d: dict[str, float]) -> str:
    """E4 IEG decision rule (brief_v2 section E4 DECISION RULE).

    Categories:
      IEG_SCZ_ENRICHED: SCZ beta > +0.30 AND q_BH < 0.05 AND DFBETAS < 1.0
      IEG_PAN_PSYCHIATRIC: SCZ beta > +0.30 AND >= 1 other psych enriched
      IEG_WEAK: SCZ beta in [+0.15, +0.30]
      IEG_NOT_ENRICHED: SCZ beta <= +0.15
      IEG_GENERIC: Height/IBD beta > +0.30 while SCZ null
    """
    scz = per_disorder.get("scz", {})
    if scz.get("status") != "ok":
        return "UNCLASSIFIED"

    scz_beta = float(scz["ols"]["beta_1"])
    scz_q = q_by_d.get("scz", 1.0)
    scz_dfbetas_max = float(
        scz["influence"].get("max_abs_dfbetas_b3", float("nan"))
    ) if scz["influence"].get("status") == "ok" else float("nan")

    # Check SCZ enrichment.
    if scz_beta > 0.30 and scz_q < 0.05 and scz_dfbetas_max < DFBETAS_CUTOFF:
        # Check pan-psychiatric.
        other_psych_enriched = []
        for d in ["bip", "mdd", "asd", "adhd"]:
            item = per_disorder.get(d, {})
            if item.get("status") == "ok":
                d_beta = float(item["ols"]["beta_1"])
                d_q = q_by_d.get(d, 1.0)
                if d_beta > 0.20 and d_q < 0.05:
                    other_psych_enriched.append(d)
        if other_psych_enriched:
            return "IEG_PAN_PSYCHIATRIC"
        return "IEG_SCZ_ENRICHED"

    if 0.15 < scz_beta <= 0.30:
        return "IEG_WEAK"

    if scz_beta <= 0.15:
        # Check generic (Height/IBD enriched while SCZ null).
        for nc in ["height", "ibd_delange2017"]:
            item = per_disorder.get(nc, {})
            if item.get("status") == "ok":
                nc_beta = float(item["ols"]["beta_1"])
                if nc_beta > 0.30:
                    return "IEG_GENERIC"
        return "IEG_NOT_ENRICHED"

    # SCZ beta > 0.30 but q >= 0.05 or DFBETAS >= 1.0: treat as WEAK.
    return "IEG_WEAK"


def classify_e5(per_disorder: dict[str, dict], q_by_d: dict[str, float]) -> str:
    """E5 GR targets decision rule (brief_v2 section E5 DECISION RULE).

    Categories:
      GR_NOT_ENRICHED: beta <= +0.15
      GR_WEAK: beta in (+0.15, +0.30]
      GR_ENRICHED: beta > +0.30 (contradicts CRP-null expectation)
      GR_MDD_SPECIFIC: MDD beta > +0.30 while SCZ null
    """
    scz = per_disorder.get("scz", {})
    if scz.get("status") != "ok":
        return "UNCLASSIFIED"

    scz_beta = float(scz["ols"]["beta_1"])

    if scz_beta > 0.30:
        return "GR_ENRICHED"
    if 0.15 < scz_beta <= 0.30:
        return "GR_WEAK"

    # SCZ <= 0.15: check MDD specificity.
    mdd = per_disorder.get("mdd", {})
    if mdd.get("status") == "ok":
        mdd_beta = float(mdd["ols"]["beta_1"])
        if mdd_beta > 0.30:
            return "GR_MDD_SPECIFIC"
    return "GR_NOT_ENRICHED"


def classify_e6(
    per_disorder_primary: dict[str, dict],
    per_disorder_full: dict[str, dict],
    q_by_d_primary: dict[str, float],
) -> str:
    """E6 complement decision rule (brief_v2 section E6 DECISION RULE).

    Uses MHC-excluded (PRIMARY) for the decision.
    Categories:
      COMPLEMENT_NULL_REPLICATED: beta <= +0.15
      COMPLEMENT_ENRICHED: beta > +0.30 (contradicts Kim/Holland prior)
      MHC_DRIVEN: full-set beta > +0.30 AND MHC-excluded beta <= +0.15
      IBD_COMPLEMENT_SIGNAL: IBD beta > +0.30 (expected, immune pathway)
    """
    scz_p = per_disorder_primary.get("scz", {})
    if scz_p.get("status") != "ok":
        return "UNCLASSIFIED"

    scz_beta_primary = float(scz_p["ols"]["beta_1"])

    # Check IBD complement signal.
    ibd_p = per_disorder_primary.get("ibd_delange2017", {})
    ibd_beta = float(ibd_p["ols"]["beta_1"]) if ibd_p.get("status") == "ok" else 0.0

    # Check MHC-driven pattern.
    scz_f = per_disorder_full.get("scz", {})
    scz_beta_full = (
        float(scz_f["ols"]["beta_1"]) if scz_f.get("status") == "ok" else 0.0
    )
    if scz_beta_full > 0.30 and scz_beta_primary <= 0.15:
        return "MHC_DRIVEN"

    if scz_beta_primary > 0.30:
        return "COMPLEMENT_ENRICHED"
    if scz_beta_primary <= 0.15:
        if ibd_beta > 0.30:
            return "IBD_COMPLEMENT_SIGNAL"
        return "COMPLEMENT_NULL_REPLICATED"

    # Intermediate zone.
    return "COMPLEMENT_WEAK"


# ===========================================================================
# Run a full gene-set battery (all 8 disorders)
# ===========================================================================
def run_gene_set_battery(
    set_name: str,
    gene_set_ensg: set[str],
    dusp1_ensgid: str | None,
    gnomad: pd.DataFrame,
    annot: pd.DataFrame,
    logger: logging.Logger,
    disorders: list[str] | None = None,
) -> dict:
    """Run the Sub-A v2.1 battery for one gene set across all disorders.

    Returns a dict with per_disorder results, BH q-values, and LOEUF
    sensitivity for any disorder reaching q < 0.05.
    """
    if disorders is None:
        disorders = DISORDERS

    per_disorder: dict[str, dict] = {}
    for d in disorders:
        per_disorder[d] = run_disorder_battery(
            d, gnomad, annot, gene_set_ensg, dusp1_ensgid, logger,
        )

    # BH-FDR across 8 disorders on OLS one-sided p.
    ok_disorders = [d for d in disorders if per_disorder[d].get("status") == "ok"]
    q_by_d: dict[str, float] = {}
    if len(ok_disorders) == len(disorders):
        pvals = [per_disorder[d]["ols"]["p_one_sided"] for d in disorders]
        qvals = bh_fdr(pvals)
        q_by_d = dict(zip(disorders, qvals))
        for d, q in q_by_d.items():
            per_disorder[d]["q_bh"] = q
    else:
        logger.warning(
            "%s: not all disorders OK (%d/%d). BH-FDR computed only on OK set.",
            set_name, len(ok_disorders), len(disorders),
        )
        if ok_disorders:
            pvals = [per_disorder[d]["ols"]["p_one_sided"] for d in ok_disorders]
            qvals = bh_fdr(pvals)
            q_by_d = dict(zip(ok_disorders, qvals))
            for d, q in q_by_d.items():
                per_disorder[d]["q_bh"] = q

    # LOEUF sensitivity for disorders reaching q < 0.05 (brief_v2).
    loeuf_results: dict[str, dict] = {}
    sig_disorders = [d for d in ok_disorders if q_by_d.get(d, 1.0) < BH_Q]
    for d in sig_disorders:
        logger.info("%s: running LOEUF sensitivity for %s (q=%.4f)",
                    set_name, d, q_by_d[d])
        loeuf_results[d] = run_loeuf_one(
            d, gnomad, annot, gene_set_ensg, logger,
        )

    # Size-matched null for disorders reaching q < 0.05 (brief_v2).
    size_matched: dict[str, dict] = {}
    for d in sig_disorders:
        item = per_disorder[d]
        observed_beta = float(item["ols"]["beta_1"])
        logger.info(
            "%s: running size-matched null for %s (beta=%.3f, q=%.4f)",
            set_name, d, observed_beta, q_by_d[d],
        )
        try:
            frame = build_sub_a_frame(d, gnomad, annot, gene_set_ensg)
            sm_result = size_matched_null(
                frame, observed_beta, len(gene_set_ensg),
                gene_set_ensg, "in_set", logger,
                seed=SEED,
            )
            size_matched[d] = sm_result
        except Exception as exc:
            logger.exception("Size-matched null failed for %s/%s", set_name, d)
            size_matched[d] = {"status": "failed", "reason": str(exc)}

    return {
        "per_disorder": per_disorder,
        "q_by_d": q_by_d,
        "loeuf_sensitivity": loeuf_results,
        "size_matched_null": size_matched,
    }


# ===========================================================================
# Format output for one gene set
# ===========================================================================
def format_per_disorder_output(per_disorder: dict, q_by_d: dict) -> dict:
    """Format per-disorder results for JSON output.

    WHY this formatting: brief_v2 specifies the output schema with per-disorder
    keys containing ols_beta, ols_ci, ols_p, q_bh, huber_beta, rank_sign,
    dfbetas_max, dfbetas_dusp1.
    """
    out: dict[str, dict] = {}
    for d, item in per_disorder.items():
        if item.get("status") != "ok":
            out[d] = {"status": item.get("status", "failed"),
                      "reason": item.get("reason", "unknown")}
            continue
        ols = item["ols"]
        huber = item.get("huber", {})
        tukey = item.get("tukey", {})
        rank_ols = item.get("rank_magma_ols", {})
        infl = item.get("influence", {})

        out[d] = {
            "ols_beta": ols.get("beta_1"),
            "ols_se": ols.get("se_1"),
            "ols_ci": [ols.get("ci_lo"), ols.get("ci_hi")],
            "ols_p_one_sided": ols.get("p_one_sided"),
            "ols_p_two_sided": ols.get("p_two_sided"),
            "q_bh": q_by_d.get(d),
            "ols_r_squared": ols.get("r_squared"),
            "ols_n": ols.get("n"),
            "huber_beta": huber.get("beta_1"),
            "huber_ci": (
                [huber.get("ci_lo"), huber.get("ci_hi")]
                if huber.get("status") == "ok" else None
            ),
            "tukey_beta": tukey.get("beta_1"),
            "tukey_ci": (
                [tukey.get("ci_lo"), tukey.get("ci_hi")]
                if tukey.get("status") == "ok" else None
            ),
            "tukey_converged": tukey.get("converged"),
            "rank_ols_beta": rank_ols.get("beta_1"),
            "rank_sign_concordance": (
                bool(
                    np.sign(ols.get("beta_1", 0))
                    == np.sign(rank_ols.get("beta_1", 0))
                )
                if rank_ols.get("status") == "ok" and ols.get("beta_1") is not None
                else None
            ),
            "dfbetas_max": infl.get("max_abs_dfbetas_b3"),
            "dfbetas_dusp1": item.get("dfbetas_dusp1"),
            "n_gene_universe": item.get("n_gene_universe"),
            "n_set_in_universe": item.get("n_set_in_universe"),
        }
    return out


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="batch_060 E4/E5/E6: environmental-axis gene-set batteries"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: SCZ + ADHD only, skip LOEUF and size-matched null.",
    )
    parser.add_argument(
        "--experiments", type=str, default="e4,e5,e6",
        help="Comma-separated list of experiments to run (default: e4,e5,e6).",
    )
    args = parser.parse_args()

    experiments = [e.strip().lower() for e in args.experiments.split(",")]
    smoke = args.smoke

    logger = setup_logger("batch_060.e4e5e6", LOGS_DIR / "e4e5e6.log")
    t0 = time.time()
    logger.info("=" * 72)
    logger.info("batch_060 E4/E5/E6 environmental-axis gene-set batteries")
    logger.info("experiments=%s smoke=%s seed=%d", experiments, smoke, SEED)
    logger.info("=" * 72)

    # Set global seed for reproducibility.
    np.random.seed(SEED)

    # -----------------------------------------------------------------------
    # Load shared data (once).
    # -----------------------------------------------------------------------
    logger.info("Loading gnomAD constraint metrics...")
    gnomad = load_gnomad_per_brief_v2()
    logger.info("gnomAD: %d genes loaded.", len(gnomad))

    logger.info("Loading gene annotations...")
    annot = load_gene_annot()
    logger.info("gene_annot: %d genes loaded.", len(annot))

    disorders_to_run = DISORDERS if not smoke else ["scz", "adhd"]

    # -----------------------------------------------------------------------
    # Reproduction gate (B3 SCZ beta_OLS).
    # WHY first: if the regression frame has drifted, all results are
    # UNINTERPRETABLE. We check this before running any gene-set battery.
    # -----------------------------------------------------------------------
    logger.info("Running reproduction gate (SCZ B3 beta_OLS)...")
    repro_gate = run_reproduction_gate(gnomad, annot, logger)
    logger.info(
        "Reproduction gate: B3 SCZ beta=%.4f, pass=%s",
        repro_gate["scz_b3_beta_ols"] or float("nan"),
        repro_gate["pass"],
    )
    if not repro_gate["pass"]:
        logger.error(
            "REPRODUCTION GATE FAILED. SCZ B3 beta=%.4f not in [%.1f, %.1f]. "
            "All results UNINTERPRETABLE.",
            repro_gate["scz_b3_beta_ols"] or float("nan"),
            REPRO_GATE_LO, REPRO_GATE_HI,
        )
        # Still proceed to write results, but flag everything as
        # UNINTERPRETABLE. WHY: Cardinal Rule 0 requires reporting failures,
        # not hiding them.

    # -----------------------------------------------------------------------
    # Gene symbol verification for all gene sets.
    # -----------------------------------------------------------------------
    # DUSP1 ENSGID (needed for DFBETAS reporting in E4 and E5).
    dusp1_row = annot[annot["NAME"] == "DUSP1"]
    dusp1_ensgid = str(dusp1_row["ENSGID"].iloc[0]) if len(dusp1_row) > 0 else None
    logger.info("DUSP1 ENSGID: %s", dusp1_ensgid)

    gene_set_info: dict[str, dict] = {}

    if "e4" in experiments:
        ensg, sym_map, unmapped, multi = verify_gene_set(
            IEG_RPRGS, annot, logger, "IEG_rPRGs",
        )
        gene_set_info["ieg"] = {
            "ensg_set": ensg, "sym_to_ensg": sym_map,
            "unmapped": unmapped, "multi_mapped": multi,
            "n_requested": len(IEG_RPRGS), "n_mapped": len(ensg),
        }

    if "e5" in experiments:
        ensg, sym_map, unmapped, multi = verify_gene_set(
            GR_TARGETS, annot, logger, "GR_targets",
        )
        gene_set_info["gr"] = {
            "ensg_set": ensg, "sym_to_ensg": sym_map,
            "unmapped": unmapped, "multi_mapped": multi,
            "n_requested": len(GR_TARGETS), "n_mapped": len(ensg),
        }

    if "e6" in experiments:
        # MHC-excluded PRIMARY set.
        complement_mhc_excl = [s for s in COMPLEMENT_FULL if s not in MHC_EXCLUDE]
        ensg_p, sym_map_p, unmapped_p, multi_p = verify_gene_set(
            complement_mhc_excl, annot, logger, "Complement_MHC_excluded",
            expected_chr=COMPLEMENT_EXPECTED_CHR,
        )
        gene_set_info["complement_primary"] = {
            "ensg_set": ensg_p, "sym_to_ensg": sym_map_p,
            "unmapped": unmapped_p, "multi_mapped": multi_p,
            "n_requested": len(complement_mhc_excl), "n_mapped": len(ensg_p),
        }
        # Full set (SENSITIVITY).
        ensg_f, sym_map_f, unmapped_f, multi_f = verify_gene_set(
            COMPLEMENT_FULL, annot, logger, "Complement_full",
            expected_chr=COMPLEMENT_EXPECTED_CHR,
        )
        gene_set_info["complement_full"] = {
            "ensg_set": ensg_f, "sym_to_ensg": sym_map_f,
            "unmapped": unmapped_f, "multi_mapped": multi_f,
            "n_requested": len(COMPLEMENT_FULL), "n_mapped": len(ensg_f),
        }

    # -----------------------------------------------------------------------
    # Check UNINTERPRETABLE threshold (>3 missing per set).
    # -----------------------------------------------------------------------
    for key, info in gene_set_info.items():
        n_missing = info["n_requested"] - info["n_mapped"]
        if n_missing > MAX_MISSING_GENES:
            logger.error(
                "UNINTERPRETABLE: %s has %d missing genes (threshold=%d). "
                "Unmapped: %s",
                key, n_missing, MAX_MISSING_GENES, info["unmapped"],
            )
            # For complement_full, the threshold is 5 (brief_v2 E6: >5/24).
            # But we use the per-set threshold consistently. The complement
            # MHC-excluded PRIMARY uses >3 as per brief_v2 E4/E5 convention.
            # E6 brief says ">5/24 missing (MHC-excluded)" for UNINTERPRETABLE.
            # We log but do NOT abort -- we produce results flagged as
            # UNINTERPRETABLE so the auditor can see what happened.

    # -----------------------------------------------------------------------
    # E4: IEG rPRGs
    # -----------------------------------------------------------------------
    if "e4" in experiments:
        logger.info("=" * 72)
        logger.info("E4: IEG rPRGs x 8-disorder battery")
        logger.info("=" * 72)

        ieg_info = gene_set_info["ieg"]
        ieg_ensg = ieg_info["ensg_set"]
        n_missing_ieg = ieg_info["n_requested"] - ieg_info["n_mapped"]
        ieg_interpretable = n_missing_ieg <= MAX_MISSING_GENES

        battery = run_gene_set_battery(
            "IEG_rPRGs", ieg_ensg, dusp1_ensgid, gnomad, annot, logger,
            disorders=disorders_to_run,
        )

        classification = (
            classify_e4(battery["per_disorder"], battery["q_by_d"])
            if ieg_interpretable and repro_gate["pass"] and not smoke
            else "UNINTERPRETABLE"
        )

        e4_results = {
            "experiment": "E4",
            "gene_set": "IEG_rPRGs",
            "source": "Tyssowski et al. 2018 Neuron [lit_doi_10.1016_j.neuron.2018.04.001]",
            "n_genes_requested": ieg_info["n_requested"],
            "n_genes_mapped": ieg_info["n_mapped"],
            "n_genes_unmapped": ieg_info["unmapped"],
            "n_genes_multi_mapped": ieg_info["multi_mapped"],
            "symbol_to_ensgid": ieg_info["sym_to_ensg"],
            "reproduction_gate": repro_gate,
            "per_disorder": format_per_disorder_output(
                battery["per_disorder"], battery["q_by_d"],
            ),
            "bh_q_by_disorder": battery["q_by_d"],
            "loeuf_sensitivity": battery["loeuf_sensitivity"],
            "size_matched_null": battery["size_matched_null"],
            "classification": classification,
            "disorders_run": disorders_to_run,
            "seed": SEED,
            "smoke": smoke,
            "wall_s": time.time() - t0,
            "model": ("MAGMA_Z ~ in_set + log10(gene_length) + lof_pLI + "
                       "log10(exp_lof+1) + log10(NSNPS+1)"),
            "brief_contract": {
                "bh_fdr_family_size": len(disorders_to_run),
                "bh_q_threshold": BH_Q,
                "dfbetas_cutoff": DFBETAS_CUTOFF,
                "max_missing_genes": MAX_MISSING_GENES,
                "repro_gate_range": [REPRO_GATE_LO, REPRO_GATE_HI],
            },
        }

        out_path = OUTPUT_DIR / "e4" / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(e4_results, out_path)
        logger.info("E4 wrote %s (classification=%s)", out_path, classification)

    # -----------------------------------------------------------------------
    # E5: GR targets
    # -----------------------------------------------------------------------
    if "e5" in experiments:
        logger.info("=" * 72)
        logger.info("E5: GR targets x 8-disorder battery")
        logger.info("=" * 72)

        gr_info = gene_set_info["gr"]
        gr_ensg = gr_info["ensg_set"]
        n_missing_gr = gr_info["n_requested"] - gr_info["n_mapped"]
        gr_interpretable = n_missing_gr <= MAX_MISSING_GENES

        battery = run_gene_set_battery(
            "GR_targets", gr_ensg, dusp1_ensgid, gnomad, annot, logger,
            disorders=disorders_to_run,
        )

        # DUSP1 leave-one-out check (brief_v2 note: if DUSP1 DFBETAS > 0.5
        # in both E4 and E5, run leave-one-out excluding DUSP1 from GR list).
        dusp1_loo = None
        if dusp1_ensgid is not None:
            # Check if DUSP1 DFBETAS > 0.5 in SCZ for GR.
            scz_item = battery["per_disorder"].get("scz", {})
            dusp1_dfb_gr = scz_item.get("dfbetas_dusp1")
            if dusp1_dfb_gr is not None and abs(dusp1_dfb_gr) > 0.5:
                logger.info(
                    "E5: DUSP1 DFBETAS=%.3f > 0.5 in SCZ. Running leave-one-out.",
                    dusp1_dfb_gr,
                )
                gr_ensg_no_dusp1 = gr_ensg - {dusp1_ensgid}
                loo_result = run_disorder_battery(
                    "scz", gnomad, annot, gr_ensg_no_dusp1,
                    None, logger,  # no DUSP1 to track
                )
                dusp1_loo = {
                    "scz_ols_beta_with_dusp1": float(scz_item["ols"]["beta_1"]),
                    "scz_ols_beta_without_dusp1": (
                        float(loo_result["ols"]["beta_1"])
                        if loo_result.get("status") == "ok" else None
                    ),
                    "n_set_without_dusp1": len(gr_ensg_no_dusp1),
                }

        classification = (
            classify_e5(battery["per_disorder"], battery["q_by_d"])
            if gr_interpretable and repro_gate["pass"] and not smoke
            else "UNINTERPRETABLE"
        )

        e5_results = {
            "experiment": "E5",
            "gene_set": "GR_targets",
            "source": "Reddy et al. 2009 Genome Res [lit_doi_10.1101_gr.085464.108]",
            "n_genes_requested": gr_info["n_requested"],
            "n_genes_mapped": gr_info["n_mapped"],
            "n_genes_unmapped": gr_info["unmapped"],
            "n_genes_multi_mapped": gr_info["multi_mapped"],
            "symbol_to_ensgid": gr_info["sym_to_ensg"],
            "reproduction_gate": repro_gate,
            "per_disorder": format_per_disorder_output(
                battery["per_disorder"], battery["q_by_d"],
            ),
            "bh_q_by_disorder": battery["q_by_d"],
            "loeuf_sensitivity": battery["loeuf_sensitivity"],
            "size_matched_null": battery["size_matched_null"],
            "dusp1_leave_one_out": dusp1_loo,
            "classification": classification,
            "disorders_run": disorders_to_run,
            "seed": SEED,
            "smoke": smoke,
            "wall_s": time.time() - t0,
            "model": ("MAGMA_Z ~ in_set + log10(gene_length) + lof_pLI + "
                       "log10(exp_lof+1) + log10(NSNPS+1)"),
            "brief_contract": {
                "bh_fdr_family_size": len(disorders_to_run),
                "bh_q_threshold": BH_Q,
                "dfbetas_cutoff": DFBETAS_CUTOFF,
                "max_missing_genes": MAX_MISSING_GENES,
                "repro_gate_range": [REPRO_GATE_LO, REPRO_GATE_HI],
            },
        }

        out_path = OUTPUT_DIR / "e5" / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(e5_results, out_path)
        logger.info("E5 wrote %s (classification=%s)", out_path, classification)

    # -----------------------------------------------------------------------
    # E6: Complement cascade
    # -----------------------------------------------------------------------
    if "e6" in experiments:
        logger.info("=" * 72)
        logger.info("E6: Complement cascade x 8-disorder battery")
        logger.info("=" * 72)

        comp_p_info = gene_set_info["complement_primary"]
        comp_f_info = gene_set_info["complement_full"]
        comp_p_ensg = comp_p_info["ensg_set"]
        comp_f_ensg = comp_f_info["ensg_set"]

        # E6 UNINTERPRETABLE threshold: >5/24 missing for MHC-excluded
        # (brief_v2 section E6).
        n_missing_primary = comp_p_info["n_requested"] - comp_p_info["n_mapped"]
        # WHY 5 not 3: brief_v2 E6 specifies ">5/24 missing (MHC-excluded)"
        # as the UNINTERPRETABLE threshold. This is more lenient than E4/E5
        # because complement genes have known aliasing issues.
        comp_interpretable = n_missing_primary <= 5

        # PRIMARY: MHC-excluded (n=24 expected).
        logger.info("E6 PRIMARY: MHC-excluded complement (n_mapped=%d)",
                    comp_p_info["n_mapped"])
        battery_primary = run_gene_set_battery(
            "Complement_MHC_excluded", comp_p_ensg, dusp1_ensgid,
            gnomad, annot, logger,
            disorders=disorders_to_run,
        )

        # SENSITIVITY: full complement (n=28 expected).
        logger.info("E6 SENSITIVITY: full complement (n_mapped=%d)",
                    comp_f_info["n_mapped"])
        battery_full = run_gene_set_battery(
            "Complement_full", comp_f_ensg, dusp1_ensgid,
            gnomad, annot, logger,
            disorders=disorders_to_run,
        )

        classification = (
            classify_e6(
                battery_primary["per_disorder"],
                battery_full["per_disorder"],
                battery_primary["q_by_d"],
            )
            if comp_interpretable and repro_gate["pass"] and not smoke
            else "UNINTERPRETABLE"
        )

        e6_results = {
            "experiment": "E6",
            "gene_set_primary": "Complement_MHC_excluded",
            "gene_set_sensitivity": "Complement_full",
            "source": ("Kim et al. 2021 Nat Neurosci [lit_doi_10.1038_s41593-021-00847-z], "
                        "Holland et al. 2019, Sekar et al. 2016 [lit_doi_10.1038_nature16549]"),
            "mhc_excluded_genes": sorted(MHC_EXCLUDE),
            "primary": {
                "n_genes_requested": comp_p_info["n_requested"],
                "n_genes_mapped": comp_p_info["n_mapped"],
                "n_genes_unmapped": comp_p_info["unmapped"],
                "n_genes_multi_mapped": comp_p_info["multi_mapped"],
                "symbol_to_ensgid": comp_p_info["sym_to_ensg"],
                "per_disorder": format_per_disorder_output(
                    battery_primary["per_disorder"],
                    battery_primary["q_by_d"],
                ),
                "bh_q_by_disorder": battery_primary["q_by_d"],
                "loeuf_sensitivity": battery_primary["loeuf_sensitivity"],
                "size_matched_null": battery_primary["size_matched_null"],
            },
            "sensitivity": {
                "n_genes_requested": comp_f_info["n_requested"],
                "n_genes_mapped": comp_f_info["n_mapped"],
                "n_genes_unmapped": comp_f_info["unmapped"],
                "n_genes_multi_mapped": comp_f_info["multi_mapped"],
                "symbol_to_ensgid": comp_f_info["sym_to_ensg"],
                "per_disorder": format_per_disorder_output(
                    battery_full["per_disorder"],
                    battery_full["q_by_d"],
                ),
                "bh_q_by_disorder": battery_full["q_by_d"],
                "loeuf_sensitivity": battery_full["loeuf_sensitivity"],
                "size_matched_null": battery_full["size_matched_null"],
            },
            "reproduction_gate": repro_gate,
            "classification": classification,
            "disorders_run": disorders_to_run,
            "seed": SEED,
            "smoke": smoke,
            "wall_s": time.time() - t0,
            "model": ("MAGMA_Z ~ in_set + log10(gene_length) + lof_pLI + "
                       "log10(exp_lof+1) + log10(NSNPS+1)"),
            "brief_contract": {
                "bh_fdr_family_size": len(disorders_to_run),
                "bh_q_threshold": BH_Q,
                "dfbetas_cutoff": DFBETAS_CUTOFF,
                "max_missing_genes_primary": 5,
                "repro_gate_range": [REPRO_GATE_LO, REPRO_GATE_HI],
            },
        }

        out_path = OUTPUT_DIR / "e6" / "results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(e6_results, out_path)
        logger.info("E6 wrote %s (classification=%s)", out_path, classification)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    wall = time.time() - t0
    logger.info("=" * 72)
    logger.info("batch_060 E4/E5/E6 complete. Wall time: %.1f s", wall)
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
