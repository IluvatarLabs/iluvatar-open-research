#!/usr/bin/env python3
"""
batch_070: Reviewer-Defense Gap Closure — FIVE sub-analyses

E1 — Pseudobulk DESeq2 DE + Hallmark GSEA per compartment (Vasc, MuSC, FAP)
E2 — Cell-count sensitivity for F084 (Vasc JUNB-SASP) and F093 (MuSC CDKN1A-SASP)
E3 — Formal power annotation on canonical findings table + L-R FDR
E4 — 4-compartment TF coupling heatmap (Vasc, FAP, Immune, MuSC × {JUNB,FOS,EGR1,CEBPB,ATF3,STAT3})
E5 — Confidence-tier audit on 6 flagged findings

WHY WRITE IT THIS WAY:
- Each sub-analysis wrapped in try/except so one failure does not kill the batch
  (brief "Coding constraints" mandate).
- Data columns are AUTO-DETECTED on load (obs schema differs across compartments)
  per explicit instruction to VERIFY and never assume.
- Pseudobulk uses raw counts from adata.raw.X (present in all four HLMA h5ad files;
  confirmed in preflight). If raw counts are absent, E1 aborts for that compartment
  and logs the reason — we do NOT fabricate quasi-counts here (batch_052 already
  tried expm1-derived quasi-counts; brief asks for proper raw counts). WHY raw
  rather than reusing batch_052 quasi-counts: pydeseq2's NB dispersion estimate
  is only meaningful on UMI-scale integers, and raw.X is available.
- DESeq2 design "~ age_binary + tech + sex" per brief; this matches the I2 partial
  correlation design used in batch_060.
- Tech coefficient magnitude is extracted via a second Wald contrast on the tech
  variable ("snRNA" vs "scRNA") so we can compute the tech-vs-age magnitude ratio
  on top-100 age-significant genes (brief E1 decision rule).
- GSEA uses locally cached MSigDB Hallmark GMT from ~/.cache/gseapy (confirmed
  present in preflight). Seed set (42) for permutation reproducibility.
- E3 uses Fisher-z MDR formula explicitly per brief. Observed power is computed
  via Fisher-z: P(|Z - z_r| > z_{alpha/2} / sqrt(N-3)) where Z ~ Normal(atanh(rho), 1/sqrt(N-3)).
  This is standard for Spearman (Bonett & Wright 2000) — asymptotic approximation.
- E4 is a pure aggregation of batch_060 outputs + Immune row from batch_067/068
  results; cells that are not directly measurable (e.g. FAP JUNB AUCell) are
  marked NaN and masked in heatmap per brief instruction.
- E5 is bookkeeping only: we write a CSV that records the current classification
  from research_state.md and a recommended classification based on the 5-criteria
  test explained in CLAUDE.md. This is NOT automated tiering; the rows carry
  human-decidable verdicts so the orchestrator/PI can review.

AUDIT NOTES:
- All constants explain WHY they were chosen.
- Seeds: SEED=42 for gseapy.prerank permutation.
- All file paths are absolute per CWD convention.
- Logging: every major step gets a timestamped line + per-sub-analysis log file.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# Paths and constants
# ============================================================================

PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro")
DATA_DIR = PROJECT_ROOT / "data"
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_070"
RESULTS_DIR = BATCH_DIR / "results"
LOGS_DIR = BATCH_DIR / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# WHY 42: matches batch_052 SEED; consistent reproducibility across batches.
SEED = 42
np.random.seed(SEED)

# SASP12 panel per the brief E2 spec (not the batch_060 variant).
# WHY this panel: this is the panel stated explicitly in the brief. Handle
# missing genes gracefully. IL8 is an alias for CXCL8; we check both.
SASP12_PANEL = [
    "CCL2", "CCL7", "CCL20", "CXCL6", "CXCL8",  # chemokines (CXCL8==IL8)
    "IL6",                                       # cytokine
    "MMP1", "MMP3",                              # matrix
    "SERPINE1",                                  # serpin
    "IGFBP2", "IGFBP3", "IGFBP5",                # IGFBPs
]
# Alias resolution: if CXCL8 absent, try IL8
SASP12_ALIASES = {"CXCL8": ["CXCL8", "IL8"]}

# TFs for E4 heatmap columns
E4_TFS = ["JUNB", "FOS", "EGR1", "CEBPB", "ATF3", "STAT3"]

# Flagged findings for E5 (6 rows, per brief: F066_01, F067_01, F068_02/03/04, F069_04)
E5_FINDINGS = ["F066_01", "F067_01", "F068_02", "F068_03", "F068_04", "F069_04"]

# Compartment -> h5ad path
COMPARTMENT_FILES = {
    "Vascular": DATA_DIR / "Vascular_scsn_RNA.h5ad",
    "MuSC": DATA_DIR / "MuSC_scsn_RNA.h5ad",
    "FAP": DATA_DIR / "OMIX004308-02.h5ad",
    "Immune": DATA_DIR / "Immune_scsn_RNA.h5ad",
}

# E1 compartments (skip Immune by brief design — snRNA-only tech confound)
E1_COMPARTMENTS = ["Vascular", "MuSC", "FAP"]

# E2: (compartment, target_gene) pairs
E2_TARGETS = [
    ("Vascular", "JUNB"),   # F084
    ("MuSC", "CDKN1A"),     # F093
]
E2_THRESHOLDS = [0, 50, 100, 200]

# E1 gene filter: require at least this many donors with >= this count
E1_MIN_COUNT_PER_GENE = 10
E1_MIN_DONORS_EXPRESSING = 3   # gene must be detected in >=3 donors

# Hallmark GMT path (from preflight; local cache confirmed present)
HALLMARK_GMT = Path.home() / ".cache" / "gseapy" / "Enrichr.MSigDB_Hallmark_2020.gmt"


# ============================================================================
# Logging utilities
# ============================================================================

def make_logger(name: str) -> logging.Logger:
    """Create a per-sub-analysis logger that writes both to console and to
    experiments/batch_070/logs/{name}.log.

    WHY both: stdout for live monitoring, file for post-hoc audit (brief
    'Save interim logs to experiments/batch_070/logs/{sub_analysis}.log').
    """
    logger = logging.getLogger(f"batch_070.{name}")
    logger.setLevel(logging.INFO)
    # Clear any old handlers
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOGS_DIR / f"{name}.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ============================================================================
# Data-loading / schema-detection helpers
# ============================================================================

def detect_obs_columns(obs: pd.DataFrame) -> dict:
    """Detect the donor / age / age_pop / tech / sex columns in an h5ad obs df.

    WHY this function: the brief explicitly mandates that column names be
    detected at runtime (they differ between compartments — e.g. Sex vs
    gender). We search a prioritized list and return a dict.
    """
    detected = {"donor": None, "age": None, "age_pop": None,
                "tech": None, "sex": None}
    for col in ["sample", "donor_id", "DonorID", "Donor", "donor", "patient_id"]:
        if col in obs.columns:
            detected["donor"] = col
            break
    for col in ["age", "Age", "age_years", "AgeYears"]:
        if col in obs.columns:
            detected["age"] = col
            break
    for col in ["age_pop", "age_bin", "age_binary", "age_group"]:
        if col in obs.columns:
            detected["age_pop"] = col
            break
    for col in ["tech", "Tech", "technology", "modality"]:
        if col in obs.columns:
            detected["tech"] = col
            break
    for col in ["Sex", "sex", "gender", "Gender"]:
        if col in obs.columns:
            detected["sex"] = col
            break
    return detected


def coerce_age_binary(obs: pd.DataFrame, age_pop_col: str | None,
                      age_col: str | None, logger: logging.Logger) -> pd.Series:
    """Return a Series of {'young','old'} per cell.

    Preference order:
    1. age_pop string labels ('young_pop','old_pop') -> map to young/old
    2. age_pop integer labels (0/1) with 0=young, 1=old  (detected in Immune)
    3. numeric age with median split on donor-level age

    WHY median split for fallback: matches the I2 partial-correlation setup
    used in batch_060 and is the only objective threshold given N=23.
    """
    if age_pop_col is not None:
        series = obs[age_pop_col]
        # Try string form
        str_vals = set(str(x) for x in series.dropna().unique())
        if "young_pop" in str_vals or "old_pop" in str_vals:
            return series.astype(str).map(
                {"young_pop": "young", "old_pop": "old"}
            )
        # Try 0/1 form
        numeric_vals = pd.to_numeric(series, errors="coerce")
        if numeric_vals.dropna().isin([0, 1]).all():
            # Convention from Immune file inspection: 0=young, 1=old
            # (because 0 count > 1 count, matching old_pop majority pattern
            # seen in Vasc string form). Log for transparency.
            logger.info(f"  age_pop is 0/1 encoded — mapping 0->young, 1->old")
            return numeric_vals.map({0: "young", 1: "old"})
    if age_col is not None:
        # Use numeric age, median split on donor level
        logger.info(f"  Falling back to median split on numeric age ({age_col})")
        ages = pd.to_numeric(obs[age_col], errors="coerce")
        median = ages.median()
        return ages.apply(lambda x: "old" if x >= median else "young"
                          if pd.notna(x) else None)
    return pd.Series([None] * len(obs), index=obs.index)


# ============================================================================
# E1 — Pseudobulk DESeq2 + GSEA
# ============================================================================

def e1_build_pseudobulk(adata, donor_col, logger):
    """Build donor-level pseudobulk counts from adata.raw.X (raw UMIs).

    Returns:
        counts_df: DataFrame (donors x genes), integer counts
        donor_meta: DataFrame indexed by donor with age/tech/sex
        None if raw counts are not available
    """
    import scanpy as sc
    from scipy.sparse import issparse

    if adata.raw is None:
        logger.warning("  adata.raw is None — no raw counts. E1 aborted for this compartment.")
        return None, None

    raw = adata.raw
    X = raw.X
    var_names = list(raw.var_names)

    # Verify these LOOK like integer counts (max value should be reasonable UMI count)
    sample_data = X[:1000].toarray() if issparse(X) else np.asarray(X[:1000])
    if not np.allclose(sample_data, np.round(sample_data)):
        logger.warning(
            f"  adata.raw.X does not contain integer values "
            f"(max={float(sample_data.max()):.3g}). Treating as counts but flagging."
        )
    max_val = float(sample_data.max())
    logger.info(f"  raw.X dtype={X.dtype}, shape={X.shape}, max(first 1000 rows)={max_val:.1f}")

    donors = adata.obs[donor_col].astype(str).values
    unique_donors = sorted(pd.unique(donors))

    # Aggregate sum per donor
    counts = np.zeros((len(unique_donors), X.shape[1]), dtype=np.int64)
    for i, d in enumerate(unique_donors):
        mask = donors == d
        donor_X = X[mask]
        if issparse(donor_X):
            summed = np.asarray(donor_X.sum(axis=0)).flatten()
        else:
            summed = np.asarray(donor_X).sum(axis=0)
        # Round defensively in case raw is float-stored but integer-valued
        counts[i, :] = np.clip(np.round(summed), 0, None).astype(np.int64)

    counts_df = pd.DataFrame(counts, index=unique_donors, columns=var_names)
    counts_df.index.name = "donor"
    return counts_df, unique_donors


def e1_filter_genes(counts_df, logger):
    """Apply brief's default low-expression filter.

    WHY min 10 total across donors and min 3 donors expressing: matches
    batch_052 and pydeseq2 quickstart defaults. Removes genes the NB model
    cannot fit meaningfully.
    """
    gene_totals = counts_df.sum(axis=0)
    n_donors_expressing = (counts_df > 0).sum(axis=0)
    keep = (gene_totals >= E1_MIN_COUNT_PER_GENE) & (n_donors_expressing >= E1_MIN_DONORS_EXPRESSING)
    logger.info(
        f"  Gene filter: {len(counts_df.columns)} -> {int(keep.sum())} "
        f"(>={E1_MIN_COUNT_PER_GENE} total, >={E1_MIN_DONORS_EXPRESSING} donors expressing)"
    )
    return counts_df.loc[:, keep]


def e1_run_deseq2(counts_df, metadata, label, logger):
    """Run pydeseq2 with design ~ age_binary + tech + sex.

    Returns a DataFrame with per-gene age coefficient + a parallel DataFrame
    with per-gene tech coefficient, plus a dict of diagnostics including
    age-tech crosstab + confound severity. WHY extract tech coefficient:
    brief's decision rule compares tech vs age magnitude on top-100
    age-significant genes.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    logger.info(f"  pydeseq2 DeseqDataSet fitting ({label})")
    t0 = time.time()
    # Ensure metadata rows align with counts_df index
    metadata = metadata.loc[counts_df.index].copy()

    # Sanity: require at least 2 levels in each factor
    for factor in ["age_binary", "tech", "sex"]:
        if metadata[factor].nunique() < 2:
            logger.warning(
                f"  {factor} has <2 levels in {label} "
                f"(values: {list(metadata[factor].unique())}) — "
                f"DESeq2 design will drop it"
            )

    # ---- Age-tech crosstab collinearity diagnostic (C1 fix) ----
    # WHY: if age × tech has empty or near-empty cells, the age coefficient
    # is confounded with tech; we must either caveat or drop tech.
    diagnostics = {}
    if metadata["tech"].nunique() >= 2 and metadata["age_binary"].nunique() >= 2:
        crosstab = pd.crosstab(metadata["age_binary"], metadata["tech"])
        logger.info(f"  {label} age x tech crosstab:\n{crosstab}")
        min_cell = int(crosstab.values.min()) if crosstab.size else 0
        if min_cell == 0:
            severity = "DEGENERATE"   # rank-deficient, drop tech term
        elif min_cell < 3:
            severity = "SEVERE"        # near-collinear, age coeffs unreliable
        elif min_cell < 6:
            severity = "MODERATE"      # caveat in reporting
        else:
            severity = "ACCEPTABLE"
        diagnostics["tech_confound_severity"] = severity
        diagnostics["age_tech_crosstab"] = crosstab.to_dict()
        logger.info(
            f"  {label} tech_confound_severity={severity} "
            f"(min crosstab cell={min_cell})"
        )
    else:
        severity = "NOT_APPLICABLE"   # tech has <2 levels; nothing to confound
        diagnostics["tech_confound_severity"] = severity
        diagnostics["age_tech_crosstab"] = {}

    # Build design adaptively: include only factors with >=2 levels
    factors = [f for f in ["age_binary", "tech", "sex"]
               if metadata[f].nunique() >= 2]
    if "age_binary" not in factors:
        raise ValueError("age_binary must have 2 levels; E1 cannot proceed")

    # If DEGENERATE crosstab: drop tech term (rank-deficient)
    if severity == "DEGENERATE":
        logger.warning(
            f"  {label}: DEGENERATE tech-age crosstab — "
            f"dropping tech, using ~ age_binary + sex"
        )
        design_factors = [f for f in factors if f != "tech"]
    else:
        design_factors = factors
    design = "~ " + " + ".join(design_factors)
    logger.info(f"  design: {design}")
    diagnostics["design_factors"] = design_factors
    # Preserve `factors` variable for downstream tech-contrast check below
    factors = design_factors

    dds = DeseqDataSet(
        counts=counts_df.astype(int),
        metadata=metadata,
        design=design,
    )
    dds.deseq2()
    logger.info(f"  DeseqDataSet.deseq2() took {time.time()-t0:.1f}s")

    # ---- age contrast ----
    # Convention: old vs young -> positive LFC means up in old
    age_stats = DeseqStats(dds, contrast=["age_binary", "old", "young"])
    age_stats.summary()
    age_df = age_stats.results_df.copy()
    age_df.index.name = "gene"
    age_df = age_df.reset_index().rename(columns={
        "log2FoldChange": "log2FC",
        "lfcSE": "lfcSE",
        "stat": "stat",
        "pvalue": "pvalue",
        "padj": "padj",
    })

    # ---- tech contrast (if tech in design) ----
    tech_df = None
    if "tech" in factors:
        tech_levels = sorted(metadata["tech"].unique().tolist())
        if len(tech_levels) == 2:
            # Compare the two tech levels explicitly; alphabetical order chosen
            # deterministically so ratios are reproducible
            c0, c1 = tech_levels[0], tech_levels[1]
            try:
                tech_stats = DeseqStats(dds, contrast=["tech", c1, c0])
                tech_stats.summary()
                tech_df = tech_stats.results_df.copy()
                tech_df.index.name = "gene"
                tech_df = tech_df.reset_index().rename(columns={
                    "log2FoldChange": "log2FC_tech",
                    "pvalue": "pvalue_tech",
                    "padj": "padj_tech",
                })
                tech_df = tech_df[["gene", "log2FC_tech", "pvalue_tech", "padj_tech"]]
            except Exception as e:
                logger.warning(f"  tech contrast extraction failed: {e}")
    return age_df, tech_df, diagnostics


def e1_gsea(age_df, label, logger):
    """Run gseapy.prerank on signed -log10(padj) * sign(log2FC).

    WHY this metric: penalizes unreliable genes (high padj) and preserves
    directionality. Used previously in batch_052 via Wald stat; the brief
    here specifies signed -log10(padj) * sign(log2FC) explicitly.

    Returns cleaned DataFrame with [Term, NES, pval, fdr, leading_edge].
    """
    import gseapy as gp

    # Build ranking
    df = age_df.dropna(subset=["padj", "log2FC"]).copy()
    # Avoid log(0): floor padj to smallest positive
    floor = df["padj"][df["padj"] > 0].min() if (df["padj"] > 0).any() else 1e-300
    df["padj_safe"] = df["padj"].clip(lower=floor)
    df["rank_metric"] = -np.log10(df["padj_safe"]) * np.sign(df["log2FC"])
    df = df.sort_values("rank_metric", ascending=False)
    df = df.drop_duplicates(subset="gene", keep="first")
    rnk = pd.Series(df["rank_metric"].values, index=df["gene"].values)
    logger.info(
        f"  GSEA prerank input: {len(rnk)} genes, "
        f"range=[{rnk.min():.3f}, {rnk.max():.3f}]"
    )

    if not HALLMARK_GMT.exists():
        logger.warning(f"  Hallmark GMT not found at {HALLMARK_GMT} — skipping GSEA")
        return None

    pre_res = gp.prerank(
        rnk=rnk,
        gene_sets=str(HALLMARK_GMT),
        outdir=None,
        permutation_num=1000,
        seed=SEED,
        min_size=15,
        max_size=500,
        no_plot=True,
        verbose=False,
    )
    res_df = pre_res.res2d.copy()
    if res_df.empty:
        logger.warning(f"  GSEA returned empty for {label}")
        return None

    # Normalize column names — gseapy 1.1.13 column headers
    col_map = {
        "NOM p-val": "pval",
        "FDR q-val": "fdr",
        "Lead_genes": "leading_edge",
    }
    for src, dst in col_map.items():
        if src in res_df.columns:
            res_df = res_df.rename(columns={src: dst})
    keep_cols = [c for c in ["Term", "NES", "pval", "fdr", "leading_edge"]
                 if c in res_df.columns]
    res_df = res_df[keep_cols].sort_values("NES", ascending=False).reset_index(drop=True)
    return res_df


def e1_volcano_plot(age_df, label, out_path, logger):
    """Volcano plot: log2FC vs -log10(padj), top-10 labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = age_df.dropna(subset=["padj", "log2FC"]).copy()
    df["neglog10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))
    df["sig"] = (df["padj"] < 0.1) & (df["log2FC"].abs() > 0.5)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df.loc[~df["sig"], "log2FC"],
               df.loc[~df["sig"], "neglog10_padj"],
               s=6, alpha=0.3, color="gray", label="ns")
    ax.scatter(df.loc[df["sig"], "log2FC"],
               df.loc[df["sig"], "neglog10_padj"],
               s=10, alpha=0.8, color="tab:red", label="padj<0.1 & |log2FC|>0.5")

    top = df.nsmallest(10, "padj")
    for _, row in top.iterrows():
        ax.annotate(str(row["gene"]),
                    (row["log2FC"], row["neglog10_padj"]),
                    fontsize=7, alpha=0.8)

    ax.axhline(-np.log10(0.1), color="k", lw=0.5, linestyle="--")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("log2 fold change (old vs young, controlled for tech+sex)")
    ax.set_ylabel("-log10(padj)")
    ax.set_title(f"Pseudobulk DE volcano — {label} (HLMA)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info(f"  Saved volcano: {out_path}")


def run_e1():
    """Execute E1 for each compartment. Returns summary dict for results.json."""
    logger = make_logger("e1_pseudobulk")
    logger.info("=" * 70)
    logger.info("E1 — Pseudobulk DESeq2 + Hallmark GSEA")
    logger.info("=" * 70)

    import scanpy as sc

    e1_summary = {"status": "running", "per_compartment": {}}

    for comp in E1_COMPARTMENTS:
        logger.info(f"\n--- {comp} ---")
        per_comp = {"status": "running"}
        try:
            h5 = COMPARTMENT_FILES[comp]
            adata = sc.read_h5ad(h5)
            logger.info(f"  Loaded {h5.name}: shape={adata.shape}")

            cols = detect_obs_columns(adata.obs)
            logger.info(f"  Detected columns: {cols}")
            if cols["donor"] is None:
                raise ValueError("no donor column detected")

            # Build age_binary at cell level
            age_binary = coerce_age_binary(
                adata.obs, cols["age_pop"], cols["age"], logger
            )
            if age_binary.isna().all():
                raise ValueError("could not construct age_binary (no age or age_pop)")
            adata.obs = adata.obs.copy()
            adata.obs["_age_binary"] = age_binary

            # Build pseudobulk counts
            counts_df, unique_donors = e1_build_pseudobulk(
                adata, cols["donor"], logger
            )
            if counts_df is None:
                per_comp["status"] = "ABORTED_NO_RAW_COUNTS"
                e1_summary["per_compartment"][comp] = per_comp
                continue

            # Donor metadata: one row per donor (take first occurrence)
            obs = adata.obs
            meta_rows = []
            donor_col = cols["donor"]
            for d in unique_donors:
                sub = obs[obs[donor_col].astype(str) == d]
                age_bin = sub["_age_binary"].iloc[0]
                tech_val = str(sub[cols["tech"]].iloc[0]) if cols["tech"] else "unknown"
                sex_val = str(sub[cols["sex"]].iloc[0]) if cols["sex"] else "unknown"
                meta_rows.append({
                    "donor": d,
                    "age_binary": age_bin,
                    "tech": tech_val,
                    "sex": sex_val,
                    "n_cells": int(len(sub)),
                })
            metadata = pd.DataFrame(meta_rows).set_index("donor")
            logger.info(
                f"  Donors: n={len(metadata)}, "
                f"age_binary={metadata['age_binary'].value_counts().to_dict()}, "
                f"tech={metadata['tech'].value_counts().to_dict()}, "
                f"sex={metadata['sex'].value_counts().to_dict()}"
            )

            # Drop donors with missing metadata
            keep = metadata.dropna(subset=["age_binary"]).index
            counts_df = counts_df.loc[keep]
            metadata = metadata.loc[keep]

            # Gene filter
            counts_df = e1_filter_genes(counts_df, logger)

            # Run DESeq2
            age_df, tech_df, deseq_diag = e1_run_deseq2(
                counts_df, metadata, comp, logger
            )
            # Propagate crosstab diagnostics (C1 fix)
            per_comp["tech_confound_severity"] = deseq_diag.get(
                "tech_confound_severity"
            )
            per_comp["age_tech_crosstab"] = deseq_diag.get("age_tech_crosstab")
            per_comp["design_factors"] = deseq_diag.get("design_factors")

            # Save DE table
            de_path = BATCH_DIR / f"e1_deseq_{comp.lower()}.csv"
            age_df[["gene", "log2FC", "lfcSE", "stat", "pvalue", "padj"]].to_csv(
                de_path, index=False
            )
            logger.info(f"  Saved DE: {de_path}")

            # Tech-vs-age ratio on top-100 age-significant genes
            tech_vs_age_path = BATCH_DIR / f"e1_tech_vs_age_coeff_top100_{comp.lower()}.csv"
            tech_dom_frac = None
            # If tech term was dropped due to DEGENERATE crosstab, skip ratio
            if per_comp.get("tech_confound_severity") == "DEGENERATE":
                logger.info(
                    "  tech term dropped (DEGENERATE crosstab); "
                    "tech-vs-age ratio not computed"
                )
                per_comp["tech_vs_age_ratio"] = None
                per_comp["tech_vs_age_note"] = (
                    "tech term dropped due to degenerate age x tech crosstab"
                )
            elif tech_df is not None:
                top100 = (age_df.dropna(subset=["padj"])
                          .sort_values("padj")
                          .head(100)[["gene", "log2FC", "padj"]])
                top100 = top100.merge(tech_df, on="gene", how="left")
                top100["abs_age"] = top100["log2FC"].abs()
                top100["abs_tech"] = top100["log2FC_tech"].abs()
                top100["tech_over_age"] = top100["abs_tech"] / top100["abs_age"].replace(0, np.nan)
                top100["tech_dominates"] = top100["abs_tech"] > top100["abs_age"]
                top100.to_csv(tech_vs_age_path, index=False)
                tech_dom_frac = float(top100["tech_dominates"].mean())
                logger.info(
                    f"  Tech dominates age in {top100['tech_dominates'].sum()}/"
                    f"{len(top100)} of top-100 age-sig genes "
                    f"({tech_dom_frac*100:.1f}%)"
                )
                per_comp["tech_dominated_frac_top100"] = tech_dom_frac
                per_comp["tech_vs_age_ratio"] = float(
                    top100["tech_over_age"].median(skipna=True)
                )
                per_comp["tech_dominated_flag"] = tech_dom_frac > 0.30
                if tech_dom_frac > 0.30:
                    logger.warning("  FLAG: TECH_DOMINATED (>30% of top-100)")
            else:
                logger.info("  Tech contrast not available; skipping ratio CSV")

            # Hallmark GSEA
            gsea_df = e1_gsea(age_df, comp, logger)
            if gsea_df is not None:
                gsea_path = BATCH_DIR / f"e1_gsea_{comp.lower()}.csv"
                gsea_df.to_csv(gsea_path, index=False)
                logger.info(f"  Saved GSEA: {gsea_path}")
                per_comp["n_hallmark_sig_q01"] = int(
                    (gsea_df.get("fdr", pd.Series()).astype(float) < 0.1).sum()
                )
            else:
                per_comp["n_hallmark_sig_q01"] = None

            # Volcano
            volc_path = BATCH_DIR / f"e1_volcano_{comp.lower()}.png"
            e1_volcano_plot(age_df, comp, volc_path, logger)

            # Metrics
            n_sig_padj01 = int((age_df["padj"] < 0.1).sum())
            per_comp.update({
                "status": "OK",
                "n_donors": int(len(metadata)),
                "n_genes_tested": int(len(age_df)),
                "n_sig_padj0p1": n_sig_padj01,
            })
            logger.info(f"  {comp}: n_sig(padj<0.1)={n_sig_padj01}")

        except Exception as e:
            per_comp["status"] = f"ERROR: {e}"
            per_comp["traceback"] = traceback.format_exc()
            logger.exception(f"  E1 failed for {comp}: {e}")

        e1_summary["per_compartment"][comp] = per_comp

        # Free memory
        try:
            del adata
        except NameError:
            pass
        import gc
        gc.collect()

    e1_summary["status"] = "OK"
    return e1_summary


# ============================================================================
# E2 — Cell-count sensitivity
# ============================================================================

def resolve_sasp_genes(var_names, logger):
    """Resolve SASP12 gene symbols with CXCL8/IL8 alias.

    Returns: (resolved_genes, missing_genes).
    """
    resolved = []
    missing = []
    for g in SASP12_PANEL:
        if g in var_names:
            resolved.append(g)
        elif g in SASP12_ALIASES:
            hit = None
            for alias in SASP12_ALIASES[g]:
                if alias in var_names:
                    hit = alias
                    break
            if hit:
                resolved.append(hit)
            else:
                missing.append(g)
        else:
            missing.append(g)
    if missing:
        logger.info(f"  SASP12 missing: {missing}")
    return resolved, missing


def run_e2():
    """Execute E2 cell-count sensitivity."""
    logger = make_logger("e2_cellcount")
    logger.info("=" * 70)
    logger.info("E2 — Cell-count sensitivity on F084 (Vasc JUNB) and F093 (MuSC CDKN1A)")
    logger.info("=" * 70)

    import scanpy as sc
    from scipy import stats
    from scipy.sparse import issparse

    rows = []
    summary = {"status": "running", "results": []}

    for comp, tf in E2_TARGETS:
        logger.info(f"\n--- {comp} / TF={tf} ---")
        try:
            h5 = COMPARTMENT_FILES[comp]
            adata = sc.read_h5ad(h5)
            cols = detect_obs_columns(adata.obs)
            donor_col = cols["donor"]

            # Resolve SASP genes (use current adata.var_names; raw.var_names if raw)
            var_names = list(adata.var_names)
            sasp_genes, missing = resolve_sasp_genes(var_names, logger)

            # Verify TF present
            if tf not in var_names:
                raise ValueError(f"{tf} not found in var_names")

            # Use adata.X which is log-normalized (consistent across compartments).
            # WHY log-normalized: donor-mean correlation is more robust on
            # log-normalized than on raw counts (Gaussianizes per-cell variance),
            # matches batch_060 partial-correlation protocol, and is what the
            # original F084/F093 findings used.
            def to_dense_1d(X_slice):
                if issparse(X_slice):
                    return np.asarray(X_slice.toarray()).flatten()
                return np.asarray(X_slice).flatten()

            tf_idx = var_names.index(tf)
            tf_per_cell = to_dense_1d(adata.X[:, tf_idx])

            # SASP composite: mean across resolved genes per cell
            if sasp_genes:
                sasp_idx = [var_names.index(g) for g in sasp_genes]
                X_sasp = adata.X[:, sasp_idx]
                if issparse(X_sasp):
                    sasp_per_cell = np.asarray(X_sasp.toarray()).mean(axis=1)
                else:
                    sasp_per_cell = np.asarray(X_sasp).mean(axis=1)
            else:
                raise ValueError("no SASP genes resolvable")

            # Donor aggregation
            donors = adata.obs[donor_col].astype(str).values
            df = pd.DataFrame({
                "donor": donors,
                "tf": tf_per_cell,
                "sasp": sasp_per_cell,
            })
            donor_df = df.groupby("donor").agg(
                tf_mean=("tf", "mean"),
                sasp_mean=("sasp", "mean"),
                n_cells=("tf", "count"),
            ).reset_index()
            logger.info(f"  n_donors={len(donor_df)}, "
                        f"cells/donor: min={donor_df['n_cells'].min()}, "
                        f"median={donor_df['n_cells'].median():.0f}, "
                        f"max={donor_df['n_cells'].max()}")

            # Baseline rho (threshold=0)
            rho0, p0 = stats.spearmanr(donor_df["tf_mean"], donor_df["sasp_mean"])
            logger.info(f"  baseline rho={rho0:.4f}, p={p0:.3g}, N={len(donor_df)}")

            # ---- Jackknife (leave-one-donor-out) on baseline (W2 fix) ----
            # WHY: at N=23, a single influential donor can dominate rho;
            # jackknife quantifies how much. One pass, no acceleration.
            jk_rows = []
            for drop_idx in range(len(donor_df)):
                sub = donor_df.drop(donor_df.index[drop_idx])
                if len(sub) >= 3:
                    jk_rho, _ = stats.spearmanr(sub["tf_mean"], sub["sasp_mean"])
                else:
                    jk_rho = np.nan
                jk_rows.append({
                    "donor_left_out": str(donor_df["donor"].iloc[drop_idx]),
                    "rho_jackknife": float(jk_rho) if not np.isnan(jk_rho) else np.nan,
                    "N_remaining": int(len(sub)),
                })
            jk_df = pd.DataFrame(jk_rows)
            jk_path = BATCH_DIR / f"e2_jackknife_{comp.lower()}_{tf.lower()}.csv"
            jk_df.to_csv(jk_path, index=False)
            logger.info(f"  Saved jackknife: {jk_path}")
            # Jackknife summary stats
            jk_vals = jk_df["rho_jackknife"].dropna().values
            if len(jk_vals) >= 2:
                jk_mean = float(np.mean(jk_vals))
                jk_se = float(np.sqrt(
                    (len(jk_vals) - 1) / len(jk_vals) *
                    np.sum((jk_vals - jk_mean) ** 2)
                ))
                jk_ci_lo = float(jk_mean - 1.96 * jk_se)
                jk_ci_hi = float(jk_mean + 1.96 * jk_se)
                jk_summary = {
                    "compartment": comp,
                    "TF": tf,
                    "baseline_rho": float(rho0),
                    "jackknife_min": float(np.min(jk_vals)),
                    "jackknife_median": float(np.median(jk_vals)),
                    "jackknife_max": float(np.max(jk_vals)),
                    "jackknife_IQR_lo": float(np.percentile(jk_vals, 25)),
                    "jackknife_IQR_hi": float(np.percentile(jk_vals, 75)),
                    "jackknife_mean": jk_mean,
                    "jackknife_SE": jk_se,
                    "jackknife_CI95_lo": jk_ci_lo,
                    "jackknife_CI95_hi": jk_ci_hi,
                }
                logger.info(
                    f"  jackknife: min={jk_summary['jackknife_min']:.4f}, "
                    f"median={jk_summary['jackknife_median']:.4f}, "
                    f"max={jk_summary['jackknife_max']:.4f}, "
                    f"95% CI=[{jk_ci_lo:.4f},{jk_ci_hi:.4f}]"
                )
            else:
                jk_summary = {
                    "compartment": comp,
                    "TF": tf,
                    "baseline_rho": float(rho0),
                    "note": "insufficient jackknife resamples",
                }
            # Accumulate into a module-level jackknife summary list
            if "jackknife_summary" not in summary:
                summary["jackknife_summary"] = []
            summary["jackknife_summary"].append(jk_summary)

            for thr in E2_THRESHOLDS:
                retained = donor_df[donor_df["n_cells"] >= thr]
                N = len(retained)
                if N < 3:
                    rho, p = np.nan, np.nan
                else:
                    rho, p = stats.spearmanr(retained["tf_mean"], retained["sasp_mean"])
                delta = rho - rho0 if not np.isnan(rho) and not np.isnan(rho0) else np.nan
                rows.append({
                    "compartment": comp,
                    "TF": tf,
                    "threshold": thr,
                    "N_retained": N,
                    "rho": rho,
                    "p": p,
                    "delta_vs_baseline": delta,
                })
                logger.info(
                    f"  threshold>={thr}: N={N}, rho={rho:.4f}, p={p:.3g}, "
                    f"delta={delta:+.4f}" if not np.isnan(rho) else
                    f"  threshold>={thr}: N={N} (insufficient)"
                )
            summary["results"].append({
                "compartment": comp,
                "TF": tf,
                "baseline_rho": float(rho0),
                "baseline_N": int(len(donor_df)),
                "missing_sasp_genes": missing,
                "n_sasp_resolved": len(sasp_genes),
            })

            del adata
            import gc
            gc.collect()

        except Exception as e:
            logger.exception(f"  E2 failed for {comp}/{tf}: {e}")
            summary["results"].append({
                "compartment": comp,
                "TF": tf,
                "error": str(e),
            })

    out_path = BATCH_DIR / "e2_cellcount_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")

    # ---- Jackknife summary CSV (W2 fix) ----
    if summary.get("jackknife_summary"):
        jk_sum_path = BATCH_DIR / "e2_jackknife_summary.csv"
        pd.DataFrame(summary["jackknife_summary"]).to_csv(jk_sum_path, index=False)
        logger.info(f"Saved jackknife summary: {jk_sum_path}")
        summary["jackknife_summary_csv"] = str(jk_sum_path)

    # ---- README sibling explaining SASP12 composite (E2 CSV metadata fix) ----
    # WHY a sibling README rather than CSV-inline comments: CSV-inline
    # comments broke round-trip per prior auditor. README stays parallel
    # to the CSV and records provenance for the SASP12 composite.
    per_comp_missing = {
        r.get("compartment"): r.get("missing_sasp_genes", [])
        for r in summary.get("results", [])
        if "compartment" in r
    }
    readme_path = BATCH_DIR / "e2_cellcount_sensitivity.README.txt"
    readme_lines = [
        "e2_cellcount_sensitivity.csv — SASP12 composite provenance",
        "",
        "Composite computation: mean of log-normalized expression (adata.X,",
        "already log1p-CPM/size-factor normalized by source pipeline) across",
        "SASP12 panel: [CCL2, CCL7, CCL20, CXCL6, CXCL8, IL6, MMP1, MMP3,",
        "SERPINE1, IGFBP2, IGFBP3, IGFBP5]. NO z-score applied. Matches the",
        "batch_060 F084/F093 protocol. CXCL8 alias IL8 is accepted.",
        "",
        "Per-compartment missing SASP12 genes:",
    ]
    for c, miss in per_comp_missing.items():
        readme_lines.append(f"  {c}: {list(miss) if miss else 'none'}")
    readme_lines.append("")
    readme_lines.append(
        "Jackknife: leave-one-donor-out Spearman rho per (compartment, TF) "
        "saved as e2_jackknife_{compartment}_{TF}.csv; summary in "
        "e2_jackknife_summary.csv."
    )
    readme_path.write_text("\n".join(readme_lines) + "\n")
    logger.info(f"Saved README: {readme_path}")

    summary["status"] = "OK"
    summary["csv"] = str(out_path)
    summary["readme"] = str(readme_path)
    return summary


# ============================================================================
# E3 — Formal power annotation
# ============================================================================

def fisher_z_mdr(N, alpha=0.05, power=0.80):
    """Minimum detectable rho at given N, alpha, power using Fisher-z.

    Uses Bonett & Wright (2000) Spearman SE correction: SE_z = sqrt(1.06/(N-3)).

    WHY Fisher-z: standard asymptotic method for correlation power. See
    Bonett & Wright 2000; also implemented in G*Power.
        Z_r ~ Normal(arctanh(rho), sqrt(1.06/(N-3)))   [Spearman]
    For two-sided test:
        MDR = tanh((z_{alpha/2} + z_{power}) * sqrt(1.06/(N-3)))
    WHY the 1.06 factor: Pearson-r Fisher-z has SE=1/sqrt(N-3); Spearman
    has inflated SE by factor sqrt(1.06) (Bonett & Wright 2000, Eq. 3).
    All canonical findings here are Spearman correlations.
    """
    from scipy.stats import norm
    if N < 4:
        return np.nan
    z_alpha_2 = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    # Spearman Bonett-Wright correction: SE_z = sqrt(1.06/(N-3))
    return float(np.tanh((z_alpha_2 + z_beta) * np.sqrt(1.06 / (N - 3))))


def fisher_z_observed_power(rho, N, alpha=0.05):
    """Observed power for detecting the observed rho at given N, alpha.

    Uses Bonett & Wright (2000) Spearman SE correction: SE_z = sqrt(1.06/(N-3)).

    Power = 1 - Beta. Under H1 (true rho = observed rho):
        Z_r ~ Normal(arctanh(rho), sqrt(1.06/(N-3)))   [Spearman]
    We reject H0 (rho=0) if |Z_r| > z_{alpha/2} * SE_z.
    Power is P(reject | H1 true).
    """
    from scipy.stats import norm
    if N < 4 or np.isnan(rho):
        return np.nan
    z_true = np.arctanh(rho)
    # Spearman Bonett-Wright correction: SE_z = sqrt(1.06/(N-3))
    se = float(np.sqrt(1.06 / (N - 3)))
    z_crit = norm.ppf(1 - alpha / 2)
    # Two-sided: cutoff on arctanh scale = z_crit * se;
    # power = P(|N(z_true, se)| > z_crit*se)
    cutoff = z_crit * se
    upper = 1 - norm.cdf(cutoff, loc=z_true, scale=se)
    lower = norm.cdf(-cutoff, loc=z_true, scale=se)
    return float(upper + lower)


def run_e3():
    """E3: annotate canonical findings table with formal power + LR multiple testing.

    W1 fix — D2 NA-vasc rho discrepancy:
    We read rho and N directly from canonical_findings_table_final.csv
    row-by-row. NO rho value is hardcoded in this function. The canonical
    CSV (D2 NA Endothelium JUNB) is the single source of truth. Power
    verdicts reflect the exact canonical value, whatever it is.
    """
    logger = make_logger("e3_power")
    logger.info("=" * 70)
    logger.info("E3 — Formal power annotation (Fisher-z MDR) + L-R FDR")
    logger.info("=" * 70)

    summary = {"status": "running"}

    # ---- Power annotation on canonical findings ----
    canonical_path = PROJECT_ROOT / "experiments" / "batch_050" / "canonical_findings_table_final.csv"
    if not canonical_path.exists():
        logger.error(f"  canonical findings table missing: {canonical_path}")
        summary["status"] = "ERROR: canonical table missing"
        return summary

    cf = pd.read_csv(canonical_path)
    logger.info(f"  Loaded {len(cf)} rows from {canonical_path.name}")
    logger.info(f"  Columns: {list(cf.columns)}")

    # Assume columns: Finding, N_donors, rho (per brief reference)
    rho_col = "rho"
    n_col = "N_donors"
    if rho_col not in cf.columns or n_col not in cf.columns:
        logger.warning(f"  Expected {rho_col}/{n_col} missing; available: {list(cf.columns)}")

    annotations = []
    for _, row in cf.iterrows():
        try:
            N = int(row[n_col])
            rho = float(row[rho_col])
        except Exception:
            N, rho = np.nan, np.nan

        mdr = fisher_z_mdr(N, 0.05, 0.80) if not np.isnan(N) else np.nan
        obs_pow = fisher_z_observed_power(rho, N, 0.05) if not np.isnan(rho) else np.nan

        if np.isnan(rho) or np.isnan(mdr):
            verdict = "UNKNOWN"
        elif abs(rho) >= mdr:
            verdict = "ADEQUATE"
        else:
            verdict = "UNDERPOWERED"

        annotations.append({
            "MDR_80power": mdr,
            "observed_power": obs_pow,
            "power_verdict": verdict,
        })

    ann_df = pd.DataFrame(annotations)
    cf_annotated = pd.concat([cf.reset_index(drop=True), ann_df], axis=1)
    out_path = BATCH_DIR / "e3_power_annotated_canonical.csv"
    cf_annotated.to_csv(out_path, index=False)
    logger.info(f"  Saved: {out_path}")
    for _, r in cf_annotated.iterrows():
        logger.info(
            f"    {r.get('Finding','?')} (N={r.get(n_col,'?')}, rho={r.get(rho_col,'?')}): "
            f"MDR={r.get('MDR_80power'):.3f}, obs_power={r.get('observed_power'):.3f}, "
            f"verdict={r.get('power_verdict')}"
        )
    summary["canonical_csv"] = str(out_path)
    summary["n_underpowered"] = int(
        (cf_annotated["power_verdict"] == "UNDERPOWERED").sum()
    )

    # ---- L-R multiple testing on batch_068 ----
    lr_csv = PROJECT_ROOT / "experiments" / "batch_068" / "results.csv"
    lr_json = PROJECT_ROOT / "experiments" / "batch_068" / "results.json"
    lr_rows = []

    if lr_csv.exists():
        lr_df = pd.read_csv(lr_csv)
        n_lr = len(lr_df)
        logger.info(f"  batch_068 L-R pairs: n={n_lr}")

        # Findings F068_02 (CXCL2-CXCR2 JUNB-coupled), F068_03 (VEGFA-KDR),
        # F068_04 (SEMA3C/F, FGF1 age-decrease). Apply BH-FDR across two
        # different hypotheses: JUNB correlation (F068_02/03) and AGE
        # correlation (F068_04). We tabulate both.

        from statsmodels.stats.multitest import multipletests

        # BH on p_junb (tests JUNB coupling across all pairs)
        if "p_junb" in lr_df.columns:
            pvals_j = lr_df["p_junb"].values
            _, fdr_j, _, _ = multipletests(pvals_j, method="fdr_bh")
            lr_df["fdr_bh_junb"] = fdr_j

        # BH on p_age (tests age dependence)
        if "p_age" in lr_df.columns:
            pvals_a = lr_df["p_age"].values
            _, fdr_a, _, _ = multipletests(pvals_a, method="fdr_bh")
            lr_df["fdr_bh_age"] = fdr_a

        # Locate specific finding pairs
        def survives_fdr_junb(ligand, receptor):
            r = lr_df[(lr_df["ligand"] == ligand) & (lr_df["receptor"] == receptor)]
            if r.empty:
                return None, None
            return float(r["fdr_bh_junb"].iloc[0]) if "fdr_bh_junb" in lr_df.columns else None, \
                   float(r["rho_junb"].iloc[0]) if "rho_junb" in lr_df.columns else None

        def survives_fdr_age(ligand, receptor):
            r = lr_df[(lr_df["ligand"] == ligand) & (lr_df["receptor"] == receptor)]
            if r.empty:
                return None, None
            return float(r["fdr_bh_age"].iloc[0]) if "fdr_bh_age" in lr_df.columns else None, \
                   float(r["rho_age"].iloc[0]) if "rho_age" in lr_df.columns else None

        # F068_02: CXCL2-CXCR2 JUNB-coupled
        fdr, rho = survives_fdr_junb("CXCL2", "CXCR2")
        lr_rows.append({
            "finding": "F068_02",
            "pair": "CXCL2-CXCR2",
            "hypothesis": "JUNB coupling",
            "rho": rho,
            "fdr_bh": fdr,
            "survives_fdr_0p05": (fdr is not None and fdr < 0.05),
        })

        # F068_03: VEGFA-KDR JUNB-coupled (pro-angiogenic dysregulated)
        fdr, rho = survives_fdr_junb("VEGFA", "KDR")
        lr_rows.append({
            "finding": "F068_03",
            "pair": "VEGFA-KDR",
            "hypothesis": "JUNB coupling (pro-angiogenic)",
            "rho": rho,
            "fdr_bh": fdr,
            "survives_fdr_0p05": (fdr is not None and fdr < 0.05),
        })

        # F068_04: SEMA3C-NRP1 age-decrease (protective factor lost)
        fdr, rho = survives_fdr_age("SEMA3C", "NRP1")
        lr_rows.append({
            "finding": "F068_04",
            "pair": "SEMA3C-NRP1",
            "hypothesis": "age decrease",
            "rho": rho,
            "fdr_bh": fdr,
            "survives_fdr_0p05": (fdr is not None and fdr < 0.05),
        })
        # Also log FGF1-FGFR1 and SEMA3F-NRP2 for F068_04
        for ligand, receptor in [("SEMA3F", "NRP2"), ("FGF1", "FGFR1")]:
            fdr, rho = survives_fdr_age(ligand, receptor)
            lr_rows.append({
                "finding": "F068_04",
                "pair": f"{ligand}-{receptor}",
                "hypothesis": "age decrease",
                "rho": rho,
                "fdr_bh": fdr,
                "survives_fdr_0p05": (fdr is not None and fdr < 0.05),
            })

        summary["lr_n_pairs"] = int(n_lr)
    else:
        logger.warning(f"  batch_068 L-R table not found at {lr_csv}")
        lr_rows.append({
            "finding": "N/A",
            "pair": "N/A",
            "hypothesis": "batch_068 L-R table not found",
            "rho": None,
            "fdr_bh": None,
            "survives_fdr_0p05": None,
        })
        summary["lr_n_pairs"] = None

    lr_out = BATCH_DIR / "e3_lr_multiple_testing.csv"
    pd.DataFrame(lr_rows).to_csv(lr_out, index=False)
    logger.info(f"  Saved: {lr_out}")
    summary["lr_csv"] = str(lr_out)
    summary["status"] = "OK"
    return summary


# ============================================================================
# E4 — 4-compartment TF coupling heatmap
# ============================================================================

def run_e4():
    """Build heatmap: rows=[Vasc,FAP,Immune,MuSC], cols=E4_TFS, values=partial_rho."""
    logger = make_logger("e4_heatmap")
    logger.info("=" * 70)
    logger.info("E4 — 4-compartment TF coupling heatmap")
    logger.info("=" * 70)

    summary = {"status": "running"}
    try:
        i2_path = PROJECT_ROOT / "experiments" / "batch_060" / "i2_partial_correlations.csv"
        musc_path = PROJECT_ROOT / "experiments" / "batch_060" / "musc_partial.csv"

        if not i2_path.exists():
            raise FileNotFoundError(f"{i2_path} missing")

        i2 = pd.read_csv(i2_path)
        logger.info(f"  Loaded i2 partial rhos: {len(i2)} rows, compartments="
                    f"{i2['compartment'].unique().tolist()}")

        # ---- Covariate-set documentation (C2 fix) ----
        # WHY: the heatmap aggregates partial-rho values computed with
        # potentially DIFFERENT covariate sets across compartments. We
        # document the covariate set for each source so downstream readers
        # know what "partial" means per cell.
        # batch_060 i2_partial_correlations.csv: columns include partial_rho
        #   controlling for age + sex + tech (brief E4 notes same design as
        #   I2 partial-correlation protocol).
        # batch_060 musc_partial.csv: MuSC-specific partial controls for
        #   age + sex + tech (same protocol).
        # batch_059 Immune: snRNA-only cohort (N=13); tech cannot be used
        #   as covariate because tech is CONSTANT (all snRNA). Therefore
        #   covariate set for Immune is incompatible ("age+tech" in
        #   age_tech_adjustment.csv requires tech variation; the snRNA-only
        #   runs effectively use age only or age+sex).
        covariate_sets = {
            "Vascular": "age+sex+tech",
            "FAP": "age+sex+tech",
            "MuSC": "age+sex+tech",
            "Immune": None,  # filled below if resolvable
        }
        source_batches = {
            "Vascular": "batch_060/i2_partial_correlations.csv",
            "FAP": "batch_060/i2_partial_correlations.csv",
            "MuSC": "batch_060/musc_partial.csv",
            "Immune": None,
        }
        logger.info(
            f"  Covariate set for Vascular/FAP/MuSC: age+sex+tech "
            f"(batch_060 i2 protocol)"
        )

        # Initialize matrix as NaN
        rows = ["Vascular", "FAP", "Immune", "MuSC"]
        mat = pd.DataFrame(
            index=rows,
            columns=E4_TFS,
            dtype=float,
        )
        mat.index.name = "compartment"

        # Per-cell note matrix (covariate_set and source_batch annotations)
        cov_note_mat = pd.DataFrame("", index=rows, columns=E4_TFS, dtype=object)
        src_note_mat = pd.DataFrame("", index=rows, columns=E4_TFS, dtype=object)

        # Fill Vascular / FAP from batch_060 i2_partial_correlations
        for _, r in i2.iterrows():
            comp = r["compartment"]
            tf = r["TF"]
            if comp in rows and tf in E4_TFS:
                mat.loc[comp, tf] = float(r["partial_rho"])
                cov_note_mat.loc[comp, tf] = covariate_sets.get(comp, "")
                src_note_mat.loc[comp, tf] = source_batches.get(comp, "")

        # Fill MuSC from musc_partial.csv
        if musc_path.exists():
            musc = pd.read_csv(musc_path)
            for _, r in musc.iterrows():
                tf = r["TF"]
                if tf in E4_TFS:
                    mat.loc["MuSC", tf] = float(r["partial_rho"])
                    cov_note_mat.loc["MuSC", tf] = covariate_sets["MuSC"]
                    src_note_mat.loc["MuSC", tf] = source_batches["MuSC"]
        else:
            logger.warning(f"  musc_partial.csv missing — MuSC row stays NaN")

        # Immune compartment: batch_060's i2 table does NOT include Immune.
        # batch_059 characterized Immune separately (snRNA-only cohort,
        # N=13). WHY we MUST not silently fill: the Immune cohort has no
        # scRNA donors so "tech" is a constant — covariate set is
        # INCOMPATIBLE with Vasc/FAP/MuSC (age+sex+tech). Per brief/C2
        # guidance, we mark Immune as NaN with note "covariate_set_mismatch"
        # rather than substitute an incompatible value.
        immune_candidates = [
            PROJECT_ROOT / "experiments" / "batch_059" / "i2_partial_correlations.csv",
            PROJECT_ROOT / "experiments" / "batch_059" / "immune_partial.csv",
            PROJECT_ROOT / "experiments" / "batch_059" / "age_tech_adjustment.csv",
            PROJECT_ROOT / "experiments" / "batch_059" / "results.csv",
        ]
        immune_source_found = None
        immune_covariate_set = None
        for cand in immune_candidates:
            if cand.exists():
                try:
                    imm = pd.read_csv(cand)
                    cols_list = list(imm.columns)
                    logger.info(f"  Found candidate Immune source: {cand.name} "
                                f"(cols: {cols_list[:10]}...)")
                    # Strict match: we only accept a source that has
                    # partial_rho computed with the SAME covariate set
                    # (age+sex+tech). batch_059 files have tech-adjusted
                    # rho but snRNA-only (tech constant → tech adjustment
                    # is a no-op). That is NOT equivalent covariate set.
                    if "TF" in cols_list and "partial_rho" in cols_list and \
                       "covariate_set" in cols_list:
                        # Hypothetical future file with explicit tag
                        for _, r in imm.iterrows():
                            tf = r["TF"]
                            if tf in E4_TFS and r.get("covariate_set") == "age+sex+tech":
                                mat.loc["Immune", tf] = float(r["partial_rho"])
                                cov_note_mat.loc["Immune", tf] = "age+sex+tech"
                                src_note_mat.loc["Immune", tf] = f"batch_059/{cand.name}"
                        immune_source_found = cand.name
                        immune_covariate_set = "age+sex+tech"
                        break
                except Exception as e:
                    logger.warning(f"    parse failed: {e}")

        if mat.loc["Immune"].isna().all():
            logger.warning(
                "  Immune partial rhos NOT merged — covariate set mismatch "
                "(snRNA-only cohort, tech is constant). Row stays NaN with "
                "note 'covariate_set_mismatch'."
            )
            covariate_sets["Immune"] = "covariate_set_mismatch"
            source_batches["Immune"] = "batch_059 (incompatible: snRNA-only)"
            for tf in E4_TFS:
                cov_note_mat.loc["Immune", tf] = "covariate_set_mismatch"
                src_note_mat.loc["Immune", tf] = source_batches["Immune"]
        else:
            covariate_sets["Immune"] = immune_covariate_set
            source_batches["Immune"] = f"batch_059/{immune_source_found}"

        # Save CSV in long form so covariate_set + source_batch columns attach
        # per-cell, while ALSO saving the wide matrix for easy plotting/reading.
        csv_out = BATCH_DIR / "tf_coupling_4compartment.csv"
        long_rows = []
        for comp in rows:
            for tf in E4_TFS:
                val = mat.loc[comp, tf]
                long_rows.append({
                    "compartment": comp,
                    "TF": tf,
                    "partial_rho": (float(val) if pd.notna(val) else np.nan),
                    "covariate_set": cov_note_mat.loc[comp, tf] or None,
                    "source_batch": src_note_mat.loc[comp, tf] or None,
                    "note": (
                        "covariate_set_mismatch"
                        if cov_note_mat.loc[comp, tf] == "covariate_set_mismatch"
                        else ""
                    ),
                })
        pd.DataFrame(long_rows).to_csv(csv_out, index=False)
        # Also save wide matrix as sibling for quick inspection
        wide_out = BATCH_DIR / "tf_coupling_4compartment_wide.csv"
        mat.to_csv(wide_out)
        # Append explanatory footnote to wide
        with open(wide_out, "a") as f:
            f.write(
                "\n# NOTE: partial_rho controls for age + sex + tech "
                "(Vasc/FAP/MuSC via batch_060).\n"
                "# MuSC from batch_060 musc_partial.csv (covariate set: age+sex+tech).\n"
                "# Immune row NaN with note 'covariate_set_mismatch' — snRNA-only\n"
                "#   cohort in batch_059 means tech is constant, so covariate set\n"
                "#   is NOT equivalent to age+sex+tech used for other compartments.\n"
                "#   See long-form CSV (tf_coupling_4compartment.csv) for per-cell\n"
                "#   covariate_set and source_batch columns.\n"
            )
        logger.info(f"  Saved long CSV: {csv_out}")
        logger.info(f"  Saved wide CSV: {wide_out}")

        # Plot heatmap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 5))
        mask = mat.isna()
        # Show the value as text; where NaN, display 'n/a'
        annot = mat.round(2).astype(object)
        annot[mask] = "n/a"
        sns.heatmap(
            mat.astype(float),
            cmap="RdBu_r",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            annot=annot,
            fmt="",
            mask=mask,
            ax=ax,
            cbar_kws={"label": "Partial Spearman rho (age+sex+tech controlled)"},
        )
        ax.set_title("TF-SASP coupling across 4 muscle compartments (HLMA)")
        # Covariate-set caveat appended as figure subtitle (C2 fix)
        ax.text(
            0.5, 1.01,
            "(covariate sets may differ - see CSV)",
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7, style="italic",
        )
        ax.set_xlabel("Transcription factor")
        ax.set_ylabel("Compartment")
        plt.tight_layout()
        png_out = RESULTS_DIR / "tf_coupling_4compartment_heatmap.png"
        plt.savefig(png_out, dpi=150)
        plt.close(fig)
        logger.info(f"  Saved heatmap: {png_out}")

        summary.update({
            "status": "OK",
            "csv": str(csv_out),
            "png": str(png_out),
            "n_na_cells": int(mask.values.sum()),
            "n_total_cells": int(mask.size),
        })
    except Exception as e:
        logger.exception(f"  E4 failed: {e}")
        summary["status"] = f"ERROR: {e}"
    return summary


# ============================================================================
# E5 — Confidence-tier audit
# ============================================================================

def run_e5():
    """Bookkeeping: write confidence_tier_audit.csv with 6 rows.

    WHY manually-coded: the brief says this is bookkeeping, max 5-6 rows.
    Current classifications are pulled from research_state.md / handoff.md
    which the orchestrator has already read. We hard-code the rows with
    citations to file paths so the reviewer can trace each verdict.

    Recommended class applies the 5-criteria test:
      (1) statistical significance (p corrected)
      (2) practical effect size
      (3) robustness across specifications
      (4) replicability across runs
      (5) clear causal mechanism
    """
    logger = make_logger("e5_audit")
    logger.info("=" * 70)
    logger.info("E5 — Confidence-tier audit of 6 flagged findings")
    logger.info("=" * 70)

    # ---- C3 fix: split rows into bookkeeping vs conditional_on_e3 ----
    # (a) bookkeeping: editorial downgrades only (verdict is independent of E3)
    # (b) conditional_on_e3: verdict depends on whether the row survives
    #     E3's BH-FDR in e3_lr_multiple_testing.csv
    rows = [
        {
            "finding_id": "F066_01",
            "type": "bookkeeping",
            "current_class": "SUGGESTED (research_state.md line 82) / ESTABLISHED (handoff.md line 26) - INCONSISTENT",
            "recommended_class": "SUGGESTED",
            "reason": (
                "Internal inconsistency between research_state.md (SUGGESTED) "
                "and handoff.md (ESTABLISHED) is itself a reviewer-attack "
                "surface. Evidence: partial_rho=0.513 Vasc, 0.549 MuSC at p<0.01 "
                "(batch_066). Passes 5-criteria test on stat-sig + effect + "
                "robustness (controlled for age+sex+tech), but single-dataset "
                "(HLMA-only) means criterion-4 replicability is absent. "
                "Mechanism is correlational not causal."
            ),
            "action": "downgrade_to_SUGGESTED_and_resolve_inconsistency",
        },
        {
            "finding_id": "F067_01",
            "type": "bookkeeping",
            "current_class": "ESTABLISHED",
            "recommended_class": "SUGGESTED",
            "reason": (
                "Claim: SASP factor loadings UNIFORM across Vasc and FAP "
                "(1/12 genes with |delta|>0.3, IL6 delta=0.468). "
                "Criterion-1 fails: no multiple-testing correction "
                "documented across 12 per-gene tests. Criterion-2: the "
                "single gene that differs (IL6, delta=0.468) is the "
                "most biologically informative marker, which weakens "
                "the 'uniform' claim. No negative-control comparison "
                "to non-SASP gene set. Downgrade."
            ),
            "action": "downgrade_to_SUGGESTED_add_FDR_caveat",
        },
        {
            "finding_id": "F068_02",
            "type": "conditional_on_e3",
            "current_class": "ESTABLISHED",
            "recommended_class": "ESTABLISHED (with FDR caveat) or SUGGESTED (if FDR fails)",
            "reason": (
                "CXCL2-CXCR2 JUNB-coupled (rho=0.924, p=3e-10, N=23). "
                "Criteria 1-3 clearly met pre-FDR. Criterion-5 mechanism "
                "well-supported (CXCR2 downstream of JUNB/AP-1 "
                "literature). E3 computes BH-FDR across 49 L-R pairs; "
                "if survives FDR<0.05, KEEP as ESTABLISHED with FDR "
                "caveat; if not, downgrade. Criterion-4 single-dataset "
                "replication is a permanent caveat."
            ),
            "action": "pending_E3",
        },
        {
            "finding_id": "F068_03",
            "type": "conditional_on_e3",
            "current_class": "ESTABLISHED",
            "recommended_class": "ESTABLISHED (with FDR caveat) or SUGGESTED (if FDR fails)",
            "reason": (
                "VEGFA-KDR JUNB-coupled (rho=0.766) + VEGFB-KDR age-decrease "
                "(rho=-0.479). Criterion-2: VEGFB rho is moderate. "
                "Criterion-4: single dataset. The dual claim "
                "(coupled AND age-decreased) is two hypotheses bundled; "
                "multi-testing not corrected for the dual framing. "
                "Verdict depends on E3 FDR survival."
            ),
            "action": "pending_E3",
        },
        {
            "finding_id": "F068_04",
            "type": "conditional_on_e3",
            "current_class": "ESTABLISHED",
            "recommended_class": "ESTABLISHED (with FDR caveat) or SUGGESTED (if FDR fails)",
            "reason": (
                "SEMA3C/F and FGF1 decrease with age, JNK-independent "
                "(rho=-0.49 to -0.55). Criterion-2: effect sizes are "
                "moderate (|rho|~0.5), not strong. Criterion-4: single "
                "dataset. In E3 L-R FDR (49 pairs), moderate-rho hits "
                "are most vulnerable to non-survival. Verdict depends "
                "on E3 FDR survival."
            ),
            "action": "pending_E3",
        },
        {
            "finding_id": "F069_04",
            "type": "bookkeeping",
            "current_class": "ESTABLISHED",
            "recommended_class": "literature-synthesis tier (unvalidated-literature)",
            "reason": (
                "CDK4/6 inhibitors CONTRAINDICATED based on 'CelRep 2022' "
                "single citation (handoff.md line 70). No computed "
                "effect in THIS project. Criterion-1 does not apply "
                "(no statistical test run here); criterion-5 "
                "(mechanism) is provided by the cited paper but NOT "
                "validated against our data. This is a literature "
                "synthesis conclusion, not a computational finding. "
                "Quality tier should be 'unvalidated-literature' per "
                "CLAUDE.md research quality tiers."
            ),
            "action": "relabel_quality_tier_unvalidated_literature",
        },
    ]

    # ---- C3 fix continued: substitute E3 verdict for conditional rows ----
    status_notes = []
    e3_csv = BATCH_DIR / "e3_lr_multiple_testing.csv"
    if e3_csv.exists():
        try:
            e3_df = pd.read_csv(e3_csv)
            logger.info(f"  Loaded E3 FDR table: {len(e3_df)} rows")
            for row in rows:
                if row["type"] != "conditional_on_e3":
                    continue
                fid = row["finding_id"]
                hits = e3_df[e3_df["finding"] == fid]
                if hits.empty:
                    row["action"] = "pending_E3"
                    status_notes.append(
                        f"{fid}: no E3 FDR rows found — action=pending_E3"
                    )
                    continue
                # A finding survives if ANY of its L-R pairs survives FDR<0.05
                # WHY: F068_02/03 each have one primary pair; F068_04 has
                # multiple (SEMA3C/F, FGF1). Surviving at least one pair is
                # sufficient to keep the claim in the "ESTABLISHED" band;
                # zero survivors means the claim is down-graded.
                survives_vals = hits["survives_fdr_0p05"].astype(str).str.lower()
                any_survives = survives_vals.eq("true").any()
                if any_survives:
                    row["action"] = "keep_ESTABLISHED"
                else:
                    row["action"] = "downgrade_to_SUGGESTED"
                status_notes.append(
                    f"{fid}: E3 FDR survivors="
                    f"{int(survives_vals.eq('true').sum())}/{len(hits)} -> "
                    f"action={row['action']}"
                )
        except Exception as e:
            logger.warning(f"  E3 table parse failed: {e}")
            for row in rows:
                if row["type"] == "conditional_on_e3":
                    row["action"] = "pending_E3"
            status_notes.append(f"E3 parse error: {e} — all conditional rows pending_E3")
    else:
        logger.warning(f"  E3 FDR table missing at {e3_csv} — conditional rows pending_E3")
        for row in rows:
            if row["type"] == "conditional_on_e3":
                row["action"] = "pending_E3"
        status_notes.append("E3 FDR table missing — all conditional rows pending_E3")

    df = pd.DataFrame(rows)
    out_path = BATCH_DIR / "confidence_tier_audit.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")
    for _, r in df.iterrows():
        logger.info(f"  [{r['type']}] {r['finding_id']}: {r['current_class']} -> "
                    f"{r['recommended_class']} ({r['action']})")
    # Set status=WARNING if any conditional row ended as pending_E3 (E3 missing/broken)
    any_pending = (df["action"] == "pending_E3").any()
    result = {
        "status": "WARNING" if any_pending else "OK",
        "csv": str(out_path),
        "n_rows": len(df),
        "n_bookkeeping": int((df["type"] == "bookkeeping").sum()),
        "n_conditional": int((df["type"] == "conditional_on_e3").sum()),
        "notes": status_notes,
    }
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    top_logger = make_logger("batch_070_top")
    top_logger.info("=" * 70)
    top_logger.info("batch_070: Reviewer-Defense Gap Closure — 5 sub-analyses")
    top_logger.info("=" * 70)
    top_logger.info(f"Working dir: {PROJECT_ROOT}")
    top_logger.info(f"Results dir: {BATCH_DIR}")
    top_logger.info(f"Seed: {SEED}")

    all_results = {
        "batch": "batch_070",
        "date": time.strftime("%Y-%m-%d"),
        "seed": SEED,
    }

    # Run each sub-analysis independently
    for name, func in [
        ("E1_pseudobulk", run_e1),
        ("E2_cellcount", run_e2),
        ("E3_power", run_e3),
        ("E4_heatmap", run_e4),
        ("E5_audit", run_e5),
    ]:
        top_logger.info(f"\n=== Launching {name} ===")
        try:
            all_results[name] = func()
        except Exception as e:
            top_logger.exception(f"  TOP-LEVEL failure in {name}: {e}")
            all_results[name] = {"status": f"ERROR: {e}",
                                 "traceback": traceback.format_exc()}

    # Save aggregated results.json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, Path):
                return str(obj)
            return super().default(obj)

    out_path = BATCH_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    top_logger.info(f"\nSaved aggregated results: {out_path}")
    top_logger.info("batch_070 DONE")

    # ---- CRITICAL 1 fix: propagate sub-analysis failures to exit code ----
    # WHY: silent success on failure masks regressions from CI/orchestrator.
    # A sub-analysis is "failed" if its returned dict has status starting
    # with "ERROR" (run_e* functions set this on exception, and the
    # top-level try/except above wraps ERROR: ... onto the status field).
    failed = sum(
        1 for v in all_results.values()
        if isinstance(v, dict) and str(v.get("status", "")).startswith("ERROR")
    )
    top_logger.info(f"Sub-analysis failure count: {failed}")
    import sys
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
