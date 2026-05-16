#!/usr/bin/env python3
"""
Batch 063 — Analysis A: Absolute SASP burden by compartment (DESCRIPTIVE).

PURPOSE (WHY): Answer "which compartment PRODUCES the most SASP at the tissue
level?" — directly addressing PI directive #1 and the reviewer-facing question
"do vascular cells really dominate SASP output vs FAPs or SMCs?" Analysis A is
DESCRIPTIVE: we report point estimates with 95% bootstrap CIs and rank
compartments; we do NOT call significance at p<0.05 given known ~20% power at
Cohen's d=0.5, n≈11 vs 12 (exact nct).

Mandatory revisions from 3-critic review (all applied here):
  - Within-tech stratification (HLMA): required, not optional. Young vs old is
    otherwise confounded with snRNA vs scRNA (F101).
  - Within-country stratification (HLMA): required (F051_01 country
    confound rho=0.626).
  - Total load = mean x n_cells is CAVEAT-ONLY; never primary ranking.
  - Regen score ONLY on MuSC donor rows; fibrotic score ONLY on FAP donor
    rows (category error to apply cross-compartment).
  - Senescence panel: ['CDKN2A','GLB1']; CDKN1A dropped to avoid double-
    count with Analysis B TF panel.
  - (Donor x compartment) with <20 cells excluded from ranking.
  - Power disclosed alongside every Cohen's d (exact noncentral t).

INPUTS:
  HLMA: data/Vascular_scsn_RNA.h5ad, data/MuSC_scsn_RNA.h5ad,
        data/OMIX004308-02.h5ad (FAP, z-scored), data/Immune_scsn_RNA.h5ad
  NA:   data/NA_Endothelium_SMC.h5ad, data/SKM_MuSC_human_2023-06-22.h5ad,
        data/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad

OUTPUTS (experiments/batch_063/):
  a_burden_per_donor.csv     -- one row per (dataset x compartment x donor)
  a_burden_rankings.csv      -- compartment-level point+CI in old donors,
                                 pooled AND within-tech AND within-country
  a_burden_summary.json      -- brief-referenced predictions vs observed;
                                 decision-rule triggers
  a_burden.log               -- progress log + full provenance

PROVENANCE: see a_burden.log. Seeds: sc.tl.score_genes (random_state=1),
bootstrap seed=42 (1000 resamples). scanpy 1.11.x; see log for exact version.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

OUTDIR = Path("experiments/batch_063")
OUTDIR.mkdir(parents=True, exist_ok=True)
LOG = OUTDIR / "a_burden.log"

# Panels (batch_050 canonical; brief §Analysis A):
SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'CXCL8', 'IL6',
          'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']
REGEN = ['PAX7', 'MYOD1', 'MYOG', 'CDK1', 'MKI67']            # MuSC ONLY
FIBROTIC = ['COL1A1', 'COL3A1', 'ACTA2', 'FN1', 'TGFBI']      # FAP ONLY
SENESCENCE = ['CDKN2A', 'GLB1']                               # CDKN1A dropped

SCORE_SEED = 1          # sc.tl.score_genes random_state
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 1000
MIN_CELLS_PER_DONOR = 20
HLMA_AGE_THRESHOLD = 50  # young < 50 yr, old >= 50 yr (per brief §Analysis A)

# Cell-type filters per dataset -> map to compartment label
# WHY these filters: use canonical annotation columns; for HLMA files the
# whole file is the compartment (already pre-subsetted). For NA we restrict
# to endothelial/MuSC/fibroblast categories at level1.
DATASETS = [
    # (dataset, compartment, file, obs_annot_col, obs_annot_keep_set_or_None,
    #  donor_col, age_col, tech_col, country_col)
    dict(dataset="HLMA", compartment="Vascular",
         file="data/Vascular_scsn_RNA.h5ad",
         cell_filter=None,  # whole file
         donor_col="sample", age_col="age", tech_col="tech",
         country_col="Country"),
    dict(dataset="HLMA", compartment="MuSC",
         file="data/MuSC_scsn_RNA.h5ad",
         cell_filter=None,
         donor_col="sample", age_col="age", tech_col="tech",
         country_col="Country"),
    dict(dataset="HLMA", compartment="FAP",
         file="data/OMIX004308-02.h5ad",
         cell_filter=("Annotation",
                      ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP',
                       'RUNX2+ FAP', 'CD99+ FAP']),
         donor_col="sample", age_col="age", tech_col="tech",
         country_col="Country"),
    dict(dataset="HLMA", compartment="Immune",
         file="data/Immune_scsn_RNA.h5ad",
         cell_filter=None,
         # Immune uses integer-encoded categoricals; we decode below
         donor_col="sample", age_col="age", tech_col="tech",
         country_col="Country"),
    dict(dataset="NA_atlas", compartment="Vascular",
         file="data/NA_Endothelium_SMC.h5ad",
         cell_filter=("annotation_level1",
                      ['VenEC', 'ArtEC', 'CapEC', 'LymphEC']),
         donor_col="donor_id", age_col="Age_bin", tech_col="assay",
         country_col=None, symbol_remap_col="feature_name"),
    dict(dataset="NA_atlas", compartment="MuSC",
         file="data/SKM_MuSC_human_2023-06-22.h5ad",
         cell_filter=("annotation_level1", ['MuSC']),
         donor_col="DonorID", age_col="Age_bin", tech_col="10X_version",
         country_col=None),
    dict(dataset="NA_atlas", compartment="FAP",  # fibroblasts ~ FAP-equivalent
         file="data/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad",
         cell_filter=("annotation_level1", ['FB', 'PnFB', 'EnFB']),
         donor_col="DonorID", age_col="Age_bin", tech_col="10X_version",
         country_col=None),
]


# ============================================================================
# Utilities
# ============================================================================

def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_line(msg):
    line = f"[{timestamp()}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def peak_mem_gb():
    """Peak resident set size in GB (Linux: ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def cohen_d(a, b):
    """Cohen's d with pooled SD. Returns np.nan if either group <2 or pooled SD=0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled


def power_two_sample_t(d, n1, n2, alpha=0.05):
    """Exact two-sample t-test power at Cohen's d via noncentral t.

    WHY noncentral t: under H1 the test statistic follows a noncentral t with
    ncp = d * sqrt(n1*n2/(n1+n2)) and df = n1+n2-2. This is the textbook
    exact-power formula (Cohen 1988). Two-sided test.
    """
    if not np.isfinite(d) or n1 < 2 or n2 < 2:
        return np.nan
    df = n1 + n2 - 2
    ncp = d * np.sqrt(n1 * n2 / (n1 + n2))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # two-sided: P(|T| > t_crit | ncp)
    p_upper = 1 - stats.nct.cdf(t_crit, df, ncp)
    p_lower = stats.nct.cdf(-t_crit, df, ncp)
    return float(p_upper + p_lower)


def bootstrap_ci(values, n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED,
                 stat_fn=np.mean, alpha=0.05):
    """Percentile bootstrap 95% CI.

    WHY percentile bootstrap: no distributional assumption; valid at these Ns
    for mean-like statistics; brief pre-specifies 1000 resamples + seed=42.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    samples = values[idx]
    stats_arr = stat_fn(samples, axis=1)
    point = stat_fn(values)
    lo = float(np.nanpercentile(stats_arr, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(stats_arr, 100 * (1 - alpha / 2)))
    return (float(point), lo, hi)


def bootstrap_d_ci(a, b, n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Bootstrap 95% CI on Cohen's d by resampling donors within each group.

    WHY donor-level resampling: the independent unit is the donor, not the
    cell. We resample with replacement separately within young and old groups.
    """
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    point = cohen_d(a, b)
    if len(a) < 2 or len(b) < 2:
        return (point, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ds = np.empty(n_resamples)
    for i in range(n_resamples):
        aa = a[rng.integers(0, len(a), size=len(a))]
        bb = b[rng.integers(0, len(b), size=len(b))]
        ds[i] = cohen_d(aa, bb)
    lo = float(np.nanpercentile(ds, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(ds, 100 * (1 - alpha / 2)))
    return (float(point) if np.isfinite(point) else np.nan, lo, hi)


# ============================================================================
# Immune obs decoding
# ============================================================================

def decode_immune_obs(obs: pd.DataFrame) -> pd.DataFrame:
    """Immune h5ad stores tech/Country/Sex/age_pop as integer codes. Decode
    them to strings consistent with the other HLMA files. Mapping was
    inferred by cross-referencing ages in decode_immune_obs_probe.

    tech: 1=snRNA, 0=scRNA (only 1 sample is 0, age=29)
    Country: 0=China, 1=Spain
    age_pop: 0=old_pop, 1=young_pop
    Sex: 0=Female, 1=Male (inferred; not used downstream)
    """
    out = obs.copy()
    if out["tech"].dtype != object:
        out["tech"] = out["tech"].map({1: "snRNA", 0: "scRNA"}).astype(str)
    if out["Country"].dtype != object:
        out["Country"] = out["Country"].map({0: "China", 1: "Spain"}).astype(str)
    if out["age_pop"].dtype != object:
        out["age_pop"] = out["age_pop"].map({0: "old_pop", 1: "young_pop"}).astype(str)
    return out


# ============================================================================
# Per-donor score computation
# ============================================================================

def _get_dense_row_chunk(adata, gene_idx):
    """Return dense np.ndarray (n_cells, len(gene_idx)) for the given genes."""
    X = adata.X
    if sp.issparse(X):
        return np.asarray(X[:, gene_idx].toarray(), dtype=np.float32)
    return np.asarray(X[:, gene_idx], dtype=np.float32)


def score_panel(adata: "sc.AnnData", panel: list, label: str,
                ctrl_size: int = 50, n_bins: int = 25, seed: int = SCORE_SEED):
    """Run sc.tl.score_genes with pre-registered hyperparameters.

    WHY score_genes: it subtracts the mean of a size-matched control pool
    binned by mean expression, making scores comparable across cells and
    robust to baseline expression differences. Critical for z-scored FAP
    data where raw-mean ignores normalization.

    Returns the list of genes actually detected in adata.var_names.
    """
    detected = [g for g in panel if g in adata.var_names]
    if len(detected) == 0:
        log_line(f"  score_panel[{label}]: 0/{len(panel)} genes detected -> all NaN")
        adata.obs[label] = np.nan
        return detected
    try:
        sc.tl.score_genes(
            adata, gene_list=detected, score_name=label,
            ctrl_size=ctrl_size, n_bins=n_bins, random_state=seed,
            use_raw=False,
        )
    except Exception as e:
        log_line(f"  score_panel[{label}] FAILED: {e}; fallback to np.mean")
        # Fallback: simple mean over detected genes
        gene_idx = [adata.var_names.get_loc(g) for g in detected]
        X = adata.X
        if sp.issparse(X):
            vals = np.asarray(X[:, gene_idx].toarray())
        else:
            vals = np.asarray(X[:, gene_idx])
        adata.obs[label] = np.mean(vals, axis=1)
    log_line(f"  score_panel[{label}]: detected {len(detected)}/{len(panel)}: {detected}")
    return detected


def process_dataset(cfg: dict) -> pd.DataFrame:
    """Load one dataset, score cells, aggregate to per-donor rows.

    Returns DataFrame of per-(donor x compartment) rows.
    """
    t0 = time.time()
    ds, comp, fp = cfg["dataset"], cfg["compartment"], cfg["file"]
    log_line(f"=== {ds}/{comp}  file={fp}")
    log_line(f"  reading (not backed) ...")
    a = sc.read_h5ad(fp)
    log_line(f"  loaded: shape={a.shape}  peak_mem={peak_mem_gb():.2f} GB")

    # Remap var_names to gene symbols if needed (NA_Endothelium uses ENSEMBL IDs)
    remap_col = cfg.get("symbol_remap_col")
    if remap_col and remap_col in a.var.columns:
        before_symbols = list(a.var_names[:3])
        new_names = a.var[remap_col].astype(str).values
        # keep unique names; for duplicates append suffix (anndata requires unique)
        a.var_names = pd.Index(new_names).astype(str)
        a.var_names_make_unique()
        log_line(f"  var_names remapped via {remap_col}; before={before_symbols}, "
                 f"after={list(a.var_names[:3])}")

    # Apply cell-type filter
    if cfg["cell_filter"] is not None:
        col, keep = cfg["cell_filter"]
        before = a.n_obs
        mask = a.obs[col].astype(str).isin(keep)
        a = a[mask].copy()
        log_line(f"  filter {col} in {keep}: {before} -> {a.n_obs} cells")

    # Decode Immune integer codes
    if ds == "HLMA" and comp == "Immune":
        a.obs = decode_immune_obs(a.obs)
        log_line(f"  Immune obs decoded (tech/Country/age_pop -> strings)")

    # Attach age_group
    age_col = cfg["age_col"]
    if ds == "HLMA":
        # HLMA ages are continuous; threshold at HLMA_AGE_THRESHOLD
        ages = pd.to_numeric(a.obs[age_col], errors="coerce")
        a.obs["age_group"] = np.where(ages >= HLMA_AGE_THRESHOLD, "old", "young")
        a.obs["age_numeric"] = ages
    else:
        # NA dataset uses Age_group directly ('young'/'old'); no continuous age
        a.obs["age_group"] = a.obs[age_col].astype(str).str.lower()
        a.obs["age_numeric"] = np.nan

    # Tech
    if cfg["tech_col"] is not None:
        a.obs["tech"] = a.obs[cfg["tech_col"]].astype(str)
    else:
        a.obs["tech"] = "NA"
    # Country
    if cfg["country_col"] is not None:
        a.obs["country"] = a.obs[cfg["country_col"]].astype(str)
    else:
        a.obs["country"] = "NA"

    # Score panels
    log_line("  scoring SASP12, senescence panels on all cells")
    score_panel(a, SASP12, "SASP12_score")
    score_panel(a, SENESCENCE, "senescence_score")

    # Regen ONLY for MuSC compartment cells
    if comp == "MuSC":
        log_line("  scoring regen panel (MuSC only)")
        score_panel(a, REGEN, "regen_score")
    else:
        a.obs["regen_score"] = np.nan

    # Fibrotic ONLY for FAP compartment cells
    if comp == "FAP":
        log_line("  scoring fibrotic panel (FAP only)")
        score_panel(a, FIBROTIC, "fibrotic_score")
    else:
        a.obs["fibrotic_score"] = np.nan

    # SASP_high threshold: 75th percentile of pooled YOUNG-compartment cells
    young_mask = (a.obs["age_group"].astype(str) == "young").to_numpy()
    n_young_cells = int(young_mask.sum())
    if n_young_cells >= 50:
        thr = float(np.nanpercentile(a.obs.loc[young_mask, "SASP12_score"].values, 75))
    else:
        thr = float(np.nanpercentile(a.obs["SASP12_score"].values, 75))
        log_line(f"  WARN: only {n_young_cells} young cells (<50), using pooled 75th pct as threshold")
    log_line(f"  SASP_high threshold (75th pct of young pool): {thr:.4f}")

    a.obs["SASP_high"] = (a.obs["SASP12_score"].values > thr).astype(int)

    # Aggregate per donor
    donor_col = cfg["donor_col"]
    grp = a.obs.groupby(donor_col, observed=True)
    rows = []
    for donor, sub in grp:
        n_cells = int(len(sub))
        if n_cells == 0:
            continue
        row = {
            "dataset": ds,
            "compartment": comp,
            "donor": str(donor),
            "n_cells": n_cells,
            "age_group": sub["age_group"].iloc[0],
            "age_numeric": float(sub["age_numeric"].iloc[0])
                          if pd.notna(sub["age_numeric"].iloc[0]) else np.nan,
            "tech": sub["tech"].iloc[0],
            "country": sub["country"].iloc[0],
            "mean_SASP": float(np.nanmean(sub["SASP12_score"].values)),
            "SASP_high_fraction": float(np.nanmean(sub["SASP_high"].values)),
            "senescence_score": float(np.nanmean(sub["senescence_score"].values)),
            "regen_score": (float(np.nanmean(sub["regen_score"].values))
                            if comp == "MuSC" else np.nan),
            "fibrotic_score": (float(np.nanmean(sub["fibrotic_score"].values))
                               if comp == "FAP" else np.nan),
            "sasp_high_threshold": thr,
        }
        # Caveat-only total load
        row["total_load_caveat"] = row["mean_SASP"] * n_cells
        # Donor-level consistency check: age_group must be constant per donor
        ag_u = sub["age_group"].unique()
        if len(ag_u) > 1:
            log_line(f"  WARN: donor {donor} has multiple age_group values {ag_u}")
        rows.append(row)
    df = pd.DataFrame(rows)
    log_line(f"  aggregated {len(df)} donor rows (t={time.time()-t0:.1f}s; peak_mem={peak_mem_gb():.2f} GB)")
    del a
    gc.collect()
    return df


# ============================================================================
# Ranking / CI output
# ============================================================================

def compartment_rankings(per_donor: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset x stratum x compartment x age_group), compute
    bootstrap 95% CI on mean_SASP and SASP_high_fraction. Also compute
    Cohen's d (young vs old) with bootstrap CI and exact-nct power at the
    observed d.

    Strata computed:
      - pooled     : no stratification (HLMA and NA)
      - within_tech_snRNA / within_tech_scRNA : HLMA only (NA has single tech)
      - within_country_China / within_country_Spain : HLMA only
    """
    rows = []
    # Exclude (donor x compartment) with <20 cells from ranking
    df = per_donor[per_donor["n_cells"] >= MIN_CELLS_PER_DONOR].copy()

    def _emit(sub, stratum, dataset, compartment, age):
        metrics = ["mean_SASP", "SASP_high_fraction",
                   "senescence_score", "regen_score", "fibrotic_score",
                   "total_load_caveat"]
        for met in metrics:
            vals = pd.to_numeric(sub[met], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            pt, lo, hi = bootstrap_ci(vals)
            rows.append(dict(
                dataset=dataset, compartment=compartment, stratum=stratum,
                age_group=age, metric=met, n_donor=int(len(vals)),
                point=pt, ci_lo=lo, ci_hi=hi,
            ))

    def _emit_effect(young_sub, old_sub, stratum, dataset, compartment):
        metrics = ["mean_SASP", "SASP_high_fraction", "senescence_score"]
        if compartment == "MuSC":
            metrics.append("regen_score")
        if compartment == "FAP":
            metrics.append("fibrotic_score")
        for met in metrics:
            y = pd.to_numeric(young_sub[met], errors="coerce").dropna().values
            o = pd.to_numeric(old_sub[met], errors="coerce").dropna().values
            if len(y) < 2 or len(o) < 2:
                continue
            # Cohen's d: old - young (positive d means old > young)
            d, dlo, dhi = bootstrap_d_ci(o, y)
            pw = power_two_sample_t(d, len(o), len(y)) if np.isfinite(d) else np.nan
            # Mann-Whitney U (descriptive p; NOT primary inference)
            try:
                u, pval = stats.mannwhitneyu(o, y, alternative="two-sided")
            except Exception:
                u, pval = np.nan, np.nan
            rows.append(dict(
                dataset=dataset, compartment=compartment, stratum=stratum,
                age_group="old_vs_young_effect", metric=met,
                n_donor=int(len(y) + len(o)), n_young=int(len(y)),
                n_old=int(len(o)), point=float(d),
                ci_lo=float(dlo) if np.isfinite(dlo) else np.nan,
                ci_hi=float(dhi) if np.isfinite(dhi) else np.nan,
                power_at_d_observed=float(pw) if np.isfinite(pw) else np.nan,
                mannwhitney_p=float(pval) if np.isfinite(pval) else np.nan,
            ))

    for (ds, comp), block in df.groupby(["dataset", "compartment"], observed=True):
        # Pooled per age_group
        for age in ["young", "old"]:
            sub = block[block["age_group"] == age]
            if len(sub):
                _emit(sub, "pooled", ds, comp, age)
        # Pooled effect size (old vs young)
        _emit_effect(block[block["age_group"] == "young"],
                     block[block["age_group"] == "old"],
                     "pooled", ds, comp)

        # Within-tech (HLMA only — NA tech column is always a single value
        # per dataset so stratification is degenerate)
        if ds == "HLMA":
            for tech in sorted(block["tech"].dropna().unique()):
                tsub = block[block["tech"] == tech]
                if len(tsub) == 0:
                    continue
                stratum = f"within_tech_{tech}"
                for age in ["young", "old"]:
                    ageblk = tsub[tsub["age_group"] == age]
                    if len(ageblk):
                        _emit(ageblk, stratum, ds, comp, age)
                _emit_effect(tsub[tsub["age_group"] == "young"],
                             tsub[tsub["age_group"] == "old"],
                             stratum, ds, comp)
            # Within-country (HLMA only)
            for country in sorted(block["country"].dropna().unique()):
                csub = block[block["country"] == country]
                if len(csub) == 0:
                    continue
                stratum = f"within_country_{country}"
                for age in ["young", "old"]:
                    ageblk = csub[csub["age_group"] == age]
                    if len(ageblk):
                        _emit(ageblk, stratum, ds, comp, age)
                _emit_effect(csub[csub["age_group"] == "young"],
                             csub[csub["age_group"] == "old"],
                             stratum, ds, comp)

    return pd.DataFrame(rows)


# ============================================================================
# Summary / decision rules
# ============================================================================

def build_summary(per_donor: pd.DataFrame, rankings: pd.DataFrame) -> dict:
    """Evaluate brief §Analysis A predictions vs observed rankings."""
    summary = {
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "scanpy_version": sc.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "score_genes_params": {"ctrl_size": 50, "n_bins": 25,
                                   "random_state": SCORE_SEED, "use_raw": False},
            "bootstrap_params": {"n_resamples": BOOTSTRAP_N,
                                 "seed": BOOTSTRAP_SEED},
            "min_cells_per_donor": MIN_CELLS_PER_DONOR,
            "hlma_age_threshold": HLMA_AGE_THRESHOLD,
            "panels": {"SASP12": SASP12, "regen_MuSC_only": REGEN,
                       "fibrotic_FAP_only": FIBROTIC,
                       "senescence": SENESCENCE},
        },
        "counts": {},
        "predictions_vs_observed": {},
        "decision_rule_triggers": {},
    }

    # Donor counts per (dataset, compartment, age_group)
    for (ds, comp, ag), sub in per_donor.groupby(["dataset", "compartment", "age_group"], observed=True):
        # only count donors passing the 20-cell filter
        sub2 = sub[sub["n_cells"] >= MIN_CELLS_PER_DONOR]
        summary["counts"][f"{ds}|{comp}|{ag}"] = {
            "n_donor_total": int(len(sub)),
            "n_donor_pass_20cells": int(len(sub2)),
            "median_n_cells": float(np.median(sub["n_cells"])) if len(sub) else np.nan,
        }

    # --- Predictions from brief §Analysis A ---
    def _get(ds, stratum, age, metric, comp):
        sel = rankings[
            (rankings["dataset"] == ds)
            & (rankings["compartment"] == comp)
            & (rankings["stratum"] == stratum)
            & (rankings["age_group"] == age)
            & (rankings["metric"] == metric)
        ]
        if len(sel) == 0:
            return None
        r = sel.iloc[0]
        return {"point": r["point"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                "n": int(r["n_donor"])}

    def _ci_above(a, b):
        """a's CI entirely above b's CI."""
        if a is None or b is None: return None
        return bool(a["ci_lo"] > b["ci_hi"])

    def _ci_overlap(a, b):
        if a is None or b is None: return None
        return not (a["ci_lo"] > b["ci_hi"] or b["ci_lo"] > a["ci_hi"])

    pred = summary["predictions_vs_observed"]

    # Prediction 1: Vascular mean_SASP CI above MuSC and FAP in OLD (pooled)
    v = _get("HLMA", "pooled", "old", "mean_SASP", "Vascular")
    m = _get("HLMA", "pooled", "old", "mean_SASP", "MuSC")
    f = _get("HLMA", "pooled", "old", "mean_SASP", "FAP")
    pred["HLMA_old_vascular_vs_musc_mean_SASP"] = {
        "vascular": v, "musc": m,
        "vascular_ci_above_musc": _ci_above(v, m),
        "ci_overlap": _ci_overlap(v, m),
    }
    pred["HLMA_old_vascular_vs_fap_mean_SASP"] = {
        "vascular": v, "fap": f,
        "vascular_ci_above_fap": _ci_above(v, f),
        "ci_overlap": _ci_overlap(v, f),
    }

    # Prediction 2: SASP_high_fraction: Vascular and FAP overlapping CIs in
    # old; MuSC lowest
    v2 = _get("HLMA", "pooled", "old", "SASP_high_fraction", "Vascular")
    m2 = _get("HLMA", "pooled", "old", "SASP_high_fraction", "MuSC")
    f2 = _get("HLMA", "pooled", "old", "SASP_high_fraction", "FAP")
    pred["HLMA_old_SASP_high_fraction"] = {
        "vascular": v2, "musc": m2, "fap": f2,
        "vascular_fap_overlap": _ci_overlap(v2, f2),
        "musc_below_both": (
            (_ci_above(v2, m2) if v2 and m2 else None),
            (_ci_above(f2, m2) if f2 and m2 else None),
        ),
    }

    # Prediction 3: within-tech HLMA — does ranking reverse?
    for tech in ["snRNA", "scRNA"]:
        strat = f"within_tech_{tech}"
        v3 = _get("HLMA", strat, "old", "mean_SASP", "Vascular")
        m3 = _get("HLMA", strat, "old", "mean_SASP", "MuSC")
        f3 = _get("HLMA", strat, "old", "mean_SASP", "FAP")
        pred[f"HLMA_old_{strat}_mean_SASP"] = {
            "vascular": v3, "musc": m3, "fap": f3,
            "vascular_ci_above_musc": _ci_above(v3, m3),
            "vascular_ci_above_fap": _ci_above(v3, f3),
            "any_ci_overlap_with_vascular": (
                _ci_overlap(v3, m3) if v3 and m3 else None,
                _ci_overlap(v3, f3) if v3 and f3 else None,
            ),
        }

    # Prediction 4: within-country HLMA
    for cc in ["China", "Spain"]:
        strat = f"within_country_{cc}"
        v4 = _get("HLMA", strat, "old", "mean_SASP", "Vascular")
        m4 = _get("HLMA", strat, "old", "mean_SASP", "MuSC")
        f4 = _get("HLMA", strat, "old", "mean_SASP", "FAP")
        pred[f"HLMA_old_{strat}_mean_SASP"] = {
            "vascular": v4, "musc": m4, "fap": f4,
            "vascular_ci_above_musc": _ci_above(v4, m4),
            "vascular_ci_above_fap": _ci_above(v4, f4),
        }

    # Prediction 5: Regen axis MuSC-only old vs young effect (expect decline)
    rsel = rankings[
        (rankings["compartment"] == "MuSC")
        & (rankings["metric"] == "regen_score")
        & (rankings["stratum"] == "pooled")
        & (rankings["age_group"] == "old_vs_young_effect")
    ]
    pred["MuSC_regen_decline_old_vs_young"] = rsel.to_dict("records")

    # Prediction 6: Fibrotic axis FAP-only old vs young effect (expect increase)
    fsel = rankings[
        (rankings["compartment"] == "FAP")
        & (rankings["metric"] == "fibrotic_score")
        & (rankings["stratum"] == "pooled")
        & (rankings["age_group"] == "old_vs_young_effect")
    ]
    pred["FAP_fibrotic_increase_old_vs_young"] = fsel.to_dict("records")

    # --- Decision rule triggers from brief ---
    triggers = summary["decision_rule_triggers"]
    triggers["vascular_per_cell_dominance_SUPPORTED"] = bool(
        pred.get("HLMA_old_vascular_vs_musc_mean_SASP", {}).get(
            "vascular_ci_above_musc") is True
        and pred.get("HLMA_old_vascular_vs_fap_mean_SASP", {}).get(
            "vascular_ci_above_fap") is True
        and all(pred.get(f"HLMA_old_within_tech_{t}_mean_SASP", {}).get(
            "vascular_ci_above_musc") is True for t in ["snRNA", "scRNA"]
            if pred.get(f"HLMA_old_within_tech_{t}_mean_SASP") is not None)
        and all(pred.get(f"HLMA_old_within_country_{c}_mean_SASP", {}).get(
            "vascular_ci_above_musc") is True for c in ["China", "Spain"]
            if pred.get(f"HLMA_old_within_country_{c}_mean_SASP") is not None)
    )
    triggers["thesis_softened_within_tech_or_country_overlap"] = bool(
        any(pred.get(f"HLMA_old_within_tech_{t}_mean_SASP", {}).get(
                "vascular_ci_above_musc") is False
            for t in ["snRNA", "scRNA"]
            if pred.get(f"HLMA_old_within_tech_{t}_mean_SASP") is not None)
        or any(pred.get(f"HLMA_old_within_country_{c}_mean_SASP", {}).get(
                "vascular_ci_above_musc") is False
               for c in ["China", "Spain"]
               if pred.get(f"HLMA_old_within_country_{c}_mean_SASP") is not None)
    )
    # Under-powered reminder — every negative result is INCONCLUSIVE, not REFUTED
    triggers["power_caveat"] = (
        "At effective n approx 9-14 donors per compartment x age, power is "
        "~20% for Cohen's d=0.5 and ~63% for d=1.0 (exact nct, alpha=0.05 "
        "two-sided). Negative results classify as INCONCLUSIVE, not REFUTED, "
        "per brief Decision Rule §Analysis A."
    )

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Fresh log file
    with open(LOG, "w") as f:
        f.write(f"[{timestamp()}] === batch_063 run_a_burden.py START ===\n")
    log_line(f"python {sys.version.split()[0]}  scanpy {sc.__version__}  "
             f"numpy {np.__version__}  pandas {pd.__version__}  "
             f"scipy {stats.__name__.split('.')[0]}={__import__('scipy').__version__}")
    log_line(f"cwd={os.getcwd()}")
    log_line(f"seeds: score_genes={SCORE_SEED}, bootstrap={BOOTSTRAP_SEED}, "
             f"n_resamples={BOOTSTRAP_N}")

    per_donor_frames = []
    for cfg in DATASETS:
        try:
            df = process_dataset(cfg)
            per_donor_frames.append(df)
        except Exception as e:
            log_line(f"FAILED {cfg['dataset']}/{cfg['compartment']}: {e}")
            log_line(traceback.format_exc())

    if not per_donor_frames:
        log_line("FATAL: no datasets processed")
        sys.exit(1)

    per_donor = pd.concat(per_donor_frames, ignore_index=True)
    pd_path = OUTDIR / "a_burden_per_donor.csv"
    per_donor.to_csv(pd_path, index=False)
    log_line(f"WROTE {pd_path}  n_rows={len(per_donor)}")

    log_line("computing rankings with bootstrap CIs ...")
    rankings = compartment_rankings(per_donor)
    rk_path = OUTDIR / "a_burden_rankings.csv"
    rankings.to_csv(rk_path, index=False)
    log_line(f"WROTE {rk_path}  n_rows={len(rankings)}")

    summary = build_summary(per_donor, rankings)
    sm_path = OUTDIR / "a_burden_summary.json"
    with open(sm_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log_line(f"WROTE {sm_path}")

    log_line(f"DONE. peak_mem={peak_mem_gb():.2f} GB")


if __name__ == "__main__":
    main()
