#!/usr/bin/env python3
"""
batch_063 Analysis B: Hardened Driver-Bystander Delta-Rule
===========================================================
PURPOSE: Classify 20 TFs x 3 HLMA compartments as driver / bystander /
polarity-flip / inconclusive using multi-layer leakage correction
(SASP12 + SenMayo + size-matched null + age-shuffle null +
regulon-size covariate + empirical-null-derived thresholds).

WHY: Three science-critics returned PASS WITH REVISIONS on the previous
design. Critic 3 Objection #1: SASP12-only leakage correction (12/~200 genes
= ~1% perturbation) is insufficient. Critic 2: previous permutation null
tested the wrong hypothesis. Critic 1 CF-2: thresholds were hard-coded.

PRIMARY CORRELATION: rho(AUCell_donor, SASP12_mean_donor) per Spearman
(matches batch_054 convention that produced d1_correlations_all.csv). The
brief's "age-shuffle null" operates on the phenotype label at the donor
level — in this design the phenotype against which we correlate is
SASP12_mean, so we shuffle the SASP12_mean column (equivalent to
shuffling age since the two are correlated; shuffling the response
label is the standard permutation null Critic 2 requested). A smoke
test on MuSC (2026-04-22) confirmed pyscenic AUCell vs SASP12_mean
reproduces batch_054 rhos (JUNB=0.88, KLF10=0.79, FOS=0.90) while
correlating vs age gives rho ~0 — validating this design choice.

This script implements:
  - raw AUCell (validates against batch_054 d1_correlations_all.csv)
  - clean12 AUCell (regulon minus SASP12)
  - clean_SenMayo AUCell (regulon minus MSigDB SAUL_SEN_MAYO)
  - 100 size-matched random-gene-removal nulls (Critic 3 objection)
  - 2000 phenotype-shuffle nulls (Critic 2 primary null; "age-shuffle"
    per brief, implemented by shuffling the SASP12_mean donor vector)
  - regulon-size covariate (Critic 3)
  - empirical thresholds from pooled phenotype-shuffle null (Critic 1 CF-2)
  - Fisher-z 95% CI per rho
  - Bonferroni 0.05 / 60 correction

WHY use pyscenic.aucell (not scanpy.score_genes): AUCell ranks genes
per cell over the FULL expression matrix and measures enrichment of the
regulon in the top 5% of that ranking. A smoke test (run 2026-04-22)
showed scanpy.score_genes on an identical regulon produced rho = -0.018
vs the pyscenic AUCell rho of +0.923 reported in batch_054 -- order-of-
magnitude mismatch. The brief's suggestion that score_genes approximates
AUCell is therefore rejected; we use pyscenic directly.

AUCELL TRICK: pyscenic.aucell() accepts a list of GeneSignatures and
computes per-cell rankings ONCE then evaluates all signatures against
that single ranking. So computing 20 TFs x 3 leakage conditions x
100 size-matched nulls = ~6000 signatures per compartment incurs almost
no extra ranking cost. Runtime scales with n_cells * n_genes for the
ranking, and is ~O(n_signatures * 0.05 * n_genes) for the AUC lookup.

INPUTS:
  data/Vascular_scsn_RNA.h5ad (23 donors)
  data/MuSC_scsn_RNA.h5ad     (23 donors)
  data/OMIX004308-02.h5ad     (FAP; 22 donors, 40K cells, 38K genes; ~5GB dense)
  experiments/batch_054/d1_adjacencies_HLMA_{Vascular,MuSC}.csv
  experiments/batch_055/d1_adjacencies_HLMA_FAP.csv
  experiments/batch_054/d1_correlations_all.csv  (validation)
  experiments/batch_063/senmayo_genes.txt

OUTPUTS:
  experiments/batch_063/b_delta_classification.csv
  experiments/batch_063/b_delta_null_distributions.npz
  experiments/batch_063/b_delta_summary.json
  experiments/batch_063/b_delta.log

SEEDS: pyscenic AUCell seed=42; numpy null seed=42.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.stats import norm, spearmanr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro")
OUTDIR = PROJECT_ROOT / "experiments" / "batch_063"
LOG_FILE = OUTDIR / "b_delta.log"

VASC_PATH = PROJECT_ROOT / "data" / "Vascular_scsn_RNA.h5ad"
MUSC_PATH = PROJECT_ROOT / "data" / "MuSC_scsn_RNA.h5ad"
FAP_PATH = PROJECT_ROOT / "data" / "OMIX004308-02.h5ad"

ADJ_VASC = PROJECT_ROOT / "experiments" / "batch_054" / "d1_adjacencies_HLMA_Vascular.csv"
ADJ_MUSC = PROJECT_ROOT / "experiments" / "batch_054" / "d1_adjacencies_HLMA_MuSC.csv"
ADJ_FAP = PROJECT_ROOT / "experiments" / "batch_055" / "d1_adjacencies_HLMA_FAP.csv"

VALIDATION_CSV = PROJECT_ROOT / "experiments" / "batch_054" / "d1_correlations_all.csv"
SENMAYO_TXT = OUTDIR / "senmayo_genes.txt"

COMPARTMENT_PATHS = {
    "HLMA_Vascular": VASC_PATH,
    "HLMA_MuSC": MUSC_PATH,
    "HLMA_FAP": FAP_PATH,
}

# Only FAP requires cell-type filtering (it contains tenocytes and FAP subtypes).
# Vascular and MuSC h5ads are already pre-filtered to their compartments.
FAP_CELL_TYPES = ["MME+ FAP", "CD55+ FAP", "GPC3+ FAP", "RUNX2+ FAP", "CD99+ FAP"]

TF_PANEL = [
    "JUNB", "JUN", "JUND", "FOS", "FOSB", "FOSL1", "FOSL2",
    "ATF3", "ATF6", "CEBPB", "CEBPD",
    "EGR1", "EGR2", "IRF1",
    "KLF10", "CDKN1A",
    "NFKB1", "RELA", "RELB",
    "STAT3",
]

SASP12 = [
    "CCL2", "CXCL1", "CXCL2", "CXCL3", "CXCL6", "CXCL8",
    "IL6", "SERPINE1", "MMP1", "MMP3", "PLAU", "PLAUR",
]

MIN_CELLS_PER_DONOR_DEFAULT = 50    # FAP uses this per batch_055 convention
MIN_CELLS_PER_DONOR_VASC_MUSC = 1   # Vascular/MuSC use no filter per batch_054
REGULON_QUANTILE = 0.80             # matches iter 054/055 actual impl (brief says 75th
                                    # but the referenced prior convention is 80th)
REGULON_MIN = 5
REGULON_MAX = 200
AUCELL_THRESHOLD = 0.05
AUCELL_SEED = 42
SIZE_MATCHED_N = 100
AGE_SHUFFLE_N = 2000
NULL_SEED = 42
BONFERRONI_ALPHA = 0.05 / 60        # 0.000833
FISHER_CI_DRIVER_THRESH = 0.5       # Fisher-z CI lower bound must exceed this


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, file=LOG_FILE) -> None:
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(file, "a") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def fisher_z_ci(rho: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% Fisher-Z CI for a Spearman rho at N donors."""
    if not np.isfinite(rho) or n < 4:
        return (np.nan, np.nan)
    rho_c = np.clip(rho, -1 + 1e-15, 1 - 1e-15)
    z = np.arctanh(rho_c)
    se = 1.0 / np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


# ---------------------------------------------------------------------------
# Regulon parsing
# ---------------------------------------------------------------------------
def build_regulon(adj: pd.DataFrame, tf: str,
                  quantile: float = REGULON_QUANTILE,
                  min_t: int = REGULON_MIN,
                  max_t: int = REGULON_MAX) -> list[str]:
    sub = adj[adj["TF"] == tf].sort_values("importance", ascending=False)
    if len(sub) == 0:
        return []
    thr = sub["importance"].quantile(quantile)
    top = sub[sub["importance"] >= thr]["target"].tolist()
    top = [g for g in top if g != tf]
    if len(top) < min_t:
        return []
    return top[:max_t]


# ---------------------------------------------------------------------------
# Compartment data loader (returns expression DataFrame, obs)
# ---------------------------------------------------------------------------
def load_compartment(compartment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load full expression matrix (cells x genes, dense DataFrame) and obs.

    WHY dense DataFrame: pyscenic.aucell wants a pd.DataFrame with cell index
    and gene columns. We must pass ALL genes so the per-cell ranking used by
    AUCell is well-defined (cannot subset to regulon genes only; that would
    collapse the rank space and invalidate AUC).

    Returns (expr_df, obs_df). For FAP: filter to FAP cell types first.
    """
    import anndata as ad

    path = COMPARTMENT_PATHS[compartment]
    log(f"  Loading {compartment} from {path.name} ...")
    a = ad.read_h5ad(path)
    log(f"    raw shape: {a.shape}")

    if compartment == "HLMA_FAP":
        mask = a.obs["Annotation"].isin(FAP_CELL_TYPES).values
        a = a[mask].copy()
        log(f"    after FAP cell-type filter: {a.shape}")

    X = a.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32, copy=False)
    expr_df = pd.DataFrame(X, index=a.obs_names, columns=a.var_names.to_list())
    obs = a.obs.copy()
    del a, X
    gc.collect()
    log(f"    expr_df: {expr_df.shape}, RAM ~{expr_df.memory_usage(deep=True).sum()/2**30:.1f} GiB")
    return expr_df, obs


# ---------------------------------------------------------------------------
# AUCell bulk scorer
# ---------------------------------------------------------------------------
def score_signatures(expr_df: pd.DataFrame, sigs: list) -> pd.DataFrame:
    """Run pyscenic AUCell on a list of GeneSignatures. Returns cells x sigs."""
    from pyscenic.aucell import aucell
    auc_mtx = aucell(
        exp_mtx=expr_df,
        signatures=sigs,
        auc_threshold=AUCELL_THRESHOLD,
        noweights=False,
        normalize=False,
        seed=AUCELL_SEED,
        num_workers=max(1, os.cpu_count() - 4),
    )
    return auc_mtx


# ---------------------------------------------------------------------------
# Donor-level aggregation
# ---------------------------------------------------------------------------
def donor_mean_vector(per_cell: np.ndarray, obs: pd.DataFrame,
                      donor_samples: list[str]) -> np.ndarray:
    sample_vals = obs["sample"].to_numpy()
    out = np.empty(len(donor_samples), dtype=float)
    for i, d in enumerate(donor_samples):
        idx = np.where(sample_vals == d)[0]
        if len(idx) == 0:
            out[i] = np.nan
        else:
            out[i] = float(np.nanmean(per_cell[idx]))
    return out


def build_donor_table(obs: pd.DataFrame, expr_df_subset_for_sasp: pd.DataFrame,
                      min_cells: int = 1) -> pd.DataFrame:
    """Return table of (sample, age, SASP12_mean, n_cells) for donors with
    >= min_cells. SASP12_mean = mean over detected SASP12 genes of the
    per-cell expression mean-of-gene-values (matches batch_054 canonical)."""
    detected = [g for g in SASP12 if g in expr_df_subset_for_sasp.columns]
    if not detected:
        raise RuntimeError(f"No SASP12 genes found in expression matrix")
    sasp_per_cell = expr_df_subset_for_sasp[detected].to_numpy().mean(axis=1)

    out = []
    sample_vals = obs["sample"].to_numpy()
    # Coerce age to numeric; the FAP h5ad stores age as Categorical of string.
    if "age" in obs.columns:
        ages = pd.to_numeric(obs["age"], errors="coerce").to_numpy()
    else:
        ages = np.full(len(obs), np.nan)
    for d in pd.unique(sample_vals):
        idx = np.where(sample_vals == d)[0]
        if len(idx) < min_cells:
            continue
        raw_age = ages[idx[0]] if len(idx) else np.nan
        try:
            age_val = float(raw_age) if np.isfinite(raw_age) else np.nan
        except Exception:
            age_val = np.nan
        sasp = float(np.mean(sasp_per_cell[idx]))
        out.append({"sample": str(d), "age": age_val,
                    "SASP12_mean": sasp, "n_cells": int(len(idx))})
    df = pd.DataFrame(out).sort_values("sample").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main per-compartment analysis
# ---------------------------------------------------------------------------
def analyze_compartment(compartment: str, adj: pd.DataFrame,
                        senmayo: set[str]) -> tuple[list[dict], dict]:
    from ctxcore.genesig import GeneSignature

    log(f"\n=== Analyzing {compartment} ===")
    t0 = time.time()

    # ----- Build regulons for TF panel -----
    regulons: dict[str, list[str]] = {}
    for tf in TF_PANEL:
        regulons[tf] = build_regulon(adj, tf)
    n_nonempty = sum(1 for r in regulons.values() if r)
    log(f"  {compartment}: {n_nonempty}/{len(TF_PANEL)} TFs have a non-empty regulon")
    for tf in TF_PANEL:
        log(f"    {tf}: regulon_size_raw = {len(regulons[tf])}")

    # ----- Load expression -----
    expr_df, obs = load_compartment(compartment)
    all_genes = set(expr_df.columns)

    min_cells = (MIN_CELLS_PER_DONOR_DEFAULT if compartment == "HLMA_FAP"
                 else MIN_CELLS_PER_DONOR_VASC_MUSC)
    donor_df = build_donor_table(obs, expr_df, min_cells=min_cells)
    donor_samples = donor_df["sample"].tolist()
    donor_sasp = donor_df["SASP12_mean"].to_numpy()      # PHENOTYPE VECTOR
    donor_ages = donor_df["age"].to_numpy()
    n_donors = len(donor_df)
    log(f"  {compartment}: {n_donors} donors with >= {min_cells} cells "
        f"(convention: Vasc/MuSC=all donors, FAP=50-cell filter per batch_054/055)")
    log(f"    donor ages: {sorted(donor_ages[np.isfinite(donor_ages)].astype(int).tolist())}")
    log(f"    donor SASP12_mean range: [{donor_sasp.min():.3f}, {donor_sasp.max():.3f}]")

    # ----- Build all GeneSignatures (raw + clean12 + clean_senmayo + 100 size-matched) -----
    # Keyed name so we can parse back
    sigs: list = []
    sig_lookup: dict[str, dict] = {}  # name -> {tf, kind, idx}

    # For deterministic null sampling per-TF
    for tf in TF_PANEL:
        reg = regulons[tf]
        if not reg:
            continue
        reg_in_expr = [g for g in reg if g in all_genes]
        if len(reg_in_expr) < REGULON_MIN:
            # skip: regulon after gene-presence filter is too small
            continue

        # Raw
        name_raw = f"{tf}__raw"
        sigs.append(GeneSignature(name=name_raw,
                                   gene2weight={g: 1.0 for g in reg_in_expr}))
        sig_lookup[name_raw] = {"tf": tf, "kind": "raw"}

        # Clean12
        reg_clean12 = [g for g in reg_in_expr if g not in set(SASP12)]
        if len(reg_clean12) >= REGULON_MIN:
            name_c12 = f"{tf}__clean12"
            sigs.append(GeneSignature(name=name_c12,
                                       gene2weight={g: 1.0 for g in reg_clean12}))
            sig_lookup[name_c12] = {"tf": tf, "kind": "clean12"}

        # Clean SenMayo
        reg_clean_sen = [g for g in reg_in_expr if g not in senmayo]
        if len(reg_clean_sen) >= REGULON_MIN:
            name_sen = f"{tf}__clean_senmayo"
            sigs.append(GeneSignature(name=name_sen,
                                       gene2weight={g: 1.0 for g in reg_clean_sen}))
            sig_lookup[name_sen] = {"tf": tf, "kind": "clean_senmayo"}

        # Size-matched nulls: remove n_rm random genes where n_rm == |SASP12 overlap|
        n_sasp_overlap = sum(1 for g in reg_in_expr if g in set(SASP12))
        # If SASP-overlap is zero, we still do size-matched removal of 12 random
        # genes to characterize the null magnitude; clean12 will equal raw in that case
        n_rm = max(n_sasp_overlap, 12 if n_sasp_overlap == 0 else n_sasp_overlap)
        # Actually: brief says "matched to SASP12 removal count; if <12 SASP-overlap
        # exist, match that count". So n_rm = n_sasp_overlap when > 0, else... we
        # still need a baseline; use 12 as the standard matched count when SASP
        # overlap is 0 (this tests regulon-size sensitivity at the canonical count).
        if n_rm < 1:
            n_rm = 1
        if len(reg_in_expr) < n_rm + REGULON_MIN:
            # cannot do size-matched null — skip
            pass
        else:
            rng = np.random.default_rng(
                NULL_SEED + (abs(hash(f"{compartment}__{tf}__sm")) % (2**31))
            )
            for k in range(SIZE_MATCHED_N):
                idx_sel = rng.choice(len(reg_in_expr), size=n_rm, replace=False)
                mask = np.ones(len(reg_in_expr), dtype=bool)
                mask[idx_sel] = False
                reg_k = [reg_in_expr[i] for i in range(len(reg_in_expr)) if mask[i]]
                if len(reg_k) < REGULON_MIN:
                    continue
                nm = f"{tf}__smn_{k:03d}"
                sigs.append(GeneSignature(name=nm,
                                           gene2weight={g: 1.0 for g in reg_k}))
                sig_lookup[nm] = {"tf": tf, "kind": "smn", "k": k}

    log(f"  {compartment}: constructed {len(sigs)} GeneSignatures for AUCell")

    # ----- Run AUCell ONCE on all signatures (per-cell ranks computed once) -----
    log(f"  {compartment}: launching pyscenic AUCell on "
        f"{expr_df.shape[0]} cells x {expr_df.shape[1]} genes ...")
    t_auc = time.time()
    auc_mtx = score_signatures(expr_df, sigs)
    log(f"  {compartment}: AUCell done in {(time.time()-t_auc)/60:.2f} min; "
        f"matrix {auc_mtx.shape}")

    # Release expr_df memory
    expr_cells_index = expr_df.index
    # We still need per-cell mRNA for the TF panel — extract now
    tf_mrna = {}
    for tf in TF_PANEL:
        if tf in expr_df.columns:
            tf_mrna[tf] = expr_df[tf].to_numpy().astype(float)
    # Free expr_df
    del expr_df
    gc.collect()

    # ----- Reduce AUCell to donor means -----
    # auc_mtx index may match expr_cells_index; handle both
    auc_mtx = auc_mtx.reindex(expr_cells_index)
    auc_values = auc_mtx.to_numpy().astype(float)      # cells x sigs
    sig_names = auc_mtx.columns.to_list()

    # Map: sample -> cell index array (positional within obs)
    sample_vals = obs["sample"].to_numpy()
    donor_cell_idx = {d: np.where(sample_vals == d)[0] for d in donor_samples}

    # Per-signature per-donor mean -> n_donors x n_sigs
    n_sigs = auc_values.shape[1]
    donor_means = np.empty((n_donors, n_sigs), dtype=float)
    for i, d in enumerate(donor_samples):
        idx = donor_cell_idx[d]
        donor_means[i, :] = np.nanmean(auc_values[idx, :], axis=0)

    # ----- Compute per-TF raw/clean12/clean_senmayo rhos + size-matched null CIs -----
    # Build reverse lookup: tf -> (raw_col, c12_col, sen_col, [smn_cols])
    sig_to_col = {nm: k for k, nm in enumerate(sig_names)}
    tf_cols: dict[str, dict] = {tf: {"smn": []} for tf in TF_PANEL}
    for nm, info in sig_lookup.items():
        if nm not in sig_to_col:
            continue
        col = sig_to_col[nm]
        tf = info["tf"]
        if info["kind"] == "raw":
            tf_cols[tf]["raw"] = col
        elif info["kind"] == "clean12":
            tf_cols[tf]["clean12"] = col
        elif info["kind"] == "clean_senmayo":
            tf_cols[tf]["clean_senmayo"] = col
        elif info["kind"] == "smn":
            tf_cols[tf]["smn"].append(col)

    # Spearman rho with SASP12_mean phenotype (matches batch_054 convention)
    def spearman_pheno(vec, pheno=donor_sasp):
        ok = np.isfinite(vec) & np.isfinite(pheno)
        if ok.sum() < 4:
            return np.nan
        rho, _ = spearmanr(pheno[ok], vec[ok])
        return float(rho) if np.isfinite(rho) else np.nan

    results: list[dict] = []
    null_dists: dict[str, np.ndarray] = {}

    # For mRNA rho, aggregate per-donor
    for tf in TF_PANEL:
        row = {
            "compartment": compartment,
            "tf": tf,
            "n_donors": n_donors,
            "regulon_size_raw": len(regulons[tf]),
        }
        reg = regulons[tf]
        reg_in_expr = [g for g in reg if g in all_genes]
        sasp_in = [g for g in reg_in_expr if g in set(SASP12)]
        sen_in = [g for g in reg_in_expr if g in senmayo]
        row["regulon_size_after_sasp12"] = len(reg_in_expr) - len(sasp_in)
        row["regulon_size_after_senmayo"] = len(reg_in_expr) - len(sen_in)
        row["n_sasp_removed"] = len(sasp_in)
        row["n_senmayo_removed"] = len(sen_in)

        # AUCell rhos
        raw_rho = spearman_pheno(donor_means[:, tf_cols[tf]["raw"]]) if "raw" in tf_cols[tf] else np.nan
        c12_rho = spearman_pheno(donor_means[:, tf_cols[tf]["clean12"]]) if "clean12" in tf_cols[tf] else np.nan
        sen_rho = spearman_pheno(donor_means[:, tf_cols[tf]["clean_senmayo"]]) if "clean_senmayo" in tf_cols[tf] else np.nan
        row["raw_aucell_rho"] = raw_rho
        row["clean12_aucell_rho"] = c12_rho
        row["clean_senmayo_aucell_rho"] = sen_rho

        # mRNA rho
        if tf in tf_mrna:
            mrna_per_cell = tf_mrna[tf]
            mrna_donor = np.array([
                np.nanmean(mrna_per_cell[donor_cell_idx[d]]) for d in donor_samples
            ])
            mrna_rho = spearman_pheno(mrna_donor)
        else:
            mrna_rho = np.nan
        row["mrna_rho"] = mrna_rho

        row["delta_clean12"] = (c12_rho - mrna_rho) if (np.isfinite(c12_rho) and np.isfinite(mrna_rho)) else np.nan
        row["delta_clean_senmayo"] = (sen_rho - mrna_rho) if (np.isfinite(sen_rho) and np.isfinite(mrna_rho)) else np.nan

        # Size-matched null CIs — compute rho for each smn and derive 95% CI on rho AND on Delta
        smn_cols = tf_cols[tf]["smn"]
        if smn_cols:
            smn_rhos = np.array([spearman_pheno(donor_means[:, c]) for c in smn_cols])
            smn_rhos = smn_rhos[np.isfinite(smn_rhos)]
            if len(smn_rhos) >= 10 and np.isfinite(mrna_rho):
                smn_deltas = smn_rhos - mrna_rho
                row["size_matched_null_ci_low"] = float(np.percentile(smn_deltas, 2.5))
                row["size_matched_null_ci_high"] = float(np.percentile(smn_deltas, 97.5))
                row["size_matched_null_n"] = int(len(smn_rhos))
                row["size_matched_null_median_delta"] = float(np.median(smn_deltas))
            else:
                row["size_matched_null_ci_low"] = np.nan
                row["size_matched_null_ci_high"] = np.nan
                row["size_matched_null_n"] = int(len(smn_rhos))
                row["size_matched_null_median_delta"] = np.nan
        else:
            row["size_matched_null_ci_low"] = np.nan
            row["size_matched_null_ci_high"] = np.nan
            row["size_matched_null_n"] = 0
            row["size_matched_null_median_delta"] = np.nan

        # Age-shuffle null (Critic 2 permutation): 2000 shuffles of the phenotype
        # vector (SASP12_mean) per donor, recomputing rho(AUCell_clean12, pheno_perm).
        # This is what the brief calls "age-shuffle" -- since the response we
        # correlate against is SASP12_mean (which is tightly age-correlated), we
        # shuffle that directly. The null is permutation of the response label.
        if "clean12" in tf_cols[tf] and np.isfinite(c12_rho):
            rng = np.random.default_rng(
                NULL_SEED + (abs(hash(f"{compartment}__{tf}__pheno")) % (2**31))
            )
            d12 = donor_means[:, tf_cols[tf]["clean12"]]
            ok = np.isfinite(d12) & np.isfinite(donor_sasp)
            pheno_valid = donor_sasp[ok]
            d12_valid = d12[ok]
            if ok.sum() < 4:
                row["age_shuffle_p_empirical"] = np.nan
                null_dists[f"{compartment}__{tf}"] = np.array([])
            else:
                null_rhos = np.empty(AGE_SHUFFLE_N, dtype=float)
                for i in range(AGE_SHUFFLE_N):
                    perm = rng.permutation(len(pheno_valid))
                    r, _ = spearmanr(pheno_valid[perm], d12_valid)
                    null_rhos[i] = r if np.isfinite(r) else 0.0
                p_emp = float((np.abs(null_rhos) >= abs(c12_rho)).mean())
                row["age_shuffle_p_empirical"] = p_emp
                null_dists[f"{compartment}__{tf}"] = null_rhos
        else:
            row["age_shuffle_p_empirical"] = np.nan
            null_dists[f"{compartment}__{tf}"] = np.array([])

        # Fisher-z CI on clean12 rho
        lo, hi = fisher_z_ci(c12_rho, n_donors)
        row["fisher_z_ci_low"] = lo
        row["fisher_z_ci_high"] = hi

        # Fisher-z CI on mRNA rho (for bystander non-overlap check)
        mlo, mhi = fisher_z_ci(mrna_rho, n_donors)
        row["mrna_fz_ci_low"] = mlo
        row["mrna_fz_ci_high"] = mhi

        results.append(row)

    elapsed = (time.time() - t0) / 60
    log(f"  {compartment}: analysis complete in {elapsed:.1f} min")

    # Free compartment-level big arrays
    del auc_mtx, auc_values, donor_means, tf_mrna, obs
    gc.collect()

    return results, null_dists


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_rows(rows: list[dict], null_dists: dict[str, np.ndarray]
                  ) -> tuple[list[dict], float, float]:
    """Apply empirical-null-derived thresholds to classify each (TF x compartment).

    t_driver = 95th percentile of |rho| pooled across all age-shuffle nulls.
    t_bystander = 5th percentile of |rho| pooled across all age-shuffle nulls.
    """
    all_null = [np.abs(v) for v in null_dists.values() if len(v) > 0]
    if all_null:
        pooled = np.concatenate(all_null)
        t_driver = float(np.percentile(pooled, 95))
        t_bystander = float(np.percentile(pooled, 5))
    else:
        t_driver = np.nan
        t_bystander = np.nan

    log(f"Empirical thresholds from pooled age-shuffle null:")
    log(f"  t_driver (95th pct of |rho|) = {t_driver:.4f}")
    log(f"  t_bystander (5th pct of |rho|) = {t_bystander:.4f}")

    for row in rows:
        flags: list[str] = []
        n = row["n_donors"]
        clean12 = row.get("clean12_aucell_rho", np.nan)
        clean_sen = row.get("clean_senmayo_aucell_rho", np.nan)
        mrna = row.get("mrna_rho", np.nan)
        delta12 = row.get("delta_clean12", np.nan)
        p_age = row.get("age_shuffle_p_empirical", np.nan)
        sm_lo = row.get("size_matched_null_ci_low", np.nan)
        sm_hi = row.get("size_matched_null_ci_high", np.nan)
        fz_lo = row.get("fisher_z_ci_low", np.nan)
        fz_hi = row.get("fisher_z_ci_high", np.nan)
        m_fz_lo = row.get("mrna_fz_ci_low", np.nan)
        m_fz_hi = row.get("mrna_fz_ci_high", np.nan)

        # Underpowered flags
        reg_after_sasp = row.get("regulon_size_after_sasp12", 0)
        reg_after_sen = row.get("regulon_size_after_senmayo", 0)
        underpowered = False
        if reg_after_sasp < 20 or reg_after_sen < 20:
            underpowered = True
            flags.append(f"regulon_too_small(sasp={reg_after_sasp},sen={reg_after_sen})")
        if n <= 16 and np.isfinite(fz_lo) and np.isfinite(fz_hi):
            if (fz_lo <= t_driver <= fz_hi) or (fz_lo <= t_bystander <= fz_hi):
                underpowered = True
                flags.append("fisher_z_ci_crosses_threshold_at_low_N")

        classification = "inconclusive"

        # Driver
        driver_ok = (
            np.isfinite(clean12) and np.isfinite(clean_sen) and
            np.isfinite(p_age) and np.isfinite(delta12) and
            np.isfinite(sm_lo) and np.isfinite(sm_hi)
        )
        if driver_ok:
            sign_agree = (np.sign(clean12) == np.sign(clean_sen)) and clean12 != 0
            both_strong = (abs(clean12) >= t_driver) and (abs(clean_sen) >= t_driver)
            bonf_pass = p_age <= BONFERRONI_ALPHA
            # Brief §Classification tree line 98: "clean12 Δ INSIDE the size-matched
            # control null 95% CI" -- i.e., SASP12 removal effect is statistically
            # indistinguishable from any-12-gene random removal, meaning the regulon
            # is a balanced driver target set (not a SASP-dominated leakage set).
            # (Note: brief line 92 says "Δ OUTSIDE this null CI" to count as
            # SASP-SPECIFIC -- that's the bystander-like direction; a true DRIVER's
            # SASP12 removal has the SAME magnitude as random removal = Δ INSIDE CI.)
            delta_inside = (sm_lo <= delta12 <= sm_hi)
            fz_above_half = (
                np.isfinite(fz_lo) and np.isfinite(fz_hi) and
                ((clean12 > 0 and fz_lo >= FISHER_CI_DRIVER_THRESH) or
                 (clean12 < 0 and fz_hi <= -FISHER_CI_DRIVER_THRESH))
            )
            if sign_agree and both_strong and bonf_pass and delta_inside and fz_above_half:
                classification = "driver"

        # Bystander
        if classification == "inconclusive" and np.isfinite(mrna) and np.isfinite(clean12):
            mrna_strong = abs(mrna) >= t_driver
            clean12_weak = abs(clean12) <= t_bystander
            if (np.isfinite(fz_lo) and np.isfinite(fz_hi) and
                np.isfinite(m_fz_lo) and np.isfinite(m_fz_hi)):
                non_overlap = (fz_hi < m_fz_lo) or (m_fz_hi < fz_lo)
            else:
                non_overlap = False
            if mrna_strong and clean12_weak and non_overlap:
                classification = "bystander"

        # Polarity-flip
        if classification == "inconclusive" and np.isfinite(mrna) and np.isfinite(clean12):
            opp = (np.sign(mrna) != np.sign(clean12)) and mrna != 0 and clean12 != 0
            both_big = (abs(mrna) >= 0.3) and (abs(clean12) >= 0.3)
            clean_excl_0 = (
                np.isfinite(fz_lo) and np.isfinite(fz_hi) and
                (fz_lo > 0 or fz_hi < 0)
            )
            mrna_excl_0 = (
                np.isfinite(m_fz_lo) and np.isfinite(m_fz_hi) and
                (m_fz_lo > 0 or m_fz_hi < 0)
            )
            if opp and both_big and clean_excl_0 and mrna_excl_0:
                classification = "polarity-flip"

        row["flag_underpowered"] = bool(underpowered)
        row["flag_note"] = "; ".join(flags)

        if underpowered and classification == "inconclusive":
            classification = "UNDERPOWERED"
        row["classification"] = classification

    return rows, t_driver, t_bystander


# ---------------------------------------------------------------------------
# Validation against d1_correlations_all.csv
# ---------------------------------------------------------------------------
def validate_against_prior(rows: list[dict]) -> dict:
    if not VALIDATION_CSV.exists():
        return {"status": "validation_csv_missing"}
    prior = pd.read_csv(VALIDATION_CSV)
    checks = []
    for row in rows:
        comp, tf = row["compartment"], row["tf"]
        if comp not in ("HLMA_Vascular", "HLMA_MuSC"):
            continue
        p = prior[(prior["dataset"] == comp) & (prior["tf"] == tf)]
        if len(p) == 0:
            continue
        prior_rho = float(p["aucell_rho"].iloc[0])
        our_rho = row.get("raw_aucell_rho", np.nan)
        if np.isfinite(our_rho) and np.isfinite(prior_rho):
            checks.append({"compartment": comp, "tf": tf,
                           "prior_rho": prior_rho, "our_rho": our_rho,
                           "abs_diff": float(abs(our_rho - prior_rho))})
    n_close = sum(1 for c in checks if c["abs_diff"] <= 0.05)
    n_loose = sum(1 for c in checks if c["abs_diff"] <= 0.15)
    n_total = len(checks)
    log(f"Validation vs d1_correlations_all.csv: {n_close}/{n_total} within 0.05, "
        f"{n_loose}/{n_total} within 0.15")
    for c in checks[:8]:
        log(f"  {c['compartment']} {c['tf']}: prior={c['prior_rho']:+.3f} "
            f"ours={c['our_rho']:+.3f} diff={c['abs_diff']:.3f}")
    return {"n_checked": n_total,
            "n_within_0.05": n_close,
            "n_within_0.15": n_loose,
            "sample_checks": checks[:12]}


# ---------------------------------------------------------------------------
# Regulon-size covariate
# ---------------------------------------------------------------------------
def regulon_size_covariate(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["delta_clean12", "regulon_size_after_sasp12"])
    if len(df) < 5:
        return {"n": int(len(df)), "rho": None, "p": None}
    rho, p = spearmanr(df["delta_clean12"], df["regulon_size_after_sasp12"])
    log(f"Regulon-size covariate: spearman(delta_clean12, regulon_size_after_sasp12) "
        f"= {rho:+.3f} (p={p:.4f}, n={len(df)})")
    return {"n": int(len(df)), "rho": float(rho), "p": float(p)}


# ---------------------------------------------------------------------------
# JSON-safe helpers
# ---------------------------------------------------------------------------
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main() -> None:
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    log("=== batch_063 Analysis B: Hardened Driver-Bystander Delta-Rule ===")
    log(f"Python: {sys.version.split()[0]}")
    try:
        import pyscenic
        log(f"pyscenic: {pyscenic.__version__}")
    except Exception as e:
        log(f"pyscenic import failed: {e}")
        raise
    log(f"numpy: {np.__version__}, pandas: {pd.__version__}")
    log(f"TF panel: {TF_PANEL}")
    log(f"SASP12: {SASP12}")

    if SENMAYO_TXT.exists():
        with open(SENMAYO_TXT) as f:
            senmayo = set(line.strip() for line in f if line.strip())
        log(f"SenMayo loaded: {len(senmayo)} genes")
    else:
        senmayo = set()
        log("SenMayo file MISSING - running with SASP12 correction only (SKIPPED SenMayo)")

    adj = {
        "HLMA_Vascular": pd.read_csv(ADJ_VASC),
        "HLMA_MuSC": pd.read_csv(ADJ_MUSC),
        "HLMA_FAP": pd.read_csv(ADJ_FAP),
    }
    for k, v in adj.items():
        log(f"  {k} adjacencies: {len(v)} rows, {v['TF'].nunique()} TFs")

    all_rows: list[dict] = []
    all_nulls: dict[str, np.ndarray] = {}

    for comp in ["HLMA_Vascular", "HLMA_MuSC", "HLMA_FAP"]:
        rows, nulls = analyze_compartment(comp, adj[comp], senmayo)
        all_rows.extend(rows)
        all_nulls.update(nulls)
        gc.collect()

    log("\n--- Applying classification ---")
    all_rows, t_driver, t_bystander = classify_rows(all_rows, all_nulls)

    validation = validate_against_prior(all_rows)
    covar = regulon_size_covariate(all_rows)

    cls_df = pd.DataFrame(all_rows)
    keep_cols = [
        "compartment", "tf", "n_donors",
        "regulon_size_raw", "regulon_size_after_sasp12", "regulon_size_after_senmayo",
        "n_sasp_removed", "n_senmayo_removed",
        "raw_aucell_rho", "clean12_aucell_rho", "clean_senmayo_aucell_rho",
        "mrna_rho", "delta_clean12", "delta_clean_senmayo",
        "size_matched_null_ci_low", "size_matched_null_ci_high",
        "size_matched_null_n", "size_matched_null_median_delta",
        "age_shuffle_p_empirical",
        "fisher_z_ci_low", "fisher_z_ci_high",
        "mrna_fz_ci_low", "mrna_fz_ci_high",
        "classification", "flag_underpowered", "flag_note",
    ]
    cls_df = cls_df[[c for c in keep_cols if c in cls_df.columns]]
    out_csv = OUTDIR / "b_delta_classification.csv"
    cls_df.to_csv(out_csv, index=False)
    log(f"Wrote {out_csv} ({len(cls_df)} rows)")

    out_npz = OUTDIR / "b_delta_null_distributions.npz"
    np.savez(out_npz, **{k: v for k, v in all_nulls.items() if len(v) > 0})
    log(f"Wrote {out_npz} ({len([k for k,v in all_nulls.items() if len(v)>0])} null arrays)")

    # Summary JSON
    by_key = {(r["compartment"], r["tf"]): r for r in all_rows}

    def get_c(comp: str, tf: str) -> str:
        r = by_key.get((comp, tf))
        return r["classification"] if r else "missing"

    preds = {
        "JUNB_vasc_driver_after_senmayo": {
            "prediction": "JUNB retains driver class in Vascular after SenMayo removal",
            "Vascular_classification": get_c("HLMA_Vascular", "JUNB"),
            "Vascular_clean12_rho": by_key.get(("HLMA_Vascular", "JUNB"), {}).get("clean12_aucell_rho"),
            "Vascular_clean_senmayo_rho": by_key.get(("HLMA_Vascular", "JUNB"), {}).get("clean_senmayo_aucell_rho"),
            "MuSC_classification": get_c("HLMA_MuSC", "JUNB"),
            "FAP_classification": get_c("HLMA_FAP", "JUNB"),
        },
        "KLF10_bystander_all_3": {
            "prediction": "KLF10 classifies as bystander in ALL 3 compartments",
            "Vascular": get_c("HLMA_Vascular", "KLF10"),
            "MuSC": get_c("HLMA_MuSC", "KLF10"),
            "FAP": get_c("HLMA_FAP", "KLF10"),
        },
        "CEBPB_fap_driver": {
            "prediction": "CEBPB classifies as driver in FAP",
            "FAP": get_c("HLMA_FAP", "CEBPB"),
            "Vascular": get_c("HLMA_Vascular", "CEBPB"),
            "MuSC": get_c("HLMA_MuSC", "CEBPB"),
        },
        "ATF3_novel_call": {
            "prediction": "ATF3 driver status in any compartment is a NOVEL finding",
            "Vascular": get_c("HLMA_Vascular", "ATF3"),
            "MuSC": get_c("HLMA_MuSC", "ATF3"),
            "FAP": get_c("HLMA_FAP", "ATF3"),
        },
        "STAT3_novel_call": {
            "prediction": "STAT3 driver status in any compartment is a NOVEL finding",
            "Vascular": get_c("HLMA_Vascular", "STAT3"),
            "MuSC": get_c("HLMA_MuSC", "STAT3"),
            "FAP": get_c("HLMA_FAP", "STAT3"),
        },
    }

    drivers = [r for r in all_rows if r["classification"] == "driver"]
    bystanders = [r for r in all_rows if r["classification"] == "bystander"]
    flips = [r for r in all_rows if r["classification"] == "polarity-flip"]
    inconc = [r for r in all_rows if r["classification"] == "inconclusive"]
    underp_only = [r for r in all_rows if r["classification"] == "UNDERPOWERED"]
    flagged = [r for r in all_rows if r["flag_underpowered"]]

    summary = {
        "batch": "batch_063",
        "analysis": "B_driver_bystander_delta_hardened",
        "date": ts(),
        "seeds": {"null_seed": NULL_SEED, "aucell_seed": AUCELL_SEED},
        "parameters": {
            "regulon_quantile": REGULON_QUANTILE,
            "regulon_min_targets": REGULON_MIN,
            "regulon_max_targets": REGULON_MAX,
            "min_cells_per_donor_vasc_musc": MIN_CELLS_PER_DONOR_VASC_MUSC,
            "min_cells_per_donor_fap": MIN_CELLS_PER_DONOR_DEFAULT,
            "aucell_threshold": AUCELL_THRESHOLD,
            "size_matched_null_n": SIZE_MATCHED_N,
            "age_shuffle_n": AGE_SHUFFLE_N,
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "fisher_ci_driver_thresh": FISHER_CI_DRIVER_THRESH,
            "sasp12_count": len(SASP12),
            "senmayo_count": len(senmayo),
            "tf_panel_count": len(TF_PANEL),
        },
        "empirical_thresholds": {
            "t_driver_from_age_shuffle_95pct": t_driver,
            "t_bystander_from_age_shuffle_5pct": t_bystander,
        },
        "validation_vs_batch054": validation,
        "regulon_size_covariate": covar,
        "counts": {
            "driver": len(drivers),
            "bystander": len(bystanders),
            "polarity_flip": len(flips),
            "inconclusive": len(inconc),
            "underpowered_label": len(underp_only),
            "flagged_underpowered_any": len(flagged),
        },
        "driver_TFs_by_compartment": {
            comp: [r["tf"] for r in drivers if r["compartment"] == comp]
            for comp in ["HLMA_Vascular", "HLMA_MuSC", "HLMA_FAP"]
        },
        "bystander_TFs_by_compartment": {
            comp: [r["tf"] for r in bystanders if r["compartment"] == comp]
            for comp in ["HLMA_Vascular", "HLMA_MuSC", "HLMA_FAP"]
        },
        "pre_registered_predictions": preds,
    }

    for tf in ["JUNB", "KLF10", "CEBPB", "ATF3", "STAT3"]:
        entries = {}
        for comp in ["HLMA_Vascular", "HLMA_MuSC", "HLMA_FAP"]:
            r = by_key.get((comp, tf))
            if r:
                entries[comp] = {
                    "regulon_size_raw": r["regulon_size_raw"],
                    "regulon_size_after_sasp12": r["regulon_size_after_sasp12"],
                    "regulon_size_after_senmayo": r["regulon_size_after_senmayo"],
                    "raw_aucell_rho": r["raw_aucell_rho"],
                    "clean12_aucell_rho": r["clean12_aucell_rho"],
                    "clean_senmayo_aucell_rho": r["clean_senmayo_aucell_rho"],
                    "mrna_rho": r["mrna_rho"],
                    "delta_clean12": r["delta_clean12"],
                    "age_shuffle_p_empirical": r["age_shuffle_p_empirical"],
                    "fisher_z_ci": [r["fisher_z_ci_low"], r["fisher_z_ci_high"]],
                    "classification": r["classification"],
                    "flag_underpowered": r["flag_underpowered"],
                    "flag_note": r["flag_note"],
                }
        summary.setdefault("tf_of_interest", {})[tf] = entries

    out_json = OUTDIR / "b_delta_summary.json"
    with open(out_json, "w") as f:
        json.dump(clean_for_json(summary), f, indent=2, default=str)
    log(f"Wrote {out_json}")

    log("=== Analysis B complete ===")
    log(f"Counts: driver={len(drivers)}, bystander={len(bystanders)}, "
        f"polarity-flip={len(flips)}, inconclusive={len(inconc)}, "
        f"UNDERPOWERED={len(underp_only)}, any_flagged={len(flagged)}")


if __name__ == "__main__":
    main()
