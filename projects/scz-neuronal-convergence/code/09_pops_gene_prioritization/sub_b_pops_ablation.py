#!/usr/bin/env python3
"""batch_058 Sub-B v2.1 — F-056-A mechanism criterion 5 via PoPS ablation +
tail-ρ + rank-partial.

Implements brief_v2.md §Sub-B v2.1 EXACTLY.

Pre-steps:
  P0. Load + categorize all PoPS features (committed to
      `scripts/pops_feature_categories.tsv`).
  P1. Shapiro-Wilk on MAGMA_Z_SCZ; if p<0.05, report + use Gaussianized
      MAGMA_Z for B.2/B.3 tail and rank comparisons.
  P2. Reproduction gate R2: raw Pearson ρ(PoPS, MAGMA_Z_SCZ) ∈ [0.495, 0.535].

B.1 CATEGORY ABLATION:
  - Load PoPS coefs (~17,427 features) and features matrix.
  - For each category k ∈ {expression, ppi, pathway, other}:
      PoPS_score_i[−k] = Σ_{j ∉ C_k} β_j X_ij
      Δρ_k = partial_ρ(PoPS_full, MAGMA | pLI) − partial_ρ(PoPS[−k], MAGMA | pLI)
  - Permutation null (n=1000, seed=20260424) preserving category counts.
  - Variance-normalization sanity (descriptive).
  - Decision: CATEGORY_DRIVES_k if observed Δρ_k exceeds 95th pct AND ≥ 0.10.

B.2 TAIL-RESTRICTED: upper/middle/lower tails; compare upper vs LS-MC null.
B.3 RANK-TRANSFORMED: rank-Gaussianized partial ρ vs raw partial.

Aggregate (first-match): 2A_CATEGORY_DRIVES_k → 2B_TAIL_ENRICHED →
  2C_DISTRIBUTION_SENSITIVE → MECHANISM_NULL → INTERMEDIATE → UNINTERPRETABLE.

Bootstrap n=1000, seed=20260424 for all primary CIs.

Outputs:
  output/sub_b/results.json
  output/sub_b/permutation_null.npz  (arrays of Δρ*_k per draw)
  output/sub_b/reconstruction_check.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BATCH054_P05_PREDS,
    BH_Q,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    POPS_COEFS_P05,
    POPS_FEATURES_MUNGED_DIR,
    REPRO_R2_RHO_HI,
    REPRO_R2_RHO_LO,
    SHAPIRO_ALPHA,
    SUB_B_BOOT_N,
    SUB_B_CATEGORY_DELTA_FLOOR,
    SUB_B_DISTRIBUTION_DIFF,
    SUB_B_LS_MC_N,
    SUB_B_LS_NULL_RHO,
    SUB_B_LS_NULL_TAU,
    SUB_B_OTHER_MASS_GATE,
    SUB_B_PERM_N,
    SUB_B_PERM_PVALUE,
    SUB_B_SEED,
    SUB_B_TAIL_ENRICHED_DIFF,
    atomic_write_json,
    bh_fdr,
    build_bootstrap_idx,
    categorize_pops_features,
    classify_feature,
    load_gnomad_per_brief_v2,
    load_magma_scz,
    load_pops_coefs,
    load_pops_features_matrix,
    load_preds,
    longin_solnik_mc_null,
    partial_pearson,
    percentile_ci,
    rank_gaussianize,
    setup_logger,
    sha256_file,
)

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, shapiro


def _ci(arr: np.ndarray) -> dict:
    """Return point, ci_lo, ci_hi dict from a bootstrap array."""
    arr = np.asarray(arr, dtype=float)
    return {
        "point": float(np.mean(arr)),
        "ci_lo": float(np.quantile(arr, 0.025)),
        "ci_hi": float(np.quantile(arr, 0.975)),
        "std": float(np.std(arr, ddof=1)),
    }


def build_shared_frame(smoke_subset: int = 0, logger=None
                        ) -> tuple[pd.DataFrame, dict]:
    """Build the shared (N, 4) frame: ENSGID, PoPS_Score, MAGMA_Z, lof_pLI.

    N expected ~17,459 per brief. We merge PoPS preds p=0.05 × MAGMA SCZ ×
    gnomAD per-brief and drop rows missing any of the three covariates.
    """
    preds = load_preds(BATCH054_P05_PREDS)
    magma = load_magma_scz()[["ENSGID", "MAGMA_Z"]]
    gnomad = load_gnomad_per_brief_v2()[["ENSGID", "lof_pLI"]]
    frame = (
        preds.merge(magma, on="ENSGID", how="inner")
             .merge(gnomad, on="ENSGID", how="inner")
             .dropna(subset=["PoPS_Score", "MAGMA_Z", "lof_pLI"])
             .drop_duplicates(subset="ENSGID", keep="first")
             .sort_values("ENSGID")
             .reset_index(drop=True)
    )
    if smoke_subset > 0:
        frame = frame.iloc[:smoke_subset].reset_index(drop=True)
    if logger is not None:
        logger.info("shared frame: N=%d (smoke_subset=%d)", len(frame),
                     smoke_subset)
    info = {"N": int(len(frame)), "smoke_subset": int(smoke_subset)}
    return frame, info


def load_coefs_and_features(cols_from_munged: set[str], logger
                              ) -> tuple[pd.Series, np.ndarray,
                                           list[str], list[str]]:
    """Load PoPS coefs and feature matrix restricted to coefs ∩ munged cols.

    Returns (coefs_series, X_matrix, gene_ensgids_rows, feature_names_cols).
    coefs_series and feature_names_cols are aligned 1:1.
    """
    coefs = load_pops_coefs(POPS_COEFS_P05)
    logger.info("PoPS coefs parsed: n_features=%d", len(coefs))
    features_in_both = [f for f in coefs.index if f in cols_from_munged]
    logger.info("Features in coefs ∩ munged cols: %d (coefs=%d, munged=%d)",
                 len(features_in_both), len(coefs), len(cols_from_munged))
    # Load only features present in both sides.
    X, gene_ensgids, kept_names = load_pops_features_matrix(
        POPS_FEATURES_MUNGED_DIR, cols_to_load=set(features_in_both),
    )
    logger.info("Loaded feature matrix X: shape=%s", X.shape)
    # Align coefs to loaded feature column order.
    coefs_aligned = coefs.reindex(kept_names)
    if coefs_aligned.isna().any():
        raise RuntimeError(
            "Coefs alignment failure — some kept_names have no coef."
        )
    return coefs_aligned, X, gene_ensgids, kept_names


def reconstruct_pops_scores(X: np.ndarray, coefs: np.ndarray,
                              gene_ensgids: list[str]) -> pd.Series:
    """Compute PoPS_score_i = Σ_j β_j · X_ij for each gene (row).

    Returns Series indexed by gene ENSGID; float64 for bootstrap stability.
    """
    scores = X.astype(np.float64) @ coefs.astype(np.float64)
    return pd.Series(scores, index=gene_ensgids, name="PoPS_score_recon")


def run_P0_category_map(feature_names: list[str], logger) -> dict:
    """P0: classify each of the loaded feature names into 4 categories.

    Writes/diffs `scripts/pops_feature_categories.tsv` (handled inside
    categorize_pops_features — v2.1 FIX #9: compute in memory + SHA256 diff
    vs committed file, warn on drift, no rewrite). Returns summary dict with
    counts + shares over the FULL munged (57,742) universe.

    IMPORTANT: this summary is the FULL-universe view (before coefs∩munged
    restriction). For the gate on `other` mass, the caller must also compute
    the loaded-subset version — see compute_other_mass_gates().
    """
    cat_map = categorize_pops_features(feature_names, logger=logger)
    counts: dict[str, int] = {"expression": 0, "ppi": 0,
                               "pathway": 0, "other": 0}
    for c in cat_map.values():
        counts[c] += 1
    total = sum(counts.values())
    shares = {k: (v / total if total else 0.0) for k, v in counts.items()}
    logger.info("P0 categories (FULL munged universe): %s (shares=%s)",
                 counts, shares)
    return {
        "counts": counts,
        "shares": shares,
        "total_features": total,
        "cat_map_size": len(cat_map),
    }


def compute_other_mass_gates(loaded_names: list[str],
                              cat_map_loaded: dict[str, str],
                              coefs: np.ndarray, X: np.ndarray,
                              logger) -> dict:
    """v2.1 FIX #1: compute BOTH other-mass fractions on the LOADED subset.

    Quality-reviewer BLOCKING fix: the gate must be evaluated over the
    ~17,427 features actually loaded (coefs ∩ munged), not the 57,742 full
    munged universe (those numbers are informational only).

    Two metrics:
      other_mass_fraction   = #{j ∈ other ∩ loaded} / |loaded|
      other_mass_weighted   = Σ_{j∈other} |β_j|·mean_i(|X_ij|)
                              / Σ_j      |β_j|·mean_i(|X_ij|)

    WHY the weighted metric: even if `other` is a small fraction of the
    feature count, it could carry disproportionate coefficient × feature-value
    mass (e.g., one high-|β| feature drives variance). Gating on EITHER metric
    catches both cases.

    Gate fires if either metric > SUB_B_OTHER_MASS_GATE (0.15).
    """
    cats = np.array([cat_map_loaded[n] for n in loaded_names])
    n_loaded = len(loaded_names)
    counts_loaded = {k: int((cats == k).sum())
                      for k in ("expression", "ppi", "pathway", "other")}
    other_mass_fraction = (counts_loaded["other"] / n_loaded
                            if n_loaded else 0.0)

    # Weighted mass: |β_j| · mean_i(|X_ij|) per column, sum over category.
    # We compute on float64 to avoid float32 rounding in sum.
    abs_coefs = np.abs(coefs.astype(np.float64))
    # mean over rows of |X_ij| for each column j.
    abs_X_col_mean = np.mean(np.abs(X.astype(np.float64)), axis=0)
    per_col_mass = abs_coefs * abs_X_col_mean
    total_mass = float(per_col_mass.sum())
    other_mask = (cats == "other")
    other_mass_sum = float(per_col_mass[other_mask].sum())
    other_mass_weighted = (other_mass_sum / total_mass
                            if total_mass > 0 else 0.0)

    mass_by_cat = {}
    for k in ("expression", "ppi", "pathway", "other"):
        mass_by_cat[k] = float(per_col_mass[cats == k].sum()
                                / total_mass if total_mass > 0 else 0.0)

    gate_fires_fraction = bool(other_mass_fraction
                                > SUB_B_OTHER_MASS_GATE)
    gate_fires_weighted = bool(other_mass_weighted
                                > SUB_B_OTHER_MASS_GATE)
    gate_fires_any = bool(gate_fires_fraction or gate_fires_weighted)
    logger.info("LOADED-subset other gates: fraction=%.4f weighted=%.4f "
                 "gate=%s (fraction-gate=%s, weighted-gate=%s)",
                 other_mass_fraction, other_mass_weighted,
                 gate_fires_any, gate_fires_fraction, gate_fires_weighted)
    return {
        "n_loaded": n_loaded,
        "counts_loaded": counts_loaded,
        "other_mass_fraction": float(other_mass_fraction),
        "other_mass_weighted": float(other_mass_weighted),
        "mass_share_by_category": mass_by_cat,
        "gate_threshold": SUB_B_OTHER_MASS_GATE,
        "gate_fires_fraction": gate_fires_fraction,
        "gate_fires_weighted": gate_fires_weighted,
        "gate_fires": gate_fires_any,
    }


def run_P1_shapiro(y: np.ndarray, logger) -> dict:
    """P1: Shapiro-Wilk on MAGMA_Z_SCZ. Use subsample if n > 5000 (scipy cap).

    WHY subsample: scipy.stats.shapiro has an upper n of ~5000 due to
    accuracy of its p-value tabulation. We draw a seeded subsample if n is
    larger, per C3 CRITICAL #3 Shapiro requirement.
    """
    n = int(y.shape[0])
    rng = np.random.default_rng(SUB_B_SEED)
    if n > 5000:
        idx = rng.choice(n, size=5000, replace=False)
        y_test = y[idx]
        used_subsample = True
    else:
        y_test = y
        used_subsample = False
    stat, pval = shapiro(y_test)
    passes_normality = bool(pval >= SHAPIRO_ALPHA)
    logger.info("P1 Shapiro: W=%.4f p=%.4e (normal=%s; subsampled=%s)",
                 stat, pval, passes_normality, used_subsample)
    return {
        "W": float(stat),
        "p": float(pval),
        "n_tested": int(y_test.shape[0]),
        "used_subsample": used_subsample,
        "alpha": SHAPIRO_ALPHA,
        "is_normal_at_alpha": passes_normality,
    }


def run_P2_R2_gate(pops: np.ndarray, magma: np.ndarray, logger) -> dict:
    """P2: raw Pearson ρ(PoPS, MAGMA) ∈ [0.495, 0.535] reproduction gate."""
    rho, _ = pearsonr(pops, magma)
    passes = bool(REPRO_R2_RHO_LO <= rho <= REPRO_R2_RHO_HI)
    logger.info("P2 R2 raw ρ(PoPS, MAGMA)=%.4f pass=%s", rho, passes)
    return {
        "raw_pearson_rho": float(rho),
        "target_lo": REPRO_R2_RHO_LO,
        "target_hi": REPRO_R2_RHO_HI,
        "pass": passes,
    }


def compute_ablated_score(X: np.ndarray, coefs: np.ndarray,
                            feature_cats: list[str], kcat: str) -> np.ndarray:
    """PoPS_score[−k]_i = Σ_{j : cat_j != kcat} β_j X_ij.

    Implemented by zeroing out coefs for category k (vectorized).
    """
    mask = np.array([c != kcat for c in feature_cats], dtype=bool)
    coefs_masked = coefs.copy()
    coefs_masked[~mask] = 0.0
    return (X.astype(np.float64) @ coefs_masked.astype(np.float64))


def partial_rho_paired_bootstrap(score_full: np.ndarray,
                                   score_ablated: np.ndarray,
                                   magma: np.ndarray, pli: np.ndarray,
                                   idx_mat: np.ndarray) -> tuple[np.ndarray,
                                                                    np.ndarray,
                                                                    np.ndarray]:
    """Paired bootstrap of partial_ρ(full|pLI), partial_ρ(ablated|pLI), Δρ.

    idx_mat: (B, N) integer indices.

    Returns (partial_full, partial_ablated, delta) arrays of length B.
    """
    B = idx_mat.shape[0]
    partial_full = np.zeros(B, dtype=float)
    partial_ablated = np.zeros(B, dtype=float)
    delta = np.zeros(B, dtype=float)
    for b in range(B):
        idx = idx_mat[b]
        p_full_i = score_full[idx]
        p_ab_i = score_ablated[idx]
        m_i = magma[idx]
        c_i = pli[idx]
        covs = c_i.reshape(-1, 1)
        rho_f = partial_pearson(p_full_i, m_i, covs)
        rho_a = partial_pearson(p_ab_i, m_i, covs)
        partial_full[b] = rho_f
        partial_ablated[b] = rho_a
        delta[b] = rho_f - rho_a
    return partial_full, partial_ablated, delta


def run_B1_category_ablation(X: np.ndarray, coefs: np.ndarray,
                               feature_names: list[str],
                               cat_map: dict[str, str],
                               pops_full: np.ndarray, magma: np.ndarray,
                               pli: np.ndarray, logger,
                               n_boot: int = SUB_B_BOOT_N,
                               n_perm: int = SUB_B_PERM_N,
                               seed: int = SUB_B_SEED) -> dict:
    """B.1 category ablation with permutation null.

    Returns dict with observed Δρ per category, bootstrap CI, and
    permutation-null distribution + p_perm + pass flag.
    """
    cats_list = [cat_map[n] for n in feature_names]
    categories = ["expression", "ppi", "pathway", "other"]
    n_genes = X.shape[0]

    # Compute ablated scores per category.
    ablated_scores: dict[str, np.ndarray] = {}
    for k in categories:
        ablated_scores[k] = compute_ablated_score(X, coefs, cats_list, k)

    # Full-sample partial ρ (reference). We recompute since `pops_full` is
    # coming in pre-computed but may have rounding drift vs coefs @ X.
    cov_pli = pli.reshape(-1, 1)
    partial_full_point = partial_pearson(pops_full, magma, cov_pli)
    observed_delta: dict[str, float] = {}
    observed_partial_ablated: dict[str, float] = {}
    for k in categories:
        pa = partial_pearson(ablated_scores[k], magma, cov_pli)
        observed_partial_ablated[k] = float(pa)
        observed_delta[k] = float(partial_full_point - pa)

    logger.info("B.1 observed partial_full=%.4f; Δρ=%s",
                 partial_full_point, observed_delta)

    # Variance-normalization sanity (descriptive).
    sd_full = float(np.std(pops_full, ddof=1))
    variance_norm: dict[str, float] = {}
    variance_norm_delta: dict[str, float] = {}
    for k in categories:
        sd_ab = float(np.std(ablated_scores[k], ddof=1))
        if sd_ab > 0:
            scaled = ablated_scores[k] * (sd_full / sd_ab)
            pa_norm = partial_pearson(scaled, magma, cov_pli)
        else:
            pa_norm = float("nan")
        variance_norm[k] = pa_norm
        variance_norm_delta[k] = partial_full_point - pa_norm

    # Paired bootstrap CI for observed Δρ.
    idx_mat = build_bootstrap_idx(n_genes, n_boot, seed)
    boot_results: dict[str, dict] = {}
    for k in categories:
        pf, pa, d = partial_rho_paired_bootstrap(
            pops_full, ablated_scores[k], magma, pli, idx_mat,
        )
        boot_results[k] = {
            "partial_full": _ci(pf),
            "partial_ablated": _ci(pa),
            "delta": _ci(d),
        }

    # PERMUTATION NULL preserving category counts.
    # For each draw, we randomly reassign category labels (preserving counts)
    # and re-compute Δρ_k per category.
    rng_perm = np.random.default_rng(seed)
    counts = {k: int(sum(1 for c in cats_list if c == k))
              for k in categories}
    logger.info("B.1 category counts: %s; running %d permutations",
                 counts, n_perm)
    n_feat = len(feature_names)
    base_labels = np.array(
        ["expression"] * counts["expression"]
        + ["ppi"] * counts["ppi"]
        + ["pathway"] * counts["pathway"]
        + ["other"] * counts["other"],
    )
    if base_labels.shape[0] != n_feat:
        raise RuntimeError(
            f"Permutation label count mismatch: {base_labels.shape[0]} vs "
            f"n_feat={n_feat}"
        )

    perm_delta: dict[str, np.ndarray] = {k: np.zeros(n_perm, dtype=float)
                                           for k in categories}
    # Pre-compute coefs * X = pops_full (we always compute ablated by
    # zeroing coefs for permuted-category indices).
    pops_recon = X.astype(np.float64) @ coefs.astype(np.float64)
    # We'll use pops_recon as the "full" when computing null deltas so the
    # null and observed are on the same reconstructed scale (removes any
    # alignment bias vs the loaded preds).
    partial_full_recon = partial_pearson(pops_recon, magma, cov_pli)

    for p_i in range(n_perm):
        perm_labels = base_labels.copy()
        rng_perm.shuffle(perm_labels)
        # Ablated scores under this permutation.
        for k in categories:
            drop_mask = (perm_labels == k)
            coefs_masked = coefs.copy()
            coefs_masked[drop_mask] = 0.0
            ab = X.astype(np.float64) @ coefs_masked.astype(np.float64)
            pa = partial_pearson(ab, magma, cov_pli)
            perm_delta[k][p_i] = partial_full_recon - pa
        if logger is not None and (p_i + 1) % 100 == 0:
            logger.info("  B.1 permutation %d/%d", p_i + 1, n_perm)

    # p_perm: fraction of null Δρ_k* ≥ observed Δρ_k.
    perm_summary: dict[str, dict] = {}
    decision_per_cat: dict[str, dict] = {}
    for k in categories:
        null = perm_delta[k]
        obs = observed_delta[k]
        # One-sided (upper): fraction ≥ observed.
        p_perm = float((null >= obs).mean())
        perm_summary[k] = {
            "null_mean": float(null.mean()),
            "null_std": float(null.std(ddof=1)),
            "null_median": float(np.median(null)),
            "null_p05": float(np.quantile(null, 0.05)),
            "null_p95": float(np.quantile(null, 0.95)),
            "observed_delta": obs,
            "p_perm_upper": p_perm,
        }
        # Decision rule B.1 (v2.1).
        meets_p = p_perm < SUB_B_PERM_PVALUE
        meets_floor = obs >= SUB_B_CATEGORY_DELTA_FLOOR
        drives = bool(meets_p and meets_floor)
        decision_per_cat[k] = {
            "category_drives": drives,
            "meets_p_perm_under_0_05": meets_p,
            "meets_delta_floor_0_10": meets_floor,
            "observed_delta": obs,
            "p_perm": p_perm,
            "delta_floor": SUB_B_CATEGORY_DELTA_FLOOR,
        }

    # Internal-consistency check: observed Δρ_k > raw partial_full → bug.
    internal_consistency = {
        k: (observed_delta[k] <= partial_full_point + 1e-9)
        for k in categories
    }
    any_ic_fail = any(not v for v in internal_consistency.values())

    # Save permutation null arrays to npz (auditor inspection).
    return {
        "partial_full_from_preds": float(partial_full_point),
        "partial_full_from_recon_at_X_beta": float(partial_full_recon),
        "observed_delta": observed_delta,
        "observed_partial_ablated": observed_partial_ablated,
        "variance_norm_diagnostic": {
            "sd_full": sd_full,
            "partial_variance_normalized_by_category": variance_norm,
            "delta_variance_normalized_by_category": variance_norm_delta,
        },
        "bootstrap": boot_results,
        "permutation_null_summary": perm_summary,
        "decision_per_category": decision_per_cat,
        "internal_consistency_per_cat": internal_consistency,
        "internal_consistency_fail": any_ic_fail,
        "_perm_arrays": perm_delta,  # pulled out by caller; not JSON-serialized
    }


def _ls_mc_null_ci_tail_resamples(tail_n: int, rho: float, tau: float,
                                    n_resample: int = 1000,
                                    seed: int = SUB_B_SEED) -> dict:
    """v2.1 FIX #3: LS-MC null distribution of tail-ρ at sample size tail_n.

    Draw `n_resample` bivariate-normal samples each of size ~tail_n by:
      1. For each resample, draw enough (X, Y) ~ N(0, Σ) with Σ=[[1,ρ],[ρ,1]]
         to accumulate tail_n observations where Y >= tau.
      2. Compute Pearson ρ on the tail observations.

    Returns {"null_rhos": array of length n_resample, "null_ci_lo": float,
    "null_ci_hi": float, "null_median": float, "null_mean": float}.

    WHY this construction: the brief's LS-MC anchor gives a single point
    estimate (0.268). For a CI-based decision we need the full sampling
    distribution of tail-ρ at the observed tail sample size under the
    bivariate-normal null. This matches what the observed bootstrap CI
    measures.

    WHY oversample-then-filter: drawing exactly tail_n from the conditional
    distribution P(·|Y>=tau) requires rejection sampling. We oversample by a
    factor that yields tail_n observations on average (1 / P(Y>=tau)), with a
    small safety margin.
    """
    from scipy.stats import pearsonr as _pr
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    p_tail = float(1.0 - norm.cdf(tau))
    if p_tail <= 0:
        return {"status": "failed", "reason": "p_tail==0"}
    # Oversample so each draw gives >= tail_n tail obs with high probability.
    # Use 1.5/p_tail as expected size, capped for memory.
    per_draw = int(max(tail_n / max(p_tail, 1e-6) * 1.5, tail_n * 2))
    null_rhos = np.full(n_resample, np.nan, dtype=float)
    for i in range(n_resample):
        samples = rng.multivariate_normal([0.0, 0.0], cov, size=per_draw)
        y = samples[:, 1]
        mask = y >= tau
        # If fewer than tail_n tail obs, skip (rare at per_draw=1.5/p_tail).
        if mask.sum() < tail_n:
            continue
        # Take first tail_n tail observations (unbiased since order is random).
        idxs = np.where(mask)[0][:tail_n]
        r, _ = _pr(samples[idxs, 0], samples[idxs, 1])
        null_rhos[i] = r
    finite = null_rhos[np.isfinite(null_rhos)]
    if finite.size == 0:
        return {"status": "failed", "reason": "no finite null rhos"}
    return {
        "status": "ok",
        "n_resample": int(n_resample),
        "n_finite": int(finite.size),
        "tail_n": int(tail_n),
        "null_mean": float(finite.mean()),
        "null_median": float(np.median(finite)),
        "null_std": float(finite.std(ddof=1)),
        "null_ci_lo": float(np.quantile(finite, 0.025)),
        "null_ci_hi": float(np.quantile(finite, 0.975)),
    }


def run_B2_tail_restricted(pops: np.ndarray, magma: np.ndarray,
                             magma_g: np.ndarray, logger,
                             use_gaussianized: bool,
                             n_boot: int = SUB_B_BOOT_N,
                             seed: int = SUB_B_SEED) -> dict:
    """B.2 tail-restricted Pearson ρ in upper / middle / lower tails.

    v2.1 FIX #3: the upper-tail decision is CI-BASED (not p-value based) to
    avoid double-counting bootstrap SD as a z-score. Specifically:
      - Observed upper-tail ρ has a 95% bootstrap CI (descriptive).
      - MC-null distribution of tail-ρ at the observed tail-N has a 95% CI
        via `_ls_mc_null_ci_tail_resamples` (1000 resamples at ρ=0.515,
        τ=0.842).
      - TAIL_ENRICHED fires iff observed 95% bootstrap CI does NOT overlap
        MC-null 95% CI AND observed point > MC-null median.

    No scalar p-value is produced for B.2 — BH family drops the 3 B.2 tests.

    Uses Gaussianized MAGMA if use_gaussianized; else raw. Tail thresholds:
    upper = norm.ppf(0.80), lower = norm.ppf(0.20). For raw MAGMA we pick
    80th / 20th percentile values for robustness.
    """
    m_use = magma_g if use_gaussianized else magma
    # Tail thresholds per brief L149-L151: norm.ppf(0.80) = 0.842 on
    # Gaussianized scale; on raw scale we use empirical 80th/20th pct.
    if use_gaussianized:
        tau_hi = float(norm.ppf(0.80))
        tau_lo = float(norm.ppf(0.20))
    else:
        tau_hi = float(np.quantile(m_use, 0.80))
        tau_lo = float(np.quantile(m_use, 0.20))
    upper_mask = m_use >= tau_hi
    lower_mask = m_use < tau_lo
    middle_mask = (~upper_mask) & (~lower_mask)

    def _rho_ci(mask: np.ndarray, tag: str) -> dict:
        n = int(mask.sum())
        if n < 10:
            return {"n": n, "status": "skipped",
                    "reason": "tail too small"}
        rho, _ = pearsonr(pops[mask], m_use[mask])
        # Bootstrap CI within the tail.
        idx_mat = build_bootstrap_idx(n, n_boot, seed + {"upper": 0,
                                                            "middle": 1,
                                                            "lower": 2}[tag])
        pops_t = pops[mask]
        m_t = m_use[mask]
        boot = np.zeros(n_boot, dtype=float)
        for b in range(n_boot):
            idx = idx_mat[b]
            r, _ = pearsonr(pops_t[idx], m_t[idx])
            boot[b] = r
        return {
            "n": n,
            "status": "ok",
            "point_pearson_rho": float(rho),
            "bootstrap": _ci(boot),
        }

    tails = {
        "upper_tail": _rho_ci(upper_mask, "upper"),
        "middle": _rho_ci(middle_mask, "middle"),
        "lower_tail": _rho_ci(lower_mask, "lower"),
    }

    # LS-MC point-estimate anchor (for audit / brief contract).
    ls_point = longin_solnik_mc_null(
        rho=SUB_B_LS_NULL_RHO, tau=SUB_B_LS_NULL_TAU,
        n_mc=SUB_B_LS_MC_N, seed=seed,
    )

    # v2.1 FIX #3: CI-based decision for upper tail.
    upper = tails["upper_tail"]
    if upper.get("status") != "ok":
        decision = {"tail_enriched": False,
                    "reason": "upper tail skipped"}
        ls_ci = {"status": "skipped"}
    else:
        tail_n = int(upper["n"])
        ls_ci = _ls_mc_null_ci_tail_resamples(
            tail_n=tail_n, rho=SUB_B_LS_NULL_RHO, tau=SUB_B_LS_NULL_TAU,
            n_resample=1000, seed=seed,
        )
        obs_rho = upper["point_pearson_rho"]
        obs_lo = upper["bootstrap"]["ci_lo"]
        obs_hi = upper["bootstrap"]["ci_hi"]
        if ls_ci.get("status") == "ok":
            null_lo = ls_ci["null_ci_lo"]
            null_hi = ls_ci["null_ci_hi"]
            null_median = ls_ci["null_median"]
            # CI intervals don't overlap AND observed point is above null.
            ci_disjoint = (obs_lo > null_hi) or (obs_hi < null_lo)
            tail_enriched = bool(ci_disjoint and obs_rho > null_median)
            decision = {
                "tail_enriched": tail_enriched,
                "upper_tail_rho": obs_rho,
                "upper_tail_ci_lo": obs_lo,
                "upper_tail_ci_hi": obs_hi,
                "ls_mc_null_point_estimate": ls_point.get("tail_rho_mc",
                                                             float("nan")),
                "ls_mc_null_median_at_tail_n": null_median,
                "ls_mc_null_ci_lo_at_tail_n": null_lo,
                "ls_mc_null_ci_hi_at_tail_n": null_hi,
                "ci_disjoint": bool(ci_disjoint),
                "decision_rule": ("CI-based: observed bootstrap CI disjoint "
                                    "from MC-null CI AND observed > null "
                                    "median (v2.1 FIX #3, no scalar p-value)"),
            }
        else:
            decision = {"tail_enriched": False,
                        "reason": f"LS-MC null failed: {ls_ci}"}
    return {
        "use_gaussianized_magma": bool(use_gaussianized),
        "thresholds": {"upper_tau": tau_hi, "lower_tau": tau_lo},
        "tails": tails,
        "ls_mc_null_point_estimate": ls_point,
        "ls_mc_null_ci_at_tail_n": ls_ci,
        "decision": decision,
    }


def run_B3_rank_partial(pops: np.ndarray, magma: np.ndarray,
                         magma_g: np.ndarray, pli: np.ndarray,
                         logger,
                         n_boot: int = SUB_B_BOOT_N,
                         seed: int = SUB_B_SEED) -> dict:
    """B.3 raw vs rank-partial ρ(PoPS, MAGMA | pLI).

    Computes:
      raw_partial   = partial_ρ(PoPS, MAGMA_raw | pLI)
      rank_partial  = partial_ρ(PoPS, MAGMA_rank | pLI)
      diff          = |rank_partial − raw_partial|
    """
    cov = pli.reshape(-1, 1)
    raw = partial_pearson(pops, magma, cov)
    rank = partial_pearson(pops, magma_g, cov)
    diff = abs(rank - raw)
    logger.info("B.3 raw=%.4f rank=%.4f |diff|=%.4f", raw, rank, diff)

    # Paired bootstrap for diff.
    n = pops.shape[0]
    idx_mat = build_bootstrap_idx(n, n_boot, seed + 3)
    boot_raw = np.zeros(n_boot, dtype=float)
    boot_rank = np.zeros(n_boot, dtype=float)
    boot_diff = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        idx = idx_mat[b]
        r_raw = partial_pearson(pops[idx], magma[idx], pli[idx].reshape(-1, 1))
        r_rank = partial_pearson(pops[idx], magma_g[idx], pli[idx].reshape(-1, 1))
        boot_raw[b] = r_raw
        boot_rank[b] = r_rank
        boot_diff[b] = abs(r_rank - r_raw)
    # Decision rule B.3.
    raw_diff_signed = rank - raw
    boot_signed = boot_rank - boot_raw
    ci_lo_signed = float(np.quantile(boot_signed, 0.025))
    ci_hi_signed = float(np.quantile(boot_signed, 0.975))
    ci_excludes_zero = (ci_lo_signed > 0) or (ci_hi_signed < 0)
    distribution_sensitive = bool(diff > SUB_B_DISTRIBUTION_DIFF
                                    and ci_excludes_zero)
    return {
        "raw_partial": float(raw),
        "rank_partial": float(rank),
        "abs_diff": float(diff),
        "signed_diff": float(raw_diff_signed),
        "bootstrap": {
            "raw_partial": _ci(boot_raw),
            "rank_partial": _ci(boot_rank),
            "abs_diff": _ci(boot_diff),
            "signed_diff": {
                "point": float(np.mean(boot_signed)),
                "ci_lo": ci_lo_signed, "ci_hi": ci_hi_signed,
                "std": float(np.std(boot_signed, ddof=1)),
            },
        },
        "decision": {
            "distribution_sensitive": distribution_sensitive,
            "threshold": SUB_B_DISTRIBUTION_DIFF,
            "ci_excludes_zero": bool(ci_excludes_zero),
        },
    }


def aggregate_sub_b(B1: dict, B2: dict, B3: dict,
                     other_gate_fires: bool,
                     other_mass_fraction: float,
                     other_mass_weighted: float,
                     shapiro_normal_raw: bool, shapiro_normal_gauss: bool,
                     ) -> dict:
    """First-match per brief_v2 §Sub-B DECISION RULE (L178-L183).

    v2.1 FIX #1: `other_gate_fires` is True iff EITHER `other_mass_fraction`
    OR `other_mass_weighted` exceeds SUB_B_OTHER_MASS_GATE on the LOADED
    subset (coefs ∩ munged).
    """
    # UNINTERPRETABLE GATES first (brief_v2 L183).
    if other_gate_fires:
        return {
            "classification": "UNINTERPRETABLE",
            "reason": (f"other mass gate fires on loaded subset: "
                        f"fraction={other_mass_fraction:.3f} "
                        f"weighted={other_mass_weighted:.3f} "
                        f"(threshold={SUB_B_OTHER_MASS_GATE})"),
        }
    if (not shapiro_normal_raw) and (not shapiro_normal_gauss):
        return {
            "classification": "UNINTERPRETABLE",
            "reason": "Shapiro fails on raw AND on Gaussianized MAGMA-Z",
        }
    if B1.get("internal_consistency_fail", False):
        return {
            "classification": "UNINTERPRETABLE",
            "reason": f"Internal consistency fail: {B1['internal_consistency_per_cat']}",
        }

    # 1. 2A_DIRECT_CATEGORY_DRIVES_k (first hit).
    decision_per_cat = B1["decision_per_category"]
    for k in ("expression", "ppi", "pathway", "other"):
        if decision_per_cat.get(k, {}).get("category_drives", False):
            return {
                "classification": f"2A_DIRECT_CATEGORY_DRIVES_{k}",
                "reason": (f"Δρ_{k}={decision_per_cat[k]['observed_delta']:.3f} "
                            f"p_perm={decision_per_cat[k]['p_perm']:.3f} "
                            f">= floor {SUB_B_CATEGORY_DELTA_FLOOR} AND "
                            f"p_perm < {SUB_B_PERM_PVALUE}"),
                "category": k,
            }
    # 2. 2B_TAIL_ENRICHED (v2.1 FIX #3: CI-based, no scalar p).
    if B2["decision"].get("tail_enriched", False):
        dec = B2["decision"]
        return {
            "classification": "2B_TAIL_ENRICHED",
            "reason": (f"upper_tail ρ={dec['upper_tail_rho']:.3f} "
                        f"CI=[{dec['upper_tail_ci_lo']:.3f}, "
                        f"{dec['upper_tail_ci_hi']:.3f}] disjoint from MC-null "
                        f"CI=[{dec['ls_mc_null_ci_lo_at_tail_n']:.3f}, "
                        f"{dec['ls_mc_null_ci_hi_at_tail_n']:.3f}]"),
        }
    # 3. 2C_DISTRIBUTION_SENSITIVE.
    if B3["decision"].get("distribution_sensitive", False):
        return {
            "classification": "2C_DISTRIBUTION_SENSITIVE",
            "reason": (f"|rank-raw|={B3['abs_diff']:.3f} > "
                        f"{SUB_B_DISTRIBUTION_DIFF}"),
        }

    # Check whether any branch partially fires (for INTERMEDIATE).
    # v2.1 FIX #3: B.2 no longer has a scalar p or abs_diff; use CI-disjoint
    # fall-through instead.
    b2_partial = False
    if "upper_tail_rho" in B2["decision"]:
        # If CIs touch (overlap only at edges) and observed > null median,
        # call it partial.
        dec = B2["decision"]
        null_median = dec.get("ls_mc_null_median_at_tail_n", float("nan"))
        if (np.isfinite(null_median)
                and dec["upper_tail_rho"] > null_median
                and not dec.get("ci_disjoint", False)):
            b2_partial = True
    partial_fire = any([
        any(decision_per_cat.get(k, {}).get("meets_p_perm_under_0_05", False)
            for k in ("expression", "ppi", "pathway", "other")),
        b2_partial,
    ])
    if partial_fire:
        return {
            "classification": "INTERMEDIATE",
            "reason": ("Partial fires (some p_perm significant but below "
                        "effect floor, or tail observed > null median but "
                        "CIs overlap). L056_02 binds iter_059."),
        }

    # 4. MECHANISM_NULL: none fire.
    return {
        "classification": "MECHANISM_NULL",
        "reason": ("No category drives (all p_perm >= 0.05 or Δρ < 0.10); "
                    "tail-null; distribution-robust. Criterion 5 UNMET."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_058 Sub-B v2.1")
    parser.add_argument("--smoke", action="store_true",
                         help="Smoke mode: reduce N (genes) and feature col "
                              "subset, n_boot/n_perm=50.")
    parser.add_argument("--smoke-genes", type=int, default=100,
                         help="Smoke gene subset size.")
    parser.add_argument("--smoke-feature-cols", type=int, default=100,
                         help="Smoke feature col subset size.")
    parser.add_argument("--smoke-boot", type=int, default=50)
    parser.add_argument("--smoke-perm", type=int, default=50)
    args = parser.parse_args()

    logger = setup_logger("batch_058.sub_b", LOGS_DIR / "sub_b.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "sub_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build shared frame (ENSGID × [PoPS, MAGMA, pLI]).
    frame, frame_info = build_shared_frame(
        smoke_subset=(args.smoke_genes if args.smoke else 0),
        logger=logger,
    )
    ensgids_shared = frame["ENSGID"].tolist()
    pops_preds = frame["PoPS_Score"].to_numpy(dtype=np.float64)
    magma_raw = frame["MAGMA_Z"].to_numpy(dtype=np.float64)
    pli = frame["lof_pLI"].to_numpy(dtype=np.float64)
    logger.info("Shared frame: N=%d", len(frame))

    # Load PoPS coefs + munged feature column names.
    coefs_full = load_pops_coefs(POPS_COEFS_P05)
    logger.info("coefs_full: %d features", len(coefs_full))

    # Collect all munged col names (single pass).
    all_munged_cols: set[str] = set()
    all_munged_list: list[str] = []
    i = 0
    while True:
        p = POPS_FEATURES_MUNGED_DIR / f"pops_features.cols.{i}.txt"
        if not p.exists():
            break
        with p.open() as fh:
            for line in fh:
                c = line.strip()
                if c:
                    all_munged_cols.add(c)
                    all_munged_list.append(c)
        i += 1
    logger.info("munged cols collected: %d", len(all_munged_list))

    # P0: Categorize ALL munged columns and write TSV.
    P0 = run_P0_category_map(all_munged_list, logger)

    # Decide which features to load: those in BOTH coefs AND munged.
    feats_to_load = [f for f in coefs_full.index if f in all_munged_cols]
    logger.info("Features in coefs ∩ munged: %d", len(feats_to_load))
    if args.smoke:
        feats_to_load = feats_to_load[:args.smoke_feature_cols]
        logger.info("  smoke: restricted to first %d features",
                     len(feats_to_load))

    # Load feature matrix restricted to coefs ∩ munged ∩ smoke subset.
    X_full, gene_rows, loaded_names = load_pops_features_matrix(
        POPS_FEATURES_MUNGED_DIR, cols_to_load=set(feats_to_load),
    )
    logger.info("Feature matrix X_full: shape=%s", X_full.shape)

    # Align to shared frame gene order (ENSGID subset from frame).
    gene_rows_idx = {g: i for i, g in enumerate(gene_rows)}
    row_indices = []
    keep_frame_rows = []
    for i_shared, g in enumerate(ensgids_shared):
        idx = gene_rows_idx.get(g)
        if idx is not None:
            row_indices.append(idx)
            keep_frame_rows.append(i_shared)
    if len(row_indices) < len(ensgids_shared):
        logger.info("Dropped %d genes not in PoPS features matrix rows",
                     len(ensgids_shared) - len(row_indices))
    X = X_full[np.asarray(row_indices)]
    ensgids_aligned = [ensgids_shared[i] for i in keep_frame_rows]
    pops_preds_aligned = pops_preds[np.asarray(keep_frame_rows)]
    magma_raw_aligned = magma_raw[np.asarray(keep_frame_rows)]
    pli_aligned = pli[np.asarray(keep_frame_rows)]
    logger.info("Aligned N=%d (rows with both PoPS preds and features mat)",
                 len(ensgids_aligned))

    # Categories only for loaded features (consistent with full P0 mapping).
    cat_map_loaded = {n: classify_feature(n) for n in loaded_names}

    # Reconstruction check: X @ β vs loaded PoPS preds.
    coefs_aligned_values = coefs_full.reindex(loaded_names).to_numpy(
        dtype=np.float64
    )
    if np.isnan(coefs_aligned_values).any():
        raise RuntimeError(
            "Coefs alignment: some loaded_names have no coef (should not "
            "happen since feats_to_load ⊆ coefs.index)"
        )
    pops_recon = X.astype(np.float64) @ coefs_aligned_values
    # Compare to loaded preds on the aligned genes.
    rho_recon, _ = pearsonr(pops_recon, pops_preds_aligned)
    max_abs_diff = float(np.nanmax(np.abs(pops_recon - pops_preds_aligned)))
    mean_abs_diff = float(np.nanmean(np.abs(pops_recon - pops_preds_aligned)))
    logger.info("Reconstruction check: ρ(X@β, preds)=%.4f max|diff|=%.3e "
                 "mean|diff|=%.3e", rho_recon, max_abs_diff, mean_abs_diff)
    reconstruction_check = {
        "rho_recon_vs_preds": float(rho_recon),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "n_features_used": int(X.shape[1]),
        "note": ("Exact match not expected when we use subset of coefs; "
                  "high ρ confirms linear-model sum is the right mechanism."),
    }
    atomic_write_json(reconstruction_check,
                       out_dir / "reconstruction_check.json")
    # v2.1 FIX #6: HALT if reconstruction ρ < 0.99. Below this threshold, the
    # X @ β sum is not tracking the loaded PoPS predictions, meaning our
    # ablation semantics ("remove category k") are not faithful to the real
    # PoPS score. Log the value regardless so failures are auditable.
    # In smoke mode we relax this since the feature subset is tiny by design
    # and reconstruction ρ is not expected to be high.
    RECON_RHO_MIN = 0.99
    if not args.smoke and (not np.isfinite(rho_recon)
                             or rho_recon < RECON_RHO_MIN):
        raise RuntimeError(
            f"Reconstruction ρ(X@β, PoPS_preds)={rho_recon:.4f} < "
            f"{RECON_RHO_MIN}; X @ β does not track loaded PoPS scores. "
            "Halting — the ablation mechanism would be uninterpretable. "
            f"See {out_dir / 'reconstruction_check.json'}."
        )

    # v2.1 FIX #1: compute BOTH other-mass metrics on the LOADED subset
    # (coefs ∩ munged), not on the full 57,742 munged universe. Gate fires
    # if EITHER fraction or weighted-mass exceeds 0.15.
    other_gates = compute_other_mass_gates(
        loaded_names, cat_map_loaded, coefs_aligned_values, X, logger,
    )

    # P1: Shapiro on MAGMA raw and Gaussianized.
    P1_raw = run_P1_shapiro(magma_raw_aligned, logger)
    magma_gauss = rank_gaussianize(magma_raw_aligned)
    P1_gauss_info = run_P1_shapiro(magma_gauss, logger)

    # P2: R2 gate using PoPS preds and raw MAGMA.
    P2 = run_P2_R2_gate(pops_preds_aligned, magma_raw_aligned, logger)

    # Determine which MAGMA to use downstream (brief_v2 L115).
    use_gaussianized = not bool(P1_raw["is_normal_at_alpha"])
    logger.info("Using %s MAGMA-Z for B.2/B.3",
                 "Gaussianized" if use_gaussianized else "raw")

    # B.1 Category ablation.
    n_boot = args.smoke_boot if args.smoke else SUB_B_BOOT_N
    n_perm = args.smoke_perm if args.smoke else SUB_B_PERM_N
    B1_raw = run_B1_category_ablation(
        X, coefs_aligned_values, loaded_names, cat_map_loaded,
        pops_preds_aligned, magma_raw_aligned, pli_aligned, logger,
        n_boot=n_boot, n_perm=n_perm, seed=SUB_B_SEED,
    )
    # Save permutation null to npz.
    perm_arrays = B1_raw.pop("_perm_arrays")
    np.savez_compressed(
        out_dir / "permutation_null.npz",
        expression=perm_arrays["expression"],
        ppi=perm_arrays["ppi"],
        pathway=perm_arrays["pathway"],
        other=perm_arrays["other"],
    )
    B1 = B1_raw  # cleaned dict

    # B.2 Tail-restricted.
    B2 = run_B2_tail_restricted(
        pops_preds_aligned, magma_raw_aligned, magma_gauss, logger,
        use_gaussianized=use_gaussianized, n_boot=n_boot, seed=SUB_B_SEED,
    )

    # B.3 Rank-transformed partial.
    B3 = run_B3_rank_partial(
        pops_preds_aligned, magma_raw_aligned, magma_gauss, pli_aligned,
        logger, n_boot=n_boot, seed=SUB_B_SEED,
    )

    # v2.1 FIX #3: Sub-B BH family is now 5 tests (4 B.1 + 1 B.3). The 3 B.2
    # tests are CI-based (MC-null vs bootstrap CI disjointness), not p-based,
    # so they are REMOVED from the BH family to avoid mixing inference modes.
    bh_pvals_labels = []
    bh_pvals = []
    for k in ("expression", "ppi", "pathway", "other"):
        bh_pvals_labels.append(f"B1_{k}")
        bh_pvals.append(B1["decision_per_category"][k]["p_perm"])
    # B.3: signed-diff bootstrap z (rank-partial − raw-partial).
    signed = B3["bootstrap"]["signed_diff"]
    z = abs(signed["point"]) / max(signed["std"], 1e-9)
    p_b3 = float(2 * (1 - norm.cdf(z)))
    bh_pvals_labels.append("B3_rank_minus_raw")
    bh_pvals.append(p_b3)
    qvals = bh_fdr(bh_pvals)

    bh_family = {
        "labels": bh_pvals_labels,
        "pvals": bh_pvals,
        "qvals": qvals,
        "q_threshold": BH_Q,
        "family_size": len(bh_pvals),
        "note": ("v2.1 FIX #3: 3 B.2 tests dropped from BH family (CI-based, "
                  "not p-based). Family is 4 B.1 + 1 B.3 = 5 tests."),
    }

    # Aggregate classification.
    aggregate = aggregate_sub_b(
        B1=B1, B2=B2, B3=B3,
        other_gate_fires=other_gates["gate_fires"],
        other_mass_fraction=other_gates["other_mass_fraction"],
        other_mass_weighted=other_gates["other_mass_weighted"],
        shapiro_normal_raw=P1_raw["is_normal_at_alpha"],
        shapiro_normal_gauss=P1_gauss_info["is_normal_at_alpha"],
    )

    # Provenance.
    provenance = {
        "gnomad_tsv": sha256_file(GNOMAD_TSV),
        "gene_annot": sha256_file(GENE_ANNOT),
        "magma_scz": sha256_file(MAGMA_SCZ_GENES_OUT),
        "pops_preds_p05": sha256_file(BATCH054_P05_PREDS),
        "pops_coefs_p05": sha256_file(POPS_COEFS_P05),
        "pops_features_dir": str(POPS_FEATURES_MUNGED_DIR),
        # We don't hash all 24 shard files; document path and let auditors
        # spot-check with sha256_file on individual shards.
    }

    results = {
        "status": "ok",
        "batch": "058", "sub": "b", "brief": "brief_v2.md (v2.1)",
        "wall_s": time.time() - t0,
        "smoke": bool(args.smoke),
        "n_shared_frame": frame_info["N"],
        "n_aligned_feature_matrix": int(X.shape[0]),
        "n_features_loaded": int(X.shape[1]),
        "n_features_munged_total": len(all_munged_list),
        "use_gaussianized_magma": use_gaussianized,
        "P0_feature_category_summary": P0,
        "loaded_subset_other_gates": other_gates,
        "P1_shapiro_raw": P1_raw,
        "P1_shapiro_gaussianized": P1_gauss_info,
        "P2_R2_reproduction_gate": P2,
        "reconstruction_check": reconstruction_check,
        "B1_category_ablation": B1,
        "B2_tail_restricted": B2,
        "B3_rank_partial": B3,
        "bh_fdr_family": bh_family,
        "aggregate_decision": aggregate,
        "provenance_sha256": provenance,
        "brief_contract": {
            "sub_b_bh_family_size": 5,
            "bh_fdr_family_size": 5,
            "bh_q": BH_Q,
            "seed": SUB_B_SEED,
            "n_boot": n_boot,
            "n_perm": n_perm,
            "delta_floor": SUB_B_CATEGORY_DELTA_FLOOR,
            "tail_enriched_diff": SUB_B_TAIL_ENRICHED_DIFF,
            "distribution_diff": SUB_B_DISTRIBUTION_DIFF,
            "other_mass_gate": SUB_B_OTHER_MASS_GATE,
            "permutation_null_category_counts_deviation": {
                "brief_v2_L133_spec": (
                    "preserve {36,772, 8,717, 8,479, 3,774} counts over "
                    "57,742 columns"
                ),
                "implemented_as": (
                    "category counts preserved within loaded coefs∩munged "
                    "subset; design-deviation from brief_v2 L133 due to "
                    "restriction to features with non-zero ridge β"
                ),
                "loaded_subset_counts": other_gates["counts_loaded"],
                "full_munged_counts": P0["counts"],
            },
            "sub_b_decision_rule_changes_v2_1": {
                "fix_1_other_gate": (
                    "other-mass gate evaluated on LOADED subset via BOTH "
                    "fraction AND weighted metrics; fires if either > 0.15"
                ),
                "fix_2_permutation_counts": (
                    "category counts preserved within loaded coefs∩munged "
                    "subset (see permutation_null_category_counts_deviation)"
                ),
                "fix_3_b2_ci_based": (
                    "B.2 decision is CI-disjointness vs MC-null CI at tail-N; "
                    "no scalar p-value; 3 B.2 tests removed from BH family"
                ),
                "fix_6_reconstruction_halt": (
                    "reconstruction ρ(X@β, PoPS_preds) >= 0.99 enforced in "
                    "non-smoke runs"
                ),
                "fix_8_ls_mc_n": (
                    f"LS-MC point-anchor n raised to {SUB_B_LS_MC_N} "
                    "(was 10,000) for ±0.0016 precision"
                ),
            },
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("Sub-B wrote %s (wall=%.1fs)",
                 out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
