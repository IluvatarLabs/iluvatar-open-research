#!/usr/bin/env python3
"""batch_059 E1 — within-expression 4-bucket sub-partition + ridge-α sweep.

Implements brief_v2.md §1 EXACTLY.

Overview (WHY):
  iter_058 F058_02 found PoPS-mediation Δρ_expression=0.201 at fitted ridge
  α=31,623. Permutation-null preserving category COUNT (0.141) ruled out
  pure mass-driven mechanism. But within the ~37k-feature `expression`
  category, we still cannot tell whether the effect is brain-biology-specific
  vs mass-share proportional vs ridge-α-peculiar.

  E1 partitions expression features into 4 mutually-exclusive sub-buckets
  {brain_human, brain_mouse, immune, other_non_brain} via regex and
  (a) computes Δρ per sub-bucket at the fitted α with permutation null, and
  (b) re-fits PoPS ridge at 7 α values to test flatness of Δρ_brain_human
      across the ridge grid.

  DECISION RULE (first-match per brief_v2 §1):
    2E_UNINTERPRETABLE → 2A_BRAIN_BIOLOGY → 2B_MASS_SHARE →
    2C_ALPHA_SPECIFIC → 2C_bis_ALPHA_MARGINAL →
    2D_POSITIVE_UNANTICIPATED → 2D_NEGATIVE → 2D_INTERMEDIATE.

Outputs:
  experiments/batch_059/output/e1/results.json
  experiments/batch_059/output/e1/permutation_null.npz
  experiments/batch_059/output/e1/alpha_sweep.json
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
    COMMON_N_OFF_MHC,
    E1_ALPHA_GRID,
    E1_ALPHA_RANGE_ALPHA_SPECIFIC,
    E1_ALPHA_RANGE_MARGINAL,
    E1_BRAIN_HUMAN_DELTA_FLOOR,
    E1_BRAIN_HUMAN_DOMINANCE_GAP,
    E1_CI_OVERLAP_FLATNESS_MIN,
    E1_FITTED_ALPHA,
    E1_MASS_SHARE_TOLERANCE,
    E1_MIN_PARTITION_FEATURES,
    E1_NEGATIVE_BRAIN_HUMAN_VS_MOUSE,
    E1_POSITIVE_UNANTICIPATED_GAP,
    E1_R1_HI,
    E1_R1_LO,
    E1_R1_TARGET,
    E1_R1_TOLERANCE,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    POPS_COEFS_P05,
    POPS_FEATURES_MUNGED_DIR,
    SEED_E1_BOOT,
    SEED_E1_PERM,
    SUB_B_BOOT_N,
    SUB_B_PERM_N,
    atomic_write_json,
    bh_fdr,
    build_bootstrap_idx,
    classify_feature,
    load_gnomad_per_brief_v2,
    load_magma_scz,
    load_pops_coefs,
    load_pops_features_matrix,
    load_preds,
    partial_pearson,
    partition_expression_features,
    setup_logger,
    sha256_file,
)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def _ci(arr: np.ndarray) -> dict:
    """Return point, ci_lo, ci_hi dict from a bootstrap array."""
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"point": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "std": float("nan"),
                "n_finite": 0}
    return {
        "point": float(np.mean(finite)),
        "ci_lo": float(np.quantile(finite, 0.025)),
        "ci_hi": float(np.quantile(finite, 0.975)),
        "std": float(np.std(finite, ddof=1)),
        "n_finite": int(finite.size),
    }


def _ci_overlap_fraction(ci_a: dict, ci_b: dict) -> float:
    """Fraction of CI_A that overlaps CI_B (symmetric is OK for flatness test).

    WHY: brief_v2 §1 L70 "bootstrap CI of Δρ_brain_human at α=31,623 overlaps
    ≥50% with CI at α=1e4 AND α=1e5". We measure overlap as
    len(intersect) / len(CI_A) where CI_A is the fitted-α CI.
    """
    a_lo, a_hi = ci_a["ci_lo"], ci_a["ci_hi"]
    b_lo, b_hi = ci_b["ci_lo"], ci_b["ci_hi"]
    if not all(np.isfinite([a_lo, a_hi, b_lo, b_hi])):
        return 0.0
    a_len = max(a_hi - a_lo, 1e-12)
    inter_lo = max(a_lo, b_lo)
    inter_hi = min(a_hi, b_hi)
    inter = max(0.0, inter_hi - inter_lo)
    return float(inter / a_len)


def build_shared_frame(smoke_subset: int = 0, logger=None
                        ) -> tuple[pd.DataFrame, dict]:
    """Build shared frame: ENSGID × [PoPS_Score, MAGMA_Z, lof_pLI]. N ~17,459."""
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
        logger.info("E1 shared frame: N=%d (smoke=%d)", len(frame), smoke_subset)
    return frame, {"N": int(len(frame)), "smoke_subset": int(smoke_subset)}


def load_coefs_and_features(logger, smoke_n_features: int = 0
                              ) -> tuple[pd.Series, np.ndarray,
                                          list[str], list[str]]:
    """Load PoPS coefs + full feature matrix restricted to coefs ∩ munged.

    Returns (coefs_series_aligned, X, gene_ensgids_rows, feature_names).
    coefs_series_aligned and feature_names are 1:1 aligned by column.
    """
    # Collect all munged col names first.
    all_munged_cols: set[str] = set()
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
        i += 1
    logger.info("PoPS munged cols: %d", len(all_munged_cols))

    coefs_full = load_pops_coefs(POPS_COEFS_P05)
    feats_to_load = [f for f in coefs_full.index if f in all_munged_cols]
    if smoke_n_features:
        feats_to_load = feats_to_load[:smoke_n_features]
    logger.info("Loading %d features (coefs ∩ munged)", len(feats_to_load))
    X, gene_ensgids, kept_names = load_pops_features_matrix(
        POPS_FEATURES_MUNGED_DIR, cols_to_load=set(feats_to_load),
    )
    coefs_aligned = coefs_full.reindex(kept_names)
    if coefs_aligned.isna().any():
        raise RuntimeError("coefs alignment failure — some kept_names unmapped")
    logger.info("Feature matrix: shape=%s", X.shape)
    return coefs_aligned, X, gene_ensgids, kept_names


def delta_rho_for_partition(X: np.ndarray, coefs: np.ndarray,
                              feature_names: list[str],
                              drop_feature_names: set[str],
                              magma: np.ndarray, pli: np.ndarray) -> dict:
    """Compute Δρ = partial_ρ(full | pLI) − partial_ρ(ablated | pLI).

    Ablation: zero out coefs for features in `drop_feature_names`.
    WHY zero-out: matches batch_058 Sub-B ablation semantics exactly (Rule 1).
    """
    pops_full = X @ coefs
    drop_mask = np.array([n in drop_feature_names for n in feature_names],
                          dtype=bool)
    coefs_ab = coefs.copy()
    coefs_ab[drop_mask] = 0.0
    pops_ab = X @ coefs_ab
    cov = pli.reshape(-1, 1)
    rho_f = partial_pearson(pops_full, magma, cov)
    rho_a = partial_pearson(pops_ab, magma, cov)
    return {
        "partial_full": float(rho_f),
        "partial_ablated": float(rho_a),
        "delta": float(rho_f - rho_a),
        "n_features_dropped": int(drop_mask.sum()),
    }


def refit_ridge(X_full: np.ndarray, y_full: np.ndarray, alpha: float) -> np.ndarray:
    """Re-fit ridge β at the given α on the FULL feature matrix.

    WHY sklearn.linear_model.Ridge: Cardinal Rule 1. The PoPS pipeline fits
    its ridge via the same closed-form normal equation. We use sklearn's
    Ridge with fit_intercept=True and solver='auto' (defaults to cholesky for
    dense inputs), matching the Weeks 2023 pipeline's ridge CV behavior.

    Returns β vector of shape (P,). Intercept is absorbed into the training
    and not returned (the PoPS score is β·X + intercept — we mirror that
    internally in the caller by computing deltas at a fixed intercept).
    """
    from sklearn.linear_model import Ridge
    # float64 for numerical stability with large P.
    ridge = Ridge(alpha=float(alpha), fit_intercept=True, solver="auto",
                   random_state=SEED_E1_BOOT)
    ridge.fit(X_full.astype(np.float64), y_full.astype(np.float64))
    return ridge.coef_.astype(np.float64)


def run_alpha_sweep(X: np.ndarray, magma_for_ridge: np.ndarray,
                     feature_names: list[str],
                     partitions: dict[str, list[str]],
                     magma: np.ndarray, pli: np.ndarray,
                     alphas: list[float], n_boot: int, seed: int,
                     logger) -> dict:
    """Re-fit ridge at each α and compute Δρ per partition + bootstrap CI.

    WHY bootstrap within each α: flatness test requires CI at fitted α, at
    min α, and at max α — we must have CI per α to compute overlap fraction.

    `magma_for_ridge` is the training target (MAGMA-Z). We intentionally
    re-fit on the full feature matrix (brief_v2 open-question 1 resolution).

    IN-SAMPLE DESCRIPTIVE. Do NOT compute q-values from this sweep. Δρ values
    are used ONLY for flatness/peak detection in 2A/2C/2C_bis. WHY: the ridge
    is re-fit on the same evaluation frame (in-sample), so per-α Δρ estimates
    are biased toward higher apparent correlation (no held-out control).
    That bias is immaterial for pattern classification (flatness / peak
    location) but would be fatal for inferential q-values. 2B_MASS_SHARE uses
    the CANONICAL α=31,623 result ONLY (which is NOT re-fit here — it uses
    the pre-fitted PoPS coefficients from POPS_COEFS_P05), not the sweep.
    (B1 audit fix; see results.json field `is_descriptive_pattern_only`.)
    """
    idx_mat = build_bootstrap_idx(X.shape[0], n_boot, seed)
    partition_sets = {k: set(v) for k, v in partitions.items()}
    sweep: dict[str, dict] = {}
    for a_i, alpha in enumerate(alphas):
        logger.info("  α-sweep %d/%d: α=%.3e", a_i + 1, len(alphas), alpha)
        beta = refit_ridge(X, magma_for_ridge, alpha)
        # Point Δρ per partition.
        per_part_point: dict[str, dict] = {}
        for k, fset in partition_sets.items():
            pt = delta_rho_for_partition(X, beta, feature_names, fset,
                                           magma, pli)
            per_part_point[k] = pt
        # Bootstrap Δρ per partition (paired over genes).
        boot_delta: dict[str, np.ndarray] = {k: np.zeros(n_boot, dtype=float)
                                               for k in partition_sets}
        for b in range(n_boot):
            idx = idx_mat[b]
            pops_full_b = (X[idx] @ beta)
            magma_b = magma[idx]
            pli_b = pli[idx]
            cov_b = pli_b.reshape(-1, 1)
            rho_f = partial_pearson(pops_full_b, magma_b, cov_b)
            for k, fset in partition_sets.items():
                drop_mask = np.array([n in fset for n in feature_names],
                                       dtype=bool)
                beta_ab = beta.copy()
                beta_ab[drop_mask] = 0.0
                pops_ab_b = (X[idx] @ beta_ab)
                rho_a = partial_pearson(pops_ab_b, magma_b, cov_b)
                boot_delta[k][b] = rho_f - rho_a
        per_part_ci = {k: _ci(arr) for k, arr in boot_delta.items()}
        sweep[f"alpha_{alpha:.3e}"] = {
            "alpha": float(alpha),
            "per_partition_point": per_part_point,
            "per_partition_bootstrap_delta": per_part_ci,
        }
    return sweep


def run_permutation_null(X: np.ndarray, coefs: np.ndarray,
                           feature_names: list[str],
                           partitions: dict[str, list[str]],
                           magma: np.ndarray, pli: np.ndarray,
                           n_perm: int, seed: int,
                           logger) -> dict:
    """Permutation null preserving per-bucket feature COUNT.

    WHY count-preserving: brief_v2 §1 L84 mandates the null preserve
    sub-category feature COUNT (not mass), following iter_058 Sub-B's
    convention.

    WHY shuffling only within the partition universe (bug-fix 2026-04-24):
    `partitions` covers the ~14,850 expression+brain-ICA features, but
    `feature_names` has length ~17,423 (all PoPS features incl. PPI,
    pathway, non-brain "other"). Shuffling bucket labels over ALL features
    would (a) break the labels/features length invariant (the original
    bug — RuntimeError at line 319 in the pre-fix code) and (b) corrupt
    the null semantics by randomly assigning partition labels to features
    that are outside the partition universe entirely. The correct null is:
    permute bucket assignment only among the 14,850 partition members;
    non-partition features are never dropped (they keep their original
    coefficient in every draw).

    For each draw:
      1. Among the `partition_indices` (size 14,850), randomly re-assign
         the per-bucket counts.
      2. For each bucket k, zero out coefs at the feature-matrix indices
         whose permuted label == k and recompute partial_ρ.

    Returns {bucket: np.ndarray[n_perm]} null distributions.
    """
    rng = np.random.default_rng(seed)
    buckets = list(partitions.keys())
    counts = {k: len(v) for k, v in partitions.items()}

    # Partition universe: only these feature indices ever get shuffled.
    partition_feature_names: set[str] = set()
    for names in partitions.values():
        partition_feature_names.update(names)
    partition_indices = np.array(
        [i for i, n in enumerate(feature_names)
         if n in partition_feature_names],
        dtype=np.int64,
    )
    n_partition = partition_indices.shape[0]
    expected_partition = sum(counts.values())
    if n_partition != expected_partition:
        raise RuntimeError(
            f"E1 partition universe size mismatch: partition_indices="
            f"{n_partition} vs sum(partition counts)={expected_partition}"
        )

    # base_labels has one label per PARTITION feature, not per feature.
    base_labels = np.array(
        sum(([k] * counts[k] for k in buckets), []), dtype=object,
    )
    if base_labels.shape[0] != n_partition:
        raise RuntimeError(
            f"E1 base_labels vs partition universe mismatch: "
            f"{base_labels.shape[0]} vs {n_partition}"
        )
    logger.info(
        "E1 permutation universe: n_partition=%d (of n_feat=%d); "
        "counts=%s; n_perm=%d",
        n_partition, len(feature_names), counts, n_perm,
    )

    perm_delta: dict[str, np.ndarray] = {k: np.zeros(n_perm, dtype=float)
                                           for k in buckets}
    pops_full = X @ coefs
    cov = pli.reshape(-1, 1)
    partial_full = partial_pearson(pops_full, magma, cov)

    for p_i in range(n_perm):
        perm_labels = base_labels.copy()
        rng.shuffle(perm_labels)
        for k in buckets:
            # Boolean mask over the partition universe (length n_partition).
            drop_partition_pos = (perm_labels == k)
            # Translate partition-universe positions to feature-matrix idx.
            drop_feat_idx = partition_indices[drop_partition_pos]
            beta_ab = coefs.copy()
            beta_ab[drop_feat_idx] = 0.0
            pops_ab = X @ beta_ab
            rho_a = partial_pearson(pops_ab, magma, cov)
            perm_delta[k][p_i] = partial_full - rho_a
        if (p_i + 1) % 100 == 0:
            logger.info("  E1 perm %d/%d", p_i + 1, n_perm)
    return perm_delta


def compute_mass_shares(X: np.ndarray, coefs: np.ndarray,
                          feature_names: list[str],
                          partitions: dict[str, list[str]]) -> dict:
    """Compute per-bucket mass share |β|·mean(|X|) / total.

    WHY: brief_v2 §1 DECISION 2B_MASS_SHARE requires us to know mass_share_k
    for each partition so we can evaluate
    |Δρ_k/Δρ_total − mass_share_k/mass_share_total| < 0.20.

    WHY normalize within the partition universe (bug-fix 2026-04-24):
    The decision rule compares Δρ_k / Δρ_total against mass_k / mass_total
    where BOTH totals sum over the 4 partition buckets (i.e., the 14,850
    partition-universe features, NOT the full 17,423 feature set). We
    therefore report `mass_share` normalized to the partition universe.
    `mass_share_all_features` is additionally reported for transparency
    but is NOT used by decide(). `_total_mass_abs` remains the all-features
    total for provenance.
    """
    abs_coefs = np.abs(coefs.astype(np.float64))
    abs_X_mean = np.mean(np.abs(X.astype(np.float64)), axis=0)
    per_col = abs_coefs * abs_X_mean
    total_all = float(per_col.sum())
    # Partition-universe total: sum of |β|·mean(|X|) over the 4 buckets only.
    partition_mass_by_bucket: dict[str, float] = {}
    n_feat_by_bucket: dict[str, int] = {}
    for k, names in partitions.items():
        name_set = set(names)
        mask = np.array([n in name_set for n in feature_names], dtype=bool)
        partition_mass_by_bucket[k] = float(per_col[mask].sum())
        n_feat_by_bucket[k] = int(mask.sum())
    total_partition = float(sum(partition_mass_by_bucket.values()))
    out: dict[str, dict] = {}
    for k in partitions.keys():
        s = partition_mass_by_bucket[k]
        out[k] = {
            "n_features": n_feat_by_bucket[k],
            "mass_abs": s,
            # Normalized to the partition universe (used by decide() 2B).
            "mass_share": float(
                s / total_partition if total_partition > 0 else 0.0
            ),
            # Descriptive: normalized to ALL features (for transparency).
            "mass_share_all_features": float(
                s / total_all if total_all > 0 else 0.0
            ),
        }
    out["_total_mass_abs_all_features"] = total_all
    out["_total_mass_abs_partition_universe"] = total_partition
    return out


def decide(
    delta_by_partition_point: dict[str, float],
    permutation_null: dict[str, np.ndarray],
    alpha_sweep_per_partition: dict[str, dict],
    mass_shares: dict[str, dict],
    r1_pass: bool,
    partition_counts: dict[str, int],
) -> dict:
    """Apply E1 decision rule (first-match per brief_v2 §1 DECISION RULE).

    Order: 2E → 2A → 2B → 2C → 2C_bis → 2D_POS → 2D_NEG → 2D_INTERMEDIATE.

    `alpha_sweep_per_partition[k]` is a dict with keys α values:
        {alpha: {"delta_point": ..., "ci": {...}}}
    """
    reasons: list[str] = []

    # 2E: R1 fails or underpowered partition.
    if not r1_pass:
        return {"classification": "2E_UNINTERPRETABLE",
                "reason": "R1 reproduction gate failed"}
    for k, n in partition_counts.items():
        if n < E1_MIN_PARTITION_FEATURES:
            return {"classification": "2E_UNINTERPRETABLE",
                    "reason": (f"partition {k} has {n} < "
                                f"{E1_MIN_PARTITION_FEATURES} features")}

    # 2A: brain-biology.
    d_bh = delta_by_partition_point["brain_human"]
    d_bm = delta_by_partition_point["brain_mouse"]
    d_im = delta_by_partition_point["immune"]
    d_ot = delta_by_partition_point["other_non_brain"]
    null_bh = permutation_null["brain_human"]
    p95_null_bh = float(np.quantile(null_bh, 0.95))
    dominance = d_bh > max(d_im, d_ot) + E1_BRAIN_HUMAN_DOMINANCE_GAP

    # Flatness: bootstrap CI of Δρ_brain_human at α=31,623 overlaps ≥50% with
    # CIs at α=1e4 AND α=1e5 (brief_v2 §1 flatness definition).
    bh_sweep = alpha_sweep_per_partition.get("brain_human", {})
    ci_fitted = None
    ci_1e4 = None
    ci_1e5 = None
    for a_key, entry in bh_sweep.items():
        a_val = entry["alpha"]
        if abs(a_val - E1_FITTED_ALPHA) / max(E1_FITTED_ALPHA, 1) < 0.01:
            ci_fitted = entry["ci"]
        elif abs(a_val - 1e4) / 1e4 < 0.01:
            ci_1e4 = entry["ci"]
        elif abs(a_val - 1e5) / 1e5 < 0.01:
            ci_1e5 = entry["ci"]
    flatness_fraction = float("nan")
    flatness_pass = False
    if ci_fitted and ci_1e4 and ci_1e5:
        overlap_1e4 = _ci_overlap_fraction(ci_fitted, ci_1e4)
        overlap_1e5 = _ci_overlap_fraction(ci_fitted, ci_1e5)
        flatness_fraction = float(min(overlap_1e4, overlap_1e5))
        flatness_pass = (overlap_1e4 >= E1_CI_OVERLAP_FLATNESS_MIN
                          and overlap_1e5 >= E1_CI_OVERLAP_FLATNESS_MIN)

    if (d_bh >= E1_BRAIN_HUMAN_DELTA_FLOOR
            and d_bh > p95_null_bh and dominance and flatness_pass):
        return {
            "classification": "2A_BRAIN_BIOLOGY",
            "reason": (f"Δρ_brain_human={d_bh:.3f} >= {E1_BRAIN_HUMAN_DELTA_FLOOR} "
                        f"AND > perm-null 95th={p95_null_bh:.3f} AND > "
                        f"max(immune={d_im:.3f}, other={d_ot:.3f}) + "
                        f"{E1_BRAIN_HUMAN_DOMINANCE_GAP} AND flatness "
                        f"overlap_min={flatness_fraction:.2f} >= "
                        f"{E1_CI_OVERLAP_FLATNESS_MIN}"),
            "f056_a_criterion_5": "FULL_MET (brain-biology-specific)",
        }

    # 2B: mass-share.
    total_delta = d_bh + d_bm + d_im + d_ot
    total_mass = sum(mass_shares[k]["mass_abs"] for k in
                      ("brain_human", "brain_mouse", "immune",
                       "other_non_brain"))
    diffs: dict[str, float] = {}
    mass_share_ok_all = True
    if abs(total_delta) > 1e-9 and total_mass > 0:
        for k in ("brain_human", "brain_mouse", "immune", "other_non_brain"):
            r_delta = delta_by_partition_point[k] / total_delta
            r_mass = mass_shares[k]["mass_abs"] / total_mass
            diff = abs(r_delta - r_mass)
            diffs[k] = float(diff)
            if diff >= E1_MASS_SHARE_TOLERANCE:
                mass_share_ok_all = False
    else:
        mass_share_ok_all = False
    if mass_share_ok_all:
        return {
            "classification": "2B_MASS_SHARE",
            "reason": (f"|Δρ_k/Δρ_total - mass_k/mass_total| < "
                        f"{E1_MASS_SHARE_TOLERANCE} for ALL k; diffs={diffs}"),
            "f058_02_reclassified": "mechanical (mass-driven)",
            "f056_a_criterion_5": "PARTIALLY_MET (unchanged)",
        }

    # 2C: ridge-α specific. Δρ_brain_human varies across α by > 0.08 AND
    # max falls within ±0.3 log-decade of fitted α.
    bh_points_by_alpha: list[tuple[float, float]] = []
    for entry in bh_sweep.values():
        bh_points_by_alpha.append((entry["alpha"], entry["delta_point"]))
    if bh_points_by_alpha:
        deltas_sorted = sorted(bh_points_by_alpha, key=lambda kv: kv[0])
        bh_range = max(d[1] for d in deltas_sorted) - min(
            d[1] for d in deltas_sorted
        )
        # α where max occurs.
        max_alpha = max(deltas_sorted, key=lambda kv: kv[1])[0]
        dist_log_dec = (abs(np.log10(max_alpha)
                           - np.log10(E1_FITTED_ALPHA))
                        if max_alpha > 0 and E1_FITTED_ALPHA > 0 else
                        float("inf"))
        if (bh_range > E1_ALPHA_RANGE_ALPHA_SPECIFIC
                and dist_log_dec <= 0.3):
            return {
                "classification": "2C_ALPHA_SPECIFIC",
                "reason": (f"Δρ_brain_human α-range={bh_range:.3f} > "
                            f"{E1_ALPHA_RANGE_ALPHA_SPECIFIC} AND max at "
                            f"α={max_alpha:.2e} within 0.3 log-dec of "
                            f"α_fitted={E1_FITTED_ALPHA:.2e}"),
                "f058_02_reclassified": "ridge-α-specific",
            }
        # 2C_bis: marginal range (0.04–0.08).
        if (E1_ALPHA_RANGE_MARGINAL <= bh_range
                <= E1_ALPHA_RANGE_ALPHA_SPECIFIC):
            return {
                "classification": "2C_bis_ALPHA_MARGINAL",
                "reason": (f"Δρ_brain_human α-range={bh_range:.3f} in "
                            f"[{E1_ALPHA_RANGE_MARGINAL}, "
                            f"{E1_ALPHA_RANGE_ALPHA_SPECIFIC}]; descriptive "
                            "no classification change"),
            }

    # 2D_POS: non-brain dominates.
    if (d_ot > d_bh + E1_POSITIVE_UNANTICIPATED_GAP
            or d_im > d_bh + E1_POSITIVE_UNANTICIPATED_GAP):
        return {
            "classification": "2D_POSITIVE_UNANTICIPATED",
            "reason": (f"non-brain bucket > brain_human by "
                        f">{E1_POSITIVE_UNANTICIPATED_GAP}: Δρ_other={d_ot:.3f} "
                        f"Δρ_immune={d_im:.3f} Δρ_brain_human={d_bh:.3f}"),
            "f058_02_descriptive_only": True,
            "L058_03_applies": True,
        }

    # 2D_NEG: brain_human < 0 OR brain_human < brain_mouse by > 0.04.
    if (d_bh < 0
            or d_bh < d_bm - E1_NEGATIVE_BRAIN_HUMAN_VS_MOUSE):
        return {
            "classification": "2D_NEGATIVE",
            "reason": (f"Δρ_brain_human={d_bh:.3f} < 0 OR < Δρ_brain_mouse="
                        f"{d_bm:.3f} by > {E1_NEGATIVE_BRAIN_HUMAN_VS_MOUSE}"),
            "f058_02_descriptive_only": True,
        }

    # 2D_INTERMEDIATE (fall-through).
    return {
        "classification": "2D_INTERMEDIATE",
        "reason": (f"No pattern matched thresholds. Δρ_brain_human={d_bh:.3f} "
                    f"Δρ_brain_mouse={d_bm:.3f} Δρ_immune={d_im:.3f} "
                    f"Δρ_other={d_ot:.3f}. F058_02 stays SUGGESTED; F-056-A "
                    "stays PARTIALLY MET."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_059 E1")
    parser.add_argument("--smoke", action="store_true",
                         help="Smoke test: N=500 genes, 500 features, fewer "
                              "α/boot/perm.")
    parser.add_argument("--smoke-genes", type=int, default=500)
    parser.add_argument("--smoke-features", type=int, default=500)
    parser.add_argument("--smoke-boot", type=int, default=50)
    parser.add_argument("--smoke-perm", type=int, default=20)
    parser.add_argument("--smoke-alphas", type=int, default=3,
                         help="Smoke: first N α values of the grid.")
    args = parser.parse_args()

    logger = setup_logger("batch_059.e1", LOGS_DIR / "e1.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "e1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build shared frame.
    frame, frame_info = build_shared_frame(
        smoke_subset=(args.smoke_genes if args.smoke else 0),
        logger=logger,
    )
    ensgids_shared = frame["ENSGID"].tolist()
    pops_preds = frame["PoPS_Score"].to_numpy(dtype=np.float64)
    magma_raw = frame["MAGMA_Z"].to_numpy(dtype=np.float64)
    pli = frame["lof_pLI"].to_numpy(dtype=np.float64)

    # Load coefs + features.
    coefs_series, X_full, gene_ensgids, feature_names = (
        load_coefs_and_features(
            logger,
            smoke_n_features=(args.smoke_features if args.smoke else 0),
        )
    )
    coefs_values = coefs_series.to_numpy(dtype=np.float64)

    # Align rows to shared frame.
    gene_idx_map = {g: i for i, g in enumerate(gene_ensgids)}
    row_indices = [gene_idx_map[g] for g in ensgids_shared
                    if g in gene_idx_map]
    kept_frame_mask = [g in gene_idx_map for g in ensgids_shared]
    X = X_full[np.asarray(row_indices)].astype(np.float64)
    magma = magma_raw[np.asarray(kept_frame_mask)]
    pli_a = pli[np.asarray(kept_frame_mask)]
    logger.info("E1 aligned N=%d (features kept=%d)", X.shape[0], X.shape[1])

    # Partition expression features.
    partitions = partition_expression_features(feature_names, logger=logger)
    partition_counts = {k: len(v) for k, v in partitions.items()}

    # Mass shares.
    mass_shares = compute_mass_shares(X, coefs_values, feature_names, partitions)

    # R1 reproduction gate: split into STRICT (iter_058-compatible) and
    # WIDENED (iter_059 partition universe).
    #
    # WHY split (bug-fix 2026-04-24): iter_058's F058_02 Δρ_expression=0.201
    # was computed by zeroing ONLY features with
    # `classify_feature(name) == 'expression'` (brain-ICA + GTEx_brain were
    # NOT included — classify_feature labels them as "other"). iter_059's
    # brief_v2 L78 widens the partition universe to ALSO include GTEx_brain.*
    # and mouse_brain_projected_pcaloadings (~778 extra features), making
    # the widened Δρ ~0.2247 — outside the R1 tolerance band [0.189, 0.213].
    # Using the widened value as the R1 gate would FALSELY fail R1.
    # The correct R1 gate is the STRICT iter_058-compatible definition.
    # The widened value is reported separately as a descriptive number:
    # if widened > strict substantially, that is itself interesting evidence
    # that brain-ICA + GTEx_brain features carry signal beyond the
    # iter_058 "expression"-classified mass (supports E1's 2A hypothesis
    # before the sub-partition decision rule even runs).

    # STRICT: iter_058-compatible — all features classified "expression"
    # by classify_feature (and ONLY those).
    strict_expression_names = [n for n in feature_names
                                  if classify_feature(n) == "expression"]
    strict_expression_set = set(strict_expression_names)
    strict_expression_indices = np.array(
        [i for i, n in enumerate(feature_names)
         if classify_feature(n) == "expression"],
        dtype=np.int64,
    )
    cov_full = pli_a.reshape(-1, 1)
    pops_full_canonical = X @ coefs_values
    partial_full_canonical = partial_pearson(pops_full_canonical, magma,
                                                cov_full)
    beta_strict = coefs_values.copy()
    beta_strict[strict_expression_indices] = 0.0
    pops_strict = X @ beta_strict
    rho_strict_ablated = partial_pearson(pops_strict, magma, cov_full)
    delta_strict = float(partial_full_canonical - rho_strict_ablated)

    # WIDENED: iter_059 partition universe (4 buckets combined).
    all_partition_names = set().union(*(set(v) for v in partitions.values()))
    widened_indices = np.array(
        [i for i, n in enumerate(feature_names) if n in all_partition_names],
        dtype=np.int64,
    )
    beta_widened = coefs_values.copy()
    beta_widened[widened_indices] = 0.0
    pops_widened = X @ beta_widened
    rho_widened_ablated = partial_pearson(pops_widened, magma, cov_full)
    delta_widened = float(partial_full_canonical - rho_widened_ablated)

    # The "extra" features that iter_059 adds beyond iter_058's strict
    # expression set — for transparency in results.json.
    extra_widened_vs_strict = sorted(
        all_partition_names - strict_expression_set
    )
    # Also note any strict-expression features NOT in the widened partition
    # (should be empty by construction since the partitioner widens,
    # never narrows — but verify defensively).
    strict_minus_widened = sorted(
        strict_expression_set - all_partition_names
    )

    r1_pass_strict = bool(E1_R1_LO <= delta_strict <= E1_R1_HI)
    delta_widened_minus_strict = float(delta_widened - delta_strict)
    brain_ica_signal_flag = bool(delta_widened_minus_strict > 0.015)

    logger.info(
        "E1 R1 STRICT (iter_058-compat): Δρ=%.4f in [%.3f, %.3f]? %s",
        delta_strict, E1_R1_LO, E1_R1_HI, r1_pass_strict,
    )
    logger.info(
        "E1 R1 WIDENED (iter_059 partition): Δρ=%.4f; "
        "widened-strict=%.4f; brain_ica_signal_flag=%s; "
        "n_strict=%d, n_widened=%d, n_extra=%d, n_strict_only=%d",
        delta_widened, delta_widened_minus_strict, brain_ica_signal_flag,
        len(strict_expression_set), len(all_partition_names),
        len(extra_widened_vs_strict), len(strict_minus_widened),
    )

    # Canonical R1 pass = STRICT pass (per pre-reg UNINTERPRETABLE_2E rule).
    r1_pass = r1_pass_strict
    # Keep `r1_delta` for backward compatibility in downstream results
    # (mapped to the STRICT value — the canonical iter_058-compatible gate).
    r1_delta = delta_strict

    # Point Δρ per partition at fitted α (using loaded coefs).
    per_partition_point: dict[str, dict] = {}
    for k, names in partitions.items():
        r = delta_rho_for_partition(X, coefs_values, feature_names,
                                      set(names), magma, pli_a)
        per_partition_point[k] = r

    # Permutation null at fitted α (preserves per-partition count).
    n_perm = args.smoke_perm if args.smoke else SUB_B_PERM_N
    perm_null = run_permutation_null(
        X, coefs_values, feature_names, partitions, magma, pli_a,
        n_perm=n_perm, seed=SEED_E1_PERM, logger=logger,
    )
    np.savez_compressed(
        out_dir / "permutation_null.npz",
        brain_human=perm_null["brain_human"],
        brain_mouse=perm_null["brain_mouse"],
        immune=perm_null["immune"],
        other_non_brain=perm_null["other_non_brain"],
    )
    perm_summary = {}
    for k in ("brain_human", "brain_mouse", "immune", "other_non_brain"):
        null = perm_null[k]
        obs = per_partition_point[k]["delta"]
        perm_summary[k] = {
            "null_mean": float(null.mean()),
            "null_std": float(null.std(ddof=1)),
            "null_p05": float(np.quantile(null, 0.05)),
            "null_p50": float(np.median(null)),
            "null_p95": float(np.quantile(null, 0.95)),
            "observed_delta": float(obs),
            "p_perm_upper": float((null >= obs).mean()),
        }

    # α-sweep: re-fit ridge at 7 α values.
    alphas_to_run = (E1_ALPHA_GRID[:args.smoke_alphas]
                     if args.smoke else E1_ALPHA_GRID)
    n_boot = args.smoke_boot if args.smoke else SUB_B_BOOT_N
    alpha_sweep_raw = run_alpha_sweep(
        X, magma, feature_names, partitions, magma, pli_a,
        alphas=alphas_to_run, n_boot=n_boot, seed=SEED_E1_BOOT, logger=logger,
    )
    # Re-shape alpha_sweep_raw by partition for decide().
    alpha_sweep_per_part: dict[str, dict] = {
        k: {} for k in ("brain_human", "brain_mouse", "immune",
                         "other_non_brain")
    }
    for a_key, entry in alpha_sweep_raw.items():
        a_val = entry["alpha"]
        for k in alpha_sweep_per_part:
            pp = entry["per_partition_point"][k]
            ci = entry["per_partition_bootstrap_delta"][k]
            alpha_sweep_per_part[k][a_key] = {
                "alpha": a_val,
                "delta_point": float(pp["delta"]),
                "ci": ci,
            }

    # Save α-sweep as JSON.
    # B1 audit fix: explicitly mark as in-sample descriptive pattern only.
    # WHY: sweep re-fits ridge on the same evaluation frame. The Δρ values
    # here are NOT inferential; they are used only for flatness / peak
    # detection in 2A/2C/2C_bis (brief_v2 §1). Any attempt to compute
    # per-α q-values from the sweep would be invalid.
    atomic_write_json(
        {
            "alphas": alphas_to_run,
            "sweep": alpha_sweep_raw,
            "is_descriptive_pattern_only": True,
            "audit_note": (
                "B1 fix: α-sweep is IN-SAMPLE DESCRIPTIVE (ridge re-fit on "
                "the same evaluation frame per-α). Δρ values are used ONLY "
                "for flatness / peak detection in 2A / 2C / 2C_bis. Do NOT "
                "compute q-values from this sweep. Only the canonical "
                "α=31,623 (the pre-fitted PoPS coefficients from "
                "POPS_COEFS_P05) is used for inferential 2B_MASS_SHARE."
            ),
        },
        out_dir / "alpha_sweep.json",
    )

    # Decision.
    delta_by_point = {k: float(v["delta"]) for k, v in per_partition_point.items()}
    decision = decide(
        delta_by_partition_point=delta_by_point,
        permutation_null=perm_null,
        alpha_sweep_per_partition=alpha_sweep_per_part,
        mass_shares=mass_shares,
        r1_pass=r1_pass,
        partition_counts=partition_counts,
    )

    # B2 audit fix: BH-FDR over 4 per-partition tests at the fitted α only.
    # WHY we dropped the 28-cell family: the ridge α-sweep re-fits on the
    # same evaluation frame (in-sample), so α != fitted Δρ values are
    # descriptive (flatness / peak detection — see alpha_sweep.json
    # `is_descriptive_pattern_only=True`). Computing q-values across the
    # 28-cell grid would conflate inferential (fitted-α, canonical PoPS
    # coefficients) and descriptive (re-fit ridge) tests. Per the audit,
    # the brief_v2 design.yaml §BH-FDR 28-cell spec is SUPERSEDED by this
    # addendum: only the 4-cell fitted-α family is used for inference.
    bh_labels: list[str] = []
    bh_pvals: list[float] = []
    for k in ("brain_human", "brain_mouse", "immune", "other_non_brain"):
        null = perm_null[k]
        obs = float(per_partition_point[k]["delta"])
        bh_labels.append(f"{k}_at_fitted_alpha")
        bh_pvals.append(float((null >= obs).mean()))
    qvals = bh_fdr(bh_pvals)
    bh_family = {
        "labels": bh_labels,
        "pvals": bh_pvals,
        "qvals": qvals,
        "q_threshold": BH_Q,
        "family_size": len(bh_pvals),
        "fitted_alpha": float(E1_FITTED_ALPHA),
        "note": (
            "B2 audit fix: BH-FDR family is 4 tests (one per partition) at "
            "the fitted α=31,623 using the canonical pre-fitted PoPS "
            "coefficients. The 28-cell family (4 buckets × 7 α) from the "
            "α-sweep is DROPPED because the sweep is in-sample descriptive "
            "(see alpha_sweep.json is_descriptive_pattern_only). The "
            "brief_v2 design.yaml 28-cell BH-FDR spec is considered "
            "SUPERSEDED; this 4-cell family is the canonical inference."
        ),
    }

    # Provenance.
    provenance = {
        "gnomad_tsv": sha256_file(GNOMAD_TSV),
        "gene_annot": sha256_file(GENE_ANNOT),
        "magma_scz": sha256_file(MAGMA_SCZ_GENES_OUT),
        "pops_preds_p05": sha256_file(BATCH054_P05_PREDS),
        "pops_coefs_p05": sha256_file(POPS_COEFS_P05),
    }

    results = {
        "status": "ok",
        "batch": "059", "sub": "e1", "brief": "brief_v2.md (v2.1)",
        "wall_s": time.time() - t0,
        "smoke": bool(args.smoke),
        "n_shared_frame": frame_info["N"],
        "n_aligned": int(X.shape[0]),
        "n_features_loaded": int(X.shape[1]),
        "partition_counts": partition_counts,
        "n_partition_features_iter059": int(len(all_partition_names)),
        "n_strict_expression_features_iter058_compat": int(
            len(strict_expression_set)
        ),
        "n_extra_in_widened_vs_strict": int(len(extra_widened_vs_strict)),
        "extra_widened_vs_strict_feature_names": extra_widened_vs_strict,
        "n_strict_only_not_in_widened": int(len(strict_minus_widened)),
        "strict_only_not_in_widened_feature_names": strict_minus_widened,
        "mass_shares": mass_shares,
        "per_partition_point_at_fitted_alpha": per_partition_point,
        "permutation_null_summary": perm_summary,
        "R1_reproduction_gate": {
            # Canonical R1 gate (pre-reg): iter_058-compatible STRICT.
            # This is the field used for UNINTERPRETABLE_2E classification.
            "delta_rho_expression_strict_iter058_compat": delta_strict,
            "R1_pass_strict": r1_pass_strict,
            "R1_strict_target": E1_R1_TARGET,
            "R1_strict_tolerance": E1_R1_TOLERANCE,
            "lo": E1_R1_LO,
            "hi": E1_R1_HI,
            # Descriptive: iter_059 widened partition universe.
            "delta_rho_expression_widened_iter059_partition":
                delta_widened,
            "delta_rho_widened_minus_strict": delta_widened_minus_strict,
            "brain_ica_signal_flag": brain_ica_signal_flag,
            # Backward-compatibility alias — the strict value is the
            # canonical R1 number.
            "delta_rho_expression_combined": r1_delta,
            "pass": r1_pass,
            "notes": (
                "R1 uses iter_058-compatible STRICT definition "
                "(classify_feature == 'expression' ONLY) as the canonical "
                "gate, per pre-reg. The WIDENED value includes the "
                "re-categorized GTEx_brain + brain ICA-loadings (brief_v2 "
                "L78). If delta_rho_widened_minus_strict > 0.015 "
                "(brain_ica_signal_flag=True), brain-ICA / GTEx_brain "
                "features carry Δρ above-and-beyond the iter_058 "
                "expression mass — descriptive support for E1's "
                "brain-specificity hypothesis before the sub-partition "
                "decision rule runs. If R1 strict fails, the decision "
                "rule classifies UNINTERPRETABLE_2E per pre-reg."
            ),
        },
        "alpha_sweep_per_partition": alpha_sweep_per_part,
        "bh_fdr_family": bh_family,
        "decision": decision,
        "provenance_sha256": provenance,
        "brief_contract": {
            "n_alphas": len(alphas_to_run),
            "n_permutations": n_perm,
            "n_bootstrap": n_boot,
            "seed_bootstrap": SEED_E1_BOOT,
            "seed_permutation": SEED_E1_PERM,
            "delta_floor_brain_human": E1_BRAIN_HUMAN_DELTA_FLOOR,
            "flatness_min_overlap": E1_CI_OVERLAP_FLATNESS_MIN,
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("E1 wrote %s (wall=%.1fs)",
                 out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
