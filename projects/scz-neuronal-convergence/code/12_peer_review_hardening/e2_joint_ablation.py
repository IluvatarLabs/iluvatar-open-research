#!/usr/bin/env python3
"""batch_060 E2 -- NOV-059-11 joint-ablation interaction test with permutation null.

Implements brief_v2.md section E2 EXACTLY. FINAL F-056-A test.

Overview (WHY):
  batch_059 E1 full-run found 2B_MASS_SHARE: all 4 partition deltas are
  proportional to mass shares (within 0.20 tolerance). F059_02 reported 35%
  non-additivity (sum of partition deltas < total expression delta). VERA
  NOV-059-11 asks: does zeroing brain_human AND other_non_brain jointly
  produce a LARGER drop than the sum of their individual drops? If so,
  there is a feature-interaction (multicollinearity) between brain_human
  and other_non_brain that inflates the non-additivity.

  NEW in batch_060 (v2): permutation null. Shuffle brain_human feature
  labels across genes (preserving column structure of other partitions),
  refit ridge, compute interaction term. 1000 permutations. This calibrates
  the expected interaction under H0 (no true interaction, just ridge
  mechanics).

  Decision: interaction_term > perm_null_95th AND bootstrap CI_lo > 0
  -> INTERACTION_DETECTED. Otherwise -> NO_INTERACTION (mass-share confirmed).

  F-056-A PERMANENTLY CLOSED after this experiment regardless of outcome.

Outputs:
  experiments/batch_060/output/e2/results.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BATCH054_P05_PREDS,
    B060_SEED_E2_BOOT,
    B060_SEED_E2_PERM,
    E1_FITTED_ALPHA,
    E2_FULL_MODEL_RHO_ANCHOR,
    E2_N_BOOTSTRAP,
    E2_N_PERMUTATIONS,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    POPS_COEFS_P05,
    POPS_FEATURES_MUNGED_DIR,
    atomic_write_json,
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


def build_shared_frame(logger) -> tuple[pd.DataFrame, dict]:
    """Build shared frame: ENSGID x [PoPS_Score, MAGMA_Z, lof_pLI].

    WHY: same construction as batch_059 E1 build_shared_frame. We need
    the aligned gene universe for PoPS ablation computations.
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
    logger.info("E2 shared frame: N=%d", len(frame))
    return frame, {"N": int(len(frame))}


def load_coefs_and_features(logger) -> tuple[pd.Series, np.ndarray,
                                                list[str], list[str]]:
    """Load PoPS coefs + full feature matrix restricted to coefs intersect munged.

    WHY: same as batch_059 E1 load_coefs_and_features. Returns
    (coefs_series_aligned, X, gene_ensgids_rows, feature_names).
    """
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
    logger.info("Loading %d features (coefs intersect munged)", len(feats_to_load))
    X, gene_ensgids, kept_names = load_pops_features_matrix(
        POPS_FEATURES_MUNGED_DIR, cols_to_load=set(feats_to_load),
    )
    coefs_aligned = coefs_full.reindex(kept_names)
    if coefs_aligned.isna().any():
        raise RuntimeError("coefs alignment failure -- some kept_names unmapped")
    logger.info("Feature matrix: shape=%s", X.shape)
    return coefs_aligned, X, gene_ensgids, kept_names


def compute_delta(X: np.ndarray, coefs: np.ndarray,
                  feature_names: list[str],
                  drop_feature_names: set[str],
                  magma: np.ndarray, pli: np.ndarray) -> float:
    """Compute delta_rho = partial_rho(full | pLI) - partial_rho(ablated | pLI).

    WHY zero-out ablation: matches batch_058/059 Sub-B ablation semantics
    (Cardinal Rule 1).
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
    return float(rho_f - rho_a)


def run_permutation_null(X: np.ndarray, coefs: np.ndarray,
                          feature_names: list[str],
                          brain_human_names: set[str],
                          other_non_brain_names: set[str],
                          magma: np.ndarray, pli: np.ndarray,
                          n_perm: int, seed: int,
                          logger) -> np.ndarray:
    """Permutation null for interaction term.

    WHY: brief_v2 section E2 MEASUREMENT specifies shuffling brain_human
    feature labels across genes (preserving column structure of other
    partitions). For each permutation:
      1. Randomly reassign the brain_human feature indices among the
         partition universe (brain_human + other_non_brain features).
      2. Compute individual deltas and joint delta with the permuted labels.
      3. Compute interaction = joint_delta - (brain_delta + other_delta).

    The permutation breaks the true association between brain_human features
    and genes while preserving the same number of features in each group.
    This gives a null distribution for the interaction term under H0 (no
    true brain_human x other_non_brain interaction).

    Returns array of shape (n_perm,) with interaction terms.
    """
    rng = np.random.default_rng(seed)

    # Identify indices of brain_human and other_non_brain in feature_names.
    bh_indices = np.array(
        [i for i, n in enumerate(feature_names) if n in brain_human_names],
        dtype=np.int64,
    )
    onb_indices = np.array(
        [i for i, n in enumerate(feature_names) if n in other_non_brain_names],
        dtype=np.int64,
    )
    # Combined pool: brain_human + other_non_brain indices.
    combined = np.concatenate([bh_indices, onb_indices])
    n_bh = len(bh_indices)

    cov = pli.reshape(-1, 1)
    pops_full = X @ coefs
    rho_full = partial_pearson(pops_full, magma, cov)

    perm_interactions = np.zeros(n_perm, dtype=np.float64)

    for p_i in range(n_perm):
        # Shuffle the combined pool and reassign first n_bh as "brain_human".
        perm_combined = combined.copy()
        rng.shuffle(perm_combined)
        perm_bh = set(perm_combined[:n_bh].tolist())
        perm_onb = set(perm_combined[n_bh:].tolist())

        # Brain-only delta (zero out permuted brain_human features).
        coefs_bh = coefs.copy()
        for idx in perm_bh:
            coefs_bh[idx] = 0.0
        rho_bh_ab = partial_pearson(X @ coefs_bh, magma, cov)
        delta_bh = rho_full - rho_bh_ab

        # Other-only delta (zero out permuted other_non_brain features).
        coefs_onb = coefs.copy()
        for idx in perm_onb:
            coefs_onb[idx] = 0.0
        rho_onb_ab = partial_pearson(X @ coefs_onb, magma, cov)
        delta_onb = rho_full - rho_onb_ab

        # Joint delta (zero out both).
        coefs_joint = coefs.copy()
        for idx in perm_bh:
            coefs_joint[idx] = 0.0
        for idx in perm_onb:
            coefs_joint[idx] = 0.0
        rho_joint_ab = partial_pearson(X @ coefs_joint, magma, cov)
        delta_joint = rho_full - rho_joint_ab

        perm_interactions[p_i] = delta_joint - (delta_bh + delta_onb)

        if (p_i + 1) % 100 == 0:
            logger.info("  E2 perm %d/%d", p_i + 1, n_perm)

    return perm_interactions


def run_bootstrap_interaction(X: np.ndarray, coefs: np.ndarray,
                               feature_names: list[str],
                               brain_human_names: set[str],
                               other_non_brain_names: set[str],
                               magma: np.ndarray, pli: np.ndarray,
                               n_boot: int, seed: int,
                               logger) -> np.ndarray:
    """Bootstrap 95% CI on interaction term (paired over genes).

    WHY paired bootstrap: brief_v2 section E2 specifies B=1000 paired bootstrap.
    The pairing preserves within-gene covariance structure (features + outcome
    come from the same gene in each bootstrap sample).

    Returns array of shape (n_boot,) with interaction terms.
    """
    idx_mat = build_bootstrap_idx(X.shape[0], n_boot, seed)
    bh_mask = np.array([n in brain_human_names for n in feature_names], dtype=bool)
    onb_mask = np.array([n in other_non_brain_names for n in feature_names], dtype=bool)
    joint_mask = bh_mask | onb_mask

    boot_interactions = np.zeros(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = idx_mat[b]
        X_b = X[idx]
        magma_b = magma[idx]
        pli_b = pli[idx]
        cov_b = pli_b.reshape(-1, 1)

        pops_full_b = X_b @ coefs
        rho_full = partial_pearson(pops_full_b, magma_b, cov_b)

        # Brain-only ablation.
        coefs_bh = coefs.copy()
        coefs_bh[bh_mask] = 0.0
        rho_bh = partial_pearson(X_b @ coefs_bh, magma_b, cov_b)
        delta_bh = rho_full - rho_bh

        # Other-only ablation.
        coefs_onb = coefs.copy()
        coefs_onb[onb_mask] = 0.0
        rho_onb = partial_pearson(X_b @ coefs_onb, magma_b, cov_b)
        delta_onb = rho_full - rho_onb

        # Joint ablation.
        coefs_joint = coefs.copy()
        coefs_joint[joint_mask] = 0.0
        rho_joint = partial_pearson(X_b @ coefs_joint, magma_b, cov_b)
        delta_joint = rho_full - rho_joint

        boot_interactions[b] = delta_joint - (delta_bh + delta_onb)

        if (b + 1) % 200 == 0:
            logger.info("  E2 bootstrap %d/%d", b + 1, n_boot)

    return boot_interactions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="batch_060 E2: joint-ablation interaction test"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke: N=500 genes, 500 features, 20 perm, 20 boot.")
    parser.add_argument("--smoke-genes", type=int, default=500)
    parser.add_argument("--smoke-features", type=int, default=500)
    parser.add_argument("--smoke-perm", type=int, default=20)
    parser.add_argument("--smoke-boot", type=int, default=20)
    args = parser.parse_args()

    logger = setup_logger("batch_060.e2", LOGS_DIR / "e2.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "e2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build shared frame.
    frame, frame_info = build_shared_frame(logger)
    if args.smoke:
        frame = frame.iloc[:args.smoke_genes].reset_index(drop=True)
        logger.info("SMOKE: truncated frame to %d genes", len(frame))
    ensgids_shared = frame["ENSGID"].tolist()
    magma_raw = frame["MAGMA_Z"].to_numpy(dtype=np.float64)
    pli = frame["lof_pLI"].to_numpy(dtype=np.float64)

    # Step 2: Load coefs + features.
    coefs_series, X_full, gene_ensgids, feature_names = (
        load_coefs_and_features(logger)
    )
    if args.smoke:
        # Limit features for smoke test.
        n_feat_limit = min(args.smoke_features, len(feature_names))
        feature_names = feature_names[:n_feat_limit]
        X_full = X_full[:, :n_feat_limit]
        coefs_series = coefs_series.iloc[:n_feat_limit]
    coefs_values = coefs_series.to_numpy(dtype=np.float64)

    # Step 3: Align rows to shared frame.
    gene_idx_map = {g: i for i, g in enumerate(gene_ensgids)}
    row_indices = [gene_idx_map[g] for g in ensgids_shared if g in gene_idx_map]
    kept_frame_mask = [g in gene_idx_map for g in ensgids_shared]
    X = X_full[np.asarray(row_indices)].astype(np.float64)
    magma = magma_raw[np.asarray(kept_frame_mask)]
    pli_a = pli[np.asarray(kept_frame_mask)]
    logger.info("E2 aligned N=%d features=%d", X.shape[0], X.shape[1])

    # Step 4: Partition expression features.
    partitions = partition_expression_features(feature_names, logger=logger)
    brain_human_names = set(partitions["brain_human"])
    other_non_brain_names = set(partitions["other_non_brain"])

    # Step 5: UNINTERPRETABLE gate -- full-model rho must be within 0.02
    # of the 0.510 anchor (brief_v2 section E2).
    pops_full = X @ coefs_values
    cov_full = pli_a.reshape(-1, 1)
    full_rho = partial_pearson(pops_full, magma, cov_full)
    rho_deviation = abs(full_rho - E2_FULL_MODEL_RHO_ANCHOR)
    rho_gate_pass = bool(rho_deviation <= 0.02)
    logger.info(
        "E2 full-model rho=%.4f (anchor=%.3f, deviation=%.4f, pass=%s)",
        full_rho, E2_FULL_MODEL_RHO_ANCHOR, rho_deviation, rho_gate_pass,
    )

    # Step 6: Compute observed deltas.
    delta_brain = compute_delta(
        X, coefs_values, feature_names, brain_human_names, magma, pli_a
    )
    delta_other = compute_delta(
        X, coefs_values, feature_names, other_non_brain_names, magma, pli_a
    )
    joint_drop_names = brain_human_names | other_non_brain_names
    delta_joint = compute_delta(
        X, coefs_values, feature_names, joint_drop_names, magma, pli_a
    )
    interaction_observed = delta_joint - (delta_brain + delta_other)
    logger.info(
        "E2 observed: delta_brain=%.6f delta_other=%.6f delta_joint=%.6f "
        "interaction=%.6f",
        delta_brain, delta_other, delta_joint, interaction_observed,
    )

    # Step 7: Permutation null (1000 permutations).
    n_perm = args.smoke_perm if args.smoke else E2_N_PERMUTATIONS
    perm_interactions = run_permutation_null(
        X, coefs_values, feature_names,
        brain_human_names, other_non_brain_names,
        magma, pli_a, n_perm=n_perm, seed=B060_SEED_E2_PERM, logger=logger,
    )
    perm_95th = float(np.quantile(perm_interactions, 0.95))
    perm_mean = float(np.mean(perm_interactions))
    perm_std = float(np.std(perm_interactions, ddof=1))
    # p_perm: fraction of null interaction terms >= observed.
    p_perm = float((perm_interactions >= interaction_observed).mean())
    logger.info(
        "E2 perm null: mean=%.6f std=%.6f 95th=%.6f p_perm=%.4f",
        perm_mean, perm_std, perm_95th, p_perm,
    )

    # Step 8: Bootstrap 95% CI on interaction term.
    n_boot = args.smoke_boot if args.smoke else E2_N_BOOTSTRAP
    boot_interactions = run_bootstrap_interaction(
        X, coefs_values, feature_names,
        brain_human_names, other_non_brain_names,
        magma, pli_a, n_boot=n_boot, seed=B060_SEED_E2_BOOT, logger=logger,
    )
    boot_ci_lo = float(np.quantile(boot_interactions, 0.025))
    boot_ci_hi = float(np.quantile(boot_interactions, 0.975))
    boot_mean = float(np.mean(boot_interactions))
    logger.info(
        "E2 bootstrap: mean=%.6f CI=[%.6f, %.6f]",
        boot_mean, boot_ci_lo, boot_ci_hi,
    )

    # Step 9: Apply decision rule.
    if not rho_gate_pass:
        verdict = {
            "verdict": "UNINTERPRETABLE",
            "reason": (
                f"Full-model rho={full_rho:.4f} deviates from anchor "
                f"{E2_FULL_MODEL_RHO_ANCHOR} by {rho_deviation:.4f} > 0.02"
            ),
        }
    elif interaction_observed > perm_95th and boot_ci_lo > 0:
        verdict = {
            "verdict": "INTERACTION_DETECTED",
            "reason": (
                f"interaction={interaction_observed:.6f} > perm_95th="
                f"{perm_95th:.6f} AND boot_CI_lo={boot_ci_lo:.6f} > 0. "
                f"brain_human contributes through multicollinearity with "
                f"other_non_brain. Single positive from 5th test = "
                f"SUGGESTED at best."
            ),
        }
    else:
        verdict = {
            "verdict": "NO_INTERACTION",
            "reason": (
                f"interaction={interaction_observed:.6f} "
                f"{'<= perm_95th=' + f'{perm_95th:.6f}' if interaction_observed <= perm_95th else 'boot_CI_lo=' + f'{boot_ci_lo:.6f} <= 0'}. "
                f"Mass-share confirmed. F-056-A brain-biology REFUTED at "
                f"expression-partition level."
            ),
        }
    logger.info("E2 verdict: %s", verdict["verdict"])

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
        "batch": "060", "sub": "e2", "brief": "brief_v2.md",
        "wall_s": time.time() - t0,
        "smoke": bool(args.smoke),
        "n_aligned": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_brain_human_features": int(len(brain_human_names)),
        "n_other_non_brain_features": int(len(other_non_brain_names)),
        "full_model_rho": float(full_rho),
        "full_model_rho_anchor": float(E2_FULL_MODEL_RHO_ANCHOR),
        "full_model_rho_deviation": float(rho_deviation),
        "rho_gate_pass": rho_gate_pass,
        "observed_deltas": {
            "delta_brain_human": delta_brain,
            "delta_other_non_brain": delta_other,
            "delta_joint": delta_joint,
            "interaction_term": interaction_observed,
            "non_additivity_fraction": (
                float((delta_joint - (delta_brain + delta_other)) / delta_joint)
                if abs(delta_joint) > 1e-12 else float("nan")
            ),
        },
        "permutation_null": {
            "n_perm": n_perm,
            "seed": B060_SEED_E2_PERM,
            "mean": perm_mean,
            "std": perm_std,
            "p05": float(np.quantile(perm_interactions, 0.05)),
            "p50": float(np.median(perm_interactions)),
            "p95": perm_95th,
            "p_perm": p_perm,
        },
        "bootstrap_ci": {
            "n_boot": n_boot,
            "seed": B060_SEED_E2_BOOT,
            "mean": boot_mean,
            "ci_lo": boot_ci_lo,
            "ci_hi": boot_ci_hi,
        },
        "verdict": verdict,
        "f056_a_closure": (
            "F-056-A PERMANENTLY CLOSED. Regardless of outcome, no further "
            "experiments on the PoPS expression-mechanism channel."
        ),
        "provenance_sha256": provenance,
        "brief_contract": {
            "n_permutations": n_perm,
            "n_bootstrap": n_boot,
            "seed_perm": B060_SEED_E2_PERM,
            "seed_boot": B060_SEED_E2_BOOT,
            "full_model_rho_anchor": E2_FULL_MODEL_RHO_ANCHOR,
            "rho_anchor_tolerance": 0.02,
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("E2 wrote %s (wall=%.1fs)",
                 out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
