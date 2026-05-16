#!/usr/bin/env python3
"""batch_056 Sub-A — ρ(PoPS_preds @ p=0.05, pLI) direct + partial-ρ framework.

Implements brief_v2.md §Sub-A exactly. Reproducibility contract:
  - Reproduction gate R1 FIRST: Spearman ρ(PoPS_preds_p0.05, MAGMA-Z) on the
    17,459-gene shared bg must match 0.5102 ± 1e-4. If fails, exit non-zero
    with clear diagnostic (per cardinal rule "no unaudited code runs").
  - Paired bootstrap uses np.random.default_rng(20260423) on the SAME n_genes
    as batch_054_A/055_A → bit-identical index matrix.
  - Partial-ρ residuals are re-fit WITHIN EACH bootstrap resample via
    numpy.linalg.lstsq (brief_v2 critic-2 clarification line 21).

Outputs
-------
experiments/batch_056/output/sub_a/results.json with:
  - reproduction_gate: R1 status and rho_observed/target/delta.
  - correlations: point estimates + CI for
      rho_pops_pli (Pearson + Spearman)
      rho_magma_pli (Pearson + Spearman)
      rho_pops_magma (Spearman, the anchor; Pearson reported for completeness)
      delta_rho_pops_vs_magma_pli (paired Pearson and Spearman).
  - partial_rho: (PoPS, MAGMA | pLI) and (PoPS, MAGMA | pLI + log10(length)).
  - decision_classification: Pattern A / B / B-weak / Intermediate /
    UNINTERPRETABLE per brief_v2 DECISION RULE.

WHY the correlation-by-correlation structure rather than one monolithic test:
  brief_v2 PREDICTION/DECISION RULE uses multiple quantities and thresholds;
  reporting them atomically lets downstream auditors (and ml-researcher) apply
  alternate thresholds without re-running the bootstrap.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Keep this file importable from the master orchestrator (sibling directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BATCH054_P05_PREDS,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    LOGS_DIR,
    OUTPUT_DIR,
    REPRO_R1_SPEARMAN_TARGET,
    REPRO_TOLERANCE,
    atomic_write_json,
    build_bootstrap_idx,
    load_common_ensgids,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    load_magma_scz,
    load_preds,
    percentile_ci,
    reproduce_spearman_anchor,
)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ---- Decision thresholds (brief_v2 §Sub-A DECISION RULE) ----
PAT_A_DELTA_MAX = 0.02      # Δρ ≤ +0.02
PAT_A_PARTIAL_MIN = 0.48    # partial ρ ≥ 0.48
PAT_A_PARTIAL_CI_NOT_EXCLUDE = 0.45

PAT_B_DELTA_MIN = 0.05      # Δρ ≥ +0.05
PAT_B_PARTIAL_MAX = 0.45

PAT_BW_DELTA_LO = 0.03
PAT_BW_DELTA_HI = 0.05
PAT_BW_PARTIAL_LO = 0.45
PAT_BW_PARTIAL_HI = 0.48
PAT_BW_PARTIAL_CI_MAX = 0.48

UNINTERPRETABLE_MIN_GENES = 15000


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_056.sub_a")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%dT%H:%M:%S")
        for h in (logging.FileHandler(LOGS_DIR / "sub_a.log"),
                  logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return residuals of y regressed on columns of X (intercept added).

    WHY numpy.linalg.lstsq: brief_v2 line 21 explicitly mandates this. It
    minimizes ||y - X β||_2 via SVD; deterministic, numerically stable, no
    matrix-inversion pitfalls.
    """
    n = X.shape[0]
    Xc = np.hstack([np.ones((n, 1), dtype=float), X.astype(float)])
    beta, *_ = np.linalg.lstsq(Xc, y.astype(float), rcond=None)
    return y.astype(float) - Xc @ beta


def partial_pearson(y1: np.ndarray, y2: np.ndarray, covariates: np.ndarray
                     ) -> float:
    """Partial Pearson ρ(y1, y2 | covariates) via residualization.

    WHY residualization (not a closed-form formula): the brief explicitly
    specifies per-resample residual recomputation; residualization makes this
    explicit and generalizes trivially when we add extra covariates.
    """
    r1 = residualize(y1, covariates)
    r2 = residualize(y2, covariates)
    rho, _ = pearsonr(r1, r2)
    return float(rho)


def classify(delta_pearson_ci: tuple[float, float],
             delta_point: float,
             partial_lr_point: float,
             partial_lr_ci: tuple[float, float],
             n_common: int) -> dict:
    """Apply the DECISION RULE (brief_v2 §Sub-A)."""
    # UNINTERPRETABLE (c): gene-universe too small.
    if n_common < UNINTERPRETABLE_MIN_GENES:
        return {
            "classification": "UNINTERPRETABLE",
            "reason": f"n_common={n_common} < {UNINTERPRETABLE_MIN_GENES}",
        }
    # UNINTERPRETABLE (a): CI on Δρ spans both 0 and +0.03.
    if delta_pearson_ci[0] < 0 and delta_pearson_ci[1] > PAT_BW_DELTA_LO:
        return {
            "classification": "UNINTERPRETABLE",
            "reason": f"Δρ CI {delta_pearson_ci} spans both 0 and {PAT_BW_DELTA_LO}",
        }
    # UNINTERPRETABLE (b): CI on partial ρ spans both 0.45 and 0.48.
    if (partial_lr_ci[0] < PAT_BW_PARTIAL_LO
            and partial_lr_ci[1] > PAT_BW_PARTIAL_HI):
        return {
            "classification": "UNINTERPRETABLE",
            "reason": f"partial ρ CI {partial_lr_ci} spans 0.45 and 0.48",
        }

    ci_lo, ci_hi = delta_pearson_ci
    p_ci_lo, p_ci_hi = partial_lr_ci

    # Pattern B: Δρ ≥ +0.05 CI > 0 AND partial ρ < 0.45 CI < 0.48.
    pat_b = (delta_point >= PAT_B_DELTA_MIN and ci_lo > 0
             and partial_lr_point < PAT_B_PARTIAL_MAX
             and p_ci_hi < PAT_BW_PARTIAL_CI_MAX)
    if pat_b:
        return {"classification": "Pattern_B",
                "reason": "Δρ ≥ +0.05 CI > 0 AND partial ρ < 0.45 CI < 0.48"}

    # Pattern A: Δρ ≤ +0.02 CI not excluding 0 AND partial ρ ≥ 0.48
    # CI not excluding 0.45.
    pat_a = (delta_point <= PAT_A_DELTA_MAX and ci_lo <= 0
             and partial_lr_point >= PAT_A_PARTIAL_MIN
             and p_ci_lo <= PAT_A_PARTIAL_CI_NOT_EXCLUDE)
    if pat_a:
        return {"classification": "Pattern_A",
                "reason": "Δρ ≤ +0.02 CI includes 0 AND partial ρ ≥ 0.48 CI "
                          "includes 0.45"}

    # Pattern B-weak: Δρ ∈ [+0.03, +0.05) CI > 0 OR partial ρ ∈ [0.45, 0.48)
    # CI < 0.48 — AND not both criteria meeting Pattern B full strength.
    bw_delta = (PAT_BW_DELTA_LO <= delta_point < PAT_BW_DELTA_HI and ci_lo > 0)
    bw_partial = (PAT_BW_PARTIAL_LO <= partial_lr_point < PAT_BW_PARTIAL_HI
                  and p_ci_hi < PAT_BW_PARTIAL_CI_MAX)
    if bw_delta or bw_partial:
        return {
            "classification": "Pattern_B_weak",
            "reason": (f"Δρ in [+0.03,+0.05) with CI>0 = {bw_delta}; "
                       f"partial ρ in [0.45,0.48) with CI<0.48 = {bw_partial}"),
        }

    return {"classification": "Intermediate",
            "reason": "None of Pattern A / B / B-weak thresholds met."}


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_056 Sub-A")
    parser.add_argument("--skip-gate", action="store_true",
                        help="skip R1 reproduction gate (DANGEROUS; audit-only)")
    args = parser.parse_args()

    logger = setup_logger()
    t0 = time.time()
    out_dir = OUTPUT_DIR / "sub_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- Loaders ---------------------
    try:
        preds = load_preds(BATCH054_P05_PREDS)
        magma = load_magma_scz()
        gnomad = load_gnomad_per_brief_v2()
        annot = load_gene_annot()
        common_ensgids = load_common_ensgids()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sub-A loader failure")
        atomic_write_json(
            {"status": "failed", "phase": "load", "error": str(exc)},
            out_dir / "results.json",
        )
        return 10

    logger.info("Loaded: preds=%d MAGMA=%d gnomAD=%d annot=%d common=%d",
                len(preds), len(magma), len(gnomad), len(annot),
                len(common_ensgids))

    # --------------------- Reproduction gate R1 ---------------------
    repro_r1 = reproduce_spearman_anchor(
        BATCH054_P05_PREDS, magma, common_ensgids,
        REPRO_R1_SPEARMAN_TARGET, REPRO_TOLERANCE,
    )
    logger.info("R1 reproduction: %s", repro_r1)
    if not args.skip_gate and not repro_r1.get("pass", False):
        atomic_write_json(
            {"status": "failed", "phase": "reproduction_gate",
             "reproduction_gate": repro_r1},
            out_dir / "results.json",
        )
        logger.error(
            "R1 FAIL: rho=%.6f target=%.4f Δ=%.6f > tol=%.0e",
            repro_r1.get("rho_observed", float("nan")),
            REPRO_R1_SPEARMAN_TARGET,
            repro_r1.get("delta", float("nan")),
            REPRO_TOLERANCE,
        )
        return 20

    # --------------------- Build analysis frame ---------------------
    # Sample = 17,459 common ENSGIDs ∩ gnomad (dedup) ∩ annot (for length).
    # WHY intersect all three: the primary gene universe for Sub-A is the
    # batch_055_A common set; we additionally require pLI AND log10(length)
    # to be available for each gene (drop-any-NaN).
    common = pd.DataFrame({"ENSGID": common_ensgids})
    merged = (common
              .merge(preds, on="ENSGID", how="left")
              .merge(magma, on="ENSGID", how="left")
              .merge(gnomad[["ENSGID", "lof_pLI", "lof_oe_ci_upper",
                             "lof_exp"]], on="ENSGID", how="left")
              .merge(annot[["ENSGID", "log10_gene_length",
                            "gene_length_bp"]], on="ENSGID", how="left"))
    merged_raw_n = len(merged)
    # Per-column NaN accounting — WHY: audit MAJOR #2 requires that Sub-A
    # log the join-loss counts for each filter so the bit-identical-
    # bootstrap-matrix claim can be checked against the ACTUAL n_final.
    drop_cols = ["PoPS_Score", "MAGMA_Z", "lof_pLI", "log10_gene_length"]
    drop_counts = {c: int(merged[c].isna().sum()) for c in drop_cols}
    merged = merged.dropna(subset=drop_cols).reset_index(drop=True)
    merged = merged.sort_values("ENSGID").reset_index(drop=True)
    n = len(merged)
    logger.info(
        "Intersection after dropna: %d (raw %d; lost %d to NaN; "
        "per-col NaN counts=%s)",
        n, merged_raw_n, merged_raw_n - n, drop_counts,
    )
    # Bit-identical bootstrap claim:
    #   The bootstrap matrix is bit-identical to batch_054_A ONLY when
    #   n == 17_459 (the batch_055_A common_ensgids size). If n < 17_459,
    #   the matrix is built with the SAME seed on the Sub-A analysis sample
    #   of size n — still deterministic and reproducible, but NOT identical
    #   to batch_054_A's matrix. Log a WARNING so auditors see this.
    EXPECTED_N = 17_459
    if n != EXPECTED_N:
        logger.warning(
            "n_final=%d != %d expected; bootstrap matrix is reproducible "
            "(seed=%d) but NOT bit-identical to batch_054_A's (%d) matrix. "
            "Join-loss breakdown: %s",
            n, EXPECTED_N, BOOTSTRAP_SEED, EXPECTED_N, drop_counts,
        )

    if n < UNINTERPRETABLE_MIN_GENES:
        atomic_write_json({
            "status": "failed", "phase": "intersection",
            "reproduction_gate": repro_r1,
            "n_after_intersect": n,
            "reason": f"<{UNINTERPRETABLE_MIN_GENES} genes after dropna; "
                      "gnomAD pLI or annot join leaked genes",
        }, out_dir / "results.json")
        logger.error("Intersection too small: n=%d", n)
        return 30

    pops = merged["PoPS_Score"].to_numpy(dtype=float)
    magma_z = merged["MAGMA_Z"].to_numpy(dtype=float)
    pli = merged["lof_pLI"].to_numpy(dtype=float)
    log10_len = merged["log10_gene_length"].to_numpy(dtype=float)

    # --------------------- Point estimates ---------------------
    point: dict = {}
    point["rho_pops_pli_pearson"], _ = pearsonr(pops, pli)
    point["rho_pops_pli_spearman"], _ = spearmanr(pops, pli)
    point["rho_magma_pli_pearson"], _ = pearsonr(magma_z, pli)
    point["rho_magma_pli_spearman"], _ = spearmanr(magma_z, pli)
    point["rho_pops_magma_spearman"], _ = spearmanr(pops, magma_z)
    point["rho_pops_magma_pearson"], _ = pearsonr(pops, magma_z)

    # Partial ρ point estimates (full-sample residualization).
    cov_pli = pli.reshape(-1, 1)
    cov_pli_len = np.column_stack([pli, log10_len])
    point["partial_rho_pops_magma_given_pli"] = partial_pearson(
        pops, magma_z, cov_pli
    )
    point["partial_rho_pops_magma_given_pli_length"] = partial_pearson(
        pops, magma_z, cov_pli_len
    )
    point["delta_rho_pops_minus_magma_vs_pli_pearson"] = float(
        point["rho_pops_pli_pearson"] - point["rho_magma_pli_pearson"]
    )
    point["delta_rho_pops_minus_magma_vs_pli_spearman"] = float(
        point["rho_pops_pli_spearman"] - point["rho_magma_pli_spearman"]
    )
    for k, v in point.items():
        point[k] = float(v)
    logger.info("Point estimates: %s", point)

    # --------------------- Paired bootstrap ---------------------
    # WHY n here (not len(common_ensgids)): brief_v2 bootstrap spec is
    # "same np.random.default_rng call" on n_genes; n_genes for Sub-A is the
    # post-dropna sample (partial ρ requires pLI + length non-NaN). If n
    # equals 17,459, the index matrix is bit-identical to batch_054_A.
    idx_mat = build_bootstrap_idx(n, BOOTSTRAP_N, BOOTSTRAP_SEED)
    logger.info("Bootstrap idx matrix: shape=%s seed=%d",
                idx_mat.shape, BOOTSTRAP_SEED)

    boot_rho_pops_pli = np.zeros(BOOTSTRAP_N)
    boot_rho_magma_pli = np.zeros(BOOTSTRAP_N)
    boot_rho_pops_pli_sp = np.zeros(BOOTSTRAP_N)
    boot_rho_magma_pli_sp = np.zeros(BOOTSTRAP_N)
    boot_partial_pli = np.zeros(BOOTSTRAP_N)
    boot_partial_pli_len = np.zeros(BOOTSTRAP_N)
    boot_rho_pops_magma_sp = np.zeros(BOOTSTRAP_N)

    for i in range(BOOTSTRAP_N):
        idx = idx_mat[i]
        p_i = pops[idx]
        m_i = magma_z[idx]
        l_i = pli[idx]
        ll_i = log10_len[idx]

        boot_rho_pops_pli[i], _ = pearsonr(p_i, l_i)
        boot_rho_magma_pli[i], _ = pearsonr(m_i, l_i)
        boot_rho_pops_pli_sp[i], _ = spearmanr(p_i, l_i)
        boot_rho_magma_pli_sp[i], _ = spearmanr(m_i, l_i)
        boot_rho_pops_magma_sp[i], _ = spearmanr(p_i, m_i)

        # Partial ρ: residuals recomputed WITHIN this resample (brief_v2 v2
        # critic-2).
        cov1 = l_i.reshape(-1, 1)
        cov2 = np.column_stack([l_i, ll_i])
        boot_partial_pli[i] = partial_pearson(p_i, m_i, cov1)
        boot_partial_pli_len[i] = partial_pearson(p_i, m_i, cov2)

        if (i + 1) % 250 == 0:
            logger.info("  bootstrap %d/%d", i + 1, BOOTSTRAP_N)

    # Paired Δρ bootstraps.
    boot_delta_pearson = boot_rho_pops_pli - boot_rho_magma_pli
    boot_delta_spearman = boot_rho_pops_pli_sp - boot_rho_magma_pli_sp

    # --------------------- CI summary ---------------------
    def _ci(samples: np.ndarray) -> dict:
        lo, hi, med = percentile_ci(samples)
        return {"ci_lo": lo, "ci_hi": hi, "median": med}

    bootstrap: dict = {
        "n_boot": BOOTSTRAP_N,
        "seed": BOOTSTRAP_SEED,
        "n_genes": int(n),
        "rho_pops_pli_pearson": _ci(boot_rho_pops_pli),
        "rho_magma_pli_pearson": _ci(boot_rho_magma_pli),
        "rho_pops_pli_spearman": _ci(boot_rho_pops_pli_sp),
        "rho_magma_pli_spearman": _ci(boot_rho_magma_pli_sp),
        "rho_pops_magma_spearman": _ci(boot_rho_pops_magma_sp),
        "delta_rho_pearson": _ci(boot_delta_pearson),
        "delta_rho_spearman": _ci(boot_delta_spearman),
        "partial_rho_given_pli": _ci(boot_partial_pli),
        "partial_rho_given_pli_length": _ci(boot_partial_pli_len),
    }

    # --------------------- Classification ---------------------
    delta_p_ci = (bootstrap["delta_rho_pearson"]["ci_lo"],
                  bootstrap["delta_rho_pearson"]["ci_hi"])
    partial_ci = (bootstrap["partial_rho_given_pli_length"]["ci_lo"],
                  bootstrap["partial_rho_given_pli_length"]["ci_hi"])
    classification = classify(
        delta_pearson_ci=delta_p_ci,
        delta_point=point["delta_rho_pops_minus_magma_vs_pli_pearson"],
        partial_lr_point=point["partial_rho_pops_magma_given_pli_length"],
        partial_lr_ci=partial_ci,
        n_common=n,
    )
    logger.info("Classification: %s", classification)

    # --------------------- Write ---------------------
    wall = time.time() - t0
    results = {
        "status": "ok",
        "batch": "056",
        "sub": "a",
        "wall_s": wall,
        "reproduction_gate_R1": repro_r1,
        "n_common": int(n),
        "n_final": int(n),  # alias; see provenance.bit_identical_claim
        "n_raw_before_dropna": int(merged_raw_n),
        "dropna_col_counts": drop_counts,
        "provenance": {
            "bit_identical_claim": (
                "bootstrap matrix built with np.random.default_rng("
                f"{BOOTSTRAP_SEED}) on the Sub-A analysis sample "
                f"(n = {int(n)}). Bit-identical to batch_054_A ONLY if "
                "n == 17459."
            ),
            "bit_identical_to_batch_054A": bool(n == 17_459),
            "expected_n_for_bit_identity": 17_459,
        },
        "point_estimates": point,
        "bootstrap": bootstrap,
        "decision_classification": classification,
        "inputs": {
            "pops_preds_p05": str(BATCH054_P05_PREDS),
            "magma_scz": "experiments/batch_053_B/output/PGC3_EUR_gene_ENSGID.genes.out",
            "gnomad": "data/item_15/gnomad.v4.1.constraint_metrics.tsv",
            "gene_annot": "data/pops_features/gene_annot_jun10.txt",
            "common_ensgids": "experiments/batch_055_A/output/common_ensgids.txt",
        },
        "gnomad_dedup_rule": "canonical AND mane_select AND ENSG prefix; "
                              "keep max lof.pLI; tie-break min lof.oe_ci.upper",
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("Sub-A wrote %s (wall=%.1fs)",
                out_dir / "results.json", wall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
