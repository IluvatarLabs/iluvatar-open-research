#!/usr/bin/env python3
"""batch_058 Sub-A v2 — F-055-01 cross-disorder B3 with robust diagnostic battery.

Implements brief_v2.md §Sub-A EXACTLY.

Per-disorder (8 disorders) battery on B3 indicator (18 genes):
  1. OLS β_1 + 95% CI (primary point estimate).
  2. HuberT-RLM β_1 + 95% CI.
  3. TukeyBiweight-RLM β_1 + 95% CI + convergence iter + final scale.
  4. Rank-within-disorder MAGMA-Z → OLS β_1_rank (SIGN-CONCORDANCE only).
  5. Max |DFBETAS_B3| (Fox 1997 cutoff 1.0).
  6. Cook's D (max + mean; supplementary).

Single 8-test BH-FDR family on OLS one-sided p.
LOEUF sensitivity on SCZ + MDD + ASD + Height.
Aggregate pattern: SCZ_SPECIFIC_ROBUST / SCZ+ONE / PAN / UNIVERSAL /
  INTERMEDIATE / UNINTERPRETABLE.
Reproduction gate R1: SCZ β_OLS ∈ [+2.5, +3.5].

Output: experiments/batch_058/output/sub_a/results.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BH_Q,
    B3_GENES,
    BATCH055B_WORK,
    CI_UPPER_STRONG,
    DFBETAS_CUTOFF,
    DISORDERS,
    FRAGILE_DIFF_THRESHOLD,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOEUF_DISORDERS,
    LOGS_DIR,
    MAGMA_GENELOC,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    REPRO_R1_SUB_A_LO,
    REPRO_R1_SUB_A_HI,
    aggregate_pattern_sub_a,
    atomic_write_json,
    bh_fdr,
    build_sub_a_frame,
    classify_disorder_v2,
    compute_dfbetas_cooks,
    fit_tukey_biweight,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    rank_gaussianize,
    setup_logger,
    sha256_file,
)

import numpy as np
import pandas as pd
import statsmodels.api as sm


MIN_N_UNIVERSE = 15000
PRIMARY_COVS = [
    "log10_gene_length", "lof_pLI",
    "log10_exp_lof_plus1", "log10_NSNPS_plus1",
]
LOEUF_COVS = [
    "log10_gene_length", "lof_oe_ci_upper",
    "log10_exp_lof_plus1", "log10_NSNPS_plus1",
]


def fit_ols(frame: pd.DataFrame, indicator_col: str,
             covs: list[str], outcome_col: str = "MAGMA_Z") -> dict:
    """OLS with 95% CI on β_1 (indicator coef)."""
    X = frame[[indicator_col] + covs].to_numpy(dtype=float)
    y = frame[outcome_col].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(y, Xc).fit()
    beta_1 = float(ols.params[1])
    se_1 = float(ols.bse[1])
    t_1 = float(ols.tvalues[1])
    p_two = float(ols.pvalues[1])
    p_one = float(p_two / 2.0 if t_1 > 0 else 1.0 - p_two / 2.0)
    ci_lo = beta_1 - 1.96 * se_1
    ci_hi = beta_1 + 1.96 * se_1
    return {
        "status": "ok",
        "beta_1": beta_1, "se_1": se_1, "t": t_1,
        "p_one_sided": p_one, "p_two_sided": p_two,
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "n": int(X.shape[0]),
        "covariates_used": covs,
        "r_squared": float(ols.rsquared),
    }


def fit_huber(frame: pd.DataFrame, indicator_col: str,
                covs: list[str], outcome_col: str = "MAGMA_Z") -> dict:
    """HuberT RLM."""
    X = frame[[indicator_col] + covs].to_numpy(dtype=float)
    y = frame[outcome_col].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    try:
        rlm = sm.RLM(y, Xc, M=sm.robust.norms.HuberT()).fit()
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    if len(rlm.params) <= 1:
        return {"status": "failed", "reason": "rank-deficient"}
    beta_1 = float(rlm.params[1])
    se_1 = float(rlm.bse[1])
    if not (np.isfinite(beta_1) and np.isfinite(se_1)):
        return {"status": "failed",
                "reason": f"non-finite β_1={beta_1} se={se_1}"}
    return {
        "status": "ok", "beta_1": beta_1, "se_1": se_1,
        "ci_lo": float(beta_1 - 1.96 * se_1),
        "ci_hi": float(beta_1 + 1.96 * se_1),
    }


def fit_rank_magma_ols(frame: pd.DataFrame, indicator_col: str,
                         covs: list[str]) -> dict:
    """OLS on rank-Gaussianized MAGMA-Z within-disorder.

    WHY within-disorder rank transform: brief_v2 L48 / L80 specifies
    rank-MAGMA-Z via norm.ppf((rank − 0.5)/N) applied PER DISORDER (not
    pooled across disorders), because MAGMA-Z scale varies across GWAS.
    """
    frame = frame.copy()
    frame["MAGMA_Z_rank"] = rank_gaussianize(frame["MAGMA_Z"].to_numpy())
    return fit_ols(frame, indicator_col, covs, outcome_col="MAGMA_Z_rank")


def influential_outlier_reconciliation(frame: pd.DataFrame,
                                        covs: list[str],
                                        indicator_col: str,
                                        dfbetas_info: dict) -> dict:
    """Post-removal OLS + Tukey after dropping max |DFBETAS| observation(s).

    Implements brief_v2 L59 "removing those observations reconciles OLS vs
    Tukey (diff < 0.2σ after removal)". We drop observations where
    |DFBETAS_B3| ≥ 1.0 (the Fox cutoff), then re-fit OLS and Tukey, and
    report whether the difference shrank to < FRAGILE_DIFF_THRESHOLD.

    Returns dict including post-removal fits and reconciliation flag.
    """
    if dfbetas_info.get("status") != "ok":
        return {"status": "skipped", "reason": "dfbetas failed"}
    # We need per-row |DFBETAS|; recompute influence here.
    from statsmodels.stats.outliers_influence import OLSInfluence
    X = frame[[indicator_col] + covs].to_numpy(dtype=float)
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    ols_fit = sm.OLS(y, Xc).fit()
    infl = OLSInfluence(ols_fit)
    dfbetas_all = np.asarray(infl.dfbetas)[:, 1]
    drop_mask = np.abs(dfbetas_all) >= DFBETAS_CUTOFF
    n_drop = int(drop_mask.sum())
    if n_drop == 0:
        return {"status": "skipped", "reason": "no obs exceed DFBETAS cutoff"}
    keep_frame = frame.loc[~drop_mask].copy()
    ols_post = fit_ols(keep_frame, indicator_col, covs)
    tukey_post = fit_tukey_biweight(keep_frame, covs, indicator_col)
    if tukey_post.get("status") == "ok":
        diff = abs(float(ols_post["beta_1"])
                   - float(tukey_post["beta_1"]))
        reconciled = diff < FRAGILE_DIFF_THRESHOLD
    else:
        diff = float("nan")
        reconciled = False
    return {
        "status": "ok",
        "n_obs_dropped": n_drop,
        "ols_post_removal": ols_post,
        "tukey_post_removal": tukey_post,
        "ols_tukey_diff_post": float(diff),
        "reconciled": bool(reconciled),
    }


def run_disorder(disorder: str, gnomad: pd.DataFrame, annot: pd.DataFrame,
                  gene_set_ensg: set[str], logger,
                  indicator_col: str = "in_set",
                  smoke_frame_size: int = 0) -> dict:
    """Run full v2.1 diagnostic battery for one disorder.

    Returns dict with all arms + DFBETAS/Cook's + influential-outlier recon.

    Args:
      smoke_frame_size: If > 0, truncate the regression frame to N randomly-
        sampled genes PLUS all gene_set_ensg members (so in_set stays
        representative). Used only for smoke tests to keep DFBETAS O(N^2)
        tractable in <60s.
    """
    try:
        frame = build_sub_a_frame(disorder, gnomad, annot, gene_set_ensg,
                                   gene_set_col=indicator_col)
    except Exception as exc:  # noqa: BLE001
        logger.exception("build_sub_a_frame failed for %s", disorder)
        return {"status": "failed", "reason": str(exc)}
    if smoke_frame_size and len(frame) > smoke_frame_size:
        # Keep all in_set=1 rows + random subsample from in_set=0.
        rng = np.random.default_rng(20260424)
        in_set_rows = frame[frame[indicator_col] == 1]
        out_rows = frame[frame[indicator_col] == 0]
        take = max(0, smoke_frame_size - len(in_set_rows))
        if take < len(out_rows):
            pick_idx = rng.choice(len(out_rows), size=take, replace=False)
            out_rows = out_rows.iloc[np.sort(pick_idx)]
        frame = pd.concat([in_set_rows, out_rows], axis=0
                            ).sort_values("ENSGID").reset_index(drop=True)
    n = len(frame)
    n_b3 = int(frame[indicator_col].sum())
    logger.info("%s: n=%d in_set=%d", disorder, n, n_b3)
    # In smoke mode we relax MIN_N_UNIVERSE.
    min_n = 100 if smoke_frame_size else MIN_N_UNIVERSE
    if n < min_n or n_b3 < 10:
        return {"status": "failed",
                "reason": f"n={n} in_set={n_b3}"}

    ols = fit_ols(frame, indicator_col, PRIMARY_COVS)
    huber = fit_huber(frame, indicator_col, PRIMARY_COVS)
    tukey = fit_tukey_biweight(frame, PRIMARY_COVS, indicator_col)
    rank_ols = fit_rank_magma_ols(frame, indicator_col, PRIMARY_COVS)
    infl = compute_dfbetas_cooks(frame, PRIMARY_COVS, indicator_col)
    recon = influential_outlier_reconciliation(
        frame, PRIMARY_COVS, indicator_col, infl,
    )
    return {
        "status": "ok",
        "n_gene_universe": n,
        "n_set_in_universe": n_b3,
        "ols": ols,
        "huber": huber,
        "tukey": tukey,
        "rank_magma_ols": rank_ols,
        "influence": infl,
        "influential_outlier_reconciliation": recon,
    }


def run_loeuf_sensitivity(disorder: str, gnomad: pd.DataFrame,
                           annot: pd.DataFrame,
                           gene_set_ensg: set[str], logger) -> dict:
    """Run OLS/Huber/Tukey with lof_pLI → lof_oe_ci_upper swap."""
    try:
        frame = build_sub_a_frame(disorder, gnomad, annot, gene_set_ensg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LOEUF build failed for %s", disorder)
        return {"status": "failed", "reason": str(exc)}
    frame = frame.dropna(subset=["lof_oe_ci_upper"]).copy()
    if frame.empty or frame["in_set"].sum() < 10:
        return {"status": "skipped",
                "reason": f"n={len(frame)} in_set={int(frame['in_set'].sum() if len(frame) else 0)}"}
    ols = fit_ols(frame, "in_set", LOEUF_COVS)
    huber = fit_huber(frame, "in_set", LOEUF_COVS)
    tukey = fit_tukey_biweight(frame, LOEUF_COVS)
    return {
        "status": "ok",
        "ols": ols, "huber": huber, "tukey": tukey,
        "covariates": LOEUF_COVS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_058 Sub-A v2.1")
    parser.add_argument("--smoke", action="store_true",
                         help="Smoke test: SCZ + ADHD only, skip LOEUF.")
    parser.add_argument("--smoke-frame-size", type=int, default=1000,
                         help="Smoke: cap regression frame size (in_set "
                              "preserved). Only used with --smoke.")
    args = parser.parse_args()
    smoke_frame = args.smoke_frame_size if args.smoke else 0

    logger = setup_logger("batch_058.sub_a", LOGS_DIR / "sub_a.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "sub_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load upstream data.
    gnomad = load_gnomad_per_brief_v2()
    annot = load_gene_annot()

    # Map B3 symbols → ENSGID via gene_annot.
    annot_by_name = annot.drop_duplicates(subset="NAME", keep="first"
                                            ).set_index("NAME")
    b3_sym_to_ensg: dict[str, str] = {}
    for s in B3_GENES:
        ensg = annot_by_name["ENSGID"].get(s)
        if ensg is not None:
            b3_sym_to_ensg[s] = ensg
    b3_ensg_set = set(b3_sym_to_ensg.values())
    logger.info("B3 mapped: %d/%d", len(b3_ensg_set), len(B3_GENES))
    if len(b3_ensg_set) < 16:
        raise RuntimeError(f"B3 mapping lost too many: {b3_sym_to_ensg}")

    disorders_to_run = DISORDERS if not args.smoke else ["scz", "adhd"]
    per_disorder: dict[str, dict] = {}
    for d in disorders_to_run:
        per_disorder[d] = run_disorder(
            d, gnomad, annot, b3_ensg_set, logger,
            smoke_frame_size=smoke_frame,
        )

    # BH-FDR across the 8-disorder family on OLS one-sided p.
    if all(per_disorder.get(d, {}).get("status") == "ok"
            for d in disorders_to_run):
        pvals = [per_disorder[d]["ols"]["p_one_sided"]
                 for d in disorders_to_run]
        qvals = bh_fdr(pvals)
        q_by_d = dict(zip(disorders_to_run, qvals))
        for d, q in q_by_d.items():
            per_disorder[d]["bh_q"] = q
    else:
        q_by_d = {}

    # Classification per disorder.
    # v2.1 FIX #5: we pre-compute `reconciled` from the post-removal OLS+Tukey
    # refit stored in item["influential_outlier_reconciliation"], and pass it
    # into classify_disorder_v2 so INFLUENTIAL_OUTLIERS only fires when
    # DFBETAS breach AND reconciliation succeeds.
    classifications: dict[str, str] = {}
    for d in disorders_to_run:
        item = per_disorder[d]
        if item.get("status") != "ok":
            classifications[d] = "UNCLASSIFIED"
            continue
        infl = item["influence"]
        max_df = (float(infl.get("max_abs_dfbetas_b3"))
                  if infl.get("status") == "ok" else float("nan"))
        q = item.get("bh_q", float("nan"))
        recon_info = item.get("influential_outlier_reconciliation", {})
        reconciled = bool(recon_info.get("status") == "ok"
                           and recon_info.get("reconciled", False))
        cls = classify_disorder_v2(
            item["ols"], item["huber"], item["tukey"],
            item["rank_magma_ols"], max_df, q,
            reconciled=reconciled,
        )
        classifications[d] = cls
        item["classification_v2"] = cls
        item["reconciled_post_removal"] = reconciled

    # Aggregate pattern.
    if not args.smoke and len(classifications) == 8:
        aggregate = aggregate_pattern_sub_a(classifications)
    else:
        aggregate = {
            "classification": "SMOKE_SKIPPED",
            "reason": "smoke test; not all 8 disorders run",
            "counts": {"smoke_disorders": list(classifications.keys())},
        }

    # LOEUF sensitivity on SCZ + MDD + ASD + Height.
    loeuf_results: dict[str, dict] = {}
    if not args.smoke:
        for d in LOEUF_DISORDERS:
            loeuf_results[d] = run_loeuf_sensitivity(
                d, gnomad, annot, b3_ensg_set, logger,
            )

    # Reproduction gate R1 (SCZ β_OLS ∈ [2.5, 3.5]).
    scz_item = per_disorder.get("scz", {})
    scz_beta = None
    if scz_item.get("status") == "ok":
        scz_beta = float(scz_item["ols"]["beta_1"])
    repro_r1 = {
        "target_lo": REPRO_R1_SUB_A_LO,
        "target_hi": REPRO_R1_SUB_A_HI,
        "scz_beta_ols": scz_beta,
        "pass": bool(scz_beta is not None
                     and REPRO_R1_SUB_A_LO <= scz_beta <= REPRO_R1_SUB_A_HI),
    }

    # Provenance SHA256s.
    provenance = {
        "gnomad_tsv": sha256_file(GNOMAD_TSV),
        "gene_annot": sha256_file(GENE_ANNOT),
        "magma_geneloc": sha256_file(MAGMA_GENELOC),
        "magma_scz": sha256_file(MAGMA_SCZ_GENES_OUT),
        "magma_per_disorder": {
            d: sha256_file(BATCH055B_WORK / d / "full.gene.genes.out")
            for d in disorders_to_run if d != "scz"
        },
    }

    results = {
        "status": "ok",
        "batch": "058", "sub": "a", "brief": "brief_v2.md (v2.1)",
        "wall_s": time.time() - t0,
        "smoke": bool(args.smoke),
        "model": ("MAGMA_Z ~ in_set + log10(gene_length) + lof_pLI + "
                   "log10(exp_lof+1) + log10(NSNPS+1)"),
        "disorders": disorders_to_run,
        "reproduction_gate_R1": repro_r1,
        "per_disorder": per_disorder,
        "per_disorder_classification": classifications,
        "bh_q_by_disorder": q_by_d,
        "aggregate_pattern": aggregate,
        "loeuf_sensitivity": loeuf_results,
        "provenance_sha256": provenance,
        "brief_contract": {
            "bh_fdr_family_size": 8,
            "bh_q": BH_Q,
            "ci_upper_strong_threshold": CI_UPPER_STRONG,
            "dfbetas_cutoff": DFBETAS_CUTOFF,
            "min_n_universe": MIN_N_UNIVERSE,
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("Sub-A wrote %s (wall=%.1fs)",
                 out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
