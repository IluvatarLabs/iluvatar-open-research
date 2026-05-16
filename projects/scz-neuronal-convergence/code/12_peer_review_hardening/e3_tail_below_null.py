#!/usr/bin/env python3
"""batch_059 E3 — tail-below-null diagnostic (Thorndike + kurtosis + MHC + cross-trait).

Implements brief_v2.md §3 EXACTLY.

Overview (WHY):
  iter_058 F058_03 reported upper-tail ρ(PoPS, MAGMA-Z_SCZ | pLI)=0.229,
  which is below the Longin-Solnik MC null of 0.270 at ρ=0.515, τ=0.842.
  Thorndike Case II single-truncation closed-form also predicts 0.2705,
  confirming the MC null is correctly calibrated. E3 tests which of three
  mechanisms explains the tail deflation:

    3A_CEILING_SATURATION: PoPS upper-tail platykurtic (kurt ≤ -0.5).
    3B_MHC_DILUTION: MHC-exclusion raises upper-tail ρ by Δ ≥ max(0.02,
        2·SE_bootstrap).
    3C_CROSS_TRAIT_GENERIC: same tail-deflation pattern on ≥ 2 of
        {IBD, BIP, Height} relative to trait-specific Thorndike null.

  3D (null-integrity) runs first INDEPENDENTLY as a sanity check (does NOT
  gate 3A/3B/3C).

Outputs:
  experiments/batch_059/output/e3/results.json
  experiments/batch_059/output/e3/mc_null.json
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
    E3_3A_KURTOSIS_UPPER_THRESHOLD,
    E3_3A_MIDDLE_KURTOSIS_FLOOR,
    E3_3A_MIDDLE_QUANTILE_HI,
    E3_3A_MIDDLE_QUANTILE_LO,
    E3_3B_DELTA_FLOOR,
    E3_3B_SE_MULTIPLIER,
    E3_3C_DELTA_THRESHOLD,
    E3_3C_MIN_DISORDERS_FIRING,
    E3_FULL_RHO_ANCHOR,
    E3_ITER058_MC_NULL,
    E3_MC_AGREEMENT_THRESHOLD,
    E3_MHC_CHR,
    E3_MHC_END,
    E3_MHC_START,
    E3_R1_HI,
    E3_R1_LO,
    E3_R1_TARGET,
    E3_R1_TOLERANCE,
    E3_THORNDIKE_EXPECTED_SINGLE_TRUNC,
    E3_THORNDIKE_Z_TAU,
    E3_TRUNCATION_QUANTILE,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    SEED_E3_BOOT,
    SEED_E3_MC,
    SUB_B_BOOT_N,
    SUB_B_LS_MC_N,
    SUB_B_LS_NULL_RHO,
    SUB_B_LS_NULL_TAU,
    atomic_write_json,
    bh_fdr,
    build_bootstrap_idx,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    load_magma_disorder,
    load_magma_scz,
    load_preds,
    longin_solnik_mc_null,
    partial_pearson,
    rank_gaussianize,
    sha256_file,
    setup_logger,
    thorndike_case_ii_single_trunc,
)

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, pearsonr, norm


def build_shared_frame(logger) -> pd.DataFrame:
    """Assemble ENSGID × [PoPS_Score, MAGMA_Z_SCZ, lof_pLI, CHR, START, END]."""
    preds = load_preds(BATCH054_P05_PREDS)
    magma = load_magma_scz()[["ENSGID", "MAGMA_Z"]]
    gnomad = load_gnomad_per_brief_v2()[["ENSGID", "lof_pLI"]]
    annot = load_gene_annot()[["ENSGID", "CHR", "START", "END"]]
    frame = (
        preds.merge(magma, on="ENSGID", how="inner")
             .merge(gnomad, on="ENSGID", how="inner")
             .merge(annot, on="ENSGID", how="inner")
             .dropna(subset=["PoPS_Score", "MAGMA_Z", "lof_pLI"])
             .drop_duplicates(subset="ENSGID", keep="first")
             .sort_values("ENSGID")
             .reset_index(drop=True)
    )
    logger.info("E3 shared frame: N=%d", len(frame))
    return frame


def mhc_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return boolean mask: True iff gene overlaps chr6:25-34Mb (GRCh37)."""
    chr_ok = frame["CHR"].astype(str) == str(E3_MHC_CHR)
    start_ok = frame["START"].astype(int) < E3_MHC_END
    end_ok = frame["END"].astype(int) > E3_MHC_START
    return (chr_ok & start_ok & end_ok).to_numpy()


def upper_tail_partial_rho(pops: np.ndarray, magma: np.ndarray,
                             pli: np.ndarray,
                             quantile: float = 0.80) -> dict:
    """Upper-tail partial_ρ(PoPS, MAGMA | pLI) at the given MAGMA quantile.

    WHY Gaussianize MAGMA for the tail: iter_058 Sub-B.2 convention — we
    Gaussianize MAGMA-Z so the 80th-percentile threshold maps cleanly to
    norm.ppf(0.80)=0.8416 on the Gaussianized scale.
    """
    magma_g = rank_gaussianize(magma)
    tau = float(norm.ppf(quantile))
    mask = magma_g >= tau
    n = int(mask.sum())
    if n < 10:
        return {"status": "failed", "n": n}
    cov = pli[mask].reshape(-1, 1)
    rho = partial_pearson(pops[mask], magma_g[mask], cov)
    return {
        "status": "ok",
        "n_tail": n,
        "tau": tau,
        "partial_rho": float(rho),
        "mask": mask,
    }


def run_3D_null_integrity(logger) -> dict:
    """3D: Thorndike Case II closed-form + re-run iter_058 MC null.

    Compare |Thorndike - MC| < 0.01 → confirmed. Does NOT gate 3A/3B/3C.
    """
    thorndike = thorndike_case_ii_single_trunc(
        full_rho=E3_FULL_RHO_ANCHOR,
        truncation_quantile=E3_TRUNCATION_QUANTILE,
    )
    expected_closed = float(thorndike["expected_tail_rho"])
    # Re-run iter_058 MC at seed 20260427 (offset +3).
    mc = longin_solnik_mc_null(
        rho=SUB_B_LS_NULL_RHO, tau=SUB_B_LS_NULL_TAU,
        n_mc=SUB_B_LS_MC_N, seed=SEED_E3_MC,
    )
    mc_rho = float(mc.get("tail_rho_mc", float("nan")))
    delta_th_mc = abs(expected_closed - mc_rho) if np.isfinite(mc_rho) else float("nan")
    delta_th_iter058 = abs(expected_closed - E3_ITER058_MC_NULL)
    agreement_ok = bool(np.isfinite(delta_th_mc)
                         and delta_th_mc < E3_MC_AGREEMENT_THRESHOLD)
    logger.info("3D: Thorndike=%.4f iter059_MC=%.4f iter058_MC=%.4f "
                 "|th - MC|=%.4f; agreement=%s",
                 expected_closed, mc_rho, E3_ITER058_MC_NULL,
                 delta_th_mc, agreement_ok)
    return {
        "thorndike_closed_form": thorndike,
        "iter059_mc": mc,
        "iter058_mc_reference": E3_ITER058_MC_NULL,
        "delta_thorndike_vs_iter059_mc": float(delta_th_mc),
        "delta_thorndike_vs_iter058_mc": float(delta_th_iter058),
        "agreement_threshold": E3_MC_AGREEMENT_THRESHOLD,
        "agreement_ok": agreement_ok,
    }


def run_3A_kurtosis(pops: np.ndarray, magma: np.ndarray, logger,
                     seed: int, n_boot: int) -> dict:
    """3A_CEILING_SATURATION: Fisher-Pearson excess kurtosis(PoPS | upper tail).

    WHY Fisher-Pearson: brief_v2 §3 MEASUREMENT (3A) specifies Fisher
    definition (excess kurtosis = raw - 3, zero for normal). scipy's
    `kurtosis(fisher=True)` = Fisher-Pearson excess.

    Middle bin control: [0.40, 0.60] quantile of Gaussianized MAGMA.
    """
    magma_g = rank_gaussianize(magma)
    tau_hi = float(norm.ppf(0.80))
    upper = pops[magma_g >= tau_hi]
    mid_lo = float(np.quantile(magma_g, E3_3A_MIDDLE_QUANTILE_LO))
    mid_hi = float(np.quantile(magma_g, E3_3A_MIDDLE_QUANTILE_HI))
    mid_mask = (magma_g >= mid_lo) & (magma_g <= mid_hi)
    middle = pops[mid_mask]
    if upper.size < 100 or middle.size < 100:
        return {"status": "failed",
                "reason": f"n_upper={upper.size} n_middle={middle.size}"}
    kurt_upper = float(kurtosis(upper, fisher=True, bias=False))
    kurt_middle = float(kurtosis(middle, fisher=True, bias=False))

    # Bootstrap CI on upper-tail kurtosis.
    rng = np.random.default_rng(seed)
    boot_upper = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, upper.size, size=upper.size)
        boot_upper[b] = kurtosis(upper[idx], fisher=True, bias=False)
    ci_lo = float(np.quantile(boot_upper, 0.025))
    ci_hi = float(np.quantile(boot_upper, 0.975))

    # Decision: upper ≤ -0.5 AND middle > -0.2.
    fires = bool(kurt_upper <= E3_3A_KURTOSIS_UPPER_THRESHOLD
                  and kurt_middle > E3_3A_MIDDLE_KURTOSIS_FLOOR)
    logger.info("3A kurtosis: upper=%.3f (95%% CI [%.3f, %.3f]) middle=%.3f "
                 "fires=%s", kurt_upper, ci_lo, ci_hi, kurt_middle, fires)
    return {
        "status": "ok",
        "kurtosis_upper": kurt_upper,
        "kurtosis_upper_ci_lo": ci_lo,
        "kurtosis_upper_ci_hi": ci_hi,
        "kurtosis_middle": kurt_middle,
        "n_upper": int(upper.size),
        "n_middle": int(middle.size),
        "thresholds": {
            "upper_ceiling": E3_3A_KURTOSIS_UPPER_THRESHOLD,
            "middle_floor": E3_3A_MIDDLE_KURTOSIS_FLOOR,
        },
        "fires_3A_ceiling_saturation": fires,
    }


def run_3B_mhc_exclusion(frame: pd.DataFrame, logger,
                           n_boot: int, seed: int) -> dict:
    """3B_MHC_DILUTION: upper-tail ρ excluding MHC region.

    WHY chr6:25-34Mb (GRCh37): brief_v2 §3 MEASUREMENT + design.yaml
    e3.mechanisms.3b_mhc_dilution.mhc_region_grch37. NCBI37.3 gene-loc
    coordinates.

    Δ = ρ_mhc_excluded - ρ_mhc_included. Fires if Δ ≥ max(0.02, 2·SE_boot).
    """
    pops_all = frame["PoPS_Score"].to_numpy(dtype=float)
    magma_all = frame["MAGMA_Z"].to_numpy(dtype=float)
    pli_all = frame["lof_pLI"].to_numpy(dtype=float)
    mhc = mhc_mask(frame)
    n_mhc = int(mhc.sum())
    logger.info("3B: MHC-region genes in frame=%d", n_mhc)

    # Upper-tail ρ including MHC.
    incl = upper_tail_partial_rho(pops_all, magma_all, pli_all,
                                    quantile=E3_TRUNCATION_QUANTILE)
    # Excluding MHC.
    keep = ~mhc
    excl = upper_tail_partial_rho(pops_all[keep], magma_all[keep],
                                    pli_all[keep],
                                    quantile=E3_TRUNCATION_QUANTILE)
    if incl.get("status") != "ok" or excl.get("status") != "ok":
        return {"status": "failed",
                "reason": f"incl={incl.get('status')} excl={excl.get('status')}"}
    rho_incl = incl["partial_rho"]
    rho_excl = excl["partial_rho"]
    delta = float(rho_excl - rho_incl)

    # Bootstrap SE for delta (paired over genes).
    rng = np.random.default_rng(seed)
    boot_delta = np.zeros(n_boot, dtype=float)
    n = frame.shape[0]
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_mhc = mhc[idx]
        sub_pops = pops_all[idx]
        sub_magma = magma_all[idx]
        sub_pli = pli_all[idx]
        i_obj = upper_tail_partial_rho(sub_pops, sub_magma, sub_pli,
                                         quantile=E3_TRUNCATION_QUANTILE)
        # Excluding MHC on this bootstrap index.
        keep_b = ~sub_mhc
        e_obj = upper_tail_partial_rho(sub_pops[keep_b], sub_magma[keep_b],
                                         sub_pli[keep_b],
                                         quantile=E3_TRUNCATION_QUANTILE)
        if i_obj.get("status") == "ok" and e_obj.get("status") == "ok":
            boot_delta[b] = e_obj["partial_rho"] - i_obj["partial_rho"]
        else:
            boot_delta[b] = np.nan
    finite = boot_delta[np.isfinite(boot_delta)]
    se_boot = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
    threshold = max(E3_3B_DELTA_FLOOR, E3_3B_SE_MULTIPLIER * se_boot)
    fires = bool(delta >= threshold)
    logger.info("3B MHC: ρ_incl=%.3f ρ_excl=%.3f Δ=%.3f SE_boot=%.3f "
                 "threshold=%.3f fires=%s",
                 rho_incl, rho_excl, delta, se_boot, threshold, fires)
    return {
        "status": "ok",
        "n_mhc_genes_in_frame": n_mhc,
        "upper_tail_rho_with_mhc": rho_incl,
        "upper_tail_rho_without_mhc": rho_excl,
        "delta": delta,
        "se_bootstrap": se_boot,
        "threshold": float(threshold),
        "ci_lo_delta": (float(np.quantile(finite, 0.025))
                         if finite.size else float("nan")),
        "ci_hi_delta": (float(np.quantile(finite, 0.975))
                         if finite.size else float("nan")),
        "fires_3B_mhc_dilution": fires,
    }


def _bootstrap_se_upper_tail_rho(
    pops: np.ndarray, magma: np.ndarray, pli: np.ndarray,
    n_boot: int, seed: int, quantile: float,
) -> float:
    """Bootstrap SE of upper-tail partial ρ (paired over genes).

    WHY a separate helper: needed for 3C per-disorder z-scoring in Stouffer
    combination. Paired bootstrap over shared ENSGID order preserves the
    tail-subsetting logic exactly.
    """
    rng = np.random.default_rng(seed)
    n = len(pops)
    boot_rhos = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        t = upper_tail_partial_rho(pops[idx], magma[idx], pli[idx],
                                    quantile=quantile)
        boot_rhos[b] = (t["partial_rho"] if t.get("status") == "ok"
                        else np.nan)
    finite = boot_rhos[np.isfinite(boot_rhos)]
    if finite.size < 2:
        return float("nan")
    return float(finite.std(ddof=1))


def run_3C_cross_trait(frame: pd.DataFrame, logger,
                         disorders: list[str], n_boot: int, seed: int
                         ) -> dict:
    """3C_CROSS_TRAIT_GENERIC: upper-tail ρ(SCZ-PoPS, MAGMA_d) vs Thorndike null.

    For each d ∈ {IBD, BIP, Height}:
      1. Build shared frame SCZ-PoPS × MAGMA_d × pLI.
      2. Compute full ρ (for Thorndike anchor) and upper-tail ρ.
      3. Compare upper-tail observed vs Thorndike Case II predicted using
         trait-specific full ρ.
      4. Bootstrap SE of upper-tail ρ (paired over genes).

    Fires if ≥ 2 of 3 disorders have upper_tail_rho - Thorndike_predicted ≤
    E3_3C_DELTA_THRESHOLD (-0.03).

    Combined p-value via Stouffer's Z-method (Stouffer et al. 1949) over
    per-disorder z_d = delta_d / SE_d where delta_d = observed - predicted.
    Tail-below-null is a LOWER-tail alternative, so we compute
    p = norm.sf(-Z_combined) (lower-tail one-sided).
    """
    pops_all = frame["PoPS_Score"].to_numpy(dtype=float)
    results_per_disorder: dict[str, dict] = {}
    below_count = 0
    z_scores: list[float] = []
    for d in disorders:
        try:
            magma_d = load_magma_disorder(d)[["ENSGID", "MAGMA_Z"]]
        except Exception as exc:  # noqa: BLE001
            logger.warning("3C: MAGMA load failed for %s: %s", d, exc)
            results_per_disorder[d] = {"status": "failed",
                                         "reason": str(exc)}
            continue
        # Merge onto frame's ENSGID order.
        merged = frame[["ENSGID", "PoPS_Score", "lof_pLI"]].merge(
            magma_d, on="ENSGID", how="inner"
        ).dropna(subset=["MAGMA_Z"])
        if len(merged) < 500:
            results_per_disorder[d] = {"status": "failed",
                                         "reason": f"merge n={len(merged)}"}
            continue
        pops_m = merged["PoPS_Score"].to_numpy(dtype=float)
        magma_m = merged["MAGMA_Z"].to_numpy(dtype=float)
        pli_m = merged["lof_pLI"].to_numpy(dtype=float)
        # Full partial ρ (for Thorndike anchor).
        full_rho = partial_pearson(pops_m, magma_m, pli_m.reshape(-1, 1))
        # Upper-tail ρ.
        tail = upper_tail_partial_rho(pops_m, magma_m, pli_m,
                                        quantile=E3_TRUNCATION_QUANTILE)
        if tail.get("status") != "ok":
            results_per_disorder[d] = {"status": "failed",
                                         "reason": "tail computation failed"}
            continue
        # Trait-specific Thorndike.
        thorndike = thorndike_case_ii_single_trunc(
            full_rho=float(full_rho),
            truncation_quantile=E3_TRUNCATION_QUANTILE,
        )
        predicted = float(thorndike["expected_tail_rho"])
        observed = float(tail["partial_rho"])
        delta = observed - predicted
        below = delta <= E3_3C_DELTA_THRESHOLD
        if below:
            below_count += 1
        # Bootstrap SE of upper-tail ρ.
        se_d = _bootstrap_se_upper_tail_rho(
            pops_m, magma_m, pli_m, n_boot=n_boot, seed=seed,
            quantile=E3_TRUNCATION_QUANTILE,
        )
        z_d = (float(delta / se_d) if np.isfinite(se_d) and se_d > 0
               else float("nan"))
        logger.info("3C %s: full_rho=%.3f upper=%.3f Thorndike=%.3f Δ=%.3f "
                     "SE_boot=%.3f z=%.3f below_threshold=%s",
                     d, full_rho, observed, predicted, delta,
                     se_d, z_d, below)
        results_per_disorder[d] = {
            "status": "ok",
            "n": int(len(merged)),
            "full_rho": float(full_rho),
            "upper_tail_rho": observed,
            "thorndike_predicted": predicted,
            "delta_observed_minus_thorndike": float(delta),
            "below_threshold_minus_0_03": bool(below),
            "se_bootstrap": float(se_d),
            "z_score": float(z_d),
        }
        if np.isfinite(z_d):
            z_scores.append(z_d)

    fires = bool(below_count >= E3_3C_MIN_DISORDERS_FIRING)

    # M4 audit fix: Stouffer's Z-method combined p-value (Stouffer SA et al.
    # 1949 "The American Soldier" Volume I, p.45). Lower-tail alternative
    # since tail-below-null means z_d = delta / SE is NEGATIVE.
    # If fewer than 2 disorders have finite z_d, the 3C family contribution
    # is marked None (insufficient data) and the caller drops 3C from the
    # BH-FDR family (reduces family from 3 to 2).
    if len(z_scores) >= 2:
        z_combined = float(sum(z_scores) / np.sqrt(len(z_scores)))
        p_3c = float(norm.sf(-z_combined))  # lower-tail one-sided
    else:
        z_combined = float("nan")
        p_3c = None

    return {
        "per_disorder": results_per_disorder,
        "n_disorders_below_threshold": below_count,
        "min_required": E3_3C_MIN_DISORDERS_FIRING,
        "fires_3C_cross_trait_generic": fires,
        "z_scores": z_scores,
        "stouffer_z_combined": z_combined,
        "stouffer_p_combined": p_3c,
        "stouffer_note": (
            "Stouffer 1949 Z-method: Z_combined = sum(z_d) / sqrt(k) where "
            "z_d = (observed - Thorndike) / SE_bootstrap(observed). "
            "One-sided lower-tail p = norm.sf(-Z_combined) since "
            "tail-below-null is the LESS-signal alternative."
        ),
    }


def run_R1_gate(frame: pd.DataFrame, logger) -> dict:
    """R1: SCZ full upper-tail ρ ∈ [0.21, 0.25] reproducing iter_058 0.229."""
    pops = frame["PoPS_Score"].to_numpy(dtype=float)
    magma = frame["MAGMA_Z"].to_numpy(dtype=float)
    pli = frame["lof_pLI"].to_numpy(dtype=float)
    tail = upper_tail_partial_rho(pops, magma, pli,
                                    quantile=E3_TRUNCATION_QUANTILE)
    if tail.get("status") != "ok":
        return {"pass": False, "reason": f"tail fail: {tail}"}
    obs = tail["partial_rho"]
    passes = bool(E3_R1_LO <= obs <= E3_R1_HI)
    logger.info("E3 R1: upper_tail_rho=%.4f in [%.2f, %.2f]? %s",
                 obs, E3_R1_LO, E3_R1_HI, passes)
    return {
        "target": E3_R1_TARGET, "tolerance": E3_R1_TOLERANCE,
        "lo": E3_R1_LO, "hi": E3_R1_HI,
        "observed_upper_tail_rho": obs,
        "pass": passes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_059 E3")
    parser.add_argument("--smoke", action="store_true",
                         help="Smoke: reduce bootstrap draws.")
    parser.add_argument("--smoke-boot", type=int, default=50)
    parser.add_argument("--smoke-mc", type=int, default=10000,
                         help="Smoke: cap MC n for speed.")
    args = parser.parse_args()

    logger = setup_logger("batch_059.e3", LOGS_DIR / "e3.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "e3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build shared SCZ frame.
    frame = build_shared_frame(logger)

    # 3D null integrity (independent).
    d_result = run_3D_null_integrity(logger)
    atomic_write_json(d_result, out_dir / "mc_null.json")

    # R1 reproduction gate.
    r1 = run_R1_gate(frame, logger)
    r1_pass = r1["pass"]

    # If R1 fails, we still compute 3A/3B/3C but classify as 3F_UNINTERPRETABLE
    # at the aggregate level (brief_v2 §3).

    n_boot = args.smoke_boot if args.smoke else SUB_B_BOOT_N

    # 3A kurtosis.
    pops = frame["PoPS_Score"].to_numpy(dtype=float)
    magma = frame["MAGMA_Z"].to_numpy(dtype=float)
    a = run_3A_kurtosis(pops, magma, logger,
                         seed=SEED_E3_BOOT, n_boot=n_boot)

    # 3B MHC-exclusion.
    b = run_3B_mhc_exclusion(frame, logger, n_boot=n_boot, seed=SEED_E3_BOOT)

    # 3C cross-trait.
    disorders_3c = ["ibd_delange2017", "bip", "height"]
    c = run_3C_cross_trait(frame, logger, disorders_3c,
                            n_boot=n_boot, seed=SEED_E3_BOOT)

    # BH-FDR on 3A/3B/3C (single 3-test family).
    # Convert each to a p-value via bootstrap-SE z-scores where possible.
    # 3A: z = (kurt_upper - 0) / bootstrap SE; one-sided for left tail.
    pvals = []
    labels = []
    if a.get("status") == "ok":
        se_a = (a["kurtosis_upper_ci_hi"] - a["kurtosis_upper_ci_lo"]) / (2 * 1.96)
        z_a = (a["kurtosis_upper"] / max(se_a, 1e-9)
               if np.isfinite(se_a) else 0.0)
        p_a = float(norm.cdf(z_a))  # left-tail (platykurtic = negative)
        pvals.append(p_a)
        labels.append("3A_kurtosis")
    if b.get("status") == "ok":
        se_b = b["se_bootstrap"]
        z_b = (b["delta"] / max(se_b, 1e-9)
               if np.isfinite(se_b) and se_b > 0 else 0.0)
        p_b = float(1 - norm.cdf(z_b))  # one-sided upper
        pvals.append(p_b)
        labels.append("3B_mhc_delta")
    # M4 audit fix: 3C combined p-value via Stouffer's Z-method (computed
    # in run_3C_cross_trait). Replace the prior hard-coded 0.05/0.50 proxy.
    # WHY: the Stouffer combination is a real inferential test — sum of
    # per-disorder z-scores scaled by sqrt(k) — rather than a made-up value.
    # If fewer than 2 disorders have finite z_d (insufficient data), we DROP
    # 3C from the BH-FDR family (reduces family size from 3 to 2).
    p_c = c.get("stouffer_p_combined")  # None if insufficient data
    dropped_3c = False
    if p_c is not None and np.isfinite(p_c):
        pvals.append(float(p_c))
        labels.append("3C_cross_trait_stouffer")
    else:
        dropped_3c = True

    qvals = bh_fdr(pvals) if pvals else []
    bh_family = {
        "labels": labels,
        "pvals": pvals,
        "qvals": qvals,
        "q_threshold": BH_Q,
        "family_size": len(pvals),
        "dropped_3c_insufficient_data": bool(dropped_3c),
        "note": (
            "3A p: left-tail z on kurtosis vs 0 using bootstrap-derived SE. "
            "3B p: right-tail z on Δ vs 0 using bootstrap SE. "
            "3C p (M4 audit fix): Stouffer 1949 Z-method combined one-sided "
            "lower-tail p from per-disorder z_d = (tail_rho - Thorndike) / "
            "SE_bootstrap. If fewer than 2 disorders have finite z_d, 3C is "
            "dropped from the family (see dropped_3c_insufficient_data)."
        ),
    }

    # Independent-firing verdicts (3A/3B/3C are NOT first-match).
    if not r1_pass:
        aggregate_verdict = {
            "classification": "3F_UNINTERPRETABLE",
            "reason": f"R1 reproduction gate failed: {r1}",
        }
    else:
        fires_A = bool(a.get("fires_3A_ceiling_saturation", False))
        fires_B = bool(b.get("fires_3B_mhc_dilution", False))
        fires_C = bool(c.get("fires_3C_cross_trait_generic", False))
        if fires_A and fires_B and fires_C:
            cls = "3ABC_ALL_JOINT"
        elif fires_A and fires_B:
            cls = "3AB_JOINT"
        elif fires_B and fires_C:
            cls = "3BC_JOINT"
        elif fires_A and fires_C:
            cls = "3AC_JOINT"
        elif fires_A:
            cls = "3A_CEILING_SATURATION"
        elif fires_B:
            cls = "3B_MHC_DILUTION"
        elif fires_C:
            cls = "3C_CROSS_TRAIT_GENERIC"
        else:
            cls = "3E_INTERMEDIATE"
        aggregate_verdict = {
            "classification": cls,
            "fires_3A": fires_A, "fires_3B": fires_B, "fires_3C": fires_C,
            "reason": (f"Independent firings: 3A={fires_A} 3B={fires_B} "
                        f"3C={fires_C}"),
        }

    provenance = {
        "gnomad_tsv": sha256_file(GNOMAD_TSV),
        "gene_annot": sha256_file(GENE_ANNOT),
        "magma_scz": sha256_file(MAGMA_SCZ_GENES_OUT),
        "pops_preds_p05": sha256_file(BATCH054_P05_PREDS),
    }

    results = {
        "status": "ok",
        "batch": "059", "sub": "e3", "brief": "brief_v2.md (v2.1)",
        "wall_s": time.time() - t0,
        "smoke": bool(args.smoke),
        "n_shared_frame": int(len(frame)),
        "R1_reproduction_gate": r1,
        "3D_null_integrity": d_result,
        "3A_kurtosis": a,
        "3B_mhc_exclusion": b,
        "3C_cross_trait": c,
        "bh_fdr_family_3abc": bh_family,
        "aggregate_verdict": aggregate_verdict,
        "provenance_sha256": provenance,
        "brief_contract": {
            "n_bootstrap": n_boot,
            "seed_bootstrap": SEED_E3_BOOT,
            "seed_mc_null": SEED_E3_MC,
            "truncation_quantile": E3_TRUNCATION_QUANTILE,
            "full_rho_anchor": E3_FULL_RHO_ANCHOR,
            "thorndike_expected": E3_THORNDIKE_EXPECTED_SINGLE_TRUNC,
            "mhc_region_grch37": f"chr{E3_MHC_CHR}:{E3_MHC_START}-{E3_MHC_END}",
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("E3 wrote %s (wall=%.1fs)",
                 out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
