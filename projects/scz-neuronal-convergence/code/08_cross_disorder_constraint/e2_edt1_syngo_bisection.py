#!/usr/bin/env python3
"""batch_059 E2 — EDT1 SynGO-only bisection + poly/rank length covariates +
length-decile-matched permutation null (PER-DISORDER classification).

Implements brief_v2.md §2 EXACTLY.

Overview (WHY):
  iter_058 F058_06 found cross-disorder extension of F147 to EDT1-ex-B3
  (SCZ β=+3.43; AD β=+0.31, IBD β=+0.16). E2 tests whether these are
  length-confound-driven vs PSD-scaffold-concentrated vs truly cross-
  disorder signals by:

    (a) Bisecting EDT1-ex-B3 into 3 SynGO-defined rings (SCAFFOLD_CORE,
        VESICLE_CORE, REMAINING).
    (b) Running the Sub-A v2.1 diagnostic battery (OLS + Huber + Tukey +
        rank-MAGMA + DFBETAS + Cook's) under 3 covariate specifications
        (V1_LINEAR, V2_POLY, V3_RANK).
    (c) Computing a length-decile-matched permutation null (10,000 draws)
        per ring × disorder.
    (d) Classifying PER-DISORDER × PER-RING (no cross-disorder first-match,
        per v2 critic 3 C2 fix).

Outputs:
  experiments/batch_059/output/e2/results.json
  experiments/batch_059/output/e2/per_disorder_per_ring_verdicts.json
  experiments/batch_059/output/e2/length_matched_null.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    B3_GENES,
    BH_Q,
    DFBETAS_CUTOFF,
    DISORDERS,
    E2_COVS_V1_LINEAR,
    E2_COVS_V2_POLY,
    E2_COVS_V3_RANK,
    E2_FERNANDEZ_2009_HUBS,
    E2_GO_SCAFFOLD,
    E2_GO_VESICLE,
    E2_LENGTH_ARTIFACT_EFFECT_MAX,
    E2_LENGTH_ARTIFACT_REL_DROP,
    E2_LENGTH_MASKING_REL_RISE,
    E2_LENGTH_N_DECILES,
    E2_LENGTH_N_DRAWS,
    E2_LENGTH_P_THRESHOLD,
    E2_MIN_N_REMAINING,
    E2_MIN_N_SCAFFOLD,
    E2_MIN_N_VESICLE,
    E2_R1_HI,
    E2_R1_LO,
    E2_R1_TARGET,
    E2_SCAFFOLD_CONCENTRATED_GAP,
    E2_SCAFFOLD_INTERMEDIATE_HI,
    E2_SCAFFOLD_INTERMEDIATE_LO,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_GENELOC,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    PGC3_XLSX,
    SEED_E2_BOOT,
    SEED_E2_PERM,
    SUB_C_MAPPING_GATE,
    atomic_write_json,
    bh_fdr,
    build_sub_a_frame,
    classify_disorder_v2,
    compute_dfbetas_cooks,
    fit_tukey_biweight,
    load_edt1,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    load_syngo_go_terms,
    rank_gaussianize,
    setup_logger,
    sha256_file,
    symbols_to_ensgids,
)

import numpy as np
import pandas as pd
import statsmodels.api as sm


# We re-use batch_058 Sub-A fit helpers (Rule 1: no reinvention).
BATCH058_SCRIPTS = (
    Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
    / "experiments" / "batch_058" / "scripts"
)
sys.path.insert(0, str(BATCH058_SCRIPTS))
from sub_a_robust_battery import (  # noqa: E402
    fit_huber, fit_ols, fit_rank_magma_ols,
    influential_outlier_reconciliation,
)


MIN_N_UNIVERSE = 15000  # same as batch_058


def add_length_polynomials(frame: pd.DataFrame) -> pd.DataFrame:
    """Add log10_gene_length_sq, log10_gene_length_cu, rank_pct_length.

    WHY: brief_v2 §2 MEASUREMENT covariate specs V2_POLY and V3_RANK need
    these derived columns on the regression frame.
    """
    df = frame.copy()
    L = df["log10_gene_length"].to_numpy(dtype=float)
    df["log10_gene_length_sq"] = L * L
    df["log10_gene_length_cu"] = L * L * L
    from scipy.stats import rankdata
    r = rankdata(L, method="average")
    df["rank_pct_length"] = (r - 0.5) / len(r)
    return df


def run_disorder_covs(disorder: str, gnomad: pd.DataFrame,
                       annot: pd.DataFrame, gene_set_ensg: set[str],
                       logger,
                       smoke_frame_size: int = 0) -> dict:
    """Run v2.1 diagnostic battery for one disorder under 3 covariate specs.

    Returns {spec_name: battery_result} plus raw frame for downstream
    length-matched null (caller gets it back).
    """
    try:
        frame = build_sub_a_frame(disorder, gnomad, annot, gene_set_ensg,
                                    gene_set_col="in_set")
    except Exception as exc:  # noqa: BLE001
        logger.exception("build_sub_a_frame failed for %s", disorder)
        return {"status": "failed", "reason": str(exc)}
    frame = add_length_polynomials(frame)
    if smoke_frame_size and len(frame) > smoke_frame_size:
        rng = np.random.default_rng(SEED_E2_BOOT)
        in_set_rows = frame[frame["in_set"] == 1]
        out_rows = frame[frame["in_set"] == 0]
        take = max(0, smoke_frame_size - len(in_set_rows))
        if take < len(out_rows):
            pick_idx = rng.choice(len(out_rows), size=take, replace=False)
            out_rows = out_rows.iloc[np.sort(pick_idx)]
        frame = pd.concat([in_set_rows, out_rows], axis=0
                            ).sort_values("ENSGID").reset_index(drop=True)
    n = len(frame)
    n_in = int(frame["in_set"].sum())
    logger.info("  %s: n=%d in_set=%d", disorder, n, n_in)
    min_n = 100 if smoke_frame_size else MIN_N_UNIVERSE
    if n < min_n or n_in < 10:
        return {"status": "failed",
                "reason": f"n={n} in_set={n_in}"}

    per_spec: dict[str, dict] = {}
    for spec_name, covs in [
        ("v1_linear", E2_COVS_V1_LINEAR),
        ("v2_poly", E2_COVS_V2_POLY),
        ("v3_rank", E2_COVS_V3_RANK),
    ]:
        ols = fit_ols(frame, "in_set", covs)
        huber = fit_huber(frame, "in_set", covs)
        tukey = fit_tukey_biweight(frame, covs, "in_set")
        rank_ols = fit_rank_magma_ols(frame, "in_set", covs)
        infl = compute_dfbetas_cooks(frame, covs, "in_set")
        recon = influential_outlier_reconciliation(frame, covs, "in_set", infl)
        per_spec[spec_name] = {
            "covariates": covs,
            "ols": ols, "huber": huber, "tukey": tukey,
            "rank_magma_ols": rank_ols,
            "influence": infl,
            "influential_outlier_reconciliation": recon,
        }
    return {
        "status": "ok",
        "n_gene_universe": n,
        "n_set_in_universe": n_in,
        "per_spec": per_spec,
        "frame": frame,  # not JSON-serialized; used for length-matched null
    }


def length_matched_permutation_null(
    frame: pd.DataFrame, n_draws: int, seed: int,
    covs: list[str], logger, deciles: int = 10,
) -> dict:
    """Length-decile-matched permutation null (RING-MATCHED per-decile counts).

    WHY ring-matched (B4 audit fix): brief_v2 §2 MEASUREMENT specifies that
    the null must preserve the RING's OWN per-decile length distribution,
    not the background's decile sizes. The prior implementation drew
    `rng.multinomial(ring_size, proportions)` where `proportions` reflected
    full-background decile sizes — which made the null a random subsample of
    the background rather than a length-matched draw. That would give a
    trivial null (mean β near 0) rather than the intended biological null,
    and could have materially inflated apparent SCAFFOLD_CONCENTRATED
    verdict confidence. Fix: for each decile d, sample EXACTLY
    `ring_per_decile[d]` genes from the BACKGROUND (~ring) in that decile,
    without replacement within a single draw, with replacement across draws.

    Args:
      frame: must include `log10_gene_length`, `MAGMA_Z`, `in_set` (0/1), and
        all covariates in `covs`. The ring is the `in_set==1` subset.
      covs: covariate list for the OLS fit.

    Returns dict with betas array and summary stats.
    """
    rng = np.random.default_rng(seed)
    L = frame["log10_gene_length"].to_numpy(dtype=float)
    in_set_arr = frame["in_set"].to_numpy(dtype=int)
    ring_bool = in_set_arr.astype(bool)
    n = len(L)
    ring_size = int(ring_bool.sum())

    # Decile bin IDs over the full frame, via qcut to guarantee equal-count
    # bins (Rule 1: use pandas' built-in rather than hand-rolling quantile
    # edges + digitize). duplicates="drop" guards against tied lengths.
    bin_ids_series = pd.qcut(L, deciles, labels=False, duplicates="drop")
    bin_ids = np.asarray(bin_ids_series, dtype=float)
    # qcut returns NaN for any positions that fall outside bins (shouldn't
    # happen since we pass the full array, but guard anyway).
    if np.any(~np.isfinite(bin_ids)):
        raise RuntimeError(
            "length_matched_permutation_null: qcut produced NaN bin_ids; "
            "check log10_gene_length for NaN/Inf."
        )
    bin_ids = bin_ids.astype(int)
    n_bins = int(bin_ids.max()) + 1

    # Ring's per-decile count distribution.
    ring_per_decile = np.bincount(bin_ids[ring_bool], minlength=n_bins)

    # Background = NOT in ring. Pre-compute per-decile background indices.
    bg_mask = ~ring_bool
    decile_bg_indices: dict[int, np.ndarray] = {
        d: np.where(bg_mask & (bin_ids == d))[0] for d in range(n_bins)
    }

    # Log the match configuration for auditor traceability.
    logger.info(
        "  length-matched null (ring-matched): %d draws × ring_size=%d over "
        "%d bins; ring_per_decile=%s; bg_per_decile=%s",
        n_draws, ring_size, n_bins,
        ring_per_decile.tolist(),
        [int(len(decile_bg_indices[d])) for d in range(n_bins)],
    )

    # Diagnostic: flag deciles where background has fewer genes than the ring
    # needs — these require sampling with replacement, which inflates var.
    insufficient_bg: list[int] = []
    for d in range(n_bins):
        n_need = int(ring_per_decile[d])
        n_have = int(len(decile_bg_indices[d]))
        if n_need > 0 and n_have < n_need:
            insufficient_bg.append(d)
    if insufficient_bg:
        logger.info(
            "  length-matched null: %d deciles have fewer background genes "
            "than needed (will sample with replacement): %s",
            len(insufficient_bg), insufficient_bg,
        )

    ols_betas = np.zeros(n_draws, dtype=float)
    ses = np.zeros(n_draws, dtype=float)
    # Pre-extract covariates once (float arrays).
    X_covs = frame[covs].to_numpy(dtype=float)
    y = frame["MAGMA_Z"].to_numpy(dtype=float)

    for di in range(n_draws):
        picks: list[int] = []
        for d in range(n_bins):
            n_needed = int(ring_per_decile[d])
            if n_needed == 0:
                continue
            bg_d = decile_bg_indices[d]
            if len(bg_d) == 0:
                # No background genes in this decile — cannot sample.
                continue
            if len(bg_d) < n_needed:
                # Insufficient background; sample with replacement (conservative).
                sampled = rng.choice(bg_d, size=n_needed, replace=True)
            else:
                sampled = rng.choice(bg_d, size=n_needed, replace=False)
            picks.extend(sampled.tolist())
        if len(picks) < 10:
            ols_betas[di] = np.nan
            ses[di] = np.nan
            continue
        in_set = np.zeros(n, dtype=int)
        in_set[picks] = 1
        X = np.column_stack([in_set, X_covs])
        Xc = sm.add_constant(X, has_constant="add")
        try:
            fit = sm.OLS(y, Xc).fit()
            ols_betas[di] = float(fit.params[1])
            ses[di] = float(fit.bse[1])
        except Exception:
            ols_betas[di] = np.nan
            ses[di] = np.nan
        if (di + 1) % 1000 == 0:
            logger.info("    length-null draw %d/%d", di + 1, n_draws)

    finite = ols_betas[np.isfinite(ols_betas)]
    return {
        "n_draws": int(n_draws),
        "ring_size": int(ring_size),
        "betas": ols_betas,
        "n_finite": int(finite.size),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        "q05": float(np.quantile(finite, 0.05)) if finite.size else float("nan"),
        "q50": float(np.quantile(finite, 0.50)) if finite.size else float("nan"),
        "q95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
        "ring_per_decile": ring_per_decile.tolist(),
        "n_bins": int(n_bins),
        "insufficient_bg_deciles": insufficient_bg,
    }


def classify_per_disorder_per_ring(
    spec_results: dict[str, dict],
    beta_remaining_per_spec: dict[str, float] | None,
    se_remaining_per_spec: dict[str, float] | None,
    perm_p_length: float | None,
    is_ring_core: bool,
) -> dict:
    """Apply E2 per-cell decision rule (brief_v2 §2 PREDICTION).

    Orthogonal verdicts: SCAFFOLD_CONCENTRATED / SCAFFOLD_INTERMEDIATE is
    about gap vs REMAINING; LENGTH_ARTIFACT / LENGTH_MASKING is about
    V3_RANK vs V1_LINEAR. Both can fire for the same d×r cell and are
    reported separately.
    """
    if spec_results.get("status") != "ok":
        return {"verdict": "UNINTERPRETABLE",
                "reason": "battery failed"}

    per_spec = spec_results["per_spec"]
    v1 = per_spec.get("v1_linear", {})
    v3 = per_spec.get("v3_rank", {})
    ols_v1 = v1.get("ols", {})
    ols_v3 = v3.get("ols", {})

    beta_v1 = float(ols_v1.get("beta_1", float("nan")))
    beta_v3 = float(ols_v3.get("beta_1", float("nan")))
    se_v1 = float(ols_v1.get("se_1", float("nan")))
    ci_lo_v1 = float(ols_v1.get("ci_lo", float("nan")))
    ci_hi_v1 = float(ols_v1.get("ci_hi", float("nan")))

    verdicts: list[str] = []
    reasons: list[str] = []

    # SCAFFOLD verdicts only apply to the core ring(s) vs REMAINING baseline.
    if is_ring_core and beta_remaining_per_spec is not None:
        beta_r = beta_remaining_per_spec.get("v1_linear", float("nan"))
        se_r = (se_remaining_per_spec.get("v1_linear", float("nan"))
                if se_remaining_per_spec else float("nan"))
        if np.isfinite(beta_v1) and np.isfinite(beta_r):
            gap = beta_v1 - beta_r
            # CI-disjoint check.
            ci_disjoint = False
            if (np.isfinite(ci_lo_v1) and np.isfinite(ci_hi_v1)
                    and np.isfinite(se_r)):
                ci_lo_r = beta_r - 1.96 * se_r
                ci_hi_r = beta_r + 1.96 * se_r
                ci_disjoint = (ci_lo_v1 > ci_hi_r) or (ci_hi_v1 < ci_lo_r)
            p_len_ok = (perm_p_length is not None
                         and perm_p_length < E2_LENGTH_P_THRESHOLD)
            if gap > E2_SCAFFOLD_CONCENTRATED_GAP and ci_disjoint and p_len_ok:
                verdicts.append("SCAFFOLD_CONCENTRATED")
                reasons.append(
                    f"β_core - β_remaining={gap:.3f} > "
                    f"{E2_SCAFFOLD_CONCENTRATED_GAP} AND CIs disjoint AND "
                    f"p_length_matched={perm_p_length} < "
                    f"{E2_LENGTH_P_THRESHOLD}"
                )
            elif (E2_SCAFFOLD_INTERMEDIATE_LO <= gap
                  <= E2_SCAFFOLD_INTERMEDIATE_HI and not ci_disjoint):
                verdicts.append("SCAFFOLD_INTERMEDIATE")
                reasons.append(
                    f"β_core - β_remaining={gap:.3f} in "
                    f"[{E2_SCAFFOLD_INTERMEDIATE_LO}, "
                    f"{E2_SCAFFOLD_INTERMEDIATE_HI}] AND CIs overlap"
                )

    # LENGTH verdicts run independently, on ALL rings.
    if np.isfinite(beta_v1) and np.isfinite(beta_v3):
        abs_v1 = abs(beta_v1)
        drop = beta_v1 - beta_v3
        # LENGTH_ARTIFACT: |drop| > 0.50*|β_v1| AND |β_v3| < 0.10.
        if (abs(drop) > E2_LENGTH_ARTIFACT_REL_DROP * abs_v1
                and abs(beta_v3) < E2_LENGTH_ARTIFACT_EFFECT_MAX):
            verdicts.append("LENGTH_ARTIFACT")
            reasons.append(
                f"|β_v1 - β_v3|={abs(drop):.3f} > "
                f"{E2_LENGTH_ARTIFACT_REL_DROP}·|β_v1|={E2_LENGTH_ARTIFACT_REL_DROP*abs_v1:.3f} "
                f"AND |β_v3|={abs(beta_v3):.3f} < "
                f"{E2_LENGTH_ARTIFACT_EFFECT_MAX}"
            )
        # LENGTH_MASKING_UNANTICIPATED: β_v3 > β_v1 + 0.30·|β_v1|.
        if beta_v3 > beta_v1 + E2_LENGTH_MASKING_REL_RISE * abs_v1:
            verdicts.append("LENGTH_MASKING_UNANTICIPATED")
            reasons.append(
                f"β_v3={beta_v3:.3f} > β_v1={beta_v1:.3f} + "
                f"{E2_LENGTH_MASKING_REL_RISE}·|β_v1|={E2_LENGTH_MASKING_REL_RISE*abs_v1:.3f}; "
                "L058_03 DESCRIPTIVE-ONLY"
            )

    if not verdicts:
        verdicts.append("INTERMEDIATE")
        reasons.append("No E2 pattern matched thresholds.")

    return {
        "verdict": "+".join(verdicts),
        "verdicts": verdicts,
        "reasons": reasons,
        "beta_v1": beta_v1,
        "beta_v3": beta_v3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_059 E2")
    parser.add_argument("--smoke", action="store_true",
                         help="Smoke: reduce disorders + length-null draws + "
                              "frame size.")
    parser.add_argument("--smoke-frame-size", type=int, default=1500)
    parser.add_argument("--smoke-length-draws", type=int, default=100)
    args = parser.parse_args()
    smoke_frame = args.smoke_frame_size if args.smoke else 0

    logger = setup_logger("batch_059.e2", LOGS_DIR / "e2.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "e2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------- Gene-set construction -------
    gnomad = load_gnomad_per_brief_v2()
    annot = load_gene_annot()

    edt1_symbols = load_edt1()
    b3 = set(B3_GENES)
    edt1_ex_b3_sym = edt1_symbols - b3
    logger.info("EDT1=%d; B3=%d; EDT1-ex-B3=%d",
                 len(edt1_symbols), len(b3), len(edt1_ex_b3_sym))

    # SynGO rings.
    scaffold_syms = load_syngo_go_terms(E2_GO_SCAFFOLD, logger=logger)
    vesicle_syms = load_syngo_go_terms(E2_GO_VESICLE, logger=logger)
    scaffold_sym_ring = scaffold_syms & edt1_ex_b3_sym
    vesicle_sym_ring = (vesicle_syms & edt1_ex_b3_sym) - scaffold_sym_ring
    remaining_sym = edt1_ex_b3_sym - scaffold_sym_ring - vesicle_sym_ring
    logger.info(
        "Rings (symbols): scaffold_core=%d vesicle_core=%d remaining=%d",
        len(scaffold_sym_ring), len(vesicle_sym_ring), len(remaining_sym),
    )

    # Fernández SECONDARY set (descriptive-only). Build superset for audit.
    fernandez_scaffold_sym = scaffold_sym_ring | (
        set(E2_FERNANDEZ_2009_HUBS) & edt1_ex_b3_sym
    )

    # Map to ENSGID.
    scaffold_ensg, scaffold_map = symbols_to_ensgids(scaffold_sym_ring)
    vesicle_ensg, vesicle_map = symbols_to_ensgids(vesicle_sym_ring)
    remaining_ensg, remaining_map = symbols_to_ensgids(remaining_sym)
    fernandez_ensg, _ = symbols_to_ensgids(fernandez_scaffold_sym)

    ring_counts = {
        "scaffold_core_primary": len(scaffold_ensg),
        "vesicle_core_primary": len(vesicle_ensg),
        "remaining": len(remaining_ensg),
        "scaffold_core_secondary": len(fernandez_ensg),
    }
    logger.info("Rings (ENSGID): %s", ring_counts)

    # UNINTERPRETABLE floors (brief_v2 §2).
    floors_fail: dict[str, str] = {}
    if ring_counts["scaffold_core_primary"] < E2_MIN_N_SCAFFOLD:
        floors_fail["scaffold_core_primary"] = (
            f"n={ring_counts['scaffold_core_primary']} < {E2_MIN_N_SCAFFOLD}"
        )
    if ring_counts["vesicle_core_primary"] < E2_MIN_N_VESICLE:
        floors_fail["vesicle_core_primary"] = (
            f"n={ring_counts['vesicle_core_primary']} < {E2_MIN_N_VESICLE}"
        )
    if ring_counts["remaining"] < E2_MIN_N_REMAINING:
        floors_fail["remaining"] = (
            f"n={ring_counts['remaining']} < {E2_MIN_N_REMAINING}"
        )

    # ------- Run battery per ring × disorder -------
    disorders_to_run = (DISORDERS if not args.smoke
                         else ["scz", "alzheimers"])
    rings_to_run = {
        "scaffold_core_primary": scaffold_ensg,
        "vesicle_core_primary": vesicle_ensg,
        "remaining": remaining_ensg,
    }

    per_ring_per_disorder: dict[str, dict] = {}
    for ring_name, ring_ensg in rings_to_run.items():
        logger.info("=== Ring: %s (n=%d) ===", ring_name, len(ring_ensg))
        if ring_name in floors_fail:
            logger.info("  ring below UNINTERPRETABLE floor: %s",
                         floors_fail[ring_name])
        per_ring_per_disorder[ring_name] = {}
        for d in disorders_to_run:
            item = run_disorder_covs(d, gnomad, annot, ring_ensg, logger,
                                       smoke_frame_size=smoke_frame)
            per_ring_per_disorder[ring_name][d] = item

    # R1 reproduction gate: SCZ × scaffold_core ring's EDT1-ex-B3-under-V1.
    # But R1 is formally defined on EDT1-ex-B3 SCZ β at V1_LINEAR. Run once.
    edt1_ex_b3_ensg, _ = symbols_to_ensgids(edt1_ex_b3_sym)
    logger.info("R1 check: running SCZ × EDT1-ex-B3 × V1_LINEAR")
    r1_item = run_disorder_covs("scz", gnomad, annot, edt1_ex_b3_ensg, logger,
                                  smoke_frame_size=smoke_frame)
    r1_scz_beta = float("nan")
    if r1_item.get("status") == "ok":
        r1_scz_beta = float(r1_item["per_spec"]["v1_linear"]["ols"]["beta_1"])
    r1_pass = bool(np.isfinite(r1_scz_beta)
                    and E2_R1_LO <= r1_scz_beta <= E2_R1_HI)
    logger.info("E2 R1 EDT1-ex-B3 SCZ V1 β=%.3f in [%.2f, %.2f]? %s",
                 r1_scz_beta, E2_R1_LO, E2_R1_HI, r1_pass)

    # ------- Length-matched null per ring × disorder -------
    # We compute the null using frame's MAGMA_Z as the outcome. For each
    # ring×disorder, sample from the same disorder's frame.
    n_draws = (args.smoke_length_draws if args.smoke else E2_LENGTH_N_DRAWS)
    length_null: dict[str, dict] = {}
    length_null_arrays: dict[str, np.ndarray] = {}
    length_null_p_by_cell: dict[str, dict[str, float]] = {}
    # M1 audit fix: use sha256 of the cell key for deterministic per-cell
    # seed offsets. Python's built-in hash() is randomized per-process
    # (PYTHONHASHSEED), so `abs(hash(key)) % 10_000_000` was non-deterministic
    # across runs. sha256 is deterministic and collision-free for our use.
    import hashlib as _hashlib
    for ring_name, ring_ensg in rings_to_run.items():
        length_null_p_by_cell[ring_name] = {}
        for d in disorders_to_run:
            item = per_ring_per_disorder[ring_name][d]
            if item.get("status") != "ok":
                length_null_p_by_cell[ring_name][d] = float("nan")
                continue
            frame = item["frame"]
            ring_size = int(frame["in_set"].sum())
            if ring_size < 10:
                length_null_p_by_cell[ring_name][d] = float("nan")
                continue
            key = f"{ring_name}__{d}"
            # Deterministic seed via sha256 first 8 hex digits (32 bits).
            cell_seed = (
                SEED_E2_PERM
                + int(_hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                % 10_000_000
            )
            null = length_matched_permutation_null(
                frame=frame, n_draws=n_draws,
                seed=cell_seed, covs=E2_COVS_V1_LINEAR, logger=logger,
                deciles=E2_LENGTH_N_DECILES,
            )
            # Empirical p (upper): fraction of null β >= observed.
            obs_beta = float(item["per_spec"]["v1_linear"]["ols"]["beta_1"])
            finite = null["betas"][np.isfinite(null["betas"])]
            p_emp = (float((finite >= obs_beta).mean())
                     if finite.size else float("nan"))
            length_null_p_by_cell[ring_name][d] = p_emp
            length_null[key] = {
                **{k: v for k, v in null.items() if k != "betas"},
                "observed_beta_v1": obs_beta,
                "p_empirical_upper": p_emp,
            }
            length_null_arrays[key] = null["betas"]

    # Save length null arrays.
    np.savez_compressed(
        out_dir / "length_matched_null.npz",
        **length_null_arrays,
    )

    # ------- BH-FDR families (brief_v2 §2 MEASUREMENT + §5) -------
    # Family 1: per covariate spec × 3 rings × 8 disorders = 24 per spec; 72 total.
    bh_pvals_by_family: dict[str, list[float]] = {}
    bh_labels_by_family: dict[str, list[str]] = {}
    for spec_name in ("v1_linear", "v2_poly", "v3_rank"):
        pvals = []
        labels = []
        for ring_name in rings_to_run:
            for d in disorders_to_run:
                item = per_ring_per_disorder[ring_name][d]
                if item.get("status") != "ok":
                    continue
                p = float(
                    item["per_spec"][spec_name]["ols"].get(
                        "p_one_sided", float("nan")
                    )
                )
                pvals.append(p)
                labels.append(f"{ring_name}__{d}__{spec_name}")
        bh_pvals_by_family[spec_name] = pvals
        bh_labels_by_family[spec_name] = labels

    qvals_by_family: dict[str, list[float]] = {}
    for spec, pvals in bh_pvals_by_family.items():
        qvals_by_family[spec] = bh_fdr(pvals)

    # Family 2: 24 length-matched-null p-values single family.
    length_pvals: list[float] = []
    length_labels: list[str] = []
    for ring_name in rings_to_run:
        for d in disorders_to_run:
            p = length_null_p_by_cell[ring_name].get(d, float("nan"))
            length_pvals.append(p)
            length_labels.append(f"{ring_name}__{d}")
    length_qvals = bh_fdr(
        [p if np.isfinite(p) else 1.0 for p in length_pvals]
    )

    # ------- Per-cell verdicts -------
    # First, get per-spec REMAINING ring β as the baseline.
    beta_remaining_per_disorder: dict[str, dict[str, float]] = {}
    se_remaining_per_disorder: dict[str, dict[str, float]] = {}
    for d in disorders_to_run:
        rem_item = per_ring_per_disorder["remaining"][d]
        if rem_item.get("status") == "ok":
            beta_remaining_per_disorder[d] = {
                spec: float(rem_item["per_spec"][spec]["ols"].get(
                    "beta_1", float("nan")
                ))
                for spec in ("v1_linear", "v2_poly", "v3_rank")
            }
            se_remaining_per_disorder[d] = {
                spec: float(rem_item["per_spec"][spec]["ols"].get(
                    "se_1", float("nan")
                ))
                for spec in ("v1_linear", "v2_poly", "v3_rank")
            }
        else:
            beta_remaining_per_disorder[d] = {}
            se_remaining_per_disorder[d] = {}

    per_cell_verdicts: dict[str, dict] = {}
    for ring_name in rings_to_run:
        for d in disorders_to_run:
            item = per_ring_per_disorder[ring_name][d]
            perm_p = length_null_p_by_cell[ring_name].get(d, float("nan"))
            perm_p_f = perm_p if np.isfinite(perm_p) else None
            is_core = ring_name in ("scaffold_core_primary",
                                     "vesicle_core_primary")
            # UNINTERPRETABLE gate: ring below floor or R1 fail.
            if ring_name in floors_fail or not r1_pass:
                per_cell_verdicts[f"{ring_name}__{d}"] = {
                    "verdict": "UNINTERPRETABLE",
                    "verdicts": ["UNINTERPRETABLE"],
                    "reasons": [
                        (f"ring below floor: {floors_fail.get(ring_name)}"
                         if ring_name in floors_fail
                         else f"R1 failed: SCZ β={r1_scz_beta}")
                    ],
                }
                continue
            verdict = classify_per_disorder_per_ring(
                item,
                beta_remaining_per_spec=beta_remaining_per_disorder.get(d),
                se_remaining_per_spec=se_remaining_per_disorder.get(d),
                perm_p_length=perm_p_f,
                is_ring_core=is_core,
            )
            per_cell_verdicts[f"{ring_name}__{d}"] = verdict

    # Strip non-JSON-serializable frames from results.
    for ring_name in rings_to_run:
        for d in disorders_to_run:
            item = per_ring_per_disorder[ring_name][d]
            if "frame" in item:
                del item["frame"]

    atomic_write_json(per_cell_verdicts,
                       out_dir / "per_disorder_per_ring_verdicts.json")

    provenance = {
        "gnomad_tsv": sha256_file(GNOMAD_TSV),
        "gene_annot": sha256_file(GENE_ANNOT),
        "magma_geneloc": sha256_file(MAGMA_GENELOC),
        "magma_scz": sha256_file(MAGMA_SCZ_GENES_OUT),
        "pgc3_xlsx": sha256_file(PGC3_XLSX),
        "syngo_gmt": sha256_file(
            Path(__file__).resolve().parents[2]
            / "batch_052_A" / "input" / "syngo_2024.gmt"
        ),
    }

    results = {
        "status": "ok",
        "batch": "059", "sub": "e2", "brief": "brief_v2.md (v2.1)",
        "wall_s": time.time() - t0,
        "smoke": bool(args.smoke),
        "ring_counts": ring_counts,
        "ring_floors_fail": floors_fail,
        "disorders": disorders_to_run,
        "R1_reproduction_gate": {
            "target": E2_R1_TARGET, "lo": E2_R1_LO, "hi": E2_R1_HI,
            "edt1_ex_b3_scz_beta_v1_linear": r1_scz_beta,
            "pass": r1_pass,
        },
        "per_ring_per_disorder": per_ring_per_disorder,
        "per_cell_verdicts": per_cell_verdicts,
        "length_matched_null_summary": length_null,
        "length_null_p_by_cell": length_null_p_by_cell,
        "bh_fdr_family_per_covariate_spec": {
            spec: {
                "labels": bh_labels_by_family[spec],
                "pvals": bh_pvals_by_family[spec],
                "qvals": qvals_by_family[spec],
                "family_size": len(bh_pvals_by_family[spec]),
                "q_threshold": BH_Q,
            } for spec in bh_pvals_by_family
        },
        "bh_fdr_family_length_matched": {
            "labels": length_labels,
            "pvals": length_pvals,
            "qvals": length_qvals,
            "family_size": len(length_pvals),
            "q_threshold": BH_Q,
        },
        "provenance_sha256": provenance,
        "brief_contract": {
            "n_length_null_draws": n_draws,
            "n_length_deciles": E2_LENGTH_N_DECILES,
            "seed_bootstrap": SEED_E2_BOOT,
            "seed_permutation": SEED_E2_PERM,
            "scaffold_concentrated_gap": E2_SCAFFOLD_CONCENTRATED_GAP,
            "length_artifact_rel_drop": E2_LENGTH_ARTIFACT_REL_DROP,
            "length_artifact_effect_max": E2_LENGTH_ARTIFACT_EFFECT_MAX,
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("E2 wrote %s (wall=%.1fs)",
                 out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
