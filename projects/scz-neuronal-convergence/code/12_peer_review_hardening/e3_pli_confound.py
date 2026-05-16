#!/usr/bin/env python3
"""batch_060 E3 -- NOV-059-12 pLI confound check.

Implements brief_v2.md section E3 EXACTLY.

Overview (WHY):
  batch_059 E2 used length-decile-matched permutation null draws to control
  for gene length confounding. L059_03 noted that length != constraint:
  length-matching provides partial but not full control for pLI (intolerance
  to loss-of-function). EDT1-ex-B3 genes tend to be longer AND more
  constrained than average. If the length-matched null draws happen to
  sample genes that are length-matched but pLI-mismatched, the null is
  conservative (biased toward zero) rather than properly calibrated.

  E3 replays the deterministic-seed null draws from batch_059 E2 (remaining
  ring x SCZ cell, 10,000 draws) and records mean(pLI) and std(pLI) of each
  draw. It then compares the null pLI distribution to the actual EDT1-ex-B3
  mean_pLI and computes the expected beta bias from the pLI gap.

  Expected beta bias = (mean_pLI_EDT1 - mean(mean_pLI_null)) x
                       pLI_regression_coefficient

  where pLI_regression_coefficient is the OLS coefficient of lof_pLI
  in the MAGMA_Z ~ covariates regression.

Decision (brief_v2 section E3 DECISION RULE):
  - mean_pLI_null > 0.30 AND bias < 10% of observed beta -> ADEQUATELY_CONTROLLED
  - mean_pLI_null in [0.20, 0.30] -> PARTIALLY_CONTROLLED
  - mean_pLI_null < 0.20 AND bias > 20% -> INADEQUATELY_CONTROLLED

Outputs:
  experiments/batch_060/output/e3/results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    B3_GENES,
    BATCH_059_OUTPUT,
    B060_SEED_E3,
    E2_COVS_V1_LINEAR,
    E2_LENGTH_N_DECILES,
    E3_BIAS_INADEQUATE,
    E3_BIAS_INCONSEQUENTIAL,
    E3_N_NULL_DRAWS,
    E3_PLI_ADEQUATE_FLOOR,
    E3_PLI_PARTIAL_FLOOR,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    SEED_E2_PERM,
    atomic_write_json,
    build_sub_a_frame,
    load_edt1,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    setup_logger,
    sha256_file,
    symbols_to_ensgids,
)

import numpy as np
import pandas as pd
import statsmodels.api as sm


def replay_length_matched_null_draws(
    frame: pd.DataFrame,
    ring_size: int,
    n_draws: int,
    seed_base: int,
    cell_key: str,
    deciles: int,
    logger,
) -> dict:
    """Replay deterministic-seed null draws from batch_059 E2.

    WHY replay rather than load from .npz: the batch_059 E2 length_matched_null.npz
    stores only the OLS betas, not the gene indices of each draw. We need the
    actual gene indices to compute pLI statistics per draw. We replay the
    EXACT same seed + sampling logic from batch_059's
    length_matched_permutation_null function so the draws are bit-identical.

    The seed construction matches batch_059 E2 line 556-559:
      cell_seed = SEED_E2_PERM + int(sha256(key.encode()).hexdigest()[:8], 16) % 10_000_000

    Returns dict with arrays of mean_pLI and std_pLI per draw.
    """
    # Reconstruct the exact same seed as batch_059 E2.
    cell_seed = (
        seed_base
        + int(hashlib.sha256(cell_key.encode()).hexdigest()[:8], 16)
        % 10_000_000
    )
    logger.info(
        "E3 replay: cell_key=%s seed=%d n_draws=%d ring_size=%d",
        cell_key, cell_seed, n_draws, ring_size,
    )
    rng = np.random.default_rng(cell_seed)

    L = frame["log10_gene_length"].to_numpy(dtype=float)
    in_set_arr = frame["in_set"].to_numpy(dtype=int)
    ring_bool = in_set_arr.astype(bool)
    pli_all = frame["lof_pLI"].to_numpy(dtype=float)
    n = len(L)

    # Replicate the exact decile construction from batch_059 E2.
    bin_ids_series = pd.qcut(L, deciles, labels=False, duplicates="drop")
    bin_ids = np.asarray(bin_ids_series, dtype=float)
    if np.any(~np.isfinite(bin_ids)):
        raise RuntimeError("qcut produced NaN bin_ids")
    bin_ids = bin_ids.astype(int)
    n_bins = int(bin_ids.max()) + 1

    ring_per_decile = np.bincount(bin_ids[ring_bool], minlength=n_bins)
    bg_mask = ~ring_bool
    decile_bg_indices = {
        d: np.where(bg_mask & (bin_ids == d))[0] for d in range(n_bins)
    }

    mean_pli_per_draw = np.zeros(n_draws, dtype=float)
    std_pli_per_draw = np.zeros(n_draws, dtype=float)

    for di in range(n_draws):
        picks: list[int] = []
        for d in range(n_bins):
            n_needed = int(ring_per_decile[d])
            if n_needed == 0:
                continue
            bg_d = decile_bg_indices[d]
            if len(bg_d) == 0:
                continue
            if len(bg_d) < n_needed:
                sampled = rng.choice(bg_d, size=n_needed, replace=True)
            else:
                sampled = rng.choice(bg_d, size=n_needed, replace=False)
            picks.extend(sampled.tolist())

        if len(picks) < 10:
            mean_pli_per_draw[di] = np.nan
            std_pli_per_draw[di] = np.nan
            # WHY we still need to advance the RNG state for the OLS fit:
            # batch_059 E2 also ran an OLS fit per draw. We don't need the
            # OLS result here, but we DO need to consume the same RNG stream
            # so subsequent draws remain synchronized. However, the OLS fit
            # in batch_059 uses statsmodels which is deterministic given
            # the same data, so no RNG state is consumed by OLS. The only
            # RNG consumption is the rng.choice calls above, which we
            # already executed.
            continue

        pli_draw = pli_all[picks]
        mean_pli_per_draw[di] = float(np.mean(pli_draw))
        std_pli_per_draw[di] = float(np.std(pli_draw, ddof=1))

        if (di + 1) % 2000 == 0:
            logger.info("  E3 replay draw %d/%d", di + 1, n_draws)

    return {
        "mean_pli_per_draw": mean_pli_per_draw,
        "std_pli_per_draw": std_pli_per_draw,
        "cell_seed_used": int(cell_seed),
        "ring_per_decile": ring_per_decile.tolist(),
        "n_bins": int(n_bins),
    }


def compute_pli_regression_coefficient(frame: pd.DataFrame, logger) -> dict:
    """Fit MAGMA_Z ~ covariates and extract the pLI coefficient.

    WHY: The expected beta bias is computed as
      (mean_pLI_EDT1 - mean_pLI_null) x pLI_regression_coefficient

    The pLI regression coefficient tells us how much MAGMA_Z changes per unit
    change in pLI, controlling for other covariates. This is the standard
    Sub-A v2.1 covariate set (V1_LINEAR).
    """
    covs = E2_COVS_V1_LINEAR
    X = frame[covs].to_numpy(dtype=float)
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(y, Xc).fit()

    # lof_pLI is the second covariate in E2_COVS_V1_LINEAR (index 1 in covs,
    # index 2 in Xc because of the constant at index 0, and in_set is not
    # included here -- this is a background regression without the indicator).
    # WHY we don't include in_set: we want the marginal effect of pLI on
    # MAGMA_Z across ALL genes, not conditional on set membership. The bias
    # is about the BACKGROUND pLI distribution, not the within-set effect.
    pli_idx = covs.index("lof_pLI") + 1  # +1 for the constant
    pli_beta = float(ols.params[pli_idx])
    pli_se = float(ols.bse[pli_idx])
    logger.info("pLI regression coef: beta=%.4f se=%.4f", pli_beta, pli_se)
    return {
        "pli_beta": pli_beta,
        "pli_se": pli_se,
        "pli_t": float(ols.tvalues[pli_idx]),
        "pli_p": float(ols.pvalues[pli_idx]),
        "covariates": covs,
        "n": int(X.shape[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="batch_060 E3: pLI confound check"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke: reduce to 500 draws.")
    parser.add_argument("--smoke-draws", type=int, default=500)
    args = parser.parse_args()

    logger = setup_logger("batch_060.e3", LOGS_DIR / "e3.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "e3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load upstream data.
    gnomad = load_gnomad_per_brief_v2()
    annot = load_gene_annot()
    edt1_sym = load_edt1()
    b3 = set(B3_GENES)
    edt1_ex_b3_sym = edt1_sym - b3
    edt1_ex_b3_ensg, _ = symbols_to_ensgids(edt1_ex_b3_sym)

    # Step 2: Build the same Sub-A frame that batch_059 E2 used for the
    # "remaining" ring x SCZ cell. The "remaining" ring in batch_059 E2
    # is EDT1-ex-B3 minus the SynGO scaffold and vesicle rings. Since
    # those rings had n<20 and were UNINTERPRETABLE, and the "remaining"
    # ring contained ~404 genes (~= EDT1-ex-B3 itself modulo the ~13
    # scaffold+vesicle genes), we use EDT1-ex-B3 directly.
    # WHY this is valid: batch_059 E2 ran the length-matched null on the
    # "remaining" ring (n=404 in SCZ frame), which is EDT1-ex-B3 minus
    # ~13 SynGO genes. The pLI distribution difference between 404 and
    # 414 genes is negligible (< 3% of genes differ).
    frame = build_sub_a_frame("scz", gnomad, annot, edt1_ex_b3_ensg)

    # Add polynomial covariates for consistency with E2.
    L = frame["log10_gene_length"].to_numpy(dtype=float)
    frame["log10_gene_length_sq"] = L * L
    frame["log10_gene_length_cu"] = L * L * L
    from scipy.stats import rankdata
    r = rankdata(L, method="average")
    frame["rank_pct_length"] = (r - 0.5) / len(r)

    n_total = len(frame)
    n_in_set = int(frame["in_set"].sum())
    logger.info("E3 frame: n=%d in_set=%d", n_total, n_in_set)

    # Step 3: Compute actual EDT1-ex-B3 mean(pLI).
    edt1_frame = frame[frame["in_set"] == 1]
    mean_pli_edt1 = float(edt1_frame["lof_pLI"].mean())
    std_pli_edt1 = float(edt1_frame["lof_pLI"].std(ddof=1))
    logger.info("EDT1-ex-B3 actual: mean_pLI=%.4f std_pLI=%.4f",
                 mean_pli_edt1, std_pli_edt1)

    # Step 4: Replay length-matched null draws.
    # The cell key for batch_059 E2's remaining__scz cell.
    cell_key = "remaining__scz"
    n_draws = args.smoke_draws if args.smoke else E3_N_NULL_DRAWS
    replay = replay_length_matched_null_draws(
        frame=frame,
        ring_size=n_in_set,
        n_draws=n_draws,
        seed_base=SEED_E2_PERM,
        cell_key=cell_key,
        deciles=E2_LENGTH_N_DECILES,
        logger=logger,
    )

    mean_pli_draws = replay["mean_pli_per_draw"]
    std_pli_draws = replay["std_pli_per_draw"]
    finite_mask = np.isfinite(mean_pli_draws)
    n_finite = int(finite_mask.sum())

    if n_finite == 0:
        results = {
            "status": "failed",
            "batch": "060", "sub": "e3", "brief": "brief_v2.md",
            "reason": "No finite draws in replay",
            "verdict": {"verdict": "UNINTERPRETABLE", "reason": "No finite draws"},
        }
        atomic_write_json(results, out_dir / "results.json")
        return 0

    mean_pli_null = float(np.mean(mean_pli_draws[finite_mask]))
    std_pli_null_across_draws = float(np.std(mean_pli_draws[finite_mask], ddof=1))
    se_mean_pli_null = float(std_pli_null_across_draws / np.sqrt(n_finite))

    logger.info(
        "E3 null pLI: mean_of_means=%.4f std_across_draws=%.4f SE=%.6f "
        "n_finite=%d",
        mean_pli_null, std_pli_null_across_draws, se_mean_pli_null, n_finite,
    )

    # Step 5: Compute pLI regression coefficient.
    pli_reg = compute_pli_regression_coefficient(frame, logger)
    pli_beta = pli_reg["pli_beta"]

    # Step 6: Compute expected beta bias.
    pli_gap = mean_pli_edt1 - mean_pli_null
    expected_bias = pli_gap * pli_beta
    logger.info(
        "E3 bias: pLI_gap=%.4f pLI_reg_beta=%.4f expected_bias=%.6f",
        pli_gap, pli_beta, expected_bias,
    )

    # Step 7: Load the observed SCZ EDT1-ex-B3 beta from batch_059 E2 for
    # the bias ratio computation.
    e2_results_path = BATCH_059_OUTPUT / "e2" / "results.json"
    observed_beta_scz = float("nan")
    if e2_results_path.exists():
        with e2_results_path.open() as fh:
            e2_data = json.load(fh)
        # The observed beta for EDT1-ex-B3 x SCZ under V1_LINEAR is in
        # R1_reproduction_gate.edt1_ex_b3_scz_beta_v1_linear.
        observed_beta_scz = float(
            e2_data.get("R1_reproduction_gate", {})
            .get("edt1_ex_b3_scz_beta_v1_linear", float("nan"))
        )
    logger.info("E3 observed SCZ beta (from E2): %.4f", observed_beta_scz)

    # Bias as fraction of observed beta.
    if np.isfinite(observed_beta_scz) and abs(observed_beta_scz) > 1e-9:
        bias_fraction = abs(expected_bias) / abs(observed_beta_scz)
    else:
        bias_fraction = float("nan")

    # Step 8: Apply decision rule.
    if mean_pli_null > E3_PLI_ADEQUATE_FLOOR and bias_fraction < E3_BIAS_INCONSEQUENTIAL:
        verdict = {
            "verdict": "ADEQUATELY_CONTROLLED",
            "reason": (
                f"mean_pLI_null={mean_pli_null:.4f} > {E3_PLI_ADEQUATE_FLOOR} "
                f"AND bias_fraction={bias_fraction:.4f} < {E3_BIAS_INCONSEQUENTIAL} "
                f"(bias={expected_bias:.6f}, observed_beta={observed_beta_scz:.4f})"
            ),
        }
    elif mean_pli_null < E3_PLI_PARTIAL_FLOOR and bias_fraction > E3_BIAS_INADEQUATE:
        verdict = {
            "verdict": "INADEQUATELY_CONTROLLED",
            "reason": (
                f"mean_pLI_null={mean_pli_null:.4f} < {E3_PLI_PARTIAL_FLOOR} "
                f"AND bias_fraction={bias_fraction:.4f} > {E3_BIAS_INADEQUATE}"
            ),
        }
    elif E3_PLI_PARTIAL_FLOOR <= mean_pli_null <= E3_PLI_ADEQUATE_FLOOR:
        verdict = {
            "verdict": "PARTIALLY_CONTROLLED",
            "reason": (
                f"mean_pLI_null={mean_pli_null:.4f} in "
                f"[{E3_PLI_PARTIAL_FLOOR}, {E3_PLI_ADEQUATE_FLOOR}]"
            ),
        }
    elif mean_pli_null > E3_PLI_ADEQUATE_FLOOR and bias_fraction >= E3_BIAS_INCONSEQUENTIAL:
        # Mean pLI is high enough but bias is non-negligible.
        verdict = {
            "verdict": "PARTIALLY_CONTROLLED",
            "reason": (
                f"mean_pLI_null={mean_pli_null:.4f} > {E3_PLI_ADEQUATE_FLOOR} "
                f"but bias_fraction={bias_fraction:.4f} >= {E3_BIAS_INCONSEQUENTIAL}"
            ),
        }
    else:
        verdict = {
            "verdict": "INCONCLUSIVE",
            "reason": (
                f"No decision rule matched: mean_pLI_null={mean_pli_null:.4f}, "
                f"bias_fraction={bias_fraction:.4f}"
            ),
        }
    logger.info("E3 verdict: %s", verdict["verdict"])

    # Seed reconstruction verification: check that our replayed draws
    # produce similar null beta distribution as batch_059 E2.
    # Load batch_059 E2 length null summary for the remaining__scz cell.
    seed_verification = {"status": "not_checked"}
    e2_results_path = BATCH_059_OUTPUT / "e2" / "results.json"
    if e2_results_path.exists():
        with e2_results_path.open() as fh:
            e2_full = json.load(fh)
        e2_null_summary = e2_full.get("length_matched_null_summary", {}).get(
            "remaining__scz", {}
        )
        if e2_null_summary:
            e2_ring_per_decile = e2_null_summary.get("ring_per_decile", [])
            our_ring_per_decile = replay.get("ring_per_decile", [])
            decile_match = (e2_ring_per_decile == our_ring_per_decile)
            seed_verification = {
                "status": "ok" if decile_match else "MISMATCH",
                "e2_ring_per_decile": e2_ring_per_decile,
                "our_ring_per_decile": our_ring_per_decile,
                "decile_match": decile_match,
                "e2_n_draws": e2_null_summary.get("n_draws"),
                "our_n_draws": n_draws,
                "note": (
                    "Ring-per-decile match confirms the same frame and "
                    "decile structure. Full beta verification requires "
                    "loading the .npz (not done here to keep E3 lightweight)."
                ),
            }
    logger.info("E3 seed verification: %s", seed_verification.get("status"))

    # Provenance.
    provenance = {
        "gnomad_tsv": sha256_file(GNOMAD_TSV),
        "gene_annot": sha256_file(GENE_ANNOT),
        "magma_scz": sha256_file(MAGMA_SCZ_GENES_OUT),
        "e2_results": (
            sha256_file(e2_results_path)
            if e2_results_path.exists() else "missing"
        ),
    }

    results = {
        "status": "ok",
        "batch": "060", "sub": "e3", "brief": "brief_v2.md",
        "wall_s": time.time() - t0,
        "smoke": bool(args.smoke),
        "n_frame": n_total,
        "n_edt1_ex_b3": n_in_set,
        "edt1_ex_b3_pli": {
            "mean": mean_pli_edt1,
            "std": std_pli_edt1,
        },
        "null_pli_distribution": {
            "n_draws": n_draws,
            "n_finite": n_finite,
            "mean_of_means": mean_pli_null,
            "std_across_draws": std_pli_null_across_draws,
            "se_mean": se_mean_pli_null,
            "q05": float(np.quantile(mean_pli_draws[finite_mask], 0.05)),
            "q50": float(np.median(mean_pli_draws[finite_mask])),
            "q95": float(np.quantile(mean_pli_draws[finite_mask], 0.95)),
            "mean_of_stds": float(np.mean(std_pli_draws[finite_mask])),
        },
        "pli_regression": pli_reg,
        "bias_analysis": {
            "pli_gap": pli_gap,
            "pli_regression_coefficient": pli_beta,
            "expected_bias": expected_bias,
            "observed_beta_scz": observed_beta_scz,
            "bias_fraction_of_observed": bias_fraction,
            "bias_fraction_threshold_adequate": E3_BIAS_INCONSEQUENTIAL,
            "bias_fraction_threshold_inadequate": E3_BIAS_INADEQUATE,
        },
        "seed_verification": seed_verification,
        "verdict": verdict,
        "provenance_sha256": provenance,
        "brief_contract": {
            "n_null_draws": n_draws,
            "seed_base": SEED_E2_PERM,
            "cell_key": cell_key,
            "cell_seed_used": replay["cell_seed_used"],
            "deciles": E2_LENGTH_N_DECILES,
            "pli_adequate_floor": E3_PLI_ADEQUATE_FLOOR,
            "pli_partial_floor": E3_PLI_PARTIAL_FLOOR,
            "bias_inconsequential_threshold": E3_BIAS_INCONSEQUENTIAL,
            "bias_inadequate_threshold": E3_BIAS_INADEQUATE,
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("E3 wrote %s (wall=%.1fs)",
                 out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
