#!/usr/bin/env python3
"""batch_056 Sub-C — extended PoPS p-grid {0.075, 0.08, 0.09}.

Reuses batch_055_A's PoPS subprocess runner (run_pops / maybe_cached_or_run)
verbatim. Writes .preds to experiments/batch_056/output/sub_c/pgrid/
cutoff_<c>/ and then computes paired Δρ (Spearman) vs the p=0.07 anchor
(batch_055_A cutoff_0.07 preds; target ρ = 0.5284).

Reproduction gate R2: Spearman ρ(batch_055_A p=0.07 preds, MAGMA-Z) on the
17,459-gene shared bg must reproduce 0.5284 ± 1e-4.

WHY we reuse batch_055_A's runner (not call its script as subprocess): the
runner is pure Python (no stateful side effects in the imported module) and
importing it preserves bit-identical flag handling + provenance logic.

Per brief_v2:
  - CUTOFFS = [0.075, 0.08, 0.09]
  - Paired bootstrap n=1000, seed=20260423, idx matrix bit-identical to
    batch_054_A/055_A when n_genes matches.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BATCH055A_P07_PREDS,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    LOGS_DIR,
    OUTPUT_DIR,
    REPRO_R2_SPEARMAN_TARGET,
    REPRO_TOLERANCE,
    atomic_write_json,
    load_common_ensgids,
    load_magma_scz,
    load_preds,
    percentile_ci,
    reproduce_spearman_anchor,
)

# Import batch_055_A's runner helpers. WHY sys.path insert: batch_055_A lives
# in a sibling directory and is not a package; this is the same sys.path
# trick used by batch_055_B for reuse of batch_048 primitives.
sys.path.insert(
    0,
    str(Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia"
             "/experiments/batch_055_A/scripts")),
)
import run_batch_055_A as b55A  # noqa: E402

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# Brief_v2 §Sub-C: 3-point grid at {0.075, 0.08, 0.09}.
CUTOFFS = [0.075, 0.08, 0.09]

# Anchor p=0.07 from batch_055_A (ρ=0.5284).
ANCHOR_CUTOFF = 0.07

# Decision thresholds (brief_v2 §Sub-C DECISION RULE).
SHAPE_A_PLATEAU_EPS = 0.005     # ρ(c) − ρ(0.07) within ±0.005
SHAPE_A_CLIFF_EPS = 0.005       # ρ(0.09) ≤ ρ(0.07) − 0.005
SHAPE_B_EXCESS_EPS = 0.005      # ρ(c) > ρ(0.07) + 0.005
SHAPE_C_PLATEAU_EPS = 0.005


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_056.sub_c")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%dT%H:%M:%S")
        for h in (logging.FileHandler(LOGS_DIR / "sub_c.log"),
                  logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def classify(rho_anchor: float, rho_per_cutoff: dict[float, float],
             cis: dict[float, tuple[float, float]]) -> dict:
    """Apply Sub-C DECISION RULE (brief_v2). Inputs are POINT ρ values per
    cutoff and corresponding Δρ (vs anchor p=0.07) CIs."""
    r07 = rho_anchor
    r075, r08, r09 = (rho_per_cutoff[0.075], rho_per_cutoff[0.08],
                       rho_per_cutoff[0.09])
    all_plateau = all(r <= r07 + SHAPE_A_PLATEAU_EPS
                       for r in (r075, r08, r09))
    cliff_09 = r09 <= r07 - SHAPE_A_CLIFF_EPS
    if all_plateau and cliff_09:
        return {"classification": "Shape_A",
                "reason": ("ρ(0.075), ρ(0.08), ρ(0.09) all ≤ ρ(0.07)+0.005 "
                           "AND ρ(0.09) ≤ ρ(0.07)−0.005")}

    # Shape B: any cutoff exceeds anchor + 0.005 with CI > 0.
    for cut in CUTOFFS:
        excess = rho_per_cutoff[cut] - r07
        ci = cis[cut]
        if excess > SHAPE_B_EXCESS_EPS and ci[0] > 0:
            return {"classification": "Shape_B",
                    "reason": (f"ρ({cut}) − ρ(0.07) = {excess:+.4f} > 0.005 "
                               f"with Δρ CI = [{ci[0]:.4f},{ci[1]:.4f}] > 0")}

    # Shape C: all 4 points within ±0.005.
    all_within = all(abs(rho_per_cutoff[c] - r07) < SHAPE_C_PLATEAU_EPS
                     for c in CUTOFFS)
    if all_within:
        return {"classification": "Shape_C",
                "reason": "All 4 points {0.07, 0.075, 0.08, 0.09} within ±0.005"}

    return {"classification": "Intermediate",
            "reason": "None of Shape_A / Shape_B / Shape_C thresholds met"}


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_056 Sub-C")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--skip-pops", action="store_true",
                        help="skip PoPS runs; assumes preds already on disk")
    parser.add_argument("--force", action="store_true",
                        help="force re-run of all PoPS cutoffs")
    args = parser.parse_args()

    logger = setup_logger()
    t0 = time.time()
    out_dir = OUTPUT_DIR / "sub_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    pgrid_dir = out_dir / "pgrid"
    pgrid_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- Loaders + reproduction gate R2 ---------------------
    try:
        magma = load_magma_scz()
        common_ensgids = load_common_ensgids()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sub-C loader failure")
        atomic_write_json(
            {"status": "failed", "phase": "load", "error": str(exc)},
            out_dir / "results.json",
        )
        return 10

    repro_r2 = reproduce_spearman_anchor(
        BATCH055A_P07_PREDS, magma, common_ensgids,
        REPRO_R2_SPEARMAN_TARGET, REPRO_TOLERANCE,
    )
    logger.info("R2: %s", repro_r2)
    if not args.skip_gate and not repro_r2.get("pass", False):
        atomic_write_json(
            {"status": "failed", "phase": "R2",
             "reproduction_gate": repro_r2},
            out_dir / "results.json",
        )
        return 20

    # --------------------- PoPS subprocess runs ---------------------
    # WHY we reuse batch_055_A helpers: collect_munged_chunks, run_pops,
    # maybe_cached_or_run, write_preflight_provenance — identical flag set
    # (including --control_features_path + remove_hla defaults). Any drift
    # would break R1 reproduction at anchor re-verification.
    try:
        _, cols_files, _ = b55A.collect_munged_chunks()
        # Count mat chunks for num_feature_chunks.
        chunks = sorted(Path(b55A.FEATURES_MUNGED_DIR).glob(
            "pops_features.mat.*.npy"))
        num_chunks = len(chunks)
        if num_chunks == 0:
            raise RuntimeError(
                "No munged feature chunks found — batch_055_A expects 12"
            )
        logger.info("PoPS num_feature_chunks = %d", num_chunks)
    except Exception as exc:  # noqa: BLE001
        logger.exception("PoPS preflight failure")
        atomic_write_json(
            {"status": "failed", "phase": "pops_preflight",
             "reproduction_gate": repro_r2, "error": str(exc)},
            out_dir / "results.json",
        )
        return 30

    run_meta: dict = {}
    if not args.skip_pops:
        for cut in CUTOFFS:
            sub_dir = pgrid_dir / f"cutoff_{cut}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            out_prefix = sub_dir / "PGC3_EUR_PoPS"
            try:
                meta = b55A.maybe_cached_or_run(
                    out_prefix=out_prefix,
                    num_chunks=num_chunks,
                    cutoff=cut,
                    logger=logger,
                    tag=f"pgrid_cutoff_{cut}",
                    subset_features_path=None,
                    force=args.force,
                    timeout_s=3600,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("PoPS run failed at cutoff=%s", cut)
                atomic_write_json({
                    "status": "failed", "phase": "pops_run",
                    "reproduction_gate": repro_r2,
                    "failing_cutoff": cut,
                    "error": str(exc),
                }, out_dir / "results.json")
                return 40
            if not meta["ok"]:
                atomic_write_json({
                    "status": "failed", "phase": "pops_run",
                    "reproduction_gate": repro_r2,
                    "failing_cutoff": cut,
                    "run_meta": meta,
                }, out_dir / "results.json")
                return 41
            run_meta[str(cut)] = meta
    else:
        for cut in CUTOFFS:
            sub_dir = pgrid_dir / f"cutoff_{cut}"
            preds_path = sub_dir / "PGC3_EUR_PoPS.preds"
            if not preds_path.exists():
                atomic_write_json({
                    "status": "failed", "phase": "skip_pops_no_cache",
                    "reproduction_gate": repro_r2,
                    "missing_preds": str(preds_path),
                }, out_dir / "results.json")
                return 42
            run_meta[str(cut)] = {"tag": f"pgrid_cutoff_{cut}", "cached": True,
                                   "preds_path": str(preds_path), "ok": True}

    # --------------------- Build analysis frame ---------------------
    anchor_preds = load_preds(BATCH055A_P07_PREDS)
    per_cutoff_preds = {
        cut: load_preds(Path(run_meta[str(cut)]["preds_path"]))
        for cut in CUTOFFS
    }

    # Intersection across: anchor preds ∪ all 3 new preds ∪ MAGMA ∪ common.
    # WHY intersection across all: paired bootstrap requires a single shared
    # gene sample so the (pops[k], magma) vectors align row-by-row.
    frame = pd.DataFrame({"ENSGID": common_ensgids})
    frame = (frame.merge(anchor_preds.rename(columns={
                "PoPS_Score": "pops_p0.07"}), on="ENSGID", how="left")
                  .merge(magma, on="ENSGID", how="left"))
    for cut in CUTOFFS:
        frame = frame.merge(per_cutoff_preds[cut].rename(
            columns={"PoPS_Score": f"pops_p{cut}"}), on="ENSGID", how="left")

    raw_n = len(frame)
    cols_to_check = ["pops_p0.07", "MAGMA_Z"] + [f"pops_p{c}" for c in CUTOFFS]
    frame = frame.dropna(subset=cols_to_check).reset_index(drop=True)
    frame = frame.sort_values("ENSGID").reset_index(drop=True)
    n = len(frame)
    logger.info("Analysis frame: n=%d (raw=%d)", n, raw_n)

    if n < 15000:
        atomic_write_json({
            "status": "failed", "phase": "intersection_small",
            "reproduction_gate": repro_r2,
            "n_after_intersect": n,
        }, out_dir / "results.json")
        return 50

    magma_z = frame["MAGMA_Z"].to_numpy(dtype=float)
    pops_anchor = frame["pops_p0.07"].to_numpy(dtype=float)
    pops_per = {cut: frame[f"pops_p{cut}"].to_numpy(dtype=float)
                for cut in CUTOFFS}

    # Point ρ.
    rho_anchor_point, _ = spearmanr(pops_anchor, magma_z)
    rho_per_point = {cut: float(spearmanr(pops_per[cut], magma_z)[0])
                      for cut in CUTOFFS}
    logger.info("ρ(anchor p=0.07) = %.6f; per-cutoff = %s",
                rho_anchor_point, rho_per_point)

    # Paired bootstrap — single shared idx matrix over n genes.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx_mat = rng.integers(0, n, size=(BOOTSTRAP_N, n))

    boot_anchor = np.zeros(BOOTSTRAP_N)
    boot_per = {cut: np.zeros(BOOTSTRAP_N) for cut in CUTOFFS}
    for i in range(BOOTSTRAP_N):
        b = idx_mat[i]
        m_b = magma_z[b]
        r_a, _ = spearmanr(pops_anchor[b], m_b)
        boot_anchor[i] = float(r_a)
        for cut in CUTOFFS:
            r, _ = spearmanr(pops_per[cut][b], m_b)
            boot_per[cut][i] = float(r)
        if (i + 1) % 250 == 0:
            logger.info("  bootstrap %d/%d", i + 1, BOOTSTRAP_N)

    per_cutoff_ci = {}
    delta_ci = {}
    for cut in CUTOFFS:
        lo, hi, med = percentile_ci(boot_per[cut])
        per_cutoff_ci[cut] = {"ci_lo": lo, "ci_hi": hi, "median": med}
        d_samples = boot_per[cut] - boot_anchor
        d_lo, d_hi, d_med = percentile_ci(d_samples)
        delta_ci[cut] = {"ci_lo": d_lo, "ci_hi": d_hi, "median": d_med}

    anchor_ci = dict(zip(
        ("ci_lo", "ci_hi", "median"),
        percentile_ci(boot_anchor),
    ))

    # Classification.
    classification = classify(
        rho_anchor=float(rho_anchor_point),
        rho_per_cutoff=rho_per_point,
        cis={c: (delta_ci[c]["ci_lo"], delta_ci[c]["ci_hi"]) for c in CUTOFFS},
    )
    logger.info("Classification: %s", classification)

    wall = time.time() - t0
    results = {
        "status": "ok",
        "batch": "056",
        "sub": "c",
        "wall_s": wall,
        "reproduction_gate_R2": repro_r2,
        "n_common": int(n),
        "cutoffs": CUTOFFS,
        "anchor_cutoff": ANCHOR_CUTOFF,
        "rho_anchor_point_spearman": float(rho_anchor_point),
        "rho_anchor_bootstrap_ci": anchor_ci,
        "rho_per_cutoff_point_spearman": rho_per_point,
        "rho_per_cutoff_bootstrap_ci": per_cutoff_ci,
        "delta_rho_per_cutoff_vs_anchor_ci": delta_ci,
        "bootstrap": {"n_boot": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
        "pops_run_meta": run_meta,
        "decision_classification": classification,
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("Sub-C wrote %s (wall=%.1fs)", out_dir / "results.json", wall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
