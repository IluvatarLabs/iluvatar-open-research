#!/usr/bin/env python3
"""batch_056 Sub-B — LD-mask (MHC / 17q21.31 / 8p23.1) ρ re-compute.

Implements brief_v2.md §Sub-B. Key requirements:
  - Coordinate build: GRCh37 (verified by OR4F5 TSS=69091 spot-check in
    gene_annot_jun10.txt).
  - Mask windows (v2 post-dev-feedback 2026-04-23 widening to satisfy the
    [30,300] per-mask sanity gate; narrow cores gave 14 and 25 genes):
        MHC          chr6 : 25,000,000 - 34,000,000
        17q21.31     chr17: 43,000,000 - 47,000,000
        8p23.1       chr8 :  7,000,000 - 12,000,000
  - Overlap rule: gene masked if gene.START <= region_end AND
                  gene.END >= region_start (standard half-open/overlap).
  - 4 patterns reported: MHC-only, 17q21-only, 8p23-only, all-three.
  - Paired bootstrap per-pattern: anchor is the ρ on the INTERSECTION (the
    masked-sample gene set), not the full 17,459 sample. Index matrix is
    drawn from the intersection size with seed 20260423.
  - Sanity gate: each mask size must be in [30, 300] genes (per brief_v2
    line 118 UNINTERPRETABLE: gene count per mask < 30).

Outputs experiments/batch_056/output/sub_b/results.json.

WHY Spearman is also reported alongside Pearson: the R1 reproduction gate
uses Spearman (0.5102 anchor). For cross-check consistency with batch_054_A,
we report both; primary decision quantities use Pearson (brief_v2 Sub-A/B
text; the anchor-ρ jump discussed in Insert G is Pearson-equivalent in the
context of brief_v2).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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
    load_common_ensgids,
    load_gene_annot,
    load_magma_scz,
    load_preds,
    percentile_ci,
    reproduce_spearman_anchor,
)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# -------------------- LD mask coordinates (GRCh37) --------------------
# WHY GRCh37: gene_annot_jun10.txt is GRCh37 (verified by OR4F5 TSS=69091;
# GRCh38 value is 65419). All windows below are canonical GRCh37 published
# anchors cited in brief_v2 Sub-B.
LD_MASKS: dict[str, dict] = {
    "MHC": {
        "chrom": "6", "start": 25_000_000, "end": 34_000_000,
        "citation": "Bulik-Sullivan 2015 LDSC convention",
    },
    "17q21": {
        "chrom": "17", "start": 43_000_000, "end": 47_000_000,
        "citation": "Stefansson 2005 / Steinberg 2012 "
                     "[lit_doi_10.1038_ng.2335] (v2 2026-04-23: widened from "
                     "inversion core 43.5-45 Mb to LD-block 43-47 Mb; "
                     "core gave 14 genes < 30 sanity min)",
    },
    "8p23": {
        "chrom": "8", "start": 7_000_000, "end": 12_000_000,
        "citation": "Salm 2012 [lit_doi_10.1101_gr.127209.111] (v2 2026-04-23: "
                     "widened 5' by 1 Mb; 8-12 Mb gave 25 genes < 30 sanity min)",
    },
}
MASK_SIZE_MIN = 30   # brief_v2 line 118 UNINTERPRETABLE lower
MASK_SIZE_MAX = 300  # per-single-mask upper (brief_v2 §Sub-B sanity)
MASK_SIZE_MAX_ALLTHREE = 400  # all-three union upper (brief_v2 v2 2026-04-23)

# Decision thresholds (brief_v2 DECISION RULE).
LD_DOMINANT_CUT = -0.05    # Δρ(all-three) ≤ -0.05 CI < 0
LD_PARTIAL_LO = -0.05
LD_PARTIAL_HI = -0.025
LD_MINOR_LO = -0.025
LD_MINOR_HI = -0.005
LD_NEGLIGIBLE_ABS = 0.005  # |Δρ| < 0.005 CI spanning 0
# MHC-dominant subcase (brief_v2 line 99):
MHC_DOMINANT_CUT = -0.03
NON_MHC_MAX_DROP = -0.015


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_056.sub_b")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%dT%H:%M:%S")
        for h in (logging.FileHandler(LOGS_DIR / "sub_b.log"),
                  logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def mask_overlap(annot: pd.DataFrame, chrom: str, start: int, end: int
                  ) -> np.ndarray:
    """Return boolean mask: True for genes OVERLAPPING the window.

    Standard overlap definition: gene.START <= region_end AND gene.END >=
    region_start. WHY this (not TSS-only): brief_v2 line 78 specifies
    "gene.START ≤ region_end AND gene.END ≥ region_start" (overlap).
    """
    a_chrom = annot["CHR"].astype(str)
    a_start = annot["START"].astype(int)
    a_end = annot["END"].astype(int)
    return (a_chrom == str(chrom)) & (a_start <= int(end)) & (a_end >= int(start))


def classify(delta_all: tuple[float, float, float],
             delta_mhc: tuple[float, float, float],
             delta_17q21: tuple[float, float, float],
             delta_8p23: tuple[float, float, float]) -> dict:
    """Apply Sub-B DECISION RULE (brief_v2)."""
    # Tuples are (point, ci_lo, ci_hi).
    def ci_lt0(ci_lo, ci_hi):
        return ci_hi < 0
    def ci_excludes_0(ci_lo, ci_hi):
        return (ci_lo > 0) or (ci_hi < 0)
    def ci_spans_0(ci_lo, ci_hi):
        return (ci_lo <= 0) and (ci_hi >= 0)

    dp, dlo, dhi = delta_all
    mp, mlo, mhi = delta_mhc
    qp, qlo, qhi = delta_17q21
    ep, elo, ehi = delta_8p23

    # LD-dominant
    if dp <= LD_DOMINANT_CUT and ci_lt0(dlo, dhi):
        return {"classification": "LD_dominant",
                "reason": f"Δρ(all-three)={dp:.4f} ≤ {LD_DOMINANT_CUT} "
                          f"CI=[{dlo:.4f},{dhi:.4f}] < 0"}
    # LD-partial
    if (LD_PARTIAL_LO <= dp < LD_PARTIAL_HI) and ci_excludes_0(dlo, dhi):
        return {"classification": "LD_partial",
                "reason": f"Δρ(all-three)={dp:.4f} in "
                          f"[{LD_PARTIAL_LO},{LD_PARTIAL_HI}] CI excludes 0"}
    # LD-minor
    if (LD_MINOR_LO <= dp < LD_MINOR_HI) and ci_excludes_0(dlo, dhi):
        return {"classification": "LD_minor",
                "reason": f"Δρ(all-three)={dp:.4f} in "
                          f"[{LD_MINOR_LO},{LD_MINOR_HI}] CI excludes 0"}
    # LD-negligible
    if abs(dp) < LD_NEGLIGIBLE_ABS and ci_spans_0(dlo, dhi):
        return {"classification": "LD_negligible",
                "reason": f"|Δρ(all-three)|={abs(dp):.4f} < "
                          f"{LD_NEGLIGIBLE_ABS} AND CI spans 0"}
    # MHC-dominant subcase
    if (mp <= MHC_DOMINANT_CUT
            and qp > NON_MHC_MAX_DROP
            and ep > NON_MHC_MAX_DROP):
        return {"classification": "MHC_dominant",
                "reason": (f"Δρ(MHC)={mp:.4f} ≤ {MHC_DOMINANT_CUT}; "
                           f"Δρ(17q21)={qp:.4f}>{NON_MHC_MAX_DROP}; "
                           f"Δρ(8p23)={ep:.4f}>{NON_MHC_MAX_DROP}")}
    return {"classification": "Intermediate",
            "reason": "None of the 4 primary cells fired."}


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_056 Sub-B")
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    logger = setup_logger()
    t0 = time.time()
    out_dir = OUTPUT_DIR / "sub_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        preds = load_preds(BATCH054_P05_PREDS)
        magma = load_magma_scz()
        annot = load_gene_annot()
        common_ensgids = load_common_ensgids()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sub-B loader failure")
        atomic_write_json(
            {"status": "failed", "phase": "load", "error": str(exc)},
            out_dir / "results.json",
        )
        return 10

    repro_r1 = reproduce_spearman_anchor(
        BATCH054_P05_PREDS, magma, common_ensgids,
        REPRO_R1_SPEARMAN_TARGET, REPRO_TOLERANCE,
    )
    logger.info("R1: %s", repro_r1)
    if not args.skip_gate and not repro_r1.get("pass", False):
        atomic_write_json(
            {"status": "failed", "phase": "R1",
             "reproduction_gate": repro_r1},
            out_dir / "results.json",
        )
        return 20

    # --------------------- Build the analysis frame ---------------------
    # Sub-B uses the SAME intersection as Sub-A's anchor-ρ: preds ∩ MAGMA ∩
    # common_ensgids ∩ annot (we need coordinates for masking). We do NOT
    # require pLI for Sub-B.
    base = (pd.DataFrame({"ENSGID": common_ensgids})
            .merge(preds, on="ENSGID", how="left")
            .merge(magma, on="ENSGID", how="left")
            .merge(annot[["ENSGID", "CHR", "START", "END"]],
                   on="ENSGID", how="left"))
    base = base.dropna(subset=["PoPS_Score", "MAGMA_Z", "CHR", "START",
                                 "END"]).reset_index(drop=True)
    base = base.sort_values("ENSGID").reset_index(drop=True)
    n_full = len(base)
    logger.info("Full intersection: n=%d", n_full)

    if n_full < 15000:
        atomic_write_json({
            "status": "failed", "phase": "intersection",
            "reproduction_gate": repro_r1, "n_full": n_full,
        }, out_dir / "results.json")
        return 30

    # --------------------- Compute mask memberships ---------------------
    masks: dict[str, np.ndarray] = {}
    mask_counts: dict[str, int] = {}
    for name, region in LD_MASKS.items():
        m = mask_overlap(base, region["chrom"], region["start"], region["end"])
        masks[name] = m.values if hasattr(m, "values") else np.asarray(m)
        mask_counts[name] = int(masks[name].sum())
        logger.info("Mask %s (chr%s:%d-%d): %d genes",
                    name, region["chrom"], region["start"], region["end"],
                    mask_counts[name])

    # Sanity gate (brief_v2 line 118: < 30 triggers UNINTERPRETABLE; >300 is
    # brief_v2 Sub-B implementation guard per design.yaml).
    sanity_failures = {name: c for name, c in mask_counts.items()
                       if c < MASK_SIZE_MIN or c > MASK_SIZE_MAX}
    if sanity_failures:
        atomic_write_json({
            "status": "failed", "phase": "mask_sanity",
            "reproduction_gate": repro_r1,
            "mask_counts": mask_counts,
            "failures": sanity_failures,
            "expected_range": [MASK_SIZE_MIN, MASK_SIZE_MAX],
            "reason": (
                f"Mask size(s) outside expected range [{MASK_SIZE_MIN},"
                f"{MASK_SIZE_MAX}]: {sanity_failures}. Likely GRCh38 coords "
                "leaked into a GRCh37 workflow, or region mis-specified."),
        }, out_dir / "results.json")
        logger.error("Mask sanity FAIL: %s", sanity_failures)
        return 40

    # --------------------- Per-pattern ρ & bootstrap ---------------------
    patterns = {
        "MHC_only": masks["MHC"],
        "17q21_only": masks["17q21"],
        "8p23_only": masks["8p23"],
        "all_three": masks["MHC"] | masks["17q21"] | masks["8p23"],
    }

    pops_full = base["PoPS_Score"].to_numpy(dtype=float)
    magma_full = base["MAGMA_Z"].to_numpy(dtype=float)

    def _ci(samples):
        lo, hi, med = percentile_ci(samples)
        return {"ci_lo": lo, "ci_hi": hi, "median": med}

    # --------------------- SHARED paired bootstrap indices ---------------
    # WHY one shared index matrix (fix for audit MAJOR #1, unpaired Δρ):
    # paired Δρ requires the SAME underlying resample of the full sample
    # for ρ_full and ρ_masked. We draw ONE matrix of full-sample indices
    # (shape n_boot × n_full) and for each pattern compute both ρ_full
    # (on the entire resample) and ρ_masked (same resample, restricted to
    # non-masked positions). Previously we drew two INDEPENDENT matrices
    # (different shapes), yielding uncorrelated ρ's and inflated Δρ CI.
    idx_mat_full = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, n_full, size=(BOOTSTRAP_N, n_full)
    )
    logger.info("Shared bootstrap idx matrix: shape=%s seed=%d",
                idx_mat_full.shape, BOOTSTRAP_SEED)

    # Point estimate of ρ on the full intersection — shared across patterns.
    rho_full_p_point, _ = pearsonr(pops_full, magma_full)
    rho_full_s_point, _ = spearmanr(pops_full, magma_full)

    results_per_pattern: dict = {}
    for pat_name, mask_any in patterns.items():
        # Boolean flag aligned with base rows: True for masked (LD-blob).
        masked_flag = np.asarray(mask_any, dtype=bool)
        n_keep_point = int((~masked_flag).sum())

        # Point estimates on the unresampled data.
        rho_int_p, _ = pearsonr(pops_full[~masked_flag],
                                magma_full[~masked_flag])
        rho_int_s, _ = spearmanr(pops_full[~masked_flag],
                                 magma_full[~masked_flag])
        point_delta_p = float(rho_int_p - rho_full_p_point)
        point_delta_s = float(rho_int_s - rho_full_s_point)

        # PAIRED bootstrap: one resample → two ρ's (full + masked) on the
        # SAME bag of indices. Mask is applied AFTER resampling.
        boot_rho_full_p = np.zeros(BOOTSTRAP_N)
        boot_rho_full_s = np.zeros(BOOTSTRAP_N)
        boot_rho_masked_p = np.zeros(BOOTSTRAP_N)
        boot_rho_masked_s = np.zeros(BOOTSTRAP_N)
        boot_n_keep = np.zeros(BOOTSTRAP_N, dtype=int)

        for i in range(BOOTSTRAP_N):
            resample_idx = idx_mat_full[i]
            x_full = pops_full[resample_idx]
            y_full = magma_full[resample_idx]
            rpp_full, _ = pearsonr(x_full, y_full)
            rss_full, _ = spearmanr(x_full, y_full)
            boot_rho_full_p[i] = float(rpp_full)
            boot_rho_full_s[i] = float(rss_full)

            # "Keep" = non-masked positions of the SAME resample.
            keep_i = ~masked_flag[resample_idx]
            boot_n_keep[i] = int(keep_i.sum())
            if keep_i.sum() < 3:
                # Degenerate resample (too few non-masked draws); record NaN.
                boot_rho_masked_p[i] = float("nan")
                boot_rho_masked_s[i] = float("nan")
                continue
            rpp_masked, _ = pearsonr(x_full[keep_i], y_full[keep_i])
            rss_masked, _ = spearmanr(x_full[keep_i], y_full[keep_i])
            boot_rho_masked_p[i] = float(rpp_masked)
            boot_rho_masked_s[i] = float(rss_masked)

        delta_p = boot_rho_masked_p - boot_rho_full_p
        delta_s = boot_rho_masked_s - boot_rho_full_s
        d_lo_p, d_hi_p, d_med_p = percentile_ci(delta_p)
        d_lo_s, d_hi_s, d_med_s = percentile_ci(delta_s)

        results_per_pattern[pat_name] = {
            "n_masked_genes": int(masked_flag.sum()),
            "n_intersection": int(n_keep_point),
            "rho_full_pearson_point": float(rho_full_p_point),
            "rho_full_spearman_point": float(rho_full_s_point),
            "rho_intersection_pearson_point": float(rho_int_p),
            "rho_intersection_spearman_point": float(rho_int_s),
            "delta_rho_pearson_point": point_delta_p,
            "delta_rho_spearman_point": point_delta_s,
            "delta_rho_pearson_ci": {"ci_lo": d_lo_p, "ci_hi": d_hi_p,
                                      "median": d_med_p},
            "delta_rho_spearman_ci": {"ci_lo": d_lo_s, "ci_hi": d_hi_s,
                                       "median": d_med_s},
            "rho_intersection_pearson_ci": _ci(boot_rho_masked_p),
            "rho_intersection_spearman_ci": _ci(boot_rho_masked_s),
            "rho_full_pearson_ci": _ci(boot_rho_full_p),
            "rho_full_spearman_ci": _ci(boot_rho_full_s),
            "bootstrap_n_keep_mean": float(boot_n_keep.mean()),
            "bootstrap_n_keep_min": int(boot_n_keep.min()),
        }
        logger.info(
            "Pattern %s: n_masked=%d n_int=%d Δρ_pearson=%.4f CI=[%.4f,%.4f] "
            "boot_keep_mean=%.0f",
            pat_name, int(masked_flag.sum()), n_keep_point,
            point_delta_p, d_lo_p, d_hi_p, boot_n_keep.mean(),
        )

    # --------------------- Classification ---------------------
    d_all = results_per_pattern["all_three"]
    d_mhc = results_per_pattern["MHC_only"]
    d_17 = results_per_pattern["17q21_only"]
    d_8 = results_per_pattern["8p23_only"]

    classification = classify(
        delta_all=(d_all["delta_rho_pearson_point"],
                   d_all["delta_rho_pearson_ci"]["ci_lo"],
                   d_all["delta_rho_pearson_ci"]["ci_hi"]),
        delta_mhc=(d_mhc["delta_rho_pearson_point"],
                   d_mhc["delta_rho_pearson_ci"]["ci_lo"],
                   d_mhc["delta_rho_pearson_ci"]["ci_hi"]),
        delta_17q21=(d_17["delta_rho_pearson_point"],
                      d_17["delta_rho_pearson_ci"]["ci_lo"],
                      d_17["delta_rho_pearson_ci"]["ci_hi"]),
        delta_8p23=(d_8["delta_rho_pearson_point"],
                     d_8["delta_rho_pearson_ci"]["ci_lo"],
                     d_8["delta_rho_pearson_ci"]["ci_hi"]),
    )
    logger.info("Classification: %s", classification)

    wall = time.time() - t0
    results = {
        "status": "ok",
        "batch": "056",
        "sub": "b",
        "wall_s": wall,
        "reproduction_gate_R1": repro_r1,
        "n_full_intersection": int(n_full),
        "mask_coordinates_build": "GRCh37",
        "mask_definitions": LD_MASKS,
        "mask_counts": mask_counts,
        "mask_size_expected_range": [MASK_SIZE_MIN, MASK_SIZE_MAX],
        "per_pattern": results_per_pattern,
        "bootstrap": {"n_boot": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
        "decision_classification": classification,
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("Sub-B wrote %s (wall=%.1fs)", out_dir / "results.json", wall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
