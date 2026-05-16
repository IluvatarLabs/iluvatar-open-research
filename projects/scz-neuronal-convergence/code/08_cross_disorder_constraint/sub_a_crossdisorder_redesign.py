#!/usr/bin/env python3
"""batch_057 Sub-A — F-055-01 cross-disorder covariate-adjusted OLS redesign.

Implements brief_v2.md §Sub-A exactly. Extends batch_056 Sub-D by:
  1. Adding log10(NSNPS+1) as a covariate (De Leeuw 2018 + VERA dream_055 #2).
  2. Replacing the relative-difference Huber check with an absolute-difference
     check: |β_OLS − β_Huber| < 0.3 · SE_OLS (brief_v2 line 24 / L056_01 fix).
  3. Running a 4-cell robustness envelope: {abs-diff / rel-diff Huber} ×
     {with NSNPS / without NSNPS}; FRAGILE flag if pattern changes.
  4. LOEUF sensitivity on SCZ + MDD + Height (swap `lof_pLI` → `lof_oe_ci_upper`).
  5. Per-category β_1 stratification on SCZ ONLY, flagged descriptive_only.
  6. BH-FDR within the 8-test Sub-A family (critic-split scope).
  7. First-match pattern classification: SCZ-SPECIFIC → SCZ+ONE-PSYCH →
     PAN-PSYCHIATRIC → UNIVERSAL-BASELINE → INTERMEDIATE.

Reproduction gate R1: SCZ β_1_adj ∈ [+2.5, +3.5].

Writes output to experiments/batch_057/output/sub_a/results.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    B3_BIOLOGICAL_CATEGORY,
    B3_GENES,
    BATCH055B_WORK,
    BH_Q,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_GENELOC,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    REPRO_R1_SCZ_BETA_HI,
    REPRO_R1_SCZ_BETA_LO,
    abs_diff_huber_check,
    atomic_write_json,
    bh_fdr,
    build_sub_a_frame,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    rel_diff_huber_check,
    setup_logger,
    sha256_file,
)

import numpy as np
import pandas as pd
import statsmodels.api as sm


DISORDERS = ["scz", "bip", "mdd", "asd", "adhd",
              "ibd_delange2017", "height", "alzheimers"]
PSYCHIATRIC = {"scz", "bip", "mdd", "asd", "adhd"}
NEGATIVE_CONTROLS = {"ibd_delange2017", "height", "alzheimers"}

# Brief_v2 decision thresholds (§Sub-A PREDICTION).
SCZ_SPEC_MIN = 1.5         # SCZ β ≥ 1.5σ
CROSS_MIN = 1.0            # other β ≥ 1.0σ for committing
PAN_PSYCH_MIN = 1.0
UNIV_BASE_MIN = 0.5
UNIV_BASE_P = 0.1
MIN_EFFECT = 0.5           # §Sub-A MEASUREMENT "Min effect size that matters"

HUBER_ABS_THRESHOLD = 0.3  # multiplier on SE_OLS
HUBER_REL_THRESHOLD = 0.20 # matches iter_056 convention
MIN_N_UNIVERSE = 15000


def fit_ols(frame: pd.DataFrame, b3_col: str, covariates: list[str]) -> dict:
    """OLS MAGMA_Z ~ b3 + covariates. β_1 is coef on b3."""
    X = frame[[b3_col] + covariates].to_numpy(dtype=float)
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(y, Xc).fit()
    beta_1 = float(ols.params[1])
    se_1 = float(ols.bse[1])
    t_1 = float(ols.tvalues[1])
    p_two = float(ols.pvalues[1])
    p_one = float(p_two / 2.0 if t_1 > 0 else 1.0 - p_two / 2.0)
    return {
        "beta_1": beta_1, "se_1": se_1, "t": t_1,
        "p_one_sided": p_one, "p_two_sided": p_two,
        "n": int(X.shape[0]),
        "covariates_used": covariates,
        "r_squared": float(ols.rsquared),
    }


def fit_huber(frame: pd.DataFrame, b3_col: str, covariates: list[str]) -> dict:
    """Huber RLM for β_1 robustness. Returns status:failed on non-finite."""
    X = frame[[b3_col] + covariates].to_numpy(dtype=float)
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
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
    return {"status": "ok", "beta_1": beta_1, "se_1": se_1}


def run_envelope(frame: pd.DataFrame) -> dict:
    """4-cell robustness envelope for one disorder.

    Cells: {abs-diff / rel-diff Huber check} × {with NSNPS / without NSNPS}.
    """
    # Primary covariate set (brief_v2 line 21-23, including NSNPS).
    full_covs = ["log10_gene_length", "lof_pLI",
                 "log10_exp_lof_plus1", "log10_NSNPS_plus1"]
    no_nsnps_covs = [c for c in full_covs if c != "log10_NSNPS_plus1"]

    results: dict = {}
    for tag, covs in (("with_nsnps", full_covs),
                       ("without_nsnps", no_nsnps_covs)):
        ols = fit_ols(frame, "in_set", covs)
        huber = fit_huber(frame, "in_set", covs)
        if huber.get("status") == "ok":
            pass_abs, ratio_abs, thr_abs = abs_diff_huber_check(
                ols["beta_1"], huber["beta_1"], ols["se_1"],
                HUBER_ABS_THRESHOLD,
            )
            pass_rel, ratio_rel, thr_rel = rel_diff_huber_check(
                ols["beta_1"], huber["beta_1"], HUBER_REL_THRESHOLD,
            )
        else:
            pass_abs = pass_rel = False
            ratio_abs = ratio_rel = float("nan")
            thr_abs = HUBER_ABS_THRESHOLD
            thr_rel = HUBER_REL_THRESHOLD

        results[tag] = {
            "covariates": covs,
            "ols": ols,
            "huber": huber,
            "huber_abs_diff": {
                "pass": pass_abs, "ratio": ratio_abs, "threshold": thr_abs,
            },
            "huber_rel_diff": {
                "pass": pass_rel, "ratio": ratio_rel, "threshold": thr_rel,
            },
        }
    return results


def run_loeuf_sensitivity(disorder: str, gnomad_df: pd.DataFrame,
                            annot_df: pd.DataFrame,
                            b3_ensg_set: set[str]) -> dict:
    """Swap lof_pLI → lof_oe_ci_upper (LOEUF) in the primary model."""
    frame = build_sub_a_frame(disorder, gnomad_df, annot_df, b3_ensg_set)
    # Drop rows without LOEUF.
    frame = frame.dropna(subset=["lof_oe_ci_upper"]).copy()
    # Rename to standardize.
    covs = ["log10_gene_length", "lof_oe_ci_upper",
            "log10_exp_lof_plus1", "log10_NSNPS_plus1"]
    if frame.empty or frame["in_set"].sum() < 10:
        return {"status": "skipped",
                "reason": f"n={len(frame)} b3={int(frame['in_set'].sum() if len(frame) else 0)}"}
    ols = fit_ols(frame, "in_set", covs)
    huber = fit_huber(frame, "in_set", covs)
    if huber.get("status") == "ok":
        pass_abs, ratio_abs, thr_abs = abs_diff_huber_check(
            ols["beta_1"], huber["beta_1"], ols["se_1"],
            HUBER_ABS_THRESHOLD,
        )
    else:
        pass_abs = False
        ratio_abs = float("nan")
        thr_abs = HUBER_ABS_THRESHOLD
    return {
        "status": "ok",
        "covariates": covs,
        "ols": ols, "huber": huber,
        "huber_abs_diff": {"pass": pass_abs, "ratio": ratio_abs,
                            "threshold": thr_abs},
    }


def run_category_stratified(frame: pd.DataFrame,
                             b3_ensg_to_cat: dict[str, str],
                             b3_ensg_to_sym: dict[str, str]) -> dict:
    """SCZ-only, descriptive: per-category β_1 (cell-adhesion / enzyme /
    scaffold).

    Reports ONLY the 3 categories explicitly named in brief_v2 line 54 for
    audit clarity (cell-adhesion n=3, enzyme n=5, scaffold n=5). Other
    categories (receptor, ion-channel, motor, misc) have n<3 per brief.
    """
    cats_of_interest = {"cell-adhesion", "enzyme", "scaffold"}
    results: dict = {}
    covs = ["log10_gene_length", "lof_pLI",
            "log10_exp_lof_plus1", "log10_NSNPS_plus1"]
    for cat in cats_of_interest:
        cat_ensg = {ensg for ensg, c in b3_ensg_to_cat.items() if c == cat}
        frame_c = frame.copy()
        frame_c["in_set"] = frame_c["ENSGID"].isin(cat_ensg).astype(int)
        n_cat_in_universe = int(frame_c["in_set"].sum())
        if n_cat_in_universe < 3:
            results[cat] = {"n": n_cat_in_universe, "status": "n_too_small"}
            continue
        ols = fit_ols(frame_c, "in_set", covs)
        ols["n_in_category"] = n_cat_in_universe
        ols["descriptive_only"] = True
        ols["not_decision_binding"] = True
        results[cat] = ols
    return results


def classify_patterns(per_disorder: dict[str, dict]) -> dict:
    """Apply brief_v2 §Sub-A DECISION RULE first-match ordering."""
    ds_ok = [d for d in DISORDERS if per_disorder.get(d, {}).get("status") == "ok"]
    if len(ds_ok) < 8:
        return {"classification": "UNINTERPRETABLE",
                "reason": f"only {len(ds_ok)}/8 disorders ran successfully"}

    # Collect primary (abs-diff Huber + with-NSNPS) β, SE, p_one.
    primary = {}
    for d in ds_ok:
        cell = per_disorder[d]["envelope"]["with_nsnps"]
        primary[d] = {
            "beta": cell["ols"]["beta_1"],
            "se": cell["ols"]["se_1"],
            "p_one": cell["ols"]["p_one_sided"],
            "abs_pass": cell["huber_abs_diff"]["pass"],
        }

    # BH-FDR within the 8-test family.
    pvals = [primary[d]["p_one"] for d in DISORDERS]
    qvals = bh_fdr(pvals)
    q_by_d = dict(zip(DISORDERS, qvals))
    for d, q in q_by_d.items():
        primary[d]["bh_q"] = q

    # UNINTERPRETABLE check (a): ≥ 2 disorders fail abs-diff Huber AND affect class.
    huber_fail = [d for d in DISORDERS if not primary[d]["abs_pass"]]
    # UNINTERPRETABLE check (c): robustness envelope pattern changes on ≥ 3 disorders.
    fragile = []
    for d in DISORDERS:
        env = per_disorder[d]["envelope"]
        # Check if any cell's β_1 disagrees with primary β_1 by > 0.5σ (FRAGILE).
        primary_beta = env["with_nsnps"]["ols"]["beta_1"]
        flips = 0
        for tag in ("without_nsnps",):
            other = env[tag]["ols"]["beta_1"]
            if abs(other - primary_beta) > 0.5:
                flips += 1
        if flips > 0:
            fragile.append(d)

    # First-match pattern classification.
    scz_b = primary["scz"]["beta"]
    scz_q = primary["scz"]["bh_q"]
    psych_others = [d for d in PSYCHIATRIC if d != "scz"]
    psych_cross = [d for d in psych_others
                    if primary[d]["beta"] >= CROSS_MIN
                    and primary[d]["bh_q"] < BH_Q]
    nc_cross_at_1 = [d for d in NEGATIVE_CONTROLS
                     if primary[d]["beta"] >= CROSS_MIN
                     and primary[d]["bh_q"] < BH_Q]
    nc_cross_at_05 = [d for d in NEGATIVE_CONTROLS
                      if primary[d]["beta"] >= UNIV_BASE_MIN
                      and primary[d]["bh_q"] < UNIV_BASE_P]

    all_huber_pass = len(huber_fail) == 0

    scz_spec = (scz_b >= SCZ_SPEC_MIN and scz_q < BH_Q
                and len([d for d in DISORDERS if d != "scz"
                         and primary[d]["beta"] >= CROSS_MIN
                         and primary[d]["bh_q"] < BH_Q]) <= 1
                and all_huber_pass)
    if scz_spec:
        return {"classification": "SCZ-SPECIFIC",
                "reason": ("SCZ β≥1.5σ q<0.05; ≤1 other crosses β≥1.0σ q<0.05; "
                           "all 8 pass abs-diff Huber"),
                "bh_q": q_by_d, "huber_fail": huber_fail,
                "fragile_disorders": fragile}

    scz_one_psych = (scz_b >= SCZ_SPEC_MIN and scz_q < BH_Q
                      and len(psych_cross) == 1
                      and len(nc_cross_at_1) == 0
                      and all_huber_pass)
    if scz_one_psych:
        return {"classification": "SCZ+ONE-PSYCHIATRIC",
                "reason": f"SCZ β≥1.5σ; exactly one psych partner: {psych_cross}",
                "bh_q": q_by_d, "huber_fail": huber_fail,
                "partner": psych_cross, "fragile_disorders": fragile}

    # PAN-PSYCHIATRIC: SCZ β≥1.5σ AND ≥ 2 of {BIP, MDD, ASD, ADHD} cross β≥1.0σ
    # q<0.05 AND all 3 non-psych NCs have β<0.5σ OR q>0.1 AND all 8 pass Huber.
    psych_2 = len(psych_cross) >= 2
    nc_small = all(
        (primary[d]["beta"] < UNIV_BASE_MIN or primary[d]["bh_q"] > UNIV_BASE_P)
        for d in NEGATIVE_CONTROLS
    )
    pan_psych = (scz_b >= SCZ_SPEC_MIN and psych_2 and nc_small
                  and all_huber_pass)
    if pan_psych:
        return {"classification": "PAN-PSYCHIATRIC",
                "reason": f"SCZ β≥1.5σ; ≥2 psych partners: {psych_cross}; "
                           "NCs small; all Huber pass",
                "bh_q": q_by_d, "huber_fail": huber_fail,
                "fragile_disorders": fragile}

    # UNIVERSAL-BASELINE: ≥ 6 of 8 disorders β≥0.5σ q<0.1 AND ≥2 of 3 NCs.
    cross_at_baseline = [d for d in DISORDERS
                          if primary[d]["beta"] >= UNIV_BASE_MIN
                          and primary[d]["bh_q"] < UNIV_BASE_P]
    univ = (len(cross_at_baseline) >= 6 and len(nc_cross_at_05) >= 2
            and all_huber_pass)
    if univ:
        return {"classification": "UNIVERSAL-BASELINE",
                "reason": f"≥6/8 disorders β≥0.5σ q<0.1 "
                           f"({cross_at_baseline}); ≥2 NCs",
                "bh_q": q_by_d, "huber_fail": huber_fail,
                "fragile_disorders": fragile}

    # UNINTERPRETABLE conditions.
    if len(huber_fail) >= 2:
        return {"classification": "UNINTERPRETABLE",
                "reason": (f"{len(huber_fail)} disorders fail abs-diff "
                            f"Huber: {huber_fail}"),
                "bh_q": q_by_d, "huber_fail": huber_fail}
    if len(fragile) >= 3:
        return {"classification": "UNINTERPRETABLE",
                "reason": f"FRAGILE on {len(fragile)} disorders: {fragile}",
                "bh_q": q_by_d, "fragile_disorders": fragile}

    # INTERMEDIATE committing conclusion (brief_v2 line 42).
    return {
        "classification": "INTERMEDIATE",
        "reason": ("None of SCZ-SPEC/+ONE-PSYCH/PAN-PSYCH/UNIV fired. "
                    "Per brief_v2: F-055-01 stays SUGGESTED; Sub-A is NOT "
                    "informative for cross-disorder classification under "
                    "current covariate set. iter_058 requires alternative "
                    "framework."),
        "bh_q": q_by_d, "huber_fail": huber_fail,
        "fragile_disorders": fragile,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_057 Sub-A")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: subset to 2 disorders, skip "
                             "LOEUF + category for speed.")
    args = parser.parse_args()

    logger = setup_logger("batch_057.sub_a", LOGS_DIR / "sub_a.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "sub_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load shared inputs.
    gnomad = load_gnomad_per_brief_v2()
    annot = load_gene_annot()
    # Map B3 symbols → ENSGID.
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

    b3_ensg_to_cat = {
        b3_sym_to_ensg[s]: c
        for s, c in B3_BIOLOGICAL_CATEGORY.items()
        if s in b3_sym_to_ensg
    }

    disorders_to_run = DISORDERS if not args.smoke else ["scz", "adhd"]

    per_disorder: dict = {}
    for d in disorders_to_run:
        try:
            frame = build_sub_a_frame(d, gnomad, annot, b3_ensg_set)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Load failed for %s", d)
            per_disorder[d] = {"status": "failed", "reason": str(exc)}
            continue
        n = len(frame)
        n_b3 = int(frame["in_set"].sum())
        logger.info("%s: n=%d B3=%d", d, n, n_b3)
        if n < MIN_N_UNIVERSE or n_b3 < 10:
            per_disorder[d] = {"status": "failed",
                                "reason": f"n={n} b3={n_b3}"}
            continue

        envelope = run_envelope(frame)
        item = {
            "status": "ok",
            "n_gene_universe": n,
            "n_b3_in_universe": n_b3,
            "envelope": envelope,
        }
        # Category stratification SCZ only.
        if d == "scz" and not args.smoke:
            item["category_stratified_descriptive_only"] = {
                "flag": {"descriptive_only": True,
                          "not_decision_binding": True},
                "results": run_category_stratified(
                    frame, b3_ensg_to_cat, b3_sym_to_ensg,
                ),
            }
        per_disorder[d] = item

    # LOEUF sensitivity (SCZ + MDD + Height).
    loeuf = {}
    if not args.smoke:
        for d in ("scz", "mdd", "height"):
            try:
                loeuf[d] = run_loeuf_sensitivity(d, gnomad, annot, b3_ensg_set)
            except Exception as exc:  # noqa: BLE001
                logger.exception("LOEUF sensitivity failed for %s", d)
                loeuf[d] = {"status": "failed", "reason": str(exc)}

    # Pattern classification + BH-FDR.
    if not args.smoke and all(d in per_disorder for d in DISORDERS):
        classification = classify_patterns(per_disorder)
    else:
        classification = {"classification": "SMOKE_SKIPPED",
                           "reason": "smoke test; not all disorders run"}

    # Reproduction gate R1.
    scz_primary_beta = (per_disorder.get("scz", {}).get("envelope", {})
                        .get("with_nsnps", {}).get("ols", {}).get("beta_1"))
    repro_r1 = {
        "target_lo": REPRO_R1_SCZ_BETA_LO,
        "target_hi": REPRO_R1_SCZ_BETA_HI,
        "scz_beta_1_adj_with_nsnps": scz_primary_beta,
        "pass": bool(scz_primary_beta is not None
                     and REPRO_R1_SCZ_BETA_LO <= scz_primary_beta <= REPRO_R1_SCZ_BETA_HI),
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
        "batch": "057", "sub": "a",
        "wall_s": time.time() - t0,
        "smoke": bool(args.smoke),
        "model": ("MAGMA_Z ~ in_set + log10(gene_length) + lof_pLI + "
                   "log10(exp_lof+1) + log10(NSNPS+1)"),
        "disorders": disorders_to_run,
        "reproduction_gate_R1": repro_r1,
        "per_disorder": per_disorder,
        "loeuf_sensitivity": loeuf,
        "decision_classification": classification,
        "provenance_sha256": provenance,
        "brief_contract": {
            "bh_fdr_family_size": 8,
            "bh_q": BH_Q,
            "huber_abs_threshold_x_se": HUBER_ABS_THRESHOLD,
            "category_stratification_flag":
                "descriptive_only, not_decision_binding",
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("Sub-A wrote %s (wall=%.1fs)",
                out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
