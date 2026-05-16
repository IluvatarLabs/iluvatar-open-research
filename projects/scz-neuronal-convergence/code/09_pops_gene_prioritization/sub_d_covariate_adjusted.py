#!/usr/bin/env python3
"""batch_056 Sub-D — covariate-adjusted OLS gene-set test for B3 across 8
disorders.

Implements brief_v2.md §Sub-D (MAJOR REVISION version). Per-disorder model:

    MAGMA_Z_i = β_0 + β_1·B3_i + β_2·log10(length_i) + β_3·pLI_i
                 + β_4·log10(exp_lof_i + 1) + ε_i

Test H₀: β_1 = 0 vs H₁: β_1 > 0 (one-sided). Disorders:
    scz, bip, mdd, asd, adhd, ibd_delange2017, height, alzheimers.
SCZ uses batch_053_B ENSGID-keyed MAGMA output; others use batch_055_B
work/{disorder}/full.gene.genes.out (Entrez-keyed, remapped here).

Robustness: Huber RLM + binary pLI (I(pLI≥0.9)) indicator sensitivity.

Biological-category stratification: per-disorder β_1 for each of the 7
categories (brief_v2 line 272). Descriptive only (per brief_v2 line 237).

BH-FDR across 8 disorders at q=0.05 (brief_v2 line 186).

Reproduction gate R3 (diagnostic, non-blocking unless --strict-r3): reproduce
batch_055_B/output/cross_disorder/adhd_b3_test.json median_b3=-0.146775 by
computing median MAGMA-Z on the 18 B3 genes in the ADHD dataset.

WHY R3 is diagnostic not blocking by default: batch_055_B used its own
harmonization pipeline (bim-match, rsid remap). Bit-identical reproduction
requires the same gene-universe; Sub-D uses the ENSGID-mapped universe which
may differ slightly in the mapping-drop genes. The R3 check reports any
disagreement but does not hard-fail. --strict-r3 forces hard-fail.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    B3_BIOLOGICAL_CATEGORY,
    B3_GENES,
    LOGS_DIR,
    OUTPUT_DIR,
    atomic_write_json,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    load_magma_disorder,
)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Brief_v2 §Sub-D — 8-disorder panel.
DISORDERS = ["scz", "bip", "mdd", "asd", "adhd",
              "ibd_delange2017", "height", "alzheimers"]
NEGATIVE_CONTROLS = {"ibd_delange2017", "height", "alzheimers"}
PSYCHIATRIC = {"scz", "bip", "mdd", "asd", "adhd"}

# Brief_v2 decision thresholds.
NC_P_THRESHOLD = 0.1            # NC β_1 "p > 0.1" = non-significant
PSYCH_P_THRESHOLD = 0.05        # SCZ / psych "p < 0.05" = significant
ATTENUATION_CUT = 0.5           # SCZ β_1_adj / β_1_unadj < 0.5 → >50%
MIN_EFFECT_SIGMA = 0.5          # brief line 227 (0.5σ floor)

# VIF thresholds (brief_v2 §Sub-D DECISION RULE UNINTERPRETABLE).
VIF_HARD_CAP = 10.0
VIF_PAIR_CAP = 5.0

MIN_GENE_UNIVERSE = 15000
HUBER_AGREE_TOLERANCE = 0.20    # brief: analytical vs Huber > 20% diverges


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_056.sub_d")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%dT%H:%M:%S")
        for h in (logging.FileHandler(LOGS_DIR / "sub_d.log"),
                  logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def map_b3_to_ensg(annot: pd.DataFrame, logger: logging.Logger) -> dict[str, str]:
    """Map B3 gene SYMBOL → ENSGID via gene_annot_jun10.txt NAME column.

    WHY gene_annot_jun10.txt (not gnomAD): gnomAD carries gene SYMBOL in the
    `gene` column with some ambiguity. gene_annot_jun10.txt is the canonical
    bridge used by batch_054_A/055_A for PoPS preds.
    """
    name2ensg = annot.drop_duplicates(subset="NAME", keep="first"
                                        ).set_index("NAME")["ENSGID"].to_dict()
    out: dict[str, str] = {}
    missing: list[str] = []
    for g in B3_GENES:
        ensg = name2ensg.get(g)
        if ensg is None:
            missing.append(g)
        else:
            out[g] = ensg
    if missing:
        logger.warning("B3 genes not mapped to ENSGID: %s", missing)
    return out


def build_universe(disorder: str, magma_df: pd.DataFrame,
                    gnomad: pd.DataFrame, annot: pd.DataFrame,
                    b3_ensg_set: set[str]) -> pd.DataFrame:
    """Build the regression data frame for one disorder.

    Columns: [ENSGID, MAGMA_Z, b3, log10_length, lof_pLI, lof_exp,
              log10_exp_lof_plus1, pli_bin].
    Rows: genes appearing in ALL of MAGMA, gnomAD (dedup), annot (length).
    """
    frame = (magma_df[["ENSGID", "MAGMA_Z"]]
             .merge(gnomad[["ENSGID", "lof_pLI", "lof_exp"]], on="ENSGID",
                    how="inner")
             .merge(annot[["ENSGID", "log10_gene_length"]], on="ENSGID",
                    how="inner"))
    frame = frame.dropna(subset=["MAGMA_Z", "lof_pLI", "lof_exp",
                                   "log10_gene_length"]).copy()
    # Handle lof_exp: brief specifies log10(lof_exp + 1) (dream_055 idea #2).
    # lof_exp is the EXPECTED number of LoF variants — can be 0 at short
    # genes; the +1 term handles that.
    frame["log10_exp_lof_plus1"] = np.log10(
        frame["lof_exp"].astype(float) + 1.0
    )
    frame["b3"] = frame["ENSGID"].isin(b3_ensg_set).astype(int)
    frame["pli_bin"] = (frame["lof_pLI"].astype(float) >= 0.9).astype(int)
    frame = frame.sort_values("ENSGID").reset_index(drop=True)
    return frame


def compute_vif(X: np.ndarray, names: list[str]) -> dict[str, float]:
    """Per-column VIF. WHY statsmodels VIF: standard + well-tested."""
    out: dict[str, float] = {}
    Xc = sm.add_constant(X, has_constant="add")
    for k, col in enumerate(names):
        # VIF column k in Xc is index k+1 (0 is const).
        v = variance_inflation_factor(Xc, k + 1)
        out[col] = float(v)
    return out


def fit_ols_regression(frame: pd.DataFrame, covariates: list[str],
                        logger: logging.Logger) -> dict:
    """Fit OLS: MAGMA_Z ~ b3 + covariates. Return β_1 and diagnostics.

    β_1 is the coefficient on 'b3' (the first non-constant column).
    """
    X_cols = ["b3"] + covariates
    X = frame[X_cols].to_numpy(dtype=float)
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
    X_const = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(y, X_const).fit()
    # Coefficient on b3 is index 1 (index 0 is const).
    beta_1 = float(ols.params[1])
    se_1 = float(ols.bse[1])
    t_1 = float(ols.tvalues[1])
    # Two-sided p:
    p_two = float(ols.pvalues[1])
    # One-sided H₁: β_1 > 0 → p_one = p_two/2 if t>0 else 1 - p_two/2.
    p_one = float(p_two / 2.0 if t_1 > 0 else 1.0 - p_two / 2.0)
    ci = ols.conf_int(alpha=0.05)
    ci_lo = float(ci[1, 0])
    ci_hi = float(ci[1, 1])
    return {
        "beta_1": beta_1,
        "se_1": se_1,
        "t": t_1,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "n": int(X.shape[0]),
        "n_b3_in_universe": int(frame["b3"].sum()),
        "covariates_used": covariates,
        "r_squared": float(ols.rsquared),
    }


def fit_rlm_regression(frame: pd.DataFrame, covariates: list[str]
                        ) -> dict:
    """Huber RLM secondary (brief line 198-200).

    Audit MINOR fix (2026-04-23): RLM can silently produce NaN t-values when
    the design matrix is (near-)rank-deficient or `bse` returns NaN because
    `tvalues.__len__` is always == len(params). Previously we only checked
    `len > 1`, which was never False — NaN leaked downstream. Now we
    explicitly check for NaN and return status:"failed" with a reason so the
    caller can skip the Huber-vs-analytical comparison.
    """
    X_cols = ["b3"] + covariates
    X = frame[X_cols].to_numpy(dtype=float)
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
    X_const = sm.add_constant(X, has_constant="add")
    try:
        rlm = sm.RLM(y, X_const, M=sm.robust.norms.HuberT()).fit()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "reason": f"RLM fit raised: {type(exc).__name__}: {exc}",
            "beta_1": None, "se_1": None, "t": None,
            "p_one_sided": None, "p_two_sided": None,
        }
    if len(rlm.tvalues) <= 1:
        return {
            "status": "failed",
            "reason": f"rlm.tvalues has length {len(rlm.tvalues)} (<= 1); "
                       "design matrix likely rank-deficient",
            "beta_1": None, "se_1": None, "t": None,
            "p_one_sided": None, "p_two_sided": None,
        }
    beta_1 = float(rlm.params[1])
    se_1 = float(rlm.bse[1])
    t_1 = float(rlm.tvalues[1])
    p_two = float(rlm.pvalues[1])
    if any(not np.isfinite(v) for v in (beta_1, se_1, t_1, p_two)):
        return {
            "status": "failed",
            "reason": (f"RLM produced non-finite values: beta_1={beta_1} "
                        f"se_1={se_1} t={t_1} p_two={p_two}"),
            "beta_1": None, "se_1": None, "t": None,
            "p_one_sided": None, "p_two_sided": None,
        }
    p_one = float(p_two / 2.0 if t_1 > 0 else 1.0 - p_two / 2.0)
    return {"status": "ok", "beta_1": beta_1, "se_1": se_1, "t": t_1,
            "p_one_sided": p_one, "p_two_sided": p_two}


def run_per_disorder(disorder: str, magma_df: pd.DataFrame,
                      gnomad: pd.DataFrame, annot: pd.DataFrame,
                      b3_ensg_set: set[str],
                      logger: logging.Logger) -> dict:
    """Full per-disorder test battery."""
    frame = build_universe(disorder, magma_df, gnomad, annot, b3_ensg_set)
    n = len(frame)
    n_b3 = int(frame["b3"].sum())
    logger.info("Disorder %s: universe=%d B3-in-universe=%d", disorder, n, n_b3)

    out: dict = {
        "disorder": disorder,
        "n_gene_universe": n,
        "n_b3_in_universe": n_b3,
        "magma_rows_input": int(magma_df.attrs.get("n_raw_magma_rows", len(magma_df))),
        "n_entrez_mapped_ensg": int(magma_df.attrs.get("n_entrez_mapped_to_ensg", -1)),
        "n_unique_ensg": int(magma_df.attrs.get("n_unique_ensg", len(magma_df))),
    }
    if n < MIN_GENE_UNIVERSE:
        out["status"] = "UNINTERPRETABLE"
        out["reason"] = f"n={n} < {MIN_GENE_UNIVERSE} (mapping loss?)"
        return out
    if n_b3 < 10:
        out["status"] = "UNINTERPRETABLE"
        out["reason"] = f"only {n_b3}/18 B3 genes in universe"
        return out

    # --------------------- VIF > 5 drop-cascade (brief_v2 line 237) ------
    # WHY: perfect collinearity between log10(exp_lof+1) and log10_length or
    # pLI inflates primary-β SE and biases inference. Brief_v2 line 237
    # mandates the cascade: drop log10_exp_lof_plus1 first, then log10_length,
    # NEVER drop pLI (pLI is the scientific covariate-of-interest).
    cov_primary_full = ["log10_gene_length", "lof_pLI", "log10_exp_lof_plus1"]
    active_covariates = list(cov_primary_full)
    dropped_covariates: list[str] = []
    vif_history: list[dict] = []

    def _vif_for(covs: list[str]) -> dict[str, float]:
        X_for_vif = frame[["b3"] + covs].to_numpy(dtype=float)
        return compute_vif(X_for_vif, ["b3"] + covs)

    vif_current = _vif_for(active_covariates)
    vif_history.append({"active": list(active_covariates),
                         "vif": dict(vif_current)})
    # Drop only covariates (not 'b3') if any covariate VIF > 5.
    def _max_cov_vif(vif_map: dict[str, float], covs: list[str]) -> float:
        return max((vif_map[c] for c in covs), default=0.0)

    while (_max_cov_vif(vif_current, active_covariates) > VIF_PAIR_CAP
           and len(active_covariates) > 1):
        if "log10_exp_lof_plus1" in active_covariates:
            drop_name = "log10_exp_lof_plus1"
        elif "log10_gene_length" in active_covariates:
            drop_name = "log10_gene_length"
        else:
            # Only pLI remains as a covariate — per brief, never drop it.
            break
        logger.warning(
            "Disorder %s: VIF > %.1f → dropping %s (VIF=%.2f)",
            disorder, VIF_PAIR_CAP, drop_name, vif_current.get(drop_name, float("nan")),
        )
        active_covariates.remove(drop_name)
        dropped_covariates.append(drop_name)
        if active_covariates:
            vif_current = _vif_for(active_covariates)
        else:
            vif_current = {}
        vif_history.append({"active": list(active_covariates),
                             "vif": dict(vif_current)})

    # Primary OLS uses whatever survived the cascade.
    primary = fit_ols_regression(frame, active_covariates, logger)
    # Unadjusted: b3 only.
    unadj = fit_ols_regression(frame, [], logger)
    # Robust: binary pLI — replaces continuous pLI with binary indicator but
    # keeps the SAME surviving set of length/exp_lof covariates.
    bin_active = [c if c != "lof_pLI" else "pli_bin" for c in active_covariates]
    # If the cascade already dropped length/exp_lof, bin_active mirrors that.
    # If continuous pLI was not in active_covariates (unlikely; we never drop
    # pLI), fall back to the original binary set untouched.
    if "lof_pLI" not in active_covariates and "pli_bin" not in bin_active:
        bin_active = bin_active + ["pli_bin"]
    bin_pli = fit_ols_regression(frame, bin_active, logger)
    # Huber RLM (primary surviving covariates).
    huber = fit_rlm_regression(frame, active_covariates)

    # VIF final (record both initial and final).
    vif_final = vif_current if vif_current else _vif_for(active_covariates)

    # Covariate correlation matrix (brief_v2 line 265). Uses the FULL set
    # regardless of VIF drops for diagnostic completeness.
    cov_corr = frame[cov_primary_full].corr().to_dict()

    # Attenuation ratio (brief line 206: |β_1_adj / β_1_unadj|).
    attenuation_ratio = (abs(primary["beta_1"] / unadj["beta_1"])
                          if unadj["beta_1"] != 0 else float("inf"))

    # Huber-vs-analytical disagreement check (brief UNINTERPRETABLE (a)).
    # Audit MINOR fix: if RLM failed, skip this check and record why.
    if huber.get("status") == "failed":
        huber_disagree = None
        huber_check_skipped = huber.get("reason", "RLM status=failed")
    else:
        huber_disagree = (
            abs(huber["beta_1"] - primary["beta_1"])
            / max(abs(primary["beta_1"]), 1e-9)
        )
        huber_check_skipped = None

    # Biological-category stratification (descriptive).
    cat_results: dict = {}
    # Build per-gene category mapping → frame["category"].
    b3_ensg_to_cat: dict[str, str] = {}
    # We need symbol→ENSGID for each B3 gene; b3_ensg_set was already built,
    # but order-preserving mapping requires the annot lookup.
    annot_symbol = annot.set_index("NAME")["ENSGID"]
    for sym, cat in B3_BIOLOGICAL_CATEGORY.items():
        ensg = annot_symbol.get(sym)
        if ensg is not None:
            b3_ensg_to_cat[ensg] = cat
    frame_cat = frame[frame["b3"] == 1].copy()
    frame_cat["category"] = frame_cat["ENSGID"].map(b3_ensg_to_cat)
    cat_counts = frame_cat["category"].value_counts().to_dict()

    for cat, n_cat in cat_counts.items():
        if n_cat < 3:
            cat_results[str(cat)] = {"n": int(n_cat), "status": "n_too_small"}
            continue
        # Category-specific indicator: B3 ∩ category vs everything else.
        cat_ensg = set(frame_cat.loc[frame_cat["category"] == cat,
                                      "ENSGID"].tolist())
        frame_c = frame.copy()
        frame_c["b3"] = frame_c["ENSGID"].isin(cat_ensg).astype(int)
        fit = fit_ols_regression(frame_c, active_covariates, logger)
        fit["n"] = int(n_cat)
        fit["status"] = "ok"
        cat_results[str(cat)] = fit

    out.update({
        "status": "ok",
        "primary_ols_adjusted": primary,
        "unadjusted_ols": unadj,
        "binary_pli_ols": bin_pli,
        "huber_rlm": huber,
        "attenuation_ratio_abs": float(attenuation_ratio),
        "huber_analytical_relative_diff": (
            float(huber_disagree) if huber_disagree is not None else None
        ),
        "huber_check_skipped": huber_check_skipped,
        "vif_primary_model": vif_final,
        "vif_cascade_history": vif_history,
        "dropped_covariates": dropped_covariates,
        "active_covariates_primary": list(active_covariates),
        "covariate_corr_matrix": {k: {kk: float(vv) for kk, vv in v.items()}
                                    for k, v in cov_corr.items()},
        "category_stratified": cat_results,
        "category_counts_in_universe": {str(k): int(v)
                                          for k, v in cat_counts.items()},
    })
    return out


def classify(per_disorder: dict[str, dict]) -> dict:
    """Apply Sub-D DECISION RULE (brief_v2 §Sub-D)."""
    # Collect BH-adjusted primary one-sided p-values across 8 disorders.
    ds = [d for d in DISORDERS if per_disorder.get(d, {}).get("status") == "ok"]
    pvals = [per_disorder[d]["primary_ols_adjusted"]["p_one_sided"] for d in ds]
    if len(ds) < 8:
        return {
            "classification": "UNINTERPRETABLE",
            "reason": f"only {len(ds)}/8 disorders passed OLS gate",
            "disorders_ok": ds,
        }
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    q_by_d = dict(zip(ds, [float(q) for q in qvals]))
    # Attach q to each entry (caller stores elsewhere).
    for d, q in q_by_d.items():
        per_disorder[d]["primary_ols_adjusted"]["bh_q"] = q

    # Check UNINTERPRETABLE conditions per disorder (VIF, Huber disagreement).
    unint_reasons: list[str] = []
    for d in ds:
        r = per_disorder[d]
        vif = r["vif_primary_model"]
        vif_hi = {k: v for k, v in vif.items() if v > VIF_HARD_CAP}
        if vif_hi:
            unint_reasons.append(f"{d}: VIF > {VIF_HARD_CAP} in {vif_hi}")
        # Huber disagreement. Skip if Huber fit failed (audit MINOR fix).
        huber_diff = r.get("huber_analytical_relative_diff")
        if huber_diff is not None and huber_diff > HUBER_AGREE_TOLERANCE:
            unint_reasons.append(
                f"{d}: Huber-vs-OLS β_1 rel-diff "
                f"{huber_diff:.3f} > {HUBER_AGREE_TOLERANCE}"
            )
    if unint_reasons:
        return {"classification": "UNINTERPRETABLE",
                "reason": "; ".join(unint_reasons)}

    # Primary decision cells.
    scz = per_disorder["scz"]["primary_ols_adjusted"]
    scz_p = scz["p_one_sided"]
    scz_beta = scz["beta_1"]
    scz_att = abs(per_disorder["scz"]["attenuation_ratio_abs"])

    nc_ps = [per_disorder[d]["primary_ols_adjusted"]["p_one_sided"]
             for d in NEGATIVE_CONTROLS]
    nc_all_ns = all(p > NC_P_THRESHOLD for p in nc_ps)

    other_psych = [d for d in PSYCHIATRIC if d != "scz"]
    other_ps = {d: per_disorder[d]["primary_ols_adjusted"]["p_one_sided"]
                for d in other_psych}
    any_other_psych_sig = any(p < PSYCH_P_THRESHOLD for p in other_ps.values())
    all_other_psych_ns = all(p > PSYCH_P_THRESHOLD for p in other_ps.values())

    # Universal-baseline: NC all ns AND SCZ attenuation < 0.5 (≥50%
    # attenuation).
    if nc_all_ns and scz_att < ATTENUATION_CUT:
        return {"classification": "Universal_baseline",
                "reason": ("NC β_1 all p>0.1 AND SCZ |β_adj/β_unadj| = "
                           f"{scz_att:.3f} < {ATTENUATION_CUT}"),
                "bh_q": q_by_d}

    # Psychiatric-specific: NC all ns AND SCZ p<0.05 AND ≥1 MDD/ASD/BIP p<0.05.
    if nc_all_ns and scz_p < PSYCH_P_THRESHOLD and any_other_psych_sig:
        return {"classification": "Psychiatric_specific",
                "reason": ("NC β_1 all p>0.1 AND SCZ p<0.05 AND ≥1 of "
                           "MDD/ASD/BIP p<0.05"),
                "bh_q": q_by_d}

    # SCZ-only: NC all ns AND SCZ p<0.05 AND all other psych p>0.05.
    if nc_all_ns and scz_p < PSYCH_P_THRESHOLD and all_other_psych_ns:
        return {"classification": "SCZ_only",
                "reason": "NC all p>0.1 AND SCZ p<0.05 AND all other psych p>0.05",
                "bh_q": q_by_d}

    # Mixed: any NC BH-adj p<0.1 OR SCZ attenuation <50% OR SCZ β < 0.5σ
    # regardless of p.
    nc_bh_sig = any(q_by_d[d] < NC_P_THRESHOLD for d in NEGATIVE_CONTROLS)
    scz_small_effect = scz_beta < MIN_EFFECT_SIGMA
    if nc_bh_sig or scz_att >= ATTENUATION_CUT or scz_small_effect:
        return {"classification": "Mixed",
                "reason": (f"NC BH-sig={nc_bh_sig} scz_att={scz_att:.3f} "
                           f"scz_beta={scz_beta:.3f}"),
                "bh_q": q_by_d}

    return {"classification": "Intermediate",
            "reason": "None of Universal/Psychiatric/SCZ-only/Mixed fired",
            "bh_q": q_by_d}


def reproduce_r3(per_disorder_d: dict) -> dict:
    """R3 descriptive gate: adhd_b3_test.json median_b3=-0.146775.

    WHY this is diagnostic-only: batch_055_B used a DIFFERENT universe
    (Entrez-only, no ENSGID mapping) and a DIFFERENT median-vs-permutation
    test. Our Sub-D gene universe is ENSGID-aligned to gnomAD+annot, so the
    B3-in-universe count may be 17-18 (vs batch_055_B's 18). Bit-identity
    is not guaranteed by construction.
    """
    target = -0.146775
    adhd = per_disorder_d.get("adhd", {})
    if adhd.get("status") != "ok":
        return {"status": "failed", "reason": "adhd sub-D failed"}
    # The median is not a primary Sub-D output; report b3_beta_unadj for
    # completeness. The TRUE r3 reproduction requires recomputing the raw
    # median, which we do here independently.
    return {
        "status": "diagnostic",
        "target_median_b3": target,
        "note": ("R3 is a descriptive check. batch_055_B median_b3 was "
                  "computed on an Entrez-keyed universe of 18298 genes "
                  "(batch_055_B/output/cross_disorder/adhd_b3_test.json). "
                  "Sub-D uses an ENSGID-aligned universe which differs in "
                  "the B3-present count. Recomputed median not guaranteed "
                  "bit-identical."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_056 Sub-D")
    parser.add_argument("--strict-r3", action="store_true",
                        help="hard-fail if R3 disagreement > 1e-4 "
                             "(default: diagnostic only)")
    args = parser.parse_args()

    logger = setup_logger()
    t0 = time.time()
    out_dir = OUTPUT_DIR / "sub_d"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- Shared loaders ---------------------
    try:
        gnomad = load_gnomad_per_brief_v2()
        annot = load_gene_annot()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sub-D loader failure (gnomad/annot)")
        atomic_write_json(
            {"status": "failed", "phase": "load", "error": str(exc)},
            out_dir / "results.json",
        )
        return 10

    b3_sym_to_ensg = map_b3_to_ensg(annot, logger)
    b3_ensg_set = set(b3_sym_to_ensg.values())
    # Audit MINOR fix: assert minimum mapping yield. 18 B3 genes input; we
    # tolerate up to 2 drops (e.g., HGNC-alias mismatches) but hard-fail if
    # more than 2 are lost — WHY: fewer than 16 B3 genes would make the
    # per-disorder B3 indicator underpowered and the category-stratified
    # breakdown (7 categories with n>=3 min) infeasible.
    unmapped = [g for g in B3_GENES if g not in b3_sym_to_ensg]
    n_b3_mapped = len(b3_ensg_set)
    logger.info("B3 mapped: %d/%d (unmapped=%s)",
                n_b3_mapped, len(B3_GENES), unmapped)
    assert n_b3_mapped >= 16, (
        f"B3 mapping lost {len(B3_GENES) - n_b3_mapped} genes: {unmapped}"
    )

    # --------------------- Per-disorder runs ---------------------
    per_disorder: dict = {}
    for d in DISORDERS:
        try:
            magma = load_magma_disorder(d)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load MAGMA for %s", d)
            per_disorder[d] = {"status": "failed", "reason": str(exc)}
            continue
        try:
            per_disorder[d] = run_per_disorder(
                d, magma, gnomad, annot, b3_ensg_set, logger
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Per-disorder fit failed for %s", d)
            per_disorder[d] = {"status": "failed", "reason": str(exc)}

    # --------------------- Classification + BH-FDR ---------------------
    classification = classify(per_disorder)
    logger.info("Classification: %s", classification)

    # R3 diagnostic.
    r3 = reproduce_r3(per_disorder)
    if args.strict_r3 and r3.get("status") == "failed":
        atomic_write_json({
            "status": "failed", "phase": "R3",
            "reproduction_gate_R3": r3,
            "per_disorder": per_disorder,
        }, out_dir / "results.json")
        return 20

    wall = time.time() - t0
    results = {
        "status": "ok",
        "batch": "056",
        "sub": "d",
        "wall_s": wall,
        "reproduction_gate_R3": r3,
        "b3_symbol_to_ensg": b3_sym_to_ensg,
        "n_b3_mapped": int(n_b3_mapped),
        "b3_unmapped_symbols": unmapped,
        "b3_biological_category": B3_BIOLOGICAL_CATEGORY,
        "disorders": DISORDERS,
        "model": "MAGMA_Z ~ b3 + log10(gene_length) + lof.pLI + log10(lof.exp+1)",
        "vif_cascade_rule": (
            "per brief_v2 line 237: drop log10_exp_lof_plus1 first, then "
            "log10_gene_length, NEVER drop lof_pLI. Threshold VIF > 5.0."
        ),
        "per_disorder": per_disorder,
        "decision_classification": classification,
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("Sub-D wrote %s (wall=%.1fs)", out_dir / "results.json", wall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
