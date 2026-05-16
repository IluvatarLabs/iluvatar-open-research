#!/usr/bin/env python3
"""batch_044 Flaw 2 fix: BH-FDR correct tau* z-scores across A_eur_sldsc_v2.

WHY: The S-LDSC tau* coefficients come from a ~95-category baselineLD+cell-type
joint model. Reporting cell-type tau p-values from z-scores without multiple
testing correction is a peer-reviewer-visible flaw because neuronal/oligo/
astro/OPC co-exist with ~90 other baselineLD categories in the same model.
Any significance claim on cell-type tau must be corrected across the full
coefficient family actually fit.

Input:  experiments/batch_044/output/A_eur_sldsc_v2.results
Output: experiments/batch_044/output/A_eur_tau_bh_corrected.json

Procedure:
  1. Parse .results (tab-separated) with pandas.
  2. For EVERY row, compute two-sided p = 2 * (1 - norm.cdf(|z|)).
  3. BH-FDR correct across all rows via statsmodels multipletests.
  4. Emit {tau, tau_se, z, raw_p_two_sided, bh_q} for the 4 cell-types plus
     a top-20 |z| table across the full model.

WHY two-sided: tau* can be positive (enriched) or negative (depleted). The
pre-registered directionality claim for cell types is "positive and non-zero";
reporting a two-sided p is conservative and matches the convention used in
Finucane 2018 / Jagadeesh 2022 for baselineLD cell-type annotations.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

BATCH_OUT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_044/output")
RESULTS = BATCH_OUT / "A_eur_sldsc_v2.results"
OUT = BATCH_OUT / "A_eur_tau_bh_corrected.json"

CELL_TYPE_CATS = {
    "neuronal": "neuronalL2_0",
    "oligodendrocyte": "oligodendrocyteL2_0",
    "astrocyte": "astrocyteL2_0",
    "OPC": "OPCL2_0",
}


def main() -> int:
    df = pd.read_csv(RESULTS, sep="\t")
    # Two-sided p from z via normal approx. WHY z -> 2*(1-Phi(|z|)): S-LDSC
    # emits Coefficient_z-score already normalized by jackknife SE; under the
    # null, z ~ N(0,1), so a two-sided test is 2*(1-Phi(|z|)).
    z = df["Coefficient_z-score"].to_numpy(dtype=float)
    # Use scipy.stats.norm.sf for numerical stability at large |z|:
    # sf(|z|) = 1 - cdf(|z|) without catastrophic cancellation.
    raw_p = 2.0 * norm.sf(np.abs(z))
    df["raw_p_two_sided"] = raw_p

    # BH-FDR across all rows. WHY across all rows (not cell-types only):
    # the question is whether a neuronal tau* z=1.73 is significant AFTER
    # accounting for the full 95+ coefficient family actually fit. Correcting
    # only across 4 cell types would be anti-conservative relative to the
    # exploratory surface S-LDSC actually scans.
    _, bh_q, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")
    df["bh_q"] = bh_q

    n_total = int(len(df))

    out = {
        "input_file": str(RESULTS),
        "n_total_coefficients": n_total,
        "method": "BH-FDR (statsmodels multipletests fdr_bh) across all tau* z-scores",
        "p_computation": "two-sided p = 2 * (1 - norm.cdf(|z|)) from Coefficient_z-score",
        "why_two_sided": "tau* can be positive (enrichment) or negative (depletion); pre-registered"
                         " neuronal directionality claim is positive but two-sided test is conservative.",
        "why_full_family": "Cell-type tau coefficients are estimated jointly with ~90 baselineLD"
                           " categories in one S-LDSC model; multiple testing correction must span"
                           " the full coefficient family actually fit.",
    }

    for label, cat in CELL_TYPE_CATS.items():
        row = df.loc[df["Category"] == cat]
        if row.empty:
            out[label] = {"error": f"Category {cat} not found in .results"}
            continue
        r = row.iloc[0]
        out[label] = {
            "category": cat,
            "tau": float(r["Coefficient"]),
            "tau_se": float(r["Coefficient_std_error"]),
            "z": float(r["Coefficient_z-score"]),
            "raw_p_two_sided": float(r["raw_p_two_sided"]),
            "bh_q": float(r["bh_q"]),
        }

    # Also report any row with |z|>2 and its q.
    signif_abs_z = df.loc[np.abs(df["Coefficient_z-score"]) > 2.0].copy()
    signif_abs_z = signif_abs_z.sort_values("bh_q").reset_index(drop=True)
    out["abs_z_gt_2"] = [
        {
            "category": str(r["Category"]),
            "tau": float(r["Coefficient"]),
            "tau_se": float(r["Coefficient_std_error"]),
            "z": float(r["Coefficient_z-score"]),
            "raw_p_two_sided": float(r["raw_p_two_sided"]),
            "bh_q": float(r["bh_q"]),
        }
        for _, r in signif_abs_z.iterrows()
    ]

    # Top-20 by |z| across the full model (sorted descending by |z|).
    top = df.assign(abs_z=np.abs(df["Coefficient_z-score"])).sort_values(
        "abs_z", ascending=False
    ).head(20)
    out["top_significant_by_q"] = [
        {
            "rank": i + 1,
            "category": str(r["Category"]),
            "tau": float(r["Coefficient"]),
            "tau_se": float(r["Coefficient_std_error"]),
            "z": float(r["Coefficient_z-score"]),
            "raw_p_two_sided": float(r["raw_p_two_sided"]),
            "bh_q": float(r["bh_q"]),
        }
        for i, (_, r) in enumerate(top.iterrows())
    ]

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    print(f"n_total_coefficients={n_total}")
    for label in CELL_TYPE_CATS:
        b = out[label]
        if "error" not in b:
            print(f"  {label:16s}  tau={b['tau']:+.3e}  z={b['z']:+.3f}  "
                  f"raw_p={b['raw_p_two_sided']:.4g}  bh_q={b['bh_q']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
