#!/usr/bin/env python3
"""batch_044 Flaw 3 fix: 95% CIs for height specificity (B_height_sldsc.results).

WHY: The pre-registered REFUTED threshold for neuronal specificity to SCZ is
"height neuronal enrichment >= 30% of SCZ neuronal enrichment" (i.e. ~1.25
given SCZ neuronal ~1.83). Reporting a point estimate without a CI is a
peer-reviewer-visible flaw: the reader needs to see whether the CI straddles
the pre-registered threshold (which would make specificity unresolvable at
the current sample size, not refuted).

Also report the minimum detectable enrichment excess at alpha=0.05 given
SE=0.275: anything below 1 + 1.96*0.275 = 1.539 is statistically
undetectable under this design.

Input:  experiments/batch_044/output/B_height_sldsc.results
Output: experiments/batch_044/output/B_height_ci_analysis.json

CI formulas used (WHY):
  * Enrichment CI: point +/- 1.96 * Enrichment_std_error (Finucane 2015
    standard jackknife SE interpretation).
  * tau* CI:        Coefficient +/- 1.96 * Coefficient_std_error.
  * tau* p:         two-sided from z = Coefficient / Coefficient_std_error
                    via norm.sf (numerically stable).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm

BATCH_OUT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_044/output")
RESULTS = BATCH_OUT / "B_height_sldsc.results"
OUT = BATCH_OUT / "B_height_ci_analysis.json"

CELL_TYPE_CATS = {
    "neuronal": "neuronalL2_0",
    "oligodendrocyte": "oligodendrocyteL2_0",
    "astrocyte": "astrocyteL2_0",
    "OPC": "OPCL2_0",
}

# Pre-registered REFUTED threshold for height specificity: "enrichment >= 30%
# of SCZ neuronal" — with SCZ neuronal enrichment ~1.83 in the same pipeline,
# 30% excess over 1.0 gives ~1.25. See brief.md §DECISION RULE / Sub-B.
REFUTED_THRESHOLD_HEIGHT_ENRICH = 1.25


def main() -> int:
    df = pd.read_csv(RESULTS, sep="\t")
    z_alpha = 1.96  # two-sided alpha=0.05

    out = {
        "input_file": str(RESULTS),
        "method": "95% CI = point +/- 1.96 * jackknife SE; two-sided p from Z = Coef/SE",
        "refuted_threshold_height_enrichment": REFUTED_THRESHOLD_HEIGHT_ENRICH,
        "why_threshold": "30% excess over 1.0 equals 1.25; pre-registered in brief.md §Sub-B"
                         " as the threshold for 'height specificity REFUTED' (i.e. height enrichment"
                         " >=1.25 would mean neuronal signal is not SCZ-specific).",
    }

    for label, cat in CELL_TYPE_CATS.items():
        row = df.loc[df["Category"] == cat]
        if row.empty:
            out[label] = {"error": f"Category {cat} not found in .results"}
            continue
        r = row.iloc[0]
        enr = float(r["Enrichment"])
        enr_se = float(r["Enrichment_std_error"])
        enr_ci_low = enr - z_alpha * enr_se
        enr_ci_high = enr + z_alpha * enr_se
        ci_upper_exceeds_refuted = bool(enr_ci_high > REFUTED_THRESHOLD_HEIGHT_ENRICH)

        tau = float(r["Coefficient"])
        tau_se = float(r["Coefficient_std_error"])
        tau_z = float(r["Coefficient_z-score"])
        tau_p_two_sided = float(2.0 * norm.sf(abs(tau_z)))
        tau_ci_low = tau - z_alpha * tau_se
        tau_ci_high = tau + z_alpha * tau_se

        out[label] = {
            "category": cat,
            "enrichment": {
                "point": enr,
                "se": enr_se,
                "ci_low_95": enr_ci_low,
                "ci_high_95": enr_ci_high,
                "ci_upper_exceeds_refuted_threshold": ci_upper_exceeds_refuted,
                "p_from_sldsc_column": float(r["Enrichment_p"]) if pd.notna(r["Enrichment_p"]) else None,
            },
            "tau_star": {
                "point": tau,
                "se": tau_se,
                "z": tau_z,
                "raw_p_two_sided": tau_p_two_sided,
                "ci_low_95": tau_ci_low,
                "ci_high_95": tau_ci_high,
            },
        }

    # Minimum detectable enrichment excess at alpha=0.05 for neuronal given the
    # actual SE of the height neuronal enrichment in this fit.
    neu_row = df.loc[df["Category"] == CELL_TYPE_CATS["neuronal"]].iloc[0]
    neu_se = float(neu_row["Enrichment_std_error"])
    min_detectable_enr_one_sided = 1.0 + z_alpha * neu_se  # one-sided alpha=0.025 ~ 1.96 SE
    out["minimum_detectable_neuronal_enrichment"] = {
        "neuronal_enrichment_se": neu_se,
        "z_alpha_two_sided_0_05": z_alpha,
        "threshold_formula": "1 + 1.96 * SE",
        "min_detectable_enrichment": min_detectable_enr_one_sided,
        "interpretation": (
            f"Any enrichment point estimate below {min_detectable_enr_one_sided:.3f} is"
            " statistically undetectable from 1.0 at alpha=0.05 (two-sided) given the"
            f" jackknife SE={neu_se:.3f}. The pre-registered REFUTED threshold"
            f" ({REFUTED_THRESHOLD_HEIGHT_ENRICH}) lies BELOW this minimum-detectable"
            " floor, so 'height neuronal enrichment is not >=1.25' cannot be concluded"
            " from a non-significant test alone (Type II not Type I)."
        ),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT}")
    for label in CELL_TYPE_CATS:
        b = out[label]
        if "error" not in b:
            e = b["enrichment"]
            print(f"  {label:16s} enr={e['point']:+.3f} (95% CI {e['ci_low_95']:+.3f}, {e['ci_high_95']:+.3f})"
                  f" -> upper>{REFUTED_THRESHOLD_HEIGHT_ENRICH}? {e['ci_upper_exceeds_refuted_threshold']}")
    md = out["minimum_detectable_neuronal_enrichment"]
    print(f"  min detectable neuronal enrichment = {md['min_detectable_enrichment']:.3f} "
          f"(SE={md['neuronal_enrichment_se']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
