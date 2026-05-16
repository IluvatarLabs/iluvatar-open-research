#!/usr/bin/env python3
"""
E8 (R8): Haldane Correction for OR=infinity

From batch_048/output/A_edt1_decomposition.json, the glutamate receptor subset
(n=3) has: a=3, b=0, c=3039, d=14092 -> OR=inf

Apply Haldane-Anscombe correction (add 0.5 to each cell) and compute:
- Corrected OR with 95% CI via Woolf (log-OR) method
- Fisher exact CI for comparison
- Scan all other decomposition results for zero-cell issues

WHY: OR=infinity is not publishable. The Haldane-Anscombe correction is the
standard approach (Haldane 1956, Anscombe 1956) for 2x2 tables with zero cells.
A reviewer would require this correction or a note explaining the infinite OR.

Output: experiments/batch_069/output/e8_haldane_corrections.json
"""

from __future__ import annotations
import json, math, pathlib
import numpy as np
from scipy import stats

PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "batch_069" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DECOMP_JSON = PROJECT_ROOT / "experiments" / "batch_048" / "output" / "A_edt1_decomposition.json"


def haldane_or(a: int, b: int, c: int, d: int) -> dict:
    """
    Compute Haldane-Anscombe corrected OR and Woolf 95% CI.

    Haldane (1956) correction: add 0.5 to all four cells.
    Woolf (1955) method for log-OR CI:
      log(OR) = log(a'd'/b'c')
      SE(log(OR)) = sqrt(1/a' + 1/b' + 1/c' + 1/d')
      95% CI on log scale: log(OR) +/- 1.96 * SE
    """
    a_c = a + 0.5
    b_c = b + 0.5
    c_c = c + 0.5
    d_c = d + 0.5

    or_corrected = (a_c * d_c) / (b_c * c_c)
    log_or = math.log(or_corrected)
    se_log_or = math.sqrt(1/a_c + 1/b_c + 1/c_c + 1/d_c)

    ci_low_log = log_or - 1.96 * se_log_or
    ci_high_log = log_or + 1.96 * se_log_or
    ci_low = math.exp(ci_low_log)
    ci_high = math.exp(ci_high_log)

    return {
        "a_corrected": a_c,
        "b_corrected": b_c,
        "c_corrected": c_c,
        "d_corrected": d_c,
        "or_corrected": round(or_corrected, 4),
        "log_or": round(log_or, 4),
        "se_log_or": round(se_log_or, 4),
        "ci_low_woolf": round(ci_low, 4),
        "ci_high_woolf": round(ci_high, 4),
    }


def fisher_exact_ci(a: int, b: int, c: int, d: int) -> dict:
    """Compute Fisher exact test OR and CI using scipy."""
    table = [[a, b], [c, d]]
    # scipy fisher_exact returns (OR, p)
    or_fisher, p_fisher = stats.fisher_exact(table, alternative="two-sided")
    # For CI, use scipy odds_ratio
    try:
        result = stats.contingency.odds_ratio(table)
        ci = result.confidence_interval(0.95)
        ci_low = float(ci.low) if ci.low is not None else 0.0
        ci_high = float(ci.high) if ci.high is not None else float("inf")
    except Exception as e:
        ci_low, ci_high = float("nan"), float("nan")

    return {
        "or_fisher": float(or_fisher) if not math.isinf(or_fisher) else "Infinity",
        "p_fisher_two_sided": float(p_fisher),
        "ci_low_fisher": ci_low,
        "ci_high_fisher": ci_high if not math.isinf(ci_high) else "Infinity",
    }


def main():
    # Load decomposition
    with open(DECOMP_JSON) as f:
        decomp = json.load(f)

    corrections = []
    zero_cell_results = []

    print("=== Scanning batch_048 decomposition for zero cells ===\n")

    for r in decomp["results"]:
        a, b, c, d = r["a"], r["b"], r["c"], r["d"]
        has_zero = any(v == 0 for v in [a, b, c, d])
        has_inf = r.get("or") in [float("inf"), "Infinity", None] or (isinstance(r.get("or"), float) and math.isinf(r.get("or")))

        gene_list = r["gene_list"]
        metric = r["constraint_metric"]

        if has_zero or has_inf:
            print(f"ZERO CELL: {gene_list} / {metric}")
            print(f"  Table: a={a}, b={b}, c={c}, d={d}")
            print(f"  Original OR: {r['or']}")

            # Apply Haldane correction
            hc = haldane_or(a, b, c, d)
            fe = fisher_exact_ci(a, b, c, d)

            entry = {
                "gene_list": gene_list,
                "constraint_metric": metric,
                "original_table": {"a": a, "b": b, "c": c, "d": d},
                "original_or": str(r["or"]),
                "original_p": r["p"],
                "original_ci": [r.get("ci_low"), r.get("ci_high")],
                "zero_cells": [cell for cell, v in zip(["a", "b", "c", "d"], [a, b, c, d]) if v == 0],
                "haldane_correction": hc,
                "fisher_exact": fe,
                "n_in_list": r["n_in_list"],
            }
            corrections.append(entry)

            print(f"  Haldane OR:  {hc['or_corrected']:.2f} (95% CI: {hc['ci_low_woolf']:.2f} - {hc['ci_high_woolf']:.2f})")
            print(f"  Fisher CI:   ({fe['ci_low_fisher']:.2f}, {fe['ci_high_fisher']})")
            print(f"  Woolf SE(logOR): {hc['se_log_or']:.4f}")
            print()

            zero_cell_results.append(entry)
        else:
            # No zero cells, but still check for near-zero b
            if b <= 2 or a <= 2:
                print(f"SMALL CELL: {gene_list} / {metric} (a={a}, b={b})")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total tests scanned: {len(decomp['results'])}")
    print(f"Zero-cell results: {len(zero_cell_results)}")

    if zero_cell_results:
        print(f"\nPrimary correction (glutamate_receptor / pLI >= 0.9):")
        primary = next((c for c in corrections if c["gene_list"] == "glutamate_receptor" and "pLI" in c["constraint_metric"]), None)
        if primary:
            hc = primary["haldane_correction"]
            print(f"  Original: OR = Infinity")
            print(f"  Haldane-corrected: OR = {hc['or_corrected']:.2f}")
            print(f"  95% CI (Woolf): [{hc['ci_low_woolf']:.2f}, {hc['ci_high_woolf']:.2f}]")
            print(f"  All 3 glutamate receptor genes (GRIN2A, GRIA1, GRM3) have pLI >= 0.9")
            print(f"  This is a small-sample extreme result — the direction is genuine")
            print(f"  but the magnitude is poorly estimated (wide CI reflects n=3)")

    # Output
    result = {
        "experiment": "E8_Haldane_Correction",
        "batch": "batch_069",
        "status": "COMPLETED",
        "n_tests_scanned": len(decomp["results"]),
        "n_zero_cell_results": len(zero_cell_results),
        "corrections": corrections,
        "method_note": (
            "Haldane-Anscombe correction (Haldane 1956, Anscombe 1956): "
            "add 0.5 to all four cells of the 2x2 table. "
            "Woolf (1955) method for log-OR SE: "
            "SE = sqrt(1/a' + 1/b' + 1/c' + 1/d'). "
            "95% CI: exp(log(OR) +/- 1.96 * SE). "
            "Fisher exact CI computed via scipy.stats.contingency.odds_ratio."
        ),
    }

    out_path = OUTPUT_DIR / "e8_haldane_corrections.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
