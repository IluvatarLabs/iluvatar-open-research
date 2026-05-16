#!/usr/bin/env python3
"""
07_summary.py — Collate per-axis results into one output/results.json.

Maps per-axis conclusions to the brief DECISION RULE branches. Interpretation
(ESTABLISHED/SUGGESTED/REFUTED/SPECULATIVE/INCONCLUSIVE) is OUT OF SCOPE for
the experimentalist — this step just summarizes raw numerics and the
pre-registered rule evaluation.
"""
import json
import time
from pathlib import Path

ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH = ROOT / "experiments" / "batch_052_A"
INPUT = BATCH / "input"
OUTPUT = BATCH / "output"
LOGS = BATCH / "logs"
LOG = LOGS / "run.log"
SEED = 20260424


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    log("=" * 70)
    log("batch_052_A/07_summary.py starting")

    assembled = load(INPUT / "assembled_inputs.json")
    axis_a = load(OUTPUT / "axis_a_results.json")
    axis_d = load(OUTPUT / "axis_d_results.json")
    axis_c = load(OUTPUT / "axis_c_results.json")
    axis_e = load(OUTPUT / "axis_e_results.json")
    heyne = load(OUTPUT / "heyne_pli_panel_results.json")

    # --- Extract key decision-rule cells ---
    # Axis A
    a_chi2_p_mc = axis_a["primary_3x3_chi2"]["p_monte_carlo_20000"]
    a_chi2_p_asym = axis_a["primary_3x3_chi2"]["p_asymptotic"]
    a_any_cell_sig = any(
        (c.get("holm_q_12cells") is not None and c["holm_q_12cells"] < 0.05)
        for c in axis_a["per_cell"]
    )

    # Axis D pairwise
    d_pairs = {r["pair"]: r for r in axis_d["pairwise_wilcoxon_holm6"]}
    ddd_schema_q = d_pairs["SCHEMA__DDD_Kaplanis"]["holm_q"]
    ddd_asd_q = d_pairs["ASD_FDR10__DDD_Kaplanis"]["holm_q"]
    d_any_sig = any(r["holm_q"] < 0.05 for r in axis_d["pairwise_wilcoxon_holm6"])
    d_sig_pairs = [r["pair"] for r in axis_d["pairwise_wilcoxon_holm6"] if r["holm_q"] < 0.05]

    # Axis C
    c_spread = axis_c["exclusive_spread"]
    c_incl_spread = axis_c["inclusive_spread"]

    # Axis E
    e_pairs = {r["pair"]: r for r in axis_e["pairwise_wilcoxon_holm6"]}
    e_schema_ddd_q = e_pairs["SCHEMA__DDD_Kaplanis"]["holm_q"]
    e_any_sig = any(r["holm_q"] < 0.05 for r in axis_e["pairwise_wilcoxon_holm6"])

    # Heyne
    heyne_row = [r for r in heyne["per_list"] if r["list"] == "Heyne_DEE"][0]
    heyne_or = heyne_row["OR_HA"]

    # --- Decision rule evaluation ---
    # Rule 1: >=1 axis has any pairwise Holm q<0.05 -> F127 promotes
    #   (primary axes: A, D, E; Axis C uses spread heuristic)
    # Note: Axis A "pairwise Holm q" is the per-cell Fisher Holm-12 here
    axis_sig = {
        "axis_A_any_cell_holm12_lt_0p05": bool(a_any_cell_sig),
        "axis_D_any_pair_holm6_lt_0p05": bool(d_any_sig),
        "axis_C_spread_ratio_max_min_ge_3": bool(c_spread["ratio_max_min"] is not None
                                                 and c_spread["ratio_max_min"] >= 3.0),
        "axis_E_any_pair_holm6_lt_0p05": bool(e_any_sig),
    }
    n_axes_sig = sum(axis_sig.values())

    # Rule 2 (Cassa 2017 saturation-robust): Axis D DDD-vs-SCHEMA Holm q < 0.05
    rule_cassa_fixed = bool(ddd_schema_q < 0.05)

    # Rule 3 (Heyne [15,60]): pre-registered range test
    heyne_in_range = bool(15.0 <= heyne_or <= 60.0)
    heyne_within_schema_class = bool(15.0 <= heyne_or <= 90.0)  # within observed 3-way class

    # --- Orthogonality (from assembled) ---
    orth_branch = assembled["heyne_preregistered_branch"]

    # --- Build final results.json ---
    results = {
        "batch": "batch_052_A",
        "completed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "preregistered": {
            "heyne_orthogonality_branch": orth_branch,
            "jaccard_heyne_ddd": assembled["jaccard"]["DDD_Kaplanis__Heyne_DEE"]["jaccard"],
            "jaccard_heyne_schema": assembled["jaccard"]["SCHEMA__Heyne_DEE"]["jaccard"],
            "jaccard_heyne_asd": assembled["jaccard"]["ASD_FDR10__Heyne_DEE"]["jaccard"],
        },
        "decision_cells": {
            "axis_A_primary_3x3_chi2_p_monte_carlo": a_chi2_p_mc,
            "axis_A_primary_3x3_chi2_p_asymptotic": a_chi2_p_asym,
            "axis_A_any_holm12_cell_sig": a_any_cell_sig,
            "axis_D_DDDvSCHEMA_holm_q": ddd_schema_q,
            "axis_D_DDDvASD_holm_q": ddd_asd_q,
            "axis_D_sig_pairs": d_sig_pairs,
            "axis_C_exclusive_spread_max_over_min": c_spread["ratio_max_min"],
            "axis_C_inclusive_spread_max_over_min": c_incl_spread["ratio_max_min"],
            "axis_E_SCHEMAvDDD_holm_q": e_schema_ddd_q,
            "axis_E_any_pair_sig": e_any_sig,
            "heyne_pLI_OR_HA": heyne_or,
            "heyne_pLI_emp_p_20000": heyne_row["emp_p_20000"],
            "heyne_fisher_p": heyne_row["fisher_p_one_sided_greater"],
            "heyne_OR_HA_CI": [heyne_row["OR_HA_CI_low"], heyne_row["OR_HA_CI_high"]],
        },
        "axis_significance_summary": axis_sig,
        "n_axes_with_significant_pair_or_cell": int(n_axes_sig),
        "pre_registered_decision_branches": {
            "F127_promotes_ge1_axis_holm_sig": int(n_axes_sig) >= 1,
            "axis_D_DDDvSCHEMA_saturation_robust_fix": rule_cassa_fixed,
            "heyne_in_preregistered_range_15_60": heyne_in_range,
            "heyne_within_observed_3way_class_15_90": heyne_within_schema_class,
        },
        "list_n_reconciliation": {
            k: {"n_raw": v["n"],
                "n_in_gnomad_canonical_mane": assembled["intersection_with_gnomad"][k]["n_in_gnomad_canonical_mane"]}
            for k, v in assembled["lists"].items()
        },
        "uninterpretable_triggers": {
            "axis_D_s_het_mapping_rate_any_lt_80": any(
                v["retention"] < 0.80 for v in axis_d["mapping_retention"].values()
            ),
            "axis_D_per_list_mapping": axis_d["mapping_retention"],
            "axis_A_brainspan_tpm_filter_any_ge_20pct_drop": any(
                (1.0 - v["retention"]) >= 0.20 for v in axis_a["list_retention"].values()
            ),
            "axis_A_per_list_retention": {k: v["retention"] for k, v in axis_a["list_retention"].items()},
            "axis_E_retention_any_lt_80": any(v["retention"] < 0.80 for v in axis_e["retention"].values()),
            "axis_E_per_list_retention": {k: v["retention"] for k, v in axis_e["retention"].items()},
        },
        "axis_a": axis_a,
        "axis_d": axis_d,
        "axis_c": axis_c,
        "axis_e": axis_e,
        "heyne_pli_panel": heyne,
    }

    out = OUTPUT / "results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {out}")

    # --- Short summary to log ---
    log("=" * 70)
    log("DECISION-CELL SUMMARY")
    log("=" * 70)
    log(f"Axis A 3x3 chi2 p_asym={a_chi2_p_asym:.4g}  p_mc={a_chi2_p_mc:.4g}  any_cell_Holm12<0.05={a_any_cell_sig}")
    log(f"Axis D DDDvSCHEMA Holm q={ddd_schema_q:.4g}  DDDvASD Holm q={ddd_asd_q:.4g}  sig_pairs={d_sig_pairs}")
    log(f"Axis C exclusive spread (max_OR/min_OR, n>=5 cells) = {c_spread['ratio_max_min']:.3f}")
    log(f"Axis E SCHEMAvDDD Holm q={e_schema_ddd_q:.4g}  any_pair_sig={e_any_sig}")
    log(f"Heyne pLI OR_HA={heyne_or:.3f}  in [15,60]={heyne_in_range}  in [15,90]={heyne_within_schema_class}")
    log(f"Heyne orthogonality branch = {orth_branch} (Jaccard_Heyne_DDD={assembled['jaccard']['DDD_Kaplanis__Heyne_DEE']['jaccard']:.4f})")
    log(f"n_axes_with_significant_pair_or_cell = {n_axes_sig}")
    log("07_summary.py DONE")


if __name__ == "__main__":
    main()
