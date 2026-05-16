#!/usr/bin/env python3
"""
05_axis_e_oe_ratio.py — Axis E per-gene log(oe_mis/oe_lof) ratio.

Per gene: `log_ratio = log((mis_obs+0.5)/mis_exp) - log((lof_obs+0.5)/lof_exp)`
  (brief Q3 wording).

Per disorder: n, median, IQR.
Pairwise Wilcoxon rank-sum over 6 pairs, Holm corrected.
Primary contrasts: SCHEMA-vs-DDD, SCHEMA-vs-ASD, SCHEMA-vs-Heyne.
"""
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

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


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(m)
    running = 0.0
    for i, pv in enumerate(ranked):
        running = max(running, pv * (m - i))
        adj[i] = min(running, 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def main():
    log("=" * 70)
    log("batch_052_A/05_axis_e_oe_ratio.py starting")

    with open(INPUT / "assembled_inputs.json") as f:
        assembled = json.load(f)
    lists = {k: set(v["genes"]) for k, v in assembled["lists"].items()}

    gc = pd.read_csv(ROOT / "data/item_15/gnomad.v4.1.constraint_metrics.tsv",
                     sep="\t", low_memory=False)
    mask = (gc["canonical"].astype(str).str.lower() == "true") & (
        gc["mane_select"].astype(str).str.lower() == "true")
    gc_mm = gc[mask].drop_duplicates(subset="gene")
    # Drop genes with zero expected values
    req_cols = ["mis.obs", "mis.exp", "lof_hc_lc.obs", "lof_hc_lc.exp"]
    sub = gc_mm.dropna(subset=req_cols).copy()
    sub = sub[(sub["mis.exp"] > 0) & (sub["lof_hc_lc.exp"] > 0)]
    log(f"gnomAD rows after exp>0 filter: {len(sub):,} / {len(gc_mm):,}")

    # Haldane-Anscombe +0.5 on obs counts (brief Q3)
    sub["log_ratio"] = (
        np.log((sub["mis.obs"].astype(float) + 0.5) / sub["mis.exp"].astype(float))
        - np.log((sub["lof_hc_lc.obs"].astype(float) + 0.5) / sub["lof_hc_lc.exp"].astype(float))
    )
    gene_to_lr = dict(zip(sub["gene"].astype(str), sub["log_ratio"].astype(float)))

    # Retention per list
    retention = {}
    for k, v in lists.items():
        n_map = sum(1 for g in v if g in gene_to_lr)
        retention[k] = {"n_in_list": len(v), "n_mapped": n_map,
                        "retention": n_map / max(1, len(v))}
        log(f"{k}: log_ratio mapped {n_map}/{len(v)} = {retention[k]['retention']:.3f}")

    # Per-list distribution stats
    per_list_stats = {}
    values_by_list = {}
    for k, v in lists.items():
        vals = np.array([gene_to_lr[g] for g in v if g in gene_to_lr], dtype=float)
        values_by_list[k] = vals
        q = {
            "n": int(len(vals)),
            "median": float(np.median(vals)) if len(vals) else None,
            "mean": float(np.mean(vals)) if len(vals) else None,
            "q25": float(np.quantile(vals, 0.25)) if len(vals) else None,
            "q75": float(np.quantile(vals, 0.75)) if len(vals) else None,
            "iqr": (float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))
                    if len(vals) else None),
        }
        per_list_stats[k] = q
        log(f"{k} log_ratio: n={q['n']} median={q['median']} IQR={q['iqr']}")

    # Pairwise Wilcoxon 6-pair Holm
    names = ["SCHEMA", "ASD_FDR10", "DDD_Kaplanis", "Heyne_DEE"]
    pairs = list(combinations(names, 2))
    wilc_rows = []
    raw_ps = []
    for a, b in pairs:
        va, vb = values_by_list[a], values_by_list[b]
        if len(va) < 2 or len(vb) < 2:
            p_raw = np.nan
            stat = np.nan
        else:
            res = mannwhitneyu(va, vb, alternative="two-sided")
            stat = float(res.statistic)
            p_raw = float(res.pvalue)
        wilc_rows.append({"pair": f"{a}__{b}", "n_a": int(len(va)),
                          "n_b": int(len(vb)),
                          "median_a": float(np.median(va)) if len(va) else None,
                          "median_b": float(np.median(vb)) if len(vb) else None,
                          "U_statistic": stat, "p_raw": p_raw})
        raw_ps.append(p_raw if not np.isnan(p_raw) else 1.0)
    q_adj = holm(raw_ps)
    for i, q in enumerate(q_adj):
        wilc_rows[i]["holm_q"] = float(q)
    for r in wilc_rows:
        log(f"Wilcoxon {r['pair']}: p_raw={r['p_raw']:.4g} holm_q={r['holm_q']:.4g} "
            f"med_a={r['median_a']} med_b={r['median_b']}")

    # Save
    pd.DataFrame([{"list": k, **s, "retention": retention[k]["retention"]}
                  for k, s in per_list_stats.items()]).to_csv(
        OUTPUT / "axis_e_oe_ratio_stats.tsv", sep="\t", index=False)
    pd.DataFrame(wilc_rows).to_csv(OUTPUT / "axis_e_oe_ratio_wilcoxon.tsv",
                                   sep="\t", index=False)

    results = {
        "axis": "E_oe_ratio",
        "formula": "log((mis_obs+0.5)/mis_exp) - log((lof_obs+0.5)/lof_exp)",
        "gnomad_rows_with_ratio": len(sub),
        "retention": retention,
        "per_list_stats": per_list_stats,
        "pairwise_wilcoxon_holm6": wilc_rows,
        "seed": SEED,
    }
    with open(OUTPUT / "axis_e_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {OUTPUT / 'axis_e_results.json'}")
    log("05_axis_e_oe_ratio.py DONE")


if __name__ == "__main__":
    main()
