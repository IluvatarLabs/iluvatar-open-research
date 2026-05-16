#!/usr/bin/env python3
"""
03_axis_d_shet.py — Axis D s_het distribution shape.

GeneBayes s_het file is keyed by HGNC: ID. Map to symbol via HGNC complete set.
Per-disorder: n, median, IQR, quartiles (q25,q50,q75,q90,q95).
Pairwise Wilcoxon rank-sum over 6 pairs, Holm corrected.
Per-list Fisher OR for s_het >= 0.1 vs background.
Mapping-rate UNINTERPRETABLE gate: 80% per list.
"""
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

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


def fisher_ha(a, b, c, d):
    a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ha = (a2 * d2) / (b2 * c2)
    try:
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    except ValueError:
        p = np.nan
    return or_ha, float(p)


def main():
    log("=" * 70)
    log("batch_052_A/03_axis_d_shet.py starting")

    with open(INPUT / "assembled_inputs.json") as f:
        assembled = json.load(f)
    lists = {k: set(v["genes"]) for k, v in assembled["lists"].items()}

    # --- Load s_het and HGNC map ---
    shet = pd.read_csv(INPUT / "s_het_genebayes.tsv", sep="\t")
    log(f"s_het rows: {len(shet)}  cols: {shet.columns.tolist()}")

    hgnc = pd.read_csv(INPUT / "hgnc_complete_set.txt", sep="\t", low_memory=False,
                       usecols=["hgnc_id", "symbol", "ensembl_gene_id"])
    hgnc = hgnc.dropna(subset=["hgnc_id", "symbol"])
    log(f"HGNC map rows: {len(hgnc)}")

    # s_het has an 'hgnc' column already with "HGNC:xxxx" format
    shet_map = shet.merge(hgnc[["hgnc_id", "symbol"]], left_on="hgnc",
                          right_on="hgnc_id", how="left")
    n_with_sym = shet_map["symbol"].notna().sum()
    log(f"s_het rows with mapped symbol: {n_with_sym}/{len(shet_map)}")

    # Keep post_mean as s_het value
    shet_clean = shet_map.dropna(subset=["symbol", "post_mean"]).drop_duplicates(subset="symbol")
    log(f"s_het unique-symbol rows: {len(shet_clean)}")

    gene_to_shet = dict(zip(shet_clean["symbol"], shet_clean["post_mean"].astype(float)))

    # --- Per-list retention ---
    retention = {}
    for k, v in lists.items():
        n_map = sum(1 for g in v if g in gene_to_shet)
        rate = n_map / max(1, len(v))
        retention[k] = {"n_in_list": len(v), "n_mapped": n_map, "retention": rate}
        log(f"{k}: s_het mapped {n_map}/{len(v)} = {rate:.3f}")
        if rate < 0.80:
            log(f"UNINTERPRETABLE TRIGGER: {k} s_het mapping rate {rate:.3f} < 0.80")

    # --- Per-disorder distribution stats ---
    per_list_stats = {}
    values_by_list = {}
    for k, v in lists.items():
        vals = np.array([gene_to_shet[g] for g in v if g in gene_to_shet], dtype=float)
        values_by_list[k] = vals
        q = {
            "n": int(len(vals)),
            "median": float(np.median(vals)) if len(vals) else None,
            "mean": float(np.mean(vals)) if len(vals) else None,
            "q25": float(np.quantile(vals, 0.25)) if len(vals) else None,
            "q50": float(np.quantile(vals, 0.50)) if len(vals) else None,
            "q75": float(np.quantile(vals, 0.75)) if len(vals) else None,
            "q90": float(np.quantile(vals, 0.90)) if len(vals) else None,
            "q95": float(np.quantile(vals, 0.95)) if len(vals) else None,
            "iqr": (float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))
                    if len(vals) else None),
        }
        per_list_stats[k] = q
        log(f"{k} s_het: n={q['n']}  median={q['median']}  IQR={q['iqr']}  "
            f"q25={q['q25']}  q75={q['q75']}  q95={q['q95']}")

    # --- Pairwise Wilcoxon rank-sum, 6 pairs, Holm ---
    names = ["SCHEMA", "ASD_FDR10", "DDD_Kaplanis", "Heyne_DEE"]
    pairs = list(combinations(names, 2))
    wilc_rows = []
    raw_ps = []
    for a, b in pairs:
        va = values_by_list[a]
        vb = values_by_list[b]
        if len(va) < 2 or len(vb) < 2:
            p_raw = np.nan
            stat = np.nan
        else:
            try:
                res = mannwhitneyu(va, vb, alternative="two-sided")
                stat = float(res.statistic)
                p_raw = float(res.pvalue)
            except Exception:
                stat = np.nan
                p_raw = np.nan
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
        log(f"Wilcoxon {r['pair']}: p_raw={r['p_raw']:.4g}  holm_q={r['holm_q']:.4g}  "
            f"med_a={r['median_a']}  med_b={r['median_b']}")

    # --- Per-list OR for s_het >= 0.1 ---
    # Background = s_het genes NOT in the list
    or_rows = []
    all_shet_symbols = set(gene_to_shet.keys())
    for k, v in lists.items():
        list_in_shet = [g for g in v if g in gene_to_shet]
        n_list = len(list_in_shet)
        list_high = sum(1 for g in list_in_shet if gene_to_shet[g] >= 0.1)
        bg_symbols = all_shet_symbols - set(list_in_shet)
        bg_high = sum(1 for g in bg_symbols if gene_to_shet[g] >= 0.1)
        a = list_high
        b = n_list - list_high
        c = bg_high
        d = len(bg_symbols) - bg_high
        or_ha, p_f = fisher_ha(a, b, c, d)
        or_rows.append({"list": k, "n_mapped": n_list,
                        "k_shet_ge_0p1": list_high,
                        "bg_n": len(bg_symbols),
                        "bg_k_shet_ge_0p1": bg_high,
                        "OR_HA": or_ha, "fisher_p_greater": p_f})
        log(f"{k}: s_het>=0.1 {list_high}/{n_list}; OR_HA={or_ha:.3f} "
            f"fisher_p={p_f:.4g}")

    # --- Save TSV ---
    rows = []
    for k, s in per_list_stats.items():
        row = {"list": k}
        row.update(s)
        or_row = [r for r in or_rows if r["list"] == k][0]
        row["k_shet_ge_0p1"] = or_row["k_shet_ge_0p1"]
        row["OR_HA_shet_ge_0p1"] = or_row["OR_HA"]
        row["fisher_p_shet_ge_0p1"] = or_row["fisher_p_greater"]
        row["s_het_mapping_retention"] = retention[k]["retention"]
        rows.append(row)
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUTPUT / "axis_d_shet_stats.tsv", sep="\t", index=False)
    log(f"Wrote {OUTPUT / 'axis_d_shet_stats.tsv'}")

    # Wilcoxon TSV
    pd.DataFrame(wilc_rows).to_csv(OUTPUT / "axis_d_shet_wilcoxon.tsv", sep="\t", index=False)
    log(f"Wrote {OUTPUT / 'axis_d_shet_wilcoxon.tsv'}")

    results = {
        "axis": "D_shet_distribution",
        "s_het_source": "GeneBayes Zeng 2024 Nat Genet; post_mean",
        "mapping_retention": retention,
        "per_list_quartiles": per_list_stats,
        "pairwise_wilcoxon_holm6": wilc_rows,
        "shet_ge_0p1_OR": or_rows,
        "seed": SEED,
    }
    with open(OUTPUT / "axis_d_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {OUTPUT / 'axis_d_results.json'}")
    log("03_axis_d_shet.py DONE")


if __name__ == "__main__":
    main()
