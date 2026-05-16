#!/usr/bin/env python3
"""
06_heyne_pli_panel.py — Heyne-DEE pLI panel extension (F127 4-way).

Runs the F120/F127 pipeline:
  - Fisher + Haldane-Anscombe pLI>=0.9 OR + Wald CI + one-sided greater p
  - 20,000-perm length-matched (decile log10 cds_length) null
  - 4-way pairwise Fisher (pLI>=0.9 indicator) and pairwise Wilcoxon on
    continuous pLI, Holm corrected over 6 pairs.
"""
import json
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, norm

ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH = ROOT / "experiments" / "batch_052_A"
INPUT = BATCH / "input"
OUTPUT = BATCH / "output"
LOGS = BATCH / "logs"

LOG = LOGS / "run.log"
SEED = 20260424
N_PERM = 20000


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


def fisher_with_ha(k_list, n_list, K_bg, N_bg):
    """List_in_bg=True convention (F127 batch_050)."""
    a = k_list
    b = n_list - k_list
    c = K_bg - k_list
    d = (N_bg - n_list) - (K_bg - k_list)
    if b == 0 or c == 0:
        or_raw = float("inf")
    else:
        or_raw = (a * d) / (b * c)
    a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ha = (a2 * d2) / (b2 * c2)
    se = math.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
    z = norm.ppf(0.975)
    ci_low = math.exp(math.log(or_ha) - z * se)
    ci_high = math.exp(math.log(or_ha) + z * se)
    try:
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    except ValueError:
        p = np.nan
    return {
        "a": int(a), "b": int(b), "c": int(c), "d": int(d),
        "OR_raw": (None if math.isinf(or_raw) else float(or_raw)),
        "OR_raw_is_inf": bool(math.isinf(or_raw)),
        "OR_HA": float(or_ha),
        "OR_HA_CI_low": float(ci_low),
        "OR_HA_CI_high": float(ci_high),
        "fisher_p_one_sided_greater": float(p),
    }


def length_matched_null(list_in_bg, bg_decile, bg_pli_flag, observed_or,
                        n_perm=N_PERM, seed=SEED):
    """
    Permute by sampling n_list genes from bg preserving decile distribution.
    `list_in_bg` is a boolean array aligned with bg.
    Returns (emp_p, null_ors).
    """
    rng = np.random.default_rng(seed)
    deciles = np.asarray(bg_decile)
    pli = np.asarray(bg_pli_flag)
    in_list = np.asarray(list_in_bg)

    # Deciles histogram among list
    unique_dec = np.arange(int(deciles.min()), int(deciles.max()) + 1)
    list_dec_counts = {d: int(((deciles == d) & in_list).sum()) for d in unique_dec}

    # Indexes per decile for sampling
    dec_idx = {d: np.where((deciles == d) & (~in_list))[0] for d in unique_dec}

    N_bg = len(deciles)
    n_list = int(in_list.sum())
    K_bg = int(pli.sum())

    null_ors = np.empty(n_perm)
    for i in range(n_perm):
        sampled_idx = []
        for d, k in list_dec_counts.items():
            pool = dec_idx[d]
            if len(pool) == 0 or k == 0:
                continue
            k_eff = min(k, len(pool))
            sampled_idx.append(rng.choice(pool, size=k_eff, replace=False))
        if sampled_idx:
            idx = np.concatenate(sampled_idx)
        else:
            idx = np.array([], dtype=int)
        a = int(pli[idx].sum())
        sampled_total = len(idx)
        b = sampled_total - a
        c = K_bg - a
        d = N_bg - sampled_total - c
        a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        null_ors[i] = (a2 * d2) / (b2 * c2)
    n_ge = int((null_ors >= observed_or).sum())
    emp_p = (1 + n_ge) / (1 + n_perm)
    return emp_p, null_ors


def main():
    log("=" * 70)
    log("batch_052_A/06_heyne_pli_panel.py starting")

    with open(INPUT / "assembled_inputs.json") as f:
        assembled = json.load(f)
    lists = {k: set(v["genes"]) for k, v in assembled["lists"].items()}

    # gnomAD canonical+MANE; cds_length is patchy on the MANE row, so aggregate
    # max cds_length across transcripts per gene (matches batch_050 policy).
    gc = pd.read_csv(ROOT / "data/item_15/gnomad.v4.1.constraint_metrics.tsv",
                     sep="\t", low_memory=False)
    cds_by_gene = gc.groupby("gene")["cds_length"].max()
    mask = (gc["canonical"].astype(str).str.lower() == "true") & (
        gc["mane_select"].astype(str).str.lower() == "true")
    gc_mm = gc[mask].drop_duplicates(subset="gene").copy()
    gc_mm["cds_length"] = gc_mm["gene"].map(cds_by_gene)
    gc_mm = gc_mm.dropna(subset=["gene", "lof.pLI", "cds_length"])
    gc_mm = gc_mm[gc_mm["cds_length"] > 0]
    log(f"gnomAD MANE+canonical with pLI+cds_length (max across transcripts): {len(gc_mm):,}")

    gc_mm = gc_mm.reset_index(drop=True)
    gc_mm["pli_ge_09"] = (gc_mm["lof.pLI"].astype(float) >= 0.9).astype(int)
    # Decile of log10(cds_length)
    gc_mm["log_len"] = np.log10(gc_mm["cds_length"].astype(float).clip(lower=1.0))
    gc_mm["decile"] = pd.qcut(gc_mm["log_len"], q=10, labels=False,
                              duplicates="drop")

    N_bg = len(gc_mm)
    K_bg = int(gc_mm["pli_ge_09"].sum())
    log(f"N_bg={N_bg}  K_bg(pLI>=0.9)={K_bg}")

    # --- Per-list Fisher + permutation ---
    per_list = []
    for k, v in lists.items():
        in_list = gc_mm["gene"].astype(str).isin(v).values
        n_list = int(in_list.sum())
        k_list = int((gc_mm.loc[in_list, "pli_ge_09"] == 1).sum())
        f = fisher_with_ha(k_list, n_list, K_bg, N_bg)
        obs_or = f["OR_HA"] if f["OR_raw_is_inf"] else (f["OR_raw"] if f["OR_raw"] else f["OR_HA"])
        # Use HA for permutation basis (matches batch_050)
        perm_basis = f["OR_HA"] if f["OR_raw_is_inf"] else f["OR_raw"]
        emp_p, null_ors = length_matched_null(in_list,
                                              gc_mm["decile"].values,
                                              gc_mm["pli_ge_09"].values,
                                              perm_basis, n_perm=N_PERM,
                                              seed=SEED + hash(k) % 1000)
        f["n_list_in_bg"] = n_list
        f["k_list_pli_high"] = k_list
        f["emp_p_20000"] = float(emp_p)
        f["permutation_basis_OR"] = "OR_HA" if f["OR_raw_is_inf"] else "OR_raw"
        f["null_or_median"] = float(np.median(null_ors))
        f["null_or_p99"] = float(np.quantile(null_ors, 0.99))
        f["list"] = k
        per_list.append(f)
        log(f"{k}: n_in_bg={n_list} k_pli_high={k_list} OR_raw={f['OR_raw']} "
            f"OR_HA={f['OR_HA']:.3f}[{f['OR_HA_CI_low']:.3f},{f['OR_HA_CI_high']:.3f}] "
            f"fisher_p={f['fisher_p_one_sided_greater']:.4g} emp_p={f['emp_p_20000']:.4g}")

    # --- 4-way pairwise Fisher on pLI>=0.9 indicator (Holm over 6) ---
    names = ["SCHEMA", "ASD_FDR10", "DDD_Kaplanis", "Heyne_DEE"]
    pairs = list(combinations(names, 2))
    pairwise_fisher = []
    raw_ps = []
    for a, b in pairs:
        a_mask = gc_mm["gene"].astype(str).isin(lists[a]).values
        b_mask = gc_mm["gene"].astype(str).isin(lists[b]).values
        na, nb = int(a_mask.sum()), int(b_mask.sum())
        ka = int((gc_mm.loc[a_mask, "pli_ge_09"] == 1).sum())
        kb = int((gc_mm.loc[b_mask, "pli_ge_09"] == 1).sum())
        # 2x2: list-a rows × pLI>=0.9 columns, vs list-b
        table = [[ka, na - ka], [kb, nb - kb]]
        try:
            stat, p_raw = fisher_exact(table, alternative="two-sided")
        except ValueError:
            stat, p_raw = np.nan, np.nan
        or_ha = ((ka + 0.5) * ((nb - kb) + 0.5)) / (((na - ka) + 0.5) * (kb + 0.5))
        pairwise_fisher.append({"pair": f"{a}__{b}", "n_a": na, "n_b": nb,
                                "k_a": ka, "k_b": kb,
                                "OR_a_vs_b_HA": float(or_ha),
                                "fisher_p_raw": float(p_raw) if p_raw is not np.nan else None})
        raw_ps.append(float(p_raw) if not np.isnan(p_raw) else 1.0)
    qs = holm(raw_ps)
    for i, q in enumerate(qs):
        pairwise_fisher[i]["holm_q"] = float(q)
    for r in pairwise_fisher:
        log(f"PairFisher {r['pair']}: OR={r['OR_a_vs_b_HA']:.2f} p_raw={r['fisher_p_raw']:.4g} holm_q={r['holm_q']:.4g}")

    # --- 4-way pairwise Wilcoxon on continuous pLI (Holm over 6) ---
    pairwise_wilc = []
    raw_ps_w = []
    pli_map = dict(zip(gc_mm["gene"].astype(str), gc_mm["lof.pLI"].astype(float)))
    for a, b in pairs:
        va = np.array([pli_map[g] for g in lists[a] if g in pli_map])
        vb = np.array([pli_map[g] for g in lists[b] if g in pli_map])
        if len(va) < 2 or len(vb) < 2:
            stat, p_raw = np.nan, np.nan
        else:
            res = mannwhitneyu(va, vb, alternative="two-sided")
            stat = float(res.statistic)
            p_raw = float(res.pvalue)
        pairwise_wilc.append({"pair": f"{a}__{b}", "n_a": int(len(va)),
                              "n_b": int(len(vb)),
                              "median_a": float(np.median(va)) if len(va) else None,
                              "median_b": float(np.median(vb)) if len(vb) else None,
                              "U_statistic": stat, "p_raw": p_raw})
        raw_ps_w.append(float(p_raw) if not np.isnan(p_raw) else 1.0)
    qs_w = holm(raw_ps_w)
    for i, q in enumerate(qs_w):
        pairwise_wilc[i]["holm_q"] = float(q)
    for r in pairwise_wilc:
        log(f"PairWilc {r['pair']}: p_raw={r['p_raw']:.4g} holm_q={r['holm_q']:.4g} "
            f"med_a={r['median_a']} med_b={r['median_b']}")

    # --- Save ---
    pl_df = pd.DataFrame(per_list)
    pl_df.to_csv(OUTPUT / "heyne_pli_panel.tsv", sep="\t", index=False)
    log(f"Wrote {OUTPUT / 'heyne_pli_panel.tsv'}")
    pd.DataFrame(pairwise_fisher).to_csv(OUTPUT / "heyne_pli_panel_pairwise_fisher.tsv",
                                         sep="\t", index=False)
    pd.DataFrame(pairwise_wilc).to_csv(OUTPUT / "heyne_pli_panel_pairwise_wilcoxon.tsv",
                                       sep="\t", index=False)

    results = {
        "axis": "heyne_pli_panel_F127_4way",
        "n_permutations": N_PERM,
        "length_matching": "decile of log10(cds_length)",
        "N_bg": N_bg, "K_bg_pli_ge_09": K_bg,
        "per_list": per_list,
        "pairwise_fisher_holm6": pairwise_fisher,
        "pairwise_wilcoxon_holm6": pairwise_wilc,
        "seed": SEED,
    }
    with open(OUTPUT / "heyne_pli_panel_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {OUTPUT / 'heyne_pli_panel_results.json'}")
    log("06_heyne_pli_panel.py DONE")


if __name__ == "__main__":
    main()
