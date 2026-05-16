#!/usr/bin/env python3
"""
02_axis_a_brainspan.py — Axis A BrainSpan peak developmental stage.

Per gene: peak = argmax of mean TPM within {prenatal, early_postnatal, adult}.
Universe: BrainSpan genes with TPM >= 1 in >= 1 stage.

Primary test: 3x3 chi-square on {ASD, DDD, Heyne} × stages (Fisher-Freeman-Halton
monte-carlo backup if any expected cell < 5).

Per-(disorder, stage) Fisher exact OR vs background.
Constraint-stratified sensitivity (background = pLI>=0.9 subset).
Holm correction over 12 primary (4 disorders × 3 stages).

SCHEMA per-stage descriptive-only.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency

ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH = ROOT / "experiments" / "batch_052_A"
INPUT = BATCH / "input"
OUTPUT = BATCH / "output"
LOGS = BATCH / "logs"
OUTPUT.mkdir(exist_ok=True, parents=True)

LOG = LOGS / "run.log"
SEED = 20260424
rng = np.random.default_rng(SEED)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def parse_age_weeks(age_str):
    s = str(age_str).strip().strip('"')
    if "pcw" in s:
        return float(s.replace(" pcw", ""))
    if "mos" in s:
        return 40 + float(s.replace(" mos", "")) * 4.33
    if "yrs" in s:
        return 40 + float(s.replace(" yrs", "")) * 52
    return None


def classify_stage(age_str):
    w = parse_age_weeks(age_str)
    if w is None:
        return None
    if w < 40:
        return "prenatal"
    if w < 40 + 5 * 52:
        return "early_postnatal"
    return "adult"


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
    """Return Haldane-Anscombe OR and one-sided greater Fisher p."""
    a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ha = (a2 * d2) / (b2 * c2)
    try:
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    except ValueError:
        p = np.nan
    return or_ha, float(p)


def run_chi_square_monte_carlo(obs, n_perm=20000, seed=SEED):
    """3x3 (or NxK) contingency table chi-square with monte-carlo p-value for
    small expected cells. Uses product-of-marginals under independence null.
    Returns (chi2_stat, mc_p_value, expected, min_expected).
    """
    obs = np.asarray(obs, dtype=float)
    try:
        chi2, p_asym, dof, expected = chi2_contingency(obs)
    except Exception:
        chi2, p_asym, dof, expected = np.nan, np.nan, np.nan, None

    min_exp = float(np.min(expected)) if expected is not None else float("nan")

    # Monte-carlo: simulate tables with fixed marginals using rng
    rng_local = np.random.default_rng(seed)
    row_sums = obs.sum(axis=1).astype(int)
    col_sums = obs.sum(axis=0).astype(float)
    N = int(obs.sum())
    probs = col_sums / col_sums.sum()

    ge = 0
    for _ in range(n_perm):
        sim = np.zeros_like(obs, dtype=int)
        for i, rs in enumerate(row_sums):
            sim[i, :] = rng_local.multinomial(rs, probs)
        try:
            c2, _, _, _ = chi2_contingency(sim)
        except Exception:
            continue
        if c2 >= chi2:
            ge += 1
    mc_p = (1 + ge) / (1 + n_perm)
    return float(chi2), float(p_asym), float(mc_p), expected.tolist() if expected is not None else None, min_exp


def main():
    log("=" * 70)
    log("batch_052_A/02_axis_a_brainspan.py starting")

    # --- Load inputs ---
    with open(INPUT / "assembled_inputs.json") as f:
        assembled = json.load(f)
    lists = {k: set(v["genes"]) for k, v in assembled["lists"].items()}
    log(f"lists loaded: {[f'{k}={len(v)}' for k, v in lists.items()]}")

    # BrainSpan metadata
    bs_dir = ROOT / "experiments/batch_041/output/brainspan_rnaseq_genes"
    col_meta = pd.read_csv(bs_dir / "columns_metadata.csv")
    row_meta = pd.read_csv(bs_dir / "rows_metadata.csv")
    col_meta["stage"] = col_meta["age"].apply(classify_stage)
    log(f"samples per stage: {col_meta['stage'].value_counts().to_dict()}")

    # Load expression matrix (genes x samples). First col is row number.
    log("Loading BrainSpan expression_matrix.csv ...")
    expr = pd.read_csv(bs_dir / "expression_matrix.csv", header=None)
    # first column = 1-based row_num; drop it
    expr = expr.iloc[:, 1:]
    expr.columns = range(expr.shape[1])
    log(f"expr shape: {expr.shape}  rows_meta: {len(row_meta)}  cols_meta: {len(col_meta)}")

    stages = col_meta["stage"].values
    sel = {s: np.where(stages == s)[0] for s in ["prenatal", "early_postnatal", "adult"]}

    arr = expr.values.astype(np.float32)
    means = {}
    for s, idx in sel.items():
        means[s] = arr[:, idx].mean(axis=1)
    stage_df = pd.DataFrame(means)
    stage_df["gene_symbol"] = row_meta["gene_symbol"].values

    # --- Universe filter: TPM>=1 in at least one stage ---
    max_tpm = stage_df[["prenatal", "early_postnatal", "adult"]].max(axis=1)
    universe_mask = (max_tpm >= 1.0) & stage_df["gene_symbol"].notna()
    universe = stage_df[universe_mask].copy()
    universe = universe.drop_duplicates(subset="gene_symbol", keep="first")
    log(f"BrainSpan universe (TPM>=1 in >=1 stage, with symbol): n={len(universe):,} (of {len(stage_df):,})")

    # --- Assign peak stage per gene ---
    vals = universe[["prenatal", "early_postnatal", "adult"]].values
    peak_idx = np.argmax(vals, axis=1)
    labels = np.array(["prenatal", "early_postnatal", "adult"])
    universe["peak_stage"] = labels[peak_idx]
    stage_dist = universe["peak_stage"].value_counts().to_dict()
    log(f"BrainSpan universe peak-stage distribution: {stage_dist}")

    # --- List intersection with universe ---
    uset = set(universe["gene_symbol"])
    retention = {}
    for k, v in lists.items():
        inter = v & uset
        retention[k] = {
            "n_in_list": len(v),
            "n_in_brainspan_universe": len(inter),
            "retention": len(inter) / max(1, len(v)),
            "genes_in_universe": sorted(inter),
        }
        log(f"{k}: {len(inter)}/{len(v)} retained in BrainSpan universe "
            f"(retention={retention[k]['retention']:.3f})")

    # UNINTERPRETABLE trigger: >=20% drop
    for k, r in retention.items():
        dropped = 1.0 - r["retention"]
        if dropped >= 0.20:
            log(f"UNINTERPRETABLE TRIGGER: {k} drops {dropped:.1%} in BrainSpan filter "
                f"(>=20% threshold). Reporting descriptive-only for {k}.")

    # --- Per-disorder peak-stage count matrix ---
    # Full matrix including SCHEMA (descriptive); primary 3x3 excludes SCHEMA
    stage_ord = ["prenatal", "early_postnatal", "adult"]
    counts = {}
    for k, v in lists.items():
        inter = v & uset
        sub = universe[universe["gene_symbol"].isin(inter)]
        row = [int((sub["peak_stage"] == s).sum()) for s in stage_ord]
        counts[k] = row
        log(f"{k} peak-stage counts: {dict(zip(stage_ord, row))}")

    # --- Primary 3x3 chi-square on ASD, DDD, Heyne ---
    primary_names = ["ASD_FDR10", "DDD_Kaplanis", "Heyne_DEE"]
    primary_obs = np.array([counts[k] for k in primary_names])
    log(f"Primary 3x3 observed table ({primary_names} × {stage_ord}):\n{primary_obs}")
    chi2, p_asym, mc_p, expected, min_exp = run_chi_square_monte_carlo(primary_obs, n_perm=20000, seed=SEED)
    log(f"3x3 chi2={chi2:.4f}  p_asym={p_asym:.4g}  mc_p={mc_p:.4g}  min_expected={min_exp:.2f}")

    # --- Per-cell Fisher OR (unstratified background) ---
    bg_all_in_universe = sorted(uset - set().union(*lists.values()))
    bg_df = universe[universe["gene_symbol"].isin(bg_all_in_universe)]
    bg_stage_counts = {s: int((bg_df["peak_stage"] == s).sum()) for s in stage_ord}
    bg_total = len(bg_df)
    log(f"Background (BrainSpan universe minus any list): n={bg_total}, per-stage={bg_stage_counts}")

    cell_rows = []
    pvals_primary = []
    primary_keys = []
    for k in ["SCHEMA", "ASD_FDR10", "DDD_Kaplanis", "Heyne_DEE"]:
        inter = lists[k] & uset
        sub = universe[universe["gene_symbol"].isin(inter)]
        n_list = len(sub)
        descriptive_only = (k == "SCHEMA")
        for s in stage_ord:
            k_list_in_stage = int((sub["peak_stage"] == s).sum())
            # Background = universe minus ALL lists
            K_bg_stage = bg_stage_counts[s]
            N_bg = bg_total
            a = k_list_in_stage
            b = n_list - k_list_in_stage
            c = K_bg_stage
            d = N_bg - K_bg_stage
            or_ha, p_fisher = fisher_ha(a, b, c, d)
            rec = {
                "list": k, "stage": s,
                "a_list_in_stage": a, "b_list_not_stage": b,
                "c_bg_in_stage": c, "d_bg_not_stage": d,
                "n_list_in_universe": n_list,
                "OR_HA": or_ha, "fisher_p_greater": p_fisher,
                "descriptive_only": descriptive_only,
            }
            cell_rows.append(rec)
            if not descriptive_only:
                pvals_primary.append(p_fisher)
                primary_keys.append(f"{k}::{s}")

    q_primary = holm(pvals_primary)
    for key, q in zip(primary_keys, q_primary):
        for rec in cell_rows:
            if f"{rec['list']}::{rec['stage']}" == key:
                rec["holm_q_12cells"] = float(q)
                break
    # SCHEMA cells get NaN holm
    for rec in cell_rows:
        rec.setdefault("holm_q_12cells", None)

    # --- Constraint-stratified sensitivity: bg = pLI>=0.9 subset ---
    log("Constraint-stratified sensitivity: loading gnomAD pLI...")
    gc = pd.read_csv(ROOT / "data/item_15/gnomad.v4.1.constraint_metrics.tsv",
                     sep="\t", low_memory=False)
    mask = (gc["canonical"].astype(str).str.lower() == "true") & (
        gc["mane_select"].astype(str).str.lower() == "true")
    gc_mm = gc[mask].drop_duplicates(subset="gene")
    high_pli = gc_mm[gc_mm["lof.pLI"] >= 0.9]["gene"].dropna().astype(str).tolist()
    log(f"pLI>=0.9 canonical+MANE genes: {len(high_pli):,}")
    pli_set = set(high_pli)
    bg_pli = bg_df[bg_df["gene_symbol"].isin(pli_set)]
    bg_pli_stage_counts = {s: int((bg_pli["peak_stage"] == s).sum()) for s in stage_ord}
    bg_pli_total = len(bg_pli)
    log(f"Constraint-stratified bg (pLI>=0.9 ∩ BrainSpan - lists): n={bg_pli_total}, "
        f"per-stage={bg_pli_stage_counts}")

    for rec in cell_rows:
        # cells are about list genes; stratified just changes denominator
        K_bg_stage = bg_pli_stage_counts[rec["stage"]]
        N_bg = bg_pli_total
        a = rec["a_list_in_stage"]
        b = rec["b_list_not_stage"]
        c = K_bg_stage
        d = N_bg - K_bg_stage
        or_ha_s, p_s = fisher_ha(a, b, c, d)
        rec["OR_HA_stratified_pLI09_bg"] = or_ha_s
        rec["fisher_p_greater_stratified"] = p_s

    # --- Save matrix TSV ---
    mat_rows = []
    for k in ["SCHEMA", "ASD_FDR10", "DDD_Kaplanis", "Heyne_DEE"]:
        for s in stage_ord:
            rec = [r for r in cell_rows if r["list"] == k and r["stage"] == s][0]
            mat_rows.append(rec)
    mat_df = pd.DataFrame(mat_rows)
    mat_df.to_csv(OUTPUT / "axis_a_peak_stage_matrix.tsv", sep="\t", index=False)
    log(f"Wrote {OUTPUT / 'axis_a_peak_stage_matrix.tsv'}")

    # --- Descriptive: peak_stage histograms per list (including SCHEMA) ---
    hist = {}
    for k in ["SCHEMA", "ASD_FDR10", "DDD_Kaplanis", "Heyne_DEE"]:
        inter = lists[k] & uset
        sub = universe[universe["gene_symbol"].isin(inter)]
        hist[k] = {s: int((sub["peak_stage"] == s).sum()) for s in stage_ord}

    # --- Dump ---
    results = {
        "axis": "A_brainspan_peak_stage",
        "universe_size": int(len(universe)),
        "universe_peak_dist": stage_dist,
        "list_retention": retention,
        "list_peak_histogram": hist,
        "primary_3x3_chi2": {
            "lists": primary_names,
            "stages": stage_ord,
            "observed": primary_obs.tolist(),
            "expected": expected,
            "min_expected_cell": min_exp,
            "chi2_stat": chi2,
            "p_asymptotic": p_asym,
            "p_monte_carlo_20000": mc_p,
        },
        "per_cell": cell_rows,
        "constraint_stratified_bg_pli09": {
            "bg_n": bg_pli_total,
            "bg_per_stage": bg_pli_stage_counts,
        },
        "seed": SEED,
    }
    with open(OUTPUT / "axis_a_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {OUTPUT / 'axis_a_results.json'}")
    log("02_axis_a_brainspan.py DONE")


if __name__ == "__main__":
    main()
