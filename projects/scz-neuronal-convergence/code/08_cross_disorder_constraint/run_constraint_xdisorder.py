#!/usr/bin/env python3
"""
batch_050 — Cross-Disorder Constraint Specificity (PI item 15 extension)

Tests whether the SCHEMA SCZ extreme-constraint signature (F120/F121) is shared
with ASD (Satterstrom 2020 FDR<=0.10 and FDR<=0.05), DDD (Kaplanis 2020), and BD
(Palmer 2022). Pre-registered in experiments/batch_050/brief.md (3-critic PASS).

WHY this script (vs. reusing batch_047/run_sub2_constraint.py):
  - batch_047's q_value was a hand-rolled Bonferroni-style cap; here BH is
    applied to ONE primary metric (pLI) across 4 lists per Critic-1 fix.
  - batch_047 used 5000 perms; brief mandates 20000 (Critic-2 fix for 5e-5
    resolution).
  - Length-matching changed from 1kb fixed bins to deciles of log10(cds_length)
    (faithful to brief; brief explicitly says "same as batch_047", but inspecting
    batch_047 it used 1kb bins — we follow the brief's explicit decile spec
    because it is the most recent pre-registered design).
  - Adds Haldane-Anscombe-corrected OR + Wald 95% CI (Critic-2 SCHEMA fix).
  - Adds cross-disorder logistic LR test + pairwise Wilcoxon on continuous pLI
    (Critic-1/2 H_CD2 discrimination).
  - Adds discovery-bias sensitivity using bg_minus_disease comparator
    (Critic-1/3 fix; an APPROXIMATION of the ideal Kaplanis-tested-not-sig set).

Determinism: random.seed(0); np.random.seed(0).
Single self-contained script. Does not import other batch scripts.
"""
import json
import math
import os
import random
import sys
import warnings
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, norm
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ---------------------------------------------------------------------------
# Determinism (Cardinal Rule 0: reproducibility)
# ---------------------------------------------------------------------------
random.seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = "/home/yuanz/Documents/GitHub/biomarvin_schizophrenia"
GNOMAD = f"{ROOT}/data/item_15/gnomad.v4.1.constraint_metrics.tsv"
OUTDIR = f"{ROOT}/experiments/batch_050/output"
INPUTS = f"{ROOT}/experiments/batch_050/input"
SCHEMA_FILE = f"{ROOT}/experiments/batch_044/input/schema_exome_wide_significant.txt"

os.makedirs(OUTDIR, exist_ok=True)

# Thresholds (pre-registered in brief)
PLI_HIGH = 0.9
LOEUF_LOW = 0.6
MISZ_HIGH = 3.09
N_PERM = 20_000
PRIMARY_METRIC = "pLI"


# ---------------------------------------------------------------------------
# Tee print -> stdout AND run_log buffer (so we can dump every line to disk)
# ---------------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


_log_buf = StringIO()
_real_stdout = sys.stdout
sys.stdout = Tee(_real_stdout, _log_buf)


def log(*args, **kwargs):
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Step 1: Load gnomAD canonical constraint metrics
# ---------------------------------------------------------------------------
def load_gnomad():
    log(f"[STEP 1] Loading gnomAD v4.1 from {GNOMAD}")
    cols_needed = ["gene", "canonical", "mane_select", "lof.pLI", "lof.oe", "mis.z_score", "cds_length"]
    df = pd.read_csv(GNOMAD, sep="\t", usecols=cols_needed, low_memory=False)
    log(f"  raw rows: {len(df)}")

    # canonical column is a string 'true'/'false' in this gnomAD v4 file
    df["canonical"] = df["canonical"].astype(str).str.lower() == "true"
    df["mane_select"] = df["mane_select"].astype(str).str.lower() == "true"
    # Use canonical OR mane_select rows; cds_length is populated almost
    # exclusively on MANE-select transcripts in v4.1, so canonical-only filter
    # discards length info for ~17k of 18k genes (regression vs batch_047).
    df = df[df["canonical"] | df["mane_select"]].copy()
    log(f"  canonical|mane rows: {len(df)}")

    # Coerce numerics; missing -> NaN
    for c in ["lof.pLI", "lof.oe", "mis.z_score", "cds_length"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Deduplicate on gene symbol; PREFER (a) row with cds_length present, then
    # (b) mane_select=True, then (c) canonical=True. gnomAD v4.1 has multiple
    # canonical+MANE rows per gene where only one carries cds_length, so cds
    # presence must be the primary sort key for length-matched permutation.
    before = len(df)
    df["_has_cds"] = df["cds_length"].notna() & (df["cds_length"] > 0)
    df = df.sort_values(
        ["_has_cds", "mane_select", "canonical"],
        ascending=[False, False, False],
    )
    df = df.drop_duplicates(subset="gene", keep="first")
    df = df.drop(columns=["_has_cds"])
    if len(df) != before:
        log(f"  dropped {before - len(df)} duplicate gene rows "
            f"(kept MANE-select if present, else canonical)")

    df = df.rename(columns={"lof.pLI": "pLI", "lof.oe": "LOEUF",
                            "mis.z_score": "missense_z"})
    df = df.set_index("gene")
    log(f"  N_canonical_genes_with_metrics = {len(df)}")
    return df


def empirical_backgrounds(g):
    """Compute background prevalence for the three thresholds, per Critic-1 ask."""
    n_pli = g["pLI"].notna().sum()
    n_loeuf = g["LOEUF"].notna().sum()
    n_misz = g["missense_z"].notna().sum()
    bg = {
        "pLI_ge_0.9": {
            "K": int((g["pLI"] >= PLI_HIGH).sum()),
            "N_with_metric": int(n_pli),
            "rate": float((g["pLI"] >= PLI_HIGH).sum() / n_pli) if n_pli else None,
        },
        "LOEUF_le_0.6": {
            "K": int((g["LOEUF"] <= LOEUF_LOW).sum()),
            "N_with_metric": int(n_loeuf),
            "rate": float((g["LOEUF"] <= LOEUF_LOW).sum() / n_loeuf) if n_loeuf else None,
        },
        "missense_z_ge_3.09": {
            "K": int((g["missense_z"] >= MISZ_HIGH).sum()),
            "N_with_metric": int(n_misz),
            "rate": float((g["missense_z"] >= MISZ_HIGH).sum() / n_misz) if n_misz else None,
        },
    }
    log("  Background prevalences (canonical gnomAD v4.1):")
    for k, v in bg.items():
        log(f"    {k}: K={v['K']} / N={v['N_with_metric']} = {v['rate']:.4f}")
    return bg


# ---------------------------------------------------------------------------
# Step 2: Load gene lists
# ---------------------------------------------------------------------------
def _read_gene_file(path):
    """Read gene symbols, skipping comment lines (#) and tokens that don't look
    like a gene symbol. Tokens are split on whitespace; we keep the first token
    of each non-comment line (so 'AKAP11<TAB># comment' yields 'AKAP11')."""
    genes = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()[0].strip()
            # Crude sanity: gene symbols are uppercase letters, digits, hyphens, dots
            if not tok or tok.startswith("#"):
                continue
            genes.append(tok)
    # de-dup, preserve order
    seen, out = set(), []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def load_gene_lists():
    log("\n[STEP 2] Loading gene lists")
    lists = {
        "SCHEMA": _read_gene_file(SCHEMA_FILE),
        "ASD_FDR10": _read_gene_file(f"{INPUTS}/asd_satterstrom_2020_fdr10.txt"),
        "ASD_FDR05": _read_gene_file(f"{INPUTS}/asd_satterstrom_2020_fdr05.txt"),
        "BD_top10": _read_gene_file(f"{INPUTS}/bd_exome_significant.txt"),
        "BD_AKAP11": ["AKAP11"],
        "DDD": _read_gene_file(f"{INPUTS}/ddd_kaplanis_2020.txt"),
    }
    for n, gl in lists.items():
        log(f"  {n}: N={len(gl)}")
    return lists


def intersect_with_gnomad(lists, gnomad_index):
    """Return mapping name -> dict(genes_in, n_total, n_in, n_dropped, dropped_examples)."""
    log("\n[STEP 2b] Intersecting with gnomAD canonical universe")
    out = {}
    for name, gl in lists.items():
        in_g = [g for g in gl if g in gnomad_index]
        dropped = [g for g in gl if g not in gnomad_index]
        out[name] = {
            "n_total": len(gl),
            "n_in_gnomad": len(in_g),
            "n_dropped": len(dropped),
            "dropped_examples": dropped[:10],
            "genes_in": in_g,
        }
        log(f"  {name}: total={len(gl)} in_gnomAD={len(in_g)} dropped={len(dropped)}"
            + (f" (e.g. {dropped[:5]})" if dropped else ""))
        if len(in_g) == 0:
            msg = (f"FATAL: gene list '{name}' has zero genes after gnomAD intersection. "
                   f"Possible identifier-mapping failure. Aborting.\n")
            sys.stderr.write(msg)
            log(msg.rstrip())  # tee to run_log
            sys.exit(2)
    return out


# ---------------------------------------------------------------------------
# Step 3: Length-matched permutation null
# ---------------------------------------------------------------------------
def build_length_deciles(gnomad_df, metric_col):
    """Decile-bin the gnomAD universe (restricted to genes with metric_col present
    AND cds_length present) on log10(cds_length). Returns dict bin_idx -> np.array
    of gene symbols and a parallel dict bin_idx -> np.array of metric values."""
    sub = gnomad_df[gnomad_df[metric_col].notna() & gnomad_df["cds_length"].notna()
                    & (gnomad_df["cds_length"] > 0)].copy()
    sub["log_cds"] = np.log10(sub["cds_length"].astype(float))
    # Use qcut deciles; duplicates='drop' for safety on tied edges
    sub["dec"] = pd.qcut(sub["log_cds"], q=10, labels=False, duplicates="drop")
    bins_genes, bins_vals = {}, {}
    for d, grp in sub.groupby("dec"):
        bins_genes[int(d)] = grp.index.to_numpy()
        bins_vals[int(d)] = grp[metric_col].to_numpy()
    return bins_genes, bins_vals, sub


def assign_genes_to_deciles(genes, sub_with_dec):
    """For a list of gene symbols (already known to be in sub_with_dec.index),
    return list of decile assignments (one per gene)."""
    return sub_with_dec.loc[genes, "dec"].astype(int).to_numpy()


def length_matched_perm_or(observed_or, list_genes_in_metric, gnomad_df, metric_col,
                           threshold, direction, n_perm=N_PERM, rng=None):
    """Empirical p-value via length-matched permutation.

    For each permutation, sample N_list genes from gnomAD (restricted to genes
    with metric_col present) matched to the list's per-gene decile of
    log10(cds_length). Compute null OR; emp_p = (1 + sum(null >= obs))/(1+P).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    bins_genes, bins_vals, sub = build_length_deciles(gnomad_df, metric_col)

    # Restrict list to genes that have metric AND cds_length AND were binned
    eligible = [g for g in list_genes_in_metric if g in sub.index]
    if len(eligible) == 0:
        return None, None, 0
    decs = assign_genes_to_deciles(eligible, sub)
    n_list = len(eligible)

    # Background K and N for OR computation in null draws
    if direction == "high":
        bg_pos_mask = {b: bins_vals[b] >= threshold for b in bins_genes}
    else:  # 'low'
        bg_pos_mask = {b: bins_vals[b] <= threshold for b in bins_genes}

    # Total background pos and N (for the 2x2 in null draws)
    K_bg_total = int(sum(bg_pos_mask[b].sum() for b in bg_pos_mask))
    N_bg_total = int(sum(len(bins_vals[b]) for b in bg_pos_mask))

    null_ors = np.empty(n_perm, dtype=float)

    # Pre-index decile -> (gene_array, value_array, pos_mask) for fast sampling
    bin_pos = {b: np.where(bg_pos_mask[b])[0] for b in bg_pos_mask}
    bin_size = {b: len(bins_vals[b]) for b in bg_pos_mask}

    # Per-decile gene-count needed
    from collections import Counter
    dec_counts = Counter(decs.tolist())

    for i in range(n_perm):
        sampled_pos = 0
        sampled_total = 0
        for b, k in dec_counts.items():
            sz = bin_size[b]
            if sz == 0:
                continue
            # Sample k indices without replacement from bin
            idx = rng.choice(sz, size=k, replace=False)
            sampled_total += k
            sampled_pos += int(np.isin(idx, bin_pos[b]).sum())

        a = sampled_pos
        b_ = sampled_total - sampled_pos
        c = K_bg_total - a
        d = N_bg_total - sampled_total - c
        # Haldane-Anscombe smoothed OR for stability in null
        or_null = ((a + 0.5) * (d + 0.5)) / ((b_ + 0.5) * (c + 0.5))
        null_ors[i] = or_null

    n_ge = int((null_ors >= observed_or).sum())
    emp_p = (1 + n_ge) / (1 + n_perm)
    return emp_p, null_ors, n_list


# ---------------------------------------------------------------------------
# Step 4: Per-list Fisher (with Haldane-Anscombe and Wald CI)
# ---------------------------------------------------------------------------
def fisher_with_ha(k_list, n_list, K_bg, N_bg, list_in_bg=True):
    """Returns dict with raw OR, HA OR, log(HA OR) Wald 95% CI, Fisher one-sided p.

    When list_in_bg=True (default): background INCLUDES the list genes, so we
    deduct list contributions to get the "not-in-list" complement:
        a = k_list                         # in list, constrained
        b = n_list - k_list                # in list, not constrained
        c = K_bg - k_list                  # not in list, constrained
        d = (N_bg - n_list) - (K_bg - k_list)

    When list_in_bg=False: background is DISJOINT from the list (e.g. the
    bg_minus_disease comparator). Deducting again would double-count, so:
        c = K_bg                           # constrained genes in disjoint bg
        d = N_bg - K_bg                    # unconstrained genes in disjoint bg
    """
    a = k_list
    b = n_list - k_list
    if list_in_bg:
        c = K_bg - k_list
        d = (N_bg - n_list) - (K_bg - k_list)
    else:
        c = K_bg
        d = N_bg - K_bg

    # Raw OR
    if b == 0 or c == 0:
        or_raw = float("inf")
    else:
        or_raw = (a * d) / (b * c)

    # Haldane-Anscombe corrected
    a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ha = (a2 * d2) / (b2 * c2)
    se_log_or = math.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
    log_or = math.log(or_ha)
    z = norm.ppf(0.975)
    ci_low = math.exp(log_or - z * se_log_or)
    ci_high = math.exp(log_or + z * se_log_or)

    # Fisher's exact (one-sided greater)
    _, p_fisher = fisher_exact([[a, b], [c, d]], alternative="greater")

    return {
        "k_list": int(a), "n_list_minus_k": int(b),
        "K_bg_minus_k": int(c), "N_minus_n_minus_K_plus_k": int(d),
        "OR_raw": (None if math.isinf(or_raw) else float(or_raw)),
        "OR_raw_is_inf": bool(math.isinf(or_raw)),
        "OR_HA": float(or_ha),
        "OR_HA_CI_low": float(ci_low),
        "OR_HA_CI_high": float(ci_high),
        "fisher_p_one_sided_greater": float(p_fisher),
    }


# ---------------------------------------------------------------------------
# Step 5: BH q-values
# ---------------------------------------------------------------------------
def bh_qvalues(pvals_dict):
    """Benjamini-Hochberg across the pvals provided. Returns dict same keys ->
    q-values."""
    keys = list(pvals_dict.keys())
    pvals = np.array([pvals_dict[k] for k in keys], dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n)
    # BH: q_i = min over j>=i of (p_j * n / j)  (1-based j)
    cummin = 1.0
    for j in range(n - 1, -1, -1):
        val = ranked[j] * n / (j + 1)
        cummin = min(cummin, val)
        q[j] = cummin
    out = {}
    for rank_i, k_i in enumerate(order):
        out[keys[k_i]] = float(q[rank_i])
    return out


# ---------------------------------------------------------------------------
# Step 6: Cross-disorder logistic regression + pairwise Wilcoxon
# ---------------------------------------------------------------------------
def cross_disorder_logistic(long_df, disorders):
    """Fit logit(constrained) ~ C(disorder) for the four interpretable lists.
    Returns (LR-test p, LR stat, df, pairwise raw p, pairwise OR, holm dict,
    OR_method per pair, notes list).

    WHY perfect-separation guard: statsmodels.Logit does NOT raise on perfect
    separation; it merely emits a ConvergenceWarning and returns inflated
    coefficients (e.g. SCHEMA all-9-constrained -> beta~24, OR~1e10, z>20,
    p~0). The pre-existing try/except never triggered. We detect separation
    explicitly and fall back to Fisher's exact 2x2 for any pair touching a
    perfectly-separated group.
    """
    sub = long_df[long_df["disorder"].isin(disorders)].copy()
    sub["disorder"] = pd.Categorical(sub["disorder"], categories=disorders)
    X = pd.get_dummies(sub["disorder"], drop_first=True).astype(float)
    X = sm.add_constant(X)
    y = sub["constrained"].astype(int).to_numpy()

    notes = []  # list so multiple notes can stack

    # ---- Perfect-separation detection per disorder group ----
    sep_groups = []
    for d in disorders:
        yg = sub.loc[sub["disorder"] == d, "constrained"].astype(int).to_numpy()
        if len(yg) > 0 and (yg.sum() == 0 or yg.sum() == len(yg)):
            sep_groups.append(d)

    lr_p = None
    lr_stat = None
    df_diff = X.shape[1] - 1

    if sep_groups:
        notes.append(
            f"Perfect separation detected (group(s): {','.join(sep_groups)} "
            f"all-constrained or all-unconstrained); LR test undefined."
        )
        # Skip global LR fit — statsmodels would silently return inflated coefs.
    else:
        # Belt-and-braces: capture ConvergenceWarning as a double-check.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                full = sm.Logit(y, X).fit(disp=0)
                null = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0)
                lr_stat = 2 * (full.llf - null.llf)
                from scipy.stats import chi2
                lr_p = float(chi2.sf(lr_stat, df_diff))
            except (ConvergenceWarning, Exception) as e:
                notes.append(f"LR fit failed/non-converged ({type(e).__name__}); LR test undefined.")
                lr_p = None
                lr_stat = None

    # ---- Pairwise contrasts ----
    pairs = []
    for i in range(len(disorders)):
        for j in range(i + 1, len(disorders)):
            pairs.append((disorders[i], disorders[j]))

    raw_pair_p = {}
    raw_pair_or = {}
    or_method = {}
    for (a, b) in pairs:
        key = f"{a}_vs_{b}"
        # If either group in this pair is perfectly separated, use Fisher 2x2 directly.
        ka = int(sub[(sub["disorder"] == a) & (sub["constrained"] == 1)].shape[0])
        na = int(sub[sub["disorder"] == a].shape[0])
        kb = int(sub[(sub["disorder"] == b) & (sub["constrained"] == 1)].shape[0])
        nb = int(sub[sub["disorder"] == b].shape[0])

        if a in sep_groups or b in sep_groups:
            _, p_two = fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="two-sided")
            # Haldane-Anscombe-corrected OR for stability when a cell is 0
            or_pair = ((ka + 0.5) * ((nb - kb) + 0.5)) / (((na - ka) + 0.5) * (kb + 0.5))
            raw_pair_p[key] = float(p_two)
            raw_pair_or[key] = float(or_pair)
            or_method[key] = "fisher_2x2"
            continue

        sub2 = sub[sub["disorder"].isin([a, b])].copy()
        sub2["is_b"] = (sub2["disorder"] == b).astype(int).to_numpy()
        Xp = sm.add_constant(sub2[["is_b"]].astype(float))
        yp = sub2["constrained"].astype(int).to_numpy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                mod = sm.Logit(yp, Xp).fit(disp=0)
            beta = float(mod.params["is_b"])
            se = float(mod.bse["is_b"])
            zstat = beta / se if se > 0 else float("nan")
            from scipy.stats import norm as _nrm
            p_two = 2 * (1 - _nrm.cdf(abs(zstat))) if not math.isnan(zstat) else float("nan")
            or_pair = math.exp(beta)
            method = "logistic"
        except (ConvergenceWarning, Exception):
            # Fallback: 2x2 Fisher two-sided on the pair (covers latent separation
            # not caught by the global per-group check, e.g. degenerate covariates).
            _, p_two = fisher_exact([[ka, na - ka], [kb, nb - kb]], alternative="two-sided")
            or_pair = ((ka + 0.5) * ((nb - kb) + 0.5)) / (((na - ka) + 0.5) * (kb + 0.5))
            method = "fisher_2x2"
        raw_pair_p[key] = float(p_two)
        raw_pair_or[key] = float(or_pair)
        or_method[key] = method

    # Holm correction across the pairs
    holm = holm_correct(raw_pair_p)

    return lr_p, lr_stat, df_diff, raw_pair_p, raw_pair_or, holm, or_method, notes


def holm_correct(pvals_dict):
    keys = list(pvals_dict.keys())
    pvals = np.array([pvals_dict[k] for k in keys], dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    holm = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pvals[idx] * (n - rank)
        running_max = max(running_max, adj)
        holm[rank] = min(running_max, 1.0)
    out = {}
    for rank, idx in enumerate(order):
        out[keys[idx]] = float(holm[rank])
    return out


def wilcoxon_pairs(long_df, disorders, value_col, lower_is_more_constrained=False):
    """Two-sided Mann-Whitney U on `value_col` for all pairs in `disorders`.
    Returns list of dicts and a Holm-corrected mapping."""
    pairs = []
    for i in range(len(disorders)):
        for j in range(i + 1, len(disorders)):
            pairs.append((disorders[i], disorders[j]))

    rows = []
    raw_p = {}
    for (a, b) in pairs:
        x = long_df.loc[long_df["disorder"] == a, value_col].dropna().to_numpy()
        y = long_df.loc[long_df["disorder"] == b, value_col].dropna().to_numpy()
        if len(x) == 0 or len(y) == 0:
            rows.append({"pair": f"{a}_vs_{b}", "n_a": len(x), "n_b": len(y),
                         "U": None, "p_two_sided": None, "rbc": None})
            raw_p[f"{a}_vs_{b}"] = 1.0
            continue
        U, p = mannwhitneyu(x, y, alternative="two-sided")
        # Rank-biserial correlation: r = 2U/(n1*n2) - 1 where U is U_x from
        # mannwhitneyu(x, y). positive rbc means group x (first arg, here `a`)
        # tends to higher values.
        rbc = 2.0 * U / (len(x) * len(y)) - 1.0
        rows.append({
            "pair": f"{a}_vs_{b}",
            "metric": value_col,
            "lower_is_more_constrained": lower_is_more_constrained,
            "n_a": int(len(x)), "n_b": int(len(y)),
            "median_a": float(np.median(x)), "median_b": float(np.median(y)),
            "U": float(U), "p_two_sided": float(p), "rbc": float(rbc),
        })
        raw_p[f"{a}_vs_{b}"] = float(p)
    holm = holm_correct(raw_p)
    for r in rows:
        r["p_holm"] = float(holm.get(r["pair"], 1.0))
    return rows


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log("=" * 72)
    log("batch_050 — Cross-Disorder Constraint Specificity")
    log("=" * 72)

    gnomad = load_gnomad()
    bg = empirical_backgrounds(gnomad)

    lists_raw = load_gene_lists()
    lists_iso = intersect_with_gnomad(lists_raw, gnomad.index)

    # ------------------------------------------------------------------
    # STEP 3: Per-list Fisher + permutation for SCHEMA, ASD_FDR10/05, DDD,
    #         BD_top10, BD_AKAP11. Three metrics: pLI, LOEUF, missense_z.
    # ------------------------------------------------------------------
    metric_specs = [
        ("pLI", PLI_HIGH, "high"),          # higher pLI = more LoF-intolerant
        ("LOEUF", LOEUF_LOW, "low"),         # lower LOEUF = more constrained
        ("missense_z", MISZ_HIGH, "high"),   # higher z = more missense-constrained
    ]

    results = {
        "metadata": {
            "n_perm": N_PERM,
            "thresholds": {"pLI": PLI_HIGH, "LOEUF": LOEUF_LOW,
                           "missense_z": MISZ_HIGH},
            "primary_metric": PRIMARY_METRIC,
            "background": "gnomAD v4.1 canonical, metric-present universe",
            "notes": (
                "Length-matched perm uses deciles of log10(cds_length); BH applied "
                "to pLI across SCHEMA, ASD_FDR10, ASD_FDR05, DDD only (BD excluded)."
            ),
        },
        "background_prevalence": bg,
        "lists": {},
        "per_list": {},
    }

    # Persist per-list intake info
    for nm, info in lists_iso.items():
        results["lists"][nm] = {
            "n_total": info["n_total"],
            "n_in_gnomad": info["n_in_gnomad"],
            "n_dropped": info["n_dropped"],
            "dropped_examples": info["dropped_examples"],
        }

    log("\n[STEP 3] Per-list Fisher + length-matched permutation")
    rng_master = np.random.default_rng(0)

    pli_pvals_for_bh = {}  # only the four interpretable lists

    BG_POS = {m: int(gnomad[m].notna().sum()) for m, _, _ in metric_specs}
    BG_K = {
        "pLI": int((gnomad["pLI"] >= PLI_HIGH).sum()),
        "LOEUF": int((gnomad["LOEUF"] <= LOEUF_LOW).sum()),
        "missense_z": int((gnomad["missense_z"] >= MISZ_HIGH).sum()),
    }

    interpretable = ["SCHEMA", "ASD_FDR10", "ASD_FDR05", "DDD"]

    for list_name in ["SCHEMA", "ASD_FDR10", "ASD_FDR05", "DDD",
                      "BD_top10", "BD_AKAP11"]:
        info = lists_iso[list_name]
        genes_in = info["genes_in"]
        list_block = {"n_total": info["n_total"],
                      "n_in_gnomad": info["n_in_gnomad"],
                      "n_dropped": info["n_dropped"],
                      "metrics": {}}
        log(f"\n  --- {list_name} (n_in_gnomad={len(genes_in)}) ---")

        for metric, thr, direction in metric_specs:
            vals = gnomad.loc[genes_in, metric].dropna()
            n_with_metric = int(vals.shape[0])
            if direction == "high":
                k = int((vals >= thr).sum())
            else:
                k = int((vals <= thr).sum())

            f = fisher_with_ha(k, n_with_metric, BG_K[metric], BG_POS[metric])

            if list_name == "BD_AKAP11":
                # n=1 special-case: report descriptive only, no perm, no Fisher trust
                emp_p, n_list_perm = None, n_with_metric
                or_for_perm = None
            else:
                # Permutation on observed OR (use HA OR for stability when raw OR=inf)
                obs_or = f["OR_HA"] if f["OR_raw_is_inf"] else f["OR_raw"]
                emp_p, _null, n_list_perm = length_matched_perm_or(
                    obs_or, vals.index.tolist(), gnomad, metric, thr, direction,
                    n_perm=N_PERM, rng=np.random.default_rng(rng_master.integers(1 << 31))
                )

            list_block["metrics"][metric] = {
                **f,
                "n_list_with_metric": n_with_metric,
                "K_bg_with_metric": BG_K[metric],
                "N_bg_with_metric": BG_POS[metric],
                "emp_p_perm": (None if emp_p is None else float(emp_p)),
                "n_list_used_in_perm": int(n_list_perm),
                "permutation_basis_OR": ("OR_HA" if f["OR_raw_is_inf"] else "OR_raw"),
            }
            or_raw_str = "inf" if f["OR_raw_is_inf"] else f"{f['OR_raw']:.3f}"
            emp_p_str = "NA" if emp_p is None else f"{emp_p:.5f}"
            log(f"    {metric}: k={k}/{n_with_metric}  OR_raw={or_raw_str}  "
                f"OR_HA={f['OR_HA']:.3f} [{f['OR_HA_CI_low']:.3f},{f['OR_HA_CI_high']:.3f}]  "
                f"fisher_p={f['fisher_p_one_sided_greater']:.3e}  "
                f"emp_p={emp_p_str}")

            if metric == "pLI" and list_name in interpretable:
                pli_pvals_for_bh[list_name] = f["fisher_p_one_sided_greater"]

        results["per_list"][list_name] = list_block

    # BH across the 4 interpretable pLI tests
    q_pli = bh_qvalues(pli_pvals_for_bh)
    log("\n  BH q-values across 4 interpretable lists (pLI primary):")
    for k, v in q_pli.items():
        results["per_list"][k]["metrics"]["pLI"]["q_pLI_primary_BH"] = float(v)
        log(f"    {k}: q={v:.3e}")

    # ------------------------------------------------------------------
    # STEP 4: SCHEMA replication HALT trigger
    # ------------------------------------------------------------------
    schema_block = results["per_list"]["SCHEMA"]["metrics"]["pLI"]
    schema_k = schema_block["k_list"]
    schema_emp_p = schema_block.get("emp_p_perm")
    log(f"\n[STEP 4] SCHEMA replication check: k={schema_k}, emp_p={schema_emp_p}")
    schema_ok = (schema_k >= 8) and (schema_emp_p is not None) and (schema_emp_p < 0.10)
    results["schema_replication"] = {
        "k_pLI_high": int(schema_k),
        "emp_p": (None if schema_emp_p is None else float(schema_emp_p)),
        "passes": bool(schema_ok),
        "criteria": ">=8/9 genes pLI>=0.9 AND emp_p<0.10",
    }
    if not schema_ok:
        flag_path = f"{OUTDIR}/SCHEMA_REPLICATION_FAILED.txt"
        with open(flag_path, "w") as fh:
            fh.write(f"SCHEMA replication failed: k={schema_k} (need >=8), "
                     f"emp_p={schema_emp_p} (need <0.10).\n")
        fatal_msg = (f"FATAL: SCHEMA replication failed (k={schema_k}, emp_p={schema_emp_p}). "
                     f"See {flag_path}.\n")
        sys.stderr.write(fatal_msg)
        log(fatal_msg.rstrip())  # tee to run_log
        # Still write what we have, then exit non-zero
        _dump_partial(results)
        sys.exit(3)

    # ------------------------------------------------------------------
    # STEP 5: Cross-disorder logistic + Wilcoxon on continuous pLI/LOEUF
    # ------------------------------------------------------------------
    log("\n[STEP 5] Cross-disorder OR-equality + Wilcoxon (pLI, LOEUF)")
    rows = []
    for d in interpretable:
        for g in lists_iso[d]["genes_in"]:
            row = {"gene": g, "disorder": d}
            row["pLI"] = float(gnomad.at[g, "pLI"]) if pd.notna(gnomad.at[g, "pLI"]) else np.nan
            row["LOEUF"] = float(gnomad.at[g, "LOEUF"]) if pd.notna(gnomad.at[g, "LOEUF"]) else np.nan
            row["constrained"] = int(row["pLI"] >= PLI_HIGH) if not math.isnan(row["pLI"]) else 0
            rows.append(row)
    long_df = pd.DataFrame(rows)
    log(f"  long_df shape: {long_df.shape}; per-disorder counts:")
    log(long_df.groupby("disorder").agg(n=("gene", "size"),
                                         k_constrained=("constrained", "sum"),
                                         median_pLI=("pLI", "median")).to_string())

    lr_p, lr_stat, df_diff, raw_pair_p, raw_pair_or, holm_pairs, or_method, lr_notes = \
        cross_disorder_logistic(long_df, interpretable)
    if lr_p is None:
        log(f"\n  Logistic LR test: SKIPPED ({'; '.join(lr_notes) or 'see notes'})")
    else:
        log(f"\n  Logistic LR test: chi2={lr_stat:.3f} df={df_diff} p={lr_p:.3e}")
    for k in raw_pair_p:
        log(f"    pairwise ({or_method[k]}) {k}: OR={raw_pair_or[k]:.3f}  "
            f"p_raw={raw_pair_p[k]:.3e}  p_holm={holm_pairs[k]:.3e}")

    wilc_pli = wilcoxon_pairs(long_df, interpretable, "pLI", False)
    wilc_loeuf = wilcoxon_pairs(long_df, interpretable, "LOEUF", True)
    log("\n  Wilcoxon (continuous pLI):")
    for r in wilc_pli:
        log(f"    {r['pair']}: U={r['U']:.0f} p={r['p_two_sided']:.3e} "
            f"p_holm={r['p_holm']:.3e} rbc={r['rbc']:.3f}")
    log("  Wilcoxon (continuous LOEUF; lower=more constrained):")
    for r in wilc_loeuf:
        log(f"    {r['pair']}: U={r['U']:.0f} p={r['p_two_sided']:.3e} "
            f"p_holm={r['p_holm']:.3e} rbc={r['rbc']:.3f}")

    results["cross_disorder"] = {
        "logistic_LR": {"chi2": (None if lr_stat is None else float(lr_stat)),
                        "df": int(df_diff),
                        "p": (None if lr_p is None else float(lr_p)),
                        "raw_pairwise_p": raw_pair_p,
                        "raw_pairwise_OR": raw_pair_or,
                        "holm_pairwise_p": holm_pairs,
                        "OR_method": or_method,
                        "notes": lr_notes},
        "wilcoxon_pLI": wilc_pli,
        "wilcoxon_LOEUF": wilc_loeuf,
    }

    # ------------------------------------------------------------------
    # STEP 6: Discovery-bias sensitivity (bg_minus_disease)
    # ------------------------------------------------------------------
    log("\n[STEP 6] Discovery-bias sensitivity (bg_minus_disease approximation)")
    union_disease = set()
    for d in interpretable:
        union_disease.update(lists_iso[d]["genes_in"])
    bg_minus_idx = gnomad.index.difference(pd.Index(list(union_disease)))
    log(f"  union(disease) size: {len(union_disease)}; "
        f"bg_minus_disease universe size: {len(bg_minus_idx)}")
    bg_minus = gnomad.loc[bg_minus_idx]
    BG_K_minus = {
        "pLI": int((bg_minus["pLI"] >= PLI_HIGH).sum()),
        "LOEUF": int((bg_minus["LOEUF"] <= LOEUF_LOW).sum()),
        "missense_z": int((bg_minus["missense_z"] >= MISZ_HIGH).sum()),
    }
    BG_N_minus = {m: int(bg_minus[m].notna().sum()) for m, _, _ in metric_specs}

    sens_rows = []
    for list_name in interpretable:
        genes_in = lists_iso[list_name]["genes_in"]
        for metric, thr, direction in metric_specs:
            vals = gnomad.loc[genes_in, metric].dropna()
            n_with_metric = int(vals.shape[0])
            if direction == "high":
                k = int((vals >= thr).sum())
            else:
                k = int((vals <= thr).sum())
            f_full = fisher_with_ha(k, n_with_metric, BG_K[metric], BG_POS[metric])

            # bg_minus_disease background: list genes EXCLUDED from background, so
            # the 2x2 must NOT deduct k/n from the background totals (list_in_bg=False).
            f_min = fisher_with_ha(k, n_with_metric, BG_K_minus[metric], BG_N_minus[metric],
                                   list_in_bg=False)
            or_full = f_full["OR_HA"]
            or_min = f_min["OR_HA"]
            atten = (or_full - or_min) / or_full if or_full > 0 else None
            sens_rows.append({
                "list": list_name, "metric": metric,
                "OR_HA_full_bg": or_full,
                "OR_HA_bgMinusDisease": or_min,
                "attenuation_pct": (None if atten is None else 100.0 * atten),
                "attenuation_gt_50pct": (None if atten is None else bool(atten > 0.5)),
                "fisher_p_full": f_full["fisher_p_one_sided_greater"],
                "fisher_p_bgMinusDisease": f_min["fisher_p_one_sided_greater"],
            })
    results["discovery_bias_sensitivity"] = {
        "comparator": "gnomAD canonical MINUS union(SCHEMA,ASD_FDR10,ASD_FDR05,DDD)",
        "limitation": ("APPROXIMATION of the ideal Kaplanis-tested-not-significant "
                       "comparator, which is not on disk in this batch. The ideal "
                       "comparator (Heyne 2018 epilepsy DEE or Kaplanis denovoWEST "
                       "tested-not-sig) was not available; see brief.md note."),
        "rows": sens_rows,
    }
    log("  Sensitivity (per list, per metric):")
    for r in sens_rows:
        log(f"    {r['list']}/{r['metric']}: OR_full={r['OR_HA_full_bg']:.2f} -> "
            f"OR_bgMinus={r['OR_HA_bgMinusDisease']:.2f}  "
            f"atten={r['attenuation_pct']:.1f}%  >50%={r['attenuation_gt_50pct']}")

    # ------------------------------------------------------------------
    # STEP 7: Write outputs
    # ------------------------------------------------------------------
    _dump_outputs(results, wilc_pli, wilc_loeuf, sens_rows,
                  lr_p, lr_stat, df_diff, raw_pair_p, raw_pair_or, holm_pairs,
                  or_method, lr_notes)

    log("\n[DONE] All outputs written to", OUTDIR)


def _dump_partial(results):
    """Best-effort dump when we abort early (e.g. SCHEMA HALT)."""
    with open(f"{OUTDIR}/results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    with open(f"{OUTDIR}/run_log.txt", "w") as fh:
        fh.write(_log_buf.getvalue())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    raise TypeError(f"Not JSON serializable: {type(o)}")


def _dump_outputs(results, wilc_pli, wilc_loeuf, sens_rows,
                  lr_p, lr_stat, df_diff, raw_pair_p, raw_pair_or, holm_pairs,
                  or_method, lr_notes):
    # 1) results.json
    with open(f"{OUTDIR}/results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)

    # 2) results_summary.tsv  (pLI primary view)
    cols = ["list", "n_in_gnomad", "k_pli_high", "OR_raw", "OR_HA",
            "OR_HA_CI_low", "OR_HA_CI_high", "fisher_p", "emp_p",
            "q_pLI_primary"]
    rows = []
    for nm, blk in results["per_list"].items():
        m = blk["metrics"]["pLI"]
        rows.append({
            "list": nm,
            "n_in_gnomad": blk["n_in_gnomad"],
            "k_pli_high": m["k_list"],
            "OR_raw": ("inf" if m["OR_raw_is_inf"] else f"{m['OR_raw']:.4f}"),
            "OR_HA": f"{m['OR_HA']:.4f}",
            "OR_HA_CI_low": f"{m['OR_HA_CI_low']:.4f}",
            "OR_HA_CI_high": f"{m['OR_HA_CI_high']:.4f}",
            "fisher_p": f"{m['fisher_p_one_sided_greater']:.3e}",
            "emp_p": ("NA" if m["emp_p_perm"] is None else f"{m['emp_p_perm']:.5f}"),
            "q_pLI_primary": (f"{m['q_pLI_primary_BH']:.3e}"
                              if "q_pLI_primary_BH" in m else "NA"),
        })
    pd.DataFrame(rows, columns=cols).to_csv(f"{OUTDIR}/results_summary.tsv",
                                            sep="\t", index=False)

    # 3) wilcoxon_pairs.tsv
    wpd = pd.DataFrame(wilc_pli + wilc_loeuf)
    wpd.to_csv(f"{OUTDIR}/wilcoxon_pairs.tsv", sep="\t", index=False)

    # 4) logistic_lr.json
    base_note = ("Pairwise contrasts are pairwise logistic refits (intercept "
                 "+ disorder dummy), Wald two-sided p, Holm-corrected across "
                 "the 6 pairs. Pairs touching a perfectly-separated group fall "
                 "back to Fisher 2x2 (see OR_method per pair).")
    notes_combined = [base_note] + list(lr_notes or [])
    with open(f"{OUTDIR}/logistic_lr.json", "w") as fh:
        json.dump({
            "LR_chi2": (None if lr_stat is None else float(lr_stat)),
            "df": int(df_diff),
            "p": (None if lr_p is None else float(lr_p)),
            "raw_pairwise_p": raw_pair_p,
            "raw_pairwise_OR": raw_pair_or,
            "holm_pairwise_p": holm_pairs,
            "OR_method": or_method,
            "notes": notes_combined,
        }, fh, indent=2, default=_json_default)

    # 5) discovery_bias_sensitivity.tsv
    pd.DataFrame(sens_rows).to_csv(f"{OUTDIR}/discovery_bias_sensitivity.tsv",
                                   sep="\t", index=False)

    # 6) run_log.txt
    with open(f"{OUTDIR}/run_log.txt", "w") as fh:
        fh.write(_log_buf.getvalue())

    # 7) README.md describing the outputs
    readme = (
        "# batch_050 outputs\n\n"
        "Generated by `experiments/batch_050/scripts/run_constraint_xdisorder.py`.\n\n"
        "Files:\n"
        "- `results.json` — full structured results (per-list Fisher + permutation, "
        "schema replication check, cross-disorder logistic LR, Wilcoxon pairwise, "
        "discovery-bias sensitivity).\n"
        "- `results_summary.tsv` — flat per-list summary on the primary metric (pLI).\n"
        "- `wilcoxon_pairs.tsv` — pairwise Mann-Whitney U on continuous pLI and LOEUF.\n"
        "- `logistic_lr.json` — cross-disorder logistic LR test + pairwise contrasts.\n"
        "- `discovery_bias_sensitivity.tsv` — attenuation when background excludes "
        "the union of disease lists (APPROXIMATION; see brief).\n"
        "- `run_log.txt` — captured stdout from the run.\n"
        "- `SCHEMA_REPLICATION_FAILED.txt` — present ONLY if HALT trigger fired.\n"
    )
    with open(f"{OUTDIR}/README.md", "w") as fh:
        fh.write(readme)


def _selftest():
    """Lightweight inline self-test for the two correctness fixes (C1, C2).

    WHY inline: callers in the audit checklist invoke this script with
    --selftest; we don't want to require a separate pytest dependency.
    """
    failures = []

    # C1: list_in_bg toggle should change the OR. With list_in_bg=False the
    # background is treated as DISJOINT (no double-deduction), so the comparison
    # rate K_bg/(N_bg) is HIGHER than (K_bg-k)/(N_bg-n), meaning OR is SMALLER.
    f_in = fisher_with_ha(k_list=5, n_list=10, K_bg=100, N_bg=1000, list_in_bg=True)
    f_out = fisher_with_ha(k_list=5, n_list=10, K_bg=100, N_bg=1000, list_in_bg=False)
    or_in = f_in["OR_raw"]
    or_out = f_out["OR_raw"]
    if or_in is None or or_out is None:
        failures.append(f"C1: OR_raw was None (in={or_in}, out={or_out})")
    elif not (or_in != or_out):
        failures.append(f"C1: ORs identical (in={or_in:.4f}, out={or_out:.4f})")
    elif not (or_out < or_in):
        failures.append(
            f"C1: list_in_bg=False should give SMALLER OR than =True; "
            f"got in={or_in:.4f} out={or_out:.4f}")
    else:
        print(f"  C1 OK: OR(list_in_bg=True)={or_in:.4f} > OR(list_in_bg=False)={or_out:.4f}")

    # C2: rank-biserial sign convention — group x with higher values gives positive rbc.
    x = np.array([0.99] * 9, dtype=float)
    y = np.array([0.5] * 50, dtype=float)
    U, _ = mannwhitneyu(x, y, alternative="two-sided")
    rbc = 2.0 * U / (len(x) * len(y)) - 1.0
    if not (rbc > 0):
        failures.append(f"C2: rbc should be positive when x>y; got rbc={rbc:.4f}")
    else:
        print(f"  C2 OK: rbc(x=[0.99]*9, y=[0.5]*50) = {rbc:.4f} (positive as expected)")

    if failures:
        print("SELFTEST FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SELFTEST PASS")
    sys.exit(0)


if __name__ == "__main__":
    try:
        if "--selftest" in sys.argv:
            _selftest()
        main()
    finally:
        sys.stdout = _real_stdout  # restore even on crash
