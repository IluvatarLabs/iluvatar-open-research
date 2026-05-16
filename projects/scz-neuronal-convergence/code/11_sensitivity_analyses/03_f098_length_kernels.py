"""03_f098_length_kernels.py — F098 gene-length-conditioned neuronal enrichment across 5 kernels.

Anchor: F098 adj_OR=6.94 (batch_040 d45 linear ±10% kernel).

Kernels (preflight):
  linear ±10% ratio (d45 default).
  log ±0.1 in log10(length).
  quantile-5bin / 10bin / 20bin (equal-size bins of bg by length).

For each kernel:
  k_obs = |SCZ ∩ neuronal|.
  Null: 10,000 permutations. Each perm draws a length-matched pseudo-SCZ gene set (same size)
  from bg by sampling each gene from the same match-bucket as the corresponding SCZ gene.
  OR_obs computed as Fisher OR of (SCZ∩neuronal, SCZ\\neuronal; bg\\SCZ∩neuronal, bg\\SCZ\\neuronal).
  adj_OR = OR_obs / median(OR_null). emp_p = frac null OR >= OR_obs.

Bg = GenCode v44 protein-coding genes with length.
Seed 20260424.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import (OUTPUT_DIR, LOGS_DIR, RNG_SEED, N_PERM,
                    load_scz_genes, load_neuronal_markers, load_gencode,
                    fisher_or_ci, log_event, load_scz_df)

def compute_or(scz_set: set[str], markers: set[str], bg: set[str]) -> float:
    a = len(scz_set & markers)
    b = len(scz_set - markers)
    c = len((bg - scz_set) & markers)
    d = len(bg - scz_set - markers)
    from math import log
    # Haldane if any zero
    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (a * d) / (b * c) if b * c > 0 else float("inf")

def build_buckets_linear(bg_genes: list[str], bg_lens: np.ndarray, scz_lens: np.ndarray,
                         tol: float = 0.1) -> list[np.ndarray]:
    """For each SCZ gene length L, return indices of bg genes within L(1±tol)."""
    buckets = []
    for L in scz_lens:
        lo, hi = L * (1 - tol), L * (1 + tol)
        idx = np.where((bg_lens >= lo) & (bg_lens <= hi))[0]
        buckets.append(idx)
    return buckets

def build_buckets_log(bg_genes, bg_lens, scz_lens, tol_log: float = 0.1):
    log_bg = np.log10(bg_lens)
    buckets = []
    for L in scz_lens:
        lg = np.log10(L)
        idx = np.where((log_bg >= lg - tol_log) & (log_bg <= lg + tol_log))[0]
        buckets.append(idx)
    return buckets

def build_buckets_quantile(bg_genes, bg_lens, scz_lens, n_bins: int):
    # Assign each bg gene to a quantile bin, and each scz gene to the same
    edges = np.quantile(bg_lens, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf; edges[-1] = np.inf
    bg_bin = np.digitize(bg_lens, edges, right=False) - 1
    bg_bin = np.clip(bg_bin, 0, n_bins - 1)
    by_bin = [np.where(bg_bin == b)[0] for b in range(n_bins)]
    buckets = []
    for L in scz_lens:
        b = int(np.clip(np.digitize(L, edges, right=False) - 1, 0, n_bins - 1))
        buckets.append(by_bin[b])
    return buckets

def run_kernel(name: str, buckets, bg_genes_arr, scz_in_bg: list[str],
               markers: set[str], rng: np.random.Generator) -> dict:
    """Vectorized null: per-perm sample one bg index per SCZ gene from its match-bucket.
    OR_obs uses set form; OR_null is computed on the integer hit-count k_null (hits among sampled)
    assuming each sample has size |scz_in_bg|. This is biased by duplicates (sample might include
    the same bg gene twice) but so does the observed side — we compensate by using same-size
    contingency with expected duplicates.

    To stay consistent with OR formula, compute contingency with the distinct sampled set size.
    """
    N_bg = len(bg_genes_arr)
    M = np.isin(bg_genes_arr, list(markers))
    n_M = int(M.sum())
    # Observed
    scz_idx = np.array([int(np.where(bg_genes_arr == g)[0][0]) for g in scz_in_bg if g in set(bg_genes_arr.tolist())])
    n_L = len(scz_idx)
    k_obs = int(M[scz_idx].sum())
    a = k_obs; b = n_L - k_obs; c = n_M - k_obs; d = N_bg - n_L - n_M + k_obs
    if 0 in (a, b, c, d):
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        aa, bb, cc, dd = a, b, c, d
    or_obs = (aa * dd) / (bb * cc) if bb * cc else float("inf")
    # Permutation: for each SCZ slot sample one index from its bucket
    # Pre-materialize bucket index arrays
    bucket_arrs = [np.asarray(b, dtype=np.int64) if len(b) else np.arange(N_bg, dtype=np.int64)
                   for b in buckets]
    empties = sum(1 for b in buckets if len(b) == 0)
    n_slots = len(bucket_arrs)
    null_or = np.empty(N_PERM, dtype=float)
    # Vectorize: precompute per-slot random integers (N_PERM x n_slots) uniform [0, len_bucket)
    lens = np.array([len(a) for a in bucket_arrs], dtype=np.int64)
    # For each (perm, slot), draw integer uniformly in [0, lens[slot]).
    # rng.integers supports broadcasting via high=lens.
    rand_u = rng.random((N_PERM, n_slots))
    idx_in_bucket = (rand_u * lens).astype(np.int64)  # broadcast: lens has shape (n_slots,)
    # Gather picks[i, j] = bucket_arrs[j][idx_in_bucket[i, j]]
    # Build a rectangular index matrix: picks has shape (N_PERM, n_slots)
    picks = np.empty((N_PERM, n_slots), dtype=np.int64)
    for j, arr in enumerate(bucket_arrs):
        picks[:, j] = arr[idx_in_bucket[:, j]]
    # Marker lookup
    hits_matrix = M[picks]  # (N_PERM, n_slots) bool
    # For set-based n_L, we need unique count per row — use Python loop but with numpy unique
    for i in range(N_PERM):
        uniq = np.unique(picks[i])
        n_Ln = len(uniq)
        k_null = int(M[uniq].sum())
        an, bn, cn, dn = k_null, n_Ln - k_null, n_M - k_null, N_bg - n_Ln - n_M + k_null
        if 0 in (an, bn, cn, dn):
            an, bn, cn, dn = an + 0.5, bn + 0.5, cn + 0.5, dn + 0.5
        null_or[i] = (an * dn) / (bn * cn) if bn * cn else float("inf")
    finite = null_or[np.isfinite(null_or)]
    med_null = float(np.median(finite)) if len(finite) else float("nan")
    adj_or = or_obs / med_null if med_null and med_null > 0 else float("nan")
    emp_p = float((null_or >= or_obs).sum() + 1) / (N_PERM + 1)
    return {
        "spec": name,
        "n_scz_in_bg": n_L,
        "a_obs": a,
        "OR_obs": or_obs,
        "OR_null_median": med_null,
        "adj_OR": adj_or,
        "emp_p": emp_p,
        "empty_buckets": empties,
    }

def main():
    log = LOGS_DIR / "run.log"; t0 = time.time()
    scz_df = load_scz_df()
    markers = load_neuronal_markers()
    gencode = load_gencode()
    # Bg = gencode PC genes with known length.
    bg_items = [(g, v["length"]) for g, v in gencode.items() if v.get("length", 0) > 0]
    bg_genes_arr = np.array([g for g, _ in bg_items])
    bg_lens = np.array([l for _, l in bg_items], dtype=float)
    bg_set = set(bg_genes_arr.tolist())
    # SCZ in bg, with length (prefer gencode length for bg consistency)
    scz_in_bg = [g for g in scz_df["hgnc_symbol"] if g in bg_set]
    # use bg length
    gene_to_len = {g: l for g, l in zip(bg_genes_arr, bg_lens)}
    scz_lens = np.array([gene_to_len[g] for g in scz_in_bg], dtype=float)
    print(f"F098 bg n={len(bg_genes_arr)}  scz_in_bg n={len(scz_in_bg)}")

    rng = np.random.default_rng(RNG_SEED)
    kernels = []
    specs = [
        ("linear_10pct",  lambda: build_buckets_linear(bg_genes_arr, bg_lens, scz_lens, 0.1)),
        ("log_0.1",       lambda: build_buckets_log(bg_genes_arr, bg_lens, scz_lens, 0.1)),
        ("quantile_5",    lambda: build_buckets_quantile(bg_genes_arr, bg_lens, scz_lens, 5)),
        ("quantile_10",   lambda: build_buckets_quantile(bg_genes_arr, bg_lens, scz_lens, 10)),
        ("quantile_20",   lambda: build_buckets_quantile(bg_genes_arr, bg_lens, scz_lens, 20)),
    ]
    for name, build in specs:
        print(f"  building {name} ...")
        buckets = build()
        r = run_kernel(name, buckets, bg_genes_arr, scz_in_bg, markers, rng)
        kernels.append(r)
        print(f"F098 | {name}: a={r['a_obs']} OR_obs={r['OR_obs']:.3g} "
              f"adj_OR={r['adj_OR']:.3g} emp_p={r['emp_p']:.4f}")
    elapsed = time.time() - t0
    out = {"finding": "F098", "headline_adj_OR": 6.94, "headline_source": "batch_040_d45",
           "cells": kernels, "elapsed_s": elapsed}
    (OUTPUT_DIR / "f098_length_kernels.json").write_text(json.dumps(out, indent=2))
    log_event(log, f"[03_f098] elapsed={elapsed:.2f}s cells={len(kernels)}")

if __name__ == "__main__":
    main()
