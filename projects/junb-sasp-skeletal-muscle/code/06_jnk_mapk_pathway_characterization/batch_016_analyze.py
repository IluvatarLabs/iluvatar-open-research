#!/usr/bin/env python3
"""
batch_016: MAP2K4/7 Gene Expression Analysis
Tests MAP2K4 (MKK4) and MAP2K7 (MKK7) expression in aged FAPs and MuSCs.
Role: Descriptive characterization of JNK pathway transcriptional state.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, ttest_ind
import warnings
import json
warnings.filterwarnings('ignore')

DATA_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data"
OUT_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_016"

GENES = {
    'JNKK': ['MAP2K4', 'MAP2K7'],  # JNKKs
    'JNK_TARGETS': ['FOS', 'JUNB'],  # Reference: known age-up
    'P38_TARGETS': ['MAPK14', 'MEF2A'],  # Reference: known null
}

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dx, dy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = np.sqrt(((nx-1)*dx**2 + (ny-1)*dy**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0

def smart_corr(x, y, min_detect=10):
    mask = (x > 0) & (y > 0)
    n = int(mask.sum())
    if n < min_detect:
        return np.nan, np.nan, 0
    r, p = spearmanr(x[mask], y[mask])
    return r, p, n

def analyze_idx(adata, young_idx, aged_idx, compartment, label):
    """Analyze age effects and correlations using integer index arrays."""
    results = {}
    all_genes = GENES['JNKK'] + GENES['JNK_TARGETS'] + GENES['P38_TARGETS']
    var_names = list(adata.var_names)
    gene_idx = {g: var_names.index(g) for g in all_genes if g in var_names}

    print(f"  {label} ({compartment}): Young={len(young_idx):,}, Aged={len(aged_idx):,}")
    print(f"  {'Gene':<10} {'Detect_Y':>8} {'Detect_A':>8} {'Mean_Y':>8} {'Mean_A':>8} {'d':>8} {'p':>8}")

    for gene in all_genes:
        if gene not in gene_idx:
            print(f"  {gene:<10} NOT FOUND")
            continue

        idx = gene_idx[gene]
        x_y = adata.X[young_idx, idx]
        x_a = adata.X[aged_idx, idx]
        y_vals = x_y.toarray().flatten() if hasattr(x_y, 'toarray') else np.asarray(x_y).flatten()
        a_vals = x_a.toarray().flatten() if hasattr(x_a, 'toarray') else np.asarray(x_a).flatten()

        pct_y = (y_vals > 0).mean() * 100
        pct_a = (a_vals > 0).mean() * 100
        mean_y = y_vals[y_vals > 0].mean() if pct_y > 0 else 0
        mean_a = a_vals[a_vals > 0].mean() if pct_a > 0 else 0

        d = cohens_d(a_vals, y_vals)
        _, p = ttest_ind(a_vals, y_vals, equal_var=False)

        print(f"  {gene:<10} {pct_y:>7.1f}% {pct_a:>7.1f}% {mean_y:>8.3f} {mean_a:>8.3f} {d:>8.3f} {p:>8.2e}")
        results[gene] = {'detect_young': pct_y, 'detect_aged': pct_a,
                         'mean_young': mean_y, 'mean_aged': mean_a,
                         'd': float(d), 'p_age': float(p), 'n_young': len(young_idx), 'n_aged': len(aged_idx)}

    # Correlations (aged cells only)
    print(f"  Correlation analysis (aged cells only):")
    print(f"  {'Pair':<25} {'N':>8} {'rho':>8} {'p':>10}")

    for jnkk in GENES['JNKK']:
        if jnkk not in gene_idx:
            continue
        for target in ['FOS', 'MAPK14', 'MEF2A', 'JUNB']:
            if target not in gene_idx:
                continue

            j_raw = adata.X[aged_idx, gene_idx[jnkk]]
            t_raw = adata.X[aged_idx, gene_idx[target]]
            j_vals = j_raw.toarray().flatten() if hasattr(j_raw, 'toarray') else np.asarray(j_raw).flatten()
            t_vals = t_raw.toarray().flatten() if hasattr(t_raw, 'toarray') else np.asarray(t_raw).flatten()

            rho, p, n_det = smart_corr(j_vals, t_vals, min_detect=10)
            pair = f"{jnkk}×{target}"
            results[pair] = {'rho': float(rho), 'p': float(p), 'n': n_det}
            if not np.isnan(rho):
                print(f"  {pair:<25} {n_det:>8} {rho:>8.3f} {p:>10.2e}")

    return results

print("=" * 70)
print("batch_016: MAP2K4/7 Gene Expression Analysis")
print("=" * 70)

results = {}

# Load data
print("\n[1] Loading data...")
fap_hlma = sc.read(f"{DATA_DIR}/OMIX004308-02.h5ad", backed=True)
print(f"  HLMA FAPs: {fap_hlma.shape[0]:,} cells")
musc = sc.read(f"{DATA_DIR}/MuSC_scsn_RNA.h5ad", backed=True)
print(f"  MuSCs: {musc.shape[0]:,} cells")
fap_na = sc.read(f"{DATA_DIR}/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad", backed=True)
print(f"  Nature Aging FAPs: {fap_na.shape[0]:,} cells")

# HLMA FAPs
print("\n" + "=" * 70)
print("[2] HLMA FAPs")
print("=" * 70)
fap_young_mask = fap_hlma.obs['age_pop'] == 'young_pop'
fap_aged_mask = fap_hlma.obs['age_pop'] == 'old_pop'
fap_y_idx = np.where(fap_young_mask.values)[0]
fap_a_idx = np.where(fap_aged_mask.values)[0]
results['hlma_fap'] = analyze_idx(fap_hlma, fap_y_idx, fap_a_idx, 'FAP', 'HLMA FAPs')

# MuSCs
print("\n" + "=" * 70)
print("[3] MuSCs")
print("=" * 70)
musc_young_mask = musc.obs['age_pop'] == 'young_pop'
musc_aged_mask = musc.obs['age_pop'] == 'old_pop'
musc_y_idx = np.where(musc_young_mask.values)[0]
musc_a_idx = np.where(musc_aged_mask.values)[0]
results['musc'] = analyze_idx(musc, musc_y_idx, musc_a_idx, 'MuSC', 'MuSCs')

# Nature Aging FAPs
print("\n" + "=" * 70)
print("[4] Nature Aging FAPs")
print("=" * 70)

young_bins = ['15-20', '25-30', '35-40']
na_young_mask = fap_na.obs['Age_group'].isin(young_bins)
na_aged_mask = ~fap_na.obs['Age_group'].isin(young_bins)
print(f"  All cells: Young={na_young_mask.sum():,}, Aged={na_aged_mask.sum():,}")

if 'PDGFRA' in fap_na.var_names and 'DCN' in fap_na.var_names:
    na_pdg_raw = fap_na[:, 'PDGFRA'].X[:]
    na_dcn_raw = fap_na[:, 'DCN'].X[:]
    na_pdg = na_pdg_raw.toarray().flatten() > 0 if hasattr(na_pdg_raw, 'toarray') else np.asarray(na_pdg_raw).flatten() > 0
    na_dcn = na_dcn_raw.toarray().flatten() > 0 if hasattr(na_dcn_raw, 'toarray') else np.asarray(na_dcn_raw).flatten() > 0
    is_fap = na_pdg & na_dcn
    n_fap = int(is_fap.sum())
    print(f"  FAPs (PDGFRA+/DCN+): {n_fap:,} / {fap_na.shape[0]:,} cells")

    na_y_idx = np.where(na_young_mask.values & is_fap)[0]
    na_a_idx = np.where(na_aged_mask.values & is_fap)[0]
    results['na_fap'] = analyze_idx(fap_na, na_y_idx, na_a_idx, 'FAP', 'NA FAPs')
else:
    print("  PDGFRA/DCN not found — using all cells")
    na_y_idx = np.where(na_young_mask.values)[0]
    na_a_idx = np.where(na_aged_mask.values)[0]
    results['na_fap'] = analyze_idx(fap_na, na_y_idx, na_a_idx, 'FAP', 'NA cells')

# Summary
print("\n" + "=" * 70)
print("[5] Summary: MAP2K4/7 Age Effects and FOS Correlations")
print("=" * 70)

rows = []
for ds, label in [('hlma_fap', 'HLMA FAPs'), ('musc', 'MuSCs'), ('na_fap', 'NA FAPs')]:
    if ds not in results:
        continue
    for gene in ['MAP2K4', 'MAP2K7', 'FOS']:
        if gene in results[ds]:
            r = results[ds][gene]
            rows.append({'Dataset': label, 'Gene': gene, 'd': round(r['d'], 3), 'p': f"{r['p_age']:.2e}",
                         'Detect_Aged': f"{r['detect_aged']:.1f}%"})

df = pd.DataFrame(rows)
print(df.to_string(index=False))

print("\n[Correlations with FOS (aged cells)]")
print(f"{'Dataset':<15} {'Pair':<15} {'N':>8} {'rho':>8}")
for ds, label in [('hlma_fap', 'HLMA FAPs'), ('musc', 'MuSCs'), ('na_fap', 'NA FAPs')]:
    if ds not in results:
        continue
    for jnkk in ['MAP2K4', 'MAP2K7']:
        pair = f"{jnkk}×FOS"
        if pair in results[ds] and not np.isnan(results[ds][pair]['rho']):
            r = results[ds][pair]
            print(f"  {label:<15} {pair:<15} {r['n']:>8} {r['rho']:>8.3f}")

# Save
output = {'batch': 'batch_016', 'date': '2026-04-09',
          'results': {k: {kk: float(vv) if isinstance(vv, np.floating) else vv
                          for kk, vv in v.items()} for k, v in results.items()}}
with open(f"{OUT_DIR}/results.json", 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n[DONE] Results saved to {OUT_DIR}/results.json")
print("\nNOTE: MAP2K4/7 mRNA does NOT reflect JNK kinase activity (post-translational).")
print("F047/F048 remain the primary therapeutic evidence.")