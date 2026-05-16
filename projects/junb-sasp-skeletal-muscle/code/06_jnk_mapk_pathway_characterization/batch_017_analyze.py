#!/usr/bin/env python3
"""
batch_017: MAP3K Upstream Activators of JNK Analysis
Tests JNK-specific MAP3K genes (MLK2/3) for age effects in FAPs and MuSCs.
Role: Descriptive characterization of upstream JNK pathway activators.

REVISED per science-critic review:
- Replace t-test with Wilcoxon rank-sum (non-parametric for scRNA)
- Add Benjamini-Hochberg correction for multiple comparisons
- Focus on JNK-specific MAP3Ks (MLK2/3) — remove p38/ERK MAP3Ks
- Report effective N for co-detected cells
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
import json
import sys
warnings.filterwarnings('ignore')

DATA_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data"
OUT_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_017"

# JNK-specific MAP3K genes (prioritized per biologist critic)
# MLK2/3 (MAP3K10/11) directly activate MKK4/7 (JNKKs)
JNK_SPECIFIC_MAP3KS = {
    'MLK2': 'MAP3K10',      # Mixed lineage kinase 2
    'MLK3': 'MAP3K11',     # Mixed lineage kinase 3
    'HPK1': 'MAP4K1',      # Hematopoietic progenitor kinase 1
    'HPK2': 'MAP4K2',      # MAP4K2 (GCK-like)
    'HPK4': 'MAP4K4',      # MAP4K4 (HPK4)
}

# Also test MAP3K5 (ASK1) and MAP3K7 (TAK1) but note they activate both JNK and p38
DUAL_SPECIFICITY_MAP3KS = {
    'ASK1': 'MAP3K5',      # Activates JNK AND p38
    'TAK1': 'MAP3K7',      # Activates JNK AND p38
}

# Reference genes
REFERENCE_GENES = {
    'JNK_TARGETS': ['FOS', 'JUNB'],  # JNK pathway targets
    'JNKK': ['MAP2K4', 'MAP2K7'],     # JNKKs (known null)
}

YOUNG_THRESHOLD = 40  # age < 40
OLD_THRESHOLD = 65    # age > 65

def cohens_d(x, y):
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dx, dy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = np.sqrt(((nx-1)*dx**2 + (ny-1)*dy**2) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0

def smart_corr(x, y, min_detect=10):
    """Compute Spearman correlation for co-detected cells."""
    mask = (x > 0) & (y > 0)
    n = int(mask.sum())
    if n < min_detect:
        return np.nan, np.nan, 0
    r, p = spearmanr(x[mask], y[mask])
    return r, p, n

def get_age_idx(adata):
    """Get young and aged indices based on age threshold.

    Handles multiple column formats:
    1. Numeric age (int/float): threshold by value
    2. String categories ('84', '79'): convert to float then threshold
    3. Age_bin ('young', 'old'): use directly
    """
    # Priority: Age_bin > Age_group > age
    age_col = None
    for col in ['Age_bin', 'age_bin', 'Age_group', 'age_group', 'age', 'Age']:
        if col in adata.obs.columns:
            age_col = col
            break

    if age_col is None:
        raise KeyError(f"No age column found. Available: {adata.obs.columns.tolist()}")

    vals = adata.obs[age_col]

    # If Age_bin with 'young'/'old', use directly
    if age_col in ['Age_bin', 'age_bin']:
        vals_str = vals.astype(str).str.lower()
        young_idx = adata.obs[vals_str == 'young'].index.tolist()
        aged_idx = adata.obs[vals_str == 'old'].index.tolist()
        return young_idx, aged_idx

    # Try to convert to numeric (handles string categories like '84', '79')
    try:
        ages = vals.astype(float)
        young_idx = adata.obs[ages < YOUNG_THRESHOLD].index.tolist()
        aged_idx = adata.obs[ages > OLD_THRESHOLD].index.tolist()
        return young_idx, aged_idx
    except (ValueError, TypeError):
        pass

    # Fallback: look for Young/Aged in string values
    vals_str = vals.astype(str).str.lower()
    if vals_str.str.contains('young').any():
        young_idx = adata.obs[vals_str.str.contains('young')].index.tolist()
        aged_idx = adata.obs[vals_str.str.contains('old|aged')].index.tolist()
        return young_idx, aged_idx

    raise KeyError(f"Cannot interpret age column '{age_col}'")

def analyze_age_effects(adata, young_idx, aged_idx, label, alpha=0.05):
    """Analyze age effects for all MAP3K genes using Wilcoxon test."""
    results = {}
    all_genes = list(JNK_SPECIFIC_MAP3KS.values()) + list(DUAL_SPECIFICITY_MAP3KS.values()) + REFERENCE_GENES['JNK_TARGETS'] + REFERENCE_GENES['JNKK']
    var_names = adata.var_names.tolist()

    # Filter to genes present in data
    present_genes = [g for g in all_genes if g in var_names]
    missing_genes = [g for g in all_genes if g not in var_names]

    if missing_genes:
        print(f"  [INFO] Genes not in {label}: {missing_genes}")

    print(f"\n  {'Gene':<12} {'DetY%':>6} {'DetA%':>6} {'d':>7} {'Wilcoxon_p':>12} {'N_Y':>6} {'N_A':>6}")
    print(f"  {'-'*70}")

    # Convert barcode indices to integer positions
    young_pos = [adata.obs.index.tolist().index(b) for b in young_idx]
    aged_pos = [adata.obs.index.tolist().index(b) for b in aged_idx]

    p_values = []
    gene_list = []

    for gene in present_genes:
        gene_idx = var_names.index(gene)

        y_vals = adata.X[young_pos, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[young_pos, gene_idx]
        a_vals = adata.X[aged_pos, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[aged_pos, gene_idx]

        detect_y = float((y_vals > 0).mean() * 100)
        detect_a = float((a_vals > 0).mean() * 100)
        d = cohens_d(y_vals, a_vals)

        # Wilcoxon rank-sum (Mann-Whitney) — non-parametric for zero-inflated scRNA
        try:
            stat, p = mannwhitneyu(y_vals, a_vals, alternative='two-sided')
        except ValueError:
            p = 1.0

        p_values.append(p)
        gene_list.append(gene)

        results[gene] = {
            'detect_young': detect_y,
            'detect_aged': detect_a,
            'd': float(d),
            'p': float(p),
            'n_young': int(len(young_pos)),
            'n_aged': int(len(aged_pos)),
        }

        marker = ''
        if p < 0.05:
            marker = '*'
        if p < alpha / len(present_genes):
            marker = '**'  # Bonferroni-significant

        print(f"  {gene:<12} {detect_y:>6.1f} {detect_a:>6.1f} {d:>7.3f} {p:>12.2e}{marker} {len(young_idx):>6} {len(aged_idx):>6}")

    # Benjamini-Hochberg correction
    if len(p_values) > 0:
        reject, p_corr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        print(f"\n  [BH correction] FDR-adjusted threshold: {alpha}")

        for i, gene in enumerate(gene_list):
            results[gene]['p_BH'] = float(p_corr[i])
            results[gene]['significant_BH'] = bool(reject[i])

        sig_genes = [g for g, r in zip(gene_list, reject) if r]
        print(f"  [BH] Significant genes: {len(sig_genes)}")
        if sig_genes:
            print(f"  [BH] {sig_genes}")

    return results

def analyze_correlations(adata, aged_idx, label, min_pairs=50):
    """Analyze correlations between MAP3Ks and FOS/JUNB in aged cells."""
    results = {}
    all_genes = list(JNK_SPECIFIC_MAP3KS.values()) + list(DUAL_SPECIFICITY_MAP3KS.values())
    var_names = adata.var_names.tolist()

    present_genes = [g for g in all_genes if g in var_names]

    print(f"\n  MAP3K-FOS correlations in aged {label} (co-detected cells only):")
    print(f"  {'MAP3K':<12} {'rho':>8} {'p':>12} {'N_pairs':>10}")
    print(f"  {'-'*45}")

    # Convert barcode indices to integer positions
    aged_pos = [adata.obs.index.tolist().index(b) for b in aged_idx]

    for gene in present_genes:
        gene_idx = var_names.index(gene)
        fos_idx = var_names.index('FOS') if 'FOS' in var_names else None

        x_vals = adata.X[aged_pos, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[aged_pos, gene_idx]

        if fos_idx is not None:
            y_vals = adata.X[aged_pos, fos_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[aged_pos, fos_idx]
            rho, p, n = smart_corr(x_vals, y_vals, min_detect=min_pairs)
            results[f'{gene}_FOS'] = {'rho': rho, 'p': p, 'n': n, 'detectable': n >= min_pairs}
            if n >= min_pairs:
                print(f"  {gene:<12} {rho:>8.3f} {p:>12.2e} {n:>10}")
            else:
                print(f"  {gene:<12} {'N/A':>8} {f'N={n}<{min_pairs}':>12}")

    return results

def main():
    results = {}

    # ============ 1. HLMA FAPs ============
    print("=" * 70)
    print("1. HLMA FAPs Analysis")
    print("=" * 70)

    adata_fap = sc.read_h5ad(f"{DATA_DIR}/OMIX004308-02.h5ad")
    print(f"FAP data: {adata_fap.n_obs:,} cells, {adata_fap.n_vars:,} genes")

    young_idx, aged_idx = get_age_idx(adata_fap)
    print(f"Young (< {YOUNG_THRESHOLD}): {len(young_idx):,}, Aged (> {OLD_THRESHOLD}): {len(aged_idx):,}")

    fap_age_results = analyze_age_effects(adata_fap, young_idx, aged_idx, "HLMA FAPs")
    fap_corr_results = analyze_correlations(adata_fap, aged_idx, "FAPs")

    results['HLMA_FAPs'] = {
        'age_effects': fap_age_results,
        'correlations': fap_corr_results,
        'n_young': len(young_idx),
        'n_aged': len(aged_idx),
    }

    # ============ 2. MuSCs ============
    print("\n" + "=" * 70)
    print("2. MuSC Analysis")
    print("=" * 70)

    adata_musc = sc.read_h5ad(f"{DATA_DIR}/MuSC_scsn_RNA.h5ad")
    print(f"MuSC data: {adata_musc.n_obs:,} cells, {adata_musc.n_vars:,} genes")

    young_idx, aged_idx = get_age_idx(adata_musc)
    print(f"Young (< {YOUNG_THRESHOLD}): {len(young_idx):,}, Aged (> {OLD_THRESHOLD}): {len(aged_idx):,}")

    musc_age_results = analyze_age_effects(adata_musc, young_idx, aged_idx, "MuSCs")
    musc_corr_results = analyze_correlations(adata_musc, aged_idx, "MuSCs")

    results['MuSCs'] = {
        'age_effects': musc_age_results,
        'correlations': musc_corr_results,
        'n_young': len(young_idx),
        'n_aged': len(aged_idx),
    }

    # ============ 3. Nature Aging FAPs ============
    print("\n" + "=" * 70)
    print("3. Nature Aging FAPs Analysis")
    print("=" * 70)

    adata_na = sc.read_h5ad(f"{DATA_DIR}/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad")
    print(f"NA FAP data: {adata_na.n_obs:,} cells, {adata_na.n_vars:,} genes")

    young_idx, aged_idx = get_age_idx(adata_na)
    print(f"Young (< {YOUNG_THRESHOLD}): {len(young_idx):,}, Aged (> {OLD_THRESHOLD}): {len(aged_idx):,}")

    na_age_results = analyze_age_effects(adata_na, young_idx, aged_idx, "Nature Aging FAPs")
    na_corr_results = analyze_correlations(adata_na, aged_idx, "NA FAPs")

    results['Nature_Aging_FAPs'] = {
        'age_effects': na_age_results,
        'correlations': na_corr_results,
        'n_young': len(young_idx),
        'n_aged': len(aged_idx),
    }

    # ============ Summary ============
    print("\n" + "=" * 70)
    print("SUMMARY: MAP3K Age Effects (d > 0.15 = detectable signal)")
    print("=" * 70)

    summary_data = []
    for dataset, data in results.items():
        for gene, res in data['age_effects'].items():
            if gene in JNK_SPECIFIC_MAP3KS.values() or gene in DUAL_SPECIFICITY_MAP3KS.values():
                summary_data.append({
                    'dataset': dataset,
                    'gene': gene,
                    'd': res['d'],
                    'p_BH': res.get('p_BH', np.nan),
                    'significant': res.get('significant_BH', False),
                    'detect_aged': res['detect_aged'],
                })

    summary_df = pd.DataFrame(summary_data)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values(['dataset', 'd'], ascending=[True, False])
        print("\nJNK-specific MAP3Ks:")
        jnk_specific = summary_df[summary_df['gene'].isin(JNK_SPECIFIC_MAP3KS.values())]
        print(jnk_specific.to_string(index=False))
        print("\nDual-specificity MAP3Ks (also activate p38):")
        dual = summary_df[summary_df['gene'].isin(DUAL_SPECIFICITY_MAP3KS.values())]
        print(dual.to_string(index=False))

    # Save results
    output_file = f"{OUT_DIR}/results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return results

if __name__ == '__main__':
    results = main()
