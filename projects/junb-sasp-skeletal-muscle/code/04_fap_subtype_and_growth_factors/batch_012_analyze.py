#!/usr/bin/env python3
"""
batch_012: FAP Pro-Regenerative Axis + JUNB Co-Expression

Tests:
1. H032: Aged FAPs downregulate HGF/IGF1/FGF7 (regenerative secretome decline)
2. H033: JUNB+ FAPs show negative correlation with regenerative genes
3. H034: JUNB+ cells show co-expression with DDR/NF-kB genes

Descriptive only - no causal claims.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import anndata as ad
import json
import warnings
warnings.filterwarnings('ignore')

def cohen_d(x, y):
    """Cohen d for age effect."""
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return np.nan
    m1, m2 = np.mean(x), np.mean(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    if pooled == 0:
        return np.nan
    return (m2 - m1) / pooled

def fdr_correction(p_vals):
    """Benjamini-Hochberg FDR correction."""
    p_vals = np.array(p_vals)
    n = len(p_vals)
    ranks = np.argsort(np.argsort(p_vals))
    adjusted = p_vals * n / (ranks + 1)
    adjusted = np.minimum(adjusted, 1.0)
    for i in range(n-2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i+1])
    return adjusted

print("=" * 60)
print("batch_012: FAP Pro-Regenerative Axis + JUNB Co-Expression")
print("=" * 60)

results = {}

# =============================================================================
# EXPERIMENT 1: FAP Secretome Age Effects
# =============================================================================
print("\n## EXPERIMENT 1: FAP Secretome Age Effects")
print("-" * 40)

# Gene sets
REGENERATIVE_GENES = ['HGF', 'IGF1', 'IGF2', 'FGF7']
SASP_GENES = ['IL6', 'CXCL8', 'CCL2', 'IL1B', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'VEGFA', 'CCL5']

reg_results = []
sasp_results = []

# --- HLMA FAPs ---
print("\n=== HLMA FAPs ===")
hlma = ad.read_h5ad('data/OMIX004308-02.h5ad', backed='r')

# Subset to FAPs - materialize to memory first
fap_types = ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP']
fap_mask = np.isin(hlma.obs.Annotation.values, fap_types)
hlma_faps = hlma[fap_mask].to_memory()
print(f"HLMA FAPs: {hlma_faps.shape[0]} cells")

# Age groups (HLMA uses 'old_pop' and 'young_pop')
old_mask = hlma_faps.obs.age_pop.values == 'old_pop'
young_mask = hlma_faps.obs.age_pop.values == 'young_pop'
hlma_old_n = old_mask.sum()
hlma_young_n = young_mask.sum()
print(f"Old: {hlma_old_n}, Young: {hlma_young_n}")

for gene in REGENERATIVE_GENES + SASP_GENES:
    if gene not in hlma_faps.var_names:
        continue

    for fap_type in fap_types:
        subtype_mask = hlma_faps.obs.Annotation.values == fap_type
        old_idx = subtype_mask & old_mask
        young_idx = subtype_mask & young_mask

        old_cells = hlma_faps[old_idx, gene].X[:]
        young_cells = hlma_faps[young_idx, gene].X[:]

        if hasattr(old_cells, 'todense'):
            old_cells = np.array(old_cells.todense()).flatten()
        if hasattr(young_cells, 'todense'):
            young_cells = np.array(young_cells.todense()).flatten()

        old_cells = old_cells[~np.isnan(old_cells)]
        young_cells = young_cells[~np.isnan(young_cells)]

        detection = 100 * (old_cells > 0).mean()

        if len(old_cells) > 10 and len(young_cells) > 10 and detection > 10:
            d = cohen_d(old_cells, young_cells)

            # JUNB correlation within old cells
            old_junb = hlma_faps[old_idx, 'JUNB'].X[:]
            if hasattr(old_junb, 'todense'):
                old_junb = np.array(old_junb.todense()).flatten()
            old_junb = old_junb[~np.isnan(old_junb)]

            if len(old_cells) == len(old_junb) and len(old_cells) > 50:
                rho, p = spearmanr(old_junb, old_cells)

                res = {
                    'atlas': 'HLMA',
                    'subtype': fap_type,
                    'gene': gene,
                    'd': float(d) if not np.isnan(d) else None,
                    'N_old': len(old_cells),
                    'N_young': len(young_cells),
                    'detection_pct': float(detection),
                    'rho_JUNB': float(rho) if not np.isnan(rho) else None,
                    'p_JUNB': float(p) if not np.isnan(p) else None,
                    'gene_type': 'regenerative' if gene in REGENERATIVE_GENES else 'SASP'
                }
                if gene in REGENERATIVE_GENES:
                    reg_results.append(res)
                else:
                    sasp_results.append(res)

# --- Nature Aging FAPs ---
print("\n=== Nature Aging FAPs ===")
na = ad.read_h5ad('data/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad', backed='r')

# FAP types - materialize to memory
na_fap_mask = np.isin(na.obs.annotation_level1.values, ['FB', 'EnFB', 'PnFB'])
na_faps = na[na_fap_mask].to_memory()
print(f"Nature Aging FAPs: {na_faps.shape[0]} cells")

# Young: 15-30, Old: 60-75
na_old_mask = np.isin(na_faps.obs.Age_group.values, ['60-65', '70-75'])
na_young_mask = np.isin(na_faps.obs.Age_group.values, ['15-20', '25-30'])
na_fap_types = ['Inter_FB', 'Par_FB', 'Adv_FB']
print(f"Old: {na_old_mask.sum()}, Young: {na_young_mask.sum()}")

for gene in REGENERATIVE_GENES + SASP_GENES:
    if gene not in na_faps.var_names:
        continue

    for fap_type in na_fap_types:
        subtype_mask = na_faps.obs.annotation_level2.values == fap_type
        old_idx = subtype_mask & na_old_mask
        young_idx = subtype_mask & na_young_mask

        old_cells = na_faps[old_idx, gene].X[:]
        young_cells = na_faps[young_idx, gene].X[:]

        if hasattr(old_cells, 'todense'):
            old_cells = np.array(old_cells.todense()).flatten()
        if hasattr(young_cells, 'todense'):
            young_cells = np.array(young_cells.todense()).flatten()

        old_cells = old_cells[~np.isnan(old_cells)]
        young_cells = young_cells[~np.isnan(young_cells)]

        detection = 100 * (old_cells > 0).mean()

        if len(old_cells) > 10 and len(young_cells) > 10 and detection > 10:
            d = cohen_d(old_cells, young_cells)

            # JUNB correlation within old cells
            old_junb = na_faps[old_idx, 'JUNB'].X[:]
            if hasattr(old_junb, 'todense'):
                old_junb = np.array(old_junb.todense()).flatten()
            old_junb = old_junb[~np.isnan(old_junb)]

            if len(old_cells) == len(old_junb) and len(old_cells) > 50:
                rho, p = spearmanr(old_junb, old_cells)

                res = {
                    'atlas': 'Nature Aging',
                    'subtype': fap_type,
                    'gene': gene,
                    'd': float(d) if not np.isnan(d) else None,
                    'N_old': len(old_cells),
                    'N_young': len(young_cells),
                    'detection_pct': float(detection),
                    'rho_JUNB': float(rho) if not np.isnan(rho) else None,
                    'p_JUNB': float(p) if not np.isnan(p) else None,
                    'gene_type': 'regenerative' if gene in REGENERATIVE_GENES else 'SASP'
                }
                if gene in REGENERATIVE_GENES:
                    reg_results.append(res)
                else:
                    sasp_results.append(res)

# --- Aggregate results ---
all_results = reg_results + sasp_results
df = pd.DataFrame(all_results)
if len(df) > 0 and 'p_JUNB' in df.columns:
    df['p_JUNB_adj'] = fdr_correction(df['p_JUNB'].values)

print("\n=== REGENERATIVE GENES: Age Effect (Cohen d) + JUNB Correlation ===")
print("(Negative d = downregulated with age; Negative rho = JUNB+ cells have low regenerative genes)")
reg_df = df[df.gene_type == 'regenerative'].copy() if len(df) > 0 else pd.DataFrame()
if len(reg_df) > 0:
    print(reg_df[['atlas', 'subtype', 'gene', 'd', 'N_old', 'rho_JUNB', 'p_JUNB_adj', 'detection_pct']].to_string(index=False))
else:
    print("No data")

print("\n=== SASP GENES: Age Effect (Cohen d) + JUNB Correlation ===")
print("(Positive d = upregulated with age; Positive rho = JUNB+ cells have high SASP)")
sasp_df = df[df.gene_type == 'SASP'].copy() if len(df) > 0 else pd.DataFrame()
if len(sasp_df) > 0:
    print(sasp_df[['atlas', 'subtype', 'gene', 'd', 'N_old', 'rho_JUNB', 'p_JUNB_adj', 'detection_pct']].to_string(index=False))
else:
    print("No data")

results['exp1_regenerative'] = reg_df.to_dict('records') if len(reg_df) > 0 else []
results['exp1_sasp'] = sasp_df.to_dict('records') if len(sasp_df) > 0 else []

# =============================================================================
# EXPERIMENT 2: JUNB Co-Expression Landscape
# =============================================================================
print("\n\n## EXPERIMENT 2: JUNB Co-Expression Landscape")
print("-" * 40)

DDR_GENES = ['TP53', 'CDKN1A', 'GADD45A', 'GADD45B', 'BBC3', 'MDM2']
NFKB_GENES = ['NFKB1', 'NFKBIA', 'NFKBIZ', 'RELA', 'BCL3']

def compute_junb_correlates(adata_cells, label):
    """Compute JUNB co-expression with pathway genes."""
    if adata_cells.shape[0] == 0:
        print(f"  {label}: 0 cells, skipping...")
        return {'DDR_rho': 0.0, 'DDR_p': 1.0, 'NFKB_rho': 0.0, 'NFKB_p': 1.0,
                'top_positive': [], 'top_negative': [], 'DDR_genes_found': [], 'NFKB_genes_found': []}

    junb_expr = adata_cells[:, 'JUNB'].X[:]
    if hasattr(junb_expr, 'todense'):
        junb_expr = np.array(junb_expr.todense()).flatten()
    else:
        junb_expr = np.array(junb_expr).flatten()
    junb_expr = np.nan_to_num(junb_expr, nan=0)
    n_cells = len(junb_expr)
    print(f"  {label}: {n_cells} cells, computing correlations...")

    results_local = {}

    # DDR score vs JUNB
    ddr_genes_found = [g for g in DDR_GENES if g in adata_cells.var_names]
    if len(ddr_genes_found) > 0 and n_cells > 0:
        ddr_vals = np.zeros(n_cells)
        for g in ddr_genes_found:
            g_vals = adata_cells[:, g].X[:]
            if hasattr(g_vals, 'todense'):
                g_vals = np.array(g_vals.todense()).flatten()
            else:
                g_vals = np.array(g_vals).flatten()
            g_vals = np.nan_to_num(g_vals, nan=0)
            if len(g_vals) == n_cells:
                ddr_vals += g_vals
        ddr_vals /= len(ddr_genes_found)
        rho_ddr, p_ddr = spearmanr(junb_expr, ddr_vals)
        results_local['DDR_rho'] = float(rho_ddr) if not np.isnan(rho_ddr) else 0.0
        results_local['DDR_p'] = float(p_ddr) if not np.isnan(p_ddr) else 1.0
        print(f"    DDR score vs JUNB: rho={rho_ddr:.3f}, p={p_ddr:.2e}")

    # NF-kB score vs JUNB
    nfkba_genes_found = [g for g in NFKB_GENES if g in adata_cells.var_names]
    if len(nfkba_genes_found) > 0 and n_cells > 0:
        nfkb_vals = np.zeros(n_cells)
        for g in nfkba_genes_found:
            g_vals = adata_cells[:, g].X[:]
            if hasattr(g_vals, 'todense'):
                g_vals = np.array(g_vals.todense()).flatten()
            else:
                g_vals = np.array(g_vals).flatten()
            g_vals = np.nan_to_num(g_vals, nan=0)
            if len(g_vals) == n_cells:
                nfkb_vals += g_vals
        nfkb_vals /= len(nfkba_genes_found)
        rho_nfkb, p_nfkb = spearmanr(junb_expr, nfkb_vals)
        results_local['NFKB_rho'] = float(rho_nfkb) if not np.isnan(rho_nfkb) else 0.0
        results_local['NFKB_p'] = float(p_nfkb) if not np.isnan(p_nfkb) else 1.0
        print(f"    NF-kB score vs JUNB: rho={rho_nfkb:.3f}, p={p_nfkb:.2e}")

    # Top correlating genes with JUNB
    n_genes = min(3000, adata_cells.shape[1])
    gene_names = list(adata_cells.var_names[:n_genes])
    corrs = []

    for i, g in enumerate(gene_names):
        if g == 'JUNB':
            continue
        g_vals = adata_cells[:, g].X[:]
        if hasattr(g_vals, 'todense'):
            g_vals = np.array(g_vals.todense()).flatten()
        else:
            g_vals = np.array(g_vals).flatten()
        g_vals = np.nan_to_num(g_vals, nan=0)

        if len(g_vals) == n_cells and (g_vals > 0).mean() > 0.05:
            rho, p = spearmanr(junb_expr, g_vals)
            if not np.isnan(rho) and not np.isnan(p):
                corrs.append((g, float(rho), float(p)))

        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n_genes} genes processed")

    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_pos = [x for x in corrs if x[1] > 0][:50]
    top_neg = [x for x in corrs if x[1] < 0][:50]

    print(f"    Top 10 positively correlated with JUNB:")
    for g, r, p in top_pos[:10]:
        print(f"      {g}: rho={r:.3f}")
    print(f"    Top 10 negatively correlated with JUNB:")
    for g, r, p in top_neg[:10]:
        print(f"      {g}: rho={r:.3f}")

    results_local['top_positive'] = [(g, r, p) for g, r, p in top_pos]
    results_local['top_negative'] = [(g, r, p) for g, r, p in top_neg]
    results_local['DDR_genes_found'] = ddr_genes_found
    results_local['NFKB_genes_found'] = nfkba_genes_found

    return results_local

# HLMA
print("\n=== HLMA Old FAPs ===")
hlma_old_mask = np.isin(hlma.obs.Annotation.values, fap_types) & (hlma.obs.age_pop.values == 'old_pop')
hlma_old_cells = hlma[hlma_old_mask].to_memory()
print(f"HLMA old FAPs: {hlma_old_cells.shape[0]} cells")
hlma_coexp = compute_junb_correlates(hlma_old_cells, "HLMA")

# Nature Aging
print("\n=== Nature Aging Old FAPs ===")
na_old_mask = np.isin(na.obs.annotation_level1.values, ['FB', 'EnFB', 'PnFB']) & np.isin(na.obs.Age_group.values, ['60-65', '70-75'])
na_old_cells = na[na_old_mask].to_memory()
print(f"Nature Aging old FAPs: {na_old_cells.shape[0]} cells")
na_coexp = compute_junb_correlates(na_old_cells, "Nature Aging")

# Cross-atlas validation
print("\n=== Cross-Atlas Validation ===")
hlma_top_genes = set([g for g, r, p in hlma_coexp.get('top_positive', [])[:100]])
na_top_genes = set([g for g, r, p in na_coexp.get('top_positive', [])[:100]])
overlap = hlma_top_genes & na_top_genes
print(f"HLMA top-100 JUNB+ genes: {len(hlma_top_genes)}")
print(f"Nature Aging top-100 JUNB+ genes: {len(na_top_genes)}")
print(f"Overlap: {len(overlap)} genes")
if len(overlap) > 0:
    print(f"Shared genes: {list(overlap)[:20]}")

results['exp2_HLMA'] = hlma_coexp
results['exp2_NA'] = na_coexp
results['exp2_cross_atlas'] = {
    'overlap_count': len(overlap),
    'overlap_genes': list(overlap),
    'hlma_top_n': len(hlma_top_genes),
    'na_top_n': len(na_top_genes)
}

# =============================================================================
# DECISION RULE EVALUATION
# =============================================================================
print("\n\n## DECISION RULE EVALUATION")
print("-" * 40)

# H032: Regenerative genes downregulated with age
print("\nH032: Aged FAPs downregulate HGF/IGF1/FGF7")
reg_downs = [r for r in reg_results if r.get('d') is not None and r['d'] < -0.2 and r['detection_pct'] > 10]
if reg_downs:
    print(f"  SUPPORTED: {len(reg_downs)} subtypes show age downregulation")
    for r in reg_downs[:5]:
        print(f"    {r['atlas']}/{r['subtype']}/{r['gene']}: d={r['d']:.3f}")
else:
    print("  NOT SUPPORTED: No subtypes show age downregulation of regenerative genes")

# H033: JUNB negatively correlates with regenerative genes
print("\nH033: JUNB+ FAPs negatively correlate with regenerative genes")
jund_anticorr = [r for r in reg_results if r.get('rho_JUNB') is not None and r['rho_JUNB'] < -0.10 and r['detection_pct'] > 10]
if jund_anticorr:
    print(f"  SUPPORTED: {len(jund_anticorr)} subtypes show JUNB-regenerative anticorrelation")
    for r in jund_anticorr[:5]:
        print(f"    {r['atlas']}/{r['subtype']}/{r['gene']}: rho={r['rho_JUNB']:.3f}")
else:
    print("  NOT SUPPORTED: No subtypes show JUNB-regenerative anticorrelation")

# H034: JUNB co-expresses with DDR/NF-kB
print("\nH034: JUNB co-expresses with DDR/NF-kB genes")
ddr_support = hlma_coexp.get('DDR_rho', 0) > 0.10 or na_coexp.get('DDR_rho', 0) > 0.10
nfkb_support = hlma_coexp.get('NFKB_rho', 0) > 0.10 or na_coexp.get('NFKB_rho', 0) > 0.10
if ddr_support:
    print(f"  DDR co-expression: SUPPORTED (HLMA rho={hlma_coexp.get('DDR_rho', 0):.3f}, NA rho={na_coexp.get('DDR_rho', 0):.3f})")
else:
    print(f"  DDR co-expression: NOT SUPPORTED (HLMA rho={hlma_coexp.get('DDR_rho', 0):.3f}, NA rho={na_coexp.get('DDR_rho', 0):.3f})")
if nfkb_support:
    print(f"  NF-kB co-expression: SUPPORTED (HLMA rho={hlma_coexp.get('NFKB_rho', 0):.3f}, NA rho={na_coexp.get('NFKB_rho', 0):.3f})")
else:
    print(f"  NF-kB co-expression: NOT SUPPORTED (HLMA rho={hlma_coexp.get('NFKB_rho', 0):.3f}, NA rho={na_coexp.get('NFKB_rho', 0):.3f})")

print("\n" + "=" * 60)

# Save results
with open('experiments/batch_012/results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to experiments/batch_012/results.json")
