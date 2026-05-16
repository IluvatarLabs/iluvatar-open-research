#!/usr/bin/env python3
"""
batch_011 Experiment 1: FOS-Collagen Cross-Validation in Nature Aging Atlas
Test FOS-collagen correlation in independent Nature Aging atlas.
"""

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Configuration
DATA_PATH = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad"
OUTPUT_PATH = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_011/exp1_results.json"

# Gene sets
COLLAGEN_GENES = ['COL1A1', 'COL3A1', 'COL6A1', 'COL6A3', 'FN1', 'LOX', 'LOXL1']
SASP_4_GENE = ['CXCL1', 'CXCL2', 'IL6', 'CXCL8']
SASP_12_GENE = SASP_4_GENE + ['IL1B', 'CCL2', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'VEGFA', 'CCL5']
AP1_MEMBERS = ['FOS', 'FOSL1', 'FOSL2', 'JUN', 'JUNB', 'JUND']

# Load data
print("Loading Nature Aging atlas...")
adata = sc.read_h5ad(DATA_PATH)
print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# Check genes exist
available_genes = adata.var_names.tolist()
collagen_present = [g for g in COLLAGEN_GENES if g in available_genes]
sasp4_present = [g for g in SASP_4_GENE if g in available_genes]
sasp12_present = [g for g in SASP_12_GENE if g in available_genes]
ap1_present = [g for g in AP1_MEMBERS if g in available_genes]

print(f"Collagen genes available: {len(collagen_present)}/{len(COLLAGEN_GENES)}")
print(f"SASP 4-gene available: {len(sasp4_present)}/{len(SASP_4_GENE)}")
print(f"SASP 12-gene available: {len(sasp12_present)}/{len(SASP_12_GENE)}")
print(f"AP-1 members available: {len(ap1_present)}/{len(AP1_MEMBERS)}")

# Identify FAPs (PDGFRA+ AND DCN+)
if 'PDGFRA' in available_genes and 'DCN' in available_genes:
    is_pdgfra = adata.obs_vector('PDGFRA') > 0
    is_dcn = adata.obs_vector('DCN') > 0
    is_fap = is_pdgfra & is_dcn
    print(f"FAPs (PDGFRA+ & DCN+): {is_fap.sum()} cells")
else:
    raise ValueError("PDGFRA or DCN not found")

# Age grouping
if 'Age_group' in adata.obs.columns:
    age_col = 'Age_group'
    age_bins = adata.obs['Age_group'].unique()
    print(f"Age bins: {sorted(age_bins)}")

    # Define young (<=40) vs old (>=55)
    young_bins = [b for b in age_bins if any(x in str(b) for x in ['15', '20', '25', '30', '35', '40'])]
    old_bins = [b for b in age_bins if any(x in str(b) for x in ['55', '60', '65', '70', '75'])]

    is_young = adata.obs['Age_group'].isin(young_bins)
    is_old = adata.obs['Age_group'].isin(old_bins)

    print(f"Young (15-40): {is_young.sum()} cells")
    print(f"Old (55-75): {is_old.sum()} cells")
else:
    raise ValueError("Age_group not found")

# Subset to old FAPs for correlation analysis
adata_old_fap = adata[is_fap & is_old].copy()
print(f"Old FAPs for analysis: {adata_old_fap.n_obs} cells")

# Extract expression matrices (use raw counts or normalized)
if adata_old_fap.raw is not None:
    X = adata_old_fap.raw.X.toarray() if hasattr(adata_old_fap.raw.X, 'toarray') else adata_old_fap.raw.X
else:
    X = adata_old_fap.X.toarray() if hasattr(adata_old_fap.X, 'toarray') else adata_old_fap.X

gene_names = adata_old_fap.var_names.tolist()

# Compute gene expression composites
def compute_composite(X, gene_names, gene_list):
    indices = [gene_names.index(g) for g in gene_list if g in gene_names]
    if len(indices) == 0:
        return None
    return np.mean(X[:, indices], axis=1)

collagen_expr = compute_composite(X, gene_names, collagen_present)
fos_expr = compute_composite(X, gene_names, ['FOS'])
junb_expr = compute_composite(X, gene_names, ['JUNB'])

sasp4_expr = compute_composite(X, gene_names, sasp4_present)
sasp12_expr = compute_composite(X, gene_names, sasp12_present)

# Compute correlations
results = {}

# FOS-collagen
if collagen_expr is not None and fos_expr is not None:
    rho, pval = stats.spearmanr(fos_expr, collagen_expr)
    results['fos_collagen_rho'] = float(rho)
    results['fos_collagen_pval'] = float(pval)
    results['fos_collagen_r2'] = float(rho ** 2)
    print(f"FOS-collagen: rho={rho:.3f}, p={pval:.2e}, r2={rho**2:.3f}")

# JUNB-collagen
if collagen_expr is not None and junb_expr is not None:
    rho, pval = stats.spearmanr(junb_expr, collagen_expr)
    results['junb_collagen_rho'] = float(rho)
    results['junb_collagen_pval'] = float(pval)
    results['junb_collagen_r2'] = float(rho ** 2)
    print(f"JUNB-collagen: rho={rho:.3f}, p={pval:.2e}, r2={rho**2:.3f}")

# JUNB-SASP 4-gene
if sasp4_expr is not None and junb_expr is not None:
    rho, pval = stats.spearmanr(junb_expr, sasp4_expr)
    results['junb_sasp4_rho'] = float(rho)
    results['junb_sasp4_pval'] = float(pval)
    results['junb_sasp4_r2'] = float(rho ** 2)
    print(f"JUNB-SASP4: rho={rho:.3f}, p={pval:.2e}, r2={rho**2:.3f}")

# JUNB-SASP 12-gene
if sasp12_expr is not None and junb_expr is not None:
    rho, pval = stats.spearmanr(junb_expr, sasp12_expr)
    results['junb_sasp12_rho'] = float(rho)
    results['junb_sasp12_pval'] = float(pval)
    results['junb_sasp12_r2'] = float(rho ** 2)
    print(f"JUNB-SASP12: rho={rho:.3f}, p={pval:.2e}, r2={rho**2:.3f}")

# Apply FDR correction
correlations = [
    ('fos_collagen', results.get('fos_collagen_pval', 1.0)),
    ('junb_collagen', results.get('junb_collagen_pval', 1.0)),
    ('junb_sasp4', results.get('junb_sasp4_pval', 1.0)),
    ('junb_sasp12', results.get('junb_sasp12_pval', 1.0)),
]

test_names = [x[0] for x in correlations]
pvals = np.array([x[1] for x in correlations])
rejected, fdr_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

for i, name in enumerate(test_names):
    results[f'{name}_fdr'] = float(fdr_corrected[i])
    print(f"{name}: FDR-corrected p = {fdr_corrected[i]:.2e}")

# Store metadata
results['n_old_fap'] = int(adata_old_fap.n_obs)
results['n_young_fap'] = int((is_fap & is_young).sum())
results['collagen_genes_used'] = collagen_present
results['sasp4_genes_used'] = sasp4_present
results['sasp12_genes_used'] = sasp12_present
results['ap1_members_present'] = ap1_present

# Decision rules
results['decision'] = {}
if results.get('fos_collagen_rho', 0) > 0.15:
    results['decision']['fos_collagen_replicated'] = True
    results['decision']['fos_as_fibrosis_driver'] = 'SUPPORTED'
elif results.get('fos_collagen_rho', 0) < 0.05:
    results['decision']['fos_collagen_replicated'] = False
    results['decision']['fos_as_fibrosis_driver'] = 'REFUTED'
else:
    results['decision']['fos_collagen_replicated'] = None
    results['decision']['fos_as_fibrosis_driver'] = 'AMBIGUOUS'

if results.get('junb_sasp12_rho', 0) >= 0.25:
    results['decision']['junb_sasp_expanded'] = 'STRENGTHENED'
elif results.get('junb_sasp12_rho', 0) < 0.20:
    results['decision']['junb_sasp_expanded'] = 'NOT_STRENGTHENED'
else:
    results['decision']['junb_sasp_expanded'] = 'AMBIGUOUS'

# Save results
import json
with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_PATH}")
print(f"\n=== DECISION ===")
print(f"FOS-collagen rho: {results.get('fos_collagen_rho', 'N/A'):.3f} (threshold: > 0.15)")
print(f"JUNB-SASP12 rho: {results.get('junb_sasp12_rho', 'N/A'):.3f} (threshold: >= 0.25)")
print(f"Decision: {results['decision']}")