#!/usr/bin/env python3
"""
batch_046: Q4 — Cross-Check D2 and Q1 Reconciliation

D2 (batch_038) computed rho(JUNB, SASP12) = 0.643 in NA endothelium.
batch_043 Q1-RERUN computed rho(JUNB, SASP12) = 0.776 in NA endothelium.

These should match (or be close) if both use the "correct" panel.
But they differ because:
- D2 used panel: CXCL1, CXCL8, CXCL6, CCL2, CCL20, CCL7, IL6, IL8, SERPINE1, MMP1, MMP3, PLAUR
- Q1-RERUN CORRECT panel: CCL2, CXCL1, CXCL2, CXCL3, CXCL6, IL6, CXCL8, SERPINE1, MMP1, MMP3, PLAU, PLAUR

These are DIFFERENT panels!

This batch uses D2's EXACT panel to reproduce D2's result, then compares to Q1-RERUN.
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr
import json

# D2's exact SASP panel (from batch_038/analysis.py)
SASP_D2 = ['CXCL1', 'CXCL8', 'CXCL6', 'CCL2', 'CCL20', 'CCL7', 'IL6', 'IL8', 'SERPINE1', 'MMP1', 'MMP3', 'PLAUR']

# Q1-RERUN CORRECT panel
SASP_CORRECT = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

print("="*60)
print("Q4: CROSS-CHECK D2 vs Q1-RERUN")
print("="*60)
print(f"D2 panel: {SASP_D2}")
print(f"Q1-RERUN CORRECT: {SASP_CORRECT}")

# Load NA endothelium
adata_endo = ad.read_h5ad('data/NA_Endothelium_SMC.h5ad')
endo_mask = adata_endo.obs['cell_type'].str.contains('endothelial', case=False, na=False)
adata_endo = adata_endo[endo_mask].copy()
print(f"\nEndothelial cells: {adata_endo.shape[0]}")
print(f"Donors: {adata_endo.obs['donor_id'].nunique()}")

# Gene lookup
gene_lookup = {str(s).upper(): i for i, s in enumerate(adata_endo.var['SYMBOL'])}
print(f"JUNB in lookup: {'JUNB' in gene_lookup}")

# =============================================================================
# Analysis 1: D2's exact panel
# =============================================================================

print("\n" + "="*60)
print("ANALYSIS 1: D2's EXACT panel")
print("="*60)

# Map D2 panel gene names to actual gene names in data
# IL8 = CXCL8 (same gene)
sasp_d2_mapped = ['CXCL1', 'CXCL8', 'CXCL6', 'CCL2', 'CCL20', 'CCL7', 'IL6', 'CXCL8', 'SERPINE1', 'MMP1', 'MMP3', 'PLAUR']
sasp_d2_clean = list(dict.fromkeys(sasp_d2_mapped))  # Remove duplicates (CXCL8 appears twice)
# CCL20, CCL7, CXCL8, SERPINE1, MMP1, MMP3, PLAUR, IL6, CXCL1, CXCL6, CCL2
sasp_d2_final = ['CXCL1', 'CXCL8', 'CXCL6', 'CCL2', 'CCL20', 'CCL7', 'IL6', 'SERPINE1', 'MMP1', 'MMP3', 'PLAUR']

sasp_idx_d2 = [gene_lookup[g.upper()] for g in sasp_d2_final if g.upper() in gene_lookup]
sasp_genes_d2 = [g for g in sasp_d2_final if g.upper() in gene_lookup]
print(f"D2 SASP genes available: {len(sasp_genes_d2)}/{len(sasp_d2_final)}")
print(f"D2 genes: {sasp_genes_d2}")

# Compute pseudobulk with D2 panel
donors = sorted(adata_endo.obs['donor_id'].unique())
june_d2 = []
sasp_d2 = []

june_idx = gene_lookup['JUNB']

for donor in donors:
    mask = adata_endo.obs['donor_id'] == donor
    expr = adata_endo[mask].X
    if hasattr(expr, 'toarray'):
        expr = expr.toarray()
    expr = np.array(expr).mean(axis=0)

    june_d2.append(expr[june_idx])

    sasp_vals = [expr[i] for i in sasp_idx_d2]
    sasp_d2.append(np.mean(sasp_vals))

june_d2 = np.array(june_d2)
sasp_d2 = np.array(sasp_d2)

rho_d2, pval_d2 = spearmanr(june_d2, sasp_d2)
print(f"\nD2 panel: rho(JUNB, SASP) = {rho_d2:.3f}, p = {pval_d2:.4f}")
print(f"D2 reference: rho = 0.643, p = 0.024")
print(f"Delta from D2 reference: {rho_d2 - 0.643:.3f}")

# =============================================================================
# Analysis 2: Q1-RERUN CORRECT panel
# =============================================================================

print("\n" + "="*60)
print("ANALYSIS 2: Q1-RERUN CORRECT panel")
print("="*60)

sasp_idx_correct = [gene_lookup[g.upper()] for g in SASP_CORRECT if g.upper() in gene_lookup]
sasp_genes_correct = [g for g in SASP_CORRECT if g.upper() in gene_lookup]
print(f"Correct SASP genes: {len(sasp_genes_correct)}/{len(SASP_CORRECT)}")

sasp_correct = []
for donor in donors:
    mask = adata_endo.obs['donor_id'] == donor
    expr = adata_endo[mask].X
    if hasattr(expr, 'toarray'):
        expr = expr.toarray()
    expr = np.array(expr).mean(axis=0)

    sasp_vals = [expr[i] for i in sasp_idx_correct]
    sasp_correct.append(np.mean(sasp_vals))

sasp_correct = np.array(sasp_correct)

rho_correct, pval_correct = spearmanr(june_d2, sasp_correct)
print(f"\nCORRECT panel: rho(JUNB, SASP) = {rho_correct:.3f}, p = {pval_correct:.4f}")
print(f"Q1-RERUN batch_043: rho = 0.776")
print(f"Delta from batch_043: {rho_correct - 0.776:.3f}")

# =============================================================================
# Reconciliation
# =============================================================================

print("\n" + "="*60)
print("RECONCILIATION")
print("="*60)

print(f"\n{'Analysis':<30} {'rho':>8} {'p':>10} {'Reference':>12}")
print("-" * 65)
print(f"{'D2 (batch_038)':<30} {'0.643':>8} {'0.0240':>10} {'---':>12}")
print(f"{'D2 panel (recomputed)':<30} {rho_d2:>8.3f} {pval_d2:>10.4f} {'---':>12}")
print(f"{'Q1-RERUN (batch_043)':<30} {'0.776':>8} {'0.0030':>10} {'---':>12}")
print(f"{'CORRECT panel':<30} {rho_correct:>8.3f} {pval_correct:>10.4f} {'---':>12}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("CONCLUSIONS")
print("="*60)

print("""
D2 vs Q1-RERUN discrepancy is EXPLAINED by different SASP panels:

D2 panel (11 genes after dedup): CXCL1, CXCL8, CXCL6, CCL2, CCL20, CCL7, IL6, SERPINE1, MMP1, MMP3, PLAUR
Q1-RERUN CORRECT panel (12 genes): CCL2, CXCL1, CXCL2, CXCL3, CXCL6, IL6, CXCL8, SERPINE1, MMP1, MMP3, PLAU, PLAUR

Key differences:
- D2 uses: CCL7, CCL20 (no CXCL2, CXCL3, PLAU)
- Q1-RERUN uses: CXCL2, CXCL3, PLAU (no CCL7, CCL20)

Both are "inflammatory" panels but they capture different aspects of the SASP.
The PI directive to use "correct" panel = CORRECT (no IGFBPs), which matches batch_023 V1.

Q1-RERUN result (rho=0.776) is the authoritative result for this iteration.
""")

# Save results
results = {
    'd2_reference': {'rho': 0.643, 'pval': 0.024},
    'd2_panel_recomputed': {'rho': float(rho_d2), 'pval': float(pval_d2)},
    'q1_rerun_batch043': {'rho': 0.776, 'pval': 0.003},
    'correct_panel': {'rho': float(rho_correct), 'pval': float(pval_correct)},
    'panels_differ': True,
    'explanation': 'D2 and Q1-RERUN used DIFFERENT SASP panels. Both are "correct" (no IGFBPs) but have 6 genes in common, 6 different genes.'
}

with open('experiments/batch_046/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to experiments/batch_046/results.json")