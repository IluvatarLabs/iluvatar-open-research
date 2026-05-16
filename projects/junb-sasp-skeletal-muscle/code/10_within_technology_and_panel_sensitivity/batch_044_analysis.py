#!/usr/bin/env python3
"""
batch_044: Q2-RERUN — FAP Per-Gene with CORRECT SASP12 Panel

Verify FAP per-gene correlations with the CORRECT panel (no IGFBPs).
Q2-RERUN: Verify/re-run FAP per-gene decomposition with correct panel.

P2 finding from batch_040 (mixed panel):
- JUNB negative to CCL7 (rho=-0.74), IL6 (rho=-0.48), MMP1 (rho=-0.51)
- JUNB positive to IGFBP3 (rho=+0.47), IGFBP5 (rho=+0.48)

With CORRECT panel (no IGFBPs):
- Negative coupling to inflammatory genes should persist
- Positive coupling to IGFBPs will be absent (IGFBPs not in panel)
- Per-gene analysis will show which inflammatory genes are specifically affected
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr

# =============================================================================
# Configuration
# =============================================================================

# CORRECT SASP12 panel (no IGFBPs)
SASP12_CORRECT = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

# Mixed panel (for comparison)
SASP12_MIXED = ['CCL2', 'CCL7', 'CXCL8', 'IL6', 'SERPINE1', 'MMP1', 'MMP3', 'IGFBP2', 'IGFBP3', 'IGFBP5', 'CXCL6', 'CCL20']

# =============================================================================
# Load Data
# =============================================================================

print("Loading OMIX004308-02.h5ad (FAP compartment)...")
adata = ad.read_h5ad('data/OMIX004308-02.h5ad')
print(f"  Total cells: {adata.shape[0]}")

# All cells are FAPs based on prior analysis
print(f"  Total donors: {adata.obs['sample'].nunique()}")

# =============================================================================
# Compute Donor-Level Pseudobulk
# =============================================================================

print("\n" + "="*60)
print("COMPUTING DONOR-LEVEL PSEUDOBULK")
print("="*60)

donors = adata.obs['sample'].unique()
n_donors = len(donors)
print(f"N donors: {n_donors}")

# Get gene indices
available_genes = {}
for gene in SASP12_CORRECT + ['JUNB', 'FOS', 'FOSB']:
    if gene in adata.var_names:
        available_genes[gene] = np.where(adata.var_names == gene)[0][0]
    else:
        print(f"  WARNING: {gene} not in var_names")

print(f"Available genes: {list(available_genes.keys())}")

# Compute pseudobulk per donor
donor_data = []
for donor in sorted(donors):
    donor_mask = adata.obs['sample'] == donor
    donor_expr = adata[donor_mask].X
    if hasattr(donor_expr, 'toarray'):
        donor_expr = donor_expr.toarray()
    donor_expr = np.array(donor_expr)

    # Mean expression per gene
    mean_expr = donor_expr.mean(axis=0)

    row = {'donor_id': donor, 'n_cells': donor_mask.sum()}
    for gene, idx in available_genes.items():
        row[f'expr_{gene}'] = mean_expr[idx]
    donor_data.append(row)

df = pd.DataFrame(donor_data)
print(f"\nPseudobulk shape: {df.shape}")

# =============================================================================
# Compute SASP12 Composite Score (Correct Panel)
# =============================================================================

sasp_cols = [f'expr_{g}' for g in SASP12_CORRECT if f'expr_{g}' in df.columns]
sasp_expr = df[sasp_cols].values

# Z-score normalize SASP genes then average
sasp_z = (sasp_expr - sasp_expr.mean(axis=0)) / (sasp_expr.std(axis=0) + 1e-10)
sasp_composite_correct = sasp_z.mean(axis=1)
df['SASP12_correct'] = sasp_composite_correct

print(f"\nSASP12 (correct) composite: mean={sasp_composite_correct.mean():.3f}, std={sasp_composite_correct.std():.3f}")

# =============================================================================
# Compute SASP12 Composite Score (Mixed Panel) for comparison
# =============================================================================

mixed_cols = [f'expr_{g}' for g in SASP12_MIXED if f'expr_{g}' in df.columns]
mixed_expr = df[mixed_cols].values

# Z-score normalize SASP genes then average
mixed_z = (mixed_expr - mixed_expr.mean(axis=0)) / (mixed_expr.std(axis=0) + 1e-10)
sasp_composite_mixed = mixed_z.mean(axis=1)
df['SASP12_mixed'] = sasp_composite_mixed

print(f"SASP12 (mixed) composite: mean={sasp_composite_mixed.mean():.3f}, std={sasp_composite_mixed.std():.3f}")

# =============================================================================
# JUNB-SASP12 Correlations (both panels)
# =============================================================================

print("\n" + "="*60)
print("JUNB-SASP12 CORRELATIONS")
print("="*60)

june_expr = df['expr_JUNB'].values

# Correct panel
rho_correct, pval_correct = spearmanr(june_expr, sasp_composite_correct)
print(f"\nCORRECT panel (no IGFBPs):")
print(f"  rho(JUNB, SASP12) = {rho_correct:.3f}, p = {pval_correct:.3f}")

# Mixed panel
rho_mixed, pval_mixed = spearmanr(june_expr, sasp_composite_mixed)
print(f"\nMIXED panel (with IGFBPs):")
print(f"  rho(JUNB, SASP12) = {rho_mixed:.3f}, p = {pval_mixed:.3f}")

# =============================================================================
# Per-Gene Correlations with JUNB
# =============================================================================

print("\n" + "="*60)
print("JUNB-PER-GENE CORRELATIONS (Correct Panel)")
print("="*60)

per_gene_results = []
for gene in SASP12_CORRECT:
    if f'expr_{gene}' in df.columns:
        gene_expr = df[f'expr_{gene}'].values
        rho, pval = spearmanr(june_expr, gene_expr)
        sig_mark = '*' if pval < 0.05 else ''
        print(f"  {gene:12s} rho={rho:+.3f} p={pval:.3f}{sig_mark}")
        per_gene_results.append({
            'gene': gene,
            'rho': rho,
            'pvalue': pval,
            'direction': 'positive' if rho > 0.2 else ('negative' if rho < -0.2 else 'flat'),
        })

# =============================================================================
# Comparison: Correct vs Mixed Panel
# =============================================================================

print("\n" + "="*60)
print("CORRECT vs MIXED PANEL COMPARISON")
print("="*60)

# Genes in both panels
common_genes = [g for g in SASP12_CORRECT if g in SASP12_MIXED]
print(f"\nCommon genes: {common_genes}")

# Genes unique to mixed (IGFBPs)
mixed_only = [g for g in SASP12_MIXED if g not in SASP12_CORRECT]
print(f"Mixed-only genes (IGFBPs): {mixed_only}")

# =============================================================================
# P2 Verification
# =============================================================================

print("\n" + "="*60)
print("P2 VERIFICATION (batch_040 findings)")
print("="*60)

# batch_040 found: JUNB negative to CCL7 (rho=-0.74), IL6 (rho=-0.48), MMP1 (rho=-0.51)
# With correct panel (no CCL7), check IL6 and MMP1

p2_genes = ['IL6', 'MMP1']
print("\nP2 genes in FAPs (correct panel):")
for gene in p2_genes:
    if f'expr_{gene}' in df.columns:
        gene_expr = df[f'expr_{gene}'].values
        rho, pval = spearmanr(june_expr, gene_expr)
        sig_mark = '*' if pval < 0.05 else ''
        expected_sign = 'negative' if gene in ['IL6', 'MMP1'] else 'positive'
        match = '✓' if (rho < 0 and expected_sign == 'negative') or (rho > 0 and expected_sign == 'positive') else '✗'
        print(f"  {gene}: rho={rho:+.3f} p={pval:.3f}{sig_mark} (expected: {expected_sign}) {match}")

# =============================================================================
# Save Results
# =============================================================================

results = {
    'n_donors': int(n_donors),
    'jUNB_rho_correct': float(rho_correct),
    'jUNB_pval_correct': float(pval_correct),
    'jUNB_rho_mixed': float(rho_mixed),
    'jUNB_pval_mixed': float(pval_mixed),
    'per_gene': per_gene_results,
}

import json
with open('experiments/batch_044/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save per-gene CSV
df_pg = pd.DataFrame(per_gene_results)
df_pg.to_csv('experiments/batch_044/per_gene_results.csv', index=False)

print("\nResults saved to experiments/batch_044/")

# =============================================================================
# Decision Rule
# =============================================================================

print("\n" + "="*60)
print("DECISION RULE")
print("="*60)

print(f"\nCorrect panel rho(JUNB, SASP12) = {rho_correct:.3f}")
if rho_correct < -0.3:
    print("  → Negative coupling confirmed (P2 finding holds for correct panel)")
elif rho_correct > 0.3:
    print("  → Positive coupling (unexpected for correct panel)")
else:
    print("  → Flat/no coupling (IGFBPs were driving mixed panel correlation)")

print(f"\nMixed panel rho(JUNB, SASP12) = {rho_mixed:.3f}")
print("  → For comparison with batch_040 result")