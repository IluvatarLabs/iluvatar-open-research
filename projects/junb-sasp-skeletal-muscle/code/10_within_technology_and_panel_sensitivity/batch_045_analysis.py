#!/usr/bin/env python3
"""
batch_045: Q3-RERUN — Permutation Test on Actual Donor Labels

Run 10,000 permutations on ACTUAL donor labels for:
1. F084 (HLMA vascular rho=0.929) - uses CORRECT SASP12 panel (batch_023 methodology)
2. F093 (HLMA MuSC rho=0.941) - uses CORRECT SASP12 panel
3. batch_043 JUNB-SASP12 (NA endothelium rho=0.776) - uses CORRECT SASP12 panel

CRITICAL: batch_023 V1 used CORRECT SASP12 panel:
['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

All three analyses use the SAME CORRECT panel (no IGFBPs).
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr
import json

np.random.seed(42)
N_PERMUTATIONS = 10000

# CORRECT SASP12 panel (matching batch_023 V1 methodology)
SASP_CORRECT = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

def permutation_test(x, y, n_perms=10000):
    """Permutation test for Spearman correlation."""
    obs_rho, obs_pval = spearmanr(x, y)
    null_rhos = []
    x_arr = np.array(x)
    y_arr = np.array(y)

    for _ in range(n_perms):
        y_perm = np.random.permutation(y_arr)
        null_rho, _ = spearmanr(x_arr, y_perm)
        null_rhos.append(null_rho)

    null_rhos = np.array(null_rhos)
    p_empirical = np.mean(np.abs(null_rhos) >= np.abs(obs_rho))

    return {
        'obs_rho': obs_rho,
        'obs_pval': obs_pval,
        'p_empirical': p_empirical,
        'null_mean': np.mean(null_rhos),
        'null_std': np.std(null_rhos),
        'null_95': np.percentile(null_rhos, 95),
    }

# =============================================================================
# Analysis 1: F084 (HLMA Vascular JUNB-SASP12) - CORRECT PANEL
# =============================================================================

print("="*60)
print("ANALYSIS 1: F084 (HLMA Vascular JUNB-SASP12)")
print("="*60)
print("Prior: rho=0.929, p=1.65×10⁻¹⁰ (CORRECT panel, batch_023 V1)")

# Load HLMA vascular data
adata_vas = ad.read_h5ad('data/Vascular_scsn_RNA.h5ad')

# Filter to endothelial cells (matching batch_023: CapEC, VenEC, ArtEC)
ec_types = ['CapEC', 'VenEC', 'ArtEC']
ec_mask = adata_vas.obs['Annotation'].isin(ec_types)
adata_ec = adata_vas[ec_mask].copy()

print(f"  Endothelial cells: {adata_ec.shape[0]}")
print(f"  Donors: {adata_ec.obs['sample'].nunique()}")

# Use CORRECT panel
sasp_cols_vas = [g for g in SASP_CORRECT if g in adata_ec.var_names]
print(f"  SASP genes (correct): {len(sasp_cols_vas)}/{len(SASP_CORRECT)}")

# Get expression data - MATCH batch_023 exactly
X_df = adata_ec.to_df()

# Compute per-cell SASP12 score (matching batch_023: mean across SASP genes)
X_df['SASP12'] = X_df[sasp_cols_vas].mean(axis=1)

# Add sample info
X_df['sample'] = adata_ec.obs['sample'].values

# Compute donor-level means (matching batch_023: mean of per-cell scores)
donor_means = X_df.groupby('sample').agg({
    'JUNB': 'mean',
    'SASP12': 'mean'
}).reset_index()

june_vas = donor_means['JUNB'].values
sasp_vas = donor_means['SASP12'].values

print(f"  Testing: {len(donor_means)} donors")

result_f084 = permutation_test(june_vas, sasp_vas, N_PERMUTATIONS)
print(f"\n  Observed: rho={result_f084['obs_rho']:.3f}, p={result_f084['obs_pval']:.2e}")
print(f"  Empirical p-value: {result_f084['p_empirical']:.6f}")
print(f"  Null distribution: mean={result_f084['null_mean']:.3f}, std={result_f084['null_std']:.3f}")
print(f"  Null 95th percentile: {result_f084['null_95']:.3f}")
sig_f084 = result_f084['p_empirical'] < 0.05
print(f"  SIGNIFICANT: {'YES' if sig_f084 else 'NO'}")

# =============================================================================
# Analysis 2: F093 (HLMA MuSC p21-SASP12) - CORRECT PANEL
# =============================================================================

print("\n" + "="*60)
print("ANALYSIS 2: F093 (HLMA MuSC p21-SASP12)")
print("="*60)
print("Prior: rho=0.941, p=5.7×10⁻⁸ (CORRECT panel)")

adata_musc = ad.read_h5ad('data/MuSC_scsn_RNA.h5ad')
print(f"  Cells: {adata_musc.shape[0]}, Donors: {adata_musc.obs['sample'].nunique()}")

sasp_cols_musc = [g for g in SASP_CORRECT if g in adata_musc.var_names]
print(f"  SASP genes (correct): {len(sasp_cols_musc)}/{len(SASP_CORRECT)}")

# Get expression data - matching F084 methodology
X_df_musc = adata_musc.to_df()
X_df_musc['SASP12'] = X_df_musc[sasp_cols_musc].mean(axis=1)
X_df_musc['sample'] = adata_musc.obs['sample'].values

# Donor-level means
donor_means_musc = X_df_musc.groupby('sample').agg({
    'CDKN1A': 'mean',
    'SASP12': 'mean'
}).reset_index()

p21_musc = donor_means_musc['CDKN1A'].values
sasp_musc = donor_means_musc['SASP12'].values

print(f"  Testing: {len(donor_means_musc)} donors")

result_f093 = permutation_test(p21_musc, sasp_musc, N_PERMUTATIONS)
print(f"\n  Observed: rho={result_f093['obs_rho']:.3f}, p={result_f093['obs_pval']:.2e}")
print(f"  Empirical p-value: {result_f093['p_empirical']:.6f}")
print(f"  Null 95th percentile: {result_f093['null_95']:.3f}")
sig_f093 = result_f093['p_empirical'] < 0.05
print(f"  SIGNIFICANT: {'YES' if sig_f093 else 'NO'}")

# =============================================================================
# Analysis 3: batch_043 JUNB-SASP12 (NA Endothelium) - CORRECT PANEL
# =============================================================================

print("\n" + "="*60)
print("ANALYSIS 3: batch_043 JUNB-SASP12 (NA Endothelium)")
print("="*60)
print("Prior: rho=0.776, p=2.99×10⁻³ (CORRECT panel)")

adata_endo = ad.read_h5ad('data/NA_Endothelium_SMC.h5ad')
endo_mask = adata_endo.obs['cell_type'].str.contains('endothelial', case=False, na=False)
adata_endo = adata_endo[endo_mask].copy()
print(f"  Endothelial cells: {adata_endo.shape[0]}, Donors: {adata_endo.obs['donor_id'].nunique()}")

# Use SYMBOL column for gene lookup
if 'SYMBOL' in adata_endo.var.columns:
    gene_lookup = {str(s).upper(): i for i, s in enumerate(adata_endo.var['SYMBOL'])}
else:
    gene_lookup = {str(s).upper(): i for i, s in enumerate(adata_endo.var_names)}

sasp_idx_endo = [gene_lookup[g.upper()] for g in SASP_CORRECT if g.upper() in gene_lookup]
print(f"  SASP genes (correct): {len(sasp_idx_endo)}/{len(SASP_CORRECT)}")

donors_endo = sorted(adata_endo.obs['donor_id'].unique())
june_endo = []
sasp_endo = []

june_idx = gene_lookup.get('JUNB')

for donor in donors_endo:
    mask = adata_endo.obs['donor_id'] == donor
    expr = adata_endo[mask].X
    if hasattr(expr, 'toarray'):
        expr = expr.toarray()
    expr = np.array(expr).mean(axis=0)

    june_endo.append(expr[june_idx])

    sasp_vals = [expr[i] for i in sasp_idx_endo]
    sasp_z = (sasp_vals - np.mean(sasp_vals)) / (np.std(sasp_vals) + 1e-10)
    sasp_endo.append(np.mean(sasp_z))

june_endo = np.array(june_endo)
sasp_endo = np.array(sasp_endo)

print(f"  Testing: {len(donors_endo)} donors")

result_junb_endo = permutation_test(june_endo, sasp_endo, N_PERMUTATIONS)
print(f"\n  Observed: rho={result_junb_endo['obs_rho']:.3f}, p={result_junb_endo['obs_pval']:.2e}")
print(f"  Empirical p-value: {result_junb_endo['p_empirical']:.6f}")
print(f"  Null 95th percentile: {result_junb_endo['null_95']:.3f}")
sig_junb_endo = result_junb_endo['p_empirical'] < 0.05
print(f"  SIGNIFICANT: {'YES' if sig_junb_endo else 'NO'}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*60)
print("SUMMARY: PERMUTATION TEST RESULTS (10,000 permutations)")
print("="*60)

print(f"\n{'Finding':<45} {'Prior':>7} {'Obs':>7} {'p-empirical':>12} {'Sig':>5}")
print("-" * 80)
results_list = [
    ("F084 HLMA Vascular JUNB-SASP (correct)", 0.929, result_f084, sig_f084),
    ("F093 HLMA MuSC p21-SASP (correct)", 0.941, result_f093, sig_f093),
    ("batch_043 NA Endo JUNB-SASP (correct)", 0.776, result_junb_endo, sig_junb_endo),
]

for name, prior_rho, result, sig in results_list:
    sig_str = '***' if result['p_empirical'] < 0.001 else ('**' if result['p_empirical'] < 0.01 else ('*' if result['p_empirical'] < 0.05 else ''))
    print(f"{name:<45} {prior_rho:>7.3f} {result['obs_rho']:>+7.3f} {result['p_empirical']:>12.6f} {sig_str:>5}")

# Save results
output = {
    'f084': {'finding': 'F084 HLMA Vascular JUNB-SASP', 'prior_rho': 0.929, 'obs_rho': float(result_f084['obs_rho']), 'p_empirical': float(result_f084['p_empirical']), 'significant': bool(sig_f084)},
    'f093': {'finding': 'F093 HLMA MuSC p21-SASP', 'prior_rho': 0.941, 'obs_rho': float(result_f093['obs_rho']), 'p_empirical': float(result_f093['p_empirical']), 'significant': bool(sig_f093)},
    'batch_043_junb_endo': {'finding': 'batch_043 NA Endothelium JUNB-SASP', 'prior_rho': 0.776, 'obs_rho': float(result_junb_endo['obs_rho']), 'p_empirical': float(result_junb_endo['p_empirical']), 'significant': bool(sig_junb_endo)},
    'n_permutations': N_PERMUTATIONS,
    'panel': 'correct (no IGFBPs)',
    'methodology_match': 'All three use CORRECT SASP12 panel (matching batch_023 V1)',
}

with open('experiments/batch_045/results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n\nAll findings use CORRECT SASP12 panel (matching batch_023 V1 methodology).")