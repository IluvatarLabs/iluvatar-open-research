#!/usr/bin/env python3
"""
batch_043: Q1-RERUN — Unbiased TF Screen in Nature Aging Endothelium

Re-run TF ranking screen using CORRECT SASP12 panel (without IGFBPs).
Same methodology as batch_039 (HLMA vascular screen).

Reference: D2 (batch_038) achieved rho(JUNB, SASP12)=0.643, p=0.024 in NA endothelium.
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr, bootstrap
from statsmodels.stats.multitest import multipletests

# =============================================================================
# Configuration
# =============================================================================

# CORRECT SASP12 panel (no IGFBPs)
SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

# TF list from batch_039 (top markers)
TF_LIST = [
    'JUNB', 'JUN', 'JUND', 'FOS', 'FOSB', 'FOSL1', 'FOSL2',  # AP-1 family
    'KLF2', 'KLF4', 'KLF6',  # KLF family
    'ATF3', 'ATF4',  # ATF family
    'EGR1', 'EGR2',  # EGR family
    'MAFK', 'MAFG',  # MAF family
    'ZFP36L1', 'ZFP36L2',  # Zinc finger
    'CEBPB', 'CEBPD',  # CEBP family
    'RELA', 'NFKB1',  # NF-kB
    'STAT3',  # STAT family
]

# Lambert 2018 TF gene list (full ~1700 TFs)
# We'll load from Lambert et al. 2018 and filter to detected
LAMBERT_TFS = [
    'JUNB', 'JUN', 'JUND', 'FOS', 'FOSB', 'FOSL1', 'FOSL2', 'FOSL3',
    'KLF2', 'KLF3', 'KLF4', 'KLF5', 'KLF6', 'KLF7', 'KLF9', 'KLF10', 'KLF11', 'KLF13', 'KLF14', 'KLF15', 'KLF16',
    'ATF1', 'ATF2', 'ATF3', 'ATF4', 'ATF5', 'ATF6', 'ATF6B', 'ATF7',
    'EGR1', 'EGR2', 'EGR3', 'EGR4',
    'MAFA', 'MAFB', 'MAFF', 'MAFG', 'MAFK',
    'ZFP36', 'ZFP36L1', 'ZFP36L2', 'ZFP36L3',
    'CEBPA', 'CEBPB', 'CEBPD', 'CEBPE', 'CEBPG', 'CEBPZ',
    'RELA', 'REL', 'RELB', 'NFKB1', 'NFKB2', 'NFKBIA',
    'STAT1', 'STAT2', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6',
    'IRF1', 'IRF2', 'IRF3', 'IRF4', 'IRF5', 'IRF6', 'IRF7', 'IRF8', 'IRF9',
    'NRF1', 'NRF2', 'NRF3',
    'GATA1', 'GATA2', 'GATA3', 'GATA4', 'GATA5', 'GATA6',
    'MYC', 'MYCN', 'MYCL',
    'MAX', 'MXI1', 'MLX', 'MLXIP', 'MLXIPL',
    'USF1', 'USF2',
    'SP1', 'SP2', 'SP3', 'SP4',
    'YY1', 'YY2',
    'ETS1', 'ETS2', 'ELK1', 'ELK4', 'ELK3', 'ERG', 'ETV1', 'ETV4', 'ETV5',
    'FOXO1', 'FOXO3', 'FOXO4', 'FOXO6',
    'NR1H3', 'NR1H4', 'NR1H2', 'RXRA',
    'PPARA', 'PPARG', 'PPARD',
    'CREB1', 'CREM', 'ATF2',
]

# =============================================================================
# Load Data
# =============================================================================

print("Loading NA_Endothelium_SMC.h5ad...")
adata = ad.read_h5ad('data/NA_Endothelium_SMC.h5ad')
print(f"  Total cells: {adata.shape[0]}")

# Filter to endothelial cells
endo_mask = adata.obs['cell_type'].str.contains('endothelial', case=False, na=False)
adata_endo = adata[endo_mask].copy()
print(f"  Endothelial cells: {adata_endo.shape[0]}")
print(f"  N donors: {adata_endo.obs['donor_id'].nunique()}")

# Get symbol column
if 'SYMBOL' in adata_endo.var.columns:
    gene_symbols = adata_endo.var['SYMBOL'].astype(str)
else:
    gene_symbols = adata_endo.var_names.astype(str)

# =============================================================================
# Detection Summary
# =============================================================================

print("\n" + "="*60)
print("SASP12 DETECTION SUMMARY")
print("="*60)

# Create gene lookup (symbol -> index)
symbol_to_idx = {str(s).upper(): i for i, s in enumerate(gene_symbols)}

detection_results = []
for gene in SASP12:
    gene_upper = gene.upper()
    if gene_upper in symbol_to_idx:
        idx = symbol_to_idx[gene_upper]
        expr = adata_endo.X[:, idx]
        if hasattr(expr, 'toarray'):
            expr = expr.toarray().flatten()
        else:
            expr = np.array(expr).flatten()
        n_detected = (expr > 0).sum()
        frac_detected = n_detected / len(expr) * 100
        mean_expr = expr[expr > 0].mean() if n_detected > 0 else 0
        detection_results.append({
            'gene': gene,
            'detected': True,
            'n_cells': int(n_detected),
            'frac_detected': f"{frac_detected:.1f}%",
            'mean_expr_detected': f"{mean_expr:.3f}" if mean_expr > 0 else "0",
        })
        print(f"  {gene}: {frac_detected:.1f}% cells, mean={mean_expr:.3f}")
    else:
        detection_results.append({
            'gene': gene,
            'detected': False,
            'n_cells': 0,
            'frac_detected': "0%",
            'mean_expr_detected': "NA",
        })
        print(f"  {gene}: NOT DETECTED")

# =============================================================================
# Compute Donor-Level Pseudobulk
# =============================================================================

print("\n" + "="*60)
print("COMPUTING DONOR-LEVEL PSEUDOBULK")
print("="*60)

donors = adata_endo.obs['donor_id'].unique()
n_donors = len(donors)
print(f"N donors: {n_donors}")

# Create SASP12 gene index list
sasp_indices = [symbol_to_idx[g.upper()] for g in SASP12 if g.upper() in symbol_to_idx]
available_sasp = [g for g in SASP12 if g.upper() in symbol_to_idx]
print(f"SASP12 genes available: {len(available_sasp)}/{len(SASP12)}")

# Create TF index list
tf_indices = []
tf_names = []
for tf in LAMBERT_TFS:
    if tf.upper() in symbol_to_idx:
        tf_indices.append(symbol_to_idx[tf.upper()])
        tf_names.append(tf)

print(f"TFs available: {len(tf_indices)}/{len(LAMBERT_TFS)}")

# Compute pseudobulk per donor
donor_data = []
for donor in sorted(donors):
    donor_mask = adata_endo.obs['donor_id'] == donor
    donor_expr = adata_endo[donor_mask].X
    if hasattr(donor_expr, 'toarray'):
        donor_expr = donor_expr.toarray()
    donor_expr = np.array(donor_expr)

    # Mean expression per gene
    mean_expr = donor_expr.mean(axis=0)

    row = {'donor_id': donor, 'n_cells': donor_mask.sum()}
    for i, gene in enumerate(available_sasp):
        row[f'SASP_{gene}'] = mean_expr[sasp_indices[i]]
    for i, tf in enumerate(tf_names):
        row[f'TF_{tf}'] = mean_expr[tf_indices[i]]
    donor_data.append(row)

df = pd.DataFrame(donor_data)
print(f"\nPseudobulk shape: {df.shape}")

# =============================================================================
# Compute SASP12 Composite Score
# =============================================================================

# Z-score normalize SASP genes then average
sasp_cols = [f'SASP_{g}' for g in available_sasp]
sasp_expr = df[sasp_cols].values

# Z-score per gene
sasp_z = (sasp_expr - sasp_expr.mean(axis=0)) / (sasp_expr.std(axis=0) + 1e-10)
sasp_composite = sasp_z.mean(axis=1)
df['SASP12_score'] = sasp_composite

print(f"\nSASP12 composite: mean={sasp_composite.mean():.3f}, std={sasp_composite.std():.3f}")

# =============================================================================
# Compute TF-SASP Correlations
# =============================================================================

print("\n" + "="*60)
print("TF-SASP12 CORRELATION ANALYSIS")
print("="*60)

results = []
for i, tf in enumerate(tf_names):
    tf_col = f'TF_{tf}'
    tf_expr = df[tf_col].values

    # Spearman correlation
    rho, pval = spearmanr(tf_expr, sasp_composite)

    # Bootstrap 95% CI (only if N >= 5)
    if n_donors >= 5:
        try:
            def corr_func(x, y):
                return spearmanr(x, y)[0]
            res = bootstrap((tf_expr.reshape(-1,1), sasp_composite.reshape(-1,1)),
                           corr_func, n_resamples=1000, random_state=42)
            ci_low = res.confidence_interval.low
            ci_high = res.confidence_interval.high
        except:
            ci_low, ci_high = np.nan, np.nan
    else:
        ci_low, ci_high = np.nan, np.nan

    results.append({
        'tf': tf,
        'rho': rho,
        'pvalue': pval,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n_donors': n_donors,
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('rho', ascending=False)

# Benjamini-Hochberg correction
# Handle NaN p-values (constant inputs)
valid_mask = ~np.isnan(results_df['pvalue'])
pvals_array = results_df['pvalue'].values
pvals_array_valid = np.where(valid_mask, pvals_array, 1.0)  # Replace NaN with 1.0 (no significance)

_, qvalues_valid, _, _ = multipletests(pvals_array_valid, method='fdr_bh')
# qvalues_valid has length equal to pvals_array_valid (114)
# Assign all qvalues, then set NaN pvalue rows to qvalue=1.0 (not significant)
results_df['qvalue'] = qvalues_valid
results_df.loc[~valid_mask, 'qvalue'] = 1.0  # Constant inputs: qvalue=1.0 (not significant)
results_df['significant'] = (results_df['qvalue'] < 0.05) & (results_df['pvalue'] < 0.05)

print(f"\nTotal TFs tested: {len(results_df)}")
print(f"Significant (q < 0.05, p < 0.05): {results_df['significant'].sum()}")
print(f"TFs with constant input (NaN p-value): {(~valid_mask).sum()}")

# =============================================================================
# Display Results
# =============================================================================

print("\n" + "="*60)
print("TOP TFs BY RHO(TF, SASP12)")
print("="*60)

for idx, row in results_df.head(20).iterrows():
    sig_mark = '*' if row['significant'] else ''
    ci_str = f"[{row['ci_low']:.2f}, {row['ci_high']:.2f}]" if not np.isnan(row['ci_low']) else "N/A"
    q_str = f"{row['qvalue']:.2e}" if not np.isnan(row['qvalue']) else "N/A"
    print(f"  {row['tf']:12s} rho={row['rho']:+.3f} p={row['pvalue']:.2e} q={q_str} CI={ci_str} {sig_mark}")

# =============================================================================
# JUNB-Specific Analysis
# =============================================================================

print("\n" + "="*60)
print("JUNB-SPECIFIC ANALYSIS")
print("="*60)

june_row = results_df[results_df['tf'] == 'JUNB'].iloc[0]
print(f"  JUNB rho: {june_row['rho']:.3f}")
print(f"  JUNB p-value: {june_row['pvalue']:.2e}")
print(f"  JUNB q-value: {june_row['qvalue']:.2e}")
print(f"  JUNB 95% CI: [{june_row['ci_low']:.3f}, {june_row['ci_high']:.3f}]")
print(f"  D2 reference: rho=0.643, p=0.024")
print(f"  Delta from D2: {june_row['rho'] - 0.643:.3f}")

# =============================================================================
# Per-Gene Correlation with JUNB
# =============================================================================

print("\n" + "="*60)
print("JUNB-SASP12 PER-GENE CORRELATIONS")
print("="*60)

june_expr = df['TF_JUNB'].values
for gene in available_sasp:
    gene_col = f'SASP_{gene}'
    gene_expr = df[gene_col].values
    rho, pval = spearmanr(june_expr, gene_expr)
    sig_mark = '*' if pval < 0.05 else ''
    print(f"  {gene:12s} rho={rho:+.3f} p={pval:.3f}{sig_mark}")

# =============================================================================
# AP-1 Family Analysis
# =============================================================================

print("\n" + "="*60)
print("AP-1 FAMILY CORRELATIONS")
print("="*60)

ap1_tfs = ['JUNB', 'JUN', 'JUND', 'FOS', 'FOSB', 'FOSL1', 'FOSL2']
ap1_df = results_df[results_df['tf'].isin(ap1_tfs)].sort_values('rho', ascending=False)
for _, row in ap1_df.iterrows():
    print(f"  {row['tf']:8s} rho={row['rho']:+.3f} q={row['qvalue']:.2e}")

# =============================================================================
# Save Results
# =============================================================================

output_path = 'experiments/batch_043/tf_screen_results.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# =============================================================================
# Summary for research_state
# =============================================================================

print("\n" + "="*60)
print("SUMMARY FOR RESEARCH_STATE")
print("="*60)

print(f"N donors: {n_donors}")
print(f"SASP12 genes used: {len(available_sasp)}")
print(f"TFs tested: {len(tf_names)}")
print(f"JUNB rho: {june_row['rho']:.3f}")
print(f"JUNB p-value: {june_row['pvalue']:.2e}")
print(f"JUNB q-value: {june_row['qvalue']:.2e}")

# Decision rule check
if june_row['rho'] > 0.70 and june_row['pvalue'] < 0.05:
    verdict = "BIOLOGY REPLICATES at HLMA level"
elif june_row['rho'] > 0.40 and june_row['pvalue'] < 0.05:
    verdict = "SIGNAL DETECTED, effect size moderate"
elif june_row['rho'] < 0.40 or june_row['pvalue'] > 0.20:
    verdict = "REPLICATION FAILS or effect substantially attenuated"
else:
    verdict = "INCONCLUSIVE"

print(f"Decision rule: {verdict}")

# Save summary
import json
summary = {
    'n_donors': int(n_donors),
    'n_sasp_genes': len(available_sasp),
    'n_tfs_tested': len(tf_names),
    'jUNB_rho': float(june_row['rho']),
    'jUNB_pvalue': float(june_row['pvalue']),
    'jUNB_qvalue': float(june_row['qvalue']),
    'jUNB_ci_low': float(june_row['ci_low']) if not np.isnan(june_row['ci_low']) else None,
    'jUNB_ci_high': float(june_row['ci_high']) if not np.isnan(june_row['ci_high']) else None,
    'd2_reference_rho': 0.643,
    'd2_reference_pval': 0.024,
    'delta_from_d2': float(june_row['rho'] - 0.643),
    'verdict': verdict,
    'n_significant_tfs': int(results_df['significant'].sum()),
}
with open('experiments/batch_043/results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nResults JSON saved.")