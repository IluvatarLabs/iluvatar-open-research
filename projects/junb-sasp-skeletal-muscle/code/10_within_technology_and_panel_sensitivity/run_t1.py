#!/usr/bin/env python3
"""
T1 Analysis: Within-Technology Correlation of F084

Compute vascular JUNB-SASP donor-level correlation using ONLY old donors
(same technology: snRNA from snRNA-seq). This directly tests whether F084
(rho=0.93) is confounded by technology differences between young (scRNA) and
old (snRNA) donors.

T1 from research_state.md: "MUST compute, CRITICAL"
"""

import json
import numpy as np
from scipy.stats import spearmanr
import scanpy as sc

SASP12 = ["CCL2", "CCL7", "CCL8", "CXCL6", "CXCL8", "IL6", "IL1B", "MMP1", "MMP3", "SERPINE1", "PLAU"]

print("=" * 60)
print("T1: Within-Technology Correlation Analysis")
print("=" * 60)

# Load vascular data
adata = sc.read_h5ad('/home/yuanz/Documents/GitHub/biomarvin_fibro/data/Vascular_scsn_RNA.h5ad')

# Filter to CapEC + VenEC + ArtEC (as in batch_023 V1)
annotation_col = 'Annotation'
vascular_types = ['CapEC', 'VenEC', 'ArtEC']
mask = adata.obs[annotation_col].isin(vascular_types)
adata_vasc = adata[mask].copy()

print(f"\nVascular cells: {adata_vasc.shape[0]} cells")

# Get donor information
donors = adata_vasc.obs['sample'].unique()
var_names = adata_vasc.var_names.tolist()

# Separate old and young donors
# OM prefix = old male, P prefix = old (other), YM prefix = young male
old_donors = [d for d in donors if d.startswith('OM') or d.startswith('P')]
young_donors = [d for d in donors if d.startswith('YM')]

print(f"Old donors: {len(old_donors)} ({old_donors})")
print(f"Young donors: {len(young_donors)} ({young_donors})")

# Check technology distribution
print("\n--- Technology Distribution ---")
tech_by_donor = adata_vasc.obs.groupby('sample')['tech'].first()
print("Old donors tech:", tech_by_donor[old_donors].value_counts().to_dict())
print("Young donors tech:", tech_by_donor[young_donors].value_counts().to_dict())

# Compute pseudobulk per donor
def compute_pseudobulk(adata, donors, gene_names):
    """Compute pseudobulk means per donor for given genes."""
    results = []
    for donor in donors:
        mask = adata.obs['sample'] == donor
        cells = adata[mask]
        n_cells = mask.sum()

        if n_cells < 10:
            continue

        gene_exprs = {}
        for gene in gene_names:
            if gene in var_names:
                expr = cells[:, gene].X.toarray() if hasattr(cells[:, gene].X, 'toarray') else np.array(cells[:, gene].X)
                gene_exprs[gene] = np.mean(expr)

        results.append({
            'donor': donor,
            'n_cells': n_cells,
            **gene_exprs
        })

    return results

# Compute for all relevant genes
gene_names = ['JUNB', 'CDKN1A'] + SASP12

# Old donors
old_data = compute_pseudobulk(adata_vasc, old_donors, gene_names)
# Young donors
young_data = compute_pseudobulk(adata_vasc, young_donors, gene_names)
# All donors
all_data = old_data + young_data

print(f"\nOld donors with adequate data: {len(old_data)}")
print(f"Young donors with adequate data: {len(young_data)}")
print(f"Total donors: {len(all_data)}")

# Extract arrays for correlations
def extract_arrays(data):
    junb = np.array([d['JUNB'] for d in data])
    sasp_vals = []
    for d in data:
        sasp_expr = 0.0
        n_genes = 0
        for gene in SASP12:
            if gene in d:
                sasp_expr += d[gene]
                n_genes += 1
        if n_genes >= 3:
            sasp_expr /= n_genes
        sasp_vals.append(sasp_expr)
    sasp = np.array(sasp_vals)
    return junb, sasp

# Compute correlations
print("\n" + "=" * 60)
print("CORRELATION RESULTS")
print("=" * 60)

results = {}

for cohort_name, data in [("old", old_data), ("young", young_data), ("all", all_data)]:
    if len(data) < 5:
        print(f"\n{cohort_name.upper()}: Insufficient donors (N={len(data)})")
        continue

    junb, sasp = extract_arrays(data)
    rho, pval = spearmanr(junb, sasp)

    # Get donor list
    donors_list = [d['donor'] for d in data]

    results[cohort_name] = {
        "n_donors": len(data),
        "rho": float(rho),
        "p_value": float(pval),
        "donors": donors_list
    }

    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"\n{cohort_name.upper()} (N={len(data)}): rho={rho:.4f}, p={pval:.4e} {sig}")

# Compare to F088 estimates
print("\n" + "=" * 60)
print("COMPARISON TO F088 ESTIMATES")
print("=" * 60)

# F088 estimates from batch_034
f088 = {
    "old_male": {"rho": 0.8893, "n": 7},
    "old_female": {"rho": 0.8531, "n": 8}
}

# Pooled estimate (weighted by N)
from scipy.stats import norm

def fisher_z_pooled(rhos, ns):
    """Pool correlations using Fisher Z transformation."""
    zs = []
    weights = []
    for rho, n in zip(rhos, ns):
        z = 0.5 * np.log((1 + rho) / (1 - rho))
        zs.append(z)
        weights.append(n - 3)  # Weight by df

    # Weighted mean
    z_pooled = np.average(zs, weights=weights)
    se_pooled = np.sqrt(1 / sum(weights))

    # Convert back to rho
    rho_pooled = (np.exp(2 * z_pooled) - 1) / (np.exp(2 * z_pooled) + 1)

    # 95% CI
    z_lower = z_pooled - 1.96 * se_pooled
    z_upper = z_pooled + 1.96 * se_pooled
    rho_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    rho_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    return rho_pooled, rho_lower, rho_upper

f088_rhos = [f088['old_male']['rho'], f088['old_female']['rho']]
f088_ns = [f088['old_male']['n'], f088['old_female']['n']]
f088_pooled, f088_ci_lower, f088_ci_upper = fisher_z_pooled(f088_rhos, f088_ns)

print(f"F088 pooled estimate (Fisher Z): rho={f088_pooled:.3f}, 95% CI [{f088_ci_lower:.3f}, {f088_ci_upper:.3f}]")
print(f"  old_male: rho={f088['old_male']['rho']:.3f} (N={f088['old_male']['n']})")
print(f"  old_female: rho={f088['old_female']['rho']:.3f} (N={f088['old_female']['n']})")

# Decision rule assessment
print("\n" + "=" * 60)
print("T1 DECISION RULE ASSESSMENT")
print("=" * 60)

if 'old' in results:
    rho_old = results['old']['rho']
    p_old = results['old']['p_value']
    n_old = results['old']['n_donors']

    print(f"\nWithin-old (N={n_old}): rho={rho_old:.4f}, p={p_old:.4e}")
    print(f"F088 pooled estimate: rho={f088_pooled:.3f}")

    if rho_old >= 0.80 and p_old < 0.05:
        print("\n>>> SUPPORTS F084: Within-technology correlation is strong (>= 0.80)")
        print("    F084 is robust to technology confound")
        decision = "SUPPORTS"
    elif rho_old >= 0.50 and p_old < 0.05:
        print("\n>>> PARTIAL SUPPORT: Correlation exists but attenuated (0.50-0.79)")
        print("    Technology may partially inflate cross-age correlation")
        decision = "PARTIAL_SUPPORT"
    else:
        print(f"\n>>> CONCERN: Correlation weak or non-significant (rho={rho_old:.4f}, p={p_old:.4e})")
        print("    Technology confound may be substantial")
        decision = "CONCERN"

    # Compare to F088
    delta = rho_old - f088_pooled
    print(f"\nComparison to F088 pooled: delta={delta:+.3f}")

    if abs(delta) < 0.10:
        print("  -> Within-old estimate is consistent with F088")
    elif abs(delta) < 0.20:
        print("  -> Within-old estimate is moderately different from F088")
    else:
        print("  -> Within-old estimate is substantially different from F088")

    results['old']['decision'] = decision
    results['old']['comparison_to_f088'] = {
        'f088_pooled': f088_pooled,
        'delta': float(delta)
    }

# Add 95% CI for within-old rho
if 'old' in results and results['old']['n_donors'] >= 5:
    n = results['old']['n_donors']
    r = results['old']['rho']
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    z_lower = z - 1.96 * se
    z_upper = z + 1.96 * se
    r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

    results['old']['rho_95ci'] = [float(r_lower), float(r_upper)]
    print(f"\n95% CI for within-old rho: [{r_lower:.3f}, {r_upper:.3f}]")

# Save results
output_path = '/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_036/t1_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
