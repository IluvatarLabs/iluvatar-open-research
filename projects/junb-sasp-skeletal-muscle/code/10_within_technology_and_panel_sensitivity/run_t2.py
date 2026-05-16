#!/usr/bin/env python3
"""
T2 Analysis: Softening the "two independent axes" language in F092.

F092 claimed "two parallel pathways: JNK→JUNB and p21→CDK4/6"
But JUNB and p21 are rho=0.91 (collinear)
Partial r(p21→SASP|JUNB) = 0.42 (p=0.046, marginal)

This script re-examines the evidence and generates revised language.
"""

import json
import numpy as np
from scipy.stats import spearmanr, pearsonr
import scanpy as sc

SASP12 = ["CCL2", "CCL7", "CCL8", "CXCL6", "CXCL8", "IL6", "IL1B", "MMP1", "MMP3", "SERPINE1", "PLAU"]

print("=" * 60)
print("T2: Re-examining F092 'two independent axes' claim")
print("=" * 60)

# Load vascular data
adata = sc.read_h5ad('/home/yuanz/Documents/GitHub/biomarvin_fibro/data/Vascular_scsn_RNA.h5ad')
annotation_col = 'Annotation'
vascular_types = ['CapEC', 'VenEC', 'ArtEC']
mask = adata.obs[annotation_col].isin(vascular_types)
adata_vasc = adata[mask].copy()

donors = adata_vasc.obs['sample'].unique()
var_names = adata_vasc.var_names.tolist()

print(f"\nVascular cells: {adata_vasc.shape[0]} cells, {len(donors)} donors")

# Compute pseudobulk per donor
donor_data = []
for donor in donors:
    donor_mask = adata_vasc.obs['sample'] == donor
    cells = adata_vasc[donor_mask]
    n_cells = donor_mask.sum()

    if n_cells < 10:
        continue

    # JUNB expression
    if 'JUNB' in var_names:
        junb_expr = np.mean(cells[:, 'JUNB'].X.toarray()) if hasattr(cells[:, 'JUNB'].X, 'toarray') else np.mean(cells[:, 'JUNB'].X)
    else:
        continue

    # p21 (CDKN1A) expression
    if 'CDKN1A' in var_names:
        p21_expr = np.mean(cells[:, 'CDKN1A'].X.toarray()) if hasattr(cells[:, 'CDKN1A'].X, 'toarray') else np.mean(cells[:, 'CDKN1A'].X)
    else:
        continue

    # SASP12 composite
    sasp_expr = 0.0
    sasp_genes_found = 0
    for gene in SASP12:
        if gene in var_names:
            expr = cells[:, gene].X.toarray() if hasattr(cells[:, gene].X, 'toarray') else np.array(cells[:, gene].X)
            sasp_expr += np.mean(expr)
            sasp_genes_found += 1

    if sasp_genes_found >= 3:
        sasp_expr /= sasp_genes_found
        donor_data.append({
            'donor': donor,
            'n_cells': n_cells,
            'JUNB': junb_expr,
            'p21': p21_expr,
            'SASP12': sasp_expr
        })

print(f"Donors with adequate data: {len(donor_data)}")

# Extract arrays
junb = np.array([d['JUNB'] for d in donor_data])
p21 = np.array([d['p21'] for d in donor_data])
sasp = np.array([d['SASP12'] for d in donor_data])

# Key correlations
print("\n--- KEY CORRELATIONS ---")
rho_junb_sasp, p_junb_sasp = spearmanr(junb, sasp)
rho_p21_sasp, p_p21_sasp = spearmanr(p21, sasp)
rho_junb_p21, p_junb_p21 = spearmanr(junb, p21)

print(f"rho(JUNB, SASP12) = {rho_junb_sasp:.4f}, p = {p_junb_sasp:.2e}")
print(f"rho(p21, SASP12) = {rho_p21_sasp:.4f}, p = {p_p21_sasp:.2e}")
print(f"rho(JUNB, p21) = {rho_junb_p21:.4f}, p = {p_junb_p21:.2e}")

# Partial correlation: p21→SASP | JUNB
print("\n--- PARTIAL CORRELATION ---")
# Using Pearson on residuals
def partial_corr(x, y, z):
    """Partial correlation: corr(x, y | z)"""
    # Residualize x and y on z
    from scipy.stats import linregress
    slope_xz, intercept_xz, _, _, _ = linregress(z, x)
    resid_x = x - (slope_xz * z + intercept_xz)
    slope_yz, intercept_yz, _, _, _ = linregress(z, y)
    resid_y = y - (slope_yz * z + intercept_yz)
    # Correlation of residuals
    r, p = pearsonr(resid_x, resid_y)
    return r, p

partial_r, partial_p = partial_corr(p21, sasp, junb)
print(f"Partial r(p21→SASP | JUNB) = {partial_r:.4f}, p = {partial_p:.4f}")

# Fisher Z comparison: are JUNB and p21 independently correlated with SASP?
print("\n--- FISHER Z COMPARISON ---")
from scipy.stats import norm

def fisher_z_test(r1, r2, n):
    """Fisher Z test: are two correlations significantly different?"""
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    se = np.sqrt(2 / (n - 3))
    z_diff = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z_diff)))
    return z_diff, p

z_diff, p_fisher = fisher_z_test(rho_junb_sasp, rho_p21_sasp, len(donor_data))
print(f"Fisher Z = {z_diff:.4f}, p = {p_fisher:.4f}")
print(f"Interpretation: {'SAME' if p_fisher > 0.05 else 'DIFFERENT'} direction at alpha=0.05")

# Revised language assessment
print("\n" + "=" * 60)
print("T2 ASSESSMENT: REVISED LANGUAGE FOR PAPER")
print("=" * 60)

print(f"""
ORIGINAL CLAIM (from F092):
"Two mechanistically distinct axes independently drive vascular SASP:
1. JNK→JUNB→AP-1 axis (kinase signaling)
2. p21→CDK4/6 axis (cell cycle checkpoint)"

PROBLEM:
- JUNB and p21 are highly collinear (rho = {rho_junb_p21:.2f})
- Partial r(p21→SASP | JUNB) = {partial_r:.2f} (p = {partial_p:.4f})
- The partial correlation is marginally significant and sensitive to outliers

REVISED CLAIM:
"Vascular SASP is strongly associated with both JUNB (rho={rho_junb_sasp:.2f})
and p21 (rho={rho_p21_sasp:.2f}). These markers are highly collinear (rho={rho_junb_p21:.2f}),
suggesting they capture the same underlying senescence program. Partial correlation
analysis suggests p21 may capture a partially independent SASP component
(partial r={partial_r:.2f}, p={partial_p:.3f}), but this requires
orthogonal validation. Dual targeting of JNK and CDK4/6 pathways may address
overlapping but potentially distinct aspects of vascular senescence."

KEY CHANGES:
1. "Two independent axes" → "strongly associated with both JUNB and p21"
2. "Mechanistically distinct" → acknowledge collinearity explicitly
3. "Dual targeting" → "may address overlapping but potentially distinct components"
4. Add explicit caveat: "requires orthogonal validation"
5. Reference F088 (sex-stratified) showing both axes are not sex-driven
""")

# Save results
results = {
    "analysis": "T2: Softening two-independent-axes language",
    "n_donors": len(donor_data),
    "correlations": {
        "JUNB_SASP12": {"rho": float(rho_junb_sasp), "p": float(p_junb_sasp)},
        "p21_SASP12": {"rho": float(rho_p21_sasp), "p": float(p_p21_sasp)},
        "JUNB_p21": {"rho": float(rho_junb_p21), "p": float(p_junb_p21)}
    },
    "partial_correlation": {
        "p21_to_SASP_given_JUNB": {"r": float(partial_r), "p": float(partial_p)}
    },
    "fisher_z": {
        "z": float(z_diff),
        "p": float(p_fisher)
    },
    "revised_language": {
        "original": "Two mechanistically distinct axes independently drive vascular SASP",
        "revised": "Vascular SASP is strongly associated with both JUNB and p21, which are highly collinear and likely capture the same senescence program"
    }
}

with open('/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_036/t2_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to experiments/batch_036/t2_results.json")
