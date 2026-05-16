"""
batch_013: MuSC Compartment Aging Analysis
==========================================
Test whether DDR/NF-κB/JUNB pattern (established in aged FAPs) extends to aged MuSCs.

Design: Within-compartment analysis only (cross-compartment comparison dropped per design review)
- Primary: Confirm DDR/NF-κB/JUNB pattern within MuSCs
- Secondary: Technology-stratified analysis (scRNA vs snRNA)

Author: Marvin (autonomous research agent)
Date: 2026-04-09
"""

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import json
from pathlib import Path

print("=" * 60)
print("batch_013: MuSC Compartment Aging Analysis")
print("=" * 60)

# =============================================================================
# Step 1: Load and Stratify MuSC Data
# =============================================================================
print("\n[1/5] Loading MuSC data...")

musc = ad.read_h5ad('data/MuSC_scsn_RNA.h5ad')
print(f"Total MuSCs: {musc.shape[0]:,} cells × {musc.shape[1]:,} genes")

# Check annotations
print(f"\nMuSC Annotations:")
print(musc.obs['Annotation'].value_counts())

print(f"\nAge distribution:")
print(musc.obs['age'].value_counts().sort_index())

print(f"\nTechnology distribution:")
print(musc.obs['tech'].value_counts())

# =============================================================================
# Step 2: Stratify by Age and Technology
# =============================================================================
print("\n[2/5] Stratifying by age and technology...")

# Define young (< 40) vs old (> 65)
young_mask = musc.obs['age'] < 40
old_mask = musc.obs['age'] > 65

musc.young = musc[young_mask].copy()
musc.old = musc[old_mask].copy()

young_donors = musc.obs.loc[young_mask, 'sample'].nunique()
old_donors = musc.obs.loc[old_mask, 'sample'].nunique()

print(f"\nYoung (age < 40): {musc.young.shape[0]:,} cells from {young_donors} donors")
print(f"Old (age > 65): {musc.old.shape[0]:,} cells from {old_donors} donors")

# Stratify by technology
musc.young_scRNA = musc.young[musc.young.obs['tech'] == 'scRNA'].copy()
musc.young_snRNA = musc.young[musc.young.obs['tech'] == 'snRNA'].copy()
musc.old_scRNA = musc.old[musc.old.obs['tech'] == 'scRNA'].copy()
musc.old_snRNA = musc.old[musc.old.obs['tech'] == 'snRNA'].copy()

print(f"\nYoung scRNA: {musc.young_scRNA.shape[0]:,} cells")
print(f"Young snRNA: {musc.young_snRNA.shape[0]:,} cells")
print(f"Old scRNA: {musc.old_scRNA.shape[0]:,} cells")
print(f"Old snRNA: {musc.old_snRNA.shape[0]:,} cells")

# =============================================================================
# Step 3: Compute Pathway Scores
# =============================================================================
print("\n[3/5] Computing pathway scores...")

# Gene sets (from batch_012 FAP analysis)
DDR_GENES = ['GADD45A', 'BTG2', 'CDKN1A', 'MDM2', 'GADD45B']
NFKB_GENES = ['NFKB1', 'NFKBIA', 'NFKBIZ', 'RELA', 'RELB']
AP1_GENES = ['JUNB', 'JUN', 'FOS']

# Filter to genes present in dataset
musc_genes = musc.var_names.tolist()
ddr_genes = [g for g in DDR_GENES if g in musc_genes]
nfkb_genes = [g for g in NFKB_GENES if g in musc_genes]
ap1_genes = [g for g in AP1_GENES if g in musc_genes]

print(f"DDR genes found: {ddr_genes}")
print(f"NF-κB genes found: {nfkb_genes}")
print(f"AP-1 genes found: {ap1_genes}")

def compute_pathway_score(adata, genes):
    """Compute mean expression of pathway genes."""
    genes_present = [g for g in genes if g in adata.var_names]
    if len(genes_present) == 0:
        return np.zeros(adata.n_obs)
    return np.mean(adata[:, genes_present].X.toarray() if hasattr(adata[:, genes_present].X, 'toarray') else adata[:, genes_present].X, axis=1)

def cohen_d(group1, group2):
    """Compute Cohen's d between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan
    return (mean2 - mean1) / pooled_sd

# =============================================================================
# Step 4: Compute Age Effects (All + by Technology)
# =============================================================================
print("\n[4/5] Computing age effects...")

def compute_age_effects(young_data, old_data, label):
    """Compute Cohen d for age effect on pathway scores and AP-1 genes."""
    results = {}

    # Pathway scores
    young_ddr = compute_pathway_score(young_data, ddr_genes)
    old_ddr = compute_pathway_score(old_data, ddr_genes)
    results['DDR_score_d'] = cohen_d(young_ddr, old_ddr)

    young_nfkb = compute_pathway_score(young_data, nfkb_genes)
    old_nfkb = compute_pathway_score(old_data, nfkb_genes)
    results['NFkB_score_d'] = cohen_d(young_nfkb, old_nfkb)

    # AP-1 genes
    for gene in ap1_genes:
        if gene in young_data.var_names:
            young_expr = young_data[:, gene].X.toarray().flatten() if hasattr(young_data[:, gene].X, 'toarray') else young_data[:, gene].X.toarray().flatten()
            old_expr = old_data[:, gene].X.toarray().flatten() if hasattr(old_data[:, gene].X, 'toarray') else old_data[:, gene].X.toarray().flatten()
            results[f'{gene}_d'] = cohen_d(young_expr, old_expr)
            results[f'{gene}_young_mean'] = float(np.mean(young_expr))
            results[f'{gene}_old_mean'] = float(np.mean(old_expr))
            results[f'{gene}_young_det'] = float(np.mean(young_expr > 0) * 100)
            results[f'{gene}_old_det'] = float(np.mean(old_expr > 0) * 100)

    results['N_young'] = int(young_data.n_obs)
    results['N_old'] = int(old_data.n_obs)

    return results

# Compute for all MuSCs
results_all = compute_age_effects(musc.young, musc.old, "All MuSCs")
print(f"\nAll MuSCs (N_young={results_all['N_young']:,}, N_old={results_all['N_old']:,}):")
for key, val in results_all.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.3f}")
    else:
        print(f"  {key}: {val}")

# Compute for scRNA only
results_scrna = compute_age_effects(musc.young_scRNA, musc.old_scRNA, "scRNA MuSCs")
print(f"\nscRNA MuSCs (N_young={results_scrna['N_young']:,}, N_old={results_scrna['N_old']:,}):")
for key, val in results_scrna.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.3f}")
    else:
        print(f"  {key}: {val}")

# Compute for snRNA only
results_snRNA = compute_age_effects(musc.young_snRNA, musc.old_snRNA, "snRNA MuSCs")
print(f"\nsnRNA MuSCs (N_young={results_snRNA['N_young']:,}, N_old={results_snRNA['N_old']:,}):")
for key, val in results_snRNA.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.3f}")
    else:
        print(f"  {key}: {val}")

# =============================================================================
# Step 5: JUNB Correlations in Aged Cells
# =============================================================================
print("\n[5/5] Computing JUNB correlations in aged MuSCs...")

def compute_correlations(old_data, label):
    """Compute Spearman correlations of JUNB with DDR and NF-κB scores in aged cells."""
    results = {}

    # Get JUNB expression
    if 'JUNB' not in old_data.var_names:
        print(f"  JUNB not detected in {label}")
        return results

    junb_expr = old_data[:, 'JUNB'].X.toarray().flatten() if hasattr(old_data[:, 'JUNB'].X, 'toarray') else old_data[:, 'JUNB'].X.flatten()

    # DDR score
    ddr_score = compute_pathway_score(old_data, ddr_genes)
    valid = ~(np.isnan(junb_expr) | np.isnan(ddr_score) | (junb_expr == 0))
    if valid.sum() > 30:
        rho_ddr, p_ddr = spearmanr(junb_expr[valid], ddr_score[valid])
        results['JUNB_DDR_rho'] = float(rho_ddr)
        results['JUNB_DDR_p'] = float(p_ddr)
        print(f"  JUNB-DDR: rho={rho_ddr:.3f}, p={p_ddr:.2e}, N={valid.sum():,}")

    # NF-κB score
    nfkb_score = compute_pathway_score(old_data, nfkb_genes)
    valid = ~(np.isnan(junb_expr) | np.isnan(nfkb_score) | (junb_expr == 0))
    if valid.sum() > 30:
        rho_nfkb, p_nfkb = spearmanr(junb_expr[valid], nfkb_score[valid])
        results['JUNB_NFkB_rho'] = float(rho_nfkb)
        results['JUNB_NFkB_p'] = float(p_nfkb)
        print(f"  JUNB-NF-κB: rho={rho_nfkb:.3f}, p={p_nfkb:.2e}, N={valid.sum():,}")

    # JUN expression
    if 'JUN' in old_data.var_names:
        jun_expr = old_data[:, 'JUN'].X.toarray().flatten() if hasattr(old_data[:, 'JUN'].X, 'toarray') else old_data[:, 'JUN'].X.flatten()
        valid = ~(np.isnan(junb_expr) | np.isnan(jun_expr) | (junb_expr == 0) | (jun_expr == 0))
        if valid.sum() > 30:
            rho_jun, p_jun = spearmanr(junb_expr[valid], jun_expr[valid])
            results['JUNB_JUN_rho'] = float(rho_jun)
            results['JUNB_JUN_p'] = float(p_jun)
            print(f"  JUNB-JUN: rho={rho_jun:.3f}, p={p_jun:.2e}, N={valid.sum():,}")

    results['N_cells'] = int(old_data.n_obs)

    return results

# Correlations in all aged MuSCs
print("\nAged MuSCs (all):")
corr_all = compute_correlations(musc.old, "All aged MuSCs")

print("\nAged MuSCs (scRNA):")
corr_scrna = compute_correlations(musc.old_scRNA, "Aged scRNA MuSCs")

print("\nAged MuSCs (snRNA):")
corr_snRNA = compute_correlations(musc.old_snRNA, "Aged snRNA MuSCs")

# =============================================================================
# Step 6: MuSC Subtype Analysis
# =============================================================================
print("\n[Bonus] MuSC Subtype Analysis...")

subtypes = ['Quiescent MuSC', 'Early Primed MuSC', 'Late Primed MuSC', 'Diff.MuSC']
subtype_results = {}

for subtype in subtypes:
    young_sub = musc.young[musc.young.obs['Annotation'] == subtype]
    old_sub = musc.old[musc.old.obs['Annotation'] == subtype]

    if young_sub.n_obs < 50 or old_sub.n_obs < 50:
        continue

    print(f"\n{subtype}:")
    results = compute_age_effects(young_sub, old_sub, subtype)
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.3f}")
    subtype_results[subtype] = results

# =============================================================================
# Summary and Comparison with FAPs
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY: MuSC vs FAP Comparison")
print("=" * 60)

# FAP findings from batch_012
fap_ddr_rho = 0.249  # HLMA FAPs
fap_nfkb_rho = 0.198  # HLMA FAPs
fap_junb_d = 0.556  # HLMA FAPs (snRNA)

print(f"\nFAP Findings (batch_012):")
print(f"  JUNB age effect (d): {fap_junb_d:.3f}")
print(f"  JUNB-DDR correlation: {fap_ddr_rho:.3f}")
print(f"  JUNB-NF-κB correlation: {fap_nfkb_rho:.3f}")

print(f"\nMuSC Findings (batch_013):")
print(f"  JUNB age effect (d): {results_all.get('JUNB_d', 'N/A'):.3f}")
print(f"  JUNB-DDR correlation: {corr_all.get('JUNB_DDR_rho', 'N/A'):.3f}")
print(f"  JUNB-NF-κB correlation: {corr_all.get('JUNB_NFkB_rho', 'N/A'):.3f}")

# Technology-matched comparison note
print(f"\nNOTE: Cross-compartment comparison (FAP vs MuSC) is INVALID due to technology mismatch.")
print(f"  FAPs: predominantly snRNA-seq")
print(f"  MuSCs: predominantly scRNA-seq (70%)")

# =============================================================================
# Save Results
# =============================================================================
results = {
    "experiment": "batch_013",
    "date": "2026-04-09",
    "description": "MuSC compartment aging: DDR/NF-κB/JUNB pattern confirmation",
    "age_effects": {
        "all_musc": results_all,
        "scrna_musc": results_scrna,
        "snrna_musc": results_snRNA
    },
    "correlations": {
        "aged_all_musc": corr_all,
        "aged_scrna_musc": corr_scrna,
        "aged_snrna_musc": corr_snRNA
    },
    "subtype_effects": subtype_results,
    "sample_sizes": {
        "young_musc_total": int(musc.young.n_obs),
        "old_musc_total": int(musc.old.n_obs),
        "young_donors": int(young_donors),
        "old_donors": int(old_donors),
        "scrna_old": int(musc.old_scRNA.n_obs),
        "snrna_old": int(musc.old_snRNA.n_obs)
    },
    "gene_sets": {
        "DDR": ddr_genes,
        "NFkB": nfkb_genes,
        "AP1": ap1_genes
    },
    "design_notes": {
        "cross_compartment_comparison": "DROPPED due to technology mismatch (scRNA vs snRNA)",
        "within_compartment_analysis": "PRIMARY analysis - valid",
        "technology_stratification": "scRNA vs snRNA separated"
    }
}

output_path = Path('experiments/batch_013/results.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

print("\n" + "=" * 60)
print("Analysis complete")
print("=" * 60)
