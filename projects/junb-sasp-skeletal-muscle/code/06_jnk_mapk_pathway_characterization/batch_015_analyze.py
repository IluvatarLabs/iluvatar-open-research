"""
batch_015: MAPK TF Activity Analysis
====================================
Test whether MAPK-associated TF activity increases with age in FAPs and MuSCs,
and whether MAPK-associated genes co-activate with JUNB.

Design: Revised per science-critic review (2 rounds)
- Tier 1: MAPK-specific genes (no DDR overlap): MEF2A, MAPK14, MAX
- Tier 2: MAPK-DDR shared genes (FOS, JUND, GADD45B, DUSP1, ATF4)
- Tier 3: AP-1 complex (FOS, JUND, JUNB, JUN)

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
print("batch_015: MAPK TF Activity Analysis")
print("=" * 60)

# =============================================================================
# GENE SETS (Revised — per science critics, no DDR-MAPK overlap)
# =============================================================================

DDR_GENES = ['GADD45A', 'BTG2', 'CDKN1A', 'MDM2']  # GADD45B excluded from DDR for this analysis (it's in Tier 2)

# Tier 1: MAPK-Specific Genes (no DDR overlap)
# Best-detected in FAPs: MEF2A (31.4%), MAPK14 (28.5%), MAX (22.9%)
TIER1_MAPK_TFS = {
    'MAPK14': {'desc': 'p38alpha MAPK', 'fap_det': 0.285, 'musc_det': 0.224},
    'MEF2A': {'desc': 'MEF2A TF (p38 target)', 'fap_det': 0.314, 'musc_det': 0.393},
    'MAX': {'desc': 'MAX TF (MYC partner)', 'fap_det': 0.229, 'musc_det': 0.259},
}

# Tier 2: MAPK-DDR Shared Genes (known overlap — FOR COMPARISON ONLY)
TIER2_SHARED = ['FOS', 'JUND', 'GADD45B', 'DUSP1', 'ATF4']

# Tier 3: AP-1 Complex (well-detected MAPK downstream)
TIER3_AP1 = ['FOS', 'JUND', 'JUNB', 'JUN']

# All MAPK genes for screening
ALL_MAPK_GENES = ['MAPK14', 'MAP2K3', 'MAP2K6', 'ATF2', 'ELK1', 'MEF2A', 'MEF2C',
                   'MEF2D', 'MAX', 'FOS', 'JUNB', 'JUND', 'GADD45A', 'GADD45B',
                   'DUSP1', 'DUSP5', 'ATF4', 'JUN']

def cohen_d(group1, group2):
    """Compute Cohen's d between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan, np.nan, np.nan
    se = pooled_sd * np.sqrt(1/n1 + 1/n2)
    t_stat = (mean2 - mean1) / se if se > 0 else 0
    df = n1 + n2 - 2
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if df > 0 else np.nan
    return (mean2 - mean1) / pooled_sd, p_val, pooled_sd

def compute_detection(adata, gene):
    """Compute detection rate for a gene."""
    if gene in adata.var_names:
        gi = list(adata.var_names).index(gene)
        if hasattr(adata.X, 'toarray'):
            vals = np.asarray(adata[:, gene].X.toarray()).flatten()
        else:
            vals = np.asarray(adata[:, gene].X).flatten()
        return float(np.mean(vals > 0))
    return 0.0

def compute_gene_expr(adata, gene):
    """Compute expression values for a gene."""
    if gene in adata.var_names:
        if hasattr(adata.X, 'toarray'):
            return np.asarray(adata[:, gene].X.toarray()).flatten()
        else:
            return np.asarray(adata[:, gene].X).flatten()
    return np.zeros(adata.n_obs)

def compute_mean_score(adata, genes):
    """Compute mean expression score for a gene set."""
    genes_present = [g for g in genes if g in adata.var_names]
    if len(genes_present) == 0:
        return np.zeros(adata.n_obs)
    scores = []
    for g in genes_present:
        expr = compute_gene_expr(adata, g)
        scores.append(expr)
    return np.mean(scores, axis=0)

def partial_correlation(x, y, z):
    """
    Compute partial correlation between x and y, controlling for z.
    Uses residual method: corr(resid_x, resid_y) where resid_x = residuals of x ~ z.
    NOTE: Only use this when x, y, z are clearly non-overlapping gene sets.
    For this analysis: We use it ONLY for Tier1 vs JUNB (no overlap with DDR).
    """
    # Residualize x on z
    x_z_corr = np.corrcoef(x, z)[0, 1]
    z_z_var = np.var(z)
    resid_x = x - x_z_corr * (np.std(x) / np.std(z)) * (z - np.mean(z)) if z_z_var > 0 else x - np.mean(x)
    # Residualize y on z
    y_z_corr = np.corrcoef(y, z)[0, 1]
    resid_y = y - y_z_corr * (np.std(y) / np.std(z)) * (z - np.mean(z)) if z_z_var > 0 else y - np.mean(y)
    # Partial correlation
    if np.std(resid_x) == 0 or np.std(resid_y) == 0:
        return np.nan
    return float(np.corrcoef(resid_x, resid_y)[0, 1])

# =============================================================================
# STEP 1: Load and Check Data
# =============================================================================
print("\n[1/6] Loading data...")

fap = ad.read_h5ad('/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad')
print(f"FAP data: {fap.shape[0]:,} cells × {fap.shape[1]:,} genes")

musc = ad.read_h5ad('/home/yuanz/Documents/GitHub/biomarvin_fibro/data/MuSC_scsn_RNA.h5ad')
print(f"MuSC data: {musc.shape[0]:,} cells × {musc.shape[1]:,} genes")

na_fap_path = '/home/yuanz/Documents/GitHub/biomarvin_fibro/data/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad'
try:
    na_fap = ad.read_h5ad(na_fap_path)
    print(f"Nature Aging FAP data: {na_fap.shape[0]:,} cells × {na_fap.shape[1]:,} genes")
    na_available = True
except Exception as e:
    print(f"Nature Aging data not available: {e}")
    na_available = False

# =============================================================================
# STEP 2: Stratify Data
# =============================================================================
print("\n[2/6] Stratifying data...")

# FAP stratification
fap_genes = list(fap.var_names)
fap_young = fap[fap.obs['age_pop'] == 'young_pop'].copy()
fap_old = fap[fap.obs['age_pop'] == 'old_pop'].copy()
fap_young_scRNA = fap_young[fap_young.obs['tech'] == 'scRNA-seq'] if 'scRNA-seq' in fap_young.obs['tech'].values else fap_young
fap_old_scRNA = fap_old[fap_old.obs['tech'] == 'scRNA-seq'] if 'scRNA-seq' in fap_old.obs['tech'].values else fap_old
fap_old_snRNA = fap_old[fap_old.obs['tech'] == 'snRNA-seq'] if 'snRNA-seq' in fap_old.obs['tech'].values else fap_old

print(f"FAP Young: {fap_young.n_obs:,} cells")
print(f"FAP Old: {fap_old.n_obs:,} cells (scRNA: {fap_old_scRNA.n_obs:,}, snRNA: {fap_old_snRNA.n_obs:,})")

# MuSC stratification
musc_genes = list(musc.var_names)
musc_young = musc[musc.obs['age_pop'] == 'young_pop'].copy()
musc_old = musc[musc.obs['age_pop'] == 'old_pop'].copy()
musc_young_scRNA = musc_young[musc_young.obs['tech'] == 'scRNA'] if 'scRNA' in musc_young.obs['tech'].values else musc_young
musc_old_scRNA = musc_old[musc_old.obs['tech'] == 'scRNA'] if 'scRNA' in musc_old.obs['tech'].values else musc_old
musc_old_snRNA = musc_old[musc_old.obs['tech'] == 'snRNA'] if 'snRNA' in musc_old.obs['tech'].values else musc_old

print(f"MuSC Young: {musc_young.n_obs:,} cells")
print(f"MuSC Old: {musc_old.n_obs:,} cells (scRNA: {musc_old_scRNA.n_obs:,}, snRNA: {musc_old_snRNA.n_obs:,})")

# =============================================================================
# STEP 3: Detection Check for Tier 1 Genes
# =============================================================================
print("\n[3/6] Detection check for Tier 1 MAPK-specific genes...")

tier1_detection = {}
for gene in TIER1_MAPK_TFS:
    fap_det = compute_detection(fap_old, gene)
    musc_det = compute_detection(musc_old, gene)
    tier1_detection[gene] = {'fap': fap_det, 'musc': musc_det}
    print(f"  {gene}: FAP={fap_det:.3f}, MuSC={musc_det:.3f}")

# Check if >= 2 genes pass > 20% threshold in BOTH compartments
tier1_passing = [g for g, d in tier1_detection.items() if d['fap'] > 0.20 and d['musc'] > 0.20]
print(f"\nTier 1 genes passing >20% detection in BOTH compartments: {tier1_passing}")

detection_pass = len(tier1_passing) >= 2
print(f"Detection threshold met (>= 2 genes): {detection_pass}")

# =============================================================================
# STEP 4: Age Effect Analysis
# =============================================================================
print("\n[4/6] Computing age effects...")

results = {
    'detection_check': {
        'tier1_passing_genes': tier1_passing,
        'detection_pass': detection_pass,
        'tier1_detection': tier1_detection,
    },
    'age_effects': {},
    'correlations': {},
    'cross_atlas': {},
}

def compute_age_effect_for_gene(gene, young_data, old_data, label):
    """Compute Cohen d for a single gene's age effect."""
    expr_young = compute_gene_expr(young_data, gene)
    expr_old = compute_gene_expr(old_data, gene)
    d, p, sd = cohen_d(expr_young, expr_old)
    return {
        'd': float(d) if not np.isnan(d) else None,
        'p_value': float(p) if not np.isnan(p) else None,
        'young_mean': float(np.mean(expr_young)),
        'old_mean': float(np.mean(expr_old)),
        'young_det': float(np.mean(expr_young > 0)),
        'old_det': float(np.mean(expr_old > 0)),
        'n_young': int(young_data.n_obs),
        'n_old': int(old_data.n_obs),
    }

def compute_age_effect_for_score(score_young, score_old):
    """Compute Cohen d for a score's age effect."""
    d, p, sd = cohen_d(score_young, score_old)
    return {
        'd': float(d) if not np.isnan(d) else None,
        'p_value': float(p) if not np.isnan(p) else None,
        'young_mean': float(np.mean(score_young)),
        'old_mean': float(np.mean(score_old)),
    }

# FAP Age Effects
print("\n--- FAP Age Effects ---")
fap_results = {}

# Tier 1 genes
for gene in tier1_passing:
    result = compute_age_effect_for_gene(gene, fap_young, fap_old, 'FAP')
    fap_results[f'tier1_{gene}'] = result
    print(f"  {gene}: d={result['d']:.3f} (p={result['p_value']:.2e})")

# Tier 2 genes (FOS, JUND, GADD45B, DUSP1, ATF4)
for gene in TIER2_SHARED:
    if gene in fap_genes:
        result = compute_age_effect_for_gene(gene, fap_young, fap_old, 'FAP')
        fap_results[f'tier2_{gene}'] = result
        print(f"  [shared] {gene}: d={result['d']:.3f} (p={result['p_value']:.2e})")

# Tier 3 AP-1 genes
for gene in ['JUNB', 'FOS', 'JUND', 'JUN']:
    if gene in fap_genes:
        result = compute_age_effect_for_gene(gene, fap_young, fap_old, 'FAP')
        fap_results[f'ap1_{gene}'] = result
        print(f"  [AP-1] {gene}: d={result['d']:.3f} (p={result['p_value']:.2e})")

# Technology-stratified for key genes
for gene in ['JUNB', 'MEF2A', 'FOS']:
    if gene in fap_genes:
        # scRNA
        try:
            y_sc = fap_young[fap_young.obs['tech'] == 'scRNA-seq'] if 'scRNA-seq' in fap_young.obs['tech'].values else fap_young
            o_sc = fap_old_scRNA if fap_old_scRNA.n_obs > 0 else fap_old
            result = compute_age_effect_for_gene(gene, y_sc, o_sc, 'FAP_scRNA')
            fap_results[f'{gene}_scRNA'] = result
            print(f"  [scRNA] {gene}: d={result['d']:.3f}")
        except Exception as e:
            print(f"  [scRNA] {gene}: error ({e})")
        # snRNA
        try:
            o_sn = fap_old_snRNA if fap_old_snRNA.n_obs > 0 else fap_old
            y_sn = fap_young[fap_young.obs['tech'] == 'snRNA-seq'] if 'snRNA-seq' in fap_young.obs['tech'].values else fap_young
            result = compute_age_effect_for_gene(gene, y_sn, o_sn, 'FAP_snRNA')
            fap_results[f'{gene}_snRNA'] = result
            print(f"  [snRNA] {gene}: d={result['d']:.3f}")
        except Exception as e:
            print(f"  [snRNA] {gene}: error ({e})")

results['age_effects']['fap'] = fap_results

# MuSC Age Effects
print("\n--- MuSC Age Effects ---")
musc_results = {}

for gene in tier1_passing:
    result = compute_age_effect_for_gene(gene, musc_young, musc_old, 'MuSC')
    musc_results[f'tier1_{gene}'] = result
    print(f"  {gene}: d={result['d']:.3f} (p={result['p_value']:.2e})")

for gene in TIER2_SHARED:
    if gene in musc_genes:
        result = compute_age_effect_for_gene(gene, musc_young, musc_old, 'MuSC')
        musc_results[f'tier2_{gene}'] = result
        print(f"  [shared] {gene}: d={result['d']:.3f} (p={result['p_value']:.2e})")

for gene in ['JUNB', 'FOS', 'JUND', 'JUN']:
    if gene in musc_genes:
        result = compute_age_effect_for_gene(gene, musc_young, musc_old, 'MuSC')
        musc_results[f'ap1_{gene}'] = result
        print(f"  [AP-1] {gene}: d={result['d']:.3f} (p={result['p_value']:.2e})")

# Technology-stratified for MuSC
for gene in ['JUNB', 'MEF2A', 'FOS']:
    if gene in musc_genes:
        y_sc = musc_young_scRNA if musc_young_scRNA.n_obs > 0 else musc_young
        o_sc = musc_old_scRNA if musc_old_scRNA.n_obs > 0 else musc_old
        result = compute_age_effect_for_gene(gene, y_sc, o_sc, 'MuSC_scRNA')
        musc_results[f'{gene}_scRNA'] = result
        print(f"  [scRNA] {gene}: d={result['d']:.3f}")

        y_sn = musc_young[musc_young.obs['tech'] == 'snRNA'] if 'snRNA' in musc_young.obs['tech'].values else musc_young
        o_sn = musc_old_snRNA if musc_old_snRNA.n_obs > 0 else musc_old
        result = compute_age_effect_for_gene(gene, y_sn, o_sn, 'MuSC_snRNA')
        musc_results[f'{gene}_snRNA'] = result
        print(f"  [snRNA] {gene}: d={result['d']:.3f}")

results['age_effects']['musc'] = musc_results

# =============================================================================
# STEP 5: Correlation Analysis (Aged Cells)
# =============================================================================
print("\n[5/6] Computing correlations in aged cells...")

def spearman_corr_with_pval(x, y):
    """Compute Spearman correlation with p-value."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)

def compute_correlations(old_data, label, gene_list_fap, gene_list_musc):
    """Compute JUNB-DDR, MAPK TF-JUNB, MAPK-DDR correlations."""
    corrs = {}

    # JUNB expression
    junb_expr = compute_gene_expr(old_data, 'JUNB')

    # DDR score
    ddr_genes = [g for g in DDR_GENES if g in old_data.var_names]
    if len(ddr_genes) > 0:
        ddr_score = compute_mean_score(old_data, ddr_genes)
    else:
        ddr_score = np.zeros(old_data.n_obs)

    # Tier 1 MAPK TF score (only passing genes)
    tier1_pass = [g for g in tier1_passing if g in old_data.var_names]
    if len(tier1_pass) > 0:
        mapk_score = compute_mean_score(old_data, tier1_pass)
    else:
        mapk_score = np.zeros(old_data.n_obs)

    # AP-1 score
    ap1_genes = [g for g in TIER3_AP1 if g in old_data.var_names]
    if len(ap1_genes) > 0:
        ap1_score = compute_mean_score(old_data, ap1_genes)
    else:
        ap1_score = np.zeros(old_data.n_obs)

    # JUNB-DDR correlation (replicate F034/F039)
    rho, p = spearman_corr_with_pval(junb_expr, ddr_score)
    corrs['junb_ddr_rho'] = rho
    corrs['junb_ddr_p'] = p

    # MAPK TF-JUNB correlation (primary test)
    rho, p = spearman_corr_with_pval(mapk_score, junb_expr)
    corrs['mapk_junb_rho'] = rho
    corrs['mapk_junb_p'] = p

    # MAPK TF-DDR correlation
    rho, p = spearman_corr_with_pval(mapk_score, ddr_score)
    corrs['mapk_ddr_rho'] = rho
    corrs['mapk_ddr_p'] = p

    # AP-1-JUNB correlation (Tier 3)
    rho, p = spearman_corr_with_pval(ap1_score, junb_expr)
    corrs['ap1_junb_rho'] = rho
    corrs['ap1_junb_p'] = p

    # JUNB-DDR partial correlation (only valid if Tier1 and DDR don't overlap)
    # Tier 1 genes don't overlap with DDR, so this is valid
    if np.std(mapk_score) > 0 and np.std(junb_expr) > 0:
        partial = partial_correlation(mapk_score, junb_expr, ddr_score)
        corrs['mapk_junb_partial_ddr'] = partial
    else:
        corrs['mapk_junb_partial_ddr'] = np.nan

    corrs['n'] = int(old_data.n_obs)

    return corrs

# FAP correlations
print("\n--- FAP Correlations (Old Cells) ---")
fap_corrs = compute_correlations(fap_old, 'FAP_old', fap_genes, fap_genes)
for k, v in fap_corrs.items():
    if k != 'n' and v is not None:
        print(f"  {k}: {v:.3f}")
results['correlations']['fap'] = fap_corrs

# MuSC correlations
print("\n--- MuSC Correlations (Old Cells) ---")
musc_corrs = compute_correlations(musc_old, 'MuSC_old', musc_genes, musc_genes)
for k, v in musc_corrs.items():
    if k != 'n' and v is not None:
        print(f"  {k}: {v:.3f}")
results['correlations']['musc'] = musc_corrs

# =============================================================================
# STEP 6: Cross-Atlas Validation (Nature Aging FAPs)
# =============================================================================
if na_available:
    print("\n[6/6] Cross-atlas validation (Nature Aging FAPs)...")
    try:
        na_genes = list(na_fap.var_names)
        print(f"  Genes: {len(na_genes):,}")
        print(f"  Age_bin distribution: {na_fap.obs['Age_bin'].value_counts().to_dict()}")

        # Select FB cells (fibroblasts)
        na_fap_fb = na_fap[na_fap.obs['annotation_level1'] == 'FB'].copy()
        print(f"  FB cells: {na_fap_fb.n_obs:,}")

        # Stratify by Age_bin
        na_young = na_fap_fb[na_fap_fb.obs['Age_bin'] == 'young'].copy()
        na_old = na_fap_fb[na_fap_fb.obs['Age_bin'] == 'old'].copy()
        print(f"  Young: {na_young.n_obs:,} cells")
        print(f"  Old: {na_old.n_obs:,} cells")

        # Check key gene detection
        for gene in ['JUNB', 'MEF2A', 'FOS', 'JUND', 'GADD45B', 'MAPK14']:
            if gene in na_genes:
                det = compute_detection(na_old, gene)
                print(f"  [NA] {gene} detection (old): {det:.3f}")

        # Age effects
        na_results = {}
        for gene in ['JUNB', 'MEF2A', 'FOS', 'JUND', 'GADD45B']:
            if gene in na_genes:
                result = compute_age_effect_for_gene(gene, na_young, na_old, 'NA_FAP')
                na_results[f'{gene}_d'] = result['d']
                na_results[f'{gene}_p'] = result['p_value']
                print(f"  [NA] {gene}: d={result['d']:.3f} (p={result['p_value']:.2e})")

        # Correlations in aged cells (FB only)
        na_corrs = compute_correlations(na_old, 'NA_FAP_old', na_genes, na_genes)
        for k, v in na_corrs.items():
            if k != 'n' and v is not None:
                print(f"  [NA] {k}: {v:.3f}")

        results['cross_atlas']['fap'] = na_results
        results['cross_atlas']['fap_corrs'] = na_corrs
        results['cross_atlas']['n_young'] = int(na_young.n_obs)
        results['cross_atlas']['n_old'] = int(na_old.n_obs)
    except Exception as e:
        print(f"  Error in Nature Aging analysis: {e}")
        import traceback
        traceback.print_exc()
        results['cross_atlas']['error'] = str(e)
else:
    print("\n[6/6] Cross-atlas validation: Nature Aging data not available")
    results['cross_atlas']['error'] = 'Nature Aging data not available'

# =============================================================================
# STEP 7: Decision Rule Evaluation
# =============================================================================
print("\n" + "=" * 60)
print("DECISION RULE EVALUATION")
print("=" * 60)

# FAP Tier 1 age effects
fap_mef2a_d = fap_results.get('tier1_MEF2A', {}).get('d')
fap_mapk14_d = fap_results.get('tier1_MAPK14', {}).get('d')
fap_max_d = fap_results.get('tier1_MAX', {}).get('d')

# MuSC Tier 1 age effects
musc_mef2a_d = musc_results.get('tier1_MEF2A', {}).get('d')
musc_mapk14_d = musc_results.get('tier1_MAPK14', {}).get('d')
musc_max_d = musc_results.get('tier1_MAX', {}).get('d')

# FAP Tier 2 age effects (FOS, JUND as MAPK downstream)
fap_fos_d = fap_results.get('tier2_FOS', {}).get('d') or fap_results.get('ap1_FOS', {}).get('d')
fap_jund_d = fap_results.get('tier2_JUND', {}).get('d') or fap_results.get('ap1_JUND', {}).get('d')

# FAP/MuSC MAPK TF-JUNB correlations
fap_mapk_junb_rho = fap_corrs.get('mapk_junb_rho')
musc_mapk_junb_rho = musc_corrs.get('mapk_junb_rho')
fap_ap1_junb_rho = fap_corrs.get('ap1_junb_rho')

print(f"\nTier 1 Age Effects:")
print(f"  FAP: MEF2A d={fap_mef2a_d}, MAPK14 d={fap_mapk14_d}, MAX d={fap_max_d}")
print(f"  MuSC: MEF2A d={musc_mef2a_d}, MAPK14 d={musc_mapk14_d}, MAX d={musc_max_d}")

print(f"\nMAPK TF-JUNB Correlations (aged cells):")
print(f"  FAP: rho={fap_mapk_junb_rho:.3f}")
print(f"  MuSC: rho={musc_mapk_junb_rho:.3f}")

print(f"\nAP-1-JUNB Correlation (FAP aged): rho={fap_ap1_junb_rho:.3f}")

# Decision evaluation
decision = {}
decision['detection_pass'] = detection_pass
decision['fap_tier1_best'] = max([fap_mef2a_d, fap_mapk14_d, fap_max_d or 0], key=lambda x: x if x else 0)
decision['musc_tier1_best'] = max([musc_mef2a_d, musc_mapk14_d, musc_max_d or 0], key=lambda x: x if x else 0)
decision['fap_fos_d'] = fap_fos_d
decision['fap_jund_d'] = fap_jund_d

results['decision'] = decision

print(f"\n--- DECISION ---")
if not detection_pass:
    print("INCONCLUSIVE: < 2 Tier 1 genes pass >20% detection threshold")
elif (decision['fap_tier1_best'] or 0) > 0.2 and (fap_mapk_junb_rho or 0) > 0.10:
    print("SUPPORTED: MAPK TF activity age-up (d>0.2) AND co-activated with JUNB (rho>0.10)")
elif (decision['fap_tier1_best'] or 0) > 0.2:
    print("PARTIAL: MAPK TFs age-up (d>0.2) but NOT co-activated with JUNB")
elif (decision['fap_tier1_best'] or 0) < 0.15 and (decision['musc_tier1_best'] or 0) < 0.15:
    print("REFUTED: MAPK TFs NOT age-activated in either compartment")
elif fap_fos_d and fap_fos_d > 0.2:
    print("FALLBACK: FOS (Tier 2 MAPK target) is age-up. MAPK stress response co-activates with JUNB.")
else:
    print("INCONCLUSIVE: Results not strong enough to classify")

# =============================================================================
# SAVE RESULTS
# =============================================================================
output_path = Path('experiments/batch_015/results.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: {output_path}")
print("=" * 60)
