#!/usr/bin/env python3
"""
Batch 022: R1-R4 computations for SM-RD skeletal muscle aging project.

Implements:
- R1: Pseudobulk donor-level correlation (FAP + Nature Aging cross-atlas)
- R2: Donor-level Cohen's d with bootstrap CI
- R3: SASP panel sensitivity analysis
- R4: Vascular JUNB-SASP correlation

Date: 2026-04-10
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr, bootstrap, ttest_ind
from scipy.stats import mannwhitneyu
import statsmodels.formula.api as smf
import statsmodels.api as sm
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_DIR = '/home/yuanz/Documents/GitHub/biomarvin_fibro/data'
OUTPUT_DIR = '/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_022'

FAP_ATLAS = os.path.join(DATA_DIR, 'OMIX004308-02.h5ad')
VASCULAR_ATLAS = os.path.join(DATA_DIR, 'Vascular_scsn_RNA.h5ad')
NATURE_AGING_ATLAS = os.path.join(DATA_DIR, 'OMIX004308-05.h5ad')

# SASP panels
SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAU']
SASP_COPPE = ['IL6', 'CXCL8', 'CXCL1', 'CCL2', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU']
SASP_BASISTY = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'IL6', 'IL1B', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'VEGFA', 'CCL5', 'CXCL8']
SASP_MINIMAL = ['CCL2', 'CXCL1', 'CXCL2', 'IL6']

# FAP subtypes to include (excluding tenocytes and rare subtypes)
FAP_SUBTYPES = ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP']

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_adata(path: str) -> ad.AnnData:
    """Load AnnData object with error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return ad.read_h5ad(path)


def compute_pseudobulk(adata: ad.AnnData, genes: List[str], groupby: str = 'sample') -> pd.DataFrame:
    """
    Compute mean expression per donor/group for specified genes.

    Args:
        adata: AnnData object
        genes: List of gene names to compute
        groupby: Column in obs to group by (default: 'sample')

    Returns:
        DataFrame with mean expression per group
    """
    # Filter genes that exist in the data
    available_genes = [g for g in genes if g in adata.var_names]
    missing_genes = [g for g in genes if g not in adata.var_names]

    if missing_genes:
        print(f"  Warning: {len(missing_genes)} genes not in data: {missing_genes[:5]}{'...' if len(missing_genes) > 5 else ''}")

    if not available_genes:
        raise ValueError(f"No genes from list found in data: {genes}")

    # Extract expression matrix efficiently
    result_dict = {}

    for gene in available_genes:
        expr = adata[:, gene].X
        # Handle sparse matrices
        if hasattr(expr, 'toarray'):
            expr = expr.toarray().ravel()
        else:
            expr = np.asarray(expr).ravel()
        result_dict[gene] = expr

    # Create DataFrame
    df = pd.DataFrame(result_dict, index=adata.obs_names)
    df[groupby] = adata.obs[groupby].values

    # Group by and compute mean
    grouped = df.groupby(groupby)[available_genes].mean()

    return grouped


def hedges_g(x1: np.ndarray, x2: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute Hedges' g effect size with pooled SD.

    Args:
        x1: Array of values from group 1 (old)
        x2: Array of values from group 2 (young)

    Returns:
        Tuple of (hedges_g, cohens_d, pooled_sd, se)
    """
    n1, n2 = len(x1), len(x2)

    # Means and standard deviations
    mean1, mean2 = np.mean(x1), np.mean(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_sd

    # Hedges' g correction factor
    df = n1 + n2 - 2
    correction = 1 - (3 / (4 * df - 1))
    hedges_g = cohens_d * correction

    # Standard error of Hedges' g
    se = np.sqrt((n1 + n2) / (n1 * n2) + (hedges_g ** 2) / (2 * (n1 + n2)))

    return hedges_g, cohens_d, pooled_sd, se


def bootstrap_ci(
    x1: np.ndarray,
    x2: np.ndarray,
    n_iterations: int = 10000,
    ci: float = 0.95,
    stat_func='hedges_g'
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for effect size.

    Args:
        x1: Array of values from group 1 (old)
        x2: Array of values from group 2 (young)
        n_iterations: Number of bootstrap iterations
        ci: Confidence level (default 0.95)
        stat_func: Which statistic to compute ('hedges_g' or 'mean_diff')

    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n1, n2 = len(x1), len(x2)

    boot_stats = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Resample with replacement within each group
        idx1 = rng.integers(0, n1, size=n1)
        idx2 = rng.integers(0, n2, size=n2)

        boot_x1 = x1[idx1]
        boot_x2 = x2[idx2]

        if stat_func == 'hedges_g':
            boot_stats[i], _, _, _ = hedges_g(boot_x1, boot_x2)
        else:
            # Mean difference
            boot_stats[i] = np.mean(boot_x1) - np.mean(boot_x2)

    # Percentile CI
    alpha = 1 - ci
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper


def mixed_effects_model(
    df: pd.DataFrame,
    formula: str = "JUNB ~ SASP_mean"
) -> Dict[str, Any]:
    """
    Fit mixed effects model: JUNB ~ SASP12 + (1|donor)

    Args:
        df: DataFrame with columns for response, predictor, and grouping variable
        formula: Model formula

    Returns:
        Dictionary with model results
    """
    try:
        # Ensure proper data types
        df = df.copy()

        # Fit mixed effects model
        model = smf.mixedlm(formula, df, groups=df['sample'])
        result = model.fit(method='powell')

        # Extract key statistics
        fixed_effects = result.fe_params
        pvalues = result.pvalues
        conf_int = result.conf_int()

        output = {
            'slope': float(fixed_effects.get('SASP_mean', np.nan)),
            'intercept': float(fixed_effects.get('Intercept', np.nan)),
            'p_value': float(pvalues.get('SASP_mean', np.nan)),
            'ci_lower': float(conf_int.loc['SASP_mean', 0]) if 'SASP_mean' in conf_int.index else np.nan,
            'ci_upper': float(conf_int.loc['SASP_mean', 1]) if 'SASP_mean' in conf_int.index else np.nan,
            'aic': float(result.aic),
            'bic': float(result.bic),
            'converged': result.converged,
            'method': 'mixedlm'
        }

        return output

    except Exception as e:
        print(f"  Warning: Mixed effects model failed: {e}")
        return {
            'slope': np.nan,
            'intercept': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'converged': False,
            'method': 'mixedlm',
            'error': str(e)
        }


def spearman_with_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute Spearman correlation with p-value."""
    rho, pval = spearmanr(x, y, nan_policy='omit')
    return float(rho), float(pval)


def compute_cell_level_correlation(
    adata: ad.AnnData,
    gene1: str,
    gene2: str
) -> Dict[str, float]:
    """
    Compute cell-level Spearman correlation between two genes.

    Returns NaN if insufficient data.
    """
    if gene1 not in adata.var_names or gene2 not in adata.var_names:
        return {'rho': np.nan, 'p_value': np.nan, 'n_cells': 0}

    expr1 = adata[:, gene1].X
    expr2 = adata[:, gene2].X

    if hasattr(expr1, 'toarray'):
        expr1 = expr1.toarray().ravel()
    else:
        expr1 = np.asarray(expr1).ravel()

    if hasattr(expr2, 'toarray'):
        expr2 = expr2.toarray().ravel()
    else:
        expr2 = np.asarray(expr2).ravel()

    # Remove NaN values
    valid_idx = ~(np.isnan(expr1) | np.isnan(expr2))
    expr1_valid = expr1[valid_idx]
    expr2_valid = expr2[valid_idx]

    if len(expr1_valid) < 10:
        return {'rho': np.nan, 'p_value': np.nan, 'n_cells': len(expr1_valid)}

    rho, pval = spearmanr(expr1_valid, expr2_valid)

    return {'rho': float(rho), 'p_value': float(pval), 'n_cells': int(len(expr1_valid))}


# ============================================================================
# R1: PSEUDOBULK DONOR-LEVEL CORRELATION
# ============================================================================

def run_r1(fap_atlas: str, nature_aging_atlas: str) -> Dict[str, Any]:
    """
    R1: Pseudobulk donor-level correlation analysis.

    Computes:
    - Donor-level Spearman correlation (all, old-only, young-only)
    - Mixed effects model: JUNB ~ SASP12 + (1|donor)
    - Cross-atlas validation (Nature Aging FB cells)
    """
    print("\n" + "="*80)
    print("R1: PSEUDOBULK DONOR-LEVEL CORRELATION")
    print("="*80)

    results = {
        'donor_level': {},
        'mixed_effects': {},
        'cross_atlas': {},
        'per_gene_rho': {}
    }

    # -------------------------------------------------------------------------
    # Load FAP atlas
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading FAP atlas...")
    fap = load_adata(fap_atlas)
    print(f"  Loaded: {fap.n_obs} cells, {fap.n_vars} genes")
    print(f"  Annotations: {fap.obs['Annotation'].value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # Filter to FAP subtypes
    # -------------------------------------------------------------------------
    print("\n[2/4] Filtering to FAP subtypes...")
    fap_filt = fap[fap.obs['Annotation'].isin(FAP_SUBTYPES)].copy()
    print(f"  After filtering: {fap_filt.n_obs} cells")

    # Check age population distribution
    if 'age_pop' in fap_filt.obs.columns:
        print(f"  Age distribution: {fap_filt.obs['age_pop'].value_counts().to_dict()}")

    # Check sample (donor) distribution
    if 'sample' in fap_filt.obs.columns:
        n_donors = fap_filt.obs['sample'].nunique()
        print(f"  Number of donors: {n_donors}")

        # Count old and young donors
        age_pop_map = fap_filt.obs.groupby('sample')['age_pop'].first()
        n_old = (age_pop_map == 'old_pop').sum()
        n_young = (age_pop_map == 'young_pop').sum()
        print(f"  Old donors: {n_old}, Young donors: {n_young}")

    # -------------------------------------------------------------------------
    # Compute pseudobulk per donor
    # -------------------------------------------------------------------------
    print("\n[3/4] Computing pseudobulk per donor...")

    # Compute JUNB mean per donor
    junb_df = compute_pseudobulk(fap_filt, ['JUNB'], 'sample')
    junb_df.columns = ['JUNB']

    # Compute SASP12 mean per donor (mean of gene means)
    sasp12_available = [g for g in SASP12 if g in fap_filt.var_names]
    print(f"  SASP12 genes available: {len(sasp12_available)}/{len(SASP12)}")

    sasp_df = compute_pseudobulk(fap_filt, sasp12_available, 'sample')

    # Create SASP12 composite score (mean across genes)
    sasp_df['SASP12_mean'] = sasp_df.mean(axis=1)

    # Merge JUNB and SASP12
    pseudobulk_df = junb_df.join(sasp_df[['SASP12_mean']])

    # Add age information
    age_map = fap_filt.obs.groupby('sample')['age_pop'].first()
    pseudobulk_df['age_pop'] = age_map

    print(f"  Pseudobulk computed for {len(pseudobulk_df)} donors")

    # -------------------------------------------------------------------------
    # Donor-level Spearman correlation (all donors)
    # -------------------------------------------------------------------------
    print("\n[4/4] Computing donor-level Spearman correlations...")

    # All donors
    rho_all, pval_all = spearman_with_pvalue(
        pseudobulk_df['JUNB'].values,
        pseudobulk_df['SASP12_mean'].values
    )

    results['donor_level'] = {
        'rho': rho_all,
        'p_value': pval_all,
        'n_donors': int(len(pseudobulk_df))
    }

    print(f"  All donors (N={len(pseudobulk_df)}): rho={rho_all:.4f}, p={pval_all:.4e}")

    # Old-only donors
    old_df = pseudobulk_df[pseudobulk_df['age_pop'] == 'old_pop']
    if len(old_df) > 2:
        rho_old, pval_old = spearman_with_pvalue(old_df['JUNB'].values, old_df['SASP12_mean'].values)
        results['donor_level']['rho_old_only'] = rho_old
        results['donor_level']['p_value_old_only'] = pval_old
        results['donor_level']['n_old_donors'] = int(len(old_df))
        print(f"  Old-only (N={len(old_df)}): rho={rho_old:.4f}, p={pval_old:.4e}")

    # Young-only donors
    young_df = pseudobulk_df[pseudobulk_df['age_pop'] == 'young_pop']
    if len(young_df) > 2:
        rho_young, pval_young = spearman_with_pvalue(young_df['JUNB'].values, young_df['SASP12_mean'].values)
        results['donor_level']['rho_young_only'] = rho_young
        results['donor_level']['p_value_young_only'] = pval_young
        results['donor_level']['n_young_donors'] = int(len(young_df))
        print(f"  Young-only (N={len(young_df)}): rho={rho_young:.4f}, p={pval_young:.4e}")

    # -------------------------------------------------------------------------
    # Per-gene correlations
    # -------------------------------------------------------------------------
    print("\n  Computing per-gene Spearman correlations...")
    per_gene_rho = {}

    for gene in sasp12_available:
        gene_expr = pseudobulk_df[gene].values
        junb_expr = pseudobulk_df['JUNB'].values

        # Only compute if sufficient non-NaN values
        valid_idx = ~(np.isnan(gene_expr) | np.isnan(junb_expr))
        if valid_idx.sum() > 2:
            rho_gene, pval_gene = spearman_with_pvalue(
                junb_expr[valid_idx],
                gene_expr[valid_idx]
            )
            per_gene_rho[gene] = {'rho': rho_gene, 'p_value': pval_gene}

    results['per_gene_rho'] = per_gene_rho
    print(f"  Computed correlations for {len(per_gene_rho)} genes")

    # -------------------------------------------------------------------------
    # Mixed effects model
    # -------------------------------------------------------------------------
    print("\n  Fitting mixed effects model: JUNB ~ SASP12 + (1|donor)...")

    # Prepare data for mixed model
    mixed_df = pseudobulk_df.reset_index()
    mixed_df = mixed_df.rename(columns={'index': 'sample'})

    mixed_results = mixed_effects_model(
        mixed_df[['sample', 'JUNB', 'SASP12_mean']],
        formula="JUNB ~ SASP12_mean"
    )

    results['mixed_effects'] = mixed_results
    print(f"  Slope: {mixed_results.get('slope', 'N/A')}, p={mixed_results.get('p_value', 'N/A')}")
    print(f"  95% CI: [{mixed_results.get('ci_lower', 'N/A')}, {mixed_results.get('ci_upper', 'N/A')}]")
    print(f"  Converged: {mixed_results.get('converged', 'N/A')}")

    # -------------------------------------------------------------------------
    # Cross-atlas validation: Nature Aging atlas
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("Cross-atlas validation (Nature Aging FB cells)")
    print("-"*60)

    try:
        print("\n  Loading Nature Aging atlas...")
        nature = load_adata(nature_aging_atlas)
        print(f"  Loaded: {nature.n_obs} cells, {nature.n_vars} genes")

        # Check annotation column
        if 'Annotation' in nature.obs.columns:
            print(f"  Annotations: {nature.obs['Annotation'].value_counts().head(10).to_dict()}")

            # Filter to FB annotation (FAP-like fibroblasts)
            if 'FB' in nature.obs['Annotation'].values:
                nature_fb = nature[nature.obs['Annotation'] == 'FB'].copy()
                print(f"  FB cells: {nature_fb.n_obs}")

                # Compute pseudobulk per donor
                if 'sample' in nature_fb.obs.columns or 'orig.ident' in nature_fb.obs.columns:
                    group_col = 'sample' if 'sample' in nature_fb.obs.columns else 'orig.ident'

                    # Check if JUNB exists
                    if 'JUNB' in nature_fb.var_names:
                        # Compute pseudobulk
                        junb_nat = compute_pseudobulk(nature_fb, ['JUNB'], group_col)
                        junb_nat.columns = ['JUNB']

                        # SASP12
                        sasp_nat_available = [g for g in SASP12 if g in nature_fb.var_names]
                        if sasp_nat_available:
                            sasp_nat = compute_pseudobulk(nature_fb, sasp_nat_available, group_col)
                            sasp_nat['SASP12_mean'] = sasp_nat.mean(axis=1)

                            # Merge
                            nat_df = junb_nat.join(sasp_nat[['SASP12_mean']])
                            nat_df = nat_df.dropna()

                            if len(nat_df) > 2:
                                rho_nat, pval_nat = spearman_with_pvalue(
                                    nat_df['JUNB'].values,
                                    nat_df['SASP12_mean'].values
                                )

                                results['cross_atlas'] = {
                                    'rho': rho_nat,
                                    'p_value': pval_nat,
                                    'n_donors': int(len(nat_df)),
                                    'source': 'Nature Aging FB cells',
                                    'n_sasp_genes': len(sasp_nat_available)
                                }

                                print(f"  Cross-atlas correlation (N={len(nat_df)}): rho={rho_nat:.4f}, p={pval_nat:.4e}")
                            else:
                                print(f"  Warning: Insufficient donors for cross-atlas correlation (N={len(nat_df)})")
                                results['cross_atlas'] = {
                                    'rho': np.nan,
                                    'p_value': np.nan,
                                    'n_donors': int(len(nat_df)),
                                    'source': 'Nature Aging FB cells',
                                    'note': 'Insufficient donors'
                                }
                        else:
                            print("  Warning: No SASP12 genes available in Nature Aging atlas")
                            results['cross_atlas'] = {
                                'rho': np.nan,
                                'p_value': np.nan,
                                'n_donors': 0,
                                'source': 'Nature Aging FB cells',
                                'note': 'No SASP12 genes available'
                            }
                    else:
                        print("  Warning: JUNB not found in Nature Aging atlas")
                        results['cross_atlas'] = {
                            'rho': np.nan,
                            'p_value': np.nan,
                            'n_donors': 0,
                            'source': 'Nature Aging FB cells',
                            'note': 'JUNB not found'
                        }
                else:
                    print("  Warning: No sample/orig.ident column found")
                    results['cross_atlas'] = {
                        'rho': np.nan,
                        'p_value': np.nan,
                        'n_donors': 0,
                        'source': 'Nature Aging FB cells',
                        'note': 'No donor identifier column'
                    }
            else:
                print("  Warning: FB annotation not found")
                results['cross_atlas'] = {
                    'rho': np.nan,
                    'p_value': np.nan,
                    'n_donors': 0,
                    'source': 'Nature Aging FB cells',
                    'note': 'FB annotation not found'
                }
        else:
            print("  Warning: No Annotation column found")
            results['cross_atlas'] = {
                'rho': np.nan,
                'p_value': np.nan,
                'n_donors': 0,
                'source': 'Nature Aging FB cells',
                'note': 'No Annotation column'
            }

    except Exception as e:
        print(f"  Error in cross-atlas analysis: {e}")
        results['cross_atlas'] = {
            'rho': np.nan,
            'p_value': np.nan,
            'n_donors': 0,
            'source': 'Nature Aging FB cells',
            'error': str(e)
        }

    # Save intermediate results
    print("\n  Saving R1 intermediate results...")
    r1_df = pseudobulk_df.reset_index()
    r1_df.to_csv(os.path.join(OUTPUT_DIR, 'r1_pseudobulk.csv'), index=False)

    return results


# ============================================================================
# R2: DONOR-LEVEL COHEN'S D WITH BOOTSTRAP CI
# ============================================================================

def run_r2(fap_atlas: str) -> Dict[str, Any]:
    """
    R2: Donor-level Cohen's d with bootstrap CI.

    Computes:
    - Hedges' g effect size (old vs young donors)
    - Bootstrap 95% CI (10,000 iterations)
    - Per-donor means
    """
    print("\n" + "="*80)
    print("R2: DONOR-LEVEL COHEN'S D WITH BOOTSTRAP CI")
    print("="*80)

    results = {}

    # -------------------------------------------------------------------------
    # Load FAP atlas
    # -------------------------------------------------------------------------
    print("\n[1/3] Loading FAP atlas...")
    fap = load_adata(fap_atlas)

    # Filter to FAP subtypes
    fap_filt = fap[fap.obs['Annotation'].isin(FAP_SUBTYPES)].copy()
    print(f"  FAP cells: {fap_filt.n_obs}")

    # -------------------------------------------------------------------------
    # Compute donor-level means
    # -------------------------------------------------------------------------
    print("\n[2/3] Computing donor-level JUNB means...")

    # Get age population per donor
    age_map = fap_filt.obs.groupby('sample')['age_pop'].first()

    # Compute pseudobulk for JUNB
    junb_df = compute_pseudobulk(fap_filt, ['JUNB'], 'sample')
    junb_df.columns = ['JUNB_mean']
    junb_df['age_pop'] = age_map

    # Separate old and young
    old_means = junb_df[junb_df['age_pop'] == 'old_pop']['JUNB_mean'].values
    young_means = junb_df[junb_df['age_pop'] == 'young_pop']['JUNB_mean'].values

    n_old = len(old_means)
    n_young = len(young_means)

    print(f"  Old donors (N={n_old}): mean={np.mean(old_means):.4f}, sd={np.std(old_means):.4f}")
    print(f"  Young donors (N={n_young}): mean={np.mean(young_means):.4f}, sd={np.std(young_means):.4f}")

    # -------------------------------------------------------------------------
    # Compute Hedges' g
    # -------------------------------------------------------------------------
    print("\n[3/3] Computing effect size...")

    hedges_g_val, cohens_d_val, pooled_sd, se = hedges_g(old_means, young_means)

    print(f"  Cohen's d: {cohens_d_val:.4f}")
    print(f"  Hedges' g: {hedges_g_val:.4f}")
    print(f"  Pooled SD: {pooled_sd:.4f}")

    # Bootstrap CI
    print("\n  Computing bootstrap 95% CI (10,000 iterations)...")
    ci_lower, ci_upper = bootstrap_ci(old_means, young_means, n_iterations=10000, ci=0.95)

    print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Save per-donor means
    per_donor_data = []
    for sample, row in junb_df.iterrows():
        per_donor_data.append({
            'donor': sample,
            'junb_mean': float(row['JUNB_mean']),
            'age_pop': row['age_pop']
        })

    results = {
        'hedges_g': float(hedges_g_val),
        'cohens_d': float(cohens_d_val),
        'pooled_sd': float(pooled_sd),
        'bootstrap_ci_lower': float(ci_lower),
        'bootstrap_ci_upper': float(ci_upper),
        'n_old': int(n_old),
        'n_young': int(n_young),
        'mean_old': float(np.mean(old_means)),
        'mean_young': float(np.mean(young_means)),
        'sd_old': float(np.std(old_means)),
        'sd_young': float(np.std(young_means)),
        'per_donor_means': per_donor_data
    }

    # Statistical test (Mann-Whitney U for robustness)
    stat_mw, pval_mw = mannwhitneyu(old_means, young_means, alternative='two-sided')
    results['mannwhitneyu_stat'] = float(stat_mw)
    results['mannwhitneyu_pvalue'] = float(pval_mw)

    # T-test for comparison
    stat_t, pval_t = ttest_ind(old_means, young_means)
    results['ttest_stat'] = float(stat_t)
    results['ttest_pvalue'] = float(pval_t)

    print(f"\n  Mann-Whitney U p-value: {pval_mw:.4e}")
    print(f"  T-test p-value: {pval_t:.4e}")

    # Interpretation
    if ci_lower > 0:
        interpretation = "Positive effect (old > young)"
    elif ci_upper < 0:
        interpretation = "Negative effect (old < young)"
    else:
        interpretation = "Effect includes zero (inconclusive)"

    results['interpretation'] = interpretation
    print(f"\n  Interpretation: {interpretation}")

    return results


# ============================================================================
# R3: SASP PANEL SENSITIVITY
# ============================================================================

def run_r3(fap_atlas: str) -> Dict[str, Any]:
    """
    R3: SASP panel sensitivity analysis.

    Tests 4 different SASP panels:
    - SASP12 (12 genes)
    - Coppé2008 (8 genes)
    - Basisty2020 (13 genes)
    - Minimal4 (4 genes)

    For each panel computes:
    - Cell-level Spearman rho
    - Donor-level Spearman rho
    """
    print("\n" + "="*80)
    print("R3: SASP PANEL SENSITIVITY ANALYSIS")
    print("="*80)

    results = {'panels': {}}

    # Define panels
    panels = {
        'SASP12': SASP12,
        'Coppé2008': SASP_COPPE,
        'Basisty2020': SASP_BASISTY,
        'Minimal4': SASP_MINIMAL
    }

    # -------------------------------------------------------------------------
    # Load FAP atlas
    # -------------------------------------------------------------------------
    print("\n[1/2] Loading FAP atlas...")
    fap = load_adata(fap_atlas)

    # Filter to FAP subtypes
    fap_filt = fap[fap.obs['Annotation'].isin(FAP_SUBTYPES)].copy()
    print(f"  FAP cells: {fap_filt.n_obs}")

    # Check JUNB availability
    if 'JUNB' not in fap_filt.var_names:
        print("  ERROR: JUNB not found in FAP atlas")
        return results

    # -------------------------------------------------------------------------
    # Analyze each panel
    # -------------------------------------------------------------------------
    print("\n[2/2] Analyzing SASP panels...")

    for panel_name, panel_genes in panels.items():
        print(f"\n  Panel: {panel_name}")

        # Filter to available genes
        available_genes = [g for g in panel_genes if g in fap_filt.var_names]
        n_available = len(available_genes)
        n_missing = len(panel_genes) - n_available

        print(f"    Genes: {n_available}/{len(panel_genes)} available")

        if n_missing > 0:
            missing = [g for g in panel_genes if g not in fap_filt.var_names]
            print(f"    Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        if n_available < 2:
            print(f"    WARNING: Less than 2 genes available, skipping")
            results['panels'][panel_name] = {
                'cell_rho': np.nan,
                'cell_p_value': np.nan,
                'donor_rho': np.nan,
                'donor_p_value': np.nan,
                'n_genes': 0,
                'genes_available': available_genes,
                'note': 'Insufficient genes'
            }
            continue

        # ---- Cell-level correlation ----
        print(f"    Computing cell-level correlation...")

        # Get JUNB expression
        junb_expr = fap_filt[:, 'JUNB'].X
        if hasattr(junb_expr, 'toarray'):
            junb_expr = junb_expr.toarray().ravel()
        else:
            junb_expr = np.asarray(junb_expr).ravel()

        # Compute mean SASP per cell
        sasp_expr_matrix = np.zeros((fap_filt.n_obs, n_available))
        for i, gene in enumerate(available_genes):
            expr = fap_filt[:, gene].X
            if hasattr(expr, 'toarray'):
                expr = expr.toarray().ravel()
            else:
                expr = np.asarray(expr).ravel()
            sasp_expr_matrix[:, i] = expr

        sasp_mean_per_cell = np.mean(sasp_expr_matrix, axis=1)

        # Spearman correlation
        cell_rho, cell_pval = spearman_with_pvalue(junb_expr, sasp_mean_per_cell)
        print(f"    Cell-level: rho={cell_rho:.4f}, p={cell_pval:.4e}")

        # ---- Donor-level correlation ----
        print(f"    Computing donor-level correlation...")

        # Pseudobulk per donor
        try:
            junb_pb = compute_pseudobulk(fap_filt, ['JUNB'], 'sample')
            sasp_pb = compute_pseudobulk(fap_filt, available_genes, 'sample')
            sasp_pb['SASP_mean'] = sasp_pb.mean(axis=1)

            pb_df = junb_pb.join(sasp_pb[['SASP_mean']])
            pb_df = pb_df.dropna()

            if len(pb_df) > 2:
                donor_rho, donor_pval = spearman_with_pvalue(
                    pb_df['JUNB'].values,
                    pb_df['SASP_mean'].values
                )
                print(f"    Donor-level: rho={donor_rho:.4f}, p={donor_pval:.4e}, N={len(pb_df)}")
            else:
                donor_rho, donor_pval = np.nan, np.nan
                print(f"    Donor-level: Insufficient donors (N={len(pb_df)})")

        except Exception as e:
            donor_rho, donor_pval = np.nan, np.nan
            print(f"    Donor-level: Error - {e}")

        # Store results
        results['panels'][panel_name] = {
            'cell_rho': float(cell_rho),
            'cell_p_value': float(cell_pval),
            'cell_n_cells': int(fap_filt.n_obs),
            'donor_rho': float(donor_rho) if not np.isnan(donor_rho) else None,
            'donor_p_value': float(donor_pval) if not np.isnan(donor_pval) else None,
            'n_genes': n_available,
            'genes_available': available_genes,
            'genes_missing': [g for g in panel_genes if g not in fap_filt.var_names]
        }

    # Summary
    print("\n  Summary:")
    print("  " + "-"*50)
    print(f"  {'Panel':<15} {'Cell ρ':<10} {'Donor ρ':<10} {'N genes'}")
    print("  " + "-"*50)
    for panel_name, panel_results in results['panels'].items():
        cell_rho_str = f"{panel_results['cell_rho']:.3f}" if not np.isnan(panel_results['cell_rho']) else "N/A"
        donor_rho_str = f"{panel_results['donor_rho']:.3f}" if panel_results['donor_rho'] is not None else "N/A"
        print(f"  {panel_name:<15} {cell_rho_str:<10} {donor_rho_str:<10} {panel_results['n_genes']}")

    return results


# ============================================================================
# R4: VASCULAR JUNB-SASP
# ============================================================================

def run_r4(vascular_atlas: str) -> Dict[str, Any]:
    """
    R4: Vascular JUNB-SASP analysis.

    Computes:
    - Cell-level JUNB-SASP12 Spearman correlation
    - Donor-level pseudobulk correlation
    - JUNB age effect (Cohen's d)
    """
    print("\n" + "="*80)
    print("R4: VASCULAR JUNB-SASP ANALYSIS")
    print("="*80)

    results = {
        'cell_level': {},
        'donor_level': {},
        'age_effect': {}
    }

    # -------------------------------------------------------------------------
    # Load Vascular atlas
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading Vascular atlas...")
    vascular = load_adata(vascular_atlas)
    print(f"  Loaded: {vascular.n_obs} cells, {vascular.n_vars} genes")

    # -------------------------------------------------------------------------
    # Explore annotations
    # -------------------------------------------------------------------------
    print("\n[2/5] Exploring cell annotations...")

    if 'Annotation' in vascular.obs.columns:
        annotations = vascular.obs['Annotation'].value_counts()
        print(f"  Available annotations:")
        for ann, count in annotations.items():
            print(f"    {ann}: {count}")

        # Identify endothelial cell types
        ec_keywords = ['CapEC', 'VenEC', 'ArtEC', 'EC', 'Endothelial']
        ec_types = [ann for ann in annotations.index if any(kw in ann for kw in ec_keywords)]

        if ec_types:
            print(f"\n  Endothelial cell types found: {ec_types}")
        else:
            print("\n  No endothelial markers found in annotations")

        results['available_annotations'] = annotations.to_dict()

    else:
        print("  Warning: No 'Annotation' column found")
        print(f"  Available obs columns: {list(vascular.obs.columns)[:10]}...")
        ec_types = []

    # -------------------------------------------------------------------------
    # Filter to endothelial cells
    # -------------------------------------------------------------------------
    print("\n[3/5] Filtering to endothelial cells...")

    # If no specific endothelial types found, use all cells
    if not ec_types:
        print("  Using all vascular cells for analysis")
        vascular_filt = vascular.copy()
    else:
        vascular_filt = vascular[vascular.obs['Annotation'].isin(ec_types)].copy()
        print(f"  Filtered to: {vascular_filt.n_obs} cells")

    results['cell_types_analyzed'] = ec_types if ec_types else ['all_vascular']
    results['n_cells_total'] = int(vascular_filt.n_obs)

    if vascular_filt.n_obs == 0:
        print("  ERROR: No cells after filtering")
        results['note'] = 'No cells after filtering'
        return results

    # -------------------------------------------------------------------------
    # Cell-level correlation
    # -------------------------------------------------------------------------
    print("\n[4/5] Computing cell-level JUNB-SASP12 correlation...")

    if 'JUNB' not in vascular_filt.var_names:
        print("  ERROR: JUNB not found in vascular atlas")
        results['cell_level'] = {'rho': np.nan, 'p_value': np.nan, 'n_cells': 0}
    else:
        # Get available SASP12 genes
        sasp_available = [g for g in SASP12 if g in vascular_filt.var_names]
        print(f"  SASP12 genes available: {len(sasp_available)}/{len(SASP12)}")

        if len(sasp_available) > 0:
            cell_result = compute_cell_level_correlation(
                vascular_filt,
                'JUNB',
                'MEAN_SASP'  # Will compute internally
            )

            # Compute SASP mean per cell for correlation
            junb_expr = vascular_filt[:, 'JUNB'].X
            if hasattr(junb_expr, 'toarray'):
                junb_expr = junb_expr.toarray().ravel()
            else:
                junb_expr = np.asarray(junb_expr).ravel()

            sasp_matrix = np.zeros((vascular_filt.n_obs, len(sasp_available)))
            for i, gene in enumerate(sasp_available):
                expr = vascular_filt[:, gene].X
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray().ravel()
                else:
                    expr = np.asarray(expr).ravel()
                sasp_matrix[:, i] = expr

            sasp_mean = np.mean(sasp_matrix, axis=1)

            # Valid cells
            valid_idx = ~(np.isnan(junb_expr) | np.isnan(sasp_mean))
            junb_valid = junb_expr[valid_idx]
            sasp_valid = sasp_mean[valid_idx]

            if len(junb_valid) > 10:
                cell_rho, cell_pval = spearman_with_pvalue(junb_valid, sasp_valid)

                results['cell_level'] = {
                    'rho': float(cell_rho),
                    'p_value': float(cell_pval),
                    'n_cells': int(valid_idx.sum()),
                    'n_sasp_genes': len(sasp_available)
                }

                print(f"  Cell-level: rho={cell_rho:.4f}, p={cell_pval:.4e}, N={valid_idx.sum()}")
            else:
                print("  ERROR: Insufficient valid cells")
                results['cell_level'] = {'rho': np.nan, 'p_value': np.nan, 'n_cells': int(valid_idx.sum())}
        else:
            print("  ERROR: No SASP12 genes available")
            results['cell_level'] = {'rho': np.nan, 'p_value': np.nan, 'n_cells': 0, 'note': 'No SASP12 genes'}

    # -------------------------------------------------------------------------
    # Donor-level correlation
    # -------------------------------------------------------------------------
    print("\n[5/5] Computing donor-level pseudobulk correlation...")

    # Identify donor column
    groupby = 'sample' if 'sample' in vascular_filt.obs.columns else 'orig.ident'

    if groupby in vascular_filt.obs.columns:
        n_donors = vascular_filt.obs[groupby].nunique()
        print(f"  Donor column: {groupby}, N={n_donors}")

        try:
            # Pseudobulk
            junb_pb = compute_pseudobulk(vascular_filt, ['JUNB'], groupby)
            sasp_available = [g for g in SASP12 if g in vascular_filt.var_names]
            sasp_pb = compute_pseudobulk(vascular_filt, sasp_available, groupby)
            sasp_pb['SASP_mean'] = sasp_pb.mean(axis=1)

            pb_df = junb_pb.join(sasp_pb[['SASP_mean']])
            pb_df = pb_df.dropna()

            if len(pb_df) > 2:
                donor_rho, donor_pval = spearman_with_pvalue(
                    pb_df['JUNB'].values,
                    pb_df['SASP_mean'].values
                )

                results['donor_level'] = {
                    'rho': float(donor_rho),
                    'p_value': float(donor_pval),
                    'n_donors': int(len(pb_df))
                }

                print(f"  Donor-level: rho={donor_rho:.4f}, p={donor_pval:.4e}, N={len(pb_df)}")
            else:
                print(f"  ERROR: Insufficient donors (N={len(pb_df)})")
                results['donor_level'] = {'rho': np.nan, 'p_value': np.nan, 'n_donors': int(len(pb_df))}

        except Exception as e:
            print(f"  ERROR in donor-level analysis: {e}")
            results['donor_level'] = {'rho': np.nan, 'p_value': np.nan, 'error': str(e)}
    else:
        print(f"  Warning: Donor column '{groupby}' not found")
        results['donor_level'] = {'rho': np.nan, 'p_value': np.nan, 'note': f'{groupby} column not found'}

    # -------------------------------------------------------------------------
    # Age effect (if age information available)
    # -------------------------------------------------------------------------
    print("\n  Computing JUNB age effect...")

    if 'age_pop' in vascular_filt.obs.columns or 'age' in vascular_filt.obs.columns:
        age_col = 'age_pop' if 'age_pop' in vascular_filt.obs.columns else 'age'

        # Get JUNB per cell
        junb_expr = vascular_filt[:, 'JUNB'].X
        if hasattr(junb_expr, 'toarray'):
            junb_expr = junb_expr.toarray().ravel()
        else:
            junb_expr = np.asarray(junb_expr).ravel()

        age_vals = vascular_filt.obs[age_col].values

        # Separate by age
        if age_col == 'age_pop':
            old_junb = junb_expr[age_vals == 'old_pop']
            young_junb = junb_expr[age_vals == 'young_pop']
        else:
            # Assume numeric age
            median_age = np.median(age_vals)
            old_junb = junb_expr[age_vals > median_age]
            young_junb = junb_expr[age_vals <= median_age]

        if len(old_junb) > 5 and len(young_junb) > 5:
            # Compute Cohen's d at cell level
            old_mean = np.mean(old_junb)
            young_mean = np.mean(young_junb)
            pooled_sd = np.sqrt((np.var(old_junb, ddof=1) + np.var(young_junb, ddof=1)) / 2)
            cohens_d = (old_mean - young_mean) / pooled_sd if pooled_sd > 0 else np.nan

            # Bootstrap CI
            ci_lower, ci_upper = bootstrap_ci(old_junb, young_junb, n_iterations=10000)

            results['age_effect'] = {
                'cohens_d': float(cohens_d),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'n_old_cells': int(len(old_junb)),
                'n_young_cells': int(len(young_junb)),
                'mean_old': float(old_mean),
                'mean_young': float(young_mean)
            }

            print(f"  Age effect (Cohen's d): {cohens_d:.4f}")
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"  Old mean: {old_mean:.4f} (N={len(old_junb)})")
            print(f"  Young mean: {young_mean:.4f} (N={len(young_junb)})")
        else:
            print(f"  Warning: Insufficient cells for age effect (old N={len(old_junb)}, young N={len(young_junb)})")
            results['age_effect'] = {'cohens_d': np.nan, 'note': 'Insufficient cells'}
    else:
        print("  Warning: No age information available")
        results['age_effect'] = {'cohens_d': np.nan, 'note': 'No age information'}

    # Summary
    print("\n  Summary:")
    print(f"    Cell-level ρ: {results['cell_level'].get('rho', 'N/A')}")
    print(f"    Donor-level ρ: {results['donor_level'].get('rho', 'N/A')}")
    print(f"    Age effect d: {results['age_effect'].get('cohens_d', 'N/A')}")
    print(f"    Total vascular cells: {results['n_cells_total']}")
    print(f"    Cell types: {results['cell_types_analyzed']}")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all R1-R4 computations and save results."""

    print("="*80)
    print("BATCH 022: R1-R4 COMPUTATIONS FOR SM-RD SKELETAL MUSCLE AGING")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Random seed: {RANDOM_SEED}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize results
    results = {
        'metadata': {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'batch': 'batch_022',
            'random_seed': RANDOM_SEED,
            'fap_subtypes': FAP_SUBTYPES,
            'sasp_panels': {
                'SASP12': SASP12,
                'Coppé2008': SASP_COPPE,
                'Basisty2020': SASP_BASISTY,
                'Minimal4': SASP_MINIMAL
            }
        },
        'r1': {},
        'r2': {},
        'r3': {},
        'r4': {}
    }

    # Run R1
    try:
        results['r1'] = run_r1(FAP_ATLAS, NATURE_AGING_ATLAS)
    except Exception as e:
        print(f"\nERROR in R1: {e}")
        import traceback
        traceback.print_exc()
        results['r1'] = {'error': str(e)}

    # Run R2
    try:
        results['r2'] = run_r2(FAP_ATLAS)
    except Exception as e:
        print(f"\nERROR in R2: {e}")
        import traceback
        traceback.print_exc()
        results['r2'] = {'error': str(e)}

    # Run R3
    try:
        results['r3'] = run_r3(FAP_ATLAS)
    except Exception as e:
        print(f"\nERROR in R3: {e}")
        import traceback
        traceback.print_exc()
        results['r3'] = {'error': str(e)}

    # Run R4
    try:
        results['r4'] = run_r4(VASCULAR_ATLAS)
    except Exception as e:
        print(f"\nERROR in R4: {e}")
        import traceback
        traceback.print_exc()
        results['r4'] = {'error': str(e)}

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    print("\n" + "="*80)
    print(f"SAVING RESULTS TO: {results_path}")
    print("="*80)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved successfully!")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF KEY FINDINGS")
    print("="*80)

    print("\nR1: Donor-level correlation")
    r1_dl = results['r1'].get('donor_level', {})
    print(f"  All donors (N={r1_dl.get('n_donors', 'N/A')}): rho={r1_dl.get('rho', 'N/A'):.4f}, p={r1_dl.get('p_value', 'N/A'):.4e}")

    r1_mixed = results['r1'].get('mixed_effects', {})
    if 'slope' in r1_mixed and not np.isnan(r1_mixed.get('slope', np.nan)):
        print(f"  Mixed effects slope: {r1_mixed.get('slope', 'N/A'):.4f}, p={r1_mixed.get('p_value', 'N/A'):.4e}")

    r1_cross = results['r1'].get('cross_atlas', {})
    if 'rho' in r1_cross and not np.isnan(r1_cross.get('rho', np.nan)):
        print(f"  Cross-atlas (Nature Aging): rho={r1_cross.get('rho', 'N/A'):.4f}, p={r1_cross.get('p_value', 'N/A'):.4e}")

    print("\nR2: Age effect (Cohen's d)")
    r2 = results['r2']
    print(f"  Hedges' g: {r2.get('hedges_g', 'N/A'):.4f}")
    print(f"  Bootstrap 95% CI: [{r2.get('bootstrap_ci_lower', 'N/A'):.4f}, {r2.get('bootstrap_ci_upper', 'N/A'):.4f}]")
    print(f"  N old: {r2.get('n_old', 'N/A')}, N young: {r2.get('n_young', 'N/A')}")

    print("\nR3: SASP panel sensitivity")
    r3 = results['r3'].get('panels', {})
    for panel_name in ['SASP12', 'Coppé2008', 'Basisty2020', 'Minimal4']:
        if panel_name in r3:
            pr = r3[panel_name]
            cell_rho = f"{pr.get('cell_rho', 'N/A'):.4f}" if pr.get('cell_rho') is not None and not np.isnan(pr.get('cell_rho', np.nan)) else "N/A"
            donor_rho = f"{pr.get('donor_rho', 'N/A'):.4f}" if pr.get('donor_rho') is not None else "N/A"
            print(f"  {panel_name}: Cell ρ={cell_rho}, Donor ρ={donor_rho}, N_genes={pr.get('n_genes', 'N/A')}")

    print("\nR4: Vascular analysis")
    r4 = results['r4']
    print(f"  Cell-level ρ: {r4.get('cell_level', {}).get('rho', 'N/A')}")
    print(f"  Donor-level ρ: {r4.get('donor_level', {}).get('rho', 'N/A')}")
    print(f"  Age effect d: {r4.get('age_effect', {}).get('cohens_d', 'N/A')}")
    print(f"  Cell types: {r4.get('cell_types_analyzed', 'N/A')}")

    print("\n" + "="*80)
    print("BATCH 022 COMPLETE")
    print("="*80)

    return results


if __name__ == '__main__':
    results = main()
