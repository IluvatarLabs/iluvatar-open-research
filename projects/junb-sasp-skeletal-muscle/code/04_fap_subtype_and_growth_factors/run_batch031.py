#!/usr/bin/env python3
"""
Batch 031: FAP Subtype-Specific Growth Factor Analysis

PURPOSE: Analyze growth factor expression (FGF7, HGF, PDGFA, IGF1, IGF2)
by FAP subtype and age, plus JUNB correlations.

DESIGN:
- Part A: Donor-level pseudobulk age effects (Cohen's d + t-tests)
- Part B: Donor-level JUNB vs GF score correlation
- Part C: JUNB+ enrichment by subtype (exploratory, cell-level)

HYPOTHESIS: Growth factor expression differs by FAP subtype and age,
           with JUNB+ cells showing distinct GF profiles.

Author: Marvin (Research Agent)
Date: 2026-04-10
"""

import json
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Configuration
DATA_PATH = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad"
OUTPUT_PATH = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_031/results.json"
RANDOM_SEED = 42

# Set seed for reproducibility
np.random.seed(RANDOM_SEED)
warnings.filterwarnings('ignore')

# Growth factor genes (from design spec)
GF_GENES = ['FGF7', 'HGF', 'PDGFA', 'IGF1', 'IGF2']

# Subtypes for analysis (from design spec)
# Part A/B: 4 subtypes (Tenocyte excluded from inferential stats)
SUBTYPES_INFERENTIAL = ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP']  # N=3 for BH correction
SUBTYPES_DESCRIPTIVE = ['Tenocyte']  # N=4 young donors - too small for inferential
SUBTYPES_PART_C = ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'Tenocyte', 'RUNX2+ FAP']  # All subtypes for Part C

# Age thresholds (from design spec)
YOUNG_THRESHOLD = 40
OLD_THRESHOLD = 77

def load_data():
    """Load FAP atlas and return AnnData object."""
    print(f"Loading data from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"  Loaded {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    return adata

def categorize_age(age_series):
    """Convert age to numeric and categorize as young/old."""
    age_numeric = pd.to_numeric(age_series)
    return np.where(age_numeric <= YOUNG_THRESHOLD, 'young',
                   np.where(age_numeric >= OLD_THRESHOLD, 'old', 'middle'))

def safe_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: safe_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_to_native(x) for x in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return safe_to_native(obj.tolist())
    elif pd.isna(obj):
        return None
    else:
        return obj

def part_a_age_effects(adata):
    """
    Part A: Donor-level pseudobulk age effects.

    Steps:
    1. Filter to 4 subtypes
    2. Split into young (<=40) and old (>=77)
    3. Compute pseudobulk mean per donor per subtype per gene
    4. Report cell counts to verify N >= 50 cells per donor-subtype
    5. Compute Cohen's d (old vs young) per subtype per gene
    6. Welch's t-test per subtype per gene
    7. BH correction across 15 tests (3 subtypes × 5 genes)
    8. Report sparsity for PDGFA and HGF
    """
    print("\n" + "="*60)
    print("PART A: Donor-Level Age Effects on Growth Factors")
    print("="*60)

    # Filter to relevant subtypes
    subtypes_all = SUBTYPES_INFERENTIAL + SUBTYPES_DESCRIPTIVE
    adata_sub = adata[adata.obs['Annotation'].isin(subtypes_all)].copy()
    print(f"\nFiltered to {subtypes_all}")
    print(f"  Cells: {adata_sub.shape[0]:,}")

    # Add age category FIRST, then filter
    adata_sub.obs['age_cat'] = categorize_age(adata_sub.obs['age'])

    # Filter out middle-aged samples
    adata_sub = adata_sub[adata_sub.obs['age_cat'].isin(['young', 'old'])].copy()
    print(f"\nAfter filtering to young (<={YOUNG_THRESHOLD}) and old (>={OLD_THRESHOLD}):")
    print(f"  Cells: {adata_sub.shape[0]:,}")
    print(f"  Age categories: {adata_sub.obs['age_cat'].value_counts().to_dict()}")

    # Verify sample age mapping after filtering
    sample_to_age = adata_sub.obs.groupby('sample')['age_cat'].first().to_dict()
    print(f"  Samples included: {len(sample_to_age)}")
    print(f"  Young samples: {[s for s,a in sample_to_age.items() if a=='young']}")
    print(f"  Old samples: {[s for s,a in sample_to_age.items() if a=='old']}")

    # Step 4: Report cell counts per donor per subtype
    print("\n--- Donor Cell Counts (per subtype) ---")
    cell_counts = adata_sub.obs.groupby(['sample', 'Annotation']).size()
    donor_counts_dict = {}
    for (donor, subtype), count in cell_counts.items():
        if subtype not in donor_counts_dict:
            donor_counts_dict[subtype] = {}
        donor_counts_dict[subtype][donor] = int(count)

    # Check for N >= 50 cells per donor-subtype
    low_count_donors = []
    for (donor, subtype), count in cell_counts.items():
        if count < 50:
            low_count_donors.append(f"{donor} × {subtype}: {count} cells")

    if low_count_donors:
        print(f"WARNING: {len(low_count_donors)} donor-subtype pairs with <50 cells:")
        for x in low_count_donors[:10]:
            print(f"  {x}")

    # Step 3: Compute pseudobulk mean per donor per subtype per gene
    print("\n--- Computing Donor Pseudobulk Means ---")

    # Get gene expression matrix
    X = adata_sub[:, GF_GENES].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float64)  # Ensure float64 for calculations

    # Add gene expression to obs DataFrame
    obs_df = adata_sub.obs.copy()
    for i, gene in enumerate(GF_GENES):
        obs_df[gene] = X[:, i]

    # Compute pseudobulk: mean expression per donor per subtype (grouped by sample, Annotation, age_cat)
    pseudobulk = obs_df.groupby(['sample', 'Annotation', 'age_cat'])[GF_GENES].mean()
    pseudobulk = pseudobulk.reset_index()

    print(f"Pseudobulk table shape: {pseudobulk.shape}")
    print(f"Donors: {pseudobulk['sample'].nunique()}")

    # Report N per subtype per age
    print("\n--- Sample Sizes (Donors) ---")
    sample_sizes = {}
    for subtype in subtypes_all:
        sample_sizes[subtype] = {}
        for age_cat in ['young', 'old']:
            n = pseudobulk[(pseudobulk['Annotation'] == subtype) &
                          (pseudobulk['age_cat'] == age_cat)].shape[0]
            sample_sizes[subtype][age_cat] = n
            print(f"  {subtype} × {age_cat}: N={n} donors")

    # Step 5: Cohen's d per subtype per gene
    print("\n--- Cohen's d (Old vs Young) ---")
    cohens_d_results = {}

    for subtype in subtypes_all:
        cohens_d_results[subtype] = {}
        young_data = pseudobulk[(pseudobulk['Annotation'] == subtype) &
                                (pseudobulk['age_cat'] == 'young')]
        old_data = pseudobulk[(pseudobulk['Annotation'] == subtype) &
                              (pseudobulk['age_cat'] == 'old')]

        for gene in GF_GENES:
            y_vals = young_data[gene].dropna().values
            o_vals = old_data[gene].dropna().values

            if len(y_vals) == 0 or len(o_vals) == 0:
                d = 0.0
                mean_y = float('nan')
                mean_o = float('nan')
            else:
                mean_y = float(np.mean(y_vals))
                mean_o = float(np.mean(o_vals))
                mean_diff = mean_o - mean_y

                # Pooled SD
                n1, n2 = len(y_vals), len(o_vals)
                var1, var2 = np.var(y_vals, ddof=1), np.var(o_vals, ddof=1)
                pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

                if pooled_sd > 0:
                    d = mean_diff / pooled_sd
                else:
                    d = 0.0

            cohens_d_results[subtype][gene] = {
                'd': round(float(d), 4),
                'mean_young': round(mean_y, 4) if not np.isnan(mean_y) else None,
                'mean_old': round(mean_o, 4) if not np.isnan(mean_o) else None,
                'n_young': len(y_vals),
                'n_old': len(o_vals)
            }
            print(f"  {subtype} × {gene}: d={d:.3f} (young={mean_y:.3f}, old={mean_o:.3f})")

    # Step 6: Welch's t-test per subtype per gene
    print("\n--- Welch's t-test Results (Uncorrected) ---")
    ttest_results = {}

    for subtype in SUBTYPES_INFERENTIAL:  # Only inferential subtypes
        ttest_results[subtype] = {}
        young_data = pseudobulk[(pseudobulk['Annotation'] == subtype) &
                                (pseudobulk['age_cat'] == 'young')]
        old_data = pseudobulk[(pseudobulk['Annotation'] == subtype) &
                              (pseudobulk['age_cat'] == 'old')]

        for gene in GF_GENES:
            y_vals = young_data[gene].dropna().values
            o_vals = old_data[gene].dropna().values

            if len(y_vals) < 2 or len(o_vals) < 2:
                t_stat, p_val = float('nan'), float('nan')
            else:
                t_stat, p_val = stats.ttest_ind(o_vals, y_vals, equal_var=False)

            ttest_results[subtype][gene] = {
                't_stat': round(float(t_stat), 4) if not np.isnan(t_stat) else None,
                'p_value': round(float(p_val), 6) if not np.isnan(p_val) else None
            }
            print(f"  {subtype} × {gene}: t={t_stat:.3f}, p={p_val:.4f}")

    # Step 7: BH correction across 15 tests (3 subtypes × 5 genes)
    print("\n--- BH-Corrected p-values (15 tests) ---")
    all_pvals = []
    test_labels = []

    for subtype in SUBTYPES_INFERENTIAL:
        for gene in GF_GENES:
            p = ttest_results[subtype][gene]['p_value']
            if p is not None and not np.isnan(p):
                all_pvals.append(p)
            else:
                all_pvals.append(1.0)  # Use 1.0 for missing values
            test_labels.append(f"{subtype} × {gene}")

    reject, p_corrected, _, _ = multipletests(all_pvals, method='fdr_bh')

    bh_corrected = {}
    for i, label in enumerate(test_labels):
        subtype, gene = label.split(' × ')
        if subtype not in bh_corrected:
            bh_corrected[subtype] = {}
        bh_corrected[subtype][gene] = {
            'p_raw': round(all_pvals[i], 6),
            'p_bh': round(float(p_corrected[i]), 6),
            'significant': bool(reject[i])
        }
        sig_str = "***" if p_corrected[i] < 0.001 else "**" if p_corrected[i] < 0.01 else "*" if p_corrected[i] < 0.05 else ""
        print(f"  {label}: p_bh={p_corrected[i]:.4f} {sig_str}")

    # Step 8: Sparsity for PDGFA and HGF
    print("\n--- Sparsity (% non-zero cells) ---")
    sparsity_results = {}
    for subtype in subtypes_all:
        sparsity_results[subtype] = {}
        for gene in ['PDGFA', 'HGF']:
            for age_cat in ['young', 'old']:
                mask = (obs_df['Annotation'] == subtype) & (obs_df['age_cat'] == age_cat)
                cells = obs_df.loc[mask, gene].values
                non_zero = (cells > 0).mean() * 100
                sparsity_results[subtype][f"{gene}_{age_cat}"] = round(float(non_zero), 1)
                print(f"  {subtype} × {gene} × {age_cat}: {non_zero:.1f}% non-zero")

    return {
        'donor_cell_counts': donor_counts_dict,
        'sample_sizes': sample_sizes,
        'sparsity': sparsity_results,
        'cohens_d': cohens_d_results,
        't_tests': ttest_results,
        'bh_corrected': bh_corrected
    }

def part_b_junb_correlation(adata):
    """
    Part B: Donor-level JUNB vs GF score correlation.

    Steps:
    1. Compute mean JUNB expression per donor per subtype
    2. Compute mean GF score (z-scored [FGF7, HGD, PDGFA]) per donor per subtype
    3. Spearman correlation across donors within each subtype
    """
    print("\n" + "="*60)
    print("PART B: JUNB vs Growth Factor Score Correlation")
    print("="*60)

    # Filter to inferential subtypes
    adata_sub = adata[adata.obs['Annotation'].isin(SUBTYPES_INFERENTIAL)].copy()

    # Get JUNB expression
    junb_expr = adata_sub[:, 'JUNB'].X
    if hasattr(junb_expr, 'toarray'):
        junb_expr = junb_expr.toarray().flatten()
    adata_sub.obs['JUNB'] = junb_expr.astype(np.float64)

    # Compute GF score: mean of GF genes per cell, then z-score across donors
    gf_genes_partb = ['FGF7', 'HGF', 'PDGFA']  # As specified in design
    X_gf = adata_sub[:, gf_genes_partb].X
    if hasattr(X_gf, 'toarray'):
        X_gf = X_gf.toarray()
    X_gf = X_gf.astype(np.float64)

    # Step 1: Compute mean GF per cell
    cell_gf_mean = X_gf.mean(axis=1)

    # Create obs DataFrame with all needed columns
    obs_df = adata_sub.obs[['sample', 'Annotation']].copy()
    obs_df['JUNB'] = adata_sub.obs['JUNB'].values
    obs_df['GF_mean'] = cell_gf_mean

    # Step 2: Pseudobulk per donor per subtype
    pseudobulk = obs_df.groupby(['sample', 'Annotation']).agg({
        'JUNB': 'mean',
        'GF_mean': 'mean'
    }).reset_index()

    # Step 3: Z-score the GF_mean across donors within each subtype
    # Handle NaN by filtering valid data per subtype
    pseudobulk['GF_score'] = np.nan

    for subtype in SUBTYPES_INFERENTIAL:
        mask = pseudobulk['Annotation'] == subtype
        sub_data = pseudobulk.loc[mask].copy()

        # Drop rows with NaN for z-score calculation
        valid_mask = ~sub_data['GF_mean'].isna()
        valid_sub = sub_data[valid_mask]

        if len(valid_sub) > 1:
            mean_val = valid_sub['GF_mean'].mean()
            std_val = valid_sub['GF_mean'].std()
            if std_val > 0:
                zscore = (valid_sub['GF_mean'].values - mean_val) / std_val
                # Assign z-scores back to original indices
                pseudobulk.loc[valid_sub.index, 'GF_score'] = zscore
            else:
                pseudobulk.loc[valid_sub.index, 'GF_score'] = 0.0

    print(f"\nPseudobulk table: {pseudobulk.shape[0]} donor-subtype pairs")

    # Spearman correlation per subtype
    print("\n--- Spearman Correlations (JUNB vs GF score) ---")
    spearman_results = {}

    for subtype in SUBTYPES_INFERENTIAL:
        sub_data = pseudobulk[pseudobulk['Annotation'] == subtype].copy()

        # Drop rows with NaN in either JUNB or GF_score
        valid_data = sub_data.dropna(subset=['JUNB', 'GF_score'])
        n_donors = len(valid_data)

        if n_donors >= 3:
            rho, p_val = stats.spearmanr(valid_data['JUNB'], valid_data['GF_score'])
            if np.isnan(rho):
                spearman_results[subtype] = {
                    'rho': None,
                    'p_value': None,
                    'n_donors': int(n_donors),
                    'note': 'Constant values in one or both variables'
                }
                print(f"  {subtype}: N={n_donors} donors - constant values (correlation undefined)")
            else:
                spearman_results[subtype] = {
                    'rho': round(float(rho), 4),
                    'p_value': round(float(p_val), 6),
                    'n_donors': int(n_donors)
                }
                print(f"  {subtype}: rho={rho:.3f}, p={p_val:.4f} (N={n_donors} donors)")
        else:
            spearman_results[subtype] = {
                'rho': None,
                'p_value': None,
                'n_donors': int(n_donors),
                'note': 'Insufficient donors'
            }
            print(f"  {subtype}: N={n_donors} donors - insufficient for correlation")

    return spearman_results

def part_c_junb_enrichment(adata):
    """
    Part C: JUNB+ Subtype Composition (Exploratory, cell-level).

    Steps:
    1. Compute JUNB threshold per donor: donor-specific median + 1 MAD
    2. For each subtype, compute JUNB+ fraction (cells above donor-specific threshold)
    3. Compute enrichment ratio: observed / expected
    4. Include all subtypes (MME+, CD55+, GPC3+, Tenocyte, RUNX2+)

    NOTE: This is EXPLORATORY - cell-level, N inflated; interpret as pattern only.
    """
    print("\n" + "="*60)
    print("PART C: JUNB+ Enrichment by Subtype (EXPLORATORY)")
    print("="*60)

    # Filter to Part C subtypes
    adata_sub = adata[adata.obs['Annotation'].isin(SUBTYPES_PART_C)].copy()

    # Get JUNB expression
    junb_expr = adata_sub[:, 'JUNB'].X
    if hasattr(junb_expr, 'toarray'):
        junb_expr = junb_expr.toarray().flatten()
    adata_sub.obs['JUNB'] = junb_expr.astype(np.float64)

    # Compute donor-specific threshold (median + 1 MAD)
    print("\n--- Computing Donor-Specific JUNB Thresholds ---")
    obs_df = adata_sub.obs.copy()

    # Calculate per-donor stats
    donor_stats = obs_df.groupby('sample')['JUNB'].agg(['median', 'std']).reset_index()
    donor_stats['mad'] = 1.4826 * donor_stats['std']  # MAD = 1.4826 * SD
    donor_stats['threshold'] = donor_stats['median'] + donor_stats['mad']
    donor_threshold_map = dict(zip(donor_stats['sample'], donor_stats['threshold']))

    obs_df['junb_threshold'] = obs_df['sample'].map(donor_threshold_map)
    obs_df['is_junb_positive'] = (obs_df['JUNB'] > obs_df['junb_threshold']).astype(int)

    # Overall JUNB+ fraction (expected)
    overall_junb_frac = obs_df['is_junb_positive'].mean()
    print(f"Overall JUNB+ fraction: {overall_junb_frac*100:.1f}%")

    # Compute enrichment per subtype
    print("\n--- JUNB+ Enrichment by Subtype ---")
    enrichment_results = {}

    for subtype in SUBTYPES_PART_C:
        sub_data = obs_df[obs_df['Annotation'] == subtype]
        n_cells = len(sub_data)
        n_junb_pos = sub_data['is_junb_positive'].sum()
        observed_frac = n_junb_pos / n_cells if n_cells > 0 else 0

        # Enrichment ratio: observed / expected
        if overall_junb_frac > 0:
            enrichment = observed_frac / overall_junb_frac
        else:
            enrichment = 0

        enrichment_results[subtype] = {
            'n_cells': int(n_cells),
            'n_junb_pos': int(n_junb_pos),
            'observed_fraction': round(float(observed_frac), 4),
            'expected_fraction': round(float(overall_junb_frac), 4),
            'enrichment_ratio': round(float(enrichment), 4)
        }

        status = "ENRICHED" if enrichment > 1.2 else "DEPLETED" if enrichment < 0.8 else "SIMILAR"
        print(f"  {subtype}: {n_cells:,} cells, {n_junb_pos:,} JUNB+ ({observed_frac*100:.1f}%), "
              f"enrichment={enrichment:.2f} [{status}]")

    print("\nNOTE: Part C is EXPLORATORY (cell-level, N inflated)")
    print("      Interpret as suggestive patterns only, not inferential conclusions.")

    return enrichment_results

def main():
    """Main execution function."""
    print("="*60)
    print("BATCH 031: FAP Subtype-Specific Growth Factor Analysis")
    print("="*60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Date: 2026-04-10")

    # Load data
    adata = load_data()

    # Run analyses
    results = {
        'metadata': {
            'batch': 'batch_031',
            'purpose': 'FAP subtype-specific GF expression by age',
            'date': '2026-04-10',
            'random_seed': RANDOM_SEED,
            'data_path': DATA_PATH,
            'n_cells_total': int(adata.shape[0]),
            'n_genes_total': int(adata.shape[1]),
            'n_donors_total': int(adata.obs['sample'].nunique())
        },
        'part_a': {},
        'part_b': {},
        'part_c': {}
    }

    # Part A
    results['part_a'] = part_a_age_effects(adata)

    # Part B
    results['part_b'] = part_b_junb_correlation(adata)

    # Part C
    results['part_c'] = part_c_junb_enrichment(adata)

    # Convert to native Python types for JSON serialization
    results = safe_to_native(results)

    # Save results
    print(f"\n\nSaving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done!")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF KEY RESULTS")
    print("="*60)

    print("\n--- Cohen's d Table (Part A) ---")
    print(f"{'Subtype':<15} {'FGF7':>8} {'HGF':>8} {'PDGFA':>8} {'IGF1':>8} {'IGF2':>8}")
    print("-" * 60)
    for subtype in SUBTYPES_INFERENTIAL + SUBTYPES_DESCRIPTIVE:
        d_vals = [results['part_a']['cohens_d'][subtype][g]['d'] for g in GF_GENES]
        print(f"{subtype:<15} " + " ".join([f"{d:>8.3f}" for d in d_vals]))

    print("\n--- Spearman Correlations (Part B) ---")
    for subtype, data in results['part_b'].items():
        if data['rho'] is not None:
            print(f"  {subtype}: rho={data['rho']:.3f}, p={data['p_value']:.4f} (N={data['n_donors']})")
        else:
            print(f"  {subtype}: {data.get('note', 'N/A')}")

    print("\n--- JUNB+ Enrichment (Part C, EXPLORATORY) ---")
    for subtype, data in results['part_c'].items():
        print(f"  {subtype}: enrichment={data['enrichment_ratio']:.2f} "
              f"({data['n_junb_pos']:,}/{data['n_cells']:,} cells)")

    return results

if __name__ == '__main__':
    results = main()
