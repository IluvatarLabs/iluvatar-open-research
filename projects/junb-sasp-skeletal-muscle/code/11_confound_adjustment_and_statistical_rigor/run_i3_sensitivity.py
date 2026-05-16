#!/usr/bin/env python3
"""
batch_056: Preprint Robustness Checks — I3 Cell Count Sensitivity + A3 QC + A2 Donor Table
========================================================================================

PURPOSE:
1. I3: Test whether primary TF-SASP donor-level correlations are robust to
   cell count per donor thresholds. Pre-registered in brief.md.
2. A3: Generate QC metrics summary table per dataset × age_group.
3. A2: Generate complete donor/cell count table per compartment × age × tech × country.

WHY: Reviewer defense for preprint submission. Table stakes.

KEY DESIGN DECISIONS (from design review):
- Report country composition at each threshold (Critic 1 CRITICAL concern)
- CEBPB elevated from "exploratory" to "primary" for FAP (Critic 1 recommendation)
- KLF10 added for FAP (Critic 1 recommendation)
- Baseline = ≥1 cell (canonical analysis threshold)

OUTPUTS:
  i3_sensitivity_results.csv    — rho, p, N, country_composition per threshold
  a3_qc_metrics.csv             — QC summary per dataset × age_group
  a2_donor_table.csv            — donor/cell counts per compartment × age × tech × country
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import norm
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# Constants
# ============================================================================

SASP12_GENES = ['CCL2', 'CCL7', 'CCL20', 'CXCL6', 'CXCL8', 'IL6',
                'MMP1', 'MMP3', 'SERPINE1', 'IGFBP2', 'IGFBP3', 'IGFBP5']

TARGET_TFS = ['JUNB', 'CDKN1A', 'CEBPB', 'KLF10']

THRESHOLDS = [1, 10, 50, 100, 200, 500]
# Note: batch_051 used [0, 50, 75, 100, 150]. We extend for consistency + CEBPB/KLF10.

DATA_FILES = {
    'Vascular': 'data/Vascular_scsn_RNA.h5ad',
    'MuSC': 'data/MuSC_scsn_RNA.h5ad',
    'FAP': 'data/OMIX004308-02.h5ad',
}

# Cell type filters to match canonical analysis
CELL_FILTERS = {
    'Vascular': ['CapEC', 'VenEC', 'IL6+ VenEC', 'ArtEC'],  # Endothelial only
    'MuSC': None,  # All MuSC subtypes
    'FAP': ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP', 'CD99+ FAP'],  # Exclude tenocytes
}

OUTDIR = Path('experiments/batch_056')
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Utility Functions
# ============================================================================

def get_expressed_values(adata, gene):
    """Extract gene expression values handling both sparse and dense matrices."""
    import scipy.sparse as sp
    col = adata[:, gene].X
    if sp.issparse(col):
        return np.asarray(col.todense()).flatten()
    return np.asarray(col).flatten()


def compute_donor_correlation(adata, tf, sasp_genes, min_cells=1):
    """
    Compute donor-level Spearman correlation between TF expression and SASP composite.
    Returns dict with rho, p, N, and donor-level metadata.
    """
    # Get expressed genes
    available_sasp = [g for g in sasp_genes if g in adata.var_names]
    available_tf = tf if tf in adata.var_names else None

    if available_tf is None or len(available_sasp) < 3:
        return None

    # Compute per-cell values
    import scipy.sparse as sp
    tf_vals = get_expressed_values(adata, available_tf)
    sasp_mat = adata[:, available_sasp].X
    if sp.issparse(sasp_mat):
        sasp_vals = np.asarray(sasp_mat.todense()).mean(axis=1).flatten()
    else:
        sasp_vals = np.asarray(sasp_mat).mean(axis=1).flatten()

    # Build donor-level table
    donors = adata.obs['sample'].values
    unique_donors = np.unique(donors)

    donor_data = []
    for d in unique_donors:
        mask = donors == d
        n_cells = mask.sum()
        if n_cells < min_cells:
            continue
        donor_data.append({
            'donor': d,
            'n_cells': n_cells,
            'tf_mean': tf_vals[mask].mean(),
            'sasp_mean': sasp_vals[mask].mean(),
        })

    if len(donor_data) < 10:  # Raised from 5 to 10 per Critic 2 (statistical adversary)
        return None

    df = pd.DataFrame(donor_data)

    # Spearman correlation
    rho, p = stats.spearmanr(df['tf_mean'], df['sasp_mean'])

    # Fisher Z 95% CI
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1 / np.sqrt(len(df) - 3)
    ci_lo = np.tanh(z - 1.96 * se)
    ci_hi = np.tanh(z + 1.96 * se)

    return {
        'tf': tf,
        'n_donors': len(df),
        'rho': rho,
        'p': p,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'min_cells': min_cells,
        'donor_data': df,
    }


def get_donor_composition(adata, min_cells=1):
    """Get country and age composition of donors at a given threshold."""
    donor_info = adata.obs.groupby('sample').agg(
        n_cells=('sample', 'size'),
        Country=('Country', 'first'),
        age_pop=('age_pop', 'first'),
        tech=('tech', 'first'),
    ).reset_index()

    subset = donor_info[donor_info['n_cells'] >= min_cells]
    n_total = len(subset)
    n_china = (subset['Country'] == 'China').sum()
    n_spain = (subset['Country'] == 'Spain').sum()
    n_young = (subset['age_pop'] == 'young_pop').sum()
    n_old = (subset['age_pop'] == 'old_pop').sum()

    return {
        'n_total': n_total,
        'n_china': n_china,
        'n_spain': n_spain,
        'frac_china': n_china / n_total if n_total > 0 else 0,
        'frac_spain': n_spain / n_total if n_total > 0 else 0,
        'n_young': n_young,
        'n_old': n_old,
    }


# ============================================================================
# Main Analysis
# ============================================================================

def run_i3_sensitivity():
    """Run I3 cell count threshold sensitivity analysis."""
    print("=" * 70)
    print("I3: Cell Count Threshold Sensitivity Analysis")
    print("=" * 70)

    results = []

    for comp_name, file_path in DATA_FILES.items():
        print(f"\n--- {comp_name} ---")
        import anndata as ad
        adata = ad.read_h5ad(file_path)
        print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

        # Filter to correct cell types (matching canonical analysis)
        if CELL_FILTERS.get(comp_name) is not None:
            mask = adata.obs['Annotation'].isin(CELL_FILTERS[comp_name])
            adata = adata[mask]
            print(f"After cell type filter: {adata.shape[0]} cells")

        # Select TFs per compartment
        if comp_name == 'FAP':
            tfs = TARGET_TFS  # JUNB, CDKN1A, CEBPB, KLF10
        else:
            tfs = ['JUNB', 'CDKN1A']  # Primary TFs for Vascular/MuSC

        for threshold in THRESHOLDS:
            # Get composition at this threshold
            comp = get_donor_composition(adata, threshold)
            print(f"\n  Threshold ≥{threshold}: N={comp['n_total']}, "
                  f"China={comp['n_china']}, Spain={comp['n_spain']}, "
                  f"Young={comp['n_young']}, Old={comp['n_old']}")

            for tf in tfs:
                result = compute_donor_correlation(adata, tf, SASP12_GENES, min_cells=threshold)
                if result is None:
                    print(f"    {tf}: INSUFFICIENT (N < 10)")
                    continue

                row = {
                    'compartment': comp_name,
                    'tf': tf,
                    'threshold': threshold,
                    'n_donors': result['n_donors'],
                    'rho': result['rho'],
                    'p': result['p'],
                    'ci_lo': result['ci_lo'],
                    'ci_hi': result['ci_hi'],
                    'n_china': comp['n_china'],
                    'n_spain': comp['n_spain'],
                    'frac_china': comp['frac_china'],
                    'n_young': comp['n_young'],
                    'n_old': comp['n_old'],
                }
                results.append(row)
                print(f"    {tf}: rho={result['rho']:.3f}, p={result['p']:.4f}, "
                      f"N={result['n_donors']}, CI=[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}]")

    # Save results
    df = pd.DataFrame(results)
    outpath = OUTDIR / 'i3_sensitivity_results.csv'
    df.to_csv(outpath, index=False)
    print(f"\nSaved: {outpath}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Primary findings across thresholds")
    print("=" * 70)

    for comp_name in DATA_FILES.keys():
        print(f"\n{comp_name}:")
        comp_df = df[df['compartment'] == comp_name]
        for tf in comp_df['tf'].unique():
            tf_df = comp_df[comp_df['tf'] == tf]
            baseline = tf_df[tf_df['threshold'] == 1]
            if len(baseline) == 0:
                continue
            baseline_rho = baseline['rho'].values[0]
            print(f"\n  {tf} (baseline rho={baseline_rho:.3f}):")
            for _, row in tf_df.iterrows():
                delta = row['rho'] - baseline_rho
                country_shift = ""
                if row['n_spain'] == 0 and row['threshold'] > 1:
                    country_shift = " [ALL CHINA]"
                elif row['frac_china'] > 0.75 and row['threshold'] > 1:
                    country_shift = f" [China-heavy {row['frac_china']:.0%}]"
                print(f"    ≥{row['threshold']:3d}: rho={row['rho']:.3f} "
                      f"(Δ={delta:+.3f}), p={row['p']:.4f}, N={row['n_donors']}"
                      f"{country_shift}")

    return df


def run_a3_qc():
    """Generate A3 QC metrics summary table."""
    print("\n" + "=" * 70)
    print("A3: QC Metrics Summary")
    print("=" * 70)

    results = []

    for comp_name, file_path in DATA_FILES.items():
        import anndata as ad
        adata = ad.read_h5ad(file_path)

        # Check available QC columns
        qc_cols = {}
        if 'nCount_RNA' in adata.obs.columns:
            qc_cols['nCount_RNA'] = 'nCount_RNA'
        if 'nFeature_RNA' in adata.obs.columns:
            qc_cols['nFeature_RNA'] = 'nFeature_RNA'
        if 'percent.mt' in adata.obs.columns:
            qc_cols['percent_mito'] = 'percent.mt'

        for age_group in ['young_pop', 'old_pop']:
            mask = adata.obs['age_pop'] == age_group
            if mask.sum() == 0:
                continue
            subset = adata.obs[mask]
            for qc_name, col in qc_cols.items():
                results.append({
                    'compartment': comp_name,
                    'age_group': age_group.replace('_pop', ''),
                    'n_cells': mask.sum(),
                    'n_donors': subset['sample'].nunique(),
                    'metric': qc_name,
                    'mean': subset[col].mean(),
                    'median': subset[col].median(),
                    'sd': subset[col].std(),
                    'min': subset[col].min(),
                    'max': subset[col].max(),
                })

    # Also add NA Endothelium
    try:
        import anndata as ad
        na = ad.read_h5ad('data/NA_Endothelium_SMC.h5ad')
        # Filter to endothelial only
        if 'annotation_level2' in na.obs.columns:
            ec = na[na.obs['annotation_level2'].str.contains('Endothelial', case=False, na=False)]
        else:
            ec = na

        for age_group in na.obs['Age_group'].unique():
            mask = ec.obs['Age_group'] == age_group
            if mask.sum() == 0:
                continue
            subset = ec.obs[mask]
            for qc_name, col in [('n_counts', 'n_counts'), ('n_genes', 'n_genes'), ('percent_mito', 'percent_mito')]:
                if col in subset.columns:
                    results.append({
                        'compartment': 'NA_Endothelium',
                        'age_group': age_group,
                        'n_cells': mask.sum(),
                        'n_donors': subset['donor_id'].nunique() if 'donor_id' in subset.columns else subset['SampleID'].nunique(),
                        'metric': qc_name,
                        'mean': pd.to_numeric(subset[col], errors='coerce').mean(),
                        'median': pd.to_numeric(subset[col], errors='coerce').median(),
                        'sd': pd.to_numeric(subset[col], errors='coerce').std(),
                        'min': pd.to_numeric(subset[col], errors='coerce').min(),
                        'max': pd.to_numeric(subset[col], errors='coerce').max(),
                    })
    except Exception as e:
        print(f"Warning: Could not process NA Endothelium: {e}")

    df = pd.DataFrame(results)
    outpath = OUTDIR / 'a3_qc_metrics.csv'
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath}")
    print(df.to_string(index=False))

    return df


def run_a2_donor_table():
    """Generate A2 complete donor/cell count table."""
    print("\n" + "=" * 70)
    print("A2: Donor/Cell Count Table")
    print("=" * 70)

    results = []

    for comp_name, file_path in DATA_FILES.items():
        import anndata as ad
        adata = ad.read_h5ad(file_path)

        donor_info = adata.obs.groupby('sample').agg(
            n_cells=('sample', 'size'),
            Country=('Country', 'first'),
            age_pop=('age_pop', 'first'),
            tech=('tech', 'first'),
            Sex=('Sex' if 'Sex' in adata.obs.columns else 'gender', 'first'),
            age=('age', 'first'),
        ).reset_index()

        for _, row in donor_info.iterrows():
            results.append({
                'compartment': comp_name,
                'donor_id': row['sample'],
                'n_cells': row['n_cells'],
                'age': row['age'],
                'age_group': row['age_pop'].replace('_pop', ''),
                'sex': row['Sex'] if isinstance(row['Sex'], str) else row.get('gender', ''),
                'country': row['Country'],
                'tech': row['tech'],
            })

    df = pd.DataFrame(results)
    outpath = OUTDIR / 'a2_donor_table.csv'
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath}")

    # Summary
    for comp in df['compartment'].unique():
        comp_df = df[df['compartment'] == comp]
        print(f"\n{comp}: {comp_df['n_cells'].sum()} cells, {len(comp_df)} donors")
        pivot = comp_df.groupby(['age_group', 'tech'])['n_cells'].agg(['sum', 'count'])
        pivot.columns = ['total_cells', 'n_donors']
        print(pivot.to_string())

    return df


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    print("batch_056: Preprint Robustness Checks")
    print("=" * 70)

    # I3: Cell count sensitivity
    i3_df = run_i3_sensitivity()

    # A3: QC metrics
    a3_df = run_a3_qc()

    # A2: Donor table
    a2_df = run_a2_donor_table()

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
