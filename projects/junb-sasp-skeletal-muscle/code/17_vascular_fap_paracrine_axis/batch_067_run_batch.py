#!/usr/bin/env python3
"""
batch_067: Cross-compartment SASP composition + Sex×compartment interaction

Design: batch_067/brief.md v2 (post 3-critic review)
- PRIMARY: Cross-compartment SASP composition (vascular, MuSC, FAP, immune)
- SECONDARY: Sex×compartment interaction in non-vascular compartments
- SCENIC+: Deferred (snATAC truncated, scenicplus not installed; F054_01 already ESTABLISHED)

Note: SCENIC+ (PI #2) deferred due to:
1. snATAC file truncated (expected 8.4GB, got 4.9GB)
2. scenicplus not installed (would require dedicated conda environment)
3. F054_01 (AUCell rho=0.923) already established pySCENIC result

This batch focuses on items that CAN be executed with current infrastructure.
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.stats import spearmanr, norm as stats_norm

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "batch_067"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# SASP12 genes (literature-curated, F047-052)
SASP12_GENES = [
    "CCL2", "CCL7", "CCL20", "CXCL6", "CXCL8",  # Chemokines
    "IL6",  # Cytokine
    "MMP1", "MMP3",  # Matrix remodeling
    "SERPINE1",  # Serpin
    "IGFBP2", "IGFBP3", "IGFBP5"  # IGFBPs
]

# Pre-specified TFs from established findings
PRE_SPECIFIED_TFS = ["JUNB", "FOS", "CEBPB"]  # F084, F047, F060_02

# Key thresholds
SASP_DELTA_THRESHOLD = 0.3  # For compartment-specific loadings
SEX_DELTA_THRESHOLD = 0.15  # For sex×compartment interaction


def load_dataset(name, path):
    """Load a dataset with correct column names."""
    print(f"  Loading {name} from {path.name}...")
    adata = sc.read_h5ad(path)

    # Find donor column
    donor_col = None
    for col in ['sample', 'donor', 'Donor', 'Subject']:
        if col in adata.obs.columns:
            donor_col = col
            break

    if donor_col:
        n_donors = adata.obs[donor_col].nunique()
        print(f"    {name}: {adata.n_obs} cells, {n_donors} donors, col={donor_col}")
    else:
        print(f"    {name}: {adata.n_obs} cells, donor_col not found")

    return adata


def compute_donor_sasp(adata, gene_list, donor_col='sample'):
    """Compute mean SASP12 score per donor."""
    available_genes = [g for g in gene_list if g in adata.var_names]
    if not available_genes:
        return pd.Series(dtype=float)

    # Get expression matrix
    if 'raw' in adata.layers:
        X = adata.layers['raw'][:, adata.var_names.get_indexer(available_genes)]
    else:
        X = adata.X[:, adata.var_names.get_indexer(available_genes)]

    # Log1p normalize
    from scipy.sparse import issparse
    if issparse(X):
        X_log = np.log1p(X.toarray())
    else:
        X_log = np.log1p(np.array(X))

    sasp_score = X_log.mean(axis=1)

    donor_df = pd.DataFrame({
        'cell': adata.obs_names,
        'donor': adata.obs[donor_col].values,
        'sasp': sasp_score
    })

    # Mean per donor
    donor_sasp = donor_df.groupby('donor')['sasp'].mean()
    return donor_sasp


def compute_donor_tf(adata, tf_name, donor_col='sample'):
    """Compute mean TF expression per donor."""
    if tf_name not in adata.var_names:
        return pd.Series(dtype=float)

    if 'raw' in adata.layers:
        X = adata.layers['raw'][:, adata.var_names.get_indexer([tf_name])]
    else:
        X = adata.X[:, adata.var_names.get_indexer([tf_name])]

    if issparse(X):
        X = X.toarray().flatten()
    else:
        X = np.array(X).flatten()

    donor_df = pd.DataFrame({
        'cell': adata.obs_names,
        'donor': adata.obs[donor_col].values,
        'tf': X
    })

    donor_tf = donor_df.groupby('donor')['tf'].mean()
    return donor_tf


def bootstrap_ci(series1, series2, n_bootstrap=10000, alpha=0.05):
    """Compute CI for Spearman correlation using Fisher z transformation."""
    data = pd.DataFrame({'x': series1, 'y': series2}).dropna()
    n = len(data)

    if n < 3:
        return (np.nan, np.nan, np.nan)

    rho, p = spearmanr(data['x'], data['y'])
    # Fisher z CI
    se = 1.0 / np.sqrt(n - 3)
    z = np.arctanh(rho)
    z_alpha = stats_norm.ppf(1 - alpha / 2)
    ci_z_low = z - z_alpha * se
    ci_z_high = z + z_alpha * se
    ci_low = np.tanh(ci_z_low)
    ci_high = np.tanh(ci_z_high)

    return (rho, ci_low, ci_high)


def load_all_datasets():
    """Load all HLMA compartments."""
    print("=" * 60)
    print("LOADING HLMA DATA")
    print("=" * 60)

    datasets = {}
    files = {
        "vascular": DATA_DIR / "Vascular_scsn_RNA.h5ad",
        "musc": DATA_DIR / "MuSC_scsn_RNA.h5ad",
        "fap": DATA_DIR / "OMIX004308-02.h5ad",
        "immune": DATA_DIR / "Immune_scsn_RNA.h5ad",
    }

    for name, path in files.items():
        if path.exists():
            datasets[name] = load_dataset(name, path)
        else:
            print(f"  WARNING: {name} not found at {path}")

    return datasets


def run_cross_compartment_sasp(datasets):
    """H2: Analyze SASP composition across compartments."""
    print("\n" + "=" * 60)
    print("H2: CROSS-COMPARTMENT SASP COMPOSITION")
    print("=" * 60)

    results = {}

    # Step 1: Compute SASP per compartment
    compartment_sasp = {}
    for name, adata in datasets.items():
        sasp = compute_donor_sasp(adata, SASP12_GENES)
        compartment_sasp[name] = sasp
        results[f'{name}_n_donors'] = len(sasp)
        results[f'{name}_sasp_mean'] = float(sasp.mean()) if len(sasp) > 0 else np.nan
        results[f'{name}_sasp_std'] = float(sasp.std()) if len(sasp) > 0 else np.nan
        print(f"  {name}: mean SASP={sasp.mean():.3f}, std={sasp.std():.3f}, N={len(sasp)}")

    # Step 2: Compare per-gene SASP factor loadings
    print("\n  Analyzing SASP factor composition by compartment...")

    gene_loadings = {}
    for name, adata in datasets.items():
        available_genes = [g for g in SASP12_GENES if g in adata.var_names]
        if not available_genes:
            continue

        if 'raw' in adata.layers:
            X = adata.layers['raw'][:, adata.var_names.get_indexer(available_genes)]
        else:
            X = adata.X[:, adata.var_names.get_indexer(available_genes)]

        if issparse(X):
            X_log = np.log1p(X.toarray())
        else:
            X_log = np.log1p(np.array(X))

        # Per-gene mean (cell-level)
        gene_means = pd.Series(X_log.mean(axis=0), index=available_genes)
        gene_loadings[name] = gene_means

    # Step 3: Compare vascular vs FAP loadings (two-compartment thesis)
    if 'vascular' in gene_loadings and 'fap' in gene_loadings:
        vasc = gene_loadings['vascular']
        fap = gene_loadings['fap']
        common_genes = vasc.index.intersection(fap.index)

        delta = (vasc[common_genes] - fap[common_genes]).abs()
        n_high_delta = (delta > SASP_DELTA_THRESHOLD).sum()
        max_delta = delta.max()
        mean_delta = delta.mean()

        results['vasc_fap_n_genes'] = len(common_genes)
        results['vasc_fap_n_high_delta'] = int(n_high_delta)
        results['vasc_fap_max_delta'] = float(max_delta)
        results['vasc_fap_mean_delta'] = float(mean_delta)

        print(f"\n  Vascular vs FAP SASP composition:")
        print(f"    Common genes: {len(common_genes)}")
        print(f"    Genes with delta > {SASP_DELTA_THRESHOLD}: {n_high_delta}")
        print(f"    Max delta: {max_delta:.3f}")
        print(f"    Mean delta: {mean_delta:.3f}")

        # Detailed per-gene comparison
        delta_df = pd.DataFrame({
            'vascular': vasc,
            'fap': fap,
            'delta': delta
        }).sort_values('delta', ascending=False)

        print(f"\n  Top 6 genes by loading difference:")
        for gene in delta_df.head(6).index:
            v = delta_df.loc[gene, 'vascular']
            f = delta_df.loc[gene, 'fap']
            d = delta_df.loc[gene, 'delta']
            higher = 'vascular' if v > f else 'fap'
            print(f"    {gene}: vascular={v:.3f}, fap={f:.3f}, delta={d:.3f} ({higher} higher)")

        # Save to CSV
        delta_df.to_csv(RESULTS_DIR / "sasp_factor_vasc_fap.csv")
        results['sasp_factor_csv'] = str(RESULTS_DIR / "sasp_factor_vasc_fap.csv")

        # Decision: compartment-specific or uniform?
        if n_high_delta >= 3:
            results['compartment_specific'] = True
            results['verdict'] = 'COMPARTMENT-SPECIFIC (supports two-compartment strategy)'
            print(f"\n  VERDICT: Compartment-specific SASP composition")
            print(f"    → Supports two-compartment therapeutic strategy (VERA novelty H2)")
        else:
            results['compartment_specific'] = False
            results['verdict'] = 'UNIFORM (single anti-SASP approach valid)'
            print(f"\n  VERDICT: Uniform SASP composition")
            print(f"    → Single anti-SASP approach valid")

    # Step 4: Compare immune vs vascular (F063_06: immune leads SASP_high_fraction)
    if 'immune' in gene_loadings and 'vascular' in gene_loadings:
        immune = gene_loadings['immune']
        vasc = gene_loadings['vascular']
        common_genes = immune.index.intersection(vasc.index)

        delta_immune_vasc = (immune[common_genes] - vasc[common_genes]).abs()

        print(f"\n  Immune vs Vascular SASP composition:")
        print(f"    Common genes: {len(common_genes)}")
        print(f"    Max delta: {delta_immune_vasc.max():.3f}")

        # Check if immune has higher overall SASP
        immune_mean = immune[common_genes].mean()
        vasc_mean = vasc[common_genes].mean()
        results['immune_vasc_diff'] = float(immune_mean - vasc_mean)
        print(f"    Immune mean: {immune_mean:.3f}, Vascular mean: {vasc_mean:.3f}")
        print(f"    Difference: {immune_mean - vasc_mean:.3f} ({'immune higher' if immune_mean > vasc_mean else 'vascular higher'})")

    return results


def run_sex_compartment_interaction(datasets):
    """H3: Test sex×compartment interaction in non-vascular compartments."""
    print("\n" + "=" * 60)
    print("H3: SEX×COMPARTMENT INTERACTION")
    print("=" * 60)

    results = {}
    compartments = ["musc", "fap", "immune"]
    tfs = PRE_SPECIFIED_TFS

    # Check for sex metadata
    for name, adata in datasets.items():
        obs_cols = list(adata.obs.columns)
        sex_candidates = [c for c in obs_cols if c.lower() in ['sex', 'gender', 'male', 'female']]
        has_sex = len(sex_candidates) > 0
        results[f'{name}_has_sex_metadata'] = has_sex
        if has_sex:
            print(f"  {name}: sex metadata found (col={sex_candidates[0]})")
            sex_vals = adata.obs[sex_candidates[0]].value_counts()
            print(f"    {dict(sex_vals)}")
        else:
            print(f"  {name}: NO sex metadata")

    # Compute within-sex correlations for each compartment
    print("\n  Computing within-sex TF-SASP correlations...")

    all_corrs = {}

    for comp in compartments:
        if comp not in datasets:
            continue

        adata = datasets[comp]

        # Find sex column
        sex_col = None
        for col in ['sex', 'Sex', 'gender', 'Gender']:
            if col in adata.obs.columns:
                sex_col = col
                break

        if not sex_col:
            print(f"\n  {comp.upper()}: No sex metadata, skipping")
            continue

        # Find donor column
        donor_col = None
        for col in ['sample', 'donor', 'Donor', 'Subject']:
            if col in adata.obs.columns:
                donor_col = col
                break

        if not donor_col:
            continue

        print(f"\n  {comp.upper()}:")

        for tf in tfs:
            if tf not in adata.var_names:
                continue

            # Compute donor-level TF and SASP
            tf_expr = compute_donor_tf(adata, tf, donor_col)
            sasp = compute_donor_sasp(adata, SASP12_GENES, donor_col)

            # Align on donors
            common_donors = tf_expr.index.intersection(sasp.index)
            if len(common_donors) < 5:
                continue

            tf_common = tf_expr[common_donors]
            sasp_common = sasp[common_donors]

            # Split by sex
            for sex in adata.obs[sex_col].unique():
                sex_donors = adata.obs[adata.obs[sex_col] == sex][donor_col].unique()
                sex_donors = [d for d in sex_donors if d in common_donors]

                if len(sex_donors) < 3:
                    continue

                tf_sex = tf_common[sex_donors]
                sasp_sex = sasp_common[sex_donors]

                rho, p = spearmanr(tf_sex, sasp_sex)

                key = f"{comp}_{tf}_{sex}"
                all_corrs[key] = {"rho": float(rho), "p": float(p), "n": len(sex_donors)}

                print(f"    {tf} ({sex}): rho={rho:.3f}, p={p:.4f}, N={len(sex_donors)}")

    results['correlations'] = all_corrs

    # Summary: sex×compartment interaction
    print("\n  Sex×compartment summary:")
    interaction_results = []

    for comp in compartments:
        for tf in tfs:
            male_key = f"{comp}_{tf}_male"
            female_key = f"{comp}_{tf}_female"

            if male_key in all_corrs and female_key in all_corrs:
                male_rho = all_corrs[male_key]['rho']
                female_rho = all_corrs[female_key]['rho']
                delta = abs(male_rho - female_rho)
                n_m = all_corrs[male_key]['n']
                n_f = all_corrs[female_key]['n']

                interaction_results.append({
                    'compartment': comp,
                    'tf': tf,
                    'male_rho': male_rho,
                    'female_rho': female_rho,
                    'delta': delta,
                    'n_male': n_m,
                    'n_female': n_f
                })

                if delta > SEX_DELTA_THRESHOLD:
                    print(f"    {comp}_{tf}: Male rho={male_rho:.3f}, Female rho={female_rho:.3f}, delta={delta:.3f} INTERACTION")
                else:
                    print(f"    {comp}_{tf}: Male rho={male_rho:.3f}, Female rho={female_rho:.3f}, delta={delta:.3f} no interaction")

    # Check for any interactions above threshold
    n_interactions = sum(1 for r in interaction_results if r['delta'] > SEX_DELTA_THRESHOLD)
    results['n_interactions_above_threshold'] = n_interactions
    results['sex_interaction_detected'] = n_interactions > 0

    if n_interactions > 0:
        results['verdict'] = 'INTERACTION DETECTED (sex modifies TF-SASP coupling in some compartments)'
    else:
        results['verdict'] = 'NO SEX INTERACTION (sex-independent across all non-vascular compartments)'

    print(f"\n  VERDICT: {results['verdict']}")

    return results


def main():
    """Main execution."""
    print("=" * 60)
    print("BATCH_067: Cross-compartment SASP + Sex×compartment")
    print("=" * 60)
    print("Date: 2026-04-23")
    print("Note: SCENIC+ deferred (snATAC truncated + scenicplus not installed)")
    print("      F054_01 (AUCell rho=0.923) already ESTABLISHED")
    print("=" * 60)

    results = {
        "batch": "batch_067",
        "date": "2026-04-23",
        "note": "SCENIC+ deferred; F054_01 already established"
    }

    # Load data
    datasets = load_all_datasets()

    # H2: Cross-compartment SASP
    print("\n" + "=" * 60)
    print("H2: CROSS-COMPARTMENT SASP")
    print("=" * 60)
    h2_results = run_cross_compartment_sasp(datasets)
    results['h2_cross_compartment_sasp'] = h2_results

    # H3: Sex×compartment interaction
    print("\n" + "=" * 60)
    print("H3: SEX×COMPARTMENT")
    print("=" * 60)
    h3_results = run_sex_compartment_interaction(datasets)
    results['h3_sex_compartment'] = h3_results

    # Save results
    results_file = RESULTS_DIR / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n" + "=" * 60)
    print(f"RESULTS SAVED: {results_file}")
    print("=" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    h2_status = "COMPARTMENT-SPECIFIC" if h2_results.get('compartment_specific', False) else "UNIFORM"
    print(f"  H2 (SASP composition): {h2_status}")
    print(f"    {h2_results.get('verdict', 'N/A')}")

    h3_status = "INTERACTION DETECTED" if h3_results.get('sex_interaction_detected', False) else "NO INTERACTION"
    print(f"  H3 (Sex×compartment): {h3_status}")
    print(f"    {h3_results.get('verdict', 'N/A')}")

    return results


if __name__ == "__main__":
    from scipy.sparse import issparse
    main()