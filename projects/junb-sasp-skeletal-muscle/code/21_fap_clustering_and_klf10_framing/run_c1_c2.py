#!/usr/bin/env python3
"""
C1: FAP Subtype Proportion Analysis
C2: snRNA-Only Subclustering (Vascular, FAP, MuSC)

Batch 053 -- SM-RD project
"""

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import warnings
warnings.filterwarnings('ignore')

import gc
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- Configuration ----------
BASE_DIR = '/home/yuanz/Documents/GitHub/biomarvin_fibro'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'batch_053')
os.makedirs(OUT_DIR, exist_ok=True)

# Reproducibility: set seeds
np.random.seed(42)
sc.settings.seed = 42

SASP12_GENES = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8',
                'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

FAP_SUBTYPES = ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP', 'CD99+ FAP']

VASCULAR_SUBTYPES = ['ArtEC', 'CapEC', 'VenEC', 'IL6+ VenEC']

LEIDEN_RESOLUTIONS = [0.3, 0.5, 0.8]


# ==================== Helpers ====================

def get_gene_expression(adata, gene_name):
    """Extract expression vector for a gene from adata.X (handles sparse/dense)."""
    if gene_name not in adata.var_names:
        return np.zeros(adata.n_obs)
    idx = list(adata.var_names).index(gene_name)
    if sparse.issparse(adata.X):
        return np.asarray(adata.X[:, idx].todense()).ravel()
    else:
        return np.asarray(adata.X[:, idx]).ravel()


def compute_sasp12_composite(adata):
    """Compute mean of detected SASP12 genes per cell."""
    detected_genes = [g for g in SASP12_GENES if g in adata.var_names]
    if not detected_genes:
        return np.zeros(adata.n_obs)
    gene_indices = [list(adata.var_names).index(g) for g in detected_genes]
    if sparse.issparse(adata.X):
        expr = np.asarray(adata.X[:, gene_indices].todense())
    else:
        expr = np.asarray(adata.X[:, gene_indices])
    return expr.mean(axis=1)


def pick_best_resolution(adata, resolutions):
    """Run Leiden at multiple resolutions, pick the one with 3-8 clusters preferring 4-6.

    If no resolution in the initial list yields 3-8 clusters, extend the search
    to progressively lower resolutions (0.2, 0.1, 0.05) until a suitable
    cluster count is found. This handles datasets where Leiden over-segments.
    """
    best_res = None
    best_n = None
    best_score = float('inf')

    # Try the provided resolutions first
    for res in resolutions:
        key = f'leiden_{res}'
        sc.tl.leiden(adata, resolution=res, key_added=key)
        n_clusters = adata.obs[key].nunique()
        print(f"    Resolution {res}: {n_clusters} clusters")

        if 3 <= n_clusters <= 8:
            score = abs(n_clusters - 5)
            if score < best_score:
                best_score = score
                best_res = res
                best_n = n_clusters

    # If none in range, try progressively lower resolutions
    if best_res is None:
        print("    No resolution yielded 3-8 clusters, extending to lower resolutions...")
        extra_resolutions = [0.2, 0.1, 0.05, 0.02]
        for res in extra_resolutions:
            key = f'leiden_{res}'
            sc.tl.leiden(adata, resolution=res, key_added=key)
            n_clusters = adata.obs[key].nunique()
            print(f"    Resolution {res}: {n_clusters} clusters")

            if 3 <= n_clusters <= 8:
                score = abs(n_clusters - 5)
                if score < best_score:
                    best_score = score
                    best_res = res
                    best_n = n_clusters
                break  # Found a suitable resolution, stop

    if best_res is None:
        # Final fallback: closest to 5 clusters across all tried resolutions
        print("    WARNING: No resolution yielded 3-8 clusters, using closest to 5")
        all_resolutions = resolutions + [0.2, 0.1, 0.05, 0.02]
        for res in all_resolutions:
            key = f'leiden_{res}'
            if key in adata.obs.columns:
                n_clusters = adata.obs[key].nunique()
                score = abs(n_clusters - 5)
                if score < best_score:
                    best_score = score
                    best_res = res
                    best_n = n_clusters

    return best_res, best_n


def detect_hvg_flavor(adata):
    """Detect whether data is log-normalized (use 'seurat') or raw counts (use 'seurat_v3')."""
    if sparse.issparse(adata.X):
        x_sample = adata.X[:100].toarray()
    else:
        x_sample = np.asarray(adata.X[:100])
    has_negatives = x_sample.min() < 0
    flavor = 'seurat' if has_negatives else 'seurat_v3'
    print(f"    Data range sample: [{x_sample.min():.2f}, {x_sample.max():.2f}], HVG flavor: {flavor}")
    return flavor


def run_de_per_cluster(adata, min_donors_per_group=3):
    """Run DE (old vs young) per cluster using scanpy's rank_genes_groups."""
    de_results = {}
    for cl in sorted(adata.obs['cluster'].unique()):
        cl_mask = adata.obs['cluster'] == cl
        cl_adata = adata[cl_mask].copy()

        if 'age_group' not in cl_adata.obs.columns:
            continue
        young_donors = cl_adata.obs[cl_adata.obs['age_group'] == 'young']['sample'].nunique()
        old_donors = cl_adata.obs[cl_adata.obs['age_group'] == 'old']['sample'].nunique()

        if young_donors < min_donors_per_group or old_donors < min_donors_per_group:
            de_results[cl] = f"Insufficient donors (young={young_donors}, old={old_donors})"
            continue

        try:
            sc.tl.rank_genes_groups(
                cl_adata, groupby='age_group', groups=['old'], reference='young',
                method='wilcoxon', key_added='de_age'
            )
            de_df = sc.get.rank_genes_groups_df(cl_adata, group='old', key='de_age')
            top10 = de_df.head(10)[['names', 'scores', 'pvals', 'logfoldchanges']].copy()
            top10.columns = ['gene', 'score', 'pval', 'log2fc']
            de_results[cl] = top10
        except Exception as e:
            de_results[cl] = f"DE failed: {str(e)}"

    return de_results


def save_de_results(de_results, out_path):
    """Save DE results to CSV."""
    rows = []
    for cl, result in de_results.items():
        if isinstance(result, str):
            rows.append({'cluster': cl, 'gene': result, 'score': np.nan,
                         'pval': np.nan, 'log2fc': np.nan})
        else:
            for _, row in result.iterrows():
                rows.append({'cluster': cl, 'gene': row['gene'],
                             'score': round(row['score'], 3),
                             'pval': row['pval'], 'log2fc': round(row['log2fc'], 3)})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  DE results saved to {out_path}")


def plot_umap_4panel(adata, compartment_name, out_path):
    """Plot 4-panel UMAP: cluster, age_group, JUNB, SASP12."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'{compartment_name} -- Subclustering UMAP', fontsize=14, y=0.98)

    sc.pl.umap(adata, color='cluster', ax=axes[0, 0], show=False, title='Cluster')
    axes[0, 0].set_xlabel('UMAP1'); axes[0, 0].set_ylabel('UMAP2')

    sc.pl.umap(adata, color='age_group', ax=axes[0, 1], show=False,
               title='Age Group', palette={'young': '#4CAF50', 'old': '#F44336'})
    axes[0, 1].set_xlabel('UMAP1'); axes[0, 1].set_ylabel('UMAP2')

    sc.pl.umap(adata, color='JUNB_expr', ax=axes[1, 0], show=False,
               title='JUNB Expression', cmap='Reds', vmin=0)
    axes[1, 0].set_xlabel('UMAP1'); axes[1, 0].set_ylabel('UMAP2')

    sc.pl.umap(adata, color='SASP12_composite', ax=axes[1, 1], show=False,
               title='SASP12 Composite', cmap='Oranges', vmin=0)
    axes[1, 1].set_xlabel('UMAP1'); axes[1, 1].set_ylabel('UMAP2')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  UMAP saved to {out_path}")


# ==================== C1: FAP Subtype Proportions ====================

def run_c1(fap_adata):
    """C1: Compute FAP subtype proportions per donor, test age association."""
    print("\n" + "=" * 60)
    print("C1: FAP Subtype Proportion Analysis")
    print("=" * 60)

    # Work with obs only (no expression data needed)
    obs = fap_adata.obs.copy()

    # Filter to FAP subtypes only
    obs = obs[obs['Annotation'].isin(FAP_SUBTYPES)].copy()
    print(f"Total FAP subtype cells: {len(obs)}")
    print(f"Subtype distribution:\n{obs['Annotation'].value_counts()}")

    # Convert categorical columns to plain strings to avoid category expansion
    obs['Annotation'] = obs['Annotation'].astype(str)
    obs['sample'] = obs['sample'].astype(str)
    obs['age'] = obs['age'].astype(str)
    obs['age_pop'] = obs['age_pop'].astype(str)
    obs['Country'] = obs['Country'].astype(str)

    # Convert age to numeric
    obs['age_num'] = pd.to_numeric(obs['age'], errors='coerce')

    # Compute per-donor subtype counts
    donor_info = (
        obs
        .groupby('sample', observed=True)
        .agg(
            age_num=('age_num', 'first'),
            age_pop=('age_pop', 'first'),
            Country=('Country', 'first'),
            n_cells=('Annotation', 'size'),
        )
        .reset_index()
    )

    # Filter: minimum 50 cells per donor
    donor_info = donor_info[donor_info['n_cells'] >= 50].copy()
    eligible_donors = donor_info['sample'].tolist()
    print(f"\nEligible donors (>= 50 cells): {len(eligible_donors)}")

    # Filter obs to eligible donors only
    eligible_obs = obs[obs['sample'].isin(eligible_donors)].copy()

    # Compute subtype proportions per eligible donor
    donor_subtype_counts = (
        eligible_obs
        .groupby(['sample', 'Annotation'], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    donor_subtype_props = donor_subtype_counts.div(donor_subtype_counts.sum(axis=1), axis=0) * 100
    donor_subtype_props = donor_subtype_props.reset_index()

    # Merge with donor info
    props_with_info = donor_subtype_props.merge(
        donor_info[['sample', 'age_num', 'age_pop', 'Country']],
        on='sample', how='left'
    )
    props_with_info['is_young'] = props_with_info['age_pop'] == 'young_pop'

    # Print donor summary
    n_total = len(props_with_info)
    n_young = props_with_info['is_young'].sum()
    n_old = (~props_with_info['is_young']).sum()
    print(f"\nDonor breakdown: {n_total} donors")
    print(f"  Young: {n_young}")
    print(f"  Old: {n_old}")
    print(f"  China: {(props_with_info['Country'] == 'China').sum()}")
    print(f"  Spain: {(props_with_info['Country'] == 'Spain').sum()}")

    # Per-subtype analysis
    results = []
    for subtype in FAP_SUBTYPES:
        col = subtype
        if col not in props_with_info.columns:
            print(f"  WARNING: {subtype} not found in proportions, skipping")
            continue

        vals = props_with_info[col]
        young_mask = props_with_info['is_young']
        old_mask = ~props_with_info['is_young']

        young_mean = vals[young_mask].mean()
        old_mean = vals[old_mask].mean()
        delta = old_mean - young_mean

        # Spearman rho(subtype_proportion, age) -- all donors
        rho_all, p_all = spearmanr(vals, props_with_info['age_num'])

        # Country-adjusted partial Spearman via residualization
        country_dummies = pd.get_dummies(props_with_info['Country'], drop_first=True)
        if country_dummies.shape[1] > 0:
            from numpy.linalg import lstsq
            X_country = np.column_stack([np.ones(len(vals)), country_dummies.values])
            coef_prop, _, _, _ = lstsq(X_country, vals.values, rcond=None)
            resid_prop = vals.values - X_country @ coef_prop
            coef_age, _, _, _ = lstsq(X_country, props_with_info['age_num'].values, rcond=None)
            resid_age = props_with_info['age_num'].values - X_country @ coef_age
            rho_adj, p_adj = spearmanr(resid_prop, resid_age)
        else:
            rho_adj, p_adj = rho_all, p_all

        # Within-China analysis
        china_mask = props_with_info['Country'] == 'China'
        china_vals = vals[china_mask]
        china_ages = props_with_info.loc[china_mask, 'age_num']
        if len(china_vals) >= 4:
            rho_china, p_china = spearmanr(china_vals, china_ages)
        else:
            rho_china, p_china = np.nan, np.nan

        # Within-Spain analysis
        spain_mask = props_with_info['Country'] == 'Spain'
        spain_vals = vals[spain_mask]
        spain_ages = props_with_info.loc[spain_mask, 'age_num']
        if len(spain_vals) >= 4:
            rho_spain, p_spain = spearmanr(spain_vals, spain_ages)
        else:
            rho_spain, p_spain = np.nan, np.nan

        results.append({
            'subtype': subtype,
            'n_donors': len(vals),
            'young_mean_pct': round(young_mean, 2),
            'old_mean_pct': round(old_mean, 2),
            'delta_pct': round(delta, 2),
            'rho_age': round(rho_all, 4),
            'p_age': round(p_all, 6),
            'rho_age_adj_country': round(rho_adj, 4),
            'p_adj': round(p_adj, 6),
            'within_china_rho': round(rho_china, 4) if not np.isnan(rho_china) else np.nan,
            'within_china_p': round(p_china, 6) if not np.isnan(p_china) else np.nan,
            'within_spain_rho': round(rho_spain, 4) if not np.isnan(rho_spain) else np.nan,
            'within_spain_p': round(p_spain, 6) if not np.isnan(p_spain) else np.nan,
        })

        print(f"\n  {subtype}:")
        print(f"    Young mean: {young_mean:.1f}%, Old mean: {old_mean:.1f}%, Delta: {delta:+.1f}%")
        print(f"    rho(age)={rho_all:.3f}, p={p_all:.4f}")
        print(f"    rho(age|country)={rho_adj:.3f}, p={p_adj:.4f}")
        print(f"    Within-China: rho={rho_china:.3f}, p={p_china:.4f}")
        print(f"    Within-Spain: rho={rho_spain:.3f}, p={p_spain:.4f}")

    df_results = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, 'c1_fap_subtype_proportions.csv')
    df_results.to_csv(out_path, index=False)
    print(f"\nC1 results saved to {out_path}")
    return df_results


# ==================== C2: Subclustering pipeline ====================

def run_subclustering_pipeline(adata, label):
    """Shared pipeline: HVG -> PCA -> Neighbors -> Leiden -> UMAP -> summary."""
    print(f"\n  Running subclustering for {label}...")
    print(f"    Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Detect HVG flavor
    flavor = detect_hvg_flavor(adata)

    # Compute HVGs
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor=flavor, subset=False)
    except Exception as e:
        print(f"    HVG with {flavor} failed ({e}), falling back to 'seurat'")
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat', subset=False)

    n_hvg = adata.var['highly_variable'].sum()
    print(f"    HVGs: {n_hvg}")

    # PCA on HVGs
    adata_hvg = adata[:, adata.var['highly_variable']].copy()
    sc.pp.pca(adata_hvg, n_comps=30, use_highly_variable=False)

    # Transfer PCA to main adata
    adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
    adata.uns['pca'] = adata_hvg.uns.get('pca', {})
    del adata_hvg
    gc.collect()

    # Neighbors
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')

    # Pick best Leiden resolution
    best_res, best_n = pick_best_resolution(adata, LEIDEN_RESOLUTIONS)
    print(f"    Best resolution: {best_res} ({best_n} clusters)")

    # Final cluster label
    adata.obs['cluster'] = adata.obs[f'leiden_{best_res}'].values
    print(f"    Cluster sizes:\n{adata.obs['cluster'].value_counts().sort_index().to_string()}")

    # UMAP
    sc.tl.umap(adata)

    # Compute markers
    adata.obs['JUNB_expr'] = get_gene_expression(adata, 'JUNB')
    adata.obs['SASP12_composite'] = compute_sasp12_composite(adata)

    # Ensure age numeric and age_group
    adata.obs['age_num'] = pd.to_numeric(adata.obs['age'], errors='coerce')
    age_pop_str = adata.obs['age_pop'].astype(str)
    adata.obs['age_group'] = age_pop_str.map(
        {'young_pop': 'young', 'old_pop': 'old'}
    ).fillna('unknown')

    # Per-cluster summary
    cluster_summary = []
    for cl in sorted(adata.obs['cluster'].unique()):
        cl_mask = adata.obs['cluster'] == cl
        cl_data = adata.obs[cl_mask]
        n_cells = cl_mask.sum()
        mean_junb = cl_data['JUNB_expr'].mean()
        mean_sasp = cl_data['SASP12_composite'].mean()
        frac_old = (cl_data['age_group'] == 'old').mean()
        frac_young = (cl_data['age_group'] == 'young').mean()

        # Dominant annotation
        if 'Annotation' in cl_data.columns:
            dominant_ann = cl_data['Annotation'].astype(str).mode().iloc[0]
        else:
            dominant_ann = 'N/A'

        cluster_summary.append({
            'cluster': cl,
            'n_cells': n_cells,
            'dominant_annotation': dominant_ann,
            'mean_JUNB': round(mean_junb, 4),
            'mean_SASP12': round(mean_sasp, 4),
            'frac_old': round(frac_old, 3),
            'frac_young': round(frac_young, 3),
        })

    df_summary = pd.DataFrame(cluster_summary)
    print(f"\n  Per-cluster summary ({label}):")
    print(df_summary.to_string(index=False))

    return adata, df_summary


def run_c2_vascular():
    """C2 for Vascular compartment: snRNA-only, filter to EC subtypes."""
    print("\n" + "=" * 60)
    print("C2: Vascular -- snRNA-Only Subclustering")
    print("=" * 60)

    adata = sc.read_h5ad(os.path.join(DATA_DIR, 'Vascular_scsn_RNA.h5ad'))
    print(f"Loaded Vascular: {adata.n_obs} cells, {adata.n_vars} genes")

    # Filter to EC subtypes then snRNA
    ec_mask = adata.obs['Annotation'].isin(VASCULAR_SUBTYPES)
    snrna_mask = adata.obs['tech'] == 'snRNA'
    combined = ec_mask & snrna_mask
    print(f"  EC subtype cells: {ec_mask.sum()}")
    print(f"  snRNA cells: {snrna_mask.sum()}")
    print(f"  Combined (EC + snRNA): {combined.sum()} cells")

    adata = adata[combined].copy()

    adata_sub, df_summary = run_subclustering_pipeline(adata, 'Vascular_snRNA')

    # Save outputs
    df_summary.to_csv(os.path.join(OUT_DIR, 'c2_vascular_snrna_clusters.csv'), index=False)
    de_results = run_de_per_cluster(adata_sub)
    save_de_results(de_results, os.path.join(OUT_DIR, 'c2_vascular_snrna_de.csv'))
    plot_umap_4panel(adata_sub, 'Vascular (snRNA-only)',
                     os.path.join(OUT_DIR, 'c2_umap_vascular.png'))

    del adata, adata_sub
    gc.collect()
    return df_summary


def run_c2_fap(fap_adata):
    """C2 for FAP compartment: snRNA-only, filter to FAP subtypes."""
    print("\n" + "=" * 60)
    print("C2: FAP -- snRNA-Only Subclustering")
    print("=" * 60)

    print(f"Loaded FAP: {fap_adata.n_obs} cells, {fap_adata.n_vars} genes")

    # Filter to FAP subtypes then snRNA
    fap_mask = fap_adata.obs['Annotation'].isin(FAP_SUBTYPES)
    snrna_mask = fap_adata.obs['tech'] == 'snRNA'
    combined = fap_mask & snrna_mask
    print(f"  FAP subtype cells: {fap_mask.sum()}")
    print(f"  snRNA cells: {snrna_mask.sum()}")
    print(f"  Combined: {combined.sum()} cells")

    adata = fap_adata[combined].copy()

    adata_sub, df_summary = run_subclustering_pipeline(adata, 'FAP_snRNA')

    # Save outputs
    df_summary.to_csv(os.path.join(OUT_DIR, 'c2_fap_snrna_clusters.csv'), index=False)
    de_results = run_de_per_cluster(adata_sub)
    save_de_results(de_results, os.path.join(OUT_DIR, 'c2_fap_snrna_de.csv'))
    plot_umap_4panel(adata_sub, 'FAP (snRNA-only)',
                     os.path.join(OUT_DIR, 'c2_umap_fap.png'))

    del adata, adata_sub
    gc.collect()
    return df_summary


def run_c2_musc():
    """C2 for MuSC compartment: ALL cells (balanced tech)."""
    print("\n" + "=" * 60)
    print("C2: MuSC -- All Cells Subclustering")
    print("=" * 60)

    adata = sc.read_h5ad(os.path.join(DATA_DIR, 'MuSC_scsn_RNA.h5ad'))
    print(f"Loaded MuSC: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"  Tech distribution: {adata.obs['tech'].value_counts().to_dict()}")

    # Use all cells (balanced tech)
    adata_sub, df_summary = run_subclustering_pipeline(adata, 'MuSC_all')

    # Save outputs
    df_summary.to_csv(os.path.join(OUT_DIR, 'c2_musc_clusters.csv'), index=False)
    de_results = run_de_per_cluster(adata_sub)
    save_de_results(de_results, os.path.join(OUT_DIR, 'c2_musc_de.csv'))
    plot_umap_4panel(adata_sub, 'MuSC (all cells)',
                     os.path.join(OUT_DIR, 'c2_umap_musc.png'))

    del adata, adata_sub
    gc.collect()
    return df_summary


# ==================== Main ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Batch 053: C1 (FAP Subtype Proportions) + C2 (Subclustering)")
    print("=" * 60)

    # Load FAP data once -- shared between C1 and C2
    print("\nLoading FAP data (OMIX004308-02.h5ad)...")
    fap_adata = sc.read_h5ad(os.path.join(DATA_DIR, 'OMIX004308-02.h5ad'),
                              backed='r')
    # C1 only needs obs, so backed mode is fine
    c1_results = run_c1(fap_adata)

    # For C2 FAP, we need the expression data. Load fully.
    print("\nRe-loading FAP data in full mode for C2...")
    del fap_adata
    gc.collect()
    fap_adata = sc.read_h5ad(os.path.join(DATA_DIR, 'OMIX004308-02.h5ad'))

    # --- C2 ---
    c2_vascular = run_c2_vascular()
    c2_fap = run_c2_fap(fap_adata)

    # Free FAP memory before loading MuSC
    del fap_adata
    gc.collect()

    c2_musc = run_c2_musc()

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("BATCH 053 SUMMARY")
    print("=" * 60)
    print(f"\nC1: FAP Subtype Proportions -- {len(c1_results)} subtypes analyzed")
    print(f"    Output: {os.path.join(OUT_DIR, 'c1_fap_subtype_proportions.csv')}")
    print(f"\nC2: Subclustering")
    print(f"    Vascular snRNA: {len(c2_vascular)} clusters")
    print(f"    FAP snRNA: {len(c2_fap)} clusters")
    print(f"    MuSC all: {len(c2_musc)} clusters")
    print(f"    Outputs in: {OUT_DIR}")
    print("\nDone.")
