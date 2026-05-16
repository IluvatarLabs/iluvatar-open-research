#!/usr/bin/env python3
"""
JUNB+ FAP Minority Subset Characterization
============================================
Hypothesis: JUNB+ FAPs are a minority inflammatory subset while JUNB- FAPs are
the majority regenerative population. This reconciles:
- F084: weak JUNB-SASP coupling at donor level (rho=0.023)
- F061: JUNB+ aged FAPs have elevated TNF
- F085: overall FAP population shows regenerative compensation

Analysis Plan:
1. Define JUNB+ threshold (mean + 1SD, top 20th percentile)
2. Quantify JUNB+ fraction by donor and age
3. Characterize JUNB+ cell identity via clustering
4. Compare SASP12 scores in JUNB+ vs JUNB- cells
5. Map FAP subtypes onto JUNB+ cells
6. Analyze secretome (TNF, IL6, FGF7, HGF, MMP1)
"""

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = 'data/OMIX004308-02.h5ad'
OUTPUT_PATH = 'experiments/batch_029/junb_fap_results.json'

# JUNB threshold methods
JUNB_THRESHOLD_METHOD = 'mean_plus_1sd'  # 'mean_plus_1sd' or 'top20percentile'

# SASP genes (from literature)
SASP_GENES = {
    'IL6': 'IL6', 'IL8': 'CXCL8', 'IL1B': 'IL1B', 'TNF': 'TNF', 'MMP1': 'MMP1',
    'MMP3': 'MMP3', 'CCL2': 'CCL2', 'CXCL1': 'CXCL1', 'CXCL2': 'CXCL2',
    'GMCSF': 'CSF2', 'GCSF': 'CSF3', 'VEGFA': 'VEGFA', 'IGFBP7': 'IGFBP7'
}

# FAP subtype markers (from F003/F008)
FAP_SUBTYPE_MARKERS = {
    'MME+ FAP': ['MME', 'CXCL14', 'PLAC8'],      # Classical regenerative FAPs
    'CD55+ FAP': ['CD55', 'FBN1', 'DCN'],         # Fibrogenic/tenogenic
    'GPC3+ FAP': ['GPC3', 'LPL', 'APOD'],         # Adipogenic/lipid processing
    'CD99+ FAP': ['CD99', 'TMSB4X', 'TAGLN2'],    # Proliferative
    'RUNX2+ FAP': ['RUNX2', 'ALPL', 'SPP1'],      # Osteogenic commitment
    'Tenocyte': ['TNMD', 'COL1A1', 'SCX', 'MKX']  # Tenocyte-like
}

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)

print("=" * 60)
print("JUNB+ FAP Minority Subset Characterization")
print("=" * 60)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1] Loading FAP atlas...")
adata = sc.read_h5ad(DATA_PATH)
print(f"  Cells: {adata.shape[0]:,} | Genes: {adata.shape[1]:,}")

# Extract JUNB expression
junb_vec = adata[:, 'JUNB'].X.flatten() if hasattr(adata[:, 'JUNB'].X, 'flatten') else np.array(adata[:, 'JUNB'].X).flatten()
print(f"  JUNB expression: mean={junb_vec.mean():.3f}, std={junb_vec.std():.3f}")

# ============================================================
# STEP 1: JUNB+ THRESHOLD
# ============================================================
print("\n[2] Defining JUNB+ threshold...")

# Method: mean + 1 SD
junb_threshold_mean_sd = junb_vec.mean() + junb_vec.std()
print(f"  Method 1 (mean + 1SD): {junb_threshold_mean_sd:.3f}")

# Method: top 20th percentile
junb_threshold_pct = np.percentile(junb_vec, 80)
print(f"  Method 2 (top 20th percentile): {junb_threshold_pct:.3f}")

# Use mean + 1SD as primary threshold
junb_threshold = junb_threshold_mean_sd
print(f"  PRIMARY THRESHOLD (mean + 1SD): {junb_threshold:.3f}")

# Classify cells
adata.obs['JUNB_positive'] = junb_vec > junb_threshold
junb_pos_count = adata.obs['JUNB_positive'].sum()
junb_neg_count = (~adata.obs['JUNB_positive']).sum()
print(f"  JUNB+ cells: {junb_pos_count:,} ({100*junb_pos_count/adata.shape[0]:.1f}%)")
print(f"  JUNB- cells: {junb_neg_count:,} ({100*junb_neg_count/adata.shape[0]:.1f}%)")

junb_results = {
    'threshold_method': JUNB_THRESHOLD_METHOD,
    'threshold_value': float(junb_threshold),
    'junb_pos_cells': int(junb_pos_count),
    'junb_neg_cells': int(junb_neg_count),
    'junb_pos_fraction': float(junb_pos_count / adata.shape[0])
}

# ============================================================
# STEP 2: JUNB+ FRACTION BY DONOR AND AGE
# ============================================================
print("\n[3] JUNB+ fraction by donor and age...")

# Compute per-donor stats
donor_stats = []
for sample in adata.obs['sample'].unique():
    sample_mask = adata.obs['sample'] == sample
    n_cells = sample_mask.sum()
    n_junb_pos = adata.obs.loc[sample_mask, 'JUNB_positive'].sum()
    frac_junb_pos = n_junb_pos / n_cells if n_cells > 0 else 0

    age_val = adata.obs.loc[sample_mask, 'age'].iloc[0]
    age_pop = adata.obs.loc[sample_mask, 'age_pop'].iloc[0]

    donor_stats.append({
        'sample': sample,
        'age': age_val,
        'age_pop': age_pop,
        'n_cells': int(n_cells),
        'n_junb_pos': int(n_junb_pos),
        'frac_junb_pos': float(frac_junb_pos)
    })

donor_df = pd.DataFrame(donor_stats).sort_values(['age_pop', 'age'])
print("\n  Per-donor JUNB+ fraction:")
print(donor_df.to_string(index=False))

# Group by age
young_df = donor_df[donor_df['age_pop'] == 'young_pop']
old_df = donor_df[donor_df['age_pop'] == 'old_pop']

young_junb_frac = young_df['frac_junb_pos'].values
old_junb_frac = old_df['frac_junb_pos'].values

young_mean = young_junb_frac.mean()
young_std = young_junb_frac.std()
old_mean = old_junb_frac.mean()
old_std = old_junb_frac.std()

# Cohen's d
pooled_std = np.sqrt(((len(young_junb_frac)-1)*young_std**2 + (len(old_junb_frac)-1)*old_std**2) / (len(young_junb_frac)+len(old_junb_frac)-2))
cohens_d = (old_mean - young_mean) / pooled_std if pooled_std > 0 else 0

# t-test
t_stat, p_value = stats.ttest_ind(old_junb_frac, young_junb_frac)

print(f"\n  Age comparison:")
print(f"  Young (N={len(young_junb_frac)} donors): {young_mean:.3f} +/- {young_std:.3f}")
print(f"  Old   (N={len(old_junb_frac)} donors): {old_mean:.3f} +/- {old_std:.3f}")
print(f"  Cohen's d: {cohens_d:.3f} (effect size)")
print(f"  t-test: t={t_stat:.3f}, p={p_value:.4f}")

junb_results['age_comparison'] = {
    'young_n_donors': len(young_junb_frac),
    'young_mean_frac': float(young_mean),
    'young_std': float(young_std),
    'old_n_donors': len(old_junb_frac),
    'old_mean_frac': float(old_mean),
    'old_std': float(old_std),
    'Cohens_d': float(cohens_d),
    't_statistic': float(t_stat),
    'p_value': float(p_value)
}

# ============================================================
# STEP 3: JUNB+ CELL IDENTITY (CLUSTERING)
# ============================================================
print("\n[4] JUNB+ cell identity via clustering...")

# Subset to JUNB+ cells
adata_junb_pos = adata[adata.obs['JUNB_positive']].copy()
print(f"  JUNB+ cells: {adata_junb_pos.shape[0]:,}")

# Normalize and compute PCA/UMAP
print("  Computing PCA and UMAP...")
sc.pp.highly_variable_genes(adata_junb_pos, flavor='seurat', n_top_genes=2000)
sc.pp.scale(adata_junb_pos, max_value=10)
sc.tl.pca(adata_junb_pos, svd_solver='arpack', random_state=SEED)
sc.pp.neighbors(adata_junb_pos, n_neighbors=15, n_pcs=30, random_state=SEED)
sc.tl.umap(adata_junb_pos, random_state=SEED)

# Leiden clustering with multiple resolutions
print("  Leiden clustering...")
for res in [0.2, 0.5, 1.0]:
    sc.tl.leiden(adata_junb_pos, resolution=res, random_state=SEED, key_added=f'leiden_{res}')

n_clusters = adata_junb_pos.obs['leiden_0.5'].nunique()
print(f"  Clusters at resolution 0.5: {n_clusters}")

# Cluster size distribution
cluster_counts = adata_junb_pos.obs['leiden_0.5'].value_counts().sort_index()
print("\n  Cluster sizes:")
for cl, cnt in cluster_counts.items():
    print(f"    Cluster {cl}: {cnt:,} cells ({100*cnt/adata_junb_pos.shape[0]:.1f}%)")

# Find markers for each cluster
print("\n  Finding cluster markers...")
sc.tl.rank_genes_groups(adata_junb_pos, groupby='leiden_0.5', method='wilcoxon', use_raw=False)
rank_genes = pd.DataFrame(adata_junb_pos.uns['rank_genes_groups']['names'])

# Get top 5 markers per cluster
cluster_markers = {}
for cluster in sorted(adata_junb_pos.obs['leiden_0.5'].unique()):
    cluster_str = str(cluster)
    if cluster_str in rank_genes.columns:
        genes = rank_genes[cluster_str].head(10).tolist()
        cluster_markers[f'cluster_{cluster}'] = genes

print("\n  Top markers per cluster:")
for cl, markers in cluster_markers.items():
    print(f"    {cl}: {', '.join(markers[:5])}")

junb_results['clustering'] = {
    'n_clusters': int(n_clusters),
    'cluster_sizes': {str(k): int(v) for k, v in cluster_counts.items()},
    'top_markers': {str(k): v for k, v in cluster_markers.items()}
}

# ============================================================
# STEP 4: SASP12 SCORES IN JUNB+ VS JUNB- CELLS
# ============================================================
print("\n[5] SASP12 scores: JUNB+ vs JUNB-...")

# Check which SASP genes are present
available_sasp = []
for gene_name, gene_id in SASP_GENES.items():
    if gene_id in adata.var_names:
        available_sasp.append(gene_id)

print(f"  Available SASP genes: {available_sasp}")

# Compute SASP12 score using available genes
# For each cell, compute mean expression of SASP genes
sasp_expr = adata[:, available_sasp].X
if hasattr(sasp_expr, 'toarray'):
    sasp_expr = sasp_expr.toarray()
sasp_expr = np.array(sasp_expr)

# Mean per cell
sasp_score_per_cell = sasp_expr.mean(axis=1)

# Split by JUNB status
sasp_junb_pos = sasp_score_per_cell[adata.obs['JUNB_positive'].values]
sasp_junb_neg = sasp_score_per_cell[~adata.obs['JUNB_positive'].values]

sasp_junb_pos_mean = sasp_junb_pos.mean()
sasp_junb_pos_std = sasp_junb_pos.std()
sasp_junb_neg_mean = sasp_junb_neg.mean()
sasp_junb_neg_std = sasp_junb_neg.std()

# t-test at cell level
t_stat_sasp, p_val_sasp = stats.ttest_ind(sasp_junb_pos, sasp_junb_neg)

# Cohen's d
pooled_sasp = np.sqrt(((len(sasp_junb_pos)-1)*sasp_junb_pos_std**2 + (len(sasp_junb_neg)-1)*sasp_junb_neg_std**2) / (len(sasp_junb_pos)+len(sasp_junb_neg)-2))
cohens_d_sasp = (sasp_junb_pos_mean - sasp_junb_neg_mean) / pooled_sasp if pooled_sasp > 0 else 0

print(f"\n  Cell-level comparison (N cells):")
print(f"    JUNB+ cells: {len(sasp_junb_pos):,} | SASP12: {sasp_junb_pos_mean:.4f} +/- {sasp_junb_pos_std:.4f}")
print(f"    JUNB- cells: {len(sasp_junb_neg):,} | SASP12: {sasp_junb_neg_mean:.4f} +/- {sasp_junb_neg_std:.4f}")
print(f"    Cohen's d: {cohens_d_sasp:.3f}")
print(f"    t-test: t={t_stat_sasp:.3f}, p={p_val_sasp:.2e}")

# Effect direction
if sasp_junb_pos_mean > sasp_junb_neg_mean:
    direction = "JUNB+ cells have HIGHER SASP score (inflammatory)"
else:
    direction = "JUNB+ cells have LOWER SASP score (anti-inflammatory/rejuvenative)"

print(f"  Direction: {direction}")

junb_results['sasp_comparison'] = {
    'genes_used': available_sasp,
    'junb_pos_n_cells': int(len(sasp_junb_pos)),
    'junb_pos_sasp_mean': float(sasp_junb_pos_mean),
    'junb_pos_sasp_std': float(sasp_junb_pos_std),
    'junb_neg_n_cells': int(len(sasp_junb_neg)),
    'junb_neg_sasp_mean': float(sasp_junb_neg_mean),
    'junb_neg_sasp_std': float(sasp_junb_neg_std),
    'Cohens_d': float(cohens_d_sasp),
    't_statistic': float(t_stat_sasp),
    'p_value': float(p_val_sasp),
    'interpretation': direction
}

# ============================================================
# STEP 5: FAP SUBTYPE COMPOSITION OF JUNB+ CELLS
# ============================================================
print("\n[6] FAP subtype composition of JUNB+ cells...")

# Check annotation in JUNB+ vs JUNB- cells
annotation_junb_pos = adata_junb_pos.obs['Annotation'].value_counts()
annotation_junb_neg = adata[~adata.obs['JUNB_positive']].obs['Annotation'].value_counts()

print("\n  JUNB+ cell annotations:")
for annot, cnt in annotation_junb_pos.items():
    frac = 100 * cnt / len(adata_junb_pos)
    print(f"    {annot}: {cnt:,} ({frac:.1f}%)")

print("\n  JUNB- cell annotations:")
for annot, cnt in annotation_junb_neg.items():
    frac = 100 * cnt / len(adata[~adata.obs['JUNB_positive']])
    print(f"    {annot}: {cnt:,} ({frac:.1f}%)")

# Enrichment: is a subtype over-represented in JUNB+ cells?
total_by_annotation = adata.obs['Annotation'].value_counts()
enrichment = {}

for annot in total_by_annotation.index:
    junb_pos_annot_count = annotation_junb_pos.get(annot, 0)
    junb_neg_annot_count = annotation_junb_neg.get(annot, 0)

    if junb_pos_annot_count + junb_neg_annot_count > 0:
        frac_junb_pos = junb_pos_annot_count / (junb_pos_annot_count + junb_neg_annot_count)
        frac_overall = total_by_annotation[annot] / adata.shape[0]
        enrichment_ratio = frac_junb_pos / frac_overall if frac_overall > 0 else 0

        enrichment[annot] = {
            'n_junb_pos': int(junb_pos_annot_count),
            'n_junb_neg': int(junb_neg_annot_count),
            'frac_in_junb_pos': float(frac_junb_pos),
            'frac_overall': float(frac_overall),
            'enrichment_ratio': float(enrichment_ratio)
        }

print("\n  FAP subtype enrichment in JUNB+ cells:")
for annot, vals in enrichment.items():
    er = vals['enrichment_ratio']
    direction = "ENRICHED" if er > 1.2 else ("DEPLETED" if er < 0.8 else "similar")
    print(f"    {annot}: ER={er:.2f} ({direction})")

junb_results['subtype_composition'] = {
    'junb_pos_annotations': {str(k): int(v) for k, v in annotation_junb_pos.items()},
    'junb_neg_annotations': {str(k): int(v) for k, v in annotation_junb_neg.items()},
    'enrichment_ratios': {str(k): v for k, v in enrichment.items()}
}

# ============================================================
# STEP 6: SECRETOME IN JUNB+ VS JUNB- CELLS
# ============================================================
print("\n[7] Secretome analysis: JUNB+ vs JUNB-...")

# Key ligands to analyze
ligands = ['TNF', 'IL6', 'FGF7', 'HGF', 'MMP1', 'IL1B', 'CXCL8', 'CCL2']
available_ligands = [g for g in ligands if g in adata.var_names]

print(f"  Available ligand genes: {available_ligands}")

ligand_results = {}

for ligand in available_ligands:
    ligand_expr_junb_pos = adata[adata.obs['JUNB_positive'], ligand].X.flatten() if hasattr(adata[adata.obs['JUNB_positive'], ligand].X, 'flatten') else np.array(adata[adata.obs['JUNB_positive'], ligand].X).flatten()
    ligand_expr_junb_neg = adata[~adata.obs['JUNB_positive'], ligand].X.flatten() if hasattr(adata[~adata.obs['JUNB_positive'], ligand].X, 'flatten') else np.array(adata[~adata.obs['JUNB_positive'], ligand].X).flatten()

    mean_pos = ligand_expr_junb_pos.mean()
    mean_neg = ligand_expr_junb_neg.mean()

    t_stat_lig, p_val_lig = stats.ttest_ind(ligand_expr_junb_pos, ligand_expr_junb_neg)

    pooled_lig = np.sqrt(((len(ligand_expr_junb_pos)-1)*ligand_expr_junb_pos.std()**2 + (len(ligand_expr_junb_neg)-1)*ligand_expr_junb_neg.std()**2) / (len(ligand_expr_junb_pos)+len(ligand_expr_junb_neg)-2))
    cd_lig = (mean_pos - mean_neg) / pooled_lig if pooled_lig > 0 else 0

    direction = "HIGHER in JUNB+" if mean_pos > mean_neg else "LOWER in JUNB+"

    ligand_results[ligand] = {
        'junb_pos_mean': float(mean_pos),
        'junb_neg_mean': float(mean_neg),
        'Cohens_d': float(cd_lig),
        'p_value': float(p_val_lig),
        'direction': direction
    }

    print(f"\n  {ligand}:")
    print(f"    JUNB+ mean: {mean_pos:.4f} | JUNB- mean: {mean_neg:.4f}")
    print(f"    Cohen's d: {cd_lig:.3f} | p={p_val_lig:.2e} | {direction}")

junb_results['secretome'] = ligand_results

# ============================================================
# SUMMARY AND INTERPRETATION
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\n  JUNB+ fraction: {100*junb_pos_count/adata.shape[0]:.1f}% of FAPs")
print(f"  JUNB+ vs JUNB- age effect:")
print(f"    Young: {100*young_mean:.1f}% +/- {100*young_std:.1f}%")
print(f"    Old: {100*old_mean:.1f}% +/- {100*old_std:.1f}%")
print(f"    Cohen's d: {cohens_d:.3f}")

print(f"\n  SASP12 scores:")
print(f"    JUNB+ cells: {sasp_junb_pos_mean:.4f}")
print(f"    JUNB- cells: {sasp_junb_neg_mean:.4f}")
print(f"    Cohen's d: {cohens_d_sasp:.3f}")

print(f"\n  Subtype enrichment:")
enriched = [k for k, v in enrichment.items() if v['enrichment_ratio'] > 1.2]
depleted = [k for k, v in enrichment.items() if v['enrichment_ratio'] < 0.8]
print(f"    Enriched in JUNB+: {enriched}")
print(f"    Depleted in JUNB+: {depleted}")

# Classify JUNB+ cells
inflammation_markers = ['TNF', 'IL1B', 'IL6', 'MMP1']
inflammatory_in_junb_pos = [m for m in inflammation_markers if m in ligand_results and ligand_results[m]['direction'].startswith('HIGHER')]
regenerative_markers = ['FGF7', 'HGF']
regenerative_in_junb_pos = [m for m in regenerative_markers if m in ligand_results and ligand_results[m]['direction'].startswith('HIGHER')]

print(f"\n  Interpretation:")
if sasp_junb_pos_mean > sasp_junb_neg_mean and len(inflammatory_in_junb_pos) > 0:
    print(f"    JUNB+ FAPs are an INFLAMMATORY MINORITY subset")
    print(f"    - Higher SASP score, TNF/IL1B elevated")
    print(f"    - JUNB- FAPs are the majority regenerative population")
    classification = "inflammatory_minority"
elif sasp_junb_pos_mean < sasp_junb_neg_mean and len(regenerative_in_junb_pos) > 0:
    print(f"    JUNB+ FAPs are a REGENERATIVE subset")
    print(f"    - Lower SASP score, FGF7/HGF elevated")
    classification = "regenerative_subset"
else:
    print(f"    Pattern is mixed; need further analysis")
    classification = "mixed_or_unclear"

junb_results['summary'] = {
    'junb_pos_fraction': float(junb_pos_count / adata.shape[0]),
    'classification': classification,
    'inflammatory_markers_higher_in_junb_pos': inflammatory_in_junb_pos,
    'regenerative_markers_higher_in_junb_pos': regenerative_in_junb_pos
}

# ============================================================
# SAVE RESULTS
# ============================================================
print(f"\n[8] Saving results to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, 'w') as f:
    json.dump(junb_results, f, indent=2)
print("  Done.")

print("\n" + "=" * 60)
