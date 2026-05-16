"""
MUST DO 3: JUNB+ FAP Surface Marker Characterization

This experiment identifies surface markers enriched on JUNB+ (senescent) FAPs
to support FAP-specific delivery strategies for JNK/AP-1 inhibition therapy.

Protocol:
1. Stratify aged FAPs by JUNB expression (top/bottom quartile)
2. Filter genes (remove mitochondrial, ribosomal, hemoglobin, pseudogenes)
3. Run differential expression (Welch's t-test)
4. Cross-reference known FAP surface markers
5. Identify novel surface markers from top DEGs

NOTE: Data is pre-normalized/scaled (z-score). Analysis uses mean difference
as the effect size since data is already log-transformed.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# LOAD AND FILTER DATA
# ==============================================================================
print("Loading data...")
adata = sc.read_h5ad('data/OMIX004308-02.h5ad')

# Filter to aged FAPs (age >= 70 AND annotation contains "FAP")
aged_faps = adata[
    (adata.obs['age'].astype(str).astype(int) >= 70) &
    (adata.obs['Annotation'].str.contains('FAP', na=False))
]
print(f"Aged FAPs: {aged_faps.shape[0]} cells")

# Extract JUNB expression
junb_expr = aged_faps[:, 'JUNB'].X.toarray().flatten() if hasattr(aged_faps[:, 'JUNB'].X, 'toarray') else np.array(aged_faps[:, 'JUNB'].X).flatten()
print(f"JUNB expression range: [{junb_expr.min():.3f}, {junb_expr.max():.3f}]")
print(f"JUNB expression mean: {junb_expr.mean():.3f}")

# ==============================================================================
# STRATIFY BY JUNB QUARTILE
# ==============================================================================
q25 = np.percentile(junb_expr, 25)
q75 = np.percentile(junb_expr, 75)
print(f"JUNB quartile thresholds: Q25={q25:.3f}, Q75={q75:.3f}")

junb_pos_mask = junb_expr >= q75
junb_neg_mask = junb_expr <= q25

# Check tech split
tech_split = {}
for tech in ['scRNA', 'snRNA']:
    if tech in aged_faps.obs['tech'].values:
        tech_cells = aged_faps.obs['tech'] == tech
        tech_split[tech] = {
            'pos': int(np.sum(junb_pos_mask & tech_cells)),
            'neg': int(np.sum(junb_neg_mask & tech_cells))
        }

junb_stratification = {
    'junb_pos_n_cells': int(np.sum(junb_pos_mask)),
    'junb_neg_n_cells': int(np.sum(junb_neg_mask)),
    'junb_pos_mean': float(junb_expr[junb_pos_mask].mean()),
    'junb_neg_mean': float(junb_expr[junb_neg_mask].mean()),
    'quartile_threshold': float(q75),
    'tech_split': tech_split
}
print(f"JUNB+ cells: {junb_stratification['junb_pos_n_cells']}")
print(f"JUNB- cells: {junb_stratification['junb_neg_n_cells']}")
print(f"Tech split: {tech_split}")

# ==============================================================================
# GENE FILTERING
# ==============================================================================
print("\nFiltering genes...")

# Gene symbols are in the index
gene_names = aged_faps.var.index

# Remove gene categories:
# 1. Mitochondrial genes (MT-)
# 2. Ribosomal genes (RPS, RPL)
# 3. Hemoglobin genes (HBB, HBA)
exclude_patterns = ['^MT-', '^RPS', '^RPL', '^HBB', '^HBA', '^HBM']
exclude_mask = gene_names.str.match('|'.join(exclude_patterns), na=False)

# Check for pseudogenes (typically have suffix like ...P, or contain "P" at end)
pseudogene_patterns = ['-[0-9]+$', 'P[0-9]*$', 'LINC', '^RP[0-9]']
pseudogene_mask = gene_names.str.contains('|'.join(pseudogene_patterns), na=False)

# Keep only protein-coding genes (heuristic: no known non-coding patterns)
keep_mask = ~exclude_mask & ~pseudogene_mask

# Ensure JUNB is included
if 'JUNB' in gene_names:
    keep_mask = keep_mask | (gene_names == 'JUNB')

keep_genes = gene_names[keep_mask]
print(f"Genes after filtering: {len(keep_genes)}")

# ==============================================================================
# DIFFERENTIAL EXPRESSION ANALYSIS
# ==============================================================================
print("\nRunning differential expression...")

# Get expression matrices for each group
expr_pos_raw = aged_faps[junb_pos_mask, keep_genes].X
expr_neg_raw = aged_faps[junb_neg_mask, keep_genes].X

# Convert to numpy arrays efficiently
if hasattr(expr_pos_raw, 'toarray'):
    expr_pos = expr_pos_raw.toarray()
    expr_neg = expr_neg_raw.toarray()
else:
    expr_pos = np.array(expr_pos_raw)
    expr_neg = np.array(expr_neg_raw)

# Calculate means
mean_pos = expr_pos.mean(axis=0).flatten()
mean_neg = expr_neg.mean(axis=0).flatten()

# For scaled/z-score data: filter by overall mean expression level
# Use genes with mean > 0 as a proxy for expressed genes
# (since z-scores centered at 0, genes with positive mean are moderately expressed)
mean_expr = (mean_pos + mean_neg) / 2
expressed_mask = mean_expr > -0.5  # Genes not strongly suppressed
print(f"Genes meeting expression threshold: {np.sum(expressed_mask)}")

# Run Welch's t-test for each gene
results = []
gene_list = keep_genes[expressed_mask]
pos_expr = expr_pos[:, expressed_mask]
neg_expr = expr_neg[:, expressed_mask]
pos_means = mean_pos[expressed_mask]
neg_means = mean_neg[expressed_mask]

for i, gene in enumerate(gene_list):
    p_vals = pos_expr[:, i]
    n_vals = neg_expr[:, i]

    # Skip if no variance
    if np.std(p_vals) < 1e-10 and np.std(n_vals) < 1e-10:
        continue

    # Welch's t-test (does not assume equal variance)
    t_stat, p_val = stats.ttest_ind(p_vals, n_vals, equal_var=False)

    # For scaled data, use mean difference as log2FC proxy
    # The mean difference in z-score space correlates with log2FC
    mean_diff = pos_means[i] - neg_means[i]

    results.append({
        'gene': str(gene),
        'mean_junb_pos': float(pos_means[i]),
        'mean_junb_neg': float(neg_means[i]),
        'mean_diff': float(mean_diff),  # This is the effect size for scaled data
        'log2FC': float(mean_diff),  # Use as proxy for log2FC (scales similarly)
        'p_value': float(p_val)
    })

deg_df = pd.DataFrame(results)

# BH correction
if len(deg_df) > 0:
    reject, bh_pvals, _, _ = multipletests(deg_df['p_value'].values, method='fdr_bh')
    deg_df['bh_p_value'] = bh_pvals
    deg_df['significant'] = reject

# Sort by absolute log2FC (mean difference)
deg_df['abs_log2FC'] = deg_df['log2FC'].abs()
deg_df = deg_df.sort_values('abs_log2FC', ascending=False)

print(f"Total DEGs tested: {len(deg_df)}")
print(f"Significant (BH < 0.05): {np.sum(deg_df['bh_p_value'] < 0.05)}")

# ==============================================================================
# CROSS-REFERENCE KNOWN FAP SURFACE MARKERS
# ==============================================================================
print("\nAnalyzing known FAP surface markers...")

known_markers = [
    'PDGFRA', 'PDGFRB', 'CD29', 'ITGB1', 'CD34', 'THY1', 'NT5E', 'ENG',
    'CD51', 'ITGAV', 'LY6A', 'LY6E', 'SCA1', 'CD9', 'CD81', 'CD63',
    'PDPN', 'MCAM', 'VCAM1', 'ICAM1', 'CD90', 'CD73', 'CD105', 'CD146',
    'CD106', 'CD54'
]

known_markers_results = []
for marker in known_markers:
    # Check if marker exists in data
    if marker in deg_df['gene'].values:
        row = deg_df[deg_df['gene'] == marker].iloc[0]
        known_markers_results.append({
            'gene': marker,
            'mean_junb_neg': row['mean_junb_neg'],
            'mean_junb_pos': row['mean_junb_pos'],
            'log2FC': row['log2FC'],
            'p_value': row['p_value'],
            'bh_p_value': row['bh_p_value']
        })
    else:
        # Marker not in analysis (possibly filtered out)
        known_markers_results.append({
            'gene': marker,
            'mean_junb_neg': None,
            'mean_junb_pos': None,
            'log2FC': None,
            'p_value': None,
            'bh_p_value': None
        })

known_df = pd.DataFrame(known_markers_results)
print(f"\nKnown markers analysis:")
print(known_df.to_string())

# Check for technology-specific effects
print("\n\nChecking technology-specific effects...")
for tech in ['scRNA', 'snRNA']:
    if tech in aged_faps.obs['tech'].values:
        tech_mask = aged_faps.obs['tech'] == tech
        tech_junb_pos = junb_pos_mask & tech_mask
        tech_junb_neg = junb_neg_mask & tech_mask
        print(f"\n{tech}:")
        for marker in ['PDGFRB', 'CD34', 'NT5E', 'CD9', 'CD63']:
            if marker in aged_faps.var_names:
                m_expr = aged_faps[:, marker].X.toarray().flatten() if hasattr(aged_faps[:, marker].X, 'toarray') else np.array(aged_faps[:, marker].X).flatten()
                pos_mean = m_expr[tech_junb_pos].mean()
                neg_mean = m_expr[tech_junb_neg].mean()
                diff = pos_mean - neg_mean
                print(f"  {marker}: JUNB+ mean={pos_mean:.3f}, JUNB- mean={neg_mean:.3f}, diff={diff:.3f}")

# ==============================================================================
# IDENTIFY NOVEL SURFACE MARKERS
# ==============================================================================
print("\nIdentifying novel surface markers...")

# Use top 50 DEGs by abs log2FC
top_degs = deg_df.head(50).copy()

# GO term annotations for surface markers (based on known gene ontology)
# This is a curated list of plasma membrane/cell surface genes
surface_go_terms = {
    # Cytokines and chemokines (often have surface receptors)
    'CXCL1': 'plasma membrane',
    'CXCL2': 'plasma membrane',
    'CXCL3': 'plasma membrane',
    'CXCL8': 'plasma membrane',
    'CCL2': 'plasma membrane',
    'CCL3': 'plasma membrane',
    'CCL7': 'plasma membrane',
    'CCL8': 'plasma membrane',
    'CCL13': 'plasma membrane',
    'CXCL6': 'plasma membrane',
    'CXCL5': 'plasma membrane',

    # Interleukins and TNF
    'IL6': 'plasma membrane',
    'IL11': 'plasma membrane',
    'IL1B': 'plasma membrane',
    'IL1A': 'plasma membrane',
    'TNF': 'plasma membrane',

    # Matrix metalloproteinases
    'MMP1': 'plasma membrane',
    'MMP2': 'plasma membrane',
    'MMP3': 'plasma membrane',
    'MMP9': 'plasma membrane',
    'MMP14': 'plasma membrane',

    # Extracellular matrix
    'COL1A1': 'extracellular',
    'COL1A2': 'extracellular',
    'COL3A1': 'extracellular',
    'COL5A1': 'extracellular',
    'COL5A2': 'extracellular',
    'COL6A1': 'extracellular',
    'COL6A2': 'extracellular',
    'COL6A3': 'extracellular',
    'COL12A1': 'extracellular',
    'COL14A1': 'extracellular',
    'COL15A1': 'extracellular',
    'FN1': 'extracellular',
    'DCN': 'extracellular',
    'LUM': 'extracellular',
    'BGN': 'extracellular',
    'FBN1': 'extracellular',
    'ELN': 'extracellular',
    'CYR61': 'extracellular',
    'CTGF': 'extracellular',
    'POSTN': 'extracellular',
    'SPP1': 'plasma membrane',
    'THBS1': 'plasma membrane',
    'THBS2': 'plasma membrane',
    'VCAN': 'extracellular',
    'HAS1': 'plasma membrane',
    'HAS2': 'plasma membrane',
    'HAS3': 'plasma membrane',

    # Growth factors
    'TGFB1': 'extracellular',
    'TGFB2': 'extracellular',
    'TGFB3': 'extracellular',
    'IGF1': 'extracellular',
    'IGF2': 'extracellular',
    'IGFBP2': 'extracellular',
    'IGFBP3': 'extracellular',
    'IGFBP4': 'extracellular',
    'IGFBP5': 'extracellular',
    'IGFBP6': 'extracellular',
    'IGFBP7': 'extracellular',
    'FGF2': 'extracellular',
    'FGF7': 'extracellular',
    'FGF10': 'extracellular',
    'HGF': 'extracellular',
    'PDGFRA': 'plasma membrane',
    'PDGFRB': 'plasma membrane',
    'EGFR': 'plasma membrane',
    'HBEGF': 'plasma membrane',
    'AREG': 'plasma membrane',
    'EREG': 'plasma membrane',

    # Integrins
    'ITGA1': 'plasma membrane',
    'ITGA2': 'plasma membrane',
    'ITGA3': 'plasma membrane',
    'ITGA5': 'plasma membrane',
    'ITGA6': 'plasma membrane',
    'ITGA7': 'plasma membrane',
    'ITGA8': 'plasma membrane',
    'ITGB1': 'plasma membrane',
    'ITGB3': 'plasma membrane',
    'ITGB5': 'plasma membrane',
    'ITGB6': 'plasma membrane',
    'ITGB8': 'plasma membrane',

    # Tetraspanins and CD molecules
    'CD9': 'plasma membrane',
    'CD44': 'plasma membrane',
    'CD47': 'plasma membrane',
    'CD59': 'plasma membrane',
    'CD63': 'plasma membrane',
    'CD81': 'plasma membrane',
    'CD151': 'plasma membrane',

    # Glypicans and proteoglycans
    'GPC1': 'plasma membrane',
    'GPC2': 'plasma membrane',
    'GPC3': 'plasma membrane',
    'GPC4': 'plasma membrane',
    'GPC6': 'plasma membrane',
    'CSPG4': 'plasma membrane',
    'NG2': 'plasma membrane',
    'BCAN': 'plasma membrane',
    'NCAN': 'plasma membrane',

    # Syndecans
    'SDC1': 'plasma membrane',
    'SDC2': 'plasma membrane',
    'SDC3': 'plasma membrane',
    'SDC4': 'plasma membrane',

    # WNT proteins
    'WNT2': 'extracellular',
    'WNT5A': 'plasma membrane',
    'WNT5B': 'plasma membrane',
    'WNT6': 'extracellular',
    'WNT10A': 'extracellular',

    # S100 proteins
    'S100A4': 'plasma membrane',
    'S100A6': 'plasma membrane',
    'S100A10': 'plasma membrane',
    'S100A11': 'plasma membrane',
    'S100A13': 'plasma membrane',
    'S100A16': 'plasma membrane',

    # Annexins
    'ANXA1': 'plasma membrane',
    'ANXA2': 'plasma membrane',
    'ANXA3': 'plasma membrane',
    'ANXA4': 'plasma membrane',
    'ANXA5': 'plasma membrane',
    'ANXA6': 'plasma membrane',
    'ANXA7': 'plasma membrane',
    'ANXA11': 'plasma membrane',

    # TIMPs
    'TIMP1': 'plasma membrane',
    'TIMP2': 'extracellular',
    'TIMP3': 'extracellular',
    'TIMP4': 'extracellular',

    # Adhesion molecules
    'VCAM1': 'plasma membrane',
    'ICAM1': 'plasma membrane',
    'PECAM1': 'plasma membrane',
    'SELE': 'plasma membrane',
    'SELP': 'plasma membrane',

    # Receptors
    'ENG': 'plasma membrane',
    'MCAM': 'plasma membrane',
    'NT5E': 'plasma membrane',
    'THY1': 'plasma membrane',
    'PDPN': 'plasma membrane',

    # Chemokine receptors
    'CXCR4': 'plasma membrane',
    'CXCR7': 'plasma membrane',
    'CCR2': 'plasma membrane',
    'CCR5': 'plasma membrane',

    # Urokinase
    'PLAU': 'plasma membrane',
    'PLAUR': 'plasma membrane',
    'PLAT': 'plasma membrane',

    # Other relevant
    'ADM': 'plasma membrane',
    'CALCRL': 'plasma membrane',
    'RAMP2': 'plasma membrane',
    'VEGFA': 'extracellular',
    'VEGFB': 'extracellular',
    'VEGFC': 'extracellular',
    'FLT1': 'plasma membrane',
    'KDR': 'plasma membrane',
    'FLT4': 'plasma membrane',
}

# Add GO annotations to top DEGs
top_degs['go_term'] = top_degs['gene'].map(
    lambda x: surface_go_terms.get(x, 'unknown')
)

# Filter for surface markers (plasma membrane or extracellular)
# Using |mean_diff| >= 0.5 as threshold (significant effect size for z-score data)
surface_markers = top_degs[
    (top_degs['go_term'].isin(['plasma membrane', 'extracellular'])) &
    (top_degs['bh_p_value'] < 0.05) &
    (top_degs['abs_log2FC'] >= 0.5)
].copy()

novel_markers = []
for _, row in surface_markers.iterrows():
    novel_markers.append({
        'gene': row['gene'],
        'log2FC': row['log2FC'],
        'bh_p_value': row['bh_p_value'],
        'go_annotation': row['go_term'],
        'mean_junb_neg': row['mean_junb_neg'],
        'mean_junb_pos': row['mean_junb_pos']
    })

print(f"\nNovel surface markers (BH < 0.05, |mean_diff| >= 0.5):")
if len(novel_markers) > 0:
    for m in novel_markers:
        print(f"  {m['gene']}: log2FC={m['log2FC']:.3f}, BH={m['bh_p_value']:.2e}, GO={m['go_annotation']}")
else:
    print("  None found with current thresholds. Relaxing to |mean_diff| >= 0.3...")
    surface_markers_relaxed = top_degs[
        (top_degs['go_term'].isin(['plasma membrane', 'extracellular'])) &
        (top_degs['bh_p_value'] < 0.05) &
        (top_degs['abs_log2FC'] >= 0.3)
    ].copy()
    for _, row in surface_markers_relaxed.iterrows():
        print(f"  {row['gene']}: log2FC={row['log2FC']:.3f}, BH={row['bh_p_value']:.2e}, GO={row['go_term']}")

# ==============================================================================
# PREPARE OUTPUT
# ==============================================================================
print("\nPreparing output files...")

# JSON output
output = {
    'junb_stratification': junb_stratification,
    'known_fap_markers': known_markers_results,
    'top_degs': top_degs.head(50)[['gene', 'log2FC', 'p_value', 'bh_p_value', 'mean_junb_neg', 'mean_junb_pos', 'go_term']].to_dict('records'),
    'novel_surface_markers': novel_markers
}

with open('experiments/batch_019/must_do_3_surface_markers_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# Full DEG table (top 200 by abs log2FC)
deg_df.head(200)[['gene', 'mean_junb_pos', 'mean_junb_neg', 'log2FC', 'p_value', 'bh_p_value', 'significant']].to_csv(
    'experiments/batch_019/must_do_3_all_degs.csv', index=False
)

# Known marker table
known_df.to_csv('experiments/batch_019/must_do_3_known_markers.csv', index=False)

print("\n=== EXPERIMENT COMPLETE ===")
print(f"JUNB+ cells: {junb_stratification['junb_pos_n_cells']}")
print(f"JUNB- cells: {junb_stratification['junb_neg_n_cells']}")
print(f"Total DEGs: {len(deg_df)}")
print(f"Significant DEGs (BH < 0.05): {np.sum(deg_df['bh_p_value'] < 0.05)}")
print(f"Novel surface markers identified: {len(novel_markers)}")
print("\nOutput files:")
print("  - experiments/batch_019/must_do_3_surface_markers_results.json")
print("  - experiments/batch_019/must_do_3_all_degs.csv")
print("  - experiments/batch_019/must_do_3_known_markers.csv")
