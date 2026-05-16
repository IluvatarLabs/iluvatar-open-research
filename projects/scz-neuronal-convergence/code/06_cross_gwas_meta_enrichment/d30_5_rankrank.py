#!/usr/bin/env python3
"""D30_5: Rank-Rank Overlap between PGC3 MAGMA p-values and neuronal marker affinity."""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
MAGMA_FILE = PROJECT_ROOT / "experiments/batch_026/gene_level_pgc3.tsv"
MARKERS_FILE = PROJECT_ROOT / "experiments/batch_009/data/markers.parquet"
OUTPUT_DIR = PROJECT_ROOT / "experiments/batch_033/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load and inspect data ----
magma = pd.read_csv(MAGMA_FILE, sep='\t')
print(f"[D30_5] MAGMA file: {len(magma)} genes, columns={magma.columns.tolist()}")

markers_df = pd.read_parquet(MARKERS_FILE)
print(f"[D30_5] Markers file: {len(markers_df)} entries, cell types={markers_df['cell_type'].unique().tolist()}")

# ---- Identify neuronal marker genes ----
neuronal_genes = set(markers_df[markers_df['cell_type'] == 'Neurons']['gene'].unique())
print(f"[D30_5] Neuronal marker genes: {len(neuronal_genes)}")

# ---- Merge MAGMA with cell-type counts ----
ct_counts = markers_df.groupby('gene').size().reset_index(name='n_celltypes')
magma_ct = magma.merge(ct_counts, on='gene', how='left')
magma_ct['n_celltypes'] = magma_ct['n_celltypes'].fillna(0)
magma_ct['is_neuronal'] = (magma_ct['n_celltypes'] > 0).astype(int)

print(f"[D30_5] MAGMA+CT merged: {len(magma_ct)} rows; neuronal overlap = {magma_ct['is_neuronal'].sum()}")

# ---- Rank genes by MAGMA p-value (ascending = most significant = rank 1) ----
# Using min_p as primary p-value (most conservative/standard MAGMA output)
p_col = 'min_p'
magma_ct = magma_ct.dropna(subset=[p_col])
magma_ct['magma_rank'] = magma_ct[p_col].rank(method='average')
print(f"[D30_5] Ranked by {p_col}, range [{magma_ct['magma_rank'].min()}, {magma_ct['magma_rank'].max()}]")

# ---- Statistical tests ----
# Mann-Whitney U: are neuronal genes ranked more significantly (lower rank)?
# H0: neuronal and non-neuronal genes have the same MAGMA rank distribution
# Lower median rank in neuronal => enriched at top of list
neuronal_ranks = magma_ct[magma_ct['is_neuronal'] == 1]['magma_rank']
non_neuronal_ranks = magma_ct[magma_ct['is_neuronal'] == 0]['magma_rank']

u_stat, p_mwu = stats.mannwhitneyu(neuronal_ranks, non_neuronal_ranks, alternative='two-sided')
print(f"\n[D30_5] Mann-Whitney U: U={u_stat:.0f}, p={p_mwu:.4e}")
print(f"[D30_5]   neuronal median rank = {neuronal_ranks.median():.1f} (p{magma_ct[p_col][magma_ct['is_neuronal']==1].median():.2e} in raw p)")
print(f"[D30_5]   non-neuronal median rank = {non_neuronal_ranks.median():.1f}")

n_total = len(magma_ct)
neuronal_median_rank_pct = 100 * neuronal_ranks.median() / n_total
print(f"[D30_5]   neuronal median at {neuronal_median_rank_pct:.1f}th percentile")

# Spearman: MAGMA rank vs neuronal cell-type count (continuous affinity)
# Lower rank = stronger SCZ association
rho, p_rho = stats.spearmanr(magma_ct['magma_rank'], magma_ct['n_celltypes'])
print(f"\n[D30_5] Spearman (MAGMA rank vs marker count): rho={rho:.4f}, p={p_rho:.4e}")

# Point-biserial: neuronal (binary) vs MAGMA rank
pb_r, p_pb = stats.pointbiserialr(magma_ct['is_neuronal'], magma_ct['magma_rank'])
print(f"[D30_5] Point-biserial (neuronal vs MAGMA rank): r={pb_r:.4f}, p={p_pb:.4e}")

# Kolmogorov-Smirnov test: two-sample KS for rank distribution difference
ks_stat, p_ks = stats.ks_2samp(neuronal_ranks, non_neuronal_ranks)
print(f"\n[D30_5] KS-test: stat={ks_stat:.4f}, p={p_ks:.4e}")

# Effect direction: lower rank in neuronal => negative r
significant = p_mwu < 0.05
enriched = neuronal_ranks.median() < non_neuronal_ranks.median()
interpretation = "Neuronal genes significantly enriched at top of MAGMA ranking" if (significant and enriched) else \
                "No significant rank enrichment for neuronal genes in MAGMA ranking"

# ---- Save results ----
results = {
    "status": "success",
    "n_total_genes": n_total,
    "n_neuronal_genes": int(magma_ct['is_neuronal'].sum()),
    "neuronal_median_rank_pct": round(float(neuronal_median_rank_pct), 2),
    "non_neuronal_median_rank_pct": round(float(100 * non_neuronal_ranks.median() / n_total), 2),
    "mannwhitney_u": float(u_stat),
    "mannwhitney_p": float(p_mwu),
    "mannwhitney_significant": bool(significant),
    "spearman_rho": float(rho),
    "spearman_p": float(p_rho),
    "pointbiserial_r": float(pb_r),
    "pointbiserial_p": float(p_pb),
    "ks_stat": float(ks_stat),
    "ks_p": float(p_ks),
    "interpretation": interpretation,
    "note": "Rank 1 = most significant MAGMA p-value; lower rank in neuronal genes indicates SCZ enrichment"
}

output_path = OUTPUT_DIR / "d30_5_rankrank_results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[D30_5] Results saved to {output_path}")
print(json.dumps(results, indent=2))