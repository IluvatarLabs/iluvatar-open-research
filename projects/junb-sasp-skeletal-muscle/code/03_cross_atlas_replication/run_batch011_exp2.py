#!/usr/bin/env python3
"""
batch_011 Experiment 2: TGF-beta/SMAD Mediation Analysis in HLMA FAPs
Test whether FOS-collagen correlation is explained by SMAD3 activity.
Uses NON-COLLAGEN SMAD targets only (to avoid circularity).
"""

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
from scipy.sparse import issparse

# Configuration
DATA_PATH = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad"
OUTPUT_PATH = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_011/exp2_results.json"
SEED = 42
np.random.seed(SEED)

# Gene sets - NON-COLLAGEN SMAD targets (avoid circularity)
SMAD_TARGETS_NO_COLLAGEN = ['ID1', 'ID2', 'ID3', 'ID4', 'SERPINE1', 'MMP2', 'MMP9', 'THBS1']
COLLAGEN_GENES = ['COL1A1', 'COL3A1', 'COL6A1', 'COL6A3', 'FN1', 'LOX', 'LOXL1']

# Load data
print("Loading HLMA FAP atlas...")
adata = sc.read_h5ad(DATA_PATH)
print(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

# Get gene names
gene_names = adata.var_names.tolist()

# Check genes
smad_present = [g for g in SMAD_TARGETS_NO_COLLAGEN if g in gene_names]
collagen_present = [g for g in COLLAGEN_GENES if g in gene_names]
print(f"SMAD targets (non-collagen) available: {len(smad_present)}/{len(SMAD_TARGETS_NO_COLLAGEN)}")
print(f"Collagen genes available: {len(collagen_present)}/{len(COLLAGEN_GENES)}")

# Subset to scRNA, old, MME+ FAPs
# Identify technology
if 'technology' in adata.obs.columns:
    is_scrna = adata.obs['technology'].str.contains('scRNA', na=False)
elif 'Tech' in adata.obs.columns:
    is_scrna = adata.obs['Tech'].str.contains('scRNA', na=False)
else:
    # Check for batch labels
    is_scrna = adata.obs.index.str.contains('scRNA', na=False)
    print("WARNING: Using index-based technology detection")

# Age grouping - age column may be categorical
age_numeric = pd.to_numeric(adata.obs['age'], errors='coerce')
is_old = age_numeric >= 60
is_young = age_numeric <= 40

# Technology
is_scrna = adata.obs['tech'] == 'scRNA'

# FAP subtype (Annotation column)
is_mme = adata.obs['Annotation'] == 'MME+ FAP'

# Subset
mask = is_scrna & is_old & is_mme
adata_old_mme = adata[mask].copy()
print(f"Old MME+ scRNA FAPs: {adata_old_mme.n_obs} cells")

mask_young = is_scrna & is_young & is_mme
adata_young_mme = adata[mask_young].copy()
print(f"Young MME+ scRNA FAPs: {adata_young_mme.n_obs} cells")

# Get expression (handle sparse)
X = adata_old_mme.X.toarray() if issparse(adata_old_mme.X) else np.asarray(adata_old_mme.X)
gnames = adata_old_mme.var_names.tolist()

# Compute SMAD score (z-score first, then mean)
smad_indices = [gnames.index(g) for g in smad_present]
smad_expr = X[:, smad_indices].astype(float)
smad_mean = np.nanmean(smad_expr, axis=1)
smad_mean = (smad_mean - np.nanmean(smad_mean)) / np.nanstd(smad_mean)

# Compute collagen composite
coll_indices = [gnames.index(g) for g in collagen_present]
coll_expr = X[:, coll_indices].astype(float)
coll_composite = np.nanmean(coll_expr, axis=1)
coll_composite_z = (coll_composite - np.nanmean(coll_composite)) / np.nanstd(coll_composite)

# Individual gene expression
fos_idx = gnames.index('FOS') if 'FOS' in gnames else None
junb_idx = gnames.index('JUNB') if 'JUNB' in gnames else None

if fos_idx is not None:
    fos_expr = np.asarray(X[:, fos_idx]).flatten()
    fos_expr_z = (fos_expr - np.nanmean(fos_expr)) / np.nanstd(fos_expr)
if junb_idx is not None:
    junb_expr = np.asarray(X[:, junb_idx]).flatten()
    junb_expr_z = (junb_expr - np.nanmean(junb_expr)) / np.nanstd(junb_expr)

results = {}

# === 1. SMAD-collagen unconditional correlation ===
rho, pval = stats.spearmanr(smad_mean, coll_composite)
results['smad_collagen_rho'] = float(rho)
results['smad_collagen_pval'] = float(pval)
results['smad_collagen_r2'] = float(rho ** 2)
print(f"SMAD-collagen unconditional: rho={rho:.3f}, p={pval:.2e}, r2={rho**2:.3f}")

# === 2. FOS-collagen unconditional ===
if fos_idx is not None:
    rho, pval = stats.spearmanr(fos_expr_z, coll_composite_z)
    results['fos_collagen_unconditional_rho'] = float(rho)
    results['fos_collagen_unconditional_pval'] = float(pval)
    results['fos_collagen_unconditional_r2'] = float(rho ** 2)
    print(f"FOS-collagen unconditional: rho={rho:.3f}, p={pval:.2e}, r2={rho**2:.3f}")

# === 3. JUNB-collagen unconditional ===
if junb_idx is not None:
    rho, pval = stats.spearmanr(junb_expr_z, coll_composite_z)
    results['junb_collagen_unconditional_rho'] = float(rho)
    results['junb_collagen_unconditional_pval'] = float(pval)
    results['junb_collagen_unconditional_r2'] = float(rho ** 2)
    print(f"JUNB-collagen unconditional: rho={rho:.3f}, p={pval:.2e}, r2={rho**2:.3f}")

# === 4. SMAD tercile stratification ===
n_terciles = 3
tercile_labels = ['low', 'mid', 'high']
tercile_boundaries = np.percentile(smad_mean, [33.3, 66.7])
tercile_assignments = np.zeros(len(smad_mean), dtype=int)
tercile_assignments[smad_mean <= tercile_boundaries[0]] = 0
tercile_assignments[(smad_mean > tercile_boundaries[0]) & (smad_mean <= tercile_boundaries[1])] = 1
tercile_assignments[smad_mean > tercile_boundaries[1]] = 2

results['smad_terciles'] = {}
for t in range(3):
    mask_t = tercile_assignments == t
    n_t = mask_t.sum()
    print(f"SMAD tercile {tercile_labels[t]}: N={n_t}")

    fos_t = fos_expr_z[mask_t] if fos_idx is not None else None
    junb_t = junb_expr_z[mask_t] if junb_idx is not None else None
    coll_t = coll_composite_z[mask_t]
    smad_t = smad_mean[mask_t]

    # FOS-collagen within tercile
    if fos_t is not None:
        rho, pval = stats.spearmanr(fos_t, coll_t)
        results['smad_terciles'][f'fos_collagen_tercile_{t}'] = {
            'rho': float(rho), 'pval': float(pval), 'n': int(n_t)
        }
        print(f"  FOS-collagen within tercile {t}: rho={rho:.3f}, p={pval:.2e}")

    # JUNB-collagen within tercile
    if junb_t is not None:
        rho, pval = stats.spearmanr(junb_t, coll_t)
        results['smad_terciles'][f'junb_collagen_tercile_{t}'] = {
            'rho': float(rho), 'pval': float(pval), 'n': int(n_t)
        }
        print(f"  JUNB-collagen within tercile {t}: rho={rho:.3f}, p={pval:.2e}")

# === 5. SMAD age effect (young vs old) ===
X_young = adata_young_mme.X.toarray() if issparse(adata_young_mme.X) else np.asarray(adata_young_mme.X)
gnames_young = adata_young_mme.var_names.tolist()

smad_idx_young = [gnames_young.index(g) for g in smad_present]
smad_young = np.nanmean(X_young[:, smad_idx_young].astype(float), axis=1)
smad_young_z = (smad_young - np.nanmean(smad_young)) / np.nanstd(smad_young)
smad_old_z = smad_mean

# Cohen d
mean_diff = np.nanmean(smad_old_z) - np.nanmean(smad_young_z)
pooled_std = np.sqrt((np.nanstd(smad_old_z)**2 + np.nanstd(smad_young_z)**2) / 2)
d = mean_diff / pooled_std if pooled_std > 0 else 0

# t-test
t_stat, pval_t = stats.ttest_ind(smad_old_z, smad_young_z, nan_policy='omit')
results['smad_age_cohen_d'] = float(d)
results['smad_age_tstat'] = float(t_stat)
results['smad_age_pval'] = float(pval_t)
results['n_young_mme'] = int(len(smad_young_z))
results['n_old_mme'] = int(len(smad_old_z))
print(f"SMAD age effect: d={d:.3f}, t={t_stat:.2f}, p={pval_t:.2e}")

# === 6. Apply FDR correction ===
correlations = [
    ('smad_collagen', results.get('smad_collagen_pval', 1.0)),
    ('fos_collagen_unconditional', results.get('fos_collagen_unconditional_pval', 1.0)),
    ('junb_collagen_unconditional', results.get('junb_collagen_unconditional_pval', 1.0)),
]

for t in range(3):
    key = f'fos_collagen_tercile_{t}'
    if key in results.get('smad_terciles', {}):
        correlations.append((key, results['smad_terciles'][key]['pval']))
    key = f'junb_collagen_tercile_{t}'
    if key in results.get('smad_terciles', {}):
        correlations.append((key, results['smad_terciles'][key]['pval']))

test_names = [x[0] for x in correlations]
pvals = np.array([x[1] for x in correlations])
rejected, fdr_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

for i, name in enumerate(test_names):
    results[f'{name}_fdr'] = float(fdr_corrected[i])

# Store metadata
results['smad_targets_used'] = smad_present
results['collagen_genes_used'] = collagen_present
results['smad_tercile_boundaries'] = [float(x) for x in tercile_boundaries]

# === 7. Decision rules ===
results['decision'] = {}

# SMAD age effect
if results.get('smad_age_cohen_d', 0) > 0.3:
    results['decision']['smad_age_effect'] = 'POSITIVE (d > 0.3)'
elif results.get('smad_age_cohen_d', 0) < 0.2:
    results['decision']['smad_age_effect'] = 'NEGATIVE (d < 0.2)'
else:
    results['decision']['smad_age_effect'] = 'AMBIGUOUS (0.2 <= d <= 0.3)'

# SMAD-collagen correlation
if results.get('smad_collagen_rho', 0) > 0.25:
    results['decision']['smad_collagen'] = 'STRONG (> 0.25)'
elif results.get('smad_collagen_rho', 0) > 0.15:
    results['decision']['smad_collagen'] = 'MODERATE (0.15-0.25)'
else:
    results['decision']['smad_collagen'] = 'WEAK (< 0.15)'

# FOS-collagen mediation
fos_cond_rhos = [results['smad_terciles'].get(f'fos_collagen_tercile_{t}', {}).get('rho', None) for t in range(3)]
fos_cond_rhos = [r for r in fos_cond_rhos if r is not None]
if fos_cond_rhos:
    mean_fos_cond = np.mean(fos_cond_rhos)
    results['fos_collagen_conditional_mean'] = float(mean_fos_cond)
    if mean_fos_cond < 0.10:
        results['decision']['fos_mediated_by_smad'] = 'LIKELY (conditional rho < 0.10)'
    elif mean_fos_cond > 0.15:
        results['decision']['fos_mediated_by_smad'] = 'UNLIKELY (conditional rho > 0.15)'
    else:
        results['decision']['fos_mediated_by_smad'] = 'AMBIGUOUS (0.10-0.15)'

# Compare SMAD vs FOS vs JUNB
results['decision']['driver_ranking'] = sorted(
    [
        ('SMAD', results.get('smad_collagen_rho', 0)),
        ('FOS', results.get('fos_collagen_unconditional_rho', 0)),
        ('JUNB', results.get('junb_collagen_unconditional_rho', 0)),
    ],
    key=lambda x: abs(x[1]),
    reverse=True
)

# Save
with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_PATH}")
print(f"\n=== DECISION ===")
print(f"SMAD age effect: d={results.get('smad_age_cohen_d', 'N/A'):.3f}")
print(f"SMAD-collagen rho: {results.get('smad_collagen_rho', 'N/A'):.3f}")
print(f"FOS-collagen conditional mean: {results.get('fos_collagen_conditional_mean', 'N/A'):.3f}")
print(f"Driver ranking: {results['decision']['driver_ranking']}")
print(f"Decision: {results['decision']}")