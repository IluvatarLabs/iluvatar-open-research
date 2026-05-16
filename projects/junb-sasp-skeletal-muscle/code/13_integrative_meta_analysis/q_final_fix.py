#!/usr/bin/env python3
"""
Q-FINAL-FIX (batch_050): Correct HLMA FAP Analysis

CRITICAL BUG FIX: batch_048 loaded SKM_fibroblasts_Schwann_human_2023-06-22.h5ad
(the Nature Aging fibroblast file, 12 donors, 20,611 cells) instead of
OMIX004308-02.h5ad (the real HLMA FAP file, 22 donors, 40,389 cells).

This script:
1. Loads the CORRECT HLMA FAP file (OMIX004308-02.h5ad)
2. Filters to FAP subtypes only (excludes Tenocyte)
3. Computes donor-level TF-SASP correlations with canonical SASP12 panel
4. Re-verifies F080 (cell-level AND donor-level JUNB-SASP)
5. Replaces HLMA FAP rows in canonical table
6. Computes NA FAP separately using Nature Aging fibroblasts
7. Produces updated canonical table with 5 correct datasets

Canonical SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8',
                    'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

Methodology: Identical to batch_048 (raw mean, Fisher Z CI, Spearman at donor level)
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, norm
import json
import warnings
import gc
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# CANONICAL CONFIGURATION
# =============================================================================
SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8',
           'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

TARGET_TFS = ['JUNB', 'JUN', 'JUND', 'FOS', 'FOSB', 'FOSL1', 'FOSL2',
              'KLF2', 'KLF4', 'KLF6', 'KLF10', 'ATF3', 'EGR1', 'EGR2',
              'IRF1', 'CEBPB', 'CEBPD', 'RELA', 'NFKB1', 'STAT3', 'CDKN1A']

# FAP subtypes to include (exclude Tenocyte per research_state directive)
FAP_SUBTYPES = ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP', 'CD99+ FAP']


def fisher_z_ci(rho, n, alpha=0.05):
    """Fisher Z-transformed 95% CI (batch_047 R8 method)."""
    if n < 4 or np.isnan(rho):
        return np.nan, np.nan
    rho_clipped = np.clip(rho, -0.999, 0.999)
    z = np.arctanh(rho_clipped)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    rho_low = np.tanh(z - z_crit * se)
    rho_high = np.tanh(z + z_crit * se)
    return rho_low, rho_high


def compute_donor_correlation(adata, tf_col, sasp_cols, group_col,
                              available_sasp=None):
    """
    Compute donor-level Spearman correlation between TF and SASP composite.
    SASP composite computed ONCE per dataset, not per TF.
    Uses RAW MEAN (no Z-score normalization).
    """
    if available_sasp is None:
        available_sasp = [g for g in sasp_cols if g in adata.var_names]

    if tf_col not in adata.var_names:
        return None
    if len(available_sasp) == 0:
        return None

    # Per-cell SASP composite (RAW MEAN)
    X = adata.to_df()
    X['SASP_composite'] = X[available_sasp].mean(axis=1)
    X['group'] = adata.obs[group_col].values

    # Donor-level means
    donor_means = X.groupby('group').agg({
        tf_col: 'mean',
        'SASP_composite': 'mean'
    }).reset_index()

    if len(donor_means) < 3:
        return None

    tf_vals = donor_means[tf_col].values
    sasp_vals = donor_means['SASP_composite'].values

    rho, p = spearmanr(tf_vals, sasp_vals)
    ci_low, ci_high = fisher_z_ci(rho, len(donor_means))

    return {
        'tf': tf_col,
        'n_donors': len(donor_means),
        'detected_sasp_genes': len(available_sasp),
        'rho': rho,
        'p_value': p,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
    }


def compute_rank_in_dataset(results_list):
    """Rank TFs by absolute rho within dataset."""
    rhos = [r['rho'] for r in results_list if r is not None]
    abs_rhos = [abs(r) for r in rhos]
    sorted_idx = np.argsort(abs_rhos)[::-1]
    ranks = [0] * len(results_list)
    for rank, idx in enumerate(sorted_idx, 1):
        ranks[idx] = rank
    for i, r in enumerate(results_list):
        if r is not None:
            r['rank_in_dataset'] = ranks[i]
    return results_list


# =============================================================================
# STEP 1: Load CORRECT HLMA FAP file (OMIX004308-02.h5ad)
# =============================================================================
print("=" * 70)
print("STEP 1: Loading CORRECT HLMA FAP file")
print("=" * 70)

# Load with only needed genes to manage memory (11.46 GB file)
genes_needed = list(set(SASP12 + TARGET_TFS))
ad_fap_hlma = sc.read_h5ad('data/OMIX004308-02.h5ad')

print(f"Loaded: {ad_fap_hlma.shape[0]} cells, {ad_fap_hlma.shape[1]} genes")
print(f"Donors (sample): {ad_fap_hlma.obs['sample'].nunique()}")
print(f"Donors: {sorted(ad_fap_hlma.obs['sample'].unique())}")
print(f"Annotation values: {ad_fap_hlma.obs['Annotation'].value_counts().to_dict()}")

# Filter to FAP subtypes only (exclude Tenocyte)
fap_mask = ad_fap_hlma.obs['Annotation'].isin(FAP_SUBTYPES)
ad_fap = ad_fap_hlma[fap_mask].copy()
print(f"\nAfter FAP subtype filter: {ad_fap.shape[0]} cells (excluded Tenocyte)")
print(f"Subtypes: {ad_fap.obs['Annotation'].value_counts().to_dict()}")
print(f"Donors after filter: {ad_fap.obs['sample'].nunique()}")

# Verify schema matches Vascular/MuSC files
for col in ['sample', 'Annotation', 'age_pop']:
    assert col in ad_fap.obs.columns, f"Missing column: {col}"

# Check gene detection
available_sasp_fap = [g for g in SASP12 if g in ad_fap.var_names]
print(f"\nSASP12 detected: {len(available_sasp_fap)}/12: {available_sasp_fap}")
missing_sasp = [g for g in SASP12 if g not in ad_fap.var_names]
print(f"Missing SASP: {missing_sasp}")

detected_tfs = [g for g in TARGET_TFS if g in ad_fap.var_names]
print(f"TFs detected: {len(detected_tfs)}/21")

# =============================================================================
# STEP 2: Compute HLMA FAP donor-level correlations (CORRECT FILE)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: HLMA FAP donor-level TF-SASP correlations (CORRECT)")
print("=" * 70)

results_hlma_fap_correct = []
for tf in TARGET_TFS:
    r = compute_donor_correlation(ad_fap, tf, SASP12, 'sample',
                                  available_sasp=available_sasp_fap)
    if r:
        results_hlma_fap_correct.append(r)
        sig = '*' if r['p_value'] < 0.05 else ''
        print(f"  {tf}: rho={r['rho']:+.3f}, p={r['p_value']:.2e}{sig}, "
              f"n={r['n_donors']}, CI=[{r['ci_95_low']:.3f}, {r['ci_95_high']:.3f}]")

results_hlma_fap_correct = compute_rank_in_dataset(results_hlma_fap_correct)

# =============================================================================
# STEP 3: F080 re-verification (cell-level AND donor-level JUNB-SASP)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: F080 Re-verification")
print("=" * 70)

# Cell-level JUNB-SASP
X_fap = ad_fap.to_df()
X_fap['SASP_composite'] = X_fap[available_sasp_fap].mean(axis=1)
rho_cell, p_cell = spearmanr(X_fap['JUNB'].values, X_fap['SASP_composite'].values)
print(f"Cell-level JUNB-SASP: rho={rho_cell:.3f}, p={p_cell:.2e}")
print(f"  (batch_022 reported: ~0.397)")

# Donor-level JUNB-SASP
june_donor = X_fap.groupby(ad_fap.obs['sample'].values).agg({
    'JUNB': 'mean',
    'SASP_composite': 'mean'
})
rho_donor, p_donor = spearmanr(june_donor['JUNB'].values, june_donor['SASP_composite'].values)
ci_low_d, ci_high_d = fisher_z_ci(rho_donor, len(june_donor))
print(f"Donor-level JUNB-SASP: rho={rho_donor:.3f}, p={p_donor:.2e}, "
      f"N={len(june_donor)}, CI=[{ci_low_d:.3f}, {ci_high_d:.3f}]")
print(f"  (batch_022 reported: ~0.023 at donor level)")
print(f"  (batch_048 reported: 0.573 — WRONG FILE)")

# Free memory
del ad_fap_hlma
gc.collect()

# =============================================================================
# STEP 4: Load NA FAP SEPARATELY (Nature Aging fibroblasts)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: NA FAP (Nature Aging fibroblasts)")
print("=" * 70)

ad_na_fib = sc.read_h5ad('data/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad')
# Filter to FB cells (fibroblasts only, exclude Schwann cells and Tenocytes)
fb_mask = ad_na_fib.obs['annotation_level1'] == 'FB'
ad_na_fb = ad_na_fib[fb_mask].copy()
available_sasp_na_fb = [g for g in SASP12 if g in ad_na_fb.var_names]

print(f"NA FB cells: {ad_na_fb.shape[0]}, Donors: {ad_na_fb.obs['DonorID'].nunique()}")
print(f"SASP12 detected: {len(available_sasp_na_fb)}/12: {available_sasp_na_fb}")

results_na_fap = []
for tf in TARGET_TFS:
    r = compute_donor_correlation(ad_na_fb, tf, SASP12, 'DonorID',
                                  available_sasp=available_sasp_na_fb)
    if r:
        results_na_fap.append(r)
        sig = '*' if r['p_value'] < 0.05 else ''
        print(f"  {tf}: rho={r['rho']:+.3f}, p={r['p_value']:.2e}{sig}, n={r['n_donors']}")

results_na_fap = compute_rank_in_dataset(results_na_fap)

# =============================================================================
# STEP 5: Load existing canonical table and REPLACE HLMA FAP + NA FAP rows
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Updating Canonical Tables")
print("=" * 70)

def make_table(results, dataset, compartment, note=""):
    """Convert results list to DataFrame."""
    rows = []
    for r in results:
        if r:
            rows.append({
                'dataset': dataset,
                'compartment': compartment,
                'note': note,
                'tf': r['tf'],
                'n_donors': r['n_donors'],
                'detected_sasp_genes': r['detected_sasp_genes'],
                'rho': round(r['rho'], 3),
                'p_value': r['p_value'],
                'ci_95_low': round(r['ci_95_low'], 3) if not np.isnan(r['ci_95_low']) else np.nan,
                'ci_95_high': round(r['ci_95_high'], 3) if not np.isnan(r['ci_95_high']) else np.nan,
                'rank_in_dataset': r.get('rank_in_dataset', 0)
            })
    return pd.DataFrame(rows)


# Load the original canonical table and keep HLMA Vascular, HLMA MuSC, NA Endothelium
orig_table = pd.read_csv('experiments/batch_048/canonical_tf_sasp_table.csv')
keep = orig_table[
    ((orig_table['dataset'] == 'HLMA') & (orig_table['compartment'] == 'Vascular')) |
    ((orig_table['dataset'] == 'HLMA') & (orig_table['compartment'] == 'MuSC')) |
    ((orig_table['dataset'] == 'NA') & (orig_table['compartment'] == 'Endothelium'))
].copy()

# Build new HLMA FAP and NA FAP rows
hlma_fap_table = make_table(results_hlma_fap_correct, 'HLMA', 'FAP',
                            note='CORRECTED: OMIX004308-02.h5ad, 22 donors')
na_fap_table = make_table(results_na_fap, 'NA', 'FAP',
                          note='Nature Aging FB (Kedlian 2024), 12 donors')

# Combine
new_canonical = pd.concat([keep, hlma_fap_table, na_fap_table], ignore_index=True)
new_canonical.to_csv('experiments/batch_050/canonical_tf_sasp_table_v2.csv', index=False)
print(f"Saved canonical_tf_sasp_table_v2.csv: {len(new_canonical)} rows")
print(f"  HLMA Vascular: {len(keep[keep['compartment'] == 'Vascular'])} rows")
print(f"  HLMA MuSC: {len(keep[keep['compartment'] == 'MuSC'])} rows")
print(f"  HLMA FAP (CORRECTED): {len(hlma_fap_table)} rows")
print(f"  NA Endothelium: {len(keep[keep['compartment'] == 'Endothelium'])} rows")
print(f"  NA FAP (NEW): {len(na_fap_table)} rows")

# =============================================================================
# STEP 6: Updated Findings Table
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Updated Findings Table")
print("=" * 70)

# Get key findings from new data
def get_tf_result(results, tf_name):
    for r in results:
        if r and r['tf'] == tf_name:
            return r
    return None

f080_new = get_tf_result(results_hlma_fap_correct, 'JUNB')
q2_new = get_tf_result(results_na_fap, 'JUNB')

findings = []

# F084: HLMA Vascular JUNB (unchanged)
f084 = orig_table[(orig_table['dataset'] == 'HLMA') & (orig_table['compartment'] == 'Vascular') & (orig_table['tf'] == 'JUNB')].iloc[0]
findings.append({
    'Finding': 'F084', 'Dataset': 'HLMA', 'Compartment': 'Vascular', 'TF': 'JUNB',
    'N_donors': int(f084['n_donors']), 'rho': float(f084['rho']),
    'CI_95': f"[{f084['ci_95_low']:.3f}, {f084['ci_95_high']:.3f}]",
    'p_value': float(f084['p_value']),
    'Classification': 'ESTABLISHED',
    'Note': 'Unchanged from batch_048'
})

# F093: HLMA MuSC CDKN1A (unchanged)
f093 = orig_table[(orig_table['dataset'] == 'HLMA') & (orig_table['compartment'] == 'MuSC') & (orig_table['tf'] == 'CDKN1A')].iloc[0]
findings.append({
    'Finding': 'F093', 'Dataset': 'HLMA', 'Compartment': 'MuSC', 'TF': 'CDKN1A',
    'N_donors': int(f093['n_donors']), 'rho': float(f093['rho']),
    'CI_95': f"[{f093['ci_95_low']:.3f}, {f093['ci_95_high']:.3f}]",
    'p_value': float(f093['p_value']),
    'Classification': 'ESTABLISHED',
    'Note': 'Unchanged from batch_048'
})

# F080: HLMA FAP JUNB (CORRECTED)
if f080_new:
    f080_class = 'MODERATE' if f080_new['rho'] > 0.3 else ('NULL' if abs(f080_new['rho']) < 0.1 else 'WEAK')
    findings.append({
        'Finding': 'F080', 'Dataset': 'HLMA', 'Compartment': 'FAP', 'TF': 'JUNB',
        'N_donors': f080_new['n_donors'], 'rho': round(f080_new['rho'], 3),
        'CI_95': f"[{f080_new['ci_95_low']:.3f}, {f080_new['ci_95_high']:.3f}]",
        'p_value': f080_new['p_value'],
        'Classification': f080_class,
        'Note': f'CORRECTED: Was {f080_new["n_donors"]} donors (was 12 in wrong file)'
    })

# D2: NA Endothelium JUNB (unchanged)
d2 = orig_table[(orig_table['dataset'] == 'NA') & (orig_table['compartment'] == 'Endothelium') & (orig_table['tf'] == 'JUNB')].iloc[0]
findings.append({
    'Finding': 'D2', 'Dataset': 'NA', 'Compartment': 'Endothelium', 'TF': 'JUNB',
    'N_donors': int(d2['n_donors']), 'rho': float(d2['rho']),
    'CI_95': f"[{d2['ci_95_low']:.3f}, {d2['ci_95_high']:.3f}]",
    'p_value': float(d2['p_value']),
    'Classification': 'MODERATE',
    'Note': 'Unchanged from batch_048'
})

# Q2: NA FAP JUNB (NEW - correct NA data)
if q2_new:
    q2_class = 'MODERATE' if q2_new['p_value'] < 0.05 else ('WEAK' if abs(q2_new['rho']) > 0.3 else 'NULL')
    findings.append({
        'Finding': 'Q2', 'Dataset': 'NA', 'Compartment': 'FAP', 'TF': 'JUNB',
        'N_donors': q2_new['n_donors'], 'rho': round(q2_new['rho'], 3),
        'CI_95': f"[{q2_new['ci_95_low']:.3f}, {q2_new['ci_95_high']:.3f}]",
        'p_value': q2_new['p_value'],
        'Classification': q2_class,
        'Note': 'CORRECTED: NA FAP is now separate from HLMA FAP'
    })

findings_df = pd.DataFrame(findings)
findings_df.to_csv('experiments/batch_050/canonical_findings_table_v2.csv', index=False)

print("\nUpdated findings table:")
for f in findings:
    sig = '*' if f['p_value'] < 0.05 else ''
    print(f"  {f['Finding']} ({f['Dataset']}, {f['Compartment']}, {f['TF']}): "
          f"rho={f['rho']:+.3f}, p={f['p_value']:.2e}{sig}, N={f['N_donors']}, "
          f"CI={f['CI_95']}, Class={f['Classification']}")
    if 'Note' in f:
        print(f"    Note: {f['Note']}")

# =============================================================================
# STEP 7: Verify D001_04 revision needed
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: D001_04 Revision Check")
print("=" * 70)

# The original D001_04 said "NA FAP = HLMA FAP (same 12 donors)"
# Now we know they are DIFFERENT datasets
# HLMA FAP: 22 donors (OM1-OM9, P3,P5,P13,P17,P21,P23,P26,P29, YM1-YM5)
# NA FAP: 12 donors (339C, 343B, 362C, 367C, 411C, 464C, 470BR, 582C, 583B, 591C, 621B, 640C)

print("D001_04 REVISED: NA FAPs and HLMA FAPs are DIFFERENT datasets")
print(f"  HLMA FAP: {f080_new['n_donors'] if f080_new else '?'} donors (OM+P+YM series)")
print(f"  NA FAP: {q2_new['n_donors'] if q2_new else '?'} donors (C-series DonorIDs)")
print("  These must be treated as separate evidence, NOT merged")

# Save summary
summary = {
    'bug_fixed': True,
    'bug_description': 'batch_048 loaded SKM_fibroblasts_Schwann (NA fibroblasts) instead of OMIX004308-02 (HLMA FAPs)',
    'hlma_fap_donors': f080_new['n_donors'] if f080_new else None,
    'na_fap_donors': q2_new['n_donors'] if q2_new else None,
    'f080_cell_rho': float(rho_cell),
    'f080_donor_rho': float(rho_donor),
    'f080_donor_p': float(p_donor),
    'f080_donor_ci': [float(ci_low_d), float(ci_high_d)],
    'd001_04_revised': True,
    'd001_04_new': 'NA FAPs and HLMA FAPs are DIFFERENT datasets with different donors'
}

with open('experiments/batch_050/q_final_fix_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved q_final_fix_summary.json")
print("\nQ-FINAL-FIX COMPLETE")
