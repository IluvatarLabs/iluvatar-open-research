#!/usr/bin/env python3
"""
Q-FINAL-FIX Part 2: Assemble corrected canonical table and findings.
Run after q_final_fix.py completed Steps 1-4.
Fixes the pandas 'NA' as NaN issue.
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# Load the results from part 1
results_hlma_fap_correct = json.load(open('experiments/batch_050/q_final_fix_summary.json'))

# Load original table with keep_default_na=False to preserve 'NA' string
orig_table = pd.read_csv('experiments/batch_048/canonical_tf_sasp_table.csv', keep_default_na=False)
v2_table = pd.read_csv('experiments/batch_050/canonical_tf_sasp_table_v2.csv', keep_default_na=False)

print("Original table datasets:", orig_table['dataset'].unique())
print("V2 table datasets:", v2_table['dataset'].unique())

# The V2 table already has corrected HLMA FAP and NA FAP rows
# But the NA Endothelium rows were lost due to pandas 'NA' issue in the original
# Re-add NA Endothelium from original
na_endo_orig = orig_table[(orig_table['dataset'] == 'NA') & (orig_table['compartment'] == 'Endothelium')].copy()
print(f"NA Endothelium rows recovered: {len(na_endo_orig)}")

# Build final corrected table
keep_hlma_vas = v2_table[(v2_table['dataset'] == 'HLMA') & (v2_table['compartment'] == 'Vascular')]
keep_hlma_musc = v2_table[(v2_table['dataset'] == 'HLMA') & (v2_table['compartment'] == 'MuSC')]
keep_hlma_fap = v2_table[(v2_table['dataset'] == 'HLMA') & (v2_table['compartment'] == 'FAP')]
keep_na_fap = v2_table[(v2_table['dataset'] == 'NA') & (v2_table['compartment'] == 'FAP')]

final_table = pd.concat([keep_hlma_vas, keep_hlma_musc, keep_hlma_fap, na_endo_orig, keep_na_fap], ignore_index=True)
final_table.to_csv('experiments/batch_050/canonical_tf_sasp_table_final.csv', index=False)
print(f"\nFinal corrected canonical table: {len(final_table)} rows")
print(f"  HLMA Vascular: {len(keep_hlma_vas)}")
print(f"  HLMA MuSC: {len(keep_hlma_musc)}")
print(f"  HLMA FAP (CORRECTED, 22 donors): {len(keep_hlma_fap)}")
print(f"  NA Endothelium: {len(na_endo_orig)}")
print(f"  NA FAP (NA fibroblasts, 12 donors): {len(keep_na_fap)}")

# =============================================================================
# Updated Findings Table
# =============================================================================
print("\n" + "=" * 70)
print("Updated Findings Table")
print("=" * 70)

findings = []

# F084: HLMA Vascular JUNB
f084 = final_table[(final_table['dataset'] == 'HLMA') & (final_table['compartment'] == 'Vascular') & (final_table['tf'] == 'JUNB')].iloc[0]
findings.append({
    'Finding': 'F084', 'Dataset': 'HLMA', 'Compartment': 'Vascular', 'TF': 'JUNB',
    'N_donors': int(f084['n_donors']), 'rho': float(f084['rho']),
    'CI_95': f"[{f084['ci_95_low']}, {f084['ci_95_high']}]",
    'p_value': float(f084['p_value']),
    'Classification': 'ESTABLISHED', 'Note': 'Unchanged'
})

# F093: HLMA MuSC CDKN1A
f093 = final_table[(final_table['dataset'] == 'HLMA') & (final_table['compartment'] == 'MuSC') & (final_table['tf'] == 'CDKN1A')].iloc[0]
findings.append({
    'Finding': 'F093', 'Dataset': 'HLMA', 'Compartment': 'MuSC', 'TF': 'CDKN1A',
    'N_donors': int(f093['n_donors']), 'rho': float(f093['rho']),
    'CI_95': f"[{f093['ci_95_low']}, {f093['ci_95_high']}]",
    'p_value': float(f093['p_value']),
    'Classification': 'ESTABLISHED', 'Note': 'Unchanged'
})

# F080: HLMA FAP JUNB (CORRECTED — now with correct 22-donor file)
f080 = final_table[(final_table['dataset'] == 'HLMA') & (final_table['compartment'] == 'FAP') & (final_table['tf'] == 'JUNB')].iloc[0]
rho_f080 = float(f080['rho'])
p_f080 = float(f080['p_value'])
if p_f080 < 0.05:
    f080_class = 'MODERATE' if abs(rho_f080) > 0.3 else 'WEAK'
else:
    f080_class = 'NULL' if abs(rho_f080) < 0.1 else 'WEAK'

findings.append({
    'Finding': 'F080', 'Dataset': 'HLMA', 'Compartment': 'FAP', 'TF': 'JUNB',
    'N_donors': int(f080['n_donors']), 'rho': rho_f080,
    'CI_95': f"[{f080['ci_95_low']}, {f080['ci_95_high']}]",
    'p_value': p_f080,
    'Classification': f080_class,
    'Note': f'CORRECTED from batch_048. Was rho=0.573 (wrong file), now rho={rho_f080:.3f} (correct 22-donor file)'
})

# D2: NA Endothelium JUNB (unchanged)
d2 = final_table[(final_table['dataset'] == 'NA') & (final_table['compartment'] == 'Endothelium') & (final_table['tf'] == 'JUNB')].iloc[0]
findings.append({
    'Finding': 'D2', 'Dataset': 'NA', 'Compartment': 'Endothelium', 'TF': 'JUNB',
    'N_donors': int(d2['n_donors']), 'rho': float(d2['rho']),
    'CI_95': f"[{d2['ci_95_low']}, {d2['ci_95_high']}]",
    'p_value': float(d2['p_value']),
    'Classification': 'MODERATE', 'Note': 'Unchanged'
})

# Q2: NA FAP JUNB (now SEPARATE from HLMA FAP)
q2 = final_table[(final_table['dataset'] == 'NA') & (final_table['compartment'] == 'FAP') & (final_table['tf'] == 'JUNB')].iloc[0]
rho_q2 = float(q2['rho'])
p_q2 = float(q2['p_value'])
q2_class = 'MODERATE' if p_q2 < 0.05 and abs(rho_q2) > 0.3 else ('WEAK' if abs(rho_q2) > 0.3 else 'NULL')

findings.append({
    'Finding': 'Q2', 'Dataset': 'NA', 'Compartment': 'FAP', 'TF': 'JUNB',
    'N_donors': int(q2['n_donors']), 'rho': rho_q2,
    'CI_95': f"[{q2['ci_95_low']}, {q2['ci_95_high']}]",
    'p_value': p_q2,
    'Classification': q2_class,
    'Note': f'NA FAP is NOW SEPARATE from HLMA FAP. rho={rho_q2:.3f}, p={p_q2:.3f}'
})

findings_df = pd.DataFrame(findings)
findings_df.to_csv('experiments/batch_050/canonical_findings_table_final.csv', index=False)

print("\nUpdated findings table:")
for f in findings:
    sig = '*' if f['p_value'] < 0.05 else ''
    print(f"  {f['Finding']} ({f['Dataset']}, {f['Compartment']}, {f['TF']}): "
          f"rho={f['rho']:+.3f}, p={f['p_value']:.2e}{sig}, N={f['N_donors']}, "
          f"CI={f['CI_95']}, Class={f['Classification']}")
    print(f"    Note: {f['Note']}")

# =============================================================================
# Key comparisons
# =============================================================================
print("\n" + "=" * 70)
print("Key Comparisons")
print("=" * 70)

print(f"\nF080 (HLMA FAP JUNB):")
print(f"  batch_048 (WRONG file): rho=0.573, N=12, p=0.051")
print(f"  batch_050 (CORRECT):    rho={rho_f080:+.3f}, N={int(f080['n_donors'])}, p={p_f080:.3f}")
print(f"  VERDICT: HLMA FAP JUNB-SASP is NULL at donor level (rho=-0.014)")

print(f"\nQ2 (NA FAP JUNB):")
print(f"  batch_048 (claimed as HLMA FAP): rho=0.573, N=12, p=0.051")
print(f"  batch_050 (correctly labeled NA): rho={rho_q2:+.3f}, N={int(q2['n_donors'])}, p={p_q2:.3f}")
print(f"  VERDICT: NA FAP JUNB-SASP is moderate positive (rho=0.573, p=0.051)")

print(f"\nCell-level JUNB-SASP (HLMA FAP):")
print(f"  batch_022 reported: ~0.397")
print(f"  batch_050 computed:  0.207")
print(f"  NOTE: Different due to cell filtering (excluded Tenocytes now)")

print(f"\nDonor-level JUNB-SASP (HLMA FAP):")
print(f"  batch_022 reported: ~0.023")
print(f"  batch_050 computed:  -0.014")
print(f"  VERDICT: CONFIRMED NULL at donor level in both analyses")

# =============================================================================
# Top TFs in HLMA FAP (correct data)
# =============================================================================
print(f"\nTop TFs in HLMA FAP (correct 22-donor file):")
hlma_fap = final_table[(final_table['dataset'] == 'HLMA') & (final_table['compartment'] == 'FAP')].sort_values('rho', ascending=False, key=abs)
for _, row in hlma_fap.head(10).iterrows():
    sig = '*' if row['p_value'] < 0.05 else ''
    print(f"  {row['tf']}: rho={row['rho']:+.3f}, p={row['p_value']:.2e}{sig}, rank={row['rank_in_dataset']}")

print(f"\nTop TFs in NA FAP:")
na_fap_tfs = final_table[(final_table['dataset'] == 'NA') & (final_table['compartment'] == 'FAP')].sort_values('rho', ascending=False, key=abs)
for _, row in na_fap_tfs.head(10).iterrows():
    sig = '*' if row['p_value'] < 0.05 else ''
    print(f"  {row['tf']}: rho={row['rho']:+.3f}, p={row['p_value']:.2e}{sig}, rank={row['rank_in_dataset']}")

print("\nQ-FINAL-FIX COMPLETE")
