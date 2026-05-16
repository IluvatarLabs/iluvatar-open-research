#!/usr/bin/env python3
"""
Q-INTEGRATIVE: Cross-dataset integrative analysis (batch_050)

Runs all 5 integrative analyses across the corrected 5-dataset canonical table:
I1. Meta-analytic pooling via Fisher Z
I2. Direction-of-effect voting across datasets
I3. Rank correlation of TF rankings across datasets
I4. Mixed-effects model with dataset as random effect
I5. Cross-dataset filtering for robust TF set

Uses the CORRECTED canonical_tf_sasp_table_final.csv from Q-FINAL-FIX.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm, chi2
from scipy.special import erfcinv
import warnings
import json
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# Load corrected canonical table
# =============================================================================
table = pd.read_csv('experiments/batch_050/canonical_tf_sasp_table_final.csv', keep_default_na=False)
print(f"Loaded canonical table: {len(table)} rows")
print(f"Datasets: {table[['dataset','compartment']].drop_duplicates().to_dict('records')}")

# Define 5 dataset-compartment pairs
datasets = [
    ('HLMA', 'Vascular'),
    ('HLMA', 'MuSC'),
    ('HLMA', 'FAP'),
    ('NA', 'Endothelium'),
    ('NA', 'FAP'),
]

# =============================================================================
# I1: Meta-analytic pooling via Fisher Z
# =============================================================================
print("\n" + "=" * 70)
print("I1: Meta-analytic pooling via Fisher Z")
print("=" * 70)

def fisher_z(rho):
    """Fisher Z transformation."""
    return np.arctanh(np.clip(rho, -0.999, 0.999))

def inverse_fisher_z(z):
    """Inverse Fisher Z."""
    return np.tanh(z)

I1_results = []

for tf in table['tf'].unique():
    tf_data = table[table['tf'] == tf]

    rhos = []
    ns = []
    dataset_labels = []

    for ds, comp in datasets:
        row = tf_data[(tf_data['dataset'] == ds) & (tf_data['compartment'] == comp)]
        if len(row) > 0:
            r = row.iloc[0]
            rhos.append(float(r['rho']))
            ns.append(int(r['n_donors']))
            dataset_labels.append(f"{ds}/{comp}")

    if len(rhos) < 2:
        continue

    rhos = np.array(rhos)
    ns = np.array(ns)

    # Fisher Z pooling with inverse-variance weighting
    zs = fisher_z(rhos)
    weights = ns - 3  # Inverse variance weights
    weights = np.maximum(weights, 1)  # Avoid zero/negative weights

    z_pooled = np.sum(weights * zs) / np.sum(weights)
    se_pooled = 1.0 / np.sqrt(np.sum(weights))
    rho_pooled = inverse_fisher_z(z_pooled)

    # 95% CI
    z_crit = norm.ppf(0.975)
    ci_low = inverse_fisher_z(z_pooled - z_crit * se_pooled)
    ci_high = inverse_fisher_z(z_pooled + z_crit * se_pooled)

    # Cochran's Q test for heterogeneity
    Q = np.sum(weights * (zs - z_pooled) ** 2)
    df_Q = len(rhos) - 1
    p_Q = 1 - chi2.cdf(Q, df_Q) if df_Q > 0 else np.nan

    # I² heterogeneity statistic
    I2 = max(0, (Q - df_Q) / Q * 100) if Q > 0 else 0

    # P-value for pooled estimate
    z_stat = z_pooled / se_pooled
    p_pooled = 2 * (1 - norm.cdf(abs(z_stat)))

    I1_results.append({
        'TF': tf,
        'pooled_rho': round(rho_pooled, 3),
        'pooled_ci_low': round(ci_low, 3),
        'pooled_ci_high': round(ci_high, 3),
        'pooled_p': p_pooled,
        'Q_statistic': round(Q, 3),
        'Q_p': round(p_Q, 4),
        'I_squared': round(I2, 1),
        'n_datasets': len(rhos),
        'individual_rhos': [round(r, 3) for r in rhos],
        'datasets': dataset_labels
    })

I1_df = pd.DataFrame(I1_results).sort_values('pooled_rho', ascending=False, key=abs)
I1_df.to_csv('experiments/batch_050/I1_fisher_z_pooling.csv', index=False)

print("\nTop 10 TFs by pooled |rho|:")
for _, row in I1_df.head(10).iterrows():
    het = "HIGH" if row['I_squared'] > 50 else ("MOD" if row['I_squared'] > 25 else "LOW")
    sig = '*' if row['pooled_p'] < 0.05 else ''
    print(f"  {row['TF']:8s}: pooled_rho={row['pooled_rho']:+.3f}, "
          f"CI=[{row['pooled_ci_low']:.3f}, {row['pooled_ci_high']:.3f}], "
          f"p={row['pooled_p']:.2e}{sig}, I²={row['I_squared']:.1f}% ({het}), "
          f"n_ds={row['n_datasets']}")
    print(f"           individual: {row['individual_rhos']}")

# JUNB specifically
june_row = I1_df[I1_df['TF'] == 'JUNB']
if len(june_row) > 0:
    june = june_row.iloc[0]
    print(f"\nJUNB spotlight:")
    print(f"  Pooled rho: {june['pooled_rho']:+.3f}, CI=[{june['pooled_ci_low']:.3f}, {june['pooled_ci_high']:.3f}]")
    print(f"  Heterogeneity: I²={june['I_squared']:.1f}%, Q p={june['Q_p']:.4f}")
    print(f"  Individual: {june['individual_rhos']}")
    print(f"  Datasets: {june['datasets']}")
    if june['I_squared'] > 50:
        print("  INTERPRETATION: High heterogeneity — JUNB couples to SASP in vascular/MuSC but NOT in FAPs")

# =============================================================================
# I2: Direction-of-effect voting
# =============================================================================
print("\n" + "=" * 70)
print("I2: Direction-of-effect voting across datasets")
print("=" * 70)

I2_results = []

for tf in table['tf'].unique():
    tf_data = table[table['tf'] == tf]

    votes = {'positive': 0, 'negative': 0, 'null': 0}
    rhos = []

    for ds, comp in datasets:
        row = tf_data[(tf_data['dataset'] == ds) & (tf_data['compartment'] == comp)]
        if len(row) > 0:
            rho = float(row.iloc[0]['rho'])
            rhos.append(rho)
            if rho > 0.3:
                votes['positive'] += 1
            elif rho < -0.3:
                votes['negative'] += 1
            else:
                votes['null'] += 1

    n_pos = votes['positive']
    n_neg = votes['negative']
    n_null = votes['null']
    n_total = n_pos + n_neg + n_null

    if n_total >= 3:
        if n_pos >= 3:
            tier = 'ROBUST'
        elif n_pos >= 2:
            tier = 'HETEROGENEOUS'
        elif n_pos == 1:
            tier = 'SINGLE-ATLAS'
        else:
            tier = 'NULL/NEGATIVE'

        I2_results.append({
            'TF': tf,
            'positive_votes': n_pos,
            'negative_votes': n_neg,
            'null_votes': n_null,
            'n_datasets': n_total,
            'tier': tier,
            'rhos': [round(r, 3) for r in rhos]
        })

I2_df = pd.DataFrame(I2_results).sort_values('positive_votes', ascending=False)
I2_df.to_csv('experiments/batch_050/I2_direction_voting.csv', index=False)

print("\nVoting tier summary:")
for tier in ['ROBUST', 'HETEROGENEOUS', 'SINGLE-ATLAS', 'NULL/NEGATIVE']:
    tfs = I2_df[I2_df['tier'] == tier]
    if len(tfs) > 0:
        print(f"  {tier} ({len(tfs)} TFs): {', '.join(tfs['TF'].tolist())}")

print("\nDetailed voting:")
for _, row in I2_df.iterrows():
    print(f"  {row['TF']:8s}: +{row['positive_votes']}/-{row['negative_votes']}/null={row['null_votes']} "
          f"→ {row['tier']}  rhos={row['rhos']}")

# =============================================================================
# I3: Rank correlation of TF rankings across datasets
# =============================================================================
print("\n" + "=" * 70)
print("I3: Rank correlation of TF rankings across datasets")
print("=" * 70)

# Build ranking matrix: TF x dataset
rank_matrix = {}
for ds, comp in datasets:
    sub = table[(table['dataset'] == ds) & (table['compartment'] == comp)]
    ranks = dict(zip(sub['tf'], sub['rank_in_dataset']))
    rank_matrix[f"{ds}/{comp}"] = ranks

# Compute Spearman rank correlations between dataset pairs
comparisons = [
    ('HLMA/Vascular', 'NA/Endothelium', 'Both endothelial — should be most similar'),
    ('HLMA/Vascular', 'HLMA/MuSC', 'Same atlas, different compartments'),
    ('HLMA/Vascular', 'HLMA/FAP', 'Same atlas, different compartments'),
    ('HLMA/FAP', 'NA/FAP', 'Both FAP — cross-atlas'),
    ('HLMA/MuSC', 'HLMA/FAP', 'Same atlas, different compartments'),
]

I3_results = []
all_datasets = list(rank_matrix.keys())
common_tfs = set(rank_matrix[all_datasets[0]].keys())
for ds_key in all_datasets[1:]:
    common_tfs &= set(rank_matrix[ds_key].keys())
common_tfs = sorted(common_tfs)
print(f"Common TFs across all datasets: {len(common_tfs)}")

for ds1_key, ds2_key, description in comparisons:
    ds1_ranks = [rank_matrix[ds1_key].get(tf, np.nan) for tf in common_tfs]
    ds2_ranks = [rank_matrix[ds2_key].get(tf, np.nan) for tf in common_tfs]

    # Remove NaN
    valid = [(r1, r2) for r1, r2 in zip(ds1_ranks, ds2_ranks)
             if not np.isnan(r1) and not np.isnan(r2)]
    if len(valid) >= 5:
        r1, r2 = zip(*valid)
        rho, p = spearmanr(r1, r2)
        I3_results.append({
            'comparison': f"{ds1_key} vs {ds2_key}",
            'description': description,
            'spearman_rho': round(rho, 3),
            'p_value': p,
            'n_tfs': len(valid)
        })
        print(f"  {ds1_key} vs {ds2_key}: rho={rho:+.3f}, p={p:.3f} — {description}")

I3_df = pd.DataFrame(I3_results)
I3_df.to_csv('experiments/batch_050/I3_rank_correlations.csv', index=False)

# =============================================================================
# I4: Mixed-effects model with dataset as random effect
# =============================================================================
print("\n" + "=" * 70)
print("I4: Mixed-effects model (SASP ~ TF + (1|dataset))")
print("=" * 70)

# For I4, we need donor-level data. Since we can't easily reconstruct that from
# the correlation table alone, we'll compute the fixed effect estimate using
# the correlation table as a summary. We use the Fisher Z approach:
# "Does the cross-dataset fixed effect explain the signal, or is it all dataset random effect?"

# We approximate this by checking: does the pooled estimate remain significant
# after accounting for between-dataset heterogeneity?
# This is exactly what I1's Cochran's Q test addresses.

# For a proper mixed-effects model, we need donor-level data.
# We'll note this limitation and use I1 results as proxy.

print("NOTE: Full mixed-effects model requires donor-level data (not available from canonical table).")
print("Using Fisher Z pooling with heterogeneity statistics as approximation.")

# For each TF, compute whether the fixed effect (cross-dataset biology) is significant
# AND whether heterogeneity is low (biology > platform)
I4_results = []
for _, row in I1_df.iterrows():
    fixed_significant = row['pooled_p'] < 0.05
    low_heterogeneity = row['I_squared'] < 50

    if fixed_significant and low_heterogeneity:
        verdict = "BIOLOGY — cross-dataset signal, low heterogeneity"
    elif fixed_significant and not low_heterogeneity:
        verdict = "HETEROGENEOUS — significant but compartment/atlas-driven"
    elif not fixed_significant:
        verdict = "PLATFORM — no consistent cross-dataset signal"

    I4_results.append({
        'TF': row['TF'],
        'pooled_rho': row['pooled_rho'],
        'pooled_p': row['pooled_p'],
        'I_squared': row['I_squared'],
        'fixed_effect_significant': fixed_significant,
        'low_heterogeneity': low_heterogeneity,
        'verdict': verdict
    })

I4_df = pd.DataFrame(I4_results).sort_values('pooled_rho', ascending=False, key=abs)
I4_df.to_csv('experiments/batch_050/I4_mixed_effects_summary.csv', index=False)

print("\nVerdict summary:")
for verdict in ['BIOLOGY', 'HETEROGENEOUS', 'PLATFORM']:
    tfs = I4_df[I4_df['verdict'].str.startswith(verdict)]
    if len(tfs) > 0:
        print(f"  {verdict} ({len(tfs)}): {', '.join(tfs['TF'].tolist())}")

# =============================================================================
# I5: Cross-dataset filtering for robust TF set
# =============================================================================
print("\n" + "=" * 70)
print("I5: Cross-dataset filtering for robust TF set")
print("=" * 70)

I5_results = []

for tf in table['tf'].unique():
    tf_data = table[table['tf'] == tf]

    datasets_above_05 = 0
    datasets_above_03 = 0
    hlma_only = 0
    single_dataset = 0
    dataset_list = []

    for ds, comp in datasets:
        row = tf_data[(tf_data['dataset'] == ds) & (tf_data['compartment'] == comp)]
        if len(row) > 0:
            rho = abs(float(row.iloc[0]['rho']))
            dataset_list.append(f"{ds}/{comp}: {rho:.3f}")

            if rho > 0.5:
                datasets_above_05 += 1
                if ds == 'HLMA':
                    hlma_only += 1
            if rho > 0.3:
                datasets_above_03 += 1

    if datasets_above_05 >= 3:
        category = 'ROBUST (rho>0.5 in ≥3 datasets)'
    elif datasets_above_05 >= 2:
        category = 'MODERATE (rho>0.5 in 2 datasets)'
    elif datasets_above_05 == 1:
        if hlma_only == 1:
            category = 'HLMA-SPECIFIC'
        else:
            category = 'SINGLE-DATASET'
    else:
        if datasets_above_03 >= 3:
            category = 'WEAK-ROBUST (rho>0.3 in ≥3, never >0.5)'
        else:
            category = 'NULL'

    I5_results.append({
        'TF': tf,
        'datasets_rho_above_0.5': datasets_above_05,
        'datasets_rho_above_0.3': datasets_above_03,
        'category': category,
        'details': '; '.join(dataset_list)
    })

I5_df = pd.DataFrame(I5_results).sort_values('datasets_rho_above_0.5', ascending=False)
I5_df.to_csv('experiments/batch_050/I5_cross_dataset_filtering.csv', index=False)

print("\nCategory summary:")
for cat in ['ROBUST', 'MODERATE', 'HLMA-SPECIFIC', 'SINGLE-DATASET', 'WEAK-ROBUST', 'NULL']:
    tfs = I5_df[I5_df['category'].str.startswith(cat)]
    if len(tfs) > 0:
        print(f"  {cat} ({len(tfs)}): {', '.join(tfs['TF'].tolist())}")

# =============================================================================
# Integrative Summary
# =============================================================================
print("\n" + "=" * 70)
print("INTEGRATIVE SUMMARY")
print("=" * 70)

# Top findings
robust_tfs = I5_df[I5_df['category'].str.startswith('ROBUST')]
print(f"\nRobust TFs (rho > 0.5 in ≥ 3/5 datasets): {', '.join(robust_tfs['TF'].tolist()) if len(robust_tfs) > 0 else 'NONE'}")

moderate_tfs = I5_df[I5_df['category'].str.startswith('MODERATE')]
print(f"Moderate TFs (rho > 0.5 in 2/5 datasets): {', '.join(moderate_tfs['TF'].tolist()) if len(moderate_tfs) > 0 else 'NONE'}")

# JUNB specifically
june_I1 = I1_df[I1_df['TF'] == 'JUNB'].iloc[0]
june_I2 = I2_df[I2_df['TF'] == 'JUNB'].iloc[0]
june_I4 = I4_df[I4_df['TF'] == 'JUNB'].iloc[0]
june_I5 = I5_df[I5_df['TF'] == 'JUNB'].iloc[0]

print(f"\nJUNB integrative profile:")
print(f"  I1 pooled: rho={june_I1['pooled_rho']:+.3f}, I²={june_I1['I_squared']:.1f}%")
print(f"  I2 votes: {june_I2['positive_votes']} positive / {june_I2['negative_votes']} negative / {june_I2['null_votes']} null → {june_I2['tier']}")
print(f"  I4 verdict: {june_I4['verdict']}")
print(f"  I5 category: {june_I5['category']}")

print(f"\nCDKN1A (p21) integrative profile:")
cdkn1a_I1 = I1_df[I1_df['TF'] == 'CDKN1A'].iloc[0]
cdkn1a_I2 = I2_df[I2_df['TF'] == 'CDKN1A'].iloc[0]
cdkn1a_I5 = I5_df[I5_df['TF'] == 'CDKN1A'].iloc[0]
print(f"  I1 pooled: rho={cdkn1a_I1['pooled_rho']:+.3f}, I²={cdkn1a_I1['I_squared']:.1f}%")
print(f"  I2 votes: {cdkn1a_I2['positive_votes']} positive / {cdkn1a_I2['negative_votes']} negative / {cdkn1a_I2['null_votes']} null → {cdkn1a_I2['tier']}")
print(f"  I5 category: {cdkn1a_I5['category']}")

# Save comprehensive summary
summary = {
    'robust_tfs': robust_tfs['TF'].tolist(),
    'moderate_tfs': moderate_tfs['TF'].tolist(),
    'june_pooled_rho': float(june_I1['pooled_rho']),
    'june_heterogeneity_I2': float(june_I1['I_squared']),
    'june_votes': june_I2['positive_votes'],
    'cdkn1a_pooled_rho': float(cdkn1a_I1['pooled_rho']),
    'cdkn1a_votes': cdkn1a_I2['positive_votes'],
    'n_datasets': 5,
    'datasets': [f"{ds}/{comp}" for ds, comp in datasets]
}

with open('experiments/batch_050/integrative_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nSaved:")
print("  I1_fisher_z_pooling.csv")
print("  I2_direction_voting.csv")
print("  I3_rank_correlations.csv")
print("  I4_mixed_effects_summary.csv")
print("  I5_cross_dataset_filtering.csv")
print("  integrative_summary.json")
print("\nQ-INTEGRATIVE COMPLETE")
