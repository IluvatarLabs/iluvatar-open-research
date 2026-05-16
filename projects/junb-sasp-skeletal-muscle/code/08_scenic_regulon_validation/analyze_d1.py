#!/usr/bin/env python3
"""
batch_054 Phase F: Analyze SCENIC D1 results against pre-registered decision rules.
Reads: d1_correlations_all.csv, d1_summary.json, d1_regulons_*.csv
Outputs: d1_analysis.json with decision rule application
"""
import json
import pandas as pd
import numpy as np
import sys
import os

OUTDIR = "experiments/batch_054"

# Pre-registered decision rules from brief.md
DECISION_RULES = {
    'JUNB': {
        'validated': 0.60,
        'inconclusive_low': 0.30,
    },
    'KLF10': {
        'validated': 0.50,
        'inconclusive_low': 0.20,
        'note': 'Underpowered at N=22; null = INCONCLUSIVE, not NOT VALIDATED',
    },
}

def apply_decision_rule(tf, dataset, rho, n):
    """Apply pre-registered decision rules for a TF in a specific compartment."""
    rules = DECISION_RULES.get(tf)
    if rules is None:
        return 'NOT PRE-REGISTERED'

    # Map datasets to expected compartments
    expected = {
        'JUNB': 'HLMA_Vascular',
        'KLF10': 'HLMA_FAP',
    }

    is_primary = (dataset == expected.get(tf))

    if not is_primary:
        return 'SECONDARY'  # Not a primary test

    if rho >= rules['validated']:
        return 'VALIDATED'
    elif rho >= rules['inconclusive_low']:
        return 'INCONCLUSIVE'
    else:
        if tf == 'KLF10':
            return 'INCONCLUSIVE (underpowered)'
        return 'NOT VALIDATED'


def compute_power(rho, n, alpha=0.05):
    """Compute statistical power for Spearman rho test."""
    from scipy.stats import norm
    z_rho = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    z_alt = z_rho / se
    power = norm.cdf(z_alt - z_crit) + norm.cdf(-z_alt - z_crit)
    return min(power, 1.0)


if __name__ == '__main__':
    print("=== batch_054 Phase F: D1 SCENIC Analysis ===\n")

    # Load correlations
    corr_path = f"{OUTDIR}/d1_correlations_all.csv"
    if not os.path.exists(corr_path):
        print(f"ERROR: {corr_path} not found. SCENIC may still be running.")
        sys.exit(1)

    corr_df = pd.read_csv(corr_path)
    print(f"Loaded {len(corr_df)} TF-compartment correlations")
    print(f"Datasets: {corr_df['dataset'].unique().tolist()}")
    print(f"TFs: {corr_df['tf'].unique().tolist()}")

    # Load summary
    summary_path = f"{OUTDIR}/d1_summary.json"
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    # === Apply decision rules ===
    print("\n=== Decision Rule Application ===\n")

    analysis_results = {}

    for tf in ['JUNB', 'KLF10']:
        expected_ds = {'JUNB': 'HLMA_Vascular', 'KLF10': 'HLMA_FAP'}[tf]
        sub = corr_df[(corr_df['tf'] == tf) & (corr_df['dataset'] == expected_ds)]

        if len(sub) == 0:
            print(f"{tf} in {expected_ds}: NOT FOUND in results")
            continue

        row = sub.iloc[0]
        rho = row['aucell_rho']
        p = row['aucell_p']
        n = row['n_donors']
        raw_rho = row.get('raw_mrna_rho', np.nan)
        delta = row.get('delta_rho', np.nan)
        regulon_source = row.get('regulon_source', 'unknown')

        power_50 = compute_power(0.50, n)
        power_60 = compute_power(0.60, n)
        power_actual = compute_power(max(rho, 0.01), n)

        verdict = apply_decision_rule(tf, expected_ds, rho, n)

        print(f"{tf} in {expected_ds}:")
        print(f"  AUCell rho = {rho:.3f} (p = {p:.2e}, N = {n})")
        print(f"  Raw mRNA rho = {raw_rho:.3f}")
        print(f"  Delta rho = {delta:+.3f} (AUCell - raw)")
        print(f"  Regulon source = {regulon_source}")
        print(f"  Power at rho=0.50: {power_50:.3f}")
        print(f"  Power at rho=0.60: {power_60:.3f}")
        print(f"  Decision: {verdict}")
        print()

        analysis_results[f"{tf}_{expected_ds}"] = {
            'tf': tf,
            'dataset': expected_ds,
            'aucell_rho': float(rho),
            'aucell_p': float(p),
            'raw_mrna_rho': float(raw_rho) if not np.isnan(raw_rho) else None,
            'delta_rho': float(delta) if not np.isnan(delta) else None,
            'n_donors': int(n),
            'regulon_source': regulon_source,
            'power_at_50': float(power_50),
            'power_at_60': float(power_60),
            'verdict': verdict,
        }

    # === Pattern matching ===
    print("\n=== Pattern Matching ===\n")

    # Check if top AUCell TFs match top raw mRNA TFs
    for ds in corr_df['dataset'].unique():
        sub = corr_df[corr_df['dataset'] == ds].dropna(subset=['aucell_rho', 'raw_mrna_rho'])
        if len(sub) < 5:
            continue

        top_aucell = set(sub.nlargest(5, 'aucell_rho')['tf'].tolist())
        top_raw = set(sub.nlargest(5, 'raw_mrna_rho')['tf'].tolist())
        overlap = top_aucell & top_raw
        print(f"{ds}: Top-5 AUCell overlap with Top-5 raw: {len(overlap)}/5 ({overlap})")

    # === JUNB compartment pattern ===
    print("\n=== JUNB Compartment Pattern ===")
    for ds in corr_df['dataset'].unique():
        sub = corr_df[(corr_df['tf'] == 'JUNB') & (corr_df['dataset'] == ds)]
        if len(sub) > 0:
            r = sub.iloc[0]
            print(f"  {ds}: AUCell rho={r['aucell_rho']:.3f}, raw rho={r['raw_mrna_rho']:.3f}")

    # === KLF10 compartment pattern ===
    print("\n=== KLF10 Compartment Pattern ===")
    for ds in corr_df['dataset'].unique():
        sub = corr_df[(corr_df['tf'] == 'KLF10') & (corr_df['dataset'] == ds)]
        if len(sub) > 0:
            r = sub.iloc[0]
            print(f"  {ds}: AUCell rho={r['aucell_rho']:.3f}, raw rho={r['raw_mrna_rho']:.3f}")

    # === Delta rho analysis ===
    print("\n=== Delta rho (AUCell - raw mRNA) ===")
    for tf in ['JUNB', 'JUN', 'FOS', 'FOSB', 'KLF10', 'ATF3', 'EGR1', 'IRF1', 'CDKN1A']:
        for ds in corr_df['dataset'].unique():
            sub = corr_df[(corr_df['tf'] == tf) & (corr_df['dataset'] == ds)]
            if len(sub) > 0:
                r = sub.iloc[0]
                delta = r.get('delta_rho', np.nan)
                if not np.isnan(delta):
                    print(f"  {tf:8s} {ds:20s}: delta={delta:+.3f}")

    # Save analysis
    analysis = {
        'batch': 'batch_054_d1',
        'analysis': 'Phase F decision rule application',
        'results': analysis_results,
    }

    with open(f"{OUTDIR}/d1_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nAnalysis saved to {OUTDIR}/d1_analysis.json")
