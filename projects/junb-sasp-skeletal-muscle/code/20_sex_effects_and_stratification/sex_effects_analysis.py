"""
Sex-Specific Effects Analysis: Vascular JUNB-SASP
Batch 029 - Investigating high-influence donor pattern (OM1, OM3, OM4, OM8)

Hypothesis: Old-male donors are driving the rho=0.93 correlation
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Define SASP12 genes (from literature)
SASP12_GENES = [
    'IL1B', 'IL6', 'IL8', 'CXCL1', 'CXCL2', 'CXCL3',  # Chemokines
    'MMP1', 'MMP3',  # Matrix remodeling
    'CSF2', 'CSF3',  # Colony stimulating factors
    'TNF', 'CCL2'  # Inflammatory cytokines
]

JUNB = 'JUNB'

print("=" * 70)
print("SEX-SPECIFIC EFFECTS: VASCULAR JUNB-SASP")
print("=" * 70)

# Load data
print("\n[1] Loading vascular data...")
adata = sc.read_h5ad('/home/yuanz/Documents/GitHub/biomarvin_fibro/data/Vascular_scsn_RNA.h5ad')
print(f"    Cells: {adata.shape[0]:,}")
print(f"    Genes: {adata.shape[1]:,}")
print(f"    Samples: {adata.obs['sample'].nunique()}")

# Create age-sex strata
print("\n[2] Creating age-sex strata...")

# Map sample to strata
def get_strata(row):
    sample = row['sample']
    sex = row['Sex']
    age = row['age']

    # Identify old (>= 60) vs young (< 60)
    if age >= 60:
        age_group = 'old'
    else:
        age_group = 'young'

    return f"{age_group}_{sex.lower()}"

adata.obs['strata'] = adata.obs.apply(get_strata, axis=1)

print("\n    Strata distribution:")
print(adata.obs.groupby('strata')['sample'].nunique())

# Step 3: Compute donor-level pseudobulk means
print("\n[3] Computing donor-level pseudobulk means...")

# Check which SASP12 genes are in the data
available_sasp = [g for g in SASP12_GENES if g in adata.var_names]
print(f"    SASP12 genes available: {len(available_sasp)}/{len(SASP12_GENES)}")
missing = [g for g in SASP12_GENES if g not in adata.var_names]
if missing:
    print(f"    Missing: {missing}")

# Compute SASP12 score per cell (using numpy for dense conversion if needed)
sasp_expr = adata[:, available_sasp].X.toarray() if hasattr(adata[:, available_sasp].X, 'toarray') else adata[:, available_sasp].X
adata.obs['SASP12'] = sasp_expr.mean(axis=1).A1 if hasattr(sasp_expr, 'A1') else np.array(sasp_expr).mean(axis=1)

# Compute JUNB per cell
junb_expr = adata[:, JUNB].X.toarray() if hasattr(adata[:, JUNB].X, 'toarray') else adata[:, JUNB].X
adata.obs['JUNB_expr'] = junb_expr.A1 if hasattr(junb_expr, 'A1') else np.array(junb_expr).flatten()

# Get strata per sample
sample_strata = adata.obs.groupby('sample')['strata'].first()

# Compute pseudobulk per donor
donor_data = []
for sample in adata.obs['sample'].unique():
    mask = adata.obs['sample'] == sample
    junb_mean = adata.obs.loc[mask, 'JUNB_expr'].mean()
    sasp_mean = adata.obs.loc[mask, 'SASP12'].mean()
    age = adata.obs.loc[mask, 'age'].iloc[0]
    sex = adata.obs.loc[mask, 'Sex'].iloc[0]
    strata = adata.obs.loc[mask, 'strata'].iloc[0]
    n_cells = mask.sum()

    donor_data.append({
        'sample': sample,
        'strata': strata,
        'sex': sex,
        'age': age,
        'n_cells': n_cells,
        'JUNB': junb_mean,
        'SASP12': sasp_mean
    })

donor_df = pd.DataFrame(donor_data)
print(f"\n    Donor-level data computed for {len(donor_df)} donors")
print(donor_df[['sample', 'strata', 'sex', 'age', 'n_cells', 'JUNB', 'SASP12']])

# Verify genes
if JUNB not in adata.var_names:
    print(f"    ERROR: {JUNB} not found in data!")
    print(f"    Available: {[g for g in adata.var_names if 'JUN' in g.upper()]}")

# Step 4: Correlation analysis within strata
print("\n[4] Correlation analysis within strata...")

strata_results = {}

for strata in sorted(donor_df['strata'].unique()):
    subset = donor_df[donor_df['strata'] == strata]
    n = len(subset)

    if n >= 3:  # Need at least 3 points for correlation
        rho, p = pearsonr(subset['JUNB'], subset['SASP12'])
        rho_s, p_s = spearmanr(subset['JUNB'], subset['SASP12'])
        mean_junb = subset['JUNB'].mean()
        mean_sasp = subset['SASP12'].mean()
        std_junb = subset['JUNB'].std()
        std_sasp = subset['SASP12'].std()
    else:
        rho, p, rho_s, p_s = np.nan, np.nan, np.nan, np.nan
        mean_junb = subset['JUNB'].mean() if n > 0 else np.nan
        mean_sasp = subset['SASP12'].mean() if n > 0 else np.nan
        std_junb = subset['JUNB'].std() if n > 1 else np.nan
        std_sasp = subset['SASP12'].std() if n > 1 else np.nan

    strata_results[strata] = {
        'n_donors': n,
        'n_cells': int(subset['n_cells'].sum()),
        'rho_pearson': float(rho) if not np.isnan(rho) else None,
        'p_pearson': float(p) if not np.isnan(p) else None,
        'rho_spearman': float(rho_s) if not np.isnan(rho_s) else None,
        'p_spearman': float(p_s) if not np.isnan(p_s) else None,
        'mean_JUNB': float(mean_junb) if not np.isnan(mean_junb) else None,
        'mean_SASP12': float(mean_sasp) if not np.isnan(mean_sasp) else None,
        'std_JUNB': float(std_junb) if not np.isnan(std_junb) else None,
        'std_SASP12': float(std_sasp) if not np.isnan(std_sasp) else None,
        'samples': list(subset['sample'])
    }

# Print strata summary
print("\n    Strata-level correlation summary:")
print("-" * 90)
print(f"{'Stratum':<15} {'N':<5} {'rho':<8} {'p':<10} {'mean_JUNB':<12} {'mean_SASP12':<12}")
print("-" * 90)
for strata, res in sorted(strata_results.items()):
    rho_str = f"{res['rho_pearson']:.4f}" if res['rho_pearson'] else "N/A"
    p_str = f"{res['p_pearson']:.4f}" if res['p_pearson'] else "N/A"
    mj = f"{res['mean_JUNB']:.4f}" if res['mean_JUNB'] else "N/A"
    ms = f"{res['mean_SASP12']:.4f}" if res['mean_SASP12'] else "N/A"
    print(f"{strata:<15} {res['n_donors']:<5} {rho_str:<8} {p_str:<10} {mj:<12} {ms:<12}")

# Step 5: Fisher Z comparison old-male vs old-female
print("\n[5] Fisher Z-transform correlation comparison...")

# Old-male vs Old-female
if 'old_male' in strata_results and 'old_female' in strata_results:
    om = strata_results['old_male']
    of = strata_results['old_female']

    if om['rho_pearson'] is not None and of['rho_pearson'] is not None:
        n1, r1 = om['n_donors'], om['rho_pearson']
        n2, r2 = of['n_donors'], of['rho_pearson']

        # Fisher Z transformation
        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
        z2 = 0.5 * np.log((1 + r2) / (1 - r2))
        se = np.sqrt(1/(n1-3) + 1/(n2-3))
        z_diff = (z1 - z2) / se
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

        print(f"\n    Old-Male: rho={r1:.4f} (n={n1})")
        print(f"    Old-Female: rho={r2:.4f} (n={n2})")
        print(f"    Fisher Z difference: {z_diff:.4f}")
        print(f"    p-value (two-tailed): {p_diff:.4f}")

        fisher_result = {
            'old_male_rho': r1,
            'old_male_n': n1,
            'old_female_rho': r2,
            'old_female_n': n2,
            'z_statistic': float(z_diff),
            'p_value': float(p_diff),
            'significant_at_0.05': bool(p_diff < 0.05)
        }
    else:
        print("    Cannot compute: missing correlation values")
        fisher_result = None
else:
    print("    Cannot compute: missing strata")
    fisher_result = None

# Step 6: T-tests old-male vs old-female
print("\n[6] T-tests: Old-Male vs Old-Female...")

om_df = donor_df[donor_df['strata'] == 'old_male']
of_df = donor_df[donor_df['strata'] == 'old_female']

# JUNB t-test
t_junb, p_junb = ttest_ind(om_df['JUNB'], of_df['JUNB'])
print(f"\n    JUNB: Old-Male mean={om_df['JUNB'].mean():.4f} vs Old-Female mean={of_df['JUNB'].mean():.4f}")
print(f"    t={t_junb:.4f}, p={p_junb:.4f}")

# SASP12 t-test
t_sasp, p_sasp = ttest_ind(om_df['SASP12'], of_df['SASP12'])
print(f"\n    SASP12: Old-Male mean={om_df['SASP12'].mean():.4f} vs Old-Female mean={of_df['SASP12'].mean():.4f}")
print(f"    t={t_sasp:.4f}, p={p_sasp:.4f}")

# Cohen's d for JUNB
pooled_std = np.sqrt(((len(om_df)-1)*om_df['JUNB'].std()**2 + (len(of_df)-1)*of_df['JUNB'].std()**2) / (len(om_df)+len(of_df)-2))
cohens_d_junb = (om_df['JUNB'].mean() - of_df['JUNB'].mean()) / pooled_std
print(f"\n    Cohen's d (JUNB): {cohens_d_junb:.4f}")

# Step 7: Old-female only correlation
print("\n[7] Correlation in Old-Female only (excluding high-influence donors)...")

if 'old_female' in strata_results and len(of_df) >= 3:
    rho_of, p_of = pearsonr(of_df['JUNB'], of_df['SASP12'])
    print(f"    Old-Female only: rho={rho_of:.4f}, p={p_of:.4f} (n={len(of_df)})")
    old_female_only = {'rho': float(rho_of), 'p': float(p_of), 'n': len(of_df)}
else:
    old_female_only = None
    print("    Insufficient data for old-female only correlation")

# Step 8: Create visualization
print("\n[8] Creating visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color and marker mapping
colors = {'Male': '#E63946', 'Female': '#457B9D'}
markers = {'old': 'D', 'young': 'o'}

# Left panel: JUNB vs SASP12 scatter by sex and age
ax = axes[0]

for strata in sorted(donor_df['strata'].unique()):
    subset = donor_df[donor_df['strata'] == strata]
    age_grp = 'old' if 'old' in strata else 'young'
    sex = 'Male' if 'male' in strata else 'Female'

    ax.scatter(subset['JUNB'], subset['SASP12'],
               c=colors[sex], marker=markers[age_grp],
               s=100, alpha=0.8, label=f"{strata} (n={len(subset)})",
               edgecolors='black', linewidth=0.5)

    # Add regression line if enough points
    if len(subset) >= 3:
        z = np.polyfit(subset['JUNB'], subset['SASP12'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset['JUNB'].min(), subset['JUNB'].max(), 100)
        ax.plot(x_line, p(x_line), c=colors[sex], linestyle='--', alpha=0.5, linewidth=2)

# Annotate high-influence donors
for idx, row in donor_df.iterrows():
    if row['sample'] in ['OM1', 'OM3', 'OM4', 'OM8']:
        ax.annotate(row['sample'], (row['JUNB'], row['SASP12']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

ax.set_xlabel('JUNB (pseudobulk mean)', fontsize=12)
ax.set_ylabel('SASP12 (pseudobulk mean)', fontsize=12)
ax.set_title('Vascular JUNB-SASP by Sex and Age', fontsize=14)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)

# Right panel: Bar chart comparing strata
ax2 = axes[1]

# Create comparison data
strata_order = ['old_male', 'old_female', 'young_male', 'young_female']
plot_data = []
for s in strata_order:
    if s in strata_results:
        r = strata_results[s]
        plot_data.append({
            'strata': s,
            'rho': r['rho_pearson'] if r['rho_pearson'] else 0,
            'n': r['n_donors'],
            'se': 1/np.sqrt(r['n_donors']-3) if r['n_donors'] > 3 else 0.5
        })

plot_df = pd.DataFrame(plot_data)

x_pos = np.arange(len(plot_df))
colors_bar = ['#E63946', '#457B9D', '#E63946', '#457B9D']
bars = ax2.bar(x_pos, plot_df['rho'], yerr=plot_df['se'], capsize=5,
               color=colors_bar, alpha=0.7, edgecolor='black')

ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.axhline(y=0.93, color='red', linestyle='--', alpha=0.5, label='Original rho=0.93')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"{r['strata']}\n(n={r['n']})" for _, r in plot_df.iterrows()], fontsize=10)
ax2.set_ylabel('Pearson rho (JUNB vs SASP12)', fontsize=12)
ax2.set_title('Correlation Strength by Stratum', fontsize=14)
ax2.set_ylim(-1, 1)
ax2.legend()

# Add significance markers
for i, (_, row) in enumerate(plot_df.iterrows()):
    if row['rho'] > 0.5:
        ax2.annotate('*', (i, row['rho'] + 0.1), ha='center', fontsize=16)

plt.tight_layout()
plt.savefig('/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_029/sex_effects_plot.pdf', dpi=150, bbox_inches='tight')
plt.savefig('/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_029/sex_effects_plot.png', dpi=150, bbox_inches='tight')
print("    Saved: sex_effects_plot.pdf, sex_effects_plot.png")

# Step 9: Compile results
print("\n[9] Compiling results...")

results = {
    'donor_data': donor_df.to_dict(orient='records'),
    'strata_results': strata_results,
    'old_male_vs_old_female': {
        'fisher_z': fisher_result,
        'ttest_JUNB': {'t': float(t_junb), 'p': float(p_junb), 'cohens_d': float(cohens_d_junb)},
        'ttest_SASP12': {'t': float(t_sasp), 'p': float(p_sasp)}
    },
    'old_female_only_correlation': old_female_only,
    'high_influence_donors': {
        'samples': ['OM1', 'OM3', 'OM4', 'OM8'],
        'all_old_male': True,
        'total_influence': '4/4 high-influence donors are old-male'
    },
    'summary': {
        'total_donors': len(donor_df),
        'old_male_donors': len(om_df),
        'old_female_donors': len(of_df),
        'young_male_donors': len(donor_df[donor_df['strata'] == 'young_male']),
        'young_female_donors': len(donor_df[donor_df['strata'] == 'young_female'])
    }
}

# Save results
output_path = '/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_029/sex_effects_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"    Saved: {output_path}")

# Print final summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nDonor breakdown:")
for strata in sorted(donor_df['strata'].unique()):
    n = (donor_df['strata'] == strata).sum()
    print(f"  {strata}: {n} donors")

print("\nCorrelation by stratum:")
for strata in sorted(strata_results.keys()):
    r = strata_results[strata]
    if r['rho_pearson'] is not None:
        sig = "sig" if r['p_pearson'] < 0.05 else "n.s."
        print(f"  {strata}: rho={r['rho_pearson']:.4f} (p={r['p_pearson']:.4f}, {sig})")
    else:
        print(f"  {strata}: insufficient data (n={r['n_donors']})")

print("\nKey finding:")
if fisher_result:
    print(f"  Old-Male rho: {fisher_result['old_male_rho']:.4f}")
    print(f"  Old-Female rho: {fisher_result['old_female_rho']:.4f}")
    print(f"  Fisher Z p-value: {fisher_result['p_value']:.4f}")
    if fisher_result['p_value'] < 0.05:
        print("  --> CORRELATIONS ARE SIGNIFICANTLY DIFFERENT between old-male and old-female")
    else:
        print("  --> Correlations are NOT significantly different")

if old_female_only:
    print(f"\n  Old-female only correlation: rho={old_female_only['rho']:.4f}")
    print(f"  --> Does old-female correlation survive without old-male donors?")

print("\nInterpretation:")
print("  The high-influence donors (OM1, OM3, OM4, OM8) are ALL OLD-MALE.")
print("  This raises the question: is the rho=0.93 driven by sex rather than age?")

plt.close()
print("\n[COMPLETE]")
