#!/usr/bin/env python3
"""
batch_073: Three PI-Directed External-Review Analyses (A1 + A2 + A3)

E1: Within-Country JUNB-Age Slopes (Vascular, MuSC)
E2: EGR2 Polarity Reversal Resolution (adjacency overlap)
E3: RUNX2+ FAP-Restricted JUNB-SASP (cell-level primary, donor-level secondary)

All analyses run on CPU using existing HLMA single-cell data.
Methodology follows batch_051 (pseudobulk donor-level means, SASP12 panel).

Sources:
  - PI directives A1/A2/A3 from research_state.md "FINAL PI DIRECTIVE"
  - batch_051/run_batch051.py for load_dataset, compute_donor_level, SASP12 panel
  - Design review revisions in batch_073/brief.md
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import anndata as ad
import scipy.sparse as sp
import json
import os
import sys
import platform
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

OUTDIR = "experiments/batch_073"
os.makedirs(OUTDIR, exist_ok=True)

# Canonical SASP12 panel — source: batch_050 q_final_fix.py, used in batch_051
SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8',
          'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

# China age groups for Mann-Whitney — source: brief.md E1 design review revision #1
CHINA_YOUNG_RANGE = (15, 34)   # YM1-YM5
CHINA_OLD_RANGE = (79, 84)     # OM1-OM9

# ============================================================
# ENVIRONMENT LOGGING (non-negotiable per ML research standards)
# ============================================================

def log_environment():
    """Log environment for reproducibility."""
    env = {
        'timestamp': datetime.now().isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scipy': __import__('scipy').__version__,
        'statsmodels': sm.__version__,
        'anndata': ad.__version__,
    }
    print("=== Environment ===")
    for k, v in env.items():
        print(f"  {k}: {v}")
    return env

env_info = log_environment()

# ============================================================
# HELPER FUNCTIONS (adapted from batch_051)
# ============================================================

def load_dataset(path, cell_type_filter=None, filter_col='Annotation'):
    """Load h5ad with backed='r' for memory efficiency. Optionally filter by
    cell type. Returns obs DataFrame, dense expression matrix, and var_names list.

    WHY backed='r': avoids loading full matrix into RAM; we slice what we need.
    WHY toarray(): downstream ops (mean, indexing) require dense arrays.
    """
    print(f"  Loading {path}...")
    adata = ad.read_h5ad(path, backed='r')
    obs = adata.obs.copy()
    var_names = list(adata.var_names)

    if cell_type_filter:
        mask = obs[filter_col].isin(cell_type_filter)
        obs = obs[mask]
    else:
        mask = pd.Series(True, index=obs.index)

    X = adata[mask.values, :].X[:, :]
    if sp.issparse(X):
        X = X.toarray()

    adata.file.close()
    print(f"    Loaded {X.shape[0]} cells x {X.shape[1]} genes")
    return obs, X, var_names


def compute_donor_level(obs, X, var_names, genes):
    """Compute donor-level pseudobulk mean expression for specified genes.

    WHY pseudobulk mean: standard approach for donor-level analysis from scRNA-seq.
    Each donor contributes one observation (mean across all cells from that donor).
    Source: batch_051 methodology.
    """
    detected = [g for g in genes if g in var_names]
    detected_sasp = [g for g in SASP12 if g in var_names]
    missing_sasp = [g for g in SASP12 if g not in var_names]
    if missing_sasp:
        print(f"    WARNING: SASP12 genes not found in var_names: {missing_sasp}")

    # Identify sample column
    sample_col = 'sample' if 'sample' in obs.columns else 'donor_id'
    if sample_col not in obs.columns:
        sample_col = 'SampleID'

    donor_data = []
    for donor in obs[sample_col].unique():
        dmask = obs[sample_col] == donor
        d_obs = obs[dmask]
        d_X = X[dmask.values]
        n_cells = dmask.sum()

        row = {'sample': donor, 'n_cells': n_cells}

        # Copy metadata from first cell of this donor
        for col in ['age', 'age_pop', 'Sex', 'gender', 'Country', 'tech']:
            if col in d_obs.columns:
                row[col] = d_obs[col].iloc[0]

        # Gene means
        for g in detected:
            idx = var_names.index(g)
            row[g] = float(np.mean(d_X[:, idx]))

        # SASP12 composite
        row['SASP12_detected'] = len(detected_sasp)
        if detected_sasp:
            row['SASP12_mean'] = float(np.mean([row[g] for g in detected_sasp]))
        else:
            row['SASP12_mean'] = np.nan
        row['SASP12_genes'] = ','.join(detected_sasp)

        donor_data.append(row)

    df = pd.DataFrame(donor_data)
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    return df


def ols_regression(y, x, label=""):
    """Run OLS regression y ~ x. Return dict with beta, CI, p, R2.

    WHY OLS with statsmodels: provides proper confidence intervals and
    inference for linear regression. Source: standard statistical methodology.
    """
    valid = np.isfinite(y) & np.isfinite(x)
    y_v = y[valid]
    x_v = x[valid]
    n = len(y_v)
    if n < 4:
        return {
            'test': 'OLS', 'estimate': np.nan, 'CI_lower': np.nan,
            'CI_upper': np.nan, 'p_value': np.nan, 'R2': np.nan,
            'N': n, 'note': f'Too few observations (N={n})'
        }
    X_design = sm.add_constant(x_v)
    model = sm.OLS(y_v, X_design).fit()
    beta = model.params[1]
    ci = model.conf_int(alpha=0.05)
    ci_lower = ci[1][0]
    ci_upper = ci[1][1]
    p_value = model.pvalues[1]
    r2 = model.rsquared

    result = {
        'test': 'OLS', 'estimate': float(beta),
        'CI_lower': float(ci_lower), 'CI_upper': float(ci_upper),
        'p_value': float(p_value), 'R2': float(r2), 'N': n,
    }
    return result


def spearman_test(y, x):
    """Compute Spearman rho with p-value. Return dict."""
    valid = np.isfinite(y) & np.isfinite(x)
    y_v = y[valid]
    x_v = x[valid]
    n = len(y_v)
    if n < 4:
        return {
            'test': 'Spearman', 'estimate': np.nan, 'CI_lower': np.nan,
            'CI_upper': np.nan, 'p_value': np.nan, 'R2': np.nan,
            'N': n, 'note': f'Too few observations (N={n})'
        }
    rho, p = stats.spearmanr(y_v, x_v)
    # Fisher Z CI for Spearman rho
    ci_lo, ci_hi = fisher_z_ci(rho, n)
    return {
        'test': 'Spearman', 'estimate': float(rho),
        'CI_lower': float(ci_lo), 'CI_upper': float(ci_hi),
        'p_value': float(p), 'R2': np.nan, 'N': n,
    }


def fisher_z_ci(rho, n, alpha=0.05):
    """95% CI for correlation via Fisher Z transform.
    Source: standard statistical methodology (Fisher 1921).
    """
    if n < 4 or np.isnan(rho):
        return np.nan, np.nan
    from scipy.stats import norm
    z_rho = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    z_lo = z_rho - z_crit * se
    z_hi = z_rho + z_crit * se
    ci_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
    ci_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
    return float(ci_lo), float(ci_hi)


def mann_whitney_test(group1, group2, label=""):
    """Mann-Whitney U test with rank-biserial effect size.

    WHY rank-biserial r = 1 - 2U/(n1*n2): standard nonparametric effect size
    for Mann-Whitney. Source: Kerby (2014), "Simple differences."
    """
    n1 = len(group1)
    n2 = len(group2)
    if n1 < 2 or n2 < 2:
        return {
            'test': 'MannWhitney', 'estimate': np.nan, 'CI_lower': np.nan,
            'CI_upper': np.nan, 'p_value': np.nan, 'R2': np.nan,
            'effect_size': np.nan, 'N': n1 + n2,
            'note': f'Too few observations (n1={n1}, n2={n2})'
        }
    U, p = mannwhitneyu(group1, group2, alternative='two-sided')
    # Rank-biserial correlation: r = 1 - 2U/(n1*n2)
    r_rb = 1.0 - (2.0 * U) / (n1 * n2)
    return {
        'test': 'MannWhitney', 'estimate': float(U),
        'CI_lower': np.nan, 'CI_upper': np.nan,
        'p_value': float(p), 'R2': np.nan,
        'effect_size': float(r_rb),
        'N': n1 + n2,
        'n_young': n1, 'n_old': n2,
    }


def cooks_distance(y, x):
    """Compute Cook's D for OLS y ~ x. Return array of Cook's D per observation.

    WHY Cook's D: identifies influential observations that disproportionately
    affect regression slope. Critical for E1 China where bimodal age distribution
    may create leverage effects. Source: design review revision #1.
    """
    valid = np.isfinite(y) & np.isfinite(x)
    y_v = np.array(y[valid], dtype=float)
    x_v = np.array(x[valid], dtype=float)
    n = len(y_v)
    if n < 4:
        return np.full(n, np.nan)
    X_design = sm.add_constant(x_v)
    model = sm.OLS(y_v, X_design).fit()
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    return cooks_d


# ============================================================
# E1: Within-Country JUNB-Age Slopes
# ============================================================

print("\n" + "=" * 70)
print("E1: Within-Country JUNB-Age Slopes")
print("=" * 70)
print("Source: PI directive A1 — compute within-country JUNB~age and SASP12~age")
print("WHY: Pooled rho(JUNB,age)=0.042 (p=0.87) is null. PI hypothesizes")
print("  within-country slopes are positive but cross-country composition attenuates.")

e1_rows = []

datasets_e1 = {
    'Vascular': {
        'path': 'data/Vascular_scsn_RNA.h5ad',
        'filter': None,  # All cell types in the file
        'filter_col': 'Annotation',
    },
    'MuSC': {
        'path': 'data/MuSC_scsn_RNA.h5ad',
        'filter': None,
        'filter_col': 'Annotation',
    },
}

for compartment, ds_info in datasets_e1.items():
    print(f"\n--- {compartment} ---")
    obs, X, var_names = load_dataset(
        ds_info['path'],
        cell_type_filter=ds_info['filter'],
        filter_col=ds_info['filter_col'],
    )

    # Compute donor-level JUNB + SASP12
    genes_needed = ['JUNB'] + SASP12
    donors = compute_donor_level(obs, X, var_names, genes_needed)

    # Standardize column names
    if 'Country' in donors.columns:
        donors['country'] = donors['Country']
    if 'Sex' in donors.columns:
        donors['sex'] = donors['Sex']
    if 'gender' in donors.columns:
        donors['sex'] = donors['gender']

    print(f"  Total donors: {len(donors)}")
    print(f"  Country distribution: {donors['country'].value_counts().to_dict()}")
    print(f"  Age range: {donors['age'].min():.0f} - {donors['age'].max():.0f}")
    print(f"  SASP12 detected: {donors['SASP12_detected'].iloc[0]} / 12")

    # Check JUNB is present
    if 'JUNB' not in donors.columns or donors['JUNB'].isna().all():
        print(f"  ERROR: JUNB not found or all NaN for {compartment}. Skipping.")
        continue

    # ---- Pooled analysis (all 23 donors) ----
    print(f"\n  POOLED (N={len(donors)}):")
    for metric in ['JUNB', 'SASP12_mean']:
        metric_label = 'JUNB' if metric == 'JUNB' else 'SASP12'
        y = donors[metric].values.astype(float)
        x = donors['age'].values.astype(float)

        # OLS
        ols_res = ols_regression(y, x)
        ols_row = {
            'compartment': compartment, 'country': 'Pooled',
            'N': len(donors),
            'age_range': f"{donors['age'].min():.0f}-{donors['age'].max():.0f}",
            'metric': metric_label,
            **ols_res,
        }
        e1_rows.append(ols_row)
        print(f"    OLS {metric_label}~age: beta={ols_res['estimate']:.6f}, "
              f"CI=[{ols_res['CI_lower']:.6f}, {ols_res['CI_upper']:.6f}], "
              f"p={ols_res['p_value']:.4f}, R2={ols_res['R2']:.4f}")

        # Spearman
        sp_res = spearman_test(y, x)
        sp_row = {
            'compartment': compartment, 'country': 'Pooled',
            'N': len(donors),
            'age_range': f"{donors['age'].min():.0f}-{donors['age'].max():.0f}",
            'metric': metric_label,
            **sp_res,
        }
        e1_rows.append(sp_row)
        print(f"    Spearman {metric_label}~age: rho={sp_res['estimate']:.4f}, "
              f"CI=[{sp_res['CI_lower']:.4f}, {sp_res['CI_upper']:.4f}], "
              f"p={sp_res['p_value']:.4f}")

    # ---- Within-country analyses ----
    for country in sorted(donors['country'].unique()):
        sub = donors[donors['country'] == country].copy()
        n_country = len(sub)
        age_range_str = f"{sub['age'].min():.0f}-{sub['age'].max():.0f}"
        print(f"\n  WITHIN-{country.upper()} (N={n_country}, ages {age_range_str}):")

        for metric in ['JUNB', 'SASP12_mean']:
            metric_label = 'JUNB' if metric == 'JUNB' else 'SASP12'
            y = sub[metric].values.astype(float)
            x = sub['age'].values.astype(float)

            # OLS
            ols_res = ols_regression(y, x)
            ols_row = {
                'compartment': compartment, 'country': country,
                'N': n_country,
                'age_range': age_range_str,
                'metric': metric_label,
                **ols_res,
            }
            e1_rows.append(ols_row)
            print(f"    OLS {metric_label}~age: beta={ols_res['estimate']:.6f}, "
                  f"CI=[{ols_res['CI_lower']:.6f}, {ols_res['CI_upper']:.6f}], "
                  f"p={ols_res['p_value']:.4f}, R2={ols_res['R2']:.4f}")

            # Spearman
            sp_res = spearman_test(y, x)
            sp_row = {
                'compartment': compartment, 'country': country,
                'N': n_country,
                'age_range': age_range_str,
                'metric': metric_label,
                **sp_res,
            }
            e1_rows.append(sp_row)
            print(f"    Spearman {metric_label}~age: rho={sp_res['estimate']:.4f}, "
                  f"CI=[{sp_res['CI_lower']:.4f}, {sp_res['CI_upper']:.4f}], "
                  f"p={sp_res['p_value']:.4f}")

            # Cook's D leverage diagnostics for OLS
            cooks_d_vals = cooks_distance(y, x)
            if not np.all(np.isnan(cooks_d_vals)):
                max_cook = np.nanmax(cooks_d_vals)
                max_cook_idx = np.nanargmax(cooks_d_vals)
                # Standard threshold: 4/n
                threshold = 4.0 / n_country
                n_influential = np.sum(cooks_d_vals > threshold)
                print(f"    Cook's D: max={max_cook:.4f} (donor {sub['sample'].iloc[max_cook_idx]}, "
                      f"age={sub['age'].iloc[max_cook_idx]:.0f}), "
                      f"threshold(4/N)={threshold:.4f}, "
                      f"n_influential={n_influential}")

        # ---- China-specific: Mann-Whitney U (young vs old) ----
        # WHY: Design review revision #1 — China has bimodal ages (5 young
        # [15-34] + 9 old [79-84]). OLS slope is mechanically a two-group
        # contrast, so report Mann-Whitney alongside.
        if country == 'China':
            young_mask = (sub['age'] >= CHINA_YOUNG_RANGE[0]) & (sub['age'] <= CHINA_YOUNG_RANGE[1])
            old_mask = (sub['age'] >= CHINA_OLD_RANGE[0]) & (sub['age'] <= CHINA_OLD_RANGE[1])
            young = sub[young_mask]
            old = sub[old_mask]
            print(f"\n    Mann-Whitney U (young [{CHINA_YOUNG_RANGE}] vs old [{CHINA_OLD_RANGE}]):")
            print(f"      N_young={len(young)}, N_old={len(old)}")

            for metric in ['JUNB', 'SASP12_mean']:
                metric_label = 'JUNB' if metric == 'JUNB' else 'SASP12'
                if len(young) >= 2 and len(old) >= 2:
                    g1 = young[metric].values.astype(float)
                    g2 = old[metric].values.astype(float)
                    mwu_res = mann_whitney_test(g1, g2)
                    mwu_row = {
                        'compartment': compartment, 'country': f'{country}_MWU',
                        'N': mwu_res['N'],
                        'age_range': f"young({CHINA_YOUNG_RANGE[0]}-{CHINA_YOUNG_RANGE[1]})_vs_old({CHINA_OLD_RANGE[0]}-{CHINA_OLD_RANGE[1]})",
                        'metric': metric_label,
                        **mwu_res,
                    }
                    e1_rows.append(mwu_row)
                    print(f"      {metric_label}: U={mwu_res['estimate']:.1f}, "
                          f"p={mwu_res['p_value']:.4f}, "
                          f"rank-biserial r={mwu_res['effect_size']:.4f}")
                    print(f"        young mean={np.mean(g1):.4f}, old mean={np.mean(g2):.4f}")
                else:
                    print(f"      {metric_label}: SKIPPED — too few observations in one group")

# Save E1 results
e1_df = pd.DataFrame(e1_rows)
e1_path = f"{OUTDIR}/e1_junb_age_slopes.csv"
e1_df.to_csv(e1_path, index=False)
print(f"\nE1 results saved to {e1_path} ({len(e1_df)} rows)")


# ============================================================
# E2: EGR2 Polarity Reversal Resolution
# ============================================================

print("\n" + "=" * 70)
print("E2: EGR2 Polarity Reversal Resolution")
print("=" * 70)
print("Source: PI directive A2 — resolve EGR2 AUCell polarity reversal")
print("IMPORTANT: These are co-expression modules (GRNBoost2), NOT motif-pruned")
print("  regulatory targets. cisTarget failed in batch_054 (pySCENIC 0.12.x bug).")
print("  Interpretation capped at SUGGESTED per design review revision #2.")

# Load adjacency files
adj_files = {
    'Vascular': 'experiments/batch_054/d1_adjacencies_HLMA_Vascular.csv',
    'MuSC': 'experiments/batch_054/d1_adjacencies_HLMA_MuSC.csv',
    'FAP': 'experiments/batch_055/d1_adjacencies_HLMA_FAP.csv',
}

egr2_targets = {}  # compartment -> DataFrame of EGR2 targets
target_lists = {}  # compartment -> {threshold -> set of target genes}

for compartment, path in adj_files.items():
    print(f"\n--- {compartment} ---")
    adj = pd.read_csv(path)
    egr2 = adj[adj['TF'] == 'EGR2'].copy()
    print(f"  Total EGR2-as-TF target genes: {len(egr2)}")

    if len(egr2) == 0:
        print(f"  WARNING: No EGR2 targets found in {path}")
        egr2_targets[compartment] = pd.DataFrame()
        target_lists[compartment] = {'median': set(), 'p80': set()}
        continue

    egr2 = egr2.sort_values('importance', ascending=False).reset_index(drop=True)
    median_imp = egr2['importance'].median()
    p80_imp = egr2['importance'].quantile(0.80)
    print(f"  Importance: median={median_imp:.4f}, p80={p80_imp:.4f}, "
          f"max={egr2['importance'].max():.4f}")

    # Apply thresholds
    # WHY two thresholds: design review revision #4 — sensitivity analysis to
    # ensure Jaccard result is not an artifact of threshold choice.
    targets_median = set(egr2[egr2['importance'] >= median_imp]['target'].values)
    targets_p80 = set(egr2[egr2['importance'] >= p80_imp]['target'].values)
    target_lists[compartment] = {'median': targets_median, 'p80': targets_p80}
    egr2_targets[compartment] = egr2

    print(f"  Targets at median threshold: {len(targets_median)}")
    print(f"  Targets at p80 threshold: {len(targets_p80)}")

    # SASP12 membership
    for thresh_name, tset in [('median', targets_median), ('p80', targets_p80)]:
        sasp_in = [g for g in SASP12 if g in tset]
        print(f"  SASP12 in {thresh_name} targets: {sasp_in if sasp_in else 'NONE'}")

    # Top-10 targets
    top10 = egr2.head(10)
    print(f"  Top-10 targets by importance:")
    for _, r in top10.iterrows():
        sasp_flag = " [SASP12]" if r['target'] in SASP12 else ""
        print(f"    {r['target']:15s}  importance={r['importance']:.4f}{sasp_flag}")

# Compute Jaccard overlaps
print("\n--- Jaccard Overlaps ---")

e2_results = {
    'caveat': 'Co-expression modules only (GRNBoost2). NOT motif-pruned regulatory targets. cisTarget failed in batch_054.',
    'compartments': {},
    'jaccard_comparisons': [],
}

# Store per-compartment info
for compartment in ['Vascular', 'MuSC', 'FAP']:
    comp_info = {}
    for thresh_name in ['median', 'p80']:
        tset = target_lists.get(compartment, {}).get(thresh_name, set())
        sasp_in = [g for g in SASP12 if g in tset]
        comp_info[thresh_name] = {
            'n_targets': len(tset),
            'sasp12_in_targets': sasp_in,
            'n_sasp12_in_targets': len(sasp_in),
        }
    e2_results['compartments'][compartment] = comp_info

    # Top-10
    egr2_df = egr2_targets.get(compartment, pd.DataFrame())
    if len(egr2_df) > 0:
        top10 = egr2_df.head(10)[['target', 'importance']].to_dict('records')
        e2_results['compartments'][compartment]['top10'] = top10

# Pairwise Jaccard
pairs = [('Vascular', 'MuSC'), ('FAP', 'Vascular'), ('FAP', 'MuSC')]
for c1, c2 in pairs:
    for thresh_name in ['median', 'p80']:
        s1 = target_lists.get(c1, {}).get(thresh_name, set())
        s2 = target_lists.get(c2, {}).get(thresh_name, set())

        if len(s1) == 0 or len(s2) == 0:
            jaccard = np.nan
            intersection_size = 0
            union_size = len(s1 | s2)
        else:
            intersection = s1 & s2
            union = s1 | s2
            intersection_size = len(intersection)
            union_size = len(union)
            jaccard = intersection_size / union_size if union_size > 0 else 0.0

        comparison = {
            'pair': f'{c1}_vs_{c2}',
            'threshold': thresh_name,
            'n_targets_1': len(s1),
            'n_targets_2': len(s2),
            'intersection_size': intersection_size,
            'union_size': union_size,
            'jaccard': float(jaccard) if not np.isnan(jaccard) else None,
        }

        # Also report which specific genes overlap (for Vascular vs MuSC)
        if c1 in ['Vascular', 'MuSC'] and c2 in ['Vascular', 'MuSC']:
            if intersection_size > 0:
                comparison['intersection_genes'] = sorted(list(s1 & s2))
            # SASP12 presence comparison
            sasp_in_1 = [g for g in SASP12 if g in s1]
            sasp_in_2 = [g for g in SASP12 if g in s2]
            comparison['sasp12_in_c1'] = sasp_in_1
            comparison['sasp12_in_c2'] = sasp_in_2
            comparison['sasp12_unique_to_c1'] = [g for g in sasp_in_1 if g not in sasp_in_2]
            comparison['sasp12_unique_to_c2'] = [g for g in sasp_in_2 if g not in sasp_in_1]

        e2_results['jaccard_comparisons'].append(comparison)

        label = f"  {c1} vs {c2} ({thresh_name}): Jaccard={jaccard:.4f}" if not np.isnan(jaccard) else f"  {c1} vs {c2} ({thresh_name}): Jaccard=N/A"
        print(f"{label}, |A intersect B|={intersection_size}, |A union B|={union_size}")

# Build target gene CSV
# WHY: provides a detailed view of which targets are in which compartment
e2_target_rows = []
all_targets = set()
for compartment in ['Vascular', 'MuSC', 'FAP']:
    egr2_df = egr2_targets.get(compartment, pd.DataFrame())
    if len(egr2_df) > 0:
        for _, r in egr2_df.iterrows():
            all_targets.add(r['target'])

for target in sorted(all_targets):
    row = {'target': target}
    for compartment in ['Vascular', 'MuSC', 'FAP']:
        egr2_df = egr2_targets.get(compartment, pd.DataFrame())
        if len(egr2_df) > 0 and target in egr2_df['target'].values:
            imp = egr2_df[egr2_df['target'] == target]['importance'].values[0]
            row[f'importance_{compartment}'] = float(imp)
            row[f'in_{compartment.lower()}'] = True
        else:
            row[f'importance_{compartment}'] = np.nan
            row[f'in_{compartment.lower()}'] = False
    row['in_sasp12'] = target in SASP12
    e2_target_rows.append(row)

e2_targets_df = pd.DataFrame(e2_target_rows)
e2_targets_path = f"{OUTDIR}/e2_egr2_targets.csv"
e2_targets_df.to_csv(e2_targets_path, index=False)
print(f"\nE2 target gene list saved to {e2_targets_path} ({len(e2_targets_df)} genes)")

# Save JSON results
e2_json_path = f"{OUTDIR}/e2_egr2_polarity.json"
with open(e2_json_path, 'w') as f:
    json.dump(e2_results, f, indent=2, default=str)
print(f"E2 structured results saved to {e2_json_path}")


# ============================================================
# E3: RUNX2+ FAP-Restricted JUNB-SASP
# ============================================================

print("\n" + "=" * 70)
print("E3: RUNX2+ FAP-Restricted JUNB-SASP")
print("=" * 70)
print("Source: PI directive A3 — RUNX2+ FAP-restricted donor-level JUNB-SASP")
print("PRIMARY: Cell-level Spearman rho(JUNB, SASP12_mean) by FAP subtype")
print("  WHY primary: design review revision #5 — cell-level is feasible and")
print("  informative, while donor-level is pre-declared INCONCLUSIVE (N~3).")
print("SECONDARY: Donor-level (pre-declared INCONCLUSIVE due to RUNX2+ sparsity)")

# Load FAP data
fap_obs, fap_X, fap_var_names = load_dataset('data/OMIX004308-02.h5ad')

# Check gene presence
junb_idx = fap_var_names.index('JUNB') if 'JUNB' in fap_var_names else None
if junb_idx is None:
    print("  FATAL: JUNB not found in FAP var_names. Cannot proceed.")
    sys.exit(1)

sasp_indices = []
sasp_detected = []
for g in SASP12:
    if g in fap_var_names:
        sasp_indices.append(fap_var_names.index(g))
        sasp_detected.append(g)
    else:
        print(f"  WARNING: SASP12 gene {g} not in FAP var_names")

print(f"  SASP12 detected: {len(sasp_detected)}/{len(SASP12)} — {sasp_detected}")

# Define FAP subtypes (from brief.md)
fap_subtypes = ['MME+ FAP', 'CD55+ FAP', 'Tenocyte', 'GPC3+ FAP',
                'CD99+ FAP', 'RUNX2+ FAP']

# ---- PRIMARY: Cell-level Spearman rho by subtype ----
print("\n--- PRIMARY: Cell-level JUNB-SASP12 Spearman by FAP subtype ---")
# WHY cell-level: Each cell is an observation. Spearman tests whether cells
# with higher JUNB tend to have higher SASP12 expression. This is the PI's
# actual question ("is JUNB-SASP coupling stronger in RUNX2+ FAPs?").

e3_cell_rows = []

for subtype in fap_subtypes:
    mask = fap_obs['Annotation'] == subtype
    n_cells = mask.sum()
    if n_cells < 10:
        print(f"  {subtype}: N={n_cells} — too few cells, skipping")
        e3_cell_rows.append({
            'subtype': subtype, 'N_cells': n_cells,
            'rho': np.nan, 'p_value': np.nan,
            'note': 'Too few cells'
        })
        continue

    sub_X = fap_X[mask.values]
    junb_vals = sub_X[:, junb_idx].astype(float)

    # Compute per-cell SASP12_mean
    if len(sasp_indices) > 0:
        sasp_matrix = sub_X[:, sasp_indices].astype(float)
        sasp_mean_vals = np.mean(sasp_matrix, axis=1)
    else:
        sasp_mean_vals = np.full(n_cells, np.nan)

    # Spearman rho(JUNB, SASP12_mean) at cell level
    rho, p = stats.spearmanr(junb_vals, sasp_mean_vals)

    e3_cell_rows.append({
        'subtype': subtype,
        'N_cells': n_cells,
        'rho': float(rho),
        'p_value': float(p),
        'junb_mean': float(np.mean(junb_vals)),
        'junb_nonzero_frac': float(np.mean(junb_vals > 0)),
        'sasp12_mean': float(np.mean(sasp_mean_vals)),
    })
    print(f"  {subtype:15s}: N={n_cells:6d}, rho={rho:.4f}, p={p:.2e}, "
          f"JUNB_mean={np.mean(junb_vals):.4f}, SASP12_mean={np.mean(sasp_mean_vals):.4f}")

e3_cell_df = pd.DataFrame(e3_cell_rows)
e3_cell_path = f"{OUTDIR}/e3_runx2_fap_cell_level.csv"
e3_cell_df.to_csv(e3_cell_path, index=False)
print(f"\nE3 cell-level results saved to {e3_cell_path}")

# Compare RUNX2+ to others
runx2_rho = e3_cell_df[e3_cell_df['subtype'] == 'RUNX2+ FAP']['rho'].values
if len(runx2_rho) > 0 and not np.isnan(runx2_rho[0]):
    print(f"\n  RUNX2+ FAP cell-level rho = {runx2_rho[0]:.4f}")
    other_rhos = e3_cell_df[
        (e3_cell_df['subtype'] != 'RUNX2+ FAP') &
        (e3_cell_df['rho'].notna())
    ]
    print(f"  Other subtypes:")
    for _, r in other_rhos.iterrows():
        delta = runx2_rho[0] - r['rho']
        print(f"    {r['subtype']:15s}: rho={r['rho']:.4f} (delta from RUNX2+: {delta:+.4f})")

# ---- SECONDARY: Donor-level analysis (pre-declared INCONCLUSIVE) ----
print("\n--- SECONDARY: Donor-level RUNX2+ FAP JUNB-SASP (PRE-DECLARED INCONCLUSIVE) ---")

# Filter to RUNX2+ FAP only
runx2_mask = fap_obs['Annotation'] == 'RUNX2+ FAP'
runx2_obs = fap_obs[runx2_mask].copy()
runx2_X = fap_X[runx2_mask.values]

# Cell count distribution (all donors)
sample_col = 'sample'
e3_count_rows = []
for donor in fap_obs[sample_col].unique():
    d_obs = fap_obs[fap_obs[sample_col] == donor]
    d_runx2 = runx2_obs[runx2_obs[sample_col] == donor]
    n_total_fap = len(d_obs)
    n_runx2 = len(d_runx2)
    age_val = d_obs['age'].iloc[0] if 'age' in d_obs.columns else np.nan
    country_val = d_obs['Country'].iloc[0] if 'Country' in d_obs.columns else ''
    sex_val = d_obs['Sex'].iloc[0] if 'Sex' in d_obs.columns else ''

    e3_count_rows.append({
        'sample': donor,
        'age': age_val,
        'country': country_val,
        'sex': sex_val,
        'n_total_fap_cells': n_total_fap,
        'n_runx2_cells': n_runx2,
        'runx2_fraction': n_runx2 / n_total_fap if n_total_fap > 0 else 0,
    })

e3_counts_df = pd.DataFrame(e3_count_rows).sort_values('n_runx2_cells', ascending=False)
e3_counts_path = f"{OUTDIR}/e3_runx2_fap_cell_counts.csv"
e3_counts_df.to_csv(e3_counts_path, index=False)
print(f"\n  Cell count distribution saved to {e3_counts_path}")
print(f"  RUNX2+ FAP cell counts per donor:")
for _, r in e3_counts_df.iterrows():
    flag = ""
    if r['n_runx2_cells'] >= 20:
        flag = " [>=20]"
    elif r['n_runx2_cells'] >= 10:
        flag = " [>=10]"
    age_display = r['age']
    try:
        age_display = f"{float(r['age']):3.0f}"
    except (ValueError, TypeError):
        age_display = str(r['age'])
    print(f"    {r['sample']:8s}: age={age_display}, country={str(r['country']):6s}, "
          f"RUNX2+={r['n_runx2_cells']:4d} / {r['n_total_fap_cells']:5d}{flag}")

# Donor-level at thresholds
e3_donor_rows = []
for min_cells in [10, 20]:
    qualifying = e3_counts_df[e3_counts_df['n_runx2_cells'] >= min_cells]
    n_qualifying = len(qualifying)
    qualifying_ids = list(qualifying['sample'].values)
    print(f"\n  Threshold >= {min_cells} cells: {n_qualifying} qualifying donors: {qualifying_ids}")

    if n_qualifying < 3:
        print(f"    DEGENERATE: N={n_qualifying} < 3, cannot compute meaningful rho")
        e3_donor_rows.append({
            'threshold': min_cells,
            'effective_N': n_qualifying,
            'donor_ids': ','.join(qualifying_ids),
            'rho': np.nan,
            'p_value': np.nan,
            'note': f'Degenerate: N={n_qualifying} < 3',
        })
        continue

    # Compute donor-level for qualifying donors
    qual_mask = runx2_obs[sample_col].isin(qualifying_ids)
    qual_obs = runx2_obs[qual_mask].copy()
    qual_X = runx2_X[qual_mask.values]

    genes_needed = ['JUNB'] + SASP12
    qual_donors = compute_donor_level(qual_obs, qual_X, fap_var_names, genes_needed)

    # Add metadata from counts table
    qual_donors = qual_donors.merge(
        e3_counts_df[['sample', 'age', 'country', 'sex', 'n_runx2_cells']],
        on='sample', how='left', suffixes=('_expr', '_meta')
    )
    # Use age from metadata if available
    if 'age_meta' in qual_donors.columns:
        qual_donors['age'] = pd.to_numeric(qual_donors['age_meta'], errors='coerce')

    print(f"    Donor-level RUNX2+ FAP (threshold={min_cells}):")
    for _, r in qual_donors.iterrows():
        print(f"      {r['sample']}: age={r.get('age', 'N/A')}, "
              f"country={r.get('country', 'N/A')}, "
              f"n_cells={r['n_cells']}, JUNB={r.get('JUNB', np.nan):.4f}, "
              f"SASP12={r.get('SASP12_mean', np.nan):.4f}")

    if n_qualifying < 5:
        # Still compute rho but flag as degenerate
        if 'JUNB' in qual_donors.columns and 'SASP12_mean' in qual_donors.columns:
            junb_vals = qual_donors['JUNB'].values.astype(float)
            sasp_vals = qual_donors['SASP12_mean'].values.astype(float)
            valid = np.isfinite(junb_vals) & np.isfinite(sasp_vals)
            if valid.sum() >= 3:
                rho, p = stats.spearmanr(junb_vals[valid], sasp_vals[valid])
                print(f"    Spearman rho(JUNB, SASP12): {rho:.4f} (p={p:.4f}) "
                      f"-- DEGENERATE (N={valid.sum()}<5)")
                e3_donor_rows.append({
                    'threshold': min_cells,
                    'effective_N': int(valid.sum()),
                    'donor_ids': ','.join(qualifying_ids),
                    'ages': ','.join([str(int(a)) for a in qual_donors['age'].dropna()]),
                    'countries': ','.join([str(c) for c in qual_donors.get('country', pd.Series()).dropna()]),
                    'cell_counts': ','.join([str(int(c)) for c in qual_donors['n_cells']]),
                    'rho': float(rho),
                    'p_value': float(p),
                    'note': f'Degenerate: N={valid.sum()}<5. Pre-declared INCONCLUSIVE.',
                })
            else:
                print(f"    Cannot compute rho: only {valid.sum()} valid observations")
                e3_donor_rows.append({
                    'threshold': min_cells,
                    'effective_N': int(valid.sum()),
                    'donor_ids': ','.join(qualifying_ids),
                    'rho': np.nan,
                    'p_value': np.nan,
                    'note': f'Degenerate: only {valid.sum()} valid observations',
                })
        else:
            e3_donor_rows.append({
                'threshold': min_cells,
                'effective_N': n_qualifying,
                'donor_ids': ','.join(qualifying_ids),
                'rho': np.nan,
                'p_value': np.nan,
                'note': 'JUNB or SASP12_mean column missing',
            })
    else:
        # N >= 5: compute properly
        junb_vals = qual_donors['JUNB'].values.astype(float)
        sasp_vals = qual_donors['SASP12_mean'].values.astype(float)
        valid = np.isfinite(junb_vals) & np.isfinite(sasp_vals)
        rho, p = stats.spearmanr(junb_vals[valid], sasp_vals[valid])
        ci_lo, ci_hi = fisher_z_ci(rho, int(valid.sum()))
        print(f"    Spearman rho(JUNB, SASP12): {rho:.4f} (p={p:.4f}), "
              f"CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        e3_donor_rows.append({
            'threshold': min_cells,
            'effective_N': int(valid.sum()),
            'donor_ids': ','.join(qualifying_ids),
            'ages': ','.join([str(int(a)) for a in qual_donors['age'].dropna()]),
            'countries': ','.join([str(c) for c in qual_donors.get('country', pd.Series()).dropna()]),
            'cell_counts': ','.join([str(int(c)) for c in qual_donors['n_cells']]),
            'rho': float(rho),
            'p_value': float(p),
            'CI_lower': float(ci_lo),
            'CI_upper': float(ci_hi),
            'note': '',
        })

e3_donor_df = pd.DataFrame(e3_donor_rows)
e3_donor_path = f"{OUTDIR}/e3_runx2_fap_donor_level.csv"
e3_donor_df.to_csv(e3_donor_path, index=False)
print(f"\nE3 donor-level results saved to {e3_donor_path}")


# ============================================================
# SUMMARY: results.json
# ============================================================

print("\n" + "=" * 70)
print("Saving results.json summary")
print("=" * 70)

# Extract key findings for summary
e1_summary = {}
for compartment in ['Vascular', 'MuSC']:
    e1_comp = {}
    sub = e1_df[e1_df['compartment'] == compartment]
    for country in sub['country'].unique():
        c_sub = sub[sub['country'] == country]
        c_results = {}
        for _, r in c_sub.iterrows():
            key = f"{r['metric']}_{r['test']}"
            c_results[key] = {
                'estimate': r.get('estimate'),
                'p_value': r.get('p_value'),
                'R2': r.get('R2') if pd.notna(r.get('R2')) else None,
                'effect_size': r.get('effect_size') if pd.notna(r.get('effect_size', np.nan)) else None,
            }
        e1_comp[country] = {'N': int(c_sub['N'].iloc[0]), 'results': c_results}
    e1_summary[compartment] = e1_comp

# E2 summary: Jaccard for Vascular vs MuSC
e2_summary = {}
for comp in e2_results['jaccard_comparisons']:
    if comp['pair'] == 'Vascular_vs_MuSC':
        e2_summary[f"jaccard_{comp['threshold']}"] = comp['jaccard']

# E3 summary
e3_summary = {
    'cell_level': e3_cell_df.to_dict('records'),
    'donor_level': e3_donor_df.to_dict('records'),
    'cell_count_summary': {
        'total_runx2_cells': int(e3_counts_df['n_runx2_cells'].sum()),
        'donors_ge20': int((e3_counts_df['n_runx2_cells'] >= 20).sum()),
        'donors_ge10': int((e3_counts_df['n_runx2_cells'] >= 10).sum()),
    },
}

results_summary = {
    'batch': 'batch_073',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'environment': env_info,
    'experiments': {
        'E1_within_country_slopes': {
            'description': 'Within-country JUNB-age and SASP12-age regression slopes',
            'source': 'PI directive A1',
            'results': e1_summary,
        },
        'E2_egr2_polarity': {
            'description': 'EGR2 co-expression module target overlap between compartments',
            'source': 'PI directive A2',
            'caveat': 'Co-expression modules only (GRNBoost2), NOT motif-pruned. Interpretation capped at SUGGESTED.',
            'results': e2_summary,
        },
        'E3_runx2_fap': {
            'description': 'RUNX2+ FAP JUNB-SASP coupling (cell-level primary, donor-level secondary)',
            'source': 'PI directive A3',
            'results': e3_summary,
        },
    },
    'output_files': [
        'e1_junb_age_slopes.csv',
        'e2_egr2_polarity.json',
        'e2_egr2_targets.csv',
        'e3_runx2_fap_cell_level.csv',
        'e3_runx2_fap_donor_level.csv',
        'e3_runx2_fap_cell_counts.csv',
        'results.json',
    ],
}

results_path = f"{OUTDIR}/results.json"
with open(results_path, 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f"Results summary saved to {results_path}")
print(f"\n=== batch_073 COMPLETE ===")
