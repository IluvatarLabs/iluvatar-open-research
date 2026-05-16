#!/usr/bin/env python3
"""
batch_060: I2 Partial Correlations (age + sex + tech simultaneously)
=====================================================================

PURPOSE: Compute partial correlations for Vascular/MuSC/FAP compartments
controlling for age, sex, and tech simultaneously. batch_058 computed age-only
partial correlations; this analysis extends to full covariate adjustment.

WHY: The immune compartment (batch_059) found a severe tech confound (1 scRNA
vs 12 snRNA donors). We must control for tech in the other compartments too.
Additionally, sex is a known confounder in many aging studies.

DATA SOURCES:
  - Vascular: /home/yuanz/Documents/GitHub/biomarvin_fibro/data/Vascular_scsn_RNA.h5ad
  - MuSC: /home/yuanz/Documents/GitHub/biomarvin_fibro/data/MuSC_scsn_RNA.h5ad
  - FAP: /home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad

TFs: JUNB, CDKN1A, CEBPB, KLF10, EGR1, FOS (canonical TF-SASP regulators)

METHODOLOGY:
  - Donor-level aggregation (mean TF, mean SASP per donor) to avoid pseudoreplication
  - Raw Spearman correlation (baseline)
  - Partial correlation via pingouin.partial_corr (preferred) or residualization
  - Partial correlation controls for: age (numeric) + sex (binary) + tech (binary)
  - Report: TF, compartment, N, raw_rho, p_raw, partial_rho, p_partial, delta

OUTPUTS:
  - i2_partial_correlations.csv
  - results.json
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import json
import os
import warnings
import time

warnings.filterwarnings('ignore')

# =============================================================================
# Setup
# =============================================================================

BASE_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro"
DATA_DIR = f"{BASE_DIR}/data"
RESULTS_DIR = f"{BASE_DIR}/experiments/batch_060"
os.makedirs(RESULTS_DIR, exist_ok=True)

# SASP12 panel (same as used in batch_058/059)
SASP12 = ['CCL2', 'CCL7', 'CCL20', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL8', 'IL6',
          'MMP1', 'MMP3', 'SERPINE1', 'PLAU']

# TFs for analysis
TF_LIST = ['JUNB', 'CDKN1A', 'CEBPB', 'KLF10', 'EGR1', 'FOS']

# Compartment definitions
COMPARTMENTS = {
    'Vascular': f"{DATA_DIR}/Vascular_scsn_RNA.h5ad",
    'MuSC': f"{DATA_DIR}/MuSC_scsn_RNA.h5ad",
    'FAP': f"{DATA_DIR}/OMIX004308-02.h5ad",
}

# Key findings from batch_050 to compare
KEY_FINDINGS = {
    'Vascular': {'JUNB': 0.929, 'CDKN1A': 0.880},  # rho from HLMA
    'MuSC': {'CDKN1A': 0.929},  # rho from HLMA
    'FAP': {'JUNB': -0.014, 'CEBPB': 0.580},  # JUNB null, CEBPB moderate
}


def timestamp():
    """Current time string for progress logging."""
    return time.strftime('%Y-%m-%d %H:%M:%S')


def load_adata(path, compartment_name):
    """Load h5ad and extract obs metadata."""
    import scanpy as sc
    adata = sc.read_h5ad(path)
    return adata


def extract_donor_metadata(obs):
    """Extract donor-level metadata from obs DataFrame.

    WHY: Each compartment has different column names for donor, age, sex, tech.
    We standardize by searching for common patterns.

    Args:
        obs: adata.obs DataFrame
    Returns:
        dict with standardized columns
    """
    result = {}

    # Donor ID
    for col in ['sample', 'DonorID', 'donor_id', 'SampleID']:
        if col in obs.columns:
            result['donor_col'] = col
            break

    # Age
    for col in ['age', 'Age', 'age_val', 'AgeYears']:
        if col in obs.columns:
            result['age_col'] = col
            break
    # Try age_pop as fallback
    if 'age_col' not in result and 'age_pop' in obs.columns:
        result['age_col'] = 'age_pop'

    # Sex
    for col in ['sex', 'Sex', 'sex_val', 'Gender']:
        if col in obs.columns:
            result['sex_col'] = col
            break

    # Tech
    for col in ['tech', 'Tech', 'Tech_RNA', 'technology']:
        if col in obs.columns:
            result['tech_col'] = col
            break

    return result


def extract_age_numeric(obs, age_col):
    """Extract numeric age from age column.

    WHY: Some datasets store age in format like "OM1_82" or "82_years".
    We extract the numeric portion.

    Args:
        obs: adata.obs DataFrame
        age_col: column name containing age
    Returns:
        Series with numeric age values (as float)
    """
    if age_col not in obs.columns:
        return pd.Series([np.nan] * len(obs), index=obs.index)

    col_data = obs[age_col]

    # If already numeric, return as-is
    if pd.api.types.is_numeric_dtype(col_data):
        return col_data.astype(float)

    # Try to extract numeric from string
    def extract_number(x):
        try:
            return float(x)
        except:
            # Try split
            parts = str(x).split('_')
            for p in parts:
                try:
                    num = float(p)
                    if 10 <= num <= 110:  # reasonable age range
                        return num
                except:
                    continue
            return np.nan

    result = col_data.apply(extract_number)
    return pd.to_numeric(result, errors='coerce')


def compute_partial_correlation_pingouin(df, x_col, y_col, covar_cols):
    """Compute partial correlation using pingouin.

    WHY pingouin: It provides a well-tested implementation of partial correlation
    with proper statistical inference (p-values, confidence intervals).

    Args:
        df: DataFrame with columns
        x_col: column name for first variable
        y_col: column name for second variable
        covar_cols: list of covariate column names
    Returns:
        (partial_r, p_value) or (nan, nan) if insufficient data
    """
    try:
        import pingouin as pg

        covar_str = covar_cols  # pingouin expects list
        result = pg.partial_corr(data=df, x=x_col, y=y_col, covar=covar_str)

        # pingouin returns DataFrame with columns 'r' and 'p-val'
        partial_r = result['r'].values[0]
        p_val = result['p-val'].values[0]
        return float(partial_r), float(p_val)

    except ImportError:
        return None, None
    except Exception as e:
        print(f"    pingouin error: {e}")
        return None, None


def compute_partial_correlation_residualization(x, y, covariates):
    """Compute partial correlation via residualization.

    WHY residualization: When pingouin is unavailable, this is the standard
    fallback method. We rank-transform before residualizing to approximate
    Spearman partial correlation.

    Args:
        x, y: arrays to correlate
        covariates: list of covariate arrays (all numeric)
    Returns:
        (partial_r, p_value) or (nan, nan) if insufficient data
    """
    n = len(x)
    if n < 5:
        return np.nan, np.nan

    # Rank transform to approximate Spearman
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)

    # Build covariate matrix with intercept
    covar_mat = np.column_stack(covariates + [np.ones(n)])

    # Residualize x on covariates
    coef_x, _, _, _ = np.linalg.lstsq(covar_mat, x_rank, rcond=None)
    x_resid = x_rank - covar_mat @ coef_x

    # Residualize y on covariates
    coef_y, _, _, _ = np.linalg.lstsq(covar_mat, y_rank, rcond=None)
    y_resid = y_rank - covar_mat @ coef_y

    # Partial correlation is correlation of residuals
    rho, p = stats.spearmanr(x_resid, y_resid)
    return float(rho), float(p)


# =============================================================================
# Load Compartments
# =============================================================================

print("=" * 70)
print("batch_060: I2 Partial Correlations (age + sex + tech)")
print("=" * 70)
print(f"[{timestamp()}] Output: {RESULTS_DIR}")

# Store donor-level data per compartment
compartment_donor_data = {}

for comp_name, h5ad_path in COMPARTMENTS.items():
    print(f"\n[{timestamp()}] Loading {comp_name} from {h5ad_path}...")

    if not os.path.exists(h5ad_path):
        print(f"  ERROR: File not found: {h5ad_path}")
        continue

    adata = load_adata(h5ad_path, comp_name)
    print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    obs = adata.obs.copy()
    var_names = list(adata.var_names)

    # Extract column names
    col_map = extract_donor_metadata(obs)
    print(f"  Donor col: {col_map.get('donor_col', 'NOT FOUND')}")
    print(f"  Age col: {col_map.get('age_col', 'NOT FOUND')}")
    print(f"  Sex col: {col_map.get('sex_col', 'NOT FOUND')}")
    print(f"  Tech col: {col_map.get('tech_col', 'NOT FOUND')}")

    # Get donor ID column
    donor_col = col_map.get('donor_col')
    if donor_col is None:
        print(f"  ERROR: No donor column found")
        continue

    # Extract numeric age
    age_col = col_map.get('age_col')
    if age_col:
        obs['age_numeric'] = extract_age_numeric(obs, age_col)
    else:
        obs['age_numeric'] = np.nan

    # Extract sex (convert to binary 0/1)
    sex_col = col_map.get('sex_col')
    if sex_col:
        sex_data = obs[sex_col]
        if hasattr(sex_data, 'cat') and sex_data.dtype.name == 'category':
            # Convert categorical to codes
            sex_binary = sex_data.cat.codes
            # Handle any -1 codes (NA)
            sex_binary = sex_binary.replace(-1, -1).astype(float)
        else:
            # Map string values
            unique_sex = sex_data.dropna().unique()
            sex_map = {}
            for i, s in enumerate(unique_sex):
                sex_map[s] = float(i)
            sex_binary = sex_data.map(sex_map)
            sex_binary = sex_binary.fillna(-1)
        obs['sex_binary'] = sex_binary
        print(f"  Sex encoding: {dict(zip(*np.unique(sex_data.dropna(), return_index=False)))}")
    else:
        obs['sex_binary'] = pd.Series([-1.0] * len(obs), index=obs.index)

    # Extract tech (convert to binary 0/1)
    tech_col = col_map.get('tech_col')
    if tech_col:
        tech_data = obs[tech_col]
        if hasattr(tech_data, 'cat') and tech_data.dtype.name == 'category':
            tech_binary = tech_data.cat.codes
            tech_binary = tech_binary.replace(-1, -1).astype(float)
        else:
            unique_tech = tech_data.dropna().unique()
            tech_map = {}
            for i, t in enumerate(sorted(unique_tech)):
                tech_map[t] = float(i)
            tech_binary = tech_data.map(tech_map)
            tech_binary = tech_binary.fillna(-1)
        obs['tech_binary'] = tech_binary
        print(f"  Tech encoding: {dict(zip(*np.unique(tech_data.dropna(), return_index=False)))}")
    else:
        obs['tech_binary'] = pd.Series([-1.0] * len(obs), index=obs.index)

    # Check available SASP genes
    available_sasp = [g for g in SASP12 if g in var_names]
    print(f"  SASP12 detected: {len(available_sasp)}/{len(SASP12)}: {available_sasp}")

    # Compute SASP composite per cell
    if available_sasp:
        sasp_mat = adata[:, available_sasp].X
        if hasattr(sasp_mat, 'toarray'):
            sasp_scores = np.asarray(sasp_mat.toarray()).mean(axis=1)
        else:
            sasp_scores = np.asarray(sasp_mat).mean(axis=1)
    else:
        sasp_scores = np.zeros(len(obs))

    obs['SASP12'] = sasp_scores

    # Compute TF expression per cell
    for tf in TF_LIST:
        if tf in var_names:
            tf_mat = adata[:, tf].X
            if hasattr(tf_mat, 'toarray'):
                obs[f'TF_{tf}'] = np.asarray(tf_mat.toarray()).flatten()
            else:
                obs[f'TF_{tf}'] = np.asarray(tf_mat).flatten()

    # Aggregate to donor level
    agg_dict = {'SASP12': 'mean', 'age_numeric': 'first', 'sex_binary': 'first',
                'tech_binary': 'first'}
    for tf in TF_LIST:
        if f'TF_{tf}' in obs.columns:
            agg_dict[f'TF_{tf}'] = 'mean'

    donor_df = obs.groupby(donor_col).agg(agg_dict).reset_index()
    donor_df = donor_df.rename(columns={donor_col: 'donor_id'})
    donor_df['compartment'] = comp_name

    # Ensure numeric columns are float (not categorical)
    for col in ['age_numeric', 'sex_binary', 'tech_binary']:
        if col in donor_df.columns:
            donor_df[col] = pd.to_numeric(donor_df[col], errors='coerce')

    # Count cells per donor
    cell_counts = obs.groupby(donor_col).size().reset_index()
    cell_counts.columns = ['donor_id', 'n_cells']
    donor_df = donor_df.merge(cell_counts, on='donor_id')

    # Drop donors with missing age or all-missing TF
    n_before = len(donor_df)
    donor_df = donor_df.dropna(subset=['age_numeric'])
    n_after = len(donor_df)
    print(f"  Donors: {n_before} -> {n_after} (after age filter)")

    compartment_donor_data[comp_name] = donor_df

    print(f"  Final: N={len(donor_df)} donors")
    print(f"  Age range: {float(donor_df['age_numeric'].min()):.0f} - {float(donor_df['age_numeric'].max()):.0f}")
    print(f"  Sex values: {sorted(donor_df['sex_binary'].unique())}")
    print(f"  Tech values: {sorted(donor_df['tech_binary'].unique())}")

# =============================================================================
# Compute Partial Correlations
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: Partial Correlation Analysis")
print("=" * 70)

# Store results
results_rows = []

for comp_name, donor_df in compartment_donor_data.items():
    print(f"\n[{comp_name}] N={len(donor_df)} donors")

    n_donors = len(donor_df)

    for tf in TF_LIST:
        tf_col = f'TF_{tf}'
        if tf_col not in donor_df.columns:
            print(f"  {tf}: TF not in data, SKIPPED")
            continue

        # Extract arrays
        tf_vals = donor_df[tf_col].values
        sasp_vals = donor_df['SASP12'].values
        age_vals = donor_df['age_numeric'].values
        sex_vals = donor_df['sex_binary'].values
        tech_vals = donor_df['tech_binary'].values

        # Remove any rows with missing values
        valid_mask = ~(np.isnan(tf_vals) | np.isnan(sasp_vals) |
                       np.isnan(age_vals) | (sex_vals < 0) | (tech_vals < 0))
        if valid_mask.sum() < 5:
            print(f"  {tf}: <5 valid donors, SKIPPED")
            continue

        tf_clean = tf_vals[valid_mask]
        sasp_clean = sasp_vals[valid_mask]
        age_clean = age_vals[valid_mask]
        sex_clean = sex_vals[valid_mask]
        tech_clean = tech_vals[valid_mask]

        n_valid = len(tf_clean)

        # Raw correlation
        rho_raw, p_raw = stats.spearmanr(tf_clean, sasp_clean)

        # Partial correlation (age + sex + tech)
        # Try pingouin first, then residualization
        partial_r, p_partial = compute_partial_correlation_pingouin(
            pd.DataFrame({
                'TF': tf_clean,
                'SASP': sasp_clean,
                'age': age_clean,
                'sex': sex_clean,
                'tech': tech_clean,
            }),
            'TF', 'SASP', ['age', 'sex', 'tech']
        )

        if partial_r is None:
            # Fallback to residualization
            partial_r, p_partial = compute_partial_correlation_residualization(
                tf_clean, sasp_clean, [age_clean, sex_clean, tech_clean]
            )

        delta = partial_r - rho_raw if not np.isnan(partial_r) else np.nan

        results_rows.append({
            'TF': tf,
            'compartment': comp_name,
            'N': n_valid,
            'raw_rho': round(rho_raw, 4),
            'p_raw': p_raw,
            'partial_rho': round(partial_r, 4) if not np.isnan(partial_r) else np.nan,
            'p_partial': p_partial if not np.isnan(p_partial) else np.nan,
            'delta': round(delta, 4) if not np.isnan(delta) else np.nan,
            'raw_sig': p_raw < 0.05,
            'partial_sig': p_partial < 0.05 if not np.isnan(p_partial) else False,
        })

        # Print summary
        sig_raw = "*" if p_raw < 0.05 else ""
        sig_partial = "*" if p_partial < 0.05 else ""
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
        partial_str = f"{partial_r:.4f}" if not np.isnan(partial_r) else "N/A"
        print(f"  {tf}: raw={rho_raw:.4f}{sig_raw} -> partial={partial_str}{sig_partial} "
              f"(delta={delta_str})")

results_df = pd.DataFrame(results_rows)

# Sort by compartment and |delta|
results_df['abs_delta'] = results_df['delta'].abs()
results_df = results_df.sort_values(['compartment', 'abs_delta'], ascending=[True, False])

# Save results
results_path = f"{RESULTS_DIR}/i2_partial_correlations.csv"
results_df.to_csv(results_path, index=False)
print(f"\n[{timestamp()}] Saved: {results_path}")

# =============================================================================
# Summary Statistics
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Per-compartment summary
for comp_name in COMPARTMENTS.keys():
    comp_results = results_df[results_df['compartment'] == comp_name]
    if len(comp_results) == 0:
        continue

    print(f"\n{comp_name} (N={comp_results['N'].iloc[0]} donors):")

    # Survival analysis: which TFs survive with |partial_rho| > 0.5 and p < 0.05?
    surviving = comp_results[comp_results['partial_sig'] & (comp_results['partial_rho'].abs() > 0.5)]
    collapsing = comp_results[comp_results['raw_sig'] & ~comp_results['partial_sig']]

    print(f"  Survive adjustment (|partial_rho| > 0.5, p < 0.05): {len(surviving)} TFs")
    for _, row in surviving.iterrows():
        print(f"    {row['TF']}: raw={row['raw_rho']:.4f} -> partial={row['partial_rho']:.4f}")

    print(f"  Collapse on adjustment (raw sig, partial not sig): {len(collapsing)} TFs")
    for _, row in collapsing.iterrows():
        print(f"    {row['TF']}: raw={row['raw_rho']:.4f} -> partial={row['partial_rho']:.4f}")

    # Compare to key findings
    print(f"  Comparison to batch_050:")
    if comp_name in KEY_FINDINGS:
        for tf, ref_rho in KEY_FINDINGS[comp_name].items():
            tf_row = comp_results[comp_results['TF'] == tf]
            if len(tf_row) > 0:
                actual_raw = tf_row['raw_rho'].iloc[0]
                actual_partial = tf_row['partial_rho'].iloc[0]
                delta = actual_partial - ref_rho if not np.isnan(actual_partial) else np.nan
                print(f"    {tf}: batch_050={ref_rho:.3f}, batch_060_raw={actual_raw:.3f}, "
                      f"batch_060_partial={actual_partial:.3f} (delta={delta:+.3f})")

# Overall summary
print("\n" + "-" * 60)
print("Key Findings:")
all_survive = results_df[results_df['partial_sig'] & (results_df['partial_rho'].abs() > 0.5)]
print(f"  TFs surviving full adjustment (|partial_rho| > 0.5, p < 0.05): {len(all_survive)}")
for _, row in all_survive.iterrows():
    print(f"    {row['compartment']}/{row['TF']}: partial_rho={row['partial_rho']:.4f}")

# =============================================================================
# Save results.json
# =============================================================================

key_findings = {
    'survive_thresholds': {
        'min_partial_rho': 0.5,
        'max_p_value': 0.05,
    },
    'total_tested': len(results_df),
    'survive_count': len(all_survive),
    'survive_list': [
        {'TF': row['TF'], 'compartment': row['compartment'],
         'partial_rho': row['partial_rho'], 'p_partial': row['p_partial']}
        for _, row in all_survive.iterrows()
    ],
    'compartment_summary': {},
}

for comp_name in COMPARTMENTS.keys():
    comp_results = results_df[results_df['compartment'] == comp_name]
    if len(comp_results) == 0:
        continue

    surviving = comp_results[comp_results['partial_sig'] & (comp_results['partial_rho'].abs() > 0.5)]
    key_findings['compartment_summary'][comp_name] = {
        'n_donors': int(comp_results['N'].iloc[0]),
        'n_tested': len(comp_results),
        'n_survive': len(surviving),
        'surviving_tfs': surviving['TF'].tolist() if len(surviving) > 0 else [],
    }

summary_json = {
    'batch': 'batch_060',
    'script': 'run_i2_partial.py',
    'date': pd.Timestamp.now().isoformat(),
    'method': {
        'approach': 'Partial correlation controlling for age + sex + tech simultaneously',
        'implementation': 'pingouin.partial_corr (fallback: residualization)',
        'correlation_type': 'Spearman partial correlation',
    },
    'data_sources': {
        'Vascular': f"{DATA_DIR}/Vascular_scsn_RNA.h5ad",
        'MuSC': f"{DATA_DIR}/MuSC_scsn_RNA.h5ad",
        'FAP': f"{DATA_DIR}/OMIX004308-02.h5ad",
    },
    'TFs_analyzed': TF_LIST,
    'results_summary': key_findings,
    'output_files': [
        'i2_partial_correlations.csv',
        'results.json',
    ],
}

results_json_path = f"{RESULTS_DIR}/results.json"
with open(results_json_path, 'w') as f:
    json.dump(summary_json, f, indent=2, default=str)
print(f"\n[{timestamp()}] Saved: {results_json_path}")

print(f"\n[{timestamp()}] batch_060 COMPLETE")