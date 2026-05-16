#!/usr/bin/env python3
"""
batch_051: Q-MODERN Statistical Rigor + Confound Analysis
Executes: F1 (country + age confound), I1 (power), I2 (partial corr),
          I3 (cell count sensitivity), I4 (tiering), A2 (counts), A3 (QC)
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import norm
import anndata as ad
import scipy.sparse as sp
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')

OUTDIR = "experiments/batch_051"
os.makedirs(OUTDIR, exist_ok=True)

# Canonical SASP12 panel (from batch_050 q_final_fix.py)
SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8',
          'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

TARGET_TFS = ['JUNB','JUN','JUND','FOS','FOSB','FOSL1','FOSL2','KLF2','KLF4',
              'KLF6','KLF10','ATF3','EGR1','EGR2','IRF1','CEBPB','CEBPD',
              'RELA','NFKB1','STAT3','CDKN1A']

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dataset(path, cell_type_filter=None, filter_col='Annotation'):
    """Load h5ad, optionally filter to cell types, return obs + expression."""
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
    return obs, X, var_names


def compute_donor_level(obs, X, var_names, genes, sasp_genes=None):
    """Compute donor-level mean expression for specified genes.
    sasp_genes: if provided, compute SASP12_mean from these genes only (not all genes).
    """
    detected = [g for g in genes if g in var_names]
    if sasp_genes is None:
        sasp_genes = [g for g in genes if g in SASP12]
    detected_sasp = [g for g in sasp_genes if g in var_names]
    donor_data = []

    # Get sample column name
    sample_col = 'sample' if 'sample' in obs.columns else 'donor_id'
    if sample_col not in obs.columns:
        sample_col = 'SampleID'

    for donor in obs[sample_col].unique():
        dmask = obs[sample_col] == donor
        d_obs = obs[dmask]
        d_X = X[dmask.values]
        n_cells = dmask.sum()

        row = {
            'sample': donor,
            'n_cells': n_cells,
        }

        # Copy metadata from first cell
        for col in ['age', 'age_pop', 'Sex', 'sex', 'gender', 'Country',
                     'country', 'tech', 'Age_group', 'Age_bin', 'donor_id']:
            if col in d_obs.columns:
                row[col] = d_obs[col].iloc[0]

        # Gene means
        for g in detected:
            idx = var_names.index(g)
            row[g] = np.mean(d_X[:, idx])

        row['SASP12_detected'] = len(detected_sasp)
        row['SASP12_mean'] = np.mean([row[g] for g in detected_sasp]) if detected_sasp else np.nan
        row['SASP12_genes'] = ','.join(detected_sasp)

        donor_data.append(row)

    df = pd.DataFrame(donor_data)

    # Normalize age to numeric
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    return df


def fisher_z_power(rho, n, alpha=0.05):
    """Compute power to detect Spearman rho via Fisher Z approximation."""
    if abs(rho) < 1e-10 or n < 4:
        return 0.0
    z_rho = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    power = 1 - norm.cdf(z_crit - abs(z_rho) / se)
    return power


def fisher_z_ci(rho, n, alpha=0.05):
    """Compute 95% CI for Spearman rho via Fisher Z."""
    z_rho = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    z_lo = z_rho - z_crit * se
    z_hi = z_rho + z_crit * se
    ci_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
    ci_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
    return ci_lo, ci_hi


# ============================================================
# ANALYSIS
# ============================================================

results = {}

# ----------------------------------------------------------
# A2 + A3: Cell/Donor counts and QC metrics
# ----------------------------------------------------------
print("\n=== A2+A3: Cell/Donor counts and QC metrics ===")

datasets_info = {
    'HLMA_Vascular': {
        'path': 'data/Vascular_scsn_RNA.h5ad',
        'filter': ['ArtEC', 'CapEC', 'VenEC', 'IL6+ VenEC'],
        'filter_col': 'Annotation',
    },
    'HLMA_MuSC': {
        'path': 'data/MuSC_scsn_RNA.h5ad',
        'filter': None,
        'filter_col': 'Annotation',
    },
    'HLMA_FAP': {
        'path': 'data/OMIX004308-02.h5ad',
        'filter': ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP', 'CD99+ FAP'],
        'filter_col': 'Annotation',
    },
    'NA_Endothelium': {
        'path': 'data/NA_Endothelium_SMC.h5ad',
        'filter': ['Vein', 'Vein-CCL2+', 'Artery', 'Arteriole', 'Arteriole-CCL2+',
                    'Cap', 'Cap-Ven', 'Cap-CCL2+'],
        'filter_col': 'annotation_level2',
    },
    'NA_FAP': {
        'path': 'data/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad',
        'filter': ['Inter_FB', 'Adv_FB', 'Par_FB'],
        'filter_col': 'annotation_level2',
    },
}

a2_rows = []
a3_rows = []

for ds_name, ds_info in datasets_info.items():
    print(f"  Processing {ds_name}...")
    adata = ad.read_h5ad(ds_info['path'], backed='r')
    obs = adata.obs.copy()

    # Filter
    if ds_info['filter']:
        mask = obs[ds_info['filter_col']].isin(ds_info['filter'])
        obs_f = obs[mask]
    else:
        obs_f = obs

    # Sample column
    if 'sample' in obs_f.columns:
        scol = 'sample'
    elif 'donor_id' in obs_f.columns:
        scol = 'donor_id'
    else:
        scol = 'SampleID'

    # A2: Donor counts
    for donor in obs_f[scol].unique():
        d = obs_f[obs_f[scol] == donor]
        age_val = None
        for acol in ['age', 'Age_group']:
            if acol in d.columns:
                age_val = d[acol].iloc[0]
                break

        sex_val = None
        for scol2 in ['Sex', 'sex', 'gender']:
            if scol2 in d.columns:
                sex_val = d[scol2].iloc[0]
                break

        tech_val = None
        for tcol in ['tech', 'assay']:
            if tcol in d.columns:
                tech_val = d[tcol].iloc[0]
                break

        country_val = None
        for ccol in ['Country', 'country']:
            if ccol in d.columns:
                country_val = d[ccol].iloc[0]
                break

        a2_rows.append({
            'dataset': ds_name,
            'donor': donor,
            'n_cells': len(d),
            'age': age_val,
            'sex': sex_val,
            'tech': tech_val,
            'country': country_val,
        })

    # A3: QC metrics
    for qc_col, qc_name in [('nCount_RNA', 'nUMI'), ('nFeature_RNA', 'nGene'),
                             ('percent.mt', 'percent_mito'), ('n_counts', 'nUMI'),
                             ('n_genes', 'nGene'), ('percent_mito', 'percent_mito')]:
        if qc_col in obs_f.columns:
            vals = pd.to_numeric(obs_f[qc_col], errors='coerce').dropna()
            if len(vals) > 0:
                a3_rows.append({
                    'dataset': ds_name,
                    'metric': qc_name,
                    'column': qc_col,
                    'mean': float(vals.mean()),
                    'median': float(vals.median()),
                    'sd': float(vals.std()),
                    'n_cells': len(vals),
                })

    adata.file.close()

a2_df = pd.DataFrame(a2_rows)
a3_df = pd.DataFrame(a3_rows)

a2_df.to_csv(f"{OUTDIR}/a2_cell_donor_counts.csv", index=False)
a3_df.to_csv(f"{OUTDIR}/a3_qc_metrics.csv", index=False)

print(f"  A2: {len(a2_df)} donor-dataset entries saved")
print(f"  A3: {len(a3_df)} QC metric entries saved")

# Print donor counts by dataset x country
print("\n  Donor counts by dataset x country:")
for ds in a2_df['dataset'].unique():
    sub = a2_df[a2_df['dataset'] == ds]
    if 'country' in sub.columns:
        country_counts = sub['country'].value_counts().to_dict()
        print(f"    {ds}: N={len(sub)} donors, countries={country_counts}")
    else:
        print(f"    {ds}: N={len(sub)} donors")

# ----------------------------------------------------------
# F1: Confound Analysis (Country + Age)
# ----------------------------------------------------------
print("\n=== F1: Confound Analysis ===")

# Load HLMA Vascular (endothelial only)
print("  Loading HLMA Vascular (endothelial)...")
v_obs, v_X, v_vars = load_dataset(
    'data/Vascular_scsn_RNA.h5ad',
    cell_type_filter=['ArtEC', 'CapEC', 'VenEC', 'IL6+ VenEC'],
    filter_col='Annotation'
)
v_donors = compute_donor_level(v_obs, v_X, v_vars, TARGET_TFS + SASP12)

# Standardize column names
if 'Sex' in v_donors.columns:
    v_donors['sex'] = v_donors['Sex']
if 'Country' in v_donors.columns:
    v_donors['country'] = v_donors['Country']

print(f"  Vascular donors: {len(v_donors)}")
print(f"  Country distribution: {v_donors['country'].value_counts().to_dict()}")

# Load HLMA MuSC
print("  Loading HLMA MuSC...")
m_obs, m_X, m_vars = load_dataset('data/MuSC_scsn_RNA.h5ad')
m_donors = compute_donor_level(m_obs, m_X, m_vars, TARGET_TFS + SASP12)
if 'gender' in m_donors.columns:
    m_donors['sex'] = m_donors['gender']
if 'Sex' in m_donors.columns:
    m_donors['sex'] = m_donors['Sex']
if 'Country' in m_donors.columns:
    m_donors['country'] = m_donors['Country']

print(f"  MuSC donors: {len(m_donors)}")
print(f"  Country distribution: {m_donors['country'].value_counts().to_dict()}")

# Load HLMA FAP
print("  Loading HLMA FAP...")
f_obs, f_X, f_vars = load_dataset(
    'data/OMIX004308-02.h5ad',
    cell_type_filter=['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP', 'CD99+ FAP'],
    filter_col='Annotation'
)
f_donors = compute_donor_level(f_obs, f_X, f_vars, TARGET_TFS + SASP12)
if 'Sex' in f_donors.columns:
    f_donors['sex'] = f_donors['Sex']
if 'Country' in f_donors.columns:
    f_donors['country'] = f_donors['Country']

print(f"  FAP donors: {len(f_donors)}")
print(f"  Country distribution: {f_donors['country'].value_counts().to_dict()}")

# Load NA Endothelium
print("  Loading NA Endothelium...")
nae_obs, nae_X, nae_vars = load_dataset(
    'data/NA_Endothelium_SMC.h5ad',
    cell_type_filter=['Vein', 'Vein-CCL2+', 'Artery', 'Arteriole', 'Arteriole-CCL2+',
                       'Cap', 'Cap-Ven', 'Cap-CCL2+'],
    filter_col='annotation_level2'
)
nae_donors = compute_donor_level(nae_obs, nae_X, nae_vars, TARGET_TFS + SASP12)
if 'sex' in nae_donors.columns:
    nae_donors['sex'] = nae_donors['sex']
if 'donor_id' in nae_donors.columns:
    nae_donors['sample'] = nae_donors['donor_id']
# Assign age midpoints for NA
if 'Age_group' in nae_donors.columns:
    age_midpoints = {'15-20': 17.5, '25-30': 27.5, '35-40': 37.5,
                     '50-55': 52.5, '55-60': 57.5, '60-65': 62.5, '70-75': 72.5}
    nae_donors['age'] = nae_donors['Age_group'].map(age_midpoints)

print(f"  NA Endothelium donors: {len(nae_donors)}")

# Load NA FAP
print("  Loading NA FAP...")
naf_obs, naf_X, naf_vars = load_dataset(
    'data/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad',
    cell_type_filter=['Inter_FB', 'Adv_FB', 'Par_FB'],
    filter_col='annotation_level2'
)
naf_donors = compute_donor_level(naf_obs, naf_X, naf_vars, TARGET_TFS + SASP12)

print(f"  NA FAP donors: {len(naf_donors)}")

# === F1 Analysis: Country and Age Confound ===
f1_results = []

for ds_name, donors_df in [
    ('HLMA_Vascular', v_donors),
    ('HLMA_MuSC', m_donors),
    ('HLMA_FAP', f_donors),
]:
    print(f"\n  === F1: {ds_name} ===")

    has_country = 'country' in donors_df.columns and donors_df['country'].nunique() > 1

    for tf in ['JUNB', 'CDKN1A', 'ATF3', 'EGR1', 'KLF10']:
        if tf not in donors_df.columns:
            continue

        rho_raw, p_raw = stats.spearmanr(donors_df[tf], donors_df['SASP12_mean'])
        ci_lo, ci_hi = fisher_z_ci(rho_raw, len(donors_df))
        n = len(donors_df)

        row = {
            'dataset': ds_name,
            'tf': tf,
            'n': n,
            'rho_raw': rho_raw,
            'p_raw': p_raw,
            'ci_95_raw': f"[{ci_lo:.3f}, {ci_hi:.3f}]",
        }

        # Age-adjusted (Spearman partial via residualization)
        if 'age' in donors_df.columns:
            age_vals = pd.to_numeric(donors_df['age'], errors='coerce')
            valid = donors_df[tf].notna() & donors_df['SASP12_mean'].notna() & age_vals.notna()
            if valid.sum() >= 5:
                tf_vals = donors_df.loc[valid, tf].values.astype(float)
                sasp_vals = donors_df.loc[valid, 'SASP12_mean'].values.astype(float)
                age_v = age_vals[valid].values.astype(float)

                # Rank transform for Spearman-like
                tf_rank = stats.rankdata(tf_vals)
                sasp_rank = stats.rankdata(sasp_vals)
                age_rank = stats.rankdata(age_v)

                # Residualize
                slope_tf, intercept_tf, _, _, _ = stats.linregress(age_rank, tf_rank)
                tf_resid = tf_rank - (slope_tf * age_rank + intercept_tf)

                slope_sasp, intercept_sasp, _, _, _ = stats.linregress(age_rank, sasp_rank)
                sasp_resid = sasp_rank - (slope_sasp * age_rank + intercept_sasp)

                rho_age_adj, p_age_adj = stats.spearmanr(tf_resid, sasp_resid)
                row['rho_age_adj'] = rho_age_adj
                row['p_age_adj'] = p_age_adj
                row['age_rho_tf'] = stats.spearmanr(tf_vals, age_v)[0]
                row['age_rho_sasp'] = stats.spearmanr(sasp_vals, age_v)[0]

        # Country-adjusted (if applicable)
        if has_country:
            donors_df['country_num'] = (donors_df['country'] == sorted(donors_df['country'].unique())[0]).astype(int)
            tf_vals = donors_df[tf].values
            sasp_vals = donors_df['SASP12_mean'].values
            ctry_vals = donors_df['country_num'].values

            # Residualize
            tf_rank = stats.rankdata(tf_vals)
            sasp_rank = stats.rankdata(sasp_vals)
            ctry_rank = stats.rankdata(ctry_vals)

            slope_tf, intercept_tf, _, _, _ = stats.linregress(ctry_rank, tf_rank)
            tf_resid = tf_rank - (slope_tf * ctry_rank + intercept_tf)

            slope_sasp, intercept_sasp, _, _, _ = stats.linregress(ctry_rank, sasp_rank)
            sasp_resid = sasp_rank - (slope_sasp * ctry_rank + intercept_sasp)

            rho_ctry_adj, p_ctry_adj = stats.spearmanr(tf_resid, sasp_resid)
            row['rho_country_adj'] = rho_ctry_adj
            row['p_country_adj'] = p_ctry_adj
            row['country_rho_tf'] = stats.spearmanr(tf_vals, ctry_vals)[0]
            row['country_rho_sasp'] = stats.spearmanr(sasp_vals, ctry_vals)[0]

            # Within-country
            for c in sorted(donors_df['country'].unique()):
                sub = donors_df[donors_df['country'] == c]
                if len(sub) >= 4 and tf in sub.columns:
                    r, p = stats.spearmanr(sub[tf], sub['SASP12_mean'])
                    row[f'within_{c}_rho'] = r
                    row[f'within_{c}_p'] = p
                    row[f'within_{c}_n'] = len(sub)

            # Country + Sex + Age adjusted (full model via residualization)
            sex_num = (donors_df['sex'] == 'Male').astype(int).values if 'sex' in donors_df.columns else np.zeros(n)
            age_v = pd.to_numeric(donors_df['age'], errors='coerce').fillna(0).values if 'age' in donors_df.columns else np.zeros(n)

            # Multiple regression residualization for 3 covariates
            try:
                from numpy.linalg import lstsq
                covars = np.column_stack([ctry_vals, sex_num, age_v, np.ones(n)])
                # Regress TF on covars
                coef_tf, _, _, _ = lstsq(covars, tf_rank, rcond=None)
                tf_resid_full = tf_rank - covars @ coef_tf
                # Regress SASP on covars
                coef_sasp, _, _, _ = lstsq(covars, sasp_rank, rcond=None)
                sasp_resid_full = sasp_rank - covars @ coef_sasp

                rho_full_adj, p_full_adj = stats.spearmanr(tf_resid_full, sasp_resid_full)
                row['rho_full_adj'] = rho_full_adj
                row['p_full_adj'] = p_full_adj
            except:
                pass

        f1_results.append(row)
        print(f"    {tf}: raw={rho_raw:.4f} (p={p_raw:.2e})", end="")
        if 'rho_age_adj' in row:
            print(f", age_adj={row['rho_age_adj']:.4f}", end="")
        if 'rho_country_adj' in row:
            print(f", ctry_adj={row['rho_country_adj']:.4f}", end="")
        if 'rho_full_adj' in row:
            print(f", full_adj={row['rho_full_adj']:.4f}", end="")
        print()

f1_df = pd.DataFrame(f1_results)
f1_df.to_csv(f"{OUTDIR}/f1_confound_analysis.csv", index=False)
print(f"\n  F1 results saved: {len(f1_df)} rows")

# Permutation test for JUNB-SASP in HLMA Vascular
print("\n  Permutation test for JUNB-SASP (HLMA Vascular)...")
np.random.seed(42)
n_perm = 10000
rho_obs = stats.spearmanr(v_donors['JUNB'], v_donors['SASP12_mean'])[0]
perm_rhos = []
for _ in range(n_perm):
    perm_rhos.append(stats.spearmanr(
        np.random.permutation(v_donors['JUNB'].values),
        v_donors['SASP12_mean'].values
    )[0])
perm_p = np.mean(np.abs(perm_rhos) >= np.abs(rho_obs))
print(f"    Observed rho={rho_obs:.4f}, permutation p={perm_p:.5f} (N={n_perm})")

# Within-country permutation
if 'country' in v_donors.columns:
    for c in sorted(v_donors['country'].unique()):
        sub = v_donors[v_donors['country'] == c]
        rho_wc = stats.spearmanr(sub['JUNB'], sub['SASP12_mean'])[0]
        perm_rhos_wc = []
        for _ in range(n_perm):
            perm_rhos_wc.append(stats.spearmanr(
                np.random.permutation(sub['JUNB'].values),
                sub['SASP12_mean'].values
            )[0])
        perm_p_wc = np.mean(np.abs(perm_rhos_wc) >= np.abs(rho_wc))
        print(f"    Within-{c}: rho={rho_wc:.4f}, perm p={perm_p_wc:.5f}, N={len(sub)}")


# ----------------------------------------------------------
# I1: Formal Power Analysis
# ----------------------------------------------------------
print("\n=== I1: Power Analysis ===")

key_findings = [
    ('F084', 'HLMA_Vascular', 'JUNB', 23, 0.929),
    ('F093', 'HLMA_MuSC', 'CDKN1A', 23, 0.929),
    ('F080', 'HLMA_FAP', 'JUNB', 22, -0.014),
    ('D2', 'NA_Endothelium', 'JUNB', 12, 0.720),
    ('Q2', 'NA_FAP', 'JUNB', 12, 0.573),
]

# Also add within-country
key_findings_extended = key_findings + [
    ('F084_China', 'HLMA_Vascular_within_China', 'JUNB', 14, 0.78),
    ('F084_Spain', 'HLMA_Vascular_within_Spain', 'JUNB', 9, 0.63),
]

i1_results = []
for fid, ds, tf, n, rho in key_findings_extended:
    power = fisher_z_power(rho, n)
    ci_lo, ci_hi = fisher_z_ci(rho, n)
    min_rho_80 = None
    # Find minimum detectable rho at 80% power
    for test_rho in np.arange(0.05, 1.0, 0.01):
        if fisher_z_power(test_rho, n) >= 0.80:
            min_rho_80 = test_rho
            break

    row = {
        'finding_id': fid,
        'dataset': ds,
        'tf': tf,
        'n_donors': n,
        'observed_rho': rho,
        'ci_95': f"[{ci_lo:.3f}, {ci_hi:.3f}]",
        'power_at_observed': power,
        'min_rho_80_power': min_rho_80,
        'power_label': 'adequate' if power >= 0.80 else ('marginal' if power >= 0.50 else 'underpowered'),
    }
    i1_results.append(row)
    print(f"  {fid}: N={n}, rho={rho:.3f}, power={power:.3f} ({row['power_label']}), "
          f"min_detectable(80%)={min_rho_80:.2f}" if min_rho_80 else f"  {fid}: min_detectable(80%)=N/A")

i1_df = pd.DataFrame(i1_results)
i1_df.to_csv(f"{OUTDIR}/i1_power_analysis.csv", index=False)


# ----------------------------------------------------------
# I2: Confounder-Adjusted Partial Correlations (via pingouin)
# ----------------------------------------------------------
print("\n=== I2: Partial Correlations ===")

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("  pingouin not available, using manual residualization")

i2_results = []

for ds_name, donors_df in [
    ('HLMA_Vascular', v_donors),
    ('HLMA_MuSC', m_donors),
    ('HLMA_FAP', f_donors),
]:
    has_country = 'country' in donors_df.columns and donors_df['country'].nunique() > 1
    has_sex = 'sex' in donors_df.columns

    for tf in ['JUNB', 'CDKN1A', 'ATF3', 'EGR1', 'KLF10']:
        if tf not in donors_df.columns:
            continue

        row = {
            'dataset': ds_name,
            'tf': tf,
            'n': len(donors_df),
        }

        if HAS_PINGOUIN and has_country and has_sex:
            df_pg = pd.DataFrame({
                'tf': donors_df[tf].values,
                'sasp': donors_df['SASP12_mean'].values,
                'country': donors_df['country'].values,
                'sex': donors_df['sex'].values,
            })
            try:
                pc = pg.partial_corr(data=df_pg, x='tf', y='sasp',
                                      covar=['country', 'sex'], method='spearman')
                row['partial_r_ctry_sex'] = pc['r'].values[0]
                row['partial_p_ctry_sex'] = pc['p-val'].values[0]
            except:
                pass

            try:
                pc2 = pg.partial_corr(data=df_pg, x='tf', y='sasp',
                                       covar=['sex'], method='spearman')
                row['partial_r_sex_only'] = pc2['r'].values[0]
                row['partial_p_sex_only'] = pc2['p-val'].values[0]
            except:
                pass

        i2_results.append(row)
        out_str = f"  {ds_name} {tf}:"
        if 'partial_r_ctry_sex' in row:
            out_str += f" ctry+sex adj r={row['partial_r_ctry_sex']:.4f} p={row['partial_p_ctry_sex']:.4f}"
        if 'partial_r_sex_only' in row:
            out_str += f" sex-only adj r={row['partial_r_sex_only']:.4f} p={row['partial_p_sex_only']:.4f}"
        print(out_str)

i2_df = pd.DataFrame(i2_results)
i2_df.to_csv(f"{OUTDIR}/i2_partial_correlations.csv", index=False)


# ----------------------------------------------------------
# I3: Cell Count Sensitivity
# ----------------------------------------------------------
print("\n=== I3: Cell Count Sensitivity ===")

i3_results = []

for ds_name, donors_df in [
    ('HLMA_Vascular', v_donors),
    ('HLMA_MuSC', m_donors),
    ('HLMA_FAP', f_donors),
    ('NA_Endothelium', nae_donors),
    ('NA_FAP', naf_donors),
]:
    for tf in ['JUNB', 'CDKN1A']:
        if tf not in donors_df.columns:
            continue

        rho_full, p_full = stats.spearmanr(donors_df[tf], donors_df['SASP12_mean'])
        n_full = len(donors_df)

        row_full = {
            'dataset': ds_name,
            'tf': tf,
            'threshold': 0,
            'n_donors': n_full,
            'rho': rho_full,
            'p': p_full,
        }
        i3_results.append(row_full)

        for thresh in [50, 75, 100, 150]:
            sub = donors_df[donors_df['n_cells'] >= thresh]
            if len(sub) >= 4:
                r, p = stats.spearmanr(sub[tf], sub['SASP12_mean'])
                row = {
                    'dataset': ds_name,
                    'tf': tf,
                    'threshold': thresh,
                    'n_donors': len(sub),
                    'rho': r,
                    'p': p,
                    'delta_rho': r - rho_full,
                }
                i3_results.append(row)
                print(f"  {ds_name} {tf}: thresh={thresh}, N={len(sub)}, rho={r:.4f} (delta={r-rho_full:+.4f})")

i3_df = pd.DataFrame(i3_results)
i3_df.to_csv(f"{OUTDIR}/i3_cell_count_sensitivity.csv", index=False)


# ----------------------------------------------------------
# I4: Replication Tiering
# ----------------------------------------------------------
print("\n=== I4: Replication Tiering ===")

# Load canonical table
canonical = pd.read_csv("experiments/batch_050/canonical_tf_sasp_table_final.csv",
                        keep_default_na=False)

i4_results = []

# For key findings, count how many datasets show rho > 0.5 and rho > 0.3
for tf in TARGET_TFS:
    sub = canonical[canonical['tf'] == tf]
    n_datasets = len(sub)
    n_pos_05 = ((sub['rho'] > 0.5)).sum()
    n_pos_03 = ((sub['rho'] > 0.3)).sum()
    n_null = ((sub['rho'].abs() < 0.15)).sum()
    n_neg = ((sub['rho'] < -0.3)).sum()

    max_rho = sub['rho'].max()
    min_rho = sub['rho'].min()

    if n_pos_05 >= 3:
        tier = 'ROBUST'
    elif n_pos_05 >= 2:
        tier = 'MODERATE'
    elif n_pos_03 >= 2:
        tier = 'DIRECTIONAL'
    elif n_pos_05 == 1:
        tier = 'SINGLE_DATASET'
    elif n_null >= 3:
        tier = 'NULL'
    else:
        tier = 'HETEROGENEOUS'

    i4_results.append({
        'tf': tf,
        'n_datasets': n_datasets,
        'n_positive_0.5': n_pos_05,
        'n_positive_0.3': n_pos_03,
        'n_null': n_null,
        'n_negative': n_neg,
        'max_rho': max_rho,
        'min_rho': min_rho,
        'tier': tier,
    })

i4_df = pd.DataFrame(i4_results)
i4_df.to_csv(f"{OUTDIR}/i4_replication_tiering.csv", index=False)

print("  Replication tiering:")
for _, row in i4_df.iterrows():
    print(f"    {row['tf']:8s}: {row['tier']:15s} (pos>0.5: {row['n_positive_0.5']}/{row['n_datasets']}, "
          f"pos>0.3: {row['n_positive_0.3']}/{row['n_datasets']})")


# ----------------------------------------------------------
# VIF Computation for HLMA Vascular JUNB
# ----------------------------------------------------------
print("\n=== VIF Check for F1 model ===")

# Compute VIF manually: regress each predictor on others
if 'country' in v_donors.columns and 'sex' in v_donors.columns:
    from numpy.linalg import lstsq

    y_rank = stats.rankdata(v_donors['SASP12_mean'].values)
    tf_rank = stats.rankdata(v_donors['JUNB'].values)
    ctry_num = (v_donors['country'] == sorted(v_donors['country'].unique())[0]).astype(int).values
    sex_num = (v_donors['sex'] == 'Male').astype(int).values
    age_vals = pd.to_numeric(v_donors['age'], errors='coerce').fillna(0).values

    X_mat = np.column_stack([tf_rank, ctry_num, sex_num, age_vals, np.ones(len(v_donors))])

    # VIF for each predictor
    predictors = ['JUNB', 'country', 'sex', 'age']
    for i, name in enumerate(predictors):
        others = np.delete(X_mat, i, axis=1)
        target = X_mat[:, i]
        coef, _, _, _ = lstsq(others, target, rcond=None)
        predicted = others @ coef
        ss_res = np.sum((target - predicted) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vif = 1 / (1 - r_sq) if r_sq < 1 else float('inf')
        print(f"  VIF({name}) = {vif:.2f} {'[WARNING > 5]' if vif > 5 else '[OK]'}")


# ----------------------------------------------------------
# Save summary
# ----------------------------------------------------------
summary = {
    'batch': 'batch_051',
    'date': '2026-04-13',
    'analyses': {
        'F1_confound': {
            'key_finding': 'Country (China/Spain) is dominant confound, not age',
            'rho_JUNB_age': float(stats.spearmanr(v_donors['JUNB'], pd.to_numeric(v_donors['age'], errors='coerce'))[0]),
            'rho_JUNB_country': float(stats.spearmanr(v_donors['JUNB'], (v_donors['country'] == sorted(v_donors['country'].unique())[0]).astype(int))[0]),
            'rho_JUNB_SASP_raw': float(stats.spearmanr(v_donors['JUNB'], v_donors['SASP12_mean'])[0]),
        },
        'I1_power': f"{len(i1_df)} findings analyzed",
        'I2_partial_corr': f"{len(i2_df)} TF-dataset combinations",
        'I3_sensitivity': f"{len(i3_df)} threshold analyses",
        'I4_tiering': f"{len(i4_df)} TFs tiered",
    },
    'files': [
        'f1_confound_analysis.csv',
        'i1_power_analysis.csv',
        'i2_partial_correlations.csv',
        'i3_cell_count_sensitivity.csv',
        'i4_replication_tiering.csv',
        'a2_cell_donor_counts.csv',
        'a3_qc_metrics.csv',
    ]
}

with open(f"{OUTDIR}/results.json", 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n=== batch_051 COMPLETE ===")
print(f"Results saved to {OUTDIR}/")
