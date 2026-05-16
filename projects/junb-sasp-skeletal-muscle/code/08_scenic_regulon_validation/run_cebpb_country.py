#!/usr/bin/env python3
"""
batch_055 C1: CEBPB Country Stratification (within-China vs within-Spain)
==========================================================================
PURPOSE: Document CEBPB country sensitivity alongside JUNB (reference) for all
3 HLMA compartments. The therapeutic thesis identifies CEBPB as a potential
secondary AP-1/SASP regulator alongside JUNB. This script quantifies whether
CEBPB's TF-SASP coupling is robust across countries or driven by China-only
demographics.

WHY CEBPB: CEBPB is a known SASP regulator (CCAAT/Enhancer Binding Protein Beta)
that co-regulates inflammatory genes with AP-1 factors. It is co-expressed with
JUNB in aged muscle (batch_041) and may represent a parallel therapeutic target.

WHY COUNTRY STRATIFICATION: The HLMA dataset has 2 countries (China=16-18 donors,
Spain=2-4 donors per compartment after min_cells filter). Country is a potential
confound because China donors were all collected by snRNA while Spain donors
were enriched for scRNA. Within-country correlations are the gold standard for
disentangling country effects from biology.

WHY INCLUDE JUNB: JUNB is the established reference (from batch_053 k1_klf10.py)
showing strong country sensitivity in some compartments. Including JUNB enables
side-by-side comparison: if CEBPB is more country-robust than JUNB, it may be
a more reliable therapeutic target.

METHODOLOGY (same as batch_053 run_k1_klf10.py):
- Donor-level mean CEBPB/JUNB and SASP12 composite (mean of 12 genes present).
- Spearman rank correlations throughout (robust to non-normality, N=15-21).
- Fisher Z for 95% CI.
- pingouin.partial_corr for formal partial correlations.
- Residualization-based partial correlation as verification.
- Minimum 50 cells per donor for inclusion.
- Bonferroni-adjusted alpha: 0.05 / (3 compartments × 2 countries) = 0.0083.

OUTPUT: cebpb_country_stratification.csv -- same schema as batch_053's
k1_klf10_country_sensitivity.csv for direct comparison.
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
import time

warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)

OUTDIR = "experiments/batch_055"

# Canonical SASP12 panel
SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8',
          'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

# Target TFs: CEBPB (primary) and JUNB (reference from batch_053)
TARGET_TFS = ['CEBPB', 'JUNB']

# Cell type filters (same as batch_053 and batch_054)
VASCULAR_TYPES = ['ArtEC', 'CapEC', 'VenEC', 'IL6+ VenEC']
FAP_TYPES = ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP', 'CD99+ FAP']

# Minimum cells per donor (same threshold as batch_053)
MIN_CELLS = 50

# Paths to 3 HLMA files (same as batch_053)
HLMA_FILES = {
    'Vascular': 'data/Vascular_scsn_RNA.h5ad',
    'MuSC': 'data/MuSC_scsn_RNA.h5ad',
    'FAP': 'data/OMIX004308-02.h5ad',
}


# ============================================================
# HELPER FUNCTIONS (same as batch_053, no execution side effects)
# ============================================================

def fisher_z_ci(rho, n, alpha=0.05):
    """95% CI for Spearman rho via Fisher Z transformation.

    WHY Fisher Z: Standard method for constructing CIs around correlation
    coefficients. Stabilizes variance across the range of rho values.
    """
    if n < 4 or abs(rho) >= 1.0:
        return np.nan, np.nan
    z_rho = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    z_lo = z_rho - z_crit * se
    z_hi = z_rho + z_crit * se
    ci_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
    ci_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
    return ci_lo, ci_hi


def timestamp():
    """Current time string for progress logging."""
    return time.strftime('%Y-%m-%d %H:%M:%S')


def load_and_compute_donors(path, cell_type_filter=None, filter_col='Annotation'):
    """Load h5ad (backed), filter cell types, compute donor-level means.

    WHY backed='r': This script only needs expression for a small set of genes
    (SASP12 + 2 TFs). Loading the full matrix into RAM is wasteful.
    We read only the required columns and close immediately.

    WHY donor-level: Country is a donor-level variable. Cell-level correlation
    with country would be pseudoreplicated. One aggregate per donor.

    Returns DataFrame with one row per donor.
    """
    print(f"  [{timestamp()}] Loading {path}...")
    adata = ad.read_h5ad(path, backed='r')
    obs = adata.obs.copy()
    var_names = list(adata.var_names)

    # Filter to specified cell types
    if cell_type_filter:
        mask = obs[filter_col].isin(cell_type_filter)
        obs = obs[mask]
    else:
        mask = pd.Series(True, index=obs.index)

    # Extract expression for filtered cells only
    X = adata[mask.values, :].X[:, :]
    if sp.issparse(X):
        X = X.toarray()

    adata.file.close()

    # Identify available genes
    sasp_present = [g for g in SASP12 if g in var_names]
    tfs_present = [g for g in TARGET_TFS if g in var_names]
    genes_needed = sasp_present + tfs_present

    # Standardize column names (same logic as batch_053)
    sample_col = 'sample' if 'sample' in obs.columns else 'donor_id'
    country_col = None
    for c in ['Country', 'country']:
        if c in obs.columns:
            country_col = c
            break
    age_col = None
    for a in ['age', 'age_pop']:
        if a in obs.columns:
            age_col = a
            break

    # Compute donor-level means
    donor_rows = []
    for donor in obs[sample_col].unique():
        dmask = obs[sample_col] == donor
        d_obs = obs[dmask]
        d_X = X[dmask.values]
        n_cells = dmask.sum()

        # Skip donors with too few cells
        if n_cells < MIN_CELLS:
            print(f"    [{timestamp()}] WARNING: Donor {donor} has {n_cells} cells "
                  f"< {MIN_CELLS} minimum, skipping")
            continue

        row = {
            'sample': donor,
            'n_cells': int(n_cells),
        }

        # Donor-level metadata from first cell
        if country_col:
            row['country'] = d_obs[country_col].iloc[0]
        if age_col:
            row['age'] = pd.to_numeric(d_obs[age_col].iloc[0], errors='coerce')
        if 'Sex' in d_obs.columns:
            row['sex'] = d_obs['Sex'].iloc[0]
        elif 'gender' in d_obs.columns:
            row['sex'] = d_obs['gender'].iloc[0]
        if 'tech' in d_obs.columns:
            row['tech'] = d_obs['tech'].iloc[0]

        # Gene means
        for g in genes_needed:
            idx = var_names.index(g)
            row[g] = float(np.mean(d_X[:, idx]))

        # SASP12 composite: mean of all detected SASP genes
        row['SASP12_mean'] = float(np.mean([row[g] for g in sasp_present]))
        row['SASP12_n_detected'] = len(sasp_present)

        donor_rows.append(row)

    df = pd.DataFrame(donor_rows)
    print(f"    [{timestamp()}] {len(df)} donors, {df['n_cells'].sum()} total cells "
          f"(min {MIN_CELLS} cells/donor)")
    if 'country' in df.columns:
        print(f"    Country: {df['country'].value_counts().to_dict()}")
    return df


def partial_corr_residualization(x, y, covars_list):
    """Compute Spearman partial correlation via residualization.

    WHY residualization: Standard approach for partial correlation when
    the covariate is binary (country) or continuous (age). We rank-transform
    first to approximate Spearman partial correlation.

    Args:
        x, y: arrays of the two variables
        covars_list: list of covariate arrays to residualize out
    Returns:
        (rho, p-value)
    """
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)
    n = len(x)

    # Build covariate matrix with intercept
    covar_mat = np.column_stack(covars_list + [np.ones(n)])

    # Residualize x
    from numpy.linalg import lstsq
    coef_x, _, _, _ = lstsq(covar_mat, x_rank, rcond=None)
    x_resid = x_rank - covar_mat @ coef_x

    # Residualize y
    coef_y, _, _, _ = lstsq(covar_mat, y_rank, rcond=None)
    y_resid = y_rank - covar_mat @ coef_y

    rho, p = stats.spearmanr(x_resid, y_resid)
    return float(rho), float(p)


# ============================================================
# MAIN ANALYSIS
# ============================================================

if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 60)
    print("batch_055 C1: CEBPB Country Stratification Analysis")
    print("=" * 60)
    print(f"[{timestamp()}] Output directory: {OUTDIR}")
    print(f"[{timestamp()}] Bonferroni note: alpha_adj = 0.05 / (3 compartments × 2 countries)")
    print(f"    = 0.0083")
    print()

    # --- Load and compute donor-level data for all 3 compartments ---
    print("--- Loading datasets ---")

    v_donors = load_and_compute_donors(
        HLMA_FILES['Vascular'],
        cell_type_filter=VASCULAR_TYPES,
        filter_col='Annotation'
    )

    m_donors = load_and_compute_donors(
        HLMA_FILES['MuSC'],
        cell_type_filter=None,
        filter_col='Annotation'
    )

    f_donors = load_and_compute_donors(
        HLMA_FILES['FAP'],
        cell_type_filter=FAP_TYPES,
        filter_col='Annotation'
    )

    # --- Compute correlations for each compartment ---
    print("\n--- Computing country-stratified correlations ---")

    results_rows = []

    for ds_name, donors_df in [
        ('Vascular', v_donors),
        ('MuSC', m_donors),
        ('FAP', f_donors),
    ]:
        print(f"\n{'='*60}")
        print(f"  [{timestamp()}] === {ds_name} (N={len(donors_df)} donors) ===")
        print(f"{'='*60}")

        # Create country numeric (1=China, 0=Spain)
        if 'country' in donors_df.columns:
            donors_df['country_num'] = (donors_df['country'] == 'China').astype(int)
        else:
            print(f"  WARNING: No country column in {ds_name}")
            continue

        has_country = donors_df['country'].nunique() > 1
        print(f"  Countries: {donors_df['country'].value_counts().to_dict()}")

        for tf in TARGET_TFS:
            if tf not in donors_df.columns:
                print(f"  [{timestamp()}] {tf}: not in dataset, skipping")
                continue

            tf_vals = donors_df[tf].values
            sasp_vals = donors_df['SASP12_mean'].values
            age_vals = donors_df['age'].values if 'age' in donors_df.columns else None
            ctry_vals = donors_df['country_num'].values
            n = len(donors_df)

            row = {
                'compartment': ds_name,
                'tf': tf,
                'n_donors': n,
            }

            # --- rho_total: total Spearman(TF, SASP12) ---
            rho_total, p_total = stats.spearmanr(tf_vals, sasp_vals)
            ci_lo, ci_hi = fisher_z_ci(rho_total, n)
            row['rho_total'] = rho_total
            row['p_total'] = p_total
            row['ci_95_total'] = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
            print(f"  [{timestamp()}] {tf}: rho_total={rho_total:.4f} (p={p_total:.2e}, N={n})")

            # --- rho(TF, country) and rho(TF, age) ---
            rho_tf_country, p_tf_country = stats.spearmanr(tf_vals, ctry_vals)
            row['rho_tf_country'] = rho_tf_country
            row['p_tf_country'] = p_tf_country
            print(f"    rho({tf},country)={rho_tf_country:.4f} (p={p_tf_country:.2e})")

            if age_vals is not None:
                rho_tf_age, p_tf_age = stats.spearmanr(tf_vals, age_vals)
                row['rho_tf_age'] = rho_tf_age
                row['p_tf_age'] = p_tf_age
                print(f"    rho({tf},age)={rho_tf_age:.4f} (p={p_tf_age:.2e})")

            # --- Country-adjusted partial correlation ---
            if has_country:
                try:
                    import pingouin as pg
                    df_pg = pd.DataFrame({
                        'tf': tf_vals,
                        'sasp': sasp_vals,
                        'country': ctry_vals.astype(float),
                    })
                    pc = pg.partial_corr(data=df_pg, x='tf', y='sasp',
                                         covar=['country'], method='spearman')
                    row['rho_ctry_adj_pg'] = float(pc['r'].values[0])
                    # pingouin uses 'p_val' column name
                    p_col = 'p_val' if 'p_val' in pc.columns else 'p-val'
                    row['p_ctry_adj_pg'] = float(pc[p_col].values[0])
                    print(f"    ctry_adj(pg)={row['rho_ctry_adj_pg']:.4f} "
                          f"(p={row['p_ctry_adj_pg']:.2e})")
                except Exception as e:
                    print(f"    WARNING: pingouin partial_corr failed: {e}")
                    row['rho_ctry_adj_pg'] = np.nan
                    row['p_ctry_adj_pg'] = np.nan

                # Residualization cross-check
                rho_resid, p_resid = partial_corr_residualization(
                    tf_vals, sasp_vals, [ctry_vals.astype(float)]
                )
                row['rho_ctry_adj_resid'] = rho_resid
                row['p_ctry_adj_resid'] = p_resid
                print(f"    ctry_adj(resid)={rho_resid:.4f} (p={p_resid:.2e})")
            else:
                row['rho_ctry_adj_pg'] = np.nan
                row['p_ctry_adj_pg'] = np.nan
                row['rho_ctry_adj_resid'] = np.nan
                row['p_ctry_adj_resid'] = np.nan

            # --- Within-country correlations ---
            for country_name in ['China', 'Spain']:
                sub = donors_df[donors_df['country'] == country_name]
                if len(sub) >= 4:
                    r_wc, p_wc = stats.spearmanr(
                        sub[tf].values, sub['SASP12_mean'].values
                    )
                    ci_lo_wc, ci_hi_wc = fisher_z_ci(r_wc, len(sub))
                    row[f'within_{country_name}_rho'] = r_wc
                    row[f'within_{country_name}_p'] = p_wc
                    row[f'within_{country_name}_n'] = len(sub)
                    row[f'within_{country_name}_ci95'] = f"[{ci_lo_wc:.3f}, {ci_hi_wc:.3f}]"
                    print(f"    within_{country_name}: rho={r_wc:.4f} "
                          f"(p={p_wc:.4f}, N={len(sub)}, "
                          f"CI95=[{ci_lo_wc:.3f}, {ci_hi_wc:.3f}])")
                else:
                    row[f'within_{country_name}_rho'] = np.nan
                    row[f'within_{country_name}_p'] = np.nan
                    row[f'within_{country_name}_n'] = len(sub)
                    row[f'within_{country_name}_ci95'] = 'N/A'
                    print(f"    within_{country_name}: N={len(sub)} < 4, not computed")

            # --- rho(TF, age) controlling for country ---
            if age_vals is not None and has_country:
                try:
                    import pingouin as pg
                    df_pg2 = pd.DataFrame({
                        'tf': tf_vals,
                        'age': age_vals.astype(float),
                        'country': ctry_vals.astype(float),
                    })
                    pc2 = pg.partial_corr(data=df_pg2, x='tf', y='age',
                                          covar=['country'], method='spearman')
                    row['rho_tf_age_ctry_adj'] = float(pc2['r'].values[0])
                    p_col2 = 'p_val' if 'p_val' in pc2.columns else 'p-val'
                    row['p_tf_age_ctry_adj'] = float(pc2[p_col2].values[0])
                except Exception as e:
                    row['rho_tf_age_ctry_adj'] = np.nan
                    row['p_tf_age_ctry_adj'] = np.nan

                # Residualization cross-check
                rho_age_resid, p_age_resid = partial_corr_residualization(
                    tf_vals, age_vals, [ctry_vals.astype(float)]
                )
                row['rho_tf_age_ctry_adj_resid'] = rho_age_resid
                row['p_tf_age_ctry_adj_resid'] = p_age_resid

            results_rows.append(row)

        # Print compartment summary
        print(f"\n  [{timestamp()}] {ds_name} compartment summary:")
        sub_df = pd.DataFrame(results_rows)
        sub_df = sub_df[sub_df['compartment'] == ds_name]
        for _, row in sub_df.iterrows():
            print(f"    {row['tf']}: rho_total={row['rho_total']:.3f}, "
                  f"rho_ctry_adj={row.get('rho_ctry_adj_pg', np.nan):.3f}, "
                  f"within_China={row.get('within_China_rho', np.nan):.3f} "
                  f"(N={int(row.get('within_China_n', 0))}), "
                  f"within_Spain={row.get('within_Spain_rho', np.nan):.3f} "
                  f"(N={int(row.get('within_Spain_n', 0))})")

    # --- Build output table ---
    print("\n--- Saving results ---")
    results_df = pd.DataFrame(results_rows)

    # Column order matching batch_053 k1_klf10_country_sensitivity.csv
    col_order = [
        'compartment', 'tf', 'n_donors',
        'rho_total', 'p_total', 'ci_95_total',
        'rho_tf_country', 'p_tf_country',
        'rho_tf_age', 'p_tf_age',
        'rho_ctry_adj_pg', 'p_ctry_adj_pg',
        'rho_ctry_adj_resid', 'p_ctry_adj_resid',
        'within_China_rho', 'within_China_p', 'within_China_n', 'within_China_ci95',
        'within_Spain_rho', 'within_Spain_p', 'within_Spain_n', 'within_Spain_ci95',
        'rho_tf_age_ctry_adj', 'p_tf_age_ctry_adj',
        'rho_tf_age_ctry_adj_resid', 'p_tf_age_ctry_adj_resid',
    ]
    # Keep only columns that exist
    col_order = [c for c in col_order if c in results_df.columns]
    results_df = results_df[col_order]

    results_path = f"{OUTDIR}/cebpb_country_stratification.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[{timestamp()}] Saved: {results_path}")
    print(f"  Shape: {results_df.shape}")
    print(f"  Columns: {list(results_df.columns)}")

    # --- Print side-by-side comparison (CEBPB vs JUNB) ---
    print("\n--- CEBPB vs JUNB Side-by-Side Comparison ---")
    print(f"{'Compartment':12s} {'TF':8s} {'rho_total':>10s} {'ctry_adj':>10s} "
          f"{'wChina':>8s} {'wSpain':>8s} {'n_d':>5s}")
    print(f"{'-'*65}")

    for ds_name in ['Vascular', 'MuSC', 'FAP']:
        for tf in TARGET_TFS:
            sub = results_df[(results_df['compartment'] == ds_name) & (results_df['tf'] == tf)]
            if len(sub) > 0:
                row = sub.iloc[0]
                rho = f"{row['rho_total']:.3f}"
                ctry = f"{row.get('rho_ctry_adj_pg', np.nan):.3f}" if pd.notna(row.get('rho_ctry_adj_pg')) else 'N/A'
                wc = f"{row.get('within_China_rho', np.nan):.3f}" if pd.notna(row.get('within_China_rho')) else 'N/A'
                ws = f"{row.get('within_Spain_rho', np.nan):.3f}" if pd.notna(row.get('within_Spain_rho')) else 'N/A'
                n = int(row['n_donors'])
                print(f"{ds_name:12s} {tf:8s} {rho:>10s} {ctry:>10s} "
                      f"{wc:>8s} {ws:>8s} {n:>5d}")

    # --- Significance interpretation ---
    print("\n--- Significance (Bonferroni alpha_adj = 0.0083) ---")
    bonf_alpha = 0.05 / (3 * 2)  # 3 compartments × 2 countries
    print(f"  Bonferroni-adjusted alpha = 0.05 / (3 × 2) = {bonf_alpha:.4f}")
    print()
    for ds_name in ['Vascular', 'MuSC', 'FAP']:
        print(f"  {ds_name}:")
        for tf in TARGET_TFS:
            sub = results_df[(results_df['compartment'] == ds_name) & (results_df['tf'] == tf)]
            if len(sub) > 0:
                row = sub.iloc[0]
                p_total = row.get('p_total', np.nan)
                p_ctry = row.get('p_ctry_adj_pg', np.nan)
                p_china = row.get('within_China_p', np.nan)
                p_spain = row.get('within_Spain_p', np.nan)

                sig_total = p_total < bonf_alpha if pd.notna(p_total) else False
                sig_ctry = p_ctry < bonf_alpha if pd.notna(p_ctry) else False
                sig_china = p_china < bonf_alpha if pd.notna(p_china) else False
                sig_spain = p_spain < bonf_alpha if pd.notna(p_spain) else False

                total_str = f"{'**' if sig_total else ''}p={p_total:.2e}" if pd.notna(p_total) else 'N/A'
                ctry_str = f"{'**' if sig_ctry else ''}p={p_ctry:.2e}" if pd.notna(p_ctry) else 'N/A'
                china_str = f"{'**' if sig_china else ''}p={p_china:.2e}" if pd.notna(p_china) else 'N/A'
                spain_str = f"{'**' if sig_spain else ''}p={p_spain:.2e}" if pd.notna(p_spain) else 'N/A'

                print(f"    {tf}: total={total_str}, ctry_adj={ctry_str}, "
                      f"China={china_str}, Spain={spain_str}")

    # --- Save summary JSON ---
    summary = {
        'batch': 'batch_055_c1',
        'script': 'run_cebpb_country.py',
        'date': pd.Timestamp.now().isoformat(),
        'analysis': 'CEBPB country stratification (within-China vs within-Spain)',
        'target_tfs': TARGET_TFS,
        'sasp12_genes': SASP12,
        'bonferroni_alpha': float(bonf_alpha),
        'bonferroni_note': 'alpha_adj = 0.05 / (3 compartments × 2 countries) = 0.0083',
        'methodology': {
            'correlation': 'Spearman rank',
            'ci_method': 'Fisher Z transformation',
            'partial_corr': 'pingouin.partial_corr (Spearman) + residualization cross-check',
            'min_cells_per_donor': MIN_CELLS,
            'cell_type_filters': {
                'Vascular': VASCULAR_TYPES,
                'FAP': FAP_TYPES,
                'MuSC': 'all cells',
            },
        },
        'data_sources': HLMA_FILES,
        'caveats': [
            'Spain N is very small after min 50 cells filter (2-4 donors/compartment)',
            'Spain donors may be from a different surgical context than China donors',
            'China is almost entirely snRNA while Spain has scRNA enrichment',
            'CEBPB is a broad transcriptional activator -- positive SASP correlation may '
            'reflect co-activation rather than direct regulatory targeting',
        ],
    }

    summary_path = f"{OUTDIR}/c1_cebpb_country_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[{timestamp()}] Summary saved: {summary_path}")

    print(f"\n=== batch_055 C1 COMPLETE ===")
