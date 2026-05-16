#!/usr/bin/env python3
"""
batch_066: T1 CDKN1A vs JUNB signal decomposition + T2 sex-stratified + T3 heterogeneity

Implements partial correlation signal decomposition to test whether CDKN1A
has independent SASP-predictive signal beyond JUNB, and characterizes donor
heterogeneity in vascular JUNB-SASP coupling.
"""

import json
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import inv

warnings.filterwarnings('ignore')

SASP12 = ['CXCL8', 'CCL2', 'CXCL1', 'IL6', 'CXCL2', 'CCL20',
          'SERPINE1', 'MMP1', 'MMP3', 'MMP10', 'PLAU', 'TNFAIP6']


def _numpy_to_python(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_python(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return _numpy_to_python(obj.tolist())
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (bool,)):
        return bool(obj)
    elif isinstance(obj, (int, float)):
        return obj
    elif obj is None:
        return None
    else:
        return obj

def load_hlma_vascular():
    """Load HLMA vascular data and compute per-donor aggregates."""
    adata = sc.read_h5ad('data/Vascular_scsn_RNA.h5ad', backed='r')
    print(f"Vascular loaded: {adata.shape}")

    # Get donor-level aggregates
    samples = adata.obs['sample'].unique()
    n_donors = len(samples)
    print(f"Donors: {n_donors}")

    # Check what genes are available
    genes = adata.var_names.tolist()
    missing = [g for g in SASP12 if g not in genes]
    if missing:
        print(f"SASP12 missing: {missing}")

    missing_cdkn1a = 'CDKN1A' not in genes
    missing_junb = 'JUNB' not in genes
    print(f"CDKN1A: {'YES' if not missing_cdkn1a else 'NO'}, JUNB: {'YES' if not missing_junb else 'NO'}")

    # Compute per-donor means (sample bulk aggregation)
    results = []
    for sample in samples:
        mask = adata.obs['sample'] == sample
        X = adata[mask].X[:]
        if hasattr(X, 'toarray'):
            X = X.toarray()

        row_mean = X.mean(axis=0).flatten()

        # SASP12 mean
        sasp_idx = [genes.index(g) for g in SASP12 if g in genes]
        sasp_mean = row_mean[sasp_idx].mean() if sasp_idx else np.nan

        cdkn1a_expr = row_mean[genes.index('CDKN1A')] if not missing_cdkn1a else np.nan
        junb_expr = row_mean[genes.index('JUNB')] if not missing_junb else np.nan

        # Get metadata
        obs_slice = adata.obs[mask].iloc[0]

        obs_slice = adata.obs[mask].iloc[0]

        # Handle sex column: Vascular uses 'Sex', MuSC uses 'gender'
        sex_val = obs_slice.get('Sex', obs_slice.get('gender', 'Unknown'))

        results.append({
            'sample': sample,
            'SASP12_mean': sasp_mean,
            'CDKN1A': cdkn1a_expr,
            'JUNB': junb_expr,
            'age': obs_slice['age'],
            'Sex': sex_val,
            'tech': obs_slice['tech'],
            'Country': obs_slice['Country'],
            'n_cells': mask.sum()
        })

    return pd.DataFrame(results)


def load_hlma_musc():
    """Load HLMA MuSC data and compute per-donor aggregates."""
    adata = sc.read_h5ad('data/MuSC_scsn_RNA.h5ad', backed='r')
    print(f"MuSC loaded: {adata.shape}")

    samples = adata.obs['sample'].unique()
    n_donors = len(samples)
    print(f"Donors: {n_donors}")

    genes = adata.var_names.tolist()
    missing = [g for g in SASP12 if g not in genes]
    if missing:
        print(f"SASP12 missing in MuSC: {missing}")

    results = []
    for sample in samples:
        mask = adata.obs['sample'] == sample
        X = adata[mask].X[:]
        if hasattr(X, 'toarray'):
            X = X.toarray()

        row_mean = X.mean(axis=0).flatten()

        sasp_idx = [genes.index(g) for g in SASP12 if g in genes]
        sasp_mean = row_mean[sasp_idx].mean() if sasp_idx else np.nan

        cdkn1a_expr = row_mean[genes.index('CDKN1A')] if 'CDKN1A' in genes else np.nan
        junb_expr = row_mean[genes.index('JUNB')] if 'JUNB' in genes else np.nan

        obs_slice = adata.obs[mask].iloc[0]
        sex_val = obs_slice.get('Sex', obs_slice.get('gender', 'Unknown'))

        results.append({
            'sample': sample,
            'SASP12_mean': sasp_mean,
            'CDKN1A': cdkn1a_expr,
            'JUNB': junb_expr,
            'age': obs_slice['age'],
            'Sex': sex_val,
            'tech': obs_slice['tech'],
            'Country': obs_slice['Country'],
            'n_cells': mask.sum()
        })

    return pd.DataFrame(results)


def partial_correlation(x, y, z, method='spearman'):
    """
    Compute partial correlation between x and y controlling for z.
    Uses residualization approach for Spearman.
    """
    if method == 'spearman':
        # Residualize x and y against z
        def resid(a, b):
            # Fit linear regression: a ~ b
            b_centered = b - b.mean()
            a_centered = a - a.mean()
            # Use Pearson for regression
            slope = np.dot(b_centered, a_centered) / (np.dot(b_centered, b_centered) + 1e-10)
            resid = a - a.mean() - slope * (b - b.mean())
            return resid

        x_resid = resid(x, z)
        y_resid = resid(y, z)

        # Spearman partial correlation = Pearson of residuals
        rho, p = spearmanr(x_resid, y_resid)
        return rho, p

    else:  # pearson
        # Partial correlation formula
        r_xy = pearsonr(x, y)[0]
        r_xz = pearsonr(x, z)[0]
        r_yz = pearsonr(y, z)[0]

        num = r_xy - r_xz * r_yz
        den = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

        if den < 1e-10:
            return 0.0, 1.0

        partial_r = num / den

        # Fisher-z transform for p-value
        n = len(x)
        se = 1.0 / np.sqrt(n - 3)
        z_stat = np.arctanh(partial_r) / se
        p_val = 2 * stats.norm.cdf(-abs(z_stat))

        return partial_r, p_val


def steiger_dependent_z(r12, r13, r23, n):
    """
    Steiger (1980) dependent-correlation z-test.
    Tests if r12 differs from r13 given their shared variable 3.
    r12 = correlation between variable 1 and 2 (e.g., CDKN1A-SASP)
    r13 = correlation between variable 1 and 3 (e.g., JUNB-SASP)
    r23 = correlation between variable 2 and 3 (e.g., CDKN1A-JUNB)
    n = sample size
    """
    # Determinant of correlation matrix
    det = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23

    if det <= 0:
        return np.nan, np.nan, "near-singular matrix"

    # Standard error
    se = np.sqrt((1 - r23**2 + (r12 - r13)**2 / 2)**2 / ((n - 3) * det))

    if np.isnan(se) or se == 0:
        return np.nan, np.nan, "SE undefined"

    # z-statistic
    z = (r12 - r13) / se
    p = 2 * stats.norm.cdf(-abs(z))

    return z, p, "ok"


def bootstrap_partial_rho(x, y, z, n_bootstrap=1000, ci=0.95):
    """Bootstrap CI for partial correlation."""
    n = len(x)
    boots = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            rho, _ = partial_correlation(x[idx], y[idx], z[idx], method='spearman')
            boots.append(rho)
        except:
            pass

    boots = np.array(boots)
    boots = boots[~np.isnan(boots)]

    if len(boots) < 100:
        return np.nan, np.nan, np.nan

    alpha = 1 - ci
    lower = np.percentile(boots, 100 * alpha / 2)
    upper = np.percentile(boots, 100 * (1 - alpha / 2))
    median = np.median(boots)

    return median, lower, upper


def vif(X):
    """Compute variance inflation factors for columns of X."""
    n_cols = X.shape[1]
    vifs = []
    for i in range(n_cols):
        # Regress column i on all others
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        X_others = np.column_stack([np.ones(len(y)), X_others])

        try:
            beta = np.linalg.lstsq(X_others, y, rcond=None)[0]
            resid = y - X_others @ beta
            rss = np.sum(resid**2)
            tss = np.sum((y - y.mean())**2)
            r2 = 1 - rss / (tss + 1e-10)
            vif_val = 1 / (1 - r2 + 1e-10)
        except:
            vif_val = np.nan
        vifs.append(vif_val)

    return np.array(vifs)


def bh_fdr(p_values):
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    p_arr = np.array(p_values)
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]

    # BH threshold
    bh_threshold = np.arange(1, n + 1) / n * 0.05

    # Find largest k where p[k] <= threshold[k]
    below = sorted_p <= bh_threshold
    if not np.any(below):
        reject = np.zeros(n, dtype=bool)
        ranked_q = np.zeros(n)
    else:
        k_max = np.where(below)[0][-1]
        reject = sorted_idx[:k_max + 1]
        reject = np.isin(np.arange(n), reject)

        # Compute q-values
        ranked_q = np.zeros(n)
        for i in range(n):
            orig_idx = sorted_idx[i]
            ranked_q[orig_idx] = sorted_p[i] * n / (i + 1)

    return reject, p_arr[ np.argsort(p_arr) ]  # sorted p-values


def main():
    print("=" * 60)
    print("batch_066: T1-T3 Implementation")
    print("=" * 60)

    # Load data
    print("\nLoading HLMA vascular...")
    vasc_df = load_hlma_vascular()
    print(f"  N = {len(vasc_df)} donors")

    print("\nLoading HLMA MuSC...")
    musc_df = load_hlma_musc()
    print(f"  N = {len(musc_df)} donors")

    # =========================================================================
    # T1: CDKN1A vs JUNB Signal Decomposition
    # =========================================================================
    print("\n" + "=" * 60)
    print("T1: Signal Decomposition (Vascular + MuSC)")
    print("=" * 60)

    t1_results = {}

    for comp_name, df in [('Vascular', vasc_df), ('MuSC', musc_df)]:
        print(f"\n--- {comp_name} ---")

        n = len(df)

        # Extract vectors
        sasp = df['SASP12_mean'].values
        cdkn1a = df['CDKN1A'].values
        junb = df['JUNB'].values

        # 1. Bivariate Spearman correlations
        r_cdk_sasp, p_cdk = spearmanr(cdkn1a, sasp)
        r_junb_sasp, p_junb = spearmanr(junb, sasp)
        r_cdk_junb, p_junb_cdk = spearmanr(cdkn1a, junb)

        print(f"  rho(CDKN1A, SASP) = {r_cdk_sasp:.4f}, p = {p_cdk:.4f}")
        print(f"  rho(JUNB, SASP)   = {r_junb_sasp:.4f}, p = {p_junb:.4f}")
        print(f"  rho(CDKN1A, JUNB) = {r_cdk_junb:.4f}, p = {p_junb_cdk:.4f}")

        # 2. Partial correlations (PRIMARY test)
        partial_cdk_given_junb, p_partial_cdk = partial_correlation(cdkn1a, sasp, junb, method='spearman')
        partial_junb_given_cdk, p_partial_junb = partial_correlation(junb, sasp, cdkn1a, method='spearman')

        print(f"  partial_rho(CDKN1A, SASP | JUNB) = {partial_cdk_given_junb:.4f}, p = {p_partial_cdk:.4f}")
        print(f"  partial_rho(JUNB, SASP | CDKN1A) = {partial_junb_given_cdk:.4f}, p = {p_partial_junb:.4f}")

        # 3. Bootstrap CIs for partial correlations
        print(f"  Computing bootstrap CIs (1000 iterations)...")
        np.random.seed(42)
        med_cdk, lo_cdk, hi_cdk = bootstrap_partial_rho(cdkn1a, sasp, junb, n_bootstrap=1000)
        np.random.seed(42)
        med_junb, lo_junb, hi_junb = bootstrap_partial_rho(junb, sasp, cdkn1a, n_bootstrap=1000)

        print(f"  CDKN1A|SASP|JUNB bootstrap CI: [{lo_cdk:.3f}, {hi_cdk:.3f}]")
        print(f"  JUNB|SASP|CDKN1A bootstrap CI: [{lo_junb:.3f}, {hi_junb:.3f}]")

        # 4. Steiger dependent-correlation z-test (SECONDARY test)
        z_steiger, p_steiger, msg = steiger_dependent_z(r_cdk_sasp, r_junb_sasp, r_cdk_junb, n)
        print(f"  Steiger z = {z_steiger:.4f}, p = {p_steiger:.4f} ({msg})")

        # 5. VIF (multicollinearity)
        X = np.column_stack([cdkn1a, junb])
        vifs = vif(X)
        print(f"  VIF: CDKN1A={vifs[0]:.1f}, JUNB={vifs[1]:.1f}")

        # 6. Decision rule
        # PRIMARY: partial correlations with bootstrap CI
        # BH-FDR for 4 tests (2 partial corr per compartment)
        p_partials = [p_partial_cdk, p_partial_junb]
        reject_fdr, sorted_p = bh_fdr(p_partials)
        q_cdk = sorted_p[0]  # smallest p-value gets smallest q
        q_junb = sorted_p[1]

        # Decision
        if abs(lo_cdk) > 0.15 and abs(hi_cdk) > 0.15:  # CI excludes ~0
            decision = "CDKN1A has independent signal"
        elif abs(lo_junb) > 0.15 and abs(hi_junb) > 0.15:
            decision = "JUNB has independent signal"
        elif abs(lo_cdk) < 0.1 and abs(lo_junb) < 0.1:
            decision = "Both signal fully mediated (co-predictors)"
        else:
            decision = "Indeterminate"

        print(f"  DECISION: {decision}")

        t1_results[comp_name] = {
            'n': n,
            'bivariate': {
                'CDKN1A_SASP': {'rho': r_cdk_sasp, 'p': p_cdk},
                'JUNB_SASP': {'rho': r_junb_sasp, 'p': p_junb},
                'CDKN1A_JUNB': {'rho': r_cdk_junb, 'p': p_junb_cdk}
            },
            'partial_correlations': {
                'CDKN1A_SASP_given_JUNB': {
                    'rho': partial_cdk_given_junb,
                    'p': p_partial_cdk,
                    'bootstrap_CI': [lo_cdk, hi_cdk]
                },
                'JUNB_SASP_given_CDKN1A': {
                    'rho': partial_junb_given_cdk,
                    'p': p_partial_junb,
                    'bootstrap_CI': [lo_junb, hi_junb]
                }
            },
            'steiger': {'z': z_steiger, 'p': p_steiger},
            'vif': {'CDKN1A': vifs[0], 'JUNB': vifs[1]},
            'decision': decision
        }

    # =========================================================================
    # T2: Sex-stratified descriptive
    # =========================================================================
    print("\n" + "=" * 60)
    print("T2: Sex-stratified Descriptive (Vascular)")
    print("=" * 60)

    t2_results = {}

    for sex in vasc_df['Sex'].unique():
        subset = vasc_df[vasc_df['Sex'] == sex]
        n_sex = len(subset)

        sasp = subset['SASP12_mean'].values
        junb = subset['JUNB'].values
        cdkn1a = subset['CDKN1A'].values

        rho_j, p_j = spearmanr(junb, sasp)
        rho_c, p_c = spearmanr(cdkn1a, sasp)

        # Fisher-z CI
        def fisher_ci(rho, n, alpha=0.05):
            z = np.arctanh(rho)
            se = 1.0 / np.sqrt(n - 3)
            z_lo = z - stats.norm.ppf(1 - alpha/2) * se
            z_hi = z + stats.norm.ppf(1 - alpha/2) * se
            return np.tanh(z_lo), np.tanh(z_hi)

        ci_j = fisher_ci(rho_j, n_sex)
        ci_c = fisher_ci(rho_c, n_sex)

        print(f"  {sex}: N={n_sex}")
        print(f"    JUNB-SASP: rho={rho_j:.3f} [CI: {ci_j[0]:.3f}, {ci_j[1]:.3f}]")
        print(f"    CDKN1A-SASP: rho={rho_c:.3f} [CI: {ci_c[0]:.3f}, {ci_c[1]:.3f}]")

        t2_results[sex] = {
            'n': n_sex,
            'JUNB': {'rho': rho_j, 'p': p_j, 'CI_95': ci_j},
            'CDKN1A': {'rho': rho_c, 'p': p_c, 'CI_95': ci_c}
        }

    # Check overlap
    sexes = list(t2_results.keys())
    if len(sexes) == 2:
        overlap_junb = max(t2_results[sexes[0]]['JUNB']['CI_95'][0],
                           t2_results[sexes[1]]['JUNB']['CI_95'][0]) <= \
                       min(t2_results[sexes[0]]['JUNB']['CI_95'][1],
                           t2_results[sexes[1]]['JUNB']['CI_95'][1])
        t2_results['CI_overlap_JUNB'] = overlap_junb
        print(f"  JUNB CI overlap: {overlap_junb}")

    # =========================================================================
    # T3: Donor heterogeneity
    # =========================================================================
    print("\n" + "=" * 60)
    print("T3: Donor Heterogeneity (Vascular)")
    print("=" * 60)

    import statsmodels.api as sm

    # Prepare design matrix
    vasc_df['tech_binary'] = (vasc_df['tech'] == vasc_df['tech'].unique()[0]).astype(int)
    vasc_df['sex_binary'] = (vasc_df['Sex'] == vasc_df['Sex'].unique()[0]).astype(int)

    # Country encoding
    countries = vasc_df['Country'].unique()
    if len(countries) == 2:
        vasc_df['country_binary'] = (vasc_df['Country'] == countries[0]).astype(int)
    else:
        # One-hot encode if >2
        for c in countries[1:]:
            vasc_df[f'country_{c}'] = (vasc_df['Country'] == c).astype(int)

    # Prepare X and y
    predictor_vars = ['JUNB']
    covars = ['age', 'sex_binary', 'tech_binary']
    if 'country_binary' in vasc_df.columns:
        covars.append('country_binary')
    else:
        covars += [c for c in vasc_df.columns if c.startswith('country_')]

    all_vars = predictor_vars + covars

    X = sm.add_constant(vasc_df[all_vars].values)
    y = vasc_df['SASP12_mean'].values

    # OLS regression
    model = sm.OLS(y, X)
    results = model.fit()

    print(f"\nOLS Regression: SASP12 ~ JUNB + covariates")
    print(f"  R-squared: {results.rsquared:.4f}")
    print(f"  N: {len(y)}, df_residual: {results.df_resid}")

    # Residuals and influence
    residuals = results.resid
    standardized = residuals / results.scale

    # Studentized residuals (use internal variant for speed)
    infl = results.get_influence()
    stud_resid = infl.resid_studentized_internal

    # Cook's distance
    cooks_d = infl.cooks_distance[0]

    # DFBETAS
    dfbetas = infl.dfbetas

    # Flag outliers (threshold 3.0 per brief)
    threshold = 3.0
    outlier_mask = np.abs(stud_resid) > threshold

    print(f"\nOutliers (|studentized residual| > {threshold}):")
    if not np.any(outlier_mask):
        print("  None")
    else:
        for i, (is_outlier, row) in enumerate(zip(outlier_mask, vasc_df.itertuples())):
            if is_outlier:
                print(f"  {row.sample}: stud_resid={stud_resid[i]:.2f}, cooks_d={cooks_d[i]:.3f}")

                # Check DFBETAS
                for j, var in enumerate(all_vars):
                    if abs(dfbetas[i, j+1]) > 0.5:  # +1 for intercept
                        print(f"    DFBETAS for {var}: {dfbetas[i, j+1]:.3f}")

    n_outliers = np.sum(outlier_mask)

    t3_results = {
        'r_squared': results.rsquared,
        'n': len(y),
        'df_residual': results.df_resid,
        'n_outliers': int(n_outliers),
        'outlier_threshold': threshold,
        'max_studentized_resid': float(np.max(np.abs(stud_resid))),
        'max_cooks_d': float(np.max(cooks_d)),
        'coefficients': {
            'intercept': float(results.params[0]),
            'JUNB': float(results.params[1]),
            'age': float(results.params[2]) if len(results.params) > 2 else np.nan,
            'sex': float(results.params[3]) if len(results.params) > 3 else np.nan,
            'tech': float(results.params[4]) if len(results.params) > 4 else np.nan
        },
        'donor_residuals': {
            'sample': vasc_df['sample'].tolist(),
            'studentized_resid': stud_resid.tolist(),
            'cooks_d': cooks_d.tolist()
        }
    }

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    results_out = {
        't1': t1_results,
        't2': t2_results,
        't3': t3_results,
        'metadata': {
            'vascular_n': len(vasc_df),
            'musc_n': len(musc_df)
        }
    }

    with open('experiments/batch_066/results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json.dump(_numpy_to_python(results_out), f, indent=2)

    # Save CSV summaries
    t1_df = pd.DataFrame([
        {
            'compartment': comp,
            'rho_CDKN1A_SASP': r['bivariate']['CDKN1A_SASP']['rho'],
            'p_CDKN1A_SASP': r['bivariate']['CDKN1A_SASP']['p'],
            'rho_JUNB_SASP': r['bivariate']['JUNB_SASP']['rho'],
            'p_JUNB_SASP': r['bivariate']['JUNB_SASP']['p'],
            'rho_CDKN1A_JUNB': r['bivariate']['CDKN1A_JUNB']['rho'],
            'partial_CDKN1A_SASP_given_JUNB': r['partial_correlations']['CDKN1A_SASP_given_JUNB']['rho'],
            'p_partial_CDKN1A': r['partial_correlations']['CDKN1A_SASP_given_JUNB']['p'],
            'CI_lower_CDKN1A': r['partial_correlations']['CDKN1A_SASP_given_JUNB']['bootstrap_CI'][0],
            'CI_upper_CDKN1A': r['partial_correlations']['CDKN1A_SASP_given_JUNB']['bootstrap_CI'][1],
            'partial_JUNB_SASP_given_CDKN1A': r['partial_correlations']['JUNB_SASP_given_CDKN1A']['rho'],
            'p_partial_JUNB': r['partial_correlations']['JUNB_SASP_given_CDKN1A']['p'],
            'CI_lower_JUNB': r['partial_correlations']['JUNB_SASP_given_CDKN1A']['bootstrap_CI'][0],
            'CI_upper_JUNB': r['partial_correlations']['JUNB_SASP_given_CDKN1A']['bootstrap_CI'][1],
            'steiger_z': r['steiger']['z'],
            'steiger_p': r['steiger']['p'],
            'vif_CDKN1A': r['vif']['CDKN1A'],
            'vif_JUNB': r['vif']['JUNB'],
            'decision': r['decision']
        }
        for comp, r in t1_results.items()
    ])
    t1_df.to_csv('experiments/batch_066/t1_signal_decomposition.csv', index=False)

    t2_df = pd.DataFrame([
        {'sex': sex, 'n': r['n'],
         'JUNB_rho': r['JUNB']['rho'], 'JUNB_p': r['JUNB']['p'],
         'JUNB_CI_lo': r['JUNB']['CI_95'][0], 'JUNB_CI_hi': r['JUNB']['CI_95'][1],
         'CDKN1A_rho': r['CDKN1A']['rho'], 'CDKN1A_p': r['CDKN1A']['p'],
         'CDKN1A_CI_lo': r['CDKN1A']['CI_95'][0], 'CDKN1A_CI_hi': r['CDKN1A']['CI_95'][1]}
        for sex, r in t2_results.items() if sex not in ['CI_overlap_JUNB']
    ])
    if 'CI_overlap_JUNB' in t2_results:
        t2_df['CI_overlap_JUNB'] = t2_results['CI_overlap_JUNB']
    t2_df.to_csv('experiments/batch_066/t2_sex_stratified.csv', index=False)

    t3_df = pd.DataFrame({
        'sample': t3_results['donor_residuals']['sample'],
        'studentized_resid': t3_results['donor_residuals']['studentized_resid'],
        'cooks_d': t3_results['donor_residuals']['cooks_d']
    })
    t3_df.to_csv('experiments/batch_066/t3_heterogeneity.csv', index=False)

    print("Done! Results saved to experiments/batch_066/")

    return results_out


if __name__ == '__main__':
    results = main()