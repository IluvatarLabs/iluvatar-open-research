#!/usr/bin/env python3
"""
batch_052 B1 (Pseudobulk DE) + B2 (GSEA) Analysis

PURPOSE:
  - B1: Pseudobulk differential expression (young vs old) per compartment,
        controlling for country confound.
  - B2: Gene set enrichment analysis on ranked DE results.

CRITICAL DATA CAVEAT:
  The HLMA h5ad files contain LOG-NORMALIZED expression (not raw UMI counts).
  X values range 0-~5 (consistent with log1p of CPM or SCTransform-corrected values).
  nCount_RNA is fractional, confirming normalization has already been applied.
  Raw counts are NOT preserved in any layer or raw slot.

  For pydeseq2 (which expects integer counts), we generate quasi-counts by:
  1. Applying expm1() to reverse the log1p transformation
  2. Summing per donor across all cells (pseudobulk aggregation)
  3. Rounding to integers and converting to float64

  This produces donor-level count-like matrices that preserve relative gene
  expression ratios. pydeseq2's negative binomial model will estimate dispersion
  from these quasi-counts. Results should be interpreted with the caveat that
  these are NOT true raw UMIs -- the normalization may affect variance estimation.

  As a validation, we also compute limma-style linear models on donor-level
  mean normalized expression.

REFERENCES:
  - Squair et al. 2021, Nat Commun (10.1038/s41467-021-25960-2): pseudobulk DE
  - pydeseq2 0.5.4 documentation
  - gseapy prerank (Subramanian et al. 2005)

RUNTIME: ~30-60 minutes (mostly pydeseq2 fitting)
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ============================================================================
# Constants
# ============================================================================

PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "batch_052"
DATA_DIR = PROJECT_ROOT / "data"

# Cell type filters per compartment
VASCULAR_CELLTYPES = ['ArtEC', 'CapEC', 'VenEC', 'IL6+ VenEC']
MUSC_CELLTYPES = None  # All cells
FAP_CELLTYPES = ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP', 'CD99+ FAP']

# Gene sets of interest
SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8',
          'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']
TARGET_TFS = ['JUNB', 'JUN', 'JUND', 'FOS', 'FOSB', 'FOSL1', 'FOSL2',
              'KLF2', 'KLF4', 'KLF6', 'KLF10', 'ATF3', 'EGR1', 'EGR2',
              'IRF1', 'CEBPB', 'CEBPD', 'RELA', 'NFKB1', 'STAT3', 'CDKN1A']

# Minimum thresholds
MIN_CELLS_PER_DONOR = 50
MIN_TOTAL_COUNTS_PER_GENE = 10

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# File paths
FILES = {
    'vascular': DATA_DIR / "Vascular_scsn_RNA.h5ad",
    'musc': DATA_DIR / "MuSC_scsn_RNA.h5ad",
    'fap': DATA_DIR / "OMIX004308-02.h5ad",
}


# ============================================================================
# Utility Functions
# ============================================================================

def log(msg):
    """Print timestamped message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_and_filter(filepath, celltypes, label):
    """
    Load h5ad file and filter to specified cell types.

    WHY: Each compartment has specific cell types relevant to the analysis.
    Vascular filters to endothelial subtypes. FAP filters to FAP subtypes.
    MuSC uses all cells (all are muscle stem cells).

    Returns filtered AnnData.
    """
    log(f"Loading {label}: {filepath}")
    adata = sc.read_h5ad(filepath)
    log(f"  Full shape: {adata.shape}")
    log(f"  Annotations: {list(adata.obs['Annotation'].unique())}")

    if celltypes is not None:
        mask = adata.obs['Annotation'].isin(celltypes)
        adata = adata[mask].copy()
        log(f"  After filtering to {celltypes}: {adata.shape}")
    else:
        log(f"  Using all cells (no cell type filter)")

    return adata


def compute_pseudobulk_quasicounts(adata, label):
    """
    Compute pseudobulk quasi-counts per donor by:
    1. Applying expm1() to reverse log1p normalization
    2. Summing across all cells per donor
    3. Rounding to integers

    WHY: The data contains log-normalized expression (not raw UMIs).
    expm1 reverses the log1p transform, giving linear-space normalized values.
    Summing across cells per donor produces quasi-counts proportional to
    total expression. Rounding is needed for pydeseq2's count-based model.

    This approach preserves the relative expression structure while producing
    integer-valued matrices that pydeseq2 can process. The dispersion estimates
    will reflect the quasi-count distribution, not true UMI overdispersion,
    but the relative DE rankings should be informative.

    Returns:
        count_df: DataFrame (donors x genes) with integer quasi-counts
        meta_df: DataFrame with donor metadata
        cell_counts: dict mapping donor -> number of cells
    """
    log(f"  Computing pseudobulk quasi-counts for {label}...")

    donors = adata.obs['sample'].values
    unique_donors = sorted(adata.obs['sample'].unique())
    gene_names = list(adata.var_names)
    n_genes = len(gene_names)

    log(f"  {len(unique_donors)} unique donors, {n_genes} genes")

    # Sum quasi-counts per donor
    # Process in chunks to avoid loading entire matrix to dense memory.
    # For sparse matrices, expm1 is applied to non-zero elements only
    # (expm1(0) = 0, so zero entries remain zero).
    count_data = {}
    cell_counts = {}

    X_sparse = adata.X
    is_sparse = issparse(X_sparse)

    for donor in unique_donors:
        mask = donors == donor
        n_cells = mask.sum()
        cell_counts[donor] = int(n_cells)

        if n_cells < MIN_CELLS_PER_DONOR:
            log(f"  WARNING: Donor {donor} has {n_cells} cells (< {MIN_CELLS_PER_DONOR}), skipping")
            continue

        # Extract donor rows and apply expm1 + sum
        donor_X = X_sparse[mask]
        if is_sparse:
            # expm1 on sparse: expm1(0)=0 so just transform data array
            donor_X = donor_X.copy()
            donor_X.data = np.expm1(donor_X.data.astype(np.float64))
            donor_sums = np.array(donor_X.sum(axis=0)).flatten()
        else:
            donor_X = donor_X.astype(np.float64)
            donor_sums = np.expm1(donor_X).sum(axis=0)
            if hasattr(donor_sums, 'A1'):
                donor_sums = donor_sums.A1

        count_data[donor] = np.clip(np.round(donor_sums), 0, None).astype(np.float64)

    if not count_data:
        raise ValueError(f"No donors with >= {MIN_CELLS_PER_DONOR} cells in {label}")

    count_df = pd.DataFrame(count_data, index=gene_names).T
    count_df.index.name = 'donor'

    log(f"  Pseudobulk shape: {count_df.shape} (donors x genes)")
    log(f"  Quasi-count range: [{count_df.values.min():.0f}, {count_df.values.max():.0f}]")
    log(f"  Quasi-count total per donor: mean={count_df.sum(axis=1).mean():.0f}, "
        f"std={count_df.sum(axis=1).std():.0f}")

    # Filter genes with low total counts
    gene_totals = count_df.sum(axis=0)
    keep_genes = gene_totals >= MIN_TOTAL_COUNTS_PER_GENE
    n_removed = (~keep_genes).sum()
    count_df = count_df.loc[:, keep_genes]
    log(f"  Removed {n_removed} genes with < {MIN_TOTAL_COUNTS_PER_GENE} total quasi-counts")
    log(f"  Remaining: {count_df.shape[1]} genes")

    # Build metadata
    valid_donors = list(count_df.index)
    meta_rows = []
    obs = adata.obs
    for donor in valid_donors:
        donor_obs = obs[obs['sample'] == donor].iloc[0]

        # Get sex column name (varies between files)
        sex_col = 'Sex' if 'Sex' in obs.columns else 'gender'

        age_pop = str(donor_obs['age_pop'])
        # Map: young_pop -> young, old_pop -> old
        age_group = 'young' if age_pop == 'young_pop' else 'old'

        meta_rows.append({
            'donor': donor,
            'country': str(donor_obs['Country']),
            'age_group': age_group,
            'age': float(donor_obs['age']),
            'tech': str(donor_obs['tech']),
            'sex': str(donor_obs[sex_col]),
            'n_cells': cell_counts[donor],
        })

    meta_df = pd.DataFrame(meta_rows).set_index('donor')

    # Print summary
    log(f"  Metadata summary:")
    log(f"    age_group: {dict(meta_df['age_group'].value_counts())}")
    log(f"    country: {dict(meta_df['country'].value_counts())}")
    log(f"    tech: {dict(meta_df['tech'].value_counts())}")

    # Cross-tabulation
    ct = pd.crosstab(meta_df['age_group'], meta_df['country'])
    log(f"    age_group x country:\n{ct.to_string()}")

    return count_df, meta_df, cell_counts


def compute_vif(meta_df, factor_cols):
    """
    Compute Variance Inflation Factor for design factors.

    WHY: VIF quantifies multicollinearity between covariates.
    VIF > 10 indicates severe collinearity that makes coefficient
    estimates unreliable. VIF > 5 is moderate concern.

    With country and age_group, if one country has mostly young
    and the other mostly old, VIF will be high.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    log(f"  Computing VIF for {factor_cols}...")

    # Create numeric design matrix
    X_design = pd.DataFrame(index=meta_df.index)
    for col in factor_cols:
        X_design[col] = pd.Categorical(meta_df[col]).codes

    X_design = add_constant(X_design)

    vif_data = []
    for i, col in enumerate(X_design.columns):
        if col == 'const':
            continue
        vif_val = variance_inflation_factor(X_design.values, i)
        vif_data.append({'factor': col, 'VIF': vif_val})
        flag = " [HIGH]" if vif_val > 10 else " [MODERATE]" if vif_val > 5 else ""
        log(f"    VIF({col}) = {vif_val:.2f}{flag}")

    return pd.DataFrame(vif_data)


def run_pydeseq2(count_df, meta_df, label):
    """
    Run pydeseq2 differential expression analysis.

    Design: ~ country + age_group
    Contrast: age_group (old vs young), controlling for country.

    WHY country-first design: Country (China vs Spain) is the dominant confound
    (rho_JUNB_country=0.626 >> rho_JUNB_age=0.042). Placing country first in the
    design absorbs this variance before testing age_group, preventing country-driven
    false positives.

    WHY pydeseq2: Standard pseudobulk DE tool (Python port of DESeq2).
    Uses negative binomial GLM with Wald test for contrasts.

    NOTE: Input quasi-counts are derived from log-normalized data (not raw UMIs).
    Dispersion estimates reflect quasi-count distribution.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    log(f"  Running pydeseq2 for {label}...")

    # Ensure count_df has no negative values (shouldn't happen, but guard)
    count_df = count_df.clip(lower=0)

    # Align metadata to count matrix
    common_donors = sorted(set(count_df.index) & set(meta_df.index))
    counts = count_df.loc[common_donors]
    metadata = meta_df.loc[common_donors].copy()

    log(f"  N donors for DE: {len(common_donors)}")
    log(f"  N genes: {counts.shape[1]}")

    # pydeseq2 expects counts as integers or float (>=0.5)
    # Our quasi-counts are already rounded
    counts_int = counts.astype(int)

    # Create DeseqDataSet
    # design: country first to absorb variance, then age_group
    # Uses formulaic formula syntax (pydeseq2 >= 0.4)
    dds = DeseqDataSet(
        counts=counts_int,
        metadata=metadata,
        design="~ country + age_group",
    )

    log(f"  Fitting model...")
    t0 = time.time()
    dds.deseq2()
    log(f"  Model fitting took {time.time() - t0:.1f}s")

    # Extract results for age_group contrast (old vs young)
    # The contrast tests the effect of age_group, controlling for country
    stat_res = DeseqStats(dds, contrast=['age_group', 'old', 'young'])
    stat_res.summary()

    # Get results DataFrame
    # Columns from pydeseq2: baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
    # padj is BH-adjusted p-value computed by pydeseq2's independent filtering
    results_df = stat_res.results_df.copy()

    # Rename columns for clarity
    results_df.index.name = 'gene'
    results_df = results_df.reset_index()
    results_df = results_df.rename(columns={
        'log2FoldChange': 'log2FC',
        'stat': 'wald_stat',
        'pvalue': 'p_value',
        'padj': 'bh_q',  # pydeseq2's BH-adjusted p-value (primary)
    })

    # Also compute manual BH as sanity check
    from statsmodels.stats.multitest import multipletests
    valid_mask = results_df['p_value'].notna()
    pvals = results_df.loc[valid_mask, 'p_value'].values

    if len(pvals) > 0:
        reject, bh_manual, _, _ = multipletests(pvals, method='fdr_bh')
        results_df.loc[valid_mask, 'bh_q_manual'] = bh_manual
    else:
        results_df['bh_q_manual'] = np.nan

    # Sort by p-value
    results_df = results_df.sort_values('p_value')

    # Clean up index (gene column already exists from first reset_index)
    results_df = results_df.reset_index(drop=True)

    # Report
    n_sig = int((results_df['bh_q'] < 0.05).sum())
    n_tested = int(results_df['p_value'].notna().sum())
    log(f"  DE results: {n_tested} genes tested (of {len(results_df)} total), "
        f"{n_sig} with BH q < 0.05")

    # Check JUNB
    junb_row = results_df[results_df['gene'] == 'JUNB']
    if not junb_row.empty:
        jr = junb_row.iloc[0]
        log(f"  JUNB: log2FC={jr['log2FC']:.3f}, p={jr['p_value']:.4g}, "
            f"BH q={jr['bh_q']:.4g}")
    else:
        log(f"  JUNB: NOT FOUND in results")

    # Check SASP12 genes
    log(f"  SASP12 gene status:")
    for gene in SASP12:
        row = results_df[results_df['gene'] == gene]
        if not row.empty:
            r = row.iloc[0]
            sig = '*' if r['bh_q'] < 0.05 else ''
            log(f"    {gene}: log2FC={r['log2FC']:.3f}, p={r['p_value']:.4g}, "
                f"q={r['bh_q']:.4g} {sig}")
        else:
            log(f"    {gene}: NOT FOUND")

    # Check TARGET_TFs
    log(f"  Target TF status:")
    for gene in TARGET_TFS:
        row = results_df[results_df['gene'] == gene]
        if not row.empty:
            r = row.iloc[0]
            sig = '*' if r['bh_q'] < 0.05 else ''
            log(f"    {gene}: log2FC={r['log2FC']:.3f}, p={r['p_value']:.4g}, "
                f"q={r['bh_q']:.4g} {sig}")
        else:
            log(f"    {gene}: NOT FOUND")

    # Top 10 DE genes
    log(f"  Top 10 DE genes (by p-value):")
    for i, row in results_df.head(10).iterrows():
        log(f"    {row['gene']}: log2FC={row['log2FC']:.3f}, "
            f"wald={row['wald_stat']:.2f}, p={row['p_value']:.4g}, q={row['bh_q']:.4g}")

    return results_df


def run_gsea(de_results, label):
    """
    Run pre-ranked GSEA on Wald-statistic-ranked gene list.

    WHY: GSEA discovers pathways beyond a priori hypotheses. The Wald statistic
    from DE provides a continuous ranking (not binary cutoff), which preserves
    information about sub-threshold effects.

    WHY Wald statistic (not log2FC): Wald statistic incorporates both effect size
    and significance, giving higher rank to genes with consistent large effects
    relative to their variance.

    WHY MSigDB Hallmarks + GO BP: Standard gene set collections in aging research.
    Hallmarks are curated, well-defined, and interpretable. GO BP provides
    comprehensive pathway coverage.
    """
    import gseapy as gp

    log(f"  Running GSEA for {label}...")

    # Create ranked gene list from Wald statistic
    # Remove NaN stats
    ranked = de_results[['gene', 'wald_stat']].dropna()
    ranked = ranked.sort_values('wald_stat', ascending=False)

    # Remove duplicates (keep first = highest Wald stat)
    ranked = ranked.drop_duplicates(subset='gene', keep='first')

    # Create Series for gseapy
    rnk = pd.Series(ranked['wald_stat'].values, index=ranked['gene'].values)

    log(f"  Ranked gene list: {len(rnk)} genes")
    log(f"  Wald stat range: [{rnk.min():.2f}, {rnk.max():.2f}]")

    # Check available GMT files in cache
    cache_dir = Path.home() / '.cache' / 'gseapy'
    gsea_results = {}

    # --- MSigDB Hallmarks ---
    hallmark_gmt = cache_dir / 'Enrichr.MSigDB_Hallmark_2020.gmt'

    if hallmark_gmt.exists():
        log(f"  Running prerank with MSigDB Hallmarks (local GMT)...")
        try:
            pre_res = gp.prerank(
                rnk=rnk,
                gene_sets=str(hallmark_gmt),
                outdir=None,
                permutation_num=1000,
                seed=SEED,
                min_size=15,
                max_size=500,
                no_plot=True,
                verbose=False,
            )

            res_df = pre_res.res2d.copy()
            if not res_df.empty:
                # gseapy 1.1.13 columns: Name, Term, ES, NES, NOM p-val, FDR q-val,
                #                         FWER p-val, Tag %, Gene %, Lead_genes
                # Keep original column names for compatibility
                gsea_results['hallmark'] = res_df
                log(f"  Hallmark GSEA: {len(res_df)} pathways tested")
            else:
                log(f"  WARNING: Hallmark GSEA returned empty results")
        except Exception as e:
            log(f"  WARNING: Hallmark GSEA failed: {e}")
    else:
        log(f"  WARNING: MSigDB Hallmark GMT not found at {hallmark_gmt}")

    # --- GO Biological Process ---
    gobp_gmt = cache_dir / 'Enrichr.GO_Biological_Process_2021.gmt'

    if gobp_gmt.exists():
        log(f"  Running prerank with GO Biological Process (local GMT)...")
        try:
            pre_res_bp = gp.prerank(
                rnk=rnk,
                gene_sets=str(gobp_gmt),
                outdir=None,
                permutation_num=1000,
                seed=SEED,
                min_size=15,
                max_size=500,
                no_plot=True,
                verbose=False,
            )

            res_bp = pre_res_bp.res2d.copy()
            if not res_bp.empty:
                # Keep original gseapy column names
                gsea_results['gobp'] = res_bp
                log(f"  GO BP GSEA: {len(res_bp)} pathways tested")
            else:
                log(f"  WARNING: GO BP GSEA returned empty results")
        except Exception as e:
            log(f"  WARNING: GO BP GSEA failed: {e}")
    else:
        log(f"  WARNING: GO BP GMT not found at {gobp_gmt}")

    # Report results
    for gs_name, res_df in gsea_results.items():
        log(f"\n  === {gs_name.upper()} Top 10 enriched pathways ===")

        # gseapy 1.1.13 actual columns: Term, NES, NOM p-val, FDR q-val, Lead_genes
        if 'NES' in res_df.columns:
            # Top positive (upregulated in old)
            pos = res_df[res_df['NES'] > 0].sort_values('NES', ascending=False).head(10)
            log(f"  Enriched in OLD (positive NES):")
            for _, row in pos.iterrows():
                nes = float(row['NES'])
                nom_p = float(row['NOM p-val'])
                fdr_q = float(row['FDR q-val'])
                term = str(row['Term'])
                log(f"    {term}: NES={nes:.3f}, NOM p={nom_p:.4g}, FDR q={fdr_q:.4g}")

            # Top negative (upregulated in young)
            neg = res_df[res_df['NES'] < 0].sort_values('NES', ascending=True).head(5)
            if not neg.empty:
                log(f"  Enriched in YOUNG (negative NES):")
                for _, row in neg.iterrows():
                    nes = float(row['NES'])
                    nom_p = float(row['NOM p-val'])
                    fdr_q = float(row['FDR q-val'])
                    term = str(row['Term'])
                    log(f"    {term}: NES={nes:.3f}, NOM p={nom_p:.4g}, FDR q={fdr_q:.4g}")

        # Check HALLMARK_INFLAMMATORY_RESPONSE
        # NOTE: Enrichr MSigDB Hallmark uses "Inflammatory Response" (space),
        # not "HALLMARK_INFLAMMATORY_RESPONSE" (underscore).
        if gs_name == 'hallmark':
            inflam = res_df[
                res_df['Term'].str.contains(
                    'inflammatory.response|^Inflammatory Response$',
                    case=False, na=False, regex=True
                )
            ]
            if not inflam.empty:
                row = inflam.iloc[0]
                nes = float(row['NES'])
                fdr_q = float(row['FDR q-val'])
                term_name = str(row['Term'])
                log(f"\n  {term_name}: NES={nes:.3f}, FDR q={fdr_q:.4g}")
                sig_flag = "SIGNIFICANT" if fdr_q < 0.05 else "not significant"
                log(f"  -> {sig_flag}")
            else:
                log(f"\n  Inflammatory Response: NOT FOUND in results")

    return gsea_results


def compute_limma_validation(count_df, meta_df, de_pydeseq2, label):
    """
    Validate pydeseq2 results with a simple linear model approach.

    WHY: Since the input data is log-normalized (not raw counts), a linear
    model on log-normalized donor-level means is methodologically more
    appropriate than pydeseq2's negative binomial model. This provides a
    cross-check: if both methods agree on the top DE genes, we have higher
    confidence in the results.

    Method: Vectorized OLS via numpy matrix algebra for speed.
    For each gene: log1p(quasi_count) ~ country + age_group
    Test the age_group coefficient using t-test.
    """
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    log(f"  Computing limma-style validation for {label}...")

    # log1p(quasi_counts) approximates the original normalized data
    # Clip to non-negative to handle floating point artifacts from expm1
    count_clipped = count_df.clip(lower=0).astype(np.float64)
    log_counts = np.log1p(count_clipped)

    # Create design matrix
    common_donors = sorted(set(count_df.index) & set(meta_df.index))
    y = log_counts.loc[common_donors].values  # (n_donors, n_genes)
    meta = meta_df.loc[common_donors]

    # Dummy coding: 0=China/young, 1=Spain/old
    country_code = (meta['country'] == 'Spain').astype(float).values
    age_code = (meta['age_group'] == 'old').astype(float).values

    # Design matrix: [intercept, country, age_group]
    n_donors = len(common_donors)
    X_design = np.column_stack([np.ones(n_donors), country_code, age_code])

    # Vectorized OLS: beta = (X'X)^{-1} X'y
    # y is (n_donors, n_genes), so beta is (3, n_genes)
    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    beta = XtX_inv @ X_design.T @ y  # (3, n_genes)

    # Residuals and standard errors
    residuals = y - X_design @ beta  # (n_donors, n_genes)
    df_resid = n_donors - X_design.shape[1]  # residual df
    mse = (residuals ** 2).sum(axis=0) / df_resid  # per-gene MSE

    # Standard error for age_group coefficient (index 2)
    se_beta = np.sqrt(mse * XtX_inv[2, 2])  # per-gene SE

    # t-statistic and p-value for age_group
    t_stat = beta[2, :] / se_beta
    pvals = 2 * scipy_stats.t.sf(np.abs(t_stat), df=df_resid)

    # BH correction
    valid = ~np.isnan(pvals)
    bh_q = np.full_like(pvals, np.nan)
    if valid.any():
        _, bh_q[valid], _, _ = multipletests(pvals[valid], method='fdr_bh')

    genes = list(log_counts.columns)
    lm_results = pd.DataFrame({
        'gene': genes,
        'log2FC_lm': beta[2, :],
        'p_value_lm': pvals,
        'bh_q_lm': bh_q,
    }).sort_values('p_value_lm')

    n_sig_lm = int((lm_results['bh_q_lm'] < 0.05).sum())
    log(f"  Limma-style: {n_sig_lm} genes with BH q < 0.05")

    # Check JUNB
    junb_row = lm_results[lm_results['gene'] == 'JUNB']
    if not junb_row.empty:
        jr = junb_row.iloc[0]
        log(f"  Limma JUNB: log2FC={jr['log2FC_lm']:.3f}, p={jr['p_value_lm']:.4g}, "
            f"q={jr['bh_q_lm']:.4g}")

    # Compare top genes with pydeseq2
    top_pydeseq2 = set(de_pydeseq2.head(50)['gene'])
    top_limma = set(lm_results.head(50)['gene'])
    overlap = top_pydeseq2 & top_limma
    log(f"  Top-50 gene overlap (pydeseq2 vs limma): {len(overlap)}/50")

    if len(overlap) >= 25:
        log(f"  -> GOOD AGREEMENT (>=50% overlap)")
    elif len(overlap) >= 15:
        log(f"  -> MODERATE AGREEMENT (30-50% overlap)")
    else:
        log(f"  -> LOW AGREEMENT (<30% overlap) -- interpret with caution")

    return lm_results


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    log("=" * 70)
    log("batch_052 B1 (Pseudobulk DE) + B2 (GSEA)")
    log("=" * 70)
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Seed: {SEED}")
    log("")
    log("CRITICAL NOTE: Input data is log-normalized (not raw UMIs).")
    log("Pseudobulk quasi-counts derived via expm1(X) + aggregation + rounding.")
    log("pydeseq2 results should be interpreted with this caveat.")
    log("Limma-style validation provided as cross-check.")
    log("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}
    all_de_tables = {}
    all_gsea_tables = {}
    all_vif = {}

    compartments = [
        ('vascular', VASCULAR_CELLTYPES, FILES['vascular']),
        ('musc', MUSC_CELLTYPES, FILES['musc']),
        ('fap', FAP_CELLTYPES, FILES['fap']),
    ]

    for label, celltypes, filepath in compartments:
        log("")
        log("=" * 70)
        log(f"COMPARTMENT: {label.upper()}")
        log("=" * 70)

        try:
            # ---- B1: Pseudobulk DE ----
            log(f"\n--- B1: Pseudobulk DE ({label}) ---")

            # Load and filter
            adata = load_and_filter(filepath, celltypes, label)

            # Compute pseudobulk quasi-counts
            count_df, meta_df, cell_counts = compute_pseudobulk_quasicounts(adata, label)

            # VIF check
            vif_df = compute_vif(meta_df, ['country', 'age_group'])
            all_vif[label] = vif_df.to_dict('records')

            # pydeseq2
            try:
                de_results = run_pydeseq2(count_df, meta_df, label)

                # Save DE table
                de_path = OUTPUT_DIR / f"b1_de_{label}.csv"
                de_results.to_csv(de_path, index=False)
                log(f"  Saved: {de_path}")

                all_de_tables[label] = de_results

                # Limma validation
                lm_results = compute_limma_validation(count_df, meta_df, de_results, label)

                # Merge limma results into DE table
                de_merged = de_results.merge(lm_results, on='gene', how='left')
                de_merged.to_csv(de_path, index=False)
                log(f"  Updated with limma columns: {de_path}")

            except Exception as e:
                log(f"  ERROR: pydeseq2 failed for {label}: {e}")
                import traceback
                traceback.print_exc()
                all_results[f"{label}_error"] = str(e)
                continue

            # ---- B2: GSEA ----
            log(f"\n--- B2: GSEA ({label}) ---")

            try:
                gsea_results = run_gsea(de_results, label)

                # Combine hallmark + GO BP into single table per compartment
                combined_gsea = []
                for gs_name, gs_df in gsea_results.items():
                    gs_df = gs_df.copy()
                    gs_df['gene_set_collection'] = gs_name
                    combined_gsea.append(gs_df)

                if combined_gsea:
                    gsea_combined = pd.concat(combined_gsea, ignore_index=True)
                    gsea_path = OUTPUT_DIR / f"b2_gsea_{label}.csv"
                    gsea_combined.to_csv(gsea_path, index=False)
                    log(f"  Saved: {gsea_path}")
                    all_gsea_tables[label] = gsea_combined
                else:
                    log(f"  WARNING: No GSEA results for {label}")

            except Exception as e:
                log(f"  ERROR: GSEA failed for {label}: {e}")
                import traceback
                traceback.print_exc()
                all_results[f"{label}_gsea_error"] = str(e)

            # Collect summary stats
            junb_row = de_results[de_results['gene'] == 'JUNB']
            junb_info = {}
            if not junb_row.empty:
                jr = junb_row.iloc[0]
                junb_info = {
                    'log2FC': float(jr['log2FC']),
                    'wald_stat': float(jr['wald_stat']),
                    'p_value': float(jr['p_value']),
                    'bh_q': float(jr['bh_q']),
                    'significant_bh005': bool(jr['bh_q'] < 0.05),
                }

            n_sig = int((de_results['bh_q'] < 0.05).sum()) if de_results['bh_q'].notna().any() else 0

            # Check HALLMARK_INFLAMMATORY_RESPONSE in GSEA
            inflam_info = {}
            if label in all_gsea_tables:
                gsea_df = all_gsea_tables[label]
                inflam = gsea_df[
                    gsea_df['Term'].str.contains(
                        'inflammatory.response|^Inflammatory Response$',
                        case=False, na=False, regex=True
                    )
                ]
                if not inflam.empty:
                    ir = inflam.iloc[0]
                    inflam_info = {
                        'NES': float(ir['NES']),
                        'FDR_q': float(ir['FDR q-val']),
                    }

            all_results[label] = {
                'n_donors': len(meta_df),
                'n_genes_tested': len(de_results),
                'n_sig_bh005': n_sig,
                'JUNB': junb_info,
                'INFLAMMATORY_RESPONSE': inflam_info,
                'VIF': all_vif.get(label, []),
                'cell_counts': cell_counts,
                'metadata_summary': {
                    'age_group': dict(meta_df['age_group'].value_counts()),
                    'country': dict(meta_df['country'].value_counts()),
                },
                'data_caveat': 'quasi-counts from log-normalized data (not raw UMIs)',
            }

            # Free memory
            del adata, count_df, meta_df
            import gc
            gc.collect()

        except Exception as e:
            log(f"  FATAL ERROR for {label}: {e}")
            import traceback
            traceback.print_exc()
            all_results[f"{label}_fatal_error"] = str(e)

    # ---- Save combined results JSON ----
    log("")
    log("=" * 70)
    log("SAVING COMBINED RESULTS")
    log("=" * 70)

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    results_path = OUTPUT_DIR / "results_b1b2.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    log(f"Saved: {results_path}")

    # ---- Final Summary ----
    log("")
    log("=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)

    for label in ['vascular', 'musc', 'fap']:
        if label in all_results and isinstance(all_results[label], dict):
            r = all_results[label]
            log(f"\n{label.upper()}:")
            log(f"  Donors: {r['n_donors']}, Genes tested: {r['n_genes_tested']}")
            log(f"  Significant (BH q<0.05): {r['n_sig_bh005']}")

            junb = r.get('JUNB', {})
            if junb:
                sig = "SIGNIFICANT" if junb.get('significant_bh005', False) else "not significant"
                log(f"  JUNB: log2FC={junb.get('log2FC', float('nan')):.3f}, "
                    f"q={junb.get('bh_q', float('nan')):.4g} [{sig}]")
            else:
                log(f"  JUNB: NOT FOUND")

            inflam = r.get('INFLAMMATORY_RESPONSE', {})
            if inflam:
                log(f"  Inflammatory Response (Hallmark): NES={inflam.get('NES', float('nan')):.3f}, "
                    f"FDR q={inflam.get('FDR_q', float('nan')):.4g}")
            else:
                log(f"  Inflammatory Response (Hallmark): NOT TESTED")

            log(f"  Data: {r.get('data_caveat', 'unknown')}")
        elif f"{label}_fatal_error" in all_results:
            log(f"\n{label.upper()}: FAILED - {all_results[f'{label}_fatal_error']}")
        else:
            log(f"\n{label.upper()}: No results")

    log("")
    log("DONE.")

    return all_results


if __name__ == '__main__':
    results = main()
