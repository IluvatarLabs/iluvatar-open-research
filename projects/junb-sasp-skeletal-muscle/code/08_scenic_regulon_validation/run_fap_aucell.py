#!/usr/bin/env python3
"""
batch_055 D1: FAP-only pySCENIC GRNBoost2 + AUCell + Donor-Level SASP Correlation
=================================================================================
PURPOSE: Run GRNBoost2 co-expression regulon inference and AUCell scoring for the
HLMA FAP compartment only. batch_054 ran this for Vascular and MuSC but the FAP
process was killed before completion.

WHY FAP ONLY: The FAP compartment is the therapeutic target (JUNB-SASP coupling
is FAP-specific per batch_051). Vascular and MuSC results are already in batch_054.
This standalone script enables FAP-specific analysis with full snRNA cells.

KEY DESIGN (from batch_054 v3):
- snRNA-only filter: FAP data has 40,389 cells split across snRNA (8,513) and scRNA
  (31,876). Only snRNA cells are used for GRNBoost2 to avoid tech confound.
- Z-scored data: FAP data contains negative values (mean ≈ 0, ~90% negative entries).
  We use variance-based HVG selection via pre-computed dispersions_norm rather than
  scanpy's seurat flavor (which requires non-negative data).
- All expressed TFs: GRNBoost2 uses ALL TFs from human_tfs.txt, not just the 21
  TARGET_TFs, to prevent misattribution of shared targets.
- Co-expression regulons only: cisTarget is skipped (broken in batch_054).
  Regulons are built from top-80th-percentile GRNBoost2 importance per TF.

RUNTIME: ~2-4 hours for GRNBoost2 + AUCell.

OUTPUTS:
  d1_adjacencies_HLMA_FAP.csv       -- GRNBoost2 TF-target pairs
  d1_aucell_HLMA_FAP.csv            -- AUCell regulon scores (cells x regulons)
  d1_donor_averages_HLMA_FAP.csv    -- per-donor mean AUCell + raw mRNA
  d1_correlations_HLMA_FAP.csv     -- Spearman rho per TF vs SASP12_mean
  d1_log.txt                        -- timestamped progress log
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import norm
import json
import sys
import time
from pathlib import Path
import os

warnings.filterwarnings('ignore')

# ============================================================================
# Constants (module-level, no execution side effects)
# ============================================================================

OUTDIR = "experiments/batch_055"
DATASET_NAME = "HLMA_FAP"

# FAP-specific cell type filter
FAP_CELL_TYPES = ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP', 'CD99+ FAP']

# FAP data path (z-scored)
FAP_PATH = 'data/OMIX004308-02.h5ad'

# SASP12 canonical panel
SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8',
          'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

# Target TFs for reporting (subset of full TF list)
TARGET_TFS = ['JUNB', 'KLF10', 'CEBPB', 'EGR2', 'CDKN1A', 'EGR1',
              'ATF3', 'FOSL1', 'IRF1', 'FOS', 'FOSB']

# All human TFs for GRNBoost2 (full TF list, not just TARGET_TFS)
TF_LIST = "databases/human_tfs.txt"

# DASK / ARBORETO SETTINGS
# Set BEFORE importing arboreto or distributed. This is critical:
# Python 3.13 defaults to 'spawn' multiprocessing which causes infinite
# re-execution on module re-import. GRNBoost2 and dask must see fork mode.
os.environ['ARBORETO_FORCE_DASK'] = 'False'


# ============================================================================
# UTILITY FUNCTIONS (no side effects at import time)
# ============================================================================

def fisher_z_ci(rho, n, alpha=0.05):
    """Compute 95% CI for Spearman rho via Fisher Z transformation.

    WHY Fisher Z: Spearman rho has bounded support [-1, 1] and non-normal
    sampling distribution. The Fisher Z transform = arctanh(rho) is approximately
    normal with SE = 1/sqrt(n-3), enabling symmetric CI construction.
    The eps=1e-15 guard prevents log(0) when |rho| = 1 exactly.
    """
    rho_clamped = np.clip(rho, -1 + 1e-15, 1 - 1e-15)
    z_rho = np.arctanh(rho_clamped)
    se = 1.0 / np.sqrt(max(n - 3, 1))
    z_crit = norm.ppf(1 - alpha / 2)
    z_lo = z_rho - z_crit * se
    z_hi = z_rho + z_crit * se
    ci_lo = np.tanh(z_lo)
    ci_hi = np.tanh(z_hi)
    return ci_lo, ci_hi


def timestamp():
    """Current time string for progress logging."""
    return time.strftime('%Y-%m-%d %H:%M:%S')


def log_print(msg, log_file=None):
    """Print to stdout and write to log file simultaneously."""
    print(f"[{timestamp()}] {msg}")
    if log_file:
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp()}] {msg}\n")
            f.flush()


def load_and_filter(path, cell_types, filter_col='Annotation'):
    """Load h5ad, optionally filter to specific cell types, return AnnData.

    WHY: We need the full AnnData object for this script because GRNBoost2
    requires the expression matrix. backed='r' would work but we need the
    variance-based HVG computation which requires full in-memory access.

    Returns: AnnData object with specified cell types.
    """
    import anndata as ad

    adata = ad.read_h5ad(path)
    if cell_types:
        mask = adata.obs[filter_col].isin(cell_types)
        adata = adata[mask].copy()

    return adata


def compute_hvg_expression(adata, n_top_genes=5000, human_tfs=None):
    """Select top HVGs and return expression DataFrame for GRNBoost2.

    WHY 5000 HVGs: All ~37K genes x all ~1200 TFs makes GRNBoost2
    O(TFs * genes * n_estimators * n_cells) prohibitively slow. The top
    5000 HVGs capture the majority of biological signal.

    WHY force-include TFs: Many TFs of interest (including JUNB) are NOT
    in the top 5000 HVGs because TFs tend to be lowly/ubiquitously expressed.
    GRNBoost2 needs TFs as both regulators and potential targets. We force-
    include all expressed TFs from human_tfs.txt.

    Z-SCORED DATA HANDLING (FAP):
    The FAP h5ad contains z-scored data (mean ≈ 0, ~90% negative entries,
    range [-6, +10]). scanpy's seurat flavor requires non-negative data and
    would produce invalid dispersion estimates here. Instead we use the
    pre-computed dispersions_norm column already stored in adata.var, which
    was computed correctly on the z-scored data during the original scanpy
    pipeline. This column is available in OMIX004308-02.h5ad.

    Returns: DataFrame (cells x genes) with HGNC symbol column names.
    """
    import scipy.sparse as sp

    adata_work = adata.copy()

    # Detect data format: if X contains negative values, it's z-scored
    X_sample = adata_work.X[:100, :]
    if sp.issparse(X_sample):
        X_sample = X_sample.toarray()
    has_negative = (X_sample < 0).any()
    fraction_negative = (X_sample < 0).mean()

    if has_negative and fraction_negative > 0.1:
        # Z-scored data (FAP). Use pre-computed dispersions_norm.
        print(f"    Z-scored data detected ({fraction_negative:.1%} negative values). "
              f"Using pre-computed dispersions_norm from adata.var.")

        if 'dispersions_norm' in adata_work.var.columns:
            # Rank genes by normalized dispersion, take top N
            print(f"    Using pre-computed dispersions_norm from adata.var...")
            top_genes = adata_work.var.nlargest(n_top_genes, 'dispersions_norm').index
            hvg_mask = adata_work.var_names.isin(top_genes)
        else:
            # Fallback: select by variance if dispersions_norm not available
            print(f"    No dispersions_norm. Falling back to variance-based selection...")
            X_all = adata_work.X
            if sp.issparse(X_all):
                X_all = X_all.toarray()
            gene_variances = np.var(X_all, axis=0)
            top_indices = np.argsort(gene_variances)[-n_top_genes:]
            hvg_mask = np.zeros(adata_work.n_vars, dtype=bool)
            hvg_mask[top_indices] = True
    else:
        # Log-normalized data. Use scanpy's seurat flavor.
        import scanpy as sc
        print(f"    Computing {n_top_genes} HVGs (flavor='seurat', log-normalized data)...")
        sc.pp.highly_variable_genes(
            adata_work,
            n_top_genes=n_top_genes,
            flavor='seurat',
            subset=False,
        )
        hvg_mask = adata_work.var['highly_variable'].values

    print(f"    HVGs selected: {hvg_mask.sum()}")

    # Force-include all expressed TFs from the human TF list
    if human_tfs is not None:
        tf_mask = adata_work.var_names.isin(human_tfs)
        tf_not_hvg = tf_mask & ~hvg_mask
        n_added = tf_not_hvg.sum()
        hvg_mask = hvg_mask | tf_mask
        print(f"    Force-including {n_added} TFs not in top {n_top_genes} HVGs")
        print(f"    Combined genes (HVGs + all TFs): {hvg_mask.sum()}")
    else:
        print(f"    HVGs (no TF list): {hvg_mask.sum()}")

    adata_work = adata_work[:, hvg_mask].copy()
    print(f"    Final genes for GRNBoost2: {adata_work.n_vars}")

    # Extract expression matrix
    X = adata_work.X
    if sp.issparse(X):
        X = X.toarray()

    gene_names = list(adata_work.var_names)
    ex_df = pd.DataFrame(X, columns=gene_names, index=adata_work.obs_names)

    return ex_df


def compute_sasp_composite(adata, sasp_genes):
    """Compute per-cell SASP12 composite score (mean of detected SASP genes).

    WHY mean (not z-score): L001 from batch_045 showed that within-donor
    z-scoring introduces artifacts when donor variance is small. Raw mean
    is the established canonical methodology from batch_050.

    Returns: (sasp_per_cell array, list of detected gene names)
    """
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    var_names = list(adata.var_names)
    detected = [g for g in sasp_genes if g in var_names]
    print(f"    SASP12 genes detected: {len(detected)}/{len(sasp_genes)} -> {detected}")

    if len(detected) == 0:
        return np.zeros(adata.n_obs), detected

    indices = [var_names.index(g) for g in detected]
    sasp_vals = X[:, indices]
    return np.mean(sasp_vals, axis=1), detected


def compute_donor_averages(adata, aucell_df, sasp_genes, tf_list):
    """Compute donor-level averages for AUCell scores and raw mRNA.

    WHY donor-level: Country is a donor-level confounder. Computing cell-level
    correlations would be pseudoreplicated. We aggregate to one value per donor.

    MIN 50 CELLS: batch_053 established this threshold. It balances donor N
    against noise from too few cells per donor. At N=22 donors, we can afford
    to drop low-coverage donors.

    Returns DataFrame with one row per donor: sample, n_cells, SASP12_mean,
    raw_{TF}, aucell_{regulon}, and metadata columns.
    """
    import scipy.sparse as sp

    obs = adata.obs.copy()
    # Standardize sample/donor column name
    sample_col = 'sample' if 'sample' in obs.columns else 'donor_id'

    sasp_per_cell, _ = compute_sasp_composite(adata, sasp_genes)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    var_names = list(adata.var_names)

    donor_data = []
    for donor in sorted(obs[sample_col].unique()):
        mask = obs[sample_col] == donor
        n_cells = int(mask.sum())

        row = {
            'sample': donor,
            'n_cells': n_cells,
        }

        # Donor-level metadata columns
        for col in ['age', 'age_pop', 'Sex', 'sex', 'gender', 'Country', 'country', 'tech']:
            if col in obs.columns:
                val = obs.loc[mask, col]
                row[col] = val.iloc[0] if len(val) > 0 else np.nan

        # Mean SASP12 composite
        row['SASP12_mean'] = float(np.mean(sasp_per_cell[mask.values]))

        # Mean raw mRNA for TARGET_TFs
        for tf in tf_list:
            if tf in var_names:
                idx = var_names.index(tf)
                row[f'raw_{tf}'] = float(np.mean(X[mask.values, idx]))
            else:
                row[f'raw_{tf}'] = np.nan

        # Mean AUCell scores
        aucell_donor = aucell_df.loc[mask.values]
        for col_name in aucell_df.columns:
            row[f'aucell_{col_name}'] = float(aucell_donor[col_name].mean())

        donor_data.append(row)

    return pd.DataFrame(donor_data)


def build_coexpression_regulons(adjacencies, tf_list,
                                  quantile=0.80, min_targets=5, max_targets=200):
    """Build co-expression regulons from GRNBoost2 adjacencies.

    USED INSTEAD OF cisTarget: cisTarget motif pruning was skipped because
    the motif databases were incompatible with the hg38 gene nomenclature
    in batch_054. This co-expression fallback builds regulons from the
    top-80th-percentile importance scores per TF.

    WHY 80th percentile: This selects the top 20% of targets per TF by
    GRNBoost2 importance, balancing sensitivity (keeping real targets)
    vs specificity (excluding noise). This was validated against cisTarget
    in batch_054 where both were available (Vascular, MuSC).

    WHY min 5, max 200 targets: < 5 targets gives unstable AUCell scores.
    > 200 targets dilutes the regulon with non-specific co-expression.

    Returns dict: {tf_name: [target_gene, ...]}
    """
    regulons = {}
    for tf_name in tf_list:
        sub = adjacencies[adjacencies['TF'] == tf_name].sort_values(
            'importance', ascending=False
        )
        if len(sub) == 0:
            continue
        threshold = sub['importance'].quantile(quantile)
        top_targets = sub[sub['importance'] >= threshold]['target'].tolist()
        # Exclude the TF itself from its own regulon
        top_targets = [g for g in top_targets if g != tf_name]
        if len(top_targets) >= min_targets:
            regulons[tf_name] = top_targets[:max_targets]
    return regulons


# ============================================================================
# MAIN ANALYSIS
# All executable code inside __name__ guard for multiprocessing safety.
# Python 3.13 defaults to 'spawn' start method which re-imports the module
# in child processes. The guard prevents infinite re-execution.
# ============================================================================

if __name__ == '__main__':
    # CRITICAL: Set multiprocessing start method to fork BEFORE importing
    # dask/arboreto. GRNBoost2 uses sklearn GBM which is CPU-bound and does
    # NOT release the GIL effectively. Fork-based processes share memory
    # pages (COW) so the expression matrix is not duplicated in RAM.
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

    os.makedirs(OUTDIR, exist_ok=True)
    LOG_FILE = f"{OUTDIR}/d1_log.txt"

    log_print("=== batch_055 D1: FAP pySCENIC AUCell Analysis ===", LOG_FILE)
    log_print(f"Output directory: {OUTDIR}", LOG_FILE)

    # ----------------------------------------------------------------
    # Load TF list
    # ----------------------------------------------------------------
    log_print("Loading human TF list...", LOG_FILE)
    with open(TF_LIST) as f:
        human_tfs = set(line.strip() for line in f if line.strip())
    log_print(f"  Loaded {len(human_tfs)} human TFs from {TF_LIST}", LOG_FILE)
    log_print(f"  JUNB in TF list: {'JUNB' in human_tfs}", LOG_FILE)

    target_tfs_in_list = sum(1 for tf in TARGET_TFS if tf in human_tfs)
    log_print(f"  Target TFs in list: {target_tfs_in_list}/{len(TARGET_TFS)}", LOG_FILE)

    missing_tfs = [tf for tf in TARGET_TFS if tf not in human_tfs]
    if missing_tfs:
        log_print(f"  Target TFs MISSING from TF list: {missing_tfs}", LOG_FILE)

    # ----------------------------------------------------------------
    # Step 1: Load FAP data
    # ----------------------------------------------------------------
    log_print(f"\nStep 1: Loading FAP data...", LOG_FILE)
    adata_full = load_and_filter(FAP_PATH, FAP_CELL_TYPES, filter_col='Annotation')
    log_print(f"  Full data shape: {adata_full.shape}", LOG_FILE)

    # Tech distribution
    log_print("  Tech distribution (FAP cells):", LOG_FILE)
    for tech_val in sorted(adata_full.obs['tech'].unique()):
        sub = adata_full.obs[adata_full.obs['tech'] == tech_val]
        n_donors = sub['sample'].nunique()
        n_cells = len(sub)
        log_print(f"    {tech_val}: {n_cells} cells, {n_donors} donors", LOG_FILE)

    # Country distribution
    if 'Country' in adata_full.obs.columns:
        log_print("  Country distribution (FAP cells):", LOG_FILE)
        for ctry in sorted(adata_full.obs['Country'].unique()):
            sub = adata_full.obs[adata_full.obs['Country'] == ctry]
            log_print(f"    {ctry}: {len(sub)} cells, {sub['sample'].nunique()} donors", LOG_FILE)

    # Split: snRNA-only for GRNBoost2
    if 'tech' in adata_full.obs.columns:
        snrna_mask = adata_full.obs['tech'] == 'snRNA'
        adata_snrna = adata_full[snrna_mask.values].copy()
        log_print(f"\n  snRNA-only (for GRNBoost2): {adata_snrna.shape[0]} cells", LOG_FILE)
        log_print(f"  All cells (for AUCell scoring): {adata_full.shape[0]} cells", LOG_FILE)

        # snRNA country distribution
        log_print("  snRNA Country distribution:", LOG_FILE)
        for ctry in sorted(adata_snrna.obs['Country'].unique()):
            sub = adata_snrna.obs[adata_snrna.obs['Country'] == ctry]
            log_print(f"    {ctry}: {len(sub)} cells, {sub['sample'].nunique()} donors", LOG_FILE)
    else:
        adata_snrna = adata_full.copy()
        log_print(f"  No tech column. Using all {adata_snrna.shape[0]} cells.", LOG_FILE)

    # ----------------------------------------------------------------
    # Step 2: HVG selection on snRNA cells
    # ----------------------------------------------------------------
    log_print(f"\nStep 2: HVG selection (top 5000 + all expressed TFs)...", LOG_FILE)
    ex_snrna = compute_hvg_expression(adata_snrna, n_top_genes=5000, human_tfs=human_tfs)

    # Filter TFs to those present in the HVG expression matrix
    tf_in_data = [g for g in ex_snrna.columns if g in human_tfs]
    log_print(f"  TFs present in HVG data: {len(tf_in_data)}", LOG_FILE)

    # Check target TFs presence
    target_tfs_present = [tf for tf in TARGET_TFS if tf in ex_snrna.columns]
    target_tfs_missing = [tf for tf in TARGET_TFS if tf not in ex_snrna.columns]
    log_print(f"  Target TFs present in HVGs: {len(target_tfs_present)}/{len(TARGET_TFS)}", LOG_FILE)
    if target_tfs_missing:
        log_print(f"  Target TFs MISSING from HVGs: {target_tfs_missing}", LOG_FILE)
        log_print(f"  NOTE: These TFs will still be scored via AUCell if present in co-expression regulons.", LOG_FILE)

    # ----------------------------------------------------------------
    # Step 3: GRNBoost2 on snRNA-only, all expressed TFs x HVGs
    # ----------------------------------------------------------------
    log_print(f"\nStep 3: GRNBoost2 inference...", LOG_FILE)
    log_print(f"  Input: {ex_snrna.shape[0]} cells x {ex_snrna.shape[1]} genes", LOG_FILE)
    log_print(f"  TFs for inference: {len(tf_in_data)}", LOG_FILE)

    from arboreto.algo import grnboost2

    t_grn_start = time.time()
    adjacencies = grnboost2(
        expression_data=ex_snrna,
        tf_names=tf_in_data,
        client_or_address='local',  # Use local mode with fork
        verbose=True,
        seed=42,
    )
    t_grn = time.time() - t_grn_start
    log_print(f"  GRNBoost2 completed in {t_grn/60:.1f} minutes", LOG_FILE)
    log_print(f"  Adjacencies: {len(adjacencies)} TF-target pairs", LOG_FILE)
    log_print(f"  Unique TFs with adjacencies: {adjacencies['TF'].nunique()}", LOG_FILE)

    # Save adjacencies
    adj_path = f"{OUTDIR}/d1_adjacencies_{DATASET_NAME}.csv"
    adjacencies.to_csv(adj_path, index=False)
    log_print(f"  Saved: {adj_path}", LOG_FILE)

    # Top adjacencies per TARGET_TF
    log_print("  Top TARGET_TF adjacencies by importance:", LOG_FILE)
    for tf in TARGET_TFS:
        sub = adjacencies[adjacencies['TF'] == tf].sort_values('importance', ascending=False)
        if len(sub) > 0:
            top_targets = sub.head(3)[['target', 'importance']].values
            targets_str = ', '.join([f"{t}({i:.4f})" for t, i in top_targets])
            log_print(f"    {tf}: {len(sub)} adjacencies. Top: {targets_str}", LOG_FILE)
        else:
            log_print(f"    {tf}: NO adjacencies", LOG_FILE)

    # ----------------------------------------------------------------
    # Step 4: cisTarget SKIPPED (broken)
    # Skip entirely: use co-expression regulons only
    # ----------------------------------------------------------------
    log_print(f"\nStep 4: cisTarget SKIPPED (motif databases incompatible with hg38).", LOG_FILE)
    log_print(f"  Using co-expression regulons (GRNBoost2 importance-based).", LOG_FILE)

    motif_regulons = {}  # Empty -- no cisTarget
    motif_tf_names = set()  # Empty set

    # ----------------------------------------------------------------
    # Step 4b: Build co-expression regulons
    # ----------------------------------------------------------------
    log_print(f"\nStep 4b: Building co-expression regulons...", LOG_FILE)
    coexpression_regulons = build_coexpression_regulons(
        adjacencies,
        tf_in_data,
        quantile=0.80,   # Top 80th percentile importance per TF
        min_targets=5,   # Minimum 5 targets for stable AUCell
        max_targets=200,  # Maximum 200 targets to avoid dilution
    )
    log_print(f"  Co-expression regulons built: {len(coexpression_regulons)}", LOG_FILE)

    # Log regulon sizes for TARGET_TFS
    log_print("  Regulon sizes for TARGET_TFS:", LOG_FILE)
    for tf in TARGET_TFS:
        if tf in coexpression_regulons:
            n_targets = len(coexpression_regulons[tf])
            source = 'motif' if tf in motif_tf_names else 'coexpression'
            log_print(f"    {tf} ({source}): {n_targets} targets", LOG_FILE)
            if n_targets > 0:
                top5 = coexpression_regulons[tf][:5]
                log_print(f"      Top targets: {', '.join(top5)}", LOG_FILE)
        else:
            log_print(f"    {tf}: NO regulon (too few targets)", LOG_FILE)

    # ----------------------------------------------------------------
    # Step 5: AUCell scoring on ALL (snRNA + scRNA) cells
    # ----------------------------------------------------------------
    log_print(f"\nStep 5: AUCell scoring on ALL cells...", LOG_FILE)

    from pyscenic.aucell import aucell
    from ctxcore.genesig import GeneSignature

    # Build gene signatures for AUCell
    signatures = []

    # Add co-expression regulons
    for tf_name, targets in coexpression_regulons.items():
        # Get importance scores for weighting
        sub = adjacencies[
            (adjacencies['TF'] == tf_name) & (adjacencies['target'].isin(targets))
        ]
        if len(sub) > 0:
            gene2weight = dict(zip(sub['target'], sub['importance']))
        else:
            gene2weight = {g: 1.0 for g in targets}

        sig = GeneSignature(
            name=f"{tf_name}(+)",
            gene2weight=gene2weight,
        )
        signatures.append(sig)

    # Ensure TARGET_TFs with no regulon are noted
    missing_from_sigs = set(TARGET_TFS) - set(coexpression_regulons.keys())
    if missing_from_sigs:
        log_print(f"  TARGET_TFs with NO regulon: {missing_from_sigs}", LOG_FILE)

    log_print(f"  Total regulon signatures for AUCell: {len(signatures)}", LOG_FILE)

    if len(signatures) == 0:
        log_print("  ERROR: No regulon signatures. Aborting.", LOG_FILE)
        raise RuntimeError("No regulon signatures for AUCell")

    # Prepare full expression matrix (ALL cells, ALL genes) for AUCell
    import scipy.sparse as sp

    X_all = adata_full.X
    if sp.issparse(X_all):
        X_all = X_all.toarray()
    ex_all_df = pd.DataFrame(
        X_all,
        columns=list(adata_full.var_names),
        index=adata_full.obs_names,
    )
    log_print(f"  Expression matrix: {ex_all_df.shape[0]} cells x {ex_all_df.shape[1]} genes", LOG_FILE)

    # Check that regulon genes are present in expression matrix
    genes_in_matrix = set(ex_all_df.columns)
    for sig in signatures:
        missing_genes = [g for g in sig.gene2weight if g not in genes_in_matrix]
        if missing_genes:
            # Remove missing genes from signature
            sig.gene2weight = {g: w for g, w in sig.gene2weight.items() if g in genes_in_matrix}
            if sig.gene2weight:
                log_print(f"  WARNING: {sig.name} had {len(missing_genes)} missing genes, removed.", LOG_FILE)

    log_print(f"  Running AUCell: {ex_all_df.shape[0]} cells x {len(signatures)} regulons...", LOG_FILE)
    t_auc_start = time.time()

    aucell_df = aucell(
        exp_mtx=ex_all_df,
        signatures=signatures,
        auc_threshold=0.05,  # Top 5% of ranked genes per cell
        noweights=False,
        normalize=False,
        seed=42,
        num_workers=max(1, os.cpu_count() - 4),
    )

    t_auc = time.time() - t_auc_start
    log_print(f"  AUCell completed in {t_auc/60:.1f} minutes", LOG_FILE)
    log_print(f"  AUCell matrix shape: {aucell_df.shape}", LOG_FILE)

    # Verify cell order matches between adata_full and aucell_df
    assert list(aucell_df.index) == list(adata_full.obs_names), \
        "Cell order mismatch between adata_full and aucell_df!"

    # Save AUCell scores
    auc_path = f"{OUTDIR}/d1_aucell_{DATASET_NAME}.csv"
    aucell_df.to_csv(auc_path)
    log_print(f"  Saved: {auc_path}", LOG_FILE)

    # Clean up large expression matrices to free memory
    del X_all, ex_all_df, ex_snrna

    # ----------------------------------------------------------------
    # Step 6: Donor-level averages and correlations
    # ----------------------------------------------------------------
    log_print(f"\nStep 6: Donor-level averages and correlations...", LOG_FILE)

    # Compute donor averages (min 50 cells per donor)
    donor_df = compute_donor_averages(adata_full, aucell_df, SASP12, TARGET_TFS)

    # Apply min 50 cells filter
    donor_df_before = len(donor_df)
    donor_df = donor_df[donor_df['n_cells'] >= 50].copy()
    donor_df_after = len(donor_df)
    log_print(f"  Donors: {donor_df_after}/{donor_df_before} (min 50 cells)", LOG_FILE)

    if 'Country' in donor_df.columns:
        log_print(f"  Country distribution (after min_cells filter):", LOG_FILE)
        for ctry in sorted(donor_df['Country'].unique()):
            n_d = (donor_df['Country'] == ctry).sum()
            log_print(f"    {ctry}: {n_d} donors", LOG_FILE)

    donor_path = f"{OUTDIR}/d1_donor_averages_{DATASET_NAME}.csv"
    donor_df.to_csv(donor_path, index=False)
    log_print(f"  Saved: {donor_path}", LOG_FILE)

    # ----------------------------------------------------------------
    # Step 7: Spearman correlations per regulon vs SASP12_mean
    # ----------------------------------------------------------------
    log_print(f"\nStep 7: Computing Spearman correlations (AUCell vs SASP12)...", LOG_FILE)

    corr_results = []

    for col in aucell_df.columns:
        tf_name = col.replace('(+)', '')
        aucell_col = f'aucell_{col}'

        if aucell_col not in donor_df.columns:
            continue

        valid = donor_df[aucell_col].notna() & donor_df['SASP12_mean'].notna()
        if valid.sum() < 4:
            log_print(f"  WARNING: {tf_name} has only {valid.sum()} valid donors, skipping.", LOG_FILE)
            continue

        # AUCell-SASP correlation
        rho_auc, p_auc = stats.spearmanr(
            donor_df.loc[valid, aucell_col],
            donor_df.loc[valid, 'SASP12_mean'],
        )
        n = int(valid.sum())
        ci_lo_auc, ci_hi_auc = fisher_z_ci(rho_auc, n)

        # Raw mRNA-SASP correlation for comparison
        raw_col = f'raw_{tf_name}'
        if raw_col in donor_df.columns:
            valid_raw = donor_df[raw_col].notna() & donor_df['SASP12_mean'].notna()
            if valid_raw.sum() >= 4:
                rho_raw, p_raw = stats.spearmanr(
                    donor_df.loc[valid_raw, raw_col],
                    donor_df.loc[valid_raw, 'SASP12_mean'],
                )
            else:
                rho_raw, p_raw = np.nan, np.nan
        else:
            rho_raw, p_raw = np.nan, np.nan

        # Regulon source
        source = 'motif' if tf_name in motif_tf_names else 'coexpression'

        corr_results.append({
            'dataset': DATASET_NAME,
            'tf': tf_name,
            'regulon_source': source,
            'n_targets': len(coexpression_regulons.get(tf_name, [])),
            'n_donors': n,
            'aucell_rho': float(rho_auc),
            'aucell_p': float(p_auc),
            'aucell_ci_lo': float(ci_lo_auc),
            'aucell_ci_hi': float(ci_hi_auc),
            'raw_mrna_rho': float(rho_raw) if not np.isnan(rho_raw) else None,
            'raw_mrna_p': float(p_raw) if not np.isnan(p_raw) else None,
            'delta_rho': float(rho_auc - rho_raw) if not np.isnan(rho_raw) else None,
        })

    corr_df = pd.DataFrame(corr_results)
    corr_path = f"{OUTDIR}/d1_correlations_{DATASET_NAME}.csv"
    corr_df.to_csv(corr_path, index=False)
    log_print(f"  Saved: {corr_path} ({len(corr_df)} regulons)", LOG_FILE)

    # ----------------------------------------------------------------
    # Print key results table
    # ----------------------------------------------------------------
    log_print(f"\n  === {DATASET_NAME}: AUCell-SASP Correlations for TARGET_TFS ===", LOG_FILE)
    log_print(f"  {'TF':10s} {'Source':12s} {'n_tgt':>5s} {'n_d':>4s} "
              f"{'AUCell_rho':>11s} {'AUCell_p':>11s} "
              f"{'Raw_rho':>10s} {'Delta':>8s}", LOG_FILE)
    log_print(f"  {'-' * 75}", LOG_FILE)

    for tf in TARGET_TFS:
        sub = corr_df[corr_df['tf'] == tf]
        if len(sub) > 0:
            row = sub.iloc[0]
            raw_str = f"{row['raw_mrna_rho']:.3f}" if row['raw_mrna_rho'] is not None else 'N/A'
            delta_str = f"{row['delta_rho']:+.3f}" if row['delta_rho'] is not None else 'N/A'
            log_print(f"  {tf:10s} {row['regulon_source']:12s} {row['n_targets']:5.0f} "
                      f"{row['n_donors']:4.0f} {row['aucell_rho']:11.3f} "
                      f"{row['aucell_p']:11.2e} {raw_str:>10s} {delta_str:>8s}", LOG_FILE)
        else:
            log_print(f"  {tf:10s} {'no regulon':12s}", LOG_FILE)

    # Print top 10 by AUCell rho
    log_print(f"\n  === Top 10 AUCell-SASP correlations (all regulons) ===", LOG_FILE)
    top10 = corr_df.nlargest(10, 'aucell_rho')
    for _, row in top10.iterrows():
        raw_str = f"{row['raw_mrna_rho']:.3f}" if row['raw_mrna_rho'] is not None else 'N/A'
        delta_str = f"{row['delta_rho']:+.3f}" if row['delta_rho'] is not None else 'N/A'
        log_print(f"  {row['tf']:15s} {row['aucell_rho']:7.3f} (p={row['aucell_p']:.2e}, "
                  f"n={row['n_donors']:.0f}) | raw={raw_str} | delta={delta_str}", LOG_FILE)

    # ----------------------------------------------------------------
    # Summary JSON
    # ----------------------------------------------------------------
    summary = {
        'batch': 'batch_055_d1',
        'script': 'run_fap_aucell.py',
        'date': pd.Timestamp.now().isoformat(),
        'dataset': DATASET_NAME,
        'analysis': 'FAP-only pySCENIC: GRNBoost2 + co-expression regulons + AUCell + SASP correlation',
        'key_metrics': {},
        'design_decisions': [
            'snRNA-only cells for GRNBoost2 (tech confound avoidance)',
            'z-scored data: variance-based HVG selection via dispersions_norm',
            'All expressed TFs from human_tfs.txt for GRNBoost2 (not just TARGET_TFS)',
            'cisTarget SKIPPED (broken in batch_054)',
            'Co-expression regulons: top-80th-percentile importance per TF, min 5, max 200 targets',
            'AUCell on ALL cells (snRNA + scRNA)',
            'Donor-level correlation: min 50 cells per donor',
            'delta_rho = aucell_rho - raw_mrna_rho',
        ],
    }

    # Key metrics for TARGET_TFS
    for tf in TARGET_TFS:
        sub = corr_df[corr_df['tf'] == tf]
        if len(sub) > 0:
            row = sub.iloc[0]
            summary['key_metrics'][tf] = {
                'aucell_rho': float(row['aucell_rho']),
                'aucell_p': float(row['aucell_p']),
                'aucell_ci_95': f"[{row['aucell_ci_lo']:.3f}, {row['aucell_ci_hi']:.3f}]",
                'raw_mrna_rho': float(row['raw_mrna_rho']) if row['raw_mrna_rho'] is not None else None,
                'delta_rho': float(row['delta_rho']) if row['delta_rho'] is not None else None,
                'regulon_source': row['regulon_source'],
                'n_targets': int(row['n_targets']),
                'n_donors': int(row['n_donors']),
            }

    summary['runtime'] = {
        'grnboost2_minutes': t_grn / 60,
        'aucell_minutes': t_auc / 60,
    }

    summary_path = f"{OUTDIR}/d1_summary_{DATASET_NAME}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log_print(f"\n  Summary saved: {summary_path}", LOG_FILE)

    log_print(f"\n=== batch_055 D1 COMPLETE ===", LOG_FILE)
