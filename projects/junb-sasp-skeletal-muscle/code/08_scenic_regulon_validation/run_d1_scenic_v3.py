#!/usr/bin/env python3
"""
batch_054 D1: pySCENIC Regulon Inference and Donor-Level AUCell-SASP Correlation
v3: Addressed 3-critic review feedback from batch_052 failure.

Key design changes from batch_052:
- HVG selection: Use top 5000 HVGs via scanpy (was all ~15K genes). Reduces GRNBoost2
  computation ~3x while keeping the most informative genes for regulon inference.
- All expressed TFs: Run GRNBoost2 with ALL expressed TFs from human_tfs.txt, not just
  the 21 TARGET_TFs. This prevents misattribution of shared targets to the wrong TF.
- cisTarget motif pruning: Proper pipeline with NES>3.0 threshold. Falls back to
  co-expression modules if cisTarget databases are incompatible.
- Single GRNBoost2 run per compartment (snRNA-only). The all-cells run from batch_052
  was removed because it doubled runtime without adding value for the scientific question.

Runtime estimate: ~2-4 hours for all 3 compartments.
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

OUTDIR = "experiments/batch_054"

SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8',
          'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

TARGET_TFS = ['JUNB', 'JUN', 'JUND', 'FOS', 'FOSB', 'FOSL1', 'FOSL2',
              'KLF2', 'KLF4', 'KLF6', 'KLF10', 'ATF3', 'EGR1', 'EGR2',
              'IRF1', 'CEBPB', 'CEBPD', 'RELA', 'NFKB1', 'STAT3', 'CDKN1A']

# cisTarget databases -- use the properly-named files in the project root.
# The copies in databases/ lack the required cisTarget filename convention
# (must end with .genes_vs_motifs.rankings.feather) and will fail to load.
RANKINGS_DBS = [
    "hg38__refseq-r80__500bp_up_and_100bp_down_tss.mc9nr.genes_vs_motifs.rankings.feather",
    "hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather",
]
MOTIF_ANNOTATIONS = "databases/motifs-v9-nr.hgnc-m0.001-o0.0.tbl"
TF_LIST = "databases/human_tfs.txt"

DATASETS = {
    'HLMA_Vascular': {
        'path': 'data/Vascular_scsn_RNA.h5ad',
        'cell_types': ['ArtEC', 'CapEC', 'VenEC', 'IL6+ VenEC'],
        'filter_col': 'Annotation',
    },
    'HLMA_MuSC': {
        'path': 'data/MuSC_scsn_RNA.h5ad',
        'cell_types': None,
        'filter_col': 'Annotation',
    },
    'HLMA_FAP': {
        'path': 'data/OMIX004308-02.h5ad',
        'cell_types': ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP', 'CD99+ FAP'],
        'filter_col': 'Annotation',
    },
}


# ============================================================================
# Utility functions (no side effects at import time)
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


def load_and_filter(path, cell_types=None, filter_col='Annotation'):
    """Load h5ad, optionally filter to specific cell types, return AnnData."""
    import anndata as ad
    print(f"  [{timestamp()}] Loading {path}...")
    adata = ad.read_h5ad(path)

    if cell_types:
        mask = adata.obs[filter_col].isin(cell_types)
        adata = adata[mask].copy()

    print(f"    Shape after filtering: {adata.shape}")
    return adata


def compute_hvg_expression(adata, n_top_genes=5000, human_tfs=None):
    """Select top HVGs and return expression DataFrame for GRNBoost2.

    WHY 5000 HVGs: All ~15K expressed genes with all ~1200 TFs makes GRNBoost2
    O(TFs * genes * n_estimators * n_cells) which is prohibitively slow. The top
    5000 HVGs capture the majority of biological signal while reducing computation
    by ~3x.

    WHY force-include TFs: Many TFs of interest (including JUNB, rank ~11K)
    are NOT in the top 5000 HVGs because TFs tend to be lowly/ubiquitously
    expressed. GRNBoost2 needs TFs in the expression matrix to serve as both
    regulators and potential targets. We force-include all expressed TFs from
    the human TF list, which adds ~600-1200 genes to the 5000 HVGs.

    Data format handling:
    - Vascular and MuSC h5ad files contain log-normalized data (non-negative,
      max ~4-8). We use scanpy's seurat flavor for HVG selection.
    - FAP h5ad file contains z-scored data (negative values, mean ~0). scanpy's
      seurat flavor requires non-negative data, so we fall back to pre-computed
      HVG annotations stored in adata.var['dispersions_norm'].

    Returns: DataFrame (cells x genes) with HGNC symbol column names.
    """
    import scanpy as sc
    import scipy.sparse as sp

    adata_work = adata.copy()

    # Detect data format: if X contains negative values, it's z-scored
    X_sample = adata_work.X[:100, :]
    if sp.issparse(X_sample):
        X_sample = X_sample.toarray()
    has_negative = (X_sample < 0).any()
    fraction_negative = (X_sample < 0).mean()

    if has_negative and fraction_negative > 0.1:
        # Z-scored data (e.g., FAP dataset). Use pre-computed HVG info if available,
        # otherwise use variance-based selection.
        print(f"    [{timestamp()}] Detected z-scored data ({fraction_negative:.1%} negative values). "
              f"Using pre-computed HVG info or variance-based selection.")

        if 'dispersions_norm' in adata_work.var.columns:
            # Use pre-computed dispersions to rank genes
            print(f"    Using pre-computed dispersions_norm from adata.var...")
            top_genes = adata_work.var.nlargest(n_top_genes, 'dispersions_norm').index
            hvg_mask = adata_work.var_names.isin(top_genes)
        else:
            # Fallback: select by variance
            print(f"    No pre-computed HVG info. Selecting by variance...")
            X_all = adata_work.X
            if sp.issparse(X_all):
                X_all = X_all.toarray()
            gene_variances = np.var(X_all, axis=0)
            top_indices = np.argsort(gene_variances)[-n_top_genes:]
            hvg_mask = np.zeros(adata_work.n_vars, dtype=bool)
            hvg_mask[top_indices] = True
    else:
        # Log-normalized data. Use scanpy's seurat flavor.
        print(f"    [{timestamp()}] Computing {n_top_genes} HVGs (flavor='seurat', log-normalized data)...")
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

    WHY mean of detected genes (not z-score): L001 from batch_045 demonstrated
    that z-scoring within donors can introduce artifacts when donor variance is
    small. The raw mean is the canonical methodology established in batch_050.
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

    Returns DataFrame with one row per donor: n_cells, SASP12_mean, raw_{TF},
    aucell_{regulon}.
    """
    import scipy.sparse as sp

    obs = adata.obs.copy()
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

        # Metadata columns
        for col in ['age', 'age_pop', 'Sex', 'sex', 'gender', 'Country', 'country', 'tech']:
            if col in obs.columns:
                val = obs.loc[mask, col]
                row[col] = val.iloc[0] if len(val) > 0 else np.nan

        # Mean SASP12
        row['SASP12_mean'] = float(np.mean(sasp_per_cell[mask.values]))

        # Mean raw mRNA for target TFs
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


def build_coexpression_regulons(adjacencies, tf_list, quantile=0.80, min_targets=5, max_targets=200):
    """Build co-expression regulons from GRNBoost2 adjacencies as fallback.

    Uses top-quartile importance scores per TF. This is the fallback when
    cisTarget motif pruning is unavailable.

    Returns dict: {tf_name: [target_gene, ...]}
    """
    regulons = {}
    for tf_name in tf_list:
        sub = adjacencies[adjacencies['TF'] == tf_name].sort_values('importance', ascending=False)
        if len(sub) == 0:
            continue
        threshold = sub['importance'].quantile(quantile)
        top_targets = sub[sub['importance'] >= threshold]['target'].tolist()
        # Exclude the TF itself from its own regulon targets
        top_targets = [g for g in top_targets if g != tf_name]
        if len(top_targets) >= min_targets:
            regulons[tf_name] = top_targets[:max_targets]
    return regulons


# ============================================================================
# MAIN ANALYSIS
# All executable code is inside this guard for multiprocessing safety.
# Python 3.13 defaults to 'spawn' start method which re-imports the module
# in child processes. The __name__ guard prevents infinite re-execution.
# ============================================================================

if __name__ == '__main__':
    # CRITICAL: Set multiprocessing start method to fork BEFORE importing
    # dask/arboreto. Python 3.13 defaults to 'spawn' which causes the
    # infinite re-execution bug seen in batch_052.
    import multiprocessing
    multiprocessing.set_start_method('fork', force=True)

    os.makedirs(OUTDIR, exist_ok=True)

    print(f"[{timestamp()}] === batch_054 D1 pySCENIC v3 ===")
    print(f"[{timestamp()}] Output directory: {OUTDIR}")

    # Load TF list
    with open(TF_LIST) as f:
        human_tfs = set(line.strip() for line in f if line.strip())
    print(f"[{timestamp()}] Loaded {len(human_tfs)} human TFs from {TF_LIST}")
    print(f"  JUNB in TF list: {'JUNB' in human_tfs}")
    print(f"  Target TFs in list: {sum(1 for tf in TARGET_TFS if tf in human_tfs)}/{len(TARGET_TFS)}")

    # Check which target TF is missing
    missing_tfs = [tf for tf in TARGET_TFS if tf not in human_tfs]
    if missing_tfs:
        print(f"  Missing from TF list: {missing_tfs}")

    # Load cisTarget ranking databases
    print(f"\n[{timestamp()}] Loading cisTarget ranking databases...")
    from ctxcore.rnkdb import FeatherRankingDatabase

    rnk_dbs = []
    cistarget_available = True
    for db_path in RANKINGS_DBS:
        db_name = Path(db_path).stem
        try:
            db = FeatherRankingDatabase(fname=db_path, name=db_name)
            rnk_dbs.append(db)
            print(f"  Loaded: {db_name} ({db.total_genes} genes)")
        except Exception as e:
            print(f"  WARNING: Failed to load {db_path}: {e}")
            cistarget_available = False

    if not rnk_dbs:
        print("  ERROR: No ranking databases loaded. Will fall back to co-expression modules.")
        cistarget_available = False
    else:
        print(f"  {len(rnk_dbs)} ranking databases ready")

    # Create dask client for GRNBoost2.
    #
    # WHY processes=True with fork: GRNBoost2 uses sklearn GBM which is CPU-bound
    # and does NOT release the GIL effectively. Threaded workers suffer from GIL
    # contention. Fork-based processes share memory pages (COW) so the expression
    # matrix is not duplicated in RAM.
    #
    # WHY N_WORKERS = min(16, cpu_count - 4): Each GBM fit uses ~0.5-2 GB peak RAM.
    # With 246 GB available and 4GB/worker limit, 16 workers keeps peak memory
    # well under budget. cpu_count - 4 leaves headroom for the scheduler and OS.
    from distributed import Client, LocalCluster

    N_WORKERS = min(16, os.cpu_count() - 4)

    print(f"\n[{timestamp()}] Creating dask LocalCluster: {N_WORKERS} workers, fork start method...")
    local_cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=True,
        diagnostics_port=None,
        memory_limit='4GB',
    )
    dask_client = Client(local_cluster)
    print(f"  Dask client ready: {dask_client}")

    # Track results across compartments
    all_donor_corrs = []
    all_donor_averages = []
    comp_results = {}

    # ================================================================
    # Process each compartment
    # ================================================================
    for ds_name, ds_info in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"[{timestamp()}] === Processing {ds_name} ===")
        print(f"{'='*70}")

        t_comp_start = time.time()

        try:
            # Initialize tracking variables for the finally block
            n_modules_count = 0
            t_ct_duration = None

            # --------------------------------------------------------
            # Step 1: Load data and split by technology
            # --------------------------------------------------------
            adata = load_and_filter(ds_info['path'], ds_info['cell_types'], ds_info['filter_col'])

            # Tech distribution
            if 'tech' in adata.obs.columns:
                print(f"\n  Tech distribution:")
                for tech_val in sorted(adata.obs['tech'].unique()):
                    sub = adata.obs[adata.obs['tech'] == tech_val]
                    n_donors = sub['sample'].nunique() if 'sample' in sub.columns else sub['donor_id'].nunique() if 'donor_id' in sub.columns else '?'
                    print(f"    {tech_val}: {len(sub)} cells, {n_donors} donors")
            else:
                print("  No 'tech' column found -- using all cells for both inference and scoring")

            # Split: snRNA-only for GRNBoost2 (tech confound avoidance)
            if 'tech' in adata.obs.columns:
                snRNA_mask = adata.obs['tech'] == 'snRNA'
                adata_snrna = adata[snRNA_mask.values].copy()
                print(f"\n  snRNA-only (for GRNBoost2): {adata_snrna.shape}")
                print(f"  All cells (for AUCell scoring): {adata.shape}")
            else:
                adata_snrna = adata.copy()
                print(f"\n  All cells (for both): {adata.shape}")

            # --------------------------------------------------------
            # Step 2: HVG selection on snRNA cells
            # --------------------------------------------------------
            print(f"\n  [{timestamp()}] Step 2: HVG selection (top 5000 + all expressed TFs)...")
            ex_snrna = compute_hvg_expression(adata_snrna, n_top_genes=5000, human_tfs=human_tfs)

            # Filter TFs to those present in the HVG expression matrix
            tf_in_data = [g for g in ex_snrna.columns if g in human_tfs]
            print(f"    TFs present in HVG data: {len(tf_in_data)}")

            # Check target TFs
            target_tfs_present = [tf for tf in TARGET_TFS if tf in ex_snrna.columns]
            target_tfs_missing = [tf for tf in TARGET_TFS if tf not in ex_snrna.columns]
            print(f"    Target TFs present: {len(target_tfs_present)}/{len(TARGET_TFS)}")
            if target_tfs_missing:
                print(f"    Target TFs missing from HVGs: {target_tfs_missing}")
                print(f"    NOTE: These TFs are not in top 5000 HVGs. They may still be scored "
                      f"via AUCell if present in the full expression matrix.")

            # --------------------------------------------------------
            # Step 3: GRNBoost2 on snRNA-only, all expressed TFs x HVGs
            # --------------------------------------------------------
            print(f"\n  [{timestamp()}] Step 3: GRNBoost2 inference...")
            print(f"    Input: {ex_snrna.shape[0]} cells x {ex_snrna.shape[1]} HVGs")
            print(f"    TFs for inference: {len(tf_in_data)}")

            from arboreto.algo import grnboost2

            t_grn_start = time.time()
            adjacencies = grnboost2(
                expression_data=ex_snrna,
                tf_names=tf_in_data,
                client_or_address=dask_client,
                verbose=True,
                seed=42,
            )
            t_grn = time.time() - t_grn_start
            print(f"  [{timestamp()}] GRNBoost2 completed in {t_grn/60:.1f} minutes")
            print(f"    Adjacencies: {len(adjacencies)} TF-target pairs")
            print(f"    Unique TFs with adjacencies: {adjacencies['TF'].nunique()}")

            # Save adjacencies
            adj_path = f"{OUTDIR}/d1_adjacencies_{ds_name}.csv"
            adjacencies.to_csv(adj_path, index=False)
            print(f"    Saved: {adj_path}")

            # --------------------------------------------------------
            # Step 4: cisTarget motif pruning
            # --------------------------------------------------------
            print(f"\n  [{timestamp()}] Step 4: cisTarget motif pruning...")

            motif_regulons = {}  # {tf_name: [(target, NES, motif), ...]}
            motif_pruned_regulon_objects = []  # Regulon objects for AUCell
            coexpression_regulons = {}  # fallback

            if cistarget_available:
                from pyscenic.utils import modules_from_adjacencies
                from pyscenic.prune import prune2df, df2regulons

                # Build co-expression modules from adjacencies.
                # modules_from_adjacencies requires TFs to be in the expression matrix.
                print(f"    [{timestamp()}] Building co-expression modules...")
                modules = modules_from_adjacencies(
                    adjacencies,
                    ex_snrna,
                    thresholds=(0.75, 0.90),
                    top_n_targets=(50,),
                    top_n_regulators=(5, 10, 50),
                    min_genes=20,
                    keep_only_activating=True,
                    rho_threshold=0.03,
                )
                print(f"    Co-expression modules: {len(modules)}")
                n_modules_count = len(modules)

                # Run cisTarget motif pruning
                print(f"    [{timestamp()}] Running cisTarget prune2df (NES > 3.0)...")
                t_ct_start = time.time()

                try:
                    df_motifs = prune2df(
                        rnk_dbs,
                        modules,
                        MOTIF_ANNOTATIONS,
                        rank_threshold=1500,
                        auc_threshold=0.05,
                        nes_threshold=3.0,
                        motif_similarity_fdr=0.001,
                        orthologuous_identity_threshold=0.0,
                        client_or_address='dask_multiprocessing',
                        num_workers=max(1, os.cpu_count() - 4),
                    )
                    t_ct = time.time() - t_ct_start
                    print(f"    [{timestamp()}] cisTarget completed in {t_ct/60:.1f} minutes")
                    t_ct_duration = t_ct / 60

                    # The output has MultiIndex columns.
                    # Columns: (Enrichment, AUC), (Enrichment, NES), (Enrichment, TargetGenes), etc.
                    # Index: module names (TF names).
                    n_motif_regulons = len(df_motifs) if df_motifs is not None else 0
                    print(f"    Motif-enriched regulons (before NES filter): {n_motif_regulons}")

                    if df_motifs is not None and len(df_motifs) > 0:
                        # Filter by NES > 3.0
                        nes_col = ('Enrichment', 'NES')
                        if nes_col in df_motifs.columns:
                            sig_mask = df_motifs[nes_col] > 3.0
                            df_motifs_sig = df_motifs[sig_mask]
                        else:
                            df_motifs_sig = df_motifs
                            print(f"    WARNING: NES column not found in expected location. "
                                  f"Columns: {df_motifs.columns.tolist()}")

                        print(f"    Significant regulons (NES > 3.0): {len(df_motifs_sig)}")

                        # Derive regulon objects from the pruned results.
                        # df2regulons extracts Regulon objects from the prune2df output,
                        # keeping only targets with cis-regulatory footprints.
                        try:
                            motif_pruned_regulon_objects = df2regulons(df_motifs_sig)
                            print(f"    Derived {len(motif_pruned_regulon_objects)} motif-pruned regulon objects")
                        except Exception as e:
                            print(f"    df2regulons failed: {e}")
                            print(f"    Attempting manual regulon extraction...")
                            motif_pruned_regulon_objects = []

                        # Extract target gene lists for reporting
                        target_genes_col = ('Enrichment', 'TargetGenes')
                        for idx_row in df_motifs_sig.index:
                            tf_name = idx_row if isinstance(idx_row, str) else str(idx_row)
                            row_data = df_motifs_sig.loc[idx_row]
                            nes_val = row_data[nes_col] if nes_col in df_motifs_sig.columns else np.nan
                            targets = row_data[target_genes_col] if target_genes_col in df_motifs_sig.columns else []

                            # targets may be a list of tuples (gene, weight) or GeneSignature
                            if hasattr(targets, 'genes'):
                                target_list = list(targets.genes)
                            elif isinstance(targets, (list, np.ndarray)):
                                target_list = [t[0] if isinstance(t, (tuple, list)) else str(t) for t in targets]
                            else:
                                target_list = []

                            motif_regulons[tf_name] = target_list

                        # Log target counts for key TFs
                        print(f"\n    Motif-pruned regulon sizes for TARGET_TFs:")
                        for tf in TARGET_TFS:
                            if tf in motif_regulons:
                                n_targets = len(motif_regulons[tf])
                                print(f"      {tf}: {n_targets} motif-pruned targets")
                                # KLF10 repressor caveat
                                if tf == 'KLF10':
                                    print(f"      KLF10 CAVEAT: These are co-expressed genes passing motif enrichment. "
                                          f"KLF10 is a transcriptional repressor; targets listed here are co-expressed, "
                                          f"not necessarily direct repression targets. Activating vs repressive "
                                          f"function requires experimental validation.")
                            else:
                                print(f"      {tf}: no motif-pruned regulon (below NES threshold)")

                        # Save regulon table
                        regulon_rows = []
                        for idx_row in df_motifs_sig.index:
                            tf_name = idx_row if isinstance(idx_row, str) else str(idx_row)
                            row_data = df_motifs_sig.loc[idx_row]
                            nes_val = float(row_data[nes_col]) if nes_col in df_motifs_sig.columns else np.nan
                            auc_val = float(row_data[('Enrichment', 'AUC')]) if ('Enrichment', 'AUC') in df_motifs_sig.columns else np.nan
                            motif_val = str(row_data[('Enrichment', 'Annotation')]) if ('Enrichment', 'Annotation') in df_motifs_sig.columns else ''

                            targets = row_data[target_genes_col] if target_genes_col in df_motifs_sig.columns else []
                            if hasattr(targets, 'genes'):
                                target_list = list(targets.genes)
                            elif isinstance(targets, (list, np.ndarray)):
                                target_list = [t[0] if isinstance(t, (tuple, list)) else str(t) for t in targets]
                            else:
                                target_list = []

                            for gene in target_list:
                                regulon_rows.append({
                                    'TF': tf_name,
                                    'target': gene,
                                    'NES': nes_val,
                                    'AUC': auc_val,
                                    'motif': motif_val,
                                })

                        regulon_df = pd.DataFrame(regulon_rows)
                        reg_path = f"{OUTDIR}/d1_regulons_{ds_name}.csv"
                        regulon_df.to_csv(reg_path, index=False)
                        print(f"    Saved regulon table: {reg_path} ({len(regulon_df)} rows)")

                    else:
                        print(f"    No motif-enriched regulons found.")

                except Exception as e:
                    print(f"    cisTarget prune2df FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"    Falling back to co-expression modules.")
                    motif_pruned_regulon_objects = []
            else:
                print(f"    cisTarget databases unavailable. Skipping motif pruning.")

            # --------------------------------------------------------
            # Step 4b: Build fallback co-expression regulons
            # --------------------------------------------------------
            # For TFs that have no motif-pruned regulon, we use co-expression
            # modules (top-quartile importance from GRNBoost2 adjacencies).
            print(f"\n  [{timestamp()}] Step 4b: Building co-expression regulon fallback...")
            coexpression_regulons = build_coexpression_regulons(
                adjacencies,
                tf_in_data,
                quantile=0.80,
                min_targets=5,
                max_targets=200,
            )
            print(f"    Co-expression regulons: {len(coexpression_regulons)}")

            # Determine which TFs need co-expression fallback
            tfs_with_motif = set(motif_regulons.keys())
            tfs_needing_fallback = set(coexpression_regulons.keys()) - tfs_with_motif
            print(f"    TFs with motif-pruned regulons: {len(tfs_with_motif)}")
            print(f"    TFs using co-expression fallback: {len(tfs_needing_fallback)}")

            for tf in TARGET_TFS:
                source = 'motif' if tf in tfs_with_motif else ('coexpression' if tf in coexpression_regulons else 'none')
                n_tgts = len(motif_regulons.get(tf, coexpression_regulons.get(tf, [])))
                print(f"      {tf}: {source}, {n_tgts} targets")

            # --------------------------------------------------------
            # Step 5: AUCell scoring on ALL cells
            # --------------------------------------------------------
            print(f"\n  [{timestamp()}] Step 5: AUCell scoring on ALL cells...")

            from pyscenic.aucell import aucell
            from ctxcore.genesig import GeneSignature

            # Build gene signatures for AUCell.
            # Priority: motif-pruned regulons > co-expression fallback.
            # Each signature is a GeneSignature with gene weights from GRNBoost2 importance.
            signatures = []

            # First, add motif-pruned regulon objects directly
            for reg in motif_pruned_regulon_objects:
                signatures.append(reg)

            # Track which TF names are already covered by motif-pruned regulons
            motif_tf_names = set()
            for reg in motif_pruned_regulon_objects:
                motif_tf_names.add(reg.transcription_factor)

            # Add co-expression regulons for TFs NOT covered by motif pruning
            for tf_name, targets in coexpression_regulons.items():
                if tf_name in motif_tf_names:
                    continue
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

            # Ensure all TARGET_TFs have a signature (even if not in HVGs)
            target_tfs_in_signatures = set()
            for sig in signatures:
                name = sig.name.replace('(+)', '')
                target_tfs_in_signatures.add(name)

            missing_from_sigs = set(TARGET_TFS) - target_tfs_in_signatures
            if missing_from_sigs:
                print(f"    WARNING: TARGET_TFs missing from regulons: {missing_from_sigs}")
                print(f"    These TFs were not in the top 5000 HVGs and had no GRNBoost2 output.")

            print(f"    Total regulon signatures for AUCell: {len(signatures)}")

            if len(signatures) == 0:
                print(f"    ERROR: No regulon signatures. Skipping AUCell for {ds_name}.")
                raise RuntimeError(f"No regulon signatures for {ds_name}")

            # Prepare expression matrix for AUCell (ALL cells, ALL genes)
            import scipy.sparse as sp

            X_all = adata.X
            if sp.issparse(X_all):
                X_all = X_all.toarray()
            ex_all_df = pd.DataFrame(
                X_all,
                columns=list(adata.var_names),
                index=adata.obs_names,
            )

            print(f"    Running AUCell: {ex_all_df.shape[0]} cells x {len(signatures)} regulons...")
            t_auc_start = time.time()

            auc_threshold = 0.05  # Default: top 5% of ranked genes
            aucell_df = aucell(
                exp_mtx=ex_all_df,
                signatures=signatures,
                auc_threshold=auc_threshold,
                noweights=False,
                normalize=False,
                seed=42,
                num_workers=max(1, os.cpu_count() - 4),
            )

            t_auc = time.time() - t_auc_start
            print(f"  [{timestamp()}] AUCell completed in {t_auc/60:.1f} minutes")
            print(f"    AUCell matrix: {aucell_df.shape}")

            # Save AUCell scores
            auc_path = f"{OUTDIR}/d1_aucell_{ds_name}.csv"
            aucell_df.to_csv(auc_path)
            print(f"    Saved: {auc_path}")

            # Clean up large expression matrices
            del X_all, ex_all_df, ex_snrna

            # --------------------------------------------------------
            # Step 6: Donor-level averages and correlations
            # --------------------------------------------------------
            print(f"\n  [{timestamp()}] Step 6: Donor-level averages and correlations...")

            donor_df = compute_donor_averages(adata, aucell_df, SASP12, TARGET_TFS)
            donor_path = f"{OUTDIR}/d1_donor_averages_{ds_name}.csv"
            donor_df.to_csv(donor_path, index=False)
            print(f"    Donors: {len(donor_df)}")
            print(f"    Saved: {donor_path}")

            # AUCell-SASP correlation per regulon
            corr_results = []
            for col in aucell_df.columns:
                tf_name = col.replace('(+)', '')
                aucell_col = f'aucell_{col}'

                if aucell_col not in donor_df.columns:
                    continue

                valid = donor_df[aucell_col].notna() & donor_df['SASP12_mean'].notna()
                if valid.sum() < 4:
                    continue

                rho, p = stats.spearmanr(
                    donor_df.loc[valid, aucell_col],
                    donor_df.loc[valid, 'SASP12_mean'],
                )
                n = int(valid.sum())
                ci_lo, ci_hi = fisher_z_ci(rho, n)

                # Raw mRNA correlation for comparison
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

                # Determine regulon source
                source = 'motif' if tf_name in motif_tf_names else 'coexpression'

                corr_results.append({
                    'dataset': ds_name,
                    'tf': tf_name,
                    'regulon_source': source,
                    'n_targets': len(motif_regulons.get(tf_name, coexpression_regulons.get(tf_name, []))),
                    'n_donors': n,
                    'aucell_rho': float(rho),
                    'aucell_p': float(p),
                    'aucell_ci_lo': float(ci_lo),
                    'aucell_ci_hi': float(ci_hi),
                    'raw_mrna_rho': float(rho_raw) if not np.isnan(rho_raw) else None,
                    'raw_mrna_p': float(p_raw) if not np.isnan(p_raw) else None,
                    'delta_rho': float(rho - rho_raw) if not np.isnan(rho_raw) else None,
                })

            corr_df = pd.DataFrame(corr_results)
            corr_path = f"{OUTDIR}/d1_correlations_{ds_name}.csv"
            corr_df.to_csv(corr_path, index=False)
            all_donor_corrs.append(corr_df)
            all_donor_averages.append(donor_df)

            # Print key results
            print(f"\n    === {ds_name}: AUCell-SASP Correlations for TARGET_TFs ===")
            print(f"    {'TF':10s} {'Source':12s} {'n_tgt':>5s} {'AUCell_rho':>10s} "
                  f"{'AUCell_p':>10s} {'Raw_rho':>10s} {'Delta':>8s}")
            print(f"    {'-'*65}")
            for tf in TARGET_TFS:
                sub = corr_df[corr_df['tf'] == tf]
                if len(sub) > 0:
                    row = sub.iloc[0]
                    raw_str = f"{row['raw_mrna_rho']:.3f}" if row['raw_mrna_rho'] is not None else 'N/A'
                    delta_str = f"{row['delta_rho']:+.3f}" if row['delta_rho'] is not None else 'N/A'
                    print(f"    {tf:10s} {row['regulon_source']:12s} {row['n_targets']:5.0f} "
                          f"{row['aucell_rho']:10.3f} {row['aucell_p']:10.2e} "
                          f"{raw_str:>10s} {delta_str:>8s}")
                else:
                    print(f"    {tf:10s} {'no regulon':12s}")

            # Store compartment results
            comp_results[ds_name] = {
                'status': 'SUCCESS',
                'n_cells_total': int(adata.shape[0]),
                'n_cells_snrna': int(adata_snrna.shape[0]),
                'n_hvgs': 5000,
                'n_tfs_inferred': len(tf_in_data),
                'n_adjacencies': len(adjacencies),
                'n_modules': n_modules_count,
                'n_motif_regulons': len(motif_regulons),
                'n_coexpression_regulons': len(coexpression_regulons),
                'n_signatures_aucell': len(signatures),
                'grnboost2_minutes': t_grn / 60,
                'cistarget_minutes': t_ct_duration,
                'aucell_minutes': t_auc / 60,
            }

            # KLF10 repressor caveat log
            if 'KLF10' in motif_regulons:
                print(f"\n    KLF10 REPRESSOR CAVEAT: KLF10 motif-pruned regulon has "
                      f"{len(motif_regulons['KLF10'])} targets. KLF10 is a known transcriptional "
                      f"repressor. Targets listed are co-expressed genes passing motif enrichment "
                      f"(NES > 3.0). These may be indirect targets or genes repressed in the same "
                      f"cells where KLF10 is active. Activating vs repressive function requires "
                      f"experimental validation (e.g., ChIP-seq).")

        except Exception as e:
            print(f"\n  [{timestamp()}] ERROR processing {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            comp_results[ds_name] = {
                'status': 'FAILED',
                'error': str(e),
            }

        finally:
            # Clean up large objects to free memory between compartments.
            # Using locals() dict to delete by string name since these are
            # local variables inside the for loop's try block.
            import gc
            _to_clean = ['adata', 'adata_snrna', 'adjacencies', 'aucell_df',
                          'donor_df', 'df_motifs', 'df_motifs_sig', 'ex_snrna',
                          'ex_all_df', 'X_all', 'modules']
            _local_vars = locals()
            for _v in _to_clean:
                if _v in _local_vars:
                    del _local_vars[_v]
            gc.collect()

            # Save checkpoint after each compartment
            if all_donor_corrs:
                checkpoint_corrs = pd.concat(all_donor_corrs)
                checkpoint_corrs.to_csv(f"{OUTDIR}/d1_correlations_all.csv", index=False)
                print(f"  [{timestamp()}] Checkpoint saved: d1_correlations_all.csv "
                      f"({len(checkpoint_corrs)} rows)")

        t_comp = time.time() - t_comp_start
        print(f"\n  [{timestamp()}] {ds_name} completed in {t_comp/60:.1f} minutes")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"[{timestamp()}] === D1 SCENIC SUMMARY ===")
    print(f"{'='*70}")

    summary = {
        'batch': 'batch_054_d1',
        'script': 'run_d1_scenic_v3.py',
        'date': pd.Timestamp.now().isoformat(),
        'analysis': 'D1 pySCENIC regulon inference (GRNBoost2 + cisTarget + AUCell)',
        'design_changes': [
            'HVG selection (top 5000 via scanpy, not all genes)',
            'All expressed TFs for GRNBoost2 (not just 21 TARGET_TFs)',
            'cisTarget motif pruning with NES > 3.0',
            'Fallback to co-expression for TFs without motif support',
            'Single GRNBoost2 run per compartment (snRNA-only)',
        ],
        'compartments': comp_results,
        'key_findings': {},
    }

    if all_donor_corrs:
        all_corrs = pd.concat(all_donor_corrs)
        all_corrs.to_csv(f"{OUTDIR}/d1_correlations_all.csv", index=False)
        print(f"\n  Combined correlations: {len(all_corrs)} TF-compartment pairs")

        # Per-dataset top results
        for ds in all_corrs['dataset'].unique():
            sub = all_corrs[all_corrs['dataset'] == ds]
            sub_sorted = sub.sort_values('aucell_rho', ascending=False)
            print(f"\n  {ds} - Top 10 AUCell-SASP TFs:")
            print(f"    {'TF':10s} {'Source':12s} {'AUCell_rho':>10s} {'Raw_rho':>10s} {'Delta':>8s} {'p':>10s}")
            print(f"    {'-'*60}")
            for _, row in sub_sorted.head(10).iterrows():
                raw_str = f"{row['raw_mrna_rho']:.3f}" if row['raw_mrna_rho'] is not None else 'N/A'
                delta_str = f"{row['delta_rho']:+.3f}" if row['delta_rho'] is not None else 'N/A'
                print(f"    {row['tf']:10s} {row['regulon_source']:12s} {row['aucell_rho']:10.3f} "
                      f"{raw_str:>10s} {delta_str:>8s} {row['aucell_p']:10.2e}")

            # JUNB decision rule
            junb = sub[sub['tf'] == 'JUNB']
            if len(junb) > 0:
                j = junb.iloc[0]
                summary['key_findings'][f'{ds}_JUNB'] = {
                    'aucell_rho': float(j['aucell_rho']),
                    'aucell_p': float(j['aucell_p']),
                    'aucell_ci_95': f"[{j['aucell_ci_lo']:.3f}, {j['aucell_ci_hi']:.3f}]",
                    'raw_mrna_rho': float(j['raw_mrna_rho']) if j['raw_mrna_rho'] is not None else None,
                    'delta_rho': float(j['delta_rho']) if j['delta_rho'] is not None else None,
                    'regulon_source': j['regulon_source'],
                    'n_targets': int(j['n_targets']),
                }
                print(f"\n    JUNB decision ({ds}):")
                if j['aucell_rho'] >= 0.60:
                    print(f"      VALIDATED (rho={j['aucell_rho']:.3f} >= 0.60)")
                elif j['aucell_rho'] >= 0.30:
                    print(f"      INCONCLUSIVE (rho={j['aucell_rho']:.3f}, 0.30-0.60)")
                else:
                    print(f"      NOT VALIDATED (rho={j['aucell_rho']:.3f} < 0.30)")

            # Report all TARGET_TFs for this compartment
            target_corrs = sub[sub['tf'].isin(TARGET_TFS)].sort_values('aucell_rho', ascending=False)
            summary['key_findings'][f'{ds}_all_targets'] = target_corrs[['tf', 'aucell_rho', 'aucell_p', 'raw_mrna_rho', 'delta_rho', 'regulon_source']].to_dict('records')

        # Cross-compartment comparison for JUNB
        print(f"\n  Cross-compartment JUNB comparison:")
        junb_all = all_corrs[all_corrs['tf'] == 'JUNB']
        for _, row in junb_all.iterrows():
            raw_junb_str = f"{row['raw_mrna_rho']:.3f}" if row['raw_mrna_rho'] is not None else 'N/A'
            print(f"    {row['dataset']:20s}: AUCell rho={row['aucell_rho']:.3f}, "
                  f"raw rho={raw_junb_str}, "
                  f"source={row['regulon_source']}")

    else:
        print("  No correlation results produced.")
        summary['status'] = 'NO_RESULTS'

    # Save summary
    summary_path = f"{OUTDIR}/d1_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved summary: {summary_path}")

    # Shutdown dask
    print(f"\n[{timestamp()}] Shutting down dask...")
    dask_client.close()
    local_cluster.close()

    print(f"\n[{timestamp()}] === batch_054 D1 COMPLETE ===")
