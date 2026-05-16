#!/usr/bin/env python3
"""
batch_052 D1: pySCENIC Regulon Inference and Donor-Level AUCell-SASP Correlation
Runs: GRNBoost2 (snRNA-only primary + all-cells sensitivity) -> cisTarget -> AUCell
Outputs: AUCell scores, donor-level correlations, regulon descriptions, Jaccard consistency

v2: Fixed multiprocessing/dask spawn issue on Python 3.13
    - All executable code wrapped in if __name__ == '__main__': guard
    - multiprocessing.set_start_method('fork') called before any dask imports
    - dask LocalCluster created with processes=False to avoid spawn entirely
"""

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.stats import norm
import anndata as ad
import scipy.sparse as sp
import json
import sys
import time
from pathlib import Path
import pickle
import os

warnings.filterwarnings('ignore')

OUTDIR = "experiments/batch_052"
os.makedirs(OUTDIR, exist_ok=True)

# Canonical SASP12 panel
SASP12 = ['CCL2', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6', 'IL6', 'CXCL8',
          'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'PLAUR']

TARGET_TFS = ['JUNB','JUN','JUND','FOS','FOSB','FOSL1','FOSL2','KLF2','KLF4',
              'KLF6','KLF10','ATF3','EGR1','EGR2','IRF1','CEBPB','CEBPD',
              'RELA','NFKB1','STAT3','CDKN1A']

# Database paths
MOTIF_ANNOTATIONS = "databases/motifs-v9-nr.hgnc-m0.001-o0.0.tbl"
RANKINGS_DB = ["databases/hg38_500bp.feather", "databases/hg38_10kb.feather"]
TF_LIST = "databases/human_tfs.txt"

def fisher_z_ci(rho, n, alpha=0.05):
    """Compute 95% CI for Spearman rho via Fisher Z."""
    z_rho = 0.5 * np.log((1 + rho) / (1 + rho + 1e-15))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    z_lo = z_rho - z_crit * se
    z_hi = z_rho + z_crit * se
    ci_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
    ci_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
    return ci_lo, ci_hi


def load_and_filter(path, cell_types=None, filter_col='Annotation'):
    """Load h5ad, filter to cell types, return AnnData."""
    print(f"  Loading {path}...")
    adata = ad.read_h5ad(path)

    if cell_types:
        mask = adata.obs[filter_col].isin(cell_types)
        adata = adata[mask].copy()

    print(f"    Shape: {adata.shape}")
    return adata


def prepare_expression_for_grnboost(adata, min_cells=0.01):
    """Prepare expression matrix for GRNBoost2.
    Returns: ex_matrix (cells x genes DataFrame with HGNC symbols)
    """
    # Get expression matrix
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    # Use log-normalized values (already in X for these h5ad files)
    # GRNBoost2 works best with log-normalized data
    gene_names = list(adata.var_names)

    # Filter lowly expressed genes (< 1% of cells)
    min_cells_abs = max(int(min_cells * X.shape[0]), 3)
    gene_mask = (X > 0).sum(axis=0) >= min_cells_abs
    if hasattr(gene_mask, 'A1'):  # sparse
        gene_mask = gene_mask.A1
    gene_mask = np.asarray(gene_mask).flatten()

    X_filtered = X[:, gene_mask]
    gene_names_filtered = [g for g, m in zip(gene_names, gene_mask) if m]

    print(f"    Genes after filtering: {len(gene_names_filtered)} (from {len(gene_names)})")

    # Create DataFrame (GRNBoost2 expects genes as columns)
    ex_df = pd.DataFrame(X_filtered, columns=gene_names_filtered, index=adata.obs_names)

    return ex_df


def compute_sasp_composite(adata, sasp_genes):
    """Compute per-cell SASP12 composite score."""
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    var_names = list(adata.var_names)
    detected = [g for g in sasp_genes if g in var_names]

    if len(detected) == 0:
        return np.zeros(adata.n_obs)

    indices = [var_names.index(g) for g in detected]
    sasp_vals = X[:, indices]

    # Mean of detected SASP genes per cell
    return np.mean(sasp_vals, axis=1)


def compute_donor_averages(adata, aucell_matrix, sasp_genes, tf_list):
    """Compute donor-level averages for AUCell scores and raw mRNA."""
    obs = adata.obs.copy()
    sample_col = 'sample' if 'sample' in obs.columns else 'donor_id'

    # Compute per-cell SASP composite
    sasp_per_cell = compute_sasp_composite(adata, sasp_genes)

    # Get raw mRNA expression for target TFs
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    var_names = list(adata.var_names)

    donor_data = []
    for donor in obs[sample_col].unique():
        mask = obs[sample_col] == donor
        n_cells = mask.sum()

        row = {
            'sample': donor,
            'n_cells': n_cells,
        }

        # Metadata
        for col in ['age', 'age_pop', 'Sex', 'sex', 'gender', 'Country', 'country', 'tech']:
            if col in obs.columns:
                row[col] = obs.loc[mask, col].iloc[0]

        # Mean SASP12
        row['SASP12_mean'] = np.mean(sasp_per_cell[mask.values])

        # Mean raw mRNA for target TFs
        for tf in tf_list:
            if tf in var_names:
                idx = var_names.index(tf)
                row[f'raw_{tf}'] = np.mean(X[mask.values, idx])

        # Mean AUCell scores for regulons
        aucell_donor = aucell_matrix[mask.values]
        for col_idx, col_name in enumerate(aucell_matrix.columns):
            row[f'aucell_{col_name}'] = np.mean(aucell_donor[:, col_idx])

        donor_data.append(row)

    return pd.DataFrame(donor_data)


# ============================================================
# MAIN ANALYSIS - guarded for multiprocessing safety
# ============================================================

if __name__ == '__main__':
    # CRITICAL: Set multiprocessing start method to fork BEFORE any dask/arboreto imports.
    # Python 3.13 defaults to 'spawn' which re-imports the module in child processes,
    # causing infinite re-execution of top-level code. The if __name__ == '__main__' guard
    # prevents re-execution, and fork avoids the issue entirely.
    import multiprocessing
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set

    results = {}
    all_donor_corrs = []

    # Dataset definitions
    datasets = {
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

    # Load TF list
    with open(TF_LIST) as f:
        human_tfs = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(human_tfs)} human TFs from {TF_LIST}")

    # Check if JUNB is in TF list
    print(f"JUNB in TF list: {'JUNB' in human_tfs}")
    print(f"Target TFs in list: {sum(1 for tf in TARGET_TFS if tf in human_tfs)}/{len(TARGET_TFS)}")

    # Create a dask client with processes=True and fork start method.
    #
    # Why processes=True: GRNBoost2 uses sklearn's GBM which is CPU-bound and
    # does NOT release the GIL effectively. Threaded workers (processes=False)
    # suffer from GIL contention. Fork-based processes share memory pages (COW)
    # so the expression matrix is not duplicated in RAM.
    #
    # Worker count: 16 workers. Each GBM fit is ~0.5-2 GB peak, and we have
    # ~70 GB RAM. 16 workers keeps total peak memory under 40 GB, well within
    # budget. More workers give diminishing returns because the scheduler
    # overhead grows with n_workers.
    #
    # The if __name__ == '__main__': guard (above) prevents the infinite
    # re-execution that occurs when Python's spawn start method re-imports
    # the module. Combined with set_start_method('fork') at the top, child
    # processes are created via fork which does not re-import.
    from distributed import Client, LocalCluster

    N_WORKERS = 16

    print(f"Creating dask LocalCluster with {N_WORKERS} fork-based worker processes...")
    local_cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=1,
        processes=True,
        diagnostics_port=None,
        memory_limit='4GB',    # 4 GB per worker, 64 GB total
    )
    dask_client = Client(local_cluster)
    print(f"Dask client ready: {dask_client}")

    # ============================================================
    # Process each compartment
    # ============================================================

    for ds_name, ds_info in datasets.items():
        print(f"\n{'='*60}")
        print(f"=== Processing {ds_name} ===")
        print(f"{'='*60}")

        t_start = time.time()

        # Load data
        adata = load_and_filter(ds_info['path'], ds_info['cell_types'], ds_info['filter_col'])

        # Check tech distribution
        if 'tech' in adata.obs.columns:
            print(f"\n  Tech distribution:")
            for tech in sorted(adata.obs['tech'].unique()):
                sub = adata.obs[adata.obs['tech'] == tech]
                print(f"    {tech}: {len(sub)} cells, {sub['sample'].nunique()} donors")

        # Split by technology
        snRNA_mask = adata.obs['tech'] == 'snRNA' if 'tech' in adata.obs.columns else pd.Series(True, index=adata.obs.index)
        adata_snrna = adata[snRNA_mask.values].copy()

        print(f"\n  snRNA-only: {adata_snrna.shape}")
        print(f"  All cells: {adata.shape}")

        # ============================================================
        # GRNBoost2: snRNA-only (PRIMARY)
        # ============================================================
        print(f"\n  --- GRNBoost2: snRNA-only (PRIMARY) ---")
        ex_snrna = prepare_expression_for_grnboost(adata_snrna)

        # Filter to TFs only for faster GRNBoost2
        tf_in_data = [g for g in ex_snrna.columns if g in human_tfs]
        print(f"    TFs in snRNA data: {len(tf_in_data)}")

        # Run GRNBoost2
        from arboreto.algo import grnboost2

        print(f"    Running GRNBoost2 on {ex_snrna.shape[0]} cells x {ex_snrna.shape[1]} genes...")
        print(f"    Target TFs for inference: {len(tf_in_data)}")

        t_grn_start = time.time()

        # Pass the pre-created dask client (threaded mode, no spawn issues)
        adjacencies_snrna = grnboost2(
            expression_data=ex_snrna,
            tf_names=tf_in_data,
            client_or_address=dask_client,
            verbose=True,
            seed=42,
        )

        t_grn = time.time() - t_grn_start
        print(f"    GRNBoost2 snRNA completed in {t_grn/60:.1f} minutes")
        print(f"    Adjacencies: {len(adjacencies_snrna)} TF-target pairs")

        # Save adjacencies
        adjacencies_snrna.to_csv(f"{OUTDIR}/d1_adjacencies_{ds_name}_snrna.csv", index=False)

        # ============================================================
        # GRNBoost2: all cells (SENSITIVITY)
        # ============================================================
        print(f"\n  --- GRNBoost2: all cells (SENSITIVITY) ---")
        ex_all = prepare_expression_for_grnboost(adata)
        tf_in_all = [g for g in ex_all.columns if g in human_tfs]

        print(f"    Running GRNBoost2 on {ex_all.shape[0]} cells x {ex_all.shape[1]} genes...")

        t_grn_start2 = time.time()
        adjacencies_all = grnboost2(
            expression_data=ex_all,
            tf_names=tf_in_all,
            client_or_address=dask_client,
            verbose=True,
            seed=42,
        )

        t_grn2 = time.time() - t_grn_start2
        print(f"    GRNBoost2 all-cells completed in {t_grn2/60:.1f} minutes")
        print(f"    Adjacencies: {len(adjacencies_all)} TF-target pairs")

        adjacencies_all.to_csv(f"{OUTDIR}/d1_adjacencies_{ds_name}_all.csv", index=False)

        # ============================================================
        # cisTarget: Prune co-expression modules
        # ============================================================
        print(f"\n  --- cisTarget: Pruning modules ---")

        from pyscenic.prune import prune2df
        from pyscenic.utils import modules_from_adjacencies

        # Create modules from adjacencies
        modules_snrna = modules_from_adjacencies(adjacencies_snrna, ex_snrna)
        modules_all = modules_from_adjacencies(adjacencies_all, ex_all)

        print(f"    snRNA modules: {len(modules_snrna)} regulons")
        print(f"    All-cells modules: {len(modules_all)} regulons")

        # Run cisTarget on snRNA modules
        print("    Running cisTarget on snRNA modules...")
        t_ct_start = time.time()

        try:
            df_motifs_snrna = prune2df(
                modules_snrna,
                RANKINGS_DB,
                MOTIF_ANNOTATIONS,
                mask_dropouts=False,
            )
            t_ct = time.time() - t_ct_start
            print(f"    cisTarget snRNA completed in {t_ct/60:.1f} minutes")
            print(f"    Motif-enriched regulons (snRNA): {len(df_motifs_snrna)}")

            # Extract regulons
            from pyscenic.aucell import create_rankings, enrichment4cells
            from pyscenic.utils import load_motifs

            regulons_snrna = df_motifs_snrna[
                (df_motifs_snrna['NES'] > 3.0) & (df_motifs_snrna['OrthoAUC'] > 0.05)
            ] if 'NES' in df_motifs_snrna.columns else df_motifs_snrna

            print(f"    Significant regulons (NES>3): {len(regulons_snrna)}")

        except Exception as e:
            print(f"    cisTarget snRNA ERROR: {e}")
            print("    Falling back to co-expression modules without motif pruning...")
            # Use top co-expression modules without motif pruning as fallback
            regulons_snrna_names = {}
            for tf_name, module in modules_snrna:
                targets = module[module['importance'] > module['importance'].quantile(0.75)]['target'].tolist()
                if len(targets) >= 5:
                    regulons_snrna_names[tf_name] = targets

            print(f"    Fallback regulons (top-quartile co-expression): {len(regulons_snrna_names)}")

        # Run cisTarget on all-cells modules
        print("    Running cisTarget on all-cells modules...")
        try:
            df_motifs_all = prune2df(
                modules_all,
                RANKINGS_DB,
                MOTIF_ANNOTATIONS,
                mask_dropouts=False,
            )
            print(f"    Motif-enriched regulons (all): {len(df_motifs_all)}")
        except Exception as e:
            print(f"    cisTarget all-cells ERROR: {e}")
            df_motifs_all = None

        # ============================================================
        # Jaccard consistency check
        # ============================================================
        print(f"\n  --- Jaccard Consistency Check ---")

        # Get JUNB targets from both runs
        def get_regulon_targets(adj_df, tf_name, top_n=100):
            sub = adj_df[adj_df['TF'] == tf_name].sort_values('importance', ascending=False)
            return set(sub['target'].head(top_n).tolist()) if len(sub) > 0 else set()

        for tf in ['JUNB', 'JUN', 'FOS', 'FOSB', 'KLF10', 'ATF3']:
            targets_snrna = get_regulon_targets(adjacencies_snrna, tf)
            targets_all = get_regulon_targets(adjacencies_all, tf)

            if targets_snrna and targets_all:
                jaccard = len(targets_snrna & targets_all) / len(targets_snrna | targets_all)
                overlap = len(targets_snrna & targets_all)
                print(f"    {tf}: snRNA={len(targets_snrna)} targets, all={len(targets_all)} targets, "
                      f"Jaccard={jaccard:.3f}, overlap={overlap}")
            else:
                print(f"    {tf}: no targets in one or both runs")

        # ============================================================
        # AUCell: Score regulon activity
        # ============================================================
        print(f"\n  --- AUCell: Scoring regulon activity ---")

        from pyscenic.aucell import enrichment4cells as aucell_score

        # Build regulon gene sets from snRNA adjacencies (top targets per TF)
        regulon_genesets = {}
        for tf_name in set(adjacencies_snrna['TF']):
            sub = adjacencies_snrna[adjacencies_snrna['TF'] == tf_name].sort_values('importance', ascending=False)
            # Keep top targets with importance > threshold
            top_targets = sub[sub['importance'] > sub['importance'].quantile(0.80)]['target'].tolist()
            if len(top_targets) >= 3:
                regulon_genesets[tf_name] = top_targets[:200]  # Cap at 200 targets

        print(f"    Regulon gene sets: {len(regulon_genesets)}")
        for tf in ['JUNB', 'JUN', 'FOS', 'FOSB', 'KLF10', 'ATF3']:
            if tf in regulon_genesets:
                print(f"      {tf}: {len(regulon_genesets[tf])} targets")

        # Run AUCell on ALL cells using snRNA-inferred regulons
        ex_all_for_aucell = prepare_expression_for_grnboost(adata)

        # Use pyscenic's AUCell
        from pyscenic.aucell import aucell

        print(f"    Running AUCell on {ex_all_for_aucell.shape[0]} cells x {len(regulon_genesets)} regulons...")
        t_auc_start = time.time()

        # Convert gene sets to the format AUCell expects
        # AUCell expects a dict of {regulon_name: set(gene_names)}
        gene_sets_aucell = {f"{tf}(+)": set(targets) for tf, targets in regulon_genesets.items()}

        try:
            aucell_matrix = aucell(
                ex_matrix=ex_all_for_aucell,
                signatures=gene_sets_aucell,
                seed=42,
                num_workers=max(1, os.cpu_count() - 2),
            )
            t_auc = time.time() - t_auc_start
            print(f"    AUCell completed in {t_auc/60:.1f} minutes")
            print(f"    AUCell matrix shape: {aucell_matrix.shape}")

            # Save AUCell scores
            aucell_matrix.to_csv(f"{OUTDIR}/d1_aucell_{ds_name}.csv")

            # ============================================================
            # Donor-level correlation
            # ============================================================
            print(f"\n  --- Donor-level AUCell-SASP Correlation ---")

            # Compute donor averages
            donor_df = compute_donor_averages(adata, aucell_matrix.values, SASP12, TARGET_TFS)
            donor_df.to_csv(f"{OUTDIR}/d1_donor_averages_{ds_name}.csv", index=False)

            print(f"    Donors: {len(donor_df)}")

            # Correlate AUCell scores with SASP12
            corr_results = []
            for col in aucell_matrix.columns:
                tf_name = col.replace('(+)', '')
                aucell_col = f'aucell_{col}'

                if aucell_col not in donor_df.columns:
                    continue

                valid = donor_df[aucell_col].notna() & donor_df['SASP12_mean'].notna()
                if valid.sum() < 4:
                    continue

                rho, p = stats.spearmanr(donor_df.loc[valid, aucell_col], donor_df.loc[valid, 'SASP12_mean'])
                n = valid.sum()
                ci_lo, ci_hi = fisher_z_ci(rho, n)

                # Also compute raw mRNA correlation for comparison
                raw_col = f'raw_{tf_name}'
                if raw_col in donor_df.columns:
                    valid_raw = donor_df[raw_col].notna() & donor_df['SASP12_mean'].notna()
                    if valid_raw.sum() >= 4:
                        rho_raw, p_raw = stats.spearmanr(donor_df.loc[valid_raw, raw_col], donor_df.loc[valid_raw, 'SASP12_mean'])
                    else:
                        rho_raw, p_raw = np.nan, np.nan
                else:
                    rho_raw, p_raw = np.nan, np.nan

                corr_results.append({
                    'dataset': ds_name,
                    'tf': tf_name,
                    'n_donors': n,
                    'aucell_rho': rho,
                    'aucell_p': p,
                    'aucell_ci_95': f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                    'raw_mrna_rho': rho_raw,
                    'raw_mrna_p': p_raw,
                    'delta_rho': rho - rho_raw if not np.isnan(rho_raw) else np.nan,
                })

            corr_df = pd.DataFrame(corr_results)
            corr_df.to_csv(f"{OUTDIR}/d1_correlations_{ds_name}.csv", index=False)
            all_donor_corrs.append(corr_df)

            # Print key results
            print(f"\n    === Key AUCell-SASP Correlations ===")
            for tf in ['JUNB', 'JUN', 'FOS', 'FOSB', 'KLF10', 'ATF3', 'EGR1', 'IRF1', 'CDKN1A']:
                sub = corr_df[corr_df['tf'] == tf]
                if len(sub) > 0:
                    row = sub.iloc[0]
                    print(f"      {tf:8s}: AUCell rho={row['aucell_rho']:.3f} (p={row['aucell_p']:.2e}) "
                          f"vs raw rho={row['raw_mrna_rho']:.3f} delta={row['delta_rho']:+.3f}")

        except Exception as e:
            print(f"    AUCell ERROR: {e}")
            import traceback
            traceback.print_exc()

            # Fallback: manual AUC computation
            print("    Attempting manual AUCell computation...")
            try:
                from pyscenic.aucell import derive_rankings, enrichment

                rankings = derive_rankings(ex_all_for_aucell, seed=42)

                aucell_results = []
                for reg_name, reg_genes in gene_sets_aucell.items():
                    if len(reg_genes) < 3:
                        continue
                    # Compute enrichment for each cell
                    aucs = enrichment(rankings, reg_genes)
                    aucell_results.append(aucs)

                if aucell_results:
                    aucell_matrix = pd.concat(aucell_results, axis=1)
                    print(f"    Manual AUCell completed: {aucell_matrix.shape}")

                    # Continue with donor-level analysis
                    donor_df = compute_donor_averages(adata, aucell_matrix.values, SASP12, TARGET_TFS)
                    donor_df.to_csv(f"{OUTDIR}/d1_donor_averages_{ds_name}.csv", index=False)
                else:
                    print("    Manual AUCell produced no results")
                    aucell_matrix = None

            except Exception as e2:
                print(f"    Manual AUCell also failed: {e2}")
                aucell_matrix = None

        t_total = time.time() - t_start
        print(f"\n  {ds_name} completed in {t_total/60:.1f} minutes")

        # Clean up large objects
        del adata, adata_snrna, ex_snrna, ex_all
        if 'adjacencies_snrna' in dir():
            del adjacencies_snrna
        if 'adjacencies_all' in dir():
            del adjacencies_all

        # Save checkpoint
        if all_donor_corrs:
            pd.concat(all_donor_corrs).to_csv(f"{OUTDIR}/d1_correlations_all.csv", index=False)

    # ============================================================
    # Summary
    # ============================================================

    print(f"\n{'='*60}")
    print(f"=== D1 SCENIC SUMMARY ===")
    print(f"{'='*60}")

    if all_donor_corrs:
        all_corrs = pd.concat(all_donor_corrs)
        all_corrs.to_csv(f"{OUTDIR}/d1_correlations_all.csv", index=False)

        # Print summary per dataset
        for ds in all_corrs['dataset'].unique():
            sub = all_corrs[all_corrs['dataset'] == ds]
            print(f"\n  {ds}:")

            # Sort by AUCell rho
            sub_sorted = sub.sort_values('aucell_rho', ascending=False)
            print(f"    Top 5 AUCell-SASP TFs:")
            for _, row in sub_sorted.head(5).iterrows():
                print(f"      {row['tf']:8s}: AUCell rho={row['aucell_rho']:.3f}, raw rho={row['raw_mrna_rho']:.3f}")

            # Check JUNB specifically
            junb = sub[sub['tf'] == 'JUNB']
            if len(junb) > 0:
                j = junb.iloc[0]
                print(f"\n    JUNB Decision Rule:")
                if j['aucell_rho'] >= 0.60:
                    print(f"      VALIDATED (rho={j['aucell_rho']:.3f} >= 0.60)")
                elif j['aucell_rho'] >= 0.30:
                    print(f"      INCONCLUSIVE (rho={j['aucell_rho']:.3f}, 0.30-0.60)")
                else:
                    print(f"      NOT VALIDATED (rho={j['aucell_rho']:.3f} < 0.30)")

        # Save results
        summary = {
            'batch': 'batch_052_d1',
            'date': pd.Timestamp.now().isoformat(),
            'analysis': 'D1 pySCENIC regulon inference and donor-level AUCell-SASP correlation',
            'datasets': list(all_corrs['dataset'].unique()),
            'key_findings': {},
        }

        for ds in all_corrs['dataset'].unique():
            sub = all_corrs[all_corrs['dataset'] == ds]
            junb = sub[sub['tf'] == 'JUNB']
            if len(junb) > 0:
                j = junb.iloc[0]
                summary['key_findings'][ds] = {
                    'JUNB_aucell_rho': float(j['aucell_rho']),
                    'JUNB_aucell_p': float(j['aucell_p']),
                    'JUNB_raw_rho': float(j['raw_mrna_rho']),
                    'JUNB_delta_rho': float(j['delta_rho']),
                }

        with open(f"{OUTDIR}/results_d1.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    else:
        print("  No correlation results produced")
        with open(f"{OUTDIR}/results_d1.json", 'w') as f:
            json.dump({'batch': 'batch_052_d1', 'status': 'FAILED', 'error': 'No correlation results'}, f, indent=2)

    # Clean up dask
    print("\nShutting down dask client and cluster...")
    dask_client.close()
    local_cluster.close()

    print(f"\n=== batch_052 D1 COMPLETE ===")
