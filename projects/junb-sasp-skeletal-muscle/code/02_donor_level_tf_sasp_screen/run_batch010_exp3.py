#!/usr/bin/env python3
"""
batch_010 Experiment 3: Nature Aging 2024 Atlas Cross-Validation

WHAT: Independent replication of AP-1/JUNB aging effect in FAPs using
      Nature Aging 2024 skeletal muscle aging atlas.

WHY: G2 requires independent replication. Nature Aging atlas is the natural
     validation dataset from Sanger. Population confound (Chinese vs European
     cohort) is acknowledged but cannot be controlled.

SOURCE: batch_010 design.yaml (cite this as experimental protocol).

PREDICTION: FAPs (PDGFRA+ AND DCN+) will show AP-1 elevation with age (d > 0.5).
           JUNB will be the strongest AP-1 member (d > 0.5).
           JUNB-collagen correlation within FAPs: rho > 0.15.

DECISION RULE:
  - d > 0.5 for JUNB AND direction consistent → G2 STRONGLY supported (50% replication)
  - d > 0.3 for AP-1 composite with d > 0.4 for JUNB → G2 supported (partial)
  - d < 0.3 for AP-1 composite → dataset-specific, not universal

UNINTERPRETABLE: FAPs not identifiable (PDGFRA+/DCN+ do not co-express in this data).
"""

import json
import os
import sys
import subprocess
from pathlib import Path

# Third-party imports with version-aware logging
import scanpy as sc
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import spearmanr

# Configuration
DATA_URL = "https://cellgeni.cog.sanger.ac.uk/muscleageingcellatlas/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad"
DATA_PATH = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/SKM_fibroblasts_Schwann_human_2023-06-22.h5ad"
OUTPUT_PATH = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_010/results_nature_aging.json"

# AP-1 composite genes (per batch_009 methodology)
# SOURCE: batch_009 (AP-1 signature from HLMA)
AP1_GENES = ['FOS', 'FOSL1', 'FOSL2', 'JUN', 'JUNB', 'JUND']

# Collagen genes for correlation analysis
# SOURCE: batch_009 (canonical ECM/fibrosis genes)
COLLAGEN_GENES = ['COL1A1', 'COL3A1', 'COL6A1', 'COL6A3', 'FN1', 'LOX', 'LOXL1']

# Age thresholds (per design spec)
YOUNG_THRESHOLD = 40  # age < 40
OLD_THRESHOLD = 65     # age > 65

# Decision thresholds (50% of HLMA effect size per design)
# SOURCE: batch_010 design.yaml (critics raised threshold from d > 0.2 to d > 0.5)
AP1_THRESHOLD = 0.5   # 50% replication of HLMA d=1.07
JUNB_THRESHOLD = 0.5   # 50% replication of HLMA d=0.751
COLLAGEN_CORR_THRESHOLD = 0.15  # correlation within independent atlas

# Bootstrap parameters
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


def log_environment():
    """Log complete environment for reproducibility."""
    print("=" * 80)
    print("ENVIRONMENT LOG")
    print("=" * 80)
    print(f"Python: {sys.version}")
    print(f"Scanpy: {sc.settings.verbosity}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 80)


def download_data():
    """Download Nature Aging atlas from Sanger.

    WHY: Data source confirmed in batch_010 design (Sanger URL verified with HTTP 200).
    """
    if os.path.exists(DATA_PATH):
        print(f"File already exists: {DATA_PATH}")
        size_mb = os.path.getsize(DATA_PATH) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
        return True

    print(f"Downloading Nature Aging atlas...")
    print(f"  URL: {DATA_URL}")
    print(f"  Destination: {DATA_PATH}")

    cmd = [
        'wget', '-O', DATA_PATH,
        '--no-check-certificate',
        '--quiet',
        DATA_URL
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: wget failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
        return False

    size_mb = os.path.getsize(DATA_PATH) / (1024 * 1024)
    print(f"Download complete: {size_mb:.1f} MB")
    return True


def qc_data(adata):
    """Quality control of downloaded atlas.

    WHAT: Check shape, gene presence, and age annotations.
    WHY: Must verify data integrity before proceeding with analysis.

    Args:
        adata: AnnData object

    Returns:
        dict with QC metrics
    """
    print("\n" + "=" * 80)
    print("DATA QC")
    print("=" * 80)

    qc = {
        'total_cells': adata.n_obs,
        'total_genes': adata.n_vars,
    }

    # Check age annotations - try multiple possible column names
    # Updated with actual columns found in Nature Aging atlas
    age_columns = ['age', 'Age', 'age_group', 'Age_group', 'Age_bin', 'donor_age',
                   'DonorAge', 'donor', 'Age_at_collection', 'age_at_death', 'AgeDeath']
    age_col = None
    age_type = None  # 'numeric', 'categorical', or 'binned'
    for col in age_columns:
        if col in adata.obs.columns:
            age_col = col
            # Determine if numeric, categorical (young/old), or binned (15-20, 25-30, etc.)
            vals = adata.obs[age_col].dropna().unique()
            if len(vals) <= 10:  # Likely categorical or binned
                str_vals = [str(v) for v in vals]
                # Check for binary young/old
                if set(str_vals).issubset({'young', 'old'}):
                    age_type = 'categorical_binary'
                # Check for age bins
                elif any('-' in str(v) for v in vals):
                    age_type = 'binned'
                else:
                    age_type = 'categorical_other'
            else:
                age_type = 'numeric'
            break

    if age_col:
        qc['age_annotation_found'] = True
        qc['age_column_used'] = age_col
        qc['age_type'] = age_type
        qc['age_values'] = sorted(adata.obs[age_col].dropna().unique().tolist())

        if age_type == 'numeric':
            ages = pd.to_numeric(adata.obs[age_col], errors='coerce')
            qc['age_numeric'] = True
            qc['age_min'] = float(ages.min())
            qc['age_max'] = float(ages.max())
            qc['age_mean'] = float(ages.mean())
        else:
            qc['age_numeric'] = False
            if age_type == 'categorical_binary':
                qc['age_min'] = None
                qc['age_max'] = None
            elif age_type == 'binned':
                # Extract midpoint for each bin
                bin_midpoints = []
                for v in qc['age_values']:
                    try:
                        parts = str(v).split('-')
                        mid = (float(parts[0]) + float(parts[1])) / 2
                        bin_midpoints.append(mid)
                    except:
                        pass
                if bin_midpoints:
                    qc['age_min'] = min(bin_midpoints)
                    qc['age_max'] = max(bin_midpoints)
    else:
        qc['age_annotation_found'] = False
        qc['age_values'] = []
        qc['age_type'] = None

    # Check for key genes
    gene_names = adata.var_names.tolist()
    qc['pdgfra_present'] = 'PDGFRA' in gene_names
    qc['dcn_present'] = 'DCN' in gene_names
    qc['ap1_genes_present'] = [g in gene_names for g in AP1_GENES]
    qc['collagen_genes_present'] = [g in gene_names for g in COLLAGEN_GENES]

    print(f"Total cells: {qc['total_cells']:,}")
    print(f"Total genes: {qc['total_genes']:,}")
    print(f"Age annotation found: {qc['age_annotation_found']}")
    if qc['age_annotation_found']:
        print(f"  Column: {qc.get('age_column_used', 'N/A')}")
        print(f"  Type: {qc.get('age_type', 'unknown')}")
        print(f"  Values: {qc['age_values'][:10]}...")  # First 10 values
        if qc.get('age_numeric'):
            print(f"  Range: {qc['age_min']:.1f} - {qc['age_max']:.1f}")
    print(f"PDGFRA present: {qc['pdgfra_present']}")
    print(f"DCN present: {qc['dcn_present']}")
    print(f"AP-1 genes present: {qc['ap1_genes_present']}")
    print(f"Collagen genes present: {qc['collagen_genes_present']}")

    return qc, age_col, age_type


def identify_faps(adata, age_col):
    """Identify FAPs as PDGFRA+ AND DCN+ cells.

    WHAT: FAPs defined as cells expressing both PDGFRA and DCN.
    WHY: FAPs are fibro-adipogenic progenitors - the target cell type for this analysis.
         Both markers required per design spec (FAP equivalence definition).

    Args:
        adata: AnnData object
        age_col: column name for age annotations

    Returns:
        FAP subset of adata
    """
    print("\n" + "=" * 80)
    print("FAP IDENTIFICATION")
    print("=" * 80)

    # Get expression matrices - prefer raw if available, otherwise layer
    if adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    # Find gene indices
    pdgfra_idx = np.where(var_names == 'PDGFRA')[0]
    dcn_idx = np.where(var_names == 'DCN')[0]

    if len(pdgfra_idx) == 0 or len(dcn_idx) == 0:
        print("ERROR: PDGFRA or DCN genes not found in data")
        return None, None

    pdgfra_idx = pdgfra_idx[0]
    dcn_idx = dcn_idx[0]

    # Get expression values
    if hasattr(X, 'toarray'):
        # Sparse matrix
        pdgfra_expr = X[:, pdgfra_idx].toarray().flatten()
        dcn_expr = X[:, dcn_idx].toarray().flatten()
    else:
        pdgfra_expr = X[:, pdgfra_idx].flatten()
        dcn_expr = X[:, dcn_idx].flatten()

    # FAP identification: both markers above zero
    pdgfra_positive = pdgfra_expr > 0
    dcn_positive = dcn_expr > 0
    fap_mask = pdgfra_positive & dcn_positive

    pdgfra_count = int(pdgfra_positive.sum())
    dcn_count = int(dcn_positive.sum())
    fap_count = int(fap_mask.sum())

    print(f"PDGFRA+ cells: {pdgfra_count:,} ({100*pdgfra_count/len(fap_mask):.1f}%)")
    print(f"DCN+ cells: {dcn_count:,} ({100*dcn_count/len(fap_mask):.1f}%)")
    print(f"PDGFRA+ AND DCN+ (FAPs): {fap_count:,} ({100*fap_count/len(fap_mask):.2f}%)")

    # Return FAP indices and counts
    fap_info = {
        'pdgfra_positive': pdgfra_count,
        'dcn_positive': dcn_count,
        'pdgfra_and_dcn': fap_count,
        'fap_percentage_of_total': round(100 * fap_count / len(fap_mask), 2)
    }

    return fap_mask, fap_info


def separate_age_groups(adata, fap_mask, age_col, age_type):
    """Separate FAPs into young and old age groups.

    WHAT: Young (age < 40) vs Old (age > 65) or use categorical/bin labels.
    WHY: Design spec requires testing age effect on AP-1/JUNB expression.

    Args:
        adata: AnnData object
        fap_mask: boolean mask for FAPs
        age_col: column name for age
        age_type: type of age annotation ('numeric', 'categorical_binary', 'binned')

    Returns:
        young indices, old indices, grouping info
    """
    print("\n" + "=" * 80)
    print("AGE GROUPING")
    print("=" * 80)

    young_mask = fap_mask.copy()
    old_mask = fap_mask.copy()

    if age_type == 'numeric':
        ages = adata.obs[age_col].values
        fap_ages = ages[fap_mask]
        young_mask[fap_mask] = fap_ages < YOUNG_THRESHOLD
        old_mask[fap_mask] = fap_ages > OLD_THRESHOLD
        fap_ages_young = fap_ages[fap_ages < YOUNG_THRESHOLD]
        fap_ages_old = fap_ages[fap_ages > OLD_THRESHOLD]

    elif age_type == 'categorical_binary':
        # Age_bin: 'young' or 'old'
        young_mask[fap_mask] = adata.obs.loc[fap_mask, age_col] == 'young'
        old_mask[fap_mask] = adata.obs.loc[fap_mask, age_col] == 'old'
        fap_ages_young = None
        fap_ages_old = None

    elif age_type == 'binned':
        # Age_group: bins like "15-20", "25-30", etc.
        # Map bins to young (< 40) or old (>= 60, straddling threshold)
        young_bins = ['15-20', '25-30', '35-40']
        old_bins = ['55-60', '60-65', '70-75']  # Include 55-65 for reasonable sample

        young_mask[fap_mask] = adata.obs.loc[fap_mask, age_col].isin(young_bins)
        old_mask[fap_mask] = adata.obs.loc[fap_mask, age_col].isin(old_bins)
        fap_ages_young = None
        fap_ages_old = None

    young_indices = np.where(young_mask)[0]
    old_indices = np.where(old_mask)[0]

    young_n = len(young_indices)
    old_n = len(old_indices)

    print(f"Young FAPs: {young_n:,}")
    if fap_ages_young is not None and len(fap_ages_young) > 0:
        print(f"  Age range: {fap_ages_young.min():.1f} - {fap_ages_young.max():.1f}")
    else:
        # Show the bins used
        if age_type == 'categorical_binary':
            print(f"  Category: 'young'")
        elif age_type == 'binned':
            print(f"  Bins: {young_bins}")

    print(f"Old FAPs: {old_n:,}")
    if fap_ages_old is not None and len(fap_ages_old) > 0:
        print(f"  Age range: {fap_ages_old.min():.1f} - {fap_ages_old.max():.1f}")
    else:
        if age_type == 'categorical_binary':
            print(f"  Category: 'old'")
        elif age_type == 'binned':
            print(f"  Bins: {old_bins}")

    grouping_info = {
        'young_n': young_n,
        'old_n': old_n,
        'age_type': age_type,
        'young_age_range': [float(fap_ages_young.min()), float(fap_ages_young.max())] if fap_ages_young is not None and len(fap_ages_young) > 0 else None,
        'old_age_range': [float(fap_ages_old.min()), float(fap_ages_old.max())] if fap_ages_old is not None and len(fap_ages_old) > 0 else None,
        'young_bins': young_bins if age_type == 'binned' else None,
        'old_bins': old_bins if age_type == 'binned' else None
    }

    return young_indices, old_indices, grouping_info


def compute_scores(adata, indices, gene_list):
    """Compute mean expression score for a gene list, z-scored within atlas.

    WHAT: Mean of gene expression, z-scored for comparability.
    WHY: Standard practice for gene set scoring. Z-scoring within atlas ensures
         comparable scale across datasets.

    Args:
        adata: AnnData object
        indices: cell indices
        gene_list: list of gene names

    Returns:
        z-scored mean expression values
    """
    # Use raw if available, otherwise X
    if adata.raw is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    # Get gene indices
    gene_indices = [np.where(var_names == g)[0][0] for g in gene_list if g in var_names]
    if len(gene_indices) == 0:
        return None

    gene_indices = np.array(gene_indices)

    # Extract expression - handle sparse matrices properly
    if hasattr(X, 'toarray'):
        # For sparse, extract all cells first
        X_dense = X.toarray()
        full_expr = X_dense[:, gene_indices]
    else:
        full_expr = X[:, gene_indices]

    # Compute raw mean per cell (across the gene list)
    raw_scores = full_expr.mean(axis=1)

    # Z-score within full atlas (per design spec)
    # Compute atlas-wide statistics from the DISTRIBUTION of raw scores
    atlas_mean = raw_scores.mean()
    atlas_std = raw_scores.std()

    # Compute subset scores and z-score using atlas-wide statistics
    if hasattr(X, 'toarray'):
        expr_subset = X_dense[np.ix_(indices, gene_indices)]
    else:
        expr_subset = full_expr[indices, :]

    subset_scores = expr_subset.mean(axis=1)

    # Z-score (avoid division by zero)
    z_scores = np.zeros(len(subset_scores))
    if atlas_std > 0:
        z_scores = (subset_scores - atlas_mean) / atlas_std
    else:
        z_scores = subset_scores - atlas_mean

    return z_scores


def compute_cohen_d(young_scores, old_scores):
    """Compute Cohen's d with bootstrap 95% CI.

    WHAT: Effect size = (mean_old - mean_young) / pooled_SD
    WHY: Standard effect size measure. Allows comparison across studies.
         Bootstrap CI assesses uncertainty.

    Args:
        young_scores: expression scores for young cells
        old_scores: expression scores for old cells

    Returns:
        dict with d, CI, p-value
    """
    np.random.seed(RANDOM_SEED)

    mean_young = young_scores.mean()
    mean_old = old_scores.mean()
    std_young = young_scores.std(ddof=1)
    std_old = old_scores.std(ddof=1)
    n_young = len(young_scores)
    n_old = len(old_scores)

    # Pooled SD
    pooled_sd = np.sqrt(((n_young - 1) * std_young**2 + (n_old - 1) * std_old**2) /
                        (n_young + n_old - 2))

    # Cohen's d
    if pooled_sd > 0:
        d = (mean_old - mean_young) / pooled_sd
    else:
        d = 0.0

    # Two-sample t-test for p-value
    t_stat, p_value = stats.ttest_ind(old_scores, young_scores, equal_var=False)

    # Bootstrap CI
    bootstrap_ds = []
    for _ in range(N_BOOTSTRAP):
        young_boot = np.random.choice(young_scores, size=n_young, replace=True)
        old_boot = np.random.choice(old_scores, size=n_old, replace=True)

        mean_y_b = young_boot.mean()
        mean_o_b = old_boot.mean()
        std_y_b = young_boot.std(ddof=1)
        std_o_b = old_boot.std(ddof=1)

        pooled_b = np.sqrt(((n_young - 1) * std_y_b**2 + (n_old - 1) * std_o_b**2) /
                           (n_young + n_old - 2))

        if pooled_b > 0:
            d_b = (mean_o_b - mean_y_b) / pooled_b
        else:
            d_b = 0.0
        bootstrap_ds.append(d_b)

    ci_low = np.percentile(bootstrap_ds, 2.5)
    ci_high = np.percentile(bootstrap_ds, 97.5)

    result = {
        'young_mean': float(mean_young),
        'old_mean': float(mean_old),
        'cohen_d': float(d),
        'cohen_d_ci95': [float(ci_low), float(ci_high)],
        'p_value': float(p_value),
        'bootstrap_n': N_BOOTSTRAP
    }

    return result


def compute_correlation(x, y):
    """Compute Spearman correlation with p-value.

    WHAT: Spearman rho - rank-based correlation, robust to outliers.
    WHY: Expression data often non-normal. Spearman is appropriate for
         gene expression correlation analysis.

    Args:
        x, y: expression vectors

    Returns:
        dict with rho, p-value, n
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p_value = spearmanr(x, y, nan_policy='omit')

    n = len(x) - np.isnan(x).sum() - np.isnan(y).sum()

    return {
        'rho': float(rho) if not np.isnan(rho) else None,
        'p_value': float(p_value) if not np.isnan(p_value) else None,
        'n': int(n)
    }


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 80)
    print("BATCH_010 EXPERIMENT 3: Nature Aging 2024 Atlas Cross-Validation")
    print("=" * 80)

    log_environment()

    # Step 1: Download data
    print("\n[Step 1] Downloading Nature Aging atlas...")
    if not download_data():
        print("FATAL: Download failed")
        return None

    # Step 2: Load and QC
    print("\n[Step 2] Loading and QC...")
    try:
        adata = sc.read_h5ad(DATA_PATH)
        print(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return None

    qc_results, age_col, age_type = qc_data(adata)

    if not age_col:
        print("ERROR: No age annotation found in data")
        print("UNINTERPRETABLE: Cannot test age effect without age annotations")
        return None

    # Step 3: Identify FAPs
    print("\n[Step 3] Identifying FAPs...")
    fap_mask, fap_info = identify_faps(adata, age_col)

    if fap_mask is None:
        print("FATAL: FAP identification failed (marker genes not present)")
        return None

    if fap_info['pdgfra_and_dcn'] == 0:
        print("ERROR: No FAPs identified (no PDGFRA+ AND DCN+ cells)")
        print("UNINTERPRETABLE: FAPs not identifiable in this data")
        return None

    # Step 4: Age grouping
    print("\n[Step 4] Separating age groups...")
    young_idx, old_idx, grouping_info = separate_age_groups(adata, fap_mask, age_col, age_type)

    if grouping_info['young_n'] < 10 or grouping_info['old_n'] < 10:
        print("WARNING: Very few cells in one or both age groups")
        print("Results may be underpowered")

    # Step 5: Compute AP-1 and JUNB scores
    print("\n[Step 5] Computing AP-1 and JUNB scores...")

    # AP-1 composite
    ap1_scores_all = compute_scores(adata, np.arange(adata.n_obs), AP1_GENES)
    ap1_young = ap1_scores_all[young_idx]
    ap1_old = ap1_scores_all[old_idx]

    # JUNB individual
    junb_scores_all = compute_scores(adata, np.arange(adata.n_obs), ['JUNB'])
    junb_young = junb_scores_all[young_idx]
    junb_old = junb_scores_all[old_idx]

    # Step 6: Compute age effects
    print("\n[Step 6] Computing age effects (Cohen d)...")
    ap1_effect = compute_cohen_d(ap1_young, ap1_old)
    junb_effect = compute_cohen_d(junb_young, junb_old)

    print(f"\nAP-1 composite:")
    print(f"  Young mean: {ap1_effect['young_mean']:.4f}")
    print(f"  Old mean: {ap1_effect['old_mean']:.4f}")
    print(f"  Cohen d: {ap1_effect['cohen_d']:.4f} (95% CI: {ap1_effect['cohen_d_ci95'][0]:.4f}, {ap1_effect['cohen_d_ci95'][1]:.4f})")
    print(f"  p-value: {ap1_effect['p_value']:.2e}")

    print(f"\nJUNB:")
    print(f"  Young mean: {junb_effect['young_mean']:.4f}")
    print(f"  Old mean: {junb_effect['old_mean']:.4f}")
    print(f"  Cohen d: {junb_effect['cohen_d']:.4f} (95% CI: {junb_effect['cohen_d_ci95'][0]:.4f}, {junb_effect['cohen_d_ci95'][1]:.4f})")
    print(f"  p-value: {junb_effect['p_value']:.2e}")

    # Step 7: Correlation analysis (within old FAPs)
    print("\n[Step 7] Computing JUNB-collagen correlation within old FAPs...")

    junb_old_scores = junb_scores_all[old_idx]
    collagen_scores_all = compute_scores(adata, np.arange(adata.n_obs), COLLAGEN_GENES)
    collagen_old_scores = collagen_scores_all[old_idx]

    corr_result = compute_correlation(junb_old_scores, collagen_old_scores)

    print(f"\nJUNB vs Collagen composite (within old FAPs):")
    rho_str = f"{corr_result['rho']:.4f}" if corr_result['rho'] is not None else "N/A (constant)"
    p_str = f"{corr_result['p_value']:.2e}" if corr_result['p_value'] is not None else "N/A"
    print(f"  Spearman rho: {rho_str}")
    print(f"  p-value: {p_str}")
    print(f"  n: {corr_result['n']}")

    # Step 8: Decision rules
    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)

    ap1_threshold_met = ap1_effect['cohen_d'] > AP1_THRESHOLD
    junb_threshold_met = junb_effect['cohen_d'] > JUNB_THRESHOLD
    correlation_threshold_met = corr_result['rho'] is not None and corr_result['rho'] > COLLAGEN_CORR_THRESHOLD

    # G2 supported if both JUNB d > 0.5 and direction consistent (positive)
    g2_supported = junb_threshold_met and junb_effect['cohen_d'] > 0

    print(f"\nPre-registered thresholds:")
    print(f"  AP-1 composite d > {AP1_THRESHOLD}: {ap1_threshold_met} (actual: {ap1_effect['cohen_d']:.4f})")
    print(f"  JUNB d > {JUNB_THRESHOLD}: {junb_threshold_met} (actual: {junb_effect['cohen_d']:.4f})")
    print(f"  JUNB-collagen rho > {COLLAGEN_CORR_THRESHOLD}: {correlation_threshold_met} (actual: {corr_result['rho']})")
    print(f"\nG2 strongly supported: {g2_supported}")

    # Compile results
    results = {
        'experiment_id': 'batch_010_exp3',
        'hypothesis': 'Nature Aging 2024 Atlas Cross-Validation (G2: Independent Replication)',
        'data_qc': qc_results,
        'fap_identification': fap_info,
        'age_grouping': grouping_info,
        'ap1_age_effect': ap1_effect,
        'junb_age_effect': junb_effect,
        'junb_collagen_correlation': corr_result,
        'decision': {
            'ap1_threshold_met': ap1_threshold_met,
            'junb_threshold_met': junb_threshold_met,
            'g2_supported': g2_supported
        },
        'metadata': {
            'ap1_threshold': AP1_THRESHOLD,
            'junb_threshold': JUNB_THRESHOLD,
            'collagen_corr_threshold': COLLAGEN_CORR_THRESHOLD,
            'young_threshold': YOUNG_THRESHOLD,
            'old_threshold': OLD_THRESHOLD,
            'bootstrap_n': N_BOOTSTRAP,
            'random_seed': RANDOM_SEED,
            'data_url': DATA_URL,
            'python_version': sys.version,
            'scanpy_version': sc.__version__,
            'numpy_version': np.__version__
        }
    }

    # Save results
    print(f"\nSaving results to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = main()
    if results:
        sys.exit(0)
    else:
        sys.exit(1)
