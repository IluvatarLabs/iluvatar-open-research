#!/usr/bin/env python3
"""
batch_038: Nature Aging Endothelial Cell JUNB-SASP Cross-Compartment Replication

Downloads and analyzes the CELLxGENE Endothelium+SMC dataset from Nature Aging 2024
to compute donor-level JUNB-SASP correlation for cross-compartment replication of F084.

Primary: rho(JUNB, SASP12) at donor level (Nature Aging endothelial cells)
Secondary: rho(FOS, SASP12), rho(JUNB, FOS)
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Configuration
DATA_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data"
OUTPUT_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_038"
LOG_DIR = f"{OUTPUT_DIR}/logs"
H5AD_FILE = f"{DATA_DIR}/NA_Endothelium_SMC.h5ad"

# SASP12 gene panel
SASP12_GENES = [
    'CXCL1', 'CXCL8', 'CXCL6', 'CCL2', 'CCL20', 'CCL7',
    'IL6', 'IL8', 'SERPINE1', 'MMP1', 'MMP3', 'PLAUR'
]

def log(msg):
    print(f"[batch_038] {msg}", flush=True)

def fisher_ci(rho, n):
    """Compute Fisher z-transformed 95% CI for Spearman rho."""
    if n <= 3 or np.isnan(rho):
        return np.nan, np.nan
    z = 0.5 * np.log((1 + rho) / (1 - rho)) if abs(rho) < 1 else 0
    se_z = 1 / np.sqrt(n - 3)
    z_lo = z - 1.96 * se_z
    z_hi = z + 1.96 * se_z
    rho_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1) if abs(z_lo) < 20 else np.nan
    rho_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1) if abs(z_hi) < 20 else np.nan
    return rho_lo, rho_hi

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        'batch_id': 'batch_038',
        'hypothesis': 'Cross-compartment replication of JUNB→SASP in Nature Aging endothelial cells (V4 resolution)',
        'status': 'pending',
        'cell_type_counts': {},
        'detection_rates': {},
        'available_sasp': [],
        'primary': {},
        'secondary': {},
        'verdict': None,
        'notes': []
    }

    # Step 1: Load
    log("Loading Nature Aging Endothelium+SMC file...")
    try:
        import scanpy as sc
        adata = sc.read_h5ad(H5AD_FILE)
    except Exception as e:
        log(f"Load error: {e}")
        results['status'] = 'error'
        results['error'] = f'Load failed: {e}'
        with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        return 1

    # This dataset uses ENSEMBL IDs as var_names — switch to SYMBOL for readability
    if 'SYMBOL' in adata.var.columns:
        adata.var_names = adata.var['SYMBOL']
        log("Switched to SYMBOL gene names")

    log(f"Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # Cell type annotations
    ct_col = None
    for col in ['cell_type', 'celltype', 'annotation', 'cell_type_annotations']:
        if col in adata.obs.columns:
            ct_col = col
            break

    if ct_col:
        log(f"Cell type column: {ct_col}")
        results['cell_type_column'] = ct_col
        results['cell_type_counts'] = adata.obs[ct_col].value_counts().head(20).to_dict()
        log(f"Cell types: {adata.obs[ct_col].value_counts().head(10).to_dict()}")

    # Metadata columns
    meta_cols = [c for c in adata.obs.columns if any(k in c.lower() for k in
                ['donor', 'individual', 'age', 'sex', 'subject', 'development'])]
    results['metadata_columns'] = meta_cols
    log(f"Metadata columns: {meta_cols}")

    # Identify donor column
    donor_col = None
    for col in ['donor_id', 'DonorID', 'donor', 'individual', 'subject_id']:
        if col in adata.obs.columns:
            donor_col = col
            break

    if not donor_col:
        results['notes'].append("No standard donor column identified")
        results['status'] = 'error'
        results['error'] = 'No donor column found'
        with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        return 1

    results['donor_column'] = donor_col

    # Filter to endothelial cells only
    if ct_col:
        ec_mask = adata.obs[ct_col].str.contains('endothelial|EC', case=False, na=False)
        adata_ec = adata[ec_mask].copy()
        n_ec = ec_mask.sum()
        log(f"Endothelial cells: {n_ec:,} / {adata.n_obs:,}")
        results['n_endothelial_cells'] = int(n_ec)
    else:
        adata_ec = adata
        results['notes'].append("No cell type column — using all cells")
        log("No cell type column found, using all cells")

    n_donors_ec = adata_ec.obs[donor_col].nunique()
    log(f"N donors with endothelial cells: {n_donors_ec}")

    if n_donors_ec < 5:
        results['notes'].append(f"Only {n_donors_ec} donors with endothelial cells — underpowered")
        log(f"WARNING: Only {n_donors_ec} donors")

    # Check gene detection
    available_sasp = [g for g in SASP12_GENES if g in adata_ec.var_names]
    results['available_sasp'] = available_sasp
    log(f"SASP12 available: {len(available_sasp)}/{len(SASP12_GENES)}")

    has_junb = 'JUNB' in adata_ec.var_names
    has_fos = 'FOS' in adata_ec.var_names
    has_cdkn1a = 'CDKN1A' in adata_ec.var_names

    results['has_junb'] = has_junb
    results['has_fos'] = has_fos
    results['has_cdkn1a'] = has_cdkn1a

    if not has_junb:
        log("ERROR: JUNB not in dataset")
        results['notes'].append("JUNB not found")
        results['status'] = 'uninterpretable'
        with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        return 1

    # Detection rates for key genes
    for gene in ['JUNB', 'FOS', 'CDKN1A'] + available_sasp:
        if gene in adata_ec.var_names:
            det = (adata_ec[:, gene].X > 0).sum() / adata_ec.n_obs
            results['detection_rates'][gene] = float(det)

    junb_det = results['detection_rates'].get('JUNB', 0)
    log(f"JUNB detection rate: {junb_det:.1%}")

    if junb_det < 0.20:
        log("WARNING: JUNB detection < 20% — results may be unreliable")
        results['notes'].append(f"JUNB detection low: {junb_det:.1%}")

    # Step 2: Donor-level pseudobulk (endothelial cells only)
    log("Computing donor-level pseudobulk (endothelial cells)...")
    df = adata_ec.to_df()
    df[donor_col] = adata_ec.obs[donor_col].values

    pbulk = df.groupby(donor_col).mean()
    results['n_donors'] = int(pbulk.shape[0])
    results['donors'] = list(pbulk.index)
    log(f"Pseudobulk shape: {pbulk.shape}")

    # SASP12 score
    pbulk['SASP12'] = pbulk[available_sasp].mean(axis=1)
    pbulk['JUNB'] = pbulk['JUNB']
    if has_fos and 'FOS' in pbulk.columns:
        pbulk['FOS'] = pbulk['FOS']
    if has_cdkn1a and 'CDKN1A' in pbulk.columns:
        pbulk['CDKN1A'] = pbulk['CDKN1A']

    # Report available SASP genes actually used
    results['available_sasp'] = available_sasp

    # Step 3: Correlation analysis
    log("Running correlation analysis...")

    # Primary: rho(JUNB, SASP12)
    rho_junb, pval_junb = spearmanr(pbulk['JUNB'], pbulk['SASP12'])
    ci_lo, ci_hi = fisher_ci(rho_junb, len(pbulk))

    results['primary'] = {
        'metric': 'rho(JUNB, SASP12)',
        'rho': float(rho_junb),
        'p_value': float(pval_junb),
        'ci_95_lo': float(ci_lo),
        'ci_95_hi': float(ci_hi),
        'n': int(len(pbulk))
    }
    log(f"rho(JUNB, SASP12) = {rho_junb:.3f}, p = {pval_junb:.2e}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], N = {len(pbulk)}")

    # Secondary correlations
    if has_fos and 'FOS' in pbulk.columns:
        rho_fos, pval_fos = spearmanr(pbulk['FOS'], pbulk['SASP12'])
        results['secondary']['fos_sasp'] = {
            'rho': float(rho_fos),
            'p_value': float(pval_fos),
            'note': 'exploratory'
        }
        log(f"rho(FOS, SASP12) = {rho_fos:.3f}, p = {pval_fos:.2e} (exploratory)")

        # JUNB-FOS collinearity
        rho_junb_fos, pval_junb_fos = spearmanr(pbulk['JUNB'], pbulk['FOS'])
        results['secondary']['junb_fos'] = {
            'rho': float(rho_junb_fos),
            'p_value': float(pval_junb_fos),
            'note': 'exploratory collinearity check'
        }
        log(f"rho(JUNB, FOS) = {rho_junb_fos:.3f}, p = {pval_junb_fos:.2e} (exploratory)")

    if has_cdkn1a and 'CDKN1A' in pbulk.columns:
        rho_p21, pval_p21 = spearmanr(pbulk['CDKN1A'], pbulk['SASP12'])
        results['secondary']['cdkn1a_sasp'] = {
            'rho': float(rho_p21),
            'p_value': float(pval_p21),
            'note': 'exploratory'
        }
        log(f"rho(CDKN1A, SASP12) = {rho_p21:.3f}, p = {pval_p21:.2e} (exploratory)")

    # Step 4: Classification
    n = len(pbulk)
    log(f"N = {n}")

    if rho_junb >= 0.70 and pval_junb < 0.05:
        verdict = "STRONG CROSS-COMPARTMENT REPLICATION"
        confidence = "SUGGESTED"
    elif rho_junb >= 0.50 and pval_junb < 0.05:
        verdict = "MODERATE REPLICATION"
        confidence = "INCONCLUSIVE"
    else:
        verdict = "WEAK/ABSENT"
        confidence = "INCONCLUSIVE"

    results['verdict'] = verdict
    results['confidence'] = confidence
    log(f"VERDICT: {verdict}")

    # Comparison to F084
    f084_rho = 0.9287
    delta = rho_junb - f084_rho
    results['comparison_to_f084'] = {
        'f084_rho': f084_rho,
        'batch038_rho': float(rho_junb),
        'delta': float(delta),
        'interpretation': f"Cross-compartment delta within Nature Aging = {delta:+.3f}"
    }
    log(f"F084 comparison: delta = {delta:+.3f}")

    # Save results
    results['status'] = 'complete'
    results_file = f"{OUTPUT_DIR}/results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {results_file}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
