#!/usr/bin/env python3
"""
batch_037: Nature Aging MuSC p21→SASP Cross-Atlas Replication

Downloads and analyzes the Nature Aging MuSC file to compute donor-level
p21→SASP correlation for cross-atlas replication of F093.

Primary: rho(p21, SASP12) at donor level
Secondary: rho(JUNB, SASP12), rho(p21, JUNB)
"""

import os
import sys
import json
import subprocess
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Configuration
DATA_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data"
OUTPUT_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_037"
LOG_DIR = f"{OUTPUT_DIR}/logs"
MUSCID_FILE = f"{DATA_DIR}/SKM_MuSC_human_2023-06-22.h5ad"
MUSCID_URL = "https://cellgeni.cog.sanger.ac.uk/muscleageingcellatlas/SKM_MuSC_human_2023-06-22.h5ad"

# SASP12 gene panel
SASP12_GENES = [
    'CXCL1', 'CXCL8', 'CXCL6', 'CCL2', 'CCL20', 'CCL7',
    'IL6', 'IL8', 'SERPINE1', 'MMP1', 'MMP3', 'PLAUR'
]

def log(msg):
    print(f"[batch_037] {msg}", flush=True)

def fisher_ci(rho, n):
    """Compute Fisher z-transformed 95% CI for Spearman rho."""
    z = 0.5 * np.log((1 + rho) / (1 - rho)) if abs(rho) < 1 else 0
    se_z = 1 / np.sqrt(n - 3) if n > 3 else np.nan
    z_lo = z - 1.96 * se_z
    z_hi = z + 1.96 * se_z
    rho_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1) if abs(z_lo) < 20 else np.nan
    rho_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1) if abs(z_hi) < 20 else np.nan
    return rho_lo, rho_hi

def download_file():
    """Download the Nature Aging MuSC file if not present."""
    if os.path.exists(MUSCID_FILE):
        log(f"File already exists: {MUSCID_FILE}")
        return True

    log(f"Downloading Nature Aging MuSC file from {MUSCID_URL}")
    try:
        result = subprocess.run(
            ['wget', '-q', '-O', MUSCID_FILE, MUSCID_URL],
            capture_output=True, text=True, timeout=600
        )
        if os.path.exists(MUSCID_FILE):
            size_mb = os.path.getsize(MUSCID_FILE) / 1e6
            log(f"Downloaded successfully: {size_mb:.1f} MB")
            return True
        else:
            log(f"Download failed: {result.stderr}")
            return False
    except Exception as e:
        log(f"Download error: {e}")
        return False

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        'batch_id': 'batch_037',
        'hypothesis': 'Cross-atlas replication of p21→SASP in MuSCs (F093)',
        'status': 'pending',
        'donors': [],
        'detection_rates': {},
        'available_sasp': [],
        'primary': {},
        'secondary': {},
        'verdict': None,
        'notes': []
    }

    # Step 1: Download
    if not download_file():
        results['status'] = 'error'
        results['error'] = 'Download failed'
        with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        return 1

    # Step 2: Load and validate
    log("Loading Nature Aging MuSC file...")
    try:
        import scanpy as sc
        adata = sc.read_h5ad(MUSCID_FILE)
    except Exception as e:
        log(f"Load error: {e}")
        results['status'] = 'error'
        results['error'] = f'Load failed: {e}'
        with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        return 1

    log(f"Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # Check cell type annotations
    ct_col = None
    for col in ['cell_type', 'celltype', 'cell_type_annotations', 'celltype_annotations']:
        if col in adata.obs.columns:
            ct_col = col
            break

    if ct_col:
        log(f"Cell type column: {ct_col}")
        results['cell_type_column'] = ct_col
        results['cell_type_counts'] = adata.obs[ct_col].value_counts().to_dict()
    else:
        results['notes'].append("No cell type annotation column found")
        log("No cell type annotation column found")

    # Check donor/age metadata
    meta_cols = [c for c in adata.obs.columns if any(k in c.lower() for k in
                ['donor', 'individual', 'age', 'sex', 'subject'])]
    results['metadata_columns'] = meta_cols
    log(f"Metadata columns: {meta_cols}")

    # Identify donor column
    donor_col = None
    for col in ['donor_id', 'DonorID', 'donor', 'individual', 'subject_id', 'subject']:
        if col in adata.obs.columns:
            donor_col = col
            break

    if not donor_col:
        results['notes'].append("No donor column identified")
        log("WARNING: No donor column identified")
        results['status'] = 'error'
        results['error'] = 'No donor column found'
        with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        return 1

    results['donor_column'] = donor_col
    n_donors = adata.obs[donor_col].nunique()
    log(f"N donors: {n_donors}")

    # Step 3: Check gene detection
    available_sasp = [g for g in SASP12_GENES if g in adata.var_names]
    results['available_sasp'] = available_sasp
    log(f"SASP12 available: {len(available_sasp)}/{len(SASP12_GENES)}")

    has_cdkn1a = 'CDKN1A' in adata.var_names
    has_junb = 'JUNB' in adata.var_names
    results['has_cdkn1a'] = has_cdkn1a
    results['has_junb'] = has_junb

    if not has_cdkn1a:
        log("ERROR: CDKN1A (p21) not in dataset")
        results['notes'].append("CDKN1A not found")
        results['status'] = 'uninterpretable'
        with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        return 1

    # Detection rates
    for gene in available_sasp + ['CDKN1A', 'JUNB']:
        if gene in adata.var_names:
            det_rate = (adata[:, gene].X > 0).sum() / adata.n_obs
            results['detection_rates'][gene] = float(det_rate)

    cdkn1a_det = results['detection_rates'].get('CDKN1A', 0)
    log(f"CDKN1A detection rate: {cdkn1a_det:.1%}")

    if cdkn1a_det < 0.20:
        log("WARNING: CDKN1A detection < 20% — results may be unreliable")
        results['notes'].append(f"CDKN1A detection low: {cdkn1a_det:.1%}")

    # Step 4: Donor-level pseudobulk
    log("Computing donor-level pseudobulk...")
    df = adata.to_df()
    df[donor_col] = adata.obs[donor_col].values

    # Pseudobulk per donor
    pbulk = df.groupby(donor_col).mean()
    results['n_donors'] = pbulk.shape[0]
    results['donors'] = list(pbulk.index)

    log(f"Pseudobulk shape: {pbulk.shape}")

    # SASP12 score
    pbulk['SASP12'] = pbulk[available_sasp].mean(axis=1)
    pbulk['CDKN1A'] = pbulk['CDKN1A']
    if has_junb and 'JUNB' in pbulk.columns:
        pbulk['JUNB'] = pbulk['JUNB']

    # Step 5: Correlation analysis
    log("Running correlation analysis...")

    # Primary: rho(p21, SASP12)
    rho_p21, pval_p21 = spearmanr(pbulk['CDKN1A'], pbulk['SASP12'])
    ci_lo, ci_hi = fisher_ci(rho_p21, len(pbulk))

    results['primary'] = {
        'metric': 'rho(CDKN1A, SASP12)',
        'rho': float(rho_p21),
        'p_value': float(pval_p21),
        'ci_95_lo': float(ci_lo),
        'ci_95_hi': float(ci_hi),
        'n': int(len(pbulk))
    }
    log(f"rho(CDKN1A, SASP12) = {rho_p21:.3f}, p = {pval_p21:.2e}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], N = {len(pbulk)}")

    # Secondary: JUNB
    if has_junb and 'JUNB' in pbulk.columns:
        rho_junb, pval_junb = spearmanr(pbulk['JUNB'], pbulk['SASP12'])
        results['secondary']['junb_sasp'] = {
            'rho': float(rho_junb),
            'p_value': float(pval_junb),
            'note': 'exploratory'
        }
        log(f"rho(JUNB, SASP12) = {rho_junb:.3f}, p = {pval_junb:.2e} (exploratory)")

        # p21-JUNB collinearity
        rho_p21_junb, pval_p21_junb = spearmanr(pbulk['CDKN1A'], pbulk['JUNB'])
        results['secondary']['cdkn1a_junb'] = {
            'rho': float(rho_p21_junb),
            'p_value': float(pval_p21_junb),
            'note': 'exploratory collinearity check'
        }
        log(f"rho(CDKN1A, JUNB) = {rho_p21_junb:.3f}, p = {pval_p21_junb:.2e} (exploratory)")
    else:
        results['secondary']['junb_sasp'] = {'note': 'JUNB not available in dataset'}
        results['secondary']['cdkn1a_junb'] = {'note': 'JUNB not available in dataset'}
        results['notes'].append("JUNB not available for secondary correlations")

    # Step 6: Classification
    n = len(pbulk)
    log(f"N = {n}")

    if rho_p21 >= 0.70 and pval_p21 < 0.05:
        verdict = "STRONG REPLICATION"
        confidence = "SUGGESTED"
    elif rho_p21 >= 0.50 and pval_p21 < 0.05:
        verdict = "MODERATE/INCONCLUSIVE"
        confidence = "INCONCLUSIVE"
    else:
        verdict = "WEAK/ABSENT"
        confidence = "INCONCLUSIVE"

    results['verdict'] = verdict
    results['confidence'] = confidence

    # Power note
    power_note = ""
    if n < 15:
        power_note = f"WARNING: N={n} provides only ~50-65% power for rho=0.50; low power limits interpretation"
        results['notes'].append(power_note)

    log(f"VERDICT: {verdict}")

    # Comparison to F093
    f093_rho = 0.9410
    delta = rho_p21 - f093_rho
    results['comparison_to_f093'] = {
        'f093_rho': f093_rho,
        'batch037_rho': float(rho_p21),
        'delta': float(delta),
        'interpretation': f"Cross-atlas delta = {delta:+.3f}"
    }
    log(f"F093 comparison: delta = {delta:+.3f}")

    # Save results
    results['status'] = 'complete'
    results_file = f"{OUTPUT_DIR}/results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {results_file}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
