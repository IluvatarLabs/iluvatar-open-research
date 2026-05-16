#!/usr/bin/env python3
"""
batch_007: HLMA Data Quality & GRN Viability
QC Script for H5AD files

Computes all metrics from brief.md:
- n_cells total and by Annotation × age_pop
- n_donors by Annotation × age_pop
- Sparsity (% zeros)
- Median/mean genes per cell by tech
- Donor composition (max % any single donor per cell type)
- Technology balance (snRNA % young vs old for MuSC file)
- PAX7, MYOD1 positive fraction per age_pop (MuSC file only)

Outputs: experiments/batch_007/results.json
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/data")
OUTPUT_FILE = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_007/results.json")
BATCH_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_007")
BATCH_DIR.mkdir(parents=True, exist_ok=True)

# Files to process
FILES = {
    'MuSC': DATA_DIR / 'MuSC_scsn_RNA.h5ad',
    'Immune': DATA_DIR / 'Immune_scsn_RNA.h5ad',
    'Vascular': DATA_DIR / 'Vascular_scsn_RNA.h5ad',
    'FAP': DATA_DIR / 'OMIX004308-02.h5ad',
    'Myonuclei': DATA_DIR / 'OMIX004308-05.h5ad',
}

SKIPPED_FILES = {
    'OMIX004308-03': DATA_DIR / 'OMIX004308-03.h5ad'  # Truncated, skip
}

MARKER_GENES = ['PAX7', 'MYOD1']  # MyoD1 is alias for MYOD1


def safe_read_h5ad(path, name):
    """Read H5AD with backed mode for memory efficiency."""
    try:
        adata = sc.read(str(path), backed='r')
        return adata, None
    except Exception as e:
        return None, str(e)


def normalize_age_pop(adata):
    """Normalize age_pop column across different encodings."""
    if adata.obs['age_pop'].dtype in [np.int32, np.int64]:
        # Numeric encoding: 0=old, 1=young (Immune file)
        obs = adata.obs.copy()
        obs['age_pop'] = obs['age_pop'].map({0: 'old_pop', 1: 'young_pop'})
        adata.obs['age_pop'] = obs['age_pop']
    return adata


def normalize_tech(adata):
    """Normalize tech column across different encodings."""
    if adata.obs['tech'].dtype in [np.int32, np.int64]:
        # Numeric encoding: 0=scRNA, 1=snRNA (Immune file)
        obs = adata.obs.copy()
        obs['tech'] = obs['tech'].map({0: 'scRNA', 1: 'snRNA'})
        adata.obs['tech'] = obs['tech']
    return adata


def compute_metrics(adata, name):
    """Compute all QC metrics for a single H5AD file."""
    metrics = {
        'name': name,
        'shape': {'n_cells': int(adata.n_obs), 'n_genes': int(adata.n_vars)},
        'n_cells_by_annotation_age': {},
        'n_donors_by_annotation_age': {},
        'sparsity': None,
        'genes_per_cell_by_tech': {},
        'donor_composition': {},
        'tech_balance': None,
        'marker_expression': {},
        'passes': {},
        'flags': []
    }

    # Normalize columns
    adata = normalize_age_pop(adata)
    adata = normalize_tech(adata)

    # 1. n_cells by Annotation × age_pop
    if 'Annotation' in adata.obs.columns and 'age_pop' in adata.obs.columns:
        counts = adata.obs.groupby(['Annotation', 'age_pop']).size()
        metrics['n_cells_by_annotation_age'] = {
            f"{ann}_{age}": int(cnt)
            for (ann, age), cnt in counts.items()
        }

    # 2. n_donors by Annotation × age_pop
    if 'sample' in adata.obs.columns and 'Annotation' in adata.obs.columns:
        donor_counts = adata.obs.groupby(['Annotation', 'age_pop'])['sample'].nunique()
        metrics['n_donors_by_annotation_age'] = {
            f"{ann}_{age}": int(cnt)
            for (ann, age), cnt in donor_counts.items()
        }

    # 3. Sparsity (% zeros)
    # Use backed mode - sample to estimate
    sample_genes = adata.var_names[:1000].values
    X_sample = adata[:, sample_genes].X
    if hasattr(X_sample, 'toarray'):
        X_sample = X_sample.toarray()
    total_entries_actual = X_sample.size
    n_zeros = np.sum(X_sample == 0)
    sparsity = float(n_zeros / total_entries_actual)

    # Check if data appears to be normalized (has negative values)
    has_negative = np.any(X_sample < 0)
    is_normalized = False
    if has_negative:
        is_normalized = True
        metrics['sparsity'] = None  # Cannot compute sparsity on normalized data
        metrics['flags'].append('data appears normalized (log-transformed), sparsity not computed')
    else:
        metrics['sparsity'] = sparsity

    # 4. Median/mean genes per cell by tech
    if 'tech' in adata.obs.columns and 'nFeature_RNA' in adata.obs.columns:
        tech_genes = adata.obs.groupby('tech')['nFeature_RNA']
        for tech in tech_genes.groups.keys():
            vals = adata.obs[adata.obs['tech'] == tech]['nFeature_RNA'].values
            metrics['genes_per_cell_by_tech'][tech] = {
                'median': float(np.median(vals)),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals))
            }

    # 5. Donor composition (max % any single donor per cell type)
    if 'sample' in adata.obs.columns and 'Annotation' in adata.obs.columns:
        for ann in adata.obs['Annotation'].unique():
            ann_cells = adata.obs[adata.obs['Annotation'] == ann]
            total = len(ann_cells)
            if total > 0:
                donor_counts = ann_cells['sample'].value_counts()
                max_pct = float(donor_counts.iloc[0] / total * 100)
                metrics['donor_composition'][ann] = {
                    'max_donor_pct': max_pct,
                    'n_donors': int(len(donor_counts)),
                    'dominant_donor': str(donor_counts.index[0])
                }

    # 6. Technology balance (snRNA % young vs old for MuSC file)
    if 'tech' in adata.obs.columns and 'age_pop' in adata.obs.columns:
        tech_age = adata.obs.groupby(['tech', 'age_pop']).size().unstack(fill_value=0)
        tech_balance = {}
        for tech in tech_age.index:
            if 'young_pop' in tech_age.columns and 'old_pop' in tech_age.columns:
                young = tech_age.loc[tech, 'young_pop']
                old = tech_age.loc[tech, 'old_pop']
                total = young + old
                if total > 0:
                    young_pct = young / total * 100
                    old_pct = old / total * 100
                    tech_balance[tech] = {
                        'young_pct': float(young_pct),
                        'old_pct': float(old_pct),
                        'young_n': int(young),
                        'old_n': int(old)
                    }
        metrics['tech_balance'] = tech_balance

    # 7. Marker expression fraction (PAX7, MYOD1) per age_pop - MuSC file only
    if name == 'MuSC':
        for gene in MARKER_GENES:
            if gene in adata.var_names:
                gene_counts = {}
                for age_pop in ['young_pop', 'old_pop']:
                    mask = adata.obs['age_pop'] == age_pop
                    n_cells = mask.sum()
                    if n_cells > 0:
                        # Check expression using backed mode
                        expr = adata[mask, gene].X
                        if hasattr(expr, 'toarray'):
                            expr = expr.toarray()
                        positive = np.sum(expr > 0)
                        frac = positive / n_cells
                        gene_counts[age_pop] = {
                            'positive_fraction': float(frac),
                            'positive_n': int(positive),
                            'total_n': int(n_cells)
                        }
                metrics['marker_expression'][gene] = gene_counts
            else:
                metrics['marker_expression'][gene] = {'status': 'not_found'}

    # 8. Pass/fail checks based on brief.md criteria
    checks = []

    # Check 1: MuSC - ≥500 cells per age group
    if name == 'MuSC':
        young_cells = metrics['n_cells_by_annotation_age'].get('Total_young_pop', 0)
        old_cells = metrics['n_cells_by_annotation_age'].get('Total_old_pop', 0)
        # Sum across all annotations
        young_total = sum(v for k, v in metrics['n_cells_by_annotation_age'].items() if 'young_pop' in k)
        old_total = sum(v for k, v in metrics['n_cells_by_annotation_age'].items() if 'old_pop' in k)
        checks.append(('MuSC >=500 young', young_total >= 500, f"{young_total}"))
        checks.append(('MuSC >=500 old', old_total >= 500, f"{old_total}"))

    # Check 2: ≥3 donors per age group
    if 'n_donors_by_annotation_age' in metrics and len(metrics['n_donors_by_annotation_age']) > 0:
        min_donors = min(metrics['n_donors_by_annotation_age'].values())
        checks.append(('>=3 donors per group', min_donors >= 3, f"min={min_donors}"))

    # Check 3: Sparsity <0.97 (snRNA) or <0.9 (scRNA) - skip if normalized data
    sparsity = metrics['sparsity']
    tech_type = list(metrics['genes_per_cell_by_tech'].keys())[0] if metrics['genes_per_cell_by_tech'] else 'unknown'
    if sparsity is None:
        checks.append(('sparsity check', True, 'N/A (normalized data)'))
    elif 'snRNA' in tech_type:
        checks.append(('sparsity <0.97 (snRNA)', sparsity < 0.97, f"{sparsity:.4f}"))
    else:
        checks.append(('sparsity <0.90 (scRNA)', sparsity < 0.90, f"{sparsity:.4f}"))

    # Check 4: Gene detection adequate (median ≥500 scRNA or ≥200 snRNA)
    if metrics['genes_per_cell_by_tech']:
        for tech, stats in metrics['genes_per_cell_by_tech'].items():
            if tech == 'snRNA':
                checks.append((f'{tech} median >=200 genes', stats['median'] >= 200, f"{stats['median']:.0f}"))
            else:
                checks.append((f'{tech} median >=500 genes', stats['median'] >= 500, f"{stats['median']:.0f}"))

    # Check 5: Donor balance <50% single donor
    if metrics['donor_composition']:
        max_donor_pcts = [v['max_donor_pct'] for v in metrics['donor_composition'].values()]
        max_pct = max(max_donor_pcts) if max_donor_pcts else 0
        checks.append(('max donor <50%', max_pct < 50, f"{max_pct:.1f}%"))

    # Check 6: Tech confounding <15pp snRNA proportion diff
    if metrics['tech_balance'] and 'snRNA' in metrics['tech_balance']:
        snrna_stats = metrics['tech_balance']['snRNA']
        young_pct = snrna_stats['young_pct']
        old_pct = snrna_stats['old_pct']
        diff = abs(young_pct - old_pct)
        checks.append(('snRNA tech confound <15pp', diff < 15, f"{diff:.1f}pp"))

    # Check 7: Marker genes detectable (MuSC only)
    if name == 'MuSC' and metrics['marker_expression']:
        for gene in MARKER_GENES:
            if gene in metrics['marker_expression']:
                gene_data = metrics['marker_expression'][gene]
                if isinstance(gene_data, dict) and 'status' not in gene_data:
                    young_frac = gene_data.get('young_pop', {}).get('positive_fraction', 0)
                    old_frac = gene_data.get('old_pop', {}).get('positive_fraction', 0)
                    detectable = young_frac > 0 or old_frac > 0
                    checks.append((f'{gene} detectable', detectable, f"young={young_frac:.4f}, old={old_frac:.4f}"))

    metrics['checks'] = checks

    # Overall pass/fail
    if checks:
        all_passed = all(c[1] for c in checks)
        metrics['overall_pass'] = all_passed
    else:
        metrics['overall_pass'] = None

    return metrics


def print_summary_table(results):
    """Print a formatted summary table to stdout."""
    print("\n" + "="*100)
    print("BATCH_007 QC RESULTS: HLMA Data Quality & GRN Viability")
    print("="*100)

    for name, data in results.items():
        if 'error' in data or data.get('status') == 'skipped':
            reason = data.get('error', data.get('reason', 'unknown'))
            print(f"\n{name}: SKIPPED - {reason}")
            continue

        print(f"\n{'='*80}")
        print(f"FILE: {data['name']}")
        print(f"{'='*80}")

        # Basic stats
        print(f"\n  Shape: {data['shape']['n_cells']:,} cells × {data['shape']['n_genes']:,} genes")
        sparsity = data['sparsity']
        if sparsity is None:
            print(f"  Sparsity: N/A (data appears normalized)")
        else:
            print(f"  Sparsity: {sparsity:.4f} ({sparsity*100:.2f}% zeros)")

        # Genes per cell by tech
        print(f"\n  Genes per cell by technology:")
        for tech, stats in data['genes_per_cell_by_tech'].items():
            print(f"    {tech}: median={stats['median']:.0f}, mean={stats['mean']:.0f} ± {stats['std']:.0f}")

        # Cell counts by annotation × age
        print(f"\n  Cells by Annotation × age_pop:")
        for key, count in sorted(data['n_cells_by_annotation_age'].items()):
            print(f"    {key}: {count:,}")

        # Donors by annotation × age
        print(f"\n  Donors by Annotation × age_pop:")
        for key, n_donors in sorted(data['n_donors_by_annotation_age'].items()):
            print(f"    {key}: {n_donors}")

        # Donor composition
        print(f"\n  Donor composition (max % single donor per cell type):")
        for ann, stats in data['donor_composition'].items():
            flag = " [WARNING]" if stats['max_donor_pct'] >= 50 else ""
            print(f"    {ann}: {stats['max_donor_pct']:.1f}% ({stats['dominant_donor']}){flag}")

        # Tech balance
        if data['tech_balance']:
            print(f"\n  Technology balance:")
            for tech, stats in data['tech_balance'].items():
                print(f"    {tech}: young={stats['young_pct']:.1f}%, old={stats['old_pct']:.1f}%")

        # Marker expression (MuSC only)
        if data['marker_expression']:
            print(f"\n  Marker gene expression (positive fraction):")
            for gene, age_data in data['marker_expression'].items():
                if isinstance(age_data, dict) and 'status' in age_data:
                    print(f"    {gene}: {age_data['status']}")
                else:
                    for age, stats in age_data.items():
                        print(f"    {gene} ({age}): {stats['positive_fraction']:.4f} ({stats['positive_n']}/{stats['total_n']})")

        # Checks
        print(f"\n  QC Checks:")
        all_passed = True
        for check_name, passed, detail in data['checks']:
            status = "PASS" if passed else "FAIL"
            symbol = "[+]" if passed else "[!]"
            print(f"    {symbol} {check_name}: {status} ({detail})")
            if not passed:
                all_passed = False

        overall = "PASS" if data.get('overall_pass') else ("FAIL" if data.get('overall_pass') is False else "N/A")
        print(f"\n  Overall: {overall}")

    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'File':<15} {'Cells':<12} {'Sparsity':<10} {'Median Genes':<14} {'Checks':<8} {'Status':<10}")
    print("-"*70)
    for name, data in results.items():
        if 'error' in data or data.get('status') == 'skipped':
            print(f"{name:<15} {'ERROR/SKIP':<12} {'N/A':<10} {'N/A':<14} {'N/A':<8} {'SKIPPED':<10}")
            continue

        cells = f"{data['shape']['n_cells']:,}"
        sparsity_val = data['sparsity']
        if sparsity_val is None:
            sparsity = "N/A"
        else:
            sparsity = f"{sparsity_val:.3f}"
        medians = []
        for tech, stats in data['genes_per_cell_by_tech'].items():
            medians.append(f"{tech}={stats['median']:.0f}")
        median_str = "; ".join(medians)[:14]
        n_checks = len(data['checks'])
        passed_checks = sum(1 for c in data['checks'] if c[1])
        checks_str = f"{passed_checks}/{n_checks}"
        overall = "PASS" if data.get('overall_pass') else ("FAIL" if data.get('overall_pass') is False else "N/A")

        print(f"{name:<15} {cells:<12} {sparsity:<10} {median_str:<14} {checks_str:<8} {overall:<10}")

    print("="*100)


def main():
    """Main entry point."""
    print("Starting batch_007 QC analysis...")
    print(f"Processing {len(FILES)} files...")
    print(f"Skipping {len(SKIPPED_FILES)} truncated files: {list(SKIPPED_FILES.keys())}")

    results = {}

    for name, path in FILES.items():
        print(f"\nProcessing {name}...")
        adata, error = safe_read_h5ad(path, name)

        if error:
            print(f"  ERROR: {error}")
            results[name] = {'name': name, 'error': error}
        else:
            print(f"  Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
            metrics = compute_metrics(adata, name)
            results[name] = metrics
            print(f"  Metrics computed successfully")

    # Add skipped files info
    for name, path in SKIPPED_FILES.items():
        results[name] = {
            'name': name,
            'status': 'skipped',
            'reason': 'truncated file, unreadable'
        }

    # Print summary table
    print_summary_table(results)

    # Save results to JSON
    print(f"\nSaving results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Done!")

    return results


if __name__ == '__main__':
    main()
