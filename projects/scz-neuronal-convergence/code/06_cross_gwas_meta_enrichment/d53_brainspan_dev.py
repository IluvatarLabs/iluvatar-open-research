#!/usr/bin/env python3
"""
D46 (rescued via D53): BrainSpan Developmental Trajectory Analysis

Tests whether SCZ gene enrichment differs by developmental stage (prenatal vs early postnatal vs adult).
This was originally BLOCKED because the API URL was wrong, but BrainSpan data IS accessible
via direct download from brainspan.org.

Groups:
- Prenatal: 8-37 pcw (post-conceptional weeks)
- Early postnatal: 4 months - 5 years
- Adult: 19-40 years

For each gene, identify peak-expression developmental stage.
Then test whether SCZ genes are enriched in one stage using Fisher's exact.

Author: Marvin (iteration 041)
"""

import json
import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact
from pathlib import Path

PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_041"
OUTPUT_DIR = BATCH_DIR / "output"
BS_DIR = OUTPUT_DIR / "brainspan_rnaseq_genes"

# Gene lists
PARDINAS_PATH = PROJECT_ROOT / "experiments" / "batch_008" / "data" / "gwas_genes.parquet"
PGC3_TABLE_PATH = PROJECT_ROOT / "data" / "19426775" / "scz2022-Extended-Data-Table1.xlsx"
NEURONAL_MARKERS_PATH = PROJECT_ROOT / "experiments" / "batch_009" / "data" / "markers.parquet"

def parse_age(age_str):
    """Parse BrainSpan age string to a comparable numeric value in weeks."""
    age_str = age_str.strip('"')
    if 'pcw' in age_str:
        return float(age_str.replace(' pcw', ''))
    elif 'mos' in age_str:
        months = float(age_str.replace(' mos', ''))
        return 40 + months * 4.33  # convert to weeks from conception
    elif 'yrs' in age_str:
        years = float(age_str.replace(' yrs', ''))
        return 40 + years * 52  # convert to weeks from conception
    return None

def classify_stage(age_str):
    """Classify age into developmental stage."""
    weeks = parse_age(age_str)
    if weeks is None:
        return None
    if weeks < 40:  # prenatal
        return 'prenatal'
    elif weeks < 40 + 5 * 52:  # 0-5 years = early postnatal
        return 'early_postnatal'
    else:  # > 5 years = adult/late
        return 'adult'

def main():
    print("=" * 70)
    print("D46/D53: BrainSpan Developmental Trajectory Analysis")
    print("=" * 70)

    # 1. Load metadata
    print("\n[1] Loading BrainSpan metadata...")
    col_meta = pd.read_csv(BS_DIR / "columns_metadata.csv")
    row_meta = pd.read_csv(BS_DIR / "rows_metadata.csv")

    # Classify developmental stages
    col_meta['stage'] = col_meta['age'].apply(classify_stage)
    stage_counts = col_meta['stage'].value_counts()
    print(f"Samples per stage: {stage_counts.to_dict()}")
    print(f"Total samples: {len(col_meta)}, with stage: {col_meta['stage'].notna().sum()}")

    # 2. Load expression matrix (large — 175MB)
    print("\n[2] Loading expression matrix...")
    expr = pd.read_csv(BS_DIR / "expression_matrix.csv", header=None)
    print(f"Expression matrix shape: {expr.shape}")  # genes × samples

    # 3. Map columns to stages
    stages = col_meta['stage'].values

    # 4. For each gene, compute median expression per stage
    print("\n[3] Computing per-stage median expression per gene...")
    prenatal_cols = np.where(stages == 'prenatal')[0]
    early_post_cols = np.where(stages == 'early_postnatal')[0]
    adult_cols = np.where(stages == 'adult')[0]

    expr_arr = expr.values  # numpy array for speed

    prenatal_med = np.median(expr_arr[:, prenatal_cols], axis=1)
    early_post_med = np.median(expr_arr[:, early_post_cols], axis=1)
    adult_med = np.median(expr_arr[:, adult_cols], axis=1)

    # 5. Assign peak stage to each gene
    print("\n[4] Assigning peak developmental stage per gene...")
    all_meds = np.column_stack([prenatal_med, early_post_med, adult_med])
    peak_idx = np.argmax(all_meds, axis=1)
    stage_labels = ['prenatal', 'early_postnatal', 'adult']
    peak_stages = [stage_labels[i] for i in peak_idx]

    row_meta['peak_stage'] = peak_stages
    row_meta['prenatal_med'] = prenatal_med
    row_meta['early_post_med'] = early_post_med
    row_meta['adult_med'] = adult_med

    # Filter to protein-coding genes only (with gene symbol)
    gene_info = row_meta[row_meta['gene_symbol'].notna()].copy()
    print(f"Genes with symbols: {len(gene_info)}")

    # Distribution of peak stages
    peak_dist = gene_info['peak_stage'].value_counts()
    print(f"Peak stage distribution (all genes): {peak_dist.to_dict()}")

    # === SCIENCE-CRITIC CORRECTION ===
    # 46% of genes have zero median expression in ALL stages.
    # np.argmax defaults to index 0 (prenatal), inflating prenatal baseline.
    # Fix: filter to genes with max median TPM >= 1.0 before analysis.
    max_med = np.max(all_meds, axis=1)
    gene_info['max_med'] = max_med
    expressed_mask = gene_info['max_med'] >= 1.0

    n_zero = (max_med == 0).sum()
    print(f"\nGenes with all-zero medians (np.argmax artifact): {n_zero} ({100*n_zero/len(gene_info):.1f}%)")

    gene_info_expressed = gene_info[expressed_mask].copy()
    peak_dist_expressed = gene_info_expressed['peak_stage'].value_counts()
    print(f"After TPM>=1.0 filter: {len(gene_info_expressed)} genes")
    print(f"Peak stage distribution (TPM>=1): {peak_dist_expressed.to_dict()}")

    # 6. Load SCZ gene lists
    print("\n[5] Loading SCZ gene lists...")
    pardinas_df = pd.read_parquet(PARDINAS_PATH)
    pardinas_genes = set(pardinas_df['hgnc_symbol'].str.upper().tolist())
    print(f"Pardiñas genes: {len(pardinas_genes)}")

    pgc3_xlsx = pd.ExcelFile(PGC3_TABLE_PATH)
    sheet1 = pd.read_excel(pgc3_xlsx, sheet_name='Extended.Data.Table.1')
    if 'gene_biotype' in sheet1.columns:
        protein_coding = sheet1[sheet1['gene_biotype'] == 'protein_coding']
        pgc3_genes = set(protein_coding['Symbol.ID'].str.upper().tolist())
    else:
        pgc3_genes = set(sheet1['Symbol.ID'].str.upper().tolist())
    print(f"PGC3 genes: {len(pgc3_genes)}")

    combined_genes = pardinas_genes | pgc3_genes
    print(f"Combined: {len(combined_genes)}, overlap: {len(pardinas_genes & pgc3_genes)}")

    # 7. Load neuronal markers
    print("\n[6] Loading neuronal markers...")
    markers_df = pd.read_parquet(NEURONAL_MARKERS_PATH)
    neuronal_markers = set(markers_df[markers_df['cell_type'].str.lower() == 'neurons']['gene'].str.upper().tolist())
    print(f"Neuronal markers: {len(neuronal_markers)}")

    # 8. Create gene universe (BrainSpan genes with symbols, TPM >= 1 filter)
    gene_info_expressed['gene_upper'] = gene_info_expressed['gene_symbol'].str.upper()
    gene_universe = set(gene_info_expressed['gene_upper'].tolist())
    print(f"Gene universe (BrainSpan, TPM>=1): {len(gene_universe)}")

    # 9. Enrichment analysis per stage × gene list
    print("\n[7] Enrichment analysis: SCZ genes by developmental stage (TPM>=1 filter)...")
    print("=" * 70)

    results = {}
    for gene_list_name, gene_list in [('Pardiñas', pardinas_genes), ('PGC3', pgc3_genes), ('Combined', combined_genes)]:
        scz_in_universe = gene_list & gene_universe
        print(f"\n{gene_list_name}: {len(scz_in_universe)} genes in BrainSpan universe (TPM>=1)")

        for stage in ['prenatal', 'early_postnatal', 'adult']:
            stage_genes = set(gene_info_expressed[gene_info_expressed['peak_stage'] == stage]['gene_upper'].tolist())
            overlap = scz_in_universe & stage_genes

            # Fisher's exact test
            k = len(overlap)
            n_scz = len(scz_in_universe)
            n_stage = len(stage_genes)
            n_bg = len(gene_universe)

            a = k  # SCZ and stage
            b = n_scz - k  # SCZ not stage
            c = n_stage - k  # stage not SCZ
            d = n_bg - n_scz - n_stage + k  # neither

            if a > 0 and b > 0:
                OR, p = fisher_exact([[a, b], [c, d]], alternative='greater')
            else:
                OR, p = 0.0, 1.0

            pct = 100 * k / n_scz if n_scz > 0 else 0

            results[f"{gene_list_name}_{stage}"] = {
                'gene_list': gene_list_name,
                'stage': stage,
                'overlap': int(k),
                'n_scz': int(n_scz),
                'n_stage': int(n_stage),
                'n_universe': int(n_bg),
                'pct_in_stage': round(pct, 1),
                'OR': round(float(OR), 3),
                'p_value': float(p),
                'significant_005': bool(p < 0.05)
            }

            print(f"  {stage:>18s}: {k:4d}/{n_scz} ({pct:5.1f}%) genes peak here | OR={OR:.2f}, p={p:.4f} {'*' if p < 0.05 else ''}")

    # 10. Also test: neuronal SCZ genes by stage
    print("\n[8] Enrichment: Neuronal SCZ genes by developmental stage (TPM>=1 filter)...")
    neuronal_scz = combined_genes & neuronal_markers & gene_universe
    print(f"Neuronal SCZ genes in BrainSpan (TPM>=1): {len(neuronal_scz)}")

    for stage in ['prenatal', 'early_postnatal', 'adult']:
        stage_genes = set(gene_info_expressed[gene_info_expressed['peak_stage'] == stage]['gene_upper'].tolist())
        overlap = neuronal_scz & stage_genes

        k = len(overlap)
        n_scz = len(neuronal_scz)
        n_stage = len(stage_genes)
        n_bg = len(gene_universe)

        a = k
        b = n_scz - k
        c = n_stage - k
        d = n_bg - n_scz - n_stage + k

        if a > 0 and b > 0:
            OR, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        else:
            OR, p = 0.0, 1.0

        pct = 100 * k / n_scz if n_scz > 0 else 0

        results[f"neuronal_scz_{stage}"] = {
            'gene_list': 'neuronal_scz',
            'stage': stage,
            'overlap': int(k),
            'n_scz': int(n_scz),
            'n_stage': int(n_stage),
            'n_universe': int(n_bg),
            'pct_in_stage': round(pct, 1),
            'OR': round(float(OR), 3),
            'p_value': float(p)
        }

        print(f"  {stage:>18s}: {k:4d}/{n_scz} ({pct:5.1f}%) neuronal SCZ genes | OR={OR:.2f}, p={p:.4f} {'*' if p < 0.05 else ''}")

    # 11. Sensitivity: expression-weighted enrichment
    print("\n[9] Expression-weighted analysis: mean expression per stage for SCZ vs non-SCZ genes (TPM>=1 filter)...")
    gene_info_expressed['is_scz'] = gene_info_expressed['gene_upper'].isin(combined_genes)
    neuronal_markers_set = set(markers_df[markers_df['cell_type'].str.lower() == 'neurons']['gene'].str.upper().tolist())
    gene_info_expressed['is_neuronal'] = gene_info_expressed['gene_upper'].isin(neuronal_markers_set)

    for stage_name, med_col in [('prenatal', 'prenatal_med'), ('early_postnatal', 'early_post_med'), ('adult', 'adult_med')]:
        scz_expr = gene_info_expressed[gene_info_expressed['is_scz']][med_col]
        nonscz_expr = gene_info_expressed[~gene_info_expressed['is_scz']][med_col]

        # Mann-Whitney U test (non-parametric)
        u_stat, u_p = stats.mannwhitneyu(scz_expr, nonscz_expr, alternative='greater')
        median_diff = scz_expr.median() - nonscz_expr.median() if len(scz_expr) > 0 else 0

        print(f"  {stage_name:>18s}: SCZ median={scz_expr.median():.2f}, non-SCZ median={nonscz_expr.median():.2f}, U p={u_p:.4f}")

    # 12. Save results
    print("\n[10] Saving results...")
    output = {
        'analysis': 'D46_brainspan_developmental_trajectory',
        'date': '2026-04-21',
        'iteration': 41,
        'data_source': 'BrainSpan Atlas of the Developing Human Brain (brainspan.org)',
        'gene_universe_size': len(gene_universe),
        'peak_stage_distribution': peak_dist_expressed.to_dict(),
        'filter': 'TPM >= 1.0 (science-critic correction: removes 24K zero-expression genes with np.argmax artifact)',
        'enrichment_results': results
    }

    with open(OUTPUT_DIR / 'd46_brainspan_dev_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Also save as TSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv(OUTPUT_DIR / 'd46_brainspan_dev_results.tsv', sep='\t')

    print(f"\nResults saved to {OUTPUT_DIR / 'd46_brainspan_dev_results.json'}")
    print(f"Results saved to {OUTPUT_DIR / 'd46_brainspan_dev_results.tsv'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, res in results.items():
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        print(f"  {key:>35s}: OR={res['OR']:6.2f}, p={res['p_value']:.4f} {sig}")

if __name__ == '__main__':
    main()
