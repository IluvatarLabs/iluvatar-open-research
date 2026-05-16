#!/usr/bin/env python3
"""
D45: Gene-Length-Matched Permutation Analysis (Optimized)

Tests whether neuronal enrichment (F013 OR=9.76, p=1.79e-10) survives gene-length conditioning.
Uses ratio-based matching (±10% tolerance) to control for the confound that longer genes
have more SNPs and are more likely to be "GWS by gene mapping."

Author: Marvin (Experimentalist)
Date: 2026-04-17
"""

import gzip
import random
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Configuration
N_PERMUTATIONS = 10000
N_SCZ_GENES = 444
LENGTH_TOLERANCE = 0.10  # ±10% ratio-based tolerance
RANDOM_SEED = 42
OUTPUT_DIR = "/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_040/output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70, flush=True)
print("D45: Gene-Length-Matched Permutation Analysis", flush=True)
print("=" * 70, flush=True)
print(f"Permutations: {N_PERMUTATIONS}", flush=True)
print(f"SCZ gene count: {N_SCZ_GENES}", flush=True)
print(f"Length tolerance: ±{LENGTH_TOLERANCE*100}%", flush=True)
print(f"Random seed: {RANDOM_SEED}", flush=True)
print(flush=True)

# =============================================================================
# Step 1: Load SCZ genes with gene lengths from batch_008
# =============================================================================
print("[1/6] Loading SCZ genes from batch_008...", flush=True)
gwas_df = pd.read_parquet("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_008/data/gwas_genes.parquet")
gwas_df['gene_length'] = gwas_df['gene_end'] - gwas_df['gene_start']

# Use hgnc_symbol as gene_name (standard gene naming)
scz_genes = set(gwas_df['hgnc_symbol'].dropna().str.upper().tolist())
scz_gene_lengths = dict(zip(gwas_df['hgnc_symbol'].str.upper(), gwas_df['gene_length']))

print(f"  Loaded {len(scz_genes)} SCZ genes", flush=True)
print(f"  SCZ gene length: mean={np.mean(list(scz_gene_lengths.values())):.0f}bp, "
      f"median={np.median(list(scz_gene_lengths.values())):.0f}bp", flush=True)

# =============================================================================
# Step 2: Extract protein-coding genes from GenCode GTF (OPTIMIZED)
# =============================================================================
print("\n[2/6] Extracting protein-coding genes from GenCode v44...", flush=True)
gencode_path = "/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_026/gencode.v44.annotation.gtf.gz"

# Cache file for faster re-runs
cache_path = f"{OUTPUT_DIR}/gencode_protein_coding_genes.json"

if os.path.exists(cache_path):
    print("  Loading from cache...", flush=True)
    with open(cache_path, 'r') as f:
        cache = json.load(f)
    gene_lengths_all = {k: {'length': v['length'], 'gene_id': v['gene_id'], 'chrom': v['chrom']}
                        for k, v in cache['genes'].items()}
    gene_names_all = set(gene_lengths_all.keys())
else:
    gene_lengths_all = {}
    gene_names_all = set()
    line_count = 0
    chunk_size = 100000

    with gzip.open(gencode_path, 'rt') as f:
        while True:
            lines = []
            for _ in range(chunk_size):
                line = f.readline()
                if not line:
                    break
                lines.append(line)

            if not lines:
                break

            line_count += len(lines)

            for line in lines:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                feature_type = fields[2]
                if feature_type != 'gene':
                    continue

                # Quick filter for protein-coding
                if 'protein_coding' not in line:
                    continue

                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                gene_length = end - start

                attr_str = fields[8]
                gene_name = None
                gene_id = None

                # Fast attribute parsing
                for attr in attr_str.split(';')[:-1]:
                    parts = attr.strip().split()
                    if len(parts) >= 2:
                        key = parts[0]
                        if key == 'gene_name':
                            gene_name = ' '.join(parts[1:]).strip('"')
                        elif key == 'gene_id':
                            gene_id = parts[1].strip('"')

                if gene_name and gene_id:
                    # Standard chromosomes only, exclude chrY
                    if chrom.startswith('chr') and chrom != 'chrY' and '_' not in chrom:
                        gene_names_all.add(gene_name.upper())
                        gene_lengths_all[gene_name.upper()] = {
                            'length': gene_length,
                            'gene_id': gene_id,
                            'chrom': chrom
                        }

            if line_count % 500000 == 0:
                print(f"    Processed {line_count:,} lines, found {len(gene_names_all)} protein-coding genes", flush=True)

    # Save cache
    cache_data = {
        'genes': {k: {'length': v['length'], 'gene_id': v['gene_id'], 'chrom': v['chrom']}
                  for k, v in gene_lengths_all.items()}
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
    print(f"  Saved cache to {cache_path}", flush=True)

# Background genes = all protein-coding genes not in SCZ gene set
background_genes = list(set(gene_names_all) - scz_genes)
print(f"  Total protein-coding genes: {len(gene_names_all)}", flush=True)
print(f"  Background genes (non-SCZ): {len(background_genes)}", flush=True)
print(f"  Background gene length: mean={np.mean([gene_lengths_all[g]['length'] for g in background_genes]):.0f}bp, "
      f"median={np.median([gene_lengths_all[g]['length'] for g in background_genes]):.0f}bp", flush=True)

# =============================================================================
# Step 3: Load neuronal markers from batch_009
# =============================================================================
print("\n[3/6] Loading neuronal markers from batch_009...", flush=True)
markers_df = pd.read_parquet("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_009/data/markers.parquet")

# Filter for Neurons
neurons_df = markers_df[markers_df['cell_type'] == 'Neurons']
neuronal_markers = set(neurons_df['gene'].dropna().str.upper().tolist())
print(f"  Neuronal markers: {len(neuronal_markers)}", flush=True)

# =============================================================================
# Step 4: Compute observed overlap
# =============================================================================
print("\n[4/6] Computing observed overlap (k_obs)...", flush=True)
# SCZ genes that are also neuronal markers
scz_neuronal = scz_genes.intersection(neuronal_markers)
k_obs = len(scz_neuronal)
print(f"  SCZ genes that are neuronal markers: {k_obs}", flush=True)
print(f"  Example SCZ-neuronal genes: {list(scz_neuronal)[:10]}", flush=True)

# =============================================================================
# Step 5: Gene-length-matched permutation (VECTORIZED)
# =============================================================================
print(f"\n[5/6] Running {N_PERMUTATIONS:,} gene-length-matched permutations...", flush=True)

# Pre-compute gene lengths array for fast sampling
bg_lengths = np.array([gene_lengths_all[g]['length'] for g in background_genes])
bg_genes_arr = np.array(background_genes)

# Get SCZ gene lengths for matching
scz_lengths_arr = np.array([scz_gene_lengths.get(g, 0) for g in scz_genes])

def sample_length_matched_null_fast(n_genes: int, scz_lengths: np.ndarray,
                                      bg_lengths: np.ndarray, bg_genes: np.ndarray,
                                      tolerance: float = 0.10) -> Tuple[set, List]:
    """Fast length-matched sampling using vectorized operations."""
    sampled_genes = set()
    matched_lengths = []

    for target_len in scz_lengths:
        min_len = target_len * (1 - tolerance)
        max_len = target_len * (1 + tolerance)

        # Find candidates within length range
        mask = (bg_lengths >= min_len) & (bg_lengths <= max_len)
        candidates = np.where(mask)[0]

        if len(candidates) > 0:
            # Pick random candidate
            idx = np.random.choice(candidates)
            gene = bg_genes[idx]
            if gene not in sampled_genes:
                sampled_genes.add(gene)
                matched_lengths.append(gene_lengths_all[gene]['length'])

    # If we need more genes, sample randomly
    if len(sampled_genes) < n_genes:
        available = [g for g in bg_genes if g not in sampled_genes]
        needed = n_genes - len(sampled_genes)
        extra = np.random.choice(available, min(needed, len(available)), replace=False)
        sampled_genes.update(extra)

    return sampled_genes, matched_lengths

# Set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Run permutations
k_null_list = []
matched_lengths_list = []

for perm_i in range(N_PERMUTATIONS):
    if (perm_i + 1) % 1000 == 0:
        print(f"    Permutation {perm_i + 1:,}/{N_PERMUTATIONS:,}...", flush=True)

    sampled_set, matched_lens = sample_length_matched_null_fast(
        N_SCZ_GENES, scz_lengths_arr, bg_lengths, bg_genes_arr, LENGTH_TOLERANCE
    )

    # Compute overlap with neuronal markers
    k_perm = len(sampled_set.intersection(neuronal_markers))
    k_null_list.append(k_perm)
    matched_lengths_list.extend(matched_lens)

print(f"  Permutations complete", flush=True)

k_null = np.array(k_null_list)

# =============================================================================
# Step 6: Compute metrics
# =============================================================================
print("\n[6/6] Computing metrics...", flush=True)

# Empirical p-value: fraction of null >= observed
empirical_p = np.mean(k_null >= k_obs)

# Compute odds ratios
def compute_odds_ratio(overlap_count: int, total_genes: int, marker_count: int, background_count: int) -> Tuple[float, int, int, int, int]:
    """Compute odds ratio for enrichment test."""
    # 2x2 table:
    #               Neuronal    Non-Neuronal
    # In set        a            b
    # Not in set    c            d
    a = overlap_count  # Genes in set AND neuronal
    b = total_genes - overlap_count  # Genes in set AND not neuronal
    c = marker_count - overlap_count  # Genes not in set AND neuronal
    d = (background_count - total_genes) - c  # Genes not in set AND not neuronal

    # Odds ratio = (a*d) / (b*c)
    if b * c > 0:
        or_val = (a * d) / (b * c)
    else:
        or_val = float('inf') if (a * d) > 0 else 1.0

    return or_val, a, b, c, d

# OR for observed SCZ genes
background_n_neurons = len(set(gene_names_all).intersection(neuronal_markers))
or_obs, a_obs, b_obs, c_obs, d_obs = compute_odds_ratio(
    k_obs, N_SCZ_GENES, background_n_neurons, len(background_genes)
)

# OR for null distribution (median)
or_null_median = np.median([compute_odds_ratio(k, N_SCZ_GENES, background_n_neurons, len(background_genes))[0]
                             for k in k_null])

# Adjusted OR = ratio of observed to null median
adjusted_or = or_obs / or_null_median if or_null_median > 0 else float('inf')

# Attenuation ratio = (OR_obs - OR_null) / OR_obs
attenuation_ratio = (or_obs - or_null_median) / or_obs if or_obs > 0 else float('inf')

# =============================================================================
# Output Results
# =============================================================================
print("\n" + "=" * 70, flush=True)
print("RESULTS", flush=True)
print("=" * 70, flush=True)

print(f"\nEmpirical p-value: {empirical_p:.6f}", flush=True)
print(f"  (Fraction of {N_PERMUTATIONS:,} permutations with k_perm >= k_obs)", flush=True)
print(f"\nObserved overlap (k_obs): {k_obs}", flush=True)
print(f"Null distribution: mean={np.mean(k_null):.1f}, std={np.std(k_null):.1f}, "
      f"min={np.min(k_null)}, max={np.max(k_null)}", flush=True)

print(f"\nOdds Ratio (observed SCZ): {or_obs:.2f}", flush=True)
print(f"  2x2 table: a={a_obs}, b={b_obs}, c={c_obs}, d={d_obs}", flush=True)
print(f"Odds Ratio (null median): {or_null_median:.2f}", flush=True)
print(f"\nAdjusted OR (relative to null): {adjusted_or:.2f}", flush=True)
print(f"Attenuation ratio: {attenuation_ratio:.2f}", flush=True)

# Interpretation
print("\n" + "-" * 70, flush=True)
print("INTERPRETATION", flush=True)
print("-" * 70, flush=True)

if empirical_p < 0.01:
    significance = "SIGNIFICANT"
else:
    significance = "NOT SIGNIFICANT"

if attenuation_ratio < 0.5:
    attenuation_status = "GENUINE (minimal attenuation)"
else:
    attenuation_status = "REDUCED (substantial attenuation)"

print(f"Empirical p-value: {empirical_p:.6f} -> {significance}", flush=True)
print(f"Attenuation ratio: {attenuation_ratio:.2f} -> {attenuation_status}", flush=True)

if empirical_p < 0.01 and attenuation_ratio < 0.5:
    conclusion = "SURVIVES"
    print("\nCONCLUSION: Neuronal enrichment SURVIVES gene-length conditioning.", flush=True)
    print("The effect is genuine and not explained by gene length confounding.", flush=True)
elif empirical_p < 0.01 and attenuation_ratio >= 0.5:
    conclusion = "REDUCED"
    print("\nCONCLUSION: Neuronal enrichment REDUCED by gene-length conditioning.", flush=True)
    print("Effect partially explained by gene length, but remains significant.", flush=True)
else:
    conclusion = "NOT_SIGNIFICANT"
    print("\nCONCLUSION: Neuronal enrichment does NOT survive gene-length conditioning.", flush=True)
    print("Effect is largely explained by gene length confounding.", flush=True)

# =============================================================================
# Save output files
# =============================================================================
# Results file
results_df = pd.DataFrame({
    'metric': ['empirical_p', 'k_obs', 'or_obs', 'or_null_median', 'adjusted_or',
               'attenuation_ratio', 'n_permutations', 'n_scz_genes', 'n_neuronal_markers',
               'length_tolerance', 'null_mean', 'null_std', 'null_min', 'null_max'],
    'value': [empirical_p, k_obs, or_obs, or_null_median, adjusted_or,
              attenuation_ratio, N_PERMUTATIONS, N_SCZ_GENES, len(neuronal_markers),
              LENGTH_TOLERANCE, np.mean(k_null), np.std(k_null), np.min(k_null), np.max(k_null)]
})
results_path = f"{OUTPUT_DIR}/d45_gene_length_results.tsv"
results_df.to_csv(results_path, sep='\t', index=False)
print(f"\nSaved: {results_path}", flush=True)

# Null distribution file
null_df = pd.DataFrame({'k_null': k_null})
null_path = f"{OUTPUT_DIR}/d45_null_distribution.tsv"
null_df.to_csv(null_path, sep='\t', index=False)
print(f"Saved: {null_path}", flush=True)

# Save full results summary as JSON for easy parsing
summary = {
    'experiment': 'D45_gene_length_permutation',
    'hypothesis': 'F013 neuronal enrichment survives gene-length conditioning',
    'empirical_p': float(empirical_p),
    'k_obs': int(k_obs),
    'or_obs': float(or_obs),
    'or_null_median': float(or_null_median),
    'adjusted_or': float(adjusted_or),
    'attenuation_ratio': float(attenuation_ratio),
    'n_permutations': N_PERMUTATIONS,
    'n_scz_genes': N_SCZ_GENES,
    'n_neuronal_markers': len(neuronal_markers),
    'length_tolerance': LENGTH_TOLERANCE,
    'random_seed': RANDOM_SEED,
    'conclusion': conclusion,
    'scz_neuronal_genes': list(scz_neuronal)
}

summary_path = f"{OUTPUT_DIR}/d45_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {summary_path}", flush=True)

print("\n" + "=" * 70, flush=True)
print("D45 Analysis Complete", flush=True)
print("=" * 70, flush=True)
