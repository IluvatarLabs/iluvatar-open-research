#!/usr/bin/env python3
"""
Batch 010: Microglia GWAS Enrichment Test (All 80 Markers)

Experiment: Test whether SCZ GWAS genes are enriched in ALL microglia marker genes
           from PanglaoDB (including sensitivity_human = 0).

Design Specification:
- GWAS genes: 444 SCZ genes (from batch_009)
- Background: 20,000 protein-coding genes (pybiomart)
- Markers: 80 microglia markers from PanglaoDB (ALL, not filtered by sensitivity_human)
- Method: Fisher's exact test
- Alpha: 0.05 (single test)

Key Rationale:
- Batch 009 excluded microglia by filtering sensitivity_human > 0
- All 80 PanglaoDB microglia markers have sensitivity_human = 0
- This is the critical cell type implicated by S-LDSC literature
- Must test regardless of human sensitivity filter

Implementation Steps:
1. Load SCZ genes from batch_008 data
2. Load ALL 80 PanglaoDB microglia markers (including sensitivity_human=0)
3. Map gene symbols to HGNC (if needed)
4. Run Fisher's exact test
5. Calculate OR and p-value
6. Report overlapping genes
7. Gene length analysis: median gene length for microglia markers vs background
8. Species breakdown: how many are human/mouse/both
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BATCH_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_010")
DATA_DIR = BATCH_DIR / "data"
RESULTS_DIR = BATCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Parameters from design specification
BACKGROUND_SIZE = 20000  # protein-coding genes
ALPHA = 0.05  # Single test

print("=" * 70)
print("Batch 010: Microglia GWAS Enrichment Test (All 80 Markers)")
print("=" * 70)

# =============================================================================
# Step 1: Load SCZ genes from batch_008 data
# =============================================================================
print("\n[Step 1] Loading SCZ GWAS genes...")
scz_genes_path = "/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_008/data/gwas_genes.parquet"
scz_df = pd.read_parquet(scz_genes_path)
scz_genes = set(scz_df['hgnc_symbol'].str.upper().tolist())
print(f"  SCZ genes loaded: {len(scz_genes)}")

# Also store gene lengths for analysis
scz_gene_lengths = {}
for _, row in scz_df.iterrows():
    gene = str(row['hgnc_symbol']).upper()
    if pd.notna(row['gene_start']) and pd.notna(row['gene_end']):
        length = row['gene_end'] - row['gene_start']
        scz_gene_lengths[gene] = length

print(f"  SCZ genes with length data: {len(scz_gene_lengths)}")

# =============================================================================
# Step 2: Load ALL 80 PanglaoDB microglia markers (including sensitivity_human=0)
# =============================================================================
print("\n[Step 2] Loading PanglaoDB microglia markers...")
panglao_df = pd.read_csv('/tmp/panglao_markers.tsv.gz', sep='\t')

# Filter to Microglia cell type - NO FILTERING by sensitivity_human
microglia_df = panglao_df[panglao_df['cell type'] == 'Microglia'].copy()
print(f"  Total microglia markers (all): {len(microglia_df)}")

# Create marker set (unique genes)
microglia_genes = microglia_df['official gene symbol'].str.upper().unique().tolist()
microglia_marker_set = set(microglia_genes)
print(f"  Unique gene symbols: {len(microglia_marker_set)}")

# =============================================================================
# Step 3: Species breakdown analysis
# =============================================================================
print("\n[Step 3] Analyzing species breakdown...")

# Get species information (handle 'Mm Hs' = Mouse+Human)
# Use set operations on unique genes per species
human_genes_set = set(microglia_df[microglia_df['species'].str.contains('Hs', na=False)]['official gene symbol'].str.upper().unique())
mouse_genes_set = set(microglia_df[microglia_df['species'].str.contains('Mm', na=False)]['official gene symbol'].str.upper().unique())
both_species = human_genes_set & mouse_genes_set
human_only = human_genes_set - mouse_genes_set
mouse_only = mouse_genes_set - human_genes_set

species_breakdown = {
    'human': len(human_genes_set),
    'mouse': len(mouse_genes_set),
    'both': len(both_species),
    'human_only': len(human_only),
    'mouse_only': len(mouse_only),
    'human_genes': sorted(human_genes_set),
    'mouse_genes': sorted(mouse_genes_set),
    'both_genes': sorted(both_species)
}

print(f"  Human markers: {len(human_genes_set)}")
print(f"  Mouse markers: {len(mouse_genes_set)}")
print(f"  Both species: {len(both_species)}")
print(f"  Human only: {len(human_only)}")
print(f"  Mouse only: {len(mouse_only)}")

# =============================================================================
# Step 4: Gene symbol harmonization
# =============================================================================
print("\n[Step 4] Harmonizing gene symbols...")

# Convert microglia genes to uppercase for matching
harmonized_markers = set(str(g).upper() for g in microglia_marker_set)

# =============================================================================
# Step 5: Run Fisher's exact test
# =============================================================================
print("\n[Step 5] Running Fisher's exact test...")

k = len(harmonized_markers)  # Marker count
m = len(scz_genes)           # SCZ gene count
n = BACKGROUND_SIZE - m     # Non-SCZ genes

# Count overlap
overlap = scz_genes & harmonized_markers
x = len(overlap)  # Observed overlap

print(f"  Background: {BACKGROUND_SIZE} protein-coding genes")
print(f"  SCZ genes (M): {m}")
print(f"  Microglia markers (N): {k}")
print(f"  Overlap (x): {x}")

# Fisher's exact test (hypergeometric)
# Parameters: M = total population, n = successes in population, N = sample size
# M = BACKGROUND_SIZE, n = m (SCZ genes), N = k (markers)
p_value = stats.hypergeom.sf(x - 1, BACKGROUND_SIZE, m, k)

# Calculate odds ratio
# OR = (x / (k - x)) / ((m - x) / (n - (m - x)))
if x > 0 and (k - x) > 0 and (m - x) > 0 and (n - (m - x)) > 0:
    or_value = (x / (k - x)) / ((m - x) / (n - (m - x)))
else:
    # Use MLE estimate with continuity correction
    or_value = (x + 0.5) / (k - x + 0.5) / ((m - x + 0.5) / (n - (m - x) + 0.5))

print(f"  Odds Ratio: {or_value:.3f}")
print(f"  P-value: {p_value:.4e}")

# Determine power status
LOW_POWER_THRESHOLD = 20
if k < LOW_POWER_THRESHOLD:
    power_status = "LOW_POWER"
else:
    power_status = "ADEQUATE"
print(f"  Power status: {power_status}")

# =============================================================================
# Step 6: Report overlapping genes
# =============================================================================
print("\n[Step 6] Overlapping genes...")

overlap_genes_list = sorted(list(overlap))
print(f"  {len(overlap_genes_list)} overlapping genes:")
for gene in overlap_genes_list:
    print(f"    - {gene}")

# =============================================================================
# Step 7: Gene length analysis
# =============================================================================
print("\n[Step 7] Gene length analysis...")

# Get proper background gene lengths from pybiomart
print("  Fetching background gene lengths from pybiomart...")
from pybiomart import Server

server = Server(host='http://www.ensembl.org')
dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']

background_genes = dataset.query(
    attributes=['hgnc_symbol', 'ensembl_gene_id', 'start_position', 'end_position'],
    filters={'biotype': 'protein_coding'}
)

# Calculate lengths and clean
background_genes = background_genes.dropna(subset=['Gene start (bp)', 'Gene end (bp)'])
background_genes['gene_length'] = background_genes['Gene end (bp)'] - background_genes['Gene start (bp)']

# Get unique genes (take first occurrence per gene symbol)
background_lengths_df = background_genes.groupby('HGNC symbol').first().reset_index()
background_lengths = background_lengths_df['gene_length'].values

background_median = np.median(background_lengths)
background_mean = np.mean(background_lengths)
background_count = len(background_lengths)

print(f"  Background loaded: {background_count} protein-coding genes")
print(f"  Background median: {background_median:.0f} bp")
print(f"  Background mean: {background_mean:.0f} bp")

# Get microglia marker lengths from pybiomart
# Map microglia genes to HGNC symbols
microglia_markers_upper = [g.upper() for g in harmonized_markers]

microglia_lengths_df = background_lengths_df[
    background_lengths_df['HGNC symbol'].isin(microglia_markers_upper)
]
all_microglia_lengths = microglia_lengths_df['gene_length'].values

if len(all_microglia_lengths) > 0:
    all_microglia_median = np.median(all_microglia_lengths)
    all_microglia_mean = np.mean(all_microglia_lengths)
else:
    all_microglia_median = None
    all_microglia_mean = None

print(f"  Microglia markers with length data: {len(all_microglia_lengths)}/{len(harmonized_markers)}")
print(f"  Microglia median: {all_microglia_median:.0f} bp" if all_microglia_median else "  Microglia median: N/A")
print(f"  Microglia mean: {all_microglia_mean:.0f} bp" if all_microglia_mean else "  Microglia mean: N/A")

# Overlap genes with length data
overlap_markers_upper = [g.upper() for g in overlap_genes_list]
overlap_lengths_df = background_lengths_df[
    background_lengths_df['HGNC symbol'].isin(overlap_markers_upper)
]
microglia_overlap_lengths = overlap_lengths_df['gene_length'].values

if len(microglia_overlap_lengths) > 0:
    microglia_overlap_median = np.median(microglia_overlap_lengths)
    microglia_overlap_mean = np.mean(microglia_overlap_lengths)
else:
    microglia_overlap_median = None
    microglia_overlap_mean = None

print(f"  Overlap genes with length data: {len(microglia_overlap_lengths)}/{len(overlap_genes_list)}")
print(f"  Overlap median: {microglia_overlap_median:.0f} bp" if microglia_overlap_median else "  Overlap median: N/A")
print(f"  Overlap mean: {microglia_overlap_mean:.0f} bp" if microglia_overlap_mean else "  Overlap mean: N/A")

gene_length_analysis = {
    'background': {
        'median_bp': float(background_median),
        'mean_bp': float(background_mean),
        'count': int(background_count)
    },
    'microglia_markers': {
        'median_bp': float(all_microglia_median) if all_microglia_median else None,
        'mean_bp': float(all_microglia_mean) if all_microglia_mean else None,
        'count_with_length_data': int(len(all_microglia_lengths)),
        'total_markers': len(harmonized_markers)
    },
    'overlap_genes': {
        'median_bp': float(microglia_overlap_median) if microglia_overlap_median else None,
        'mean_bp': float(microglia_overlap_mean) if microglia_overlap_mean else None,
        'count': int(len(microglia_overlap_lengths))
    }
}

# Check for gene length bias
if all_microglia_median and background_median:
    if all_microglia_median > background_median * 1.5:
        gene_length_warning = "Microglia markers may have longer genes than background (potential bias)"
    elif all_microglia_median < background_median * 0.5:
        gene_length_warning = "Microglia markers have shorter genes than background (bias in opposite direction)"
    else:
        gene_length_warning = None

    if gene_length_warning:
        print(f"\n  WARNING: {gene_length_warning}")

# =============================================================================
# Step 8: Classification
# =============================================================================
print("\n[Step 8] Classification...")

# Single test alpha = 0.05
OR_THRESHOLD = 5.0

if p_value < ALPHA and or_value > OR_THRESHOLD:
    classification = "POSITIVE"
elif p_value < ALPHA and or_value > 1.0:
    classification = "MARGINAL"
elif p_value > ALPHA:
    classification = "NEGATIVE"
else:
    classification = "INCONCLUSIVE"

print(f"  Classification: {classification}")
print(f"  Significance: p-value {'<' if p_value < ALPHA else '>='} {ALPHA}")
print(f"  Effect size: OR = {or_value:.3f} ({'>' if or_value > OR_THRESHOLD else '<='} threshold of {OR_THRESHOLD})")

# =============================================================================
# Step 9: Compile and save results
# =============================================================================
print("\n[Step 9] Saving results...")

results = {
    'experiment': 'batch_010_microglia_enrichment',
    'cell_type': 'Microglia',
    'method': 'Fisher exact test (hypergeometric)',
    'alpha': ALPHA,
    'parameters': {
        'background_size': BACKGROUND_SIZE,
        'scz_genes': m,
        'marker_count': k,
        'low_power_threshold': LOW_POWER_THRESHOLD
    },
    'test_results': {
        'overlap': x,
        'odds_ratio': float(or_value),
        'p_value': float(p_value),
        'power_status': power_status
    },
    'classification': classification,
    'overlapping_genes': overlap_genes_list,
    'gene_length_analysis': gene_length_analysis,
    'species_breakdown': species_breakdown,
    'all_microglia_genes': sorted(list(harmonized_markers))
}

# Save JSON results
results_path = RESULTS_DIR / "results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {results_path}")

# =============================================================================
# Step 10: Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nCell type: Microglia")
print(f"Markers tested: {k} (ALL from PanglaoDB, including sensitivity_human = 0)")
print(f"Overlap with SCZ genes: {x}")
print(f"Odds Ratio: {or_value:.3f}")
print(f"P-value: {p_value:.4e}")
print(f"Classification: {classification}")

print(f"\nSpecies breakdown:")
print(f"  Human markers: {species_breakdown['human']}")
print(f"  Mouse markers: {species_breakdown['mouse']}")
print(f"  Both: {species_breakdown['both']}")

print(f"\nGene length comparison:")
print(f"  Background median: {background_median:.0f} bp ({background_count} genes)")
print(f"  Microglia markers median: {all_microglia_median:.0f} bp ({len(all_microglia_lengths)} genes with data)" if all_microglia_median else "  Microglia markers median: N/A")

if overlap_genes_list:
    print(f"\nOverlapping genes ({len(overlap_genes_list)}):")
    print(f"  {', '.join(overlap_genes_list)}")

print("\n[Complete]")
