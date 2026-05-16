#!/usr/bin/env python3
"""
Batch 009: GWAS-Based Cell-Type Enrichment with PanglaoDB Markers

Experiment: Test whether SCZ GWAS genes are enriched in specific brain cell type marker genes.
Method: Fisher's exact test (hypergeometric) with Benjamini-Hochberg FDR correction.

Parameters (from design.yaml):
- OR threshold (POSITIVE): 5.0
- FDR threshold: 0.05
- BH per-test α: 0.0083 (0.05/6 tests)
- Min markers: 5 per cell type
- Low power threshold: k < 20
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BATCH_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_009")
DATA_DIR = BATCH_DIR / "data"
RESULTS_DIR = BATCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Parameters from design.yaml
OR_THRESHOLD_POSITIVE = 5.0
FDR_THRESHOLD = 0.05
MIN_MARKERS = 5
LOW_POWER_THRESHOLD = 20
BACKGROUND_SIZE = 20000  # protein-coding genes

print("=" * 60)
print("Batch 009: GWAS-Based Cell-Type Enrichment")
print("=" * 60)

# Step 1: Load PanglaoDB markers
print("\n[Step 1] Loading PanglaoDB markers...")
panglao_df = pd.read_csv('/tmp/panglao_markers.tsv.gz', sep='\t')
print(f"  Total markers: {len(panglao_df)}")

# Filter: Brain organ AND human sensitivity > 0
brain_human = panglao_df[
    (panglao_df['organ'] == 'Brain') &
    (panglao_df['sensitivity_human'] > 0)
].copy()
print(f"  Brain + Human sensitivity > 0: {len(brain_human)}")

# Step 2: Create marker sets per cell type
print("\n[Step 2] Creating marker sets per cell type...")

# Group by cell type
cell_type_groups = brain_human.groupby('cell type')['official gene symbol'].apply(list).to_dict()

# Filter to cell types with k >= MIN_MARKERS
qualifying_cell_types = {
    ct: genes for ct, genes in cell_type_groups.items()
    if len(genes) >= MIN_MARKERS
}

print(f"  Cell types with k >= {MIN_MARKERS}: {len(qualifying_cell_types)}")
for ct, genes in sorted(qualifying_cell_types.items(), key=lambda x: -len(x[1])):
    print(f"    {ct}: k = {len(genes)}")

# Create marker sets (unique genes per cell type)
marker_sets = {}
for ct, genes in qualifying_cell_types.items():
    marker_sets[ct] = set(genes)

# Step 3: Load SCZ genes
print("\n[Step 3] Loading SCZ genes...")
scz_genes_path = "/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_008/data/gwas_genes.parquet"
scz_df = pd.read_parquet(scz_genes_path)
scz_genes = set(scz_df['hgnc_symbol'].str.upper().tolist())
print(f"  SCZ genes: {len(scz_genes)}")

# Step 4: Gene symbol harmonization
print("\n[Step 4] Harmonizing gene symbols...")

# Convert marker genes to uppercase for matching
harmonized_marker_sets = {}
for ct, genes in marker_sets.items():
    harmonized = set(str(g).upper() for g in genes)
    harmonized_marker_sets[ct] = harmonized

# Count overlaps
print("  Overlaps between SCZ genes and marker sets:")
for ct in sorted(harmonized_marker_sets.keys()):
    overlap = scz_genes & harmonized_marker_sets[ct]
    print(f"    {ct}: {len(overlap)} overlapping genes")

# Step 5: Run Fisher's exact test
print("\n[Step 5] Running Fisher's exact tests...")

results = []
n_tests = len(harmonized_marker_sets)
bh_alpha = FDR_THRESHOLD / n_tests  # Per-test alpha

print(f"  Number of tests: {n_tests}")
print(f"  BH-corrected alpha per test: {bh_alpha:.4f}")

for cell_type, marker_genes in harmonized_marker_sets.items():
    k = len(marker_genes)  # Marker count
    m = len(scz_genes)     # SCZ gene count
    n = BACKGROUND_SIZE - m  # Non-SCZ genes

    # Overlap count
    overlap = scz_genes & marker_genes
    x = len(overlap)  # Observed overlap

    # Hypergeometric test
    # Population: N = BACKGROUND_SIZE (20,000)
    # Successes in population: M = m (SCZ genes)
    # Sample size: n = k (marker genes)
    # Observed successes: x = overlap
    # Alternative: greater (testing for enrichment)

    # scipy.stats hypergeom: P(X = x)
    # hypergeom.sf(x-1, M, n, N) gives P(X >= x)
    # Using hypergeom for one-sided test

    # Parameters: M = total population, n = successes in population, N = sample size
    # M = BACKGROUND_SIZE, n = m (SCZ genes), N = k (markers)
    p_value = stats.hypergeom.sf(x - 1, BACKGROUND_SIZE, m, k)

    # Calculate odds ratio
    # OR = (x / (k - x)) / ((m - x) / (n - (m - x)))
    # Handle edge cases
    if x > 0 and (k - x) > 0 and (m - x) > 0 and (n - (m - x)) > 0:
        or_value = (x / (k - x)) / ((m - x) / (n - (m - x)))
    else:
        # Use MLE estimate
        or_value = (x + 0.5) / (k - x + 0.5) / ((m - x + 0.5) / (n - (m - x) + 0.5))

    # Determine power status
    if k < LOW_POWER_THRESHOLD:
        power_status = "LOW_POWER"
    else:
        power_status = "ADEQUATE"

    results.append({
        'cell_type': cell_type,
        'k': k,
        'overlap': x,
        'overlap_genes': sorted(list(overlap)),
        'odds_ratio': or_value,
        'p_value': p_value,
        'power_status': power_status
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Step 6: Apply Benjamini-Hochberg FDR correction
print("\n[Step 6] Applying Benjamini-Hochberg FDR correction...")

reject, fdr_values, _, _ = multipletests(
    results_df['p_value'].values,
    alpha=FDR_THRESHOLD,
    method='fdr_bh'
)
results_df['fdr'] = fdr_values
results_df['significant'] = reject

# Step 7: Classify results
print("\n[Step 7] Classifying results...")

def classify_row(row):
    or_val = row['odds_ratio']
    fdr = row['fdr']
    k = row['k']
    power = row['power_status']

    if k < MIN_MARKERS:
        return "EXCLUDED"
    elif or_val > OR_THRESHOLD_POSITIVE and fdr < FDR_THRESHOLD:
        return "POSITIVE"
    elif or_val < 1.5 or fdr > 0.1:
        return "NEGATIVE"
    else:
        return "INCONCLUSIVE"

results_df['classification'] = results_df.apply(classify_row, axis=1)

# Print results
print("\n" + "=" * 60)
print("ENRICHMENT RESULTS")
print("=" * 60)

for _, row in results_df.sort_values('odds_ratio', ascending=False).iterrows():
    print(f"\n{row['cell_type']}")
    print(f"  Markers (k): {row['k']}")
    print(f"  Overlap: {row['overlap']}")
    print(f"  Odds Ratio: {row['odds_ratio']:.3f}")
    print(f"  P-value: {row['p_value']:.4e}")
    print(f"  FDR: {row['fdr']:.4f}")
    print(f"  Power: {row['power_status']}")
    print(f"  Classification: {row['classification']}")
    if row['overlap_genes']:
        print(f"  Overlapping genes: {', '.join(row['overlap_genes'][:10])}")

# Step 8: Save results
print("\n[Step 8] Saving results...")

# Save markers parquet
markers_output = []
for ct, genes in marker_sets.items():
    for gene in genes:
        markers_output.append({
            'cell_type': ct,
            'gene': gene
        })
markers_df = pd.DataFrame(markers_output)
markers_df.to_parquet(DATA_DIR / "markers.parquet", index=False)
print(f"  Saved: {DATA_DIR / 'markers.parquet'}")

# Save enrichment results JSON
enrichment_results = results_df[[
    'cell_type', 'k', 'overlap', 'odds_ratio', 'p_value', 'fdr',
    'power_status', 'classification'
]].to_dict('records')

# Add overlapping genes to JSON
for i, row in results_df.iterrows():
    enrichment_results[i]['overlap_genes'] = row['overlap_genes']

with open(RESULTS_DIR / "enrichment_results.json", 'w') as f:
    json.dump(enrichment_results, f, indent=2)
print(f"  Saved: {RESULTS_DIR / 'enrichment_results.json'}")

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

positive = results_df[results_df['classification'] == 'POSITIVE']
negative = results_df[results_df['classification'] == 'NEGATIVE']
inconclusive = results_df[results_df['classification'] == 'INCONCLUSIVE']
low_power = results_df[results_df['power_status'] == 'LOW_POWER']

print(f"Total cell types tested: {len(results_df)}")
print(f"POSITIVE: {len(positive)}")
print(f"NEGATIVE: {len(negative)}")
print(f"INCONCLUSIVE: {len(inconclusive)}")
print(f"LOW POWER: {len(low_power)}")

# Determine overall classification
print("\n" + "=" * 60)
print("OVERALL CLASSIFICATION")
print("=" * 60)

adequate_power = results_df[results_df['power_status'] == 'ADEQUATE']
positive_adequate = adequate_power[adequate_power['classification'] == 'POSITIVE']
negative_adequate = adequate_power[adequate_power['classification'] == 'NEGATIVE']

if len(positive_adequate) >= 2:
    overall = "ESTABLISHED"
    print(f"Classification: {overall}")
    print(f"Rationale: {len(positive_adequate)} cell types POSITIVE with adequate power (k >= 20)")
elif len(positive_adequate) == 1:
    overall = "SUGGESTED"
    print(f"Classification: {overall}")
    print(f"Rationale: 1 cell type POSITIVE with adequate power")
elif len(negative_adequate) == len(adequate_power) and len(adequate_power) > 0:
    overall = "REFUTED"
    print(f"Classification: {overall}")
    print(f"Rationale: All cell types with adequate power are NEGATIVE")
else:
    overall = "INCONCLUSIVE"
    print(f"Classification: {overall}")
    print(f"Rationale: Only INCONCLUSIVE cell types or all LOW POWER")

# Store for challenge.md
print(f"\nOverall classification: {overall}")

# Save overall classification
with open(RESULTS_DIR / "classification.txt", 'w') as f:
    f.write(overall)

print("\n[Complete]")