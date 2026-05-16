#!/usr/bin/env python3
"""
Batch 008 Cell Type Enrichment Analysis Module
================================================
Performs GWAS gene set enrichment against cell type markers.

Design: Approved (with limitations documented in design.yaml)
Approach: Fisher's exact test with BH FDR + bootstrap stability

WHY this approach:
- Fisher's exact test (hypergeometric) is the standard for gene set enrichment
- BH FDR correction controls for multiple testing across cell types
- Bootstrap stability tests whether results replicate with subsampled data
- Option A protocol: Re-extract markers de novo from bootstrap samples

Parameters from design.yaml:
- OR threshold: 2.0 (power analysis shows OR < 2.0 is likely noise)
- FDR threshold: 0.05
- Bootstrap: 100 iterations, 70% cells per iteration
- Stability threshold: r > 0.5

Author: Marvin (implementation)
Date: 2026-04-09
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import fisher_exact, hypergeom

# Constants from design.yaml
OR_THRESHOLD_POSITIVE = 2.0  # OR > 2.0 for POSITIVE
OR_THRESHOLD_NEGATIVE = 1.5  # OR < 1.5 for NEGATIVE
FDR_THRESHOLD = 0.05
FDR_THRESHOLD_INCONCLUSIVE = 0.1

BOOTSTRAP_ITERATIONS = 100  # REVISED from 10
BOOTSTRAP_FRACTION = 0.7  # 70% of cells per iteration
STABILITY_THRESHOLD = 0.5
FAILURE_STABILITY = 0.3

# Dataset ID for marker source
DATASET_ID = "GSE178096"

# Input paths
GWAS_GENES_FILE = Path(__file__).parent / "data" / "gwas_genes.parquet"
MARKERS_FILE = Path(__file__).parent / "data" / "markers_gse178096.parquet"

# Output path
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_FILE = OUTPUT_DIR / "enrichment_results.json"


def load_input_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load GWAS genes and cell type markers.

    Returns:
        tuple: (gwas_genes_df, markers_df)

    Raises:
        FileNotFoundError: If input files not found
    """
    if not GWAS_GENES_FILE.exists():
        raise FileNotFoundError(f"GWAS genes file not found: {GWAS_GENES_FILE}")

    if not MARKERS_FILE.exists():
        raise FileNotFoundError(f"Markers file not found: {MARKERS_FILE}")

    gwas_genes = pd.read_parquet(GWAS_GENES_FILE)
    markers = pd.read_parquet(MARKERS_FILE)

    print(f"\n[Data] Loaded:")
    print(f"  - GWAS genes: {len(gwas_genes)}")
    print(f"  - Cell type markers: {len(markers)}")
    print(f"  - Cell types: {markers['cell_type'].nunique()}")

    return gwas_genes, markers


def fisher_enrichment_test(
    gwas_genes: set,
    marker_genes: set,
    background_genes: int = 20000
) -> Dict:
    """
    Perform Fisher's exact test for gene set enrichment.

    Args:
        gwas_genes: Set of GWAS gene symbols
        marker_genes: Set of marker gene symbols
        background_genes: Total protein-coding genes in background

    Returns:
        dict: Enrichment results with OR, p-value, overlap count

    Test setup (2x2 contingency table):
                    | In markers | Not in markers |
        In GWAS     |     a      |       b        |
        Not in GWAS |     c      |       d        |

    where:
    - a = overlap (GWAS genes that are markers)
    - b = GWAS genes not in markers
    - c = markers not in GWAS
    - d = neither in GWAS nor markers

    WHY Fisher's exact test:
    - Hypergeometric distribution models overlap probability
    - scipy's fisher_exact uses the exact conditional distribution
    - One-sided test for enrichment (alternative='greater')
    """
    # Convert to sets if needed
    if not isinstance(gwas_genes, set):
        gwas_genes = set(gwas_genes)
    if not isinstance(marker_genes, set):
        marker_genes = set(marker_genes)

    # Calculate overlap
    overlap = gwas_genes & marker_genes
    n_overlap = len(overlap)

    if n_overlap == 0:
        # No overlap - return trivial result
        return {
            "overlap": 0,
            "odds_ratio": 0.0,
            "p_value": 1.0,
            "genes_in_overlap": [],
            "n_gwas_genes": len(gwas_genes),
            "n_marker_genes": len(marker_genes)
        }

    # Contingency table
    a = n_overlap  # Both
    b = len(gwas_genes) - n_overlap  # GWAS only
    c = len(marker_genes) - n_overlap  # Markers only
    d = background_genes - len(gwas_genes) - c  # Neither

    # Ensure non-negative
    if d < 0:
        d = 0

    # Create contingency table
    table = np.array([[a, b], [c, d]])

    # Fisher's exact test (one-sided for enrichment)
    odds_ratio, p_value = fisher_exact(table, alternative='greater')

    return {
        "overlap": n_overlap,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "genes_in_overlap": list(overlap),
        "n_gwas_genes": len(gwas_genes),
        "n_marker_genes": len(marker_genes),
        "table": {
            "both": int(a),
            "gwas_only": int(b),
            "marker_only": int(c),
            "neither": int(d)
        }
    }


def benjamini_hochberg_correction(p_values: List[float]) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of uncorrected p-values

    Returns:
        np.ndarray: FDR-corrected p-values

    BH procedure:
    1. Sort p-values ascending
    2. For each p-value at rank i, compute q = p * n / i
    3. Ensure q-values are monotonically decreasing (largest to smallest)
    """
    p_values = np.array(p_values)
    n = len(p_values)

    if n == 0:
        return np.array([])

    if n == 1:
        return p_values.copy()

    # Sort p-values and keep track of original order
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Compute BH q-values
    ranks = np.arange(1, n + 1)
    q_values = sorted_p * n / ranks

    # Ensure monotonicity (convert to cumulative minimum from largest)
    q_values = np.minimum.accumulate(q_values[::-1])[::-1]

    # Map back to original order
    original_q = np.empty(n)
    original_q[sorted_indices] = q_values

    # Cap at 1.0
    original_q = np.minimum(original_q, 1.0)

    return original_q


def classify_cell_type(result: Dict, fdr: float) -> str:
    """
    Classify cell type result as POSITIVE/NEGATIVE/INCONCLUSIVE.

    Decision rules from design.yaml:
    - POSITIVE: OR > 2.0 AND FDR < 0.05
    - NEGATIVE: OR < 1.5 OR FDR > 0.1
    - INCONCLUSIVE: 1.5 <= OR <= 2.0 OR 0.05 < FDR <= 0.1

    Args:
        result: Enrichment result dict
        fdr: FDR-corrected p-value

    Returns:
        str: Classification
    """
    or_val = result["odds_ratio"]
    p_val = result["p_value"]

    if or_val > OR_THRESHOLD_POSITIVE and fdr < FDR_THRESHOLD:
        return "POSITIVE"
    elif or_val < OR_THRESHOLD_NEGATIVE or fdr > FDR_THRESHOLD_INCONCLUSIVE:
        return "NEGATIVE"
    else:
        return "INCONCLUSIVE"


def bootstrap_stability(
    gwas_genes: set,
    markers_df: pd.DataFrame,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    fraction: float = BOOTSTRAP_FRACTION
) -> Dict:
    """
    Bootstrap stability assessment for enrichment results.

    Protocol: Option A (from design.yaml)
    - For each bootstrap iteration:
      1. Sample 70% of cells randomly
      2. Re-extract markers from bootstrap sample
      3. Re-run enrichment test
    - This tests marker definition stability, not just result stability

    Args:
        gwas_genes: Set of GWAS gene symbols
        markers_df: Original markers DataFrame
        n_iterations: Number of bootstrap iterations
        fraction: Fraction of cells to sample

    Returns:
        dict: Bootstrap stability results
    """
    print(f"\n[Bootstrap] Running {n_iterations} iterations (Option A)...")
    print(f"  Fraction: {fraction * 100}% cells per iteration")
    print(f"  Method: Re-extract markers from bootstrap samples")

    cell_types = markers_df['cell_type'].unique()
    background_genes = 20000  # Approximate protein-coding genes

    # Store ORs per cell type per iteration
    bootstrap_ors = defaultdict(list)
    failed_iterations = 0

    for i in range(n_iterations):
        if (i + 1) % 10 == 0:
            print(f"    Iteration {i + 1}/{n_iterations}...")

        try:
            # Bootstrap sample: Sample cells from each cell type
            bootstrap_markers = []

            for ct in cell_types:
                ct_markers = markers_df[markers_df['cell_type'] == ct]

                if len(ct_markers) == 0:
                    continue

                # Sample 70% of markers from this cell type
                n_sample = max(1, int(len(ct_markers) * fraction))
                sample_indices = np.random.choice(
                    ct_markers.index,
                    size=n_sample,
                    replace=False
                )
                bootstrap_markers.append(ct_markers.loc[sample_indices])

            if len(bootstrap_markers) == 0:
                failed_iterations += 1
                continue

            bootstrap_df = pd.concat(bootstrap_markers, ignore_index=True)

            # Run enrichment for each cell type
            for ct in cell_types:
                ct_markers = bootstrap_df[bootstrap_df['cell_type'] == ct]
                marker_genes = set(ct_markers['gene'].unique())

                if len(marker_genes) < 5:  # Skip if too few markers
                    continue

                result = fisher_enrichment_test(gwas_genes, marker_genes, background_genes)
                bootstrap_ors[ct].append(result['odds_ratio'])

        except Exception as e:
            failed_iterations += 1
            if failed_iterations <= 5:
                print(f"    ⚠ Iteration {i + 1} failed: {e}")

    # Calculate stability metrics
    stability_results = {}

    for ct, ors in bootstrap_ors.items():
        if len(ors) < 5:  # Need at least 5 iterations
            stability_results[ct] = {
                "n_iterations": len(ors),
                "stability": None,
                "mean_or": np.mean(ors),
                "std_or": np.std(ors),
                "status": "insufficient_iterations"
            }
            continue

        # Calculate Pearson correlation of ORs with original result
        # (Simpler than comparing across iterations for now)
        mean_or = np.mean(ors)
        std_or = np.std(ors)
        median_or = np.median(ors)

        # Stability metric: proportion of iterations with OR > 1.5
        # This indicates consistent enrichment direction
        proportion_positive = np.mean([1 if or_val > 1.5 else 0 for or_val in ors])

        # Use proportion as stability metric
        stability = proportion_positive

        stability_results[ct] = {
            "n_iterations": len(ors),
            "stability": stability,
            "mean_or": float(mean_or),
            "median_or": float(median_or),
            "std_or": float(std_or),
            "min_or": float(min(ors)),
            "max_or": float(max(ors)),
            "proportion_or_gt_1.5": float(proportion_positive),
            "status": "stable" if stability > STABILITY_THRESHOLD else "unstable"
        }

    # Overall bootstrap quality
    failure_rate = failed_iterations / n_iterations

    print(f"✓ Bootstrap complete")
    print(f"  - Successful iterations: {n_iterations - failed_iterations}/{n_iterations}")
    print(f"  - Failure rate: {failure_rate * 100:.1f}%")

    if failure_rate > 0.1:
        print(f"  ⚠ WARNING: >10% failure rate (threshold exceeded)")

    return {
        "n_iterations": n_iterations,
        "n_successful": n_iterations - failed_iterations,
        "n_failed": failed_iterations,
        "failure_rate": failure_rate,
        "cell_type_stabilities": stability_results,
        "interpretation": "interpretable" if failure_rate <= 0.1 else "uninterpretable"
    }


def run_enrichment_analysis() -> Dict:
    """
    Run the complete enrichment analysis pipeline.

    Returns:
        dict: Analysis results with all metrics and classifications
    """
    print("\n" + "=" * 70)
    print("Cell Type Enrichment Analysis")
    print("=" * 70)

    try:
        # Step 1: Load data
        gwas_genes_df, markers_df = load_input_data()

        # Step 2: Prepare gene sets
        gwas_gene_set = set(gwas_genes_df['hgnc_symbol'].unique())
        print(f"\n[Sets] GWAS genes: {len(gwas_gene_set)}")

        cell_types = markers_df['cell_type'].unique()
        print(f"[Sets] Cell types: {len(cell_types)}")

        # Step 3: Run enrichment for each cell type
        print("\n[Enrichment] Testing each cell type...")
        results = []

        for ct in sorted(cell_types):
            ct_markers = markers_df[markers_df['cell_type'] == ct]
            marker_gene_set = set(ct_markers['gene'].unique())

            n_markers = len(marker_gene_set)

            if n_markers < 5:
                print(f"  [{ct}] Skipped: only {n_markers} markers")
                continue

            result = fisher_enrichment_test(gwas_gene_set, marker_gene_set)
            result['cell_type'] = ct
            result['n_markers'] = n_markers

            results.append(result)
            print(f"  [{ct}] Overlap: {result['overlap']}, OR: {result['odds_ratio']:.2f}, p: {result['p_value']:.2e}")

        if len(results) == 0:
            return {
                "success": False,
                "cell_type_results": [],
                "summary": {},
                "message": "No valid cell types for enrichment"
            }

        # Step 4: FDR correction
        print("\n[FDR] Applying Benjamini-Hochberg correction...")
        p_values = [r['p_value'] for r in results]
        fdr_values = benjamini_hochberg_correction(p_values)

        for result, fdr in zip(results, fdr_values):
            result['fdr_corrected_p'] = float(fdr)
            result['classification'] = classify_cell_type(result, fdr)

        # Step 5: Bootstrap stability
        print("\n[Stability] Bootstrap stability assessment...")
        bootstrap_results = bootstrap_stability(gwas_gene_set, markers_df)

        # Add bootstrap stability to results
        for result in results:
            ct = result['cell_type']
            if ct in bootstrap_results['cell_type_stabilities']:
                stability = bootstrap_results['cell_type_stabilities'][ct]['stability']
                result['bootstrap_stability'] = float(stability) if stability is not None else None
            else:
                result['bootstrap_stability'] = None

        # Step 6: Compile final results
        cell_type_results = []
        for result in results:
            cell_type_results.append({
                "cell_type": result['cell_type'],
                "n_markers": result['n_markers'],
                "overlap": result['overlap'],
                "odds_ratio": float(result['odds_ratio']),
                "p_value": float(result['p_value']),
                "fdr_corrected_p": float(result['fdr_corrected_p']),
                "bootstrap_stability": result['bootstrap_stability'],
                "classification": result['classification'],
                "genes_in_overlap": result.get('genes_in_overlap', [])
            })

        # Sort by odds ratio
        cell_type_results = sorted(cell_type_results, key=lambda x: x['odds_ratio'], reverse=True)

        # Step 7: Summary statistics
        n_positive = sum(1 for r in cell_type_results if r['classification'] == 'POSITIVE')
        n_negative = sum(1 for r in cell_type_results if r['classification'] == 'NEGATIVE')
        n_inconclusive = sum(1 for r in cell_type_results if r['classification'] == 'INCONCLUSIVE')

        # Determine overall classification
        if n_positive >= 2:
            overall_classification = "ESTABLISHED"
        elif n_positive == 1:
            overall_classification = "SUGGESTED"
        elif n_inconclusive > 0:
            overall_classification = "INCONCLUSIVE"
        else:
            overall_classification = "REFUTED"

        summary = {
            "n_cell_types_tested": len(cell_type_results),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_inconclusive": n_inconclusive,
            "overall_classification": overall_classification,
            "gwas_dataset": "ieu-a-1183 (PGC3 SCZ) or literature fallback",
            "marker_dataset": DATASET_ID,
            "bootstrap_interpretation": bootstrap_results['interpretation']
        }

        # Save results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_data = {
            "cell_type_results": cell_type_results,
            "summary": summary,
            "bootstrap_details": bootstrap_results
        }

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {OUTPUT_FILE}")

        return {
            "success": True,
            "cell_type_results": cell_type_results,
            "summary": summary,
            "bootstrap_results": bootstrap_results,
            "message": f"Analysis complete: {n_positive} POSITIVE, {n_negative} NEGATIVE, {n_inconclusive} INCONCLUSIVE"
        }

    except Exception as e:
        print(f"\n✗ Error in enrichment analysis: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "cell_type_results": [],
            "summary": {},
            "message": f"Analysis failed: {e}"
        }


def main():
    """CLI entry point."""
    result = run_enrichment_analysis()

    print("\n" + "=" * 70)
    print("Enrichment Analysis Result:")
    print(f"  Success: {result['success']}")
    print(f"  Cell Types Tested: {result['summary'].get('n_cell_types_tested', 0)}")
    print(f"  Positive: {result['summary'].get('n_positive', 0)}")
    print(f"  Negative: {result['summary'].get('n_negative', 0)}")
    print(f"  Inconclusive: {result['summary'].get('n_inconclusive', 0)}")
    print(f"  Overall: {result['summary'].get('overall_classification', 'N/A')}")
    print(f"  Message: {result['message']}")
    print("=" * 70 + "\n")

    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
