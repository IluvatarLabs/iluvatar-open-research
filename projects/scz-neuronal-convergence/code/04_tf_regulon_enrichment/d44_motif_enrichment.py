#!/usr/bin/env python3
"""
D44: TF Motif Enrichment Analysis for SCZ Convergence Project

Hypothesis: SCZ GWAS genes are enriched for specific TF motif occurrences
in their promoters compared to background genes.

Design (from brief):
- 12 primary TFs: EGR1, CTCF, RELA, NFKB1, SPI1, CEBPA, CEBPB, CREB1, NR3C1, STAT1, STAT3, TCF4, MEF2C
- 4 secondary TFs: JUN, FOS, ELF1, ETS1
- Threshold: median TF rank ≤ 500 (top 1.85% of promoters)
- Test: Fisher's exact on 2x2 contingency table
- Multiple testing: BH-FDR across 16 TFs
- Permutations: 1,000 for top 3 TFs

Sources:
- Fisher's exact: Fisher (1925) "Statistical Methods for Research Workers"
- BH-FDR: Benjamini & Hochberg (1995) "Controlling the False Discovery Rate"
- Motif atlas: MC9nr (Molecular signatures, 2019)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests  # For Benjamini-Hochberg FDR

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Input files
GWAS_GENES_PATH = "experiments/batch_008/data/gwas_genes.parquet"
MOTIF_ATLAS_PATH = "data/hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather"

# Output directory
OUTPUT_DIR = Path("experiments/batch_040/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
# Threshold: median rank ≤ 500 = top 1.85% (500/27090 ≈ 0.0184)
# Source: design brief specifies this threshold
RANK_THRESHOLD = 500
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

# Target TFs
# Primary TFs: from design brief
PRIMARY_TFS = [
    "EGR1", "CTCF", "RELA", "NFKB1", "SPI1",
    "CEBPA", "CEBPB", "CREB1", "NR3C1",
    "STAT1", "STAT3", "TCF4", "MEF2C"
]

# Secondary TFs: from design brief
SECONDARY_TFS = ["JUN", "FOS", "ELF1", "ETS1"]

ALL_TFS = PRIMARY_TFS + SECONDARY_TFS

# ============================================================================
# Helper Functions
# ============================================================================

def load_data():
    """Load gwas_genes.parquet and motif atlas feather file."""
    print("=" * 70)
    print("D44: TF Motif Enrichment Analysis")
    print("=" * 70)

    # Load GWAS genes
    print("\n[1/6] Loading GWAS genes...")
    gwas_genes = pd.read_parquet(GWAS_GENES_PATH)
    print(f"  - Loaded {len(gwas_genes)} GWAS genes")
    print(f"  - Columns: {list(gwas_genes.columns)}")

    # Load motif atlas
    print("\n[2/6] Loading motif atlas (this may take a moment)...")
    motif_atlas = pd.read_feather(MOTIF_ATLAS_PATH)

    # Motif atlas structure: rows=motifs, columns=genes, with 'motifs' column containing motif names
    # Transpose to get genes as index, motifs as columns
    print(f"  - Raw atlas shape: {motif_atlas.shape}")
    print(f"  - Columns (first 10): {list(motif_atlas.columns[:10])}")

    # Set index to motif names (first column)
    motif_atlas = motif_atlas.set_index('motifs')
    print(f"  - Set motifs as index, shape now: {motif_atlas.shape}")

    # Transpose: genes become index, motifs become columns
    motif_atlas = motif_atlas.T
    motif_atlas.index.name = 'gene'
    motif_atlas.columns.name = 'motif'

    print(f"  - Transposed to genes×motifs: {motif_atlas.shape}")
    print(f"  - Gene index name: {motif_atlas.index.name}")
    print(f"  - Sample genes: {list(motif_atlas.index[:5])}")

    return gwas_genes, motif_atlas


def find_tf_motifs(motif_atlas, tf_name):
    """
    Find all motifs matching a TF name (case-insensitive string contains).

    Motif naming convention: typically contains TF name (e.g., "EGR1_01", "EGR1_02")
    Some motifs may have variants or partial matches.
    """
    tf_upper = tf_name.upper()
    motif_cols = motif_atlas.columns.tolist()

    # Find all motifs containing the TF name
    matching_motifs = [
        col for col in motif_cols
        if tf_upper in col.upper()
    ]

    return matching_motifs


def compute_median_ranks(motif_atlas, tf_motifs, genes):
    """
    Compute median rank per gene across all motifs for a TF.

    Parameters:
    - motif_atlas: DataFrame with genes as index, motifs as columns
    - tf_motifs: list of motif column names for this TF
    - genes: list of gene names to compute ranks for

    Returns:
    - Dictionary mapping gene_name -> median rank (lower = stronger binding)
    """
    # Subset to the TF's motifs
    tf_data = motif_atlas.loc[genes, tf_motifs]

    # Compute median across motifs for each gene
    median_ranks = tf_data.median(axis=1)

    return median_ranks.to_dict()


def run_fisher_test(k_scz, n_scz, k_bg, n_bg):
    """
    Run Fisher's exact test on 2x2 contingency table.

    Table structure:
                  | High affinity | Low affinity |
    SCZ genes      |     k_scz    |   n_scz - k  |
    Background     |     k_bg     |   n_bg - k   |

    Source: Fisher (1925) "Statistical Methods for Research Workers"
    Implementation: scipy.stats.fisher_exact or statsmodels equivalent
    """
    # Build contingency table
    # [[SCZ high, SCZ low], [BG high, BG low]]
    table = np.array([
        [k_scz, n_scz - k_scz],
        [k_bg, n_bg - k_bg]
    ])

    # Use scipy.stats.fisher_exact for odds ratio and p-value
    # Source: Fisher (1925) "Statistical Methods for Research Workers"
    # scipy.stats.fisher_exact returns (odds_ratio, p_value)
    odds_ratio, p_value = stats.fisher_exact(table)

    # Compute confidence interval manually using log OR
    # For Fisher's exact, CI is computed via exact method
    # We use a mid-p correction approximation
    try:
        # Compute log OR and its variance
        log_or = np.log(odds_ratio)

        # Variance of log OR (Cochran-Mantel-Haenszel style)
        # var(log OR) = 1/a + 1/b + 1/c + 1/d
        a, b = k_scz, n_scz - k_scz
        c, d = k_bg, n_bg - k_bg

        if all([a > 0, b > 0, c > 0, d > 0]):
            var_log_or = 1/a + 1/b + 1/c + 1/d
            se_log_or = np.sqrt(var_log_or)

            # 95% CI (Woolf method)
            ci_low = np.exp(log_or - 1.96 * se_log_or)
            ci_high = np.exp(log_or + 1.96 * se_log_or)
        else:
            # Fallback: use scipy's Fisher result if available
            ci_low = np.nan
            ci_high = np.nan
    except:
        ci_low = np.nan
        ci_high = np.nan

    return odds_ratio, ci_low, ci_high, p_value, table


def run_permutations(motif_atlas, tf_motifs, scz_genes, background_genes,
                     n_perms=RANDOM_SEED, rank_threshold=RANK_THRESHOLD):
    """
    Run permutation test for a TF.

    Shuffle gene labels 1,000 times to compute null distribution of OR.
    Source: design brief specifies 1,000 permutations for top 3 TFs.
    """
    np.random.seed(n_perms)

    # Compute median ranks for all genes
    all_genes = list(scz_genes) + list(background_genes)
    median_ranks = compute_median_ranks(motif_atlas, tf_motifs, all_genes)

    # Count high-affinity genes in original SCZ vs background
    k_scz_orig = sum(1 for g in scz_genes if median_ranks.get(g, float('inf')) <= rank_threshold)
    k_bg_orig = sum(1 for g in background_genes if median_ranks.get(g, float('inf')) <= rank_threshold)

    n_scz = len(scz_genes)
    n_bg = len(background_genes)

    # Original OR
    orig_table = np.array([
        [k_scz_orig, n_scz - k_scz_orig],
        [k_bg_orig, n_bg - k_bg_orig]
    ])
    orig_or, _ = stats.fisher_exact(orig_table)

    # Permutation loop
    permuted_ors = []
    combined_genes = list(scz_genes) + list(background_genes)
    gene_labels = np.array([1] * len(scz_genes) + [0] * len(background_genes))

    for i in range(N_PERMUTATIONS):
        # Shuffle gene labels
        shuffled_labels = np.random.permutation(gene_labels)

        # Compute counts for this permutation
        k_perm_scz = sum(1 for j, g in enumerate(combined_genes)
                         if shuffled_labels[j] == 1 and median_ranks.get(g, float('inf')) <= rank_threshold)
        k_perm_bg = sum(1 for j, g in enumerate(combined_genes)
                        if shuffled_labels[j] == 0 and median_ranks.get(g, float('inf')) <= rank_threshold)

        # Run Fisher's test
        perm_table = np.array([
            [k_perm_scz, n_scz - k_perm_scz],
            [k_perm_bg, n_bg - k_perm_bg]
        ])
        perm_or, _ = stats.fisher_exact(perm_table)
        permuted_ors.append(perm_or)

    permuted_ors = np.array(permuted_ors)

    # Compute empirical p-value: proportion of permuted ORs >= original OR
    emp_pvalue = np.mean(permuted_ors >= orig_or)

    return {
        'original_or': orig_or,
        'permuted_or_mean': float(np.mean(permuted_ors)),
        'permuted_or_std': float(np.std(permuted_ors)),
        'permuted_or_min': float(np.min(permuted_ors)),
        'permuted_or_max': float(np.max(permuted_ors)),
        'empirical_pvalue': float(emp_pvalue),
        'n_permutations': N_PERMUTATIONS
    }


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    # Load data
    gwas_genes, motif_atlas = load_data()

    # Extract gene lists (hgnc_symbol column in Pardiñas dataset)
    scz_genes = gwas_genes['hgnc_symbol'].dropna().unique().tolist()
    print(f"\n  - SCZ genes: {len(scz_genes)}")

    # Check how many SCZ genes are in motif atlas
    atlas_genes = set(motif_atlas.index)
    scz_in_atlas = [g for g in scz_genes if g in atlas_genes]
    scz_missing = [g for g in scz_genes if g not in atlas_genes]
    print(f"  - SCZ genes found in atlas: {len(scz_in_atlas)}")
    print(f"  - SCZ genes NOT in atlas: {len(scz_missing)}")
    if scz_missing:
        print(f"    Missing: {scz_missing[:5]}{'...' if len(scz_missing) > 5 else ''}")

    # Background genes: all atlas genes minus SCZ genes
    background_genes = [g for g in motif_atlas.index if g not in scz_genes]
    print(f"  - Background genes: {len(background_genes)}")

    n_scz = len(scz_in_atlas)
    n_bg = len(background_genes)

    # =========================================================================
    print("\n[3/6] Finding TF motifs in atlas...")
    # =========================================================================

    tf_motif_map = {}
    for tf in ALL_TFS:
        matching = find_tf_motifs(motif_atlas, tf)
        tf_motif_map[tf] = matching
        print(f"  - {tf}: {len(matching)} motifs")

    # Check for TFs with no matches
    no_match_tfs = [tf for tf, motifs in tf_motif_map.items() if len(motifs) == 0]
    if no_match_tfs:
        print(f"\n  WARNING: No motifs found for: {no_match_tfs}")

    # =========================================================================
    print("\n[4/6] Computing Fisher's exact test for all TFs...")
    # =========================================================================

    results = []
    for tf in ALL_TFS:
        tf_motifs = tf_motif_map[tf]

        if len(tf_motifs) == 0:
            print(f"  - {tf}: SKIPPED (no motifs found)")
            continue

        # Compute median ranks for all genes of interest
        all_genes = list(scz_in_atlas) + list(background_genes)
        median_ranks = compute_median_ranks(motif_atlas, tf_motifs, all_genes)

        # Count high-affinity genes
        k_scz = sum(1 for g in scz_in_atlas if median_ranks.get(g, float('inf')) <= RANK_THRESHOLD)
        k_bg = sum(1 for g in background_genes if median_ranks.get(g, float('inf')) <= RANK_THRESHOLD)

        # Run Fisher's exact test
        or_val, ci_low, ci_high, p_val, table = run_fisher_test(k_scz, n_scz, k_bg, n_bg)

        results.append({
            'tf': tf,
            'tf_type': 'primary' if tf in PRIMARY_TFS else 'secondary',
            'n_motifs': len(tf_motifs),
            'motifs': ','.join(tf_motifs[:10]) + ('...' if len(tf_motifs) > 10 else ''),
            'k_scz': k_scz,
            'n_scz': n_scz,
            'k_bg': k_bg,
            'n_bg': n_bg,
            'odds_ratio': or_val,
            'ci_low': ci_low if not np.isnan(ci_low) else None,
            'ci_high': ci_high if not np.isnan(ci_high) else None,
            'p_value': p_val,
            'pct_scz': k_scz / n_scz * 100,
            'pct_bg': k_bg / n_bg * 100
        })

        print(f"  - {tf}: OR={or_val:.2f}, p={p_val:.4f}, {k_scz}/{n_scz} SCZ vs {k_bg}/{n_bg} BG")

    results_df = pd.DataFrame(results)

    # =========================================================================
    print("\n[5/6] Applying Benjamini-Hochberg FDR correction...")
    # =========================================================================

    # Source: Benjamini & Hochberg (1995) "Controlling the False Discovery Rate"
    rejected, p_adjusted, _, _ = multipletests(
        results_df['p_value'].values,
        method='fdr_bh'
    )

    results_df['p_adjusted'] = p_adjusted
    results_df['significant_bh'] = rejected
    results_df['significant_str'] = ['Yes' if r else 'No' for r in rejected]

    # Sort by p-value
    results_df = results_df.sort_values('p_value')

    print("\n  BH-FDR Results:")
    print("  " + "-" * 70)
    for _, row in results_df.iterrows():
        sig_marker = "*" if row['significant_bh'] else " "
        print(f"  {sig_marker} {row['tf']:<10} OR={row['odds_ratio']:.2f}  "
              f"p={row['p_value']:.4f}  BH-FDR={row['p_adjusted']:.4f}  "
              f"{row['significant_str']}")

    # =========================================================================
    print("\n[6/6] Running permutation analysis for top 3 TFs...")
    # =========================================================================

    # Identify top 3 TFs by lowest p-value
    top_3_tfs = results_df.head(3)['tf'].tolist()
    print(f"  Top 3 TFs for permutation: {top_3_tfs}")

    perm_results = []
    for tf in top_3_tfs:
        print(f"  - Running {N_PERMUTATIONS} permutations for {tf}...")
        tf_motifs = tf_motif_map[tf]

        perm_result = run_permutations(
            motif_atlas, tf_motifs, scz_in_atlas, background_genes,
            n_perms=RANDOM_SEED + hash(tf) % 1000  # Slight offset per TF
        )

        perm_results.append({
            'tf': tf,
            'n_motifs': len(tf_motifs),
            **perm_result
        })

        print(f"    Original OR: {perm_result['original_or']:.2f}")
        print(f"    Permuted OR mean ± std: {perm_result['permuted_or_mean']:.2f} ± {perm_result['permuted_or_std']:.2f}")
        print(f"    Permuted OR range: [{perm_result['permuted_or_min']:.2f}, {perm_result['permuted_or_max']:.2f}]")
        print(f"    Empirical p-value: {perm_result['empirical_pvalue']:.4f}")

    perm_results_df = pd.DataFrame(perm_results)

    # =========================================================================
    # Save Results
    # =========================================================================

    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    # Main results TSV
    output_cols = [
        'tf', 'tf_type', 'n_motifs', 'motifs',
        'k_scz', 'n_scz', 'k_bg', 'n_bg',
        'pct_scz', 'pct_bg',
        'odds_ratio', 'ci_low', 'ci_high',
        'p_value', 'p_adjusted', 'significant_str'
    ]
    results_path = OUTPUT_DIR / "d44_motif_enrichment_results.tsv"
    results_df[output_cols].to_csv(results_path, sep='\t', index=False)
    print(f"  - Main results: {results_path}")

    # Contingency tables
    contingency_cols = ['tf', 'n_motifs', 'k_scz', 'n_scz', 'k_bg', 'n_bg', 'pct_scz', 'pct_bg']
    contingency_path = OUTPUT_DIR / "d44_contingency_tables.tsv"
    results_df[contingency_cols].to_csv(contingency_path, sep='\t', index=False)
    print(f"  - Contingency tables: {contingency_path}")

    # Permutation results
    perm_path = OUTPUT_DIR / "d44_permutation_sensitivity.tsv"
    perm_results_df.to_csv(perm_path, sep='\t', index=False)
    print(f"  - Permutation results: {perm_path}")

    # =========================================================================
    # Summary Statistics
    # =========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    significant_tfs = results_df[results_df['significant_bh']]['tf'].tolist()
    print(f"\nTFs with BH-FDR < 0.05: {len(significant_tfs)}")
    if significant_tfs:
        for tf in significant_tfs:
            row = results_df[results_df['tf'] == tf].iloc[0]
            print(f"  - {tf}: OR={row['odds_ratio']:.2f} [{row['ci_low']:.2f}-{row['ci_high']:.2f}], "
                  f"p={row['p_value']:.4f}, FDR={row['p_adjusted']:.4f}")
    else:
        print("  None")

    # Check for EGR1 and CTCF specifically
    egr1_row = results_df[results_df['tf'] == 'EGR1']
    ctcf_row = results_df[results_df['tf'] == 'CTCF']

    print(f"\nEGR1 (primary TF hypothesis):")
    if len(egr1_row) > 0:
        r = egr1_row.iloc[0]
        print(f"  OR={r['odds_ratio']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}], "
              f"p={r['p_value']:.4f}, BH-FDR={r['p_adjusted']:.4f}, "
              f"Significant: {r['significant_str']}")
    else:
        print("  Not tested (no motifs found)")

    print(f"\nCTCF (primary TF hypothesis):")
    if len(ctcf_row) > 0:
        r = ctcf_row.iloc[0]
        print(f"  OR={r['odds_ratio']:.2f} [{r['ci_low']:.2f}-{r['ci_high']:.2f}], "
              f"p={r['p_value']:.4f}, BH-FDR={r['p_adjusted']:.4f}, "
              f"Significant: {r['significant_str']}")
    else:
        print("  Not tested (no motifs found)")

    # Permutation summary
    print(f"\nPermutation analysis (top 3 TFs, {N_PERMUTATIONS} permutations):")
    for _, row in perm_results_df.iterrows():
        print(f"  - {row['tf']}: empirical p={row['empirical_pvalue']:.4f} "
              f"(null OR mean={row['permuted_or_mean']:.2f} ± {row['permuted_or_std']:.2f})")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)

    # Return results for potential further analysis
    return results_df, perm_results_df


if __name__ == "__main__":
    results_df, perm_results_df = main()