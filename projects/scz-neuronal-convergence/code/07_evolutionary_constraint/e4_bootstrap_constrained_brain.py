#!/usr/bin/env python3
"""
E4 (R12): 100-Seed Bootstrap of Constrained-Brain Comparator

Hypothesis tested:
  Whether the single-draw (seed=42) motif enrichment for EGR1 and CTCF
  in the constrained-brain gene set (batch_069 E2) is representative
  of the distribution, or an outlier.

Design:
  - Run 100 random draws (seeds 1-100) of 300 genes from the eligible
    constrained-brain-expressed gene pool (same pool as E2)
  - For each draw, compute Fisher's exact EGR1 and CTCF enrichment
  - Report distribution statistics and where seed=42 falls

Method sources:
  - Fisher's exact: Fisher (1925) "Statistical Methods for Research Workers"
  - Bootstrap CI via percentile method: Efron & Tibshirani (1993) "An Introduction
    to the Bootstrap", Chapman & Hall
  - gnomAD v4.1 pLI: Karczewski et al. 2020 Nature 581:434
  - BrainSpan expression: Miller et al. 2014 Nature 508:199
  - Rank threshold 500 (top 1.85%): from batch_040/d44 design brief
  - Motif atlas: MC9nr cisTarget (Imrichova et al. 2015)

Parameters (all inherited from batch_069/E2):
  - RANK_THRESHOLD = 500 (source: batch_040/d44)
  - PLI_THRESHOLD = 0.5 (source: spec, less stringent than gnomAD 0.9 convention)
  - BRAIN_EXPR_PERCENTILE = 50 (source: top 50% in BrainSpan)
  - TARGET_CONSTRAINED_N = 300 (source: spec)
  - N_BOOTSTRAP = 100 (source: R12 design requirement)
  - SEEDS = 1-100 (source: R12 design requirement)

Predecessor: experiments/batch_069/scripts/e2_motif_specificity.py
"""

import os
import sys
import json
import time
import platform
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration (inherited from batch_069 E2)
# ============================================================================

BASE_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")

# Input files (same as E2)
GWAS_GENES_PATH = BASE_DIR / "experiments/batch_008/data/gwas_genes.parquet"
MOTIF_ATLAS_PATH = BASE_DIR / "data/hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather"
ASD_GENES_PATH = BASE_DIR / "experiments/batch_050/input/asd_satterstrom_2020_fdr05.txt"
DDD_GENES_PATH = BASE_DIR / "experiments/batch_050/input/ddd_kaplanis_2020.txt"
GNOMAD_PATH = BASE_DIR / "data/item_15/gnomad.v4.1.constraint_metrics.tsv"
BRAINSPAN_DIR = BASE_DIR / "data/brainspan/rnaseq"

# Output
OUTPUT_DIR = BASE_DIR / "experiments/batch_070/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "e4_bootstrap_results.json"

# Analysis parameters (all from batch_069 E2 / batch_040 d44)
RANK_THRESHOLD = 500        # Source: batch_040/d44 design brief
PLI_THRESHOLD = 0.5         # Source: batch_069 E2 spec
BRAIN_EXPR_PERCENTILE = 50  # Source: batch_069 E2 spec
TARGET_CONSTRAINED_N = 300  # Source: batch_069 E2 spec

# Bootstrap parameters (source: R12 design requirement from science-critic)
N_BOOTSTRAP = 100
SEEDS = list(range(1, N_BOOTSTRAP + 1))

# Only EGR1 and CTCF needed for R12
TARGET_TFS = ["EGR1", "CTCF"]

# Reference seed from E2 single-draw
REFERENCE_SEED = 42


# ============================================================================
# Environment logging
# ============================================================================

def log_environment():
    """Log environment details for reproducibility."""
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "rank_threshold": RANK_THRESHOLD,
        "pli_threshold": PLI_THRESHOLD,
        "brain_expr_percentile": BRAIN_EXPR_PERCENTILE,
        "target_constrained_n": TARGET_CONSTRAINED_N,
        "n_bootstrap": N_BOOTSTRAP,
        "seeds_range": "1-100",
        "reference_seed": REFERENCE_SEED,
    }
    try:
        import scipy
        env["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    return env


# ============================================================================
# Data Loading (adapted from batch_069 E2)
# ============================================================================

def load_gene_list(path, comment_char="#"):
    """Load gene list from text file, one gene per line, skip comments."""
    genes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(comment_char):
                genes.append(line)
    return genes


def load_brainspan_expression():
    """
    Load BrainSpan RNA-seq expression data and compute mean expression per gene.
    Returns set of gene symbols considered 'brain-expressed' (top 50% percentile).

    Source: BrainSpan Atlas (Miller et al. 2014 Nature 508:199)
    """
    all_gene_means = {}

    for donor_dir in sorted(BRAINSPAN_DIR.iterdir()):
        if not donor_dir.is_dir():
            continue

        tpm_path = donor_dir / "RNAseqTPM.csv"
        genes_path = donor_dir / "Genes.csv"

        if not tpm_path.exists() or not genes_path.exists():
            continue

        print(f"  Loading BrainSpan: {donor_dir.name}")

        # Load gene names
        genes_df = pd.read_csv(genes_path)

        # Load TPM matrix (genes x samples)
        tpm_df = pd.read_csv(tpm_path, header=None)
        tpm_df.columns = ['gene_symbol'] + [f'sample_{i}' for i in range(tpm_df.shape[1] - 1)]

        # Compute mean expression per gene across all brain samples
        sample_cols = [c for c in tpm_df.columns if c != 'gene_symbol']
        tpm_df['mean_expr'] = tpm_df[sample_cols].mean(axis=1)

        for _, row in tpm_df.iterrows():
            gene = row['gene_symbol']
            expr = row['mean_expr']
            if gene in all_gene_means:
                all_gene_means[gene] = (all_gene_means[gene] + expr) / 2
            else:
                all_gene_means[gene] = expr

    # Convert to Series and determine threshold
    expr_series = pd.Series(all_gene_means)
    threshold = expr_series.quantile(1.0 - BRAIN_EXPR_PERCENTILE / 100.0)

    brain_expressed = set(expr_series[expr_series >= threshold].index)
    print(f"  BrainSpan: {len(expr_series)} genes total, "
          f"{len(brain_expressed)} brain-expressed (top {BRAIN_EXPR_PERCENTILE}%)")

    return brain_expressed


def build_eligible_pool(scz_genes, asd_genes, ddd_genes, atlas_genes):
    """
    Build the eligible gene pool for constrained-brain sampling.
    This is done ONCE and then sampled from repeatedly.

    Returns:
        eligible_list: sorted list of eligible genes
        metadata: dict with pool construction info
    """
    print("\n  Building eligible constrained-brain gene pool...")

    # Step 1: Load gnomAD constraint
    print("  Loading gnomAD v4.1 constraint metrics...")
    gnomad = pd.read_csv(GNOMAD_PATH, sep='\t', usecols=['gene', 'canonical', 'lof.pLI'])

    # Use canonical transcripts only
    gnomad_canonical = gnomad[gnomad['canonical'] == True].copy()
    if len(gnomad_canonical) == 0:
        gnomad_canonical = gnomad[gnomad['canonical'].astype(str).str.lower() == 'true'].copy()

    gnomad_canonical['lof.pLI'] = pd.to_numeric(gnomad_canonical['lof.pLI'], errors='coerce')
    gnomad_canonical = gnomad_canonical.dropna(subset=['lof.pLI'])

    # Deduplicate by gene name (take max pLI)
    gnomad_dedup = gnomad_canonical.groupby('gene')['lof.pLI'].max().reset_index()

    constrained = set(gnomad_dedup[gnomad_dedup['lof.pLI'] >= PLI_THRESHOLD]['gene'])
    print(f"  gnomAD: {len(gnomad_dedup)} genes with pLI, "
          f"{len(constrained)} with pLI >= {PLI_THRESHOLD}")

    # Step 2: Load brain expression
    brain_expressed = load_brainspan_expression()

    # Step 3: Intersect
    constrained_brain = constrained & brain_expressed
    print(f"  Constrained AND brain-expressed: {len(constrained_brain)}")

    # Step 4: Restrict to atlas and exclude SCZ/ASD/DDD
    exclude = set(scz_genes) | set(asd_genes) | set(ddd_genes)
    eligible = constrained_brain & set(atlas_genes) - exclude
    print(f"  After excluding SCZ/ASD/DDD and restricting to atlas: {len(eligible)}")

    # Sort for reproducibility
    eligible_list = sorted(eligible)

    metadata = {
        "n_gnomad_genes": len(gnomad_dedup),
        "n_constrained": len(constrained),
        "n_brain_expressed": len(brain_expressed),
        "n_constrained_brain": len(constrained_brain),
        "n_eligible_after_exclusion": len(eligible),
        "pli_threshold": PLI_THRESHOLD,
        "brain_expression_source": "BrainSpan (Miller et al. 2014)",
        "constraint_source": "gnomAD v4.1 (Karczewski et al. 2020)",
    }

    return eligible_list, metadata


# ============================================================================
# Analysis Functions (from batch_069 E2)
# ============================================================================

def find_tf_motifs(motif_atlas, tf_name):
    """Find all motifs matching a TF name (case-insensitive substring match)."""
    tf_upper = tf_name.upper()
    return [col for col in motif_atlas.columns if tf_upper in col.upper()]


def compute_fisher_enrichment(motif_atlas, tf_motifs, test_genes, all_atlas_genes,
                              rank_threshold=RANK_THRESHOLD):
    """
    Compute Fisher's exact test for motif enrichment.
    Gene is 'enriched' if median rank across TF motifs <= threshold.
    Background = all atlas genes NOT in test_genes.

    Source: Fisher (1925), Woolf (1955) for CI on log-OR.
    """
    test_in_atlas = [g for g in test_genes if g in all_atlas_genes]
    bg_genes = [g for g in all_atlas_genes if g not in set(test_genes)]

    n_test = len(test_in_atlas)
    n_bg = len(bg_genes)

    if n_test == 0 or len(tf_motifs) == 0:
        return {"odds_ratio": None, "p_value": None, "k_test": 0, "n_test": n_test}

    # Compute median ranks for test genes
    test_data = motif_atlas.loc[test_in_atlas, tf_motifs]
    test_median = test_data.median(axis=1)
    k_test = int((test_median <= rank_threshold).sum())

    # Compute median ranks for background genes
    bg_data = motif_atlas.loc[bg_genes, tf_motifs]
    bg_median = bg_data.median(axis=1)
    k_bg = int((bg_median <= rank_threshold).sum())

    # Fisher's exact test
    table = np.array([
        [k_test, n_test - k_test],
        [k_bg, n_bg - k_bg]
    ])
    odds_ratio, p_value = stats.fisher_exact(table)

    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "k_test": k_test,
        "n_test": n_test,
        "k_bg": k_bg,
        "n_bg": n_bg,
    }


# ============================================================================
# Bootstrap
# ============================================================================

def run_bootstrap(eligible_list, motif_atlas, tf_motif_map, all_atlas_genes):
    """
    Run 100 bootstrap draws from the eligible pool. For each draw:
    - Sample 300 genes (without replacement) using a different seed
    - Compute EGR1 and CTCF Fisher's exact enrichment

    Returns list of dicts, one per seed.
    """
    results = []

    for i, seed in enumerate(SEEDS):
        np.random.seed(seed)
        sampled = list(np.random.choice(eligible_list, size=TARGET_CONSTRAINED_N, replace=False))

        draw_result = {"seed": seed}

        for tf in TARGET_TFS:
            tf_motifs = tf_motif_map[tf]
            enrichment = compute_fisher_enrichment(
                motif_atlas, tf_motifs, sampled, all_atlas_genes
            )
            draw_result[f"{tf}_OR"] = enrichment["odds_ratio"]
            draw_result[f"{tf}_p"] = enrichment["p_value"]
            draw_result[f"{tf}_k_test"] = enrichment["k_test"]
            draw_result[f"{tf}_n_test"] = enrichment["n_test"]

        results.append(draw_result)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{N_BOOTSTRAP} draws...")

    return results


def compute_distribution_stats(bootstrap_results, tf_name):
    """
    Compute summary statistics for the OR distribution across 100 draws.
    Also locates where the reference seed=42 falls.
    """
    ors = [r[f"{tf_name}_OR"] for r in bootstrap_results if r[f"{tf_name}_OR"] is not None]
    ps = [r[f"{tf_name}_p"] for r in bootstrap_results if r[f"{tf_name}_p"] is not None]

    # Find seed=42 result
    seed42_result = next((r for r in bootstrap_results if r["seed"] == REFERENCE_SEED), None)
    seed42_or = seed42_result[f"{tf_name}_OR"] if seed42_result else None
    seed42_p = seed42_result[f"{tf_name}_p"] if seed42_result else None

    or_array = np.array(ors)
    p_array = np.array(ps)

    mean_or = float(np.mean(or_array))
    median_or = float(np.median(or_array))
    sd_or = float(np.std(or_array, ddof=1))
    ci_low = float(np.percentile(or_array, 2.5))
    ci_high = float(np.percentile(or_array, 97.5))
    min_or = float(np.min(or_array))
    max_or = float(np.max(or_array))

    # p-value distribution
    mean_p = float(np.mean(p_array))
    median_p = float(np.median(p_array))
    pct_significant = float(np.mean(p_array < 0.05) * 100)

    # Where does seed=42 fall?
    if seed42_or is not None:
        percentile_rank = float(np.mean(or_array <= seed42_or) * 100)
        z_score = (seed42_or - mean_or) / sd_or if sd_or > 0 else 0.0
        if abs(z_score) <= 1.0:
            deviation_band = "within_1SD"
        elif abs(z_score) <= 2.0:
            deviation_band = "between_1SD_and_2SD"
        else:
            deviation_band = "outside_2SD"
    else:
        percentile_rank = None
        z_score = None
        deviation_band = None

    return {
        "n_valid_draws": len(ors),
        "or_mean": mean_or,
        "or_median": median_or,
        "or_sd": sd_or,
        "or_95ci_low": ci_low,
        "or_95ci_high": ci_high,
        "or_min": min_or,
        "or_max": max_or,
        "p_mean": mean_p,
        "p_median": median_p,
        "pct_draws_significant_p05": pct_significant,
        "seed42_or": seed42_or,
        "seed42_p": seed42_p,
        "seed42_percentile_rank": percentile_rank,
        "seed42_z_score": float(z_score) if z_score is not None else None,
        "seed42_deviation_band": deviation_band,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()

    print("=" * 78)
    print("E4 (R12): 100-Seed Bootstrap of Constrained-Brain Comparator")
    print("=" * 78)

    env = log_environment()
    print(f"\nEnvironment: Python {sys.version.split()[0]}, "
          f"NumPy {np.__version__}, Pandas {pd.__version__}")
    print(f"Bootstrap: {N_BOOTSTRAP} draws, seeds 1-{N_BOOTSTRAP}")
    print(f"Reference seed from E2: {REFERENCE_SEED}")

    # ------------------------------------------------------------------
    # Step 1: Load motif atlas (ONCE)
    # ------------------------------------------------------------------
    print("\n[1/4] Loading motif atlas...")
    motif_atlas = pd.read_feather(MOTIF_ATLAS_PATH)
    motif_atlas = motif_atlas.set_index('motifs').T
    motif_atlas.index.name = 'gene'
    all_atlas_genes = set(motif_atlas.index)
    print(f"  Atlas shape: {motif_atlas.shape} (genes x motifs)")

    # ------------------------------------------------------------------
    # Step 2: Load gene sets and build eligible pool (ONCE)
    # ------------------------------------------------------------------
    print("\n[2/4] Loading gene sets and building eligible pool...")

    # SCZ genes
    scz_df = pd.read_parquet(GWAS_GENES_PATH)
    scz_genes = scz_df['hgnc_symbol'].dropna().unique().tolist()
    print(f"  SCZ: {len(scz_genes)} genes")

    # ASD genes
    asd_genes = load_gene_list(ASD_GENES_PATH)
    print(f"  ASD: {len(asd_genes)} genes")

    # DDD genes
    ddd_genes = load_gene_list(DDD_GENES_PATH)
    print(f"  DDD: {len(ddd_genes)} genes")

    # Build eligible pool
    eligible_list, pool_meta = build_eligible_pool(
        scz_genes, asd_genes, ddd_genes, all_atlas_genes
    )
    print(f"\n  Eligible pool size: {len(eligible_list)} genes")
    print(f"  Target sample size per draw: {TARGET_CONSTRAINED_N}")

    if len(eligible_list) < TARGET_CONSTRAINED_N:
        print(f"  WARNING: Pool ({len(eligible_list)}) < target ({TARGET_CONSTRAINED_N}). "
              f"Using all eligible genes for every draw (no variability).")
        # This would mean bootstrap is pointless -- flag it
        pool_meta["warning"] = "pool_smaller_than_target"

    # ------------------------------------------------------------------
    # Step 3: Find TF motifs (ONCE)
    # ------------------------------------------------------------------
    print("\n[3/4] Finding TF motifs...")
    tf_motif_map = {}
    for tf in TARGET_TFS:
        matching = find_tf_motifs(motif_atlas, tf)
        tf_motif_map[tf] = matching
        print(f"  {tf}: {len(matching)} motifs found")

    # ------------------------------------------------------------------
    # Step 4: Run 100-draw bootstrap
    # ------------------------------------------------------------------
    print(f"\n[4/4] Running {N_BOOTSTRAP}-draw bootstrap...")
    print("-" * 78)

    bootstrap_results = run_bootstrap(eligible_list, motif_atlas, tf_motif_map, all_atlas_genes)

    # ------------------------------------------------------------------
    # Compute distribution statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BOOTSTRAP DISTRIBUTION RESULTS")
    print("=" * 78)

    distribution_stats = {}
    for tf in TARGET_TFS:
        tf_stats = compute_distribution_stats(bootstrap_results, tf)
        distribution_stats[tf] = tf_stats

        print(f"\n{tf}:")
        print(f"  OR distribution (N={tf_stats['n_valid_draws']} draws):")
        print(f"    Mean:   {tf_stats['or_mean']:.3f}")
        print(f"    Median: {tf_stats['or_median']:.3f}")
        print(f"    SD:     {tf_stats['or_sd']:.3f}")
        print(f"    95% CI: [{tf_stats['or_95ci_low']:.3f}, {tf_stats['or_95ci_high']:.3f}]")
        print(f"    Range:  [{tf_stats['or_min']:.3f}, {tf_stats['or_max']:.3f}]")
        print(f"  p-value distribution:")
        print(f"    Mean p:   {tf_stats['p_mean']:.4f}")
        print(f"    Median p: {tf_stats['p_median']:.4f}")
        print(f"    % draws with p<0.05: {tf_stats['pct_draws_significant_p05']:.1f}%")
        print(f"  Seed=42 placement:")
        print(f"    OR={tf_stats['seed42_or']:.3f}, p={tf_stats['seed42_p']:.4f}")
        print(f"    Percentile rank: {tf_stats['seed42_percentile_rank']:.1f}th")
        print(f"    Z-score: {tf_stats['seed42_z_score']:.2f}")
        print(f"    Band: {tf_stats['seed42_deviation_band']}")

    # ------------------------------------------------------------------
    # Interpretive summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("INTERPRETIVE SUMMARY")
    print("=" * 78)

    for tf in TARGET_TFS:
        s = distribution_stats[tf]
        print(f"\n{tf}:")
        if s['seed42_deviation_band'] == 'within_1SD':
            print(f"  Seed=42 is WITHIN 1 SD of the bootstrap mean.")
            print(f"  --> The E2 single-draw result IS REPRESENTATIVE of the distribution.")
        elif s['seed42_deviation_band'] == 'between_1SD_and_2SD':
            print(f"  Seed=42 is between 1 and 2 SD from the bootstrap mean.")
            print(f"  --> The E2 single-draw result is somewhat typical but slightly extreme.")
        else:
            print(f"  Seed=42 is OUTSIDE 2 SD from the bootstrap mean.")
            print(f"  --> The E2 single-draw result is an OUTLIER. Conclusions should be revised.")

        # Overall pattern
        if s['pct_draws_significant_p05'] >= 80:
            print(f"  {s['pct_draws_significant_p05']:.0f}% of draws yield p<0.05 "
                  f"--> enrichment is ROBUST across draws.")
        elif s['pct_draws_significant_p05'] >= 50:
            print(f"  {s['pct_draws_significant_p05']:.0f}% of draws yield p<0.05 "
                  f"--> enrichment is MODERATELY ROBUST.")
        elif s['pct_draws_significant_p05'] >= 20:
            print(f"  {s['pct_draws_significant_p05']:.0f}% of draws yield p<0.05 "
                  f"--> enrichment is DRAW-DEPENDENT (not robust).")
        else:
            print(f"  {s['pct_draws_significant_p05']:.0f}% of draws yield p<0.05 "
                  f"--> enrichment is ABSENT in most draws.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    t_elapsed = time.time() - t_start

    output = {
        "experiment_id": "batch_070_e4_bootstrap_constrained_brain",
        "hypothesis": "E2 seed=42 single-draw is representative of constrained-brain enrichment distribution",
        "design": {
            "n_bootstrap_draws": N_BOOTSTRAP,
            "seeds": "1-100",
            "reference_seed": REFERENCE_SEED,
            "target_sample_size": TARGET_CONSTRAINED_N,
            "tfs_tested": TARGET_TFS,
            "rank_threshold": RANK_THRESHOLD,
            "rank_threshold_source": "batch_040/d44 design brief (top 1.85%)",
            "pli_threshold": PLI_THRESHOLD,
            "brain_expr_percentile": BRAIN_EXPR_PERCENTILE,
        },
        "eligible_pool": pool_meta,
        "distribution_stats": distribution_stats,
        "all_draws": bootstrap_results,
        "environment": env,
        "runtime_seconds": round(t_elapsed, 1),
        "status": "completed",
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Runtime: {t_elapsed:.1f} seconds")
    print("=" * 78)
    print("E4 COMPLETE")
    print("=" * 78)

    return output


if __name__ == "__main__":
    main()
