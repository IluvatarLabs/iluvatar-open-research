#!/usr/bin/env python3
"""
E2 (R2): EGR1/CTCF/MEF2C Motif Specificity Comparators

Hypothesis tested:
  Whether EGR1 (OR=4.98), CTCF (OR=3.15), and MEF2C (OR=3.84) motif
  enrichment in SCZ genes is SCZ-specific or appears at any constrained
  brain-expressed gene set.

Design:
  Compare motif enrichment (median-rank <= 500, Fisher's exact) across:
    1. SCZ GWAS genes (reference, from batch_008)
    2. ASD Satterstrom 2020 FDR<0.05 (n=78)
    3. DDD Kaplanis 2020 (n~285)
    4. Non-SCZ constrained brain-expressed genes (n~300, constructed)

  TFs tested: EGR1, CTCF, MEF2C
  Background: all atlas genes NOT in the test set (varies per comparator)

Method sources:
  - Fisher's exact: Fisher (1925) "Statistical Methods for Research Workers"
  - Woolf CI for log-OR: Woolf (1955) Ann Hum Genet 19:251
  - Rank threshold 500 (top 1.85%): from batch_040/d44 design brief
  - Motif atlas: MC9nr cisTarget (Imrichova et al. 2015)
  - gnomAD v4.1 pLI: Karczewski et al. 2020 Nature 581:434
  - BrainSpan expression: Miller et al. 2014 Nature 508:199

Pre-registered power notes:
  - ASD EGR1: expected k ~ 0.6 at background rate (~0.8%). Likely UNINTERPRETABLE.
  - DDD EGR1: expected k ~ 2 at background rate. Marginal power.
  - ASD CTCF: expected k ~ 1.2 at background rate (~1.6%). Also marginal.
  - Non-SCZ constrained: expected k ~ 2.4 (EGR1), ~4.7 (CTCF). Testable.

Adapted from: experiments/batch_040/d44_motif_enrichment.py
"""

import os
import sys
import json
import platform
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")

# Input files
GWAS_GENES_PATH = BASE_DIR / "experiments/batch_008/data/gwas_genes.parquet"
MOTIF_ATLAS_PATH = BASE_DIR / "data/hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather"
ASD_GENES_PATH = BASE_DIR / "experiments/batch_050/input/asd_satterstrom_2020_fdr05.txt"
DDD_GENES_PATH = BASE_DIR / "experiments/batch_050/input/ddd_kaplanis_2020.txt"
GNOMAD_PATH = BASE_DIR / "data/item_15/gnomad.v4.1.constraint_metrics.tsv"
BRAINSPAN_DIR = BASE_DIR / "data/brainspan/rnaseq"

# Output
OUTPUT_DIR = BASE_DIR / "experiments/batch_069/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "e2_motif_specificity.json"

# Analysis parameters
# Rank threshold 500: top 1.85% of promoters (500/27090)
# Source: batch_040/d44 design brief
RANK_THRESHOLD = 500

# TFs to test
# EGR1 OR=4.98, CTCF OR=3.15, MEF2C OR=3.84 from batch_040 results
TARGET_TFS = ["EGR1", "CTCF", "MEF2C"]

# pLI threshold for constrained genes
# Source: gnomAD convention: pLI >= 0.9 is "constrained"; we use >= 0.5
# because the spec says pLI >= 0.5 (less stringent to get enough genes)
PLI_THRESHOLD = 0.5

# Target size for non-SCZ constrained set
# Source: spec says n ~ 300
TARGET_CONSTRAINED_N = 300

# Brain expression threshold: mean TPM > 1 across brain regions
# Source: GTEx convention for "expressed" genes is TPM > 0.5-1.0
# (Consortium 2020 Science 369:1318). Using 1.0 as BrainSpan provides
# fraction-of-total not raw TPM, so we threshold on relative expression.
BRAIN_EXPR_PERCENTILE = 50  # top 50% expressed in brain = brain-expressed

RANDOM_SEED = 42


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
        "scipy_version": stats.scipy.__version__ if hasattr(stats, 'scipy') else "unknown",
        "random_seed": RANDOM_SEED,
        "rank_threshold": RANK_THRESHOLD,
        "pli_threshold": PLI_THRESHOLD,
    }
    try:
        import scipy
        env["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    return env


# ============================================================================
# Data Loading
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
    Load BrainSpan RNA-seq expression data (both donors) and compute
    mean expression per gene across all brain samples.

    Returns set of gene symbols considered 'brain-expressed'.

    Source: BrainSpan Atlas (Miller et al. 2014 Nature 508:199)
    The TPM values in BrainSpan are fraction-of-total (not standard TPM),
    so we use within-dataset percentile rank to define brain-expressed.
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
        gene_symbols = genes_df['gene_symbol'].tolist()

        # Load TPM matrix (genes x samples)
        # First column is gene symbol, rest are samples
        tpm_df = pd.read_csv(tpm_path, header=None)
        tpm_df.columns = ['gene_symbol'] + [f'sample_{i}' for i in range(tpm_df.shape[1] - 1)]

        # Compute mean expression per gene across all brain samples
        sample_cols = [c for c in tpm_df.columns if c != 'gene_symbol']
        tpm_df['mean_expr'] = tpm_df[sample_cols].mean(axis=1)

        for _, row in tpm_df.iterrows():
            gene = row['gene_symbol']
            expr = row['mean_expr']
            if gene in all_gene_means:
                # Average across donors
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


def construct_constrained_brain_set(scz_genes, asd_genes, ddd_genes, atlas_genes):
    """
    Construct non-SCZ constrained brain-expressed gene set.

    Steps:
    1. Load gnomAD v4.1 constraint (pLI >= 0.5)
    2. Load BrainSpan expression (top 50%)
    3. Intersect constrained + brain-expressed
    4. Exclude SCZ, ASD, DDD genes
    5. Random sample to ~300 if needed

    Sources:
    - pLI >= 0.5: spec requirement (less stringent than gnomAD convention of 0.9)
    - gnomAD v4.1: Karczewski et al. 2020 Nature 581:434
    - BrainSpan: Miller et al. 2014 Nature 508:199
    """
    print("\n  Constructing non-SCZ constrained brain-expressed gene set...")

    # Step 1: Load gnomAD constraint
    print("  Loading gnomAD v4.1 constraint metrics...")
    gnomad = pd.read_csv(GNOMAD_PATH, sep='\t', usecols=['gene', 'canonical', 'lof.pLI'])

    # Use canonical transcripts only (to avoid duplicates)
    # canonical column is 'true'/'false' string in gnomAD v4.1
    gnomad_canonical = gnomad[gnomad['canonical'] == True].copy()
    if len(gnomad_canonical) == 0:
        # Try string 'true'
        gnomad_canonical = gnomad[gnomad['canonical'].astype(str).str.lower() == 'true'].copy()

    # Convert pLI to numeric, drop NAs
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

    # Step 4: Restrict to genes in atlas AND exclude SCZ/ASD/DDD
    exclude = set(scz_genes) | set(asd_genes) | set(ddd_genes)
    eligible = constrained_brain & set(atlas_genes) - exclude
    print(f"  After excluding SCZ/ASD/DDD and restricting to atlas: {len(eligible)}")

    # Step 5: Sample to target size if needed
    np.random.seed(RANDOM_SEED)
    eligible_list = sorted(eligible)  # Sort for reproducibility before sampling

    if len(eligible_list) > TARGET_CONSTRAINED_N:
        sampled = list(np.random.choice(eligible_list, size=TARGET_CONSTRAINED_N, replace=False))
        print(f"  Sampled {TARGET_CONSTRAINED_N} from {len(eligible_list)} eligible genes")
    else:
        sampled = eligible_list
        print(f"  Using all {len(sampled)} eligible genes (< target {TARGET_CONSTRAINED_N})")

    return sampled, {
        "n_gnomad_genes": len(gnomad_dedup),
        "n_constrained": len(constrained),
        "n_brain_expressed": len(brain_expressed),
        "n_constrained_brain": len(constrained_brain),
        "n_eligible_after_exclusion": len(eligible),
        "n_final": len(sampled),
        "pli_threshold": PLI_THRESHOLD,
        "brain_expression_source": "BrainSpan (Miller et al. 2014)",
        "constraint_source": "gnomAD v4.1 (Karczewski et al. 2020)",
    }


# ============================================================================
# Analysis Functions (adapted from batch_040/d44_motif_enrichment.py)
# ============================================================================

def find_tf_motifs(motif_atlas, tf_name):
    """
    Find all motifs matching a TF name.
    Adapted from d44_motif_enrichment.py find_tf_motifs().
    """
    tf_upper = tf_name.upper()
    return [col for col in motif_atlas.columns if tf_upper in col.upper()]


def compute_fisher_enrichment(motif_atlas, tf_motifs, test_genes, all_atlas_genes,
                              rank_threshold=RANK_THRESHOLD):
    """
    Compute Fisher's exact test for motif enrichment in test_genes vs background.

    For each gene, compute the median rank across all motifs for the TF.
    A gene is 'enriched' if median rank <= threshold (lower rank = stronger binding).

    Background = all atlas genes NOT in test_genes.

    Returns dict with OR, p-value, CI, and contingency cell counts.

    Sources:
    - Fisher's exact: Fisher (1925)
    - Woolf CI for log-OR: Woolf (1955) Ann Hum Genet 19:251
    """
    # Restrict to genes present in atlas
    test_in_atlas = [g for g in test_genes if g in all_atlas_genes]
    bg_genes = [g for g in all_atlas_genes if g not in set(test_genes)]

    n_test = len(test_in_atlas)
    n_bg = len(bg_genes)

    if n_test == 0 or len(tf_motifs) == 0:
        return {
            "odds_ratio": None, "p_value": None,
            "ci_low": None, "ci_high": None,
            "k_test": 0, "n_test": n_test,
            "k_bg": 0, "n_bg": n_bg,
            "pct_test": 0, "pct_bg": 0,
            "error": "no genes or motifs"
        }

    # Compute median ranks for test genes
    test_data = motif_atlas.loc[test_in_atlas, tf_motifs]
    test_median = test_data.median(axis=1)
    k_test = int((test_median <= rank_threshold).sum())

    # Compute median ranks for background genes
    bg_data = motif_atlas.loc[bg_genes, tf_motifs]
    bg_median = bg_data.median(axis=1)
    k_bg = int((bg_median <= rank_threshold).sum())

    # Fisher's exact test
    # [[test enriched, test not], [bg enriched, bg not]]
    table = np.array([
        [k_test, n_test - k_test],
        [k_bg, n_bg - k_bg]
    ])
    odds_ratio, p_value = stats.fisher_exact(table)

    # Woolf CI for log-OR (Woolf 1955)
    a, b = k_test, n_test - k_test
    c, d = k_bg, n_bg - k_bg
    if all(x > 0 for x in [a, b, c, d]):
        log_or = np.log(odds_ratio)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_low = float(np.exp(log_or - 1.96 * se))
        ci_high = float(np.exp(log_or + 1.96 * se))
    else:
        ci_low = None
        ci_high = None

    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "k_test": k_test,
        "n_test": n_test,
        "k_bg": k_bg,
        "n_bg": n_bg,
        "pct_test": round(k_test / n_test * 100, 2) if n_test > 0 else 0,
        "pct_bg": round(k_bg / n_bg * 100, 2) if n_bg > 0 else 0,
    }


# ============================================================================
# Power Analysis
# ============================================================================

def compute_expected_counts(n_genes, bg_rate):
    """Compute expected count at background rate for power assessment."""
    return n_genes * bg_rate


def assess_interpretability(n_genes, k_observed, bg_rate, tf_name, gene_set_name):
    """
    Assess whether the result has enough counts for interpretation.

    With Fisher's exact, very small cell counts (expected < 5 or observed < 3)
    yield wide CIs and results should be flagged as UNINTERPRETABLE or MARGINAL.

    However, if the observed count far exceeds expectation (k_observed >> expected),
    the result is informative despite low expected counts -- it demonstrates strong
    enrichment. In that case, mark as MARGINAL_BUT_INFORMATIVE.
    """
    expected = compute_expected_counts(n_genes, bg_rate)

    if expected < 1.0 and k_observed < 1:
        return "UNINTERPRETABLE", f"Expected k={expected:.1f}, observed k={k_observed}"
    elif expected < 1.0 and k_observed >= 3:
        # Low expected but high observed = strong enrichment signal
        return "MARGINAL_BUT_INFORMATIVE", (
            f"Expected k={expected:.1f} but observed k={k_observed}; "
            f"low power to detect null but enrichment is clear"
        )
    elif expected < 1.0:
        return "UNINTERPRETABLE", f"Expected k={expected:.1f} at bg rate {bg_rate:.4f}"
    elif expected < 3.0 or (k_observed < 3):
        return "MARGINAL", f"Expected k={expected:.1f}, observed k={k_observed}"
    else:
        return "INTERPRETABLE", f"Expected k={expected:.1f}, observed k={k_observed}"


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 78)
    print("E2 (R2): EGR1/CTCF/MEF2C Motif Specificity Comparators")
    print("=" * 78)

    env = log_environment()
    print(f"\nEnvironment: Python {sys.version.split()[0]}, "
          f"NumPy {np.__version__}, Pandas {pd.__version__}")
    print(f"Seed: {RANDOM_SEED}")

    # ------------------------------------------------------------------
    # Step 1: Load motif atlas
    # ------------------------------------------------------------------
    print("\n[1/5] Loading motif atlas...")
    motif_atlas = pd.read_feather(MOTIF_ATLAS_PATH)
    motif_atlas = motif_atlas.set_index('motifs').T
    motif_atlas.index.name = 'gene'
    all_atlas_genes = set(motif_atlas.index)
    print(f"  Atlas shape: {motif_atlas.shape} (genes x motifs)")

    # ------------------------------------------------------------------
    # Step 2: Load gene sets
    # ------------------------------------------------------------------
    print("\n[2/5] Loading gene sets...")

    # SCZ genes
    scz_df = pd.read_parquet(GWAS_GENES_PATH)
    scz_genes = scz_df['hgnc_symbol'].dropna().unique().tolist()
    scz_in_atlas = [g for g in scz_genes if g in all_atlas_genes]
    print(f"  SCZ: {len(scz_genes)} total, {len(scz_in_atlas)} in atlas")

    # ASD genes
    asd_genes = load_gene_list(ASD_GENES_PATH)
    asd_in_atlas = [g for g in asd_genes if g in all_atlas_genes]
    print(f"  ASD (Satterstrom 2020 FDR<0.05): {len(asd_genes)} total, "
          f"{len(asd_in_atlas)} in atlas")

    # DDD genes
    ddd_genes = load_gene_list(DDD_GENES_PATH)
    ddd_in_atlas = [g for g in ddd_genes if g in all_atlas_genes]
    print(f"  DDD (Kaplanis 2020): {len(ddd_genes)} total, "
          f"{len(ddd_in_atlas)} in atlas")

    # Non-SCZ constrained brain-expressed
    constrained_genes, constrained_meta = construct_constrained_brain_set(
        scz_genes, asd_genes, ddd_genes, all_atlas_genes
    )
    constrained_in_atlas = [g for g in constrained_genes if g in all_atlas_genes]
    print(f"  Non-SCZ constrained brain-expressed: {len(constrained_genes)} total, "
          f"{len(constrained_in_atlas)} in atlas")

    # Gene set overlap check
    scz_set = set(scz_genes)
    asd_set = set(asd_genes)
    ddd_set = set(ddd_genes)
    const_set = set(constrained_genes)
    print(f"\n  Overlaps:")
    print(f"    SCZ & ASD: {len(scz_set & asd_set)} genes")
    print(f"    SCZ & DDD: {len(scz_set & ddd_set)} genes")
    print(f"    ASD & DDD: {len(asd_set & ddd_set)} genes")
    print(f"    SCZ & Constrained: {len(scz_set & const_set)} genes")
    print(f"    ASD & Constrained: {len(asd_set & const_set)} genes")
    print(f"    DDD & Constrained: {len(ddd_set & const_set)} genes")

    # ------------------------------------------------------------------
    # Step 3: Find TF motifs
    # ------------------------------------------------------------------
    print("\n[3/5] Finding TF motifs in atlas...")
    tf_motif_map = {}
    for tf in TARGET_TFS:
        matching = find_tf_motifs(motif_atlas, tf)
        tf_motif_map[tf] = matching
        print(f"  {tf}: {len(matching)} motifs found")

    # ------------------------------------------------------------------
    # Step 4: Run Fisher's exact for all gene sets x TFs
    # ------------------------------------------------------------------
    print("\n[4/5] Running Fisher's exact tests...")
    print("-" * 78)

    gene_sets = {
        "SCZ_GWAS": scz_in_atlas,
        "ASD_Satterstrom2020": asd_in_atlas,
        "DDD_Kaplanis2020": ddd_in_atlas,
        "NonSCZ_Constrained_Brain": constrained_in_atlas,
    }

    all_results = {}
    comparison_table = []

    for gs_name, gs_genes in gene_sets.items():
        all_results[gs_name] = {"n_genes_in_atlas": len(gs_genes)}
        row = {"Gene_Set": gs_name, "n_genes": len(gs_genes)}

        for tf in TARGET_TFS:
            tf_motifs = tf_motif_map[tf]
            result = compute_fisher_enrichment(
                motif_atlas, tf_motifs, gs_genes, all_atlas_genes
            )
            all_results[gs_name][tf] = result

            # Compute interpretability
            bg_rate = result["k_bg"] / result["n_bg"] if result["n_bg"] > 0 else 0
            interp_status, interp_note = assess_interpretability(
                len(gs_genes), result["k_test"], bg_rate, tf, gs_name
            )
            result["interpretability"] = interp_status
            result["interpretability_note"] = interp_note

            or_val = result["odds_ratio"]
            p_val = result["p_value"]

            or_str = f"{or_val:.2f}" if or_val is not None else "N/A"
            p_str = f"{p_val:.4f}" if p_val is not None else "N/A"

            row[f"{tf}_OR"] = or_val
            row[f"{tf}_p"] = p_val
            row[f"{tf}_k"] = result["k_test"]
            row[f"{tf}_interp"] = interp_status

            print(f"  {gs_name:30s} | {tf:6s} | OR={or_str:>7s} | "
                  f"p={p_str:>8s} | k={result['k_test']:3d}/{result['n_test']:5d} "
                  f"vs {result['k_bg']:4d}/{result['n_bg']:5d} | {interp_status}")

        comparison_table.append(row)

    # ------------------------------------------------------------------
    # Step 5: Summary comparison table
    # ------------------------------------------------------------------
    print("\n[5/5] Comparison Table")
    print("=" * 78)

    header = (f"{'Gene_Set':30s} | {'n':>5s} | "
              f"{'EGR1_OR':>8s} {'EGR1_p':>10s} | "
              f"{'CTCF_OR':>8s} {'CTCF_p':>10s} | "
              f"{'MEF2C_OR':>8s} {'MEF2C_p':>10s}")
    print(header)
    print("-" * len(header))

    for row in comparison_table:
        def fmt_or(v):
            return f"{v:.2f}" if v is not None else "N/A"
        def fmt_p(v):
            return f"{v:.6f}" if v is not None else "N/A"

        line = (f"{row['Gene_Set']:30s} | {row['n_genes']:5d} | "
                f"{fmt_or(row.get('EGR1_OR')):>8s} {fmt_p(row.get('EGR1_p')):>10s} | "
                f"{fmt_or(row.get('CTCF_OR')):>8s} {fmt_p(row.get('CTCF_p')):>10s} | "
                f"{fmt_or(row.get('MEF2C_OR')):>8s} {fmt_p(row.get('MEF2C_p')):>10s}")
        print(line)

    # ------------------------------------------------------------------
    # Interpretive summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("INTERPRETIVE SUMMARY")
    print("=" * 78)

    scz_ref = all_results["SCZ_GWAS"]

    for tf in TARGET_TFS:
        scz_or = scz_ref[tf]["odds_ratio"]
        scz_p = scz_ref[tf]["p_value"]
        print(f"\n{tf}:")
        print(f"  SCZ reference: OR={scz_or:.2f}, p={scz_p:.6f}")

        specificity_assessment = "SCZ-SPECIFIC"
        enriched_elsewhere = []

        for gs_name in ["ASD_Satterstrom2020", "DDD_Kaplanis2020",
                        "NonSCZ_Constrained_Brain"]:
            comp = all_results[gs_name][tf]
            comp_or = comp["odds_ratio"]
            comp_p = comp["p_value"]
            interp = comp["interpretability"]

            if interp == "UNINTERPRETABLE":
                print(f"  {gs_name}: OR={comp_or:.2f}, p={comp_p:.4f} -- "
                      f"UNINTERPRETABLE (insufficient counts)")
            elif interp == "MARGINAL_BUT_INFORMATIVE" and comp_p < 0.05 and comp_or > 1.5:
                specificity_assessment = "SHARED"
                enriched_elsewhere.append(gs_name)
                print(f"  {gs_name}: OR={comp_or:.2f}, p={comp_p:.4f} -- "
                      f"ALSO ENRICHED (low expected but observed far exceeds null)")
            elif comp_p is not None and comp_p < 0.05 and comp_or is not None and comp_or > 1.5:
                specificity_assessment = "SHARED"
                enriched_elsewhere.append(gs_name)
                print(f"  {gs_name}: OR={comp_or:.2f}, p={comp_p:.4f} -- "
                      f"ALSO ENRICHED (NOT SCZ-specific)")
            else:
                print(f"  {gs_name}: OR={comp_or:.2f}, p={comp_p:.4f} -- "
                      f"NOT enriched ({interp})")

        if specificity_assessment == "SCZ-SPECIFIC":
            print(f"  --> ASSESSMENT: {tf} enrichment appears SCZ-SPECIFIC "
                  f"(not replicated in comparators)")
        else:
            print(f"  --> ASSESSMENT: {tf} enrichment is SHARED with: "
                  f"{', '.join(enriched_elsewhere)}")
            print(f"      This suggests the enrichment may reflect general "
                  f"neurodevelopmental constraint rather than SCZ specificity.")

    # ------------------------------------------------------------------
    # Pre-registered power notes in output
    # ------------------------------------------------------------------
    power_notes = {
        "ASD_EGR1": "Expected k~0.6 at background rate. Likely UNINTERPRETABLE per pre-registration.",
        "DDD_EGR1": "Expected k~2 at background rate. Marginal power per pre-registration.",
        "ASD_CTCF": "Expected k~1.2 at background rate. Also marginal per pre-registration.",
        "NonSCZ_Constrained_EGR1": "Expected k~2.4 at background rate. Testable.",
        "NonSCZ_Constrained_CTCF": "Expected k~4.7 at background rate. Testable.",
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "experiment_id": "batch_069_e2_motif_specificity",
        "hypothesis": "EGR1/CTCF/MEF2C motif enrichment in SCZ genes is SCZ-specific",
        "tfs_tested": TARGET_TFS,
        "rank_threshold": RANK_THRESHOLD,
        "rank_threshold_source": "batch_040/d44 design brief (top 1.85% of promoters)",
        "gene_sets": {
            gs_name: {
                "n_total": len(genes),
                "n_in_atlas": all_results[gs_name]["n_genes_in_atlas"],
            }
            for gs_name, genes in gene_sets.items()
        },
        "constrained_set_construction": constrained_meta,
        "results": all_results,
        "comparison_table": comparison_table,
        "power_notes": power_notes,
        "environment": env,
        "status": "completed",
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("=" * 78)
    print("E2 COMPLETE")
    print("=" * 78)

    return output


if __name__ == "__main__":
    main()
