#!/usr/bin/env python3
"""
batch_037 — Ancestry + Sex Stratification Gene-Level Analysis (D18)
====================================================================
Run Stouffer's Z gene-level aggregation on 4 stratified PGC3 SCZ GWAS
summary statistics, then test cell-type enrichment where powered.

Strata:
  1. AFR  — PGC3 African American (VCF format: BETA/SE -> Z)
  2. LAT  — PGC3 Latino (VCF format: BETA/SE -> Z)
  3. EUR female — PGC3 European female (daner format: log(OR)/SE -> Z)
  4. EUR male   — PGC3 European male (daner format: log(OR)/SE -> Z)

Power gates (mandatory from brief):
  - AFR hard gate: if mean chi^2 < 1.0, descriptive stats only, no enrichment.
  - For any stratum: compute min detectable OR at k_overlap >= 3 with 80% power.
    If fewer GWS genes than needed for 80% power, classify UNINTERPRETABLE.

Reference pipeline: experiments/batch_026/run_batch026.py
"""

import os
import sys
import gzip
import json
import warnings
import bisect
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import fisher_exact, norm

warnings.filterwarnings("ignore")

# =============================================================================
# Paths
# =============================================================================
PROJ = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
DATA_DIR = PROJ / "data" / "19426775"
BATCH_DIR = PROJ / "experiments" / "batch_037"
OUTPUT_DIR = BATCH_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENCODE_BED = PROJ / "experiments" / "batch_026" / "gencode_genes.bed"
MARKERS_PATH = PROJ / "experiments" / "batch_009" / "data" / "markers.parquet"

# Strata definitions with file paths and format type
STRATA = {
    "afr": {
        "path": DATA_DIR / "PGC3_SCZ_wave3.afram.autosome.public.v3.vcf.tsv.gz",
        "format": "vcf",  # CHROM, ID, POS, A1, A2, FCAS, FCON, IMPINFO, BETA, SE, PVAL
        "label": "AFR (African American)",
    },
    "lat": {
        "path": DATA_DIR / "PGC3_SCZ_wave3.latino.autosome.public.v3.vcf.tsv.gz",
        "format": "vcf",
        "label": "LAT (Latino)",
    },
    "eur_female": {
        "path": DATA_DIR / "daner_PGC_SCZ_w3_75_0618a_eur_female.gz",
        "format": "daner",  # CHR, SNP, BP, A1, A2, FRQ_*, INFO, OR, SE, P, ...
        "label": "EUR Female",
    },
    "eur_male": {
        "path": DATA_DIR / "daner_PGC_SCZ_w3_75_0618a_eur_male.gz",
        "format": "daner",
        "label": "EUR Male",
    },
}

# Window size for SNP->gene mapping (same as batch_026)
WINDOW_SIZE = 50_000

# Minimum number of SNPs per gene for Stouffer aggregation
MIN_SNPS_PER_GENE = 2

# Cell types to test (from brief: Neurons=95, Oligodendrocytes=36, OPCs=10, Astrocytes=15)
CELL_TYPES_MAP = {
    "Neurons": "Neurons",
    "Oligodendrocytes": "Oligodendrocytes",
    "Oligodendrocyte progenitor cells": "OPCs",
    "Astrocytes": "Astrocytes",
}


# =============================================================================
# Utility functions
# =============================================================================

def load_gencode_genes(bed_path):
    """Load GENCODE gene annotations from cached BED file.

    The BED file was produced by batch_026 from GENCODE v44 GTF.
    Format: chrom, start, end, gene_name (tab-separated, no header).
    """
    print(f"  Loading GENCODE genes from: {bed_path}")
    df = pd.read_csv(bed_path, sep="\t", header=None,
                     names=["chrom", "start", "end", "gene"])
    # Normalize chromosome format (ensure "chr" prefix)
    df["chrom"] = df["chrom"].astype(str)
    print(f"  Loaded {len(df):,} gene annotations")
    return df


def load_celltype_markers(parquet_path):
    """Load cell-type markers from batch_009 PanglaoDB parquet.

    Returns dict mapping short name -> set of gene symbols.
    """
    print(f"  Loading cell-type markers from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    markers = {}
    for full_name, short_name in CELL_TYPES_MAP.items():
        genes = set(df.loc[df["cell_type"] == full_name, "gene"].dropna().astype(str))
        markers[short_name] = genes
        print(f"    {short_name}: {len(genes)} markers")
    return markers


def parse_vcf_sumstats(vcf_path):
    """Parse PGC3 VCF-format summary statistics (AFR, LAT).

    Columns: CHROM, ID, POS, A1, A2, FCAS, FCON, IMPINFO, BETA, SE, PVAL
    Filter: IMPINFO >= 0.9, SE > 0
    Z = BETA / SE

    Returns DataFrame with columns: CHR, SNP, BP, Z, P, IMPINFO
    """
    print(f"  Parsing VCF: {vcf_path.name}")
    records = []
    n_total = 0
    n_pass = 0
    n_skip_info = 0
    n_skip_se = 0

    with gzip.open(str(vcf_path), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            n_total += 1

            # Skip header row (in case it is not comment-prefixed)
            if parts[0] in ("CHROM", "#CHROM", "SNP"):
                continue
            if len(parts) < 11:
                continue

            try:
                impinfo = float(parts[7])
            except ValueError:
                continue

            if impinfo < 0.9:
                n_skip_info += 1
                continue

            try:
                beta_val = float(parts[8])
                se_val = float(parts[9])
                pval_val = float(parts[10])
            except (ValueError, IndexError):
                continue

            if se_val <= 0:
                n_skip_se += 1
                continue

            chrom = parts[0]
            # Normalize to chr prefix for matching GENCODE
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom

            records.append({
                "CHR": chrom,
                "SNP": parts[1],
                "BP": int(parts[2]),
                "Z": beta_val / se_val,
                "P": pval_val,
                "IMPINFO": impinfo,
            })
            n_pass += 1

            if n_total % 1_000_000 == 0:
                print(f"    Processed {n_total:,} rows, {n_pass:,} pass filters")

    df = pd.DataFrame(records)
    print(f"    Total: {n_total:,} | Pass: {n_pass:,} | "
          f"Skip INFO: {n_skip_info:,} | Skip SE<=0: {n_skip_se:,}")
    return df


def parse_daner_sumstats(daner_path):
    """Parse PGC3 daner-format summary statistics (EUR female/male).

    Columns: CHR, SNP, BP, A1, A2, FRQ_A_*, FRQ_U_*, INFO, OR, SE, P, ...
    Filter: INFO >= 0.9, SE > 0
    Z = log(OR) / SE

    Returns DataFrame with columns: CHR, SNP, BP, Z, P, INFO
    """
    print(f"  Parsing daner: {daner_path.name}")
    records = []
    n_total = 0
    n_pass = 0
    n_skip_info = 0
    n_skip_se = 0

    with gzip.open(str(daner_path), "rt") as f:
        header = None
        for line in f:
            parts = line.rstrip().split("\t")
            if header is None:
                header = parts
                # Map column names
                col_idx = {name: i for i, name in enumerate(header)}
                continue

            n_total += 1

            try:
                info_val = float(parts[col_idx["INFO"]])
            except (ValueError, KeyError):
                continue

            if info_val < 0.9:
                n_skip_info += 1
                continue

            try:
                or_val = float(parts[col_idx["OR"]])
                se_val = float(parts[col_idx["SE"]])
                pval_val = float(parts[col_idx["P"]])
            except (ValueError, KeyError):
                continue

            if se_val <= 0 or or_val <= 0:
                n_skip_se += 1
                continue

            chrom = parts[col_idx["CHR"]]
            # Normalize to chr prefix
            if not chrom.startswith("chr"):
                chrom = "chr" + chrom

            z_score = np.log(or_val) / se_val

            records.append({
                "CHR": chrom,
                "SNP": parts[col_idx["SNP"]],
                "BP": int(parts[col_idx["BP"]]),
                "Z": z_score,
                "P": pval_val,
                "INFO": info_val,
            })
            n_pass += 1

            if n_total % 1_000_000 == 0:
                print(f"    Processed {n_total:,} rows, {n_pass:,} pass filters")

    df = pd.DataFrame(records)
    print(f"    Total: {n_total:,} | Pass: {n_pass:,} | "
          f"Skip INFO: {n_skip_info:,} | Skip SE<=0: {n_skip_se:,}")
    return df


def map_snps_to_genes(df_snps, df_genes, window_size=50_000):
    """Map SNPs to genes using +/- window_size around gene midpoints.

    Uses binary search (bisect) for efficient interval overlap detection.

    Returns dict: gene_name -> list of Z-scores for SNPs in window.
    Also returns dict: gene_name -> number of SNPs (including genes with 1 SNP).
    """
    print(f"  Building per-chromosome SNP index...")
    # Group SNPs by chromosome, sorted by position
    chr_snps = {}
    for _, row in df_snps.iterrows():
        chrom = str(row["CHR"])
        if chrom not in chr_snps:
            chr_snps[chrom] = []
        chr_snps[chrom].append((int(row["BP"]), row["Z"], row["P"]))

    for chrom in chr_snps:
        chr_snps[chrom].sort(key=lambda x: x[0])

    # Extract sorted position arrays for binary search
    chr_pos = {}
    for chrom, snps in chr_snps.items():
        chr_pos[chrom] = [s[0] for s in snps]

    print(f"  Chromosomes with SNPs: {len(chr_snps)}")

    # Map genes to SNPs using binary search
    gene_snp_z = {}  # gene -> list of Z values
    gene_snp_count = {}  # gene -> total SNP count

    for _, row in df_genes.iterrows():
        chrom = str(row["chrom"])
        start = int(row["start"])
        end = int(row["end"])
        gene = str(row["gene"])

        # Gene midpoint
        midpoint = (start + end) // 2
        win_start = midpoint - window_size
        win_end = midpoint + window_size

        if chrom not in chr_snps:
            continue

        pos_list = chr_pos[chrom]
        snps = chr_snps[chrom]

        # Binary search for window boundaries
        lo = bisect.bisect_left(pos_list, win_start)
        hi = bisect.bisect_right(pos_list, win_end)
        n_in_window = hi - lo

        if n_in_window >= 1:
            z_vals = [snps[i][1] for i in range(lo, hi)]
            gene_snp_z[gene] = z_vals
            gene_snp_count[gene] = n_in_window

    n_with_snps = len(gene_snp_count)
    n_with_min = sum(1 for v in gene_snp_count.values() if v >= MIN_SNPS_PER_GENE)
    print(f"  Genes with >= 1 SNP in window: {n_with_snps:,}")
    print(f"  Genes with >= {MIN_SNPS_PER_GENE} SNPs in window: {n_with_min:,}")

    return gene_snp_z, gene_snp_count


def compute_gene_level(gene_snp_z):
    """Compute Stouffer's Z aggregation per gene.

    Stouffer's Z = sum(Z_i) / sqrt(k)
    Only includes genes with >= MIN_SNPS_PER_GENE SNPs.

    Returns DataFrame with gene-level statistics, sorted by Stouffer p-value.
    """
    results = []
    for gene, z_vals in gene_snp_z.items():
        k = len(z_vals)
        if k < MIN_SNPS_PER_GENE:
            continue

        z_arr = np.array(z_vals, dtype=np.float64)

        # Stouffer's Z: assumes SNP independence (conservative approximation)
        stouffer_z = z_arr.sum() / np.sqrt(k)
        stouffer_p = 2.0 * norm.cdf(-abs(stouffer_z))

        results.append({
            "gene": gene,
            "n_snps": k,
            "stouffer_z": stouffer_z,
            "stouffer_p": stouffer_p,
            "mean_z": float(z_arr.mean()),
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    df = df.sort_values("stouffer_p").reset_index(drop=True)
    n_total = len(df)

    # Bonferroni correction
    df["stouffer_p_bonf"] = (df["stouffer_p"] * n_total).clip(upper=1.0)

    # FDR (Benjamini-Hochberg)
    ranks = np.arange(1, n_total + 1)
    df["stouffer_p_fdr"] = (df["stouffer_p"] * n_total / ranks).clip(upper=1.0)

    return df


def compute_power_metrics(n_gws_genes, n_background_genes, markers):
    """Compute minimum detectable OR at k_overlap >= 3 with 80% power.

    Uses Fisher's exact power approximation. For a 2x2 contingency table:
      a = GWS genes in cell type
      b = cell-type genes NOT in GWS
      c = GWS genes NOT in cell type
      d = background genes in neither

    We need OR such that P(reject H0 | OR) >= 0.80 with observed counts
    constrained by k_overlap = a >= 3.

    For simplicity and conservatism, we compute power for each cell type
    using the normal approximation to log(OR):
      Under H1 with true OR = OR_true:
        E[log OR_hat] = log(OR_true)
        Var[log OR_hat] ~= 1/a + 1/b + 1/c + 1/d
      Power = P(|Z| > z_{alpha/2}) under H1

    We solve for the minimum OR_true such that power >= 0.80 with a >= 3.
    """
    power_info = {}
    for cell_type, cell_genes in markers.items():
        n_cell = len(cell_genes)

        # Expected a under various OR values
        # Under null: E[a] = n_gws * n_cell / n_background
        # We need a >= 3 (k_overlap threshold)
        if n_gws_genes < 3 or n_background_genes < n_cell + n_gws_genes:
            power_info[cell_type] = {
                "min_detectable_or": float("inf"),
                "power_at_or3": 0.0,
                "k_overlap_needed": 3,
                "status": "INSUFFICIENT_GWS",
            }
            continue

        # Solve for minimum OR with 80% power, a >= 3
        # Use grid search over OR values
        best_or = None
        for or_test in np.arange(1.5, 20.0, 0.1):
            # Expected a under this OR
            # Under H0: p_a = n_gws * n_cell / n_background
            # Under H1 with OR: p_a is inflated
            # Simplified: a = min(n_gws, n_cell, max(3, round(n_gws * n_cell / n_background * or_test)))
            a = max(3, round(n_gws_genes * n_cell / n_background_genes * or_test))
            a = min(a, n_gws_genes, n_cell)

            if a < 3:
                continue

            b = n_cell - a
            c = n_gws_genes - a
            d = n_background_genes - a - b - c

            if b <= 0 or c <= 0 or d <= 0:
                continue

            # Power computation (normal approximation)
            # log(OR_true) under H1
            log_or_true = np.log(or_test)
            se_log_or = np.sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d)

            # z_alpha/2 for Bonferroni-corrected alpha
            n_cell_types = len(markers)
            alpha_bonf = 0.05 / n_cell_types
            z_alpha = norm.ppf(1 - alpha_bonf / 2)

            # Non-centrality parameter
            ncp = log_or_true / se_log_or

            # Power = P(|Z| > z_alpha) under H1
            power = 1.0 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)

            if power >= 0.80:
                best_or = or_test
                break

        # Also compute power at OR=3.0 specifically (benchmark from brief)
        a3 = max(3, round(n_gws_genes * n_cell / n_background_genes * 3.0))
        a3 = min(a3, n_gws_genes, n_cell)
        b3 = n_cell - a3
        c3 = n_gws_genes - a3
        d3 = n_background_genes - a3 - b3 - c3
        if a3 >= 3 and b3 > 0 and c3 > 0 and d3 > 0:
            se3 = np.sqrt(1.0/a3 + 1.0/b3 + 1.0/c3 + 1.0/d3)
            ncp3 = np.log(3.0) / se3
            n_cell_types = len(markers)
            alpha_bonf = 0.05 / n_cell_types
            z_alpha = norm.ppf(1 - alpha_bonf / 2)
            power_at_or3 = 1.0 - norm.cdf(z_alpha - ncp3) + norm.cdf(-z_alpha - ncp3)
        else:
            power_at_or3 = 0.0

        power_info[cell_type] = {
            "min_detectable_or": round(best_or, 2) if best_or is not None else float("inf"),
            "power_at_or3": round(power_at_or3, 3),
            "k_overlap_needed": 3,
            "status": "ADEQUATE" if best_or is not None and best_or <= 5.0 else "LOW_POWER",
        }

    return power_info


def run_enrichment(gws_genes, background_genes, markers):
    """Run Fisher's exact test for cell-type enrichment.

    Parameters:
        gws_genes: set of genome-wide significant genes (Bonferroni p < 0.05)
        background_genes: set of all genes tested in gene-level analysis
        markers: dict mapping cell_type -> set of gene symbols

    Returns list of result dicts with OR, CI, p-values.
    """
    results = []
    n_tests = len(markers)

    for cell_type, cell_genes in markers.items():
        # Compute overlap
        overlap = gws_genes & cell_genes
        a = len(overlap)  # GWS genes in cell type
        b = len(cell_genes - gws_genes)  # cell-type genes NOT GWS
        c = len(gws_genes - cell_genes)  # GWS genes NOT in cell type
        d = len(background_genes - gws_genes - cell_genes)  # neither

        # Need all cells > 0 for Fisher's exact
        if a == 0 or b == 0 or c == 0 or d == 0:
            results.append({
                "cell_type": cell_type,
                "k_overlap": a,
                "n_cell_markers": len(cell_genes),
                "odds_ratio": None,
                "ci_lo": None,
                "ci_hi": None,
                "p_raw": None,
                "p_bonf": None,
                "note": f"Degenerate table (a={a}, b={b}, c={c}, d={d})",
            })
            continue

        table = [[a, b], [c, d]]
        odds_ratio, p_val = fisher_exact(table, alternative="two-sided")

        # Woolf 95% CI for log(OR)
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1.0/a + 1.0/b + 1.0/c + 1.0/d)
        ci_lo = np.exp(log_or - 1.96 * se_log_or)
        ci_hi = np.exp(log_or + 1.96 * se_log_or)

        p_bonf = min(p_val * n_tests, 1.0)

        results.append({
            "cell_type": cell_type,
            "k_overlap": a,
            "n_cell_markers": len(cell_genes),
            "odds_ratio": round(odds_ratio, 3),
            "ci_lo": round(ci_lo, 3),
            "ci_hi": round(ci_hi, 3),
            "p_raw": p_val,
            "p_bonf": round(p_bonf, 4),
        })

    return results


# =============================================================================
# Main analysis
# =============================================================================

def main():
    print("=" * 72)
    print("batch_037 — Ancestry + Sex Stratification Gene-Level Analysis")
    print("=" * 72)

    # Load shared resources
    print("\n## Loading shared resources")
    df_genes = load_gencode_genes(GENCODE_BED)
    markers = load_celltype_markers(MARKERS_PATH)

    # Results container
    all_results = {"strata": {}, "comparison_with_sldsc": {}}

    # Process each stratum
    for stratum_id, config in STRATA.items():
        print(f"\n{'=' * 72}")
        print(f"## Stratum: {config['label']} ({stratum_id})")
        print(f"{'=' * 72}")

        # Verify file exists
        if not config["path"].exists():
            print(f"  FILE NOT FOUND: {config['path']}")
            all_results["strata"][stratum_id] = {
                "error": f"File not found: {config['path']}",
                "power_status": "FILE_MISSING",
            }
            continue

        # Parse summary statistics
        if config["format"] == "vcf":
            df_snps = parse_vcf_sumstats(config["path"])
        else:
            df_snps = parse_daner_sumstats(config["path"])

        if len(df_snps) == 0:
            print(f"  WARNING: No SNPs passed filters for {stratum_id}")
            all_results["strata"][stratum_id] = {
                "mean_chi2": None,
                "n_snps": 0,
                "n_genes_tested": 0,
                "n_gws_genes": 0,
                "enrichment": None,
                "power_status": "NO_DATA",
            }
            continue

        # Descriptive statistics
        mean_chi2 = float((df_snps["Z"] ** 2).mean())
        n_snps = len(df_snps)
        print(f"\n  Mean chi^2: {mean_chi2:.4f}")
        print(f"  N SNPs (post-filter): {n_snps:,}")
        print(f"  Z range: [{df_snps['Z'].min():.2f}, {df_snps['Z'].max():.2f}]")

        # --- POWER GATE 1: AFR hard gate ---
        is_afr = (stratum_id == "afr")
        if is_afr and mean_chi2 < 1.0:
            print(f"\n  AFR HARD GATE: chi^2 = {mean_chi2:.4f} < 1.0, no enrichment analysis.")
            all_results["strata"][stratum_id] = {
                "mean_chi2": round(mean_chi2, 4),
                "n_snps": n_snps,
                "n_genes_tested": None,
                "n_gws_genes": 0,
                "min_detectable_or": None,
                "enrichment": None,
                "power_status": "UNINTERPRETABLE_AFR_HARD_GATE",
            }
            continue

        # SNP -> gene mapping
        gene_snp_z, gene_snp_count = map_snps_to_genes(df_snps, df_genes, WINDOW_SIZE)

        # Gene-level aggregation
        df_gene_level = compute_gene_level(gene_snp_z)

        if len(df_gene_level) == 0:
            print(f"  WARNING: No genes with >= {MIN_SNPS_PER_GENE} SNPs for {stratum_id}")
            all_results["strata"][stratum_id] = {
                "mean_chi2": round(mean_chi2, 4),
                "n_snps": n_snps,
                "n_genes_tested": 0,
                "n_gws_genes": 0,
                "enrichment": None,
                "power_status": "NO_GENE_RESULTS",
            }
            continue

        n_genes_tested = len(df_gene_level)
        n_gws = int((df_gene_level["stouffer_p_bonf"] < 0.05).sum())
        n_fdr = int((df_gene_level["stouffer_p_fdr"] < 0.05).sum())

        print(f"\n  Gene-level results:")
        print(f"    Genes tested (>= {MIN_SNPS_PER_GENE} SNPs): {n_genes_tested:,}")
        print(f"    Bonferroni-significant (p < 0.05): {n_gws}")
        print(f"    FDR-significant (q < 0.05): {n_fdr}")
        print(f"    Genome-wide threshold: {0.05 / n_genes_tested:.2e}")

        # Top genes
        print(f"\n  Top 15 genes by Stouffer p-value:")
        for _, row in df_gene_level.head(15).iterrows():
            flag = "***" if row["stouffer_p_bonf"] < 0.05 else (
                   "**" if row["stouffer_p_fdr"] < 0.05 else (
                   "*" if row["stouffer_p"] < 1e-4 else ""))
            print(f"    {row['gene']:25s} n={row['n_snps']:4d}  "
                  f"Z={row['stouffer_z']:7.2f}  P={row['stouffer_p']:.2e}  "
                  f"Bonf={row['stouffer_p_bonf']:.2e} {flag}")

        # Save gene-level results per stratum
        gene_out = OUTPUT_DIR / f"gene_level_{stratum_id}.tsv"
        df_gene_level.to_csv(gene_out, sep="\t", index=False)
        print(f"  Saved gene-level results: {gene_out}")

        # --- POWER GATE 2: Minimum detectable OR ---
        background_genes = set(df_gene_level["gene"])
        gws_genes = set(df_gene_level.loc[
            df_gene_level["stouffer_p_bonf"] < 0.05, "gene"
        ])

        power_info = compute_power_metrics(n_gws, n_genes_tested, markers)

        # Determine overall power status
        # If ANY key cell type (Neurons) has min_detectable_or > 5.0, UNINTERPRETABLE
        neuron_power = power_info.get("Neurons", {})
        if neuron_power.get("power_at_or3", 0) < 0.80:
            power_status = "UNINTERPRETABLE"
        else:
            power_status = "ADEQUATE"

        print(f"\n  Power analysis:")
        print(f"    GWS genes: {n_gws}")
        print(f"    Power status: {power_status}")
        for ct, pinfo in power_info.items():
            print(f"    {ct:20s}: min_OR={pinfo['min_detectable_or']}, "
                  f"power@OR3={pinfo['power_at_or3']:.3f}, "
                  f"status={pinfo['status']}")

        # --- Cell-type enrichment ---
        enrichment_results = None
        if n_gws >= 3:
            print(f"\n  Running cell-type enrichment (Fisher's exact)...")
            enrichment_results = run_enrichment(gws_genes, background_genes, markers)
            for er in enrichment_results:
                if er["odds_ratio"] is not None:
                    print(f"    {er['cell_type']:20s}: OR={er['odds_ratio']:6.2f} "
                          f"CI=[{er['ci_lo']:.2f},{er['ci_hi']:.2f}] "
                          f"P={er['p_raw']:.2e} Bonf={er['p_bonf']:.4f} "
                          f"k={er['k_overlap']}")
                else:
                    print(f"    {er['cell_type']:20s}: {er.get('note', 'skipped')}")
        else:
            print(f"\n  Skipping enrichment: only {n_gws} GWS genes (need >= 3)")

        # Store results
        all_results["strata"][stratum_id] = {
            "mean_chi2": round(mean_chi2, 4),
            "n_snps": n_snps,
            "n_genes_tested": n_genes_tested,
            "n_gws_genes": n_gws,
            "n_fdr_genes": n_fdr,
            "min_detectable_or_neuron": neuron_power.get("min_detectable_or"),
            "power_at_or3_neuron": neuron_power.get("power_at_or3"),
            "power_status": power_status,
            "power_detail": power_info,
            "enrichment": enrichment_results,
            "top_genes": df_gene_level.head(10)[
                ["gene", "n_snps", "stouffer_z", "stouffer_p", "stouffer_p_bonf"]
            ].to_dict(orient="records"),
        }

    # =========================================================================
    # Comparison with S-LDSC
    # =========================================================================
    print(f"\n{'=' * 72}")
    print("## Comparison with S-LDSC (F076, F082)")
    print(f"{'=' * 72}")

    all_results["comparison_with_sldsc"] = {
        "eur_male_gene_level_vs_F082": (
            "Gene-level results for EUR male are supplementary to S-LDSC F082 "
            "(male neuronal 1.78x, p=0.010). Gene-level Stouffer Z does not "
            "model LD structure, so S-LDSC is the authoritative method."
        ),
        "eur_female_gene_level_vs_F082": (
            "Gene-level results for EUR female are supplementary to S-LDSC F082 "
            "(female neuronal 1.70x, p=0.239). Gene-level is less powerful."
        ),
        "afr_lat_note": (
            "AFR and LAT gene-level results are NOT comparable to EUR S-LDSC "
            "due to different LD structure, sample sizes, and ancestries."
        ),
        "note": (
            "Gene-level is supplementary to S-LDSC (F076, F082). "
            "S-LDSC models LD and is more powerful. When gene-level and "
            "S-LDSC disagree, defer to S-LDSC."
        ),
    }

    # =========================================================================
    # Save results
    # =========================================================================
    results_path = BATCH_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved results: {results_path}")

    # =========================================================================
    # Summary table
    # =========================================================================
    print(f"\n{'=' * 72}")
    print("SUMMARY TABLE")
    print(f"{'=' * 72}")
    print(f"{'Stratum':<15} {'chi^2':>8} {'N_SNPs':>10} {'Genes':>8} "
          f"{'GWS':>5} {'FDR':>5} {'Power':>18} {'Neuron minOR':>12}")
    print("-" * 90)

    for stratum_id, sdata in all_results["strata"].items():
        if "error" in sdata:
            print(f"{stratum_id:<15} {'ERROR':>8} {'ERROR':>10} {'ERROR':>8} "
                  f"{'ERROR':>5} {'ERROR':>5} {'ERROR':>18} {'ERROR':>12}")
            continue

        chi2 = sdata.get("mean_chi2", "N/A")
        n_snps = sdata.get("n_snps", 0)
        n_genes = sdata.get("n_genes_tested", 0)
        n_gws = sdata.get("n_gws_genes", 0)
        n_fdr = sdata.get("n_fdr_genes", 0)
        power = sdata.get("power_status", "N/A")
        min_or = sdata.get("min_detectable_or_neuron", "N/A")

        chi2_str = f"{chi2:.4f}" if isinstance(chi2, (int, float)) else str(chi2)
        min_or_str = (f"{min_or:.1f}" if isinstance(min_or, (int, float))
                      else str(min_or))

        print(f"{stratum_id:<15} {chi2_str:>8} {n_snps:>10,} {n_genes:>8,} "
              f"{n_gws:>5} {n_fdr:>5} {power:>18} {min_or_str:>12}")

    print("-" * 90)
    print("\nNotes:")
    print("  - Gene-level uses Stouffer's Z aggregation (does NOT model LD)")
    print("  - S-LDSC (F076, F082) is the authoritative method for cell-type enrichment")
    print("  - AFR/LAT results are NOT directly comparable to EUR due to LD confound")
    print("  - Power status UNINTERPRETABLE: < 80% power to detect OR=3.0 neuronal enrichment")

    print(f"\nDONE")


if __name__ == "__main__":
    main()
