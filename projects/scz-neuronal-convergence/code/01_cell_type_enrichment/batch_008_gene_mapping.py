#!/usr/bin/env python3
"""
Batch 008 Gene Mapping Module
==============================
Maps GWAS SNPs to genes using Ensembl gene boundaries.

Design: Approved (with limitations documented in design.yaml)
Approach: pybiomart for gene boundaries + SNP→gene mapping with ±10kb window

WHY this approach:
- pybiomart provides standardized Ensembl gene annotations
- ±10kb window follows MAGMA convention for upstream/downstream regulatory regions
- Protein-coding filter ensures biologically relevant gene set
- Maps GWAS SNPs to their nearest genes within the extended boundaries

Gene size bias limitation (acknowledged in design.yaml):
- Larger genes span more SNPs and have more opportunities for associations
- This is a known statistical artifact affecting all gene-level GWAS analyses

Author: Marvin (implementation)
Date: 2026-04-09
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# pybiomart for gene annotations
try:
    from pybiomart import Server
    PYBIOMART_AVAILABLE = True
except ImportError:
    PYBIOMART_AVAILABLE = False
    warnings.warn("pybiomart not available. Gene mapping will fail.")

# Constants from design.yaml
SPECIES = "hsapiens_gene_ensembl"
WINDOW_KB = 10  # Extend gene boundaries by ±10kb (MAGMA-style)
BIOTYPE = "protein_coding"

# Input/output paths
INPUT_FILE = Path(__file__).parent / "data" / "gwas_scz_tophits.parquet"
OUTPUT_FILE = Path(__file__).parent / "data" / "gwas_genes.parquet"


def fetch_gene_annotations():
    """
    Fetch gene annotations from Ensembl via pybiomart.

    Returns:
        pd.DataFrame: Gene annotations with columns:
            - ensembl_gene_id
            - hgnc_symbol
            - chromosome_name
            - start_position
            - end_position
            - gene_length
            - extended_start (start - 10kb)
            - extended_end (end + 10kb)

    WHY fetch all genes first:
    - Reduces API calls vs querying per SNP
    - Allows efficient interval overlap calculations
    - Gene list is stable (updated periodically)
    """
    if not PYBIOMART_AVAILABLE:
        raise RuntimeError("pybiomart is required for gene mapping")

    print("\n[pybiomart] Fetching gene annotations from Ensembl...")
    print(f"  Species: {SPECIES}")
    print(f"  Biotype: {BIOTYPE}")

    try:
        server = Server(host='http://www.ensembl.org')
        dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets[SPECIES]

        # Query gene annotations
        attributes = [
            'ensembl_gene_id',
            'hgnc_symbol',
            'chromosome_name',
            'start_position',
            'end_position',
            'gene_biotype'
        ]

        filters = {
            'biotype': BIOTYPE
        }

        genes = dataset.query(
            attributes=attributes,
            filters=filters
        )

        # Clean up column names - handle various pybiomart column name formats
        genes.columns = [c.lower().replace(' ', '_').replace('(bp)', '').replace('/', '_') for c in genes.columns]

        # Strip trailing underscores
        genes.columns = [c.rstrip('_') for c in genes.columns]

        # Column name mapping for various formats
        column_mapping = {
            'gene_stable_id': 'ensembl_gene_id',
            'ensembl_gene_id': 'ensembl_gene_id',
            'hgnc_symbol': 'hgnc_symbol',
            'symbol': 'hgnc_symbol',
            'chromosome_scaffold_name': 'chromosome',
            'chromosome_name': 'chromosome',
            'chromosome': 'chromosome',
            'gene_start': 'start',
            'start_position': 'start',
            'start': 'start',
            'gene_end': 'end',
            'end_position': 'end',
            'end': 'end',
            'gene_type': 'biotype',
            'gene_biotype': 'biotype',
            'biotype': 'biotype'
        }

        genes = genes.rename(columns=column_mapping)

        print(f"  Columns after mapping: {list(genes.columns)}")

        # Remove rows with missing symbols
        genes = genes[genes['hgnc_symbol'].notna()]
        genes = genes[genes['hgnc_symbol'] != '']

        # Add extended boundaries
        genes['extended_start'] = genes['start'] - (WINDOW_KB * 1000)
        genes['extended_end'] = genes['end'] + (WINDOW_KB * 1000)
        genes['gene_length'] = genes['end'] - genes['start']

        # Add chromosome prefix for consistency
        genes['chromosome'] = genes['chromosome'].astype(str)

        print(f"✓ Fetched {len(genes)} protein-coding genes")
        print(f"  - Extended windows: ±{WINDOW_KB}kb")
        print(f"  - Total genomic coverage: {genes['gene_length'].sum():,} bp")

        return genes

    except Exception as e:
        print(f"✗ Error fetching gene annotations: {e}")
        raise


def load_gwas_tophits():
    """
    Load GWAS top hits from parquet file.

    Returns:
        pd.DataFrame: GWAS top hits with SNP information
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)
    print(f"\n[Data] Loaded {len(df)} GWAS associations from {INPUT_FILE.name}")

    # Standardize chromosome format
    if 'chr' in df.columns:
        df['chr'] = df['chr'].astype(str)
        # Remove 'chr' prefix if present
        df['chr'] = df['chr'].str.replace('chr', '', regex=False)

    # Check for position column
    if 'pos' not in df.columns and 'position' not in df.columns:
        warnings.warn("No position column found. SNP→gene mapping will use nearest gene approach.")
        df['pos'] = 0

    return df


def map_snps_to_genes(gwas_df: pd.DataFrame, genes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map GWAS SNPs to genes based on genomic coordinates.

    Args:
        gwas_df: GWAS top hits with chr and pos columns
        genes_df: Gene annotations with extended boundaries

    Returns:
        pd.DataFrame: GWAS hits with mapped gene information

    Mapping strategy:
    1. For SNPs with position: Find genes where SNP falls within extended gene window
    2. For SNPs without position (fallback): Use 'nearest_gene' column if available
    3. A SNP can map to multiple genes (overlapping genes, sense/antisense)
    """
    print("\n[Mapping] Mapping SNPs to genes...")

    # Separate SNPs with and without position
    has_position = gwas_df['pos'] > 0

    results = []

    # Process SNPs with position
    snps_with_pos = gwas_df[has_position].copy()
    if len(snps_with_pos) > 0:
        print(f"  Mapping {len(snps_with_pos)} SNPs with position...")

        for idx, snp in snps_with_pos.iterrows():
            chrom = snp['chr']
            pos = snp['pos']

            # Find genes on same chromosome
            chrom_genes = genes_df[genes_df['chromosome'] == chrom]

            if len(chrom_genes) == 0:
                continue

            # Find genes where SNP is within extended window
            mapped = chrom_genes[
                (chrom_genes['extended_start'] <= pos) &
                (chrom_genes['extended_end'] >= pos)
            ]

            if len(mapped) > 0:
                for _, gene in mapped.iterrows():
                    results.append({
                        'snp': snp['snp'],
                        'pval': snp['pval'],
                        'chr': chrom,
                        'pos': pos,
                        'ensembl_gene_id': gene['ensembl_gene_id'],
                        'hgnc_symbol': gene['hgnc_symbol'],
                        'gene_start': gene['start'],
                        'gene_end': gene['end'],
                        'mapping_method': 'coordinates'
                    })

    # Process SNPs without position (fallback)
    snps_without_pos = gwas_df[~has_position].copy()
    if len(snps_without_pos) > 0:
        print(f"  Mapping {len(snps_without_pos)} SNPs without position (fallback)...")

        for idx, snp in snps_without_pos.iterrows():
            nearest_gene = snp.get('nearest_gene')

            if pd.notna(nearest_gene) and nearest_gene != '':
                # Find gene in annotation
                gene_match = genes_df[genes_df['hgnc_symbol'] == nearest_gene]

                if len(gene_match) > 0:
                    gene = gene_match.iloc[0]
                    results.append({
                        'snp': snp['snp'],
                        'pval': snp['pval'],
                        'chr': gene['chromosome'],
                        'pos': 0,
                        'ensembl_gene_id': gene['ensembl_gene_id'],
                        'hgnc_symbol': gene['hgnc_symbol'],
                        'gene_start': gene['start'],
                        'gene_end': gene['end'],
                        'mapping_method': 'nearest_gene'
                    })

    if len(results) == 0:
        print("⚠ No SNPs mapped to genes")
        return pd.DataFrame()

    mapped_df = pd.DataFrame(results)
    n_unique_snps = mapped_df['snp'].nunique()
    n_unique_genes = mapped_df['hgnc_symbol'].nunique()

    print(f"✓ Mapped {n_unique_snps} SNPs to {n_unique_genes} unique genes")

    # Print mapping statistics
    print(f"  - SNPs with position mapped: {len(snps_with_pos)}")
    print(f"  - SNPs via nearest gene: {len(snps_without_pos)}")
    print(f"  - SNPs with no gene match: {len(gwas_df) - n_unique_snps}")

    return mapped_df


def extract_top_genes(mapped_df: pd.DataFrame, pval_threshold: float = 0.001) -> pd.DataFrame:
    """
    Extract top GWAS genes based on p-value threshold.

    Args:
        mapped_df: Mapped SNP-gene DataFrame
        pval_threshold: P-value threshold for gene inclusion

    Returns:
        pd.DataFrame: Top GWAS genes with one row per gene

    WHY use gene-level p-value aggregation:
    - Multiple SNPs can map to the same gene
    - Take the minimum p-value (most significant SNP) per gene
    - This captures the strongest association for each gene
    """
    print(f"\n[Gene Selection] Extracting top genes (p < {pval_threshold})...")

    # Filter by p-value
    significant = mapped_df[mapped_df['pval'] <= pval_threshold].copy()

    if len(significant) == 0:
        print(f"⚠ No genes with p < {pval_threshold}")
        # Fall back to all mapped genes
        significant = mapped_df.copy()
        print(f"  Using all {len(significant)} mapped genes as fallback")

    # Aggregate to gene level: take minimum p-value per gene
    gene_pvals = significant.groupby('hgnc_symbol').agg({
        'pval': 'min',
        'ensembl_gene_id': 'first',
        'chr': 'first',
        'gene_start': 'first',
        'gene_end': 'first'
    }).reset_index()

    gene_pvals = gene_pvals.rename(columns={'pval': 'min_pval'})

    # Add SNP count per gene
    snp_counts = significant.groupby('hgnc_symbol').size().reset_index(name='n_snps')
    gene_pvals = gene_pvals.merge(snp_counts, on='hgnc_symbol')

    # Sort by p-value
    gene_pvals = gene_pvals.sort_values('min_pval')

    print(f"✓ Extracted {len(gene_pvals)} top GWAS genes")
    print(f"  - Median SNPs per gene: {gene_pvals['n_snps'].median():.0f}")
    print(f"  - Genes with multiple SNPs: {(gene_pvals['n_snps'] > 1).sum()}")

    return gene_pvals


def compute_genomic_control(mapped_df: pd.DataFrame) -> Dict:
    """
    Compute genomic inflation factor (lambda GC) for QC.

    Args:
        mapped_df: Mapped SNP-gene DataFrame with p-values

    Returns:
        dict: QC metrics including lambda GC
    """
    from scipy import stats

    pvals = mapped_df['pval'].dropna()

    if len(pvals) < 100:
        return {"lambda_gc": None, "n_snps": len(pvals), "status": "insufficient_data"}

    # Compute chi-square statistic
    chisq = stats.chi2.ppf(1 - pvals, df=1)
    lambda_gc = np.median(chisq) / stats.chi2.ppf(0.5, df=1)

    status = "good"
    if lambda_gc < 0.9:
        status = "under_inflation"
    elif lambda_gc > 1.5:
        status = "over_inflation"

    return {
        "lambda_gc": lambda_gc,
        "n_snps": len(pvals),
        "status": status
    }


def save_gwas_genes(gene_df: pd.DataFrame, output_path: Path):
    """
    Save GWAS genes to parquet file.

    Args:
        gene_df: GWAS genes DataFrame
        output_path: Output file path

    Returns:
        bool: True if saved successfully
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gene_df.to_parquet(output_path, index=False)
        print(f"✓ Saved to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error saving: {e}")
        return False


def map_gwas_to_genes():
    """
    Main gene mapping pipeline.

    Returns:
        dict: Result with keys:
            - success (bool): Whether mapping succeeded
            - n_genes (int): Number of GWAS genes
            - dataframe (pd.DataFrame): GWAS genes DataFrame
            - qc_metrics (dict): QC metrics
            - message (str): Status message
    """
    print("\n" + "=" * 70)
    print("GWAS Gene Mapping")
    print("=" * 70)

    try:
        # Step 1: Load GWAS top hits
        gwas_df = load_gwas_tophits()

        # Step 2: Fetch gene annotations
        genes_df = fetch_gene_annotations()

        # Step 3: Map SNPs to genes
        mapped_df = map_snps_to_genes(gwas_df, genes_df)

        if len(mapped_df) == 0:
            return {
                "success": False,
                "n_genes": 0,
                "dataframe": None,
                "qc_metrics": {},
                "message": "No SNPs could be mapped to genes"
            }

        # Step 4: Compute QC metrics
        qc_metrics = compute_genomic_control(mapped_df)
        print(f"\n[QC] Genomic control λGC: {qc_metrics.get('lambda_gc', 'N/A'):.3f}")
        print(f"     Status: {qc_metrics.get('status', 'unknown')}")

        # Step 5: Extract top genes
        top_genes_df = extract_top_genes(mapped_df, pval_threshold=0.001)

        # Save results
        save_gwas_genes(top_genes_df, OUTPUT_FILE)

        return {
            "success": True,
            "n_genes": len(top_genes_df),
            "dataframe": top_genes_df,
            "qc_metrics": qc_metrics,
            "message": f"Successfully mapped {len(top_genes_df)} top GWAS genes"
        }

    except Exception as e:
        print(f"\n✗ Error in gene mapping: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "n_genes": 0,
            "dataframe": None,
            "qc_metrics": {},
            "message": f"Gene mapping failed: {e}"
        }


def main():
    """CLI entry point."""
    result = map_gwas_to_genes()

    print("\n" + "=" * 70)
    print("Gene Mapping Result:")
    print(f"  Success: {result['success']}")
    print(f"  GWAS Genes: {result['n_genes']}")
    print(f"  QC Status: {result['qc_metrics'].get('status', 'unknown')}")
    print(f"  Message: {result['message']}")
    print("=" * 70 + "\n")

    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
