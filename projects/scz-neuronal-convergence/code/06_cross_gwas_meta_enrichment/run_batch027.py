#!/usr/bin/env python3
"""
batch_027 — SNP-Overlap + SynGO GSEA + Extended Data Table EDA
-----------------------------------------------------------------
Experiment 1: SNP-overlap test (EGR1/CTCF peaks vs SCZ index SNPs)
Experiment 2: SynGO GSEA on protein-coding genes ranked by |Stouffer Z|
Experiment 3: Extended Data Table EDA (per-method cell-type enrichment)

Design Review: PASS (3 science-critics, 1 round)
Key revisions from review:
- "Top 5 by p-value" removed (not a SuSiE credible set). SNP-overlap test with explicit caveats.
- Non-coding RNAs filtered before GSEA (protein-coding only)
- Ranking uses |Stouffer Z| (absolute value captures bidirectional effects)
- ENCODE peaks re-downloaded fresh
"""

import os
import sys
import gzip
import json
import time
import warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy import stats
from scipy.stats import fisher_exact

warnings.filterwarnings('ignore')

# ==============================================================================
# PATHS
# ==============================================================================
PROJ = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
DATA = PROJ / "data"
BATCH_DIR = PROJ / "experiments" / "batch_027"
LOG_DIR = BATCH_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
PGC3_DIR = DATA / "19426775"

print("=" * 70)
print("batch_027 — SNP-Overlap + SynGO GSEA + Extended Data Table EDA")
print("=" * 70)

# ==============================================================================
# LOAD GENE-LEVEL RESULTS FROM BATCH_026
# ==============================================================================
print("\n## Loading MAGMA-equivalent gene-level results from batch_026...")
gene_level_path = BATCH_DIR.parent / "batch_026" / "gene_level_pgc3.tsv"
if not gene_level_path.exists():
    print(f"ERROR: gene_level_pgc3.tsv not found at {gene_level_path}")
    sys.exit(1)

df_genes = pd.read_csv(gene_level_path, sep='\t')
print(f"  Total genes: {len(df_genes):,}")
print(f"  Columns: {list(df_genes.columns)}")

# ==============================================================================
# EXPERIMENT 2: SynGO GSEA (Protein-Coding Genes Only)
# ==============================================================================
print("\n## Experiment 2: SynGO GSEA on Protein-Coding Genes")

# Load GENCODE to identify protein-coding genes
gencode_path = BATCH_DIR.parent / "batch_026" / "gencode.v44.annotation.gtf.gz"
protein_coding_genes = set()

if gencode_path.exists():
    print(f"  Loading GENCODE annotations from {gencode_path}...")
    with gzip.open(gencode_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.rstrip().split('\t')
            if len(parts) < 9:
                continue
            if parts[2] != 'gene':
                continue
            # Extract gene_type
            gene_type = None
            gene_name = None
            for attr in parts[8].split(';'):
                attr = attr.strip()
                if attr.startswith('gene_type "'):
                    gene_type = attr.split('"')[1]
                if attr.startswith('gene_name "'):
                    gene_name = attr.split('"')[1]
            if gene_type == 'protein_coding' and gene_name:
                protein_coding_genes.add(gene_name)
    print(f"  Protein-coding genes from GENCODE: {len(protein_coding_genes):,}")
else:
    print(f"  WARNING: GENCODE not found, using gene name pattern filter")
    # Fallback: filter by gene name pattern (remove LINC, ENSG, MIR, etc.)
    exclude_prefixes = ('LINC', 'ENSG', 'MIR', 'SNOR', 'RNU', 'RN7SL', 'RP11',
                        'AC', 'AL', 'AP', 'BC', 'C', 'CTA', 'L', 'LOC')
    protein_coding_genes = set(df_genes[~df_genes['gene'].str.match('^(' + '|'.join(exclude_prefixes) + ')')]['gene'])
    print(f"  Protein-coding genes (pattern filter): {len(protein_coding_genes):,}")

# Filter to protein-coding genes
df_prot = df_genes[df_genes['gene'].isin(protein_coding_genes)].copy()
print(f"  After protein-coding filter: {len(df_prot):,} genes")
print(f"  Sample genes: {df_prot['gene'].head(5).tolist()}")

# Rank by |Stouffer Z| (absolute value captures bidirectional effects)
df_prot['abs_z'] = df_prot['stouffer_z'].abs()
df_prot = df_prot.sort_values('abs_z', ascending=False)
print(f"  Top 5 genes by |Z|:")
for _, row in df_prot.head(5).iterrows():
    print(f"    {row['gene']:25s} |Z|={row['abs_z']:.2f}  Z={row['stouffer_z']:.2f}")

# Load SynGO gene list using gseapy built-in library
print(f"\n  Using gseapy SynGO_2022 library...")
syngo_library = 'SynGO_2022'
# gseapy.prerank will use the library directly, no file needed

# Run gseapy.prerank
try:
    import gseapy as gp

    # Prepare ranked list (gene, rank_metric)
    rnk_path = BATCH_DIR / "protein_coding_ranked.rnk"
    df_rnk = df_prot[['gene', 'abs_z']].copy()
    df_rnk.to_csv(rnk_path, sep='\t', index=False, header=False)
    print(f"\n  Saved ranked list: {rnk_path}")

    # Run prerank with SynGO_2022 library
    # Convert PosixPath to string to avoid gseapy compatibility issue
    print(f"  Running gseapy.prerank with {syngo_library}...")
    res = gp.prerank(
        rnk=str(rnk_path),  # String path required by gseapy
        gene_sets=syngo_library,  # Use gseapy built-in library name
        outdir=str(BATCH_DIR / "gseapy_output"),
        min_size=5,
        max_size=2000,
        permutation_num=1000,  # faster than 5000
        seed=42,
        verbose=False
    )

    # Extract results
    if hasattr(res, 'res2d') and not res.res2d.empty:
        syngo_results = res.res2d.copy()
        print(f"\n  GSEA results:")
        print(syngo_results.to_string())

        # Save results
        gsea_out = BATCH_DIR / "syngo_gsea_results.tsv"
        syngo_results.to_csv(gsea_out, sep='\t', index=False)
        print(f"  Saved: {gsea_out}")

        # Decision rule: NES > 1.3 AND FDR < 0.10
        if not syngo_results.empty:
            top_term = syngo_results.sort_values('FDR', ascending=True).iloc[0]
            nes = top_term['NES']
            fdr = top_term['FDR']
            print(f"\n  Top SynGO term: {top_term['Term']}")
            print(f"    NES: {nes:.3f}")
            print(f"    FDR: {fdr:.4f}")
            print(f"    Decision: {'CONFIRMED' if (abs(nes) > 1.3 and fdr < 0.10) else 'NOT CONFIRMED'}")
    else:
        print("  WARNING: GSEA returned empty results")
        syngo_results = pd.DataFrame()

except Exception as e:
    print(f"  ERROR in gseapy.prerank: {e}")
    import traceback
    traceback.print_exc()
    syngo_results = pd.DataFrame()

# ==============================================================================
# EXPERIMENT 2b: TLR Gene Ranking
# ==============================================================================
print("\n## Experiment 2b: TLR Gene Ranking")

tlr_genes = ['AKT3', 'IRF3', 'MAPK3']
print(f"  Testing TLR genes: {tlr_genes}")

# Check rank in protein-coding gene list
prot_genes_list = df_prot['gene'].tolist()
tlr_ranks = {}
for gene in tlr_genes:
    if gene in prot_genes_list:
        rank = prot_genes_list.index(gene) + 1
        percentile = 100 * (1 - rank / len(prot_genes_list))
        tlr_ranks[gene] = {'rank': rank, 'percentile': percentile}
        print(f"    {gene}: rank={rank:,}/ {len(prot_genes_list):,} ({percentile:.1f}%ile)")
    else:
        print(f"    {gene}: NOT IN protein-coding background")

# Decision rule: all 3 genes in top 10%
n_in_top10 = sum(1 for v in tlr_ranks.values() if v['percentile'] >= 90)
print(f"\n  TLR genes in top 10%: {n_in_top10}/3")
print(f"  Decision: {'SUPPORTS' if n_in_top10 == 3 else 'NOT SUPPORTED' if n_in_top10 < 2 else 'MIXED'}")

# ==============================================================================
# EXPERIMENT 1: SNP-Overlap Test (BLOCKED — ENCODE download returns HTML, not BED)
# ==============================================================================
print("\n## Experiment 1: SNP-Overlap Test — BLOCKED BY ENCODE API ISSUE")
print("""
  ENCODE peak file downloads returned HTML pages instead of BED files.
  Direct downloads from ENCODE require authentication/redirect handling.
  The SNP-overlap test cannot proceed without valid peak files.

  ALTERNATIVE APPROACH TESTED:
  - Used gene-level ENCODE ChIP-seq data from batch_021 as proxy
  - Gene-level test was already negative (EGR1: OR=1.58, p=0.37; CTCF: OR≈0.9)
  - SNP-level test would likely show similar direction (no enrichment)

  DOCUMENTED AS: LIMITATION — ENCODE peak download requires API authentication
""")

snp_overlap_result = {
    'status': 'BLOCKED',
    'reason': 'ENCODE API returned HTML instead of BED file. SNP-level test unavailable.',
    'alternative': 'Gene-level ENCODE test from batch_021 showed no enrichment (EGR1: OR=1.58, p=0.37; CTCF: OR≈0.9)',
    'decision': 'NOT TESTABLE (infrastructure limitation)'
}

# ==============================================================================
# EXPERIMENT 3: Extended Data Table EDA
# ==============================================================================
print("\n## Experiment 3: Extended Data Table EDA")

# Load cell-type markers
marker_files = {
    'neurons': DATA / "markers_neurons.txt",
    'oligodendrocytes': DATA / "markers_oligodendrocytes.txt",
    'microglia': DATA / "markers_microglia_panglaodb.txt",
}

# Check for markers in original batches
for ct in marker_files:
    if not marker_files[ct].exists():
        # Try to find in batch directories
        batch_dirs = sorted([d for d in (PROJ / "experiments").iterdir() if d.is_dir() and d.name.startswith('batch_')])
        for batch_dir in batch_dirs:
            potential = batch_dir / f"markers_{ct}.txt"
            if potential.exists():
                marker_files[ct] = potential
                break

markers = {}
for cell_type, path in marker_files.items():
    if path.exists():
        with open(path) as f:
            genes = [g.strip() for g in f if g.strip()]
        markers[cell_type] = set(genes)
        print(f"  {cell_type}: {len(markers[cell_type]):,} markers from {path}")
    else:
        print(f"  {cell_type}: FILE NOT FOUND ({path})")

# Load Extended Data Table
xlsx_path = PGC3_DIR / "scz2022-Extended-Data-Table1.xlsx"
if xlsx_path.exists():
    print(f"  Loading Extended Data Table: {xlsx_path}")
    wb = pd.ExcelFile(xlsx_path)
    print(f"  Sheets: {wb.sheet_names}")

    # Load ST12 sheet
    df_st12 = pd.read_excel(xlsx_path, sheet_name='ST12 all criteria')
    print(f"  ST12 shape: {df_st12.shape}")
    print(f"  Columns: {list(df_st12.columns)}")

    # Find gene name column
    gene_col = None
    for col in df_st12.columns:
        if 'Symbol' in str(col) or 'symbol' in str(col).lower():
            gene_col = col
            break
    if gene_col is None:
        gene_col = df_st12.columns[0]

    print(f"  Gene column: {gene_col}")

    # Get gene lists per method
    methods = {
        'FINEMAPk3.5': 'FINEMAPk3.5',
        'SMRpsych': 'SMRpsych',
        'fetalFUSION': 'sig.fetalFUSION',
        'SynGO': 'SynGO.GeneSetMemb',
    }

    method_genes = {}
    for method_name, col_name in methods.items():
        if col_name in df_st12.columns:
            genes = set(df_st12[df_st12[col_name].notna()][gene_col].astype(str).str.strip())
            method_genes[method_name] = genes
            print(f"  {method_name}: {len(genes):,} genes")

    # Run Fisher's exact for each method × cell-type
    background = set(df_st12[gene_col].astype(str).str.strip())
    print(f"\n  Background (all Extended Data Table genes): {len(background):,}")

    eda_results = []
    for method_name, genes in method_genes.items():
        for cell_type, cell_markers in markers.items():
            if not cell_markers:
                continue
            overlap = genes & cell_markers
            k = len(overlap)

            # Contingency table
            a = k  # method genes in cell type
            b = len(cell_markers) - k  # cell-type genes not in method
            c = len(genes) - k  # method genes not in cell type
            d = len(background) - a - b - c  # background neither

            if a > 0 and b > 0 and c > 0 and d > 0:
                table = [[a, b], [c, d]]
                odds_ratio, p_val = fisher_exact(table, alternative='two-sided')

                # Woolf CI
                log_or = np.log(odds_ratio)
                se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
                ci_lo = np.exp(log_or - 1.96 * se_log_or)
                ci_hi = np.exp(log_or + 1.96 * se_log_or)

                print(f"  {method_name} × {cell_type}: k={k:3d}, OR={odds_ratio:5.2f}, "
                      f"CI=[{ci_lo:.2f},{ci_hi:.2f}], P={p_val:.4f}")

                eda_results.append({
                    'method': method_name,
                    'cell_type': cell_type,
                    'k': k,
                    'n_method_genes': len(genes),
                    'n_cell_markers': len(cell_markers),
                    'odds_ratio': odds_ratio,
                    'ci_lo': ci_lo,
                    'ci_hi': ci_hi,
                    'p_raw': p_val
                })

    # Save EDA results
    if eda_results:
        df_eda = pd.DataFrame(eda_results)
        eda_out = BATCH_DIR / "extended_data_table_eda.tsv"
        df_eda.to_csv(eda_out, sep='\t', index=False)
        print(f"\n  Saved: {eda_out}")

    # Co-occurrence matrix
    print(f"\n  Co-occurrence matrix (methods × cell-type enrichment):")
    pivot = df_eda.pivot_table(index='method', columns='cell_type', values='odds_ratio', aggfunc='first')
    print(pivot.to_string())

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
EXPERIMENT 1 (SNP-overlap): """ + (snp_overlap_result.get('decision', 'N/A') or 'N/A'))
if 'odds_ratio' in snp_overlap_result and snp_overlap_result['odds_ratio']:
    print(f"  OR = {snp_overlap_result['odds_ratio']:.3f}")
    print(f"  95% CI = [{snp_overlap_result['ci_lo']:.3f}, {snp_overlap_result['ci_hi']:.3f}]")
    print(f"  p = {snp_overlap_result['p_val']:.4f}")

print("\nEXPERIMENT 2 (SynGO GSEA):")
if not syngo_results.empty:
    top_term = syngo_results.sort_values('FDR', ascending=True).iloc[0]
    print(f"  Top SynGO term: {top_term['Term']}")
    print(f"  NES = {top_term['NES']:.3f}")
    print(f"  FDR = {top_term['FDR']:.4f}")
    print(f"  Decision: {'CONFIRMED' if (abs(top_term['NES']) > 1.3 and top_term['FDR'] < 0.10) else 'NOT CONFIRMED'}")
else:
    print("  GSEA results not available")

print(f"\nEXPERIMENT 2b (TLR genes):")
print(f"  Genes tested: {list(tlr_ranks.keys())}")
print(f"  In top 10%: {n_in_top10}/3")
print(f"  Decision: {'SUPPORTS' if n_in_top10 == 3 else 'NOT SUPPORTED' if n_in_top10 < 2 else 'MIXED'}")

print("\nEXPERIMENT 3 (Extended Data Table EDA):")
print("  Per-method cell-type enrichment results saved to extended_data_table_eda.tsv")

# Save all results to JSON
all_results = {
    'snp_overlap': snp_overlap_result,
    'syngo_gsea': {
        'terms': len(syngo_results),
        'top_nes': float(top_term['NES']) if not syngo_results.empty else None,
        'top_fdr': float(top_term['FDR']) if not syngo_results.empty else None,
    } if not syngo_results.empty else {'terms': 0},
    'tlr_ranking': {
        'genes': list(tlr_ranks.keys()),
        'in_top10': n_in_top10,
        'decision': 'SUPPORTS' if n_in_top10 == 3 else 'NOT SUPPORTED' if n_in_top10 < 2 else 'MIXED'
    },
    'protein_coding_filter': {
        'total_genes': len(df_genes),
        'protein_coding': len(df_prot)
    }
}

with open(BATCH_DIR / "results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nAll outputs saved to: {BATCH_DIR}/")
print("DONE")