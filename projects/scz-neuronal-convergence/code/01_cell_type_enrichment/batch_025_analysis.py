#!/usr/bin/env python3
"""
Batch 025: D11 Step 1 — PGC3 MAGMA Gene List Enrichment Analysis

Extract PGC3 prioritised gene list from Extended Data Table xlsx
(Trubetskoy et al. 2022 Nature Genetics) and run Fisher's exact test
enrichments against cell-type markers and immune pathways.

Compares results to prior enrichments with Pardiñas 2018 (batch_009/020)
and Ripke/PGC2 (batch_012/018) to assess replication.

Author: Marvin autonomous research agent
Project: SCZ Convergence Mapping
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# ============================================================
# CONFIGURATION
# ============================================================
BACKGROUND_SIZE = 20297  # Entrez protein-coding genes (same as all prior batches)
OUTPUT_DIR = "experiments/batch_025/results"
DATA_DIR = "experiments/batch_025/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# FUNCTIONS
# ============================================================

def fisher_enrichment(marker_genes, gwas_genes, universe_size,
                       alternative='greater', label='test'):
    """
    Run Fisher's exact test for gene set enrichment.

    Uses the hypergeometric formulation matching batch_009:
    - N = universe_size (total protein-coding genes in background)
    - M = len(gwas_genes) (SCZ GWAS genes = "successes in population")
    - n = len(marker_genes) (cell-type markers = "sample size")
    - x = overlap between markers and GWAS genes = "observed successes"

    Tests whether GWAS genes are enriched among cell-type marker genes
    (i.e., do cell-type markers contain more GWAS genes than expected by chance?).

    Contingency table:
      a = x (overlap: marker AND GWAS)
      b = n - x (marker but NOT GWAS)
      c = M - x (GWAS but NOT marker)
      d = N - n - c (neither)
    """
    N = universe_size
    M = len(gwas_genes)
    n = len(marker_genes)
    overlap = marker_genes & gwas_genes
    x = len(overlap)
    a = x
    b = n - x
    c = M - x
    d = N - n - c

    assert a >= 0 and b >= 0 and c >= 0 and d >= 0, \
        f"Negative counts! a={a}, b={b}, c={c}, d={d}"

    table = [[a, b], [c, d]]

    try:
        odds_ratio, p_value = fisher_exact(table, alternative=alternative)
    except Exception as e:
        print(f"  WARNING: Fisher's exact failed for {label}: {e}")
        return None

    # Woolf 95% CI (only valid when all cells > 0)
    if a > 0 and b > 0 and c > 0 and d > 0:
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lo = np.exp(log_or - 1.96 * se_log_or)
        ci_hi = np.exp(log_or + 1.96 * se_log_or)
    else:
        ci_lo = None
        ci_hi = None

    return {
        'label': label,
        'marker_n': n,
        'overlap': x,
        'gwas_n': M,
        'a': a, 'b': b, 'c': c, 'd': d,
        'odds_ratio': odds_ratio,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'p_value': p_value,
        'overlap_genes': sorted(overlap)
    }


def print_result(r, indent=2):
    """Pretty-print a Fisher's exact result."""
    if r is None:
        print(f"{' '*indent}{'FAILED'}")
        return
    ci_str = ""
    if r['ci_lo'] is not None:
        ci_str = f", 95% CI=[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
    sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
    print(f"{' '*indent}{r['label']}: "
          f"OR={r['odds_ratio']:.2f}{ci_str}, "
          f"p={r['p_value']:.3e}, k={r['overlap']}/{r['marker_n']}{sig}")


# ============================================================
# STEP 1: LOAD PGC3 GENE LIST FROM XLSX
# ============================================================
print("=" * 70)
print("BATCH 025: D11 Step 1 — PGC3 Gene List Enrichment Analysis")
print(f"Timestamp: {datetime.now().isoformat()}")
print("=" * 70)

xlsx_path = "data/19426775/scz2022-Extended-Data-Table1.xlsx"

df_ext = pd.read_excel(xlsx_path, sheet_name='Extended.Data.Table.1')
print(f"\nExtended Data Table 1: {len(df_ext)} rows, {df_ext['Symbol.ID'].nunique()} unique genes")
print(f"Gene biotypes: {df_ext['gene_biotype'].value_counts().to_dict()}")

# Filter protein-coding for primary analysis
pgc3_pc = set(df_ext[df_ext['gene_biotype'] == 'protein_coding']['Symbol.ID'].dropna().unique())
print(f"PGC3 protein-coding genes: {len(pgc3_pc)}")

# Load ST12 for comparison
df_st12 = pd.read_excel(xlsx_path, sheet_name='ST12 all criteria')
pgc3_prioritised = set(df_st12[df_st12['Prioritised'] == 1]['Symbol.ID'].dropna().unique())
pgc3_prioritised_pc = set(df_st12[(df_st12['Prioritised'] == 1) &
                                    (df_st12['gene_biotype'] == 'protein_coding')]['Symbol.ID'].dropna().unique())
pgc3_extended = set(df_st12[(df_st12['Extended.GWAS'] == 'YES') &
                              (df_st12['gene_biotype'] == 'protein_coding')]['Symbol.ID'].dropna().unique())
print(f"PGC3 prioritised (all biotypes): {len(pgc3_prioritised)}")
print(f"PGC3 prioritised (protein-coding): {len(pgc3_prioritised_pc)}")
print(f"PGC3 Extended.GWAS=YES (protein-coding): {len(pgc3_extended)}")

# Use protein-coding prioritised as primary gene list (matches Extended Data Table)
pgc3_genes = pgc3_pc
print(f"\n>>> Using PGC3 gene list: {len(pgc3_genes)} protein-coding prioritised genes")

pgc3_df = pd.DataFrame({'gene': sorted(pgc3_genes)})
pgc3_df.to_csv(f"{DATA_DIR}/pgc3_gene_list.csv", index=False)
print(f"Saved PGC3 gene list to {DATA_DIR}/pgc3_gene_list.csv")


# ============================================================
# STEP 2: LOAD CELL-TYPE MARKERS FROM BATCH_009
# ============================================================
print("\n" + "-" * 70)
print("STEP 2: Loading cell-type markers from batch_009")
print("-" * 70)

markers = pd.read_parquet("experiments/batch_009/data/markers.parquet")
print(f"Loaded {len(markers)} marker entries from batch_009")
print(f"Cell types: {markers['cell_type'].value_counts().to_dict()}")

cell_markers = {}
for ct, group in markers.groupby('cell_type'):
    cell_markers[ct] = set(group['gene'].dropna().unique())
    print(f"  {ct}: {len(cell_markers[ct])} marker genes")

# Load microglia markers from cached PanglaoDB (from batch_010)
# batch_010 used ALL 80 PanglaoDB microglia markers (including sensitivity_human=0)
# These were excluded from batch_009 due to sensitivity filter
try:
    panglao = pd.read_csv('/tmp/panglao_markers.tsv.gz', sep='\t', compression='gzip')
    microglia_df = panglao[panglao['cell type'] == 'Microglia']
    microglia_genes = set(microglia_df['official gene symbol'].dropna().str.upper().unique())
    cell_markers['Microglia (PanglaoDB)'] = microglia_genes
    print(f"  Microglia (PanglaoDB): {len(microglia_genes)} marker genes")
except Exception as e:
    print(f"  WARNING: Could not load microglia markers from PanglaoDB: {e}")
    microglia_genes = set()


# ============================================================
# STEP 3: LOAD PATHWAY GENE SETS
# ============================================================
print("\n" + "-" * 70)
print("STEP 3: Loading pathway gene sets")
print("-" * 70)

import gseapy as gp
import decoupler as dc

# NF-κB from Reactome (MSigDB 2023)
reactome = gp.Msigdb().get_gmt(category='c2.cp.reactome', dbver='2023.1.Hs')
nfkb_keys = [k for k in reactome.keys() if 'NF_KAPPA' in k.upper() or 'NFKAPPAB' in k.upper()]
nfkb_genes = set()
for k in nfkb_keys:
    nfkb_genes.update(reactome[k])
print(f"NF-κB (Reactome, combined {len(nfkb_keys)} pathways): {len(nfkb_genes)} genes")

# TLR from KEGG (MSigDB 2023)
kegg = gp.Msigdb().get_gmt(category='c2.cp.kegg', dbver='2023.1.Hs')
tlr_key = 'KEGG_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY'
tlr_genes = set(kegg[tlr_key])
print(f"TLR (KEGG): {len(tlr_genes)} genes")

# SPI1 regulon from DoRothEA (decoupler)
dor = dc.op.dorothea(organism='human')
spi1_all = dor[dor['source'] == 'SPI1']
spi1_genes = set(spi1_all['target'].unique())
print(f"SPI1 regulon (DoRothEA): {len(spi1_genes)} genes")

rela_targets = set(dor[dor['source'] == 'RELA']['target'].unique())
nfkb1_targets = set(dor[dor['source'] == 'NFKB1']['target'].unique())
print(f"RELA DoRothEA targets: {len(rela_targets)}")
print(f"NFKB1 DoRothEA targets: {len(nfkb1_targets)}")

# ============================================================
# STEP 4: RUN FISHER'S EXACT TESTS
# ============================================================
print("\n" + "-" * 70)
print("STEP 4: Running Fisher's Exact Enrichment Tests")
print("-" * 70)
print(f"Background universe: {BACKGROUND_SIZE} protein-coding genes (Entrez)")
print(f"PGC3 gene list: {len(pgc3_genes)} protein-coding genes")

results = []

# --- Cell-type enrichment ---
print("\n=== CELL-TYPE ENRICHMENT ===")
cell_types_to_test = ['Neurons', 'Oligodendrocytes', 'Astrocytes', 'Oligodendrocyte progenitor cells', 'Microglia (PanglaoDB)']
for ct in cell_types_to_test:
    if ct not in cell_markers:
        print(f"  Skipping {ct}: not in markers")
        continue
    r = fisher_enrichment(
        marker_genes=cell_markers[ct],
        gwas_genes=pgc3_genes,
        universe_size=BACKGROUND_SIZE,
        label=f"PGC3 × {ct}"
    )
    if r:
        print_result(r)
        results.append(r)


# --- Pathway enrichment ---
print("\n=== PATHWAY ENRICHMENT ===")
pathway_tests = [
    ('NF-κB (Reactome, combined)', nfkb_genes),
    ('NF-κB TF RELA (DoRothEA)', rela_targets),
    ('NF-κB TF NFKB1 (DoRothEA)', nfkb1_targets),
    ('TLR (KEGG)', tlr_genes),
    ('SPI1 regulon (DoRothEA)', spi1_genes),
]

for label, gene_set in pathway_tests:
    r = fisher_enrichment(
        marker_genes=gene_set,
        gwas_genes=pgc3_genes,
        universe_size=BACKGROUND_SIZE,
        label=f"PGC3 × {label}"
    )
    if r:
        print_result(r)
        results.append(r)


# ============================================================
# STEP 5: CELL-TYPE FDR CORRECTION
# ============================================================
print("\n" + "-" * 70)
print("STEP 5: FDR Correction for Cell-Type Tests")
print("-" * 70)

# Find cell-type results
cell_labels = ['Neurons', 'Oligodendrocytes', 'Astrocytes', 'Oligodendrocyte progenitor cells', 'Microglia (PanglaoDB)']
cell_results = [r for r in results
                if r and any(l in r['label'] for l in cell_labels)]

if len(cell_results) >= 2:
    pvals = np.array([r['p_value'] for r in cell_results])
    reject, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    for r, p_fdr in zip(cell_results, pvals_fdr):
        r['fdr_corrected'] = p_fdr
        sig = "***" if p_fdr < 0.001 else "**" if p_fdr < 0.01 else "*" if p_fdr < 0.05 else ""
        print(f"  {r['label']}: FDR={p_fdr:.3e} {sig}")
else:
    print("  Fewer than 2 cell-type results — skipping FDR correction")


# ============================================================
# STEP 6: COMPARE WITH PGC3 Extended.GWAS Gene Set
# ============================================================
print("\n" + "-" * 70)
print("STEP 6: PGC3 Extended.GWAS Analysis")
print("-" * 70)

# The Extended.GWAS gene set (641 genes) is very large — this is the full MAGMA gene set
# from PGC3, not just the prioritised ones. Test the same enrichments.
print(f"PGC3 Extended.GWAS (protein-coding): {len(pgc3_extended)} genes")
print("(Note: very large gene set — results expected to be attenuated)")

for label, gene_set in pathway_tests:
    r = fisher_enrichment(
        marker_genes=gene_set,
        gwas_genes=pgc3_extended,
        universe_size=BACKGROUND_SIZE,
        label=f"PGC3.ExtGWAS × {label}"
    )
    if r:
        print_result(r)
        results.append(r)


# ============================================================
# STEP 7: SAVE RESULTS
# ============================================================
print("\n" + "-" * 70)
print("STEP 7: Saving Results")
print("-" * 70)

results_json = {}
for r in results:
    if r:
        key = r['label']
        r_copy = {k: v for k, v in r.items() if k != 'label'}  # copy without label
        results_json[key] = r_copy

output = {
    'batch_id': 'batch_025',
    'timestamp': datetime.now().isoformat(),
    'pgc3_gene_list_n': len(pgc3_genes),
    'pgc3_extended_n': len(pgc3_extended),
    'background_size': BACKGROUND_SIZE,
    'cell_type_markers': {k: len(v) for k, v in cell_markers.items()},
    'pathway_gene_set_sizes': {
        'NF-kB (Reactome)': len(nfkb_genes),
        'NF-kB TF RELA (DoRothEA)': len(rela_targets),
        'NF-kB TF NFKB1 (DoRothEA)': len(nfkb1_targets),
        'TLR (KEGG)': len(tlr_genes),
        'SPI1 regulon (DoRothEA)': len(spi1_genes),
    },
    'results': results_json
}

with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"Results saved to {OUTPUT_DIR}/results.json")


# ============================================================
# STEP 8: SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: PGC3 Enrichment Results")
print("=" * 70)
print(f"{'Test':<45} {'OR':>8} {'95% CI':>18} {'p-value':>12} {'k':>6}")
print("-" * 85)
for r in sorted(results, key=lambda x: x['p_value'] if x else 1):
    if r:
        ci_str = ""
        if r['ci_lo'] is not None:
            ci_str = f"[{r['ci_lo']:.2f}-{r['ci_hi']:.2f}]"
        sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
        print(f"{r['label']:<45} {r['odds_ratio']:>8.2f} {ci_str:>18} {r['p_value']:>12.2e} {r['overlap']:>5}/{r['marker_n']}{sig}")

print("\n" + "=" * 70)
print("PGC3 Replication Assessment:")
print("=" * 70)

neurons_result = next((r for r in results if r and 'Neurons' in r['label'] and '× N' in r['label'] and 'PGC3.ExtGWAS' not in r['label']), None)
microglia_result = next((r for r in results if r and 'Microglia' in r['label'] and '× M' in r['label'] and 'PGC3.ExtGWAS' not in r['label']), None)
oligo_result = next((r for r in results if r and 'Oligodendrocytes' in r['label'] and '× O' in r['label'] and 'PGC3.ExtGWAS' not in r['label']), None)
nfkb_result = next((r for r in results if r and 'NF-κB' in r['label'] and 'Reactome' in r['label'] and '× N' in r['label'] and 'PGC3.ExtGWAS' not in r['label']), None)
tlr_result = next((r for r in results if r and 'TLR' in r['label'] and '× T' in r['label'] and 'PGC3.ExtGWAS' not in r['label']), None)

if neurons_result:
    print(f"\nNeuronal enrichment (PGC3, {neurons_result['overlap']}/{neurons_result['marker_n']}):")
    print(f"  OR = {neurons_result['odds_ratio']:.2f}, p = {neurons_result['p_value']:.2e}")
    if neurons_result['p_value'] < 0.05:
        print(f"  STATUS: >>> NEURONAL REPLICATION SUCCESSFUL (3rd independent GWAS) <<<")
    else:
        print(f"  STATUS: Marginal / not significant with PGC3")

if microglia_result:
    print(f"\nMicroglial enrichment (PGC3, {microglia_result['overlap']}/{microglia_result['marker_n']}):")
    print(f"  OR = {microglia_result['odds_ratio']:.2f}, p = {microglia_result['p_value']:.2e}")
    if microglia_result['p_value'] > 0.05:
        print(f"  STATUS: >>> MICROGLIA NEGATIVE CONFIRMED with PGC3 <<<")
    else:
        print(f"  STATUS: WARNING — microglia significant with PGC3")

if oligo_result:
    print(f"\nOligodendrocyte enrichment (PGC3, {oligo_result['overlap']}/{oligo_result['marker_n']}):")
    print(f"  OR = {oligo_result['odds_ratio']:.2f}, p = {oligo_result['p_value']:.2e}")

if nfkb_result:
    print(f"\nNF-κB enrichment (PGC3, {nfkb_result['overlap']}/{nfkb_result['marker_n']}):")
    print(f"  OR = {nfkb_result['odds_ratio']:.2f}, p = {nfkb_result['p_value']:.2e}")

if tlr_result:
    print(f"\nTLR enrichment (PGC3, {tlr_result['overlap']}/{tlr_result['marker_n']}):")
    print(f"  OR = {tlr_result['odds_ratio']:.2f}, p = {tlr_result['p_value']:.2e}")

print("\n" + "=" * 70)
print("Analysis complete.")
print("=" * 70)
