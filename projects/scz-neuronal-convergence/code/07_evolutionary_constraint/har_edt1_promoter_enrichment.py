#!/usr/bin/env python3
"""
batch_049 Sub-B: HAR × EDT1/SynGO_EDT1 Promoter Direct Test

Test whether Human Accelerated Regions (HARs) overlap with SCZ gene set
promoters directly (±50kb from TSS). F125 tested HAR × PWM target genes
(null). This is the direct EDT1/SynGO_EDT1 promoter test (D048_02).

Gene sets:
  - SynGO_EDT1 (n=14): EDT1 × SynGO intersection — the constrained anchor
  - EDT1 (n=470): All PGC3 MAGMA significant protein-coding genes
  - SCHEMA (n=10): Exome-wide significant rare variant genes
  - EDT1_constrained_tier: SynGO_EDT1 + glutamate receptor + TF regulators

Data:
  - HAR BED: data/item_15/reference_assets/harsRichard2020.GRCh37.bed (3,129 HARs)
  - Gene TSS: data/ldsc/gene_tss_grch37.csv
  - PGC3 EDT1: data/19426775/scz2022-Extended-Data-Table1.xlsx (ST12, protein-coding)

Output:
  - experiments/batch_049/output/har_edt1_promoter_results.json
"""

from __future__ import annotations
import datetime as _dt
import json
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ------------------------------------------------------------------------------ Paths
PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_049"
OUTPUT_DIR = BATCH_DIR / "output"
LOG_DIR = BATCH_DIR / "logs"

HAR_BED = PROJECT_ROOT / "data" / "item_15" / "reference_assets" / "harsRichard2020.GRCh37.bed"
TSS_CSV = PROJECT_ROOT / "data" / "ldsc" / "gene_tss_grch37.csv"
PGC3_XLSX = PROJECT_ROOT / "data" / "19426775" / "scz2022-Extended-Data-Table1.xlsx"

# ------------------------------------------------------------------------------ Gene Lists (pre-registered from batch_047/048)
# SynGO_EDT1: EDT1 × SynGO intersection (F121: pLI OR=26.44)
SYNOGO_EDT1 = [
    "DLGAP1", "GRIN2A", "NRXN1", "CNTNAP2", "ARC", "DLG4",
    "NRXN2", "NLGN1", "NLGN2", "SHANK1", "SHANK3", "HOMER1", "SYN1", "GAP43"
]

# SCHEMA genes: exome-wide significant (Singh 2022, batch_040)
SCHEMA_GENES = [
    "CACNA1G", "CACUL1", "DLGAP1", "GRIN2A", "GRM5", "ITPR1",
    "RB1CC1", "SATB1", "SCN2A", "TGM5"
]

# F124 constrained tier: SynGO_EDT1 + glutamate receptor genes + TF regulators
# Glutamate receptor genes from EDT1 decomposition (batch_048, n=3)
GLUTAMATE_RECEPTOR_EDT1 = ["GRIN2A", "GRIN2B", "GRIN3A"]  # from batch_048
# Transcriptional regulator genes from EDT1 decomposition (batch_048, n=4)
# These were TCF4, MEF2C, and others based on the batch_048 decomposition
TF_REGULATOR_EDT1 = ["TCF4", "MEF2C", "MEF2D", "CHD1"]  # from batch_048 decomposition keywords

# Build constrained tier as union
EDT1_CONSTRAINED_TIER = list(set(SYNOGO_EDT1 + GLUTAMATE_RECEPTOR_EDT1 + TF_REGULATOR_EDT1))

# ------------------------------------------------------------------------------ Config
HAR_WINDOW_BP = 50_000  # ±50kb from TSS (same as Doan 2016 window)
RNG_SEED = 20260423

log = logging.getLogger(__name__)


def load_edt1_genes() -> set:
    """Load EDT1 gene list from PGC3 xlsx (ST12, protein-coding)."""
    df = pd.read_excel(PGC3_XLSX, sheet_name="ST12 all criteria")
    pc = df[df["gene_biotype"] == "protein_coding"]
    genes = set(pc["Symbol.ID"].dropna().str.upper().unique())
    log.info(f"Loaded {len(genes)} EDT1 protein-coding genes from PGC3 xlsx")
    return genes


def load_har_bed() -> pd.DataFrame:
    """Load HAR BED file."""
    har = pd.read_csv(HAR_BED, sep="\t", header=None,
                      names=["chrom", "start", "end", "source"])
    log.info(f"Loaded {len(har)} HARs from {HAR_BED.name}")
    return har


def load_tss() -> pd.DataFrame:
    """Load gene TSS coordinates, deduplicate to canonical TSS per gene."""
    tss = pd.read_csv(TSS_CSV)
    tss.columns = ["gene", "chrom", "tss", "strand", "biotype"]
    # Filter to protein-coding with valid chromosomes
    tss = tss[tss["biotype"] == "protein_coding"].copy()
    tss["gene"] = tss["gene"].str.upper()
    # Remove genes on patches/scaffolds
    valid_chroms = {str(i) for i in range(1, 23)} | {"X", "Y"}
    tss = tss[tss["chrom"].isin(valid_chroms)].copy()
    # Deduplicate: keep first TSS per gene
    tss = tss.drop_duplicates(subset="gene", keep="first")
    log.info(f"Loaded {len(tss)} protein-coding genes with valid TSS coordinates")
    return tss


def compute_har_overlap(har: pd.DataFrame, tss: pd.DataFrame) -> pd.DataFrame:
    """Compute which genes have ≥1 HAR within ±window_bp of their TSS."""
    # Build HAR lookup by chromosome
    har_by_chrom = {}
    for _, row in har.iterrows():
        chrom = str(row["chrom"])
        if chrom not in har_by_chrom:
            har_by_chrom[chrom] = []
        har_by_chrom[chrom].append((int(row["start"]), int(row["end"])))

    # For each gene, check HAR overlap
    has_har = []
    for _, row in tss.iterrows():
        chrom = str(row["chrom"])
        tss_pos = int(row["tss"])
        gene_start = tss_pos - HAR_WINDOW_BP
        gene_end = tss_pos + HAR_WINDOW_BP

        found = False
        if chrom in har_by_chrom:
            for h_start, h_end in har_by_chrom[chrom]:
                if not (gene_end < h_start or gene_start > h_end):
                    found = True
                    break
        has_har.append(found)

    tss = tss.copy()
    tss["has_har"] = has_har
    return tss


def fisher_test(gene_set: set, tss_df: pd.DataFrame) -> dict:
    """Fisher's exact test for HAR overlap in gene set vs background."""
    bg_genes = set(tss_df["gene"].values)
    bg_with_har = set(tss_df[tss_df["has_har"]]["gene"].values)

    n_bg = len(bg_genes)
    n_bg_har = len(bg_with_har)

    # Gene set genes that are in the background
    gs_in_bg = gene_set & bg_genes
    n_gs = len(gs_in_bg)
    n_gs_har = len(gs_in_bg & bg_with_har)

    # Fisher table
    a = n_gs_har                   # gene set + HAR
    b = n_gs - n_gs_har           # gene set, no HAR
    c = n_bg_har - n_gs_har       # background HAR, not in gene set
    d = n_bg - n_bg_har - (n_gs - n_gs_har)  # background, no HAR, not in gene set

    # Clamp to non-negative
    c = max(0, c)
    d = max(0, d)

    table = [[a, b], [c, d]]

    try:
        or_val, p_val = stats.fisher_exact(table)
    except Exception:
        or_val, p_val = np.nan, np.nan

    # CI (Woolf method)
    if a > 0 and b > 0 and c > 0 and d > 0:
        log_or = np.log(or_val)
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_low = np.exp(log_or - 1.96 * se)
        ci_high = np.exp(log_or + 1.96 * se)
    else:
        ci_low, ci_high = np.nan, np.nan

    # List overlapping genes
    har_genes = sorted(gs_in_bg & bg_with_har)

    return {
        "n_genes": n_gs,
        "n_har_overlap": n_gs_har,
        "har_rate": n_gs_har / n_gs if n_gs > 0 else 0,
        "bg_har_rate": n_bg_har / n_bg,
        "fisher_or": or_val,
        "fisher_p": p_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "har_overlap_genes": har_genes,
        "table_a": a, "table_b": b, "table_c": c, "table_d": d
    }


def run_analysis():
    """Main analysis."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== Batch 049 Sub-B: HAR × EDT1/SynGO_EDT1 Promoter Direct Test ===")

    # Load data
    har = load_har_bed()
    tss = load_tss()
    edt1_genes = load_edt1_genes()

    # Compute HAR overlap for all genes
    log.info(f"Computing HAR overlap with ±{HAR_WINDOW_BP/1000:.0f}kb promoter windows...")
    tss = compute_har_overlap(har, tss)

    n_with_har = tss["has_har"].sum()
    bg_rate = n_with_har / len(tss)
    log.info(f"Background: {n_with_har}/{len(tss)} genes ({100*bg_rate:.1f}%) have ≥1 HAR within ±{HAR_WINDOW_BP/1000:.0f}kb")

    # Define gene sets
    gene_sets = {
        "SynGO_EDT1": set(g.upper() for g in SYNOGO_EDT1),
        "EDT1": edt1_genes,
        "SCHEMA": set(g.upper() for g in SCHEMA_GENES),
        "EDT1_constrained_tier": set(g.upper() for g in EDT1_CONSTRAINED_TIER),
    }

    # Run Fisher's exact test
    results = {}
    for name, gs in gene_sets.items():
        r = fisher_test(gs, tss)
        r["gene_set"] = name
        results[name] = r
        log.info(f"  {name}: n={r['n_genes']}, har={r['n_har_overlap']}, "
                 f"OR={r['fisher_or']:.2f}, p={r['fisher_p']:.4f}, "
                 f"genes={r['har_overlap_genes']}")

    # BH correction
    raw_p = [results[k]["fisher_p"] for k in gene_sets]
    _, bh_q, _, _ = multipletests(raw_p, method="fdr_bh")
    for i, k in enumerate(gene_sets):
        results[k]["bh_q"] = bh_q[i]

    # Save output
    output = {
        "batch": "batch_049",
        "sub": "HAR_EDT1_promoter_direct_test",
        "generated_at": str(_dt.datetime.now()),
        "har_bed": str(HAR_BED),
        "tss_csv": str(TSS_CSV),
        "har_window_bp": HAR_WINDOW_BP,
        "n_hars": len(har),
        "n_genes_background": len(tss),
        "n_genes_with_har": int(n_with_har),
        "bg_har_rate": bg_rate,
        "gene_sets": [results[k] for k in gene_sets]
    }

    out_path = OUTPUT_DIR / "har_edt1_promoter_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    log.info("\n=== SUMMARY ===")
    log.info(f"{'Gene Set':<25} {'N':>5} {'HAR':>4} {'Rate':>7} {'BG%':>6} "
             f"{'OR':>8} {'P':>10} {'BH-q':>10}")
    log.info("-" * 85)
    for k in gene_sets:
        r = results[k]
        log.info(f"{k:<25} {r['n_genes']:>5} {r['n_har_overlap']:>4} "
                 f"{r['har_rate']:>7.3f} {r['bg_har_rate']:>6.3f} "
                 f"{r['fisher_or']:>8.2f} {r['fisher_p']:>10.4f} {r['bh_q']:>10.4f}")

    log.info(f"\nResults saved to {out_path}")
    return output


if __name__ == "__main__":
    run_analysis()
