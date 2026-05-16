#!/usr/bin/env python3
"""
Batch 069 E1 (R1): SynGO Constraint Specificity Test
=====================================================
Hypothesis: Whether SynGO∩EDT1 pLI enrichment (OR=26.44, batch_047 F121)
is SCZ-specific or a general property of synaptic gene class membership.

Three analyses:
  1. PRIMARY:      SynGO-minus-EDT1 pLI>=0.9 enrichment vs ALL gnomAD
                   canonical genes. Fisher's exact + 5000 length-stratified
                   permutations.
  2. WITHIN-CLASS: SynGO∩EDT1 (n=14) vs SynGO-minus-EDT1 — direct
                   comparison of pLI>=0.9 rates WITHIN SynGO class.
                   Fisher's exact.
  3. BRAIN-BG:     SynGO-minus-EDT1 pLI>=0.9 enrichment vs
                   brain-expressed genes only (GTEx v8, TPM>1 in any
                   Brain tissue). Fisher's exact.

All three repeated for LOEUF<=0.35.

Data sources:
  - gnomAD v4.1 constraint:  data/item_15/gnomad.v4.1.constraint_metrics.tsv
  - SynGO 2024 GMT:          experiments/batch_052_A/input/syngo_2024.gmt
  - EDT1 genes:              experiments/batch_018/pgc3_genes.txt
  - GTEx v8 median TPM:      data/GTEx_v8_gene_median_tpm.gct.gz
  - SynGO∩EDT1 (n=14):       hardcoded from batch_047 (Trubetskoy 2022 EDT1)

Hyperparameter citations:
  - pLI>=0.9 threshold:  Lek et al. 2016 (Nature) standard for constrained genes
  - LOEUF<=0.35 threshold: Karczewski et al. 2020 (Nature), most constrained decile
  - 5000 permutations: project standard from batch_047 review; sufficient for
    empirical p resolution to 2e-4
  - GTEx TPM>=1 brain expression: standard filter used in batch_040, batch_052_B
  - Length-stratified permutation: controls for CDS length confound per batch_047

WHY this experiment: The OR=26.44 for SynGO∩EDT1 could reflect either (a) SCZ-
specific constraint excess in these 14 genes, or (b) SynGO genes in general being
highly constrained (synaptic genes tend to be long, dosage-sensitive). If SynGO-
minus-EDT1 shows similarly high constraint enrichment, the SynGO∩EDT1 result is
NOT SCZ-specific — it's a property of synaptic gene class membership.
"""
import gzip
import json
import math
import os
import platform
import random
import statistics
import sys
import time

import numpy as np
from scipy.stats import fisher_exact

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = "/home/yuanz/Documents/GitHub/biomarvin_schizophrenia"
GNOMAD_FILE = os.path.join(PROJECT_ROOT, "data/item_15/gnomad.v4.1.constraint_metrics.tsv")
SYNGO_GMT = os.path.join(PROJECT_ROOT, "experiments/batch_052_A/input/syngo_2024.gmt")
EDT1_FILE = os.path.join(PROJECT_ROOT, "experiments/batch_018/pgc3_genes.txt")
GTEX_FILE = os.path.join(PROJECT_ROOT, "data/GTEx_v8_gene_median_tpm.gct.gz")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments/batch_069/output")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "e1_syngo_constraint_specificity.json")

PLI_THRESHOLD = 0.9       # Lek et al. 2016
LOEUF_THRESHOLD = 0.35    # Karczewski et al. 2020
N_PERM = 5000             # batch_047 review standard
SEED = 42

# SynGO∩EDT1 genes (n=14, hardcoded from batch_047)
SYNGO_EDT1 = [
    'DLGAP1', 'GRIN2A', 'NRXN1', 'CNTNAP2', 'ARC', 'DLG4',
    'NRXN2', 'NLGN1', 'NLGN2', 'SHANK1', 'SHANK3', 'HOMER1',
    'SYN1', 'GAP43'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ENVIRONMENT LOGGING
# ============================================================
env_info = {
    "python_version": sys.version,
    "numpy_version": np.__version__,
    "scipy_version": __import__("scipy").__version__,
    "platform": platform.platform(),
    "seed": SEED,
    "n_permutations": N_PERM,
    "script": os.path.abspath(__file__),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}

print("=" * 70)
print("Batch 069 E1: SynGO Constraint Specificity Test")
print("=" * 70)
print(f"Python: {sys.version}")
print(f"NumPy:  {np.__version__}")
print(f"SciPy:  {__import__('scipy').__version__}")
print(f"Seed:   {SEED}")
print(f"Perms:  {N_PERM}")
print()

# ============================================================
# STEP 1: Load SynGO gene universe from GMT
# ============================================================
print("[1/5] Loading SynGO 2024 gene universe from GMT...")


def parse_syngo_gmt(path):
    """Parse SynGO GMT — union of all set members.
    Parser mirrors batch_052_A/scripts/04_axis_c_functional_tier.py:parse_syngo_gmt
    WHY: Rule 1, reuse canonical parser.
    """
    genes = set()
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            for tok in parts[2:]:
                tok = tok.strip()
                if tok:
                    genes.add(tok)
    return genes


syngo_all = parse_syngo_gmt(SYNGO_GMT)
print(f"  SynGO total unique genes: {len(syngo_all)}")

# ============================================================
# STEP 2: Load EDT1 genes
# ============================================================
print("[2/5] Loading EDT1 gene list...")
with open(EDT1_FILE) as f:
    edt1_genes = set(line.strip() for line in f if line.strip() and len(line.strip()) > 2)
print(f"  EDT1 total genes: {len(edt1_genes)}")

syngo_edt1_set = set(SYNGO_EDT1)
syngo_minus_edt1 = syngo_all - syngo_edt1_set
print(f"  SynGO ∩ EDT1 (hardcoded): {len(syngo_edt1_set)} genes")
print(f"  SynGO \\ EDT1:             {len(syngo_minus_edt1)} genes")

# ============================================================
# STEP 3: Load gnomAD v4.1 constraint metrics (canonical transcripts)
# ============================================================
print("[3/5] Loading gnomAD v4.1 constraint metrics...")

gene_data = {}  # gene -> {pLI, LOEUF, cds_length}

with open(GNOMAD_FILE) as f:
    header = f.readline().strip().split('\t')
    col_idx = {col: i for i, col in enumerate(header)}

    required_cols = ['gene', 'canonical', 'lof.pLI', 'lof.oe_ci.upper', 'cds_length']
    for rc in required_cols:
        if rc not in col_idx:
            print(f"  WARNING: Column '{rc}' not found in gnomAD file")

    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < len(header):
            continue

        gene = parts[col_idx['gene']]
        is_canonical = parts[col_idx['canonical']].lower() == 'true'

        if not is_canonical:
            continue

        def safe_float(col_name):
            try:
                val = parts[col_idx[col_name]]
                if val in ('', 'NA', 'nan', 'None'):
                    return None
                return float(val)
            except (ValueError, IndexError, KeyError):
                return None

        pli = safe_float('lof.pLI')
        loeuf = safe_float('lof.oe_ci.upper')
        cds_length = safe_float('cds_length')

        gene_data[gene] = {
            'pLI': pli,
            'LOEUF': loeuf,
            'cds_length': cds_length,
        }

print(f"  Canonical genes loaded: {len(gene_data)}")
pli_available = sum(1 for g in gene_data if gene_data[g]['pLI'] is not None)
loeuf_available = sum(1 for g in gene_data if gene_data[g]['LOEUF'] is not None)
cds_available = sum(1 for g in gene_data if gene_data[g]['cds_length'] is not None)
print(f"  pLI available:   {pli_available}")
print(f"  LOEUF available: {loeuf_available}")
print(f"  CDS length available: {cds_available}")

# Map gene sets to gnomAD
syngo_in_gnomad = syngo_all & set(gene_data.keys())
syngo_minus_edt1_in_gnomad = syngo_minus_edt1 & set(gene_data.keys())
syngo_edt1_in_gnomad = syngo_edt1_set & set(gene_data.keys())
all_gnomad_genes = set(gene_data.keys())

print(f"\n  SynGO in gnomAD:          {len(syngo_in_gnomad)}/{len(syngo_all)}")
print(f"  SynGO\\EDT1 in gnomAD:     {len(syngo_minus_edt1_in_gnomad)}/{len(syngo_minus_edt1)}")
print(f"  SynGO∩EDT1 in gnomAD:     {len(syngo_edt1_in_gnomad)}/{len(syngo_edt1_set)}")
print(f"  SynGO∩EDT1 genes in gnomAD: {sorted(syngo_edt1_in_gnomad)}")
print(f"  SynGO∩EDT1 genes NOT in gnomAD: {sorted(syngo_edt1_set - set(gene_data.keys()))}")

# ============================================================
# STEP 4: Load GTEx v8 brain-expressed genes
# ============================================================
print("\n[4/5] Loading GTEx v8 brain-expressed genes (TPM>=1 in any Brain tissue)...")

import pandas as pd

# Parser mirrors batch_052_B/scripts/common.py:load_gtex_brain_and_any
# WHY: Rule 1, reuse canonical project parser approach
with gzip.open(GTEX_FILE, "rt") as f:
    _ = f.readline()  # "#1.2"
    _ = f.readline()  # "n_rows\tn_cols"
gtex_df = pd.read_csv(GTEX_FILE, sep="\t", skiprows=2, low_memory=False)
brain_cols = [c for c in gtex_df.columns if c.startswith("Brain")]
print(f"  GTEx brain tissues: {len(brain_cols)}")

# Brain-expressed = TPM>=1 in any brain tissue
brain_tpm = gtex_df[brain_cols].apply(pd.to_numeric, errors="coerce")
brain_mask = (brain_tpm >= 1).any(axis=1)
brain_expressed_genes = set(gtex_df.loc[brain_mask, "Description"].astype(str))
print(f"  Brain-expressed genes (TPM>=1): {len(brain_expressed_genes)}")

# Brain-expressed genes in gnomAD
brain_in_gnomad = brain_expressed_genes & all_gnomad_genes
print(f"  Brain-expressed in gnomAD: {len(brain_in_gnomad)}")

# Free memory
del gtex_df, brain_tpm
print()


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def fisher_or_ci(a, b, c, d, alternative="greater"):
    """Fisher exact test with Haldane-Anscombe correction for zero cells.
    Returns OR, p, 95% CI, cell counts.
    WHY Haldane-Anscombe: standard correction when any cell=0, avoids
    infinite OR while preserving interpretability (batch_052_B common.py).
    """
    halda = any(v == 0 for v in (a, b, c, d))
    aa, bb, cc, dd = (a + 0.5, b + 0.5, c + 0.5, d + 0.5) if halda else (a, b, c, d)
    or_val = (aa * dd) / (bb * cc) if bb * cc > 0 else float("inf")

    # Woolf 95% CI on log(OR)
    if or_val > 0 and not math.isinf(or_val):
        se = math.sqrt(1/aa + 1/bb + 1/cc + 1/dd)
        ln_or = math.log(or_val)
        ci_low = math.exp(ln_or - 1.96 * se)
        ci_high = math.exp(ln_or + 1.96 * se)
    else:
        ci_low, ci_high = float("nan"), float("nan")

    _, p_val = fisher_exact([[a, b], [c, d]], alternative=alternative)

    return {
        "a": a, "b": b, "c": c, "d": d,
        "OR": round(or_val, 4),
        "OR_ci_low": round(ci_low, 4) if not math.isnan(ci_low) else None,
        "OR_ci_high": round(ci_high, 4) if not math.isnan(ci_high) else None,
        "p_value": float(p_val),
        "haldane_applied": halda,
    }


def length_stratified_permutation(test_genes, background_genes, metric_key,
                                  threshold, direction, n_perm, seed):
    """Length-stratified permutation test.
    WHY length-stratified: longer genes have more LoF variants observed,
    biasing constraint scores. Stratifying by CDS length controls this
    confound (Karczewski 2020).

    Bins: 1kb CDS length bins (0-999, 1000-1999, ..., 99000+).
    For each permutation: sample from same length bins as test genes,
    matching the per-bin count. This preserves the CDS length distribution.
    """
    rng = random.Random(seed)

    # Build length bins for background
    length_bins = {}
    for g in background_genes:
        cds = gene_data[g].get('cds_length')
        if cds is not None and cds > 0:
            bin_idx = min(int(cds / 1000), 99)
            if bin_idx not in length_bins:
                length_bins[bin_idx] = []
            length_bins[bin_idx].append(g)

    # Get test gene bin counts
    test_bin_counts = {}
    test_genes_with_length = []
    for g in test_genes:
        cds = gene_data[g].get('cds_length')
        if cds is not None and cds > 0:
            bin_idx = min(int(cds / 1000), 99)
            test_bin_counts[bin_idx] = test_bin_counts.get(bin_idx, 0) + 1
            test_genes_with_length.append(g)

    # Observed count
    if direction == "high":
        obs_count = sum(1 for g in test_genes_with_length
                        if gene_data[g][metric_key] is not None
                        and gene_data[g][metric_key] >= threshold)
    else:  # low
        obs_count = sum(1 for g in test_genes_with_length
                        if gene_data[g][metric_key] is not None
                        and gene_data[g][metric_key] <= threshold)

    # Run permutations
    perm_counts = []
    for _ in range(n_perm):
        sampled = []
        for bin_idx, n_needed in test_bin_counts.items():
            pool = length_bins.get(bin_idx, [])
            if len(pool) >= n_needed:
                sampled.extend(rng.sample(pool, n_needed))
            else:
                # If bin has fewer genes than needed, take all + sample remainder
                # from adjacent bins (fallback)
                sampled.extend(pool)
                shortfall = n_needed - len(pool)
                # Try adjacent bins
                for adj in [bin_idx - 1, bin_idx + 1, bin_idx - 2, bin_idx + 2]:
                    if shortfall <= 0:
                        break
                    adj_pool = [g for g in length_bins.get(adj, []) if g not in sampled]
                    take = min(shortfall, len(adj_pool))
                    if take > 0:
                        sampled.extend(rng.sample(adj_pool, take))
                        shortfall -= take

        if direction == "high":
            perm_count = sum(1 for g in sampled
                             if gene_data[g][metric_key] is not None
                             and gene_data[g][metric_key] >= threshold)
        else:
            perm_count = sum(1 for g in sampled
                             if gene_data[g][metric_key] is not None
                             and gene_data[g][metric_key] <= threshold)
        perm_counts.append(perm_count)

    # Empirical p (one-sided, >= observed)
    # WHY +1/+1: Phipson & Smyth 2010, corrects for anti-conservative
    # behavior of naive empirical p-values
    emp_p = (sum(1 for pc in perm_counts if pc >= obs_count) + 1) / (n_perm + 1)

    return {
        "observed_count": obs_count,
        "n_test_genes_with_length": len(test_genes_with_length),
        "empirical_p": round(emp_p, 6),
        "perm_mean": round(statistics.mean(perm_counts), 2) if perm_counts else None,
        "perm_sd": round(statistics.stdev(perm_counts), 2) if len(perm_counts) > 1 else None,
        "perm_median": round(statistics.median(perm_counts), 2) if perm_counts else None,
        "perm_max": max(perm_counts) if perm_counts else None,
    }


def run_enrichment(test_gene_set, background_gene_set, metric_key, threshold,
                   direction, label, run_perm=True):
    """Run Fisher's exact enrichment test + optional length-stratified permutations.

    WHY Fisher's exact over chi-square: several cells may be small (n=14 for
    SynGO∩EDT1), so the chi-square approximation is unreliable. Fisher's exact
    is valid for all sample sizes.

    Parameters
    ----------
    test_gene_set : set of gene symbols (foreground)
    background_gene_set : set of gene symbols (full background incl. test)
    metric_key : 'pLI' or 'LOEUF'
    threshold : float
    direction : 'high' (>=threshold) or 'low' (<=threshold)
    label : str for printing
    run_perm : bool, whether to run length-stratified permutations
    """
    # Filter to genes with non-None metric values
    test_with_metric = [g for g in test_gene_set
                        if g in gene_data and gene_data[g][metric_key] is not None]
    bg_non_test = [g for g in background_gene_set - test_gene_set
                   if g in gene_data and gene_data[g][metric_key] is not None]

    if direction == "high":
        a = sum(1 for g in test_with_metric if gene_data[g][metric_key] >= threshold)
        c = sum(1 for g in bg_non_test if gene_data[g][metric_key] >= threshold)
    else:
        a = sum(1 for g in test_with_metric if gene_data[g][metric_key] <= threshold)
        c = sum(1 for g in bg_non_test if gene_data[g][metric_key] <= threshold)

    b = len(test_with_metric) - a
    d = len(bg_non_test) - c

    result = fisher_or_ci(a, b, c, d)
    result["test_n"] = len(test_with_metric)
    result["bg_n"] = len(bg_non_test)
    result["test_rate"] = round(a / len(test_with_metric), 4) if test_with_metric else None
    result["bg_rate"] = round(c / len(bg_non_test), 4) if bg_non_test else None
    result["metric"] = metric_key
    result["threshold"] = threshold
    result["direction"] = direction

    # Print
    sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
    print(f"  {label}:")
    print(f"    Test: {a}/{len(test_with_metric)} ({result['test_rate']*100:.1f}%) | "
          f"BG: {c}/{len(bg_non_test)} ({result['bg_rate']*100:.1f}%)")
    print(f"    OR={result['OR']:.2f} [{result['OR_ci_low']}-{result['OR_ci_high']}], "
          f"p={result['p_value']:.2e} {sig}")
    print(f"    Table: [[{a},{b}],[{c},{d}]]"
          f"{' (Haldane +0.5 applied)' if result['haldane_applied'] else ''}")

    # Permutation test
    if run_perm:
        perm_result = length_stratified_permutation(
            test_gene_set, background_gene_set,
            metric_key, threshold, direction, N_PERM, SEED
        )
        result["permutation"] = perm_result
        print(f"    Empirical p (length-stratified, {N_PERM} perms): "
              f"{perm_result['empirical_p']:.4f}")
        print(f"    Perm null: mean={perm_result['perm_mean']}, "
              f"sd={perm_result['perm_sd']}, max={perm_result['perm_max']}")

    return result


# ============================================================
# STEP 5: Run all three analyses x two metrics
# ============================================================
print("=" * 70)
print("[5/5] Running analyses...")
print("=" * 70)

results = {
    "gene_counts": {
        "syngo_total": len(syngo_all),
        "syngo_in_gnomad": len(syngo_in_gnomad),
        "syngo_minus_edt1_total": len(syngo_minus_edt1),
        "syngo_minus_edt1_in_gnomad": len(syngo_minus_edt1_in_gnomad),
        "syngo_edt1_total": len(syngo_edt1_set),
        "syngo_edt1_in_gnomad": len(syngo_edt1_in_gnomad),
        "syngo_edt1_genes": sorted(syngo_edt1_set),
        "syngo_edt1_genes_in_gnomad": sorted(syngo_edt1_in_gnomad),
        "gnomad_canonical_total": len(all_gnomad_genes),
        "brain_expressed_total": len(brain_expressed_genes),
        "brain_expressed_in_gnomad": len(brain_in_gnomad),
    },
    "analyses": {},
    "environment": env_info,
}

# ------ ANALYSIS 1: PRIMARY — SynGO\EDT1 vs ALL gnomAD ------
print("\n--- Analysis 1: PRIMARY — SynGO\\EDT1 vs ALL gnomAD canonical ---")

print("\n  [pLI >= 0.9]")
r1_pli = run_enrichment(
    syngo_minus_edt1_in_gnomad, all_gnomad_genes,
    'pLI', PLI_THRESHOLD, 'high',
    'SynGO\\EDT1 vs gnomAD (pLI>=0.9)', run_perm=True
)

print("\n  [LOEUF <= 0.35]")
r1_loeuf = run_enrichment(
    syngo_minus_edt1_in_gnomad, all_gnomad_genes,
    'LOEUF', LOEUF_THRESHOLD, 'low',
    'SynGO\\EDT1 vs gnomAD (LOEUF<=0.35)', run_perm=True
)

results["analyses"]["primary_syngo_minus_edt1_vs_genome"] = {
    "description": "SynGO-minus-EDT1 pLI/LOEUF enrichment vs ALL gnomAD canonical genes",
    "pLI_ge_0.9": r1_pli,
    "LOEUF_le_0.35": r1_loeuf,
}

# ------ ANALYSIS 2: WITHIN-CLASS — SynGO∩EDT1 vs SynGO\EDT1 ------
print("\n--- Analysis 2: WITHIN-CLASS — SynGO∩EDT1 vs SynGO\\EDT1 ---")
# Here the foreground is SynGO∩EDT1, background is all SynGO genes
# This directly tests whether EDT1 genes are MORE constrained than
# other SynGO genes — the SCZ-specificity question.

print("\n  [pLI >= 0.9]")
r2_pli = run_enrichment(
    syngo_edt1_in_gnomad, syngo_in_gnomad,
    'pLI', PLI_THRESHOLD, 'high',
    'SynGO∩EDT1 vs SynGO\\EDT1 (pLI>=0.9)', run_perm=False
)

print("\n  [LOEUF <= 0.35]")
r2_loeuf = run_enrichment(
    syngo_edt1_in_gnomad, syngo_in_gnomad,
    'LOEUF', LOEUF_THRESHOLD, 'low',
    'SynGO∩EDT1 vs SynGO\\EDT1 (LOEUF<=0.35)', run_perm=False
)

results["analyses"]["within_class_syngo_edt1_vs_syngo_minus_edt1"] = {
    "description": "SynGO-intersect-EDT1 vs SynGO-minus-EDT1 — within-class comparison",
    "pLI_ge_0.9": r2_pli,
    "LOEUF_le_0.35": r2_loeuf,
}

# ------ ANALYSIS 3: BRAIN-BG — SynGO\EDT1 vs brain-expressed ------
print("\n--- Analysis 3: BRAIN-BG — SynGO\\EDT1 vs brain-expressed genes ---")

# Background: brain-expressed genes in gnomAD
print("\n  [pLI >= 0.9]")
r3_pli = run_enrichment(
    syngo_minus_edt1_in_gnomad, brain_in_gnomad,
    'pLI', PLI_THRESHOLD, 'high',
    'SynGO\\EDT1 vs brain-expressed (pLI>=0.9)', run_perm=False
)

print("\n  [LOEUF <= 0.35]")
r3_loeuf = run_enrichment(
    syngo_minus_edt1_in_gnomad, brain_in_gnomad,
    'LOEUF', LOEUF_THRESHOLD, 'low',
    'SynGO\\EDT1 vs brain-expressed (LOEUF<=0.35)', run_perm=False
)

results["analyses"]["syngo_minus_edt1_vs_brain_expressed"] = {
    "description": "SynGO-minus-EDT1 vs brain-expressed genes (GTEx TPM>=1)",
    "pLI_ge_0.9": r3_pli,
    "LOEUF_le_0.35": r3_loeuf,
}

# ============================================================
# SUPPLEMENTARY: SynGO∩EDT1 vs ALL gnomAD (reproduction reference)
# ============================================================
print("\n--- Supplementary: SynGO∩EDT1 vs ALL gnomAD (batch_047 reference) ---")

print("\n  [pLI >= 0.9]")
r_ref_pli = run_enrichment(
    syngo_edt1_in_gnomad, all_gnomad_genes,
    'pLI', PLI_THRESHOLD, 'high',
    'SynGO∩EDT1 vs gnomAD (pLI>=0.9) [REFERENCE]', run_perm=False
)

print("\n  [LOEUF <= 0.35]")
r_ref_loeuf = run_enrichment(
    syngo_edt1_in_gnomad, all_gnomad_genes,
    'LOEUF', LOEUF_THRESHOLD, 'low',
    'SynGO∩EDT1 vs gnomAD (LOEUF<=0.35) [REFERENCE]', run_perm=False
)

results["analyses"]["reference_syngo_edt1_vs_genome"] = {
    "description": "SynGO-intersect-EDT1 vs ALL gnomAD (batch_047 comparison reference)",
    "note": "LOEUF here uses lof.oe_ci.upper (Karczewski 2020), batch_047 used lof.oe — values may differ",
    "pLI_ge_0.9": r_ref_pli,
    "LOEUF_le_0.35": r_ref_loeuf,
}

# ============================================================
# INTERPRETIVE SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("INTERPRETIVE SUMMARY")
print("=" * 70)

print("\nKey question: Is the SynGO∩EDT1 constraint enrichment SCZ-specific?")
print()

# Extract key ORs
ref_or = r_ref_pli["OR"]
syngo_or = r1_pli["OR"]
within_or = r2_pli["OR"]
brain_or = r3_pli["OR"]

print(f"  Reference: SynGO∩EDT1 vs genome      OR = {ref_or:.2f} (p = {r_ref_pli['p_value']:.2e})")
print(f"  Analysis 1: SynGO\\EDT1 vs genome     OR = {syngo_or:.2f} (p = {r1_pli['p_value']:.2e}, emp_p = {r1_pli['permutation']['empirical_p']:.4f})")
print(f"  Analysis 2: SynGO∩EDT1 vs SynGO\\EDT1 OR = {within_or:.2f} (p = {r2_pli['p_value']:.2e})")
print(f"  Analysis 3: SynGO\\EDT1 vs brain       OR = {brain_or:.2f} (p = {r3_pli['p_value']:.2e})")

print()
print("Interpretation (pLI >= 0.9):")

# Decision logic
if syngo_or > 2.0 and r1_pli["p_value"] < 0.05:
    print("  SynGO\\EDT1 genes ARE enriched for constraint vs genome (OR > 2, p < 0.05).")
    print("  This means high constraint is a GENERAL property of SynGO membership,")
    print("  not specific to SCZ-linked genes.")
    if within_or > 2.0 and r2_pli["p_value"] < 0.05:
        print("  HOWEVER, SynGO∩EDT1 shows ADDITIONAL constraint beyond the SynGO baseline")
        print("  (within-class OR > 2, p < 0.05), suggesting a SCZ-specific component.")
    elif within_or > 1.0:
        print("  SynGO∩EDT1 does NOT show significantly higher constraint than SynGO\\EDT1")
        print(f"  (within-class OR = {within_or:.2f}, p = {r2_pli['p_value']:.2e}).")
        print("  The original OR=26.44 is LARGELY attributable to SynGO class membership.")
    else:
        print("  SynGO∩EDT1 shows LOWER or equal constraint vs SynGO\\EDT1.")
        print("  The original enrichment is ENTIRELY attributable to SynGO class membership.")
else:
    print("  SynGO\\EDT1 genes are NOT enriched for constraint vs genome.")
    print("  This would support SCZ-specificity of the original finding.")

print()
if brain_or > 2.0 and r3_pli["p_value"] < 0.05:
    print("  Brain-expressed background: SynGO\\EDT1 still enriched vs brain genes.")
    print("  Constraint excess is synaptic-specific, not just brain-expression-driven.")
elif brain_or < syngo_or * 0.5:
    print("  Brain-expressed background reduces OR substantially.")
    print("  Part of the genome-wide enrichment was brain-expression confounding.")
else:
    print("  Brain-expressed background: moderate attenuation of OR.")

# LOEUF summary
print()
print("LOEUF <= 0.35 summary:")
ref_loeuf = r_ref_loeuf["OR"]
syngo_loeuf = r1_loeuf["OR"]
within_loeuf = r2_loeuf["OR"]
brain_loeuf = r3_loeuf["OR"]
print(f"  Reference: SynGO∩EDT1 vs genome      OR = {ref_loeuf:.2f} (p = {r_ref_loeuf['p_value']:.2e})")
print(f"  Analysis 1: SynGO\\EDT1 vs genome     OR = {syngo_loeuf:.2f} (p = {r1_loeuf['p_value']:.2e}, emp_p = {r1_loeuf['permutation']['empirical_p']:.4f})")
print(f"  Analysis 2: SynGO∩EDT1 vs SynGO\\EDT1 OR = {within_loeuf:.2f} (p = {r2_loeuf['p_value']:.2e})")
print(f"  Analysis 3: SynGO\\EDT1 vs brain       OR = {brain_loeuf:.2f} (p = {r3_loeuf['p_value']:.2e})")

# Per-gene detail for SynGO∩EDT1
print()
print("Per-gene constraint detail (SynGO∩EDT1):")
print(f"  {'Gene':<12} {'pLI':>8} {'LOEUF':>8} {'pLI>=0.9':>10} {'LOEUF<=0.35':>12}")
print("  " + "-" * 52)
for g in sorted(SYNGO_EDT1):
    if g in gene_data:
        pli = gene_data[g]['pLI']
        loeuf = gene_data[g]['LOEUF']
        pli_str = f"{pli:.4f}" if pli is not None else "NA"
        loeuf_str = f"{loeuf:.4f}" if loeuf is not None else "NA"
        pli_flag = "YES" if pli is not None and pli >= 0.9 else "no"
        loeuf_flag = "YES" if loeuf is not None and loeuf <= 0.35 else "no"
        print(f"  {g:<12} {pli_str:>8} {loeuf_str:>8} {pli_flag:>10} {loeuf_flag:>12}")
    else:
        print(f"  {g:<12} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'N/A':>12}")

# ============================================================
# SAVE RESULTS
# ============================================================
# Convert any sets to sorted lists for JSON serialization
def make_serializable(obj):
    if isinstance(obj, set):
        return sorted(obj)
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


results_serializable = make_serializable(results)

with open(OUTPUT_JSON, 'w') as f:
    json.dump(results_serializable, f, indent=2, default=str)
print(f"\nResults saved to: {OUTPUT_JSON}")
print("DONE.")
