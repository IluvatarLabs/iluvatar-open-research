#!/usr/bin/env python3
"""
Batch 034: S-LDSC Cell-Type Partitioned Heritability + Genetic Correlation + Forest Plots

Addresses D35 (S-LDSC annotation fix), G2 (genetic correlation), G4 (forest plots)
"""

import subprocess
import os
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================================
# PATHS
# ============================================================================
PROJECT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
LDSC = "/home/yuanz/torchml/bin/ldsc.py"
MUNGE = "/home/yuanz/torchml/bin/munge_sumstats.py"

# Input data
PLINK_REF = str(PROJECT / "data/ldsc/1000G_EUR_Phase3_plink/1000G.EUR.QC")
FRQ_REF = str(PROJECT / "data/ldsc/1000G_Phase3_frq/1000G.EUR.QC")
ANNOT_DIR = str(PROJECT / "experiments/batch_029/output/annotations_rsID")
BASELINE_LD = str(PROJECT / "data/ldsc/baselineLD/baselineLD.")
WEIGHTS = str(PROJECT / "data/ldsc/weights/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.")
HM3_SNPS = str(PROJECT / "data/ldsc/weights/1000G_Phase3_weights_hm3_no_MHC/hm3_snps.txt")
PGC3_SUMSTATS = str(PROJECT / "data/ldsc/PGC3_sumstats/PGC3_EUR_v2.sumstats.gz")

# Output
OUT_DIR = str(PROJECT / "experiments/batch_034/output")
LDSCORES_DIR = str(PROJECT / "experiments/batch_034/output/ld_scores")
SUMSTATS_DIR = str(PROJECT / "data/ldsc/comparator_sumstats")


def run_cmd(cmd, description="", timeout=3600):
    """Run a command and return output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"CMD: {cmd[:200]}...")
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - start
        print(f"TIME: {elapsed:.1f}s | EXIT: {result.returncode}")
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[:500]}")
        return result
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return None


# ============================================================================
# ANALYSIS 1: S-LDSC CELL-TYPE PARTITIONED HERITABILITY
# ============================================================================

def compute_ld_scores(chr_num):
    """Compute LD scores for one chromosome."""
    cmd = (
        f"python3 {LDSC} "
        f"--l2 "
        f"--bfile {PLINK_REF}.{chr_num} "
        f"--annot {ANNOT_DIR}/celltype.{chr_num}.annot.gz "
        f"--ld-wind-cm 1 "
        f"--out {LDSCORES_DIR}/celltype.{chr_num}"
    )
    return run_cmd(cmd, f"LD scores chr{chr_num}", timeout=600)


def run_partitioned_heritability():
    """Run S-LDSC partitioned heritability with cell-type + baselineLD."""
    cmd = (
        f"python3 {LDSC} "
        f"--h2 {PGC3_SUMSTATS} "
        f"--ref-ld-chr {LDSCORES_DIR}/celltype.,{BASELINE_LD} "
        f"--w-ld-chr {WEIGHTS} "
        f"--overlap-annot "
        f"--frqfile-chr {FRQ_REF} "
        f"--out {OUT_DIR}/celltype_partitioned"
    )
    return run_cmd(cmd, "S-LDSC partitioned heritability", timeout=1800)


def analysis1_sldsc():
    """Full S-LDSC pipeline."""
    print("\n" + "="*60)
    print("ANALYSIS 1: S-LDSC CELL-TYPE PARTITIONED HERITABILITY")
    print("="*60)

    os.makedirs(LDSCORES_DIR, exist_ok=True)

    # Step 1: Compute LD scores for all 22 chromosomes in parallel
    print("\n--- Step 1: Computing LD scores (22 chromosomes in parallel) ---")
    with ProcessPoolExecutor(max_workers=11) as executor:
        futures = {executor.submit(compute_ld_scores, c): c for c in range(1, 23)}
        results = {}
        for future in as_completed(futures):
            chr_num = futures[future]
            try:
                result = future.result()
                results[chr_num] = result.returncode if result else -1
            except Exception as e:
                results[chr_num] = -1
                print(f"ERROR chr{chr_num}: {e}")

    # Check results
    successful = sum(1 for v in results.values() if v == 0)
    print(f"\nLD score computation: {successful}/22 chromosomes successful")

    if successful < 22:
        failed = [k for k, v in results.items() if v != 0]
        print(f"FAILED chromosomes: {failed}")
        if successful < 20:
            print("TOO MANY FAILURES - cannot proceed with partitioned heritability")
            return None

    # Step 2: Run partitioned heritability
    print("\n--- Step 2: Running partitioned heritability ---")
    result = run_partitioned_heritability()

    if result and result.returncode == 0:
        # Parse results
        results_file = f"{OUT_DIR}/celltype_partitioned.results"
        if os.path.exists(results_file):
            with open(results_file) as f:
                print(f.read())
            return results_file
    else:
        print("Partitioned heritability FAILED")
        return None


# ============================================================================
# ANALYSIS 2: LDSC GENETIC CORRELATION
# ============================================================================

# IEU OpenGWAS IDs for comparator traits (to be verified)
COMPARATOR_TRAITS = {
    "BIP": {
        "id": "ieu-a-1014",
        "name": "Bipolar Disorder",
        "expected_rg": (0.6, 0.8),
    },
    "MDD": {
        "id": "ieu-a-1007",
        "name": "Major Depressive Disorder",
        "expected_rg": (0.3, 0.4),
    },
    "ADHD": {
        "id": "ieu-a-1009",
        "name": "ADHD",
        "expected_rg": (0.2, 0.3),
    },
    "ASD": {
        "id": "ieu-a-811",
        "name": "Autism Spectrum Disorder",
        "expected_rg": (0.2, 0.25),
    },
    "EDU": {
        "id": "ieu-a-1021",
        "name": "Educational Attainment",
        "expected_rg": (0.1, 0.2),
    },
    "CRP": {
        "id": "ieu-a-7",
        "name": "C-Reactive Protein",
        "expected_rg": (-0.1, 0.0),
    },
    "COG": {
        "id": "ieu-a-899",
        "name": "Cognitive Performance",
        "expected_rg": (0.2, 0.3),
    },
    "BMI": {
        "id": "ieu-a-2",
        "name": "Body Mass Index",
        "expected_rg": (-0.05, 0.05),
    },
}


def download_comparator_sumstats(trait_key, trait_info):
    """Download and munge comparator GWAS summary statistics."""
    ieu_id = trait_info["id"]
    out_prefix = f"{SUMSTATS_DIR}/{trait_key}"

    # Check if already munged
    if os.path.exists(f"{out_prefix}.sumstats.gz"):
        print(f"  {trait_key}: Already munged, skipping")
        return True

    print(f"  {trait_key}: Downloading from IEU OpenGWAS ({ieu_id})...")

    # Try downloading via ieugwaspy or curl
    # Method 1: Direct API download
    raw_file = f"{SUMSTATS_DIR}/{trait_key}_raw.txt"

    # Try downloading via API
    cmd = f"curl -sL 'https://gwas.mrcieu.ac.uk/files/{ieu_id}/{ieu_id}.tsv.gz' -o {raw_file}.gz"
    result = run_cmd(cmd, f"Download {trait_key}", timeout=300)

    if not result or result.returncode != 0:
        print(f"  {trait_key}: Download FAILED")
        return False

    # Check file size
    size = os.path.getsize(f"{raw_file}.gz") if os.path.exists(f"{raw_file}.gz") else 0
    if size < 1000:
        print(f"  {trait_key}: Downloaded file too small ({size} bytes)")
        return False

    # Munge sumstats
    print(f"  {trait_key}: Munging summary statistics...")
    cmd = (
        f"python3 {MUNGE} "
        f"--sumstats {raw_file}.gz "
        f"--merge-alleles {HM3_SNPS} "
        f"--out {out_prefix} "
        f"--snp SNP --a1 A1 --a2 A2 --p P --beta beta --se se --N N "
        f"--frq-filter-b 0.5"
    )
    result = run_cmd(cmd, f"Munge {trait_key}", timeout=300)

    if result and result.returncode == 0:
        print(f"  {trait_key}: Munged successfully")
        return True
    else:
        # Try alternative column names
        cmd2 = (
            f"python3 {MUNGE} "
            f"--sumstats {raw_file}.gz "
            f"--merge-alleles {HM3_SNPS} "
            f"--out {out_prefix} "
            f"--snp SNP --a1 A1 --a2 A2 --p P --or OR --se SE --Ntotal N "
            f"--frq-filter-b 0.5"
        )
        result2 = run_cmd(cmd2, f"Munge {trait_key} (alt format)", timeout=300)
        if result2 and result2.returncode == 0:
            print(f"  {trait_key}: Munged successfully (alt format)")
            return True
        print(f"  {trait_key}: Munge FAILED")
        return False


def analysis2_genetic_correlation():
    """Run LDSC genetic correlation."""
    print("\n" + "="*60)
    print("ANALYSIS 2: LDSC GENETIC CORRELATION")
    print("="*60)

    os.makedirs(SUMSTATS_DIR, exist_ok=True)

    # Step 1: Download and munge all comparator traits
    print("\n--- Step 1: Downloading and munging comparator sumstats ---")
    munged_traits = []
    for key, info in COMPARATOR_TRAITS.items():
        if download_comparator_sumstats(key, info):
            munged_traits.append(key)

    print(f"\nMunged traits: {munged_traits} ({len(munged_traits)}/{len(COMPARATOR_TRAITS)})")

    if len(munged_traits) < 3:
        print("TOO FEW TRAITS MUNGED - cannot proceed with genetic correlation")
        return None

    # Step 2: Run LDSC rg
    sumstats_list = [PGC3_SUMSTATS] + [
        f"{SUMSTATS_DIR}/{t}.sumstats.gz" for t in munged_traits
    ]
    sumstats_str = ",".join(sumstats_list)

    print(f"\n--- Step 2: Running LDSC rg with {len(sumstats_list)} traits ---")
    cmd = (
        f"python3 {LDSC} "
        f"--rg {sumstats_str} "
        f"--ref-ld-chr {WEIGHTS} "
        f"--w-ld-chr {WEIGHTS} "
        f"--out {OUT_DIR}/scz_rg_all"
    )
    result = run_cmd(cmd, "LDSC genetic correlation", timeout=1800)

    if result and result.returncode == 0:
        results_file = f"{OUT_DIR}/scz_rg_all.log"
        if os.path.exists(results_file):
            with open(results_file) as f:
                print(f.read())
            return results_file

    print("Genetic correlation FAILED")
    return None


# ============================================================================
# ANALYSIS 3: FOREST PLOT
# ============================================================================

def analysis3_forest_plot():
    """Generate forest plot for cross-GWAS meta-enrichment."""
    print("\n" + "="*60)
    print("ANALYSIS 3: FOREST PLOT")
    print("="*60)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available")
        return None

    # Meta-enrichment data from batch_033 D30_1
    data = {
        "Pardiñas": {"neuronal": {"OR": 10.09, "k": 17, "n_total": 444, "n_marker": 95}},
        "PGC2": {"neuronal": {"OR": 1.07, "k": 1, "n_total": 128, "n_marker": 95}},
        "PGC3": {"neuronal": {"OR": 4.16, "k": 2, "n_total": 106, "n_marker": 95}},
    }

    # Compute ORs and CIs from 2x2 tables
    studies = []
    for name, gwas_data in data.items():
        for cell_type, d in gwas_data.items():
            # 2x2 table: k = overlap, n_total - k = SCZ non-marker, n_marker - k = marker non-SCZ
            k = d["k"]
            a = k  # SCZ + marker
            b = d["n_total"] - k  # SCZ + non-marker
            c = d["n_marker"] - k  # non-SCZ + marker
            n_bg = 20197 - d["n_total"] - d["n_marker"] + k
            d_val = n_bg  # non-SCZ + non-marker

            or_val = (a * d_val) / (b * c) if b * c > 0 else float('inf')

            # Woolf CI
            var_log_or = 1/a + 1/b + 1/c + 1/d_val if min(a,b,c,d_val) > 0 else 999
            ci_lo = np.exp(np.log(or_val) - 1.96 * np.sqrt(var_log_or))
            ci_hi = np.exp(np.log(or_val) + 1.96 * np.sqrt(var_log_or))

            studies.append({
                "name": name,
                "OR": or_val,
                "CI_lo": ci_lo,
                "CI_hi": ci_hi,
                "k": k,
                "n": d["n_total"],
            })

    # Meta-analytic OR (from batch_033: meta_OR=4.73)
    meta_or = 4.73

    # Create forest plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    y_positions = list(range(len(studies)))
    for i, s in enumerate(studies):
        ax.plot([s["CI_lo"], s["CI_hi"]], [i, i], 'b-', linewidth=2)
        ax.plot(s["OR"], i, 'bs', markersize=8)
        ax.text(s["CI_hi"] * 1.1, i, f'{s["name"]} (k={s["k"]}, OR={s["OR"]:.2f})',
                va='center', fontsize=9)

    # Meta-analytic estimate
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=meta_or, color='red', linestyle='--', alpha=0.5)
    ax.text(meta_or, len(studies) + 0.3, f'Meta OR={meta_or:.2f}, p=0.011',
            ha='center', fontsize=10, fontweight='bold', color='red')

    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11)
    ax.set_ylabel('')
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s["name"] for s in studies])
    ax.set_title('Neuronal Enrichment Across GWAS Datasets', fontsize=12, fontweight='bold')
    ax.set_xscale('log')

    plt.tight_layout()
    out_path = f"{OUT_DIR}/forest_plot_neuronal_enrichment.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Forest plot saved to {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LDSCORES_DIR, exist_ok=True)
    os.makedirs(SUMSTATS_DIR, exist_ok=True)

    results = {}

    # Run all analyses
    results["sldsc"] = analysis1_sldsc()
    results["rg"] = analysis2_genetic_correlation()
    results["forest"] = analysis3_forest_plot()

    # Save results summary
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump({
            "analysis1_sldsc": str(results.get("sldsc", "FAILED")),
            "analysis2_rg": str(results.get("rg", "FAILED")),
            "analysis3_forest": str(results.get("forest", "FAILED")),
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("BATCH 034 COMPLETE")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v}")
