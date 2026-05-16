#!/usr/bin/env python3
"""batch_060 E7 -- H-MAGMA developmental analysis using Won Lab annotations.

Implements brief_v2.md section E7 EXACTLY.

Steps:
  a) Download H-MAGMA annotation files from Won Lab GitHub (6 annotations).
  b) For each annotation, run MAGMA gene analysis using PGC3 SCZ GWAS sumstats.
  c) For each annotation's gene-Z output, apply gene-set OLS tests
     (EDT1-ex-B3, B3, IEG, GR, complement-MHC-excluded).
  d) Compute Spearman rho between each H-MAGMA gene-Z and standard MAGMA gene-Z.
  e) Paired bootstrap (B=1000) on beta_fetal - beta_adult for each gene set.

WHY H-MAGMA: Standard MAGMA assigns SNPs to nearest gene. H-MAGMA uses
Hi-C-derived chromatin interactions to assign distal regulatory SNPs to their
target genes, producing developmentally-specific gene scores. This tests
whether our gene sets (EDT1-ex-B3, B3, etc.) show differential enrichment
when distal enhancer-gene links are accounted for.

Source: Sey et al. 2020 (Nature Neuroscience) [lit_doi_10.1038_s41593-020-0603-0].

Output: experiments/batch_060/output/e7/results.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    B3_GENES,
    BATCH054_P05_PREDS,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    PROJECT_ROOT,
    atomic_write_json,
    bh_fdr,
    build_sub_a_frame,
    load_edt1,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    load_koopmans_ex_B3,
    load_magma_scz,
    setup_logger,
    sha256_file,
    symbols_to_ensgids,
    B060_SEED_MASTER,
)

import statsmodels.api as sm

# =============================================================================
# Constants
# =============================================================================
E7_OUTPUT_DIR = OUTPUT_DIR / "e7"
E7_DATA_DIR = PROJECT_ROOT / "data" / "hmagma"

# MAGMA binary and reference data paths (from batch_059/_common).
MAGMA_BIN = PROJECT_ROOT / "tools" / "magma_bin" / "magma"
MAGMA_1000G_EUR = PROJECT_ROOT / "tools" / "magma_bin" / "g1000_eur" / "g1000_eur"

# PGC3 SCZ GWAS p-value file (created by batch_052_C).
# WHY this specific file: batch_052_C's MAGMA log shows it used this file
# with `use=SNP,PVAL ncol=N` flags. We reuse the same input for consistency.
PGC3_PVAL_FILE = PROJECT_ROOT / "experiments" / "batch_052_C" / "work" / "magma_pval.txt"

# H-MAGMA annotation files from Won Lab GitHub.
# WHY these specific files: brief_v2 section E7 lists fetal_brain, adult_brain,
# cortical_neuron, iPSC_neuron, iPSC_astro as the 5 H-MAGMA annotations,
# plus conventional MAGMA as the 6th (already available).
# Source: https://github.com/thewonlab/H-MAGMA
# WHY these URLs: Verified via GitHub API (2026-04-24). Annotation files live
# under Input_Files/ in the Won Lab H-MAGMA repository. Filenames match the
# API listing exactly. Note: iPSC_derived_neuro has a leading space in the
# repo filename (" iPSC_derived_neuro.genes.annot") which we handle below.
HMAGMA_GITHUB_BASE = (
    "https://raw.githubusercontent.com/thewonlab/H-MAGMA/master/Input_Files/"
)
HMAGMA_ANNOTATIONS = {
    "fetal_brain": "Fetal_brain.genes.annot",
    "adult_brain": "Adult_brain.genes.annot",
    "cortical_neuron": "Cortical_Neuron.genes.annot",
    "iPSC_neuron": "%20iPSC_derived_neuro.genes.annot",  # Leading space in repo filename.
    "iPSC_astro": "iPSC_derived_astro.genes.annot",
}
# Local filenames (without URL-encoding artifacts).
HMAGMA_LOCAL_FILENAMES = {
    "fetal_brain": "Fetal_brain.genes.annot",
    "adult_brain": "Adult_brain.genes.annot",
    "cortical_neuron": "Cortical_Neuron.genes.annot",
    "iPSC_neuron": "iPSC_derived_neuro.genes.annot",
    "iPSC_astro": "iPSC_derived_astro.genes.annot",
}

# Gene sets to test (brief_v2 section E7: EDT1-ex-B3, B3, IEG, GR, complement).
IEG_SYMBOLS = [
    "FOS", "FOSB", "JUNB", "EGR1", "EGR2", "EGR3", "EGR4",
    "NR4A1", "NR4A2", "NR4A3", "ARC", "NPAS4", "BTG2", "DUSP1",
    "DUSP5", "GADD45B", "GADD45G", "ATF3", "PPP1R15A",
]
GR_SYMBOLS = [
    "FKBP5", "TSC22D3", "SGK1", "PER1", "DUSP1", "KLF15", "ZBTB16",
    "NFKBIA", "CDKN1A", "TXNIP", "DDIT4", "MT2A", "IL1R2", "VIPR1",
    "ANGPTL4", "GLUL", "PDK4", "ERRFI1", "SCNN1A", "CEBPD", "KLF9",
    "TIPARP",
]
# Complement MHC-excluded (n=24): full 28 minus C4A, C4B, C2, CFB.
COMPLEMENT_MHC_EXCLUDED_SYMBOLS = [
    "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C3", "C5", "C6", "C7",
    "C8A", "C8B", "C8G", "C9", "CFD", "CFP", "CFH", "CFI", "CR1",
    "CR2", "CD46", "CD55", "CD59", "SERPING1", "CLU",
]

# OLS covariates (same as Sub-A v2.1 primary spec, brief_v2 Shared Design).
PRIMARY_COVS = [
    "log10_gene_length", "lof_pLI",
    "log10_exp_lof_plus1", "log10_NSNPS_plus1",
]

# Bootstrap parameters.
N_BOOTSTRAP = 1000  # brief_v2 section E7: B=1000
SEED = B060_SEED_MASTER


# =============================================================================
# Helpers
# =============================================================================
def download_annotation(name: str, url_filename: str, local_filename: str,
                        data_dir: Path, logger) -> Path:
    """Download one H-MAGMA annotation file from GitHub if not already present.

    WHY download rather than bundle: H-MAGMA files are ~5-50 MB each.
    They are freely available from the Won Lab GitHub repository.
    We download to data/hmagma/ for reproducibility.

    Args:
        name: Human-readable annotation name (e.g., "fetal_brain").
        url_filename: Filename as it appears in the URL (may have URL-encoding).
        local_filename: Clean filename to save locally.
        data_dir: Directory to save the file.
    """
    local_path = data_dir / local_filename
    if local_path.exists():
        logger.info("H-MAGMA annotation %s already exists at %s", name, local_path)
        return local_path

    url = HMAGMA_GITHUB_BASE + url_filename
    logger.info("Downloading H-MAGMA annotation %s from %s", name, url)
    try:
        urllib.request.urlretrieve(url, str(local_path))
        logger.info("Downloaded %s (%d bytes)", local_filename, local_path.stat().st_size)
    except Exception as exc:
        logger.error("Failed to download %s: %s", url, exc)
        raise
    return local_path


def run_magma_gene_analysis(annot_path: Path, output_prefix: Path,
                            logger) -> Path:
    """Run MAGMA gene analysis with a given annotation file.

    Returns path to the .genes.out file.

    WHY we run MAGMA fresh for each annotation: H-MAGMA annotation files
    define different SNP-to-gene mappings (based on Hi-C data), so the gene
    analysis must be recomputed for each annotation context. The --gene-annot
    flag determines which SNPs contribute to each gene's test statistic.
    """
    genes_out = Path(str(output_prefix) + ".genes.out")
    if genes_out.exists():
        logger.info("MAGMA output already exists: %s", genes_out)
        return genes_out

    cmd = [
        str(MAGMA_BIN),
        "--bfile", str(MAGMA_1000G_EUR),
        "--gene-annot", str(annot_path),
        "--pval", str(PGC3_PVAL_FILE),
        "use=SNP,PVAL", "ncol=N",
        "--out", str(output_prefix),
    ]
    logger.info("Running MAGMA: %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600,
    )
    elapsed = time.time() - t0
    logger.info("MAGMA completed in %.1fs, returncode=%d", elapsed, result.returncode)
    if result.returncode != 0:
        logger.error("MAGMA stderr: %s", result.stderr[:2000])
        raise RuntimeError(
            f"MAGMA gene analysis failed for {annot_path.name}: "
            f"returncode={result.returncode}"
        )
    if not genes_out.exists():
        raise FileNotFoundError(
            f"MAGMA did not produce expected output: {genes_out}"
        )
    return genes_out


def fit_ols_geneset(frame: pd.DataFrame, gene_set_ensg: set[str],
                    indicator_col: str = "in_set",
                    outcome_col: str = "MAGMA_Z") -> dict:
    """OLS regression of MAGMA-Z on gene-set indicator + covariates.

    WHY OLS: This is the same Sub-A v2.1 primary point estimate used throughout
    the project (batch_058 sub_a_robust_battery.py). beta_1 on the indicator
    column measures the mean excess MAGMA-Z of genes in the set vs. background,
    controlling for gene length, pLI, expected LoF, and NSNPS.
    """
    frame = frame.copy()
    frame[indicator_col] = frame["ENSGID"].isin(gene_set_ensg).astype(int)
    n_in_set = int(frame[indicator_col].sum())

    X = frame[[indicator_col] + PRIMARY_COVS].to_numpy(dtype=float)
    y = frame[outcome_col].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    try:
        ols = sm.OLS(y, Xc).fit()
    except Exception as exc:
        return {"status": "failed", "reason": str(exc), "n_in_set": n_in_set}

    beta_1 = float(ols.params[1])
    se_1 = float(ols.bse[1])
    t_1 = float(ols.tvalues[1])
    p_two = float(ols.pvalues[1])
    p_one = float(p_two / 2.0 if t_1 > 0 else 1.0 - p_two / 2.0)
    ci_lo = beta_1 - 1.96 * se_1
    ci_hi = beta_1 + 1.96 * se_1
    return {
        "status": "ok",
        "beta_1": round(beta_1, 6),
        "se_1": round(se_1, 6),
        "t": round(t_1, 4),
        "p_one_sided": round(p_one, 6),
        "p_two_sided": round(p_two, 6),
        "ci_lo": round(ci_lo, 6),
        "ci_hi": round(ci_hi, 6),
        "n_total": int(X.shape[0]),
        "n_in_set": n_in_set,
    }


def load_hmagma_gene_z_with_nsnps(genes_out: Path, logger) -> pd.DataFrame:
    """Load H-MAGMA .genes.out with NSNPS column preserved.

    WHY preserve NSNPS: We need NSNPS from the H-MAGMA output for the
    log10_NSNPS_plus1 covariate. H-MAGMA may assign different SNPs to genes
    than standard MAGMA (due to Hi-C annotations), so using H-MAGMA's own
    NSNPS is more accurate for controlling SNP-count confounding.

    H-MAGMA annotation files use ENSGID directly (verified from file header:
    ENSG00000000419\t20:49551404:49575092\trs...). Therefore the MAGMA output
    GENE column contains ENSGIDs, NOT Entrez IDs. This simplifies the loader
    compared to the standard MAGMA pipeline which uses Entrez IDs.
    """
    df = pd.read_csv(genes_out, sep=r"\s+")
    required_min = {"GENE", "ZSTAT", "P"}
    available = set(df.columns)
    if not required_min <= available:
        raise RuntimeError(
            f"MAGMA schema drift in {genes_out}: columns={list(df.columns)}"
        )
    has_nsnps = "NSNPS" in available
    cols_to_keep = ["GENE", "ZSTAT", "P"]
    if has_nsnps:
        cols_to_keep.append("NSNPS")

    df = df[cols_to_keep].copy()

    # Detect whether GENE column contains ENSGIDs or Entrez IDs.
    # H-MAGMA annotations use ENSGIDs, standard MAGMA uses Entrez.
    sample_gene = str(df["GENE"].iloc[0]) if len(df) > 0 else ""
    is_ensgid = sample_gene.startswith("ENSG")

    if is_ensgid:
        # H-MAGMA output: GENE is already ENSGID.
        df = df.rename(columns={"GENE": "ENSGID", "ZSTAT": "MAGMA_Z",
                                "P": "MAGMA_P"})
        df["ENSGID"] = df["ENSGID"].astype(str)
        n_raw = len(df)
        logger.info(
            "H-MAGMA gene-Z loaded (ENSGID-keyed): %d genes", n_raw,
        )
    else:
        # Standard MAGMA output: GENE is Entrez. Map to ENSGID.
        df = df.rename(columns={"GENE": "entrez", "ZSTAT": "MAGMA_Z",
                                "P": "MAGMA_P"})
        df["entrez"] = df["entrez"].astype(str)

        geneloc_path = PROJECT_ROOT / "tools" / "magma_bin" / "refs" / "NCBI37.3.gene.loc"
        geneloc = pd.read_csv(
            geneloc_path, sep="\t", header=None,
            names=["entrez", "chr", "start", "end", "strand", "symbol"],
            dtype={"entrez": str, "symbol": str},
        )
        ent2sym = geneloc.drop_duplicates(subset="entrez", keep="first")[
            ["entrez", "symbol"]
        ]
        df = df.merge(ent2sym, on="entrez", how="left")

        annot = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME"]]
        sym2ensg = annot.drop_duplicates(subset="NAME", keep="first")
        df = df.merge(sym2ensg.rename(columns={"NAME": "symbol"}),
                      on="symbol", how="left")

        n_raw = len(df)
        df = df.dropna(subset=["ENSGID"]).copy()
        n_mapped = len(df)
        logger.info(
            "H-MAGMA gene-Z loaded (Entrez-keyed): %d raw, %d mapped to "
            "ENSGID (%.1f%%)",
            n_raw, n_mapped, 100 * n_mapped / max(n_raw, 1),
        )

    out_cols = ["ENSGID", "MAGMA_Z", "MAGMA_P"]
    if has_nsnps:
        out_cols.append("NSNPS")
    return df[out_cols].copy()


def build_hmagma_frame_v2(hmagma_df: pd.DataFrame, gnomad: pd.DataFrame,
                          annot_df: pd.DataFrame) -> pd.DataFrame:
    """Build regression frame using H-MAGMA's own NSNPS when available.

    WHY v2: Improved over build_hmagma_frame by using H-MAGMA's own NSNPS
    column rather than proxying from standard MAGMA. This is more accurate
    because H-MAGMA assigns different SNPs to genes via Hi-C annotations.
    """
    has_nsnps = "NSNPS" in hmagma_df.columns
    merge_cols = ["ENSGID", "MAGMA_Z"]
    if has_nsnps:
        merge_cols.append("NSNPS")

    frame = (
        hmagma_df[merge_cols]
        .merge(gnomad[["ENSGID", "lof_pLI", "lof_oe_ci_upper", "lof_exp"]],
               on="ENSGID", how="inner")
        .merge(annot_df[["ENSGID", "log10_gene_length"]], on="ENSGID", how="inner")
    )

    drop_subset = ["MAGMA_Z", "lof_pLI", "lof_exp", "log10_gene_length"]
    if has_nsnps:
        drop_subset.append("NSNPS")
    frame = frame.dropna(subset=drop_subset).copy()

    frame["log10_exp_lof_plus1"] = np.log10(
        frame["lof_exp"].astype(float) + 1.0
    )

    if has_nsnps:
        frame["log10_NSNPS_plus1"] = np.log10(
            frame["NSNPS"].astype(float) + 1.0
        )
    else:
        # Fallback: use standard MAGMA NSNPS.
        std_raw = pd.read_csv(MAGMA_SCZ_GENES_OUT, sep=r"\s+")
        std_nsnps = std_raw.rename(columns={"GENE": "ENSGID"})[["ENSGID", "NSNPS"]]
        frame = frame.merge(std_nsnps, on="ENSGID", how="left")
        median_nsnps = frame["NSNPS"].median()
        frame["NSNPS"] = frame["NSNPS"].fillna(median_nsnps)
        frame["log10_NSNPS_plus1"] = np.log10(
            frame["NSNPS"].astype(float) + 1.0
        )

    frame = frame.sort_values("ENSGID").reset_index(drop=True)
    return frame


def paired_bootstrap_delta(
    frame_fetal: pd.DataFrame, frame_adult: pd.DataFrame,
    gene_set_ensg: set[str], n_boot: int, seed: int,
    logger,
) -> dict:
    """Paired bootstrap on beta_fetal - beta_adult for one gene set.

    WHY paired: brief_v2 section E7 MEASUREMENT mandates "paired bootstrap
    on beta_fetal - beta_adult with 95% CI for the difference." Paired means
    we resample the same gene indices for both fetal and adult frames, then
    compute delta = beta_fetal(boot) - beta_adult(boot).

    WHY bootstrap rather than analytic SE: The fetal and adult beta estimates
    share the same genes and covariates (only the MAGMA-Z outcome differs),
    so their correlation structure is complex. Bootstrap naturally captures
    this dependence.
    """
    # Intersect genes present in both frames.
    common_ensg = set(frame_fetal["ENSGID"]) & set(frame_adult["ENSGID"])
    f_sub = frame_fetal[frame_fetal["ENSGID"].isin(common_ensg)].sort_values("ENSGID").reset_index(drop=True)
    a_sub = frame_adult[frame_adult["ENSGID"].isin(common_ensg)].sort_values("ENSGID").reset_index(drop=True)

    # Verify alignment.
    assert (f_sub["ENSGID"].values == a_sub["ENSGID"].values).all(), \
        "Fetal and adult frames not aligned after sort"

    n = len(f_sub)
    n_in_set = int(f_sub["ENSGID"].isin(gene_set_ensg).sum())

    if n_in_set < 3:
        return {
            "status": "too_few_genes",
            "n_common": n,
            "n_in_set": n_in_set,
        }

    # Prepare indicator and covariates (shared between fetal and adult).
    indicator = f_sub["ENSGID"].isin(gene_set_ensg).astype(int).values
    covs_f = f_sub[PRIMARY_COVS].to_numpy(dtype=float)
    covs_a = a_sub[PRIMARY_COVS].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        # Fetal OLS.
        X_f = np.column_stack([np.ones(n), indicator[idx], covs_f[idx]])
        y_f = f_sub["MAGMA_Z"].values[idx]
        try:
            beta_f, _, _, _ = np.linalg.lstsq(X_f, y_f, rcond=None)
        except np.linalg.LinAlgError:
            deltas[b] = np.nan
            continue
        # Adult OLS.
        X_a = np.column_stack([np.ones(n), indicator[idx], covs_a[idx]])
        y_a = a_sub["MAGMA_Z"].values[idx]
        try:
            beta_a, _, _, _ = np.linalg.lstsq(X_a, y_a, rcond=None)
        except np.linalg.LinAlgError:
            deltas[b] = np.nan
            continue
        # beta[1] is the indicator coefficient.
        deltas[b] = beta_f[1] - beta_a[1]

    valid = deltas[np.isfinite(deltas)]
    if len(valid) < n_boot * 0.5:
        return {
            "status": "failed",
            "reason": f"Too many NaN bootstraps: {n_boot - len(valid)}/{n_boot}",
            "n_common": n,
            "n_in_set": n_in_set,
        }

    ci_lo = float(np.percentile(valid, 2.5))
    ci_hi = float(np.percentile(valid, 97.5))
    mean_delta = float(np.mean(valid))
    excludes_zero = bool(ci_lo > 0 or ci_hi < 0)

    return {
        "status": "ok",
        "mean_delta_fetal_minus_adult": round(mean_delta, 6),
        "ci_lo_2.5": round(ci_lo, 6),
        "ci_hi_97.5": round(ci_hi, 6),
        "excludes_zero": excludes_zero,
        "n_valid_boots": int(len(valid)),
        "n_common": n,
        "n_in_set": n_in_set,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="E7: H-MAGMA developmental")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading annotations (use existing)")
    parser.add_argument("--skip-magma", action="store_true",
                        help="Skip MAGMA runs (use existing outputs)")
    args = parser.parse_args()

    E7_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("e7_hmagma", LOGS_DIR / "e7_hmagma.log")
    logger.info("=== E7 H-MAGMA developmental analysis ===")
    t0 = time.time()

    results: dict = {
        "experiment": "e7_hmagma",
        "brief": "brief_v2.md section E7",
        "seed": SEED,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ------------------------------------------------------------------
    # Pre-checks: verify MAGMA binary and reference data exist.
    # ------------------------------------------------------------------
    blockers = []
    if not MAGMA_BIN.exists():
        blockers.append(f"MAGMA binary not found: {MAGMA_BIN}")
    if not Path(str(MAGMA_1000G_EUR) + ".bed").exists():
        blockers.append(f"1000G EUR reference not found: {MAGMA_1000G_EUR}.bed")
    if not PGC3_PVAL_FILE.exists():
        blockers.append(f"PGC3 SCZ pval file not found: {PGC3_PVAL_FILE}")

    if blockers:
        results["verdict"] = "BLOCKED"
        results["blockers"] = blockers
        logger.error("BLOCKED: %s", blockers)
        atomic_write_json(results, E7_OUTPUT_DIR / "results.json")
        return

    # ------------------------------------------------------------------
    # Step (a): Download H-MAGMA annotation files.
    # ------------------------------------------------------------------
    logger.info("Step (a): Downloading H-MAGMA annotations...")
    annotation_paths: dict[str, Path] = {}
    download_failures: dict[str, str] = {}

    for annot_name, url_filename in HMAGMA_ANNOTATIONS.items():
        local_filename = HMAGMA_LOCAL_FILENAMES[annot_name]
        if args.skip_download and (E7_DATA_DIR / local_filename).exists():
            annotation_paths[annot_name] = E7_DATA_DIR / local_filename
            continue
        try:
            path = download_annotation(
                annot_name, url_filename, local_filename,
                E7_DATA_DIR, logger,
            )
            annotation_paths[annot_name] = path
        except Exception as exc:
            download_failures[annot_name] = str(exc)
            logger.error("Failed to download %s: %s", annot_name, exc)

    results["annotations_downloaded"] = {
        k: str(v) for k, v in annotation_paths.items()
    }
    if download_failures:
        results["download_failures"] = download_failures

    if not annotation_paths:
        results["verdict"] = "BLOCKED"
        results["blockers"] = ["All H-MAGMA annotation downloads failed"]
        logger.error("BLOCKED: No annotations available")
        atomic_write_json(results, E7_OUTPUT_DIR / "results.json")
        return

    # ------------------------------------------------------------------
    # Step (b): Run MAGMA gene analysis for each annotation.
    # ------------------------------------------------------------------
    logger.info("Step (b): Running MAGMA gene analysis...")
    hmagma_outputs: dict[str, Path] = {}
    magma_failures: dict[str, str] = {}

    for annot_name, annot_path in annotation_paths.items():
        output_prefix = E7_OUTPUT_DIR / f"hmagma_{annot_name}"
        if args.skip_magma and Path(str(output_prefix) + ".genes.out").exists():
            hmagma_outputs[annot_name] = Path(str(output_prefix) + ".genes.out")
            logger.info("Using existing MAGMA output for %s", annot_name)
            continue
        try:
            genes_out = run_magma_gene_analysis(annot_path, output_prefix, logger)
            hmagma_outputs[annot_name] = genes_out
        except Exception as exc:
            magma_failures[annot_name] = str(exc)
            logger.error("MAGMA failed for %s: %s", annot_name, exc)

    results["magma_outputs"] = {k: str(v) for k, v in hmagma_outputs.items()}
    if magma_failures:
        results["magma_failures"] = magma_failures

    if not hmagma_outputs:
        results["verdict"] = "BLOCKED"
        results["blockers"] = ["All MAGMA runs failed"]
        logger.error("BLOCKED: No MAGMA outputs")
        atomic_write_json(results, E7_OUTPUT_DIR / "results.json")
        return

    # ------------------------------------------------------------------
    # Load shared resources.
    # ------------------------------------------------------------------
    logger.info("Loading shared resources (gnomAD, gene_annot, gene sets)...")
    gnomad = load_gnomad_per_brief_v2()
    annot_df = load_gene_annot()

    # Load standard SCZ MAGMA gene-Z for Spearman comparison.
    std_magma = load_magma_scz()

    # Prepare gene sets (symbol -> ENSGID).
    edt1_symbols = load_edt1()
    edt1_ex_b3_symbols = edt1_symbols - set(B3_GENES)
    koopmans_ex_b3_symbols = load_koopmans_ex_B3()
    # EDT1-ex-B3 is defined as EDT1 genes minus B3. The brief_v2 gene-set
    # tests for E7 use "EDT1-ex-B3" which is the same as what we compute here.
    # However, load_koopmans_ex_B3 returns the Koopmans SynGO set minus B3.
    # Brief_v2 §E7 says "Apply gene-set tests (EDT1-ex-B3, B3, IEG, GR,
    # complement)". EDT1-ex-B3 = EDT1 minus B3 genes. Let's use that.

    gene_sets: dict[str, set[str]] = {}
    gene_set_mapping_info: dict[str, dict] = {}

    # EDT1-ex-B3.
    edt1_ex_b3_ensg, edt1_ex_b3_map = symbols_to_ensgids(edt1_ex_b3_symbols)
    gene_sets["EDT1_ex_B3"] = edt1_ex_b3_ensg
    gene_set_mapping_info["EDT1_ex_B3"] = {
        "n_symbols": len(edt1_ex_b3_symbols),
        "n_mapped": len(edt1_ex_b3_ensg),
    }

    # B3.
    b3_ensg, b3_map = symbols_to_ensgids(set(B3_GENES))
    gene_sets["B3"] = b3_ensg
    gene_set_mapping_info["B3"] = {
        "n_symbols": len(B3_GENES),
        "n_mapped": len(b3_ensg),
    }

    # IEG rPRGs.
    ieg_ensg, ieg_map = symbols_to_ensgids(set(IEG_SYMBOLS))
    gene_sets["IEG_rPRGs"] = ieg_ensg
    gene_set_mapping_info["IEG_rPRGs"] = {
        "n_symbols": len(IEG_SYMBOLS),
        "n_mapped": len(ieg_ensg),
        "unmapped": sorted(set(IEG_SYMBOLS) - set(ieg_map.keys())),
    }

    # GR targets.
    gr_ensg, gr_map = symbols_to_ensgids(set(GR_SYMBOLS))
    gene_sets["GR_targets"] = gr_ensg
    gene_set_mapping_info["GR_targets"] = {
        "n_symbols": len(GR_SYMBOLS),
        "n_mapped": len(gr_ensg),
        "unmapped": sorted(set(GR_SYMBOLS) - set(gr_map.keys())),
    }

    # Complement MHC-excluded.
    comp_ensg, comp_map = symbols_to_ensgids(set(COMPLEMENT_MHC_EXCLUDED_SYMBOLS))
    gene_sets["complement_MHC_excluded"] = comp_ensg
    gene_set_mapping_info["complement_MHC_excluded"] = {
        "n_symbols": len(COMPLEMENT_MHC_EXCLUDED_SYMBOLS),
        "n_mapped": len(comp_ensg),
        "unmapped": sorted(
            set(COMPLEMENT_MHC_EXCLUDED_SYMBOLS) - set(comp_map.keys())
        ),
    }

    results["gene_set_mapping"] = gene_set_mapping_info
    logger.info("Gene sets prepared: %s",
                {k: len(v) for k, v in gene_sets.items()})

    # ------------------------------------------------------------------
    # Step (c): Gene-set OLS tests per annotation.
    # ------------------------------------------------------------------
    logger.info("Step (c): Gene-set OLS tests per annotation...")
    per_annotation_results: dict[str, dict] = {}

    # Also load standard MAGMA frame for comparison.
    for annot_name, genes_out_path in hmagma_outputs.items():
        logger.info("Processing annotation: %s", annot_name)
        hmagma_df = load_hmagma_gene_z_with_nsnps(genes_out_path, logger)
        frame = build_hmagma_frame_v2(hmagma_df, gnomad, annot_df)
        logger.info("  Regression frame: N=%d genes", len(frame))

        annot_results: dict = {
            "n_genes": len(frame),
            "gene_set_tests": {},
        }

        for gs_name, gs_ensg in gene_sets.items():
            ols_result = fit_ols_geneset(frame, gs_ensg)
            annot_results["gene_set_tests"][gs_name] = ols_result
            if ols_result["status"] == "ok":
                logger.info(
                    "  %s: beta=%.4f (SE=%.4f), p_one=%.4g, n_in_set=%d",
                    gs_name, ols_result["beta_1"], ols_result["se_1"],
                    ols_result["p_one_sided"], ols_result["n_in_set"],
                )
            else:
                logger.warning("  %s: %s", gs_name, ols_result.get("reason"))

        per_annotation_results[annot_name] = annot_results

    # Also run on standard MAGMA for comparison.
    logger.info("Processing standard MAGMA (reference)...")
    std_frame = build_hmagma_frame_v2(std_magma, gnomad, annot_df)
    std_results: dict = {
        "n_genes": len(std_frame),
        "gene_set_tests": {},
    }
    for gs_name, gs_ensg in gene_sets.items():
        ols_result = fit_ols_geneset(std_frame, gs_ensg)
        std_results["gene_set_tests"][gs_name] = ols_result
        if ols_result["status"] == "ok":
            logger.info(
                "  standard/%s: beta=%.4f (SE=%.4f), p_one=%.4g, n_in_set=%d",
                gs_name, ols_result["beta_1"], ols_result["se_1"],
                ols_result["p_one_sided"], ols_result["n_in_set"],
            )
    per_annotation_results["standard_magma"] = std_results

    results["per_annotation"] = per_annotation_results

    # ------------------------------------------------------------------
    # Step (d): Spearman rho between each H-MAGMA gene-Z and standard MAGMA.
    # ------------------------------------------------------------------
    logger.info("Step (d): Spearman rho vs standard MAGMA...")
    spearman_results: dict[str, dict] = {}

    for annot_name, genes_out_path in hmagma_outputs.items():
        hmagma_df = load_hmagma_gene_z_with_nsnps(genes_out_path, logger)
        # Inner join on ENSGID to get paired gene-Z values.
        merged = pd.merge(
            std_magma[["ENSGID", "MAGMA_Z"]].rename(columns={"MAGMA_Z": "Z_standard"}),
            hmagma_df[["ENSGID", "MAGMA_Z"]].rename(columns={"MAGMA_Z": "Z_hmagma"}),
            on="ENSGID", how="inner",
        )
        n_common = len(merged)
        if n_common < 100:
            spearman_results[annot_name] = {
                "status": "too_few_genes",
                "n_common": n_common,
            }
            continue

        rho, pval = stats.spearmanr(merged["Z_standard"], merged["Z_hmagma"])
        spearman_results[annot_name] = {
            "status": "ok",
            "spearman_rho": round(float(rho), 6),
            "spearman_p": float(pval),
            "n_common": n_common,
        }
        logger.info(
            "  %s vs standard: rho=%.4f, p=%.2e, n=%d",
            annot_name, rho, pval, n_common,
        )

    results["spearman_vs_standard"] = spearman_results

    # ------------------------------------------------------------------
    # Step (e): Paired bootstrap beta_fetal - beta_adult for each gene set.
    # ------------------------------------------------------------------
    logger.info("Step (e): Paired bootstrap delta (fetal - adult)...")
    bootstrap_results: dict[str, dict] = {}

    # Check that we have both fetal_brain and adult_brain.
    if "fetal_brain" in hmagma_outputs and "adult_brain" in hmagma_outputs:
        fetal_df = load_hmagma_gene_z_with_nsnps(
            hmagma_outputs["fetal_brain"], logger
        )
        adult_df = load_hmagma_gene_z_with_nsnps(
            hmagma_outputs["adult_brain"], logger
        )
        fetal_frame = build_hmagma_frame_v2(fetal_df, gnomad, annot_df)
        adult_frame = build_hmagma_frame_v2(adult_df, gnomad, annot_df)

        for gs_name, gs_ensg in gene_sets.items():
            logger.info("  Bootstrap delta for %s...", gs_name)
            boot_result = paired_bootstrap_delta(
                fetal_frame, adult_frame, gs_ensg,
                n_boot=N_BOOTSTRAP, seed=SEED, logger=logger,
            )
            bootstrap_results[gs_name] = boot_result
            if boot_result["status"] == "ok":
                logger.info(
                    "    delta=%.4f [%.4f, %.4f] excludes_zero=%s",
                    boot_result["mean_delta_fetal_minus_adult"],
                    boot_result["ci_lo_2.5"], boot_result["ci_hi_97.5"],
                    boot_result["excludes_zero"],
                )
    else:
        missing = []
        if "fetal_brain" not in hmagma_outputs:
            missing.append("fetal_brain")
        if "adult_brain" not in hmagma_outputs:
            missing.append("adult_brain")
        bootstrap_results["status"] = "BLOCKED"
        bootstrap_results["reason"] = (
            f"Missing annotations for bootstrap: {missing}"
        )
        logger.warning("Bootstrap BLOCKED: missing %s", missing)

    results["paired_bootstrap_fetal_minus_adult"] = bootstrap_results

    # ------------------------------------------------------------------
    # Decision rule (brief_v2 section E7 DECISION RULE).
    # ------------------------------------------------------------------
    logger.info("Applying decision rule...")

    # Check |beta_fetal - beta_standard| for EDT1-ex-B3.
    verdict_parts: list[str] = []

    if ("fetal_brain" in per_annotation_results
            and "standard_magma" in per_annotation_results):
        fetal_gs = per_annotation_results["fetal_brain"]["gene_set_tests"]
        std_gs = per_annotation_results["standard_magma"]["gene_set_tests"]

        if (fetal_gs.get("EDT1_ex_B3", {}).get("status") == "ok"
                and std_gs.get("EDT1_ex_B3", {}).get("status") == "ok"):
            beta_fetal = fetal_gs["EDT1_ex_B3"]["beta_1"]
            beta_std = std_gs["EDT1_ex_B3"]["beta_1"]
            delta = abs(beta_fetal - beta_std)

            if delta > 0.5:
                verdict_parts.append(
                    f"DEVELOPMENTAL_MATTERS: |beta_fetal - beta_standard| = "
                    f"{delta:.3f} > 0.5 for EDT1_ex_B3"
                )
            elif delta < 0.2:
                verdict_parts.append(
                    f"ANNOTATION_IRRELEVANT: |beta_fetal - beta_standard| = "
                    f"{delta:.3f} < 0.2 for EDT1_ex_B3"
                )
            else:
                verdict_parts.append(
                    f"INTERMEDIATE: |beta_fetal - beta_standard| = "
                    f"{delta:.3f} (between 0.2 and 0.5) for EDT1_ex_B3"
                )

    # Check paired bootstrap for developmental differential.
    if "EDT1_ex_B3" in bootstrap_results:
        br = bootstrap_results["EDT1_ex_B3"]
        if br.get("status") == "ok" and br.get("excludes_zero"):
            verdict_parts.append(
                "DEVELOPMENTAL_DIFFERENTIAL: bootstrap CI on "
                "beta_fetal - beta_adult excludes 0 for EDT1_ex_B3"
            )

    if not verdict_parts:
        verdict_parts.append("INCOMPLETE: insufficient data to apply decision rule")

    results["verdict_components"] = verdict_parts
    # Summarize verdict.
    if any("DEVELOPMENTAL_MATTERS" in v for v in verdict_parts):
        results["verdict"] = "DEVELOPMENTAL_MATTERS"
    elif any("ANNOTATION_IRRELEVANT" in v for v in verdict_parts):
        results["verdict"] = "ANNOTATION_IRRELEVANT"
    elif any("BLOCKED" in str(v) for v in verdict_parts):
        results["verdict"] = "BLOCKED"
    else:
        results["verdict"] = "INTERMEDIATE"

    results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["elapsed_seconds"] = round(time.time() - t0, 1)

    atomic_write_json(results, E7_OUTPUT_DIR / "results.json")
    logger.info("E7 complete. Verdict: %s. Elapsed: %.1fs",
                results["verdict"], results["elapsed_seconds"])


if __name__ == "__main__":
    main()
