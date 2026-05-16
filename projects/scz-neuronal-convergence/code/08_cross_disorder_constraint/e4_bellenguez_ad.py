#!/usr/bin/env python3
"""batch_059 E4 — Bellenguez 2022 AD independent-cohort replication.

Implements brief_v2.md §4 EXACTLY.

Overview (WHY):
  iter_058 F058_06 cross-disorder arm reports AD β=+0.31 on EDT1-ex-B3.
  Bellenguez 2022 (GCST90027158, N_eff ≈ 382k) provides an independent
  AD cohort ~2× the power of the existing pipeline (Jansen-like N ≈ 200k).
  E4 tests whether the AD signal replicates.

  Pre-steps (in order, each with fail-fast gates):
    1. Munge GRCh38 sumstats → SNP-loc.
    2. Liftover GRCh38 → GRCh37 (pyliftover; 1000G EUR is GRCh37).
    3. Run MAGMA v1.10 annotation + gene analysis (35kb-up/10kb-down,
       snp-wise=mean), 1000G EUR .bfile.
    4. QC: Bellenguez vs existing AD gene-Z Spearman ρ on shared ENSGIDs
       ≥ 0.6; ENSGID overlap ≥ 90%. If fails → E4_DEFERRED.

  Primary:
    - Sub-C v2.1 battery on EDT1-ex-B3 with AD Bellenguez gene-Z.
  Secondary:
    - Same battery on B3, EDT1-FULL, Koopmans-ex-B3-ex-EDT1.

  E2/E4 coupling (pre-reg):
    - If E2 classifies AD × EDT1-ex-B3 as LENGTH_ARTIFACT, relabel AD_REPLICATED
      as REPLICATED_LENGTH_ARTIFACT.

  Wall-clock guard: abort after 90min with E4_DEFERRED; save partial state.

Outputs:
  experiments/batch_059/output/e4/results.json
  experiments/batch_059/output/e4/magma_gene_z.tsv
  experiments/batch_059/output/e4/liftover_stats.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    B3_GENES,
    BATCH055B_WORK,
    BELLENGUEZ_SUMSTATS,
    BH_Q,
    DISORDERS,
    E4_AD_HYPER_LO,
    E4_AD_INCONSISTENT_HI_BAND,
    E4_AD_INCONSISTENT_LO_BAND,
    E4_AD_NOT_REPLICATED_ABS_CAP,
    E4_AD_REPLICATED_HI,
    E4_AD_REPLICATED_LO,
    E4_AD_WRONG_DIRECTION_CAP,
    E4_ENSG_OVERLAP_MIN,
    E4_GENE_Z_SPEARMAN_MIN,
    E4_MAX_WALL_MIN,
    E4_R1_HI,
    E4_R1_LO,
    E4_R1_TARGET,
    E4_R1_TOLERANCE,
    E4_SNP_MATCH_RATE_MIN,
    E4_WALL_CAP_SECONDS,
    N_CASES_BELLENGUEZ_2022,
    N_CONTROLS_BELLENGUEZ_2022,
    N_EFFECTIVE_BELLENGUEZ_2022,
    EXISTING_ALZHEIMERS_MAGMA,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_1000G_EUR,
    MAGMA_ANNOT_WINDOW_DOWN_KB,
    MAGMA_ANNOT_WINDOW_UP_KB,
    MAGMA_BIN,
    MAGMA_GENE_MODEL,
    MAGMA_GENELOC_GRCH37,
    MAGMA_GENELOC_GRCH38,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    SEED_E4_BOOT,
    atomic_write_json,
    atomic_write_text,
    bh_fdr,
    build_sub_a_frame,
    classify_disorder_v2,
    compute_dfbetas_cooks,
    fit_tukey_biweight,
    load_edt1,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    load_koopmans_ex_B3,
    setup_logger,
    sha256_file,
    symbols_to_ensgids,
)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BATCH058_SCRIPTS = (
    Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
    / "experiments" / "batch_058" / "scripts"
)
sys.path.insert(0, str(BATCH058_SCRIPTS))
from sub_a_robust_battery import (  # noqa: E402
    fit_huber, fit_ols, fit_rank_magma_ols,
    influential_outlier_reconciliation,
)


class WallClockGuard:
    """Abort E4 after max_seconds with E4_DEFERRED.

    WHY a guard object: the brief requires a hard 90-min wall cap. We check
    `guard.expired()` at each sub-step boundary so callers can save partial
    state before raising.
    """
    def __init__(self, max_seconds: float) -> None:
        self.start = time.time()
        self.max_seconds = float(max_seconds)

    def remaining(self) -> float:
        return self.max_seconds - (time.time() - self.start)

    def expired(self) -> bool:
        return self.remaining() <= 0.0


def munge_bellenguez_sumstats(guard: WallClockGuard, logger,
                                 work_dir: Path,
                                 smoke_limit: int = 0) -> dict:
    """Pre-step 1: parse Bellenguez GRCh38 TSV.gz → SNP-loc DataFrame.

    Columns in source (verified): variant_id, p_value, chromosome,
    base_pair_location, effect_allele, other_allele, [...], beta,
    standard_error, n_cases, n_controls, ..., variant_alternate_id.

    Returns {"snpdf": DataFrame[SNP, CHR, POS_hg38, P], "sha256": ...}.
    """
    if not BELLENGUEZ_SUMSTATS.exists():
        raise FileNotFoundError(
            f"Bellenguez sumstats missing: {BELLENGUEZ_SUMSTATS}"
        )
    logger.info("Bellenguez: parsing %s", BELLENGUEZ_SUMSTATS)
    sha = sha256_file(BELLENGUEZ_SUMSTATS)
    # Stream-load with pandas in chunks to cap memory at 21M rows.
    chunks = []
    # Use c engine with gzip decompression + subset columns.
    usecols = ["variant_id", "chromosome", "base_pair_location", "p_value"]
    dtypes = {"variant_id": str, "chromosome": str,
              "base_pair_location": "int64", "p_value": "float64"}
    iter_ = pd.read_csv(
        BELLENGUEZ_SUMSTATS, sep="\t", compression="gzip",
        usecols=usecols, dtype=dtypes, chunksize=2_000_000,
    )
    total_rows = 0
    for i_c, chunk in enumerate(iter_):
        if guard.expired():
            return {"status": "E4_DEFERRED",
                    "reason": "wall-clock during munge"}
        chunk = chunk.rename(columns={
            "variant_id": "SNP",
            "chromosome": "CHR",
            "base_pair_location": "POS_hg38",
            "p_value": "P",
        })
        # Drop non-autosomal (X/Y/MT) for 1000G EUR panel.
        chunk = chunk[chunk["CHR"].isin(
            [str(x) for x in range(1, 23)]
        )]
        chunk = chunk.dropna(subset=["SNP", "CHR", "POS_hg38", "P"])
        chunks.append(chunk)
        total_rows += len(chunk)
        logger.info("  chunk %d: %d rows (running total=%d)",
                     i_c, len(chunk), total_rows)
        if smoke_limit and total_rows >= smoke_limit:
            break
    snpdf = pd.concat(chunks, ignore_index=True)
    logger.info("Bellenguez: parsed %d autosomal SNPs", len(snpdf))
    return {"status": "ok", "snpdf": snpdf, "sha256": sha,
            "n_rows": int(len(snpdf))}


def liftover_hg38_to_hg19(snpdf: pd.DataFrame, guard: WallClockGuard,
                           logger) -> dict:
    """Pre-step 2: liftover GRCh38 → GRCh37 via pyliftover.

    WHY pyliftover: brief_v2 §4 L234 specifies it (installed and verified
    at 0.4.1). 1000G EUR reference is GRCh37 only.

    Returns {"snpdf_hg19": lifted DataFrame, "pct_lost": ..., ...}.
    """
    from pyliftover import LiftOver
    lo = LiftOver("hg38", "hg19")
    n0 = len(snpdf)
    chrs = snpdf["CHR"].astype(str).to_numpy()
    positions = snpdf["POS_hg38"].to_numpy(dtype=int)
    new_chrs = np.empty(n0, dtype=object)
    new_pos = np.full(n0, -1, dtype=np.int64)
    for i in range(n0):
        if i % 2_000_000 == 0:
            if guard.expired():
                return {"status": "E4_DEFERRED",
                        "reason": "wall-clock during liftover"}
            logger.info("  liftover progress: %d/%d", i, n0)
        chrom = f"chr{chrs[i]}"
        conv = lo.convert_coordinate(chrom, int(positions[i]))
        if conv:
            new_chrs[i] = conv[0][0].replace("chr", "")
            new_pos[i] = int(conv[0][1])
    keep = new_pos >= 0
    lost = int(n0 - keep.sum())
    pct_lost = float(lost / n0) if n0 else float("nan")
    snpdf_hg19 = snpdf.loc[keep].copy()
    snpdf_hg19["CHR"] = new_chrs[keep]
    snpdf_hg19["POS_hg19"] = new_pos[keep]
    logger.info("Liftover: kept %d/%d (pct_lost=%.4f)",
                 int(keep.sum()), n0, pct_lost)
    return {
        "status": "ok",
        "snpdf_hg19": snpdf_hg19,
        "pct_lost": pct_lost,
        "n_in": int(n0),
        "n_out": int(keep.sum()),
    }


def write_magma_inputs(snpdf_hg19: pd.DataFrame, work_dir: Path,
                        logger) -> dict:
    """Write MAGMA SNP-loc + p-value files (GRCh37 coordinates).

    MAGMA SNP-loc format (space/tab-separated): SNP CHR POS
    MAGMA p-value file: SNP P
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    snp_loc_path = work_dir / "bellenguez.snp_loc"
    snp_pval_path = work_dir / "bellenguez.pvals"
    snpdf_hg19 = snpdf_hg19.drop_duplicates(subset="SNP", keep="first")
    loc_df = snpdf_hg19[["SNP", "CHR", "POS_hg19"]].copy()
    loc_df.to_csv(snp_loc_path, sep="\t", index=False, header=False)
    p_df = snpdf_hg19[["SNP", "P"]].copy()
    p_df.to_csv(snp_pval_path, sep="\t", index=False, header=False)
    logger.info("MAGMA inputs written: %d SNPs → %s / %s",
                 len(snpdf_hg19), snp_loc_path, snp_pval_path)
    return {
        "snp_loc_path": str(snp_loc_path),
        "snp_pval_path": str(snp_pval_path),
        "n_snps": int(len(snpdf_hg19)),
    }


def run_magma(snp_loc: Path, snp_pval: Path, work_dir: Path,
               n_cases: int, n_controls: int,
               guard: WallClockGuard, logger) -> dict:
    """Run MAGMA annotation + gene analysis.

    Pipeline (from batch_055_B conventions):
      1. magma --annotate window=35,10 --snp-loc <snp_loc>
         --gene-loc <NCBI37.3.gene.loc> --out <prefix>
      2. magma --bfile <1000G EUR> --pval <snp_pval> N=<N_eff>
         --gene-annot <prefix>.genes.annot --out <prefix>

    WHY this pipeline: cardinal Rule 1 — matches the MAGMA v1.10 default
    pipeline used by batch_055_B and brief_v2 L235.

    M3 audit fix: N_effective uses the Bellenguez 2022 paper-reported value
    (N_EFFECTIVE_BELLENGUEZ_2022 = 382,188) rather than a 100-row median of
    the per-SNP n_cases/n_controls columns. WHY: paper-reported values are
    canonical and reproducible; per-SNP medians can drift with which rows
    are sampled (e.g., filtering / chunking order).
    Citation: Bellenguez C et al. 2022 Nature Genetics 54:412-436
    [lit_doi_10.1038_s41588-022-01024-z].

    MN5 audit fix: replaced `subprocess.call` with `subprocess.run
    (capture_output=True)` so stderr is logged on failure for diagnostics.

    Returns dict with gene-Z output path + diagnostics.
    """
    if guard.expired():
        return {"status": "E4_DEFERRED", "reason": "wall-clock before MAGMA"}
    prefix = work_dir / "bellenguez_magma"
    # Step 1: annotate.
    annot_cmd = [
        str(MAGMA_BIN), "--annotate",
        f"window={MAGMA_ANNOT_WINDOW_UP_KB},{MAGMA_ANNOT_WINDOW_DOWN_KB}",
        "--snp-loc", str(snp_loc),
        "--gene-loc", str(MAGMA_GENELOC_GRCH37),
        "--out", str(prefix),
    ]
    logger.info("MAGMA annotate: %s", " ".join(annot_cmd))
    try:
        result = subprocess.run(
            annot_cmd, timeout=guard.remaining(),
            capture_output=True, text=True, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "E4_DEFERRED", "reason": "annotate timeout"}
    if result.returncode != 0:
        logger.error("MAGMA annotate stderr:\n%s", result.stderr)
        logger.error("MAGMA annotate stdout:\n%s", result.stdout)
        return {
            "status": "failed",
            "reason": f"annotate rc={result.returncode}",
            "stderr_tail": (result.stderr or "")[-1000:],
        }

    # Step 2: gene-based. M3 audit fix: use paper-reported N_effective.
    # The n_cases/n_controls params are retained for auditor traceability
    # (they record which values were observed in the sumstats), but they
    # are NOT used to derive N_eff for MAGMA — we use the paper constant.
    n_effective = int(N_EFFECTIVE_BELLENGUEZ_2022)
    gene_cmd = [
        str(MAGMA_BIN),
        "--bfile", str(MAGMA_1000G_EUR),
        "--pval", str(snp_pval), f"N={n_effective}",
        "--gene-annot", str(prefix) + ".genes.annot",
        "--gene-model", MAGMA_GENE_MODEL,
        "--out", str(prefix),
    ]
    logger.info("MAGMA gene-based: %s", " ".join(gene_cmd))
    try:
        result = subprocess.run(
            gene_cmd, timeout=guard.remaining(),
            capture_output=True, text=True, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "E4_DEFERRED", "reason": "gene-analysis timeout"}
    if result.returncode != 0:
        logger.error("MAGMA gene-based stderr:\n%s", result.stderr)
        logger.error("MAGMA gene-based stdout:\n%s", result.stdout)
        return {
            "status": "failed",
            "reason": f"gene rc={result.returncode}",
            "stderr_tail": (result.stderr or "")[-1000:],
        }
    genes_out = Path(str(prefix) + ".genes.out")
    if not genes_out.exists():
        return {"status": "failed", "reason": "genes.out missing"}
    return {
        "status": "ok",
        "genes_out_path": str(genes_out),
        "n_effective_N": n_effective,
        "n_effective_source": "Bellenguez 2022 paper-reported",
        "n_cases_observed": int(n_cases),
        "n_controls_observed": int(n_controls),
        "n_cases_paper": int(N_CASES_BELLENGUEZ_2022),
        "n_controls_paper": int(N_CONTROLS_BELLENGUEZ_2022),
    }


def load_bellenguez_gene_z(genes_out_path: Path, logger) -> pd.DataFrame:
    """Load MAGMA `.genes.out` → DataFrame[ENSGID, MAGMA_Z].

    MAGMA gene-loc uses Entrez GENE IDs; we map to ENSGID via gene_annot.
    WHY Entrez-keyed: NCBI37.3.gene.loc uses Entrez IDs as the first column
    (verified head of file). For symbol-based ENSGID mapping, we use the
    6th column (symbol) and `load_gene_annot()`'s NAME↔ENSGID mapping.
    """
    if not genes_out_path.exists():
        raise FileNotFoundError(f"genes.out missing: {genes_out_path}")
    df = pd.read_csv(genes_out_path, sep=r"\s+")
    if not {"GENE", "ZSTAT"} <= set(df.columns):
        raise RuntimeError(f"MAGMA schema drift: {list(df.columns)}")
    # Read gene-loc (Entrez → symbol mapping).
    geneloc = pd.read_csv(
        MAGMA_GENELOC_GRCH37, sep="\t", header=None,
        names=["entrez", "chr", "start", "end", "strand", "symbol"],
        dtype={"entrez": str, "symbol": str},
    )
    ent2sym = geneloc.drop_duplicates(subset="entrez", keep="first")[
        ["entrez", "symbol"]
    ]
    df = df.rename(columns={"GENE": "entrez", "ZSTAT": "MAGMA_Z"})
    df["entrez"] = df["entrez"].astype(str)
    df = df.merge(ent2sym, on="entrez", how="left")
    annot = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME"]]
    sym2ensg = annot.drop_duplicates(subset="NAME", keep="first").rename(
        columns={"NAME": "symbol"}
    )
    df = df.merge(sym2ensg, on="symbol", how="left")
    df = df.dropna(subset=["ENSGID"]).drop_duplicates(subset="ENSGID",
                                                         keep="first")
    logger.info("Bellenguez gene-Z: %d ENSGIDs mapped", len(df))
    return df[["ENSGID", "MAGMA_Z"]].reset_index(drop=True)


def qc_against_existing_ad(bellenguez_df: pd.DataFrame, logger) -> dict:
    """QC: ENSGID overlap + Spearman ρ against existing AD pipeline."""
    if not EXISTING_ALZHEIMERS_MAGMA.exists():
        return {"status": "failed",
                "reason": f"existing AD MAGMA missing: {EXISTING_ALZHEIMERS_MAGMA}"}
    ex = pd.read_csv(EXISTING_ALZHEIMERS_MAGMA, sep=r"\s+")
    ex = ex.rename(columns={"GENE": "entrez", "ZSTAT": "MAGMA_Z"})
    ex["entrez"] = ex["entrez"].astype(str)
    # Map Entrez → ENSGID for existing pipeline.
    geneloc = pd.read_csv(
        MAGMA_GENELOC_GRCH37, sep="\t", header=None,
        names=["entrez", "chr", "start", "end", "strand", "symbol"],
        dtype={"entrez": str, "symbol": str},
    )
    ent2sym = geneloc.drop_duplicates(subset="entrez", keep="first")[
        ["entrez", "symbol"]
    ]
    ex = ex.merge(ent2sym, on="entrez", how="left")
    annot = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME"]]
    sym2ensg = annot.drop_duplicates(subset="NAME", keep="first").rename(
        columns={"NAME": "symbol"}
    )
    ex = ex.merge(sym2ensg, on="symbol", how="left").dropna(
        subset=["ENSGID"]
    )
    # Inner-join on ENSGID.
    merged = bellenguez_df.rename(
        columns={"MAGMA_Z": "MAGMA_Z_bellenguez"}
    ).merge(
        ex.rename(columns={"MAGMA_Z": "MAGMA_Z_existing"})[
            ["ENSGID", "MAGMA_Z_existing"]
        ],
        on="ENSGID", how="inner",
    )
    n_bell = len(bellenguez_df)
    n_ex = len(ex)
    overlap = len(merged) / max(n_bell, 1)
    if len(merged) < 10:
        return {"status": "failed", "reason": f"overlap only {len(merged)}"}
    rho, pval = spearmanr(
        merged["MAGMA_Z_bellenguez"], merged["MAGMA_Z_existing"]
    )
    logger.info("QC: Bellenguez n=%d existing n=%d overlap=%d (%.3f) "
                 "Spearman ρ=%.3f",
                 n_bell, n_ex, len(merged), overlap, rho)
    return {
        "status": "ok",
        "n_bellenguez": int(n_bell),
        "n_existing": int(n_ex),
        "n_overlap": int(len(merged)),
        "ensg_overlap_fraction": float(overlap),
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "overlap_pass": bool(overlap >= E4_ENSG_OVERLAP_MIN),
        "rho_pass": bool(rho >= E4_GENE_Z_SPEARMAN_MIN),
    }


def fit_battery_on_set(bellenguez_df: pd.DataFrame, gnomad: pd.DataFrame,
                         annot: pd.DataFrame, gene_set_ensg: set[str],
                         logger, tag: str) -> dict:
    """Run Sub-A v2.1 battery with Bellenguez MAGMA-Z as outcome.

    We hijack `build_sub_a_frame` by monkey-loading Bellenguez as the
    disorder MAGMA. Cleaner path: build the frame manually using the
    existing helper logic for gnomAD/NSNPS/length.
    """
    # Build frame: ENSGID × [MAGMA_Z (Bellenguez), in_set, covariates].
    # Use batch_055_B alzheimers NSNPS for structural covariate (same LD
    # geometry; differences in N are absorbed by MAGMA_Z standardization).
    # WHY: NSNPS varies with LD pruning and 1000G panel, not with sumstats
    # N; we reuse existing AD NSNPS file to avoid re-running a second
    # MAGMA step.
    from _common import load_nsnps_per_disorder
    nsnps = load_nsnps_per_disorder("alzheimers")
    frame = (
        bellenguez_df.merge(nsnps, on="ENSGID", how="inner")
        .merge(
            gnomad[["ENSGID", "lof_pLI", "lof_oe_ci_upper", "lof_exp"]],
            on="ENSGID", how="inner",
        )
        .merge(annot[["ENSGID", "log10_gene_length"]], on="ENSGID", how="inner")
    )
    frame = frame.dropna(
        subset=["MAGMA_Z", "lof_pLI", "lof_exp", "log10_gene_length", "NSNPS"]
    ).copy()
    frame["in_set"] = frame["ENSGID"].isin(gene_set_ensg).astype(int)
    frame["log10_exp_lof_plus1"] = np.log10(
        frame["lof_exp"].astype(float) + 1.0
    )
    frame["log10_NSNPS_plus1"] = np.log10(
        frame["NSNPS"].astype(float) + 1.0
    )
    frame = frame.sort_values("ENSGID").reset_index(drop=True)
    n = len(frame)
    n_in = int(frame["in_set"].sum())
    logger.info("E4 battery %s: n=%d in_set=%d", tag, n, n_in)
    if n < 1000 or n_in < 10:
        return {"status": "failed",
                "reason": f"n={n} in_set={n_in}"}
    covs = ["log10_gene_length", "lof_pLI",
            "log10_exp_lof_plus1", "log10_NSNPS_plus1"]
    ols = fit_ols(frame, "in_set", covs)
    huber = fit_huber(frame, "in_set", covs)
    tukey = fit_tukey_biweight(frame, covs, "in_set")
    rank_ols = fit_rank_magma_ols(frame, "in_set", covs)
    infl = compute_dfbetas_cooks(frame, covs, "in_set")
    recon = influential_outlier_reconciliation(frame, covs, "in_set", infl)
    return {
        "status": "ok",
        "tag": tag,
        "n_gene_universe": n,
        "n_set_in_universe": n_in,
        "ols": ols, "huber": huber, "tukey": tukey,
        "rank_magma_ols": rank_ols,
        "influence": infl,
        "influential_outlier_reconciliation": recon,
    }


def classify_ad_beta(beta: float, huber_beta: float, q: float,
                      ad_e2_is_length_artifact: bool) -> dict:
    """Apply E4 AD decision rule (brief_v2 §4 / design.yaml e4.decision_rule).

    Boundaries: replicated [0.20, 0.42] inclusive; hyper > 0.80; inconsistent
    (0.08, 0.20) ∪ (0.42, 0.80]; not_replicated |β|<=0.08; wrong < -0.08.
    """
    if not np.isfinite(beta):
        return {"verdict": "UNINTERPRETABLE",
                "reason": "β non-finite"}
    abs_b = abs(beta)
    huber_ok = (np.isfinite(huber_beta)
                 and abs_b > 0
                 and abs(beta - huber_beta) / abs_b < 0.20)
    q_ok = (np.isfinite(q) and q < 0.05)

    if beta < E4_AD_WRONG_DIRECTION_CAP:
        v = "AD_WRONG_DIRECTION"
        reason = f"β={beta:.3f} < {E4_AD_WRONG_DIRECTION_CAP}"
    elif abs_b <= E4_AD_NOT_REPLICATED_ABS_CAP:
        v = "AD_NOT_REPLICATED"
        reason = f"|β|={abs_b:.3f} <= {E4_AD_NOT_REPLICATED_ABS_CAP}"
    elif beta > E4_AD_HYPER_LO:
        v = "AD_HYPER_REPLICATED"
        reason = f"β={beta:.3f} > {E4_AD_HYPER_LO}"
    elif (E4_AD_REPLICATED_LO <= beta <= E4_AD_REPLICATED_HI
           and q_ok and huber_ok):
        v = "AD_REPLICATED"
        reason = (f"β={beta:.3f} in [{E4_AD_REPLICATED_LO}, "
                   f"{E4_AD_REPLICATED_HI}] AND q<0.05 AND |β_OLS-β_Huber|/|β|<0.20")
        # Pre-reg coupling: if E2 says AD × EDT1-ex-B3 is LENGTH_ARTIFACT,
        # relabel.
        if ad_e2_is_length_artifact:
            v = "REPLICATED_LENGTH_ARTIFACT"
            reason += (" — E2 coupling: AD × EDT1-ex-B3 = LENGTH_ARTIFACT; "
                        "F058_06 AD arm REFUTED-AS-BIOLOGICAL-SIGNAL")
    elif (E4_AD_INCONSISTENT_LO_BAND[0] < beta
            < E4_AD_INCONSISTENT_LO_BAND[1]):
        v = "AD_INCONSISTENT"
        reason = (f"β={beta:.3f} in ({E4_AD_INCONSISTENT_LO_BAND[0]}, "
                   f"{E4_AD_INCONSISTENT_LO_BAND[1]})")
    elif (E4_AD_INCONSISTENT_HI_BAND[0] < beta
            <= E4_AD_INCONSISTENT_HI_BAND[1]):
        v = "AD_INCONSISTENT"
        reason = (f"β={beta:.3f} in ({E4_AD_INCONSISTENT_HI_BAND[0]}, "
                   f"{E4_AD_INCONSISTENT_HI_BAND[1]}]")
    else:
        v = "AD_INCONSISTENT"
        reason = (f"β={beta:.3f} between bands; q_ok={q_ok} huber_ok={huber_ok}")
    return {"verdict": v, "reason": reason,
            "beta": float(beta), "huber_beta": float(huber_beta),
            "q": float(q)}


def load_e2_ad_verdict() -> dict:
    """Load E2 results.json (if present) and check AD × EDT1-ex-B3 verdicts.

    M6 audit fix: checks ALL 3 rings (scaffold_core_primary, vesicle_core_primary,
    remaining) for LENGTH_ARTIFACT. WHY conservative: EDT1-ex-B3 in E4 is
    the full union; in E2 it is bisected into 3 rings. A length artifact in
    ANY of the 3 rings could bleed into the full-union E4 fit. Flagging
    coupling on ANY of the 3 rings firing LENGTH_ARTIFACT avoids missing a
    legitimate coupling case (conservative toward null hypothesis of
    biological signal).

    Returns dict with:
      - is_length_artifact: bool (True iff ANY ring fires LENGTH_ARTIFACT)
      - per_ring_verdicts: dict[ring_name -> verdict_str]
      - rings_firing: list[ring_name] of rings with LENGTH_ARTIFACT
    """
    empty = {"is_length_artifact": False, "per_ring_verdicts": {},
             "rings_firing": [], "status": "no_e2_results"}
    p = OUTPUT_DIR / "e2" / "per_disorder_per_ring_verdicts.json"
    if not p.exists():
        return empty
    try:
        with p.open() as fh:
            data = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return {**empty, "status": f"read_failed: {exc}"}

    # AD is stored as `alzheimers` (batch_055_B convention).
    rings = ("scaffold_core_primary", "vesicle_core_primary", "remaining")
    per_ring: dict[str, str] = {}
    rings_firing: list[str] = []
    for ring in rings:
        cell = data.get(f"{ring}__alzheimers", {})
        verdict = str(cell.get("verdict", ""))
        per_ring[ring] = verdict
        if "LENGTH_ARTIFACT" in verdict:
            rings_firing.append(ring)
    return {
        "is_length_artifact": len(rings_firing) > 0,
        "per_ring_verdicts": per_ring,
        "rings_firing": rings_firing,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_059 E4")
    parser.add_argument("--smoke", action="store_true",
                         help="Smoke: limit SNP munge to 50k rows, skip MAGMA.")
    parser.add_argument("--smoke-munge-limit", type=int, default=50000)
    parser.add_argument("--skip-magma", action="store_true",
                         help="Skip MAGMA pipeline (use existing AD gene-Z for "
                              "testing downstream steps).")
    args = parser.parse_args()

    logger = setup_logger("batch_059.e4", LOGS_DIR / "e4.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "e4"
    work_dir = out_dir / "magma_work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    guard = WallClockGuard(max_seconds=E4_WALL_CAP_SECONDS)

    partial_state: dict = {
        "status": "partial",
        "batch": "059", "sub": "e4", "brief": "brief_v2.md (v2.1)",
        "smoke": bool(args.smoke),
        "wall_cap_min": E4_MAX_WALL_MIN,
    }

    # Pre-step 1: munge sumstats.
    if args.skip_magma:
        logger.info("SKIP-MAGMA: using existing AD gene-Z as proxy")
        bellenguez_df = None
        munge = {"status": "skipped"}
        liftover = {"status": "skipped"}
        magma = {"status": "skipped"}
    else:
        munge = munge_bellenguez_sumstats(
            guard, logger, work_dir,
            smoke_limit=(args.smoke_munge_limit if args.smoke else 0),
        )
        partial_state["pre_step_1_munge"] = {
            k: v for k, v in munge.items() if k != "snpdf"
        }
        if munge.get("status") != "ok":
            partial_state["status"] = "E4_DEFERRED"
            partial_state["verdict"] = {
                "verdict": "E4_DEFERRED",
                "reason": f"munge failed: {munge.get('reason', 'unknown')}",
            }
            atomic_write_json(partial_state, out_dir / "results.json")
            return 0

        # Pre-step 2: liftover.
        liftover = liftover_hg38_to_hg19(munge["snpdf"], guard, logger)
        partial_state["pre_step_2_liftover"] = {
            k: v for k, v in liftover.items() if k != "snpdf_hg19"
        }
        atomic_write_json(
            partial_state["pre_step_2_liftover"],
            out_dir / "liftover_stats.json",
        )
        if liftover.get("status") != "ok":
            partial_state["status"] = "E4_DEFERRED"
            partial_state["verdict"] = {
                "verdict": "E4_DEFERRED",
                "reason": f"liftover: {liftover.get('reason', 'unknown')}",
            }
            atomic_write_json(partial_state, out_dir / "results.json")
            return 0

        # Pre-step 3: write MAGMA inputs + run.
        snpdf_hg19 = liftover["snpdf_hg19"]
        inputs = write_magma_inputs(snpdf_hg19, work_dir, logger)
        # N extraction: read one row of Bellenguez to get n_cases + n_controls
        # (they are constant per-SNP at the bulk GWAS level — use median).
        src_sample = pd.read_csv(
            BELLENGUEZ_SUMSTATS, sep="\t", compression="gzip",
            usecols=["n_cases", "n_controls"], nrows=100,
        )
        n_cases = int(src_sample["n_cases"].median())
        n_controls = int(src_sample["n_controls"].median())
        magma = run_magma(
            snp_loc=Path(inputs["snp_loc_path"]),
            snp_pval=Path(inputs["snp_pval_path"]),
            work_dir=work_dir,
            n_cases=n_cases, n_controls=n_controls,
            guard=guard, logger=logger,
        )
        partial_state["pre_step_3_magma"] = magma
        if magma.get("status") != "ok":
            partial_state["status"] = "E4_DEFERRED"
            partial_state["verdict"] = {
                "verdict": "E4_DEFERRED",
                "reason": f"MAGMA: {magma.get('reason')}",
            }
            atomic_write_json(partial_state, out_dir / "results.json")
            return 0

        # Load Bellenguez gene-Z.
        bellenguez_df = load_bellenguez_gene_z(
            Path(magma["genes_out_path"]), logger,
        )
        # Save gene-Z TSV.
        bellenguez_df.to_csv(
            out_dir / "magma_gene_z.tsv", sep="\t", index=False,
        )

    # Pre-step 4: QC.
    if not args.skip_magma and bellenguez_df is not None:
        qc = qc_against_existing_ad(bellenguez_df, logger)
        partial_state["pre_step_4_qc"] = qc
        if qc.get("status") != "ok" or not (qc["overlap_pass"]
                                               and qc["rho_pass"]):
            partial_state["status"] = "E4_DEFERRED"
            partial_state["verdict"] = {
                "verdict": "E4_DEFERRED",
                "reason": (f"QC failed: overlap_pass={qc.get('overlap_pass')} "
                            f"rho_pass={qc.get('rho_pass')}"),
            }
            atomic_write_json(partial_state, out_dir / "results.json")
            return 0
    else:
        # SKIP-MAGMA path: do NOT run primary regression; mark as deferred.
        partial_state["status"] = "E4_DEFERRED"
        partial_state["verdict"] = {
            "verdict": "E4_DEFERRED",
            "reason": "--skip-magma flag set; nothing to replicate against",
        }
        atomic_write_json(partial_state, out_dir / "results.json")
        return 0

    # Primary: Sub-C v2.1 battery on EDT1-ex-B3.
    gnomad = load_gnomad_per_brief_v2()
    annot = load_gene_annot()
    edt1_sym = load_edt1()
    b3 = set(B3_GENES)
    edt1_ex_b3_sym = edt1_sym - b3
    edt1_full_sym = edt1_sym
    koop_ex_sym = load_koopmans_ex_B3() - edt1_sym
    b3_sym = b3

    def run_battery(sym_set: set[str], tag: str) -> dict:
        ensg, _ = symbols_to_ensgids(sym_set)
        return fit_battery_on_set(bellenguez_df, gnomad, annot, ensg,
                                    logger, tag)

    primary = run_battery(edt1_ex_b3_sym, "EDT1_ex_B3")
    secondary = {
        "B3": run_battery(b3_sym, "B3"),
        "EDT1_FULL": run_battery(edt1_full_sym, "EDT1_FULL"),
        "KOOP_ex_B3_ex_EDT1": run_battery(koop_ex_sym, "KOOP_ex_B3_ex_EDT1"),
    }

    # BH-FDR on 3 primary/secondary-1 rings (EDT1-ex-B3, B3, EDT1-FULL) single family.
    bh_pvals = []
    bh_labels = []
    for tag, item in [
        ("EDT1_ex_B3", primary),
        ("B3", secondary["B3"]),
        ("EDT1_FULL", secondary["EDT1_FULL"]),
    ]:
        if item.get("status") == "ok":
            bh_pvals.append(float(item["ols"]["p_one_sided"]))
            bh_labels.append(tag)
    qvals = bh_fdr(bh_pvals) if bh_pvals else []
    q_by_tag = dict(zip(bh_labels, qvals))

    # R1 reproduction gate: B3 × SCZ from EXISTING pipeline (not Bellenguez).
    # We check it by re-running the B3 × SCZ Sub-A battery under the existing
    # SCZ MAGMA. WHY: R1 tests that the *existing* pipeline's β is stable at
    # +3.24 ± 0.15 before we trust Bellenguez.
    from _common import load_magma_scz as _load_scz  # noqa: E402
    b3_ensg, _ = symbols_to_ensgids(b3_sym)
    r1_frame = build_sub_a_frame("scz", gnomad, annot, b3_ensg)
    r1_ols = fit_ols(r1_frame, "in_set",
                      ["log10_gene_length", "lof_pLI",
                       "log10_exp_lof_plus1", "log10_NSNPS_plus1"])
    r1_beta = float(r1_ols.get("beta_1", float("nan")))
    r1_pass = bool(np.isfinite(r1_beta)
                    and E4_R1_LO <= r1_beta <= E4_R1_HI)
    logger.info("E4 R1: B3 SCZ β=%.3f in [%.2f, %.2f]? %s",
                 r1_beta, E4_R1_LO, E4_R1_HI, r1_pass)

    # E2 coupling check (AD × EDT1-ex-B3). M6 audit fix: check ALL 3 rings.
    e2_ad_coupling = load_e2_ad_verdict()
    ad_e2_is_length_artifact = bool(e2_ad_coupling.get("is_length_artifact",
                                                        False))
    logger.info("E4 coupling: E2 AD ANY-ring LENGTH_ARTIFACT? %s "
                 "(rings_firing=%s, per_ring_verdicts=%s)",
                 ad_e2_is_length_artifact,
                 e2_ad_coupling.get("rings_firing"),
                 e2_ad_coupling.get("per_ring_verdicts"))

    # MN11 audit fix: re-check wall-clock guard before classification.
    # If the pipeline drifted past the 90-min cap between MAGMA completion
    # and here, classify as E4_DEFERRED and save partial state so downstream
    # audits see the wall-expiry cause clearly.
    if guard.expired():
        partial_state["status"] = "E4_DEFERRED"
        partial_state["verdict"] = {
            "verdict": "E4_DEFERRED",
            "reason": (f"wall-clock expired before classify_ad_beta "
                        f"(cap={E4_MAX_WALL_MIN}min); elapsed="
                        f"{time.time() - t0:.1f}s"),
        }
        partial_state["primary_EDT1_ex_B3"] = primary
        partial_state["secondary"] = secondary
        partial_state["e2_coupling_ad_length_artifact"] = ad_e2_is_length_artifact
        partial_state["e2_ad_coupling"] = e2_ad_coupling
        atomic_write_json(partial_state, out_dir / "results.json")
        logger.info("E4 E4_DEFERRED (wall-clock) written to %s",
                     out_dir / "results.json")
        return 0

    # Apply verdict.
    if not r1_pass:
        verdict = {
            "verdict": "UNINTERPRETABLE",
            "reason": f"R1 failed: B3 SCZ β={r1_beta:.3f} not in "
                       f"[{E4_R1_LO}, {E4_R1_HI}]",
        }
    elif primary.get("status") != "ok":
        verdict = {
            "verdict": "UNINTERPRETABLE",
            "reason": f"primary battery failed: {primary.get('reason')}",
        }
    else:
        beta = float(primary["ols"]["beta_1"])
        huber_beta = float(primary["huber"].get("beta_1", float("nan")))
        q = q_by_tag.get("EDT1_ex_B3", float("nan"))
        verdict = classify_ad_beta(beta, huber_beta, q,
                                    ad_e2_is_length_artifact)

    provenance = {
        "bellenguez_sumstats": sha256_file(BELLENGUEZ_SUMSTATS),
        "ncbi37_gene_loc": sha256_file(MAGMA_GENELOC_GRCH37),
        "ncbi38_gene_loc": sha256_file(MAGMA_GENELOC_GRCH38),
        "gnomad_tsv": sha256_file(GNOMAD_TSV),
        "gene_annot": sha256_file(GENE_ANNOT),
        "magma_scz": sha256_file(MAGMA_SCZ_GENES_OUT),
        "existing_alzheimers_magma": (
            sha256_file(EXISTING_ALZHEIMERS_MAGMA)
            if EXISTING_ALZHEIMERS_MAGMA.exists() else "missing"
        ),
    }

    results = {
        "status": "ok",
        "batch": "059", "sub": "e4", "brief": "brief_v2.md (v2.1)",
        "wall_s": time.time() - t0,
        "wall_cap_min": E4_MAX_WALL_MIN,
        "smoke": bool(args.smoke),
        "pre_step_1_munge": (
            {k: v for k, v in munge.items() if k != "snpdf"}
            if isinstance(munge, dict) else {}
        ),
        "pre_step_2_liftover": (
            {k: v for k, v in liftover.items() if k != "snpdf_hg19"}
            if isinstance(liftover, dict) else {}
        ),
        "pre_step_3_magma": magma if isinstance(magma, dict) else {},
        "pre_step_4_qc": partial_state.get("pre_step_4_qc", {}),
        "R1_reproduction_gate": {
            "target": E4_R1_TARGET, "tolerance": E4_R1_TOLERANCE,
            "lo": E4_R1_LO, "hi": E4_R1_HI,
            "b3_scz_beta_ols": r1_beta, "pass": r1_pass,
        },
        "e2_coupling_ad_length_artifact": ad_e2_is_length_artifact,
        "e2_ad_coupling": e2_ad_coupling,
        "primary_EDT1_ex_B3": primary,
        "secondary": secondary,
        "bh_fdr_family": {
            "labels": bh_labels, "pvals": bh_pvals, "qvals": qvals,
            "family_size": len(bh_pvals), "q_threshold": BH_Q,
        },
        "verdict": verdict,
        "provenance_sha256": provenance,
        "brief_contract": {
            "seed_bootstrap": SEED_E4_BOOT,
            "snp_match_rate_min": E4_SNP_MATCH_RATE_MIN,
            "gene_z_spearman_min": E4_GENE_Z_SPEARMAN_MIN,
            "ensg_overlap_min": E4_ENSG_OVERLAP_MIN,
            "liftover_tool": "pyliftover",
            "magma_window_up_kb": MAGMA_ANNOT_WINDOW_UP_KB,
            "magma_window_down_kb": MAGMA_ANNOT_WINDOW_DOWN_KB,
            "magma_gene_model": MAGMA_GENE_MODEL,
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("E4 wrote %s (wall=%.1fs)",
                 out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
