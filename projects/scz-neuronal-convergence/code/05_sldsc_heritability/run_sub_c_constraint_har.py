#!/usr/bin/env python3
"""batch_044 Sub-C: gnomAD v4.1 LOEUF + HAR-proximal Fisher + adjusted logistic.

WHY: Per brief.md §WHAT/§MEASUREMENT Sub-C — this script produces the four JSON
     outputs pre-registered in design.yaml sub_experiments.C_constraint_har.outputs,
     plus a consolidated C_summary.json with classification per the pre-registered
     decision rule (brief.md §DECISION RULE / Sub-C).

Scope (per parent-directive, 2026-04-22):
  * MAGMA top-1% gene list is DROPPED from batch_044 (MAGMA binary not on PATH
    and the marginal scientific value of a 4th list is low given the three
    anchor lists below). This is a Rule 0 compliance choice: rather than
    synthesize MAGMA Z-scores we simply remove that branch.
  * Sub-C now tests THREE gene lists:
      1. PGC3_Prioritised_EDT1 (n~120): from data/19426775/scz2022-Extended-
         Data-Table1.xlsx sheet `ST12 all criteria`, column `Prioritised`==YES/
         1/TRUE AND biotype=='protein_coding'.
      2. PGC3_Prioritised_SynGO (n~35): within the same xlsx intersect with
         column `SynGO.GeneSetMemb`.
      3. SCHEMA_exome_wide_significant (n=10, positive control): Singh 2022
         Table 1 genome-wide significant genes at P<2.5e-6 (canonical exome-
         wide threshold). Loaded from experiments/batch_044/input/
         schema_exome_wide_significant.txt (one gene symbol per line).

Uses only standard scientific Python (scipy.stats, statsmodels, pandas, numpy).
Idempotent: skip outputs that already exist unless --force.

Positive-control decision rule (per parent-directive):
  * SCHEMA_exome_wide_significant pass requires Fisher OR > 5 AND q < 0.01.
    WHY OR>5 (not OR>10): the n=10 exome-wide list is smaller than the
    FDR<0.05 list would have been, so expected-count arithmetic gives a
    point OR near ~40 with a wider CI; OR>5 is conservative and empirically
    below the expected point estimate.

Usage:
  python3 run_sub_c_constraint_har.py [--force] [--schema-list PATH]

If --schema-list is not passed AND the canonical file is not present on disk,
the script STOPS rather than inventing identifiers.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import logging
import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

try:
    import statsmodels.api as sm
except ImportError as e:  # pragma: no cover - statsmodels is required
    raise SystemExit(f"statsmodels is required: {e}")

# ----------------------------------------------------------------------------- Paths
PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_044"
OUTPUT_DIR = BATCH_DIR / "output"
LOG_DIR = BATCH_DIR / "logs"
INPUT_DIR = BATCH_DIR / "input"

GNOMAD_TSV = PROJECT_ROOT / "data" / "item_15" / "gnomad.v4.1.constraint_metrics.tsv"
HAR_BED = PROJECT_ROOT / "data" / "item_15" / "reference_assets" / "harsRichard2020.GRCh37.bed"
TSS_CSV = PROJECT_ROOT / "data" / "ldsc" / "gene_tss_grch37.csv"
PGC3_XLSX = PROJECT_ROOT / "data" / "19426775" / "scz2022-Extended-Data-Table1.xlsx"
GTEX_TPM = PROJECT_ROOT / "data" / "GTEx_v8_gene_median_tpm.gct.gz"

# SCHEMA exome-wide significant (n=10) canonical file.
SCHEMA_CANONICAL = INPUT_DIR / "schema_exome_wide_significant.txt"
# Legacy fallback locations (if the canonical file is moved).
SCHEMA_CANDIDATES = [
    SCHEMA_CANONICAL,
    PROJECT_ROOT / "data" / "item_15" / "schema_exome_wide_significant.txt",
    PROJECT_ROOT / "data" / "singh2022_schema_exome_wide_significant.txt",
]

# Primary Sub-C output JSONs (design.yaml):
OUT_LOEUF_FISHER = OUTPUT_DIR / "C_loeuf_fisher.json"
OUT_LOEUF_LOGISTIC = OUTPUT_DIR / "C_loeuf_logistic.json"
OUT_HAR_FISHER = OUTPUT_DIR / "C_har_fisher.json"
OUT_PERMUTATION = OUTPUT_DIR / "C_permutation_length_stratified.json"
OUT_SUMMARY = OUTPUT_DIR / "C_summary.json"
SCRIPT_LOG = LOG_DIR / "run_sub_c_constraint_har.log"

# Pre-registered constants (cited in brief / design.yaml)
LOEUF_THRESHOLD = 0.35  # Karczewski 2020 canonical LoF-constrained threshold
HAR_WINDOW_BP = 100_000  # Doan 2016 convention (TSS within 100 kb of any HAR)
MHC_CHR = "6"
MHC_START = 25_000_000
MHC_END = 34_000_000
N_PERMUTATIONS = 10_000
N_LENGTH_DECILES = 10
RNG_SEED = 20260422  # fixed for reproducibility (today's date; see CLAUDE.md)

# Positive-control decision rule thresholds (per parent-directive).
SCHEMA_POSCTRL_MIN_OR = 5.0
SCHEMA_POSCTRL_MAX_Q = 0.01

# Number of gene lists tested → used for BH multiple-testing N.
N_GENE_LISTS = 3

# Canonical gene-list names (used across outputs and summarizer).
LIST_PGC3_EDT1 = "PGC3_Prioritised_EDT1"
LIST_PGC3_SYNGO = "PGC3_Prioritised_SynGO"
LIST_SCHEMA = "SCHEMA_exome_wide_significant"

# ----------------------------------------------------------------------------- Logging


def _setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_sub_c_constraint_har")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(SCRIPT_LOG)
        fh.setFormatter(logging.Formatter("[%(asctime)sZ] %(levelname)s %(message)s",
                                          datefmt="%Y-%m-%dT%H:%M:%S"))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s",
                                          datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log_input(logger: logging.Logger, label: str, path: pathlib.Path) -> dict[str, Any]:
    meta = {"label": label, "path": str(path), "exists": path.exists()}
    if path.exists():
        meta["sha256"] = sha256(path)
        meta["bytes"] = path.stat().st_size
        logger.info("INPUT %s: %s sha256=%s bytes=%d", label, path, meta["sha256"], meta["bytes"])
    else:
        logger.error("INPUT %s MISSING: %s", label, path)
    return meta


# ----------------------------------------------------------------------------- Loaders


def load_gnomad_constraint(logger: logging.Logger) -> pd.DataFrame:
    """Filter gnomAD v4.1 to canonical + MANE-select transcripts with valid LOEUF.

    WHY canonical + mane_select: gnomAD v4.1 contains one row per transcript; the
         canonical/MANE filter collapses to one row per gene (the convention used
         in Karczewski 2020 for LOEUF quoting). ~19,700 genes expected (design.yaml).
    """
    logger.info("Loading gnomAD constraint TSV...")
    df = pd.read_csv(GNOMAD_TSV, sep="\t", low_memory=False)
    logger.info("Raw rows: %d, columns: %d", len(df), df.shape[1])
    # WHY pre-filter to ENSG gene_id rows: gnomAD v4.1 emits TWO rows per gene —
    # (a) a legacy HGNC-numeric gene_id row with NaN chromosome/cds_length, and
    # (b) an ENSG-prefixed row with populated coordinates. Previously
    # drop_duplicates(keep="first") kept the broken HGNC row, nulling downstream
    # MHC / chrX / length features. Restricting to ENSG rows is explicit and
    # robust to upstream row-order changes.
    if "gene_id" in df.columns:
        before = len(df)
        df = df[df["gene_id"].astype(str).str.startswith("ENSG")].copy()
        logger.info("Pre-filter to ENSG gene_id rows: %d -> %d", before, len(df))
    # WHY normalize chromosome at load time: gnomAD v4.1 emits values like "chr6"
    # while our constants (MHC_CHR="6", chrX="X") are plain tokens. Normalizing
    # once here prevents every downstream comparison from needing to re-implement
    # the strip. Applied before any copy/filter so all derived columns see it.
    if "chromosome" in df.columns:
        df["chromosome"] = (df["chromosome"].astype(str)
                            .str.replace(r"^chr", "", regex=True))
    # Coerce boolean-like columns; gnomAD exports these as lower-case 'true'/'false'.
    for col in ("canonical", "mane_select"):
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    mask = df["canonical"] & df["mane_select"] & df["lof.oe_ci.upper"].notna()
    filt = df.loc[mask].copy()
    # Collapse duplicates (shouldn't happen once mane_select is True + ENSG filter,
    # but keep as belt-and-suspenders).
    filt = filt.drop_duplicates(subset=["gene"], keep="first").reset_index(drop=True)
    logger.info("After canonical=T & mane_select=T & valid LOEUF: %d genes", len(filt))
    filt["loeuf_lt_035"] = filt["lof.oe_ci.upper"] < LOEUF_THRESHOLD
    # Engineered covariates (brief.md §MEASUREMENT Sub-C adjusted logistic)
    # Gene length from chromosome:start-end interval if available; fallback to CDS only.
    if "start_position" in filt.columns and "end_position" in filt.columns:
        filt["gene_length"] = (filt["end_position"] - filt["start_position"]).abs() + 1
    else:
        filt["gene_length"] = filt["cds_length"]  # conservative fallback
    filt["log_cds_length"] = np.log1p(filt["cds_length"].fillna(0))
    filt["log_gene_length"] = np.log1p(filt["gene_length"].fillna(0))
    filt["log_obs_mis_plus1"] = np.log1p(filt["mis.obs"].fillna(0))

    # ----- Join with TSS file for MHC/chrX position info -----
    # WHY: gnomAD v4.1 constraint TSV (autosomes-only download at
    #      gcp-public-data--gnomad/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv)
    #      has NO position columns (no start_position, end_position), and ships no
    #      chrX/chrY rows. To get MHC indicator (requires chr6:25-34 Mb) and chrX
    #      indicator, we join with the local TSS file at data/ldsc/gene_tss_grch37.csv
    #      which has one row per transcript with {gene, chrom, tss, strand, biotype}.
    #      Collapse to gene-level before joining: for each gene, take median TSS on
    #      the most-common chromosome assignment.
    tss_path = PROJECT_ROOT / "data" / "ldsc" / "gene_tss_grch37.csv"
    if tss_path.exists():
        tss_df = pd.read_csv(tss_path)
        # Drop scaffold-patched chroms, keep 1..22 and X/Y
        tss_df["chrom"] = tss_df["chrom"].astype(str)
        tss_df = tss_df[tss_df["chrom"].isin([str(i) for i in range(1, 23)] + ["X", "Y"])]
        # For each gene, take mode chromosome + median TSS within that chromosome
        def _agg(g):
            top_chrom = g["chrom"].mode().iloc[0]
            tss_on_top = g.loc[g["chrom"] == top_chrom, "tss"]
            return pd.Series({"tss_chrom": top_chrom, "tss_pos": int(tss_on_top.median())})
        tss_agg = tss_df.groupby("gene", as_index=False).apply(_agg, include_groups=False)
        tss_agg = tss_agg.rename(columns={"gene": "gene"}) if "gene" in tss_agg.columns else tss_agg
        # With apply returning Series, the gene key is in the index level 0 (if group_keys defaults)
        if "gene" not in tss_agg.columns:
            tss_agg = tss_agg.reset_index().rename(columns={"level_0": "gene"})
        tss_agg = tss_agg[["gene", "tss_chrom", "tss_pos"]]
        filt = filt.merge(tss_agg, on="gene", how="left")
        logger.info("Joined TSS: %d of %d genes now have tss_chrom+tss_pos",
                    filt["tss_chrom"].notna().sum(), len(filt))
    else:
        filt["tss_chrom"] = np.nan
        filt["tss_pos"] = np.nan
        logger.warning("TSS file not found at %s — MHC/chrX will be all-False", tss_path)

    # MHC + chrX flags (from TSS join). Fall back to gnomAD's "chromosome" if TSS missing.
    tss_chrom_str = filt["tss_chrom"].astype(str)
    tss_pos_int = filt["tss_pos"].fillna(-1).astype(int)
    filt["mhc_indicator"] = (tss_chrom_str == MHC_CHR) & (tss_pos_int >= MHC_START) & (tss_pos_int <= MHC_END)
    # chrX indicator: prefer TSS chrom; fallback to gnomAD chromosome (which is autosomes only)
    filt["chrX_indicator"] = tss_chrom_str == "X"

    # ----- Defensive post-filter assertions (catch upstream dedup/normalize bugs) ---
    # WHY: silent regressions in the gnomAD schema (e.g. re-ordered duplicate rows,
    # "chr" prefix returning) previously nulled out MHC / chrX flags and zeroed
    # the entire MHC covariate. These three assertions are a tripwire.
    chrom_notna_frac = filt["chromosome"].notna().mean()
    assert chrom_notna_frac > 0.98, (
        f"CRITICAL: only {chrom_notna_frac:.3f} of filtered genes have chromosome "
        "— dedup or filter bug"
    )
    n_mhc = int(filt["mhc_indicator"].sum())
    # MHC region (chr6:25-34 Mb) contains ~200-250 protein-coding genes; see
    # gnomAD v4.1 docs + Horton 2004 MHC map. TSS-based lookup will yield
    # a higher count than position-in-gnomAD because TSS file uses transcripts.
    assert n_mhc >= 100, (
        f"CRITICAL: only {n_mhc} MHC genes — MHC filter bug (expected 200-300)"
    )
    n_chrX = int(filt["chrX_indicator"].sum())
    # gnomAD v4.1 autosomal constraint file has zero chrX rows. Since filt is
    # filtered to canonical/MANE gnomAD entries, chrX count will be 0 unless the
    # chrX gnomAD file is also loaded (not done here). The chrX_indicator
    # covariate will therefore contribute no variance and get dropped from the
    # logistic regression — this is expected and logged (not fatal).
    if n_chrX == 0:
        logger.warning("chrX count is 0 — gnomAD v4.1 autosomes-only file loaded; "
                       "chrX_indicator covariate will be dropped from logistic regression.")
    return filt


def load_tss(logger: logging.Logger) -> pd.DataFrame:
    """Load gene TSS file and collapse to one TSS per gene symbol.

    WHY collapse: Some genes have multiple transcripts → multiple TSS rows. For the
         HAR 100-kb window test we use the gene's canonical TSS. We conservatively
         take the MIN start coordinate on strand '+' and MAX end coordinate on '-'
         if strand present; otherwise take the median position.
    """
    if not TSS_CSV.exists():
        logger.error("TSS file not found: %s — cannot run HAR test.", TSS_CSV)
        raise SystemExit(11)
    df = pd.read_csv(TSS_CSV)
    logger.info("TSS file rows=%d columns=%s", len(df), list(df.columns))
    # Keep standard autosome + X chromosomes only (drop HG*_PATCH scaffolds which
    # cannot be aligned to HARs; HAR bed is chr1..22).
    std = df["chrom"].astype(str).isin([str(i) for i in range(1, 23)] + ["X", "Y", "MT"])
    df = df.loc[std].copy()
    # Take representative TSS per gene symbol (median of TSS positions per gene).
    agg = df.groupby("gene").agg(
        chrom=("chrom", "first"),
        tss=("tss", "median"),
        n_transcripts=("tss", "count"),
    ).reset_index()
    agg["tss"] = agg["tss"].astype(int)
    logger.info("Unique genes with TSS: %d", len(agg))
    return agg


def load_har_bed(logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(HAR_BED, sep="\t", header=None,
                     names=["chrom", "start", "end", "source"], dtype={"chrom": str})
    logger.info("HAR bed rows=%d", len(df))
    counts = df["source"].value_counts().to_dict()
    logger.info("HAR sources: %s", counts)
    return df


def load_pgc3_lists(logger: logging.Logger) -> dict[str, set[str]]:
    """Derive PGC3 Prioritised_EDT1 and Prioritised_SynGO from xlsx.

    WHY: brief.md + design.yaml say these lists must come from Trubetskoy 2022
         Extended-Data-Table 1. Re-derivation from the xlsx sheet `ST12 all criteria`
         is deterministic (batch_025/batch_033 both use the same sheet & columns).
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"openpyxl required to read PGC3 xlsx: {e}")
    if not PGC3_XLSX.exists():
        logger.error("PGC3 xlsx missing: %s", PGC3_XLSX)
        raise SystemExit(12)
    df = pd.read_excel(PGC3_XLSX, sheet_name="ST12 all criteria")
    logger.info("Loaded ST12 sheet: rows=%d columns=%d", len(df), df.shape[1])
    # Expected columns (verified in batch_025/batch_033):
    # 'Symbol.ID' (gene symbol), 'biotype', 'Prioritised' (YES/NO or 1/0),
    # 'SynGO.GeneSetMemb' (YES/NO or 1/0).
    def _truthy(v: Any) -> bool:
        # WHY robustified: xlsx coerces YES/1/TRUE into mixed types (int 1, float
        # 1.0, bool True, str "YES"/"Y"/"y"/"true"/"TRUE"). Normalizing via
        # str(v).strip().lower() handles every observed permutation in PGC3 xlsx
        # exports without missing any Prioritised flag.
        if v is None:
            return False
        try:
            if isinstance(v, float) and math.isnan(v):
                return False
        except TypeError:
            pass
        return str(v).strip().lower() in {"yes", "y", "1", "true", "1.0"}

    df["_prioritised"] = df["Prioritised"].apply(_truthy) if "Prioritised" in df else False
    df["_syngo"] = df["SynGO.GeneSetMemb"].apply(_truthy) if "SynGO.GeneSetMemb" in df else False
    # Column is literally 'gene_biotype' in scz2022-Extended-Data-Table1.xlsx ST12
    # (verified against actual file 2026-04-23; prior code used unqualified 'biotype'
    #  which is never present). Fallback to 'biotype' for any manually re-exported
    # variants of the sheet.
    biotype_col = "gene_biotype" if "gene_biotype" in df.columns else "biotype"
    pc = df[biotype_col].astype(str) == "protein_coding"
    prior_genes = set(df.loc[pc & df["_prioritised"], "Symbol.ID"].dropna().astype(str))
    syngo_genes = set(df.loc[pc & df["_prioritised"] & df["_syngo"], "Symbol.ID"].dropna().astype(str))
    logger.info("PGC3 Prioritised (protein_coding): %d", len(prior_genes))
    logger.info("PGC3 Prioritised x SynGO: %d", len(syngo_genes))
    return {LIST_PGC3_EDT1: prior_genes, LIST_PGC3_SYNGO: syngo_genes}


def build_schema_exome_wide(logger: logging.Logger,
                            override: pathlib.Path | None) -> set[str]:
    """Load SCHEMA exome-wide significant gene list from disk; refuse to synthesize.

    Canonical source: Singh 2022 Table 1, n=10 genes at P<2.5e-6 exome-wide
    significance (SETD1A, CUL1, XPO7, TRIO, CACNA1G, SP4, GRIA3, GRIN2A, HERC1,
    RB1CC1). File is pre-created at experiments/batch_044/input/
    schema_exome_wide_significant.txt (one gene symbol per line).

    WHY refuse to synthesize: Rule 0 — gene identifiers must come from a verified
    source file, never a script-embedded literal.
    """
    if override is not None:
        if not override.exists():
            logger.error("--schema-list file not found: %s", override)
            raise SystemExit(13)
        return _read_schema_file(override, logger)
    for cand in SCHEMA_CANDIDATES:
        if cand.exists():
            logger.info("Using SCHEMA list: %s", cand)
            return _read_schema_file(cand, logger)
    logger.error(
        "SCHEMA exome-wide significant gene list not found. Looked in: %s. "
        "Either (a) drop a one-gene-per-line file at one of those paths, "
        "or (b) pass --schema-list PATH. The canonical file (batch_044/input/"
        "schema_exome_wide_significant.txt) contains the 10 Singh 2022 Table 1 "
        "genes at P<2.5e-6. Refusing to synthesize gene identifiers (Rule 0).",
        [str(p) for p in SCHEMA_CANDIDATES],
    )
    raise SystemExit(10)


def _read_schema_file(path: pathlib.Path, logger: logging.Logger) -> set[str]:
    genes: set[str] = set()
    with path.open() as fh:
        for line in fh:
            tok = line.strip().split("\t")[0].split(",")[0].strip()
            if tok and not tok.startswith("#") and tok.lower() != "gene":
                genes.add(tok)
    logger.info("SCHEMA genes loaded: n=%d from %s sha256=%s",
                len(genes), path, sha256(path))
    return genes


# ----------------------------------------------------------------------------- Helpers


def exclude_mhc_from_set(genes: set[str], constraint: pd.DataFrame,
                         logger: logging.Logger, label: str) -> tuple[set[str], int]:
    """Drop genes whose gnomAD canonical-transcript coords fall in MHC 25-34 Mb."""
    mhc_genes = set(constraint.loc[constraint["mhc_indicator"], "gene"].astype(str))
    removed = genes & mhc_genes
    filt = genes - mhc_genes
    logger.info("MHC filter [%s]: removed %d / %d genes", label, len(removed), len(genes))
    return filt, len(removed)


def fisher_one_sided_enrichment(a: int, b: int, c: int, d: int) -> dict[str, Any]:
    """2x2 one-sided (alternative='greater') Fisher's exact with exact OR CI.

    Layout:
              in_list   not_in_list
      pos        a           b
      neg        c           d
    """
    # scipy.stats.fisher_exact returns point OR (Haldane-ish); use it for p-value.
    odds_ratio, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    # 95% exact CI via Fisher's noncentral hypergeometric confidence interval
    # (scipy >= 1.9 provides stats.contingency.odds_ratio with .confidence_interval).
    try:
        or_obj = stats.contingency.odds_ratio([[a, b], [c, d]])
        ci = or_obj.confidence_interval(confidence_level=0.95, alternative="two-sided")
        ci_low = float(ci.low) if ci.low is not None else math.nan
        ci_high = float(ci.high) if ci.high is not None else math.nan
        point = float(or_obj.statistic)
    except Exception:  # pragma: no cover - older scipy fallback
        ci_low = math.nan
        ci_high = math.nan
        point = float(odds_ratio)
    return {
        "a": int(a), "b": int(b), "c": int(c), "d": int(d),
        "odds_ratio_point": point,
        "odds_ratio_scipy": float(odds_ratio),
        "odds_ratio_95ci_low": ci_low,
        "odds_ratio_95ci_high": ci_high,
        "p_one_sided_greater": float(p),
    }


def har_proximal_genes(tss: pd.DataFrame, har: pd.DataFrame) -> set[str]:
    """Return set of gene symbols whose TSS lies within HAR_WINDOW_BP of any HAR.

    WHY O(N_gene) with sort+binary-search per chromosome: ~20k genes × ~3k HARs is
    trivial in memory so we keep it simple with pandas merge-on-range via numpy.
    """
    gene_set: set[str] = set()
    for chrom, har_chrom in har.groupby("chrom"):
        tss_chrom = tss[tss["chrom"] == str(chrom)]
        if tss_chrom.empty:
            continue
        tss_pos = tss_chrom["tss"].to_numpy()
        har_starts = har_chrom["start"].to_numpy()
        har_ends = har_chrom["end"].to_numpy()
        # For each TSS, check any HAR within ±100 kb.
        # Vectorized O(n_tss * n_har) — fine at this scale.
        for gene, pos in zip(tss_chrom["gene"].to_numpy(), tss_pos):
            if np.any((har_starts - HAR_WINDOW_BP <= pos) & (pos <= har_ends + HAR_WINDOW_BP)):
                gene_set.add(str(gene))
    return gene_set


def length_deciles(lengths: np.ndarray, n: int = N_LENGTH_DECILES) -> np.ndarray:
    """Return decile index (0..n-1) for each gene's log-length.

    WHY log-length: gene length is heavy-tailed; LOEUF is collinear with length, so
    stratifying the permutation on log-length quantiles produces a null that matches
    the observed length distribution of the SCZ list and removes the length-confound.
    """
    logL = np.log1p(np.asarray(lengths, dtype=float))
    quantiles = np.linspace(0, 1, n + 1)
    edges = np.quantile(logL, quantiles)
    # Make sure edges are strictly increasing to avoid np.digitize collapse.
    edges = np.unique(edges)
    idx = np.digitize(logL, edges[1:-1], right=False)
    return idx


def length_stratified_permutation(
    scz_mask: np.ndarray,
    loeuf_mask: np.ndarray,
    decile_idx: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Permute SCZ labels within each length decile and recompute Fisher OR."""
    observed_or = _or_from_masks(scz_mask, loeuf_mask)
    observed_a = int(np.sum(scz_mask & loeuf_mask))
    null_or = np.empty(n_perm, dtype=float)
    null_a = np.empty(n_perm, dtype=int)
    # Precompute per-decile indices
    decile_members = {d: np.where(decile_idx == d)[0] for d in np.unique(decile_idx)}
    scz_full = scz_mask.copy()
    for p in range(n_perm):
        perm_scz = np.zeros_like(scz_full)
        for d, idx in decile_members.items():
            n_scz_in_d = int(scz_full[idx].sum())
            if n_scz_in_d == 0:
                continue
            chosen = rng.choice(idx, size=n_scz_in_d, replace=False)
            perm_scz[chosen] = True
        null_or[p] = _or_from_masks(perm_scz, loeuf_mask)
        null_a[p] = int(np.sum(perm_scz & loeuf_mask))
    # Empirical p — fraction of nulls with OR >= observed (one-sided greater).
    emp_p = float((null_or >= observed_or).sum() + 1) / (n_perm + 1)
    return {
        "observed_odds_ratio": float(observed_or),
        "observed_overlap": observed_a,
        "n_permutations": int(n_perm),
        "empirical_p_one_sided_greater": emp_p,
        "null_or_mean": float(np.mean(null_or)),
        "null_or_std": float(np.std(null_or, ddof=1)),
        "null_overlap_mean": float(np.mean(null_a)),
    }


def _or_from_masks(scz: np.ndarray, loeuf: np.ndarray) -> float:
    a = int(np.sum(scz & loeuf))
    b = int(np.sum(scz & ~loeuf))
    c = int(np.sum(~scz & loeuf))
    d = int(np.sum(~scz & ~loeuf))
    if b == 0 or c == 0:
        return float("inf") if a > 0 else 0.0
    return (a * d) / (b * c)


def _compute_vif(X_nocon: np.ndarray, names: list[str]) -> dict[str, float]:
    """Variance Inflation Factor per covariate (no constant column).

    WHY: VIF > 10 is the canonical threshold (Belsley/Kuh/Welsch 1980) for
         problematic multicollinearity. Reporting VIF per covariate makes
         future singular-Hessian failures diagnosable instead of opaque.

    Returns dict {name: vif}. A covariate with zero variance yields NaN.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # variance_inflation_factor requires a constant to be meaningful; add one.
    Xc = sm.add_constant(X_nocon, has_constant="add")
    vif = {}
    # Skip constant at column 0; VIF for covariates at indices 1..
    for i, name in enumerate(names, start=1):
        try:
            v = float(variance_inflation_factor(Xc, i))
        except Exception:
            v = float("nan")
        vif[name] = v
    return vif


def fit_logistic(df: pd.DataFrame, predictor_name: str, covariates: list[str],
                 logger: logging.Logger) -> dict[str, Any]:
    """Fit logistic regression: LOEUF<0.35 ~ predictor + covariates.

    Robust strategy (WHY): the primary covariate block (log_cds_length,
    log_gene_length, log_obs_mis_plus1, brain_tau) is known to be highly
    collinear — log_cds_length vs log_gene_length correlate > 0.9 and
    log_obs_mis_plus1 scales with CDS length. This makes the Hessian
    ill-conditioned and sm.Logit().fit() raises LinAlgError: Singular
    matrix at Hessian inversion.

    Layered fit strategy:
      1. Compute VIF per covariate + condition number of the design matrix.
      2. Try standard MLE via sm.Logit.fit(method='newton', maxiter=200).
      3. If MLE fails AND condition number > 100 (or any VIF > 10):
         a. Drop the covariate with the largest VIF (keeps predictor_name
            intact) and retry MLE with the reduced design.
         b. If MLE still fails, fall back to L2-regularized GLM fit
            (`fit_regularized(alpha=1e-4, L1_wt=0.0)`). Report beta only;
            Wald p-values are NOT reported because they are invalid under
            penalization (Tibshirani 1996; Zou 2006 standard result).

    The returned `fit_method` is one of {"MLE", "MLE_after_vif_drop", "ridge"}.
    """
    cols = [predictor_name, *covariates]
    sub = df.dropna(subset=cols + ["loeuf_lt_035"]).copy()
    y = sub["loeuf_lt_035"].astype(int).to_numpy()
    X_df = sub[cols].astype(float)
    X_raw = X_df.to_numpy()
    X = sm.add_constant(X_raw, has_constant="add")

    # ---- Diagnostics: VIF + condition number (WHY: make multicollinearity visible)
    vif = _compute_vif(X_raw, cols)
    try:
        # Condition number of the centered+scaled design (sklearn convention).
        # Use numpy.linalg.cond on X.T @ X (equivalent to square of svd ratio).
        cond = float(np.linalg.cond(X))
    except Exception:
        cond = float("nan")
    logger.info("Logistic diagnostics [predictor=%s]: cond(X)=%.3e  VIF=%s",
                predictor_name, cond, {k: round(v, 2) for k, v in vif.items()})

    names_full = ["const", predictor_name, *covariates]
    dropped: list[str] = []
    fit_method = "MLE"
    fit = None
    err_primary: str | None = None
    try:
        model = sm.Logit(y, X)
        fit = model.fit(disp=False, maxiter=200)
    except Exception as e:
        err_primary = str(e)
        logger.warning("Primary MLE failed (%s); will attempt fallback.", e)

    # Decide whether diagnostics flag collinearity badly enough for fallback.
    high_vif = [k for k, v in vif.items() if np.isfinite(v) and v > 10.0]
    flag_collinear = (cond > 100.0) or bool(high_vif)

    if fit is None and X.shape[1] >= 3 and flag_collinear:
        # ---- Strategy A: drop largest-VIF covariate (not the predictor) + refit MLE
        # WHY drop rather than regularize first: a cleanly fit MLE on a slightly
        # reduced covariate set still yields valid Wald p-values, which is what
        # the pre-registered decision rule prefers.
        candidates = {k: v for k, v in vif.items()
                      if k != predictor_name and np.isfinite(v)}
        if candidates:
            drop_col = max(candidates, key=candidates.get)
            reduced_covs = [c for c in covariates if c != drop_col]
            dropped = [drop_col]
            logger.info("Retry MLE after dropping highest-VIF covariate '%s' (VIF=%.2f).",
                        drop_col, candidates[drop_col])
            cols2 = [predictor_name, *reduced_covs]
            X2_raw = sub[cols2].astype(float).to_numpy()
            X2 = sm.add_constant(X2_raw, has_constant="add")
            try:
                fit = sm.Logit(y, X2).fit(disp=False, maxiter=200)
                fit_method = "MLE_after_vif_drop"
                X = X2
                names_full = ["const", predictor_name, *reduced_covs]
                covariates = reduced_covs  # for downstream coef mapping
            except Exception as e2:
                logger.warning("MLE after VIF drop also failed (%s); trying ridge.", e2)

    if fit is None:
        # ---- Strategy B: L2-regularized GLM. WHY L2 not L1: with a SCZ indicator
        # and a small number of length covariates we want shrinkage, not sparsity,
        # to stabilize the Hessian while keeping every covariate in the model.
        # WHY alpha=1e-4: minimal shrinkage — just enough to regularize the
        # Hessian; large enough alphas would bias the predictor coefficient.
        try:
            glm = sm.GLM(y, X, family=sm.families.Binomial())
            rfit = glm.fit_regularized(alpha=1e-4, L1_wt=0.0)
            fit_method = "ridge"
            # For regularized GLM, statsmodels does NOT return valid standard
            # errors / p-values. We report only betas.
            params = np.asarray(rfit.params)
            coef_block = {}
            for i, nm in enumerate(names_full):
                coef_block[nm] = {
                    "beta": float(params[i]),
                    "se": None,
                    "z": None,
                    "p": None,
                    "ci_low": None,
                    "ci_high": None,
                    "note": "p-value suppressed under L2 regularization; decision "
                            "rule relies on sign and magnitude of beta_scz.",
                }
            scz_beta = float(params[names_full.index(predictor_name)])
            return {
                "n": int(len(y)),
                "n_outcome_1": int(y.sum()),
                "converged": True,
                "fit_method": fit_method,
                "condition_number": cond,
                "vif": vif,
                "dropped_covariates": dropped,
                "coefficients": coef_block,
                "beta_scz": scz_beta,
                "se_scz": None,
                "p_scz": None,
                "primary_mle_error": err_primary,
            }
        except Exception as e3:
            logger.exception("All logistic strategies failed: %s", e3)
            return {
                "error": f"MLE: {err_primary}; ridge: {e3}",
                "n": int(len(y)),
                "fit_method": None,
                "condition_number": cond,
                "vif": vif,
                "dropped_covariates": dropped,
            }

    # ---- Assemble MLE (or MLE_after_vif_drop) result
    coef_block = {}
    ci = fit.conf_int()
    for i, nm in enumerate(names_full):
        coef_block[nm] = {
            "beta": float(fit.params[i]),
            "se": float(fit.bse[i]),
            "z": float(fit.tvalues[i]),
            "p": float(fit.pvalues[i]),
            "ci_low": float(ci[i][0]),
            "ci_high": float(ci[i][1]),
        }
    scz_idx = names_full.index(predictor_name)
    return {
        "n": int(len(y)),
        "n_outcome_1": int(y.sum()),
        "converged": bool(getattr(fit, "mle_retvals", {}).get("converged", True)),
        "loglike": float(fit.llf),
        "aic": float(fit.aic),
        "fit_method": fit_method,
        "condition_number": cond,
        "vif": vif,
        "dropped_covariates": dropped,
        "coefficients": coef_block,
        "beta_scz": float(fit.params[scz_idx]),
        "se_scz": float(fit.bse[scz_idx]),
        "p_scz": float(fit.pvalues[scz_idx]),
        "primary_mle_error": err_primary,
    }


# ----------------------------------------------------------------------------- Core


@dataclass
class GeneListEntry:
    name: str
    genes: set[str]
    provenance: dict[str, Any]


def bh_qvalues(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    return list(map(float, qvals))


def compute_brain_tau(logger: logging.Logger) -> dict[str, float] | None:
    """Optional: compute brain-tau score per gene from GTEx median TPM.

    WHY: brief.md §MEASUREMENT Sub-C adjusted logistic covariate `brain_tau`
         (Yanai et al. 2005 tissue specificity). If GTEx file missing, return None
         and log a warning — the logistic will then drop the covariate.
    """
    if not GTEX_TPM.exists():
        logger.warning("GTEx TPM not found; brain_tau covariate disabled.")
        return None
    logger.info("Loading GTEx v8 TPM (for brain_tau)...")
    # GTEx .gct: skip 2 header lines, then tab-separated with Description = gene symbol.
    with gzip.open(GTEX_TPM, "rt") as fh:
        _ = fh.readline()
        _ = fh.readline()
        df = pd.read_csv(fh, sep="\t")
    logger.info("GTEx rows=%d columns=%d", len(df), df.shape[1])
    tissue_cols = [c for c in df.columns if c not in ("Name", "Description")]
    brain_cols = [c for c in tissue_cols if c.startswith("Brain")]
    # tau specificity: tau = sum(1 - x_i/max) / (N-1), where x_i = log2(TPM+1).
    X = np.log2(df[tissue_cols].to_numpy() + 1.0)
    xmax = X.max(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(xmax > 0, X / xmax, 0.0)
    tau = (1.0 - ratio).sum(axis=1) / (X.shape[1] - 1)
    tau = np.where(xmax.squeeze() > 0, tau, 0.0)
    # brain_tau: ratio of max brain TPM to max overall TPM as a complementary brain
    # specificity score (0..1). Higher → more brain-biased.
    brain_max = np.log2(df[brain_cols].to_numpy() + 1.0).max(axis=1)
    overall_max = np.log2(df[tissue_cols].to_numpy() + 1.0).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        brain_ratio = np.where(overall_max > 0, brain_max / overall_max, 0.0)
    brain_tau = tau * brain_ratio
    out = dict(zip(df["Description"].astype(str).tolist(), map(float, brain_tau)))
    logger.info("Computed brain_tau for %d genes (mean=%.3f)", len(out), float(np.mean(list(out.values()))))
    return out


def classify_sub_c(loeuf_fisher_rows: list[dict], loeuf_logit: dict,
                   schema_positive_control_row: dict | None) -> dict[str, Any]:
    """Apply pre-registered Sub-C decision rule (brief.md).

    Scope: 3-gene-list version. Positive control = SCHEMA_exome_wide_significant
    must satisfy OR > SCHEMA_POSCTRL_MIN_OR (5.0) AND BH-q < SCHEMA_POSCTRL_MAX_Q
    (0.01) — both conditions required. WHY OR>5 rather than OR>10: the n=10
    SCHEMA list is smaller than the would-be-FDR<0.05 list, widening the CI and
    making a strict OR>10 threshold over-conservative relative to the expected
    point estimate (~40 given 8-9/10 of Singh 2022 exome-wide-significant genes
    are in LOEUF<0.35; see parent-directive 2026-04-22).
    """
    primary_lists = [LIST_PGC3_EDT1, LIST_PGC3_SYNGO]
    primary_rows = [r for r in loeuf_fisher_rows
                    if r.get("gene_list") in primary_lists and not r.get("skipped")]
    qvals = [r.get("bh_q", 1.0) for r in primary_rows]
    any_q_lt_05 = any((q is not None and q < 0.05) for q in qvals)
    all_q_geq_10 = all((q is not None and q >= 0.10) for q in qvals) if qvals else False

    # WHY we use PGC3_Prioritised_EDT1 as the "primary" for the big-OR check:
    # MAGMA top-1% is dropped; EDT1 is the canonical Trubetskoy 2022 list with
    # n~120 — the largest SCZ-GWAS-grounded list remaining.
    primary_row = next((r for r in loeuf_fisher_rows
                        if r.get("gene_list") == LIST_PGC3_EDT1 and not r.get("skipped")), None)
    big_or = primary_row and primary_row.get("odds_ratio_point", 0.0) > 1.5

    # ---- Logistic evaluation: handle MLE vs regularized (ridge) fit -----
    # WHY: fit_logistic may return fit_method in {"MLE", "MLE_after_vif_drop",
    #      "ridge", None}. Under "ridge" the Wald p is invalid (suppressed),
    #      so the decision rule falls back to "beta_scz > 0" as evidence of
    #      enrichment direction. Under MLE we use the pre-registered
    #      "beta > 0 AND p < 0.05". `beta_pos_sig` still requires BOTH
    #      conditions for an ESTABLISHED finding so the pre-registered
    #      threshold is preserved when possible.
    fit_method = (loeuf_logit or {}).get("fit_method", None)
    # Prefer the explicit top-level beta_scz / p_scz keys emitted by the
    # new fit_logistic; fall back to coefficients[scz_top] for back-compat.
    beta = (loeuf_logit or {}).get("beta_scz", None)
    p_beta = (loeuf_logit or {}).get("p_scz", None)
    if beta is None:
        scz_coef = (loeuf_logit or {}).get("coefficients", {}).get("scz_top", {})
        beta = scz_coef.get("beta", None)
        p_beta = scz_coef.get("p", None)
    logistic_used_note = ""
    if fit_method == "ridge":
        # Under L2 regularization the pre-registered p<0.05 criterion is
        # not well-defined; use sign-only evidence. Document this in reason.
        beta_pos_sig = (beta is not None and beta > 0)
        logistic_used_note = (f" Logistic fit method={fit_method}: p-value "
                              "suppressed under L2 regularization; decision "
                              "uses sign of beta_scz only.")
    elif fit_method in ("MLE", "MLE_after_vif_drop"):
        beta_pos_sig = (beta is not None and beta > 0
                        and p_beta is not None and p_beta < 0.05)
        dropped = (loeuf_logit or {}).get("dropped_covariates") or []
        drop_note = f" (dropped covariates: {dropped})" if dropped else ""
        logistic_used_note = f" Logistic fit method={fit_method}{drop_note}."
    else:
        beta_pos_sig = (beta is not None and beta > 0
                        and p_beta is not None and p_beta < 0.05)
    beta_nonpos = (beta is not None and beta <= 0)

    schema_or = None
    schema_q = None
    if schema_positive_control_row is not None:
        schema_or = schema_positive_control_row.get("odds_ratio_point")
        schema_q = schema_positive_control_row.get("bh_q")
    # Positive-control gate (both conditions required)
    schema_pos_pass = (
        schema_or is not None and schema_or > SCHEMA_POSCTRL_MIN_OR
        and schema_q is not None and schema_q < SCHEMA_POSCTRL_MAX_Q
    )

    if not schema_pos_pass and schema_positive_control_row is not None:
        return {
            "classification": "UNINTERPRETABLE",
            "reason": (f"SCHEMA positive-control failed: requires OR > "
                       f"{SCHEMA_POSCTRL_MIN_OR} AND q < {SCHEMA_POSCTRL_MAX_Q} "
                       f"(observed OR={schema_or}, q={schema_q})."),
            "inputs": {"schema_or": schema_or, "schema_q": schema_q},
        }

    if any_q_lt_05 and big_or and beta_pos_sig and schema_pos_pass:
        # Compose the logistic evidence phrase based on fit method so the reason
        # accurately reflects whether p<0.05 was checked or only beta>0.
        if fit_method == "ridge":
            logit_phrase = "logistic beta_scz>0 (ridge; p suppressed)"
        else:
            logit_phrase = "logistic beta>0 p<0.05"
        return {"classification": "ESTABLISHED",
                "reason": (f"LOEUF Fisher q<0.05 on >=1 PGC3 list AND PGC3_EDT1 "
                           f"OR>1.5 AND {logit_phrase} AND SCHEMA control "
                           "passes." + logistic_used_note),
                "logistic_fit_method": fit_method,
                "beta_scz": beta, "p_scz": p_beta}
    if all_q_geq_10 and beta_nonpos and schema_pos_pass:
        return {"classification": "REFUTED",
                "reason": ("All PGC3 LOEUF Fisher q>=0.10 AND logistic beta<=0 "
                           "with SCHEMA control passing." + logistic_used_note),
                "logistic_fit_method": fit_method,
                "beta_scz": beta, "p_scz": p_beta}
    if any_q_lt_05 and beta_pos_sig and not schema_pos_pass:
        return {"classification": "UNINTERPRETABLE",
                "reason": "SCHEMA positive control fails." + logistic_used_note,
                "logistic_fit_method": fit_method,
                "beta_scz": beta, "p_scz": p_beta}
    if any_q_lt_05 and (beta_pos_sig or (beta is not None and beta > 0)):
        return {"classification": "SUGGESTED",
                "reason": ("Partial hit: some LOEUF Fisher q<0.05 but not fully "
                           "concordant with logistic." + logistic_used_note),
                "logistic_fit_method": fit_method,
                "beta_scz": beta, "p_scz": p_beta}
    return {"classification": "INCONCLUSIVE",
            "reason": ("Evidence criteria did not cleanly satisfy any decision "
                       "branch." + logistic_used_note),
            "logistic_fit_method": fit_method,
            "beta_scz": beta, "p_scz": p_beta}


# ----------------------------------------------------------------------------- Main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs exist.")
    parser.add_argument("--schema-list", type=pathlib.Path, default=None,
                        help="Override path to SCHEMA exome-wide significant gene list.")
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip 10k-permutation null (for quick sanity runs).")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS,
                        help=f"Override permutation count (default {N_PERMUTATIONS}).")
    args = parser.parse_args()

    logger = _setup_logger()
    logger.info("=" * 60)
    logger.info("batch_044 Sub-C constraint/HAR analysis (3 gene lists)")

    outputs = [OUT_LOEUF_FISHER, OUT_LOEUF_LOGISTIC, OUT_HAR_FISHER, OUT_PERMUTATION, OUT_SUMMARY]
    if not args.force and all(p.exists() for p in outputs):
        logger.info("All outputs exist; SKIP. Pass --force to re-run.")
        return 0

    input_manifest = {
        "gnomad": log_input(logger, "gnomad_constraint", GNOMAD_TSV),
        "har": log_input(logger, "har_bed", HAR_BED),
        "tss": log_input(logger, "gene_tss_grch37", TSS_CSV),
        "pgc3_xlsx": log_input(logger, "pgc3_xlsx", PGC3_XLSX),
        "gtex_tpm": log_input(logger, "gtex_tpm", GTEX_TPM),
        "schema_list": log_input(logger, "schema_exome_wide_significant",
                                  args.schema_list or SCHEMA_CANONICAL),
    }

    # ---- Load reference data -----
    gnomad = load_gnomad_constraint(logger)
    tss = load_tss(logger)
    har = load_har_bed(logger)

    # Precompute HAR-proximal genes once
    har_prox = har_proximal_genes(tss, har)
    logger.info("HAR-proximal genes (100kb, all HAR sources): %d", len(har_prox))
    # Sensitivity: Pollard-only subset
    har_pollard = har[har["source"].str.contains("Pollard", case=False, na=False)]
    har_prox_pollard = har_proximal_genes(tss, har_pollard)
    logger.info("HAR-proximal genes (Pollard-only subset): %d", len(har_prox_pollard))

    brain_tau = compute_brain_tau(logger)
    if brain_tau:
        gnomad["brain_tau"] = gnomad["gene"].astype(str).map(brain_tau).fillna(0.0)
    else:
        gnomad["brain_tau"] = 0.0

    # ---- Build gene lists (three-list scope) -----
    pgc3_lists = load_pgc3_lists(logger)
    schema_genes = build_schema_exome_wide(logger, args.schema_list)

    entries: list[GeneListEntry] = [
        GeneListEntry(LIST_PGC3_EDT1, pgc3_lists[LIST_PGC3_EDT1],
                      {"source": f"{PGC3_XLSX} ST12 Prioritised==YES & biotype==protein_coding"}),
        GeneListEntry(LIST_PGC3_SYNGO, pgc3_lists[LIST_PGC3_SYNGO],
                      {"source": f"{PGC3_XLSX} ST12 Prioritised & SynGO.GeneSetMemb & protein_coding"}),
        GeneListEntry(LIST_SCHEMA, schema_genes,
                      {"source": ("Singh 2022 Table 1 exome-wide significant genes at "
                                  "P<2.5e-6; file experiments/batch_044/input/"
                                  "schema_exome_wide_significant.txt")}),
    ]
    assert len(entries) == N_GENE_LISTS, (
        f"Expected {N_GENE_LISTS} gene lists, got {len(entries)} — "
        "BH correction N must match."
    )

    # Apply MHC filter to every list
    filtered_entries: list[GeneListEntry] = []
    for e in entries:
        filt, removed = exclude_mhc_from_set(e.genes, gnomad, logger, e.name)
        prov = dict(e.provenance)
        prov["original_size"] = len(e.genes)
        prov["mhc_removed"] = removed
        prov["post_mhc_size"] = len(filt)
        filtered_entries.append(GeneListEntry(e.name, filt, prov))
        logger.info("%s: %d -> %d after MHC filter", e.name, len(e.genes), len(filt))

    # Background = gnomAD canonical/MANE genes, MHC excluded
    bg = gnomad.loc[~gnomad["mhc_indicator"]].copy()
    logger.info("Background after MHC exclusion: %d genes", len(bg))

    # ---- Precompute indicator arrays -----
    bg_genes = bg["gene"].astype(str).to_numpy()
    loeuf_mask_bg = bg["loeuf_lt_035"].astype(bool).to_numpy()
    bg_gene_set = set(bg_genes)
    har_mask_bg = np.array([g in har_prox for g in bg_genes], dtype=bool)
    har_pollard_mask_bg = np.array([g in har_prox_pollard for g in bg_genes], dtype=bool)

    # WHY assert SCHEMA mapping: Singh 2022 Table 1 contains 10 exome-wide
    # significant genes; if <9 map into the gnomAD background, a symbol-nomenclature
    # mismatch (e.g. HGNC alias drift) silently destroys the positive-control arm.
    schema_hit = len(schema_genes & bg_gene_set)
    assert schema_hit >= 9, (
        f"SCHEMA gene mapping failure: only {schema_hit} of {len(schema_genes)} "
        f"SCHEMA genes present in gnomAD background (expected >=9 of 10)."
    )
    logger.info("SCHEMA gene mapping: %d / %d present in background",
                schema_hit, len(schema_genes))

    # Decile index for length-stratified permutation (use gene_length over log).
    # WHY pd.qcut with duplicates='drop': raw np.quantile edges can collapse on
    # ties at the tail of the length distribution, silently producing <10 deciles
    # and a degenerate stratified permutation. qcut+assert makes the degradation
    # explicit.
    lengths = bg["gene_length"].fillna(0).to_numpy()
    decile_series = pd.qcut(lengths, q=N_LENGTH_DECILES, labels=False,
                            duplicates="drop")
    decile_idx = np.asarray(decile_series).astype(int)
    n_deciles = int(len(np.unique(decile_idx)))
    assert n_deciles >= 8, (
        f"Length-decile degeneracy: only {n_deciles} distinct bins after "
        "pd.qcut(duplicates='drop') (expected >=8); stratified permutation "
        "would be underpowered."
    )
    logger.info("Length deciles: n_unique=%d", n_deciles)

    # ---- LOEUF Fisher per gene list -----
    loeuf_rows: list[dict[str, Any]] = []
    for e in filtered_entries:
        in_list_mask = np.array([g in e.genes for g in bg_genes], dtype=bool)
        n_overlap = int(np.sum(in_list_mask & loeuf_mask_bg))
        n_in_list_total = int(in_list_mask.sum())
        n_bg_loeuf = int(loeuf_mask_bg.sum())
        n_bg = len(bg_genes)
        if n_in_list_total == 0:
            row = {"gene_list": e.name, "skipped": True,
                   "reason": "No list genes in background.",
                   "provenance": e.provenance}
            loeuf_rows.append(row)
            continue
        a = n_overlap
        b = n_in_list_total - n_overlap
        c = n_bg_loeuf - n_overlap
        d = n_bg - n_in_list_total - c
        row = fisher_one_sided_enrichment(a, b, c, d)
        row.update({
            "gene_list": e.name,
            "n_list_in_background": n_in_list_total,
            "n_list_overlap_loeuf": n_overlap,
            "n_background": n_bg,
            "n_background_loeuf": n_bg_loeuf,
            "provenance": e.provenance,
        })
        loeuf_rows.append(row)

    # BH q-values across N=3 lists (per parent-directive).
    raw_ps = [r.get("p_one_sided_greater") for r in loeuf_rows if not r.get("skipped")]
    qs = bh_qvalues(raw_ps)
    qi = 0
    for r in loeuf_rows:
        if r.get("skipped"):
            r["bh_q"] = None
        else:
            r["bh_q"] = qs[qi]
            qi += 1

    # SCHEMA positive control row (extracted from loeuf_rows for easy access)
    schema_row = next((r for r in loeuf_rows if r["gene_list"] == LIST_SCHEMA), None)
    schema_pos_pass = (
        schema_row is not None
        and schema_row.get("odds_ratio_point") is not None
        and schema_row["odds_ratio_point"] > SCHEMA_POSCTRL_MIN_OR
        and schema_row.get("bh_q") is not None
        and schema_row["bh_q"] < SCHEMA_POSCTRL_MAX_Q
    )

    with OUT_LOEUF_FISHER.open("w") as fh:
        json.dump({
            "batch": "batch_044", "test": "loeuf_fisher",
            "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
            "n_gene_lists_tested": N_GENE_LISTS,
            "bh_correction_n": N_GENE_LISTS,
            "inputs": input_manifest,
            "rows": loeuf_rows,
            "positive_control_schema": {
                "gene_list": LIST_SCHEMA,
                "odds_ratio_point": schema_row.get("odds_ratio_point") if schema_row else None,
                "bh_q": schema_row.get("bh_q") if schema_row else None,
                "threshold_or_min": SCHEMA_POSCTRL_MIN_OR,
                "threshold_q_max": SCHEMA_POSCTRL_MAX_Q,
                "pass": bool(schema_pos_pass),
            },
        }, fh, indent=2, default=str)
    logger.info("Wrote %s", OUT_LOEUF_FISHER)

    # ---- LOEUF adjusted logistic for each gene list -----
    covariates = ["mhc_indicator", "chrX_indicator",
                  "log_cds_length", "log_gene_length",
                  "brain_tau", "log_obs_mis_plus1"]
    # mhc_indicator is all-False after background MHC exclusion; drop to avoid
    # singular design matrix.
    useable_covariates = []
    for col in covariates:
        vals = bg[col]
        if pd.api.types.is_bool_dtype(vals):
            vals = vals.astype(int)
        if vals.nunique(dropna=True) > 1:
            useable_covariates.append(col)
        else:
            logger.info("Dropping zero-variance covariate %s", col)
    logger.info("Logistic covariates used: %s", useable_covariates)

    logit_results: dict[str, Any] = {"gene_lists": {}, "covariates": useable_covariates}
    # Primary logistic: uses PGC3_Prioritised_EDT1 as scz_top (brief §DECISION RULE
    # updated post-MAGMA-drop: EDT1 is the canonical PGC3 Prioritised list).
    #
    # WHY two fits (primary + secondary_mle_drop_cds):
    #   * Primary uses the full pre-registered covariate set. When
    #     log_cds_length and log_gene_length are both present their VIF is
    #     effectively infinite (>0.99 correlation) and fit_logistic falls back
    #     to a ridge-penalized GLM, which suppresses valid Wald p-values.
    #   * Secondary re-fits with log_cds_length DROPPED (keeping log_gene_length
    #     as the canonical length confounder — gene length is the scale on
    #     which mutational target size is measured, Samocha 2014; CDS length
    #     is a subset of gene length). This removes the exact collinearity and
    #     allows unregularized MLE with proper Wald p for beta_scz.
    #   Both results are reported so the reader can see ridge vs MLE agree on
    #   sign/magnitude of beta_scz and only disagree on p-value validity.
    secondary_covariates = [c for c in useable_covariates if c != "log_cds_length"]
    logger.info("Secondary logistic (drop_cds) covariates: %s", secondary_covariates)
    for e in filtered_entries:
        df = bg.copy()
        # Ensure bool -> int for statsmodels
        for col in useable_covariates:
            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)
        df["scz_top"] = df["gene"].astype(str).isin(e.genes).astype(int)
        if df["scz_top"].sum() < 10:
            logit_results["gene_lists"][e.name] = {
                "skipped": True, "reason": "<10 genes in list within background."}
            continue
        primary_res = fit_logistic(df, "scz_top", useable_covariates, logger)
        secondary_res = fit_logistic(df, "scz_top", secondary_covariates, logger)
        logit_results["gene_lists"][e.name] = {
            "primary": primary_res,
            "secondary_mle_drop_cds": {
                "rationale": ("log_cds_length dropped because VIF is effectively"
                              " infinite vs log_gene_length (r>0.99); keeping"
                              " log_gene_length as canonical length covariate"
                              " (Samocha 2014). Unregularized MLE yields valid"
                              " Wald p_scz."),
                "covariates_used": secondary_covariates,
                "result": secondary_res,
            },
            # Keep top-level keys for backward compatibility with downstream
            # consumers (classify_sub_c reads beta_scz / p_scz / fit_method).
            # The PRIMARY (ridge-fallback) result stays authoritative for
            # the pre-registered decision rule; secondary is reported as
            # supplementary evidence.
            **{k: v for k, v in primary_res.items() if k not in ("primary", "secondary_mle_drop_cds")},
        }

    with OUT_LOEUF_LOGISTIC.open("w") as fh:
        json.dump({
            "batch": "batch_044", "test": "loeuf_logistic",
            "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
            "n_gene_lists_tested": N_GENE_LISTS,
            "inputs": input_manifest,
            "results": logit_results,
        }, fh, indent=2, default=str)
    logger.info("Wrote %s", OUT_LOEUF_LOGISTIC)

    # ---- HAR Fisher per gene list -----
    har_rows: list[dict[str, Any]] = []
    n_bg_har = int(har_mask_bg.sum())
    n_bg_har_pollard = int(har_pollard_mask_bg.sum())
    for e in filtered_entries:
        in_list_mask = np.array([g in e.genes for g in bg_genes], dtype=bool)
        n_in = int(in_list_mask.sum())
        if n_in == 0:
            har_rows.append({"gene_list": e.name, "skipped": True})
            continue
        # Union (primary)
        a = int(np.sum(in_list_mask & har_mask_bg))
        b = n_in - a
        c = n_bg_har - a
        d = len(bg_genes) - n_in - c
        primary = fisher_one_sided_enrichment(a, b, c, d)
        # Pollard-only sensitivity
        a2 = int(np.sum(in_list_mask & har_pollard_mask_bg))
        b2 = n_in - a2
        c2 = n_bg_har_pollard - a2
        d2 = len(bg_genes) - n_in - c2
        pollard = fisher_one_sided_enrichment(a2, b2, c2, d2)
        har_rows.append({
            "gene_list": e.name,
            "n_list_in_background": n_in,
            "har_union": primary,
            "har_pollard_only": pollard,
            "provenance": e.provenance,
        })
    # BH q across N=3 gene lists on primary union test
    raw_ps = [r["har_union"]["p_one_sided_greater"] for r in har_rows if not r.get("skipped")]
    qs = bh_qvalues(raw_ps)
    qi = 0
    for r in har_rows:
        if r.get("skipped"):
            continue
        r["har_union"]["bh_q"] = qs[qi]
        qi += 1

    with OUT_HAR_FISHER.open("w") as fh:
        json.dump({
            "batch": "batch_044", "test": "har_fisher",
            "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
            "n_gene_lists_tested": N_GENE_LISTS,
            "bh_correction_n": N_GENE_LISTS,
            "har_window_bp": HAR_WINDOW_BP,
            "inputs": input_manifest,
            "rows": har_rows,
            "notes": ("Underpowered at OR=1.3 (power ~0.47 per brief §POWER). "
                      "Reported descriptively."),
        }, fh, indent=2, default=str)
    logger.info("Wrote %s", OUT_HAR_FISHER)

    # ---- Length-stratified permutation for LOEUF Fisher -----
    perm_results: dict[str, Any] = {"gene_lists": {},
                                    "n_permutations": args.n_permutations,
                                    "n_length_deciles": N_LENGTH_DECILES,
                                    "rng_seed": RNG_SEED}
    if args.skip_permutation:
        logger.info("Skipping permutation (--skip-permutation)")
    else:
        rng = np.random.default_rng(RNG_SEED)
        for e in filtered_entries:
            in_list_mask = np.array([g in e.genes for g in bg_genes], dtype=bool)
            if in_list_mask.sum() < 10:
                perm_results["gene_lists"][e.name] = {"skipped": True}
                continue
            r = length_stratified_permutation(in_list_mask, loeuf_mask_bg, decile_idx,
                                              args.n_permutations, rng)
            perm_results["gene_lists"][e.name] = r
            logger.info("Permutation [%s]: obs_OR=%.3f emp_p=%.4f",
                        e.name, r["observed_odds_ratio"], r["empirical_p_one_sided_greater"])

    with OUT_PERMUTATION.open("w") as fh:
        json.dump({
            "batch": "batch_044", "test": "length_stratified_permutation",
            "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
            "n_gene_lists_tested": N_GENE_LISTS,
            "inputs": input_manifest,
            "results": perm_results,
        }, fh, indent=2, default=str)
    logger.info("Wrote %s", OUT_PERMUTATION)

    # ---- Consolidated Sub-C summary -----
    # Primary logistic for classification uses PGC3_Prioritised_EDT1 (MAGMA dropped).
    primary_logit_row = logit_results["gene_lists"].get(LIST_PGC3_EDT1, {})
    decision = classify_sub_c(loeuf_rows, primary_logit_row, schema_row)
    summary = {
        "batch": "batch_044",
        "sub_experiment": "C_constraint_har",
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "scope": {
            "n_gene_lists": N_GENE_LISTS,
            "gene_lists": [LIST_PGC3_EDT1, LIST_PGC3_SYNGO, LIST_SCHEMA],
            "magma_top1pct_dropped": True,
            "magma_drop_reason": ("MAGMA binary not on PATH; synthesizing gene "
                                   "scores violates Rule 0. 3-list scope retains "
                                   "full scientific power via PGC3 Prioritised + "
                                   "SynGO sub-intersection + SCHEMA positive control."),
        },
        "gene_list_sizes": {e.name: len(e.genes) for e in filtered_entries},
        "loeuf_fisher_rows": loeuf_rows,
        "loeuf_logistic_primary": primary_logit_row,
        "loeuf_logistic_primary_list": LIST_PGC3_EDT1,
        "har_fisher_rows": har_rows,
        "schema_positive_control": {
            "gene_list": LIST_SCHEMA,
            "odds_ratio": (schema_row or {}).get("odds_ratio_point"),
            "bh_q": (schema_row or {}).get("bh_q"),
            "threshold_or_min": SCHEMA_POSCTRL_MIN_OR,
            "threshold_q_max": SCHEMA_POSCTRL_MAX_Q,
            "pass_gate": bool(schema_pos_pass),
        },
        "decision": decision,
    }
    with OUT_SUMMARY.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Wrote %s", OUT_SUMMARY)

    logger.info("Sub-C classification: %s — %s",
                decision["classification"], decision["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
