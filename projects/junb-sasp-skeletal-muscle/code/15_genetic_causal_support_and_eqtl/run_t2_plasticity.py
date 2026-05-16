"""Batch 065 Analysis T2 — Five-cohort Fisher-combined plasticity meta-analysis.

Tests whether the HLMA aging-UP/DOWN signatures (Vascular, FAP, MuSC) are
reversed in aged exercise cohorts across five independent studies:

    ACUTE stratum:   MoTrPAC 24-hr (from iter 064) + Chambers GSE151066 (0/3h post cycling)
    CHRONIC stratum: Robinson GSE97084 (12-wk HIIT/RT/CT) + Melov GSE8479 (6-mo RT) +
                     Trappe GSE28422 (12-wk RT)

Primary criterion (brief v2, post-Critic 2 B3): SIGN-CONCORDANCE (NES<0 for
aging-UP; NES>0 for aging-DOWN) across >= 3 of 5 cohorts AND Fisher-combined
p < 2.08e-3 (Bonferroni 0.05/24) WITHIN stratum. |NES|>=1.5 was ex-post tuned
on F064_04 and is reported but not used as the decision gate.

Design source: `experiments/batch_065/brief.md` §T2 (lines 131-206).

MANDATORY pre-run steps (brief §T2):
  (a) Independence audit: extract PI/institution/donor IDs per cohort, flag
      donor-overlap risk (esp. Trappe lab multi-study overlap).
  (b) Gene coverage pre-check: per cohort x per signature, report coverage =
      (# probed genes) / 100. If <0.70 use restricted-signature GSEA. If <0.50
      BLOCK that cell.

Outputs (all under experiments/batch_065/):
    t2_de_per_cohort/GSE*_de.csv     per-cohort gene-level DE tables
    t2_coverage.csv                  platform gene-coverage per cohort x signature
    t2_independence_audit.csv        PI/institution/donor-overlap audit
    t2_gsea_per_cohort.csv           NES + p for all 6 signatures x 5 cohorts
    t2_fisher_combined.csv           meta-p per (stratum, compartment, direction)
    t2_summary.json                  final verdict + key numbers
    logs/t2_stdout.log               full stdout log

WHY reuse batch_064 MoTrPAC GSEA: iter 064 already ran GSEA for MoTrPAC 24-hr
using identical signatures and identical nulls (gseapy.prerank + p_sig_rand).
Re-running would be a Rule-1 violation; we cite `c_gsea_results.csv` directly.

WHY re-implement DE rather than fgsea/limma-R: R is not installed locally
(verified in batch_064 preflight line 29). We replicate limma eBayes moderated
t (Smyth 2004) using statsmodels OLS per gene + Newton-Raphson empirical-Bayes
shrinkage of the variance (Phipson 2016 linearized). This matches the brief
measurement spec line 179 ("limma-style eBayes via custom").

WHY paired-design where available: within-subject pre/post contrast controls
for between-subject variation (standard for pre-post exercise studies; Appleton
2019 J Appl Physiol). For GSE8479 (Melov), the PRIMARY paired contrast is
aged-post vs aged-pre (24 old subjects, 12 pre + 12 post); GSE28422 (Trappe)
is trained-aged vs untrained-aged within-subject; GSE97084 (Robinson) is
aged-post vs aged-pre; GSE151066 (Chambers) is acute-post vs pre within subject.

CITATIONS:
    Fisher 1925, Fisher's combined probability test.
    Stouffer 1949, SSRC Studies in Social Psychology in WWII (Z-method).
    Smyth 2004, Linear models and empirical bayes for microarray.
    Subramanian 2005, GSEA (weighted-KS).
    Liberzon 2015, MSigDB Hallmark collection.
    Phipson 2016, Robust hyperparameter estimation for limma.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
BATCH_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_065")
DATA_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/data/exercise_aged")
DOWNLOAD_DIR = BATCH_DIR / "downloads"
DE_DIR = BATCH_DIR / "t2_de_per_cohort"
LOG_FP = BATCH_DIR / "logs" / "t2_stdout.log"

# WHY reuse: iter 064 c_gsea_results.csv already has MoTrPAC 24-hr GSEA NES
# per compartment x direction (see brief §T2 line 155).
MOTRPAC_GSEA = Path(
    "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_064/c_gsea_results.csv"
)
MOTRPAC_TIMEPOINT_KEY = "post_24_hr"

# HLMA 6 signatures — identical to iter 064 (brief §T2 line 178: "HLMA 6-signature").
HLMA_SIGNATURES = Path(
    "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_064/c_signatures.csv"
)

# GEO SOFT files already downloaded (brief pre-registered list).
SOFT_FILES = {
    "GSE97084": DATA_DIR / "GSE97084_family.soft.gz",
    "GSE151066": DATA_DIR / "GSE151066_family.soft.gz",
    "GSE28422": DATA_DIR / "GSE28422_family.soft.gz",
    "GSE8479": DATA_DIR / "GSE8479_family.soft.gz",
}

# RNA-seq supplementary count files (NCBI GEO FTP; downloaded on first run).
RNASEQ_SUPP = {
    "GSE97084": [
        ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE97nnn/GSE97084/suppl/"
         "GSE97084_GeneCount_raw.tsv.gz"),
        ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE97nnn/GSE97084/suppl/"
         "GSE97084_GeneCount_raw_2.tsv.gz"),
    ],
    "GSE151066": [
        ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE151nnn/GSE151066/suppl/"
         "GSE151066_rsem_genes_count.txt.gz"),
    ],
}

# Melov GPL2700 annotation file (RefSeq -> gene symbol) from NCBI GEO FTP.
# WHY external fetch: GSE8479 SOFT contains only GB_ACC (RefSeq) on its platform
# table; no gene symbol. GPL2700.annot.gz is the canonical NCBI-curated annotation.
GPL2700_ANNOT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL2nnn/GPL2700/annot/GPL2700.annot.gz"
)

# Output files.
OUT_COVERAGE = BATCH_DIR / "t2_coverage.csv"
OUT_AUDIT = BATCH_DIR / "t2_independence_audit.csv"
OUT_GSEA = BATCH_DIR / "t2_gsea_per_cohort.csv"
OUT_FISHER = BATCH_DIR / "t2_fisher_combined.csv"
OUT_SUMMARY = BATCH_DIR / "t2_summary.json"

# Pre-registered constants (brief §T2).
COVERAGE_RESTRICT_THRESHOLD = 0.70     # below -> restricted-signature GSEA
COVERAGE_BLOCK_THRESHOLD = 0.50        # below -> BLOCK cell (brief line 193)
NES_MAGNITUDE_DISCLOSED = 1.5          # ex-post; reported, not gate (brief line 160)
BONFERRONI_ALPHA = 0.05 / 24           # = 2.08e-3 (brief line 185)
N_PERMUTATIONS = 1000                  # matches batch_064/c_run_c_motrpac.py
N_SIG_RAND_PERM = 1000                 # matches batch_064/c_run_c_motrpac.py
SEED = 42                              # brief line 98 + §T2 measurement

# Cohort definitions for meta-analysis.
STRATA = {
    "ACUTE": ["MoTrPAC_24hr", "GSE151066"],
    "CHRONIC": ["GSE97084", "GSE8479", "GSE28422"],
}
ALL_COHORTS = ["MoTrPAC_24hr"] + list(SOFT_FILES.keys())

# Compartments and directions (HLMA 6 signatures).
COMPARTMENTS = ["Vascular", "MuSC", "FAP"]
DIRECTIONS = ["UP", "DOWN"]


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    LOG_FP.parent.mkdir(parents=True, exist_ok=True)
    fmt = "[%(asctime)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FP, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("t2")


# -----------------------------------------------------------------------------
# Generic fetch helper
# -----------------------------------------------------------------------------
def fetch_if_missing(url: str, local_path: Path, log: logging.Logger,
                     timeout: int = 120, max_retries: int = 3) -> Path:
    """Download file from URL to local_path if not already present.

    WHY: brief §T2 MEASUREMENT line 174 states all SOFT files pre-downloaded but
    supplementary count matrices and GPL annotations need to be fetched at run
    time. Transient-errors rule (CLAUDE.md) mandates 3 retries with backoff.
    """
    if local_path.exists() and local_path.stat().st_size > 0:
        log.info("  [cache hit] %s (%.1f KB)", local_path.name,
                 local_path.stat().st_size / 1024)
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            log.info("  [download] attempt %d: %s", attempt, url)
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            tmp = local_path.with_suffix(local_path.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.rename(local_path)
            log.info("    -> %s (%.1f KB, %.1fs)", local_path.name,
                     len(data) / 1024, time.time() - t0)
            return local_path
        except Exception as e:  # noqa: BLE001
            log.warning("    attempt %d failed: %s", attempt, e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {url} after {max_retries} attempts")


# -----------------------------------------------------------------------------
# SOFT metadata parsing (independence audit + sample design)
# -----------------------------------------------------------------------------
def parse_soft_metadata(soft_path: Path, log: logging.Logger) -> Tuple[pd.DataFrame, dict]:
    """Parse a GSE SOFT file into (samples_df, series_meta).

    samples_df columns:
      sample_id, title, age_group, age_numeric, timepoint, exercise_type,
      subject_id, gender, contact_name, contact_institute

    series_meta dict:
      pi (best-guess: first Series_contributor or platform contact),
      institute, contributors[], platform, series_id

    WHY SOFT not GEOparse: GEOparse loads everything into memory (>6M lines for
    GSE28422); we only need metadata per sample. Manual streaming is O(samples)
    memory and faster. We still use GEOparse for the embedded platform/sample
    value tables (microarrays only).
    """
    samples: List[dict] = []
    current: Optional[dict] = None
    series: Dict[str, List[str]] = {}
    in_series = False
    in_sample = False
    with gzip.open(soft_path, "rt") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("^SERIES"):
                in_series = True
                in_sample = False
                continue
            if line.startswith("^SAMPLE"):
                in_series = False
                in_sample = True
                if current is not None:
                    samples.append(current)
                sample_id = line.split("=", 1)[1].strip()
                current = {"sample_id": sample_id}
                continue
            if line.startswith("^PLATFORM"):
                in_series = False
                in_sample = False
                series.setdefault("platform", []).append(line.split("=", 1)[1].strip())
                continue
            if not line.startswith("!"):
                continue
            if "=" not in line:
                continue
            key, val = line[1:].split("=", 1)
            key = key.strip()
            val = val.strip()
            if in_series:
                series.setdefault(key, []).append(val)
            elif in_sample and current is not None:
                # Accumulate multi-valued characteristics
                if key in current:
                    if not isinstance(current[key], list):
                        current[key] = [current[key]]
                    current[key].append(val)
                else:
                    current[key] = val
    if current is not None:
        samples.append(current)

    # Build tidy DataFrame
    rows: List[dict] = []
    for s in samples:
        r: Dict[str, object] = {
            "sample_id": s["sample_id"],
            "title": s.get("Sample_title", ""),
            "contact_name": s.get("Sample_contact_name", ""),
            "contact_institute": s.get("Sample_contact_institute", ""),
        }
        # Characteristics are multi-valued "key: value" strings
        char = s.get("Sample_characteristics_ch1", [])
        if isinstance(char, str):
            char = [char]
        kv: Dict[str, str] = {}
        for c in char:
            if ":" in c:
                k, v = c.split(":", 1)
                kv[k.strip().lower()] = v.strip()
        r["characteristics"] = kv
        rows.append(r)
    df = pd.DataFrame(rows)

    series_meta = {
        "series_id": series.get("Series_geo_accession", [""])[0],
        "contributors": series.get("Series_contributor", []),
        "platform": series.get("platform", [""])[0],
        "pubmed_id": series.get("Series_pubmed_id", [""])[0],
    }
    log.info("  parsed SOFT: %d samples, platform=%s, pmid=%s",
             len(df), series_meta["platform"], series_meta["pubmed_id"])
    return df, series_meta


def extract_design_columns(samples_df: pd.DataFrame, cohort_id: str,
                           log: logging.Logger) -> pd.DataFrame:
    """Cohort-specific regex-based extraction of (subject_id, age_group,
    timepoint, exercise_type) from title + characteristics.

    WHY per-cohort parsing: sample titles follow different conventions per GEO
    submitter (Trappe uses "T1_Pre_Male_Young (81373)"; Robinson uses
    "10B Young Combined PreTraining"; Chambers uses cryptic barcodes with
    subject.id in characteristics; Melov uses "Subject code A30" with age
    numeric). A unified regex is not tractable; explicit per-cohort parsing is
    correct. WHY extract: paired DE contrast requires knowing subject ID per
    pre/post pair.
    """
    df = samples_df.copy()
    df["cohort_id"] = cohort_id
    df["subject_id"] = ""
    df["age_group"] = ""
    df["age_numeric"] = np.nan
    df["timepoint"] = ""
    df["exercise_type"] = ""

    if cohort_id == "GSE97084":
        # Title: "10B Young Combined PreTraining" ; characteristics: age/exercise type/biopsy timepoint
        for i, r in df.iterrows():
            title = r["title"]
            kv = r["characteristics"]
            # First token before space is the subject code (e.g. 10A, 10B, 11A)
            m = re.match(r"^(\S+)\s+(Young|Old)\s+(\S+)\s+(\S+)", title)
            if m:
                # Strip last char (A/B/C replicate letter) to get subject_id. Actually
                # per Mayo convention subject is numeric-only; the trailing letter
                # indicates biopsy replicate. Preserve as-is but capture numeric.
                token = m.group(1)
                num_match = re.match(r"(\d+)", token)
                df.at[i, "subject_id"] = num_match.group(1) if num_match else token
            df.at[i, "age_group"] = kv.get("age", "")
            df.at[i, "exercise_type"] = kv.get("exercise type", "")
            df.at[i, "timepoint"] = kv.get("biopsy timepoint", "")

    elif cohort_id == "GSE151066":
        # subject.id is explicit in characteristics. cohort = Active|Sedentary.
        # time.point = Pre|Post|3hrPost.
        for i, r in df.iterrows():
            kv = r["characteristics"]
            df.at[i, "subject_id"] = kv.get("subject.id", "")
            df.at[i, "timepoint"] = kv.get("time.point", "")
            df.at[i, "exercise_type"] = kv.get("cohort", "")  # Active vs Sedentary
            # All subjects in GSE151066 are older adults (mean ~70 per paper);
            # no young control arm. Assign Old.
            df.at[i, "age_group"] = "Old"

    elif cohort_id == "GSE28422":
        # Title: "T1_Pre_Male_Young (81373)" or "T2_Pre4hr_Male_Old (XXXXX)"
        # Characteristics: age: Young|Old; time point: Basal|4hr post-RE
        # training state: Untrained|Trained
        for i, r in df.iterrows():
            title = r["title"]
            kv = r["characteristics"]
            m = re.search(r"\((\d+)\)\s*$", title)
            if m:
                df.at[i, "subject_id"] = m.group(1)
            df.at[i, "age_group"] = kv.get("age", "")
            df.at[i, "timepoint"] = kv.get("time point", "")
            df.at[i, "exercise_type"] = kv.get("training state", "")

    elif cohort_id == "GSE8479":
        # Title: "Subject code A30" ; Age numeric; Gender; Sample Group Y/O
        for i, r in df.iterrows():
            title = r["title"]
            kv = r["characteristics"]
            m = re.search(r"[A-Z]\d+", title)
            if m:
                df.at[i, "subject_id"] = m.group(0)
            try:
                df.at[i, "age_numeric"] = float(kv.get("age", ""))
            except (ValueError, TypeError):
                pass
            group = kv.get("sample group", "")
            # Y=young, O=old pre, OE=old post-exercise
            if group == "Y":
                df.at[i, "age_group"] = "Young"
                df.at[i, "timepoint"] = "NA"
            elif group == "O":
                df.at[i, "age_group"] = "Old"
                df.at[i, "timepoint"] = "Pre"
            elif group in ("OE", "OX"):
                df.at[i, "age_group"] = "Old"
                df.at[i, "timepoint"] = "Post"
            else:
                df.at[i, "age_group"] = ""
                df.at[i, "timepoint"] = group

    log.info("  [%s] extracted design: age=%s timepoint=%s exercise=%s subjects=%d",
             cohort_id,
             sorted(df["age_group"].unique().tolist()),
             sorted(df["timepoint"].unique().tolist()),
             sorted(df["exercise_type"].unique().tolist()),
             df["subject_id"].nunique())
    return df


# -----------------------------------------------------------------------------
# Independence audit (brief §T2 line 176, Critic 1 C3)
# -----------------------------------------------------------------------------
def run_independence_audit(all_samples: Dict[str, pd.DataFrame],
                           all_meta: Dict[str, dict],
                           log: logging.Logger) -> pd.DataFrame:
    """Audit donor-overlap risk across cohorts.

    Pre-registered criteria:
      HIGH   : same PI/lab + overlapping sample-ID nomenclature
      MEDIUM : same PI/lab OR overlapping institution OR known published overlap
      LOW    : distinct PI + distinct institution + no shared sample IDs

    KNOWN a priori concerns (from brief v2 §T2 and pubmed metadata):
      - Trappe lab (GSE28422) has authored multiple aged-RT studies;
        GSE151066 has Todd Trappe + Scott Trappe as contributors -> MEDIUM risk
      - MoTrPAC has no overlapping contributors documented here
      - Mayo (GSE97084), Buck Institute (GSE8479), Ball State (GSE28422),
        Mount Sinai/AdventHealth (GSE151066) are distinct institutions
    """
    rows: List[dict] = []
    # Build a quick sample-ID set per cohort to intersect
    id_sets = {c: set(df["subject_id"].dropna().astype(str).tolist())
               for c, df in all_samples.items()}
    # Include MoTrPAC placeholder (no donor IDs available locally)
    motrpac_pi = "Snyder/Bodine/Sanford (MoTrPAC consortium)"
    motrpac_inst = "MoTrPAC DCC (Stanford/UAB/others)"

    all_rows = []
    # MoTrPAC row
    all_rows.append({
        "cohort_id": "MoTrPAC_24hr",
        "PI": motrpac_pi,
        "institution": motrpac_inst,
        "n_samples": "N/A (aggregate DA only)",
        "n_subjects_parsed": 0,
        "contributors": "MoTrPAC consortium",
        "donor_overlap_risk": "LOW",
        "donor_overlap_rationale": ("MoTrPAC HUMA cohort (human aerobic/resistance "
                                     "acute exercise); no known donor overlap with "
                                     "Trappe/Robinson/Melov/Chambers; different "
                                     "consortium."),
    })
    for cohort, df in all_samples.items():
        meta = all_meta[cohort]
        contribs = "; ".join(meta["contributors"])
        inst_set = sorted(set(df["contact_institute"].dropna().tolist()) - {""})
        inst_str = "; ".join(inst_set) if inst_set else "UNKNOWN"
        pi = meta["contributors"][0] if meta["contributors"] else "UNKNOWN"

        # Donor ID overlap across cohorts
        overlaps = []
        for other, other_ids in id_sets.items():
            if other == cohort or not other_ids or not id_sets[cohort]:
                continue
            shared = id_sets[cohort] & other_ids
            # Filter trivial overlaps (empty or pure-alpha short codes that can collide)
            # by ignoring codes shorter than 3 chars.
            shared = {s for s in shared if len(s) >= 3 and s.strip()}
            if shared:
                overlaps.append(f"{other}:{len(shared)}")

        # Trappe-lab shared-authorship check
        trappe_present = any("Trappe" in c for c in meta["contributors"])

        risk = "LOW"
        rationales = []
        if overlaps:
            risk = "HIGH"
            rationales.append(f"donor-ID overlap detected: {', '.join(overlaps)}")
        if trappe_present and cohort != "GSE28422":
            risk = max(risk, "MEDIUM", key=["LOW", "MEDIUM", "HIGH"].index)
            rationales.append(
                f"Trappe-lab contributor present (cross-study Trappe-lab participation "
                f"possible with GSE28422); contributors={contribs}"
            )
        if cohort == "GSE28422":
            rationales.append("Trappe lab primary investigator; known multi-study "
                              "participant lab; sample IDs are 5-digit numeric codes.")
        if cohort == "GSE151066":
            rationales.append("Chambers/Coen-Rubenstein study at Mount Sinai/Translational "
                              "Research Institute; Trappe co-authored paper but this "
                              "is a distinct subject cohort (N=19 older adults).")
        if cohort == "GSE97084":
            rationales.append("Robinson/Nair Mayo Clinic cohort; distinct institution and "
                              "PI from all other cohorts.")
        if cohort == "GSE8479":
            rationales.append("Melov Buck Institute cohort; distinct institution.")

        rat = " | ".join(rationales) if rationales else "distinct PI, institution, and sample-ID space"
        all_rows.append({
            "cohort_id": cohort,
            "PI": pi,
            "institution": inst_str,
            "n_samples": len(df),
            "n_subjects_parsed": df["subject_id"].replace("", np.nan).dropna().nunique(),
            "contributors": contribs,
            "donor_overlap_risk": risk,
            "donor_overlap_rationale": rat,
        })

    audit = pd.DataFrame(all_rows)
    log.info("  independence audit: %s", audit[["cohort_id", "donor_overlap_risk"]].to_dict("records"))
    return audit


# -----------------------------------------------------------------------------
# Microarray expression matrix loaders (GEOparse)
# -----------------------------------------------------------------------------
def load_microarray_matrix(cohort_id: str, soft_path: Path,
                            log: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load expression matrix and platform probe->symbol map for a microarray cohort.

    Returns (expr_df, probe_map):
      expr_df: rows=probe_id, cols=sample_id, values=log-intensity (quantile-normalized
               for GPL2700; RMA-equivalent for GPL570 — GEO submitters preprocess)
      probe_map: DataFrame[probe_id, gene_symbol]

    WHY GEOparse for microarrays: these SOFT files have full embedded platform
    tables + per-sample value tables; GEOparse parses them correctly. Rule 1.
    WHY log transform (GPL2700): Melov submitted linear-scale Illumina values;
    log2(x+1) is required for linear-model DE (Du 2008 Lumi).
    WHY RMA check (GPL570): GSE28422 submitters uploaded RMA-processed values
    (confirmed by VALUE range in sample tables). No further transform needed.
    """
    import GEOparse
    log.info("  [%s] loading GEOparse SOFT (microarray)", cohort_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gse = GEOparse.get_GEO(filepath=str(soft_path), silent=True)

    # Build probe -> symbol map
    gpl = list(gse.gpls.values())[0]
    gpl_tbl = gpl.table
    log.info("  [%s] platform=%s probes=%d cols=%s",
             cohort_id, gpl.name, len(gpl_tbl), list(gpl_tbl.columns))

    if "Gene Symbol" in gpl_tbl.columns:
        probe_map = gpl_tbl[["ID", "Gene Symbol"]].rename(
            columns={"ID": "probe_id", "Gene Symbol": "gene_symbol"}
        )
        # For GPL570 multi-symbol entries "A /// B" prefer first symbol
        probe_map["gene_symbol"] = probe_map["gene_symbol"].astype(str).str.split(" /// ").str[0]
    elif "GB_ACC" in gpl_tbl.columns:
        # GPL2700 — need external RefSeq->symbol mapping
        probe_map = gpl_tbl[["ID", "GB_ACC"]].rename(
            columns={"ID": "probe_id", "GB_ACC": "refseq"}
        )
        # Fetch GPL2700.annot.gz for gene symbols
        annot_path = DOWNLOAD_DIR / "GPL2700.annot.gz"
        fetch_if_missing(GPL2700_ANNOT_URL, annot_path, log)
        annot_map = parse_gpl_annot(annot_path, log)
        probe_map = probe_map.merge(annot_map, on="probe_id", how="left")
    else:
        raise ValueError(f"cannot find gene-symbol column in GPL table for {cohort_id}")

    probe_map = probe_map.dropna(subset=["probe_id"])
    probe_map["probe_id"] = probe_map["probe_id"].astype(str)
    probe_map["gene_symbol"] = probe_map["gene_symbol"].fillna("").astype(str)
    probe_map = probe_map[probe_map["gene_symbol"] != ""]
    log.info("  [%s] probe_map: %d probes with gene symbol", cohort_id, len(probe_map))

    # Build expression matrix from sample value tables
    expr_cols: Dict[str, pd.Series] = {}
    for gsm_id, gsm in gse.gsms.items():
        tbl = gsm.table
        if tbl.empty or "VALUE" not in tbl.columns:
            continue
        s = pd.Series(tbl["VALUE"].values, index=tbl["ID_REF"].astype(str).values, name=gsm_id)
        expr_cols[gsm_id] = s
    if not expr_cols:
        raise ValueError(f"no sample tables found in {cohort_id}")
    expr = pd.DataFrame(expr_cols)
    # Force numeric; some Illumina tables use '.' or blanks
    expr = expr.apply(pd.to_numeric, errors="coerce")
    log.info("  [%s] expression matrix: %d probes x %d samples (raw)",
             cohort_id, expr.shape[0], expr.shape[1])

    # Log-transform for GPL2700 (Illumina linear-scale). GPL570 is already log2 (Affymetrix RMA).
    if gpl.name == "GPL2700":
        # If any values are large (>50), assume linear; apply log2(x+1).
        if np.nanmedian(expr.values) > 50:
            log.info("  [%s] applying log2(x+1) (detected linear-scale values)", cohort_id)
            expr = np.log2(expr.clip(lower=0) + 1.0)

    # Drop all-NaN rows (sometimes controls)
    expr = expr.dropna(axis=0, how="all")
    return expr, probe_map


def parse_gpl_annot(annot_path: Path, log: logging.Logger) -> pd.DataFrame:
    """Parse NCBI GEO GPL.annot.gz to extract probe_id -> gene_symbol mapping.

    File format: comments (^DATABASE, ^PLATFORM), then table header, then
    tab-separated data. Relevant cols: 'ID' and 'Gene symbol'.
    """
    rows: List[Tuple[str, str]] = []
    header = None
    in_table = False
    id_idx = sym_idx = None
    with gzip.open(annot_path, "rt", errors="ignore") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("!") or line.startswith("^"):
                continue
            if header is None:
                header = line.rstrip("\n").split("\t")
                # Locate columns (case-insensitive)
                lower = [h.lower() for h in header]
                if "id" in lower:
                    id_idx = lower.index("id")
                if "gene symbol" in lower:
                    sym_idx = lower.index("gene symbol")
                elif "gene_symbol" in lower:
                    sym_idx = lower.index("gene_symbol")
                if id_idx is None or sym_idx is None:
                    raise ValueError(f"GPL annot missing ID/Gene symbol cols: {header}")
                in_table = True
                continue
            if in_table:
                parts = line.rstrip("\n").split("\t")
                if len(parts) > max(id_idx, sym_idx):
                    pid = parts[id_idx].strip()
                    sym = parts[sym_idx].strip()
                    if pid and sym:
                        rows.append((pid, sym))
    df = pd.DataFrame(rows, columns=["probe_id", "gene_symbol"])
    log.info("  parsed GPL annot: %d probe->symbol rows", len(df))
    return df


# -----------------------------------------------------------------------------
# RNA-seq count matrix loaders
# -----------------------------------------------------------------------------
def load_rnaseq_matrix(cohort_id: str, samples_df: pd.DataFrame,
                       log: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download and load the RNA-seq count matrix for Robinson/Chambers.

    Returns (counts_df, probe_map) where probe_map is gene-symbol-indexed
    identity (gene_id=gene_symbol).

    WHY CPM+log2 normalization (not DESeq2): we need a t-like statistic per
    gene for GSEA ranking. `statsmodels.OLS` on log2(CPM+1) produces a Wald t
    that is a valid ranking metric (Law 2014 voom philosophy). DESeq2/limma-voom
    would be more rigorous but require R or additional dependencies; log2CPM+OLS
    is an acceptable approximation for GSEA ranking (Ritchie 2015 limma manual
    acknowledges t-statistic is the key input).
    """
    urls = RNASEQ_SUPP[cohort_id]
    dfs = []
    for url in urls:
        fname = url.rsplit("/", 1)[1]
        local = DOWNLOAD_DIR / fname
        fetch_if_missing(url, local, log)
        log.info("  [%s] parsing %s", cohort_id, local.name)
        # Detect separator
        with gzip.open(local, "rt", errors="ignore") as fh:
            header = fh.readline()
        sep = "\t" if "\t" in header else ","
        df = pd.read_csv(local, sep=sep, index_col=0, low_memory=False)
        log.info("    loaded: %d genes x %d samples", df.shape[0], df.shape[1])
        dfs.append(df)
    if len(dfs) > 1:
        # GSE97084 split across 2 files — concat along sample axis (union of samples);
        # row-index (gene) intersection across files.
        common_genes = dfs[0].index
        for d in dfs[1:]:
            common_genes = common_genes.intersection(d.index)
        counts = pd.concat([d.loc[common_genes] for d in dfs], axis=1)
        log.info("  [%s] merged: %d common genes x %d samples", cohort_id, *counts.shape)
    else:
        counts = dfs[0]

    # Coerce to numeric; force column/row labels to str
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)
    counts = counts.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Align to samples_df: match columns by GSM accession or by sample title suffix
    gsm_list = samples_df["sample_id"].tolist()
    # Columns may be GSM_* or just sample titles
    matched = [c for c in counts.columns if c in gsm_list]
    if len(matched) < 0.5 * len(gsm_list):
        # Try matching via title tokens
        title_map = dict(zip(samples_df["title"], samples_df["sample_id"]))
        remap = {}
        for c in counts.columns:
            if c in gsm_list:
                remap[c] = c
            elif c in title_map:
                remap[c] = title_map[c]
            else:
                # best-effort substring match against titles
                for t, gsm in title_map.items():
                    if t and (t in c or c in t):
                        remap[c] = gsm
                        break
        counts = counts.rename(columns=remap)
        matched = [c for c in counts.columns if c in gsm_list]

    log.info("  [%s] matched %d / %d samples to count columns",
             cohort_id, len(matched), len(gsm_list))
    # Keep only matched columns
    counts = counts[[c for c in counts.columns if c in gsm_list]]

    # Gene-ID handling: GSE97084 rows look like ENSG or symbols; GSE151066 uses
    # ENSG with symbol. We'll return symbol-indexed matrix when possible.
    # Detect format of index
    first_index = counts.index[0] if len(counts.index) else ""
    if first_index.startswith("ENSG"):
        # Need ENSG -> symbol mapping (from GTEx GCT, same as batch_064).
        ensg_to_sym = load_ensg_to_symbol(log)
        # Strip version suffix for matching
        base_idx = counts.index.str.split(".").str[0]
        sym = base_idx.map(ensg_to_sym)
        valid = sym.notna() & (sym != "")
        log.info("  [%s] ENSG->symbol: %d/%d mapped", cohort_id, int(valid.sum()), len(sym))
        counts = counts.loc[valid]
        counts.index = sym[valid].values
        # Collapse duplicate symbols by summing counts (standard practice)
        counts = counts.groupby(level=0).sum()

    probe_map = pd.DataFrame({
        "probe_id": counts.index.astype(str),
        "gene_symbol": counts.index.astype(str),
    })
    return counts, probe_map


_ENSG_CACHE: Optional[Dict[str, str]] = None


def load_ensg_to_symbol(log: logging.Logger) -> Dict[str, str]:
    """Build ENSG (unversioned) -> gene_symbol dict from GTEx GCT.

    WHY GTEx GCT: same rationale as batch_064/run_c_motrpac.py — canonical
    GENCODE source, no network. Rule 1 reuse of an existing file.
    """
    global _ENSG_CACHE
    if _ENSG_CACHE is not None:
        return _ENSG_CACHE
    gct = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/data/GTEx/muscle/"
               "gene_tpm_muscle_skeletal.gct.gz")
    if not gct.exists():
        log.warning("  GTEx GCT not found at %s — ENSG mapping will be empty", gct)
        _ENSG_CACHE = {}
        return _ENSG_CACHE
    m: Dict[str, str] = {}
    with gzip.open(gct, "rt") as fh:
        fh.readline()  # version
        fh.readline()  # dims
        header = fh.readline().rstrip("\n").split("\t")
        try:
            name_idx = header.index("Name")
            desc_idx = header.index("Description")
        except ValueError:
            _ENSG_CACHE = {}
            return _ENSG_CACHE
        needed = max(name_idx, desc_idx) + 1
        for line in fh:
            parts = line.split("\t", needed)
            if len(parts) > max(name_idx, desc_idx):
                ensg_v = parts[name_idx]
                sym = parts[desc_idx]
                base = ensg_v.split(".")[0]
                if base not in m:
                    m[base] = sym
    _ENSG_CACHE = m
    log.info("  ENSG->symbol cache: %d entries", len(m))
    return m


# -----------------------------------------------------------------------------
# Per-cohort DE: paired pre/post (or aged/young) OLS with moderated t
# -----------------------------------------------------------------------------
def build_design(samples_df: pd.DataFrame, cohort_id: str,
                 log: logging.Logger) -> Optional[pd.DataFrame]:
    """Return the subset of samples used for the aged pre/post OR aged/young
    contrast, with columns: sample_id, subject_id, contrast_val (0/1).

    contrast_val=1 is the "post-intervention" arm (expected to attenuate aging);
    contrast_val=0 is the "pre-intervention / baseline / young control" arm.

    Paired design: when subject_id maps 1:1 across levels, we fit
      y ~ beta * contrast + subject FE.
    Unpaired: y ~ contrast only.

    WHY this mapping per cohort:
      GSE97084 Robinson: Old + {HIIT, Resistance, Combined} × {PreTraining, PostTraining}.
          Contrast: Old post-training vs Old pre-training (paired, pooled across
          exercise types per brief's "aged-Post-HIIT/RT vs aged-Pre-HIIT/RT"
          spec line 180).
      GSE151066 Chambers: all Old; time.point Pre/Post/3hrPost.
          ACUTE contrast: Post (immediately post) vs Pre (paired).
          3hrPost excluded to match MoTrPAC 24-hr acute framing while keeping a
          consistent acute-response definition per cohort.
      GSE28422 Trappe: Old × {Untrained Basal, Trained Basal, Trained 4hr-post}.
          CHRONIC contrast: Old Trained-Basal vs Old Untrained-Basal (within
          subject paired; 4hr-post excluded as acute-superimposed).
      GSE8479 Melov: Old Pre vs Old Post (paired, 6-mo RT). Young excluded.
    """
    df = samples_df.copy()
    if cohort_id == "GSE97084":
        sub = df[df["age_group"].str.lower() == "old"].copy()
        sub["contrast_val"] = np.where(
            sub["timepoint"].str.lower() == "posttraining", 1,
            np.where(sub["timepoint"].str.lower() == "pretraining", 0, -1)
        )
    elif cohort_id == "GSE151066":
        sub = df.copy()
        sub["contrast_val"] = np.where(
            sub["timepoint"].str.lower() == "post", 1,
            np.where(sub["timepoint"].str.lower() == "pre", 0, -1)
        )
    elif cohort_id == "GSE28422":
        # Old + training state Trained (basal) vs Untrained (basal)
        sub = df[(df["age_group"].str.lower() == "old") &
                 (df["timepoint"].str.lower() == "basal")].copy()
        sub["contrast_val"] = np.where(
            sub["exercise_type"].str.lower() == "trained", 1,
            np.where(sub["exercise_type"].str.lower() == "untrained", 0, -1)
        )
    elif cohort_id == "GSE8479":
        sub = df[df["age_group"].str.lower() == "old"].copy()
        sub["contrast_val"] = np.where(
            sub["timepoint"].str.lower() == "post", 1,
            np.where(sub["timepoint"].str.lower() == "pre", 0, -1)
        )
    else:
        return None
    sub = sub[sub["contrast_val"].isin([0, 1])].copy()
    log.info("  [%s] design: %d samples (0=%d, 1=%d); paired subjects=%d",
             cohort_id, len(sub),
             int((sub["contrast_val"] == 0).sum()),
             int((sub["contrast_val"] == 1).sum()),
             sub["subject_id"].replace("", np.nan).dropna().nunique())
    return sub[["sample_id", "subject_id", "contrast_val"]].reset_index(drop=True)


def fit_limma_ebayes(expr: pd.DataFrame, design: pd.DataFrame,
                     log: logging.Logger) -> pd.DataFrame:
    """Per-gene OLS with subject fixed effects + Smyth 2004 empirical-Bayes
    variance shrinkage (moderated t).

    Inputs:
      expr: rows = feature_id, cols = sample_id (values on log scale)
      design: sample_id, subject_id, contrast_val

    Returns DataFrame[feature_id, logFC, t, moderated_t, p_value].

    WHY moderated t: limma eBayes (Smyth 2004) shrinks the per-gene variance
    toward a prior estimated across all genes; this is the standard approach
    for microarray DE with small N. We implement the closed-form from Smyth
    2004 eq. 4 (posterior variance = (d0*s0^2 + d*s^2) / (d0+d)) using
    Newton-Raphson to estimate d0, s0 following Phipson 2016.

    Paired via subject FE: for each gene, fit y_ij = mu + beta*x_ij + alpha_i
    where alpha_i is subject effect. Degrees of freedom = n_samples -
    n_subjects - 1.
    """
    # Align columns
    common = [s for s in design["sample_id"] if s in expr.columns]
    expr_al = expr[common].values  # (n_genes, n_samples)
    des = design.set_index("sample_id").loc[common]
    x = des["contrast_val"].values.astype(float)
    subjects = des["subject_id"].fillna("").values
    unique_subj = [s for s in pd.unique(subjects) if s != ""]

    # Build design matrix
    if len(unique_subj) >= 3 and (subjects != "").all():
        # Paired: intercept + contrast + subject dummies
        subj_idx = {s: i for i, s in enumerate(unique_subj)}
        S = np.zeros((len(common), len(unique_subj) - 1))
        for j, s in enumerate(subjects):
            k = subj_idx.get(s, -1)
            if 0 < k:
                S[j, k - 1] = 1.0
        X = np.column_stack([np.ones(len(common)), x, S])
        log.info("    paired design matrix: %d samples x %d cols (intercept, contrast, %d subj FE)",
                 X.shape[0], X.shape[1], X.shape[1] - 2)
    else:
        X = np.column_stack([np.ones(len(common)), x])
        log.info("    unpaired design matrix: %d samples x 2 cols", X.shape[0])

    n, p = X.shape
    dof_resid = n - p
    if dof_resid < 1:
        log.warning("    DOF=%d <1; abort this cohort", dof_resid)
        return pd.DataFrame(columns=["feature_id", "logFC", "t", "moderated_t", "p_value"])

    # Precompute (X'X)^-1 and hat matrix diagonal
    XtX_inv = np.linalg.pinv(X.T @ X)
    XtX_inv_contrast = XtX_inv[1, 1]  # variance scale for beta_contrast

    # Per-gene OLS
    Y = expr_al  # (G, n)
    # beta = (X'X)^-1 X' y
    beta_all = Y @ X @ XtX_inv  # (G, p)
    resid = Y - beta_all @ X.T  # (G, n)
    rss = (resid ** 2).sum(axis=1)  # (G,)
    sigma2 = rss / dof_resid  # (G,)
    beta_contrast = beta_all[:, 1]
    se = np.sqrt(sigma2 * XtX_inv_contrast)
    t_raw = beta_contrast / np.where(se > 0, se, np.nan)

    # Empirical-Bayes variance moderation (Smyth 2004)
    # d = dof_resid (same for all genes); s^2 = sigma2
    # Estimate prior (d0, s0^2) by fitting scaled F distribution to s^2.
    # Closed-form robust estimator from Smyth 2004 / Phipson 2016.
    log_s2 = np.log(sigma2[sigma2 > 0])
    d0, s0_sq = _fit_eb_prior(log_s2, dof_resid)
    moderated_var = (d0 * s0_sq + dof_resid * sigma2) / (d0 + dof_resid)
    moderated_se = np.sqrt(moderated_var * XtX_inv_contrast)
    t_mod = beta_contrast / np.where(moderated_se > 0, moderated_se, np.nan)
    dof_mod = dof_resid + d0
    p_mod = 2 * stats.t.sf(np.abs(t_mod), df=dof_mod)

    de = pd.DataFrame({
        "feature_id": expr.index.astype(str),
        "logFC": beta_contrast,
        "t": t_raw,
        "moderated_t": t_mod,
        "p_value": p_mod,
    })
    log.info("    eBayes prior: d0=%.2f, s0^2=%.4f; dof_mod=%.2f; DE rows=%d",
             d0, s0_sq, dof_mod, len(de))
    return de


def _fit_eb_prior(log_s2: np.ndarray, d: float) -> Tuple[float, float]:
    """Estimate (d0, s0^2) from observed log residual variances.

    Smyth 2004 eq. 3 + eq. 4 closed-form Newton-Raphson solution from Phipson
    2016. For numerical stability, falls back to naive prior if fit fails.
    """
    if len(log_s2) < 10:
        return float(d), float(np.exp(np.median(log_s2))) if len(log_s2) else 1.0
    # Observed mean & variance of log(s^2)
    # Under the model s^2 | sigma^2 ~ sigma^2 * chi2(d)/d
    # and sigma^2 ~ s0^2 * d0 / chi2(d0)
    # -> z = log(s^2) has theoretical moments:
    #   E[z] = log(s0^2) + digamma(d/2) - log(d/2) + log(d0/2) - digamma(d0/2)
    #   Var[z] = trigamma(d/2) + trigamma(d0/2)
    obs_var = np.var(log_s2, ddof=1)
    target = obs_var - stats.chi2.stats(d, moments="v")  # noqa: unused? Use trigamma via scipy
    # scipy doesn't expose polygamma easily; use digamma/trigamma from scipy.special
    from scipy.special import digamma, polygamma
    trigamma_d = polygamma(1, d / 2.0)
    # Solve trigamma(d0/2) = obs_var - trigamma(d/2)  (moment match on variance)
    rhs = obs_var - trigamma_d
    if rhs <= 0:
        # Very stable residuals across genes (unlikely in real data); default to small prior
        return 1.0, float(np.exp(np.median(log_s2)))
    # Newton-Raphson on f(x) = trigamma(x) - rhs, x = d0/2
    x = 1.0 / rhs  # initialization per Smyth
    for _ in range(50):
        f = polygamma(1, x) - rhs
        fp = polygamma(2, x)
        if fp == 0:
            break
        dx = f / fp
        x_new = max(x - dx, 1e-6)
        if abs(x_new - x) < 1e-8:
            x = x_new
            break
        x = x_new
    d0 = 2.0 * x
    # Solve for s0^2 from mean equation
    mean_z = np.mean(log_s2)
    log_s0_sq = mean_z - digamma(d / 2.0) + np.log(d / 2.0) - np.log(d0 / 2.0) + digamma(d0 / 2.0)
    s0_sq = float(np.exp(log_s0_sq))
    return float(d0), s0_sq


# -----------------------------------------------------------------------------
# Probe-to-gene collapse
# -----------------------------------------------------------------------------
def collapse_to_gene(de: pd.DataFrame, probe_map: pd.DataFrame,
                     log: logging.Logger) -> pd.DataFrame:
    """Collapse probe-level DE to gene-level by selecting the probe with max
    |moderated_t| per gene.

    WHY max-|t|: standard practice (Miller 2011 WGCNA manual); preserves the
    most statistically-grounded signal per gene without invoking mean-of-probes
    (which dampens real signal when probes have different specificity).
    """
    merged = de.merge(probe_map.rename(columns={"probe_id": "feature_id"}),
                      on="feature_id", how="left")
    merged = merged.dropna(subset=["gene_symbol"])
    merged = merged[merged["gene_symbol"] != ""]
    merged["abs_t"] = merged["moderated_t"].abs()
    merged = merged.sort_values("abs_t", ascending=False).drop_duplicates(subset=["gene_symbol"])
    gene_de = merged[["gene_symbol", "logFC", "moderated_t", "p_value"]].reset_index(drop=True)
    log.info("    collapsed %d probes -> %d unique genes", len(de), len(gene_de))
    return gene_de


# -----------------------------------------------------------------------------
# Gene-coverage pre-check (brief §T2 line 178, Critic 2 B2)
# -----------------------------------------------------------------------------
def gene_coverage(signature: List[str], cohort_genes: set) -> Tuple[int, List[str]]:
    """Return (# signature genes present, list of missing genes)."""
    present = [g for g in signature if g in cohort_genes]
    missing = [g for g in signature if g not in cohort_genes]
    return len(present), missing


# -----------------------------------------------------------------------------
# GSEA via gseapy.prerank (identical to batch_064)
# -----------------------------------------------------------------------------
def run_gsea_for_cohort(gene_de: pd.DataFrame,
                        signatures: Dict[str, List[str]],
                        cohort_id: str,
                        log: logging.Logger) -> pd.DataFrame:
    """Run gseapy.prerank + p_sig_rand null on ranked gene list."""
    import gseapy
    # Dedupe + dropna
    df = gene_de.dropna(subset=["gene_symbol", "moderated_t"]).drop_duplicates(
        subset=["gene_symbol"])
    ranked = pd.Series(df["moderated_t"].values, index=df["gene_symbol"].values)
    rnk = pd.DataFrame({"gene": ranked.index, "rank": ranked.values})

    gseapy_res = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pr = gseapy.prerank(
                rnk=rnk,
                gene_sets=signatures,
                permutation_num=N_PERMUTATIONS,
                outdir=None,
                seed=SEED,
                min_size=5,
                max_size=max(500, max(len(v) for v in signatures.values()) + 1),
                verbose=False,
                no_plot=True,
            )
        gseapy_res = pr.res2d.copy()
    except Exception as e:  # noqa: BLE001
        log.error("  [%s] gseapy.prerank FAILED: %s", cohort_id, e)
        return pd.DataFrame()

    # p_sig_rand null (identical to batch_064/c_run_c_motrpac.py)
    rows: List[dict] = []
    universe = ranked.index.tolist()
    scores = ranked.values.astype(float)
    order = np.argsort(-scores)
    scores_ordered = scores[order]
    abs_scores_ordered = np.abs(scores_ordered)
    gene_to_idx = {g: i for i, g in enumerate(ranked.index)}
    n_total = len(ranked)
    rng = np.random.default_rng(SEED)

    for name, genes in signatures.items():
        row = {"cohort_id": cohort_id, "signature": name}
        # gseapy NES + p_gene_perm
        gp = gseapy_res[gseapy_res["Term"] == name]
        if len(gp) > 0:
            g = gp.iloc[0]
            row["NES_gseapy"] = float(g["NES"])
            row["ES_gseapy"] = float(g["ES"])
            row["p_gene_perm"] = float(g["NOM p-val"])
            row["fdr_q"] = float(g["FDR q-val"])
            row["leading_edge_n"] = int(len(str(g["Lead_genes"]).split(";"))) if g["Lead_genes"] else 0
        else:
            row.update({"NES_gseapy": np.nan, "ES_gseapy": np.nan,
                        "p_gene_perm": np.nan, "fdr_q": np.nan, "leading_edge_n": 0})

        # p_sig_rand: size-matched random-signature null
        sig_idx = np.array([gene_to_idx[g] for g in genes if g in gene_to_idx], dtype=int)
        sig_size = len(sig_idx)
        if sig_size < 5:
            row["p_sig_rand"] = np.nan
            row["observed_es"] = np.nan
            row["observed_nes_local"] = np.nan
            rows.append(row)
            continue
        sig_mask = np.zeros(n_total, dtype=bool)
        sig_mask[sig_idx] = True
        hits_obs = sig_mask[order]
        observed_es = _walking_ks_es(scores_ordered, abs_scores_ordered, hits_obs, n_total)
        null_es = np.empty(N_SIG_RAND_PERM, dtype=float)
        for i in range(N_SIG_RAND_PERM):
            pick = rng.choice(n_total, size=sig_size, replace=False)
            m = np.zeros(n_total, dtype=bool)
            m[pick] = True
            null_es[i] = _walking_ks_es(scores_ordered, abs_scores_ordered,
                                          m[order], n_total)
        hits = int((np.abs(null_es) >= abs(observed_es)).sum())
        p_emp = (hits + 1) / (N_SIG_RAND_PERM + 1)
        pos = null_es[null_es > 0]
        neg = null_es[null_es < 0]
        pos_mean = pos.mean() if len(pos) else 1.0
        neg_mean = abs(neg.mean()) if len(neg) else 1.0
        if observed_es > 0:
            obs_nes_local = observed_es / pos_mean if pos_mean else np.nan
        elif observed_es < 0:
            obs_nes_local = observed_es / neg_mean if neg_mean else np.nan
        else:
            obs_nes_local = 0.0
        row["p_sig_rand"] = float(p_emp)
        row["observed_es"] = float(observed_es)
        row["observed_nes_local"] = float(obs_nes_local)
        row["signature_size_effective"] = sig_size
        rows.append(row)

    return pd.DataFrame(rows)


def _walking_ks_es(scores_ordered: np.ndarray, abs_scores_ordered: np.ndarray,
                   hits_ordered: np.ndarray, n_total: int) -> float:
    """Weighted-KS enrichment score (Subramanian 2005, p=1).

    WHY copy from batch_064: null-scale-mismatch bug fix (review_run_c.md item 1).
    Observed and null ES must be computed by the same routine to be comparable.
    """
    K = int(hits_ordered.sum())
    if K == 0 or K >= n_total:
        return 0.0
    hit_weights = abs_scores_ordered * hits_ordered
    hw_sum = hit_weights.sum()
    if hw_sum <= 0:
        return 0.0
    hit_weights = hit_weights / hw_sum
    miss_weights = (~hits_ordered).astype(float) / (n_total - K)
    running = np.cumsum(hit_weights - miss_weights)
    max_pos = float(running.max())
    max_neg = float(running.min())
    return max_pos if abs(max_pos) >= abs(max_neg) else max_neg


# -----------------------------------------------------------------------------
# Fisher + Stouffer combination
# -----------------------------------------------------------------------------
def fisher_combine(pvals: List[float]) -> Tuple[float, float]:
    """Return (fisher_combined_p, stouffer_z_combined_p).

    WHY both: Fisher 1925 is the default (brief line 183) but Stouffer is more
    robust when p-values span many orders of magnitude. Pre-registered as
    sensitivity check per brief §T2 line 184.
    """
    pvals = [p for p in pvals if p is not None and not np.isnan(p) and 0 < p <= 1]
    if len(pvals) < 2:
        return (np.nan, np.nan)
    _, p_fisher = stats.combine_pvalues(pvals, method="fisher")
    _, p_stouffer = stats.combine_pvalues(pvals, method="stouffer")
    return float(p_fisher), float(p_stouffer)


def signed_pvalue(p: float, nes: float, expected_sign: float) -> float:
    """Return a directional p-value: uncorrected p if sign matches expected,
    else (1 - p/2)*2 treated as nominal 2-sided (effectively deflates wrong-sign
    contributions).

    WHY: for Fisher combination we want to PENALIZE wrong-sign results without
    zero-inflating the meta-p. Standard approach: convert to one-sided p
    aligned with expected direction: p_dir = p/2 if sign matches else 1-p/2.
    """
    if np.isnan(p) or np.isnan(nes):
        return np.nan
    if nes * expected_sign > 0:
        return p / 2.0
    else:
        return 1.0 - p / 2.0


# -----------------------------------------------------------------------------
# MoTrPAC import (reuse iter 064 output)
# -----------------------------------------------------------------------------
def load_motrpac_gsea(log: logging.Logger) -> pd.DataFrame:
    """Load MoTrPAC 24-hr GSEA from batch_064/c_gsea_results.csv.

    Returns rows with columns (cohort_id, signature, NES_gseapy, ES_gseapy,
    p_gene_perm, p_sig_rand, observed_es, observed_nes_local, fdr_q,
    leading_edge_n, signature_size_effective) matching other cohorts.
    """
    df = pd.read_csv(MOTRPAC_GSEA)
    sub = df[df["timepoint"] == MOTRPAC_TIMEPOINT_KEY].copy()
    sub["cohort_id"] = "MoTrPAC_24hr"
    sub["signature"] = sub["compartment"] + "_" + sub["direction"]
    rename = {
        "NES": "NES_gseapy",
        "ES": "ES_gseapy",
        "p_sig_rand_perm": "p_sig_rand",
        "observed_es_local": "observed_es",
        "observed_nes_local": "observed_nes_local",
        "signature_size": "signature_size_effective",
    }
    sub = sub.rename(columns=rename)
    cols = ["cohort_id", "signature", "NES_gseapy", "ES_gseapy", "p_gene_perm",
            "p_sig_rand", "observed_es", "observed_nes_local", "fdr_q",
            "leading_edge_n", "signature_size_effective"]
    for c in cols:
        if c not in sub.columns:
            sub[c] = np.nan
    log.info("  MoTrPAC GSEA reloaded: %d rows (timepoint=%s)",
             len(sub), MOTRPAC_TIMEPOINT_KEY)
    return sub[cols]


# -----------------------------------------------------------------------------
# Decision logic
# -----------------------------------------------------------------------------
def apply_decision_rule(fisher_df: pd.DataFrame, coverage_df: pd.DataFrame,
                        log: logging.Logger) -> Dict[str, object]:
    """Apply brief §T2 DECISION RULE (revised per Critic 2 B2-B3).

    PRIMARY: SIGN-CONCORDANCE (>=3 of 5 cohorts) AND Fisher p < 2.08e-3 in
    ACUTE or CHRONIC stratum.
    """
    verdict_per_cell: List[dict] = []
    for _, row in fisher_df.iterrows():
        cell = f"{row['compartment']}_{row['direction']}_{row['stratum']}"
        concordant_n = int(row["sign_concordant_n"])
        total_n = int(row["n_cohorts"])
        fisher_p = float(row["fisher_p"])
        # Coverage check for this compartment/direction
        sig_name = f"{row['compartment']}_{row['direction']}"
        low_cov = coverage_df[
            (coverage_df["signature"] == sig_name) &
            (coverage_df["coverage"] < COVERAGE_RESTRICT_THRESHOLD)
        ]["cohort_id"].tolist()
        block_cov = coverage_df[
            (coverage_df["signature"] == sig_name) &
            (coverage_df["coverage"] < COVERAGE_BLOCK_THRESHOLD)
        ]["cohort_id"].tolist()
        verdict = "INCONCLUSIVE"
        if len(block_cov) >= 2:
            verdict = "BLOCKED_COVERAGE"
        elif concordant_n >= 3 and fisher_p < BONFERRONI_ALPHA:
            verdict = "SUGGESTED"
        elif row.get("sign_discordant_with_motrpac_n", 0) >= 2:
            verdict = "REFUTED"
        elif concordant_n >= 3 and fisher_p >= BONFERRONI_ALPHA:
            verdict = "INCONCLUSIVE"
        elif len(low_cov) >= 2:
            verdict = "INCONCLUSIVE_LOW_COVERAGE"
        verdict_per_cell.append({
            "cell": cell,
            "verdict": verdict,
            "concordant_n": concordant_n,
            "total_n": total_n,
            "fisher_p": fisher_p,
            "low_coverage_cohorts": low_cov,
            "blocked_coverage_cohorts": block_cov,
        })

    # Aggregate verdict: SUGGESTED if any UP cell (primary direction for F064_04
    # was Vascular_UP, NES<0) reaches SUGGESTED in either stratum.
    any_suggested = any(v["verdict"] == "SUGGESTED" for v in verdict_per_cell)
    any_refuted = any(v["verdict"] == "REFUTED" for v in verdict_per_cell)
    both_strata_sig = False
    # Check specifically Vascular_UP
    v_acute = [v for v in verdict_per_cell
               if v["cell"] == "Vascular_UP_ACUTE" and v["verdict"] == "SUGGESTED"]
    v_chronic = [v for v in verdict_per_cell
                 if v["cell"] == "Vascular_UP_CHRONIC" and v["verdict"] == "SUGGESTED"]
    if v_acute and v_chronic:
        both_strata_sig = True

    overall = "INCONCLUSIVE"
    if both_strata_sig:
        overall = "SUGGESTED-STRONG"
    elif any_suggested:
        overall = "SUGGESTED"
    elif any_refuted:
        overall = "REFUTED"

    return {
        "verdict_per_cell": verdict_per_cell,
        "overall_verdict": overall,
        "any_suggested": any_suggested,
        "any_refuted": any_refuted,
        "both_strata_sig": both_strata_sig,
    }


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def main() -> int:
    log = setup_logging()
    log.info("=== batch_065 run_t2_plasticity.py START ===")
    import scipy
    log.info("python %s  numpy %s  pandas %s  scipy %s",
             sys.version.split()[0], np.__version__, pd.__version__, scipy.__version__)
    try:
        import gseapy
        import GEOparse
        log.info("gseapy %s  GEOparse %s", gseapy.__version__, GEOparse.__version__)
    except ImportError as e:
        log.error("missing dep: %s", e)
        return 2

    # ------------------------------------------------------------
    # 0. Load HLMA 6 signatures (100 genes each) — same as batch_064
    # ------------------------------------------------------------
    log.info("[0] Loading HLMA 6 signatures")
    hlma = pd.read_csv(HLMA_SIGNATURES)
    hlma_by_cell: Dict[str, List[str]] = {}
    for (comp, direction), sub in hlma.groupby(["compartment", "direction"]):
        name = f"{comp}_{direction}"
        hlma_by_cell[name] = sub["gene_symbol"].dropna().astype(str).tolist()
        log.info("  %s: %d genes", name, len(hlma_by_cell[name]))

    # ------------------------------------------------------------
    # 1. Parse SOFT metadata for all 4 GEO cohorts
    # ------------------------------------------------------------
    log.info("[1] Parsing SOFT metadata")
    all_samples: Dict[str, pd.DataFrame] = {}
    all_meta: Dict[str, dict] = {}
    for cohort, soft_path in SOFT_FILES.items():
        log.info(" cohort=%s  file=%s", cohort, soft_path.name)
        samples, meta = parse_soft_metadata(soft_path, log)
        samples = extract_design_columns(samples, cohort, log)
        all_samples[cohort] = samples
        all_meta[cohort] = meta

    # ------------------------------------------------------------
    # 2. Independence audit (brief §T2 line 176)
    # ------------------------------------------------------------
    log.info("[2] Independence audit")
    audit = run_independence_audit(all_samples, all_meta, log)
    audit.to_csv(OUT_AUDIT, index=False)
    log.info("  wrote %s", OUT_AUDIT)

    # ------------------------------------------------------------
    # 3. Per-cohort DE
    # ------------------------------------------------------------
    log.info("[3] Per-cohort DE")
    de_per_cohort: Dict[str, pd.DataFrame] = {}
    cohort_gene_universe: Dict[str, set] = {}
    DE_DIR.mkdir(parents=True, exist_ok=True)

    for cohort in SOFT_FILES.keys():
        log.info(" --- cohort=%s ---", cohort)
        samples = all_samples[cohort]
        design = build_design(samples, cohort, log)
        if design is None or len(design) < 4:
            log.warning("  [%s] insufficient design; SKIP", cohort)
            continue

        try:
            if cohort in ("GSE97084", "GSE151066"):
                counts, probe_map = load_rnaseq_matrix(cohort, samples, log)
                # log2(CPM+1) normalization (standard for voom-style OLS)
                lib_size = counts.sum(axis=0).replace(0, np.nan)
                cpm = counts.div(lib_size, axis=1) * 1e6
                expr = np.log2(cpm.fillna(0) + 1.0)
                # Subset columns to those in design
                expr = expr[[s for s in design["sample_id"] if s in expr.columns]]
            else:
                expr, probe_map = load_microarray_matrix(cohort, SOFT_FILES[cohort], log)
                expr = expr[[s for s in design["sample_id"] if s in expr.columns]]
        except Exception as e:
            log.exception("  [%s] expression load FAILED: %s", cohort, e)
            continue

        if expr.shape[1] < 4:
            log.warning("  [%s] only %d samples matched; SKIP", cohort, expr.shape[1])
            continue

        de = fit_limma_ebayes(expr, design, log)
        gene_de = collapse_to_gene(de, probe_map, log)
        gene_de.to_csv(DE_DIR / f"{cohort}_de.csv", index=False)
        log.info("  [%s] wrote DE: %d genes", cohort, len(gene_de))
        de_per_cohort[cohort] = gene_de
        cohort_gene_universe[cohort] = set(gene_de["gene_symbol"].astype(str).tolist())

    # ------------------------------------------------------------
    # 4. Gene-coverage pre-check (brief §T2 line 178, Critic 2 B2)
    # ------------------------------------------------------------
    log.info("[4] Gene-coverage pre-check")
    coverage_rows: List[dict] = []
    # MoTrPAC coverage: we know iter 064 used ENSG-versioned ranks from MoTrPAC
    # feature universe; coverage per signature was effectively 'signature_size' in
    # c_gsea_results.csv. Re-derive from that file.
    motrpac_df = pd.read_csv(MOTRPAC_GSEA)
    motrpac_24h = motrpac_df[motrpac_df["timepoint"] == MOTRPAC_TIMEPOINT_KEY]
    for _, row in motrpac_24h.iterrows():
        sig_name = f"{row['compartment']}_{row['direction']}"
        n_total = len(hlma_by_cell[sig_name])
        n_present = int(row["signature_size"])  # size after ENSG universe restriction
        coverage_rows.append({
            "cohort_id": "MoTrPAC_24hr",
            "signature": sig_name,
            "n_signature_genes": n_total,
            "n_probed": n_present,
            "coverage": n_present / n_total,
            "restricted_gsea_flag": int(n_present / n_total < COVERAGE_RESTRICT_THRESHOLD),
            "blocked_flag": int(n_present / n_total < COVERAGE_BLOCK_THRESHOLD),
        })

    for cohort, universe in cohort_gene_universe.items():
        for sig_name, genes in hlma_by_cell.items():
            present, _ = gene_coverage(genes, universe)
            cov = present / len(genes)
            coverage_rows.append({
                "cohort_id": cohort,
                "signature": sig_name,
                "n_signature_genes": len(genes),
                "n_probed": present,
                "coverage": cov,
                "restricted_gsea_flag": int(cov < COVERAGE_RESTRICT_THRESHOLD),
                "blocked_flag": int(cov < COVERAGE_BLOCK_THRESHOLD),
            })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(OUT_COVERAGE, index=False)
    log.info("  wrote %s (%d rows; %d restricted; %d blocked)",
             OUT_COVERAGE, len(coverage_df),
             int(coverage_df["restricted_gsea_flag"].sum()),
             int(coverage_df["blocked_flag"].sum()))

    # ------------------------------------------------------------
    # 5. Per-cohort GSEA against HLMA 6 signatures
    # ------------------------------------------------------------
    log.info("[5] GSEA per cohort")
    all_gsea: List[pd.DataFrame] = []
    # MoTrPAC (reuse batch_064 results)
    all_gsea.append(load_motrpac_gsea(log))

    for cohort, gene_de in de_per_cohort.items():
        log.info(" --- GSEA %s ---", cohort)
        # Restrict signatures to genes actually probed in this cohort
        universe = cohort_gene_universe[cohort]
        restricted_sigs: Dict[str, List[str]] = {}
        for name, genes in hlma_by_cell.items():
            # If coverage<block_threshold, skip (record will be NaN in GSEA output)
            cov = sum(1 for g in genes if g in universe) / len(genes)
            if cov < COVERAGE_BLOCK_THRESHOLD:
                log.warning("  [%s] %s coverage=%.2f <%.2f BLOCKED",
                            cohort, name, cov, COVERAGE_BLOCK_THRESHOLD)
                continue
            if cov < COVERAGE_RESTRICT_THRESHOLD:
                # Restricted-signature GSEA: intersect with probed genes
                restricted_sigs[name] = [g for g in genes if g in universe]
                log.info("  [%s] %s coverage=%.2f <%.2f -> RESTRICTED (%d genes)",
                         cohort, name, cov, COVERAGE_RESTRICT_THRESHOLD,
                         len(restricted_sigs[name]))
            else:
                restricted_sigs[name] = genes
        if not restricted_sigs:
            log.warning("  [%s] all signatures blocked; SKIP GSEA", cohort)
            continue
        gsea = run_gsea_for_cohort(gene_de, restricted_sigs, cohort, log)
        all_gsea.append(gsea)

    gsea_df = pd.concat(all_gsea, ignore_index=True) if all_gsea else pd.DataFrame()
    # Parse compartment/direction from signature name
    if not gsea_df.empty:
        parts = gsea_df["signature"].str.rsplit("_", n=1, expand=True)
        gsea_df["compartment"] = parts[0]
        gsea_df["direction"] = parts[1]
    gsea_df.to_csv(OUT_GSEA, index=False)
    log.info("  wrote %s (%d rows)", OUT_GSEA, len(gsea_df))

    # ------------------------------------------------------------
    # 6. Fisher + Stouffer combination (ACUTE, CHRONIC, POOLED)
    # ------------------------------------------------------------
    log.info("[6] Fisher + Stouffer combination")
    fisher_rows: List[dict] = []
    # Expected sign: aging-UP goes DOWN with exercise (NES<0). aging-DOWN goes UP (NES>0).
    for comp in COMPARTMENTS:
        for direction in DIRECTIONS:
            expected_sign = -1.0 if direction == "UP" else 1.0
            for stratum_name in ["ACUTE", "CHRONIC", "POOLED"]:
                cohorts = STRATA.get(stratum_name, ALL_COHORTS)
                sub = gsea_df[
                    (gsea_df["compartment"] == comp) &
                    (gsea_df["direction"] == direction) &
                    (gsea_df["cohort_id"].isin(cohorts))
                ]
                if len(sub) == 0:
                    continue
                # Sign concordance
                nes_vals = sub["NES_gseapy"].values
                concordant_n = int(np.sum(np.sign(nes_vals) == expected_sign))
                discordant_n = int(np.sum(np.sign(nes_vals) == -expected_sign))
                # MoTrPAC-discordance count (vs non-MoTrPAC cohorts)
                motrpac_sub = sub[sub["cohort_id"] == "MoTrPAC_24hr"]
                motrpac_sign = (np.sign(motrpac_sub["NES_gseapy"].iloc[0])
                                if len(motrpac_sub) else 0)
                others = sub[sub["cohort_id"] != "MoTrPAC_24hr"]
                discordant_vs_motrpac = int(
                    np.sum(np.sign(others["NES_gseapy"]) == -motrpac_sign)
                ) if motrpac_sign != 0 else 0

                # Directional p-values: one-sided aligned with expected sign
                p_dir_gene: List[float] = []
                p_dir_sig: List[float] = []
                for _, r in sub.iterrows():
                    p_dir_gene.append(signed_pvalue(float(r["p_gene_perm"]),
                                                     float(r["NES_gseapy"]), expected_sign))
                    p_dir_sig.append(signed_pvalue(float(r["p_sig_rand"]),
                                                    float(r["observed_nes_local"]),
                                                    expected_sign))
                fisher_gene, stouffer_gene = fisher_combine(p_dir_gene)
                fisher_sig, stouffer_sig = fisher_combine(p_dir_sig)
                fisher_rows.append({
                    "stratum": stratum_name,
                    "compartment": comp,
                    "direction": direction,
                    "expected_sign": expected_sign,
                    "n_cohorts": len(sub),
                    "sign_concordant_n": concordant_n,
                    "sign_discordant_n": discordant_n,
                    "sign_discordant_with_motrpac_n": discordant_vs_motrpac,
                    "cohorts_included": ";".join(sub["cohort_id"].tolist()),
                    "NES_values": ";".join(f"{x:.3f}" for x in nes_vals),
                    "fisher_p": fisher_gene,
                    "stouffer_p": stouffer_gene,
                    "fisher_p_sig_rand": fisher_sig,
                    "stouffer_p_sig_rand": stouffer_sig,
                    "bonferroni_alpha": BONFERRONI_ALPHA,
                    "pass_bonferroni": int((fisher_gene or 1.0) < BONFERRONI_ALPHA),
                })
    fisher_df = pd.DataFrame(fisher_rows)
    fisher_df.to_csv(OUT_FISHER, index=False)
    log.info("  wrote %s (%d rows)", OUT_FISHER, len(fisher_df))

    # ------------------------------------------------------------
    # 7. Apply decision rule + write summary
    # ------------------------------------------------------------
    log.info("[7] Decision rule")
    decision = apply_decision_rule(fisher_df, coverage_df, log)
    summary = {
        "iteration": 65,
        "test": "T2",
        "hypothesis": "H065_02",
        "overall_verdict": decision["overall_verdict"],
        "any_suggested": decision["any_suggested"],
        "any_refuted": decision["any_refuted"],
        "both_strata_sig": decision["both_strata_sig"],
        "verdict_per_cell": decision["verdict_per_cell"],
        "cohorts_analyzed": list(de_per_cohort.keys()) + ["MoTrPAC_24hr"],
        "cohorts_blocked_or_failed": [c for c in SOFT_FILES if c not in de_per_cohort],
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "nes_magnitude_disclosed_threshold": NES_MAGNITUDE_DISCLOSED,
        "nes_threshold_disclosure": ("|NES|>=1.5 was ex-post tuned on F064_04 "
                                     "(NES=-1.644); primary criterion is "
                                     "SIGN-CONCORDANCE per Critic 2 B3."),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "seed": SEED,
            "n_permutations": N_PERMUTATIONS,
            "n_sig_rand_perm": N_SIG_RAND_PERM,
        },
    }
    with OUT_SUMMARY.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("  wrote %s", OUT_SUMMARY)

    # Pretty-print overall verdict
    log.info("=" * 60)
    log.info("OVERALL VERDICT: %s", decision["overall_verdict"])
    log.info("  any_suggested=%s  any_refuted=%s  both_strata_sig=%s",
             decision["any_suggested"], decision["any_refuted"],
             decision["both_strata_sig"])
    for v in decision["verdict_per_cell"]:
        log.info("  %-40s %-25s concordant=%d/%d fisher_p=%.3e",
                 v["cell"], v["verdict"], v["concordant_n"], v["total_n"],
                 v["fisher_p"])
    log.info("=== run_t2_plasticity.py END ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
