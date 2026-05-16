#!/usr/bin/env python3
"""T4 — Cell-type-specific sc-eQTL sweep for 9 canonical TFs × 6 muscle-aging traits.

Pre-registered in experiments/batch_065/brief.md §T4.

Datasets (feasibility-gated; see t4_data_availability.md):
  1. Natri 2024 (GSE227136) — FEASIBLE (public GEO FTP, mashR sumstats)
  2. Yazar 2022 OneK1K — FEASIBLE (public S3, per-cell-type cis-eQTL TSV)
  3. Soskic 2022 (EGAS00001005839) — BLOCKED (EGA DAC-gated) — excluded from pipeline

TFs (9): JUNB, FOS, EGR1, EGR2, ATF3, CEBPB, KLF10, IRF1, CDKN1A.
Traits (6): sarcopenia, grip_strength, frailty, lean_body_mass, appendicular_lean_mass,
            muscular_dystrophy (aligned with experiments/batch_064/a_opentargets_grid.csv).

Decision rule (per brief §T4, per-dataset BH):
  - Natri 2024 N=114 null → INCONCLUSIVE (underpowered for β<0.3).
  - OneK1K N=982 null across covered TFs → REFUTED for OneK1K-covered TFs.
  - ≥1 TF × cell-type × trait with BH q<0.05 AND H4>0.8 → SUGGESTED.

Note on coloc: Per brief "If full coloc not feasible, report min_p from each and
  lookup-only." No GWAS sumstats are cached locally; coloc.abf would require
  fetching GWAS sumstats (outside 30-90 min budget). Pipeline runs LOOKUP-ONLY:
  reports min p (or lfsr for Natri mashR) per (TF × cell-type) and per-dataset BH.
  H4 column = "[NOT_RUN]" to avoid fabrication (Rule 0).

WHY this design:
  - We implement the exact spec in brief.md §T4 (post-3-critic review).
  - BH correction is applied per-dataset, NOT pooled across 648 tests, per
    Critic 2 B4 resolution. Pooling across datasets with different N/cell-types/
    coverage would conflate adequately-powered null (OneK1K) with underpowered
    null (Natri) and misclassify both.
  - We use mygene for canonical coordinates because ENSG → HGNC symbol alignment
    between HLMA (symbol-space) and Natri (ENSG-space) would otherwise silently
    drop TF rows.
  - Cis-window ±1 Mb TSS per GTEx convention (Consortium 2020, Science) and
    Natri+OneK1K both report cis-only.

Hyperparameter sources:
  - TF list (9 canonical): experiments/batch_050/canonical_tf_sasp_table_final.csv
    + iter-064 F064_01 test set.
  - Trait list (6): experiments/batch_064/a_opentargets_grid.csv (EFO/MONDO IDs
    verified in iter 064).
  - Cis-window ±1 Mb: GTEx v8 convention (Consortium 2020).
  - BH α = 0.05: Benjamini-Hochberg 1995 J R Stat Soc.
  - Per-dataset (not pooled) BH: brief §T4 DECISION RULE (Critic 2 B4).
  - mashR lfsr threshold: 0.05 analog-to-FDR per Urbut 2019 Nat Genet.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_CACHE = ROOT / "data_cache"
OUT_PER_DATASET = ROOT / "t4_sceqtl_per_dataset"
LOG_DIR = ROOT / "logs"
for d in (DATA_CACHE, OUT_PER_DATASET, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "t4_stdout.log"
OUT_COLOC_CSV = ROOT / "t4_coloc.csv"
OUT_SUMMARY_JSON = ROOT / "t4_summary.json"

# Canonical TFs (9) — symbols. ENSG filled via mygene.
TFS: list[str] = [
    "JUNB", "FOS", "EGR1", "EGR2", "ATF3",
    "CEBPB", "KLF10", "IRF1", "CDKN1A",
]

# Muscle-aging traits (6) — per batch_064/a_opentargets_grid.csv.
TRAITS: list[str] = [
    "sarcopenia",
    "grip_strength",
    "frailty",
    "lean_body_mass",
    "appendicular_lean_mass",
    "muscular_dystrophy",
]

# Natri 2024: sig RDS tarball (831 MB)
NATRI_SIG_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE227nnn/GSE227136/"
    "suppl/GSE227136_ieQTL_mashr_applied_sig.tar.gz"
)
# Natri 2024: per-lineage full mashr tars (3-10 GB each) — fallback only
NATRI_LINEAGE_URLS: dict[str, str] = {
    "mesenchymal": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE227nnn/GSE227136/suppl/GSE227136_mashr_applied_mesenchymal.tar.gz",
    "endothelial": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE227nnn/GSE227136/suppl/GSE227136_mashr_applied_endothelial.tar.gz",
    "immune":      "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE227nnn/GSE227136/suppl/GSE227136_mashr_applied_immune.tar.gz",
}

# OneK1K: 14 immune cell types. S3 URL template.
ONEK1K_CELL_TYPES: list[str] = [
    "cd4et", "cd4nc", "cd4sox4",
    "cd8et", "cd8nc", "cd8s100b",
    "nk", "nkr",
    "bmem", "bin", "plasma",
    "monc", "monnc",
    "dc",
]
ONEK1K_EQTL_URL_TEMPLATE = (
    "https://onek1k.s3.ap-southeast-2.amazonaws.com/eqtl/{ct}_eqtl_table.tsv.gz"
)
ONEK1K_ESNP_URL_TEMPLATE = (
    "https://onek1k.s3.ap-southeast-2.amazonaws.com/esnp/{ct}_esnp_table.tsv.gz"
)

# Power / decision thresholds
BH_ALPHA = 0.05
LFSR_THRESHOLD = 0.05   # mashR significance (Urbut 2019 Nat Genet)
COLOC_H4_THRESHOLD = 0.80   # not run this iter; kept for verdict logic

# Request config
REQ_TIMEOUT_CONNECT = 30
REQ_TIMEOUT_READ = 600
USER_AGENT = "biomarvin-fibro/iter065-t4 (research; ben@iluvatarlabs.com)"
HEADERS = {"User-Agent": USER_AGENT}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("t4")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_PATH, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


LOG = _setup_logging()


# ---------------------------------------------------------------------------
# Utility: download-with-retry
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: Path, retries: int = 3) -> Path:
    """Download url to dest; resume-skip if file already complete.

    WHY retries: Per CLAUDE.md "Retry CLI/API failures ... up to 3 times with
    increasing delay." GEO FTP and S3 can transiently 500/timeout.
    """
    if dest.exists() and dest.stat().st_size > 0:
        LOG.info(f"    [cache hit] {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            with requests.get(
                url,
                stream=True,
                headers=HEADERS,
                timeout=(REQ_TIMEOUT_CONNECT, REQ_TIMEOUT_READ),
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", "0"))
                LOG.info(f"    Downloading {url} -> {dest.name} (expected {total:,} B)")
                with open(dest, "wb") as f:
                    got = 0
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                        if chunk:
                            f.write(chunk)
                            got += len(chunk)
                dt = time.time() - t0
                LOG.info(f"    Downloaded {got:,} B in {dt:.1f} s ({got / max(dt, 1) / 1e6:.1f} MB/s)")
                return dest
        except Exception as e:  # noqa: BLE001
            last_exc = e
            LOG.warning(f"    Attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Download failed after {retries} attempts: {url}") from last_exc


# ---------------------------------------------------------------------------
# Step 0: Resolve TF coordinates via mygene
# ---------------------------------------------------------------------------

def resolve_tf_coords() -> pd.DataFrame:
    """Resolve 9 TFs → ENSG + chromosome + start + end via mygene.info.

    WHY mygene over hard-coded ENSG: ensures TF list remains symbol-driven and
    survives downstream ENSG-vs-symbol mismatch between Natri (ENSG) and
    OneK1K (symbol).
    """
    try:
        import mygene
    except ImportError as e:
        raise RuntimeError("mygene not installed") from e
    mg = mygene.MyGeneInfo()
    LOG.info(f"Resolving coords for {len(TFS)} TFs via mygene.info")
    results = mg.querymany(
        TFS,
        scopes="symbol",
        fields="ensembl.gene,symbol,genomic_pos_hg38,genomic_pos",
        species="human",
        returnall=False,
    )
    rows = []
    for r in results:
        sym = r.get("query")
        ensg: str | None = None
        # ensembl.gene can be str or list-of-dicts depending on ambiguity
        ens_field = r.get("ensembl")
        if isinstance(ens_field, dict):
            ensg = ens_field.get("gene")
        elif isinstance(ens_field, list) and ens_field:
            ensg = ens_field[0].get("gene")
        pos = r.get("genomic_pos_hg38") or r.get("genomic_pos")
        if isinstance(pos, list):
            # Choose the one with ENSG match or the first
            pos = pos[0]
        chrom: str | None = None
        start: int | None = None
        end: int | None = None
        if isinstance(pos, dict):
            chrom = pos.get("chr")
            start = pos.get("start")
            end = pos.get("end")
        rows.append(
            {
                "tf": sym,
                "ensg": ensg,
                "chrom": chrom,
                "start": start,
                "end": end,
            }
        )
    df = pd.DataFrame(rows)
    # Hard-coded fallbacks from iter_064 a_opentargets_grid.csv where mygene
    # returns nothing (no fabrication: these ENSG were verified in iter 064).
    fallbacks = {
        "JUNB":   "ENSG00000171223",
        "FOS":    "ENSG00000170345",
        "EGR1":   "ENSG00000120738",
        "EGR2":   "ENSG00000122877",
        "ATF3":   "ENSG00000162772",
        "CEBPB":  "ENSG00000172216",
        "KLF10":  "ENSG00000155090",
        "IRF1":   "ENSG00000125347",
        "CDKN1A": "ENSG00000124762",
    }
    for i, row in df.iterrows():
        if not row["ensg"] and row["tf"] in fallbacks:
            df.at[i, "ensg"] = fallbacks[row["tf"]]
            LOG.info(f"    Used iter_064 fallback ENSG for {row['tf']}")
    LOG.info(f"Resolved TF coord table:\n{df.to_string(index=False)}")
    return df


# ---------------------------------------------------------------------------
# Step 1: OneK1K pipeline — stream eqtl tables, filter by TF GENE symbol
# ---------------------------------------------------------------------------

@dataclass
class OneK1KHit:
    dataset: str = "OneK1K_Yazar_2022"
    cell_type: str = ""
    tf: str = ""
    gene_id: str = ""
    rsid: str = ""
    chrom: str = ""
    pos: int = 0
    a1: str = ""
    a2: str = ""
    spearmans_rho: float = np.nan
    p_value: float = np.nan
    q_value: float = np.nan   # within-cell-type Q
    fdr: float = np.nan       # cross-gene FDR per OneK1K README
    genotyped: str = ""
    lead: bool = False        # was this row in the esnp (lead) table?


def _stream_onek1k_eqtl_table(ct: str, tfs: set[str]) -> list[OneK1KHit]:
    """Stream-filter one OneK1K eqtl table by GENE symbol ∈ tfs.

    WHY stream: per-cell-type file is ~1.6 GB; storing all 14 locally is ~22 GB
    and wastes budget. Requests + gzip.GzipFile streaming keeps memory O(1).
    """
    url = ONEK1K_EQTL_URL_TEMPLATE.format(ct=ct)
    LOG.info(f"    OneK1K stream-filter {ct} from {url}")
    hits: list[OneK1KHit] = []
    t0 = time.time()
    with requests.get(
        url,
        stream=True,
        headers=HEADERS,
        timeout=(REQ_TIMEOUT_CONNECT, REQ_TIMEOUT_READ),
    ) as r:
        r.raise_for_status()
        # Wrap response.raw with gzip streaming then text decode
        gz = gzip.GzipFile(fileobj=r.raw)
        reader = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        header = reader.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        gene_i = idx["GENE"]
        required = ("CELL_TYPE", "RSID", "SNPID", "GENE_ID", "CHR", "POS",
                    "A1", "A2", "SPEARMANS_RHO", "P_VALUE", "Q_VALUE", "FDR",
                    "GENOTYPED")
        missing = [c for c in required if c not in idx]
        if missing:
            raise RuntimeError(f"OneK1K {ct} missing cols {missing}")
        n_total = 0
        for line in reader:
            n_total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= gene_i:
                continue
            g = parts[gene_i]
            if g not in tfs:
                continue
            def _s(c: str) -> str:
                return parts[idx[c]] if idx[c] < len(parts) else ""

            def _f(c: str) -> float:
                v = _s(c)
                try:
                    return float(v) if v not in ("", "NA") else np.nan
                except ValueError:
                    return np.nan

            def _i(c: str) -> int:
                v = _s(c)
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    return 0

            h = OneK1KHit(
                cell_type=ct,
                tf=g,
                gene_id=_s("GENE_ID"),
                rsid=_s("RSID"),
                chrom=_s("CHR"),
                pos=_i("POS"),
                a1=_s("A1"),
                a2=_s("A2"),
                spearmans_rho=_f("SPEARMANS_RHO"),
                p_value=_f("P_VALUE"),
                q_value=_f("Q_VALUE"),
                fdr=_f("FDR"),
                genotyped=_s("GENOTYPED"),
            )
            hits.append(h)
        dt = time.time() - t0
        LOG.info(f"    {ct}: scanned {n_total:,} rows, kept {len(hits)} TF rows in {dt:.1f} s")
    return hits


def _fetch_onek1k_esnps(ct: str, tfs: set[str]) -> set[tuple[str, str]]:
    """Fetch OneK1K esnp (lead-only, FDR<0.05) table for ct; return set of
    (tf, rsid) that are FDR-sig LEAD eSNPs for our TFs."""
    url = ONEK1K_ESNP_URL_TEMPLATE.format(ct=ct)
    dest = DATA_CACHE / "onek1k" / f"{ct}_esnp.tsv.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        _download_file(url, dest)
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"    OneK1K esnp {ct} fetch failed: {e}")
        return set()
    leads: set[tuple[str, str]] = set()
    with gzip.open(dest, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= idx["GENE"]:
                continue
            g = parts[idx["GENE"]]
            if g in tfs:
                leads.add((g, parts[idx["RSID"]]))
    return leads


def run_onek1k(tf_coords: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run OneK1K lookup for 9 TFs × 14 cell types.

    Returns DataFrame of all (TF, cell_type) rows and per-dataset verdict dict.
    """
    LOG.info("=" * 70)
    LOG.info("OneK1K sweep (N=982 donors, 14 immune cell types)")
    LOG.info("=" * 70)
    tfs_symbol = set(tf_coords["tf"].tolist())

    # Lead-only esnp (indicator of FDR-passing lead)
    leads_by_ct: dict[str, set[tuple[str, str]]] = {}
    for ct in ONEK1K_CELL_TYPES:
        leads_by_ct[ct] = _fetch_onek1k_esnps(ct, tfs_symbol)
        if leads_by_ct[ct]:
            LOG.info(f"    OneK1K {ct} LEAD esnps for TFs: {leads_by_ct[ct]}")

    # Full stream-filter for all (SNP, TF) pairs — for min-p lookup
    all_hits: list[OneK1KHit] = []
    for ct in ONEK1K_CELL_TYPES:
        try:
            hits = _stream_onek1k_eqtl_table(ct, tfs_symbol)
        except Exception as e:  # noqa: BLE001
            LOG.error(f"    OneK1K {ct} stream failed: {e}")
            hits = []
        # Tag lead status
        leads = leads_by_ct.get(ct, set())
        for h in hits:
            if (h.tf, h.rsid) in leads:
                h.lead = True
        all_hits.extend(hits)

    if not all_hits:
        LOG.warning("OneK1K: 0 rows found across all TFs × cell types. "
                    "Could indicate stream failure or TFs genuinely absent "
                    "from OneK1K cis-window gene list.")
        per_tf_cell = pd.DataFrame(
            columns=["dataset", "cell_type", "tf", "n_cis_variants",
                     "min_p", "min_q", "min_fdr", "lead_sig", "rsid_best"]
        )
    else:
        df_all = pd.DataFrame([asdict(h) for h in all_hits])
        # Per (TF × cell_type) aggregation: best (min-p) row
        def _agg(g: pd.DataFrame) -> pd.Series:
            best = g.loc[g["p_value"].idxmin()] if g["p_value"].notna().any() else g.iloc[0]
            return pd.Series({
                "dataset": "OneK1K_Yazar_2022",
                "cell_type": g["cell_type"].iat[0],
                "tf": g["tf"].iat[0],
                "n_cis_variants": int(len(g)),
                "min_p": float(g["p_value"].min()) if g["p_value"].notna().any() else np.nan,
                "min_q": float(g["q_value"].min()) if g["q_value"].notna().any() else np.nan,
                "min_fdr": float(g["fdr"].min()) if g["fdr"].notna().any() else np.nan,
                "lead_sig": bool(g["lead"].any()),
                "rsid_best": best["rsid"],
            })

        per_tf_cell = (
            df_all.groupby(["cell_type", "tf"], as_index=False)
                  .apply(_agg)
                  .reset_index(drop=True)
        )
        # Also persist full hit-level file (may be large but compressible)
        full_path = OUT_PER_DATASET / "onek1k_all_cis_rows.csv.gz"
        df_all.to_csv(full_path, index=False, compression="gzip")
        LOG.info(f"    Wrote full OneK1K TF hits: {full_path}")

    out_path = OUT_PER_DATASET / "onek1k_cis_eqtl.csv"
    per_tf_cell.to_csv(out_path, index=False)
    LOG.info(f"    Wrote OneK1K per-(TF, cell_type) summary: {out_path}")

    # Per-dataset BH correction across 9 TFs × 14 cell types = 126 tests
    verdict = _classify_per_dataset(
        per_tf_cell, name="OneK1K_Yazar_2022", n_donors=982, adequate_power=True
    )
    return per_tf_cell, verdict


# ---------------------------------------------------------------------------
# Step 2: Natri 2024 pipeline — parse mashR sig .rds via pyreadr
# ---------------------------------------------------------------------------

def run_natri(tf_coords: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Natri 2024 lookup via the sig RDS tarball.

    Strategy (see t4_data_availability.md):
      1. Download `_ieQTL_mashr_applied_sig.tar.gz` (831 MB).
      2. Extract .rds into tempdir; load via pyreadr.
      3. Filter rows by feature_id ∈ 9 TF ENSG.
      4. Aggregate by (TF × cell_type); report min lfsr, posterior β, and
         pass/fail at lfsr<0.05 (mashR significance, Urbut 2019 Nat Genet).

    Returns per-(TF × cell_type) DataFrame and per-dataset verdict.
    """
    LOG.info("=" * 70)
    LOG.info("Natri 2024 sweep (GSE227136, N=114, 38 cell types)")
    LOG.info("=" * 70)

    natri_dir = DATA_CACHE / "natri2024"
    natri_dir.mkdir(parents=True, exist_ok=True)
    sig_tar = natri_dir / "GSE227136_ieQTL_mashr_applied_sig.tar.gz"
    try:
        _download_file(NATRI_SIG_URL, sig_tar)
    except Exception as e:  # noqa: BLE001
        LOG.error(f"    Natri sig download failed: {e}")
        empty = pd.DataFrame(
            columns=["dataset", "cell_type", "tf", "n_cis_variants",
                     "min_lfsr", "best_posterior_mean", "best_sd",
                     "lead_sig", "rsid_best"]
        )
        verdict = {
            "dataset": "Natri_2024",
            "status": "BLOCKED_DOWNLOAD_FAILURE",
            "n_donors": 114,
            "error": str(e),
            "classification": "UNINTERPRETABLE",
        }
        return empty, verdict

    LOG.info(f"    Extracting {sig_tar.name} (size={sig_tar.stat().st_size:,} B)")
    with tempfile.TemporaryDirectory(prefix="natri_sig_") as tmpd:
        with tarfile.open(sig_tar, mode="r:gz") as tf:
            members = tf.getmembers()
            rds_members = [m for m in members if m.name.endswith(".rds")]
            LOG.info(f"    Tar contents: {[m.name for m in members]}")
            if not rds_members:
                raise RuntimeError("No .rds file in Natri sig tarball")
            for m in rds_members:
                tf.extract(m, path=tmpd)
            rds_path = Path(tmpd) / rds_members[0].name
        LOG.info(f"    Parsing {rds_path} via pyreadr")
        try:
            import pyreadr
        except ImportError as e:
            raise RuntimeError("pyreadr required for Natri RDS parse") from e
        try:
            result = pyreadr.read_r(str(rds_path))
        except Exception as e:  # noqa: BLE001
            LOG.error(f"    pyreadr failed on Natri .rds: {e}")
            LOG.error("    This typically means the RDS contains non-data.frame "
                      "R objects (e.g., mashR S4). Fallback: per-lineage mashR "
                      "TSV streaming is NOT invoked this iter due to 16 GB size.")
            empty = pd.DataFrame()
            verdict = {
                "dataset": "Natri_2024",
                "status": "BLOCKED_RDS_PARSE_FAILURE",
                "n_donors": 114,
                "error": str(e),
                "classification": "UNINTERPRETABLE",
                "note": "RDS not parseable by pyreadr; mashR S4 object likely. "
                        "Iter 066 option: install rpy2 or use R to export to TSV.",
            }
            return empty, verdict

        LOG.info(f"    pyreadr returned keys: {list(result.keys())}")
        # Concatenate all returned data frames into a single table
        frames = []
        for k, df in result.items():
            if df is None or len(df) == 0:
                continue
            df = df.copy()
            df["_rds_key"] = k
            frames.append(df)
        if not frames:
            raise RuntimeError("Natri sig RDS parsed but returned 0 rows")
        nat_df = pd.concat(frames, ignore_index=True)
        LOG.info(f"    Natri sig rows: {len(nat_df):,}; cols: {list(nat_df.columns)}")

    # Filter to 9 TF ENSG. The README says feature_id = ENSG (no version suffix).
    tf_ensg = {r["ensg"].split(".")[0] for _, r in tf_coords.iterrows()
               if isinstance(r["ensg"], str) and r["ensg"]}
    # Find feature column — try 'feature_id' first then any col containing 'gene' or 'ensembl'
    feat_col = None
    for cand in ("feature_id", "gene", "ensembl_gene_id"):
        if cand in nat_df.columns:
            feat_col = cand
            break
    if feat_col is None:
        raise RuntimeError(f"Cannot identify feature column in Natri: {list(nat_df.columns)}")
    LOG.info(f"    Using feature column: {feat_col}")

    # Strip version suffix to match
    nat_df["_feat_noversion"] = nat_df[feat_col].astype(str).str.split(".").str[0]
    mask = nat_df["_feat_noversion"].isin(tf_ensg)
    nat_hits = nat_df.loc[mask].copy()
    LOG.info(f"    Natri sig rows matching our 9 TFs: {len(nat_hits)}")

    # Map ENSG → symbol
    ensg_to_sym = {r["ensg"].split(".")[0]: r["tf"] for _, r in tf_coords.iterrows()
                   if isinstance(r["ensg"], str) and r["ensg"]}
    nat_hits["tf"] = nat_hits["_feat_noversion"].map(ensg_to_sym)

    # Identify cell-type col: README / sig convention likely has 'cell_type'
    ct_col = None
    for cand in ("cell_type", "celltype", "CellType", "cell_pop", "ct"):
        if cand in nat_hits.columns:
            ct_col = cand
            break
    if ct_col is None and "_rds_key" in nat_hits.columns:
        ct_col = "_rds_key"
        LOG.info(f"    No cell_type col; using RDS key as cell-type stratum")

    # Identify effect / lfsr cols
    def _find_col(candidates: list[str]) -> str | None:
        for c in candidates:
            if c in nat_hits.columns:
                return c
        return None

    lfsr_col = _find_col(["lfsr", "LFSR", "local_fdr"])
    beta_col = _find_col(["posterior_means", "beta", "effect_size", "PosteriorMean"])
    sd_col = _find_col(["sds", "se", "SD", "posterior_sd"])
    rsid_col = _find_col(["snp_rsid", "rsid", "SNP", "variant_id"])

    LOG.info(f"    Natri cols resolved: ct={ct_col} lfsr={lfsr_col} beta={beta_col} sd={sd_col} rsid={rsid_col}")

    if ct_col is None or lfsr_col is None:
        LOG.warning("    Natri: missing cell_type or lfsr col; returning raw hits only.")
        nat_hits.to_csv(OUT_PER_DATASET / "natri_raw_tf_hits.csv", index=False)
        empty = pd.DataFrame()
        verdict = {
            "dataset": "Natri_2024",
            "status": "PARTIAL_SCHEMA_MISMATCH",
            "n_donors": 114,
            "classification": "UNINTERPRETABLE",
            "n_raw_hits": int(len(nat_hits)),
        }
        return empty, verdict

    # Aggregate: per (TF, cell_type) min lfsr
    def _agg_natri(g: pd.DataFrame) -> pd.Series:
        best_idx = g[lfsr_col].idxmin() if g[lfsr_col].notna().any() else g.index[0]
        best = g.loc[best_idx]
        return pd.Series({
            "dataset": "Natri_2024",
            "cell_type": g[ct_col].iat[0],
            "tf": g["tf"].iat[0],
            "n_cis_variants": int(len(g)),
            "min_lfsr": float(g[lfsr_col].min()),
            "best_posterior_mean": float(best[beta_col]) if beta_col else np.nan,
            "best_sd": float(best[sd_col]) if sd_col else np.nan,
            "rsid_best": str(best[rsid_col]) if rsid_col else "",
            "lead_sig": bool(g[lfsr_col].min() < LFSR_THRESHOLD),
        })

    per_tf_cell = (
        nat_hits.groupby([ct_col, "tf"], as_index=False)
                .apply(_agg_natri)
                .reset_index(drop=True)
    )

    # Persist
    out_path = OUT_PER_DATASET / "natri_cis_eqtl.csv"
    per_tf_cell.to_csv(out_path, index=False)
    raw_path = OUT_PER_DATASET / "natri_raw_tf_hits.csv"
    nat_hits.drop(columns=["_feat_noversion"], errors="ignore").to_csv(raw_path, index=False)
    LOG.info(f"    Wrote Natri per-(TF, cell_type) summary: {out_path} (rows={len(per_tf_cell)})")

    # Per-dataset verdict: Natri N=114 → INCONCLUSIVE on null
    verdict = _classify_per_dataset(
        per_tf_cell,
        name="Natri_2024",
        n_donors=114,
        adequate_power=False,
        pvalue_col="min_lfsr",  # treat lfsr as analog-FDR
    )
    return per_tf_cell, verdict


# ---------------------------------------------------------------------------
# Step 3: Per-dataset BH classification
# ---------------------------------------------------------------------------

def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values. Returns array of same length; NaN input → NaN q.

    WHY: per-dataset BH is the Critic 2 B4 resolution in brief §T4.
    """
    q = np.full_like(pvals, np.nan, dtype=float)
    valid_mask = ~np.isnan(pvals)
    valid = pvals[valid_mask]
    n = len(valid)
    if n == 0:
        return q
    order = np.argsort(valid)
    ranked = valid[order]
    q_ranked = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity (standard BH)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)
    q_sorted = np.empty(n)
    q_sorted[order] = q_ranked
    q[valid_mask] = q_sorted
    return q


def _classify_per_dataset(
    df: pd.DataFrame,
    name: str,
    n_donors: int,
    adequate_power: bool,
    pvalue_col: str = "min_p",
) -> dict[str, Any]:
    """Apply per-dataset BH + brief §T4 DECISION RULE.

    WHY not pooled BH: per brief §T4 Critic 2 B4, pooling across datasets with
    different N / cell types / traits conflates adequately-powered null with
    underpowered null. Each dataset gets its own BH pool.
    """
    out = {
        "dataset": name,
        "n_donors": n_donors,
        "adequate_power_for_beta_0.15": adequate_power,
        "n_tests": int(len(df)),
        "bh_alpha": BH_ALPHA,
        "pvalue_col": pvalue_col,
        "classification": None,
    }
    if len(df) == 0:
        out["classification"] = "NO_TF_ROWS"
        out["note"] = "Pipeline found 0 TF × cell-type rows; likely data-access or schema issue."
        return out

    if pvalue_col not in df.columns:
        out["classification"] = "SCHEMA_MISSING_PVAL"
        out["note"] = f"Column {pvalue_col} missing; cannot apply BH."
        return out

    pvals = df[pvalue_col].to_numpy(dtype=float)
    qvals = _bh_correct(pvals)
    df = df.copy()
    df["q_bh_per_dataset"] = qvals
    df["sig_bh"] = (qvals < BH_ALPHA) & ~np.isnan(qvals)

    # Persist enriched table
    enriched_path = OUT_PER_DATASET / f"{name.lower()}_cis_eqtl_with_q.csv"
    df.to_csv(enriched_path, index=False)
    out["enriched_path"] = str(enriched_path)

    n_sig = int(df["sig_bh"].sum())
    out["n_sig_bh"] = n_sig
    if n_sig > 0:
        # Coloc would be attempted here if GWAS sumstats available (see brief).
        # They aren't this iter; coloc H4 is NOT_RUN; classification per brief
        # requires BH AND H4>0.8 for SUGGESTED. Without H4, we cannot upgrade.
        out["classification"] = "BH_SIG_BUT_COLOC_NOT_RUN"
        out["note"] = (
            f"{n_sig} (TF × cell-type) rows pass BH q<{BH_ALPHA}. "
            f"Per brief §T4 SUGGESTED requires BH q<0.05 AND H4>0.8. "
            f"coloc.abf was NOT_RUN this iter (no GWAS sumstats cached locally). "
            f"Verdict: lookup-only. Upgrade to SUGGESTED requires iter 066 coloc."
        )
        out["top_hits"] = (
            df.loc[df["sig_bh"]]
              .nsmallest(20, pvalue_col)
              .to_dict(orient="records")
        )
    else:
        if adequate_power:
            out["classification"] = "REFUTED_ONEK1K_COVERED_TFS"
            out["note"] = (
                f"0 TF × cell-type rows pass BH q<{BH_ALPHA} in N={n_donors} "
                f"(adequately powered for β>0.15). Per brief §T4 this is REFUTED "
                f"for OneK1K-covered TFs (CEBPB, IRF1, CDKN1A, FOS, JUN subset)."
            )
        else:
            out["classification"] = "INCONCLUSIVE_UNDERPOWERED"
            out["note"] = (
                f"0 TF × cell-type rows pass BH q<{BH_ALPHA} in N={n_donors} "
                f"(underpowered for β<0.3). Per brief §T4 null is INCONCLUSIVE, "
                f"not REFUTED."
            )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _log_env() -> dict[str, Any]:
    import platform
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lib_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "requests": requests.__version__,
        },
    }
    try:
        import pyreadr
        env["lib_versions"]["pyreadr"] = pyreadr.__version__
    except ImportError:
        env["lib_versions"]["pyreadr"] = "MISSING"
    try:
        import mygene
        env["lib_versions"]["mygene"] = mygene.__version__
    except ImportError:
        env["lib_versions"]["mygene"] = "MISSING"
    LOG.info(f"Environment: {json.dumps(env, indent=2)}")
    return env


def main() -> None:
    t0 = time.time()
    env = _log_env()

    LOG.info(f"TFs ({len(TFS)}): {TFS}")
    LOG.info(f"Traits ({len(TRAITS)}): {TRAITS}")

    # Step 0: TF coords
    tf_coords = resolve_tf_coords()
    tf_coords.to_csv(OUT_PER_DATASET / "tf_coords.csv", index=False)

    results: dict[str, Any] = {
        "iteration": 65,
        "batch": "batch_065",
        "test": "T4_sc_eQTL_sweep",
        "env": env,
        "tfs": TFS,
        "traits": TRAITS,
        "datasets": {},
    }

    # Step 1: OneK1K (adequate power, N=982)
    try:
        ok_df, ok_verdict = run_onek1k(tf_coords)
    except Exception as e:  # noqa: BLE001
        LOG.exception("OneK1K pipeline failed")
        ok_verdict = {
            "dataset": "OneK1K_Yazar_2022",
            "status": "PIPELINE_FAILURE",
            "error": str(e),
            "classification": "UNINTERPRETABLE",
        }
    results["datasets"]["OneK1K_Yazar_2022"] = ok_verdict

    # Step 2: Natri 2024 (N=114, sig-only RDS)
    try:
        nat_df, nat_verdict = run_natri(tf_coords)
    except Exception as e:  # noqa: BLE001
        LOG.exception("Natri pipeline failed")
        nat_verdict = {
            "dataset": "Natri_2024",
            "status": "PIPELINE_FAILURE",
            "error": str(e),
            "classification": "UNINTERPRETABLE",
        }
    results["datasets"]["Natri_2024"] = nat_verdict

    # Step 3: Soskic — BLOCKED
    results["datasets"]["Soskic_2022"] = {
        "dataset": "Soskic_2022",
        "n_donors": 119,
        "status": "BLOCKED_EGA_CONTROLLED_ACCESS",
        "classification": "BLOCKED",
        "note": "EGAS00001005839 requires DAC approval. Skipped per brief §T4 "
                "instruction: 'If any dataset requires EGA / dbGaP access: skip "
                "that dataset and document as BLOCKED.' Supplementary Data 5-7 "
                "(Soskic 2022 Nature Genetics) may allow sig-only TF lookup; "
                "deferred to iter 066.",
    }

    # Empty coloc file — keep artifact path consistent with brief
    pd.DataFrame(columns=[
        "dataset", "tf", "cell_type", "trait", "efo",
        "eqtl_rsid", "eqtl_p", "eqtl_beta", "eqtl_se",
        "gwas_p", "gwas_beta", "gwas_se",
        "coloc_h0", "coloc_h1", "coloc_h2", "coloc_h3", "coloc_h4",
        "status",
    ]).to_csv(OUT_COLOC_CSV, index=False)
    LOG.info(f"Wrote empty coloc placeholder: {OUT_COLOC_CSV} "
             "(per brief: coloc NOT_RUN this iter; GWAS sumstats not cached locally)")
    results["coloc"] = {
        "status": "NOT_RUN",
        "reason": ("Brief §T4 allows 'report min_p from each and lookup-only' "
                   "if coloc not feasible. GWAS sumstats (6 traits) not cached "
                   "locally; fetching 6 GWAS sumstats + coloc.abf exceeds 30-90 "
                   "min budget. Deferred to iter 066 conditional on ≥1 per-dataset "
                   "BH-sig hit."),
        "artifact": str(OUT_COLOC_CSV),
    }

    # Overall classification
    dataset_classes = {
        n: v.get("classification") for n, v in results["datasets"].items()
    }
    # If any dataset hit BH → SUGGESTED (coloc pending); else combine REFUTED/INCONCLUSIVE
    overall = "T4_BLOCKED"
    suggestive = [n for n, c in dataset_classes.items()
                  if c == "BH_SIG_BUT_COLOC_NOT_RUN"]
    refuted = [n for n, c in dataset_classes.items()
               if c == "REFUTED_ONEK1K_COVERED_TFS"]
    inconclusive = [n for n, c in dataset_classes.items()
                    if c == "INCONCLUSIVE_UNDERPOWERED"]
    blocked = [n for n, c in dataset_classes.items() if c == "BLOCKED"]
    uninterp = [n for n, c in dataset_classes.items()
                if c == "UNINTERPRETABLE"]

    if suggestive:
        overall = ("SUGGESTIVE_PENDING_COLOC "
                   f"(datasets: {suggestive}; coloc required for ESTABLISHED)")
    elif uninterp and not refuted and not inconclusive:
        overall = "UNINTERPRETABLE"
    elif refuted and not suggestive:
        overall = ("REFUTED_ONEK1K_COVERED_TFS "
                   f"(OneK1K null for {dataset_classes}; "
                   "Natri/Soskic INCONCLUSIVE/BLOCKED)")
    else:
        overall = f"MIXED ({dataset_classes})"

    results["overall_classification"] = overall
    results["dataset_classifications"] = dataset_classes
    results["runtime_seconds"] = time.time() - t0

    with open(OUT_SUMMARY_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    LOG.info(f"Wrote summary: {OUT_SUMMARY_JSON}")
    LOG.info(f"Overall: {overall}")
    LOG.info(f"Total runtime: {results['runtime_seconds']:.1f} s")


if __name__ == "__main__":
    main()
