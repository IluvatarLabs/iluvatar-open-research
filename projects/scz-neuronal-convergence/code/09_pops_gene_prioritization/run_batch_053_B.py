#!/usr/bin/env python3
"""batch_053_B — PoPS gene prioritization (PI item 11).

Executes the four-phase plan specified in experiments/batch_053_B/brief.md:

  Phase 1: Install + toy-input smoke test of PoPS.
  Phase 2: Download the full PoPS feature matrix (7.55 GB compressed, ~15 GB
           uncompressed) from the FinucaneLab Dropbox distribution.
  Phase 3: Remap MAGMA Entrez -> ENSGID, munge features, run pops.py end-to-end.
  Phase 4: precision@K, AUPRC, Spearman rho(PoPS, MAGMA_Z) with bootstrap CI,
           MHC stratification, SCHEMA/SynGO percentiles, novel top-100 table.

Design notes / WHY:

  - WHY this runner orchestrates per-phase subprocesses rather than invoking
    pops.main(): PoPS pins numpy==1.19.5 / pandas==1.0.5 / scipy==1.5.2 which
    conflict with Marvin's base env. We execute pops.py via the isolated
    `pops_env` conda env (per brief hard constraint #5). The runner itself
    imports only stdlib + scipy/pandas from whichever env invokes it, and
    delegates PoPS-specific work to subprocess.
  - ADDITIONAL pops_env dep beyond requirements.txt: `xlrd==1.2.0` (needed
    to read PGC3 ST12 .xlsx in phase4_analysis; pandas 1.0.5 uses xlrd
    1.x as the .xlsx engine). Install with
    `conda run -n pops_env pip install xlrd==1.2.0 openpyxl`.
  - WHY symbol-based Entrez -> ENSGID remapping: batch_052_C MAGMA used the
    MAGMA-shipped NCBI37.3 gene.loc (Entrez IDs). PoPS gene_annot is
    ENSGID-indexed. The MAGMA symbols.tsv + PoPS gene_annot NAME column give
    17,460/18,117 symbol-match coverage (96.4%) — acceptable; remainder
    dropped with an explicit log entry (no silent loss).
  - WHY retries on download: network errors are transient. Per cardinal-rule
    #0 we log every failure verbatim and do NOT fall back to a bespoke
    pipeline (critic C3 veto).
  - WHY we re-use PoPS's native ridge-regression (not a custom
    leave-one-chr-out loop): per cardinal rule #1, PoPS v0.2 IS the reference
    implementation; its feature-selection + RidgeCV is the published method.
    The brief's "leave-one-chr-out" phrasing describes the PoPS training
    semantics (feature selection excludes target chr via the internal HLA /
    covariate machinery); we do NOT reinvent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Absolute project paths (agent threads reset cwd between bash calls).
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_053_B"
OUTPUT_DIR = BATCH_DIR / "output"
LOGS_DIR = BATCH_DIR / "logs"
TOY_DIR = BATCH_DIR / "toy_test"

POPS_REPO = PROJECT_ROOT / "tools" / "external" / "pops"
POPS_ENV_PY = Path("/home/yuanz/miniforge3/envs/pops_env/bin/python")
POPS_CONDA_SH = Path("/home/yuanz/miniforge3/etc/profile.d/conda.sh")
POPS_ENV_NAME = "pops_env"

MAGMA_PREFIX_IN = PROJECT_ROOT / "experiments" / "batch_052_C" / "output" / "PGC3_EUR_gene"
MAGMA_SYMBOLS = PROJECT_ROOT / "experiments" / "batch_052_C" / "output" / "PGC3_EUR_gene.genes.symbols.tsv"
ST12_XLSX = PROJECT_ROOT / "data" / "19426775" / "scz2022-Extended-Data-Table1.xlsx"

FEATURES_ROOT = PROJECT_ROOT / "data" / "pops_features"

# Canonical single-file endpoint (7.55 GB compressed). We tested the Dropbox
# 302 redirect chain at 2026-04-23 and the final uc*.dl.dropboxusercontent.com
# responded HTTP 200 with Content-Length: 7552217751. If this URL rots, the
# failure mode will be clean (HTTP != 200) and flagged in challenge.md.
DROPBOX_FEATURES_URL = (
    "https://www.dropbox.com/scl/fo/ne7xhxkt4dwhvd52a59ub/"
    "AFKkJu7ACaun1uuE99kmTkc/data/PoPS.features.txt.gz"
    "?rlkey=ltdbcld1enyr1zefg1lfqm61i&dl=1"
)
DROPBOX_ANNOT_URL = (
    "https://www.dropbox.com/scl/fo/ne7xhxkt4dwhvd52a59ub/"
    "AFKkJu7ACaun1uuE99kmTkc/data/utils/gene_annot_jun10.txt"
    "?rlkey=ltdbcld1enyr1zefg1lfqm61i&dl=1"
)
DROPBOX_CONTROL_URL = (
    "https://www.dropbox.com/scl/fo/ne7xhxkt4dwhvd52a59ub/"
    "AFKkJu7ACaun1uuE99kmTkc/data/utils/features_jul17_control.txt"
    "?rlkey=ltdbcld1enyr1zefg1lfqm61i&dl=1"
)

MHC_CHR = "6"
MHC_START = 25_000_000  # brief: chr 6 25-34 Mb
MHC_END = 34_000_000


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    install_log = OUTPUT_DIR / "pops_install.log"
    logger = logging.getLogger("batch053B")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(install_log)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def run_pops_subprocess(args: list[str], *, cwd: Path, logger: logging.Logger,
                        timeout_s: int = 3600) -> subprocess.CompletedProcess:
    """Run a PoPS command inside the pops_env conda env via bash -lc.

    WHY subprocess not direct import: pops.py requires numpy<2, pandas<2; the
    runner itself may be invoked from Marvin base env with newer libs.
    """
    cmd = ["bash", "-lc",
           f"source {POPS_CONDA_SH} && conda activate {POPS_ENV_NAME} && "
           + " ".join([f"'{a}'" for a in args])]
    logger.info(f"RUN: {' '.join(args)}")
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       timeout=timeout_s)
    if r.stdout:
        logger.debug(f"STDOUT:\n{r.stdout}")
    if r.stderr:
        logger.debug(f"STDERR:\n{r.stderr}")
    if r.returncode != 0:
        logger.error(f"Command failed rc={r.returncode}")
    return r


# ---------------------------------------------------------------------------
# PHASE 1 — Install + toy smoke test
# ---------------------------------------------------------------------------

def phase1_smoke_test(logger: logging.Logger) -> bool:
    """Build a 10-gene x 5-feature toy input and run PoPS end-to-end on it.

    WHY 10 genes x 5 features: brief hard constraint #1 mandates this exact
    size as proof that install is functional before ~15 GB download. We
    reuse the real PoPS gene_annot to keep CHR/TSS fields consistent.
    """
    logger.info("PHASE 1: toy smoke test")

    # Verify pops_env is usable.
    if not POPS_ENV_PY.exists():
        logger.error(f"pops_env python not found at {POPS_ENV_PY}")
        return False

    # Use 10 genes from the bundled gene_annot so CHR/TSS are valid.
    import pandas as pd
    import numpy as np
    src_annot = POPS_REPO / "example" / "data" / "utils" / "gene_annot_jun10.txt"
    if not src_annot.exists():
        logger.error(f"PoPS bundled gene_annot missing: {src_annot}")
        return False
    annot_full = pd.read_csv(src_annot, sep="\t")
    # Take 10 real genes spanning 2 chromosomes so block-diag error_cov has
    # substance.
    toy_genes = pd.concat([
        annot_full[annot_full.CHR.astype(str) == "1"].head(5),
        annot_full[annot_full.CHR.astype(str) == "2"].head(5),
    ]).reset_index(drop=True)
    assert len(toy_genes) == 10, f"toy gene count = {len(toy_genes)}"

    TOY_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["features_raw", "features_munged", "magma", "out", "utils"]:
        (TOY_DIR / sub).mkdir(exist_ok=True)

    toy_annot = TOY_DIR / "utils" / "gene_annot.txt"
    toy_genes.to_csv(toy_annot, sep="\t", index=False)

    # 5 synthetic features. WHY seed=42: matches PoPS default seed so toy
    # reproducibility is locked.
    rng = np.random.default_rng(42)
    feat_mat = rng.normal(size=(10, 5))
    feat_df = pd.DataFrame(feat_mat, columns=[f"toy.{i}" for i in range(1, 6)])
    feat_df.insert(0, "ENSGID", toy_genes["ENSGID"].values)
    feat_path = TOY_DIR / "features_raw" / "toy_features.txt"
    feat_df.to_csv(feat_path, sep="\t", index=False)

    # Synthetic MAGMA .genes.out. WHY structure: must contain columns GENE,
    # CHR, START, STOP, NSNPS, NPARAM, N, ZSTAT, P exactly as MAGMA emits.
    magma_out_rows = []
    zstats = rng.normal(size=10)
    for i, row in toy_genes.iterrows():
        magma_out_rows.append({
            "GENE": row.ENSGID,
            "CHR": str(row.CHR),
            "START": int(row.START),
            "STOP": int(row.END),
            "NSNPS": 50,
            "NPARAM": 5,
            "N": 50000,
            "ZSTAT": float(zstats[i]),
            "P": 0.1,
        })
    toy_magma_out = TOY_DIR / "magma" / "toy.genes.out"
    pd.DataFrame(magma_out_rows).to_csv(toy_magma_out, sep="\t", index=False)

    # Synthetic MAGMA .genes.raw. PoPS parser expects:
    #   header: "# VERSION = X" then "# COVAR = NSAMP MAC"
    #   body: "GENE CHR START STOP NSNPS NPARAM N MAC ZSTAT [gene_corrs...]"
    # Genes MUST be sequential by chromosome (PoPS asserts this).
    toy_magma_raw = TOY_DIR / "magma" / "toy.genes.raw"
    # Build ENSGID -> index to look up ZSTAT by ENSGID.
    ensgid_to_idx = {row["ENSGID"]: i for i, row in toy_genes.iterrows()}
    with open(toy_magma_raw, "w") as fh:
        fh.write("# VERSION = 110\n")
        fh.write("# COVAR = NSAMP MAC\n")
        # Per-chromosome: first gene has no corrs, subsequent genes have
        # (ind_in_chr) correlations. We emit zeros (independent).
        rows_by_chr = toy_genes.groupby("CHR", sort=False)
        for chr_name, chr_rows in rows_by_chr:
            chr_rows = chr_rows.reset_index(drop=True)
            for j, r in chr_rows.iterrows():
                z = zstats[ensgid_to_idx[r["ENSGID"]]]
                fields = [str(r["ENSGID"]), str(r["CHR"]),
                          str(int(r["START"])), str(int(r["END"])),
                          "50", "5", "50000", "50", f"{z:.6f}"]
                # Add j zero correlations (with preceding genes in this chr).
                fields += ["0.0"] * j
                fh.write(" ".join(fields) + "\n")

    # Control features: none for the toy.
    toy_control = TOY_DIR / "utils" / "features_control.txt"
    toy_control.write_text("")

    # Munge toy features.
    toy_munge_prefix = TOY_DIR / "features_munged" / "toy"
    r = run_pops_subprocess(
        ["python", str(POPS_REPO / "munge_feature_directory.py"),
         "--gene_annot_path", str(toy_annot),
         "--feature_dir", str(TOY_DIR / "features_raw"),
         "--save_prefix", str(toy_munge_prefix),
         "--nan_policy", "zero",
         "--max_cols", "10"],
        cwd=POPS_REPO, logger=logger, timeout_s=120,
    )
    if r.returncode != 0:
        logger.error("Toy munge FAILED")
        return False

    # Run PoPS on toy.
    toy_out_prefix = TOY_DIR / "out" / "toy_pops"
    r = run_pops_subprocess(
        ["python", str(POPS_REPO / "pops.py"),
         "--gene_annot_path", str(toy_annot),
         "--feature_mat_prefix", str(toy_munge_prefix),
         "--num_feature_chunks", "1",
         "--magma_prefix", str(TOY_DIR / "magma" / "toy"),
         "--feature_selection_p_cutoff", "0.99",  # 5 features must survive
         "--feature_selection_keep_hla",
         "--training_keep_hla",
         "--project_out_covariates_keep_hla",
         "--out_prefix", str(toy_out_prefix),
         "--verbose"],
        cwd=POPS_REPO, logger=logger, timeout_s=300,
    )
    if r.returncode != 0:
        logger.error("Toy pops.py FAILED")
        return False

    preds_path = Path(str(toy_out_prefix) + ".preds")
    if not preds_path.exists():
        logger.error(f"Toy preds missing: {preds_path}")
        return False

    preds = pd.read_csv(preds_path, sep="\t")
    if len(preds) != 10:
        logger.error(f"Toy preds has {len(preds)} rows, expected 10")
        return False
    if preds["PoPS_Score"].isna().any():
        logger.error("Toy preds has NaN PoPS_Score")
        return False
    logger.info(f"PHASE 1 PASS: toy preds file has 10 rows, all finite "
                f"(score range {preds.PoPS_Score.min():.4f} to "
                f"{preds.PoPS_Score.max():.4f})")
    return True


# ---------------------------------------------------------------------------
# PHASE 2 — Feature matrix download
# ---------------------------------------------------------------------------

def sha256_file(path: Path, block_size: int = 2 ** 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_with_retries(url: str, dest: Path, logger: logging.Logger,
                          retries: int = 3) -> tuple[bool, dict]:
    """Download url -> dest, with exponential backoff; return (ok, meta)."""
    meta: dict = {"url": url, "dest": str(dest), "attempts": []}
    for attempt in range(retries):
        start = time.time()
        logger.info(f"Downloading {url} attempt {attempt + 1}/{retries}")
        # WHY curl vs requests: streaming multi-GB with resume is trivially
        # handled by curl; we don't need HTTP-library features.
        cmd = ["curl", "-fL", "-C", "-", "-o", str(dest), url,
               "--max-time", "14400",  # 4h per attempt ceiling
               "--retry", "0",  # we manage retries externally
               "--connect-timeout", "60",
               "--speed-limit", "1024", "--speed-time", "120",
               "-s", "-w", "HTTP_STATUS=%{http_code}\nTIME=%{time_total}\n"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        dur = time.time() - start
        status = None
        for line in (r.stdout or "").splitlines():
            if line.startswith("HTTP_STATUS="):
                try:
                    status = int(line.split("=", 1)[1])
                except ValueError:
                    pass
        attempt_info = {
            "attempt": attempt + 1,
            "returncode": r.returncode,
            "http_status": status,
            "duration_s": dur,
            "stderr_tail": (r.stderr or "")[-500:],
        }
        meta["attempts"].append(attempt_info)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            sz = dest.stat().st_size
            logger.info(f"Download OK: {dest} ({sz} bytes in {dur:.0f}s)")
            meta["final_size_bytes"] = sz
            meta["sha256"] = sha256_file(dest)
            return True, meta
        logger.warning(f"Download failed rc={r.returncode} status={status}; "
                       f"backing off")
        time.sleep(5 * (attempt + 1))
    return False, meta


def phase2_download(logger: logging.Logger) -> tuple[bool, dict]:
    """Attempt to acquire the PoPS feature matrix + utility files.

    Per brief: Dropbox -> GCP -> GitHub release. FinucaneLab/pops has no
    GitHub releases (verified via GitHub API 2026-04-23) and no public GCP
    bucket is cited in the README; so only Dropbox is actually an available
    source. We record that fact explicitly rather than faking alternative
    attempts.

    WHY use bundled utils: Dropbox small-file URLs (as of 2026-04-23) return
    an HTML interstitial page, not the underlying file — verified empirically
    in an earlier run. The PoPS repo ships identical utility files
    (gene_annot_jun10.txt, features_jul17_control.txt), so we use those
    rather than re-implement Dropbox's JS interstitial. This is Rule 1
    (don't reinvent) and Rule 0 (don't fake) working together: use the
    real file that the upstream repo ships.
    """
    logger.info("PHASE 2: feature matrix download")
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    sha_log = OUTPUT_DIR / "features_sha256.txt"
    all_meta: dict = {"sources": []}

    # Gene annotation: use bundled PoPS copy (Dropbox interstitials corrupt
    # small-file direct downloads).
    annot_dest = FEATURES_ROOT / "gene_annot_jun10.txt"
    if not annot_dest.exists() or annot_dest.stat().st_size < 100_000:
        src = POPS_REPO / "example" / "data" / "utils" / "gene_annot_jun10.txt"
        shutil.copy2(src, annot_dest)
        logger.info(f"Copied bundled gene_annot: {src} -> {annot_dest} "
                    f"({annot_dest.stat().st_size} bytes)")
        all_meta["sources"].append({"name": "gene_annot",
                                    "method": "bundled_pops_repo",
                                    "src": str(src)})

    # Control features list: use bundled PoPS copy.
    ctrl_dest = FEATURES_ROOT / "features_jul17_control.txt"
    if not ctrl_dest.exists() or ctrl_dest.stat().st_size < 500:
        src = POPS_REPO / "example" / "data" / "utils" / "features_jul17_control.txt"
        shutil.copy2(src, ctrl_dest)
        logger.info(f"Copied bundled control_features: {src} -> {ctrl_dest}")
        all_meta["sources"].append({"name": "control_features",
                                    "method": "bundled_pops_repo",
                                    "src": str(src)})

    # Main feature matrix (7.55 GB compressed).
    features_gz = FEATURES_ROOT / "PoPS.features.txt.gz"
    if not features_gz.exists():
        ok, meta = download_with_retries(DROPBOX_FEATURES_URL, features_gz,
                                          logger, retries=3)
        all_meta["sources"].append({"name": "features_gz", **meta})
        if not ok:
            # Record non-Dropbox alternatives explicitly.
            all_meta["gcp_attempted"] = "N/A — README cites Dropbox only; no public GCP bucket advertised"
            all_meta["github_releases_attempted"] = "N/A — github.com/FinucaneLab/pops has 0 releases (API check 2026-04-23)"
            logger.error("Dropbox features download FAILED; no alternative sources available")
            return False, all_meta
    else:
        logger.info(f"features_gz already present ({features_gz.stat().st_size} bytes)")

    # Checksum log
    with open(sha_log, "w") as fh:
        fh.write(f"# SHA256 of downloaded PoPS features\n")
        for p in [annot_dest, ctrl_dest, features_gz]:
            if p.exists():
                fh.write(f"{sha256_file(p)}  {p.name}  {p.stat().st_size}\n")
    logger.info(f"Wrote {sha_log}")

    # Decompress features. WHY: munge_feature_directory.py needs one TSV file
    # per feature group in a directory. The tarball/gzip unpacks into one big
    # TSV (per Weeks 2023 README); we inspect it post-decompress.
    features_raw_dir = FEATURES_ROOT / "features_raw"
    if not features_raw_dir.exists() or not any(features_raw_dir.iterdir()):
        features_raw_dir.mkdir(exist_ok=True)
        # Peek at the file type
        import gzip
        try:
            with gzip.open(features_gz, "rt") as fh:
                first_line = fh.readline()
                logger.info(f"features_gz first line (truncated 200c): "
                            f"{first_line[:200]}")
        except Exception as e:
            logger.error(f"Could not read features_gz as gzip: {e}")
            return False, all_meta

        # If it's a single TSV, write it to features_raw/ as one file.
        # WHY single file: munge_feature_directory.py iterates glob pattern
        # "feature_dir/*" so one file works; it assigns chunks by max_cols.
        features_tsv = features_raw_dir / "PoPS.features.txt"
        if not features_tsv.exists():
            logger.info(f"Decompressing {features_gz} -> {features_tsv}")
            with gzip.open(features_gz, "rb") as src, open(features_tsv, "wb") as dst:
                shutil.copyfileobj(src, dst, length=2 ** 22)
            logger.info(f"Decompressed size: {features_tsv.stat().st_size} bytes")

    return True, all_meta


# ---------------------------------------------------------------------------
# PHASE 3 — Remap MAGMA + munge + run PoPS
# ---------------------------------------------------------------------------

def remap_magma_to_ensgid(logger: logging.Logger,
                          pops_annot_path: Path,
                          magma_in_prefix: Path,
                          magma_out_prefix: Path) -> dict:
    """Rewrite batch_052_C MAGMA .genes.out and .genes.raw with ENSGID ids.

    WHY symbol-based mapping: our MAGMA used NCBI37 gene.loc which uses
    Entrez IDs. PoPS gene_annot uses ENSGID. The only bridge is the gene
    symbol column (MAGMA symbols.tsv vs PoPS NAME). We filter to genes
    present in PoPS annot, deduplicate 1:1 by symbol (when PoPS annot has
    multiple ENSGIDs for one symbol, pick the first to preserve
    deterministic ordering).
    """
    import pandas as pd

    logger.info("PHASE 3a: remap MAGMA Entrez -> ENSGID by symbol")
    symbols = pd.read_csv(MAGMA_SYMBOLS, sep="\t")
    logger.info(f"MAGMA symbols.tsv rows: {len(symbols)}")

    pops_annot = pd.read_csv(pops_annot_path, sep="\t")
    logger.info(f"PoPS gene_annot rows: {len(pops_annot)}")

    # Symbol -> first ENSGID (deterministic). WHY first: gene_annot_jun10
    # is already deduplicated for the vast majority; ties are rare and
    # arbitrary in either direction.
    sym2ens = (pops_annot[["NAME", "ENSGID", "CHR"]]
               .drop_duplicates(subset="NAME", keep="first")
               .set_index("NAME"))

    symbols = symbols.merge(sym2ens, left_on="symbol", right_index=True,
                            how="inner")
    symbols = symbols.rename(columns={"GENE": "Entrez", "CHR_x": "MAGMA_CHR",
                                      "CHR_y": "POPS_CHR"})
    # Our MAGMA CHR and PoPS CHR should agree; check and drop mismatches.
    symbols["MAGMA_CHR"] = symbols["MAGMA_CHR"].astype(str)
    symbols["POPS_CHR"] = symbols["POPS_CHR"].astype(str)
    n_before = len(symbols)
    symbols = symbols[symbols["MAGMA_CHR"] == symbols["POPS_CHR"]].copy()
    logger.info(f"CHR-agreement filter: {n_before} -> {len(symbols)}")

    # Deduplicate any ENSGID that maps to multiple MAGMA Entrez IDs (rare).
    # WHY: PoPS asserts GENE column unique.
    symbols = symbols.drop_duplicates(subset="ENSGID", keep="first")
    logger.info(f"After ENSGID dedup: {len(symbols)}")

    # Build Entrez -> ENSGID map
    entrez_to_ens = dict(zip(symbols["Entrez"].astype(str),
                             symbols["ENSGID"].astype(str)))
    keep_entrez = set(entrez_to_ens.keys())

    # ---- Rewrite .genes.out ----
    in_out = Path(str(magma_in_prefix) + ".genes.out")
    out_out = Path(str(magma_out_prefix) + ".genes.out")
    with open(in_out) as fh_in, open(out_out, "w") as fh_out:
        header = fh_in.readline()
        fh_out.write(header)
        kept = 0
        dropped = 0
        for line in fh_in:
            parts = line.split()
            if not parts:
                continue
            eid = parts[0]
            if eid in keep_entrez:
                parts[0] = entrez_to_ens[eid]
                fh_out.write("  ".join(parts) + "\n")
                kept += 1
            else:
                dropped += 1
    logger.info(f"genes.out: kept={kept} dropped={dropped}")

    # ---- Rewrite .genes.raw ----
    # Per munge_magma_covariance_metadata: genes must be sequential per
    # chromosome. MAGMA already emits them sequentially. Dropping genes is
    # OK as long as ordering is preserved. Covariance columns refer to
    # preceding genes in the SAME chromosome by position; when we drop a
    # gene we must also drop its correlation entry from subsequent genes'
    # rows. WHY: Sigma matrix is indexed by kept-gene-position, not Entrez.
    in_raw = Path(str(magma_in_prefix) + ".genes.raw")
    out_raw = Path(str(magma_out_prefix) + ".genes.raw")
    with open(in_raw) as fh_in, open(out_raw, "w") as fh_out:
        header1 = fh_in.readline()  # VERSION
        header2 = fh_in.readline()  # COVAR
        fh_out.write(header1)
        fh_out.write(header2)

        # Track per-chromosome keep/drop mask so we can prune covariance
        # columns correctly.
        curr_chr = None
        chr_keep_mask: list[bool] = []  # True for each gene encountered in curr_chr
        for line in fh_in:
            parts = line.strip("\n").split(" ")
            if not parts or parts == [""]:
                continue
            eid, chr_ = parts[0], parts[1]
            if chr_ != curr_chr:
                curr_chr = chr_
                chr_keep_mask = []
            keep = eid in keep_entrez
            # Covariance columns are positions 9..end (0-indexed: indices 9+)
            # and refer to genes in this chr at positions
            # [pos - n_corrs, pos - 1]. We drop corrs referring to
            # previously-dropped genes.
            n_fixed = 9
            gene_corrs = parts[n_fixed:] if len(parts) > n_fixed else []
            n_corrs = len(gene_corrs)
            # The i-th gene has up to i preceding-gene corrs. gene_corrs[k]
            # refers to gene at chr-position (len(chr_keep_mask) - n_corrs + k).
            new_corrs = []
            base = len(chr_keep_mask) - n_corrs
            for k, c in enumerate(gene_corrs):
                ref_pos = base + k
                if 0 <= ref_pos < len(chr_keep_mask) and chr_keep_mask[ref_pos]:
                    new_corrs.append(c)
            chr_keep_mask.append(keep)
            if not keep:
                continue
            # Rewrite parts[0] to ENSGID, replace correlations.
            new_parts = [entrez_to_ens[eid]] + parts[1:n_fixed] + new_corrs
            fh_out.write(" ".join(new_parts) + "\n")

    logger.info(f"Wrote {out_out} and {out_raw}")

    # Sanity: CHR ordering preserved?
    import pandas as pd
    df = pd.read_csv(out_out, delim_whitespace=True)
    logger.info(f"Remapped genes.out rows: {len(df)}")
    # Check CHR sequential (per PoPS assert)
    chrs = df["CHR"].astype(str).tolist()
    breaks = sum(1 for i in range(len(chrs) - 1) if chrs[i] != chrs[i + 1])
    if breaks != len(set(chrs)) - 1:
        logger.error(f"CHR not sequential: breaks={breaks} unique={len(set(chrs))}")
    return {"kept": len(df), "dropped_no_symbol_match": n_before - len(symbols),
            "entrez_to_ens_size": len(entrez_to_ens)}


def phase3_execute(logger: logging.Logger) -> bool:
    """Munge features + run pops.py on remapped MAGMA."""
    logger.info("PHASE 3: execute PoPS on real PGC3 MAGMA")

    pops_annot = FEATURES_ROOT / "gene_annot_jun10.txt"
    ctrl = FEATURES_ROOT / "features_jul17_control.txt"
    features_raw_dir = FEATURES_ROOT / "features_raw"
    features_munged_prefix = FEATURES_ROOT / "features_munged" / "pops_features"
    (FEATURES_ROOT / "features_munged").mkdir(exist_ok=True)

    for p in [pops_annot, ctrl, features_raw_dir]:
        if not p.exists():
            logger.error(f"Missing input for Phase 3: {p}")
            return False

    # Remap MAGMA
    magma_remapped = OUTPUT_DIR / "PGC3_EUR_gene_ENSGID"
    stats = remap_magma_to_ensgid(logger, pops_annot, MAGMA_PREFIX_IN,
                                  magma_remapped)
    (OUTPUT_DIR / "magma_remap_stats.json").write_text(
        json.dumps(stats, indent=2))

    # Munge features (unless already munged)
    existing_chunks = sorted((FEATURES_ROOT / "features_munged").glob("pops_features.mat.*.npy"))
    if not existing_chunks:
        logger.info("Munging feature directory (may take minutes; ~15 GB input)")
        r = run_pops_subprocess(
            ["python", str(POPS_REPO / "munge_feature_directory.py"),
             "--gene_annot_path", str(pops_annot),
             "--feature_dir", str(features_raw_dir),
             "--save_prefix", str(features_munged_prefix),
             "--nan_policy", "mean",  # WHY mean: Weeks 2023 default impute
             "--max_cols", "5000"],
            cwd=POPS_REPO, logger=logger, timeout_s=7200,
        )
        if r.returncode != 0:
            logger.error("munge_feature_directory FAILED")
            return False
        existing_chunks = sorted((FEATURES_ROOT / "features_munged").glob("pops_features.mat.*.npy"))
    num_chunks = len(existing_chunks)
    logger.info(f"Num feature chunks: {num_chunks}")

    # Run PoPS
    out_prefix = OUTPUT_DIR / "PGC3_EUR_PoPS"
    r = run_pops_subprocess(
        ["python", str(POPS_REPO / "pops.py"),
         "--gene_annot_path", str(pops_annot),
         "--feature_mat_prefix", str(features_munged_prefix),
         "--num_feature_chunks", str(num_chunks),
         "--magma_prefix", str(magma_remapped),
         "--control_features_path", str(ctrl),
         "--feature_selection_p_cutoff", "0.001",
         "--out_prefix", str(out_prefix),
         "--verbose"],
        cwd=POPS_REPO, logger=logger, timeout_s=14400,
    )
    if r.returncode != 0:
        logger.error("pops.py main run FAILED")
        return False
    preds_path = Path(str(out_prefix) + ".preds")
    if not preds_path.exists():
        logger.error(f"Preds missing: {preds_path}")
        return False
    logger.info(f"PHASE 3 PASS: {preds_path}")
    return True


# ---------------------------------------------------------------------------
# PHASE 4 — Analysis
# ---------------------------------------------------------------------------

def phase4_analysis(logger: logging.Logger) -> dict:
    """Compute precision@K, AUPRC, Spearman rho, MHC stratification, etc.

    Primary metric (per critic C2 revision): precision@K and AUPRC with
    PGC3 Prioritised (n=120) as positives.
    """
    logger.info("PHASE 4: analysis")
    import pandas as pd
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score

    # Load PoPS preds
    preds = pd.read_csv(OUTPUT_DIR / "PGC3_EUR_PoPS.preds", sep="\t")
    logger.info(f"PoPS preds rows: {len(preds)}")
    preds = preds.dropna(subset=["PoPS_Score"]).copy()
    logger.info(f"Finite PoPS scores: {len(preds)}")

    # Load gene_annot for CHR/NAME lookup
    annot = pd.read_csv(FEATURES_ROOT / "gene_annot_jun10.txt", sep="\t")
    preds = preds.merge(annot[["ENSGID", "NAME", "CHR", "TSS"]], on="ENSGID", how="left")

    # Load original MAGMA (pre-remap) for MAGMA_Z and P — we keep the remapped
    # one (which has ENSGID matching PoPS preds).
    magma = pd.read_csv(str(OUTPUT_DIR / "PGC3_EUR_gene_ENSGID") + ".genes.out",
                        delim_whitespace=True)
    magma = magma.rename(columns={"GENE": "ENSGID", "ZSTAT": "MAGMA_Z", "P": "MAGMA_P"})
    preds = preds.merge(magma[["ENSGID", "MAGMA_Z", "MAGMA_P"]], on="ENSGID", how="left")

    # ---- Build PGC3 Prioritised gene list ----
    # Brief §6: Prioritised column in ST12, combined with protein_coding filter
    # as the "Published Prioritised" set of n=120 (all biotypes). We report
    # both versions (n=120 all-biotype and n=106 protein-coding) but use
    # protein_coding as the primary positive class because PoPS features
    # are protein-coding biased.
    st12 = pd.read_excel(ST12_XLSX, sheet_name="ST12 all criteria")
    prior_all = st12[st12["Prioritised"] == 1].copy()
    prior_pc = prior_all[prior_all["gene_biotype"] == "protein_coding"].copy()
    logger.info(f"Prioritised (all biotypes): {len(prior_all)}")
    logger.info(f"Prioritised (protein_coding): {len(prior_pc)}")

    prior_symbols = set(prior_all["Symbol.ID"].astype(str))
    prior_pc_symbols = set(prior_pc["Symbol.ID"].astype(str))
    prior_ensgids = set(prior_all["Ensembl.ID"].astype(str))
    prior_pc_ensgids = set(prior_pc["Ensembl.ID"].astype(str))

    # Positive class membership (by ENSGID primary, symbol fallback)
    preds["is_prior"] = preds["ENSGID"].isin(prior_ensgids) | preds["NAME"].isin(prior_symbols)
    preds["is_prior_pc"] = preds["ENSGID"].isin(prior_pc_ensgids) | preds["NAME"].isin(prior_pc_symbols)
    n_pos = int(preds["is_prior"].sum())
    n_pos_pc = int(preds["is_prior_pc"].sum())
    logger.info(f"Positives in preds (all biotypes): {n_pos} / {len(prior_all)}")
    logger.info(f"Positives in preds (protein_coding): {n_pos_pc} / {len(prior_pc)}")

    # ---- Sort by PoPS score ----
    preds_sorted = preds.sort_values("PoPS_Score", ascending=False).reset_index(drop=True)
    N_total = len(preds_sorted)

    # ---- precision@K (primary) ----
    precisions: dict = {}
    for positives_label, pos_mask in [("all_prioritised", preds_sorted["is_prior"]),
                                      ("pc_prioritised", preds_sorted["is_prior_pc"])]:
        precisions[positives_label] = {}
        n_positives = int(pos_mask.sum())
        for K in [50, 100, 200]:
            topK = pos_mask.iloc[:K]
            hits = int(topK.sum())
            p_at_k = hits / K
            baseline = n_positives / N_total
            precisions[positives_label][f"P@{K}"] = {
                "hits": hits, "K": K, "precision": p_at_k,
                "baseline": baseline, "lift": p_at_k / baseline if baseline else None,
            }
            logger.info(f"P@{K} ({positives_label}): {hits}/{K} = {p_at_k:.4f} "
                        f"(baseline {baseline:.4f}, lift {p_at_k / baseline if baseline else 0:.1f}x)")

    # ---- AUPRC ----
    auprc_all = float(average_precision_score(preds_sorted["is_prior"].values,
                                              preds_sorted["PoPS_Score"].values))
    auprc_pc = float(average_precision_score(preds_sorted["is_prior_pc"].values,
                                              preds_sorted["PoPS_Score"].values))
    logger.info(f"AUPRC (all): {auprc_all:.4f}; AUPRC (protein_coding): {auprc_pc:.4f}")

    # ---- Spearman rho with bootstrap 95% CI ----
    both = preds_sorted.dropna(subset=["MAGMA_Z"])
    pops_vals = both["PoPS_Score"].values
    magma_vals = both["MAGMA_Z"].values
    rho, rho_p = spearmanr(pops_vals, magma_vals)
    rho = float(rho)
    logger.info(f"Spearman rho(PoPS, MAGMA_Z): {rho:.4f} (p={rho_p:.3g}, "
                f"n={len(both)})")

    # Bootstrap 1000 iters
    rng = np.random.default_rng(42)
    boot_rhos = []
    n = len(both)
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        b_rho, _ = spearmanr(pops_vals[idx], magma_vals[idx])
        if not np.isnan(b_rho):
            boot_rhos.append(float(b_rho))
    rho_lo, rho_hi = (float(np.percentile(boot_rhos, 2.5)),
                     float(np.percentile(boot_rhos, 97.5)))
    logger.info(f"Bootstrap 95% CI: [{rho_lo:.4f}, {rho_hi:.4f}]")

    # ---- MHC stratification ----
    # chr 6, 25-34 Mb by TSS. WHY TSS: annot has TSS field; MHC is defined
    # by genomic window containing the gene.
    preds_sorted["is_mhc"] = (
        (preds_sorted["CHR"].astype(str) == MHC_CHR)
        & (preds_sorted["TSS"] >= MHC_START)
        & (preds_sorted["TSS"] <= MHC_END)
    )
    mhc_results = {}
    for K in [50, 100, 200]:
        topK = preds_sorted.iloc[:K]
        mhc_count = int(topK["is_mhc"].sum())
        non_mhc = topK[~topK["is_mhc"]]
        non_mhc_hits = int(non_mhc["is_prior"].sum())
        mhc_results[f"K={K}"] = {
            "mhc_top_K_count": mhc_count,
            "non_mhc_top_K_count": K - mhc_count,
            "non_mhc_prioritised_hits": non_mhc_hits,
            "non_mhc_precision": non_mhc_hits / max(K - mhc_count, 1),
        }
        logger.info(f"MHC strat K={K}: MHC_in_topK={mhc_count}, "
                    f"non_MHC_hits={non_mhc_hits}/{K - mhc_count}")

    # ---- SCHEMA / SynGO_EDT1 percentiles (descriptive) ----
    SCHEMA = ["SETD1A", "CUL1", "XPO7", "TRIO", "CACNA1G", "SP4", "GRIA3",
              "GRIN2A", "HERC1", "RB1CC1"]
    SYNGO_EDT1 = ['DLGAP1', 'GRIN2A', 'NRXN1', 'CNTNAP2', 'ARC', 'DLG4',
                  'NRXN2', 'NLGN1', 'NLGN2', 'SHANK1', 'SHANK3', 'HOMER1',
                  'SYN1', 'GAP43']
    preds_sorted["rank"] = np.arange(1, len(preds_sorted) + 1)
    preds_sorted["pctile"] = 1.0 - (preds_sorted["rank"] - 1) / len(preds_sorted)
    schema_pctiles = preds_sorted[preds_sorted["NAME"].isin(SCHEMA)]["pctile"].tolist()
    syngo_pctiles = preds_sorted[preds_sorted["NAME"].isin(SYNGO_EDT1)]["pctile"].tolist()
    logger.info(f"SCHEMA median pctile: {float(np.median(schema_pctiles)) if schema_pctiles else 'NA'} (n={len(schema_pctiles)})")
    logger.info(f"SynGO_EDT1 median pctile: {float(np.median(syngo_pctiles)) if syngo_pctiles else 'NA'} (n={len(syngo_pctiles)})")

    # ---- Novel top-100 (not in Prioritised) ----
    top100 = preds_sorted.iloc[:100]
    novel = top100[~top100["is_prior"]][["ENSGID", "NAME", "CHR", "TSS",
                                         "PoPS_Score", "MAGMA_Z", "MAGMA_P"]]
    novel_list = novel.to_dict(orient="records")

    # ---- Write pops_scores.tsv per brief Engineering Constraint §6 ----
    out = preds[["NAME", "ENSGID", "CHR", "PoPS_Score", "MAGMA_Z", "MAGMA_P"]].copy()
    out.columns = ["gene_symbol", "gene_id", "chrom", "pops_score", "magma_z", "magma_p"]
    out = out.sort_values("pops_score", ascending=False)
    out_path = OUTPUT_DIR / "pops_scores.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Wrote {out_path} ({len(out)} rows)")

    # ---- top100_comparison.json ----
    primary_K = 100
    primary_p = precisions["all_prioritised"][f"P@{primary_K}"]
    comparison = {
        "K": primary_K,
        "n_total": N_total,
        "n_prioritised_total": n_pos,
        "n_prioritised_protein_coding": n_pos_pc,
        "precision_at_K": precisions,
        "auprc_all": auprc_all,
        "auprc_protein_coding": auprc_pc,
        "spearman_rho_PoPS_vs_MAGMA_Z": {
            "rho": rho, "p": float(rho_p),
            "ci_95": [rho_lo, rho_hi], "n_bootstrap": len(boot_rhos),
            "n_genes": n,
        },
        "mhc_stratified": mhc_results,
        "schema_median_pctile": float(np.median(schema_pctiles)) if schema_pctiles else None,
        "schema_n": len(schema_pctiles),
        "syngo_edt1_median_pctile": float(np.median(syngo_pctiles)) if syngo_pctiles else None,
        "syngo_edt1_n": len(syngo_pctiles),
        "novel_top100": novel_list,
    }
    (OUTPUT_DIR / "top100_comparison.json").write_text(json.dumps(comparison, indent=2,
                                                                  default=str))
    logger.info("Wrote top100_comparison.json")

    return comparison


# ---------------------------------------------------------------------------
# DRIVER + challenge.md
# ---------------------------------------------------------------------------

def classify_outcome(comparison: dict) -> str:
    """Per brief §DECISION RULE."""
    primary_prec = comparison["precision_at_K"]["all_prioritised"]["P@100"]["precision"]
    rho = comparison["spearman_rho_PoPS_vs_MAGMA_Z"]["rho"]
    non_mhc_100 = comparison["mhc_stratified"]["K=100"]["non_mhc_precision"]
    if primary_prec >= 0.10 and 0.55 <= rho <= 0.80 and non_mhc_100 >= 0.05:
        return "PASS"
    if primary_prec < 0.05 or rho < 0.30 or rho > 0.95:
        return "UNINTERPRETABLE"
    # Everything else → still report but flag
    return "UNINTERPRETABLE"


def write_challenge_md(phase_status: dict, comparison: dict | None,
                       blocked_reason: str | None, t_start: float) -> None:
    wall = time.time() - t_start
    path = BATCH_DIR / "challenge.md"
    lines = ["# batch_053_B — challenge.md", ""]
    lines.append(f"**Wall time**: {wall / 60:.1f} min")
    lines.append("")
    lines.append("## Phase status")
    for k, v in phase_status.items():
        lines.append(f"- {k}: **{v}**")
    lines.append("")
    if blocked_reason:
        lines.append("## REPORTED_TO_PI — Blocking cause")
        lines.append("")
        lines.append(blocked_reason)
        lines.append("")
    if comparison:
        prec_all = comparison["precision_at_K"]["all_prioritised"]
        prec_pc = comparison["precision_at_K"]["pc_prioritised"]
        rho = comparison["spearman_rho_PoPS_vs_MAGMA_Z"]
        mhc = comparison["mhc_stratified"]["K=100"]
        cls = classify_outcome(comparison)
        lines += [
            "## Primary metrics",
            f"- **Precision@100 (all Prioritised)**: {prec_all['P@100']['hits']}/100 = {prec_all['P@100']['precision']:.4f} (baseline {prec_all['P@100']['baseline']:.4f})",
            f"- **Precision@100 (protein_coding Prioritised)**: {prec_pc['P@100']['hits']}/100 = {prec_pc['P@100']['precision']:.4f}",
            f"- **Precision@50 (all)**: {prec_all['P@50']['hits']}/50 = {prec_all['P@50']['precision']:.4f}",
            f"- **Precision@200 (all)**: {prec_all['P@200']['hits']}/200 = {prec_all['P@200']['precision']:.4f}",
            f"- **AUPRC (all)**: {comparison['auprc_all']:.4f}",
            f"- **AUPRC (pc)**: {comparison['auprc_protein_coding']:.4f}",
            f"- **Spearman ρ(PoPS, MAGMA_Z)**: {rho['rho']:.4f} (95% CI [{rho['ci_95'][0]:.4f}, {rho['ci_95'][1]:.4f}]; n={rho['n_genes']})",
            "",
            "## MHC stratification (K=100)",
            f"- MHC-in-top100: {mhc['mhc_top_K_count']}",
            f"- Non-MHC-top100: {mhc['non_mhc_top_K_count']}",
            f"- Non-MHC Prioritised hits: {mhc['non_mhc_prioritised_hits']}",
            f"- Non-MHC precision: {mhc['non_mhc_precision']:.4f}",
            "",
            "## Descriptive gene-set percentiles (no CI)",
            f"- SCHEMA median percentile (n={comparison['schema_n']}): {comparison['schema_median_pctile']}",
            f"- SynGO_EDT1 median percentile (n={comparison['syngo_edt1_n']}): {comparison['syngo_edt1_median_pctile']}",
            "",
            "## Novel top-100 (SPECULATIVE, no biological claim)",
            f"See `output/top100_comparison.json` key `novel_top100` (n={len(comparison['novel_top100'])}).",
            "",
            f"## Classification (per brief §DECISION RULE): **{cls}**",
            "",
        ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def write_blocked_requests(reason: str) -> None:
    path = PROJECT_ROOT / "docs" / "blocked_requests.md"
    # Append rather than overwrite; preserve prior entries.
    existing = path.read_text() if path.exists() else "# Blocked Requests\n\n"
    entry = "\n---\n\n## 2026-04-23 batch_053_B PoPS PI item 11 BLOCKED\n\n" + reason + "\n"
    path.write_text(existing + entry)
    print(f"Updated {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-smoke", action="store_true",
                        help="skip Phase 1 toy test (only use if already verified)")
    parser.add_argument("--skip-download", action="store_true",
                        help="skip Phase 2 (data already on disk)")
    parser.add_argument("--skip-run", action="store_true",
                        help="skip Phase 3 (run already complete)")
    parser.add_argument("--only-analysis", action="store_true",
                        help="only run Phase 4")
    args = parser.parse_args()

    t_start = time.time()
    logger = setup_logging()
    logger.info(f"batch_053_B runner started at {time.ctime()}")

    phase_status = {"Phase1": "NOT_RUN", "Phase2": "NOT_RUN",
                    "Phase3": "NOT_RUN", "Phase4": "NOT_RUN"}
    comparison = None
    blocked_reason = None

    try:
        if args.only_analysis:
            comparison = phase4_analysis(logger)
            phase_status["Phase4"] = "PASS"
        else:
            if not args.skip_smoke:
                ok = phase1_smoke_test(logger)
                phase_status["Phase1"] = "PASS" if ok else "FAIL"
                if not ok:
                    blocked_reason = ("Phase 1 smoke test failed. PoPS install "
                                      "or toy-input execution blocked. "
                                      "See output/pops_install.log for details.")
                    raise SystemExit(1)
            else:
                phase_status["Phase1"] = "SKIPPED"

            if not args.skip_download:
                ok, meta = phase2_download(logger)
                (OUTPUT_DIR / "download_meta.json").write_text(
                    json.dumps(meta, indent=2, default=str))
                phase_status["Phase2"] = "PASS" if ok else "FAIL"
                if not ok:
                    blocked_reason = ("Phase 2 feature download failed. See "
                                      "`output/download_meta.json` for HTTP "
                                      "codes, retry attempts, and URLs tried.\n\n"
                                      "URLs tried:\n"
                                      f"  1. Dropbox: {DROPBOX_FEATURES_URL}\n"
                                      "  2. GCP: not advertised in README\n"
                                      "  3. GitHub release: 0 releases in "
                                      "FinucaneLab/pops per GitHub API")
                    raise SystemExit(2)
            else:
                phase_status["Phase2"] = "SKIPPED"

            if not args.skip_run:
                ok = phase3_execute(logger)
                phase_status["Phase3"] = "PASS" if ok else "FAIL"
                if not ok:
                    blocked_reason = ("Phase 3 PoPS execution failed. See "
                                      "output/pops_install.log for the full "
                                      "pops.py stderr trace.")
                    raise SystemExit(3)
            else:
                phase_status["Phase3"] = "SKIPPED"

            # Phase 4 only if Phase 3 produced a preds file.
            preds_path = OUTPUT_DIR / "PGC3_EUR_PoPS.preds"
            if preds_path.exists():
                comparison = phase4_analysis(logger)
                phase_status["Phase4"] = "PASS"
            else:
                phase_status["Phase4"] = "NOT_RUN"

    except SystemExit as e:
        logger.error(f"Halted with SystemExit {e.code}")
    except Exception as e:
        logger.exception("Uncaught exception in driver")
        blocked_reason = blocked_reason or f"Uncaught exception: {e}"

    if blocked_reason:
        write_blocked_requests(blocked_reason)
    write_challenge_md(phase_status, comparison, blocked_reason, t_start)


if __name__ == "__main__":
    main()
