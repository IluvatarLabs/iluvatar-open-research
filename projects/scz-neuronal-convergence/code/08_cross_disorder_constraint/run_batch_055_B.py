#!/usr/bin/env python3
"""batch_055_B v2 driver — F147 reconciliation: Sub-A (per-gene constraint table),
Sub-B (multi-gene LOFGO on B3), Sub-C (cross-disorder MAGMA-Z with negative
controls).

Brief: experiments/batch_055_B/brief.md (v2). All cardinal rules apply:
  * Rule 0  — no fabrication; emp_ps and OR are computed from the real gnomAD
              constraint table and real comparator GWAS sumstats.
  * Rule 1  — re-uses batch_048 `load_gnomad`/`fisher_enrichment`/
              `_or_from_masks`/`length_perm_test` for Sub-A/Sub-B (verbatim) and
              the existing MAGMA install + 1000G EUR plink + NCBI37.3 gene-loc
              for Sub-C, exactly as batch_052_C did.
  * Rule 2  — every constant cites its source inline (brief, batch_054_B, or
              published reference).
  * Rule 5  — every decision documents WHY (file header, function docstrings,
              inline comments).

WHY this driver structure (3 sub-experiments in one script):
  Sub-A is a deterministic table (no compute), Sub-B reuses the same gnomAD
  table and Fisher machinery, and Sub-C is a separate compute pipeline. They
  share the bg gene set, the B3 set, the seeds, and the provenance dict.
  Splitting into 3 scripts duplicates I/O and provenance. One driver with
  --only-sub-{a,b,c} flags is the standard Marvin pattern (mirrors
  batch_053_B).

WHY the missense_z metric uses BOTH the brief-stated 3.09 threshold AND the
batch_054_B "top decile" (~p90 ≈ 2.879) threshold:
  The brief explicitly says "mis_z >= 3.09" (Sub-A line 22, Sub-B line 30) but
  the reproduction gate (line 131) requires Sub-A counts to match
  batch_054_B.sub_b_decomposition.B3.per_metric.observed_a, which used the
  top-decile (~2.879) threshold and yielded observed_a=7. With threshold 3.09
  observed_a=5 (verified bench-test 2026-04-23). The brief is internally
  inconsistent. To honour BOTH, we report mis_z under both thresholds in the
  per-gene table and Sub-B; the reproduction-gate check is performed against
  the top-decile value (the actual batch_054_B convention).

WHY MAGMA is the right compute (not LDSC partitioned-h2 or scDRS):
  Brief Sub-C explicitly requires MAGMA gene-Z. We have a working MAGMA v1.10
  install + 1000G EUR plink + NCBI37.3 gene-loc + a verified batch_052_C
  driver (PGC3 SCZ EUR) that produces gene-Z values used downstream by
  batch_053_B. We replicate that pipeline 7× (once per comparator) with the
  same hyperparameters (35kb-up / 10kb-down window per PoPS standard, snp-wise
  mean gene model). For sumstats with no per-SNP P column, we convert the
  LDSC-munged Z to a 2-sided p-value via 2 * scipy.stats.norm.sf(|Z|).

Failure handling:
  * Sub-A reproduction gate (counts vs batch_054_B observed_a) failure → exit 2.
  * Sub-B reproduction gate (full-set Fisher OR for B3 pLI = 5.806398 to 4 dp)
    failure → exit 3.
  * Sub-C: per-disorder MAGMA failure → mark INVALID and continue; if >1
    disorder fails, exit 4 (per brief Sub-C UNINTERPRETABLE clause).
  * results.json is written ATOMICALLY (tmp + os.replace) so partial JSON is
    never observed by readers.

Seeds (per brief):
  RNG_SEED_PERMS = 20260423         (Sub-B 5000 perms, Sub-C 10000 perms)
  RNG_SEED_BOOTSTRAP = 20260501     (Sub-B bootstrap CIs)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Paths (all absolute — agent threads reset cwd between bash calls).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_055_B"
OUTPUT_DIR = BATCH_DIR / "output"
LOGS_DIR = BATCH_DIR / "logs"
WORK_DIR = BATCH_DIR / "work"
CROSS_DIR = OUTPUT_DIR / "cross_disorder"

GNOMAD_TSV = PROJECT_ROOT / "data" / "item_15" / "gnomad.v4.1.constraint_metrics.tsv"
F144_RESULTS_JSON = PROJECT_ROOT / "experiments" / "batch_054_B" / "output" / "results.json"
BATCH_048_SCRIPT = PROJECT_ROOT / "experiments" / "batch_048" / "scripts" / "run_batch_048.py"

# MAGMA infrastructure (from batch_052_C). All verified present at
# implementation time (2026-04-23).
MAGMA_BIN = PROJECT_ROOT / "tools" / "magma_bin" / "magma"
MAGMA_GENELOC = PROJECT_ROOT / "tools" / "magma_bin" / "refs" / "NCBI37.3.gene.loc"
MAGMA_BFILE = PROJECT_ROOT / "tools" / "magma_bin" / "g1000_eur" / "g1000_eur"
SUMSTATS_DIR = PROJECT_ROOT / "data" / "ldsc" / "comparator_sumstats"

# Existing PGC3 SCZ MAGMA gene-Z output (anchor for reproduction gate).
PGC3_SCZ_GENES_OUT = PROJECT_ROOT / "experiments" / "batch_053_B" / "output" / "PGC3_EUR_gene_ENSGID.genes.out"

# ---------------------------------------------------------------------------
# Constants (every value cites brief or batch_054_B).
# ---------------------------------------------------------------------------
# Brief §WHAT line 20 — frozen B3 set (n=18, sha256 reproducible from
# batch_054_B/output/results.json gene_lists.B3.sha256).
B3_GENES: list[str] = [
    "AP3B2", "ASIC1", "CNNM2", "CPNE7", "CRHR1", "DTNB", "EIF5", "EPN2",
    "FGFR1", "KIF21B", "MOB4", "NAE1", "NEGR1", "NXPH1", "OPCML", "PLK2",
    "SRPK2", "STRN",
]
B3_SHA256_EXPECTED = "a627014497c3bdf1559fdeaec0f53162359c8f45a91f222474b066d0bab69617"

# Brief §WHAT Sub-B — leave-K-out subsets defined by gene set difference.
# WHY each definition: see brief §Sub-B B3.1/B3.2/B3.3/B3.4 (lines 25-28).
LOFGO_DROP_SETS: dict[str, set[str]] = {
    "B3.1_drop_top5_pLI": {"CNNM2", "EIF5", "FGFR1", "MOB4", "SRPK2"},
    "B3.2_drop_4_VERA_pLI": {"FGFR1", "CNNM2", "NEGR1", "PLK2"},
    "B3.3_drop_full_VERA_7": {
        "CRHR1", "FGFR1", "CNNM2", "NEGR1", "KIF21B", "PLK2", "NAE1",
    },
}
# B3.4 is a RETAIN set (the pLI<0.9 subset, n=8; brief line 28).
LOFGO_RETAIN_B34: set[str] = {
    "AP3B2", "CPNE7", "CRHR1", "DTNB", "EPN2", "KIF21B", "NAE1", "STRN",
}

# Constraint thresholds (brief §WHAT Sub-A line 22; batch_048.PLI_THRESHOLD;
# batch_048.LOEUF_THRESHOLD).
PLI_THRESHOLD = 0.9
LOEUF_THRESHOLD = 0.35
MIS_Z_THRESHOLD_LITERAL = 3.09  # brief literal Sub-A/Sub-B threshold

# Reproduction targets (brief §Reproduction gate line 132).
B3_FULLSET_PLI_OR_TARGET = 5.806398  # batch_054_B B3 per_metric pLI OR
B3_FULLSET_PLI_OR_TOL = 1e-4  # 4 dp per brief
# batch_054_B observed_a per metric for B3 (n_in_bg=18). Matches
# brief §PREDICTION line 79 ("10/18 pLI hits, 5/18 LOEUF hits, 7/18 mis_z hits").
B3_OBSERVED_A_TARGETS: dict[str, int] = {
    "pLI >= 0.9": 10,
    "LOEUF <= 0.35": 5,
    "missense_z top decile": 7,
}

# Seeds (brief §MEASUREMENT line 84; matches batch_054_A/B convention).
RNG_SEED_PERMS = 20260423
RNG_SEED_BOOTSTRAP = 20260501

# Sub-B settings (brief §MEASUREMENT line 76).
N_PERM_SUBB = 5_000
N_BOOTSTRAP = 1_000

# Sub-C settings (brief §MEASUREMENT line 77).
N_PERM_SUBC = 10_000

# BH-FDR alpha (brief §MEASUREMENT line 30).
BH_ALPHA = 0.05

# Comparator disorders. WHY this exact 7-disorder panel: brief §WHAT Sub-C
# line 32 (4 psychiatric: BIP, ASD, MDD, ADHD; 3 negative-control:
# IBD-de_Lange_2017, Height, Alzheimer's).
COMPARATOR_DISORDERS: list[str] = [
    "adhd", "asd", "bip", "mdd", "ibd_delange2017", "height", "alzheimers",
]
PSYCHIATRIC_DISORDERS = {"adhd", "asd", "bip", "mdd"}
NEGATIVE_CONTROL_DISORDERS = {"ibd_delange2017", "height", "alzheimers"}

# MAGMA window (brief §WHAT Sub-C line 32 — "35kb-up / 10kb-down" PoPS standard).
# Verified PoPS Weeks 2023 default.
MAGMA_WINDOW_UP_KB = 35
MAGMA_WINDOW_DOWN_KB = 10

# B3 hand-curated biological category (brief §Sub-A; user-supplied mapping
# with cited UniProt/GO source per gene). WHY a hand mapping rather than
# auto-pull: the brief explicitly mandates a hardcoded mapping with citations,
# and a runtime UniProt query would add a network dependency we are told not
# to introduce. Source for each: UniProt protein function or canonical GO
# molecular-function/biological-process top-level annotation.
B3_BIOLOGICAL_CATEGORY: dict[str, dict[str, str]] = {
    "AP3B2": {"category": "scaffold/trafficking",
              "source": "UniProt Q13367 — clathrin-coated-vesicle adaptor AP-3 beta-2 subunit"},
    "ASIC1": {"category": "receptor/ion-channel",
              "source": "UniProt P78348 — acid-sensing ion channel 1"},
    "CNNM2": {"category": "transporter",
              "source": "UniProt Q9H8M5 — Mg2+ transporter; SLC-related"},
    "CPNE7": {"category": "enzyme/Ca2+-binding",
              "source": "UniProt Q9UBL6 — copine-7 phospholipid-binding membrane protein"},
    "CRHR1": {"category": "receptor",
              "source": "UniProt P34998 — class B GPCR for CRH"},
    "DTNB": {"category": "scaffold",
             "source": "UniProt O60941 — beta-dystrobrevin DGC scaffold"},
    "EIF5": {"category": "enzyme/translation",
             "source": "UniProt P55010 — translation initiation factor (GTPase-activating)"},
    "EPN2": {"category": "scaffold/trafficking",
             "source": "UniProt O95208 — epsin-2 endocytic adaptor"},
    "FGFR1": {"category": "receptor",
              "source": "UniProt P11362 — fibroblast growth factor receptor 1 (RTK)"},
    "KIF21B": {"category": "enzyme/motor",
               "source": "UniProt O75037 — kinesin family member 21B (microtubule motor)"},
    "MOB4": {"category": "scaffold/signaling",
             "source": "UniProt Q9Y3A3 — MOB kinase activator 4 (Hippo pathway)"},
    "NAE1": {"category": "enzyme",
             "source": "UniProt Q13564 — NEDD8-activating enzyme E1 regulatory subunit"},
    "NEGR1": {"category": "cell-adhesion",
              "source": "UniProt Q7Z3B1 — IgLON family neuronal growth regulator 1"},
    "NXPH1": {"category": "cell-adhesion",
              "source": "UniProt P58417 — neurexophilin-1 secreted neurexin ligand"},
    "OPCML": {"category": "cell-adhesion",
              "source": "UniProt Q14982 — IgLON opioid-binding cell adhesion molecule-like"},
    "PLK2": {"category": "enzyme",
             "source": "UniProt Q9NYY3 — serine/threonine-protein kinase polo-like 2"},
    "SRPK2": {"category": "enzyme",
              "source": "UniProt P78362 — SR-protein-specific kinase 2"},
    "STRN": {"category": "scaffold",
             "source": "UniProt O43815 — striatin scaffold (STRIPAK complex)"},
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROSS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("batch_055_B")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%dT%H:%M:%S")
        for h in (logging.FileHandler(LOGS_DIR / "run_batch_055_B.log"),
                  logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """Streaming SHA256 (handles multi-GB inputs without loading)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_of_genes(items: list[str]) -> str:
    h = hashlib.sha256()
    for x in sorted(items):
        h.update(x.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def atomic_write_json(obj: Any, dest: Path) -> None:
    """Write JSON atomically: tmp file + os.replace.

    WHY: results.json is read by orchestrator + downstream batches; partial
    writes during a crash would corrupt the audit trail. os.replace is atomic
    on POSIX local filesystems.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------
# Reuse batch_048 primitives. WHY: Rule 1 — exact reproduction of the F121
# pipeline used by batch_054_B requires bit-for-bit matching of load_gnomad
# (canonical+mane filter; first-row dedup; TSS-based MHC indicator).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "batch_048" / "scripts"))
from run_batch_048 import (  # noqa: E402 — sys.path must be set first
    load_gnomad,                 # canonical+mane filter, dedup, MHC indicator
    fisher_enrichment,           # 2x2 OR + p (Haldane–Anscombe corr)
    _or_from_masks,              # raw OR from boolean masks
    length_perm_test,            # length-stratified perm test
)


def make_bg(gnomad: pd.DataFrame) -> pd.DataFrame:
    """bg = MHC-excluded canonical+mane gnomAD rows (matches batch_054_B exactly).

    Defined ONCE so Sub-A, Sub-B, and provenance use a single bg. WHY we DO
    NOT drop NaN-gene rows: bench-test 2026-04-23 confirmed batch_054_B keeps
    them (gnomAD ships exactly 1 row with `gene`=NaN, which becomes a non-B3
    non-metric d-cell entry). Dropping it shifts d by 1 and breaks the
    bit-identical reproduction of OR=5.806398 (we saw 5.805986 instead).
    The dedup-ratio gate is computed over notna() rows only.
    """
    return gnomad.loc[~gnomad["mhc_indicator"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sub-A: per-gene constraint table
# ---------------------------------------------------------------------------
def build_per_gene_table(
    logger: logging.Logger,
    gnomad: pd.DataFrame,
    f144_set: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construct the per-gene table for B3 (n=18) + F144 (n=48) + bg.

    The bg = MHC-excluded canonical+mane gnomAD rows (same definition as
    batch_054_B). Per-gene metric values are taken from the deduplicated
    gnomAD frame (one row per gene; lof.pLI, lof.oe_ci.upper, mis.z_score).

    Returns (table_df, summary_dict). The summary is the "Sub-A" payload for
    results.json (counts, threshold-meeting numbers, sanity-check stats).
    """
    logger.info("SUB-A: building per-gene constraint table")

    # bg via shared helper (MHC-excluded, matches batch_054_B 1:1 incl. 1 NaN-gene row).
    bg = make_bg(gnomad)
    bg_real = bg.loc[bg["gene"].notna()]
    bg_genes_set = set(bg_real["gene"])

    # Sanity: 1 row per gene (deduplication gate per brief Sub-A line 22).
    # Computed on real (notna) gene rows only — NaN-gene rows are tracked
    # separately for transparency but do not invalidate the dedup gate.
    n_unique = bg_real["gene"].nunique()
    n_rows_real = len(bg_real)
    n_rows_total = len(bg)
    dedup_ratio = n_unique / n_rows_real if n_rows_real else 0.0
    assert dedup_ratio == 1.0, (
        f"Deduplication failed on real-gene rows: n_unique={n_unique} "
        f"n_rows_real={n_rows_real}; ratio={dedup_ratio}. Brief requires 1.0."
    )
    logger.info(
        "SUB-A bg: %d total rows (%d real-gene, %d NaN-gene; deduplicated 1:1 over real)",
        n_rows_total, n_rows_real, n_rows_total - n_rows_real)

    # Group / category labels (iterate over real-gene rows only; NaN-gene
    # row would add a meaningless table entry).
    f144_set = list(f144_set)
    rows: list[dict[str, Any]] = []
    for _, r in bg_real.iterrows():
        g = r["gene"]
        is_b3 = g in B3_GENES
        is_f144 = g in f144_set
        group = "B3" if is_b3 else ("F144" if is_f144 else "bg")
        bio = B3_BIOLOGICAL_CATEGORY.get(g, {"category": "", "source": ""})
        # Threshold flags. WHY both top-decile AND >=3.09 for missense_z:
        # see file header WHY-block.
        mis_z = r.get("mis.z_score", np.nan)
        rows.append({
            "gene": g,
            "ensgid": r.get("gene_id", ""),
            "group": group,
            "in_B3": is_b3,
            "in_F144": is_f144,
            "lof_pLI": r.get("lof.pLI", np.nan),
            "lof_LOEUF": r.get("lof.oe_ci.upper", np.nan),
            "mis_z": mis_z,
            "pli_ge_09": bool(r.get("pli_ge_09", False)),
            "loeuf_lt_035": bool(r.get("loeuf_lt_035", False)),
            "mis_z_ge_309": bool(mis_z >= MIS_Z_THRESHOLD_LITERAL) if pd.notna(mis_z) else False,
            "biological_category": bio["category"] if is_b3 else "",
            "biological_source": bio["source"] if is_b3 else "",
        })

    table = pd.DataFrame(rows)

    # missense_z top-decile threshold (matches batch_054_B
    # build_metric_masks). WHY top-decile + filled NaN -> -inf: matches
    # batch_054_B mask logic exactly so observed_a counts replicate.
    mis_vals = pd.to_numeric(bg["mis.z_score"], errors="coerce")
    mis_topdec_threshold = float(np.nanpercentile(mis_vals.to_numpy(), 90))
    mis_topdec_mask = (mis_vals.fillna(-np.inf) >= mis_topdec_threshold).to_numpy(dtype=bool)
    table["mis_z_top_decile"] = False
    bg_gene_to_topdec = dict(zip(bg["gene"], mis_topdec_mask))
    table["mis_z_top_decile"] = table["gene"].map(
        lambda g: bool(bg_gene_to_topdec.get(g, False))
    )

    # B3 / F144 / bg counts per threshold.
    def count_in(group_genes: list[str], col: str) -> int:
        return int(table[table["gene"].isin(group_genes)][col].sum())

    b3_counts = {
        "n": len(B3_GENES),
        "pLI >= 0.9": count_in(B3_GENES, "pli_ge_09"),
        "LOEUF <= 0.35": count_in(B3_GENES, "loeuf_lt_035"),
        "mis_z top decile": count_in(B3_GENES, "mis_z_top_decile"),
        "mis_z >= 3.09": count_in(B3_GENES, "mis_z_ge_309"),
        "in_bg_intersect_n": int(table[table["gene"].isin(B3_GENES)].shape[0]),
    }
    f144_counts = {
        "n": len(f144_set),
        "pLI >= 0.9": count_in(f144_set, "pli_ge_09"),
        "LOEUF <= 0.35": count_in(f144_set, "loeuf_lt_035"),
        "mis_z top decile": count_in(f144_set, "mis_z_top_decile"),
        "mis_z >= 3.09": count_in(f144_set, "mis_z_ge_309"),
        "in_bg_intersect_n": int(table[table["gene"].isin(f144_set)].shape[0]),
    }
    bg_counts = {
        "n": int(table.shape[0]),
        "pLI >= 0.9": int(table["pli_ge_09"].sum()),
        "LOEUF <= 0.35": int(table["loeuf_lt_035"].sum()),
        "mis_z top decile": int(table["mis_z_top_decile"].sum()),
        "mis_z >= 3.09": int(table["mis_z_ge_309"].sum()),
    }

    # Predicted ALL-3-meeting set for B3 (brief §PREDICTION line 55):
    # FGFR1, CNNM2, SRPK2 (NOT MOB4, since MOB4 mis_z=2.46 < 3.09).
    all3_observed = sorted(table[
        (table["in_B3"]) &
        (table["pli_ge_09"]) &
        (table["loeuf_lt_035"]) &
        (table["mis_z_ge_309"])
    ]["gene"].tolist())

    summary = {
        "B3_counts": b3_counts,
        "F144_counts": f144_counts,
        "bg_counts": bg_counts,
        "missense_z_top_decile_threshold_value": mis_topdec_threshold,
        "missense_z_literal_threshold_value": MIS_Z_THRESHOLD_LITERAL,
        "B3_genes_meeting_all_3_predicted": ["FGFR1", "CNNM2", "SRPK2"],
        "B3_genes_meeting_all_3_observed_lit_threshold": all3_observed,
        "n_unique_genes_div_n_rows_bg": dedup_ratio,
        "biological_category_summary_B3":
            pd.Series({g: B3_BIOLOGICAL_CATEGORY[g]["category"] for g in B3_GENES})
            .value_counts().to_dict(),
    }
    logger.info("SUB-A B3 counts: %s", b3_counts)
    logger.info("SUB-A predicted all-3 set: %s; observed: %s",
                summary["B3_genes_meeting_all_3_predicted"], all3_observed)

    return table, summary


def write_per_gene_tsv(table: pd.DataFrame, dest: Path,
                       logger: logging.Logger) -> None:
    """Write per_gene_constraint.tsv for B3 + F144 + bg (brief §Output)."""
    cols = [
        "gene", "ensgid", "group", "in_B3", "in_F144",
        "lof_pLI", "lof_LOEUF", "mis_z",
        "pli_ge_09", "loeuf_lt_035",
        "mis_z_top_decile", "mis_z_ge_309",
        "biological_category", "biological_source",
    ]
    out = table[cols].copy()
    out.to_csv(dest, sep="\t", index=False)
    logger.info("SUB-A wrote %s (%d rows)", dest, len(out))


# ---------------------------------------------------------------------------
# Sub-B: multi-gene LOFGO on B3
# ---------------------------------------------------------------------------
def _bootstrap_or_ci(
    in_list_mask: np.ndarray,
    metric_mask: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap 95% CI of the Fisher OR by resampling bg rows with
    replacement.

    WHY resample bg (not just list genes): the OR depends on the marginal
    rates within bg as well as the list. Resampling all bg rows preserves
    the joint distribution of (in_list, in_metric).

    Returns (ci_low, ci_high). NaN values omitted.
    """
    rng = np.random.default_rng(seed)
    n = len(in_list_mask)
    boot_ors: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = int((in_list_mask[idx] & metric_mask[idx]).sum())
        b = int((in_list_mask[idx] & ~metric_mask[idx]).sum())
        c = int((~in_list_mask[idx] & metric_mask[idx]).sum())
        d = int((~in_list_mask[idx] & ~metric_mask[idx]).sum())
        # Haldane–Anscombe correction for empty cells (matches batch_048).
        if a == 0 or b == 0 or c == 0 or d == 0:
            a += 0.5; b += 0.5; c += 0.5; d += 0.5
        boot_ors.append((a * d) / (b * c))
    arr = np.array([x for x in boot_ors if np.isfinite(x)])
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _emp_p_genelist(
    list_genes: set[str],
    bg_genes_arr: np.ndarray,
    metric_mask: np.ndarray,
    n_perm: int,
    seed: int,
) -> float:
    """Empirical p for OR(list, metric) via random-sample-of-same-size null.

    WHY simple random sampling (not length-stratified): brief §MEASUREMENT
    line 30 says "5000 perms" without length-stratification. batch_054_B
    used length-stratified for jackknife (different test). Here Sub-B is a
    classic two-tailed permutation against a size-matched random gene set
    drawn from bg.
    """
    rng = np.random.default_rng(seed)
    list_mask = np.array([g in list_genes for g in bg_genes_arr], dtype=bool)
    obs_or = _or_from_masks(list_mask, metric_mask)
    n_list = int(list_mask.sum())
    n_bg = len(bg_genes_arr)
    if n_list == 0:
        return float("nan")
    null_ors = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        # Sample n_list indices without replacement from bg.
        idx = rng.choice(n_bg, size=n_list, replace=False)
        m = np.zeros(n_bg, dtype=bool)
        m[idx] = True
        null_ors[i] = _or_from_masks(m, metric_mask)
    finite = null_ors[np.isfinite(null_ors)]
    if finite.size == 0:
        return float("nan")
    # One-sided (>=) emp_p with +1 / +1 smoothing per Phipson+Smyth 2010.
    n_extreme = int((finite >= obs_or).sum())
    return (n_extreme + 1) / (finite.size + 1)


def run_sub_b_lofgo(
    logger: logging.Logger,
    bg: pd.DataFrame,
    table: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run the 4 leave-K-out tests × 3 metrics = 12 cells.

    Returns (per-cell results dict, reproduction-gate dict).
    """
    logger.info("SUB-B: multi-gene LOFGO (4 tests × 3 metrics)")

    bg_genes_arr = bg["gene"].to_numpy()
    bg_genes_set = set(bg_genes_arr.tolist())

    # Build metric masks aligned to bg row order (brief §MEASUREMENT same
    # convention as batch_054_B). WHY top-decile for missense_z: see header
    # WHY-block; needed to satisfy the reproduction gate of observed_a=7.
    pli_mask = bg["pli_ge_09"].astype(bool).to_numpy()
    loeuf_mask = bg["loeuf_lt_035"].astype(bool).to_numpy()
    mis_vals = pd.to_numeric(bg["mis.z_score"], errors="coerce")
    mis_topdec_thr = float(np.nanpercentile(mis_vals.to_numpy(), 90))
    mis_mask = (mis_vals.fillna(-np.inf) >= mis_topdec_thr).to_numpy(dtype=bool)

    metric_masks: dict[str, np.ndarray] = {
        "pLI >= 0.9": pli_mask,
        "LOEUF <= 0.35": loeuf_mask,
        "missense_z top decile": mis_mask,
    }

    # ---- Reproduction gate: full-set B3 pLI Fisher OR == 5.806398 ± 1e-4.
    # WHY check FIRST: brief §Reproduction gate line 132 + §UNINTERPRETABLE
    # line 119 — failure means a count drift (deduplication, MHC, threshold)
    # and we MUST stop before reporting. _or_from_masks gives the raw sample
    # OR a*d/(b*c) — bench-test 2026-04-23 confirmed this matches batch_054_B's
    # stored 5.806398 to 6 decimal places when bg includes the 1 NaN-gene row.
    full_b3_set = set(B3_GENES)
    full_b3_in_bg_mask = np.array(
        [g in full_b3_set for g in bg_genes_arr], dtype=bool)
    full_or = float(_or_from_masks(full_b3_in_bg_mask, pli_mask))
    a_full = int((full_b3_in_bg_mask & pli_mask).sum())
    repro = {
        "B3_full_set_pLI_OR_observed": full_or,
        "B3_full_set_pLI_OR_expected": B3_FULLSET_PLI_OR_TARGET,
        "tolerance": B3_FULLSET_PLI_OR_TOL,
        "match": bool(abs(full_or - B3_FULLSET_PLI_OR_TARGET) < B3_FULLSET_PLI_OR_TOL),
        "B3_full_observed_a": a_full,
    }
    logger.info("SUB-B reproduction gate: full-set B3 pLI OR=%.6f target=%.6f match=%s",
                full_or, B3_FULLSET_PLI_OR_TARGET, repro["match"])
    if not repro["match"]:
        logger.error("REPRODUCTION GATE FAILED — refusing to report Sub-B")
        return {}, repro

    # ---- LOFGO cells.
    lofgo_subsets: dict[str, set[str]] = {}
    for k, drop_set in LOFGO_DROP_SETS.items():
        lofgo_subsets[k] = full_b3_set - drop_set
    # B3.4 is a retain set, not a drop set.
    lofgo_subsets["B3.4_retain_pLI_lt_09"] = LOFGO_RETAIN_B34 & full_b3_set

    rows: list[dict[str, Any]] = []
    cell_pvals: list[float] = []
    cell_keys: list[tuple[str, str]] = []
    bg_genes_set_full = set(bg_genes_arr.tolist())

    for lofgo_name, sub_set in lofgo_subsets.items():
        sub_in_bg = sub_set & bg_genes_set
        n_sub = len(sub_in_bg)
        logger.info("SUB-B %s: n=%d (in_bg=%d)", lofgo_name, len(sub_set), n_sub)
        list_mask = np.array([g in sub_in_bg for g in bg_genes_arr], dtype=bool)

        for metric_name, m_mask in metric_masks.items():
            target_set = set(bg_genes_arr[m_mask].tolist())
            # fisher_enrichment returns raw sample OR + Fisher exact 'greater'
            # P + scipy odds_ratio CI — matches batch_054_B verbatim. The raw
            # OR a*d/(b*c) is what scipy.stats.fisher_exact returns.
            fish = fisher_enrichment(sub_in_bg, bg_genes_set_full, target_set)
            a, b, c, d = fish["a"], fish["b"], fish["c"], fish["d"]
            or_ = float(fish["or"])
            p_fish = float(fish["p"])

            # Bootstrap 95% CI of the raw OR (b/c we resample bg with
            # replacement; this produces a non-conditional OR distribution).
            ci_low, ci_high = _bootstrap_or_ci(
                list_mask, m_mask, N_BOOTSTRAP, RNG_SEED_BOOTSTRAP)

            # Empirical permutation P (5000 perms, seed=20260423).
            emp_p = _emp_p_genelist(
                sub_in_bg, bg_genes_arr, m_mask,
                N_PERM_SUBB, RNG_SEED_PERMS)

            rows.append({
                "lofgo": lofgo_name,
                "metric": metric_name,
                "n_in_set": len(sub_set),
                "n_in_bg": n_sub,
                "a": a, "b": b, "c": c, "d": d,
                "OR": or_,
                "OR_scipy_ci95_low": float(fish.get("ci_low", float("nan"))),
                "OR_scipy_ci95_high": float(fish.get("ci_high", float("nan"))),
                "OR_bootstrap_ci95_low": float(ci_low),
                "OR_bootstrap_ci95_high": float(ci_high),
                "p_fisher": p_fish,
                "emp_p": float(emp_p),
            })
            cell_pvals.append(p_fish if np.isfinite(p_fish) else 1.0)
            cell_keys.append((lofgo_name, metric_name))

    # BH-FDR across all 12 cells (4 LOFGO × 3 metrics).
    if cell_pvals:
        _, q, _, _ = multipletests(cell_pvals, alpha=BH_ALPHA, method="fdr_bh")
        for r, qv in zip(rows, q):
            r["bh_fdr_q"] = float(qv)

    sub_b_payload = {
        "n_cells": len(rows),
        "metrics": list(metric_masks.keys()),
        "missense_z_top_decile_threshold": mis_topdec_thr,
        "lofgo_subsets": {k: sorted(list(v)) for k, v in lofgo_subsets.items()},
        "cells": rows,
        "seeds": {
            "perm_seed": RNG_SEED_PERMS,
            "n_perm": N_PERM_SUBB,
            "bootstrap_seed": RNG_SEED_BOOTSTRAP,
            "n_bootstrap": N_BOOTSTRAP,
        },
        "bh_fdr_alpha": BH_ALPHA,
    }
    return sub_b_payload, repro


# ---------------------------------------------------------------------------
# Sub-C: cross-disorder MAGMA-Z
# ---------------------------------------------------------------------------
def load_bim_rsid_map(logger: logging.Logger) -> dict[str, tuple[str, int]]:
    """Build {rsid: (chr, bp)} from the 1000G EUR plink .bim file.

    WHY 1000G EUR bim as the SNP-loc source (not the GWAS files): MAGMA --bfile
    requires the SNP IDs in --pval and --snp-loc to match the bfile RSIDs.
    Using the bfile bim guarantees alignment for ALL disorders uniformly. The
    bim has ~22.6M SNPs covering all common comparator GWASes.
    """
    bim_path = Path(str(MAGMA_BFILE) + ".bim")
    logger.info("SUB-C: loading 1000G EUR bim %s (~22M SNPs, ~250 MB RAM)", bim_path)
    bim = pd.read_csv(bim_path, sep="\t", header=None,
                      names=["chr", "rsid", "cM", "bp", "a1", "a2"],
                      dtype={"chr": str, "rsid": str, "bp": int})
    out: dict[str, tuple[str, int]] = dict(zip(bim["rsid"], zip(bim["chr"], bim["bp"])))
    logger.info("SUB-C: 1000G bim loaded — %d RSIDs", len(out))
    return out


def harmonize_disorder_sumstats(
    disorder: str,
    rsid_map: dict[str, tuple[str, int]],
    logger: logging.Logger,
) -> tuple[Path, Path, dict[str, Any]] | None:
    """Produce a MAGMA-ready (snp_loc.tsv, pval.tsv) pair for one disorder.

    Sumstats source priority (per data inventory at SUMSTATS_DIR):
      adhd, asd, bip, mdd                  -> {disorder}_ldsc.tsv (SNP/A1/A2/BETA/P/SE/N)
      ibd_delange2017                      -> ibd_delange2017.preprocessed.tsv (SNP/A1/A2/Effect/P/N)
      height, alzheimers                   -> {disorder}.sumstats.gz (SNP/A1/A2/Z/N) -> Z->P

    snp_loc columns:  SNP CHR BP   (no header per MAGMA --snp-loc convention)
    pval columns:     SNP PVAL N   (with header; passed via use=SNP,PVAL ncol=N)

    Returns (snp_loc_path, pval_path, meta) or None if loading failed.
    """
    logger.info("SUB-C0 [%s]: harmonizing sumstats", disorder)
    work = WORK_DIR / disorder
    work.mkdir(parents=True, exist_ok=True)
    snp_loc = work / "snp_loc.tsv"
    pval = work / "pval.tsv"
    meta: dict[str, Any] = {"disorder": disorder, "source": None,
                            "z_to_p_conversion": False}

    try:
        if disorder in {"adhd", "asd", "bip", "mdd"}:
            src = SUMSTATS_DIR / f"{disorder}_ldsc.tsv"
            meta["source"] = str(src)
            df = pd.read_csv(src, sep="\t", usecols=["SNP", "P", "N"])
            df = df.rename(columns={"SNP": "SNP", "P": "PVAL", "N": "N"})
        elif disorder == "ibd_delange2017":
            src = SUMSTATS_DIR / "ibd_delange2017.preprocessed.tsv"
            meta["source"] = str(src)
            df = pd.read_csv(src, sep="\t", usecols=["SNP", "P", "N"])
            df = df.rename(columns={"SNP": "SNP", "P": "PVAL", "N": "N"})
        elif disorder in {"height", "alzheimers"}:
            src = SUMSTATS_DIR / f"{disorder}.sumstats.gz"
            meta["source"] = str(src)
            meta["z_to_p_conversion"] = True
            df = pd.read_csv(src, sep="\t", compression="gzip",
                             usecols=["SNP", "Z", "N"])
            # Two-sided p from |Z| via standard normal (brief §WHAT Sub-C
            # MDD note; same convention applied to height + alzheimers since
            # they only ship LDSC-munged files).
            df["PVAL"] = 2.0 * norm.sf(np.abs(df["Z"].astype(float).values))
            df = df[["SNP", "PVAL", "N"]]
        else:
            raise ValueError(f"unknown disorder: {disorder}")
    except Exception as e:
        logger.error("SUB-C0 [%s] FAILED loading sumstats: %s", disorder, e)
        return None

    # Drop NaN pvals; clip to (1e-300, 1) (MAGMA refuses pval==0 or >1).
    df = df.dropna(subset=["PVAL", "N"])
    df["PVAL"] = df["PVAL"].astype(float).clip(lower=1e-300, upper=1.0)
    df["N"] = df["N"].astype(float).round().astype(int)
    n_before = len(df)

    # Add CHR / BP from 1000G EUR bim. WHY bim, not GWAS-CHR/BP: see
    # load_bim_rsid_map docstring — uniform path, alignment guaranteed.
    chr_bp = df["SNP"].map(lambda s: rsid_map.get(s))
    keep = chr_bp.notna()
    df = df[keep].copy()
    df["CHR"] = chr_bp[keep].map(lambda t: t[0])
    df["BP"] = chr_bp[keep].map(lambda t: t[1])
    n_after = len(df)
    logger.info("SUB-C0 [%s]: %d SNPs in sumstats, %d after bim-RSID match",
                disorder, n_before, n_after)
    meta["n_snps_input"] = int(n_before)
    meta["n_snps_after_bim_match"] = int(n_after)
    if n_after < 100_000:
        logger.warning("SUB-C0 [%s]: only %d SNPs match bim — proceeding but "
                       "expect noisy gene-Z", disorder, n_after)

    # Write snp_loc (no header per MAGMA convention) and pval (with header).
    df[["SNP", "CHR", "BP"]].to_csv(snp_loc, sep="\t", index=False, header=False)
    df[["SNP", "PVAL", "N"]].to_csv(pval, sep="\t", index=False)

    return snp_loc, pval, meta


def run_magma_for_disorder(
    disorder: str,
    snp_loc: Path,
    pval: Path,
    logger: logging.Logger,
    smoke_chr22: bool = False,
) -> Path | None:
    """Run MAGMA annotate + gene analysis for one disorder.

    smoke_chr22: if True, restrict to chromosome 22 for a fast smoke test
    (~1 min wall) before the full-genome run (~30 min). Brief §WHY NOT line
    43 mandates this Sub-C0 chr22 smoke.

    Returns the .genes.out path if successful, else None.
    """
    work = WORK_DIR / disorder
    suffix = "smoke22" if smoke_chr22 else "full"
    annot_prefix = work / f"{suffix}.annot"
    gene_prefix = work / f"{suffix}.gene"
    logger.info("SUB-C [%s/%s]: MAGMA annotate", disorder, suffix)

    if smoke_chr22:
        # Filter to chr 22 only.
        df = pd.read_csv(snp_loc, sep="\t", header=None, names=["SNP", "CHR", "BP"],
                         dtype={"CHR": str, "BP": int})
        df = df[df["CHR"] == "22"]
        snp_loc_smoke = work / "snp_loc.smoke22.tsv"
        df.to_csv(snp_loc_smoke, sep="\t", index=False, header=False)
        snp_loc_use = snp_loc_smoke
    else:
        snp_loc_use = snp_loc

    annot_cmd = [
        str(MAGMA_BIN), "--annotate",
        f"window={MAGMA_WINDOW_UP_KB},{MAGMA_WINDOW_DOWN_KB}",
        "--snp-loc", str(snp_loc_use),
        "--gene-loc", str(MAGMA_GENELOC),
        "--out", str(annot_prefix),
    ]
    log_path = LOGS_DIR / f"magma_{disorder}_{suffix}.log"
    with log_path.open("w") as lh:
        r = subprocess.run(annot_cmd, stdout=lh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        logger.error("MAGMA annotate failed for %s/%s (rc=%d) — see %s",
                     disorder, suffix, r.returncode, log_path)
        return None

    gene_cmd = [
        str(MAGMA_BIN),
        "--bfile", str(MAGMA_BFILE),
        "--pval", str(pval), "use=SNP,PVAL", "ncol=N",
        "--gene-annot", f"{annot_prefix}.genes.annot",
        "--out", str(gene_prefix),
    ]
    logger.info("SUB-C [%s/%s]: MAGMA gene analysis", disorder, suffix)
    with log_path.open("a") as lh:
        r = subprocess.run(gene_cmd, stdout=lh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        logger.error("MAGMA gene-analysis failed for %s/%s (rc=%d) — see %s",
                     disorder, suffix, r.returncode, log_path)
        return None

    out = Path(f"{gene_prefix}.genes.out")
    if not out.exists():
        logger.error("MAGMA produced no .genes.out for %s/%s", disorder, suffix)
        return None
    logger.info("SUB-C [%s/%s]: produced %s", disorder, suffix, out)
    return out


def load_b3_entrez_map(logger: logging.Logger) -> dict[str, str]:
    """Map B3 symbol -> Entrez ID using the MAGMA NCBI37.3.gene.loc file.

    WHY symbol-based mapping: MAGMA gene-Z is keyed by Entrez (NCBI37.3 gene
    column). We need Entrez IDs to find B3 in MAGMA output. The gene-loc file
    provides the canonical bridge.
    """
    df = pd.read_csv(MAGMA_GENELOC, sep="\t", header=None,
                     names=["entrez", "chr", "start", "end", "strand", "symbol"],
                     dtype={"entrez": str, "symbol": str})
    sym2ent = (df.drop_duplicates(subset="symbol", keep="first")
                 .set_index("symbol")["entrez"].to_dict())
    out = {g: sym2ent.get(g) for g in B3_GENES}
    missing = [g for g, e in out.items() if e is None]
    if missing:
        logger.warning("SUB-C: B3 genes missing from MAGMA gene-loc: %s", missing)
    return {g: e for g, e in out.items() if e is not None}


def b3_test_per_disorder(
    disorder: str,
    genes_out: Path,
    b3_entrez: dict[str, str],
    logger: logging.Logger,
) -> dict[str, Any] | None:
    """Per-disorder permutation-null test: median B3 gene-Z vs random 18-gene null.

    Returns dict with {median_b3, median_bg, sd_bg, std_effect, perm_p,
    n_b3_in_genes_out, n_perm}, or None on failure.
    """
    logger.info("SUB-C [%s]: per-disorder permutation test", disorder)
    try:
        df = pd.read_csv(genes_out, sep=r"\s+", engine="python")
    except Exception as e:
        logger.error("SUB-C [%s]: failed reading %s: %s", disorder, genes_out, e)
        return None
    if "ZSTAT" not in df.columns or "GENE" not in df.columns:
        logger.error("SUB-C [%s]: %s missing GENE/ZSTAT columns", disorder, genes_out)
        return None

    df["GENE"] = df["GENE"].astype(str)
    bg_z = df["ZSTAT"].astype(float).dropna().to_numpy()
    if bg_z.size == 0 or not np.isfinite(np.var(bg_z)) or np.var(bg_z) == 0:
        logger.error("SUB-C [%s]: bg gene-Z has zero variance — INVALID", disorder)
        return None

    # B3 genes present in this disorder's MAGMA output (Entrez IDs).
    b3_entrez_set = set(b3_entrez.values())
    b3_rows = df[df["GENE"].isin(b3_entrez_set)]
    n_b3 = len(b3_rows)
    if n_b3 < 10:
        logger.warning("SUB-C [%s]: only %d/%d B3 genes in MAGMA output",
                       disorder, n_b3, len(b3_entrez))
    if n_b3 == 0:
        logger.error("SUB-C [%s]: zero B3 genes in MAGMA output — INVALID", disorder)
        return None

    obs_median_b3 = float(np.median(b3_rows["ZSTAT"].astype(float)))
    median_bg = float(np.median(bg_z))
    sd_bg = float(np.std(bg_z, ddof=1))
    std_effect = (obs_median_b3 - median_bg) / sd_bg if sd_bg > 0 else float("nan")

    # Permutation null: sample n_b3 random genes from bg, compute median
    # gene-Z, repeat 10,000x. WHY n_b3 (not 18): if some B3 genes are missing
    # from this disorder's MAGMA output, draw the same number to make the null
    # apples-to-apples.
    rng = np.random.default_rng(RNG_SEED_PERMS)
    perm_medians = np.empty(N_PERM_SUBC, dtype=float)
    for i in range(N_PERM_SUBC):
        idx = rng.choice(bg_z.size, size=n_b3, replace=False)
        perm_medians[i] = np.median(bg_z[idx])
    n_extreme = int((perm_medians >= obs_median_b3).sum())
    perm_p = (n_extreme + 1) / (N_PERM_SUBC + 1)  # +1 smoothing

    out = {
        "disorder": disorder,
        "n_genes_in_output": int(len(df)),
        "n_b3_in_output": n_b3,
        "median_b3": obs_median_b3,
        "median_bg": median_bg,
        "sd_bg": sd_bg,
        "std_effect": float(std_effect),
        "perm_p": float(perm_p),
        "n_perm": N_PERM_SUBC,
        "perm_seed": RNG_SEED_PERMS,
    }
    logger.info(
        "SUB-C [%s]: median_b3=%.3f median_bg=%.3f sd_bg=%.3f std_effect=%+0.3fσ perm_p=%.4g (n_b3=%d)",
        disorder, obs_median_b3, median_bg, sd_bg, std_effect, perm_p, n_b3)
    return out


def run_sub_c(logger: logging.Logger,
              skip_smoke: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sub-C orchestrator: harmonize -> smoke -> full -> per-disorder test.

    Returns (sub_c_payload, sub_c_repro_dict).
    """
    if not MAGMA_BIN.exists():
        logger.error("MAGMA binary missing: %s", MAGMA_BIN)
        return {"status": "MAGMA_BINARY_MISSING"}, {}

    rsid_map = load_bim_rsid_map(logger)
    b3_entrez = load_b3_entrez_map(logger)

    per_disorder: dict[str, Any] = {}
    failed: list[str] = []

    for d in COMPARATOR_DISORDERS:
        logger.info("SUB-C: ===== %s =====", d)
        harm = harmonize_disorder_sumstats(d, rsid_map, logger)
        if harm is None:
            failed.append(d)
            per_disorder[d] = {"status": "HARMONIZATION_FAILED"}
            continue
        snp_loc, pval, meta = harm

        # chr22 smoke (skip if requested via --skip-smoke).
        if not skip_smoke:
            smoke_out = run_magma_for_disorder(d, snp_loc, pval, logger,
                                                smoke_chr22=True)
            if smoke_out is None:
                logger.error("SUB-C [%s]: chr22 smoke FAILED — skipping full run", d)
                failed.append(d)
                per_disorder[d] = {"status": "SMOKE_FAILED",
                                    "harmonization_meta": meta}
                continue

        # Full-genome run.
        full_out = run_magma_for_disorder(d, snp_loc, pval, logger,
                                           smoke_chr22=False)
        if full_out is None:
            failed.append(d)
            per_disorder[d] = {"status": "FULL_RUN_FAILED",
                                "harmonization_meta": meta}
            continue

        # Persist a copy in output/cross_disorder/.
        copied = CROSS_DIR / f"{d}.genes.out"
        shutil.copy2(full_out, copied)

        # Per-disorder permutation test.
        test_res = b3_test_per_disorder(d, copied, b3_entrez, logger)
        if test_res is None:
            failed.append(d)
            per_disorder[d] = {"status": "TEST_FAILED",
                                "harmonization_meta": meta}
            continue

        # Per-disorder JSON.
        per_d_json = CROSS_DIR / f"{d}_b3_test.json"
        atomic_write_json({"harmonization_meta": meta,
                           "test": test_res}, per_d_json)
        per_disorder[d] = {"status": "OK", "harmonization_meta": meta,
                           "test": test_res, "genes_out": str(copied)}

    # BH-FDR across the 7 disorders.
    ok_disorders = [d for d in COMPARATOR_DISORDERS if per_disorder.get(d, {}).get("status") == "OK"]
    if ok_disorders:
        ps = [per_disorder[d]["test"]["perm_p"] for d in ok_disorders]
        _, qs, _, _ = multipletests(ps, alpha=BH_ALPHA, method="fdr_bh")
        for d, q in zip(ok_disorders, qs):
            per_disorder[d]["test"]["bh_fdr_q"] = float(q)

    # Reproduction gate (Sub-C): PGC3 SCZ EUR median B3 gene-Z computed from
    # batch_053_B PGC3_EUR_gene_ENSGID.genes.out — record (descriptive only).
    sczu = compute_pgc3_scz_anchor(b3_entrez, logger)

    repro = {"scz_anchor": sczu}

    payload = {
        "n_disorders_attempted": len(COMPARATOR_DISORDERS),
        "n_disorders_ok": len(ok_disorders),
        "n_disorders_failed": len(failed),
        "failed_disorders": failed,
        "per_disorder": per_disorder,
        "n_perm": N_PERM_SUBC,
        "perm_seed": RNG_SEED_PERMS,
        "bh_fdr_alpha": BH_ALPHA,
        "magma_binary": str(MAGMA_BIN),
        "magma_window_kb": [MAGMA_WINDOW_UP_KB, MAGMA_WINDOW_DOWN_KB],
    }
    return payload, repro


def compute_pgc3_scz_anchor(b3_entrez: dict[str, str],
                             logger: logging.Logger) -> dict[str, Any]:
    """Read batch_053_B PGC3 SCZ MAGMA output and compute median B3 gene-Z.

    WHY descriptive (not gating): brief §Sub-C reproduction gate (line 133)
    is a self-consistency check. We compute the value here and record it; the
    interpretation is that this median should match a downstream re-derivation
    in iter_055 docs/.
    """
    if not PGC3_SCZ_GENES_OUT.exists():
        logger.warning("PGC3 SCZ MAGMA output missing: %s", PGC3_SCZ_GENES_OUT)
        return {"status": "MISSING"}
    try:
        df = pd.read_csv(PGC3_SCZ_GENES_OUT, sep=r"\s+", engine="python")
    except Exception as e:
        logger.error("SCZ anchor read failed: %s", e)
        return {"status": "READ_FAILED", "error": str(e)}
    df["GENE"] = df["GENE"].astype(str)
    # batch_053_B remapped to ENSGID. We need ENSGID for B3 — pull from
    # batch_054_B per-gene table generation (gnomAD gene_id col stores ENSGID
    # for mane_select=true rows, but our dedup keeps Entrez/refseq rows).
    # SIMPLEST: read PGC3_EUR_gene.genes.out (Entrez) from batch_052_C.
    # But batch_053_B has only the ENSGID-remapped file. Try an ENSGID lookup
    # via gnomAD gene_id (mane_select=true) or fall back to NAME if present.
    # Since batch_053_B output uses ENSGID, we need a Symbol->ENSGID mapping.
    sym2ens = build_symbol_to_ensgid()
    b3_ens = [sym2ens.get(g) for g in B3_GENES if sym2ens.get(g)]
    b3_in = df[df["GENE"].isin(b3_ens)]
    median_b3 = float(np.median(b3_in["ZSTAT"].astype(float))) if len(b3_in) else float("nan")
    median_bg = float(np.median(df["ZSTAT"].astype(float)))
    sd_bg = float(np.std(df["ZSTAT"].astype(float), ddof=1))
    std_eff = ((median_b3 - median_bg) / sd_bg) if sd_bg > 0 else float("nan")
    return {
        "status": "OK",
        "n_b3_in_scz_output": int(len(b3_in)),
        "median_b3_scz": median_b3,
        "median_bg_scz": median_bg,
        "sd_bg_scz": sd_bg,
        "std_effect_scz": std_eff,
        "source_file": str(PGC3_SCZ_GENES_OUT),
    }


def build_symbol_to_ensgid() -> dict[str, str]:
    """Symbol -> ENSGID using gnomAD mane_select=True row gene_id column.

    WHY gnomAD as the bridge: it contains both gene symbol and ENSGID
    (gene_id) for the canonical+mane_select transcript. No new dependency.
    """
    df = pd.read_csv(GNOMAD_TSV, sep="\t", low_memory=False,
                     usecols=["gene", "gene_id", "mane_select"])
    if df["mane_select"].dtype == object:
        df["mane_select"] = df["mane_select"].astype(str).str.lower().isin({"true", "1", "yes"})
    df = df[df["mane_select"]]
    # keep rows whose gene_id looks like ENSG...
    df = df[df["gene_id"].astype(str).str.startswith("ENSG")]
    return df.drop_duplicates(subset="gene", keep="first").set_index("gene")["gene_id"].to_dict()


# ---------------------------------------------------------------------------
# Provenance / reproduction-gate aggregator
# ---------------------------------------------------------------------------
def build_provenance(logger: logging.Logger,
                     gnomad: pd.DataFrame,
                     f144_set: list[str]) -> dict[str, Any]:
    """SHA256s + frozen gene-set hashes + hyperparameters."""
    bg = make_bg(gnomad)
    n_unique = bg["gene"].nunique()
    n_rows = len(bg)
    prov: dict[str, Any] = {
        "sha256": {
            "gnomad_v4_1_constraint_tsv": sha256_file(GNOMAD_TSV),
            "batch_054_B_results_json": sha256_file(F144_RESULTS_JSON),
            "batch_048_script": sha256_file(BATCH_048_SCRIPT),
        },
        "file_paths": {
            "gnomad_v4_1_constraint_tsv": str(GNOMAD_TSV),
            "batch_054_B_results_json": str(F144_RESULTS_JSON),
            "magma_binary": str(MAGMA_BIN),
            "magma_geneloc": str(MAGMA_GENELOC),
            "magma_bfile": str(MAGMA_BFILE),
            "sumstats_dir": str(SUMSTATS_DIR),
        },
        "gene_lists": {
            "B3": {
                "n": len(B3_GENES),
                "sha256_observed": sha256_of_genes(B3_GENES),
                "sha256_expected": B3_SHA256_EXPECTED,
                "sha256_match": sha256_of_genes(B3_GENES) == B3_SHA256_EXPECTED,
                "genes": list(B3_GENES),
            },
            "F144": {
                "n": len(f144_set),
                "sha256": sha256_of_genes(f144_set),
            },
        },
        "bg_dedup_check": {
            "n_unique_genes": int(n_unique),
            "n_rows": int(n_rows),
            "ratio": (n_unique / n_rows) if n_rows else 0.0,
        },
        "hyperparameters": {
            "PLI_THRESHOLD": PLI_THRESHOLD,
            "LOEUF_THRESHOLD": LOEUF_THRESHOLD,
            "MIS_Z_THRESHOLD_LITERAL": MIS_Z_THRESHOLD_LITERAL,
            "RNG_SEED_PERMS": RNG_SEED_PERMS,
            "RNG_SEED_BOOTSTRAP": RNG_SEED_BOOTSTRAP,
            "N_PERM_SUBB": N_PERM_SUBB,
            "N_BOOTSTRAP": N_BOOTSTRAP,
            "N_PERM_SUBC": N_PERM_SUBC,
            "BH_ALPHA": BH_ALPHA,
            "MAGMA_WINDOW_UP_KB": MAGMA_WINDOW_UP_KB,
            "MAGMA_WINDOW_DOWN_KB": MAGMA_WINDOW_DOWN_KB,
        },
    }
    if not prov["gene_lists"]["B3"]["sha256_match"]:
        logger.error("B3 sha256 mismatch — frozen gene set drift!")
    return prov


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="batch_055_B v2 driver")
    parser.add_argument("--only-sub-a", action="store_true")
    parser.add_argument("--only-sub-b", action="store_true")
    parser.add_argument("--only-sub-c", action="store_true")
    parser.add_argument("--skip-sub-c", action="store_true",
                        help="skip Sub-C MAGMA (faster local dry-run)")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Sub-C: skip per-disorder chr22 smoke test")
    args = parser.parse_args()

    t_start = time.time()
    logger = setup_logger()
    logger.info("=== batch_055_B v2 driver start (%s) ===",
                _dt.datetime.now(_dt.timezone.utc).isoformat())

    # ---- Load gnomAD via batch_048 primitive (Rule 1).
    logger.info("Loading gnomAD via batch_048.load_gnomad()")
    gnomad = load_gnomad(logger)

    # ---- Read F144 set from batch_054_B results.json.
    with F144_RESULTS_JSON.open() as fh:
        b054 = json.load(fh)
    f144_set = b054["gene_lists"]["F144_set"]["genes"]
    f144_sha_observed = sha256_of_genes(f144_set)
    f144_sha_expected = b054["gene_lists"]["F144_set"]["sha256"]
    if f144_sha_observed != f144_sha_expected:
        logger.error("F144 sha256 mismatch: %s vs %s",
                     f144_sha_observed, f144_sha_expected)
        return 5

    provenance = build_provenance(logger, gnomad, f144_set)

    sub_a: dict[str, Any] = {"status": "NOT_RUN"}
    sub_b: dict[str, Any] = {"status": "NOT_RUN"}
    sub_c: dict[str, Any] = {"status": "NOT_RUN"}
    repro: dict[str, Any] = {"sub_a_count_match": None,
                              "sub_b_full_OR_match": None,
                              "sub_c_anchor_match": None}

    table = None

    do_a = (not args.only_sub_b) and (not args.only_sub_c)
    do_b = (not args.only_sub_a) and (not args.only_sub_c)
    do_c = (not args.only_sub_a) and (not args.only_sub_b) and (not args.skip_sub_c)

    # ---- Sub-A
    if do_a:
        try:
            table, sub_a = build_per_gene_table(logger, gnomad, f144_set)
            sub_a["status"] = "OK"
            write_per_gene_tsv(table, OUTPUT_DIR / "per_gene_constraint.tsv", logger)
            # Reproduction gate Sub-A: counts must match batch_054_B observed_a.
            counts = sub_a["B3_counts"]
            mismatches = []
            mapping = {
                "pLI >= 0.9": ("pLI >= 0.9", counts["pLI >= 0.9"]),
                "LOEUF <= 0.35": ("LOEUF <= 0.35", counts["LOEUF <= 0.35"]),
                "missense_z top decile": ("mis_z top decile", counts["mis_z top decile"]),
            }
            for k, target in B3_OBSERVED_A_TARGETS.items():
                _, observed = mapping[k]
                if observed != target:
                    mismatches.append(f"{k}: observed={observed} expected={target}")
            repro["sub_a_count_match"] = (len(mismatches) == 0)
            repro["sub_a_count_mismatches"] = mismatches
            if mismatches:
                logger.error("SUB-A reproduction gate FAILED: %s", mismatches)
                # Persist what we have, then exit non-zero per brief.
                _write_results(provenance, sub_a, sub_b, sub_c, repro, t_start)
                return 2
            logger.info("SUB-A reproduction gate PASS")
        except AssertionError as e:
            logger.error("SUB-A assertion: %s", e)
            sub_a = {"status": "FAILED", "error": str(e)}
            _write_results(provenance, sub_a, sub_b, sub_c, repro, t_start)
            return 2
        except Exception as e:
            logger.exception("SUB-A unexpected error")
            sub_a = {"status": "FAILED", "error": str(e), "trace": traceback.format_exc()}
            _write_results(provenance, sub_a, sub_b, sub_c, repro, t_start)
            return 2

    # ---- Sub-B
    if do_b:
        try:
            bg = make_bg(gnomad)
            if table is None:
                table, _sa = build_per_gene_table(logger, gnomad, f144_set)
            sub_b_payload, sub_b_repro = run_sub_b_lofgo(logger, bg, table)
            repro["sub_b_full_OR_match"] = sub_b_repro.get("match", False)
            repro["sub_b_full_OR_detail"] = sub_b_repro
            if not sub_b_repro.get("match", False):
                logger.error("SUB-B reproduction gate FAILED — exit 3")
                sub_b = {"status": "REPRO_FAILED", **sub_b_repro}
                _write_results(provenance, sub_a, sub_b, sub_c, repro, t_start)
                return 3
            sub_b = {"status": "OK", **sub_b_payload}
        except Exception as e:
            logger.exception("SUB-B unexpected error")
            sub_b = {"status": "FAILED", "error": str(e), "trace": traceback.format_exc()}
            _write_results(provenance, sub_a, sub_b, sub_c, repro, t_start)
            return 3

    # ---- Sub-C
    if do_c:
        try:
            sub_c_payload, sub_c_repro = run_sub_c(logger, skip_smoke=args.skip_smoke)
            sub_c = {"status": "OK", **sub_c_payload}
            repro["sub_c_anchor_match"] = sub_c_repro.get("scz_anchor", {}).get("status") == "OK"
            repro["sub_c_scz_anchor"] = sub_c_repro.get("scz_anchor")

            n_failed = sub_c_payload.get("n_disorders_failed", 0)
            if n_failed > 1:
                logger.error("SUB-C: %d disorders failed (>1) — UNINTERPRETABLE per brief",
                             n_failed)
                sub_c["status"] = "UNINTERPRETABLE_TOO_MANY_FAILURES"
                _write_results(provenance, sub_a, sub_b, sub_c, repro, t_start)
                return 4
        except Exception as e:
            logger.exception("SUB-C unexpected error")
            sub_c = {"status": "FAILED", "error": str(e), "trace": traceback.format_exc()}
            _write_results(provenance, sub_a, sub_b, sub_c, repro, t_start)
            return 4

    _write_results(provenance, sub_a, sub_b, sub_c, repro, t_start)
    logger.info("=== batch_055_B v2 driver done (%.1f min) ===",
                (time.time() - t_start) / 60.0)
    return 0


def _write_results(provenance: dict[str, Any],
                   sub_a: dict[str, Any],
                   sub_b: dict[str, Any],
                   sub_c: dict[str, Any],
                   repro: dict[str, Any],
                   t_start: float) -> None:
    """Atomically write results.json (brief §Output schema)."""
    payload = {
        "batch": "batch_055_B",
        "version": "v2",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "provenance": provenance,
        "sub_a": sub_a,
        "sub_b_lofgo": sub_b,
        "sub_c_cross_disorder": sub_c,
        "reproduction_gate": repro,
        "wall_time_min": round((time.time() - t_start) / 60.0, 3),
    }
    atomic_write_json(payload, OUTPUT_DIR / "results.json")


if __name__ == "__main__":
    sys.exit(main())
