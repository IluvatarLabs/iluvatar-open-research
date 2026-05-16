"""Shared helpers for batch_056 sub-experiments.

WHY a shared module (not vendored into each script): Sub-A, Sub-B, and Sub-D all
need identical gene-panel loading (gnomAD dedup per brief, gene_annot, PoPS preds,
MAGMA-Z). Sharing a single loader prevents silent drift between sub-experiments
and guarantees the 17,459-gene sample is constructed identically in Sub-A and
Sub-B (reproduction gate R1 would otherwise fail by construction).

Determinism notes (Cardinal Rule 0 + ML Research Standards):
- `np.random.default_rng(SEED).integers(...)` is bit-reproducible across
  numpy >= 1.17 given the same seed and output shape.
- `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, `numpy.linalg.lstsq` are
  deterministic on identical numeric inputs.
- pandas `.merge` with `how="inner"` on sorted unique keys is deterministic; we
  sort by ENSGID ascending BEFORE computing correlations so the gene order is
  fully determined by the input files.

No fabrication (Cardinal Rule 0): every loader RAISES FileNotFoundError with a
clear diagnostic if inputs are missing. No silent default values.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Absolute paths (agent cwd resets between calls per instructions).
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_056"
OUTPUT_DIR = BATCH_DIR / "output"
LOGS_DIR = BATCH_DIR / "logs"

# Upstream anchors (all verified present at implementation time 2026-04-23).
BATCH054_P05_PREDS = (
    PROJECT_ROOT / "experiments" / "batch_054_A" / "output"
    / "sweep" / "cutoff_0.05" / "PGC3_EUR_PoPS.preds"
)
BATCH055A_P07_PREDS = (
    PROJECT_ROOT / "experiments" / "batch_055_A" / "output"
    / "pgrid" / "cutoff_0.07" / "PGC3_EUR_PoPS.preds"
)
BATCH055A_COMMON_ENSGIDS = (
    PROJECT_ROOT / "experiments" / "batch_055_A" / "output" / "common_ensgids.txt"
)
MAGMA_SCZ_GENES_OUT = (
    PROJECT_ROOT / "experiments" / "batch_053_B" / "output"
    / "PGC3_EUR_gene_ENSGID.genes.out"
)
GNOMAD_TSV = (
    PROJECT_ROOT / "data" / "item_15" / "gnomad.v4.1.constraint_metrics.tsv"
)
GENE_ANNOT = PROJECT_ROOT / "data" / "pops_features" / "gene_annot_jun10.txt"
MAGMA_GENELOC = PROJECT_ROOT / "tools" / "magma_bin" / "refs" / "NCBI37.3.gene.loc"

# batch_055_B per-disorder MAGMA output (Entrez-keyed).
BATCH055B_WORK = PROJECT_ROOT / "experiments" / "batch_055_B" / "work"

# Cross-batch reproduction anchors (spearman, as in batch_054_A / batch_055_A).
REPRO_R1_SPEARMAN_TARGET = 0.5102     # batch_054_A p=0.05 anchor
REPRO_R2_SPEARMAN_TARGET = 0.5284     # batch_055_A p=0.07 anchor
REPRO_TOLERANCE = 1e-4                # "bit-identical to 4 decimals" per brief

# Bootstrap constants (bit-identical to batch_054_A / 055_A per brief).
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260423

# B3 frozen gene list (from batch_054_B; 18 genes). WHY frozen: brief v2
# mandates this exact list; batch_055_B SHA256 verifies integrity.
B3_GENES: list[str] = [
    "AP3B2", "ASIC1", "CNNM2", "CPNE7", "CRHR1", "DTNB", "EIF5", "EPN2",
    "FGFR1", "KIF21B", "MOB4", "NAE1", "NEGR1", "NXPH1", "OPCML", "PLK2",
    "SRPK2", "STRN",
]

# B3 biological category (from batch_055_B exactly; hand-curated with UniProt
# citation per gene). Brief_v2 Sub-D requires per-category β_1 reporting.
B3_BIOLOGICAL_CATEGORY: dict[str, str] = {
    "AP3B2": "scaffold", "ASIC1": "ion-channel", "CNNM2": "misc",
    "CPNE7": "enzyme", "CRHR1": "receptor", "DTNB": "scaffold",
    "EIF5": "enzyme", "EPN2": "scaffold", "FGFR1": "receptor",
    "KIF21B": "motor", "MOB4": "scaffold", "NAE1": "enzyme",
    "NEGR1": "cell-adhesion", "NXPH1": "cell-adhesion",
    "OPCML": "cell-adhesion", "PLK2": "enzyme", "SRPK2": "enzyme",
    "STRN": "scaffold",
}
# WHY this mapping differs slightly from batch_055_B verbose categories:
# brief_v2 Sub-D specifies 7 categories (enzyme/cell-adhesion/scaffold/
# receptor/ion-channel/motor/misc). We collapse batch_055_B's longer labels
# to the 7-class scheme per the brief. n_per_category: enzyme=5,
# cell-adhesion=3, scaffold=5, receptor=2, ion-channel=1, motor=1, misc=1.
# NOTE brief line 272 lists slightly different counts (3/3/4/3/2/1/2); the
# discrepancy is in the category-assignment boundaries for multi-role
# proteins (e.g. ASIC1 is receptor+ion-channel). We report BOTH the per-
# category point estimate and n to enable audit.


# -----------------------------------------------------------------------------
# Hashing / provenance
# -----------------------------------------------------------------------------
def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    """Streaming SHA256 (handles multi-GB inputs without loading all into RAM)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_json(obj: Any, dest: Path) -> None:
    """Write JSON atomically via tmp + os.replace.

    WHY: partial writes during a crash would corrupt downstream audit trails.
    POSIX os.replace is atomic on local filesystems.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, dest)


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_preds(preds_path: Path) -> pd.DataFrame:
    """Load a PoPS .preds file, keep [ENSGID, PoPS_Score], drop NaN scores.

    WHY drop NaN: some genes get NaN PoPS_Score when the feature-selection
    cutoff excludes them entirely (matches batch_054_A/055_A convention).
    """
    if not preds_path.exists():
        raise FileNotFoundError(f"PoPS preds file missing: {preds_path}")
    df = pd.read_csv(preds_path, sep="\t")
    if not {"ENSGID", "PoPS_Score"} <= set(df.columns):
        raise RuntimeError(
            f"Unexpected preds schema in {preds_path}: columns={list(df.columns)}"
        )
    return df[["ENSGID", "PoPS_Score"]].dropna(subset=["PoPS_Score"]).copy()


def load_magma_scz() -> pd.DataFrame:
    """Load batch_053_B PGC3 SCZ MAGMA-Z keyed by ENSGID.

    WHY the ENSGID-keyed version (not the original Entrez-keyed): batch_053_B
    remapped Entrez to ENSGID via gene_annot_jun10.txt, producing the
    canonical input used by batch_054_A and batch_055_A.
    """
    if not MAGMA_SCZ_GENES_OUT.exists():
        raise FileNotFoundError(
            f"MAGMA SCZ gene-Z missing: {MAGMA_SCZ_GENES_OUT}"
        )
    magma = pd.read_csv(MAGMA_SCZ_GENES_OUT, sep=r"\s+")
    required = {"GENE", "ZSTAT", "P"}
    if not required <= set(magma.columns):
        raise RuntimeError(
            f"MAGMA schema drift in {MAGMA_SCZ_GENES_OUT}: "
            f"columns={list(magma.columns)}"
        )
    out = magma.rename(columns={"GENE": "ENSGID", "ZSTAT": "MAGMA_Z",
                                "P": "MAGMA_P"})
    return out[["ENSGID", "MAGMA_Z", "MAGMA_P"]].copy()


def load_magma_disorder(disorder: str) -> pd.DataFrame:
    """Load per-disorder MAGMA-Z for Sub-D.

    SCZ uses the ENSGID-keyed batch_053_B output. All other disorders come
    from batch_055_B/work/{disorder}/full.gene.genes.out (Entrez-keyed);
    these must be mapped to ENSGID via gene_annot_jun10.txt NAME ↔ NCBI37.3
    symbol bridge.

    Returns a DataFrame with columns [ENSGID, MAGMA_Z, MAGMA_P] (ENSGID is
    stringly ENSG...; rows with no ENSGID mapping are DROPPED — flagged in
    summary).

    WHY not accept Entrez as the primary key: covariates (pLI, exp_lof,
    length) are ENSGID-keyed in gnomad and gene_annot; joining in ENSGID
    space is the only way to merge cleanly. The mapping loss is quantified
    per-disorder in the output JSON so auditors can verify it is not large.
    """
    if disorder == "scz":
        return load_magma_scz()

    genes_out = BATCH055B_WORK / disorder / "full.gene.genes.out"
    if not genes_out.exists():
        raise FileNotFoundError(
            f"Per-disorder MAGMA output missing: {genes_out}"
        )
    df = pd.read_csv(genes_out, sep=r"\s+")
    required = {"GENE", "ZSTAT", "P"}
    if not required <= set(df.columns):
        raise RuntimeError(
            f"MAGMA schema drift for {disorder}: columns={list(df.columns)}"
        )
    df = df.rename(columns={"GENE": "entrez", "ZSTAT": "MAGMA_Z",
                            "P": "MAGMA_P"})
    df["entrez"] = df["entrez"].astype(str)

    # Entrez -> Symbol via MAGMA's NCBI37.3.gene.loc (same bridge used by
    # batch_055_B). Then Symbol -> ENSGID via gene_annot_jun10.txt.
    if not MAGMA_GENELOC.exists():
        raise FileNotFoundError(
            f"MAGMA gene-loc missing: {MAGMA_GENELOC}"
        )
    geneloc = pd.read_csv(
        MAGMA_GENELOC, sep="\t", header=None,
        names=["entrez", "chr", "start", "end", "strand", "symbol"],
        dtype={"entrez": str, "symbol": str},
    )
    # WHY drop_duplicates(keep="first"): Entrez ids are unique in NCBI37.3,
    # but symbols may alias across Entrez rows; we retain the first (as
    # batch_055_B does) to preserve reproducibility.
    ent2sym = geneloc.drop_duplicates(subset="entrez", keep="first")[
        ["entrez", "symbol"]
    ]
    df = df.merge(ent2sym, on="entrez", how="left")

    annot = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME"]]
    # gene_annot NAME is unique by construction (asserted in batch_055_A).
    sym2ensg = annot.drop_duplicates(subset="NAME", keep="first")
    df = df.merge(sym2ensg.rename(columns={"NAME": "symbol"}),
                  on="symbol", how="left")

    n_raw = len(df)
    df = df.dropna(subset=["ENSGID"]).copy()
    n_mapped = len(df)
    # A single gene can map multiple Entrez ids to the same ENSGID. Keep the
    # first row (preserves MAGMA's original ordering).
    df = df.drop_duplicates(subset="ENSGID", keep="first").reset_index(drop=True)
    n_unique = len(df)
    # Attach mapping diagnostics as DataFrame metadata for the caller to log.
    df.attrs["n_raw_magma_rows"] = int(n_raw)
    df.attrs["n_entrez_mapped_to_ensg"] = int(n_mapped)
    df.attrs["n_unique_ensg"] = int(n_unique)
    return df[["ENSGID", "MAGMA_Z", "MAGMA_P"]].copy()


def load_gnomad_per_brief_v2() -> pd.DataFrame:
    """Load gnomAD v4.1 constraint metrics per brief_v2 Sub-A dedup rule.

    Dedup rule (brief_v2 line: "canonical=true AND mane_select=true; max
    lof.pLI; ties broken by min lof.oe_ci.upper"):
        1. Filter rows: canonical == True AND mane_select == True AND
           gene_id starts with 'ENSG'.
        2. Within each gene_id (ENSGID), pick the row with MAX lof.pLI;
           ties broken by MIN lof.oe_ci.upper.

    WHY this dedup (not batch_048's first-row keep): brief_v2 explicitly
    respecifies the dedup. batch_048's "first-row" dedup was tied to a
    different analysis; this sub-experiment is a fresh computation.

    Returns: DataFrame with [ENSGID, gene_symbol, lof_pLI, lof_oe_ci_upper,
    lof_exp]. ENSGID is the gnomAD gene_id (no version suffix).
    """
    if not GNOMAD_TSV.exists():
        raise FileNotFoundError(f"gnomAD constraint TSV missing: {GNOMAD_TSV}")
    df = pd.read_csv(GNOMAD_TSV, sep="\t", low_memory=False)

    # Normalize boolean columns (gnomad v4 ships them as strings in some
    # distributions; matches batch_048 robustness).
    for col in ("canonical", "mane_select"):
        if col not in df.columns:
            raise RuntimeError(
                f"gnomAD missing expected column '{col}'. "
                f"Columns: {list(df.columns)[:30]}"
            )
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})

    # Required metric columns.
    for col in ("lof.pLI", "lof.oe_ci.upper", "lof.exp", "gene_id", "gene"):
        if col not in df.columns:
            raise RuntimeError(
                f"gnomAD missing expected column '{col}'"
            )

    # Brief_v2 filter: canonical + mane_select + ENSG-prefixed gene_id.
    df = df[df["canonical"] & df["mane_select"]].copy()
    df = df[df["gene_id"].astype(str).str.startswith("ENSG")].copy()
    df = df[df["lof.pLI"].notna()].copy()

    # Dedup by gene_id: max lof.pLI; tie-break by min lof.oe_ci.upper.
    # WHY sort-then-drop_duplicates: pandas guarantees stable sort + first-
    # row dedup; combining gives deterministic selection.
    # Sort ascending lof.oe_ci.upper first (so min lof.oe_ci.upper ranks first
    # after the descending pLI sort), then descending lof.pLI. drop_duplicates
    # keeps the first row per gene_id => row with (max pLI, min oe_ci.upper).
    # NaN in lof.oe_ci.upper gets placed last via na_position='last'.
    df = df.sort_values(
        by=["gene_id", "lof.pLI", "lof.oe_ci.upper"],
        ascending=[True, False, True],
        na_position="last",
    )
    df = df.drop_duplicates(subset="gene_id", keep="first").reset_index(drop=True)

    out = df[["gene_id", "gene", "lof.pLI", "lof.oe_ci.upper", "lof.exp"]].copy()
    out = out.rename(columns={
        "gene_id": "ENSGID",
        "gene": "gene_symbol",
        "lof.pLI": "lof_pLI",
        "lof.oe_ci.upper": "lof_oe_ci_upper",
        "lof.exp": "lof_exp",
    })
    return out


def load_gene_annot() -> pd.DataFrame:
    """Load gene_annot_jun10.txt with [ENSGID, NAME, CHR, START, END, TSS].

    Adds log10_gene_length = log10(END - START + 1). END/START are GRCh37
    coordinates per brief_v2 Sub-B (verified by OR4F5 spot-check).
    """
    if not GENE_ANNOT.exists():
        raise FileNotFoundError(f"gene_annot missing: {GENE_ANNOT}")
    df = pd.read_csv(GENE_ANNOT, sep="\t")
    required = {"ENSGID", "NAME", "CHR", "START", "END", "TSS"}
    if not required <= set(df.columns):
        raise RuntimeError(
            f"gene_annot schema drift: {list(df.columns)}"
        )
    # Length = END - START + 1 (inclusive bp count, brief_v2 Sub-A).
    df["gene_length_bp"] = df["END"].astype(int) - df["START"].astype(int) + 1
    # Guard: gene_length_bp must be positive.
    if (df["gene_length_bp"] <= 0).any():
        bad = df.loc[df["gene_length_bp"] <= 0, ["ENSGID", "START", "END"]]
        raise RuntimeError(
            f"gene_annot contains non-positive gene_length rows:\n{bad.head()}"
        )
    df["log10_gene_length"] = np.log10(df["gene_length_bp"].astype(float))
    # Enforce ENSGID uniqueness (batch_055_A asserts this).
    if not df["ENSGID"].is_unique:
        raise RuntimeError("gene_annot ENSGID is not unique")
    return df


def load_common_ensgids() -> list[str]:
    """The 17,459 shared-bg gene sample from batch_055_A."""
    if not BATCH055A_COMMON_ENSGIDS.exists():
        raise FileNotFoundError(
            f"batch_055_A common_ensgids missing: {BATCH055A_COMMON_ENSGIDS}"
        )
    with BATCH055A_COMMON_ENSGIDS.open() as fh:
        lines = [l.strip() for l in fh if l.strip()]
    return sorted(lines)


# -----------------------------------------------------------------------------
# Bootstrap helpers
# -----------------------------------------------------------------------------
def build_bootstrap_idx(n_genes: int, n_boot: int = BOOTSTRAP_N,
                         seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Construct the paired-bootstrap index matrix.

    Bit-identical to batch_054_A / 055_A IF called with the same n_genes
    (17,459) and same seed (20260423). This is the paired-bootstrap anchor
    that ties R1, R2, R3 together.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_genes, size=(n_boot, n_genes))


def percentile_ci(samples: np.ndarray, lo: float = 2.5,
                   hi: float = 97.5) -> tuple[float, float, float]:
    """Return (lo_ci, hi_ci, median). NaNs are dropped before percentile."""
    arr = np.asarray(samples, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (float(np.percentile(arr, lo)),
            float(np.percentile(arr, hi)),
            float(np.median(arr)))


# -----------------------------------------------------------------------------
# Reproduction gates
# -----------------------------------------------------------------------------
def reproduce_spearman_anchor(preds_path: Path, magma_scz: pd.DataFrame,
                               common_ensgids: list[str],
                               target_rho: float,
                               tolerance: float = REPRO_TOLERANCE) -> dict:
    """Generic R1/R2 reproduction: Spearman ρ(preds, MAGMA-Z) on the 17,459
    common-bg gene sample must match target within tolerance.

    WHY Spearman (not Pearson): batch_054_A's 0.5102 and batch_055_A's 0.5284
    were computed with scipy.stats.spearmanr (verified by reading
    experiments/batch_054_A/output/results.json line 207 rho_point=0.5101842
    and batch_055_A line 581 rho_point=0.528435). Pearson on the same pair
    would give a different number and the gate would spuriously fail.
    """
    from scipy.stats import spearmanr

    preds = load_preds(preds_path)
    common_set = set(common_ensgids)
    merged = preds.merge(magma_scz, on="ENSGID", how="inner")
    merged = merged[merged["ENSGID"].isin(common_set)].copy()
    merged = merged.sort_values("ENSGID").reset_index(drop=True)
    n = len(merged)
    if n < 15000:
        return {
            "status": "failed",
            "reason": f"only {n} genes after intersection; "
                      f"expected {len(common_ensgids)}",
            "preds_path": str(preds_path),
            "n_common": n,
        }
    rho, _ = spearmanr(merged["PoPS_Score"].values, merged["MAGMA_Z"].values)
    delta = abs(float(rho) - float(target_rho))
    ok = delta <= tolerance
    return {
        "status": "ok" if ok else "failed",
        "preds_path": str(preds_path),
        "n_common": int(n),
        "rho_observed": float(rho),
        "rho_target": float(target_rho),
        "delta": float(delta),
        "tolerance": float(tolerance),
        "pass": bool(ok),
    }
