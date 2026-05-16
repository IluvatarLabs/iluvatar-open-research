"""Shared helpers for batch_057 sub-experiments.

WHY a shared module (not vendored into each script): Sub-A, Sub-B, and Sub-C
all need identical gene-panel loading (gnomAD dedup, gene_annot, PoPS preds,
MAGMA-Z, NSNPS, STRING PPI, GTEx brain TPM). Sharing a single loader prevents
silent drift between sub-experiments and guarantees the 17,459-gene sample is
constructed identically in Sub-B (reproduction gate R2) and the 16,556-gene
sample in Sub-A/C (reproduction gate R1 / iter_056 Sub-D anchor).

WHY we import from batch_056/_common rather than duplicating: cardinal Rule 1
(never reinvent). batch_056/_common already provides `load_gnomad_per_brief_v2`,
`load_gene_annot`, `load_magma_disorder`, `load_common_ensgids`, `B3_GENES`,
`B3_BIOLOGICAL_CATEGORY`, `atomic_write_json`, `sha256_file`,
`load_preds`, `build_bootstrap_idx`, `percentile_ci`, `load_magma_scz`.
We re-export these at the module level for downstream scripts.

Determinism notes (Cardinal Rule 0 + ML Research Standards):
- `np.random.default_rng(SEED).integers(...)` is bit-reproducible across
  numpy >= 1.17 given the same seed and output shape.
- pandas `.merge` with `how="inner"` on sorted unique keys is deterministic; we
  sort by ENSGID ascending BEFORE computing correlations so the gene order is
  fully determined by the input files.

No fabrication (Cardinal Rule 0): every loader RAISES FileNotFoundError with a
clear diagnostic if inputs are missing. No silent default values.
"""
from __future__ import annotations

import gzip
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Absolute paths (agent cwd resets between calls per instructions).
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_057"
OUTPUT_DIR = BATCH_DIR / "output"
LOGS_DIR = BATCH_DIR / "logs"

# Reuse batch_056 _common via importlib (Rule 1: no reinvention).
# WHY importlib (not sys.path): the batch_056 file is ALSO named `_common.py`.
# Adding its directory to sys.path would shadow THIS module. importlib.util
# with a module spec lets us load the file under a distinct module name.
import importlib.util as _ilu

_BATCH056_COMMON_PATH = (
    PROJECT_ROOT / "experiments" / "batch_056" / "scripts" / "_common.py"
)
if not _BATCH056_COMMON_PATH.exists():
    raise FileNotFoundError(
        f"batch_056 _common.py missing: {_BATCH056_COMMON_PATH}"
    )
_spec = _ilu.spec_from_file_location(
    "batch056_common", str(_BATCH056_COMMON_PATH)
)
_batch056_common = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_batch056_common)

B3_BIOLOGICAL_CATEGORY = _batch056_common.B3_BIOLOGICAL_CATEGORY
B3_GENES = _batch056_common.B3_GENES
BOOTSTRAP_N = _batch056_common.BOOTSTRAP_N
BOOTSTRAP_SEED = _batch056_common.BOOTSTRAP_SEED
GENE_ANNOT = _batch056_common.GENE_ANNOT
GNOMAD_TSV = _batch056_common.GNOMAD_TSV
MAGMA_GENELOC = _batch056_common.MAGMA_GENELOC
MAGMA_SCZ_GENES_OUT = _batch056_common.MAGMA_SCZ_GENES_OUT
REPRO_TOLERANCE = _batch056_common.REPRO_TOLERANCE
atomic_write_json = _batch056_common.atomic_write_json
build_bootstrap_idx = _batch056_common.build_bootstrap_idx
load_common_ensgids = _batch056_common.load_common_ensgids
load_gene_annot = _batch056_common.load_gene_annot
load_gnomad_per_brief_v2 = _batch056_common.load_gnomad_per_brief_v2
load_magma_disorder = _batch056_common.load_magma_disorder
load_magma_scz = _batch056_common.load_magma_scz
load_preds = _batch056_common.load_preds
percentile_ci = _batch056_common.percentile_ci
sha256_file = _batch056_common.sha256_file

# batch_054_A PoPS p=0.05 preds (for Sub-B).
BATCH054_P05_PREDS = (
    PROJECT_ROOT / "experiments" / "batch_054_A" / "output"
    / "sweep" / "cutoff_0.05" / "PGC3_EUR_PoPS.preds"
)

# batch_055_B per-disorder MAGMA (Entrez-keyed).
BATCH055B_WORK = PROJECT_ROOT / "experiments" / "batch_055_B" / "work"

# STRING v12 (pre-staged by _prestage_string.py).
STRING_DIR = PROJECT_ROOT / "data" / "string_v12"
STRING_FULL = STRING_DIR / "9606.protein.links.v12.0.txt.gz"
STRING_PHYSICAL = STRING_DIR / "9606.protein.physical.links.v12.0.txt.gz"
STRING_ALIASES = STRING_DIR / "9606.protein.aliases.v12.0.txt.gz"

# STRING confidence threshold (high-confidence). Source: STRING v12.0 documentation
# (https://version-12-0.string-db.org/cgi/info), "high confidence" ≥ 0.700.
STRING_HIGH_CONF = 700  # combined_score in 0..1000 integer scale

# GTEx v8 median TPM file.
GTEX_TPM = PROJECT_ROOT / "data" / "GTEx_v8_gene_median_tpm.gct.gz"

# The 13 brain region columns in GTEx (verified in brief_v2 line 238).
# GTEx column names use spaces and punctuation; we canonicalize during load.
# Source: GTEx portal tissue list.
GTEX_BRAIN_REGIONS_13 = [
    "Brain - Cortex",
    "Brain - Frontal Cortex (BA9)",
    "Brain - Hippocampus",
    "Brain - Amygdala",
    "Brain - Anterior cingulate cortex (BA24)",
    "Brain - Caudate (basal ganglia)",
    "Brain - Cerebellar Hemisphere",
    "Brain - Cerebellum",
    "Brain - Hypothalamus",
    "Brain - Nucleus accumbens (basal ganglia)",
    "Brain - Putamen (basal ganglia)",
    "Brain - Spinal cord (cervical c-1)",
    "Brain - Substantia nigra",
]

# EDT1 source: Trubetskoy 2022 ST12 xlsx (per batch_048/run_batch_048.py L51).
PGC3_XLSX = PROJECT_ROOT / "data" / "19426775" / "scz2022-Extended-Data-Table1.xlsx"
# Koopmans SynGO 2024 GMT source (per batch_054_B/run_batch_054_B.py L75).
SYNGO_GMT = PROJECT_ROOT / "experiments" / "batch_052_A" / "input" / "syngo_2024.gmt"

# Reproduction anchors.
REPRO_R1_SCZ_BETA_LO = 2.5  # brief_v2 line 226
REPRO_R1_SCZ_BETA_HI = 3.5
REPRO_R2_PARTIAL_RHO_TARGET = 0.510  # brief_v2 line 59 / design.yaml R2
REPRO_R2_PARTIAL_TOLERANCE = 1e-3    # brief_v2 line 59

# BH-FDR alpha.
BH_Q = 0.05


# -----------------------------------------------------------------------------
# NSNPS loader
# -----------------------------------------------------------------------------
def load_nsnps_per_disorder(disorder: str) -> pd.DataFrame:
    """Return DataFrame[ENSGID, NSNPS] for a disorder.

    WHY column 5 (NSNPS): MAGMA gene-analysis output schema is
        GENE CHR START STOP NSNPS NPARAM N ZSTAT P
    verified for both batch_053_B (SCZ, ENSGID-keyed) and batch_055_B
    (per-disorder, Entrez-keyed).

    For SCZ we read the ENSGID-keyed file directly (GENE is already ENSGID).
    For other disorders we apply the same Entrez→Symbol→ENSGID bridge as
    `load_magma_disorder` so NSNPS keys align with MAGMA-Z keys 1:1.
    """
    if disorder == "scz":
        if not MAGMA_SCZ_GENES_OUT.exists():
            raise FileNotFoundError(
                f"MAGMA SCZ gene-Z missing: {MAGMA_SCZ_GENES_OUT}"
            )
        df = pd.read_csv(MAGMA_SCZ_GENES_OUT, sep=r"\s+")
        if "NSNPS" not in df.columns or "GENE" not in df.columns:
            raise RuntimeError(
                f"SCZ MAGMA missing NSNPS/GENE; cols={list(df.columns)}"
            )
        out = df.rename(columns={"GENE": "ENSGID"})[["ENSGID", "NSNPS"]].copy()
        out["NSNPS"] = out["NSNPS"].astype(int)
        return out

    genes_out = BATCH055B_WORK / disorder / "full.gene.genes.out"
    if not genes_out.exists():
        raise FileNotFoundError(f"Per-disorder MAGMA missing: {genes_out}")
    df = pd.read_csv(genes_out, sep=r"\s+")
    if "NSNPS" not in df.columns or "GENE" not in df.columns:
        raise RuntimeError(
            f"{disorder} MAGMA missing NSNPS/GENE; cols={list(df.columns)}"
        )
    df = df.rename(columns={"GENE": "entrez"})[["entrez", "NSNPS"]].copy()
    df["entrez"] = df["entrez"].astype(str)
    df["NSNPS"] = df["NSNPS"].astype(int)

    if not MAGMA_GENELOC.exists():
        raise FileNotFoundError(f"MAGMA geneloc missing: {MAGMA_GENELOC}")
    geneloc = pd.read_csv(
        MAGMA_GENELOC, sep="\t", header=None,
        names=["entrez", "chr", "start", "end", "strand", "symbol"],
        dtype={"entrez": str, "symbol": str},
    )
    ent2sym = geneloc.drop_duplicates(subset="entrez", keep="first")[
        ["entrez", "symbol"]
    ]
    df = df.merge(ent2sym, on="entrez", how="left")

    annot = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME"]]
    sym2ensg = annot.drop_duplicates(subset="NAME", keep="first").rename(
        columns={"NAME": "symbol"}
    )
    df = df.merge(sym2ensg, on="symbol", how="left")
    df = df.dropna(subset=["ENSGID"])
    df = df.drop_duplicates(subset="ENSGID", keep="first")
    return df[["ENSGID", "NSNPS"]].reset_index(drop=True)


# -----------------------------------------------------------------------------
# STRING PPI loaders
# -----------------------------------------------------------------------------
def _load_string_ensp_to_ensg() -> dict[str, str]:
    """Build ENSP (9606.ENSP...) → ENSGID map using STRING aliases.

    WHY aliases (not the protein.info file): STRING's aliases TSV
    carries explicit 'Ensembl_gene' entries per ENSP. We prefer the
    Ensembl-sourced ENSG alias when multiple aliases exist.

    Returns dict mapping `9606.ENSPxxxxxxxxxxx` to the ENSGID string. Genes
    with no Ensembl_gene alias are dropped (cardinal Rule 0: no silent
    invention of mappings).
    """
    if not STRING_ALIASES.exists():
        raise FileNotFoundError(
            f"STRING aliases missing: {STRING_ALIASES} (run _prestage_string.py)"
        )
    # File is TSV: `#string_protein_id\talias\tsource`.
    # We can filter source to the Ensembl-gene alias to guarantee ENSG.
    rows = []
    with gzip.open(STRING_ALIASES, "rt") as fh:
        # Header line starts with '#'.
        header = fh.readline().strip()
        if not header.startswith("#"):
            raise RuntimeError(
                f"Unexpected STRING aliases header: {header!r}"
            )
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sp_id, alias, source = parts[0], parts[1], parts[2]
            # Accept both 'Ensembl_gene' and 'Ensembl_HGNC_ensembl_gene_id'
            # to maximize coverage (both point to an ENSG). Prefer entries
            # where alias starts with 'ENSG' — this is the actual ENSG.
            if "Ensembl" in source and alias.startswith("ENSG"):
                rows.append((sp_id, alias))
    if not rows:
        raise RuntimeError("No ENSP→ENSG rows found in STRING aliases")
    df = pd.DataFrame(rows, columns=["ensp", "ensg"])
    # Multiple ENSG can map to one ENSP when STRING includes HGNC aliases.
    # Keep first in sorted order for determinism.
    df = df.sort_values(["ensp", "ensg"]).drop_duplicates(
        subset="ensp", keep="first"
    )
    return dict(zip(df["ensp"].tolist(), df["ensg"].tolist()))


def load_string_ppi_degree(physical_only: bool) -> pd.DataFrame:
    """Compute per-gene STRING PPI degree.

    Threshold: combined_score >= 700 (high-confidence) for both channels.
    WHY 700 threshold for both: brief_v2 line 103 specifies 700 for full; the
    physical subchannel's combined_score is the same 0..1000 scale. Using the
    same 700 threshold keeps the two degree measures directly comparable.

    Degree count is built by counting distinct STRING edges per gene (STRING
    lists each edge once as `a b score`). Self-loops are dropped.

    Returns DataFrame[ENSGID, PPI_degree].
    """
    path = STRING_PHYSICAL if physical_only else STRING_FULL
    if not path.exists():
        raise FileNotFoundError(
            f"STRING links file missing: {path} (run _prestage_string.py)"
        )
    ensp2ensg = _load_string_ensp_to_ensg()

    # Stream-parse the gzip to avoid loading the whole file into memory.
    degree: dict[str, int] = {}
    with gzip.open(path, "rt") as fh:
        header = fh.readline().strip().split()
        # Expect: "protein1 protein2 combined_score" (space-separated).
        if header[:3] != ["protein1", "protein2", "combined_score"]:
            raise RuntimeError(
                f"Unexpected STRING links header: {header!r}"
            )
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            p1, p2, score_s = parts[0], parts[1], parts[2]
            try:
                score = int(score_s)
            except ValueError:
                continue
            if score < STRING_HIGH_CONF:
                continue
            if p1 == p2:
                continue
            g1 = ensp2ensg.get(p1)
            g2 = ensp2ensg.get(p2)
            if g1 is None or g2 is None:
                continue
            # Each directed line a->b; STRING exports both (a,b) and (b,a)
            # so simply count per endpoint. To avoid double-counting we use
            # the canonical (min,max) pair via a set; but since STRING
            # exports ~2 entries per edge, we still want distinct edges. Use
            # per-gene SET of partners for correctness:
            degree.setdefault(g1, set()).add(g2)
            degree.setdefault(g2, set()).add(g1)

    # Collapse to integer counts.
    data = [(g, len(partners)) for g, partners in degree.items()]
    return pd.DataFrame(data, columns=["ENSGID", "PPI_degree"]).sort_values(
        "ENSGID"
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# GTEx brain TPM
# -----------------------------------------------------------------------------
def load_brain_tpm_median_13regions() -> pd.DataFrame:
    """Return DataFrame[ENSGID, brain_TPM_median] from GTEx v8 13 brain regions.

    GTEx rows are ENSGID.version; we strip '.version' to match ENSGID
    (matches batch_*/common practice).

    WHY median (not mean): brief_v2 line 100 + critic-1 MINOR M3 adopted
    median for robustness across 13 regions (some regions have much higher
    TPM). Median is less influenced by cerebellum outlier and spinal cord
    low-expression tail.
    """
    if not GTEX_TPM.exists():
        raise FileNotFoundError(f"GTEx median TPM missing: {GTEX_TPM}")
    # The GCT has 2 header lines then a wide TSV.
    df = pd.read_csv(
        GTEX_TPM, sep="\t", skiprows=2, compression="gzip", low_memory=False
    )
    missing = [r for r in GTEX_BRAIN_REGIONS_13 if r not in df.columns]
    if missing:
        raise RuntimeError(
            f"GTEx is missing expected brain-region columns: {missing}"
        )
    if "Name" not in df.columns:
        raise RuntimeError("GTEx missing 'Name' (ENSGID.version) column")
    df = df[["Name"] + list(GTEX_BRAIN_REGIONS_13)].copy()
    df["ENSGID"] = df["Name"].astype(str).str.split(".").str[0]
    mat = df[GTEX_BRAIN_REGIONS_13].to_numpy(dtype=float)
    df["brain_TPM_median"] = np.nanmedian(mat, axis=1)
    out = df[["ENSGID", "brain_TPM_median"]].copy()
    # Drop duplicates (GTEx occasionally lists two entries per ENSGID when
    # .version differs; keep the first for determinism).
    out = out.drop_duplicates(subset="ENSGID", keep="first").reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Gene-set loaders: Koopmans + EDT1
# -----------------------------------------------------------------------------
def load_koopmans_full_symbols() -> set[str]:
    """Parse Koopmans SynGO 2024 GMT → union of all set members (symbols).

    Verbatim logic from batch_054_B/run_batch_054_B.py:parse_syngo_gmt — Rule 1.
    Returns a set of gene symbols (HGNC-like).
    """
    if not SYNGO_GMT.exists():
        raise FileNotFoundError(f"SynGO GMT missing: {SYNGO_GMT}")
    union: set[str] = set()
    with SYNGO_GMT.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            # Layout: name, description, gene1, gene2, ...
            for g in parts[2:]:
                g = g.strip()
                if g:
                    union.add(g)
    return union


def load_koopmans_ex_B3() -> set[str]:
    """Koopmans SynGO union MINUS the 18 B3 genes.

    WHY exclude B3: brief_v2 §Sub-C (critic-3 MAJOR-2) demands the B3-ex
    variant so that Sub-C's scope-extension conclusion is B3-independent.
    """
    full = load_koopmans_full_symbols()
    return full - set(B3_GENES)


def load_edt1() -> set[str]:
    """EDT1_all_pc symbol set (Trubetskoy 2022 ST12 all criteria, PC only).

    Verbatim logic from batch_048/run_batch_048.py:load_pgc3_edt1 — Rule 1.
    """
    if not PGC3_XLSX.exists():
        raise FileNotFoundError(f"PGC3 ST12 xlsx missing: {PGC3_XLSX}")
    df = pd.read_excel(PGC3_XLSX, sheet_name="ST12 all criteria")
    pc = df[df["gene_biotype"] == "protein_coding"]
    return set(pc["Symbol.ID"].dropna().astype(str).tolist())


# -----------------------------------------------------------------------------
# Symbol → ENSGID via gene_annot
# -----------------------------------------------------------------------------
def symbols_to_ensgids(symbols: set[str]) -> tuple[set[str], dict[str, str]]:
    """Map symbols → ENSGIDs using gene_annot NAME↔ENSGID.

    Returns (ensg_set, symbol_to_ensg dict). Symbols unmapped are dropped; the
    caller can compute mapping rate from the returned dict size.
    """
    annot = pd.read_csv(GENE_ANNOT, sep="\t")
    name2ensg = annot.drop_duplicates(subset="NAME", keep="first").set_index(
        "NAME"
    )["ENSGID"].to_dict()
    out: dict[str, str] = {}
    for s in symbols:
        ensg = name2ensg.get(s)
        if ensg is not None:
            out[s] = ensg
    return set(out.values()), out


# -----------------------------------------------------------------------------
# Huber abs-diff check (brief_v2 L056_01 fix)
# -----------------------------------------------------------------------------
def abs_diff_huber_check(beta_ols: float, beta_huber: float,
                          se_ols: float,
                          threshold: float = 0.3
                          ) -> tuple[bool, float, float]:
    """Return (pass_flag, abs_diff_over_se, threshold).

    Pass when |β_OLS − β_Huber| < threshold · SE_OLS (brief_v2 line 24).
    Threshold is a multiplier on SE_OLS (not a fixed absolute), per brief.

    If any input is NaN/inf, returns pass_flag=False and diagnostic NaN.
    """
    try:
        diff = float(abs(beta_ols - beta_huber))
        se_ols_f = float(se_ols)
        if not np.isfinite(diff) or not np.isfinite(se_ols_f) or se_ols_f == 0:
            return False, float("nan"), float(threshold)
        ratio = diff / se_ols_f
        return bool(ratio < threshold), float(ratio), float(threshold)
    except Exception:
        return False, float("nan"), float(threshold)


def rel_diff_huber_check(beta_ols: float, beta_huber: float,
                          threshold: float = 0.20
                          ) -> tuple[bool, float, float]:
    """Relative-difference Huber check (iter_056 Sub-D style).

    Pass when |β_OLS − β_Huber| / max(|β_OLS|, 1e-9) < threshold.
    threshold default 0.20 matches batch_056 HUBER_AGREE_TOLERANCE.
    """
    try:
        diff = float(abs(beta_ols - beta_huber))
        denom = float(max(abs(beta_ols), 1e-9))
        if not np.isfinite(diff):
            return False, float("nan"), float(threshold)
        ratio = diff / denom
        return bool(ratio < threshold), float(ratio), float(threshold)
    except Exception:
        return False, float("nan"), float(threshold)


# -----------------------------------------------------------------------------
# BH-FDR (deterministic)
# -----------------------------------------------------------------------------
def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up FDR. Returns q-values in original order.

    WHY our own (instead of statsmodels) — determinism + simplicity. Result
    is bit-identical to `statsmodels.stats.multitest.multipletests(method="fdr_bh")`.
    """
    arr = np.asarray(pvals, dtype=float)
    n = arr.size
    if n == 0:
        return []
    order = np.argsort(arr, kind="stable")
    ranked = arr[order]
    q_ranked = ranked * n / (np.arange(n) + 1)
    # Ensure monotonic non-increasing when walking from largest to smallest.
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = q_ranked
    return [float(x) for x in q.tolist()]


# -----------------------------------------------------------------------------
# Convenience: build the 8-disorder analysis frame
# -----------------------------------------------------------------------------
def build_sub_a_frame(
    disorder: str,
    gnomad: pd.DataFrame,
    annot: pd.DataFrame,
    gene_set_ensg: set[str],
    gene_set_col: str = "in_set",
) -> pd.DataFrame:
    """Assemble the regression frame for one disorder for Sub-A/C.

    Columns (in order):
        ENSGID, MAGMA_Z, <gene_set_col>, log10_gene_length, lof_pLI,
        lof_oe_ci_upper, log10_exp_lof_plus1, log10_NSNPS_plus1,
        missense_z (None unless attached later), NSNPS
    """
    magma = load_magma_disorder(disorder)[["ENSGID", "MAGMA_Z"]]
    nsnps = load_nsnps_per_disorder(disorder)
    frame = (
        magma.merge(nsnps, on="ENSGID", how="inner")
        .merge(gnomad[["ENSGID", "lof_pLI", "lof_oe_ci_upper", "lof_exp"]],
               on="ENSGID", how="inner")
        .merge(annot[["ENSGID", "log10_gene_length"]], on="ENSGID", how="inner")
    )
    frame = frame.dropna(
        subset=["MAGMA_Z", "lof_pLI", "lof_exp", "log10_gene_length", "NSNPS"]
    ).copy()
    frame[gene_set_col] = frame["ENSGID"].isin(gene_set_ensg).astype(int)
    frame["log10_exp_lof_plus1"] = np.log10(
        frame["lof_exp"].astype(float) + 1.0
    )
    frame["log10_NSNPS_plus1"] = np.log10(
        frame["NSNPS"].astype(float) + 1.0
    )
    frame = frame.sort_values("ENSGID").reset_index(drop=True)
    return frame


# -----------------------------------------------------------------------------
# Logger factory
# -----------------------------------------------------------------------------
def setup_logger(name: str, logfile: Path) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# Re-export for downstream convenience (avoid callers hunting batch_056 imports).
__all__ = [
    "PROJECT_ROOT", "BATCH_DIR", "OUTPUT_DIR", "LOGS_DIR",
    "BATCH054_P05_PREDS", "BATCH055B_WORK",
    "STRING_DIR", "STRING_FULL", "STRING_PHYSICAL",
    "STRING_ALIASES", "STRING_HIGH_CONF", "GTEX_TPM", "GTEX_BRAIN_REGIONS_13",
    "PGC3_XLSX", "SYNGO_GMT", "REPRO_R1_SCZ_BETA_LO", "REPRO_R1_SCZ_BETA_HI",
    "REPRO_R2_PARTIAL_RHO_TARGET", "REPRO_R2_PARTIAL_TOLERANCE", "BH_Q",
    "BOOTSTRAP_N", "BOOTSTRAP_SEED", "REPRO_TOLERANCE",
    "GENE_ANNOT", "GNOMAD_TSV", "MAGMA_GENELOC", "MAGMA_SCZ_GENES_OUT",
    # Re-exported from batch_056/_common
    "B3_GENES", "B3_BIOLOGICAL_CATEGORY",
    "atomic_write_json", "sha256_file", "load_preds",
    "load_gnomad_per_brief_v2", "load_gene_annot", "load_common_ensgids",
    "load_magma_disorder", "load_magma_scz",
    "build_bootstrap_idx", "percentile_ci",
    # New
    "load_nsnps_per_disorder", "load_string_ppi_degree",
    "load_brain_tpm_median_13regions",
    "load_koopmans_full_symbols", "load_koopmans_ex_B3", "load_edt1",
    "symbols_to_ensgids",
    "abs_diff_huber_check", "rel_diff_huber_check", "bh_fdr",
    "build_sub_a_frame", "setup_logger",
]
