"""Shared helpers for batch_052_B specification-curve pipeline.

WHY this module: avoid duplicating gene-list / background builders across
per-finding scripts. Every helper returns a dict suitable for JSON dump so
results are provenance-traceable. Keep stateless.
"""
from __future__ import annotations
import hashlib, json, gzip
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BATCH_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BATCH_DIR / "input"
OUTPUT_DIR = BATCH_DIR / "output"
LOGS_DIR = BATCH_DIR / "logs"
for d in (OUTPUT_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RNG_SEED = 20260424
N_PERM = 10_000

# ---- path constants from preflight ----
GWAS_PARQUET   = ROOT / "experiments/batch_008/data/gwas_genes.parquet"
MARKERS_PARQ   = ROOT / "experiments/batch_009/data/markers.parquet"
GENCODE_JSON   = ROOT / "experiments/batch_040/output/gencode_protein_coding_genes.json"
GNOMAD_TSV     = ROOT / "data/item_15/gnomad.v4.1.constraint_metrics.tsv"
GTEX_GCT       = ROOT / "data/GTEx_v8_gene_median_tpm.gct.gz"
SCHEMA_TXT     = ROOT / "experiments/batch_044/input/schema_exome_wide_significant.txt"
PGC3_XLSX      = ROOT / "data/19426775/scz2022-Extended-Data-Table1.xlsx"
SYNGO_GMT      = Path.home() / ".cache/gseapy/Enrichr.SynGO_2022.gmt"
HGNC_FAMILY    = INPUT_DIR / "hgnc_family.tsv"
BATCH048_JSON  = ROOT / "experiments/batch_048/output/A_edt1_decomposition.json"

# ---- hashing ----
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ---- gene list loaders ----
def load_scz_genes() -> set[str]:
    g = pd.read_parquet(GWAS_PARQUET)
    return set(g["hgnc_symbol"].dropna().astype(str))

def load_scz_df() -> pd.DataFrame:
    g = pd.read_parquet(GWAS_PARQUET)
    g["length"] = g["gene_end"].astype(int) - g["gene_start"].astype(int)
    g = g[g["length"] > 0].copy()
    return g[["hgnc_symbol", "length"]].dropna().drop_duplicates("hgnc_symbol")

def load_neuronal_markers() -> set[str]:
    m = pd.read_parquet(MARKERS_PARQ)
    return set(m.loc[m["cell_type"] == "Neurons", "gene"].astype(str))

def load_gencode() -> dict[str, dict]:
    with open(GENCODE_JSON) as f:
        d = json.load(f)
    return d["genes"]

def load_gnomad() -> pd.DataFrame:
    """Canonical + MANE filtered gnomAD v4.1 constraint metrics."""
    df = pd.read_csv(GNOMAD_TSV, sep="\t", low_memory=False)
    df = df[(df["canonical"] == True) & (df["mane_select"] == True)].copy()
    df["pLI"] = pd.to_numeric(df["lof.pLI"], errors="coerce")
    df["LOEUF"] = pd.to_numeric(df["lof.oe_ci.upper"] if "lof.oe_ci.upper" in df.columns else df.columns[0], errors="coerce")
    # Fallback: compute LOEUF from 90% CI upper — gnomAD exposes it as lof.oe_ci.upper in v4.1
    for candidate in ("lof.oe_ci.upper", "lof.oe_upper_ci", "oe_lof_upper", "lof.oe_ci_upper"):
        if candidate in df.columns:
            df["LOEUF"] = pd.to_numeric(df[candidate], errors="coerce")
            break
    df = df.drop_duplicates("gene")
    return df[["gene", "pLI", "LOEUF"]].dropna(subset=["gene"])

def load_schema() -> set[str]:
    with open(SCHEMA_TXT) as f:
        return {l.strip() for l in f if l.strip()}

def load_pgc3():
    xl = pd.ExcelFile(PGC3_XLSX)
    st12 = xl.parse("ST12 all criteria")
    edt1_sheet = xl.parse("Extended.Data.Table.1")
    return {"st12": st12, "edt1_sheet": edt1_sheet}

def load_syngo() -> dict[str, set[str]]:
    """Parse SynGO 2022 Enrichr GMT. Terms are suffixed ' BP' or ' CC'."""
    terms: dict[str, set[str]] = {}
    with open(SYNGO_GMT) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = {g for g in parts[2:] if g}
            terms[name] = genes
    return terms

def load_hgnc_family() -> pd.DataFrame:
    df = pd.read_csv(HGNC_FAMILY, sep="\t", dtype=str)
    df = df.rename(columns={"Approved symbol": "symbol", "Gene group name": "family"})
    return df[["symbol", "family"]].dropna(subset=["symbol"])

def load_gtex_brain_and_any() -> tuple[set[str], set[str]]:
    """Return (brain_expressed, any_tissue_expressed) at TPM >= 1.

    Uses GTEx v8 median TPM file. Brain = any of 13 Brain tissues. Any = any column.
    """
    with gzip.open(GTEX_GCT, "rt") as f:
        _ = f.readline()  # "#1.2"
        _ = f.readline()  # "n_rows\tn_cols"
        header = f.readline().rstrip("\n").split("\t")
    df = pd.read_csv(GTEX_GCT, sep="\t", skiprows=2, low_memory=False)
    # Map Ensembl to symbol via "Description" column
    brain_cols = [c for c in df.columns if c.startswith("Brain")]
    tissue_cols = [c for c in df.columns if c not in ("Name", "Description")]
    tpm_vals = df[tissue_cols].apply(pd.to_numeric, errors="coerce")
    any_mask = (tpm_vals >= 1).any(axis=1)
    brain_mask = (df[brain_cols].apply(pd.to_numeric, errors="coerce") >= 1).any(axis=1)
    any_genes = set(df.loc[any_mask, "Description"].astype(str))
    brain_genes = set(df.loc[brain_mask, "Description"].astype(str))
    return brain_genes, any_genes

# ---- Fisher + Haldane-Anscombe ----
def fisher_or_ci(a: int, b: int, c: int, d: int) -> dict[str, float]:
    """Exact-Fisher OR with Haldane-Anscombe +0.5 if any cell is 0. 95% CI (Woolf)."""
    from scipy.stats import fisher_exact
    halda = any(v == 0 for v in (a, b, c, d))
    aa, bb, cc, dd = (a + 0.5, b + 0.5, c + 0.5, d + 0.5) if halda else (a, b, c, d)
    or_val = (aa * dd) / (bb * cc) if bb * cc > 0 else float("inf")
    se = (1 / aa + 1 / bb + 1 / cc + 1 / dd) ** 0.5 if halda or min(a, b, c, d) > 0 else float("nan")
    from math import log, exp, sqrt
    if or_val not in (0, float("inf")) and not np.isnan(or_val):
        ln = log(or_val)
        lo = exp(ln - 1.96 * se); hi = exp(ln + 1.96 * se)
    else:
        lo, hi = float("nan"), float("nan")
    stat, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return {"a": a, "b": b, "c": c, "d": d,
            "OR": or_val, "OR_low": lo, "OR_high": hi,
            "raw_p": float(p), "haldane_applied": bool(halda)}

# ---- BH-FDR ----
def bh_qvalues(pvals: list[float]) -> list[float]:
    p = np.array(pvals, dtype=float)
    n = len(p); order = np.argsort(p); ranks = np.empty(n, int); ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    # monotone
    s = np.argsort(-p)
    qmin = np.inf; out = np.empty(n, float)
    for idx in np.argsort(p)[::-1]:
        qmin = min(qmin, q[idx])
        out[idx] = min(qmin, 1.0)
    return out.tolist()

# ---- permutation helper ----
def perm_enrichment_p(
    list_genes: set[str], marker_genes: set[str], bg_genes: set[str],
    k_obs: int, n_perm: int = N_PERM, rng: np.random.Generator | None = None
) -> float:
    """Empirical p: shuffle list-membership among bg preserving |list ∩ bg|, count overlaps with markers.

    WHY this null: Simonsohn §3 asks "under null of no association what's the spec-wise p?"; here the
    natural null is "random gene list of same size drawn from bg universe" and effect is the overlap with markers.
    """
    rng = rng or np.random.default_rng(RNG_SEED)
    bg_list = list(bg_genes)
    n_bg = len(bg_list)
    marker_idx = np.array([i for i, g in enumerate(bg_list) if g in marker_genes], dtype=int)
    n_hits = len(list_genes & bg_genes)
    if n_hits == 0 or len(marker_idx) == 0 or n_bg == 0:
        return 1.0
    # Precompute marker membership as boolean
    mark_bool = np.zeros(n_bg, dtype=bool); mark_bool[marker_idx] = True
    count = 0
    for _ in range(n_perm):
        perm = rng.choice(n_bg, size=n_hits, replace=False)
        if mark_bool[perm].sum() >= k_obs:
            count += 1
    return (count + 1) / (n_perm + 1)

def jaccard(a: Iterable, b: Iterable) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    u = A | B
    return len(A & B) / len(u) if u else 0.0

def log_event(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(msg.rstrip() + "\n")
