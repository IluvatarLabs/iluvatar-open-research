"""Analysis B — NNLS signature-based deconvolution of GTEx v8 muscle bulk.

Iter 064, batch_064. Implements B1 (composition) + B1s (fraction-adjusted bulk TF-age).
B2 BayesPrism is DEFERRED to iter 065 because R is not installed on this system.

WHY THIS SCRIPT:
    Population-scale replication of HLMA compartment-level aging claims at N~500
    GTEx skeletal-muscle donors. HLMA N=23 is below benchmarked safety thresholds
    (Tran 2023 [lit_doi_10.1038_s41467-023-41385-5]); GTEx provides an orthogonal
    bulk cohort to test whether aged muscle shows compartment-fraction drift
    (FAP up, Vascular up, MuSC down) and whether TF-age effects survive
    compartment-fraction adjustment.

METHOD CITATION CHAIN:
    - NNLS as canonical GTEx deconvolution: Donovan 2020
      [lit_doi_10.1038_s41467-020-14561-0]
    - GTEx aging + technical confounders (ischemic time dominant):
      [lit_doi_10.1038_s41467-022-33509-0] (Vinuela 2022),
      [lit_doi_10.1101_2022.05.17.492324] (Coronary-Artery case study).

DATA-SCHEMA SURPRISES (documented here, not suppressed — Rule 0):

1. HLMA h5ads contain LOG-NORMALIZED data in both .X and .raw.X (not raw counts).
   Per-cell sums of .X are ~2000-2800 (consistent with log1p of normalize-to-10000),
   and value ranges are 0.4-3.7 (Vascular) / 0.8-2.9 (MuSC). The FAP file
   (OMIX004308-02) even has negative .X values (scaled data) but .raw.X matches
   the scaled (not counts) form too. We therefore build the compartment signature
   as the MEAN log1p-normalized expression across cells per compartment
   (equivalent in rank structure to mean-CPM for our purposes). This is
   documented in the output b_summary.json under 'signature_basis_caveat'.

2. HLMA donor column is 'orig.ident' (e.g., 'OM3_N1' = donor OM3 subregion N1);
   coarse donor label is 'sample' (e.g., 'OM3'). Because signature derivation
   averages across cells regardless of donor (Rule 1: do not re-invent pseudobulk
   averaging if NNLS only needs mean expression per compartment), donor structure
   is not required to build the signature. It is retained in b_summary.json for
   auditability.

3. No Myofiber h5ad is available in the HLMA bundle. HLMA is nuclei-biased and
   myofibers are underrepresented. We therefore run NNLS WITHOUT a Myofiber basis
   (fractions across Vascular, FAP, MuSC, Immune sum to 1); the "missing mass"
   is absorbed across compartments, which inflates non-myofiber fractions.
   This is documented, NOT hidden. Interpretation must therefore focus on
   RELATIVE age-slope direction, not absolute fraction magnitudes.
   Brief's smoke-test plausibility ranges (Vascular 3-10%, etc.) assume
   Myofiber+Other presence and are NOT applicable under this constraint.

4. GTEx v8 public SubjectPhenotypes contains only SUBJID, SEX, AGE, DTHHRDY.
   BMI and RACE are dbGaP-protected and NOT available in the local files.
   The primary model therefore drops BMI and RACE and uses what IS available:
      age + SMTSISCH + SMRIN + SEX + SMCENTER (as COLLECTION_SITE proxy) + DTHHRDY.
   This deviation from the brief is declared in b_summary.json under
   'confounder_deviation'. DTHHRDY (Hardy death-scale 0-4) is a standard GTEx
   tissue-quality covariate (GTEx Consortium 2020).

5. GCT header is '#1.3' not '#1.2'; parser still skips 2 header lines. No impact.

NO EMOJIS. Absolute paths. Type hints. Explicit WHY in docstrings.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls

# statsmodels is used for OLS with cluster-robust SE (WHY: Critic 3 B1 pre-
# requisite — cluster on COLLECTION_SITE to handle residual site-level
# correlation not absorbed by the fixed-effects dummies).
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --------------------------------------------------------------------------- #
# Paths (all absolute per repo convention).
# --------------------------------------------------------------------------- #
REPO_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro")
BATCH_DIR = REPO_ROOT / "experiments" / "batch_064"
LOG_DIR = BATCH_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

GTEX_DIR = REPO_ROOT / "data" / "GTEx" / "muscle"
GTEX_TPM_GCT = GTEX_DIR / "gene_tpm_muscle_skeletal.gct.gz"
GTEX_COUNTS_GCT = GTEX_DIR / "gene_reads_muscle_skeletal.gct.gz"
GTEX_SAMPLE_ATTR = GTEX_DIR / "GTEx_v8_SampleAttributes.txt"
GTEX_SUBJ_PHENO = GTEX_DIR / "GTEx_v8_SubjectPhenotypes.txt"

HLMA_FILES: Dict[str, Path] = {
    # Compartment -> h5ad. WHY these assignments: files are organized by
    # compartment already (Li 2025 HLMA). FAP uses OMIX004308-02 which has
    # only FAP/Tenocyte subpopulations in the Annotation column.
    "Vascular": REPO_ROOT / "data" / "Vascular_scsn_RNA.h5ad",
    "MuSC": REPO_ROOT / "data" / "MuSC_scsn_RNA.h5ad",
    "FAP": REPO_ROOT / "data" / "OMIX004308-02.h5ad",
    "Immune": REPO_ROOT / "data" / "Immune_scsn_RNA.h5ad",
}

# Outputs.
OUT_SIGNATURE = BATCH_DIR / "b_hlma_signature.csv"
OUT_SIGNATURE_SOURCE = BATCH_DIR / "b_hlma_signature_source.csv"
OUT_FRACTIONS = BATCH_DIR / "b_gtex_fractions.csv"
OUT_AGE_PRIMARY = BATCH_DIR / "b_age_regression_primary.csv"
OUT_AGE_TERTILE = BATCH_DIR / "b_age_regression_tertile.csv"
OUT_B1S = BATCH_DIR / "b_b1s_fraction_adjusted.csv"
OUT_SUMMARY = BATCH_DIR / "b_summary.json"
OUT_STDOUT_LOG = LOG_DIR / "b_stdout.log"

# --------------------------------------------------------------------------- #
# Constants.
# --------------------------------------------------------------------------- #
SEED = 42  # WHY: reproducibility — all bootstraps/RNGs derive from this.
BOOTSTRAP_B = 500  # WHY: brief spec; B=500 gives 95% CI percentile SE < 0.02.
BOOTSTRAP_GENE_FRAC = 0.80  # WHY: brief — 80% gene resample without replacement.
SMOKE_N = 10  # WHY: brief — sanity check first 10 samples.
SMOKE_B = 50
CHUNK_SIZE = 100  # WHY: brief OOM mitigation — chunk GTEx samples.

# TF panel from B1s spec.
TF_PANEL = ["JUNB", "FOS", "EGR1", "EGR2", "CEBPB", "ATF3", "KLF10", "CDKN1A"]
# SASP12 canonical panel (Coppe 2010 [lit_doi_10.1146_annurev-pathol-121808-102144]
# core SASP; Basisty 2020 [lit_doi_10.1371_journal.pbio.3000599] extended).
# Exact members per batch_052 convention.
SASP12 = [
    "IL6", "IL8", "CXCL1", "CXCL2", "CCL2", "CCL20",
    "MMP1", "MMP3", "SERPINE1", "TIMP1", "ICAM1", "TNFSF10",
]

# Gene-symbol prefix exclusion list (WHY: brief — protein-coding focus by symbol
# in absence of a pre-loaded GTF annotation; ribo/mito/HLA/sex-chrom dominate
# variance and are known deconvolution confounders per Donovan 2020
# [lit_doi_10.1038_s41467-020-14561-0] QC section).
EXCLUDE_PREFIXES = ("RPS", "RPL", "MT-", "HLA-", "MRPS", "MRPL")
EXCLUDE_EXACT = {"XIST", "TSIX", "MALAT1", "NEAT1"}

# Primary compartments for B1s (vascular, FAP, MuSC per PI thesis).
B1S_COMPARTMENTS = ["Vascular", "FAP", "MuSC"]

COMPARTMENTS = list(HLMA_FILES.keys())  # order fixes column ordering everywhere.

# --------------------------------------------------------------------------- #
# FAP marker-gene fallback panel.
# --------------------------------------------------------------------------- #
# WHY a marker-based fallback is required (Iter 064 resolution):
#   OMIX004308-02 (the only FAP h5ad in the HLMA bundle) stores SCALED data
#   (per-gene z-scored, clipped to [-5, 10]) in BOTH .X and .raw.X. No raw
#   counts or log1p-normalized-only matrix is available; OMIX004308-03 is
#   truncated on disk (reader fails with "Unable to synchronously open file:
#   eof=4.93e9, stored_eof=8.37e9"); OMIX004308-01.tar.gz contains only the
#   three non-FAP compartment h5ads (verified 2026-04-22); OMIX004308-05
#   contains the MYOFIBER compartment (Type I / Type II / MTJ / NMJ labels),
#   not FAPs. ReLU-clipping the scaled FAP matrix collapses the FAP signature
#   to near-zero for most genes (row-sums go negative), yielding a degenerate
#   signature column that produces FAP fraction = 0 in the smoke test (B=50,
#   N=10 -> FAP mean = 0.000; abort gate fires at min<0.005).
#
# FALLBACK DESIGN (option 3 in the brief):
#   Build the FAP signature row from a CURATED MARKER PANEL, using
#   per-marker-gene ReLU(scaled-expression) mean across FAP-labeled cells.
#   WHY ReLU on scaled data is valid here:
#     - Scaled expression is z-score-like: cells that express the marker
#       ABOVE the gene's mean have positive values, below-mean cells have
#       negative values. ReLU keeps only the "positively expressing"
#       contribution, then averaging gives a non-negative intensity that
#       preserves per-gene rank structure on the FAP subpopulation.
#     - Because we restrict to CANONICAL FAP markers (highly expressed in
#       FAPs relative to other compartments per HLMA 2025 and Schwalie 2018),
#       ReLU-mean concentrates signal where the biology says it should.
#     - We then L1-normalize the FAP vector to match the median L1 norm of
#       the Vascular/MuSC/Immune CPM-like signatures (1e6 scale by
#       construction), so NNLS sees a comparably-scaled FAP column.
#   WHY exclude Tenocyte cells: OMIX004308-02 contains FAP + Tenocyte
#   populations (Annotation in {'MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP',
#   'CD99+ FAP', 'RUNX2+ FAP', 'Tenocyte'}); Tenocytes are a distinct
#   mesenchymal population (SCX+, MKX+) and diluting the FAP signature with
#   them contaminates the marker intensity estimate.
#
# MARKER SOURCES (cite every weight):
#   - PDGFRA: canonical pan-FAP marker (Uezumi 2010, Joe 2010,
#     [lit_doi_10.1038_s41586-024-07348-6] HLMA — near-universal FAP marker).
#     WEIGHT 3.0 (strongest single marker).
#   - CD34, DCN, LUM, COL6A1/2/3, MFAP5, SERPINF1: core stromal/ECM markers
#     enriched in FAPs per HLMA 2025 compartment definition and Schwalie
#     2018 Nature FAP subpopulation panel. WEIGHT 1.0 each.
#   - DLK1, THY1: FAP subpopulation markers (Malecova 2018, Giuliani 2021).
#     WEIGHT 1.0 each.
# WHY this panel (10 genes + PDGFRA 3x-weighted): broad enough to survive
# log-space expression filter even when 1-2 markers are zeroed by
# post-QC filtering, narrow enough to not admit non-FAP ECM genes from
# vascular/immune compartments.
FAP_MARKER_GENES: Dict[str, float] = {
    "PDGFRA": 3.0,
    "CD34": 1.0,
    "DCN": 1.0,
    "LUM": 1.0,
    "DLK1": 1.0,
    "COL6A1": 1.0,
    "COL6A2": 1.0,
    "COL6A3": 1.0,
    "THY1": 1.0,
    "MFAP5": 1.0,
    "SERPINF1": 1.0,
}

# FAP cell-type labels in OMIX004308-02.obs['Annotation']. Tenocyte excluded.
FAP_ANNOTATION_KEEP = {
    "MME+ FAP", "CD55+ FAP", "GPC3+ FAP", "CD99+ FAP", "RUNX2+ FAP",
}

# --------------------------------------------------------------------------- #
# Logging.
# --------------------------------------------------------------------------- #
logger = logging.getLogger("run_b_nnls")
logger.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_file_h = logging.FileHandler(OUT_STDOUT_LOG, mode="w")
_file_h.setFormatter(_formatter)
_stream_h = logging.StreamHandler(sys.stdout)
_stream_h.setFormatter(_formatter)
logger.addHandler(_file_h)
logger.addHandler(_stream_h)


# --------------------------------------------------------------------------- #
# Utility.
# --------------------------------------------------------------------------- #
def _elapsed(t0: float) -> str:
    dt = time.time() - t0
    return f"{dt:.1f}s"


def _set_seeds(seed: int = SEED) -> None:
    """WHY: Rule 0 reproducibility — pin Python/NumPy entropy."""
    import random
    random.seed(seed)
    np.random.seed(seed)


def _gene_filter_mask(symbols: Iterable[str]) -> np.ndarray:
    """Return a boolean mask True for genes to KEEP.

    WHY: exclude ribo/mito/HLA/sex-chrom by symbol; brief spec.
    """
    arr = np.asarray(list(symbols), dtype=object)
    mask = np.ones(arr.shape[0], dtype=bool)
    for p in EXCLUDE_PREFIXES:
        mask &= ~np.char.startswith(arr.astype(str), p)
    for g in EXCLUDE_EXACT:
        mask &= arr != g
    return mask


# --------------------------------------------------------------------------- #
# Step 1 — HLMA compartment signature.
# --------------------------------------------------------------------------- #
@dataclass
class SignatureBuildReport:
    """Per-compartment build telemetry (for b_summary.json)."""
    compartment: str
    n_cells: int
    n_genes_in: int
    donor_col_used: Optional[str]
    n_donors: int
    x_matrix_source: str  # 'raw.X', '.X', 'raw.X[clipped]', '.X[clipped]',
                          # or 'marker_panel_ReLU_mean' for FAP fallback.
    x_min: float
    x_max: float
    sample_row_sum: float
    negative_clip_applied: bool = False  # True iff we ReLU'd negatives
    x_min_pre_clip: float = 0.0  # min observed in probe before clipping
    marker_based: bool = False  # True iff this compartment uses the curated
                                # FAP-marker fallback (option 3).
    marker_genes_used: List[str] = field(default_factory=list)
    marker_genes_missing: List[str] = field(default_factory=list)


def _probe_matrix(X, n_probe: int = 2000) -> Tuple[float, float]:
    """Return (min_value, sum_total) from a row-probe of X.

    WHY: some HLMA files have scaled (negative-containing) matrices in BOTH
    .X and .raw.X. We need to choose the matrix that (a) has NO negatives
    and (b) has a positive total sum. A small row-probe is sufficient to
    detect the presence of negatives because scaling is cell-wise.

    Implementation note: `a.raw.X` from a backed anndata is neither a plain
    numpy array nor always detected as sparse by `scipy.sparse.issparse`
    (it may be an anndata SparseDataset wrapper). We normalize by slicing
    first (which materializes a concrete sparse or dense block) and then
    branching on sparsity of the realized block.
    """
    n_rows = X.shape[0]
    end = min(n_probe, n_rows)
    block = X[0:end, :]
    # After slicing, a SparseDataset returns a concrete csr/csc matrix.
    if sp.issparse(block):
        if block.nnz > 0:
            mn = float(block.data.min())
        else:
            mn = 0.0
        total = float(np.asarray(block.sum()).ravel()[0])
    else:
        block = np.asarray(block)
        mn = float(block.min()) if block.size else 0.0
        total = float(block.sum())
    return mn, total


def _choose_matrix(
    a: ad.AnnData,
) -> Tuple[str, "sp.spmatrix | np.ndarray", bool, float]:
    """Pick the matrix to sum over.

    Returns: (src_name, X, needs_clip, probe_min_pre_clip).

    WHY (audit C1 fix): .raw.X is NOT always preferable. The FAP file
    (OMIX004308-02) has scaled (negative-containing) values in both .X and
    .raw.X; blindly preferring .raw.X would propagate negatives through
    colsum, yield non-CPM normalization, and silently emit NaN at the
    downstream log2(x+1) step (values in [-1, 0) log to NaN).

    Decision rule: inspect BOTH matrices on a row-probe; pick the one with
    min>=0 AND positive total sum. Prefer .raw.X when both qualify (canonical
    location for counts). If NEITHER qualifies, pick whichever has the larger
    positive-sum AND apply ReLU clipping (needs_clip=True) with a WARN. This
    is documented (NOT hidden) in the per-compartment report. Rule 0: the
    clip is lossy; we must surface it so reviewers can interpret the FAP
    signature under the "scaled-data ReLU" caveat.
    """
    candidates: List[Tuple[str, "sp.spmatrix | np.ndarray", float, float]] = []
    if a.raw is not None:
        mn, tot = _probe_matrix(a.raw.X)
        candidates.append(("raw.X", a.raw.X, mn, tot))
    mn, tot = _probe_matrix(a.X)
    candidates.append((".X", a.X, mn, tot))

    # Prefer in order: non-negative AND positive sum, then non-negative.
    valid = [c for c in candidates if c[2] >= 0.0 and c[3] > 0.0]
    if valid:
        name, X, probe_min, _tot = valid[0]
        return name, X, False, probe_min
    nonneg = [c for c in candidates if c[2] >= 0.0]
    if nonneg:
        name, X, probe_min, tot = nonneg[0]
        logger.warning("[signature] %s chosen with non-positive total sum=%s",
                       name, tot)
        return name, X, False, probe_min
    # No non-negative matrix available: apply ReLU clipping.
    diag = "; ".join(f"{n}: min={m:.3g} sum={t:.3g}" for n, _, m, t in candidates)
    # Choose the candidate with the least-negative minimum (closest to
    # non-negative); if tied, prefer the one whose raw data is more canonical
    # (raw.X appears first in candidates).
    candidates_sorted = sorted(candidates, key=lambda c: -c[2])
    name, X, probe_min, tot = candidates_sorted[0]
    logger.warning(
        "[signature] NO non-negative matrix found (%s). Falling back to "
        "ReLU clipping of %s (probe_min=%.3g). Signature will be lossy "
        "but documented in b_hlma_signature_source.csv.",
        diag, name, probe_min,
    )
    return name + "[clipped]", X, True, probe_min


def _pseudobulk_compartment(
    path: Path, compartment_name: str
) -> Tuple[pd.Series, SignatureBuildReport]:
    """Sum the expression matrix across all cells in the file.

    WHY sum across all cells: each h5ad is already compartment-specific
    (Vascular/MuSC/FAP-Tenocyte/Immune), so pooling all cells yields a single
    pseudobulk expression vector per compartment. Brief step 1 explicitly
    allows this because "per-compartment signature averaged across donors is
    fine for NNLS."

    Returns normalized signature vector (sum to 1e6, like CPM) indexed by
    gene symbol.
    """
    logger.info("[signature] Loading %s (backed='r')...", path.name)
    t0 = time.time()
    a = ad.read_h5ad(path, backed="r")
    try:
        src_name, X, needs_clip, probe_min_pre_clip = _choose_matrix(a)
        n_cells, n_genes = X.shape
        # Derive donor count (cosmetic — for audit only).
        donor_col = None
        for c in ("sample", "orig.ident", "donor", "patient"):
            if c in a.obs.columns:
                donor_col = c
                break
        n_donors = int(a.obs[donor_col].nunique()) if donor_col else -1

        # Sum over cells in one pass with chunking (robust for both backed
        # sparse and dense; issparse(X) may return False for backed
        # SparseDataset wrappers, so we detect sparsity post-slice).
        # If needs_clip is True, apply ReLU per chunk before summing (audit
        # C1 fix: keeps colsum >= 0 and prevents NaN at log step).
        colsum = np.zeros(n_genes, dtype=np.float64)
        chunk = 5000
        for start in range(0, n_cells, chunk):
            end = min(start + chunk, n_cells)
            block = X[start:end, :]
            if sp.issparse(block):
                if needs_clip:
                    # Materialize only non-negative entries: set negatives to 0.
                    block = block.copy()
                    block.data = np.maximum(block.data, 0.0)
                    block.eliminate_zeros()
                colsum += np.asarray(block.sum(axis=0)).ravel()
            else:
                arr = np.asarray(block)
                if needs_clip:
                    arr = np.maximum(arr, 0.0)
                colsum += arr.sum(axis=0)

        # Sample diagnostic row sum for audit (from first cell).
        first_row = X[0:1, :]
        if sp.issparse(first_row):
            sample_row_sum = float(np.asarray(first_row.sum(axis=1)).ravel()[0])
        else:
            sample_row_sum = float(np.asarray(first_row).sum())

        # Audit C1 fix: ensure non-negative colsum before normalization.
        # After optional ReLU clipping this must hold (modulo tiny fp noise);
        # we assert to catch any future regression.
        assert np.all(colsum >= -1e-9), (
            f"Compartment {compartment_name}: colsum contains negatives "
            f"(min={colsum.min():.3e}); ReLU clip should have prevented this."
        )
        colsum = np.maximum(colsum, 0.0)  # clamp tiny negative fp noise

        # Normalize to CPM-like scale (sum to 1e6). WHY: per brief step 1
        # "Normalize to mean TPM-like". Because the input is already log1p-
        # normalized per cell, this renormalization yields a relative
        # expression signature valid for NNLS rank-comparison across
        # compartments.
        total = colsum.sum()
        if total <= 0:
            raise RuntimeError(
                f"Compartment {compartment_name}: colsum total is <= 0 "
                f"after optional clipping. Cannot build CPM-like signature."
            )
        sig_cpm = colsum / total * 1e6
        assert np.all(sig_cpm >= 0), "negative values survived preprocessing"

        # var gene symbols: prefer 'features' col if present, else var_names.
        if "features" in a.var.columns:
            gene_symbols = a.var["features"].astype(str).values
        else:
            gene_symbols = np.asarray(a.var_names, dtype=str)

        ser = pd.Series(sig_cpm, index=gene_symbols, dtype=np.float64)
        # If duplicate symbols exist (common in scRNA refs), aggregate by sum.
        # WHY: duplicates typically reflect multiple Ensembl IDs collapsing to
        # same symbol; summing preserves total expression for that symbol.
        if not ser.index.is_unique:
            ser = ser.groupby(ser.index).sum()

        # Diagnostic min/max from a 100-row probe; use the same sparse-safe
        # pattern as _probe_matrix to avoid csr_matrix .min() type errors.
        probe = X[0:100, :]
        if sp.issparse(probe):
            probe_mn = float(probe.data.min()) if probe.nnz > 0 else 0.0
            probe_mx = float(probe.data.max()) if probe.nnz > 0 else 0.0
        else:
            probe_arr = np.asarray(probe)
            probe_mn = float(probe_arr.min()) if probe_arr.size else 0.0
            probe_mx = float(probe_arr.max()) if probe_arr.size else 0.0
        report = SignatureBuildReport(
            compartment=compartment_name,
            n_cells=int(n_cells),
            n_genes_in=int(n_genes),
            donor_col_used=donor_col,
            n_donors=n_donors,
            x_matrix_source=src_name,
            x_min=probe_mn,
            x_max=probe_mx,
            sample_row_sum=sample_row_sum,
            negative_clip_applied=bool(needs_clip),
            x_min_pre_clip=float(probe_min_pre_clip),
        )
        logger.info(
            "[signature] %s: n_cells=%d n_genes=%d donors=%d src=%s (%s)",
            compartment_name, n_cells, n_genes, n_donors, src_name,
            _elapsed(t0),
        )
        return ser, report
    finally:
        a.file.close()


def _build_fap_marker_signature(
    path: Path,
    target_l1_norm: float,
) -> Tuple[pd.Series, SignatureBuildReport]:
    """Option-3 fallback FAP signature from a curated marker panel.

    Rationale: OMIX004308-02 stores scaled (z-score-like) data with negatives
    in both .X and .raw.X; no usable raw-count FAP matrix exists in the local
    HLMA bundle (see FAP_MARKER_GENES docstring above). We therefore build the
    FAP column from canonical markers using per-gene ReLU(scaled)-mean over
    FAP-labeled cells (Tenocytes excluded).

    Steps (each with WHY):
      1) Open backed; subset obs to FAP_ANNOTATION_KEEP (drop Tenocytes —
         they are a distinct mesenchymal population and would dilute
         FAP-specific marker intensity).
      2) For each marker gene in FAP_MARKER_GENES, slice the column from
         .X (scaled data), compute mean of max(x, 0) across FAP cells.
         WHY: ReLU preserves per-gene rank (cells with above-mean expression
         contribute) while keeping values non-negative so the downstream
         log2(x+1) step cannot produce NaN (Audit C1 invariant).
      3) Multiply by per-marker weight (PDGFRA = 3.0, others 1.0) to upweight
         the strongest canonical FAP marker (Uezumi 2010, Joe 2010).
      4) Build a zero-initialized gene Series over the full var_names index
         and fill in the marker values. All non-marker genes are 0 -- this
         is WHY the fallback is weaker but defensible: NNLS sees FAP as a
         panel-restricted "anchor" column.
      5) L1-normalize the non-zero entries so sum(FAP_row) == target_l1_norm
         (the CPM-like 1e6 scale used by the other compartments). WHY: keeps
         FAP on a comparable magnitude scale with Vascular/MuSC/Immune in
         NNLS, so the non-negative least squares fit does not trivially
         zero-out FAP on a scale mismatch.

    Parameters
    ----------
    path : Path
        OMIX004308-02.h5ad.
    target_l1_norm : float
        Target sum for the FAP signature row (pre-log-transform). Use the
        CPM-like 1e6 scale for parity with the other compartments.

    Returns
    -------
    ser : pd.Series
        gene_symbol -> signature value (most entries 0).
    report : SignatureBuildReport
        marker_based=True, records which markers were found/missing.
    """
    logger.info("[signature] Building FAP marker-based fallback from %s ...",
                path.name)
    t0 = time.time()
    a = ad.read_h5ad(path, backed="r")
    try:
        # Resolve gene symbols in the h5ad.
        if "features" in a.var.columns:
            gene_symbols = a.var["features"].astype(str).values
        else:
            gene_symbols = np.asarray(a.var_names, dtype=str)
        gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}

        # Subset obs to FAP-labeled cells (drop Tenocytes).
        if "Annotation" not in a.obs.columns:
            raise RuntimeError(
                f"[signature] FAP marker path requires 'Annotation' column "
                f"in {path.name}.obs; present cols: {list(a.obs.columns)}"
            )
        ann = a.obs["Annotation"].astype(str).values
        fap_mask = np.isin(ann, list(FAP_ANNOTATION_KEEP))
        n_fap = int(fap_mask.sum())
        if n_fap < 100:
            raise RuntimeError(
                f"[signature] FAP mask has only {n_fap} cells; refuse to "
                f"build a signature from <100 cells. Check FAP_ANNOTATION_KEEP."
            )
        fap_row_idx = np.where(fap_mask)[0]
        logger.info("[signature] FAP marker path: N_fap_cells=%d (of %d total; "
                    "Tenocytes excluded)", n_fap, len(ann))

        # Donor count (audit only).
        donor_col = None
        for c in ("sample", "orig.ident", "donor", "patient"):
            if c in a.obs.columns:
                donor_col = c
                break
        if donor_col is not None:
            n_donors = int(
                pd.Series(a.obs[donor_col].values[fap_mask]).nunique()
            )
        else:
            n_donors = -1

        # X matrix. .X and .raw.X are the same scaled data per the earlier
        # probe; use .X (slightly faster path in some AnnData versions since
        # no .raw indirection).
        X = a.X

        # For each marker: compute ReLU-mean across FAP cells.
        markers_found: List[str] = []
        markers_missing: List[str] = []
        marker_values: Dict[str, float] = {}

        # Slicing backed sparse by row-list is supported; we materialize the
        # FAP subset once per marker column because column-slicing inside an
        # anndata SparseDataset requires a dense probe. To avoid repeated IO,
        # we fetch FAP rows as a dense subset only ONCE and slice in RAM.
        # Memory estimate: n_fap ~ 37000, n_genes = 37841, float32 -> ~5.2 GB
        # if we densified ALL genes. We instead slice only the MARKER columns
        # post-load which keeps memory to n_fap * n_markers * 4 ~ 1.6 MB.
        # We must load FAP rows once as sparse, then densify per column.
        logger.info("[signature] Loading FAP cells as sparse block (n=%d)...",
                    n_fap)
        fap_block = X[fap_row_idx, :]  # backed indexing -> concrete sparse
        if not sp.issparse(fap_block):
            fap_block = sp.csr_matrix(fap_block)
        fap_block = fap_block.tocsc()  # fast column slicing

        # Diagnostics for report.
        probe_mn = float(fap_block.data.min()) if fap_block.nnz > 0 else 0.0
        probe_mx = float(fap_block.data.max()) if fap_block.nnz > 0 else 0.0
        sample_row_sum = float(
            np.asarray(fap_block[0:1, :].sum(axis=1)).ravel()[0]
        )

        for gene, weight in FAP_MARKER_GENES.items():
            idx = gene_to_idx.get(gene)
            if idx is None:
                markers_missing.append(gene)
                continue
            col = fap_block[:, idx]
            col_dense = col.toarray().ravel() if sp.issparse(col) else \
                np.asarray(col).ravel()
            # ReLU then mean across FAP cells.
            relu_mean = float(np.maximum(col_dense, 0.0).mean())
            marker_values[gene] = float(weight) * relu_mean
            markers_found.append(gene)

        if len(markers_found) == 0:
            raise RuntimeError(
                "[signature] FAP marker fallback: ZERO markers found in "
                "gene symbols. Check FAP_MARKER_GENES vs h5ad var["
                "'features']."
            )

        logger.info(
            "[signature] FAP marker found=%d missing=%d; missing=%s",
            len(markers_found), len(markers_missing), markers_missing,
        )

        # Assemble signature. Start zero, fill markers.
        sig_vec = np.zeros(len(gene_symbols), dtype=np.float64)
        for g, v in marker_values.items():
            idx = gene_to_idx[g]
            sig_vec[idx] = v

        # Non-negative guard.
        assert np.all(sig_vec >= 0), (
            "[signature] FAP marker path produced negatives; ReLU skipped?"
        )

        # L1-normalize the nonzero entries to match the target norm used by
        # the other compartments (1e6 CPM-like scale).
        current_l1 = float(sig_vec.sum())
        if current_l1 <= 0:
            raise RuntimeError(
                "[signature] FAP marker L1 norm is zero — all markers had "
                "ReLU-mean = 0. This indicates either wrong annotation "
                "filtering or a catastrophic scale-data artifact."
            )
        sig_vec = sig_vec * (target_l1_norm / current_l1)

        ser = pd.Series(sig_vec, index=gene_symbols, dtype=np.float64)
        # Duplicate-symbol aggregation (same policy as _pseudobulk_compartment).
        if not ser.index.is_unique:
            ser = ser.groupby(ser.index).sum()

        report = SignatureBuildReport(
            compartment="FAP",
            n_cells=n_fap,
            n_genes_in=int(len(gene_symbols)),
            donor_col_used=donor_col,
            n_donors=n_donors,
            x_matrix_source="marker_panel_ReLU_mean",
            x_min=probe_mn,
            x_max=probe_mx,
            sample_row_sum=sample_row_sum,
            negative_clip_applied=True,  # ReLU is applied to scaled data
            x_min_pre_clip=probe_mn,
            marker_based=True,
            marker_genes_used=markers_found,
            marker_genes_missing=markers_missing,
        )
        logger.info(
            "[signature] FAP marker-based done: n_cells=%d n_markers=%d "
            "donors=%d L1=%.3e -> %.3e (%s)",
            n_fap, len(markers_found), n_donors, current_l1, target_l1_norm,
            _elapsed(t0),
        )
        return ser, report
    finally:
        a.file.close()


def build_hlma_signature() -> Tuple[pd.DataFrame, List[SignatureBuildReport]]:
    """Build the compartment x gene signature matrix.

    WHY: NNLS needs a single canonical expression vector per compartment.
    Gene filter: mean normalized-count > 1 across compartments AND not in
    exclude list (ribo/mito/HLA/sex-chrom).

    FAP compartment ALWAYS uses the marker-gene fallback in iter 064 because
    the only FAP h5ad (OMIX004308-02) stores scaled data with negatives in
    BOTH .X and .raw.X and no usable raw-count FAP matrix is available
    locally (OMIX004308-03 is truncated; OMIX004308-05 is the myofiber
    compartment, not FAPs). Documented in b_summary.json under
    'fap_signature_method'. Iter 065 should replace with BayesPrism (R) or
    external FAP raw counts.

    Returns:
        signature: DataFrame (compartments rows x filtered genes cols),
        reports: per-compartment build telemetry.
    """
    reports: List[SignatureBuildReport] = []
    per_comp: Dict[str, pd.Series] = {}

    # Build non-FAP compartments first so we can compute a reference L1 norm
    # to which the FAP marker vector can be normalized. WHY: the per-
    # compartment pseudobulk normalizes colsum to 1e6 CPM-like scale already;
    # using that exact same target keeps FAP comparable under NNLS.
    non_fap_l1: List[float] = []
    for comp, path in HLMA_FILES.items():
        if comp == "FAP":
            continue
        ser, rep = _pseudobulk_compartment(path, comp)
        per_comp[comp] = ser
        reports.append(rep)
        non_fap_l1.append(float(ser.sum()))

    # Target L1 for FAP: median of non-FAP L1 norms (which equal 1e6 by
    # construction, but compute defensively in case upstream changes).
    target_l1 = float(np.median(non_fap_l1)) if non_fap_l1 else 1e6
    logger.info("[signature] FAP fallback target L1 = %.3e "
                "(median of non-FAP L1 norms)", target_l1)
    fap_ser, fap_rep = _build_fap_marker_signature(
        HLMA_FILES["FAP"], target_l1_norm=target_l1
    )
    per_comp["FAP"] = fap_ser
    reports.append(fap_rep)
    # Reorder reports to match COMPARTMENTS for stable CSV/JSON layout
    # (auditors expect the fixed iteration order).
    by_name = {r.compartment: r for r in reports}
    reports = [by_name[c] for c in COMPARTMENTS if c in by_name]

    # Align on common gene symbol union; missing genes filled with 0.
    sig = pd.DataFrame(per_comp).fillna(0.0)  # index=gene_symbol, cols=comp
    sig = sig.T  # compartments x genes
    logger.info(
        "[signature] Pre-filter signature shape: %s (compartments x genes)",
        sig.shape,
    )

    # Filter gene-symbol prefix exclusions.
    mask_sym = _gene_filter_mask(sig.columns)
    sig = sig.loc[:, mask_sym]
    n_after_symbol = sig.shape[1]

    # AUDIT C2 FIX: original `mean_cpm > 1.0` filter was inert post-1e6
    # normalization (mean across 4 compartments ~ 1e6/n_genes ~ 33 for ~30k
    # genes, so every gene was retained). Replace with an explicit log-space
    # threshold: after log2(x+1), require mean > 0.5 across compartments.
    # WHY 0.5 in log-space: matches the brief's intent of "min expression
    # threshold" in the log-domain where NNLS actually operates; log2(x+1)>0.5
    # corresponds to CPM-raw >= ~0.414, i.e., genes detectably expressed
    # above near-zero baseline in at least part of the panel.
    log_sig = np.log2(sig.values + 1.0)
    mean_log = log_sig.mean(axis=0)
    log_thresh = 0.5
    keep_mask = mean_log > log_thresh
    n_kept = int(keep_mask.sum())
    logger.warning(
        "[signature] Log-space mean log2(x+1) > %.2f filter: kept %d / %d "
        "genes (after symbol-prefix exclusion). Prior `mean_cpm>1` filter "
        "was inert post-1e6 normalization; this fix applies the intended "
        "min-expression cutoff in log-space.",
        log_thresh, n_kept, n_after_symbol,
    )
    sig = sig.loc[:, keep_mask]
    logger.info("[signature] Post-filter signature shape: %s", sig.shape)

    # Attach metadata for downstream summary.
    sig.attrs["log_space_threshold"] = float(log_thresh)
    sig.attrs["n_genes_after_symbol_filter"] = int(n_after_symbol)
    sig.attrs["n_genes_after_expression_filter"] = int(n_kept)
    return sig, reports


# --------------------------------------------------------------------------- #
# Step 2 — GTEx bulk prep.
# --------------------------------------------------------------------------- #
def _parse_gct(path: Path) -> pd.DataFrame:
    """Parse a GTEx .gct.gz file.

    WHY: GTEx v8 TPM/counts are distributed as GCT v1.3; first line is the
    format tag (#1.2 or #1.3), second line is '<nrow>\\t<ncol>', third line
    is the header row with 'Name\\tDescription\\t<sample_ids>...'.
    We skip 2 header lines and use pandas read_csv.
    """
    logger.info("[gtex] Parsing %s ...", path.name)
    t0 = time.time()
    # Skip first 2 lines (#1.x and dims).
    df = pd.read_csv(path, sep="\t", skiprows=2, low_memory=False,
                     compression="gzip")
    # GCT layout resolution. WHY this fix (smoke-test surfaced bug): the
    # previous code mislabeled GCT v1.3's `Name` column as gene_symbol and
    # dropped `Description` — but in GTEx v8 GCT files, `Name` holds the
    # ENSG Ensembl ID and `Description` holds the HGNC gene symbol. The
    # mislabel produced zero intersection with HLMA symbols.
    col_names = list(df.columns)
    if "Name" in col_names and "Description" in col_names:
        # v1.3 layout: id, Name (=ENSG), Description (=symbol), samples...
        # v1.2 layout:      Name (=ENSG), Description (=symbol), samples...
        rename_map = {"Name": "gene_id", "Description": "gene_symbol"}
        if "id" in col_names:
            rename_map["id"] = "row_id"
        df = df.rename(columns=rename_map)
        # Drop incidental `row_id` to keep downstream schema simple.
        if "row_id" in df.columns:
            df = df.drop(columns=["row_id"])
    else:
        # Fallback: assume first col is id-like, second col is symbol.
        df = df.rename(columns={col_names[0]: "gene_id",
                                col_names[1]: "gene_symbol"})
    logger.info("[gtex] Parsed shape=%s cols=%s... (%s)", df.shape,
                list(df.columns[:5]), _elapsed(t0))
    return df


def _age_bin_to_midpoint(bin_str: str) -> float:
    """'60-69' -> 64.5. WHY: brief uses integer midpoint (25, 35, ..., 75);
    we use the true midpoint of the bin (24.5, 34.5, ...) for precision.

    Brief spec said 25/35/45/55/65/75 — we honor that exactly (not 24.5).
    """
    mapping = {
        "20-29": 25.0, "30-39": 35.0, "40-49": 45.0,
        "50-59": 55.0, "60-69": 65.0, "70-79": 75.0,
    }
    return mapping.get(str(bin_str).strip(), np.nan)


def load_gtex_bulk() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load GTEx bulk TPM + sample attrs + subject phenotypes; join and QC.

    Returns:
        tpm:     (genes x samples) DataFrame, gene index = gene_symbol,
                 duplicate symbols aggregated by sum.
        meta:    per-sample metadata joined with subject phenotypes.
        qc_log:  {'n_before', 'n_after', 'n_after_rin', 'n_after_tissue'}
    """
    tpm_df = _parse_gct(GTEX_TPM_GCT)
    attr = pd.read_csv(GTEX_SAMPLE_ATTR, sep="\t", low_memory=False)
    pheno = pd.read_csv(GTEX_SUBJ_PHENO, sep="\t", low_memory=False)

    # Subject ID extraction: SAMPID format 'GTEX-XXXXX-...' -> SUBJID 'GTEX-XXXXX'.
    attr["SUBJID"] = attr["SAMPID"].str.split("-").str[:2].str.join("-")
    meta = attr.merge(pheno, on="SUBJID", how="left")

    # Build gene-symbol TPM matrix.
    sample_cols = [c for c in tpm_df.columns
                   if c not in ("gene_id", "gene_symbol")]
    tpm = (tpm_df.groupby("gene_symbol")[sample_cols].sum())  # aggregate dups
    logger.info("[gtex] TPM gene-symbol matrix shape: %s", tpm.shape)

    # Subset metadata to sample_cols present in TPM (alignment check).
    meta = meta[meta["SAMPID"].isin(sample_cols)].copy()
    n_before = len(meta)

    # Filter to skeletal muscle tissue.
    meta = meta[meta["SMTSD"] == "Muscle - Skeletal"].copy()
    n_after_tissue = len(meta)

    # RIN >= 6 QC.
    meta["SMRIN"] = pd.to_numeric(meta["SMRIN"], errors="coerce")
    meta = meta[meta["SMRIN"] >= 6.0].copy()
    n_after_rin = len(meta)

    # Age bin midpoint.
    meta["age_midpoint"] = meta["AGE"].map(_age_bin_to_midpoint)
    meta = meta[meta["age_midpoint"].notna()].copy()

    # Ischemic time.
    meta["SMTSISCH"] = pd.to_numeric(meta["SMTSISCH"], errors="coerce")

    # Sex as categorical: 1=Male, 2=Female per GTEx codebook.
    meta["SEX"] = meta["SEX"].astype("Int64").astype(str)

    # Collection site proxy. SMCENTER is GTEx center code (B1, C1, etc.).
    # WHY (surprise #4 in module docstring): COLLECTION_SITE column is NOT
    # present in public GTEx v8 SampleAttributes. SMCENTER is the closest
    # available proxy — it identifies the collection center.
    meta["SMCENTER"] = meta["SMCENTER"].fillna("UNK").astype(str)

    # DTHHRDY (death-scale 0-4) retained as numeric-then-string for dummy.
    meta["DTHHRDY"] = meta["DTHHRDY"].fillna(-1).astype(int).astype(str)

    # Align TPM columns to meta['SAMPID'].
    kept_samples = meta["SAMPID"].tolist()
    tpm = tpm[kept_samples]

    qc_log = pd.DataFrame([{
        "n_before_qc": int(n_before),
        "n_after_tissue_filter": int(n_after_tissue),
        "n_after_rin_filter": int(n_after_rin),
        "n_after_age_filter": int(len(meta)),
        "n_final": int(len(meta)),
    }])
    logger.info("[gtex] QC: %s", qc_log.to_dict(orient="records")[0])
    return tpm, meta, qc_log


# --------------------------------------------------------------------------- #
# Step 3 — NNLS deconvolution.
# --------------------------------------------------------------------------- #
def _prepare_nnls_matrices(
    signature: pd.DataFrame, tpm: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Intersect genes, log-transform, return S (genes x compartments) and
    B (genes x samples).

    WHY log2(TPM+1): Donovan 2020 recommends log-transform for NNLS to
    stabilize variance across dynamic range; pseudocount 1 is standard.
    """
    common = signature.columns.intersection(tpm.index)
    logger.info("[nnls] %d genes common to signature and bulk.", len(common))
    sig_sub = signature.loc[:, common]  # compartments x genes
    bulk_sub = tpm.loc[common, :]        # genes x samples

    # AUDIT C1 FIX: explicit assertion before log-transform. Negatives would
    # silently produce NaN via log2(x+1) for x in [-1, 0) or raise invalid
    # warnings for x < -1. Guard against any preprocessing regression.
    assert np.all(sig_sub.values >= 0), (
        "negative values survived preprocessing in signature"
    )
    assert np.all(bulk_sub.values >= 0), (
        "negative values survived preprocessing in GTEx TPM"
    )

    # Log-transform both.
    S = np.log2(sig_sub.values + 1.0).T  # genes x compartments
    B = np.log2(bulk_sub.values + 1.0)   # genes x samples
    assert not np.any(np.isnan(S)), "NaN in S after log transform"
    assert not np.any(np.isnan(B)), "NaN in B after log transform"
    gene_order = list(common)
    return S, B, gene_order


def _nnls_one(
    S: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Run NNLS for one bulk sample.

    Returns (fractions_sum1, residual_l2, residual_ratio).
    residual_ratio = median(|resid_per_gene|) / median(|b|); used to flag
    poor fits (brief threshold 0.5).

    WHY median ratio: robust to outlier genes; brief-spec early-fail criterion.
    """
    # scipy.optimize.nnls minimizes ||S x - b||_2 subject to x >= 0.
    # To keep signatures on comparable scales, we do not pre-scale beyond
    # the log transform.
    x, _rnorm = nnls(S, b, maxiter=2000)
    total = x.sum()
    if total > 0:
        frac = x / total
    else:
        frac = np.zeros_like(x)
    resid = b - S @ x
    med_abs_resid = float(np.median(np.abs(resid)))
    med_abs_b = float(np.median(np.abs(b)))
    ratio = med_abs_resid / max(med_abs_b, 1e-9)
    return frac, float(np.linalg.norm(resid)), ratio


def deconvolve_all(
    S: np.ndarray,
    B: np.ndarray,
    sample_ids: List[str],
    compartments: List[str],
    bootstrap_b: int = BOOTSTRAP_B,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Run NNLS + bootstrap CI for every sample in B.

    Bootstrap design: resample 80% of signature genes without replacement per
    iteration; refit NNLS on resampled sub-problem. WHY 80% without
    replacement: brief — this probes sensitivity to gene selection without
    generating large bootstrap-bias effects from duplicated rows.

    Returns long-form DataFrame:
        SAMPID, compartment, fraction_point,
        fraction_boot_mean, fraction_boot_ci_low, fraction_boot_ci_high,
        residual_l2, residual_ratio, flag_poor_fit, bootstrap_cv
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    n_genes, n_comp = S.shape
    n_samples = B.shape[1]
    logger.info(
        "[nnls] Deconvolving N=%d samples, %d compartments, %d genes, B=%d",
        n_samples, n_comp, n_genes, bootstrap_b,
    )

    # Point estimates for every sample.
    point_frac = np.zeros((n_samples, n_comp), dtype=np.float64)
    residual_l2 = np.zeros(n_samples, dtype=np.float64)
    residual_ratio = np.zeros(n_samples, dtype=np.float64)

    # Chunk samples to control memory; even though NNLS is per-sample O(n_genes
    # * n_comp), this chunking aids logging cadence.
    for start in range(0, n_samples, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_samples)
        for j in range(start, end):
            frac, r2, rr = _nnls_one(S, B[:, j])
            point_frac[j] = frac
            residual_l2[j] = r2
            residual_ratio[j] = rr
        if (end % 100) < CHUNK_SIZE:
            logger.info("[nnls]  point estimates %d/%d", end, n_samples)

    # Bootstrap: sample gene subsets outside the sample loop so every sample
    # uses the same bootstrap gene-subset schedule (paired bootstrap).
    # WHY paired: preserves across-sample comparability of bootstrap
    # variance estimates.
    boot_subsets = []
    keep_n = int(round(BOOTSTRAP_GENE_FRAC * n_genes))
    for _ in range(bootstrap_b):
        idx = rng.choice(n_genes, size=keep_n, replace=False)
        boot_subsets.append(idx)

    boot_frac = np.zeros((bootstrap_b, n_samples, n_comp), dtype=np.float64)
    for b_iter, idx in enumerate(boot_subsets):
        Sb = S[idx, :]
        Bb = B[idx, :]
        for j in range(n_samples):
            frac, _r2, _rr = _nnls_one(Sb, Bb[:, j])
            boot_frac[b_iter, j] = frac
        if (b_iter + 1) % max(1, bootstrap_b // 10) == 0:
            logger.info("[nnls]  bootstrap %d/%d", b_iter + 1, bootstrap_b)

    boot_mean = boot_frac.mean(axis=0)              # n_samples x n_comp
    boot_std = boot_frac.std(axis=0, ddof=1)
    ci_low = np.quantile(boot_frac, 0.025, axis=0)
    ci_high = np.quantile(boot_frac, 0.975, axis=0)
    # CV = sd / mean, robust against mean=0 by eps floor.
    boot_cv = boot_std / np.maximum(boot_mean, 1e-9)

    rows = []
    for j, sid in enumerate(sample_ids):
        flag_poor = residual_ratio[j] > 0.5
        for k, comp in enumerate(compartments):
            rows.append({
                "SAMPID": sid,
                "compartment": comp,
                "fraction_point": float(point_frac[j, k]),
                "fraction_boot_mean": float(boot_mean[j, k]),
                "fraction_boot_ci_low": float(ci_low[j, k]),
                "fraction_boot_ci_high": float(ci_high[j, k]),
                "fraction_boot_cv": float(boot_cv[j, k]),
                "residual_l2": float(residual_l2[j]),
                "residual_ratio": float(residual_ratio[j]),
                "flag_poor_fit": bool(flag_poor),
            })
    # Also keep the raw bootstrap tensor in memory for downstream B1
    # bootstrap-CI propagation by returning via a closure-attached attribute.
    df = pd.DataFrame(rows)
    df.attrs["boot_frac"] = boot_frac  # shape (B, n_samples, n_comp)
    df.attrs["sample_order"] = sample_ids
    df.attrs["compartment_order"] = compartments
    return df


# --------------------------------------------------------------------------- #
# Step 4 — Age regression.
# --------------------------------------------------------------------------- #
def _build_reg_frame(
    fractions_wide: pd.DataFrame, meta: pd.DataFrame
) -> pd.DataFrame:
    """Join fractions (wide, one row per sample) with meta covariates for
    regression. Drop rows with any required covariate missing.

    WHY a single curated frame: avoids repeated alignment per compartment
    regression; keeps sample set consistent across comparisons.
    """
    keep_cols = ["SAMPID", "age_midpoint", "SMTSISCH", "SMRIN", "SEX",
                 "SMCENTER", "DTHHRDY", "AGE"]
    m = meta[keep_cols].copy()
    df = fractions_wide.merge(m, on="SAMPID", how="inner")
    # Drop rows with NaN in required columns (numeric confounders).
    df = df.dropna(subset=["age_midpoint", "SMTSISCH", "SMRIN"])
    logger.info("[reg] Regression frame N = %d", len(df))
    return df


def _fit_primary_age(
    df: pd.DataFrame, compartment: str
) -> Dict[str, float]:
    """Fit OLS: fraction_<comp> ~ age + SMTSISCH + SMRIN + C(SEX) + C(SMCENTER)
       + C(DTHHRDY), with cluster-robust SE clustered on SMCENTER.

    Returns a dict with beta_age, ci_low, ci_high, p, n, r2.
    """
    col = f"frac_{compartment}"
    # Skip if the column has no variance.
    if df[col].std() < 1e-12:
        return {"beta_age": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "n": len(df), "r2": np.nan,
                "note": "zero_variance_fraction"}
    formula = (f"{col} ~ age_midpoint + SMTSISCH + SMRIN "
               f"+ C(SEX) + C(SMCENTER) + C(DTHHRDY)")
    groups = df["SMCENTER"].astype(str).values
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    b = float(model.params.get("age_midpoint", np.nan))
    ci = model.conf_int().loc["age_midpoint"].values if "age_midpoint" in model.params.index else [np.nan, np.nan]
    p = float(model.pvalues.get("age_midpoint", np.nan))
    return {
        "beta_age": b,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": p,
        "n": int(model.nobs),
        "r2": float(model.rsquared),
    }


def _fit_ordinal_sensitivity(
    df: pd.DataFrame, compartment: str
) -> Dict[str, float]:
    """Ordinal sensitivity: OrderedModel on 6 age bins if available, else
    Spearman of fraction vs ordered age bin.

    WHY fallback: statsmodels OrderedModel requires scipy + sometimes fails
    on small-N strata. Spearman gives ordinal rank correspondence as a
    coarse sensitivity per brief ("can use statsmodels OrderedModel or just
    ordinal-Spearman as approximation if OrderedModel unavailable; document").
    """
    col = f"frac_{compartment}"
    age_order = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"]
    bin_rank = df["AGE"].astype(str).map({a: i for i, a in enumerate(age_order)})
    if bin_rank.isna().any():
        return {"method": "spearman", "rho": np.nan, "p_value": np.nan,
                "note": "age_bin_map_failed"}
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        exog = df[["SMTSISCH", "SMRIN"]].copy()
        # OrderedModel requires sorted integer response.
        m = OrderedModel(bin_rank.values, exog, distr="logit").fit(disp=False)
        # Read slope for the fraction column by switching the response:
        # We cannot trivially swap with age as response AND fraction as input
        # inside OrderedModel; simpler approach: use proportional-odds with
        # AGE ordinal as response and fraction + confounders as exog.
        exog2 = df[[col, "SMTSISCH", "SMRIN"]].copy()
        m2 = OrderedModel(bin_rank.values, exog2, distr="logit").fit(
            disp=False
        )
        beta = float(m2.params.get(col, np.nan))
        p = float(m2.pvalues.get(col, np.nan))
        return {"method": "ordered_logit", "beta_fraction": beta,
                "p_value": p, "note": "proportional_odds"}
    except Exception as e:
        from scipy.stats import spearmanr
        rho, p = spearmanr(df[col].values, bin_rank.values)
        return {"method": "spearman", "rho": float(rho),
                "p_value": float(p), "note": f"ordered_logit_failed: {e!s}"}


def _bootstrap_propagate_age(
    df_base: pd.DataFrame,
    boot_frac: np.ndarray,                     # (B, n_samples, n_comp)
    sample_order: List[str],
    compartment_order: List[str],
    compartment: str,
) -> Tuple[float, float, float, float]:
    """Refit primary age regression on each bootstrap fraction estimate.

    Returns (boot_beta_mean, boot_ci_low, boot_ci_high, boot_cv).

    WHY: Critic 2 + Critic 3 variance-inflated CI. The analytic OLS CI
    conditions on the NNLS fraction point estimate as if error-free; the
    bootstrap propagation injects fraction-estimation noise into beta_age.
    """
    # AUDIT C3 FIX: vectorize sample -> bootstrap row mapping. Original
    # implementation iterated Python-over-rows per bootstrap iter; at B=500,
    # 4 compartments, ~500 samples, that was 200-600 min total. We now
    # compute the full (len(df),) bootstrap fraction column in ONE numpy
    # indexing op per bootstrap iteration; the per-iteration cost is then
    # dominated by a single cluster-robust OLS fit (~0.1-0.3 s). Expected
    # total: ~5-10 min per compartment.
    B = boot_frac.shape[0]
    k = compartment_order.index(compartment)
    sid_to_frac_row = {sid: i for i, sid in enumerate(sample_order)}
    df = df_base.copy()
    df_sids = df["SAMPID"].values
    # Vectorized index lookup; -1 sentinel for missing sample ids.
    sid_rows = np.array(
        [sid_to_frac_row.get(sid, -1) for sid in df_sids], dtype=np.int64
    )
    valid_mask = sid_rows >= 0
    n_valid = int(valid_mask.sum())
    if n_valid < len(df):
        logger.warning(
            "[bootstrap] %d/%d regression rows have no bootstrap fraction "
            "(SAMPID not in deconvolution sample_order); they will be dropped.",
            len(df) - n_valid, len(df),
        )
    # Only keep rows present in boot_frac mapping; re-order arrays consistently.
    df = df.loc[valid_mask].reset_index(drop=True)
    sid_rows = sid_rows[valid_mask]
    groups = df["SMCENTER"].astype(str).values

    # --- Singleton cluster guard (non-blocking W2 fix) ---
    # WHY: a single-row cluster produces a zero within-cluster residual and
    # inflates cluster-robust SE (or triggers LinAlgError). If the smallest
    # cluster is size 1, fall back to HC3 heteroskedasticity-robust SE.
    _, cluster_counts = np.unique(groups, return_counts=True)
    use_cluster = cluster_counts.min() >= 2
    if not use_cluster:
        logger.warning(
            "[bootstrap] compartment=%s: smallest SMCENTER cluster has N=1; "
            "falling back to HC3 robust SE for bootstrap fits.", compartment,
        )

    # Precompute design matrix once (it does not depend on the bootstrap).
    # Using smf here each iteration is still cheap because the design-matrix
    # build dominated by formula parsing is amortized by statsmodels when
    # `data` is re-used; we keep the formula approach for interpretability.
    formula = ("__boot_frac__ ~ age_midpoint + SMTSISCH + SMRIN "
               "+ C(SEX) + C(SMCENTER) + C(DTHHRDY)")

    betas = np.full(B, np.nan, dtype=np.float64)
    for b_iter in range(B):
        # VECTORIZED: one indexing op replaces the inner Python loop.
        col_vals = boot_frac[b_iter, sid_rows, k]
        if np.nanstd(col_vals) < 1e-12:
            continue
        df["__boot_frac__"] = col_vals
        try:
            if use_cluster:
                m = smf.ols(formula, data=df).fit(
                    cov_type="cluster", cov_kwds={"groups": groups}
                )
            else:
                m = smf.ols(formula, data=df).fit(cov_type="HC3")
            betas[b_iter] = m.params.get("age_midpoint", np.nan)
        except Exception:
            continue
    mask = ~np.isnan(betas)
    if mask.sum() < 10:
        return (np.nan, np.nan, np.nan, np.nan)
    mean_beta = float(betas[mask].mean())
    sd_beta = float(betas[mask].std(ddof=1))
    cv = sd_beta / max(abs(mean_beta), 1e-12)
    ci_low = float(np.quantile(betas[mask], 0.025))
    ci_high = float(np.quantile(betas[mask], 0.975))
    return mean_beta, ci_low, ci_high, cv


def _fit_tertile(
    df: pd.DataFrame, compartment: str
) -> List[Dict[str, float]]:
    """SMTSISCH tertile-stratified primary regression.

    WHY: Critic 3 B1 — age-fraction beta must replicate in lowest ischemic-
    time tertile INDEPENDENTLY or result is "ischemic-confounded". We report
    every tertile here plus the full-sample interaction test.
    """
    tertiles = pd.qcut(df["SMTSISCH"].rank(method="first"), q=3,
                       labels=["T1_low", "T2_mid", "T3_high"])
    out = []
    for name, mask in [("T1_low", tertiles == "T1_low"),
                       ("T2_mid", tertiles == "T2_mid"),
                       ("T3_high", tertiles == "T3_high")]:
        sub = df.loc[mask].copy()
        if len(sub) < 30:
            out.append({"tertile": name, "n": int(len(sub)),
                        "beta_age": np.nan, "ci_low": np.nan,
                        "ci_high": np.nan, "p_value": np.nan,
                        "note": "n_below_30_uninterpretable"})
            continue
        res = _fit_primary_age(sub, compartment)
        res["tertile"] = name
        out.append(res)

    # Age x SMTSISCH interaction on full data.
    col = f"frac_{compartment}"
    formula = (f"{col} ~ age_midpoint * SMTSISCH + SMRIN "
               f"+ C(SEX) + C(SMCENTER) + C(DTHHRDY)")
    groups = df["SMCENTER"].astype(str).values
    try:
        m = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        interaction_key = "age_midpoint:SMTSISCH"
        inter_p = float(m.pvalues.get(interaction_key, np.nan))
        inter_beta = float(m.params.get(interaction_key, np.nan))
    except Exception as e:
        inter_p = np.nan
        inter_beta = np.nan
        logger.warning("[reg] interaction fit failed: %s", e)
    out.append({"tertile": "interaction_age_x_smtsisch",
                "n": int(len(df)), "beta_age": inter_beta,
                "ci_low": np.nan, "ci_high": np.nan,
                "p_value": inter_p, "note": "age_midpoint:SMTSISCH term"})
    return out


# --------------------------------------------------------------------------- #
# Step 5 — B1s fraction-adjusted bulk TF-age regression.
# --------------------------------------------------------------------------- #
def _fit_b1s(
    reg_df: pd.DataFrame,
    tpm: pd.DataFrame,
    tf_list: List[str],
    compartments: List[str],
) -> pd.DataFrame:
    """For each TF in tf_list, compare age-beta BEFORE and AFTER adding
    compartment-fraction covariates.

    WHY drop-1 F-test design: brief Alternative — simpler interpretable
    test. If bulk TF-age effect attenuates after adjusting for fractions,
    the bulk age signal is compositional; if it survives, there is
    within-compartment activity change (SPECULATIVE evidence — full answer
    requires BayesPrism Z-tensor, deferred to iter 065).

    Per-compartment reporting: we report 24 cells (8 TFs x 3 primary
    compartments) by examining the TF x fraction-of-compartment INTERACTION
    term — the coefficient quantifies how that compartment modulates the
    TF-age slope.

    Returns a DataFrame with:
        TF, compartment, beta_age_before, p_before,
        beta_age_after, p_after, delta_r2, interaction_beta,
        interaction_p, n
    """
    rows = []
    # Build log2(TPM+1) for each TF aligned to reg_df SAMPIDs.
    sids = reg_df["SAMPID"].tolist()
    missing_tfs = [tf for tf in tf_list if tf not in tpm.index]
    if missing_tfs:
        logger.warning("[b1s] TFs missing from GTEx TPM: %s", missing_tfs)
    tf_vals = {}
    for tf in tf_list:
        if tf not in tpm.index:
            continue
        v = tpm.loc[tf, sids].values.astype(float)
        tf_vals[tf] = np.log2(v + 1.0)

    # Precompute SASP12 sum (log2 of sum of TPMs, matching brief).
    sasp_genes_present = [g for g in SASP12 if g in tpm.index]
    if len(sasp_genes_present) > 0:
        sasp_sum = tpm.loc[sasp_genes_present, sids].sum(axis=0).values
        tf_vals["SASP12"] = np.log2(sasp_sum + 1.0)

    groups = reg_df["SMCENTER"].astype(str).values

    for tf, y in tf_vals.items():
        df_local = reg_df.copy()
        df_local["__y__"] = y

        # BEFORE: age + standard confounders (no compartment fractions).
        before_formula = ("__y__ ~ age_midpoint + SMTSISCH + SMRIN "
                          "+ C(SEX) + C(SMCENTER) + C(DTHHRDY)")
        try:
            m_before = smf.ols(before_formula, data=df_local).fit(
                cov_type="cluster", cov_kwds={"groups": groups}
            )
            beta_before = float(m_before.params.get("age_midpoint", np.nan))
            p_before = float(m_before.pvalues.get("age_midpoint", np.nan))
            r2_before = float(m_before.rsquared)
        except Exception as e:
            beta_before = p_before = r2_before = np.nan
            logger.warning("[b1s] before-fit failed TF=%s: %s", tf, e)

        # AFTER: include fractions (use 3 primary; leave 1 as reference to
        # avoid perfect collinearity since fractions sum to 1).
        frac_cols = [f"frac_{c}" for c in compartments[:-1]]
        frac_terms = " + ".join(frac_cols)
        after_formula = (f"__y__ ~ age_midpoint + SMTSISCH + SMRIN "
                         f"+ C(SEX) + C(SMCENTER) + C(DTHHRDY) "
                         f"+ {frac_terms}")
        try:
            m_after = smf.ols(after_formula, data=df_local).fit(
                cov_type="cluster", cov_kwds={"groups": groups}
            )
            beta_after = float(m_after.params.get("age_midpoint", np.nan))
            p_after = float(m_after.pvalues.get("age_midpoint", np.nan))
            r2_after = float(m_after.rsquared)
            # Non-blocking W6 fix: condition number diagnostic. WHY: the AFTER
            # model includes 3 of 4 fractions, which are near-collinear since
            # the 4 fractions sum to 1; a high kappa inflates SE and mutes
            # beta_after. Flag at kappa > 1000 (standard threshold).
            try:
                kappa = float(np.linalg.cond(m_after.model.exog))
                if kappa > 1000.0:
                    logger.warning(
                        "[b1s] TF=%s AFTER design matrix kappa=%.1f (>1000); "
                        "near-collinear fractions (sum-to-1 constraint); SE "
                        "may be inflated.", tf, kappa,
                    )
            except Exception:
                pass
        except Exception as e:
            beta_after = p_after = r2_after = np.nan
            logger.warning("[b1s] after-fit failed TF=%s: %s", tf, e)

        delta_r2 = r2_after - r2_before if (not np.isnan(r2_after)
                                            and not np.isnan(r2_before)) else np.nan

        # Per-compartment TF-age interaction: fraction-of-compartment x age.
        for comp in compartments:
            if comp not in B1S_COMPARTMENTS and tf != "SASP12":
                continue
            inter_formula = (f"__y__ ~ age_midpoint * frac_{comp} + SMTSISCH "
                             f"+ SMRIN + C(SEX) + C(SMCENTER) + C(DTHHRDY)")
            try:
                m_int = smf.ols(inter_formula, data=df_local).fit(
                    cov_type="cluster", cov_kwds={"groups": groups}
                )
                ikey = f"age_midpoint:frac_{comp}"
                inter_beta = float(m_int.params.get(ikey, np.nan))
                inter_p = float(m_int.pvalues.get(ikey, np.nan))
            except Exception as e:
                inter_beta = np.nan
                inter_p = np.nan
                logger.warning("[b1s] interaction fit failed TF=%s comp=%s: %s",
                               tf, comp, e)
            rows.append({
                "TF": tf,
                "compartment": comp,
                "beta_age_before": beta_before,
                "p_before": p_before,
                "beta_age_after": beta_after,
                "p_after": p_after,
                "delta_r2": delta_r2,
                "interaction_beta": inter_beta,
                "interaction_p": inter_p,
                "n": int(len(df_local)),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Decision logic (brief DECISION RULE).
# --------------------------------------------------------------------------- #
def _apply_decision(
    primary: pd.DataFrame, tertile: pd.DataFrame, b1s: pd.DataFrame
) -> Dict[str, object]:
    """Translate numerical results into a pre-registered classification.

    WHY a function: keeps the decision surface explicit and auditable so a
    reviewer can read exactly which threshold was crossed.
    """
    # Bonferroni alpha across compartments (6 targeted in brief; we only have
    # 4 — document ratio).
    n_comp_tested = primary["compartment"].nunique()
    alpha_bonf = 0.05 / max(n_comp_tested, 1)

    sig_comp = primary[(primary["p_value"] < alpha_bonf) &
                       (primary["beta_age"].abs() > 5e-4)]  # 0.05% per year
    n_sig = len(sig_comp)

    # SMTSISCH-lowest-tertile replication test — primary compartment must
    # also reach p < 0.05 in T1_low for that compartment (weaker bar because
    # of reduced N; brief calls for "replicates in lowest tertile").
    replicated = []
    for _, row in sig_comp.iterrows():
        c = row["compartment"]
        t1 = tertile[(tertile["compartment"] == c) &
                     (tertile["tertile"] == "T1_low")]
        if len(t1) == 1 and t1.iloc[0]["p_value"] < 0.05 and \
                np.sign(t1.iloc[0]["beta_age"]) == np.sign(row["beta_age"]):
            replicated.append(c)

    if len(replicated) >= 2:
        b1_class = "ESTABLISHED_compositional_aging"
    elif n_sig >= 1 and len(replicated) >= 1:
        b1_class = "SUGGESTED_compositional_aging"
    elif n_sig >= 1 and len(replicated) == 0:
        b1_class = "INCONCLUSIVE_ischemic_confounded"
    else:
        b1_class = "REFUTES_or_null"

    # B1s: any TF-age surviving fraction adjustment with Bonferroni across
    # 24 cells (alpha 0.00208).
    alpha_b1s = 0.05 / 24.0
    b1s_sig = b1s[b1s["p_after"] < alpha_b1s]
    if b1_class.startswith("ESTABLISHED") or b1_class.startswith("SUGGESTED"):
        if len(b1s_sig) >= 1:
            b1s_class = "SPECULATIVE_activity_signal_pending_iter_065"
        else:
            b1s_class = "null_activity_signal"
    else:
        b1s_class = "null_activity_signal_or_not_applicable"

    return {
        "b1_classification": b1_class,
        "b1_n_sig_compartments": int(n_sig),
        "b1_replicated_in_t1_low": replicated,
        "b1_alpha_bonf_used": float(alpha_bonf),
        "b1s_classification": b1s_class,
        "b1s_n_sig_after_fractions": int(len(b1s_sig)),
        "b1s_alpha_bonf_used": float(alpha_b1s),
    }


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main() -> int:
    t0 = time.time()
    _set_seeds(SEED)
    logger.info("=== Analysis B (NNLS) start. Seed=%d B=%d ===",
                SEED, BOOTSTRAP_B)

    summary: Dict[str, object] = {
        "seed": SEED,
        "bootstrap_b": BOOTSTRAP_B,
        "compartments": COMPARTMENTS,
        "signature_basis_caveat": (
            "HLMA h5ads contain log1p-normalized data in .raw.X (not raw counts). "
            "Signature is mean log1p-normalized expression per compartment, "
            "renormalized to CPM-like 1e6 scale. Rank-preserving for NNLS; "
            "absolute fraction values not directly comparable to count-based "
            "deconvolutions (BayesPrism iter 065 will compare)."
        ),
        "missing_myofiber_caveat": (
            "No Myofiber reference h5ad available in HLMA bundle; NNLS runs "
            "with 4 compartments (Vascular, FAP, MuSC, Immune). Absolute "
            "fractions inflated; interpretation focuses on relative age-slope "
            "direction, not absolute fraction magnitudes."
        ),
        "confounder_deviation": (
            "GTEx v8 public SubjectPhenotypes contains only SUBJID, SEX, AGE, "
            "DTHHRDY; BMI and RACE are dbGaP-protected and unavailable here. "
            "Primary model uses: age_midpoint + SMTSISCH + SMRIN + SEX + "
            "SMCENTER (as COLLECTION_SITE proxy) + DTHHRDY. Cluster-robust "
            "SE clustered on SMCENTER."
        ),
        "fap_signature_method": (
            "FAP column built via option-3 fallback (curated marker-gene "
            "panel + per-cell ReLU-mean of scaled expression, L1-normalized "
            "to 1e6 CPM-like scale). WHY: OMIX004308-02 stores scaled "
            "(z-score-like) data with negatives in both .X and .raw.X; no "
            "raw-count FAP matrix is available locally (OMIX004308-03 is "
            "truncated on disk; OMIX004308-05 is the myofiber compartment). "
            "Marker panel: PDGFRA (weight 3, Uezumi 2010 / Joe 2010 / HLMA "
            "lit_doi_10.1038_s41586-024-07348-6), CD34, DCN, LUM, DLK1, "
            "COL6A1/2/3, THY1, MFAP5, SERPINF1 (weight 1, Schwalie 2018 "
            "Nature + HLMA). Tenocytes excluded. This is WEAKER than a "
            "count-based pseudobulk; iter 065 should replace with BayesPrism "
            "(R) or external FAP raw counts. Documented in "
            "b_hlma_signature_source.csv column 'marker_based'."
        ),
    }

    # --- 1. Signature ------------------------------------------------------ #
    signature, sig_reports = build_hlma_signature()
    signature.to_csv(OUT_SIGNATURE)
    summary["signature_path"] = str(OUT_SIGNATURE)
    summary["signature_shape"] = list(signature.shape)
    summary["signature_reports"] = [r.__dict__ for r in sig_reports]
    # Record per-compartment matrix source (audit C1 requirement).
    src_df = pd.DataFrame([{
        "compartment": r.compartment,
        "x_matrix_source": r.x_matrix_source,
        "negative_clip_applied": r.negative_clip_applied,
        "x_min_pre_clip": r.x_min_pre_clip,
        "n_cells": r.n_cells,
        "n_genes_in": r.n_genes_in,
        "x_min_probe_post": r.x_min,
        "x_max_probe_post": r.x_max,
        "marker_based": r.marker_based,
        "marker_genes_used": ";".join(r.marker_genes_used),
        "marker_genes_missing": ";".join(r.marker_genes_missing),
    } for r in sig_reports])
    src_df.to_csv(OUT_SIGNATURE_SOURCE, index=False)
    summary["signature_source_path"] = str(OUT_SIGNATURE_SOURCE)
    summary["signature_log_space_threshold"] = float(
        signature.attrs.get("log_space_threshold", float("nan"))
    )
    summary["signature_n_genes_after_symbol_filter"] = int(
        signature.attrs.get("n_genes_after_symbol_filter", -1)
    )
    summary["signature_n_genes_after_expression_filter"] = int(
        signature.attrs.get("n_genes_after_expression_filter", -1)
    )

    # --- 2. GTEx bulk ------------------------------------------------------ #
    tpm, meta, qc_log = load_gtex_bulk()
    summary["gtex_qc"] = qc_log.to_dict(orient="records")[0]

    # Per-bin counts audit (brief uninterpretable gate: N >= 30/bin).
    bin_counts = meta.groupby("AGE").size().to_dict()
    summary["gtex_age_bin_counts"] = {k: int(v) for k, v in bin_counts.items()}
    undercount_bins = [k for k, v in bin_counts.items() if v < 30]
    summary["gtex_age_bins_below_30"] = undercount_bins

    # --- 3. Prepare NNLS matrices ----------------------------------------- #
    S, B, gene_order = _prepare_nnls_matrices(signature, tpm)
    summary["nnls_n_common_genes"] = int(len(gene_order))

    # --- 3a. Smoke test on first SMOKE_N samples -------------------------- #
    smoke_ids = meta["SAMPID"].tolist()[:SMOKE_N]
    smoke_B = B[:, :SMOKE_N]
    logger.info("[smoke] Running smoke test on N=%d samples, B=%d ...",
                SMOKE_N, SMOKE_B)
    smoke_df = deconvolve_all(S, smoke_B, smoke_ids, COMPARTMENTS,
                              bootstrap_b=SMOKE_B,
                              rng=np.random.default_rng(SEED + 1))
    smoke_means = smoke_df.groupby("compartment")["fraction_point"].mean()
    logger.info("[smoke] mean fractions: %s",
                smoke_means.to_dict())
    summary["smoke_test_mean_fractions"] = smoke_means.to_dict()

    # Non-blocking W4 fix: abort gate on wildly implausible smoke-test means.
    # WHY: If any compartment mean > 0.95 (one compartment swallows all mass)
    # or < 0.005 (compartment invisible), the signature is degenerate and
    # continuing to full deconvolution wastes compute and produces
    # un-interpretable fractions.
    smoke_max = float(smoke_means.max())
    smoke_min = float(smoke_means.min())
    if smoke_max > 0.95 or smoke_min < 0.005:
        summary["smoke_test_abort"] = True
        summary["smoke_test_abort_reason"] = (
            f"implausible fraction range: min={smoke_min:.4f} "
            f"max={smoke_max:.4f}; thresholds min<0.005 or max>0.95"
        )
        with open(OUT_SUMMARY, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.error(
            "[smoke] ABORT: compartment fraction means outside plausible "
            "[0.005, 0.95] range (min=%.4f max=%.4f). Likely signature "
            "degeneracy. Summary written; full deconvolution NOT run.",
            smoke_min, smoke_max,
        )
        return 2
    summary["smoke_test_abort"] = False

    # Early-exit for smoke-only invocations. WHY: supports audit/CI runs that
    # want only the first-10 sanity check without paying for full N~500
    # deconvolution + bootstrap propagation.
    if os.environ.get("RUN_B_SMOKE_ONLY", "0") == "1":
        summary["smoke_only_exit"] = True
        summary["elapsed_seconds"] = round(time.time() - t0, 1)
        with open(OUT_SUMMARY, "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.info("[smoke] RUN_B_SMOKE_ONLY=1 -> exit after smoke stage. "
                    "Summary -> %s", OUT_SUMMARY)
        return 0

    # --- 3b. Full deconvolution ------------------------------------------- #
    all_ids = meta["SAMPID"].tolist()
    frac_df = deconvolve_all(S, B, all_ids, COMPARTMENTS,
                             bootstrap_b=BOOTSTRAP_B,
                             rng=np.random.default_rng(SEED))
    frac_df.to_csv(OUT_FRACTIONS, index=False)
    summary["fractions_path"] = str(OUT_FRACTIONS)

    # Wide format for regression: one row per SAMPID, one col per compartment.
    frac_wide = (frac_df.pivot_table(
        index="SAMPID", columns="compartment", values="fraction_point"
    ).reset_index())
    frac_wide.columns = (["SAMPID"]
                         + [f"frac_{c}" for c in frac_wide.columns[1:]])
    # Ensure all expected frac_<c> cols exist.
    for c in COMPARTMENTS:
        if f"frac_{c}" not in frac_wide.columns:
            frac_wide[f"frac_{c}"] = np.nan

    # Fraction of poor-fit samples.
    poor = frac_df[frac_df["flag_poor_fit"]]["SAMPID"].nunique()
    total = frac_df["SAMPID"].nunique()
    summary["n_samples_total"] = int(total)
    summary["n_samples_poor_fit"] = int(poor)
    summary["frac_poor_fit"] = float(poor / max(total, 1))

    # --- 4. Regression ---------------------------------------------------- #
    reg_df = _build_reg_frame(frac_wide, meta)
    primary_rows: List[Dict[str, object]] = []
    tertile_rows: List[Dict[str, object]] = []
    boot_frac_tensor = frac_df.attrs["boot_frac"]
    sample_order = frac_df.attrs["sample_order"]
    compartment_order = frac_df.attrs["compartment_order"]

    for comp in COMPARTMENTS:
        logger.info("[reg] ===== Compartment %s =====", comp)
        primary = _fit_primary_age(reg_df, comp)
        ordinal = _fit_ordinal_sensitivity(reg_df, comp)
        boot_mean, boot_cilow, boot_cihigh, boot_cv = _bootstrap_propagate_age(
            reg_df, boot_frac_tensor, sample_order, compartment_order, comp
        )
        primary_rows.append({
            "compartment": comp,
            **primary,
            "ordinal_method": ordinal.get("method"),
            "ordinal_beta_or_rho": ordinal.get("beta_fraction",
                                               ordinal.get("rho")),
            "ordinal_p": ordinal.get("p_value"),
            "ordinal_note": ordinal.get("note"),
            "bootstrap_beta_mean": boot_mean,
            "bootstrap_ci_low": boot_cilow,
            "bootstrap_ci_high": boot_cihigh,
            "bootstrap_cv": boot_cv,
        })
        tertile_res = _fit_tertile(reg_df, comp)
        for r in tertile_res:
            r["compartment"] = comp
            tertile_rows.append(r)

    primary_df = pd.DataFrame(primary_rows)
    tertile_df = pd.DataFrame(tertile_rows)
    primary_df.to_csv(OUT_AGE_PRIMARY, index=False)
    tertile_df.to_csv(OUT_AGE_TERTILE, index=False)
    summary["age_primary_path"] = str(OUT_AGE_PRIMARY)
    summary["age_tertile_path"] = str(OUT_AGE_TERTILE)

    # --- 5. B1s fraction-adjusted bulk TF-age ----------------------------- #
    b1s_df = _fit_b1s(reg_df, tpm, TF_PANEL, COMPARTMENTS)
    b1s_df.to_csv(OUT_B1S, index=False)
    summary["b1s_path"] = str(OUT_B1S)

    # --- 6. Decision ------------------------------------------------------ #
    decision = _apply_decision(primary_df, tertile_df, b1s_df)
    summary["decision"] = decision

    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(OUT_SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("=== Analysis B DONE in %s. Summary -> %s ===",
                _elapsed(t0), OUT_SUMMARY)
    return 0


if __name__ == "__main__":
    # Suppress statsmodels rank-deficient warnings that would drown the log.
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, module="statsmodels"
    )
    sys.exit(main())
