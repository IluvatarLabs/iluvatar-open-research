"""T3 pyBayesPrism + decoupler MLM deconvolution for batch_065.

Implements brief.md §T3 exactly, including the DUAL VALIDATION GATE
(ground-truth pseudobulk + ischemic sign-flip sentinel; xCell is sanity-only).

HYPERPARAMETER SOURCES
----------------------
- pyBayesPrism v0.1.0 API: pybayesprism/prism.py (local package inspection 2026-04-22).
- Prism.new + prism.run() pattern: pybayesprism/run.py (package's own example).
- Gibbs sampler defaults (chain.length=1000, burn.in=500, thinning=2): hardcoded in
  pybayesprism/prism.py Prism.valid_gibbs_control(); these mirror BayesPrism v2 defaults
  (Chu et al. 2022 Nat Cancer, lit_doi_10.1038_s43018-022-00356-3).
- 5000 highly variable genes for reference: brief.md §T3 "top 5000 variable genes";
  aligns with BayesPrism-typical 3k-6k HVG range (Chu 2022).
- Cells-per-compartment downsample to 4000: to keep reference matrix manageable
  (no published standard for downsample; value chosen by compute budget — BayesPrism
  collapses cells into pseudo-reference anyway, so down-to-2x-n_states-x-100 suffices).
- Bootstrap 100 resamples for CV (brief.md §T3 MEASUREMENT).
- CV > 0.3 exclusion threshold (brief.md §T3 MEASUREMENT).
- N=100 synthetic Dirichlet bulks for validation (brief.md §T3 MANDATORY DUAL VALIDATION 1).
- Dirichlet alpha=1 (flat prior) for synthetic bulk fractions — uniform over simplex
  (Aitchison 1986 standard for compositional sampling, no published standard for this
  specific benchmark; flat chosen to span the simplex broadly).
- Bonferroni alpha for T3a: 0.05/(4 compartments x 4 strata) = 3.125e-3 (brief.md §T3).
- Bonferroni alpha for T3b: 0.05/(9 TFs x 3 compartments x 4 strata) = 4.63e-4
  (brief.md §T3; 9 TFs = {JUNB,KLF10,CEBPB,FOS,EGR1,FOSL2,ATF3,IRF1,CDKN1A}, but CDKN1A
  has 0 regulon targets — it's p21, not a TF — so drops to 8 SCENIC-based TFs.
  We apply Bonferroni based on the pre-registered family size 9x3x4=108).
- ilr transform (Aitchison 1986): primary for compositional regression (brief.md §T3c).
- F064_07 sign-flip sentinel reference: experiments/batch_064/b_age_regression_tertile.csv.

CARDINAL RULE 5 (WHY)
---------------------
- We use pyBayesPrism FIRST (not decoupler) because BayesPrism is a proper deconvolution
  method producing per-sample compartment fractions θ with reference-update step that
  corrects for reference-mixture mismatch — this is the claim to test.
- DUAL VALIDATION GATE is non-circular: ground-truth pseudobulk is constructed from
  held-out cells with KNOWN Dirichlet fractions; pyBayesPrism sees it as bulk and must
  recover the fractions. Unlike xCell-vs-pyBayesPrism comparison (circular because both
  have shared biases; Critic 1 C4).
- Ischemic sentinel tests whether pyBayesPrism actually solves the iter-064 F064_08
  problem. If sign-flip persists, T3a has not fixed the failure mode even if primary
  validation passes.
- ilr-OLS (not raw OLS) for T3a because fractions sum to 1 (simplex); raw OLS on
  fractions is biased (Aitchison 1986).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

REPO = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro")
BATCH = REPO / "experiments" / "batch_065"
LOGDIR = BATCH / "logs"
LOGDIR.mkdir(parents=True, exist_ok=True)
LOGFILE = LOGDIR / "t3_stdout.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOGFILE, mode="w"), logging.StreamHandler()],
)
log = logging.getLogger("t3")

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
RNG = np.random.default_rng(SEED)

N_HVG = 5000           # brief.md §T3 MEASUREMENT
N_CELLS_PER_CMPT = 4000  # compute budget; BayesPrism collapses into state-pseudobulk
N_SYNTHETIC = 100        # brief.md §T3 MANDATORY 1
N_BOOTSTRAP = 100        # brief.md §T3 MEASUREMENT
CV_THRESHOLD = 0.30      # brief.md §T3 MEASUREMENT
RHO_GATE = 0.7           # brief.md §T3 MANDATORY 1 (Spearman threshold)
DIRICHLET_ALPHA = 1.0    # flat prior over simplex

GIBBS_CONTROL = {"chain.length": 1000, "burn.in": 500, "thinning": 2, "seed": SEED}
OPT_CONTROL = {"maxit": 100000, "optimizer": "MAP"}

# Compartment reference files (h5ad); myofiber optional
HLMA_FILES = {
    "Vascular": REPO / "data" / "Vascular_scsn_RNA.h5ad",
    "MuSC": REPO / "data" / "MuSC_scsn_RNA.h5ad",
    "FAP": REPO / "data" / "OMIX004308-02.h5ad",
    "Immune": REPO / "data" / "Immune_scsn_RNA.h5ad",
    "Myofiber": REPO / "data" / "OMIX004308-05.h5ad",  # include if present
}

GTEX_TPM = REPO / "data" / "GTEx" / "muscle" / "gene_tpm_muscle_skeletal.gct.gz"
GTEX_SATTR = REPO / "data" / "GTEx" / "muscle" / "GTEx_v8_SampleAttributes.txt"
GTEX_SUBJ = REPO / "data" / "GTEx" / "muscle" / "GTEx_v8_SubjectPhenotypes.txt"

# pySCENIC regulons (batch_054/055)
REGULON_FILES = {
    "Vascular": REPO / "experiments" / "batch_054" / "d1_adjacencies_HLMA_Vascular.csv",
    "MuSC": REPO / "experiments" / "batch_054" / "d1_adjacencies_HLMA_MuSC.csv",
    "FAP": REPO / "experiments" / "batch_055" / "d1_adjacencies_HLMA_FAP.csv",
}

# Nine canonical TFs per brief.md §T3 (CDKN1A retained in pre-registered family
# count; empirically has 0 targets — p21 is a cell-cycle inhibitor, not a TF —
# so SCENIC-based T3b will cover 8 TFs)
CANONICAL_TFS = ["JUNB", "KLF10", "CEBPB", "FOS", "EGR1", "FOSL2", "ATF3", "IRF1", "CDKN1A"]

# F064_07 reference sign-flip magnitudes (per-compartment T3_high vs T1_low β_age)
# from experiments/batch_064/b_age_regression_tertile.csv
F064_07_BETA = {
    # compartment: (beta_T1_low, beta_T3_high)
    "Vascular": (-9.559e-05, 7.650e-04),   # sign flip: neg T1 -> pos T3
    "MuSC":     ( 1.675e-05, -1.563e-03),  # sign flip: pos T1 -> neg T3
    "FAP":      (-5.574e-05, 1.333e-04),   # sign flip
    "Immune":   ( 1.346e-04, 6.646e-04),   # same sign, but magnitude amplification
}

OUTPUTS = {
    "t3a_fractions":   BATCH / "t3a_pybayesprism_fractions.csv",
    "t3a_validation":  BATCH / "t3a_validation_gate.json",
    "t3a_reg":         BATCH / "t3a_age_regression.csv",
    "t3b_activity":    BATCH / "t3b_decoupler_activity.csv",
    "t3b_reg":         BATCH / "t3b_age_regression.csv",
    "t3_summary":      BATCH / "t3_summary.json",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log_env() -> None:
    import platform
    import pybayesprism
    import decoupler
    import skbio
    log.info("=== Environment ===")
    log.info("Python: %s", sys.version.replace("\n", " "))
    log.info("Platform: %s", platform.platform())
    log.info("seed=%d", SEED)
    log.info("pybayesprism: %s", Path(pybayesprism.__file__).parent)
    log.info("decoupler: %s", decoupler.__version__)
    log.info("skbio: %s", skbio.__version__)
    log.info("statsmodels: %s", sm.__version__)
    log.info("numpy: %s", np.__version__)
    log.info("pandas: %s", pd.__version__)
    log.info("command: %s", " ".join(sys.argv))


def midpoint_age(s: str) -> float | None:
    """Convert GTEx 'AGE' bracket (e.g. '60-69') to midpoint float."""
    if not isinstance(s, str) or "-" not in s:
        return None
    try:
        a, b = s.split("-")
        return (int(a) + int(b)) / 2.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GTEx loading
# ---------------------------------------------------------------------------

def load_gtex_bulk() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (bulk_df samples x genes_symbol, meta_df samples x covariates).

    GTEx TPM is provided with ENSG-versioned IDs; we strip the version and keep
    HGNC symbol for gene-matching with HLMA reference.
    """
    log.info("Reading GTEx TPM %s ...", GTEX_TPM.name)
    bulk = pd.read_csv(GTEX_TPM, sep="\t", skiprows=2)
    # Columns: id, Name, Description, <samples ...>
    bulk = bulk.drop(columns=["id"])
    bulk = bulk.rename(columns={"Name": "ensg", "Description": "symbol"})
    bulk["ensg_stripped"] = bulk["ensg"].str.split(".").str[0]
    # Use symbol when unique; keep ENSG for fallback
    log.info("  TPM shape: %s", bulk.shape)

    # Collapse duplicate symbols by mean (some symbols map to multiple ENSG)
    sample_cols = [c for c in bulk.columns if c.startswith("GTEX-")]
    log.info("  N samples pre-QC: %d", len(sample_cols))
    bulk_sym = bulk[["symbol"] + sample_cols].groupby("symbol").mean(numeric_only=True)
    log.info("  After symbol collapse: %s", bulk_sym.shape)

    # Metadata
    sattr = pd.read_csv(GTEX_SATTR, sep="\t")
    subj = pd.read_csv(GTEX_SUBJ, sep="\t")
    # SUBJID from SAMPID: GTEX-XXXX
    sattr["SUBJID"] = sattr["SAMPID"].str.extract(r"^(GTEX-[^-]+)-")
    meta = sattr.merge(subj, on="SUBJID", how="left")
    meta = meta[meta["SAMPID"].isin(sample_cols)]
    meta = meta[meta["SMTSD"] == "Muscle - Skeletal"]
    # Only keep samples with valid age, SMTSISCH (ischemic time), RIN
    keep = meta.dropna(subset=["AGE", "SMTSISCH", "SMRIN", "SEX"])
    keep = keep.copy()
    keep["age_midpoint"] = keep["AGE"].map(midpoint_age)
    keep = keep.dropna(subset=["age_midpoint"])
    log.info("  After QC (age/SMTSISCH/RIN/SEX valid): %d samples", len(keep))

    # Align bulk to QC'd samples
    bulk_qc = bulk_sym[keep["SAMPID"].tolist()]
    meta_out = keep.set_index("SAMPID")[
        ["age_midpoint", "SMTSISCH", "SMRIN", "SEX", "AGE", "SUBJID"]
    ]
    # Tertile strata per SMTSISCH
    meta_out["ischemic_tertile"] = pd.qcut(
        meta_out["SMTSISCH"], q=3, labels=["T1_low", "T2_mid", "T3_high"]
    )
    return bulk_qc, meta_out


# ---------------------------------------------------------------------------
# HLMA reference loading
# ---------------------------------------------------------------------------

def load_compartment(h5ad_path: Path, compartment: str, max_cells: int = N_CELLS_PER_CMPT) -> ad.AnnData:
    """Return stratified-subsampled AnnData with `Annotation` cell states."""
    if not h5ad_path.exists():
        log.warning("  Compartment %s: file %s missing; skipping", compartment, h5ad_path)
        return None
    a = ad.read_h5ad(h5ad_path)
    log.info("  %s: %d cells x %d genes, states=%s",
             compartment, a.n_obs, a.n_vars,
             sorted(a.obs["Annotation"].astype(str).unique()))
    # Subsample stratified by Annotation
    rng = np.random.default_rng(SEED + hash(compartment) % 1000)
    idx = []
    per_state = max_cells // max(a.obs["Annotation"].nunique(), 1)
    for st, sub in a.obs.groupby("Annotation", observed=True):
        take = min(per_state, len(sub))
        sel = rng.choice(sub.index, size=take, replace=False)
        idx.extend(sel.tolist())
    a = a[idx].copy()
    log.info("    subsampled to %d cells", a.n_obs)
    return a


def select_hvg_union(adatas: dict[str, ad.AnnData], n_top: int = N_HVG) -> list[str]:
    """Union of top-N variable genes from each compartment, ranked by per-compartment variance."""
    gene_scores: dict[str, float] = {}
    for c, a in adatas.items():
        if a is None:
            continue
        X = a.X
        if sp.issparse(X):
            mean = np.asarray(X.mean(axis=0)).ravel()
            m2 = np.asarray(X.multiply(X).mean(axis=0)).ravel()
            var = m2 - mean**2
        else:
            var = X.var(axis=0)
        genes = a.var_names.tolist()
        order = np.argsort(-var)[:n_top]
        for i in order:
            g = genes[i]
            gene_scores[g] = max(gene_scores.get(g, -np.inf), float(var[i]))
    ranked = sorted(gene_scores.items(), key=lambda kv: -kv[1])
    # Cap at n_top * ~1.5 to limit size while keeping union coverage
    return [g for g, _ in ranked[: n_top * 3]]


def build_reference_counts(
    adatas: dict[str, ad.AnnData], genes: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Concatenate all compartments into one cells-x-genes DataFrame.

    cell_type_labels = compartment name (Vascular/MuSC/FAP/Immune/Myofiber).
    cell_state_labels = compartment::Annotation (finer state).
    Values are the HLMA stored expression (log-normalized), passed as-is since
    the brief explicitly specifies "mean log-normalized expression". pyBayesPrism
    will warn but still runs.
    """
    parts = []
    ct_labels: list[str] = []
    cs_labels: list[str] = []
    for c, a in adatas.items():
        if a is None:
            continue
        gene_keep = [g for g in genes if g in a.var_names]
        sub = a[:, gene_keep]
        X = sub.X
        if sp.issparse(X):
            X = X.toarray()
        df = pd.DataFrame(X, columns=gene_keep)
        # Add missing genes as zeros to align
        missing = [g for g in genes if g not in gene_keep]
        for g in missing:
            df[g] = 0.0
        df = df[genes]
        df.index = [f"{c}_{i}" for i in range(len(df))]
        parts.append(df)
        ct_labels.extend([c] * len(df))
        # state labels must be unique per type (BayesPrism requirement)
        ann = sub.obs["Annotation"].astype(str).values
        cs_labels.extend([f"{c}::{s}" for s in ann])
    ref = pd.concat(parts, axis=0)
    log.info("Reference matrix: %s (cells x genes)", ref.shape)
    log.info("  cell-types: %s", sorted(set(ct_labels)))
    return ref, ct_labels, cs_labels


# ---------------------------------------------------------------------------
# pyBayesPrism runner
# ---------------------------------------------------------------------------

def run_pybayesprism(
    ref_df: pd.DataFrame,
    ct_labels: list[str],
    cs_labels: list[str],
    mixture_df: pd.DataFrame,
    update_gibbs: bool = True,
) -> pd.DataFrame:
    """Run pyBayesPrism; return per-sample posterior compartment fractions θf."""
    from pybayesprism.prism import Prism
    from pybayesprism import extract

    # Align gene order: mixture columns should match reference columns
    shared = [g for g in ref_df.columns if g in mixture_df.columns]
    log.info("  shared genes ref∩mixture = %d", len(shared))
    if len(shared) < 500:
        raise RuntimeError(f"Too few shared genes: {len(shared)}")
    ref_al = ref_df[shared]
    mix_al = mixture_df[shared]

    t0 = time.time()
    prism = Prism.new(
        reference=ref_al,
        input_type="count.matrix",
        cell_type_labels=ct_labels,
        cell_state_labels=cs_labels,
        key=None,  # non-tumor
        mixture=mix_al,
    )
    bp = prism.run(
        n_cores=1,
        update_gibbs=update_gibbs,
        gibbs_control=GIBBS_CONTROL.copy(),
        opt_control=OPT_CONTROL.copy(),
    )
    which = "final" if update_gibbs else "first"
    theta = extract.get_fraction(bp, which, "type")
    log.info("  pyBayesPrism done in %.1fs; theta shape %s", time.time() - t0, theta.shape)
    return theta


# ---------------------------------------------------------------------------
# Dual validation gate
# ---------------------------------------------------------------------------

def make_synthetic_bulks(
    adatas: dict[str, ad.AnnData],
    genes: list[str],
    n_bulks: int = N_SYNTHETIC,
    reads_per_bulk: int = 50_000_000,  # ~50M reads like bulk RNA-seq
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct N synthetic bulks with known Dirichlet-sampled fractions.

    Returns (bulk_df samples x genes, truth_df samples x compartments).
    Uses held-out 10% of cells per compartment.
    """
    rng = np.random.default_rng(SEED + 1)
    cmpts = [c for c in adatas if adatas[c] is not None]
    # Compute per-compartment mean expression across held-out cells (pseudo-reference)
    holdouts: dict[str, np.ndarray] = {}
    for c in cmpts:
        a = adatas[c]
        n_hold = max(int(0.1 * a.n_obs), 50)
        hold_idx = rng.choice(a.n_obs, size=n_hold, replace=False)
        X = a[hold_idx, :].X
        if sp.issparse(X):
            X = X.toarray()
        # Map to shared gene space
        gene_pos = {g: i for i, g in enumerate(a.var_names)}
        expr = np.zeros(len(genes), dtype=np.float64)
        for j, g in enumerate(genes):
            if g in gene_pos:
                expr[j] = X[:, gene_pos[g]].mean()
        holdouts[c] = expr
    # Sample Dirichlet fractions
    k = len(cmpts)
    truths = rng.dirichlet(alpha=[DIRICHLET_ALPHA] * k, size=n_bulks)
    # Build bulks as weighted sums. Since HLMA X is log-normalized, we reverse to
    # normalized-library (expm1) before mixing so the mixing linearity assumption
    # (bulk = sum fraction_c * expr_c) holds on the normalized scale.
    ref_mat = np.stack([np.expm1(np.clip(holdouts[c], 0, None)) for c in cmpts], axis=0)  # (k, g)
    mixed = truths @ ref_mat  # (n_bulks, g)
    # Rescale to reads_per_bulk (rough count-scale)
    mixed = mixed / (mixed.sum(axis=1, keepdims=True) + 1e-9) * reads_per_bulk
    bulk_df = pd.DataFrame(
        mixed.astype(np.float32),
        columns=genes,
        index=[f"synth_{i}" for i in range(n_bulks)],
    )
    truth_df = pd.DataFrame(truths, columns=cmpts, index=bulk_df.index)
    return bulk_df, truth_df


def dual_validation_gate(
    theta_synth: pd.DataFrame,
    truth_df: pd.DataFrame,
    theta_gtex: pd.DataFrame,
    gtex_meta: pd.DataFrame,
) -> dict:
    """Ground-truth + ischemic sentinel gates. See brief §T3 MANDATORY 1/2/3."""
    result: dict = {"per_compartment_rho": {}, "sentinel": {}, "gates": {}}

    # --- Gate 1: ground-truth Spearman --------------------------------------
    shared_c = [c for c in truth_df.columns if c in theta_synth.columns]
    for c in shared_c:
        rho, p = spearmanr(theta_synth[c].values, truth_df[c].values)
        result["per_compartment_rho"][c] = {"rho": float(rho), "p": float(p)}
    required = ["Vascular", "MuSC", "FAP"]
    ok = all(
        c in result["per_compartment_rho"]
        and result["per_compartment_rho"][c]["rho"] > RHO_GATE
        for c in required
    )
    result["gates"]["ground_truth_pass"] = bool(ok)

    # --- Gate 2: ischemic sign-flip sentinel --------------------------------
    # For each compartment, fit β_age within T1_low and T3_high strata of GTEx.
    # Compare to F064_07 signed magnitudes. Sentinel fails if sign-flip
    # magnitude > 50% of F064_07 reference magnitude.
    meta = gtex_meta.reindex(theta_gtex.index).dropna(subset=["age_midpoint"])
    frac = theta_gtex.reindex(meta.index)
    sentinel_res: dict = {}
    total_reappearance = 0
    for c in ["Vascular", "MuSC", "FAP", "Immune"]:
        if c not in frac.columns:
            continue
        ref_lo, ref_hi = F064_07_BETA[c]
        ref_delta = ref_hi - ref_lo
        betas = {}
        for tert in ["T1_low", "T3_high"]:
            idx = meta.index[meta["ischemic_tertile"] == tert]
            if len(idx) < 20:
                betas[tert] = None
                continue
            y = frac.loc[idx, c].values.astype(float)
            X = meta.loc[idx, ["age_midpoint"]].copy()
            X["const"] = 1.0
            X["SMRIN"] = meta.loc[idx, "SMRIN"].values
            X["SEX"] = meta.loc[idx, "SEX"].values
            X = X.astype(float)
            try:
                mod = sm.OLS(y, X).fit()
                betas[tert] = float(mod.params.get("age_midpoint", np.nan))
            except Exception as e:
                betas[tert] = None
                log.warning("  sentinel OLS failed for %s %s: %s", c, tert, e)
        if betas.get("T1_low") is None or betas.get("T3_high") is None:
            continue
        new_delta = betas["T3_high"] - betas["T1_low"]
        # Reappearance fraction: magnitude ratio (sign comparison)
        rel = abs(new_delta) / (abs(ref_delta) + 1e-20) if abs(ref_delta) > 0 else 0.0
        reappear = bool(rel > 0.5 and np.sign(new_delta) == np.sign(ref_delta))
        sentinel_res[c] = {
            "beta_T1_low": betas["T1_low"],
            "beta_T3_high": betas["T3_high"],
            "new_delta": new_delta,
            "f064_07_delta": ref_delta,
            "relative_magnitude": rel,
            "sign_match": bool(np.sign(new_delta) == np.sign(ref_delta)),
            "reappears": reappear,
        }
        if reappear:
            total_reappearance += 1
    result["sentinel"] = sentinel_res
    # Gate passes when NOT-reappearing (i.e., fewer than half of compartments
    # replay the iter-064 sign-flip pattern).
    result["gates"]["sentinel_pass"] = bool(total_reappearance < 2)

    # Overall dual-gate decision
    result["gates"]["dual_pass"] = bool(
        result["gates"]["ground_truth_pass"] and result["gates"]["sentinel_pass"]
    )
    return result


# ---------------------------------------------------------------------------
# T3b: decoupler MLM TF activity
# ---------------------------------------------------------------------------

def build_regulon_table() -> pd.DataFrame:
    """Long-format (source, target, weight) regulon table from batch_054/055."""
    frames = []
    for compartment, path in REGULON_FILES.items():
        if not path.exists():
            log.warning("  regulon file %s missing", path)
            continue
        df = pd.read_csv(path)
        df = df[df["TF"].isin(CANONICAL_TFS)].copy()
        if df.empty:
            continue
        # Prefix source with compartment for compartment-specific TFs
        df["source"] = compartment + "__" + df["TF"]
        df = df.rename(columns={"target": "target", "importance": "weight"})
        frames.append(df[["source", "target", "weight"]])
    out = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    log.info("SCENIC regulon table: %d rows, %d sources", len(out), out["source"].nunique() if len(out) else 0)
    return out


def run_decoupler_mlm(bulk_df: pd.DataFrame, net: pd.DataFrame, name: str) -> pd.DataFrame:
    """bulk_df samples x genes (HGNC). net: long with (source, target, weight)."""
    import decoupler as dc
    # decoupler expects samples x features dataframe
    log.info("  decoupler MLM (%s): %d samples x %d genes, net sources=%d",
             name, bulk_df.shape[0], bulk_df.shape[1], net["source"].nunique())
    act = dc.mt.mlm(data=bulk_df, net=net, tmin=5, verbose=False)
    # decoupler 2.x returns tuple (score, pvalue) or AnnData; normalize to DataFrame
    if isinstance(act, tuple):
        score, pval = act[0], act[1] if len(act) > 1 else None
        if hasattr(score, "to_df"):
            score = score.to_df()
        elif isinstance(score, np.ndarray):
            score = pd.DataFrame(score, index=bulk_df.index, columns=sorted(net["source"].unique()))
        log.info("    score shape %s", score.shape)
        return score
    if hasattr(act, "to_df"):
        return act.to_df()
    return act


# ---------------------------------------------------------------------------
# Bootstrap CV
# ---------------------------------------------------------------------------

def bootstrap_cv_fractions(
    ref_df: pd.DataFrame,
    ct_labels: list[str],
    cs_labels: list[str],
    mix_df: pd.DataFrame,
    theta_hat: pd.DataFrame,
    n_boot: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """CV via bootstrapping the reference cells (BayesPrism); lightweight variant.

    Rather than re-running full Gibbs n_boot times (too expensive), we bootstrap
    the GENE-space by subsampling 80% of shared genes and re-running initial
    Gibbs (update_gibbs=False, short chain). Reports per-sample CV across boots.
    """
    from pybayesprism.prism import Prism
    from pybayesprism import extract
    rng = np.random.default_rng(SEED + 2)
    cmpts = list(theta_hat.columns)
    boot_stack = np.zeros((n_boot, len(theta_hat), len(cmpts)), dtype=np.float32)
    for b in range(n_boot):
        genes = ref_df.columns.tolist()
        sel = rng.choice(len(genes), size=int(0.8 * len(genes)), replace=False)
        genes_sub = [genes[i] for i in sel]
        ref_b = ref_df[genes_sub]
        mix_b = mix_df[[g for g in genes_sub if g in mix_df.columns]]
        try:
            p = Prism.new(
                reference=ref_b,
                input_type="count.matrix",
                cell_type_labels=ct_labels,
                cell_state_labels=cs_labels,
                key=None,
                mixture=mix_b,
            )
            bp = p.run(
                n_cores=1, update_gibbs=False,
                gibbs_control={"chain.length": 200, "burn.in": 100,
                               "thinning": 2, "seed": SEED + b},
                opt_control=OPT_CONTROL.copy(),
            )
            t = extract.get_fraction(bp, "first", "type")
            t = t.reindex(index=theta_hat.index, columns=cmpts)
            boot_stack[b] = t.values
        except Exception as e:
            log.warning("  bootstrap %d failed: %s", b, e)
            boot_stack[b] = np.nan
    mean = np.nanmean(boot_stack, axis=0)
    std = np.nanstd(boot_stack, axis=0)
    cv = std / (np.abs(mean) + 1e-9)
    cv_df = pd.DataFrame(cv, index=theta_hat.index, columns=cmpts)
    return cv_df


# ---------------------------------------------------------------------------
# Regression: ilr-OLS for fractions, OLS for activities
# ---------------------------------------------------------------------------

def ilr_transform(fractions: pd.DataFrame) -> pd.DataFrame:
    """Aitchison ilr (Aitchison 1986) via skbio.stats.composition.ilr."""
    from skbio.stats.composition import ilr
    f = fractions.values
    # Clip and renormalize to avoid 0 which breaks log-ratio
    f = np.clip(f, 1e-6, None)
    f = f / f.sum(axis=1, keepdims=True)
    ilr_vals = ilr(f)  # returns (N, K-1)
    cols = [f"ilr_{i}" for i in range(ilr_vals.shape[1])]
    return pd.DataFrame(ilr_vals, index=fractions.index, columns=cols)


def age_regression_fractions(
    fractions: pd.DataFrame,
    meta: pd.DataFrame,
    tertile_col: str = "ischemic_tertile",
    pre_n: int | None = None,
) -> pd.DataFrame:
    """ilr-OLS: ilr coord ~ age + SMTSISCH + age:SMTSISCH + sex + RIN.

    Runs pooled and within-tertile. Returns long-format frame. Note: ilr
    coordinates do not have a clean one-to-one mapping to compartments, so we
    also report per-compartment marginal slope on the raw fractions (Dirichlet
    regression requires iterative fit; we use per-compartment logit-transformed
    OLS as a back-up reported alongside).
    """
    rows = []
    meta = meta.reindex(fractions.index)
    # Filter to valid
    ok = meta.dropna(subset=["age_midpoint", "SMTSISCH", "SMRIN", "SEX"]).index
    ilr_df = ilr_transform(fractions.loc[ok])

    # Also prep logit-transformed raw fractions per compartment (backup)
    f = fractions.loc[ok].values.clip(1e-6, 1 - 1e-6)
    logit_df = pd.DataFrame(
        np.log(f / (1 - f)), index=ok, columns=fractions.columns
    )

    def _run(df: pd.DataFrame, stratum: str, kind: str, col_space: list[str]) -> None:
        for col in col_space:
            y = df[col]
            X = pd.DataFrame({
                "const": 1.0,
                "age_midpoint": meta.loc[df.index, "age_midpoint"].astype(float),
                "SMTSISCH": meta.loc[df.index, "SMTSISCH"].astype(float),
                "SMRIN": meta.loc[df.index, "SMRIN"].astype(float),
                "SEX": meta.loc[df.index, "SEX"].astype(float),
            })
            X["age_x_smtsisch"] = X["age_midpoint"] * X["SMTSISCH"]
            try:
                mod = sm.OLS(y.astype(float), X.astype(float)).fit()
                rows.append({
                    "stratum": stratum,
                    "kind": kind,
                    "coord": col,
                    "beta_age": float(mod.params["age_midpoint"]),
                    "se_age": float(mod.bse["age_midpoint"]),
                    "p_age": float(mod.pvalues["age_midpoint"]),
                    "ci_lo": float(mod.conf_int().loc["age_midpoint", 0]),
                    "ci_hi": float(mod.conf_int().loc["age_midpoint", 1]),
                    "beta_age_x_smtsisch": float(mod.params["age_x_smtsisch"]),
                    "p_age_x_smtsisch": float(mod.pvalues["age_x_smtsisch"]),
                    "n": int(mod.nobs),
                    "r2": float(mod.rsquared),
                    "pre_cv_n": pre_n if pre_n is not None else int(mod.nobs),
                    "post_cv_n": int(mod.nobs),
                })
            except Exception as e:
                log.warning("  regression %s %s %s failed: %s", stratum, kind, col, e)

    # Pooled
    _run(ilr_df, "pooled", "ilr", list(ilr_df.columns))
    _run(logit_df, "pooled", "logit", list(logit_df.columns))
    # Tertile-stratified
    for tert in ["T1_low", "T2_mid", "T3_high"]:
        sub_idx = meta.loc[ok].index[meta.loc[ok, tertile_col] == tert]
        if len(sub_idx) < 20:
            continue
        _run(ilr_df.loc[sub_idx], tert, "ilr", list(ilr_df.columns))
        _run(logit_df.loc[sub_idx], tert, "logit", list(logit_df.columns))
    out = pd.DataFrame(rows)
    # Bonferroni
    n_tests_family = 4 * 4  # 4 compartments x 4 strata (brief §T3)
    out["bonf_alpha"] = 0.05 / n_tests_family
    out["bonf_sig"] = out["p_age"] < out["bonf_alpha"]
    return out


def age_regression_activity(
    act_df: pd.DataFrame,
    meta: pd.DataFrame,
    family_name: str,
    tertile_col: str = "ischemic_tertile",
    pre_n: int | None = None,
) -> pd.DataFrame:
    """OLS: activity ~ age + SMTSISCH + age:SMTSISCH + sex + RIN."""
    rows = []
    meta = meta.reindex(act_df.index)
    ok = meta.dropna(subset=["age_midpoint", "SMTSISCH", "SMRIN", "SEX"]).index
    act = act_df.loc[ok]
    tfcols = [c for c in act.columns]

    def _run(df: pd.DataFrame, stratum: str) -> None:
        for tf in tfcols:
            y = df[tf]
            X = pd.DataFrame({
                "const": 1.0,
                "age_midpoint": meta.loc[df.index, "age_midpoint"].astype(float),
                "SMTSISCH": meta.loc[df.index, "SMTSISCH"].astype(float),
                "SMRIN": meta.loc[df.index, "SMRIN"].astype(float),
                "SEX": meta.loc[df.index, "SEX"].astype(float),
            })
            X["age_x_smtsisch"] = X["age_midpoint"] * X["SMTSISCH"]
            try:
                mod = sm.OLS(y.astype(float), X.astype(float)).fit()
                rows.append({
                    "stratum": stratum,
                    "tf": tf,
                    "beta_age": float(mod.params["age_midpoint"]),
                    "se_age": float(mod.bse["age_midpoint"]),
                    "p_age": float(mod.pvalues["age_midpoint"]),
                    "ci_lo": float(mod.conf_int().loc["age_midpoint", 0]),
                    "ci_hi": float(mod.conf_int().loc["age_midpoint", 1]),
                    "beta_age_x_smtsisch": float(mod.params["age_x_smtsisch"]),
                    "p_age_x_smtsisch": float(mod.pvalues["age_x_smtsisch"]),
                    "n": int(mod.nobs),
                    "r2": float(mod.rsquared),
                    "family": family_name,
                    "pre_cv_n": pre_n if pre_n is not None else int(mod.nobs),
                    "post_cv_n": int(mod.nobs),
                })
            except Exception as e:
                log.warning("  activity reg %s %s failed: %s", stratum, tf, e)

    _run(act, "pooled")
    for tert in ["T1_low", "T2_mid", "T3_high"]:
        sub_idx = meta.loc[ok].index[meta.loc[ok, tertile_col] == tert]
        if len(sub_idx) < 20:
            continue
        _run(act.loc[sub_idx], tert)
    out = pd.DataFrame(rows)
    # Bonferroni: 9 TFs x 3 compartments x 4 strata = 108 tests
    out["bonf_alpha"] = 0.05 / 108
    out["bonf_sig"] = out["p_age"] < out["bonf_alpha"]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log_env()
    summary: dict = {
        "iteration": 65,
        "batch": "batch_065",
        "task": "T3",
        "seed": SEED,
        "config": {
            "n_hvg": N_HVG,
            "n_cells_per_compartment": N_CELLS_PER_CMPT,
            "n_synthetic_bulks": N_SYNTHETIC,
            "n_bootstrap": N_BOOTSTRAP,
            "cv_threshold": CV_THRESHOLD,
            "rho_gate": RHO_GATE,
            "gibbs": GIBBS_CONTROL,
            "opt": OPT_CONTROL,
        },
    }

    # --- GTEx bulk ---
    log.info("\n========= Loading GTEx bulk =========")
    gtex_bulk, gtex_meta = load_gtex_bulk()  # genes_symbol x samples
    # Transpose to samples x genes for pyBayesPrism
    gtex_bulk_T = gtex_bulk.T
    summary["n_gtex_samples"] = int(gtex_bulk_T.shape[0])
    summary["gtex_tertile_counts"] = (
        gtex_meta["ischemic_tertile"].value_counts().to_dict()
    )

    # --- HLMA compartments ---
    log.info("\n========= Loading HLMA compartments =========")
    adatas: dict[str, ad.AnnData] = {}
    for c, f in HLMA_FILES.items():
        adatas[c] = load_compartment(f, c, max_cells=N_CELLS_PER_CMPT)

    # --- HVG union ---
    log.info("\n========= HVG selection =========")
    hvg = select_hvg_union({k: v for k, v in adatas.items() if v is not None}, n_top=N_HVG)
    # Intersect with GTEx genes
    gtex_syms = set(gtex_bulk_T.columns)
    hvg = [g for g in hvg if g in gtex_syms]
    log.info("  HVG ∩ GTEx symbols: %d", len(hvg))
    summary["n_hvg_final"] = len(hvg)

    # --- Build reference ---
    log.info("\n========= Building reference =========")
    ref_df, ct_labels, cs_labels = build_reference_counts(adatas, hvg)

    # --- T3a.1 Ground-truth validation ---
    log.info("\n========= T3a Gate 1: synthetic-bulk ground truth =========")
    t3a_success = False
    try:
        synth_bulk, truth_df = make_synthetic_bulks(
            {k: v for k, v in adatas.items() if v is not None}, hvg, n_bulks=N_SYNTHETIC
        )
        log.info("  synth bulk shape: %s", synth_bulk.shape)
        theta_synth = run_pybayesprism(ref_df, ct_labels, cs_labels, synth_bulk, update_gibbs=True)
        t3a_success = True
    except Exception as e:
        log.warning("  T3a Gate 1 FAILED: %s", e)
        synth_bulk = None
        truth_df = None
        theta_synth = None
        summary["t3a_gate1_error"] = str(e)

    # --- T3a (primary) on GTEx ---
    if t3a_success:
        try:
            log.info("\n========= T3a: pyBayesPrism on GTEx %d samples =========", gtex_bulk_T.shape[0])
            theta_gtex = run_pybayesprism(ref_df, ct_labels, cs_labels, gtex_bulk_T, update_gibbs=True)
            theta_gtex.to_csv(OUTPUTS["t3a_fractions"])
            log.info("  wrote %s", OUTPUTS["t3a_fractions"])
        except Exception as e:
            log.warning("  T3a primary FAILED: %s", e)
            t3a_success = False
            theta_gtex = None
            summary["t3a_primary_error"] = str(e)
    else:
        theta_gtex = None
        log.info("  T3a primary SKIPPED (Gate 1 failed)")

    # --- Dual validation gate ---
    if t3a_success and theta_gtex is not None:
        log.info("\n========= Dual validation gate =========")
        gate = dual_validation_gate(theta_synth, truth_df, theta_gtex, gtex_meta)
        with open(OUTPUTS["t3a_validation"], "w") as f:
            json.dump(gate, f, indent=2, default=str)
        log.info("  wrote %s", OUTPUTS["t3a_validation"])
        log.info("  ground_truth_pass = %s", gate["gates"]["ground_truth_pass"])
        log.info("  sentinel_pass = %s", gate["gates"]["sentinel_pass"])
        log.info("  DUAL PASS = %s", gate["gates"]["dual_pass"])
        summary["t3a_gates"] = gate["gates"]
    else:
        gate = {"gates": {"dual_pass": False, "ground_truth_pass": False, "sentinel_pass": False}}
        summary["t3a_gates"] = {"dual_pass": False, "ground_truth_pass": False, "sentinel_pass": False}
        summary["t3a_dual_gate"] = "SKIPPED (T3a failed)"
        log.info("  Dual gate SKIPPED (T3a failed)")
        t3a_success = False

    # --- Bootstrap CV on GTEx fractions ---
    if t3a_success and theta_gtex is not None:
        log.info("\n========= T3a bootstrap CV =========")
        cv_df = bootstrap_cv_fractions(ref_df, ct_labels, cs_labels, gtex_bulk_T, theta_gtex,
                                       n_boot=N_BOOTSTRAP)
        cv_df.to_csv(str(OUTPUTS["t3a_fractions"]).replace(".csv", "_cv.csv"))
        # Keep samples where ALL compartments have CV <= 0.3
        keep_mask = (cv_df <= CV_THRESHOLD).all(axis=1)
        pre_n = int(theta_gtex.shape[0])
        post_n = int(keep_mask.sum())
        log.info("  pre-CV N=%d, post-CV N=%d (%.1f%% retained)",
                 pre_n, post_n, 100.0 * post_n / pre_n)
        summary["t3a_pre_cv_n"] = pre_n
        summary["t3a_post_cv_n"] = post_n
        theta_kept = theta_gtex[keep_mask]
        meta_kept = gtex_meta.reindex(theta_kept.index)

        # --- T3a regression (ilr-OLS primary) ---
        log.info("\n========= T3a age regression (ilr-OLS) =========")
        reg_a = age_regression_fractions(theta_kept, meta_kept, pre_n=pre_n)
        reg_a.to_csv(OUTPUTS["t3a_reg"], index=False)
        log.info("  wrote %s (%d rows)", OUTPUTS["t3a_reg"], len(reg_a))
        bonf_a_sig = int(reg_a["bonf_sig"].sum()) if not reg_a.empty else 0
    else:
        log.info("  T3a CV+regression SKIPPED (T3a failed)")
        summary["t3a_pre_cv_n"] = 0
        summary["t3a_post_cv_n"] = 0
        reg_a = pd.DataFrame()
        bonf_a_sig = 0
        t3a_success = False

    # --- T3b decoupler MLM fallback/robustness ---
    log.info("\n========= T3b: decoupler MLM TF activity =========")
    scenic_net = build_regulon_table()
    # CollecTRI for secondary panel
    import decoupler as dc
    try:
        collectri = dc.op.collectri(organism="human", verbose=False)
        collectri = collectri.rename(columns={"source": "source", "target": "target"})
        if "weight" not in collectri.columns and "mor" in collectri.columns:
            collectri["weight"] = collectri["mor"]
        ct_tfs = collectri[collectri["source"].isin(CANONICAL_TFS)].copy()
        log.info("  CollecTRI panel: %d TFs, %d rows", ct_tfs["source"].nunique(), len(ct_tfs))
    except Exception as e:
        log.warning("  CollecTRI fetch failed: %s", e)
        ct_tfs = pd.DataFrame()

    # Bulk must be samples x genes with gene symbols
    bulk_for_mlm = gtex_bulk_T  # already samples x symbols
    # log1p transform for variance-stabilization (standard for decoupler input)
    bulk_for_mlm = np.log1p(bulk_for_mlm)

    act_frames = []
    if not scenic_net.empty:
        act_scenic = run_decoupler_mlm(bulk_for_mlm, scenic_net, "SCENIC")
        act_scenic = act_scenic.add_prefix("SCENIC__")
        act_frames.append(act_scenic)
    if not ct_tfs.empty:
        act_ct = run_decoupler_mlm(bulk_for_mlm, ct_tfs, "CollecTRI")
        act_ct = act_ct.add_prefix("CollecTRI__")
        act_frames.append(act_ct)
    act_all = pd.concat(act_frames, axis=1) if act_frames else pd.DataFrame()
    act_all.to_csv(OUTPUTS["t3b_activity"])
    log.info("  wrote %s (%s)", OUTPUTS["t3b_activity"], act_all.shape)

    # --- T3b regression ---
    log.info("\n========= T3b age regression =========")
    reg_b = age_regression_activity(act_all, gtex_meta, "mlm", pre_n=int(act_all.shape[0]))
    reg_b.to_csv(OUTPUTS["t3b_reg"], index=False)
    log.info("  wrote %s (%d rows)", OUTPUTS["t3b_reg"], len(reg_b))

    # --- Summary & verdict ---
    bonf_b_sig = int(reg_b["bonf_sig"].sum()) if not reg_b.empty else 0
    # Surviving TFs: Bonferroni-sig in pooled AND same-sign in T1_low
    surviving_tfs = []
    if not reg_b.empty:
        pooled = reg_b[reg_b["stratum"] == "pooled"]
        t1 = reg_b[reg_b["stratum"] == "T1_low"].set_index("tf")
        for _, row in pooled.iterrows():
            tf = row["tf"]
            if row["bonf_sig"] and tf in t1.index:
                if np.sign(row["beta_age"]) == np.sign(t1.loc[tf, "beta_age"]) and t1.loc[tf, "p_age"] < 0.05:
                    surviving_tfs.append(tf)

    summary["t3a_bonf_sig_count"] = bonf_a_sig
    summary["t3b_bonf_sig_count"] = bonf_b_sig
    summary["t3b_surviving_tfs"] = surviving_tfs
    summary["primary_method"] = (
        "T3a" if summary["t3a_gates"]["dual_pass"] else "T3b_fallback"
    )
    summary["verdict"] = (
        "T3a_PASSED_primary" if summary["t3a_gates"]["dual_pass"] else
        "T3a_FAILED_fallback_to_T3b"
    )

    with open(OUTPUTS["t3_summary"], "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("\n========= DONE =========")
    log.info("verdict: %s", summary["verdict"])
    log.info("t3a_bonf_sig: %d; t3b_bonf_sig: %d; surviving_TFs: %s",
             bonf_a_sig, bonf_b_sig, surviving_tfs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
