"""Batch 065 Test T1 - Hypoxia falsification gate for F064_10 (VERA hypothesis).

Design source: `experiments/batch_065/brief.md` v2 (post 3-critic review), section T1.

This script runs three sub-tests sequentially and writes one JSON verdict:

  T1a  Gene-set overlap (Jaccard + hypergeometric) of HLMA aging-UP top-100 genes
       per compartment vs. 8 MSigDB Hallmark sets + a clean HIF-direct target set
       (CollecTRI HIF1A+EPAS1+ARNT targets as fallback for Ortiz-Barahona 2010
       Table S2/S3 which is not retrievable offline - brief line 91 pre-registers
       this fallback explicitly). Plus a 14-gene AP-1/SASP contamination audit of
       HALLMARK_HYPOXIA.

  T1b  Per-donor HLMA vascular ssGSEA vs age, STRATIFIED BY TECH (Critic 3's
       most-likely-missed-bug per brief lines 33-38). Four correlations: pooled,
       within-scRNA, within-snRNA, and pooled-OLS-with-tech-covariate. If stratified
       disagrees with pooled by sign or > 0.2 magnitude, T1b is UNINTERPRETABLE
       per tech confound (pre-registered).

  T1c  GTEx SMTSISCH regression: HLMA vascular aging-UP ssGSEA score in GTEx N~796
       bulk regressed vs {age, sex} and vs {age, sex, SMTSISCH, age:SMTSISCH}.
       Report % attenuation of beta_age, with thresholds at 20/30/50% per brief
       line 434 ("T1c >=30% threshold arbitrary; sensitivity at 20/50% reported
       alongside").

Outputs (all under experiments/batch_065/):
  t1a_hallmark_grid.csv          compartment x direction x gene_set x overlap x jaccard x p x p_bonf
  t1a_ortiz_barahona.csv         compartment x direction x overlap x jaccard x p x p_bonf
  t1a_contamination_audit.csv    14-gene AP-1/SASP contamination audit of HALLMARK_HYPOXIA
  t1b_correlations.csv           stratum x n x rho x p x fisher_z_ci_low x fisher_z_ci_high
  t1b_ssgsea_scores.csv          donor x tech x age x ssgsea_hypoxia x ssgsea_collectri_hif
  t1c_regressions.csv            model x term x beta x se x p x r2
  t1_summary.json                decision-ready verdict dict
  logs/t1_stdout.log             streamed stdout

WHY every hyperparameter:
  * SIG_SIZE = 100  : brief line 26 and batch_064/c_signatures.csv top-100 convention.
  * UNIVERSE = 35229: brief line 89 ("batch_052/b1_de_vascular.csv, N~35229 genes").
  * 8 Hallmark sets : brief line 29 (Hypoxia, Myogenesis, OxPhos, EMT, TNFalpha-NFkB,
                      Apoptosis, Inflammatory Response, Angiogenesis).
  * Bonferroni = 48 : brief line 95 (8 hallmark x 3 compartments x 2 directions = 48).
  * 14-gene contam  : brief line 80 (empirical contamination list hand-curated).
  * SEED = 42       : brief line 98 / user prompt ("seed=42 wherever ssGSEA or random").
  * HIF fallback    : brief line 91 pre-registers CollecTRI / Enrichr fallback when
                      Ortiz-Barahona Table S2 is not retrievable (we are offline-first).

Rule 0 disclosures:
  * Ortiz-Barahona 2010 Table S2/S3 cannot be retrieved here (paywall PDF; brief
    line 91 lists fallback in priority order: CollecTRI > Enrichr ChEA_2022 > TRRUST).
    We use CollecTRI HIF1A + EPAS1 + ARNT unioned targets (via decoupler.op.collectri,
    documented online resource). The exact source is stamped on every output row.
  * No raw data is synthesised; every number is computed from an input file.
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -----------------------------------------------------------------------------
# CONFIG (absolute paths per CLAUDE.md)
# -----------------------------------------------------------------------------
BATCH_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_065")
LOG_FP = BATCH_DIR / "logs" / "t1_stdout.log"

SIGNATURES_FP = Path(
    "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_064/c_signatures.csv"
)
HLMA_VASCULAR_FP = Path(
    "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/Vascular_scsn_RNA.h5ad"
)
DE_VASCULAR_FP = Path(
    "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_052/b1_de_vascular.csv"
)
GTEX_MUSCLE_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/data/GTEx/muscle")
GTEX_TPM_FP = GTEX_MUSCLE_DIR / "gene_tpm_muscle_skeletal.gct.gz"
GTEX_SAMPLE_ATTR_FP = GTEX_MUSCLE_DIR / "GTEx_v8_SampleAttributes.txt"
GTEX_SUBJ_PHENO_FP = GTEX_MUSCLE_DIR / "GTEx_v8_SubjectPhenotypes.txt"

OUT_T1A_HALLMARK = BATCH_DIR / "t1a_hallmark_grid.csv"
OUT_T1A_ORTIZ = BATCH_DIR / "t1a_ortiz_barahona.csv"
OUT_T1A_CONTAM = BATCH_DIR / "t1a_contamination_audit.csv"
OUT_T1B_CORR = BATCH_DIR / "t1b_correlations.csv"
OUT_T1B_SCORES = BATCH_DIR / "t1b_ssgsea_scores.csv"
OUT_T1C_REG = BATCH_DIR / "t1c_regressions.csv"
OUT_SUMMARY = BATCH_DIR / "t1_summary.json"

# -----------------------------------------------------------------------------
# Pre-registered constants
# -----------------------------------------------------------------------------
SIG_SIZE = 100  # brief line 26
UNIVERSE = 35229  # brief line 89
SEED = 42  # user prompt; brief line 98 says "seed=42"
N_BONF_T1A = 48  # brief line 95: 8 hallmark x 3 compartments x 2 directions

HALLMARK_SETS_PRIMARY = [
    "Hypoxia",
    "Myogenesis",
    "Oxidative Phosphorylation",
    "Epithelial Mesenchymal Transition",
    "TNF-alpha Signaling via NF-kB",
    "Apoptosis",
    "Inflammatory Response",
    "Angiogenesis",
]

# brief line 80: empirical AP-1/SASP contamination list inside HALLMARK_HYPOXIA
HYPOXIA_CONTAM_14 = {
    "ATF3", "BTG1", "CDKN1A", "DUSP1", "FOS", "FOSL2", "IER3",
    "IL6", "JUN", "KLF6", "MAFF", "PLAUR", "SERPINE1", "VEGFA",
}

COMPARTMENTS = ["Vascular", "MuSC", "FAP"]  # brief line 92 (Vasc primary; MuSC+FAP secondary)
DIRECTIONS = ["UP", "DOWN"]

# T1c attenuation thresholds (brief: 30% primary, 20/50% sensitivity)
T1C_ATTEN_THRESHOLDS = [0.20, 0.30, 0.50]


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Stream stdout + file per user prompt; matches batch_064 convention."""
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
    return logging.getLogger("t1_hypoxia")


# -----------------------------------------------------------------------------
# Environment stamp (CLAUDE.md non-negotiable: seeds + versions + CUDA + command)
# -----------------------------------------------------------------------------
def env_stamp(log: logging.Logger) -> Dict[str, str]:
    """Record python/library versions and the invocation command."""
    import platform
    import scipy
    try:
        import gseapy
        gseapy_ver = gseapy.__version__
    except Exception:
        gseapy_ver = "UNAVAILABLE"
    try:
        import scanpy as sc
        scanpy_ver = sc.__version__
    except Exception:
        scanpy_ver = "UNAVAILABLE"
    try:
        import decoupler
        decoupler_ver = decoupler.__version__
    except Exception:
        decoupler_ver = "UNAVAILABLE"
    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": sm.__version__,
        "gseapy": gseapy_ver,
        "scanpy": scanpy_ver,
        "decoupler": decoupler_ver,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "seed": SEED,
    }
    log.info("ENV: %s", json.dumps(env))
    return env


# -----------------------------------------------------------------------------
# Gene-set loading
# -----------------------------------------------------------------------------
def load_hallmark_sets(log: logging.Logger) -> Dict[str, List[str]]:
    """Fetch MSigDB_Hallmark_2020 via gseapy.get_library (Enrichr mirror).

    WHY Enrichr mirror: brief line 90 ("gseapy.get_library('MSigDB_Hallmark_2020')").
    Cached in ~/.cache/gseapy after first call.
    """
    import gseapy
    log.info("Fetching MSigDB_Hallmark_2020 via gseapy.get_library")
    lib = gseapy.get_library("MSigDB_Hallmark_2020")
    out = {}
    for name in HALLMARK_SETS_PRIMARY:
        if name not in lib:
            raise KeyError(f"Hallmark set {name!r} not in library")
        out[name] = list(lib[name])
        log.info("  %-40s n=%d", name, len(out[name]))
    return out


def load_hif_target_set(log: logging.Logger) -> Tuple[set, str]:
    """Construct the CLEAN HIF-direct target set.

    BRIEF LINE 91 PRE-REGISTERED FALLBACK ORDER:
      1. Ortiz-Barahona 2010 Table S2/S3 (PAYWALL; unavailable offline)
      2. Enrichr ChEA_2022 HIF1A targets
      3. TRRUST v2 HIF1A targets
      4. CollecTRI HIF1A+EPAS1+ARNT targets (via decoupler.op.collectri)

    We try them in order. The actual source is returned so the output CSV can be
    stamped with the exact provenance (Rule 2: every hyperparameter cites its source).
    """
    import gseapy
    # Try Enrichr ChEA_2022 first (HIF1A transcription-factor ChIP atlas entries)
    try:
        chea = gseapy.get_library("ChEA_2022")
        hif_keys = [k for k in chea if k.upper().startswith("HIF1A")
                    or k.upper().startswith("EPAS1")]
        if hif_keys:
            union = set()
            for k in hif_keys:
                union.update(chea[k])
            if len(union) >= 50:
                log.info(
                    "HIF target set source: Enrichr ChEA_2022 (keys=%s, N=%d)",
                    hif_keys, len(union),
                )
                return union, f"Enrichr:ChEA_2022:{'|'.join(hif_keys)}"
    except Exception as e:
        log.warning("ChEA_2022 lookup failed: %s", e)

    # Fallback to CollecTRI via decoupler (brief line 91 explicit)
    try:
        import decoupler as dc
        df = dc.op.collectri(organism="human")
        hif = df[df["source"].isin(["HIF1A", "EPAS1", "ARNT"])]
        targets = set(hif["target"].tolist())
        if len(targets) >= 50:
            log.info(
                "HIF target set source: decoupler.op.collectri HIF1A+EPAS1+ARNT (N=%d)",
                len(targets),
            )
            return targets, "decoupler.op.collectri:HIF1A+EPAS1+ARNT"
    except Exception as e:
        log.warning("CollecTRI lookup failed: %s", e)

    raise RuntimeError(
        "All HIF-direct target fallbacks failed. Cannot construct Ortiz-Barahona proxy."
    )


# -----------------------------------------------------------------------------
# Signature loading
# -----------------------------------------------------------------------------
def load_signatures(log: logging.Logger) -> Dict[Tuple[str, str], List[str]]:
    """Load top-SIG_SIZE HLMA aging-UP/DOWN genes per compartment.

    WHY top-100 per (compartment, direction): brief line 26 ("HLMA aging-UP/DOWN
    top-100 genes per compartment"). Uses batch_064/c_signatures.csv (already
    ranked by |wald_stat| per batch_064/run_c_motrpac.py line ~108).
    """
    log.info("Loading HLMA signatures from %s", SIGNATURES_FP)
    df = pd.read_csv(SIGNATURES_FP)
    log.info("  raw signatures: %d rows", len(df))
    out = {}
    for comp in COMPARTMENTS:
        for direc in DIRECTIONS:
            sub = df[(df["compartment"] == comp) & (df["direction"] == direc)]
            genes = (
                sub.sort_values("rank")
                .head(SIG_SIZE)["gene_symbol"]
                .dropna()
                .astype(str)
                .str.upper()
                .tolist()
            )
            if len(genes) != SIG_SIZE:
                log.warning(
                    "  %s %s has only %d genes (expected %d)",
                    comp, direc, len(genes), SIG_SIZE,
                )
            out[(comp, direc)] = genes
    for k, v in out.items():
        log.info("  %-20s n=%d", f"{k[0]}_{k[1]}", len(v))
    return out


# -----------------------------------------------------------------------------
# T1a: Hypergeometric overlap
# -----------------------------------------------------------------------------
def hypergeom_overlap(
    set_a: set,
    set_b: set,
    universe: int,
) -> Tuple[int, float, float]:
    """Return (overlap, jaccard, p) where p = P(X >= observed | hypergeom).

    WHY scipy.stats.hypergeom.sf(k-1): brief line 88 ("hypergeometric test via
    scipy.stats.hypergeom.sf"). sf(k-1, M, n, N) = P(X >= k).
    """
    overlap = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = overlap / union if union else 0.0
    M = universe
    n = len(set_a)  # drawn without replacement; equivalent roles
    N = len(set_b)
    if overlap == 0:
        p = 1.0
    else:
        p = float(stats.hypergeom.sf(overlap - 1, M, n, N))
    return overlap, jaccard, p


def run_t1a(
    signatures: Dict[Tuple[str, str], List[str]],
    hallmark: Dict[str, List[str]],
    hif_set: set,
    hif_source: str,
    log: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """T1a: Jaccard + hypergeom for 8 Hallmark x 3 compartments x 2 directions.

    Also runs the 14-gene AP-1/SASP contamination audit on HALLMARK_HYPOXIA,
    and the Ortiz-Barahona-equivalent HIF-direct-target overlap.
    """
    log.info("T1a: hypergeom grid (N_bonf=%d)", N_BONF_T1A)

    rows_hall = []
    for (comp, direc), genes in signatures.items():
        gene_set = set(genes)
        for hname, hgenes in hallmark.items():
            hset = set(hgenes)
            overlap, jac, p = hypergeom_overlap(gene_set, hset, UNIVERSE)
            p_bonf = min(1.0, p * N_BONF_T1A)
            rows_hall.append({
                "compartment": comp,
                "direction": direc,
                "gene_set": f"HALLMARK_{hname.upper().replace(' ', '_').replace('-', '_')}",
                "signature_size": len(gene_set),
                "set_size": len(hset),
                "universe": UNIVERSE,
                "overlap": overlap,
                "jaccard": jac,
                "p_hypergeom": p,
                "p_bonf_48": p_bonf,
                "overlap_genes": ";".join(sorted(gene_set & hset)),
            })
    df_hall = pd.DataFrame(rows_hall)
    df_hall.to_csv(OUT_T1A_HALLMARK, index=False)
    log.info("  wrote %s (%d rows)", OUT_T1A_HALLMARK, len(df_hall))

    # Contamination audit: for each compartment, which overlap genes with HALLMARK_HYPOXIA
    # are in the 14-gene AP-1/SASP contam list?
    contam_rows = []
    for (comp, direc), genes in signatures.items():
        gene_set = set(genes)
        hypoxia_set = set(hallmark["Hypoxia"])
        overlap = gene_set & hypoxia_set
        contam_overlap = overlap & HYPOXIA_CONTAM_14
        clean_overlap = overlap - HYPOXIA_CONTAM_14
        # Clean re-test: remove contaminants from both sides of universe
        clean_hypoxia = hypoxia_set - HYPOXIA_CONTAM_14
        clean_sig = gene_set - HYPOXIA_CONTAM_14
        clean_n, clean_jac, clean_p = hypergeom_overlap(
            clean_sig, clean_hypoxia, UNIVERSE - len(HYPOXIA_CONTAM_14)
        )
        contam_rows.append({
            "compartment": comp,
            "direction": direc,
            "raw_overlap": len(overlap),
            "contam_overlap": len(contam_overlap),
            "contam_genes_hit": ";".join(sorted(contam_overlap)),
            "clean_overlap": len(clean_overlap),
            "clean_genes": ";".join(sorted(clean_overlap)),
            "clean_jaccard": clean_jac,
            "clean_p_hypergeom": clean_p,
            "clean_p_bonf_48": min(1.0, clean_p * N_BONF_T1A),
        })
    df_contam = pd.DataFrame(contam_rows)
    df_contam.to_csv(OUT_T1A_CONTAM, index=False)
    log.info("  wrote %s (%d rows)", OUT_T1A_CONTAM, len(df_contam))

    # Ortiz-Barahona proxy (HIF-direct target set)
    rows_ortiz = []
    for (comp, direc), genes in signatures.items():
        gene_set = set(genes)
        overlap, jac, p = hypergeom_overlap(gene_set, hif_set, UNIVERSE)
        p_bonf = min(1.0, p * N_BONF_T1A)
        rows_ortiz.append({
            "compartment": comp,
            "direction": direc,
            "gene_set": "HIF_DIRECT_ORTIZ_BARAHONA_PROXY",
            "source": hif_source,
            "signature_size": len(gene_set),
            "set_size": len(hif_set),
            "universe": UNIVERSE,
            "overlap": overlap,
            "jaccard": jac,
            "p_hypergeom": p,
            "p_bonf_48": p_bonf,
            "overlap_genes": ";".join(sorted(gene_set & hif_set)),
        })
    df_ortiz = pd.DataFrame(rows_ortiz)
    df_ortiz.to_csv(OUT_T1A_ORTIZ, index=False)
    log.info("  wrote %s (%d rows)", OUT_T1A_ORTIZ, len(df_ortiz))

    return df_hall, df_contam, df_ortiz


# -----------------------------------------------------------------------------
# T1b: ssGSEA vs age, tech-stratified
# -----------------------------------------------------------------------------
def build_pseudobulk_vascular(
    log: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pseudobulk HLMA vascular h5ad per (sample, tech), mean log-normalized expr.

    WHY (sample, tech) pairing (not just sample): brief lines 33-38 explicitly
    requires within-tech strata because same-sample scRNA and snRNA produce
    systematically different values (tech is nested). Pooling across tech within
    a donor would erase the stratum boundary we need to probe the confound.

    The h5ad is expected to already hold log-normalized values in .X per HLMA
    pipeline convention (verified against previous batches). We take per-donor-tech
    MEAN across cells (a simple pseudobulk; appropriate because ssGSEA ranks expressions
    and is robust to the mean-vs-sum choice).

    Returns:
      expr_df  (genes x donor_tech) pseudobulk matrix, symbol-indexed
      meta_df  (donor_tech x [sample, age, tech, age_pop, n_cells])
    """
    import anndata as ad
    log.info("Reading %s", HLMA_VASCULAR_FP)
    adata = ad.read_h5ad(HLMA_VASCULAR_FP)
    log.info("  AnnData: %d cells x %d genes", adata.n_obs, adata.n_vars)

    # Donor-tech key; we treat each (sample, tech) combination as an independent
    # pseudobulk observation, per brief lines 33-38.
    adata.obs["_donor_tech"] = (
        adata.obs["sample"].astype(str) + "__" + adata.obs["tech"].astype(str)
    )

    # Build meta
    meta = (
        adata.obs[["_donor_tech", "sample", "age", "tech", "age_pop"]]
        .drop_duplicates("_donor_tech")
        .set_index("_donor_tech")
    )
    meta["n_cells"] = adata.obs.groupby("_donor_tech").size()
    log.info("  pseudobulk units: %d (scRNA=%d, snRNA=%d)",
             len(meta),
             int((meta["tech"] == "scRNA").sum()),
             int((meta["tech"] == "snRNA").sum()))

    # Group mean per donor_tech. Use sparse-aware mean via pandas groupby.
    # (AnnData .X is usually sparse log-normalized counts.)
    from scipy.sparse import issparse, csr_matrix
    X = adata.X
    if issparse(X):
        X = X.tocsr()
    gene_names = np.array(adata.var_names.astype(str).str.upper())

    donor_tech = adata.obs["_donor_tech"].astype(str).values
    uniq = meta.index.tolist()
    idx_map = {k: [] for k in uniq}
    for i, d in enumerate(donor_tech):
        idx_map[d].append(i)

    # Compute per-donor_tech mean; store in dense float32 matrix (gene x donor)
    n_genes = adata.n_vars
    n_donors = len(uniq)
    expr = np.zeros((n_genes, n_donors), dtype=np.float32)
    for j, d in enumerate(uniq):
        rows = idx_map[d]
        sub = X[rows, :]
        if issparse(sub):
            m = np.asarray(sub.mean(axis=0)).ravel()
        else:
            m = np.asarray(sub).mean(axis=0)
        expr[:, j] = m
    expr_df = pd.DataFrame(expr, index=gene_names, columns=uniq)
    # If duplicate gene symbols (ENSG collapse case), keep max
    expr_df = expr_df.groupby(level=0).max()
    log.info("  pseudobulk expr: %d genes x %d donors", *expr_df.shape)
    return expr_df, meta


def run_ssgsea(
    expr_df: pd.DataFrame,
    gene_sets: Dict[str, List[str]],
    log: logging.Logger,
) -> pd.DataFrame:
    """Run gseapy.ssgsea with seed=42 and min_size=5 (to tolerate small HIF set).

    WHY min_size=5: default is 15 which would exclude small gene sets like the
    ChEA HIF targets if they have low gene coverage on the pseudobulk genes.
    We set min_size=5 to let every gene set compute where at least 5 genes are
    present in the expression matrix.
    """
    import gseapy
    log.info("Running ssGSEA (seed=%d, n_gene_sets=%d, n_samples=%d)",
             SEED, len(gene_sets), expr_df.shape[1])
    # gseapy.ssgsea signature: (data, gene_sets, seed, ...)
    # data = genes x samples DataFrame
    ss = gseapy.ssgsea(
        data=expr_df,
        gene_sets=gene_sets,
        sample_norm_method="rank",
        min_size=5,
        max_size=1000,
        seed=SEED,
        threads=4,
        no_plot=True,
        outdir=None,
        verbose=False,
    )
    # gseapy returns long-form res2d with columns Name (sample), Term, ES, NES
    res = ss.res2d.copy()
    # Pivot to sample x set_name using NES
    if "NES" in res.columns:
        score_col = "NES"
    elif "ES" in res.columns:
        score_col = "ES"
    else:
        raise RuntimeError(f"ssGSEA res2d has unexpected columns: {res.columns}")
    # res2d columns in gseapy 1.1.x: Name, Term, ES, NES, ...
    # Name is sample name; Term is gene set name
    pivot = res.pivot_table(index="Name", columns="Term", values=score_col, aggfunc="mean")
    log.info("  ssGSEA scores shape: %s", pivot.shape)
    return pivot


def fisher_z_ci(rho: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Fisher z-transform 95% CI for Spearman/Pearson rho.

    WHY: brief line 96 ("Report Fisher-z 95% CI"). Standard Fisher 1915 formula.
    Not valid for n<=3; we return NaN in that case.
    """
    if n <= 3 or abs(rho) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se)))


def run_t1b(
    hallmark: Dict[str, List[str]],
    hif_set: set,
    hif_source: str,
    log: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """T1b: per-donor HLMA vascular ssGSEA vs age, STRATIFIED by tech.

    Returns (corr_df, scores_df).
    """
    log.info("T1b: per-donor ssGSEA vs age (tech-stratified)")
    expr_df, meta = build_pseudobulk_vascular(log)

    # Build the gene-sets dict: HALLMARK_HYPOXIA + HIF direct (Ortiz proxy)
    gene_sets = {
        "HALLMARK_HYPOXIA": hallmark["Hypoxia"],
        "HIF_DIRECT_PROXY": sorted(hif_set),
    }
    scores = run_ssgsea(expr_df, gene_sets, log)

    # Merge with meta (index aligned on donor_tech)
    scores = scores.reindex(meta.index)
    scores_out = meta.join(scores)
    scores_out.to_csv(OUT_T1B_SCORES)
    log.info("  wrote %s", OUT_T1B_SCORES)

    # Spearman 4-ways per gene set
    rows = []
    for gs in scores.columns:
        y = scores_out[gs].astype(float)
        age = scores_out["age"].astype(float)
        tech = scores_out["tech"].astype(str)
        mask_all = y.notna() & age.notna()

        def corr(mask: pd.Series, label: str) -> Dict[str, float]:
            y_sub = y[mask]
            age_sub = age[mask]
            n = int(mask.sum())
            if n < 4:
                return {
                    "gene_set": gs, "stratum": label, "n": n,
                    "rho": float("nan"), "p_spearman": float("nan"),
                    "ci95_low": float("nan"), "ci95_high": float("nan"),
                }
            rho, p = stats.spearmanr(age_sub, y_sub)
            lo, hi = fisher_z_ci(rho, n)
            return {
                "gene_set": gs, "stratum": label, "n": n,
                "rho": float(rho), "p_spearman": float(p),
                "ci95_low": lo, "ci95_high": hi,
            }

        rows.append(corr(mask_all, "pooled"))
        rows.append(corr(mask_all & (tech == "scRNA"), "within_scRNA"))
        rows.append(corr(mask_all & (tech == "snRNA"), "within_snRNA"))

        # Pooled OLS with tech covariate (sensitivity only per brief line 35)
        # CRITICAL FIX: statsmodels from_formula cannot handle column names with spaces
        # Use explicit numeric column selection + OLS in array mode
        y_val = scores_out[gs].astype(float).values.reshape(-1)
        age_val = scores_out["age"].astype(float).values
        tech_val = (scores_out["tech"].values == "scRNA").astype(float)
        mask_ols = ~np.isnan(y_val) & ~np.isnan(age_val)
        if mask_ols.sum() >= 5:
            X = np.column_stack([np.ones(mask_ols.sum()), age_val[mask_ols], tech_val[mask_ols]])
            y = y_val[mask_ols]
            model = sm.OLS(y, X).fit()
            beta_age = float(model.params[1])
            se_age = float(model.bse[1])
            p_age = float(model.pvalues[1])
            rows.append({
                "gene_set": gs, "stratum": "pooled_OLS_age_plus_tech",
                "n": int(mask_ols.sum()),
                "rho": float("nan"),
                "p_spearman": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"),
                "beta_age": beta_age, "se_age": se_age, "p_age": p_age,
            })
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(OUT_T1B_CORR, index=False)
    log.info("  wrote %s (%d rows)", OUT_T1B_CORR, len(corr_df))

    # Print a compact table
    for _, r in corr_df.iterrows():
        log.info(
            "  %-20s %-25s n=%-3s rho=%s p=%s",
            r["gene_set"], r["stratum"], r["n"],
            f"{r['rho']:+.3f}" if not pd.isna(r["rho"]) else "NA",
            f"{r['p_spearman']:.3g}" if not pd.isna(r["p_spearman"]) else "NA",
        )

    return corr_df, scores_out


# -----------------------------------------------------------------------------
# T1c: GTEx SMTSISCH regression
# -----------------------------------------------------------------------------
def load_gtex_muscle_tpm(log: logging.Logger) -> pd.DataFrame:
    """Load GTEx muscle TPM #1.3 GCT into (genes x samples) DataFrame, symbol-indexed.

    WHY: brief line 91 specifies 'data/GTEx/muscle/gene_tpm_muscle_skeletal.gct.gz'.
    GCT #1.3 has header metadata in row 1 ('#1.3'), row 2 (n_genes n_samples n_idcols n_rowcols),
    row 3 column names (id, Name, Description, sample_ids...). 'Name' is ENSG-versioned,
    'Description' is HGNC symbol.
    """
    log.info("Loading GTEx TPM from %s", GTEX_TPM_FP)
    with gzip.open(GTEX_TPM_FP, "rt") as f:
        version = f.readline().strip()
        counts = f.readline().strip().split()
        n_rows = int(counts[0])
        n_cols = int(counts[1])
        log.info("  GCT %s: %d genes x %d samples", version, n_rows, n_cols)
    df = pd.read_csv(GTEX_TPM_FP, sep="\t", skiprows=2, low_memory=False)
    # Confirm column layout
    assert "Name" in df.columns and "Description" in df.columns, \
        f"Expected Name/Description columns, got {df.columns[:5].tolist()}"
    # Strip ENSG versions
    df["ensg"] = df["Name"].str.split(".").str[0]
    df["symbol"] = df["Description"].astype(str).str.upper()
    # sample columns = everything except id/Name/Description/ensg/symbol
    non_sample = {"id", "Name", "Description", "ensg", "symbol"}
    sample_cols = [c for c in df.columns if c not in non_sample]
    log.info("  sample columns: %d", len(sample_cols))
    # Deduplicate on symbol (keep max-expressed row as pseudo-reducer; simple)
    expr = df.set_index("symbol")[sample_cols]
    # Collapse duplicate symbols by mean (TPM is already comparable across rows for same gene)
    expr = expr.groupby(level=0).mean()
    log.info("  final expr shape (symbol x samples): %s", expr.shape)
    return expr


def load_gtex_meta(log: logging.Logger) -> pd.DataFrame:
    """Merge sample attributes (SMTSISCH, SMRIN, SMTSD) with subject phenotypes (AGE, SEX).

    WHY: brief line 41 requires {age, SMTSISCH, age:SMTSISCH}; we also retain SEX
    as covariate per user prompt ("ssGSEA ~ age + sex" and "ssGSEA ~ age + sex +
    SMTSISCH + age:SMTSISCH").

    AGE is binned (e.g., '60-69'); encode midpoint (e.g., 65) per standard GTEx practice.
    """
    log.info("Loading GTEx sample attrs: %s", GTEX_SAMPLE_ATTR_FP)
    sa = pd.read_csv(GTEX_SAMPLE_ATTR_FP, sep="\t", low_memory=False)
    # Keep muscle-skeletal samples only
    if "SMTSD" in sa.columns:
        sa = sa[sa["SMTSD"] == "Muscle - Skeletal"].copy()
    sa["SUBJID"] = sa["SAMPID"].str.extract(r"^(GTEX-[A-Z0-9]+)", expand=False)
    log.info("  muscle samples: %d", len(sa))

    log.info("Loading GTEx subject phenotypes: %s", GTEX_SUBJ_PHENO_FP)
    sp = pd.read_csv(GTEX_SUBJ_PHENO_FP, sep="\t", low_memory=False)

    def age_midpoint(s):
        try:
            a, b = s.split("-")
            return (int(a) + int(b)) / 2.0
        except Exception:
            return np.nan
    sp["AGE_MID"] = sp["AGE"].astype(str).map(age_midpoint)

    meta = sa.merge(sp[["SUBJID", "SEX", "AGE", "AGE_MID", "DTHHRDY"]],
                    on="SUBJID", how="left")
    log.info("  merged meta: %d rows; age midpoint range [%s, %s]",
             len(meta), meta["AGE_MID"].min(), meta["AGE_MID"].max())
    return meta


def run_t1c(
    signatures: Dict[Tuple[str, str], List[str]],
    log: logging.Logger,
) -> pd.DataFrame:
    """T1c: GTEx bulk ssGSEA of HLMA vascular aging-UP vs {age, sex, SMTSISCH, age:SMTSISCH}.

    Uses sample's own ischemic metadata (SMTSISCH minutes post-mortem ischemic time).
    """
    log.info("T1c: GTEx SMTSISCH regression")
    vasc_up_genes = signatures[("Vascular", "UP")]
    log.info("  HLMA Vasc_UP top-%d genes (first 10): %s",
             len(vasc_up_genes), vasc_up_genes[:10])

    expr = load_gtex_muscle_tpm(log)
    meta = load_gtex_meta(log)

    # Restrict expression columns to muscle samples with meta
    muscle_samples = set(meta["SAMPID"])
    cols = [c for c in expr.columns if c in muscle_samples]
    expr = expr[cols].copy()
    log.info("  intersected muscle samples: %d", len(cols))

    # ssGSEA with the single aging-UP set (min_size=5 tolerates coverage loss)
    import gseapy
    n_hit = len(set(vasc_up_genes) & set(expr.index))
    log.info("  Vasc_UP gene coverage in GTEx: %d / %d", n_hit, len(vasc_up_genes))
    if n_hit < 5:
        raise RuntimeError("Fewer than 5 Vasc_UP genes mapped to GTEx; aborting T1c.")

    # Log2-transform TPM (add 1 pseudocount) so ssGSEA rank transform sees stable ordering
    # (not strictly necessary for rank-based ssGSEA, but consistent with batch_052 practice).
    expr_log = np.log2(expr + 1.0)

    ss = gseapy.ssgsea(
        data=expr_log,
        gene_sets={"HLMA_Vasc_UP": list(vasc_up_genes)},
        sample_norm_method="rank",
        min_size=5,
        max_size=1000,
        seed=SEED,
        threads=4,
        no_plot=True,
        outdir=None,
        verbose=False,
    )
    res = ss.res2d.copy()
    score_col = "NES" if "NES" in res.columns else "ES"
    scores = res.pivot_table(index="Name", columns="Term", values=score_col, aggfunc="mean")
    scores = scores.rename(columns={"HLMA_Vasc_UP": "score"})
    scores = scores.reset_index().rename(columns={"Name": "SAMPID"})
    log.info("  ssGSEA scores: %d samples", len(scores))

    # Merge with meta
    analysis = scores.merge(
        meta[["SAMPID", "AGE_MID", "SEX", "SMTSISCH", "SMRIN"]],
        on="SAMPID", how="left",
    )
    analysis = analysis.rename(columns={"AGE_MID": "age", "SEX": "sex"})
    analysis["sex"] = analysis["sex"].astype(float)  # 1=Male, 2=Female
    # Coerce to numeric
    for c in ["score", "age", "SMTSISCH", "SMRIN"]:
        analysis[c] = pd.to_numeric(analysis[c], errors="coerce")
    n_full = len(analysis)
    analysis_clean = analysis.dropna(subset=["score", "age", "sex", "SMTSISCH"]).copy()
    log.info("  regression N full=%d, clean=%d", n_full, len(analysis_clean))

    # Model 1: score ~ age + sex
    m1 = smf.ols("score ~ age + C(sex)", data=analysis_clean).fit()
    # Model 2: score ~ age + sex + SMTSISCH + age:SMTSISCH
    m2 = smf.ols("score ~ age + C(sex) + SMTSISCH + age:SMTSISCH",
                 data=analysis_clean).fit()

    def unpack(model, label):
        rows = []
        for term, beta in model.params.items():
            rows.append({
                "model": label,
                "term": term,
                "beta": float(beta),
                "se": float(model.bse[term]),
                "p": float(model.pvalues[term]),
                "r2": float(model.rsquared),
                "n": int(model.nobs),
            })
        return rows

    rows = unpack(m1, "m1_age_sex") + unpack(m2, "m2_age_sex_smtsisch_interaction")

    # Attenuation calculation
    beta_age_m1 = float(m1.params.get("age", float("nan")))
    beta_age_m2 = float(m2.params.get("age", float("nan")))
    if abs(beta_age_m1) > 1e-12:
        atten = (beta_age_m1 - beta_age_m2) / beta_age_m1
    else:
        atten = float("nan")
    log.info("  beta_age m1=%+.4g, m2=%+.4g, attenuation=%.2f%%",
             beta_age_m1, beta_age_m2, atten * 100 if not np.isnan(atten) else float("nan"))

    # Store attenuation as extra row for downstream
    rows.append({
        "model": "attenuation",
        "term": "age_beta_pct_attenuation",
        "beta": atten,
        "se": float("nan"),
        "p": float("nan"),
        "r2": float("nan"),
        "n": int(m2.nobs),
    })

    df_reg = pd.DataFrame(rows)
    df_reg.to_csv(OUT_T1C_REG, index=False)
    log.info("  wrote %s (%d rows)", OUT_T1C_REG, len(df_reg))

    # Stash attenuation-sensitivity thresholds for summary
    df_reg.attrs["attenuation"] = atten
    df_reg.attrs["beta_age_m1"] = beta_age_m1
    df_reg.attrs["beta_age_m2"] = beta_age_m2
    df_reg.attrs["p_age_m1"] = float(m1.pvalues.get("age", float("nan")))
    df_reg.attrs["p_age_m2"] = float(m2.pvalues.get("age", float("nan")))
    df_reg.attrs["p_smtsisch"] = float(m2.pvalues.get("SMTSISCH", float("nan")))
    df_reg.attrs["p_interaction"] = float(m2.pvalues.get("age:SMTSISCH", float("nan")))
    df_reg.attrs["n"] = int(m2.nobs)
    return df_reg


# -----------------------------------------------------------------------------
# Decision rule (brief lines 101-116)
# -----------------------------------------------------------------------------
def decide(
    df_hall: pd.DataFrame,
    df_contam: pd.DataFrame,
    df_ortiz: pd.DataFrame,
    corr_df: pd.DataFrame,
    df_reg: pd.DataFrame,
    log: logging.Logger,
) -> Dict:
    """Apply brief's H065_01 decision rule.

    POSITIVE requires ALL three:
      1. Jaccard(Vasc_UP, HALLMARK_HYPOXIA) >= 0.15 with p_bonf < 1e-8
         AND Jaccard(Vasc_UP, Ortiz-Barahona) >= 0.10 with p_bonf < 1e-4.
      2. HLMA ssGSEA-HYPOXIA vs age rho >= 0.40 with p < 0.05 (N=22).
      3. GTEx beta_age significant (p<0.05) AND |beta_age| attenuates by >=30%.

    NEGATIVE requires ANY TWO of:
      1. Ortiz-Barahona Jaccard < 0.05.
      2. HLMA ssGSEA-vs-age |rho| < 0.25.
      3. GTEx age:SMTSISCH no attenuation (<10%).

    AMBIGUOUS-SHARED-AP1: Hypoxia Jaccard >=0.15 BUT Ortiz Jaccard <0.05.
    BROAD-STRESS: Jaccard > 0.10 for >=4 of 8 Hallmark sets (Vasc_UP).
    UNINTERPRETABLE-TECH-CONFOUND: stratified disagree from pooled in sign or >0.2 magnitude.
    """
    verdict = {}

    # Focal Vasc_UP rows
    vasc_up_hall = df_hall[
        (df_hall["compartment"] == "Vascular") & (df_hall["direction"] == "UP")
    ].set_index("gene_set")
    vasc_up_ortiz = df_ortiz[
        (df_ortiz["compartment"] == "Vascular") & (df_ortiz["direction"] == "UP")
    ].iloc[0].to_dict()

    jac_hypoxia = float(vasc_up_hall.loc["HALLMARK_HYPOXIA", "jaccard"])
    p_bonf_hypoxia = float(vasc_up_hall.loc["HALLMARK_HYPOXIA", "p_bonf_48"])
    jac_ortiz = float(vasc_up_ortiz["jaccard"])
    p_bonf_ortiz = float(vasc_up_ortiz["p_bonf_48"])

    verdict["t1a_vasc_up_hallmark_hypoxia"] = {
        "jaccard": jac_hypoxia,
        "overlap": int(vasc_up_hall.loc["HALLMARK_HYPOXIA", "overlap"]),
        "p_bonf_48": p_bonf_hypoxia,
        "hit_positive_criterion": (jac_hypoxia >= 0.15 and p_bonf_hypoxia < 1e-8),
    }
    verdict["t1a_vasc_up_ortiz_barahona"] = {
        "jaccard": jac_ortiz,
        "overlap": int(vasc_up_ortiz["overlap"]),
        "p_bonf_48": p_bonf_ortiz,
        "source": vasc_up_ortiz.get("source", ""),
        "hit_positive_criterion": (jac_ortiz >= 0.10 and p_bonf_ortiz < 1e-4),
        "hit_negative_criterion": (jac_ortiz < 0.05),
    }

    # Broad-stress: Vasc_UP Jaccard>0.10 across Hallmark sets
    n_broad = int((vasc_up_hall["jaccard"] > 0.10).sum())
    verdict["t1a_vasc_up_broad_stress"] = {
        "n_hallmark_sets_jaccard_gt_010": n_broad,
        "total_hallmark_sets": len(vasc_up_hall),
        "trigger_broad_stress": n_broad >= 4,
    }

    # Contamination audit (Vasc_UP row)
    contam_vasc_up = df_contam[
        (df_contam["compartment"] == "Vascular") & (df_contam["direction"] == "UP")
    ].iloc[0].to_dict()
    verdict["t1a_contamination_audit"] = {
        "raw_overlap_hypoxia": int(contam_vasc_up["raw_overlap"]),
        "contam_overlap_14ap1sasp": int(contam_vasc_up["contam_overlap"]),
        "contam_genes_hit": contam_vasc_up["contam_genes_hit"],
        "clean_overlap": int(contam_vasc_up["clean_overlap"]),
        "clean_jaccard": float(contam_vasc_up["clean_jaccard"]),
        "clean_p_bonf_48": float(contam_vasc_up["clean_p_bonf_48"]),
    }

    # T1b rows for HALLMARK_HYPOXIA
    hypoxia_corr = corr_df[corr_df["gene_set"] == "HALLMARK_HYPOXIA"].copy()
    by_stratum = {r["stratum"]: r for _, r in hypoxia_corr.iterrows()}
    pooled_rho = float(by_stratum.get("pooled", {}).get("rho", float("nan")))
    sc_rho = float(by_stratum.get("within_scRNA", {}).get("rho", float("nan")))
    sn_rho = float(by_stratum.get("within_snRNA", {}).get("rho", float("nan")))
    pooled_p = float(by_stratum.get("pooled", {}).get("p_spearman", float("nan")))
    pooled_n = int(by_stratum.get("pooled", {}).get("n", 0))

    # Tech-confound uninterpretable check
    tech_confound_flags = []
    for stratum_rho in (sc_rho, sn_rho):
        if np.isnan(stratum_rho) or np.isnan(pooled_rho):
            continue
        if np.sign(stratum_rho) != np.sign(pooled_rho) and abs(stratum_rho) > 0.1:
            tech_confound_flags.append(f"sign_flip(stratum={stratum_rho:+.3f}, pooled={pooled_rho:+.3f})")
        if abs(stratum_rho - pooled_rho) > 0.2:
            tech_confound_flags.append(f"magnitude_gap({abs(stratum_rho - pooled_rho):.2f})")
    verdict["t1b_hallmark_hypoxia"] = {
        "pooled_rho": pooled_rho,
        "pooled_p": pooled_p,
        "pooled_n": pooled_n,
        "within_scRNA_rho": sc_rho,
        "within_scRNA_n": int(by_stratum.get("within_scRNA", {}).get("n", 0)),
        "within_snRNA_rho": sn_rho,
        "within_snRNA_n": int(by_stratum.get("within_snRNA", {}).get("n", 0)),
        "tech_confound_flags": tech_confound_flags,
        "tech_confound_unstable": len(tech_confound_flags) > 0,
        "hit_positive_criterion": (not np.isnan(pooled_rho)
                                   and pooled_rho >= 0.40 and pooled_p < 0.05),
        "hit_intermediate_criterion": (not np.isnan(pooled_rho)
                                       and 0.25 <= abs(pooled_rho) < 0.40 and pooled_p < 0.1),
        "hit_negative_criterion": (not np.isnan(pooled_rho)
                                   and abs(pooled_rho) < 0.25),
    }

    # T1c attenuation
    atten = df_reg.attrs.get("attenuation", float("nan"))
    p_age_m1 = df_reg.attrs.get("p_age_m1", float("nan"))
    p_age_m2 = df_reg.attrs.get("p_age_m2", float("nan"))
    p_smtsisch = df_reg.attrs.get("p_smtsisch", float("nan"))
    p_interaction = df_reg.attrs.get("p_interaction", float("nan"))
    verdict["t1c_gtex"] = {
        "beta_age_m1": df_reg.attrs.get("beta_age_m1", float("nan")),
        "beta_age_m2": df_reg.attrs.get("beta_age_m2", float("nan")),
        "p_age_m1": p_age_m1,
        "p_age_m2": p_age_m2,
        "p_smtsisch": p_smtsisch,
        "p_age_smtsisch_interaction": p_interaction,
        "attenuation_pct": float(atten) if not np.isnan(atten) else None,
        "hit_positive_criterion": (
            not np.isnan(p_age_m1) and p_age_m1 < 0.05
            and not np.isnan(atten) and atten >= 0.30
        ),
        "hit_negative_criterion": (not np.isnan(atten) and atten < 0.10),
        "sensitivity_20pct": (not np.isnan(atten) and atten >= 0.20),
        "sensitivity_50pct": (not np.isnan(atten) and atten >= 0.50),
        "n": df_reg.attrs.get("n", 0),
    }

    # Apply decision rule
    pos_crit_1 = (verdict["t1a_vasc_up_hallmark_hypoxia"]["hit_positive_criterion"]
                  and verdict["t1a_vasc_up_ortiz_barahona"]["hit_positive_criterion"])
    pos_crit_2 = verdict["t1b_hallmark_hypoxia"]["hit_positive_criterion"]
    pos_crit_3 = verdict["t1c_gtex"]["hit_positive_criterion"]

    neg_crit_1 = verdict["t1a_vasc_up_ortiz_barahona"]["hit_negative_criterion"]
    neg_crit_2 = verdict["t1b_hallmark_hypoxia"]["hit_negative_criterion"]
    neg_crit_3 = verdict["t1c_gtex"]["hit_negative_criterion"]

    verdict["criteria_summary"] = {
        "positive_1_dual_jaccard": pos_crit_1,
        "positive_2_hlma_rho": pos_crit_2,
        "positive_3_gtex_attenuation": pos_crit_3,
        "negative_1_ortiz_low": neg_crit_1,
        "negative_2_hlma_rho_low": neg_crit_2,
        "negative_3_gtex_no_attenuation": neg_crit_3,
    }

    if verdict["t1a_vasc_up_broad_stress"]["trigger_broad_stress"]:
        pre_verdict = "BROAD-STRESS"
    elif verdict["t1b_hallmark_hypoxia"]["tech_confound_unstable"]:
        pre_verdict = "UNINTERPRETABLE-TECH-CONFOUND"
    elif pos_crit_1 and pos_crit_2 and pos_crit_3:
        pre_verdict = "POSITIVE"
    elif sum([neg_crit_1, neg_crit_2, neg_crit_3]) >= 2:
        pre_verdict = "REFUTED"
    elif (verdict["t1a_vasc_up_hallmark_hypoxia"]["hit_positive_criterion"]
          and verdict["t1a_vasc_up_ortiz_barahona"]["hit_negative_criterion"]):
        pre_verdict = "AMBIGUOUS-SHARED-AP1"
    else:
        pre_verdict = "INCONCLUSIVE"

    verdict["pre_registered_verdict"] = pre_verdict
    log.info("PRE-REGISTERED VERDICT: %s", pre_verdict)
    return verdict


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    log = setup_logging()
    t0 = time.time()
    log.info("=" * 70)
    log.info("BATCH_065 T1: Hypoxia falsification gate (brief v2, post-3-critic)")
    log.info("=" * 70)

    env = env_stamp(log)

    try:
        # --- T1a ---
        log.info("\n--- T1a: Gene-set overlap ---")
        hallmark = load_hallmark_sets(log)
        hif_set, hif_source = load_hif_target_set(log)
        signatures = load_signatures(log)
        df_hall, df_contam, df_ortiz = run_t1a(signatures, hallmark, hif_set, hif_source, log)

        # --- T1b ---
        log.info("\n--- T1b: HLMA ssGSEA-vs-age, tech-stratified ---")
        corr_df, _ = run_t1b(hallmark, hif_set, hif_source, log)

        # --- T1c ---
        log.info("\n--- T1c: GTEx SMTSISCH regression ---")
        df_reg = run_t1c(signatures, log)

        # --- Decision ---
        log.info("\n--- Decision ---")
        verdict = decide(df_hall, df_contam, df_ortiz, corr_df, df_reg, log)

        summary = {
            "env": env,
            "t1a_hallmark_grid": {
                "csv": str(OUT_T1A_HALLMARK),
                "n_rows": len(df_hall),
            },
            "t1a_ortiz_barahona": {
                "csv": str(OUT_T1A_ORTIZ),
                "source": hif_source,
                "set_size": len(hif_set),
            },
            "t1a_contamination_audit": {
                "csv": str(OUT_T1A_CONTAM),
            },
            "t1b_correlations": {
                "csv": str(OUT_T1B_CORR),
                "scores_csv": str(OUT_T1B_SCORES),
            },
            "t1c_regressions": {
                "csv": str(OUT_T1C_REG),
            },
            "verdict": verdict,
            "pre_registered_verdict": verdict["pre_registered_verdict"],
            "elapsed_sec": round(time.time() - t0, 2),
        }
        with open(OUT_SUMMARY, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Wrote %s", OUT_SUMMARY)
        log.info("Elapsed: %.1fs", time.time() - t0)
        return 0
    except Exception as e:
        log.exception("FATAL: %s", e)
        err_summary = {
            "env": env,
            "error": str(e),
            "pre_registered_verdict": "ERROR",
            "elapsed_sec": round(time.time() - t0, 2),
        }
        with open(OUT_SUMMARY, "w") as f:
            json.dump(err_summary, f, indent=2, default=str)
        return 1


if __name__ == "__main__":
    sys.exit(main())
