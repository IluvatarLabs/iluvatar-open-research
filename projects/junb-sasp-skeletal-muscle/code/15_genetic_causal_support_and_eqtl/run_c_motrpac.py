"""Batch 064 Analysis C — MoTrPAC plasticity / reversal (PI #15).

Tests whether HLMA aging-UP/DOWN signatures (Vascular, FAP, MuSC) are reversed in
MoTrPAC human acute-exercise transcriptome.

    C1  24-hr post-exercise reversal test (age-stratified PRIMARY, pooled SENSITIVITY)
    C2  cross-sectional Endurance-baseline vs Control-baseline comparison
        (NOT a chronic-training test; lifestyle correlate only — per brief v2, post-Critic 3)

Design source: `experiments/batch_064/brief.md` v2, Analysis C section (lines 113-161).
Preflight: `experiments/batch_064/preflight.md`.

DATA-LIMITATION PRE-COMMIT (Critic 1/2/3 C-triad, as negotiated in brief v2):
  The MoTrPAC local artefacts `MUSCLE_TRANSCRIPT_RNA_SEQ_SUM_STATS.rda` and
  `MUSCLE_TRNSCRPT_DA.rda` are PER-CONTRAST AGGREGATES — they contain neither raw
  per-participant counts nor participant covariates (age, BMI, sex, site).
  Consequences:
    * IEG pre-check cannot be age-stratified locally; we run at POOLED level and flag
      `ieg_precheck_pooled` verdict. Brief v2 pre-registered this path when covariates
      are not locally available (preflight line 33, brief line 119 stratified-ideal).
    * Between-subject arm-label permutation null (brief line 143) would require raw counts.
      We substitute a SIGNATURE-RANDOMIZATION null (1000 size-matched random gene sets from
      MoTrPAC-detected genes, NES recomputed against the same EE-vs-CON ranking) IN ADDITION
      to the default gseapy gene-label permutation. Both nulls are reported. Documented as
      limitation; full sample-permutation deferred to a future iter with DCC raw-data access.
    * C2 propensity-score balance check cannot be performed (no covariates). C2 is therefore
      classified UNINTERPRETABLE-FOR-CAUSAL under brief v2 decision rules and is run as
      DESCRIPTIVE-ONLY. A `BALANCE_UNKNOWN=True` flag is stamped on every C2 output row.

WHY the substitute nulls are defensible:
    Signature-randomization null tests the specific signature against the OBSERVED ranking,
    controlling for ranking shape but not between-subject variability. This is weaker than
    an arm-label permutation null but strictly stronger than no null. We report both so the
    reader can see whether gene-label (gseapy) and signature-randomization disagree on sign.
    Per brief UNINTERPRETABLE clause: if the two nulls disagree on sign, cell is INCONCLUSIVE.

WHY gseapy.prerank: Python-native GSEA implementation with an established API; the brief
    (line 140) originally specified fgsea in R, and the preflight (line 29) pivoted to
    gseapy 1.1.13 after confirming R is not installed. gseapy.prerank implements the same
    Subramanian 2005 weighted-KS statistic. No wheel to reinvent (Rule 1).

Silmaril context: F047 FOS/AP-1 ESTABLISHED (d=1.357 FAP); finding_cell_type_specificity
    (Vascular/FAP/MuSC-specific JUNB-SASP coupling); iter 054/055 AUCell signatures.

Outputs (all under experiments/batch_064/):
  c_signatures.csv       compartment x direction x rank x gene_symbol x ensembl x logFC x t
  c_ieg_precheck.csv     gene x timepoint x logFC_EE_vs_CON x pass_threshold x verdict
  c_gsea_results.csv     compartment x direction x timepoint x NES x p_gene_perm x
                         p_sig_rand x leading_edge_n x leading_edge_genes
  c_c2_baseline.csv      compartment x direction x NES x p_gene_perm x p_sig_rand x
                         BALANCE_UNKNOWN x leading_edge_n
  c_correlation.csv      TF x compartment x HLMA_age_logFC x HLMA_t x MoTrPAC_EE_logFC_24hr +
                         compartment-level Pearson r (9 TFs per compartment)
  c_summary.json         per-brief decision rules + data-limitation disclosures
  logs/c_stdout.log      full stdout log
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# -----------------------------------------------------------------------------
# CONFIG (all paths absolute; brief/preflight pins the values)
# -----------------------------------------------------------------------------
BATCH_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_064")
LOG_FP = BATCH_DIR / "logs" / "c_stdout.log"

MOTRPAC_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/data/MoTrPAC")
RDA_SUMSTATS = MOTRPAC_DIR / "MUSCLE_TRANSCRIPT_RNA_SEQ_SUM_STATS.rda"
RDA_DA = MOTRPAC_DIR / "MUSCLE_TRNSCRPT_DA.rda"

BATCH_052 = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_052")
HLMA_DE_FILES = {
    "Vascular": BATCH_052 / "b1_de_vascular.csv",
    "FAP": BATCH_052 / "b1_de_fap.csv",
    "MuSC": BATCH_052 / "b1_de_musc.csv",
}

# GTEx gct provides local ENSG->symbol mapping (standard GCT header: Name=ENSG,
# Description=symbol) — avoids any network dependency on mygene/MyGene.info.
GTEX_GCT = Path(
    "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/GTEx/muscle/gene_tpm_muscle_skeletal.gct.gz"
)

# Output filepaths
OUT_SIGNATURES = BATCH_DIR / "c_signatures.csv"
OUT_IEG = BATCH_DIR / "c_ieg_precheck.csv"
OUT_GSEA = BATCH_DIR / "c_gsea_results.csv"
OUT_C2 = BATCH_DIR / "c_c2_baseline.csv"
OUT_CORR = BATCH_DIR / "c_correlation.csv"
OUT_SUMMARY = BATCH_DIR / "c_summary.json"

# -----------------------------------------------------------------------------
# Pre-registered constants
# -----------------------------------------------------------------------------
# WHY top-100 per direction by |t-stat| (wald_stat in HLMA DE): preflight line 39
# (Subramanian 2005 canonical GSEA set size). Bounded to p<0.05 per brief Step 3.
SIG_SIZE = 100

# WHY ±0.3 logFC IEG tolerance: preflight line 42 (HLMA JUNB d=0.901 ~ 0.7 logFC, half
# magnitude is a defensible "near-baseline" criterion).
IEG_LOGFC_TOLERANCE = 0.3

# WHY 1000 permutations: matches brief line 143 (gene-label permutation 1000 perms;
# signature-randomization null 1000 random size-matched gene sets).
N_PERMUTATIONS = 1000
N_SIG_RAND_PERM = 1000

# WHY |NES|>2.0 Bonferroni-corrected: Critic 2 C2 resolution (brief line 130, 148).
NES_THRESHOLD = 2.0
P_THRESHOLD = 0.05

# WHY seed=1: convention from `marvin.yaml` rigor.seeds and batch_063 scripts.
SEED = 1

# 9-TF panel (brief line 44-ish canonical, also batch_052 / batch_063 usage).
TF_PANEL = ["JUNB", "FOS", "EGR1", "EGR2", "ATF3", "CEBPB", "KLF10", "IRF1", "CDKN1A"]

# IEG subset for C1 24-hr pre-check (brief line 119).
IEG_SUBSET = ["JUNB", "FOS", "EGR1"]

# Timepoints in MoTrPAC DA table (verified via schema probe).
TIMEPOINTS_POST = ["post_15_30_45_min", "post_3.5_4_hr", "post_24_hr"]
TIMEPOINT_24HR = "post_24_hr"
TIMEPOINT_BASELINE = "pre_exercise"

# Contrast labels (verified via schema probe):
#   EE-CON post-timepoints use the delta-delta contrast_short:
#     "Endur.<tp> - Control.<tp> (delta-delta)"
#   EE-CON baseline uses: "Endur.pre_exercise - Control.pre_exercise"
def _ee_con_contrast_label(timepoint: str) -> str:
    """Return contrast_short for EE-CON at a given timepoint.

    WHY this helper: DA has TWO different contrast_short naming schemes for EE-CON —
    delta-delta (post timepoints) vs pre-exercise (baseline). Hardcoding avoids bugs.
    """
    if timepoint == "pre_exercise":
        return "Endur.pre_exercise - Control.pre_exercise"
    return f"Endur.{timepoint} - Control.{timepoint} (delta-delta)"


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure root logger to write both to stdout and logs/c_stdout.log.

    WHY: matches batch_063 convention and preflight's Monitor kill-conditions
    (silence >45 min triggers investigation).
    """
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
    return logging.getLogger("c_motrpac")


# -----------------------------------------------------------------------------
# ENSG <-> symbol mapping (from GTEx GCT; no network)
# -----------------------------------------------------------------------------
def load_ensg_symbol_map(gct_fp: Path, log: logging.Logger) -> pd.DataFrame:
    """Read the (Name, Description) columns of a GCT.gz file to produce a
    mapping dataframe with columns: ensg_versioned, ensg, symbol.

    WHY: GTEx GCT file is a canonical GENCODE source for ENSG<->symbol; using it
    avoids a network call to mygene (and therefore avoids a reproducibility-in-CI
    concern on air-gapped compute). The MoTrPAC DA feature_id is ENSG-with-version
    (e.g., ENSG00000171223.7).

    BUG FIX (2026-04-22): this GCT is #1.3 format (not #1.2) and carries FOUR
    leading non-sample columns on row 3: [id, Name, Description, <sample_ids>...].
    The previous implementation hardcoded column 0 / column 1 as (Name, Description)
    which — for a #1.3 file — silently assigned the integer row-index column as the
    "ENSG" and the ENSG column as the "symbol". Result: 0/600 signature symbols
    mapped because the map's "symbol" side actually held versioned ENSG strings.
    Fix: parse the header line and locate the "Name" and "Description" columns by
    name rather than by position. This is robust to both #1.2 and #1.3 GCTs.
    """
    log.info("Loading ENSG-symbol map from %s", gct_fp)
    with gzip.open(gct_fp, "rt") as fh:
        version = fh.readline().rstrip("\n")
        _ = fh.readline()  # "<nrows>\t<ncols>" — ignored
        header = fh.readline().rstrip("\n").split("\t")
        try:
            name_idx = header.index("Name")
            desc_idx = header.index("Description")
        except ValueError as e:
            raise ValueError(
                f"GCT header missing Name/Description columns: {header[:6]}"
            ) from e
        needed = max(name_idx, desc_idx) + 1
        rows: List[Tuple[str, str]] = []
        for line in fh:
            parts = line.split("\t", needed)
            if len(parts) > max(name_idx, desc_idx):
                rows.append((parts[name_idx], parts[desc_idx]))
    df = pd.DataFrame(rows, columns=["ensg_versioned", "symbol"])
    df["ensg"] = df["ensg_versioned"].str.split(".").str[0]
    df = df.drop_duplicates(subset=["ensg_versioned"])
    log.info(
        "  loaded %d ENSG->symbol rows (GCT version=%s; Name=col%d Description=col%d; header cols=%d)",
        len(df), version, name_idx, desc_idx, len(header),
    )
    return df


def symbol_to_ensg_versioned(
    symbols: List[str],
    mapping: pd.DataFrame,
    motrpac_feature_ids: set,
    log: logging.Logger,
) -> Dict[str, Optional[str]]:
    """Map a list of gene symbols to ENSG-versioned IDs present in MoTrPAC.

    Priority: prefer a candidate whose ENSG-versioned matches a MoTrPAC feature_id
    exactly; else prefer a candidate whose UNVERSIONED ENSG matches any unversioned
    MoTrPAC feature_id (handles GENCODE-release version skew between GTEx and
    MoTrPAC, e.g. GTEx v8 -> ENSG00000006210.7 vs MoTrPAC -> ENSG00000006210.15);
    else fall back to the GTEx-versioned ENSG so the symbol is still represented.
    Returns dict symbol -> ensg_versioned (or None if unmappable).

    WHY unversioned fallback (bug fix, 2026-04-22): even with the GCT header bug
    fixed, we cannot assume GTEx and MoTrPAC pin the same GENCODE release. Matching
    version-suffix-exact leaves any version-skewed gene unmapped, which silently
    produced empty gene sets in downstream GSEA. The unversioned-base match recovers
    those; we prefer the version present in MoTrPAC so the ranking vector can be
    queried directly without extra lookup.
    """
    motrpac_unversioned_to_versioned: Dict[str, str] = {}
    for fid in motrpac_feature_ids:
        base = fid.split(".")[0]
        # Keep the first-seen version; MoTrPAC feature_ids are already unique per base.
        motrpac_unversioned_to_versioned.setdefault(base, fid)

    out: Dict[str, Optional[str]] = {}
    grouped = mapping.groupby("symbol")
    unmapped: List[str] = []
    for sym in symbols:
        if sym not in grouped.groups:
            out[sym] = None
            unmapped.append(sym)
            continue
        candidates = grouped.get_group(sym)[["ensg_versioned", "ensg"]].values.tolist()
        picked: Optional[str] = None
        # Priority 1: exact versioned match to MoTrPAC
        for ensg_v, _ in candidates:
            if ensg_v in motrpac_feature_ids:
                picked = ensg_v
                break
        # Priority 2: unversioned match -> substitute MoTrPAC's version
        if picked is None:
            for _, ensg_base in candidates:
                if ensg_base in motrpac_unversioned_to_versioned:
                    picked = motrpac_unversioned_to_versioned[ensg_base]
                    break
        # Priority 3: fall back to GTEx-versioned ENSG (may not be in MoTrPAC)
        if picked is None:
            picked = candidates[0][0]
        out[sym] = picked
    if unmapped:
        log.warning(
            "  %d symbols unmapped (GTEx GCT has no match): %s",
            len(unmapped),
            ", ".join(unmapped[:12]) + ("..." if len(unmapped) > 12 else ""),
        )
    return out


# -----------------------------------------------------------------------------
# MoTrPAC loaders
# -----------------------------------------------------------------------------
def load_motrpac_da(log: logging.Logger) -> pd.DataFrame:
    """Load MUSCLE_TRNSCRPT_DA.rda with pyreadr and return the inner DataFrame.

    WHY pyreadr: preflight line 27 verified availability; pure-Python reader,
    no R dependency. Rule 1 — reuse the library, do not reinvent RDS parsing.

    Schema (verified via probe 2026-04-22): 20 cols, 346689 rows. Key cols:
      contrast_category in {EE-CON, RE-CON, EE-EE, RE-RE, EE-RE, CON-CON}
      contrast_short   e.g. "Endur.post_24_hr - Control.post_24_hr (delta-delta)"
      Timepoint        {post_15_30_45_min, post_3.5_4_hr, post_24_hr, pre_exercise}
      feature_id       ENSG with version suffix
      logFC, CI.L, CI.R, t, p_value, adj_p_value
    """
    import pyreadr  # local import — heavy dep, isolate

    log.info("Reading %s via pyreadr", RDA_DA)
    t0 = time.time()
    result = pyreadr.read_r(str(RDA_DA))
    # pyreadr returns OrderedDict[name -> DataFrame]; .rda name matches R variable name
    key = next(iter(result.keys()))
    da = result[key]
    log.info(
        "  loaded DA: key=%s  shape=%s  cols=%s  elapsed=%.1fs",
        key, da.shape, list(da.columns), time.time() - t0,
    )
    # Coerce Categorical to object for safer string compares downstream
    for c in ["contrast_short", "contrast_category", "Timepoint", "randomGroupCode",
              "contrast_type", "feature_id"]:
        if c in da.columns:
            da[c] = da[c].astype(str)
    return da


def slice_ee_con(da: pd.DataFrame, timepoint: str, log: logging.Logger) -> pd.DataFrame:
    """Slice the EE-vs-CON contrast for a single timepoint.

    WHY: the DA table carries multiple contrasts interleaved; slicing must filter
    BOTH contrast_category AND contrast_short for unambiguous match.
    """
    label = _ee_con_contrast_label(timepoint)
    sub = da[(da["contrast_category"] == "EE-CON") & (da["contrast_short"] == label)].copy()
    log.info("  EE-CON slice timepoint=%s  label=%s  rows=%d", timepoint, label, len(sub))
    return sub


# -----------------------------------------------------------------------------
# HLMA signature extraction
# -----------------------------------------------------------------------------
def load_hlma_signatures(
    hlma_de_paths: Dict[str, Path],
    sig_size: int,
    log: logging.Logger,
) -> pd.DataFrame:
    """Read batch_052 DE CSVs and return top-N UP/DOWN genes per compartment.

    Ranking: |wald_stat| (DESeq2-style Wald t-equivalent), filtered to p_value<0.05
    and |log2FC|>0. Signed log2FC determines direction.

    WHY wald_stat over log2FC alone: preflight line 39 (avoid bias from large-fold-
    small-t signal; Subramanian 2005 recommends a t-like statistic).

    Returns long-form DataFrame: compartment, direction, rank, gene_symbol, log2FC,
    wald_stat, p_value.
    """
    rows: List[dict] = []
    for compartment, fp in hlma_de_paths.items():
        log.info("  Reading HLMA DE %s: %s", compartment, fp)
        df = pd.read_csv(fp)
        # Confirmed columns (batch_052): gene, baseMean, log2FC, lfcSE, wald_stat,
        # p_value, bh_q, bh_q_manual, log2FC_lm, p_value_lm, bh_q_lm
        required = {"gene", "log2FC", "wald_stat", "p_value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"HLMA DE {fp} missing columns: {missing}")
        df = df.dropna(subset=["gene", "log2FC", "wald_stat", "p_value"]).copy()
        df = df[df["p_value"] < P_THRESHOLD]
        up = df[df["log2FC"] > 0].copy()
        down = df[df["log2FC"] < 0].copy()
        up = up.reindex(up["wald_stat"].abs().sort_values(ascending=False).index).head(sig_size)
        down = down.reindex(down["wald_stat"].abs().sort_values(ascending=False).index).head(sig_size)
        for rank_i, (_, r) in enumerate(up.iterrows(), start=1):
            rows.append({
                "compartment": compartment, "direction": "UP", "rank": rank_i,
                "gene_symbol": r["gene"], "log2FC": r["log2FC"],
                "wald_stat": r["wald_stat"], "p_value": r["p_value"],
            })
        for rank_i, (_, r) in enumerate(down.iterrows(), start=1):
            rows.append({
                "compartment": compartment, "direction": "DOWN", "rank": rank_i,
                "gene_symbol": r["gene"], "log2FC": r["log2FC"],
                "wald_stat": r["wald_stat"], "p_value": r["p_value"],
            })
        log.info(
            "    %s: selected UP=%d DOWN=%d (post p<%.2g, |log2FC|>0 filter)",
            compartment, len(up), len(down), P_THRESHOLD,
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# GSEA with gseapy.prerank
# -----------------------------------------------------------------------------
def run_prerank_gsea(
    ranked: pd.Series,
    gene_sets: Dict[str, List[str]],
    seed: int,
    n_perm: int,
    log: logging.Logger,
) -> pd.DataFrame:
    """Run gseapy.prerank on a pre-ranked gene series.

    Parameters
    ----------
    ranked
        pd.Series indexed by gene_identifier (ENSG-versioned, matching gene_sets),
        values = ranking metric (t-statistic). Must have no duplicates.
    gene_sets
        dict[name] -> list[gene_id]

    WHY: gseapy.prerank is the Python re-implementation of weighted-KS GSEA
    (Subramanian 2005). Rule 1 — do not reinvent.
    """
    import gseapy

    # gseapy.prerank requires a 2-col DataFrame (gene, rank) with no NaN/duplicates.
    rdf = pd.DataFrame({"gene": ranked.index, "rank": ranked.values})
    rdf = rdf.dropna().drop_duplicates(subset=["gene"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr = gseapy.prerank(
            rnk=rdf,
            gene_sets=gene_sets,
            permutation_num=n_perm,
            outdir=None,  # do not dump files to disk
            seed=seed,
            min_size=5,
            max_size=max(1000, max(len(v) for v in gene_sets.values()) + 1),
            verbose=False,
            no_plot=True,
        )
    return pr.res2d.copy()


def _walking_ks_es(
    scores_ordered: np.ndarray,
    abs_scores_ordered: np.ndarray,
    hits_ordered: np.ndarray,
    n_total: int,
) -> float:
    """Compute a single weighted-KS enrichment score (Subramanian 2005, p=1).

    Parameters
    ----------
    scores_ordered, abs_scores_ordered
        The ranking metric and its absolute value, pre-sorted in descending order.
    hits_ordered
        Boolean vector over the same descending order marking signature-gene positions.
    n_total
        Length of the full ranked universe.

    WHY factored out: fix for null-scale mismatch (review_run_c.md item 1). The
    observed ES used to form the empirical p-value MUST come from the same walking
    routine that produces the null draws; otherwise we compare gseapy's NES (which
    is normalized against gseapy's own gene-label-permutation null) to our locally
    NES-normalized null, and the two live on different scales. Centralizing the
    routine makes that scale mismatch impossible.
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


def signature_randomization_null(
    ranked: pd.Series,
    signature_size: int,
    signature_genes: List[str],
    n_perm: int,
    seed: int,
    rng_universe: List[str],
) -> Tuple[float, float, float]:
    """Empirical p-value for enrichment using a size-matched signature-randomization null.

    Returns
    -------
    (p_emp, observed_es, observed_nes_local)
        p_emp: two-sided empirical p-value with +1 pseudocount.
        observed_es: raw ES for the TRUE signature against this ranking (walking-KS,
            same routine used for null draws).
        observed_nes_local: observed_es normalized by the sign-conditional mean of
            the null ES (reported for reference; the p-value uses raw ES, not this).

    CRITICAL FIX (review_run_c.md item 1): the previous implementation accepted the
    `observed_nes` returned by gseapy.prerank (which normalizes by a gene-label-
    permutation null) and compared it against a null that was normalized by
    sign-conditional means of its OWN draws. Those two NES scales are not
    commensurable, which meant the empirical p-value was computed against the wrong
    reference. The fix is to compute the observed ES for the true signature using
    the SAME walking-KS routine used for the null draws, then compare raw observed
    ES to raw null ES (both unnormalized) for the empirical p. This guarantees
    observed and null live on the same scale.

    WHY this substitute null at all: the brief (line 143) pre-registered a
    between-subject arm-label permutation null, which requires raw counts. The
    .rda DA aggregate does NOT provide these (verified). This null controls for
    signature SIZE and the ranking DISTRIBUTION but NOT for between-subject
    variability. Reported alongside gseapy's default gene-label permutation
    p-value so the reader can see whether the two disagree (brief UNINTERPRETABLE
    clause at line 159).

    Implementation: sample `signature_size` genes uniformly without replacement
    from rng_universe; compute weighted-KS enrichment (Subramanian 2005, p=1)
    against the ranked vector; repeat `n_perm` times. The observed ES is computed
    identically using the TRUE signature genes.
    """
    rng = np.random.default_rng(seed)
    genes = ranked.index.to_numpy()
    scores = ranked.values.astype(float)
    n_total = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    universe_idx = np.array(
        [gene_to_idx[g] for g in rng_universe if g in gene_to_idx], dtype=int
    )
    if len(universe_idx) < signature_size + 10:
        return float("nan"), float("nan"), float("nan")

    # Pre-sort once: everything walks the same descending order.
    order = np.argsort(-scores)
    scores_ordered = scores[order]
    abs_scores_ordered = np.abs(scores_ordered)

    # --- Observed ES using the SAME routine that generates the null draws ---
    sig_idx = np.array(
        [gene_to_idx[g] for g in signature_genes if g in gene_to_idx], dtype=int
    )
    if len(sig_idx) == 0:
        return float("nan"), float("nan"), float("nan")
    sig_mask = np.zeros(n_total, dtype=bool)
    sig_mask[sig_idx] = True
    hits_obs_ordered = sig_mask[order]
    observed_es = _walking_ks_es(scores_ordered, abs_scores_ordered, hits_obs_ordered, n_total)

    # --- Null ES draws ---
    null_es = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        pick = rng.choice(universe_idx, size=signature_size, replace=False)
        pick_set = np.zeros(n_total, dtype=bool)
        pick_set[pick] = True
        hits_ordered = pick_set[order]
        null_es[i] = _walking_ks_es(scores_ordered, abs_scores_ordered, hits_ordered, n_total)

    # --- Empirical p on RAW ES (same scale on both sides) ---
    hits = int((np.abs(null_es) >= abs(observed_es)).sum())
    p_emp = (hits + 1) / (n_perm + 1)

    # Local NES (sign-conditional mean normalization) — reported for reference only.
    pos = null_es[null_es > 0]
    neg = null_es[null_es < 0]
    pos_mean = pos.mean() if len(pos) else 1.0
    neg_mean = abs(neg.mean()) if len(neg) else 1.0
    if observed_es > 0:
        observed_nes_local = observed_es / pos_mean if pos_mean else float("nan")
    elif observed_es < 0:
        observed_nes_local = observed_es / neg_mean if neg_mean else float("nan")
    else:
        observed_nes_local = 0.0

    return float(p_emp), float(observed_es), float(observed_nes_local)


# -----------------------------------------------------------------------------
# IEG pre-check (C1 gate)
# -----------------------------------------------------------------------------
def ieg_precheck(
    da: pd.DataFrame,
    sym_to_ensg: Dict[str, Optional[str]],
    log: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """For JUNB/FOS/EGR1, extract POOLED EE-vs-CON logFC at 15-45min (positive control)
    and 24-hr (primary gate).

    Returns (rows_df, verdict_dict):
        verdict_dict['ieg_precheck_pooled'] = 'PASS' | 'FAIL' | 'INCOMPLETE'
            PASS: all 3 IEGs |logFC_24hr| < IEG_LOGFC_TOLERANCE
            FAIL: any IEG |logFC_24hr| >= IEG_LOGFC_TOLERANCE
            INCOMPLETE: at least one IEG ENSG unmapped or missing at 24-hr row

    WHY pooled-only: MoTrPAC .rda DA lacks per-participant age covariates
    (verified: no such column). Brief v2 line 119 pre-registered that age-stratified
    re-fit requires DCC raw-counts access, which we do not have. Pooled IEG pre-check
    is the agreed fallback (brief Analysis-C Step 2 in this task spec).
    """
    rows: List[dict] = []
    verdict = "PASS"
    for gene_sym in IEG_SUBSET:
        ensg = sym_to_ensg.get(gene_sym)
        for tp in ["post_15_30_45_min", TIMEPOINT_24HR]:
            if ensg is None:
                rows.append({
                    "gene_symbol": gene_sym, "ensg_versioned": None, "timepoint": tp,
                    "logFC_EE_vs_CON": float("nan"), "p_value": float("nan"),
                    "adj_p_value": float("nan"), "status": "ENSG_UNMAPPED",
                })
                if tp == TIMEPOINT_24HR:
                    verdict = "INCOMPLETE"
                continue
            sub = slice_ee_con(da, tp, log)
            hit = sub[sub["feature_id"] == ensg]
            if len(hit) == 0:
                rows.append({
                    "gene_symbol": gene_sym, "ensg_versioned": ensg, "timepoint": tp,
                    "logFC_EE_vs_CON": float("nan"), "p_value": float("nan"),
                    "adj_p_value": float("nan"), "status": "MISSING_IN_MOTRPAC",
                })
                if tp == TIMEPOINT_24HR:
                    verdict = "INCOMPLETE"
                continue
            r = hit.iloc[0]
            logfc = float(r["logFC"])
            rows.append({
                "gene_symbol": gene_sym, "ensg_versioned": ensg, "timepoint": tp,
                "logFC_EE_vs_CON": logfc,
                "p_value": float(r["p_value"]),
                "adj_p_value": float(r["adj_p_value"]),
                "status": "OK",
            })
            # Only the 24-hr row drives the gate
            if tp == TIMEPOINT_24HR and verdict == "PASS":
                if abs(logfc) >= IEG_LOGFC_TOLERANCE:
                    verdict = "FAIL"
    return pd.DataFrame(rows), {"ieg_precheck_pooled": verdict}


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def main() -> int:  # noqa: C901 — orchestration is linear; better read top-to-bottom
    """Run the full C pipeline.

    WHY single main(): auditability — reviewers can follow the order of operations
    (load MoTrPAC -> build mappings -> IEG precheck -> signatures -> GSEA C1 -> C2
    descriptive -> correlation secondary -> summary).
    """
    random.seed(SEED)
    np.random.seed(SEED)

    log = setup_logging()
    log.info("=== batch_064 run_c_motrpac.py START ===")
    log.info("python %s  numpy %s  pandas %s  scipy %s",
             sys.version.split()[0], np.__version__, pd.__version__,
             __import__("scipy").__version__)
    try:
        import gseapy
        import pyreadr
        log.info("gseapy %s  pyreadr %s", gseapy.__version__, pyreadr.__version__)
    except ImportError as e:  # pragma: no cover
        log.error("Missing required dependency: %s", e)
        return 2

    # ------------------------------------------------------------------
    # 1. Load MoTrPAC DA
    # ------------------------------------------------------------------
    log.info("[1] Loading MoTrPAC DA table")
    da = load_motrpac_da(log)
    log.info("  unique contrast_category: %s", sorted(da["contrast_category"].unique()))
    log.info("  unique Timepoint: %s", sorted(da["Timepoint"].unique()))
    motrpac_feature_ids = set(da["feature_id"].unique())
    log.info("  unique feature_ids (ENSG-versioned): %d", len(motrpac_feature_ids))

    # ------------------------------------------------------------------
    # 2. ENSG <-> symbol mapping from GTEx GCT (no network)
    # ------------------------------------------------------------------
    log.info("[2] Building ENSG<->symbol map")
    ensg_map = load_ensg_symbol_map(GTEX_GCT, log)

    # ------------------------------------------------------------------
    # 3. HLMA signatures (top UP + top DOWN per compartment)
    # ------------------------------------------------------------------
    log.info("[3] Building HLMA aging signatures (top-%d |wald_stat|, p<%.2g)",
             SIG_SIZE, P_THRESHOLD)
    sig_df = load_hlma_signatures(HLMA_DE_FILES, SIG_SIZE, log)

    # Map all signature gene symbols to ENSG-versioned
    all_syms = sorted(set(sig_df["gene_symbol"].unique()) | set(TF_PANEL))
    sym2ensg = symbol_to_ensg_versioned(all_syms, ensg_map, motrpac_feature_ids, log)
    sig_df["ensg_versioned"] = sig_df["gene_symbol"].map(sym2ensg)
    n_unmapped_sig = int(sig_df["ensg_versioned"].isna().sum())
    log.info(
        "  signature genes mapped: %d/%d (unmapped=%d)",
        len(sig_df) - n_unmapped_sig, len(sig_df), n_unmapped_sig,
    )
    sig_df.to_csv(OUT_SIGNATURES, index=False)
    log.info("  wrote %s", OUT_SIGNATURES)

    # ------------------------------------------------------------------
    # 4. IEG pre-check (C1 24-hr gate + 15-45min positive control)
    # ------------------------------------------------------------------
    log.info("[4] IEG pre-check (pooled; age stratification deferred — brief v2 line 119)")
    ieg_df, ieg_verdict = ieg_precheck(da, sym2ensg, log)
    ieg_df.to_csv(OUT_IEG, index=False)
    log.info("  ieg_precheck_pooled=%s", ieg_verdict["ieg_precheck_pooled"])
    log.info("  wrote %s", OUT_IEG)

    # ------------------------------------------------------------------
    # 5. GSEA C1 across 3 compartments x 3 timepoints
    # ------------------------------------------------------------------
    log.info("[5] GSEA C1: 3 compartments x 3 timepoints")

    # Build gene sets in ENSG-versioned space per compartment+direction.
    # WHY dedupe (review_run_c.md non-blocking #2): gseapy treats duplicate entries
    # as separate hits, which would inflate both observed ES and null ES for any
    # signature where the same ENSG appears twice. Symbol->ENSG mapping is
    # many-to-one, but collisions at the signature level can still occur if
    # upstream produced duplicates. dict.fromkeys preserves the first-seen order
    # (which corresponds to |wald_stat| ranking) so leading-edge interpretability
    # is preserved.
    gene_sets_full: Dict[str, List[str]] = {}
    for (comp, direction), sub in sig_df.groupby(["compartment", "direction"]):
        ensgs_raw = sub["ensg_versioned"].dropna().tolist()
        ensgs = list(dict.fromkeys(ensgs_raw))
        if len(ensgs) != len(ensgs_raw):
            log.warning(
                "  %s %s: removed %d duplicate ENSG before GSEA (raw=%d -> unique=%d)",
                comp, direction, len(ensgs_raw) - len(ensgs), len(ensgs_raw), len(ensgs),
            )
        gene_sets_full[f"{comp}_{direction}"] = ensgs

    gsea_rows: List[dict] = []
    for tp in TIMEPOINTS_POST:
        da_tp = slice_ee_con(da, tp, log)
        # Drop NaN t, dedupe feature_id (MoTrPAC is already unique but guard)
        da_tp = da_tp.dropna(subset=["t", "feature_id"]).drop_duplicates(subset=["feature_id"])
        ranked = pd.Series(da_tp["t"].values, index=da_tp["feature_id"].values, name=tp)
        log.info("  [tp=%s] ranking length=%d  min_t=%.3f  max_t=%.3f",
                 tp, len(ranked), float(ranked.min()), float(ranked.max()))

        try:
            res = run_prerank_gsea(ranked, gene_sets_full, seed=SEED,
                                   n_perm=N_PERMUTATIONS, log=log)
        except Exception as e:  # pragma: no cover — logged, not swallowed
            log.error("  gseapy.prerank failed at tp=%s: %s", tp, e)
            continue

        # res2d columns (gseapy 1.1.13): Name, Term, ES, NES, NOM p-val, FDR q-val,
        # FWER p-val, Tag %, Gene %, Lead_genes
        universe = ranked.index.tolist()
        for term, row in res.set_index("Term").iterrows():
            nes = float(row["NES"])
            p_gene = float(row["NOM p-val"])
            lead = str(row.get("Lead_genes", ""))
            lead_n = 0 if not lead else len(lead.split(";"))
            # Signature-randomization null — pass signature genes so the function
            # can compute the observed ES on the SAME scale as its null draws
            # (review_run_c.md fix #1).
            sig_name = term
            sig_genes = gene_sets_full.get(sig_name, [])
            sig_size_i = len(sig_genes)
            if sig_size_i < 5:
                p_sig_rand = float("nan")
                observed_es_local = float("nan")
                observed_nes_local = float("nan")
            else:
                p_sig_rand, observed_es_local, observed_nes_local = signature_randomization_null(
                    ranked=ranked,
                    signature_size=sig_size_i,
                    signature_genes=sig_genes,
                    n_perm=N_SIG_RAND_PERM,
                    seed=SEED,
                    rng_universe=universe,
                )
            comp, direction = term.rsplit("_", 1)
            gsea_rows.append({
                "compartment": comp, "direction": direction, "timepoint": tp,
                "NES": nes, "ES": float(row["ES"]),
                "p_gene_perm": p_gene,
                "p_sig_rand_perm": p_sig_rand,
                "observed_es_local": observed_es_local,
                "observed_nes_local": observed_nes_local,
                "fdr_q": float(row.get("FDR q-val", float("nan"))),
                "leading_edge_n": lead_n, "leading_edge_genes": lead,
                "signature_size": sig_size_i,
            })
            log.info(
                "    %s: NES=%.3f  p_gene=%.3g  p_sig_rand=%.3g  lead_n=%d",
                term, nes, p_gene, p_sig_rand, lead_n,
            )

    gsea_df = pd.DataFrame(gsea_rows)
    gsea_df.to_csv(OUT_GSEA, index=False)
    log.info("  wrote %s (rows=%d)", OUT_GSEA, len(gsea_df))

    # ------------------------------------------------------------------
    # 6. C2 descriptive baseline (BALANCE_UNKNOWN, no causal claim)
    # ------------------------------------------------------------------
    log.info("[6] C2 descriptive: EE-pre vs CON-pre (BALANCE_UNKNOWN=True)")
    c2_rows: List[dict] = []
    da_bl = slice_ee_con(da, TIMEPOINT_BASELINE, log)
    da_bl = da_bl.dropna(subset=["t", "feature_id"]).drop_duplicates(subset=["feature_id"])
    ranked_bl = pd.Series(da_bl["t"].values, index=da_bl["feature_id"].values,
                          name=TIMEPOINT_BASELINE)
    try:
        res_bl = run_prerank_gsea(ranked_bl, gene_sets_full, seed=SEED,
                                  n_perm=N_PERMUTATIONS, log=log)
        universe_bl = ranked_bl.index.tolist()
        for term, row in res_bl.set_index("Term").iterrows():
            nes = float(row["NES"])
            p_gene = float(row["NOM p-val"])
            sig_genes_bl = gene_sets_full.get(term, [])
            sig_size_i = len(sig_genes_bl)
            if sig_size_i < 5:
                p_sig_rand = float("nan")
                observed_es_local = float("nan")
                observed_nes_local = float("nan")
            else:
                p_sig_rand, observed_es_local, observed_nes_local = signature_randomization_null(
                    ranked=ranked_bl, signature_size=sig_size_i,
                    signature_genes=sig_genes_bl,
                    n_perm=N_SIG_RAND_PERM,
                    seed=SEED, rng_universe=universe_bl,
                )
            comp, direction = term.rsplit("_", 1)
            lead = str(row.get("Lead_genes", ""))
            c2_rows.append({
                "compartment": comp, "direction": direction,
                "NES": nes, "p_gene_perm": p_gene, "p_sig_rand_perm": p_sig_rand,
                "observed_es_local": observed_es_local,
                "observed_nes_local": observed_nes_local,
                "fdr_q": float(row.get("FDR q-val", float("nan"))),
                "signature_size": sig_size_i,
                "leading_edge_n": 0 if not lead else len(lead.split(";")),
                "BALANCE_UNKNOWN": True,
                "reason_flag": (
                    "MoTrPAC .rda DA lacks per-participant age/BMI/sex/site — "
                    "propensity-score balance check impossible; classified "
                    "DESCRIPTIVE-ONLY per brief v2 C-triad resolution."
                ),
            })
            log.info("    [C2] %s: NES=%.3f  p_gene=%.3g  p_sig_rand=%.3g",
                     term, nes, p_gene, p_sig_rand)
    except Exception as e:  # pragma: no cover
        log.error("  C2 gseapy.prerank failed: %s", e)
    c2_df = pd.DataFrame(c2_rows)
    c2_df.to_csv(OUT_C2, index=False)
    log.info("  wrote %s (rows=%d)", OUT_C2, len(c2_df))

    # ------------------------------------------------------------------
    # 7. TF correlation secondary (SPECULATIVE-only, N=9)
    # ------------------------------------------------------------------
    log.info("[7] TF correlation secondary (9 TFs x 3 compartments)")
    corr_rows: List[dict] = []
    # Pre-cache HLMA DE per compartment for TF lookup
    hlma_de: Dict[str, pd.DataFrame] = {
        comp: pd.read_csv(fp).set_index("gene") for comp, fp in HLMA_DE_FILES.items()
    }
    da_24 = slice_ee_con(da, TIMEPOINT_24HR, log)
    da_24 = da_24.set_index("feature_id")

    for comp in HLMA_DE_FILES:
        tf_vecs = []
        for tf in TF_PANEL:
            h = hlma_de[comp]
            hlma_beta = float(h.loc[tf, "log2FC"]) if tf in h.index else float("nan")
            hlma_t = float(h.loc[tf, "wald_stat"]) if tf in h.index else float("nan")
            ensg = sym2ensg.get(tf)
            motrpac_lfc = float("nan")
            if ensg is not None and ensg in da_24.index:
                motrpac_lfc = float(da_24.loc[ensg, "logFC"])
            # WHY SPECULATIVE-ONLY flag (review_run_c.md non-blocking #1): the
            # compartment-level Pearson r is computed over only 9 TFs, i.e. the
            # effective N for the correlation is 9 (or less after NaN filtering).
            # That is far below what is required for a stable correlation coefficient
            # (rule of thumb n >= 25 for r confidence intervals of usable width), so
            # downstream consumers must not treat this as an ESTABLISHED finding.
            corr_rows.append({
                "TF": tf, "compartment": comp,
                "HLMA_age_log2FC": hlma_beta, "HLMA_wald_t": hlma_t,
                "MoTrPAC_EE_logFC_24hr": motrpac_lfc,
                "effective_n": len(TF_PANEL),
                "classification": "SPECULATIVE_ONLY",
                "classification_reason": (
                    "N=9 TFs per compartment — correlation is underpowered; "
                    "reported as exploratory descriptor, not an ESTABLISHED "
                    "finding (brief v2 rigor clause on n<25 correlations)."
                ),
            })
            tf_vecs.append((hlma_beta, motrpac_lfc))
        arr = np.array(tf_vecs, dtype=float)
        mask = np.isfinite(arr).all(axis=1)
        if mask.sum() >= 3:
            r, p = stats.pearsonr(arr[mask, 0], arr[mask, 1])
        else:
            r, p = float("nan"), float("nan")
        # Stamp the compartment-level r on each of this compartment's rows (last 9).
        for row in corr_rows[-len(TF_PANEL):]:
            row["pearson_r_compartment"] = float(r)
            row["pearson_p_compartment"] = float(p)
            row["n_usable"] = int(mask.sum())
        log.info("  %s: r=%.3f  p=%.3g  n_usable=%d",
                 comp, r if not np.isnan(r) else float("nan"),
                 p if not np.isnan(p) else float("nan"), int(mask.sum()))

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_CORR, index=False)
    log.info("  wrote %s", OUT_CORR)

    # ------------------------------------------------------------------
    # 8. Decision / summary per brief decision rules
    # ------------------------------------------------------------------
    log.info("[8] Compiling c_summary.json")
    def classify_cell(row: pd.Series) -> str:
        """Apply brief decision rule per (compartment, direction, timepoint) cell.

        WHY this helper: the brief line 148-150 rule is on PAIRED UP/DOWN signs at the
        young stratum at 24-hr. We compute a simpler per-cell evidence tag; the final
        decision call is a roll-up in the summary.

        Null-disagreement rule (review_run_c.md fix #2): a naive
        `(p_gene<0.05) != (p_sig<0.05)` mis-triggers on borderline pairs like
        p=0.049 vs p=0.051, and MISSES genuine conflicts like p=0.04 vs p=0.19
        where both fall on the same side of 0.05 only by arithmetic coincidence.
        We require a meaningful gap: NULLS_DISAGREE when one null is "significant"
        (<0.05) AND the other is "clearly non-significant" (>0.2). Pairs both
        under 0.2 are treated as broadly consistent; pairs both over 0.05 agree on
        non-enrichment.

        UNINTERPRETABLE guard (review_run_c.md fix #3): if both p-values are NaN,
        `nanmax` emits a warning and returns NaN, which silently degrades the
        downstream abs(nes) check. We classify that row UNINTERPRETABLE explicitly.
        """
        nes = row["NES"]
        p_gene = row["p_gene_perm"]
        p_sig = row["p_sig_rand_perm"]
        if np.isnan(nes):
            return "NA"
        p_gene_nan = bool(np.isnan(p_gene))
        p_sig_nan = bool(np.isnan(p_sig))
        if p_gene_nan and p_sig_nan:
            return "UNINTERPRETABLE"
        # Gap-based disagreement: one null clearly significant, the other clearly not.
        if not p_gene_nan and not p_sig_nan:
            p_min = min(p_gene, p_sig)
            p_max_raw = max(p_gene, p_sig)
            if p_min < 0.05 and p_max_raw > 0.2:
                return "NULLS_DISAGREE"
        # p_max for the enrichment gate: require BOTH available nulls to pass.
        # Use nanmax but with the NaN-both case already handled above.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p_max = float(np.nanmax([p_gene, p_sig]))
        if abs(nes) >= NES_THRESHOLD and p_max < P_THRESHOLD:
            return "ENRICHED"
        return "NOT_ENRICHED"

    gsea_df_c1 = gsea_df.copy()
    # Guard against empty GSEA result (e.g. gseapy filtered all sets out). The
    # previous version crashed here because `apply` on an empty DataFrame returns
    # an empty DataFrame-shaped result that pandas refuses to assign as a column.
    if gsea_df_c1.empty or not {"NES", "p_gene_perm", "p_sig_rand_perm"}.issubset(gsea_df_c1.columns):
        gsea_df_c1 = gsea_df_c1.reindex(
            columns=list(gsea_df_c1.columns) + ["cell_verdict"]
        )
        gsea_df_c1["cell_verdict"] = pd.Series(dtype=str)
        log.warning(
            "  gsea_df_c1 is empty or missing required columns; downstream rollup "
            "will flag all compartments NA."
        )
    else:
        gsea_df_c1["cell_verdict"] = gsea_df_c1.apply(classify_cell, axis=1)

    summary = {
        "iteration": 64,
        "batch": "batch_064",
        "analysis": "C",
        "seed": SEED,
        "n_permutations_gene_label": N_PERMUTATIONS,
        "n_permutations_signature_rand": N_SIG_RAND_PERM,
        "nes_threshold": NES_THRESHOLD,
        "p_threshold": P_THRESHOLD,
        "signature_size": SIG_SIZE,
        "ieg_verdict": ieg_verdict,
        "data_limitations": {
            "age_stratification": (
                "NOT AVAILABLE — MoTrPAC .rda DA aggregates do not contain "
                "per-participant age covariates. Pooled-only analysis run."
            ),
            "between_subject_permutation_null": (
                "NOT AVAILABLE — requires per-participant raw counts. "
                "Signature-randomization null substituted as secondary null. "
                "Gene-label permutation (gseapy default) also reported."
            ),
            "c2_propensity_score_balance": (
                "NOT AVAILABLE — no participant covariates in .rda DA. "
                "C2 is DESCRIPTIVE-ONLY with BALANCE_UNKNOWN=True."
            ),
        },
        "c1_24hr_cells": [
            {
                "compartment": r["compartment"],
                "direction": r["direction"],
                "NES": r["NES"], "p_gene": r["p_gene_perm"],
                "p_sig_rand": r["p_sig_rand_perm"],
                "verdict": r["cell_verdict"],
            }
            for _, r in gsea_df_c1[gsea_df_c1["timepoint"] == TIMEPOINT_24HR].iterrows()
        ],
        "outputs": {
            "signatures": str(OUT_SIGNATURES),
            "ieg_precheck": str(OUT_IEG),
            "gsea_results": str(OUT_GSEA),
            "c2_baseline": str(OUT_C2),
            "correlation": str(OUT_CORR),
            "log": str(LOG_FP),
        },
        "decision_pre_commit": (
            "Per brief v2 decision rules: because age stratification and "
            "between-subject permutation null are both unavailable from the .rda "
            "aggregate, ESTABLISHED calls are structurally unreachable this iter. "
            "Maximum attainable classification is SUGGESTED at the pooled level "
            "(if NES<-2.0 aging-UP AND NES>+2.0 aging-DOWN, both p<0.05 under BOTH "
            "nulls, AND ieg_precheck_pooled==PASS). Otherwise INCONCLUSIVE."
        ),
    }

    # Roll-up final call per compartment at 24-hr pooled
    rollup: Dict[str, str] = {}
    for comp in HLMA_DE_FILES:
        up = gsea_df_c1[(gsea_df_c1["compartment"] == comp)
                        & (gsea_df_c1["direction"] == "UP")
                        & (gsea_df_c1["timepoint"] == TIMEPOINT_24HR)]
        dn = gsea_df_c1[(gsea_df_c1["compartment"] == comp)
                        & (gsea_df_c1["direction"] == "DOWN")
                        & (gsea_df_c1["timepoint"] == TIMEPOINT_24HR)]
        if len(up) == 0 or len(dn) == 0:
            rollup[comp] = "NA"
            continue
        if ieg_verdict["ieg_precheck_pooled"] != "PASS":
            rollup[comp] = "INCONCLUSIVE_IEG_GATE"
            continue
        u = up.iloc[0]
        d = dn.iloc[0]
        if u["cell_verdict"] == "NULLS_DISAGREE" or d["cell_verdict"] == "NULLS_DISAGREE":
            rollup[comp] = "INCONCLUSIVE_NULLS_DISAGREE"
            continue
        # WHY this clause (review_run_c.md fix #3 follow-through): if either paired
        # cell is UNINTERPRETABLE (both nulls returned NaN), we cannot form a
        # paired-sign judgement and the compartment rolls up as inconclusive.
        if u["cell_verdict"] == "UNINTERPRETABLE" or d["cell_verdict"] == "UNINTERPRETABLE":
            rollup[comp] = "INCONCLUSIVE_UNINTERPRETABLE"
            continue
        up_ok = (u["NES"] <= -NES_THRESHOLD
                 and max(u["p_gene_perm"], u["p_sig_rand_perm"]) < P_THRESHOLD)
        dn_ok = (d["NES"] >= NES_THRESHOLD
                 and max(d["p_gene_perm"], d["p_sig_rand_perm"]) < P_THRESHOLD)
        if up_ok and dn_ok:
            rollup[comp] = "SUGGESTED_POOLED"
        else:
            rollup[comp] = "INCONCLUSIVE"
    summary["rollup_pooled_24hr"] = rollup
    log.info("  rollup pooled 24-hr: %s", rollup)

    with open(OUT_SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("  wrote %s", OUT_SUMMARY)

    log.info("=== batch_064 run_c_motrpac.py DONE ===")
    return 0


def _write_partial_summary(exc: BaseException) -> None:
    """Emit a PARTIAL_FAILURE c_summary.json if main() crashes before writing it.

    WHY: downstream consumers (Marvin PI review, Vera audit) rely on a canonical
    c_summary.json to decide whether analysis C is usable. When the script dies
    mid-flow the old behaviour was to leave no summary file, forcing a manual
    re-run just to recover state. A partial-failure record is more honest and
    preserves the information needed to debug and re-try.
    """
    import traceback
    partial = {
        "iteration": 64,
        "batch": "batch_064",
        "analysis": "C",
        "status": "PARTIAL_FAILURE",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "outputs": {
            "signatures": str(OUT_SIGNATURES),
            "ieg_precheck": str(OUT_IEG),
            "gsea_results": str(OUT_GSEA),
            "c2_baseline": str(OUT_C2),
            "correlation": str(OUT_CORR),
            "log": str(LOG_FP),
        },
    }
    try:
        with open(OUT_SUMMARY, "w") as fh:
            json.dump(partial, fh, indent=2, default=str)
    except Exception:  # pragma: no cover — last-ditch, never raise out of finally
        pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as e:
        _write_partial_summary(e)
        raise
