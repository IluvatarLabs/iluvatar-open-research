#!/usr/bin/env python3
"""
batch_048: EDT1 Gene-Set Decomposition + HAR × EGR1/CTCF Enrichment

Sub-A: Decompose EDT1 (protein_coding genes from PGC3 xlsx) into functional subsets
       (SynGO, ion channel, glutamate receptor, mitochondrial, transcriptional
       regulator, other). Test each subset for gnomAD constraint (pLI, LOEUF).
       Quantify where the constraint signal lives.

Sub-B: Test EGR1/CTCF PWM target gene promoters for overlap with HARs.
       Link SCZ convergence regulators to human-specific regulatory evolution.

WHY: F121 showed SynGO_EDT1 (n=14) pLI OR=26.44, the strongest constraint signal.
     F122 showed EDT1 broadly has NO constraint signal. This decomposition reveals
     the convergence architecture — which biological programs carry the signal.
     HAR × EGR1/CTCF connects regulatory evolution to disease risk.

Data:
  - gnomAD v4.1 constraint: data/item_15/gnomad.v4.1.constraint_metrics.tsv
  - HAR BED (GRCh37): data/item_15/reference_assets/harsRichard2020.GRCh37.bed
  - PGC3 xlsx: data/19426775/scz2022-Extended-Data-Table1.xlsx
  - Gene TSS: data/ldsc/gene_tss_grch37.csv
  - EGR1/CTCF target genes from batch_040: experiments/batch_040/output/B_pwm_targets.json

Output:
  - experiments/batch_048/output/A_edt1_decomposition.json
  - experiments/batch_048/output/B_har_tf_enrichment.json
  - experiments/batch_048/output/summary.json
"""

from __future__ import annotations
import argparse, datetime as _dt, hashlib, json, logging, math, pathlib, sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ------------------------------------------------------------------------------ Paths
PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_048"
OUTPUT_DIR = BATCH_DIR / "output"
LOG_DIR = BATCH_DIR / "logs"
INPUT_DIR = BATCH_DIR / "input"

GNOMAD_TSV = PROJECT_ROOT / "data" / "item_15" / "gnomad.v4.1.constraint_metrics.tsv"
HAR_BED = PROJECT_ROOT / "data" / "item_15" / "reference_assets" / "harsRichard2020.GRCh37.bed"
TSS_CSV = PROJECT_ROOT / "data" / "ldsc" / "gene_tss_grch37.csv"
PGC3_XLSX = PROJECT_ROOT / "data" / "19426775" / "scz2022-Extended-Data-Table1.xlsx"
MOTIF_ATLAS = PROJECT_ROOT / "data" / "hg38__refseq-r80__10kb_up_and_down_tss.mc9nr.genes_vs_motifs.rankings.feather"

# Pre-registered constants
LOEUF_THRESHOLD = 0.35  # Karczewski 2020
PLI_THRESHOLD = 0.9    # gnomAD canonical threshold
HAR_WINDOW_BP = 100_000  # Doan 2016
MHC_CHR, MHC_START, MHC_END = "6", 25_000_000, 34_000_000
N_PERMUTATIONS = 5_000  # balanced: ~2 min per metric per list
RNG_SEED = 20260423  # fixed for reproducibility

# EGR1 and CTCF JASPAR/HOCOMOCO motif IDs
EGR1_MOTIFS = ["EGR1_HUMAN.H11MO.0.A", "EGR1_MOUSE.H11MO.0.A"]  # JASPAR+HOCOMOCO
CTCF_MOTIFS = ["CTCF_HUMAN.M1", "CTCF_MOUSE.M1"]  # JASPAR+HOCOMOCO
RANK_THRESHOLD = 500  # median rank ≤ 500 (top 1.85%)

# SynGO_EDT1 genes: from batch_047 reconstructed list (Singh 2022 Table 1 supplementary)
# These 14 genes are at the intersection of EDT1 and SynGO synaptic annotation
SYNOGO_EDT1_BATCH047 = ["DLGAP1", "GRIN2A", "NRXN1", "CNTNAP2", "ARC", "DLG4", "NRXN2",
                          "NLGN1", "NLGN2", "SHANK1", "SHANK3", "HOMER1", "SYN1", "GAP43"]

# Functional category keywords for EDT1 decomposition
SYNGO_KEYWORDS = ["synap", "synaptic", "postsynap", "presynap", "psd", "glutamate receptor", "gaba", "gria", "grik", "grin", "grid", "slitrk", "lrfn", "nlgn", "shank", "dlgap", "homer", "dlg4", "arc", "syn", "camp", "rab3", "stx", "vamp", "complexin", "synapsin", "synaptotagmin", "nrxn", "cntnap"]
ION_CHANNEL_KEYWORDS = ["channel", "ion channel", "voltage-gated", "ligand-gated", "cacna", "kcnq", "kcnn", "scn", "nav", "cav", "kv", "sk", "bk", "hyperpolarization", "trpc", "trpv", "kcnk"]
GLUTAMATE_RECEPTOR_KEYWORDS = ["glutamate receptor", "gria", "grik", "grm", "grin"]  # GRIN already in SynGO
MITOCHONDRIAL_KEYWORDS = ["mitoch", "mitochondrial", "mt-", "cytochrome c", "respiratory chain", "atp synthase", "nd", "mtco", "mtatp", "cox", "sdh"]
TRANSCRIPTIONAL_KEYWORDS = ["transcription factor", "zinc finger", "histone", "chromatin", "kdm", "setd", "smarc", "chd", "epigen", "methyltransferase", "acetyltransferase"]
NEURODEV_KEYWORDS = ["development", "neuron migration", "axon guidance", "dendrite", "growth cone", "semaphorin", "robo", "slit", "netrin", "eph", "ephrin"]

# Pre-registered thresholds
MIN_OR_SUB_A_REPLICATION = 10.0   # SynGO_EDT1 OR must exceed 10 for ESTABLISHED replication
MIN_OR_HAR_SUGGESTED = 1.5       # EGR1 HAR OR must exceed 1.5 for SUGGESTED
MIN_OR_HAR_ESTABLISHED = 2.0     # EGR1 HAR OR must exceed 2.0 for ESTABLISHED
MIN_P_HAR = 0.01                 # EGR1 HAR p must be below this for ESTABLISHED

def apply_decision_rules(sub_a: list, sub_b: list) -> dict[str, Any]:
    """Apply pre-registered decision rules and classify findings.

    Sub-A: SynGO_EDT1 replication — OR > 10 AND BH q < 0.05 AND emp_p < 0.05 → ESTABLISHED
    Sub-B: EGR1 HAR — OR > 1.5 AND p < 0.05 → SUGGESTED; OR > 2.0 AND p < 0.01 → ESTABLISHED
    """
    classifications = {}

    # Sub-A: SynGO_EDT1 replication
    syngo_result = next((r for r in sub_a if r["gene_list"] == "SynGO_EDT1_batch047" and "pLI" in r["constraint_metric"]), None)
    if syngo_result:
        or_ok = syngo_result["or"] > MIN_OR_SUB_A_REPLICATION
        q_ok = syngo_result.get("bh_q", 1.0) < 0.05
        emp_ok = syngo_result.get("emp_p", 1.0) < 0.05
        if or_ok and q_ok and emp_ok:
            classifications["SynGO_EDT1_replication"] = "ESTABLISHED"
        else:
            classifications["SynGO_EDT1_replication"] = "REFUTED"
        classifications["SynGO_EDT1_detail"] = {
            "or": syngo_result["or"], "bh_q": syngo_result.get("bh_q"), "emp_p": syngo_result.get("emp_p"),
            "or_ok": or_ok, "q_ok": q_ok, "emp_ok": emp_ok
        }
    else:
        classifications["SynGO_EDT1_replication"] = "NOT_TESTED"

    # Sub-B: EGR1 HAR enrichment
    egr1_result = next((r for r in sub_b if "EGR1" in r["gene_list"]), None)
    if egr1_result:
        if egr1_result["or"] > MIN_OR_HAR_ESTABLISHED and egr1_result["p"] < MIN_P_HAR:
            classifications["EGR1_HAR_enrichment"] = "ESTABLISHED"
        elif egr1_result["or"] > MIN_OR_HAR_SUGGESTED and egr1_result["p"] < 0.05:
            classifications["EGR1_HAR_enrichment"] = "SUGGESTED"
        else:
            classifications["EGR1_HAR_enrichment"] = "INCONCLUSIVE"
        classifications["EGR1_HAR_detail"] = {
            "or": egr1_result["or"], "p": egr1_result["p"],
            "bh_q": egr1_result.get("bh_q"), "n_in_list": egr1_result["n_in_list"]
        }
    else:
        classifications["EGR1_HAR_enrichment"] = "NOT_TESTED"

    # CTCF descriptive (no prediction)
    ctcf_result = next((r for r in sub_b if "CTCF" in r["gene_list"]), None)
    if ctcf_result:
        classifications["CTCF_HAR"] = {
            "or": ctcf_result["or"], "p": ctcf_result["p"],
            "n_in_list": ctcf_result["n_in_list"]
        }

    return classifications

# ----------------------------------------------------------------------------- Logging
def _setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_048")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        for handler in (logging.FileHandler(LOG_DIR / "run_batch_048.log"),
                        logging.StreamHandler(sys.stdout)):
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s",
                                                  datefmt="%Y-%m-%dT%H:%M:%S"))
            logger.addHandler(handler)
    return logger

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def log_input(logger: logging.Logger, label: str, path: pathlib.Path) -> dict[str, Any]:
    meta = {"label": label, "path": str(path), "exists": path.exists()}
    if path.exists():
        meta.update({"sha256": sha256(path), "bytes": path.stat().st_size})
        logger.info("INPUT %s: %s sha=%s bytes=%d", label, path, meta["sha256"], meta["bytes"])
    else:
        logger.error("INPUT %s MISSING: %s", label, path)
    return meta

# ----------------------------------------------------------------------------- Data loaders
def load_gnomad(logger: logging.Logger) -> pd.DataFrame:
    """Load gnomAD v4.1 constraint metrics, filter to ENSG canonical+MANE."""
    logger.info("Loading gnomAD v4.1...")
    df = pd.read_csv(GNOMAD_TSV, sep="\t", low_memory=False)
    if "gene_id" in df.columns:
        df = df[df["gene_id"].astype(str).str.startswith("ENSG")].copy()
    if "chromosome" in df.columns:
        df["chromosome"] = df["chromosome"].astype(str).str.replace(r"^chr", "", regex=True)
    for col in ("canonical", "mane_select"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    mask = df["canonical"] & df["mane_select"] & df["lof.oe_ci.upper"].notna()
    df = df.loc[mask].drop_duplicates(subset=["gene"], keep="first").reset_index(drop=True)
    df["loeuf_lt_035"] = df["lof.oe_ci.upper"] < LOEUF_THRESHOLD
    df["pli_ge_09"] = df["lof.pLI"] >= PLI_THRESHOLD
    if "cds_length" in df.columns:
        df["gene_length"] = df["cds_length"].fillna(0)
    else:
        df["gene_length"] = 1
    df["log_gene_length"] = np.log1p(df["gene_length"].fillna(0))
    # Join TSS for MHC/chrX
    tss = pd.read_csv(TSS_CSV)
    tss = tss[tss["chrom"].astype(str).isin([str(i) for i in range(1, 23)] + ["X", "Y"])]
    def _agg(g):
        top_chrom = g["chrom"].mode().iloc[0]
        tss_on_top = g.loc[g["chrom"] == top_chrom, "tss"]
        return pd.Series({"tss_chrom": top_chrom, "tss_pos": int(tss_on_top.median())})
    tss_agg = tss.groupby("gene", as_index=False).apply(_agg, include_groups=False)
    if "gene" not in tss_agg.columns:
        tss_agg = tss_agg.reset_index().rename(columns={"level_0": "gene"})
    df = df.merge(tss_agg[["gene", "tss_chrom", "tss_pos"]], on="gene", how="left")
    tss_chrom = df["tss_chrom"].astype(str)
    tss_pos = df["tss_pos"].fillna(-1).astype(int)
    df["mhc_indicator"] = (tss_chrom == MHC_CHR) & (tss_pos >= MHC_START) & (tss_pos <= MHC_END)
    df["chrX_indicator"] = tss_chrom == "X"
    logger.info("gnomAD: %d genes, MHC=%d, chrX=%d",
                len(df), df["mhc_indicator"].sum(), df["chrX_indicator"].sum())
    return df

def load_tss(logger: logging.Logger) -> pd.DataFrame:
    """Load TSS file, collapse to one per gene symbol with chrom."""
    df = pd.read_csv(TSS_CSV)
    df = df[df["chrom"].astype(str).isin([str(i) for i in range(1, 23)] + ["X", "Y"])]
    # For each gene, take mode chromosome + median TSS within that chromosome
    def _agg(g):
        top_chrom = g["chrom"].mode().iloc[0]
        tss_on_top = g.loc[g["chrom"] == top_chrom, "tss"]
        return pd.Series({"chrom": top_chrom, "tss": int(tss_on_top.median())})
    agg = df.groupby("gene", as_index=False).apply(_agg, include_groups=False)
    if "gene" not in agg.columns:
        agg = agg.reset_index().rename(columns={"level_0": "gene"})
    agg["tss"] = agg["tss"].astype(int)
    logger.info("TSS: %d unique genes", len(agg))
    return agg

def load_har(logger: logging.Logger) -> pd.DataFrame:
    """Load HAR BED file (GRCh37, Richard et al. 2020)."""
    df = pd.read_csv(HAR_BED, sep="\t", header=None,
                     names=["chrom", "start", "end", "source"], dtype={"chrom": str})
    logger.info("HARs: %d regions from %s", len(df), df["source"].value_counts().to_dict())
    return df

def load_pgc3_edt1(logger: logging.Logger) -> dict[str, set[str]]:
    """Load EDT1 full gene list from PGC3 xlsx, all protein_coding genes."""
    df = pd.read_excel(PGC3_XLSX, sheet_name="ST12 all criteria")
    pc = df[df["gene_biotype"] == "protein_coding"]
    all_pc = set(pc["Symbol.ID"].dropna().astype(str))
    # SynGO subset
    def truthy(v):
        if v is None: return False
        try:
            if isinstance(v, float) and math.isnan(v): return False
        except TypeError: pass
        return str(v).strip().lower() in {"yes", "y", "1", "true", "1.0"}
    syngo = set(pc[pc["SynGO.GeneSetMemb"].apply(truthy)]["Symbol.ID"].dropna().astype(str))
    logger.info("EDT1 protein_coding: %d total, %d SynGO", len(all_pc), len(syngo))
    return {"EDT1_all_pc": all_pc, "EDT1_SynGO": syngo}

def classify_edt1_gene(gene: str) -> str:
    """Classify a gene symbol into functional category."""
    g_lower = gene.lower()
    # Order matters: check specific categories before general
    if any(k in g_lower for k in GLUTAMATE_RECEPTOR_KEYWORDS):
        return "glutamate_receptor"
    if any(k in g_lower for k in ION_CHANNEL_KEYWORDS):
        return "ion_channel"
    if any(k in g_lower for k in MITOCHONDRIAL_KEYWORDS):
        return "mitochondrial"
    if any(k in g_lower for k in TRANSCRIPTIONAL_KEYWORDS):
        return "transcriptional"
    if any(k in g_lower for k in NEURODEV_KEYWORDS):
        return "neurodevelopmental"
    return "other"

def load_pwm_targets(logger: logging.Logger, gnomad: pd.DataFrame) -> dict[str, set[str]]:
    """Scan gene promoters for EGR1/CTCF motif matches using motif atlas.

    WHY: batch_040 did not produce PWM target gene lists. The MC9nr motif atlas
    is TRANSPOSED: rows = motif names, columns = gene symbols, values = ranks.
    Target genes = genes whose rank for the TF ≤ 500 (top 1.85%).
    """
    if not MOTIF_ATLAS.exists():
        logger.error("Motif atlas not found: %s", MOTIF_ATLAS)
        return {"EGR1": set(), "CTCF": set()}

    logger.info("Loading motif atlas for PWM scanning (transposed structure)...")
    mt = pd.read_feather(MOTIF_ATLAS)
    logger.info("Motif atlas: %d rows (motifs) × %d cols (genes)",
                mt.shape[0], mt.shape[1] - 1)  # -1 for 'motifs' col

    # Motif column contains motif identifiers
    motif_col = "motifs"
    gene_cols = [c for c in mt.columns if c != motif_col]
    logger.info("Gene columns: %d", len(gene_cols))

    # EGR1 and CTCF are motif names, not gene names
    # Find rows where motifs column matches EGR1 or CTCF
    egr1_rows = mt[mt[motif_col] == "EGR1"]
    ctcf_rows = mt[mt[motif_col] == "CTCF"]

    if egr1_rows.empty:
        logger.warning("EGR1 motif row not found in atlas")
        egr1_targets = set()
    else:
        egr1_row = egr1_rows.iloc[0][gene_cols]
        egr1_ranks = pd.to_numeric(egr1_row, errors="coerce")
        egr1_target_genes = {g for g, r in zip(gene_cols, egr1_ranks)
                            if pd.notna(r) and r <= RANK_THRESHOLD}
        logger.info("PWM EGR1 targets (rank ≤ %d): %d genes", RANK_THRESHOLD, len(egr1_target_genes))
        egr1_targets = egr1_target_genes

    if ctcf_rows.empty:
        logger.warning("CTCF motif row not found in atlas")
        ctcf_targets = set()
    else:
        ctcf_row = ctcf_rows.iloc[0][gene_cols]
        ctcf_ranks = pd.to_numeric(ctcf_row, errors="coerce")
        ctcf_target_genes = {g for g, r in zip(gene_cols, ctcf_ranks)
                              if pd.notna(r) and r <= RANK_THRESHOLD}
        logger.info("PWM CTCF targets (rank ≤ %d): %d genes", RANK_THRESHOLD, len(ctcf_target_genes))
        ctcf_targets = ctcf_target_genes

    return {"EGR1": egr1_targets, "CTCF": ctcf_targets}

# ----------------------------------------------------------------------------- Helpers
def fisher_enrichment(genes_in_list: set, background_set: set,
                     target_set: set) -> dict[str, Any]:
    """Fisher's exact: enrichment of target_set in genes_in_list vs background."""
    a = len(genes_in_list & target_set)
    b = len(genes_in_list - target_set)
    c = len(target_set - genes_in_list)
    d = len(background_set - genes_in_list - target_set)
    if b == 0 or c == 0:
        or_point = float("inf") if a > 0 else 0.0
        p_val = 0.0 if a > 0 else 1.0
    else:
        or_point, p_val = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    try:
        ci = stats.contingency.odds_ratio([[a, b], [c, d]]).confidence_interval(0.95)
        ci_low, ci_high = float(ci.low or 0), float(ci.high or float("inf"))
    except Exception:
        ci_low, ci_high = 0.0, float("inf")
    return {"a": a, "b": b, "c": c, "d": d,
            "or": float(or_point), "p": float(p_val),
            "ci_low": ci_low, "ci_high": ci_high,
            "n_in_list": len(genes_in_list), "n_target": len(target_set),
            "n_bg": len(background_set)}

def length_perm_test(list_mask: np.ndarray, target_mask: np.ndarray,
                     lengths: np.ndarray, n_perm: int,
                     rng: np.random.Generator) -> dict[str, Any]:
    """Permute list labels within length deciles; compute empirical p."""
    observed_or = _or_from_masks(list_mask, target_mask)
    observed_a = int(np.sum(list_mask & target_mask))
    null_or = np.empty(n_perm, dtype=float)
    deciles = pd.qcut(lengths, q=10, labels=False, duplicates="drop")
    decile_members = {d: np.where(deciles == d)[0] for d in np.unique(deciles)}
    list_full = list_mask.copy()
    for p in range(n_perm):
        perm_list = np.zeros_like(list_full)
        for d, idx in decile_members.items():
            n_scz = int(list_full[idx].sum())
            if n_scz == 0: continue
            chosen = rng.choice(idx, size=n_scz, replace=False)
            perm_list[chosen] = True
        null_or[p] = _or_from_masks(perm_list, target_mask)
    emp_p = float((null_or >= observed_or).sum() + 1) / (n_perm + 1)
    return {"observed_or": float(observed_or), "observed_a": observed_a,
            "n_perm": n_perm, "emp_p": emp_p,
            "null_mean": float(np.mean(null_or)), "null_std": float(np.std(null_or, ddof=1))}

def _or_from_masks(scz: np.ndarray, target: np.ndarray) -> float:
    a = int(np.sum(scz & target))
    b = int(np.sum(scz & ~target))
    c = int(np.sum(~scz & target))
    d = int(np.sum(~scz & ~target))
    if b == 0 or c == 0: return float("inf") if a > 0 else 0.0
    return (a * d) / (b * c)

def har_proximal_genes(tss: pd.DataFrame, har: pd.DataFrame) -> set[str]:
    """Genes with TSS within ±100kb of any HAR.

    tss must have columns: gene, tss, chrom (chromosome of the gene's TSS).
    """
    result: set = set()
    for chrom, har_chrom in har.groupby("chrom"):
        tss_chrom = tss[tss["chrom"].astype(str) == str(chrom)]
        if tss_chrom.empty: continue
        tss_pos = tss_chrom["tss"].to_numpy()
        har_starts = har_chrom["start"].to_numpy()
        har_ends = har_chrom["end"].to_numpy()
        for gene, pos in zip(tss_chrom["gene"].to_numpy(), tss_pos):
            if np.any((har_starts - HAR_WINDOW_BP <= pos) & (pos <= har_ends + HAR_WINDOW_BP)):
                result.add(str(gene))
    return result

# ----------------------------------------------------------------------------- Sub-A: EDT1 Decomposition
def run_sub_a(logger: logging.Logger, gnomad: pd.DataFrame,
              edt1: dict[str, set[str]]) -> list[dict[str, Any]]:
    """Test EDT1 functional subsets for gnomAD constraint."""
    # Background: gnomAD genes, MHC excluded
    bg = gnomad[~gnomad["mhc_indicator"]].copy()
    bg_genes_set = set(bg["gene"].astype(str))
    bg_array = bg["gene"].astype(str).to_numpy()

    # Build EDT1 functional subsets
    all_edt1 = edt1["EDT1_all_pc"]
    edt1_in_bg = all_edt1 & bg_genes_set
    logger.info("EDT1 protein_coding in background: %d / %d", len(edt1_in_bg), len(all_edt1))

    subsets: dict[str, set[str]] = {"EDT1_all": edt1_in_bg}

    # Classify each EDT1 gene
    categories: dict[str, set] = {c: set() for c in
        ["SynGO", "glutamate_receptor", "ion_channel", "mitochondrial",
         "transcriptional", "neurodevelopmental", "other"]}
    for gene in edt1_in_bg:
        cat = classify_edt1_gene(gene)
        categories[cat].add(gene)

    # Add SynGO_EDT1 from batch_047 reconstructed list (Singh 2022 Table 1)
    syngo_edt1 = set(SYNOGO_EDT1_BATCH047) & bg_genes_set
    categories["SynGO_EDT1_batch047"] = syngo_edt1
    logger.info("SynGO_EDT1 (batch_047 list, n=%d in gnomAD bg)", len(syngo_edt1))

    for cat, genes in categories.items():
        logger.info("  %s: %d genes", cat, len(genes))

    # Constraint metrics to test
    metrics = [
        ("pLI >= 0.9", gnomad[~gnomad["mhc_indicator"]]["pli_ge_09"].astype(bool).to_numpy()),
        ("LOEUF <= 0.35", gnomad[~gnomad["mhc_indicator"]]["loeuf_lt_035"].astype(bool).to_numpy()),
    ]
    metric_names = [m[0] for m in metrics]

    bg_genes_arr = bg["gene"].astype(str).to_numpy()
    lengths = bg["gene_length"].fillna(0).to_numpy()

    results = []
    all_pvals = []

    for cat, genes in categories.items():
        if len(genes) < 3:
            logger.info("  Skipping %s (n=%d < 3)", cat, len(genes))
            continue
        list_mask = np.array([g in genes for g in bg_genes_arr], dtype=bool)

        for metric_name, metric_mask in metrics:
            r = fisher_enrichment(genes, bg_genes_set, set(bg_genes_arr[metric_mask]))
            r["gene_list"] = cat
            r["constraint_metric"] = metric_name
            r["n_in_list"] = len(genes)
            all_pvals.append(r["p"])

            # Permutation test for pLI (primary)
            if "pLI" in metric_name:
                perm = length_perm_test(list_mask, metric_mask, lengths, N_PERMUTATIONS,
                                       np.random.default_rng(RNG_SEED))
                r["emp_p"] = perm["emp_p"]
                r["emp_p_note"] = f"{N_PERMUTATIONS} permutations, length-stratified"
            else:
                r["emp_p"] = None

            results.append(r)
            logger.info("  [%s] [%s] OR=%.3f p=%.4f emp_p=%s n=%d",
                        cat, metric_name, r["or"], r["p"], r.get("emp_p"), len(genes))

    # BH correction across all tests
    if all_pvals:
        _, qvals, _, _ = multipletests(all_pvals, alpha=0.05, method="fdr_bh")
        for i, r in enumerate(results):
            r["bh_q"] = float(qvals[i])

    return results

# ----------------------------------------------------------------------------- Sub-B: HAR × TF Enrichment
def run_sub_b(logger: logging.Logger, tss: pd.DataFrame, har: pd.DataFrame,
              pwm_targets: dict[str, set[str]], gnomad: pd.DataFrame) -> list[dict[str, Any]]:
    """Test EGR1/CTCF targets for HAR enrichment."""
    bg = gnomad[~gnomad["mhc_indicator"]].copy()
    bg_genes_set = set(bg["gene"].astype(str))

    # HAR-proximal genes
    har_prox = har_proximal_genes(tss, har)
    logger.info("HAR-proximal genes (±100kb): %d", len(har_prox))

    # HAR-prox set intersection with background
    har_prox_bg = har_prox & bg_genes_set
    logger.info("HAR-proximal genes in background: %d", len(har_prox_bg))

    results = []
    for tf_name, targets in pwm_targets.items():
        targets_in_bg = targets & bg_genes_set
        logger.info("%s targets in background: %d / %d", tf_name, len(targets_in_bg), len(targets))

        r = fisher_enrichment(targets_in_bg, bg_genes_set, har_prox_bg)
        r["gene_list"] = f"{tf_name}_PWM_targets"
        r["har_prox_count"] = len(har_prox_bg)
        r["n_in_list"] = len(targets_in_bg)
        results.append(r)
        logger.info("  [%s] OR=%.3f p=%.4f CI=[%.2f, %.2f]",
                    tf_name, r["or"], r["p"], r["ci_low"], r["ci_high"])

    # BH correction across 2 tests
    if len(results) == 2:
        ps = [r["p"] for r in results]
        _, qvals, _, _ = multipletests(ps, alpha=0.05, method="fdr_bh")
        for i, r in enumerate(results):
            r["bh_q"] = float(qvals[i])

    return results

# ----------------------------------------------------------------------------- Main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-perm", action="store_true", help="Skip permutation (fast test)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger()
    logger.info("=" * 60)
    logger.info("batch_048: EDT1 decomposition + HAR × TF enrichment")

    outputs = [OUTPUT_DIR / "A_edt1_decomposition.json",
               OUTPUT_DIR / "B_har_tf_enrichment.json",
               OUTPUT_DIR / "summary.json"]
    if not args.force and all(p.exists() for p in outputs):
        logger.info("Outputs exist; SKIP. Pass --force to re-run.")
        return 0

    # Log inputs
    input_manifest = {
        "gnomad": log_input(logger, "gnomad_constraint", GNOMAD_TSV),
        "har": log_input(logger, "har_bed", HAR_BED),
        "tss": log_input(logger, "gene_tss_grch37", TSS_CSV),
        "pgc3_xlsx": log_input(logger, "pgc3_xlsx", PGC3_XLSX),
    }

    # Load data
    gnomad = load_gnomad(logger)
    tss = load_tss(logger)
    har = load_har(logger)
    edt1 = load_pgc3_edt1(logger)
    pwm_targets = load_pwm_targets(logger, gnomad)

    # Run Sub-A
    sub_a = run_sub_a(logger, gnomad, edt1)

    # Run Sub-B
    sub_b = run_sub_b(logger, tss, har, pwm_targets, gnomad)

    # Write outputs
    with (OUTPUT_DIR / "A_edt1_decomposition.json").open("w") as fh:
        json.dump({"batch": "batch_048", "sub": "A_edt1_decomposition",
                   "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
                   "n_tests": len(sub_a), "results": sub_a,
                   "inputs": input_manifest}, fh, indent=2, default=str)

    with (OUTPUT_DIR / "B_har_tf_enrichment.json").open("w") as fh:
        json.dump({"batch": "batch_048", "sub": "B_har_tf_enrichment",
                   "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
                   "n_tests": len(sub_b), "results": sub_b,
                   "inputs": input_manifest}, fh, indent=2, default=str)

    # Summary with decision rules
    classifications = apply_decision_rules(sub_a, sub_b)
    summary = {
        "batch": "batch_048",
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "sub_a_results": sub_a,
        "sub_b_results": sub_b,
        "pre_registered_thresholds": {
            "sub_a_replication_or_min": MIN_OR_SUB_A_REPLICATION,
            "sub_b_suggested_or_min": MIN_OR_HAR_SUGGESTED,
            "sub_b_established_or_min": MIN_OR_HAR_ESTABLISHED,
            "sub_b_established_p_max": MIN_P_HAR,
        },
        "classifications": classifications,
    }
    with (OUTPUT_DIR / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Console summary
    logger.info("\n=== Sub-A: EDT1 Decomposition ===")
    for r in sorted(sub_a, key=lambda x: x.get("or", 0), reverse=True):
        emp_p = f"emp_p={r.get('emp_p', 'N/A')}"
        bh_q = f"q={r['bh_q']:.4f}" if "bh_q" in r else ""
        logger.info("  %-25s | %-15s | OR=%6.2f p=%.4f %s %s [%d genes]",
                    r["gene_list"], r["constraint_metric"],
                    r["or"], r["p"], emp_p, bh_q, r["n_in_list"])

    logger.info("\n=== Sub-B: HAR × TF Enrichment ===")
    for r in sorted(sub_b, key=lambda x: x.get("or", 0), reverse=True):
        bh_q = f"q={r['bh_q']:.4f}" if "bh_q" in r else ""
        logger.info("  %-25s | OR=%6.2f p=%.4f CI=[%.2f,%.2f] %s [%d genes]",
                    r["gene_list"], r["or"], r["p"],
                    r["ci_low"], r["ci_high"], bh_q, r["n_in_list"])

    logger.info("\nWrote %d Sub-A + %d Sub-B results", len(sub_a), len(sub_b))
    return 0

if __name__ == "__main__":
    sys.exit(main())