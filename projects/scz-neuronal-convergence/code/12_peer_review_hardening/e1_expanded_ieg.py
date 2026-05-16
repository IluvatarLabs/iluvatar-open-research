#!/usr/bin/env python3
"""batch_061 E1 — Expanded IEG ARG gene-set x 8-disorder battery.

Tests whether SCZ polygenic risk is enriched in activity-regulated genes (ARGs)
from Tyssowski et al. 2018 Neuron [lit_doi_10.1016_j.neuron.2018.04.001] using
MAGMA competitive gene-set analysis across 8 disorders.

Gene sets tested:
  all_ARG:  all mapped ARGs (rPRG + dPRG + SRG)
  rPRG:     rapid primary response genes only
  dPRG:     delayed primary response genes only
  SRG:      secondary response genes only
  all_PRG:  rPRG + dPRG combined

Per gene set:
  1. Mouse->human ortholog mapping via uppercase + known ortholog table.
  2. Sub-A v2.1 battery per disorder (8 disorders):
     - OLS beta + 95% CI (PRIMARY estimator)
     - HuberT beta + 95% CI
     - TukeyBiweight beta + 95% CI
     - Rank-MAGMA OLS (sign concordance)
  3. BH-FDR across 8 disorders (one family per gene set).
  4. Size-matched random gene set null (1000 draws) for q < 0.05 hits.
  5. Estimator concordance for SCZ (all 4 estimators).

WHY this script uses canonical Tyssowski 2018 gene lists from the txt files
rather than the batch_060 IEG_RPRGS list: the batch_060 E4 list was a hybrid
that included GADD45B (borderline in Tyssowski, reclassified by batch_060) and
excluded IER2/KLF4/MAFF/AMIGO3. Per critic review, this script uses the exact
Tyssowski Table S3 classifications.

WHY these 5 gene sets and not 4: all_PRG (rPRG + dPRG combined) tests whether
the primary-response program as a whole shows enrichment beyond just the
canonical rPRGs. This is the key test of temporal differentiation that
brief_v2 describes as exploratory.

Output:
  experiments/batch_061/output/e1/results.json

Cardinal rules:
  - Rule 0: No fabrication. Unmapped genes logged, never silently dropped.
  - Rule 1: Import Sub-A v2.1 battery functions from batch_058, not reimplemented.
  - Rule 5: Every decision documented with WHY.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Imports from batch_060/_common (which re-exports batch_059/058/057/056).
# WHY importlib instead of sys.path: all _common.py files share the same
# filename; sys.path would shadow this script's _common imports. importlib
# loads under distinct module names.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")

_B060_COMMON_PATH = PROJECT_ROOT / "experiments" / "batch_060" / "scripts" / "_common.py"
_spec060 = _ilu.spec_from_file_location("batch060_common", str(_B060_COMMON_PATH))
_b060 = _ilu.module_from_spec(_spec060)  # type: ignore[arg-type]
assert _spec060 is not None and _spec060.loader is not None
_spec060.loader.exec_module(_b060)

# Load batch_058 sub_a_robust_battery for fit_ols, fit_huber, etc.
# WHY: Cardinal Rule 1. These functions are parameterized by indicator column
# and are reusable for arbitrary gene-set indicators.
_B058_SUBA_PATH = (
    PROJECT_ROOT / "experiments" / "batch_058" / "scripts" / "sub_a_robust_battery.py"
)
_spec058suba = _ilu.spec_from_file_location("batch058_sub_a", str(_B058_SUBA_PATH))
_b058_suba = _ilu.module_from_spec(_spec058suba)  # type: ignore[arg-type]
assert _spec058suba is not None and _spec058suba.loader is not None
_spec058suba.loader.exec_module(_b058_suba)

# ---------------------------------------------------------------------------
# Re-bind names for clarity and auditor traceability.
# ---------------------------------------------------------------------------
# From batch_060 _common (transitive):
GENE_ANNOT = _b060.GENE_ANNOT
GNOMAD_TSV = _b060.GNOMAD_TSV
DISORDERS = _b060.DISORDERS
PSYCHIATRIC = _b060.PSYCHIATRIC
BH_Q = _b060.BH_Q
DFBETAS_CUTOFF = _b060.DFBETAS_CUTOFF
B3_GENES = _b060.B3_GENES
REPRO_R1_SUB_A_LO = _b060.REPRO_R1_SUB_A_LO
REPRO_R1_SUB_A_HI = _b060.REPRO_R1_SUB_A_HI

load_gene_annot = _b060.load_gene_annot
load_gnomad_per_brief_v2 = _b060.load_gnomad_per_brief_v2
build_sub_a_frame = _b060.build_sub_a_frame
bh_fdr = _b060.bh_fdr
atomic_write_json = _b060.atomic_write_json
sha256_file = _b060.sha256_file
rank_gaussianize = _b060.rank_gaussianize
fit_tukey_biweight = _b060.fit_tukey_biweight

# From batch_058 sub_a_robust_battery:
fit_ols = _b058_suba.fit_ols
fit_huber = _b058_suba.fit_huber
fit_rank_magma_ols = _b058_suba.fit_rank_magma_ols

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_061"
OUTPUT_DIR = BATCH_DIR / "output"
LOGS_DIR = BATCH_DIR / "logs"

# Tyssowski 2018 gene lists.
TYSSOWSKI_DIR = PROJECT_ROOT / "data" / "tyssowski_2018"
TYSSOWSKI_RPRG = TYSSOWSKI_DIR / "tyssowski2018_rPRG.txt"
TYSSOWSKI_DPRG = TYSSOWSKI_DIR / "tyssowski2018_dPRG.txt"
TYSSOWSKI_SRG = TYSSOWSKI_DIR / "tyssowski2018_SRG.txt"
TYSSOWSKI_ALL_ARGS = TYSSOWSKI_DIR / "tyssowski2018_all_ARGs.txt"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Design.yaml seed. WHY 20260424: matches batch_060 master seed for
# reproducibility lineage.
SEED = 20260424

# Covariates: same as Sub-A v2.1 (brief_v2).
# WHY these 4: established in batch_058 as the standard MAGMA competitive
# gene-set covariate vector. log10_gene_length controls for gene-length
# confound, lof_pLI for constraint, log10_exp_lof_plus1 for expected LoF
# density, log10_NSNPS_plus1 for SNP density.
PRIMARY_COVS = [
    "log10_gene_length", "lof_pLI",
    "log10_exp_lof_plus1", "log10_NSNPS_plus1",
]
LOEUF_COVS = [
    "log10_gene_length", "lof_oe_ci_upper",
    "log10_exp_lof_plus1", "log10_NSNPS_plus1",
]

MIN_N_UNIVERSE = 15000  # minimum gene universe size for interpretable result

# Size-matched null: 1000 draws (brief_v2 E1 specifies 1000, not 10000).
# WHY 1000 not 10000: brief_v2 E1 MEASUREMENT section says "Size-matched null
# (1000 random sets of matched size)". 10000 would take ~10x longer for
# marginal precision gain on the empirical percentile.
SIZE_MATCHED_N_DRAWS = 1000
SIZE_MATCHED_N_DECILES = 10

# Reproduction gate range (brief_v2 shared design elements).
REPRO_GATE_LO = 2.5
REPRO_GATE_HI = 3.5

# UNINTERPRETABLE gate: brief_v2 E1 says "Ortholog mapping yields < 80 human
# genes (insufficient power)." We apply this to all_ARG. For sub-sets, a
# proportional floor is used.
MIN_ALL_ARG_GENES = 80


# ===========================================================================
# Mouse-to-human ortholog mapping
# ===========================================================================

# Known non-trivial mouse->human ortholog mappings that differ from simple
# uppercase conversion. Source: Ensembl BioMart GRCh37/GRCm38 1:1 orthologs,
# cross-checked against NCBI Gene.
#
# WHY a hardcoded table rather than runtime BioMart query: (1) reproducibility
# -- the mapping is deterministic and auditable; (2) offline operation --
# no network dependency during experiment execution; (3) gene_annot_jun10.txt
# uses older gene names for some genes (e.g., WDR96 instead of CFAP43).
KNOWN_ORTHOLOGS: dict[str, str | None] = {
    # Mouse symbol: human symbol in gene_annot_jun10.txt (or None if no mapping)
    #
    # Renamed genes where mouse and human symbols differ:
    "Cfap43": "WDR96",      # Cilia and flagella associated protein 43; gene_annot uses WDR96
    "Glt28d2": "COLGALT2",  # Collagen beta(1-O)galactosyltransferase 2
    "Rab39": "RAB39A",      # Mouse Rab39 is 1:1 ortholog of human RAB39A
    "Kitl": "KITLG",        # Kit ligand (mouse "Kitl" vs human "KITLG")
    #
    # RIKEN clones: no 1:1 human ortholog in Ensembl BioMart.
    "4931440P22Rik": None,
    "5430416O09Rik": None,
    "9430020K01Rik": None,
    "D16Ertd472e": None,
    #
    # X-linked genes: gene_annot_jun10.txt contains only autosomal genes (chr1-22).
    # These have valid human orthologs but cannot be mapped to ENSGIDs in our
    # annotation file, so they are excluded from the analysis.
    "Acsl4": None,     # ACSL4, Xq22.3
    "Klhl4": None,     # KLHL4, Xq21.3
    "Nefl": None,      # NEFL, chr8p21.2 -- absent from this gene_annot build
    "Rnf128": None,    # RNF128/GRAIL, Xq22.3
    "Slitrk4": None,   # SLITRK4, Xq27.3
    "Tbc1d8b": None,   # TBC1D8B, Xq25-26
    #
    # No clear 1:1 human ortholog in gene_annot:
    "Ccdc184": None,   # CCDC184/CFAP65 not in gene_annot_jun10.txt
}


def load_tyssowski_gene_list(filepath: Path) -> list[str]:
    """Load a Tyssowski 2018 gene list from a text file (one gene per line).

    WHY strip and skip blanks: the text files have no header and may have
    trailing whitespace or empty lines.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Tyssowski gene list not found: {filepath}")
    genes = []
    with open(filepath) as f:
        for line in f:
            gene = line.strip()
            if gene:
                genes.append(gene)
    return genes


def map_mouse_to_human(
    mouse_genes: list[str],
    annot: pd.DataFrame,
    logger: logging.Logger,
    set_name: str,
) -> tuple[set[str], dict[str, str], list[str], list[str]]:
    """Map mouse gene symbols to human ENSGIDs.

    Strategy (per brief_v2 E1 ortholog mapping procedure):
      1. Check KNOWN_ORTHOLOGS table for non-trivial mappings.
      2. If not in KNOWN_ORTHOLOGS, uppercase the mouse symbol and look up
         in gene_annot_jun10.txt NAME column.
      3. Genes with no mapping are documented (never silently dropped).

    Returns:
      ensg_set: set of mapped ENSGIDs
      sym_to_ensg: dict mapping mouse_symbol -> ENSGID
      unmapped: list of mouse symbols with no human ENSGID
      mapping_details: list of strings documenting each mapping for audit

    WHY gene_annot (not MAGMA geneloc): gene_annot_jun10.txt is the canonical
    ENSGID<->symbol bridge used by build_sub_a_frame and all downstream
    loaders. Using MAGMA geneloc would give Entrez IDs requiring a second
    mapping step.
    """
    # Build NAME -> ENSGID lookup from gene_annot.
    # WHY drop_duplicates(keep="first"): matches the convention in the shared
    # symbols_to_ensgids function from batch_057/_common.py.
    name_dedup = (
        annot.drop_duplicates(subset="NAME", keep="first")
        .set_index("NAME")
    )
    annot_names = set(name_dedup.index)

    sym_to_ensg: dict[str, str] = {}
    unmapped: list[str] = []
    mapping_details: list[str] = []

    for mouse_sym in mouse_genes:
        # Step 1: check known orthologs table.
        if mouse_sym in KNOWN_ORTHOLOGS:
            human_sym = KNOWN_ORTHOLOGS[mouse_sym]
            if human_sym is None:
                unmapped.append(mouse_sym)
                mapping_details.append(
                    f"{mouse_sym} -> [no 1:1 human ortholog]"
                )
                continue
            if human_sym in annot_names:
                ensg = str(name_dedup.loc[human_sym, "ENSGID"])
                sym_to_ensg[mouse_sym] = ensg
                mapping_details.append(
                    f"{mouse_sym} -> {human_sym} -> {ensg} (known ortholog)"
                )
                continue
            else:
                unmapped.append(mouse_sym)
                mapping_details.append(
                    f"{mouse_sym} -> {human_sym} (known ortholog, "
                    f"but {human_sym} not in gene_annot)"
                )
                continue

        # Step 2: uppercase conversion.
        human_sym = mouse_sym.upper()
        if human_sym in annot_names:
            ensg = str(name_dedup.loc[human_sym, "ENSGID"])
            sym_to_ensg[mouse_sym] = ensg
            mapping_details.append(
                f"{mouse_sym} -> {human_sym} -> {ensg} (uppercase)"
            )
        else:
            unmapped.append(mouse_sym)
            mapping_details.append(
                f"{mouse_sym} -> {human_sym} (uppercase, not in gene_annot)"
            )

    ensg_set = set(sym_to_ensg.values())

    logger.info(
        "%s: mapped %d/%d mouse genes to human ENSGIDs. Unmapped (%d): %s",
        set_name,
        len(ensg_set),
        len(mouse_genes),
        len(unmapped),
        unmapped if unmapped else "none",
    )

    return ensg_set, sym_to_ensg, unmapped, mapping_details


# ===========================================================================
# Run one disorder through the battery (4 estimators)
# ===========================================================================
def run_disorder_battery(
    disorder: str,
    gnomad: pd.DataFrame,
    annot: pd.DataFrame,
    gene_set_ensg: set[str],
    logger: logging.Logger,
    indicator_col: str = "in_set",
) -> dict:
    """Run OLS + Huber + Tukey + rank-MAGMA for one disorder and gene set.

    WHY not directly call batch_060's run_disorder_battery: that function
    includes DFBETAS/Cook's D computation (via OLSInfluence) which is O(N^2)
    and adds ~60s per disorder. For E1 we have 5 gene sets x 8 disorders = 40
    disorder runs, and DFBETAS is not part of the E1 decision rule. We run
    only the 4 estimators specified by the E1 brief.

    The individual fit functions (fit_ols, fit_huber, fit_rank_magma_ols,
    fit_tukey_biweight) are imported from batch_058 (Cardinal Rule 1).
    """
    try:
        frame = build_sub_a_frame(
            disorder, gnomad, annot, gene_set_ensg,
            gene_set_col=indicator_col,
        )
    except Exception as exc:
        logger.exception("build_sub_a_frame failed for %s", disorder)
        return {"status": "failed", "reason": str(exc)}

    n = len(frame)
    n_in_set = int(frame[indicator_col].sum())
    logger.info("%s: n=%d in_set=%d", disorder, n, n_in_set)

    if n < MIN_N_UNIVERSE:
        return {"status": "failed", "reason": f"n={n} < {MIN_N_UNIVERSE}"}
    # WHY n_in_set >= 5: regression with fewer than 5 indicator=1 observations
    # yields unstable coefficient estimates. 5 is the standard floor used
    # across all batch_058+ gene-set analyses.
    if n_in_set < 5:
        return {"status": "failed", "reason": f"in_set={n_in_set} < 5"}

    ols = fit_ols(frame, indicator_col, PRIMARY_COVS)
    huber = fit_huber(frame, indicator_col, PRIMARY_COVS)
    tukey = fit_tukey_biweight(frame, PRIMARY_COVS, indicator_col)
    rank_ols = fit_rank_magma_ols(frame, indicator_col, PRIMARY_COVS)

    return {
        "status": "ok",
        "n_gene_universe": n,
        "n_set_in_universe": n_in_set,
        "ols": ols,
        "huber": huber,
        "tukey": tukey,
        "rank_magma_ols": rank_ols,
    }


# ===========================================================================
# LOEUF sensitivity for a single disorder + gene set
# ===========================================================================
def run_loeuf_one(
    disorder: str,
    gnomad: pd.DataFrame,
    annot: pd.DataFrame,
    gene_set_ensg: set[str],
    logger: logging.Logger,
    indicator_col: str = "in_set",
) -> dict:
    """OLS + Huber + Tukey with lof_pLI swapped to lof_oe_ci_upper.

    WHY: tests whether enrichment signal is robust to constraint metric
    choice. If signal vanishes with LOEUF, it may be confounded by
    selective constraint rather than reflecting genuine pathway enrichment.
    """
    try:
        frame = build_sub_a_frame(
            disorder, gnomad, annot, gene_set_ensg,
            gene_set_col=indicator_col,
        )
    except Exception as exc:
        logger.exception("LOEUF build failed for %s", disorder)
        return {"status": "failed", "reason": str(exc)}
    frame = frame.dropna(subset=["lof_oe_ci_upper"]).copy()
    n_in_set = int(frame[indicator_col].sum()) if len(frame) else 0
    if frame.empty or n_in_set < 5:
        return {
            "status": "skipped",
            "reason": f"n={len(frame)} in_set={n_in_set}",
        }
    ols = fit_ols(frame, indicator_col, LOEUF_COVS)
    huber = fit_huber(frame, indicator_col, LOEUF_COVS)
    tukey = fit_tukey_biweight(frame, LOEUF_COVS, indicator_col)
    return {
        "status": "ok",
        "ols": ols,
        "huber": huber,
        "tukey": tukey,
        "covariates": LOEUF_COVS,
    }


# ===========================================================================
# Size-matched random gene set null
# ===========================================================================
def size_matched_null(
    frame: pd.DataFrame,
    observed_beta: float,
    gene_set_size: int,
    gene_set_ensg: set[str],
    indicator_col: str,
    logger: logging.Logger,
    seed: int = SEED,
) -> dict:
    """Draw 1000 random gene sets matched on gene-length decile distribution.

    WHY length-decile matching: brief_v2 E1 mandates matching on the
    gene-length decile distribution. Gene length is a known confound in
    MAGMA -- longer genes have more SNPs and mechanically higher Z.
    Matching on length-decile ensures the null distribution controls for
    this confound.

    Algorithm:
      1. Compute gene-length deciles for all genes in the regression frame.
      2. Count how many test-set genes fall in each decile.
      3. For each of 1000 draws, sample the same number of genes from each
         decile (sampling without replacement within each decile).
      4. Run OLS on each draw via np.linalg.lstsq (fast).
      5. Report empirical percentile of observed beta.
    """
    rng = np.random.default_rng(seed)

    frame = frame.copy()
    frame["length_decile"] = pd.qcut(
        frame["log10_gene_length"],
        q=SIZE_MATCHED_N_DECILES,
        labels=False,
        duplicates="drop",
    )

    # Count test-set genes per decile.
    in_set_mask = frame[indicator_col] == 1
    decile_counts = (
        frame.loc[in_set_mask, "length_decile"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # Available genes per decile (excluding test set to avoid self-sampling).
    out_set_mask = ~in_set_mask
    decile_pools: dict[int, np.ndarray] = {}
    for d_val in sorted(decile_counts.keys()):
        pool_idx = frame.index[
            out_set_mask & (frame["length_decile"] == d_val)
        ].to_numpy()
        decile_pools[d_val] = pool_idx
        if len(pool_idx) < decile_counts[d_val]:
            logger.warning(
                "Decile %d: need %d genes but only %d available. "
                "Will sample with replacement for this decile.",
                d_val,
                decile_counts[d_val],
                len(pool_idx),
            )

    # Pre-compute regression components for speed.
    # WHY np.linalg.lstsq: 1000 iterations of sm.OLS would take ~10s
    # (10ms each). lstsq runs in ~0.5ms each (~0.5s total). The results
    # are identical for OLS (same normal equations).
    y = frame["MAGMA_Z"].to_numpy(dtype=float)
    X_cov = frame[PRIMARY_COVS].to_numpy(dtype=float)
    n = len(frame)
    # Layout: [const, indicator, cov1, cov2, ...]. indicator at col 1.
    X_base = np.column_stack([np.ones(n), np.zeros(n), X_cov])

    # Positional index lookup for fast indicator assignment.
    idx_to_pos = {idx: i for i, idx in enumerate(frame.index)}

    null_betas: list[float] = []
    n_failed = 0

    for _ in range(SIZE_MATCHED_N_DRAWS):
        draw_idx: list[int] = []
        for d_val, count in decile_counts.items():
            pool = decile_pools[d_val]
            if len(pool) == 0:
                continue
            replace = len(pool) < count
            sampled = rng.choice(pool, size=count, replace=replace)
            draw_idx.extend(sampled.tolist())

        if len(draw_idx) == 0:
            n_failed += 1
            continue

        X_base[:, 1] = 0.0
        for idx in draw_idx:
            X_base[idx_to_pos[idx], 1] = 1.0

        try:
            params, _, _, _ = np.linalg.lstsq(X_base, y, rcond=None)
            beta = float(params[1])
            if np.isfinite(beta):
                null_betas.append(beta)
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

    if not null_betas:
        return {"status": "failed", "reason": "all draws failed"}

    null_arr = np.array(null_betas)
    # Empirical percentile: fraction of null betas <= observed beta.
    # WHY <=: one-sided test direction matches the MAGMA competitive test
    # (positive beta = enrichment).
    emp_pct = float(np.mean(null_arr <= observed_beta))

    return {
        "status": "ok",
        "observed_beta": float(observed_beta),
        "null_mean": float(np.mean(null_arr)),
        "null_sd": float(np.std(null_arr)),
        "null_median": float(np.median(null_arr)),
        "empirical_percentile": emp_pct,
        "robust": bool(emp_pct > 0.95),
        "n_draws": len(null_betas),
        "n_draws_failed": n_failed,
        "seed": seed,
    }


# ===========================================================================
# Reproduction gate: SCZ B3 beta_OLS
# ===========================================================================
def run_reproduction_gate(
    gnomad: pd.DataFrame,
    annot: pd.DataFrame,
    logger: logging.Logger,
) -> dict:
    """Verify SCZ B3 beta_OLS is in [+2.5, +3.5] (iter_058 anchor 3.24).

    WHY: brief_v2 shared design elements mandate this. If the regression frame
    has drifted (e.g., gnomAD version mismatch, gene_annot change), all results
    are UNINTERPRETABLE.
    """
    annot_by_name = (
        annot.drop_duplicates(subset="NAME", keep="first").set_index("NAME")
    )
    b3_ensg: dict[str, str] = {}
    for s in B3_GENES:
        ensg = annot_by_name["ENSGID"].get(s)
        if ensg is not None:
            b3_ensg[s] = ensg
    b3_ensg_set = set(b3_ensg.values())
    logger.info(
        "Reproduction gate: B3 mapped %d/%d", len(b3_ensg_set), len(B3_GENES)
    )

    frame = build_sub_a_frame("scz", gnomad, annot, b3_ensg_set)
    ols = fit_ols(frame, "in_set", PRIMARY_COVS)
    beta = float(ols["beta_1"]) if ols.get("status") == "ok" else None

    return {
        "b3_n_mapped": len(b3_ensg_set),
        "b3_n_total": len(B3_GENES),
        "scz_b3_beta_ols": beta,
        "target_lo": REPRO_GATE_LO,
        "target_hi": REPRO_GATE_HI,
        "pass": bool(
            beta is not None and REPRO_GATE_LO <= beta <= REPRO_GATE_HI
        ),
    }


# ===========================================================================
# Run a full gene-set battery (all 8 disorders)
# ===========================================================================
def run_gene_set_battery(
    set_name: str,
    gene_set_ensg: set[str],
    gnomad: pd.DataFrame,
    annot: pd.DataFrame,
    logger: logging.Logger,
    disorders: list[str] | None = None,
) -> dict:
    """Run the 4-estimator battery for one gene set across all disorders.

    Returns a dict with per_disorder results, BH q-values, LOEUF
    sensitivity, and size-matched null for any disorder reaching q < 0.05.
    """
    if disorders is None:
        disorders = DISORDERS

    per_disorder: dict[str, dict] = {}
    for d in disorders:
        logger.info("  %s / %s ...", set_name, d)
        per_disorder[d] = run_disorder_battery(
            d, gnomad, annot, gene_set_ensg, logger,
        )

    # BH-FDR across disorders on OLS one-sided p.
    # WHY within each gene-set family: brief_v2 E1 specifies "BH-corrected
    # q-value within each gene-set family (8 disorders per family, 4
    # families)". NOT across all gene sets.
    ok_disorders = [
        d for d in disorders if per_disorder[d].get("status") == "ok"
    ]
    q_by_d: dict[str, float] = {}
    if len(ok_disorders) == len(disorders):
        pvals = [per_disorder[d]["ols"]["p_one_sided"] for d in disorders]
        qvals = bh_fdr(pvals)
        q_by_d = dict(zip(disorders, qvals))
        for d, q in q_by_d.items():
            per_disorder[d]["q_bh"] = q
    else:
        logger.warning(
            "%s: not all disorders OK (%d/%d). BH-FDR on OK subset only.",
            set_name,
            len(ok_disorders),
            len(disorders),
        )
        if ok_disorders:
            pvals = [
                per_disorder[d]["ols"]["p_one_sided"] for d in ok_disorders
            ]
            qvals = bh_fdr(pvals)
            q_by_d = dict(zip(ok_disorders, qvals))
            for d, q in q_by_d.items():
                per_disorder[d]["q_bh"] = q

    # LOEUF sensitivity for disorders reaching q < 0.05.
    loeuf_results: dict[str, dict] = {}
    sig_disorders = [d for d in ok_disorders if q_by_d.get(d, 1.0) < BH_Q]
    for d in sig_disorders:
        logger.info(
            "%s: running LOEUF sensitivity for %s (q=%.4f)",
            set_name,
            d,
            q_by_d[d],
        )
        loeuf_results[d] = run_loeuf_one(
            d, gnomad, annot, gene_set_ensg, logger,
        )

    # Size-matched null for disorders reaching q < 0.05.
    size_matched: dict[str, dict] = {}
    for d in sig_disorders:
        item = per_disorder[d]
        observed_beta = float(item["ols"]["beta_1"])
        logger.info(
            "%s: running size-matched null for %s (beta=%.3f, q=%.4f)",
            set_name,
            d,
            observed_beta,
            q_by_d[d],
        )
        try:
            frame = build_sub_a_frame(d, gnomad, annot, gene_set_ensg)
            sm_result = size_matched_null(
                frame,
                observed_beta,
                len(gene_set_ensg),
                gene_set_ensg,
                "in_set",
                logger,
                seed=SEED,
            )
            size_matched[d] = sm_result
        except Exception as exc:
            logger.exception(
                "Size-matched null failed for %s/%s", set_name, d
            )
            size_matched[d] = {"status": "failed", "reason": str(exc)}

    return {
        "per_disorder": per_disorder,
        "q_by_d": q_by_d,
        "loeuf_sensitivity": loeuf_results,
        "size_matched_null": size_matched,
    }


# ===========================================================================
# Format output for one gene set
# ===========================================================================
def format_per_disorder_output(
    per_disorder: dict, q_by_d: dict
) -> dict:
    """Format per-disorder results for JSON output.

    WHY this schema: matches the batch_060 E4/E5/E6 output format for
    consistency, minus DFBETAS (not computed in E1).
    """
    out: dict[str, dict] = {}
    for d, item in per_disorder.items():
        if item.get("status") != "ok":
            out[d] = {
                "status": item.get("status", "failed"),
                "reason": item.get("reason", "unknown"),
            }
            continue
        ols = item["ols"]
        huber = item.get("huber", {})
        tukey = item.get("tukey", {})
        rank_ols = item.get("rank_magma_ols", {})

        out[d] = {
            "ols_beta": ols.get("beta_1"),
            "ols_se": ols.get("se_1"),
            "ols_ci": [ols.get("ci_lo"), ols.get("ci_hi")],
            "ols_p_one_sided": ols.get("p_one_sided"),
            "ols_p_two_sided": ols.get("p_two_sided"),
            "q_bh": q_by_d.get(d),
            "ols_r_squared": ols.get("r_squared"),
            "ols_n": ols.get("n"),
            "huber_beta": huber.get("beta_1"),
            "huber_ci": (
                [huber.get("ci_lo"), huber.get("ci_hi")]
                if huber.get("status") == "ok"
                else None
            ),
            "tukey_beta": tukey.get("beta_1"),
            "tukey_ci": (
                [tukey.get("ci_lo"), tukey.get("ci_hi")]
                if tukey.get("status") == "ok"
                else None
            ),
            "tukey_converged": tukey.get("converged"),
            "rank_ols_beta": rank_ols.get("beta_1"),
            "rank_sign_concordance": (
                bool(
                    np.sign(ols.get("beta_1", 0))
                    == np.sign(rank_ols.get("beta_1", 0))
                )
                if rank_ols.get("status") == "ok"
                and ols.get("beta_1") is not None
                else None
            ),
            "n_gene_universe": item.get("n_gene_universe"),
            "n_set_in_universe": item.get("n_set_in_universe"),
        }
    return out


# ===========================================================================
# Estimator concordance (all 4 estimators for SCZ)
# ===========================================================================
def compute_estimator_concordance(disorder_result: dict) -> dict:
    """Compute estimator concordance summary for one disorder.

    WHY: brief_v2 E1 output spec requires "Estimator concordance for SCZ
    (all 4 estimators)". This provides a single summary of whether all
    4 estimators agree on sign and approximate magnitude.
    """
    if disorder_result.get("status") != "ok":
        return {"status": "failed", "reason": "disorder not ok"}

    betas = {}
    for name, key in [
        ("ols", "ols"),
        ("huber", "huber"),
        ("tukey", "tukey"),
        ("rank_magma", "rank_magma_ols"),
    ]:
        est = disorder_result.get(key, {})
        if est.get("status") == "ok" and est.get("beta_1") is not None:
            betas[name] = float(est["beta_1"])

    if not betas:
        return {"status": "failed", "reason": "no estimators succeeded"}

    signs = {name: np.sign(b) for name, b in betas.items()}
    all_positive = all(s > 0 for s in signs.values())
    all_negative = all(s < 0 for s in signs.values())
    sign_concordant = all_positive or all_negative

    beta_values = list(betas.values())
    beta_range = max(beta_values) - min(beta_values) if len(beta_values) > 1 else 0.0

    return {
        "status": "ok",
        "betas": betas,
        "sign_concordant": sign_concordant,
        "all_positive": all_positive,
        "beta_range": float(beta_range),
        "n_estimators": len(betas),
    }


# ===========================================================================
# Decision rule (per brief_v2 E1 DECISION RULE)
# ===========================================================================
def classify_e1(
    battery_results: dict[str, dict],
    set_name: str,
    q_by_d_all: dict[str, dict[str, float]],
    logger: logging.Logger,
) -> str:
    """E1 decision rule (brief_v2 E1 DECISION RULE).

    Applied to all_ARG as the primary endpoint:
      IEG_ENRICHED (SUGGESTED): q_SCZ < 0.05 (within-gene-set BH, family=8)
      IEG_SUGGESTIVE: q_SCZ in [0.05, 0.20)
      IEG_NULL: q_SCZ > 0.20

    WHY classify on all_ARG: this is the primary endpoint with the most
    statistical power (~140-160 genes). Sub-set comparisons (rPRG vs dPRG
    vs SRG) are exploratory per brief_v2.
    """
    all_arg_q = q_by_d_all.get("all_ARG", {})
    scz_q = all_arg_q.get("scz", 1.0)

    all_arg_per_d = battery_results.get("all_ARG", {}).get("per_disorder", {})
    scz_result = all_arg_per_d.get("scz", {})

    if scz_result.get("status") != "ok":
        return "UNINTERPRETABLE"

    scz_beta = float(scz_result["ols"]["beta_1"])

    # Check UNINTERPRETABLE conditions from brief_v2:
    # "IBD/Height positive (generic confound)"
    for neg_ctrl in ["ibd_delange2017", "height"]:
        neg_result = all_arg_per_d.get(neg_ctrl, {})
        neg_q = all_arg_q.get(neg_ctrl, 1.0)
        if neg_result.get("status") == "ok":
            neg_beta = float(neg_result["ols"]["beta_1"])
            if neg_beta > 0.30 and neg_q < 0.05:
                logger.warning(
                    "E1 UNINTERPRETABLE: %s beta=%.3f q=%.4f "
                    "(generic confound signal).",
                    neg_ctrl,
                    neg_beta,
                    neg_q,
                )
                return "IEG_GENERIC_CONFOUND"

    # Primary decision on SCZ q-value.
    if scz_q < 0.05:
        logger.info(
            "E1 classification: IEG_ENRICHED (SCZ beta=%.3f, q=%.4f)",
            scz_beta,
            scz_q,
        )
        return "IEG_ENRICHED"
    elif scz_q < 0.20:
        logger.info(
            "E1 classification: IEG_SUGGESTIVE (SCZ beta=%.3f, q=%.4f)",
            scz_beta,
            scz_q,
        )
        return "IEG_SUGGESTIVE"
    else:
        logger.info(
            "E1 classification: IEG_NULL (SCZ beta=%.3f, q=%.4f)",
            scz_beta,
            scz_q,
        )
        return "IEG_NULL"


# ===========================================================================
# Logger factory
# ===========================================================================
def setup_logger(name: str, logfile: Path) -> logging.Logger:
    """Logger emitting to logfile and stdout.

    WHY separate from batch_060: keep batch_061 logs isolated under
    batch_061/logs for auditor traceability.
    """
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


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="batch_061 E1: expanded IEG ARG gene-set x 8-disorder battery",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: SCZ + ADHD only, skip LOEUF and size-matched null.",
    )
    args = parser.parse_args()
    smoke = args.smoke

    logger = setup_logger("batch_061.e1", LOGS_DIR / "e1_expanded_ieg.log")
    t0 = time.time()
    logger.info("=" * 72)
    logger.info("batch_061 E1: expanded IEG ARG gene-set x 8-disorder battery")
    logger.info("smoke=%s seed=%d", smoke, SEED)
    logger.info("=" * 72)

    # Set global seed for reproducibility.
    np.random.seed(SEED)

    # -------------------------------------------------------------------
    # Load shared data (once).
    # -------------------------------------------------------------------
    logger.info("Loading gnomAD constraint metrics...")
    gnomad = load_gnomad_per_brief_v2()
    logger.info("gnomAD: %d genes loaded.", len(gnomad))

    logger.info("Loading gene annotations...")
    annot = load_gene_annot()
    logger.info("gene_annot: %d genes loaded.", len(annot))

    disorders_to_run = DISORDERS if not smoke else ["scz", "adhd"]

    # -------------------------------------------------------------------
    # Reproduction gate (B3 SCZ beta_OLS).
    # WHY first: if the regression frame has drifted, all results are
    # UNINTERPRETABLE. Check before running any gene-set battery.
    # -------------------------------------------------------------------
    logger.info("Running reproduction gate (SCZ B3 beta_OLS)...")
    repro_gate = run_reproduction_gate(gnomad, annot, logger)
    logger.info(
        "Reproduction gate: B3 SCZ beta=%.4f, pass=%s",
        repro_gate["scz_b3_beta_ols"] or float("nan"),
        repro_gate["pass"],
    )
    if not repro_gate["pass"]:
        logger.error(
            "REPRODUCTION GATE FAILED. SCZ B3 beta=%.4f not in [%.1f, %.1f]. "
            "All results UNINTERPRETABLE.",
            repro_gate["scz_b3_beta_ols"] or float("nan"),
            REPRO_GATE_LO,
            REPRO_GATE_HI,
        )
        # Proceed to write results flagged as UNINTERPRETABLE (Rule 0).

    # -------------------------------------------------------------------
    # Load and map Tyssowski 2018 gene lists.
    # WHY load from txt files: these are the canonical Tyssowski Table S3
    # classifications, one gene per line. Using the txt files rather than
    # the JSON avoids any preprocessing artifacts.
    # -------------------------------------------------------------------
    logger.info("Loading Tyssowski 2018 gene lists...")
    mouse_rprg = load_tyssowski_gene_list(TYSSOWSKI_RPRG)
    mouse_dprg = load_tyssowski_gene_list(TYSSOWSKI_DPRG)
    mouse_srg = load_tyssowski_gene_list(TYSSOWSKI_SRG)
    mouse_all_args = load_tyssowski_gene_list(TYSSOWSKI_ALL_ARGS)
    # all_PRG = rPRG + dPRG (combined primary response genes).
    mouse_all_prg = mouse_rprg + mouse_dprg

    logger.info(
        "Mouse gene counts: rPRG=%d, dPRG=%d, SRG=%d, all_ARG=%d, all_PRG=%d",
        len(mouse_rprg),
        len(mouse_dprg),
        len(mouse_srg),
        len(mouse_all_args),
        len(mouse_all_prg),
    )

    # Verify all_ARG = rPRG + dPRG + SRG (sanity check).
    combined_set = set(mouse_rprg) | set(mouse_dprg) | set(mouse_srg)
    all_arg_set = set(mouse_all_args)
    if combined_set != all_arg_set:
        # WHY log but not abort: brief notes some borderline genes may
        # differ. We use the canonical all_ARGs.txt as ground truth.
        diff_in_combined = combined_set - all_arg_set
        diff_in_allarg = all_arg_set - combined_set
        logger.warning(
            "all_ARG != rPRG+dPRG+SRG. "
            "In combined but not all_ARG: %s. "
            "In all_ARG but not combined: %s.",
            diff_in_combined or "none",
            diff_in_allarg or "none",
        )

    # -------------------------------------------------------------------
    # Map mouse genes to human ENSGIDs.
    # -------------------------------------------------------------------
    gene_sets_mouse = {
        "rPRG": mouse_rprg,
        "dPRG": mouse_dprg,
        "SRG": mouse_srg,
        "all_ARG": mouse_all_args,
        "all_PRG": mouse_all_prg,
    }

    gene_sets_human: dict[str, set[str]] = {}
    mapping_info: dict[str, dict] = {}

    for name, mouse_list in gene_sets_mouse.items():
        ensg_set, sym_map, unmapped, details = map_mouse_to_human(
            mouse_list, annot, logger, name,
        )
        gene_sets_human[name] = ensg_set
        mapping_info[name] = {
            "total_mouse": len(mouse_list),
            "mapped_human": len(sym_map),
            "mapped_ensgid": len(ensg_set),
            "unmapped_count": len(unmapped),
            "unmapped_genes": unmapped,
            "mapping_details": details,
            "symbol_to_ensgid": {
                k: v for k, v in sym_map.items()
            },
        }

    # -------------------------------------------------------------------
    # UNINTERPRETABLE gate: all_ARG must have >= 80 mapped genes.
    # WHY 80: brief_v2 E1 says "Ortholog mapping yields < 80 human genes
    # (insufficient power)." At n=80, MDE at alpha=0.05 one-sided is ~0.40
    # which is close to the predicted beta of 0.454.
    # -------------------------------------------------------------------
    all_arg_n = len(gene_sets_human["all_ARG"])
    if all_arg_n < MIN_ALL_ARG_GENES:
        logger.error(
            "UNINTERPRETABLE: all_ARG mapped only %d genes (< %d threshold).",
            all_arg_n,
            MIN_ALL_ARG_GENES,
        )
        # Still proceed to write results, flagged as UNINTERPRETABLE.

    # -------------------------------------------------------------------
    # SHA256 provenance for input files.
    # -------------------------------------------------------------------
    provenance = {
        "gene_annot_sha256": sha256_file(GENE_ANNOT),
        "gnomad_sha256": sha256_file(GNOMAD_TSV),
        "tyssowski_rprg_sha256": sha256_file(TYSSOWSKI_RPRG),
        "tyssowski_dprg_sha256": sha256_file(TYSSOWSKI_DPRG),
        "tyssowski_srg_sha256": sha256_file(TYSSOWSKI_SRG),
        "tyssowski_all_args_sha256": sha256_file(TYSSOWSKI_ALL_ARGS),
    }

    # -------------------------------------------------------------------
    # Run gene-set batteries.
    # -------------------------------------------------------------------
    gene_set_names = ["all_ARG", "rPRG", "dPRG", "SRG", "all_PRG"]
    battery_results: dict[str, dict] = {}
    q_by_d_all: dict[str, dict[str, float]] = {}

    for gs_name in gene_set_names:
        logger.info("=" * 72)
        logger.info("Gene set: %s (n_ensg=%d)", gs_name, len(gene_sets_human[gs_name]))
        logger.info("=" * 72)

        # Skip sub-sets with < 5 genes (would fail the n_in_set floor).
        if len(gene_sets_human[gs_name]) < 5:
            logger.warning(
                "%s: only %d mapped genes, skipping (< 5).",
                gs_name,
                len(gene_sets_human[gs_name]),
            )
            battery_results[gs_name] = {
                "per_disorder": {},
                "q_by_d": {},
                "loeuf_sensitivity": {},
                "size_matched_null": {},
                "skipped": True,
                "reason": f"n_ensg={len(gene_sets_human[gs_name])} < 5",
            }
            q_by_d_all[gs_name] = {}
            continue

        battery = run_gene_set_battery(
            gs_name,
            gene_sets_human[gs_name],
            gnomad,
            annot,
            logger,
            disorders=disorders_to_run,
        )
        battery_results[gs_name] = battery
        q_by_d_all[gs_name] = battery["q_by_d"]

    # -------------------------------------------------------------------
    # Estimator concordance for SCZ (all 4 estimators, per gene set).
    # -------------------------------------------------------------------
    estimator_concordance: dict[str, dict] = {}
    for gs_name in gene_set_names:
        per_d = battery_results[gs_name].get("per_disorder", {})
        scz_result = per_d.get("scz", {})
        estimator_concordance[gs_name] = compute_estimator_concordance(
            scz_result
        )

    # -------------------------------------------------------------------
    # Classification (applied to all_ARG as primary endpoint).
    # -------------------------------------------------------------------
    interpretable = (
        repro_gate["pass"]
        and all_arg_n >= MIN_ALL_ARG_GENES
        and not smoke
    )
    classification = (
        classify_e1(battery_results, "all_ARG", q_by_d_all, logger)
        if interpretable
        else "UNINTERPRETABLE"
    )

    # -------------------------------------------------------------------
    # Temporal differentiation (exploratory).
    # WHY exploratory: brief_v2 says "Power for formal temporal
    # differentiation is <15% (rPRG n=19 is underpowered for q<0.10);
    # do NOT classify as ESTABLISHED even if dPRG > rPRG."
    # -------------------------------------------------------------------
    temporal_comparison = {}
    for gs_a, gs_b in [("rPRG", "dPRG"), ("rPRG", "SRG"), ("dPRG", "SRG")]:
        a_per_d = battery_results.get(gs_a, {}).get("per_disorder", {})
        b_per_d = battery_results.get(gs_b, {}).get("per_disorder", {})
        a_scz = a_per_d.get("scz", {})
        b_scz = b_per_d.get("scz", {})
        if a_scz.get("status") == "ok" and b_scz.get("status") == "ok":
            a_beta = float(a_scz["ols"]["beta_1"])
            b_beta = float(b_scz["ols"]["beta_1"])
            temporal_comparison[f"{gs_a}_vs_{gs_b}"] = {
                f"{gs_a}_scz_beta": a_beta,
                f"{gs_b}_scz_beta": b_beta,
                "difference": a_beta - b_beta,
                "note": "EXPLORATORY — underpowered for formal comparison",
            }

    # -------------------------------------------------------------------
    # Assemble output JSON.
    # -------------------------------------------------------------------
    wall = time.time() - t0

    # Per gene-set results.
    per_gene_set_output: dict[str, dict] = {}
    for gs_name in gene_set_names:
        battery = battery_results[gs_name]
        per_gene_set_output[gs_name] = {
            "ortholog_mapping": {
                k: v
                for k, v in mapping_info[gs_name].items()
                if k != "mapping_details"
                # WHY exclude mapping_details: they are verbose per-gene
                # strings useful for debug but cluttering the JSON. Included
                # separately in ortholog_mapping_details.
            },
            "per_disorder": format_per_disorder_output(
                battery.get("per_disorder", {}),
                battery.get("q_by_d", {}),
            ),
            "bh_q_by_disorder": battery.get("q_by_d", {}),
            "loeuf_sensitivity": battery.get("loeuf_sensitivity", {}),
            "size_matched_null": battery.get("size_matched_null", {}),
            "estimator_concordance_scz": estimator_concordance.get(
                gs_name, {}
            ),
        }

    results = {
        "experiment": "E1",
        "title": "Expanded IEG ARG gene-set x 8-disorder battery",
        "source": (
            "Tyssowski et al. 2018 Neuron 98(3):530-546 "
            "[lit_doi_10.1016_j.neuron.2018.04.001]"
        ),
        "gene_set_names": gene_set_names,
        "per_gene_set": per_gene_set_output,
        "classification": classification,
        "temporal_differentiation": temporal_comparison,
        "reproduction_gate": repro_gate,
        "ortholog_mapping_summary": {
            gs_name: {
                "total_mouse": mapping_info[gs_name]["total_mouse"],
                "mapped_human": mapping_info[gs_name]["mapped_human"],
                "mapped_ensgid": mapping_info[gs_name]["mapped_ensgid"],
                "unmapped_genes": mapping_info[gs_name]["unmapped_genes"],
            }
            for gs_name in gene_set_names
        },
        "ortholog_mapping_details": {
            gs_name: mapping_info[gs_name]["mapping_details"]
            for gs_name in gene_set_names
        },
        "provenance": provenance,
        "disorders_run": disorders_to_run,
        "seed": SEED,
        "smoke": smoke,
        "wall_s": wall,
        "model": (
            "MAGMA_Z ~ in_set + log10(gene_length) + lof_pLI + "
            "log10(exp_lof+1) + log10(NSNPS+1)"
        ),
        "brief_contract": {
            "bh_fdr_family_size": len(disorders_to_run),
            "bh_q_threshold": BH_Q,
            "size_matched_n_draws": SIZE_MATCHED_N_DRAWS,
            "min_all_arg_genes": MIN_ALL_ARG_GENES,
            "repro_gate_range": [REPRO_GATE_LO, REPRO_GATE_HI],
            "decision_rule": (
                "q_SCZ < 0.05 -> IEG_ENRICHED; "
                "q_SCZ in [0.05, 0.20) -> IEG_SUGGESTIVE; "
                "q_SCZ > 0.20 -> IEG_NULL; "
                "IBD/Height q<0.05 with beta>0.30 -> IEG_GENERIC_CONFOUND"
            ),
        },
    }

    # -------------------------------------------------------------------
    # Write output.
    # -------------------------------------------------------------------
    out_path = OUTPUT_DIR / "e1" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(results, out_path)
    logger.info("E1 wrote %s (classification=%s)", out_path, classification)

    # -------------------------------------------------------------------
    # Summary to log.
    # -------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("batch_061 E1 SUMMARY")
    logger.info("=" * 72)
    logger.info("Classification: %s", classification)
    for gs_name in gene_set_names:
        gs_out = per_gene_set_output[gs_name]
        scz_info = gs_out["per_disorder"].get("scz", {})
        if isinstance(scz_info, dict) and "ols_beta" in scz_info:
            logger.info(
                "  %s: SCZ OLS beta=%.3f, p_one=%.4f, q_bh=%.4f, "
                "n_in_set=%s, n_universe=%s",
                gs_name,
                scz_info["ols_beta"],
                scz_info["ols_p_one_sided"],
                scz_info.get("q_bh", float("nan")),
                scz_info.get("n_set_in_universe", "?"),
                scz_info.get("n_gene_universe", "?"),
            )
        else:
            logger.info("  %s: SCZ result: %s", gs_name, scz_info)
    logger.info("Wall time: %.1f s", wall)
    logger.info("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
