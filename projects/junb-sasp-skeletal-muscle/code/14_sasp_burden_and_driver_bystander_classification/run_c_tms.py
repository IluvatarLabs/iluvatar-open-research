"""
Batch 063 Analysis C — TMS cross-species descriptive AP-1 polarity replication.

Design: DESCRIPTIVE directional check (not confirmatory).
  - AP-1 family ONLY (CEBP dropped per Critic 1 R-3).
  - Pseudobulk per (mouse x age_bin x compartment). >=50 cells per pseudobulk.
  - For each (compartment x TF): Spearman rho(pseudobulk mean TF expr, pseudobulk mean SASP12 score).
  - Classify direction vs human HLMA expected polarity (EC +, MuSC +, FAP 0/-).
  - Replication call: DIRECTIONAL MATCH (>=4/5), DIRECTIONAL MISMATCH (<=2/5), AMBIGUOUS (3/5).

Outputs:
  - c_tms_direction_table.csv
  - c_tms_summary.json
  - c_tms.log

WHY log-normalized X: matches iter 054/055 human analysis which used scanpy-normalized .X
for both AP-1 TF expression and sc.tl.score_genes (score_genes requires normalized data).
Mixing raw counts for TFs and normalized for SASP score would create an apples-vs-oranges
correlation. We use .X (log-normalized) throughout.

WHY >=50 cells/pseudobulk: dropout dominates below that; pseudobulk mean becomes noisy
(brief Analysis C UNINTERPRETABLE criterion).

WHY Spearman: matches human HLMA rho methodology (iter 054/055).

WHY Fisher-z 95% CI: standard large-sample CI for Spearman; at N=12-16 the CI is wide,
which is THE POINT — we report descriptive magnitude + uncertainty.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
OUT_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_063")
DATA_FP = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/data/TMS_muscle/TMS_Droplet_Limb_Muscle.h5ad")
LOG_FP = OUT_DIR / "c_tms.log"
TABLE_FP = OUT_DIR / "c_tms_direction_table.csv"
SUMMARY_FP = OUT_DIR / "c_tms_summary.json"

MIN_CELLS_PER_PSEUDOBULK = 50
SEED = 1  # for sc.tl.score_genes (rigor.seeds=1 per marvin.yaml)

# AP-1 family — mouse symbols
AP1_PANEL = ["Junb", "Fos", "Fosb", "Jund", "Fosl1"]

# Mouse SASP panel: human SASP12 mapped to mouse orthologs
#  - CXCL8/IL8: NO mouse ortholog (functional analogs are Cxcl1/Cxcl2 which are already in panel)
#  - MMP1: no 1:1 mouse ortholog; paralogs Mmp1a and Mmp1b (include BOTH)
#  - CXCL6: check availability; if missing, drop and document
#  - All others: direct 1:1 ortholog (lowercase)
MOUSE_SASP_CANDIDATES = [
    "Ccl2", "Cxcl1", "Cxcl2", "Cxcl3", "Cxcl6",
    "Il6", "Serpine1", "Mmp1a", "Mmp1b", "Mmp3", "Plau", "Plaur",
]

# Human HLMA expected polarity per compartment (from iter 054/055 AUCell rho results)
# For each compartment, expected sign of rho(AP-1 TF, SASP12 score) across donors.
# EC/Vascular: positive (human Vascular AUCell JUNB rho = +0.923)
# MuSC: positive (human MuSC AUCell JUNB rho = +0.885)
# FAP-equivalent (mesenchymal stem cell): negative/near-zero (human FAP AUCell JUNB rho = -0.394)
# SMC: no human-compartment-specific prediction (report descriptively; no expected sign set)
# Macrophage: human Immune showed OPPOSITE direction per MEMORY finding_cell_type_specificity;
#   we set expected sign to NEGATIVE for macrophage so a positive mouse result = mismatch.
EXPECTED_HUMAN_SIGN = {
    "endothelial cell": "+",
    "skeletal muscle satellite cell": "+",
    "mesenchymal stem cell": "-",
    "smooth muscle cell": None,       # no pre-registered expectation
    "macrophage": "-",                # human immune direction was opposite to FAP/EC
}

# Compartments to analyse — mapping to report-friendly labels
COMPARTMENTS = {
    "endothelial cell": "EC",
    "mesenchymal stem cell": "FAP_equivalent",
    "skeletal muscle satellite cell": "MuSC",
    "smooth muscle cell": "SMC",
    "macrophage": "Macrophage",
}


# ----------------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    LOG_FP.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_c_tms")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_FP, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def fisher_z_ci(rho: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% Fisher-z CI for a Spearman rho.

    WHY: Spearman rho does not have a closed-form CI but the Fisher z-transform
    is the standard approximation used widely (and in iter 054/055 for human HLMA).
    Requires n >= 4.
    """
    if n < 4 or not np.isfinite(rho) or abs(rho) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return (float(lo), float(hi))


def sign_of(rho: float) -> str:
    if not np.isfinite(rho):
        return "NA"
    if abs(rho) < 0.1:
        return "~0"
    return "+" if rho > 0 else "-"


def matches_expected(observed_sign: str, expected: str | None) -> str | None:
    """Return match classification vs human expectation.

    - None expected (SMC) -> None (not counted).
    - observed '~0' vs expected '+' -> NOT MATCH (we only match if sign agrees).
    - observed '~0' vs expected '-' -> MATCH (the FAP prediction is 'negative OR near-zero').
      The brief says FAP-equivalent expected "negative or near-zero". So when expected is '-',
      we count both '-' and '~0' as match.
    """
    if expected is None:
        return None
    if observed_sign == "NA":
        return None
    if expected == "+":
        return observed_sign == "+"
    if expected == "-":
        return observed_sign in ("-", "~0")
    return None


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main() -> int:
    logger = setup_logger()
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("Batch 063 Analysis C — TMS cross-species AP-1 polarity replication")
    logger.info("=" * 80)
    logger.info("Data: %s", DATA_FP)
    logger.info("Min cells per pseudobulk: %d", MIN_CELLS_PER_PSEUDOBULK)
    logger.info("AP-1 panel (mouse): %s", AP1_PANEL)

    # ------------------------------------------------------------------
    # Load — keep X as-is (log-normalized per iter 054/055 convention)
    # ------------------------------------------------------------------
    logger.info("Loading h5ad...")
    adata = sc.read_h5ad(str(DATA_FP))
    logger.info("Loaded: shape=%s, %d var, %d obs", adata.shape, adata.n_vars, adata.n_obs)

    # Document X nature
    sample_x = adata.X[:200].toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X[:200])
    logger.info(
        "X diagnostics (first 200 rows): min=%.3f max=%.3f mean=%.3f; row_sum mean=%.1f",
        float(sample_x.min()), float(sample_x.max()), float(sample_x.mean()),
        float(sample_x.sum(axis=1).mean()),
    )
    logger.info("USING .X (log-normalized) for TF expression and SASP score_genes — matches iter 054/055.")

    # ------------------------------------------------------------------
    # Gene availability (fail-fast and document actually-used panel)
    # ------------------------------------------------------------------
    var_set = set(adata.var_names)
    ap1_present = [g for g in AP1_PANEL if g in var_set]
    ap1_missing = [g for g in AP1_PANEL if g not in var_set]
    sasp_present = [g for g in MOUSE_SASP_CANDIDATES if g in var_set]
    sasp_missing = [g for g in MOUSE_SASP_CANDIDATES if g not in var_set]
    logger.info("AP-1 present: %s", ap1_present)
    if ap1_missing:
        logger.warning("AP-1 MISSING (excluded): %s", ap1_missing)
    logger.info("SASP present: %s", sasp_present)
    if sasp_missing:
        logger.info("SASP MISSING (dropped): %s", sasp_missing)
    if len(sasp_present) < 6:
        logger.error("Too few SASP genes (%d < 6); abort.", len(sasp_present))
        return 2

    # ------------------------------------------------------------------
    # Required obs columns
    # ------------------------------------------------------------------
    required = ["cell_ontology_class", "age", "mouse.id"]
    for c in required:
        if c not in adata.obs.columns:
            logger.error("Missing required obs column: %s", c)
            return 2

    # Restrict to target compartments to save memory/time for score_genes
    keep_mask = adata.obs["cell_ontology_class"].isin(list(COMPARTMENTS.keys()))
    logger.info("Cells in target compartments: %d / %d", int(keep_mask.sum()), adata.n_obs)
    adata = adata[keep_mask].copy()

    # ------------------------------------------------------------------
    # SASP score (score_genes over .X) — per-cell, then pseudobulk-averaged
    # ------------------------------------------------------------------
    logger.info("Computing SASP score via sc.tl.score_genes (n_bins=25 default, seed=%d)", SEED)
    sc.tl.score_genes(
        adata,
        gene_list=sasp_present,
        score_name="SASP_score",
        random_state=SEED,
        n_bins=25,
    )
    logger.info(
        "SASP score stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(adata.obs["SASP_score"].mean()),
        float(adata.obs["SASP_score"].std()),
        float(adata.obs["SASP_score"].min()),
        float(adata.obs["SASP_score"].max()),
    )

    # ------------------------------------------------------------------
    # Extract AP-1 TF expression per cell (dense small matrix)
    # ------------------------------------------------------------------
    tf_idx = [adata.var_names.get_loc(g) for g in ap1_present]
    Xtf = adata.X[:, tf_idx]
    if hasattr(Xtf, "toarray"):
        Xtf = Xtf.toarray()
    Xtf = np.asarray(Xtf, dtype=np.float32)
    tf_df = pd.DataFrame(Xtf, columns=ap1_present, index=adata.obs_names)

    # Keep only the columns needed for pseudobulk
    obs_slim = adata.obs[["cell_ontology_class", "age", "mouse.id", "SASP_score"]].copy()
    obs_slim["cell_ontology_class"] = obs_slim["cell_ontology_class"].astype(str)
    obs_slim["age"] = obs_slim["age"].astype(str)
    obs_slim["mouse.id"] = obs_slim["mouse.id"].astype(str)
    full = pd.concat([obs_slim, tf_df], axis=1)

    # ------------------------------------------------------------------
    # Pseudobulk: one row per (compartment, mouse, age)
    # ------------------------------------------------------------------
    logger.info("Building pseudobulks per (compartment x mouse x age)...")
    group_cols = ["cell_ontology_class", "mouse.id", "age"]
    agg_cols = {c: "mean" for c in ap1_present + ["SASP_score"]}
    # Count cells per pseudobulk
    counts = full.groupby(group_cols, observed=True).size().rename("n_cells").reset_index()
    means = full.groupby(group_cols, observed=True).agg(agg_cols).reset_index()
    pb = means.merge(counts, on=group_cols, how="left")
    logger.info("Total pseudobulks (pre-filter): %d", len(pb))
    pb_pass = pb[pb["n_cells"] >= MIN_CELLS_PER_PSEUDOBULK].copy()
    logger.info(
        "Pseudobulks with n_cells >= %d: %d (dropped %d for low cell count)",
        MIN_CELLS_PER_PSEUDOBULK, len(pb_pass), len(pb) - len(pb_pass),
    )

    # Report pseudobulk distribution
    logger.info("Pseudobulks per compartment (post-filter):")
    for comp, sub in pb_pass.groupby("cell_ontology_class", observed=True):
        logger.info(
            "  %-35s n_pseudobulks=%2d  n_mice=%2d  ages=%s  mean_cells/pb=%.0f",
            comp, len(sub), sub["mouse.id"].nunique(),
            sorted(sub["age"].unique().tolist()),
            float(sub["n_cells"].mean()),
        )

    # ------------------------------------------------------------------
    # Spearman correlations per (compartment x TF) over pseudobulks
    # ------------------------------------------------------------------
    logger.info("Computing Spearman(TF mean expr, SASP score) per (compartment x TF)...")
    rows = []
    for comp_label, comp_short in COMPARTMENTS.items():
        sub = pb_pass[pb_pass["cell_ontology_class"] == comp_label]
        n = len(sub)
        mean_cells = float(sub["n_cells"].mean()) if n else float("nan")
        expected_sign = EXPECTED_HUMAN_SIGN[comp_label]
        for tf in ap1_present:
            if n < 4:
                rho, p = float("nan"), float("nan")
                ci_lo, ci_hi = float("nan"), float("nan")
                observed_sign = "NA"
                match = None
            else:
                rho_p = stats.spearmanr(sub[tf].values, sub["SASP_score"].values)
                rho = float(rho_p.statistic)
                p = float(rho_p.pvalue)
                ci_lo, ci_hi = fisher_z_ci(rho, n)
                observed_sign = sign_of(rho)
                match = matches_expected(observed_sign, expected_sign)
            rows.append({
                "compartment_ontology": comp_label,
                "compartment_short": comp_short,
                "TF": tf,
                "n_pseudobulks": n,
                "mean_cells_per_pb": mean_cells,
                "rho_spearman": rho,
                "p_value_descriptive": p,
                "CI95_lo_fisher_z": ci_lo,
                "CI95_hi_fisher_z": ci_hi,
                "direction_observed": observed_sign,
                "direction_expected_human": expected_sign,
                "match_with_human": match,
            })
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_FP, index=False)
    logger.info("Wrote direction table: %s (%d rows)", TABLE_FP, len(table))

    # ------------------------------------------------------------------
    # Per-compartment classification
    # ------------------------------------------------------------------
    logger.info("Classifying replication per compartment...")
    summary = {
        "analysis": "batch_063 Analysis C — TMS cross-species AP-1 polarity replication",
        "framing": "DESCRIPTIVE directional only; NOT confirmatory. Power at N=12-16 pseudobulks for rho=0.5 is ~45-60% (Fisher-z, alpha=0.05 two-sided). Do not over-interpret.",
        "data": str(DATA_FP),
        "min_cells_per_pseudobulk": MIN_CELLS_PER_PSEUDOBULK,
        "ap1_panel_used": ap1_present,
        "ap1_missing": ap1_missing,
        "mouse_sasp_panel_used": sasp_present,
        "mouse_sasp_missing": sasp_missing,
        "notes_on_sasp_mapping": {
            "CXCL8/IL8": "no mouse ortholog; functional analogs Cxcl1/Cxcl2 already in panel",
            "MMP1": "no 1:1 mouse ortholog; paralogs Mmp1a + Mmp1b both included",
            "CXCL6": "mouse gene not in TMS var_names (dropped)" if "Cxcl6" not in sasp_present else "present",
        },
        "compartments": {},
        "power_disclaimer": (
            "At N=12-16 pseudobulks the Fisher-z 95% CI on a point estimate rho=0.5 has lower bound "
            "~-0.07 (N=12) to ~0.00 (N=16), i.e. CIs will typically cross zero. The classification "
            "reported here is SIGN-BASED directional only. Per Analysis C decision rule: DIRECTIONAL "
            "MATCH if >=4/5 AP-1 TFs match human sign expectation; DIRECTIONAL MISMATCH if <=2/5; "
            "AMBIGUOUS if 3/5. We do NOT claim statistical replication."
        ),
    }

    for comp_label, comp_short in COMPARTMENTS.items():
        sub = table[table["compartment_ontology"] == comp_label].copy()
        expected_sign = EXPECTED_HUMAN_SIGN[comp_label]
        n_pb_list = sub["n_pseudobulks"].unique().tolist()
        n_pb = int(n_pb_list[0]) if n_pb_list else 0

        if expected_sign is None:
            # SMC — no pre-registered direction; report descriptively
            rhos = sub.set_index("TF")["rho_spearman"].to_dict()
            summary["compartments"][comp_short] = {
                "ontology_label": comp_label,
                "n_pseudobulks": n_pb,
                "expected_human_direction": None,
                "rho_per_TF": {k: (None if not np.isfinite(v) else float(v)) for k, v in rhos.items()},
                "direction_per_TF": sub.set_index("TF")["direction_observed"].to_dict(),
                "classification": "NO_PREDICTION (no pre-registered human expectation for SMC)",
                "n_matches": None,
                "n_tested": None,
            }
            continue

        # Count sign matches
        matches = sub["match_with_human"].tolist()
        n_tested = sum(1 for m in matches if m is not None)
        n_match = sum(1 for m in matches if m is True)

        if n_tested == 0:
            classification = "INSUFFICIENT_DATA"
        elif n_match >= 4:
            classification = "DIRECTIONAL_MATCH"
        elif n_match <= 2:
            classification = "DIRECTIONAL_MISMATCH"
        else:
            classification = "AMBIGUOUS"

        rhos = sub.set_index("TF")["rho_spearman"].to_dict()
        directions = sub.set_index("TF")["direction_observed"].to_dict()
        match_map = {r["TF"]: r["match_with_human"] for _, r in sub.iterrows()}

        summary["compartments"][comp_short] = {
            "ontology_label": comp_label,
            "n_pseudobulks": n_pb,
            "expected_human_direction": expected_sign,
            "rho_per_TF": {k: (None if not np.isfinite(v) else float(v)) for k, v in rhos.items()},
            "direction_per_TF": directions,
            "match_per_TF": match_map,
            "n_matches": int(n_match),
            "n_tested": int(n_tested),
            "classification": classification,
        }

        logger.info(
            "  %-35s N=%2d  matches=%d/%d  -> %s",
            comp_label, n_pb, n_match, n_tested, classification,
        )

    # Paper-level headline
    ec = summary["compartments"].get("EC", {})
    musc = summary["compartments"].get("MuSC", {})
    fap = summary["compartments"].get("FAP_equivalent", {})
    headline_parts = []
    for short, s in [("EC", ec), ("MuSC", musc), ("FAP_equivalent", fap)]:
        headline_parts.append(f"{short}={s.get('classification', 'NA')} ({s.get('n_matches', 0)}/{s.get('n_tested', 0)})")
    summary["headline"] = "; ".join(headline_parts)

    # Publishable binary call per brief §Analysis C decision rule
    ec_cls = ec.get("classification", "")
    musc_cls = musc.get("classification", "")
    fap_cls = fap.get("classification", "")
    if ec_cls == "DIRECTIONAL_MATCH" and musc_cls == "DIRECTIONAL_MATCH":
        paper_call = "DIRECTIONAL_REPLICATION_SUPPORTED: >=4/5 AP-1 TFs match human polarity in EC and MuSC."
    elif ec_cls == "DIRECTIONAL_MISMATCH" or musc_cls == "DIRECTIONAL_MISMATCH":
        paper_call = "DIRECTIONAL_REPLICATION_FAILS: human-specific (mismatch in EC or MuSC)."
    else:
        paper_call = f"DIRECTIONAL_REPLICATION_INCONCLUSIVE: EC={ec_cls}, MuSC={musc_cls}, FAP_equivalent={fap_cls}."
    summary["paper_call"] = paper_call

    logger.info("HEADLINE: %s", summary["headline"])
    logger.info("PAPER CALL: %s", paper_call)

    with open(SUMMARY_FP, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote summary: %s", SUMMARY_FP)

    logger.info("Total runtime: %.1fs", time.time() - t0)
    logger.info("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
