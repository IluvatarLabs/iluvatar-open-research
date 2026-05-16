#!/usr/bin/env python3
"""
batch_054_B: F144 jackknife + bootstrap (Sub-A) + 4-construct Koopmans x EDT1
              decomposition (Sub-B).

Sub-A — Leave-one-out jackknife (analytic OR + diagnostic length-perm emp_p)
        and bootstrap (1000 resamples, analytic OR) of the ~48-gene
        ST12_SynGO ∩ EDT1 set (F144_set) against three constraint metrics
        (pLI>=0.9, LOEUF<=0.35, missense_z top decile).

Sub-B — Four Koopmans x EDT1 decomposition constructs:
        B1 = Koopmans ∩ EDT1_all_pc
        B2 = (Koopmans − Hand) ∩ EDT1_all_pc
        B3 = (Koopmans − Hand − EDT1_SynGO) ∩ EDT1_all_pc
        B4 = EDT1_SynGO − Hand
        Each tested vs pLI, LOEUF, missense_z via length-stratified 5000-perm
        test. B4 is replicated across 5 seeds for MC-precision.

Primitives are imported VERBATIM from experiments/batch_048/scripts/run_batch_048.py
(PLI_THRESHOLD, LOEUF_THRESHOLD, RNG_SEED, SYNOGO_EDT1_BATCH047, fisher_enrichment,
length_perm_test, _or_from_masks, load_gnomad, load_pgc3_edt1). Rule 1 —
no re-implementation; every decision point has explicit WHY inline.

Pre-flight provenance (output/provenance.json) gates execution: if the hand-list
pLI reproduction OR does not match F121 (20.9050445), we STOP with an error
before running the expensive jackknife/bootstrap/perm pipeline.

Outputs:
  - experiments/batch_054_B/output/provenance.json
  - experiments/batch_054_B/output/results.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Import primitives verbatim from batch_048. WHY: Rule 1 — do not re-implement
# Fisher / permutation machinery; F121/F144 reproduction requires bit-for-bit
# matching of the batch_048 data loaders and stats functions.
# ---------------------------------------------------------------------------
sys.path.insert(0, "/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_048/scripts")
from run_batch_048 import (  # noqa: E402 — sys.path must be set first
    PLI_THRESHOLD,
    LOEUF_THRESHOLD,
    RNG_SEED,
    SYNOGO_EDT1_BATCH047,
    _or_from_masks,
    fisher_enrichment,
    length_perm_test,
    load_gnomad,
    load_pgc3_edt1,
    GNOMAD_TSV,  # path used by load_gnomad — needed for provenance SHA256
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_054_B"
OUTPUT_DIR = BATCH_DIR / "output"
LOG_DIR = BATCH_DIR / "logs"

SYNGO_GMT = PROJECT_ROOT / "experiments" / "batch_052_A" / "input" / "syngo_2024.gmt"
BATCH_048_SCRIPT = PROJECT_ROOT / "experiments" / "batch_048" / "scripts" / "run_batch_048.py"

# ---------------------------------------------------------------------------
# Hyperparameters (brief §MEASUREMENT). All constants explicit + justified.
# ---------------------------------------------------------------------------
# Sub-A jackknife emp_p uses 1000 perms (diagnostic only; brief §MEASUREMENT).
# WHY 1000 (not 5000): emp_p is a diagnostic, not a decision gate; 48 jackknife
# iterations × 5000 perms = 240k perms is wasteful for a diagnostic number.
N_PERM_JACKKNIFE = 1000

# Sub-A bootstrap: 1000 resamples x analytic Fisher OR (no inner perm). WHY:
# brief §MEASUREMENT — analytic OR is sufficient for OR 95% CI; inner perm
# is optional/time-bound. Seed 20260501 per brief.
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260501

# Sub-B: 5000 perms per cell, RNG_SEED=20260423 (inherited from batch_048 / F121).
# WHY 5000: matches F121/F144 convention; enables apples-to-apples replication
# of F144 via B4 construct (brief §SOURCE, §PREDICTION).
N_PERM_SUBB = 5_000

# B4 MC-precision replicates (brief §MEASUREMENT).
B4_EXTRA_SEEDS = [20260501, 20260502, 20260503, 20260504, 20260505]

# Reproduction target. WHY 20.9050445: this is the exact F121 value computed
# bit-for-bit in iter_053 (docs/summary_053.md, batch_053_A) from the same
# gnomAD v4.1 TSV + SYNOGO_EDT1_BATCH047 constant + batch_048 load_gnomad.
REPRODUCTION_TARGET_OR = 20.9050445
REPRODUCTION_TOL = 1e-4  # absolute OR tolerance; bit-for-bit expected

# BH-FDR alpha for Sub-B. WHY 0.05: conventional; matches batch_048 / batch_053_A.
BH_ALPHA = 0.05

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def _setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_054_B")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        for handler in (
            logging.FileHandler(LOG_DIR / "run_batch_054_B.log"),
            logging.StreamHandler(sys.stdout),
        ):
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
            )
            logger.addHandler(handler)
    return logger


def sha256(path: pathlib.Path) -> str:
    """SHA256 of a file — streaming to support multi-GB gnomAD TSV."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_list(items: list[str]) -> str:
    """SHA256 of a sorted list — for gene-list provenance without disk I/O."""
    h = hashlib.sha256()
    for item in sorted(items):
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Koopmans KB parser (SynGO 2024 GMT format)
# ---------------------------------------------------------------------------
def parse_syngo_gmt(path: pathlib.Path) -> set[str]:
    """Parse SynGO 2024 GMT — take union of all set members.

    GMT format per line: <set_name>\t<description>\t<gene1>\t<gene2>...
    (The 2024 snapshot leaves the description field empty, so gene cols start at 2.)
    Parser logic mirrors batch_052_A/scripts/04_axis_c_functional_tier.py:parse_syngo_gmt
    verbatim — WHY: Rule 1, matches the canonical project parser; the 1,555 unique-gene
    count in the brief §SOURCE derives from this parser.
    """
    genes: set[str] = set()
    with path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            # Fields: [set_name, description, gene1, gene2, ...]. Empty strings
            # filtered out to handle trailing tabs.
            for tok in parts[2:]:
                tok = tok.strip()
                if tok:
                    genes.add(tok)
    return genes


# ---------------------------------------------------------------------------
# Provenance + reproduction gate
# ---------------------------------------------------------------------------
def build_provenance(
    logger: logging.Logger,
    gnomad: pd.DataFrame,
    bg_genes_set: set[str],
    bg_genes_arr: np.ndarray,
    pli_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute SHA256s and run the F121 hand-list pLI reproduction gate.

    Gate logic: the hand list (SYNOGO_EDT1_BATCH047, n=14) intersected with the
    gnomAD bg should give exactly the 11-gene subset whose pLI OR = 20.9050445
    per F121 / iter_053. Any deviation means the imported pipeline or input files
    have drifted — STOP before running expensive tests.
    """
    hand_list = set(SYNOGO_EDT1_BATCH047)
    hand_in_bg = hand_list & bg_genes_set
    target_set = set(bg_genes_arr[pli_mask])
    fisher = fisher_enrichment(hand_in_bg, bg_genes_set, target_set)
    hand_or = float(fisher["or"])
    match = abs(hand_or - REPRODUCTION_TARGET_OR) < REPRODUCTION_TOL

    prov = {
        "sha256": {
            "syngo_2024_gmt": sha256(SYNGO_GMT),
            "gnomad_v4_1_constraint_tsv": sha256(GNOMAD_TSV),
            "batch_048_script": sha256(BATCH_048_SCRIPT),
        },
        "file_paths": {
            "syngo_2024_gmt": str(SYNGO_GMT),
            "gnomad_v4_1_constraint_tsv": str(GNOMAD_TSV),
            "batch_048_script": str(BATCH_048_SCRIPT),
        },
        "reproduction_check": {
            "gene_list": "SYNOGO_EDT1_BATCH047 (hand list, n=14)",
            "n_in_hand_list": len(hand_list),
            "n_hand_in_bg": len(hand_in_bg),
            "hand_list_pLI_OR": hand_or,
            "target_OR": REPRODUCTION_TARGET_OR,
            "absolute_tolerance": REPRODUCTION_TOL,
            "match": bool(match),
            "fisher_detail": {
                "a": fisher["a"], "b": fisher["b"], "c": fisher["c"], "d": fisher["d"],
                "p": float(fisher["p"]),
                "ci_low": float(fisher["ci_low"]),
                "ci_high": float(fisher["ci_high"]),
            },
        },
        "hyperparameters": {
            "PLI_THRESHOLD": PLI_THRESHOLD,
            "LOEUF_THRESHOLD": LOEUF_THRESHOLD,
            "RNG_SEED_subB_primary": RNG_SEED,
            "N_PERM_SUBB": N_PERM_SUBB,
            "N_PERM_JACKKNIFE": N_PERM_JACKKNIFE,
            "N_BOOTSTRAP": N_BOOTSTRAP,
            "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
            "B4_EXTRA_SEEDS": B4_EXTRA_SEEDS,
            "BH_ALPHA": BH_ALPHA,
        },
    }
    logger.info(
        "PROVENANCE hand-list reproduction: OR=%.7f target=%.7f match=%s (n_in_bg=%d)",
        hand_or, REPRODUCTION_TARGET_OR, match, len(hand_in_bg),
    )
    return prov


# ---------------------------------------------------------------------------
# Metric mask builder. WHY a helper: three metrics need the SAME bg-row order
# so masks align across pLI / LOEUF / missense_z. This mirrors batch_053_A.
# ---------------------------------------------------------------------------
def build_metric_masks(bg: pd.DataFrame, gnomad: pd.DataFrame | None = None) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    """Return {metric_name: (mask_over_bg, meta)} — bg = MHC-excluded gnomAD rows.

    WHY MHC exclusion: F121/F144/batch_048 all exclude MHC from the background
    (extended LD / HLA diversity confounds constraint interpretation; see
    Karczewski 2020 + PGC3 SCZ treatment). Matching bg definition is the whole
    point of verbatim import.

    WHY bg passed in (not re-derived): prevents silent mask/bg misalignment if
    this function is ever modified to sort/reorder rows — main() owns bg row
    order and all downstream arrays (bg_genes_arr, lengths) derive from the
    same single bg DataFrame. The `gnomad` parameter is retained for API
    backward-compatibility but is not used.
    """
    pli_mask = bg["pli_ge_09"].astype(bool).to_numpy()
    loeuf_mask = bg["loeuf_lt_035"].astype(bool).to_numpy()
    # missense_z top decile over bg (matches batch_053_A._missense_z_top_decile_mask).
    # WHY top decile (not bottom): higher mis.z_score = fewer observed missense
    # than expected = more constrained (Karczewski 2020). 90th pct of NON-NaN
    # values; NaN -> False.
    mis_vals = pd.to_numeric(bg["mis.z_score"], errors="coerce")
    mis_threshold = float(np.nanpercentile(mis_vals.to_numpy(), 90))
    mis_mask = (mis_vals.fillna(-np.inf) >= mis_threshold).to_numpy(dtype=bool)
    return {
        "pLI >= 0.9": (pli_mask, {"threshold": PLI_THRESHOLD, "n_hits": int(pli_mask.sum())}),
        "LOEUF <= 0.35": (loeuf_mask, {"threshold": LOEUF_THRESHOLD, "n_hits": int(loeuf_mask.sum())}),
        "missense_z top decile": (mis_mask, {"threshold": mis_threshold, "n_hits": int(mis_mask.sum())}),
    }


# ---------------------------------------------------------------------------
# Sub-A: jackknife + bootstrap
# ---------------------------------------------------------------------------
def run_sub_a(
    logger: logging.Logger,
    f144_set: list[str],
    bg_genes_arr: np.ndarray,
    bg_genes_set: set[str],
    metric_masks: dict[str, tuple[np.ndarray, dict[str, Any]]],
    lengths: np.ndarray,
) -> dict[str, Any]:
    """Jackknife (analytic OR + diagnostic emp_p) + bootstrap OR.

    F144_set is the 48-gene ST12_SynGO ∩ EDT1 bg-intersected list (set order
    lost — we sort to make jackknife reproducible).
    """
    f144_sorted = sorted(f144_set)  # deterministic jackknife index
    n = len(f144_sorted)
    logger.info("SUB-A: F144_set size=%d (sorted for reproducibility)", n)

    sub_a: dict[str, Any] = {"F144_set_n": n, "per_metric": {}}

    # Pre-compute full-set observed OR per metric (reference point).
    def compute_OR(genes_subset: set[str], metric_mask: np.ndarray) -> float:
        """Analytic Fisher OR via _or_from_masks (verbatim batch_048)."""
        list_mask = np.array([g in genes_subset for g in bg_genes_arr], dtype=bool)
        return _or_from_masks(list_mask, metric_mask)

    for metric_name, (metric_mask, meta) in metric_masks.items():
        logger.info("SUB-A metric=%s", metric_name)
        full_set = set(f144_sorted)
        observed_or = compute_OR(full_set, metric_mask)

        # ---- Jackknife: drop one gene at a time, analytic OR + diagnostic emp_p.
        jackknife_ORs: list[float] = []
        jackknife_emp_ps: list[float] = []
        for i, g in enumerate(f144_sorted):
            dropped = full_set - {g}
            or_i = compute_OR(dropped, metric_mask)
            jackknife_ORs.append(float(or_i))

            # Diagnostic length-perm emp_p — 1000 perms per drop (see N_PERM_JACKKNIFE WHY).
            list_mask_drop = np.array([x in dropped for x in bg_genes_arr], dtype=bool)
            perm = length_perm_test(
                list_mask_drop, metric_mask, lengths,
                N_PERM_JACKKNIFE, np.random.default_rng(RNG_SEED),
            )
            jackknife_emp_ps.append(float(perm["emp_p"]))

        jackknife_ORs_arr = np.asarray(jackknife_ORs, dtype=float)
        # WHY finite-mask guard: _or_from_masks returns inf when b==0 or c==0
        # (the dropped gene happens to be the ONLY non-target or the ONLY target).
        # A single inf value in the 48 jackknife ORs would poison jk_mean, jk_std,
        # jk_cv, and max_influence_delta silently via numpy's inf arithmetic.
        # We compute summary stats on finite values only and report the inf count.
        finite_mask = np.isfinite(jackknife_ORs_arr)
        n_inf_jk = int((~finite_mask).sum())
        finite_jk = jackknife_ORs_arr[finite_mask]
        if len(finite_jk) < 2:
            # Almost all inf; jk stats are not meaningful. Report and skip.
            jk_mean = float("nan")
            jk_std = float("nan")
            jk_cv = float("nan")
        else:
            jk_mean = float(finite_jk.mean())
            jk_std = float(finite_jk.std(ddof=1))
            jk_cv = float(jk_std / jk_mean) if jk_mean > 0 else float("nan")
        # Influence: which gene's removal changes OR most (abs delta). Computed
        # over finite values only; if observed_or itself is inf, fall back to NaN.
        if np.isfinite(observed_or) and finite_mask.any():
            finite_deltas = np.abs(finite_jk - observed_or)
            finite_indices = np.where(finite_mask)[0]
            local_max = int(np.argmax(finite_deltas))
            max_idx = int(finite_indices[local_max])
            max_delta = float(finite_deltas[local_max])
        else:
            max_idx = 0
            max_delta = float("nan")

        # ---- Bootstrap: 1000 resamples of size 48 with replacement.
        boot_rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot_ORs = np.empty(N_BOOTSTRAP, dtype=float)
        f144_arr = np.array(f144_sorted)
        finite_count = 0
        for b in range(N_BOOTSTRAP):
            idx = boot_rng.integers(0, n, size=n)
            # With-replacement sample; duplicates collapse in set(). This matches
            # brief §MEASUREMENT ("1000 bootstrap samples of size 48 ... WITH
            # REPLACEMENT ... compute Fisher OR analytically").
            sample_set = set(f144_arr[idx].tolist())
            or_b = compute_OR(sample_set, metric_mask)
            boot_ORs[b] = or_b
            if np.isfinite(or_b):
                finite_count += 1

        # WHY nanpercentile: some bootstrap draws can produce b=0 or c=0 (inf OR)
        # when duplicates collapse. We compute percentiles only over finite values
        # and report the inf count separately — this is NOT a bug, it's a real
        # property of the gene list (all bootstrap genes happen to be high-pLI).
        finite_boot = boot_ORs[np.isfinite(boot_ORs)]
        if len(finite_boot) > 0:
            boot_median = float(np.median(finite_boot))
            boot_ci = [float(np.percentile(finite_boot, 2.5)), float(np.percentile(finite_boot, 97.5))]
        else:
            boot_median = float("nan")
            boot_ci = [float("nan"), float("nan")]

        # min/max over finite values only (matches mean/std/CV guard above).
        jk_min = float(finite_jk.min()) if len(finite_jk) else float("nan")
        jk_max = float(finite_jk.max()) if len(finite_jk) else float("nan")
        sub_a["per_metric"][metric_name] = {
            "observed_or_full_set": float(observed_or),
            "jackknife": {
                "ORs": jackknife_ORs,
                "emp_ps": jackknife_emp_ps,
                "emp_p_n_perm_note": f"{N_PERM_JACKKNIFE} perms per drop (diagnostic; not decision gate)",
                "max_influence_gene": f144_sorted[max_idx],
                "max_influence_delta_OR": max_delta,
                "mean_OR": jk_mean,
                "std_OR": jk_std,
                "CV_OR": jk_cv,
                "min_OR": jk_min,
                "max_OR": jk_max,
                "n_inf_jackknife": n_inf_jk,
            },
            "bootstrap": {
                "n_bootstraps": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "median_OR": boot_median,
                "ci95_OR": boot_ci,
                "n_finite_samples": int(len(finite_boot)),
                "n_inf_or_nan_samples": int(N_BOOTSTRAP - len(finite_boot)),
            },
            "metric_meta": meta,
        }
        logger.info(
            "SUB-A %s: obs_OR=%.3f jk_mean=%.3f jk_CV=%.3f jk_min=%.3f jk_max=%.3f "
            "n_inf_jk=%d max_influence=%s (dOR=%.3f) | boot_median=%.3f boot_CI95=[%.3f, %.3f]",
            metric_name, observed_or, jk_mean, jk_cv,
            jk_min, jk_max, n_inf_jk,
            f144_sorted[max_idx], max_delta,
            boot_median, boot_ci[0], boot_ci[1],
        )
    return sub_a


# ---------------------------------------------------------------------------
# Sub-B: 4-construct Koopmans x EDT1 decomposition
# ---------------------------------------------------------------------------
def build_sub_b_constructs(
    logger: logging.Logger,
    koopmans_full: set[str],
    edt1: dict[str, set[str]],
    bg_genes_set: set[str],
) -> dict[str, dict[str, Any]]:
    """Build the four Sub-B constructs per brief §WHAT.

    Returns dict keyed by B1/B2/B3/B4. Each value carries the gene set (bg-
    intersected) and its membership lists.
    """
    hand = set(SYNOGO_EDT1_BATCH047)
    edt1_all_pc = edt1["EDT1_all_pc"]
    edt1_syngo = edt1["EDT1_SynGO"]

    b1_raw = koopmans_full & edt1_all_pc
    # Sanity-halt: brief §UNINTERPRETABLE anticipates ~73 for Koopmans ∩ EDT1_all_pc.
    # A dramatically smaller intersection (e.g. symbol drift in Koopmans GMT or
    # EDT1 list) would silently invalidate the whole decomposition — fail LOUD.
    assert len(b1_raw) >= 50, (
        f"Koopmans ∩ EDT1_all_pc has unexpected size {len(b1_raw)}; "
        f"brief expected ~73"
    )
    b2_raw = (koopmans_full - hand) & edt1_all_pc
    b3_raw = (koopmans_full - hand - edt1_syngo) & edt1_all_pc
    # B4: brief — "EDT1_SynGO − Hand" (expected n=54 pc, n~47 in bg).
    b4_raw = edt1_syngo - hand

    constructs = {}
    for name, raw in [("B1", b1_raw), ("B2", b2_raw), ("B3", b3_raw), ("B4", b4_raw)]:
        in_bg = raw & bg_genes_set
        constructs[name] = {
            "raw_set": raw,
            "in_bg_set": in_bg,
            "n_raw": len(raw),
            "n_in_bg": len(in_bg),
            "genes_in_bg_sorted": sorted(in_bg),
            "sha256_in_bg": sha256_of_list(sorted(in_bg)),
        }
        logger.info("SUB-B construct %s: n_raw=%d n_in_bg=%d", name, len(raw), len(in_bg))
    return constructs


def run_sub_b(
    logger: logging.Logger,
    constructs: dict[str, dict[str, Any]],
    bg_genes_arr: np.ndarray,
    bg_genes_set: set[str],
    metric_masks: dict[str, tuple[np.ndarray, dict[str, Any]]],
    lengths: np.ndarray,
) -> dict[str, Any]:
    """Run Fisher + 5000-perm length_perm_test for each (construct × metric).

    B4 additionally replicated across 5 seeds for MC-precision. BH-FDR applied
    across the 12 primary emp_p values (4 constructs × 3 metrics).
    """
    sub_b: dict[str, Any] = {}
    primary_pvals: list[float] = []          # for BH-FDR
    primary_keys: list[tuple[str, str]] = [] # (construct, metric) keys aligned to primary_pvals

    for cname, cmeta in constructs.items():
        c_in_bg = cmeta["in_bg_set"]
        sub_b[cname] = {"n_in_bg": cmeta["n_in_bg"], "per_metric": {}}
        if cmeta["n_in_bg"] < 3:
            logger.warning("SUB-B %s: n_in_bg=%d < 3 — UNINTERPRETABLE", cname, cmeta["n_in_bg"])
            for mname in metric_masks:
                sub_b[cname]["per_metric"][mname] = {
                    "OR": None, "OR_ci95": [None, None], "p_fisher": None,
                    "emp_p": None, "note": "n_in_bg<3",
                }
                primary_pvals.append(1.0)  # neutral placeholder for BH alignment
                primary_keys.append((cname, mname))
            continue

        list_mask = np.array([g in c_in_bg for g in bg_genes_arr], dtype=bool)

        for mname, (metric_mask, _meta) in metric_masks.items():
            target_set = set(bg_genes_arr[metric_mask])
            fisher = fisher_enrichment(c_in_bg, bg_genes_set, target_set)

            # Primary emp_p at RNG_SEED (shared with F121/F144 convention).
            perm = length_perm_test(
                list_mask, metric_mask, lengths,
                N_PERM_SUBB, np.random.default_rng(RNG_SEED),
            )

            cell = {
                "n_in_bg": cmeta["n_in_bg"],
                "OR": float(fisher["or"]),
                "OR_ci95": [float(fisher["ci_low"]), float(fisher["ci_high"])],
                "p_fisher": float(fisher["p"]),
                "emp_p": float(perm["emp_p"]),
                "emp_p_n_perm": N_PERM_SUBB,
                "emp_p_seed": RNG_SEED,
                "null_mean": float(perm["null_mean"]),
                "null_std": float(perm["null_std"]),
                "observed_a": int(fisher["a"]),
            }

            # B4 extra seeds (MC-precision check). WHY only B4: B4 is the
            # explicit replication of F144 ST12_minus_hand (brief §WHAT);
            # the extra seeds confirm the emp_p precision floor at 5000 perms.
            if cname == "B4":
                extra = {}
                for seed in B4_EXTRA_SEEDS:
                    p_extra = length_perm_test(
                        list_mask, metric_mask, lengths,
                        N_PERM_SUBB, np.random.default_rng(seed),
                    )
                    extra[str(seed)] = float(p_extra["emp_p"])
                cell["p_values_across_seeds"] = extra

            sub_b[cname]["per_metric"][mname] = cell
            primary_pvals.append(cell["emp_p"])
            primary_keys.append((cname, mname))
            logger.info(
                "SUB-B %s × %s: OR=%.3f CI=[%.3f,%.3f] p_fisher=%.4g emp_p=%.4g (a=%d)",
                cname, mname, cell["OR"], cell["OR_ci95"][0], cell["OR_ci95"][1],
                cell["p_fisher"], cell["emp_p"], cell["observed_a"],
            )

    # BH-FDR across 12 primary emp_p cells. WHY BH over the 12 cells (not 12+5):
    # B4 extra seeds are MC replicates of the SAME cell, not additional
    # hypothesis tests — the brief §MEASUREMENT explicitly says "12 Sub-B
    # primary tests". Placeholder 1.0s for UNINTERPRETABLE cells keep index
    # alignment without spuriously inflating FDR of real cells.
    if primary_pvals:
        _, qvals, _, _ = multipletests(primary_pvals, alpha=BH_ALPHA, method="fdr_bh")
        for (cname, mname), q in zip(primary_keys, qvals):
            if sub_b[cname]["per_metric"].get(mname) is not None:
                sub_b[cname]["per_metric"][mname]["bh_fdr_q"] = float(q)
    return sub_b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output/results.json if present.")
    parser.add_argument("--skip-jackknife-perm", action="store_true",
                        help="Skip Sub-A jackknife emp_p diagnostic (saves ~1-2 min).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger()
    logger.info("=" * 60)
    logger.info("batch_054_B: F144 jackknife/bootstrap + Koopmans x EDT1 4-construct")

    out_path = OUTPUT_DIR / "results.json"
    if out_path.exists() and not args.force:
        logger.info("Output %s exists; pass --force to re-run. Exiting.", out_path)
        return 0

    # -- Load data (verbatim batch_048 primitives) --
    gnomad = load_gnomad(logger)
    edt1 = load_pgc3_edt1(logger)
    koopmans_full = parse_syngo_gmt(SYNGO_GMT)
    logger.info("Koopmans (syngo_2024.gmt): %d unique gene symbols", len(koopmans_full))

    bg = gnomad[~gnomad["mhc_indicator"]].copy()
    bg_genes_set = set(bg["gene"].astype(str))
    bg_genes_arr = bg["gene"].astype(str).to_numpy()
    lengths = bg["gene_length"].fillna(0).to_numpy()
    # WHY pass bg (not gnomad): single source of truth for bg row order. Prevents
    # mask/bg misalignment if build_metric_masks is ever modified to sort/reorder.
    metric_masks = build_metric_masks(bg, gnomad)
    # Sanity: all metric masks MUST align with bg_genes_arr (same length, same order).
    pli_mask_check, _ = metric_masks["pLI >= 0.9"]
    assert len(pli_mask_check) == len(bg_genes_arr), "mask/bg length mismatch"

    # -- Provenance + reproduction gate (FIRST, before any expensive test) --
    pli_mask, _ = metric_masks["pLI >= 0.9"]
    provenance = build_provenance(logger, gnomad, bg_genes_set, bg_genes_arr, pli_mask)
    prov_path = OUTPUT_DIR / "provenance.json"
    with prov_path.open("w") as fh:
        json.dump(provenance, fh, indent=2, default=str)
    logger.info("Wrote provenance: %s", prov_path)

    if not provenance["reproduction_check"]["match"]:
        logger.error(
            "REPRODUCTION GATE FAILED — hand-list pLI OR=%.7f != target %.7f. "
            "Pipeline has drifted vs F121; STOP.",
            provenance["reproduction_check"]["hand_list_pLI_OR"],
            REPRODUCTION_TARGET_OR,
        )
        return 1

    # -- Build F144_set: EDT1_SynGO ∩ bg (48-gene target per brief) --
    f144_set = sorted(edt1["EDT1_SynGO"] & bg_genes_set)
    logger.info(
        "F144_set (EDT1_SynGO ∩ bg): n=%d (brief expected ~48)", len(f144_set),
    )

    # -- Sub-A --
    if args.skip_jackknife_perm:
        logger.warning("--skip-jackknife-perm set; Sub-A jackknife emp_ps will be null.")
    sub_a = run_sub_a(
        logger, f144_set, bg_genes_arr, bg_genes_set, metric_masks, lengths,
    )

    # -- Sub-B --
    constructs = build_sub_b_constructs(logger, koopmans_full, edt1, bg_genes_set)
    sub_b = run_sub_b(
        logger, constructs, bg_genes_arr, bg_genes_set, metric_masks, lengths,
    )

    # -- Gene-list membership (for audit) --
    gene_lists = {
        "F144_set": {
            "n": len(f144_set),
            "sha256": sha256_of_list(f144_set),
            "genes": f144_set,
        },
    }
    for cname, cmeta in constructs.items():
        gene_lists[cname] = {
            "n": cmeta["n_in_bg"],
            "sha256": cmeta["sha256_in_bg"],
            "genes": cmeta["genes_in_bg_sorted"],
        }

    # -- Assemble results.json --
    results = {
        "batch": "batch_054_B",
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "provenance": provenance,
        "sub_a_jackknife_bootstrap": sub_a,
        "sub_b_decomposition": sub_b,
        "gene_lists": gene_lists,
        "hyperparameters": provenance["hyperparameters"],
    }
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Wrote results: %s", out_path)

    # -- Compact console summary --
    logger.info("\n=== SUB-A summary ===")
    for m, d in sub_a["per_metric"].items():
        jk = d["jackknife"]; bt = d["bootstrap"]
        logger.info(
            "  %-24s obs_OR=%.2f | jk[mean=%.2f CV=%.3f min=%.2f max=%.2f max_infl=%s] | boot[med=%.2f CI95=%.2f-%.2f]",
            m, d["observed_or_full_set"],
            jk["mean_OR"], jk["CV_OR"], jk["min_OR"], jk["max_OR"], jk["max_influence_gene"],
            bt["median_OR"], bt["ci95_OR"][0], bt["ci95_OR"][1],
        )
    logger.info("\n=== SUB-B summary ===")
    for cname, cdata in sub_b.items():
        for mname, cell in cdata["per_metric"].items():
            if cell.get("OR") is None:
                logger.info("  %s × %-22s n=%d UNINTERPRETABLE", cname, mname, cdata["n_in_bg"])
            else:
                logger.info(
                    "  %s × %-22s n=%d OR=%.2f CI=[%.2f,%.2f] emp_p=%.4g q=%.4g",
                    cname, mname, cdata["n_in_bg"],
                    cell["OR"], cell["OR_ci95"][0], cell["OR_ci95"][1],
                    cell["emp_p"], cell.get("bh_fdr_q", float("nan")),
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
