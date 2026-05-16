#!/usr/bin/env python3
"""batch_060 E1 -- Bellenguez 2022 AD replication with relaxed QC gate.

Implements brief_v2.md section E1 EXACTLY.

Overview (WHY):
  iter_059 E4 produced Spearman rho=0.306 between Bellenguez and existing AD
  gene-Z. The original QC gate (rho >= 0.60) rejected this. L059_01
  established that rho ~ 0.25-0.50 is expected for cross-generation GWAS
  (different sample composition, QC pipelines, LD structure). E1 re-evaluates
  with a relaxed gate (rho >= 0.25) plus a top-20 locus spot-check (by
  absolute MAGMA-Z overlap with existing top-40).

  Steps:
    1. Load Bellenguez gene-Z from batch_059/output/e4/magma_gene_z.tsv.
    2. QC: rho >= 0.25 AND top-20 spot-check overlap >= 10.
    3. Run Sub-A v2.1 battery on EDT1-ex-B3, B3, EDT1-FULL,
       Koopmans-ex-B3-ex-EDT1 with Bellenguez MAGMA-Z as outcome.
    4. Apply decision rules from brief_v2 section E1.

Outputs:
  experiments/batch_060/output/e1/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    B3_GENES,
    BATCH055B_WORK,
    BH_Q,
    B060_SEED_E1,
    DISORDERS,
    E1_AD_NOT_REPLICATED_CAP,
    E1_AD_REPLICATED_HI,
    E1_AD_REPLICATED_LO,
    E1_AD_STRONGER_LO,
    E1_BELLENGUEZ_GENE_Z_TSV,
    E1_EXISTING_AD_TOP_K,
    E1_QC_RHO_MIN,
    E1_R1_HI,
    E1_R1_LO,
    E1_TOP_K_OVERLAP_MIN,
    E1_TOP_K_SPOTCHECK,
    EXISTING_ALZHEIMERS_MAGMA,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_GENELOC,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    atomic_write_json,
    bh_fdr,
    build_sub_a_frame,
    classify_disorder_v2,
    compute_dfbetas_cooks,
    fit_tukey_biweight,
    load_edt1,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    load_koopmans_ex_B3,
    load_nsnps_per_disorder,
    setup_logger,
    sha256_file,
    symbols_to_ensgids,
)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Import Sub-A v2.1 battery fit functions from batch_058 (Rule 1: reuse).
BATCH058_SCRIPTS = (
    Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
    / "experiments" / "batch_058" / "scripts"
)
sys.path.insert(0, str(BATCH058_SCRIPTS))
from sub_a_robust_battery import (  # noqa: E402
    fit_huber, fit_ols, fit_rank_magma_ols,
    influential_outlier_reconciliation,
)


def load_bellenguez_gene_z(path: Path, logger) -> pd.DataFrame:
    """Load Bellenguez gene-Z TSV produced by batch_059 E4.

    WHY we load from the E4 output rather than re-running MAGMA: the MAGMA
    pipeline already ran in iter_059 and produced valid gene-Z (rho=0.306,
    overlap=99.9%). Re-running would waste 30+ minutes of compute for
    identical output. Cardinal Rule 1.

    Returns DataFrame[ENSGID, MAGMA_Z].
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Bellenguez gene-Z missing: {path}. "
            f"Run batch_059 E4 first (or check output path)."
        )
    df = pd.read_csv(path, sep="\t")
    if not {"ENSGID", "MAGMA_Z"} <= set(df.columns):
        raise RuntimeError(
            f"Bellenguez gene-Z schema drift: {list(df.columns)}. "
            f"Expected columns ENSGID and MAGMA_Z."
        )
    df = df.dropna(subset=["ENSGID", "MAGMA_Z"]).drop_duplicates(
        subset="ENSGID", keep="first"
    )
    logger.info("Bellenguez gene-Z loaded: %d ENSGIDs from %s", len(df), path)
    return df[["ENSGID", "MAGMA_Z"]].reset_index(drop=True)


def load_existing_ad_gene_z(logger) -> pd.DataFrame:
    """Load existing AD pipeline gene-Z (Jansen-like) for QC comparison.

    WHY: We need the existing AD gene-Z to compute QC Spearman rho and
    the top-K spot-check overlap. Reuses the same mapping logic as
    batch_059 E4's qc_against_existing_ad.
    """
    from _common import MAGMA_GENELOC as _GENELOC
    geneloc_path = Path(
        "/home/yuanz/Documents/GitHub/biomarvin_schizophrenia"
        "/tools/magma_bin/refs/NCBI37.3.gene.loc"
    )
    if not EXISTING_ALZHEIMERS_MAGMA.exists():
        raise FileNotFoundError(
            f"Existing AD MAGMA missing: {EXISTING_ALZHEIMERS_MAGMA}"
        )
    ex = pd.read_csv(EXISTING_ALZHEIMERS_MAGMA, sep=r"\s+")
    ex = ex.rename(columns={"GENE": "entrez", "ZSTAT": "MAGMA_Z"})
    ex["entrez"] = ex["entrez"].astype(str)
    geneloc = pd.read_csv(
        geneloc_path, sep="\t", header=None,
        names=["entrez", "chr", "start", "end", "strand", "symbol"],
        dtype={"entrez": str, "symbol": str},
    )
    ent2sym = geneloc.drop_duplicates(subset="entrez", keep="first")[
        ["entrez", "symbol"]
    ]
    ex = ex.merge(ent2sym, on="entrez", how="left")
    annot = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME"]]
    sym2ensg = annot.drop_duplicates(subset="NAME", keep="first").rename(
        columns={"NAME": "symbol"}
    )
    ex = ex.merge(sym2ensg, on="symbol", how="left").dropna(subset=["ENSGID"])
    ex = ex.drop_duplicates(subset="ENSGID", keep="first")
    logger.info("Existing AD gene-Z loaded: %d ENSGIDs", len(ex))
    return ex[["ENSGID", "MAGMA_Z"]].reset_index(drop=True)


def qc_bellenguez(bellenguez_df: pd.DataFrame,
                   existing_df: pd.DataFrame,
                   logger) -> dict:
    """QC: Spearman rho >= 0.25 AND top-20 spot-check overlap >= 10.

    WHY two gates: brief_v2 section E1 DECISION RULE requires both. The rho
    gate tests global concordance. The top-20 spot-check tests whether the
    highest-signal loci are consistent (which is what matters for gene-set
    enrichment tests downstream).

    Top-20 spot-check: rank Bellenguez genes by |MAGMA_Z|, take top-20.
    Rank existing AD genes by |MAGMA_Z|, take top-40. Count overlap.
    """
    merged = bellenguez_df.rename(
        columns={"MAGMA_Z": "MAGMA_Z_bellenguez"}
    ).merge(
        existing_df.rename(columns={"MAGMA_Z": "MAGMA_Z_existing"})[
            ["ENSGID", "MAGMA_Z_existing"]
        ],
        on="ENSGID", how="inner",
    )
    n_overlap = len(merged)
    if n_overlap < 100:
        return {
            "status": "failed",
            "reason": f"Too few overlapping genes: {n_overlap}",
        }
    rho, rho_p = spearmanr(
        merged["MAGMA_Z_bellenguez"], merged["MAGMA_Z_existing"]
    )
    rho_pass = bool(rho >= E1_QC_RHO_MIN)

    # Top-K spot-check.
    bell_top = set(
        merged.nlargest(E1_TOP_K_SPOTCHECK, "MAGMA_Z_bellenguez",
                        keep="first")["ENSGID"]
    )
    # WHY |MAGMA_Z| for the spot-check: brief_v2 says "rank genes by
    # |MAGMA_Z|". For Bellenguez top-20, we use absolute value.
    merged["abs_MAGMA_Z_bellenguez"] = merged["MAGMA_Z_bellenguez"].abs()
    merged["abs_MAGMA_Z_existing"] = merged["MAGMA_Z_existing"].abs()
    bell_top_abs = set(
        merged.nlargest(E1_TOP_K_SPOTCHECK, "abs_MAGMA_Z_bellenguez",
                        keep="first")["ENSGID"]
    )
    existing_top_abs = set(
        merged.nlargest(E1_EXISTING_AD_TOP_K, "abs_MAGMA_Z_existing",
                        keep="first")["ENSGID"]
    )
    top_k_overlap = len(bell_top_abs & existing_top_abs)
    top_k_pass = bool(top_k_overlap >= E1_TOP_K_OVERLAP_MIN)

    logger.info(
        "QC: rho=%.4f (pass=%s); top-%d vs top-%d overlap=%d (pass=%s)",
        rho, rho_pass, E1_TOP_K_SPOTCHECK, E1_EXISTING_AD_TOP_K,
        top_k_overlap, top_k_pass,
    )
    return {
        "status": "ok",
        "n_overlap_genes": int(n_overlap),
        "n_bellenguez": int(len(bellenguez_df)),
        "n_existing": int(len(existing_df)),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "rho_threshold": float(E1_QC_RHO_MIN),
        "rho_pass": rho_pass,
        "bell_top_k": E1_TOP_K_SPOTCHECK,
        "existing_top_k": E1_EXISTING_AD_TOP_K,
        "top_k_overlap": int(top_k_overlap),
        "top_k_overlap_min": int(E1_TOP_K_OVERLAP_MIN),
        "top_k_pass": top_k_pass,
        "qc_overall_pass": bool(rho_pass and top_k_pass),
        "bell_top_k_ensgids": sorted(bell_top_abs),
        "existing_top_k_ensgids": sorted(existing_top_abs),
        "overlap_ensgids": sorted(bell_top_abs & existing_top_abs),
    }


def fit_battery_on_set(bellenguez_df: pd.DataFrame, gnomad: pd.DataFrame,
                        annot: pd.DataFrame, gene_set_ensg: set[str],
                        logger, tag: str) -> dict:
    """Run Sub-A v2.1 battery with Bellenguez MAGMA-Z as outcome.

    WHY reuse batch_059 E4's approach: Cardinal Rule 1. The fit_battery_on_set
    pattern from batch_059 E4 builds a regression frame with Bellenguez MAGMA-Z
    as the outcome variable, then runs OLS + Huber + Tukey + rank-MAGMA +
    DFBETAS diagnostics. We replicate that pattern exactly.

    Returns dict with battery results or failure reason.
    """
    nsnps = load_nsnps_per_disorder("alzheimers")
    frame = (
        bellenguez_df.merge(nsnps, on="ENSGID", how="inner")
        .merge(
            gnomad[["ENSGID", "lof_pLI", "lof_oe_ci_upper", "lof_exp"]],
            on="ENSGID", how="inner",
        )
        .merge(annot[["ENSGID", "log10_gene_length"]], on="ENSGID", how="inner")
    )
    frame = frame.dropna(
        subset=["MAGMA_Z", "lof_pLI", "lof_exp", "log10_gene_length", "NSNPS"]
    ).copy()
    frame["in_set"] = frame["ENSGID"].isin(gene_set_ensg).astype(int)
    frame["log10_exp_lof_plus1"] = np.log10(
        frame["lof_exp"].astype(float) + 1.0
    )
    frame["log10_NSNPS_plus1"] = np.log10(
        frame["NSNPS"].astype(float) + 1.0
    )
    frame = frame.sort_values("ENSGID").reset_index(drop=True)
    n = len(frame)
    n_in = int(frame["in_set"].sum())
    logger.info("E1 battery %s: n=%d in_set=%d", tag, n, n_in)
    if n < 1000 or n_in < 10:
        return {"status": "failed",
                "reason": f"n={n} in_set={n_in}"}
    covs = ["log10_gene_length", "lof_pLI",
            "log10_exp_lof_plus1", "log10_NSNPS_plus1"]
    ols = fit_ols(frame, "in_set", covs)
    huber = fit_huber(frame, "in_set", covs)
    tukey = fit_tukey_biweight(frame, covs, "in_set")
    rank_ols = fit_rank_magma_ols(frame, "in_set", covs)
    infl = compute_dfbetas_cooks(frame, covs, "in_set")
    recon = influential_outlier_reconciliation(frame, covs, "in_set", infl)
    return {
        "status": "ok",
        "tag": tag,
        "n_gene_universe": n,
        "n_set_in_universe": n_in,
        "ols": ols, "huber": huber, "tukey": tukey,
        "rank_magma_ols": rank_ols,
        "influence": infl,
        "influential_outlier_reconciliation": recon,
    }


def classify_ad_beta(beta: float, logger) -> dict:
    """Apply E1 AD decision rule from brief_v2 section E1.

    Decision boundaries:
      - beta in [+0.10, +0.50] -> AD_REPLICATED
      - beta < +0.10           -> AD_NOT_REPLICATED
      - beta > +0.50           -> AD_STRONGER

    WHY these thresholds: brief_v2 section E1 DECISION RULE specifies them,
    derived from the iter_058 F058_06 anchor of AD beta=+0.31 with a
    +/- tolerance band.
    """
    if not np.isfinite(beta):
        return {"verdict": "UNINTERPRETABLE", "reason": "beta non-finite"}
    if beta < E1_AD_NOT_REPLICATED_CAP:
        return {
            "verdict": "AD_NOT_REPLICATED",
            "reason": f"beta={beta:.4f} < {E1_AD_NOT_REPLICATED_CAP}",
        }
    if beta > E1_AD_STRONGER_LO:
        return {
            "verdict": "AD_STRONGER",
            "reason": f"beta={beta:.4f} > {E1_AD_STRONGER_LO}",
        }
    if E1_AD_REPLICATED_LO <= beta <= E1_AD_REPLICATED_HI:
        return {
            "verdict": "AD_REPLICATED",
            "reason": (
                f"beta={beta:.4f} in [{E1_AD_REPLICATED_LO}, "
                f"{E1_AD_REPLICATED_HI}]"
            ),
        }
    # Shouldn't reach here given the thresholds cover all reals, but
    # guard defensively.
    return {
        "verdict": "UNINTERPRETABLE",
        "reason": f"beta={beta:.4f} falls in no decision band (unexpected)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_060 E1: Bellenguez AD rerun")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: skip secondary gene sets.")
    args = parser.parse_args()

    logger = setup_logger("batch_060.e1", LOGS_DIR / "e1.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "e1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load Bellenguez gene-Z.
    bellenguez_df = load_bellenguez_gene_z(E1_BELLENGUEZ_GENE_Z_TSV, logger)

    # Step 2: Load existing AD gene-Z for QC.
    existing_df = load_existing_ad_gene_z(logger)

    # Step 3: QC with relaxed gate.
    qc = qc_bellenguez(bellenguez_df, existing_df, logger)
    if not qc.get("qc_overall_pass", False):
        results = {
            "status": "E1_QC_FAILED",
            "batch": "060", "sub": "e1", "brief": "brief_v2.md",
            "wall_s": time.time() - t0,
            "qc": qc,
            "verdict": {
                "verdict": "E1_QC_FAILED",
                "reason": (
                    f"QC gate failed: rho_pass={qc.get('rho_pass')}, "
                    f"top_k_pass={qc.get('top_k_pass')}"
                ),
            },
        }
        atomic_write_json(results, out_dir / "results.json")
        logger.info("E1 QC FAILED, wrote %s", out_dir / "results.json")
        return 0

    # Step 4: Load upstream data for battery.
    gnomad = load_gnomad_per_brief_v2()
    annot = load_gene_annot()
    edt1_sym = load_edt1()
    b3 = set(B3_GENES)
    edt1_ex_b3_sym = edt1_sym - b3
    edt1_full_sym = edt1_sym
    koop_ex_sym = load_koopmans_ex_B3() - edt1_sym

    # Step 5: Run batteries.
    def run_battery(sym_set: set[str], tag: str) -> dict:
        ensg, _ = symbols_to_ensgids(sym_set)
        return fit_battery_on_set(bellenguez_df, gnomad, annot, ensg,
                                    logger, tag)

    primary = run_battery(edt1_ex_b3_sym, "EDT1_ex_B3")
    if args.smoke:
        secondary = {"B3": {"status": "skipped_smoke"}}
    else:
        secondary = {
            "B3": run_battery(b3, "B3"),
            "EDT1_FULL": run_battery(edt1_full_sym, "EDT1_FULL"),
            "KOOP_ex_B3_ex_EDT1": run_battery(koop_ex_sym, "KOOP_ex_B3_ex_EDT1"),
        }

    # Step 6: BH-FDR on primary + secondary batteries.
    bh_pvals = []
    bh_labels = []
    for tag, item in [("EDT1_ex_B3", primary)] + [
        (k, v) for k, v in secondary.items() if isinstance(v, dict)
    ]:
        if item.get("status") == "ok":
            p = float(item["ols"]["p_one_sided"])
            bh_pvals.append(p)
            bh_labels.append(tag)
    qvals = bh_fdr(bh_pvals) if bh_pvals else []
    q_by_tag = dict(zip(bh_labels, qvals))

    # Step 7: R1 reproduction gate -- B3 x SCZ beta_OLS using existing
    # SCZ pipeline (not Bellenguez). WHY: R1 tests pipeline stability.
    b3_ensg, _ = symbols_to_ensgids(b3)
    r1_frame = build_sub_a_frame("scz", gnomad, annot, b3_ensg)
    r1_ols = fit_ols(r1_frame, "in_set",
                      ["log10_gene_length", "lof_pLI",
                       "log10_exp_lof_plus1", "log10_NSNPS_plus1"])
    r1_beta = float(r1_ols.get("beta_1", float("nan")))
    r1_pass = bool(np.isfinite(r1_beta) and E1_R1_LO <= r1_beta <= E1_R1_HI)
    logger.info("E1 R1: B3 SCZ beta=%.3f in [%.2f, %.2f]? %s",
                 r1_beta, E1_R1_LO, E1_R1_HI, r1_pass)

    # Step 8: Apply verdict.
    if not r1_pass:
        verdict = {
            "verdict": "UNINTERPRETABLE",
            "reason": f"R1 failed: B3 SCZ beta={r1_beta:.3f} not in "
                       f"[{E1_R1_LO}, {E1_R1_HI}]",
        }
    elif primary.get("status") != "ok":
        verdict = {
            "verdict": "UNINTERPRETABLE",
            "reason": f"Primary battery failed: {primary.get('reason')}",
        }
    else:
        beta = float(primary["ols"]["beta_1"])
        verdict = classify_ad_beta(beta, logger)
    logger.info("E1 verdict: %s", verdict.get("verdict"))

    # Provenance.
    provenance = {
        "bellenguez_gene_z_tsv": sha256_file(E1_BELLENGUEZ_GENE_Z_TSV),
        "existing_ad_magma": (
            sha256_file(EXISTING_ALZHEIMERS_MAGMA)
            if EXISTING_ALZHEIMERS_MAGMA.exists() else "missing"
        ),
        "gnomad_tsv": sha256_file(GNOMAD_TSV),
        "gene_annot": sha256_file(GENE_ANNOT),
        "magma_scz": sha256_file(MAGMA_SCZ_GENES_OUT),
    }

    results = {
        "status": "ok",
        "batch": "060", "sub": "e1", "brief": "brief_v2.md",
        "wall_s": time.time() - t0,
        "smoke": bool(args.smoke),
        "qc": qc,
        "R1_reproduction_gate": {
            "lo": E1_R1_LO, "hi": E1_R1_HI,
            "b3_scz_beta_ols": r1_beta, "pass": r1_pass,
        },
        "primary_EDT1_ex_B3": primary,
        "secondary": secondary,
        "bh_fdr_family": {
            "labels": bh_labels, "pvals": bh_pvals, "qvals": qvals,
            "family_size": len(bh_pvals), "q_threshold": BH_Q,
        },
        "verdict": verdict,
        "provenance_sha256": provenance,
        "brief_contract": {
            "qc_rho_threshold": E1_QC_RHO_MIN,
            "top_k_spotcheck": E1_TOP_K_SPOTCHECK,
            "top_k_overlap_min": E1_TOP_K_OVERLAP_MIN,
            "ad_replicated_band": [E1_AD_REPLICATED_LO, E1_AD_REPLICATED_HI],
            "seed": B060_SEED_E1,
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("E1 wrote %s (wall=%.1fs)",
                 out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
