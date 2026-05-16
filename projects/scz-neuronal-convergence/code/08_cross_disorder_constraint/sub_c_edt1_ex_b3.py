#!/usr/bin/env python3
"""batch_058 Sub-C v2 — F147 scope via EDT1-ex-B3 replication.

Applies the Sub-A v2.1 diagnostic battery (OLS + HuberT + TukeyBiweight +
rank-MAGMA OLS + DFBETAS + Cook's D) to THREE gene sets on all 8 disorders:
  1. PRIMARY: EDT1-ex-B3 (EDT1 set with B3's 18 genes removed).
  2. SECONDARY (reproduction gate R1): EDT1 FULL (with B3 included).
  3. SECONDARY (descriptive purification): Koopmans-ex-B3-ex-EDT1.

Single 8-test BH-FDR family PER GENE SET on OLS one-sided p (brief_v2 L260).
Aggregate classification via `aggregate_pattern_sub_c` (first-match):
  F147_NARROW_CONFIRMED → F147_EDT1_EXTENDED →
  F147_CROSS_DISORDER_EXTENDED → INTERMEDIATE → UNINTERPRETABLE.

Reproduction gate R1 (Sub-C): EDT1 FULL SCZ β_OLS ∈ [+3.0, +3.8]
  (anchor F-057-D β=+3.43).

Output: experiments/batch_058/output/sub_c/results.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BH_Q,
    B3_GENES,
    BATCH055B_WORK,
    DFBETAS_CUTOFF,
    DISORDERS,
    GENE_ANNOT,
    GNOMAD_TSV,
    LOGS_DIR,
    MAGMA_GENELOC,
    MAGMA_SCZ_GENES_OUT,
    OUTPUT_DIR,
    PGC3_XLSX,
    REPRO_R1_SUB_C_FULL_HI,
    REPRO_R1_SUB_C_FULL_LO,
    SUB_C_FRAGILE_UNINTERPRETABLE_COUNT,
    SUB_C_MAPPING_GATE,
    SUB_C_MIN_EFFECT,
    aggregate_pattern_sub_c,
    atomic_write_json,
    bh_fdr,
    classify_disorder_v2,
    compute_dfbetas_cooks,
    fit_tukey_biweight,
    load_edt1,
    load_gene_annot,
    load_gnomad_per_brief_v2,
    load_koopmans_full_symbols,
    setup_logger,
    sha256_file,
    symbols_to_ensgids,
)

# We re-use Sub-A's fit_ols / fit_huber / fit_rank_magma_ols helpers.
import sub_a_robust_battery as sub_a  # noqa: E402
from sub_a_robust_battery import (  # noqa: E402
    PRIMARY_COVS,
    fit_huber,
    fit_ols,
    fit_rank_magma_ols,
    influential_outlier_reconciliation,
    run_disorder,
)

import numpy as np
import pandas as pd
import statsmodels.api as sm


MIN_N_UNIVERSE = 15000


def ensgid_set_from_symbols(symbols: set[str], logger,
                              label: str) -> tuple[set[str], dict]:
    """Map a symbol set to ENSGID via gene_annot. Log mapping rate."""
    ensg_set, sym2ensg = symbols_to_ensgids(symbols)
    rate = (len(sym2ensg) / len(symbols)) if symbols else 0.0
    logger.info("%s: %d symbols → %d ENSGID (rate=%.3f)",
                 label, len(symbols), len(ensg_set), rate)
    return ensg_set, {
        "n_symbols": len(symbols),
        "n_mapped": len(sym2ensg),
        "mapping_rate": float(rate),
        "gate_pass": bool(rate >= SUB_C_MAPPING_GATE),
    }


def run_battery_for_set(gene_set_name: str, gene_set_ensg: set[str],
                         gnomad: pd.DataFrame, annot: pd.DataFrame,
                         disorders_to_run: list[str], logger,
                         smoke_frame_size: int = 0) -> dict:
    """Run Sub-A diagnostic battery for all disorders on a given gene set.

    Re-uses sub_a_robust_battery.run_disorder for the per-disorder work,
    then BH-FDRs within the 8-test family, then classifies each disorder
    via classify_disorder_v2 and aggregates via aggregate_pattern_sub_c.
    """
    per_disorder: dict[str, dict] = {}
    for d in disorders_to_run:
        item = run_disorder(d, gnomad, annot, gene_set_ensg, logger,
                             indicator_col="in_set",
                             smoke_frame_size=smoke_frame_size)
        per_disorder[d] = item

    # BH-FDR on OLS one-sided p, within this set's 8-test family.
    if all(per_disorder.get(d, {}).get("status") == "ok"
            for d in disorders_to_run):
        pvals = [per_disorder[d]["ols"]["p_one_sided"]
                 for d in disorders_to_run]
        qvals = bh_fdr(pvals)
        q_by_d = dict(zip(disorders_to_run, qvals))
        for d, q in q_by_d.items():
            per_disorder[d]["bh_q"] = q
    else:
        q_by_d = {}

    # Classify each disorder.
    # v2.1 FIX #5: pre-compute `reconciled` from run_disorder's post-removal
    # OLS+Tukey refit and pass it in.
    classifications: dict[str, str] = {}
    for d in disorders_to_run:
        item = per_disorder[d]
        if item.get("status") != "ok":
            classifications[d] = "UNCLASSIFIED"
            continue
        infl = item["influence"]
        max_df = (float(infl.get("max_abs_dfbetas_b3"))
                  if infl.get("status") == "ok" else float("nan"))
        q = item.get("bh_q", float("nan"))
        recon_info = item.get("influential_outlier_reconciliation", {})
        reconciled = bool(recon_info.get("status") == "ok"
                           and recon_info.get("reconciled", False))
        cls = classify_disorder_v2(
            item["ols"], item["huber"], item["tukey"],
            item["rank_magma_ols"], max_df, q,
            reconciled=reconciled,
        )
        classifications[d] = cls
        item["classification_v2"] = cls
        item["reconciled_post_removal"] = reconciled

    # Aggregate.
    scz_beta = float("nan")
    if per_disorder.get("scz", {}).get("status") == "ok":
        scz_beta = float(per_disorder["scz"]["ols"]["beta_1"])
    if len(classifications) == len(disorders_to_run):
        aggregate = aggregate_pattern_sub_c(classifications, scz_beta)
    else:
        aggregate = {
            "classification": "UNINTERPRETABLE",
            "reason": "not all disorders classified",
        }
    return {
        "gene_set": gene_set_name,
        "n_gene_set_ensg": len(gene_set_ensg),
        "per_disorder": per_disorder,
        "per_disorder_classification": classifications,
        "bh_q_by_disorder": q_by_d,
        "aggregate_pattern": aggregate,
        "scz_beta_ols": scz_beta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_058 Sub-C v2")
    parser.add_argument("--smoke", action="store_true",
                         help="Smoke: SCZ only on first 50 EDT1-ex-B3 genes")
    parser.add_argument("--smoke-genes", type=int, default=50,
                         help="Smoke: cap gene set size.")
    parser.add_argument("--smoke-frame-size", type=int, default=800,
                         help="Smoke: cap regression frame size.")
    args = parser.parse_args()
    smoke_frame = args.smoke_frame_size if args.smoke else 0

    logger = setup_logger("batch_058.sub_c", LOGS_DIR / "sub_c.log")
    t0 = time.time()
    out_dir = OUTPUT_DIR / "sub_c"
    out_dir.mkdir(parents=True, exist_ok=True)

    gnomad = load_gnomad_per_brief_v2()
    annot = load_gene_annot()

    # Load gene sets.
    edt1_symbols = load_edt1()
    koopmans_symbols = load_koopmans_full_symbols()
    b3_set = set(B3_GENES)
    logger.info("EDT1 symbols=%d; Koopmans symbols=%d; B3=%d",
                 len(edt1_symbols), len(koopmans_symbols), len(b3_set))

    # Derived sets:
    edt1_ex_b3_symbols = edt1_symbols - b3_set
    koop_ex_b3_ex_edt1_symbols = (koopmans_symbols - b3_set) - edt1_symbols
    logger.info("EDT1-ex-B3 symbols=%d; Koopmans-ex-B3-ex-EDT1=%d",
                 len(edt1_ex_b3_symbols), len(koop_ex_b3_ex_edt1_symbols))

    # Map to ENSGID.
    edt1_full_ensg, edt1_full_map = ensgid_set_from_symbols(
        edt1_symbols, logger, "EDT1_FULL",
    )
    edt1_ex_b3_ensg, edt1_ex_b3_map = ensgid_set_from_symbols(
        edt1_ex_b3_symbols, logger, "EDT1_ex_B3",
    )
    koop_ex_b3_ex_edt1_ensg, koop_map = ensgid_set_from_symbols(
        koop_ex_b3_ex_edt1_symbols, logger, "Koopmans_ex_B3_ex_EDT1",
    )

    # Mapping gate for PRIMARY (EDT1-ex-B3) per brief_v2 L230.
    if not edt1_ex_b3_map["gate_pass"] and not args.smoke:
        # Brief mandates UNINTERPRETABLE if mapping rate < 80%.
        logger.error("EDT1-ex-B3 mapping gate FAILED (rate=%.3f < %.2f)",
                      edt1_ex_b3_map["mapping_rate"], SUB_C_MAPPING_GATE)
        results = {
            "status": "uninterpretable",
            "batch": "058", "sub": "c", "brief": "brief_v2.md (v2.1)",
            "wall_s": time.time() - t0,
            "reason": (f"EDT1-ex-B3 symbol→ENSGID mapping rate "
                        f"{edt1_ex_b3_map['mapping_rate']:.3f} < "
                        f"{SUB_C_MAPPING_GATE}"),
            "edt1_ex_b3_mapping": edt1_ex_b3_map,
        }
        atomic_write_json(results, out_dir / "results.json")
        return 0

    # Smoke: truncate gene sets to --smoke-genes first elements.
    if args.smoke:
        edt1_ex_b3_ensg = set(sorted(edt1_ex_b3_ensg)[:args.smoke_genes])
        edt1_full_ensg = set(sorted(edt1_full_ensg)[:args.smoke_genes])
        koop_ex_b3_ex_edt1_ensg = set(
            sorted(koop_ex_b3_ex_edt1_ensg)[:args.smoke_genes]
        )
        disorders_to_run = ["scz"]
    else:
        disorders_to_run = DISORDERS

    # PRIMARY: EDT1-ex-B3.
    primary_edt1_ex_b3 = run_battery_for_set(
        "EDT1_ex_B3", edt1_ex_b3_ensg, gnomad, annot,
        disorders_to_run, logger, smoke_frame_size=smoke_frame,
    )

    # SECONDARY: EDT1 FULL (for reproduction gate R1).
    secondary_edt1_full = run_battery_for_set(
        "EDT1_FULL", edt1_full_ensg, gnomad, annot,
        disorders_to_run, logger, smoke_frame_size=smoke_frame,
    )

    # SECONDARY: Koopmans-ex-B3-ex-EDT1.
    secondary_koop = run_battery_for_set(
        "Koopmans_ex_B3_ex_EDT1", koop_ex_b3_ex_edt1_ensg, gnomad, annot,
        disorders_to_run, logger, smoke_frame_size=smoke_frame,
    )

    # Reproduction gate R1 (Sub-C) on EDT1 FULL SCZ.
    scz_beta_full = secondary_edt1_full.get("scz_beta_ols", float("nan"))
    repro_r1 = {
        "gene_set": "EDT1_FULL",
        "target_lo": REPRO_R1_SUB_C_FULL_LO,
        "target_hi": REPRO_R1_SUB_C_FULL_HI,
        "scz_beta_ols_edt1_full": scz_beta_full,
        "pass": bool(np.isfinite(scz_beta_full)
                     and REPRO_R1_SUB_C_FULL_LO <= scz_beta_full
                     <= REPRO_R1_SUB_C_FULL_HI),
    }

    # Provenance SHA256s.
    provenance = {
        "gnomad_tsv": sha256_file(GNOMAD_TSV),
        "gene_annot": sha256_file(GENE_ANNOT),
        "magma_geneloc": sha256_file(MAGMA_GENELOC),
        "magma_scz": sha256_file(MAGMA_SCZ_GENES_OUT),
        "pgc3_xlsx": sha256_file(PGC3_XLSX),
        "magma_per_disorder": {
            d: sha256_file(BATCH055B_WORK / d / "full.gene.genes.out")
            for d in disorders_to_run if d != "scz"
        },
    }

    results = {
        "status": "ok",
        "batch": "058", "sub": "c", "brief": "brief_v2.md (v2.1)",
        "wall_s": time.time() - t0,
        "smoke": bool(args.smoke),
        "disorders": disorders_to_run,
        "gene_set_mappings": {
            "EDT1_FULL": edt1_full_map,
            "EDT1_ex_B3": edt1_ex_b3_map,
            "Koopmans_ex_B3_ex_EDT1": koop_map,
        },
        "primary_EDT1_ex_B3": primary_edt1_ex_b3,
        "secondary_EDT1_FULL": secondary_edt1_full,
        "secondary_Koopmans_ex_B3_ex_EDT1": secondary_koop,
        "reproduction_gate_R1_sub_c": repro_r1,
        "provenance_sha256": provenance,
        "brief_contract": {
            "bh_fdr_family_size": 8,
            "bh_q": BH_Q,
            "sub_c_min_effect": SUB_C_MIN_EFFECT,
            "mapping_gate": SUB_C_MAPPING_GATE,
            "fragile_uninterpretable_count": SUB_C_FRAGILE_UNINTERPRETABLE_COUNT,
            "dfbetas_cutoff": DFBETAS_CUTOFF,
        },
    }
    atomic_write_json(results, out_dir / "results.json")
    logger.info("Sub-C wrote %s (wall=%.1fs)",
                 out_dir / "results.json", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
