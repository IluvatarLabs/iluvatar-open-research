#!/usr/bin/env python3
"""
batch_051_B: parse 7 S-LDSC runs, compute Holm correction and CIs, and
emit the deliverable JSON + flat neuronal TSV.

WHY this design:
- Primary metric is τ*-z on the neuronal row (brief MEASUREMENT section).
- --print-coefficients emits Coefficient, Coefficient_std_error, Coefficient_z-score
  per category in the .results file. We also read Enrichment, Enrichment_std_error,
  Enrichment_p to compute 95% CIs (Enr ± 1.96·SE).
- Holm correction is applied twice: within-disorder across 4 cell types (τ*-p),
  and cross-disorder (6 disorders, AD_primary = AD_noAPOE) on neuronal τ*-p.
- τ*-p and τ*-z conversion: LDSC prints the signed z-score; we derive a one-sided
  p from z (upper tail) — the biological prior is positive enrichment.

Caveats baked into output per brief R5/R6:
  * AD on a 4-cell annotation lacking microglia is descriptive, not specific.
  * BIP is coverage_flag=True if post-merge SNPs < 700k.
  * M_eff not computed; Holm is conservative.
"""
from __future__ import annotations

import gzip
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
OUT_DIR = ROOT / "experiments/batch_051_B/output"
LOG_DIR = ROOT / "experiments/batch_051_B/logs"

# tag -> (display_name, sumstats_path, citation_short)
DISORDERS: list[tuple[str, str, str, str]] = [
    ("scz",       "SCZ",       str(ROOT / "data/ldsc/PGC3_sumstats/PGC3_EUR_v2.sumstats.gz"),
        "Trubetskoy et al. 2022, Nature, PGC3 EUR SCZ (doi:10.1038/s41586-022-04434-5)"),
    ("mdd",       "MDD",       str(ROOT / "data/ldsc/comparator_sumstats/mdd.sumstats.gz"),
        "Wray et al. 2018 / Howard et al. 2019 PGC MDD (doi:10.1038/s41588-018-0090-3)"),
    ("bip",       "BIP",       str(ROOT / "data/ldsc/comparator_sumstats/bip.sumstats.gz"),
        "Mullins et al. 2021 PGC BIP (doi:10.1038/s41588-021-00857-4)"),
    ("asd",       "ASD",       str(ROOT / "data/ldsc/comparator_sumstats/asd.sumstats.gz"),
        "Grove et al. 2019 iPSYCH/PGC ASD (doi:10.1038/s41588-019-0344-8)"),
    ("adhd",      "ADHD",      str(ROOT / "data/ldsc/comparator_sumstats/adhd.sumstats.gz"),
        "Demontis et al. 2023 iPSYCH/PGC ADHD (doi:10.1038/s41588-022-01285-8)"),
    ("ad_full",   "AD_full",   str(ROOT / "data/ldsc/comparator_sumstats/alzheimers.sumstats.gz"),
        "Wightman et al. 2021 AD proxy-GWAS (doi:10.1038/s41588-021-00921-z)"),
    ("ad_noapoe", "AD_noAPOE", str(ROOT / "data/ldsc/comparator_sumstats/alzheimers_noAPOE.sumstats.gz"),
        "Wightman et al. 2021 AD proxy-GWAS, chr19:44411941-46386942 excluded (APOE)"),
]

CELL_TYPES = ["neuronal", "oligodendrocyte", "astrocyte", "OPC"]
ROW_NAMES = {
    "neuronal":        "neuronalL2_0",
    "oligodendrocyte": "oligodendrocyteL2_0",
    "astrocyte":       "astrocyteL2_0",
    "OPC":             "OPCL2_0",
}

COVERAGE_FLAG_MIN_SNPS = 700_000  # brief R3


def normal_upper_p(z: float) -> float:
    """One-sided upper-tail p from z (biological prior: positive enrichment)."""
    if math.isnan(z):
        return float("nan")
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down, return adjusted p in the original order.
    NaN p's are preserved as NaN and excluded from counting."""
    idx = list(range(len(pvals)))
    good = [i for i in idx if not math.isnan(pvals[i])]
    m = len(good)
    sorted_good = sorted(good, key=lambda i: pvals[i])
    adj: list[float] = [float("nan")] * len(pvals)
    running_max = 0.0
    for rank, i in enumerate(sorted_good):
        raw = (m - rank) * pvals[i]
        raw = min(1.0, raw)
        running_max = max(running_max, raw)
        adj[i] = running_max
    return adj


def parse_results_file(path: Path) -> dict[str, dict[str, float]]:
    """Parse an LDSC .results table. Return {row_name: {col: value}}."""
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        out: dict[str, dict[str, float]] = {}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            name = parts[0]
            row: dict[str, float] = {}
            for c, v in zip(header[1:], parts[1:]):
                try:
                    row[c] = float(v)
                except ValueError:
                    row[c] = float("nan")
            out[name] = row
    return out


def parse_log_file(path: Path) -> dict[str, Any]:
    """Extract SNP counts, h2, lambda_GC, intercept from an LDSC .log."""
    out: dict[str, Any] = {
        "sumstats_snps": None,
        "ref_ld_snps": None,
        "weight_ld_snps": None,
        "post_merge_snps": None,
        "post_weight_merge_snps": None,
        "lambda_gc": None,
        "mean_chi2": None,
        "intercept": None,
        "intercept_se": None,
        "total_h2": None,
        "total_h2_se": None,
    }
    if not path.exists():
        return out
    text = path.read_text()
    def grab(pat: str, cast=float) -> Any:
        m = re.search(pat, text)
        if not m:
            return None
        try:
            return cast(m.group(1))
        except ValueError:
            return None

    out["sumstats_snps"] = grab(r"Read summary statistics for (\d+) SNPs", int)
    out["ref_ld_snps"] = grab(r"Read reference panel LD Scores for (\d+) SNPs", int)
    out["weight_ld_snps"] = grab(r"Read regression weight LD Scores for (\d+) SNPs", int)
    out["post_merge_snps"] = grab(r"After merging with reference panel LD, (\d+) SNPs remain", int)
    out["post_weight_merge_snps"] = grab(r"After merging with regression SNP LD, (\d+) SNPs remain", int)
    out["lambda_gc"] = grab(r"Lambda GC:\s+([0-9.eE+-]+)", float)
    out["mean_chi2"] = grab(r"Mean Chi\^2:\s+([0-9.eE+-]+)", float)
    m = re.search(r"Intercept:\s+([0-9.eE+-]+)\s+\(([0-9.eE+-]+)\)", text)
    if m:
        out["intercept"] = float(m.group(1))
        out["intercept_se"] = float(m.group(2))
    m = re.search(r"Total Observed scale h2:\s+([0-9.eE+-]+)\s+\(([0-9.eE+-]+)\)", text)
    if m:
        out["total_h2"] = float(m.group(1))
        out["total_h2_se"] = float(m.group(2))
    return out


def count_sumstats_rows(path: str) -> int:
    n = 0
    with gzip.open(path, "rt") as fh:
        fh.readline()  # header
        for _ in fh:
            n += 1
    return n


def extract_celltype_metrics(results: dict[str, dict[str, float]],
                             loginfo: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ct in CELL_TYPES:
        row = results.get(ROW_NAMES[ct], {})
        enr = row.get("Enrichment", float("nan"))
        enr_se = row.get("Enrichment_std_error", float("nan"))
        enr_p = row.get("Enrichment_p", float("nan"))
        prop_snps = row.get("Prop._SNPs", float("nan"))
        prop_h2 = row.get("Prop._h2", float("nan"))
        prop_h2_se = row.get("Prop._h2_std_error", float("nan"))
        tau = row.get("Coefficient", float("nan"))
        tau_se = row.get("Coefficient_std_error", float("nan"))
        tau_z = row.get("Coefficient_z-score", float("nan"))
        # one-sided upper-tail p (biological prior: positive enrichment)
        tau_p = normal_upper_p(tau_z) if not math.isnan(tau_z) else float("nan")
        ci_lo = enr - 1.96 * enr_se if not math.isnan(enr_se) else float("nan")
        ci_hi = enr + 1.96 * enr_se if not math.isnan(enr_se) else float("nan")
        out[ct] = {
            "prop_snps": prop_snps,
            "prop_h2": prop_h2,
            "prop_h2_se": prop_h2_se,
            "enrichment": enr,
            "enrichment_se": enr_se,
            "enrichment_95ci_lo": ci_lo,
            "enrichment_95ci_hi": ci_hi,
            "enrichment_p": enr_p,
            "tau_star": tau,
            "tau_star_se": tau_se,
            "tau_star_z": tau_z,
            "tau_star_p_onesided": tau_p,
        }
    return out


def main() -> int:
    # Verify all 7 runs finished
    missing = []
    runs: dict[str, dict[str, Any]] = {}
    for tag, name, ss, cit in DISORDERS:
        rp = OUT_DIR / f"d_{tag}_celltype_partitioned.results"
        lp = OUT_DIR / f"d_{tag}_celltype_partitioned.log"
        if not rp.exists():
            missing.append(str(rp))
            continue
        res = parse_results_file(rp)
        info = parse_log_file(lp)
        post_merge = info.get("post_weight_merge_snps") or info.get("post_merge_snps")
        coverage_flag = False
        if post_merge is not None and post_merge < COVERAGE_FLAG_MIN_SNPS:
            coverage_flag = True
        cts = extract_celltype_metrics(res, info)
        # Within-disorder Holm across 4 cell types (tau*-p)
        tau_p_list = [cts[c]["tau_star_p_onesided"] for c in CELL_TYPES]
        adj = holm(tau_p_list)
        for c, a in zip(CELL_TYPES, adj):
            cts[c]["tau_star_p_holm_within_disorder"] = a
        munge_rows = count_sumstats_rows(ss)
        runs[name] = {
            "source": ss,
            "citation": cit,
            "sumstats_rows": munge_rows,
            "sumstats_snps_read": info["sumstats_snps"],
            "ref_ld_snps": info["ref_ld_snps"],
            "weight_ld_snps": info["weight_ld_snps"],
            "post_merge_snps": info["post_merge_snps"],
            "post_weight_merge_snps": info["post_weight_merge_snps"],
            "coverage_flag": coverage_flag,
            "lambda_gc": info["lambda_gc"],
            "mean_chi2": info["mean_chi2"],
            "intercept": info["intercept"],
            "intercept_se": info["intercept_se"],
            "total_h2": info["total_h2"],
            "total_h2_se": info["total_h2_se"],
            "cell_types": cts,
        }

    if missing:
        print("MISSING results files:", missing, file=sys.stderr)
        return 2

    # Cross-disorder Holm on neuronal tau*-p across 6 disorders
    # (AD_primary = AD_noAPOE; AD_full reported for transparency)
    primary_order = ["SCZ", "MDD", "BIP", "ASD", "ADHD", "AD_noAPOE"]
    neuronal_p = [runs[d]["cell_types"]["neuronal"]["tau_star_p_onesided"] for d in primary_order]
    neuronal_holm = holm(neuronal_p)
    for d, h in zip(primary_order, neuronal_holm):
        runs[d]["cell_types"]["neuronal"]["tau_star_p_holm_cross_disorder"] = h

    # Replication check against F076 anchor (threshold from brief R5)
    scz_neu_z = runs["SCZ"]["cell_types"]["neuronal"]["tau_star_z"]
    if math.isnan(scz_neu_z):
        repl = {"scz_neuronal_tau_star_z": None,
                "replication_success": None,
                "comment": "SCZ run missing or NaN; cannot evaluate replication"}
    elif scz_neu_z < 1.0:
        repl = {"scz_neuronal_tau_star_z": scz_neu_z,
                "replication_success": False,
                "comment": "PIPELINE REGRESSION: SCZ neuronal tau*-z below halt threshold 1.0. Do not interpret cross-disorder."}
    elif scz_neu_z >= 1.5:
        repl = {"scz_neuronal_tau_star_z": scz_neu_z,
                "replication_success": True,
                "comment": f"Replicates F076 (observed tau*-z={scz_neu_z:.3f} vs F076 reference 1.83; threshold 1.5 per brief R5)."}
    else:
        repl = {"scz_neuronal_tau_star_z": scz_neu_z,
                "replication_success": False,
                "comment": f"Between halt (1.0) and replication (1.5) thresholds — inconclusive replication (z={scz_neu_z:.3f})."}

    # Cross-disorder descriptive tables
    def build_table(ct: str) -> list[dict[str, Any]]:
        order = ["SCZ", "MDD", "BIP", "ASD", "ADHD", "AD_full", "AD_noAPOE"]
        rows: list[dict[str, Any]] = []
        # Cross-disorder Holm only applies across primary_order; for table display
        # we also compute an analogous Holm across 6 primary rows for NON-neuronal
        # cells (keeps symmetry with neuronal table). AD_full omitted from Holm.
        ct_p = [runs[d]["cell_types"][ct]["tau_star_p_onesided"] for d in primary_order]
        ct_holm = holm(ct_p)
        holm_map = dict(zip(primary_order, ct_holm))
        for d in order:
            e = runs[d]["cell_types"][ct]
            rows.append({
                "disorder": d,
                "enrichment": e["enrichment"],
                "enrichment_95ci_lo": e["enrichment_95ci_lo"],
                "enrichment_95ci_hi": e["enrichment_95ci_hi"],
                "enrichment_p": e["enrichment_p"],
                "tau_star": e["tau_star"],
                "tau_star_se": e["tau_star_se"],
                "tau_star_z": e["tau_star_z"],
                "tau_star_p_onesided": e["tau_star_p_onesided"],
                "holm_p_across_6": holm_map.get(d),  # None for AD_full
                "coverage_flag": runs[d]["coverage_flag"],
            })
        return rows

    payload = {
        "metadata": {
            "date": datetime.now(timezone.utc).isoformat(),
            "annotation_source": "batch_034 PanglaoDB 4-cell (neuronal, oligodendrocyte, astrocyte, OPC)",
            "baseline_ld": "baselineLD v2.2",
            "weights": "1000G_Phase3_weights_hm3_no_MHC",
            "frq": "1000G_Phase3_frq (EUR)",
            "ldsc_version": "2.0.0",
            "ldsc_binary": "/home/yuanz/torchml/bin/ldsc.py",
            "n_disorders": 7,
            "primary_cross_disorder_set": primary_order,
            "runs_included": [d[1] for d in DISORDERS],
        },
        "pre_registered_thresholds": {
            "replication_success_scz_tau_z": 1.5,
            "replication_halt_scz_tau_z": 1.0,
            "coverage_flag_min_snps": COVERAGE_FLAG_MIN_SNPS,
            "positive_effect_tau_z": 1.5,
            "null_effect_tau_z": 1.0,
        },
        "disorders": runs,
        "replication_check": repl,
        "cross_disorder_descriptive": {
            "neuronal_table": build_table("neuronal"),
            "oligodendrocyte_table": build_table("oligodendrocyte"),
            "astrocyte_table": build_table("astrocyte"),
            "OPC_table": build_table("OPC"),
        },
        "caveats": {
            "ad_microglia_missing":
                "AD null on PanglaoDB 4-cell (no microglia) is descriptive, not a specific negative control. "
                "The panel cannot discriminate 'AD is microglia-driven' from 'AD signal is in an un-annotated cell type'.",
            "bip_coverage_limit":
                "BIP sumstats has ~497k rows pre-merge; post-HM3 intersection is below the 700k pre-registered coverage gate.",
            "sample_overlap":
                "PGC cohorts share participants across psychiatric disorders; cross-disorder comparison is descriptive only. Cochran Q inference dropped per brief R7.",
            "no_m_eff":
                "M_eff not computed for cell-type annotation correlation; Holm correction is conservative.",
            "ad_provenance":
                "alzheimers.sumstats.gz is LDSC-munged format but contains 12,069,723 rows — not HM3-intersected in advance. "
                "HM3 intersection is performed by LDSC merge; provenance_status = 'raw-unfiltered; HM3 intersection via S-LDSC merge'. "
                "Source matches Wightman 2021 (N=455,258 proxy-AD).",
            "scz_sumstats_substitution":
                "Brief specified PGC3_EUR.sumstats.gz, which lacks a Z column. We used PGC3_EUR_v2.sumstats.gz (same GWAS, has Z). "
                "This is the file used by F076 (the replication anchor), so the replication test is self-consistent.",
        },
    }

    out_json = OUT_DIR / "d_cross_disorder_celltype_sldsc.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out_json}")

    # Flat neuronal TSV for manuscript
    neu_tsv = OUT_DIR / "cross_disorder_neuronal_table.tsv"
    with neu_tsv.open("w") as fh:
        fh.write("disorder\tenrichment\tenr_ci_lo\tenr_ci_hi\tenr_p\t"
                "tau_star\ttau_star_se\ttau_star_z\ttau_star_p_onesided\t"
                "holm_p_across_6\tcoverage_flag\tpost_merge_snps\n")
        for r in payload["cross_disorder_descriptive"]["neuronal_table"]:
            d = r["disorder"]
            fh.write("\t".join(str(x) for x in [
                d, r["enrichment"], r["enrichment_95ci_lo"], r["enrichment_95ci_hi"],
                r["enrichment_p"], r["tau_star"], r["tau_star_se"], r["tau_star_z"],
                r["tau_star_p_onesided"], r["holm_p_across_6"], r["coverage_flag"],
                runs[d]["post_weight_merge_snps"],
            ]) + "\n")
    print(f"Wrote {neu_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
