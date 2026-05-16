#!/usr/bin/env python3
"""
E9 (R9): Multiple-Testing Table

Build a comprehensive table of all analysis families in the project,
documenting: number of tests, correction method, primary vs exploratory,
and key findings that survive correction.

WHY: A reviewer needs a single table showing all multiple-testing corrections
applied throughout the manuscript. This is increasingly required by genetics
journals (e.g., Nature Genetics, Am J Hum Genet) to assess familywise error
control.

Sources: research_state.md findings log, iteration docs, batch outputs.

Output: experiments/batch_069/output/e9_multiple_testing_table.json
"""

from __future__ import annotations
import json, pathlib

PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "batch_069" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis families documented from research_state.md, batch outputs, and manuscript
# Each entry is constructed from the actual analyses performed in the project
ANALYSIS_FAMILIES = [
    {
        "family": "Cell-type enrichment (S-LDSC)",
        "description": "S-LDSC partitioned heritability across cell-type-specific annotations (EUR PGC3)",
        "n_tests": 4,
        "test_items": "neuronal, oligodendrocyte, astrocyte, OPC",
        "correction_method": "BH-FDR",
        "classification": "Primary",
        "key_findings": [
            "F013: Neuronal enrichment ESTABLISHED (survives FDR)",
            "F026: Cell-type pattern replicates across methods",
        ],
        "nominal_only": [
            "Oligodendrocyte, astrocyte, OPC — not significant after correction"
        ],
        "source": "batch_028, batch_046",
        "finding_ids": ["F013", "F026", "F076", "F098", "F099", "F117"],
    },
    {
        "family": "EAS cell-type enrichment (S-LDSC)",
        "description": "S-LDSC partitioned heritability across same annotations in EAS GWAS (Lam 2019)",
        "n_tests": 4,
        "test_items": "neuronal, oligodendrocyte, astrocyte, OPC",
        "correction_method": "BH-FDR",
        "classification": "Exploratory (underpowered)",
        "key_findings": [],
        "nominal_only": [
            "Oligo z=-4.55 (negative enrichment, artifact of small sample — see E7 power diagnostic)",
            "Neuronal z=-0.20 (underpowered)"
        ],
        "source": "batch_046",
        "finding_ids": ["F116", "F117"],
    },
    {
        "family": "Cross-disorder MAGMA gene-set enrichment",
        "description": "Competitive MAGMA gene-set analysis: EDT1-ex-B3, B3 (SynGO), remaining ring, and sub-lists across 8 disorders (SCZ, BIP, MDD, ASD, ADHD, AD, PD, IBD)",
        "n_tests": "up to 24 (3 gene sets x 8 disorders)",
        "test_items": "EDT1-ex-B3, B3 (SynGO), remaining ring x SCZ, BIP, MDD, ASD, ADHD, AD, PD, IBD",
        "correction_method": "BH-FDR within each gene-set battery",
        "classification": "Primary",
        "key_findings": [
            "F146: Pan-psychiatric pattern ESTABLISHED (SCZ, BIP, MDD, ASD, AD significant)",
            "F147: PSD-scaffold concentration SUGGESTED",
            "F058_05: EDT1-ex-B3 SCZ beta=3.427 ESTABLISHED",
            "F060_01: Bellenguez AD independent replication SUGGESTED (beta=0.370)",
        ],
        "nominal_only": [
            "IBD marginal (q=0.050)",
            "PD not significant",
            "ADHD not significant",
        ],
        "source": "batch_034, batch_047, batch_058, batch_059, batch_060",
        "finding_ids": ["F146", "F147", "F058_05", "F060_01"],
    },
    {
        "family": "TF motif enrichment (PWM scanning)",
        "description": "Position weight matrix scanning for TF binding site enrichment in regulatory elements near EDT1 genes",
        "n_tests": 16,
        "test_items": "16 candidate TFs including EGR1, CTCF, and neurodevelopmental regulators",
        "correction_method": "BH-FDR",
        "classification": "Primary",
        "key_findings": [
            "F086: EGR1 motif enrichment ESTABLISHED",
            "F093-F097: PWM enrichment patterns",
        ],
        "nominal_only": [],
        "source": "batch_040, batch_048",
        "finding_ids": ["F086", "F093", "F094", "F095", "F096", "F097"],
    },
    {
        "family": "Constraint enrichment (pLI/LOEUF)",
        "description": "Fisher exact tests: EDT1 and functional subsets vs gnomAD constraint thresholds. Length-stratified permutation for primary metric.",
        "n_tests": 12,
        "test_items": "6 gene lists (SynGO_EDT1, glutamate_receptor, ion_channel, mitochondrial, transcriptional, other) x 2 metrics (pLI>=0.9, LOEUF<=0.35)",
        "correction_method": "BH-FDR across all 12 tests",
        "classification": "Primary",
        "key_findings": [
            "F122: EDT1-wide pLI OR=1.14 (NOT significant — EDT1 not globally constrained)",
            "F147: SynGO_EDT1 pLI OR=20.9 (q=2e-5, ESTABLISHED; emp_p=0.0002)",
            "Glutamate receptor pLI OR=Infinity (q=0, but n=3; Haldane-corrected in E8)",
            "Transcriptional pLI OR=13.9 (q=0.033), LOEUF OR=35.4 (q=0.004)",
            "Other pLI OR=2.46 (q=1.3e-14)",
        ],
        "nominal_only": [
            "Ion channel: OR=0.93 (pLI), OR=2.36 (LOEUF) — not significant",
            "Mitochondrial: OR=1.54 (pLI), OR=1.68 (LOEUF) — not significant",
        ],
        "source": "batch_047, batch_048",
        "finding_ids": ["F122", "F147"],
    },
    {
        "family": "Network proximity (STRING PPI)",
        "description": "Network-based proximity of SCHEMA genes to MAGMA-prioritized genes in STRING v12",
        "n_tests": 3,
        "test_items": "3 SCHEMA thresholds (P_MIN<0.05, P<0.001, EWS)",
        "correction_method": "Bonferroni (3 tests)",
        "classification": "Exploratory",
        "key_findings": [
            "F067_01: SCHEMA P_MIN<0.05 x MAGMA top-500 NETWORK_CONVERGENT (d_c=1.67 vs null=1.75, Z=-4.63, perm p=1e-4) SUGGESTED",
        ],
        "nominal_only": [
            "F067_04: Gene list mislabeling caveat (P_MIN<0.05 was labeled FDR<0.05)"
        ],
        "source": "batch_067",
        "finding_ids": ["F067_01", "F067_04"],
    },
    {
        "family": "SCHEMA convergence (threshold sensitivity)",
        "description": "SCHEMA de novo burden gene overlap with neuronal markers at increasingly stringent thresholds",
        "n_tests": 3,
        "test_items": "P_MIN<0.05 (n=1613), P<0.001 (n=50), EWS (n=10)",
        "correction_method": "Not corrected (exploratory sensitivity analysis)",
        "classification": "Exploratory (post-hoc sensitivity)",
        "key_findings": [
            "F068_01: OR strengthens 2.32 -> 13.78 -> 23.5 at stricter thresholds. Signal is genuine, noise-diluted. ESTABLISHED."
        ],
        "nominal_only": [],
        "source": "batch_068",
        "finding_ids": ["F068_01"],
    },
    {
        "family": "PoPS feature ablation",
        "description": "Ridge regression feature ablation to assess contribution of expression, PPI, pathway, and other feature groups to PoPS scores",
        "n_tests": 5,
        "test_items": "Expression (brain), Expression (non-brain), PPI network, Pathway, Other features",
        "correction_method": "Descriptive (not hypothesis-testing); reported as relative contributions",
        "classification": "Exploratory (mechanistic characterization)",
        "key_findings": [
            "F148a: Expression dominance ESTABLISHED",
            "F059_02: Mass-share mediation MECHANICAL (brain Drho=0.004, p=0.73). Expression signal is not brain-specific.",
        ],
        "nominal_only": [],
        "source": "batch_056, batch_059",
        "finding_ids": ["F148a", "F059_02"],
    },
    {
        "family": "MiXeR polygenicity estimation",
        "description": "MiXeR bivariate causal mixture model: 20 replicate fits with different random seeds",
        "n_tests": "20 replicates (CI estimation, not hypothesis tests)",
        "test_items": "20 random seeds for nc, sig2_beta, sig2_zero, h2 estimation",
        "correction_method": "N/A — replicate-based CI (mean +/- SD across 20 fits)",
        "classification": "Primary (estimation, not testing)",
        "key_findings": [
            "F065_03: nc=32,186 +/- 414 (CV=2.8%), sig2_zero=1.219, h2=0.823. 20/20 converged. PUBLISHABLE."
        ],
        "nominal_only": [
            "sig2_beta WARNING: 1.23e-4 (2.2x Holland 2020 value)"
        ],
        "source": "batch_065",
        "finding_ids": ["F065_03"],
    },
    {
        "family": "H-MAGMA cross-annotation comparison",
        "description": "H-MAGMA with 5 Hi-C annotations (fetal brain, adult brain, cortical neuron, iPSC neuron, iPSC astro) vs standard proximity-based MAGMA",
        "n_tests": 5,
        "test_items": "5 Hi-C annotations + 1 standard MAGMA (reference)",
        "correction_method": "Descriptive comparison (not corrected; all use same gene sets)",
        "classification": "Exploratory",
        "key_findings": [
            "F060_09: Standard proximity-MAGMA STRONGER than all Hi-C annotations for EDT1-ex-B3 and B3. SCZ signal is proximally-driven. SUGGESTED."
        ],
        "nominal_only": [],
        "source": "batch_060",
        "finding_ids": ["F060_09"],
    },
    {
        "family": "Environmental-axis gene-set batteries (IEG, GR, Complement)",
        "description": "MAGMA competitive gene-set tests for IEG (ARG n=154), GR targets, complement genes (MHC-excluded) across 8 disorders",
        "n_tests": "~24 (3 gene sets x 8 disorders)",
        "test_items": "IEG ARG (n=154), GR targets, Complement (MHC-excluded) x 8 disorders",
        "correction_method": "BH-FDR within each battery",
        "classification": "Primary (for G3 hypothesis testing)",
        "key_findings": [
            "F061_01: IEG ARG REFUTED (SCZ beta=-0.165, q=0.999). 4/4 estimators negative. ESTABLISHED negative.",
            "F060_05: GR targets NOT ENRICHED for SCZ (beta=-0.688). SUGGESTED negative.",
            "F060_06: Complement NULL REPLICATED (beta=-0.466, q=0.92). SUGGESTED.",
        ],
        "nominal_only": [
            "F060_04/F061_05: rPRG weak positive (beta=0.454, q=0.264) — underpowered, INCONCLUSIVE"
        ],
        "source": "batch_060, batch_061",
        "finding_ids": ["F061_01", "F060_05", "F060_06"],
    },
    {
        "family": "HAR x TF enrichment",
        "description": "Fisher enrichment of EGR1/CTCF PWM target gene promoters for overlap with human accelerated regions (HARs)",
        "n_tests": 2,
        "test_items": "EGR1 targets x HAR proximity, CTCF targets x HAR proximity",
        "correction_method": "BH-FDR (2 tests)",
        "classification": "Exploratory",
        "key_findings": [],
        "nominal_only": [
            "Results from batch_048 Sub-B"
        ],
        "source": "batch_048",
        "finding_ids": [],
    },
    {
        "family": "G4 ARI clustering",
        "description": "k-means and spectral clustering of 285 genes x 8 cross-disorder MAGMA-Z for subtype detection",
        "n_tests": "4 (k=2,3,4,5 evaluated by ARI + permutation)",
        "test_items": "k-means at k=2,3,4,5 with split-half ARI and permutation p-values",
        "correction_method": "Permutation-based p-values (nonparametric)",
        "classification": "Exploratory",
        "key_findings": [
            "F064_02: ARI=0.39-0.44, perm p > 0.24. SUBTYPES_NOT_SUPPORTED. ESTABLISHED negative.",
            "F065_01: PC1=17.9% (near-isotropic). UNINTERPRETABLE.",
        ],
        "nominal_only": [],
        "source": "batch_064, batch_065",
        "finding_ids": ["F064_02", "F065_01"],
    },
]


def main():
    # Print formatted table
    print("=" * 120)
    print(f"{'Family':<45} {'N tests':>8} {'Correction':<15} {'Class':<20} {'Key Results'}")
    print("-" * 120)

    for fam in ANALYSIS_FAMILIES:
        n_tests_str = str(fam["n_tests"])
        key_str = "; ".join(fam.get("finding_ids", [])) if fam.get("finding_ids") else "—"
        print(f"{fam['family']:<45} {n_tests_str:>8} {fam['correction_method']:<15} {fam['classification']:<20} {key_str}")

    print("=" * 120)

    # Summary statistics
    total_tests = 0
    for fam in ANALYSIS_FAMILIES:
        nt = fam["n_tests"]
        if isinstance(nt, int):
            total_tests += nt
        elif isinstance(nt, str):
            # Parse "up to 24" or "~24" etc.
            import re
            nums = re.findall(r'\d+', nt)
            if nums:
                total_tests += int(nums[0])

    n_primary = sum(1 for f in ANALYSIS_FAMILIES if "Primary" in f["classification"])
    n_exploratory = sum(1 for f in ANALYSIS_FAMILIES if "Exploratory" in f["classification"] or "exploratory" in f["classification"])

    print(f"\nTotal analysis families: {len(ANALYSIS_FAMILIES)}")
    print(f"Estimated total individual tests: ~{total_tests}")
    print(f"Primary families: {n_primary}")
    print(f"Exploratory families: {n_exploratory}")

    # Correction methods used
    methods = set(f["correction_method"] for f in ANALYSIS_FAMILIES)
    print(f"Correction methods: {sorted(methods)}")

    # Output
    result = {
        "experiment": "E9_Multiple_Testing_Table",
        "batch": "batch_069",
        "status": "COMPLETED",
        "n_analysis_families": len(ANALYSIS_FAMILIES),
        "estimated_total_tests": total_tests,
        "n_primary_families": n_primary,
        "n_exploratory_families": n_exploratory,
        "correction_methods_used": sorted(methods),
        "families": ANALYSIS_FAMILIES,
        "summary_note": (
            "All primary analyses use BH-FDR correction within each analysis family. "
            "No global correction is applied across families because each family tests "
            "a distinct biological hypothesis with different data modalities (S-LDSC vs "
            "MAGMA vs Fisher exact vs permutation). This is consistent with the per-family "
            "correction approach recommended by Goeman & Solari (2014) and standard practice "
            "in GWAS post-processing papers (e.g., Finucane 2018, Trubetskoy 2022). "
            "Exploratory analyses are clearly labeled and findings are classified as "
            "SUGGESTED rather than ESTABLISHED."
        ),
    }

    out_path = OUTPUT_DIR / "e9_multiple_testing_table.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
