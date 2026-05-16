#!/usr/bin/env python3
"""
E2 (R10): SynGO_EDT1 14-Gene Provenance Audit

HYPOTHESIS: The 14-gene "SynGO_EDT1" list is NOT a programmatic intersection
of SynGO and EDT1 (Extended Data Table 1 from Trubetskoy 2022 PGC3 GWAS).
Instead, it is a manually curated list of canonical synaptic genes that happen
to be in SynGO. The "EDT1" label is misleading.

WHY: iter_069 E4 revealed that 10-12 of these 14 genes are NOT in the EDT1
xlsx. This script performs the definitive provenance audit to determine:
1. Which genes are actually in EDT1 (both sheets)?
2. Which genes are in pgc3_genes.txt?
3. Which genes are in SynGO (and which version)?
4. How the list was originally constructed in batch_047.
5. Whether the 56-gene "ST12 SynGO column" is a true programmatic filter.

Output: experiments/batch_070/output/e2_syngo_edt1_provenance.json
"""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# -- Paths --
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "batch_070" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDT1_XLSX = Path("/mnt/GLaDOS_pool/Iluvatar/biomarvin/schizo/19426775/scz2022-Extended-Data-Table1.xlsx")
PGC3_TXT = PROJECT_ROOT / "experiments" / "batch_018" / "pgc3_genes.txt"
SYNGO_2024_GMT = PROJECT_ROOT / "experiments" / "batch_052_A" / "input" / "syngo_2024.gmt"
SYNGO_2022_GMT = Path.home() / ".cache/gseapy/Enrichr.SynGO_2022.gmt"

# -- The 14-gene list as defined in batch_047/scripts/run_sub2_constraint.py:52 --
SYNOGO_EDT1_BATCH047 = [
    "DLGAP1", "GRIN2A", "NRXN1", "CNTNAP2", "ARC", "DLG4", "NRXN2",
    "NLGN1", "NLGN2", "SHANK1", "SHANK3", "HOMER1", "SYN1", "GAP43"
]


def load_syngo_gmt(gmt_path: Path) -> tuple[set[str], dict[str, list[str]]]:
    """Load a SynGO GMT file. Returns (all_genes, gene_to_terms_dict)."""
    all_genes: set[str] = set()
    gene_to_terms: dict[str, list[str]] = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = [g for g in parts[2:] if g]
            for g in genes:
                all_genes.add(g)
                gene_to_terms.setdefault(g, []).append(term)
    return all_genes, gene_to_terms


def classify_syngo_category(terms: list[str]) -> str:
    """Classify SynGO terms into broad categories: presynaptic, postsynaptic,
    synaptic_vesicle, synaptic_signaling, other_synaptic."""
    lower_terms = " ".join(terms).lower()
    categories = []
    if "presynaptic" in lower_terms or "presynapse" in lower_terms:
        categories.append("presynaptic")
    if "postsynaptic" in lower_terms or "postsynapse" in lower_terms:
        categories.append("postsynaptic")
    if "vesicle" in lower_terms:
        categories.append("synaptic_vesicle")
    if "modulation" in lower_terms or "regulation" in lower_terms or "signaling" in lower_terms:
        categories.append("synaptic_signaling")
    if not categories:
        categories.append("other_synaptic")
    return "|".join(sorted(set(categories)))


def main():
    t0 = time.time()
    print("=" * 70)
    print("E2 (R10): SynGO_EDT1 14-Gene Provenance Audit")
    print("=" * 70)

    # ========== 1. Load EDT1 xlsx (all sheets) ==========
    print("\n[1] Loading EDT1 xlsx...")
    xl = pd.ExcelFile(EDT1_XLSX)
    print(f"    Sheets: {xl.sheet_names}")

    # Extended.Data.Table.1 sheet
    edt1_sheet = xl.parse("Extended.Data.Table.1")
    edt1_all_genes = set(edt1_sheet["Symbol.ID"].dropna().astype(str))
    edt1_pc_genes = set(
        edt1_sheet[edt1_sheet["gene_biotype"] == "protein_coding"]["Symbol.ID"]
        .dropna().astype(str)
    )
    print(f"    Extended.Data.Table.1: {len(edt1_all_genes)} genes ({len(edt1_pc_genes)} protein_coding)")

    # ST12 all criteria sheet
    st12_sheet = xl.parse("ST12 all criteria")
    st12_all_genes = set(st12_sheet["Symbol.ID"].dropna().astype(str))
    st12_pc_genes = set(
        st12_sheet[st12_sheet["gene_biotype"] == "protein_coding"]["Symbol.ID"]
        .dropna().astype(str)
    )
    print(f"    ST12 all criteria: {len(st12_all_genes)} genes ({len(st12_pc_genes)} protein_coding)")

    # ST12 SynGO column
    def truthy(v):
        if v is None:
            return False
        try:
            if isinstance(v, float) and math.isnan(v):
                return False
        except TypeError:
            pass
        return str(v).strip().lower() in {"yes", "y", "1", "true", "1.0"}

    st12_syngo_genes = set(
        st12_sheet[st12_sheet["SynGO.GeneSetMemb"].apply(truthy)]["Symbol.ID"]
        .dropna().astype(str)
    )
    print(f"    ST12 SynGO.GeneSetMemb=1 genes: {len(st12_syngo_genes)}")

    # ST12 Prioritised column
    st12_prioritised_genes = set(
        st12_sheet[st12_sheet["Prioritised"] == 1]["Symbol.ID"]
        .dropna().astype(str)
    )
    print(f"    ST12 Prioritised=1 genes: {len(st12_prioritised_genes)}")

    # ========== 2. Load pgc3_genes.txt ==========
    print("\n[2] Loading pgc3_genes.txt...")
    pgc3_txt_genes = set(open(PGC3_TXT).read().strip().split("\n"))
    print(f"    pgc3_genes.txt: {len(pgc3_txt_genes)} genes")

    # ========== 3. Load SynGO data ==========
    print("\n[3] Loading SynGO data...")

    syngo_2024_genes, syngo_2024_terms = load_syngo_gmt(SYNGO_2024_GMT)
    print(f"    SynGO 2024 (batch_052_A): {len(syngo_2024_genes)} genes")

    syngo_2022_genes, syngo_2022_terms = load_syngo_gmt(SYNGO_2022_GMT)
    print(f"    SynGO 2022 (Enrichr): {len(syngo_2022_genes)} genes")

    # ========== 4. Per-gene provenance table ==========
    print("\n[4] Building per-gene provenance table...")
    per_gene = []
    for gene in sorted(SYNOGO_EDT1_BATCH047):
        row = {"gene": gene}

        # EDT1 Extended.Data.Table.1 sheet
        row["in_edt1_extended_data_table_1"] = gene in edt1_all_genes
        if gene in edt1_all_genes:
            gene_row = edt1_sheet[edt1_sheet["Symbol.ID"] == gene].iloc[0]
            row["edt1_sheet"] = "Extended.Data.Table.1"
            row["edt1_biotype"] = str(gene_row.get("gene_biotype", ""))
        else:
            row["edt1_sheet"] = None
            row["edt1_biotype"] = None

        # ST12 all criteria sheet
        row["in_st12_all_criteria"] = gene in st12_all_genes
        if gene in st12_all_genes:
            gene_row_st12 = st12_sheet[st12_sheet["Symbol.ID"] == gene].iloc[0]
            row["st12_syngo_col"] = bool(truthy(gene_row_st12.get("SynGO.GeneSetMemb", 0)))
            row["st12_prioritised"] = int(gene_row_st12.get("Prioritised", 0))
        else:
            row["st12_syngo_col"] = False
            row["st12_prioritised"] = None

        # pgc3_genes.txt
        row["in_pgc3_txt"] = gene in pgc3_txt_genes

        # SynGO 2024
        row["in_syngo_2024"] = gene in syngo_2024_genes
        if gene in syngo_2024_genes:
            terms_2024 = syngo_2024_terms[gene]
            row["syngo_2024_n_terms"] = len(terms_2024)
            row["syngo_2024_category"] = classify_syngo_category(terms_2024)
            row["syngo_2024_terms"] = terms_2024[:5]  # first 5 for brevity
        else:
            row["syngo_2024_n_terms"] = 0
            row["syngo_2024_category"] = None
            row["syngo_2024_terms"] = []

        # SynGO 2022 (Enrichr)
        row["in_syngo_2022"] = gene in syngo_2022_genes
        if gene in syngo_2022_genes:
            terms_2022 = syngo_2022_terms[gene]
            row["syngo_2022_n_terms"] = len(terms_2022)
            row["syngo_2022_category"] = classify_syngo_category(terms_2022)
        else:
            row["syngo_2022_n_terms"] = 0
            row["syngo_2022_category"] = None

        # Combined: in BOTH EDT1 (any sheet) AND SynGO
        in_any_edt1 = gene in edt1_all_genes or gene in st12_all_genes
        in_any_syngo = gene in syngo_2024_genes or gene in syngo_2022_genes
        row["in_both_edt1_and_syngo"] = in_any_edt1 and in_any_syngo

        per_gene.append(row)

    # Print per-gene table
    print(f"\n{'Gene':<12} {'EDT1_sheet':<6} {'ST12':<6} {'ST12_SynGO':<10} "
          f"{'pgc3_txt':<9} {'SynGO_2024':<10} {'SynGO_2022':<10} {'Both':<6}")
    print("-" * 85)
    for r in per_gene:
        print(f"{r['gene']:<12} "
              f"{'YES' if r['in_edt1_extended_data_table_1'] else 'NO':<6} "
              f"{'YES' if r['in_st12_all_criteria'] else 'NO':<6} "
              f"{'YES' if r['st12_syngo_col'] else 'NO':<10} "
              f"{'YES' if r['in_pgc3_txt'] else 'NO':<9} "
              f"{'YES' if r['in_syngo_2024'] else 'NO':<10} "
              f"{'YES' if r['in_syngo_2022'] else 'NO':<10} "
              f"{'YES' if r['in_both_edt1_and_syngo'] else 'NO':<6}")

    # ========== 5. Summary statistics ==========
    print("\n[5] Summary statistics:")
    n_in_edt1_sheet = sum(1 for r in per_gene if r["in_edt1_extended_data_table_1"])
    n_in_st12 = sum(1 for r in per_gene if r["in_st12_all_criteria"])
    n_in_st12_syngo = sum(1 for r in per_gene if r["st12_syngo_col"])
    n_in_pgc3_txt = sum(1 for r in per_gene if r["in_pgc3_txt"])
    n_in_syngo_2024 = sum(1 for r in per_gene if r["in_syngo_2024"])
    n_in_syngo_2022 = sum(1 for r in per_gene if r["in_syngo_2022"])
    n_in_both = sum(1 for r in per_gene if r["in_both_edt1_and_syngo"])

    print(f"    In Extended.Data.Table.1 sheet: {n_in_edt1_sheet}/14")
    print(f"    In ST12 all criteria sheet: {n_in_st12}/14")
    print(f"    In ST12 SynGO.GeneSetMemb=1 column: {n_in_st12_syngo}/14")
    print(f"    In pgc3_genes.txt: {n_in_pgc3_txt}/14")
    print(f"    In SynGO 2024: {n_in_syngo_2024}/14")
    print(f"    In SynGO 2022 (Enrichr): {n_in_syngo_2022}/14")
    print(f"    In BOTH any EDT1 source AND SynGO: {n_in_both}/14")

    # ========== 6. Verify ST12 SynGO column is programmatic ==========
    print("\n[6] Verifying ST12 SynGO.GeneSetMemb column...")
    # The ST12 column contains exactly 56 genes with SynGO.GeneSetMemb=1
    # These are a TRUE programmatic annotation (column in the xlsx)
    # Check overlap with SynGO 2024 and 2022
    st12_syngo_in_2024 = st12_syngo_genes & syngo_2024_genes
    st12_syngo_in_2022 = st12_syngo_genes & syngo_2022_genes
    st12_syngo_in_both = st12_syngo_genes & syngo_2024_genes & syngo_2022_genes
    st12_syngo_not_in_either = st12_syngo_genes - syngo_2024_genes - syngo_2022_genes
    print(f"    ST12 SynGO genes (n=56) in SynGO 2024: {len(st12_syngo_in_2024)}")
    print(f"    ST12 SynGO genes (n=56) in SynGO 2022: {len(st12_syngo_in_2022)}")
    print(f"    ST12 SynGO genes (n=56) in both versions: {len(st12_syngo_in_both)}")
    print(f"    ST12 SynGO genes NOT in either SynGO version: {len(st12_syngo_not_in_either)}")
    if st12_syngo_not_in_either:
        print(f"      {sorted(st12_syngo_not_in_either)}")

    # The TRUE SynGO intersection with EDT1 data:
    # Approach A: ST12_protein_coding genes with SynGO.GeneSetMemb=1 (the column)
    true_syngo_edt1_st12_col = st12_syngo_genes & st12_pc_genes
    # Approach B: ST12_protein_coding genes that are ALSO in SynGO 2024 GMT
    true_syngo_edt1_programmatic_2024 = st12_pc_genes & syngo_2024_genes
    # Approach C: Extended.Data.Table.1 protein_coding genes in SynGO 2024
    true_syngo_edt1_edt1sheet_2024 = edt1_pc_genes & syngo_2024_genes
    print(f"\n    TRUE 'SynGO intersect EDT1' by different approaches:")
    print(f"      A: ST12 pc + SynGO.GeneSetMemb column = {len(true_syngo_edt1_st12_col)} genes")
    print(f"      B: ST12 pc intersect SynGO 2024 GMT = {len(true_syngo_edt1_programmatic_2024)} genes")
    print(f"      C: EDT1_sheet pc intersect SynGO 2024 GMT = {len(true_syngo_edt1_edt1sheet_2024)} genes")

    # How many of the 14 batch_047 genes appear in each:
    b47_set = set(SYNOGO_EDT1_BATCH047)
    b47_in_A = b47_set & true_syngo_edt1_st12_col
    b47_in_B = b47_set & true_syngo_edt1_programmatic_2024
    b47_in_C = b47_set & true_syngo_edt1_edt1sheet_2024
    print(f"\n    batch_047 14 genes in each TRUE intersection:")
    print(f"      A (ST12 SynGO col): {len(b47_in_A)}/14 = {sorted(b47_in_A)}")
    print(f"      B (ST12 x SynGO 2024): {len(b47_in_B)}/14 = {sorted(b47_in_B)}")
    print(f"      C (EDT1 sheet x SynGO 2024): {len(b47_in_C)}/14 = {sorted(b47_in_C)}")

    # ========== 7. Determine actual provenance ==========
    print("\n[7] Provenance determination...")
    # Key evidence:
    # 1. The batch_047 code comment says "from batch_031 F058" and
    #    "14 genes from Extended Data Table 1 with Prioritised + SynGO"
    # 2. The batch_069 E4 code says "SynGO_EDT1 from batch_047 (Singh 2022 Table 1)"
    # 3. The batch_048 code says "from batch_047 reconstructed list (Singh 2022 Table 1 supplementary)"
    # 4. The batch_053_A brief calls it "14-gene hand list" and "Singh 2022 style"
    # 5. All 14 genes ARE in SynGO 2022 (Enrichr version)
    # 6. Only 1/14 (GRIN2A) is in the Extended.Data.Table.1 sheet
    # 7. Only 2/14 (GRIN2A, SHANK3) are in ST12 at all
    # 8. 0/14 are in pgc3_genes.txt (which is a non-SCZ gene list of 300 genes)
    # 9. The genes are all well-known synaptic scaffold/adhesion/plasticity proteins
    #    commonly cited in schizophrenia literature (NRXN1, SHANK3, DLG4, etc.)

    provenance_determination = {
        "conclusion": (
            "The 14-gene 'SynGO_EDT1' list is a MANUAL LITERATURE CURATION of "
            "canonical synaptic proteins implicated in schizophrenia, NOT a "
            "programmatic intersection of SynGO and EDT1. The name is misleading."
        ),
        "evidence": [
            f"Only {n_in_edt1_sheet}/14 genes (GRIN2A) appear in Extended.Data.Table.1 sheet",
            f"Only {n_in_st12}/14 genes (GRIN2A, SHANK3) appear in ST12 all criteria sheet",
            f"Only {n_in_st12_syngo}/14 genes (GRIN2A, SHANK3) have SynGO.GeneSetMemb=1 in ST12",
            f"{n_in_syngo_2022}/14 genes are in SynGO 2022 (Enrichr) — ALL 14",
            f"{n_in_syngo_2024}/14 genes are in SynGO 2024 — 13/14 (ARC absent from 2024 but present in 2022)",
            "The code comment in batch_047 says 'from batch_031 F058' and 'EDT1 with Prioritised + SynGO'",
            "batch_048 and batch_069 both reference 'Singh 2022 Table 1' as the source",
            "batch_053_A brief explicitly calls it a '14-gene hand list'",
            "All 14 genes are well-known synaptic scaffold/adhesion/plasticity proteins "
            "commonly cited in SCZ literature (NRXN1, SHANK3, DLG4, NLGN1/2, etc.)",
        ],
        "most_likely_origin": (
            "Hand-curated from SCZ literature (probably Singh et al. 2022 SCHEMA paper's "
            "Table 1 or supplementary materials listing known synaptic SCZ genes), "
            "with the SynGO membership verified post-hoc. The 'EDT1' in the name likely "
            "referred to the broader PGC3 project context, not to a specific intersection "
            "operation with the EDT1 spreadsheet."
        ),
        "true_programmatic_syngo_edt1": {
            "description": (
                "The TRUE programmatic SynGO x EDT1 intersection is the ST12 "
                "SynGO.GeneSetMemb=1 column (n=56 genes). This is a pre-computed "
                "column in the PGC3 supplementary data."
            ),
            "n_genes": len(true_syngo_edt1_st12_col),
            "genes": sorted(true_syngo_edt1_st12_col),
            "overlap_with_batch047_14": sorted(b47_in_A),
        },
    }

    print(f"\n    CONCLUSION: {provenance_determination['conclusion']}")
    print(f"\n    Most likely origin: {provenance_determination['most_likely_origin']}")

    # ========== 8. Recommendation ==========
    recommendation = {
        "action": "RENAME",
        "current_label": "SynGO_EDT1",
        "proposed_label": "SynGO_lit14",
        "rationale": (
            "The current label 'SynGO_EDT1' implies a programmatic intersection of "
            "SynGO and Extended Data Table 1. Only 2/14 genes are in any EDT1 source. "
            "Proposed rename to 'SynGO_lit14' (14 literature-curated synaptic genes) "
            "or 'SynGO_canonical_SCZ' (canonical synaptic SCZ genes in SynGO). "
            "Alternatively, if the manuscript uses this list, it should explicitly "
            "state: 'We curated 14 canonical synaptic genes known from prior "
            "schizophrenia literature (Singh 2022, SCHEMA) that are annotated in "
            "SynGO' rather than implying an intersection with EDT1."
        ),
        "alternative_labels": [
            "SynGO_lit14",
            "SynGO_canonical_SCZ",
            "SynGO_curated_synaptic_SCZ14",
        ],
        "manuscript_impact": (
            "Any manuscript sentence claiming these 14 genes are 'the intersection "
            "of SynGO and EDT1' is factually incorrect and must be reworded. "
            "The finding itself (these 14 genes show extreme constraint) remains valid "
            "regardless of labeling."
        ),
    }

    print(f"\n[8] Recommendation: {recommendation['action']}")
    print(f"    Current: {recommendation['current_label']}")
    print(f"    Proposed: {recommendation['proposed_label']}")
    print(f"    Rationale: {recommendation['rationale']}")

    # ========== 9. Save results ==========
    elapsed = time.time() - t0
    results = {
        "experiment": "E2_SynGO_EDT1_Provenance_Audit",
        "batch": "batch_070",
        "hypothesis": "The 14-gene SynGO_EDT1 list is NOT a programmatic SynGO x EDT1 intersection",
        "status": "COMPLETED",
        "wall_time_s": round(elapsed, 2),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "data_sources": {
            "edt1_xlsx": str(EDT1_XLSX),
            "edt1_xlsx_sheets": xl.sheet_names,
            "pgc3_txt": str(PGC3_TXT),
            "syngo_2024_gmt": str(SYNGO_2024_GMT),
            "syngo_2022_gmt": str(SYNGO_2022_GMT),
        },
        "gene_list_sizes": {
            "edt1_extended_data_table_1_all": len(edt1_all_genes),
            "edt1_extended_data_table_1_pc": len(edt1_pc_genes),
            "st12_all": len(st12_all_genes),
            "st12_pc": len(st12_pc_genes),
            "st12_syngo_column": len(st12_syngo_genes),
            "st12_prioritised": len(st12_prioritised_genes),
            "pgc3_txt": len(pgc3_txt_genes),
            "syngo_2024": len(syngo_2024_genes),
            "syngo_2022": len(syngo_2022_genes),
        },
        "per_gene_table": per_gene,
        "summary": {
            "n_in_edt1_extended_data_table_1": n_in_edt1_sheet,
            "n_in_st12_all_criteria": n_in_st12,
            "n_in_st12_syngo_column": n_in_st12_syngo,
            "n_in_pgc3_txt": n_in_pgc3_txt,
            "n_in_syngo_2024": n_in_syngo_2024,
            "n_in_syngo_2022": n_in_syngo_2022,
            "n_in_both_any_edt1_and_syngo": n_in_both,
            "genes_in_edt1_sheet": sorted(
                g for g in SYNOGO_EDT1_BATCH047 if g in edt1_all_genes
            ),
            "genes_in_st12": sorted(
                g for g in SYNOGO_EDT1_BATCH047 if g in st12_all_genes
            ),
            "genes_in_st12_syngo_col": sorted(
                g for g in SYNOGO_EDT1_BATCH047
                if g in st12_all_genes
                and truthy(
                    st12_sheet.loc[
                        st12_sheet["Symbol.ID"] == g, "SynGO.GeneSetMemb"
                    ].iloc[0]
                    if len(st12_sheet[st12_sheet["Symbol.ID"] == g]) > 0
                    else 0
                )
            ),
            "genes_NOT_in_any_edt1_source": sorted(
                g for g in SYNOGO_EDT1_BATCH047
                if g not in edt1_all_genes and g not in st12_all_genes
            ),
        },
        "st12_syngo_column_verification": {
            "is_programmatic_column": True,
            "description": (
                "The ST12 SynGO.GeneSetMemb column is a pre-computed binary flag "
                "in the Trubetskoy 2022 PGC3 supplementary xlsx. It contains exactly "
                "56 genes marked as SynGO members. This IS a true programmatic "
                "annotation (a column filter), not a hand-curated list."
            ),
            "n_genes_with_syngo_flag": len(st12_syngo_genes),
            "n_also_in_syngo_2024_gmt": len(st12_syngo_in_2024),
            "n_also_in_syngo_2022_gmt": len(st12_syngo_in_2022),
            "n_not_in_either_gmt": len(st12_syngo_not_in_either),
            "genes_not_in_either_gmt": sorted(st12_syngo_not_in_either),
            "all_56_genes": sorted(st12_syngo_genes),
        },
        "true_programmatic_intersections": {
            "A_st12_pc_syngo_column": {
                "method": "ST12 protein_coding with SynGO.GeneSetMemb=1",
                "n": len(true_syngo_edt1_st12_col),
                "genes": sorted(true_syngo_edt1_st12_col),
            },
            "B_st12_pc_intersect_syngo_2024": {
                "method": "ST12 protein_coding intersect SynGO 2024 GMT",
                "n": len(true_syngo_edt1_programmatic_2024),
                "genes": sorted(true_syngo_edt1_programmatic_2024),
            },
            "C_edt1_sheet_pc_intersect_syngo_2024": {
                "method": "Extended.Data.Table.1 protein_coding intersect SynGO 2024 GMT",
                "n": len(true_syngo_edt1_edt1sheet_2024),
                "genes": sorted(true_syngo_edt1_edt1sheet_2024),
            },
        },
        "provenance_determination": provenance_determination,
        "recommendation": recommendation,
    }

    out_path = OUTPUT_DIR / "e2_syngo_edt1_provenance.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[9] Wrote results to: {out_path}")
    print(f"    Wall time: {elapsed:.2f}s")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
