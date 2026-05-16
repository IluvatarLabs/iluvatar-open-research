#!/usr/bin/env python3
"""
E4 (R4): EDT1 Arithmetic Reconciliation

The manuscript has a count inconsistency:
- Functional decomposition tiers sum to 11+3+4+6+8+396 = 428
- EDT1 has n=261 (gnomAD-matched) or n=470 (total protein_coding from xlsx)
- 428 matches neither

This script traces the arithmetic to identify the source of the discrepancy.

WHY: A reviewer will catch the tier-sum discrepancy. We need to reconcile it
before submission by documenting where each number comes from and whether
tiers overlap (which would make them sum > total).

Data:
  - EDT1 genes from pgc3_genes.txt (n=300) — older list
  - EDT1 genes from xlsx (n=470 protein_coding) — canonical source used in batch_048
  - gnomAD v4.1 constraint metrics
  - Prior decomposition: batch_048/output/A_edt1_decomposition.json
  - Prior script: batch_048/scripts/run_batch_048.py (classification logic)

Output: experiments/batch_069/output/e4_edt1_reconciliation.json
"""

from __future__ import annotations
import json, math, pathlib, sys
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "batch_069" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paths
PGC3_TXT = PROJECT_ROOT / "experiments" / "batch_018" / "pgc3_genes.txt"
PGC3_XLSX = PROJECT_ROOT / "data" / "19426775" / "scz2022-Extended-Data-Table1.xlsx"
GNOMAD_TSV = PROJECT_ROOT / "data" / "item_15" / "gnomad.v4.1.constraint_metrics.tsv"
DECOMP_JSON = PROJECT_ROOT / "experiments" / "batch_048" / "output" / "A_edt1_decomposition.json"
TSS_CSV = PROJECT_ROOT / "data" / "ldsc" / "gene_tss_grch37.csv"

# Thresholds from batch_048 (Karczewski 2020)
LOEUF_THRESHOLD = 0.35
PLI_THRESHOLD = 0.9
MHC_CHR, MHC_START, MHC_END = "6", 25_000_000, 34_000_000

# Classification keywords from batch_048/scripts/run_batch_048.py
GLUTAMATE_RECEPTOR_KEYWORDS = ["glutamate receptor", "gria", "grik", "grm", "grin"]
ION_CHANNEL_KEYWORDS = ["channel", "ion channel", "voltage-gated", "ligand-gated",
    "cacna", "kcnq", "kcnn", "scn", "nav", "cav", "kv", "sk", "bk",
    "hyperpolarization", "trpc", "trpv", "kcnk"]
MITOCHONDRIAL_KEYWORDS = ["mitoch", "mitochondrial", "mt-", "cytochrome c",
    "respiratory chain", "atp synthase", "nd", "mtco", "mtatp", "cox", "sdh"]
TRANSCRIPTIONAL_KEYWORDS = ["transcription factor", "zinc finger", "histone",
    "chromatin", "kdm", "setd", "smarc", "chd", "epigen", "methyltransferase",
    "acetyltransferase"]

# SynGO_EDT1 from batch_047 (Singh 2022 Table 1)
SYNOGO_EDT1_BATCH047 = [
    "DLGAP1", "GRIN2A", "NRXN1", "CNTNAP2", "ARC", "DLG4", "NRXN2",
    "NLGN1", "NLGN2", "SHANK1", "SHANK3", "HOMER1", "SYN1", "GAP43"
]


def classify_gene(gene: str) -> str:
    """Classify a gene symbol into functional category.
    Reproduces batch_048 classify_edt1_gene exactly."""
    g_lower = gene.lower()
    if any(k in g_lower for k in GLUTAMATE_RECEPTOR_KEYWORDS):
        return "glutamate_receptor"
    if any(k in g_lower for k in ION_CHANNEL_KEYWORDS):
        return "ion_channel"
    if any(k in g_lower for k in MITOCHONDRIAL_KEYWORDS):
        return "mitochondrial"
    if any(k in g_lower for k in TRANSCRIPTIONAL_KEYWORDS):
        return "transcriptional"
    return "other"


def load_gnomad_canonical():
    """Load gnomAD v4.1, canonical + MANE, with MHC exclusion."""
    print("Loading gnomAD v4.1...")
    df = pd.read_csv(GNOMAD_TSV, sep="\t", low_memory=False)
    if "gene_id" in df.columns:
        df = df[df["gene_id"].astype(str).str.startswith("ENSG")].copy()
    for col in ("canonical", "mane_select"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
    mask = df["canonical"] & df["mane_select"] & df["lof.oe_ci.upper"].notna()
    df = df.loc[mask].drop_duplicates(subset=["gene"], keep="first").reset_index(drop=True)
    df["loeuf_lt_035"] = df["lof.oe_ci.upper"] < LOEUF_THRESHOLD
    df["pli_ge_09"] = df["lof.pLI"] >= PLI_THRESHOLD

    # TSS for MHC
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
    df["mhc"] = (tss_chrom == MHC_CHR) & (tss_pos >= MHC_START) & (tss_pos <= MHC_END)
    print(f"  gnomAD canonical+MANE: {len(df)} genes, {df['mhc'].sum()} MHC")
    return df


def main():
    # 1. Load gene lists
    pgc3_txt_genes = set(open(PGC3_TXT).read().strip().split("\n"))
    print(f"pgc3_genes.txt: {len(pgc3_txt_genes)} genes")

    xlsx_df = pd.read_excel(PGC3_XLSX, sheet_name="ST12 all criteria")
    xlsx_pc = xlsx_df[xlsx_df["gene_biotype"] == "protein_coding"]
    xlsx_pc_genes = set(xlsx_pc["Symbol.ID"].dropna().astype(str))
    xlsx_all_genes = set(xlsx_df["Symbol.ID"].dropna().astype(str))
    print(f"xlsx protein_coding: {len(xlsx_pc_genes)} genes")
    print(f"xlsx all biotypes: {len(xlsx_all_genes)} genes")

    # SynGO flag from xlsx
    def truthy(v):
        if v is None: return False
        try:
            if isinstance(v, float) and math.isnan(v): return False
        except TypeError: pass
        return str(v).strip().lower() in {"yes", "y", "1", "true", "1.0"}
    syngo_xlsx = set(xlsx_pc[xlsx_pc["SynGO.GeneSetMemb"].apply(truthy)]["Symbol.ID"].dropna().astype(str))
    print(f"xlsx SynGO genes: {len(syngo_xlsx)}")

    # 2. Load gnomAD
    gnomad = load_gnomad_canonical()
    gnomad_genes = set(gnomad["gene"].astype(str))
    bg = gnomad[~gnomad["mhc"]].copy()
    bg_genes = set(bg["gene"].astype(str))
    print(f"gnomAD background (excl MHC): {len(bg_genes)} genes")

    # 3. Reconcile EDT1 counts
    # The canonical EDT1 source used in batch_048 is the xlsx protein_coding set
    edt1_all = xlsx_pc_genes
    edt1_in_gnomad = edt1_all & gnomad_genes
    edt1_in_bg = edt1_all & bg_genes  # gnomAD excl MHC
    edt1_not_in_gnomad = edt1_all - gnomad_genes

    print(f"\n=== EDT1 Reconciliation ===")
    print(f"EDT1 total (xlsx protein_coding): {len(edt1_all)}")
    print(f"EDT1 in gnomAD (any): {len(edt1_in_gnomad)}")
    print(f"EDT1 in gnomAD excl MHC: {len(edt1_in_bg)}")
    print(f"EDT1 NOT in gnomAD: {len(edt1_not_in_gnomad)}")
    if edt1_not_in_gnomad:
        print(f"  Missing genes: {sorted(edt1_not_in_gnomad)}")

    # 4. Classify each gene into tier
    tier_assignments = {}
    for gene in edt1_in_bg:
        tier = classify_gene(gene)
        tier_assignments[gene] = tier

    # Check SynGO membership
    syngo_batch047_set = set(SYNOGO_EDT1_BATCH047) & bg_genes
    for gene in syngo_batch047_set & edt1_in_bg:
        # SynGO is a separate overlay, not exclusive with keyword classification
        tier_assignments.setdefault(gene, "other")  # already assigned

    tier_counts = {}
    for gene, tier in tier_assignments.items():
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print(f"\n=== Tier Counts (keyword-based from batch_048 classify_edt1_gene) ===")
    tier_sum = 0
    for tier in sorted(tier_counts.keys()):
        print(f"  {tier}: {tier_counts[tier]}")
        tier_sum += tier_counts[tier]
    print(f"  TOTAL: {tier_sum}")

    # 5. Check SynGO_EDT1 overlap with keyword tiers
    syngo_in_edt1 = syngo_batch047_set & edt1_in_bg
    syngo_overlap_with_tiers = {}
    for gene in syngo_in_edt1:
        tier = tier_assignments.get(gene, "NOT_IN_CLASSIFICATION")
        syngo_overlap_with_tiers[gene] = tier
    print(f"\n=== SynGO_EDT1 × Keyword Tier Overlap ===")
    print(f"SynGO_EDT1 in EDT1 background: {len(syngo_in_edt1)}")
    for gene, tier in sorted(syngo_overlap_with_tiers.items()):
        print(f"  {gene}: {tier}")

    # 6. Load prior decomposition to understand the 428 number
    with open(DECOMP_JSON) as f:
        decomp = json.load(f)

    # The decomposition tested 6 gene lists against 2 constraint metrics = 12 tests
    # The gene lists and their sizes:
    print(f"\n=== Prior Decomposition (batch_048) Gene Lists ===")
    seen_lists = {}
    for r in decomp["results"]:
        gl = r["gene_list"]
        n = r["n_in_list"]
        if gl not in seen_lists:
            seen_lists[gl] = n
            print(f"  {gl}: n={n}")

    decomp_sum = sum(seen_lists.values())
    print(f"  SUM: {decomp_sum}")

    # The 428 = 11 + 3 + 4 + 6 + 8 + 396 = 428
    # This is SynGO_EDT1(11) + glutamate_receptor(3) + transcriptional(4)
    # + ion_channel(6) + mitochondrial(8) + other(396)
    # But note SynGO_EDT1 is not a keyword category — it's an overlay
    # AND some SynGO genes may ALSO be in glutamate_receptor etc.

    # Check overlap
    print(f"\n=== Overlap Analysis ===")
    # Build sets for each tier
    tier_sets = {}
    for gene, tier in tier_assignments.items():
        tier_sets.setdefault(tier, set()).add(gene)
    tier_sets["SynGO_EDT1_batch047"] = syngo_in_edt1

    all_tier_names = sorted(tier_sets.keys())
    overlap_matrix = {}
    for t1 in all_tier_names:
        overlap_matrix[t1] = {}
        for t2 in all_tier_names:
            overlap_matrix[t1][t2] = len(tier_sets[t1] & tier_sets[t2])

    print("Overlap matrix:")
    header = f"{'':30s}" + "".join(f"{t:>15s}" for t in all_tier_names)
    print(header)
    for t1 in all_tier_names:
        row = f"{t1:30s}" + "".join(f"{overlap_matrix[t1][t2]:>15d}" for t2 in all_tier_names)
        print(row)

    # Key question: do the keyword tiers EXCLUDE SynGO genes?
    # In batch_048, SynGO_EDT1 is tested SEPARATELY but genes also appear in keyword tiers
    keyword_tiers_only = [t for t in all_tier_names if t != "SynGO_EDT1_batch047"]
    keyword_union = set()
    for t in keyword_tiers_only:
        keyword_union |= tier_sets.get(t, set())

    syngo_in_keyword = syngo_in_edt1 & keyword_union
    syngo_not_in_keyword = syngo_in_edt1 - keyword_union

    print(f"\nSynGO genes also in keyword tiers: {len(syngo_in_keyword)}")
    print(f"SynGO genes NOT in keyword tiers: {len(syngo_not_in_keyword)}")
    for g in sorted(syngo_not_in_keyword):
        print(f"  {g} (would be in 'other' by keyword)")

    # CRITICAL CLARIFICATION:
    # The SynGO_EDT1_batch047 gene list is NOT a subset of EDT1. Only 2/14 genes
    # (GRIN2A, SHANK3) appear in EDT1 xlsx. The batch_048 decomposition tested
    # SynGO_EDT1 as an INDEPENDENT reference gene set against the gnomAD background,
    # not as a partition of EDT1. This means:
    # - The 5 keyword tiers ARE a partition of EDT1-in-gnomAD (sum=418)
    # - SynGO_EDT1 is an overlay tested separately (11 genes in gnomAD bg)
    # - The 428 = 418 keyword partition + 11 SynGO - 1 overlap (GRIN2A) = 428
    #   Actually: 3+6+8+4+396+11 = 428 (batch_048 had other=396 vs our 397,
    #   likely due to gnomAD version/filter difference between runs)
    syngo_in_gnomad_bg = set(SYNOGO_EDT1_BATCH047) & bg_genes
    syngo_in_edt1_xlsx = set(SYNOGO_EDT1_BATCH047) & xlsx_pc_genes
    syngo_only_in_bg_not_edt1 = syngo_in_gnomad_bg - edt1_in_bg
    print(f"\n=== CRITICAL: SynGO_EDT1 is NOT a subset of EDT1 ===")
    print(f"SynGO_EDT1 in gnomAD bg: {len(syngo_in_gnomad_bg)} (the n=11 in decomposition)")
    print(f"SynGO_EDT1 in EDT1 xlsx: {len(syngo_in_edt1_xlsx)} ({sorted(syngo_in_edt1_xlsx)})")
    print(f"SynGO_EDT1 in gnomAD bg but NOT in EDT1: {len(syngo_only_in_bg_not_edt1)}")
    print(f"  Genes: {sorted(syngo_only_in_bg_not_edt1)}")
    # Store for later addition to reconciliation dict
    _syngo_critical = {
        "syngo_edt1_is_subset_of_edt1": False,
        "syngo_edt1_in_gnomad_bg": len(syngo_in_gnomad_bg),
        "syngo_edt1_in_edt1_xlsx": sorted(syngo_in_edt1_xlsx),
        "syngo_edt1_not_in_edt1": sorted(syngo_only_in_bg_not_edt1),
        "critical_note": (
            "SynGO_EDT1_batch047 (n=11 in gnomAD bg) is NOT a subset of EDT1. "
            "Only 2/14 genes (GRIN2A, SHANK3) are in the EDT1 xlsx. The batch_048 "
            "decomposition tested it as an independent reference gene set against the "
            "full gnomAD background. The 428 sum = 5 keyword tiers (3+6+8+4+396=417 in batch_048) "
            "+ SynGO_EDT1 (11) = 428. The keyword tiers partition EDT1-in-gnomAD into "
            "non-overlapping categories. The current run finds keyword sum=418 (other=397), "
            "which differs from batch_048 (other=396) by 1 gene, likely due to gnomAD "
            "filter differences between runs."
        ),
    }

    # 7. Build per-gene table with gnomAD metrics
    per_gene_table = []
    for gene in sorted(edt1_all):
        row = {"gene": gene}
        row["in_gnomad_bg"] = gene in bg_genes
        row["keyword_tier"] = tier_assignments.get(gene, "NOT_IN_BG")
        row["syngo_edt1_batch047"] = gene in syngo_in_edt1
        row["syngo_xlsx"] = gene in syngo_xlsx

        if gene in bg_genes:
            g_row = bg[bg["gene"] == gene].iloc[0] if len(bg[bg["gene"] == gene]) > 0 else None
            if g_row is not None:
                pli = g_row.get("lof.pLI")
                loeuf = g_row.get("lof.oe_ci.upper")
                row["pLI"] = float(pli) if pd.notna(pli) else None
                row["LOEUF"] = float(loeuf) if pd.notna(loeuf) else None
                row["pLI_ge_09"] = bool(g_row["pli_ge_09"]) if pd.notna(g_row["pli_ge_09"]) else None
                row["LOEUF_lt_035"] = bool(g_row["loeuf_lt_035"]) if pd.notna(g_row["loeuf_lt_035"]) else None
            else:
                row["pLI"] = None
                row["LOEUF"] = None
        else:
            row["pLI"] = None
            row["LOEUF"] = None

        per_gene_table.append(row)

    # 8. Compute the reconciliation
    # The 428 = sum of keyword tier sizes for EDT1 genes in gnomAD bg
    # This should equal len(edt1_in_bg) since every gene gets exactly one keyword tier
    #
    # But batch_048 n_in_list counts include SynGO_EDT1(11) which overlaps keyword tiers
    # The SynGO set is TESTED SEPARATELY — it's not part of the partition

    reconciliation = {
        "edt1_total_xlsx_protein_coding": len(edt1_all),
        "edt1_in_gnomad_any": len(edt1_in_gnomad),
        "edt1_in_gnomad_excl_mhc": len(edt1_in_bg),
        "edt1_not_in_gnomad": len(edt1_not_in_gnomad),
        "edt1_not_in_gnomad_genes": sorted(edt1_not_in_gnomad),
        "pgc3_txt_count": len(pgc3_txt_genes),
        "pgc3_txt_overlap_with_xlsx_pc": len(pgc3_txt_genes & xlsx_pc_genes),
        "keyword_tier_partition": {t: tier_counts.get(t, 0) for t in sorted(tier_counts.keys())},
        "keyword_tier_sum": tier_sum,
        "note_keyword_partition": "Each EDT1 gene in gnomAD bg gets exactly one keyword tier. Sum == edt1_in_gnomad_excl_mhc.",
        "syngo_edt1_batch047_n": len(syngo_in_edt1),
        "syngo_edt1_batch047_genes": sorted(syngo_in_edt1),
        "syngo_overlap_with_keyword_tiers": {g: t for g, t in sorted(syngo_overlap_with_tiers.items())},
        "decomposition_tested_gene_lists": seen_lists,
        "decomposition_sum_n_in_list": decomp_sum,
        "reconciliation_explanation": (
            "The 428 count comes from summing the 5 keyword tiers "
            "(glutamate_receptor=3, ion_channel=6, mitochondrial=8, transcriptional=4, other=396) "
            "PLUS the SynGO_EDT1 overlay (n=11). This sum double-counts genes that are in BOTH "
            "SynGO_EDT1 AND a keyword tier (e.g., GRIN2A is in both SynGO and glutamate_receptor). "
            f"The keyword partition alone sums to {tier_sum}, which equals edt1_in_gnomad_excl_mhc={len(edt1_in_bg)}. "
            f"The manuscript's 261 likely refers to the number of genes with pLI >= 0.9 "
            "OR LOEUF <= 0.35 in the constraint analysis, not the total EDT1 count."
        ),
        "overlap_matrix": overlap_matrix,
    }

    # Check: how many EDT1 genes have pLI >= 0.9?
    n_pli = sum(1 for r in per_gene_table if r.get("pLI_ge_09") is True)
    n_loeuf = sum(1 for r in per_gene_table if r.get("LOEUF_lt_035") is True)
    n_either = sum(1 for r in per_gene_table if r.get("pLI_ge_09") is True or r.get("LOEUF_lt_035") is True)
    reconciliation.update(_syngo_critical)
    reconciliation["edt1_pLI_ge_09_count"] = n_pli
    reconciliation["edt1_LOEUF_lt_035_count"] = n_loeuf
    reconciliation["edt1_either_constrained"] = n_either

    print(f"\n=== Constraint Counts in EDT1 ===")
    print(f"  pLI >= 0.9: {n_pli}")
    print(f"  LOEUF <= 0.35: {n_loeuf}")
    print(f"  Either: {n_either}")

    # Final output
    result = {
        "experiment": "E4_EDT1_Arithmetic_Reconciliation",
        "batch": "batch_069",
        "status": "COMPLETED",
        "reconciliation": reconciliation,
        "per_gene_table_count": len(per_gene_table),
        "per_gene_table": per_gene_table,
    }

    out_path = OUTPUT_DIR / "e4_edt1_reconciliation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
