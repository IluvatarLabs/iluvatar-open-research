#!/usr/bin/env python3
"""
04_axis_c_functional_tier.py — Axis C 6-tier functional decomposition.

Tiers (exclusive priority, highest wins):
  1. glutamate_receptor
  2. ion_channel
  3. synaptic (residual, SynGO minus 1-2)
  4. chromatin_TF
  5. kinase
  6. mitochondrial

Per (disorder × tier): Fisher pLI>=0.9 OR vs background (gene in gnomAD
canonical+MANE tier, not in list). Produce both exclusive and inclusive
(multi-membership) versions.

Off-diagonal variance reported as max_OR / min_OR across 24 cells (descriptive
cells with n<5 flagged).
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH = ROOT / "experiments" / "batch_052_A"
INPUT = BATCH / "input"
OUTPUT = BATCH / "output"
LOGS = BATCH / "logs"

LOG = LOGS / "run.log"
SEED = 20260424


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fisher_ha(a, b, c, d):
    a2, b2, c2, d2 = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ha = (a2 * d2) / (b2 * c2)
    try:
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    except ValueError:
        p = np.nan
    return or_ha, float(p)


def parse_syngo_gmt(p):
    genes = set()
    with open(p) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            # Skip header-like lines; SynGO fields: term, desc, gene1, gene2...
            gs = [g.strip() for g in parts[2:] if g.strip()]
            genes.update(gs)
    return genes


def main():
    log("=" * 70)
    log("batch_052_A/04_axis_c_functional_tier.py starting")

    with open(INPUT / "assembled_inputs.json") as f:
        assembled = json.load(f)
    lists = {k: set(v["genes"]) for k, v in assembled["lists"].items()}

    # --- gnomAD canonical+MANE with pLI ---
    gc = pd.read_csv(ROOT / "data/item_15/gnomad.v4.1.constraint_metrics.tsv",
                     sep="\t", low_memory=False)
    mask = (gc["canonical"].astype(str).str.lower() == "true") & (
        gc["mane_select"].astype(str).str.lower() == "true")
    gc_mm = gc[mask].drop_duplicates(subset="gene").dropna(subset=["gene", "lof.pLI"])
    pli_map = dict(zip(gc_mm["gene"].astype(str), gc_mm["lof.pLI"].astype(float)))
    universe = set(gc_mm["gene"].astype(str))
    log(f"gnomAD canonical+MANE universe: {len(universe):,}")

    # --- SynGO (synaptic) ---
    syngo_genes = parse_syngo_gmt(INPUT / "syngo_2024.gmt")
    log(f"SynGO universe: {len(syngo_genes):,}")

    # --- HGNC gene groups ---
    hf = pd.read_csv(ROOT / "experiments/batch_052_B/input/hgnc_family.tsv", sep="\t")
    hf = hf.rename(columns={"Approved symbol": "symbol", "Gene group name": "group"})
    hf["group"] = hf["group"].fillna("")

    def genes_where(pattern):
        mask = hf["group"].str.contains(pattern, case=False, regex=True, na=False)
        return set(hf.loc[mask, "symbol"].dropna().astype(str))

    glu_hgnc = genes_where(r"Glutamate (ionotropic|metabotropic)")
    ion_hgnc = genes_where(r"channel")  # all channels
    chrom_hgnc = genes_where(r"Zinc finger|Homeobox|HMG box|Bromodomain|Chromatin|Helix-loop-helix|transcription factor|BTB/POZ|SWI/SNF|basic helix")
    kinase_hgnc = genes_where(r"[Kk]inase")
    mito_hgnc = genes_where(r"Mitochondrial")

    # Augment with prefix regex for ion channels and GluR
    glu_prefix_re = re.compile(r"^(GRIA|GRIK|GRIN|GRM|DLGAP|)\d+$")
    def matches_prefix(regex, symbol):
        return regex.match(str(symbol)) is not None

    ion_prefix_re = re.compile(
        r"^(SCN|KCN|CACN|HCN|CLCN|CNG|TRPM|TRPA|TRPC|TRPV|TRPP|ASIC|KCNMA|KCNN|KCNQ|KCNH|KCNK|KCNJ|KCNC|KCND|KCNB|KCNA|KCNG|KCNV|KCNS|CATSPER|NALCN|CLIC|BEST|ANO|SLC26A)\d*[A-Z]*\d*$"
    )

    # Build GluR tier
    glu = set()
    for g in universe:
        if g in glu_hgnc or glu_prefix_re.match(g):
            glu.add(g)
    # Add explicit GRIA/K/N/M prefixes
    for g in universe:
        if re.match(r"^(GRIA|GRIK|GRIN|GRM)\d+[A-Z]*$", g):
            glu.add(g)

    ion = set()
    for g in universe:
        if g in ion_hgnc or ion_prefix_re.match(g):
            ion.add(g)
    ion -= glu  # remove overlap

    syn = (syngo_genes & universe) - glu - ion

    chrom = (chrom_hgnc & universe) - glu - ion - syn
    kin = (kinase_hgnc & universe) - glu - ion - syn - chrom
    mito = (mito_hgnc & universe) - glu - ion - syn - chrom - kin

    tiers_exclusive = {
        "glutamate_receptor": glu,
        "ion_channel": ion,
        "synaptic_residual": syn,
        "chromatin_TF": chrom,
        "kinase": kin,
        "mitochondrial": mito,
    }
    tiers_inclusive = {
        "glutamate_receptor": (glu_hgnc & universe) | {g for g in universe if re.match(r"^(GRIA|GRIK|GRIN|GRM)\d+[A-Z]*$", g)},
        "ion_channel": (ion_hgnc & universe) | {g for g in universe if ion_prefix_re.match(g)},
        "synaptic": syngo_genes & universe,
        "chromatin_TF": chrom_hgnc & universe,
        "kinase": kinase_hgnc & universe,
        "mitochondrial": mito_hgnc & universe,
    }

    log("Exclusive tier sizes (within gnomAD canonical+MANE):")
    for t, s in tiers_exclusive.items():
        log(f"  {t}: {len(s):,}")
    log("Inclusive tier sizes (within gnomAD canonical+MANE):")
    for t, s in tiers_inclusive.items():
        log(f"  {t}: {len(s):,}")

    # --- Build per-tier × per-list Fisher OR matrix ---
    def build_matrix(tiers_dict, label):
        rows = []
        for list_name, glist in lists.items():
            list_in_u = glist & universe
            for tier, tier_genes in tiers_dict.items():
                tier_in_u = tier_genes & universe
                list_in_tier = list_in_u & tier_in_u
                bg_in_tier = tier_in_u - list_in_u
                # pLI >= 0.9
                list_high = sum(1 for g in list_in_tier if pli_map.get(g, np.nan) >= 0.9)
                bg_high = sum(1 for g in bg_in_tier if pli_map.get(g, np.nan) >= 0.9)
                a = list_high
                b = len(list_in_tier) - list_high
                c = bg_high
                d = len(bg_in_tier) - bg_high
                or_ha, p_f = fisher_ha(a, b, c, d)
                rows.append({
                    "assignment": label,
                    "list": list_name,
                    "tier": tier,
                    "tier_size_in_universe": len(tier_in_u),
                    "list_in_tier_n": len(list_in_tier),
                    "list_in_tier_pli_high": list_high,
                    "bg_in_tier_n": len(bg_in_tier),
                    "bg_in_tier_pli_high": bg_high,
                    "OR_HA": or_ha,
                    "fisher_p_greater": p_f,
                    "descriptive_only_n_lt_5": len(list_in_tier) < 5,
                })
        return rows

    excl_rows = build_matrix(tiers_exclusive, "exclusive")
    incl_rows = build_matrix(tiers_inclusive, "inclusive")

    # --- Off-diagonal variance (max/min OR across 24 cells), n>=5 only ---
    def spread_metrics(rows, n_min=5):
        ors = [r["OR_HA"] for r in rows if r["list_in_tier_n"] >= n_min and np.isfinite(r["OR_HA"]) and r["OR_HA"] > 0]
        if len(ors) < 2:
            return {"n_cells_valid": len(ors), "max_OR": None, "min_OR": None, "ratio_max_min": None}
        return {"n_cells_valid": len(ors),
                "max_OR": float(max(ors)), "min_OR": float(min(ors)),
                "ratio_max_min": float(max(ors) / min(ors))}

    excl_spread = spread_metrics(excl_rows)
    incl_spread = spread_metrics(incl_rows)
    log(f"Exclusive matrix spread (n>=5 cells only): {excl_spread}")
    log(f"Inclusive matrix spread (n>=5 cells only): {incl_spread}")

    # --- Save TSVs ---
    df_excl = pd.DataFrame(excl_rows)
    df_incl = pd.DataFrame(incl_rows)
    df_excl.to_csv(OUTPUT / "axis_c_tier_matrix_exclusive.tsv", sep="\t", index=False)
    df_incl.to_csv(OUTPUT / "axis_c_tier_matrix_inclusive.tsv", sep="\t", index=False)
    # Also a combined 'axis_c_tier_matrix.tsv' (primary = exclusive per brief)
    df_excl.to_csv(OUTPUT / "axis_c_tier_matrix.tsv", sep="\t", index=False)
    log(f"Wrote {OUTPUT / 'axis_c_tier_matrix_exclusive.tsv'}")
    log(f"Wrote {OUTPUT / 'axis_c_tier_matrix_inclusive.tsv'}")

    # Pivot to matrix for quick inspection in log
    excl_pivot_or = df_excl.pivot(index="list", columns="tier", values="OR_HA")
    excl_pivot_n = df_excl.pivot(index="list", columns="tier", values="list_in_tier_n")
    log("Exclusive OR_HA matrix:\n" + str(excl_pivot_or))
    log("Exclusive list-in-tier n matrix:\n" + str(excl_pivot_n))

    results = {
        "axis": "C_functional_tier",
        "tier_sizes_exclusive": {t: len(s) for t, s in tiers_exclusive.items()},
        "tier_sizes_inclusive": {t: len(s) for t, s in tiers_inclusive.items()},
        "gnomad_universe_n": len(universe),
        "syngo_genes_total": len(syngo_genes),
        "per_cell_exclusive": excl_rows,
        "per_cell_inclusive": incl_rows,
        "exclusive_spread": excl_spread,
        "inclusive_spread": incl_spread,
        "seed": SEED,
    }
    with open(OUTPUT / "axis_c_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Wrote {OUTPUT / 'axis_c_results.json'}")
    log("04_axis_c_functional_tier.py DONE")


if __name__ == "__main__":
    main()
