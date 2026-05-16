"""06_f124_tier_definitions.py — F124 4-tier decomposition across 4 tier-definition sources.

Headline (batch_048): synaptic 20.9 / channel 0.93 / tf 13.9 / other 2.46.

Tier definition sources (preflight):
  F124_curated:   batch_048 keyword classifier (SynGO-like keywords + ion_channel/TF/other).
  HGNC_family:    partition by HGNC "Gene group name" substring match (synaptic/channel/TF/other).
  KEGG_path:      top KEGG pathway hits (Synapse/Ion channel related).
  Reactome_path:  Neuronal System / Ion channel transport hierarchy.

For each source: re-run 4-tier decomposition on EDT1_all_pc (ST12 pc, n=470);
compute pLI≥0.9 OR per tier vs gnomAD bg. Record tier-sharing Jaccard across sources.
Seed 20260424.
"""
from __future__ import annotations
import json, sys, time, re
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import (OUTPUT_DIR, LOGS_DIR, RNG_SEED, N_PERM,
                    load_pgc3, load_gnomad, load_hgnc_family, fisher_or_ci, jaccard, log_event)

# --- F124-curated keyword lists (batch_048 mirror) ---
GLUTAMATE_RECEPTOR_KEYWORDS = ["grin", "gria", "grik", "grm", "grip"]
ION_CHANNEL_KEYWORDS = ["cacn", "kcn", "scn", "hcn", "catsper", "ryr", "itpr", "trp"]
MITOCHONDRIAL_KEYWORDS = ["nduf", "sdha", "sdhb", "uqcr", "cox", "atp5", "mt-"]
TRANSCRIPTIONAL_KEYWORDS = ["foxp", "tbr", "satb", "pou", "sox", "zic", "bcl", "zfp", "ezh", "setd", "kdm", "mll", "ctcf", "egr", "klf"]
NEURODEV_KEYWORDS = ["dcx", "reln", "lis1", "pafah1b1", "sema", "plxn", "ntn", "slit", "robo"]

def classify_curated(gene: str) -> str:
    g = gene.lower()
    if any(k in g for k in GLUTAMATE_RECEPTOR_KEYWORDS): return "synaptic"
    if any(k in g for k in ION_CHANNEL_KEYWORDS): return "ion_channel"
    if any(k in g for k in MITOCHONDRIAL_KEYWORDS): return "other"  # batch_048 doesn't treat mito as a tier
    if any(k in g for k in TRANSCRIPTIONAL_KEYWORDS): return "transcriptional"
    if any(k in g for k in NEURODEV_KEYWORDS): return "other"
    return "other"

def classify_hgnc(hgnc_df: pd.DataFrame):
    """Return a dict gene -> tier using HGNC 'Gene group name' pattern match."""
    # Build gene -> family set
    fam_map = hgnc_df.groupby("symbol")["family"].apply(lambda s: " | ".join(str(x) for x in s if isinstance(x, str)))
    def classify(gene):
        fam = fam_map.get(gene, "")
        fl = fam.lower()
        if not fl:
            return "other"
        if any(k in fl for k in ["synap", "glutamate recept", "gabaa receptor", "postsynaptic", "presynaptic", "neurotransmitter receptor"]):
            return "synaptic"
        if any(k in fl for k in ["ion channel", "potassium channel", "sodium channel", "calcium channel", "chloride channel", "voltage-gated"]):
            return "ion_channel"
        if any(k in fl for k in ["transcription factor", "zinc finger", "homeobox", "bhlh", "forkhead", "chromatin", "histone"]):
            return "transcriptional"
        return "other"
    return classify

def classify_kegg():
    """Use gseapy to pull KEGG_2021_Human library (pathway -> gene list)."""
    import gseapy
    lib = gseapy.get_library("KEGG_2021_Human", "Human")
    # Build gene -> set of pathway names
    g2p = {}
    for path, genes in lib.items():
        for gene in genes:
            g2p.setdefault(gene, set()).add(path)
    # Synaptic KEGG pathways
    SYN = {"Glutamatergic synapse", "GABAergic synapse", "Cholinergic synapse", "Dopaminergic synapse",
           "Serotonergic synapse", "Synaptic vesicle cycle", "Neuroactive ligand-receptor interaction"}
    ION = {"Calcium signaling pathway", "cAMP signaling pathway", "cGMP-PKG signaling pathway"}
    TF = {"Basal transcription factors", "Transcriptional misregulation in cancer"}
    def classify(gene):
        paths = g2p.get(gene, set())
        if paths & SYN: return "synaptic"
        if paths & ION: return "ion_channel"
        if paths & TF: return "transcriptional"
        return "other"
    return classify

def classify_reactome():
    import gseapy
    lib = gseapy.get_library("Reactome_2022", "Human")
    g2p = {}
    for path, genes in lib.items():
        for gene in genes:
            g2p.setdefault(gene, set()).add(path)
    # Reactome hierarchy: "Neuronal System" and sub-paths; "Ion channel transport"; "Generic Transcription Pathway"
    def classify(gene):
        paths = g2p.get(gene, set())
        has_syn = any("neuronal system" in p.lower() or "neurotransmitter" in p.lower()
                      or "synapse" in p.lower() for p in paths)
        has_ion = any("ion channel" in p.lower() for p in paths)
        has_tf = any("transcription" in p.lower() and "regulation" in p.lower() for p in paths) or \
                 any("rna polymerase ii transcription" in p.lower() for p in paths) or \
                 any("chromatin" in p.lower() for p in paths)
        if has_syn: return "synaptic"
        if has_ion: return "ion_channel"
        if has_tf: return "transcriptional"
        return "other"
    return classify

def run_tier_or(tier_name: str, tier_genes: set[str], gnomad, rng) -> dict:
    bg_genes = set(gnomad["gene"].astype(str))
    pli_hi = set(gnomad.loc[gnomad["pLI"] >= 0.9, "gene"].astype(str))
    in_bg = tier_genes & bg_genes
    a = len(in_bg & pli_hi)
    b = len(in_bg - pli_hi)
    control = bg_genes - in_bg
    c = len(control & pli_hi)
    d = len(control - pli_hi)
    fish = fisher_or_ci(a, b, c, d)
    descriptive = (len(in_bg) < 5)
    bg_arr = np.array(sorted(bg_genes))
    hi_mask = np.isin(bg_arr, list(pli_hi))
    n_tier = len(in_bg); k_obs = a
    if n_tier == 0 or not hi_mask.any():
        emp_p = 1.0
    else:
        hits = np.empty(N_PERM, dtype=np.int32)
        for i in range(N_PERM):
            idx = rng.choice(len(bg_arr), size=n_tier, replace=False)
            hits[i] = hi_mask[idx].sum()
        emp_p = float((hits >= k_obs).sum() + 1) / (N_PERM + 1)
    return {"tier": tier_name, "n_in_bg": n_tier, "descriptive_only": descriptive,
            **fish, "emp_p": emp_p}

def main():
    log = LOGS_DIR / "run.log"; t0 = time.time()
    pgc3 = load_pgc3()
    st12 = pgc3["st12"]
    edt1 = set(st12.loc[st12["gene_biotype"] == "protein_coding", "Symbol.ID"].astype(str))
    print(f"F124 EDT1_all_pc n={len(edt1)}")
    gnomad = load_gnomad()
    hgnc = load_hgnc_family()
    cls_curated = classify_curated
    cls_hgnc = classify_hgnc(hgnc)
    try:
        cls_kegg = classify_kegg()
        kegg_ok = True
    except Exception as e:
        print(f"KEGG load failed: {e}"); cls_kegg = None; kegg_ok = False
    try:
        cls_reactome = classify_reactome()
        reac_ok = True
    except Exception as e:
        print(f"Reactome load failed: {e}"); cls_reactome = None; reac_ok = False

    rng = np.random.default_rng(RNG_SEED)
    sources_cells: dict = {}
    source_funcs = [("F124_curated", cls_curated)]
    source_funcs.append(("HGNC_family", cls_hgnc))
    if kegg_ok: source_funcs.append(("KEGG_path", cls_kegg))
    if reac_ok: source_funcs.append(("Reactome_path", cls_reactome))

    per_source_tier_sets: dict[str, dict[str, set]] = {}
    for sname, fn in source_funcs:
        tiers = {"synaptic": set(), "ion_channel": set(), "transcriptional": set(), "other": set()}
        for g in edt1:
            tiers[fn(g)].add(g)
        per_source_tier_sets[sname] = tiers
        cells = []
        for tier, tg in tiers.items():
            r = run_tier_or(tier, tg, gnomad, rng)
            r["spec"] = f"{sname}__{tier}"
            cells.append(r)
            flag = "DESC" if r["descriptive_only"] else "OK  "
            print(f"F124 | {flag} {r['spec']:>35} n={r['n_in_bg']:4d} OR={r['OR']:.3g} raw_p={r['raw_p']:.3g} emp_p={r['emp_p']:.4f}")
        sources_cells[sname] = cells

    # Pairwise tier-sharing Jaccard between sources (per tier)
    tier_jacc = {}
    source_names = list(per_source_tier_sets)
    for tier in ("synaptic", "ion_channel", "transcriptional", "other"):
        for i, a in enumerate(source_names):
            for b in source_names[i + 1:]:
                tier_jacc[f"{tier}|{a}|{b}"] = jaccard(per_source_tier_sets[a][tier], per_source_tier_sets[b][tier])

    elapsed = time.time() - t0
    out = {"finding": "F124", "headline_OR_by_tier": {"synaptic": 20.9, "ion_channel": 0.93, "transcriptional": 13.9, "other": 2.46},
           "headline_source": "batch_048",
           "sources": sources_cells, "tier_jaccard": tier_jacc, "elapsed_s": elapsed,
           "source_tier_counts": {s: {t: len(v) for t, v in tiers.items()} for s, tiers in per_source_tier_sets.items()}}
    (OUTPUT_DIR / "f124_tier_definitions.json").write_text(json.dumps(out, indent=2, default=str))
    log_event(log, f"[06_f124] elapsed={elapsed:.2f}s sources={list(sources_cells)}")

if __name__ == "__main__":
    main()
