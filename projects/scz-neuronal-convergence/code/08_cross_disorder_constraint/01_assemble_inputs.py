#!/usr/bin/env python3
"""
01_assemble_inputs.py — batch_052_A

Loads the 4 cross-disorder gene lists, all data resources, computes Jaccard
overlaps (for brief H3 orthogonality check), and writes
`input/assembled_inputs.json` with sha256 fingerprints.

Why this step first: the brief mandates Jaccard(Heyne,DDD)/(Heyne,SCHEMA)/
(Heyne,ASD) are computed BEFORE any axis is tested so Heyne's "orthogonal"
framing stands or falls on the pre-registered threshold (0.2), not on
post-hoc re-interpretation.

Also writes env/libraries to logs/run.log.
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH = ROOT / "experiments" / "batch_052_A"
INPUT = BATCH / "input"
OUTPUT = BATCH / "output"
LOGS = BATCH / "logs"
OUTPUT.mkdir(exist_ok=True, parents=True)
LOGS.mkdir(exist_ok=True, parents=True)

LOG = LOGS / "run.log"
SEED = 20260424


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_gene_list(path, skip_comments=True):
    genes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if skip_comments and line.startswith("#"):
                continue
            genes.append(line.split("\t")[0].split(",")[0].strip())
    return sorted(set(g for g in genes if g))


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    log("=" * 70)
    log("batch_052_A/01_assemble_inputs.py starting")
    log(f"python={platform.python_version()} platform={platform.platform()}")
    try:
        import scipy, statsmodels  # noqa
        log(f"numpy={np.__version__} pandas={pd.__version__} scipy={scipy.__version__} statsmodels={statsmodels.__version__}")
    except Exception as e:
        log(f"version capture partial: {e}")
    log(f"seed={SEED}")

    # -------- Gene lists --------
    schema_p = ROOT / "experiments/batch_044/input/schema_exome_wide_significant.txt"
    asd_p = ROOT / "experiments/batch_050/input/asd_satterstrom_2020_fdr10.txt"
    ddd_p = ROOT / "experiments/batch_050/input/ddd_kaplanis_2020.txt"
    heyne_p = INPUT / "heyne_2018_dee_33.txt"

    schema = read_gene_list(schema_p)
    asd = read_gene_list(asd_p)
    ddd = read_gene_list(ddd_p)
    heyne = read_gene_list(heyne_p)

    lists = {
        "SCHEMA": {"genes": schema, "n": len(schema), "source": str(schema_p)},
        "ASD_FDR10": {"genes": asd, "n": len(asd), "source": str(asd_p)},
        "DDD_Kaplanis": {"genes": ddd, "n": len(ddd), "source": str(ddd_p)},
        "Heyne_DEE": {"genes": heyne, "n": len(heyne), "source": str(heyne_p)},
    }
    for k, v in lists.items():
        log(f"list {k}: n={v['n']}  src={v['source']}")

    # -------- Jaccard matrix --------
    names = list(lists.keys())
    jacc = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            j = jaccard(lists[a]["genes"], lists[b]["genes"])
            key = f"{a}__{b}"
            jacc[key] = {"jaccard": j,
                         "intersection": sorted(set(lists[a]["genes"]) & set(lists[b]["genes"]))}
            log(f"Jaccard({a},{b}) = {j:.4f}  (|∩|={len(jacc[key]['intersection'])})")

    # -------- File fingerprints --------
    data_files = {
        "gnomad_constraint": ROOT / "data/item_15/gnomad.v4.1.constraint_metrics.tsv",
        "s_het_genebayes": INPUT / "s_het_genebayes.tsv",
        "hgnc_complete": INPUT / "hgnc_complete_set.txt",
        "hgnc_family": ROOT / "experiments/batch_052_B/input/hgnc_family.tsv",
        "syngo_gmt": INPUT / "syngo_2024.gmt",
        "brainspan_expr": ROOT / "experiments/batch_041/output/brainspan_rnaseq_genes/expression_matrix.csv",
        "brainspan_rows": ROOT / "experiments/batch_041/output/brainspan_rnaseq_genes/rows_metadata.csv",
        "brainspan_cols": ROOT / "experiments/batch_041/output/brainspan_rnaseq_genes/columns_metadata.csv",
    }
    shas = {}
    for label, p in data_files.items():
        if not p.exists():
            log(f"WARNING missing file: {label} -> {p}")
            shas[label] = {"path": str(p), "exists": False}
            continue
        sha = sha256_file(p)
        size = p.stat().st_size
        shas[label] = {"path": str(p), "sha256": sha, "bytes": size}
        log(f"sha256({label}) = {sha[:16]}...  bytes={size:,}")

    # -------- Heyne orthogonality pre-registered branch --------
    j_heyne_ddd = jacc["DDD_Kaplanis__Heyne_DEE"]["jaccard"]
    j_heyne_schema = jacc["SCHEMA__Heyne_DEE"]["jaccard"]
    j_heyne_asd = jacc["ASD_FDR10__Heyne_DEE"]["jaccard"]
    heyne_branch = (
        "DDD_SUBTYPE_NOT_ORTHOGONAL" if j_heyne_ddd > 0.2 else "ORTHOGONAL_COMPARATOR"
    )
    log(f"Heyne pre-registered branch = {heyne_branch}  (Jaccard_Heyne_DDD={j_heyne_ddd:.4f}, threshold=0.2)")

    # -------- gnomAD canonical+MANE universe size (quick sanity check) --------
    log("Scanning gnomAD constraint metrics for canonical+MANE universe...")
    gc = pd.read_csv(data_files["gnomad_constraint"], sep="\t", low_memory=False)
    mask = (gc["canonical"].astype(str).str.lower() == "true") & (
        gc["mane_select"].astype(str).str.lower() == "true"
    )
    gc_mm = gc[mask].drop_duplicates(subset="gene")
    log(f"gnomAD v4.1 canonical+MANE unique genes: {len(gc_mm):,}")
    gnomad_universe = sorted(gc_mm["gene"].dropna().astype(str).unique())

    # -------- List intersection with gnomAD universe (baseline for axes C/E) --------
    intersect = {}
    for k, v in lists.items():
        inter = sorted(set(v["genes"]) & set(gnomad_universe))
        intersect[k] = {"n_in_gnomad_canonical_mane": len(inter),
                        "genes_in_gnomad": inter}
        log(f"{k}: {len(inter)}/{v['n']} genes in gnomAD canonical+MANE")

    # -------- Dump --------
    payload = {
        "seed": SEED,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lists": lists,
        "jaccard": jacc,
        "heyne_preregistered_branch": heyne_branch,
        "heyne_preregistered_jaccard_threshold": 0.2,
        "file_fingerprints": shas,
        "gnomad_canonical_mane_n": len(gc_mm),
        "intersection_with_gnomad": intersect,
    }
    out = INPUT / "assembled_inputs.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log(f"Wrote {out}")
    log("01_assemble_inputs.py DONE")


if __name__ == "__main__":
    main()
