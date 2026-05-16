#!/usr/bin/env python3
"""
FAP NFKBIA/NFKBIZ within-subtype correlations - Exploratory Analysis
Run via: python3 batch_035/run.py
"""

import json
import numpy as np
from scipy.stats import spearmanr
import scanpy as sc
import warnings
import sys

warnings.filterwarnings('ignore')

# SASP12 gene list
SASP12 = ["CCL2", "CCL7", "CCL8", "CXCL6", "CXCL8", "IL6", "IL1B", "MMP1", "MMP3", "SERPINE1", "PLAU"]
TARGET_GENES = ["NFKBIA", "NFKBIZ", "JUNB", "CEBPB", "RUNX2"]

print("=" * 60, file=sys.stderr)
print("batch_035: FAP NFKB/NFKBIZ within-subtype correlations", file=sys.stderr)
print("=" * 60, file=sys.stderr)

# Load with backed mode
print("\n[1] Loading FAP atlas with backed mode...", file=sys.stderr)
adata = sc.read_h5ad('/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad', backed='r')
print(f"    Shape: {adata.shape}", file=sys.stderr)

# Check available annotations
print(f"    Obs columns: {list(adata.obs.columns)}", file=sys.stderr)

# Identify subtype column
subtype_col = None
for col in ['subtype', 'cluster', 'cell_type', 'leiden', 'celltype', 'CellType', 'Annotation']:
    if col in adata.obs.columns:
        subtype_col = col
        print(f"    Using subtype column: {col}", file=sys.stderr)
        break

if subtype_col is None:
    print("    WARNING: No subtype column found", file=sys.stderr)
    subtypes = ['all']
else:
    subtypes = adata.obs[subtype_col].cat.categories.tolist()
    print(f"    Subtypes found: {subtypes}", file=sys.stderr)

# Get gene indices
var_names = adata.var_names.tolist()
gene_idx = {}
for gene in SASP12 + TARGET_GENES:
    if gene in var_names:
        gene_idx[gene] = var_names.index(gene)

print(f"\n[2] Found {len(gene_idx)} of {len(SASP12) + len(TARGET_GENES)} target genes", file=sys.stderr)
missing = [g for g in SASP12 + TARGET_GENES if g not in gene_idx]
if missing:
    print(f"    Missing: {missing}", file=sys.stderr)

# Subsample if needed
n_cells = adata.shape[0]
MAX_CELLS = 10000
if n_cells > MAX_CELLS:
    print(f"\n[3] Subsampling from {n_cells} to {MAX_CELLS} cells", file=sys.stderr)
    np.random.seed(42)
    idx = np.random.choice(n_cells, MAX_CELLS, replace=False)
    idx = np.sort(idx)
else:
    idx = None

# Results structure
results = {
    "analysis": "FAP NFKBIA/NFKBIZ within-subtype correlations",
    "subtypes_found": subtypes,
    "correlations": {},
    "exploratory": True
}

# Process each subtype
for subtype in subtypes:
    print(f"\n[4] Processing {subtype}...", file=sys.stderr)

    if subtype == 'all':
        mask = np.ones(n_cells, dtype=bool)
    else:
        mask = (adata.obs[subtype_col] == subtype).values

    if idx is not None:
        mask = mask[idx]

    cell_count = mask.sum()
    print(f"    {cell_count} cells in mask", file=sys.stderr)

    if cell_count < 50:
        print(f"    Skipping - too few cells", file=sys.stderr)
        continue

    # Extract expression
    exprs = {}
    for gene in gene_idx:
        if idx is not None:
            gene_expr = adata.X[idx, gene_idx[gene]]
        else:
            gene_expr = adata.X[:, gene_idx[gene]]

        if hasattr(gene_expr, 'toarray'):
            gene_expr = gene_expr.toarray().flatten()
        else:
            gene_expr = np.array(gene_expr).flatten()

        exprs[gene] = gene_expr[mask]

    # Compute SASP12 composite
    sasp_genes_found = [g for g in SASP12 if g in gene_idx]
    if sasp_genes_found:
        sasp_expr = np.zeros(cell_count)
        for g in sasp_genes_found:
            sasp_expr += exprs[g]
        sasp_expr = sasp_expr / len(sasp_genes_found)
    else:
        print(f"    ERROR: No SASP12 genes found", file=sys.stderr)
        continue

    # Compute correlations
    corrs = {
        "n_cells": int(cell_count),
        "sasp_genes_found": len(sasp_genes_found)
    }

    for gene in TARGET_GENES:
        if gene in exprs:
            rho, pval = spearmanr(exprs[gene], sasp_expr)
            corrs[f"{gene}_rho"] = float(rho)
            corrs[f"{gene}_pval"] = float(pval)

    results["correlations"][subtype] = corrs

    # Print summary
    nfkb_rho = corrs.get("NFKBIA_rho", "NA")
    nfkbiz_rho = corrs.get("NFKBIZ_rho", "NA")
    junb_rho = corrs.get("JUNB_rho", "NA")
    print(f"    NFKBIA rho: {nfkb_rho:.3f}" if isinstance(nfkb_rho, float) else f"    NFKBIA: {nfkb_rho}", file=sys.stderr)
    print(f"    NFKBIZ rho: {nfkbiz_rho:.3f}" if isinstance(nfkbiz_rho, float) else f"    NFKBIZ: {nfkbiz_rho}", file=sys.stderr)
    print(f"    JUNB rho: {junb_rho:.3f}" if isinstance(junb_rho, float) else f"    JUNB: {junb_rho}", file=sys.stderr)

# Key finding
results["key_finding"] = "NFKBIA/NFKBIZ correlation with SASP12 analyzed across FAP subtypes"

# Save
import os
outdir = '/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_035'
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'results.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[5] Results saved to {outpath}", file=sys.stderr)
print("\nDONE", file=sys.stderr)
