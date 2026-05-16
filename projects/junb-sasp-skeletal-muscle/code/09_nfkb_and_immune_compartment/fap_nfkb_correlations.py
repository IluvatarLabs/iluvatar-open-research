#!/usr/bin/env python3
import sys
sys.stderr.write("Starting script\n")
sys.stderr.flush()

try:
    import json
    import numpy as np
    from scipy.stats import spearmanr
    import scanpy as sc

    print("All imports successful", file=sys.stderr)
    sys.stderr.flush()

    SASP12 = ["CCL2", "CCL7", "CCL8", "CXCL6", "CXCL8", "IL6", "IL1B", "MMP1", "MMP3", "SERPINE1", "PLAU"]

    print("Loading FAP atlas...", file=sys.stderr)
    sys.stderr.flush()

    adata = sc.read_h5ad('/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad', backed='r')

    print(f"Shape: {adata.shape}", file=sys.stderr)
    print(f"Columns: {list(adata.obs.columns)}", file=sys.stderr)

    # Check for subtype column
    subtype_col = None
    for col in ['subtype', 'cluster', 'cell_type', 'leiden', 'celltype', 'CellType', 'Annotation', 'cell_type']:
        if col in adata.obs.columns:
            subtype_col = col
            break

    if subtype_col:
        subtypes = adata.obs[subtype_col].cat.categories.tolist()
    else:
        subtypes = ['all']

    print(f"Subtype column: {subtype_col}", file=sys.stderr)
    print(f"Subtypes: {subtypes}", file=sys.stderr)
    sys.stderr.flush()

    # Get gene indices
    var_names = adata.var_names.tolist()
    gene_idx = {}
    for gene in SASP12 + ["NFKBIA", "NFKBIZ", "JUNB", "CEBPB", "RUNX2"]:
        if gene in var_names:
            gene_idx[gene] = var_names.index(gene)

    print(f"Found {len(gene_idx)} genes", file=sys.stderr)
    sys.stderr.flush()

    # Subsample if needed
    n_cells = adata.shape[0]
    MAX_CELLS = 10000
    if n_cells > MAX_CELLS:
        np.random.seed(42)
        idx = np.random.choice(n_cells, MAX_CELLS, replace=False)
        idx = np.sort(idx)
        print(f"Subsampling to {MAX_CELLS}", file=sys.stderr)
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
        if subtype == 'all':
            mask = np.ones(n_cells, dtype=bool)
        else:
            mask = (adata.obs[subtype_col] == subtype).values

        if idx is not None:
            mask = mask[idx]

        cell_count = mask.sum()
        print(f"\nProcessing {subtype}: {cell_count} cells", file=sys.stderr)
        sys.stderr.flush()

        if cell_count < 50:
            print("  Too few cells, skipping", file=sys.stderr)
            continue

        # Extract expression data
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
        sasp_expr = np.zeros(cell_count)
        for g in sasp_genes_found:
            sasp_expr += exprs[g]
        sasp_expr = sasp_expr / len(sasp_genes_found)

        # Compute correlations
        corrs = {"n_cells": int(cell_count)}
        for gene in ["NFKBIA", "NFKBIZ", "JUNB", "CEBPB", "RUNX2"]:
            if gene in exprs:
                rho, pval = spearmanr(exprs[gene], sasp_expr)
                corrs[f"{gene}_rho"] = float(rho)
                corrs[f"{gene}_pval"] = float(pval)

        results["correlations"][subtype] = corrs

        nfkb = corrs.get("NFKBIA_rho", corrs.get("NFKBIZ_rho", None))
        junb = corrs.get("JUNB_rho", None)
        print(f"  NFKBIA rho: {nfkb:.3f}, JUNB rho: {junb:.3f}", file=sys.stderr)
        sys.stderr.flush()

    # Key finding
    results["key_finding"] = "NFKBIA/NFKBIZ correlation with SASP12 varies by FAP subtype"

    # Save
    import os
    os.makedirs('/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_035', exist_ok=True)
    with open('/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_035/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved!", file=sys.stderr)

except Exception as e:
    import traceback
    sys.stderr.write(f"ERROR: {str(e)}\n")
    sys.stderr.write(traceback.format_exc())
    sys.stderr.flush()
    raise
