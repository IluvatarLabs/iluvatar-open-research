#!/usr/bin/env python3
"""
batch_057 Part 2: Re-run CEBPB AUCell with ALL HLMA FAP donors (N=22)
Uses batch_055 GRNBoost2 network, applied to all cells (snRNA + scRNA).
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CEBPB AUCell with N=22 (all HLMA FAP donors)")
print("=" * 60)

# Step 1: Extract CEBPB regulon from batch_055 adjacency file
adj_file = "experiments/batch_055/d1_adjacencies_HLMA_FAP.csv"
print(f"Loading GRNBoost2 adjacencies: {adj_file}")
adj = pd.read_csv(adj_file)
print(f"  Shape: {adj.shape}")
print(f"  Columns: {list(adj.columns)}")

# Extract CEBPB regulon targets (top 80th percentile importance)
cepb_adj = adj[adj["TF"] == "CEBPB"].copy()
print(f"  CEBPB edges: {len(cepb_adj)}")

if len(cepb_adj) > 0:
    # Use same threshold as batch_055: top 80th percentile
    threshold = cepb_adj["importance"].quantile(0.80)
    cebpb_targets = cepb_adj[cepb_adj["importance"] >= threshold]["target"].tolist()
    # Cap at 200 targets (same as batch_055)
    cebpb_targets = cebpb_targets[:200]
    print(f"  CEBPB regulon targets (80th pct): {len(cepb_targets)}")
else:
    print("  WARNING: No CEBPB edges found. Using all TFs.")
    cebpb_targets = []

# Also extract top TFs for comparison
target_tfs = ["CEBPB", "JUNB", "KLF10", "EGR2", "CDKN1A", "ATF3", "FOS", "FOSL1"]
regulons = {}
for tf in target_tfs:
    tf_adj = adj[adj["TF"] == tf]
    if len(tf_adj) > 0:
        threshold = tf_adj["importance"].quantile(0.80)
        targets = tf_adj[tf_adj["importance"] >= threshold]["target"].tolist()[:200]
        regulons[tf] = targets
        print(f"  {tf}: {len(targets)} targets")

# Step 2: Load ALL FAP cells (snRNA + scRNA)
print("\nLoading ALL FAP cells...")
adata = sc.read_h5ad("data/OMIX004308-02.h5ad")
print(f"  Cells: {adata.n_obs:,}")

# SASP12
SASP12 = ["CXCL1", "CXCL2", "CXCL3", "IL8", "IL1B", "CCL2", "CCL20",
          "CXCL6", "PLAU", "PLAUR", "TIMP1", "MMP1"]
detected = [g for g in SASP12 if g in adata.var_names]
adata.obs["SASP_score"] = np.asarray(adata[:, detected].X).mean(axis=1).flatten()

# Step 3: Compute AUCell per regulon
# AUCell: for each cell, rank genes by expression, compute AUC of the gene set in the ranking
from itertools import repeat
import multiprocessing as mp

def compute_aucell_single(args):
    """Compute AUCell for one gene set on one cell's ranked expression."""
    gene_set, expression, all_genes = args
    # Rank genes by expression (descending)
    ranked = np.argsort(-expression)
    # Find positions of gene set members in the ranking
    gene_set_idx = set()
    for g in gene_set:
        matches = np.where(all_genes == g)[0]
        if len(matches) > 0:
            gene_set_idx.add(matches[0])
    if len(gene_set_idx) == 0:
        return 0.0
    # Compute AUC
    n_genes = len(expression)
    n_set = len(gene_set_idx)
    # Recovery curve: at each rank, how many gene set members have been seen?
    ranks = np.zeros(n_genes, dtype=bool)
    for idx in gene_set_idx:
        rank_pos = np.searchsorted(ranked, idx)
        if rank_pos < n_genes:
            ranks[rank_pos] = True
    # Cumulative sum of recovery
    cumsum = np.cumsum(ranks).astype(float)
    # Normalize by max possible AUC
    max_auc = n_set * (n_genes - n_set + n_set) / 2.0  # wrong, use proper formula
    # AUCell formula: AUC = sum of (n_genes - rank_i) for each gene set member
    # Simplified: area under the recovery curve
    total = cumsum.sum()
    # Normalize to [0, 1]
    max_possible = float(n_set * (n_genes - n_set + 1)) / 2.0
    if max_possible == 0:
        return 0.0
    auc = total / max_possible
    return auc

# Actually, let's use a simpler approach with AUCell from pySCENIC
try:
    from ctxcore.genesig import GeneSignature
    from pyscenic.aucell import aucell

    print("\nUsing pySCENIC AUCell implementation...")

    # Create gene signatures
    gs_list = []
    for tf, targets in regulons.items():
        # Filter to expressed genes
        expr_targets = [g for g in targets if g in adata.var_names]
        if len(expr_targets) >= 5:
            gs = GeneSignature(name=f"{tf}(+)", gene2weight={g: 1.0 for g in expr_targets})
            gs_list.append(gs)

    print(f"  Computing AUCell for {len(gs_list)} regulons on {adata.n_obs:,} cells...")

    # Create expression matrix for AUCell
    exp_matrix = pd.DataFrame(
        adata.X.T,
        index=adata.var_names,
        columns=adata.obs.index
    )

    # Run AUCell
    aucell_df = aucell(exp_matrix, gs_list, seed=42, num_workers=4)
    print(f"  AUCell complete: {aucell_df.shape}")

    # Add AUCell scores to adata.obs
    for col in aucell_df.columns:
        adata.obs[f"aucell_{col}"] = aucell_df[col].values

    # Step 4: Donor-level correlations (N=22)
    print("\n--- Donor-level AUCell-SASP correlations (N=22) ---")

    donor_means = adata.obs.groupby("sample").agg(
        n_cells=("SASP_score", "size"),
        SASP_mean=("SASP_score", "mean"),
        Country=("Country", "first"),
        age=("age", "first"),
        **{f"aucell_{tf}(+)": (f"aucell_{tf}(+)", "mean") for tf in regulons.keys() if f"aucell_{tf}(+)" in adata.obs.columns}
    )

    # Filter min 50 cells
    donor_means = donor_means[donor_means["n_cells"] >= 50]
    print(f"  Donors with ≥50 cells: {len(donor_means)}")

    # Correlate each TF AUCell with SASP
    results = []
    for tf in regulons.keys():
        col = f"aucell_{tf}(+)"
        if col not in donor_means.columns:
            continue
        valid = donor_means[[col, "SASP_mean"]].dropna()
        if len(valid) < 5:
            continue
        rho, p = spearmanr(valid[col], valid["SASP_mean"])
        # Also raw mRNA
        raw_col = f"raw_{tf}"
        if tf in adata.var_names:
            adata.obs[raw_col] = np.asarray(adata[:, tf].X).flatten()
            raw_donor = adata.obs.groupby("sample").agg({raw_col: "mean"})
            merged = valid.join(raw_donor, how="inner")
            raw_rho, raw_p = spearmanr(merged[raw_col], merged["SASP_mean"])
        else:
            raw_rho = np.nan
        delta = rho - raw_rho

        results.append({
            "TF": tf,
            "n_donors": len(valid),
            "aucell_rho": round(rho, 4),
            "aucell_p": p,
            "raw_mrna_rho": round(raw_rho, 4),
            "delta_rho": round(delta, 4)
        })
        print(f"  {tf}: N={len(valid)}, AUCell rho={rho:.4f} (p={p:.2e}), mRNA rho={raw_rho:.4f}, delta={delta:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv("experiments/batch_057/cebpb_aucell_n22.csv", index=False)

    # Country stratification for CEBPB
    print("\n--- CEBPB country stratification (N=22) ---")
    cebpb_col = "aucell_CEBPB(+)"
    if cebpb_col in donor_means.columns:
        for country in ["China", "Spain"]:
            sub = donor_means[donor_means["Country"] == country]
            if len(sub) >= 5:
                rho, p = spearmanr(sub[cebpb_col], sub["SASP_mean"])
                print(f"  {country} (N={len(sub)}): rho={rho:.4f}, p={p:.2e}")
            else:
                print(f"  {country} (N={len(sub)}): too few donors")

    # Save donor means
    donor_means.to_csv("experiments/batch_057/donor_aucell_means_n22.csv")

except ImportError:
    print("pySCENIC AUCell not available — using manual AUCell implementation")

    # Manual AUCell implementation
    print("\nComputing manual AUCell...")

    X = adata.X  # (n_cells, n_genes)
    var_names = adata.var_names.values

    for tf, targets in regulons.items():
        # Filter targets to expressed genes
        target_idx = [i for i, g in enumerate(var_names) if g in targets]
        if len(target_idx) < 5:
            print(f"  {tf}: too few targets ({len(target_idx)})")
            continue

        n_cells = X.shape[0]
        n_genes = X.shape[1]
        n_set = len(target_idx)

        # For each cell, compute AUC
        aucs = np.zeros(n_cells)
        for i in range(n_cells):
            expr = X[i]
            # Rank genes
            order = np.argsort(-expr)
            # Positions of target genes in ranking
            ranks_of_targets = np.zeros(n_set, dtype=int)
            for j, idx in enumerate(target_idx):
                rank_pos = np.searchsorted(order, idx)
                ranks_of_targets[j] = rank_pos

            # AUC = sum(max_rank - rank_i) for each target / max_possible
            max_auc = n_set * (n_genes - (n_set - 1) / 2)
            auc = np.sum(n_genes - ranks_of_targets) / max_auc
            aucs[i] = auc

        adata.obs[f"aucell_{tf}(+)"] = aucs
        print(f"  {tf}: {n_set} targets, AUC range [{aucs.min():.4f}, {aucs.max():.4f}]")

    # Donor-level correlation
    print("\n--- Donor-level AUCell-SASP correlations (manual) ---")
    donor_means = adata.obs.groupby("sample").agg(
        n_cells=("SASP_score", "size"),
        SASP_mean=("SASP_score", "mean"),
        Country=("Country", "first"),
    )

    # Add AUCell columns
    for tf in regulons.keys():
        col = f"aucell_{tf}(+)"
        if col in adata.obs.columns:
            donor_means[col] = adata.obs.groupby("sample")[col].mean()

    donor_means = donor_means[donor_means["n_cells"] >= 50]
    print(f"  Donors: {len(donor_means)}")

    for tf in regulons.keys():
        col = f"aucell_{tf}(+)"
        if col not in donor_means.columns:
            continue
        valid = donor_means[[col, "SASP_mean"]].dropna()
        if len(valid) >= 5:
            rho, p = spearmanr(valid[col], valid["SASP_mean"])
            print(f"  {tf}: N={len(valid)}, rho={rho:.4f}, p={p:.2e}")

print("\nDONE")
