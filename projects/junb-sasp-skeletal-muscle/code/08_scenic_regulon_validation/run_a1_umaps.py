#!/usr/bin/env python3
"""
batch_052 A1: UMAP Figures per Compartment (HLMA Vascular, MuSC, FAP)

Generates 4-panel UMAP figures for each compartment:
  A: cell type annotation
  B: age group (young_pop / old_pop)
  C: JUNB expression (normalized, per-compartment z-scored)
  D: SASP12 composite score (mean of detected SASP genes)

Also saves donor-level summary statistics to CSV.

Data state (verified):
  - Vascular: raw counts (0.0 min, max 4.4, all non-negative)
  - MuSC:    raw counts (0.0 min, max 8.26, all non-negative)
  - FAP:     already log-normalized (min=-15.7, max=10, ~89% negative values)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.sparse import issparse
import warnings
import os

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=150, facecolor="white")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTDIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_052"
os.makedirs(OUTDIR, exist_ok=True)

DATA_FILES = {
    "vascular": "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/Vascular_scsn_RNA.h5ad",
    "musc":     "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/MuSC_scsn_RNA.h5ad",
    "fap":      "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad",
}

SASP12 = ["CCL2", "CXCL1", "CXCL2", "CXCL3", "CXCL6", "IL6",
          "CXCL8", "SERPINE1", "MMP1", "MMP3", "PLAU", "PLAUR"]

VASCULAR_ENDO_TYPES   = ["ArtEC", "CapEC", "VenEC", "IL6+ VenEC"]
FAP_SUBTYPES          = ["MME+ FAP", "CD55+ FAP", "GPC3+ FAP", "RUNX2+ FAP", "CD99+ FAP"]
# MuSC: all cells used

RANDOM_STATE = 42
N_PCS        = 30
N_NEIGHBORS  = 15
MIN_DIST     = 0.3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_filter_adata(path, name):
    """
    Load h5ad and return AnnData, plus a boolean mask of cells to KEEP.

    Filtering logic per compartment:
      Vascular: keep endothelial types only
      MuSC:     keep all cells
      FAP:      keep FAP subtypes only
    """
    ad = sc.read_h5ad(path)
    ad.var_names = ad.var["features"].tolist()

    if name == "vascular":
        mask = ad.obs["Annotation"].isin(VASCULAR_ENDO_TYPES)
        ad._inplace_subset_obs(mask)
        print(f"  [{name}] Kept {ad.n_obs} endothelial cells (from {mask.sum()}/{len(mask)} matched)")
    elif name == "musc":
        # Keep all MuSC cells
        print(f"  [{name}] Keeping all {ad.n_obs} cells")
    elif name == "fap":
        mask = ad.obs["Annotation"].isin(FAP_SUBTYPES)
        ad._inplace_subset_obs(mask)
        print(f"  [{name}] Kept {ad.n_obs} FAP cells (from {mask.sum()}/{len(mask)} matched)")

    return ad


def is_already_normalized(adata):
    """
    Heuristic: if >30% of values are negative, data is already log-normalized.
    Raw counts are non-negative. Log-normalized data has negative values
    where raw count < 1.
    """
    x = adata.X
    if issparse(x):
        x = x.toarray().ravel()
    else:
        x = np.asarray(x).ravel()
    pct_neg = np.sum(x < 0) / x.size * 100
    return pct_neg > 30


def compute_sasp12_score(adata, genes):
    """
    Per-cell SASP12 composite score:
    - For each gene, count it as detected if expression > 0 (after log-normalization).
    - Composite = mean expression of detected genes.
    - Returns vector of scores (same length as n_cells).
    """
    gene_idx = []
    present_genes = []
    for g in genes:
        if g in adata.var_names:
            gene_idx.append(adata.var_names.get_loc(g))
            present_genes.append(g)

    if not present_genes:
        return np.zeros(adata.n_obs)

    X = adata[:, present_genes].X
    if issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    # Detected = positive expression
    detected = X > 0
    # Composite = mean of detected genes; if none detected, score = 0
    scores = np.zeros(adata.n_obs)
    for j in range(len(present_genes)):
        scores += np.where(detected[:, j], X[:, j], 0.0)
    # Normalize by number of detected genes per cell
    n_detected = detected.sum(axis=1)
    scores = np.divide(scores, np.maximum(n_detected, 1),
                       out=scores, where=np.maximum(n_detected, 1) > 0)
    return scores


def run_pca_umap(adata, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS,
                 min_dist=MIN_DIST, random_state=RANDOM_STATE):
    """
    Compute PCA (highly variable genes, n_pcs) then UMAP.
    Stores obs['pca'] and obs['umap'] coordinates.
    """
    # Highly variable genes
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
    adata_hvg = adata[:, adata.var.highly_variable]

    # PCA on HVG
    n_pcs_actual = min(n_pcs, adata_hvg.n_vars - 1, adata_hvg.n_obs - 1)
    sc.tl.pca(adata_hvg, n_comps=n_pcs_actual, random_state=random_state, svd_solver="arpack")
    # Transfer PCA to full object
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    # UMAP on PCA
    n_neighbors_actual = min(n_neighbors, adata.n_obs - 1)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors_actual, n_pcs=n_pcs_actual,
                    random_state=random_state)
    sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

    return adata


def plot_4panel(adata, compart, junb_expr, sasp12_score, outpath):
    """
    Create 4-panel figure:
      A: cell type annotation
      B: age_pop (young_pop / old_pop)
      C: JUNB expression (z-scored per-compartment)
      D: SASP12 composite score

    UMAP coordinates from adata.obsm['X_umap'].
    """
    umap_coords = adata.obsm["X_umap"]
    n_cells = adata.n_obs

    # JUNB z-score per-compartment (for coloring scale)
    if junb_expr.size > 0:
        j_mean, j_std = np.mean(junb_expr), np.std(junb_expr)
        junb_z = (junb_expr - j_mean) / (j_std + 1e-12)
    else:
        junb_z = np.zeros(n_cells)

    # SASP12 score already per-cell; z-score for coloring
    if sasp12_score.size > 0:
        s_mean, s_std = np.mean(sasp12_score), np.std(sasp12_score)
        sasp_z = (sasp12_score - s_mean) / (s_std + 1e-12)
    else:
        sasp_z = np.zeros(n_cells)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(f"HLMA {compart.upper()} UMAP (batch_052 A1)", fontsize=14, y=1.02)

    panel_labels = ["A", "B", "C", "D"]
    titles = [
        "A: Cell Type Annotation",
        "B: Age Group",
        "C: JUNB Expression (z-score)",
        "D: SASP12 Composite Score (z-score)",
    ]

    # ---- Panel A: cell type annotation ----
    ax = axes[0]
    annotations = adata.obs["Annotation"].astype(str)
    cats = annotations.unique()
    cmap_palette = plt.cm.tab20(np.linspace(0, 1, len(cats)))
    color_map = {c: cmap_palette[i] for i, c in enumerate(cats)}
    colors = annotations.map(color_map)
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=colors,
               s=1.5, alpha=0.6, rasterized=True)
    # Legend outside
    handles = [matplotlib.patches.Patch(color=color_map[c], label=c) for c in cats]
    ax.legend(handles=handles, loc="upper left", fontsize=5.5,
              title="Annotation", title_fontsize=6, framealpha=0.7,
              markerscale=3)
    ax.set_title(titles[0], fontsize=10)
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # ---- Panel B: age group ----
    ax = axes[1]
    # Convert Categorical to str to avoid TypeError on fillna
    age_col = adata.obs["age_pop"].astype(str)
    age_color_map = {"young_pop": "#1e90ff", "old_pop": "#ff4500"}
    # Default gray for any unexpected categories
    age_colors = age_col.map(age_color_map).fillna("#808080")
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=age_colors,
               s=1.5, alpha=0.6, rasterized=True)
    age_cats = [c for c in age_color_map if c in age_col.values]
    handles = [matplotlib.patches.Patch(color=age_color_map[c], label=c) for c in age_cats]
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              title="Age Group", title_fontsize=8, framealpha=0.7,
              markerscale=3)
    ax.set_title(titles[1], fontsize=10)
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # ---- Panel C: JUNB z-score ----
    ax = axes[2]
    vmin, vmax = np.percentile(junb_z, 1), np.percentile(junb_z, 99)
    vmax = max(vmax, vmin + 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    sc_plot = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                         c=junb_z, cmap="RdBu_r", s=1.5, alpha=0.6,
                         norm=norm, rasterized=True)
    plt.colorbar(sc_plot, ax=ax, fraction=0.03, pad=0.04, label="JUNB z-score")
    ax.set_title(titles[2], fontsize=10)
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # ---- Panel D: SASP12 z-score ----
    ax = axes[3]
    vmin, vmax = np.percentile(sasp_z, 1), np.percentile(sasp_z, 99)
    vmax = max(vmax, vmin + 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    sc_plot = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                         c=sasp_z, cmap="RdBu_r", s=1.5, alpha=0.6,
                         norm=norm, rasterized=True)
    plt.colorbar(sc_plot, ax=ax, fraction=0.03, pad=0.04, label="SASP12 z-score")
    ax.set_title(titles[3], fontsize=10)
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def compute_donor_stats(adata, sasp12_score, compart):
    """
    Compute donor-level summary stats for CSV output.
    """
    obs = adata.obs.copy()
    obs["sasp12_score"] = sasp12_score

    # JUNB expression
    if "JUNB" in adata.var_names:
        X_junb = adata[:, "JUNB"].X
        if issparse(X_junb):
            X_junb = X_junb.toarray().ravel()
        else:
            X_junb = np.asarray(X_junb).ravel()
        obs["junb_expr"] = X_junb
    else:
        obs["junb_expr"] = np.nan

    # Aggregate per donor
    group_cols = ["sample", "age_pop", "Country", "tech"]
    available = [c for c in group_cols if c in obs.columns]

    grp = obs.groupby("sample").agg(
        n_cells=("sasp12_score", "count"),
        sasp12_mean=("sasp12_score", "mean"),
        sasp12_std=("sasp12_score", "std"),
        junb_mean=("junb_expr", "mean"),
        age_pop=("age_pop", "first"),
        country=("Country", "first"),
        tech=("tech", "first"),
    )
    grp = grp.reset_index()
    grp["compartment"] = compart

    return grp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("batch_052 A1: UMAP Figures per Compartment")
    print("=" * 60)

    all_donor_stats = []
    results = {}

    for compart, path in DATA_FILES.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {compart.upper()}")
        print(f"Data: {path}")
        print("=" * 60)

        # --- Load & filter ---
        ad = load_and_filter_adata(path, compart)

        # --- Check normalization state ---
        already_norm = is_already_normalized(ad)
        print(f"  Data appears log-normalized: {already_norm}")

        if already_norm:
            # FAP already log-normalized: use as-is
            print(f"  [{compart}] Using data as-is (already log-normalized)")
        else:
            # Vascular/MuSC raw counts: normalize + log1p
            print(f"  [{compart}] Normalizing (total counts + log1p)")
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)

        # --- SASP12 composite score (computed on current X) ---
        print(f"  [{compart}] Computing SASP12 composite score...")
        sasp12_score = compute_sasp12_score(ad, SASP12)
        ad.obs["sasp12_score"] = sasp12_score
        print(f"    SASP12 score range: [{sasp12_score.min():.4f}, {sasp12_score.max():.4f}]")
        print(f"    SASP12 score mean:  {sasp12_score.mean():.4f}")

        # --- JUNB expression (on normalized X) ---
        if "JUNB" in ad.var_names:
            X_junb = ad[:, "JUNB"].X
            if issparse(X_junb):
                junb_expr = np.asarray(X_junb.toarray().ravel())
            else:
                junb_expr = np.asarray(X_junb.ravel())
            ad.obs["junb_expr"] = junb_expr
            print(f"    JUNB expr range: [{junb_expr.min():.4f}, {junb_expr.max():.4f}]")
        else:
            junb_expr = np.array([])
            ad.obs["junb_expr"] = np.nan

        # --- PCA + UMAP ---
        print(f"  [{compart}] Running PCA ({N_PCS} PCs) + UMAP...")
        ad = run_pca_umap(ad)

        # --- Print cell counts ---
        print(f"\n  [{compart}] Cell counts:")
        for col in ["Annotation", "age_pop", "tech", "Country"]:
            if col in ad.obs.columns:
                vc = ad.obs[col].value_counts()
                print(f"    {col}:")
                for val, cnt in vc.items():
                    print(f"      {val}: {cnt}")
        print(f"    Total: {ad.n_obs} cells")

        # --- Plot ---
        out_png = os.path.join(OUTDIR, f"a1_umap_{compart}.png")
        plot_4panel(ad, compart, junb_expr, sasp12_score, out_png)

        # --- Donor stats ---
        donor_df = compute_donor_stats(ad, sasp12_score, compart)
        all_donor_stats.append(donor_df)

        # Store adata with UMAP coords for summary
        results[compart] = {
            "n_cells": ad.n_obs,
            "n_annots": ad.obs["Annotation"].nunique(),
            "annotations": ad.obs["Annotation"].value_counts().to_dict(),
            "age_pops": ad.obs["age_pop"].value_counts().to_dict(),
            "techs": ad.obs["tech"].value_counts().to_dict(),
            "n_donors": ad.obs["sample"].nunique(),
        }

    # -------------------------------------------------------------------------
    # Save donor-level summary CSV
    # -------------------------------------------------------------------------
    all_donor_df = pd.concat(all_donor_stats, ignore_index=True)
    csv_path = os.path.join(OUTDIR, "a1_donor_summary.csv")
    all_donor_df.to_csv(csv_path, index=False)
    print(f"\n  Saved donor summary: {csv_path}")

    # -------------------------------------------------------------------------
    # Print final summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for compart, info in results.items():
        print(f"\n  {compart.upper()}:")
        print(f"    Cells: {info['n_cells']:,}")
        print(f"    Donors: {info['n_donors']}")
        print(f"    Annotations: {info['n_annots']}")
        for annot, cnt in info["annotations"].items():
            print(f"      {annot}: {cnt}")
        print(f"    Age groups:")
        for age, cnt in info["age_pops"].items():
            print(f"      {age}: {cnt}")
        print(f"    Tech:")
        for tech, cnt in info["techs"].items():
            print(f"      {tech}: {cnt}")

    print(f"\n  Output files:")
    for compart in DATA_FILES:
        png = os.path.join(OUTDIR, f"a1_umap_{compart}.png")
        if os.path.exists(png):
            size_kb = os.path.getsize(png) // 1024
            print(f"    {png}  ({size_kb} KB)")
    print(f"    {csv_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()