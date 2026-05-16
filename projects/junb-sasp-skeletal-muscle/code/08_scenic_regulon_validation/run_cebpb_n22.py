#!/usr/bin/env python3
"""
batch_057: CEBPB with N=22 (all HLMA FAP donors)
Compares N=16 (snRNA-only) vs N=22 (all cells) for CEBPB-SASP.
Also computes raw mRNA correlations for all 22 donors.
"""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
import os

# ============================================================
# PART 1: Raw mRNA correlations with all 22 donors
# ============================================================
print("=" * 60)
print("CEBPB mRNA-SASP: N=16 vs N=22 comparison")
print("=" * 60)

adata = sc.read_h5ad("data/OMIX004308-02.h5ad")
print(f"Total cells: {adata.n_obs:,}")

SASP12 = ["CXCL1", "CXCL2", "CXCL3", "IL8", "IL1B", "CCL2", "CCL20",
          "CXCL6", "PLAU", "PLAUR", "TIMP1", "MMP1"]
detected = [g for g in SASP12 if g in adata.var_names]
adata.obs["SASP_score"] = np.asarray(adata[:, detected].X).mean(axis=1).flatten()

# Target TFs
TF_LIST = ["CEBPB", "JUNB", "KLF10", "EGR2", "CDKN1A", "ATF3", "FOSL1", "FOS"]
for tf in TF_LIST:
    adata.obs[f"raw_{tf}"] = np.asarray(adata[:, tf].X).flatten()

# Per-donor means (ALL cells, N=22)
donor_all = adata.obs.groupby("sample").agg(
    n_cells=("SASP_score", "size"),
    SASP=("SASP_score", "mean"),
    Country=("Country", "first"),
    age=("age", "first"),
    **{f"raw_{tf}": (f"raw_{tf}", "mean") for tf in TF_LIST}
)
# Filter min 50 cells
donor_all = donor_all[donor_all["n_cells"] >= 50]
print(f"\nDonors (N=22 all, ≥50 cells): {len(donor_all)}")
print(f"  China: {(donor_all['Country']=='China').sum()}")
print(f"  Spain: {(donor_all['Country']=='Spain').sum()}")

# Compute rho for each TF (all 22)
results_all = []
for tf in TF_LIST:
    valid = donor_all[[f"raw_{tf}", "SASP"]].dropna()
    rho, p = spearmanr(valid[f"raw_{tf}"], valid["SASP"])
    results_all.append({"TF": tf, "N": len(valid), "rho": round(rho,4), "p_value": p})

# Country stratification
results_china = []
results_spain = []
for tf in TF_LIST:
    col = f"raw_{tf}"
    cn = donor_all[donor_all["Country"]=="China"]
    es = donor_all[donor_all["Country"]=="Spain"]
    if len(cn) >= 5:
        rho, p = spearmanr(cn[col], cn["SASP"])
        results_china.append({"TF": tf, "N": len(cn), "rho": round(rho,4), "p_value": p})
    if len(es) >= 5:
        rho, p = spearmanr(es[col], es["SASP"])
        results_spain.append({"TF": tf, "N": len(es), "rho": round(rho,4), "p_value": p})

print("\n--- All donors (N=22) ---")
for r in results_all:
    print(f"  {r['TF']}: N={r['N']}, rho={r['rho']:+.4f}, p={r['p_value']:.2e}")

print("\n--- China only ---")
for r in results_china:
    print(f"  {r['TF']}: N={r['N']}, rho={r['rho']:+.4f}, p={r['p_value']:.2e}")

print("\n--- Spain only ---")
for r in results_spain:
    print(f"  {r['TF']}: N={r['N']}, rho={r['rho']:+.4f}, p={r['p_value']:.2e}")

# ============================================================
# PART 2: CEBPB AUCell — load batch_055 (N=16) vs compute new for N=22
# ============================================================
print("\n" + "=" * 60)
print("CEBPB AUCell: N=16 (snRNA-only) vs N=22 (all cells)")
print("=" * 60)

# Load batch_055 AUCell donor averages (N=16 snRNA-only)
batch055 = pd.read_csv("experiments/batch_055/d1_donor_averages_HLMA_FAP.csv")
print(f"batch_055 N=16 (snRNA-only): {[s for s in batch055['sample'].values[:5]]}...")

# The AUCell scores from batch_055 are already donor-averaged.
# We need to compare: can we compute AUCell for the 6 additional scRNA-only donors?
# The GRNBoost2 network was trained on snRNA-only cells.
# AUCell can be computed on any cell using the same regulons.

# Check which donors from N=22 are NOT in batch_055
batch055_donors = set(batch055["sample"].values)
all_donors = set(donor_all.index)
new_donors = all_donors - batch055_donors
print(f"\nNew donors (in N=22 but not N=16): {new_donors}")

# Check tech for new donors
new_df = donor_all.loc[list(new_donors)]
print(f"  Tech breakdown:")
if "tech" in adata.obs.columns:
    tech_map = adata.obs.groupby("sample")["tech"].first()
    for d in new_donors:
        if d in tech_map.index:
            print(f"    {d}: {tech_map[d]}")

# For CEBPB AUCell comparison:
# Option A: Only compare donors present in BOTH (intersection)
# Option B: Re-run AUCell on ALL cells using batch_055 regulons

# Let's do Option A: N=16 intersection
common_donors = batch055_donors & all_donors
print(f"\nCommon donors: {len(common_donors)}")

# Correlate CEBPB AUCell vs SASP in common donors
if "aucell_CEBPB(+)" in batch055.columns:
    batch055_idx = batch055.set_index("sample")
    common_idx = donor_all.loc[list(common_donors)]

    merged = common_idx[["SASP"]].join(batch055_idx[["aucell_CEBPB(+)", "raw_CEBPB"]], how="inner")
    print(f"  Merged: {len(merged)} donors")

    rho_aucell, p_aucell = spearmanr(merged["aucell_CEBPB(+)"], merged["SASP"])
    rho_raw, p_raw = spearmanr(merged["raw_CEBPB"], merged["SASP"])
    print(f"  AUCell (N={len(merged)}): rho={rho_aucell:.4f}, p={p_aucell:.2e}")
    print(f"  Raw mRNA (N={len(merged)}): rho={rho_raw:.4f}, p={p_raw:.2e}")

# Key comparison: CEBPB raw mRNA N=22 vs N=16
cebp_n22 = [r for r in results_all if r["TF"] == "CEBPB"][0]
print(f"\n--- CEBPB comparison ---")
print(f"  CEBPB raw mRNA N=22: rho={cebp_n22['rho']:+.4f}, p={cebp_n22['p_value']:.2e}")
print(f"  CEBPB AUCell N=16:   rho={rho_aucell:.4f}, p={p_aucell:.2e}")
print(f"  Delta AUCell-mRNA (N=16): {rho_aucell - rho_raw:.4f}")

# Also show all TFs at N=22 for comparison
print("\n--- All TFs raw mRNA at N=22 ---")
results_df = pd.DataFrame(results_all)
print(results_df.to_string(index=False))

# ============================================================
# PART 3: FAP subtype-level CEBPB with N=22
# ============================================================
print("\n" + "=" * 60)
print("CEBPB mRNA-SASP per FAP subtype (N=22, donor-level)")
print("=" * 60)

for subtype in sorted(adata.obs["Annotation"].unique()):
    mask = adata.obs["Annotation"] == subtype
    sub = adata[mask]
    if sub.n_obs < 100:
        continue
    # Per-donor means
    sub_donor = sub.obs.groupby("sample").agg(
        n_cells=("SASP_score", "size"),
        CEBPB=("raw_CEBPB", "mean"),
        SASP=("SASP_score", "mean"),
        Country=("Country", "first")
    ).dropna()
    sub_donor = sub_donor[sub_donor["n_cells"] >= 50]
    if len(sub_donor) >= 5:
        rho, p = spearmanr(sub_donor["CEBPB"], sub_donor["SASP"])
        # China-only
        cn = sub_donor[sub_donor["Country"]=="China"]
        rho_cn = np.nan
        if len(cn) >= 5:
            rho_cn, _ = spearmanr(cn["CEBPB"], cn["SASP"])
        print(f"  {subtype}: N={len(sub_donor)}, rho={rho:.4f} (p={p:.2e}), China N={len(cn)}, rho_cn={rho_cn:.4f if not np.isnan(rho_cn) else 'NS'}")

# ============================================================
# Save
# ============================================================
outdir = "experiments/batch_057"
os.makedirs(outdir, exist_ok=True)
donor_all.to_csv(f"{outdir}/donor_level_all_tfs_n22.csv")
results_df.to_csv(f"{outdir}/tf_correlations_n22.csv", index=False)

print(f"\nSaved to {outdir}/")
print("DONE")