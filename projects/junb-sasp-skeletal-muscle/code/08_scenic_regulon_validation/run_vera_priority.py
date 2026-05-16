#!/usr/bin/env python3
"""
batch_057: VERA Priority — JUNB-SASP within FAP subtypes + CEBPB full N analysis
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')
import os

# ============================================================
# PART 1: JUNB-SASP within FAP subtypes at cell level
# ============================================================
print("=" * 60)
print("PART 1: JUNB-SASP within FAP subtypes (cell-level)")
print("=" * 60)

adata = sc.read_h5ad("data/OMIX004308-02.h5ad")
print(f"FAP cells: {adata.n_obs:,}")

SASP12 = ["CXCL1", "CXCL2", "CXCL3", "IL8", "IL1B", "CCL2", "CCL20",
          "CXCL6", "PLAU", "PLAUR", "TIMP1", "MMP1"]

detected = [g for g in SASP12 if g in adata.var_names]
print(f"SASP12 detected: {len(detected)}/12 ({[g for g in SASP12 if g not in adata.var_names]} missing)")

# Compute scores
sasp_matrix = adata[:, detected].X  # (n_cells, n_genes)
adata.obs["SASP_score"] = np.asarray(sasp_matrix).mean(axis=1).flatten()
adata.obs["JUNB"] = np.asarray(adata[:, "JUNB"].X).flatten()

# Verify
print(f"SASP_score range: [{adata.obs['SASP_score'].min():.3f}, {adata.obs['SASP_score'].max():.3f}]")
print(f"JUNB range: [{adata.obs['JUNB'].min():.3f}, {adata.obs['JUNB'].max():.3f}]")

# Per-subtype cell-level correlation
print("\n--- Cell-level JUNB-SASP correlation per FAP subtype ---")
results = []
for subtype in sorted(adata.obs["Annotation"].unique()):
    mask = adata.obs["Annotation"] == subtype
    n_cells = mask.sum()
    if n_cells < 50:
        print(f"  {subtype}: N={n_cells} < 50, skip")
        continue
    sub = adata[mask]
    junb = sub.obs["JUNB"].values
    sasp = sub.obs["SASP_score"].values
    valid = ~(np.isnan(junb) | np.isnan(sasp))
    if valid.sum() < 50:
        continue
    rho, p = spearmanr(junb[valid], sasp[valid])
    results.append({
        "subtype": subtype,
        "n_cells": int(valid.sum()),
        "rho": round(rho, 4),
        "p_value": p
    })
    print(f"  {subtype}: N={valid.sum():,}, rho={rho:.4f}, p={p:.2e}")

results_df = pd.DataFrame(results)

# Sign-flip test
print("\n--- SIGN-FLIP TEST (VERA hypothesis) ---")
runx2_rows = [r for r in results if "RUNX2" in r["subtype"]]
mme_rows = [r for r in results if "MME" in r["subtype"]]

if runx2_rows and mme_rows:
    r_rho = runx2_rows[0]["rho"]
    m_rho = mme_rows[0]["rho"]
    print(f"  RUNX2+ rho: {r_rho:+.4f}")
    print(f"  MME+ rho:  {m_rho:+.4f}")
    if r_rho > 0 and m_rho < 0:
        print("  VERDICT: SIGN-FLIP IS COMPOSITIONAL (VERA CONFIRMED)")
        print("  → JUNB drives SASP in RUNX2+ (inflammatory), represses in MME+ (regenerative)")
    elif r_rho < 0 and m_rho < 0:
        print("  VERDICT: MECHANISTIC — JUNB is a SASP repressor in ALL FAP subtypes")
        print("  → AP-1 inhibitors for FAPs are CONTRAINDICATED")
    elif r_rho > 0 and m_rho > 0:
        print("  VERDICT: BOTH POSITIVE — donor-level null is from composition mixing")
    else:
        print(f"  VERDICT: MIXED")
else:
    print("  Insufficient subtypes for comparison")

# Per-subtype donor-level analysis
print("\n--- Donor-level JUNB-SASP per FAP subtype ---")
donor_results = []
for subtype in sorted(adata.obs["Annotation"].unique()):
    mask = adata.obs["Annotation"] == subtype
    sub = adata[mask]
    if sub.n_obs < 50:
        continue
    donor_means = sub.obs.groupby("sample").agg({"JUNB": "mean", "SASP_score": "mean"}).dropna()
    if len(donor_means) >= 5:
        rho, p = spearmanr(donor_means["JUNB"], donor_means["SASP_score"])
        donor_results.append({
            "subtype": subtype,
            "n_donors": len(donor_means),
            "rho": round(rho, 4),
            "p_value": p
        })
        print(f"  {subtype}: N={len(donor_means)}, rho={rho:.4f}, p={p:.2e}")

donor_df = pd.DataFrame(donor_results)

# ============================================================
# PART 2: CEBPB mRNA analysis with ALL donors (N=22)
# ============================================================
print("\n" + "=" * 60)
print("PART 2: CEBPB mRNA-SASP with N=22 (all HLMA FAP donors)")
print("=" * 60)

# CEBPB donor-level correlation with all 22 donors
adata.obs["CEBPB"] = np.asarray(adata[:, "CEBPB"].X).flatten()
donor_cebpb = adata.obs.groupby("sample").agg({
    "CEBPB": "mean",
    "SASP_score": "mean",
    "age": "first",
    "Country": "first"
}).dropna()

print(f"Donors: {len(donor_cebpb)}")
rho_all, p_all = spearmanr(donor_cebpb["CEBPB"], donor_cebpb["SASP_score"])
print(f"All donors: rho={rho_all:.4f}, p={p_all:.2e}")

# Within-China
china = donor_cebpb[donor_cebpb["Country"] == "China"]
if len(china) >= 5:
    rho_cn, p_cn = spearmanr(china["CEBPB"], china["SASP_score"])
    print(f"China only: rho={rho_cn:.4f}, p={p_cn:.2e} (N={len(china)})")

# Within-Spain
spain = donor_cebpb[donor_cebpb["Country"] == "Spain"]
if len(spain) >= 5:
    rho_es, p_es = spearmanr(spain["CEBPB"], spain["SASP_score"])
    print(f"Spain only: rho={rho_es:.4f}, p={p_es:.2e} (N={len(spain)})")
else:
    print(f"Spain only: N={len(spain)} — too few for correlation")

# Country-adjusted (partial correlation)
try:
    from pingouin import partial_corr
    pc = partial_corr(data=donor_cebpb.reset_index(), x="CEBPB", y="SASP_score", covar="age")
    print(f"\nAge-adjusted: r={pc['r'].values[0]:.4f}, p={pc['p-val'].values[0]:.2e}")
except:
    print("\nAge-adjusted: pingouin not available")

# ============================================================
# PART 3: Per-subtype CEBPB donor analysis
# ============================================================
print("\n--- CEBPB per FAP subtype (donor-level) ---")
for subtype in sorted(adata.obs["Annotation"].unique()):
    mask = adata.obs["Annotation"] == subtype
    sub = adata[mask]
    if sub.n_obs < 100:
        continue
    donor_sub = sub.obs.groupby("sample").agg({
        "CEBPB": "mean", "SASP_score": "mean", "Country": "first"
    }).dropna()
    if len(donor_sub) >= 5:
        rho, p = spearmanr(donor_sub["CEBPB"], donor_sub["SASP_score"])
        print(f"  {subtype}: N={len(donor_sub)}, rho={rho:.4f}, p={p:.2e}")

# ============================================================
# Save
# ============================================================
outdir = "experiments/batch_057"
os.makedirs(outdir, exist_ok=True)
results_df.to_csv(f"{outdir}/juns_sasp_cell_level.csv", index=False)
if len(donor_df) > 0:
    donor_df.to_csv(f"{outdir}/juns_sasp_donor_level.csv", index=False)
donor_cebpb.to_csv(f"{outdir}/cebpb_donor_level_n22.csv")

print(f"\nResults saved to {outdir}/")
print("DONE")
