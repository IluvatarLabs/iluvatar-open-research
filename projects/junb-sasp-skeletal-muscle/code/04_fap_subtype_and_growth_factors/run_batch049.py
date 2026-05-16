#!/usr/bin/env python3
"""
batch_049: Three PI-directed MUST-DO analyses for SM-RD project.

A1: FAP Subtype JUNB-SASP Donor-Level Spearman (EXPLORATORY)
A2: JUNB+ FAP Surface Marker Characterization (Donor-Level Aggregation)
A3: FAP->MuSC Ligand-Receptor Age-Differential (Within-Tech Only)

Design review mandated:
- A1 is exploratory (N<=12 per subtype, powered only for rho>0.7)
- A2 uses donor-level paired comparisons (true N=donors, not cells)
- A3 restricts to within-technology strata (scRNA-only primary, snRNA validation)
"""

import json
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Trying to modify attribute")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro")
DATA = PROJECT / "data"
OUT = PROJECT / "experiments" / "batch_049"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SASP12 = ["CCL2", "CCL7", "CCL20", "CXCL6", "CXCL8", "IL6",
           "MMP1", "MMP3", "SERPINE1", "IGFBP2", "IGFBP3", "IGFBP5"]

# HLMA FAP subtypes (exclude Tenocyte -- not a FAP)
HLMA_SUBTYPES = ["MME+ FAP", "CD55+ FAP", "GPC3+ FAP", "RUNX2+ FAP", "CD99+ FAP"]

# Nature Aging FB subtypes (only the FB-like ones)
NA_SUBTYPES = ["Inter_FB", "Par_FB", "Adv_FB", "Perineural_FB"]

# Surface marker gene list (curated from brief + literature)
# Aliases resolved to official gene symbols where needed
SURFACE_GENES = sorted(set([
    # Known FAP markers
    "PDGFRA", "PDGFRB", "CD55", "CD63", "THY1", "ENG", "NT5E", "DPP4",
    "VCAM1", "ICAM1",
    # Receptor tyrosine kinases
    "MET", "IGF1R", "FGFR1", "FGFR2", "FGFR3", "FGFR4", "EGFR", "ERBB2",
    # Integrins
    "ITGA1", "ITGA2", "ITGA3", "ITGA4", "ITGA5", "ITGA6", "ITGA7",
    "ITGA8", "ITGA9", "ITGA10", "ITGA11",
    "ITGB1", "ITGB2", "ITGB3", "ITGB4", "ITGB5", "ITGB6", "ITGB7", "ITGB8",
    "ITGAV",
    # CD molecules
    "CD44", "CD47", "CD82", "CD151", "CD9", "CD81", "CD24", "CD46", "CD59", "CD99",
    # Transporters/channels
    "SLC2A1",  # GLUT1
    "ABCB1", "ABCC1", "ABCG2",
    # Surface proteoglycans
    "GPC1", "GPC2", "GPC3", "GPC4", "GPC5", "GPC6",
    "SDC1", "SDC2", "SDC3", "SDC4",
]))

# Ligand-Receptor pairs (curated from F087 + literature)
LR_PAIRS = [
    ("HGF", "MET"), ("FGF7", "FGFR2"), ("FGF7", "FGFR1"),
    ("IGF2", "IGF1R"), ("IGF1", "IGF1R"),
    ("PDGFA", "PDGFRA"), ("PDGFB", "PDGFRB"),
    ("CCL2", "CCR2"), ("CCL7", "CCR2"),
    ("CXCL8", "CXCR1"), ("CXCL8", "CXCR2"),
    ("CXCL6", "CXCR1"), ("CXCL6", "CXCR2"),
    ("IL6", "IL6R"),
    ("TNF", "TNFRSF1A"), ("TNF", "TNFRSF1B"),
    ("TGFB1", "TGFBR1"), ("TGFB2", "TGFBR1"),
    ("WNT5A", "FZD2"), ("WNT5A", "FZD7"),
    ("JAG1", "NOTCH2"), ("DLL1", "NOTCH2"),
    ("BMP4", "BMPR1A"), ("GDF15", "GFRAL"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense_vec(adata, gene):
    """Extract gene expression as dense 1D numpy array. Return zeros if missing."""
    if gene not in adata.var_names:
        return np.zeros(adata.shape[0])
    X = adata[:, gene].X
    if hasattr(X, "toarray"):
        return X.toarray().ravel()
    if hasattr(X, "A"):
        return X.A.ravel()
    return np.asarray(X).ravel()


def fisher_z_ci(rho, n, alpha=0.05):
    """95% CI for Spearman rho via Fisher Z-transform."""
    # Clip rho to avoid inf in atanh
    rho = np.clip(rho, -0.9999, 0.9999)
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return [round(float(lo), 4), round(float(hi), 4)]


def achieved_power(rho_alt, n, alpha=0.05):
    """
    Achieved power for Spearman rho test via Fisher Z approximation.
    H0: rho=0, H1: rho=rho_alt (two-sided).
    """
    if abs(rho_alt) < 1e-9:
        return alpha  # power = alpha under null
    z_alt = np.arctanh(rho_alt)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    # Power = P(|Z| > z_crit | true rho = rho_alt)
    # Under H1, Z ~ N(z_alt, se^2)
    power = 1 - stats.norm.cdf(z_crit - z_alt / se) + stats.norm.cdf(-z_crit - z_alt / se)
    return round(power, 4)


def cohen_d_paired(diff_array):
    """Cohen's d for paired differences: mean(diff) / std(diff)."""
    diff = np.asarray(diff_array, dtype=float)
    if len(diff) < 2:
        return np.nan
    s = np.std(diff, ddof=1)
    if s < 1e-12:
        return 0.0
    return np.mean(diff) / s


def cohen_d_two_group(x, y):
    """Cohen's d for two independent groups (pooled SD)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if pooled_sd < 1e-12:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_sd


# ===========================================================================
# ANALYSIS 1: FAP Subtype JUNB-SASP (EXPLORATORY)
# ===========================================================================

def analysis1():
    """
    For each FAP subtype in HLMA and Nature Aging:
    1. Compute donor-level mean JUNB and donor-level mean SASP12 composite
    2. Spearman rho(JUNB, SASP12) across donors
    3. Report N, rho, p, 95% CI, power
    4. Exclude subtypes with N<5 donors
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: FAP Subtype JUNB-SASP (EXPLORATORY)")
    print("="*70)

    results = {"HLMA": {}, "Nature_Aging": {}}

    # --- HLMA ---
    print("\n--- HLMA FAP Data ---")
    adata = sc.read_h5ad(DATA / "OMIX004308-02.h5ad")
    adata.obs["age_num"] = pd.to_numeric(adata.obs["age"], errors="coerce")

    # Pre-compute expression vectors for JUNB and SASP12 genes
    junb_vec = _to_dense_vec(adata, "JUNB")
    sasp_vecs = {}
    for g in SASP12:
        sasp_vecs[g] = _to_dense_vec(adata, g)

    for subtype in HLMA_SUBTYPES:
        mask = adata.obs["Annotation"] == subtype
        n_cells = mask.sum()
        donors = adata.obs.loc[mask, "sample"].unique()
        n_donors = len(donors)
        print(f"\n  {subtype}: {n_cells} cells, {n_donors} donors")

        if n_donors < 5:
            print(f"    EXCLUDED: N={n_donors} < 5 donors")
            results["HLMA"][subtype] = {"excluded": True, "reason": f"N={n_donors}<5"}
            continue

        # Donor-level JUNB and SASP12
        donor_junb = {}
        donor_sasp = {}
        for d in donors:
            d_mask = mask & (adata.obs["sample"] == d)
            if d_mask.sum() == 0:
                continue
            idx = np.where(d_mask)[0]
            # JUNB mean across cells for this donor
            donor_junb[d] = np.mean(junb_vec[idx])
            # SASP12 composite: mean of detected genes per cell, then mean across cells
            # "Detected" = gene has non-zero expression in at least some cells
            sasp_cell_means = []
            for i in idx:
                gene_vals = [sasp_vecs[g][i] for g in SASP12]
                # Use all SASP12 genes (including zeros) -- this is mean expression
                # rationale: using only detected genes introduces bias at low detection
                sasp_cell_means.append(np.mean(gene_vals))
            donor_sasp[d] = np.mean(sasp_cell_means)

        junb_arr = np.array([donor_junb[d] for d in sorted(donor_junb)])
        sasp_arr = np.array([donor_sasp[d] for d in sorted(donor_junb)])
        n = len(junb_arr)

        if n < 5:
            results["HLMA"][subtype] = {"excluded": True, "reason": f"N={n}<5 after filtering"}
            continue

        rho, pval = stats.spearmanr(junb_arr, sasp_arr)
        ci = fisher_z_ci(rho, n)
        pwr5 = achieved_power(0.5, n)
        pwr7 = achieved_power(0.7, n)

        print(f"    N={n}, rho={rho:.3f}, p={pval:.4f}, CI95={ci}, "
              f"power(rho=0.5)={pwr5:.3f}, power(rho=0.7)={pwr7:.3f}")

        results["HLMA"][subtype] = {
            "N": n, "rho": round(float(rho), 4), "p": round(float(pval), 4),
            "ci95": ci, "power_0.5": pwr5, "power_0.7": pwr7
        }

    # --- Nature Aging ---
    print("\n--- Nature Aging FB Data ---")
    na = sc.read_h5ad(DATA / "SKM_fibroblasts_Schwann_human_2023-06-22.h5ad")

    # Pre-compute expression vectors
    junb_na = _to_dense_vec(na, "JUNB")
    sasp_na = {}
    for g in SASP12:
        sasp_na[g] = _to_dense_vec(na, g)

    for subtype in NA_SUBTYPES:
        mask = na.obs["annotation_level2"] == subtype
        n_cells = mask.sum()
        donors = na.obs.loc[mask, "DonorID"].unique()
        n_donors = len(donors)
        print(f"\n  {subtype}: {n_cells} cells, {n_donors} donors")

        if n_donors < 5:
            print(f"    EXCLUDED: N={n_donors} < 5 donors")
            results["Nature_Aging"][subtype] = {"excluded": True, "reason": f"N={n_donors}<5"}
            continue

        donor_junb = {}
        donor_sasp = {}
        for d in donors:
            d_mask = mask & (na.obs["DonorID"] == d)
            if d_mask.sum() == 0:
                continue
            idx = np.where(d_mask)[0]
            donor_junb[d] = np.mean(junb_na[idx])
            sasp_cell_means = []
            for i in idx:
                gene_vals = [sasp_na[g][i] for g in SASP12]
                sasp_cell_means.append(np.mean(gene_vals))
            donor_sasp[d] = np.mean(sasp_cell_means)

        junb_arr = np.array([donor_junb[d] for d in sorted(donor_junb)])
        sasp_arr = np.array([donor_sasp[d] for d in sorted(donor_junb)])
        n = len(junb_arr)

        rho, pval = stats.spearmanr(junb_arr, sasp_arr)
        ci = fisher_z_ci(rho, n)
        pwr5 = achieved_power(0.5, n)
        pwr7 = achieved_power(0.7, n)

        print(f"    N={n}, rho={rho:.3f}, p={pval:.4f}, CI95={ci}, "
              f"power(rho=0.5)={pwr5:.3f}, power(rho=0.7)={pwr7:.3f}")

        results["Nature_Aging"][subtype] = {
            "N": n, "rho": round(float(rho), 4), "p": round(float(pval), 4),
            "ci95": ci, "power_0.5": pwr5, "power_0.7": pwr7
        }

    # Save CSV
    rows = []
    for dataset in ["HLMA", "Nature_Aging"]:
        for subtype, info in results[dataset].items():
            row = {"dataset": dataset, "subtype": subtype}
            row.update(info)
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "A1_subtype_junb_sasp.csv", index=False)
    print(f"\n  CSV saved: {OUT / 'A1_subtype_junb_sasp.csv'}")

    return results


# ===========================================================================
# ANALYSIS 2: JUNB+ Surface Markers (Donor-Level Aggregation)
# ===========================================================================

def analysis2():
    """
    Donor-level aggregation to avoid pseudoreplication.
    1. Define JUNB+ as JUNB > mean + 1*SD within old donors only
    2. For each old donor: mean expression in JUNB+ vs JUNB- cells
    3. Paired Cohen's d across donors for each gene
    4. BH-correct across all tested genes
    5. Filter to surface markers
    6. Sensitivity at mean+0.5SD and mean+1.5SD
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: JUNB+ Surface Markers (Donor-Level Aggregation)")
    print("="*70)

    adata = sc.read_h5ad(DATA / "OMIX004308-02.h5ad")
    adata.obs["age_num"] = pd.to_numeric(adata.obs["age"], errors="coerce")

    # Restrict to old donors (age >= 60)
    old_mask = adata.obs["age_num"] >= 60
    adata_old = adata[old_mask].copy()
    print(f"Old donors: {adata_old.shape[0]} cells, {adata_old.obs['sample'].nunique()} donors")

    # Get all old donor IDs
    old_donors = sorted(adata_old.obs["sample"].unique())
    print(f"Donors: {old_donors}")

    # Get JUNB expression for old cells
    junb_expr = _to_dense_vec(adata_old, "JUNB")
    junb_mean = np.mean(junb_expr)
    junb_sd = np.std(junb_expr, ddof=1)
    print(f"JUNB in old FAPs: mean={junb_mean:.3f}, SD={junb_sd:.3f}")

    # Get ALL gene names for testing
    all_genes = list(adata_old.var_names)

    def run_surface_analysis(threshold_sd, label):
        """Run surface marker analysis at a given SD threshold."""
        threshold = junb_mean + threshold_sd * junb_sd
        junb_plus_mask = junb_expr > threshold

        n_plus = junb_plus_mask.sum()
        n_minus = (~junb_plus_mask).sum()
        print(f"\n  [{label}] Threshold={threshold:.3f} (mean+{threshold_sd}SD)")
        print(f"  [{label}] JUNB+={n_plus}, JUNB-={n_minus} cells")

        # For each donor, compute mean expression in JUNB+ and JUNB- cells
        # Only include donors that have at least 5 cells in BOTH groups
        gene_results = []

        # Build donor masks
        valid_donors = []
        for d in old_donors:
            d_mask = adata_old.obs["sample"] == d
            d_plus = d_mask & junb_plus_mask
            d_minus = d_mask & (~junb_plus_mask)
            if d_plus.sum() >= 3 and d_minus.sum() >= 3:
                valid_donors.append(d)

        print(f"  [{label}] Valid donors (>=3 cells in both groups): {len(valid_donors)}")

        if len(valid_donors) < 5:
            print(f"  [{label}] ERROR: Too few valid donors ({len(valid_donors)}<5)")
            return [], []

        # For efficiency, extract expression matrix once
        # Build donor indices for JUNB+ and JUNB-
        donor_plus_idx = {}
        donor_minus_idx = {}
        for d in valid_donors:
            d_mask = adata_old.obs["sample"] == d
            plus_mask = d_mask.values & junb_plus_mask
            minus_mask = d_mask.values & (~junb_plus_mask)
            donor_plus_idx[d] = np.where(plus_mask)[0]
            donor_minus_idx[d] = np.where(minus_mask)[0]

        # Batch extract expression matrix (dense for speed)
        X_full = adata_old.X.toarray() if hasattr(adata_old.X, "toarray") else np.asarray(adata_old.X)

        # Test ALL genes
        n_genes = len(all_genes)
        print(f"  [{label}] Testing {n_genes} genes across {len(valid_donors)} donors...")

        gene_stats = []
        for gi, gene in enumerate(all_genes):
            if gi % 5000 == 0 and gi > 0:
                print(f"    ... {gi}/{n_genes} genes processed")

            expr_vec = X_full[:, gi]

            # Per-donor means
            plus_means = []
            minus_means = []
            for d in valid_donors:
                pidx = donor_plus_idx[d]
                midx = donor_minus_idx[d]
                plus_means.append(np.mean(expr_vec[pidx]))
                minus_means.append(np.mean(expr_vec[midx]))

            plus_means = np.array(plus_means)
            minus_means = np.array(minus_means)

            # Paired differences
            diffs = plus_means - minus_means
            d_val = cohen_d_paired(diffs)

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(plus_means, minus_means)

            # Log2 fold change: data is already log-transformed (confirmed negative values),
            # so FC = mean(JUNB+) - mean(JUNB-) directly gives log2FC
            mean_plus = np.mean(plus_means)
            mean_minus = np.mean(minus_means)
            log2fc = mean_plus - mean_minus

            is_surface = gene in SURFACE_GENES

            gene_stats.append({
                "gene": gene,
                "d": round(float(d_val), 4) if not np.isnan(d_val) else None,
                "log2fc": round(float(log2fc), 4) if not np.isnan(log2fc) else None,
                "p_raw": round(float(p_val), 6),
                "is_surface": is_surface,
                "mean_junb_plus": round(float(mean_plus), 4),
                "mean_junb_minus": round(float(mean_minus), 4),
            })

        # BH correction across all genes
        p_vals = [g["p_raw"] for g in gene_stats]
        if len(p_vals) > 0:
            reject, p_bh, _, _ = multipletests(p_vals, method="fdr_bh")
            for i, g in enumerate(gene_stats):
                g["p_bh"] = round(float(p_bh[i]), 6)

        # Filter to surface markers with results
        surface_hits = [g for g in gene_stats if g["is_surface"]]
        surface_hits.sort(key=lambda x: abs(x["d"]) if x["d"] is not None else 0, reverse=True)

        n_sig = sum(1 for g in surface_hits if g["p_bh"] is not None and g["p_bh"] < 0.05 and abs(g.get("d", 0) or 0) > 0.8)
        print(f"  [{label}] Surface genes tested: {len(surface_hits)}")
        print(f"  [{label}] Surface genes with |d|>0.8 and BH p<0.05: {n_sig}")

        # Print top 10 surface markers
        for g in surface_hits[:10]:
            log2fc_str = f"{g['log2fc']:.3f}" if g['log2fc'] is not None else "N/A"
            print(f"    {g['gene']}: d={g['d']:.3f}, log2FC={log2fc_str}, "
                  f"p_bh={g['p_bh']:.4f}")

        return gene_stats, surface_hits

    # Primary threshold: mean + 1 SD
    primary_all, primary_surface = run_surface_analysis(1.0, "primary_1.0SD")

    # Sensitivity: mean + 0.5 SD
    try:
        result_05 = run_surface_analysis(0.5, "sensitivity_0.5SD")
        if isinstance(result_05, tuple):
            sens_05_all, sens_05_surface = result_05
        else:
            sens_05_all, sens_05_surface = [], []
    except Exception as e:
        print(f"  Sensitivity 0.5SD failed: {e}")
        sens_05_all, sens_05_surface = [], []

    # Sensitivity: mean + 1.5 SD
    try:
        result_15 = run_surface_analysis(1.5, "sensitivity_1.5SD")
        if isinstance(result_15, tuple):
            sens_15_all, sens_15_surface = result_15
        else:
            sens_15_all, sens_15_surface = [], []
    except Exception as e:
        print(f"  Sensitivity 1.5SD failed: {e}")
        sens_15_all, sens_15_surface = [], []

    # Concordance analysis: genes significant at primary and both sensitivities
    if primary_surface and sens_05_surface and sens_15_surface:
        primary_sig = {g["gene"] for g in primary_surface
                       if g["p_bh"] is not None and g["p_bh"] < 0.05 and abs(g.get("d", 0) or 0) > 0.5}
        sens05_sig = {g["gene"] for g in sens_05_surface
                      if g["p_bh"] is not None and g["p_bh"] < 0.05 and abs(g.get("d", 0) or 0) > 0.5}
        sens15_sig = {g["gene"] for g in sens_15_surface
                      if g["p_bh"] is not None and g["p_bh"] < 0.05 and abs(g.get("d", 0) or 0) > 0.5}

        all_three = primary_sig & sens05_sig & sens15_sig
        print(f"\n  Concordance: {len(all_three)} surface markers significant at all 3 thresholds")
        print(f"    Primary significant: {len(primary_sig)}")
        print(f"    0.5SD significant: {len(sens05_sig)}")
        print(f"    1.5SD significant: {len(sens15_sig)}")
        for g in sorted(all_three):
            print(f"    CONCORDANT: {g}")

    # Save results
    results = {
        "primary_threshold": list(primary_surface) if primary_surface else [],
        "sensitivity_0.5sd": list(sens_05_surface) if sens_05_surface else [],
        "sensitivity_1.5sd": list(sens_15_surface) if sens_15_surface else [],
    }

    # Save CSVs
    pd.DataFrame(primary_all).to_csv(OUT / "A2_all_genes_primary.csv", index=False)
    pd.DataFrame(primary_surface).to_csv(OUT / "A2_surface_markers_primary.csv", index=False)
    print(f"\n  CSV saved: A2_all_genes_primary.csv, A2_surface_markers_primary.csv")

    return results


# ===========================================================================
# ANALYSIS 3: FAP->MuSC LR Age-Differential (Within-Tech Only)
# ===========================================================================

def analysis3():
    """
    Within-technology donor-level pseudobulk analysis.
    scRNA-only primary, snRNA validation.
    Donor-level pseudobulk: for each donor, mean ligand in FAPs, mean receptor in MuSCs.
    Compare young (<40) vs old (>=60).
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: FAP->MuSC LR Age-Differential (Within-Tech)")
    print("="*70)

    # Load data
    fap = sc.read_h5ad(DATA / "OMIX004308-02.h5ad")
    musc = sc.read_h5ad(DATA / "MuSC_scsn_RNA.h5ad")

    fap.obs["age_num"] = pd.to_numeric(fap.obs["age"], errors="coerce")
    # MuSC age is already numeric

    # Get all LR genes
    lr_genes = set()
    for lig, rec in LR_PAIRS:
        lr_genes.add(lig)
        lr_genes.add(rec)
    lr_genes = sorted(lr_genes)

    print(f"LR pairs: {len(LR_PAIRS)}")
    print(f"Unique LR genes: {len(lr_genes)}")

    # Check which LR genes are in each dataset
    fap_genes = set(fap.var_names)
    musc_genes = set(musc.var_names)
    missing_fap = [g for g in lr_genes if g not in fap_genes]
    missing_musc = [g for g in lr_genes if g not in musc_genes]
    print(f"LR genes missing from FAP: {missing_fap}")
    print(f"LR genes missing from MuSC: {missing_musc}")

    def run_lr_analysis(tech_str):
        """Run LR analysis for a specific technology stratum."""
        print(f"\n--- {tech_str} Analysis ---")

        # Filter to specific tech
        fap_tech = fap[fap.obs["tech"] == tech_str].copy()
        musc_tech = musc[musc.obs["tech"] == tech_str].copy()

        print(f"  FAP {tech_str}: {fap_tech.shape[0]} cells, {fap_tech.obs['sample'].nunique()} donors")
        print(f"  MuSC {tech_str}: {musc_tech.shape[0]} cells, {musc_tech.obs['sample'].nunique()} donors")

        # Shared donors
        fap_donors = set(fap_tech.obs["sample"].unique())
        musc_donors = set(musc_tech.obs["sample"].unique())
        shared = sorted(fap_donors & musc_donors)
        print(f"  Shared donors: {len(shared)} -> {shared}")

        # Age groups: young <40, old >=60
        donor_info = {}
        for d in shared:
            age = fap_tech.obs.loc[fap_tech.obs["sample"] == d, "age_num"].iloc[0]
            group = "young" if age < 40 else ("old" if age >= 60 else "mid")
            donor_info[d] = {"age": float(age), "group": group}

        young_donors = [d for d in shared if donor_info[d]["group"] == "young"]
        old_donors = [d for d in shared if donor_info[d]["group"] == "old"]
        mid_donors = [d for d in shared if donor_info[d]["group"] == "mid"]

        print(f"  Young donors (<40): {len(young_donors)} -> {young_donors}")
        print(f"  Old donors (>=60): {len(old_donors)} -> {old_donors}")
        if mid_donors:
            print(f"  Mid-age donors (excluded): {mid_donors}")

        # Verify cell counts
        n_young_fap = sum((fap_tech.obs["sample"] == d).sum() for d in young_donors)
        n_old_fap = sum((fap_tech.obs["sample"] == d).sum() for d in old_donors)
        n_young_musc = sum((musc_tech.obs["sample"] == d).sum() for d in young_donors)
        n_old_musc = sum((musc_tech.obs["sample"] == d).sum() for d in old_donors)
        print(f"  FAP young cells: {n_young_fap}, old cells: {n_old_fap}")
        print(f"  MuSC young cells: {n_young_musc}, old cells: {n_old_musc}")

        # Check interpretability thresholds
        uninterpretable = False
        reasons = []
        if n_young_fap < 500:
            uninterpretable = True
            reasons.append(f"FAP young cells={n_young_fap}<500")
        if n_old_fap < 500:
            uninterpretable = True
            reasons.append(f"FAP old cells={n_old_fap}<500")
        if n_young_musc < 500:
            uninterpretable = True
            reasons.append(f"MuSC young cells={n_young_musc}<500")
        if n_old_musc < 500:
            uninterpretable = True
            reasons.append(f"MuSC old cells={n_old_musc}<500")

        if uninterpretable:
            print(f"  UNINTERPRETABLE: {reasons}")
            print(f"  Proceeding with caution but flagging results")

        # Pre-extract expression matrices for efficiency
        fap_X = fap_tech.X.toarray() if hasattr(fap_tech.X, "toarray") else np.asarray(fap_tech.X)
        musc_X = musc_tech.X.toarray() if hasattr(musc_tech.X, "toarray") else np.asarray(musc_tech.X)

        # Build gene-to-column mapping
        fap_gene_idx = {g: i for i, g in enumerate(fap_tech.var_names) if g in lr_genes}
        musc_gene_idx = {g: i for i, g in enumerate(musc_tech.var_names) if g in lr_genes}

        # Donor-level pseudobulk
        # For each donor: mean expression of each LR gene in FAPs and MuSCs
        test_donors = young_donors + old_donors

        fap_pseudobulk = {}  # donor -> {gene: mean_expr}
        musc_pseudobulk = {}

        for d in test_donors:
            # FAP pseudobulk
            fap_d_mask = fap_tech.obs["sample"] == d
            fap_d_idx = np.where(fap_d_mask)[0]
            if len(fap_d_idx) == 0:
                continue
            fap_d_expr = {}
            for gene, col_idx in fap_gene_idx.items():
                fap_d_expr[gene] = np.mean(fap_X[fap_d_idx, col_idx])
            fap_pseudobulk[d] = fap_d_expr

            # MuSC pseudobulk
            musc_d_mask = musc_tech.obs["sample"] == d
            musc_d_idx = np.where(musc_d_mask)[0]
            if len(musc_d_idx) == 0:
                continue
            musc_d_expr = {}
            for gene, col_idx in musc_gene_idx.items():
                musc_d_expr[gene] = np.mean(musc_X[musc_d_idx, col_idx])
            musc_pseudobulk[d] = musc_d_expr

        # Now compute age-differential for each LR pair
        pair_results = []
        for lig, rec in LR_PAIRS:
            lig_str = f"{lig}->{rec}"

            # Ligand in FAPs: young vs old
            lig_young = [fap_pseudobulk[d].get(lig, np.nan) for d in young_donors if d in fap_pseudobulk]
            lig_old = [fap_pseudobulk[d].get(lig, np.nan) for d in old_donors if d in fap_pseudobulk]

            # Receptor in MuSCs: young vs old
            rec_young = [musc_pseudobulk[d].get(rec, np.nan) for d in young_donors if d in musc_pseudobulk]
            rec_old = [musc_pseudobulk[d].get(rec, np.nan) for d in old_donors if d in musc_pseudobulk]

            # Remove NaN
            lig_young = [x for x in lig_young if not np.isnan(x)]
            lig_old = [x for x in lig_old if not np.isnan(x)]
            rec_young = [x for x in rec_young if not np.isnan(x)]
            rec_old = [x for x in rec_old if not np.isnan(x)]

            # Cohen's d for ligand age effect (old - young; positive = higher in old)
            if len(lig_young) >= 2 and len(lig_old) >= 2:
                d_lig = cohen_d_two_group(lig_old, lig_young)
                # Mann-Whitney U test (small N, non-parametric preferred)
                try:
                    u_lig, p_lig = stats.mannwhitneyu(lig_old, lig_young, alternative="two-sided")
                except ValueError:
                    p_lig = 1.0
            else:
                d_lig = np.nan
                p_lig = np.nan

            # Cohen's d for receptor age effect
            if len(rec_young) >= 2 and len(rec_old) >= 2:
                d_rec = cohen_d_two_group(rec_old, rec_young)
                try:
                    u_rec, p_rec = stats.mannwhitneyu(rec_old, rec_young, alternative="two-sided")
                except ValueError:
                    p_rec = 1.0
            else:
                d_rec = np.nan
                p_rec = np.nan

            pair_results.append({
                "pair": lig_str,
                "d_ligand": round(float(d_lig), 4) if not np.isnan(d_lig) else None,
                "d_receptor": round(float(d_rec), 4) if not np.isnan(d_rec) else None,
                "p_ligand": round(float(p_lig), 6) if not np.isnan(p_lig) else None,
                "p_receptor": round(float(p_rec), 6) if not np.isnan(p_rec) else None,
                "n_young": len(lig_young),
                "n_old": len(lig_old),
                "mean_ligand_young": round(float(np.mean(lig_young)), 4) if lig_young else None,
                "mean_ligand_old": round(float(np.mean(lig_old)), 4) if lig_old else None,
                "mean_receptor_young": round(float(np.mean(rec_young)), 4) if rec_young else None,
                "mean_receptor_old": round(float(np.mean(rec_old)), 4) if rec_old else None,
            })

        # BH correction across all LR pairs (correct ligand p-values and receptor p-values separately)
        lig_pvals = [r["p_ligand"] for r in pair_results if r["p_ligand"] is not None]
        rec_pvals = [r["p_receptor"] for r in pair_results if r["p_receptor"] is not None]

        if lig_pvals:
            _, lig_bh, _, _ = multipletests(lig_pvals, method="fdr_bh")
            lig_idx = 0
            for r in pair_results:
                if r["p_ligand"] is not None:
                    r["p_ligand_bh"] = round(float(lig_bh[lig_idx]), 6)
                    lig_idx += 1
                else:
                    r["p_ligand_bh"] = None

        if rec_pvals:
            _, rec_bh, _, _ = multipletests(rec_pvals, method="fdr_bh")
            rec_idx = 0
            for r in pair_results:
                if r["p_receptor"] is not None:
                    r["p_receptor_bh"] = round(float(rec_bh[rec_idx]), 6)
                    rec_idx += 1
                else:
                    r["p_receptor_bh"] = None

        # Report pairs with |d|>0.5 in BOTH ligand and receptor
        sig_pairs = [r for r in pair_results
                     if r["d_ligand"] is not None and r["d_receptor"] is not None
                     and abs(r["d_ligand"]) > 0.5 and abs(r["d_receptor"]) > 0.5]

        print(f"\n  LR pairs with |d|>0.5 in BOTH ligand and receptor: {len(sig_pairs)}")
        for r in pair_results:
            flag = ""
            if r["d_ligand"] is not None and r["d_receptor"] is not None:
                if abs(r["d_ligand"]) > 0.5 and abs(r["d_receptor"]) > 0.5:
                    flag = " *** BOTH >0.5 ***"
                elif abs(r["d_ligand"]) > 0.5 or abs(r["d_receptor"]) > 0.5:
                    flag = " * one >0.5 *"
            print(f"    {r['pair']}: d_lig={r['d_ligand']}, d_rec={r['d_receptor']}, "
                  f"p_lig={r.get('p_ligand_bh', r['p_ligand'])}, "
                  f"p_rec={r.get('p_receptor_bh', r['p_receptor'])}{flag}")

        return pair_results

    # Run scRNA analysis (primary)
    scrna_results = run_lr_analysis("scRNA")

    # Run snRNA analysis (validation)
    snrna_results = run_lr_analysis("snRNA")

    # Cross-validate: pairs significant in BOTH strata
    scrna_sig = {r["pair"] for r in scrna_results
                 if r["d_ligand"] is not None and r["d_receptor"] is not None
                 and abs(r["d_ligand"]) > 0.5 and abs(r["d_receptor"]) > 0.5}
    snrna_sig = {r["pair"] for r in snrna_results
                 if r["d_ligand"] is not None and r["d_receptor"] is not None
                 and abs(r["d_ligand"]) > 0.5 and abs(r["d_receptor"]) > 0.5}

    concordant = scrna_sig & snrna_sig
    print(f"\n  Cross-tech concordant pairs (|d|>0.5 in both): {len(concordant)}")
    if concordant:
        for p in sorted(concordant):
            print(f"    CONCORDANT: {p}")

    # Save CSVs
    pd.DataFrame(scrna_results).to_csv(OUT / "A3_lr_scrna.csv", index=False)
    pd.DataFrame(snrna_results).to_csv(OUT / "A3_lr_snrna.csv", index=False)
    print(f"\n  CSVs saved: A3_lr_scrna.csv, A3_lr_snrna.csv")

    return {"scrna_only": scrna_results, "snrna_validation": snrna_results}


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("batch_049: Starting three analyses")
    print(f"Output directory: {OUT}")

    results = {}

    # Analysis 1
    results["A1_subtype"] = analysis1()

    # Analysis 2
    results["A2_surface"] = analysis2()

    # Analysis 3
    results["A3_lr"] = analysis3()

    # Save JSON
    json_path = OUT / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")
    print("batch_049: Complete")
