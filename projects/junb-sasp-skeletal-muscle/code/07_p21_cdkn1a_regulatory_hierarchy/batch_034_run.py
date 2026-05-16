#!/usr/bin/env python3
"""
batch_034: Cross-Compartment p21→SASP Analysis

HYPOTHESIS: Is the p21→SASP axis vascular-specific, or does it generalize to FAPs and MuSCs?

PRIMARY METRIC:
  - rho(CDKN1A, SASP12) at donor level for FAPs and MuSCs
  - Compare to F092 vascular reference: rho=0.9126

SECONDARY METRICS:
  - rho(CDKN1A, JUNB) per compartment
  - Partial correlation r(p21→SASP | JUNB) in FAPs

DECISION RULES (Pre-registered):
  - IF rho(p21, SASP) >= 0.50 in FAPs OR MuSCs → p21→SASP is NOT vascular-specific
  - IF rho(p21, SASP) < 0.50 in BOTH compartments → p21→SASP is vascular-specific

UNINTERPRETABLE GATE:
  - IF CDKN1A detection < 20% in either compartment → STOP, UNINTERPRETABLE

DATA:
  - FAP: OMIX004308-02.h5ad (40,389 cells, 22 donors)
  - MuSC: MuSC_scsn_RNA.h5ad (9,559 cells, 23 donors)

BASELINES:
  - F092 (vascular): rho(p21, SASP) = 0.9126
  - F080 (FAP JUNB→SASP): rho = 0.023 (NULL at donor level)
  - F074 (MuSC JUNB→SASP): rho = 0.072 (weak)

Date: 2026-04-11
"""

import json
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.stats import norm as normal_dist
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# All hyperparameters cited from design.yaml
# ============================================================================

# SASP12 genes — same 12-gene set as F092 (batch_033)
# Why: Enables direct comparison across compartments
SASP12 = [
    "CCL2", "CCL7", "CCL8", "CXCL6", "CXCL8",  # chemokines
    "IL6", "IL1B",  # interleukins
    "MMP1", "MMP3",  # matrix metalloproteinases
    "SERPINE1", "PLAU"  # urokinase/plasminogen system
]

# Gene symbols — same as batch_033
CDKN1A_GENE = "CDKN1A"  # p21
JUNB_GENE = "JUNB"  # AP-1 component

# Age thresholds — standard human aging definition, consistent with batch_021/023/033
YOUNG_THRESHOLD = 40  # Age < 40 = young
OLD_THRESHOLD = 60    # Age >= 60 = old

# Cell filters — from design.yaml
FAP_MIN_CELLS = 50   # Minimum FAPs per donor for stable pseudobulk
MuSC_MIN_CELLS = 20  # Minimum MuSCs per donor (smaller population)

# Quality gates — from design.yaml
DETECTION_THRESHOLD = 0.20  # Stop if CDKN1A detection < 20%

# Statistical thresholds — from design.yaml
PRIMARY_RHO_THRESHOLD = 0.50   # Decision threshold for p21→SASP
SECONDARY_RHO_THRESHOLD = 0.30  # Lower threshold for p21→JUNB
PARTIAL_R_THRESHOLD = 0.20     # For partial correlation
ALPHA_PRIMARY = 0.0125         # Bonferroni-corrected for 4 tests

# Data paths
FAP_DATA = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad"
MuSC_DATA = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/MuSC_scsn_RNA.h5ad"
OUTPUT_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_034"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fisher_z(r):
    """
    Fisher Z transformation for correlation comparison.
    Why: Stabilizes variance for small samples; enables hypothesis testing.
    Source: Fisher 1921, as used in batch_033.
    """
    # Clip to avoid log(0) when r = -1 or r = 1
    r_clipped = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))


def partial_spearman(x, y, z, n):
    """
    Compute partial Spearman correlation: r(xy | z).

    Formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    Why partial correlation: We want to know if p21 provides INDEPENDENT information
    about SASP beyond what JUNB captures. If p21 and JUNB are collinear (as in vascular),
    the raw correlation may be driven by the shared JUNB component.

    SE computation: Using Fisher Z transformation with SE = 1/sqrt(n-3)
    Source: design.yaml brief, consistent with batch_033 methodology.
    """
    r_xy, _ = stats.spearmanr(x, y)
    r_xz, _ = stats.spearmanr(x, z)
    r_yz, _ = stats.spearmanr(y, z)

    numerator = r_xy - (r_xz * r_yz)
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    # Handle edge cases where denominator is near zero
    if denominator < 1e-10:
        return np.nan, np.nan

    partial_r = numerator / denominator

    # Approximate p-value using Fisher Z transformation
    # df = n - 3 for simple correlation; for partial with 1 control: n - 4
    z_partial = fisher_z(partial_r)
    se_partial = 1 / np.sqrt(n - 3 - 1)  # n - k - 1, k=1 control variable
    p_partial = 2 * (1 - normal_dist.cdf(abs(z_partial)))

    return partial_r, p_partial


def compute_donor_means(adata, gene_list, donor_col, min_cells=50):
    """
    Compute donor-level pseudobulk means for specified genes.

    Why pseudobulk: Single-cell RNA-seq has high dropout noise. Donor-level means
    reduce technical noise and provide biological replicates (N = donors, not cells).

    Why minimum cells: Ensures pseudobulk means are representative, not dominated by
    a few cells. Source: design.yaml cell_filters.
    """
    # Filter cells by minimum count
    cell_counts = adata.obs[donor_col].value_counts()
    valid_donors = cell_counts[cell_counts >= min_cells].index.tolist()

    if len(valid_donors) == 0:
        return None

    # Filter to valid donors
    mask = adata.obs[donor_col].isin(valid_donors)
    adata_filtered = adata[mask].copy()

    # Get gene expression matrix
    X = adata_filtered.to_df()
    X["donor"] = adata_filtered.obs[donor_col].values

    # Filter to genes that exist in the data
    available_genes = [g for g in gene_list if g in X.columns]

    if len(available_genes) == 0:
        return None

    # Compute mean per donor
    donor_means = X[available_genes].groupby(X["donor"]).mean()

    # Add cell count per donor
    donor_means["n_cells"] = X.groupby("donor").size()

    return donor_means


# ============================================================================
# SMOKE TEST (Preflight checks)
# ============================================================================
print("=" * 70)
print("batch_034: Cross-Compartment p21→SASP Analysis")
print("=" * 70)

print("\n[SMOKE TEST] Pre-flight checks...")

# Check FAP data
try:
    ad_fap = sc.read_h5ad(FAP_DATA)
    print(f"  FAP data loaded: {ad_fap.n_obs:,} cells x {ad_fap.n_vars:,} genes")
except Exception as e:
    print(f"  ERROR: Could not load FAP data: {e}")
    raise

# Check MuSC data
try:
    ad_musc = sc.read_h5ad(MuSC_DATA)
    print(f"  MuSC data loaded: {ad_musc.n_obs:,} cells x {ad_musc.n_vars:,} genes")
except Exception as e:
    print(f"  ERROR: Could not load MuSC data: {e}")
    raise

# Verify required genes exist
required_genes = SASP12 + [CDKN1A_GENE, JUNB_GENE]
fap_genes = set(ad_fap.var_names)
musc_genes = set(ad_musc.var_names)

fap_missing = [g for g in required_genes if g not in fap_genes]
musc_missing = [g for g in required_genes if g not in musc_genes]

if fap_missing:
    print(f"  WARNING: FAP missing genes: {fap_missing}")
if musc_missing:
    print(f"  WARNING: MuSC missing genes: {musc_missing}")

print("  Smoke test: PASSED")

# ============================================================================
# LOAD AND FILTER DATA
# ============================================================================
print("\n[1] Loading and filtering data...")

# --- FAP: Filter to PDGFRA/COL13A1+ cells ---
# Why: FAPs identified by these markers in the HLMA atlas
print("\n  [FAP] Filtering to PDGFRA/COL13A1+ cells...")

fap_marker_genes = ["PDGFRA", "COL13A1"]
fap_has_marker = pd.Series(False, index=ad_fap.obs_names)
for gene in fap_marker_genes:
    if gene in ad_fap.var_names:
        fap_has_marker = fap_has_marker | (ad_fap.to_df()[gene] > 0)

ad_fap_fap = ad_fap[fap_has_marker].copy()
print(f"    FAP cells (PDGFRA/COL13A1+): {ad_fap_fap.n_obs:,}")

# Check CDKN1A detection in FAPs
fap_cdkn1a_detection = (ad_fap_fap.to_df()[CDKN1A_GENE] > 0).mean()
print(f"    CDKN1A detection rate: {fap_cdkn1a_detection * 100:.1f}%")

# Check JUNB detection in FAPs
fap_junb_detection = (ad_fap_fap.to_df()[JUNB_GENE] > 0).mean()
print(f"    JUNB detection rate: {fap_junb_detection * 100:.1f}%")

# --- MuSC: Filter to PAX7+ cells ---
# Why: PAX7 is the canonical MuSC marker
print("\n  [MuSC] Filtering to PAX7+ cells...")

musc_marker_genes = ["PAX7"]
musc_has_marker = pd.Series(False, index=ad_musc.obs_names)
for gene in musc_marker_genes:
    if gene in ad_musc.var_names:
        musc_has_marker = musc_has_marker | (ad_musc.to_df()[gene] > 0)

ad_musc_musc = ad_musc[musc_has_marker].copy()
print(f"    MuSC cells (PAX7+): {ad_musc_musc.n_obs:,}")

# Check CDKN1A detection in MuSCs
musc_cdkn1a_detection = (ad_musc_musc.to_df()[CDKN1A_GENE] > 0).mean()
print(f"    CDKN1A detection rate: {musc_cdkn1a_detection * 100:.1f}%")

# Check JUNB detection in MuSCs
musc_junb_detection = (ad_musc_musc.to_df()[JUNB_GENE] > 0).mean()
print(f"    JUNB detection rate: {musc_junb_detection * 100:.1f}%")

# ============================================================================
# QUALITY GATE 0: Detection Rate Check
# ============================================================================
print("\n[2] Quality Gate 0: CDKN1A Detection Rate Check...")

uninterpretable_fap = False
uninterpretable_musc = False
decision_rule_triggered = None

# FAP check
if fap_cdkn1a_detection < DETECTION_THRESHOLD:
    print(f"  UNINTERPRETABLE: FAP CDKN1A detection {fap_cdkn1a_detection * 100:.1f}% < {DETECTION_THRESHOLD * 100}%")
    uninterpretable_fap = True
    decision_rule_triggered = "FAP CDKN1A detection < 20% - UNINTERPRETABLE"
else:
    print(f"  FAP CDKN1A detection OK: {fap_cdkn1a_detection * 100:.1f}% >= {DETECTION_THRESHOLD * 100}%")

# MuSC check
if musc_cdkn1a_detection < DETECTION_THRESHOLD:
    print(f"  UNINTERPRETABLE: MuSC CDKN1A detection {musc_cdkn1a_detection * 100:.1f}% < {DETECTION_THRESHOLD * 100}%")
    uninterpretable_musc = True
    if decision_rule_triggered:
        decision_rule_triggered += "; MuSC CDKN1A detection < 20% - UNINTERPRETABLE"
    else:
        decision_rule_triggered = "MuSC CDKN1A detection < 20% - UNINTERPRETABLE"
else:
    print(f"  MuSC CDKN1A detection OK: {musc_cdkn1a_detection * 100:.1f}% >= {DETECTION_THRESHOLD * 100}%")

# If both compartments are uninterpretable, we must stop
if uninterpretable_fap and uninterpretable_musc:
    print("\n  FATAL: Both compartments UNINTERPRETABLE. Stopping analysis.")
    results = {
        "status": "UNINTERPRETABLE",
        "reason": "CDKN1A detection < 20% in both FAP and MuSC compartments",
        "fap_detection": float(fap_cdkn1a_detection),
        "musc_detection": float(musc_cdkn1a_detection),
        "interpretation": {
            "fap_p21_sasp_interpretation": "UNINTERPRETABLE - insufficient CDKN1A detection",
            "musc_p21_sasp_interpretation": "UNINTERPRETABLE - insufficient CDKN1A detection",
            "overall_interpretation": "Cannot determine if p21->SASP axis is vascular-specific",
            "decision_rule_triggered": decision_rule_triggered
        }
    }
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    raise SystemExit("Analysis stopped: Both compartments UNINTERPRETABLE")

# ============================================================================
# COMPUTE DONOR PSEUDOBULK MEANS
# ============================================================================
print("\n[3] Computing donor pseudobulk means...")

# --- FAP donor means ---
# Identify donor column (check common names)
fap_donor_col = None
for col in ["sample", "donor", "Sample", "Donor"]:
    if col in ad_fap_fap.obs.columns:
        fap_donor_col = col
        break

if fap_donor_col is None:
    print(f"    Available obs columns: {list(ad_fap_fap.obs.columns)[:20]}")
    raise ValueError("Could not identify FAP donor column")

print(f"    FAP donor column: {fap_donor_col}")
print(f"    FAP donors: {ad_fap_fap.obs[fap_donor_col].nunique()}")

# Compute FAP donor means
fap_analysis_genes = SASP12 + [CDKN1A_GENE, JUNB_GENE]
fap_donor_means = compute_donor_means(ad_fap_fap, fap_analysis_genes, fap_donor_col, min_cells=FAP_MIN_CELLS)

if fap_donor_means is None:
    raise ValueError("No FAP donors with sufficient cells")

print(f"    FAP donors with >={FAP_MIN_CELLS} cells: {len(fap_donor_means)}")

# --- MuSC donor means ---
musc_donor_col = None
for col in ["sample", "donor", "Sample", "Donor"]:
    if col in ad_musc_musc.obs.columns:
        musc_donor_col = col
        break

if musc_donor_col is None:
    print(f"    Available obs columns: {list(ad_musc_musc.obs.columns)[:20]}")
    raise ValueError("Could not identify MuSC donor column")

print(f"    MuSC donor column: {musc_donor_col}")
print(f"    MuSC donors: {ad_musc_musc.obs[musc_donor_col].nunique()}")

# Compute MuSC donor means
musc_analysis_genes = SASP12 + [CDKN1A_GENE, JUNB_GENE]
musc_donor_means = compute_donor_means(ad_musc_musc, musc_analysis_genes, musc_donor_col, min_cells=MuSC_MIN_CELLS)

if musc_donor_means is None:
    raise ValueError("No MuSC donors with sufficient cells")

print(f"    MuSC donors with >={MuSC_MIN_CELLS} cells: {len(musc_donor_means)}")

# ============================================================================
# ADD AGE INFORMATION
# ============================================================================
print("\n[4] Adding age information...")

# --- FAP age ---
fap_age_col = None
for col in ["age", "age_group", "age_pop", "Age"]:
    if col in ad_fap_fap.obs.columns:
        fap_age_col = col
        break

if fap_age_col:
    fap_age_map = ad_fap_fap.obs.groupby(fap_donor_col)[fap_age_col].first()
    # Convert to numeric, handling potential string values
    fap_age_map = pd.to_numeric(fap_age_map, errors='coerce')
    fap_donor_means["age"] = fap_donor_means.index.map(fap_age_map)
    print(f"    FAP age column: {fap_age_col}")
    print(f"    FAP age distribution: {fap_donor_means['age'].value_counts().to_dict()}")
else:
    print(f"    WARNING: No FAP age column found")
    fap_donor_means["age"] = np.nan

# --- MuSC age ---
musc_age_col = None
for col in ["age", "age_group", "age_pop", "Age"]:
    if col in ad_musc_musc.obs.columns:
        musc_age_col = col
        break

if musc_age_col:
    musc_age_map = ad_musc_musc.obs.groupby(musc_donor_col)[musc_age_col].first()
    # Convert to numeric, handling potential string values
    musc_age_map = pd.to_numeric(musc_age_map, errors='coerce')
    musc_donor_means["age"] = musc_donor_means.index.map(musc_age_map)
    print(f"    MuSC age column: {musc_age_col}")
    print(f"    MuSC age distribution: {musc_donor_means['age'].value_counts().to_dict()}")
else:
    print(f"    WARNING: No MuSC age column found")
    musc_donor_means["age"] = np.nan

# Classify age groups
def classify_age(age_val):
    if pd.isna(age_val):
        return "unknown"
    if age_val < YOUNG_THRESHOLD:
        return "young"
    elif age_val >= OLD_THRESHOLD:
        return "old"
    else:
        return "middle"

fap_donor_means["age_group"] = fap_donor_means["age"].apply(classify_age)
musc_donor_means["age_group"] = musc_donor_means["age"].apply(classify_age)

# ============================================================================
# COMPUTE SASP12 COMPOSITE SCORE
# ============================================================================
print("\n[5] Computing SASP12 composite scores...")

# FAP SASP12
fap_sasp_cols = [g for g in SASP12 if g in fap_donor_means.columns]
fap_donor_means["SASP12_mean"] = fap_donor_means[fap_sasp_cols].mean(axis=1)
fap_donor_means["SASP12_n_genes"] = len(fap_sasp_cols)
print(f"    FAP SASP12 computed from {len(fap_sasp_cols)}/12 genes")

# MuSC SASP12
musc_sasp_cols = [g for g in SASP12 if g in musc_donor_means.columns]
musc_donor_means["SASP12_mean"] = musc_donor_means[musc_sasp_cols].mean(axis=1)
musc_donor_means["SASP12_n_genes"] = len(musc_sasp_cols)
print(f"    MuSC SASP12 computed from {len(musc_sasp_cols)}/12 genes")

# ============================================================================
# PRIMARY ANALYSIS: rho(p21, SASP) and rho(p21, JUNB)
# ============================================================================
print("\n[6] Primary correlation analysis...")

results = {
    "metadata": {
        "batch": "batch_034",
        "purpose": "Cross-Compartment p21->SASP Analysis",
        "date": "2026-04-11",
        "alpha_primary": ALPHA_PRIMARY,
        "primary_rho_threshold": PRIMARY_RHO_THRESHOLD,
        "baselines": {
            "f092_vascular_p21_sasp": 0.9126,
            "f080_fap_junb_sasp": 0.023,
            "f074_musc_junb_sasp": 0.072
        }
    },
    "fap": {},
    "musc": {},
    "interpretation": {}
}

# --- FAP correlations ---
fap_n = len(fap_donor_means)
print(f"\n  [FAP] N = {fap_n} donors")

fap_cdkn1a = fap_donor_means[CDKN1A_GENE].values
fap_junb = fap_donor_means[JUNB_GENE].values
fap_sasp = fap_donor_means["SASP12_mean"].values

# rho(p21, SASP)
fap_rho_p21_sasp, fap_p_p21_sasp = stats.spearmanr(fap_cdkn1a, fap_sasp)

# rho(p21, JUNB)
fap_rho_p21_junb, fap_p_p21_junb = stats.spearmanr(fap_cdkn1a, fap_junb)

# Partial correlation: r(p21->SASP | JUNB)
if not np.isnan(fap_rho_p21_junb):
    fap_partial_r, fap_partial_p = partial_spearman(fap_cdkn1a, fap_sasp, fap_junb, fap_n)
else:
    fap_partial_r, fap_partial_p = np.nan, np.nan

results["fap"] = {
    "n_donors": int(fap_n),
    "n_cells": int(ad_fap_fap.n_obs),
    "cdkn1a_detection_rate": float(fap_cdkn1a_detection),
    "junb_detection_rate": float(fap_junb_detection),
    "rho_p21_sasp": {"estimate": float(fap_rho_p21_sasp), "p_value": float(fap_p_p21_sasp)},
    "rho_p21_junb": {"estimate": float(fap_rho_p21_junb), "p_value": float(fap_p_p21_junb)},
    "partial_r_p21_sasp_given_junb": {"estimate": float(fap_partial_r), "p_value": float(fap_partial_p)},
}

print(f"    rho(p21, SASP12): {fap_rho_p21_sasp:.4f}, p={fap_p_p21_sasp:.2e}")
print(f"    rho(p21, JUNB):   {fap_rho_p21_junb:.4f}, p={fap_p_p21_junb:.2e}")
print(f"    Partial r(p21->SASP|JUNB): {fap_partial_r:.4f}, p={fap_partial_p:.4f}")

# --- MuSC correlations ---
musc_n = len(musc_donor_means)
print(f"\n  [MuSC] N = {musc_n} donors")

musc_cdkn1a = musc_donor_means[CDKN1A_GENE].values
musc_junb = musc_donor_means[JUNB_GENE].values
musc_sasp = musc_donor_means["SASP12_mean"].values

# rho(p21, SASP)
musc_rho_p21_sasp, musc_p_p21_sasp = stats.spearmanr(musc_cdkn1a, musc_sasp)

# rho(p21, JUNB)
musc_rho_p21_junb, musc_p_p21_junb = stats.spearmanr(musc_cdkn1a, musc_junb)

results["musc"] = {
    "n_donors": int(musc_n),
    "n_cells": int(ad_musc_musc.n_obs),
    "cdkn1a_detection_rate": float(musc_cdkn1a_detection),
    "junb_detection_rate": float(musc_junb_detection),
    "rho_p21_sasp": {"estimate": float(musc_rho_p21_sasp), "p_value": float(musc_p_p21_sasp)},
    "rho_p21_junb": {"estimate": float(musc_rho_p21_junb), "p_value": float(musc_p_p21_junb)},
}

print(f"    rho(p21, SASP12): {musc_rho_p21_sasp:.4f}, p={musc_p_p21_sasp:.2e}")
print(f"    rho(p21, JUNB):   {musc_rho_p21_junb:.4f}, p={musc_p_p21_junb:.2e}")

# ============================================================================
# AGE-STRATIFIED ANALYSIS
# ============================================================================
print("\n[7] Age-stratified analysis...")

# FAP age-stratified
fap_old_mask = fap_donor_means["age_group"] == "old"
fap_young_mask = fap_donor_means["age_group"] == "young"

fap_young_n = fap_old_n = 0
fap_young_rho = fap_young_p = np.nan
fap_old_rho = fap_old_p = np.nan

if fap_young_mask.sum() >= 5:
    sub = fap_donor_means[fap_young_mask]
    fap_young_n = len(sub)
    fap_young_rho, fap_young_p = stats.spearmanr(sub[CDKN1A_GENE], sub["SASP12_mean"])
    print(f"    FAP young (N={fap_young_n}): rho(p21,SASP)={fap_young_rho:.4f}, p={fap_young_p:.4f}")
else:
    print(f"    FAP young: insufficient donors (N={fap_young_mask.sum()}, need >=5)")

if fap_old_mask.sum() >= 5:
    sub = fap_donor_means[fap_old_mask]
    fap_old_n = len(sub)
    fap_old_rho, fap_old_p = stats.spearmanr(sub[CDKN1A_GENE], sub["SASP12_mean"])
    print(f"    FAP old (N={fap_old_n}): rho(p21,SASP)={fap_old_rho:.4f}, p={fap_old_p:.4f}")
else:
    print(f"    FAP old: insufficient donors (N={fap_old_mask.sum()}, need >=5)")

results["fap"]["young_rho_p21_sasp"] = {"estimate": float(fap_young_rho), "p_value": float(fap_young_p), "n": int(fap_young_n)}
results["fap"]["old_rho_p21_sasp"] = {"estimate": float(fap_old_rho), "p_value": float(fap_old_p), "n": int(fap_old_n)}

# MuSC age-stratified
musc_old_mask = musc_donor_means["age_group"] == "old"
musc_young_mask = musc_donor_means["age_group"] == "young"

musc_young_n = musc_old_n = 0
musc_young_rho = musc_young_p = np.nan
musc_old_rho = musc_old_p = np.nan

if musc_young_mask.sum() >= 5:
    sub = musc_donor_means[musc_young_mask]
    musc_young_n = len(sub)
    musc_young_rho, musc_young_p = stats.spearmanr(sub[CDKN1A_GENE], sub["SASP12_mean"])
    print(f"    MuSC young (N={musc_young_n}): rho(p21,SASP)={musc_young_rho:.4f}, p={musc_young_p:.4f}")
else:
    print(f"    MuSC young: insufficient donors (N={musc_young_mask.sum()}, need >=5)")

if musc_old_mask.sum() >= 5:
    sub = musc_donor_means[musc_old_mask]
    musc_old_n = len(sub)
    musc_old_rho, musc_old_p = stats.spearmanr(sub[CDKN1A_GENE], sub["SASP12_mean"])
    print(f"    MuSC old (N={musc_old_n}): rho(p21,SASP)={musc_old_rho:.4f}, p={musc_old_p:.4f}")
else:
    print(f"    MuSC old: insufficient donors (N={musc_old_mask.sum()}, need >=5)")

results["musc"]["young_rho_p21_sasp"] = {"estimate": float(musc_young_rho), "p_value": float(musc_young_p), "n": int(musc_young_n)}
results["musc"]["old_rho_p21_sasp"] = {"estimate": float(musc_old_rho), "p_value": float(musc_old_p), "n": int(musc_old_n)}

# ============================================================================
# PER-GENE EXPLORATORY ANALYSIS
# ============================================================================
print("\n[8] Per-gene exploratory correlations (EXPLORATORY - not corrected)...")

# FAP per-gene
fap_per_gene_results = {}
for gene in fap_sasp_cols:
    gene_vals = fap_donor_means[gene].values
    rho, pval = stats.spearmanr(fap_cdkn1a, gene_vals)
    fap_per_gene_results[gene] = {
        "rho_p21": float(rho),
        "p_value": float(pval),
        "note": "EXPLORATORY"
    }
    print(f"  FAP {gene}: rho={rho:.4f}, p={pval:.4f}")

# MuSC per-gene
musc_per_gene_results = {}
for gene in musc_sasp_cols:
    gene_vals = musc_donor_means[gene].values
    rho, pval = stats.spearmanr(musc_cdkn1a, gene_vals)
    musc_per_gene_results[gene] = {
        "rho_p21": float(rho),
        "p_value": float(pval),
        "note": "EXPLORATORY"
    }
    print(f"  MuSC {gene}: rho={rho:.4f}, p={pval:.4f}")

# ============================================================================
# INTERPRETATION AND DECISION RULES
# ============================================================================
print("\n[9] Applying pre-registered decision rules...")

# FAP interpretation based on pre-registered decision criteria
# Source: design.yaml decision thresholds
if uninterpretable_fap:
    fap_p21_sasp_interpretation = "UNINTERPRETABLE - insufficient CDKN1A detection"
elif fap_p_p21_sasp < 0:
    fap_p21_sasp_interpretation = "INCONCLUSIVE - NaN p-value"
elif fap_p_p21_sasp < ALPHA_PRIMARY:
    if fap_rho_p21_sasp >= PRIMARY_RHO_THRESHOLD:
        fap_p21_sasp_interpretation = "ESTABLISHED"
    else:
        fap_p21_sasp_interpretation = "SUGGESTED"
elif fap_p_p21_sasp < 0.05:
    fap_p21_sasp_interpretation = "PRELIMINARY"
else:
    fap_p21_sasp_interpretation = "INCONCLUSIVE"

# MuSC interpretation
if uninterpretable_musc:
    musc_p21_sasp_interpretation = "UNINTERPRETABLE - insufficient CDKN1A detection"
elif musc_p_p21_sasp < 0:
    musc_p21_sasp_interpretation = "INCONCLUSIVE - NaN p-value"
elif musc_p_p21_sasp < ALPHA_PRIMARY:
    if musc_rho_p21_sasp >= PRIMARY_RHO_THRESHOLD:
        musc_p21_sasp_interpretation = "ESTABLISHED"
    else:
        musc_p21_sasp_interpretation = "SUGGESTED"
elif musc_p_p21_sasp < 0.05:
    musc_p21_sasp_interpretation = "PRELIMINARY"
else:
    musc_p21_sasp_interpretation = "INCONCLUSIVE"

# --- Apply decision rules ---
# IF rho(p21, SASP) >= 0.50 -> p21->SASP is NOT vascular-specific
# IF rho(p21, SASP) < 0.50 in BOTH compartments -> p21->SASP is vascular-specific

fap_strong = (not uninterpretable_fap and
              fap_p_p21_sasp >= 0 and
              fap_p_p21_sasp < ALPHA_PRIMARY and
              fap_rho_p21_sasp >= PRIMARY_RHO_THRESHOLD)

musc_strong = (not uninterpretable_musc and
               musc_p_p21_sasp >= 0 and
               musc_p_p21_sasp < ALPHA_PRIMARY and
               musc_rho_p21_sasp >= PRIMARY_RHO_THRESHOLD)

# Overall interpretation
if fap_strong and musc_strong:
    overall_interpretation = "p21->SASP axis is NOT vascular-specific - coupling detected in FAPs and MuSCs"
elif fap_strong and not musc_strong:
    overall_interpretation = "p21->SASP axis is PARTIALLY vascular-specific - coupling in FAPs but not MuSCs"
elif not fap_strong and musc_strong:
    overall_interpretation = "p21->SASP axis is PARTIALLY vascular-specific - coupling in MuSCs but not FAPs"
elif uninterpretable_fap and uninterpretable_musc:
    overall_interpretation = "UNINTERPRETABLE - CDKN1A detection insufficient in both compartments"
elif uninterpretable_fap:
    overall_interpretation = "p21->SASP is likely VASCULAR-SPECIFIC - FAP analysis uninterpretable"
elif uninterpretable_musc:
    overall_interpretation = "p21->SASP is VASCULAR-SPECIFIC in FAPs - MuSC analysis uninterpretable"
else:
    overall_interpretation = "p21->SASP axis is VASCULAR-SPECIFIC - no significant coupling in FAPs or MuSCs"

print(f"\n  FAP rho(p21,SASP): {fap_rho_p21_sasp:.4f}")
print(f"  MuSC rho(p21,SASP): {musc_rho_p21_sasp:.4f}")
print(f"  Vascular reference (F092): 0.9126")
print(f"\n  Interpretation: {overall_interpretation}")

# ============================================================================
# COMPILE FINAL RESULTS
# ============================================================================
results["interpretation"] = {
    "fap_p21_sasp_interpretation": fap_p21_sasp_interpretation,
    "musc_p21_sasp_interpretation": musc_p21_sasp_interpretation,
    "overall_interpretation": overall_interpretation,
    "decision_rule_triggered": decision_rule_triggered if decision_rule_triggered else "No UNINTERPRETABLE gate triggered"
}

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[10] Saving results...")

# Save results.json
with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved: {OUTPUT_DIR}/results.json")

# Save donor means
fap_donor_means_out = fap_donor_means.copy()
fap_donor_means_out["compartment"] = "FAP"
fap_donor_means_out["donor_id"] = fap_donor_means_out.index

musc_donor_means_out = musc_donor_means.copy()
musc_donor_means_out["compartment"] = "MuSC"
musc_donor_means_out["donor_id"] = musc_donor_means_out.index

# Align columns
common_cols = ["donor_id", "compartment", "age", "age_group", CDKN1A_GENE, JUNB_GENE, "SASP12_mean", "n_cells"]
for col in common_cols:
    if col not in fap_donor_means_out.columns:
        fap_donor_means_out[col] = np.nan
    if col not in musc_donor_means_out.columns:
        musc_donor_means_out[col] = np.nan

all_donor_means = pd.concat([fap_donor_means_out[common_cols], musc_donor_means_out[common_cols]])
all_donor_means.to_csv(f"{OUTPUT_DIR}/donor_means.csv", index=False)
print(f"  Donor means saved: {OUTPUT_DIR}/donor_means.csv")

# Save per-gene results
per_gene_rows = []
for gene in SASP12:
    if gene in fap_per_gene_results:
        per_gene_rows.append({
            "gene": gene,
            "compartment": "FAP",
            "rho_p21": fap_per_gene_results[gene]["rho_p21"],
            "p_value": fap_per_gene_results[gene]["p_value"],
            "note": "EXPLORATORY"
        })
    if gene in musc_per_gene_results:
        per_gene_rows.append({
            "gene": gene,
            "compartment": "MuSC",
            "rho_p21": musc_per_gene_results[gene]["rho_p21"],
            "p_value": musc_per_gene_results[gene]["p_value"],
            "note": "EXPLORATORY"
        })

per_gene_df = pd.DataFrame(per_gene_rows)
per_gene_df.to_csv(f"{OUTPUT_DIR}/per_gene_results.csv", index=False)
print(f"  Per-gene results saved: {OUTPUT_DIR}/per_gene_results.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Cross-Compartment p21->SASP Analysis")
print("=" * 70)
print(f"\n  FAP:")
print(f"    N donors: {fap_n}")
print(f"    CDKN1A detection: {fap_cdkn1a_detection * 100:.1f}%")
print(f"    rho(p21, SASP12): {fap_rho_p21_sasp:.4f}, p={fap_p_p21_sasp:.4f}")
print(f"    rho(p21, JUNB): {fap_rho_p21_junb:.4f}")
print(f"    Partial r(p21->SASP|JUNB): {fap_partial_r:.4f}, p={fap_partial_p:.4f}")
print(f"\n  MuSC:")
print(f"    N donors: {musc_n}")
print(f"    CDKN1A detection: {musc_cdkn1a_detection * 100:.1f}%")
print(f"    rho(p21, SASP12): {musc_rho_p21_sasp:.4f}, p={musc_p_p21_sasp:.4f}")
print(f"    rho(p21, JUNB): {musc_rho_p21_junb:.4f}")
print(f"\n  BASELINE (Vascular, F092):")
print(f"    rho(p21, SASP12): 0.9126")
print(f"\n  CONCLUSION:")
print(f"    {overall_interpretation}")
print("=" * 70)
