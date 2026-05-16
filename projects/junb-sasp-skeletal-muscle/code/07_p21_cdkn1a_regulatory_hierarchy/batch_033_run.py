#!/usr/bin/env python3
"""
batch_033: p21 vs JUNB as Vascular Senescence Biomarker

Hypothesis (H092): rho(p21, SASP12) ≥ rho(JUNB, SASP12) at donor level.
If TRUE: p21 is superior biomarker; therapeutic thesis shifts to p21/CDK4/6.
If FALSE: JUNB remains primary; p21 is supplementary.

Data: Vascular_scsn_RNA.h5ad (same as F084/batch_023)
Cell types: CapEC, VenEC, ArtEC (excluding IL6+ VenEC)
N donors: 23

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

np.random.seed(42)

SASP12 = ["CCL2", "CXCL1", "CXCL2", "CXCL3", "CXCL6", "IL6", "CXCL8",
          "SERPINE1", "MMP1", "MMP3", "PLAU", "PLAUR"]

EC_TYPES = ["CapEC", "VenEC", "ArtEC"]

print("=" * 70)
print("batch_033: p21 vs JUNB as Vascular Senescence Biomarker")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading vascular data...")
ad = sc.read_h5ad("/home/yuanz/Documents/GitHub/biomarvin_fibro/data/Vascular_scsn_RNA.h5ad")
print(f"  Total cells: {ad.n_obs:,}")
print(f"  Annotations: {ad.obs['Annotation'].value_counts().to_dict()}")

# Filter to endothelial cells
ec_mask = ad.obs["Annotation"].isin(EC_TYPES)
ad_ec = ad[ec_mask].copy()
print(f"  Endothelial cells: {ad_ec.n_obs:,}")

# Check CDKN1A detection
cdkn1a_detection = (ad_ec.to_df()["CDKN1A"] > 0).mean() * 100
print(f"  CDKN1A detection rate: {cdkn1a_detection:.1f}%")

# Check available SASP genes
available_sasp = [g for g in SASP12 if g in ad_ec.var_names]
print(f"  Available SASP genes: {len(available_sasp)}/{len(SASP12)}")

# Check JUNB availability
junb_available = "JUNB" in ad_ec.var_names
print(f"  JUNB available: {junb_available}")
cdkn1a_available = "CDKN1A" in ad_ec.var_names
print(f"  CDKN1A available: {cdkn1a_available}")

# ============================================================================
# DONOR-LEVEL PSEUDOBULK
# ============================================================================
print("\n[2] Computing donor-level pseudobulk...")

X = ad_ec.to_df()
X["sample"] = ad_ec.obs["sample"].values

# Genes for analysis
analysis_genes = ["JUNB", "CDKN1A"] + available_sasp
analysis_genes = [g for g in analysis_genes if g in X.columns]

# Compute mean per donor (genes only, sample is groupby key)
donor_means = X[analysis_genes].groupby(X["sample"]).mean()

# Compute SASP12 composite (mean across available SASP genes)
sasp_cols = [g for g in available_sasp if g in donor_means.columns]
donor_means["SASP12_mean"] = donor_means[sasp_cols].mean(axis=1)

# Add age information from ad_ec
age_map = ad_ec.obs.groupby("sample")["age_pop"].first()
donor_means["age_pop"] = age_map

print(f"  Donors: {len(donor_means)}")
print(f"  Age distribution: {donor_means['age_pop'].value_counts().to_dict()}")

# Check for zero-variance genes
for gene in ["JUNB", "CDKN1A"]:
    if gene in donor_means.columns:
        if donor_means[gene].std() < 1e-10:
            print(f"  WARNING: {gene} has near-zero variance at donor level!")
            print(f"    Mean: {donor_means[gene].mean():.6f}, Std: {donor_means[gene].std():.6f}")

# ============================================================================
# PRIMARY ANALYSIS: rho(p21, SASP) vs rho(JUNB, SASP)
# ============================================================================
print("\n[3] Primary correlation analysis...")

results = {
    "metadata": {
        "batch": "batch_033",
        "purpose": "p21 vs JUNB as vascular senescence biomarker",
        "date": "2026-04-11",
        "n_donors": int(len(donor_means)),
        "n_sasp_genes": len(available_sasp),
        "cdkn1a_detection_rate_pct": float(cdkn1a_detection),
        "data_source": "Vascular_scsn_RNA.h5ad",
        "cell_types": EC_TYPES
    },
    "primary": {},
    "secondary": {},
    "per_gene": {},
    "age_stratified": {},
    "decision": {}
}

# --- Primary correlations ---
junb_vals = donor_means["JUNB"].values
cdkn1a_vals = donor_means["CDKN1A"].values
sasp_vals = donor_means["SASP12_mean"].values

# rho(JUNB, SASP12)
rho_junb, p_junb = stats.spearmanr(junb_vals, sasp_vals)
results["primary"]["rho_junb_sasp"] = float(rho_junb)
results["primary"]["p_junb_sasp"] = float(p_junb)
print(f"  rho(JUNB, SASP12): {rho_junb:.4f}, p={p_junb:.2e}")

# rho(CDKN1A, SASP12)
rho_p21, p_p21 = stats.spearmanr(cdkn1a_vals, sasp_vals)
results["primary"]["rho_p21_sasp"] = float(rho_p21)
results["primary"]["p_p21_sasp"] = float(p_p21)
print(f"  rho(CDKN1A, SASP12): {rho_p21:.4f}, p={p_p21:.2e}")

# rho(JUNB, CDKN1A)
rho_junb_p21, p_junb_p21 = stats.spearmanr(junb_vals, cdkn1a_vals)
results["primary"]["rho_junb_p21"] = float(rho_junb_p21)
results["primary"]["p_junb_p21"] = float(p_junb_p21)
print(f"  rho(JUNB, CDKN1A): {rho_junb_p21:.4f}, p={p_junb_p21:.2e}")

# Delta
delta = rho_p21 - rho_junb
results["primary"]["delta"] = float(delta)
print(f"  Delta (p21 - JUNB): {delta:+.4f}")

# ============================================================================
# SECONDARY ANALYSIS: Fisher Z comparison
# ============================================================================
print("\n[4] Fisher Z comparison of two correlations...")

# Fisher Z transformation
def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_z_var(n):
    return 1 / (n - 3)  # variance of Fisher Z

n = len(donor_means)
z_junb = fisher_z(rho_junb)
z_p21 = fisher_z(rho_p21)
se_diff = np.sqrt(fisher_z_var(n) + fisher_z_var(n))

# Two-tailed test: z_diff = (z1 - z2) / SE_diff
z_diff = (z_p21 - z_junb) / se_diff
p_diff = 2 * (1 - normal_dist.cdf(abs(z_diff)))  # two-tailed p-value

results["secondary"]["fisher_z_comparison"] = {
    "z_junb": float(z_junb),
    "z_p21": float(z_p21),
    "z_diff": float(z_diff),
    "p_value_two_tailed": float(p_diff),
    "se_diff": float(se_diff),
    "n": int(n)
}
print(f"  Fisher Z diff: {z_diff:.3f}, p={p_diff:.4f}")
print(f"  Interpretation: {'Significant difference' if p_diff < 0.05 else 'No significant difference'}")

# ============================================================================
# PARTIAL CORRELATION: p21→SASP controlling for JUNB
# ============================================================================
print("\n[5] Partial correlation: CDKN1A→SASP controlling for JUNB...")

# Partial correlation: r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz²)(1-r_yz²))
r_p21_sasp = rho_p21
r_p21_junb = rho_junb_p21
r_junb_sasp = rho_junb

numerator = r_p21_sasp - (r_p21_junb * r_junb_sasp)
denominator = np.sqrt((1 - r_p21_junb**2) * (1 - r_junb_sasp**2))
partial_r = numerator / denominator

# Approximate p-value using Fisher Z on partial r
n_vars = len(donor_means)
z_partial = fisher_z(partial_r)
se_partial = 1 / np.sqrt(n_vars - 3 - 1)  # df = n - k - 1, k=1 control variable
p_partial = 2 * (1 - normal_dist.cdf(abs(z_partial)))

results["secondary"]["partial_correlation"] = {
    "r_partial_p21_sasp_given_junb": float(partial_r),
    "z_partial": float(z_partial),
    "p_partial": float(p_partial),
    "interpretation": "p21 provides independent information about SASP if partial_r differs substantially from 0"
}
print(f"  Partial r(CDKN1A, SASP | JUNB): {partial_r:.4f}, p={p_partial:.4f}")
if abs(partial_r) < 0.10:
    print("  Interpretation: p21 provides NO additional information beyond JUNB")
elif abs(partial_r) < 0.30:
    print("  Interpretation: p21 provides MINIMAL additional information beyond JUNB")
else:
    print("  Interpretation: p21 provides NON-TRIVIAL additional information beyond JUNB")

# ============================================================================
# PER-GENE ANALYSIS: p21 vs JUNB correlation with each SASP gene
# ============================================================================
print("\n[6] Per-gene correlation comparison...")

per_gene_results = {}
for gene in available_sasp:
    gene_vals = donor_means[gene].values

    # rho(JUNB, gene)
    r_junb_gene, p_jg = stats.spearmanr(junb_vals, gene_vals)

    # rho(CDKN1A, gene)
    r_p21_gene, p_pg = stats.spearmanr(cdkn1a_vals, gene_vals)

    per_gene_results[gene] = {
        "rho_junb": float(r_junb_gene),
        "p_junb": float(p_jg),
        "rho_p21": float(r_p21_gene),
        "p_p21": float(p_pg),
        "delta": float(r_p21_gene - r_junb_gene),
        "winner": "CDKN1A" if abs(r_p21_gene) > abs(r_junb_gene) else "JUNB"
    }
    print(f"  {gene}: JUNB={r_junb_gene:.3f}, CDKN1A={r_p21_gene:.3f}, "
          f"delta={r_p21_gene - r_junb_gene:+.3f}, winner={per_gene_results[gene]['winner']}")

results["per_gene"] = per_gene_results

# Count genes where p21 > JUNB
n_p21_wins = sum(1 for g in per_gene_results.values() if g["winner"] == "CDKN1A")
n_junb_wins = sum(1 for g in per_gene_results.values() if g["winner"] == "JUNB")
results["secondary"]["per_gene_summary"] = {
    "n_p21_wins": n_p21_wins,
    "n_junb_wins": n_junb_wins,
    "n_total": len(available_sasp)
}
print(f"\n  Summary: CDKN1A wins {n_p21_wins}/{len(available_sasp)}, JUNB wins {n_junb_wins}/{len(available_sasp)}")

# ============================================================================
# AGE-STRATIFIED ANALYSIS
# ============================================================================
print("\n[7] Age-stratified correlations...")

old_mask = donor_means["age_pop"] == "old_pop"
young_mask = donor_means["age_pop"] == "young_pop"

age_strat = {}
for label, mask in [("old", old_mask), ("young", young_mask)]:
    sub = donor_means[mask]
    n_sub = len(sub)
    if n_sub >= 3:
        rho_j, p_j = stats.spearmanr(sub["JUNB"], sub["SASP12_mean"])
        rho_p, p_p = stats.spearmanr(sub["CDKN1A"], sub["SASP12_mean"])
        age_strat[label] = {
            "n_donors": n_sub,
            "rho_junb_sasp": float(rho_j),
            "p_junb": float(p_j),
            "rho_p21_sasp": float(rho_p),
            "p_p21": float(p_p)
        }
        print(f"  {label} (N={n_sub}): rho(JUNB,SASP)={rho_j:.3f}, rho(p21,SASP)={rho_p:.3f}")
    else:
        age_strat[label] = {"n_donors": n_sub, "note": "Insufficient donors for correlation"}

results["age_stratified"] = age_strat

# ============================================================================
# DECISION RULE (REVISED per Science Critics)
# ============================================================================
print("\n[8] Applying decision rule (REVISED)...")
print("  NOTE: Delta comparison is EXPLORATORY (power=0.50 for delta=0.05)")
print("  NOTE: Partial correlation is PRIMARY test")

# Thresholds from revised brief
rho_biomarker_threshold = 0.50  # p21 must exceed this to be useful
partial_r_independent = 0.30    # p21 provides non-trivial independent signal
partial_r_marginal = 0.10       # p21 provides marginal independent signal
delta_threshold = 0.05           # EXPLORATORY only

decision = {
    "primary_test": "partial_correlation",
    "exploratory_test": "delta_comparison",
    "rho_threshold": rho_biomarker_threshold,
    "partial_r_independent_threshold": partial_r_independent,
    "partial_r_marginal_threshold": partial_r_marginal,
    "delta_threshold_exploratory": delta_threshold,
    "power_note": "Delta comparison power=0.50 for delta=0.05 (N=23). EXPLORATORY ONLY.",
    "rho_junb_sasp": float(rho_junb),
    "rho_p21_sasp": float(rho_p21),
    "partial_r": float(partial_r),
    "p_partial": float(p_partial),
    "delta": float(delta)
}

# --- Step 1: Is p21 a useful biomarker? (CONFIRMATORY) ---
print(f"\n  Step 1: Is p21 a useful biomarker?")
print(f"  Threshold: rho(p21,SASP) > {rho_biomarker_threshold}")
print(f"  Observed: rho(p21,SASP) = {rho_p21:.4f}")

if rho_p21 < rho_biomarker_threshold:
    biomarker_verdict = "NOT_USEFUL"
    biomarker_reasoning = f"rho(p21,SASP)={rho_p21:.4f} < {rho_biomarker_threshold} — p21 is NOT a useful biomarker"
    finding = "REFUTED"
    print(f"  >> {biomarker_verdict}: {biomarker_reasoning}")
    decision["step1"] = {
        "question": "Is p21 a useful biomarker?",
        "rho_p21_sasp": float(rho_p21),
        "threshold": rho_biomarker_threshold,
        "verdict": biomarker_verdict,
        "reasoning": biomarker_reasoning
    }
    decision["finding"] = finding
    decision["biomarker_verdict"] = biomarker_verdict
    decision["biomarker_reasoning"] = biomarker_reasoning
    decision["verdict"] = "F092_REFUTED"
    # Stop here — p21 is not useful
    print("\n  DECISION: F092 REFUTED — p21 is not a useful biomarker for SASP")
else:
    biomarker_verdict = "USEFUL"
    biomarker_reasoning = f"rho(p21,SASP)={rho_p21:.4f} >= {rho_biomarker_threshold} — p21 IS a useful biomarker"
    print(f"  >> {biomarker_verdict}: {biomarker_reasoning}")
    decision["step1"] = {
        "question": "Is p21 a useful biomarker?",
        "rho_p21_sasp": float(rho_p21),
        "threshold": rho_biomarker_threshold,
        "verdict": biomarker_verdict,
        "reasoning": biomarker_reasoning
    }

    # --- Step 2: Does p21 provide independent signal? (PRIMARY) ---
    print(f"\n  Step 2: Does p21 provide independent SASP signal beyond JUNB? (PRIMARY)")
    print(f"  Partial r(p21→SASP | JUNB) = {partial_r:.4f}, p={p_partial:.4f}")
    print(f"  Thresholds: >{partial_r_independent}=independent, >{partial_r_marginal}=marginal, ≤{partial_r_marginal}=redundant")

    if partial_r > partial_r_independent:
        step2_verdict = "INDEPENDENT"
        step2_reasoning = f"partial_r={partial_r:.4f} > {partial_r_independent} — p21 provides non-trivial independent SASP signal"
        finding = "SUGGESTED"
    elif partial_r > partial_r_marginal:
        step2_verdict = "MARGINAL"
        step2_reasoning = f"partial_r={partial_r:.4f} > {partial_r_marginal} — p21 provides marginal independent signal"
        finding = "SUGGESTED"
    elif partial_r > 0:
        step2_verdict = "REDUNDANT"
        step2_reasoning = f"0 < partial_r={partial_r:.4f} ≤ {partial_r_marginal} — p21 provides negligible independent signal"
        finding = "SUGGESTED"
    else:
        step2_verdict = "MEDIATED"
        step2_reasoning = f"partial_r={partial_r:.4f} ≤ 0 — p21 is fully mediated by JUNB"
        finding = "SPECULATIVE"

    print(f"  >> {step2_verdict}: {step2_reasoning}")
    print(f"  >> Finding: {finding}")

    decision["step2"] = {
        "question": "Does p21 provide independent SASP signal beyond JUNB?",
        "partial_r": float(partial_r),
        "p_partial": float(p_partial),
        "verdict": step2_verdict,
        "reasoning": step2_reasoning
    }
    decision["finding"] = finding

    # --- Step 3: Delta comparison (EXPLORATORY) ---
    print(f"\n  Step 3: Raw delta comparison (EXPLORATORY — power=0.50 for delta=0.05)")
    print(f"  rho(p21,SASP)={rho_p21:.4f}, rho(JUNB,SASP)={rho_junb:.4f}")
    print(f"  Delta = {delta:+.4f}")
    print(f"  NOTE: Cannot support confirmatory claims. Interpret with caution.")

    decision["step3"] = {
        "question": "Delta comparison (EXPLORATORY)",
        "rho_junb_sasp": float(rho_junb),
        "rho_p21_sasp": float(rho_p21),
        "delta": float(delta),
        "note": "EXPLORATORY — power=0.50 for delta=0.05, N=23. Cannot support confirmatory claims."
    }

    # --- Overall Verdict ---
    if step2_verdict == "INDEPENDENT":
        decision["verdict"] = "p21_INDEPENDENT"
        print(f"\n  OVERALL: p21 provides INDEPENDENT SASP signal beyond JUNB")
        print(f"  Therapeutic implication: p21→CDK4/6 is complementary to JNK→JUNB")
    elif step2_verdict == "MARGINAL":
        decision["verdict"] = "p21_MARGINAL"
        print(f"\n  OVERALL: p21 provides MARGINAL independent SASP signal")
        print(f"  Therapeutic implication: JUNB remains primary target; p21 supplementary")
    elif step2_verdict == "REDUNDANT":
        decision["verdict"] = "p21_REDUNDANT"
        print(f"\n  OVERALL: p21 is REDUNDANT — JUNB captures all SASP signal")
        print(f"  Therapeutic implication: JUNB is the primary target; p21 adds no information")
    else:
        decision["verdict"] = "p21_MEDIATED"
        print(f"\n  OVERALL: p21 is MEDIATED by JUNB")
        print(f"  Therapeutic implication: JUNB is the upstream driver; p21 is downstream readout")

results["decision"] = decision

# ============================================================================
# OVERALL SUMMARY
# ============================================================================
print(f"\n  Biomarker verdict: {biomarker_verdict}")
print(f"  Finding: {decision.get('finding', 'N/A')}")
print(f"  Verdict: {decision.get('verdict', 'N/A')}")
print("\n" + "=" * 70)
print("SUMMARY: p21 vs JUNB as Vascular Senescence Biomarker")
print("=" * 70)
print(f"  rho(JUNB, SASP12): {rho_junb:.4f}")
print(f"  rho(p21, SASP12): {rho_p21:.4f}")
print(f"  rho(JUNB, p21):   {rho_junb_p21:.4f} (multicollinearity note)")
print(f"  Partial r(p21→SASP|JUNB): {partial_r:.4f}, p={p_partial:.4f} [PRIMARY]")
print(f"  Delta (EXPLORATORY): {delta:+.4f}")
print(f"  Fisher Z p: {p_diff:.4f} [EXPLORATORY — power=0.50]")
print(f"  Biomarker verdict: {biomarker_verdict}")
print(f"  Finding: {decision.get('finding', 'N/A')}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
output_path = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_033/results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_path}")

# Also save donor-level data for reference
donor_path = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_033/donor_means.csv"
donor_means.to_csv(donor_path)
print(f"Donor means saved to: {donor_path}")

print("\n" + "=" * 70)
