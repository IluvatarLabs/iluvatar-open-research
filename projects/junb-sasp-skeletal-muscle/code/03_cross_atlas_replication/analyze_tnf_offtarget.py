#!/usr/bin/env python3
"""
GAP 2: TNF→TNFRSF1A Off-Target / Cell-Type Specificity Analysis
Experiment ID: EXP-GAP2-020

Pre-registered thresholds:
- Cohen d >= 0.20 = meaningful age effect
- TNF-TNFRSF1A rho >= 0.15 = meaningful co-expression
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data"
OUTPUT_DIR = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_020/tnf_offtarget"

COMPARTMENTS = {
    "FAPs": "OMIX004308-02.h5ad",
    "MuSCs": "MuSC_scsn_RNA.h5ad",
    "Immune": "Immune_scsn_RNA.h5ad",
    "Vascular": "Vascular_scsn_RNA.h5ad"
}

YOUNG_THRESHOLD = 40
OLD_THRESHOLD = 70

# ============================================================
# Utility Functions
# ============================================================
def cohens_d(group1, group2):
    """Compute Cohen's d for two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan, np.nan
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan, np.nan, np.nan, np.nan
    d = (mean2 - mean1) / pooled_std
    # Welch's t-test
    se = np.sqrt(var1 / n1 + var2 / n2)
    t_stat = (mean2 - mean1) / se if se > 0 else 0
    df = ((var1 / n1 + var2 / n2) ** 2) / (
        (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    )
    p_val = 2 * stats.t.sf(np.abs(t_stat), df)
    return d, t_stat, df, p_val

def pearson_rho(x, y):
    """Compute Pearson correlation between two arrays."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p

# ============================================================
# Main Analysis
# ============================================================
results = {
    "experiment_id": "EXP-GAP2-020",
    "thresholds": {
        "cohen_d_min": 0.20,
        "pearson_rho_min": 0.15
    },
    "compartments": {}
}

for comp_name, filename in COMPARTMENTS.items():
    print(f"\n{'='*60}")
    print(f"Analyzing: {comp_name}")
    print(f"{'='*60}")

    # Load data
    adata = sc.read_h5ad(f"{DATA_DIR}/{filename}")
    print(f"Total cells: {adata.n_obs}")

    # Get age information
    if 'age' in adata.obs.columns:
        age_vals = adata.obs['age']
        ages = pd.to_numeric(age_vals, errors='coerce').values
    elif 'Age' in adata.obs.columns:
        age_vals = adata.obs['Age']
        ages = pd.to_numeric(age_vals, errors='coerce').values
    else:
        # Try to find age column
        age_cols = [c for c in adata.obs.columns if 'age' in c.lower()]
        if age_cols:
            age_vals = adata.obs[age_cols[0]]
            ages = pd.to_numeric(age_vals, errors='coerce').values
        else:
            print(f"  WARNING: No age column found in {comp_name}")
            ages = None

    # Define age groups
    if ages is not None:
        young_mask = ages <= YOUNG_THRESHOLD
        old_mask = ages >= OLD_THRESHOLD
        young_n = young_mask.sum()
        old_n = old_mask.sum()
        print(f"  Young (≤{YOUNG_THRESHOLD}): {young_n} cells")
        print(f"  Old (≥{OLD_THRESHOLD}): {old_n} cells")
    else:
        young_n, old_n = 0, 0

    # Get TNF and TNFRSF1A expression
    genes = adata.var_names.tolist()
    has_tnf = 'TNF' in genes
    has_tnfrsf1a = 'TNFRSF1A' in genes
    print(f"  TNF present: {has_tnf}, TNFRSF1A present: {has_tnfrsf1a}")

    comp_result = {
        "total_cells": int(adata.n_obs),
        "young_n": int(young_n),
        "old_n": int(old_n),
        "tnf_present": has_tnf,
        "tnfrsf1a_present": has_tnfrsf1a,
        "tnf": {},
        "tnfrsf1a": {},
        "co_expression": {}
    }

    if young_n > 0 and old_n > 0:
        expr_df = pd.DataFrame(index=adata.obs_names)
        expr_df['age'] = ages

        if has_tnf:
            expr_df['TNF'] = adata[:, 'TNF'].X.toarray().flatten() if hasattr(adata[:, 'TNF'].X, 'toarray') else adata[:, 'TNF'].X.flatten()
        if has_tnfrsf1a:
            expr_df['TNFRSF1A'] = adata[:, 'TNFRSF1A'].X.toarray().flatten() if hasattr(adata[:, 'TNFRSF1A'].X, 'toarray') else adata[:, 'TNFRSF1A'].X.flatten()

        young_expr = expr_df[young_mask]
        old_expr = expr_df[old_mask]

        # TNF analysis
        if has_tnf:
            tnf_young_mean = young_expr['TNF'].mean()
            tnf_old_mean = old_expr['TNF'].mean()
            tnf_d, tnf_t, tnf_df, tnf_p = cohens_d(young_expr['TNF'].values, old_expr['TNF'].values)

            print(f"\n  TNF:")
            print(f"    Young mean: {tnf_young_mean:.4f}")
            print(f"    Old mean: {tnf_old_mean:.4f}")
            print(f"    Cohen d: {tnf_d:.3f} (t={tnf_t:.2f}, p={tnf_p:.4f})")

            comp_result['tnf'] = {
                "young_mean": float(tnf_young_mean),
                "old_mean": float(tnf_old_mean),
                "cohen_d": float(tnf_d),
                "t_stat": float(tnf_t),
                "df": float(tnf_df),
                "p_value": float(tnf_p),
                "significant": bool(tnf_p < 0.05),
                "meaningful": bool(abs(tnf_d) >= 0.20)
            }

        # TNFRSF1A analysis
        if has_tnfrsf1a:
            tnfrsf1a_young_mean = young_expr['TNFRSF1A'].mean()
            tnfrsf1a_old_mean = old_expr['TNFRSF1A'].mean()
            tnfrsf1a_d, tnfrsf1a_t, tnfrsf1a_df, tnfrsf1a_p = cohens_d(young_expr['TNFRSF1A'].values, old_expr['TNFRSF1A'].values)

            print(f"\n  TNFRSF1A:")
            print(f"    Young mean: {tnfrsf1a_young_mean:.4f}")
            print(f"    Old mean: {tnfrsf1a_old_mean:.4f}")
            print(f"    Cohen d: {tnfrsf1a_d:.3f} (t={tnfrsf1a_t:.2f}, p={tnfrsf1a_p:.4f})")

            comp_result['tnfrsf1a'] = {
                "young_mean": float(tnfrsf1a_young_mean),
                "old_mean": float(tnfrsf1a_old_mean),
                "cohen_d": float(tnfrsf1a_d),
                "t_stat": float(tnfrsf1a_t),
                "df": float(tnfrsf1a_df),
                "p_value": float(tnfrsf1a_p),
                "significant": bool(tnfrsf1a_p < 0.05),
                "meaningful": bool(abs(tnfrsf1a_d) >= 0.20)
            }

        # Co-expression analysis
        if has_tnf and has_tnfrsf1a:
            rho, rho_p = pearson_rho(expr_df['TNF'].values, expr_df['TNFRSF1A'].values)
            print(f"\n  TNF-TNFRSF1A co-expression:")
            print(f"    Pearson r: {rho:.3f} (p={rho_p:.4f})")

            comp_result['co_expression'] = {
                "pearson_r": float(rho),
                "p_value": float(rho_p),
                "meaningful": bool(abs(rho) >= 0.15)
            }

    results["compartments"][comp_name] = comp_result

# ============================================================
# Vascular Subtype Analysis
# ============================================================
print(f"\n{'='*60}")
print("Vascular Subtype Analysis")
print(f"{'='*60}")

vasc_adata = sc.read_h5ad(f"{DATA_DIR}/Vascular_scsn_RNA.h5ad")

# Find subtype annotation
subtype_col = None
for col in ['cell_type', 'CellType', 'celltype', 'subtype', 'Subtype', 'cell_type', 'Annotation', 'annotation']:
    if col in vasc_adata.obs.columns:
        subtype_col = col
        break

if subtype_col:
    subtypes = vasc_adata.obs[subtype_col].unique()
    print(f"  Subtype column: {subtype_col}")
    print(f"  Subtypes found: {list(subtypes)}")

    vasc_subtype_results = {}

    if 'age' in vasc_adata.obs.columns:
        vasc_ages = pd.to_numeric(vasc_adata.obs['age'], errors='coerce').values
    elif 'Age' in vasc_adata.obs.columns:
        vasc_ages = pd.to_numeric(vasc_adata.obs['Age'], errors='coerce').values
    else:
        age_cols = [c for c in vasc_adata.obs.columns if 'age' in c.lower()]
        vasc_ages = pd.to_numeric(vasc_adata.obs[age_cols[0]], errors='coerce').values if age_cols else None

    for subtype in subtypes:
        subtype_mask = vasc_adata.obs[subtype_col] == subtype
        subtype_n = subtype_mask.sum()
        print(f"\n  {subtype}: {subtype_n} cells")

        subtype_result = {"n": int(subtype_n)}

        if vasc_ages is not None:
            young_mask = (vasc_ages <= YOUNG_THRESHOLD) & subtype_mask
            old_mask = (vasc_ages >= OLD_THRESHOLD) & subtype_mask
            young_n = young_mask.sum()
            old_n = old_mask.sum()
            subtype_result["young_n"] = int(young_n)
            subtype_result["old_n"] = int(old_n)

            if young_n >= 3 and old_n >= 3:
                expr_df = pd.DataFrame(index=vasc_adata.obs_names)
                expr_df['age'] = vasc_ages

                if 'TNF' in vasc_adata.var_names:
                    expr_df['TNF'] = vasc_adata[:, 'TNF'].X.toarray().flatten() if hasattr(vasc_adata[:, 'TNF'].X, 'toarray') else vasc_adata[:, 'TNF'].X.flatten()
                if 'TNFRSF1A' in vasc_adata.var_names:
                    expr_df['TNFRSF1A'] = vasc_adata[:, 'TNFRSF1A'].X.toarray().flatten() if hasattr(vasc_adata[:, 'TNFRSF1A'].X, 'toarray') else vasc_adata[:, 'TNFRSF1A'].X.flatten()

                young_expr = expr_df[young_mask]
                old_expr = expr_df[old_mask]

                if 'TNF' in vasc_adata.var_names:
                    tnf_d, _, _, tnf_p = cohens_d(young_expr['TNF'].values, old_expr['TNF'].values)
                    subtype_result['tnf_cohen_d'] = float(tnf_d)
                    subtype_result['tnf_p'] = float(tnf_p)
                    print(f"    TNF Cohen d: {tnf_d:.3f} (p={tnf_p:.4f})")

                if 'TNFRSF1A' in vasc_adata.var_names:
                    tnfrsf1a_d, _, _, tnfrsf1a_p = cohens_d(young_expr['TNFRSF1A'].values, old_expr['TNFRSF1A'].values)
                    subtype_result['tnfrsf1a_cohen_d'] = float(tnfrsf1a_d)
                    subtype_result['tnfrsf1a_p'] = float(tnfrsf1a_p)
                    print(f"    TNFRSF1A Cohen d: {tnfrsf1a_d:.3f} (p={tnfrsf1a_p:.4f})")

        vasc_subtype_results[str(subtype)] = subtype_result

    results["vascular_subtypes"] = vasc_subtype_results

# ============================================================
# Cross-Compartment Summary
# ============================================================
print(f"\n{'='*60}")
print("Cross-Compartment Summary")
print(f"{'='*60}")

summary_table = []
for comp_name, comp_data in results["compartments"].items():
    row = {
        "compartment": comp_name,
        "n_young": comp_data.get("young_n", 0),
        "n_old": comp_data.get("old_n", 0),
        "tnf_d": comp_data.get("tnf", {}).get("cohen_d", np.nan),
        "tnf_p": comp_data.get("tnf", {}).get("p_value", np.nan),
        "tnfrsf1a_d": comp_data.get("tnfrsf1a", {}).get("cohen_d", np.nan),
        "tnfrsf1a_p": comp_data.get("tnfrsf1a", {}).get("p_value", np.nan),
        "coexpr_r": comp_data.get("co_expression", {}).get("pearson_r", np.nan)
    }
    summary_table.append(row)

summary_df = pd.DataFrame(summary_table)
print("\n" + summary_df.to_string(index=False))

results["summary_table"] = summary_table

# ============================================================
# Save Results
# ============================================================
output_path = f"{OUTPUT_DIR}/results.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✓ Results saved to: {output_path}")

# ============================================================
# Generate Interpretation
# ============================================================
print(f"\n{'='*60}")
print("Key Findings for Challenge Doc")
print(f"{'='*60}")

# Find FAP TNF d
fap_tnf_d = results["compartments"].get("FAPs", {}).get("tnf", {}).get("cohen_d", np.nan)
vasc_tnf_d = results["compartments"].get("Vascular", {}).get("tnf", {}).get("cohen_d", np.nan)
immune_tnf_d = results["compartments"].get("Immune", {}).get("tnf", {}).get("cohen_d", np.nan)
musc_tnf_d = results["compartments"].get("MuSCs", {}).get("tnf", {}).get("cohen_d", np.nan)

print(f"\nTNF Cohen d by compartment:")
print(f"  FAPs:     {fap_tnf_d:.3f}")
print(f"  Vascular: {vasc_tnf_d:.3f}")
print(f"  Immune:   {immune_tnf_d:.3f}")
print(f"  MuSCs:    {musc_tnf_d:.3f}")

# FAP-specific determination
if not np.isnan(fap_tnf_d):
    max_other_d = max([d for d in [vasc_tnf_d, immune_tnf_d, musc_tnf_d] if not np.isnan(d)], default=0)
    if abs(fap_tnf_d) > abs(max_other_d) * 1.5:
        print(f"\n✓ FAP-specific: FAP TNF d ({fap_tnf_d:.3f}) is 1.5x larger than next compartment")
    elif abs(fap_tnf_d) > abs(max_other_d):
        print(f"\n✓ FAP-dominant: FAP TNF d ({fap_tnf_d:.3f}) larger than other compartments")
    else:
        print(f"\n⚠ NOT FAP-specific: Other compartment(s) show comparable or larger TNF effect")

print("\n✓ Analysis complete")
