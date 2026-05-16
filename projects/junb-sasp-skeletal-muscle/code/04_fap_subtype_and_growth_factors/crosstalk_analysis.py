"""
MuSC-FAP Crosstalk Analysis (MUST DO 2)
========================================
Iteration 19: Tests whether JUNB+ aged FAPs produce ligands that signal to MuSCs

Hypothesis: JUNB+ aged FAPs produce elevated SASP ligands that can signal
to MuSCs via expressed receptors, potentially impairing MuSC function.

Author: Marvin
Date: 2026-04-09
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
FAP_PATH = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad"
MuSC_PATH = "/home/yuanz/Documents/GitHub/biomarvin_fibro/data/MuSC_scsn_RNA.h5ad"
OUTPUT_JSON = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_019/must_do_2_crosstalk_results.json"
OUTPUT_CSV = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_019/must_do_2_ligand_expression.csv"

# SASP ligand focus list (from protocol)
SASP_LIGANDS = [
    "IL6", "IL1B", "TNF", "CCL2", "CXCL1", "CXCL2", "CXCL3", "CXCL8",
    "SERPINE1", "MMP1", "MMP3", "PLAU", "PLAUR", "FGF7", "HGF", "PDGFA",
    "CXCL6", "CXCL10", "IL1RN"
]

# Ligand-receptor mappings (from protocol)
LIGAND_RECEPTORS = {
    "IL6": ["IL6R", "IL6ST"],
    "IL1B": ["IL1R1", "IL1R2", "IL1RAP"],
    "TNF": ["TNFRSF1A", "TNFRSF1B"],
    "CCL2": ["CCR2", "CCR4"],
    "CXCL1": ["CXCR1", "CXCR2"],
    "CXCL2": ["CXCR1", "CXCR2"],
    "CXCL3": ["CXCR1", "CXCR2"],
    "CXCL8": ["CXCR1", "CXCR2"],
    "SERPINE1": ["LRP1", "PLAUR"],
    "FGF7": ["FGFR1", "FGFR2", "FGFR3", "FGFR4"],
    "HGF": ["MET"],
    "PDGFA": ["PDGFRA", "PDGFRB"],
    "CXCL6": ["CXCR1", "CXCR2"],
    "CXCL10": ["CXCR3"],
    "IL1RN": ["IL1R1", "IL1R2"]  # IL1RN is antagonist
}

# MuSC activation states
MuSC_STATES = ["Quiescent MuSC", "Early Primed MuSC", "Late Primed MuSC", "Diff.MuSC"]

print("=" * 60)
print("MuSC-FAP Crosstalk Analysis")
print("=" * 60)

# =============================================================================
# STEP 1: Load FAP data and stratify by JUNB
# =============================================================================
print("\n[STEP 1] Loading FAP data and stratifying by JUNB expression...")

# Use backed mode for large file
fap_adata = sc.read_h5ad(FAP_PATH, backed='r')
print(f"FAP data: {fap_adata.n_obs} cells, {fap_adata.n_vars} genes (backed mode)")

# Get obs as DataFrame for filtering
obs_df = fap_adata.obs
print(f"FAP obs columns: {list(obs_df.columns)}")

# Convert age to numeric (it's stored as category with string values)
age_numeric = pd.to_numeric(obs_df['age'], errors='coerce')

# Filter to aged FAPs (age >= 70, Annotation contains 'FAP')
aged_fap_mask = (age_numeric >= 70) & obs_df['Annotation'].str.contains('FAP', na=False)
aged_fap_indices = np.where(aged_fap_mask)[0]
print(f"Aged FAPs: {len(aged_fap_indices)} cells")

# Get JUNB expression for aged FAPs only
junb_gene = "JUNB"
if junb_gene not in fap_adata.var_names:
    print(f"WARNING: {junb_gene} not found in FAP data!")
    print(f"Available genes containing JUN: {[g for g in fap_adata.var_names if 'JUN' in g.upper()]}")
    raise ValueError(f"Gene {junb_gene} not found")

junb_var_idx = list(fap_adata.var_names).index(junb_gene)
junb_expr = fap_adata.X[aged_fap_indices, junb_var_idx]
if hasattr(junb_expr, 'toarray'):
    junb_expr = junb_expr.toarray().flatten()
else:
    junb_expr = np.asarray(junb_expr).flatten()

# Define quartiles
q75 = np.percentile(junb_expr, 75)
q25 = np.percentile(junb_expr, 25)
print(f"JUNB expression: mean={np.mean(junb_expr):.4f}, median={np.median(junb_expr):.4f}")
print(f"Quartile thresholds: Q25={q25:.4f}, Q75={q75:.4f}")

# Create stratification indices
junb_pos_idx = aged_fap_indices[junb_expr >= q75]
junb_neg_idx = aged_fap_indices[junb_expr <= q25]

print(f"JUNB+ (top quartile): {len(junb_pos_idx)} cells, mean JUNB={junb_expr[junb_expr >= q75].mean():.4f}")
print(f"JUNB- (bottom quartile): {len(junb_neg_idx)} cells, mean JUNB={junb_expr[junb_expr <= q25].mean():.4f}")

junb_stratification = {
    "junb_pos_n_cells": int(len(junb_pos_idx)),
    "junb_neg_n_cells": int(len(junb_neg_idx)),
    "junb_pos_mean": float(junb_expr[junb_expr >= q75].mean()),
    "junb_neg_mean": float(junb_expr[junb_expr <= q25].mean()),
    "quartile_threshold": float(q75)
}

# =============================================================================
# STEP 2: Identify SASP ligands enriched in JUNB+ FAPs
# =============================================================================
print("\n[STEP 2] Computing ligand enrichment in JUNB+ vs JUNB- FAPs...")

# Get all genes in the dataset
available_genes = set(fap_adata.var_names)
print(f"Total genes in FAP data: {len(available_genes)}")

# Get ligand indices
ligand_indices = []
ligand_names = []
for ligand in SASP_LIGANDS:
    if ligand in available_genes:
        ligand_indices.append(list(fap_adata.var_names).index(ligand))
        ligand_names.append(ligand)
    else:
        print(f"  WARNING: {ligand} not in FAP data, skipping")

print(f"Found {len(ligand_names)}/{len(SASP_LIGANDS)} ligands in data")

# Extract expression for JUNB+ and JUNB- groups
expr_pos = fap_adata.X[junb_pos_idx, :][:, ligand_indices]
expr_neg = fap_adata.X[junb_neg_idx, :][:, ligand_indices]

if hasattr(expr_pos, 'toarray'):
    expr_pos = expr_pos.toarray()
    expr_neg = expr_neg.toarray()
else:
    expr_pos = np.asarray(expr_pos)
    expr_neg = np.asarray(expr_neg)

# Compute mean expression for each ligand
ligand_results = []

for i, ligand in enumerate(ligand_names):
    mean_pos = float(np.mean(expr_pos[:, i]))
    mean_neg = float(np.mean(expr_neg[:, i]))

    # Log2 fold change with pseudocount
    log2fc = np.log2(mean_pos + 0.1) - np.log2(mean_neg + 0.1)

    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(expr_pos[:, i], expr_neg[:, i], equal_var=False)

    ligand_results.append({
        "gene": ligand,
        "mean_junb_neg": mean_neg,
        "mean_junb_pos": mean_pos,
        "log2FC": float(log2fc),
        "p_value": float(p_val)
    })

# Sort by log2FC descending
ligand_results = sorted(ligand_results, key=lambda x: x['log2FC'], reverse=True)

# Apply BH correction for multiple testing
n_tests = len(ligand_results)
_, p_corrected, _, _ = multipletests(
    [r['p_value'] for r in ligand_results],
    alpha=0.05,
    method='fdr_bh'
)

for i, r in enumerate(ligand_results):
    r['p_value_corrected'] = float(p_corrected[i])
    r['significant'] = r['log2FC'] >= 0.5 and r['p_value_corrected'] < 0.05

print(f"\nTested {n_tests} ligands with BH correction")
print("\nLigands enriched in JUNB+ FAPs (log2FC >= 0.5, p_adj < 0.05):")
enriched = [r for r in ligand_results if r['significant']]
for r in enriched:
    print(f"  {r['gene']}: log2FC={r['log2FC']:.3f}, p_adj={r['p_value_corrected']:.2e}")

# =============================================================================
# STEP 3: Identify MuSC receptors for FAP SASP ligands
# =============================================================================
print("\n[STEP 3] Loading MuSC data and computing receptor expression...")

musc_adata = sc.read_h5ad(MuSC_PATH)
print(f"MuSC data: {musc_adata.n_obs} cells, {musc_adata.n_vars} genes")
print(f"MuSC annotations: {musc_adata.obs['Annotation'].value_counts().to_dict()}")

# Collect all unique receptors needed
all_receptors = set()
for ligand in SASP_LIGANDS:
    if ligand in LIGAND_RECEPTORS:
        all_receptors.update(LIGAND_RECEPTORS[ligand])
all_receptors = list(all_receptors)

print(f"Receptors to check: {sorted(all_receptors)}")

# Check which receptors are in the data
available_musc_genes = set(musc_adata.var_names)
receptors_in_data = [r for r in all_receptors if r in available_musc_genes]
receptors_missing = [r for r in all_receptors if r not in available_musc_genes]

print(f"Receptors found in MuSC data: {receptors_in_data}")
if receptors_missing:
    print(f"WARNING - Receptors not found: {receptors_missing}")

# Compute mean expression by MuSC state
musc_receptor_expression = {}
for state in MuSC_STATES:
    state_mask = musc_adata.obs['Annotation'] == state
    state_adata = musc_adata[state_mask]

    if state_adata.n_obs == 0:
        print(f"WARNING: No cells for state '{state}'")
        musc_receptor_expression[state] = {}
        continue

    state_expr = {}
    for receptor in receptors_in_data:
        expr_vals = state_adata[:, receptor].X
        if hasattr(expr_vals, 'toarray'):
            expr_vals = expr_vals.toarray().flatten()
        else:
            expr_vals = np.asarray(expr_vals).flatten()
        mean_expr = float(np.mean(expr_vals))
        state_expr[receptor] = mean_expr

    musc_receptor_expression[state] = state_expr
    print(f"  {state} (n={state_adata.n_obs}): {len(receptors_in_data)} receptors measured")

# =============================================================================
# STEP 4: Create crosstalk summary
# =============================================================================
print("\n[STEP 4] Building crosstalk summary...")

crosstalk_summary = []

# For each enriched ligand, find MuSC receptors
for ligand_result in ligand_results:
    ligand = ligand_result['gene']

    if ligand_result['significant']:
        # Find receptors for this ligand
        receptors = LIGAND_RECEPTORS.get(ligand, [])

        for receptor in receptors:
            if receptor not in receptors_in_data:
                continue

            # Check which MuSC states express this receptor
            for state in MuSC_STATES:
                if state not in musc_receptor_expression:
                    continue

                receptor_expr = musc_receptor_expression[state].get(receptor, 0)

                if receptor_expr > 0.1:  # Expressed threshold
                    # Determine implication based on known biology
                    if ligand in ["IL6", "IL1B", "TNF"]:
                        implication = "suppresses activation"
                    elif ligand in ["FGF7", "HGF", "PDGFA"]:
                        implication = "promotes activation"
                    elif ligand == "IL1RN":
                        implication = "neutralizes IL1B"
                    else:
                        implication = "context-dependent"

                    crosstalk_summary.append({
                        "ligand": ligand,
                        "receptor": receptor,
                        "musc_state": state,
                        "receptor_expression": receptor_expr,
                        "ligand_log2FC": ligand_result['log2FC'],
                        "implication": implication
                    })

print(f"\nCrosstalk pairs identified: {len(crosstalk_summary)}")

# Print summary table
print("\n" + "=" * 80)
print("CROSSTALK SUMMARY: JUNB+ FAP Ligands -> MuSC Receptors")
print("=" * 80)

# Group by implication
for implication in ["suppresses activation", "promotes activation", "neutralizes IL1B", "context-dependent"]:
    pairs = [p for p in crosstalk_summary if p['implication'] == implication]
    if pairs:
        print(f"\n{implication.upper()}:")
        for p in sorted(pairs, key=lambda x: x['ligand']):
            print(f"  {p['ligand']} -> {p['receptor']} ({p['musc_state']}, expr={p['receptor_expression']:.3f})")

# =============================================================================
# Save results
# =============================================================================
print("\n[Saving results...]")

# Prepare final JSON structure
results_json = {
    "junb_stratification": junb_stratification,
    "ligand_enrichment": [
        {
            "gene": r['gene'],
            "mean_junb_neg": r['mean_junb_neg'],
            "mean_junb_pos": r['mean_junb_pos'],
            "log2FC": r['log2FC'],
            "p_value": r['p_value'],
            "p_value_corrected": r['p_value_corrected'],
            "significant": r['significant']
        }
        for r in ligand_results
    ],
    "musc_receptor_expression": musc_receptor_expression,
    "crosstalk_summary": crosstalk_summary,
    "metadata": {
        "fap_n_cells_total": int(fap_adata.n_obs),
        "aged_fap_n_cells": int(len(aged_fap_indices)),
        "musc_n_cells_total": int(musc_adata.n_obs),
        "n_ligands_tested": len(ligand_results),
        "n_ligands_enriched": len(enriched),
        "n_crosstalk_pairs": len(crosstalk_summary)
    }
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"JSON saved: {OUTPUT_JSON}")

# Save ligand expression as CSV
ligand_df = pd.DataFrame([{
    'gene': r['gene'],
    'mean_JUNB_neg': r['mean_junb_neg'],
    'mean_JUNB_pos': r['mean_junb_pos'],
    'log2FC': r['log2FC'],
    'p_value': r['p_value'],
    'p_value_corrected': r['p_value_corrected'],
    'significant': r['significant']
} for r in ligand_results])

ligand_df.to_csv(OUTPUT_CSV, index=False)
print(f"CSV saved: {OUTPUT_CSV}")

# =============================================================================
# Key findings summary
# =============================================================================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

suppressive_pairs = [p for p in crosstalk_summary if p['implication'] == 'suppresses activation']
promoting_pairs = [p for p in crosstalk_summary if p['implication'] == 'promotes activation']

print(f"\n1. JUNB stratification: {junb_stratification['junb_pos_n_cells']} JUNB+ vs {junb_stratification['junb_neg_n_cells']} JUNB- aged FAPs")
print(f"2. Ligands enriched in JUNB+ FAPs: {len(enriched)}/{len(ligand_results)}")
print(f"3. Crosstalk pairs identified: {len(crosstalk_summary)} total")
print(f"   - Suppressive pairs: {len(suppressive_pairs)}")
print(f"   - Promoting pairs: {len(promoting_pairs)}")

if suppressive_pairs:
    ligands_suppressing = set([p['ligand'] for p in suppressive_pairs])
    print(f"\n4. SUPPRESSIVE LIGANDS (key finding): {sorted(ligands_suppressing)}")
    print("   These are elevated in JUNB+ aged FAPs and signal to MuSCs")

if promoting_pairs:
    ligands_promoting = set([p['ligand'] for p in promoting_pairs])
    print(f"\n5. PROMOTING LIGANDS: {sorted(ligands_promoting)}")
    print("   These are also elevated but would support MuSC function")

print("\n" + "=" * 80)
print("Analysis complete.")
print("=" * 80)
