#!/usr/bin/env python3
"""
batch_010 Experiments 1 & 2: FAP Subtype × Technology Stratified Analysis + JUNB Specificity

WHAT: Tests AP-1 age elevation within FAP subtypes stratified by technology, then tests
      whether JUNB elevation is specific to FAP fibrosis axis vs generic stress response.

WHY:
  - Exp 1: Batch_009 showed CD55+ reversal (d=-0.17 scRNA vs d=+1.31 snRNA) and RUNX2+
    contamination. Must stratify to distinguish biological from technical signal.
  - Exp 2: Critic 2 (GRN) identified AP-1 may be generic stress marker. Need to verify
    whether JUNB elevation is specific or part of coordinated AP-1 response.

SOURCE: batch_009. Design revisions per 3-critic adversarial review (batch_010/brief.md).

PREDICTION:
  - MME+ old > young: d > 0.35 in scRNA (within-technology interpretation)
  - GPC3+ old > young: d > 0.35 in scRNA (validation)
  - JUNB within MME+ scRNA: d > 0.40 (most consistent member)
  - JUNB-collagen rho > FOS-collagen rho → JUNB specificity supported

DECISION RULES:
  - MME+ scRNA d > 0.35 → AP-1 aging effect present in scRNA-clean subtype
  - GPC3+ scRNA d > 0.35 → replication in second subtype
  - JUNB within MME+ scRNA d > 0.40 → JUNB specificity supported
  - JUNB-collagen rho > FOS-collagen rho → JUNB drives FAP fibrosis axis

IF WRONG:
  - MME+ d < 0.15 → AP-1 FAP effect artifact-driven
  - All AP-1 members correlated → generic stress signal, JUNB specificity not supported
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.stats import spearmanr
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = '/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad'
OUTPUT_PATH = '/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_010/results.json'

# Gene sets (from design.yaml)
AP1_MEMBERS = ['FOS', 'FOSL1', 'FOSL2', 'JUN', 'JUNB', 'JUND']
COLLAGEN_GENES = ['COL1A1', 'COL3A1', 'COL6A1', 'COL6A3', 'FN1', 'LOX', 'LOXL1']
SASP_GENES = ['CXCL1', 'CXCL2', 'IL6', 'IL8']  # IL8 may not be in dataset

# Subtype annotations
FAP_SUBTYPES = ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP']

# Bootstrap parameters
N_BOOTSTRAP = 1000
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def cohens_d(x, y):
    """
    Compute Cohen's d for independent samples.

    WHY: Standard effect size measure for comparing two group means.
    SOURCE: Cohen (1988), widely used in biomedical research.

    Returns: Cohen's d and pooled SD
    """
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan

    mean1, mean2 = np.mean(x), np.mean(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan, np.nan

    d = (mean1 - mean2) / pooled_std
    return d, pooled_std


def bootstrap_ci(x, y, statistic_func, n_bootstrap=1000, ci=0.95):
    """
    Compute bootstrap confidence interval for a comparison statistic.

    WHY: Non-parametric CI estimation, robust to non-normality.
    SOURCE: Efron & Tibshirani (1994), standard bootstrap methodology.

    Returns: (lower_bound, upper_bound)
    """
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return [np.nan, np.nan]

    observed_stat = statistic_func(x, y)
    # Handle case where statistic_func returns tuple (stat, other_value)
    if isinstance(observed_stat, tuple):
        observed_stat = observed_stat[0]
    if np.isnan(observed_stat):
        return [np.nan, np.nan]

    # Combine and resample
    combined = np.concatenate([x, y])
    indices = np.arange(len(combined))

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_indices = np.random.choice(indices, size=len(combined), replace=True)
        boot_x = combined[boot_indices[:n1]]
        boot_y = combined[boot_indices[n1:]]

        if len(boot_x) >= 2 and len(boot_y) >= 2:
            boot_stat = statistic_func(boot_x, boot_y)
            # Handle tuple return values
            if isinstance(boot_stat, tuple):
                boot_stat = boot_stat[0]
            if not np.isnan(boot_stat):
                bootstrap_stats.append(boot_stat)

    if len(bootstrap_stats) < 10:
        return [np.nan, np.nan]

    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return [round(lower, 4), round(upper, 4)]


def compute_ap1_composite(adata, gene_list):
    """
    Compute AP-1 composite score as mean of member genes.

    WHY: Composite scores reduce noise compared to single gene analysis.
    SOURCE: Standard practice in transcription factor signature analysis.
    """
    genes_present = [g for g in gene_list if g in adata.var_names]
    if len(genes_present) < len(gene_list):
        missing = set(gene_list) - set(genes_present)
        print(f"  Warning: Missing genes {missing}")

    if len(genes_present) == 0:
        return None

    # Extract expression matrix for these genes
    gene_indices = [list(adata.var_names).index(g) for g in genes_present]
    expression = adata.X[:, gene_indices].toarray() if hasattr(adata.X, 'toarray') else adata.X[:, gene_indices]

    # Mean across genes
    composite = np.mean(expression, axis=1)
    return composite


def run_experiment1_subtype_analysis(adata):
    """
    Experiment 1: FAP Subtype × Technology Stratified Analysis

    For each subtype × technology combination:
      - Compute AP-1 composite score (z-scored per technology)
      - Compute Cohen d (old vs young)
      - Compute 95% CI via bootstrap
      - Compute JUNB individual d

    IMPORTANT INTERPRETATION NOTES (from brief.md):
      - Within-technology comparisons ONLY
      - Do NOT compare magnitudes across technologies
      - Exclude RUNX2+ from snRNA old (contamination: 10.4% vs 0.9% in young)
      - Exclude CD55+ from scRNA (artifact region, d=-0.17 reversal)
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: FAP Subtype × Technology Stratified Analysis")
    print("="*80)

    results = {'scRNA': {}, 'snRNA': {}}

    # Define age groups using age_pop column
    old_mask = adata.obs['age_pop'] == 'old_pop'
    young_mask = adata.obs['age_pop'] == 'young_pop'

    # Define subtypes to analyze per technology
    # scRNA: MME+, CD55+, GPC3+ (exclude RUNX2+ - rare and possibly artifact)
    # snRNA: MME+, GPC3+ (exclude RUNX2+ from old - contamination, exclude CD55+ per design)
    subtypes_by_tech = {
        'scRNA': ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP'],
        'snRNA': ['MME+ FAP', 'CD55+ FAP', 'GPC3+ FAP', 'RUNX2+ FAP']  # Will handle exclusions below
    }

    for tech in ['scRNA', 'snRNA']:
        print(f"\n--- {tech} Analysis ---")

        # Filter to technology
        tech_mask = adata.obs['tech'] == tech
        adata_tech = adata[tech_mask].copy()

        for subtype in subtypes_by_tech[tech]:
            subtype_label = subtype.replace('+ FAP', '_plus').replace('+ ', '_')
            print(f"\n  Subtype: {subtype}")

            # Filter to subtype
            subtype_mask = adata_tech.obs['Annotation'] == subtype
            adata_sub = adata_tech[subtype_mask].copy()

            # Define age groups
            old_sub_mask = adata_sub.obs['age_pop'] == 'old_pop'
            young_sub_mask = adata_sub.obs['age_pop'] == 'young_pop'

            n_old = old_sub_mask.sum()
            n_young = young_sub_mask.sum()

            print(f"    n_old: {n_old}, n_young: {n_young}")

            # Apply exclusion rules
            exclude = False
            if tech == 'snRNA' and subtype == 'RUNX2+ FAP':
                print(f"    [EXCLUDED: RUNX2+ from snRNA old - contamination artifact]")
                exclude = True
            elif tech == 'scRNA' and subtype == 'CD55+ FAP':
                print(f"    [EXCLUDED: CD55+ from scRNA - artifact region per design]")
                exclude = True

            if exclude or n_old < 10 or n_young < 10:
                results[tech][subtype_label] = {
                    'n_old': int(n_old),
                    'n_young': int(n_young),
                    'ap1_d': None,
                    'ap1_d_ci95': None,
                    'junb_d': None,
                    'excluded': True if (n_old >= 10 and n_young >= 10) else 'insufficient_cells'
                }
                continue

            # Compute AP-1 composite score (z-scored per technology - already done in data)
            ap1_score = compute_ap1_composite(adata_sub, AP1_MEMBERS)

            old_values = ap1_score[old_sub_mask.values]
            young_values = ap1_score[young_sub_mask.values]

            # Cohen d for AP-1 composite
            ap1_d, _ = cohens_d(old_values, young_values)

            # Bootstrap 95% CI
            ap1_ci = bootstrap_ci(old_values, young_values, cohens_d, N_BOOTSTRAP)

            print(f"    AP-1 composite d: {ap1_d:.4f} (95% CI: {ap1_ci})")

            # JUNB individual d
            if 'JUNB' in adata_sub.var_names:
                junb_idx = list(adata_sub.var_names).index('JUNB')
                if hasattr(adata_sub.X, 'toarray'):
                    junb_expr = adata_sub.X[:, junb_idx].toarray().flatten()
                else:
                    junb_expr = adata_sub.X[:, junb_idx].flatten()

                old_junb = junb_expr[old_sub_mask.values]
                young_junb = junb_expr[young_sub_mask.values]

                junb_d, _ = cohens_d(old_junb, young_junb)
                print(f"    JUNB d: {junb_d:.4f}")
            else:
                junb_d = None
                print(f"    JUNB d: [NOT FOUND]")

            results[tech][subtype_label] = {
                'n_old': int(n_old),
                'n_young': int(n_young),
                'ap1_d': round(ap1_d, 4) if ap1_d is not None else None,
                'ap1_d_ci95': ap1_ci,
                'junb_d': round(junb_d, 4) if junb_d is not None else None
            }

    return results


def run_experiment2_junb_specificity(adata):
    """
    Experiment 2: JUNB Specificity Analysis

    Within scRNA old MME+ cells (cleanest scRNA subtype):
      - Extract AP-1 member expression (z-scored)
      - Extract collagen composite score
      - Extract SASP composite score (where available)
      - Compute pairwise Spearman correlations among AP-1 members
      - Compute AP-1 member correlations vs collagen and SASP composites
      - Compare: Is JUNB-collagen rho > FOS-collagen rho?

    DECISION RULES (from brief.md):
      - JUNB-collagen rho > FOS-collagen rho → JUNB specificity supported
      - JUNB-SASP rho > FOS-SASP rho → JUNB drives senescence/fibrosis axis
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: JUNB Specificity Analysis")
    print("="*80)

    results = {}

    # Filter to: scRNA, old, MME+ FAP
    sc_mask = adata.obs['tech'] == 'scRNA'
    old_mask = adata.obs['age_pop'] == 'old_pop'
    mme_mask = adata.obs['Annotation'] == 'MME+ FAP'

    subset_mask = sc_mask & old_mask & mme_mask
    adata_sub = adata[subset_mask].copy()

    n_cells = subset_mask.sum()
    print(f"\n  scRNA old MME+ cells: {n_cells}")

    if n_cells < 50:
        print("  [INSUFFICIENT CELLS - need at least 50 for meaningful correlations]")
        return None

    results['within_mme_plus_scRNA_old'] = {'n': int(n_cells)}

    # Extract AP-1 member expression
    ap1_expr = {}
    for gene in AP1_MEMBERS:
        if gene in adata_sub.var_names:
            idx = list(adata_sub.var_names).index(gene)
            if hasattr(adata_sub.X, 'toarray'):
                expr = adata_sub.X[:, idx].toarray().flatten()
            else:
                expr = adata_sub.X[:, idx].flatten()
            ap1_expr[gene] = expr
            print(f"    {gene}: mean={np.mean(expr):.3f}, std={np.std(expr):.3f}")

    # Compute AP-1 composite
    ap1_indices = [list(adata_sub.var_names).index(g) for g in AP1_MEMBERS if g in adata_sub.var_names]
    if ap1_indices:
        if hasattr(adata_sub.X, 'toarray'):
            ap1_mat = adata_sub.X[:, ap1_indices].toarray()
        else:
            ap1_mat = adata_sub.X[:, ap1_indices]
        ap1_composite = np.mean(ap1_mat, axis=1)
    else:
        ap1_composite = None

    # Compute collagen composite
    coll_indices = [list(adata_sub.var_names).index(g) for g in COLLAGEN_GENES if g in adata_sub.var_names]
    if coll_indices:
        if hasattr(adata_sub.X, 'toarray'):
            coll_mat = adata_sub.X[:, coll_indices].toarray()
        else:
            coll_mat = adata_sub.X[:, coll_indices]
        collagen_composite = np.mean(coll_mat, axis=1)
        print(f"\n  Collagen composite: mean={np.mean(collagen_composite):.3f}")
    else:
        collagen_composite = None

    # Compute SASP composite (only available genes)
    sasp_available = [g for g in SASP_GENES if g in adata_sub.var_names]
    if sasp_available:
        sasp_indices = [list(adata_sub.var_names).index(g) for g in sasp_available]
        if hasattr(adata_sub.X, 'toarray'):
            sasp_mat = adata_sub.X[:, sasp_indices].toarray()
        else:
            sasp_mat = adata_sub.X[:, sasp_indices]
        sasp_composite = np.mean(sasp_mat, axis=1)
        print(f"  SASP composite ({', '.join(sasp_available)}): mean={np.mean(sasp_composite):.3f}")
    else:
        sasp_composite = None
        print("  SASP composite: [NO SASP GENES AVAILABLE]")

    # 1. Pairwise AP-1 member correlations (15 pairs)
    print("\n  AP-1 member pairwise correlations (Spearman rho):")
    ap1_correlations = {}
    genes_list = list(ap1_expr.keys())
    for i in range(len(genes_list)):
        for j in range(i+1, len(genes_list)):
            g1, g2 = genes_list[i], genes_list[j]
            rho, pval = spearmanr(ap1_expr[g1], ap1_expr[g2])
            key = f"{g1}_{g2}"
            ap1_correlations[key] = round(rho, 4)
            print(f"    {g1} vs {g2}: rho={rho:.4f}, p={pval:.4f}")

    results['ap1_member_correlations'] = ap1_correlations

    # 2. AP-1 member correlations vs collagen composite
    print("\n  AP-1 member correlations vs collagen composite:")
    ap1_collagen = {}
    for gene in genes_list:
        rho, pval = spearmanr(ap1_expr[gene], collagen_composite)
        ap1_collagen[gene] = round(rho, 4)
        print(f"    {gene} vs collagen: rho={rho:.4f}, p={pval:.4f}")

    results['ap1_vs_collagen'] = ap1_collagen

    # 3. AP-1 member correlations vs SASP composite
    if sasp_composite is not None:
        print("\n  AP-1 member correlations vs SASP composite:")
        ap1_sasp = {}
        for gene in genes_list:
            rho, pval = spearmanr(ap1_expr[gene], sasp_composite)
            ap1_sasp[gene] = round(rho, 4)
            print(f"    {gene} vs SASP: rho={rho:.4f}, p={pval:.4f}")

        results['ap1_vs_sasp'] = ap1_sasp
    else:
        results['ap1_vs_sasp'] = None
        print("\n  SASP correlations: [NOT COMPUTED - no SASP genes available]")

    # 4. Specificity tests
    print("\n  SPECIFICITY TESTS:")

    # JUNB-collagen vs FOS-collagen
    if 'JUNB' in ap1_collagen and 'FOS' in ap1_collagen:
        junb_collagen = ap1_collagen['JUNB']
        fos_collagen = ap1_collagen['FOS']
        junb_specificity = junb_collagen > fos_collagen
        print(f"    JUNB-collagen rho ({junb_collagen:.4f}) vs FOS-collagen rho ({fos_collagen:.4f})")
        print(f"    → JUNB > FOS: {junb_specificity}")
        results['specificity_test'] = {
            'junb_collagen_rho': junb_collagen,
            'fos_collagen_rho': fos_collagen,
            'junb_gt_fos_collagen': junb_specificity
        }

    # JUNB-SASP vs FOS-SASP
    if sasp_composite is not None and 'JUNB' in ap1_expr and 'FOS' in ap1_expr:
        junb_sasp_rho = ap1_sasp.get('JUNB')
        fos_sasp_rho = ap1_sasp.get('FOS')
        if junb_sasp_rho is not None and fos_sasp_rho is not None:
            sasp_specificity = junb_sasp_rho > fos_sasp_rho
            print(f"    JUNB-SASP rho ({junb_sasp_rho:.4f}) vs FOS-SASP rho ({fos_sasp_rho:.4f})")
            print(f"    → JUNB > FOS: {sasp_specificity}")
            results['specificity_test']['junb_sasp_rho'] = junb_sasp_rho
            results['specificity_test']['fos_sasp_rho'] = fos_sasp_rho
            results['specificity_test']['junb_gt_fos_sasp'] = sasp_specificity
        else:
            results['specificity_test']['junb_gt_fos_sasp'] = None
    else:
        results['specificity_test']['junb_gt_fos_sasp'] = None
        print("    SASP specificity: [NOT COMPUTED - no SASP genes]")

    return results


def main():
    """Run both experiments and save results."""
    print("="*80)
    print("BATCH_010: Experiments 1 & 2")
    print("FAP Subtype × Technology Stratified Analysis + JUNB Specificity")
    print("="*80)

    # Environment logging
    import platform
    import sys
    print(f"\nEnvironment:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Python: {sys.version}")
    print(f"  NumPy: {np.__version__}")
    print(f"  Scanpy: {sc.__version__}")
    print(f"  Data: {DATA_PATH}")

    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"  Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Verify key columns exist
    assert 'tech' in adata.obs.columns, "Missing 'tech' column"
    assert 'age_pop' in adata.obs.columns, "Missing 'age_pop' column"
    assert 'Annotation' in adata.obs.columns, "Missing 'Annotation' column"

    print(f"\n  Tech distribution: {adata.obs['tech'].value_counts().to_dict()}")
    print(f"  Age population: {adata.obs['age_pop'].value_counts().to_dict()}")
    print(f"  Annotations: {adata.obs['Annotation'].value_counts().to_dict()}")

    # Run Experiment 1
    exp1_results = run_experiment1_subtype_analysis(adata)

    # Run Experiment 2
    exp2_results = run_experiment2_junb_specificity(adata)

    # Compile final results - convert numpy types to native Python types
    def convert_to_serializable(obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    final_results = {
        'experiment1_subtype_analysis': convert_to_serializable(exp1_results),
        'experiment2_junb_specificity': convert_to_serializable(exp2_results)
    }

    # Save results
    print(f"\nSaving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(final_results, f, indent=2)
    print("Done.")

    # Summary interpretation
    print("\n" + "="*80)
    print("SUMMARY & DECISION RULES")
    print("="*80)

    print("\n--- Experiment 1: Subtype × Technology ---")

    # MME+ scRNA
    mme_sc = exp1_results.get('scRNA', {}).get('MME_plus', {})
    if mme_sc.get('ap1_d') is not None:
        mme_d = mme_sc['ap1_d']
        mme_ci = mme_sc.get('ap1_d_ci95', [None, None])
        print(f"  MME+ scRNA: d={mme_d:.4f} (95% CI: {mme_ci})")
        if mme_d > 0.35:
            print(f"    → POSITIVE: d > 0.35 threshold met")
        elif mme_d > 0.15:
            print(f"    → AMBIGUOUS: 0.15 < d < 0.35 (weak signal)")
        else:
            print(f"    → NEGATIVE: d < 0.15 (effect may be artifact-driven)")

    # GPC3+ scRNA
    gpc3_sc = exp1_results.get('scRNA', {}).get('GPC3_plus', {})
    if gpc3_sc.get('ap1_d') is not None:
        gpc3_d = gpc3_sc['ap1_d']
        print(f"  GPC3+ scRNA: d={gpc3_d:.4f}")
        if gpc3_d > 0.35:
            print(f"    → REPLICATION: Effect extends to second subtype")

    # JUNB within MME+ scRNA
    if mme_sc.get('junb_d') is not None:
        junb_d = mme_sc['junb_d']
        print(f"  JUNB within MME+ scRNA: d={junb_d:.4f}")
        if junb_d > 0.40:
            print(f"    → JUNB specificity supported: d > 0.40 threshold met")

    print("\n--- Experiment 2: JUNB Specificity ---")

    if exp2_results:
        spec = exp2_results.get('specificity_test', {})
        if spec.get('junb_gt_fos_collagen') is not None:
            print(f"  JUNB-collagen rho: {spec.get('junb_collagen_rho', 'N/A')}")
            print(f"  FOS-collagen rho: {spec.get('fos_collagen_rho', 'N/A')}")
            if spec['junb_gt_fos_collagen']:
                print(f"    → JUNB specificity SUPPORTED: rho_JUNB > rho_FOS")
            else:
                print(f"    → JUNB specificity NOT supported: rho_JUNB ≤ rho_FOS")

        if spec.get('junb_gt_fos_sasp') is not None:
            print(f"  JUNB-SASP rho: {spec.get('junb_sasp_rho', 'N/A')}")
            print(f"  FOS-SASP rho: {spec.get('fos_sasp_rho', 'N/A')}")
            if spec['junb_gt_fos_sasp']:
                print(f"    → JUNB-SASP link SUPPORTED: rho_JUNB > rho_FOS")
            else:
                print(f"    → JUNB-SASP link NOT supported: rho_JUNB ≤ rho_FOS")
    else:
        print("  [Not computed - insufficient cells]")

    print("\n" + "="*80)

    return final_results


if __name__ == '__main__':
    results = main()
