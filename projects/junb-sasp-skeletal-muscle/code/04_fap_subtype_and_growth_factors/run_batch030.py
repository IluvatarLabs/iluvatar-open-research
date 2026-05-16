#!/usr/bin/env python3
"""
batch_030: FAP Growth Factor Compensation Sufficiency Analysis (U89)

Tests whether aged FAP growth factor compensation (FGF7, HGF upregulation) is
sufficient to maintain MuSC regenerative function.

Three-part sufficiency assessment:
- Part A: FAP growth factor score vs MuSC activation (correlation)
- Part B: Crosstalk axis magnitude vs literature benchmarks
- Part C: Absolute growth factor expression vs young FAP
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import json
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = '/home/yuanz/Documents/GitHub/biomarvin_fibro/data'
OUT_DIR = '/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_030'

print("=" * 80)
print("batch_030: FAP Growth Factor Compensation Sufficiency (U89)")
print("=" * 80)

# ============================================================
# PART A: Check available markers in both atlases
# ============================================================
print("\n--- PART A: Checking available markers ---")

fap_ad = sc.read_h5ad(f'{DATA_DIR}/OMIX004308-02.h5ad')
musc_ad = sc.read_h5ad(f'{DATA_DIR}/MuSC_scsn_RNA.h5ad')

print(f"FAP atlas: {fap_ad.n_obs:,} cells, {fap_ad.obs['sample'].nunique()} donors")
print(f"MuSC atlas: {musc_ad.n_obs:,} cells, {musc_ad.obs['sample'].nunique()} donors")

# Check available MuSC activation markers
ACTIVATION_MARKERS = ['MKI67', 'MYOD1', 'DESMIN', 'MYOG', 'PAX7', 'PCNA', 'CK', 'MYH3', 'ACTA2']
available_activation = [g for g in ACTIVATION_MARKERS if g in musc_ad.var_names]
print(f"\nAvailable MuSC activation markers: {available_activation}")

# Also check inflammatory markers as control
INFLAMMATION_MARKERS = ['IL6', 'CXCL8', 'TNF', 'CCL2', 'IL1B']
available_inflam = [g for g in INFLAMMATION_MARKERS if g in musc_ad.var_names]
print(f"Available MuSC inflammation markers: {available_inflam}")

# Check FAP growth factors
GF_LIGANDS = ['FGF7', 'HGF', 'PDGFA', 'IGF1', 'IGF2', 'FGF2', 'VEGFA']
available_gf = [g for g in GF_LIGANDS if g in fap_ad.var_names]
print(f"Available FAP growth factors: {available_gf}")

# ============================================================
# PART B: Compute per-donor pseudobulk scores
# ============================================================
print("\n--- PART B: Per-donor pseudobulk computation ---")

def pseudobulk_by_donor(adata, genes, donor_col='sample'):
    """Compute per-donor mean expression."""
    available = [g for g in genes if g in adata.var_names]
    if not available:
        return pd.DataFrame()
    X = adata[:, available].X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    df = pd.DataFrame(X, columns=available, index=adata.obs_names)
    df[donor_col] = adata.obs[donor_col].values
    # Handle age columns (age_pop is 'old_pop'/'young_pop' in both FAP and MuSC)
    age_col = 'age_pop' if 'age_pop' in adata.obs.columns else ('age_group' if 'age_group' in adata.obs.columns else None)
    if age_col:
        df['age_group'] = adata.obs[age_col].values
    result = df.groupby(donor_col)[available].mean()
    if age_col and 'age_group' in df.columns:
        age_map = df.groupby(donor_col)['age_group'].first()
        result['age_group'] = age_map
    return result.reset_index()

# FAP growth factors per donor
fap_gf_df = pseudobulk_by_donor(fap_ad, available_gf, 'sample')
fap_gf_df['gf_score'] = fap_gf_df[available_gf].mean(axis=1)
fap_gf_df.rename(columns={'sample': 'donor_id'}, inplace=True)

# FAP senescence markers per donor (control)
SASP_MARKERS = ['JUNB', 'FOS', 'CXCL8', 'IL6', 'CCL2', 'SERPINE1', 'MMP1', 'MMP3']
available_sasp = [g for g in SASP_MARKERS if g in fap_ad.var_names]
fap_sasp_df = pseudobulk_by_donor(fap_ad, available_sasp, 'sample')
fap_sasp_df['sasp_score'] = fap_sasp_df[available_sasp].mean(axis=1)
fap_sasp_df.rename(columns={'sample': 'donor_id'}, inplace=True)

# MuSC activation markers per donor
musc_act_df = pseudobulk_by_donor(musc_ad, available_activation, 'sample')
if available_activation:
    musc_act_df['activation_score'] = musc_act_df[available_activation].mean(axis=1)
musc_act_df.rename(columns={'sample': 'donor_id'}, inplace=True)

# MuSC inflammation markers per donor
musc_inflam_df = pseudobulk_by_donor(musc_ad, available_inflam, 'sample')
if available_inflam:
    musc_inflam_df['inflammation_score'] = musc_inflam_df[available_inflam].mean(axis=1)
musc_inflam_df.rename(columns={'sample': 'donor_id'}, inplace=True)

# MuSC JUNB (control)
if 'JUNB' in musc_ad.var_names:
    musc_junb_df = pseudobulk_by_donor(musc_ad, ['JUNB'], 'sample')
    musc_junb_df.rename(columns={'sample': 'donor_id'}, inplace=True)

print(f"FAP donors: {len(fap_gf_df)}, Age groups: {fap_gf_df['age_group'].value_counts().to_dict()}")
print(f"MuSC donors: {len(musc_act_df)}, Age groups: {musc_act_df['age_group'].value_counts().to_dict()}")

# ============================================================
# PART C: Donor matching between FAP and MuSC atlases
# ============================================================
print("\n--- PART C: Donor matching ---")

# Find overlapping donors
fap_donors = set(fap_gf_df['donor_id'].values)
musc_donors = set(musc_act_df['donor_id'].values)
overlapping = fap_donors & musc_donors
print(f"FAP donors: {len(fap_donors)}")
print(f"MuSC donors: {len(musc_donors)}")
print(f"Overlapping donors: {len(overlapping)}")

if len(overlapping) >= 5:
    # Match by donor_id
    merged = fap_gf_df.merge(fap_sasp_df[['donor_id', 'sasp_score']], on='donor_id', how='left')
    merged = merged.merge(musc_act_df[['donor_id', 'activation_score'] + available_activation],
                          on='donor_id', how='left', suffixes=('_fap', '_musc'))
    merged = merged.merge(musc_inflam_df[['donor_id', 'inflammation_score'] + available_inflam],
                          on='donor_id', how='left')
    if 'JUNB' in musc_ad.var_names:
        merged = merged.merge(musc_junb_df[['donor_id', 'JUNB']], on='donor_id', how='left')

    print(f"\nMatched donors: {len(merged)}")
    print(merged[['donor_id', 'age_group', 'gf_score', 'activation_score']].to_string())

    # Correlations for Part A
    valid = merged.dropna(subset=['gf_score', 'activation_score'])
    print(f"\nDonors with both FAP GF score and MuSC activation score: {len(valid)}")

    if len(valid) >= 5:
        rho_pearson, p_pearson = pearsonr(valid['gf_score'], valid['activation_score'])
        rho_spearman, p_spearman = spearmanr(valid['gf_score'], valid['activation_score'])
        print(f"\nFAP Growth Factor Score vs MuSC Activation Score:")
        print(f"  Pearson rho={rho_pearson:.4f}, p={p_pearson:.4e}")
        print(f"  Spearman rho={rho_spearman:.4f}, p={p_spearman:.4e}")

        # Within-old correlation (more relevant for compensation question)
        old_only = valid[valid['age_group'] == 'old_pop']
        if len(old_only) >= 5:
            rho_old, p_old = pearsonr(old_only['gf_score'], old_only['activation_score'])
            print(f"\nWithin-old correlation (N={len(old_only)}):")
            print(f"  Pearson rho={rho_old:.4f}, p={p_old:.4e}")
        else:
            rho_old, p_old = np.nan, np.nan
            print(f"\nWithin-old correlation: N={len(old_only)} (<5, insufficient)")
    else:
        print("Insufficient matched donors for Part A analysis.")
        rho_pearson = p_pearson = rho_spearman = p_spearman = np.nan
        rho_old = p_old = np.nan
        valid = pd.DataFrame()
else:
    print("Insufficient overlapping donors. Checking if atlases use different sample IDs...")
    print("FAP samples:", sorted(list(fap_donors))[:10])
    print("MuSC samples:", sorted(list(musc_donors))[:10])
    rho_pearson = p_pearson = rho_spearman = p_spearman = np.nan
    rho_old = p_old = np.nan

# ============================================================
# PART D: Null control — FAP growth factors vs MuSC inflammation
# ============================================================
print("\n--- PART D: Null control ---")
if len(overlapping) >= 5:
    valid_ctrl = merged.dropna(subset=['gf_score', 'inflammation_score'])
    if len(valid_ctrl) >= 5:
        rho_ctrl, p_ctrl = pearsonr(valid_ctrl['gf_score'], valid_ctrl['inflammation_score'])
        print(f"FAP GF Score vs MuSC Inflammation (null control):")
        print(f"  Pearson rho={rho_ctrl:.4f}, p={p_ctrl:.4e}")
    else:
        rho_ctrl = p_ctrl = np.nan
else:
    rho_ctrl = p_ctrl = np.nan

# ============================================================
# PART E: Crosstalk axis sufficiency (from batch_024 data)
# ============================================================
print("\n--- PART E: Crosstalk axis sufficiency (literature benchmarks) ---")

# From batch_024 results
crosstalk_results = {
    'HGF->MET': {
        'fap_ligand_d': 1.104,
        'musc_receptor_d': 0.599,
        'crosstalk_score': 0.661,
        'ligand_bh_sig': False,
        'receptor_bh_sig': False
    },
    'FGF7->FGFR1': {
        'fap_ligand_d': 1.290,
        'musc_receptor_d': 0.133,
        'crosstalk_score': 0.171,
        'ligand_bh_sig': False,
        'receptor_bh_sig': False
    },
    'PDGFA->PDGFRA': {
        'fap_ligand_d': 0.444,
        'musc_receptor_d': 0.702,
        'crosstalk_score': 0.312,
        'ligand_bh_sig': False,
        'receptor_bh_sig': False
    }
}

# Literature benchmarks for biological effect size:
# - d > 0.5: large effect (biological relevance)
# - d > 0.8: very large effect (strong biological relevance)
# - Both ligand AND receptor up (d > 0.2): coordinated compensation
# - Crosstalk score > 0.3: biologically meaningful signal

sufficiency_results = {}
for axis, data in crosstalk_results.items():
    ligand_up = data['fap_ligand_d'] > 0.2
    receptor_up = data['musc_receptor_d'] > 0.2
    both_up = ligand_up and receptor_up
    large_crosstalk = data['crosstalk_score'] > 0.3
    coordinated = both_up and large_crosstalk

    sufficiency_results[axis] = {
        'ligand_d': data['fap_ligand_d'],
        'receptor_d': data['musc_receptor_d'],
        'crosstalk_score': data['crosstalk_score'],
        'ligand_up': ligand_up,
        'receptor_up': receptor_up,
        'both_up': both_up,
        'large_crosstalk': large_crosstalk,
        'coordinated': coordinated
    }

    print(f"\n{axis}:")
    print(f"  FAP {axis.split('->')[0]} d={data['fap_ligand_d']:.3f} ({'UP' if ligand_up else 'down/no change'})")
    print(f"  MuSC {axis.split('->')[1]} d={data['musc_receptor_d']:.3f} ({'UP' if receptor_up else 'down/no change'})")
    print(f"  Crosstalk score={data['crosstalk_score']:.3f}")
    print(f"  Coordinated compensation: {'YES' if coordinated else 'NO'}")

# ============================================================
# PART F: Absolute growth factor expression analysis
# ============================================================
print("\n--- PART F: Absolute growth factor expression (aged vs young FAP) ---")

abs_results = {}
for gene in available_gf:
    young_vals = fap_gf_df[fap_gf_df['age_group'] == 'young_pop'][gene].values
    old_vals = fap_gf_df[fap_gf_df['age_group'] == 'old_pop'][gene].values

    if len(young_vals) >= 2 and len(old_vals) >= 2:
        mean_young = np.mean(young_vals)
        mean_old = np.mean(old_vals)
        log2fc = np.log2(mean_old / mean_young) if mean_young > 0 else 0

        # Compute Cohen's d
        pooled_std = np.sqrt((np.var(young_vals) + np.var(old_vals)) / 2)
        cohens_d = (mean_old - mean_young) / pooled_std if pooled_std > 0 else 0

        t_stat, p_val = ttest_ind(old_vals, young_vals)

        # Direction: positive d means OLD > YOUNG (compensation)
        direction = 'COMPENSATED' if cohens_d > 0.2 else ('DECLINED' if cohens_d < -0.2 else 'UNCHANGED')

        abs_results[gene] = {
            'mean_young': mean_young,
            'mean_old': mean_old,
            'log2fc': log2fc,
            'cohens_d': cohens_d,
            'p_value': p_val,
            'n_young': len(young_vals),
            'n_old': len(old_vals),
            'direction': direction
        }

        print(f"\n{gene}:")
        print(f"  Young mean: {mean_young:.4f}, Old mean: {mean_old:.4f}")
        print(f"  log2FC(Old/Young): {log2fc:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f} ({direction})")
        print(f"  p-value: {p_val:.4e}")

# ============================================================
# PART G: Net FAP growth factor score by age
# ============================================================
print("\n--- PART G: Net FAP growth factor score by age ---")

if 'gf_score' in fap_gf_df.columns:
    young_gf = fap_gf_df[fap_gf_df['age_group'] == 'young_pop']['gf_score'].values
    old_gf = fap_gf_df[fap_gf_df['age_group'] == 'old_pop']['gf_score'].values

    pooled_std = np.sqrt((np.var(young_gf) + np.var(old_gf)) / 2)
    net_d = (np.mean(old_gf) - np.mean(young_gf)) / pooled_std if pooled_std > 0 else 0
    t_stat, p_val = ttest_ind(old_gf, young_gf)

    print(f"Net FAP growth factor score:")
    print(f"  Young (N={len(young_gf)}): mean={np.mean(young_gf):.4f} ± {np.std(young_gf):.4f}")
    print(f"  Old (N={len(old_gf)}): mean={np.mean(old_gf):.4f} ± {np.std(old_gf):.4f}")
    print(f"  Cohen's d: {net_d:.4f}")
    print(f"  p-value: {p_val:.4e}")
    print(f"  Interpretation: {'COMPENSATED' if net_d > 0.2 else ('DECLINED' if net_d < -0.2 else 'UNCHANGED')}")

# ============================================================
# PART H: Sufficiency verdict
# ============================================================
print("\n" + "=" * 80)
print("SUFFICIENCY VERDICT")
print("=" * 80)

# Part A verdict
part_a_adequate = not np.isnan(rho_pearson) and abs(rho_pearson) > 0.3 and p_pearson < 0.05
rho_str = f"{rho_pearson:.4f}" if not np.isnan(rho_pearson) else "N/A"
p_str = f"{p_pearson:.4e}" if not np.isnan(p_pearson) else "N/A"
print(f"\nPart A (FAP GF vs MuSC activation): rho={rho_str}, p={p_str}")
print(f"  Adequate: {'YES' if part_a_adequate else 'NO (weak/no correlation)'}")

# Part B verdict
n_coordinated = sum(1 for r in sufficiency_results.values() if r['coordinated'])
part_b_adequate = n_coordinated >= 1
print(f"\nPart B (Crosstalk axis coordination): {n_coordinated}/3 axes coordinated")
print(f"  Adequate: {'YES' if part_b_adequate else 'NO (no coordinated compensation)'}")

# Part C verdict
n_compensated = sum(1 for r in abs_results.values() if r['direction'] == 'COMPENSATED')
part_c_adequate = n_compensated >= 1
print(f"\nPart C (Absolute GF expression): {n_compensated}/{len(abs_results)} genes compensated (d > 0.2)")
print(f"  Adequate: {'YES' if part_c_adequate else 'NO (no growth factor compensation)'}")

# Net verdict
n_adequate = sum([part_a_adequate, part_b_adequate, part_c_adequate])
print(f"\n{'='*40}")
print(f"NET VERDICT: {n_adequate}/3 parts support 'adequate'")
if n_adequate >= 2:
    verdict = "ADEQUATE"
    print("FINDING F090: FAP growth factor compensation is ADEQUATE")
elif n_adequate == 1:
    verdict = "PARTIAL"
    print("FINDING F090: FAP growth factor compensation is PARTIAL")
else:
    verdict = "INADEQUATE"
    print("FINDING F090: FAP growth factor compensation is INADEQUATE")

# ============================================================
# PART I: MuSC-intrinsic compensation (IGF1R, MET receptors up)
# ============================================================
print("\n--- PART I: MuSC-intrinsic compensation ---")

# From batch_024: MuSC receptor data
musc_receptor_data = {
    'IGF1R': {'d': 0.893, 'p': 0.036, 'bh_p': 0.381},
    'MET': {'d': 0.599, 'p': 0.675, 'bh_p': 0.849},
    'PDGFRA': {'d': 0.702, 'p': 0.146, 'bh_p': 0.438},
    'FGFR2': {'d': 0.500, 'p': 0.134, 'bh_p': 0.438},
    'NTRK2': {'d': 0.953, 'p': 0.042, 'bh_p': 0.381}
}

print("\nMuSC receptor upregulation with age:")
for gene, data in musc_receptor_data.items():
    sig = '***' if data['p'] < 0.001 else ('**' if data['p'] < 0.01 else ('*' if data['p'] < 0.05 else ''))
    print(f"  {gene}: d={data['d']:.3f}, p={data['p']:.4e} {sig}")

# ============================================================
# ASSEMBLE RESULTS
# ============================================================
results = {
    'metadata': {
        'batch': 'batch_030',
        'purpose': 'FAP growth factor compensation sufficiency (U89)',
        'date': '2026-04-10',
        'random_seed': RANDOM_SEED,
        'fap_n_donors': int(fap_ad.obs['sample'].nunique()),
        'musc_n_donors': int(musc_ad.n_obs),
        'overlapping_donors': int(len(overlapping)),
        'fap_gf_genes': available_gf,
        'musc_activation_genes': available_activation
    },
    'part_a_correlation': {
        'fap_gf_vs_musc_activation': {
            'pearson_rho': float(rho_pearson) if not np.isnan(rho_pearson) else None,
            'pearson_p': float(p_pearson) if not np.isnan(p_pearson) else None,
            'spearman_rho': float(rho_spearman) if not np.isnan(rho_spearman) else None,
            'spearman_p': float(p_spearman) if not np.isnan(p_spearman) else None,
            'n_matched_donors': int(len(valid)) if len(overlapping) >= 5 else 0,
            'within_old_rho': float(rho_old) if not np.isnan(rho_old) else None,
            'within_old_p': float(p_old) if not np.isnan(p_old) else None,
            'adequate': part_a_adequate
        },
        'fap_gf_vs_musc_inflammation': {
            'pearson_rho': float(rho_ctrl) if not np.isnan(rho_ctrl) else None,
            'pearson_p': float(p_ctrl) if not np.isnan(p_ctrl) else None,
            'interpretation': 'null control - should not correlate'
        }
    },
    'part_b_crosstalk_sufficiency': sufficiency_results,
    'part_c_absolute_expression': abs_results,
    'verdict': {
        'part_a_adequate': part_a_adequate,
        'part_b_adequate': part_b_adequate,
        'part_c_adequate': part_c_adequate,
        'n_adequate': n_adequate,
        'verdict': verdict
    }
}

# Save results
with open(f'{OUT_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUT_DIR}/results.json")
print("\nDone.")
