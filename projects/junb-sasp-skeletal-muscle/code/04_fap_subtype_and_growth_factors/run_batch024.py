#!/usr/bin/env python3
"""
batch_024: MuSC-FAP Crosstalk Analysis

Tests whether aged FAPs shift toward inflammatory + away from regenerative
ligand secretion, and whether MuSCs become more responsive to inflammatory signals.

Data:
- FAP atlas: OMIX004308-02.h5ad (40,389 cells, 22 donors)
- MuSC atlas: MuSC_scsn_RNA.h5ad (9,559 cells)
"""

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import json
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SASP_LIGANDS = [
    'TNF', 'IL6', 'IL1B', 'IL1A', 'CXCL8', 'CCL2', 'CXCL1', 'CXCL2',
    'CXCL3', 'CXCL6', 'SERPINE1', 'MMP1', 'MMP3', 'PLAU', 'VEGFA',
    'CCL5', 'CXCL16', 'ICAM1', 'VCAM1'
]

REGENERATIVE_LIGANDS = [
    'PDGFA', 'PDGFB', 'IGF1', 'IGF2', 'FGF7', 'HGF', 'GDNF', 'BDNF', 'FGF2'
]

FAP_RECEPTORS = [
    'TNFRSF1A', 'TNFRSF1B', 'IL6ST', 'IL6R', 'IL6RBP', 'CCR2',
    'PDGFRA', 'PDGFRB', 'IGF1R', 'IGF2R', 'FGFR1', 'FGFR2', 'FGFR3',
    'FGFR4', 'MET', 'RET', 'NTRK1', 'NTRK2', 'NTRK3'
]

print("=" * 80)
print("batch_024: MuSC-FAP Crosstalk Analysis")
print("=" * 80)


def compute_donor_means(adata, genes, donor_col='sample', obs_col=None):
    """Compute per-donor mean expression for specified genes."""
    available_genes = [g for g in genes if g in adata.var_names]
    if not available_genes:
        return pd.DataFrame()

    subset = adata[:, available_genes].copy()

    # Get expression matrix as DataFrame
    expr_df = pd.DataFrame(
        subset.X.toarray() if hasattr(subset.X, 'toarray') else subset.X,
        columns=available_genes,
        index=subset.obs_names
    )
    expr_df[donor_col] = subset.obs[donor_col].values
    if obs_col:
        expr_df['age_group'] = subset.obs[obs_col].values

    donor_means = expr_df.groupby(donor_col)[available_genes].mean()
    if obs_col:
        age_map = expr_df.groupby(donor_col)['age_group'].first()
        donor_means['age_group'] = age_map

    return donor_means.reset_index()


def compute_age_effect(donor_means, gene, age_col='age_group'):
    """Compute Cohen's d and Mann-Whitney p for a gene by age group."""
    young = donor_means[donor_means[age_col] == 'young'][gene].values
    old = donor_means[donor_means[age_col] == 'old'][gene].values

    if len(young) < 2 or len(old) < 2:
        return None

    mean_y, mean_o = np.mean(young), np.mean(old)
    var_y, var_o = np.var(young, ddof=1), np.var(old, ddof=1)
    n_y, n_o = len(young), len(old)
    std_pooled = np.sqrt(((n_y - 1) * var_y + (n_o - 1) * var_o) / (n_y + n_o - 2))
    cohens_d = (mean_o - mean_y) / std_pooled if std_pooled > 0 else 0.0

    if len(young) >= 3 and len(old) >= 3:
        _, pval = mannwhitneyu(old, young, alternative='two-sided')
    else:
        pval = 1.0

    return {
        'mean_young': float(mean_y),
        'mean_old': float(mean_o),
        'log2fc': float(mean_o - mean_y),
        'cohens_d': float(cohens_d),
        'p_value': float(pval),
        'n_young_donors': int(n_y),
        'n_old_donors': int(n_o),
    }


# =============================================================================
# PART A: FAP Aging Secretome
# =============================================================================
print("\n## PART A: FAP Aging Secretome")
print("-" * 40)

fap_ad = sc.read_h5ad('data/OMIX004308-02.h5ad')
print(f"FAP atlas: {fap_ad.shape[0]} cells, {fap_ad.shape[1]} genes")
print(f"Annotations: {fap_ad.obs['Annotation'].value_counts().to_dict()}")

# Define age groups
if 'age_pop' in fap_ad.obs.columns:
    fap_ad.obs['age_group'] = fap_ad.obs['age_pop'].map({'young_pop': 'young', 'old_pop': 'old'})
elif 'Age_group' in fap_ad.obs.columns:
    fap_ad.obs['age_group'] = fap_ad.obs['Age_group'].map({'Young': 'young', 'Old': 'old'})
elif 'age' in fap_ad.obs.columns:
    fap_ad.obs['age_group'] = fap_ad.obs['age'].apply(lambda x: 'old' if float(x) >= 65 else 'young')
else:
    print("WARNING: No age column found, using all cells as 'old'")
    fap_ad.obs['age_group'] = 'old'

print(f"Age group distribution: {fap_ad.obs['age_group'].value_counts().to_dict()}")

# Filter to FAP cells
fap_mask = ~fap_ad.obs['Annotation'].isin(['Tenocyte'])
fap_cells = fap_ad[fap_mask].copy()
print(f"FAP cells: {fap_cells.shape[0]} cells")

donor_col = 'sample'
print(f"Donor column: {donor_col}")
print(f"N donors: {fap_cells.obs[donor_col].nunique()}")

all_ligands = SASP_LIGANDS + REGENERATIVE_LIGANDS
available_ligands = [g for g in all_ligands if g in fap_ad.var_names]
print(f"Available ligands: {len(available_ligands)}/{len(all_ligands)}")

# Donor-level means
fap_donor_means = compute_donor_means(fap_cells, available_ligands, donor_col, 'age_group')
print(f"Donor-level df: {fap_donor_means.shape}")
print(f"Young donors: {(fap_donor_means['age_group']=='young').sum()}, "
      f"Old donors: {(fap_donor_means['age_group']=='old').sum()}")

# Compute age effects
fap_ligand_results = {}
for ligand in available_ligands:
    res = compute_age_effect(fap_donor_means, ligand)
    if res:
        fap_ligand_results[ligand] = res

# BH correction
if fap_ligand_results:
    pvals = [v['p_value'] for v in fap_ligand_results.values()]
    _, bh_pvals, _, _ = multipletests(pvals, method='fdr_bh')
    for i, ligand in enumerate(fap_ligand_results):
        fap_ligand_results[ligand]['bh_p_value'] = float(bh_pvals[i])
        fap_ligand_results[ligand]['bh_significant'] = bool(bh_pvals[i] < 0.05)

sorted_ligands = sorted(fap_ligand_results.items(),
                        key=lambda x: abs(x[1]['cohens_d']), reverse=True)

print(f"\nTop FAP ligand age effects (|d| > 0.20):")
print(f"{'Ligand':<12} {'Log2FC':>8} {'d':>8} {'p':>10} {'BH':>10} {'Direction':<15}")
print("-" * 70)
for ligand, res in sorted_ligands:
    if abs(res['cohens_d']) > 0.20:
        direction = "↑ OLD" if res['cohens_d'] > 0 else "↓ OLD"
        sig = "*" if res['bh_significant'] else ""
        print(f"{ligand:<12} {res['log2fc']:>8.3f} {res['cohens_d']:>8.3f} "
              f"{res['p_value']:>10.2e} {res['bh_p_value']:>10.2e} {direction:<15}{sig}")


# =============================================================================
# PART B: MuSC Receptor Expression by Age
# =============================================================================
print("\n## PART B: MuSC Receptor Expression by Age")
print("-" * 40)

musc_ad = sc.read_h5ad('data/MuSC_scsn_RNA.h5ad')
print(f"MuSC atlas: {musc_ad.shape[0]} cells, {musc_ad.shape[1]} genes")
print(f"Annotations: {musc_ad.obs['Annotation'].value_counts().to_dict()}")
print(f"Available obs: {list(musc_ad.obs.columns)}")

# Infer age groups from sample naming
if 'age_pop' in musc_ad.obs.columns:
    musc_ad.obs['age_group'] = musc_ad.obs['age_pop'].map({'young_pop': 'young', 'old_pop': 'old'})
elif 'Age_group' in musc_ad.obs.columns:
    musc_ad.obs['age_group'] = musc_ad.obs['Age_group'].map({'Young': 'young', 'Old': 'old'})
elif 'age' in musc_ad.obs.columns:
    musc_ad.obs['age_group'] = musc_ad.obs['age'].apply(lambda x: 'old' if float(x) >= 65 else 'young')
else:
    # Infer from sample: YM=young male, YF=young female, OM=old male, OF=old female
    def infer_age(s):
        s = str(s)
        if s.startswith('Y'):
            return 'young'
        elif s.startswith('O') or s.startswith('P'):
            return 'old'
        return 'unknown'

    musc_ad.obs['age_group'] = musc_ad.obs['sample'].apply(infer_age)

print(f"Age group distribution: {musc_ad.obs['age_group'].value_counts().to_dict()}")
n_unknown = (musc_ad.obs['age_group'] == 'unknown').sum()
if n_unknown > 0:
    print(f"WARNING: {n_unknown} cells with unknown age group")

# Donor-level means
available_receptors = [r for r in FAP_RECEPTORS if r in musc_ad.var_names]
print(f"Available receptors: {len(available_receptors)}/{len(FAP_RECEPTORS)}")

musc_donor_means = compute_donor_means(musc_ad, available_receptors, 'sample', 'age_group')
print(f"MuSC donor-level df: {musc_donor_means.shape}")
print(f"Young donors: {(musc_donor_means['age_group']=='young').sum()}, "
      f"Old donors: {(musc_donor_means['age_group']=='old').sum()}")

musc_receptor_results = {}
for receptor in available_receptors:
    res = compute_age_effect(musc_donor_means, receptor)
    if res:
        musc_receptor_results[receptor] = res

# BH correction
if musc_receptor_results:
    pvals = [v['p_value'] for v in musc_receptor_results.values()]
    _, bh_pvals, _, _ = multipletests(pvals, method='fdr_bh')
    for i, receptor in enumerate(musc_receptor_results):
        musc_receptor_results[receptor]['bh_p_value'] = float(bh_pvals[i])
        musc_receptor_results[receptor]['bh_significant'] = bool(bh_pvals[i] < 0.05)

sorted_receptors = sorted(musc_receptor_results.items(),
                          key=lambda x: abs(x[1]['cohens_d']), reverse=True)

print(f"\nTop MuSC receptor age effects (|d| > 0.20):")
print(f"{'Receptor':<12} {'Mean Young':>12} {'Mean Old':>12} {'d':>8} {'p':>10} {'Direction':<15}")
print("-" * 75)
for receptor, res in sorted_receptors:
    if abs(res['cohens_d']) > 0.20:
        direction = "↑ OLD" if res['cohens_d'] > 0 else "↓ OLD"
        sig = "*" if res.get('bh_significant', False) else ""
        print(f"{receptor:<12} {res['mean_young']:>12.4f} {res['mean_old']:>12.4f} "
              f"{res['cohens_d']:>8.3f} {res['p_value']:>10.2e} {direction:<15}{sig}")


# =============================================================================
# PART C: FAP→MuSC Crosstalk Scores
# =============================================================================
print("\n## PART C: FAP→MuSC Crosstalk Scores")
print("-" * 40)

CROSSTALK_AXES = {
    'TNF->TNFRSF1A': ('TNF', 'TNFRSF1A'),
    'IL6->IL6ST': ('IL6', 'IL6ST'),
    'CCL2->CCR2': ('CCL2', 'CCR2'),
    'PDGFA->PDGFRA': ('PDGFA', 'PDGFRA'),
    'IGF1->IGF1R': ('IGF1', 'IGF1R'),
    'FGF7->FGFR1': ('FGF7', 'FGFR1'),
    'HGF->MET': ('HGF', 'MET'),
    'VEGFA->FGFR1': ('VEGFA', 'FGFR1'),
    'VEGFA->FGFR2': ('VEGFA', 'FGFR2'),
    'CXCL8->CXCR1': ('CXCL8', 'CXCR1'),
    'CXCL8->CXCR2': ('CXCL8', 'CXCR2'),
}

crosstalk_results = {}
for axis_name, (ligand, receptor) in CROSSTALK_AXES.items():
    ligand_data = fap_ligand_results.get(ligand, {})
    receptor_data = musc_receptor_results.get(receptor, {})

    if not ligand_data:
        continue

    ligand_d = ligand_data['cohens_d']
    receptor_d = receptor_data.get('cohens_d', 0.0) if receptor_data else 0.0
    crosstalk_score = ligand_d * receptor_d

    if ligand_d > 0.2 and receptor_d > 0.2:
        net_direction = "INFLAMMATORY BOOST"
    elif ligand_d < -0.2 and receptor_d < -0.2:
        net_direction = "REGENERATIVE LOSS"
    elif ligand_d > 0.2 and receptor_d < -0.2:
        net_direction = "INFLAM + LOST RESPONSE"
    elif ligand_d < -0.2 and receptor_d > 0.2:
        net_direction = "LOST SIGNAL + SENSITIZED"
    elif abs(ligand_d) > 0.2:
        net_direction = f"LIGAND: {'UP' if ligand_d > 0 else 'DOWN'}"
    elif abs(receptor_d) > 0.2:
        net_direction = f"RECEPTOR: {'UP' if receptor_d > 0 else 'DOWN'}"
    else:
        net_direction = "MINIMAL CHANGE"

    crosstalk_results[axis_name] = {
        'ligand': ligand,
        'receptor': receptor,
        'ligand_d': float(ligand_d),
        'receptor_d': float(receptor_d),
        'ligand_log2fc': float(ligand_data.get('log2fc', 0)),
        'receptor_log2fc': float(receptor_data.get('mean_old', 0) - receptor_data.get('mean_young', 0)) if receptor_data else 0.0,
        'crosstalk_score': float(crosstalk_score),
        'net_direction': net_direction,
        'ligand_bh_sig': ligand_data.get('bh_significant', False),
        'receptor_bh_sig': receptor_data.get('bh_significant', False) if receptor_data else False,
    }

print(f"\n{'Axis':<20} {'Lig d':>8} {'Rec d':>8} {'Score':>8} {'Direction':<30}")
print("-" * 80)
for axis, res in sorted(crosstalk_results.items(), key=lambda x: abs(x[1]['crosstalk_score']), reverse=True):
    ligand_sig = "*" if res['ligand_bh_sig'] else ""
    receptor_sig = "*" if res['receptor_bh_sig'] else ""
    print(f"{axis:<20} {res['ligand_d']:>7.3f}{ligand_sig} {res['receptor_d']:>7.3f}{receptor_sig} "
          f"{res['crosstalk_score']:>8.3f} {res['net_direction']:<30}")


# =============================================================================
# PART D: FAP Secretome Profile Summary
# =============================================================================
print("\n## PART D: FAP Aging Profile")
print("-" * 40)

inflammatory_ligands = ['TNF', 'IL6', 'IL1B', 'CXCL8', 'CCL2', 'CXCL1', 'CXCL2', 'SERPINE1', 'MMP1']
regenerative_ligands_list = ['PDGFA', 'FGF7', 'HGF', 'IGF1', 'IGF2', 'VEGFA']

infl_d = [fap_ligand_results.get(l, {}).get('cohens_d', 0) for l in inflammatory_ligands if l in fap_ligand_results]
regen_d = [fap_ligand_results.get(l, {}).get('cohens_d', 0) for l in regenerative_ligands_list if l in fap_ligand_results]

infl_gain = np.mean(infl_d) if infl_d else 0.0
regen_loss = np.mean(regen_d) if regen_d else 0.0
net_score = infl_gain - regen_loss

print(f"Inflammatory gain (mean d across {len(infl_d)} ligands): {infl_gain:.3f}")
print(f"Regenerative decline (mean d across {len(regen_d)} ligands): {regen_loss:.3f}")
print(f"Net aging score (infl. gain - regen. decline): {net_score:.3f}")

# Count significant axes
sig_infl = sum(1 for l in inflammatory_ligands
               if l in fap_ligand_results and fap_ligand_results[l]['bh_significant'] and fap_ligand_results[l]['cohens_d'] > 0.2)
sig_regen = sum(1 for l in regenerative_ligands_list
               if l in fap_ligand_results and fap_ligand_results[l]['bh_significant'] and fap_ligand_results[l]['cohens_d'] < -0.2)

print(f"Significant inflammatory axes (BH sig, d>0.2): {sig_infl}/{len(inflammatory_ligands)}")
print(f"Significant regenerative decline axes (BH sig, d<-0.2): {sig_regen}/{len(regenerative_ligands_list)}")

if net_score > 0.1:
    interpretation = "Pro-inflammatory FAP secretome shift"
elif net_score < -0.1:
    interpretation = "Regenerative decline FAP secretome shift"
else:
    interpretation = "Mixed/balanced FAP secretome shift"
print(f"Interpretation: {interpretation}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nSignificant FAP secretome changes (BH p<0.05, |d|>0.20):")
sig_ligands = [(l, r) for l, r in fap_ligand_results.items()
               if r['bh_significant'] and abs(r['cohens_d']) > 0.20]
for ligand, res in sorted(sig_ligands, key=lambda x: abs(x[1]['cohens_d']), reverse=True):
    direction = "↑" if res['cohens_d'] > 0 else "↓"
    print(f"  {direction} {ligand}: d={res['cohens_d']:.3f}, log2FC={res['log2fc']:.3f}, BHp={res['bh_p_value']:.2e}")

print("\nSignificant MuSC receptor changes (BH p<0.05, |d|>0.20):")
sig_receptors = [(r, res) for r, res in musc_receptor_results.items()
                 if res.get('bh_significant', False) and abs(res['cohens_d']) > 0.20]
for receptor, res in sorted(sig_receptors, key=lambda x: abs(x[1]['cohens_d']), reverse=True):
    direction = "↑" if res['cohens_d'] > 0 else "↓"
    print(f"  {direction} {receptor}: d={res['cohens_d']:.3f}, BHp={res['bh_p_value']:.2e}")

print(f"\nNet FAP aging score: {net_score:.3f}")
print(f"Interpretation: {interpretation}")

results = {
    'metadata': {
        'batch': 'batch_024',
        'date': '2026-04-10',
        'purpose': 'MuSC-FAP crosstalk analysis',
        'random_seed': RANDOM_SEED,
        'fap_n_cells': int(fap_cells.shape[0]),
        'fap_n_donors': int(fap_cells.obs[donor_col].nunique()),
        'musc_n_cells': int(musc_ad.shape[0]),
        'musc_n_donors': int(musc_ad.obs['sample'].nunique()),
        'young_fap_donors': int((fap_donor_means['age_group']=='young').sum()),
        'old_fap_donors': int((fap_donor_means['age_group']=='old').sum()),
        'young_musc_donors': int((musc_donor_means['age_group']=='young').sum()),
        'old_musc_donors': int((musc_donor_means['age_group']=='old').sum()),
    },
    'fap_ligands': fap_ligand_results,
    'musc_receptors': musc_receptor_results,
    'crosstalk': crosstalk_results,
    'net_aging_score': {
        'inflammatory_gain': float(infl_gain),
        'regenerative_loss': float(regen_loss),
        'net_score': float(net_score),
        'interpretation': interpretation,
        'sig_inflammatory_axes': int(sig_infl),
        'sig_regenerative_axes': int(sig_regen),
    }
}

with open('experiments/batch_024/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to experiments/batch_024/results.json")
print("=" * 80)
