#!/usr/bin/env python3
"""
batch_068: Ligand-Receptor Analysis for Vascular→FAP Paracrine Crosstalk

Design: PI #6 (L-R analysis) + PI #13 (cross-compartment signaling)
Therapeutic thesis: JNK→AP-1→CDKN1A→SASP in vascular endothelium (ESTABLISHED, rho=0.929)

Hypothesis: Vascular-derived ligands increase with age and drive FAP dysfunction
via specific receptor pathways. JNK inhibition should reduce these pro-aging signals.

Approach:
1. Identify vascular ligands (cellphoneDB-style) relevant to muscle aging
2. Identify FAP receptors (FGFR1/2, MET, EGFR, PDGFR, etc.)
3. Compute age-dependent ligand expression in vascular (Spearman correlation)
4. Compute predicted L-R interaction strength (ligand × receptor)
5. Correlate with vascular JUNB/SASP activation (F084: rho=0.929)
6. Identify which pathways are most affected by JNK inhibition

L-R pairs based on: CellPhoneDB, Nature Communications 2020 (muscle aging),
and prior FAP characterization (F087: HGF→MET dominant crosstalk).
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from scipy.sparse import issparse

warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro")
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "batch_068"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# SASP12 genes (literature-curated)
SASP12_GENES = [
    "CCL2", "CCL7", "CCL20", "CXCL6", "CXCL8",  # Chemokines
    "IL6",  # Cytokine
    "MMP1", "MMP3",  # Matrix remodeling
    "SERPINE1",  # Serpin
    "IGFBP2", "IGFBP3", "IGFBP5"  # IGFBPs
]

# Ligand-Receptor pairs (vascular ligands → FAP receptors)
# Based on: CellPhoneDB v5.0, muscle-relevant pathways, prior FAP characterization
LR_PAIRS = {
    # VEGF/KDR axis (angiogenesis)
    ("VEGFA", "KDR"): {"pathway": "VEGF", "function": "angiogenesis, EC survival"},
    ("VEGFB", "KDR"): {"pathway": "VEGF", "function": "angiogenesis, metabolic"},
    ("VEGFC", "KDR"): {"pathway": "VEGF", "function": "lymphangiogenesis"},
    ("VEGFA", "FLT1"): {"pathway": "VEGF", "function": "VEGF sequestration"},

    # ANGPT/TIE axis (vascular stability)
    ("ANGPT1", "TEK"): {"pathway": "ANGPT", "function": "vessel stabilization"},
    ("ANGPT2", "TEK"): {"pathway": "ANGPT", "function": "vessel destabilization"},
    ("ANGPT1", "TIE1"): {"pathway": "ANGPT", "function": "TIE1 trans-activation"},
    ("ANGPT2", "TIE1"): {"pathway": "ANGPT", "function": "TIE1 signaling"},

    # Ephrin/Eph (vascular patterning)
    ("EFNA1", "EPHA2"): {"pathway": "Ephrin", "function": "vascular patterning"},
    ("EFNA1", "EPHA4"): {"pathway": "Ephrin", "function": "vascular patterning"},
    ("EFNB2", "EPHA2"): {"pathway": "Ephrin", "function": "tip cell guidance"},
    ("EFNB2", "EPHA4"): {"pathway": "Ephrin", "function": "tip cell guidance"},

    # Notch/JAG (vascular differentiation)
    ("JAG1", "NOTCH1"): {"pathway": "Notch", "function": "EC arterial spec"},
    ("JAG1", "NOTCH2"): {"pathway": "Notch", "function": "mesenchymal Notch"},
    ("JAG2", "NOTCH1"): {"pathway": "Notch", "function": "EC arterial spec"},
    ("JAG2", "NOTCH2"): {"pathway": "Notch", "function": "mesenchymal Notch"},

    # Semaphorin/Neuropilin (axon guidance, angiogenesis)
    ("SEMA3A", "NRP1"): {"pathway": "SEMA", "function": "axon guidance, anti-angiogenesis"},
    ("SEMA3C", "NRP1"): {"pathway": "SEMA", "function": "angiogenesis"},
    ("SEMA3F", "NRP2"): {"pathway": "SEMA", "function": "anti-angiogenesis"},
    ("SEMA3A", "NRP2"): {"pathway": "SEMA", "function": "axon guidance"},

    # FGF/FGFR (fibroblast growth, F087 confirmed FAP expresses FGFR1/2)
    ("FGF2", "FGFR1"): {"pathway": "FGF", "function": "fibroblast proliferation"},
    ("FGF2", "FGFR2"): {"pathway": "FGF", "function": "fibroblast proliferation"},
    ("FGF2", "FGFR3"): {"pathway": "FGF", "function": "fibroblast differentiation"},
    ("FGF6", "FGFR1"): {"pathway": "FGF", "function": "muscle precursor"},
    ("FGF6", "FGFR2"): {"pathway": "FGF", "function": "muscle precursor"},
    ("FGF7", "FGFR2"): {"pathway": "FGF", "function": "epithelial/fibroblast migration"},
    ("FGF10", "FGFR2"): {"pathway": "FGF", "function": "progenitor expansion"},
    ("FGF1", "FGFR1"): {"pathway": "FGF", "function": "broad FGFR1 activation"},
    ("FGF1", "FGFR2"): {"pathway": "FGF", "function": "broad FGFR2 activation"},

    # HGF/MET (F087: dominant FAP→MuSC crosstalk; vascular source?)
    ("HGF", "MET"): {"pathway": "HGF", "function": "fibroblast motility, FAP→MuSC"},

    # IL6/IL6R (inflammation, SASP component)
    ("IL6", "IL6R"): {"pathway": "IL6", "function": "inflammation, acute phase"},
    ("IL11", "IL11RA1"): {"pathway": "IL6", "function": "fibrosis, STAT3"},

    # CXC chemokines (inflammation, SASP components)
    ("CXCL8", "CXCR1"): {"pathway": "CXCL", "function": "neutrophil chemotaxis"},
    ("CXCL8", "CXCR2"): {"pathway": "CXCL", "function": "neutrophil chemotaxis"},
    ("CXCL1", "CXCR2"): {"pathway": "CXCL", "function": "neutrophil chemotaxis"},
    ("CXCL2", "CXCR2"): {"pathway": "CXCL", "function": "neutrophil chemotaxis"},
    ("CXCL12", "CXCR4"): {"pathway": "CXCL12", "function": "stem cell homing"},
    ("CXCL12", "ACKR3"): {"pathway": "CXCL12", "function": "decoy receptor"},  # ACKR3=CXCR7

    # TNF/TNFR (inflammation)
    ("TNF", "TNFRSF1A"): {"pathway": "TNF", "function": "pro-inflammatory, NF-κB"},
    ("TNF", "TNFRSF1B"): {"pathway": "TNF", "function": "anti-inflammatory decoy"},

    # PDGF (fibroblast proliferation)
    ("PDGFA", "PDGFRA"): {"pathway": "PDGF", "function": "fibroblast proliferation"},
    ("PDGFB", "PDGFRB"): {"pathway": "PDGF", "function": "pericyte recruitment"},
    ("PDGFA", "PDGFRB"): {"pathway": "PDGF", "function": "fibroblast/pericyte"},
    ("PDGFB", "PDGFRA"): {"pathway": "PDGF", "function": "fibroblast proliferation"},

    # TGF-beta (fibrosis, context-dependent)
    ("TGFB1", "TGFBR1"): {"pathway": "TGF-beta", "function": "fibrosis, SMAD"},
    ("TGFB1", "TGFBR2"): {"pathway": "TGF-beta", "function": "fibrosis, SMAD"},
    ("TGFB2", "TGFBR1"): {"pathway": "TGF-beta", "function": "fibrosis, SMAD"},
    ("TGFB2", "TGFBR2"): {"pathway": "TGF-beta", "function": "fibrosis, SMAD"},
    ("TGFB3", "TGFBR1"): {"pathway": "TGF-beta", "function": "wound healing"},
    ("TGFB3", "TGFBR2"): {"pathway": "TGF-beta", "function": "wound healing"},
}


def load_dataset(name, path):
    """Load dataset with metadata extraction."""
    print(f"  Loading {name} from {path.name}...")
    adata = sc.read_h5ad(path)

    # Find donor and age columns
    donor_col = None
    for col in ['sample', 'donor', 'Donor', 'Subject', 'orig.ident']:
        if col in adata.obs.columns:
            donor_col = col
            break

    # Find continuous age column (prefer float over categorical)
    age_col = None
    for col in ['age']:
        if col in adata.obs.columns:
            if adata.obs[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                age_col = col
                break
            elif adata.obs[col].dtype.name == 'category':
                # Try to convert
                try:
                    ages = pd.to_numeric(adata.obs[col].cat.categories)
                    if not ages.isna().all():
                        age_col = col
                except:
                    pass

    n_donors = adata.obs[donor_col].nunique() if donor_col else 0
    print(f"    {name}: {adata.n_obs} cells, {n_donors} donors")
    print(f"    Donor col: {donor_col}, Age col: {age_col}")

    if age_col and adata.obs[age_col].dtype.name == 'category':
        ages = pd.to_numeric(adata.obs[age_col].cat.categories)
        print(f"    Age values: {sorted(ages[~np.isnan(ages)].tolist())[:10]}")
    elif age_col:
        print(f"    Age range: {adata.obs[age_col].min():.1f} - {adata.obs[age_col].max():.1f}")

    return adata, donor_col, age_col


def get_donor_age_mapping(adata, donor_col, age_col):
    """Get donor-level age (continuous)."""
    if age_col and adata.obs[age_col].dtype.name == 'category':
        # Create mapping from category to numeric
        cat_to_age = {cat: float(val) for cat, val in
                      zip(adata.obs[age_col].cat.categories,
                          pd.to_numeric(adata.obs[age_col].cat.categories))
                      if not np.isnan(float(val))}
        donor_age = adata.obs[donor_col].map(lambda x: cat_to_age.get(x, np.nan))
    else:
        donor_age = adata.obs.groupby(donor_col)[age_col].first()

    return donor_age


def compute_donor_expression(adata, genes, donor_col):
    """Compute mean log1p expression per donor."""
    available = [g for g in genes if g in adata.var_names]
    if not available:
        return pd.Series(dtype=float)

    idx = adata.var_names.get_indexer(available)
    if 'raw' in adata.layers:
        X = adata.layers['raw'][:, idx]
    else:
        X = adata.X[:, idx]

    if issparse(X):
        X = X.toarray()
    X_log = np.log1p(X)

    # Per-donor mean
    donor_expr = pd.DataFrame({
        'donor': adata.obs[donor_col].values,
        'expr': X_log.mean(axis=1)
    }).groupby('donor')['expr'].mean()

    return donor_expr


def compute_sasp_score(adata, donor_col):
    """Compute SASP12 score per donor (mean of log1p expression)."""
    return compute_donor_expression(adata, SASP12_GENES, donor_col)


def compute_junb_score(adata, donor_col):
    """Compute JUNB TF score per donor."""
    return compute_donor_expression(adata, ["JUNB"], donor_col)


def correlate_with_age(donor_expr, donor_age):
    """Compute Spearman correlation with age."""
    # Ensure both are numeric Series with proper alignment
    donor_age = pd.Series(donor_age.values, index=donor_age.index, dtype=float)
    donor_expr = pd.Series(donor_expr.values, index=donor_expr.index, dtype=float)

    common = donor_expr.index.intersection(donor_age.index)
    if len(common) < 5:
        return (np.nan, np.nan, 0)

    x = donor_expr.loc[list(common)].values.astype(float)
    y = donor_age.loc[list(common)].values.astype(float)

    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 5:
        return (np.nan, np.nan, 0)

    rho, p = spearmanr(x[valid], y[valid])
    return (rho, p, int(valid.sum()))


def compute_interaction_score(ligand_expr_vasc, receptor_expr_fap):
    """Compute predicted L-R interaction (product of mean expressions)."""
    ligand_mean = ligand_expr_vasc.mean()
    receptor_mean = receptor_expr_fap.mean()
    return ligand_mean * receptor_mean


def main():
    print("=" * 70)
    print("BATCH_068: Ligand-Receptor Analysis (Vascular → FAP Crosstalk)")
    print("=" * 70)
    print("Date: 2026-04-23")
    print("Therapeutic thesis: JNK→AP-1→SASP in vascular (rho=0.929, F084)")
    print("Question: Which vascular ligands increase with age and drive FAP dysfunction?")
    print("=" * 70)

    # Load datasets
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    vasc_path = DATA_DIR / "Vascular_scsn_RNA.h5ad"
    fap_path = DATA_DIR / "OMIX004308-02.h5ad"

    vasc, vasc_donor_col, vasc_age_col = load_dataset("vascular", vasc_path)
    fap, fap_donor_col, fap_age_col = load_dataset("fap", fap_path)

    # Get donor age mapping (use vascular continuous age)
    donor_age = get_donor_age_mapping(vasc, vasc_donor_col, vasc_age_col)
    print(f"\nDonor age mapping (N={len(donor_age)} donors):")
    print(donor_age.describe())

    # Extract unique ligands and receptors from LR_PAIRS
    ligands = list(set([pair[0] for pair in LR_PAIRS.keys()]))
    receptors = list(set([pair[1] for pair in LR_PAIRS.keys()]))

    print(f"\n{len(LR_PAIRS)} L-R pairs, {len(ligands)} unique ligands, {len(receptors)} unique receptors")

    # Compute donor-level expression in vascular (ligands)
    print("\n" + "=" * 70)
    print("COMPUTING VASCULAR LIGAND EXPRESSION")
    print("=" * 70)

    vasc_ligand_expr = {}
    for ligand in ligands:
        expr = compute_donor_expression(vasc, [ligand], vasc_donor_col)
        vasc_ligand_expr[ligand] = expr

    # Compute vascular JUNB and SASP scores
    vasc_junb = compute_junb_score(vasc, vasc_donor_col)
    vasc_sasp = compute_sasp_score(vasc, vasc_donor_col)

    print(f"Vascular JUNB (donor level): mean={vasc_junb.mean():.3f}, std={vasc_junb.std():.3f}")
    print(f"Vascular SASP12 (donor level): mean={vasc_sasp.mean():.3f}, std={vasc_sasp.std():.3f}")

    # Correlate JUNB and SASP (should replicate F084: rho=0.929)
    common_junb_sasp = vasc_junb.index.intersection(vasc_sasp.index)
    if len(common_junb_sasp) >= 5:
        rho_junb_sasp, p_junb_sasp = spearmanr(vasc_junb[common_junb_sasp], vasc_sasp[common_junb_sasp])
        print(f"\nVascular JUNB↔SASP12 correlation: rho={rho_junb_sasp:.3f}, p={p_junb_sasp:.4f}")

    # Compute donor-level expression in FAP (receptors)
    print("\n" + "=" * 70)
    print("COMPUTING FAP RECEPTOR EXPRESSION")
    print("=" * 70)

    fap_receptor_expr = {}
    for receptor in receptors:
        expr = compute_donor_expression(fap, [receptor], fap_donor_col)
        fap_receptor_expr[receptor] = expr

    # Report receptor expression in FAPs
    receptor_expr_summary = []
    for receptor in receptors:
        if receptor in fap_receptor_expr and len(fap_receptor_expr[receptor]) > 0:
            mean_expr = fap_receptor_expr[receptor].mean()
            std_expr = fap_receptor_expr[receptor].std()
            receptor_expr_summary.append({
                'receptor': receptor,
                'mean': mean_expr,
                'std': std_expr
            })

    receptor_df = pd.DataFrame(receptor_expr_summary).sort_values('mean', ascending=False)
    print("\nTop FAP receptors by expression:")
    for _, row in receptor_df.head(10).iterrows():
        print(f"  {row['receptor']:15s}: {row['mean']:.3f} ± {row['std']:.3f}")

    # Analyze each L-R pair
    print("\n" + "=" * 70)
    print("LIGAND-RECEPTOR PAIR ANALYSIS")
    print("=" * 70)

    results = []

    for (ligand, receptor), info in LR_PAIRS.items():
        # Get ligand expression in vascular
        if ligand not in vasc_ligand_expr or len(vasc_ligand_expr[ligand]) == 0:
            continue
        ligand_expr = vasc_ligand_expr[ligand]

        # Get receptor expression in FAP
        if receptor not in fap_receptor_expr or len(fap_receptor_expr[receptor]) == 0:
            continue
        receptor_expr = fap_receptor_expr[receptor]

        # Correlation with age (vascular ligand)
        rho_age, p_age, n_age = correlate_with_age(ligand_expr, donor_age)

        # Correlation with JUNB (vascular)
        rho_junb, p_junb, n_junb = correlate_with_age(ligand_expr, vasc_junb)

        # Correlation with SASP (vascular)
        rho_sasp, p_sasp, n_sasp = correlate_with_age(ligand_expr, vasc_sasp)

        # Interaction score (vascular ligand × FAP receptor)
        interaction = compute_interaction_score(ligand_expr, receptor_expr)

        # Age × Interaction: if ligand increases with age, interaction increases
        # (this predicts what happens in aged vasculature)

        results.append({
            'ligand': ligand,
            'receptor': receptor,
            'pathway': info['pathway'],
            'function': info['function'],
            'ligand_mean_vasc': float(ligand_expr.mean()),
            'ligand_std_vasc': float(ligand_expr.std()),
            'receptor_mean_fap': float(receptor_expr.mean()),
            'receptor_std_fap': float(receptor_expr.std()),
            'interaction_score': float(interaction),
            'rho_age': float(rho_age) if not np.isnan(rho_age) else None,
            'p_age': float(p_age) if not np.isnan(p_age) else None,
            'n_age': n_age,
            'rho_junb': float(rho_junb) if not np.isnan(rho_junb) else None,
            'p_junb': float(p_junb) if not np.isnan(p_junb) else None,
            'n_junb': n_junb,
            'rho_sasp': float(rho_sasp) if not np.isnan(rho_sasp) else None,
            'p_sasp': float(p_sasp) if not np.isnan(p_sasp) else None,
            'n_sasp': n_sasp,
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by age correlation (most age-dependent first)
    results_df = results_df.sort_values('rho_age', ascending=False, na_position='last')

    # Multiple testing correction (FDR)
    from scipy.stats import false_discovery_control
    valid_p = results_df['p_age'].dropna()
    if len(valid_p) > 0:
        fdr = false_discovery_control(valid_p.values, method='bh')
        results_df.loc[valid_p.index, 'fdr_age'] = fdr

    print(f"\nAnalyzed {len(results_df)} L-R pairs")

    # Summary statistics
    n_sig_age = (results_df['p_age'] < 0.05).sum() if 'p_age' in results_df.columns else 0
    n_sig_junb = (results_df['p_junb'] < 0.05).sum() if 'p_junb' in results_df.columns else 0
    n_sig_sasp = (results_df['p_sasp'] < 0.05).sum() if 'p_sasp' in results_df.columns else 0

    print(f"\nSignificant correlations (p < 0.05):")
    print(f"  With age (vascular ligand): {n_sig_age} pairs")
    print(f"  With JUNB (vascular): {n_sig_junb} pairs")
    print(f"  With SASP (vascular): {n_sig_sasp} pairs")

    # Top pairs by age correlation
    print("\n" + "=" * 70)
    print("TOP PAIRS BY AGE-DEPENDENT LIGAND EXPRESSION (Vascular)")
    print("=" * 70)
    print(f"{'Ligand':12s} → {'Receptor':12s} | {'Pathway':12s} | {'rho(age)':8s} | {'p(age)':8s} | {'Interaction':10s}")
    print("-" * 70)

    for _, row in results_df.head(15).iterrows():
        rho = f"{row['rho_age']:.3f}" if not pd.isna(row['rho_age']) else "NA"
        p = f"{row['p_age']:.4f}" if not pd.isna(row['p_age']) else "NA"
        fdr_val = f"[FDR={row['fdr_age']:.3f}]" if 'fdr_age' in row and not pd.isna(row['fdr_age']) else ""
        print(f"{row['ligand']:12s} → {row['receptor']:12s} | {row['pathway']:12s} | {rho:8s} | {p:8s} {fdr_val} | {row['interaction_score']:.3f}")

    # Pathway-level aggregation
    print("\n" + "=" * 70)
    print("PATHWAY-LEVEL SUMMARY (Vascular → FAP)")
    print("=" * 70)

    pathway_summary = results_df.groupby('pathway').agg({
        'rho_age': ['mean', 'max'],
        'p_age': 'min',
        'interaction_score': 'mean',
        'ligand': 'count'
    }).round(4)
    pathway_summary.columns = ['rho_age_mean', 'rho_age_max', 'p_age_min', 'interaction_mean', 'n_pairs']
    pathway_summary = pathway_summary.sort_values('rho_age_mean', ascending=False)

    print(f"\n{'Pathway':15s} | {'N pairs':8s} | {'rho(age) mean':12s} | {'rho(age) max':12s} | {'p(min)':10s} | {'Interaction':10s}")
    print("-" * 80)

    for pathway, row in pathway_summary.iterrows():
        rho_mean = f"{row['rho_age_mean']:.3f}" if not np.isnan(row['rho_age_mean']) else "NA"
        rho_max = f"{row['rho_age_max']:.3f}" if not np.isnan(row['rho_age_max']) else "NA"
        p_min = f"{row['p_age_min']:.4f}" if not np.isnan(row['p_age_min']) else "NA"
        print(f"{pathway:15s} | {int(row['n_pairs']):8d} | {rho_mean:12s} | {rho_max:12s} | {p_min:10s} | {row['interaction_mean']:.3f}")

    # Save results
    results_csv = RESULTS_DIR / "results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")

    # Key findings summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Identify pairs with significant age correlation (positive = increase with age)
    age_up = results_df[(results_df['p_age'] < 0.05) & (results_df['rho_age'] > 0)]
    age_down = results_df[(results_df['p_age'] < 0.05) & (results_df['rho_age'] < 0)]

    print(f"\n1. Vascular ligands INCREASING with age (rho > 0, p < 0.05): {len(age_up)} pairs")
    if len(age_up) > 0:
        top_age_up = age_up.nlargest(5, 'rho_age')
        for _, row in top_age_up.iterrows():
            print(f"   {row['ligand']}→{row['receptor']} ({row['pathway']}): rho={row['rho_age']:.3f}, p={row['p_age']:.4f}")

    print(f"\n2. Vascular ligands DECREASING with age (rho < 0, p < 0.05): {len(age_down)} pairs")
    if len(age_down) > 0:
        top_age_down = age_down.nsmallest(5, 'rho_age')
        for _, row in top_age_down.iterrows():
            print(f"   {row['ligand']}→{row['receptor']} ({row['pathway']}): rho={row['rho_age']:.3f}, p={row['p_age']:.4f}")

    # Pairs correlated with JUNB (proxies for JNK activation)
    junb_corr = results_df[(results_df['p_junb'] < 0.05) & (results_df['rho_junb'] > 0)]
    print(f"\n3. Ligands correlated with JUNB (JNK pathway, rho > 0, p < 0.05): {len(junb_corr)} pairs")
    if len(junb_corr) > 0:
        top_junb = junb_corr.nlargest(5, 'rho_junb')
        for _, row in top_junb.iterrows():
            print(f"   {row['ligand']}→{row['receptor']} ({row['pathway']}): rho={row['rho_junb']:.3f}, p={row['p_junb']:.4f}")

    # Pairs correlated with SASP
    sasp_corr = results_df[(results_df['p_sasp'] < 0.05) & (results_df['rho_sasp'] > 0)]
    print(f"\n4. Ligands correlated with SASP (rho > 0, p < 0.05): {len(sasp_corr)} pairs")
    if len(sasp_corr) > 0:
        top_sasp = sasp_corr.nlargest(5, 'rho_sasp')
        for _, row in top_sasp.iterrows():
            print(f"   {row['ligand']}→{row['receptor']} ({row['pathway']}): rho={row['rho_sasp']:.3f}, p={row['p_sasp']:.4f}")

    # Therapeutic implications
    print("\n" + "=" * 70)
    print("THERAPEUTIC IMPLICATIONS (JNK Inhibition)")
    print("=" * 70)

    # If JNK→AP-1→SASP is established, reducing JNK should reduce SASP
    # Which ligands would be reduced?
    jnk_targets = results_df[(results_df['rho_junb'] > 0) & (results_df['p_junb'] < 0.1)]

    if len(jnk_targets) > 0:
        print(f"\nL-R pairs predicted to be REDUCED by JNK inhibition: {len(jnk_targets)}")
        print("(Based on positive correlation with JUNB, which is downstream of JNK)")
        for _, row in jnk_targets.nlargest(10, 'rho_junb').iterrows():
            effect = "PROTECTIVE" if row['rho_age'] < 0 else "PRO-AGING"
            print(f"   {row['ligand']}→{row['receptor']} ({row['pathway']}): rho(JUNB)={row['rho_junb']:.3f} [{effect}]")
    else:
        print("No L-R pairs reached significance for JUNB correlation (p < 0.1)")
        print("This suggests ligand expression may not be JUNB-dependent")
        print("OR the sample size (N=23 donors) is underpowered")

    # Save JSON summary
    summary = {
        "batch": "batch_068",
        "date": "2026-04-23",
        "n_lr_pairs_analyzed": len(results_df),
        "n_vascular_donors": len(donor_age),
        "n_fap_donors": len(fap_receptor_expr[receptors[0]]) if receptors else 0,
        "junb_sasp_rho": float(rho_junb_sasp) if 'rho_junb_sasp' in dir() else None,
        "junb_sasp_p": float(p_junb_sasp) if 'p_junb_sasp' in dir() else None,
        "n_age_increasing": int(len(age_up)),
        "n_age_decreasing": int(len(age_down)),
        "n_junb_correlated": int(len(junb_corr)),
        "n_sasp_correlated": int(len(sasp_corr)),
        "top_age_dependent_pairs": [
            {"ligand": row['ligand'], "receptor": row['receptor'],
             "rho_age": row['rho_age'], "p_age": row['p_age']}
            for _, row in results_df.head(10).iterrows()
        ],
        "results_csv": str(results_csv)
    }

    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nJSON summary saved to: {RESULTS_DIR / 'results.json'}")

    return results_df, summary


if __name__ == "__main__":
    results_df, summary = main()
