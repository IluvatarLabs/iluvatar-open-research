#!/usr/bin/env python3
"""
batch_039: P1 — Unbiased TF Screen at Donor Level (Optimized)
=============================================================
For each compartment (FAP, MuSC, vascular), compute donor-level
Spearman rho(TF, SASP12) for all detected transcription factors.
Uses compartment-specific data files for efficiency.

Fixed from v1: Uses separate data files for each compartment.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import json
import warnings
warnings.filterwarnings('ignore')

# Constants
SASP12_GENES = ['CCL2', 'CCL7', 'CXCL8', 'IL6', 'SERPINE1', 'MMP1',
                'MMP3', 'IGFBP2', 'IGFBP3', 'IGFBP5', 'CXCL6', 'CCL20']
AP1_FAMILY = ['JUNB', 'FOS', 'FOSB', 'FOSL1', 'FOSL2', 'JUN', 'JUND']

# Comprehensive TF list (Lambert 2018 + curated additions)
TF_LIST = [
    'AHR', 'AHRR', 'AJUBA', 'ALX1', 'ALX3', 'ALX4',
    'AR', 'ARID1A', 'ARID1B', 'ARID2', 'ARID3A', 'ARID3B', 'ARID3C', 'ARID5A', 'ARID5B',
    'ATF1', 'ATF2', 'ATF3', 'ATF4', 'ATF5', 'ATF6', 'ATF6B', 'ATF7',
    'BACH1', 'BACH2', 'BATF', 'BATF2', 'BATF3',
    'BCL6', 'BPTF',
    'CUX1', 'CUX2',
    'DBP',
    'DLX1', 'DLX2', 'DLX3', 'DLX4', 'DLX5', 'DLX6',
    'DMRT1', 'DMRT2', 'DMRT3', 'DUX4',
    'E2F1', 'E2F2', 'E2F3', 'E2F4', 'E2F5', 'E2F6', 'E2F7', 'E2F8',
    'EGR1', 'EGR2', 'EGR3', 'EGR4',
    'ELF1', 'ELF2', 'ELF3', 'ELF4', 'ELK1', 'ELK3', 'ELK4',
    'EPAS1', 'ERF', 'ERG',
    'ESR1', 'ESR2', 'ESRRA', 'ESRRB', 'ESRRG',
    'ETV1', 'ETV2', 'ETV3', 'ETV4', 'ETV5', 'ETV6',
    'FEV', 'FLI1',
    'FOS', 'FOSB', 'FOSL1', 'FOSL2',
    'FOXA1', 'FOXA2', 'FOXA3', 'FOXB1', 'FOXC1', 'FOXC2',
    'FOXD1', 'FOXD2', 'FOXD3', 'FOXD4', 'FOXD5',
    'FOXE1', 'FOXE3', 'FOXF1', 'FOXF2',
    'FOXH1', 'FOXI1', 'FOXK1', 'FOXK2',
    'FOXL1', 'FOXL2', 'FOXM1', 'FOXN1', 'FOXN2',
    'FOXO1', 'FOXO3', 'FOXO4', 'FOXO6',
    'FOXP1', 'FOXP2', 'FOXP3', 'FOXP4',
    'GATA1', 'GATA2', 'GATA3', 'GATA4', 'GATA5', 'GATA6',
    'GCM1', 'GCM2',
    'GFI1', 'GFI1B',
    'GLI1', 'GLI2', 'GLI3', 'GLI4',
    'HAND1', 'HAND2', 'HBZ',
    'HES1', 'HES2', 'HES3', 'HES4', 'HES5', 'HES6', 'HES7',
    'HEY1', 'HEY2', 'HEYL',
    'HIF1A', 'HIF2A', 'HIF3A',
    'HLF',
    'HNF1A', 'HNF1B', 'HNF4A', 'HNF4G',
    'HOMEZ',
    'HOXA1', 'HOXA10', 'HOXA11', 'HOXA13', 'HOXA2', 'HOXA3', 'HOXA4', 'HOXA5', 'HOXA6', 'HOXA7', 'HOXA9',
    'HOXB1', 'HOXB13', 'HOXB2', 'HOXB3', 'HOXB4', 'HOXB5', 'HOXB6', 'HOXB7', 'HOXB8', 'HOXB9',
    'HOXC10', 'HOXC11', 'HOXC12', 'HOXC13', 'HOXC4', 'HOXC5', 'HOXC6', 'HOXC8', 'HOXC9',
    'HOXD1', 'HOXD10', 'HOXD11', 'HOXD12', 'HOXD13', 'HOXD3', 'HOXD4', 'HOXD8', 'HOXD9',
    'HSF1', 'HSF2', 'HSF3', 'HSF4', 'HSF5',
    'ICER',
    'IRF1', 'IRF2', 'IRF3', 'IRF4', 'IRF5', 'IRF6', 'IRF7', 'IRF8', 'IRF9',
    'JDP2',
    'JUN', 'JUNB', 'JUND',
    'KAT2A', 'KAT2B', 'KAT6A', 'KAT6B', 'KAT7',
    'KLF1', 'KLF10', 'KLF11', 'KLF12', 'KLF13', 'KLF14', 'KLF15', 'KLF16', 'KLF17',
    'KLF2', 'KLF3', 'KLF4', 'KLF5', 'KLF6', 'KLF7', 'KLF8', 'KLF9',
    'LHX1', 'LHX2', 'LHX3', 'LHX4', 'LHX5', 'LHX6', 'LHX8', 'LHX9',
    'MAF', 'MAFA', 'MAFB', 'MAFF', 'MAFG', 'MAFK',
    'MAX', 'MAZ',
    'MECOM',
    'MEF2A', 'MEF2B', 'MEF2C', 'MEF2D',
    'MLX', 'MLXIP', 'MLXIPL',
    'MSC', 'MTF1',
    'MXI1',
    'MYB', 'MYC', 'MYCN',
    'MYF5', 'MYF6', 'MYOD1', 'MYOG',
    'NANOG',
    'NFE2', 'NFE2L1', 'NFE2L2', 'NFE2L3',
    'NFIA', 'NFIB', 'NFIC', 'NFIL3', 'NFIX',
    'NFKB1', 'NFKB2', 'NFKBIA', 'NFKBIAP1',
    'NFX1', 'NFXL1',
    'NFYA', 'NFYB', 'NFYC',
    'NHLH1', 'NHLH2',
    'NKX2-1', 'NKX2-2', 'NKX2-5', 'NKX2-8', 'NKX3-1', 'NKX3-2', 'NKX6-1', 'NKX6-2', 'NKX6-3',
    'NOBOX', 'NOTO',
    'NR0B1', 'NR0B2', 'NR1D1', 'NR1D2', 'NR1H2', 'NR1H3', 'NR1H4', 'NR1I2', 'NR1I3',
    'NR2C1', 'NR2C2', 'NR2E1', 'NR2E3', 'NR2F1', 'NR2F2', 'NR2F6',
    'NR3C1', 'NR3C2',
    'NR4A1', 'NR4A2', 'NR4A3',
    'NR5A1', 'NR5A2', 'NR6A1',
    'NRF1',
    'ONECUT1', 'ONECUT2', 'ONECUT3',
    'OTX1', 'OTX2',
    'Ovol1', 'Ovol2', 'Ovol3',
    'PARP1', 'PATZ1',
    'PAX1', 'PAX2', 'PAX3', 'PAX4', 'PAX5', 'PAX6', 'PAX7', 'PAX8', 'PAX9',
    'PBX1', 'PBX2', 'PBX3', 'PBX4',
    'PLAG1', 'PLAGL1', 'PLAGL2',
    'PML',
    'POU1F1', 'POU2F1', 'POU2F2', 'POU2F3', 'POU3F1', 'POU3F2', 'POU3F3', 'POU3F4',
    'POU4F1', 'POU4F2', 'POU4F3', 'POU5F1', 'POU5F1B', 'POU6F1', 'POU6F2',
    'PPARA', 'PPARD', 'PPARG',
    'PRDM1', 'PRDM2', 'PRDM4', 'PRDM5', 'PRDM6', 'PRDM7', 'PRDM8', 'PRDM9', 'PRDM10', 'PRDM11', 'PRDM12', 'PRDM13', 'PRDM14', 'PRDM15', 'PRDM16',
    'PROX1', 'PROX2',
    'RAX', 'RAX2',
    'RBPJ', 'RBPJL',
    'REL', 'RELA', 'RELB',
    'RERA',
    'RFX1', 'RFX2', 'RFX3', 'RFX4', 'RFX5', 'RFX6', 'RFX7', 'RFX8',
    'RHOX1', 'RHOXF1', 'RHOXF2', 'RHOXF2B',
    'RUNX1', 'RUNX2', 'RUNX3', 'RUNX4',
    'RXRA', 'RXRB', 'RXRG',
    'SCX',
    'SIX1', 'SIX2', 'SIX3', 'SIX4', 'SIX5', 'SIX6',
    'SOX1', 'SOX10', 'SOX11', 'SOX12', 'SOX13', 'SOX14', 'SOX15', 'SOX17', 'SOX18', 'SOX2', 'SOX21', 'SOX3', 'SOX30', 'SOX4', 'SOX5', 'SOX6', 'SOX7', 'SOX8', 'SOX9',
    'SP1', 'SP2', 'SP3', 'SP4', 'SP5', 'SP7', 'SP8', 'SP9',
    'SPI1', 'SPIB', 'SPIC',
    'SREBF1', 'SREBF2',
    'ST18',
    'STAT1', 'STAT2', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6',
    'SUB1',
    'T', 'TAL1', 'TAL2',
    'TBP', 'TBPL1', 'TBPL2',
    'TBX1', 'TBX10', 'TBX15', 'TBX18', 'TBX19', 'TBX2', 'TBX20', 'TBX21', 'TBX3', 'TBX4', 'TBX5', 'TBX6', 'TBX7',
    'TCF12', 'TCF15', 'TCF3', 'TCF4', 'TCF7', 'TCF7L1', 'TCF7L2',
    'TEAD1', 'TEAD2', 'TEAD3', 'TEAD4',
    'TFAP2A', 'TFAP2B', 'TFAP2C', 'TFAP2D', 'TFAP2E', 'TFAP4',
    'TFDP1', 'TFDP2', 'TFDP3', 'TFDP4',
    'TFEC', 'TFE3', 'TFEB',
    'TGIF1', 'TGIF2',
    'THAP1', 'THAP11', 'THAP12', 'THAP2', 'THAP3', 'THAP4', 'THAP5', 'THAP6', 'THAP7', 'THAP8', 'THAP9',
    'THRA', 'THRB',
    'TLX1', 'TLX2', 'TLX3',
    'TP63', 'TP73', 'TFCP2', 'TFCP2L1', 'TFCP2L2',
    'TP53',
    'TRPS1',
    'TSHZ1', 'TSHZ2', 'TSHZ3',
    'TWIST1', 'TWIST2',
    'USF1', 'USF2',
    'VDR', 'VEZF1',
    'VGLL1', 'VGLL2', 'VGLL3', 'VGLL4',
    'XBP1', 'XBP2',
    'ZBTB1', 'ZBTB10', 'ZBTB11', 'ZBTB12', 'ZBTB14', 'ZBTB16', 'ZBTB17', 'ZBTB18', 'ZBTB2',
    'ZBTB20', 'ZBTB21', 'ZBTB22', 'ZBTB24', 'ZBTB25', 'ZBTB26', 'ZBTB3', 'ZBTB32', 'ZBTB33', 'ZBTB34',
    'ZBTB38', 'ZBTB39', 'ZBTB4', 'ZBTB40', 'ZBTB41', 'ZBTB42', 'ZBTB43', 'ZBTB44', 'ZBTB45', 'ZBTB46',
    'ZBTB47', 'ZBTB48', 'ZBTB49', 'ZBTB5', 'ZBTB6', 'ZBTB7A', 'ZBTB7B', 'ZBTB7C', 'ZBTB8', 'ZBTB9',
    'ZEB1', 'ZEB2',
    'ZFP1', 'ZFP2', 'ZFP3', 'ZFP36', 'ZFP36L1', 'ZFP36L2', 'ZFP37', 'ZFP42', 'ZFP62',
    'ZFX',
    'ZIC1', 'ZIC2', 'ZIC3', 'ZIC4', 'ZIC5',
    'ZIM1', 'ZIM2', 'ZIM3',
]

# Data file mapping
COMPARTMENT_FILES = {
    'FAP': '/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad',
    'MuSC': '/home/yuanz/Documents/GitHub/biomarvin_fibro/data/MuSC_scsn_RNA.h5ad',
    'vascular': '/home/yuanz/Documents/GitHub/biomarvin_fibro/data/Vascular_scsn_RNA.h5ad'
}

def load_compartment(name, path):
    """Load a compartment-specific data file."""
    print(f"\n{'='*60}")
    print(f"  Loading {name}...")
    print(f"{'='*60}")
    adata = sc.read_h5ad(path, backed=False)
    print(f"  Shape: {adata.shape}")
    print(f"  Annotation: {adata.obs['Annotation'].unique()}")
    print(f"  Samples: {adata.obs['sample'].nunique()} donors")
    return adata

def compute_tf_screen(adata, compartment_name, output_dir):
    """Compute TF-SASP screen for a compartment."""
    print(f"\n  Running {compartment_name} TF-SASP screen...")

    # Get available genes
    available_genes = set(adata.var_names)
    print(f"  Total genes: {len(available_genes)}")

    # Filter TF list to those present in data
    detected_tfs = [tf for tf in TF_LIST if tf in available_genes]
    print(f"  TFs from list detected: {len(detected_tfs)}")

    # Check SASP12 genes
    sasp_available = [g for g in SASP12_GENES if g in available_genes]
    print(f"  SASP12 genes available: {len(sasp_available)}/12")

    if len(sasp_available) < 6:
        print(f"  ERROR: Less than 6 SASP12 genes available")
        return None

    # Compute donor-level mean expression
    donors = adata.obs['sample'].unique()
    n_donors = len(donors)
    print(f"  N donors: {n_donors}")

    # Build donor-level expression matrix
    donor_data = {}
    for donor in donors:
        donor_cells = adata[adata.obs['sample'] == donor]
        if len(donor_cells) < 5:
            continue

        row = {}
        # TFs
        for tf in detected_tfs:
            if tf in available_genes:
                row[tf] = donor_cells[:, tf].X.mean()
        # SASP genes
        for gene in sasp_available:
            row[gene] = donor_cells[:, gene].X.mean()

        donor_data[donor] = row

    donor_df = pd.DataFrame(donor_data).T
    print(f"  Valid donors: {len(donor_df)}")

    if len(donor_df) < 10:
        print(f"  ERROR: Only {len(donor_df)} donors — insufficient")
        return None

    # Compute SASP12 composite
    donor_df['SASP12'] = donor_df[sasp_available].mean(axis=1)

    # Compute Spearman for each TF
    results = []
    sasp_vals = donor_df['SASP12'].values

    for tf in detected_tfs:
        if tf not in donor_df.columns:
            continue

        tf_vals = donor_df[tf].values

        # Remove NaN
        valid = ~(np.isnan(tf_vals) | np.isnan(sasp_vals))
        if valid.sum() < 8:
            continue

        rho, pval = spearmanr(tf_vals[valid], sasp_vals[valid])
        results.append({
            'TF': tf,
            'rho': rho,
            'pvalue': pval,
            'n_donors': int(valid.sum())
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print(f"  ERROR: No valid correlations")
        return None

    # Sort by absolute rho
    results_df['abs_rho'] = results_df['rho'].abs()
    results_df = results_df.sort_values('abs_rho', ascending=False)

    # BH-FDR correction
    _, qvals, _, _ = multipletests(
        results_df['pvalue'].values, alpha=0.05, method='fdr_bh'
    )
    results_df['qvalue'] = qvals
    results_df['significant'] = results_df['qvalue'] < 0.05
    results_df['rank'] = range(1, len(results_df) + 1)

    # Print summary
    print(f"\n  Results for {compartment_name}:")
    print(f"  Total TFs: {len(results_df)}")
    print(f"  Significant (q < 0.05): {results_df['significant'].sum()}")

    junb_row = results_df[results_df['TF'] == 'JUNB']
    if len(junb_row) > 0:
        junb_rank = results_df[results_df['TF'] == 'JUNB'].index[0]
        junb_rank = results_df.index.tolist().index('JUNB') + 1
        print(f"\n  JUNB rank: {junb_rank}/{len(results_df)}")
        print(f"  JUNB rho: {junb_row['rho'].values[0]:.4f}")
        print(f"  JUNB p-value: {junb_row['pvalue'].values[0]:.2e}")
        print(f"  JUNB q-value: {junb_row['qvalue'].values[0]:.2e}")

    print(f"\n  Top 10 TFs:")
    for i, (_, row) in enumerate(results_df.head(10).iterrows()):
        sig = "***" if row['qvalue'] < 0.001 else "**" if row['qvalue'] < 0.01 else "*" if row['qvalue'] < 0.05 else ""
        print(f"    {i+1}. {row['TF']:<8} rho={row['rho']:>7.4f} p={row['pvalue']:.2e} q={row['qvalue']:.2e} {sig}")

    print(f"\n  AP-1 Family:")
    for _, row in results_df[results_df['TF'].isin(AP1_FAMILY)].sort_values('rank').iterrows():
        sig = "***" if row['qvalue'] < 0.001 else "**" if row['qvalue'] < 0.01 else "*" if row['qvalue'] < 0.05 else ""
        print(f"    {row['TF']:<8} rho={row['rho']:>7.4f} rank={row['rank']:3d} q={row['qvalue']:.2e} {sig}")

    # Individual SASP gene correlations with JUNB
    print(f"\n  Individual SASP gene correlations with JUNB:")
    if 'JUNB' in donor_df.columns:
        junb_vals = donor_df['JUNB'].values
        for gene in sasp_available:
            if gene in donor_df.columns:
                gene_vals = donor_df[gene].values
                valid = ~(np.isnan(junb_vals) | np.isnan(gene_vals))
                if valid.sum() >= 5:
                    rho, pval = spearmanr(junb_vals[valid], gene_vals[valid])
                    print(f"    {gene:<10} rho={rho:>7.4f} p={pval:.2e}")

    # Save results
    csv_file = f"{output_dir}/{compartment_name.lower()}_tf_screen.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"\n  Saved: {csv_file}")

    return results_df

def main():
    output_dir = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_039"
    results = {}

    # Process each compartment
    for comp_name, path in COMPARTMENT_FILES.items():
        try:
            adata = load_compartment(comp_name, path)
            result = compute_tf_screen(adata, comp_name, output_dir)
            results[comp_name] = result
        except Exception as e:
            print(f"  ERROR in {comp_name}: {e}")
            import traceback
            traceback.print_exc()
            results[comp_name] = None

    # Summary
    print("\n" + "="*70)
    print("TF SCREEN SUMMARY")
    print("="*70)

    summary = {}
    for comp, df in results.items():
        if df is not None:
            junb_row = df[df['TF'] == 'JUNB']
            n_sig = df['significant'].sum()
            top_row = df.iloc[0]

            junb_rank = df.index.tolist().index('JUNB') + 1 if 'JUNB' in df['TF'].values else None

            summary[comp] = {
                'n_donors': int(len(df['n_donors'].iloc[0])) if len(df) > 0 else 0,
                'n_tfs_tested': int(len(df)),
                'n_significant': int(n_sig),
                'junb_rank': int(junb_rank) if junb_rank else None,
                'junb_rho': float(junb_row['rho'].values[0]) if len(junb_row) > 0 else None,
                'junb_qvalue': float(junb_row['qvalue'].values[0]) if len(junb_row) > 0 else None,
                'top_tf': top_row['TF'],
                'top_rho': float(top_row['rho']),
                'top_qvalue': float(top_row['qvalue']),
            }

            print(f"\n{comp}:")
            print(f"  N donors: {summary[comp]['n_donors']}")
            print(f"  TFs tested: {summary[comp]['n_tfs_tested']}")
            print(f"  Significant (q<0.05): {summary[comp]['n_significant']}")
            print(f"  Top TF: {summary[comp]['top_tf']} (rho={summary[comp]['top_rho']:.4f}, q={summary[comp]['top_qvalue']:.2e})")
            print(f"  JUNB rank: {summary[comp]['junb_rank']} (rho={summary[comp]['junb_rho']:.4f})")

    # Save summary JSON
    summary_file = f"{output_dir}/tf_screen_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {summary_file}")

    return results

if __name__ == "__main__":
    main()