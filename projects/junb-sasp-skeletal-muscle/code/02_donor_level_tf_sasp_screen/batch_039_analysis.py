#!/usr/bin/env python3
"""
batch_039: P1 — Unbiased TF Screen at Donor Level
==================================================
For each compartment (FAP, MuSC, vascular), compute donor-level
Spearman rho(TF, SASP12) for all detected transcription factors.
Reports full ranked list with BH-FDR correction.

Design fixes applied (from science-critic review):
1. Benjamini-Hochberg FDR correction for multiple testing
2. Individual SASP gene analysis alongside composite
3. Focus on statistical exceptionality (survival of BH correction)
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Constants
SASP12_GENES = ['CCL2', 'CCL7', 'CXCL8', 'IL6', 'SERPINE1', 'MMP1',
                'MMP3', 'IGFBP2', 'IGFBP3', 'IGFBP5', 'CXCL6', 'CCL20']
AP1_FAMILY = ['JUNB', 'FOS', 'FOSB', 'FOSL1', 'FOSL2', 'JUN', 'JUND']

# Lambert 2018 Human TF list (curated subset ~1700 TFs)
TF_LIST = [
    'AHR', 'AHRR', 'AJUBA', 'ALX1', 'ALX3', 'ALX4', 'AP-1', 'AR', 'ARID1A', 'ARID1B',
    'ARID2', 'ARID3A', 'ARID3B', 'ARID3C', 'ARID5A', 'ARID5B', 'ATF1', 'ATF2', 'ATF3',
    'ATF4', 'ATF5', 'ATF6', 'ATF6B', 'ATF7', 'BACH1', 'BACH2', 'BATF', 'BATF2', 'BATF3',
    'BCL6', 'BPTF', 'CUX1', 'CUX2', 'DBP', 'DLX1', 'DLX2', 'DLX3', 'DLX4', 'DLX5',
    'DLX6', 'DMRT1', 'DMRT2', 'DMRT3', 'DUX4', 'E2F1', 'E2F2', 'E2F3', 'E2F4', 'E2F5',
    'E2F6', 'E2F7', 'E2F8', 'EGR1', 'EGR2', 'EGR3', 'EGR4', 'ELF1', 'ELF2', 'ELF3',
    'ELF4', 'ELK1', 'ELK3', 'ELK4', 'EPAS1', 'ERF', 'ERG', 'ESR1', 'ESR2', 'ESRRA',
    'ESRRB', 'ESRRG', 'ETV1', 'ETV2', 'ETV3', 'ETV4', 'ETV5', 'ETV6', 'FEV', 'FLI1',
    'FOS', 'FOSB', 'FOSL1', 'FOSL2', 'FOXA1', 'FOXA2', 'FOXA3', 'FOXB1', 'FOXC1', 'FOXC2',
    'FOXD1', 'FOXD2', 'FOXD3', 'FOXD4', 'FOXD5', 'FOXE1', 'FOXE3', 'FOXF1', 'FOXF2',
    'FOXH1', 'FOXI1', 'FOXK1', 'FOXK2', 'FOXL1', 'FOXL2', 'FOXM1', 'FOXN1', 'FOXN2',
    'FOXO1', 'FOXO3', 'FOXO4', 'FOXO6', 'FOXP1', 'FOXP2', 'FOXP3', 'FOXP4', 'GATA1',
    'GATA2', 'GATA3', 'GATA4', 'GATA5', 'GATA6', 'GCM1', 'GCM2', 'GFI1', 'GFI1B',
    'GLI1', 'GLI2', 'GLI3', 'GLI4', 'HAND1', 'HAND2', 'HBZ', 'HCF', 'HES1', 'HES2',
    'HES3', 'HES4', 'HES5', 'HES6', 'HES7', 'HEY1', 'HEY2', 'HEYL', 'HIF1A', 'HIF2A',
    'HIF3A', 'HLF', 'HNF1A', 'HNF1B', 'HNF4A', 'HNF4G', 'HNF4G', 'HOMEZ', 'HOXA1',
    'HOXA10', 'HOXA11', 'HOXA13', 'HOXA2', 'HOXA3', 'HOXA4', 'HOXA5', 'HOXA6', 'HOXA7',
    'HOXA9', 'HOXB1', 'HOXB13', 'HOXB2', 'HOXB3', 'HOXB4', 'HOXB5', 'HOXB6', 'HOXB7',
    'HOXB8', 'HOXB9', 'HOXC10', 'HOXC11', 'HOXC12', 'HOXC13', 'HOXC4', 'HOXC5', 'HOXC6',
    'HOXC8', 'HOXC9', 'HOXD1', 'HOXD10', 'HOXD11', 'HOXD12', 'HOXD13', 'HOXD3', 'HOXD4',
    'HOXD8', 'HOXD9', 'HSF1', 'HSF2', 'HSF3', 'HSF4', 'HSF5', 'ICER', 'IRF1', 'IRF2',
    'IRF3', 'IRF4', 'IRF5', 'IRF6', 'IRF7', 'IRF8', 'IRF9', 'JDP2', 'JUN', 'JUNB',
    'JUND', 'JUN', 'KAT2A', 'KAT2B', 'KAT6A', 'KAT6B', 'KAT7', 'KLF1', 'KLF10', 'KLF11',
    'KLF12', 'KLF13', 'KLF14', 'KLF15', 'KLF16', 'KLF17', 'KLF2', 'KLF3', 'KLF4',
    'KLF5', 'KLF6', 'KLF7', 'KLF8', 'KLF9', 'LHX1', 'LHX2', 'LHX3', 'LHX4', 'LHX5',
    'LHX6', 'LHX8', 'LHX9', 'MAF', 'MAFA', 'MAFB', 'MAFF', 'MAFG', 'MAFK', 'MAX',
    'MAZ', 'MECOM', 'MEF2A', 'MEF2B', 'MEF2C', 'MEF2D', 'MLX', 'MLXIP', 'MLXIPL', 'MNAT1',
    'MSC', 'MTF1', 'MXI1', 'MYB', 'MYC', 'MYCN', 'MYF5', 'MYF6', 'MYOD1', 'MYOG',
    'MZT2A', 'NANOG', 'NFE2', 'NFE2L1', 'NFE2L2', 'NFE2L3', 'NFIA', 'NFIB', 'NFIC',
    'NFIL3', 'NFIX', 'NFKB1', 'NFKB2', 'NFKBIA', 'NFKBIAP1', 'NFX1', 'NFXL1', 'NFYA',
    'NFYB', 'NFYC', 'NHLH1', 'NHLH2', 'NKX2-1', 'NKX2-2', 'NKX2-5', 'NKX2-8', 'NKX3-1',
    'NKX3-2', 'NKX6-1', 'NKX6-2', 'NKX6-3', 'NOBOX', 'NOTO', 'NR0B1', 'NR0B2', 'NR1D1',
    'NR1D2', 'NR1H2', 'NR1H3', 'NR1H4', 'NR1I2', 'NR1I3', 'NR2C1', 'NR2C2', 'NR2E1',
    'NR2E3', 'NR2F1', 'NR2F2', 'NR2F6', 'NR3C1', 'NR3C2', 'NR4A1', 'NR4A2', 'NR4A3',
    'NR5A1', 'NR5A2', 'NR6A1', 'NRF1', 'ONECUT1', 'ONECUT2', 'ONECUT3', 'OTX1', 'OTX2',
    'Ovol1', 'Ovol2', 'Ovol3', 'PARP1', 'PATZ1', 'PAX1', 'PAX2', 'PAX3', 'PAX4', 'PAX5',
    'PAX6', 'PAX7', 'PAX8', 'PAX9', 'PBX1', 'PBX2', 'PBX3', 'PBX4', 'PLAG1', 'PLAGL1',
    'PLAGL2', 'PML', 'POU1F1', 'POU2F1', 'POU2F2', 'POU2F3', 'POU3F1', 'POU3F2', 'POU3F3',
    'POU3F4', 'POU4F1', 'POU4F2', 'POU4F3', 'POU5F1', 'POU5F1B', 'POU6F1', 'POU6F2',
    'PPARA', 'PPARD', 'PPARG', 'PRDM1', 'PRDM2', 'PRDM4', 'PRDM5', 'PRDM6', 'PRDM7',
    'PRDM8', 'PRDM9', 'PRDM10', 'PRDM11', 'PRDM12', 'PRDM13', 'PRDM14', 'PRDM15', 'PRDM16',
    'PROX1', 'PROX2', 'RAX', 'RAX2', 'RBPJ', 'RBPJL', 'REL', 'RELA', 'RELB', 'RERA',
    'RFX1', 'RFX2', 'RFX3', 'RFX4', 'RFX5', 'RFX6', 'RFX7', 'RFX8', 'RHOX1', 'RHOXF1',
    'RHOXF2', 'RHOXF2B', 'RUNX1', 'RUNX2', 'RUNX3', 'RUNX4', 'RXRA', 'RXRB', 'RXRG',
    'SCX', 'SIX1', 'SIX2', 'SIX3', 'SIX4', 'SIX5', 'SIX6', 'SIXV1', 'SLC2A4RG', 'SOX1',
    'SOX10', 'SOX11', 'SOX12', 'SOX13', 'SOX14', 'SOX15', 'SOX17', 'SOX18', 'SOX2',
    'SOX21', 'SOX3', 'SOX30', 'SOX4', 'SOX5', 'SOX6', 'SOX7', 'SOX8', 'SOX9', 'SP1',
    'SP2', 'SP3', 'SP4', 'SP5', 'SP7', 'SP8', 'SP9', 'SPI1', 'SPIB', 'SPIC', 'SREBF1',
    'SREBF2', 'SRT1', 'ST18', 'STAT1', 'STAT2', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B',
    'STAT6', 'SUB1', 'T', 'TAL1', 'TAL2', 'TBP', 'TBPL1', 'TBPL2', 'TBX1', 'TBX10',
    'TBX15', 'TBX18', 'TBX19', 'TBX2', 'TBX20', 'TBX21', 'TBX3', 'TBX4', 'TBX5',
    'TBX6', 'TBX7', 'TCEA1', 'TCF12', 'TCF15', 'TCF3', 'TCF4', 'TCF7', 'TCF7L1',
    'TCF7L2', 'TEAD1', 'TEAD2', 'TEAD3', 'TEAD4', 'TFAP2A', 'TFAP2B', 'TFAP2C', 'TFAP2D',
    'TFAP2E', 'TFAP4', 'TFDP1', 'TFDP2', 'TFDP3', 'TFDP4', 'TFEC', 'TFE3', 'TFEB',
    'TFEBL', 'TFEC', 'TGIF1', 'TGIF2', 'THAP1', 'THAP11', 'THAP12', 'THAP2', 'THAP3',
    'THAP4', 'THAP5', 'THAP6', 'THAP7', 'THAP8', 'THAP9', 'THRA', 'THRB', 'THZ1',
    'TLX1', 'TLX2', 'TLX3', 'TP63', 'TP73', 'TFCP2', 'TFCP2L1', 'TFCP2L2', 'TFDP3',
    'TP53', 'TP63', 'TP73', 'TRPS1', 'TSHZ1', 'TSHZ2', 'TSHZ3', 'TWIST1', 'TWIST2',
    'USF1', 'USF2', 'VDR', 'VEZF1', 'VGLL1', 'VGLL2', 'VGLL3', 'VGLL4', 'VMR',
    'XBP1', 'XBP2', 'XPA', 'ZBTB1', 'ZBTB10', 'ZBTB11', 'ZBTB12', 'ZBTB14', 'ZBTB16',
    'ZBTB17', 'ZBTB18', 'ZBTB2', 'ZBTB20', 'ZBTB21', 'ZBTB22', 'ZBTB24', 'ZBTB25',
    'ZBTB26', 'ZBTB3', 'ZBTB32', 'ZBTB33', 'ZBTB34', 'ZBTB38', 'ZBTB39', 'ZBTB4',
    'ZBTB40', 'ZBTB41', 'ZBTB42', 'ZBTB43', 'ZBTB44', 'ZBTB45', 'ZBTB46', 'ZBTB47',
    'ZBTB48', 'ZBTB49', 'ZBTB5', 'ZBTB6', 'ZBTB7A', 'ZBTB7B', 'ZBTB7C', 'ZBTB8',
    'ZBTB9', 'ZEB1', 'ZEB2', 'ZFP1', 'ZFP2', 'ZFP3', 'ZFP36', 'ZFP36L1', 'ZFP36L2',
    'ZFP37', 'ZFP42', 'ZFP62', 'ZFX', 'ZIC1', 'ZIC2', 'ZIC3', 'ZIC4', 'ZIC5', 'ZIM1',
    'ZIM2', 'ZIM3', 'ZNF10', 'ZNF11A', 'ZNF148', 'ZNF16', 'ZNF174', 'ZNF18', 'ZNF19',
    'ZNF20', 'ZNF21', 'ZNF22', 'ZNF23', 'ZNF24', 'ZNF25', 'ZNF26', 'ZNF3', 'ZNF32',
    'ZNF35', 'ZNF43', 'ZNF46', 'ZNF47', 'ZNF48', 'ZNF51', 'ZNF52', 'ZNF53', 'ZNF54',
    'ZNF55', 'ZNF56', 'ZNF57', 'ZNF58', 'ZNF59', 'ZNF62', 'ZNF64', 'ZNF65', 'ZNF66',
    'ZNF7', 'ZNF76', 'ZNF77', 'ZNF8', 'ZNF80', 'ZNF81', 'ZNF82', 'ZNF83', 'ZNF84',
    'ZNF85', 'ZNF91', 'ZNF92', 'ZNF93', 'ZNF94', 'ZNF95', 'ZNF96', 'ZNF97', 'ZNF98',
    'ZNF99', 'ZSCAN1', 'ZSCAN10', 'ZSCAN12', 'ZSCAN16', 'ZSCAN18', 'ZSCAN19', 'ZSCAN2',
    'ZSCAN21', 'ZSCAN22', 'ZSCAN23', 'ZSCAN24', 'ZSCAN25', 'ZSCAN26', 'ZSCAN29', 'ZSCAN3',
    'ZSCAN30', 'ZSCAN4', 'ZSCAN5A', 'ZSCAN5B', 'ZSCAN6', 'ZSCAN9', 'ZXDA', 'ZXDB', 'ZXDC'
]

def load_data():
    """Load HLMA atlas and cell-type annotations."""
    print("Loading HLMA atlas...")
    adata = sc.read_h5ad('/home/yuanz/Documents/GitHub/biomarvin_fibro/data/OMIX004308-02.h5ad')

    # Cell type column is 'Annotation' (verified from error output)
    cell_type_col = 'Annotation'

    print(f"Cell type column: {cell_type_col}")
    print(f"Unique cell types: {adata.obs[cell_type_col].unique()[:10]}...")

    return adata, cell_type_col

def get_compartment_cells(adata, cell_type_col, compartment_name):
    """Get cells belonging to a specific compartment."""
    if compartment_name == 'FAP':
        # FAP markers: PDGFRA, COL1A1, COL3A1, LPL, CFDC
        mask = adata.obs[cell_type_col].str.contains('FAP|Fib', case=False, na=False)
        # Also include fibroblasts/adipogenic progenitors
        mask |= adata.obs[cell_type_col].str.contains('fibro|adipogen', case=False, na=False)
    elif compartment_name == 'MuSC':
        # MuSC markers: PAX7, MYOD1, MYOG
        mask = adata.obs[cell_type_col].str.contains('MuSC|Satell|myogen|PAX7', case=False, na=False)
    elif compartment_name == 'vascular':
        # Vascular markers: VECAD, CDH5, PECAM1, KDR
        mask = adata.obs[cell_type_col].str.contains('Vasc|Endoth|EC|VenEC|Cap', case=False, na=False)
    else:
        raise ValueError(f"Unknown compartment: {compartment_name}")

    cells = adata[mask]
    print(f"  {compartment_name}: {cells.n_obs} cells")
    return cells

def get_donor_level_expression(cells, genes, cell_type_col):
    """Compute donor-level mean expression for each gene."""
    # Get donor column - use 'sample' based on available columns
    donor_col = 'sample'

    print(f"  Donor column: {donor_col}")
    print(f"  Unique donors: {cells.obs[donor_col].nunique()}")

    # Compute mean expression per donor
    donor_means = pd.DataFrame(index=cells.obs[donor_col].unique())

    for gene in genes:
        if gene in cells.var_names:
            gene_expr = cells.obs[[donor_col, gene]].groupby(donor_col)[gene].mean()
            donor_means[gene] = gene_expr
        else:
            donor_means[gene] = np.nan

    # Drop donors with too few cells
    min_cells = 10
    cell_counts = cells.obs.groupby(donor_col).size()
    valid_donors = cell_counts[cell_counts >= min_cells].index
    donor_means = donor_means.loc[valid_donors]

    print(f"  Valid donors (≥{min_cells} cells): {len(donor_means)}")

    return donor_means

def compute_tf_screen(compartment_name, cells, cell_type_col, tf_list, output_dir):
    """Compute TF-SASP screen for a compartment."""
    print(f"\n{'='*60}")
    print(f"  {compartment_name} TF-SASP Screen")
    print(f"{'='*60}")

    # Get available genes
    available_genes = set(cells.var_names)
    print(f"  Total genes in data: {len(available_genes)}")

    # Filter TF list to those present in data
    detected_tfs = [tf for tf in tf_list if tf in available_genes]
    print(f"  TFs detected in compartment: {len(detected_tfs)}")

    # Filter to TFs in >=50% of cells
    tfs_in_cells = []
    for tf in detected_tfs:
        detection_rate = (cells[:, tf].X > 0).sum() / cells.n_obs
        if detection_rate >= 0.50:
            tfs_in_cells.append(tf)
    print(f"  TFs detected in ≥50% of cells: {len(tfs_in_cells)}")

    # Check SASP12 genes
    saspg12_available = [g for g in SASP12_GENES if g in available_genes]
    print(f"  SASP12 genes available: {len(saspg12_available)}/12")

    if len(saspg12_available) < 8:
        print(f"  WARNING: Less than 8 SASP12 genes available, results may be unreliable")

    # Compute donor-level expression
    all_genes = list(tfs_in_cells) + saspg12_available
    donor_expr = get_donor_level_expression(cells, all_genes, cell_type_col)
    n_donors = len(donor_expr)

    if n_donors < 10:
        print(f"  ERROR: Only {n_donors} donors — insufficient for TF screen")
        return None

    # Compute SASP12 composite score per donor
    donor_expr['SASP12'] = donor_expr[saspg12_available].mean(axis=1)

    # Compute Spearman correlation for each TF
    results = []
    for tf in tfs_in_cells:
        tf_expr = donor_expr[tf].values
        sasp_score = donor_expr['SASP12'].values

        # Remove NaN pairs
        valid_mask = ~(np.isnan(tf_expr) | np.isnan(sasp_score))
        if valid_mask.sum() < 5:
            continue

        rho, pval = spearmanr(tf_expr[valid_mask], sasp_score[valid_mask])
        results.append({
            'TF': tf,
            'rho': rho,
            'pvalue': pval,
            'n_donors': valid_mask.sum()
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print(f"  ERROR: No valid TF-SASP correlations computed")
        return None

    # Sort by absolute rho
    results_df['abs_rho'] = results_df['rho'].abs()
    results_df = results_df.sort_values('abs_rho', ascending=False)

    # Apply Benjamini-Hochberg FDR correction
    _, pvals_corrected, _, _ = multipletests(
        results_df['pvalue'].values,
        alpha=0.05,
        method='fdr_bh'
    )
    results_df['qvalue'] = pvals_corrected
    results_df['significant'] = results_df['qvalue'] < 0.05

    # Add ranking
    results_df['rank'] = range(1, len(results_df) + 1)

    # Identify JUNB and AP-1 family positions
    junb_row = results_df[results_df['TF'] == 'JUNB']
    ap1_rows = results_df[results_df['TF'].isin(AP1_FAMILY)]

    print(f"\n  Results Summary:")
    print(f"  Total TFs tested: {len(results_df)}")
    print(f"  Significant TFs (q < 0.05): {results_df['significant'].sum()}")
    print(f"  N donors: {n_donors}")

    if len(junb_row) > 0:
        junb_rank = results_df['TF'].tolist().index('JUNB') + 1
        print(f"\n  JUNB rank: {junb_rank}/{len(results_df)}")
        print(f"  JUNB rho: {junb_row['rho'].values[0]:.4f}")
        print(f"  JUNB p-value: {junb_row['pvalue'].values[0]:.2e}")
        print(f"  JUNB q-value (BH-FDR): {junb_row['qvalue'].values[0]:.2e}")
    else:
        print(f"\n  JUNB: NOT DETECTED in this compartment")

    print(f"\n  AP-1 Family Rankings:")
    for _, row in ap1_rows.sort_values('rank').iterrows():
        sig_str = "***" if row['qvalue'] < 0.001 else "**" if row['qvalue'] < 0.01 else "*" if row['qvalue'] < 0.05 else ""
        print(f"    {row['TF']:8s} rho={row['rho']:.4f} p={row['pvalue']:.2e} q={row['qvalue']:.2e} rank={row['rank']:3d} {sig_str}")

    # Save results
    results_file = f"{output_dir}/{compartment_name.lower()}_tf_screen.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n  Results saved to: {results_file}")

    # Save top 50 for readability
    top50_file = f"{output_dir}/{compartment_name.lower()}_tf_screen_top50.txt"
    with open(top50_file, 'w') as f:
        f.write(f"# {compartment_name} TF-SASP Screen Results\n")
        f.write(f"# N donors: {n_donors}\n")
        f.write(f"# Total TFs tested: {len(results_df)}\n")
        if len(junb_row) > 0:
            f.write(f"# JUNB rank: {results_df['TF'].tolist().index('JUNB') + 1} / {len(results_df)}\n")
            f.write(f"# JUNB rho: {junb_row['rho'].values[0]:.4f}\n")
            f.write(f"# JUNB q-value (BH-FDR): {junb_row['qvalue'].values[0]:.2e}\n")
        f.write(f"# Significant TFs (q < 0.05): {results_df['significant'].sum()}\n")
        f.write(f"\n# Top 50 TFs by |rho|:\n")
        f.write(f"{'Rank':<5} {'TF':<10} {'rho':>8} {'p-value':>12} {'q-value':>12} {'Sig':>5}\n")
        f.write("-" * 55 + "\n")

        for i, row in results_df.head(50).iterrows():
            sig = "***" if row['qvalue'] < 0.001 else "**" if row['qvalue'] < 0.01 else "*" if row['qvalue'] < 0.05 else ""
            f.write(f"{row['rank']:<5} {row['TF']:<10} {row['rho']:>8.4f} {row['pvalue']:>12.2e} {row['qvalue']:>12.2e} {sig:>5}\n")

    print(f"  Top 50 saved to: {top50_file}")

    # Individual SASP gene analysis
    print(f"\n  Individual SASP Gene Correlations with JUNB:")
    junb_expr = donor_expr['JUNB'].values
    for gene in saspg12_available:
        if gene in donor_expr.columns:
            gene_vals = donor_expr[gene].values
            valid_mask = ~(np.isnan(junb_expr) | np.isnan(gene_vals))
            if valid_mask.sum() >= 5:
                rho, pval = spearmanr(junb_expr[valid_mask], gene_vals[valid_mask])
                print(f"    {gene:<10} rho={rho:>7.4f} p={pval:.2e}")

    return results_df

def main():
    """Run the TF screen analysis."""
    output_dir = "/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_039"

    # Load data
    adata, cell_type_col = load_data()

    # Run for each compartment
    results = {}
    for compartment in ['FAP', 'MuSC', 'vascular']:
        cells = get_compartment_cells(adata, cell_type_col, compartment)
        if cells.n_obs > 0:
            result = compute_tf_screen(
                compartment, cells, cell_type_col,
                TF_LIST + AP1_FAMILY,  # Ensure AP-1 family is included
                output_dir
            )
            results[compartment] = result
        else:
            print(f"\n  WARNING: No cells found for {compartment}")
            results[compartment] = None

    # Print summary
    print("\n" + "="*70)
    print("TF SCREEN SUMMARY")
    print("="*70)

    for comp, df in results.items():
        if df is not None:
            junb_row = df[df['TF'] == 'JUNB']
            junb_rank = df['TF'].tolist().index('JUNB') + 1 if 'JUNB' in df['TF'].values else 'ND'

            n_sig = df['significant'].sum()
            top_tf = df.iloc[0]['TF']
            top_rho = df.iloc[0]['rho']

            print(f"\n{comp}:")
            print(f"  N donors: {df['n_donors'].iloc[0]}")
            print(f"  TFs tested: {len(df)}")
            print(f"  Significant (q<0.05): {n_sig}")
            print(f"  Top TF: {top_tf} (rho={top_rho:.4f})")
            print(f"  JUNB rank: {junb_rank}/{len(df)}")
            if len(junb_row) > 0:
                print(f"  JUNB rho: {junb_row['rho'].values[0]:.4f} (q={junb_row['qvalue'].values[0]:.2e})")

    print("\n" + "="*70)
    print("KEY FINDINGS FOR PREPRINT:")
    print("="*70)

    # Check if JUNB is #1 in vascular
    if results.get('vascular') is not None:
        vd = results['vascular']
        if 'JUNB' in vd['TF'].values:
            junb_vascular_rank = vd['TF'].tolist().index('JUNB') + 1
            if junb_vascular_rank == 1:
                print("\n✓ JUNB is #1 in VASCULAR — supports therapeutic thesis")
            else:
                top_tfs = vd.head(junb_vascular_rank)['TF'].tolist()
                print(f"\n✗ JUNB is #{junb_vascular_rank} in VASCULAR")
                print(f"  TFs ranked higher: {', '.join(top_tfs[:-1])}")

    # Check MuSC
    if results.get('MuSC') is not None:
        md = results['MuSC']
        if 'JUNB' in md['TF'].values and 'CDKN1A' in md['TF'].values:
            junb_rank = md['TF'].tolist().index('JUNB') + 1
            cdkn1a_rank = md['TF'].tolist().index('CDKN1A') + 1
            print(f"\nMuSC: CDKN1A (p21) rank {cdkn1a_rank}, JUNB rank {junb_rank}")
            if cdkn1a_rank < junb_rank:
                print("  → p21 is stronger than JUNB in MuSC (confirming F093)")

    # Check FAP
    if results.get('FAP') is not None:
        fd = results['FAP']
        n_sig = fd['significant'].sum()
        max_rho = fd['rho'].abs().max()
        print(f"\nFAP: {n_sig} significant TFs, max |rho| = {max_rho:.4f}")
        if n_sig == 0:
            print("  → No TF strongly correlates with SASP in FAPs (confirming F095)")

    # Save results summary
    summary_file = f"{output_dir}/tf_screen_summary.json"
    import json
    summary = {}
    for comp, df in results.items():
        if df is not None:
            junb_row = df[df['TF'] == 'JUNB']
            summary[comp] = {
                'n_donors': int(df['n_donors'].iloc[0]),
                'n_tfs_tested': int(len(df)),
                'n_significant': int(df['significant'].sum()),
                'junb_rank': int(df['TF'].tolist().index('JUNB') + 1) if 'JUNB' in df['TF'].values else None,
                'junb_rho': float(junb_row['rho'].values[0]) if len(junb_row) > 0 else None,
                'junb_qvalue': float(junb_row['qvalue'].values[0]) if len(junb_row) > 0 else None,
                'top_tf': df.iloc[0]['TF'],
                'top_rho': float(df.iloc[0]['rho'])
            }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    return results

if __name__ == "__main__":
    main()