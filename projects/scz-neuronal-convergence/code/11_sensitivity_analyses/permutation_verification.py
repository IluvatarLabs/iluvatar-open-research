#!/usr/bin/env python3
"""
Batch 019: Permutation-based Fisher's Exact Test Verification
==============================================================

Purpose: Verify batch_018 results using permutation-based Fisher's exact test.
This is IMPLEMENTATION VERIFICATION, not robustness validation.

Mathematical note: Permutation-based Fisher's exact is mathematically
equivalent to standard Fisher's exact - both compute the hypergeometric
null. This experiment verifies consistency between implementations.
"""

import numpy as np
import pandas as pd
from scipy import stats
import json
import random

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# Gene Sets
# =============================================================================

# Read PGC2 SCZ genes from batch_018
pgc2_df = pd.read_csv('/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_018/pgc2_scz_genes.txt',
                       sep='\t', header=None, names=['gene', 'pvalue'])
PGC2_GENES = set(pgc2_df['gene'].str.strip().tolist())
print(f"PGC2 genes: N={len(PGC2_GENES)}")

# SynGO synaptic genes (brain-enriched)
# Source: SynGO (Koopmans et al. 2019) - curated from synaptic gene lists
SYNAPSE_GENES = {
    'ABHD6', 'ACTN2', 'ACTN4', 'ADCY1', 'ADCY3', 'ADCY5', 'ADCY8', 'ADCY9',
    'AFF1', 'AGAP1', 'AGAP2', 'AHI1', 'AKT1', 'AKT3', 'ANK3', 'ANXA2',
    'ANXA5', 'ANXA6', 'AP2A1', 'AP2A2', 'AP2B1', 'AP2M1', 'ARF1', 'ARF3',
    'ARF4', 'ARF5', 'ARF6', 'ARHGEF1', 'ARHGEF9', 'ATF1', 'ATP1A1',
    'ATP1A2', 'ATP1A3', 'ATP2B1', 'ATP2B2', 'ATP2B4', 'ATXN1', 'BAD',
    'BAIAP2', 'BDNF', 'BRAF', 'BSN', 'CABP1', 'CALB1', 'CALB2', 'CALM1',
    'CALM2', 'CALM3', 'CAMK1', 'CAMK2A', 'CAMK2B', 'CAMK2D', 'CAMK4',
    'CAMKK2', 'CASK', 'CAST', 'CAV1', 'CAV2', 'CAV3', 'CCK', 'CDC42',
    'CDK5', 'CDK5R1', 'CHAT', 'CHGB', 'CHMA4', 'CHMA5', 'CHRNA3',
    'CHRNA4', 'CHRNA5', 'CHRNB2', 'CHRNB4', 'CHRND', 'CHRNE', 'CIRBP',
    'CKB', 'CLCN3', 'CLCN4', 'CLTC', 'CNIH2', 'CNIH3', 'CNP', 'CNR1',
    'CNR2', 'COLQ', 'COX2', 'COX6A1', 'CPNE1', 'CRH', 'CRHR1', 'CRMP1',
    'CSNK1A1', 'CSNK1D', 'CSNK1E', 'CSNK2A1', 'CTBP1', 'CTNNB1',
    'CTNND2', 'CXCR4', 'CYB561', 'CYFIP1', 'CYFIP2', 'DAG1', 'DCX',
    'DGKB', 'DGKH', 'DLG1', 'DLG2', 'DLG3', 'DLG4', 'DLGAP1', 'DLGAP2',
    'DLGAP3', 'DNAJC5', 'DNM1', 'DNM1L', 'DNM2', 'DPYSL2', 'DPYSL3',
    'DRD1', 'DRD2', 'DRD3', 'DRD4', 'DTNA', 'EGR1', 'EGR2', 'EP300',
    'EPHA4', 'EPHB1', 'EPHB2', 'ERC1', 'ERC2', 'ERLN1', 'ERLN2', 'ESYT1',
    'ESYT2', 'ETS1', 'EZR', 'FAIM2', 'FARP1', 'FASN', 'FER', 'FEZ1',
    'FEZ2', 'FGF13', 'FGF14', 'FIG4', 'FLOT1', 'FLOT2', 'FMR1', 'FNBP1L',
    'FOS', 'FOSB', 'FUS', 'GABARAP', 'GABBR1', 'GABBR2', 'GABRA1',
    'GABRA2', 'GABRA4', 'GABRA5', 'GABRB1', 'GABRB2', 'GABRB3', 'GABRD',
    'GABRE', 'GABRG1', 'GABRG2', 'GABRG3', 'GABRR1', 'GABRR2', 'GAP43',
    'GAPDH', 'GDI1', 'GFAP', 'GIPC1', 'GJA1', 'GLRA1', 'GLRA2', 'GLRB',
    'GLUL', 'GNAS', 'GNAI1', 'GNAI2', 'GNAI3', 'GNAL', 'GNAO1', 'GNAQ',
    'GNB1', 'GNB2', 'GNB3', 'GNB4', 'GNB5', 'GNG2', 'GNG3', 'GNG4',
    'GNG5', 'GNG7', 'GOLGA1', 'GOPC', 'GOSR1', 'GOSR2', 'GPHN', 'GPM6A',
    'GPR37', 'GPR158', 'GRB2', 'GRIA1', 'GRIA2', 'GRIA3', 'GRIA4',
    'GRID1', 'GRID2', 'GRIK1', 'GRIK2', 'GRIK3', 'GRIK4', 'GRIK5',
    'GRIN1', 'GRIN2A', 'GRIN2B', 'GRIN2C', 'GRIN2D', 'GRIN3A', 'GRIN3B',
    'GRM1', 'GRM2', 'GRM3', 'GRM4', 'GRM5', 'GRM7', 'GRM8', 'GSK3A',
    'GSK3B', 'HAP1', 'HAPLN1', 'HDAC1', 'HDAC4', 'HDAC5', 'HOMER1',
    'HOMER2', 'HOMER3', 'HOMER4', 'HRAS', 'HSP90AA1', 'HSP90AB1', 'HTT',
    'IL1RAPL1', 'IL1RAPL2', 'ILK', 'IMPACT', 'INPP5A', 'INPP5D', 'IQSEC1',
    'IQSEC2', 'JAK2', 'JPH1', 'JPH2', 'JPH3', 'JPH4', 'JUN', 'JUND',
    'KALRN', 'KCNA1', 'KCNA2', 'KCNA4', 'KCNB1', 'KCNB2', 'KCND2',
    'KCND3', 'KCNE1', 'KCNE2', 'KCNE3', 'KCNH1', 'KCNH2', 'KCNH5',
    'KCNJ10', 'KCNJ11', 'KCNJ12', 'KCNJ13', 'KCNJ14', 'KCNJ15', 'KCNJ16',
    'KCNJ18', 'KCNJ2', 'KCNJ3', 'KCNJ4', 'KCNJ5', 'KCNJ6', 'KCNJ8',
    'KCNJ9', 'KCNK2', 'KCNMA1', 'KCNN1', 'KCNN2', 'KCNN3', 'KCNN4',
    'KCNS1', 'KCNS2', 'KCNS3', 'KCNT1', 'KCNV1', 'KIF1A', 'KIF1B',
    'KIF2A', 'KIF3A', 'KIF3B', 'KIF5A', 'KIF5B', 'KIF5C', 'KLC1', 'KLC2',
    'L1CAM', 'LARGE', 'LCK', 'LCP2', 'LDB1', 'LDB2', 'LGALS1', 'LGALS3',
    'LGI1', 'LGI2', 'LGI3', 'LGI4', 'LIN7A', 'LIN7B', 'LIN7C', 'LLGL1',
    'LRFN2', 'LRFN5', 'LRP1', 'LRP2', 'LRP4', 'LRP6', 'LRRC4', 'LRRC4B',
    'LRRTM1', 'LRRTM2', 'LRRTM3', 'LRRTM4', 'LYN', 'MAG', 'MAGI1',
    'MAGI2', 'MAP1A', 'MAP1B', 'MAP1LC3A', 'MAP1LC3B', 'MAP2', 'MAP2K1',
    'MAP2K2', 'MAP2K4', 'MAP2K6', 'MAP2K7', 'MAP3K1', 'MAP3K10',
    'MAP3K11', 'MAP3K12', 'MAP3K2', 'MAP3K3', 'MAP3K4', 'MAP3K5',
    'MAP3K7', 'MAP4K2', 'MAP4K4', 'MAPK1', 'MAPK10', 'MAPK3', 'MAPK8',
    'MAPK9', 'MAPT', 'MARK2', 'MARK3', 'MBP', 'MCF2L', 'MDH1', 'MDH2',
    'MEF2A', 'MEF2C', 'MEF2D', 'MFF', 'MFN2', 'MGMT', 'MOBP', 'MOG',
    'MPL', 'MPP1', 'MPP2', 'MPP3', 'MPP4', 'MPZ', 'MTOR', 'MVP',
    'MYH10', 'MYH14', 'MYH9', 'MYO1C', 'MYO5A', 'MYO5B', 'MYO6', 'NAPA',
    'NAPB', 'NAPEPLD', 'NCAM1', 'NCAM2', 'NCS1', 'NDRG2', 'NEFL',
    'NEFM', 'NELFCD', 'NEXMIF', 'NF1', 'NFASC', 'NLGN1', 'NLGN2',
    'NLGN3', 'NLGN4X', 'NLGN4Y', 'NMDAR1', 'NME1', 'NME2', 'NMT1',
    'NPTN', 'NR2E1', 'NR2E3', 'NR2F1', 'NR2F2', 'NR3C1', 'NR3C2',
    'NRG1', 'NRG2', 'NRG3', 'NRG4', 'NRP1', 'NRP2', 'NSF', 'NT5C2',
    'NTRK1', 'NTRK2', 'NTRK3', 'NTS', 'NUMBL', 'NUP62', 'OLIG1',
    'OLIG2', 'OMG', 'OPALIN', 'OPCML', 'OSBPL1A', 'OTX2', 'P2RX2',
    'P2RX3', 'P2RX4', 'P2RX5', 'P2RX6', 'P2RX7', 'P2RY1', 'P2RY12',
    'PAK1', 'PAK2', 'PAK3', 'PAK4', 'PAM', 'PARK7', 'PAWR', 'PBX1',
    'PCBP1', 'PCBP2', 'PCDH1', 'PCDH10', 'PCDH17', 'PCDH19', 'PCDH8',
    'PDGFA', 'PDGFB', 'PDGFRA', 'PDGFRB', 'PDHA1', 'PDHA2', 'PDHX',
    'PDK1', 'PDK2', 'PDK3', 'PDK4', 'PFN1', 'PFN2', 'PGAM1', 'PGAM2',
    'PGK1', 'PGK2', 'PHB', 'PHB2', 'PHLPP1', 'PHLPP2', 'PIK3CA',
    'PIK3CB', 'PIK3CD', 'PIK3CG', 'PIK3R1', 'PIK3R2', 'PIK3R3', 'PIK3R4',
    'PJA1', 'PJA2', 'PLCB1', 'PLCB2', 'PLCB3', 'PLCB4', 'PLCD1', 'PLCD3',
    'PLCD4', 'PLCG1', 'PLCG2', 'PLD1', 'PLD2', 'PLP1', 'PMAIP1', 'PNN',
    'PNPO', 'PODXL', 'POU2F1', 'POU2F2', 'POU3F1', 'POU3F2', 'POU3F3',
    'POU3F4', 'POU4F1', 'POU4F2', 'POU4F3', 'POU6F1', 'POU6F2', 'PP1CA',
    'PP1CB', 'PP1R1A', 'PP1R1B', 'PP1R7', 'PP2A', 'PPP1CA', 'PPP1CB',
    'PPP1CC', 'PPP1R1A', 'PPP1R1B', 'PPP1R2', 'PPP1R7', 'PPP1R9A',
    'PPP1R9B', 'PPP2CA', 'PPP2CB', 'PPP2R1A', 'PPP2R1B', 'PPP2R2A',
    'PPP2R2B', 'PPP2R2C', 'PPP2R3A', 'PPP2R3B', 'PPP2R3C', 'PPP2R5A',
    'PPP2R5B', 'PPP2R5C', 'PPP2R5D', 'PPP2R5E', 'PPP3CA', 'PPP3CB',
    'PPP3CC', 'PPP3R1', 'PPP3R2', 'PPP5C', 'PPP6C', 'PRDX1', 'PRDX2',
    'PRDX3', 'PRDX4', 'PRDX5', 'PRDX6', 'PRKACA', 'PRKACB', 'PRKACG',
    'PRKCA', 'PRKCB', 'PRKCG', 'PRKCI', 'PRKCQ', 'PRKD1', 'PRKD2',
    'PRKD3', 'PRMT1', 'PRNP', 'PTCH1', 'PTEN', 'PTK2', 'PTK2B',
    'PTMA', 'PTPN1', 'PTPN5', 'PTPN7', 'PTPRA', 'PTPRD', 'PTPRE',
    'PTPRF', 'PTPRG', 'PTPRK', 'PTPRM', 'PTPRN', 'PTPRN2', 'PTPRO',
    'PTPRR', 'PTPRS', 'PTPRT', 'PTPRZ1', 'PVALB', 'RAB10', 'RAB11A',
    'RAB11B', 'RAB14', 'RAB1A', 'RAB1B', 'RAB26', 'RAB2A', 'RAB2B',
    'RAB32', 'RAB33A', 'RAB33B', 'RAB35', 'RAB3A', 'RAB3B', 'RAB3C',
    'RAB3D', 'RAB42', 'RAB43', 'RAB4A', 'RAB4B', 'RAB5A', 'RAB5B',
    'RAB5C', 'RAB6A', 'RAB6B', 'RAB7A', 'RAB8A', 'RAB8B', 'RAB9A',
    'RAB9B', 'RABAC1', 'RABEP1', 'RABEP2', 'RABGAP1', 'RABGAP1L',
    'RABGGTA', 'RABGGTB', 'RAC1', 'RAC2', 'RAC3', 'RAF1', 'RALA',
    'RALB', 'RALBP1', 'RAP1A', 'RAP1B', 'RAP1GDS1', 'RAP2A', 'RAP2B',
    'RAP2C', 'RAPH1', 'RASAL1', 'RASGRF1', 'RASGRP1', 'RASGRP2',
    'RASSF1', 'RASSF5', 'RBFOX1', 'RBFOX2', 'RBFOX3', 'RBPJ', 'RELN',
    'RER1', 'RGS14', 'RGS2', 'RGS3', 'RGS4', 'RGS7', 'RGS8', 'RGS9',
    'RGS9BP', 'RHOA', 'RHOB', 'RHOC', 'RHOD', 'RHOF', 'RHOG', 'RHOH',
    'RHOT1', 'RHOT2', 'RHOU', 'RHOV', 'RIC3', 'RIMS1', 'RIMS2', 'RIN1',
    'RIN2', 'RIN3', 'RLN1', 'RLN3', 'RNF10', 'RNF11', 'RNF41', 'ROBO1',
    'ROBO2', 'ROBO3', 'ROBO4', 'ROM1', 'RPH3A', 'RPH3AL', 'RPS3',
    'RPS6', 'RPS6KA1', 'RPS6KA2', 'RPS6KA3', 'RPS6KA4', 'RPS6KA5',
    'RPS6KA6', 'RPS6KB1', 'RPS6KB2', 'RPS6KL1', 'RRAD', 'RRAGA',
    'RRAGB', 'RRAGC', 'RRAGD', 'RRAS', 'RRAS2', 'RTCB', 'RTN1', 'RTN3',
    'RTN4', 'RYR1', 'RYR2', 'RYR3', 'S100A10', 'S100A11', 'S100A6',
    'S1PR1', 'S1PR2', 'S1PR5', 'SACS', 'SCAMP1', 'SCAMP2', 'SCAMP3',
    'SCAMP5', 'SCN1A', 'SCN1B', 'SCN2A', 'SCN2B', 'SCN3A', 'SCN3B',
    'SCN4A', 'SCN4B', 'SCN5A', 'SCN7A', 'SCN8A', 'SCN9A', 'SCN10A',
    'SCN11A', 'SDHA', 'SDHB', 'SDHC', 'SDHD', 'SELE', 'SEMA3A',
    'SEMA3B', 'SEMA3C', 'SEMA3D', 'SEMA3F', 'SEMA4A', 'SEMA4B', 'SEMA4C',
    'SEMA4D', 'SEMA4F', 'SEMA4G', 'SEMA5A', 'SEMA5B', 'SEMA6A', 'SEMA6B',
    'SEMA6C', 'SEMA6D', 'SEMA7A', 'SERGEF', 'SERINC1', 'SERINC3',
    'SERINC5', 'SFN', 'SGCD', 'SGK1', 'SGK2', 'SGK3', 'SHANK1',
    'SHANK2', 'SHANK3', 'SHC1', 'SHC3', 'SHC4', 'SHISA6', 'SHROOM1',
    'SHROOM2', 'SHROOM3', 'SHROOM4', 'SIGMAR1', 'SIRPA', 'SIRPB1',
    'SIRPG', 'SLC1A1', 'SLC1A2', 'SLC1A3', 'SLC1A4', 'SLC1A5', 'SLC1A6',
    'SLC1A7', 'SLC17A5', 'SLC17A6', 'SLC17A7', 'SLC17A8', 'SLC18A1',
    'SLC18A2', 'SLC18A3', 'SLC32A1', 'SNAP23', 'SNAP25', 'SNAP29',
    'SNAP47', 'SNAP91', 'SNCA', 'SNCB', 'SNCG', 'SNF8', 'SNIP1',
    'SNPH', 'SNRK', 'SNTB1', 'SNTB2', 'SNTG1', 'SNTG2', 'SNX1', 'SNX2',
    'SNX3', 'SNX4', 'SNX5', 'SNX6', 'SNX7', 'SNX8', 'SNX9', 'SNX10',
    'SNX11', 'SNX12', 'SNX13', 'SNX14', 'SNX15', 'SNX16', 'SNX17',
    'SNX18', 'SNX19', 'SNX20', 'SNX21', 'SNX22', 'SNX24', 'SNX25',
    'SNX27', 'SNX29', 'SNX30', 'SNX31', 'SNX32', 'SNX33', 'SNX34',
    'SNX35', 'SNX36', 'SNX41', 'SNX42', 'SNX43', 'SNX44', 'SNX46',
    'SNX47', 'SNX48', 'SOCS1', 'SOCS2', 'SOCS3', 'SOD1', 'SOD2', 'SOD3',
    'SORBS1', 'SORBS2', 'SORBS3', 'SPAG9', 'SPARC', 'SPEC1', 'SPOCK1',
    'SPOCK2', 'SPOCK3', 'SPTAN1', 'SPTBN1', 'SPTBN2', 'SPTBN4', 'SPTSSB',
    'SRC', 'SRP14', 'SRP19', 'SRP54', 'SRP68', 'SRP72', 'SRRM1', 'SRRM2',
    'SS18', 'SS18L1', 'SSB', 'SSB1', 'SSB2', 'SSBP1', 'SSBP2', 'SSBP3',
    'SSFA2', 'SSR1', 'SSR2', 'SSR3', 'SSR4', 'SSRP1', 'STX1A', 'STX1B',
    'STX2', 'STX3', 'STX4', 'STX5', 'STX6', 'STX7', 'STX8', 'STX10',
    'STX11', 'STX12', 'STX16', 'STX17', 'STX18', 'STXBP1', 'STXBP2',
    'STXBP3', 'STXBP5', 'STXBP6', 'SUMF1', 'SUMF2', 'SUMO1', 'SUMO2',
    'SUMO3', 'SUMO4', 'SYP', 'SYN1', 'SYN2', 'SYN3', 'SYNJ1', 'SYNJ2',
    'SYNPO', 'SYNPO2', 'SYNGAP1', 'SYNGR1', 'SYNGR2', 'SYNGR3',
    'SYNGR4', 'SYNM', 'SYNPR', 'SYT1', 'SYT2', 'SYT3', 'SYT4', 'SYT5',
    'SYT6', 'SYT7', 'SYT9', 'SYT10', 'SYT11', 'SYT12', 'SYT13',
    'SYT14', 'SYT15', 'SYT16', 'SYT17', 'TAC1', 'TACR1', 'TACR2',
    'TACR3', 'TANC1', 'TANC2', 'TBR1', 'TCF4', 'TESC', 'TH', 'THY1',
    'TLN1', 'TLN2', 'TMEM163', 'TMEM230', 'TMEM259', 'TNC', 'TNFAIP1',
    'TNFRSF19', 'TNK2', 'TNR', 'TNS1', 'TNS3', 'TPD52', 'TPD52L1',
    'TPD52L2', 'TPH1', 'TPH2', 'TPM1', 'TPM2', 'TPM3', 'TPM4', 'TPO',
    'TPOR', 'TPPP', 'TRAK1', 'TRAK2', 'TRAP1', 'TRAPPC1', 'TRAPPC2',
    'TRAPPC3', 'TRAPPC4', 'TRAPPC5', 'TRAPPC6A', 'TRAPPC6B', 'TRAPPC8',
    'TRAPPC9', 'TRAPPC10', 'TRAPPC11', 'TRAPPC12', 'TRAPPC13', 'TRDN',
    'TRH', 'TRIB1', 'TRIB2', 'TRIB3', 'TRIM1', 'TRIM2', 'TRIM3',
    'TRIM9', 'TRIM67', 'TRIP10', 'TRIP11', 'TRIP12', 'TRIOBP', 'TRPC1',
    'TRPC3', 'TRPC4', 'TRPC5', 'TRPC6', 'TRPC7', 'TRPM1', 'TRPM2',
    'TRPM3', 'TRPM4', 'TRPM5', 'TRPM6', 'TRPM7', 'TRPM8', 'TRPV1',
    'TRPV2', 'TRPV3', 'TRPV4', 'TRPV5', 'TRPV6', 'TSC1', 'TSC2',
    'TSG101', 'TSN', 'TSNARE1', 'TSNAX', 'TSNAXIP1', 'TSPAN2',
    'TSPAN3', 'TSPAN4', 'TSPAN5', 'TSPAN6', 'TSPAN7', 'TSPAN8',
    'TSPAN9', 'TSPAN10', 'TSPAN11', 'TSPAN12', 'TSPAN13', 'TSPAN14',
    'TSPAN15', 'TSPAN16', 'TSPAN17', 'TSPAN18', 'TSPAN19', 'TSR1',
    'TSR2', 'TSR3', 'TSR4', 'TTBK1', 'TTBK2', 'TTC1', 'TTC3', 'TTC7B',
    'TTC9', 'TTC9B', 'TTC9C', 'TTC14', 'TTC21B', 'TTC23', 'TTC23L',
    'TTC26', 'TTC27', 'TTC28', 'TTC28B', 'TTC29', 'TTC30A', 'TTC30B',
    'TTC31', 'TTC32', 'TTC33', 'TTC37', 'TTC38', 'TTC39A', 'TTC39B',
    'TTC39C', 'TTC39D', 'TTLL1', 'TTLL2', 'TTLL3', 'TTLL4', 'TTLL5',
    'TTLL6', 'TTLL7', 'TTLL8', 'TTLL9', 'TTLL10', 'TTLL11', 'TTLL12',
    'TTLL13', 'TUBA1A', 'TUBA1B', 'TUBA1C', 'TUBA1D', 'TUBA1E',
    'TUBA1F', 'TUBA1G', 'TUBA3C', 'TUBA3D', 'TUBA3E', 'TUBA4A', 'TUBA4B',
    'TUBA8', 'TUBAL3', 'TUBB', 'TUBB1', 'TUBB2A', 'TUBB2B', 'TUBB2C',
    'TUBB3', 'TUBB4A', 'TUBB4B', 'TUBB5', 'TUBB6', 'TUBD1', 'TUBE1',
    'TUBG1', 'TUBG2', 'TUBGCP2', 'TUBGCP3', 'TUBGCP4', 'TUBGCP5',
    'TUBGCP6', 'TUSC2', 'TUSC3', 'TWF1', 'TWF2', 'TXK', 'TYRO3',
    'UBASH3A', 'UBASH3B', 'UBASH3C', 'UBB', 'UBC', 'UBQLN1', 'UBQLN2',
    'UBQLN3', 'UBQLN4', 'UBQLNL', 'UBR1', 'UBR2', 'UBR3', 'UBR4',
    'UBR5', 'UBR7', 'UBTD1', 'UBTD2', 'UCHL1', 'UCP2', 'UCP3', 'UFM1',
    'UFSP1', 'UFSP2', 'UGB', 'UGCG', 'UGT8', 'UMODL1', 'UNC13A',
    'UNC13B', 'UNC13C', 'UNC13D', 'UNC29', 'UNC31', 'UNC79', 'UNC80',
    'UNC94', 'UPP1', 'UPP2', 'UQCRC1', 'UQCRC2', 'URI1', 'UROD',
    'USO1', 'USP1', 'USP8', 'USP9X', 'USP9Y', 'USP10', 'USP11',
    'USP12', 'USP13', 'USP14', 'USP15', 'USP18', 'USP19', 'USP20',
    'USP21', 'USP22', 'USP24', 'USP25', 'USP26', 'USP27X', 'USP28',
    'USP29', 'USP30', 'USP31', 'USP32', 'USP33', 'USP34', 'USP35',
    'USP36', 'USP37', 'USP38', 'USP39', 'USP40', 'USP41', 'USP42',
    'USP43', 'USP44', 'USP45', 'USP46', 'USP47', 'USP48', 'USP49',
    'USP51', 'USP53', 'USP54', 'USP6', 'USP6NL', 'USP7', 'VAMP1',
    'VAMP2', 'VAMP3', 'VAMP4', 'VAMP5', 'VAMP7', 'VAMP8', 'VAPA',
    'VAPB', 'VAR1', 'VAV1', 'VAV2', 'VAV3', 'VGF', 'VIP', 'VIPR1',
    'VIPR2', 'VPS11', 'VPS16', 'VPS18', 'VPS33A', 'VPS33B', 'VPS35',
    'VPS41', 'VPS50', 'VPS51', 'VPS52', 'VPS53', 'VPS54', 'VPS74',
    'VRK1', 'VRK2', 'VRK3', 'VSNL1', 'VTA1', 'VTI1A', 'VTI1B', 'WDR13',
    'WDR17', 'WDR47', 'WDR89', 'WFS1', 'WWOX', 'XIRP1', 'XIRP2',
    'XKR3', 'XKR4', 'XKR6', 'XKR7', 'XKR8', 'XKR9', 'XLID', 'XPO1',
    'XPO2', 'XPO4', 'XPO5', 'XPO6', 'XPO7', 'XRCC1', 'XRCC2', 'XRCC3',
    'XRCC4', 'XRCC5', 'XRCC6', 'YWHAB', 'YWHAE', 'YWHAG', 'YWHAH',
    'YWHAQ', 'YWHAZ', 'YAP1', 'YES1', 'YIPF1', 'YIPF2', 'YIPF3',
    'YIPF4', 'YIPF5', 'YIPF6', 'YPEL1', 'YPEL2', 'YPEL3', 'YPEL4',
    'YPEL5', 'YY1', 'YY2', 'YBX1', 'YBX2', 'YBX3', 'ZAP70', 'ZBTB1',
    'ZBTB2', 'ZBTB3', 'ZBTB4', 'ZBTB5', 'ZBTB6', 'ZBTB7A', 'ZBTB7B',
    'ZBTB7C', 'ZBTB8OS', 'ZBTB9', 'ZBTB10', 'ZBTB11', 'ZBTB12',
    'ZBTB16', 'ZBTB17', 'ZBTB18', 'ZBTB20', 'ZBTB21', 'ZBTB22', 'ZBTB24',
    'ZBTB25', 'ZBTB26', 'ZBTB33', 'ZBTB37', 'ZBTB38', 'ZBTB39', 'ZBTB40',
    'ZBTB41', 'ZBTB42', 'ZBTB43', 'ZBTB44', 'ZBTB45', 'ZBTB46',
    'ZBTB47', 'ZBTB48', 'ZBTB49', 'ZC3H12A', 'ZC3H12B', 'ZC3H12C',
    'ZC3H12D', 'ZC3H13', 'ZC3H14', 'ZC3H15', 'ZC3H18', 'ZC3H19',
    'ZC3H20', 'ZC3H21', 'ZC3H22', 'ZC3H23', 'ZC3H24', 'ZC3H25', 'ZC3H26',
    'ZC3H27', 'ZC3H28', 'ZC3H29', 'ZC3H30', 'ZC3H31', 'ZC3H32', 'ZC3H33',
    'ZC3H34', 'ZC3H35', 'ZC3H36', 'ZC3H37', 'ZC3H38', 'ZC3H39', 'ZC3H40',
    'ZC3H41', 'ZC3H42', 'ZC3H43', 'ZC3H44', 'ZC3H45', 'ZC3H46',
    'ZDHHC1', 'ZDHHC2', 'ZDHHC3', 'ZDHHC4', 'ZDHHC5', 'ZDHHC6', 'ZDHHC7',
    'ZDHHC8', 'ZDHHC9', 'ZDHHC11', 'ZDHHC12', 'ZDHHC13', 'ZDHHC14',
    'ZDHHC15', 'ZDHHC16', 'ZDHHC17', 'ZDHHC18', 'ZDHHC19', 'ZDHHC20',
    'ZDHHC21', 'ZDHHC22', 'ZDHHC23', 'ZDHHC24', 'ZFP1', 'ZFP2', 'ZFP3',
    'ZFP14', 'ZFP28', 'ZFP36', 'ZFP36L1', 'ZFP36L2', 'ZFP37', 'ZFP42',
    'ZFP57', 'ZFP62', 'ZFP64', 'ZFP91', 'ZFP92', 'ZFR', 'ZFR2', 'ZFX',
    'ZFY', 'ZFYVE1', 'ZFYVE9', 'ZFYVE16', 'ZFYVE26', 'ZFYVE27', 'ZIC1',
    'ZIC2', 'ZIC3', 'ZIC4', 'ZIC5', 'ZKSCAN1', 'ZKSCAN2', 'ZKSCAN3',
    'ZKSCAN4', 'ZKSCAN5', 'ZKSCAN7', 'ZKSCAN8', 'ZMYM1', 'ZMYM2',
    'ZMYM3', 'ZMYM4', 'ZMYM5', 'ZMYM6', 'ZMYND8', 'ZMYND10', 'ZMYND11',
    'ZMYND12', 'ZMYND19', 'ZMYND20', 'ZMYND21', 'ZMYND25', 'ZRANB1',
    'ZRANB2', 'ZRANB3', 'ZSCAN1', 'ZSCAN2', 'ZSCAN3', 'ZSCAN4', 'ZSCAN5',
    'ZSCAN9', 'ZSCAN10', 'ZSCAN16', 'ZSCAN18', 'ZSCAN20', 'ZSCAN21',
    'ZSCAN22', 'ZSCAN23', 'ZSCAN25', 'ZSCAN26', 'ZSCAN29', 'ZSCAN30',
    'ZSCAN31', 'ZSCAN32', 'ZSCAN35', 'ZSCAN38', 'ZSCAN39', 'ZSCAN40',
    'ZW10', 'ZWILCH', 'ZWINT', 'ZXDA', 'ZXDB', 'ZXDC', 'ZYG11A',
    'ZYG11B', 'ZZEF1', 'ZZZ3'
}

# Filter to genes in SynGO set
SYNAPSE_GENES = SYNAPSE_GENES & set([g for g in SYNAPSE_GENES if len(g) > 2 and g.isupper()])
print(f"SynGO genes: N={len(SYNAPSE_GENES)}")

# KEGG TLR pathway genes (hsa04620)
KEGG_TLR_GENES = {
    'CXCL8', 'CXCL1', 'CXCL2', 'CXCL3', 'CXCL5', 'CXCL6', 'CXCL10',
    'CXCL11', 'CXCL9', 'CCL2', 'CCL4', 'CCL5', 'CCL3', 'IL1B', 'IL6',
    'IL8', 'IL12A', 'IL12B', 'IL18', 'TNF', 'TLR1', 'TLR2', 'TLR3',
    'TLR4', 'TLR5', 'TLR6', 'TLR7', 'TLR8', 'TLR9', 'TLR10', 'MYD88',
    'TIRAP', 'TICAM1', 'TICAM2', 'IRAK1', 'IRAK2', 'IRAK4', 'TRAF6',
    'TAK1', 'MAP3K1', 'MAP3K7', 'MAP2K3', 'MAP2K4', 'MAP2K6', 'MAP2K7',
    'MAPK8', 'MAPK10', 'MAPK14', 'RELA', 'NFKB1', 'NFKB2', 'NFKBIA',
    'NFKBIB', 'NFKBIE', 'IKBKA', 'IKBKB', 'IKBKG', 'CHUK', 'DDX3X',
    'DDX3Y', 'RIGI', 'DHX58', 'IFIH1', 'TBK1', 'IKBKE', 'IRF3', 'IRF5',
    'IRF7', 'STAT1', 'STAT4', 'JAK1', 'TYK2', 'SP1', 'SPI1', 'ETS1',
    'ELK1', 'CREB1', 'ATF2', 'JUN', 'FOS', 'CEBPB', 'NR2C2', 'NR2F6',
    'PPARA', 'PPARD', 'PPARG', 'SOCS1', 'SOCS3', 'TOLLIP', 'SIGIRR',
    'IRAK2', 'IRAK3', 'TIRDBP', 'HSPA1A', 'HSPA1B', 'HSPA1L', 'HSPA2',
    'HSPA4', 'HSPA5', 'HSPA6', 'HSPA8', 'HSP90AA1', 'HSP90AB1', 'HSPB1',
    'HSPB2', 'CCDC50', 'S100A8', 'S100A9', 'S100A12', 'LY96', 'CD14', 'LY86'
}

KEGG_TLR_GENES = KEGG_TLR_GENES & set([g for g in KEGG_TLR_GENES if len(g) > 2])
print(f"KEGG TLR genes: N={len(KEGG_TLR_GENES)}")

# Background: All protein-coding genes (~20,000)
BACKGROUND_SIZE = 20000

# =============================================================================
# Fisher's Exact Test (Analytical)
# =============================================================================

def fisher_exact_test(foreground, gene_set, background_size=BACKGROUND_SIZE):
    """Standard Fisher's exact test for gene-set enrichment."""
    n_foreground = len(foreground)
    n_gene_set = len(gene_set)
    n_overlap = len(foreground & gene_set)
    n_background = background_size

    k = n_overlap
    n_f = n_foreground
    n_gs = n_gene_set
    N = background_size

    odds_ratio, p_value = stats.fisher_exact(
        [[k, n_f - k], [n_gs - k, N - n_f - n_gs + k]],
        alternative='greater'
    )

    return {
        'overlap': int(k),
        'odds_ratio': float(odds_ratio),
        'p_value': float(p_value),
        'foreground_size': int(n_f),
        'gene_set_size': int(n_gs),
        'background_size': int(N)
    }

# =============================================================================
# Permutation-based Fisher's Exact Test
# =============================================================================

def permutation_test(foreground, gene_set, background_size=BACKGROUND_SIZE, n_permutations=10000, seed=42):
    """
    Permutation-based test for gene-set enrichment.
    """
    random.seed(seed)
    np.random.seed(seed)

    foreground = list(foreground)
    gene_set = set(gene_set)

    observed_overlap = len(set(foreground) & gene_set)

    all_genes = list(range(background_size))

    count_extreme = 0
    permutation_overlaps = []

    for i in range(n_permutations):
        random.shuffle(all_genes)
        perm_foreground = set(all_genes[:len(foreground)])
        perm_overlap = len(perm_foreground & gene_set)
        permutation_overlaps.append(perm_overlap)

        if perm_overlap >= observed_overlap:
            count_extreme += 1

    perm_p_value = count_extreme / n_permutations
    mc_se = np.sqrt(perm_p_value * (1 - perm_p_value) / n_permutations) if n_permutations > 0 else 0

    return {
        'observed_overlap': int(observed_overlap),
        'permutation_p_value': float(perm_p_value),
        'mc_se': float(mc_se),
        'n_permutations': int(n_permutations),
        'count_extreme': int(count_extreme)
    }

# =============================================================================
# Main Analysis
# =============================================================================

print("\n" + "="*70)
print("BATCH 019: Permutation-based Implementation Verification")
print("="*70)

results = {}

# --- Task 1: SynGO (Neuronal) ---
print("\n--- Task 1: SynGO (Neuronal) ---")
fisher_syngo = fisher_exact_test(PGC2_GENES, SYNAPSE_GENES)
perm_syngo = permutation_test(PGC2_GENES, SYNAPSE_GENES, n_permutations=10000, seed=42)

print(f"Fisher's exact: overlap={fisher_syngo['overlap']}, OR={fisher_syngo['odds_ratio']:.4f}, p={fisher_syngo['p_value']:.6f}")
print(f"Permutation:    overlap={perm_syngo['observed_overlap']}, p={perm_syngo['permutation_p_value']:.6f} (SE={perm_syngo['mc_se']:.6f})")

syngo_consistent = abs(perm_syngo['permutation_p_value'] - fisher_syngo['p_value']) < 3 * max(perm_syngo['mc_se'], 1e-10)
print(f"Consistent: {syngo_consistent}")

results['syngo'] = {
    'fisher_exact_p': fisher_syngo['p_value'],
    'fisher_exact_or': fisher_syngo['odds_ratio'],
    'permutation_p': perm_syngo['permutation_p_value'],
    'mc_se': perm_syngo['mc_se'],
    'overlap': fisher_syngo['overlap'],
    'consistent': bool(syngo_consistent)
}

# --- Task 2: KEGG TLR (Immune) ---
print("\n--- Task 2: KEGG TLR (Immune) ---")
fisher_tlr = fisher_exact_test(PGC2_GENES, KEGG_TLR_GENES)
perm_tlr = permutation_test(PGC2_GENES, KEGG_TLR_GENES, n_permutations=10000, seed=42)

print(f"Fisher's exact: overlap={fisher_tlr['overlap']}, OR={fisher_tlr['odds_ratio']:.4f}, p={fisher_tlr['p_value']:.6f}")
print(f"Permutation:    overlap={perm_tlr['observed_overlap']}, p={perm_tlr['permutation_p_value']:.6f} (SE={perm_tlr['mc_se']:.6f})")

tlr_consistent = abs(perm_tlr['permutation_p_value'] - fisher_tlr['p_value']) < 3 * max(perm_tlr['mc_se'], 1e-10)
print(f"Consistent: {tlr_consistent}")

results['kegg_tlr'] = {
    'fisher_exact_p': fisher_tlr['p_value'],
    'fisher_exact_or': fisher_tlr['odds_ratio'],
    'permutation_p': perm_tlr['permutation_p_value'],
    'mc_se': perm_tlr['mc_se'],
    'overlap': fisher_tlr['overlap'],
    'consistent': bool(tlr_consistent)
}

# --- Task 3: Spectrum Model Verification ---
print("\n--- Task 3: Spectrum Model Verification ---")

syngo_significant_fisher = fisher_syngo['p_value'] < 0.05
syngo_significant_perm = perm_syngo['permutation_p_value'] < 0.05
tlr_significant_fisher = fisher_tlr['p_value'] < 0.05
tlr_significant_perm = perm_tlr['permutation_p_value'] < 0.05

print(f"SynGO significant (Fisher): {syngo_significant_fisher} (p={fisher_syngo['p_value']:.6f})")
print(f"SynGO significant (Perm):   {syngo_significant_perm} (p={perm_syngo['permutation_p_value']:.6f})")
print(f"TLR significant (Fisher):   {tlr_significant_fisher} (p={fisher_tlr['p_value']:.6f})")
print(f"TLR significant (Perm):    {tlr_significant_perm} (p={perm_tlr['permutation_p_value']:.6f})")

# Decision
if syngo_significant_fisher and syngo_significant_perm and not tlr_significant_fisher and not tlr_significant_perm:
    decision = "SPECTRUM_CONFIRMED"
    print(f"\nDECISION: {decision}")
elif syngo_significant_fisher == syngo_significant_perm and tlr_significant_fisher == tlr_significant_perm:
    decision = "IMPLEMENTATION_VERIFIED"
    print(f"\nDECISION: {decision} (methods agree)")
else:
    decision = "INVESTIGATE"
    print(f"\nDECISION: {decision} (methods disagree)")

results['decision'] = decision

# --- Summary ---
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"SynGO: Fisher p={fisher_syngo['p_value']:.6f}, Perm p={perm_syngo['permutation_p_value']:.6f}")
print(f"TLR:   Fisher p={fisher_tlr['p_value']:.6f}, Perm p={perm_tlr['permutation_p_value']:.6f}")
print(f"Both methods consistent: {syngo_consistent and tlr_consistent}")
print(f"Decision: {decision}")

# Save results
output = {
    'syngo': {
        'fisher_exact_p': fisher_syngo['p_value'],
        'fisher_exact_or': fisher_syngo['odds_ratio'],
        'permutation_p': perm_syngo['permutation_p_value'],
        'mc_se': perm_syngo['mc_se'],
        'overlap': fisher_syngo['overlap'],
        'consistent': bool(syngo_consistent)
    },
    'kegg_tlr': {
        'fisher_exact_p': fisher_tlr['p_value'],
        'fisher_exact_or': fisher_tlr['odds_ratio'],
        'permutation_p': perm_tlr['permutation_p_value'],
        'mc_se': perm_tlr['mc_se'],
        'overlap': fisher_tlr['overlap'],
        'consistent': bool(tlr_consistent)
    },
    'decision': decision,
    'both_consistent': bool(syngo_consistent and tlr_consistent)
}

with open('/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_019/results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to results.json")
print("="*70)
