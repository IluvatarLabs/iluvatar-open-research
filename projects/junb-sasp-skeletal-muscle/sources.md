# Data Sources — Skeletal Muscle Aging

All analyses use publicly available datasets. No original experimental data were generated.

## Cycle 1

### Primary datasets

| Dataset | Source | Accession / URL | Samples | Used for |
|---------|--------|----------------|---------|----------|
| Human Lifemap Muscle Atlas (HLMA) | Lai et al. 2024 | CNGBdb OMIX004308 | 387,000+ cells/nuclei, 23 donors aged 15-99 | Primary analysis: donor-level TF-SASP correlations, pySCENIC module inference, partial-correlation decomposition, ligand-receptor crosstalk |
| Nature Aging 2024 atlas | Kedlian et al. 2024, Sanger cellxgene | cellxgene portal | 90,902 nuclei, 17 donors | Cross-atlas replication (endothelial and fibroblast compartments) |
| GTEx v8 bulk muscle | GTEx Consortium | gtexportal.org | N=803 | Post-mortem ischemic-time confound triangulation (SMTSISCH covariate). Bulk avenue ultimately closed due to confound structure. |
| OneK1K single-cell eQTL | Yazar et al. | N=982, 14 immune cell types | cis-eQTL validation: AP-1 TFs null at cis-eQTL level, CDKN1A carries significant cis-eQTL (p=3.4e-10) |
| Open Targets / Genetics Portal | genetics.opentargets.org | — | Bulk muscle eQTL colocalizations: 54 TF x sarcopenia/lean-body-mass trait combinations, 0 hits |

### Compartment-specific file details (HLMA)

| Compartment | Donors | Cells | Notes |
|-------------|--------|-------|-------|
| Vascular endothelial | N=23 | 16,157 | After IL6+ venular exclusion |
| FAP (fibro-adipogenic progenitor) | N=22 | 40,389 | N=16 after snRNA-only filter for SCENIC |
| MuSC (muscle stem cell) | N=23 | 9,559 | |
| Immune | N=12 | 13,773 | snRNA-only subset |

### Cross-species reference

| Dataset | Source | Used for |
|---------|--------|----------|
| Tabula Muris Senis | Tabula Muris Consortium | Cross-species AP-1 direction-of-effect comparison (1/5 matches in EC, 2/5 in FAP). Protocol differences preclude definitive species comparison. |

### Software

- pySCENIC 0.12.x (GRNBoost2 + AUCell; cisTarget motif pruning non-functional in this run)
- Python: scipy, scanpy, pandas, numpy, matplotlib
- R: partial correlation, Fisher-z meta-analysis

### Key literature

- Lai et al. 2024. HLMA multimodal cell atlas of ageing human skeletal muscle.
- Kedlian et al. 2024. Nature Aging single-cell atlas of human skeletal muscle.
- Thoma et al. 2020. BML-260 as a DUSP22 modulator protective against skeletal muscle wasting.
- Li et al. 2025. Single-nucleus multiomic study of AP-1 and NF-kB in aged human skeletal muscle. *Nat Commun* 16:6207.

Full corpus in the [project repository](https://github.com/IluvatarLabs/iluvatar-open-research/tree/main/projects/junb-sasp-skeletal-muscle) and on the [project page](https://iluvatarlabs.github.io/iori/junb-sasp-skeletal-muscle/).
