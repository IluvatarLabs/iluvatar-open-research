# Analysis Code

Scripts from the Marvin autonomous research agent's experiment batches, organized into logical analysis steps. Each step contains Python scripts and their corresponding outputs (in `../results/<step>/`).

All scripts reference local data files (`.h5ad` single-cell objects, GTEx matrices, eQTL tables) that must be obtained from the original sources listed in [`../sources.md`](../sources.md). Scripts are published as-is from the experiment runs and have not been modified; hardcoded paths reflect the Marvin workspace layout.

---

## Analysis steps

### 01_data_qc_and_preprocessing
HLMA atlas quality control and GRN viability assessment. Cell/nuclei counts, gene detection rates, and feasibility checks for downstream regulatory network inference.
- **Findings supported:** Prerequisite for all findings

### 02_donor_level_tf_sasp_screen
Unbiased donor-level Spearman correlation screen of transcription factors against SASP composite scores across vascular, FAP, and MuSC compartments. Includes JUNB specificity testing and methodological reviewer responses.
- **Findings supported:** 1, 2, 3

### 03_cross_atlas_replication
Replication of TF-SASP correlations in the independent Nature Aging atlas. Includes vascular JUNB-SASP replication, FAP subtype cross-atlas validation, MuSC cross-atlas analysis, unbiased TF screen in Nature Aging endothelium, and canonical reconciliation of panel discrepancies across prior runs.
- **Findings supported:** 1, 2, 3

### 04_fap_subtype_and_growth_factors
FAP subtype-specific analyses: JUNB-SASP quantification within MME+/CD55+/GPC3+ subtypes, pro-regenerative secretome characterization (FGF7, HGF, IGF1 age trajectories), growth factor compensation sufficiency testing, ligand-receptor crosstalk, and surface marker identification.
- **Findings supported:** 2, 4, 5

### 05_musc_compartment_analysis
MuSC (muscle stem cell) compartment aging analysis. Within-compartment JUNB-SASP correlation, DDR/NF-kB/JUNB pattern extension from FAPs to MuSCs.
- **Findings supported:** 2, 6

### 06_jnk_mapk_pathway_characterization
JNK/MAPK signaling pathway characterization: p38 MAPK transcriptional co-activation with JUNB, MAP2K4/MAP2K7 expression analysis, MAP3K upstream activator profiling.
- **Findings supported:** 7, 8

### 07_p21_cdkn1a_regulatory_hierarchy
CDKN1A/p21 regulatory hierarchy analysis: p21 vs JUNB as vascular senescence biomarkers, cross-compartment p21-SASP generalization, partial-correlation decomposition of the JUNB-p21-SASP cascade.
- **Findings supported:** 6, 7

### 08_scenic_regulon_validation
pySCENIC (GRNBoost2 + AUCell) regulon inference and validation across all three compartments. Includes AUCell-SASP donor-level correlations, CEBPB country stratification, KLF10 driver-bystander validation at the regulon level, differential expression, GSEA, and UMAP visualizations.
- **Findings supported:** 1, 2, 3

### 09_nfkb_and_immune_compartment
NF-kB pathway activity in FAPs (alternative SASP mechanism) and immune compartment TF-SASP analysis. Includes technology adjustment, power analysis, and cross-compartment comparison.
- **Findings supported:** 2

### 10_within_technology_and_panel_sensitivity
Technical sensitivity analyses: within-technology (snRNA-only) replication of vascular JUNB-SASP, SASP gene panel reconciliation across batches, FAP per-gene decomposition with corrected panel, and 10,000-permutation null distribution testing.
- **Findings supported:** 1, 2

### 11_confound_adjustment_and_statistical_rigor
Confound analysis (age, sex, sequencing technology), cell-count sensitivity, power analysis, continuous age regression, replication tiering across datasets, and QC metrics compilation.
- **Findings supported:** 1, 2, 7

### 12_partial_correlations_and_mediation
Joint partial correlations adjusting for sex, technology, and age simultaneously. JUNB and CEBPB partial rho computation across vascular, FAP, and MuSC compartments.
- **Findings supported:** 2, 7

### 13_integrative_meta_analysis
Cross-dataset integrative meta-analysis: Fisher-z pooling, direction voting, rank correlations, mixed-effects modeling, and cross-dataset filtering. Canonical findings table reconciliation.
- **Findings supported:** 1, 2, 3

### 14_sasp_burden_and_driver_bystander_classification
SASP burden quantification per donor, driver-bystander delta classification for 20+ TFs (mRNA rho vs AUCell rho), permutation-based null distributions, and cross-species (Tabula Muris Senis) direction-of-effect comparison.
- **Findings supported:** 1, 3

### 15_genetic_causal_support_and_eqtl
Genetic causal evidence: Open Targets colocalization grid, Mendelian randomization feasibility, OneK1K single-cell eQTL validation, GTEx NNLS deconvolution with age regression, MoTrPAC exercise plasticity analysis, hypoxia pathway contamination audit, and ssGSEA scoring.
- **Findings supported:** 6, 7, 8

### 16_signal_decomposition_and_pharmacology
Signal decomposition (age vs technology contributions), sex-stratified analysis, I-squared heterogeneity, CMap/LINCS connectivity query for therapeutic candidates, and GWAS colocalization.
- **Findings supported:** 1, 2, 8

### 17_vascular_fap_paracrine_axis
Vascular-to-FAP paracrine signaling: ligand-receptor analysis mapping age-dependent vascular ligands to FAP receptors, JUNB-dependent vs JUNB-independent axes (CXCL2/CXCR2 vs ANGPT2/Tie1/Tie2), cross-compartment SASP composition comparison.
- **Findings supported:** 4, 5

### 18_reviewer_defense_and_robustness
Pre-submission robustness hardening: DESeq2 age-effect DE with technology covariates, jackknife donor-leave-one-out stability, cell-count sensitivity, ligand-receptor multiple testing correction, four-compartment TF coupling analysis, JUNB age slopes, EGR2 polarity check, and RUNX2 FAP analysis.
- **Findings supported:** 1, 2, 4

### 19_supplementary_tables
Supplementary tables 1-5 and reviewer items (R5 partial-rho donors, R7 tone review, R8 confidence intervals).
- **Findings supported:** All (summary tables)

### 20_sex_effects_and_stratification
Sex-stratified analysis of JUNB-SASP correlations, JUNB-FAP characterization, and cross-atlas v4 validation results.
- **Findings supported:** 1, 2

### 21_fap_clustering_and_klf10_framing
FAP subtype proportion analysis, de novo clustering with snRNA-only cells (FAP, MuSC, vascular), cluster-level DE, KLF10 driver-bystander framing (JUNB vs KLF10 comparison), and KLF10 country-level sensitivity analysis.
- **Findings supported:** 2, 3

---

## Batch-to-step mapping

| Step | Source batches |
|------|---------------|
| 01 | 007 |
| 02 | 010, 022, 023, 039 |
| 03 | 011, 020, 037, 038, 043, 048 |
| 04 | 012, 019, 024, 030, 031, 049 |
| 05 | 013, 021 |
| 06 | 015, 016, 017 |
| 07 | 033, 034 |
| 08 | 052, 054, 055, 057 |
| 09 | 035, 059 |
| 10 | 036, 040, 044, 045, 046 |
| 11 | 051, 056, 058 |
| 12 | 060 |
| 13 | 050 |
| 14 | 063 |
| 15 | 064, 065 |
| 16 | 066 |
| 17 | 067, 068 |
| 18 | 070, 073 |
| 19 | 047 |
| 20 | 029 |
| 21 | 053 |
