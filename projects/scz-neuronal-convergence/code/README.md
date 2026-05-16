# Schizophrenia Neuronal Convergence: Analysis Code

## Note on data file references

Scripts in this repository reference local data files (GWAS summary statistics,
PanglaoDB marker databases, gnomAD constraint tables, 1000 Genomes reference panels,
etc.) that are not included due to size or licensing. To reproduce, download the
corresponding public datasets listed in [sources.md](../sources.md) and update paths
in each script. No scripts have been modified from their original experimental versions.

---

## Analysis Steps

| Step | Directory | Description | Findings Supported |
|------|-----------|-------------|-------------------|
| 01 | `01_cell_type_enrichment` | Fisher's exact cell-type GWAS enrichment across three independent GWAS gene lists (Pardinas 2018, PGC2, PGC3) and PanglaoDB marker sets. Includes cross-dataset replication, ancestry-stratified enrichment. | 1 |
| 02 | `02_microglia_negative_result` | Microglia enrichment tested with PanglaoDB markers (k=80, OR=1.11), snRNA-seq markers (Mathys 2019), and an independent marker set. All negative. | 2 |
| 03 | `03_nfkb_circularity` | NF-kB pathway enrichment tested for circularity: the initial signal collapsed when tested against independent Pardinas 2018 GWAS genes and failed to replicate in PGC3 (RELA OR=0.79, NFKB1 OR=0.76). | 3 |
| 04 | `04_tf_regulon_enrichment` | DoRothEA TF regulon enrichment, EGR1/CTCF convergence tests, ENCODE ChIP-seq validation, motif enrichment at promoters. | 4 |
| 05 | `05_sldsc_heritability` | S-LDSC partitioned heritability with cell-type annotations, height negative control, cross-disorder S-LDSC panel (SCZ, BIP, MDD, ASD, ADHD, AD). | 1, 5 |
| 06 | `06_cross_gwas_meta_enrichment` | Cross-GWAS random-effects meta-analysis, SynGO GSEA, LDSC genetic correlation panel (17 traits), STRING PPI null model, BrainSpan developmental expression, ancestry stratification. | 1, 6 |
| 07 | `07_evolutionary_constraint` | gnomAD constraint (pLI, LOEUF) enrichment in SCZ gene sets, HAR overlap, EDT1 gene-set decomposition, SynGO/EDT1 intersection constraint, jackknife resampling, SynGO-EDT1 provenance audit, bootstrap constrained-brain test. | 5 |
| 08 | `08_cross_disorder_constraint` | Cross-disorder constraint specificity: SCHEMA-SCZ vs ASD vs DDD vs BD vs DEE. BrainSpan developmental timing, multi-axis discrimination. Cross-disorder MAGMA across 7 disorders. | 6 |
| 09 | `09_pops_gene_prioritization` | PoPS gene prioritization, feature-selection cutoff sweep, LOFGO semantic-group ablation, PoPS-pLI correlation, LD-mask sensitivity. | 5, 7 |
| 10 | `10_drug_target_overlap` | Drug-target enrichment under 4 background definitions. Demonstrates definition-dependency of drug-target overlap claims. | 7 |
| 11 | `11_sensitivity_analyses` | Permutation verification, gene-length permutation controls, brain-expressed background sensitivity, FDR correction audit, specification-curve analysis across 5 key findings. | 1, 3, 5 |
| 12 | `12_peer_review_hardening` | 8-disorder MAGMA competitive battery, expression ablation, Monte Carlo tail analysis, environmental-axis gene-set batteries, H-MAGMA, expanded IEG battery, complement/myelin sensitivity, SynGO constraint specificity, multiple-testing accounting. | All |

---

## Notes

All 7 findings are supported by computational analysis steps above. Finding 3
(NF-kB circularity) is primarily evidenced by the absence of replication rather
than a dedicated circularity-detection experiment; the key evidence comes from
the circularity test in step 03 and the PGC3 non-replication results.
