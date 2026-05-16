# Data Sources — Mapping Schizophrenia Risk

All analyses use publicly available GWAS summary statistics, annotation databases, and curated gene sets. No original experimental data were generated.

## Cycle 1

### Primary datasets

| Dataset | Source | Accession / URL | Samples | Used for |
|---------|--------|----------------|---------|----------|
| PGC3 EUR SCZ GWAS | Trubetskoy et al. 2022 | figshare DOI 10.6084/m9.figshare.19426775 | 76,755 cases, 243,649 controls | S-LDSC heritability partitioning, MAGMA gene-level analysis |
| PGC2 SCZ GWAS | Schizophrenia Working Group 2014 | pgc.unc.edu | 36,989 cases, 113,075 controls | Meta-analysis sensitivity testing |
| Pardinas et al. 2018 | Nature Genetics 50:381-389 | — | Common-variant prioritized gene set | Fisher enrichment, constraint analysis |
| PanglaoDB | panglaodb.se | panglaodb.se | Cell-type marker sets | Cell-type enrichment framework |
| DoRothEA / OmniPath | omnipathdb.org | — | TF regulon target sets | Regulon-based enrichment |
| JASPAR 2022 | jaspar.genereg.net | — | Position-weight matrices | Motif enrichment (independent of DoRothEA) |
| gnomAD v4.1 | gnomad.broadinstitute.org | — | pLI, LOEUF, missense Z, synonymous Z | Evolutionary constraint analysis. Cross-version concordance with v2.1.1: pLI rho=0.833, LOEUF rho=0.843 |
| SCHEMA | schema.broadinstitute.org | — | Rare coding variant gene-level statistics | Rare-variant constraint validation |
| SynGO | syngoportal.org | — | Curated synaptic gene annotations (Koopmans et al. 2019) | Synaptic subset definition for constraint analysis |
| BrainSpan | brainspan.org | — | Allen Brain Atlas developmental expression | Developmental-stage profiling |
| S-LDSC annotations | Zenodo record 7768714 | — | Baseline-LD v2.2 | Conditional heritability partitioning |
| ENCODE | encodeproject.org | — | ChIP-seq peak data | EGR1/CTCF binding at SCZ gene promoters |
| DGIdb | dgidb.org | — | Drug-gene interaction database (strict: DrugBank, GtoPdb, ChEMBL, TTD) | Drug-target overlap analysis across four backgrounds |

### Cross-disorder comparators

| Dataset | Used for |
|---------|----------|
| ASD (Satterstrom et al. 2020) | Cross-disorder constraint and motif enrichment comparison |
| Dominant developmental disorders (Kaplanis et al. 2020) | Cross-disorder constraint comparison |
| BIP, MDD, ADHD, AD, IBD, height GWAS | Disease-specificity controls for S-LDSC and genetic correlation |

### Software

- S-LDSC (Python)
- MAGMA (gene-level analysis)
- PoPS (ridge regression gene prioritization)
- MiXeR (Gaussian causal mixture modeling)
- Fisher's exact test / permutation testing (Python/SciPy)

### Key literature

- Trubetskoy et al. 2022. PGC3 SCZ GWAS. *Nature* 604:502-508.
- Pardinas et al. 2018. Common SCZ alleles enriched in mutation-intolerant genes. *Nature Genetics* 50:381-389.
- Singh et al. 2022 (SCHEMA). Rare coding variants in 10 genes. *Nature* 604:509-516.
- Skene et al. 2018. Genetic identification of brain cell types underlying SCZ. *Nature Genetics* 50:825-833.
- Koopmans et al. 2019. SynGO: an evidence-based synapse knowledge base. *Neuron* 103:217-234.
- Weeks et al. 2023. PoPS: polygenic enrichment of gene features. *Nature Genetics* 55:1267-1276.
- Sekar et al. 2016. SCZ risk from complex variation of complement component 4. *Nature* 530:177-183.
- Satterstrom et al. 2020. Large-scale exome sequencing study in autism. *Cell* 180:568-584.
- Kaplanis et al. 2020. Evidence for 28 genetic disorders. *Nature* 586:757-762.
- Simonsohn et al. 2020. Specification curve analysis. *Nature Human Behaviour* 4:1208-1214.

Full corpus in the [project repository](https://github.com/IluvatarLabs/iluvatar-open-research/tree/main/projects/scz-neuronal-convergence) and on the [project page](https://iluvatarlabs.github.io/iori/scz-neuronal-convergence/).
