# Manifest — Skeletal Muscle Aging

Per-cycle record of what went into each release: community contributions incorporated, datasets added, hypotheses published or revised, and links to the relevant PRs.

---

## Cycle 1

**Date:** 2026-05-15
**Status:** Initial publication

### Community contributions incorporated

None (initial cycle).

### Datasets used

- HLMA atlas (CNGBdb OMIX004308, Lai et al. 2024): 387,000+ cells, 23 donors
- Nature Aging 2024 atlas (Kedlian et al.): 90,902 nuclei, 12 donors
- GTEx v8 bulk muscle (N=803)
- OneK1K single-cell eQTL (N=982)
- Open Targets / Genetics Portal

Full provenance in [sources.md](sources.md).

### Hypotheses published

8 hypotheses (3 Strong, 2 High, 3 Moderate):

1. Vascular JUNB-SASP dominance — Strong
2. Three-compartment regulatory model — Strong / High
3. KLF10 TGF-b bystander — Strong (negative result)
4. Two vascular-to-FAP paracrine axes — High
5. FAP growth factor compensation inadequate — Moderate
6. CDK4/6 satellite-cell safety — Moderate
7. JNK/AP-1/CDKN1A hierarchy — Moderate
8. BML-260 + ANGPT2 blockade combination — Moderate

Full cards in [hypotheses.md](hypotheses.md).

### Key limitations acknowledged

- All findings are correlational. Causal validation requires perturbation in primary human cells.
- pySCENIC cisTarget motif pruning was non-functional. Module-level claims use co-expression modules only and await motif validation.
- Country of origin (China vs Spain) is the dominant confound in HLMA, not age. Within-country correlations hold.
- Cross-atlas replication (Nature Aging) is directionally concordant but formally underpowered for vascular JUNB-SASP.
- Cross-species AP-1 polarity was not conserved in Tabula Muris Senis (protocol differences preclude definitive comparison).

### Open questions for the community

6 specific validation experiments listed on the [project page](https://iluvatarlabs.github.io/iori/junb-sasp-skeletal-muscle/#open-questions-for-the-community), ranging from JNK perturbation in primary vascular cells to iPSC-derived vascular organoid models.
