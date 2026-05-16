# Manifest — Mapping Schizophrenia Risk

Per-cycle record of what went into each release: community contributions incorporated, datasets added, hypotheses published or revised, and links to the relevant PRs.

---

## Cycle 1

**Date:** 2026-05-15
**Status:** Initial publication

### Community contributions incorporated

None (initial cycle).

### Datasets used

13 datasets. See [sources.md](sources.md) for full provenance.

### Hypotheses published

7 hypotheses (5 Strong, 1 Strong/Moderate, 2 Moderate):

1. Neurons are the primary enriched cell type — Strong
2. Microglia are NOT enriched — Strong (negative result)
3. NF-kB pathway enrichment is a circular false positive — Strong (methodological)
4. EGR1, MEF2C, CTCF enriched at SCZ promoters — Strong (enrichment), Moderate (specificity)
5. Two-layer constraint architecture at synaptic loci — Strong
6. Cross-disorder constraint is shared — Moderate
7. Drug-target overlap is definition-dependent — Moderate

Full cards in [hypotheses.md](hypotheses.md).

### Key limitations acknowledged

- All analyses are purely computational, based on summary statistics and annotation databases. No original experimental data were generated.
- EGR1 ChIP-seq null in post-mortem tissue is interpretable as an activity-dependent false negative, not a true negative for EGR1 binding at SCZ loci.
- Cross-disorder constraint equivalence rests on absence of significant pairwise differences (absence of evidence, not evidence of absence).
- CTCF SCZ-specificity claim is underpowered for ASD comparison (k=3 overlap genes).
- East Asian S-LDSC neuronal coefficient is near zero — cross-ancestry robustness of neuronal enrichment is uncertain.
- pySCENIC cisTarget motif pruning not used (DoRothEA regulons + JASPAR PWM used instead; independent frameworks but both annotation-dependent).

### Open questions for the community

7 specific validation experiments listed on the [project page](https://iluvatarlabs.github.io/iori/scz-neuronal-convergence/#open-questions-for-the-community), ranging from EGR1/CTCF perturbation in iPSC neurons to East Asian ancestry S-LDSC replication.
