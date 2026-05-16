# Hypotheses — Mapping Schizophrenia Risk

Each card is a specific, testable prediction from this project's Cycle 1 analysis. Status tracks community engagement. "Contribute" links open a pre-filled GitHub issue.

Full findings with figures and detailed evidence: [project page on the IORI website](https://iluvatarlabs.github.io/iori/scz-neuronal-convergence/).

---

### 1. Neurons are the primary enriched cell type

**Prediction:** Schizophrenia common-variant risk enriches in neuronal synaptic programs, not other brain cell types.

**Evidence:** Six orthogonal lines: Fisher overlap OR=9.76 (FDR=1.79e-10), cross-dataset replication OR=29.88, brain-expressed background OR=7.87, gene-length-matched permutation adjusted OR=6.94, S-LDSC neuronal 1.83-fold enrichment (p=0.009), sex-stratified S-LDSC directionally concordant (Cohen's h=0.006).

**Confidence:** Strong

**Validation needed:** Cell-type-resolved cortical eQTL to test whether expression effects localize to neurons.

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)

---

### 2. Microglia are NOT enriched

**Prediction:** Microglia do not carry a cell-autonomous SCZ genetic signal. Immune-associated findings (complement, TLR) persist within neuronal risk sets.

**Evidence:** OR=1.11 (p=0.53). Zero overlap between PGC3 prioritized genes and PanglaoDB microglia markers. TLR pathway signal (OR=5.91, raw p=0.016) does not survive FDR correction (0.147).

**Confidence:** Strong (negative result)

**Validation needed:** Independent computational replication with updated cell-type markers.

**Status:** Awaiting independent replication

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=scientific-critique.yml)

---

### 3. NF-kB pathway enrichment is a circular false positive

**Prediction:** Prior NF-kB enrichment findings were circular: the input gene list encoded NF-kB biology before the enrichment test was run.

**Evidence:** 14 of 88 genes in the working set were pre-selected from immune-pathway analyses. 86% of RELA regulon overlaps attributable to those 14 genes. RELA against PGC3 proper: OR=0.79 (p=0.72).

**Confidence:** Strong (methodological contribution)

**Validation needed:** Independent methodological review. This is a computational audit with implications for any project using regulon-based enrichment on curated gene sets.

**Status:** Awaiting independent replication

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=scientific-critique.yml)

---

### 4. EGR1, MEF2C, and CTCF are enriched at SCZ gene promoters, with cross-disorder context

**Prediction:** EGR1 and MEF2C are pan-neurodevelopmental regulators (shared across SCZ, ASD, DDD). CTCF is the only tested factor showing possibly SCZ-preferential motif enrichment.

**Evidence:** All three recur across DoRothEA regulon AND JASPAR PWM motif enrichment (EGR1 OR=4.98, corrected p=4.5e-6; CTCF OR=3.15, corrected p=6.9e-5; MEF2C OR=3.84). Cross-disorder: EGR1 DDD OR=13.89, ASD OR=11.90 (both exceed SCZ). CTCF comparators all non-significant (bootstrap-confirmed, 100 draws).

**Confidence:** Strong (enrichment), Moderate (SCZ-specificity)

**Validation needed:** Single-cell perturbation of EGR1 and CTCF in human iPSC-derived cortical neurons. Activity-dependent EGR1 binding maps in live cortical neurons (stimulus-evoked ChIP-seq or CUT&Tag).

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)

---

### 5. Evolutionary constraint concentrates at synaptic loci via two-layer architecture

**Prediction:** SCZ risk does not uniformly constrain the risk gene set. Constraint concentrates specifically at synaptic genes through two stacking effects: synaptic genes as a class are broadly constrained (pLI OR=4.45 vs genome), and within that class, SCZ-associated synaptic genes show additional concentration (within-class OR=6.94, p=0.004).

**Evidence:** Full EDT1 set pLI OR=1.14 (p=0.41, not significant). B3 synaptic intersection pLI OR=26.44 (p=2.22e-7). SCHEMA rare-variant genes all pLI>0.9. Functional decomposition: synaptic scaffold OR=20.91, glutamate receptor Haldane-corrected OR=32.5, ion channel OR=0.93 (no constraint).

**Confidence:** Strong

**Validation needed:** PSD proteomics from SCZ patient-derived iPSC neurons to test whether constraint translates to protein-level changes.

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)

---

### 6. Cross-disorder constraint is shared, not SCZ-exclusive

**Prediction:** The constraint architecture is shared with autism and developmental disorders at comparable magnitude. It is a neurodevelopmental substrate, not a schizophrenia-specific feature.

**Evidence:** ASD OR=25.4 (BH q=6e-49), developmental disorders OR=31.3 (BH q=2e-125), no Holm-significant pairwise differences with SCZ. SCZ-distinctive features reside at within-class concentration (OR=6.94), possibly CTCF motif enrichment, and cortical cell-type S-LDSC where only SCZ reaches significance.

**Confidence:** Moderate

**Validation needed:** A tested-but-not-significant comparator gene set to strengthen the specificity claim. Is CTCF SCZ-specificity real or will it dissolve with larger ASD gene sets?

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)

---

### 7. Drug-target overlap is definition-dependent; regulatory targeting may be more tractable

**Prediction:** SCZ risk genes are enriched for drug targets relative to the genome (DGIdb OR=2.33) but show no enrichment relative to the druggable universe (OR=1.03). The regulatory layer (EGR1, CTCF) may be the more tractable therapeutic axis.

**Evidence:** Four-background DGIdb analysis: genome OR=2.33 (p=3e-9), brain-expressed OR=1.63 (p=0.003), druggable universe OR=1.03 (p=0.93), expression+length matched OR=1.22 (p=0.29). Constrained synaptic proteins are by definition intolerant to perturbation.

**Confidence:** Moderate

**Validation needed:** EGR1 and CTCF perturbation studies in disease-relevant neuronal models to test whether modulating regulators is a viable therapeutic axis.

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)
