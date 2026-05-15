# Hypotheses — Skeletal Muscle Aging

Each card is a specific, testable prediction from this project's Cycle 1 analysis. Status tracks community engagement. "Contribute" links open a pre-filled GitHub issue.

Full findings with figures and detailed evidence: [project page on the IORI website](https://iluvatarlabs.github.io/iori/junb-sasp-skeletal-muscle/).

---

### 1. Vascular endothelial cells show the strongest SASP coupling

**Prediction:** JUNB/AP-1 co-expression-module activity in vascular endothelial cells is the tightest donor-level correlate of the senescence-associated secretory phenotype (SASP) in human skeletal muscle, not fibroblasts as previously assumed.

**Evidence:** JUNB module AUCell rho=0.923 (p=3.64e-10, N=23 donors). Survived IL6+ venular exclusion, leave-one-out, senescence-marker comparison, cross-atlas replication (Nature Aging rho=0.720, p=0.008, N=12), within-country stratification (China rho=0.947, N=14).

**Confidence:** Strong

**Validation needed:** JNK perturbation in primary human vascular endothelial cells via kinase/phosphatase targeting (not siJUNB/CRISPRi). BML-260 is the lead candidate.

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)

---

### 2. Three-compartment regulatory model

**Prediction:** Vascular cells run an AP-1 axis (JUNB-dominated), MuSCs run a dual AP-1 + p21 axis, and FAPs operate through C/EBPb with negative AP-1 module-level coupling. Each compartment requires a different therapeutic strategy.

**Evidence:** Vascular JUNB partial rho=0.912, FAP CEBPB partial rho=0.888, MuSC EGR1 partial rho=0.622 (all joint-adjusted for age, sex, sequencing technology). All seven AP-1 subunits show negative AUCell coupling in FAPs. Cross-compartment module polarity may reflect co-expression structure rather than regulatory opposition (36.8% of TFs show zero target overlap between compartments).

**Confidence:** Strong (vascular + FAP), High (MuSC)

**Validation needed:** C/EBPb perturbation in primary human FAPs. Module-level confirmation is currently underpowered (AUCell rho=0.474, p=0.064 at N=16 snRNA-filtered donors).

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)

---

### 3. KLF10 is a TGF-b bystander biomarker, not a SASP driver

**Prediction:** Despite the strongest mRNA-SASP correlation across five datasets (rho=0.813, I-squared=27.2%), KLF10 does not drive SASP. Therapeutic strategies targeting KLF10 in FAPs would be misdirected.

**Evidence:** KLF10 mRNA rho=0.862 in FAPs but AUCell module rho=0.079 (delta=-0.782). KLF10 co-expression module contains zero SASP genes vs JUNB module includes CXCL2 and CCL2. Generalizes via the driver-bystander delta rule across 20 TFs.

**Confidence:** Strong (negative result)

**Validation needed:** Independent computational replication with motif-validated regulons (cisTarget was non-functional in this run).

**Status:** Awaiting independent replication

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=scientific-critique.yml)

---

### 4. Two distinct vascular-to-FAP paracrine axes

**Prediction:** Axis 1 (JNK-dependent: CXCL2/CXCR2, IL6/IL6R) is reducible by JNK inhibition. Axis 2 (JNK-independent: ANGPT2/Tie1/Tie2) persists under JNK therapy and requires separate intervention.

**Evidence:** CXCL2/CXCR2 rho_JUNB=0.924 (survives 49-pair BH-FDR). ANGPT2/TIE1 rho_age=+0.465 but ANGPT2 is not a JNK-downstream AP-1 target in this data.

**Confidence:** High

**Validation needed:** JNK inhibitor treatment in primary vascular-FAP co-culture to demonstrate Axis 2 persistence. MEDI3617 or AKB-9778 for Axis 2 independently.

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)

---

### 5. FAP growth factor compensation is inadequate

**Prediction:** FAPs upregulate FGF7 and HGF with age but the signal doesn't reach MuSCs. Exogenous growth factor supplementation should bypass this failure.

**Evidence:** FGF7 Cohen's d=+1.29 in FAPs but FGFR d=+0.13 (flat) in MuSCs. IGF2 declines d=-0.60 to -1.24 across FAP subtypes while MuSC IGF1R rises d=+0.89. FAP growth factor score does not predict MuSC activation (rho=0.189, p=0.41, N=21).

**Confidence:** Moderate

**Validation needed:** FGF7/HGF/IGF2 supplementation in MuSC activation assays. FGF7 (palifermin) is FDA-approved for oral mucositis. Note: FGF7 may signal through FGFR2 rather than FGFR1 in satellite cells.

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)

---

### 6. CDK4/6 inhibitors require satellite-cell safety evaluation

**Prediction:** CDK4/6 activity is required for MyoD-driven satellite cell re-entry. Pairing a JNK inhibitor with a CDK4/6 inhibitor in muscle-aging populations requires satellite-cell safety screens.

**Evidence:** CDKN1A/p21 is consistent with a position downstream of JUNB in the regulatory cascade (partial-correlation decomposition). CDK4/6 is required for satellite cell cycle re-entry (established myogenesis literature). CDKN1A carries a significant cis-eQTL (p=3.4e-10 in OneK1K) while AP-1 TFs do not, supporting p21 as the more genetically grounded pharmacodynamic readout.

**Confidence:** Moderate

**Validation needed:** CDK4/6 + JNK inhibitor satellite-cell safety screens in primary human cells.

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)

---

### 7. JNK/AP-1/CDKN1A/SASP regulatory hierarchy

**Prediction:** JUNB's SASP signal is largely mediated through CDKN1A. CDKN1A/p21 is a pharmacodynamic biomarker for JNK-directed trials, not a therapeutic target.

**Evidence:** Partial rho(JUNB\|SASP\|CDKN1A)=0.137 (p=0.531) vs partial rho(CDKN1A\|SASP\|JUNB)=0.513 (p=0.012). Three independent genetic-regulation layers converge on AP-1 TFs being post-translationally, not transcriptionally, regulated. Ordering cannot be established from cross-sectional data alone.

**Confidence:** Moderate

**Validation needed:** Direct JNK perturbation with CDKN1A readout in vascular cells.

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)

---

### 8. BML-260 + ANGPT2 blockade as leading combination

**Prediction:** BML-260 addresses the JNK-dependent vascular axis; ANGPT2 blockade (MEDI3617 or AKB-9778) addresses the JNK-independent axis. Neither alone covers both paracrine pathways.

**Evidence:** Inferred from the two-axis paracrine architecture (Finding 4). BML-260 targets DUSP22/JNK. First-generation systemic JNK inhibitors (tanzisertib, CC-90001) encountered translational bottlenecks. CMap/LINCS connectivity query was structurally inconclusive due to cell-line mismatch.

**Confidence:** Moderate

**Validation needed:** BML-260 + ANGPT2 blockade combination testing in aged vascular-FAP co-culture or organoid models.

**Status:** Awaiting validation

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)
