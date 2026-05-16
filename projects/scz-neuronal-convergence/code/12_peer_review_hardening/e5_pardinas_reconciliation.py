#!/usr/bin/env python3
"""
E5 (R5): Pardinas Reconciliation Paragraph

Write a reconciliation paragraph explaining why:
- EDT1 pLI null (OR=1.14, from batch_047) is NOT contradictory with
- Pardinas 2018 headline: common SCZ alleles enriched in mutation-intolerant genes

WHY: A reviewer familiar with Pardinas 2018 (PMID: 29483656) would flag our null
pLI result as contradicting an established finding. The reconciliation is that the
two results measure different things (SNP-level heritability partitioning vs
gene-set overlap of genome-wide significant loci).

Output: experiments/batch_069/output/e5_pardinas_reconciliation.txt
"""

from __future__ import annotations
import pathlib

PROJECT_ROOT = pathlib.Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "batch_069" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAGRAPH = """Reconciliation: EDT1 pLI Null vs. Pardinas 2018 Constraint Enrichment

Our finding that EDT1 genes are not enriched for mutation-intolerant genes by
pLI (OR = 1.14, Fisher exact p > 0.05; batch_047) does not contradict the
Pardinas et al. (2018) result that common SCZ risk alleles are enriched in
genes intolerant to loss-of-function mutation. The apparent discrepancy reflects
a fundamental difference in statistical test and object of analysis. Pardinas
employed stratified LD score regression (S-LDSC), which partitions SNP-level
heritability across the entire polygenic architecture, including the vast
majority of subthreshold signal that does not reach genome-wide significance.
In contrast, our analysis tests whether the subset of genes prioritized at
genome-wide significant loci (EDT1, defined by PGC3 Extended Data Table 1) is
enriched for high-pLI genes using Fisher's exact test against all gnomAD genes.
These are different tests on different objects: S-LDSC integrates continuous
genetic signal across all common variants, while Fisher overlap tests discrete
membership in a genome-wide significant gene set against a discrete pLI
threshold. The polygenic signal CAN be enriched for constrained genes even when
the genome-wide significant subset alone is not, precisely because most
heritability resides in the subthreshold tail (Trubetskoy et al. 2022 report
that PGC3 GWAS significant loci explain only a fraction of total h2_SNP).
Furthermore, EDT1 gene prioritization criteria (positional mapping, eQTL, Hi-C,
SMR, fine-mapping) are not filtered by constraint, so the EDT1 set includes many
unconstrained genes that happen to lie near GWAS loci. Our SynGO-intersected
subset (n = 11, OR = 20.9, batch_048) does show extreme constraint enrichment,
consistent with the Pardinas finding that the constraint signal concentrates in
specific biological programs. In summary: Pardinas detected that polygenic SCZ
risk broadly favors constrained genes via genome-wide heritability partitioning;
our null result shows that the specific gene set mapped from genome-wide
significant loci is not enriched as a whole, though its synaptic core is. These
findings are complementary, not contradictory.

Key references:
- Pardinas et al. (2018) Nat Genet 50:381-389. PMID: 29483656.
- Trubetskoy et al. (2022) Nature 604:502-508 (PGC3).
- Karczewski et al. (2020) Nature 581:434-443 (gnomAD pLI/LOEUF).

Relevant project findings:
- F122 (batch_047): EDT1 pLI OR = 1.14, not significant.
- F147 (batch_048): SynGO_EDT1 pLI OR = 20.9, p = 6.7e-6. Constraint is PSD-scaffold-concentrated.
- F059_03 (batch_059): EDT1-ex-B3 signal is length-stable (survives length-matched null).
"""

def main():
    out_path = OUTPUT_DIR / "e5_pardinas_reconciliation.txt"
    with open(out_path, "w") as f:
        f.write(PARAGRAPH.strip() + "\n")
    print(f"Wrote: {out_path}")
    print(PARAGRAPH)


if __name__ == "__main__":
    main()
