"""
Supplementary Analysis: Extended Crosstalk Including Growth Factors
===================================================================
This extends MUST DO 2 to also check growth factor receptors (FGF7, HGF, PDGFA)
which could promote MuSC function - addressing whether aged FAPs have mixed
secretome (suppressive + supportive).
"""

import json
import numpy as np

# Load the results from main analysis
with open("/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_019/must_do_2_crosstalk_results.json") as f:
    results = json.load(f)

print("=" * 80)
print("EXTENDED CROSSTALK ANALYSIS: Checking Growth Factor Pathways")
print("=" * 80)

# Check receptor expression for growth factors
growth_factor_ligands = ["FGF7", "HGF", "PDGFA"]
growth_factor_receptors = {
    "FGF7": ["FGFR1", "FGFR2", "FGFR3", "FGFR4"],
    "HGF": ["MET"],
    "PDGFA": ["PDGFRA", "PDGFRB"]
}

print("\nGrowth Factor Receptor Expression in MuSCs:")
print("-" * 60)

for ligand, receptors in growth_factor_receptors.items():
    print(f"\n{ligand} receptors:")
    for receptor in receptors:
        for state, expr_dict in results['musc_receptor_expression'].items():
            if receptor in expr_dict:
                expr = expr_dict[receptor]
                status = "EXPRESSED" if expr > 0.1 else "low/absent"
                print(f"  {receptor} in {state}: {expr:.3f} ({status})")

# Check if growth factors are elevated in JUNB+ FAPs
print("\n" + "=" * 80)
print("Growth Factor Expression in JUNB+ vs JUNB- FAPs")
print("=" * 80)

for ligand in growth_factor_ligands:
    for entry in results['ligand_enrichment']:
        if entry['gene'] == ligand:
            fc = entry.get('log2FC', 'N/A')
            if fc == fc:  # Check not NaN
                direction = "UP in JUNB+" if fc > 0 else "DOWN in JUNB+"
                print(f"{ligand}: log2FC={fc:.3f} ({direction}), p_adj={entry['p_value_corrected']:.2e}")
            else:
                # Estimate direction from means
                if entry['mean_junb_pos'] > entry['mean_junb_neg']:
                    print(f"{ligand}: elevated in JUNB+ (mean_neg={entry['mean_junb_neg']:.3f}, mean_pos={entry['mean_junb_pos']:.3f})")
                else:
                    print(f"{ligand}: reduced in JUNB+ (mean_neg={entry['mean_junb_neg']:.3f}, mean_pos={entry['mean_junb_pos']:.3f})")

# Final interpretation
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

print("""
Key finding from MUST DO 2:
- JUNB+ aged FAPs have elevated TNF (log2FC=0.74, p_adj=0.007)
- TNF signals via TNFRSF1A, which is expressed in all MuSC states
- TNF is a known MuSC activation suppressor

Extended finding:
- Growth factors (FGF7, HGF, PDGFA) are NOT elevated in JUNB+ FAPs
- Some actually decrease (PDGFA: log2FC=-0.65)
- FGFRs are expressed in MuSCs, but the ligands aren't elevated

Biological implication:
JUNB+ aged FAPs have a SUPPRESSIVE secretome:
- Elevated: TNF (suppresses activation)
- Not elevated: FGF7, HGF, PDGFA (would promote activation)

This suggests aged FAPs with high JUNB shift from supportive to suppressive,
impairing MuSC-mediated regeneration.
""")

print("=" * 80)
