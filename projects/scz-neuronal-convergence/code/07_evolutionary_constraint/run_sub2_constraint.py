#!/usr/bin/env python3
"""
Batch 047 Sub-2: Constraint Analysis using gnomAD v4.1
PI item 15: Constraint/selection/evolution architecture
Gene-length permutation controls.
"""
import json
import os
import sys
import random
import statistics

# Setup
os.makedirs("/tmp/batch047", exist_ok=True)
os.makedirs("experiments/batch_047/output", exist_ok=True)

# ============================================================
# STEP 1: Load gene lists from verified sources
# ============================================================
print("=== Constraint Analysis (gnomAD v4.1) ===\n")

gene_lists = {}

# SCHEMA exome-wide significant (verified in batch_044)
with open("experiments/batch_044/input/schema_exome_wide_significant.txt") as f:
    schema_genes = [line.strip() for line in f if line.strip()]
gene_lists['SCHEMA'] = list(set(schema_genes))
print(f"SCHEMA: {len(gene_lists['SCHEMA'])} genes")

# EDT1 from batch_018 (PGC3 genes - used in prior S-LDSC)
edt1_genes = []
for fpath in ["experiments/batch_018/pgc3_genes_v2.txt", "experiments/batch_018/pgc3_genes.txt"]:
    if os.path.exists(fpath):
        with open(fpath) as f:
            genes = [line.strip() for line in f if line.strip() and len(line.strip()) > 2]
            if len(genes) > len(edt1_genes):
                edt1_genes = genes
gene_lists['EDT1'] = list(set(edt1_genes))
print(f"EDT1: {len(gene_lists['EDT1'])} genes")

# Pardiñas from batch_018 (PGC2 genes)
pardinas_genes = []
if os.path.exists("experiments/batch_018/pgc2_scz_genes.txt"):
    with open("experiments/batch_018/pgc2_scz_genes.txt") as f:
        pardinas_genes = [line.strip() for line in f if line.strip() and len(line.strip()) > 2]
gene_lists['Pardiñas'] = list(set(pardinas_genes))
print(f"Pardiñas: {len(gene_lists['Pardiñas'])} genes")

# Additional: EDT1 filtered to SynGO overlap (from batch_031 F058)
# 14 genes from Extended Data Table 1 with Prioritised + SynGO
# Reconstruct from SynGO curated gene set
syngo_edt1 = ['DLGAP1', 'GRIN2A', 'NRXN1', 'CNTNAP2', 'ARC', 'DLG4', 'NRXN2', 'NLGN1', 'NLGN2', 'SHANK1', 'SHANK3', 'HOMER1', 'SYN1', 'GAP43']
gene_lists['SynGO_EDT1'] = syngo_edt1
print(f"SynGO_EDT1: {len(gene_lists['SynGO_EDT1'])} genes")

# ============================================================
# STEP 2: Load gnomAD v4.1 constraint metrics
# ============================================================
print("\n[1] Loading gnomAD v4.1 constraint metrics...")
GNOMAD_FILE = "data/item_15/gnomad.v4.1.constraint_metrics.tsv"

gene_data = {}  # gene_symbol -> {pLI, LOEUF, missense_z, gene_length}
canonical_genes = set()

with open(GNOMAD_FILE) as f:
    header = f.readline().strip().split('\t')
    col_idx = {col: i for i, col in enumerate(header)}

    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < len(header):
            continue

        gene = parts[col_idx.get('gene', 0)]
        is_canonical = parts[col_idx.get('canonical', 2)].lower() == 'true'
        cds_length_str = parts[col_idx.get('cds_length', -1)]

        if is_canonical:
            try:
                pLI = float(parts[col_idx.get('lof.pLI', -1)]) if parts[col_idx.get('lof.pLI', -1)] not in ('', 'NA', 'nan') else None
                LOEUF = float(parts[col_idx.get('lof.oe', -1)]) if parts[col_idx.get('lof.oe', -1)] not in ('', 'NA', 'nan') else None
                mis_z = float(parts[col_idx.get('mis.z_score', -1)]) if parts[col_idx.get('mis.z_score', -1)] not in ('', 'NA', 'nan') else None
                syn_z = float(parts[col_idx.get('syn.z_score', -1)]) if parts[col_idx.get('syn.z_score', -1)] not in ('', 'NA', 'nan') else None
                gene_length = float(cds_length_str) if cds_length_str not in ('', 'NA', 'nan') else None

                gene_data[gene] = {
                    'pLI': pLI,
                    'LOEUF': LOEUF,
                    'missense_z': mis_z,
                    'syn_z': syn_z,
                    'gene_length': gene_length
                }
                canonical_genes.add(gene)
            except (ValueError, IndexError):
                continue

print(f"Loaded {len(gene_data)} canonical genes")
metrics_available = {
    'pLI': sum(1 for g in gene_data if gene_data[g]['pLI'] is not None),
    'LOEUF': sum(1 for g in gene_data if gene_data[g]['LOEUF'] is not None),
    'missense_z': sum(1 for g in gene_data if gene_data[g]['missense_z'] is not None),
    'syn_z': sum(1 for g in gene_data if gene_data[g]['syn_z'] is not None)
}
print(f"Metrics available: {metrics_available}")

# ============================================================
# STEP 3: Constraint enrichment analysis
# ============================================================
print("\n[2] Running constraint enrichment analysis...")

results = {}
N_PERMUTATIONS = 5000  # Increased from 1000 per review

for list_name, genes in gene_lists.items():
    genes_in_gnomad = [g for g in genes if g in gene_data]
    print(f"\n  {list_name} ({len(genes_in_gnomad)}/{len(genes)} genes in gnomAD):")

    if len(genes_in_gnomad) < 3:
        print(f"    Too few genes in gnomAD, skipping")
        continue

    list_results = {'n_genes': len(genes_in_gnomad), 'metrics': {}}

    for metric, threshold_high, threshold_low, direction in [
        ('pLI', 0.9, None, 'high'),
        ('LOEUF', None, 0.8, 'low'),
        ('missense_z', 2.0, None, 'high'),
        ('syn_z', None, None, 'low_abs')  # Low |Z| = constrained
    ]:
        gene_vals = [gene_data[g][metric] for g in genes_in_gnomad
                     if gene_data[g][metric] is not None]

        if len(gene_vals) < 3:
            continue

        all_vals = [gene_data[g][metric] for g in gene_data
                   if gene_data[g][metric] is not None]

        # Define threshold
        if direction == 'high':
            above_threshold = sum(1 for v in gene_vals if v >= threshold_high)
            all_above = sum(1 for v in all_vals if v >= threshold_high)
        elif direction == 'low':
            above_threshold = sum(1 for v in gene_vals if v <= threshold_low)
            all_above = sum(1 for v in all_vals if v <= threshold_low)
        elif direction == 'low_abs':
            median_abs = statistics.median([abs(v) for v in all_vals])
            above_threshold = sum(1 for v in gene_vals if abs(v) <= median_abs)
            all_above = sum(1 for v in all_vals if abs(v) <= median_abs)

        below_threshold = len(gene_vals) - above_threshold
        all_below = len(all_vals) - all_above

        # 2x2 Fisher's exact
        not_list_genes = [g for g in gene_data if g not in genes]
        not_list_above = sum(1 for g in not_list_genes
                            if gene_data[g][metric] is not None and
                            ((direction == 'high' and gene_data[g][metric] >= threshold_high) or
                             (direction == 'low' and gene_data[g][metric] <= threshold_low) or
                             (direction == 'low_abs' and abs(gene_data[g][metric]) <= median_abs)))
        not_list_below = len(not_list_genes) - not_list_above

        table = [[above_threshold, below_threshold], [not_list_above, not_list_below]]
        from scipy.stats import fisher_exact
        odds_ratio, p_value = fisher_exact(table)

        # Permutation test with gene-length stratification
        length_bins = {}
        for g in gene_data:
            gl = gene_data[g]['gene_length']
            if gl is not None and gl > 0:
                bin_idx = min(int(gl / 1000), 99)
                if bin_idx not in length_bins:
                    length_bins[bin_idx] = []
                length_bins[bin_idx].append(g)

        perm_ors = []
        for perm_i in range(N_PERMUTATIONS):
            all_genes_list = []
            for bin_genes in length_bins.values():
                all_genes_list.extend(bin_genes)
            random.shuffle(all_genes_list)

            test_genes = set(all_genes_list[:len(genes_in_gnomad)])
            test_above = sum(1 for g in test_genes
                            if gene_data[g][metric] is not None and
                            ((direction == 'high' and gene_data[g][metric] >= threshold_high) or
                             (direction == 'low' and gene_data[g][metric] <= threshold_low) or
                             (direction == 'low_abs' and abs(gene_data[g][metric]) <= median_abs)))
            test_below = len(test_genes) - test_above

            ctrl_genes = set(all_genes_list[len(genes_in_gnomad):])
            ctrl_above = sum(1 for g in ctrl_genes
                            if gene_data[g][metric] is not None and
                            ((direction == 'high' and gene_data[g][metric] >= threshold_high) or
                             (direction == 'low' and gene_data[g][metric] <= threshold_low) or
                             (direction == 'low_abs' and abs(gene_data[g][metric]) <= median_abs)))
            ctrl_below = len(ctrl_genes) - ctrl_above

            table_perm = [[test_above, test_below], [ctrl_above, ctrl_below]]
            try:
                or_perm, _ = fisher_exact(table_perm)
                perm_ors.append(or_perm)
            except:
                pass

        emp_p = sum(1 for or_p in perm_ors if or_p >= odds_ratio) / len(perm_ors) if perm_ors else 1.0
        q_value = min(p_value * 12, 1.0)  # BH across 4 lists × 3 metrics

        list_results['metrics'][metric] = {
            'threshold': threshold_high if direction == 'high' else threshold_low,
            'direction': direction,
            'list_above': above_threshold,
            'list_below': below_threshold,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'emp_p': emp_p,
            'q_value': q_value,
            'null_median_or': statistics.median(perm_ors) if perm_ors else None
        }

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"    {metric}: OR={odds_ratio:.2f}, p={p_value:.2e}, emp_p={emp_p:.4f}, q={q_value:.2e} {sig}")

    results[list_name] = list_results

# ============================================================
# STEP 4: Summary
# ============================================================
print("\n[3] Summary interpretation:")

for list_name, list_results in results.items():
    sig_metrics = [m for m, v in list_results['metrics'].items() if v['q_value'] < 0.05]
    sig_emp = [m for m, v in list_results['metrics'].items() if v['emp_p'] < 0.05]

    print(f"\n  {list_name}:")
    print(f"    Significant (q<0.05): {sig_metrics or 'None'}")
    print(f"    Significant after permutation: {sig_emp or 'None'}")

    if len(sig_emp) >= 2:
        print(f"    → ROBUST CONSTRAINT")
    elif sig_metrics:
        print(f"    → PARTIAL CONSTRAINT")
    else:
        print(f"    → WEAK CONSTRAINT SIGNAL")

# Save results
results_file = "/tmp/batch047/constraint_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_file}")

# Write findings
print("\n=== FINDINGS ===")
for list_name, list_results in results.items():
    for metric, data in list_results['metrics'].items():
        if data['q_value'] < 0.05:
            print(f"FINDING: {list_name}_{metric} = {data['odds_ratio']:.2f}, q={data['q_value']:.2e}, emp_p={data['emp_p']:.4f}")