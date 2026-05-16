#!/usr/bin/env python3
"""
batch_035: Drug Target Enrichment + STRING PPI + Sex-Stratified S-LDSC

Three analyses addressing field-standard gaps:
- G8: Drug target enrichment via OpenTargets Platform API
- G10: STRING PPI enrichment via STRING API
- D18: Sex-stratified cell-type enrichment via S-LDSC partitioned heritability

Author: Marvin (autonomous research agent)
Date: 2026-04-15
"""

import json
import time
import gzip
import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact
import requests
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
DATA_DIR = PROJECT_ROOT / "data"
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_035"
OUTPUT_DIR = BATCH_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Gene lists
PARDINAS_GENES_PATH = PROJECT_ROOT / "experiments" / "batch_008" / "data" / "gwas_genes.parquet"
PGC3_TABLE_PATH = DATA_DIR / "19426775" / "scz2022-Extended-Data-Table1.xlsx"

# Sex-stratified sumstats
EUR_FEMALE_PATH = DATA_DIR / "19426775" / "daner_PGC_SCZ_w3_75_0618a_eur_female.gz"
EUR_MALE_PATH = DATA_DIR / "19426775" / "daner_PGC_SCZ_w3_75_0618a_eur_male.gz"

# Background gene universe
BACKGROUND_N = 20197  # Entrez protein-coding genes

# OpenTargets API
OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"

# STRING API
STRING_API = "https://string-db.org/api"

# ============================================================================
# S-LDSC Configuration
# ============================================================================
LDSC = "/home/yuanz/torchml/bin/ldsc.py"
MUNGE = "/home/yuanz/torchml/bin/munge_sumstats.py"

# LD scores from batch_034 (celltype annotations + ld scores already computed)
CELLTYPE_LDSCORES = str(PROJECT_ROOT / "experiments" / "batch_034" / "output" / "ld_scores_final" / "celltype.")
BASELINE_LD = str(PROJECT_ROOT / "data" / "ldsc" / "baselineLD" / "baselineLD.")
WEIGHTS = str(PROJECT_ROOT / "data" / "ldsc" / "weights" / "1000G_Phase3_weights_hm3_no_MHC" / "weights.hm3_noMHC.")
HM3_SNPS = str(PROJECT_ROOT / "data" / "ldsc" / "weights" / "1000G_Phase3_weights_hm3_no_MHC" / "hm3_snps.txt")
# Batch_034 created a custom baselineLD_hm3_snps.txt that includes MHC SNPs
BASELINELD_HM3_SNPS = str(PROJECT_ROOT / "experiments" / "batch_034" / "output" / "baselineLD_hm3_snps.txt")
FRQ_REF = str(PROJECT_ROOT / "data" / "ldsc" / "1000G_Phase3_frq" / "1000G.EUR.QC")


# ============================================================================
# Utility functions
# ============================================================================
def load_gene_lists():
    """Load Pardinas 444 and PGC3 106 gene lists."""
    # Pardinas 444
    pardinas_df = pd.read_parquet(PARDINAS_GENES_PATH)
    pardinas_genes = set(pardinas_df['hgnc_symbol'].str.upper().tolist())
    print(f"Pardinas genes loaded: {len(pardinas_genes)}")

    # PGC3 106 (from Extended Data Table 1, protein-coding only)
    pgc3_xlsx = pd.ExcelFile(PGC3_TABLE_PATH)
    sheet1 = pd.read_excel(pgc3_xlsx, sheet_name='Extended.Data.Table.1')
    # Column is 'Symbol.ID', biotype column is 'gene_biotype'
    if 'gene_biotype' in sheet1.columns:
        protein_coding = sheet1[sheet1['gene_biotype'] == 'protein_coding']
        pgc3_genes = set(protein_coding['Symbol.ID'].str.upper().tolist())
    elif 'Symbol.ID' in sheet1.columns:
        pgc3_genes = set(sheet1['Symbol.ID'].str.upper().tolist())
    else:
        pgc3_genes = set()
    print(f"PGC3 genes loaded: {len(pgc3_genes)}")

    # Combined
    combined = pardinas_genes | pgc3_genes
    print(f"Combined gene list: {len(combined)}")
    print(f"Overlap: {len(pardinas_genes & pgc3_genes)}")

    return pardinas_genes, pgc3_genes, combined


def fisher_exact_test(k, n, K, N):
    """
    Fisher's exact test for enrichment or depletion.
    k = overlap (SCZ genes that are drug targets)
    n = total SCZ genes
    K = total drug targets in background
    N = background gene universe size

    Uses two-sided test to detect both enrichment (OR > 1) and depletion (OR < 1).
    Returns OR, p-value, 95% CI
    """
    # Contingency table:
    # [[k, K-k], [n-k, N-n-K+k]]
    a = k
    b = K - k
    c = n - k
    d = N - n - K + k

    # Ensure non-negative
    if any(x < 0 for x in [a, b, c, d]):
        print(f"WARNING: Negative cell in contingency table: a={a}, b={b}, c={c}, d={d}")
        b = max(b, 0)
        c = max(c, 0)
        d = max(d, 0)

    table = [[a, b], [c, d]]
    oddsratio, pvalue = fisher_exact(table, alternative='two-sided')

    # Woolf 95% CI
    if a > 0 and c > 0 and b > 0 and d > 0:
        se = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_low = np.exp(np.log(oddsratio) - 1.96 * se)
        ci_high = np.exp(np.log(oddsratio) + 1.96 * se)
    else:
        ci_low, ci_high = np.nan, np.nan

    return oddsratio, pvalue, ci_low, ci_high


def run_cmd(cmd, description="", timeout=3600):
    """Run a shell command and return the subprocess result."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"CMD: {cmd[:300]}...")
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        elapsed = time.time() - start
        print(f"TIME: {elapsed:.1f}s | EXIT: {result.returncode}")
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[:1000]}")
        return result
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s")
        return None


# ============================================================================
# G8: Drug Target Enrichment via OpenTargets
# ============================================================================
# WHY tractability-based approach instead of knownDrugs:
# The OpenTargets v4 GraphQL API removed the `knownDrugs` field from the
# Target type. The replacement approach uses the `tractability` field, which
# reports whether a gene has approved drugs (SM/AB/OC modalities with label
# "Approved Drug"). This is actually more reliable because it directly
# reflects clinical-stage drug development status.
#
# A gene is classified as a "drug target" if it has tractability entry with
# label "Approved Drug" and value=True in any modality (SM=small molecule,
# AB=antibody, OC=other clinical).

def query_opentargets_targets(gene_list):
    """Query OpenTargets for drug target information for each gene.

    Uses tractability data (v4 API) instead of the removed knownDrugs field.
    A gene is a drug target if tractability contains "Approved Drug: True"
    in any modality.
    """
    session = requests.Session()
    drug_targets = {}  # gene -> {has_drug, drug_modalities, disease_count, ...}

    for gene in tqdm(gene_list, desc="Querying OpenTargets"):
        try:
            # Step 1: Search for gene to get Ensembl ID
            search_query = {
                "query": """
                query($queryString: String!) {
                    search(queryString: $queryString, entityNames: ["target"]) {
                        hits { id entity name }
                    }
                }
                """,
                "variables": {"queryString": gene}
            }
            resp = session.post(OPENTARGETS_API, json=search_query, timeout=30)
            data = resp.json()

            # Find matching target (exact name match, case-insensitive)
            ensembl_id = None
            for hit in data.get('data', {}).get('search', {}).get('hits', []):
                if hit.get('entity') == 'target' and hit.get('name', '').upper() == gene:
                    ensembl_id = hit['id']
                    break

            if not ensembl_id:
                drug_targets[gene] = {'has_drug': False, 'drug_modalities': [],
                                       'ensembl_id': None, 'disease_count': 0,
                                       'has_ligand': False, 'has_clinical': False}
                continue

            # Step 2: Get tractability and disease associations using v4 API
            # The v4 API requires page: {index: N, size: M} for pagination
            detail_query = {
                "query": """
                query($ensemblId: String!) {
                    target(ensemblId: $ensemblId) {
                        id approvedSymbol
                        tractability { modality label value }
                        associatedDiseases(page: {index: 0, size: 1}) {
                            count
                        }
                    }
                }
                """,
                "variables": {"ensemblId": ensembl_id}
            }
            resp = session.post(OPENTARGETS_API, json=detail_query, timeout=30)
            detail_data = resp.json()
            target = detail_data.get('data', {}).get('target', {})

            if target:
                # Parse tractability: check for Approved Drug
                tract = target.get('tractability', [])
                approved_modalities = []
                has_ligand = False
                has_clinical = False

                for t in tract:
                    if t.get('value'):
                        label = t.get('label', '')
                        modality = t.get('modality', '')
                        if 'Approved Drug' in label:
                            approved_modalities.append(modality)
                        if label in ['High-Quality Ligand', 'Structure with Ligand']:
                            has_ligand = True
                        if 'Clinical' in label or 'Phase' in label:
                            has_clinical = True

                disease_count = target.get('associatedDiseases', {}).get('count', 0)
                drug_targets[gene] = {
                    'has_drug': len(approved_modalities) > 0,
                    'drug_modalities': approved_modalities,
                    'ensembl_id': ensembl_id,
                    'disease_count': disease_count,
                    'has_ligand': has_ligand,
                    'has_clinical': has_clinical,
                }
            else:
                drug_targets[gene] = {'has_drug': False, 'drug_modalities': [],
                                       'ensembl_id': ensembl_id, 'disease_count': 0,
                                       'has_ligand': False, 'has_clinical': False}

            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            print(f"  Error querying {gene}: {e}")
            drug_targets[gene] = {'has_drug': False, 'drug_modalities': [],
                                   'ensembl_id': None, 'disease_count': 0,
                                   'has_ligand': False, 'has_clinical': False,
                                   'error': str(e)}

    return drug_targets


def get_detailed_drug_info(ensembl_ids):
    """
    Get detailed disease association info for genes with known drugs.

    Since the v4 API removed knownDrugs, we get the top disease associations
    instead to provide context about the drug target.
    """
    session = requests.Session()
    drug_details = {}

    for gene, ensembl_id in tqdm(ensembl_ids.items(), desc="Getting disease details"):
        if not ensembl_id:
            continue
        try:
            query = {
                "query": """
                query($ensemblId: String!) {
                    target(ensemblId: $ensemblId) {
                        approvedSymbol
                        associatedDiseases(page: {index: 0, size: 5}) {
                            count
                            rows {
                                score
                                disease { id name }
                            }
                        }
                    }
                }
                """,
                "variables": {"ensemblId": ensembl_id}
            }
            resp = session.post(OPENTARGETS_API, json=query, timeout=30)
            data = resp.json()
            target = data.get('data', {}).get('target', {})
            diseases = []
            for row in target.get('associatedDiseases', {}).get('rows', []):
                diseases.append({
                    'disease': row.get('disease', {}).get('name', '?'),
                    'score': row.get('score', 0),
                })
            drug_details[gene] = diseases
            time.sleep(0.1)
        except Exception as e:
            print(f"  Error getting details for {gene}: {e}")

    return drug_details


def run_drug_target_enrichment(gene_list, gene_list_name):
    """Run G8: Drug target enrichment analysis.

    Uses OpenTargets v4 API tractability data. A gene is a drug target if
    it has "Approved Drug: True" in any tractability modality (SM, AB, OC).
    """
    print(f"\n{'='*60}")
    print(f"G8: Drug Target Enrichment -- {gene_list_name} (N={len(gene_list)})")
    print(f"{'='*60}")

    # Query OpenTargets
    drug_info = query_opentargets_targets(gene_list)

    # Count drug targets
    genes_with_drugs = [g for g, info in drug_info.items() if info.get('has_drug', False)]
    genes_with_ensembl = [g for g, info in drug_info.items() if info.get('ensembl_id')]
    genes_with_ligand = [g for g, info in drug_info.items() if info.get('has_ligand', False)]
    genes_with_clinical = [g for g, info in drug_info.items() if info.get('has_clinical', False)]
    print(f"\nGenes found in OpenTargets: {len(genes_with_ensembl)}/{len(gene_list)}")
    print(f"Genes with approved drugs: {len(genes_with_drugs)}/{len(gene_list)}")
    print(f"Genes with known ligands: {len(genes_with_ligand)}/{len(gene_list)}")
    print(f"Genes in clinical development: {len(genes_with_clinical)}/{len(gene_list)}")

    # Get detailed disease info for genes with drugs
    ensembl_ids = {g: info['ensembl_id'] for g, info in drug_info.items()
                   if info.get('ensembl_id') and info.get('has_drug')}
    drug_details = get_detailed_drug_info(ensembl_ids)

    # Print top drug targets (sorted by disease association count as proxy)
    print(f"\nTop drug targets in {gene_list_name}:")
    for gene in sorted(genes_with_drugs,
                       key=lambda g: drug_info[g].get('disease_count', 0), reverse=True)[:20]:
        modalities = drug_info[gene].get('drug_modalities', [])
        diseases = drug_info[gene].get('disease_count', 0)
        top_diseases = drug_details.get(gene, [])
        disease_names = [d['disease'] for d in top_diseases[:3]]
        print(f"  {gene}: modalities={modalities}, {diseases} disease associations. Top: {disease_names}")

    # Estimate background drug target rate.
    # OpenTargets defines ~3,500 genes as having approved drugs (conservative),
    # and ~4,500 when including clinical candidates (permissive).
    # Source: OpenTargets Platform statistics (https://platform.opentargets.org/statistics)
    K_background = 3500  # Conservative: ~17% of protein-coding genes
    N_background = BACKGROUND_N

    # Fisher's exact test
    k = len(genes_with_drugs)
    n = len(gene_list)
    K = K_background
    N = N_background

    or_val, p_val, ci_low, ci_high = fisher_exact_test(k, n, K, N)

    print(f"\nEnrichment test (conservative background K={K}):")
    print(f"  k={k} SCZ genes with approved drugs / {n} total SCZ genes")
    print(f"  K={K} drug targets / {N} background genes")
    print(f"  OR={or_val:.2f}, p={p_val:.4f}")
    if not np.isnan(ci_low):
        print(f"  95% CI: [{ci_low:.2f}, {ci_high:.2f}]")

    # Also test with more permissive background (include clinical candidates)
    K_permissive = 4500  # ~22% including clinical-phase drugs
    or_perm, p_perm, ci_low_perm, ci_high_perm = fisher_exact_test(k, n, K_permissive, N_background)
    print(f"\nPermissive background (K={K_permissive}, incl. clinical candidates):")
    print(f"  OR={or_perm:.2f}, p={p_perm:.4f}")
    if not np.isnan(ci_low_perm):
        print(f"  95% CI: [{ci_low_perm:.2f}, {ci_high_perm:.2f}]")

    return {
        'gene_list': gene_list_name,
        'n_genes': n,
        'n_with_drugs': k,
        'n_with_ligands': len(genes_with_ligand),
        'n_with_clinical': len(genes_with_clinical),
        'background_K_conservative': K,
        'background_K_permissive': K_permissive,
        'background_N': N_background,
        'or_conservative': round(or_val, 4),
        'p_conservative': round(p_val, 6),
        'ci_conservative': [round(ci_low, 4) if not np.isnan(ci_low) else None,
                            round(ci_high, 4) if not np.isnan(ci_high) else None],
        'or_permissive': round(or_perm, 4),
        'p_permissive': round(p_perm, 6),
        'ci_permissive': [round(ci_low_perm, 4) if not np.isnan(ci_low_perm) else None,
                          round(ci_high_perm, 4) if not np.isnan(ci_high_perm) else None],
        'drug_targets': {g: info for g, info in drug_info.items() if info.get('has_drug')},
        'drug_details': drug_details,
        'genes_found_in_opentargets': len(genes_with_ensembl),
    }


# ============================================================================
# G10: STRING PPI Enrichment
# ============================================================================
def query_string_enrichment(gene_list, gene_list_name):
    """Run G10: STRING PPI enrichment analysis."""
    print(f"\n{'='*60}")
    print(f"G10: STRING PPI Enrichment -- {gene_list_name} (N={len(gene_list)})")
    print(f"{'='*60}")

    session = requests.Session()

    # Step 1: Get STRING IDs for our genes
    # WHY POST with 'identifiers' (plural) instead of GET with 'identifier':
    # The STRING API requires POST for batch requests with multiple genes
    # separated by \r. GET requests only work for single gene lookups.
    identifiers = "\r".join(sorted(gene_list))
    try:
        resp = session.post(
            f"{STRING_API}/json/resolve",
            data={
                "identifiers": identifiers,
                "species": "9606",
            },
            timeout=120
        )
        resp.raise_for_status()
        string_ids = resp.json()
        mapped_genes = {r['queryItem']: r['stringId'] for r in string_ids if r.get('stringId')}
        print(f"Genes mapped to STRING: {len(mapped_genes)}/{len(gene_list)}")
    except Exception as e:
        print(f"Error resolving STRING IDs: {e}")
        return None

    if len(mapped_genes) < 10:
        print(f"WARNING: Only {len(mapped_genes)} genes mapped. Insufficient for enrichment.")
        return None

    # Step 2: Get PPI enrichment
    string_id_list = list(mapped_genes.values())
    identifiers_str = "\r".join(string_id_list)

    try:
        resp = session.post(
            f"{STRING_API}/json/ppi_enrichment",
            data={
                "identifiers": identifiers_str,
                "species": "9606",
            },
            timeout=60
        )
        resp.raise_for_status()
        enrichment = resp.json()

        if isinstance(enrichment, list) and len(enrichment) > 0:
            result = enrichment[0]
            print(f"\nPPI Enrichment Result:")
            print(f"  Number of nodes: {result.get('number_of_nodes', '?')}")
            print(f"  Number of edges: {result.get('number_of_edges', '?')}")
            print(f"  Expected edges: {result.get('expected_number_of_edges', '?')}")
            print(f"  Enrichment p-value: {result.get('p_value', '?')}")
            print(f"  FDR: {result.get('fdr', '?')}")
        else:
            result = enrichment
            print(f"Unexpected enrichment response: {enrichment}")

    except Exception as e:
        print(f"Error getting PPI enrichment: {e}")
        result = None

    # Step 3: Get functional enrichment (KEGG/GO)
    try:
        resp = session.post(
            f"{STRING_API}/json/enrichment",
            data={
                "identifiers": identifiers_str,
                "species": "9606",
                "category": "Process",  # GO Biological Process
            },
            timeout=60
        )
        resp.raise_for_status()
        go_enrichment = resp.json()

        print(f"\nTop GO Biological Process enrichments:")
        for term in go_enrichment[:15]:
            print(f"  {term.get('term', '?')}: FDR={term.get('fdr', '?')}, "
                  f"n={term.get('number_of_genes', '?')}, "
                  f"n_bg={term.get('number_of_genes_in_background', '?')}")
    except Exception as e:
        print(f"Error getting GO enrichment: {e}")
        go_enrichment = []

    # Step 4: Get KEGG pathway enrichment
    try:
        resp = session.post(
            f"{STRING_API}/json/enrichment",
            data={
                "identifiers": identifiers_str,
                "species": "9606",
                "category": "Pathway",
            },
            timeout=60
        )
        resp.raise_for_status()
        pathway_enrichment = resp.json()

        print(f"\nTop KEGG Pathway enrichments:")
        for term in pathway_enrichment[:10]:
            print(f"  {term.get('term', '?')}: FDR={term.get('fdr', '?')}, "
                  f"n={term.get('number_of_genes', '?')}")
    except Exception as e:
        print(f"Error getting KEGG enrichment: {e}")
        pathway_enrichment = []

    # Step 5: Get interaction network
    try:
        resp = session.post(
            f"{STRING_API}/json/network",
            data={
                "identifiers": identifiers_str,
                "species": "9606",
                "limit": "500",
            },
            timeout=60
        )
        resp.raise_for_status()
        interactions = resp.json()

        print(f"\nInteraction network: {len(interactions)} edges retrieved")

        # Find highest-confidence interactions
        high_conf = [i for i in interactions if i.get('score', 0) > 0.7]
        print(f"High-confidence interactions (score > 0.7): {len(high_conf)}")

        # Print top interactions
        print("\nTop interactions:")
        for inter in sorted(interactions, key=lambda x: x.get('score', 0), reverse=True)[:10]:
            print(f"  {inter.get('preferredName_A', '?')} <-> {inter.get('preferredName_B', '?')}: "
                  f"score={inter.get('score', 0):.3f}")

    except Exception as e:
        print(f"Error getting interaction network: {e}")
        interactions = []
        high_conf = []

    return {
        'gene_list': gene_list_name,
        'n_input_genes': len(gene_list),
        'n_mapped_genes': len(mapped_genes),
        'ppi_enrichment': result,
        'go_enrichment': go_enrichment[:50] if isinstance(go_enrichment, list) else [],
        'pathway_enrichment': pathway_enrichment[:50] if isinstance(pathway_enrichment, list) else [],
        'n_interactions': len(interactions) if isinstance(interactions, list) else 0,
        'n_high_conf_interactions': len(high_conf),
        'top_interactions': sorted(interactions, key=lambda x: x.get('score', 0), reverse=True)[:50]
                           if isinstance(interactions, list) else [],
        'mapped_genes': mapped_genes
    }


# ============================================================================
# D18: Sex-Stratified S-LDSC Partitioned Heritability
# ============================================================================
def munge_daner_to_ldsc(daner_path, output_path, label):
    """
    Convert PGC daner format summary statistics to LDSC sumstats format.

    daner columns: CHR, SNP, BP, A1, A2, FRQ_A_*, FRQ_U_*, INFO, OR, SE, P,
                   ngt, Direction, HetISqt, HetDf, HetPVa, Nca, Nco, Neff

    LDSC sumstats columns needed: SNP, CHR, BP, A1, A2, FRQ, BETA, SE, P, N

    WHY this approach:
    - Uses FRQ_U (control frequency) as population frequency estimate because
      cases have disease-affected allele frequencies; controls better reflect
      population allele frequencies for LD score computation.
    - BETA = log(OR) because LDSC requires effect sizes on the linear scale;
      daner provides OR, so we take the natural log.
    - Uses Neff (effective sample size) because it accounts for case/control
      imbalance and is the appropriate N for LDSC regression.
    - Filters INFO >= 0.9 and P > 0 to ensure quality SNPs only.
    """
    print(f"\n--- Munging {label} daner sumstats ---")
    print(f"Input: {daner_path}")
    print(f"Output: {output_path}")

    # Load daner file (tab-separated, gzipped)
    df = pd.read_csv(daner_path, sep='\t', compression='gzip')
    print(f"Loaded {len(df)} SNPs")

    # Quality filters
    # INFO >= 0.9: standard imputation quality threshold
    # P > 0: remove monomorphic or failed SNPs
    before = len(df)
    df = df[df['INFO'] >= 0.9].copy()
    print(f"After INFO >= 0.9 filter: {len(df)} (removed {before - len(df)})")
    before = len(df)
    df = df[df['P'] > 0].copy()
    print(f"After P > 0 filter: {len(df)} (removed {before - len(df)})")

    # Find the FRQ_U column (control frequency)
    # Column name format: FRQ_U_<N> where N is sample size
    frq_u_cols = [c for c in df.columns if c.startswith('FRQ_U_')]
    if not frq_u_cols:
        print(f"WARNING: No FRQ_U column found. Available: {list(df.columns)}")
        print("Falling back to FRQ_A columns...")
        frq_a_cols = [c for c in df.columns if c.startswith('FRQ_A_')]
        frq_col = frq_a_cols[0]
    else:
        frq_col = frq_u_cols[0]
    print(f"Using frequency column: {frq_col}")

    # Compute BETA = log(OR)
    # Guard against OR <= 0 (should not happen but numerical safety)
    df = df[df['OR'] > 0].copy()
    df['BETA'] = np.log(df['OR'])

    # Build LDSC sumstats
    sumstats = pd.DataFrame({
        'SNP': df['SNP'],
        'CHR': df['CHR'],
        'BP': df['BP'],
        'A1': df['A1'],
        'A2': df['A2'],
        'FRQ': df[frq_col],
        'BETA': df['BETA'],
        'SE': df['SE'],
        'P': df['P'],
        'N': df['Neff'],
    })

    # Remove any rows with NaN in critical columns
    sumstats = sumstats.dropna(subset=['SNP', 'CHR', 'BP', 'A1', 'A2', 'BETA', 'SE', 'P', 'N'])
    print(f"Final sumstats: {len(sumstats)} SNPs")

    # Write as tab-separated, UNCOMPRESSED.
    # WHY uncompressed: munge_sumstats.py has a Python 3 compatibility bug in its
    # read_header function when opening gzipped files (bytes vs str). Writing
    # uncompressed avoids this issue.
    output_path_uncompressed = output_path.replace('.sumstats.gz', '.sumstats')
    sumstats.to_csv(output_path_uncompressed, sep='\t', index=False, float_format='%.6f')
    print(f"Written to {output_path_uncompressed}")
    return output_path_uncompressed

    # Print summary stats
    print(f"\n  N median: {sumstats['N'].median():.1f}")
    print(f"  N range: [{sumstats['N'].min():.1f}, {sumstats['N'].max():.1f}]")
    print(f"  BETA range: [{sumstats['BETA'].min():.4f}, {sumstats['BETA'].max():.4f}]")
    print(f"  P range: [{sumstats['P'].min():.2e}, {sumstats['P'].max():.2e}]")
    print(f"  Mean chi2: {np.mean(sumstats['P'].apply(lambda p: stats.chi2.isf(p, 1))):.4f}")

    return output_path


def munge_sumstats_with_ldsc(raw_sumstats_path, out_prefix, merge_snps):
    """
    Run ldsc munge_sumstats.py on the converted sumstats.

    WHY we use munge_sumstats.py rather than feeding raw sumstats to --h2:
    LDSC requires sumstats in its internal format (with Z-scores computed,
    SNPs matched to the LD reference panel, and allele harmonization).
    munge_sumstats.py handles all of this.

    WHY --signed-sumstats BETA,0:
    The LDSC munge_sumstats.py does not have --beta/--se flags. Instead it
    uses --signed-sumstats which takes column_name,null_value format.
    BETA,0 means "use the BETA column, where 0 is the null value."
    """
    cmd = (
        f"python3 {MUNGE} "
        f"--sumstats {raw_sumstats_path} "
        f"--merge-alleles {merge_snps} "
        f"--out {out_prefix} "
        f"--snp SNP --a1 A1 --a2 A2 --p P "
        f"--signed-sumstats BETA,0 "
        f"--N-col N --frq FRQ"
    )
    result = run_cmd(cmd, f"Munge sumstats: {out_prefix}", timeout=600)

    if result and result.returncode == 0:
        munged_path = f"{out_prefix}.sumstats.gz"
        if os.path.exists(munged_path):
            print(f"Munged sumstats: {munged_path}")
            return munged_path
    else:
        # Print the log for debugging
        log_path = f"{out_prefix}.log"
        if os.path.exists(log_path):
            with open(log_path) as f:
                print(f"Munge log:\n{f.read()[-2000:]}")

    return None


def run_sldsc_partitioned_heritability(munged_sumstats, out_prefix, label):
    """
    Run S-LDSC partitioned heritability with cell-type + baselineLD annotations.

    This reuses the exact same pipeline as batch_034, which produced:
    - neuronal enrichment 1.83x, p=0.009 (combined EUR)
    - oligodendrocyte 0.52x, p=0.089
    - astrocyte 0.78x, p=0.734
    - OPC 3.15x, p=0.315

    WHY these parameters:
    - --ref-ld-chr celltype.,baselineLD.: Joint model with cell-type annotations
      alongside the baselineLD model (100+ annotations). This controls for known
      functional genomic annotations when testing cell-type enrichment.
    - --overlap-annot: Required because cell-type annotations partially overlap
      with baselineLD annotations (e.g., both may tag coding regions).
    - --w-ld-chr weights.hm3_noMHC.: LD score regression weights computed from
      1000G Phase 3 EUR, restricted to HapMap3 SNPs outside MHC.
    - --frqfile-chr: Population allele frequencies for MAF-stratified analyses.
    """
    print(f"\n--- S-LDSC partitioned heritability: {label} ---")

    # Use the baselineLD_hm3_snps.txt from batch_034 which includes MHC SNPs
    # (batch_034 discovered that weights hm3_snps.txt excludes MHC, causing
    # chr6 row count mismatches)
    if os.path.exists(BASELINELD_HM3_SNPS):
        merge_snps = BASELINELD_HM3_SNPS
        print(f"Using baselineLD_hm3_snps.txt (includes MHC): {merge_snps}")
    else:
        merge_snps = HM3_SNPS
        print(f"baselineLD_hm3_snps.txt not found, falling back to: {merge_snps}")

    # Step 1: Munge sumstats
    munged = munge_sumstats_with_ldsc(munged_sumstats, out_prefix + "_munged", merge_snps)
    if not munged:
        print(f"ERROR: Failed to munge sumstats for {label}")
        return None

    # Step 2: Run partitioned h2
    cmd = (
        f"python3 {LDSC} "
        f"--h2 {munged} "
        f"--ref-ld-chr {CELLTYPE_LDSCORES},{BASELINE_LD} "
        f"--w-ld-chr {WEIGHTS} "
        f"--overlap-annot "
        f"--frqfile-chr {FRQ_REF} "
        f"--out {out_prefix}_partitioned"
    )
    result = run_cmd(cmd, f"S-LDSC partitioned h2: {label}", timeout=1800)

    if result and result.returncode == 0:
        results_file = f"{out_prefix}_partitioned.results"
        if os.path.exists(results_file):
            # Parse and display results
            results_df = pd.read_csv(results_file, sep='\t')
            print(f"\n{label} S-LDSC Results:")
            print(results_df.to_string(index=False))

            # Extract cell-type specific results (first 4 rows: neuronal, oligodendrocyte, astrocyte, OPC)
            celltype_results = {}
            celltype_names = ['neuronal', 'oligodendrocyte', 'astrocyte', 'OPC']
            for i, row in results_df.head(4).iterrows():
                ct_name = celltype_names[i] if i < len(celltype_names) else f"celltype_{i}"
                celltype_results[ct_name] = {
                    'prop_snps': row.get('Prop._SNPs', float('nan')),
                    'prop_h2': row.get('Prop._h2', float('nan')),
                    'prop_h2_se': row.get('Prop._h2_std_error', float('nan')),
                    'enrichment': row.get('Enrichment', float('nan')),
                    'enrichment_se': row.get('Enrichment_std_error', float('nan')),
                    'enrichment_p': row.get('Enrichment_p', float('nan')),
                }
                print(f"\n  {ct_name}: enrichment={row.get('Enrichment', float('nan')):.4f}, "
                      f"p={row.get('Enrichment_p', float('nan')):.4f}, "
                      f"prop_h2={row.get('Prop._h2', float('nan')):.4f}")

            # Also get overall h2 from log file
            log_file = f"{out_prefix}_partitioned.log"
            overall_h2 = None
            if os.path.exists(log_file):
                with open(log_file) as f:
                    log_text = f.read()
                # Parse h2 from log
                for line in log_text.split('\n'):
                    if 'Total Observed scale h2' in line:
                        parts = line.split(':')[-1].strip()
                        try:
                            h2_val = float(parts.split('(')[0].strip())
                            overall_h2 = h2_val
                        except (ValueError, IndexError):
                            pass
                    if 'Mean Chi^2' in line:
                        print(f"  {line.strip()}")
                    if 'Lambda GC' in line:
                        print(f"  {line.strip()}")

            print(f"\n  Overall h2: {overall_h2}")

            return {
                'results_file': results_file,
                'celltype_enrichment': celltype_results,
                'overall_h2': overall_h2,
                'label': label,
            }

    print(f"ERROR: S-LDSC partitioned h2 failed for {label}")
    return None


def run_sex_stratified_sldsc():
    """
    Run D18: Sex-stratified S-LDSC partitioned heritability.

    WHY S-LDSC instead of minP gene-level approach:
    The minP approach (batch_033/d18_sex_stratification.py) failed because
    sex-stratified GWAS are underpowered at the single-gene level. S-LDSC
    tests for aggregate enrichment of heritability in cell-type annotations,
    which is more powerful for detecting cell-type effects even when
    individual genes do not reach genome-wide significance.

    The cell-type LD scores are already computed in batch_034, so we only
    need to munge the sex-stratified sumstats and run the partitioned h2
    regression.
    """
    print(f"\n{'='*60}")
    print(f"D18: Sex-Stratified S-LDSC Partitioned Heritability")
    print(f"{'='*60}")

    results = {}

    for sex, daner_path in [('female', EUR_FEMALE_PATH), ('male', EUR_MALE_PATH)]:
        label = f"EUR_{sex}"
        print(f"\n\n{'='*40}")
        print(f"Processing: {label}")
        print(f"{'='*40}")

        if not daner_path.exists():
            print(f"ERROR: File not found: {daner_path}")
            results[sex] = {'error': 'file not found'}
            continue

        # Step 1: Convert daner to LDSC sumstats format
        raw_sumstats = str(OUTPUT_DIR / f"sumstats_{sex}.sumstats.gz")
        munge_result = munge_daner_to_ldsc(daner_path, raw_sumstats, label)

        if not munge_result:
            results[sex] = {'error': 'daner conversion failed'}
            continue

        # Step 2: Run S-LDSC partitioned heritability
        # Use munge_result (uncompressed path) not raw_sumstats (may be .gz)
        out_prefix = str(OUTPUT_DIR / f"sldsc_{sex}")
        sldsc_result = run_sldsc_partitioned_heritability(munge_result, out_prefix, label)

        if sldsc_result:
            results[sex] = sldsc_result
        else:
            results[sex] = {'error': 'S-LDSC failed'}

    # Compare male vs female
    print(f"\n\n{'='*60}")
    print(f"MALE vs FEMALE COMPARISON")
    print(f"{'='*60}")

    if 'female' in results and 'male' in results:
        f_result = results['female']
        m_result = results['male']

        if 'celltype_enrichment' in f_result and 'celltype_enrichment' in m_result:
            print(f"\nCell-type enrichment comparison:")
            print(f"{'Cell Type':<20} {'Female Enrich':>15} {'Female p':>12} {'Male Enrich':>15} {'Male p':>12}")
            print("-" * 75)

            for ct in ['neuronal', 'oligodendrocyte', 'astrocyte', 'OPC']:
                f_ct = f_result['celltype_enrichment'].get(ct, {})
                m_ct = m_result['celltype_enrichment'].get(ct, {})

                f_enrich = f_ct.get('enrichment', float('nan'))
                f_p = f_ct.get('enrichment_p', float('nan'))
                m_enrich = m_ct.get('enrichment', float('nan'))
                m_p = m_ct.get('enrichment_p', float('nan'))

                print(f"{ct:<20} {f_enrich:>15.4f} {f_p:>12.4f} {m_enrich:>15.4f} {m_p:>12.4f}")

            # Reference: combined EUR from batch_034
            print(f"\nReference (combined EUR, batch_034):")
            print(f"  Neuronal: 1.83x, p=0.009")
            print(f"  Oligodendrocyte: 0.52x, p=0.089")
            print(f"  Astrocyte: 0.78x, p=0.734")
            print(f"  OPC: 3.15x, p=0.315")

            # h2 comparison
            print(f"\nHeritability comparison:")
            print(f"  Female h2: {f_result.get('overall_h2', 'N/A')}")
            print(f"  Male h2: {m_result.get('overall_h2', 'N/A')}")
            print(f"  Combined EUR h2 (batch_034): 0.8215")

    return results


# ============================================================================
# BH-FDR Correction
# ============================================================================
def apply_bh_fdr(p_values):
    """
    Apply Benjamini-Hochberg FDR correction to a list of p-values.

    Returns list of (original_p, fdr_q) tuples.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # BH procedure: q_i = p_i * n / rank_i
    fdr = [0.0] * n
    for rank_minus_1, (orig_idx, p) in enumerate(indexed):
        rank = rank_minus_1 + 1
        fdr[orig_idx] = p * n / rank

    # Enforce monotonicity (step-up): q_i = min(q_i, q_{i+1})
    # Process from largest rank to smallest
    prev_min = 1.0
    for rank_minus_1 in range(n - 1, -1, -1):
        orig_idx = indexed[rank_minus_1][0]
        fdr[orig_idx] = min(fdr[orig_idx], prev_min)
        prev_min = fdr[orig_idx]

    return list(zip(p_values, fdr))


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("batch_035: Drug Target Enrichment + STRING PPI + Sex-Stratified S-LDSC")
    print("=" * 70)

    # Load gene lists
    pardinas_genes, pgc3_genes, combined_genes = load_gene_lists()

    all_results = {}

    # ---- G8: Drug Target Enrichment ----
    print("\n\n" + "=" * 70)
    print("ANALYSIS 1: Drug Target Enrichment (G8)")
    print("=" * 70)

    # Run for Pardinas gene list
    all_results['G8_pardinas'] = run_drug_target_enrichment(pardinas_genes, "Pardinas_444")

    # Run for PGC3 gene list
    all_results['G8_pgc3'] = run_drug_target_enrichment(pgc3_genes, "PGC3_106")

    # BH-FDR correction across both gene lists (2 tests for conservative + 2 for permissive)
    print(f"\n--- BH-FDR correction for G8 (2 gene lists x 2 backgrounds = 4 tests) ---")
    g8_pvals = []
    g8_labels = []
    for key, bg_type in [('G8_pardinas', 'conservative'), ('G8_pardinas', 'permissive'),
                          ('G8_pgc3', 'conservative'), ('G8_pgc3', 'permissive')]:
        r = all_results.get(key, {})
        p_val = r.get(f'p_{bg_type}', 1.0)
        g8_pvals.append(p_val)
        g8_labels.append(f"{key}/{bg_type}")

    g8_fdr = apply_bh_fdr(g8_pvals)
    print(f"{'Test':<40} {'p-value':>12} {'FDR q':>12}")
    print("-" * 65)
    for label, (p, q) in zip(g8_labels, g8_fdr):
        sig = " *" if q < 0.05 else ""
        print(f"{label:<40} {p:>12.6f} {q:>12.6f}{sig}")

    all_results['G8_fdr'] = {label: {'p': p, 'q': q} for label, (p, q) in zip(g8_labels, g8_fdr)}

    # ---- G10: STRING PPI Enrichment ----
    print("\n\n" + "=" * 70)
    print("ANALYSIS 2: STRING PPI Enrichment (G10)")
    print("=" * 70)

    # Run for Pardinas gene list
    all_results['G10_pardinas'] = query_string_enrichment(pardinas_genes, "Pardinas_444")

    # Run for PGC3 gene list
    all_results['G10_pgc3'] = query_string_enrichment(pgc3_genes, "PGC3_106")

    # ---- D18: Sex-Stratified S-LDSC ----
    print("\n\n" + "=" * 70)
    print("ANALYSIS 3: Sex-Stratified S-LDSC Partitioned Heritability (D18)")
    print("=" * 70)

    all_results['D18_sex_stratified'] = run_sex_stratified_sldsc()

    # ---- Save results ----
    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, (float, int)) and (np.isnan(obj) if isinstance(obj, float) else False):
            return None
        return obj

    all_results = make_serializable(all_results)

    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # G8 summary
    for key in ['G8_pardinas', 'G8_pgc3']:
        if key in all_results:
            r = all_results[key]
            print(f"\n{key}:")
            print(f"  Drug targets: {r.get('n_with_drugs', '?')}/{r.get('n_genes', '?')} genes")
            print(f"  With ligands: {r.get('n_with_ligands', '?')}")
            print(f"  In clinical dev: {r.get('n_with_clinical', '?')}")
            print(f"  Conservative: OR={r.get('or_conservative', '?')}, p={r.get('p_conservative', '?')}")
            print(f"  Permissive:   OR={r.get('or_permissive', '?')}, p={r.get('p_permissive', '?')}")
            # FDR
            fdr_key = f"{key}/conservative"
            if fdr_key in all_results.get('G8_fdr', {}):
                print(f"  FDR (conservative): q={all_results['G8_fdr'][fdr_key].get('q', '?')}")

    # G10 summary
    for key in ['G10_pardinas', 'G10_pgc3']:
        if key in all_results and all_results[key]:
            r = all_results[key]
            print(f"\n{key}:")
            ppi = r.get('ppi_enrichment', {})
            if ppi:
                print(f"  Nodes: {ppi.get('number_of_nodes', '?')}")
                print(f"  Edges: {ppi.get('number_of_edges', '?')} (expected: {ppi.get('expected_number_of_edges', '?')})")
                print(f"  PPI enrichment p: {ppi.get('p_value', '?')}")

    # D18 summary
    if 'D18_sex_stratified' in all_results:
        print("\nD18_sex_stratified:")
        for sex in ['female', 'male']:
            r = all_results['D18_sex_stratified'].get(sex, {})
            if 'celltype_enrichment' in r:
                ct = r['celltype_enrichment']
                neur = ct.get('neuronal', {})
                print(f"  {sex}: h2={r.get('overall_h2', '?')}, "
                      f"neuronal enrichment={neur.get('enrichment', '?')}, "
                      f"p={neur.get('enrichment_p', '?')}")
            elif 'error' in r:
                print(f"  {sex}: ERROR - {r['error']}")


if __name__ == '__main__':
    main()
