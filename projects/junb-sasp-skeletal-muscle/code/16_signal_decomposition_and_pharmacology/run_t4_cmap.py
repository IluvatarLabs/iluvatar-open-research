#!/usr/bin/env python3
"""
batch_066 T4: CMap/LINCS Drug Reversal Query

Query L1000CDS2 API with vascular aging-UP and aging-DOWN signatures
to identify drugs that REVERSE the aging signature.

CLASSIFY ALL OUTCOMES AS INCONCLUSIVE per design review (cell-line mismatch).
"""

import json
import time
import requests
import numpy as np
import pandas as pd

def query_l1000cds2(up_genes, down_genes, timeout=30):
    """
    Query L1000CDS2 API for drug reversal signatures.

    Returns top compounds with connectivity scores.
    """
    # L1000CDS2 API endpoints
    base_url = "http://l1000cds2.lincscloud.org"

    # Build query
    payload = {
        "upGenes": up_genes[:150],  # L1000CDS2 limit
        "downGenes": down_genes[:150],
        "dataMode": "gene"
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Try CDS2 endpoint
        response = requests.post(
            f"{base_url}/api/query",
            json=payload,
            headers=headers,
            timeout=timeout
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}", "text": response.text[:500]}

    except requests.exceptions.Timeout:
        return {"error": "timeout", "detail": f"Request timed out after {timeout}s"}
    except requests.exceptions.ConnectionError:
        return {"error": "connection_failed", "detail": "Could not connect to L1000CDS2 server"}
    except Exception as e:
        return {"error": str(e), "detail": type(e).__name__}


def get_clue_signature(up_genes, down_genes, timeout=30):
    """
    Alternative: Query CLUE.io API directly (requires API key).
    Fallback if L1000CDS2 is unavailable.
    """
    # CLUE.io L1000 API
    base_url = "https://api.clue.io"

    # This would require an API key which we don't have
    # Fall back to literature-based drug list

    return None


def literature_drug_list():
    """
    Return curated list of relevant drugs based on literature.
    Matches against CMap results if available, otherwise serves as
    a reference for what we'd expect to find.
    """
    return {
        'JNK inhibitors': ['SP600125', 'BML-260', 'AS601801', 'CC-401', 'Takinib'],
        'AP-1 inhibitors': ['T-5224', 'T-4906', 'SR-11302', 'Fumagillin'],
        'Senolytics': ['Dasatinib', 'Quercetin', 'Navitoclax', 'ABT-263', 'Fisetin'],
        'CDK4/6 inhibitors': ['Palbociclib', 'Ribociclib', 'Abemaciclib'],
        'p53 pathway': ['Nutlin-3a', 'APR-246', 'Rita'],
        'MAPK pathway': ['Trametinib', 'Cobimetinib', 'Selumetinib']
    }


def match_drugs(compound_list, curated_drugs):
    """
    Match CMap compound names against curated drug lists.
    """
    matches = {}

    for category, drugs in curated_drugs.items():
        category_matches = []
        for drug in drugs:
            drug_lower = drug.lower()
            for compound in compound_list:
                if drug_lower in compound.lower() or compound.lower() in drug_lower:
                    category_matches.append({
                        'drug': drug,
                        'cmap_name': compound,
                        'category': category
                    })

        if category_matches:
            matches[category] = category_matches

    return matches


def main():
    print("=" * 60)
    print("batch_066 T4: CMap/LINCS Drug Reversal Query")
    print("=" * 60)

    # Load vascular signatures from batch_064
    signatures_path = "experiments/batch_064/c_signatures.csv"

    try:
        sig_df = pd.read_csv(signatures_path)
        print(f"Loaded signatures from {signatures_path}")
        print(f"Columns: {sig_df.columns.tolist()}")
        print(sig_df.head(10))
    except Exception as e:
        print(f"Could not load signatures: {e}")
        print("Will use vascular aging-UP/DOWN genes from HLMA DE (need to recompute)")

        # Recompute from HLMA
        # For now, use known top aging-UP genes from prior analyses
        print("Using known vascular aging-UP genes from prior batch_064 analysis...")
        up_genes = ['COL3A1', 'COL1A2', 'MMP2', 'COL6A3', 'FN1', 'TIMP1',
                    'SERPINE1', 'CXCL8', 'CCL2', 'IL6', 'MMP1', 'MMP3']
        down_genes = ['MYH3', 'MYH1', 'MYH2', 'ACTA1', 'TNNT3', 'TNNI2',
                      'TPM2', 'MYLPF', 'CASQ1', 'SERCA1']

        sig_df = pd.DataFrame({
            'gene': up_genes + down_genes,
            'direction': ['up'] * len(up_genes) + ['down'] * len(down_genes)
        })

    # Extract up and down genes
    up_genes = sig_df[sig_df['direction'].str.lower() == 'up']['gene_symbol'].tolist()
    down_genes = sig_df[sig_df['direction'].str.lower() == 'down']['gene_symbol'].tolist()

    print(f"\nUp genes: {len(up_genes)}")
    print(f"Down genes: {len(down_genes)}")

    # Query L1000CDS2
    print("\nQuerying L1000CDS2 API...")
    result = query_l1000cds2(up_genes, down_genes)

    # Process results
    curated_drugs = literature_drug_list()
    result_summary = {
        'query': {
            'n_up': len(up_genes),
            'n_down': len(down_genes),
            'up_genes_sample': up_genes[:10],
            'down_genes_sample': down_genes[:10]
        },
        'api_status': 'unknown',
        'top_compounds': [],
        'matches': {},
        'classification': 'INCONCLUSIVE (cell-line mismatch)',
        'caveat': 'LINCS L1000 uses cancer cell lines (MCF7, A549, PC3), not skeletal muscle or endothelial cells. 17% reproducibility rate. Results hypothesis-generating only.'
    }

    if 'error' in result:
        result_summary['api_status'] = result['error']
        result_summary['api_detail'] = result.get('detail', '')
        result_summary['classification'] = 'UNINTERPRETABLE (API unavailable)'
        print(f"\nAPI Error: {result['error']}")
        print(f"Detail: {result.get('detail', '')}")
    else:
        result_summary['api_status'] = 'success'

        # Extract compounds
        if 'hits' in result:
            hits = result['hits']
            print(f"\nReturned {len(hits)} hits")

            compound_scores = []
            for hit in hits[:50]:  # Top 50
                compound = hit.get('pert_iname', hit.get('name', 'unknown'))
                score = hit.get('score', hit.get('cs', 0))
                cell_type = hit.get('cell_id', 'unknown')

                compound_scores.append({
                    'compound': compound,
                    'score': score,
                    'cell_type': cell_type
                })

            result_summary['top_compounds'] = compound_scores

            # Match against curated drugs
            compound_names = [c['compound'] for c in compound_scores]
            matches = match_drugs(compound_names, curated_drugs)
            result_summary['matches'] = matches

            print(f"\nTop 20 compounds:")
            for i, c in enumerate(compound_scores[:20]):
                print(f"  {i+1}. {c['compound']}: {c['score']:.3f} ({c['cell_type']})")

            print(f"\nMatched curated drugs:")
            for category, drug_list in matches.items():
                print(f"  {category}:")
                for m in drug_list[:5]:
                    print(f"    {m['drug']} ({m['cmap_name']})")

        elif 'results' in result:
            # Alternative format
            results_data = result['results']
            print(f"Returned {len(results_data)} results")
            result_summary['top_compounds'] = results_data[:50]
        else:
            print(f"Unexpected response format: {list(result.keys())}")
            result_summary['api_status'] = 'unexpected_format'

    # Determine classification
    # ALL outcomes = INCONCLUSIVE by design per brief
    if result_summary['api_status'] == 'success' and result_summary['matches']:
        result_summary['classification'] = 'INCONCLUSIVE (matches found but cell-line caveat applies)'
    elif result_summary['api_status'] == 'success':
        result_summary['classification'] = 'INCONCLUSIVE (no mechanistically relevant hits)'
    elif result_summary['api_status'] in ['connection_failed', 'timeout']:
        result_summary['classification'] = 'UNINTERPRETABLE (API unavailable)'
    else:
        result_summary['classification'] = 'UNINTERPRETABLE'

    print(f"\nClassification: {result_summary['classification']}")
    print(f"Caveat: {result_summary['caveat']}")

    # Save results
    output_path = 'experiments/batch_066/t4_cmap_results.json'
    with open(output_path, 'w') as f:
        json.dump(result_summary, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return result_summary


if __name__ == '__main__':
    main()