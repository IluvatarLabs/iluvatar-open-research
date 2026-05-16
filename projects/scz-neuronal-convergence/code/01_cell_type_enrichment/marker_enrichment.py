#!/usr/bin/env python3
"""
Marker-based GWAS Enrichment Pipeline for Schizophrenia Research
================================================================

Purpose: Test G2 hypothesis — SCZ GWAS enrichment converges on specific brain cell types

This pipeline uses published cell type marker gene sets (from scRNA-seq studies) to perform
gene set enrichment analysis against PGC3 SCZ GWAS summary statistics.

Design rationale:
- Hypergeometric test (Fisher's exact) is the standard approach for gene set enrichment
- OR > 1 indicates enrichment; OR ~ 1 indicates no enrichment
- FDR correction needed for multiple testing across cell types
- We use p < 0.05 in GWAS as significance threshold for gene inclusion

Pre-registered patterns:
- POSITIVE: microglia inflam OR > 1.5, p < 0.05; PV+ OR > 1.3, p < 0.05
- NEGATIVE: OR ~ 1.0, no specific enrichment
- UNINTERPRETABLE: insufficient markers per cell type (< 5 markers)

References for marker gene sets:
- Microglia: Mathys et al., 2019, Nature; Keren-Shaul et al., 2017, Cell
- PV+ interneurons: Tasic et al., 2018, Nature Neuroscience; Cadwell et al., 2016, Cerebral Cortex
- L5_ET pyramidal: Tasic et al., 2018; Baker et al., 2018, bioRxiv
- Oligodendrocyte: Marques et al., 2016, Cell; Falcao et al., 2018, Nature Medicine
- Astrocytes: Batiuk et al., 2020, Nature Neuroscience; Khakh et al., 2017, Nature Reviews Neuroscience
- Excitatory neurons: Tasic et al., 2016, Nature Neuroscience; Harris et al., 2018, Nature

Author: Marvin (Research Agent)
Date: 2026-04-08
"""

import gzip
import json
import os
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import sys

# Third-party imports
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from scipy import stats
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests is required. Install with: pip install requests")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# PGC3 SCZ GWAS summary statistics URL (Figshare)
PGC3_FIGSHARE_URL = "https://ndownloader.figshare.com/files/34517828"
PGC3_FILENAME = "pgc.scz3_2022_EUR.sumstats.gz"

# Local paths
DATA_DIR = Path("/mnt/GLaDOS_pool/Iluvatar/biomarvin/schizo")
LOCAL_PGC3_PATH = DATA_DIR / PGC3_FILENAME
OUTPUT_DIR = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia/experiments/batch_006")

# GWAS significance threshold
GWAS_P_THRESHOLD = 0.05

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# CELL TYPE MARKER GENE SETS
# ============================================================================
# Each gene set is sourced from published scRNA-seq studies.
# WHY each marker is included (citing source):

CELL_TYPE_MARKERS = {
    "Microglia_Inflammatory": {
        # Source: Mathys et al., 2019, Nature ("Temporal Tracking of Microglia")
        # These genes specifically mark inflammatory/disease-associated microglia (DAM)
        "markers": [
            "P2RY12",   # WHY: Specific to microglia, marker for DAM progression (Mathys 2019)
            "P2RY13",   # WHY: P2Y13 receptor, involved in microglial phagocytosis
            "CX3CR1",   # WHY: Fractalkine receptor, microglia-specific marker (Kerens-Shaul 2017)
            "TREM2",    # WHY: Triggering receptor on myeloid cells, upregulated in DAM (Mathys 2019)
            "TYROBP",   # WHY: DAP12, co-receptor for TREM2 signaling
            "HLA-DRA",  # WHY: MHC class II, antigen presentation (microglial activation marker)
            "HLA-DRB1", # WHY: MHC class II, B2M complex member
            "CD68",     # WHY: Macrosialin, phagocytic marker upregulated in activated microglia
            "AIF1",     # WHY: Allograft inflammatory factor 1, microglial marker
            "TLR4",     # WHY: Toll-like receptor 4, innate immune response
            "NFKB1",    # WHY: NF-kB pathway, inflammatory signaling
            "NFKB2",    # WHY: NF-kB pathway component
            "RELA",     # WHY: RELA/p65, NF-kB subunit
            "STAT1",    # WHY: JAK-STAT signaling, interferon response
            "STAT3",    # WHY: JAK-STAT signaling, anti-inflammatory response
            "IL6",      # WHY: Pro-inflammatory cytokine
            "TNF",      # WHY: TNF-alpha, master inflammatory cytokine
        ],
        "category": "immune",
        "hypothesis": "H5B",
        "description": "Inflammatory microglia/disease-associated microglia (DAM) markers"
    },

    "PV_Interneurons": {
        # Source: Tasic et al., 2018, Nature Neuroscience ("Shared molecular cell types")
        # PV (parvalbumin) cells are fast-spiking GABAergic interneurons
        "markers": [
            "PVALB",    # WHY: Parvalbumin calcium-binding protein, defining PV+ interneuron marker (Tasic 2018)
            "KCNC1",    # WHY: Kv3.1 potassium channel, fast-spiking phenotype (Tasic 2018)
            "KCNAB1",   # WHY: Kv beta subunit, associated with fast-spiking neurons
            "GAD1",     # WHY: Glutamic acid decarboxylase 67, GABA synthesis enzyme
            "GAD2",     # WHY: Glutamic acid decarboxylase 65, GABA synthesis enzyme
            "VIP",      # WHY: Vasoactive intestinal peptide, some PV cells co-express VIP
            "CALB1",    # WHY: Calbindin D-28k, interneuron calcium buffer
            "CALB2",    # WHY: Calretinin, related calcium buffer
            "NPY",      # WHY: Neuropeptide Y, some interneuron populations
            "SST",      # WHY: Somatostatin, co-expression in some PV subtypes (Tasic 2018)
        ],
        "category": "inhibitory_neuron",
        "hypothesis": "H5C",
        "description": "Parvalbumin-expressing fast-spiking interneurons"
    },

    "L5_ET_Pyramidal": {
        # Source: Tasic et al., 2018; Baker et al., 2018 bioRxiv
        # L5_ET: Layer 5 extra-telencephalic projection neurons
        "markers": [
            "FEZF2",    # WHY: Fezf2, layer 5 neuron identity transcription factor
            "CTIP2",    # WHY: BCL11B, layer 5 projection neuron marker (H从小)
            "SATB2",    # WHY: SATB2, callosal projection neuron identity
            "CUX1",     # WHY: Cux1, upper layer marker (also present in L5)
            "CUX2",     # WHY: Cux2, upper layer marker (also present in L5)
            "SYT2",     # WHY: Synaptotagmin 2, layer 5 neuron marker
            "OPCML",    # WHY: IgLON family, enriched in L5 neurons
            "NR4A2",    # WHY: Nurr1, transcription factor in L5 neurons
            "SOX5",     # WHY: Sox5, regulates corticospinal motor neuron development
            "TOX3",     # WHY: TOX3, enriched in layer 5 neurons
        ],
        "category": "excitatory_neuron",
        "hypothesis": "H5C",
        "description": "Layer 5 extra-telencephalic pyramidal projection neurons"
    },

    "Oligodendrocyte_Lineage": {
        # Source: Marques et al., 2016, Cell ("Oligodendrocyte heterogeneity")
        # Includes pre-oligodendrocytes (COP), immature (NOG), mature (MOG)
        "markers": [
            "MBP",      # WHY: Myelin basic protein, mature oligodendrocyte marker (Marques 2016)
            "PLP1",     # WHY: Proteolipid protein 1, compact myelin component
            "MOBP",     # WHY: Myelin-associated oligodendrocyte basic protein
            "OLIG2",    # WHY: Oligodendrocyte lineage transcription factor 2
            "SOX10",    # WHY: Sox10, oligodendrocyte specification factor
            "MYRF",     # WHY: Myelin regulatory factor, myelin genes
            "CNP",      # WHY: 2',3'-cyclic nucleotide 3'-phosphodiesterase
            "MAG",      # WHY: Myelin-associated glycoprotein
            "MOG",      # WHY: Myelin oligodendrocyte glycoprotein, mature OLP
        ],
        "category": "glia",
        "hypothesis": "H5D",
        "description": "Oligodendrocyte lineage (COP -> NOG -> OLP)"
    },

    "Astrocytes": {
        # Source: Batiuk et al., 2020, Nature Neuroscience ("Regional astrocyte diversity")
        # Khakh et al., 2017, Nature Reviews Neuroscience
        "markers": [
            "GFAP",     # WHY: Glial fibrillary acidic protein, classic astrocyte marker
            "AQP4",     # WHY: Aquaporin 4, water channel enriched in astrocyte endfeet
            "ALDOC",    # WHY: Aldolase C, astrocyte-specific glycolytic enzyme
            "GJB6",     # WHY: Connexin 30, gap junction protein in astrocytes
            "SLC1A3",   # WHY: GLAST/EAAT1, glutamate transporter
            "GLUL",     # WHY: Glutamine synthetase, glutamate metabolism
            "APOE",     # WHY: Apolipoprotein E, lipid transport in glia
            "S100B",    # WHY: S100 calcium-binding protein B, astrocyte marker
        ],
        "category": "glia",
        "hypothesis": "H5D",
        "description": "Astrocytes (includes multiple subtypes)"
    },

    "Excitatory_Neurons_L23_L4": {
        # Source: Tasic et al., 2016, Nature Neuroscience; Harris et al., 2018, Nature
        # Upper layer excitatory neurons
        "markers": [
            "CUX2",     # WHY: Cux2, upper layer (L2/3/4) excitatory neuron marker (Tasic 2016)
            "RORB",     # WHY: ROR-beta, layer 4 excitatory neuron marker
            "ETV1",     # WHY: Etv1/Er81, layer 5/6 marker shared with some L4 neurons
            "LMO3",     # WHY: LMO3, upper layer excitatory neurons
            "HTR2A",    # WHY: Serotonin receptor 2A, enriched in L4 excitatory
        ],
        "category": "excitatory_neuron",
        "hypothesis": "H5C",
        "description": "Excitatory neurons in layers 2/3 and 4"
    },

    "Broad_Neuronal": {
        # Source: Universal pan-neuronal markers (multiple scRNA-seq studies)
        # Synaptic and neuronal identity markers
        "markers": [
            "SNAP25",   # WHY: SNAP25, presynaptic terminal protein
            "SYN1",     # WHY: Synapsin I, synaptic vesicle protein
            "SYP",      # WHY: Synaptophysin, synaptic vesicle protein
            "GRIN1",    # WHY: NMDA receptor subunit 1, glutamatergic synapses
            "GRIN2A",   # WHY: NMDA receptor subunit 2A
            "GRIN2B",   # WHY: NMDA receptor subunit 2B
        ],
        "category": "pan_neuronal",
        "hypothesis": "baseline",
        "description": "Pan-neuronal markers (baseline for comparison)"
    },

    "Dopaminergic_Neurons": {
        # Source: Accepted marker genes for dopaminergic neurons (various studies)
        "markers": [
            "TH",       # WHY: Tyrosine hydroxylase, rate-limiting enzyme in dopamine synthesis
            "SLC6A3",   # WHY: DAT, dopamine transporter
            "DBH",      # WHY: Dopamine beta-hydroxylase (noradrenergic marker)
            "EN1",      # WHY: Engrailed 1, midbrain patterning
            "NR4A2",    # WHY: Nurr1, dopaminergic transcription factor
            "CALB1",    # WHY: Calbindin, some dopaminergic populations
        ],
        "category": "neuron",
        "hypothesis": "SPECULATIVE",
        "description": "Dopaminergic neurons (midbrain)"
    },

    "VLMC": {
        # Source: Vascular leptomeningeal cells (Cappelli et al., preprint)
        "markers": [
            "COLEC11",  # WHY: Collectin 11, VLMC marker
            "COLEC12",  # WHY: Collectin 12, VLMC marker
            "CTSD",     # WHY: Cathepsin D, lysosomal enzyme in VLMC
            "LGALS1",   # WHY: Galectin 1, secreted by VLMC
            "S100A6",   # WHY: S100 calcium binding protein A6
        ],
        "category": "vascular",
        "hypothesis": "SPECULATIVE",
        "description": "Vascular leptomeningeal cells"
    },
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EnrichmentResult:
    """Results from hypergeometric enrichment test for one cell type."""
    cell_type: str
    marker_count: int
    markers_in_gwas: int
    gwas_total: int
    odds_ratio: float
    p_value: float
    fdr: float
    category: str
    hypothesis: str
    is_significant: bool
    pattern: str  # POSITIVE, NEGATIVE, UNINTERPRETABLE

@dataclass
class PipelineOutput:
    """Complete output from the enrichment pipeline."""
    version: str
    timestamp: str
    gwas_source: str
    gwas_file: str
    gwas_p_threshold: float
    total_genes_tested: int
    genes_passing_threshold: int
    cell_types_tested: int
    results: List[Dict]
    interpretation: Dict
    pre_registered_patterns: Dict

# ============================================================================
# DATA DOWNLOAD FUNCTIONS
# ============================================================================

def download_pgc3_if_needed(url: str, output_path: Path, force_redownload: bool = False) -> Path:
    """
    Download PGC3 SCZ GWAS summary statistics if not already present.

    WHY: We need the actual GWAS summary statistics to perform enrichment.
    The file should be ~200-300 MB compressed. We verify the download by checking file size.
    """
    if output_path.exists() and not force_redownload:
        file_size = output_path.stat().st_size
        print(f"  Found existing PGC3 file: {output_path} ({file_size:,} bytes)")
        # Verify it's not empty (placeholder files are 0 bytes)
        if file_size == 0:
            print("  WARNING: File is 0 bytes, re-downloading...")
            output_path.unlink()
        else:
            return output_path

    print(f"  Downloading PGC3 SCZ GWAS from {url}...")
    print(f"  This may take several minutes (expected size: ~200-300 MB)")

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download in chunks
        total_size = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
                    if total_size % (10 * 1024 * 1024) == 0:  # Progress every 10 MB
                        print(f"    Downloaded: {total_size:,} bytes...")

        print(f"  Download complete: {total_size:,} bytes")
        return output_path

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Failed to download PGC3: {e}")
        raise

# ============================================================================
# GWAS DATA PARSING
# ============================================================================

def parse_gwas_sumstats(gz_path: Path, skip_snps: bool = True) -> Tuple[Counter, int]:
    """
    Parse GWAS summary statistics file.

    File format (standard GWAS catalog):
    - Columns: SNP, CHR, BP, A1, A2, FRQ, INFO, OR, SE, P, N, ... (varies by file)
    - Often has header row with # or SNP ID column

    Returns:
        genes_with_pvalue: Counter of genes passing threshold
        total_genes: Total number of genes in file

    WHY we use Counter: Some genes may appear multiple times (multiple SNPs)
    We count unique genes that pass the threshold.
    """
    genes_with_pvalue = Counter()
    total_genes = 0
    gwas_genes_set = set()

    print(f"  Parsing GWAS file: {gz_path}")

    try:
        with gzip.open(gz_path, 'rt') as f:
            header = None
            line_num = 0

            for line in f:
                line_num += 1

                # Skip comment lines
                if line.startswith('#'):
                    continue

                # Parse header
                if header is None:
                    parts = line.strip().split()
                    header = {name: idx for idx, name in enumerate(parts)}

                    # Find gene and P columns
                    # Common column names: GENE, gene, Gene, gene_name, GeneName
                    gene_col = None
                    p_col = None

                    for col_name in ['GENE', 'gene', 'Gene', 'gene_name', 'GeneName', 'nearestGene']:
                        if col_name in header:
                            gene_col = header[col_name]
                            break

                    # Find P-value column
                    for col_name in ['P', 'p', 'PVALUE', 'P-value', 'pvalue']:
                        if col_name in header:
                            p_col = header[col_name]
                            break

                    if gene_col is None or p_col is None:
                        print(f"  WARNING: Could not identify GENE or P column")
                        print(f"  Available columns: {list(header.keys())}")
                        # Fall back to SNP column as gene identifier
                        if 'SNP' in header:
                            gene_col = header['SNP']
                        if p_col is None:
                            print("  ERROR: Cannot proceed without P-value column")
                            raise ValueError("Missing required columns")

                    print(f"  Gene column: {list(header.keys())[gene_col]}")
                    print(f"  P-value column: {list(header.keys())[p_col]}")
                    continue

                parts = line.strip().split()

                try:
                    gene = parts[gene_col]
                    p_value = float(parts[p_col])
                except (IndexError, ValueError):
                    continue

                total_genes += 1

                if gene and gene != 'NA' and gene != '.' and len(gene) > 1:
                    gwas_genes_set.add(gene.upper())

                    if p_value < GWAS_P_THRESHOLD:
                        genes_with_pvalue[gene.upper()] = 1

                if line_num % 100000 == 0:
                    print(f"    Processed {line_num:,} rows...")

        print(f"  Total rows: {line_num:,}")
        print(f"  Unique genes in GWAS: {len(gwas_genes_set):,}")
        print(f"  Genes with p < {GWAS_P_THRESHOLD}: {len(genes_with_pvalue):,}")

        return genes_with_pvalue, len(gwas_genes_set)

    except Exception as e:
        print(f"  ERROR parsing GWAS file: {e}")
        raise

# ============================================================================
# HYPERGEOMETRIC ENRICHMENT TEST
# ============================================================================

def hypergeometric_enrichment_test(
    marker_set: List[str],
    gwas_significant_genes: Counter,
    total_genes: int
) -> Tuple[float, float]:
    """
    Perform hypergeometric test for gene set enrichment.

    Model: Fisher's exact test / hypergeometric distribution
    - Population: all genes in GWAS
    - Successes: genes passing GWAS p < threshold
    - Sample: genes in cell type marker set
    - Question: Are marker genes over-represented in GWAS hits?

    Returns:
        odds_ratio: How much more likely marker genes are GWAS hits
        p_value: Statistical significance

    WHY Fisher's exact test: Standard for gene set enrichment (GO, KEGG, etc.)
    It's appropriate when we have binary outcomes (in GWAS hit / not in GWAS hit)
    """
    # Convert marker list to uppercase set
    marker_set = {g.upper() for g in marker_set}

    # Count how many markers are in GWAS significant genes
    markers_in_gwas = sum(1 for g in marker_set if g in gwas_significant_genes)

    # Count how many markers are NOT in GWAS significant
    markers_not_in_gwas = len(marker_set) - markers_in_gwas

    # Count non-marker genes that are in GWAS significant
    nonmarkers_in_gwas = len(gwas_significant_genes) - markers_in_gwas

    # Count non-marker genes that are NOT in GWAS significant
    nonmarkers_not_in_gwas = total_genes - len(gwas_significant_genes) - markers_not_in_gwas

    # 2x2 contingency table for Fisher's exact test
    #                 | In GWAS hit | Not in GWAS hit |
    # -----------------------------------------------|
    # In marker set   |   a (hits)  |   b (misses)    |
    # Not in marker  |   c         |   d              |

    contingency_table = [
        [markers_in_gwas, markers_not_in_gwas],
        [nonmarkers_in_gwas, nonmarkers_not_in_gwas]
    ]

    # Fisher's exact test
    odds_ratio, p_value = stats.fisher_exact(contingency_table, alternative='greater')

    return odds_ratio, p_value

# ============================================================================
# FDR CORRECTION
# ============================================================================

def fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """
    Benjamini-Hochberg FDR correction.

    WHY FDR: We test multiple cell types simultaneously.
    Without correction, the chance of false positives increases.
    FDR controls the expected proportion of false discoveries.
    """
    n = len(p_values)
    p_values = np.array(p_values)

    # Sort p-values and keep track of original order
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH procedure: find largest k where p[k] <= (k+1)/n * alpha
    fdr_adjusted = np.zeros(n)

    # Calculate adjusted p-values
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            fdr_adjusted[sorted_indices[i]] = sorted_p[i]
        else:
            fdr_adjusted[sorted_indices[i]] = min(
                sorted_p[i] * n / (i + 1),
                fdr_adjusted[sorted_indices[i + 1]]
            )

    return fdr_adjusted.tolist()

# ============================================================================
# PATTERN CLASSIFICATION
# ============================================================================

def classify_pattern(
    cell_type: str,
    odds_ratio: float,
    p_value: float,
    marker_count: int
) -> str:
    """
    Classify result against pre-registered patterns.

    WHY pre-registered patterns: Prevents post-hoc interpretation bias.
    We define what we would consider POSITIVE, NEGATIVE, or UNINTERPRETABLE
    BEFORE seeing the results.

    POSITIVE patterns (specific thresholds):
    - Microglia inflammatory: OR > 1.5, p < 0.05
    - PV+ interneurons: OR > 1.3, p < 0.05

    NEGATIVE patterns:
    - OR ~ 1.0 (no enrichment), regardless of p-value

    UNINTERPRETABLE patterns:
    - Fewer than 5 markers per cell type
    """
    if marker_count < 5:
        return "UNINTERPRETABLE"

    if p_value >= 0.05:
        return "NEGATIVE"

    # Cell-type-specific positive thresholds
    if cell_type == "Microglia_Inflammatory":
        if odds_ratio > 1.5:
            return "POSITIVE"
    elif cell_type in ["PV_Interneurons", "L5_ET_Pyramidal"]:
        if odds_ratio > 1.3:
            return "POSITIVE"
    else:
        # Generic positive threshold
        if odds_ratio > 1.2 and p_value < 0.05:
            return "POSITIVE"

    # If we get here, it's not positive
    if abs(odds_ratio - 1.0) < 0.15:  # OR close to 1.0
        return "NEGATIVE"

    return "INCONCLUSIVE"

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_enrichment_pipeline(
    pgc3_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    force_download: bool = False
) -> PipelineOutput:
    """
    Main pipeline: Download GWAS, parse, test enrichment for each cell type.

    Steps:
    1. Download PGC3 GWAS (if needed)
    2. Parse to get genes passing p < 0.05
    3. For each cell type: hypergeometric test
    4. FDR correction across cell types
    5. Classify patterns and generate interpretation
    """
    from datetime import datetime

    print("=" * 70)
    print("MARKER-BASED GWAS ENRICHMENT PIPELINE FOR SCHIZOPHRENIA")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Step 1: Get PGC3 data
    print("[1/5] Obtaining PGC3 SCZ GWAS summary statistics...")
    if pgc3_path is None:
        pgc3_path = LOCAL_PGC3_PATH

    pgc3_path = download_pgc3_if_needed(PGC3_FIGSHARE_URL, pgc3_path, force_download)

    # Step 2: Parse GWAS file
    print("\n[2/5] Parsing GWAS summary statistics...")
    gwas_significant_genes, total_genes = parse_gwas_sumstats(pgc3_path)
    print(f"  Total genes in GWAS: {total_genes:,}")
    print(f"  Genes with p < {GWAS_P_THRESHOLD}: {len(gwas_significant_genes):,}")

    # Step 3: Run enrichment for each cell type
    print("\n[3/5] Running hypergeometric enrichment tests...")
    results = []

    for cell_type_name, cell_type_data in CELL_TYPE_MARKERS.items():
        markers = cell_type_data["markers"]
        category = cell_type_data["category"]
        hypothesis = cell_type_data["hypothesis"]
        description = cell_type_data["description"]

        print(f"  Testing {cell_type_name} ({len(markers)} markers)...")

        # Run hypergeometric test
        odds_ratio, p_value = hypergeometric_enrichment_test(
            markers,
            gwas_significant_genes,
            total_genes
        )

        # Count how many markers are in GWAS significant
        markers_upper = {m.upper() for m in markers}
        markers_in_gwas = sum(1 for m in markers_upper if m in gwas_significant_genes)

        results.append({
            "cell_type": cell_type_name,
            "marker_count": len(markers),
            "markers_in_gwas": markers_in_gwas,
            "category": category,
            "hypothesis": hypothesis,
            "description": description,
            "markers": markers,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
        })

    # Step 4: FDR correction
    print("\n[4/5] Applying FDR correction...")
    p_values = [r["p_value"] for r in results]
    fdr_adjusted = fdr_correction(p_values)

    for i, r in enumerate(results):
        r["fdr"] = fdr_adjusted[i]

    # Step 5: Classify patterns and generate interpretation
    print("\n[5/5] Generating interpretation...")

    for r in results:
        r["pattern"] = classify_pattern(
            r["cell_type"],
            r["odds_ratio"],
            r["p_value"],
            r["marker_count"]
        )
        r["is_significant"] = r["fdr"] < 0.05

    # Sort by odds ratio
    results_sorted = sorted(results, key=lambda x: x["odds_ratio"], reverse=True)

    # Identify top enriched cell types
    top_enriched = [
        r for r in results_sorted
        if r["pattern"] == "POSITIVE"
    ]

    # Check specific hypotheses
    microglia_result = next((r for r in results if r["cell_type"] == "Microglia_Inflammatory"), None)
    pv_result = next((r for r in results if r["cell_type"] == "PV_Interneurons"), None)
    l5et_result = next((r for r in results if r["cell_type"] == "L5_ET_Pyramidal"), None)

    interpretation = {
        "h5b_microglia_inflam": {
            "supported": microglia_result["pattern"] == "POSITIVE" if microglia_result else False,
            "odds_ratio": microglia_result["odds_ratio"] if microglia_result else None,
            "p_value": microglia_result["p_value"] if microglia_result else None,
            "fdr": microglia_result["fdr"] if microglia_result else None,
        },
        "h5c_pv_l5et": {
            "supported": (
                (pv_result["pattern"] == "POSITIVE" if pv_result else False) or
                (l5et_result["pattern"] == "POSITIVE" if l5et_result else False)
            ),
            "pv_odds_ratio": pv_result["odds_ratio"] if pv_result else None,
            "pv_p_value": pv_result["p_value"] if pv_result else None,
            "l5et_odds_ratio": l5et_result["odds_ratio"] if l5et_result else None,
            "l5et_p_value": l5et_result["p_value"] if l5et_result else None,
        }
    }

    # Build output
    output = PipelineOutput(
        version="1.0",
        timestamp=datetime.now().isoformat(),
        gwas_source=PGC3_FIGSHARE_URL,
        gwas_file=str(pgc3_path),
        gwas_p_threshold=GWAS_P_THRESHOLD,
        total_genes_tested=total_genes,
        genes_passing_threshold=len(gwas_significant_genes),
        cell_types_tested=len(CELL_TYPE_MARKERS),
        results=results_sorted,
        interpretation=interpretation,
        pre_registered_patterns={
            "POSITIVE": {
                "Microglia_Inflammatory": "OR > 1.5, p < 0.05",
                "PV_Interneurons": "OR > 1.3, p < 0.05",
                "L5_ET_Pyramidal": "OR > 1.3, p < 0.05",
            },
            "NEGATIVE": "OR ~ 1.0 (no enrichment)",
            "UNINTERPRETABLE": "Fewer than 5 markers"
        }
    )

    # Save output
    if output_path is None:
        output_path = OUTPUT_DIR / "enrichment_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict for JSON serialization
    output_dict = {
        "version": output.version,
        "timestamp": output.timestamp,
        "gwas_source": output.gwas_source,
        "gwas_file": output.gwas_file,
        "gwas_p_threshold": output.gwas_p_threshold,
        "total_genes_tested": output.total_genes_tested,
        "genes_passing_threshold": output.genes_passing_threshold,
        "cell_types_tested": output.cell_types_tested,
        "results": output.results,
        "interpretation": output.interpretation,
        "pre_registered_patterns": output.pre_registered_patterns,
    }

    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=2)

    print(f"\n  Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("ENRICHMENT SUMMARY")
    print("=" * 70)
    print(f"Total genes in GWAS: {total_genes:,}")
    print(f"Genes with p < {GWAS_P_THRESHOLD}: {len(gwas_significant_genes):,}")
    print(f"Cell types tested: {len(CELL_TYPE_MARKERS)}")
    print()

    print("Top enriched cell types (by OR):")
    print("-" * 70)
    print(f"{'Cell Type':<30} {'OR':>8} {'p-value':>12} {'FDR':>12} {'Pattern':<15}")
    print("-" * 70)

    for r in results_sorted[:5]:
        status = "***" if r["pattern"] == "POSITIVE" else ""
        print(f"{r['cell_type']:<30} {r['odds_ratio']:>8.3f} {r['p_value']:>12.2e} {r['fdr']:>12.2e} {r['pattern']:<15} {status}")

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if interpretation["h5b_microglia_inflam"]["supported"]:
        print("H5B SUPPORTED: Microglia inflammatory markers are enriched in SCZ GWAS")
    else:
        print("H5B NOT SUPPORTED: No significant enrichment of microglia inflammatory markers")

    if interpretation["h5c_pv_l5et"]["supported"]:
        print("H5C SUPPORTED: PV+ and/or L5_ET markers are enriched in SCZ GWAS")
    else:
        print("H5C NOT SUPPORTED: No significant enrichment of PV+ or L5_ET markers")

    print("\n" + "=" * 70)

    return output

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Marker-based GWAS enrichment pipeline for schizophrenia research"
    )
    parser.add_argument(
        "--pgc3-path",
        type=Path,
        default=None,
        help=f"Path to PGC3 GWAS file (default: {LOCAL_PGC3_PATH})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output JSON path (default: {OUTPUT_DIR}/enrichment_results.json)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of PGC3 file even if present"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})"
    )

    args = parser.parse_args()

    # Update seed if provided
    if args.seed != RANDOM_SEED:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Run pipeline
    output = run_enrichment_pipeline(
        pgc3_path=args.pgc3_path,
        output_path=args.output,
        force_download=args.force_download
    )

    print("\nPipeline completed successfully!")
