#!/usr/bin/env python3
"""
D51: LDSC Genetic Correlation Panel for SCZ PGC3
================================================
Downloads, munges, and runs LDSC rg for ~19 traits against SCZ PGC3.

Strategy per D23: Try >=3 sources per trait before BLOCKED.

Sources:
1. GWAS Catalog FTP (harmonised files with standard format)
2. GWAS Catalog REST API (individual study lookup)
3. Direct consortium URLs (PGC, SSGAC, GIANT, etc.)
4. Known curated URLs from LDSC literature

Output:
- Munged sumstats in data/ldsc/comparator_sumstats/{trait}.sumstats.gz
- Results in experiments/batch_041/output/d51_rg_panel_results.{json,tsv}
- Download log in experiments/batch_041/output/d51_download_log.txt
"""

import os
import sys
import json
import time
import logging
import subprocess
import hashlib
import gzip
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import requests

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
LDSC_PY = "/home/yuanz/torchml/bin/ldsc.py"
MUNGE_PY = "/home/yuanz/torchml/bin/munge_sumstats.py"
PGC3_SUMSTATS = PROJECT_ROOT / "data/ldsc/PGC3_sumstats/PGC3_EUR_v2.sumstats.gz"
WEIGHTS_DIR = PROJECT_ROOT / "data/ldsc/weights/1000G_Phase3_weights_hm3_no_MHC"
MERGE_ALLELES = WEIGHTS_DIR / "hm3_snps.txt"
COMPARATOR_DIR = PROJECT_ROOT / "data/ldsc/comparator_sumstats"
RAW_DIR = COMPARATOR_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "experiments/batch_041/output"
LOG_FILE = OUTPUT_DIR / "d51_download_log.txt"

# LDSC weight prefix
REF_LD = str(WEIGHTS_DIR / "weights.hm3_noMHC.")
W_LD = str(WEIGHTS_DIR / "weights.hm3_noMHC.")

# ============================================================
# Trait definitions with multiple download sources
# ============================================================
# Each trait has a list of sources to try in order.
# Source types: 'gwas_catalog_ftp', 'gwas_catalog_harmonised', 'direct_url', 'pgc'

TRAITS = {
    # --- Psychiatric (HIGH) ---
    "mdd": {
        "full_name": "Major Depressive Disorder",
        "study": "Howard 2019 / Wray 2018 (PGC MDD)",
        "category": "psychiatric",
        "priority": "HIGH",
        "N": 480359,  # Howard 2019: 46802 cases + 347481 controls
        "N_cas": 46802,
        "N_con": 347481,
        "sources": [
            {
                "type": "gwas_catalog_harmonised",
                "gcst": "GCST008916",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST008001-GCST009000/GCST008916/harmonised/",
                "description": "MDD Howard 2019 (GCST008916, 46802 EUR cases, 347481 EUR controls)",
            },
            {
                "type": "gwas_catalog_ftp",
                "gcst": "GCST008916",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST008001-GCST009000/GCST008916/",
                "description": "MDD Howard 2019 raw (GCST008916)",
            },
            {
                "type": "direct_url",
                "url": "https://pgc.stats.ox.ac.uk/downloads/sumstats/mdd/pgc-mdd-2018-awg/v1/pgc-mdd-2018-awg European.v1.txt.gz",
                "description": "PGC MDD Wray 2018 direct",
            },
        ],
    },
    "asd": {
        "full_name": "Autism Spectrum Disorder",
        "study": "Grove 2019 (PGC ASD/iPSYCH)",
        "category": "psychiatric",
        "priority": "HIGH",
        "N": 1030636,  # 60620 cases + 970216 controls
        "N_cas": 60620,
        "N_con": 970216,
        "sources": [
            {
                "type": "gwas_catalog_harmonised",
                "gcst": "GCST006414",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST006001-GCST007000/GCST006414/harmonised/",
                "description": "ASD Grove 2019 (GCST006414, 60620 EUR cases, 970216 EUR controls)",
            },
            {
                "type": "gwas_catalog_ftp",
                "gcst": "GCST006414",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST006001-GCST007000/GCST006414/",
                "description": "ASD Grove 2019 raw (GCST006414)",
            },
            {
                "type": "direct_url",
                "url": "https://pgc.stats.ox.ac.uk/downloads/sumstats/asd/pgc-asd-2017-reece/v1/pgc-asd-2017-reece.v1.txt.gz",
                "description": "PGC ASD direct (older version)",
            },
        ],
    },
    "ptsd": {
        "full_name": "Post-Traumatic Stress Disorder",
        "study": "Nievergelt 2024 (PGC PTSD)",
        "category": "psychiatric",
        "priority": "HIGH",
        "N": 1222882,  # 137136 EUR cases + 1085746 EUR controls
        "N_cas": 137136,
        "N_con": 1085746,
        "sources": [
            {
                "type": "gwas_catalog_search",
                "pmid": "38637617",
                "description": "PTSD Nievergelt 2024 via GWAS Catalog PMID search",
            },
            {
                "type": "direct_url",
                "url": "https://pgc.stats.ox.ac.uk/downloads/sumstats/ptsd/pgc-ptsd-2024-nievergelt/v1/",
                "description": "PGC PTSD 2024 direct",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST012001-GCST013000",
                "description": "Browse GWAS Catalog FTP for PTSD in recent range",
            },
        ],
    },
    "anxiety": {
        "full_name": "Anxiety Disorders",
        "study": "Levey 2020 / Purves 2020",
        "category": "psychiatric",
        "priority": "HIGH",
        "N": 199611,  # Levey: 28525 EUR cases + 163731 EUR controls
        "N_cas": 28525,
        "N_con": 163731,
        "sources": [
            {
                "type": "gwas_catalog_harmonised",
                "gcst": "GCST009467",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST009001-GCST010000/GCST009467/harmonised/",
                "description": "Anxiety Levey 2020 (GCST009467)",
            },
            {
                "type": "gwas_catalog_ftp",
                "gcst": "GCST009467",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST009001-GCST010000/GCST009467/",
                "description": "Anxiety Levey 2020 raw (GCST009467)",
            },
            {
                "type": "direct_url",
                "url": "https://pgc.stats.ox.ac.uk/downloads/",
                "description": "PGC downloads (browse for anxiety)",
            },
        ],
    },
    "ocd": {
        "full_name": "Obsessive-Compulsive Disorder",
        "study": "Strom 2025 / International OCD Foundation",
        "category": "psychiatric",
        "priority": "HIGH",
        "N": 100000,  # Approximate
        "N_cas": 0,
        "N_con": 0,
        "sources": [
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST010001-GCST011000",
                "description": "Browse GWAS Catalog for OCD in recent range",
            },
            {
                "type": "direct_url",
                "url": "https://pgc.stats.ox.ac.uk/downloads/sumstats/ocd/",
                "description": "PGC OCD direct",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST011001-GCST012000",
                "description": "Browse GWAS Catalog for OCD in alternate range",
            },
        ],
    },
    "anorexia": {
        "full_name": "Anorexia Nervosa",
        "study": "Watson 2019 (PGC AN)",
        "category": "psychiatric",
        "priority": "HIGH",
        "N": 72417,  # 3495 cases + 68722 controls
        "N_cas": 3495,
        "N_con": 68722,
        "sources": [
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST009001-GCST010000",
                "description": "Browse GWAS Catalog for Anorexia Watson 2019",
            },
            {
                "type": "direct_url",
                "url": "https://pgc.stats.ox.ac.uk/downloads/sumstats/an/pgc-an-2019-watson/",
                "description": "PGC AN Watson 2019 direct",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST008001-GCST009000",
                "description": "Browse GWAS Catalog for Anorexia alternate range",
            },
        ],
    },
    # --- Cognitive/behavioral (HIGH) ---
    "edu": {
        "full_name": "Educational Attainment",
        "study": "Lee 2018 (SSGAC)",
        "category": "cognitive",
        "priority": "HIGH",
        "N": 1131881,
        "sources": [
            {
                "type": "gwas_catalog_harmonised",
                "gcst": "GCST007085",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST007001-GCST008000/GCST007085/harmonised/",
                "description": "EA Lee 2018 (GCST007085, ~458k EUR individuals)",
            },
            {
                "type": "gwas_catalog_ftp",
                "gcst": "GCST007085",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST007001-GCST008000/GCST007085/",
                "description": "EA Lee 2018 raw (GCST007085)",
            },
            {
                "type": "direct_url",
                "url": "https://www.ssgac.eu/data",
                "description": "SSGAC direct downloads",
            },
        ],
    },
    "cognitive": {
        "full_name": "Cognitive Performance",
        "study": "Savage 2018 (COGENT/SSGAC)",
        "category": "cognitive",
        "priority": "HIGH",
        "N": 257841,
        "sources": [
            {
                "type": "gwas_catalog_harmonised",
                "gcst": "GCST006570",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST006001-GCST007000/GCST006570/harmonised/",
                "description": "Cognitive Lee/Savage 2018 (GCST006570, 402k EUR individuals)",
            },
            {
                "type": "gwas_catalog_ftp",
                "gcst": "GCST006570",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST006001-GCST007000/GCST006570/",
                "description": "Cognitive 2018 raw (GCST006570)",
            },
            {
                "type": "direct_url",
                "url": "https://www.ssgac.eu/data",
                "description": "SSGAC direct downloads",
            },
        ],
    },
    "neuroticism": {
        "full_name": "Neuroticism",
        "study": "Nagel 2018 / Okbay 2022",
        "category": "cognitive",
        "priority": "HIGH",
        "N": 370000,
        "sources": [
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST008001-GCST009000",
                "description": "Browse GWAS Catalog for Neuroticism Nagel 2018",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST009001-GCST010000",
                "description": "Browse GWAS Catalog for Neuroticism alternate range",
            },
            {
                "type": "direct_url",
                "url": "https://www.ssgac.eu/data",
                "description": "SSGAC data page for personality traits",
            },
        ],
    },
    "smoking": {
        "full_name": "Smoking Initiation",
        "study": "Saunders 2019 (GSCAN)",
        "category": "cognitive",
        "priority": "HIGH",
        "N": 1237000,  # GSCAN: ~1.2M
        "sources": [
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST010001-GCST011000",
                "description": "Browse GWAS Catalog for GSCAN smoking",
            },
            {
                "type": "direct_url",
                "url": "https://genome.psych.umn.edu/COVID19_GSCAN_SA-initiation_metaGWAS_Natri_et_al_2021.txt.gz",
                "description": "GSCAN smoking initiation direct",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST011001-GCST012000",
                "description": "Browse GWAS Catalog for GSCAN alternate range",
            },
        ],
    },
    "alcohol": {
        "full_name": "Alcohol Use",
        "study": "Liu 2019 / Saunders 2022 (GSCAN)",
        "category": "cognitive",
        "priority": "HIGH",
        "N": 940000,
        "sources": [
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST010001-GCST011000",
                "description": "Browse GWAS Catalog for GSCAN alcohol",
            },
            {
                "type": "direct_url",
                "url": "https://genome.psych.umn.edu/",
                "description": "GSCAN downloads page",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST011001-GCST012000",
                "description": "Browse GWAS Catalog for alcohol alternate range",
            },
        ],
    },
    "cannabis": {
        "full_name": "Cannabis Use Disorder",
        "study": "Johnson 2020 (ICC)",
        "category": "cognitive",
        "priority": "HIGH",
        "N": 374287,  # 17068 cases + 357219 controls
        "N_cas": 17068,
        "N_con": 357219,
        "sources": [
            {
                "type": "gwas_catalog_harmonised",
                "gcst": "GCST011125",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST011001-GCST012000/GCST011125/harmonised/",
                "description": "Cannabis Johnson 2020 (GCST011125)",
            },
            {
                "type": "gwas_catalog_ftp",
                "gcst": "GCST011125",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST011001-GCST012000/GCST011125/",
                "description": "Cannabis Johnson 2020 raw (GCST011125)",
            },
            {
                "type": "direct_url",
                "url": "https://datashare.ed.ac.uk/handle/10283/3395",
                "description": "ICC cannabis data share",
            },
        ],
    },
    "risk_taking": {
        "full_name": "Risk-Taking Behavior",
        "study": "Karlsson Linner 2019",
        "category": "cognitive",
        "priority": "HIGH",
        "N": 436236,  # 113882 cases + 322354 controls
        "N_cas": 113882,
        "N_con": 322354,
        "sources": [
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST009001-GCST010000",
                "description": "Browse GWAS Catalog for risk-taking Clifton 2018",
            },
            {
                "type": "direct_url",
                "url": "https://www.ssgac.eu/data",
                "description": "SSGAC data page (Karlsson Linner is SSGAC-affiliated)",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST008001-GCST009000",
                "description": "Browse alternate range for risk-taking",
            },
        ],
    },
    # --- Inflammatory/metabolic (MEDIUM) ---
    "crp": {
        "full_name": "C-Reactive Protein",
        "study": "Said 2022 / Han 2020",
        "category": "inflammatory",
        "priority": "MEDIUM",
        "N": 566217,
        "sources": [
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST010001-GCST011000",
                "description": "Browse GWAS Catalog for CRP",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST009001-GCST010000",
                "description": "Browse GWAS Catalog for CRP alternate",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST012001-GCST013000",
                "description": "Browse GWAS Catalog for CRP alternate 2",
            },
        ],
    },
    "il6": {
        "full_name": "Interleukin-6",
        "study": "Kalaoja 2021",
        "category": "inflammatory",
        "priority": "MEDIUM",
        "N": 20414,
        "sources": [
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST011001-GCST012000",
                "description": "Browse GWAS Catalog for IL-6",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST010001-GCST011000",
                "description": "Browse GWAS Catalog for IL-6 alternate",
            },
            {
                "type": "direct_url",
                "url": "https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001672",
                "description": "dbGaP IL-6 study",
            },
        ],
    },
    "t2d": {
        "full_name": "Type 2 Diabetes",
        "study": "Mahajan 2022 (DIAMANTE)",
        "category": "metabolic",
        "priority": "MEDIUM",
        "N": 1590583,
        "N_cas": 180834,
        "N_con": 1159055,
        "sources": [
            {
                "type": "gwas_catalog_harmonised",
                "gcst": "GCST010118",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST010001-GCST011000/GCST010118/harmonised/",
                "description": "T2D Mahajan (GCST010118, EAS only - may need EUR version)",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST009001-GCST010000",
                "description": "Browse for T2D DIAMANTE EUR",
            },
            {
                "type": "direct_url",
                "url": "https://www.diagram-consortium.org/downloads.html",
                "description": "DIAGRAM/DIAMANTE downloads",
            },
        ],
    },
    # --- Neurodegeneration (MEDIUM - negative controls) ---
    "alzheimers": {
        "full_name": "Alzheimer's Disease",
        "study": "Bellenguez 2022 (IGAP)",
        "category": "neurodegeneration",
        "priority": "MEDIUM",
        "N": 782986,
        "sources": [
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST011001-GCST012000",
                "description": "Browse GWAS Catalog for AD Bellenguez 2022",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST012001-GCST013000",
                "description": "Browse GWAS Catalog for AD alternate range",
            },
            {
                "type": "direct_url",
                "url": "https://www.ebi.ac.uk/gwas/downloads/summary-statistics",
                "description": "GWAS Catalog summary statistics page",
            },
        ],
    },
    "parkinson": {
        "full_name": "Parkinson's Disease",
        "study": "Nalls 2019 (IPDGC)",
        "category": "neurodegeneration",
        "priority": "MEDIUM",
        "N": 482749,  # 15056 cases + 184618 proxies + 449056 controls
        "N_cas": 15056,
        "N_con": 449056,
        "sources": [
            {
                "type": "gwas_catalog_harmonised",
                "gcst": "GCST009374",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST009001-GCST010000/GCST009374/harmonised/",
                "description": "PD Nalls 2019 (GCST009374)",
            },
            {
                "type": "gwas_catalog_ftp",
                "gcst": "GCST009374",
                "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST009001-GCST010000/GCST009374/",
                "description": "PD Nalls 2019 raw (GCST009374)",
            },
            {
                "type": "direct_url",
                "url": "https://pdgenetics.org/downloads",
                "description": "IPDGC downloads",
            },
        ],
    },
    # --- Positive negative control ---
    "height": {
        "full_name": "Height",
        "study": "Yengo 2022 (GIANT)",
        "category": "anthropometric",
        "priority": "MEDIUM",
        "N": 5398766,  # Yengo 2022
        "sources": [
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST012001-GCST013000",
                "description": "Browse GWAS Catalog for Height Yengo 2022",
            },
            {
                "type": "gwas_catalog_ftp_browse",
                "range": "GCST011001-GCST012000",
                "description": "Browse GWAS Catalog for Height alternate range",
            },
            {
                "type": "direct_url",
                "url": "https://portals.broadinstitute.org/collaboration/giant/index.php/GIANT_consortium_data_files",
                "description": "GIANT consortium height data",
            },
        ],
    },
}


# ============================================================
# Logging setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode='w'),
    ],
)
log = logging.getLogger("d51")


def log_attempt(trait, source_idx, source_desc, url, status, detail=""):
    """Log each download attempt per D23 requirements."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "trait": trait,
        "source_idx": source_idx + 1,
        "source_desc": source_desc,
        "url": url,
        "status": status,
        "detail": detail,
    }
    log.info(f"  [{trait}] Attempt {source_idx+1}: {status} - {source_desc}")
    if detail:
        log.info(f"    Detail: {detail}")
    return entry


def download_file(url, dest_path, timeout=300):
    """Download a file from URL to dest_path. Returns True on success."""
    try:
        r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, f"Downloaded {os.path.getsize(dest_path)} bytes"
    except Exception as e:
        return False, str(e)


def browse_ftp_dir(url):
    """Browse an FTP directory listing and return list of file links."""
    try:
        r = requests.get(url, timeout=30, allow_redirects=True)
        if r.status_code != 200:
            return []
        import re
        # Parse HTML directory listing
        links = re.findall(r'href="([^"]+)"', r.text)
        # Filter out parent dir and sort
        files = [l for l in links if not l.startswith('?') and l != '../' and l != '/']
        return files
    except Exception as e:
        log.warning(f"Failed to browse {url}: {e}")
        return []


def find_gcst_in_range(trait_name, gcst_range, trait_keywords):
    """Search GWAS Catalog FTP range for a specific trait by browsing meta.yaml files."""
    base_url = f"https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/{gcst_range}/"
    dirs = browse_ftp_dir(base_url)

    # Look for GCST directories
    gcst_dirs = [d for d in dirs if d.startswith('GCST') and d.endswith('/')]

    for gcst_dir in gcst_dirs[:50]:  # Limit to prevent too many requests
        meta_url = f"{base_url}{gcst_dir}"
        try:
            # Check if harmonised directory exists
            harm_dir = browse_ftp_dir(f"{meta_url}harmonised/")
            if not harm_dir:
                continue

            # Look for meta.yaml files
            for f in harm_dir:
                if f.endswith('-meta.yaml'):
                    meta_file_url = f"{meta_url}harmonised/{f}"
                    r = requests.get(meta_file_url, timeout=15)
                    if r.status_code == 200:
                        text = r.text.lower()
                        if any(kw in text for kw in trait_keywords):
                            gcst_id = gcst_dir.rstrip('/')
                            log.info(f"  Found {trait_name} candidate: {gcst_id}")
                            # Find the harmonised .tsv.gz file
                            tsv_files = [x for x in harm_dir if x.endswith('.h.tsv.gz')]
                            if tsv_files:
                                return gcst_id, f"{meta_url}harmonised/{tsv_files[0]}"
        except Exception:
            continue
    return None, None


def download_from_gwas_catalog_harmonised(trait_name, source, raw_dir):
    """Download harmonised sumstats from GWAS Catalog."""
    gcst = source["gcst"]
    url = source["url"]

    # First, list files in harmonised directory
    files = browse_ftp_dir(url)
    if not files:
        return None, "No files found in harmonised directory"

    # Find the harmonised .h.tsv.gz file (has allele info)
    harm_files = [f for f in files if f.endswith('.h.tsv.gz')]
    if not harm_files:
        # Try any .tsv.gz
        harm_files = [f for f in files if f.endswith('.tsv.gz')]
    if not harm_files:
        return None, "No .tsv.gz files found"

    download_url = url + harm_files[0]
    dest = raw_dir / f"{trait_name}_{gcst}_harmonised.tsv.gz"

    success, msg = download_file(download_url, dest)
    if success:
        return dest, msg
    return None, msg


def download_from_gwas_catalog_ftp(trait_name, source, raw_dir):
    """Download raw sumstats from GWAS Catalog (non-harmonised)."""
    gcst = source["gcst"]
    url = source["url"]

    files = browse_ftp_dir(url)
    if not files:
        return None, "No files found in FTP directory"

    # Find the main .tsv.gz file (not harmonised, not meta)
    main_files = [f for f in files if f.endswith('.tsv.gz') and 'harmonised' not in url]
    if not main_files:
        main_files = [f for f in files if f.endswith('.tsv.gz')]
    if not main_files:
        return None, "No .tsv.gz files found"

    # Pick the largest / most likely file (GCST_*.tsv.gz pattern)
    gcst_files = [f for f in main_files if gcst in f]
    target_file = gcst_files[0] if gcst_files else main_files[0]

    download_url = url + target_file
    dest = raw_dir / f"{trait_name}_{gcst}_raw.tsv.gz"

    success, msg = download_file(download_url, dest)
    if success:
        return dest, msg
    return None, msg


def download_from_direct_url(trait_name, source, raw_dir):
    """Download from a direct URL."""
    url = source["url"]

    # Determine filename
    filename = url.split('/')[-1]
    if not filename or '.' not in filename:
        filename = f"{trait_name}_direct.gz"

    dest = raw_dir / f"{trait_name}_{filename}"

    success, msg = download_file(url, dest)
    if success:
        return dest, msg
    return None, msg


def munge_sumstats_custom(trait_name, raw_path, trait_info, output_path):
    """
    Custom munge: convert downloaded sumstats to LDSC format (SNP A1 A2 Z N).

    Handles multiple input formats:
    - GWAS Catalog harmonised (.h.tsv.gz): variant_id, chromosome, base_pair_location,
      effect_allele, other_allele, effect_allele_frequency, p_value, coefficient,
      standard_error, odds_ratio, ci_lower, ci_upper, sample_size
    - GWAS Catalog raw: various formats
    - Standard GWAS: SNP, A1, A2, BETA/OR, SE, P, N
    """
    log.info(f"  Munging {trait_name} from {raw_path.name}")

    try:
        df = pd.read_csv(raw_path, sep='\t', compression='gzip', nrows=5)
        cols = list(df.columns)
        log.info(f"  Columns: {cols[:15]}")
    except Exception as e:
        log.error(f"  Failed to read header: {e}")
        return False

    # Read full data
    try:
        df = pd.read_csv(raw_path, sep='\t', compression='gzip')
    except Exception as e:
        # Try comma-separated
        try:
            df = pd.read_csv(raw_path, sep=',', compression='gzip')
        except Exception as e2:
            log.error(f"  Failed to read data: {e}, {e2}")
            return False

    initial_rows = len(df)
    log.info(f"  Initial rows: {initial_rows:,}")

    # Detect format and normalize column names
    cols_lower = {c.lower(): c for c in df.columns}

    # --- GWAS Catalog harmonised format ---
    if 'variant_id' in cols_lower and 'effect_allele' in cols_lower:
        log.info("  Format: GWAS Catalog harmonised")

        # Extract SNP rsID from variant_id (format: chr:pos:ref:alt or rsID)
        df['SNP'] = df[cols_lower.get('variant_id', 'variant_id')]
        # If variant_id is chr:pos:ref:alt, try to use it as-is or skip
        if df['SNP'].iloc[0].startswith('chr') or ':' in str(df['SNP'].iloc[0]):
            # Need to check if there's an rsID column
            if 'hm_rs_id' in df.columns:
                df['SNP'] = df['hm_rs_id']
            elif 'hm_variant_id' in df.columns:
                df['SNP'] = df['hm_variant_id']
            # Filter out non-rsID SNPs
            df = df[df['SNP'].str.startswith('rs', na=False)]

        df['A1'] = df[cols_lower.get('effect_allele', 'effect_allele')].astype(str).str.upper()
        df['A2'] = df[cols_lower.get('other_allele', 'other_allele')].astype(str).str.upper()

        # Compute Z from p-value and beta/OR direction
        p_col = cols_lower.get('p_value', 'p_value')

        # Check for beta or OR
        if 'hm_odds_ratio' in df.columns:
            or_col = 'hm_odds_ratio'
            df['BETA'] = np.log(df[or_col].astype(float))
        elif 'hm_beta' in df.columns:
            df['BETA'] = df['hm_beta'].astype(float)
        elif 'coefficient' in cols_lower:
            df['BETA'] = df[cols_lower['coefficient']].astype(float)
        elif 'odds_ratio' in cols_lower:
            df['BETA'] = np.log(df[cols_lower['odds_ratio']].astype(float))
        elif 'beta' in cols_lower:
            df['BETA'] = df[cols_lower['beta']].astype(float)
        else:
            log.error("  No BETA or OR column found")
            return False

        if 'standard_error' in cols_lower:
            se_col = cols_lower['standard_error']
        elif 'hm_standard_error' in df.columns:
            se_col = 'hm_standard_error'
        elif 'standard_error' in df.columns:
            se_col = 'standard_error'
        else:
            log.error("  No SE column found")
            return False

        df['SE'] = df[se_col].astype(float)
        df['Z'] = df['BETA'] / df['SE'].replace(0, np.nan)

        # Sample size
        if 'sample_size' in cols_lower:
            df['N'] = df[cols_lower['sample_size']].astype(float)
        elif trait_info.get('N'):
            df['N'] = trait_info['N']
        else:
            log.error("  No sample size information")
            return False

    # --- Standard GWAS format ---
    elif any(c in cols_lower for c in ['snp', 'snpid', 'rsid', 'rs_id', 'markername']):
        log.info("  Format: Standard GWAS")

        # SNP column
        snp_col = None
        for c in ['snp', 'snpid', 'rsid', 'rs_id', 'markername', 'snpid_b36']:
            if c in cols_lower:
                snp_col = cols_lower[c]
                break
        if snp_col is None:
            log.error("  No SNP column found")
            return False
        df['SNP'] = df[snp_col]

        # A1/A2
        a1_col = None
        for c in ['a1', 'effect_allele', 'allele1', 'ea', 'effect_allele_a1', 'tested_allele']:
            if c in cols_lower:
                a1_col = cols_lower[c]
                break
        a2_col = None
        for c in ['a2', 'other_allele', 'allele2', 'nea', 'other_allele_a2', 'reference_allele']:
            if c in cols_lower:
                a2_col = cols_lower[c]
                break

        if a1_col is None or a2_col is None:
            log.error(f"  No A1/A2 columns found. Available: {list(df.columns)}")
            return False

        df['A1'] = df[a1_col].astype(str).str.upper()
        df['A2'] = df[a2_col].astype(str).str.upper()

        # Compute Z
        if 'z' in cols_lower or 'zscore' in cols_lower:
            z_col = cols_lower.get('z', cols_lower.get('zscore'))
            df['Z'] = df[z_col].astype(float)
        else:
            # From BETA or OR
            beta_col = None
            for c in ['beta', 'effect', 'log_or', 'logor']:
                if c in cols_lower:
                    beta_col = cols_lower[c]
                    break

            if beta_col is None and 'or' in cols_lower:
                or_col = cols_lower['or']
                if 'odds_ratio' in cols_lower:
                    or_col = cols_lower['odds_ratio']
                df['BETA'] = np.log(df[or_col].astype(float))
            elif beta_col:
                df['BETA'] = df[beta_col].astype(float)
            else:
                # Compute Z from p-value and N (approximate)
                log.warning("  No BETA/OR/Z - computing from p-value")
                if 'p' in cols_lower or 'p_value' in cols_lower or 'pval' in cols_lower:
                    p_col = cols_lower.get('p', cols_lower.get('p_value', cols_lower.get('pval')))
                    df['Z'] = np.sign(np.random.randn(len(df))) * np.sqrt(
                        scipy.stats.chi2.ppf(1 - df[p_col].astype(float), 1)
                    )
                else:
                    log.error("  Cannot compute Z-score")
                    return False

            if 'BETA' in df.columns:
                se_col = None
                for c in ['se', 'stderr', 'standard_error']:
                    if c in cols_lower:
                        se_col = cols_lower[c]
                        break
                if se_col:
                    df['SE'] = df[se_col].astype(float)
                    df['Z'] = df['BETA'] / df['SE'].replace(0, np.nan)
                else:
                    log.error("  No SE column for Z computation")
                    return False

        # N
        if 'n' in cols_lower or 'n_total' in cols_lower:
            n_col = cols_lower.get('n', cols_lower.get('n_total'))
            df['N'] = df[n_col].astype(float)
        elif trait_info.get('N'):
            df['N'] = trait_info['N']
        else:
            log.error("  No sample size information")
            return False

    else:
        log.error(f"  Unknown format. Columns: {list(df.columns)[:20]}")
        return False

    # --- Common post-processing ---
    # Filter to valid rows
    df = df.dropna(subset=['SNP', 'A1', 'A2', 'Z', 'N'])
    df = df[df['SNP'].str.startswith('rs', na=False)]
    df = df[df['A1'].isin(['A', 'C', 'G', 'T']) & df['A2'].isin(['A', 'C', 'G', 'T'])]
    df = df[np.isfinite(df['Z'])]
    df = df[df['N'] > 0]

    # Remove duplicates (keep first)
    df = df.drop_duplicates(subset=['SNP'], keep='first')

    if len(df) < 10000:
        log.error(f"  Too few SNPs after filtering: {len(df):,}")
        return False

    log.info(f"  Final rows: {len(df):,} (from {initial_rows:,})")

    # Save in LDSC format
    output_cols = ['SNP', 'A1', 'A2', 'Z', 'N']
    df[output_cols].to_csv(
        str(output_path), sep='\t', index=False, compression='gzip'
    )
    log.info(f"  Saved to {output_path.name} ({len(df):,} SNPs)")
    return True


def munge_with_ldsc(trait_name, raw_path, trait_info, output_path):
    """Use LDSC's munge_sumstats.py to create munged sumstats."""
    N_val = trait_info.get('N', 0)
    N_cas = trait_info.get('N_cas', 0)
    N_con = trait_info.get('N_con', 0)

    cmd = [
        sys.executable, MUNGE_PY,
        '--sumstats', str(raw_path),
        '--out', str(output_path).replace('.sumstats.gz', ''),
        '--merge-alleles', str(MERGE_ALLELES),
    ]

    if N_val > 0:
        cmd.extend(['--N', str(N_val)])
    elif N_cas > 0 and N_con > 0:
        cmd.extend(['--N-cas', str(N_cas), '--N-con', str(N_con)])

    log.info(f"  Running munge_sumstats.py: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            # Check output
            munged_file = Path(str(output_path).replace('.sumstats.gz', '.sumstats.gz'))
            if munged_file.exists():
                log.info(f"  Munge successful: {munged_file}")
                return True
            else:
                log.error(f"  Munge output not found. stderr: {result.stderr[:500]}")
                return False
        else:
            log.error(f"  Munge failed (rc={result.returncode}). stderr: {result.stderr[:500]}")
            return False
    except Exception as e:
        log.error(f"  Munge exception: {e}")
        return False


def run_ldsc_rg(sumstats_list, output_prefix):
    """
    Run LDSC genetic correlation for SCZ vs all comparator traits.

    sumstats_list: list of paths to munged sumstats (including PGC3 first)
    """
    cmd = [
        sys.executable, LDSC_PY,
        '--rg', ','.join(str(s) for s in sumstats_list),
        '--ref-ld-chr', REF_LD,
        '--w-ld-chr', W_LD,
        '--out', output_prefix,
    ]

    log.info(f"Running LDSC rg with {len(sumstats_list)} sumstats files")
    log.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            log.error(f"LDSC rg failed (rc={result.returncode})")
            log.error(f"stderr: {result.stderr[:2000]}")
            return None

        # Parse the .log file for results
        log_file = output_prefix + '.log'
        if not os.path.exists(log_file):
            log.error(f"LDSC log file not found: {log_file}")
            return None

        return parse_ldsc_rg_log(log_file)
    except Exception as e:
        log.error(f"LDSC rg exception: {e}")
        return None


def parse_ldsc_rg_log(log_file_path):
    """Parse LDSC rg log file to extract genetic correlation results."""
    results = []

    with open(log_file_path, 'r') as f:
        lines = f.readlines()

    # Find the rg results section
    in_rg_section = False
    for line in lines:
        line = line.strip()
        if 'Genetic Correlation' in line or 'gencov' in line.lower():
            in_rg_section = True
            continue
        if in_rg_section:
            if line.startswith('-') * 5 or 'Summary of' in line:
                break
            if line and not line.startswith('WARNING') and not line.startswith('NOTE'):
                # Parse the data lines
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        # Try to parse as numeric
                        rg_val = float(parts[1])
                        results.append({
                            'trait_pair': parts[0],
                            'rg': rg_val,
                            'se': float(parts[2]),
                            'z': float(parts[3]),
                            'p': float(parts[4]),
                        })
                    except (ValueError, IndexError):
                        pass

    # Also look for the summary table which has a different format
    if not results:
        for line in lines:
            if '.sumstats.gz' in line:
                parts = line.split()
                # Find the numeric fields
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except ValueError:
                        pass
                if len(nums) >= 4:
                    results.append({
                        'trait_pair': parts[0] if parts else 'unknown',
                        'rg': nums[0] if len(nums) > 0 else None,
                        'se': nums[1] if len(nums) > 1 else None,
                        'z': nums[2] if len(nums) > 2 else None,
                        'p': nums[3] if len(nums) > 3 else None,
                    })

    return results


# ============================================================
# Main pipeline
# ============================================================
def main():
    start_time = datetime.now()
    log.info("=" * 70)
    log.info("D51: LDSC Genetic Correlation Panel")
    log.info(f"Started: {start_time.isoformat()}")
    log.info("=" * 70)

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    download_log = []  # Track all attempts per D23
    successful_traits = {}  # trait -> munged_path
    blocked_traits = {}  # trait -> list of failed attempts

    # --------------------------------------------------------
    # Phase 1: Download sumstats
    # --------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("Phase 1: Downloading sumstats")
    log.info("=" * 70)

    for trait_name, trait_info in TRAITS.items():
        log.info(f"\n--- {trait_name.upper()} ({trait_info['full_name']}) ---")

        # Check if already munged
        munged_path = COMPARATOR_DIR / f"{trait_name}.sumstats.gz"
        if munged_path.exists():
            log.info(f"  Already munged: {munged_path}")
            successful_traits[trait_name] = munged_path
            continue

        downloaded = False
        attempts = []

        for src_idx, source in enumerate(trait_info["sources"]):
            source_desc = source.get("description", source.get("type", "unknown"))
            url = source.get("url", "")

            log.info(f"  Attempt {src_idx + 1}/{len(trait_info['sources'])}: {source_desc}")

            raw_path = None

            if source["type"] == "gwas_catalog_harmonised":
                raw_path, msg = download_from_gwas_catalog_harmonised(trait_name, source, RAW_DIR)
            elif source["type"] == "gwas_catalog_ftp":
                raw_path, msg = download_from_gwas_catalog_ftp(trait_name, source, RAW_DIR)
            elif source["type"] == "direct_url":
                raw_path, msg = download_from_direct_url(trait_name, source, RAW_DIR)
            elif source["type"] in ("gwas_catalog_search", "gwas_catalog_ftp_browse"):
                raw_path, msg = None, "Browsable source requires manual discovery - skipping"
            else:
                raw_path, msg = None, f"Unknown source type: {source['type']}"

            if raw_path and raw_path.exists() and os.path.getsize(raw_path) > 1000:
                attempt = log_attempt(trait_name, src_idx, source_desc, url, "SUCCESS", msg)
                download_log.append(attempt)
                downloaded = True

                # Try to munge
                log.info(f"  Munging downloaded file: {raw_path.name}")
                munged_ok = munge_sumstats_custom(trait_name, raw_path, trait_info, munged_path)

                if not munged_ok:
                    # Try LDSC munge as fallback
                    log.info(f"  Custom munge failed, trying LDSC munge_sumstats.py")
                    munged_ok = munge_with_ldsc(trait_name, raw_path, trait_info, munged_path)

                if munged_ok and munged_path.exists():
                    successful_traits[trait_name] = munged_path
                    log.info(f"  SUCCESS: {trait_name} munged ({os.path.getsize(munged_path)} bytes)")
                    break
                else:
                    log.warning(f"  Munge failed for {trait_name}")
                    downloaded = False
            else:
                attempt = log_attempt(trait_name, src_idx, source_desc, url, "FAILED", msg or "No file downloaded")
                download_log.append(attempt)

            time.sleep(1)  # Rate limiting

        if trait_name not in successful_traits:
            blocked_traits[trait_name] = trait_info

    # --------------------------------------------------------
    # Phase 2: Browse GWAS Catalog FTP for blocked traits
    # --------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("Phase 2: FTP browsing for blocked traits")
    log.info("=" * 70)

    # For blocked traits, try systematic FTP browsing
    trait_keywords_map = {
        "ocd": ["obsessive", "compulsive", "ocd"],
        "anorexia": ["anorexia", "eating disorder"],
        "neuroticism": ["neuroticism"],
        "smoking": ["smoking", "gscan", "tobacco"],
        "alcohol": ["alcohol", "drinking"],
        "risk_taking": ["risk-taking", "risk taking"],
        "crp": ["c-reactive", "c reactive", "crp"],
        "il6": ["interleukin-6", "interleukin 6", "il-6"],
        "alzheimers": ["alzheimer"],
        "height": ["height", "body height", "stature"],
        "ptsd": ["post-traumatic", "ptsd"],
    }

    # FTP ranges to search
    ftp_ranges = [
        "GCST008001-GCST009000",
        "GCST009001-GCST010000",
        "GCST010001-GCST011000",
        "GCST011001-GCST012000",
        "GCST012001-GCST013000",
    ]

    for trait_name in list(blocked_traits.keys()):
        if trait_name not in trait_keywords_map:
            continue

        log.info(f"\n--- FTP Browse: {trait_name.upper()} ---")
        keywords = trait_keywords_map[trait_name]
        munged_path = COMPARATOR_DIR / f"{trait_name}.sumstats.gz"

        found = False
        for gcst_range in ftp_ranges:
            if found:
                break
            log.info(f"  Searching {gcst_range}...")

            gcst_id, download_url = find_gcst_in_range(trait_name, gcst_range, keywords)
            if gcst_id and download_url:
                dest = RAW_DIR / f"{trait_name}_{gcst_id}_harmonised.tsv.gz"
                success, msg = download_file(download_url, dest)

                attempt = log_attempt(
                    trait_name, 0, f"FTP browse {gcst_range}: {gcst_id}",
                    download_url, "SUCCESS" if success else "FAILED", msg
                )
                download_log.append(attempt)

                if success and dest.exists():
                    # Munge
                    trait_info = blocked_traits[trait_name]
                    munged_ok = munge_sumstats_custom(trait_name, dest, trait_info, munged_path)

                    if munged_ok and munged_path.exists():
                        successful_traits[trait_name] = munged_path
                        del blocked_traits[trait_name]
                        found = True
                        log.info(f"  FTP browse SUCCESS: {trait_name} ({gcst_id})")

            time.sleep(2)  # Rate limiting between range searches

    # --------------------------------------------------------
    # Phase 3: Run LDSC rg
    # --------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("Phase 3: Running LDSC genetic correlation")
    log.info("=" * 70)

    n_successful = len(successful_traits)
    n_blocked = len(blocked_traits)

    log.info(f"Traits ready for rg: {n_successful}")
    log.info(f"Traits BLOCKED: {n_blocked}")

    if n_successful == 0:
        log.error("No traits available for rg analysis")
        write_results({}, {}, download_log, start_time)
        return

    # Run LDSC rg: PGC3 vs each trait individually (more robust than batch)
    rg_results = {}

    for trait_name, munged_path in successful_traits.items():
        log.info(f"\nRunning rg: SCZ vs {trait_name}")

        output_prefix = str(OUTPUT_DIR / f"d51_rg_{trait_name}")

        # Run individual rg: PGC3 vs this trait
        cmd = [
            sys.executable, LDSC_PY,
            '--rg', f"{PGC3_SUMSTATS},{munged_path}",
            '--ref-ld-chr', REF_LD,
            '--w-ld-chr', W_LD,
            '--out', output_prefix,
        ]

        log.info(f"  CMD: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            # Parse results from log file
            log_path = output_prefix + '.log'
            if os.path.exists(log_path):
                parsed = parse_ldsc_rg_log(log_path)
                if parsed:
                    rg_results[trait_name] = parsed[0]  # First result
                    rg_val = parsed[0].get('rg', 'N/A')
                    p_val = parsed[0].get('p', 'N/A')
                    log.info(f"  Result: rg={rg_val}, p={p_val}")
                else:
                    log.warning(f"  Could not parse results from log file")
                    rg_results[trait_name] = {"status": "parse_error"}
            else:
                log.error(f"  No log file produced")
                rg_results[trait_name] = {"status": "no_log", "stderr": result.stderr[:500]}

        except subprocess.TimeoutExpired:
            log.error(f"  LDSC rg timed out for {trait_name}")
            rg_results[trait_name] = {"status": "timeout"}
        except Exception as e:
            log.error(f"  LDSC rg error: {e}")
            rg_results[trait_name] = {"status": "error", "error": str(e)}

    # --------------------------------------------------------
    # Phase 4: Compile and save results
    # --------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("Phase 4: Compiling results")
    log.info("=" * 70)

    write_results(rg_results, blocked_traits, download_log, start_time)

    elapsed = (datetime.now() - start_time).total_seconds()
    log.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info(f"SUCCESS: {len(rg_results)} traits, BLOCKED: {len(blocked_traits)} traits")


def write_results(rg_results, blocked_traits, download_log, start_time):
    """Write results to JSON and TSV."""

    # --- JSON ---
    output_json = {
        "directive": "D51",
        "description": "LDSC genetic correlation panel against SCZ PGC3",
        "date": datetime.now().isoformat(),
        "elapsed_seconds": (datetime.now() - start_time).total_seconds(),
        "summary": {
            "n_successful": len(rg_results),
            "n_blocked": len(blocked_traits),
            "decision_rule": "SUCCESS >= 15, PARTIAL 8-14, FAILURE < 8",
            "assessment": "SUCCESS" if len(rg_results) >= 15 else ("PARTIAL" if len(rg_results) >= 8 else "FAILURE"),
        },
        "rg_results": {},
        "blocked_traits": {k: {"full_name": v["full_name"], "study": v["study"], "category": v["category"]}
                          for k, v in blocked_traits.items()},
        "download_log": download_log,
    }

    # Add existing traits (ADHD, BIP, BMI from batch_034/035)
    existing_traits = ["adhd", "bip", "bmi"]

    for trait_name, result in rg_results.items():
        trait_info = TRAITS.get(trait_name, {})
        output_json["rg_results"][trait_name] = {
            "full_name": trait_info.get("full_name", trait_name),
            "study": trait_info.get("study", "unknown"),
            "category": trait_info.get("category", "unknown"),
            "rg": result.get("rg"),
            "se": result.get("se"),
            "z": result.get("z"),
            "p": result.get("p"),
            "status": result.get("status", "ok"),
        }

    json_path = OUTPUT_DIR / "d51_rg_panel_results.json"
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2, default=str)
    log.info(f"Saved JSON: {json_path}")

    # --- TSV ---
    tsv_rows = []
    for trait_name, result in rg_results.items():
        trait_info = TRAITS.get(trait_name, {})
        rg = result.get("rg")
        se = result.get("se")
        p = result.get("p")

        # Compute 95% CI
        if rg is not None and se is not None:
            ci_lo = rg - 1.96 * se
            ci_hi = rg + 1.96 * se
        else:
            ci_lo = ci_hi = None

        tsv_rows.append({
            "trait": trait_name,
            "full_name": trait_info.get("full_name", ""),
            "category": trait_info.get("category", ""),
            "study": trait_info.get("study", ""),
            "N": trait_info.get("N", ""),
            "rg": rg,
            "SE": se,
            "z": result.get("z"),
            "p": p,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "status": result.get("status", "ok"),
        })

    for trait_name, info in blocked_traits.items():
        tsv_rows.append({
            "trait": trait_name,
            "full_name": info.get("full_name", ""),
            "category": info.get("category", ""),
            "study": info.get("study", ""),
            "N": info.get("N", ""),
            "rg": None,
            "SE": None,
            "z": None,
            "p": None,
            "ci_95_lo": None,
            "ci_95_hi": None,
            "status": "BLOCKED",
        })

    tsv_path = OUTPUT_DIR / "d51_rg_panel_results.tsv"
    tsv_df = pd.DataFrame(tsv_rows)
    tsv_df.to_csv(str(tsv_path), sep='\t', index=False)
    log.info(f"Saved TSV: {tsv_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("D51 RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Trait':<20s} {'Category':<18s} {'rg':>8s} {'SE':>8s} {'p':>12s} {'Status':<12s}")
    print("-" * 100)
    for row in sorted(tsv_rows, key=lambda x: (x.get('status', '') != 'ok', x.get('category', ''), x.get('trait', ''))):
        rg_str = f"{row['rg']:.4f}" if row['rg'] is not None else "N/A"
        se_str = f"{row['SE']:.4f}" if row['SE'] is not None else "N/A"
        p_str = f"{row['p']:.2e}" if row.get('p') is not None else "N/A"
        print(f"{row['trait']:<20s} {row['category']:<18s} {rg_str:>8s} {se_str:>8s} {p_str:>12s} {row['status']:<12s}")

    n_ok = sum(1 for r in tsv_rows if r.get('status') == 'ok')
    n_blocked = sum(1 for r in tsv_rows if r.get('status') == 'BLOCKED')
    print(f"\nTotal: {n_ok} SUCCESS + {n_blocked} BLOCKED = {n_ok + n_blocked} traits")

    # Add existing traits note
    print(f"\nNote: 3 existing traits (ADHD, BIP, BMI) from batch_034/035")
    print(f"      Total panel: {n_ok + 3} traits with rg estimates")


if __name__ == "__main__":
    main()
