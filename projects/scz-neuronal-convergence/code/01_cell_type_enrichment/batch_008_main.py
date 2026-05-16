#!/usr/bin/env python3
"""
Batch 008 Main Orchestration Script
====================================
Runs the complete GWAS-based cell-type enrichment pipeline.

Design: Approved (with limitations documented in design.yaml)
Pipeline: Auth → Download → Gene Mapping → Marker Extraction → Enrichment

WHY this orchestration:
- Each module can run independently for debugging
- Sequential execution ensures proper dependencies
- Comprehensive error handling and logging
- Progress reporting for long-running operations

Author: Marvin (implementation)
Date: 2026-04-09
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"batch_008_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_header():
    """Print experiment header."""
    print("\n" + "=" * 80)
    print("BATCH 008: GWAS-BASED CELL-TYPE ENRICHMENT")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Location: {Path(__file__).parent}")
    print("=" * 80)


def print_stage(stage_num: int, total: int, name: str):
    """Print stage header."""
    print(f"\n{'─' * 80}")
    print(f"STAGE {stage_num}/{total}: {name}")
    print(f"{'─' * 80}")


def run_auth():
    """
    Stage 1: Authentication

    Returns:
        dict: Authentication result
    """
    print_stage(1, 5, "OpenGWAS JWT Authentication")

    try:
        from batch_008_auth import setup_auth
        result = setup_auth()

        if result["success"]:
            print(f"\n✓ Authentication successful (method: {result['method']})")
            return {"success": True, "stage": "auth", "result": result}
        else:
            print(f"\n⚠ Authentication failed but continuing...")
            print(f"  Using fallback approach without JWT")
            return {"success": False, "stage": "auth", "result": result}

    except Exception as e:
        logger.error(f"Auth stage failed: {e}")
        print(f"\n✗ Auth error: {e}")
        return {"success": False, "stage": "auth", "error": str(e)}


def run_download():
    """
    Stage 2: GWAS Data Download

    Returns:
        dict: Download result
    """
    print_stage(2, 5, "GWAS Data Download")

    try:
        from batch_008_download import download_gwas_data
        result = download_gwas_data()

        if result["success"]:
            print(f"\n✓ Download complete")
            print(f"  Method: {result['method']}")
            print(f"  SNPs/Genes: {result['n_snps']}")
            return {"success": True, "stage": "download", "result": result}
        else:
            print(f"\n✗ Download failed")
            return {"success": False, "stage": "download", "error": result.get("message", "Unknown error")}

    except Exception as e:
        logger.error(f"Download stage failed: {e}")
        print(f"\n✗ Download error: {e}")
        return {"success": False, "stage": "download", "error": str(e)}


def run_gene_mapping():
    """
    Stage 3: Gene Mapping

    Returns:
        dict: Gene mapping result
    """
    print_stage(3, 5, "GWAS Gene Mapping")

    try:
        from batch_008_gene_mapping import map_gwas_to_genes
        result = map_gwas_to_genes()

        if result["success"]:
            print(f"\n✓ Gene mapping complete")
            print(f"  GWAS genes: {result['n_genes']}")
            print(f"  QC status: {result['qc_metrics'].get('status', 'unknown')}")
            return {"success": True, "stage": "gene_mapping", "result": result}
        else:
            print(f"\n✗ Gene mapping failed")
            return {"success": False, "stage": "gene_mapping", "error": result.get("message", "Unknown error")}

    except Exception as e:
        logger.error(f"Gene mapping stage failed: {e}")
        print(f"\n✗ Gene mapping error: {e}")
        return {"success": False, "stage": "gene_mapping", "error": str(e)}


def run_marker_extraction():
    """
    Stage 4: Cell Type Marker Extraction

    Returns:
        dict: Marker extraction result
    """
    print_stage(4, 5, "Cell Type Marker Extraction")

    try:
        from batch_008_markers import extract_cell_type_markers
        result = extract_cell_type_markers()

        if result["success"]:
            print(f"\n✓ Marker extraction complete")
            print(f"  Cell types: {result['n_cell_types']}")
            print(f"  Total markers: {result['n_markers']}")
            return {"success": True, "stage": "markers", "result": result}
        else:
            print(f"\n✗ Marker extraction failed")
            return {"success": False, "stage": "markers", "error": result.get("message", "Unknown error")}

    except Exception as e:
        logger.error(f"Marker extraction stage failed: {e}")
        print(f"\n✗ Marker extraction error: {e}")
        return {"success": False, "stage": "markers", "error": str(e)}


def run_enrichment():
    """
    Stage 5: Enrichment Analysis

    Returns:
        dict: Enrichment analysis result
    """
    print_stage(5, 5, "Cell Type Enrichment Analysis")

    try:
        from batch_008_enrichment import run_enrichment_analysis
        result = run_enrichment_analysis()

        if result["success"]:
            print(f"\n✓ Enrichment analysis complete")
            summary = result["summary"]
            print(f"  Cell types tested: {summary.get('n_cell_types_tested', 0)}")
            print(f"  Positive: {summary.get('n_positive', 0)}")
            print(f"  Negative: {summary.get('n_negative', 0)}")
            print(f"  Inconclusive: {summary.get('n_inconclusive', 0)}")
            print(f"  Overall: {summary.get('overall_classification', 'N/A')}")
            return {"success": True, "stage": "enrichment", "result": result}
        else:
            print(f"\n✗ Enrichment analysis failed")
            return {"success": False, "stage": "enrichment", "error": result.get("message", "Unknown error")}

    except Exception as e:
        logger.error(f"Enrichment stage failed: {e}")
        print(f"\n✗ Enrichment error: {e}")
        return {"success": False, "stage": "enrichment", "error": str(e)}


def generate_preflight_report(stage_results: List[Dict]):
    """
    Generate preflight check report.

    Args:
        stage_results: List of stage results

    Returns:
        dict: Report data
    """
    print("\n" + "=" * 80)
    print("PREFLIGHT CHECK REPORT")
    print("=" * 80)

    report = {
        "experiment_id": "batch_008",
        "date": datetime.now().isoformat(),
        "stages": [],
        "overall_status": "unknown"
    }

    all_success = True
    any_success = False

    for result in stage_results:
        stage_report = {
            "stage": result["stage"],
            "success": result["success"],
            "message": result.get("result", {}).get("message", result.get("error", "Unknown"))
        }
        report["stages"].append(stage_report)

        if result["success"]:
            print(f"✓ {result['stage'].upper()}: SUCCESS")
            any_success = True
        else:
            print(f"✗ {result['stage'].upper()}: FAILED")
            print(f"    Error: {result.get('error', 'Unknown')}")
            all_success = False

    if all_success:
        report["overall_status"] = "READY"
        print("\n✓ ALL STAGES PASSED - Experiment ready")
    elif any_success:
        report["overall_status"] = "PARTIAL"
        print("\n⚠ PARTIAL SUCCESS - Some stages failed but continuing")
    else:
        report["overall_status"] = "BLOCKED"
        print("\n✗ ALL STAGES FAILED - Experiment blocked")

    return report


def save_results(stage_results: List[Dict], preflight_report: Dict):
    """
    Save pipeline results to JSON.

    Args:
        stage_results: List of stage results
        preflight_report: Preflight report data
    """
    OUTPUT_DIR = Path(__file__).parent / "results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save preflight report
    preflight_file = OUTPUT_DIR / "preflight_report.json"
    with open(preflight_file, 'w') as f:
        json.dump(preflight_report, f, indent=2)
    print(f"\n✓ Preflight report saved to {preflight_file}")

    # Save stage results
    stages_file = OUTPUT_DIR / "pipeline_stages.json"
    with open(stages_file, 'w') as f:
        json.dump(stage_results, f, indent=2)
    print(f"✓ Stage results saved to {stages_file}")


def main():
    """
    Main orchestration function.

    Runs all pipeline stages in sequence and generates reports.
    """
    print_header()

    # Track stage results
    stage_results = []

    # Stage 1: Authentication
    result = run_auth()
    stage_results.append(result)

    # If auth fails, we can still continue with fallback
    # Don't abort the pipeline

    # Stage 2: Download
    result = run_download()
    stage_results.append(result)

    if not result["success"]:
        logger.warning("Download failed - may need fallback approach")

    # Stage 3: Gene Mapping
    result = run_gene_mapping()
    stage_results.append(result)

    if not result["success"]:
        logger.error("Gene mapping failed - cannot proceed to enrichment")
        # Generate report and exit
        report = generate_preflight_report(stage_results)
        save_results(stage_results, report)
        return 1

    # Stage 4: Marker Extraction
    result = run_marker_extraction()
    stage_results.append(result)

    if not result["success"]:
        logger.error("Marker extraction failed - cannot proceed to enrichment")
        report = generate_preflight_report(stage_results)
        save_results(stage_results, report)
        return 1

    # Stage 5: Enrichment Analysis
    result = run_enrichment()
    stage_results.append(result)

    # Generate preflight report
    report = generate_preflight_report(stage_results)

    # Save results
    save_results(stage_results, report)

    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Overall status: {report['overall_status']}")
    print(f"Stages completed: {sum(1 for r in stage_results if r['success'])}/{len(stage_results)}")
    print("=" * 80 + "\n")

    return 0 if report['overall_status'] in ['READY', 'PARTIAL'] else 1


if __name__ == "__main__":
    sys.exit(main())
