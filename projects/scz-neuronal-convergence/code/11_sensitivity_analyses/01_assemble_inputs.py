"""01_assemble_inputs.py — verify every input per preflight table, SHA-256, and dump to JSON.

WHY: Cardinal Rule 0 — never fabricate. Every downstream script depends on a frozen,
verifiable input set. We stamp SHA-256 so anomalies reproduce exactly.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import (ROOT, OUTPUT_DIR, LOGS_DIR, sha256_file,
                    GWAS_PARQUET, MARKERS_PARQ, GENCODE_JSON, GNOMAD_TSV,
                    GTEX_GCT, SCHEMA_TXT, PGC3_XLSX, SYNGO_GMT, HGNC_FAMILY,
                    BATCH048_JSON,
                    load_scz_genes, load_neuronal_markers, load_gencode,
                    load_gnomad, load_schema, load_pgc3, load_syngo,
                    load_hgnc_family, load_gtex_brain_and_any)

def main():
    log = LOGS_DIR / "run.log"
    t0 = time.time()
    report = {"generated_at": int(t0), "inputs": {}, "counts": {}}
    files = {
        "gwas_parquet": GWAS_PARQUET,
        "markers_parquet": MARKERS_PARQ,
        "gencode_json": GENCODE_JSON,
        "gnomad_tsv": GNOMAD_TSV,
        "gtex_gct": GTEX_GCT,
        "schema_txt": SCHEMA_TXT,
        "pgc3_xlsx": PGC3_XLSX,
        "syngo_gmt": SYNGO_GMT,
        "hgnc_family_tsv": HGNC_FAMILY,
        "batch_048_decomposition": BATCH048_JSON,
    }
    for k, p in files.items():
        p = Path(p)
        if not p.exists():
            report["inputs"][k] = {"path": str(p), "exists": False, "BLOCKED": True}
            print(f"BLOCKED: {k} -> {p}")
            continue
        report["inputs"][k] = {
            "path": str(p), "exists": True, "bytes": p.stat().st_size,
            "sha256": sha256_file(p)
        }

    # Counts and sanity checks
    scz = load_scz_genes()
    report["counts"]["scz_genes"] = len(scz)
    markers = load_neuronal_markers()
    report["counts"]["panglao_neuronal"] = len(markers)
    gencode = load_gencode()
    report["counts"]["gencode_pc_genes"] = len(gencode)
    gnomad = load_gnomad()
    report["counts"]["gnomad_canonical_mane"] = len(gnomad)
    report["counts"]["gnomad_pLI_ge_0.9"] = int((gnomad["pLI"] >= 0.9).sum())
    schema = load_schema()
    report["counts"]["schema"] = len(schema)
    pgc3 = load_pgc3()
    st12 = pgc3["st12"]
    edt1sheet = pgc3["edt1_sheet"]
    report["counts"]["st12_rows"] = int(len(st12))
    report["counts"]["st12_protein_coding"] = int((st12["gene_biotype"] == "protein_coding").sum())
    report["counts"]["st12_prioritised_eq_1"] = int((st12["Prioritised"] == 1).sum())
    report["counts"]["st12_prioritised_pc"] = int(((st12["Prioritised"] == 1) & (st12["gene_biotype"] == "protein_coding")).sum())
    report["counts"]["edt1_sheet_rows"] = int(len(edt1sheet))
    report["counts"]["edt1_sheet_pc"] = int((edt1sheet["gene_biotype"] == "protein_coding").sum())
    syngo = load_syngo()
    report["counts"]["syngo_terms"] = len(syngo)
    n_bp = sum(1 for k in syngo if k.rstrip().endswith(" BP"))
    n_cc = sum(1 for k in syngo if k.rstrip().endswith(" CC"))
    report["counts"]["syngo_BP_terms"] = n_bp
    report["counts"]["syngo_CC_terms"] = n_cc
    # SynGO gene sets
    from common import jaccard
    all_syngo = set().union(*syngo.values())
    bp_genes = set().union(*[v for k, v in syngo.items() if k.rstrip().endswith(" BP")])
    cc_genes = set().union(*[v for k, v in syngo.items() if k.rstrip().endswith(" CC")])
    report["counts"]["syngo_all_genes"] = len(all_syngo)
    report["counts"]["syngo_BP_genes"] = len(bp_genes)
    report["counts"]["syngo_CC_genes"] = len(cc_genes)

    hgnc = load_hgnc_family()
    report["counts"]["hgnc_family_rows"] = len(hgnc)

    brain, any_tiss = load_gtex_brain_and_any()
    report["counts"]["gtex_brain_expressed"] = len(brain)
    report["counts"]["gtex_any_expressed"] = len(any_tiss)

    # Note EDT1 provenance discrepancy (brief says n=261, actual Prioritised=1 is 120)
    report["edt1_provenance_note"] = {
        "brief_claim_n": 261,
        "actual_prioritised_eq_1_pc": report["counts"]["st12_prioritised_pc"],
        "edt1_sheet_pc": report["counts"]["edt1_sheet_pc"],
        "st12_all_pc": report["counts"]["st12_protein_coding"],
        "resolution": "Using EDT1 sheet pc (n=106, Prioritised==1) as 'prioritised', and ST12 all pc (~470) as 'broader'. Brief's n=261 does not correspond to any ST12 binary column; documented here instead of fabricated.",
    }

    out = OUTPUT_DIR / "assembled_inputs.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    elapsed = time.time() - t0
    from common import log_event
    log_event(log, f"[01_assemble] elapsed={elapsed:.2f}s | counts={report['counts']}")
    print(json.dumps({"status": "ok", "elapsed_s": elapsed, "counts": report['counts']}, indent=2))

if __name__ == "__main__":
    main()
