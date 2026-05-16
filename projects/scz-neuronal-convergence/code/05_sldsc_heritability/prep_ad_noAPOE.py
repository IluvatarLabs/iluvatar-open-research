#!/usr/bin/env python3
"""
batch_051_B: AD sumstats provenance check + APOE exclusion.

WHY (per brief R2): APOE region chr19:44411941-46386942 (GRCh37) dominates
AD heritability (~50%) and violates the LDSC polygenicity assumption.
Per Wightman 2021 convention we exclude SNPs within this interval before
S-LDSC, and we run AD both with APOE (AD_full) and without (AD_noAPOE).

WHY chose BP-range source = baselineLD chr19 annot: the AD sumstats file
(LDSC munged: SNP/A1/A2/Z/N) has NO BP column. We therefore resolve each
SNP's position via the baselineLD annot and then filter. This is the
deterministic and reproducible mapping used already by LDSC merges.

Outputs:
  data/ldsc/comparator_sumstats/alzheimers_noAPOE.sumstats.gz
  experiments/batch_051_B/output/apoe_filter_report.txt
"""
from __future__ import annotations

import gzip
from pathlib import Path
import sys

ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
AD_IN = ROOT / "data/ldsc/comparator_sumstats/alzheimers.sumstats.gz"
AD_OUT = ROOT / "data/ldsc/comparator_sumstats/alzheimers_noAPOE.sumstats.gz"
REPORT = ROOT / "experiments/batch_051_B/output/apoe_filter_report.txt"

# APOE region, GRCh37 — Wightman 2021 convention
APOE_CHR = 19
APOE_START = 44_411_941
APOE_END = 46_386_942

# Source of truth for SNP->BP mapping: baselineLD chr19 annot
BASELINE_CHR19 = ROOT / "data/ldsc/baselineLD/baselineLD.19.annot.gz"


def load_apoe_snps() -> set[str]:
    """Collect rsIDs on chr19 within APOE interval from baselineLD annot."""
    snps: set[str] = set()
    with gzip.open(BASELINE_CHR19, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        # Expected: CHR BP SNP CM ...
        i_chr = header.index("CHR")
        i_bp = header.index("BP")
        i_snp = header.index("SNP")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            try:
                bp = int(parts[i_bp])
            except ValueError:
                continue
            if parts[i_chr] == str(APOE_CHR) and APOE_START <= bp <= APOE_END:
                snps.add(parts[i_snp])
    return snps


def main() -> int:
    if not AD_IN.exists():
        print(f"ERROR: {AD_IN} not found", file=sys.stderr)
        return 1
    if not BASELINE_CHR19.exists():
        print(f"ERROR: {BASELINE_CHR19} not found", file=sys.stderr)
        return 1

    print("Loading APOE SNPs from baselineLD chr19 ...")
    apoe_snps = load_apoe_snps()
    print(f"  APOE region SNPs in baselineLD chr19: {len(apoe_snps):,}")

    total = 0
    kept = 0
    dropped = 0
    header_line = ""
    print("Filtering AD sumstats ...")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(AD_IN, "rt") as fin, gzip.open(AD_OUT, "wt") as fout:
        header_line = fin.readline()
        fout.write(header_line)
        cols = header_line.rstrip("\n").split("\t")
        i_snp = cols.index("SNP")
        for line in fin:
            total += 1
            parts = line.rstrip("\n").split("\t")
            if parts[i_snp] in apoe_snps:
                dropped += 1
                continue
            fout.write(line)
            kept += 1

    with REPORT.open("w") as rf:
        rf.write("APOE exclusion report (batch_051_B)\n")
        rf.write(f"Input : {AD_IN}\n")
        rf.write(f"Output: {AD_OUT}\n")
        rf.write(f"APOE coords (GRCh37): chr{APOE_CHR}:{APOE_START}-{APOE_END}\n")
        rf.write(f"APOE SNPs from baselineLD chr19 : {len(apoe_snps):,}\n")
        rf.write(f"AD sumstats total rows         : {total:,}\n")
        rf.write(f"AD sumstats rows dropped       : {dropped:,}\n")
        rf.write(f"AD sumstats rows kept          : {kept:,}\n")
        rf.write(f"Header preserved              : {header_line.strip()}\n")

    print(f"  total:  {total:,}")
    print(f"  kept :  {kept:,}")
    print(f"  drop :  {dropped:,}")
    print(f"Report -> {REPORT}")
    print(f"Output -> {AD_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
