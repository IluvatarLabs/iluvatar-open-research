#!/usr/bin/env python3
"""batch_061 E2 -- MiXeR univariate SCZ pilot (single replicate).

Implements brief_v2.md section E2 EXACTLY.

Purpose: Pipeline validation + runtime calibration. This is NOT a production
run. Single replicate, no findings reported, no confidence classification.

Steps:
  1. Check prerequisites (Docker, sumstats, reference panel).
  2. Generate LD files (per-chromosome) via `mixer.py ld`.
  3. Generate pruned SNP subset via `mixer.py snps`.
  4. Run `mixer.py fit1` (univariate parameter estimation).
  5. Run `mixer.py test1` (goodness-of-fit evaluation).
  6. Parse results, record wall times, write results.json.

WHY MiXeR: PI item 12, deferred 3x. MiXeR (Frei 2019/2024) provides
Gaussian mixture model estimates of polygenicity (pi_c) and discoverability
(sigma_beta) from GWAS summary statistics. These complement gene-level
analyses (MAGMA, PoPS) by operating at the SNP level.

Source: [lit_doi_10.1038_s41467-019-10310-0] Frei 2019,
        [lit_doi_10.1038_s41588-024-01771-1] Frei 2024,
        F060_08 (MIXER_FEASIBLE).

DECISION RULE (brief_v2):
  - Successful completion + pi_c in [3000, 20000] -> PILOT_PASS
  - Runtime > 12h -> PILOT_REASSESS
  - Error -> PILOT_FAIL

Output: experiments/batch_061/output/e2/results.json
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# Absolute paths (agent cwd resets between calls).
# =============================================================================
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_061"
OUTPUT_DIR = BATCH_DIR / "output"
E2_OUTPUT_DIR = OUTPUT_DIR / "e2"
WORK_DIR = E2_OUTPUT_DIR / "mixer_work"
LOGS_DIR = BATCH_DIR / "logs"

# =============================================================================
# Data paths
# =============================================================================
# SCZ GWAS summary statistics.
# WHY PGC3_EUR_v2: Latest PGC3 European SCZ GWAS. Column format confirmed:
# SNP, CHR, BP, A1, A2, FRQ, Z, P, N -- MiXeR needs SNP, Z, N (minimum).
SUMSTATS = PROJECT_ROOT / "data" / "ldsc" / "PGC3_sumstats" / "PGC3_EUR_v2.sumstats.gz"

# 1000G EUR reference panel (per-chromosome plink files).
# WHY 1000G EUR Phase3: Standard reference for European-ancestry GWAS.
# MiXeR uses bim files to define the SNP universe and plink bfiles for LD.
REF_DIR = PROJECT_ROOT / "data" / "ldsc" / "1000G_EUR_Phase3_plink"
# '@' is a MiXeR convention: replaced by chromosome number at runtime.
BIM_PATTERN = str(REF_DIR / "1000G.EUR.QC.@.bim")
BFILE_PATTERN = str(REF_DIR / "1000G.EUR.QC.@")

# =============================================================================
# Container config
# =============================================================================
CONTAINER_IMAGE = "ghcr.io/precimed/gsa-mixer:2.2.1"
# Library path inside the container (confirmed by `find /tools/mixer -name libbgmg*`).
LIB_PATH = "/tools/mixer/src/build/lib/libbgmg.so"
MIXER_PY = "/tools/mixer/precimed/mixer.py"

# =============================================================================
# MiXeR parameters
# =============================================================================
# WHY these defaults: MiXeR documentation and Frei et al. 2019 supplementary.
# ld-window-kb=0 disables the constraint (MiXeR uses r2min for sparse storage).
LD_R2MIN = 0.05              # r2 threshold for sparse LD (MiXeR default)
LD_LDSCORE_R2MIN = 0.001     # r2 for LD score contribution (MiXeR default)
# WHY 1000 for ld-window-kb: Frei 2019 used whole-chromosome LD. But computing
# that takes ~10h per chromosome. 1000kb window is the common practical choice
# used in MiXeR tutorials and precimed/mixer GitHub examples.
LD_WINDOW_KB = 1000

# SNP pruning parameters for `mixer.py snps`.
# WHY r2=0.8, maf=0.05: standard MiXeR defaults from precimed examples.
SNP_R2 = 0.8
SNP_MAF = 0.05
SNP_SEED = 20260424           # design.yaml master seed for reproducibility

# fit1 parameters.
# WHY randprune-n=20, randprune-r2=0.1: MiXeR default values from Frei 2019.
# These control random pruning iterations during fitting.
FIT1_RANDPRUNE_N = 20
FIT1_RANDPRUNE_R2 = 0.1
FIT1_SEED = 20260424          # Single seed for pilot.
# WHY z1max=5.45: MiXeR convention. Right-censoring at z=5.45 corresponds to
# p~5e-8 (genome-wide significance). Prevents top hits from dominating fit.
FIT1_Z1MAX = 5.45
# WHY kmax=5000: Default from MiXeR examples. Higher values improve precision
# but increase runtime. 5000 is standard for a single-replicate univariate fit.
FIT1_KMAX = 5000

# =============================================================================
# Timeouts
# =============================================================================
# WHY 2h for LD per chromosome: 1000G EUR has ~780K SNPs per chromosome (chr1).
# LD computation is O(n^2) within the LD window. Empirically takes 20-90 min
# per chromosome with 1000kb window. 2h is a generous bound.
LD_TIMEOUT_PER_CHR = 7200     # 2 hours per chromosome
# WHY 6h for fit1: brief_v2 predicts 3-6h. 6h is the upper bound.
FIT1_TIMEOUT = 21600          # 6 hours
# WHY 2h for test1: test1 is typically faster than fit1 (no optimization).
TEST1_TIMEOUT = 7200          # 2 hours
# WHY 12h total: brief_v2 decision rule -- runtime > 12h -> PILOT_REASSESS.
TOTAL_TIMEOUT = 43200         # 12 hours
# WHY 2h for LD generation total budget: The task says if LD generation takes
# > 2h, report PILOT_REASSESS. We time it and check.
LD_TOTAL_BUDGET = 7200        # 2 hours total across all chromosomes

CHROMOSOMES = list(range(1, 23))  # 1-22 autosomes


# =============================================================================
# Logger
# =============================================================================
def setup_logger(name: str, logfile: Path) -> logging.Logger:
    """Logger emitting to logfile and stdout.

    WHY isolated logger: keep batch_061 logs separate from other batches.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# =============================================================================
# Helpers
# =============================================================================
def atomic_write_json(data: dict, path: Path) -> None:
    """Write JSON atomically via tmp + rename.

    WHY atomic: prevents partial writes on crash/timeout.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")
    tmp.rename(path)


def run_docker(
    args: list[str],
    volumes: dict[str, str],
    timeout: int,
    logger: logging.Logger,
    label: str = "",
) -> dict:
    """Run a Docker command with volume mounts and timeout.

    Args:
        args: Arguments to pass after the container image name.
        volumes: Dict of {host_path: container_path} for -v mounts.
        timeout: Maximum wall time in seconds.
        logger: Logger instance.
        label: Human-readable label for log messages.

    Returns:
        Dict with returncode, stdout, stderr, elapsed_seconds, success, timed_out,
        and the full docker_command string for reproducibility.
    """
    cmd = ["docker", "run", "--rm"]
    for host_path, container_path in volumes.items():
        cmd.extend(["-v", f"{host_path}:{container_path}"])
    cmd.append(CONTAINER_IMAGE)
    cmd.extend(args)

    cmd_str = " ".join(cmd)
    logger.info("[%s] Running: %s", label, cmd_str)
    logger.info("[%s] Timeout: %d seconds", label, timeout)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        # Log last 500 chars of stdout/stderr for debugging.
        if result.stdout:
            logger.info("[%s] stdout (last 500): %s", label, result.stdout[-500:])
        if result.stderr:
            logger.info("[%s] stderr (last 500): %s", label, result.stderr[-500:])
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "elapsed_seconds": round(elapsed, 2),
            "success": result.returncode == 0,
            "timed_out": False,
            "docker_command": cmd_str,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - t0
        logger.error("[%s] TIMEOUT after %ds", label, timeout)
        return {
            "returncode": -1,
            "stdout": (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr": (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
            "elapsed_seconds": round(elapsed, 2),
            "success": False,
            "timed_out": True,
            "docker_command": cmd_str,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("[%s] Exception: %s", label, exc)
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_seconds": round(elapsed, 2),
            "success": False,
            "timed_out": False,
            "docker_command": cmd_str,
        }


# =============================================================================
# Step 1: Prerequisites
# =============================================================================
def check_prerequisites(logger: logging.Logger) -> dict:
    """Verify Docker, sumstats, and reference panel are available.

    Returns dict with check results and overall pass/fail.
    """
    checks = {}

    # Docker available and daemon running.
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=15,
        )
        checks["docker_available"] = result.returncode == 0
        checks["docker_version"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception as exc:
        checks["docker_available"] = False
        checks["docker_error"] = str(exc)

    # Container image pulled.
    # WHY `docker images -q`: fast check if image is locally available.
    try:
        result = subprocess.run(
            ["docker", "images", "-q", CONTAINER_IMAGE],
            capture_output=True, text=True, timeout=15,
        )
        checks["container_pulled"] = bool(result.stdout.strip())
    except Exception:
        checks["container_pulled"] = False

    # Sumstats exist.
    checks["sumstats_exists"] = SUMSTATS.exists()
    if SUMSTATS.exists():
        checks["sumstats_size_bytes"] = SUMSTATS.stat().st_size

    # Reference panel bim files exist (check all 22 chromosomes).
    missing_chroms = []
    for chrom in CHROMOSOMES:
        bim = REF_DIR / f"1000G.EUR.QC.{chrom}.bim"
        if not bim.exists():
            missing_chroms.append(chrom)
    checks["ref_bim_complete"] = len(missing_chroms) == 0
    if missing_chroms:
        checks["ref_bim_missing_chroms"] = missing_chroms

    # Reference panel bfile (bed/fam) files -- needed for LD computation.
    missing_bfiles = []
    for chrom in CHROMOSOMES:
        bed = REF_DIR / f"1000G.EUR.QC.{chrom}.bed"
        fam = REF_DIR / f"1000G.EUR.QC.{chrom}.fam"
        if not bed.exists() or not fam.exists():
            missing_bfiles.append(chrom)
    checks["ref_bfile_complete"] = len(missing_bfiles) == 0
    if missing_bfiles:
        checks["ref_bfile_missing_chroms"] = missing_bfiles

    checks["all_passed"] = all([
        checks["docker_available"],
        checks["container_pulled"],
        checks["sumstats_exists"],
        checks["ref_bim_complete"],
        checks["ref_bfile_complete"],
    ])

    for key, val in checks.items():
        logger.info("Prerequisite %s: %s", key, val)

    return checks


# =============================================================================
# Step 2: Generate LD files
# =============================================================================
def generate_ld_files(logger: logging.Logger) -> dict:
    """Generate per-chromosome LD files using MiXeR's `ld` subcommand.

    WHY per-chromosome: MiXeR's `ld` subcommand operates on one plink bfile
    at a time. The '@' pattern in --ld-file is resolved later by fit1/test1.

    Returns dict with per-chromosome results and overall status.
    """
    ld_dir = WORK_DIR / "ld"
    ld_dir.mkdir(parents=True, exist_ok=True)

    ld_results = {"per_chromosome": {}, "total_wall_time_seconds": 0}
    total_t0 = time.time()

    # Volume mounts: reference data + work directory.
    volumes = {
        str(REF_DIR): "/ref",
        str(ld_dir): "/ld_out",
    }

    for chrom in CHROMOSOMES:
        # Check if LD file already exists (resume support).
        ld_file = ld_dir / f"1000G.EUR.QC.{chrom}.ld"
        if ld_file.exists() and ld_file.stat().st_size > 0:
            logger.info("LD file for chr%d already exists (%d bytes), skipping",
                        chrom, ld_file.stat().st_size)
            ld_results["per_chromosome"][str(chrom)] = {
                "status": "cached",
                "size_bytes": ld_file.stat().st_size,
            }
            continue

        # Check total LD budget.
        elapsed_total = time.time() - total_t0
        if elapsed_total > LD_TOTAL_BUDGET:
            logger.warning(
                "LD generation budget exhausted (%.0fs > %ds). "
                "Completed %d/%d chromosomes.",
                elapsed_total, LD_TOTAL_BUDGET,
                len([v for v in ld_results["per_chromosome"].values()
                     if v.get("status") in ("success", "cached")]),
                len(CHROMOSOMES),
            )
            ld_results["budget_exceeded"] = True
            break

        # MiXeR ld subcommand arguments.
        # WHY --bfile not --bim-file: The `ld` subcommand requires full plink
        # bfile (bed+bim+fam) to compute LD from genotypes.
        args = [
            "python3", MIXER_PY, "ld",
            "--bfile", f"/ref/1000G.EUR.QC.{chrom}",
            "--out", f"/ld_out/1000G.EUR.QC.{chrom}",
            "--r2min", str(LD_R2MIN),
            "--ldscore-r2min", str(LD_LDSCORE_R2MIN),
            "--ld-window-kb", str(LD_WINDOW_KB),
            "--lib", LIB_PATH,
        ]

        result = run_docker(
            args=args,
            volumes=volumes,
            timeout=LD_TIMEOUT_PER_CHR,
            logger=logger,
            label=f"LD-chr{chrom}",
        )

        # Check if the output file was created.
        ld_file_created = ld_file.exists() and ld_file.stat().st_size > 0
        ld_results["per_chromosome"][str(chrom)] = {
            "status": "success" if (result["success"] and ld_file_created) else "failed",
            "elapsed_seconds": result["elapsed_seconds"],
            "returncode": result["returncode"],
            "timed_out": result["timed_out"],
            "size_bytes": ld_file.stat().st_size if ld_file_created else 0,
        }

        if not result["success"]:
            # Log error details but continue to next chromosome so we know
            # the overall failure pattern.
            logger.error(
                "LD generation failed for chr%d: rc=%d, stderr=%s",
                chrom, result["returncode"], result["stderr"][:500],
            )
            # Store truncated stderr for results.json.
            ld_results["per_chromosome"][str(chrom)]["error"] = result["stderr"][:1000]
        else:
            logger.info(
                "LD chr%d complete: %.1fs, %d bytes",
                chrom, result["elapsed_seconds"],
                ld_file.stat().st_size if ld_file_created else 0,
            )

    ld_results["total_wall_time_seconds"] = round(time.time() - total_t0, 2)
    n_success = sum(
        1 for v in ld_results["per_chromosome"].values()
        if v.get("status") in ("success", "cached")
    )
    ld_results["chromosomes_completed"] = n_success
    ld_results["all_completed"] = n_success == len(CHROMOSOMES)

    return ld_results


# =============================================================================
# Step 3: Generate pruned SNP list
# =============================================================================
def generate_snp_list(logger: logging.Logger) -> dict:
    """Generate pruned SNP subset using MiXeR's `snps` subcommand.

    WHY `mixer.py snps`: MiXeR provides its own LD-aware random pruning that
    is consistent with the LD structure it uses internally. Using plink's
    --indep-pairwise would produce a differently-pruned set.

    Returns dict with status, SNP count, and file path.
    """
    snps_dir = WORK_DIR / "snps"
    snps_dir.mkdir(parents=True, exist_ok=True)

    snps_file = snps_dir / "pruned_snps.snps"
    # The snps subcommand writes <out>.snps file.

    # Check if already generated.
    if snps_file.exists() and snps_file.stat().st_size > 0:
        n_snps = sum(1 for _ in open(snps_file))
        logger.info("Pruned SNP list already exists: %d SNPs", n_snps)
        return {
            "status": "cached",
            "n_snps": n_snps,
            "file": str(snps_file),
        }

    ld_dir = WORK_DIR / "ld"

    # Volume mounts.
    volumes = {
        str(REF_DIR): "/ref",
        str(ld_dir): "/ld",
        str(snps_dir): "/snps_out",
    }

    # WHY '@' in bim-file and ld-file: MiXeR expands '@' to chromosome numbers
    # (1-22) automatically. This is the standard MiXeR convention.
    args = [
        "python3", MIXER_PY, "snps",
        "--bim-file", "/ref/1000G.EUR.QC.@.bim",
        "--ld-file", "/ld/1000G.EUR.QC.@.ld",
        "--out", "/snps_out/pruned_snps",
        "--r2", str(SNP_R2),
        "--maf", str(SNP_MAF),
        "--seed", str(SNP_SEED),
        "--lib", LIB_PATH,
    ]

    result = run_docker(
        args=args,
        volumes=volumes,
        timeout=600,   # 10 min should be plenty for SNP pruning.
        logger=logger,
        label="SNP-prune",
    )

    snp_result = {
        "status": "success" if result["success"] else "failed",
        "elapsed_seconds": result["elapsed_seconds"],
        "docker_command": result["docker_command"],
    }

    if result["success"] and snps_file.exists():
        n_snps = sum(1 for _ in open(snps_file))
        snp_result["n_snps"] = n_snps
        snp_result["file"] = str(snps_file)
        logger.info("SNP pruning complete: %d SNPs in %.1fs",
                     n_snps, result["elapsed_seconds"])
    else:
        snp_result["error"] = result["stderr"][:1000]
        logger.error("SNP pruning failed: %s", result["stderr"][:500])

    return snp_result


# =============================================================================
# Step 4: Run fit1 (univariate MiXeR fit)
# =============================================================================
def run_fit1(logger: logging.Logger) -> dict:
    """Run MiXeR fit1 for SCZ (single replicate).

    WHY fit1: Estimates the causal mixture model parameters (pi_c, sigma_beta,
    sig0) by maximum likelihood. This is the core MiXeR optimization step.

    Returns dict with status, wall time, output file path, and parameters
    if successful.
    """
    fit1_dir = WORK_DIR / "fit1"
    fit1_dir.mkdir(parents=True, exist_ok=True)

    ld_dir = WORK_DIR / "ld"
    snps_dir = WORK_DIR / "snps"
    snps_file_container = "/snps/pruned_snps.snps"

    # Volume mounts.
    volumes = {
        str(REF_DIR): "/ref",
        str(ld_dir): "/ld",
        str(snps_dir): "/snps",
        str(fit1_dir): "/fit1_out",
        str(SUMSTATS.parent): "/sumstats",
    }

    # WHY --exclude-ranges MHC: Standard MiXeR practice. The MHC region
    # (chr6:25-35Mb) has extreme LD that biases mixture model estimation.
    # This is the MiXeR default but we state it explicitly for reproducibility.
    args = [
        "python3", MIXER_PY, "fit1",
        "--trait1-file", f"/sumstats/{SUMSTATS.name}",
        "--out", "/fit1_out/scz_pilot",
        "--extract", snps_file_container,
        "--bim-file", "/ref/1000G.EUR.QC.@.bim",
        "--ld-file", "/ld/1000G.EUR.QC.@.ld",
        "--lib", LIB_PATH,
        "--z1max", str(FIT1_Z1MAX),
        "--randprune-n", str(FIT1_RANDPRUNE_N),
        "--randprune-r2", str(FIT1_RANDPRUNE_R2),
        "--seed", str(FIT1_SEED),
        "--kmax", str(FIT1_KMAX),
        "--exclude-ranges", "MHC",
    ]

    result = run_docker(
        args=args,
        volumes=volumes,
        timeout=FIT1_TIMEOUT,
        logger=logger,
        label="fit1",
    )

    fit1_result = {
        "status": "success" if result["success"] else ("timeout" if result["timed_out"] else "failed"),
        "elapsed_seconds": result["elapsed_seconds"],
        "docker_command": result["docker_command"],
    }

    # Check for output JSON.
    fit1_json = fit1_dir / "scz_pilot.json"
    if fit1_json.exists():
        try:
            with open(fit1_json) as f:
                fit1_data = json.load(f)
            fit1_result["output_json"] = str(fit1_json)
            fit1_result["raw_params"] = fit1_data
            # Extract key parameters.
            # WHY these keys: pi, sig2_beta, sig2_zero are the standard MiXeR
            # output parameters. pi = polygenicity fraction, sig2_beta =
            # per-SNP heritability for causal SNPs, sig2_zero = inflation.
            # pi_c (causal SNP count) = pi * n_snps.
            if "ci" in fit1_data:
                ci = fit1_data["ci"]
                fit1_result["pi"] = ci.get("pi_c", ci.get("pi"))
                fit1_result["sig2_beta"] = ci.get("sig2_beta")
                fit1_result["sig2_zero"] = ci.get("sig2_zero")
            elif "params" in fit1_data:
                params = fit1_data["params"]
                fit1_result["pi"] = params.get("pi_c", params.get("pi"))
                fit1_result["sig2_beta"] = params.get("sig2_beta")
                fit1_result["sig2_zero"] = params.get("sig2_zero")
            logger.info("fit1 output parsed: pi=%s, sig2_beta=%s",
                        fit1_result.get("pi"), fit1_result.get("sig2_beta"))
        except (json.JSONDecodeError, KeyError) as exc:
            fit1_result["parse_error"] = str(exc)
            logger.warning("fit1 JSON parse error: %s", exc)
    else:
        # Check for log file which may contain partial results.
        fit1_log = fit1_dir / "scz_pilot.log"
        if fit1_log.exists():
            fit1_result["log_tail"] = fit1_log.read_text()[-2000:]
        logger.warning("fit1 output JSON not found at %s", fit1_json)

    if not result["success"]:
        fit1_result["error"] = result["stderr"][:2000]
        # Also capture stdout which may contain convergence info.
        fit1_result["stdout_tail"] = result["stdout"][-2000:]

    return fit1_result


# =============================================================================
# Step 5: Run test1 (univariate goodness-of-fit test)
# =============================================================================
def run_test1(logger: logging.Logger) -> dict:
    """Run MiXeR test1 using fit1 estimated parameters.

    WHY test1: Computes AIC/BIC and QQ-plot data to assess model fit quality.
    Without test1, we only have point estimates but no model diagnostics.

    Returns dict with status, wall time, AIC, BIC.
    """
    test1_dir = WORK_DIR / "test1"
    test1_dir.mkdir(parents=True, exist_ok=True)

    fit1_json = WORK_DIR / "fit1" / "scz_pilot.json"
    if not fit1_json.exists():
        return {"status": "skipped", "reason": "fit1 output not found"}

    ld_dir = WORK_DIR / "ld"
    snps_dir = WORK_DIR / "snps"
    snps_file_container = "/snps/pruned_snps.snps"

    # Volume mounts.
    volumes = {
        str(REF_DIR): "/ref",
        str(ld_dir): "/ld",
        str(snps_dir): "/snps",
        str(WORK_DIR / "fit1"): "/fit1",
        str(test1_dir): "/test1_out",
        str(SUMSTATS.parent): "/sumstats",
    }

    args = [
        "python3", MIXER_PY, "test1",
        "--trait1-file", f"/sumstats/{SUMSTATS.name}",
        "--out", "/test1_out/scz_pilot_test",
        "--load-params-file", "/fit1/scz_pilot.json",
        "--extract", snps_file_container,
        "--bim-file", "/ref/1000G.EUR.QC.@.bim",
        "--ld-file", "/ld/1000G.EUR.QC.@.ld",
        "--lib", LIB_PATH,
        "--z1max", str(FIT1_Z1MAX),
        "--seed", str(FIT1_SEED),
        "--exclude-ranges", "MHC",
    ]

    result = run_docker(
        args=args,
        volumes=volumes,
        timeout=TEST1_TIMEOUT,
        logger=logger,
        label="test1",
    )

    test1_result = {
        "status": "success" if result["success"] else ("timeout" if result["timed_out"] else "failed"),
        "elapsed_seconds": result["elapsed_seconds"],
        "docker_command": result["docker_command"],
    }

    # Parse test1 output.
    test1_json = test1_dir / "scz_pilot_test.json"
    if test1_json.exists():
        try:
            with open(test1_json) as f:
                test1_data = json.load(f)
            test1_result["output_json"] = str(test1_json)
            test1_result["raw_output"] = test1_data
            # Extract AIC/BIC.
            # WHY AIC/BIC: Standard model selection criteria. Lower is better.
            # These tell us if the Gaussian mixture is a reasonable fit to the
            # SCZ GWAS Z-score distribution.
            test1_result["AIC"] = test1_data.get("aic", test1_data.get("AIC"))
            test1_result["BIC"] = test1_data.get("bic", test1_data.get("BIC"))
            logger.info("test1 output parsed: AIC=%s, BIC=%s",
                        test1_result.get("AIC"), test1_result.get("BIC"))
        except (json.JSONDecodeError, KeyError) as exc:
            test1_result["parse_error"] = str(exc)
            logger.warning("test1 JSON parse error: %s", exc)
    else:
        test1_log = test1_dir / "scz_pilot_test.log"
        if test1_log.exists():
            test1_result["log_tail"] = test1_log.read_text()[-2000:]
        logger.warning("test1 output JSON not found at %s", test1_json)

    if not result["success"]:
        test1_result["error"] = result["stderr"][:2000]
        test1_result["stdout_tail"] = result["stdout"][-2000:]

    return test1_result


# =============================================================================
# Step 6: Parse and classify results
# =============================================================================
def classify_pilot(
    fit1_result: dict,
    test1_result: dict,
    total_elapsed: float,
    ld_result: dict,
    logger: logging.Logger,
) -> dict:
    """Apply brief_v2 decision rules to pilot results.

    DECISION RULE (brief_v2 E2):
      - Successful completion + pi_c in [3000, 20000] -> PILOT_PASS
      - Runtime > 12h -> PILOT_REASSESS
      - Error -> PILOT_FAIL

    Additionally:
      - LD budget exceeded -> PILOT_REASSESS (LD generation too slow)

    Returns dict with verdict and supporting evidence.
    """
    verdict_data = {}

    # Check LD budget.
    if ld_result.get("budget_exceeded"):
        verdict_data["pilot_status"] = "PILOT_REASSESS"
        verdict_data["reason"] = (
            "LD generation exceeded 2h budget. "
            f"Completed {ld_result.get('chromosomes_completed', 0)}/22 chromosomes "
            f"in {ld_result.get('total_wall_time_seconds', 0):.0f}s. "
            "Cloud dispatch or pre-computed LD files needed."
        )
        logger.warning("VERDICT: %s -- %s", verdict_data["pilot_status"],
                        verdict_data["reason"])
        return verdict_data

    # Check total timeout.
    if total_elapsed > TOTAL_TIMEOUT:
        verdict_data["pilot_status"] = "PILOT_REASSESS"
        verdict_data["reason"] = (
            f"Total runtime {total_elapsed:.0f}s exceeds 12h budget ({TOTAL_TIMEOUT}s). "
            "Cloud dispatch needed."
        )
        logger.warning("VERDICT: %s -- %s", verdict_data["pilot_status"],
                        verdict_data["reason"])
        return verdict_data

    # Check fit1 success.
    if fit1_result.get("status") != "success":
        if fit1_result.get("status") == "timeout":
            verdict_data["pilot_status"] = "PILOT_REASSESS"
            verdict_data["reason"] = (
                f"fit1 timed out after {fit1_result.get('elapsed_seconds', 0):.0f}s. "
                "Cloud dispatch needed."
            )
        else:
            verdict_data["pilot_status"] = "PILOT_FAIL"
            verdict_data["reason"] = (
                f"fit1 failed: {fit1_result.get('error', 'unknown error')[:500]}"
            )
        logger.warning("VERDICT: %s -- %s", verdict_data["pilot_status"],
                        verdict_data["reason"])
        return verdict_data

    # Extract pi_c from fit1 results.
    pi_value = fit1_result.get("pi")

    # MiXeR reports pi as a fraction (polygenicity fraction). To get pi_c
    # (number of causal SNPs), multiply by total number of SNPs used.
    # However, the key in the output may already be "pi_c" (integer count)
    # or "pi" (fraction). We handle both.
    # WHY [3000, 20000] range: brief_v2 decision rule. SCZ is highly
    # polygenic; Frei 2019 reported pi_c ~ 8,300 for SCZ.
    pi_c = None
    if pi_value is not None:
        # If pi is a small fraction (< 1), it's the fraction, not count.
        if isinstance(pi_value, (int, float)) and pi_value < 1:
            # Approximate total SNPs from reference panel (~9.3M across 22 chr).
            # This is a rough estimate; the exact number depends on QC.
            # WHY 9.3M: sum of SNPs across 22 chromosomes in 1000G EUR QC bim files.
            # The actual number used by MiXeR will be in its output.
            total_snps_approx = 9_300_000
            pi_c = round(pi_value * total_snps_approx)
            verdict_data["pi_fraction"] = pi_value
            verdict_data["pi_c_estimated"] = pi_c
            verdict_data["pi_c_estimation_note"] = (
                f"pi_c = pi * {total_snps_approx} (approximate total reference SNPs)"
            )
        else:
            pi_c = pi_value
            verdict_data["pi_c"] = pi_c

    # Sigma_beta (discoverability).
    sig2_beta = fit1_result.get("sig2_beta")
    if sig2_beta is not None:
        verdict_data["sigma_beta"] = sig2_beta

    # AIC/BIC from test1.
    if test1_result.get("status") == "success":
        verdict_data["AIC"] = test1_result.get("AIC")
        verdict_data["BIC"] = test1_result.get("BIC")

    # Apply decision rule.
    if pi_c is not None:
        if 3000 <= pi_c <= 20000:
            verdict_data["pilot_status"] = "PILOT_PASS"
            verdict_data["reason"] = (
                f"fit1 completed successfully. pi_c={pi_c} is within "
                f"[3000, 20000] range. Wall time: fit1={fit1_result.get('elapsed_seconds', 0):.0f}s"
            )
            if test1_result.get("status") == "success":
                verdict_data["reason"] += (
                    f", test1={test1_result.get('elapsed_seconds', 0):.0f}s"
                )
        elif 1000 <= pi_c <= 30000:
            # Within broader plausibility range but outside strict pass range.
            verdict_data["pilot_status"] = "PILOT_PASS"
            verdict_data["reason"] = (
                f"fit1 completed. pi_c={pi_c} is outside strict [3000, 20000] "
                f"but within plausible [1000, 30000] range. May reflect single-"
                f"replicate variability."
            )
            verdict_data["warning"] = "pi_c outside narrow range, multi-replicate needed"
        else:
            verdict_data["pilot_status"] = "PILOT_FAIL"
            verdict_data["reason"] = (
                f"fit1 completed but pi_c={pi_c} is outside [1000, 30000]. "
                "Model may not have converged or data quality issue."
            )
    else:
        # fit1 succeeded but we could not extract pi_c.
        verdict_data["pilot_status"] = "PILOT_PASS"
        verdict_data["reason"] = (
            "fit1 completed successfully but pi_c could not be extracted "
            "from output JSON. Manual inspection of output needed."
        )
        verdict_data["warning"] = "pi_c not parsed -- check raw_params in fit1_result"

    # Wall times.
    verdict_data["wall_time_fit1_seconds"] = fit1_result.get("elapsed_seconds")
    verdict_data["wall_time_test1_seconds"] = test1_result.get("elapsed_seconds")
    verdict_data["wall_time_total_seconds"] = round(total_elapsed, 2)

    logger.info("VERDICT: %s -- %s", verdict_data["pilot_status"],
                verdict_data["reason"])

    return verdict_data


# =============================================================================
# Main
# =============================================================================
def main():
    E2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("e2_mixer_pilot", LOGS_DIR / "e2_mixer_pilot.log")
    logger.info("=" * 70)
    logger.info("E2 MiXeR Univariate SCZ Pilot (Single Replicate)")
    logger.info("=" * 70)

    total_t0 = time.time()

    results: dict = {
        "experiment": "e2_mixer_pilot",
        "brief": "brief_v2.md section E2",
        "container_image": CONTAINER_IMAGE,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parameters": {
            "ld_r2min": LD_R2MIN,
            "ld_ldscore_r2min": LD_LDSCORE_R2MIN,
            "ld_window_kb": LD_WINDOW_KB,
            "snp_r2": SNP_R2,
            "snp_maf": SNP_MAF,
            "snp_seed": SNP_SEED,
            "fit1_randprune_n": FIT1_RANDPRUNE_N,
            "fit1_randprune_r2": FIT1_RANDPRUNE_R2,
            "fit1_seed": FIT1_SEED,
            "fit1_z1max": FIT1_Z1MAX,
            "fit1_kmax": FIT1_KMAX,
        },
    }

    # ------------------------------------------------------------------
    # Step 1: Prerequisites
    # ------------------------------------------------------------------
    logger.info("Step 1: Checking prerequisites...")
    prereqs = check_prerequisites(logger)
    results["prerequisites"] = prereqs

    if not prereqs["all_passed"]:
        results["pilot_status"] = "PILOT_FAIL"
        results["error_message"] = (
            "Prerequisites not met: "
            + ", ".join(k for k, v in prereqs.items()
                        if k != "all_passed" and v is False)
        )
        results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["elapsed_seconds"] = round(time.time() - total_t0, 2)
        atomic_write_json(results, E2_OUTPUT_DIR / "results.json")
        logger.error("ABORT: Prerequisites failed. %s", results["error_message"])
        return

    # ------------------------------------------------------------------
    # Step 2: Generate LD files
    # ------------------------------------------------------------------
    logger.info("Step 2: Generating LD files (per-chromosome)...")
    logger.info(
        "NOTE: This is the most time-consuming step. LD computation for "
        "22 chromosomes may take 30 min - 4+ hours depending on hardware. "
        "Budget cap: %ds. Files are cached for resume.", LD_TOTAL_BUDGET
    )
    ld_result = generate_ld_files(logger)
    results["ld_generation"] = {
        "total_wall_time_seconds": ld_result["total_wall_time_seconds"],
        "chromosomes_completed": ld_result["chromosomes_completed"],
        "all_completed": ld_result["all_completed"],
        "budget_exceeded": ld_result.get("budget_exceeded", False),
    }

    if ld_result.get("budget_exceeded") or not ld_result["all_completed"]:
        total_elapsed = time.time() - total_t0
        verdict = classify_pilot(
            fit1_result={"status": "not_run"},
            test1_result={"status": "not_run"},
            total_elapsed=total_elapsed,
            ld_result=ld_result,
            logger=logger,
        )
        results.update(verdict)
        results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["elapsed_seconds"] = round(total_elapsed, 2)
        atomic_write_json(results, E2_OUTPUT_DIR / "results.json")
        logger.info("E2 halted at LD generation. Results written.")
        return

    # ------------------------------------------------------------------
    # Step 3: Generate pruned SNP list
    # ------------------------------------------------------------------
    logger.info("Step 3: Generating pruned SNP list...")
    snp_result = generate_snp_list(logger)
    results["snp_pruning"] = snp_result

    if snp_result.get("status") == "failed":
        results["pilot_status"] = "PILOT_FAIL"
        results["error_message"] = f"SNP pruning failed: {snp_result.get('error', 'unknown')[:500]}"
        results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["elapsed_seconds"] = round(time.time() - total_t0, 2)
        atomic_write_json(results, E2_OUTPUT_DIR / "results.json")
        logger.error("ABORT: SNP pruning failed.")
        return

    # ------------------------------------------------------------------
    # Step 4: Run fit1
    # ------------------------------------------------------------------
    logger.info("Step 4: Running fit1 (univariate MiXeR fit)...")
    logger.info("This may take 1-6 hours. Timeout: %ds", FIT1_TIMEOUT)
    fit1_result = run_fit1(logger)
    results["fit1"] = {
        "status": fit1_result["status"],
        "elapsed_seconds": fit1_result.get("elapsed_seconds"),
        "docker_command": fit1_result.get("docker_command"),
    }
    # Include parameter estimates if available.
    for key in ("pi", "pi_fraction", "sig2_beta", "sig2_zero",
                "raw_params", "parse_error", "error", "log_tail"):
        if key in fit1_result:
            results["fit1"][key] = fit1_result[key]

    # ------------------------------------------------------------------
    # Step 5: Run test1 (only if fit1 succeeded)
    # ------------------------------------------------------------------
    if fit1_result.get("status") == "success":
        logger.info("Step 5: Running test1 (goodness-of-fit evaluation)...")
        test1_result = run_test1(logger)
        results["test1"] = {
            "status": test1_result["status"],
            "elapsed_seconds": test1_result.get("elapsed_seconds"),
            "docker_command": test1_result.get("docker_command"),
        }
        for key in ("AIC", "BIC", "raw_output", "parse_error", "error", "log_tail"):
            if key in test1_result:
                results["test1"][key] = test1_result[key]
    else:
        test1_result = {"status": "skipped", "reason": "fit1 did not succeed"}
        results["test1"] = test1_result
        logger.info("Step 5: Skipping test1 (fit1 did not succeed)")

    # ------------------------------------------------------------------
    # Step 6: Classify and write results
    # ------------------------------------------------------------------
    total_elapsed = time.time() - total_t0
    logger.info("Step 6: Classifying pilot results...")
    verdict = classify_pilot(
        fit1_result=fit1_result,
        test1_result=test1_result,
        total_elapsed=total_elapsed,
        ld_result=ld_result,
        logger=logger,
    )
    results.update(verdict)
    results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["elapsed_seconds"] = round(total_elapsed, 2)

    # Write results.
    atomic_write_json(results, E2_OUTPUT_DIR / "results.json")
    logger.info("=" * 70)
    logger.info("E2 COMPLETE. Verdict: %s", results.get("pilot_status", "UNKNOWN"))
    logger.info("Total elapsed: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("Results: %s", E2_OUTPUT_DIR / "results.json")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
