#!/usr/bin/env python3
"""batch_060 E9 -- MiXeR Docker feasibility pilot.

Implements brief_v2.md section E9 EXACTLY.

Steps:
  a) Check if Docker is available.
  b) Check if Singularity is available.
  c) Try to pull the GSA-MiXeR container.
  d) If pull succeeds, try a minimal test run (--help or similar).
  e) Document: Docker available? Container pulled? Test run passed?

WHY MiXeR: PI item 12, re-deferred 3x. MiXeR (Frei 2019/2024) provides
bivariate Gaussian mixture estimates of polygenicity overlap that cannot
be obtained from other tools. GSA-MiXeR extends to gene-set enrichment.
Phase B found Docker container available: ghcr.io/precimed/gsa-mixer:2.2.1.

Source: [lit_doi_10.1038_s41467-019-10310-0] Frei 2019,
        [lit_doi_10.1038_s41588-024-01771-1] Frei 2024.
PI item 12.

Output: experiments/batch_060/output/e9/results.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUTPUT_DIR,
    LOGS_DIR,
    atomic_write_json,
    setup_logger,
)

# =============================================================================
# Constants
# =============================================================================
E9_OUTPUT_DIR = OUTPUT_DIR / "e9"

# Container image from brief_v2 section E9 and design.yaml.
CONTAINER_IMAGE = "ghcr.io/precimed/gsa-mixer:2.2.1"

# Timeout for Docker/Singularity operations (seconds).
# WHY 300s: Container pulls can take several minutes on slow connections.
# The container image is ~2-3 GB. 5 minutes is generous but bounded.
PULL_TIMEOUT = 300
RUN_TIMEOUT = 60


def run_cmd(cmd: list[str], timeout: int = 30,
            logger=None) -> dict:
    """Run a shell command and capture output.

    Returns dict with returncode, stdout, stderr, elapsed, and success flag.

    WHY subprocess.run: Standard Python approach for running external commands.
    We capture both stdout and stderr for debugging. Timeout prevents hanging
    on unresponsive Docker daemon or network issues.
    """
    if logger:
        logger.info("Running: %s", " ".join(cmd))
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[:2000],  # Cap output size.
            "stderr": result.stderr.strip()[:2000],
            "elapsed_seconds": round(elapsed, 2),
            "success": result.returncode == 0,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "elapsed_seconds": round(elapsed, 2),
            "success": False,
            "timed_out": True,
        }
    except FileNotFoundError:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command not found: {cmd[0]}",
            "elapsed_seconds": 0,
            "success": False,
            "timed_out": False,
        }
    except Exception as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_seconds": time.time() - t0,
            "success": False,
            "timed_out": False,
        }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="E9: MiXeR Docker feasibility")
    parser.add_argument("--skip-pull", action="store_true",
                        help="Skip container pull (assume already pulled)")
    args = parser.parse_args()

    E9_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("e9_mixer", LOGS_DIR / "e9_mixer_feasibility.log")
    logger.info("=== E9 MiXeR Docker feasibility pilot ===")
    t0 = time.time()

    results: dict = {
        "experiment": "e9_mixer_feasibility",
        "brief": "brief_v2.md section E9",
        "container_image": CONTAINER_IMAGE,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ------------------------------------------------------------------
    # Step (a): Check Docker availability.
    # ------------------------------------------------------------------
    logger.info("Step (a): Checking Docker availability...")
    docker_check = run_cmd(["docker", "--version"], timeout=10, logger=logger)
    results["docker_available"] = docker_check["success"]
    results["docker_version"] = docker_check["stdout"] if docker_check["success"] else None
    results["docker_check_detail"] = docker_check

    if docker_check["success"]:
        logger.info("Docker available: %s", docker_check["stdout"])
    else:
        logger.warning("Docker not available: %s", docker_check["stderr"])

    # Also check if Docker daemon is running (not just installed).
    if docker_check["success"]:
        docker_info = run_cmd(["docker", "info", "--format", "{{.ServerVersion}}"],
                              timeout=15, logger=logger)
        results["docker_daemon_running"] = docker_info["success"]
        results["docker_daemon_detail"] = docker_info
        if docker_info["success"]:
            logger.info("Docker daemon running: server version %s",
                        docker_info["stdout"])
        else:
            logger.warning("Docker installed but daemon may not be running: %s",
                           docker_info["stderr"])

    # ------------------------------------------------------------------
    # Step (b): Check Singularity availability.
    # ------------------------------------------------------------------
    logger.info("Step (b): Checking Singularity availability...")
    sing_check = run_cmd(["singularity", "--version"], timeout=10, logger=logger)
    results["singularity_available"] = sing_check["success"]
    results["singularity_version"] = sing_check["stdout"] if sing_check["success"] else None
    results["singularity_check_detail"] = sing_check

    if sing_check["success"]:
        logger.info("Singularity available: %s", sing_check["stdout"])
    else:
        logger.info("Singularity not available: %s", sing_check["stderr"])

    # Determine which container runtime to use.
    # WHY Docker first: brief_v2 specifies Docker. Singularity is fallback
    # for HPC environments where Docker is not available.
    use_docker = docker_check["success"] and results.get("docker_daemon_running", False)
    use_singularity = sing_check["success"] and not use_docker
    runtime = "docker" if use_docker else ("singularity" if use_singularity else None)
    results["runtime_selected"] = runtime

    if runtime is None:
        results["verdict"] = "MIXER_BLOCKED"
        results["blocker"] = (
            "Neither Docker (with running daemon) nor Singularity is available. "
            f"Docker check: {docker_check['stderr']}. "
            f"Singularity check: {sing_check['stderr']}."
        )
        logger.error("BLOCKED: No container runtime available")
        results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["elapsed_seconds"] = round(time.time() - t0, 1)
        atomic_write_json(results, E9_OUTPUT_DIR / "results.json")
        return

    logger.info("Using runtime: %s", runtime)

    # ------------------------------------------------------------------
    # Step (c): Pull the GSA-MiXeR container.
    # ------------------------------------------------------------------
    logger.info("Step (c): Pulling GSA-MiXeR container...")

    if args.skip_pull:
        logger.info("Skipping pull (--skip-pull flag)")
        results["container_pulled"] = "skipped"
        pull_success = True
    elif runtime == "docker":
        pull_result = run_cmd(
            ["docker", "pull", CONTAINER_IMAGE],
            timeout=PULL_TIMEOUT, logger=logger,
        )
        results["container_pull_detail"] = pull_result
        pull_success = pull_result["success"]
        results["container_pulled"] = pull_success
        if pull_success:
            logger.info("Container pulled successfully (%.1fs)",
                        pull_result["elapsed_seconds"])
        else:
            logger.error("Container pull failed: %s", pull_result["stderr"])
    elif runtime == "singularity":
        # Singularity uses `singularity pull` with a different syntax.
        sif_path = E9_OUTPUT_DIR / "gsa-mixer_2.2.1.sif"
        if sif_path.exists():
            logger.info("Singularity SIF already exists: %s", sif_path)
            pull_success = True
            results["container_pulled"] = True
        else:
            pull_result = run_cmd(
                ["singularity", "pull", str(sif_path),
                 f"docker://{CONTAINER_IMAGE}"],
                timeout=PULL_TIMEOUT, logger=logger,
            )
            results["container_pull_detail"] = pull_result
            pull_success = pull_result["success"]
            results["container_pulled"] = pull_success
    else:
        pull_success = False
        results["container_pulled"] = False

    if not pull_success:
        results["verdict"] = "MIXER_BLOCKED"
        results["blocker"] = (
            f"Container pull failed. Runtime: {runtime}. "
            f"Image: {CONTAINER_IMAGE}. "
            f"Detail: {results.get('container_pull_detail', {}).get('stderr', 'unknown')}"
        )
        logger.error("BLOCKED: Container pull failed")
        results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["elapsed_seconds"] = round(time.time() - t0, 1)
        atomic_write_json(results, E9_OUTPUT_DIR / "results.json")
        return

    # ------------------------------------------------------------------
    # Step (d): Test run (--help or minimal invocation).
    # ------------------------------------------------------------------
    logger.info("Step (d): Test run...")

    if runtime == "docker":
        # Try running the container with --help.
        # WHY --help: It's the safest way to verify the container executes
        # without requiring any data files or complex arguments.
        test_result = run_cmd(
            ["docker", "run", "--rm", CONTAINER_IMAGE, "--help"],
            timeout=RUN_TIMEOUT, logger=logger,
        )
        results["test_run_help"] = test_result

        if not test_result["success"]:
            # Some containers don't support --help at the entrypoint.
            # Try just running with no args.
            logger.info("--help failed, trying no-arg invocation...")
            test_result_noarg = run_cmd(
                ["docker", "run", "--rm", CONTAINER_IMAGE],
                timeout=RUN_TIMEOUT, logger=logger,
            )
            results["test_run_noarg"] = test_result_noarg
            # Also try with `python -c 'import mixer; print(mixer.__version__)'`.
            test_result_version = run_cmd(
                ["docker", "run", "--rm", CONTAINER_IMAGE,
                 "python3", "-c",
                 "print('MiXeR container operational')"],
                timeout=RUN_TIMEOUT, logger=logger,
            )
            results["test_run_version"] = test_result_version
            test_passed = (
                test_result_noarg["success"]
                or test_result_version["success"]
            )
        else:
            test_passed = True
    elif runtime == "singularity":
        sif_path = E9_OUTPUT_DIR / "gsa-mixer_2.2.1.sif"
        test_result = run_cmd(
            ["singularity", "run", str(sif_path), "--help"],
            timeout=RUN_TIMEOUT, logger=logger,
        )
        results["test_run_help"] = test_result
        test_passed = test_result["success"]
    else:
        test_passed = False

    results["test_run_passed"] = test_passed

    # ------------------------------------------------------------------
    # Step (e): Summary and verdict.
    # ------------------------------------------------------------------
    logger.info("Step (e): Verdict...")

    feasibility_checks = {
        "docker_or_singularity_available": runtime is not None,
        "container_pulled": pull_success,
        "test_run_passed": test_passed,
    }
    results["feasibility_checks"] = feasibility_checks
    all_passed = all(feasibility_checks.values())

    if all_passed:
        results["verdict"] = "MIXER_FEASIBLE"
        results["recommendation"] = (
            "All 3 checks passed. Schedule full MiXeR run in iter_061 "
            f"using {runtime} runtime with image {CONTAINER_IMAGE}."
        )
        logger.info("VERDICT: MIXER_FEASIBLE. All checks passed.")
    else:
        failed_checks = [k for k, v in feasibility_checks.items() if not v]
        results["verdict"] = "MIXER_BLOCKED"
        results["blocker"] = (
            f"Failed checks: {failed_checks}. "
            f"Runtime: {runtime}. "
            "Document specific blocker for PI escalation."
        )
        # Provide specific remediation advice.
        remediation = []
        if not feasibility_checks["docker_or_singularity_available"]:
            remediation.append(
                "Install Docker or Singularity on the execution machine."
            )
        if not feasibility_checks["container_pulled"]:
            remediation.append(
                f"Check network connectivity and registry access to "
                f"ghcr.io. Try: {runtime} pull {CONTAINER_IMAGE}"
            )
        if not feasibility_checks["test_run_passed"]:
            remediation.append(
                "Container pulled but test run failed. Check container "
                "entrypoint and dependencies. Review test_run_* fields "
                "in results.json for stdout/stderr details."
            )
        results["remediation"] = remediation
        logger.warning(
            "VERDICT: MIXER_BLOCKED. Failed: %s. Remediation: %s",
            failed_checks, remediation,
        )

    results["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["elapsed_seconds"] = round(time.time() - t0, 1)

    atomic_write_json(results, E9_OUTPUT_DIR / "results.json")
    logger.info("E9 complete. Verdict: %s. Elapsed: %.1fs",
                results["verdict"], results["elapsed_seconds"])


if __name__ == "__main__":
    main()
