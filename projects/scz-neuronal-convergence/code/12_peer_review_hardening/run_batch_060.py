#!/usr/bin/env python3
"""batch_060 driver -- run all sub-experiments sequentially.

Execution order per brief_v2.md:
  Track A (carry-over closures): E1 -> E2 -> E3
  Track B (environmental-axis): E4 -> E5 -> E6 -> E8
  Track C (PI items): E7 -> E9

WHY sequential (not parallel): experiments in the same track may have
data dependencies (E3 replays E2's null draws; E4-E6 share the same
battery framework and compete for memory).

E4-E9 are stubs awaiting implementation by other developers.

Usage:
  python3 experiments/batch_060/scripts/run_batch_060.py [--smoke]
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_script(name: str, script_path: Path, extra_args: list[str],
               label: str) -> int:
    """Run a sub-experiment script as a subprocess.

    WHY subprocess rather than import: each experiment manages its own
    sys.path, logging handlers, and argparse. Running as a subprocess
    provides clean isolation and prevents handler accumulation.
    """
    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"\n{'='*60}")
    print(f"[batch_060] Starting {label}: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"[batch_060] {label} {status} in {elapsed:.1f}s")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="batch_060 driver: run all sub-experiments"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Pass --smoke to all sub-experiments.")
    parser.add_argument("--track-a-only", action="store_true",
                        help="Run only Track A (E1, E2, E3).")
    args = parser.parse_args()

    smoke_args = ["--smoke"] if args.smoke else []
    t0_total = time.time()
    rc_all: dict[str, int] = {}

    # =========================================================================
    # Track A: Carry-over closures (E1 -> E2 -> E3)
    # =========================================================================
    rc_all["e1"] = run_script(
        "e1", SCRIPTS_DIR / "e1_bellenguez_rerun.py", smoke_args,
        "E1: Bellenguez AD rerun",
    )

    rc_all["e2"] = run_script(
        "e2", SCRIPTS_DIR / "e2_joint_ablation.py", smoke_args,
        "E2: Joint-ablation interaction (FINAL F-056-A test)",
    )

    rc_all["e3"] = run_script(
        "e3", SCRIPTS_DIR / "e3_pli_confound.py", smoke_args,
        "E3: pLI confound check",
    )

    if args.track_a_only:
        print(f"\n[batch_060] Track A complete in {time.time() - t0_total:.1f}s")
        print(f"[batch_060] Return codes: {rc_all}")
        return max(rc_all.values()) if rc_all else 0

    # =========================================================================
    # Track B: Environmental-axis gene-set batteries + G4 (E4 -> E5 -> E6 -> E8)
    # STUB: these scripts will be implemented by other developers.
    # =========================================================================
    for eid in ["e4", "e5", "e6", "e8"]:
        script = SCRIPTS_DIR / f"{eid}_stub.py"
        if script.exists():
            rc_all[eid] = run_script(eid, script, smoke_args,
                                      f"{eid.upper()} (Track B)")
        else:
            print(f"\n[batch_060] {eid.upper()} STUB: script not yet implemented, skipping.")
            rc_all[eid] = 0  # Non-blocking: stubs don't fail the batch.

    # =========================================================================
    # Track C: PI items (E7 -> E9)
    # STUB: these scripts will be implemented by other developers.
    # =========================================================================
    for eid in ["e7", "e9"]:
        script = SCRIPTS_DIR / f"{eid}_stub.py"
        if script.exists():
            rc_all[eid] = run_script(eid, script, smoke_args,
                                      f"{eid.upper()} (Track C)")
        else:
            print(f"\n[batch_060] {eid.upper()} STUB: script not yet implemented, skipping.")
            rc_all[eid] = 0

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed_total = time.time() - t0_total
    failed = {k: v for k, v in rc_all.items() if v != 0}
    print(f"\n{'='*60}")
    print(f"[batch_060] BATCH COMPLETE in {elapsed_total:.1f}s")
    print(f"[batch_060] Return codes: {rc_all}")
    if failed:
        print(f"[batch_060] FAILED experiments: {failed}")
    else:
        print("[batch_060] All experiments succeeded.")
    print(f"{'='*60}")
    return max(rc_all.values()) if rc_all else 0


if __name__ == "__main__":
    sys.exit(main())
