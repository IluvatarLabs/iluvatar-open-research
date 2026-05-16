#!/usr/bin/env python3
"""batch_056 master orchestrator.

Runs any subset of {sub_a, sub_b, sub_c, sub_d} and aggregates outputs into
experiments/batch_056/output/results.json with a decision matrix row for the
overall outcome.

Usage:
    python3 run_batch_056.py --sub a          # just Sub-A
    python3 run_batch_056.py --sub all        # all four
    python3 run_batch_056.py --sub c --skip-pops   # sub-C analysis only

WHY separate sub-scripts exist as standalone entry points: each sub-experiment
is independently auditable (Gate 2 per marvin.md). The orchestrator only
composes them; it does NOT recompute anything the individual scripts produce.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    LOGS_DIR,
    OUTPUT_DIR,
    atomic_write_json,
)

import sub_a_pops_pli as sub_a
import sub_b_ldmask as sub_b
import sub_c_pgrid as sub_c
import sub_d_covariate_adjusted as sub_d


ALL_SUBS = ("a", "b", "c", "d")


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_056.run")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                datefmt="%Y-%m-%dT%H:%M:%S")
        for h in (logging.FileHandler(LOGS_DIR / "run_batch_056.log"),
                  logging.StreamHandler(sys.stdout)):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def _load_sub_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "not_run", "path": str(path)}
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed_read", "error": str(exc), "path": str(path)}


def build_master_decision(agg: dict) -> dict:
    """Compose the 4-cell outcome matrix row per brief_v2 IF-WRONG table.

    Returns {sub_a, sub_b, sub_c, sub_d} classifications + a manuscript-
    action row pointer (the brief pre-registers specific actions per cell).
    """
    def _class(sub_key: str) -> str:
        sub = agg.get(sub_key, {})
        if not isinstance(sub, dict):
            return "NOT_RUN"
        status = sub.get("status", "not_run")
        if status != "ok":
            return f"NOT_OK({status})"
        c = sub.get("decision_classification", {})
        return c.get("classification", "Unknown")

    c_a = _class("sub_a")
    c_b = _class("sub_b")
    c_c = _class("sub_c")
    c_d = _class("sub_d")

    # Brief_v2 IF-WRONG 4-cell matrix is Sub-A × Sub-D. Sub-B and Sub-C are
    # modifiers (Insert G hedges).
    scenario_key = f"A={c_a} / D={c_d}"

    return {
        "sub_a": c_a,
        "sub_b": c_b,
        "sub_c": c_c,
        "sub_d": c_d,
        "scenario_key_A_x_D": scenario_key,
        "manuscript_action_anchor": (
            "See brief_v2.md §Sub-D IF WRONG table and §Sub-B IF WRONG for "
            "pre-registered text substitutions per cell."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="batch_056 orchestrator")
    parser.add_argument("--sub", choices=list(ALL_SUBS) + ["all"],
                        default="all",
                        help="which sub-experiment to run")
    parser.add_argument("--skip-gate", action="store_true",
                        help="pass --skip-gate to sub-scripts (DANGEROUS)")
    parser.add_argument("--skip-pops", action="store_true",
                        help="Sub-C only: skip PoPS runs; use cached preds")
    parser.add_argument("--force", action="store_true",
                        help="Sub-C only: force PoPS re-run")
    parser.add_argument("--strict-r3", action="store_true",
                        help="Sub-D: hard-fail on R3 disagreement")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="don't run subs; only re-aggregate existing "
                             "sub-results into master results.json")
    args = parser.parse_args()

    logger = setup_logger()
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subs_to_run = list(ALL_SUBS) if args.sub == "all" else [args.sub]
    logger.info("Orchestrator will run: %s", subs_to_run)

    rcs: dict[str, int | None] = {s: None for s in ALL_SUBS}

    if not args.aggregate_only:
        for s in subs_to_run:
            logger.info("===== STARTING SUB-%s =====", s.upper())
            # WHY we pass argv manually: each sub-script uses its own
            # argparse. We slice the orchestrator's flags that apply.
            old_argv = sys.argv
            try:
                if s == "a":
                    sys.argv = ["sub_a_pops_pli.py"]
                    if args.skip_gate:
                        sys.argv.append("--skip-gate")
                    rcs["a"] = sub_a.main()
                elif s == "b":
                    sys.argv = ["sub_b_ldmask.py"]
                    if args.skip_gate:
                        sys.argv.append("--skip-gate")
                    rcs["b"] = sub_b.main()
                elif s == "c":
                    sys.argv = ["sub_c_pgrid.py"]
                    if args.skip_gate:
                        sys.argv.append("--skip-gate")
                    if args.skip_pops:
                        sys.argv.append("--skip-pops")
                    if args.force:
                        sys.argv.append("--force")
                    rcs["c"] = sub_c.main()
                elif s == "d":
                    sys.argv = ["sub_d_covariate_adjusted.py"]
                    if args.strict_r3:
                        sys.argv.append("--strict-r3")
                    rcs["d"] = sub_d.main()
            except SystemExit as se:
                rcs[s] = int(se.code) if se.code is not None else 0
            except Exception as exc:  # noqa: BLE001
                logger.exception("Sub-%s raised unhandled exception", s)
                rcs[s] = -1
                # Write a failure stub so aggregation still happens.
                (OUTPUT_DIR / f"sub_{s}" / "results.json").parent.mkdir(
                    parents=True, exist_ok=True)
                atomic_write_json(
                    {"status": "failed", "phase": "orchestrator",
                     "error": str(exc)},
                    OUTPUT_DIR / f"sub_{s}" / "results.json",
                )
            finally:
                sys.argv = old_argv
            logger.info("===== FINISHED SUB-%s rc=%s =====", s.upper(), rcs[s])

    # --------------------- Aggregate ---------------------
    agg = {
        "batch": "056",
        "iteration": 56,
        "date": "2026-04-23",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime()),
        "wall_s_total": time.time() - t0,
        "sub_rcs": rcs,
        "sub_a": _load_sub_json(OUTPUT_DIR / "sub_a" / "results.json"),
        "sub_b": _load_sub_json(OUTPUT_DIR / "sub_b" / "results.json"),
        "sub_c": _load_sub_json(OUTPUT_DIR / "sub_c" / "results.json"),
        "sub_d": _load_sub_json(OUTPUT_DIR / "sub_d" / "results.json"),
    }

    # Reproduction gates roll-up.
    gates = {
        "R1_anchor_p05_spearman": (
            agg["sub_a"].get("reproduction_gate_R1")
            or agg["sub_b"].get("reproduction_gate_R1")
        ),
        "R2_anchor_p07_spearman": agg["sub_c"].get("reproduction_gate_R2"),
        "R3_adhd_median_b3": agg["sub_d"].get("reproduction_gate_R3"),
    }
    agg["reproduction_gates"] = gates

    # Decision matrix row.
    agg["decision_matrix_row"] = build_master_decision(agg)
    logger.info("Master decision: %s", agg["decision_matrix_row"])

    atomic_write_json(agg, OUTPUT_DIR / "results.json")
    logger.info("Wrote %s", OUTPUT_DIR / "results.json")

    # Non-zero exit if any sub failed.
    if any(rc not in (0, None) for rc in rcs.values()):
        logger.error("At least one sub-experiment failed: %s", rcs)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
