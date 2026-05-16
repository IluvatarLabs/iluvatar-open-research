"""04_f120_pli_thresholds.py — F120 SCHEMA (n=10) pLI threshold gradient + LOEUF orthogonal check.

Anchor: F120 OR=∞ at pLI≥0.9 (batch_047).

Specs (preflight):
  pLI gradient: 0.5, 0.7, 0.8, 0.9, 0.95.
  LOEUF < 0.35 (Karczewski 2020 preferred continuous metric).

Per cell: 2x2 Fisher (greater); Haldane-Anscombe +0.5 where any cell is 0.
10,000 permutations for empirical p (shuffle SCHEMA-membership label among gnomAD bg).
Report saturation (OR=∞ at high pLI where all 9 SCHEMA > threshold) descriptively.
Seed 20260424.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import (OUTPUT_DIR, LOGS_DIR, RNG_SEED, N_PERM,
                    load_schema, load_gnomad, fisher_or_ci, log_event)

def run_threshold(kind: str, thr: float, gnomad, schema_set: set[str], rng: np.random.Generator) -> dict:
    col = "pLI" if kind == "pLI_ge" else "LOEUF"
    if kind == "pLI_ge":
        high = gnomad[gnomad[col] >= thr]["gene"].astype(str)
    else:
        high = gnomad[gnomad[col] < thr]["gene"].astype(str)
    high_set = set(high)
    bg_genes = set(gnomad["gene"].astype(str))
    schema_in_bg = schema_set & bg_genes
    a = len(schema_in_bg & high_set)
    b = len(schema_in_bg - high_set)
    c = len(high_set - schema_in_bg)
    d = len(bg_genes - schema_in_bg - high_set)
    fish = fisher_or_ci(a, b, c, d)
    # Permutation — draw random "pseudo-SCHEMA" of size |schema_in_bg| from bg and count overlap with high_set
    bg_arr = np.array(sorted(bg_genes))
    n_bg = len(bg_arr)
    hi_mask = np.isin(bg_arr, list(high_set))
    n_s = len(schema_in_bg); k_obs = a
    if n_s == 0 or not hi_mask.any():
        emp_p = 1.0
    else:
        hits = np.empty(N_PERM, dtype=np.int32)
        for i in range(N_PERM):
            idx = rng.choice(n_bg, size=n_s, replace=False)
            hits[i] = hi_mask[idx].sum()
        emp_p = float((hits >= k_obs).sum() + 1) / (N_PERM + 1)
    saturated = (b == 0)  # all SCHEMA pass threshold
    return {"spec": f"{kind}_{thr}", "kind": kind, "threshold": thr,
            "n_schema_in_bg": n_s, "n_high_set": len(high_set), "n_bg": len(bg_genes),
            **fish, "emp_p": emp_p, "saturated": saturated}

def main():
    log = LOGS_DIR / "run.log"; t0 = time.time()
    schema = load_schema()
    gnomad = load_gnomad()
    print(f"F120 bg n={len(gnomad)} schema n={len(schema)} schema_in_bg={len(schema & set(gnomad['gene']))}")
    rng = np.random.default_rng(RNG_SEED)
    cells = []
    for thr in (0.5, 0.7, 0.8, 0.9, 0.95):
        r = run_threshold("pLI_ge", thr, gnomad, schema, rng)
        cells.append(r)
        print(f"F120 | pLI≥{thr}: a={r['a']}/b={r['b']} OR={r['OR']:.3g} raw_p={r['raw_p']:.3g} emp_p={r['emp_p']:.4f} sat={r['saturated']}")
    # LOEUF orthogonal
    r = run_threshold("LOEUF_lt", 0.35, gnomad, schema, rng)
    cells.append(r)
    print(f"F120 | LOEUF<0.35: a={r['a']}/b={r['b']} OR={r['OR']:.3g} raw_p={r['raw_p']:.3g} emp_p={r['emp_p']:.4f} sat={r['saturated']}")
    elapsed = time.time() - t0
    out = {"finding": "F120", "headline_OR": float("inf"), "headline_source": "batch_047",
           "cells": cells, "elapsed_s": elapsed}
    (OUTPUT_DIR / "f120_pli_thresholds.json").write_text(json.dumps(out, indent=2, default=str))
    log_event(log, f"[04_f120] elapsed={elapsed:.2f}s cells={len(cells)}")

if __name__ == "__main__":
    main()
