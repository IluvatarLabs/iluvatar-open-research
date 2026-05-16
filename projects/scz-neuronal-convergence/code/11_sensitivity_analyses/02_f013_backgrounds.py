"""02_f013_backgrounds.py — F013 (neuronal PanglaoDB marker enrichment) across 5 backgrounds.

Headline: F013 OR=9.76 (batch_009).

Backgrounds (W2 revision, preflight):
  1. all-coding = GenCode v44 protein-coding (n≈19,975) ∪ SCZ genes (union so SCZ not dropped).
  2. protein-only = gnomAD canonical+MANE (n≈17,485).
  3. brain-expressed = GTEx Brain TPM≥1 in ≥1 of 13 brain tissues.
  4. GTEx-any = GTEx TPM≥1 in ≥1 of 54 tissues.
  5. constrained = gnomAD pLI ≥ 0.9 (~3065).

Per cell: 2x2 Fisher (greater). Empirical p via 10,000 permutations shuffling SCZ-list membership
among bg-eligible-non-SCZ preserving |SCZ∩bg| and counting overlap with PanglaoDB neuronal markers.
Seed 20260424.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import (OUTPUT_DIR, LOGS_DIR, RNG_SEED, N_PERM,
                    load_scz_genes, load_neuronal_markers, load_gencode,
                    load_gnomad, load_gtex_brain_and_any, fisher_or_ci, log_event)

def run_spec(name: str, bg: set[str], scz: set[str], markers: set[str],
             rng: np.random.Generator) -> dict:
    scz_bg = scz & bg
    mk_bg = markers & bg
    a = len(scz_bg & mk_bg)
    b = len(scz_bg - mk_bg)
    c = len(mk_bg - scz_bg)
    d = len(bg - scz_bg - mk_bg)
    fish = fisher_or_ci(a, b, c, d)
    # Permutation: draw random "pseudo-SCZ" sets of |scz_bg| from bg and count overlaps with markers.
    bg_arr = np.array(sorted(bg))
    n_bg = len(bg_arr)
    mk_mask = np.isin(bg_arr, list(mk_bg))
    n_scz = len(scz_bg)
    k_obs = a
    if n_scz == 0 or not mk_mask.any() or n_bg == 0:
        emp_p = 1.0
    else:
        hits = np.empty(N_PERM, dtype=np.int32)
        for i in range(N_PERM):
            idx = rng.choice(n_bg, size=n_scz, replace=False)
            hits[i] = mk_mask[idx].sum()
        emp_p = float((hits >= k_obs).sum() + 1) / (N_PERM + 1)
    return {
        "spec": name,
        "n_bg": len(bg),
        "n_scz_in_bg": len(scz_bg),
        "n_markers_in_bg": len(mk_bg),
        "numerator_genes": sorted(scz_bg & mk_bg),
        **fish,
        "emp_p": emp_p,
    }

def main():
    log = LOGS_DIR / "run.log"; t0 = time.time()
    scz = load_scz_genes()
    markers = load_neuronal_markers()
    gencode = load_gencode()
    gnomad = load_gnomad()
    brain, any_t = load_gtex_brain_and_any()
    all_coding = set(gencode.keys()) | scz
    protein_only = set(gnomad["gene"]) | scz
    constrained = set(gnomad.loc[gnomad["pLI"] >= 0.9, "gene"]) | scz
    # brain/any: do NOT union SCZ — this background is supposed to be an independent denominator.
    # To avoid 0-intersections we keep SCZ as-is and report what fraction are in-bg.
    specs = {
        "all_coding":        all_coding,
        "protein_only":      protein_only,
        "brain_expressed":   brain,
        "gtex_any":          any_t,
        "constrained_pLI09": constrained,
    }
    rng = np.random.default_rng(RNG_SEED)
    cells = []
    for name, bg in specs.items():
        r = run_spec(name, bg, scz, markers, rng)
        cells.append(r)
        print(f"F013 | {name}: a={r['a']} OR={r['OR']:.3g} raw_p={r['raw_p']:.3g} emp_p={r['emp_p']:.4f}")
    elapsed = time.time() - t0
    out = {"finding": "F013", "headline_OR": 9.76, "headline_source": "batch_009",
           "cells": cells, "elapsed_s": elapsed}
    (OUTPUT_DIR / "f013_backgrounds.json").write_text(json.dumps(out, indent=2))
    log_event(log, f"[02_f013] elapsed={elapsed:.2f}s cells={len(cells)}")

if __name__ == "__main__":
    main()
