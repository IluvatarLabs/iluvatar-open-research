"""05_f121_syngo_scopes.py — F121 SynGO_EDT1 pLI enrichment across 4 SynGO scopes × 2 EDT1 sizes.

Anchor: F121 OR=26.44 (batch_047) for SynGO_EDT1 (n=14) pLI≥0.9.

Specs:
  SynGO scopes: all (BP∪CC), BP-only, CC-only, BP∪CC (identical to all — confirm via Jaccard, C1 revision).
  EDT1 sizes:
    prioritised = ST12 Prioritised==1 & protein_coding (n=106; preflight fallback, brief says ~261).
    broader = ST12 all gene_biotype=='protein_coding' (n=470, matches batch_048).
  Bg = gnomAD canonical+MANE ~17,485.

Jaccard-drop: if pairwise Jaccard(scope_genes) > 0.9 among {all, BP-only, CC-only, BP∪CC},
drop duplicate pairs from effective axis. Record pairs.

For each effective (scope × EDT1): intersect SynGO scope ∩ EDT1_in_bg → test pLI≥0.9 vs bg.
W4: cells with n<5 reported descriptively only, excluded from significance aggregation.
Seed 20260424.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import (OUTPUT_DIR, LOGS_DIR, RNG_SEED, N_PERM,
                    load_syngo, load_pgc3, load_gnomad, fisher_or_ci, jaccard, log_event)

def build_syngo_scopes() -> dict[str, set[str]]:
    terms = load_syngo()
    bp = [k for k in terms if k.rstrip().endswith(" BP")]
    cc = [k for k in terms if k.rstrip().endswith(" CC")]
    all_genes = set().union(*terms.values())
    bp_genes = set().union(*[terms[k] for k in bp])
    cc_genes = set().union(*[terms[k] for k in cc])
    return {"all": all_genes, "BP_only": bp_genes, "CC_only": cc_genes, "BP_union_CC": bp_genes | cc_genes}

def build_edt1_sizes(pgc3) -> dict[str, set[str]]:
    st12 = pgc3["st12"]
    pri = set(st12.loc[(st12["Prioritised"] == 1) & (st12["gene_biotype"] == "protein_coding"), "Symbol.ID"].astype(str))
    broader = set(st12.loc[st12["gene_biotype"] == "protein_coding", "Symbol.ID"].astype(str))
    return {"prioritised_n106": pri, "broader_n470": broader}

def pairwise_jaccard(d: dict[str, set]) -> dict[str, float]:
    names = list(d); out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}|{b}"] = jaccard(d[a], d[b])
    return out

def run_cell(scope_name: str, scope_genes: set[str], edt_name: str, edt_set: set[str],
             gnomad, rng) -> dict:
    bg_genes = set(gnomad["gene"].astype(str))
    pli_hi = set(gnomad.loc[gnomad["pLI"] >= 0.9, "gene"].astype(str))
    inter = scope_genes & edt_set & bg_genes
    # 2x2: within EDT (test set) vs outside-EDT within-bg (control), pLI≥0.9 yes/no
    a = len(inter & pli_hi)                     # in EDT∩scope, pLI hi
    b = len(inter - pli_hi)                     # in EDT∩scope, pLI low
    control = bg_genes - inter
    c = len(control & pli_hi)
    d = len(control - pli_hi)
    fish = fisher_or_ci(a, b, c, d)
    descriptive = (len(inter) < 5)
    # Permutation: draw random pseudo-"EDT∩scope" of size |inter| from bg and count pLI≥0.9
    bg_arr = np.array(sorted(bg_genes))
    hi_mask = np.isin(bg_arr, list(pli_hi))
    n_inter = len(inter); k_obs = a
    if n_inter == 0 or not hi_mask.any():
        emp_p = 1.0
    else:
        hits = np.empty(N_PERM, dtype=np.int32)
        for i in range(N_PERM):
            idx = rng.choice(len(bg_arr), size=n_inter, replace=False)
            hits[i] = hi_mask[idx].sum()
        emp_p = float((hits >= k_obs).sum() + 1) / (N_PERM + 1)
    return {
        "spec": f"{scope_name}__{edt_name}",
        "scope": scope_name, "edt": edt_name,
        "n_inter": n_inter, "descriptive_only": descriptive,
        "numerator_genes": sorted(inter & pli_hi),
        **fish, "emp_p": emp_p,
    }

def main():
    log = LOGS_DIR / "run.log"; t0 = time.time()
    scopes = build_syngo_scopes()
    pgc3 = load_pgc3(); edt1s = build_edt1_sizes(pgc3)
    gnomad = load_gnomad()
    # Jaccard matrix on full SynGO bg
    scope_jacc = pairwise_jaccard(scopes)
    print("SynGO scope pairwise Jaccard:", scope_jacc)
    # Drop duplicates (>0.9) — keep the first of the pair, remove second.
    drop = set()
    names = list(scopes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if scope_jacc[f"{a}|{b}"] > 0.9:
                drop.add(b)
    effective_scopes = {k: v for k, v in scopes.items() if k not in drop}
    print(f"Dropped scopes (Jaccard>0.9 dup): {sorted(drop)}")

    rng = np.random.default_rng(RNG_SEED)
    cells = []
    for sn, sg in effective_scopes.items():
        for en, eg in edt1s.items():
            c = run_cell(sn, sg, en, eg, gnomad, rng)
            cells.append(c)
            flag = "DESC" if c["descriptive_only"] else "OK  "
            print(f"F121 | {flag} {c['spec']:>40} n={c['n_inter']:3d} OR={c['OR']:.3g} raw_p={c['raw_p']:.3g} emp_p={c['emp_p']:.4f}")
    elapsed = time.time() - t0
    out = {"finding": "F121", "headline_OR": 26.44, "headline_source": "batch_047",
           "scope_jaccard": scope_jacc, "scopes_dropped": sorted(drop),
           "cells": cells, "elapsed_s": elapsed}
    (OUTPUT_DIR / "f121_syngo_scopes.json").write_text(json.dumps(out, indent=2, default=str))
    log_event(log, f"[05_f121] elapsed={elapsed:.2f}s cells={len(cells)}")

if __name__ == "__main__":
    main()
