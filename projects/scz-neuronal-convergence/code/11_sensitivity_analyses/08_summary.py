"""08_summary.py — produce specification_table.tsv, jaccard_matrices.json, results.json.

Decision rule per finding (W1 revised):
  >= K-1 specs same direction (OR > 1) AND |effect| >= 50% of headline OR for those specs.
  n<5 cells are not counted toward significance (W4).
"""
from __future__ import annotations
import json, sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from common import OUTPUT_DIR, LOGS_DIR, jaccard, bh_qvalues, log_event

def load(name): return json.loads((OUTPUT_DIR / name).read_text())

def headline_or(finding):
    return {"F013": 9.76, "F098": 6.94, "F120": float("inf"), "F121": 26.44}.get(finding)

def decide(finding: str, cells: list[dict]) -> dict:
    """Apply W1 revised rule."""
    keep = [c for c in cells if not c.get("descriptive_only", False)]
    K = len(keep)
    if K == 0:
        return {"K": 0, "verdict": "UNINTERPRETABLE", "reason": "all cells descriptive (n<5)"}
    same_direction = sum(1 for c in keep if c.get("OR", math.nan) > 1)
    hl = headline_or(finding)
    # F124 headline is per-tier so we apply the rule on "tier-wise" comparison in summary_f124.
    if finding == "F124" or hl is None:
        return {"K": K, "same_direction": same_direction,
                "pct_same": same_direction / K if K else 0.0,
                "verdict": "see_per_tier_summary"}
    # For saturated headline (OR=inf), use "OR>=5" as a practical threshold proxy on each spec
    # to test "strong-effect consistency". Simonsohn 2020 does not mandate exact-OR thresholds;
    # we adopt |OR| >= 50% of a reference = max(OR across non-infinite specs) when headline is inf.
    if not math.isfinite(hl):
        finite_ors = [c.get("OR", 0) for c in keep if math.isfinite(c.get("OR", 0))]
        ref = max(finite_ors) if finite_ors else 0
        thresh = 0.5 * ref if ref else 0.0
        ref_used = f"max_finite_OR={ref:.3g}"
    else:
        thresh = 0.5 * hl
        ref_used = f"headline_OR={hl:.3g}"
    strong = sum(1 for c in keep if math.isfinite(c.get("OR", 0)) and c["OR"] >= thresh)
    saturated = sum(1 for c in keep if not math.isfinite(c.get("OR", 0)))
    # For F120 saturation: treat inf OR as passing the 50% threshold (inf > any threshold).
    strong_incl_sat = strong + saturated
    # W1 rule: K-1 specs same direction AND strong effect
    rule_pass = (same_direction >= K - 1) and (strong_incl_sat >= K - 1)
    return {
        "K": K, "same_direction": same_direction,
        "strong_effect_count": strong_incl_sat, "threshold_for_strong": thresh,
        "reference_used": ref_used,
        "verdict": "PASS" if rule_pass else "FAIL",
    }

def main():
    log = LOGS_DIR / "run.log"; t0 = time.time()
    f13 = load("f013_backgrounds.json")
    f98 = load("f098_length_kernels.json")
    f120 = load("f120_pli_thresholds.json")
    f121 = load("f121_syngo_scopes.json")
    f124 = load("f124_tier_definitions.json")

    rows = []
    # Long-format table
    def row(finding, c, n_num_key="a", n_denom_key=None, or_key="OR",
            or_lo_key="OR_low", or_hi_key="OR_high"):
        or_ = c.get(or_key, c.get("OR"))
        if finding == "F098":
            or_ = c.get("adj_OR")
            n_num = c.get("a_obs")
            n_denom = c.get("n_scz_in_bg")
        else:
            n_num = c.get(n_num_key)
            if n_denom_key is None:
                n_denom = c.get("n_bg", c.get("n_scz_in_bg") or c.get("n_in_bg"))
            else:
                n_denom = c.get(n_denom_key)
        return {
            "finding": finding,
            "spec": c.get("spec"),
            "n_num": n_num,
            "n_denom": n_denom,
            "OR": or_,
            "OR_low": c.get(or_lo_key),
            "OR_high": c.get(or_hi_key),
            "raw_p": c.get("raw_p"),
            "emp_p": c.get("emp_p"),
            "descriptive_only": bool(c.get("descriptive_only", False)),
        }

    # F013
    for c in f13["cells"]:
        rows.append(row("F013", c))
    for c in f98["cells"]:
        rows.append(row("F098", c))
    for c in f120["cells"]:
        rows.append(row("F120", c))
    for c in f121["cells"]:
        rows.append(row("F121", c))
    for src, cells in f124["sources"].items():
        for c in cells:
            rr = row("F124", c, n_denom_key="n_in_bg")
            rr["source"] = src
            rows.append(rr)

    df = pd.DataFrame(rows)
    # BH q per finding
    df["BH_q"] = math.nan
    for fnd, sub in df.groupby("finding"):
        p = sub["raw_p"].astype(float).fillna(1.0).tolist()
        q = bh_qvalues(p)
        df.loc[sub.index, "BH_q"] = q
    tsv_path = OUTPUT_DIR / "specification_table.tsv"
    df.to_csv(tsv_path, sep="\t", index=False, float_format="%.6g")
    print(f"wrote {tsv_path} ({len(df)} rows)")

    # Jaccard matrices: per-axis pairwise Jaccard of gene sets (C1 + W3)
    jacc = {}
    # F013 backgrounds: include the 5 backgrounds and SCZ∩bg (numerator) for W3
    jacc["F013_background_universe"] = {}
    bg_sets = {c["spec"]: set([]) for c in f13["cells"]}  # placeholder; bg genes not saved in json
    # Use numerator_genes (SCZ ∩ markers ∩ bg) as a proxy — that's the W3 numerator Jaccard
    num_sets = {c["spec"]: set(c.get("numerator_genes", [])) for c in f13["cells"]}
    jacc["F013_numerator_jaccard_W3"] = {f"{a}|{b}": jaccard(num_sets[a], num_sets[b])
                                          for i, a in enumerate(num_sets)
                                          for b in list(num_sets)[i + 1:]}
    # F098: numerator = neuronal ∩ scz is common across kernels (same SCZ, same markers).
    jacc["F098_numerator_jaccard_W3_note"] = "SCZ list and markers identical; numerator invariant across kernels. Differences are in null sampling, not numerator."
    # F120: numerator set = SCHEMA ∩ high across thresholds
    # (gene lists not saved explicitly — use count-based "self-Jaccard" from a/b cells in results text)
    # F121: numerator Jaccard on (SynGO∩EDT1∩pLI≥0.9) across specs
    num_121 = {c["spec"]: set(c.get("numerator_genes", [])) for c in f121["cells"]}
    jacc["F121_numerator_jaccard_W3"] = {f"{a}|{b}": jaccard(num_121[a], num_121[b])
                                          for i, a in enumerate(num_121)
                                          for b in list(num_121)[i + 1:]}
    jacc["F121_scope_jaccard_C1"] = f121["scope_jaccard"]
    jacc["F121_scopes_dropped"] = f121["scopes_dropped"]
    jacc["F124_tier_jaccard"] = f124["tier_jaccard"]
    (OUTPUT_DIR / "jaccard_matrices.json").write_text(json.dumps(jacc, indent=2))
    print(f"wrote jaccard_matrices.json")

    # Decision rules
    decisions = {
        "F013": decide("F013", f13["cells"]),
        "F098": decide_f098(f98["cells"]),
        "F120": decide("F120", f120["cells"]),
        "F121": decide("F121", f121["cells"]),
        "F124": decide_f124(f124["sources"], f124["headline_OR_by_tier"]),
    }

    results = {
        "batch": "batch_052_B",
        "generated_at": int(time.time()),
        "decision_rule": "W1 revised: >= K-1 specs same direction AND |effect| >= 50% of headline",
        "decisions": decisions,
        "f013": f13, "f098": f98, "f120": f120, "f121": f121, "f124": f124,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    elapsed = time.time() - t0
    log_event(log, f"[08_summary] elapsed={elapsed:.2f}s decisions={decisions}")
    print(json.dumps({"decisions": decisions}, indent=2, default=str))

def decide_f098(cells):
    """F098 uses adj_OR as effect."""
    K = len(cells)
    same_direction = sum(1 for c in cells if (c.get("adj_OR") or 0) > 1)
    hl = 6.94
    strong = sum(1 for c in cells if (c.get("adj_OR") or 0) >= 0.5 * hl)
    return {"K": K, "same_direction": same_direction, "strong_effect_count": strong,
            "threshold_for_strong": 0.5 * hl, "reference_used": f"headline_adj_OR={hl:.3g}",
            "verdict": "PASS" if (same_direction >= K - 1 and strong >= K - 1) else "FAIL"}

def decide_f124(sources, headlines):
    """Per-tier decision across sources: tier X is robust if K-1 sources same direction & OR >= 50% headline."""
    results = {}
    source_names = list(sources)
    # Invert: tier -> list of (source, OR, desc)
    per_tier = {}
    for s, cells in sources.items():
        for c in cells:
            per_tier.setdefault(c["tier"], []).append((s, c.get("OR"), c.get("descriptive_only", False)))
    for tier, lst in per_tier.items():
        keep = [x for x in lst if not x[2]]  # exclude descriptive
        K = len(keep)
        hl = headlines.get(tier)
        same_dir = sum(1 for _, or_, _ in keep if (or_ or 0) > 1)
        strong = sum(1 for _, or_, _ in keep if (or_ or 0) >= 0.5 * hl) if hl else 0
        verdict = "PASS" if (K and same_dir >= K - 1 and (hl is None or strong >= K - 1)) else "FAIL"
        results[tier] = {"K_eligible": K, "K_total": len(lst), "same_direction": same_dir,
                          "strong_count": strong, "headline_OR": hl, "verdict": verdict}
    return results

if __name__ == "__main__":
    main()
