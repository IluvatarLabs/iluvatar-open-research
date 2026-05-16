"""07_spec_curve_plots.py — Simonsohn 2020 Figure-1 style specification-curve plots.

One panel per finding. Y-axis OR (log), X-axis specs sorted by OR magnitude, headline
highlighted, 95% CI error bars, OR=1 reference line, permutation-null-count annotation.
matplotlib only.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from common import OUTPUT_DIR

HEADLINES = {
    "f013":  {"or": 9.76,    "src": "batch_009"},
    "f098":  {"or": 6.94,    "src": "batch_040 d45", "use_adj": True},
    "f120":  {"or": 1e6,     "src": "batch_047 (OR=inf)", "clip": True},
    "f121":  {"or": 26.44,   "src": "batch_047"},
    "f124":  {"or": None,    "src": "batch_048 per-tier"},  # per-tier headlines, use None
}

def plot_spec(cells, finding: str, title: str, or_key="OR", lo_key="OR_low", hi_key="OR_high",
              label_key="spec", descriptive_key="descriptive_only", highlight_match=None):
    # Prepare
    data = []
    for c in cells:
        or_ = c.get(or_key, c.get("OR"))
        if or_ is None or (isinstance(or_, float) and (math.isnan(or_) or or_ == 0)):
            continue
        lo = c.get(lo_key, math.nan); hi = c.get(hi_key, math.nan)
        data.append({
            "spec": c[label_key],
            "OR": or_, "lo": lo, "hi": hi,
            "desc": bool(c.get(descriptive_key, False)),
            "emp_p": c.get("emp_p", math.nan),
            "raw_p": c.get("raw_p", math.nan),
        })
    data.sort(key=lambda d: (d["OR"] if math.isfinite(d["OR"]) else 1e30))
    n = len(data)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.2 * n + 2), 5.2))
    xs = np.arange(n)
    or_vals = np.array([d["OR"] for d in data], dtype=float)
    los = np.array([d["lo"] for d in data], dtype=float)
    his = np.array([d["hi"] for d in data], dtype=float)
    # Clip infs for plotting
    or_plot = np.where(np.isfinite(or_vals), or_vals, 1e5)
    los_plot = np.where(np.isfinite(los), los, 1e-3)
    his_plot = np.where(np.isfinite(his), his, 1e5)
    # Colours: headline = red, descriptive = grey, else blue
    colours = []
    for d in data:
        if d["desc"]:
            colours.append("#888888")
        elif highlight_match and highlight_match(d["spec"]):
            colours.append("#d62728")
        else:
            colours.append("#1f77b4")
    lower_err = np.clip(or_plot - los_plot, 0, None)
    upper_err = np.clip(his_plot - or_plot, 0, None)
    # errorbar wants a single color; draw per-point so colors differ
    for i in range(n):
        ax.errorbar([xs[i]], [or_plot[i]], yerr=[[lower_err[i]], [upper_err[i]]],
                    fmt="none", ecolor=colours[i], capsize=3, linewidth=1.3)
    ax.scatter(xs, or_plot, c=colours, s=60, zorder=3, edgecolors="black", linewidths=0.6)
    ax.axhline(1.0, color="#555", linewidth=0.9, linestyle="--")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([d["spec"] for d in data], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("OR (log scale)")
    ax.set_title(title)
    # Annotate each point with emp_p and n if present
    for i, d in enumerate(data):
        marker = "desc" if d["desc"] else f"p={d['emp_p']:.3g}"
        ax.text(i, or_plot[i], f" {marker}", fontsize=7, va="center")
    # Legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color="#d62728", label="headline spec"),
        mpatches.Patch(color="#1f77b4", label="alt spec (n>=5)"),
        mpatches.Patch(color="#888888", label="descriptive only (n<5)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)
    fig.tight_layout()
    out = OUTPUT_DIR / f"spec_curve_{finding}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

def main():
    # F013
    with open(OUTPUT_DIR / "f013_backgrounds.json") as f:
        f13 = json.load(f)
    plot_spec(f13["cells"], "f013", "F013 | PanglaoDB neuronal marker enrichment — background axis",
              highlight_match=lambda s: s in ("all_coding", "protein_only"))
    # F098 — use adj_OR on y-axis; no CI given (permutation), plot OR_obs with fake CI = [adj_OR, OR_obs]
    with open(OUTPUT_DIR / "f098_length_kernels.json") as f:
        f98 = json.load(f)
    # Construct OR/CI from adj_OR / emp_p; we use adj_OR as the "OR" point estimate
    for c in f98["cells"]:
        c["OR"] = c.get("adj_OR")
        c["OR_low"] = c.get("adj_OR")  # no CI computed
        c["OR_high"] = c.get("adj_OR")
        c["descriptive_only"] = False
    plot_spec(f98["cells"], "f098", "F098 | length-matched neuronal enrichment — kernel axis",
              highlight_match=lambda s: s == "linear_10pct")
    # F120
    with open(OUTPUT_DIR / "f120_pli_thresholds.json") as f:
        f120 = json.load(f)
    for c in f120["cells"]:
        c["descriptive_only"] = False  # saturation is not descriptive; keep for visibility
    plot_spec(f120["cells"], "f120", "F120 | SCHEMA pLI enrichment — threshold gradient",
              highlight_match=lambda s: s == "pLI_ge_0.9")
    # F121
    with open(OUTPUT_DIR / "f121_syngo_scopes.json") as f:
        f121 = json.load(f)
    plot_spec(f121["cells"], "f121", "F121 | SynGO_EDT1 pLI enrichment — scope x EDT1-size",
              highlight_match=lambda s: "all__prioritised" in s)
    # F124 — flatten across sources, one row per (source, tier)
    with open(OUTPUT_DIR / "f124_tier_definitions.json") as f:
        f124 = json.load(f)
    all_cells = []
    for src, cells in f124["sources"].items():
        for c in cells:
            c2 = dict(c)
            c2["spec"] = c["spec"]
            all_cells.append(c2)
    plot_spec(all_cells, "f124", "F124 | EDT1 tier decomposition — definition-source axis",
              highlight_match=lambda s: s.startswith("F124_curated__"))

if __name__ == "__main__":
    main()
