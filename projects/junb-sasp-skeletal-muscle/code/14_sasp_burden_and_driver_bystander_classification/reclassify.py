#!/usr/bin/env python3
"""Re-run classification only (without re-computing AUCell).

WHY: Classification logic had a sign inversion (delta_outside vs delta_inside).
Re-running the full pipeline would recompute AUCell unnecessarily. This script
reloads the existing classification CSV + null distributions, re-applies
classification, re-writes CSV/JSON.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_063")
CSV = ROOT / "b_delta_classification.csv"
NPZ = ROOT / "b_delta_null_distributions.npz"
SUM = ROOT / "b_delta_summary.json"
LOG = ROOT / "b_delta.log"

BONFERRONI_ALPHA = 0.05 / 60
FISHER_CI_DRIVER_THRESH = 0.5


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fisher_z_ci(rho, n, alpha=0.05):
    if not np.isfinite(rho) or n < 4:
        return (np.nan, np.nan)
    rho_c = np.clip(rho, -1 + 1e-15, 1 - 1e-15)
    z = np.arctanh(rho_c)
    se = 1.0 / np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def main():
    log("=== Re-classification only (classification logic fix: delta_inside for driver) ===")
    df = pd.read_csv(CSV)
    nulls = np.load(NPZ)
    null_dists = {k: nulls[k] for k in nulls.files}

    all_null = [np.abs(v) for v in null_dists.values() if len(v) > 0]
    pooled = np.concatenate(all_null) if all_null else np.array([])
    t_driver = float(np.percentile(pooled, 95)) if len(pooled) else np.nan
    t_bystander = float(np.percentile(pooled, 5)) if len(pooled) else np.nan
    log(f"Empirical thresholds: t_driver (95th pct of |rho|) = {t_driver:.4f}, "
        f"t_bystander (5th pct) = {t_bystander:.4f}")

    classifications = []
    underpowered_flags = []
    flag_notes = []

    for _, row in df.iterrows():
        n = int(row["n_donors"])
        clean12 = row.get("clean12_aucell_rho", np.nan)
        clean_sen = row.get("clean_senmayo_aucell_rho", np.nan)
        mrna = row.get("mrna_rho", np.nan)
        delta12 = row.get("delta_clean12", np.nan)
        p_age = row.get("age_shuffle_p_empirical", np.nan)
        sm_lo = row.get("size_matched_null_ci_low", np.nan)
        sm_hi = row.get("size_matched_null_ci_high", np.nan)
        fz_lo = row.get("fisher_z_ci_low", np.nan)
        fz_hi = row.get("fisher_z_ci_high", np.nan)
        m_fz_lo = row.get("mrna_fz_ci_low", np.nan)
        m_fz_hi = row.get("mrna_fz_ci_high", np.nan)

        reg_after_sasp = row.get("regulon_size_after_sasp12", 0)
        reg_after_sen = row.get("regulon_size_after_senmayo", 0)
        flags = []
        underpowered = False
        if reg_after_sasp < 20 or reg_after_sen < 20:
            underpowered = True
            flags.append(f"regulon_too_small(sasp={reg_after_sasp},sen={reg_after_sen})")
        if n <= 16 and np.isfinite(fz_lo) and np.isfinite(fz_hi):
            if (fz_lo <= t_driver <= fz_hi) or (fz_lo <= t_bystander <= fz_hi):
                underpowered = True
                flags.append("fisher_z_ci_crosses_threshold_at_low_N")

        classification = "inconclusive"

        driver_ok = all(np.isfinite(v) for v in [clean12, clean_sen, p_age, delta12, sm_lo, sm_hi])
        if driver_ok:
            sign_agree = (np.sign(clean12) == np.sign(clean_sen)) and clean12 != 0
            both_strong = (abs(clean12) >= t_driver) and (abs(clean_sen) >= t_driver)
            bonf_pass = p_age <= BONFERRONI_ALPHA
            delta_inside = (sm_lo <= delta12 <= sm_hi)
            fz_above_half = (
                np.isfinite(fz_lo) and np.isfinite(fz_hi) and
                ((clean12 > 0 and fz_lo >= FISHER_CI_DRIVER_THRESH) or
                 (clean12 < 0 and fz_hi <= -FISHER_CI_DRIVER_THRESH))
            )
            if sign_agree and both_strong and bonf_pass and delta_inside and fz_above_half:
                classification = "driver"

        if classification == "inconclusive" and np.isfinite(mrna) and np.isfinite(clean12):
            mrna_strong = abs(mrna) >= t_driver
            clean12_weak = abs(clean12) <= t_bystander
            if all(np.isfinite(v) for v in [fz_lo, fz_hi, m_fz_lo, m_fz_hi]):
                non_overlap = (fz_hi < m_fz_lo) or (m_fz_hi < fz_lo)
            else:
                non_overlap = False
            if mrna_strong and clean12_weak and non_overlap:
                classification = "bystander"

        if classification == "inconclusive" and np.isfinite(mrna) and np.isfinite(clean12):
            opp = (np.sign(mrna) != np.sign(clean12)) and mrna != 0 and clean12 != 0
            both_big = (abs(mrna) >= 0.3) and (abs(clean12) >= 0.3)
            clean_excl_0 = (
                np.isfinite(fz_lo) and np.isfinite(fz_hi) and
                (fz_lo > 0 or fz_hi < 0)
            )
            mrna_excl_0 = (
                np.isfinite(m_fz_lo) and np.isfinite(m_fz_hi) and
                (m_fz_lo > 0 or m_fz_hi < 0)
            )
            if opp and both_big and clean_excl_0 and mrna_excl_0:
                classification = "polarity-flip"

        if underpowered and classification == "inconclusive":
            classification = "UNDERPOWERED"

        classifications.append(classification)
        underpowered_flags.append(bool(underpowered))
        flag_notes.append("; ".join(flags))

    df["classification"] = classifications
    df["flag_underpowered"] = underpowered_flags
    df["flag_note"] = flag_notes
    df.to_csv(CSV, index=False)
    log(f"Wrote {CSV}")

    # Re-build summary.json
    rows = df.to_dict(orient="records")
    by_key = {(r["compartment"], r["tf"]): r for r in rows}

    def get_c(comp, tf):
        r = by_key.get((comp, tf))
        return r["classification"] if r else "missing"

    drivers = [r for r in rows if r["classification"] == "driver"]
    bystanders = [r for r in rows if r["classification"] == "bystander"]
    flips = [r for r in rows if r["classification"] == "polarity-flip"]
    inconc = [r for r in rows if r["classification"] == "inconclusive"]
    underp = [r for r in rows if r["classification"] == "UNDERPOWERED"]
    flagged = [r for r in rows if r["flag_underpowered"]]

    # Load the existing summary and merge updates
    prev = json.load(open(SUM))

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        return obj

    prev["empirical_thresholds"] = {
        "t_driver_from_age_shuffle_95pct": t_driver,
        "t_bystander_from_age_shuffle_5pct": t_bystander,
    }
    prev["counts"] = {
        "driver": len(drivers),
        "bystander": len(bystanders),
        "polarity_flip": len(flips),
        "inconclusive": len(inconc),
        "underpowered_label": len(underp),
        "flagged_underpowered_any": len(flagged),
    }
    prev["driver_TFs_by_compartment"] = {
        comp: [r["tf"] for r in drivers if r["compartment"] == comp]
        for comp in ["HLMA_Vascular", "HLMA_MuSC", "HLMA_FAP"]
    }
    prev["bystander_TFs_by_compartment"] = {
        comp: [r["tf"] for r in bystanders if r["compartment"] == comp]
        for comp in ["HLMA_Vascular", "HLMA_MuSC", "HLMA_FAP"]
    }
    prev["polarity_flip_TFs_by_compartment"] = {
        comp: [r["tf"] for r in flips if r["compartment"] == comp]
        for comp in ["HLMA_Vascular", "HLMA_MuSC", "HLMA_FAP"]
    }
    prev["pre_registered_predictions"] = {
        "JUNB_vasc_driver_after_senmayo": {
            "prediction": "JUNB retains driver class in Vascular after SenMayo removal",
            "Vascular_classification": get_c("HLMA_Vascular", "JUNB"),
            "Vascular_clean12_rho": by_key.get(("HLMA_Vascular","JUNB"),{}).get("clean12_aucell_rho"),
            "Vascular_clean_senmayo_rho": by_key.get(("HLMA_Vascular","JUNB"),{}).get("clean_senmayo_aucell_rho"),
            "MuSC_classification": get_c("HLMA_MuSC", "JUNB"),
            "FAP_classification": get_c("HLMA_FAP", "JUNB"),
        },
        "KLF10_bystander_all_3": {
            "prediction": "KLF10 classifies as bystander in ALL 3 compartments",
            "Vascular": get_c("HLMA_Vascular", "KLF10"),
            "MuSC": get_c("HLMA_MuSC", "KLF10"),
            "FAP": get_c("HLMA_FAP", "KLF10"),
        },
        "CEBPB_fap_driver": {
            "prediction": "CEBPB classifies as driver in FAP",
            "FAP": get_c("HLMA_FAP", "CEBPB"),
            "Vascular": get_c("HLMA_Vascular", "CEBPB"),
            "MuSC": get_c("HLMA_MuSC", "CEBPB"),
        },
        "ATF3_novel_call": {
            "prediction": "ATF3 driver status in any compartment is a NOVEL finding",
            "Vascular": get_c("HLMA_Vascular", "ATF3"),
            "MuSC": get_c("HLMA_MuSC", "ATF3"),
            "FAP": get_c("HLMA_FAP", "ATF3"),
        },
        "STAT3_novel_call": {
            "prediction": "STAT3 driver status in any compartment is a NOVEL finding",
            "Vascular": get_c("HLMA_Vascular", "STAT3"),
            "MuSC": get_c("HLMA_MuSC", "STAT3"),
            "FAP": get_c("HLMA_FAP", "STAT3"),
        },
    }

    # Update TF of interest block
    for tf in ["JUNB","KLF10","CEBPB","ATF3","STAT3"]:
        entries = {}
        for comp in ["HLMA_Vascular","HLMA_MuSC","HLMA_FAP"]:
            r = by_key.get((comp, tf))
            if r:
                entries[comp] = {
                    "regulon_size_raw": r["regulon_size_raw"],
                    "regulon_size_after_sasp12": r["regulon_size_after_sasp12"],
                    "regulon_size_after_senmayo": r["regulon_size_after_senmayo"],
                    "raw_aucell_rho": r["raw_aucell_rho"],
                    "clean12_aucell_rho": r["clean12_aucell_rho"],
                    "clean_senmayo_aucell_rho": r["clean_senmayo_aucell_rho"],
                    "mrna_rho": r["mrna_rho"],
                    "delta_clean12": r["delta_clean12"],
                    "size_matched_null_ci_low": r["size_matched_null_ci_low"],
                    "size_matched_null_ci_high": r["size_matched_null_ci_high"],
                    "age_shuffle_p_empirical": r["age_shuffle_p_empirical"],
                    "fisher_z_ci": [r["fisher_z_ci_low"], r["fisher_z_ci_high"]],
                    "classification": r["classification"],
                    "flag_underpowered": r["flag_underpowered"],
                    "flag_note": r["flag_note"],
                }
        prev.setdefault("tf_of_interest", {})[tf] = entries

    prev["classification_revision_note"] = (
        "2026-04-22: classification logic corrected. Brief's driver tree "
        "requires Δ INSIDE the size-matched null 95% CI (i.e., SASP12 removal "
        "indistinguishable from random 12-gene removal = balanced driver "
        "regulon, not a SASP-dominated leakage set). Prior run inverted this. "
        "AUCell and null-distribution numerics unchanged; only classification "
        "labels re-derived."
    )

    with open(SUM, "w") as f:
        json.dump(clean(prev), f, indent=2, default=str)
    log(f"Wrote {SUM}")
    log(f"Counts: driver={len(drivers)}, bystander={len(bystanders)}, "
        f"polarity-flip={len(flips)}, inconclusive={len(inconc)}, "
        f"UNDERPOWERED={len(underp)}, any_flagged={len(flagged)}")


if __name__ == "__main__":
    main()
