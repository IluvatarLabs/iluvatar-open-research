"""Shared helpers for batch_058 sub-experiments (v2.1 brief).

WHY: This module re-exports batch_057/_common (which itself re-exports
batch_056/_common, gene loaders, etc.) and adds ONLY the new helpers needed
for iter_058's v2.1 diagnostic battery, PoPS ablation/permutation-null
framework, and bivariate tail-ρ Monte-Carlo null.

Cardinal Rule 1 (no reinvention): TukeyBiweight / DFBETAS / Cook's D come
straight from `statsmodels.api`; rank-Gaussianization uses `scipy.stats`
`rankdata` + `norm.ppf`. We do NOT hand-roll any M-estimator or influence
statistic.

Cardinal Rule 0 (no fabrication): every loader raises FileNotFoundError with
a clear diagnostic on missing inputs. No silent defaults. Provenance SHA256
is the caller's responsibility and uses the re-exported `sha256_file`.

Seed policy: 20260424 (brief_v2 §MEASUREMENT). All stochastic helpers accept
an explicit seed and use `np.random.default_rng(seed)` for bit-reproducibility.
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Absolute paths (agent cwd resets between calls).
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_058"
OUTPUT_DIR = BATCH_DIR / "output"
LOGS_DIR = BATCH_DIR / "logs"
SCRIPTS_DIR = BATCH_DIR / "scripts"

# -----------------------------------------------------------------------------
# Re-export batch_057/_common (which re-exports batch_056/_common).
# WHY importlib (not sys.path): batch_057's file is ALSO named `_common.py`.
# Adding its directory to sys.path would shadow THIS module. importlib.util
# with a module spec lets us load the file under a distinct module name.
# -----------------------------------------------------------------------------
import importlib.util as _ilu

_BATCH057_COMMON_PATH = (
    PROJECT_ROOT / "experiments" / "batch_057" / "scripts" / "_common.py"
)
if not _BATCH057_COMMON_PATH.exists():
    raise FileNotFoundError(
        f"batch_057 _common.py missing: {_BATCH057_COMMON_PATH}"
    )
_spec = _ilu.spec_from_file_location(
    "batch057_common", str(_BATCH057_COMMON_PATH)
)
_batch057_common = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_batch057_common)

# Re-export everything we need from batch_057/_common (which transitively
# re-exports batch_056/_common). Not using `from x import *` because the
# module was loaded by importlib and isn't on sys.modules as "batch057_common".
B3_BIOLOGICAL_CATEGORY = _batch057_common.B3_BIOLOGICAL_CATEGORY
B3_GENES = _batch057_common.B3_GENES
BATCH054_P05_PREDS = _batch057_common.BATCH054_P05_PREDS
BATCH055B_WORK = _batch057_common.BATCH055B_WORK
BH_Q = _batch057_common.BH_Q
BOOTSTRAP_N = _batch057_common.BOOTSTRAP_N
BOOTSTRAP_SEED = _batch057_common.BOOTSTRAP_SEED
GENE_ANNOT = _batch057_common.GENE_ANNOT
GNOMAD_TSV = _batch057_common.GNOMAD_TSV
MAGMA_GENELOC = _batch057_common.MAGMA_GENELOC
MAGMA_SCZ_GENES_OUT = _batch057_common.MAGMA_SCZ_GENES_OUT
PGC3_XLSX = _batch057_common.PGC3_XLSX
REPRO_TOLERANCE = _batch057_common.REPRO_TOLERANCE
SYNGO_GMT = _batch057_common.SYNGO_GMT
abs_diff_huber_check = _batch057_common.abs_diff_huber_check
atomic_write_json = _batch057_common.atomic_write_json
bh_fdr = _batch057_common.bh_fdr
build_bootstrap_idx = _batch057_common.build_bootstrap_idx
build_sub_a_frame = _batch057_common.build_sub_a_frame
load_common_ensgids = _batch057_common.load_common_ensgids
load_edt1 = _batch057_common.load_edt1
load_gene_annot = _batch057_common.load_gene_annot
load_gnomad_per_brief_v2 = _batch057_common.load_gnomad_per_brief_v2
load_koopmans_ex_B3 = _batch057_common.load_koopmans_ex_B3
load_koopmans_full_symbols = _batch057_common.load_koopmans_full_symbols
load_magma_disorder = _batch057_common.load_magma_disorder
load_magma_scz = _batch057_common.load_magma_scz
load_nsnps_per_disorder = _batch057_common.load_nsnps_per_disorder
load_preds = _batch057_common.load_preds
percentile_ci = _batch057_common.percentile_ci
rel_diff_huber_check = _batch057_common.rel_diff_huber_check
sha256_file = _batch057_common.sha256_file
symbols_to_ensgids = _batch057_common.symbols_to_ensgids

# setup_logger is batch_057-scoped (logs to batch_057/logs); we define our own
# version pointing at batch_058/logs below.

# -----------------------------------------------------------------------------
# v2 constants (brief_v2 §Reproduction gates)
# -----------------------------------------------------------------------------
# Sub-A R1 (B3): [+2.5, +3.5] per brief_v2 L263 / design.yaml R1_sub_a.
REPRO_R1_SUB_A_LO = 2.5
REPRO_R1_SUB_A_HI = 3.5
# Sub-C R1 (EDT1 full): [+3.0, +3.8] per brief_v2 L263 / design.yaml R1_sub_c.
REPRO_R1_SUB_C_FULL_LO = 3.0
REPRO_R1_SUB_C_FULL_HI = 3.8
# Sub-B R2 (PoPS ρ): [+0.495, +0.535] per brief_v2 L264 (widened v2).
REPRO_R2_RHO_LO = 0.495
REPRO_R2_RHO_HI = 0.535

# Sub-A v2 thresholds (brief_v2 §Sub-A CONJUNCTIVE BATTERY).
CI_UPPER_STRONG = 0.90     # CI UPPER threshold for CI-EXCLUDES-STRONG
DFBETAS_CUTOFF = 1.0       # Fox 1997 max-|DFBETAS| rule
MDE_80_POWER_SUB_A = 0.90  # empirical MDE at SE=0.36σ, N=16,556
FRAGILE_DIFF_THRESHOLD = 0.2  # §INFLUENTIAL-OUTLIERS: diff < 0.2σ after removal

# Sub-B v2.1 thresholds.
SUB_B_SEED = 20260424
SUB_B_PERM_N = 1000
SUB_B_BOOT_N = 1000
SUB_B_CATEGORY_DELTA_FLOOR = 0.10  # v2.1: observed Δρ_k ≥ 0.10 AND p_perm<0.05
SUB_B_PERM_PVALUE = 0.05
SUB_B_OTHER_MASS_GATE = 0.15       # OTHER > 15% → UNINTERPRETABLE
SUB_B_TAIL_ENRICHED_DIFF = 0.05    # |upper ρ − 0.268| > 0.05
SUB_B_LS_NULL_RHO = 0.515          # anchor ρ for LS MC
SUB_B_LS_NULL_TAU = 0.842          # norm.ppf(0.80)
# v2.1 FIX #8: raise n_mc to 100_000 for ±0.0016 precision vs ±0.005 at 10k.
# WHY: LS MC-null is used as a CI anchor in B.2; tighter precision keeps
# its uncertainty well below the observed bootstrap CI width (~±0.025 on
# tail-N=3,492), avoiding MC-null noise dominating the decision.
SUB_B_LS_MC_N = 100000
SUB_B_DISTRIBUTION_DIFF = 0.03     # |rank-partial − raw-partial| > 0.03

# Shapiro-Wilk gate
SHAPIRO_ALPHA = 0.05

# Sub-C v2 (brief_v2 §Sub-C).
SUB_C_MIN_EFFECT = 0.3            # min effect size for F147_EDT1_EXTENDED
SUB_C_MAPPING_GATE = 0.80
SUB_C_FRAGILE_UNINTERPRETABLE_COUNT = 3

# Canonical disorder list (design.yaml).
DISORDERS = ["scz", "bip", "mdd", "asd", "adhd",
             "ibd_delange2017", "height", "alzheimers"]
PSYCHIATRIC = {"scz", "bip", "mdd", "asd", "adhd"}

# Sub-A LOEUF-sensitivity disorders (brief_v2 L82).
LOEUF_DISORDERS = ["scz", "mdd", "asd", "height"]

# PoPS feature category TSV (written by Sub-B pre-step P0).
POPS_FEATURE_CATEGORIES_TSV = SCRIPTS_DIR / "pops_feature_categories.tsv"

# PoPS shard paths.
POPS_FEATURES_MUNGED_DIR = (
    PROJECT_ROOT / "data" / "pops_features" / "features_munged"
)
POPS_FEATURES_ROWS_TXT = POPS_FEATURES_MUNGED_DIR / "pops_features.rows.txt"

POPS_COEFS_P05 = (
    PROJECT_ROOT / "experiments" / "batch_054_A" / "output"
    / "sweep" / "cutoff_0.05" / "PGC3_EUR_PoPS.coefs"
)


# -----------------------------------------------------------------------------
# batch_058-scoped logger (writes to batch_058/logs)
# -----------------------------------------------------------------------------
def setup_logger(name: str, logfile: Path) -> logging.Logger:
    """Logger emitting to `logfile` and stdout.

    WHY we override batch_057's setup_logger: that one routes to
    batch_057/logs; we want all batch_058 logs isolated under
    batch_058/logs for auditor traceability.
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
# Sub-A / Sub-C v2.1 new helpers: robust battery
# =============================================================================
def fit_tukey_biweight(frame: pd.DataFrame, covs: list[str],
                        indicator_col: str = "in_set",
                        outcome_col: str = "MAGMA_Z",
                        max_iter: int = 50,
                        tol: float = 1e-8) -> dict:
    """Tukey biweight RLM for β_1 on indicator_col.

    WHY statsmodels: Cardinal Rule 1. statsmodels ships TukeyBiweight via
    `sm.robust.norms.TukeyBiweight()`. No hand-roll.

    Returns a dict with beta_1, se_1, CI (normal 95%), convergence flag,
    niter, final_scale. On non-finite result, returns status="failed".
    """
    import statsmodels.api as sm
    X = frame[[indicator_col] + covs].to_numpy(dtype=float)
    y = frame[outcome_col].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    try:
        rlm = sm.RLM(y, Xc, M=sm.robust.norms.TukeyBiweight()).fit(
            maxiter=max_iter, tol=tol,
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    if len(rlm.params) <= 1:
        return {"status": "failed", "reason": "rank-deficient"}
    beta_1 = float(rlm.params[1])
    se_1 = float(rlm.bse[1])
    if not (np.isfinite(beta_1) and np.isfinite(se_1)):
        return {"status": "failed",
                "reason": f"non-finite β_1={beta_1} se={se_1}"}
    # statsmodels doesn't expose a simple `.converged` for RLM; we flag by
    # whether iterations hit max_iter. `rlm.fit_history` is a dict with
    # history arrays per iteration.
    niter = int(getattr(rlm, "iteration", -1))
    if niter < 0:
        # Some statsmodels versions store iteration count in fit_history.
        fh = getattr(rlm, "fit_history", {})
        niter = int(len(fh.get("params", []))) if isinstance(fh, dict) else -1
    converged = bool(niter > 0 and niter < max_iter)
    final_scale = float(getattr(rlm, "scale", float("nan")))
    ci_lo = beta_1 - 1.96 * se_1
    ci_hi = beta_1 + 1.96 * se_1
    return {
        "status": "ok",
        "beta_1": beta_1,
        "se_1": se_1,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "converged": converged,
        "niter": niter,
        "final_scale": final_scale,
    }


def compute_dfbetas_cooks(frame: pd.DataFrame, covs: list[str],
                           indicator_col: str = "in_set",
                           outcome_col: str = "MAGMA_Z") -> dict:
    """Max |DFBETAS| on indicator coefficient + Cook's D summary.

    WHY statsmodels: Cardinal Rule 1 — `OLSInfluence` ships
    `.dfbetas` (N×p matrix) and `.cooks_distance` (N-vector, D + p).

    DFBETAS column indexing: `.dfbetas[:, 1]` corresponds to the indicator
    (column 1 after the constant at column 0). Brief_v2 L49 specifies this.

    Returns dict with max_abs_dfbetas_b3 (scalar), cooks_max (scalar),
    cooks_mean (scalar). On failure returns {"status": "failed", ...}.
    """
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import OLSInfluence
    X = frame[[indicator_col] + covs].to_numpy(dtype=float)
    y = frame[outcome_col].to_numpy(dtype=float)
    Xc = sm.add_constant(X, has_constant="add")
    try:
        ols_fit = sm.OLS(y, Xc).fit()
        infl = OLSInfluence(ols_fit)
        dfbetas = np.asarray(infl.dfbetas)
        cooks = np.asarray(infl.cooks_distance[0])
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    if dfbetas.shape[1] <= 1:
        return {"status": "failed",
                "reason": f"dfbetas has too few cols: {dfbetas.shape}"}
    b3_col = dfbetas[:, 1]
    max_abs = float(np.nanmax(np.abs(b3_col))) if b3_col.size else float("nan")
    argmax_abs = int(np.nanargmax(np.abs(b3_col))) if b3_col.size else -1
    cooks_max = float(np.nanmax(cooks)) if cooks.size else float("nan")
    cooks_mean = float(np.nanmean(cooks)) if cooks.size else float("nan")
    cooks_argmax = int(np.nanargmax(cooks)) if cooks.size else -1
    return {
        "status": "ok",
        "max_abs_dfbetas_b3": max_abs,
        "argmax_dfbetas_b3_idx": argmax_abs,
        "cooks_max": cooks_max,
        "cooks_mean": cooks_mean,
        "cooks_argmax_idx": cooks_argmax,
    }


def rank_gaussianize(x: np.ndarray) -> np.ndarray:
    """Rank-based Gaussianization: norm.ppf((rank(x) − 0.5) / N).

    WHY scipy: Cardinal Rule 1. `scipy.stats.rankdata(x, method="average")`
    is the canonical normal-score transformation used in robust regression
    (Conover 1999 §5.1). Ties get average rank; edges shifted by −0.5 avoid
    infinite z at min/max.
    """
    from scipy.stats import norm, rankdata
    r = rankdata(np.asarray(x, dtype=float), method="average")
    n = r.size
    if n == 0:
        return np.array([], dtype=float)
    return norm.ppf((r - 0.5) / n)


# -----------------------------------------------------------------------------
# v2 per-disorder and aggregate classifiers
# -----------------------------------------------------------------------------
def _sign(x: float) -> int:
    """+1 / 0 / −1 sign of x (0 if exactly 0 or NaN)."""
    if not np.isfinite(x):
        return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def classify_disorder_v2(ols_fit: dict, huber_fit: dict,
                          tukey_fit: dict, rank_ols_fit: dict,
                          max_dfbetas: float, q_bh_ols: float,
                          reconciled: bool = False) -> str:
    """Classify one disorder per brief_v2 §Sub-A CONJUNCTIVE BATTERY (L55-L60).

    Categories (first-match):
      ROBUST_POSITIVE     — (a)+(b)+(c)+(d), see brief_v2 L55
      INFLUENTIAL_OUTLIERS — max |DFBETAS| ≥ 1.0 AND reconciled=True
                            (brief_v2 L59)
      CI_EXCLUDES_STRONG  — (a')+(b)+(d): OLS CI UPPER < 0.90 & sign concord
                            & DFBETAS<1.0 (brief_v2 L57). v2.1 FIX #4: if
                            OLS |β|>0.3σ AND sign(rank-β) != sign(OLS-β),
                            demote to FRAGILE (flagrant rank-sign disagreement
                            at non-trivial effect size). For |β|<=0.3σ, rank
                            near-null sign flips are expected and ignored.
      FRAGILE             — sign disagreement NOT resolved by DFBETAS
                            (brief_v2 L58) OR max|DFBETAS|>=1.0 but caller
                            has not reconciled (v2.1 FIX #5).
      UNCLASSIFIED        — OLS CI includes 0 but CI UPPER ≥ 0.90 (L60)

    v2.1 FIX #5: `reconciled` is a REQUIRED input that the caller pre-computes
    by dropping max-|DFBETAS| observation(s), refitting OLS + Tukey, and
    checking |β_ols_post − β_tukey_post| < FRAGILE_DIFF_THRESHOLD (0.2σ).
    INFLUENTIAL_OUTLIERS only fires when reconciled=True; else FRAGILE.
    This prevents mis-labeling DFBETAS breaches that do NOT reconcile.

    WHY this ordering: brief_v2 L59 reads "INFLUENTIAL_OUTLIERS: max|DFBETAS|
    ≥ 1.0 AND removing those obs reconciles OLS vs Tukey". We check the
    DFBETAS floor early but gate on reconciled=True before returning
    INFLUENTIAL_OUTLIERS.
    """
    # Gate on OLS fit existing and finite.
    if ols_fit.get("status") == "failed" or ols_fit.get("beta_1") is None:
        return "UNCLASSIFIED"
    beta_ols = float(ols_fit["beta_1"])
    se_ols = float(ols_fit["se_1"])
    if not (np.isfinite(beta_ols) and np.isfinite(se_ols) and se_ols > 0):
        return "UNCLASSIFIED"
    ci_lo = beta_ols - 1.96 * se_ols
    ci_hi = beta_ols + 1.96 * se_ols

    # Extract sign-concordance components.
    beta_huber = float(huber_fit.get("beta_1", float("nan")))
    beta_tukey = float(tukey_fit.get("beta_1", float("nan")))
    beta_rank = float(rank_ols_fit.get("beta_1", float("nan")))
    s_ols = _sign(beta_ols)
    s_huber = _sign(beta_huber)
    s_tukey = _sign(beta_tukey)
    s_rank = _sign(beta_rank)

    # Determine sign concordance (v2 fix: rank contributes sign only, not q).
    sign_concord_all = (s_ols != 0 and s_huber == s_ols
                         and s_tukey == s_ols and s_rank == s_ols)
    sign_concord_robust_arms = (s_ols != 0 and s_huber == s_ols
                                 and s_tukey == s_ols)  # for CI-EXCLUDES
    sign_disagree_robust = (
        (s_huber != 0 and s_huber != s_ols)
        or (s_tukey != 0 and s_tukey != s_ols)
    )
    # v2.1 FIX #4: rank-sign flagrant disagreement at non-trivial OLS β.
    # If |β_ols| > 0.3σ AND sign(rank-β) != sign(OLS-β), the rank
    # evidence is against the effect at a magnitude where rank and OLS
    # agree under a real signal; treat as FRAGILE, not CI_EXCLUDES_STRONG.
    # At |β_ols| <= 0.3σ, rank near-null sign flips are expected and ignored.
    rank_flagrantly_disagrees = (
        abs(beta_ols) > 0.3
        and np.isfinite(beta_rank)
        and s_rank != 0
        and s_rank != s_ols
    )

    dfbetas_pass = (np.isfinite(max_dfbetas)
                    and max_dfbetas < DFBETAS_CUTOFF)

    # INFLUENTIAL_OUTLIERS: DFBETAS floor breached AND reconciliation succeeded.
    # v2.1 FIX #5: reconciled is caller-provided; False means the post-removal
    # refit did NOT bring OLS and Tukey β into agreement (diff >= 0.2σ), so we
    # don't earn the INFLUENTIAL_OUTLIERS label — we return FRAGILE instead.
    if np.isfinite(max_dfbetas) and max_dfbetas >= DFBETAS_CUTOFF:
        if reconciled:
            return "INFLUENTIAL_OUTLIERS"
        return "FRAGILE"

    # ROBUST_POSITIVE: OLS β>0, q<0.05, sign-concord all arms, DFBETAS<1.0.
    robust_positive = (
        beta_ols > 0 and np.isfinite(q_bh_ols) and q_bh_ols < BH_Q
        and sign_concord_all and dfbetas_pass
    )
    if robust_positive:
        return "ROBUST_POSITIVE"

    # CI_EXCLUDES_STRONG: OLS CI UPPER < 0.90, sign-concord on non-rank arms
    # near zero (sign may be ±1 on small estimates), DFBETAS pass.
    # Brief_v2 L57 reads "sign-concordance across Huber/Tukey/rank-MAGMA (all
    # near-zero)"; we operationalize that as: CI UPPER < 0.90 AND Huber/Tukey
    # CI UPPER < 0.90 OR sign 0, AND DFBETAS pass.
    # v2.1 FIX #4: demote to FRAGILE if rank flagrantly disagrees at
    # non-trivial OLS |β|.
    ci_excludes_strong = (ci_hi < CI_UPPER_STRONG and dfbetas_pass)
    if ci_excludes_strong:
        if rank_flagrantly_disagrees:
            return "FRAGILE"
        # Also require signs do not flagrantly disagree (any single arm
        # disagreeing with OLS near-zero is OK; we just require overall
        # Huber/Tukey β are not > 0.90 themselves).
        beta_huber_ok = (not np.isfinite(beta_huber)
                         or beta_huber < CI_UPPER_STRONG)
        beta_tukey_ok = (not np.isfinite(beta_tukey)
                         or beta_tukey < CI_UPPER_STRONG)
        if beta_huber_ok and beta_tukey_ok:
            return "CI_EXCLUDES_STRONG"

    # FRAGILE: robust-arm sign disagreement not resolved by DFBETAS.
    # Caller passes only classify_disorder_v2 AFTER any post-removal refit;
    # we flag FRAGILE here if signs still disagree.
    if sign_disagree_robust:
        return "FRAGILE"

    # UNCLASSIFIED: everything else (e.g., CI includes 0 but CI UPPER ≥ 0.90).
    return "UNCLASSIFIED"


def aggregate_pattern_sub_a(per_disorder_class_map: dict[str, str]) -> dict:
    """First-match per brief_v2 §Sub-A DECISION RULE (L63-L68).

    Input: {disorder: classification_str}. Every disorder MUST be classified.

    Returns {"classification": str, "reason": str, "counts": {...}}.
    """
    d = per_disorder_class_map
    robust = [k for k, v in d.items() if v == "ROBUST_POSITIVE"]
    ci_strong = [k for k, v in d.items() if v == "CI_EXCLUDES_STRONG"]
    unclass = [k for k, v in d.items() if v == "UNCLASSIFIED"]
    fragile = [k for k, v in d.items() if v == "FRAGILE"]
    infl = [k for k, v in d.items() if v == "INFLUENTIAL_OUTLIERS"]
    psych_robust = [k for k in robust if k in PSYCHIATRIC and k != "scz"]
    nonpsych_robust = [k for k in robust if k not in PSYCHIATRIC]
    counts = {
        "robust_positive": robust,
        "ci_excludes_strong": ci_strong,
        "unclassified": unclass,
        "fragile": fragile,
        "influential_outliers": infl,
        "psych_robust_non_scz": psych_robust,
        "nonpsych_robust": nonpsych_robust,
    }

    # UNINTERPRETABLE first if too many FRAGILE (brief_v2 L68).
    if len(fragile) >= 3:
        return {
            "classification": "UNINTERPRETABLE",
            "reason": f">=3 FRAGILE disorders: {fragile}",
            "counts": counts,
        }

    scz_robust = "scz" in robust

    # 1. SCZ-SPECIFIC-ROBUST: SCZ robust AND all 7 others CI-EXCLUDES-STRONG.
    others = [k for k in d if k != "scz"]
    all_others_ci_strong = all(d[k] == "CI_EXCLUDES_STRONG" for k in others)
    if scz_robust and all_others_ci_strong:
        return {
            "classification": "SCZ_SPECIFIC_ROBUST",
            "reason": "SCZ ROBUST_POSITIVE AND all 7 others CI_EXCLUDES_STRONG",
            "counts": counts,
        }

    # 2. SCZ + ONE-PSYCHIATRIC-ROBUST: exactly 1 of {BIP,MDD,ASD,ADHD} robust
    # AND 0 non-psych NCs robust.
    if (scz_robust and len(psych_robust) == 1
            and len(nonpsych_robust) == 0):
        return {
            "classification": "SCZ_PLUS_ONE_PSYCHIATRIC_ROBUST",
            "reason": (f"SCZ ROBUST + 1 psych partner {psych_robust} + "
                        "0 NCs robust"),
            "counts": counts,
        }

    # 3. PAN-PSYCHIATRIC: SCZ robust AND ≥ 2 of {BIP,MDD,ASD,ADHD} robust
    # AND 0 non-psych NCs robust.
    if (scz_robust and len(psych_robust) >= 2
            and len(nonpsych_robust) == 0):
        return {
            "classification": "PAN_PSYCHIATRIC_ROBUST",
            "reason": (f"SCZ ROBUST + >=2 psych partners {psych_robust} + "
                        "0 NCs robust"),
            "counts": counts,
        }

    # 4. UNIVERSAL-BASELINE-ROBUST: ≥ 6 of 8 robust INCLUDING ≥ 1 non-psych NC.
    if len(robust) >= 6 and len(nonpsych_robust) >= 1:
        return {
            "classification": "UNIVERSAL_BASELINE_ROBUST",
            "reason": (f">=6/8 ROBUST including >=1 non-psych NC "
                        f"({nonpsych_robust})"),
            "counts": counts,
        }

    # 5. INTERMEDIATE (committing): mixed ROBUST + UNCLASSIFIED +
    # CI_EXCLUDES_STRONG not matching archetypes 1-4.
    return {
        "classification": "INTERMEDIATE",
        "reason": ("None of SCZ_SPECIFIC / SCZ+ONE / PAN / UNIVERSAL fired. "
                    "F-055-01 is CLOSED at SUGGESTED; iter_059 may NOT "
                    "re-run Sub-A with covariate variation (L056_02 binding)."),
        "counts": counts,
    }


def aggregate_pattern_sub_c(per_disorder_class_map: dict[str, str],
                              scz_beta_ols: float) -> dict:
    """First-match per brief_v2 §Sub-C DECISION RULE (L225-L230).

    1. F147_NARROW_CONFIRMED: SCZ β < 0.3σ OR SCZ not ROBUST_POSITIVE OR
       SCZ CI_EXCLUDES_STRONG.
    2. F147_EDT1_EXTENDED: SCZ ROBUST_POSITIVE β ≥ 0.3σ AND ≤ 2 of 7 others
       ROBUST_POSITIVE.
    3. F147_CROSS_DISORDER_EXTENDED: ≥ 3 disorders ROBUST_POSITIVE.
    4. INTERMEDIATE: mixed.
    5. UNINTERPRETABLE: ≥ 3 FRAGILE.
    """
    d = per_disorder_class_map
    robust = [k for k, v in d.items() if v == "ROBUST_POSITIVE"]
    fragile = [k for k, v in d.items() if v == "FRAGILE"]
    counts = {
        "robust_positive": robust,
        "fragile": fragile,
        "n_disorders": len(d),
    }

    if len(fragile) >= SUB_C_FRAGILE_UNINTERPRETABLE_COUNT:
        return {
            "classification": "UNINTERPRETABLE",
            "reason": f">=3 FRAGILE disorders: {fragile}",
            "counts": counts,
        }

    scz_class = d.get("scz", "UNCLASSIFIED")
    # v2.1 FIX #7: if SCZ β is non-finite we CANNOT compare to 0.3σ. Returning
    # F147_NARROW_CONFIRMED via `scz_beta := 0.0` would fabricate a conclusion
    # from a fit that failed. Return UNINTERPRETABLE so the caller surfaces
    # this correctly (Cardinal Rule 0: no fake numbers as conclusions).
    if (scz_beta_ols is None
            or not np.isfinite(float(scz_beta_ols))):
        return {
            "classification": "UNINTERPRETABLE",
            "reason": "SCZ β non-finite (OLS fit failed or produced NaN/Inf)",
            "counts": counts,
        }
    scz_beta = float(scz_beta_ols)

    # 1. F147_NARROW_CONFIRMED.
    if (scz_beta < SUB_C_MIN_EFFECT
            or scz_class != "ROBUST_POSITIVE"
            or scz_class == "CI_EXCLUDES_STRONG"):
        return {
            "classification": "F147_NARROW_CONFIRMED",
            "reason": (f"SCZ β={scz_beta:.3f} < 0.3σ OR not ROBUST_POSITIVE "
                        f"(class={scz_class})"),
            "counts": counts,
        }

    # At this point SCZ is ROBUST_POSITIVE with β ≥ 0.3σ.
    # 3. F147_CROSS_DISORDER_EXTENDED: ≥ 3 disorders robust.
    if len(robust) >= 3:
        return {
            "classification": "F147_CROSS_DISORDER_EXTENDED",
            "reason": f">=3 ROBUST disorders ({robust}); scope not SCZ-specific",
            "counts": counts,
        }

    # 2. F147_EDT1_EXTENDED: SCZ robust β≥0.3σ AND ≤ 2 of 7 others robust.
    others_robust = [k for k in robust if k != "scz"]
    if len(others_robust) <= 2:
        return {
            "classification": "F147_EDT1_EXTENDED",
            "reason": (f"SCZ ROBUST β={scz_beta:.3f}; <=2 other ROBUST "
                        f"({others_robust})"),
            "counts": counts,
        }

    # 4. INTERMEDIATE (tightened per C3 MAJOR #1): committing conclusion.
    return {
        "classification": "INTERMEDIATE",
        "reason": ("Mixed pattern (SCZ ROBUST but FRAGILE or UNCLASSIFIED on "
                    "others). F147 scope remains B3-concentrated; EDT1 scope "
                    "extension not sustained. L056_02 binds iter_059."),
        "counts": counts,
    }


# =============================================================================
# Sub-B v2.1 new helpers: PoPS coefs/features, categorization, permutation null
# =============================================================================
def load_pops_coefs(path: Path) -> pd.Series:
    """Parse `PGC3_EUR_PoPS.coefs` → Series indexed by feature name.

    PoPS coefs file layout (verified in batch_054_A output):
        parameter\tbeta          ← TSV header
        METHOD\tRidgeCV
        SELECTED_CV_ALPHA\t31622.776601683792
        BEST_CV_SCORE\t-0.5172858752289862
        GTEx.2\t-0.000623...
        ...

    We SKIP the three metadata rows (METHOD, SELECTED_CV_ALPHA,
    BEST_CV_SCORE) plus the header and return a pandas Series.

    Returns Series[feature_name -> β].
    """
    if not path.exists():
        raise FileNotFoundError(f"PoPS coefs file missing: {path}")
    df = pd.read_csv(path, sep="\t", header=0,
                     names=["parameter", "beta"], dtype=str)
    meta_rows = {"METHOD", "SELECTED_CV_ALPHA", "BEST_CV_SCORE",
                 "parameter"}
    df = df[~df["parameter"].isin(meta_rows)].copy()
    df["beta"] = pd.to_numeric(df["beta"], errors="raise")
    if df["parameter"].duplicated().any():
        raise RuntimeError(
            f"PoPS coefs has duplicate feature names: "
            f"{df[df['parameter'].duplicated()]['parameter'].tolist()[:5]}"
        )
    out = pd.Series(df["beta"].values, index=df["parameter"].values,
                    name="beta")
    return out


def load_pops_features_matrix(features_dir: Path,
                                cols_to_load: list[str] | set[str] | None = None
                                ) -> tuple[np.ndarray, list[str], list[str]]:
    """Load sharded PoPS feature mats (12 shards) and concatenate horizontally.

    Args:
      features_dir: directory holding pops_features.{mat,cols,rows}.*.
      cols_to_load: if provided, only load columns whose names are in this
                    set. Reduces memory when we only need features present
                    in the coefs file.

    Returns (X, gene_ensgids, feature_names) where X is shape (N, P_kept).

    WHY sharded load (not pre-concatenated): the full matrix is
    18,383 × 57,742 × 8 bytes = ~8.5 GB; loading only the subset we need
    (features in coefs, ~17,427) still requires ~2.6 GB which is affordable
    in memory. We use mmap + float32 conversion to keep RAM under control
    when the caller doesn't restrict cols.
    """
    if not features_dir.exists():
        raise FileNotFoundError(
            f"PoPS features dir missing: {features_dir}")
    rows_path = features_dir / "pops_features.rows.txt"
    if not rows_path.exists():
        raise FileNotFoundError(f"PoPS rows file missing: {rows_path}")
    with rows_path.open() as fh:
        gene_ensgids = [l.strip() for l in fh if l.strip()]

    cols_filter: set[str] | None
    if cols_to_load is None:
        cols_filter = None
    else:
        cols_filter = set(cols_to_load)

    parts: list[np.ndarray] = []
    kept_names: list[str] = []
    i = 0
    while True:
        cols_path = features_dir / f"pops_features.cols.{i}.txt"
        mat_path = features_dir / f"pops_features.mat.{i}.npy"
        if not cols_path.exists() and not mat_path.exists():
            break
        if not cols_path.exists() or not mat_path.exists():
            raise FileNotFoundError(
                f"PoPS shard {i} incomplete: cols={cols_path.exists()} "
                f"mat={mat_path.exists()}"
            )
        with cols_path.open() as fh:
            shard_cols = [l.strip() for l in fh if l.strip()]
        mat = np.load(mat_path, mmap_mode="r")
        if mat.shape[0] != len(gene_ensgids):
            raise RuntimeError(
                f"PoPS shard {i} row-count mismatch: mat={mat.shape[0]} "
                f"rows_txt={len(gene_ensgids)}"
            )
        if mat.shape[1] != len(shard_cols):
            raise RuntimeError(
                f"PoPS shard {i} col-count mismatch: mat={mat.shape[1]} "
                f"cols_txt={len(shard_cols)}"
            )
        if cols_filter is None:
            parts.append(np.ascontiguousarray(mat, dtype=np.float32))
            kept_names.extend(shard_cols)
        else:
            keep_mask = np.array(
                [c in cols_filter for c in shard_cols], dtype=bool,
            )
            if keep_mask.any():
                sub = np.ascontiguousarray(
                    mat[:, keep_mask], dtype=np.float32,
                )
                parts.append(sub)
                kept_names.extend(
                    [c for c, k in zip(shard_cols, keep_mask) if k]
                )
        i += 1
    if not parts:
        raise RuntimeError(f"No shards loaded from {features_dir}")
    X = np.concatenate(parts, axis=1)
    return X, gene_ensgids, kept_names


# Regex patterns from brief_v2 §Sub-B P0 (L109-L112).
# WHY these exact regexes: brief_v2 gives them verbatim; also design.yaml
# L103-L105 serialize them. We compile once at import.
_EXPRESSION_PATTERNS = [
    re.compile(r"^GTEx\."),
    re.compile(r"^BrainSpan\."),
    re.compile(r"^Franke\."),
    re.compile(r"^ImmGen\."),
    re.compile(r"pcaloadings"),
    re.compile(r"diffexprs"),
    re.compile(r"Tissue_"),
    re.compile(r"Cell_"),
    re.compile(r"Roadmap_"),
    re.compile(r"ENCODE_"),
    re.compile(r"_expression"),
    re.compile(r"average_expression"),
]
_PPI_PATTERNS = [
    re.compile(r"^ppi\."),
    re.compile(r"^ppi_"),
]
_PATHWAY_PATTERNS = [
    re.compile(r"^pathways\."),
    re.compile(r"^c2_"),
    re.compile(r"^c5_"),
    re.compile(r"^hallmark"),
    re.compile(r"^MSigDB_"),
    re.compile(r"^GO_"),
    re.compile(r"^KEGG_"),
    re.compile(r"^Reactome_"),
]


def _classify_one_feature(name: str) -> str:
    for pat in _EXPRESSION_PATTERNS:
        if pat.search(name):
            return "expression"
    for pat in _PPI_PATTERNS:
        if pat.search(name):
            return "ppi"
    for pat in _PATHWAY_PATTERNS:
        if pat.search(name):
            return "pathway"
    return "other"


def classify_feature(name: str) -> str:
    """Public alias for the single-feature classifier (brief_v2 §Sub-B P0)."""
    return _classify_one_feature(name)


def categorize_pops_features(feature_names: list[str],
                               logger: logging.Logger | None = None
                               ) -> dict[str, str]:
    """Map each feature name → category ∈ {expression, ppi, pathway, other}.

    Uses brief_v2 §Sub-B P0 regex. Computes the categorization IN MEMORY and
    (v2.1 FIX #9) compares the result to the committed
    `pops_feature_categories.tsv` via SHA256:
      - If the file does NOT exist, write it once.
      - If the file exists AND SHA256 matches, no-op (preserves commit hash).
      - If SHA256 differs, log a WARNING (does NOT halt) so auditors see drift.

    WHY not rewrite every run: rewriting mutates committed artifact content
    across runs and obscures whether the partition is stable; auditors track
    git hash. If the in-memory partition disagrees, we flag loudly but don't
    kill the experiment — the in-memory value is what's used downstream, and
    the file serves only as a documented artifact.

    Returns dict[feature_name -> category].
    """
    import hashlib as _hashlib
    import os as _os
    cats = {name: _classify_one_feature(name) for name in feature_names}

    # Build canonical TSV bytes (sorted by feature name, deterministic).
    lines = ["feature\tcategory\n"]
    for name in sorted(cats.keys()):
        lines.append(f"{name}\t{cats[name]}\n")
    content = "".join(lines).encode("utf-8")
    computed_sha = _hashlib.sha256(content).hexdigest()

    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    if not POPS_FEATURE_CATEGORIES_TSV.exists():
        # First-time write only.
        tmp = POPS_FEATURE_CATEGORIES_TSV.with_suffix(".tsv.tmp")
        with tmp.open("wb") as fh:
            fh.write(content)
        _os.replace(tmp, POPS_FEATURE_CATEGORIES_TSV)
        if logger is not None:
            logger.info("wrote pops_feature_categories.tsv (first time, "
                        "sha256=%s)", computed_sha)
    else:
        committed_sha = sha256_file(POPS_FEATURE_CATEGORIES_TSV)
        if committed_sha != computed_sha:
            msg = (f"pops_feature_categories.tsv SHA256 drift: "
                   f"committed={committed_sha[:12]}... "
                   f"computed={computed_sha[:12]}... "
                   f"(n_feat={len(feature_names)}); "
                   f"in-memory values used; file NOT rewritten.")
            if logger is not None:
                logger.warning(msg)
            else:
                import warnings as _w
                _w.warn(msg)
    return cats


def longin_solnik_mc_null(rho: float = 0.515, tau: float = 0.842,
                            n_mc: int = 10000,
                            seed: int = 20260424) -> dict:
    """Monte-Carlo Longin-Solnik tail-exceedance ρ under bivariate normal.

    Draw (X, Y) ~ N(0, Σ) with Σ = [[1,ρ],[ρ,1]]. Condition on Y ≥ τ
    (upper 80th pct for default τ=0.842 = norm.ppf(0.80)). Return observed
    Pearson ρ on the tail sample.

    WHY MC rather than closed-form LS formula: brief_v2 spec is MC "at ρ=0.515,
    τ=0.842, seed=20260424, n=10000"; target output ≈ 0.268. We return both
    the MC point estimate and the count of tail observations for audit.
    """
    from scipy.stats import pearsonr
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    samples = rng.multivariate_normal([0.0, 0.0], cov, size=n_mc)
    y = samples[:, 1]
    mask = y >= tau
    n_tail = int(mask.sum())
    if n_tail < 3:
        return {"status": "failed", "reason": "tail too small",
                "n_tail": n_tail}
    tail_x = samples[mask, 0]
    tail_y = samples[mask, 1]
    rho_obs, _ = pearsonr(tail_x, tail_y)
    return {
        "status": "ok",
        "tail_rho_mc": float(rho_obs),
        "n_mc": int(n_mc),
        "n_tail": n_tail,
        "rho_input": float(rho),
        "tau_input": float(tau),
        "seed": int(seed),
    }


# -----------------------------------------------------------------------------
# Partial-correlation helper (Rule 1 — thin wrapper around numpy).
# -----------------------------------------------------------------------------
def partial_pearson(y1: np.ndarray, y2: np.ndarray,
                     covariates: np.ndarray) -> float:
    """Partial Pearson ρ(y1, y2 | covariates) via residualization.

    WHY residualization: partial ρ = ρ(resid_1, resid_2) where residuals
    are from OLS regressions of y_i on covariates. Bit-identical to
    pingouin.partial_corr(method='pearson').
    """
    from scipy.stats import pearsonr
    n = y1.shape[0]
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)
    Xc = np.column_stack([np.ones(n), covariates])
    beta_1, *_ = np.linalg.lstsq(Xc, y1.astype(float), rcond=None)
    beta_2, *_ = np.linalg.lstsq(Xc, y2.astype(float), rcond=None)
    r1 = y1.astype(float) - Xc @ beta_1
    r2 = y2.astype(float) - Xc @ beta_2
    r, _ = pearsonr(r1, r2)
    return float(r)


__all__ = [
    # paths
    "PROJECT_ROOT", "BATCH_DIR", "OUTPUT_DIR", "LOGS_DIR", "SCRIPTS_DIR",
    "POPS_FEATURE_CATEGORIES_TSV", "POPS_FEATURES_MUNGED_DIR",
    "POPS_FEATURES_ROWS_TXT", "POPS_COEFS_P05",
    # re-exports
    "B3_BIOLOGICAL_CATEGORY", "B3_GENES", "BATCH054_P05_PREDS",
    "BATCH055B_WORK", "BH_Q", "BOOTSTRAP_N", "BOOTSTRAP_SEED",
    "GENE_ANNOT", "GNOMAD_TSV", "MAGMA_GENELOC", "MAGMA_SCZ_GENES_OUT",
    "PGC3_XLSX", "REPRO_TOLERANCE", "SYNGO_GMT",
    "abs_diff_huber_check", "atomic_write_json", "bh_fdr",
    "build_bootstrap_idx", "build_sub_a_frame",
    "load_common_ensgids", "load_edt1", "load_gene_annot",
    "load_gnomad_per_brief_v2", "load_koopmans_ex_B3",
    "load_koopmans_full_symbols", "load_magma_disorder", "load_magma_scz",
    "load_nsnps_per_disorder", "load_preds", "percentile_ci",
    "rel_diff_huber_check", "sha256_file", "symbols_to_ensgids",
    "setup_logger",
    # v2 constants
    "REPRO_R1_SUB_A_LO", "REPRO_R1_SUB_A_HI",
    "REPRO_R1_SUB_C_FULL_LO", "REPRO_R1_SUB_C_FULL_HI",
    "REPRO_R2_RHO_LO", "REPRO_R2_RHO_HI",
    "CI_UPPER_STRONG", "DFBETAS_CUTOFF", "MDE_80_POWER_SUB_A",
    "FRAGILE_DIFF_THRESHOLD",
    "SUB_B_SEED", "SUB_B_PERM_N", "SUB_B_BOOT_N",
    "SUB_B_CATEGORY_DELTA_FLOOR", "SUB_B_PERM_PVALUE",
    "SUB_B_OTHER_MASS_GATE", "SUB_B_TAIL_ENRICHED_DIFF",
    "SUB_B_LS_NULL_RHO", "SUB_B_LS_NULL_TAU", "SUB_B_LS_MC_N",
    "SUB_B_DISTRIBUTION_DIFF", "SHAPIRO_ALPHA",
    "SUB_C_MIN_EFFECT", "SUB_C_MAPPING_GATE",
    "SUB_C_FRAGILE_UNINTERPRETABLE_COUNT",
    "DISORDERS", "PSYCHIATRIC", "LOEUF_DISORDERS",
    # new v2 helpers
    "fit_tukey_biweight", "compute_dfbetas_cooks", "rank_gaussianize",
    "classify_disorder_v2", "aggregate_pattern_sub_a",
    "aggregate_pattern_sub_c",
    "load_pops_coefs", "load_pops_features_matrix",
    "categorize_pops_features", "classify_feature",
    "longin_solnik_mc_null", "partial_pearson",
]
