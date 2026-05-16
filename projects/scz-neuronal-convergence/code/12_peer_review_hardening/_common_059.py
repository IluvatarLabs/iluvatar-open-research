"""Shared helpers for batch_059 sub-experiments (iter_059 VERA E1/E2/E3 + Bellenguez E4).

WHY this module exists: Cardinal Rule 1. Re-export batch_058/_common (which
transitively re-exports batch_057/_common and batch_056/_common). All loaders
already exist; we only add iter_059-specific constants (E1 partition regex,
E2 SynGO GO IDs, E3 Thorndike Case II closed-form constants, E4 Bellenguez
paths) and a new SynGO GO-term subset loader.

Determinism + seeds per design.yaml: master=20260424, E1 perm offset +1,
E2 perm offset +2, E3 MC offset +3, E4 bootstrap uses master.

No fabrication (Cardinal Rule 0): every loader raises FileNotFoundError with
a clear diagnostic on missing inputs. SHA256 provenance via re-exported
`sha256_file`.
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
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_059"
OUTPUT_DIR = BATCH_DIR / "output"
LOGS_DIR = BATCH_DIR / "logs"
SCRIPTS_DIR = BATCH_DIR / "scripts"

# -----------------------------------------------------------------------------
# Re-export batch_058/_common via importlib (same pattern as batch_058
# re-exporting batch_057, which re-exports batch_056).
# WHY importlib: all ancestor files are ALSO named `_common.py`; sys.path
# tricks would shadow THIS module. importlib lets us bind them as distinct
# modules.
# -----------------------------------------------------------------------------
import importlib.util as _ilu

_BATCH058_COMMON_PATH = (
    PROJECT_ROOT / "experiments" / "batch_058" / "scripts" / "_common.py"
)
if not _BATCH058_COMMON_PATH.exists():
    raise FileNotFoundError(
        f"batch_058 _common.py missing: {_BATCH058_COMMON_PATH}"
    )
_spec = _ilu.spec_from_file_location(
    "batch058_common", str(_BATCH058_COMMON_PATH)
)
_batch058_common = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_batch058_common)

# Re-export everything we need from batch_058/_common (which transitively
# re-exports batch_057/_common and batch_056/_common).
B3_BIOLOGICAL_CATEGORY = _batch058_common.B3_BIOLOGICAL_CATEGORY
B3_GENES = _batch058_common.B3_GENES
BATCH054_P05_PREDS = _batch058_common.BATCH054_P05_PREDS
BATCH055B_WORK = _batch058_common.BATCH055B_WORK
BH_Q = _batch058_common.BH_Q
BOOTSTRAP_N = _batch058_common.BOOTSTRAP_N
BOOTSTRAP_SEED = _batch058_common.BOOTSTRAP_SEED
GENE_ANNOT = _batch058_common.GENE_ANNOT
GNOMAD_TSV = _batch058_common.GNOMAD_TSV
MAGMA_GENELOC = _batch058_common.MAGMA_GENELOC
MAGMA_SCZ_GENES_OUT = _batch058_common.MAGMA_SCZ_GENES_OUT
PGC3_XLSX = _batch058_common.PGC3_XLSX
POPS_COEFS_P05 = _batch058_common.POPS_COEFS_P05
POPS_FEATURES_MUNGED_DIR = _batch058_common.POPS_FEATURES_MUNGED_DIR
POPS_FEATURES_ROWS_TXT = _batch058_common.POPS_FEATURES_ROWS_TXT
POPS_FEATURE_CATEGORIES_TSV = _batch058_common.POPS_FEATURE_CATEGORIES_TSV
REPRO_TOLERANCE = _batch058_common.REPRO_TOLERANCE
SYNGO_GMT = _batch058_common.SYNGO_GMT

# v2 / v2.1 constants from batch_058
CI_UPPER_STRONG = _batch058_common.CI_UPPER_STRONG
DFBETAS_CUTOFF = _batch058_common.DFBETAS_CUTOFF
FRAGILE_DIFF_THRESHOLD = _batch058_common.FRAGILE_DIFF_THRESHOLD
MDE_80_POWER_SUB_A = _batch058_common.MDE_80_POWER_SUB_A
SHAPIRO_ALPHA = _batch058_common.SHAPIRO_ALPHA
SUB_B_BOOT_N = _batch058_common.SUB_B_BOOT_N
SUB_B_CATEGORY_DELTA_FLOOR = _batch058_common.SUB_B_CATEGORY_DELTA_FLOOR
SUB_B_DISTRIBUTION_DIFF = _batch058_common.SUB_B_DISTRIBUTION_DIFF
SUB_B_LS_MC_N = _batch058_common.SUB_B_LS_MC_N
SUB_B_LS_NULL_RHO = _batch058_common.SUB_B_LS_NULL_RHO
SUB_B_LS_NULL_TAU = _batch058_common.SUB_B_LS_NULL_TAU
SUB_B_OTHER_MASS_GATE = _batch058_common.SUB_B_OTHER_MASS_GATE
SUB_B_PERM_N = _batch058_common.SUB_B_PERM_N
SUB_B_PERM_PVALUE = _batch058_common.SUB_B_PERM_PVALUE
SUB_B_SEED = _batch058_common.SUB_B_SEED
SUB_B_TAIL_ENRICHED_DIFF = _batch058_common.SUB_B_TAIL_ENRICHED_DIFF
SUB_C_FRAGILE_UNINTERPRETABLE_COUNT = (
    _batch058_common.SUB_C_FRAGILE_UNINTERPRETABLE_COUNT
)
SUB_C_MAPPING_GATE = _batch058_common.SUB_C_MAPPING_GATE
SUB_C_MIN_EFFECT = _batch058_common.SUB_C_MIN_EFFECT

DISORDERS = _batch058_common.DISORDERS
LOEUF_DISORDERS = _batch058_common.LOEUF_DISORDERS
PSYCHIATRIC = _batch058_common.PSYCHIATRIC

# Anchors
REPRO_R1_SUB_A_LO = _batch058_common.REPRO_R1_SUB_A_LO
REPRO_R1_SUB_A_HI = _batch058_common.REPRO_R1_SUB_A_HI
REPRO_R1_SUB_C_FULL_LO = _batch058_common.REPRO_R1_SUB_C_FULL_LO
REPRO_R1_SUB_C_FULL_HI = _batch058_common.REPRO_R1_SUB_C_FULL_HI
REPRO_R2_RHO_LO = _batch058_common.REPRO_R2_RHO_LO
REPRO_R2_RHO_HI = _batch058_common.REPRO_R2_RHO_HI

# Loaders / helpers
abs_diff_huber_check = _batch058_common.abs_diff_huber_check
aggregate_pattern_sub_a = _batch058_common.aggregate_pattern_sub_a
aggregate_pattern_sub_c = _batch058_common.aggregate_pattern_sub_c
atomic_write_json = _batch058_common.atomic_write_json
bh_fdr = _batch058_common.bh_fdr
build_bootstrap_idx = _batch058_common.build_bootstrap_idx
build_sub_a_frame = _batch058_common.build_sub_a_frame
categorize_pops_features = _batch058_common.categorize_pops_features
classify_disorder_v2 = _batch058_common.classify_disorder_v2
classify_feature = _batch058_common.classify_feature
compute_dfbetas_cooks = _batch058_common.compute_dfbetas_cooks
fit_tukey_biweight = _batch058_common.fit_tukey_biweight
load_common_ensgids = _batch058_common.load_common_ensgids
load_edt1 = _batch058_common.load_edt1
load_gene_annot = _batch058_common.load_gene_annot
load_gnomad_per_brief_v2 = _batch058_common.load_gnomad_per_brief_v2
load_koopmans_ex_B3 = _batch058_common.load_koopmans_ex_B3
load_koopmans_full_symbols = _batch058_common.load_koopmans_full_symbols
load_magma_disorder = _batch058_common.load_magma_disorder
load_magma_scz = _batch058_common.load_magma_scz
load_nsnps_per_disorder = _batch058_common.load_nsnps_per_disorder
load_pops_coefs = _batch058_common.load_pops_coefs
load_pops_features_matrix = _batch058_common.load_pops_features_matrix
load_preds = _batch058_common.load_preds
longin_solnik_mc_null = _batch058_common.longin_solnik_mc_null
partial_pearson = _batch058_common.partial_pearson
percentile_ci = _batch058_common.percentile_ci
rank_gaussianize = _batch058_common.rank_gaussianize
rel_diff_huber_check = _batch058_common.rel_diff_huber_check
sha256_file = _batch058_common.sha256_file
symbols_to_ensgids = _batch058_common.symbols_to_ensgids


# =============================================================================
# batch_059-scoped logger (writes to batch_059/logs)
# =============================================================================
def setup_logger(name: str, logfile: Path) -> logging.Logger:
    """Logger emitting to `logfile` and stdout.

    WHY we override: keep batch_059 logs isolated under batch_059/logs for
    auditor traceability.
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
# Seeds (design.yaml §seed)
# =============================================================================
SEED_MASTER = 20260424  # design.yaml seed.master
SEED_E1_BOOT = 20260424  # design.yaml seed.e1_bootstrap (no offset)
SEED_E1_PERM = 20260425  # design.yaml seed.e1_permutation (offset +1)
SEED_E2_BOOT = 20260424  # design.yaml seed.e2_bootstrap
SEED_E2_PERM = 20260426  # design.yaml seed.e2_permutation_length (offset +2)
SEED_E3_BOOT = 20260424  # design.yaml seed.e3_bootstrap
SEED_E3_MC = 20260427    # design.yaml seed.e3_mc_null (offset +3)
SEED_E4_BOOT = 20260424  # design.yaml seed.e4_bootstrap

COMMON_N_OFF_MHC = 16556  # F148a anchor (design.yaml)


# =============================================================================
# E1 — within-expression 4-bucket partition + ridge-α sweep
# WHY these regexes: brief_v2 §1 MEASUREMENT (L77-L82); design.yaml e1.partitions.
# =============================================================================
# Brain_human: human_brain* + named cortex regions + GTEx_brain (14 features
# currently in "other" category — re-categorize to brain per SynGO convention,
# brief_v2 L78).
E1_BRAIN_HUMAN_REGEX = [
    r"^human_brain\d*",
    r"^human_hippocampus",
    r"^human_brain_cerebellarhem",
    r"^human_brain_frontalcortex",
    r"^human_brain_visualcortex",
    r"^GTEx_brain",
]
E1_BRAIN_MOUSE_REGEX = [
    r"^mouse_brain\d*",
    r"^mouse_brain4_neurons",
    r"^mouse_brain4_bnst",
    r"^mouse_brain_projected_pcaloadings",
]
E1_IMMUNE_REGEX = [
    r"^ImmGen",
    r"_immune",
    r"_fetalblood",
    r"_myeloid",
    r"_tcell",
    r"_bcell",
    r"_thymus",
    r"_spleen",
]
# Fallback pattern for "other_non_brain" is "does not match any above AND
# classify_feature returned 'expression'". Handled in the partitioner below.

# Ridge-α sweep grid (brief_v2 L83; design.yaml e1.alpha_grid).
E1_ALPHA_GRID = [1000.0, 3162.0, 10000.0, 31623.0, 100000.0, 316230.0, 1000000.0]
E1_FITTED_ALPHA = 31623.0

# Reproduction gate R1: Δρ_expression combined ∈ [0.189, 0.213]
# (brief_v2 L85; design.yaml e1.reproduction_gate_r1).
E1_R1_TARGET = 0.201
E1_R1_TOLERANCE = 0.012
E1_R1_LO = 0.189
E1_R1_HI = 0.213

# Decision-rule thresholds (brief_v2 §1 DECISION RULE; design.yaml e1.decision_rule).
E1_BRAIN_HUMAN_DELTA_FLOOR = 0.10       # 2A
E1_BRAIN_HUMAN_DOMINANCE_GAP = 0.04     # 2A: must exceed max(immune, other) + 0.04
E1_MASS_SHARE_TOLERANCE = 0.20          # 2B: v2-widened
E1_ALPHA_RANGE_ALPHA_SPECIFIC = 0.08    # 2C: varies by > 0.08
E1_ALPHA_RANGE_MARGINAL = 0.04          # 2C_bis lower band
E1_POSITIVE_UNANTICIPATED_GAP = 0.03    # 2D_POS
E1_NEGATIVE_BRAIN_HUMAN_VS_MOUSE = 0.04  # 2D_NEG
E1_CI_OVERLAP_FLATNESS_MIN = 0.50       # flatness def: bootstrap CI overlap >= 50%
E1_MIN_PARTITION_FEATURES = 200         # UNINTERPRETABLE 2E floor


def partition_expression_features(
    feature_names: list[str],
    logger: logging.Logger | None = None,
) -> dict[str, list[str]]:
    """Partition PoPS expression features into 4 mutually-exclusive buckets.

    WHY: brief_v2 §1 MEASUREMENT (L77-L82) mandates the 4-bucket split
    {brain_human, brain_mouse, immune, other_non_brain}. Ordering of
    classification: brain_human first (SynGO convention + GTEx_brain
    re-categorization), then brain_mouse, then immune, then other.

    Precondition: this operates on features classified as `expression` by
    batch_058's `classify_feature` (batch_058 §Sub-B P0). BUT: brief_v2 L78
    mandates re-categorizing the 14 GTEx_brain.* features (which
    `classify_feature` returns as "other") into brain_human. We therefore
    widen the universe to ALSO include any feature name matching any of the
    4 E1 bucket regexes, regardless of `classify_feature`'s verdict. This
    matches the brief's "37,204 expression + 14 GTEx_brain re-categorized =
    37,218 features" assertion (brief_v2 L82).

    Returns dict[bucket_name -> list[feature_name]]. Buckets are mutually
    exclusive; union covers all expression features ∪ any GTEx_brain.*.
    """
    # Pre-compile (cheap; only 4 sets).
    bh = [re.compile(p) for p in E1_BRAIN_HUMAN_REGEX]
    bm = [re.compile(p) for p in E1_BRAIN_MOUSE_REGEX]
    im = [re.compile(p) for p in E1_IMMUNE_REGEX]

    out: dict[str, list[str]] = {
        "brain_human": [],
        "brain_mouse": [],
        "immune": [],
        "other_non_brain": [],
    }
    for n in feature_names:
        # The partition universe is expression ∪ any-bucket-regex-hit, so
        # GTEx_brain.* (which classify_feature calls "other") is
        # re-categorized to brain_human per brief_v2 L78.
        is_expression = classify_feature(n) == "expression"
        matches_bh = any(p.search(n) for p in bh)
        matches_bm = any(p.search(n) for p in bm)
        matches_im = any(p.search(n) for p in im)
        if not (is_expression or matches_bh or matches_bm or matches_im):
            continue
        if matches_bh:
            out["brain_human"].append(n)
        elif matches_bm:
            out["brain_mouse"].append(n)
        elif matches_im:
            out["immune"].append(n)
        else:
            out["other_non_brain"].append(n)

    if logger is not None:
        logger.info(
            "E1 partition counts: brain_human=%d brain_mouse=%d "
            "immune=%d other_non_brain=%d (total=%d)",
            len(out["brain_human"]), len(out["brain_mouse"]),
            len(out["immune"]), len(out["other_non_brain"]),
            sum(len(v) for v in out.values()),
        )

    # Assert mutually exclusive (defensive — regex could overlap).
    seen: set[str] = set()
    for k, names in out.items():
        for nm in names:
            if nm in seen:
                raise RuntimeError(
                    f"E1 partition not mutually exclusive: {nm} in {k} and prior"
                )
            seen.add(nm)
    return out


# =============================================================================
# E2 — EDT1 SynGO-only bisection + polynomial/rank length covariates
# WHY these GO IDs: brief_v2 §2 (L113-L115).
# =============================================================================
# SCAFFOLD_CORE_PRIMARY = GO:0098839 ∪ GO:0014069
E2_GO_SCAFFOLD = ["GO:0098839", "GO:0014069"]
# VESICLE_CORE_PRIMARY = GO:0008021 ∪ GO:0045202 (presynaptic subset)
# WHY full GO:0045202 here: the 2024 GMT does not sub-split synapse by
# pre/postsynaptic annotation tag. We take the full set and note that the
# REMAINING ring control absorbs any mis-classification.
E2_GO_VESICLE = ["GO:0008021", "GO:0045202"]

# Fernández 2009 TAP-MS hub list (brief_v2 §2 SCAFFOLD_CORE_SECONDARY).
# DESCRIPTIVE ONLY — not used in the decision rule.
# WHY these genes: brief_v2 L116 lists them verbatim.
E2_FERNANDEZ_2009_HUBS = [
    "DLG1", "DLG2", "DLG3", "DLG4",
    "SHANK1", "SHANK2", "SHANK3",
    "HOMER1", "HOMER2", "HOMER3",
    "DLGAP1", "DLGAP2", "DLGAP3", "DLGAP4",
    "GRIN2A", "GRIN2B",
    "SYNGAP1",
    "CAMK2A", "CAMK2B",
    "ARC",
    "NLGN1", "NLGN2", "NLGN3", "NLGN4",
    "NRXN1", "NRXN2", "NRXN3",
    "IQSEC1",
    "BAIAP2",
    "PICK1",
]

# UNINTERPRETABLE floors (brief_v2 §2 + design.yaml e2.rings).
E2_MIN_N_SCAFFOLD = 20    # v2 fix (was n<15 in v1)
E2_MIN_N_VESICLE = 20     # v2 fix (was n<15 in v1)
E2_MIN_N_REMAINING = 30   # below this, residual-pool comparison unreliable

# Covariate specifications (brief_v2 §2 MEASUREMENT; design.yaml e2.covariate_specs).
E2_COVS_V1_LINEAR = ["log10_gene_length", "lof_pLI",
                     "log10_exp_lof_plus1", "log10_NSNPS_plus1"]
# V2_POLY adds log10_length^2 and log10_length^3.
E2_COVS_V2_POLY = [
    "log10_gene_length", "lof_pLI",
    "log10_exp_lof_plus1", "log10_NSNPS_plus1",
    "log10_gene_length_sq", "log10_gene_length_cu",
]
# V3_RANK adds rank-percentile of length.
E2_COVS_V3_RANK = [
    "log10_gene_length", "lof_pLI",
    "log10_exp_lof_plus1", "log10_NSNPS_plus1",
    "rank_pct_length",
]

# Length-decile-matched permutation null (brief_v2 §2 MEASUREMENT;
# design.yaml e2.length_matched_null).
E2_LENGTH_N_DECILES = 10
E2_LENGTH_N_DRAWS = 10000
E2_LENGTH_P_THRESHOLD = 0.01

# Reproduction gate R1 (brief_v2 §2; design.yaml e2.reproduction_gate_r1).
# EDT1-ex-B3 SCZ β_OLS under V1 ∈ [3.0, 3.8]; anchor F058_05 +3.43.
E2_R1_TARGET = 3.43
E2_R1_LO = 3.0
E2_R1_HI = 3.8

# Decision-rule thresholds (brief_v2 §2 DECISION RULE; design.yaml).
E2_SCAFFOLD_CONCENTRATED_GAP = 2.0     # σ-units
E2_SCAFFOLD_INTERMEDIATE_LO = 1.0
E2_SCAFFOLD_INTERMEDIATE_HI = 2.0
E2_LENGTH_ARTIFACT_REL_DROP = 0.50      # |β_v3 - β_v1| > 0.50 * |β_v1|
E2_LENGTH_ARTIFACT_EFFECT_MAX = 0.10    # AND |β_v3| < 0.10
E2_LENGTH_MASKING_REL_RISE = 0.30       # β_v3 > β_v1 + 0.30*|β_v1|


def load_syngo_go_terms(go_id_list: list[str],
                          logger: logging.Logger | None = None
                          ) -> set[str]:
    """Load the union of gene-symbols annotated to any GO ID in `go_id_list`.

    WHY: brief_v2 §2 (L113-L114) defines SCAFFOLD_CORE_PRIMARY and
    VESICLE_CORE_PRIMARY as unions over specific GO IDs from the SynGO 2024
    ontologies file.

    SynGO 2024 GMT layout (verified by head/grep inspection at implementation
    time): each line is `{name} (GO:xxxxxxx) {BP|CC|MF}\\t\\t{gene1}\\t{gene2}...`
    where `name` is human-readable, the GO ID is in parens, and genes start
    at split-index 2 (after two tab-separated description fields).

    Returns a set of HGNC-like gene symbols (uppercase convention). Callers
    must subsequently map symbols → ENSGID via `symbols_to_ensgids`.

    Raises FileNotFoundError if SYNGO_GMT missing, or RuntimeError if any
    requested GO ID is not found in the file (Cardinal Rule 0: no silent
    empty sets).
    """
    if not SYNGO_GMT.exists():
        raise FileNotFoundError(f"SynGO GMT missing: {SYNGO_GMT}")
    # Compile GO-ID matcher. GO IDs appear in the first field like
    # `... (GO:0098839) CC`.
    want = set(go_id_list)
    found_ids: set[str] = set()
    union: set[str] = set()
    go_re = re.compile(r"\(GO:\d+\)")
    with SYNGO_GMT.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            header_field = parts[0]
            m = go_re.search(header_field)
            if not m:
                continue
            # Strip "()" to get pure GO ID string.
            go_id = m.group(0).strip("()")
            if go_id not in want:
                continue
            found_ids.add(go_id)
            for g in parts[2:]:
                g = g.strip()
                if g:
                    union.add(g)
    missing = want - found_ids
    if missing:
        raise RuntimeError(
            f"SynGO GO IDs not found in {SYNGO_GMT}: {sorted(missing)}"
        )
    if logger is not None:
        logger.info("load_syngo_go_terms: GO IDs=%s → n_symbols=%d",
                     sorted(want), len(union))
    return union


# =============================================================================
# E3 — Tail-below-null diagnostic (Thorndike closed-form)
# WHY these constants: brief_v2 §3 MEASUREMENT (L186).
# =============================================================================
# Thorndike Case II single-truncation closed-form (pre-computed; verified by
# scipy: z=norm.ppf(0.80)=0.8416; λ(z)=φ(z)/(1-Φ(z))=1.3998;
# var_trunc=1-λ·(λ-z)=0.2186; s_S=√var_trunc=0.4676;
# r_trunc = R·s_S / √(1 - R² + R²·var_trunc) = 0.2705 at R=0.515).
E3_THORNDIKE_EXPECTED_SINGLE_TRUNC = 0.2705
E3_THORNDIKE_Z_TAU = 0.8416    # norm.ppf(0.80)
E3_THORNDIKE_LAMBDA = 1.3998
E3_THORNDIKE_VAR_TRUNC = 0.2186
E3_THORNDIKE_S_S = 0.4676
E3_FULL_RHO_ANCHOR = 0.515     # F148a off-MHC anchor (design.yaml e3.full_rho_anchor)
E3_TRUNCATION_QUANTILE = 0.80  # norm.ppf(0.80) = 0.8416

# iter_058 MC anchor (brief_v2 §3; design.yaml e3.null_integrity.iter058_mc_null).
E3_ITER058_MC_NULL = 0.270
E3_MC_AGREEMENT_THRESHOLD = 0.01

# Reproduction gate R1 (brief_v2 §3; design.yaml e3.reproduction_gate_r1).
E3_R1_TARGET = 0.229
E3_R1_TOLERANCE = 0.02
E3_R1_LO = 0.21
E3_R1_HI = 0.25

# Mechanism thresholds (brief_v2 §3 DECISION RULE).
E3_3A_KURTOSIS_UPPER_THRESHOLD = -0.5       # fires if kurt(PoPS | upper) <= -0.5
E3_3A_MIDDLE_KURTOSIS_FLOOR = -0.2          # middle bin kurt > -0.2 (not platykurtic)
E3_3A_MIDDLE_QUANTILE_LO = 0.40             # middle bin [0.4, 0.6]
E3_3A_MIDDLE_QUANTILE_HI = 0.60
E3_3B_DELTA_FLOOR = 0.02                    # Δ ≥ max(0.02, 2·SE_bootstrap)
E3_3B_SE_MULTIPLIER = 2.0
# MHC region (GRCh37, NCBI37.3 gene-loc coordinates, brief_v2 L188/design.yaml).
E3_MHC_CHR = 6
E3_MHC_START = 25_000_000
E3_MHC_END = 34_000_000
E3_3C_DELTA_THRESHOLD = -0.03               # fires if ≤ -0.03 below Thorndike
E3_3C_MIN_DISORDERS_FIRING = 2              # need ≥ 2 of {ibd, bip, height}


def thorndike_case_ii_single_trunc(
    full_rho: float, truncation_quantile: float = 0.80,
) -> dict:
    """Thorndike Case II single-truncation closed-form expected tail-ρ.

    WHY closed-form: brief_v2 §3 RESOLVED (v2 changelog C2) mandates a
    closed-form Thorndike prediction alongside MC verification. This function
    returns the analytical expected tail-ρ conditional on truncating on one
    variable (MAGMA-Z, per iter_058 Sub-B.2 line 606).

    Formula (Thorndike 1949 Case II):
        z = norm.ppf(truncation_quantile)     # standard normal quantile
        λ(z) = φ(z) / (1 - Φ(z))              # inverse Mills ratio
        var_trunc = 1 - λ(z) · (λ(z) - z)     # variance after truncation
        s_S = sqrt(var_trunc)                  # SD of selection variable after trunc
        r_trunc = R · s_S / sqrt(1 - R² + R² · var_trunc)

    Returns dict with all intermediate constants for auditor traceability.

    WHY numpy/scipy: Cardinal Rule 1 (no hand-roll norm.pdf/cdf).
    """
    from scipy.stats import norm
    z_tau = float(norm.ppf(truncation_quantile))
    lam = float(norm.pdf(z_tau) / (1.0 - norm.cdf(z_tau)))
    var_trunc = 1.0 - lam * (lam - z_tau)
    s_S = float(np.sqrt(var_trunc))
    R = float(full_rho)
    denom = float(np.sqrt(1.0 - R ** 2 + R ** 2 * var_trunc))
    r_trunc = R * s_S / denom
    return {
        "full_rho_R": R,
        "truncation_quantile": float(truncation_quantile),
        "z_tau": z_tau,
        "lambda_inverse_mills": lam,
        "var_trunc": float(var_trunc),
        "s_S": s_S,
        "expected_tail_rho": float(r_trunc),
    }


# =============================================================================
# E4 — Bellenguez 2022 AD independent-cohort replication
# WHY these paths: design.yaml e4 + brief_v2 §4 (L233-L236).
# =============================================================================
BELLENGUEZ_SUMSTATS = (
    PROJECT_ROOT / "data" / "bellenguez_ad_2022"
    / "GCST90027158_buildGRCh38.tsv.gz"
)
MAGMA_BIN = PROJECT_ROOT / "tools" / "magma_bin" / "magma"
MAGMA_GENELOC_GRCH37 = PROJECT_ROOT / "tools" / "magma_bin" / "refs" / "NCBI37.3.gene.loc"
MAGMA_GENELOC_GRCH38 = PROJECT_ROOT / "tools" / "magma_bin" / "refs" / "NCBI38.gene.loc"
MAGMA_1000G_EUR = PROJECT_ROOT / "tools" / "magma_bin" / "g1000_eur" / "g1000_eur"

# Existing Alzheimers pipeline MAGMA (Jansen-like) for comparison.
EXISTING_ALZHEIMERS_MAGMA = (
    BATCH055B_WORK / "alzheimers" / "full.gene.genes.out"
)

# Walltime cap (design.yaml e4.max_wall_min).
E4_MAX_WALL_MIN = 90
E4_WALL_CAP_SECONDS = 60 * E4_MAX_WALL_MIN

# QC gates (design.yaml e4).
E4_SNP_MATCH_RATE_MIN = 0.90
E4_GENE_Z_SPEARMAN_MIN = 0.6  # min Spearman ρ to existing AD pipeline
E4_ENSG_OVERLAP_MIN = 0.90    # Rule 1 consistency check

# Reproduction gate R1 (brief_v2 §4; design.yaml e4.reproduction_gate_r1).
# B3 SCZ β unchanged at +3.24 ± 0.15 → [3.09, 3.39]. Uses existing pipeline;
# this is a pipeline-regression gate, not a new computation.
E4_R1_TARGET = 3.24
E4_R1_TOLERANCE = 0.15
E4_R1_LO = 3.09
E4_R1_HI = 3.39

# AD decision thresholds (brief_v2 §4 + design.yaml e4.decision_rule).
# iter_058 F058_06 anchor AD β=+0.31.
E4_AD_REPLICATED_LO = 0.20   # inclusive
E4_AD_REPLICATED_HI = 0.42   # inclusive (boundary-closed per v2 M5)
E4_AD_HYPER_LO = 0.80        # strict >
E4_AD_INCONSISTENT_LO_BAND = (0.08, 0.20)   # (exclusive, exclusive)
E4_AD_INCONSISTENT_HI_BAND = (0.42, 0.80)   # (exclusive, inclusive)
E4_AD_NOT_REPLICATED_ABS_CAP = 0.08  # |β| ≤ 0.08
E4_AD_WRONG_DIRECTION_CAP = -0.08    # β < -0.08

# MAGMA hyperparameters (from batch_055_B; brief_v2 §4 L235).
MAGMA_ANNOT_WINDOW_UP_KB = 35
MAGMA_ANNOT_WINDOW_DOWN_KB = 10
MAGMA_GENE_MODEL = "snp-wise=mean"

# Bellenguez 2022 Nature Genetics reported effective N.
# WHY: Bellenguez C et al. (2022) "New insights into the genetic etiology of
# Alzheimer's disease and related dementias" Nature Genetics 54:412-436.
# Meta-analysis reports N=111,326 AD cases + 677,663 controls (total 788,989).
# Effective N = 4 / (1/N_case + 1/N_ctrl) = 4 / (1/111326 + 1/677663) ≈ 382,188.
# Citation: [lit_doi_10.1038_s41588-022-01024-z]. Using paper-reported value
# avoids per-row median (MAJOR M3 fix — medians of per-SNP n_cases/n_controls
# can vary with filtering; paper value is canonical).
N_EFFECTIVE_BELLENGUEZ_2022 = 382188
N_CASES_BELLENGUEZ_2022 = 111326
N_CONTROLS_BELLENGUEZ_2022 = 677663


# =============================================================================
# Atomic write shortcut for non-JSON (e.g., .tsv)
# =============================================================================
def atomic_write_text(text: str, path: Path) -> None:
    """Write text atomically via `.tmp` → rename.

    WHY: our atomic_write_json only handles dicts. For TSV outputs we need
    the same atomic pattern.
    """
    import os as _os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        fh.write(text)
    _os.replace(tmp, path)


# =============================================================================
# Exported symbols
# =============================================================================
__all__ = [
    # Paths
    "PROJECT_ROOT", "BATCH_DIR", "OUTPUT_DIR", "LOGS_DIR", "SCRIPTS_DIR",
    "POPS_FEATURE_CATEGORIES_TSV", "POPS_FEATURES_MUNGED_DIR",
    "POPS_FEATURES_ROWS_TXT", "POPS_COEFS_P05", "BATCH054_P05_PREDS",
    "BATCH055B_WORK", "GENE_ANNOT", "GNOMAD_TSV", "MAGMA_GENELOC",
    "MAGMA_SCZ_GENES_OUT", "PGC3_XLSX", "SYNGO_GMT",
    "BELLENGUEZ_SUMSTATS", "MAGMA_BIN", "MAGMA_GENELOC_GRCH37",
    "MAGMA_GENELOC_GRCH38", "MAGMA_1000G_EUR", "EXISTING_ALZHEIMERS_MAGMA",
    # Re-exports
    "B3_BIOLOGICAL_CATEGORY", "B3_GENES", "BH_Q", "BOOTSTRAP_N",
    "BOOTSTRAP_SEED", "REPRO_TOLERANCE",
    "REPRO_R1_SUB_A_LO", "REPRO_R1_SUB_A_HI",
    "REPRO_R1_SUB_C_FULL_LO", "REPRO_R1_SUB_C_FULL_HI",
    "REPRO_R2_RHO_LO", "REPRO_R2_RHO_HI",
    "CI_UPPER_STRONG", "DFBETAS_CUTOFF", "FRAGILE_DIFF_THRESHOLD",
    "MDE_80_POWER_SUB_A", "SHAPIRO_ALPHA",
    "SUB_B_BOOT_N", "SUB_B_CATEGORY_DELTA_FLOOR", "SUB_B_DISTRIBUTION_DIFF",
    "SUB_B_LS_MC_N", "SUB_B_LS_NULL_RHO", "SUB_B_LS_NULL_TAU",
    "SUB_B_OTHER_MASS_GATE", "SUB_B_PERM_N", "SUB_B_PERM_PVALUE",
    "SUB_B_SEED", "SUB_B_TAIL_ENRICHED_DIFF",
    "SUB_C_FRAGILE_UNINTERPRETABLE_COUNT", "SUB_C_MAPPING_GATE",
    "SUB_C_MIN_EFFECT",
    "DISORDERS", "LOEUF_DISORDERS", "PSYCHIATRIC",
    # Helpers (re-exported)
    "abs_diff_huber_check", "aggregate_pattern_sub_a",
    "aggregate_pattern_sub_c", "atomic_write_json", "bh_fdr",
    "build_bootstrap_idx", "build_sub_a_frame",
    "categorize_pops_features", "classify_disorder_v2", "classify_feature",
    "compute_dfbetas_cooks", "fit_tukey_biweight",
    "load_common_ensgids", "load_edt1", "load_gene_annot",
    "load_gnomad_per_brief_v2", "load_koopmans_ex_B3",
    "load_koopmans_full_symbols", "load_magma_disorder", "load_magma_scz",
    "load_nsnps_per_disorder", "load_pops_coefs",
    "load_pops_features_matrix", "load_preds", "longin_solnik_mc_null",
    "partial_pearson", "percentile_ci", "rank_gaussianize",
    "rel_diff_huber_check", "sha256_file", "symbols_to_ensgids",
    "setup_logger",
    # Seeds
    "SEED_MASTER", "SEED_E1_BOOT", "SEED_E1_PERM",
    "SEED_E2_BOOT", "SEED_E2_PERM", "SEED_E3_BOOT", "SEED_E3_MC",
    "SEED_E4_BOOT", "COMMON_N_OFF_MHC",
    # E1
    "E1_BRAIN_HUMAN_REGEX", "E1_BRAIN_MOUSE_REGEX", "E1_IMMUNE_REGEX",
    "E1_ALPHA_GRID", "E1_FITTED_ALPHA",
    "E1_R1_TARGET", "E1_R1_TOLERANCE", "E1_R1_LO", "E1_R1_HI",
    "E1_BRAIN_HUMAN_DELTA_FLOOR", "E1_BRAIN_HUMAN_DOMINANCE_GAP",
    "E1_MASS_SHARE_TOLERANCE", "E1_ALPHA_RANGE_ALPHA_SPECIFIC",
    "E1_ALPHA_RANGE_MARGINAL", "E1_POSITIVE_UNANTICIPATED_GAP",
    "E1_NEGATIVE_BRAIN_HUMAN_VS_MOUSE",
    "E1_CI_OVERLAP_FLATNESS_MIN", "E1_MIN_PARTITION_FEATURES",
    "partition_expression_features",
    # E2
    "E2_GO_SCAFFOLD", "E2_GO_VESICLE", "E2_FERNANDEZ_2009_HUBS",
    "E2_MIN_N_SCAFFOLD", "E2_MIN_N_VESICLE", "E2_MIN_N_REMAINING",
    "E2_COVS_V1_LINEAR", "E2_COVS_V2_POLY", "E2_COVS_V3_RANK",
    "E2_LENGTH_N_DECILES", "E2_LENGTH_N_DRAWS", "E2_LENGTH_P_THRESHOLD",
    "E2_R1_TARGET", "E2_R1_LO", "E2_R1_HI",
    "E2_SCAFFOLD_CONCENTRATED_GAP", "E2_SCAFFOLD_INTERMEDIATE_LO",
    "E2_SCAFFOLD_INTERMEDIATE_HI", "E2_LENGTH_ARTIFACT_REL_DROP",
    "E2_LENGTH_ARTIFACT_EFFECT_MAX", "E2_LENGTH_MASKING_REL_RISE",
    "load_syngo_go_terms",
    # E3
    "E3_THORNDIKE_EXPECTED_SINGLE_TRUNC", "E3_THORNDIKE_Z_TAU",
    "E3_THORNDIKE_LAMBDA", "E3_THORNDIKE_VAR_TRUNC",
    "E3_THORNDIKE_S_S", "E3_FULL_RHO_ANCHOR",
    "E3_TRUNCATION_QUANTILE", "E3_ITER058_MC_NULL",
    "E3_MC_AGREEMENT_THRESHOLD",
    "E3_R1_TARGET", "E3_R1_TOLERANCE", "E3_R1_LO", "E3_R1_HI",
    "E3_3A_KURTOSIS_UPPER_THRESHOLD", "E3_3A_MIDDLE_KURTOSIS_FLOOR",
    "E3_3A_MIDDLE_QUANTILE_LO", "E3_3A_MIDDLE_QUANTILE_HI",
    "E3_3B_DELTA_FLOOR", "E3_3B_SE_MULTIPLIER",
    "E3_MHC_CHR", "E3_MHC_START", "E3_MHC_END",
    "E3_3C_DELTA_THRESHOLD", "E3_3C_MIN_DISORDERS_FIRING",
    "thorndike_case_ii_single_trunc",
    # E4
    "E4_MAX_WALL_MIN", "E4_WALL_CAP_SECONDS",
    "E4_SNP_MATCH_RATE_MIN", "E4_GENE_Z_SPEARMAN_MIN",
    "E4_ENSG_OVERLAP_MIN",
    "E4_R1_TARGET", "E4_R1_TOLERANCE", "E4_R1_LO", "E4_R1_HI",
    "E4_AD_REPLICATED_LO", "E4_AD_REPLICATED_HI",
    "E4_AD_HYPER_LO", "E4_AD_INCONSISTENT_LO_BAND",
    "E4_AD_INCONSISTENT_HI_BAND", "E4_AD_NOT_REPLICATED_ABS_CAP",
    "E4_AD_WRONG_DIRECTION_CAP",
    "MAGMA_ANNOT_WINDOW_UP_KB", "MAGMA_ANNOT_WINDOW_DOWN_KB",
    "MAGMA_GENE_MODEL",
    "N_EFFECTIVE_BELLENGUEZ_2022",
    "N_CASES_BELLENGUEZ_2022", "N_CONTROLS_BELLENGUEZ_2022",
    # New helpers
    "atomic_write_text",
]
