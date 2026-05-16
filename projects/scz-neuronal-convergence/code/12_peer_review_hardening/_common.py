"""Shared helpers for batch_060 sub-experiments (iter_060 brief_v2).

WHY this module exists: Cardinal Rule 1. Re-export batch_059/_common (which
transitively re-exports batch_058/_common, batch_057, batch_056). All loaders
and helpers already exist; we only add iter_060-specific constants (E1
relaxed QC gate, E2 permutation/bootstrap parameters, E3 pLI null replay
parameters) and batch_060-scoped paths.

Determinism: all seeds sourced from design.yaml. Master seed = 20260424.

No fabrication (Cardinal Rule 0): every loader raises FileNotFoundError with
a clear diagnostic on missing inputs. SHA256 provenance via re-exported
`sha256_file`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Absolute paths (agent cwd resets between calls).
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_060"
OUTPUT_DIR = BATCH_DIR / "output"
LOGS_DIR = BATCH_DIR / "logs"
SCRIPTS_DIR = BATCH_DIR / "scripts"

# Upstream batch directories used by carry-over experiments.
BATCH_059_DIR = PROJECT_ROOT / "experiments" / "batch_059"
BATCH_059_OUTPUT = BATCH_059_DIR / "output"

# -----------------------------------------------------------------------------
# Re-export batch_059/_common via importlib (same pattern used by batch_059
# re-exporting batch_058, which re-exports batch_057 etc.).
# WHY importlib: all ancestor files are ALSO named `_common.py`; sys.path
# tricks would shadow THIS module.
# -----------------------------------------------------------------------------
import importlib.util as _ilu

_BATCH059_COMMON_PATH = (
    PROJECT_ROOT / "experiments" / "batch_059" / "scripts" / "_common.py"
)
if not _BATCH059_COMMON_PATH.exists():
    raise FileNotFoundError(
        f"batch_059 _common.py missing: {_BATCH059_COMMON_PATH}"
    )
_spec = _ilu.spec_from_file_location(
    "batch059_common", str(_BATCH059_COMMON_PATH)
)
_batch059_common = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_batch059_common)

# Re-export everything needed from batch_059/_common (which transitively
# re-exports batch_058/_common and earlier).
# Paths
B3_GENES = _batch059_common.B3_GENES
BATCH054_P05_PREDS = _batch059_common.BATCH054_P05_PREDS
BATCH055B_WORK = _batch059_common.BATCH055B_WORK
GENE_ANNOT = _batch059_common.GENE_ANNOT
GNOMAD_TSV = _batch059_common.GNOMAD_TSV
MAGMA_GENELOC = _batch059_common.MAGMA_GENELOC
MAGMA_SCZ_GENES_OUT = _batch059_common.MAGMA_SCZ_GENES_OUT
PGC3_XLSX = _batch059_common.PGC3_XLSX
POPS_COEFS_P05 = _batch059_common.POPS_COEFS_P05
POPS_FEATURES_MUNGED_DIR = _batch059_common.POPS_FEATURES_MUNGED_DIR
SYNGO_GMT = _batch059_common.SYNGO_GMT
EXISTING_ALZHEIMERS_MAGMA = _batch059_common.EXISTING_ALZHEIMERS_MAGMA

# Constants
BH_Q = _batch059_common.BH_Q
BOOTSTRAP_N = _batch059_common.BOOTSTRAP_N
BOOTSTRAP_SEED = _batch059_common.BOOTSTRAP_SEED
COMMON_N_OFF_MHC = _batch059_common.COMMON_N_OFF_MHC
DISORDERS = _batch059_common.DISORDERS
LOEUF_DISORDERS = _batch059_common.LOEUF_DISORDERS
PSYCHIATRIC = _batch059_common.PSYCHIATRIC
REPRO_TOLERANCE = _batch059_common.REPRO_TOLERANCE

# Sub-A/Sub-B/Sub-C constants
CI_UPPER_STRONG = _batch059_common.CI_UPPER_STRONG
DFBETAS_CUTOFF = _batch059_common.DFBETAS_CUTOFF
FRAGILE_DIFF_THRESHOLD = _batch059_common.FRAGILE_DIFF_THRESHOLD
MDE_80_POWER_SUB_A = _batch059_common.MDE_80_POWER_SUB_A
SHAPIRO_ALPHA = _batch059_common.SHAPIRO_ALPHA
SUB_B_BOOT_N = _batch059_common.SUB_B_BOOT_N
SUB_B_PERM_N = _batch059_common.SUB_B_PERM_N
SUB_B_SEED = _batch059_common.SUB_B_SEED

# Reproduction gate anchors
REPRO_R1_SUB_A_LO = _batch059_common.REPRO_R1_SUB_A_LO
REPRO_R1_SUB_A_HI = _batch059_common.REPRO_R1_SUB_A_HI

# Helpers (re-exported)
abs_diff_huber_check = _batch059_common.abs_diff_huber_check
aggregate_pattern_sub_a = _batch059_common.aggregate_pattern_sub_a
atomic_write_json = _batch059_common.atomic_write_json
atomic_write_text = _batch059_common.atomic_write_text
bh_fdr = _batch059_common.bh_fdr
build_bootstrap_idx = _batch059_common.build_bootstrap_idx
build_sub_a_frame = _batch059_common.build_sub_a_frame
categorize_pops_features = _batch059_common.categorize_pops_features
classify_disorder_v2 = _batch059_common.classify_disorder_v2
classify_feature = _batch059_common.classify_feature
compute_dfbetas_cooks = _batch059_common.compute_dfbetas_cooks
fit_tukey_biweight = _batch059_common.fit_tukey_biweight
load_common_ensgids = _batch059_common.load_common_ensgids
load_edt1 = _batch059_common.load_edt1
load_gene_annot = _batch059_common.load_gene_annot
load_gnomad_per_brief_v2 = _batch059_common.load_gnomad_per_brief_v2
load_koopmans_ex_B3 = _batch059_common.load_koopmans_ex_B3
load_koopmans_full_symbols = _batch059_common.load_koopmans_full_symbols
load_magma_disorder = _batch059_common.load_magma_disorder
load_magma_scz = _batch059_common.load_magma_scz
load_nsnps_per_disorder = _batch059_common.load_nsnps_per_disorder
load_pops_coefs = _batch059_common.load_pops_coefs
load_pops_features_matrix = _batch059_common.load_pops_features_matrix
load_preds = _batch059_common.load_preds
partial_pearson = _batch059_common.partial_pearson
percentile_ci = _batch059_common.percentile_ci
rank_gaussianize = _batch059_common.rank_gaussianize
sha256_file = _batch059_common.sha256_file
symbols_to_ensgids = _batch059_common.symbols_to_ensgids

# batch_059 E1-specific functions/constants needed by batch_060 E2
partition_expression_features = _batch059_common.partition_expression_features
E1_BRAIN_HUMAN_REGEX = _batch059_common.E1_BRAIN_HUMAN_REGEX
E1_FITTED_ALPHA = _batch059_common.E1_FITTED_ALPHA

# batch_059 E2-specific constants needed by batch_060 E3
E2_COVS_V1_LINEAR = _batch059_common.E2_COVS_V1_LINEAR
E2_LENGTH_N_DECILES = _batch059_common.E2_LENGTH_N_DECILES
E2_LENGTH_N_DRAWS = _batch059_common.E2_LENGTH_N_DRAWS

# batch_059 seeds (used by E2/E3 for replay)
SEED_MASTER = _batch059_common.SEED_MASTER
SEED_E1_BOOT = _batch059_common.SEED_E1_BOOT
SEED_E1_PERM = _batch059_common.SEED_E1_PERM
SEED_E2_BOOT = _batch059_common.SEED_E2_BOOT
SEED_E2_PERM = _batch059_common.SEED_E2_PERM


# =============================================================================
# batch_060-scoped logger (writes to batch_060/logs)
# =============================================================================
def setup_logger(name: str, logfile: Path) -> logging.Logger:
    """Logger emitting to `logfile` and stdout.

    WHY we override: keep batch_060 logs isolated under batch_060/logs for
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
# Seeds (design.yaml; master=20260424)
# =============================================================================
B060_SEED_MASTER = 20260424       # design.yaml seed
B060_SEED_E1 = 20260424           # E1 Bellenguez rerun
B060_SEED_E2_PERM = 20260425      # E2 permutation null (offset +1)
B060_SEED_E2_BOOT = 20260426      # E2 bootstrap (offset +2)
B060_SEED_E3 = 20260427           # E3 pLI confound (offset +3)

# =============================================================================
# E1 — Bellenguez AD rerun with relaxed QC gate
# WHY relaxed: brief_v2 §E1. iter_059 E4 produced rho=0.306 which failed
# the old gate of 0.60. L059_01 established rho~0.25-0.50 is expected for
# cross-generation GWAS. We relax to 0.25.
# =============================================================================
E1_QC_RHO_MIN = 0.25              # brief_v2 §E1: relaxed from 0.80
E1_TOP_K_SPOTCHECK = 20           # brief_v2 §E1: rank by |MAGMA_Z|
E1_TOP_K_OVERLAP_MIN = 10         # brief_v2 §E1: require >=10 overlap
E1_EXISTING_AD_TOP_K = 40         # compare Bellenguez top-20 vs existing top-40
# Bellenguez gene-Z from batch_059 E4 pipeline output.
E1_BELLENGUEZ_GENE_Z_TSV = BATCH_059_OUTPUT / "e4" / "magma_gene_z.tsv"

# Decision thresholds (brief_v2 §E1 DECISION RULE).
E1_AD_REPLICATED_LO = 0.10        # inclusive
E1_AD_REPLICATED_HI = 0.50        # inclusive
E1_AD_STRONGER_LO = 0.50          # exclusive (beta > 0.50)
E1_AD_NOT_REPLICATED_CAP = 0.10   # beta < 0.10

# Reproduction gate: B3 SCZ beta_OLS in [+2.5, +3.5] (brief_v2 §Shared)
E1_R1_LO = 2.5
E1_R1_HI = 3.5

# =============================================================================
# E2 — Joint ablation interaction test
# WHY: brief_v2 §E2. F059_02 found 35% non-additivity. This tests whether
# brain_human + other_non_brain interact beyond simple sum.
# =============================================================================
E2_N_PERMUTATIONS = 1000          # brief_v2 §E2: 1000 permutations
E2_N_BOOTSTRAP = 1000             # brief_v2 §E2: 1000 bootstrap
E2_FULL_MODEL_RHO_ANCHOR = 0.510  # brief_v2 §E2: UNINTERPRETABLE if deviates >0.02

# =============================================================================
# E3 — pLI confound check
# WHY: brief_v2 §E3. Length-matched null from E2 does not control for
# constraint (pLI). We measure pLI distribution of null draws.
# =============================================================================
E3_N_NULL_DRAWS = 10000           # brief_v2 §E3: replay 10,000 draws
E3_PLI_ADEQUATE_FLOOR = 0.30      # brief_v2 §E3: mean_pLI_null > 0.30
E3_PLI_PARTIAL_FLOOR = 0.20       # brief_v2 §E3: mean_pLI_null in [0.20, 0.30]
E3_BIAS_INCONSEQUENTIAL = 0.10    # brief_v2 §E3: bias < 10% of observed beta
E3_BIAS_INADEQUATE = 0.20         # brief_v2 §E3: bias > 20% of observed beta


# =============================================================================
# Exported symbols
# =============================================================================
__all__ = [
    # Paths
    "PROJECT_ROOT", "BATCH_DIR", "OUTPUT_DIR", "LOGS_DIR", "SCRIPTS_DIR",
    "BATCH_059_DIR", "BATCH_059_OUTPUT",
    "GENE_ANNOT", "GNOMAD_TSV", "MAGMA_GENELOC", "MAGMA_SCZ_GENES_OUT",
    "PGC3_XLSX", "POPS_COEFS_P05", "POPS_FEATURES_MUNGED_DIR",
    "BATCH054_P05_PREDS", "BATCH055B_WORK", "EXISTING_ALZHEIMERS_MAGMA",
    "SYNGO_GMT",
    # Constants
    "B3_GENES", "BH_Q", "BOOTSTRAP_N", "BOOTSTRAP_SEED",
    "COMMON_N_OFF_MHC", "DISORDERS", "LOEUF_DISORDERS", "PSYCHIATRIC",
    "REPRO_TOLERANCE",
    "CI_UPPER_STRONG", "DFBETAS_CUTOFF", "FRAGILE_DIFF_THRESHOLD",
    "MDE_80_POWER_SUB_A", "SHAPIRO_ALPHA",
    "SUB_B_BOOT_N", "SUB_B_PERM_N", "SUB_B_SEED",
    "REPRO_R1_SUB_A_LO", "REPRO_R1_SUB_A_HI",
    # Re-exported helpers
    "abs_diff_huber_check", "aggregate_pattern_sub_a",
    "atomic_write_json", "atomic_write_text", "bh_fdr",
    "build_bootstrap_idx", "build_sub_a_frame",
    "categorize_pops_features", "classify_disorder_v2", "classify_feature",
    "compute_dfbetas_cooks", "fit_tukey_biweight",
    "load_common_ensgids", "load_edt1", "load_gene_annot",
    "load_gnomad_per_brief_v2", "load_koopmans_ex_B3",
    "load_koopmans_full_symbols", "load_magma_disorder", "load_magma_scz",
    "load_nsnps_per_disorder", "load_pops_coefs", "load_pops_features_matrix",
    "load_preds", "partial_pearson", "percentile_ci", "rank_gaussianize",
    "sha256_file", "symbols_to_ensgids",
    "partition_expression_features",
    "E1_BRAIN_HUMAN_REGEX", "E1_FITTED_ALPHA",
    "E2_COVS_V1_LINEAR", "E2_LENGTH_N_DECILES", "E2_LENGTH_N_DRAWS",
    "SEED_MASTER", "SEED_E1_BOOT", "SEED_E1_PERM",
    "SEED_E2_BOOT", "SEED_E2_PERM",
    "setup_logger",
    # batch_060 seeds
    "B060_SEED_MASTER", "B060_SEED_E1", "B060_SEED_E2_PERM",
    "B060_SEED_E2_BOOT", "B060_SEED_E3",
    # E1 constants
    "E1_QC_RHO_MIN", "E1_TOP_K_SPOTCHECK", "E1_TOP_K_OVERLAP_MIN",
    "E1_EXISTING_AD_TOP_K", "E1_BELLENGUEZ_GENE_Z_TSV",
    "E1_AD_REPLICATED_LO", "E1_AD_REPLICATED_HI",
    "E1_AD_STRONGER_LO", "E1_AD_NOT_REPLICATED_CAP",
    "E1_R1_LO", "E1_R1_HI",
    # E2 constants
    "E2_N_PERMUTATIONS", "E2_N_BOOTSTRAP", "E2_FULL_MODEL_RHO_ANCHOR",
    # E3 constants
    "E3_N_NULL_DRAWS", "E3_PLI_ADEQUATE_FLOOR", "E3_PLI_PARTIAL_FLOOR",
    "E3_BIAS_INCONSEQUENTIAL", "E3_BIAS_INADEQUATE",
]
