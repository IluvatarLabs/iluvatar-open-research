#!/usr/bin/env python3
"""batch_055_A — PoPS SEMANTIC-group LOFGO @ p=0.05 + finer p-grid {0.02,0.03,0.07}.

Implements experiments/batch_055_A/brief.md (v2). Two parallel sub-experiments
share the same PoPS pipeline, MAGMA-Z input, munged feature chunks, and paired-
bootstrap framework as batch_054_A:

  Reproduction gate (MUST pass first):
      - Re-run pops.py at p=0.05 with the FULL feature matrix (no LOFGO, no
        subset_features). Verify rho(PoPS, MAGMA-Z) on the shared 17,459-gene
        sample matches batch_054_A's published anchor (0.5102) to 4 decimals.
      - WHY first: the brief UNINTERPRETABLE rule says any LOFGO/finer-p-grid
        run that drifts >0.005 from anchor implies SHA256 / pipeline drift.
        Reproducing the anchor in the SAME run is the cheapest way to confirm
        the env+inputs are still equivalent to batch_054_A. If reproduction
        fails -> exit non-zero, skip downstream analysis.

  Sub-A (SEMANTIC LOFGO @ p=0.05):
      - 8 PoPS runs, each leaving out one biologically-meaningful feature
        block (defined by regex on feature names).
      - Implementation: --subset_features_path = list of features to KEEP
        (i.e. the complement of the LOFGO drop set). PoPS handles this
        natively in select_features_from_marginal_assoc_df (pops.py L346).
        WHY this over rewriting cols/mat chunks: (a) Rule 1 — re-uses an
        existing PoPS feature; (b) marginal-association p-values are per-
        feature OLS independent of which other features are present, so
        subsetting AFTER marginal-assoc is mathematically equivalent to
        munging a smaller feature matrix; (c) avoids 12 chunk-rewrites per
        LOFGO run = ~10x faster + no chunk-layout edge cases.
      - 8 groups: L1 ALL_GTEx, L2 GTEx_brain, L3 ALL_human_brain,
        L4 ALL_mouse_brain, L5 ALL_brain_total (broad regex), L6 ALL_PPI,
        L7 ALL_Pathways, L8 ImmGen (CNS-irrelevant negative control).

  Sub-B (finer p-grid):
      - 3 PoPS runs at cutoffs {0.02, 0.03, 0.07} on the FULL feature matrix.
      - Trivial extension of batch_054_A's CUTOFFS list.

  Phase 4 (analysis):
      - Reuse batch_054_A's paired-bootstrap framework. Bootstrap INDEX
        matrix (n=1000, seed=20260423) is bit-for-bit identical (same
        np.random.default_rng + same n_genes) so the bootstrap *resamples*
        match. Paired Delta-rho values are NEW: anchor is the p=0.05
        production run from THIS batch (the reproduction-gate run), not
        batch_054_A's anchor (which was p=0.001).
      - BH-FDR over the 8 LOFGO cells (Sub-A) using
        statsmodels.stats.multitest.multipletests (method='fdr_bh').
      - Per-cell secondary metrics: P@100 vs PGC3 ST12 Prioritised, SCHEMA
        median percentile (n=10), SynGO_EDT1 hand-list median percentile
        (n=14), Top-100 Jaccard vs the p=0.05 anchor.

WHY this script imports almost nothing from run_batch_054_A.py: the parent
script's imports / paths assume batch_054_A's output dir. We DUPLICATE the
small set of helpers (sha256, load_preds, load_magma_z, paired_bootstrap)
rather than couple the two scripts. Each batch directory remains a self-
contained reproducible artifact (Cardinal Rule 0: no hidden cross-batch
runtime coupling).

Resume / safety:
  - Per-run cache: if a sub-A LOFGO or sub-B cutoff already produced a
    .preds with >= 15,000 non-NaN rows, skip the PoPS subprocess.
  - Atomic writes for results.json (write to .tmp -> os.replace).
  - L8 (ImmGen) negative-control failure (|Delta-rho| >= 0.010) is RECORDED
    in results.json but does NOT exit; orchestrator decides next step.
  - Reproduction-gate failure -> non-zero exit + clear stderr message.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# -------------------- Absolute paths (agent cwd resets between calls) -------
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_055_A"
OUTPUT_DIR = BATCH_DIR / "output"
LOFGO_DIR = OUTPUT_DIR / "lofgo"
PGRID_DIR = OUTPUT_DIR / "pgrid"
ANCHOR_DIR = OUTPUT_DIR / "anchor_repro"
SUBSET_LISTS_DIR = OUTPUT_DIR / "subset_lists"  # one keep-list file per LOFGO
LOGS_DIR = BATCH_DIR / "logs"

POPS_REPO = PROJECT_ROOT / "tools" / "external" / "pops"
POPS_ENV_NAME = "pops_env"
POPS_CONDA_SH = Path("/home/yuanz/miniforge3/etc/profile.d/conda.sh")

FEATURES_ROOT = PROJECT_ROOT / "data" / "pops_features"
FEATURES_MUNGED_DIR = FEATURES_ROOT / "features_munged"
FEATURES_MUNGED_PREFIX = FEATURES_MUNGED_DIR / "pops_features"
GENE_ANNOT = FEATURES_ROOT / "gene_annot_jun10.txt"
CONTROL_FEATURES = FEATURES_ROOT / "features_jul17_control.txt"

# iter_053 canonical inputs (same as batch_054_A — provenance-locked)
ITER053_OUTPUT = PROJECT_ROOT / "experiments" / "batch_053_B" / "output"
MAGMA_REMAPPED_PREFIX = ITER053_OUTPUT / "PGC3_EUR_gene_ENSGID"
MAGMA_REMAPPED_RAW = Path(str(MAGMA_REMAPPED_PREFIX) + ".genes.raw")
MAGMA_REMAPPED_OUT = Path(str(MAGMA_REMAPPED_PREFIX) + ".genes.out")
ST12_XLSX = PROJECT_ROOT / "data" / "19426775" / "scz2022-Extended-Data-Table1.xlsx"

# batch_054_A anchor (production p=0.05 reference for the reproduction gate)
BATCH054_RESULTS_JSON = (
    PROJECT_ROOT / "experiments" / "batch_054_A" / "output" / "results.json"
)
BATCH054_P05_PREDS = (
    PROJECT_ROOT / "experiments" / "batch_054_A" / "output" / "sweep"
    / "cutoff_0.05" / "PGC3_EUR_PoPS.preds"
)
# Brief: reproduction gate must match this rho to 4 decimals.
BATCH054_P05_RHO_TARGET = 0.5102
REPRO_TOLERANCE = 1e-4  # "matches to 4 decimals"

# Brief MEASUREMENT (Sub-B): 3 finer cutoffs + p=0.05 production anchor.
PRODUCTION_CUTOFF = 0.05
FINER_CUTOFFS = [0.02, 0.03, 0.07]

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260423  # brief-specified; matches batch_054_A bit-for-bit

# Gene panels (identical to batch_054_A)
SCHEMA_GENES = ["SETD1A", "CUL1", "XPO7", "TRIO", "CACNA1G", "SP4", "GRIA3",
                "GRIN2A", "HERC1", "RB1CC1"]
SYNGO_EDT1_GENES = ['DLGAP1', 'GRIN2A', 'NRXN1', 'CNTNAP2', 'ARC', 'DLG4',
                    'NRXN2', 'NLGN1', 'NLGN2', 'SHANK1', 'SHANK3', 'HOMER1',
                    'SYN1', 'GAP43']

# -------------------- Semantic LOFGO group definitions ---------------------
# WHY regex (not literal lists): brief mandates these EXACT patterns. Each
# pattern uses Python re.search semantics (anchor with ^ where the brief
# specified anchored matching). Empirical counts on the live feature list
# (verified at implementation time, see batch_055_A scripts/README inline):
#   L1 ALL_GTEx           : 66 features  (53 GTEx.NN + 13 GTEx_brain.NN;
#                                          .control entries excluded since
#                                          brief regex anchors with [0-9]+$)
#   L2 GTEx_brain only    : 14 features  (^GTEx_brain\. catches .control too)
#   L3 ALL_human_brain    : 2834 features
#   L4 ALL_mouse_brain    : 2436 features
#   L5 ALL_brain_total    : 6062 features (broad regex below, case-insensitive)
#   L6 ALL_PPI            : 8717 features
#   L7 ALL_Pathways       : 8479 features
#   L8 ImmGen             : 293 features
# Brief stated L1=68; actual regex match is 66 (brief approximation).
# We honor the literal regex spec since "exact pattern" > "approximate count"
# and the verbatim brief regexes are documented per-group below for audit.
LOFGO_GROUPS: list[dict[str, Any]] = [
    {
        "id": "L1",
        "name": "ALL_GTEx",
        # WHY two patterns OR'd: brief specifies "drop GTEx.* + GTEx_brain.*"
        # but as two anchored regexes; we keep them separate for clarity.
        "patterns": [r"^GTEx\.[0-9]+$", r"^GTEx_brain\.[0-9]+$"],
        "case_insensitive": False,
        "expected_count": 66,
        "purpose": "Tests tissue-expression-driven hypothesis.",
    },
    {
        "id": "L2",
        "name": "GTEx_brain",
        "patterns": [r"^GTEx_brain\."],
        "case_insensitive": False,
        "expected_count": 14,
        "purpose": "Tests brain-tissue-specific (GTEx) hypothesis.",
    },
    {
        "id": "L3",
        "name": "ALL_human_brain",
        "patterns": [r"^human_brain[0-9]*"],
        "case_insensitive": False,
        "expected_count": 2834,
        "purpose": "Tests human-brain-single-cell-driven hypothesis.",
    },
    {
        "id": "L4",
        "name": "ALL_mouse_brain",
        "patterns": [r"^mouse_brain[0-9]*"],
        "case_insensitive": False,
        "expected_count": 2436,
        "purpose": "Tests mouse-brain-single-cell-driven hypothesis.",
    },
    {
        "id": "L5",
        "name": "ALL_brain_total",
        # WHY this exact regex: brief MEASUREMENT lists this as the
        # CNS / nervous-system semantic-block test (the strongest VERA A1
        # distinguishing test). It catches GTEx_brain + human_brain* +
        # mouse_brain* PLUS broader nervous-system features (cortex,
        # cerebell, hippocamp, etc.).
        "patterns": [
            r"(brain|cortex|cerebell|hippocamp|amyg|hypoth|thalam|midbrain"
            r"|nucleus_acc|pallidum|cerebr|substanti|neuron|astro|microgli"
            r"|oligo|olfactor)"
        ],
        "case_insensitive": True,
        "expected_count": 6062,
        "purpose": "Tests CNS-driven vs broad-signal hypothesis (VERA A1).",
    },
    {
        "id": "L6",
        "name": "ALL_PPI",
        "patterns": [r"^ppi\."],
        "case_insensitive": False,
        "expected_count": 8717,
        "purpose": "Tests PPI-network-driven hypothesis.",
    },
    {
        "id": "L7",
        "name": "ALL_Pathways",
        "patterns": [r"^pathways\."],
        "case_insensitive": False,
        "expected_count": 8479,
        "purpose": "Tests pathway-membership-driven hypothesis.",
    },
    {
        "id": "L8",
        "name": "ImmGen",
        "patterns": [r"^ImmGen\."],
        "case_insensitive": False,
        "expected_count": 293,
        "purpose": "CNS-irrelevant negative control; |Delta-rho| MUST be <0.010.",
    },
]


# -------------------- Logging ----------------------------------------------

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOFGO_DIR.mkdir(parents=True, exist_ok=True)
    PGRID_DIR.mkdir(parents=True, exist_ok=True)
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    SUBSET_LISTS_DIR.mkdir(parents=True, exist_ok=True)
    main_log = LOGS_DIR / "run_batch_055_A.log"
    logger = logging.getLogger("batch055A")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(main_log)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def log_step(logger: logging.Logger, step: str, phase: str) -> None:
    """Emit STEP X BEGIN/END markers per brief SAFETY requirement."""
    logger.info(f"===== STEP {step} {phase} =====")


# -------------------- Hashing + provenance ---------------------------------

def sha256_file(path: Path, block_size: int = 2 ** 20) -> str:
    """Stream-hash. WHY streaming: .npy chunks are ~700MB each; slurping all
    12 into memory would peak ~8GB."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_munged_chunks() -> tuple[list[Path], list[Path], Path]:
    """Return (mat chunks, cols chunks, rows file)."""
    chunks = sorted(FEATURES_MUNGED_DIR.glob("pops_features.mat.*.npy"))
    cols = sorted(FEATURES_MUNGED_DIR.glob("pops_features.cols.*.txt"))
    rows = FEATURES_MUNGED_DIR / "pops_features.rows.txt"
    if not chunks or not cols or not rows.exists():
        raise FileNotFoundError(
            f"Munged feature artifacts incomplete in {FEATURES_MUNGED_DIR}; "
            "STOP and re-run iter_053 munging first."
        )
    if len(chunks) != len(cols):
        raise RuntimeError(
            f"Chunk count mismatch: {len(chunks)} mat vs {len(cols)} cols."
        )
    return chunks, cols, rows


def load_all_feature_names(cols_files: list[Path]) -> list[str]:
    """Concatenate all .cols.K.txt into one ordered list (the in-matrix order
    is preserved chunk-by-chunk; PoPS hstacks chunks the same way).

    WHY ordered list (not set): determinism. The keep-lists we write to disk
    will be reproducible and diffable across runs.
    """
    feats: list[str] = []
    for cf in cols_files:
        with open(cf) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    feats.append(line)
    return feats


def write_preflight_provenance(logger: logging.Logger) -> dict:
    """Provenance: SHA256 of munged chunks + MAGMA inputs + iter_053 anchor
    preds + batch_054_A anchor preds (the reproduction-gate reference)."""
    logger.info("PREFLIGHT: computing provenance hashes")
    prov: dict = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "production_cutoff": PRODUCTION_CUTOFF,
        "finer_cutoffs": FINER_CUTOFFS,
        "lofgo_groups": [
            {k: v for k, v in g.items()} for g in LOFGO_GROUPS
        ],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "paths": {
            "features_munged_dir": str(FEATURES_MUNGED_DIR),
            "magma_remapped_raw": str(MAGMA_REMAPPED_RAW),
            "magma_remapped_out": str(MAGMA_REMAPPED_OUT),
            "gene_annot": str(GENE_ANNOT),
            "control_features": str(CONTROL_FEATURES),
            "batch054_results_json": str(BATCH054_RESULTS_JSON),
            "batch054_p05_preds": str(BATCH054_P05_PREDS),
        },
        "munged_feature_artifacts": [],
        "magma_inputs": [],
        "batch054_artifacts": [],
        "pops_env_pip_freeze": None,
        "pops_version_info": None,
    }

    chunks, cols, rows = collect_munged_chunks()
    for p in chunks + cols + [rows]:
        if not p.exists():
            raise FileNotFoundError(f"Expected munged artifact missing: {p}")
        sz = p.stat().st_size
        h = sha256_file(p)
        prov["munged_feature_artifacts"].append({
            "path": str(p), "size_bytes": sz, "sha256": h,
        })
        logger.info(f"  sha256 {h[:12]}... {p.name} ({sz} bytes)")

    for p in [MAGMA_REMAPPED_RAW, MAGMA_REMAPPED_OUT]:
        if not p.exists():
            raise FileNotFoundError(f"Expected MAGMA input missing: {p}")
        prov["magma_inputs"].append({
            "path": str(p), "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    # batch_054_A anchor artifacts: required for reproduction-gate comparison.
    # WHY hash these: if the on-disk anchor differs from what we compare
    # against, the anchor-vs-rerun delta interpretation collapses.
    for p in [BATCH054_RESULTS_JSON, BATCH054_P05_PREDS]:
        if not p.exists():
            raise FileNotFoundError(
                f"Expected batch_054_A anchor missing: {p}. Reproduction gate "
                "cannot run without the published p=0.05 anchor."
            )
        prov["batch054_artifacts"].append({
            "path": str(p), "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    try:
        r = subprocess.run(
            ["conda", "run", "-n", POPS_ENV_NAME, "pip", "freeze"],
            capture_output=True, text=True, timeout=120,
        )
        prov["pops_env_pip_freeze"] = r.stdout
        (OUTPUT_DIR / "pops_env_pip_freeze.txt").write_text(r.stdout)
        logger.info(f"  pip freeze captured ({len(r.stdout.splitlines())} pkgs)")
    except Exception as e:  # noqa: BLE001
        logger.error(f"pip freeze failed: {e}")
        prov["pops_env_pip_freeze"] = f"ERROR: {e}"

    try:
        r = subprocess.run(
            ["bash", "-lc",
             f"source {POPS_CONDA_SH} && conda activate {POPS_ENV_NAME} && "
             f"cd {POPS_REPO} && python -c 'import sys; "
             f"import pops; print(sys.version); print(pops.__file__)'"],
            capture_output=True, text=True, timeout=60,
        )
        prov["pops_version_info"] = {"stdout": r.stdout, "stderr": r.stderr,
                                     "rc": r.returncode}
    except Exception as e:  # noqa: BLE001
        prov["pops_version_info"] = {"error": str(e)}

    prov_path = OUTPUT_DIR / "provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2, default=str))
    logger.info(f"Wrote {prov_path}")
    return prov


# -------------------- LOFGO subset-list generation -------------------------

def _compile_patterns(group: dict) -> list[re.Pattern]:
    flags = re.IGNORECASE if group.get("case_insensitive") else 0
    return [re.compile(p, flags) for p in group["patterns"]]


def _matches_any(name: str, compiled: list[re.Pattern]) -> bool:
    return any(p.search(name) for p in compiled)


def build_lofgo_subset_list(group: dict, all_features: list[str],
                            logger: logging.Logger) -> tuple[Path, dict]:
    """For LOFGO group K, write a one-feature-per-line file containing the
    features to KEEP (i.e. NOT matching the drop regex).

    Returns (path_to_keep_list, stats_dict).

    WHY a keep-list (not drop-list): pops.py --subset_features_path expects
    features to RETAIN. We invert the regex match here.

    WHY we record the actual drop count: brief gives expected counts; if the
    live regex match disagrees by >1% from the brief estimate, log a warning
    so the operator catches a potential data drift before interpretation.
    Match within 1% is treated as agreement (brief counts were from Phase B
    inspection which used the same data snapshot).
    """
    compiled = _compile_patterns(group)
    drop_set = set()
    keep: list[str] = []
    for f in all_features:
        if _matches_any(f, compiled):
            drop_set.add(f)
        else:
            keep.append(f)
    n_drop = len(drop_set)
    n_keep = len(keep)
    expected = group["expected_count"]
    rel_diff = abs(n_drop - expected) / max(expected, 1)
    if rel_diff > 0.01:
        logger.warning(
            f"  LOFGO {group['id']} ({group['name']}): drop count {n_drop} "
            f"differs from brief expected {expected} by {rel_diff:.1%}. "
            "Possible data drift — investigate before interpreting Sub-A."
        )

    out_path = SUBSET_LISTS_DIR / f"keep_{group['id']}_{group['name']}.txt"
    # Atomic write
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        for f in keep:
            fh.write(f + "\n")
    os.replace(tmp, out_path)
    logger.info(
        f"  LOFGO {group['id']} ({group['name']}): "
        f"drop={n_drop} keep={n_keep} -> {out_path.name}"
    )
    return out_path, {
        "id": group["id"],
        "name": group["name"],
        "drop_count": n_drop,
        "keep_count": n_keep,
        "expected_drop_count": expected,
        "drop_count_relative_diff": rel_diff,
        "subset_path": str(out_path),
    }


# -------------------- PoPS subprocess helper -------------------------------

def run_pops(out_prefix: Path, num_chunks: int, cutoff: float,
             logger: logging.Logger,
             subset_features_path: Path | None = None,
             tag: str = "run", timeout_s: int = 3600) -> dict:
    """Invoke pops.py. Mirrors batch_054_A's run_pops_for_cutoff with one
    addition: optional --subset_features_path for LOFGO.

    WHY identical flag set otherwise: brief reproduction-gate requires the
    p=0.05 anchor to bit-for-bit reproduce batch_054_A's anchor. Any flag
    drift (e.g. HLA handling) would invalidate the gate.
    """
    log_path = LOGS_DIR / f"pops_{tag}.log"

    args = [
        "python", str(POPS_REPO / "pops.py"),
        "--gene_annot_path", str(GENE_ANNOT),
        "--feature_mat_prefix", str(FEATURES_MUNGED_PREFIX),
        "--num_feature_chunks", str(num_chunks),
        "--magma_prefix", str(MAGMA_REMAPPED_PREFIX),
        "--control_features_path", str(CONTROL_FEATURES),
        "--method", "ridge",
        "--feature_selection_p_cutoff", str(cutoff),
        "--out_prefix", str(out_prefix),
        "--verbose",
    ]
    if subset_features_path is not None:
        args.extend(["--subset_features_path", str(subset_features_path)])

    cmd = ["bash", "-lc",
           f"source {POPS_CONDA_SH} && conda activate {POPS_ENV_NAME} && "
           + " ".join([f"'{a}'" for a in args])]
    logger.info(f"RUN {tag}: {' '.join(args)}")
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"# batch_055_A {tag}\n# cmd: {' '.join(args)}\n\n")
        r = subprocess.run(cmd, cwd=str(POPS_REPO), stdout=lf,
                           stderr=subprocess.STDOUT, timeout=timeout_s)
    dur = time.time() - t0
    logger.info(f"  {tag} rc={r.returncode} wall={dur:.1f}s log={log_path}")
    preds_path = Path(str(out_prefix) + ".preds")
    ok = r.returncode == 0 and preds_path.exists()
    return {"tag": tag, "cutoff": cutoff,
            "subset_features_path": (str(subset_features_path)
                                     if subset_features_path else None),
            "rc": r.returncode, "wall_s": dur,
            "log_path": str(log_path),
            "preds_path": str(preds_path), "ok": ok}


def maybe_cached_or_run(out_prefix: Path, num_chunks: int, cutoff: float,
                        logger: logging.Logger, tag: str,
                        subset_features_path: Path | None,
                        force: bool, timeout_s: int = 3600) -> dict:
    """Resume logic: if .preds exists with >=15,000 non-NaN rows, skip the
    PoPS run (brief SAFETY requirement). --force overrides.

    WHY 15,000: matches batch_054_A's UNINTERPRETABLE threshold; a preds
    with fewer non-NaN rows is suspect (brief MEASUREMENT lower bound on
    common-bg sample is 15k).
    """
    preds_path = Path(str(out_prefix) + ".preds")
    if preds_path.exists() and not force:
        try:
            n_rows = len(load_preds(preds_path))
            if n_rows >= 15000:
                logger.info(
                    f"  {tag}: cached preds reused ({n_rows} non-NaN rows)"
                )
                return {"tag": tag, "cutoff": cutoff,
                        "subset_features_path": (str(subset_features_path)
                                                 if subset_features_path
                                                 else None),
                        "rc": 0, "wall_s": 0.0,
                        "log_path": str(LOGS_DIR / f"pops_{tag}.log"),
                        "preds_path": str(preds_path),
                        "ok": True, "cached": True}
            logger.info(f"  {tag}: cached preds has only {n_rows} rows; rerunning")
        except Exception as e:  # noqa: BLE001
            logger.info(f"  {tag}: cached preds unreadable ({e}); rerunning")
    meta = run_pops(out_prefix=out_prefix, num_chunks=num_chunks,
                    cutoff=cutoff, logger=logger,
                    subset_features_path=subset_features_path,
                    tag=tag, timeout_s=timeout_s)
    meta["cached"] = False
    return meta


# -------------------- Data loaders (duplicated from batch_054_A) -----------

def load_magma_z(logger: logging.Logger):
    """Load MAGMA_Z keyed by ENSGID. See batch_054_A run_batch_054_A.py for
    the schema-drift defensive checks rationale."""
    import pandas as pd
    magma = pd.read_csv(MAGMA_REMAPPED_OUT, sep=r"\s+")
    assert {"GENE", "ZSTAT", "P"} <= set(magma.columns), (
        f"MAGMA schema drift: columns={list(magma.columns)}"
    )
    magma = magma.rename(columns={"GENE": "ENSGID", "ZSTAT": "MAGMA_Z",
                                  "P": "MAGMA_P"})
    logger.info(f"Loaded MAGMA: {len(magma)} rows")
    return magma[["ENSGID", "MAGMA_Z", "MAGMA_P"]]


def load_preds(preds_path: Path):
    import pandas as pd
    df = pd.read_csv(preds_path, sep="\t")
    return df[["ENSGID", "PoPS_Score"]].dropna(subset=["PoPS_Score"]).copy()


# -------------------- Aligned matrix builder + bootstrap -------------------

def build_intersected_matrix(preds_by_label: dict, magma_df,
                             logger: logging.Logger):
    """Common-ENSGID intersection across all preds + MAGMA.

    Returns (label_order, common_ensgids, pops_mat[L, n], magma_z[n]).
    """
    import numpy as np
    sets = [set(df["ENSGID"]) for df in preds_by_label.values()]
    sets.append(set(magma_df["ENSGID"]))
    common = set.intersection(*sets)
    common_list = sorted(common)
    n = len(common_list)
    logger.info(f"Common ENSGID rows across {len(preds_by_label)} preds + MAGMA: {n}")
    if n < 15000:
        raise RuntimeError(
            f"Common ENSGID set has only {n} rows (UNINTERPRETABLE). "
            "One or more configs dropped a large number of genes."
        )
    idx_map = {g: i for i, g in enumerate(common_list)}
    mag_z = np.full(n, np.nan, dtype=float)
    for ensg, z in zip(magma_df["ENSGID"].values, magma_df["MAGMA_Z"].values):
        i = idx_map.get(ensg)
        if i is not None:
            mag_z[i] = float(z)
    pops_mat = np.full((len(preds_by_label), n), np.nan, dtype=float)
    label_order = list(preds_by_label.keys())
    for k, (lab, df) in enumerate(preds_by_label.items()):
        for ensg, sc in zip(df["ENSGID"].values, df["PoPS_Score"].values):
            i = idx_map.get(ensg)
            if i is not None:
                pops_mat[k, i] = float(sc)
    if np.isnan(mag_z).any():
        bad = int(np.isnan(mag_z).sum())
        raise RuntimeError(f"MAGMA_Z has {bad} NaNs after alignment.")
    if np.isnan(pops_mat).any():
        bad = int(np.isnan(pops_mat).sum())
        raise RuntimeError(f"PoPS matrix has {bad} NaNs after alignment.")
    return label_order, common_list, pops_mat, mag_z


def paired_bootstrap(pops_mat, mag_z, n_boot: int, seed: int,
                     logger: logging.Logger):
    """Paired bootstrap. Returns (point_rhos[L], rho_boot[L, n_boot]).

    WHY same RNG init as batch_054_A: brief mandates bootstrap INDICES are
    bit-for-bit reusable. np.random.default_rng(20260423).integers(0, n, ...)
    on the SAME n_genes will produce the same matrix. n_genes is determined
    by the common-bg intersection -> as long as the shared sample is the
    same 17,459 ENSGIDs, the indices match batch_054_A exactly.
    """
    import numpy as np
    from scipy.stats import spearmanr

    n_cut, n_gene = pops_mat.shape
    assert mag_z.shape[0] == n_gene
    rng = np.random.default_rng(seed)
    logger.info(f"Generating paired bootstrap idx matrix: {n_boot} x {n_gene} "
                f"(seed={seed})")
    idx_mat = rng.integers(0, n_gene, size=(n_boot, n_gene))
    point_rhos = np.full(n_cut, np.nan, dtype=float)
    for k in range(n_cut):
        r, _ = spearmanr(pops_mat[k], mag_z)
        point_rhos[k] = float(r)
    logger.info(f"Point-estimate rhos: {point_rhos.tolist()}")
    rho_boot = np.full((n_cut, n_boot), np.nan, dtype=float)
    for i in range(n_boot):
        idx = idx_mat[i]
        mag_sample = mag_z[idx]
        for k in range(n_cut):
            r, _ = spearmanr(pops_mat[k][idx], mag_sample)
            rho_boot[k, i] = float(r)
        if (i + 1) % 100 == 0:
            logger.info(f"  bootstrap progress: {i+1}/{n_boot}")
    return point_rhos, rho_boot


def percentile_ci(v, lo=2.5, hi=97.5):
    import numpy as np
    arr = np.asarray(v, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (None, None, None)
    return (float(np.percentile(arr, lo)),
            float(np.percentile(arr, hi)),
            float(np.median(arr)))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR. WHY statsmodels: brief mandates BH-FDR over
    the 8 LOFGO cells (Sub-A); statsmodels.stats.multitest is the standard
    well-tested implementation. Returns q-values in the same order."""
    from statsmodels.stats.multitest import multipletests
    _, q, _, _ = multipletests(pvals, method="fdr_bh")
    return [float(x) for x in q]


# -------------------- Per-cell secondary metrics ---------------------------

def compute_secondary_metrics(preds_df, annot_df, prior_ensgids: set,
                              prior_symbols: set, anchor_top100: set,
                              logger: logging.Logger) -> dict:
    """P@100, SCHEMA pctile, SynGO pctile, top-100 Jaccard vs anchor.

    Replicates batch_054_A.phase4_analysis per_cutoff_metrics block.
    """
    import numpy as np
    df = preds_df.merge(annot_df, on="ENSGID", how="left")
    df_sorted = df.sort_values("PoPS_Score", ascending=False).reset_index(drop=True)
    df_sorted = df_sorted.drop_duplicates("ENSGID", keep="first").reset_index(drop=True)
    N = len(df_sorted)
    is_pos = (df_sorted["ENSGID"].isin(prior_ensgids)
              | df_sorted["NAME"].isin(prior_symbols))
    n_pos = int(is_pos.sum())
    baseline = n_pos / N if N else None

    precs = {}
    for K in [50, 100, 200]:
        hits = int(is_pos.iloc[:K].sum())
        precs[f"P@{K}"] = {
            "hits": hits, "K": K,
            "precision": (hits / K) if K else None,
            "baseline": baseline,
            "lift": ((hits / K) / baseline) if baseline else None,
        }
    df_sorted["rank"] = np.arange(1, N + 1)
    df_sorted["pctile"] = 1.0 - (df_sorted["rank"] - 1) / N
    schema_pct = df_sorted[df_sorted["NAME"].isin(SCHEMA_GENES)]["pctile"].tolist()
    syngo_pct = df_sorted[df_sorted["NAME"].isin(SYNGO_EDT1_GENES)]["pctile"].tolist()
    top100_ensg = set(df_sorted.head(100)["ENSGID"].tolist())
    overlap = len(top100_ensg & anchor_top100)
    jac = jaccard(top100_ensg, anchor_top100)
    return {
        "n_total_rows": N,
        "n_prioritised_in_rows": n_pos,
        "precision_at_K": precs,
        "schema_n": len(schema_pct),
        "schema_median_pctile": float(np.median(schema_pct)) if schema_pct else None,
        "schema_pctiles": schema_pct,
        "syngo_edt1_n": len(syngo_pct),
        "syngo_edt1_median_pctile": float(np.median(syngo_pct)) if syngo_pct else None,
        "syngo_edt1_pctiles": syngo_pct,
        "top100_jaccard_vs_anchor": jac,
        "top100_overlap_count_vs_anchor": overlap,
        "top100_ensgids": sorted(top100_ensg),
    }


# -------------------- Orchestrator phases ----------------------------------

def phase_anchor_repro(num_chunks: int, force: bool, skip: bool,
                       logger: logging.Logger) -> dict:
    """Reproduction gate: rerun p=0.05 with FULL feature matrix; verify
    rho matches batch_054_A's published anchor (0.5102) to 4 decimals.

    WHY before any LOFGO/finer-p-grid run: brief explicit mandate. The
    reproduction-gate's purpose is to catch pipeline drift (env, features,
    MAGMA inputs) BEFORE we commit ~70 min of compute that would be
    uninterpretable in the presence of drift.
    """
    log_step(logger, "1/4", "BEGIN — anchor reproduction gate")
    out_prefix = ANCHOR_DIR / "PGC3_EUR_PoPS_p0.05_repro"
    if skip:
        logger.info("Anchor reproduction SKIPPED (--skip-anchor-repro)")
        return {
            "skipped": True, "preds_path": str(Path(str(out_prefix) + ".preds")),
        }
    meta = maybe_cached_or_run(
        out_prefix=out_prefix, num_chunks=num_chunks,
        cutoff=PRODUCTION_CUTOFF, logger=logger, tag="anchor_repro_p0.05",
        subset_features_path=None, force=force, timeout_s=3600,
    )
    if not meta["ok"]:
        raise RuntimeError(f"Anchor reproduction PoPS run failed: {meta}")

    # Compute rho on the same shared-bg gene set as batch_054_A.
    # WHY use only this preds + batch_054_A preds + MAGMA: we need the SAME
    # 17,459-gene shared bg. The cleanest definition is "intersection of
    # rerun preds + batch_054_A preds + MAGMA" — if the rerun reproduces
    # bit-for-bit, the intersection equals batch_054_A's common set.
    import pandas as pd
    repro_preds = load_preds(Path(meta["preds_path"]))
    batch054_preds = load_preds(BATCH054_P05_PREDS)
    magma = load_magma_z(logger)
    label_order, common_ensg, pops_mat, mag_z = build_intersected_matrix(
        {"repro_p0.05": repro_preds, "batch054_p0.05": batch054_preds},
        magma, logger,
    )
    from scipy.stats import spearmanr
    rho_repro, _ = spearmanr(pops_mat[label_order.index("repro_p0.05")], mag_z)
    rho_054, _ = spearmanr(pops_mat[label_order.index("batch054_p0.05")], mag_z)

    # Compare to brief target (0.5102). WHY both checks: target is the
    # published number; the side-by-side match-vs-batch054 catches any
    # subtle drift that left the published number stale.
    delta_target = abs(float(rho_repro) - BATCH054_P05_RHO_TARGET)
    delta_054 = abs(float(rho_repro) - float(rho_054))
    pass_target = delta_target <= REPRO_TOLERANCE
    pass_054 = delta_054 <= REPRO_TOLERANCE
    logger.info(f"  rho(repro) = {rho_repro:.6f}")
    logger.info(f"  rho(batch_054_A on shared bg) = {rho_054:.6f}")
    logger.info(f"  target rho = {BATCH054_P05_RHO_TARGET}")
    logger.info(f"  |Δ vs target| = {delta_target:.6f} (tol {REPRO_TOLERANCE})")
    logger.info(f"  |Δ vs batch_054_A rerun| = {delta_054:.6f}")
    repro = {
        "preds_path": meta["preds_path"],
        "rho_repro": float(rho_repro),
        "rho_batch054_recomputed": float(rho_054),
        "rho_batch054_target": BATCH054_P05_RHO_TARGET,
        "delta_vs_target": float(delta_target),
        "delta_vs_batch054_recomputed": float(delta_054),
        "tolerance": REPRO_TOLERANCE,
        "pass_target": pass_target,
        "pass_batch054_recomputed": pass_054,
        "pass": pass_target and pass_054,
        "n_genes_common": len(common_ensg),
        "pops_run_meta": meta,
    }
    if not repro["pass"]:
        log_step(logger, "1/4", "FAIL — reproduction gate failed")
        raise RuntimeError(
            f"REPRODUCTION GATE FAIL: rho_repro={rho_repro:.6f} differs from "
            f"target {BATCH054_P05_RHO_TARGET} by {delta_target:.6f} (tol "
            f"{REPRO_TOLERANCE}) and/or from re-computed batch_054_A rho "
            f"{rho_054:.6f} by {delta_054:.6f}. Likely SHA256 drift in "
            "features or MAGMA input. STOP and audit."
        )
    log_step(logger, "1/4", "END — reproduction gate PASS")
    return repro


def phase_run_pops(num_chunks: int, force: bool, skip: bool,
                   subset_lists_meta: list[dict],
                   logger: logging.Logger) -> dict:
    """Sub-A (8 LOFGO) + Sub-B (3 finer cutoffs) PoPS subprocess invocations."""
    log_step(logger, "2/4", "BEGIN — Sub-A LOFGO + Sub-B finer-pgrid PoPS runs")
    if skip:
        logger.info("PoPS runs SKIPPED (--skip-pops); reusing existing preds")
        return {"skipped": True}

    runs: dict = {"sub_a_lofgo": {}, "sub_b_finer_pgrid": {}}

    # ---- Sub-A: 8 LOFGO @ p=0.05 ----
    for group, sl_meta in zip(LOFGO_GROUPS, subset_lists_meta):
        gid = group["id"]
        out_subdir = LOFGO_DIR / f"{gid}_{group['name']}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_prefix = out_subdir / "PGC3_EUR_PoPS"
        meta = maybe_cached_or_run(
            out_prefix=out_prefix, num_chunks=num_chunks,
            cutoff=PRODUCTION_CUTOFF, logger=logger,
            tag=f"lofgo_{gid}_{group['name']}",
            subset_features_path=Path(sl_meta["subset_path"]),
            force=force, timeout_s=3600,
        )
        runs["sub_a_lofgo"][gid] = meta
        if not meta["ok"]:
            raise RuntimeError(
                f"Sub-A LOFGO {gid} ({group['name']}) PoPS failed: {meta}"
            )

    # ---- Sub-B: 3 finer cutoffs (full feature matrix) ----
    for cut in FINER_CUTOFFS:
        out_subdir = PGRID_DIR / f"cutoff_{cut}"
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_prefix = out_subdir / "PGC3_EUR_PoPS"
        meta = maybe_cached_or_run(
            out_prefix=out_prefix, num_chunks=num_chunks, cutoff=cut,
            logger=logger, tag=f"pgrid_cutoff_{cut}",
            subset_features_path=None, force=force, timeout_s=3600,
        )
        runs["sub_b_finer_pgrid"][str(cut)] = meta
        if not meta["ok"]:
            raise RuntimeError(f"Sub-B cutoff {cut} PoPS failed: {meta}")

    (OUTPUT_DIR / "pops_run_meta.json").write_text(json.dumps(runs, indent=2))
    log_step(logger, "2/4", "END — all PoPS runs complete")
    return runs


def phase_analysis(repro_meta: dict, run_meta: dict,
                   logger: logging.Logger) -> dict:
    """Phase 4: paired-bootstrap Delta-rho + secondary metrics + BH-FDR."""
    log_step(logger, "3/4", "BEGIN — Phase 4 analysis")
    import numpy as np
    import pandas as pd

    # Anchor for THIS batch is the reproduction-gate p=0.05 rerun.
    # WHY not batch_054_A's preds directly: brief mandates "anchor is p=0.05
    # production from batch_055_A's reproduction run" so paired bootstrap
    # operates on a self-contained set of preds files generated under the
    # same env/code path.
    if repro_meta.get("skipped"):
        anchor_preds_path = Path(repro_meta["preds_path"])
        if not anchor_preds_path.exists():
            raise RuntimeError(
                "--skip-anchor-repro was passed but no cached anchor preds "
                "exist; cannot continue analysis."
            )
    else:
        anchor_preds_path = Path(repro_meta["preds_path"])

    magma_df = load_magma_z(logger)
    annot = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME", "CHR", "TSS"]]
    assert annot["ENSGID"].is_unique, "ENSGID annotation has duplicates"

    # PGC3 ST12 Prioritised set for P@K
    try:
        st12 = pd.read_excel(ST12_XLSX, sheet_name="ST12 all criteria")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"ST12 read failed: {e}")
    prior_all = st12[st12["Prioritised"] == 1]
    prior_symbols = set(prior_all["Symbol.ID"].astype(str))
    prior_ensgids = set(prior_all["Ensembl.ID"].astype(str))

    # ---- Build the JOINT preds dict for the paired bootstrap ----
    # Layout: row 0 = anchor (production p=0.05); rows 1..8 = LOFGO L1..L8;
    # rows 9..11 = finer-pgrid cutoffs. ONE bootstrap idx matrix shared
    # across all rows -> Delta-rho is paired against row 0.
    preds_by_label: dict[str, Any] = {}
    preds_by_label["anchor_p0.05"] = load_preds(anchor_preds_path)
    for group in LOFGO_GROUPS:
        gid = group["id"]
        if run_meta.get("skipped"):
            preds_path = (LOFGO_DIR / f"{gid}_{group['name']}"
                          / "PGC3_EUR_PoPS.preds")
        else:
            preds_path = Path(run_meta["sub_a_lofgo"][gid]["preds_path"])
        if not preds_path.exists():
            raise RuntimeError(f"Missing LOFGO preds: {preds_path}")
        preds_by_label[f"lofgo_{gid}"] = load_preds(preds_path)
    for cut in FINER_CUTOFFS:
        if run_meta.get("skipped"):
            preds_path = (PGRID_DIR / f"cutoff_{cut}"
                          / "PGC3_EUR_PoPS.preds")
        else:
            preds_path = Path(run_meta["sub_b_finer_pgrid"][str(cut)]["preds_path"])
        if not preds_path.exists():
            raise RuntimeError(f"Missing finer-pgrid preds: {preds_path}")
        preds_by_label[f"pgrid_p{cut}"] = load_preds(preds_path)

    label_order, common_ensg, pops_mat, mag_z = build_intersected_matrix(
        preds_by_label, magma_df, logger,
    )
    n_genes = len(common_ensg)
    (OUTPUT_DIR / "common_ensgids.txt").write_text("\n".join(common_ensg) + "\n")

    # ---- Anchor top-100 for Jaccard secondary metric ----
    anchor_df = preds_by_label["anchor_p0.05"].merge(annot, on="ENSGID", how="left")
    anchor_top100 = set(
        anchor_df.sort_values("PoPS_Score", ascending=False)
        .drop_duplicates("ENSGID", keep="first").head(100)["ENSGID"].tolist()
    )

    # ---- Paired bootstrap (single shared idx matrix) ----
    point_rhos, rho_boot = paired_bootstrap(
        pops_mat, mag_z, BOOTSTRAP_N, BOOTSTRAP_SEED, logger,
    )
    anchor_idx = label_order.index("anchor_p0.05")
    anchor_boot = rho_boot[anchor_idx]
    p_floor = 1.0 / BOOTSTRAP_N

    # ---- Per-config rho point + CI ----
    per_cfg_rho: dict = {}
    for k, lab in enumerate(label_order):
        lo, hi, med = percentile_ci(rho_boot[k])
        per_cfg_rho[lab] = {
            "rho_point": float(point_rhos[k]),
            "rho_boot_median": med,
            "rho_ci95_lo": lo, "rho_ci95_hi": hi,
        }
        logger.info(f"  {lab}: rho={point_rhos[k]:.4f} CI=[{lo:.4f},{hi:.4f}]")

    # ---- Sub-A: per-LOFGO Delta-rho + paired CI + bootstrap p ----
    sub_a: dict = {}
    sub_a_pvals: list[float] = []
    sub_a_keys: list[str] = []
    for group in LOFGO_GROUPS:
        gid = group["id"]
        lab = f"lofgo_{gid}"
        k = label_order.index(lab)
        delta = rho_boot[k] - anchor_boot
        delta_point = float(point_rhos[k] - point_rhos[anchor_idx])
        lo, hi, med = percentile_ci(delta)
        # Two-sided bootstrap p (floored at 1/n_boot per batch_054_A convention)
        frac_pos = float((delta > 0).mean())
        frac_neg = float((delta < 0).mean())
        p_raw = min(2.0 * min(frac_pos, frac_neg), 1.0)
        p_two_sided = max(p_raw, p_floor)
        p_note = (f"floored to 1/n_boot={p_floor:g} (raw={p_raw:g})"
                  if p_raw < p_floor else None)
        # Per-cell secondary metrics (LOFGO preds vs anchor top-100)
        sec = compute_secondary_metrics(
            preds_by_label[lab], annot, prior_ensgids, prior_symbols,
            anchor_top100, logger,
        )
        sub_a[gid] = {
            "id": gid, "name": group["name"],
            "purpose": group["purpose"],
            "expected_drop_count": group["expected_count"],
            "rho_point": float(point_rhos[k]),
            "rho_anchor": float(point_rhos[anchor_idx]),
            "delta_rho_point": delta_point,
            "delta_rho_ci95_lo": lo, "delta_rho_ci95_hi": hi,
            "delta_rho_median": med,
            "p_two_sided_bootstrap": p_two_sided,
            "p_note": p_note,
            "secondary": sec,
        }
        sub_a_pvals.append(p_two_sided)
        sub_a_keys.append(gid)

    # ---- BH-FDR over the 8 LOFGO cells ----
    qvals = bh_fdr(sub_a_pvals)
    for gid, q in zip(sub_a_keys, qvals):
        sub_a[gid]["bh_q"] = float(q)
    logger.info(f"  BH-FDR q-values (sub-A): "
                + ", ".join([f"{k}={sub_a[k]['bh_q']:.4f}" for k in sub_a_keys]))

    # ---- L8 (ImmGen) negative-control sanity ----
    # WHY only RECORD (not exit): brief SAFETY says orchestrator decides.
    l8 = sub_a["L8"]
    l8_neg_control_pass = abs(l8["delta_rho_point"]) < 0.010
    l8["negative_control_pass"] = l8_neg_control_pass
    if not l8_neg_control_pass:
        logger.warning(
            f"L8 (ImmGen) negative-control |Delta-rho| = "
            f"{abs(l8['delta_rho_point']):.4f} >= 0.010 — recorded but "
            "orchestrator decides next step (per brief)."
        )

    # ---- Sub-B: per-cutoff Delta-rho + secondary ----
    sub_b: dict = {}
    sub_b_pvals: list[float] = []
    sub_b_keys: list[str] = []
    for cut in FINER_CUTOFFS:
        lab = f"pgrid_p{cut}"
        k = label_order.index(lab)
        delta = rho_boot[k] - anchor_boot
        delta_point = float(point_rhos[k] - point_rhos[anchor_idx])
        lo, hi, med = percentile_ci(delta)
        frac_pos = float((delta > 0).mean())
        frac_neg = float((delta < 0).mean())
        p_raw = min(2.0 * min(frac_pos, frac_neg), 1.0)
        p_two_sided = max(p_raw, p_floor)
        p_note = (f"floored to 1/n_boot={p_floor:g} (raw={p_raw:g})"
                  if p_raw < p_floor else None)
        sec = compute_secondary_metrics(
            preds_by_label[lab], annot, prior_ensgids, prior_symbols,
            anchor_top100, logger,
        )
        sub_b[str(cut)] = {
            "cutoff": cut,
            "rho_point": float(point_rhos[k]),
            "rho_anchor": float(point_rhos[anchor_idx]),
            "delta_rho_point": delta_point,
            "delta_rho_ci95_lo": lo, "delta_rho_ci95_hi": hi,
            "delta_rho_median": med,
            "p_two_sided_bootstrap": p_two_sided,
            "p_note": p_note,
            "secondary": sec,
        }
        sub_b_pvals.append(p_two_sided)
        sub_b_keys.append(str(cut))

    # WHY BH-FDR also for Sub-B: brief decision rule for Sub-B uses paired
    # CI not bootstrap-p directly; we still report a BH-q across the 3
    # cutoffs for symmetry with Sub-A and to make the results.json
    # interpretable downstream.
    qvals_b = bh_fdr(sub_b_pvals)
    for k, q in zip(sub_b_keys, qvals_b):
        sub_b[k]["bh_q"] = float(q)

    log_step(logger, "3/4", "END — Phase 4 analysis complete")
    return {
        "n_genes_common": n_genes,
        "anchor_label": "anchor_p0.05",
        "anchor_preds_path": str(anchor_preds_path),
        "label_order": label_order,
        "bootstrap": {"n_boot": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
        "per_config_rho": per_cfg_rho,
        "sub_a_lofgo": sub_a,
        "sub_b_finer_pgrid": sub_b,
    }


# -------------------- Driver -----------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="batch_055_A LOFGO + finer-p-grid")
    parser.add_argument("--skip-anchor-repro", action="store_true",
                        help="skip the anchor reproduction gate (DANGEROUS — "
                             "only if anchor preds already validated)")
    parser.add_argument("--skip-pops", action="store_true",
                        help="skip all PoPS runs; assumes preds already exist")
    parser.add_argument("--only-analysis", action="store_true",
                        help="only Phase 4; preds + anchor must already exist")
    parser.add_argument("--force", action="store_true",
                        help="re-run pops.py even if cached valid preds exist")
    args = parser.parse_args()

    t_start = time.time()
    logger = setup_logging()
    logger.info(f"batch_055_A runner start at {time.ctime()}")

    # -- Preflight provenance --
    try:
        prov = write_preflight_provenance(logger)
    except Exception as e:  # noqa: BLE001
        logger.exception("Preflight provenance FAILED")
        (OUTPUT_DIR / "FAILED_preflight.txt").write_text(str(e))
        return 10

    num_chunks = sum(
        1 for a in prov["munged_feature_artifacts"] if a["path"].endswith(".npy")
    )
    logger.info(f"Detected {num_chunks} munged feature chunks")

    # -- Generate the 8 LOFGO keep-lists (cheap, deterministic) --
    log_step(logger, "0/4", "BEGIN — LOFGO subset-list generation")
    _, cols_files, _ = collect_munged_chunks()
    all_features = load_all_feature_names(cols_files)
    logger.info(f"Loaded {len(all_features)} feature names from {len(cols_files)} cols files")
    if len(all_features) != 57742:
        # WHY warn (not error): brief expects 57,742 features. A different
        # count signals data drift; flag it but let downstream LOFGO-count
        # checks decide whether to abort.
        logger.warning(
            f"Total features ({len(all_features)}) differs from brief's 57,742; "
            "data drift suspected."
        )
    subset_lists_meta = []
    for group in LOFGO_GROUPS:
        _, sl_meta = build_lofgo_subset_list(group, all_features, logger)
        subset_lists_meta.append(sl_meta)
    (OUTPUT_DIR / "subset_lists_meta.json").write_text(
        json.dumps(subset_lists_meta, indent=2))
    log_step(logger, "0/4", "END — subset lists written")

    repro: dict = {}
    runs: dict = {}

    if args.only_analysis:
        # Reconstruct repro metadata from cached anchor preds
        anchor_preds_path = ANCHOR_DIR / "PGC3_EUR_PoPS_p0.05_repro.preds"
        if not anchor_preds_path.exists():
            logger.error(
                "--only-analysis requires cached anchor preds at "
                f"{anchor_preds_path}; halting."
            )
            return 41
        repro = {"preds_path": str(anchor_preds_path), "skipped": True,
                 "pass": True, "note": "only-analysis mode; gate not enforced"}
        runs = {"skipped": True}
    else:
        # -- Phase 1: Anchor reproduction gate --
        try:
            repro = phase_anchor_repro(num_chunks, args.force,
                                       args.skip_anchor_repro, logger)
        except Exception as e:  # noqa: BLE001
            logger.exception("Anchor reproduction gate FAILED")
            (OUTPUT_DIR / "FAILED_anchor_repro.txt").write_text(str(e))
            return 20

        # -- Phase 2: PoPS runs (Sub-A LOFGO + Sub-B finer pgrid) --
        try:
            runs = phase_run_pops(num_chunks, args.force, args.skip_pops,
                                  subset_lists_meta, logger)
        except Exception as e:  # noqa: BLE001
            logger.exception("PoPS runs FAILED")
            (OUTPUT_DIR / "FAILED_pops_runs.txt").write_text(str(e))
            return 30

    # -- Phase 3: Analysis --
    try:
        analysis = phase_analysis(repro, runs, logger)
    except Exception as e:  # noqa: BLE001
        logger.exception("Phase 4 analysis FAILED")
        (OUTPUT_DIR / "FAILED_analysis.txt").write_text(str(e))
        return 50

    # -- Phase 4: Atomic write of results.json --
    log_step(logger, "4/4", "BEGIN — write results.json")
    wall = time.time() - t_start
    results = {
        "batch": "055_A",
        "version": "v2",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_min": wall / 60.0,
        "provenance": prov,
        "subset_lists_meta": subset_lists_meta,
        "anchor_reproduction": repro,
        "pops_run_meta": runs,
        "analysis": analysis,
    }
    out_path = OUTPUT_DIR / "results.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(results, indent=2, default=str))
    os.replace(tmp_path, out_path)
    logger.info(f"Wrote {out_path} (wall={wall/60:.1f} min)")
    log_step(logger, "4/4", "END — results.json written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
