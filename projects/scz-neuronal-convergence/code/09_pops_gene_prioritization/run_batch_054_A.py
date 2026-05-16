#!/usr/bin/env python3
"""batch_054_A - PoPS feature-selection cutoff sweep with paired-bootstrap Delta-rho.

Executes the plan in experiments/batch_054_A/brief.md:

  Pre-flight: Provenance (SHA256 of munged feature chunks + MAGMA input,
              plus `pip freeze` of pops_env) -> output/provenance.json.
  Smoke test: ONE pops.py run at p=0.001 with --num_feature_chunks 2 to
              verify environment health (NOT used for analysis; the
              iter_053 preds file is the anchor).
  Phase 3:    5 pops.py runs at cutoffs {0.001, 0.01, 0.05, 0.10, 0.20}
              using the SAME munged feature chunks + remapped MAGMA input
              as iter_053. Outputs go to output/sweep/cutoff_<X>/.
  Phase 4:    Paired-bootstrap rho(PoPS, MAGMA_Z) on the common 17,459
              ENSGID set. ONE (1000 x 17,459) index matrix seeded with
              20260423 is reused across all 5 cutoffs so Delta-rho =
              rho(c) - rho(0.001) is a PAIRED statistic.
              Per-cutoff: precision@{50,100,200}, SCHEMA median pctile
              (n=9), SynGO_EDT1 hand-list median pctile (n=14), top-100
              Jaccard vs iter_053 anchor.

Design / WHY:

  - WHY reuse iter_053 munged chunks unchanged: Rule 1 (don't reinvent).
    Munging is deterministic given the raw features + gene_annot; redoing
    it would only waste 30+ minutes AND would break the SHA256 provenance
    link to iter_053. A pre-flight hash compares what's on disk NOW vs.
    what iter_053 used; if they differ, STOP with a clear log message.
  - WHY a paired bootstrap index matrix: Brief MEASUREMENT section. Same
    gene sample across cutoffs lets us directly estimate Var(Delta-rho)
    instead of Var(rho_1) + Var(rho_2). The nested-feature-set structure
    (p=0.001 subset of p=0.01 subset of ...) makes per-cutoff rho's highly
    correlated (~0.8), so paired Delta-rho CI is ~10x tighter than
    unpaired difference of independent CIs (Efron & Tibshirani 1993).
    Shared seed=20260423 is the brief-specified anchor.
  - WHY subprocess invocation of pops.py: iter_053 established that
    pops_env pins numpy==1.19.5 / pandas==1.0.5 / scipy==1.5.2; our
    orchestrator (this file) uses the base Marvin env. Direct import
    would fail.
  - WHY NOT re-run MAGMA or re-download features: brief IMPORTANT
    CONSTRAINTS forbid it. Pre-flight SHA256 is the check-don't-trust
    discipline for that assumption.
  - WHY we do NOT pass --project_out_covariates_keep_hla / --training_keep_hla
    flags: iter_053's real Phase 3 run (run_batch_053_B.py line 622-633)
    did not pass them either; defaults (remove_hla=True) apply. We match
    that exactly to isolate the cutoff effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# -------------------- Absolute paths (agent cwd resets between calls) -------
PROJECT_ROOT = Path("/home/yuanz/Documents/GitHub/biomarvin_schizophrenia")
BATCH_DIR = PROJECT_ROOT / "experiments" / "batch_054_A"
OUTPUT_DIR = BATCH_DIR / "output"
SWEEP_DIR = OUTPUT_DIR / "sweep"
LOGS_DIR = BATCH_DIR / "logs"

POPS_REPO = PROJECT_ROOT / "tools" / "external" / "pops"
POPS_ENV_NAME = "pops_env"
POPS_CONDA_SH = Path("/home/yuanz/miniforge3/etc/profile.d/conda.sh")
POPS_ENV_PY = Path("/home/yuanz/miniforge3/envs/pops_env/bin/python")

FEATURES_ROOT = PROJECT_ROOT / "data" / "pops_features"
FEATURES_MUNGED_DIR = FEATURES_ROOT / "features_munged"
FEATURES_MUNGED_PREFIX = FEATURES_MUNGED_DIR / "pops_features"
GENE_ANNOT = FEATURES_ROOT / "gene_annot_jun10.txt"
CONTROL_FEATURES = FEATURES_ROOT / "features_jul17_control.txt"

# iter_053 canonical anchor inputs
ITER053_OUTPUT = PROJECT_ROOT / "experiments" / "batch_053_B" / "output"
MAGMA_REMAPPED_PREFIX = ITER053_OUTPUT / "PGC3_EUR_gene_ENSGID"
MAGMA_REMAPPED_RAW = Path(str(MAGMA_REMAPPED_PREFIX) + ".genes.raw")
MAGMA_REMAPPED_OUT = Path(str(MAGMA_REMAPPED_PREFIX) + ".genes.out")
ITER053_TOP100_JSON = ITER053_OUTPUT / "top100_comparison.json"
ITER053_PREDS = ITER053_OUTPUT / "PGC3_EUR_PoPS.preds"  # anchor p=0.001 reference
ST12_XLSX = PROJECT_ROOT / "data" / "19426775" / "scz2022-Extended-Data-Table1.xlsx"

# Brief MEASUREMENT: 5 cutoffs; 0.001 is the iter_053 anchor.
CUTOFFS = [0.001, 0.01, 0.05, 0.10, 0.20]
ANCHOR_CUTOFF = 0.001
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260423  # brief-specified

# Gene-set panels (identical to iter_053 Phase 4)
SCHEMA_GENES = ["SETD1A", "CUL1", "XPO7", "TRIO", "CACNA1G", "SP4", "GRIA3",
                "GRIN2A", "HERC1", "RB1CC1"]
SYNGO_EDT1_GENES = ['DLGAP1', 'GRIN2A', 'NRXN1', 'CNTNAP2', 'ARC', 'DLG4',
                    'NRXN2', 'NLGN1', 'NLGN2', 'SHANK1', 'SHANK3', 'HOMER1',
                    'SYN1', 'GAP43']


# -------------------- Logging ----------------------------------------------

def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    main_log = LOGS_DIR / "run_batch_054_A.log"
    logger = logging.getLogger("batch054A")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(main_log)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# -------------------- Hashing + provenance ---------------------------------

def sha256_file(path: Path, block_size: int = 2 ** 20) -> str:
    """Stream-hash a file. WHY streaming: .npy chunks are ~700 MB each;
    slurping into memory would peak at ~8 GB for 12 chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_munged_chunks() -> list[Path]:
    """Return sorted list of munged feature-matrix chunks.

    Iter_053 produced 12 chunks: pops_features.mat.{0..11}.npy plus
    matching .cols.{k}.txt and a single .rows.txt. We verify all three
    groups are present."""
    chunks = sorted(FEATURES_MUNGED_DIR.glob("pops_features.mat.*.npy"))
    cols = sorted(FEATURES_MUNGED_DIR.glob("pops_features.cols.*.txt"))
    rows = FEATURES_MUNGED_DIR / "pops_features.rows.txt"
    missing = []
    if not chunks:
        missing.append("mat.*.npy")
    if not cols:
        missing.append("cols.*.txt")
    if not rows.exists():
        missing.append("rows.txt")
    if missing:
        raise FileNotFoundError(
            f"Munged feature artifacts missing from {FEATURES_MUNGED_DIR}: "
            f"{missing}. Brief assumes iter_053 munging is complete; if not, "
            f"STOP and run phase2/3 munging first. See iter_053 runner "
            f"phase3_execute() for the munging command."
        )
    if len(chunks) != len(cols):
        raise RuntimeError(
            f"Chunk count mismatch: {len(chunks)} mat.*.npy vs {len(cols)} "
            f"cols.*.txt. Munging likely incomplete."
        )
    return chunks + cols + [rows]


def write_preflight_provenance(logger: logging.Logger) -> dict:
    """Write output/provenance.json with file SHA256s + pops_env pip freeze.

    WHY this MUST run before any PoPS invocation: the brief's UNINTERPRETABLE
    condition explicitly names "SHA256 mismatch on feature chunks vs iter_053"
    as a halt trigger. We cannot compare to iter_053 after-the-fact; we
    must record the hashes NOW and flag drift at run-time."""
    logger.info("PREFLIGHT: computing provenance hashes")
    prov: dict = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cutoffs": CUTOFFS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "paths": {
            "features_munged_dir": str(FEATURES_MUNGED_DIR),
            "magma_remapped_raw": str(MAGMA_REMAPPED_RAW),
            "magma_remapped_out": str(MAGMA_REMAPPED_OUT),
            "gene_annot": str(GENE_ANNOT),
            "control_features": str(CONTROL_FEATURES),
            "iter053_preds_anchor": str(ITER053_PREDS),
        },
        "munged_feature_artifacts": [],
        "magma_inputs": [],
        "pops_env_pip_freeze": None,
        "pops_version_info": None,
    }

    # Munged feature chunks
    artifacts = collect_munged_chunks()
    for p in artifacts:
        if not p.exists():
            raise FileNotFoundError(f"Expected munged artifact missing: {p}")
        sz = p.stat().st_size
        h = sha256_file(p)
        prov["munged_feature_artifacts"].append({
            "path": str(p), "size_bytes": sz, "sha256": h,
        })
        logger.info(f"  sha256 {h[:12]}... {p.name} ({sz} bytes)")

    # MAGMA inputs (we rely on both .genes.raw for pops.py and .genes.out
    # for MAGMA_Z values downstream).
    for p in [MAGMA_REMAPPED_RAW, MAGMA_REMAPPED_OUT]:
        if not p.exists():
            raise FileNotFoundError(f"Expected MAGMA input missing: {p}")
        prov["magma_inputs"].append({
            "path": str(p), "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    # pip freeze + pops version. WHY both: brief W2 addressing requires
    # env pinning; pops.__file__ confirms we're loading the repo copy
    # (not a stray pip-installed fork).
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
    prov_path.write_text(json.dumps(prov, indent=2))
    logger.info(f"Wrote {prov_path}")
    return prov


# -------------------- PoPS subprocess helper -------------------------------

def run_pops_for_cutoff(cutoff: float, out_prefix: Path, num_chunks: int,
                        logger: logging.Logger, smoke: bool = False,
                        timeout_s: int = 3600) -> dict:
    """Invoke pops.py --method ridge --feature_selection_p_cutoff <cutoff>.

    WHY same flags as iter_053: brief hard constraint. We pass HLA flags
    in the DEFAULT state (remove_hla=True); iter_053's real Phase 3 run
    did the same. See run_batch_053_B.py phase3_execute() invocation at
    lines 623-635.

    WHY --num_feature_chunks 2 for smoke: brief SMOKE TEST requirement.
    A 2-chunk run completes in ~1-2 min and exercises the full pipeline
    (munge load -> feature selection -> Ridge fit -> preds write).
    """
    tag = f"smoke_{cutoff}" if smoke else f"cutoff_{cutoff}"
    log_path = LOGS_DIR / f"pops_{tag}.log"

    # Quote each arg to survive bash -lc; we pipe stdout+stderr into a
    # per-run log. The returncode is captured; the orchestrator halts on
    # nonzero exit so we never interpret an incomplete preds file.
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
    cmd = ["bash", "-lc",
           f"source {POPS_CONDA_SH} && conda activate {POPS_ENV_NAME} && "
           + " ".join([f"'{a}'" for a in args])]
    logger.info(f"RUN {tag}: {' '.join(args)}")
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"# batch_054_A {tag}\n# cmd: {' '.join(args)}\n\n")
        r = subprocess.run(cmd, cwd=str(POPS_REPO), stdout=lf,
                           stderr=subprocess.STDOUT, timeout=timeout_s)
    dur = time.time() - t0
    logger.info(f"  {tag} rc={r.returncode} wall={dur:.1f}s log={log_path}")
    preds_path = Path(str(out_prefix) + ".preds")
    ok = r.returncode == 0 and preds_path.exists()
    return {"cutoff": cutoff, "tag": tag, "rc": r.returncode,
            "wall_s": dur, "log_path": str(log_path),
            "preds_path": str(preds_path), "ok": ok}


# -------------------- Smoke test -------------------------------------------

def run_smoke_test(logger: logging.Logger, num_chunks_total: int) -> dict:
    """p=0.001 with --num_feature_chunks 2 before the real sweep.

    WHY: brief IMPORTANT CONSTRAINTS explicitly requires a smoke test.
    It does NOT aim to reproduce the iter_053 ρ within 0.01 literally --
    using only 2 of 12 chunks means the feature-selection pool is 1/6
    the size, so the resulting ρ will differ. What we actually verify:
    (a) pops.py exits 0, (b) preds file is written, (c) preds has
    >= 15,000 ENSGID rows (brief UNINTERPRETABLE threshold). Full-sweep
    runs then use all chunks.
    """
    smoke_dir = OUTPUT_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_prefix = smoke_dir / "smoke_p0.001"
    meta = run_pops_for_cutoff(
        cutoff=0.001, out_prefix=smoke_prefix,
        num_chunks=min(2, num_chunks_total),
        logger=logger, smoke=True, timeout_s=1200,
    )
    if not meta["ok"]:
        raise RuntimeError(
            f"Smoke test FAILED (rc={meta['rc']}). See {meta['log_path']}. "
            "Environment is not healthy; halting before full sweep."
        )
    # Check row count. WHY load_preds() rather than raw pd.read_csv:
    # load_preds() drops rows with NaN PoPS_Score. A NaN-filled preds
    # file would pass a naive len() check but be analytically worthless,
    # so we count only usable rows against the 15,000 UNINTERPRETABLE
    # threshold (Cardinal Rule 0: fake metrics must not pass silently).
    preds = load_preds(Path(meta["preds_path"]))
    n_rows = len(preds)
    meta["n_rows"] = n_rows
    if n_rows < 15000:
        raise RuntimeError(
            f"Smoke preds has only {n_rows} non-NaN rows (threshold 15,000). "
            "Feature-matrix corruption suspected per brief UNINTERPRETABLE."
        )
    logger.info(f"SMOKE PASS: non-NaN preds rows={n_rows}, rc=0")
    return meta


# -------------------- Phase 4 analysis -------------------------------------

def load_magma_z(logger: logging.Logger):
    """Load MAGMA_Z keyed by ENSGID.

    WHY sep=r'\\s+' not delim_whitespace: pandas 2.2+ deprecates the
    latter; the regex form is the forward-compatible replacement and
    matches identically on whitespace-delimited MAGMA .genes.out.

    WHY the explicit column-set assertion: if a future MAGMA version
    renames GENE/ZSTAT/P, the subsequent .rename() would silently no-op
    and the [['ENSGID','MAGMA_Z','MAGMA_P']] selection would KeyError
    without a schema-drift diagnosis. We fail fast with column names in
    the message (Cardinal Rule 0: expose, don't hide).
    """
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


def build_intersected_matrix(preds_by_cutoff: dict, magma_df, logger):
    """Return (ordered ENSGID list, pops_matrix [n_cut x n_genes],
    magma_z vector [n_genes]) on the common ENSGID set across all cutoffs
    AND MAGMA.

    WHY common intersection: brief MEASUREMENT says "the SAME 17,459
    ENSGID rows" -- this is the iter_053 anchor n. If any cutoff's
    preds subset diverges, we use the strictest common set and log
    the delta.
    """
    import numpy as np
    sets = [set(df["ENSGID"]) for df in preds_by_cutoff.values()]
    sets.append(set(magma_df["ENSGID"]))
    common = set.intersection(*sets)
    common_list = sorted(common)
    n = len(common_list)
    logger.info(f"Common ENSGID rows across 5 cutoffs + MAGMA: {n}")
    if n < 15000:
        raise RuntimeError(
            f"Common ENSGID set has only {n} rows (UNINTERPRETABLE threshold). "
            "One or more cutoffs dropped a large number of genes."
        )

    # Aligned matrices
    idx_map = {g: i for i, g in enumerate(common_list)}
    mag_z = np.full(n, np.nan, dtype=float)
    for ensg, z in zip(magma_df["ENSGID"].values, magma_df["MAGMA_Z"].values):
        i = idx_map.get(ensg)
        if i is not None:
            mag_z[i] = float(z)

    pops_mat = np.full((len(preds_by_cutoff), n), np.nan, dtype=float)
    cutoff_order = list(preds_by_cutoff.keys())
    for k, (cut, df) in enumerate(preds_by_cutoff.items()):
        for ensg, sc in zip(df["ENSGID"].values, df["PoPS_Score"].values):
            i = idx_map.get(ensg)
            if i is not None:
                pops_mat[k, i] = float(sc)

    # Sanity: no NaN in the aligned matrices. WHY include offending
    # ENSGIDs in the error: a bare "N NaNs" count forces the operator to
    # rerun interactively to locate the bad rows; first-5 samples make
    # the MAGMA-vs-PoPS intersection mismatch diagnosable in one shot.
    if np.isnan(mag_z).any():
        bad_idx = np.where(np.isnan(mag_z))[0]
        bad = int(bad_idx.size)
        bad_ensgids = [common_list[i] for i in bad_idx[:5].tolist()]
        raise RuntimeError(
            f"MAGMA_Z has {bad} NaNs after alignment; "
            f"first {min(5, bad)} ENSGIDs: {bad_ensgids}"
        )
    if np.isnan(pops_mat).any():
        bad_mask = np.isnan(pops_mat)
        bad = int(bad_mask.sum())
        # Report offenders from the first affected cutoff row
        first_bad_cut = int(np.argmax(bad_mask.any(axis=1)))
        bad_gene_idx = np.where(bad_mask[first_bad_cut])[0]
        bad_ensgids = [common_list[i] for i in bad_gene_idx[:5].tolist()]
        raise RuntimeError(
            f"PoPS matrix has {bad} NaNs after alignment "
            f"(cutoff_order[{first_bad_cut}]={cutoff_order[first_bad_cut]}); "
            f"first {min(5, len(bad_gene_idx))} offending ENSGIDs: "
            f"{bad_ensgids}"
        )

    return cutoff_order, common_list, pops_mat, mag_z


def paired_bootstrap(pops_mat, mag_z, n_boot: int, seed: int, logger):
    """Paired bootstrap: ONE index matrix of shape (n_boot, n_genes)
    sampled with replacement, reused across all cutoffs.

    Returns rho_boot dict {cutoff_index: ndarray(n_boot)} plus the point
    estimates array [n_cutoffs].

    WHY scipy.stats.spearmanr: brief MEASUREMENT explicit. For a 17k x 1k
    bootstrap this is O(n log n) per iteration per cutoff -- ~O(5 * 1000 *
    17000 log 17000) ~= tractable on CPU. We avoid per-iter nan_policy=
    ambiguity by feeding dense pre-cleaned arrays.
    """
    import numpy as np
    from scipy.stats import spearmanr

    n_cut, n_gene = pops_mat.shape
    assert mag_z.shape[0] == n_gene
    assert n_boot >= 1

    rng = np.random.default_rng(seed)
    logger.info(f"Generating paired bootstrap index matrix: "
                f"{n_boot} x {n_gene} (seed={seed})")
    idx_mat = rng.integers(0, n_gene, size=(n_boot, n_gene))

    # Point estimates (unresampled)
    point_rhos = np.full(n_cut, np.nan, dtype=float)
    for k in range(n_cut):
        r, _ = spearmanr(pops_mat[k], mag_z)
        point_rhos[k] = float(r)
    logger.info(f"Point-estimate rhos: {point_rhos.tolist()}")

    # Bootstrap
    rho_boot = np.full((n_cut, n_boot), np.nan, dtype=float)
    mag_mat_cache = None  # we re-index mag_z each iter; cheap
    for i in range(n_boot):
        idx = idx_mat[i]
        mag_sample = mag_z[idx]
        for k in range(n_cut):
            r, _ = spearmanr(pops_mat[k][idx], mag_sample)
            rho_boot[k, i] = float(r)
        if (i + 1) % 100 == 0:
            logger.info(f"  bootstrap progress: {i + 1}/{n_boot}")

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
    if not u:
        return 0.0
    return len(a & b) / len(u)


def phase4_analysis(preds_paths: dict, logger: logging.Logger) -> dict:
    """Compute Delta-rho + per-cutoff metrics.

    preds_paths: ordered dict {cutoff_float: Path(.preds)}.
    """
    import numpy as np
    import pandas as pd

    logger.info("PHASE 4: paired-bootstrap Delta-rho analysis")

    # Load MAGMA (17,460 rows)
    magma_df = load_magma_z(logger)

    # Load each cutoff's preds
    preds_by_cutoff = {}
    for cut, p in preds_paths.items():
        df = load_preds(p)
        preds_by_cutoff[cut] = df
        logger.info(f"  cutoff={cut}: {len(df)} preds rows from {p.name}")

    # Build aligned matrices
    cutoff_order, common_ensgids, pops_mat, mag_z = build_intersected_matrix(
        preds_by_cutoff, magma_df, logger,
    )
    n_genes = len(common_ensgids)
    (OUTPUT_DIR / "common_ensgids.txt").write_text(
        "\n".join(common_ensgids) + "\n",
    )
    logger.info(f"Wrote common_ensgids.txt ({n_genes} genes)")

    # Paired bootstrap
    point_rhos, rho_boot = paired_bootstrap(
        pops_mat, mag_z, BOOTSTRAP_N, BOOTSTRAP_SEED, logger,
    )

    # ---- Per-cutoff CIs ----
    per_cutoff: dict = {}
    anchor_idx = cutoff_order.index(ANCHOR_CUTOFF)
    anchor_boot = rho_boot[anchor_idx]

    for k, cut in enumerate(cutoff_order):
        lo, hi, med = percentile_ci(rho_boot[k])
        per_cutoff[str(cut)] = {
            "cutoff": cut,
            "rho_point": float(point_rhos[k]),
            "rho_boot_median": med,
            "rho_ci95_lo": lo,
            "rho_ci95_hi": hi,
        }
        logger.info(f"  cutoff={cut}: rho={point_rhos[k]:.4f} "
                    f"CI95=[{lo:.4f}, {hi:.4f}] (median {med:.4f})")

    # ---- Paired Delta-rho vs anchor ----
    delta_rho: dict = {}
    p_floor = 1.0 / BOOTSTRAP_N  # WHY: bootstrap resolution cap
    for k, cut in enumerate(cutoff_order):
        delta = rho_boot[k] - anchor_boot  # element-wise; SAME bootstrap sample
        delta_point = float(point_rhos[k] - point_rhos[anchor_idx])
        lo, hi, med = percentile_ci(delta)
        # Two-sided p: 2 * min(P(Delta>0), P(Delta<0)); reported as % of
        # bootstrap draws consistent with null (Delta=0). For anchor vs
        # itself this is meaningless -> emit 1.0.
        #
        # WHY floor p at 1/n_boot: with n_boot=1000, p=0.0 is not
        # achievable — the smallest resolvable nonzero p is 1/1000. We
        # report p = max(raw_p, 1/n_boot) and annotate via p_note so
        # downstream readers know the floor was applied (Cardinal Rule 5:
        # never present "p < floor" as "p = 0").
        p_note = None
        if cut == ANCHOR_CUTOFF:
            p_two_sided = 1.0
        else:
            frac_pos = float((delta > 0).mean())
            frac_neg = float((delta < 0).mean())
            p_raw = 2.0 * min(frac_pos, frac_neg)
            p_raw = min(p_raw, 1.0)
            p_two_sided = max(p_raw, p_floor)
            if p_raw < p_floor:
                p_note = (f"p floored to 1/n_boot = {p_floor:g} "
                          f"(raw bootstrap p = {p_raw:g})")
        delta_rho[str(cut)] = {
            "cutoff": cut,
            "vs_anchor": ANCHOR_CUTOFF,
            "delta_point": delta_point,
            "delta_median": med,
            "delta_ci95_lo": lo,
            "delta_ci95_hi": hi,
            "p_two_sided_bootstrap": p_two_sided,
            "p_note": p_note,
        }
        logger.info(f"  Delta-rho({cut} vs {ANCHOR_CUTOFF}): "
                    f"point={delta_point:+.4f} "
                    f"CI95=[{lo:+.4f}, {hi:+.4f}] p2s={p_two_sided:.4g}"
                    + (f" [{p_note}]" if p_note else ""))

    # ---- Per-cutoff Phase-4 metrics ----
    # Load gene_annot for NAME mapping + PGC3 ST12 Prioritised for P@K.
    # WHY assert ENSGID uniqueness in annot: the per-cutoff merge below
    # is `preds.merge(annot, on='ENSGID')`. If annot had duplicates, the
    # merge would row-multiply silently and the precision@K numerator
    # would be inflated / deflated arbitrarily.
    annot = pd.read_csv(GENE_ANNOT, sep="\t")[["ENSGID", "NAME", "CHR", "TSS"]]
    assert annot["ENSGID"].is_unique, "ENSGID annotation has duplicates"

    # WHY hard-fail on ST12: brief MEASUREMENT names precision@K on the
    # PGC3 Prioritised list as the primary biological-signal metric. A
    # silently-empty prior set produces 0-hit "numbers" that look real
    # but are fabricated (Cardinal Rule 0). "I don't know" by halting
    # beats populating results.json with meaningless zeros.
    try:
        st12 = pd.read_excel(ST12_XLSX, sheet_name="ST12 all criteria")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"ST12 read failed: {e}")
    prior_all = st12[st12["Prioritised"] == 1]
    prior_symbols = set(prior_all["Symbol.ID"].astype(str))
    prior_ensgids = set(prior_all["Ensembl.ID"].astype(str))
    have_st12 = True

    # Iter_053 anchor top-100 ENSGIDs for Jaccard
    iter053_preds = load_preds(ITER053_PREDS).merge(annot, on="ENSGID", how="left")
    iter053_top100 = set(
        iter053_preds.sort_values("PoPS_Score", ascending=False)
        .head(100)["ENSGID"].tolist()
    )
    logger.info(f"iter_053 anchor top-100 ENSGID set size: {len(iter053_top100)}")

    per_cutoff_metrics: dict = {}
    for cut in cutoff_order:
        df = preds_by_cutoff[cut].merge(annot, on="ENSGID", how="left")
        df_sorted = df.sort_values("PoPS_Score", ascending=False).reset_index(drop=True)
        # WHY drop_duplicates AFTER sort: we keep the highest-scoring row
        # per ENSGID. Without this, any duplicated ENSGID in preds would
        # double-count toward the top-K slice and inflate P@K hits. This
        # is defensive — iter_053 preds were unique — but cheap and
        # makes the metric robust to future PoPS output changes.
        df_sorted = df_sorted.drop_duplicates("ENSGID", keep="first").reset_index(drop=True)
        N = len(df_sorted)

        # Positive mask (by ENSGID primary + symbol fallback)
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

        # SCHEMA + SynGO_EDT1 pctiles (rank 1 = top; pctile = 1 - (rank-1)/N)
        df_sorted["rank"] = np.arange(1, N + 1)
        df_sorted["pctile"] = 1.0 - (df_sorted["rank"] - 1) / N
        schema_pct = df_sorted[df_sorted["NAME"].isin(SCHEMA_GENES)]["pctile"].tolist()
        syngo_pct = df_sorted[df_sorted["NAME"].isin(SYNGO_EDT1_GENES)]["pctile"].tolist()

        top100_ensg = set(df_sorted.head(100)["ENSGID"].tolist())
        overlap = len(top100_ensg & iter053_top100)
        jac = jaccard(top100_ensg, iter053_top100)

        per_cutoff_metrics[str(cut)] = {
            "cutoff": cut,
            "n_total_rows": N,
            "n_prioritised_in_rows": n_pos,
            "have_st12": have_st12,
            "precision_at_K": precs,
            "schema_n": len(schema_pct),
            "schema_median_pctile": float(np.median(schema_pct)) if schema_pct else None,
            "schema_pctiles": schema_pct,
            "syngo_edt1_n": len(syngo_pct),
            "syngo_edt1_median_pctile": float(np.median(syngo_pct)) if syngo_pct else None,
            "syngo_edt1_pctiles": syngo_pct,
            "top100_jaccard_vs_iter053": jac,
            "top100_overlap_count_vs_iter053": overlap,
            "top100_ensgids": sorted(top100_ensg),
        }
        logger.info(
            f"  cutoff={cut}: P@100={precs['P@100']['precision']}, "
            f"SCHEMA_med={per_cutoff_metrics[str(cut)]['schema_median_pctile']}, "
            f"SynGO_med={per_cutoff_metrics[str(cut)]['syngo_edt1_median_pctile']}, "
            f"top100_jac={jac:.3f}"
        )

    return {
        "n_genes_common": n_genes,
        "cutoff_order": [float(c) for c in cutoff_order],
        "anchor_cutoff": ANCHOR_CUTOFF,
        "bootstrap": {
            "n_boot": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
        },
        "per_cutoff_rho": per_cutoff,
        "delta_rho_vs_anchor": delta_rho,
        "per_cutoff_metrics": per_cutoff_metrics,
    }


# -------------------- Driver -----------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="batch_054_A cutoff sweep")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="skip smoke test (only if env already verified)")
    parser.add_argument("--skip-pops", action="store_true",
                        help="skip the 5 pops.py runs (use existing preds)")
    parser.add_argument("--only-analysis", action="store_true",
                        help="only Phase 4; preds must already exist")
    parser.add_argument("--force", action="store_true",
                        help="re-run pops.py even if cached valid preds exist")
    args = parser.parse_args()

    t_start = time.time()
    logger = setup_logging()
    logger.info(f"batch_054_A runner start at {time.ctime()}")

    # -- Preflight provenance (ALWAYS) --
    try:
        prov = write_preflight_provenance(logger)
    except Exception as e:  # noqa: BLE001
        logger.exception("Preflight provenance FAILED")
        (OUTPUT_DIR / "FAILED_preflight.txt").write_text(str(e))
        return 10

    num_chunks = len([a for a in prov["munged_feature_artifacts"]
                      if a["path"].endswith(".npy")])
    logger.info(f"Detected {num_chunks} munged feature chunks")

    preds_paths: dict = {}

    if not args.only_analysis:
        # -- Smoke test --
        if not args.skip_smoke:
            try:
                run_smoke_test(logger, num_chunks)
            except Exception as e:  # noqa: BLE001
                logger.exception("Smoke test FAILED")
                (OUTPUT_DIR / "FAILED_smoke.txt").write_text(str(e))
                return 20
        else:
            logger.info("Smoke test SKIPPED (--skip-smoke)")

        # -- 5 real runs --
        if not args.skip_pops:
            run_meta = []
            for cut in CUTOFFS:
                out_subdir = SWEEP_DIR / f"cutoff_{cut}"
                out_subdir.mkdir(parents=True, exist_ok=True)
                out_prefix = out_subdir / "PGC3_EUR_PoPS"
                preds_path = Path(str(out_prefix) + ".preds")

                # Resume/cache check. WHY: each pops.py run is ~5-10 min;
                # re-running all 5 after a Phase-4 crash wastes 25-50 min.
                # A valid cached preds (>= 15,000 non-NaN rows) is
                # equivalent to what we would re-produce, so skip.
                # --force overrides (e.g., if provenance hashes changed).
                cached_ok = False
                if preds_path.exists() and not args.force:
                    try:
                        cached_n = len(load_preds(preds_path))
                        if cached_n >= 15000:
                            cached_ok = True
                            logger.info(
                                f"cutoff={cut}: skipped (cached) — "
                                f"{cached_n} non-NaN rows in {preds_path}"
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.info(
                            f"cutoff={cut}: cached preds unreadable ({e}); "
                            "re-running"
                        )

                if cached_ok:
                    meta = {
                        "cutoff": cut, "tag": f"cutoff_{cut}", "rc": 0,
                        "wall_s": 0.0,
                        "log_path": str(LOGS_DIR / f"pops_cutoff_{cut}.log"),
                        "preds_path": str(preds_path), "ok": True,
                        "cached": True,
                    }
                else:
                    meta = run_pops_for_cutoff(
                        cutoff=cut, out_prefix=out_prefix,
                        num_chunks=num_chunks, logger=logger,
                        smoke=False, timeout_s=3600,
                    )
                    meta["cached"] = False

                run_meta.append(meta)
                if not meta["ok"]:
                    (OUTPUT_DIR / "FAILED_sweep.txt").write_text(
                        json.dumps(meta, indent=2))
                    logger.error(f"Sweep run failed at cutoff={cut}; halting")
                    return 30
                preds_paths[cut] = Path(meta["preds_path"])
            (OUTPUT_DIR / "sweep_run_meta.json").write_text(
                json.dumps(run_meta, indent=2))
        else:
            logger.info("Sweep runs SKIPPED (--skip-pops); reusing existing preds")
    # collect preds paths if skipped
    if not preds_paths:
        for cut in CUTOFFS:
            p = SWEEP_DIR / f"cutoff_{cut}" / "PGC3_EUR_PoPS.preds"
            if not p.exists():
                logger.error(f"Missing expected preds for cutoff={cut}: {p}")
                return 40
            preds_paths[cut] = p

    # -- Phase 4 analysis --
    try:
        analysis = phase4_analysis(preds_paths, logger)
    except Exception as e:  # noqa: BLE001
        logger.exception("Phase 4 analysis FAILED")
        (OUTPUT_DIR / "FAILED_analysis.txt").write_text(str(e))
        return 50

    # -- Assemble results.json --
    wall = time.time() - t_start
    results = {
        "batch": "054_A",
        "wall_time_min": wall / 60.0,
        "provenance": prov,
        "preds_paths": {str(k): str(v) for k, v in preds_paths.items()},
        "analysis": analysis,
    }
    out_path = OUTPUT_DIR / "results.json"
    # Atomic write. WHY: a SIGINT / OOM during write_text would leave a
    # truncated JSON on disk that downstream tooling (e.g., Silmaril
    # ingest) would load as corrupt data. Writing to a .tmp sibling then
    # os.replace() is atomic within one filesystem on POSIX, so either
    # the old results.json survives intact or the new one is complete.
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(results, indent=2, default=str))
    os.replace(tmp_path, out_path)
    logger.info(f"Wrote {out_path} (wall={wall / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
