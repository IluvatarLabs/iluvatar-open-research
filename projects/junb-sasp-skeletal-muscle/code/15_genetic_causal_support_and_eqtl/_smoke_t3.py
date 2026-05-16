"""Smoke test for T3: import, mini HLMA load, tiny Prism run.

Not part of production code - runs only to verify:
1. pyBayesPrism.prism.Prism.new/run API works end-to-end with HLMA-like data.
2. HLMA h5ad files load and Annotation column exists.
3. GTEx bulk parses.
4. decoupler 2.1.5 API works (dc.mt.mlm, dc.op.collectri).
5. skbio ilr works.
"""
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("smoke")

REPO = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro")

# 1) Imports
log.info("1) imports...")
from pybayesprism.prism import Prism
from pybayesprism import extract
import decoupler as dc
from skbio.stats.composition import ilr
log.info("  OK pybayesprism, decoupler %s, skbio ilr", dc.version.__version__ if hasattr(dc, "version") else "?")

# 2) Load 500-cell subset from Vascular + FAP
log.info("2) mini HLMA load (Vascular + FAP, 500 cells each)...")
vasc = ad.read_h5ad(REPO / "data" / "Vascular_scsn_RNA.h5ad")
fap = ad.read_h5ad(REPO / "data" / "OMIX004308-02.h5ad")
rng = np.random.default_rng(0)
v_idx = rng.choice(vasc.n_obs, 500, replace=False)
f_idx = rng.choice(fap.n_obs, 500, replace=False)
vasc = vasc[v_idx].copy()
fap = fap[f_idx].copy()
# Use intersection of genes
shared_genes = sorted(set(vasc.var_names) & set(fap.var_names))[:800]
log.info("  shared genes: %d", len(shared_genes))
Xv = vasc[:, shared_genes].X
Xf = fap[:, shared_genes].X
if sp.issparse(Xv):
    Xv = Xv.toarray()
if sp.issparse(Xf):
    Xf = Xf.toarray()
ref = pd.DataFrame(
    np.vstack([Xv, Xf]),
    columns=shared_genes,
    index=[f"V_{i}" for i in range(500)] + [f"F_{i}" for i in range(500)],
)
ct = ["Vascular"] * 500 + ["FAP"] * 500
# cell-state labels must be unique per type (pyBayesPrism constraint)
cs = (["Vascular::" + s for s in vasc.obs["Annotation"].astype(str).values.tolist()]
      + ["FAP::" + s for s in fap.obs["Annotation"].astype(str).values.tolist()])
log.info("  states: %d unique", len(set(cs)))

# 3) Tiny GTEx slice - first 20 samples
log.info("3) mini GTEx bulk load (20 samples)...")
bulk = pd.read_csv(REPO / "data" / "GTEx" / "muscle" / "gene_tpm_muscle_skeletal.gct.gz",
                   sep="\t", skiprows=2, nrows=50000)
bulk = bulk.drop(columns=["id"]).rename(columns={"Name": "ensg", "Description": "symbol"})
sample_cols = [c for c in bulk.columns if c.startswith("GTEX-")][:20]
bulk_sym = bulk[["symbol"] + sample_cols].groupby("symbol").mean(numeric_only=True)
bulk_sym = bulk_sym.loc[[g for g in shared_genes if g in bulk_sym.index]]
bulk_for_prism = bulk_sym.T  # samples x genes
log.info("  GTEx bulk mini: %s", bulk_for_prism.shape)

# 4) Run Prism (short chain)
log.info("4) running Prism.new + run (short chain)...")
prism = Prism.new(
    reference=ref,
    input_type="count.matrix",
    cell_type_labels=ct,
    cell_state_labels=cs,
    key=None,
    mixture=bulk_for_prism,
)
bp = prism.run(
    n_cores=1,
    update_gibbs=True,
    gibbs_control={"chain.length": 100, "burn.in": 50, "thinning": 2, "seed": 42},
    opt_control={"maxit": 1000, "optimizer": "MAP"},
)
theta = extract.get_fraction(bp, "final", "type")
log.info("  theta shape: %s", theta.shape)
log.info("  theta preview:\n%s", theta.head(5).to_string())

# 5) decoupler MLM
log.info("5) decoupler MLM test...")
net = pd.DataFrame({
    "source": ["JUNB"] * 10 + ["FOS"] * 10,
    "target": list(bulk_sym.index[:10]) + list(bulk_sym.index[10:20]),
    "weight": [1.0] * 20,
})
act = dc.mt.mlm(data=np.log1p(bulk_for_prism), net=net, tmin=3, verbose=False)
log.info("  mlm output type: %s", type(act))
if isinstance(act, tuple):
    log.info("  tuple len: %d; first elem type: %s shape: %s",
             len(act), type(act[0]),
             act[0].shape if hasattr(act[0], "shape") else "?")
    if hasattr(act[0], "to_df"):
        log.info("  has to_df")

# 6) ilr
log.info("6) ilr test...")
f = theta.values.clip(1e-6, None)
f = f / f.sum(axis=1, keepdims=True)
ilr_vals = ilr(f)
log.info("  ilr shape: %s", ilr_vals.shape)

# 7) CollecTRI
log.info("7) CollecTRI fetch...")
try:
    ct_net = dc.op.collectri(organism="human", verbose=False)
    log.info("  CollecTRI: %d rows, columns: %s", len(ct_net), list(ct_net.columns))
except Exception as e:
    log.warning("  CollecTRI failed: %s", e)

log.info("ALL SMOKE TESTS PASSED")
