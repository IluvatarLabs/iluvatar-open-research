"""Analysis A — Open Targets v4 GraphQL genetic causal support (iter 064).

WHY this script exists: Convert correlational TF-SASP findings from the HLMA
re-analysis into orthogonal genetic causal support by querying Open Targets
(OT 26.03.1, verified via `data/opentargets_probe.py`) for 9 canonical TFs x
6 muscle-aging traits = 54 grid cells. Per brief `experiments/batch_064/brief.md`
Analysis A, PI directive #10. Citations:
  - OT platform: [lit_doi_10.1093_nar_gkac1046] (Ochoa 2023 NAR)
  - MR guidelines:  [lit_doi_10.12688_wellcomeopenres.15555.3] (Burgess 2023)
  - cis-MR drug targets: [lit_doi_10.1038_s41467-020-16969-0] (Schmidt 2020)
  - muscle-aging MR: [lit_doi_10.1111_acel.13923] (Ye 2023)

WHY this shape: the verified probe (`data/opentargets_probe.py`) returns
`target.evidences(efoIds:[$efo])` rows plus `associatedDiseases` scores. We
extend to pull `credibleSet { qtlGeneId, isTransQtl, study{studyType,biosample}
, colocalisation{rows{h4,h3,clpp,colocalisationMethod,rightStudyType}} }`
inside each evidence row so the skeletal-muscle eQTL filter
(UBERON_0001134) and coloc H4 thresholds are applied client-side.

WHY we do NOT run full MR this iter: brief explicitly restricts to an MR
*feasibility* report. GWAS summary stats are not present in `data/` (see
preflight `ls data/` — no sumstats file). Script documents the download URL
and defers actual MR to iter 065 per brief decision rules.

Outputs (all under experiments/batch_064/):
  - a_opentargets_grid.csv   : 54 cells with evidence/datasource scores
  - a_coloc_hits.csv         : credibleSet rows with muscle eQTL coloc H4>0.6
  - a_mr_feasibility.csv     : CDKN1A + CEBPB x {grip_strength, ALM} rows
  - a_summary.json           : verdicts + top-3 TF ranking + counters
  - logs/a_stdout.log        : mirrored stdout with timings
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import requests

# ---------------------------------------------------------------------------
# CONSTANTS (verbatim from brief §Analysis A; TF_ENSG matches opentargets_probe.py)
# ---------------------------------------------------------------------------
API_URL = "https://api.platform.opentargets.org/api/v4/graphql"
USER_AGENT = "sm-rd-research/0.64 (https://github.com/biomarvin/fibro)"

TF_ENSG: dict[str, str] = {
    "JUNB": "ENSG00000171223",
    "FOS": "ENSG00000170345",
    "EGR1": "ENSG00000120738",
    "EGR2": "ENSG00000122877",
    "ATF3": "ENSG00000162772",
    "CEBPB": "ENSG00000172216",
    "KLF10": "ENSG00000155090",
    "IRF1": "ENSG00000125347",
    "CDKN1A": "ENSG00000124762",
}

TRAIT_EFO: dict[str, str] = {
    "sarcopenia": "EFO_1000653",
    "grip_strength": "EFO_0006941",
    "frailty": "EFO_0009885",
    "lean_body_mass": "EFO_0004995",
    "appendicular_lean_mass": "EFO_0004980",
    "muscular_dystrophy": "MONDO_0020121",
}

# Skeletal muscle reference tissue ontology (UBERON) for eQTL biosample filter.
# WHY UBERON_0001134: brief §Analysis A MEASUREMENT pre-registers this id; it is
# the canonical "skeletal muscle tissue" UBERON term used by OT + GTEx.
MUSCLE_UBERON = "UBERON_0001134"

# Cardiometabolic traits used for pleiotropy screen per brief Critic 3 A2 fix.
# EFO ids verified against EBI OLS 2026-04-22; used only for matching coloc study
# disease ids client-side (we fetch non-muscle colocs via the credibleSet's
# colocalisation.rows which already include the cross-trait QTL/GWAS studies).
CARDIOMETABOLIC_EFO: dict[str, str] = {
    "T2D": "MONDO_0005148",            # type 2 diabetes
    "coronary_artery_disease": "EFO_0001645",
    "BMI": "EFO_0004340",
    "LDL": "EFO_0004611",
}

# MR-feasibility target set restricted by brief to CDKN1A + CEBPB.
MR_TARGETS = ["CDKN1A", "CEBPB"]
MR_OUTCOMES = {
    # WHY grip_strength + ALM: brief explicitly names these two outcomes;
    # they are the UK Biobank phenotypes with published sumstats.
    "grip_strength": "EFO_0006941",
    "appendicular_lean_mass": "EFO_0004980",
}

# Published GWAS Catalog accessions for UK Biobank grip strength and ALM.
# URLs are documented only, NOT auto-downloaded this iter (brief: defer to 065).
# WHY these studies: GCST90025994 (Jones 2021, UKB grip) and GCST90025979
# (Pei 2020, UKB appendicular lean mass) are the largest single-trait sumstats
# listed in the GWAS Catalog at time of preflight (2026-04-22). If iter 065
# implements MR, it SHOULD re-verify the accessions before download.
GWAS_CATALOG_SUMSTATS: dict[str, str] = {
    "grip_strength": (
        "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
        "GCST90025001-GCST90026000/GCST90025994/"
    ),
    "appendicular_lean_mass": (
        "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
        "GCST90025001-GCST90026000/GCST90025979/"
    ),
}

# Retry parameters per brief Robustness section.
MAX_RETRIES = 3
BACKOFF_SECONDS = (1, 5, 15)
HTTP_TIMEOUT = 60  # wall-clock per request; OT responses under 5s typical.

# Thresholds (pre-registered in brief).
H4_STRICT = 0.80  # primary muscle-coloc threshold
H4_MULTI_TRAIT = 0.60  # pleiotropy screen threshold for non-muscle colocs
MIN_CELLS_FOR_SUGGESTED = 3  # brief decision rule

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path("/home/yuanz/Documents/GitHub/biomarvin_fibro/experiments/batch_064")
LOG_DIR = HERE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

GRID_CSV = HERE / "a_opentargets_grid.csv"
COLOC_CSV = HERE / "a_coloc_hits.csv"
MR_CSV = HERE / "a_mr_feasibility.csv"
SUMMARY_JSON = HERE / "a_summary.json"
STDOUT_LOG = LOG_DIR / "a_stdout.log"


# ---------------------------------------------------------------------------
# Logging: mirror all prints to stdout AND STDOUT_LOG with timings.
# WHY: brief requires stdout+file logging for auditability.
# ---------------------------------------------------------------------------
class TeeLogger:
    """Minimal tee: write to real stdout + file, flush per line for live tail."""

    def __init__(self, path: Path) -> None:
        self._fh = open(path, "a", buffering=1)
        self._start = time.time()
        self._real_stdout = sys.__stdout__

    def log(self, msg: str) -> None:
        elapsed = time.time() - self._start
        line = f"[{elapsed:8.2f}s] {msg}"
        print(line, file=self._real_stdout, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# HTTP with retry + fallback on schema errors.
# ---------------------------------------------------------------------------
def build_session() -> requests.Session:
    """WHY a session: requests.Session pools TCP connections; OT requires ~54
    sequential POSTs, so pooling shaves ~10-30% of wall time."""
    s = requests.Session()
    s.headers.update({
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    })
    return s


def gql_post(
    session: requests.Session,
    query: str,
    variables: dict[str, Any],
    logger: TeeLogger,
    context: str,
) -> dict | None:
    """Single POST with retry on 5xx/429/timeout.

    Returns the parsed JSON dict (OT responses always include 'data' and
    optionally 'errors'). Returns None only when retries are exhausted -- a
    None should be treated as a transient failure for the cell, not as a
    semantic zero.
    """
    body = {"query": query, "variables": variables}
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.post(API_URL, json=body, timeout=HTTP_TIMEOUT)
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.log(f"WARN [{context}] network error attempt {attempt+1}: {exc}")
            if attempt + 1 < MAX_RETRIES:
                time.sleep(BACKOFF_SECONDS[attempt])
                continue
            return None

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError as exc:
                logger.log(f"WARN [{context}] JSON decode failed: {exc}")
                return None

        if resp.status_code in (429, 500, 502, 503, 504):
            logger.log(
                f"WARN [{context}] HTTP {resp.status_code} attempt {attempt+1}"
                f" body[:160]={resp.text[:160]!r}"
            )
            if attempt + 1 < MAX_RETRIES:
                # WHY honor Retry-After: on 429 the server tells us its preferred
                # cool-down. Ignoring it burns our retry budget inside the rate
                # window and drops cells. Floor at BACKOFF_SECONDS[attempt] so we
                # never sleep shorter than the pre-registered schedule, and floor
                # at 5s when the header is missing (brief Robustness section).
                retry_after_raw = resp.headers.get("Retry-After", "0")
                try:
                    retry_after = int(retry_after_raw)
                except (TypeError, ValueError):
                    retry_after = 0
                sleep_for = max(BACKOFF_SECONDS[attempt], retry_after, 5)
                time.sleep(sleep_for)
                continue
            return None

        # 4xx other than 429: not retryable.
        logger.log(
            f"WARN [{context}] HTTP {resp.status_code} non-retryable:"
            f" {resp.text[:160]!r}"
        )
        return None

    return None


# ---------------------------------------------------------------------------
# GraphQL queries.
#
# WHY two shapes: OT schemas occasionally drop fields between releases. We
# attempt the FULL shape (with credibleSet + colocalisation) first, and if
# OT returns a "Cannot query field ..." error we fall back to the minimal
# shape matching the verified probe (no credibleSet). This matches brief
# Robustness: "On schema errors, fall back to simpler query and log WARN."
# ---------------------------------------------------------------------------
QUERY_FULL = """
query TargetDiseaseFull($ensg: String!, $efo: String!) {
  target(ensemblId: $ensg) {
    id
    approvedSymbol
    evidences(efoIds: [$efo], size: 100) {
      count
      rows {
        disease { id name }
        datatypeId
        datasourceId
        score
        literature
        studyId
        credibleSet {
          studyLocusId
          qtlGeneId
          isTransQtl
          pValueMantissa
          pValueExponent
          study {
            id
            studyType
            biosample { biosampleId biosampleName }
          }
          colocalisation {
            rows {
              h4
              h3
              clpp
              colocalisationMethod
              rightStudyType
            }
          }
        }
      }
    }
    associatedDiseases(page: {index: 0, size: 50}, orderByScore: "score") {
      count
      rows {
        disease { id name }
        score
        datatypeScores { id score }
        datasourceScores { id score }
      }
    }
  }
}
"""

QUERY_MINIMAL = """
query TargetDiseaseMinimal($ensg: String!, $efo: String!) {
  target(ensemblId: $ensg) {
    id
    approvedSymbol
    evidences(efoIds: [$efo], size: 100) {
      count
      rows {
        disease { id name }
        datatypeId
        datasourceId
        score
        literature
        studyId
      }
    }
    associatedDiseases(page: {index: 0, size: 50}, orderByScore: "score") {
      count
      rows {
        disease { id name }
        score
        datatypeScores { id score }
        datasourceScores { id score }
      }
    }
  }
}
"""


def _has_schema_error(payload: dict | None) -> bool:
    """Return True iff the OT response contains a GraphQL 'Cannot query field'
    error, i.e. the full schema shape is unsupported by the running OT release."""
    if not payload or "errors" not in payload:
        return False
    for err in payload.get("errors") or []:
        msg = (err.get("message") or "").lower()
        if "cannot query field" in msg or "unknown field" in msg:
            return True
    return False


def fetch_cell(
    session: requests.Session, ensg: str, efo: str, logger: TeeLogger, context: str
) -> tuple[dict | None, str]:
    """Try FULL, fall back to MINIMAL on schema error. Return (payload, shape)."""
    payload = gql_post(session, QUERY_FULL, {"ensg": ensg, "efo": efo}, logger, context)
    if _has_schema_error(payload):
        logger.log(
            f"WARN [{context}] OT schema rejected credibleSet fields; falling back"
        )
        payload = gql_post(
            session, QUERY_MINIMAL, {"ensg": ensg, "efo": efo}, logger, context
        )
        return payload, "minimal"
    return payload, "full"


# ---------------------------------------------------------------------------
# Extraction helpers.
# ---------------------------------------------------------------------------
def _overall_score_for_disease(
    associated: dict | None, efo: str
) -> tuple[float | None, dict[str, float], dict[str, float]]:
    """Scan `associatedDiseases.rows` and return (score, datatype_scores,
    datasource_scores) for the disease matching `efo`. Returns (None, {}, {})
    if the disease is not in the top-50 associated diseases for this target.

    WHY client-side filter: the OT 26.03.1 `associatedDiseases` field has no
    efoIds filter (confirmed by probe). We fetch the top 50 by score and scan.
    If the disease rank is >50 for the target, overall_score for that cell is
    not retrievable via this endpoint and is reported as null (not zero -- this
    is missing data, not measured zero, per Rule 0).
    """
    if not associated or "rows" not in associated:
        return None, {}, {}
    for row in associated.get("rows") or []:
        if (row.get("disease") or {}).get("id") == efo:
            datatype = {d["id"]: d["score"] for d in (row.get("datatypeScores") or [])}
            datasource = {
                d["id"]: d["score"] for d in (row.get("datasourceScores") or [])
            }
            return row.get("score"), datatype, datasource
    return None, {}, {}


def _aggregate_evidence_scores(
    evidences: dict | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate evidence-level scores per datasourceId AND per datatypeId
    (max score per key).

    WHY both dicts: the brief's grid columns (`genetic_association_score`,
    `rna_expression_score`, `literature_score`, `affected_pathway_score`)
    are OT **datatype** IDs, not datasource IDs. Returning a datatype-keyed
    dict lets `_src` populate those columns even when the disease is below
    the associatedDiseases top-50 cutoff (which is the case for all
    muscle-aging traits against the 9 TFs in this sweep — they rank deep in
    each target's long-tail of 1k+ associated diseases). A datasource-keyed
    dict is retained for callers who reference specific datasources (e.g.
    ot_genetics_portal) in future extensions.

    WHY max: OT returns one row per piece of evidence; per-key max is the
    conservative summary for "is there any supporting evidence of this type
    or source?" It matches how OT's own overall-score aggregation takes
    the harmonic-sum-scaled max per category.

    Returns: (by_datasource, by_datatype).
    """
    by_datasource: dict[str, float] = {}
    by_datatype: dict[str, float] = {}
    if not evidences or "rows" not in evidences:
        return by_datasource, by_datatype
    for row in evidences.get("rows") or []:
        score = row.get("score")
        if score is None:
            continue
        s = float(score)
        dsid = row.get("datasourceId")
        dtid = row.get("datatypeId")
        if dsid is not None:
            by_datasource[dsid] = max(by_datasource.get(dsid, 0.0), s)
        if dtid is not None:
            by_datatype[dtid] = max(by_datatype.get(dtid, 0.0), s)
    return by_datasource, by_datatype


@dataclass
class ColocHit:
    tf: str
    trait: str
    evidence_datasource: str
    credible_set_id: str | None
    qtl_gene_id: str | None
    is_trans_qtl: bool | None
    study_id: str | None
    study_type: str | None
    biosample_ontology: str | None
    biosample_name: str | None
    h4: float
    h3: float | None
    clpp: float | None
    coloc_method: str | None
    right_study_type: str | None
    p_value_mantissa: float | None
    p_value_exponent: int | None
    pleiotropic: bool | None = None
    non_muscle_cardiometabolic_h4: dict[str, float] = field(default_factory=dict)


def _iter_coloc_rows(evidence_row: dict) -> Iterable[dict]:
    cs = evidence_row.get("credibleSet") or {}
    coloc = cs.get("colocalisation") or {}
    for r in coloc.get("rows") or []:
        yield r


def extract_muscle_coloc_hits(
    tf: str, trait: str, evidences: dict | None
) -> list[ColocHit]:
    """Return credibleSet rows where biosample.biosampleId == UBERON_0001134
    AND any colocalisation h4 >= H4_MULTI_TRAIT (0.6). Stricter H4_STRICT
    (0.8) filtering is applied by callers for decision rules.

    WHY biosampleId (not ontologyId): OT 26.03.1 Biosample type exposes the
    UBERON string via `biosampleId` (introspected 2026-04-22). Prior code
    used `ontologyId` which does not exist on this release and caused HTTP
    400 "Cannot query field 'ontologyId' on type 'Biosample'".

    WHY H4>=0.6 in-collection: brief §Analysis A MEASUREMENT requires
    `a_coloc_hits.csv` to include "credibleSets with skeletal muscle eQTL
    coloc H4>0.6" (the pleiotropy-screen threshold). Strict H4>0.8 is a
    downstream filter.
    """
    hits: list[ColocHit] = []
    if not evidences:
        return hits
    for row in evidences.get("rows") or []:
        cs = row.get("credibleSet") or {}
        if not cs:
            continue
        study = cs.get("study") or {}
        biosample = study.get("biosample") or {}
        ontology = biosample.get("biosampleId")
        if ontology != MUSCLE_UBERON:
            continue
        if (study.get("studyType") or "").lower() not in ("eqtl", "qtl"):
            # brief: filter eQTL evidence only; skip GWAS/pqtl credible sets.
            # WHY substring-match: OT has used both "eqtl" and "QTL" labels
            # across releases; we permit both.
            continue
        for croloc in _iter_coloc_rows(row):
            h4 = croloc.get("h4")
            if h4 is None or h4 < H4_MULTI_TRAIT:
                continue
            hits.append(
                ColocHit(
                    tf=tf,
                    trait=trait,
                    evidence_datasource=row.get("datasourceId") or "",
                    # WHY studyLocusId: OT 26.03.1 CredibleSet has no `id` field
                    # (introspected 2026-04-22); studyLocusId is the NON_NULL
                    # primary key used as the argument to Query.credibleSet().
                    credible_set_id=cs.get("studyLocusId"),
                    qtl_gene_id=cs.get("qtlGeneId"),
                    is_trans_qtl=cs.get("isTransQtl"),
                    study_id=study.get("id"),
                    study_type=study.get("studyType"),
                    biosample_ontology=ontology,
                    biosample_name=biosample.get("biosampleName"),
                    h4=float(h4),
                    h3=croloc.get("h3"),
                    clpp=croloc.get("clpp"),
                    coloc_method=croloc.get("colocalisationMethod"),
                    right_study_type=croloc.get("rightStudyType"),
                    p_value_mantissa=cs.get("pValueMantissa"),
                    p_value_exponent=cs.get("pValueExponent"),
                )
            )
    return hits


# ---------------------------------------------------------------------------
# Pleiotropy check: for each muscle-coloc hit, query the credibleSet again via
# OT's credibleSet root to retrieve its non-muscle coloc partners.
#
# WHY a second query: the evidence-row embed gives us coloc rows under the
# muscle-trait credible set, but to check pleiotropy we need the full coloc
# table including non-muscle GWAS partners. OT exposes this via
# `credibleSet(studyLocusId: $id) { colocalisation { rows { ... } } }`.
# If that field isn't available in the running schema, we log WARN and leave
# pleiotropic=None (unknown) rather than fabricate a value (Rule 0).
# ---------------------------------------------------------------------------
# WHY runtime version probe: the hardcoded "26.03.1" is from the preflight probe
# dated 2026-04-22. If OT silently upgraded between probe and execution, we want
# the summary to record the LIVE version, not a fabricated tag. On schema failure
# we record "unknown" (Rule 0: never fabricate) and continue.
QUERY_META = """
query Meta {
  meta {
    apiVersion {
      x
      y
      z
    }
  }
}
"""


def fetch_ot_api_version(
    session: requests.Session, logger: TeeLogger
) -> str:
    """Return the live OT apiVersion string "x.y.z" or "unknown" on failure."""
    payload = gql_post(session, QUERY_META, {}, logger, "meta_version")
    if payload is None or _has_schema_error(payload):
        return "unknown"
    meta = ((payload.get("data") or {}).get("meta")) or {}
    v = meta.get("apiVersion") or {}
    x, y, z = v.get("x"), v.get("y"), v.get("z")
    if x is None or y is None or z is None:
        return "unknown"
    return f"{x}.{y}.{z}"


QUERY_CRED_SET = """
query CredSet($id: String!) {
  credibleSet(studyLocusId: $id) {
    studyLocusId
    colocalisation {
      rows {
        h4
        h3
        rightStudyType
        otherStudyLocus {
          study {
            id
            studyType
            diseases { id name }
            traitFromSource
          }
        }
      }
    }
  }
}
"""


def pleiotropy_check(
    session: requests.Session, hit: ColocHit, logger: TeeLogger
) -> None:
    """Populate hit.pleiotropic and hit.non_muscle_cardiometabolic_h4 in place.

    Logic (brief): for the credibleSet, list all non-muscle-trait colocs at
    H4>=0.6; count how many map to {T2D, CAD, BMI, LDL}. If >=2 cardiometabolic
    coloc partners present, set pleiotropic=True; else False.
    """
    if not hit.credible_set_id:
        hit.pleiotropic = None
        return
    ctx = f"pleio:{hit.tf}/{hit.trait}/{hit.credible_set_id}"
    payload = gql_post(
        session, QUERY_CRED_SET, {"id": hit.credible_set_id}, logger, ctx
    )
    if _has_schema_error(payload) or payload is None:
        logger.log(f"WARN [{ctx}] pleiotropy query unsupported; leaving pleiotropic=None")
        hit.pleiotropic = None
        return
    cs = ((payload.get("data") or {}).get("credibleSet")) or {}
    rows = ((cs.get("colocalisation") or {}).get("rows")) or []
    cm_hits: dict[str, float] = {}
    for r in rows:
        h4 = r.get("h4")
        if h4 is None or h4 < H4_MULTI_TRAIT:
            continue
        other = ((r.get("otherStudyLocus") or {}).get("study")) or {}
        diseases = other.get("diseases") or []
        for d in diseases:
            d_id = d.get("id")
            for cm_name, cm_id in CARDIOMETABOLIC_EFO.items():
                if d_id == cm_id:
                    # Keep the max H4 per cardiometabolic category.
                    cm_hits[cm_name] = max(cm_hits.get(cm_name, 0.0), float(h4))
    hit.non_muscle_cardiometabolic_h4 = cm_hits
    hit.pleiotropic = len(cm_hits) >= 2


# ---------------------------------------------------------------------------
# MR feasibility (cis-eQTL count + F-stat estimate + coloc H4 w/ outcome).
# Per brief: do NOT run MR; document gap in GWAS sumstats.
# ---------------------------------------------------------------------------
def _f_stat_from_p(p_mantissa: float | None, p_exponent: int | None) -> float | None:
    """Estimate F-statistic from a cis-eQTL p-value using F ~= chi^2(1 df) for
    a single-IV Wald z-score. We invert the two-sided normal CDF:
        z = Phi^{-1}(1 - p/2); F ~= z^2.
    WHY: brief Robustness specifies "F ~= chi^2(1) from p; mark as estimate."
    We rely only on stdlib (math.erfcinv not in stdlib, so use Newton iteration
    on math.erfc). Mark as estimate because actual F uses beta^2 / se^2.
    """
    if p_mantissa is None or p_exponent is None:
        return None
    try:
        p = float(p_mantissa) * (10.0 ** int(p_exponent))
    except (TypeError, ValueError):
        return None
    if p <= 0 or p >= 1:
        return None
    # Newton on erfc: we want z such that erfc(z/sqrt(2)) = p (two-sided).
    # erfc is strictly decreasing so Newton converges quickly from z=4.
    import statistics  # noqa: F401  (kept for future bootstrap if expanded)

    # Use math-only: Phi^{-1}(1 - p/2) via series for extreme tails, else Newton.
    # For numerical stability, bound p to [1e-300, 1-1e-16].
    p = max(1e-300, min(p, 1 - 1e-16))
    target = p  # erfc(z/sqrt(2)) == p
    z = 4.0
    for _ in range(60):
        f = math.erfc(z / math.sqrt(2.0)) - target
        # derivative: d/dz erfc(z/sqrt(2)) = -sqrt(2/pi) * exp(-z^2/2)
        fprime = -math.sqrt(2.0 / math.pi) * math.exp(-(z * z) / 2.0)
        if fprime == 0:
            break
        step = f / fprime
        z_new = z - step
        if z_new < 0:
            z_new = z / 2.0
        if abs(z_new - z) < 1e-10:
            z = z_new
            break
        z = z_new
    return z * z


def mr_feasibility_rows(
    cell_records: dict[tuple[str, str], dict],
    logger: TeeLogger,
) -> list[dict]:
    """Build MR feasibility rows for MR_TARGETS x MR_OUTCOMES = 2x2 = 4 rows.

    Uses already-fetched payloads in `cell_records` (no extra HTTP).
    """
    rows: list[dict] = []
    for tf in MR_TARGETS:
        for outcome_name, outcome_efo in MR_OUTCOMES.items():
            rec = cell_records.get((tf, outcome_name), {})
            evidences = rec.get("evidences") or {}
            muscle_eqtls: list[dict] = []
            coloc_h4_with_outcome: list[float] = []
            f_stats: list[float] = []
            for row in (evidences.get("rows") or []):
                cs = row.get("credibleSet") or {}
                if not cs:
                    continue
                study = cs.get("study") or {}
                biosample = study.get("biosample") or {}
                if biosample.get("biosampleId") != MUSCLE_UBERON:
                    continue
                if (study.get("studyType") or "").lower() not in ("eqtl", "qtl"):
                    continue
                muscle_eqtls.append(cs)
                f = _f_stat_from_p(
                    cs.get("pValueMantissa"), cs.get("pValueExponent")
                )
                if f is not None:
                    f_stats.append(f)
                for croloc in (cs.get("colocalisation") or {}).get("rows") or []:
                    h4 = croloc.get("h4")
                    if h4 is not None:
                        coloc_h4_with_outcome.append(float(h4))

            n_iv = len(muscle_eqtls)
            mean_f = sum(f_stats) / len(f_stats) if f_stats else None
            max_h4 = max(coloc_h4_with_outcome) if coloc_h4_with_outcome else None
            sumstats_url = GWAS_CATALOG_SUMSTATS.get(outcome_name, "")
            sumstats_local_path = None  # brief: not downloaded this iter.
            # Verdict per brief decision rule (caps at SUGGESTED if <3 IVs).
            # WHY: ESTABLISHED requires 6 conditions incl. MR runs — impossible
            # without sumstats on disk. We encode the ceiling given available data.
            # WHY hoist mean_f format spec out of the f-string: Python parses
            # `{x:.1f if cond else ...}` as a single format-spec token and raises
            # ValueError. Format first, interpolate second.
            mean_f_str = f"{mean_f:.1f}" if mean_f is not None else "nan"
            if n_iv == 0:
                verdict = "INCONCLUSIVE"
                reason = "No cis-eQTL in UBERON_0001134 found in OT evidences."
            elif n_iv <= 2:
                # WHY SUGGESTED (not SPECULATIVE): brief.md line 47 "1-2 IVs ->
                # SUGGESTED regardless of p-value". Aligns with pre-registered
                # decision rule.
                verdict = "SUGGESTED"
                reason = (
                    f"{n_iv} IV(s) in muscle; brief caps <3 IV at SUGGESTED."
                )
            elif mean_f is not None and mean_f < 10:
                verdict = "INCONCLUSIVE"
                reason = f"Weak-instrument: mean F~{mean_f_str} < 10."
            else:
                verdict = "SUGGESTED"
                reason = (
                    f"{n_iv} cis-eQTL IVs in muscle; mean F~{mean_f_str}. "
                    "MR not run (no local sumstats); defer to iter 065."
                )
            rows.append(
                {
                    "tf": tf,
                    "outcome": outcome_name,
                    "outcome_efo": outcome_efo,
                    "n_cis_eqtls_muscle": n_iv,
                    "mean_F_stat_estimate": mean_f,
                    "max_coloc_H4_with_outcome": max_h4,
                    "gwas_sumstats_url": sumstats_url,
                    "gwas_sumstats_local_path": sumstats_local_path,
                    "verdict": verdict,
                    "reason": reason,
                }
            )
            logger.log(
                f"MR-feas [{tf} x {outcome_name}] n_iv={n_iv} mean_F="
                f"{mean_f} max_H4={max_h4} verdict={verdict}"
            )
    return rows


# ---------------------------------------------------------------------------
# CSV writers (hand-rolled to keep the script dependency-light).
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict], header: list[str]) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------
def run(dry_run: bool = False) -> int:
    logger = TeeLogger(STDOUT_LOG)
    logger.log("START run_a_opentargets.py")
    logger.log(f"TFs={list(TF_ENSG)} traits={list(TRAIT_EFO)}")
    logger.log(
        f"Output: grid={GRID_CSV.name} coloc={COLOC_CSV.name} "
        f"mr={MR_CSV.name} summary={SUMMARY_JSON.name}"
    )
    if dry_run:
        logger.log("DRY-RUN: exiting without HTTP calls.")
        logger.close()
        return 0

    session = build_session()
    grid_rows: list[dict] = []
    all_coloc_hits: list[ColocHit] = []
    cell_records: dict[tuple[str, str], dict] = {}

    # WHY probe apiVersion up front: if OT upgraded between preflight and now,
    # record the live version instead of the stale hardcoded tag (Rule 0).
    ot_api_version_live = fetch_ot_api_version(session, logger)
    logger.log(f"OT live apiVersion = {ot_api_version_live}")

    # WHY try/finally around the main loop: if any cell trips an uncaught
    # exception, we still want to emit a_summary.json reflecting partial state
    # (brief requires summary on API-down scenarios). CSV writes also happen in
    # the finally block so in-memory rows are not lost.
    cells_populated = 0
    run_error: Exception | None = None
    mr_rows: list[dict] = []
    try:
        for tf, ensg in TF_ENSG.items():
            for trait, efo in TRAIT_EFO.items():
                context = f"cell:{tf}x{trait}"
                t0 = time.time()
                payload, shape = fetch_cell(session, ensg, efo, logger, context)
                dt = time.time() - t0
                if payload is None or "data" not in payload:
                    logger.log(f"WARN [{context}] no data after retries (dt={dt:.1f}s)")
                    grid_rows.append(_empty_grid_row(tf, trait, ensg, efo, shape="FAILED"))
                    continue

                target = (payload.get("data") or {}).get("target")
                if not target:
                    logger.log(
                        f"WARN [{context}] target missing for ensg={ensg};"
                        " OT may not know this id"
                    )
                    grid_rows.append(
                        _empty_grid_row(tf, trait, ensg, efo, shape=shape)
                    )
                    continue

                evidences = target.get("evidences") or {}
                associated = target.get("associatedDiseases") or {}
                overall_score, datatype_scores, datasource_scores = (
                    _overall_score_for_disease(associated, efo)
                )
                agg_by_datasource, agg_by_datatype = _aggregate_evidence_scores(evidences)

                # WHY prefer datatype_scores: the grid columns
                # (genetic_association_score, rna_expression_score,
                # literature_score, affected_pathway_score) are OT DATATYPE
                # IDs, not datasource IDs. When the disease is in the
                # target's associatedDiseases top-50 we use the official
                # per-datatype score; otherwise we fall back to the
                # per-datatype max aggregated from `evidences.rows`. A
                # final fallback to the datasource aggregate is retained
                # only for safety (never hit in practice for these IDs).
                def _src(name: str) -> float | None:
                    if name in datatype_scores:
                        return datatype_scores[name]
                    if name in agg_by_datatype:
                        return agg_by_datatype[name]
                    if name in datasource_scores:
                        return datasource_scores[name]
                    return agg_by_datasource.get(name)

                muscle_hits = extract_muscle_coloc_hits(tf, trait, evidences)
                n_h4_strict = sum(1 for h in muscle_hits if h.h4 >= H4_STRICT)

                grid_rows.append({
                    "tf": tf,
                    "ensg": ensg,
                    "trait": trait,
                    "efo": efo,
                    "query_shape": shape,
                    "evidence_count": evidences.get("count"),
                    "overall_score": overall_score,
                    "overall_score_source": (
                        "associatedDiseases"
                        if overall_score is not None
                        else "not_in_top50"
                    ),
                    "genetic_association_score": _src("genetic_association"),
                    "rna_expression_score": _src("rna_expression"),
                    "literature_score": _src("literature"),
                    "affected_pathway_score": _src("affected_pathway"),
                    "n_coloc_h4_gt_0.8_muscle": n_h4_strict,
                    "muscle_eqtl_credible_set_ids": ";".join(
                        sorted({h.credible_set_id or "" for h in muscle_hits if h.h4 >= H4_STRICT})
                    ),
                    "fetch_seconds": round(dt, 2),
                })
                cell_records[(tf, trait)] = {
                    "evidences": evidences,
                    "associated": associated,
                }
                cells_populated += 1
                all_coloc_hits.extend(muscle_hits)
                logger.log(
                    f"[{context}] shape={shape} evidence_count={evidences.get('count')} "
                    f"overall_score={overall_score} muscle_coloc_H4>0.8={n_h4_strict} "
                    f"dt={dt:.1f}s"
                )

        # ----- Pleiotropy screen on strict-H4 hits only. -----
        logger.log(
            f"Pleiotropy screen on {sum(1 for h in all_coloc_hits if h.h4 >= H4_STRICT)} strict hits"
        )
        for hit in all_coloc_hits:
            if hit.h4 >= H4_STRICT:
                pleiotropy_check(session, hit, logger)

        # ----- MR feasibility (no extra HTTP; reuses cell_records). -----
        mr_rows = mr_feasibility_rows(cell_records, logger)

    except Exception as exc:  # noqa: BLE001 — we rethrow after emitting summary
        # WHY catch-all: any uncaught exception mid-sweep would otherwise leave
        # zero files on disk despite having populated grid_rows in memory.
        # We log, stash the error for the summary, and fall through to finally.
        run_error = exc
        logger.log(
            f"ERROR run() aborted mid-loop: {type(exc).__name__}: {exc}"
        )

    finally:
        # ----- Write CSVs (partial-state safe). -----
        # WHY in finally: even on exception we want any rows we already built to
        # land on disk so the audit trail is preserved (Rule 0: report failures
        # with the same detail as successes).
        try:
            _write_csv(
                GRID_CSV,
                grid_rows,
                header=[
                    "tf", "ensg", "trait", "efo", "query_shape",
                    "evidence_count", "overall_score", "overall_score_source",
                    "genetic_association_score", "rna_expression_score",
                    "literature_score", "affected_pathway_score",
                    "n_coloc_h4_gt_0.8_muscle", "muscle_eqtl_credible_set_ids",
                    "fetch_seconds",
                ],
            )
        except Exception as csv_exc:  # noqa: BLE001
            logger.log(f"ERROR writing GRID_CSV: {csv_exc}")

        coloc_rows = [
            {
                "tf": h.tf,
                "trait": h.trait,
                "evidence_datasource": h.evidence_datasource,
                "credible_set_id": h.credible_set_id,
                "qtl_gene_id": h.qtl_gene_id,
                "is_trans_qtl": h.is_trans_qtl,
                "study_id": h.study_id,
                "study_type": h.study_type,
                "biosample_ontology": h.biosample_ontology,
                "biosample_name": h.biosample_name,
                "h4": h.h4,
                "h3": h.h3,
                "clpp": h.clpp,
                "coloc_method": h.coloc_method,
                "right_study_type": h.right_study_type,
                "p_value_mantissa": h.p_value_mantissa,
                "p_value_exponent": h.p_value_exponent,
                "pleiotropic": h.pleiotropic,
                "cardiometabolic_non_muscle_h4": json.dumps(
                    h.non_muscle_cardiometabolic_h4, sort_keys=True
                ),
            }
            for h in all_coloc_hits
        ]
        try:
            _write_csv(
                COLOC_CSV,
                coloc_rows,
                header=[
                    "tf", "trait", "evidence_datasource", "credible_set_id",
                    "qtl_gene_id", "is_trans_qtl", "study_id", "study_type",
                    "biosample_ontology", "biosample_name", "h4", "h3", "clpp",
                    "coloc_method", "right_study_type", "p_value_mantissa",
                    "p_value_exponent", "pleiotropic", "cardiometabolic_non_muscle_h4",
                ],
            )
        except Exception as csv_exc:  # noqa: BLE001
            logger.log(f"ERROR writing COLOC_CSV: {csv_exc}")

        try:
            _write_csv(
                MR_CSV,
                mr_rows,
                header=[
                    "tf", "outcome", "outcome_efo", "n_cis_eqtls_muscle",
                    "mean_F_stat_estimate", "max_coloc_H4_with_outcome",
                    "gwas_sumstats_url", "gwas_sumstats_local_path",
                    "verdict", "reason",
                ],
            )
        except Exception as csv_exc:  # noqa: BLE001
            logger.log(f"ERROR writing MR_CSV: {csv_exc}")

        # ----- Summary JSON (always written; reflects partial state on error). -----
        total_coloc_h4_strict_muscle = sum(
            1 for h in all_coloc_hits if h.h4 >= H4_STRICT
        )
        pleiotropic_hits = sum(
            1 for h in all_coloc_hits if h.h4 >= H4_STRICT and h.pleiotropic is True
        )
        clean_muscle_cells = len(
            {
                (h.tf, h.trait)
                for h in all_coloc_hits
                if h.h4 >= H4_STRICT and h.pleiotropic is not True
            }
        )
        if run_error is not None:
            grid_verdict = "PARTIAL_FAILURE"
            grid_reason = (
                f"run() aborted: {type(run_error).__name__}: {run_error}. "
                f"{cells_populated}/{len(TF_ENSG) * len(TRAIT_EFO)} cells populated."
            )
        elif clean_muscle_cells >= MIN_CELLS_FOR_SUGGESTED:
            grid_verdict = "SUGGESTED"
            grid_reason = (
                f"{clean_muscle_cells} TFxtrait cells pass muscle-coloc H4>0.8 "
                "AND pleiotropy screen."
            )
        elif 1 <= clean_muscle_cells < MIN_CELLS_FOR_SUGGESTED:
            grid_verdict = "SPECULATIVE"
            grid_reason = (
                f"{clean_muscle_cells} cell(s) pass both filters; brief requires "
                ">=3 for SUGGESTED."
            )
        elif total_coloc_h4_strict_muscle == 0:
            grid_verdict = "NEGATIVE"
            grid_reason = (
                "0 muscle-coloc cells; pre-registered interpretation: IEG post-"
                "translational regulation confirmed — mechanism constraint, not refutation."
            )
        else:
            grid_verdict = "INCONCLUSIVE"
            grid_reason = (
                f"{total_coloc_h4_strict_muscle} coloc hits but all flagged pleiotropic "
                "(cardiometabolic-dominated)."
            )

        # Top-3 TFs by genetic_association score summed across traits.
        ga_by_tf: dict[str, float] = {tf: 0.0 for tf in TF_ENSG}
        for r in grid_rows:
            v = r.get("genetic_association_score")
            if v is not None:
                ga_by_tf[r["tf"]] += float(v)
        top_tfs = sorted(ga_by_tf.items(), key=lambda kv: -kv[1])[:3]

        mr_verdicts = {
            f"{r['tf']}_x_{r['outcome']}": {"verdict": r["verdict"], "reason": r["reason"]}
            for r in mr_rows
        }

        summary = {
            "iteration": 64,
            "analysis": "A_opentargets",
            "ot_endpoint": API_URL,
            "ot_release_probe": "26.03.1 (probe dated 2026-04-22)",
            "ot_release_live": ot_api_version_live,
            "status": "partial_failure" if run_error is not None else "complete",
            "error": (
                f"{type(run_error).__name__}: {run_error}"
                if run_error is not None
                else None
            ),
            "total_cells_populated": cells_populated,
            "total_cells_total": len(TF_ENSG) * len(TRAIT_EFO),
            "total_coloc_h4_gt_0.8_muscle": total_coloc_h4_strict_muscle,
            "pleiotropic_hits": pleiotropic_hits,
            "clean_muscle_cells": clean_muscle_cells,
            "grid_verdict": grid_verdict,
            "grid_reason": grid_reason,
            "mr_feasibility_verdicts": mr_verdicts,
            "top3_tfs_by_genetic_association": [
                {"tf": tf, "sum_ga_score": round(s, 4)} for tf, s in top_tfs
            ],
            "notes": [
                "MR not executed this iter: no GWAS summary stats present in data/. "
                "Download URLs documented in a_mr_feasibility.csv; MR deferred to iter 065.",
                "Pleiotropy check leaves pleiotropic=None when OT credibleSet root "
                "schema is unsupported by the running release (schema fallback logged).",
            ],
        }
        try:
            SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
        except Exception as js_exc:  # noqa: BLE001
            logger.log(f"ERROR writing SUMMARY_JSON: {js_exc}")

        logger.log(
            f"END cells_populated={cells_populated}/54 "
            f"coloc_H4>0.8={total_coloc_h4_strict_muscle} "
            f"clean_muscle_cells={clean_muscle_cells} verdict={grid_verdict}"
        )
        logger.close()

    # Re-raise after finally so the shell exit code reflects failure; this is
    # intentional — we want CI / dispatch to see a non-zero exit, but only
    # after summary/CSVs have been persisted.
    if run_error is not None:
        raise run_error
    return 0


def _empty_grid_row(tf: str, trait: str, ensg: str, efo: str, shape: str) -> dict:
    return {
        "tf": tf,
        "ensg": ensg,
        "trait": trait,
        "efo": efo,
        "query_shape": shape,
        "evidence_count": None,
        "overall_score": None,
        "overall_score_source": "unavailable",
        "genetic_association_score": None,
        "rna_expression_score": None,
        "literature_score": None,
        "affected_pathway_score": None,
        "n_coloc_h4_gt_0.8_muscle": 0,
        "muscle_eqtl_credible_set_ids": "",
        "fetch_seconds": None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate imports and paths without hitting the OT API.",
    )
    args = p.parse_args()
    return run(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
