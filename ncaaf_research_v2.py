"""NCAAF Research V2.31.1 — deep objective context source repair for System Miner.

Research-only architecture built around the frozen NCAAF Production V1 benchmark.
This module cannot mutate CORE/model probability. Historically qualified system families may emit bounded
confirmation/conflict votes for Bet Authority; no research family can create a CORE candidate.

Core contracts
--------------
* Production V1 remains frozen and separate.
* Discovery is capped at 2023; 2024-2025 are untouched confirmation seasons.
* 2026+ is prospective only and never participates in search, threshold choice,
  family selection, direction choice, or confirmation.
* STAT research predicts residual error of the incumbent OOF fair-value backbones.
* System mining searches interpretable pregame conditions, applies FDR and
  season robustness, then collapses dependent/near-duplicate systems into
  mechanism families before confirmation.
* Market-rich inputs are accepted only when already present leakage-safely in
  the historical research frame. Current market data continues to originate in
  Utils/Move Master; this module does not become a second market backend.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

NCAAF_RESEARCH_V2_SOURCE_TAG = "ncaaf-research-v2.31.1-deep-context-raw-source-repair-20261010"
NCAAF_RESEARCH_V2_VERSION = "2.31.1"
NCAAF_MINER_LIVE_AUTHORITY_POLICY = "NCAAF_MINER_LIVE_AUTHORITY_V2_20_SOURCE_NEUTRAL_STRONG_VALIDATED_20261007"
NCAAF_MINER_LIVE_MIN_CONFIRMATION_N = 60
NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE = 0.56
DISCOVERY_MAX_SEASON = 2023
CONFIRMATION_SEASONS = (2024, 2025)
PROSPECTIVE_MIN_SEASON = 2026
REPORT_CURRENT_BLOB = "research/ncaaf/v2/current_report.json"
BUNDLE_CURRENT_BLOB = "research/ncaaf/v2/current_bundle.pkl"
REPORT_HISTORY_PREFIX = "research/ncaaf/v2/history"

# Big Al observation ledger. Observed 2026 selections may seed generic research
# vocabulary/hypothesis search, but the picks' outcomes and ratings have zero
# qualification/authority weight. Every candidate must earn its edge entirely
# from the sealed historical discovery/confirmation framework.
BIG_AL_OBSERVATION_LEDGER_BLOB = "research/expert_observations/big_al/observations.jsonl"
BIG_AL_LOCAL_LEDGER_FILENAME = "big_al_observations.jsonl"
BIG_AL_ALLOWED_RATINGS = {2, 4, 6, 8, 10}
BIG_AL_RATING_MIN_FOR_PATTERN_ANALYSIS = 15
BIG_AL_OBSERVATION_ATOM_FAMILIES = {
    "OBS_PRIOR_SCORING_EXTREME",
    "OBS_PRIOR_UPSET_STATE",
    "OBS_ATS_SEASON_STATE",
    "OBS_SPREAD_MARKET_DIRECTION",
    "OBS_TOTAL_MARKET_DIRECTION",
    "OBS_DUAL_PRIOR_RESULT",
    "OBS_PRIOR_POINTS_FOR",
    "OBS_PRIOR_POINTS_ALLOWED",
}
BIG_AL_OBSERVATION_SUPPORT_FAMILIES = {
    "VENUE", "MARKET_ROLE", "MARKET_PRICE", "TOTAL_REGIME",
    "MATCHUP_HISTORY", "RIVALRY", "TRAVEL_CONTEXT", "ROAD_SEQUENCE", "REST",
    "RESULT_QUALITY_REGRESSION", "SCHEDULE_RESUME_QUALITY", "RECENT_VS_SEASON", "MATCHUP_DIFFERENTIAL",
    "PRIOR_RESULT", "PRIOR_MARGIN_MAGNITUDE", "PRIOR_ATS_MAGNITUDE",
    "ATS_FORM", "SU_FORM", "ATS_SEQUENCE", "SU_SEQUENCE",
    "OPP_PRIOR_SU1", "OPP_PRIOR_ATS1", "TEAM_STATE", "OPP_STATE",
    "SEASON_TIMING", "CONFERENCE_PAIR", "ROLE_CHANGE",
    "SEASON_RECORD_STATE", "ROLE_TRANSITION_BOUNCEBACK",
}

# Prediction Tracker external-rating metamodel. PT itself has no blanket model
# weight and never rewrites CORE. PT-derived Miner systems are source-neutral:
# each must independently clear the same frozen historical authority gate as
# other Miner mechanisms, after which all correlated PT variants collapse to
# one EXTERNAL_RATINGS_FAMILY vote. The five fixed weights remain the published
# external benchmark; we do not re-fit them on our NCAAF outcomes.
PT_BASE_URL = "https://www.thepredictiontracker.com"
PT_ARCHIVE_URL = PT_BASE_URL + "/ncaa{season}.csv"
PT_ARCHIVE_PAGE_URL = PT_BASE_URL + "/ncaaarchive.html"
PT_LIVE_CSV_URL = PT_BASE_URL + "/ncaapredictions.csv"
PT_LIVE_PAGE_URL = PT_BASE_URL + "/predncaa.php"
PT_LIVE_PAGE_ALT_URL = PT_BASE_URL + "/predncaa.html"
PT_RAW_PREFIX = "research/ncaaf/external/prediction_tracker/raw"
PT_NORMALIZED_PREFIX = "research/ncaaf/external/prediction_tracker/normalized"
PT_CURRENT_BLOB = "research/ncaaf/external/prediction_tracker/current.csv"  # merged current-season archive + live week
PT_CURRENT_ARCHIVE_BLOB = "research/ncaaf/external/prediction_tracker/current_archive.csv"
PT_CURRENT_LIVE_BLOB = "research/ncaaf/external/prediction_tracker/current_live.csv"
PT_CURRENT_META_BLOB = "research/ncaaf/external/prediction_tracker/current_meta.json"
PT_LIVE_HTML_RAW_BLOB = "research/ncaaf/external/prediction_tracker/raw/predncaa_live_page.txt"
PT_HEADER_MANIFEST_BLOB = "research/ncaaf/external/prediction_tracker/header_manifest.json"
PT_FEEDER_MANIFEST_BLOB = "research/ncaaf/external/prediction_tracker/feeder_manifest.json"
PT_FEEDER_SNAPSHOT_PREFIX = "research/ncaaf/external/prediction_tracker/snapshots"
PT_FEEDER_CURRENT_MAX_AGE_HOURS = float(os.getenv("PT_FEEDER_CURRENT_MAX_AGE_HOURS", "72"))
PT_CURRENT_ARCHIVE_MAX_AGE_HOURS = float(os.getenv("PT_CURRENT_ARCHIVE_MAX_AGE_HOURS", "168"))
PT_ALLOW_WEB_FALLBACK = str(os.getenv("PT_ALLOW_WEB_FALLBACK", "0")).strip().lower() not in {"0","false","no","off"}
PT_PUBLISHED_WEIGHTS = {
    "DOKTER": 0.242406,
    "PI_RATE_BIAS": 0.281205,
    "KEEPER": 0.135398,
    "ESPN_FPI": 0.163639,
    "PIGSKIN_INDEX": 0.114519,
}
# Exact system identity contract. System columns are NEVER selected by
# position and NEVER by substring/fuzzy matching. Prediction Tracker's own CSV
# exports use stable source-native header IDs; those exact IDs are explicitly
# whitelisted here. No live HTML page is required for the normal contract.
PT_SYSTEM_HEADER_ALIASES = {
    "DOKTER": ("Dokter", "Dokter Entropy", "linedokter"),
    "PI_RATE_BIAS": ("Pi-Ratings Bias", "Pi Ratings Bias", "Pi-Rating Bias", "Pi Rate Bias", "linepibias"),
    "KEEPER": ("Keeper", "Keeper Ratings", "linekeep"),
    "ESPN_FPI": ("ESPN FPI", "FPI", "lineespn", "linefpi"),
    "PIGSKIN_INDEX": ("Pigskin Index", "Pigskin", "linepig"),
}

# V2.18 full Prediction Tracker research universe.
# The five published benchmark systems above remain frozen and are still the only
# inputs to META_MARGIN. Separately, every source-native Prediction Tracker
# `line*` predictor is retained for research/Miner use. Market/summary fields are
# excluded so they cannot masquerade as independent models.
PT_INDEX_RESERVED_HEADERS = {"line", "lineopen", "lineavg", "linemedian", "linestd"}
PT_INDEX_CANONICAL_HEADER_ALIASES = {
    # Same provider, renamed by Prediction Tracker across seasons.
    "lineespn": "linefpi",
    "linemass": "linemassey",
    "linebill": "linebillings",
}
PT_INDEX_DISPLAY_NAMES = {
    "linehow": "Howell",
    "linedokter": "Dokter",
    "linebihl": "Bihl System",
    "linemidweek": "Midweek",
    "linekeep": "Keeper",
    "linecong": "Congrove Computer Rankings",
    "linebillings": "Billingsley",
    "lineharville": "David Harville",
    "linebig200": "Big 200",
    "linepve": "PvE Sports Ratings",
    "linepiratings": "Pi-Ratings",
    "linepimean": "Pi-Ratings Mean",
    "linepibias": "Pi-Ratings Bias",
    "linefei": "FEI Projections",
    "linerwp": "Laffaye RWP",
    "linemassey": "Massey Ratings",
    "lineclean": "Cleanup Hitter",
    "linecraig": "Craig",
    "linedonchess": "Donchess",
    "linesag": "Sagarin",
    "linesagpred": "Sagarin Predictor",
    "linesaggm": "Sagarin GM",
    "linesagr": "Sagarin Recent",
    "linenewbury": "Max Newbury",
    "lineversus": "Versus Sports Simulator",
    "linedwig": "Dwiggins",
    "linefpi": "ESPN FPI",
    "linetalis": "Talisman Red",
    "linecfp": "CFP",
    "linepig": "Pigskin Index",
    "linefidler": "Fidler Book",
    "linedial": "Odds Dial",
    "linel2": "Least Squares",
    "linepfz": "PerformanZ Ratings",
    "linemoore": "Sonny Moore",
    "lineelo": "Beck Elo",
    "linelaz": "Laz Index",
    "linekam": "Edward Kambour",
    "linel2hf": "Least Squares w/HFA",
    "linekerns": "Stephen Kerns",
    "lineteamrank": "TeamRankings",
    "linefluker": "Slate Fluker",
    "linelog": "Logistic Regression",
    "linecons": "Massey Consensus",
    "lineloud": "Loudsound",
    "linecurry": "Daniel Curry Index",
    "linedunk": "Dunkel Index",
    "linewayward": "Waywardtrends",
    "linedoi": "Director of Information",
    "lineborn": "Born",
    "lineca": "Computer Adjusted Line",
    # Historical/source-retired IDs preserved by exact source header.
    "linepayne": "Payne Power Ratings",
    "linepaynep": "Payne Power Ratings P",
    "linepaynewl": "Payne Power Ratings W/L",
    "linefox": "Fox",
    "linepugh": "Pugh",
    "linecather": "Cather",
    "lineargh": "ARGH",
}
PT_INDEX_SOURCE_CLUSTERS = {
    "SAGARIN": {"linesag", "linesagpred", "linesaggm", "linesagr"},
    "PI_RATINGS": {"linepiratings", "linepimean", "linepibias"},
    "REGRESSION": {"linel2", "linel2hf", "linelog"},
    "PAYNE": {"linepayne", "linepaynep", "linepaynewl"},
}

# Exact Prediction Tracker indices that are already constituents of the frozen
# five-system META_MARGIN. They remain available for individual behavior study,
# but are never counted as independent evidence from META itself.
PT_META_CONSTITUENT_INDEX_IDS = frozenset({
    "linedokter",
    "linepibias",
    "linekeep",
    "linefpi",   # lineespn canonicalizes to linefpi
    "linepig",
})


def _pt_index_canonical_header(x: Any) -> str:
    k=str(x).strip().lower()
    return PT_INDEX_CANONICAL_HEADER_ALIASES.get(k,k)

def _pt_index_display_name(canon: str) -> str:
    return PT_INDEX_DISPLAY_NAMES.get(str(canon), str(canon))

def _pt_index_source_cluster(canon: str) -> str:
    c=str(canon)
    for cluster,members in PT_INDEX_SOURCE_CLUSTERS.items():
        if c in members:
            return cluster
    # Every other provider is its own source cluster. This prevents Sagarin/Pi
    # variants from receiving multiple consensus votes while retaining all systems.
    return c.upper()

PT_EXTERNAL_CONSENSUS_MIN_CLUSTERS = 8
PT_EXTERNAL_CONSENSUS_CONTRACT = "META_SOURCE_CLUSTER_OUT_CLUSTER_BALANCED_MEDIAN_MIN8"


def _pt_attach_current_external_consensus_fields(out: pd.DataFrame) -> pd.DataFrame:
    """Attach a current-board consensus independent of the frozen five-system META."""
    x=out.copy()
    idx_cols=[str(c) for c in x.columns if str(c).startswith("ptidx__")]
    meta_clusters={_pt_index_source_cluster(k) for k in PT_META_CONSTITUENT_INDEX_IDS}
    eligible=[]
    cluster_to_cols={}
    for c in idx_cols:
        canon=str(c)[len("ptidx__"):]
        cluster=_pt_index_source_cluster(canon)
        if cluster in meta_clusters:
            continue
        eligible.append(c)
        cluster_to_cols.setdefault(cluster,[]).append(c)

    n=len(x)
    raw_index_count=np.zeros(n,dtype=float)
    if eligible:
        _raw=np.column_stack([pd.to_numeric(x[c],errors="coerce").to_numpy(float) for c in eligible])
        raw_index_count=np.isfinite(_raw).sum(axis=1).astype(float)

    cluster_names=sorted(cluster_to_cols)
    cluster_mat=np.full((n,len(cluster_names)),np.nan,dtype=float)
    for j,cluster in enumerate(cluster_names):
        cols=cluster_to_cols[cluster]
        vals=np.column_stack([pd.to_numeric(x[c],errors="coerce").to_numpy(float) for c in cols])
        for i in range(n):
            v=vals[i,np.isfinite(vals[i])]
            if v.size:
                cluster_mat[i,j]=float(np.median(v))

    cluster_count=np.isfinite(cluster_mat).sum(axis=1).astype(float) if cluster_mat.size else np.zeros(n,dtype=float)
    consensus=np.full(n,np.nan,dtype=float)
    dispersion=np.full(n,np.nan,dtype=float)
    iqr=np.full(n,np.nan,dtype=float)
    for i in range(n):
        v=cluster_mat[i,np.isfinite(cluster_mat[i])] if cluster_mat.size else np.asarray([],dtype=float)
        if v.size>=PT_EXTERNAL_CONSENSUS_MIN_CLUSTERS:
            consensus[i]=float(np.median(v))
            if v.size>=2:
                dispersion[i]=float(np.std(v))
                iqr[i]=float(np.percentile(v,75)-np.percentile(v,25))

    x["external_consensus_home_margin"]=consensus
    x["external_consensus_index_count"]=raw_index_count
    x["external_consensus_cluster_count"]=cluster_count
    x["external_consensus_cluster_std"]=dispersion
    x["external_consensus_cluster_iqr"]=iqr
    x["external_consensus_status"]=np.where(np.isfinite(consensus),"READY","INSUFFICIENT_CLUSTERS")
    x["external_consensus_contract"]=PT_EXTERNAL_CONSENSUS_CONTRACT
    return x
PT_IDENTITY_HEADER_ALIASES = {
    "HOME": ("Home", "Home Team", "HomeTeam"),
    "AWAY": ("Road", "Away", "Visitor", "Visiting Team", "Visitor Team", "Away Team"),
}
PT_HISTORY_SEASONS = tuple(range(2022, 2026))
PT_CURRENT_SEASON = 2026
# PT join-quality contract. The primary target is coverage of games actually
# published by Prediction Tracker, not all internal NCAA games (the source does
# not publish every FCS/FBS matchup). Aim for complete source-row matching and
# treat anything below 90% as a coverage defect requiring review.
PT_SOURCE_MATCH_MIN_COVERAGE = float(os.getenv("PT_SOURCE_MATCH_MIN_COVERAGE", "0.90"))
PT_SOURCE_MATCH_GOAL_COVERAGE = float(os.getenv("PT_SOURCE_MATCH_GOAL_COVERAGE", "0.99"))
# Cloud-hosted runtimes can be denied directly by the source site (HTTP 403).
# The relay is read-only and only transports the original public source bytes/text;
# all model identity is still validated against Prediction Tracker headers/values.
PT_RELAY_PREFIX = os.getenv("PT_RELAY_PREFIX", "https://r.jina.ai/https://www.thepredictiontracker.com").rstrip("/")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index, dtype=float)


def _txt(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).str.upper().str.strip() if col in df.columns else pd.Series("", index=df.index, dtype=str)


def _corr(a, b) -> float:
    aa=np.asarray(a,dtype=float); bb=np.asarray(b,dtype=float)
    ok=np.isfinite(aa)&np.isfinite(bb)
    if ok.sum()<20 or np.nanstd(aa[ok])<1e-12 or np.nanstd(bb[ok])<1e-12: return float("nan")
    return float(np.corrcoef(aa[ok],bb[ok])[0,1])


def _mae(y,p) -> float:
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float); ok=np.isfinite(y)&np.isfinite(p)
    return float(np.mean(np.abs(y[ok]-p[ok]))) if ok.any() else float("nan")


def _rmse(y,p) -> float:
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float); ok=np.isfinite(y)&np.isfinite(p)
    return float(np.sqrt(np.mean((y[ok]-p[ok])**2))) if ok.any() else float("nan")


def _american_profit(odds: float) -> float:
    try: o=float(odds)
    except Exception: return float("nan")
    if not np.isfinite(o) or abs(o)<100: return float("nan")
    return o/100.0 if o>0 else 100.0/abs(o)


def _bh_qvalues(pvals: Iterable[float]) -> np.ndarray:
    a=np.asarray(list(pvals),dtype=float); out=np.full(len(a),np.nan,dtype=float)
    ok=np.flatnonzero(np.isfinite(a))
    if not len(ok): return out
    order=ok[np.argsort(a[ok])]; m=len(order); prev=1.0
    for rank in range(m,0,-1):
        i=order[rank-1]; q=min(prev,float(a[i])*m/rank,1.0); out[i]=q; prev=q
    return out


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict): return {str(k):_json_safe(v) for k,v in x.items()}
    if isinstance(x, (list,tuple,set)): return [_json_safe(v) for v in x]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return None if not np.isfinite(float(x)) else float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)): return x.isoformat()
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def _stable_id(prefix: str, parts: Iterable[str]) -> str:
    payload="|".join(str(x) for x in parts)
    return prefix+hashlib.blake2b(payload.encode("utf-8"),digest_size=5).hexdigest().upper()


def _normalize_big_al_observation(raw: dict[str,Any]) -> dict[str,Any] | None:
    """Normalize one immutable expert observation without outcome fields.

    Ratings are descriptive only. Invalid ratings are retained as None rather than
    coerced. Outcome/postgame fields are deliberately discarded so the ledger can
    safely participate in hypothesis generation without contaminating qualification.
    """
    if not isinstance(raw,dict): return None
    sport=str(raw.get("sport") or "").upper().strip()
    market=str(raw.get("market") or "").lower().strip()
    if sport not in {"NCAAF","NFL"} or market not in {"spreads","totals","h2h"}: return None
    rating=raw.get("rating")
    try: rating=int(rating) if rating is not None and str(rating).strip()!="" else None
    except Exception: rating=None
    if rating not in BIG_AL_ALLOWED_RATINGS: rating=None
    line=raw.get("line")
    try: line=float(line) if line is not None and str(line).strip()!="" else None
    except Exception: line=None
    game_date=str(raw.get("game_date") or "").strip()[:10]
    sel_team=str(raw.get("selection_team") or "").strip()
    team_a=str(raw.get("team_a") or "").strip()
    team_b=str(raw.get("team_b") or "").strip()
    direction=str(raw.get("selection_direction") or "").upper().strip()
    oid=str(raw.get("observation_id") or "").strip()
    if not oid:
        oid=_stable_id("BIGAL-OBS-",[sport,game_date,market,sel_team,team_a,team_b,direction,line,rating])
    return {
        "observation_id":oid,"sport":sport,"game_date":game_date,"market":market,
        "selection_team":sel_team or None,"team_a":team_a or None,"team_b":team_b or None,
        "selection_direction":direction or None,"line":line,"rating":rating,
        "release_label":str(raw.get("release_label") or "").strip() or None,
        "source":str(raw.get("source") or "USER_OBSERVED_BIG_AL").strip(),
        "card_complete":bool(raw.get("card_complete",False)),
        "notes":str(raw.get("notes") or "").strip() or None,
        "outcome_used_for_selection":False,"production_authority":0,"rating_authority":0,
    }


def _local_big_al_observations() -> list[dict[str,Any]]:
    path=Path(__file__).resolve().with_name(BIG_AL_LOCAL_LEDGER_FILENAME)
    if not path.exists(): return []
    rows=[]
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if not line: continue
            try: obj=json.loads(line)
            except Exception: continue
            z=_normalize_big_al_observation(obj)
            if z: rows.append(z)
    except Exception:
        return []
    return rows


def _load_big_al_observation_ledger(*, storage_client=None, bucket_name="sharp-models", log_func=print) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    """Merge bundled observations with the persistent GCS research ledger.

    The merge is append/upsert by immutable observation_id. No outcome fields are
    accepted. This is a research ledger only and cannot mutate production state.
    """
    local=_local_big_al_observations(); remote=[]; b=None; blob=None; sync="LOCAL_ONLY"
    if storage_client is not None:
        try:
            b=storage_client.bucket(bucket_name); blob=b.blob(BIG_AL_OBSERVATION_LEDGER_BLOB)
            if blob.exists():
                txt=blob.download_as_text()
                for line in txt.splitlines():
                    try: obj=json.loads(line)
                    except Exception: continue
                    z=_normalize_big_al_observation(obj)
                    if z: remote.append(z)
            sync="REMOTE_READ"
        except Exception as exc:
            sync=f"REMOTE_READ_ERROR:{type(exc).__name__}"
    merged={}
    for row in remote+local:
        oid=str(row.get("observation_id") or "")
        if oid: merged[oid]=row
    rows=sorted(merged.values(),key=lambda z:(str(z.get("game_date") or ""),str(z.get("sport") or ""),str(z.get("observation_id") or "")))
    if blob is not None:
        try:
            payload="\n".join(json.dumps(_json_safe(z),sort_keys=True,separators=(",",":")) for z in rows)+("\n" if rows else "")
            remote_ids={str(x.get("observation_id") or "") for x in remote}
            local_ids={str(x.get("observation_id") or "") for x in local}
            if local_ids-remote_ids or (not blob.exists() and rows):
                blob.upload_from_string(payload,content_type="application/x-ndjson")
                sync="REMOTE_SYNCED"
        except Exception as exc:
            sync=f"REMOTE_SYNC_ERROR:{type(exc).__name__}"
    ratings={}
    for x in rows:
        k=str(x.get("rating")) if x.get("rating") is not None else "UNRATED"
        ratings[k]=ratings.get(k,0)+1
    diag={"status":"PASS","total_observations":len(rows),"ncaaf_observations":sum(x.get("sport")=="NCAAF" for x in rows),
          "nfl_observations":sum(x.get("sport")=="NFL" for x in rows),"rating_counts":ratings,"sync":sync,
          "outcome_fields_allowed":False,"rating_authority":0,"production_authority":0}
    log_func(f"[BIGAL-OBSERVATION-LEDGER] status=PASS total={len(rows)} ncaaf={diag['ncaaf_observations']} nfl={diag['nfl_observations']} ratings={json.dumps(ratings,sort_keys=True)} sync={sync} outcome_authority=0 rating_authority=0")
    return rows,diag


def _big_al_rating_observation_report(observations: list[dict[str,Any]], *, miner_lane_audit: dict[str,Any] | None=None,
                                      context_games: pd.DataFrame | None=None, dashboard_module=None) -> dict[str,Any]:
    """Describe observed ratings and, when possible, fingerprint pregame atoms.

    Fingerprints are descriptive only. We never grade the observed 2026 pick here,
    never use its result, and never use the rating to qualify a historical system.
    """
    ncaaf=[x for x in observations if x.get("sport")=="NCAAF"]
    rated=[x for x in ncaaf if x.get("rating") in BIG_AL_ALLOWED_RATINGS]
    by_rating={}; by_market={}
    for x in ncaaf:
        rk=str(x.get("rating")) if x.get("rating") is not None else "UNRATED"
        by_rating[rk]=by_rating.get(rk,0)+1
        mk=str(x.get("market") or ""); by_market[mk]=by_market.get(mk,0)+1

    obs_rows=[]; atom_counts_by_rating={}; matched_rated=0
    ctx=context_games.copy() if isinstance(context_games,pd.DataFrame) and not context_games.empty else None
    atoms_by_market={}
    if ctx is not None:
        for mk in ("spreads","totals"):
            try:
                aa=_extended_atoms(ctx,dashboard_module=dashboard_module,for_live=False,market=mk)
                atoms_by_market[mk]=[(a["name"],str(a.get("family") or ""),np.asarray(a["mask"],dtype=bool)) for a in aa
                                     if str(a.get("family") or "") in (BIG_AL_OBSERVATION_ATOM_FAMILIES|BIG_AL_OBSERVATION_SUPPORT_FAMILIES)
                                     and str(a.get("family") or "") not in {"OBS_SPREAD_MARKET_DIRECTION","OBS_TOTAL_MARKET_DIRECTION"}]
            except Exception:
                atoms_by_market[mk]=[]
        _dates=pd.to_datetime(ctx.get("Game_Date",ctx.get("Game_Start",pd.Series(pd.NaT,index=ctx.index))),errors="coerce",utc=True).dt.strftime("%Y-%m-%d")
        _teams=ctx.get("Team_Norm",ctx.get("Team",pd.Series("",index=ctx.index))).astype(str).map(_pt_team_key)
        _opps=ctx.get("Opponent_Norm",ctx.get("Opponent",pd.Series("",index=ctx.index))).astype(str).map(_pt_team_key)
        _home=pd.to_numeric(ctx.get("Is_Home",pd.Series(np.nan,index=ctx.index)),errors="coerce")
    else:
        _dates=_teams=_opps=_home=None

    for x in ncaaf:
        rec={k:x.get(k) for k in ("observation_id","game_date","market","selection_team","team_a","team_b","selection_direction","line","rating","release_label")}
        rec["context_match"]="UNAVAILABLE"; rec["active_atoms"]=[]
        if ctx is not None and x.get("game_date"):
            mk=str(x.get("market") or "")
            cand=np.flatnonzero(_dates.eq(str(x.get("game_date"))).to_numpy())
            sel=_pt_team_key(x.get("selection_team")) if x.get("selection_team") else ""
            ta=_pt_team_key(x.get("team_a")) if x.get("team_a") else ""
            tb=_pt_team_key(x.get("team_b")) if x.get("team_b") else ""
            chosen=None; selection_orientation=None
            if mk=="spreads" and sel:
                # Miner rows are canonical-home oriented. Match Big Al road picks
                # to the same physical game instead of silently dropping them;
                # keep orientation explicit so observed spread movement can be
                # converted into the Miner's home-oriented sign convention.
                for i in cand:
                    if _teams.iloc[i]==sel:
                        chosen=int(i); selection_orientation="TEAM"; break
                    if _opps.iloc[i]==sel:
                        chosen=int(i); selection_orientation="OPP"; break
            elif mk=="totals" and ta and tb:
                pair={ta,tb}
                pair_rows=[int(i) for i in cand if {_teams.iloc[i],_opps.iloc[i]}==pair]
                home_rows=[i for i in pair_rows if pd.notna(_home.iloc[i]) and float(_home.iloc[i])==1.0]
                chosen=(home_rows or pair_rows or [None])[0]
            if chosen is not None:
                rec["selection_orientation"]=selection_orientation or "GAME"
                active=[]
                for nm,fam,mask in atoms_by_market.get(mk,[]):
                    if chosen<len(mask) and bool(mask[chosen]): active.append(nm)
                # Market movement in the fingerprint is derived from the OBSERVED
                # release line versus the historical opener, never from a later close.
                try: _oline=float(x.get("line"))
                except Exception: _oline=np.nan
                if np.isfinite(_oline):
                    if mk=="spreads":
                        _open=pd.to_numeric(pd.Series([ctx.iloc[chosen].get("Consensus_Open_Spread",ctx.iloc[chosen].get("Opening_Spread",np.nan))]),errors="coerce").iloc[0]
                        if pd.notna(_open):
                            _oriented_line=(-float(_oline)) if selection_orientation=="OPP" else float(_oline)
                            _mv=_oriented_line-float(_open); rec["observed_release_move_from_open"]=_mv
                            rec["observed_release_line_home_orientation"]=_oriented_line
                            if _mv<=-1: active.append("SPREAD_MOVED_TOWARD_TEAM_1_PLUS")
                            if _mv<=-2: active.append("SPREAD_MOVED_TOWARD_TEAM_2_PLUS")
                            if _mv>=1: active.append("SPREAD_MOVED_AWAY_FROM_TEAM_1_PLUS")
                            if _mv>=2: active.append("SPREAD_MOVED_AWAY_FROM_TEAM_2_PLUS")
                    elif mk=="totals":
                        _open=pd.to_numeric(pd.Series([ctx.iloc[chosen].get("Consensus_Open_Total",ctx.iloc[chosen].get("Opening_Total",np.nan))]),errors="coerce").iloc[0]
                        if pd.notna(_open):
                            _mv=float(_oline)-float(_open); rec["observed_release_move_from_open"]=_mv
                            if _mv<=-1: active.append("TOTAL_MOVED_DOWN_1_PLUS")
                            if _mv<=-2: active.append("TOTAL_MOVED_DOWN_2_PLUS")
                            if _mv<=-3: active.append("TOTAL_MOVED_DOWN_3_PLUS")
                            if _mv>=1: active.append("TOTAL_MOVED_UP_1_PLUS")
                            if _mv>=2: active.append("TOTAL_MOVED_UP_2_PLUS")
                            if _mv>=3: active.append("TOTAL_MOVED_UP_3_PLUS")
                rec["context_match"]="MATCHED_PREGAME_FINGERPRINT"; rec["active_atoms"]=sorted(set(active))
                if x.get("rating") in BIG_AL_ALLOWED_RATINGS:
                    matched_rated+=1; rk=str(x.get("rating")); atom_counts_by_rating.setdefault(rk,{})
                    for nm in active: atom_counts_by_rating[rk][nm]=atom_counts_by_rating[rk].get(nm,0)+1
            else:
                rec["context_match"]="PENDING_OR_UNMATCHED"
        obs_rows.append(rec)

    enough=matched_rated>=BIG_AL_RATING_MIN_FOR_PATTERN_ANALYSIS
    atom_rates={}
    for rk,counts in atom_counts_by_rating.items():
        denom=max(1,sum(1 for r in obs_rows if str(r.get("rating"))==rk and r.get("context_match")=="MATCHED_PREGAME_FINGERPRINT"))
        atom_rates[rk]=[{"atom":nm,"count":cnt,"share":cnt/denom} for nm,cnt in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0]))[:30]]
    return {
        "status":"DESCRIPTIVE_READY" if enough else "COLLECTING_OBSERVATIONS",
        "ncaaf_observations":len(ncaaf),"rated_observations":len(rated),"matched_rated_observations":matched_rated,
        "rating_counts":by_rating,"market_counts":by_market,
        "complete_card_observations":sum(bool(x.get("card_complete")) for x in ncaaf),
        "negative_control_strength":"FULL_CARD_CONTROLS_AVAILABLE" if any(bool(x.get("card_complete")) for x in ncaaf) else "OBSERVED_SELECTIONS_ONLY__UNSELECTED_GAMES_NOT_ASSUMED_REJECTED",
        "rating_pattern_minimum":BIG_AL_RATING_MIN_FOR_PATTERN_ANALYSIS,"rating_pattern_analysis_enabled":bool(enough),
        "rating_atom_frequency":atom_rates,
        "rating_used_for_system_qualification":False,"rating_used_for_authority":False,"outcomes_used_for_hypothesis_generation":False,
        "observed_2026_picks_may_seed_generic_vocabulary":True,
        "hypothesis_lane":miner_lane_audit or {},
        "observations":obs_rows,
        "production_authority":0,
    }


# ---------------------------------------------------------------------------
# External rating metamodel — Prediction Tracker automatic bridge
# ---------------------------------------------------------------------------
def _pt_col_key(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(x).lower()).strip()


def _pt_team_key(x: Any) -> str:
    s=_pt_col_key(x)
    # Conservative school-name normalization. Mascots are intentionally not
    # stripped; fuzzy matching below maps short external school names to the
    # longer canonical team names used by the historical frame.
    repl={
        "st":"state","mich":"michigan","fla":"florida","car":"carolina",
        "ill":"illinois","ind":"indiana","mass":"massachusetts","miss":"mississippi",
        "mo":"missouri","tenn":"tennessee","tex":"texas","va":"virginia",
        "wash":"washington","wis":"wisconsin","conn":"connecticut","colo":"colorado",
        "ark":"arkansas","cal":"california","la":"louisiana","neb":"nebraska",
    }
    toks=[]
    for t in s.split(): toks.append(repl.get(t,t))
    s=" ".join(toks)
    # Common tracker spellings.
    s=s.replace("n c state","north carolina state").replace("nc state","north carolina state")
    s=s.replace("s c state","south carolina state").replace("sc state","south carolina state")
    s=s.replace("e michigan","eastern michigan").replace("w michigan","western michigan")
    s=s.replace("c michigan","central michigan").replace("n illinois","northern illinois")
    s=s.replace("s mississippi","southern mississippi").replace("southern miss","southern mississippi")
    return re.sub(r"\s+"," ",s).strip()


def _pt_find_col(cols: Iterable[str], aliases: Iterable[str]) -> str | None:
    """Legacy/general resolver for non-system convenience fields only.

    System identity must use _pt_resolve_strict_col; this helper may use a
    contained match for generic fields such as opening line/date.
    """
    keyed={str(c):_pt_col_key(c) for c in cols}
    aa=[_pt_col_key(x) for x in aliases]
    for c,k in keyed.items():
        if k in aa: return c
    for a in aa:
        if len(a)<3: continue
        hits=[c for c,k in keyed.items() if a in k]
        if len(hits)==1: return hits[0]
    return None


def _pt_resolve_strict_col(cols: Iterable[str], aliases: Iterable[str]) -> tuple[str | None,list[str]]:
    """Resolve a column only by an exact normalized header identity.

    Returns (unique_match, all_matches).  More than one match is ambiguous and
    therefore fails closed.  No position, substring, or fuzzy fallback exists.
    """
    alias_keys={_pt_col_key(x) for x in aliases}
    hits=[str(c) for c in cols if _pt_col_key(c) in alias_keys]
    uniq=[]
    for c in hits:
        if c not in uniq: uniq.append(c)
    return (uniq[0] if len(uniq)==1 else None),uniq


def _pt_read_csv_frame(raw: bytes) -> tuple[pd.DataFrame | None,list[str]]:
    errs=[]
    for kwargs in (
        {"engine":"python","on_bad_lines":"skip"},
        {"engine":"python","on_bad_lines":"skip","encoding":"latin1"},
    ):
        try:
            df=pd.read_csv(io.BytesIO(raw),**kwargs)
            if not df.empty: return df.dropna(axis=1,how="all").copy(),errs
        except Exception as e:
            errs.append(f"{type(e).__name__}:{e}")
    return None,errs


def _pt_extract_live_html_table(raw: bytes) -> tuple[pd.DataFrame | None,dict[str,Any]]:
    # Direct source HTML uses read_html. Reader/proxy fallback is markdown, so
    # detect and parse that named table without weakening the identity contract.
    try:
        _txt0=raw.decode("utf-8",errors="ignore") if raw is not None else ""
    except Exception:
        _txt0=""
    if "|" in _txt0 and "ESPN FPI" in _txt0 and "Pi-Ratings Bias" in _txt0 and "Pigskin Index" in _txt0:
        md,mdiag=_pt_extract_live_markdown_table(raw)
        if md is not None:
            return md,mdiag
    try:
        tables=pd.read_html(io.BytesIO(raw))
    except Exception as exc:
        md,mdiag=_pt_extract_live_markdown_table(raw)
        if md is not None:
            return md,mdiag
        return None,{"status":"HTML_PARSE_FAIL","error":f"{type(exc).__name__}:{exc}","markdown":mdiag}
    best=None; best_score=-1
    required_names=[a[0] for a in PT_SYSTEM_HEADER_ALIASES.values()]
    for t in tables:
        x=t.copy()
        if isinstance(x.columns,pd.MultiIndex):
            x.columns=[" ".join(str(v) for v in tup if str(v).lower() not in {"nan","none"} and not str(v).startswith("Unnamed")).strip() for tup in x.columns]
        else:
            x.columns=[str(c) for c in x.columns]
        cols=list(map(str,x.columns))
        home,_=_pt_resolve_strict_col(cols,PT_IDENTITY_HEADER_ALIASES["HOME"])
        away,_=_pt_resolve_strict_col(cols,PT_IDENTITY_HEADER_ALIASES["AWAY"])
        if home is None or away is None: continue
        score=0
        for canon,aliases in PT_SYSTEM_HEADER_ALIASES.items():
            c,_=_pt_resolve_strict_col(cols,aliases)
            score+=int(c is not None)
        if score>best_score:
            best=x; best_score=score
    if best is None or best_score<3:
        return None,{"status":"HTML_SYSTEM_TABLE_MISSING","table_count":len(tables),"best_system_count":best_score}
    return best,{"status":"PASS","table_count":len(tables),"matched_system_headers":best_score,"columns":list(map(str,best.columns))}


def _pt_infer_verified_header_map(csv_raw: bytes, html_raw: bytes) -> tuple[dict[str,str],dict[str,Any]]:
    """Infer cryptic CSV->system names by comparing actual prediction vectors.

    A mapping is accepted only when a single CSV column reproduces the named
    live-HTML system values across at least 8 games with >=98% exact-to-0.01
    agreement.  This turns column order into validation evidence, not identity.
    """
    cdf,errs=_pt_read_csv_frame(csv_raw)
    hdf,hdiag=_pt_extract_live_html_table(html_raw)
    if cdf is None or hdf is None:
        return {},{"status":"UNAVAILABLE","csv_errors":errs,"html":hdiag}
    ccols=list(map(str,cdf.columns)); hcols=list(map(str,hdf.columns))
    ch,_=_pt_resolve_strict_col(ccols,PT_IDENTITY_HEADER_ALIASES["HOME"]); ca,_=_pt_resolve_strict_col(ccols,PT_IDENTITY_HEADER_ALIASES["AWAY"])
    hh,_=_pt_resolve_strict_col(hcols,PT_IDENTITY_HEADER_ALIASES["HOME"]); ha,_=_pt_resolve_strict_col(hcols,PT_IDENTITY_HEADER_ALIASES["AWAY"])
    if None in (ch,ca,hh,ha):
        return {},{"status":"IDENTITY_COLUMNS_MISSING","csv_columns":ccols,"html_columns":hcols}
    c=cdf.copy(); h=hdf.copy()
    c["__pair"]=[_pt_team_key(a)+"|"+_pt_team_key(b) for a,b in zip(c[ch],c[ca])]
    h["__pair"]=[_pt_team_key(a)+"|"+_pt_team_key(b) for a,b in zip(h[hh],h[ha])]
    c=c.loc[c["__pair"].ne("") & ~c["__pair"].duplicated(keep=False)].set_index("__pair",drop=False)
    h=h.loc[h["__pair"].ne("") & ~h["__pair"].duplicated(keep=False)].set_index("__pair",drop=False)
    common=sorted(set(c.index)&set(h.index))
    if len(common)<8:
        return {},{"status":"TOO_FEW_OVERLAP_GAMES","overlap_games":len(common)}
    excluded={ch,ca}
    mapping={}; evidence={}; ambiguous={}
    for canon,aliases in PT_SYSTEM_HEADER_ALIASES.items():
        hc,hhits=_pt_resolve_strict_col(hcols,aliases)
        if hc is None:
            evidence[canon]={"status":"HTML_HEADER_MISSING_OR_AMBIGUOUS","hits":hhits}; continue
        y=_pt_numeric(h.loc[common,hc]).to_numpy(float)
        candidates=[]
        for col in ccols:
            if col in excluded: continue
            x=_pt_numeric(c.loc[common,col]).to_numpy(float)
            ok=np.isfinite(x)&np.isfinite(y)
            n=int(ok.sum())
            if n<8: continue
            diff=np.abs(x[ok]-y[ok])
            rate=float(np.mean(diff<=0.011)); mae=float(np.mean(diff)); mx=float(np.max(diff))
            if rate>=0.98 and mae<=0.011:
                candidates.append((col,n,rate,mae,mx))
        candidates=sorted(candidates,key=lambda z:(-z[1],z[3],z[4],z[0]))
        if len(candidates)==1:
            mapping[canon]=candidates[0][0]
            evidence[canon]={"status":"VERIFIED_BY_VALUES","csv_column":candidates[0][0],"n":candidates[0][1],"match_rate":candidates[0][2],"mae":candidates[0][3],"max_abs":candidates[0][4],"html_column":hc}
        elif len(candidates)>1:
            ambiguous[canon]=[z[0] for z in candidates]
            evidence[canon]={"status":"AMBIGUOUS_VALUE_MATCH","candidates":ambiguous[canon],"html_column":hc}
        else:
            # Exact self-identifying CSV headers are still safe even when the
            # HTML table contains missing values that prevent vector validation.
            dc,dhits=_pt_resolve_strict_col(ccols,aliases)
            if dc is not None:
                mapping[canon]=dc; evidence[canon]={"status":"VERIFIED_EXACT_HEADER","csv_column":dc,"html_column":hc}
            else:
                evidence[canon]={"status":"NO_VERIFIED_CSV_MATCH","html_column":hc,"exact_hits":dhits}
    status="PASS" if len(mapping)==len(PT_PUBLISHED_WEIGHTS) and not ambiguous else "PARTIAL"
    return mapping,{"status":status,"overlap_games":len(common),"mapping":mapping,"evidence":evidence,"ambiguous":ambiguous,"csv_headers":ccols,"html_headers":hcols}


def _pt_load_header_manifest(storage_client, bucket_name: str) -> dict[str,str]:
    try:
        raw=_pt_blob_bytes(storage_client,bucket_name,PT_HEADER_MANIFEST_BLOB)
        if not raw: return {}
        obj=json.loads(raw.decode("utf-8")); mp=obj.get("mapping",{}) if isinstance(obj,dict) else {}
        return {str(k):str(v) for k,v in mp.items() if k in PT_PUBLISHED_WEIGHTS and str(v).strip()}
    except Exception:
        return {}


def _pt_save_header_manifest(storage_client, bucket_name: str, mapping: dict[str,str], evidence: dict[str,Any]) -> None:
    try:
        payload={"updated_utc":_now(),"source":"LIVE_HTML_CSV_VALUE_VALIDATION","mapping":mapping,"evidence":evidence}
        storage_client.bucket(bucket_name).blob(PT_HEADER_MANIFEST_BLOB).upload_from_string(json.dumps(payload,sort_keys=True,default=str).encode(),content_type="application/json")
    except Exception:
        pass

def _pt_numeric(s: pd.Series) -> pd.Series:
    if s is None: return pd.Series(dtype=float)
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9+\-.]","",regex=True).replace({"":"nan",".":"nan","-":"nan","+":"nan"}),errors="coerce")


def _pt_http_fetch(url: str, timeout: int=25, attempts: int=3, referer: str | None=None) -> bytes:
    """Browser-like same-origin fetch for Prediction Tracker.

    The site can reject direct Cloud Run CSV hotlinks with HTTP 403.  We therefore
    establish a same-origin session on the parent HTML page first, retain any
    cookies, and then request the CSV with a real browser Referer.
    """
    import time
    import requests
    last=None
    parent=referer or (PT_LIVE_PAGE_URL if "ncaapredictions" in url else PT_ARCHIVE_PAGE_URL)
    ua=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36")
    base_headers={
        "User-Agent":ua,"Accept-Language":"en-US,en;q=0.9","Cache-Control":"no-cache",
        "Pragma":"no-cache","Connection":"keep-alive",
    }
    for i in range(max(1,int(attempts))):
        try:
            sess=requests.Session()
            # Same-origin warm-up is intentionally best effort.  The CSV request
            # still runs if the parent page is temporarily unavailable.
            try:
                sess.get(parent,headers={**base_headers,"Accept":"text/html,application/xhtml+xml,*/*;q=0.8","Referer":PT_BASE_URL+"/"},timeout=timeout)
            except Exception:
                pass
            rr=sess.get(url,headers={**base_headers,"Accept":"text/csv,text/plain,*/*","Referer":parent,"Sec-Fetch-Site":"same-origin","Sec-Fetch-Mode":"navigate"},timeout=timeout,allow_redirects=True)
            rr.raise_for_status(); data=rr.content
            if len(data)<100: raise RuntimeError(f"short response bytes={len(data)}")
            if _pt_is_challenge_payload(data):
                raise RuntimeError("anti-bot challenge page returned instead of Prediction Tracker data")
            return data
        except Exception as exc:
            last=exc
            if i+1<int(attempts): time.sleep(1.25*(i+1))
    raise RuntimeError(f"Prediction Tracker fetch failed url={url}: {type(last).__name__}:{last}")



def _pt_relay_urls(url: str) -> list[str]:
    """Read-only relay variants; HTTPS is preferred, HTTP-origin is fallback."""
    try:
        from urllib.parse import urlparse
        u=urlparse(str(url)); path=u.path or "/"
        if u.query: path += "?" + u.query
        https_url=PT_RELAY_PREFIX + path
        http_url="https://r.jina.ai/http://www.thepredictiontracker.com" + path
        return list(dict.fromkeys([https_url,http_url]))
    except Exception:
        leaf=str(url).rsplit("/",1)[-1]
        return [PT_RELAY_PREFIX+"/"+leaf,"https://r.jina.ai/http://www.thepredictiontracker.com/"+leaf]

def _pt_relay_url(url: str) -> str:
    """Backward-compatible primary relay URL used by diagnostics."""
    return _pt_relay_urls(url)[0]


def _pt_extract_csv_payload(raw: bytes) -> bytes:
    """Recover a CSV payload from a relay response without trusting column position.

    Reader/proxy services may prepend metadata or markdown fences.  We retain the
    source header and all following rows starting at the first exact Home/Road CSV
    header.  If the body is already CSV it is returned unchanged.
    """
    if raw is None:
        return b""
    text=raw.decode("utf-8",errors="replace").replace("\r\n","\n")
    lines=text.split("\n")
    for i,line in enumerate(lines):
        z=line.strip().strip("`").lstrip("\ufeff")
        cells=[c.strip().strip('"').lower() for c in z.split(",")]
        if len(cells)>=3 and cells[0] in {"home","home team"} and cells[1] in {"road","away","visitor","away team"}:
            payload="\n".join(lines[i:]).strip()
            payload=re.sub(r"\n```\s*$","",payload).strip()
            return payload.encode("utf-8")
    return raw


def _pt_is_challenge_payload(raw: bytes) -> bool:
    """Reject anti-bot/interstitial pages before they can be parsed or cached as PT data."""
    if not raw:
        return True
    head=raw[:12000].decode("utf-8",errors="ignore").lower()
    markers=(
        "title: just a moment", "<title>just a moment", "cf-chl-", "challenge-platform",
        "cloudflare ray id", "enable javascript and cookies", "checking your browser",
    )
    return any(m in head for m in markers)


def _pt_relay_fetch(url: str, timeout: int=35) -> bytes:
    """Read-only relay fallback; try HTTPS-origin then HTTP-origin variants."""
    import requests
    errs=[]
    for ru in _pt_relay_urls(url):
        try:
            rr=requests.get(ru,headers={"User-Agent":"Mozilla/5.0","Accept":"text/plain,text/markdown,*/*"},timeout=timeout,allow_redirects=True)
            rr.raise_for_status()
            if len(rr.content)<100:
                raise RuntimeError(f"short relay response bytes={len(rr.content)}")
            if _pt_is_challenge_payload(rr.content):
                raise RuntimeError("anti-bot challenge page")
            return rr.content
        except Exception as exc:
            errs.append(f"{ru}=>{type(exc).__name__}:{exc}")
    raise RuntimeError("all relay variants failed: "+" | ".join(errs))


def _pt_fetch_csv_resilient(url: str, *, referer: str | None=None, timeout: int=25) -> tuple[bytes,str]:
    """Fetch source CSV directly first, then through the read-only relay."""
    direct_exc=None
    try:
        return _pt_http_fetch(url,timeout=timeout,referer=referer),"WEB_SESSION"
    except Exception as exc:
        direct_exc=exc
    try:
        raw=_pt_extract_csv_payload(_pt_relay_fetch(url,timeout=max(timeout,35)))
        df,errs=_pt_read_csv_frame(raw)
        if df is None or df.empty:
            raise RuntimeError(f"relay payload not parseable as CSV errors={errs}")
        return raw,"WEB_RELAY"
    except Exception as relay_exc:
        raise RuntimeError(
            f"Prediction Tracker direct+relay failed url={url}; direct={type(direct_exc).__name__}:{direct_exc}; "
            f"relay={type(relay_exc).__name__}:{relay_exc}"
        )


def _pt_clean_md_cell(cell: str) -> str:
    x=str(cell or "").strip()
    x=re.sub(r"!\[[^\]]*\]\([^)]*\)","",x)
    x=re.sub(r"\[([^\]]+)\]\([^)]*\)",r"\1",x)
    x=re.sub(r"<[^>]+>"," ",x)
    return re.sub(r"\s+"," ",x).strip()


def _pt_extract_live_markdown_table(raw: bytes) -> tuple[pd.DataFrame | None,dict[str,Any]]:
    """Extract Prediction Tracker's named individual-model matrix from reader markdown."""
    try:
        text=raw.decode("utf-8",errors="replace").replace("\r\n","\n")
    except Exception as exc:
        return None,{"status":"MARKDOWN_DECODE_FAIL","error":f"{type(exc).__name__}:{exc}"}
    lines=text.split("\n")
    hidx=None; header_line=""
    # The detailed matrix is the only table containing all five canonical systems.
    required=("ESPN FPI","Pi-Ratings Bias","Dokter","Keeper","Pigskin Index")
    for i,line in enumerate(lines):
        window=" ".join(lines[max(0,i-1):min(len(lines),i+2)])
        if all(x.lower() in window.lower() for x in required) and "home" in window.lower() and "road" in window.lower():
            hidx=max(0,i-1) if "home" in lines[max(0,i-1)].lower() and "road" in lines[max(0,i-1)].lower() else i
            header_line=" ".join(lines[hidx:i+1])
            break
    if hidx is None:
        return None,{"status":"MARKDOWN_SYSTEM_TABLE_MISSING"}
    # Find separator and parse the header.  Joining the wrapped header keeps the
    # human system labels while remaining independent of the cryptic CSV order.
    header=[_pt_clean_md_cell(x) for x in header_line.strip().strip("|").split("|")]
    header=[x for x in header if x!=""]
    if len(header)<9:
        return None,{"status":"MARKDOWN_HEADER_TOO_SHORT","header":header}
    # Make duplicate Home/Road labels unique; system identity names stay exact.
    seen={}; cols=[]
    for c in header:
        k=c
        n=seen.get(k,0); seen[k]=n+1
        cols.append(k if n==0 else f"{k}__dup{n}")
    sep=None
    for j in range(i+1,min(len(lines),i+5)):
        if "---" in lines[j] and "|" in lines[j]: sep=j; break
    if sep is None:
        return None,{"status":"MARKDOWN_SEPARATOR_MISSING","header":cols}
    rows=[]
    for line in lines[sep+1:]:
        st=line.strip()
        if not st or st.startswith("* * *") or st.startswith("### ") or st.startswith("## "):
            if rows: break
            continue
        if "|" not in st: continue
        vals=[_pt_clean_md_cell(x) for x in st.strip().strip("|").split("|")]
        if len(vals)<3: continue
        if vals[0].lower() in {"home","---"}: continue
        if len(vals)<len(cols): vals += [""]*(len(cols)-len(vals))
        elif len(vals)>len(cols): vals=vals[:len(cols)]
        rows.append(vals)
    if not rows:
        return None,{"status":"MARKDOWN_NO_DATA_ROWS","header":cols}
    return pd.DataFrame(rows,columns=cols),{"status":"PASS","parser":"RELAY_MARKDOWN","rows":len(rows),"columns":cols}

def _pt_fetch_live_html(timeout: int=25) -> bytes:
    """Fetch named live table from .php or static .html, direct then relay."""
    import requests
    ua=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36")
    errs=[]
    for page in (PT_LIVE_PAGE_URL,PT_LIVE_PAGE_ALT_URL):
        try:
            rr=requests.get(page,headers={"User-Agent":ua,"Accept":"text/html,application/xhtml+xml,*/*;q=0.8","Accept-Language":"en-US,en;q=0.9","Referer":PT_BASE_URL+"/"},timeout=timeout)
            rr.raise_for_status()
            if len(rr.content)<500: raise RuntimeError(f"short live HTML bytes={len(rr.content)}")
            if _pt_is_challenge_payload(rr.content): raise RuntimeError("anti-bot challenge page")
            return rr.content
        except Exception as exc:
            errs.append(f"direct:{page}=>{type(exc).__name__}:{exc}")
        try:
            raw=_pt_relay_fetch(page,timeout=max(timeout,35))
            if _pt_is_challenge_payload(raw): raise RuntimeError("anti-bot challenge page")
            return raw
        except Exception as exc:
            errs.append(f"relay:{page}=>{type(exc).__name__}:{exc}")
    raise RuntimeError("Prediction Tracker live page variants failed: "+" | ".join(errs))

def _pt_parse_live_html(raw: bytes, season: int, *, log_func=print) -> tuple[pd.DataFrame,dict[str,Any]]:
    """Extract the named individual-system table from predncaa.php.

    The HTML table is preferred for identity because its headers contain the
    human system names.  The five-system metamodel is computed only when all
    five exact named headers resolve uniquely.
    """
    target,tdiag=_pt_extract_live_html_table(raw)
    if target is None:
        return pd.DataFrame(),{"status":tdiag.get("status","HTML_SYSTEM_TABLE_MISSING"),"season":int(season),**tdiag}
    frame,diag=_pt_parse_csv(target.to_csv(index=False).encode(),season,verified_header_map={},source_context="LIVE_HTML",log_func=log_func)
    diag["parser"]="LIVE_HTML_NAMED_TABLE"
    diag["html_table_diagnostic"]=tdiag
    return frame,diag


def _pt_parse_csv(raw: bytes, season: int, *, verified_header_map: dict[str,str] | None=None, source_context: str="CSV", log_func=print) -> tuple[pd.DataFrame, dict[str,Any]]:
    """Parse Prediction Tracker without positional system assumptions.

    Home/Road and all five benchmark systems are resolved by exact header name.
    Prediction Tracker source-native CSV IDs are accepted only through the
    explicit exact-header whitelist above. No positional/fuzzy inference is used.
    """
    df,errs=_pt_read_csv_frame(raw)
    if df is None or df.empty:
        return pd.DataFrame(),{"status":"PARSE_FAIL","season":int(season),"errors":errs}
    cols=list(map(str,df.columns))
    log_func(f"[NCAAF-PT-HEADER] season={int(season)} source={source_context} columns={json.dumps(cols,separators=(',',':'))}")

    home,home_hits=_pt_resolve_strict_col(cols,PT_IDENTITY_HEADER_ALIASES["HOME"])
    away,away_hits=_pt_resolve_strict_col(cols,PT_IDENTITY_HEADER_ALIASES["AWAY"])
    if home is None or away is None:
        diag={"status":"IDENTITY_COLUMNS_MISSING","season":int(season),"columns":cols,"home_hits":home_hits,"away_hits":away_hits}
        log_func(f"[NCAAF-PT-COLUMN-MAP] season={int(season)} source={source_context} status=FAIL identity=HOME:{home_hits},AWAY:{away_hits} authority=0")
        return pd.DataFrame(),diag

    verified_header_map=verified_header_map or {}
    syscols={}; ambiguity={}; resolution={}
    for canon,aliases in PT_SYSTEM_HEADER_ALIASES.items():
        exact,exact_hits=_pt_resolve_strict_col(cols,aliases)
        manifest_col=str(verified_header_map.get(canon,"") or "")
        manifest_match=manifest_col if manifest_col in cols else None
        candidates=[]
        if exact: candidates.append((exact,"EXACT_HEADER"))
        if manifest_match and manifest_match not in [x[0] for x in candidates]: candidates.append((manifest_match,"VERIFIED_MANIFEST"))
        if len(candidates)==1:
            syscols[canon]=candidates[0][0]; resolution[canon]=candidates[0][1]
        elif len(candidates)>1 and len({x[0] for x in candidates})==1:
            syscols[canon]=candidates[0][0]; resolution[canon]="EXACT_AND_MANIFEST"
        elif len(candidates)>1:
            syscols[canon]=None; ambiguity[canon]=[x[0] for x in candidates]; resolution[canon]="AMBIGUOUS"
        else:
            syscols[canon]=None; resolution[canon]="MISSING"
    reverse={}
    for canon,col in syscols.items():
        if col: reverse.setdefault(col,[]).append(canon)
    duplicate_assign={c:v for c,v in reverse.items() if len(v)>1}
    if duplicate_assign:
        for c,canons in duplicate_assign.items():
            for canon in canons:
                syscols[canon]=None; resolution[canon]="DUPLICATE_SOURCE_COLUMN"; ambiguity[canon]=[c]

    missing=[k for k,v in syscols.items() if v is None]
    map_status="FULL_FIVE_VERIFIED" if not missing and not ambiguity else "INCOMPLETE_FAIL_CLOSED"
    log_func(
        f"[NCAAF-PT-COLUMN-MAP] season={int(season)} source={source_context} status={map_status} "
        f"DOKTER={syscols.get('DOKTER')} PI_RATE_BIAS={syscols.get('PI_RATE_BIAS')} KEEPER={syscols.get('KEEPER')} "
        f"ESPN_FPI={syscols.get('ESPN_FPI')} PIGSKIN_INDEX={syscols.get('PIGSKIN_INDEX')} "
        f"missing={missing} ambiguous={ambiguity} resolution={resolution} authority=0"
    )

    line_col=_pt_find_col(cols,["line","updated line","current line","spread"])
    open_col=_pt_find_col(cols,["lineopen","line open","opening line","open line","opening"])
    avg_col=_pt_find_col(cols,["lineavg","prediction avg","prediction average","system average","average prediction"])
    med_col=_pt_find_col(cols,["linemedian","prediction median","median prediction"])
    std_col=_pt_find_col(cols,["linestd","prediction std","prediction standard deviation","system std"])
    date_col=_pt_find_col(cols,["date","game date","gamedate"])
    out=pd.DataFrame(index=df.index)
    out["season"]=int(season)
    out["home_raw"]=df[home].astype(str).str.strip()
    out["away_raw"]=df[away].astype(str).str.strip()
    out["home_key"]=out["home_raw"].map(_pt_team_key)
    out["away_key"]=out["away_raw"].map(_pt_team_key)
    out["game_date"]=pd.to_datetime(df[date_col],errors="coerce").dt.strftime("%Y-%m-%d") if date_col else ""
    out["tracker_line_home"]=_pt_numeric(df[line_col]) if line_col else np.nan
    out["tracker_open_home"]=_pt_numeric(df[open_col]) if open_col else np.nan
    out["prediction_avg_home"]=_pt_numeric(df[avg_col]) if avg_col else np.nan
    out["prediction_median_home"]=_pt_numeric(df[med_col]) if med_col else np.nan
    out["prediction_std_home"]=_pt_numeric(df[std_col]) if std_col else np.nan
    for k,c in syscols.items(): out[k]=_pt_numeric(df[c]) if c else np.nan

    # Preserve the full source-native Prediction Tracker predictor universe.
    # Only exact `line*` fields are eligible, with market/aggregate line fields
    # explicitly excluded. No positional or fuzzy system identity is introduced.
    _idx_sources={}; _idx_alias_conflicts={}
    for _raw in cols:
        _rk=str(_raw).strip().lower()
        if not _rk.startswith("line") or _rk in PT_INDEX_RESERVED_HEADERS:
            continue
        _canon=_pt_index_canonical_header(_rk)
        _oc=f"ptidx__{_canon}"
        _vv=_pt_numeric(df[_raw])
        _idx_sources.setdefault(_canon,[]).append(str(_raw))
        if _oc not in out.columns:
            out[_oc]=_vv
        else:
            _prev=pd.to_numeric(out[_oc],errors="coerce")
            _new=pd.to_numeric(_vv,errors="coerce")
            _ov=_prev.notna()&_new.notna()
            if _ov.any():
                _bad=(_prev[_ov]-_new[_ov]).abs().gt(.011)
                if bool(_bad.any()):
                    _idx_alias_conflicts[_canon]=int(_bad.sum())
            out[_oc]=_prev.where(_prev.notna(),_new)

    out=_pt_attach_current_external_consensus_fields(out)
    _ext_ready=int(pd.to_numeric(out.get("external_consensus_home_margin"),errors="coerce").notna().sum())
    _ext_clusters=pd.to_numeric(out.get("external_consensus_cluster_count"),errors="coerce")
    log_func(
        f"[NCAAF-PT-EXTERNAL-CONSENSUS] season={int(season)} source={source_context} "
        f"ready_rows={_ext_ready}/{len(out)} min_clusters={PT_EXTERNAL_CONSENSUS_MIN_CLUSTERS} "
        f"median_cluster_count={float(_ext_clusters.median()) if len(_ext_clusters) else 0.0:.1f} "
        f"contract={PT_EXTERNAL_CONSENSUS_CONTRACT} production_authority=0"
    )

    vals=np.column_stack([pd.to_numeric(out[k],errors="coerce").to_numpy(float) for k in PT_PUBLISHED_WEIGHTS])
    w=np.asarray([PT_PUBLISHED_WEIGHTS[k] for k in PT_PUBLISHED_WEIGHTS],dtype=float)
    full=np.isfinite(vals).all(axis=1) if map_status=="FULL_FIVE_VERIFIED" else np.zeros(len(out),dtype=bool)
    meta=np.full(len(out),np.nan,dtype=float)
    if full.any(): meta[full]=vals[full]@w
    out["meta_margin_home"]=meta
    out["meta_system_count"]=np.isfinite(vals).sum(axis=1).astype(int)
    out["pt_header_contract"]=map_status
    bad=out["home_key"].isin({"","home","home team"})|out["away_key"].isin({"","road","away","visitor","visitor team"})
    out=out.loc[~bad].reset_index(drop=True)
    out["source_row"]=np.arange(len(out),dtype=int)
    _component_counts=pd.to_numeric(out["meta_system_count"],errors="coerce").fillna(0).astype(int)
    _idx_cols=[c for c in out.columns if str(c).startswith("ptidx__")]
    _idx_populated=[c for c in _idx_cols if pd.to_numeric(out[c],errors="coerce").notna().any()]
    _idx_any=(out[_idx_cols].apply(pd.to_numeric,errors="coerce").notna().any(axis=1) if _idx_cols else pd.Series(False,index=out.index))
    log_func(
        f"[NCAAF-PT-INDEX-UNIVERSE] season={int(season)} source={source_context} "
        f"canonical_indices={len(_idx_cols)} populated_indices={len(_idx_populated)} "
        f"rows_with_any_index={int(_idx_any.sum())}/{len(out)} alias_conflicts={_idx_alias_conflicts} authority=0"
    )
    return out,{
        "status":"PASS","season":int(season),"rows":int(len(out)),
        "any_component_rows":int((_component_counts>0).sum()),
        "partial_component_rows":int(((_component_counts>0)&(_component_counts<5)).sum()),
        "full_five_rows":int(np.isfinite(out["meta_margin_home"]).sum()),
        "metamodel_status":map_status,"system_columns":syscols,"system_column_resolution":resolution,"missing_systems":missing,"ambiguous_systems":ambiguity,
        "home_column":home,"away_column":away,"date_column":date_col,"line_column":line_col,"open_line_column":open_col,
        "prediction_avg_column":avg_col,"prediction_median_column":med_col,"prediction_std_column":std_col,
        "weights":dict(PT_PUBLISHED_WEIGHTS),"weight_sum":float(sum(PT_PUBLISHED_WEIGHTS.values())),
        "index_columns":{k:v for k,v in sorted(_idx_sources.items())},
        "index_display_names":{k:_pt_index_display_name(k) for k in sorted(_idx_sources)},
        "index_alias_conflicts":_idx_alias_conflicts,
        "external_consensus_contract":PT_EXTERNAL_CONSENSUS_CONTRACT,
        "external_consensus_ready_rows":int(pd.to_numeric(out.get("external_consensus_home_margin"),errors="coerce").notna().sum()),
        "external_consensus_min_clusters":PT_EXTERNAL_CONSENSUS_MIN_CLUSTERS,
        "raw_headers":cols,"source_context":source_context,
    }

def _pt_blob_bytes(storage_client, bucket_name: str, path: str) -> bytes | None:
    try:
        b=storage_client.bucket(bucket_name).blob(path)
        if not b.exists(): return None
        return b.download_as_bytes()
    except Exception: return None


def _pt_gcs_raw(storage_client, bucket_name: str, path: str, *, max_age_hours: float | None=None) -> tuple[bytes | None,dict[str,Any]]:
    """Read a validated raw Prediction Tracker artifact from GCS.

    V2.17 makes validated GCS uploads the model-side contract. Heavy/Weekly do not
    require direct Prediction Tracker network access. Challenge/interstitial payloads
    are rejected even if a bad blob somehow exists. Current sources are allowed a
    practical manual-upload freshness window because Prediction Tracker coverage is
    intentionally sparse and may be published incrementally.
    """
    meta={"status":"MISSING","path":path,"age_hours":None,"updated_utc":None}
    try:
        blob=storage_client.bucket(bucket_name).blob(path)
        if not blob.exists():
            return None,meta
        try: blob.reload()
        except Exception: pass
        updated=getattr(blob,"updated",None)
        age_hours=None
        if updated is not None:
            try:
                import datetime as _dt
                now=_dt.datetime.now(_dt.timezone.utc)
                if getattr(updated,"tzinfo",None) is None:
                    updated=updated.replace(tzinfo=_dt.timezone.utc)
                age_hours=max(0.0,(now-updated).total_seconds()/3600.0)
            except Exception:
                age_hours=None
        raw=blob.download_as_bytes()
        if _pt_is_challenge_payload(raw):
            return None,{**meta,"status":"REJECTED_CHALLENGE","age_hours":age_hours,"updated_utc":str(updated) if updated is not None else None}
        if max_age_hours is not None and age_hours is not None and age_hours>float(max_age_hours):
            return None,{**meta,"status":"STALE","age_hours":age_hours,"updated_utc":str(updated) if updated is not None else None}
        return raw,{"status":"PASS","path":path,"age_hours":age_hours,"updated_utc":str(updated) if updated is not None else None,"bytes":len(raw)}
    except Exception as exc:
        return None,{**meta,"status":"ERROR","error":f"{type(exc).__name__}:{exc}"}


def _pt_load_feeder_manifest(storage_client,bucket_name: str) -> dict[str,Any]:
    try:
        raw=_pt_blob_bytes(storage_client,bucket_name,PT_FEEDER_MANIFEST_BLOB)
        if not raw: return {}
        obj=json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj,dict) else {}
    except Exception:
        return {}


def _pt_load_season(season: int, *, storage_client, bucket_name: str, force_web: bool=False, log_func=print) -> tuple[pd.DataFrame,dict[str,Any]]:
    season=int(season); raw_path=f"{PT_RAW_PREFIX}/ncaa{season}.csv"; norm_path=f"{PT_NORMALIZED_PREFIX}/ncaa{season}.csv"
    current=season>=PT_CURRENT_SEASON
    raw=None; source=""; web_exc=None

    # V2.17: GCS is the primary model-side interface for every season. Completed
    # historical seasons are immutable. The current-season YTD archive can remain
    # valid for a week; the separate live/current feed has the tighter freshness gate.
    max_age=PT_CURRENT_ARCHIVE_MAX_AGE_HOURS if current else None
    raw,gdiag=_pt_gcs_raw(storage_client,bucket_name,raw_path,max_age_hours=max_age)
    if raw is not None:
        source="GCS_FEEDER_RAW"
        log_func(f"[NCAAF-PT-GCS] season={season} kind=ARCHIVE status=PASS age_hours={gdiag.get('age_hours')} path=gs://{bucket_name}/{raw_path} authority=0")
    else:
        log_func(f"[NCAAF-PT-GCS] season={season} kind=ARCHIVE status={gdiag.get('status')} age_hours={gdiag.get('age_hours')} path=gs://{bucket_name}/{raw_path} authority=0")

    # Direct/relay web remains a best-effort fallback only.  It is no longer the
    # contract Heavy/Weekly rely on, and can be disabled with PT_ALLOW_WEB_FALLBACK=0.
    if raw is None and PT_ALLOW_WEB_FALLBACK:
        try:
            raw,web_source=_pt_fetch_csv_resilient(PT_ARCHIVE_URL.format(season=season),referer=PT_ARCHIVE_PAGE_URL)
            source=web_source
            try: storage_client.bucket(bucket_name).blob(raw_path).upload_from_string(raw,content_type="text/csv")
            except Exception: pass
        except Exception as exc:
            web_exc=exc
    if raw is None:
        err=(f"{type(web_exc).__name__}:{web_exc}" if web_exc else f"GCS_{gdiag.get('status')}")
        log_func(f"[NCAAF-PT-SEASON] season={season} status=UNAVAILABLE error={err} gcs_status={gdiag.get('status')} authority=0")
        return pd.DataFrame(),{"status":"UNAVAILABLE","season":season,"error":err,"gcs":gdiag,"authority":0}

    _verified_map=_pt_load_header_manifest(storage_client,bucket_name)
    frame,diag=_pt_parse_csv(raw,season,verified_header_map=_verified_map,source_context=f"ARCHIVE_{source}",log_func=log_func)
    diag["source"]=source; diag["url"]=PT_ARCHIVE_URL.format(season=season); diag["gcs"]=gdiag; diag["authority"]=0
    if not frame.empty:
        try: storage_client.bucket(bucket_name).blob(norm_path).upload_from_string(frame.to_csv(index=False).encode(),content_type="text/csv")
        except Exception: pass
    log_func(f"[NCAAF-PT-SEASON] season={season} status={diag.get('status')} source={source} rows={len(frame)} full_five={int(np.isfinite(pd.to_numeric(frame.get('meta_margin_home'),errors='coerce')).sum()) if not frame.empty else 0} authority=0")
    return frame,diag



def _pt_load_live_current(*, storage_client, bucket_name: str, log_func=print) -> tuple[pd.DataFrame,dict[str,Any]]:
    """Load current-week ratings with GCS-first, name-safe identity validation.

    The manual uploader writes the exact live CSV to GCS. Heavy/Weekly consume
    that validated CSV directly; no named live HTML page is required. Missing games are
    normal: Prediction Tracker is treated as a sparse external signal, not as the
    authoritative NCAA schedule.
    """
    raw_path=f"{PT_RAW_PREFIX}/ncaapredictions.csv"
    raw,gcsv=_pt_gcs_raw(storage_client,bucket_name,raw_path,max_age_hours=PT_FEEDER_CURRENT_MAX_AGE_HOURS)
    hraw=None
    ghtml={"status":"NOT_REQUIRED_CSV_ONLY","age_hours":None}
    source=""; csv_exc=None; html_exc=None; infer_diag={"status":"NOT_REQUIRED_CSV_ONLY"}; verified_map={}
    if raw is not None:
        source="GCS_FEEDER_LIVE"
        log_func(f"[NCAAF-PT-GCS] season={PT_CURRENT_SEASON} kind=LIVE_CSV status=PASS age_hours={gcsv.get('age_hours')} path=gs://{bucket_name}/{raw_path} authority=0")
    else:
        log_func(f"[NCAAF-PT-GCS] season={PT_CURRENT_SEASON} kind=LIVE_CSV status={gcsv.get('status')} age_hours={gcsv.get('age_hours')} path=gs://{bucket_name}/{raw_path} authority=0")
    log_func(f"[NCAAF-PT-GCS] season={PT_CURRENT_SEASON} kind=LIVE_HTML status=NOT_REQUIRED_CSV_ONLY authority=0")

    if raw is None and PT_ALLOW_WEB_FALLBACK:
        try:
            raw,_live_source=_pt_fetch_csv_resilient(PT_LIVE_CSV_URL,referer=PT_LIVE_PAGE_URL); source=_live_source.replace("WEB_SESSION","WEB_LIVE_SESSION")
            try: storage_client.bucket(bucket_name).blob(raw_path).upload_from_string(raw,content_type="text/csv")
            except Exception: pass
        except Exception as exc:
            csv_exc=exc
    # V2.17.1: live HTML is not part of the operational contract.
    # Current data comes from the manually uploaded Prediction Tracker CSV only.

    if raw is not None and hraw is not None:
        inferred,infer_diag=_pt_infer_verified_header_map(raw,hraw)
        if inferred:
            verified_map={**verified_map,**inferred}
            _pt_save_header_manifest(storage_client,bucket_name,verified_map,infer_diag)
        log_func(f"[NCAAF-PT-HEADER-VERIFY] season={PT_CURRENT_SEASON} status={infer_diag.get('status')} overlap={infer_diag.get('overlap_games',0)} mapping={inferred} ambiguous={infer_diag.get('ambiguous',{})} authority=0")

    frame=pd.DataFrame(); diag={"status":"UNAVAILABLE","season":PT_CURRENT_SEASON}
    if raw is not None:
        frame,diag=_pt_parse_csv(raw,PT_CURRENT_SEASON,verified_header_map=verified_map,source_context="LIVE_GCS_FIRST",log_func=log_func)
        source=source or "GCS_FEEDER_LIVE"
        if diag.get("metamodel_status")!="FULL_FIVE_VERIFIED" and hraw is not None:
            hframe,hdiag=_pt_parse_live_html(hraw,PT_CURRENT_SEASON,log_func=log_func)
            if not hframe.empty and hdiag.get("metamodel_status")=="FULL_FIVE_VERIFIED":
                frame,diag=hframe,hdiag; source="GCS_FEEDER_LIVE_HTML_NAMED" if gcsv.get("status")=="PASS" or ghtml.get("status")=="PASS" else "WEB_LIVE_HTML_NAMED"
    elif hraw is not None:
        frame,diag=_pt_parse_live_html(hraw,PT_CURRENT_SEASON,log_func=log_func); source="GCS_FEEDER_LIVE_HTML_NAMED" if ghtml.get("status")=="PASS" else "WEB_LIVE_HTML_FALLBACK"

    if frame.empty:
        err_csv=(f"{type(csv_exc).__name__}:{csv_exc}" if csv_exc else f"GCS_{gcsv.get('status')}")
        err_html=(f"{type(html_exc).__name__}:{html_exc}" if html_exc else f"GCS_{ghtml.get('status')}")
        log_func(f"[NCAAF-PT-LIVE] season={PT_CURRENT_SEASON} status=UNAVAILABLE csv_error={err_csv} html_error={err_html} authority=0")
        return pd.DataFrame(),{"status":"UNAVAILABLE","season":PT_CURRENT_SEASON,"error":f"CSV={err_csv}; HTML={err_html}","gcs_csv":gcsv,"gcs_html":ghtml,"url":PT_LIVE_CSV_URL,"page_url":PT_LIVE_PAGE_URL,"authority":0}

    frame=frame.copy(); frame["source_kind"]="LIVE_CURRENT"; frame["source_priority"]=2
    try: storage_client.bucket(bucket_name).blob(PT_CURRENT_LIVE_BLOB).upload_from_string(frame.to_csv(index=False).encode(),content_type="text/csv")
    except Exception: pass
    diag.update({"source":source,"source_kind":"LIVE_CURRENT","url":PT_LIVE_CSV_URL,"page_url":PT_LIVE_PAGE_URL,"header_verification":infer_diag,"verified_header_map":verified_map,"gcs_csv":gcsv,"gcs_html":ghtml,"authority":0})
    log_func(f"[NCAAF-PT-LIVE] season={PT_CURRENT_SEASON} status={diag.get('status')} source={source} header_contract={diag.get('metamodel_status')} rows={len(frame)} full_five={int(np.isfinite(pd.to_numeric(frame.get('meta_margin_home'),errors='coerce')).sum()) if not frame.empty else 0} authority=0")
    return frame,diag


def _pt_merge_current_season(archive: pd.DataFrame, live: pd.DataFrame, *, log_func=print) -> tuple[pd.DataFrame,dict[str,Any]]:
    """Merge 2026 season archive with the separate live/current-week feed.

    All archive rows are retained except an exact home/away pair also present in
    the live feed; for that overlap the live row wins because its ratings/line are
    newer.  This preserves old 2026 ratings while exposing the current week.
    """
    a=archive.copy() if isinstance(archive,pd.DataFrame) else pd.DataFrame()
    l=live.copy() if isinstance(live,pd.DataFrame) else pd.DataFrame()
    if not a.empty:
        a["source_kind"]="ARCHIVE_SEASON_TO_DATE"; a["source_priority"]=1
    if not l.empty:
        l["source_kind"]="LIVE_CURRENT"; l["source_priority"]=2
    def pair(df):
        if df.empty: return pd.Series(dtype=str)
        return df.get("home_key",pd.Series("",index=df.index)).astype(str)+"|"+df.get("away_key",pd.Series("",index=df.index)).astype(str)
    lp=pair(l) if not l.empty else pd.Series(dtype=str)
    live_counts=lp.loc[lp.str.len().gt(1)].value_counts().to_dict() if not l.empty else {}
    removed=0
    if not a.empty and live_counts:
        ap=pair(a)
        drop_idx=[]
        # Replace only the newest N archive occurrence(s) for each live pair.
        # Earlier same-season rematches are preserved rather than being wiped out.
        for pkey,n_live in live_counts.items():
            hits=list(a.index[ap.eq(pkey)])
            if hits:
                take=min(len(hits),int(n_live))
                drop_idx.extend(hits[-take:])
        if drop_idx:
            removed=len(drop_idx); a=a.drop(index=drop_idx).copy()
    merged=pd.concat([a,l],ignore_index=True,sort=False) if (not a.empty or not l.empty) else pd.DataFrame()
    diag={
        "status":"PASS" if not merged.empty else "UNAVAILABLE","season":PT_CURRENT_SEASON,
        "archive_rows":int(len(archive)) if isinstance(archive,pd.DataFrame) else 0,
        "live_rows":int(len(live)) if isinstance(live,pd.DataFrame) else 0,
        "archive_overlap_replaced_by_live":removed,"merged_rows":int(len(merged)),
        "live_full_five_rows":int(np.isfinite(pd.to_numeric(l.get("meta_margin_home"),errors="coerce")).sum()) if not l.empty else 0,
        "merged_full_five_rows":int(np.isfinite(pd.to_numeric(merged.get("meta_margin_home"),errors="coerce")).sum()) if not merged.empty else 0,
        "authority":0,
    }
    log_func(f"[NCAAF-PT-CURRENT-MERGE] status={diag['status']} archive_rows={diag['archive_rows']} live_rows={diag['live_rows']} overlap_replaced={removed} merged_rows={diag['merged_rows']} live_full_five={diag['live_full_five_rows']} authority=0")
    return merged,diag

def _pt_query_df(client, sql: str) -> pd.DataFrame:
    """Best-effort BigQuery -> DataFrame adapter used only for identity metadata."""
    try:
        job=client.query(sql)
        if hasattr(job,"to_dataframe"):
            return job.to_dataframe()
        res=job.result() if hasattr(job,"result") else job
        if hasattr(res,"to_dataframe"):
            return res.to_dataframe()
        return pd.DataFrame([dict(r.items()) for r in res])
    except Exception:
        return pd.DataFrame()


def _pt_resolve_internal_target(raw_target: Any, valid_internal: set[str]) -> str | None:
    """Resolve an identity-table target to the exact key used by the research cache."""
    from difflib import SequenceMatcher
    k=_pt_team_key(raw_target)
    if not k:
        return None
    if k in valid_internal:
        return k
    kt=set(k.split())
    pref=[c for c in valid_internal if c.startswith(k+" ") or k.startswith(c+" ")]
    if len(pref)==1:
        return pref[0]
    subs=[c for c in valid_internal if kt and kt.issubset(set(c.split()))]
    if len(subs)==1:
        return subs[0]
    scored=sorted(((SequenceMatcher(None,k,c).ratio(),c) for c in valid_internal),reverse=True)
    if scored and scored[0][0]>=0.96 and (len(scored)==1 or scored[0][0]-scored[1][0]>=0.05):
        return scored[0][1]
    return None


def _pt_hard_external_aliases(valid_internal: set[str]) -> tuple[dict[str,str],dict[str,list[str]]]:
    """Resolve the eight remaining Prediction Tracker historical aliases.

    These are explicit source-name identities observed in the V2.21 coverage audit.
    Each mapping still has to resolve to exactly one team key present in the current
    NCAAF research cache; ambiguous or missing targets fail closed.
    """
    valid=sorted({_pt_team_key(x) for x in valid_internal if _pt_team_key(x)})

    def pick(*rules):
        hits=[]
        for cand in valid:
            for rule in rules:
                if rule(cand):
                    hits.append(cand)
                    break
        hits=sorted(set(hits))
        return hits[0] if len(hits)==1 else None, hits

    specs={
        "miami ohio": (
            lambda c: c=="miami oh redhawks",
            lambda c: c.startswith("miami oh "),
            lambda c: c.startswith("miami ohio "),
        ),
        "miami florida": (
            lambda c: c=="miami hurricanes",
            lambda c: c.startswith("miami fl "),
            lambda c: c.startswith("miami florida "),
        ),
        "mississippi": (
            lambda c: c=="ole mississippi rebels",
            lambda c: c.startswith("ole miss "),
            lambda c: c.startswith("ole mississippi "),
        ),
        "texas san antonio": (
            lambda c: c.startswith("utsa "),
            lambda c: ("texas" in c.split() and "san" in c.split() and "antonio" in c.split()),
        ),
        "troy state": (
            lambda c: c.startswith("troy "),
        ),
        "louisiana lafayette": (
            lambda c: c=="louisiana ragin cajuns",
            lambda c: c.startswith("louisiana ragin "),
            lambda c: c.startswith("louisiana lafayette "),
        ),
        "central florida": (
            lambda c: c.startswith("ucf "),
            lambda c: c.startswith("central florida "),
        ),
        "florida intl": (
            lambda c: c.startswith("fiu "),
            lambda c: c.startswith("florida international "),
        ),
    }

    resolved={}
    unresolved={}
    for source,rules in specs.items():
        target,hits=pick(*rules)
        if target:
            resolved[_pt_team_key(source)]=target
        else:
            unresolved[_pt_team_key(source)]=hits
    return resolved,unresolved


def _pt_load_canonical_team_aliases(dashboard_module, internal_names: Iterable[str], *, log_func=print) -> dict[str,str]:
    """Reuse canonical NCAAF identity data instead of maintaining a PT-only name map.

    We read season-aware alignment identities, the persisted NCAAF source alias
    table, and already-validated historical raw source mappings. Only aliases whose
    destination resolves uniquely to a team in the current research cache are used.
    Conflicting aliases are discarded rather than guessed.
    """
    valid={_pt_team_key(x) for x in internal_names if _pt_team_key(x)}
    if not valid or dashboard_module is None:
        return {}
    client=getattr(dashboard_module,"bq_client",None)
    if client is None:
        return {}
    project=(getattr(dashboard_module,"GCP_PROJECT_ID",None) or getattr(dashboard_module,"PROJECT_ID",None)
             or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or "sharplogger")
    dataset=(getattr(dashboard_module,"BQ_DATASET",None) or os.getenv("BQ_DATASET") or "sharp_data")
    alias_map={}
    conflicts=set()

    def add(source_name: Any, target_name: Any):
        sk=_pt_team_key(source_name)
        target=_pt_resolve_internal_target(target_name,valid)
        if not sk or not target:
            return
        if sk not in alias_map:
            alias_map[sk]=target
        elif alias_map[sk]!=target:
            conflicts.add(sk)
            alias_map[sk]=None

    align=_pt_query_df(client,f"""
      SELECT Team, Team_Norm, Canonical_Team_Name, Canonical_Team_Norm
      FROM `{project}.{dataset}.team_sport_alignment_history`
      WHERE UPPER(Sport)='NCAAF'
    """)
    for _,r in align.iterrows():
        target=None
        for c in ("Team_Norm","Canonical_Team_Norm","Team","Canonical_Team_Name"):
            target=_pt_resolve_internal_target(r.get(c),valid)
            if target:
                break
        if not target:
            continue
        for c in ("Team","Team_Norm","Canonical_Team_Name","Canonical_Team_Norm"):
            add(r.get(c),target)

    alias_rows=_pt_query_df(client,f"""
      SELECT Source_Team_Name, Team_Norm
      FROM `{project}.{dataset}.ncaaf_source_team_aliases`
      WHERE Source_Team_Name IS NOT NULL AND Team_Norm IS NOT NULL
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY LOWER(TRIM(Source_Team_Name))
        ORDER BY Updated_At DESC
      )=1
    """)
    for _,r in alias_rows.iterrows():
        add(r.get("Source_Team_Name"),r.get("Team_Norm"))

    raw_rows=_pt_query_df(client,f"""
      SELECT Source_Team_Name, Team_Norm
      FROM `{project}.{dataset}.ncaaf_historical_game_side_raw`
      WHERE Source_Team_Name IS NOT NULL AND Team_Norm IS NOT NULL
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY LOWER(TRIM(Source_Team_Name))
        ORDER BY Season DESC, Game_Date DESC
      )=1
    """)
    for _,r in raw_rows.iterrows():
        add(r.get("Source_Team_Name"),r.get("Team_Norm"))

    # Common external abbreviations. These still resolve fail-closed to a unique
    # team in the current cache before being admitted.
    static={
        "app state":"appalachian state",
        "uconn":"connecticut",
        "umass":"massachusetts",
        "ole miss":"mississippi",
        "nc state":"north carolina state",
        "pitt":"pittsburgh",
        "fiu":"florida international",
        "fau":"florida atlantic",
        "utsa":"texas san antonio",
        "utep":"texas el paso",
        "uab":"alabama birmingham",
        "ucf":"central florida",
        "usf":"south florida",
        "smu":"southern methodist",
        "tcu":"texas christian",
        "wku":"western kentucky",
        "mtsu":"middle tennessee",
        "niu":"northern illinois",
        "la tech":"louisiana tech",
        "ul monroe":"louisiana monroe",
        "ul lafayette":"louisiana lafayette",
        "miami fl":"miami florida",
        "miami oh":"miami ohio",
        "southern miss":"southern mississippi",
    }
    for a,t in static.items():
        add(a,t)

    # V2.22: explicit PT source identities discovered by the V2.21 audit.
    # These are cache-validated exact identities, not fuzzy guesses.
    hard_resolved,hard_unresolved=_pt_hard_external_aliases(valid)
    for a,t in hard_resolved.items():
        if a not in conflicts:
            alias_map[a]=t
    log_func(
        f"[NCAAF-PT-HARD-ALIASES] resolved={len(hard_resolved)}/8 "
        f"mappings={json.dumps(hard_resolved,sort_keys=True)} "
        f"unresolved={json.dumps(hard_unresolved,sort_keys=True)} authority=0"
    )

    out={k:v for k,v in alias_map.items() if v and k not in conflicts}
    log_func(
        f"[NCAAF-PT-ALIASES] status=PASS alignment_rows={len(align)} persisted_alias_rows={len(alias_rows)} "
        f"historical_alias_rows={len(raw_rows)} usable_aliases={len(out)} conflicts={len(conflicts)} authority=0"
    )
    if conflicts:
        log_func(
            f"[NCAAF-PT-ALIAS-CONFLICTS] count={len(conflicts)} "
            f"names={json.dumps(sorted(conflicts)[:50])} fail_closed=TRUE authority=0"
        )
    return out


def _pt_candidate_suggestions(unresolved: Iterable[str], internal_names: Iterable[str], limit: int=3) -> dict[str,list[tuple[str,float]]]:
    from difflib import SequenceMatcher
    ints=sorted({_pt_team_key(x) for x in internal_names if _pt_team_key(x)})
    out={}
    for raw in unresolved:
        scored=sorted(((SequenceMatcher(None,str(raw),cand).ratio(),cand) for cand in ints),reverse=True)
        out[str(raw)]=[(c,float(s)) for s,c in scored[:limit]]
    return out


def _pt_candidate_map(external_names: Iterable[str], internal_names: Iterable[str], alias_hints: dict[str,str] | None=None) -> tuple[dict[str,str],list[str]]:
    from difflib import SequenceMatcher
    ints=sorted({_pt_team_key(x) for x in internal_names if _pt_team_key(x)})
    hints={_pt_team_key(k):_pt_team_key(v) for k,v in (alias_hints or {}).items() if _pt_team_key(k) and _pt_team_key(v)}
    mapping={}; unresolved=[]
    for raw in sorted({_pt_team_key(x) for x in external_names if _pt_team_key(x)}):
        if raw in ints: mapping[raw]=raw; continue
        hinted=hints.get(raw)
        if hinted in ints:
            mapping[raw]=hinted; continue
        rt=set(raw.split()); scored=[]
        for cand in ints:
            ct=set(cand.split())
            subset=rt.issubset(ct) and len(rt)>=1
            tok=(len(rt&ct)/max(len(rt),1)) if rt else 0.0
            seq=SequenceMatcher(None,raw,cand).ratio()
            prefix=1.0 if (cand.startswith(raw+" ") or raw.startswith(cand+" ")) else 0.0
            score=max(seq,0.88 if subset else 0.0,0.92 if prefix else 0.0,0.55*tok+0.45*seq)
            scored.append((score,cand))
        scored.sort(reverse=True)
        if scored and scored[0][0]>=0.78 and (len(scored)==1 or scored[0][0]-scored[1][0]>=0.035): mapping[raw]=scored[0][1]
        else: unresolved.append(raw)
    return mapping,unresolved


def _pt_attach_history_to_cache(dashboard_module, ext: pd.DataFrame, *, log_func=print) -> dict[str,Any]:
    """Attach sparse Prediction Tracker history without shrinking the game universe.

    A matched Prediction Tracker row may contain any number of source-native external
    systems. Every available individual index is preserved for research. The fixed
    published META_MARGIN remains a separate frozen five-system benchmark and is attached
    only when all five exact named benchmark inputs are present. Missing Prediction
    Tracker games remain NaN/absent external context and never remove or neutralize an
    internal NCAAF game.
    """
    cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}
    games=cache.get("games"); mg=cache.get("miner_games")
    if not isinstance(games,pd.DataFrame) or games.empty or not isinstance(mg,pd.DataFrame) or len(mg)!=len(games):
        return {"status":"CACHE_UNAVAILABLE","matched_rows":0,"full_five_matched_rows":0,"partial_matched_rows":0,"no_pt_rows":0,"authority":0}
    g=games.copy(); m=mg.copy()
    season=pd.to_numeric(g.get("Season"),errors="coerce")
    team=g.get("Team_Norm",g.get("Team",pd.Series("",index=g.index))).astype(str).map(_pt_team_key)
    opp=g.get("Opponent_Norm",g.get("Opponent",pd.Series("",index=g.index))).astype(str).map(_pt_team_key)
    is_home=pd.to_numeric(g.get("Is_Home",pd.Series(np.nan,index=g.index)),errors="coerce")
    internal_names=pd.concat([team,opp],ignore_index=True).dropna().astype(str).tolist()
    ext_source=ext.copy()
    ext_source["season"]=pd.to_numeric(ext_source.get("season"),errors="coerce")
    alias_hints=_pt_load_canonical_team_aliases(dashboard_module,internal_names,log_func=log_func)
    emap,unresolved=_pt_candidate_map(
        pd.concat([ext_source.get("home_key",pd.Series(dtype=str)),ext_source.get("away_key",pd.Series(dtype=str))],ignore_index=True),
        internal_names,alias_hints=alias_hints
    )
    ex=ext_source.copy()
    ex["_pt_src_row_id"]=np.arange(len(ex),dtype=int)
    ex["home_i"]=ex["home_key"].map(emap); ex["away_i"]=ex["away_key"].map(emap)
    _alias_ok=ex["home_i"].notna()&ex["away_i"].notna()
    alias_resolved_source_rows=int(_alias_ok.sum())
    source_rows=int(len(ex))
    unresolved_source_rows=int((~_alias_ok).sum())
    if unresolved:
        _usage=pd.concat(
            [ext_source.get("home_key",pd.Series(dtype=str)),ext_source.get("away_key",pd.Series(dtype=str))],
            ignore_index=True
        ).map(_pt_team_key).value_counts()
        _sugg=_pt_candidate_suggestions(unresolved,internal_names,limit=3)
        _detail=[]
        for _u in sorted(unresolved,key=lambda x:int(_usage.get(x,0)),reverse=True):
            _detail.append({
                "team":_u,
                "appearances":int(_usage.get(_u,0)),
                "candidates":[{"team":c,"score":round(s,4)} for c,s in _sugg.get(_u,[])]
            })
        log_func(
            f"[NCAAF-PT-UNRESOLVED] count={len(unresolved)} source_rows_impacted={unresolved_source_rows} "
            f"details={json.dumps(_detail[:50],sort_keys=True)} authority=0"
        )
    ex=ex.loc[_alias_ok].copy()
    ex["pair_key"]=ex["season"].astype(int).astype(str)+"|"+ex["home_i"].astype(str)+"|"+ex["away_i"].astype(str)
    counts=ex["pair_key"].value_counts(); unique=ex.loc[ex["pair_key"].map(counts).eq(1)].copy().set_index("pair_key",drop=False)
    # Duplicate season matchups (e.g., conference-title rematches) are matched only
    # when both sources expose the same calendar date; otherwise they fail closed.
    ex_date=ex.copy(); ex_date["date_key"]=ex_date["pair_key"].astype(str)+"|"+ex_date.get("game_date",pd.Series("",index=ex_date.index)).fillna("").astype(str)
    dvc=ex_date["date_key"].value_counts(); ex_date=ex_date.loc[ex_date["date_key"].str.len().gt(ex_date["pair_key"].str.len()+1)&ex_date["date_key"].map(dvc).eq(1)].set_index("date_key",drop=False)
    home=np.where(is_home.eq(1),team,np.where(is_home.eq(0),opp,"")); away=np.where(is_home.eq(1),opp,np.where(is_home.eq(0),team,""))
    key=pd.Series([f"{int(s)}|{h}|{a}" if np.isfinite(s) and h and a else "" for s,h,a in zip(season.to_numpy(float),home,away)],index=g.index)
    gd=pd.to_datetime(g.get("Game_Date",pd.Series(pd.NaT,index=g.index)),errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

    meta=np.full(len(g),np.nan); cnt=np.full(len(g),np.nan); pavg=np.full(len(g),np.nan); pmed=np.full(len(g),np.nan); pstd=np.full(len(g),np.nan); extline=np.full(len(g),np.nan)
    listed=np.zeros(len(g),dtype=float); full_five=np.zeros(len(g),dtype=float)
    comp={k:np.full(len(g),np.nan) for k in PT_PUBLISHED_WEIGHTS}

    # Full external research universe. Columns were preserved by _pt_parse_frame
    # using exact source-native Prediction Tracker headers.
    _idx_ext_cols=sorted([c for c in ex.columns if str(c).startswith("ptidx__")])
    _idx_keys=[str(c)[len("ptidx__"):] for c in _idx_ext_cols]
    idx_comp={k:np.full(len(g),np.nan) for k in _idx_keys}

    matched=0; matched_any_component=0; full_five_matched=0; partial_matched=0
    matched_any_index=0; matched_by_season={}; matched_source_ids=set()

    for i,k in enumerate(key.astype(str)):
        if not k: continue
        dk=f"{k}|{gd.iloc[i]}" if gd.iloc[i] else ""
        if dk and dk in ex_date.index: r=ex_date.loc[dk]
        elif k in unique.index: r=unique.loc[k]
        else: continue
        if isinstance(r,pd.DataFrame): continue

        listed[i]=1.0
        matched+=1
        try:
            matched_source_ids.add(int(r.get("_pt_src_row_id")))
        except Exception:
            pass
        _sy=int(season.iloc[i]) if pd.notna(season.iloc[i]) else None
        if _sy is not None:
            matched_by_season[_sy]=matched_by_season.get(_sy,0)+1
        orient=1.0 if is_home.iloc[i]==1 else -1.0

        component_count=0
        for _k in PT_PUBLISHED_WEIGHTS:
            _v=pd.to_numeric(pd.Series([r.get(_k,np.nan)]),errors="coerce").iloc[0]
            if pd.notna(_v):
                comp[_k][i]=orient*float(_v)
                component_count+=1

        _index_count=0
        for _k,_c in zip(_idx_keys,_idx_ext_cols):
            _v=pd.to_numeric(pd.Series([r.get(_c,np.nan)]),errors="coerce").iloc[0]
            if pd.notna(_v):
                idx_comp[_k][i]=orient*float(_v)
                _index_count+=1
        if _index_count>0:
            matched_any_index+=1

        cnt[i]=float(component_count)
        if component_count>0:
            matched_any_component+=1
        if component_count==5:
            full_five[i]=1.0
            full_five_matched+=1
        elif component_count>0:
            partial_matched+=1

        # Preserve the published META contract: no reweighting, imputation, or
        # renormalization when fewer than five exact systems are available.
        hm=pd.to_numeric(pd.Series([r.get("meta_margin_home")]),errors="coerce").iloc[0]
        if pd.notna(hm) and component_count==5:
            meta[i]=orient*float(hm)

        _pa=pd.to_numeric(pd.Series([r.get("prediction_avg_home",np.nan)]),errors="coerce").iloc[0]
        if pd.notna(_pa): pavg[i]=orient*float(_pa)
        _pm=pd.to_numeric(pd.Series([r.get("prediction_median_home",np.nan)]),errors="coerce").iloc[0]
        if pd.notna(_pm): pmed[i]=orient*float(_pm)
        _ps=pd.to_numeric(pd.Series([r.get("prediction_std_home",np.nan)]),errors="coerce").iloc[0]
        if pd.notna(_ps): pstd[i]=float(_ps)
        _ol=pd.to_numeric(pd.Series([r.get("tracker_open_home",np.nan)]),errors="coerce").iloc[0]
        if pd.notna(_ol): extline[i]=orient*float(_ol)

    market_margin=pd.to_numeric(g.get("Market_Open_Margin"),errors="coerce").to_numpy(float) if "Market_Open_Margin" in g.columns else -pd.to_numeric(g.get("Consensus_Open_Spread"),errors="coerce").to_numpy(float)
    meta_edge=meta-market_margin
    comp_edges={k:(v-market_margin) for k,v in comp.items()}
    comp_mat=np.column_stack([comp[k] for k in PT_PUBLISHED_WEIGHTS])
    edge_mat=np.column_stack([comp_edges[k] for k in PT_PUBLISHED_WEIGHTS])
    finite_comp=np.isfinite(comp_mat)
    comp_n=finite_comp.sum(axis=1)
    comp_std=np.full(len(g),np.nan); _std_ok=comp_n>=2
    if _std_ok.any(): comp_std[_std_ok]=np.nanstd(comp_mat[_std_ok],axis=1)
    agree_team=np.sum(np.isfinite(edge_mat)&(edge_mat>0),axis=1).astype(float)
    agree_opp=np.sum(np.isfinite(edge_mat)&(edge_mat<0),axis=1).astype(float)

    # Full Prediction Tracker index-universe aggregates. All of these remain one
    # correlated external-rating family. Multi-variant providers (Sagarin, Pi,
    # regression, Payne) are also collapsed to one source-cluster vote for the
    # cluster-balanced consensus diagnostics.
    if _idx_keys:
        idx_mat=np.column_stack([idx_comp[k] for k in _idx_keys])
        idx_edge=idx_mat-market_margin[:,None]
        idx_finite=np.isfinite(idx_mat)
        idx_n=idx_finite.sum(axis=1).astype(float)
        idx_team_count=np.sum(np.isfinite(idx_edge)&(idx_edge>0),axis=1).astype(float)
        idx_opp_count=np.sum(np.isfinite(idx_edge)&(idx_edge<0),axis=1).astype(float)
        idx_strong_team_count=np.sum(np.isfinite(idx_edge)&(idx_edge>=2),axis=1).astype(float)
        idx_strong_opp_count=np.sum(np.isfinite(idx_edge)&(idx_edge<=-2),axis=1).astype(float)

        idx_mean_margin=np.full(len(g),np.nan); idx_median_margin=np.full(len(g),np.nan)
        idx_std=np.full(len(g),np.nan); idx_iqr=np.full(len(g),np.nan)
        for _i in range(len(g)):
            _v=idx_mat[_i,np.isfinite(idx_mat[_i])]
            if _v.size:
                idx_mean_margin[_i]=float(np.mean(_v))
                idx_median_margin[_i]=float(np.median(_v))
                if _v.size>=2:
                    idx_std[_i]=float(np.std(_v))
                    idx_iqr[_i]=float(np.percentile(_v,75)-np.percentile(_v,25))
        idx_mean_edge=idx_mean_margin-market_margin
        idx_median_edge=idx_median_margin-market_margin
        with np.errstate(divide="ignore",invalid="ignore"):
            idx_team_frac=np.where(idx_n>0,idx_team_count/idx_n,np.nan)
            idx_opp_frac=np.where(idx_n>0,idx_opp_count/idx_n,np.nan)
            idx_strong_team_frac=np.where(idx_n>0,idx_strong_team_count/idx_n,np.nan)
            idx_strong_opp_frac=np.where(idx_n>0,idx_strong_opp_count/idx_n,np.nan)

        _cluster_members={}
        for _j,_k in enumerate(_idx_keys):
            _cluster_members.setdefault(_pt_index_source_cluster(_k),[]).append(_j)
        _cluster_names=sorted(_cluster_members)
        _cluster_mat=np.full((len(g),len(_cluster_names)),np.nan)
        for _cj,_cn in enumerate(_cluster_names):
            _cols=_cluster_members[_cn]
            _sub=idx_mat[:,_cols]
            for _i in range(len(g)):
                _v=_sub[_i,np.isfinite(_sub[_i])]
                if _v.size:
                    _cluster_mat[_i,_cj]=float(np.mean(_v))
        _cluster_edges=_cluster_mat-market_margin[:,None]
        idx_cluster_n=np.isfinite(_cluster_mat).sum(axis=1).astype(float)
        idx_cluster_median_margin=np.full(len(g),np.nan)
        for _i in range(len(g)):
            _v=_cluster_mat[_i,np.isfinite(_cluster_mat[_i])]
            if _v.size:
                idx_cluster_median_margin[_i]=float(np.median(_v))
        idx_cluster_median_edge=idx_cluster_median_margin-market_margin
        _ct=np.sum(np.isfinite(_cluster_edges)&(_cluster_edges>0),axis=1).astype(float)
        _co=np.sum(np.isfinite(_cluster_edges)&(_cluster_edges<0),axis=1).astype(float)
        with np.errstate(divide="ignore",invalid="ignore"):
            idx_cluster_team_frac=np.where(idx_cluster_n>0,_ct/idx_cluster_n,np.nan)
            idx_cluster_opp_frac=np.where(idx_cluster_n>0,_co/idx_cluster_n,np.nan)

        # V2.18.2: consensus independent of the frozen META constituents.
        # We remove the entire source cluster for every META constituent, not
        # merely the exact five columns. This prevents Pi Bias in META from being
        # "independently" confirmed by Pi Mean / Pi Ratings in the broad consensus.
        _meta_source_clusters={_pt_index_source_cluster(k) for k in PT_META_CONSTITUENT_INDEX_IDS}
        _exmeta_cols=[
            j for j,k in enumerate(_idx_keys)
            if _pt_index_source_cluster(k) not in _meta_source_clusters
        ]
        _exmeta_mat=idx_mat[:,_exmeta_cols] if _exmeta_cols else np.empty((len(g),0))
        _exmeta_edge=_exmeta_mat-market_margin[:,None] if _exmeta_cols else np.empty((len(g),0))
        exmeta_n=np.isfinite(_exmeta_mat).sum(axis=1).astype(float)
        exmeta_mean_margin=np.full(len(g),np.nan)
        exmeta_median_margin=np.full(len(g),np.nan)
        exmeta_std=np.full(len(g),np.nan)
        exmeta_iqr=np.full(len(g),np.nan)
        for _i in range(len(g)):
            _v=_exmeta_mat[_i,np.isfinite(_exmeta_mat[_i])]
            if _v.size:
                exmeta_mean_margin[_i]=float(np.mean(_v))
                exmeta_median_margin[_i]=float(np.median(_v))
                if _v.size>=2:
                    exmeta_std[_i]=float(np.std(_v))
                    exmeta_iqr[_i]=float(np.percentile(_v,75)-np.percentile(_v,25))
        exmeta_mean_edge=exmeta_mean_margin-market_margin
        exmeta_median_edge=exmeta_median_margin-market_margin
        _ext=np.sum(np.isfinite(_exmeta_edge)&(_exmeta_edge>0),axis=1).astype(float)
        _exo=np.sum(np.isfinite(_exmeta_edge)&(_exmeta_edge<0),axis=1).astype(float)
        _exst=np.sum(np.isfinite(_exmeta_edge)&(_exmeta_edge>=2),axis=1).astype(float)
        _exso=np.sum(np.isfinite(_exmeta_edge)&(_exmeta_edge<=-2),axis=1).astype(float)
        with np.errstate(divide="ignore",invalid="ignore"):
            exmeta_team_frac=np.where(exmeta_n>0,_ext/exmeta_n,np.nan)
            exmeta_opp_frac=np.where(exmeta_n>0,_exo/exmeta_n,np.nan)
            exmeta_strong_team_frac=np.where(exmeta_n>0,_exst/exmeta_n,np.nan)
            exmeta_strong_opp_frac=np.where(exmeta_n>0,_exso/exmeta_n,np.nan)

        _exmeta_cluster_cols=[
            j for j,cn in enumerate(_cluster_names)
            if cn not in _meta_source_clusters
        ]
        _exmeta_cluster_mat=_cluster_mat[:,_exmeta_cluster_cols] if _exmeta_cluster_cols else np.empty((len(g),0))
        _exmeta_cluster_edge=_exmeta_cluster_mat-market_margin[:,None] if _exmeta_cluster_cols else np.empty((len(g),0))
        exmeta_cluster_n=np.isfinite(_exmeta_cluster_mat).sum(axis=1).astype(float)
        exmeta_cluster_median_margin=np.full(len(g),np.nan)
        for _i in range(len(g)):
            _v=_exmeta_cluster_mat[_i,np.isfinite(_exmeta_cluster_mat[_i])]
            if _v.size:
                exmeta_cluster_median_margin[_i]=float(np.median(_v))
        exmeta_cluster_median_edge=exmeta_cluster_median_margin-market_margin
        _exct=np.sum(np.isfinite(_exmeta_cluster_edge)&(_exmeta_cluster_edge>0),axis=1).astype(float)
        _exco=np.sum(np.isfinite(_exmeta_cluster_edge)&(_exmeta_cluster_edge<0),axis=1).astype(float)
        with np.errstate(divide="ignore",invalid="ignore"):
            exmeta_cluster_team_frac=np.where(exmeta_cluster_n>0,_exct/exmeta_cluster_n,np.nan)
            exmeta_cluster_opp_frac=np.where(exmeta_cluster_n>0,_exco/exmeta_cluster_n,np.nan)
    else:
        idx_mat=np.empty((len(g),0)); idx_edge=np.empty((len(g),0)); idx_n=np.zeros(len(g),dtype=float)
        idx_mean_margin=np.full(len(g),np.nan); idx_median_margin=np.full(len(g),np.nan)
        idx_mean_edge=np.full(len(g),np.nan); idx_median_edge=np.full(len(g),np.nan)
        idx_std=np.full(len(g),np.nan); idx_iqr=np.full(len(g),np.nan)
        idx_team_frac=np.full(len(g),np.nan); idx_opp_frac=np.full(len(g),np.nan)
        idx_strong_team_frac=np.full(len(g),np.nan); idx_strong_opp_frac=np.full(len(g),np.nan)
        idx_cluster_n=np.zeros(len(g),dtype=float)
        idx_cluster_median_edge=np.full(len(g),np.nan)
        idx_cluster_team_frac=np.full(len(g),np.nan); idx_cluster_opp_frac=np.full(len(g),np.nan)
        _cluster_names=[]
        exmeta_n=np.zeros(len(g),dtype=float)
        exmeta_mean_edge=np.full(len(g),np.nan); exmeta_median_edge=np.full(len(g),np.nan)
        exmeta_std=np.full(len(g),np.nan); exmeta_iqr=np.full(len(g),np.nan)
        exmeta_team_frac=np.full(len(g),np.nan); exmeta_opp_frac=np.full(len(g),np.nan)
        exmeta_strong_team_frac=np.full(len(g),np.nan); exmeta_strong_opp_frac=np.full(len(g),np.nan)
        exmeta_cluster_n=np.zeros(len(g),dtype=float)
        exmeta_cluster_median_edge=np.full(len(g),np.nan)
        exmeta_cluster_team_frac=np.full(len(g),np.nan); exmeta_cluster_opp_frac=np.full(len(g),np.nan)

    for df in (g,m):
        df["_V210_PT_META_MARGIN_TEAM"]=meta
        df["_V210_PT_META_EDGE_POINTS"]=meta_edge
        df["_V210_PT_META_SYSTEM_COUNT"]=cnt
        df["_V210_PT_PREDICTION_AVG_TEAM"]=pavg
        df["_V218_PT_PREDICTION_MEDIAN_TEAM"]=pmed
        df["_V218_PT_PREDICTION_STD"]=pstd
        df["_V210_PT_ARCHIVE_OPEN_MARGIN_TEAM"]=extline
        df["_V212_PT_COMPONENT_STD"]=comp_std
        df["_V212_PT_COMPONENT_TEAM_AGREE_COUNT"]=agree_team
        df["_V212_PT_COMPONENT_OPP_AGREE_COUNT"]=agree_opp
        df["_V217_PT_GAME_LISTED"]=listed
        df["_V217_PT_COMPONENT_AVAILABLE_COUNT"]=comp_n.astype(float)
        df["_V217_PT_FULL_FIVE"]=full_five

        # V2.18 full external index universe and aggregate consensus diagnostics.
        df["_V218_PTIDX_AVAILABLE_COUNT"]=idx_n
        df["_V218_PTIDX_MEAN_MARGIN_TEAM"]=idx_mean_margin
        df["_V218_PTIDX_MEAN_EDGE_POINTS"]=idx_mean_edge
        df["_V218_PTIDX_MEDIAN_MARGIN_TEAM"]=idx_median_margin
        df["_V218_PTIDX_MEDIAN_EDGE_POINTS"]=idx_median_edge
        df["_V218_PTIDX_STD"]=idx_std
        df["_V218_PTIDX_IQR"]=idx_iqr
        df["_V218_PTIDX_TEAM_AGREE_FRAC"]=idx_team_frac
        df["_V218_PTIDX_OPP_AGREE_FRAC"]=idx_opp_frac
        df["_V218_PTIDX_STRONG_TEAM_FRAC"]=idx_strong_team_frac
        df["_V218_PTIDX_STRONG_OPP_FRAC"]=idx_strong_opp_frac
        df["_V218_PTIDX_CLUSTER_COUNT"]=idx_cluster_n
        df["_V218_PTIDX_CLUSTER_MEDIAN_EDGE_POINTS"]=idx_cluster_median_edge
        df["_V218_PTIDX_CLUSTER_TEAM_AGREE_FRAC"]=idx_cluster_team_frac
        df["_V218_PTIDX_CLUSTER_OPP_AGREE_FRAC"]=idx_cluster_opp_frac

        # META-source-out consensus. These are the cleanest aggregate external
        # comparators to the frozen five-system META because none of META's
        # constituent source clusters are included.
        df["_V2182_PT_EXMETA_AVAILABLE_COUNT"]=exmeta_n
        df["_V2182_PT_EXMETA_MEAN_EDGE_POINTS"]=exmeta_mean_edge
        df["_V2182_PT_EXMETA_MEDIAN_EDGE_POINTS"]=exmeta_median_edge
        df["_V2182_PT_EXMETA_STD"]=exmeta_std
        df["_V2182_PT_EXMETA_IQR"]=exmeta_iqr
        df["_V2182_PT_EXMETA_TEAM_AGREE_FRAC"]=exmeta_team_frac
        df["_V2182_PT_EXMETA_OPP_AGREE_FRAC"]=exmeta_opp_frac
        df["_V2182_PT_EXMETA_STRONG_TEAM_FRAC"]=exmeta_strong_team_frac
        df["_V2182_PT_EXMETA_STRONG_OPP_FRAC"]=exmeta_strong_opp_frac
        df["_V2182_PT_EXMETA_CLUSTER_COUNT"]=exmeta_cluster_n
        df["_V2182_PT_EXMETA_CLUSTER_MEDIAN_EDGE_POINTS"]=exmeta_cluster_median_edge
        df["_V2182_PT_EXMETA_CLUSTER_TEAM_AGREE_FRAC"]=exmeta_cluster_team_frac
        df["_V2182_PT_EXMETA_CLUSTER_OPP_AGREE_FRAC"]=exmeta_cluster_opp_frac

        df["_V218_PT_TRACKER_AVG_EDGE_POINTS"]=pavg-market_margin
        df["_V218_PT_TRACKER_MEDIAN_EDGE_POINTS"]=pmed-market_margin

        for _k in PT_PUBLISHED_WEIGHTS:
            df[f"_V212_PT_{_k}_MARGIN_TEAM"]=comp[_k]
            df[f"_V212_PT_{_k}_EDGE_POINTS"]=comp_edges[_k]

    # Bulk-attach the expanded index matrix to avoid DataFrame fragmentation.
    _idx_extra={}
    for _k in _idx_keys:
        _slug=re.sub(r"[^A-Z0-9]+","_",str(_k).upper())
        _arr=idx_comp[_k]
        _idx_extra[f"_V218_PTIDX_{_slug}_MARGIN_TEAM"]=_arr
        _idx_extra[f"_V218_PTIDX_{_slug}_EDGE_POINTS"]=_arr-market_margin
    if _idx_extra:
        _idx_extra_df=pd.DataFrame(_idx_extra,index=g.index)
        g=pd.concat([g,_idx_extra_df],axis=1)
        m=pd.concat([m,_idx_extra_df.copy()],axis=1)

    # Core bridge exists only on miner frame; attach meta-vs-core state there.
    if "_V29_CORE_INCUMBENT_EDGE_POINTS" in m.columns:
        ce=pd.to_numeric(m["_V29_CORE_INCUMBENT_EDGE_POINTS"],errors="coerce").to_numpy(float)
        m["_V210_PT_META_MINUS_CORE_EDGE"]=meta_edge-ce

    cache["games"]=g; cache["miner_games"]=m
    cache["pt_index_catalog"]=[
        {"id":_k,"name":_pt_index_display_name(_k),"cluster":_pt_index_source_cluster(_k),
         "matched_nonnull":int(np.isfinite(idx_comp[_k]).sum())}
        for _k in _idx_keys
    ]
    try: setattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",cache)
    except Exception: pass

    no_pt=max(0,int(len(g)-matched))

    # V2.22: once team identity is resolved, diagnose every remaining source row
    # that still did not attach. This separates alias defects from pair/date defects.
    _internal_pair_keys={str(x) for x in key.astype(str) if str(x)}
    _internal_date_keys={
        f"{str(k)}|{str(d)}"
        for k,d in zip(key.astype(str),gd.astype(str))
        if str(k) and str(d)
    }
    _post_alias_unmatched=ex.loc[~ex["_pt_src_row_id"].isin(matched_source_ids)].copy()
    _residual_reason_counts={}
    _residual_detail=[]
    for _,_r in _post_alias_unmatched.iterrows():
        _pk=str(_r.get("pair_key") or "")
        _dt=str(_r.get("game_date") or "")
        _dk=f"{_pk}|{_dt}" if _pk and _dt else ""
        if _pk not in _internal_pair_keys:
            _reason="PAIR_NOT_IN_INTERNAL_CACHE"
        elif int(counts.get(_pk,0))>1 and not _dt:
            _reason="REMATCH_MISSING_SOURCE_DATE"
        elif int(counts.get(_pk,0))>1 and _dk not in _internal_date_keys:
            _reason="REMATCH_DATE_MISMATCH"
        elif int(counts.get(_pk,0))>1:
            _reason="REMATCH_AMBIGUOUS"
        else:
            _reason="OTHER_KEY_MISMATCH"
        _residual_reason_counts[_reason]=_residual_reason_counts.get(_reason,0)+1
        if len(_residual_detail)<50:
            _residual_detail.append({
                "season":int(_r["season"]) if pd.notna(_r.get("season")) else None,
                "home":str(_r.get("home_i") or _r.get("home_key") or ""),
                "away":str(_r.get("away_i") or _r.get("away_key") or ""),
                "date":_dt,
                "reason":_reason,
            })
    log_func(
        f"[NCAAF-PT-POST-ALIAS-UNMATCHED] rows={len(_post_alias_unmatched)} "
        f"reasons={json.dumps(_residual_reason_counts,sort_keys=True)} "
        f"examples={json.dumps(_residual_detail,sort_keys=True)} authority=0"
    )

    _src_seasons=sorted({int(x) for x in pd.to_numeric(ext_source.get("season"),errors="coerce").dropna().tolist()})
    _eligible_internal_mask=season.isin(_src_seasons)
    _eligible_internal=int(_eligible_internal_mask.sum())
    _source_match_cov=float(matched/max(source_rows,1))
    _eligible_internal_cov=float(matched/max(_eligible_internal,1))
    _all_cache_cov=float(matched/max(len(g),1))
    _alias_resolution_cov=float(alias_resolved_source_rows/max(source_rows,1))
    _coverage_gate="PASS" if _source_match_cov>=PT_SOURCE_MATCH_MIN_COVERAGE else "FAIL"
    _goal="PASS" if _source_match_cov>=PT_SOURCE_MATCH_GOAL_COVERAGE else "OPEN"
    _season_diag={}
    for _yr in _src_seasons:
        _src_n=int((pd.to_numeric(ext_source.get("season"),errors="coerce")==_yr).sum())
        _alias_n=int((pd.to_numeric(ex.get("season"),errors="coerce")==_yr).sum())
        _int_n=int((season==_yr).sum())
        _mat_n=int(matched_by_season.get(_yr,0))
        _season_diag[str(_yr)]={
            "source_rows":_src_n,
            "alias_resolved_source_rows":_alias_n,
            "matched_rows":_mat_n,
            "source_match_coverage":float(_mat_n/max(_src_n,1)),
            "internal_games":_int_n,
            "internal_coverage":float(_mat_n/max(_int_n,1)),
        }
        log_func(
            f"[NCAAF-PT-MATCH-SEASON] season={_yr} source_rows={_src_n} alias_resolved={_alias_n} matched={_mat_n} "
            f"source_match_coverage={_mat_n/max(_src_n,1):.3f} internal_games={_int_n} "
            f"internal_coverage={_mat_n/max(_int_n,1):.3f} authority=0"
        )

    diag={
        "status":"PASS",
        "coverage_gate":_coverage_gate,
        "coverage_goal":_goal,
        "coverage_target":PT_SOURCE_MATCH_MIN_COVERAGE,
        "coverage_goal_target":PT_SOURCE_MATCH_GOAL_COVERAGE,
        "matched_rows":int(matched),
        "matched_any_component_rows":int(matched_any_component),
        "matched_any_index_rows":int(matched_any_index),
        "external_index_count":int(len(_idx_keys)),
        "external_source_cluster_count":int(len(_cluster_names)),
        "full_five_matched_rows":int(full_five_matched),
        "partial_matched_rows":int(partial_matched),
        "no_pt_rows":int(no_pt),
        "total_rows":int(len(g)),
        "coverage":_all_cache_cov,
        "all_cache_coverage":_all_cache_cov,
        "source_rows":source_rows,
        "source_match_coverage":_source_match_cov,
        "alias_resolved_source_rows":alias_resolved_source_rows,
        "alias_resolution_coverage":_alias_resolution_cov,
        "unresolved_source_rows":unresolved_source_rows,
        "post_alias_unmatched_rows":int(len(_post_alias_unmatched)),
        "post_alias_unmatched_reasons":_residual_reason_counts,
        "eligible_internal_rows":_eligible_internal,
        "eligible_internal_coverage":_eligible_internal_cov,
        "season_match_diagnostics":_season_diag,
        "full_five_coverage":float(full_five_matched/max(len(g),1)),
        "mapped_external_teams":int(len(emap)),
        "alias_hint_count":int(len(alias_hints)),
        "unresolved_external_teams":unresolved[:50],
        "sparse_policy":"MISSING_PT_IS_NO_EXTERNAL_SIGNAL; PARTIAL_COMPONENTS_PRESERVED; META_REQUIRES_EXACT_5_OF_5",
        "authority":0,
    }
    log_func(
        f"[NCAAF-PT-MATCH] status=PASS matched_rows={matched} source_rows={source_rows} "
        f"source_match_coverage={_source_match_cov:.3f} coverage_gate={_coverage_gate} "
        f"target={PT_SOURCE_MATCH_MIN_COVERAGE:.2f} goal={PT_SOURCE_MATCH_GOAL_COVERAGE:.2f} "
        f"alias_resolved={alias_resolved_source_rows}/{source_rows} alias_resolution_coverage={_alias_resolution_cov:.3f} "
        f"post_alias_unmatched={len(_post_alias_unmatched)} "
        f"eligible_internal={_eligible_internal} eligible_internal_coverage={_eligible_internal_cov:.3f} "
        f"all_cache={len(g)} all_cache_coverage={_all_cache_cov:.3f} any_component={matched_any_component} "
        f"any_index={matched_any_index} indices={len(_idx_keys)} clusters={len(_cluster_names)} "
        f"partial={partial_matched} full_five={full_five_matched} mapped_teams={len(emap)} "
        f"unresolved_teams={len(unresolved)} alias_hints={len(alias_hints)} sparse=TRUE authority=0"
    )
    if _coverage_gate!="PASS":
        log_func(
            f"[NCAAF-PT-COVERAGE-GATE] status=FAIL source_match_coverage={_source_match_cov:.3f} "
            f"required={PT_SOURCE_MATCH_MIN_COVERAGE:.2f} unresolved_teams={len(unresolved)} "
            f"unresolved_source_rows={unresolved_source_rows} "
            f"action=FIX_TEAM_ALIASES_OR_MATCH_KEYS_BEFORE_JUDGING_PT_SIGNAL production_authority=0"
        )
    else:
        log_func(
            f"[NCAAF-PT-COVERAGE-GATE] status=PASS source_match_coverage={_source_match_cov:.3f} "
            f"required={PT_SOURCE_MATCH_MIN_COVERAGE:.2f} goal={PT_SOURCE_MATCH_GOAL_COVERAGE:.2f} "
            f"production_authority=0"
        )
    return diag

def _pt_research_metrics(g: pd.DataFrame) -> dict[str,Any]:
    meta=pd.to_numeric(g.get("_V210_PT_META_MARGIN_TEAM"),errors="coerce").to_numpy(float)
    actual=pd.to_numeric(g.get("Actual_Margin"),errors="coerce").to_numpy(float)
    spread=pd.to_numeric(g.get("Consensus_Open_Spread"),errors="coerce").to_numpy(float)
    market=pd.to_numeric(g.get("Market_Open_Margin"),errors="coerce").to_numpy(float) if "Market_Open_Margin" in g.columns else -spread
    season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    ishome=pd.to_numeric(g.get("Is_Home",pd.Series(np.nan,index=g.index)),errors="coerce").to_numpy(float)
    core_edge=pd.to_numeric(g.get("_V29_CORE_INCUMBENT_EDGE_POINTS",pd.Series(np.nan,index=g.index)),errors="coerce").to_numpy(float)
    core_margin=market+core_edge
    physical=(ishome==1) if np.isfinite(ishome).any() else np.ones(len(g),dtype=bool)
    rows=[]
    for yr in sorted({int(x) for x in season[np.isfinite(season)] if int(x)<=2025}):
        mk=physical&(season==yr)&np.isfinite(meta)&np.isfinite(actual)
        if not mk.any(): continue
        mm={"season":yr,"n":int(mk.sum()),"meta_mae":_mae(actual[mk],meta[mk]),"meta_rmse":_rmse(actual[mk],meta[mk]),"market_mae":_mae(actual[mk],market[mk]),"market_rmse":_rmse(actual[mk],market[mk])}
        ck=mk&np.isfinite(core_margin)
        if ck.any(): mm.update({"core_n":int(ck.sum()),"core_mae":_mae(actual[ck],core_margin[ck]),"core_rmse":_rmse(actual[ck],core_margin[ck]),"meta_minus_core_mae_gain":_mae(actual[ck],core_margin[ck])-_mae(actual[ck],meta[ck]),"meta_minus_core_rmse_gain":_rmse(actual[ck],core_margin[ck])-_rmse(actual[ck],meta[ck])})
        edge=meta-market; cover=actual+spread; sel=mk&np.isfinite(edge)&np.isfinite(cover)&(np.abs(edge)>3)&(~np.isclose(cover,0,atol=1e-9))
        if sel.any(): mm.update({"edge_gt3_n":int(sel.sum()),"edge_gt3_ats":float(np.mean(np.sign(edge[sel])*cover[sel]>0))})
        else: mm.update({"edge_gt3_n":0,"edge_gt3_ats":np.nan})
        rows.append(mm)
    pooled=physical&np.isfinite(meta)&np.isfinite(actual)&(season<=2025)
    pp={"n":int(pooled.sum()),"meta_mae":_mae(actual[pooled],meta[pooled]),"meta_rmse":_rmse(actual[pooled],meta[pooled]),"market_mae":_mae(actual[pooled],market[pooled]),"market_rmse":_rmse(actual[pooled],market[pooled])}
    cp=pooled&np.isfinite(core_margin)
    if cp.any(): pp.update({"core_n":int(cp.sum()),"core_mae":_mae(actual[cp],core_margin[cp]),"core_rmse":_rmse(actual[cp],core_margin[cp]),"meta_minus_core_mae_gain":_mae(actual[cp],core_margin[cp])-_mae(actual[cp],meta[cp]),"meta_minus_core_rmse_gain":_rmse(actual[cp],core_margin[cp])-_rmse(actual[cp],meta[cp])})
    return {"status":"PASS" if pooled.any() else "NO_MATCHED_ROWS","season_metrics":rows,"pooled":pp,"authority":0,"selection_influence":0}


# ---------------------------------------------------------------------------
# V2.19 Prediction Tracker incremental-information / residual challenger
# ---------------------------------------------------------------------------
PT_INCREMENTAL_FIT_SEASON = 2022
PT_INCREMENTAL_SELECTION_SEASON = 2023
PT_INCREMENTAL_VALIDATION_SEASONS = (2024, 2025)
PT_INCREMENTAL_RIDGE_ALPHAS = (1.0, 10.0, 50.0, 100.0)
PT_INCREMENTAL_BLEND_WEIGHTS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.0)


def _ptiv_empirical_prob(edge, residual_pool):
    e=np.asarray(edge,float); r=np.asarray(residual_pool,float); r=r[np.isfinite(r)]
    out=np.full(len(e),np.nan)
    if len(r)<50: return out
    rs=np.sort(r); ok=np.isfinite(e); ix=np.searchsorted(rs,-e[ok],side="right")
    out[ok]=(len(rs)-ix)/float(len(rs)); eps=0.5/(len(rs)+1.0); out[ok]=np.clip(out[ok],eps,1-eps)
    return out


def _ptiv_binary_metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.isfinite(y)&np.isfinite(p)
    if not m.any(): return {"n":0,"brier":None,"logloss":None}
    yy=y[m]; pp=np.clip(p[m],1e-6,1-1e-6)
    return {"n":int(len(yy)),"brier":float(np.mean((pp-yy)**2)),"logloss":float(-np.mean(yy*np.log(pp)+(1-yy)*np.log(1-pp)))}


def _ptiv_metric_row(actual,pred,market,residual_pool):
    actual=np.asarray(actual,float); pred=np.asarray(pred,float); market=np.asarray(market,float)
    ok=np.isfinite(actual)&np.isfinite(pred)&np.isfinite(market)
    if not ok.any(): return {"n":0,"mae":None,"rmse":None,"brier":None,"logloss":None,"direction_hit":None}
    edge=pred-market; settle=actual-market; nonpush=ok&~np.isclose(settle,0.0,atol=1e-9)
    y=np.where(settle>0,1.0,np.where(settle<0,0.0,np.nan)); prob=_ptiv_empirical_prob(edge,residual_pool); bm=_ptiv_binary_metrics(y[nonpush],prob[nonpush])
    dm=nonpush&~np.isclose(edge,0.0,atol=1e-9); hit=float(np.mean(np.sign(edge[dm])==np.sign(settle[dm]))) if dm.any() else np.nan
    return {"n":int(ok.sum()),"mae":_mae(actual[ok],pred[ok]),"rmse":_rmse(actual[ok],pred[ok]),"brier":bm.get("brier"),"logloss":bm.get("logloss"),"direction_hit":hit}


def _ptiv_bootstrap_gain(y,base,pred,reps=800,seed=20261007):
    y=np.asarray(y,float); b=np.asarray(base,float); p=np.asarray(pred,float); m=np.isfinite(y)&np.isfinite(b)&np.isfinite(p)
    diff=np.abs(y[m]-b[m])-np.abs(y[m]-p[m])
    if len(diff)<30: return {"n":int(len(diff)),"mean_mae_gain":None,"ci95":[None,None]}
    rng=np.random.default_rng(seed); n=len(diff); vals=np.empty(reps,float)
    for i in range(reps): vals[i]=float(np.mean(diff[rng.integers(0,n,n)]))
    return {"n":n,"mean_mae_gain":float(np.mean(diff)),"ci95":[float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))]}


def _ptiv_fit_ridge(Xtr,ytr,Xsc,alpha):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),("ridge",Ridge(alpha=float(alpha)))])
    pipe.fit(Xtr,ytr); return np.asarray(pipe.predict(Xsc),float),pipe


def _ptiv_select_alpha(X,resid,seasons,cols):
    tr=(seasons==PT_INCREMENTAL_FIT_SEASON)&np.isfinite(resid); va=(seasons==PT_INCREMENTAL_SELECTION_SEASON)&np.isfinite(resid)
    if tr.sum()<80 or va.sum()<40: return 50.0,{"status":"FALLBACK_FIXED_ALPHA","train_n":int(tr.sum()),"selection_n":int(va.sum())}
    grid=[]
    for a in PT_INCREMENTAL_RIDGE_ALPHAS:
        try:
            q,_=_ptiv_fit_ridge(X.loc[tr,cols],resid[tr],X.loc[va,cols],a); score=float(np.mean(np.abs(resid[va]-q))); grid.append({"alpha":float(a),"selection_residual_mae":score})
        except Exception as exc: grid.append({"alpha":float(a),"selection_residual_mae":None,"error":f"{type(exc).__name__}:{exc}"})
    good=[z for z in grid if z.get("selection_residual_mae") is not None]
    if not good: return 50.0,{"status":"FALLBACK_FIXED_ALPHA","grid":grid}
    best=min(good,key=lambda z:(z["selection_residual_mae"],z["alpha"])); return float(best["alpha"]),{"status":"DISCOVERY_FORWARD_SELECTED","grid":grid,"selected_alpha":float(best["alpha"])}


def _ptiv_production_v1_oof(games,seasons,dashboard_module,production_module,log_func=print):
    if production_module is None: return np.full(len(games),np.nan),{"status":"PRODUCTION_MODULE_UNAVAILABLE"}
    sp=list(getattr(production_module,"PROD_SPREAD_FEATURES",()) or ()); tt=list(getattr(production_module,"PROD_TOTAL_FEATURES",()) or ())
    if not sp or any(c not in games.columns for c in sp): return np.full(len(games),np.nan),{"status":"FIXED_FEATURES_MISSING","missing":[c for c in sp if c not in games.columns]}
    oof=np.full(len(games),np.nan); by={}; lw=float(getattr(production_module,"PROD_LINEAR_WEIGHT",0.75))
    for sy in sorted({int(x) for x in seasons[np.isfinite(seasons)] if int(x)<=2025}):
        tr=np.isfinite(seasons)&(seasons<float(sy)); va=np.isfinite(seasons)&(seasons==float(sy))
        if tr.sum()<500 or va.sum()<50: continue
        try:
            mm,_=dashboard_module._ncaaf_stat_fit_models_for_rows(games,sp,tt,tr,target_mode="MARKET_ERROR_RESIDUAL")
            edge=dashboard_module._ncaaf_stat_blend_predict(mm,games.loc[va,sp],lw)
            market=pd.to_numeric(games.loc[va,"Market_Open_Margin"],errors="coerce").to_numpy(float)
            oof[np.where(va)[0]]=np.where(np.isfinite(market),market+edge,np.nan); by[str(sy)]={"train_n":int(tr.sum()),"score_n":int(va.sum())}
        except Exception as exc: by[str(sy)]={"status":"ERROR","error":f"{type(exc).__name__}:{exc}"}
    return oof,{"status":"PASS" if np.isfinite(oof).any() else "NO_OOF","feature_cols":sp,"linear_weight":lw,"by_season":by,"contract":"EXACT_NCAAF_PRODUCTION_V1_FIXED_SPREAD_FEATURES_SEASON_FORWARD"}


def run_pt_incremental_value_research(*,games,miner_games,seasons,dashboard_module,production_module=None,log_func=print):
    """Forecast-encompassing test: does PT add information after frozen Production V1?

    BASE_CALIBRATION is an explicit no-PT residual control. PT is therefore not
    credited for gains that can be obtained by simply recalibrating Production V1.
    Pathi/Big Al remain in the separate System Miner for interaction discovery and
    are used here only for attribution slices, not as residual-model predictors.
    """
    if games is None or getattr(games,"empty",True) or miner_games is None or getattr(miner_games,"empty",True):
        return {"status":"CACHE_UNAVAILABLE","production_authority":0}
    if len(games)!=len(miner_games) or len(seasons)!=len(games): raise RuntimeError("NCAAF_PT_INCREMENTAL_ALIGNMENT_FAIL")
    if np.nanmax(seasons)>2025: raise RuntimeError("NCAAF_PT_INCREMENTAL_2026_LEAK")
    base,base_diag=_ptiv_production_v1_oof(games,seasons,dashboard_module,production_module,log_func=log_func)
    mg=miner_games.reset_index(drop=True); g=games.reset_index(drop=True)
    edge_col=next((c for c in ("_V2182_PT_EXMETA_CLUSTER_MEDIAN_EDGE_POINTS","_V218_PTIDX_CLUSTER_MEDIAN_EDGE_POINTS") if c in mg.columns),None)
    count_col=next((c for c in ("_V2182_PT_EXMETA_CLUSTER_COUNT","_V218_PTIDX_CLUSTER_COUNT") if c in mg.columns),None)
    disp_col=next((c for c in ("_V2182_PT_EXMETA_STD","_V218_PTIDX_STD") if c in mg.columns),None)
    agree_col=next((c for c in ("_V2182_PT_EXMETA_CLUSTER_TEAM_AGREE_FRAC","_V218_PTIDX_CLUSTER_TEAM_AGREE_FRAC") if c in mg.columns),None)
    if not edge_col or not count_col:
        return {"status":"PT_CONSENSUS_FIELDS_UNAVAILABLE","base_oof":base_diag,"production_authority":0}
    actual=pd.to_numeric(g.get("Actual_Margin"),errors="coerce").to_numpy(float)
    market=pd.to_numeric(g.get("Market_Open_Margin"),errors="coerce").to_numpy(float) if "Market_Open_Margin" in g.columns else -pd.to_numeric(g.get("Consensus_Open_Spread"),errors="coerce").to_numpy(float)
    pt_edge=pd.to_numeric(mg.get(edge_col),errors="coerce").to_numpy(float); pt_fair=market+pt_edge; base_edge=base-market; gap=pt_fair-base
    count=pd.to_numeric(mg.get(count_col),errors="coerce").to_numpy(float); disp=pd.to_numeric(mg.get(disp_col,pd.Series(np.nan,index=mg.index)),errors="coerce").to_numpy(float); agree=pd.to_numeric(mg.get(agree_col,pd.Series(np.nan,index=mg.index)),errors="coerce").to_numpy(float)
    same=np.where(np.isfinite(pt_edge)&np.isfinite(base_edge)&(np.abs(pt_edge)>1e-9)&(np.abs(base_edge)>1e-9),(np.sign(pt_edge)==np.sign(base_edge)).astype(float),0.0)
    # Expert systems are attribution-only here. Their interaction mining remains in
    # the independent external PT Miner lane, preventing false PT incremental credit.
    pathi_cols=[c for c in mg.columns if str(c).startswith("Pathi_")]; bigal_cols=[c for c in mg.columns if str(c).startswith("BigAl_")]
    pathi=np.zeros(len(mg),float); bigal=np.zeros(len(mg),float)
    if pathi_cols: pathi=np.nansum(np.column_stack([pd.to_numeric(mg[c],errors="coerce").fillna(0).to_numpy(float) for c in pathi_cols]),axis=1)
    if bigal_cols: bigal=np.nansum(np.column_stack([pd.to_numeric(mg[c],errors="coerce").fillna(0).to_numpy(float) for c in bigal_cols]),axis=1)
    X=pd.DataFrame({
        "base_edge":base_edge,"pt_edge":pt_edge,"pt_dispersion":disp,"pt_cluster_count":count,"pt_positive_frac":agree,
        "abs_pt_edge":np.abs(pt_edge),"abs_base_edge":np.abs(base_edge),"abs_gap":np.abs(gap),"same_side":same,"edge_product":pt_edge*base_edge,
    })
    ishome=pd.to_numeric(g.get("Is_Home",pd.Series(np.nan,index=g.index)),errors="coerce").to_numpy(float); physical=(ishome==1) if np.isfinite(ishome).any() else np.ones(len(g),bool)
    valid=physical&np.isfinite(actual)&np.isfinite(base)&np.isfinite(pt_fair)&(count>=PT_EXTERNAL_CONSENSUS_MIN_CLUSTERS)
    fit=valid&(seasons==PT_INCREMENTAL_FIT_SEASON); sel=valid&(seasons==PT_INCREMENTAL_SELECTION_SEASON); val=valid&np.isin(seasons,PT_INCREMENTAL_VALIDATION_SEASONS)
    if fit.sum()<80 or sel.sum()<40 or val.sum()<100:
        return {"status":"INSUFFICIENT_MATCHED_ROWS","fit_n":int(fit.sum()),"selection_n":int(sel.sum()),"validation_n":int(val.sum()),"base_oof":base_diag,"production_authority":0}
    resid=actual-base
    control=["base_edge"]
    basic=control+["pt_edge","pt_dispersion","pt_cluster_count","pt_positive_frac"]
    conditional=basic+["abs_pt_edge","abs_base_edge","abs_gap","same_side","edge_product"]
    a0,a0diag=_ptiv_select_alpha(X,resid,seasons,control); ab,abdiag=_ptiv_select_alpha(X,resid,seasons,basic); ac,acdiag=_ptiv_select_alpha(X,resid,seasons,conditional)
    blends=[]
    for w in PT_INCREMENTAL_BLEND_WEIGHTS:
        p=base+float(w)*(pt_fair-base); blends.append({"weight":float(w),"selection_mae":_mae(actual[sel],p[sel])})
    bw=float(min(blends,key=lambda z:(z["selection_mae"],z["weight"]))["weight"]); pblend=base+bw*(pt_fair-base)
    fit_sel=valid&np.isin(seasons,(PT_INCREMENTAL_FIT_SEASON,PT_INCREMENTAL_SELECTION_SEASON)); pctl=base.copy(); pb=base.copy(); pc=base.copy()
    q0,_=_ptiv_fit_ridge(X.loc[fit_sel,control],resid[fit_sel],X.loc[valid,control],a0); pctl[valid]=base[valid]+q0
    qb,_=_ptiv_fit_ridge(X.loc[fit_sel,basic],resid[fit_sel],X.loc[valid,basic],ab); pb[valid]=base[valid]+qb
    qc,_=_ptiv_fit_ridge(X.loc[fit_sel,conditional],resid[fit_sel],X.loc[valid,conditional],ac); pc[valid]=base[valid]+qc
    def sel_mae(cols,a):
        q,_=_ptiv_fit_ridge(X.loc[fit,cols],resid[fit],X.loc[sel,cols],a); return _mae(actual[sel],base[sel]+q)
    scores={"BASELINE":_mae(actual[sel],base[sel]),"BASE_CALIBRATION":sel_mae(control,a0),"SIMPLE_BLEND":_mae(actual[sel],pblend[sel]),"PT_RESIDUAL":sel_mae(basic,ab),"PT_CONDITIONAL_RESIDUAL":sel_mae(conditional,ac)}
    no_pt_best=min(("BASELINE","BASE_CALIBRATION"),key=lambda k:scores[k]); pt_candidates=["PT_RESIDUAL","PT_CONDITIONAL_RESIDUAL"]
    if bw>0: pt_candidates.append("SIMPLE_BLEND")
    pt_best=min(pt_candidates,key=lambda k:scores[k])
    pt_selected=bool(scores[pt_best]+0.05<scores["BASELINE"] and scores[pt_best]+0.025<scores[no_pt_best])
    selected=pt_best if pt_selected else ("BASE_CALIBRATION_NO_PT" if no_pt_best=="BASE_CALIBRATION" and scores["BASE_CALIBRATION"]+0.025<scores["BASELINE"] else "BASELINE_NO_PT")
    preds={"BASELINE":base,"BASE_CALIBRATION":pctl,"PT_STANDALONE":pt_fair,"SIMPLE_BLEND":pblend,"PT_RESIDUAL":pb,"PT_CONDITIONAL_RESIDUAL":pc}; metrics={}; by={}
    for name,p in preds.items():
        pool=(actual-p)[fit_sel&np.isfinite(actual)&np.isfinite(p)]; metrics[name]={"selection":_ptiv_metric_row(actual[sel],p[sel],market[sel],pool),"validation_2024_2025":_ptiv_metric_row(actual[val],p[val],market[val],pool)}; by[name]={str(sy):_ptiv_metric_row(actual[valid&(seasons==sy)],p[valid&(seasons==sy)],market[valid&(seasons==sy)],pool) for sy in PT_INCREMENTAL_VALIDATION_SEASONS}
    selected_key="BASE_CALIBRATION" if selected=="BASE_CALIBRATION_NO_PT" else ("BASELINE" if selected=="BASELINE_NO_PT" else selected); sp=preds[selected_key]
    control_key=min(("BASELINE","BASE_CALIBRATION"),key=lambda k:metrics[k]["validation_2024_2025"].get("mae") if metrics[k]["validation_2024_2025"].get("mae") is not None else 1e9); cp=preds[control_key]
    boot_frozen=_ptiv_bootstrap_gain(actual[val],base[val],sp[val]); boot_control=_ptiv_bootstrap_gain(actual[val],cp[val],sp[val])
    pos_frozen=sum(1 for sy in PT_INCREMENTAL_VALIDATION_SEASONS if by["BASELINE"][str(sy)].get("mae") is not None and by[selected_key][str(sy)].get("mae") is not None and by["BASELINE"][str(sy)]["mae"]-by[selected_key][str(sy)]["mae"]>0)
    pos_control=sum(1 for sy in PT_INCREMENTAL_VALIDATION_SEASONS if by[control_key][str(sy)].get("mae") is not None and by[selected_key][str(sy)].get("mae") is not None and by[control_key][str(sy)]["mae"]-by[selected_key][str(sy)]["mae"]>0)
    bm=metrics["BASELINE"]["validation_2024_2025"]; cm=metrics[control_key]["validation_2024_2025"]; sm=metrics[selected_key]["validation_2024_2025"]
    selected_uses_pt=selected in {"SIMPLE_BLEND","PT_RESIDUAL","PT_CONDITIONAL_RESIDUAL"}
    strict=bool(selected_uses_pt and bm.get("mae") is not None and cm.get("mae") is not None and sm.get("mae") is not None and bm["mae"]-sm["mae"]>=0.05 and cm["mae"]-sm["mae"]>=0.025 and bm["rmse"]-sm["rmse"]>=0 and cm["rmse"]-sm["rmse"]>=0 and (bm.get("brier") is None or sm.get("brier") is None or sm["brier"]<=bm["brier"]) and (cm.get("brier") is None or sm.get("brier") is None or sm["brier"]<=cm["brier"]) and pos_frozen==len(PT_INCREMENTAL_VALIDATION_SEASONS) and pos_control==len(PT_INCREMENTAL_VALIDATION_SEASONS) and boot_frozen.get("ci95",[None])[0] is not None and boot_frozen["ci95"][0]>0 and boot_control.get("ci95",[None])[0] is not None and boot_control["ci95"][0]>0)
    disp_cut=float(np.nanmedian(disp[fit_sel])); regimes={"CORE_PT_SAME_SIDE":same>0.5,"CORE_PT_CONFLICT":(same<0.5)&(np.abs(base_edge)>1)&(np.abs(pt_edge)>1),"PT_EDGE_3_PLUS":np.abs(pt_edge)>=3,"CORE_PT_GAP_4_PLUS":np.abs(gap)>=4,"PT_HIGH_DISPERSION":disp>=disp_cut,"PATHI_ACTIVE":pathi>=1,"BIGAL_ACTIVE":bigal>=1}
    attr={}; rp=pc
    for name,mask in regimes.items():
        mm=val&np.asarray(mask,bool); bmae=_mae(actual[mm],base[mm]); c0=_mae(actual[mm],pctl[mm]); cmae=_mae(actual[mm],rp[mm]); attr[name]={"n":int(mm.sum()),"baseline_mae":bmae,"base_calibration_mae":c0,"conditional_pt_mae":cmae,"pt_gain_vs_frozen":(bmae-cmae if np.isfinite(bmae) and np.isfinite(cmae) else None),"pt_gain_vs_calibration":(c0-cmae if np.isfinite(c0) and np.isfinite(cmae) else None)}
    out={"status":"PASS","benchmark":"NCAAF_PRODUCTION_V1_FIXED_SPREAD_BACKBONE_SEASON_FORWARD","fit_season":PT_INCREMENTAL_FIT_SEASON,"selection_season":PT_INCREMENTAL_SELECTION_SEASON,"validation_seasons":list(PT_INCREMENTAL_VALIDATION_SEASONS),"fit_n":int(fit.sum()),"selection_n":int(sel.sum()),"validation_n":int(val.sum()),"pt_edge_field":edge_col,"pt_cluster_count_field":count_col,"base_oof":base_diag,"forecast_encompassing_control":"BASE_CALIBRATION_NO_PT","blend_selection_grid":blends,"selected_blend_weight":bw,"ridge_control":a0diag,"ridge_basic":abdiag,"ridge_conditional":acdiag,"selection_scores_mae":scores,"selection_best_no_pt":no_pt_best,"selection_best_pt":pt_best,"discovery_selected_challenger":selected,"metrics":metrics,"validation_by_season":by,"selected_validation_mae_bootstrap_gain_vs_frozen":boot_frozen,"selected_validation_mae_bootstrap_gain_vs_no_pt_control":boot_control,"positive_validation_seasons_vs_frozen":int(pos_frozen),"positive_validation_seasons_vs_no_pt_control":int(pos_control),"validation_best_no_pt_control":control_key,"strict_incremental_signal":strict,"conditional_attribution":attr,"feature_contract":{"control":control,"basic":basic,"conditional":conditional,"expert_systems":"ATTRIBUTION_ONLY_HERE__INTERACTIONS_TESTED_IN_SEPARATE_PT_EXTERNAL_MINER"},"totals_status":"NO_NCAAF_PT_TOTAL_FEED","production_authority":0,"automatic_promotion":False,"year_2026_queried":False}
    log_func("[NCAAF-PT-INCREMENTAL] "+json.dumps({"status":out["status"],"selected":selected,"strict_incremental_signal":strict,"fit_n":out["fit_n"],"selection_n":out["selection_n"],"validation_n":out["validation_n"],"baseline_validation_mae":bm.get("mae"),"no_pt_control":control_key,"no_pt_control_validation_mae":cm.get("mae"),"selected_validation_mae":sm.get("mae"),"mae_gain_vs_frozen":None if bm.get("mae") is None or sm.get("mae") is None else bm["mae"]-sm["mae"],"mae_gain_vs_no_pt_control":None if cm.get("mae") is None or sm.get("mae") is None else cm["mae"]-sm["mae"],"bootstrap_vs_frozen":boot_frozen.get("ci95"),"bootstrap_vs_control":boot_control.get("ci95"),"production_authority":0,"year_2026_queried":False},sort_keys=True,default=str))
    return out

def refresh_prediction_tracker_external(*, dashboard_module=None, storage_client=None, bucket_name="sharp-models", include_history=True, include_current=True, force=False, log_func=print) -> dict[str,Any]:
    """Read validated Prediction Tracker GCS artifacts and attach research fields.

    V2.17 treats Prediction Tracker as sparse optional external intelligence. Games
    absent from the source remain in the NCAAF model with no external signal. When a
    listed game has only some of the five benchmark systems, available components are
    preserved but META_MARGIN remains missing; the published five-system META is never
    renormalized or imputed. Failure is non-fatal and always fail-closed: no external
    field can mutate Production V1 or grant Bet Authority.
    """
    try:
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        # Reuse already attached history within the same process unless forced.
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{}) if dashboard_module is not None else {}
        prior=(cache or {}).get("prediction_tracker_external") if isinstance(cache,dict) else None
        if include_history and (not include_current) and not force and isinstance(prior,dict) and prior.get("status")=="PASS" and prior.get("history_attached"):
            return prior
        frames=[]; diags=[]
        feeder_manifest=_pt_load_feeder_manifest(storage_client,bucket_name)
        if feeder_manifest:
            log_func(f"[NCAAF-PT-FEEDER-MANIFEST] status=READY updated_utc={feeder_manifest.get('updated_utc')} host={feeder_manifest.get('host','')} source_count={len(feeder_manifest.get('sources') or {})} authority=0")
        else:
            log_func(f"[NCAAF-PT-FEEDER-MANIFEST] status=MISSING path=gs://{bucket_name}/{PT_FEEDER_MANIFEST_BLOB} authority=0")
        current_frame=pd.DataFrame(); current_diag=None; current_archive=pd.DataFrame(); current_live=pd.DataFrame(); current_merge={"status":"NOT_RUN","authority":0}
        if include_current:
            # Fetch the named live table first.  When the live CSV is also
            # available this creates/refreshes the value-validated cryptic
            # header manifest BEFORE any archive CSV is parsed.
            current_live,current_live_diag=_pt_load_live_current(storage_client=storage_client,bucket_name=bucket_name,log_func=log_func)
            current_archive,current_archive_diag=_pt_load_season(PT_CURRENT_SEASON,storage_client=storage_client,bucket_name=bucket_name,force_web=False,log_func=log_func)
            if not current_archive.empty:
                current_archive=current_archive.copy(); current_archive["source_kind"]="ARCHIVE_SEASON_TO_DATE"; current_archive["source_priority"]=1
                try: storage_client.bucket(bucket_name).blob(PT_CURRENT_ARCHIVE_BLOB).upload_from_string(current_archive.to_csv(index=False).encode(),content_type="text/csv")
                except Exception: pass
            current_frame,current_merge=_pt_merge_current_season(current_archive,current_live,log_func=log_func)
            current_diag={"status":current_merge.get("status"),"season":PT_CURRENT_SEASON,"archive":current_archive_diag,"live":current_live_diag,"merge":current_merge,"authority":0}
            diags.extend([current_archive_diag,current_live_diag,current_diag])
            if not current_frame.empty:
                try:
                    storage_client.bucket(bucket_name).blob(PT_CURRENT_BLOB).upload_from_string(current_frame.to_csv(index=False).encode(),content_type="text/csv")
                    _cc=pd.to_numeric(current_frame.get("meta_system_count"),errors="coerce").fillna(0) if "meta_system_count" in current_frame.columns else pd.Series(0,index=current_frame.index,dtype=float)
                    _lcc=pd.to_numeric(current_live.get("meta_system_count"),errors="coerce").fillna(0) if isinstance(current_live,pd.DataFrame) and "meta_system_count" in current_live.columns else pd.Series(dtype=float)
                    meta={"season":PT_CURRENT_SEASON,"updated_utc":_now(),"rows":len(current_frame),"archive_rows":len(current_archive),"live_rows":len(current_live),
                          "any_component_rows":int((_cc>0).sum()),"partial_component_rows":int(((_cc>0)&(_cc<5)).sum()),
                          "full_five_rows":int((_cc==5).sum()),"live_any_component_rows":int((_lcc>0).sum()) if len(_lcc) else 0,
                          "live_partial_component_rows":int(((_lcc>0)&(_lcc<5)).sum()) if len(_lcc) else 0,
                          "live_full_five_rows":int((_lcc==5).sum()) if len(_lcc) else 0,
                          "external_consensus_ready_rows":int(pd.to_numeric(current_frame.get("external_consensus_home_margin"),errors="coerce").notna().sum()) if "external_consensus_home_margin" in current_frame.columns else 0,
                          "live_external_consensus_ready_rows":int(pd.to_numeric(current_live.get("external_consensus_home_margin"),errors="coerce").notna().sum()) if isinstance(current_live,pd.DataFrame) and "external_consensus_home_margin" in current_live.columns else 0,
                          "external_consensus_contract":PT_EXTERNAL_CONSENSUS_CONTRACT,
                          "source":"Prediction Tracker via manual validated GCS upload","archive_url":PT_ARCHIVE_URL.format(season=PT_CURRENT_SEASON),"live_csv_url":PT_LIVE_CSV_URL,"live_page_url":PT_LIVE_PAGE_URL,"header_manifest_gcs":f"gs://{bucket_name}/{PT_HEADER_MANIFEST_BLOB}","live_page_raw_gcs":f"gs://{bucket_name}/{PT_LIVE_HTML_RAW_BLOB}","relay_prefix":PT_RELAY_PREFIX,"sparse_source":True,"authority":0}
                    storage_client.bucket(bucket_name).blob(PT_CURRENT_META_BLOB).upload_from_string(json.dumps(meta,sort_keys=True).encode(),content_type="application/json")
                except Exception: pass
        if include_history:
            # Archives parse only after the live name contract has had a chance
            # to refresh the verified cryptic-header manifest.
            for sy in PT_HISTORY_SEASONS:
                f,d=_pt_load_season(sy,storage_client=storage_client,bucket_name=bucket_name,force_web=False,log_func=log_func); diags.append(d)
                if not f.empty: frames.append(f)
        match={"status":"NOT_ATTACHED","authority":0}; metrics={"status":"NOT_RUN","authority":0}
        if include_history and dashboard_module is not None and frames:
            hist=pd.concat(frames,ignore_index=True,sort=False)
            match=_pt_attach_history_to_cache(dashboard_module,hist,log_func=log_func)
            c=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}; gg=c.get("miner_games")
            if isinstance(gg,pd.DataFrame) and not gg.empty: metrics=_pt_research_metrics(gg)
        _cc=pd.to_numeric(current_frame.get("meta_system_count"),errors="coerce").fillna(0) if isinstance(current_frame,pd.DataFrame) and "meta_system_count" in current_frame.columns else pd.Series(dtype=float)
        _lcc=pd.to_numeric(current_live.get("meta_system_count"),errors="coerce").fillna(0) if isinstance(current_live,pd.DataFrame) and "meta_system_count" in current_live.columns else pd.Series(dtype=float)
        result={"status":"PASS" if any((d or {}).get('status')=='PASS' for d in diags) else "UNAVAILABLE","source":"THE_PREDICTION_TRACKER","history_attached":bool(include_history and match.get('status')=='PASS'),"season_diagnostics":diags,"match":match,"metrics":metrics,
                "coverage_policy":"SOURCE_MATCH_TARGET>=90%; GOAL>=99%; SPARSE_SOURCE_MAY_NOT_PUBLISH_ALL_INTERNAL_GAMES; MISSING_GAME=NO_EXTERNAL_SIGNAL; PARTIAL_COMPONENTS_PRESERVED; META=EXACT_5_OF_5_ONLY",
                "current":{"rows":int(len(current_frame)),"archive_rows":int(len(current_archive)),"live_rows":int(len(current_live)),
                           "any_component_rows":int((_cc>0).sum()) if len(_cc) else 0,"partial_component_rows":int(((_cc>0)&(_cc<5)).sum()) if len(_cc) else 0,
                           "full_five_rows":int((_cc==5).sum()) if len(_cc) else 0,
                           "live_any_component_rows":int((_lcc>0).sum()) if len(_lcc) else 0,"live_partial_component_rows":int(((_lcc>0)&(_lcc<5)).sum()) if len(_lcc) else 0,
                           "live_full_five_rows":int((_lcc==5).sum()) if len(_lcc) else 0,
                           "external_consensus_ready_rows":int(pd.to_numeric(current_frame.get("external_consensus_home_margin"),errors="coerce").notna().sum()) if "external_consensus_home_margin" in current_frame.columns else 0,
                           "live_external_consensus_ready_rows":int(pd.to_numeric(current_live.get("external_consensus_home_margin"),errors="coerce").notna().sum()) if isinstance(current_live,pd.DataFrame) and "external_consensus_home_margin" in current_live.columns else 0,
                           "external_consensus_contract":PT_EXTERNAL_CONSENSUS_CONTRACT,
                           "merge":current_merge,"gcs":f"gs://{bucket_name}/{PT_CURRENT_BLOB}","archive_gcs":f"gs://{bucket_name}/{PT_CURRENT_ARCHIVE_BLOB}","live_gcs":f"gs://{bucket_name}/{PT_CURRENT_LIVE_BLOB}"},
                "published_weights":dict(PT_PUBLISHED_WEIGHTS),"feeder_manifest":feeder_manifest,"production_authority":0,"bet_authority_vote":False,"automatic_promotion":False,"selection_influence":0}
        if dashboard_module is not None:
            c=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}; c["prediction_tracker_external"]=result
            try: setattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",c)
            except Exception: pass
        log_func(
            f"[NCAAF-PT-CONTRACT] status={result['status']} history_attached={result['history_attached']} "
            f"current_rows={result['current']['rows']} archive_rows={result['current'].get('archive_rows',0)} live_rows={result['current'].get('live_rows',0)} "
            f"current_partial={result['current'].get('partial_component_rows',0)} current_full_five={result['current'].get('full_five_rows',0)} "
            f"live_partial={result['current'].get('live_partial_component_rows',0)} live_full_five={result['current'].get('live_full_five_rows',0)} "
            f"external_consensus_ready={result['current'].get('external_consensus_ready_rows',0)} "
            f"live_external_consensus_ready={result['current'].get('live_external_consensus_ready_rows',0)} "
            f"matched_rows={match.get('matched_rows',0)} source_rows={match.get('source_rows',0)} "
            f"source_match_coverage={match.get('source_match_coverage',0):.3f} coverage_gate={match.get('coverage_gate','NA')} "
            f"matched_partial={match.get('partial_matched_rows',0)} matched_full_five={match.get('full_five_matched_rows',0)} "
            f"missing_games_expected=TRUE sparse=TRUE production_authority=0 bet_authority_vote=FALSE"
        )
        return result
    except Exception as exc:
        log_func(f"[NCAAF-PT-FAIL] {type(exc).__name__}: {exc} authority=0 fail_closed=TRUE")
        return {"status":"FAILED","error":f"{type(exc).__name__}:{exc}","production_authority":0,"bet_authority_vote":False,"selection_influence":0}


# ---------------------------------------------------------------------------
# Orthogonal STAT research
# ---------------------------------------------------------------------------
STAT_FAMILY_TOKENS: dict[str, tuple[str,...]] = {
    "OPPONENT_ADJUSTED": ("opp_","opponent","sos","strength_of_schedule","adj_"),
    "RUN_PASS_MATCHUP": ("rush","rushing","pass","passing","yards_per_rush","yards_per_pass","ypa","ypc"),
    "PACE_EFFICIENCY": ("pace","plays_per","seconds_per_play","efficiency","success_rate","ppp","points_per_play","yards_per_play"),
    "TURNOVER_REGRESSION": ("turnover","giveaway","takeaway","interception","fumble","luck"),
    "EXPLOSIVENESS": ("explosive","big_play","20_plus","30_plus","40_plus","iso_ppp"),
    "FINISHING_DRIVES": ("finishing","points_per_trip","scoring_opportunity","inside_40","drive_eff","points_per_drive"),
    "RED_ZONE": ("red_zone","redzone","rz_"),
    "THIRD_FOURTH_DOWN": ("third_down","3rd_down","fourth_down","4th_down"),
    "DISCIPLINE": ("penalty","penalties","flag_","discipline"),
    "NONOFFENSIVE_SCORING": ("defensive_td","special_teams_td","nonoffensive","non_offensive","return_td"),
    "SPECIAL_TEAMS": ("special_team","punt","kickoff","field_goal","fg_","return_yards"),
    "PRESSURE_SACKS": ("sack","pressure","havoc","tfl","tackle_for_loss"),
    "QUARTER_HALF_PROFILE": ("q1_","q2_","q3_","q4_","first_half","second_half","1h_","2h_"),
    "CLOSE_GAME_STATE": ("close_game","one_score","late_game","clutch","garbage"),
    "VENUE_FORM": ("home_","away_","road_","venue","neutral"),
    "SCHEDULE_SEQUENCE": ("rest","days_since","bye","travel","sequence","short_week","lookahead","sandwich"),
    "VOLATILITY_TREND": ("volatility","std","variance","consistency","trend","rolling","last3","last5","last10"),
    "TEAM_IDENTITY": ("team_game","team_ats","team_su","team_fav","team_dog","role_price","team_history"),
    "CONFERENCE_IDENTITY": ("conference","conf_","division","rivalry"),
    "H2H_HISTORY": ("h2h","revenge","last_matchup","meetings_since"),
    "MARKET_MICROSTRUCTURE": ("line_move","sharp_","soft_","direction_changes","current_vs_best","current_vs_worst","key_cross","impliedprob","price_zscore","market_leader","limit"),
}

EXCLUDE_STAT_TOKENS=("actual_","final_","cover_result","hit_bool","scored","winner","result_","postgame","post_game")


def _classify_feature_families(cols: Iterable[str]) -> dict[str,list[str]]:
    out={k:[] for k in STAT_FAMILY_TOKENS}
    for c in cols:
        lc=str(c).lower()
        if any(t in lc for t in EXCLUDE_STAT_TOKENS): continue
        for fam,toks in STAT_FAMILY_TOKENS.items():
            if any(t in lc for t in toks): out[fam].append(str(c))
    # Deduplicate and cap gigantic broad families by deterministic name order.
    return {k:sorted(set(v))[:32] for k,v in out.items() if len(set(v))>=2}


def _fit_ridge_predict(train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe=Pipeline([
        ("impute",SimpleImputer(strategy="median", add_indicator=True)),
        ("scale",StandardScaler()),
        ("ridge",Ridge(alpha=20.0)),
    ])
    pipe.fit(train_x,train_y)
    return np.asarray(pipe.predict(test_x),dtype=float)


def _season_forward_family(g: pd.DataFrame, seasons: np.ndarray, base: np.ndarray, actual: np.ndarray,
                           features: list[str], market: str, clip: float, min_features: int=2) -> dict[str,Any]:
    """Chronological residual challenger with a frozen <=2023 discovery rule.

    With NCAAF history beginning in 2022, 2023 is the first fully season-forward
    discovery fold. 2024 and 2025 are both required as untouched confirmation
    folds. Confirmation can reject a discovery but can never select features or
    alter thresholds.
    """
    market=str(market).upper(); n=len(g)
    pred=np.full(n,np.nan,dtype=float); fold_rows=[]
    for sy in sorted({int(x) for x in seasons[np.isfinite(seasons)] if int(x)<=max(CONFIRMATION_SEASONS)}):
        tr=np.isfinite(seasons)&(seasons<sy)&np.isfinite(base)&np.isfinite(actual)
        va=np.isfinite(seasons)&(seasons==sy)&np.isfinite(base)&np.isfinite(actual)
        if tr.sum()<250 or va.sum()<50: continue
        Xtr=g.loc[tr,features].apply(pd.to_numeric,errors="coerce")
        Xva=g.loc[va,features].apply(pd.to_numeric,errors="coerce")
        usable=[c for c in features if Xtr[c].notna().sum()>=100 and Xtr[c].nunique(dropna=True)>=4]
        if len(usable)<int(min_features): continue
        target=(actual-base)[tr]
        ok=np.isfinite(target)
        if ok.sum()<200: continue
        try: corr=_fit_ridge_predict(Xtr.loc[ok,usable],target[ok],Xva[usable])
        except Exception: continue
        corr=np.clip(corr,-clip,clip); pred[va]=base[va]+corr
        bm=_mae(actual[va],base[va]); cm=_mae(actual[va],pred[va])
        br=_rmse(actual[va],base[va]); cr=_rmse(actual[va],pred[va])
        fold_rows.append({
            "season":sy,"n":int(va.sum()),
            "baseline_mae":bm,"corrected_mae":cm,"mae_improvement":bm-cm,
            "baseline_rmse":br,"corrected_rmse":cr,"rmse_improvement":br-cr,
            "mean_abs_correction":float(np.nanmean(np.abs(corr))) if len(corr) else np.nan,
            "usable_features":usable,
        })
    disc=[x for x in fold_rows if x["season"]<=DISCOVERY_MAX_SEASON]
    conf=[x for x in fold_rows if x["season"] in CONFIRMATION_SEASONS]
    d_mae=[x["mae_improvement"] for x in disc if np.isfinite(x["mae_improvement"])]
    d_rmse=[x["rmse_improvement"] for x in disc if np.isfinite(x["rmse_improvement"])]
    c_mae=[x["mae_improvement"] for x in conf if np.isfinite(x["mae_improvement"])]
    c_rmse=[x["rmse_improvement"] for x in conf if np.isfinite(x["rmse_improvement"])]
    # 2023 is the first season-forward discovery fold because 2022 is the
    # initial training season. Both 2024 AND 2025 must then confirm.
    discovery_pass=bool(len(d_mae)>=1 and len(d_rmse)>=1 and np.mean(d_mae)>0 and np.mean(d_rmse)>0)
    confirmation_pass=bool(
        discovery_pass and len(c_mae)>=2 and len(c_rmse)>=2 and
        all(x>0 for x in c_mae) and all(x>0 for x in c_rmse)
    )
    valid=np.isfinite(pred)&np.isfinite(actual)&np.isfinite(base)
    correction=pred-base
    open_line = _num(g,"Consensus_Open_Spread").to_numpy(dtype=float) if market=="SPREADS" else _num(g,"Consensus_Open_Total").to_numpy(dtype=float)
    incumbent_edge=(base+open_line) if market=="SPREADS" else (base-open_line)
    return {
        "market":market,"feature_count":len(features),"features":features,"folds":fold_rows,
        "discovery_rule":"FIRST_SEASON_FORWARD_DISCOVERY_FOLD_MAE_AND_RMSE_GT_0__2024_AND_2025_CONFIRM_BOTH",
        "discovery_pass":discovery_pass,"confirmation_pass":confirmation_pass,
        "discovery_mean_improvement":float(np.mean(d_mae)) if d_mae else np.nan,
        "discovery_mean_rmse_improvement":float(np.mean(d_rmse)) if d_rmse else np.nan,
        "confirmation_mean_improvement":float(np.mean(c_mae)) if c_mae else np.nan,
        "confirmation_mean_rmse_improvement":float(np.mean(c_rmse)) if c_rmse else np.nan,
        "all_oof_n":int(valid.sum()),
        "all_oof_mae_improvement":_mae(actual[valid],base[valid])-_mae(actual[valid],pred[valid]) if valid.any() else np.nan,
        "all_oof_rmse_improvement":_rmse(actual[valid],base[valid])-_rmse(actual[valid],pred[valid]) if valid.any() else np.nan,
        "correction_vs_incumbent_edge_corr":_corr(correction,incumbent_edge),
        "authority_state":"CONFIRMED_SHADOW" if confirmation_pass else ("DISCOVERY_CANDIDATE" if discovery_pass else "RESEARCH"),
        "production_authority":0,
    }


def run_orthogonal_stat_research(games: pd.DataFrame, seasons: np.ndarray, oof_margin: np.ndarray,
                                 oof_total: np.ndarray, candidate_cols: list[str], log_func=print) -> dict[str,Any]:
    families=_classify_feature_families(candidate_cols)
    actual_margin=_num(games,"Actual_Margin").to_numpy(dtype=float)
    actual_total=_num(games,"Actual_Total").to_numpy(dtype=float)
    out={"version":"NCAAF-RV2.1-ORTHOGONAL-STAT","selection_freeze":DISCOVERY_MAX_SEASON,"confirmation_seasons":list(CONFIRMATION_SEASONS),
         "prospective_min_season":PROSPECTIVE_MIN_SEASON,"production_authority":0,"families":{}}
    for fam,cols in families.items():
        sp=_season_forward_family(games,seasons,oof_margin,actual_margin,cols,"SPREADS",6.0,min_features=2)
        tot=_season_forward_family(games,seasons,oof_total,actual_total,cols,"TOTALS",8.0,min_features=2)
        out["families"][fam]={"SPREADS":sp,"TOTALS":tot}
        log_func(f"[NCAAF-RV21-STAT-FAMILY] family={fam} features={len(cols)} spread_discovery={sp['discovery_pass']} spread_confirm={sp['confirmation_pass']} spread_conf_mae={sp['confirmation_mean_improvement']:.5f} spread_conf_rmse={sp['confirmation_mean_rmse_improvement']:.5f} totals_discovery={tot['discovery_pass']} totals_confirm={tot['confirmation_pass']} totals_conf_mae={tot['confirmation_mean_improvement']:.5f} totals_conf_rmse={tot['confirmation_mean_rmse_improvement']:.5f} authority=0")
    out["confirmed_spread_families"]=[k for k,v in out["families"].items() if v["SPREADS"]["confirmation_pass"]]
    out["confirmed_totals_families"]=[k for k,v in out["families"].items() if v["TOTALS"]["confirmation_pass"]]
    return out


SPARSE_STAT_CANDIDATES={
    "SPREADS":["Diff_RawRecent3_Off_YPP","B_RawSeason_GameAdj_Def_Rush_YPA"],
    "TOTALS":["A_RawRecent3_Def_Rush_YPA_Allowed"],
}

def run_sparse_stat_research(games: pd.DataFrame, seasons: np.ndarray, oof_margin: np.ndarray,
                             oof_total: np.ndarray, log_func=print) -> dict[str,Any]:
    """V2.1 sparse residual pass seeded only by features that survived the prior gate.

    The candidate list is frozen before 2024-2025 confirmation. We test singleton
    challengers plus one predeclared spread interaction; no broad AutoFS search is
    reintroduced.
    """
    g=games.copy()
    actual_margin=_num(g,"Actual_Margin").to_numpy(dtype=float)
    actual_total=_num(g,"Actual_Total").to_numpy(dtype=float)
    tests=[]
    spread=[c for c in SPARSE_STAT_CANDIDATES["SPREADS"] if c in g.columns]
    totals=[c for c in SPARSE_STAT_CANDIDATES["TOTALS"] if c in g.columns]
    for c in spread:
        tests.append(("SPREADS",c,[c]))
    if len(spread)==2:
        inter="RV21_YPP_X_OPP_RUSH"
        a=pd.to_numeric(g[spread[0]],errors="coerce"); b=pd.to_numeric(g[spread[1]],errors="coerce")
        g[inter]=a*b
        tests.append(("SPREADS","YPP_PLUS_OPP_RUSH",spread.copy()))
        tests.append(("SPREADS","YPP_X_OPP_RUSH",[inter]))
    for c in totals:
        tests.append(("TOTALS",c,[c]))
    rows=[]
    for market,name,features in tests:
        base=oof_margin if market=="SPREADS" else oof_total
        actual=actual_margin if market=="SPREADS" else actual_total
        r=_season_forward_family(g,seasons,base,actual,features,market,6.0 if market=="SPREADS" else 8.0,min_features=1)
        r["candidate_id"]=_stable_id("NCAAF-STAT21-",[market,name]+features); r["candidate_name"]=name
        rows.append(r)
        log_func(f"[NCAAF-RV21-SPARSE-STAT] market={market} candidate={name} features={','.join(features)} discovery={r['discovery_pass']} confirm={r['confirmation_pass']} d_mae={r['discovery_mean_improvement']:.5f} d_rmse={r['discovery_mean_rmse_improvement']:.5f} c_mae={r['confirmation_mean_improvement']:.5f} c_rmse={r['confirmation_mean_rmse_improvement']:.5f} authority=0")
    return {
        "version":"NCAAF-RV2.1-SPARSE-RESIDUAL-STAT",
        "candidate_freeze":"PRIOR_GATE_SURVIVORS_ONLY__NO_AUTOFs",
        "candidates":rows,
        "confirmed_candidates":[x["candidate_id"] for x in rows if x.get("confirmation_pass")],
        "production_authority":0,
    }




# One-process cache for exact historical team-side expert flags.  This is derived
# from the same validated historical source used by the dashboard's Pathi/Big Al
# W/L engine, then projected onto the Miner's HOME-oriented physical-game frame.
_V214_EXPERT_SIDE_CACHE: dict[str,Any] = {}


def _v214_side_key(df: pd.DataFrame, team_col: str, opp_col: str | None=None) -> pd.Series:
    sy=pd.to_numeric(df.get("Season"),errors="coerce").round().astype("Int64").astype(str)
    gid=df.get("Source_Game_ID",pd.Series("",index=df.index)).astype(str).str.strip()
    gid=gid.mask(gid.str.lower().isin({"","nan","none","<na>"}),"")
    date=pd.to_datetime(df.get("Game_Date",pd.Series(pd.NaT,index=df.index)),errors="coerce",utc=True).dt.strftime("%Y-%m-%d").fillna("")
    team=df.get(team_col,pd.Series("",index=df.index)).map(_pt_team_key)
    opp=df.get(opp_col,pd.Series("",index=df.index)).map(_pt_team_key) if opp_col else pd.Series("",index=df.index)
    # Prefer the stable source game id.  If it is absent, use date+oriented pair;
    # never collapse a whole team's season into one key.
    game_token=pd.Series(np.where(gid.ne(""),"ID:"+gid,"DATE:"+date+"|"+team+"|"+opp),index=df.index,dtype=str)
    valid=sy.ne("<NA>") & team.ne("") & (gid.ne("") | date.ne(""))
    return (sy+"|"+game_token+"|"+team).where(valid,"")


def _v214_occurrence_token(x: Any) -> str:
    """Match dashboard _hc_team_token exactly: lowercase alphanumeric only."""
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def _v214_occurrence_key(df: pd.DataFrame, team_col: str, opp_col: str) -> pd.Series:
    """Key compatible with dashboard historical-system occurrence ledgers.

    IMPORTANT: the dashboard ledger uses _hc_team_token, which removes all
    punctuation/whitespace.  Do not use the Prediction-Tracker normalizer here;
    its space-preserving/abbreviation-expanding semantics produce non-matching
    keys even when the same physical game is present.
    """
    season=pd.to_numeric(df.get("Season"),errors="coerce")
    date=pd.to_datetime(df.get("Game_Date",df.get("Game_Start",pd.Series(pd.NaT,index=df.index))),errors="coerce",utc=True)
    team=df.get(team_col,pd.Series("",index=df.index)).map(_v214_occurrence_token)
    opp=df.get(opp_col,pd.Series("",index=df.index)).map(_v214_occurrence_token)
    yr=season.round().astype("Int64").astype(str)
    ds=date.dt.strftime("%Y-%m-%d").fillna("")
    valid=yr.ne("<NA>") & ds.ne("") & team.ne("") & opp.ne("")
    return (yr+"|"+ds+"|"+team+"|"+opp).where(valid,"")


def _attach_exact_expert_flags_to_miner(dashboard_module, miner_games: pd.DataFrame, *, log_func=print) -> tuple[pd.DataFrame,dict[str,Any]]:
    """Attach exact Pathi/Big Al historical flags to the Miner's game frame.

    The historical W/L engine operates one-row-per-team-side, while the research
    Miner is one HOME-oriented row per physical game.  Earlier bridges tried to
    recreate flags from the lean Miner frame and could silently lose all expert
    atoms.  V2.14 instead reuses the same validated historical source/builders as
    the W/L engine, then maps HOME-side and ROAD-side triggers independently.

    ROAD-side flags receive a ``__ROAD_SIDE`` suffix.  The Miner remains free to
    learn PLAY_ON vs FADE direction; no expert flag receives authority directly.
    """
    if miner_games is None or miner_games.empty:
        return miner_games,{"status":"EMPTY","authority":0}
    out=miner_games.copy()

    # Primary source: reuse the exact validated directional-system occurrence
    # ledger already used by the dashboard W/L engine.  This prevents a second
    # reconstruction from drifting away from the systems we actually graded.
    hist_cache=getattr(dashboard_module,"_V143_SYSTEM_HISTORY_CACHE",None)
    if isinstance(hist_cache,dict) and hist_cache:
        try:
            home_occ=_v214_occurrence_key(out,"Team_Norm","Opponent_Norm")
            road_occ=_v214_occurrence_key(out,"Opponent_Norm","Team_Norm")
            hp=rp=hb=rb=0; pc=bc=0
            for name,st in hist_cache.items():
                if not isinstance(st,dict) or str(st.get("role","")).lower()!="directional":
                    continue
                fam=str(st.get("family",""))
                if fam not in {"Pathi","BigAl"}:
                    continue
                occ=st.get("occurrences") or []
                keys=set()
                for r in occ:
                    if not isinstance(r,dict):
                        continue
                    try:
                        yr=str(int(float(r.get("season"))))
                    except Exception:
                        continue
                    ds=str(r.get("date") or "").strip()[:10]
                    tm=_v214_occurrence_token(r.get("team",""))
                    op=_v214_occurrence_token(r.get("opponent",""))
                    if ds and tm and op:
                        keys.add(f"{yr}|{ds}|{tm}|{op}")
                if not keys:
                    continue
                hv=home_occ.isin(keys).astype("int8")
                rv=road_occ.isin(keys).astype("int8")
                out[name]=hv
                out[name+"__ROAD_SIDE"]=rv
                if fam=="Pathi":
                    pc+=1; hp+=int(hv.sum()); rp+=int(rv.sum())
                else:
                    bc+=1; hb+=int(hv.sum()); rb+=int(rv.sum())
            if pc or bc:
                eligible=[st for st in hist_cache.values()
                          if isinstance(st,dict) and str(st.get("role",""))=="directional"
                          and str(st.get("family","")) in {"Pathi","BigAl"}]
                expected_fired=sum(int(st.get("fired",0) or 0) for st in eligible)
                expected_graded=sum(int(st.get("graded",st.get("sample",0)) or 0) for st in eligible)
                occurrence_records=sum(len(st.get("occurrences") or []) for st in eligible)
                projected_fires=hp+rp+hb+rb
                ungraded_fires=expected_fired-expected_graded
                projection_delta=projected_fires-occurrence_records
                ledger_dedup_delta=occurrence_records-expected_graded
                pathi_fired=sum(int(st.get("fired",0) or 0) for st in eligible if str(st.get("family"))=="Pathi")
                pathi_graded=sum(int(st.get("graded",st.get("sample",0)) or 0) for st in eligible if str(st.get("family"))=="Pathi")
                bigal_fired=sum(int(st.get("fired",0) or 0) for st in eligible if str(st.get("family"))=="BigAl")
                bigal_graded=sum(int(st.get("graded",st.get("sample",0)) or 0) for st in eligible if str(st.get("family"))=="BigAl")
                recon_ok=bool(projected_fires==occurrence_records)
                diag={"status":("PASS_OCCURRENCE_LEDGER_RECONCILED" if recon_ok else "FAIL_OCCURRENCE_RECONCILIATION"),
                      "source_rows":int(len(out)*2),
                      "matched_home":int(home_occ.ne("").sum()),"matched_road":int(road_occ.ne("").sum()),
                      "pathi_cols":pc,"bigal_cols":bc,"home_pathi_fires":hp,"road_pathi_fires":rp,
                      "home_bigal_fires":hb,"road_bigal_fires":rb,
                      "expected_fired":expected_fired,"expected_graded":expected_graded,"occurrence_records":occurrence_records,
                      "ungraded_fires":ungraded_fires,"projection_delta":projection_delta,"ledger_dedup_delta":ledger_dedup_delta,
                      "pathi_fired":pathi_fired,"pathi_graded":pathi_graded,"bigal_fired":bigal_fired,"bigal_graded":bigal_graded,"authority":0}
                log_func(
                    f"[NCAAF-RV215-EXPERT-OCCURRENCE-RECON] status={'PASS' if recon_ok else 'FAIL'} "
                    f"fired={expected_fired} graded={expected_graded} occurrence_records={occurrence_records} projected={projected_fires} "
                    f"ungraded={ungraded_fires} projection_delta={projection_delta} ledger_dedup_delta={ledger_dedup_delta} "
                    f"pathi_fired={pathi_fired} pathi_graded={pathi_graded} bigal_fired={bigal_fired} bigal_graded={bigal_graded} authority=0"
                )
                if recon_ok:
                    log_func(
                        f"[NCAAF-RV215-EXPERT-OCCURRENCE-BRIDGE] status=PASS pathi_cols={pc} bigal_cols={bc} "
                        f"home_pathi_fires={hp} road_pathi_fires={rp} home_bigal_fires={hb} road_bigal_fires={rb} "
                        f"graded_occurrences={occurrence_records} ungraded_fires={ungraded_fires} authority=0"
                    )
                    return out,diag
                # Fail closed: do not expose partially projected expert atoms to Miner.
                for _c in list(out.columns):
                    if str(_c).startswith("Pathi_FB_") or str(_c).startswith("BigAl_"):
                        out.drop(columns=[_c],inplace=True,errors="ignore")
                log_func(
                    f"[NCAAF-RV215-EXPERT-OCCURRENCE-BRIDGE] status=FAIL_RECONCILIATION "
                    f"occurrence_records={occurrence_records} projected={projected_fires} delta={projection_delta} fallback=TRUE authority=0"
                )
        except Exception as _occ_exc:
            log_func(f"[NCAAF-RV2142-EXPERT-OCCURRENCE-BRIDGE] status=FALLBACK error={type(_occ_exc).__name__}:{_occ_exc} authority=0")

    pathi_default=[
        "Pathi_FB_Crossed_Key_Toward_Team","Pathi_FB_Crossed_Key_Away_From_Team",
        "Pathi_FB_Dog_Hook_Above_3","Pathi_FB_Dog_Hook_Above_7","Pathi_FB_Dog_Hook_Above_10",
        "Pathi_FB_Favorite_Below_Key_3","Pathi_FB_Favorite_Below_Key_7","Pathi_FB_Favorite_Below_Key_10",
        "Pathi_FB_Dog_Below_Key_3","Pathi_FB_Dog_Below_Key_7",
        "Pathi_FB_Favorite_Laying_Hook_3","Pathi_FB_Favorite_Laying_Hook_7",
        "Pathi_FB_Usually_Dog_Now_Favorite","Pathi_FB_Usually_Favorite_Now_Dog",
        "Pathi_FB_Dog_TotalSpread_Gap_LE10",
    ]
    bigal_cols=["BigAl_CF1_Week2Home42Win","BigAl_CF2_LateSeasonRevengeDog","BigAl_CF3_Fade19PlusFavoriteUpsetLoss"]
    try:
        cache=_V214_EXPERT_SIDE_CACHE.get("side")
        if not isinstance(cache,dict):
            bq=getattr(dashboard_module,"bq_client",None)
            view=getattr(dashboard_module,"HISTORICAL_NCAAF_CORE_VIEW",None)
            if bq is None or not view:
                raise RuntimeError("dashboard historical BQ source unavailable")
            h=bq.query(f"SELECT * FROM `{view}` WHERE Historical_Core_Eligible = 1").to_dataframe()
            if h is None or h.empty:
                raise RuntimeError("historical expert source returned zero rows")
            # Exact Pathi context restoration: validated context first, historical
            # precomputed flags second, deterministic pregame reconstruction last.
            pstate=h.copy(); pok=False; pdiag={}
            if hasattr(dashboard_module,"_hc_restore_authoritative_pathi_flags"):
                pstate,pok,pdiag=dashboard_module._hc_restore_authoritative_pathi_flags(h,log_func=log_func)
            elif hasattr(dashboard_module,"add_pathi_football_key_features"):
                pstate=pstate.copy(); pstate["Sport"]="NCAAF"; pstate["Market"]="spreads"
                pstate=dashboard_module.add_pathi_football_key_features(pstate); pok=True
            # Exact Big Al historical state reconstruction + deterministic rules.
            bstate=h.copy()
            if hasattr(dashboard_module,"_hc_prepare_bigal_history_state"):
                bstate=dashboard_module._hc_prepare_bigal_history_state(h,log_func=log_func)
            if hasattr(dashboard_module,"add_pathi_bigal_rule_flags"):
                bstate=dashboard_module.add_pathi_bigal_rule_flags(bstate)
            pathi_cols=list(getattr(dashboard_module,"PATHI_DIRECTIONAL_MEMORY_COLS",()) or pathi_default)
            pathi_cols=[c for c in pathi_cols if c in pstate.columns]
            bigal_use=[c for c in bigal_cols if c in bstate.columns]

            def make_lookup(frame, cols):
                if frame is None or frame.empty or not cols:
                    return pd.DataFrame()
                z=frame.copy()
                if "Team_Norm" not in z.columns and "Team" in z.columns: z["Team_Norm"]=z["Team"]
                z["__side_key"]=_v214_side_key(z,"Team_Norm","Opponent_Norm")
                keep=["__side_key"]+cols
                zz=z[keep].copy()
                zz=zz.loc[zz["__side_key"].ne("")].copy()
                for c in cols: zz[c]=pd.to_numeric(zz[c],errors="coerce")
                # Duplicate source snapshots collapse conservatively: a trigger is
                # present if any authoritative row says it fired.
                return zz.groupby("__side_key",as_index=True,sort=False)[cols].max()

            plook=make_lookup(pstate,pathi_cols); blook=make_lookup(bstate,bigal_use)
            cache={"pathi_lookup":plook,"bigal_lookup":blook,"pathi_cols":pathi_cols,"bigal_cols":bigal_use,
                   "source_rows":int(len(h)),"pathi_source_ok":bool(pok),"pathi_diag":pdiag}
            _V214_EXPERT_SIDE_CACHE["side"]=cache

        home_key=_v214_side_key(out,"Team_Norm","Opponent_Norm")
        road_key=_v214_side_key(out,"Opponent_Norm","Team_Norm")
        matched_home=matched_road=0; hp=rp=hb=rb=0
        for family in ("pathi","bigal"):
            look=cache.get(f"{family}_lookup")
            cols=list(cache.get(f"{family}_cols") or [])
            if not isinstance(look,pd.DataFrame) or look.empty: continue
            hidx=pd.Index(home_key); ridx=pd.Index(road_key)
            matched_home=max(matched_home,int(hidx.isin(look.index).sum()))
            matched_road=max(matched_road,int(ridx.isin(look.index).sum()))
            for c in cols:
                hvals=pd.to_numeric(pd.Series(home_key.map(look[c]),index=out.index),errors="coerce")
                rvals=pd.to_numeric(pd.Series(road_key.map(look[c]),index=out.index),errors="coerce")
                out[c]=hvals.fillna(0).eq(1).astype("int8")
                out[c+"__ROAD_SIDE"]=rvals.fillna(0).eq(1).astype("int8")
                if family=="pathi": hp+=int(hvals.fillna(0).eq(1).sum()); rp+=int(rvals.fillna(0).eq(1).sum())
                else: hb+=int(hvals.fillna(0).eq(1).sum()); rb+=int(rvals.fillna(0).eq(1).sum())
        diag={"status":"PASS","source_rows":int(cache.get("source_rows",0)),"matched_home":matched_home,"matched_road":matched_road,
              "pathi_cols":len(cache.get("pathi_cols") or []),"bigal_cols":len(cache.get("bigal_cols") or []),
              "home_pathi_fires":hp,"road_pathi_fires":rp,"home_bigal_fires":hb,"road_bigal_fires":rb,"authority":0}
        log_func(
            f"[NCAAF-RV214-EXPERT-SIDE-BRIDGE] status=PASS source_rows={diag['source_rows']} matched_home={matched_home}/{len(out)} "
            f"matched_road={matched_road}/{len(out)} pathi_cols={diag['pathi_cols']} bigal_cols={diag['bigal_cols']} "
            f"home_pathi_fires={hp} road_pathi_fires={rp} home_bigal_fires={hb} road_bigal_fires={rb} authority=0"
        )
        return out,diag
    except Exception as exc:
        log_func(f"[NCAAF-RV214-EXPERT-SIDE-BRIDGE] status=UNAVAILABLE error={type(exc).__name__}:{exc} authority=0 fail_closed=TRUE")
        return out,{"status":"UNAVAILABLE","error":f"{type(exc).__name__}:{exc}","authority":0}


# V2.28 historical state bridge. The research Miner is one HOME-oriented row
# per physical game, while the validated historical core view is one row per
# team side. Reconstruct all prior-only team states on the richer side history,
# then project HOME-side state to Team_* fields and ROAD-side state to Opp_*
# fields on the Miner frame. No current-game outcome is ever used in a row's
# pregame state.
_V228_HISTORICAL_STATE_CACHE: dict[str,Any] = {}


def _attach_historical_state_context_to_miner(dashboard_module, miner_games: pd.DataFrame, *, log_func=print) -> tuple[pd.DataFrame,dict[str,Any]]:
    if miner_games is None or miner_games.empty:
        return miner_games,{"status":"EMPTY","source_rows":0,"selection_influence":0}
    out=miner_games.copy()
    bq=getattr(dashboard_module,"bq_client",None) if dashboard_module is not None else None
    view=getattr(dashboard_module,"HISTORICAL_NCAAF_CORE_VIEW",None) if dashboard_module is not None else None
    if bq is None or not view:
        return out,{"status":"NO_HISTORICAL_SIDE_SOURCE","source_rows":0,"selection_influence":0}

    cache=_V228_HISTORICAL_STATE_CACHE.get("side_state")
    if not isinstance(cache,dict):
        try:
            q=bq.query(f"SELECT * FROM `{view}` WHERE Historical_Core_Eligible = 1 AND Season <= {max(CONFIRMATION_SEASONS)}")
            try:
                h=q.to_dataframe(create_bqstorage_client=False)
            except TypeError:
                h=q.to_dataframe()
            if h is None or h.empty:
                raise RuntimeError("historical side source returned zero rows")
            h=h.copy().reset_index(drop=True)
            raw_rows=int(len(h))

            def _txt(df,*cols):
                z=pd.Series("",index=df.index,dtype="string")
                for c in cols:
                    if c in df.columns:
                        v=df[c].astype("string").fillna("").str.lower().str.strip()
                        z=z.where(z.ne(""),v)
                return z
            def _numc(df,*cols):
                z=pd.Series(np.nan,index=df.index,dtype=float)
                for c in cols:
                    if c in df.columns:
                        v=pd.to_numeric(df[c],errors="coerce")
                        z=z.where(z.notna(),v)
                return z

            if "Team_Norm" not in h.columns and "Team" in h.columns: h["Team_Norm"]=h["Team"]
            if "Opponent_Norm" not in h.columns and "Opponent" in h.columns: h["Opponent_Norm"]=h["Opponent"]
            h["__team"]=_txt(h,"Team_Norm","Team")
            h["__opp"]=_txt(h,"Opponent_Norm","Opponent")
            h["__season"]=_numc(h,"Season")
            h["__date"]=pd.to_datetime(h.get("Game_Date",h.get("Game_Start",pd.Series(pd.NaT,index=h.index))),errors="coerce",utc=True)
            h["__home"]=_numc(h,"Is_Home")
            h["__spread"]=_numc(h,"Opening_Spread","Consensus_Open_Spread")
            h["__pf"]=_numc(h,"Team_Score","Points_For")
            h["__pa"]=_numc(h,"Opponent_Score","Points_Against")
            h["__margin"]=h["__pf"]-h["__pa"]
            h["__ats_margin"]=h["__margin"]+h["__spread"]
            h["__orig_idx"]=np.arange(len(h),dtype=int)
            h["__side_key"]=_v214_side_key(h,"Team_Norm","Opponent_Norm")
            # Collapse duplicate snapshots of the same team-side physical game
            # before chronology so one game can never count twice in prior state.
            h=h.loc[h["__team"].ne("")&h["__season"].notna()&h["__date"].notna()&h["__side_key"].ne("")].copy()
            h=h.sort_values(["__season","__team","__date","__orig_idx"],kind="stable")
            h=h.drop_duplicates("__side_key",keep="last").copy()
            dedup_rows=int(len(h))

            state_cols=[
                "games_prior","winpct_prior","ats_pct_prior","ats_loss_streak_prior","ats_win_streak_prior",
                "su_win_streak_prior","su_loss_streak_prior","prev_home","prev_fav","prev_dog","prev_margin",
                "prev_ats_margin","prev_pf","prev_pa",
            ]
            for c in state_cols: h[c]=np.nan

            for (_sy,_tm),gg in h.groupby(["__season","__team"],sort=False,dropna=False):
                games=wins=ties=0
                ats_graded=ats_wins=0
                ats_loss_streak=ats_win_streak=0
                su_win_streak=su_loss_streak=0
                prev=None
                for ix,row in gg.iterrows():
                    h.at[ix,"games_prior"]=float(games)
                    h.at[ix,"winpct_prior"]=(wins+0.5*ties)/games if games else np.nan
                    h.at[ix,"ats_pct_prior"]=ats_wins/ats_graded if ats_graded else np.nan
                    h.at[ix,"ats_loss_streak_prior"]=float(ats_loss_streak)
                    h.at[ix,"ats_win_streak_prior"]=float(ats_win_streak)
                    h.at[ix,"su_win_streak_prior"]=float(su_win_streak)
                    h.at[ix,"su_loss_streak_prior"]=float(su_loss_streak)
                    if prev is not None:
                        for dst,src in (("prev_home","__home"),("prev_margin","__margin"),("prev_ats_margin","__ats_margin"),("prev_pf","__pf"),("prev_pa","__pa")):
                            v=prev.get(src,np.nan); h.at[ix,dst]=float(v) if pd.notna(v) else np.nan
                        ps=prev.get("__spread",np.nan)
                        if pd.notna(ps):
                            h.at[ix,"prev_fav"]=float(float(ps)<0)
                            h.at[ix,"prev_dog"]=float(float(ps)>0)
                    # Update only after writing the current row's pregame state.
                    m=row.get("__margin",np.nan)
                    if pd.notna(m):
                        games+=1
                        if float(m)>0:
                            wins+=1; su_win_streak+=1; su_loss_streak=0
                        elif float(m)<0:
                            su_loss_streak+=1; su_win_streak=0
                        else:
                            ties+=1; su_win_streak=0; su_loss_streak=0
                    am=row.get("__ats_margin",np.nan)
                    if pd.notna(am) and not np.isclose(float(am),0.0,atol=1e-9):
                        ats_graded+=1
                        if float(am)>0:
                            ats_wins+=1; ats_win_streak+=1; ats_loss_streak=0
                        else:
                            ats_loss_streak+=1; ats_win_streak=0
                    else:
                        # A push/ungraded game makes exact 0-X coverless false.
                        ats_loss_streak=0; ats_win_streak=0
                    prev=row

            lookup=h.set_index("__side_key",drop=False)
            cache={"lookup":lookup,"source_rows_raw":raw_rows,"source_rows_dedup":dedup_rows}
            _V228_HISTORICAL_STATE_CACHE["side_state"]=cache
        except Exception as exc:
            log_func(f"[NCAAF-HISTORICAL-STATE-BRIDGE] status=UNAVAILABLE error={type(exc).__name__}:{exc} selection_influence=0")
            return out,{"status":"UNAVAILABLE","error":f"{type(exc).__name__}:{exc}","source_rows":0,"selection_influence":0}

    lookup=cache.get("lookup")
    if not isinstance(lookup,pd.DataFrame) or lookup.empty:
        return out,{"status":"EMPTY_LOOKUP","source_rows":0,"selection_influence":0}
    home_key=_v214_side_key(out,"Team_Norm","Opponent_Norm")
    road_key=_v214_side_key(out,"Opponent_Norm","Team_Norm")

    def _mapped(keys: pd.Series, src: str) -> pd.Series:
        try:
            vals=keys.map(lookup[src])
            return pd.to_numeric(pd.Series(vals,index=out.index),errors="coerce")
        except Exception:
            return pd.Series(np.nan,index=out.index,dtype=float)
    def _attach(dst: str, src: str, keys: pd.Series):
        vals=_mapped(keys,src)
        if dst in out.columns:
            cur=pd.to_numeric(out[dst],errors="coerce")
            out[dst]=cur.where(cur.notna(),vals)
        else:
            out[dst]=vals

    # HOME-side team state.
    for dst,src in (
        ("Team_Game_Number_Prior","games_prior"),("Team_WinPct_Prior","winpct_prior"),("ATS_WinPct_Prior","ats_pct_prior"),
        ("ATS_Loss_Streak_Prior","ats_loss_streak_prior"),("ATS_Win_Streak_Prior","ats_win_streak_prior"),
        ("Current_Win_Streak_Prior","su_win_streak_prior"),("Current_Loss_Streak_Prior","su_loss_streak_prior"),
        ("Prev_Is_Home","prev_home"),("Prev_Is_ML_Favorite","prev_fav"),("Prev_Is_ML_Dog","prev_dog"),
        ("Prev_SU_Margin","prev_margin"),("Prev_ATS_Margin","prev_ats_margin"),("Prev_Team_Score","prev_pf"),("Prev_Opponent_Score","prev_pa"),
    ): _attach(dst,src,home_key)
    # ROAD-side opponent state projected onto Opp_* fields.
    for dst,src in (
        ("Opp_Game_Number_Prior","games_prior"),("Opp_WinPct_Prior","winpct_prior"),("Opp_ATS_WinPct_Prior","ats_pct_prior"),
        ("Opp_ATS_Loss_Streak_Prior","ats_loss_streak_prior"),("Opp_ATS_Win_Streak_Prior","ats_win_streak_prior"),
        ("Opp_Current_Win_Streak_Prior","su_win_streak_prior"),("Opp_Current_Loss_Streak_Prior","su_loss_streak_prior"),
        ("Opp_Prev_Is_Home","prev_home"),("Opp_Prev_Is_ML_Favorite","prev_fav"),("Opp_Prev_Is_ML_Dog","prev_dog"),
        ("Opp_Prev_SU_Margin","prev_margin"),("Opp_Prev_ATS_Margin","prev_ats_margin"),("Opp_Prev_Team_Score","prev_pf"),("Opp_Prev_Opponent_Score","prev_pa"),
    ): _attach(dst,src,road_key)

    _season_vals=pd.to_numeric(out.get("Season",pd.Series(np.nan,index=out.index)),errors="coerce")
    _historical_target=np.isfinite(_season_vals.to_numpy(float))&(_season_vals.to_numpy(float)<=max(CONFIRMATION_SEASONS))
    _historical_target_rows=int(_historical_target.sum()) if _historical_target.any() else int(len(out))
    _prospective_cache_rows=int(len(out)-_historical_target_rows)
    matched_home=int((home_key.isin(lookup.index).to_numpy(dtype=bool)&_historical_target).sum()) if len(out) else 0
    matched_road=int((road_key.isin(lookup.index).to_numpy(dtype=bool)&_historical_target).sum()) if len(out) else 0
    state_ready=int(pd.to_numeric(out.get("Team_Game_Number_Prior"),errors="coerce").notna().sum())
    opp_state_ready=int(pd.to_numeric(out.get("Opp_Game_Number_Prior"),errors="coerce").notna().sum())
    bounce_ready=int((pd.to_numeric(out.get("Prev_Is_Home"),errors="coerce").notna() & pd.to_numeric(out.get("Prev_Is_ML_Favorite"),errors="coerce").notna() & pd.to_numeric(out.get("Prev_SU_Margin"),errors="coerce").notna()).sum())
    opp_bounce_ready=int((pd.to_numeric(out.get("Opp_Prev_Is_Home"),errors="coerce").notna() & pd.to_numeric(out.get("Opp_Prev_Is_ML_Favorite"),errors="coerce").notna() & pd.to_numeric(out.get("Opp_Prev_SU_Margin"),errors="coerce").notna()).sum())
    diag={"status":"PASS","source_rows_raw":int(cache.get("source_rows_raw",0)),"source_rows_dedup":int(cache.get("source_rows_dedup",0)),
          "matched_home":matched_home,"matched_road":matched_road,"historical_target_rows":_historical_target_rows,"all_cache_rows":int(len(out)),"prospective_cache_rows":_prospective_cache_rows,
          "miner_rows":int(len(out)),"state_ready":state_ready,"opp_state_ready":opp_state_ready,
          "bounceback_ready":bounce_ready,"opp_bounceback_ready":opp_bounce_ready,"selection_influence":0,"outcomes_2026_used":False}
    log_func(f"[NCAAF-HISTORICAL-STATE-BRIDGE] status=PASS source_rows_raw={diag['source_rows_raw']} source_rows_dedup={diag['source_rows_dedup']} matched_home={matched_home}/{_historical_target_rows} matched_road={matched_road}/{_historical_target_rows} all_cache_games={len(out)} prospective_2026plus_excluded={_prospective_cache_rows} state_ready={state_ready} opp_state_ready={opp_state_ready} bounce_ready={bounce_ready} opp_bounce_ready={opp_bounce_ready} outcomes_2026_used=FALSE selection_influence=0")
    return out,diag

# V2.31 deep objective context bridge. These fields are reconstructed from the
# validated team-side history using only information available before each game.
# The bridge supplies System Miner with the missing objective vocabulary exposed
# by expert examples: result quality/regression, schedule/resume quality,
# recent-vs-season divergence, and offense-vs-defense matchup differentials.
# 2026+ completed games may update current trigger context, but never participate
# in discovery, confirmation, threshold selection, or system promotion.
_V231_DEEP_CONTEXT_CACHE: dict[str,Any] = {}


NCAAF_HISTORICAL_RAW_TABLE = os.getenv("NCAAF_HISTORICAL_RAW_TABLE", "sharplogger.sharp_data.ncaaf_historical_game_side_raw")


def _v231_finite_mean(values) -> float:
    a=pd.to_numeric(pd.Series(list(values),dtype="object"),errors="coerce").to_numpy(dtype=float)
    a=a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else np.nan


def _v231_deep_context_source(dashboard_module, *, log_func=print) -> dict[str,Any]:
    cache=_V231_DEEP_CONTEXT_CACHE.get("deep_source")
    if isinstance(cache,dict): return cache
    bq=getattr(dashboard_module,"bq_client",None) if dashboard_module is not None else None
    view=getattr(dashboard_module,"HISTORICAL_NCAAF_CORE_VIEW",None) if dashboard_module is not None else None
    if bq is None or not view:
        return {"status":"NO_HISTORICAL_SIDE_SOURCE","lookup":pd.DataFrame(),"latest":pd.DataFrame(),"source_rows":0}
    try:
        # V2.31.1 source repair: the Historical Core view intentionally exposes
        # leakage-safe PRE-GAME context, not the full current-game BigDataBall
        # box score. V2.31 incorrectly tried to reconstruct prior-only deep
        # context from that view, which left result-quality/recent/matchup raw
        # metrics empty in Cloud Run even though the raw table contains them.
        #
        # Read the raw game-side box score only for rows that are eligible in the
        # Historical Core view. All outcome/stat fields are shifted/rolled below
        # before becoming Miner inputs, so no current-game result can enter its
        # own pregame context.
        raw_table=NCAAF_HISTORICAL_RAW_TABLE
        q=bq.query(f"""
            SELECT r.*
            FROM `{raw_table}` r
            WHERE r.Season <= 2026
              AND EXISTS (
                SELECT 1
                FROM `{view}` c
                WHERE c.Historical_Core_Eligible = 1
                  AND c.Season = r.Season
                  AND c.Source_Game_ID = r.Source_Game_ID
                  AND LOWER(TRIM(c.Team_Norm)) = LOWER(TRIM(r.Team_Norm))
              )
        """)
        try: h=q.to_dataframe(create_bqstorage_client=False)
        except TypeError: h=q.to_dataframe()
        if h is None or h.empty: raise RuntimeError("eligible raw historical side source returned zero rows")
        h=h.copy().reset_index(drop=True); raw_rows=int(len(h))
        def _ready_count(col):
            return int(pd.to_numeric(h[col],errors="coerce").notna().sum()) if col in h.columns else 0
        log_func(
            f"[NCAAF-DEEP-CONTEXT-V2311-SOURCE] status=PASS source=RAW_JOINED_TO_ELIGIBLE_CORE rows={raw_rows} "
            f"yards={_ready_count('Postgame_Total_Yards')} plays={_ready_count('Postgame_Total_Plays')} "
            f"first_downs={_ready_count('Postgame_First_Downs')} turnovers={_ready_count('Postgame_Turnovers')} "
            f"pass_att={_ready_count('Postgame_Pass_Att')} rush_att={_ready_count('Postgame_Rush_Att')} "
            f"sacks={_ready_count('Postgame_Sacks')} authority=0"
        )

        def _txt(df,*cols):
            z=pd.Series("",index=df.index,dtype="string")
            for c in cols:
                if c in df.columns:
                    v=df[c].astype("string").fillna("").str.lower().str.strip()
                    z=z.where(z.ne(""),v)
            return z
        def _numc(df,*cols):
            z=pd.Series(np.nan,index=df.index,dtype=float)
            for c in cols:
                if c in df.columns:
                    v=pd.to_numeric(df[c],errors="coerce")
                    z=z.where(z.notna(),v)
            return z
        def _ratio(a,b):
            aa=pd.to_numeric(a,errors="coerce"); bb=pd.to_numeric(b,errors="coerce")
            return aa.div(bb.where(bb.gt(0)))

        if "Team_Norm" not in h.columns and "Team" in h.columns: h["Team_Norm"]=h["Team"]
        if "Opponent_Norm" not in h.columns and "Opponent" in h.columns: h["Opponent_Norm"]=h["Opponent"]
        h["__team"]=_txt(h,"Team_Norm","Team")
        h["__opp"]=_txt(h,"Opponent_Norm","Opponent")
        h["__season"]=_numc(h,"Season")
        h["__date"]=pd.to_datetime(h.get("Game_Date",h.get("Game_Start",pd.Series(pd.NaT,index=h.index))),errors="coerce",utc=True)
        h["__pf"]=_numc(h,"Team_Score","Points_For")
        h["__pa"]=_numc(h,"Opponent_Score","Points_Against")
        h["__margin"]=h["__pf"]-h["__pa"]
        h["__yards"]=_numc(h,"Postgame_Total_Yards","Total_Yards")
        h["__plays"]=_numc(h,"Postgame_Total_Plays","Total_Plays")
        h["__ypp"]=_ratio(h["__yards"],h["__plays"])
        h["__rush_yards"]=_numc(h,"Postgame_Rush_Yards","Rush_Yards")
        h["__rush_att"]=_numc(h,"Postgame_Rush_Att","Rush_Att")
        h["__rush_ypa"]=_ratio(h["__rush_yards"],h["__rush_att"])
        h["__pass_yards"]=_numc(h,"Postgame_Pass_Yards","Pass_Yards")
        h["__pass_att"]=_numc(h,"Postgame_Pass_Att","Pass_Att")
        h["__pass_ypa"]=_ratio(h["__pass_yards"],h["__pass_att"])
        h["__first_downs"]=_numc(h,"Postgame_First_Downs","First_Downs")
        h["__turnovers"]=_numc(h,"Postgame_Turnovers","Turnovers")
        h["__def_sacks"]=_numc(h,"Postgame_Sacks","Def_Sacks")
        _nonoff_cols=[c for c in ("Postgame_Def_INT_TD","Postgame_Fumble_Return_TD","Postgame_Kick_Return_TD","Postgame_Punt_Return_TD") if c in h.columns]
        _nonoff=[pd.to_numeric(h[c],errors="coerce") for c in _nonoff_cols]
        h["__nonoff_td"]=pd.concat(_nonoff,axis=1).fillna(0).sum(axis=1) if _nonoff else pd.Series(np.nan,index=h.index,dtype=float)
        h["__orig_idx_v231"]=np.arange(len(h),dtype=int)
        h["__side_key"]=_v214_side_key(h,"Team_Norm","Opponent_Norm")
        h=h.loc[h["__team"].ne("")&h["__opp"].ne("")&h["__season"].notna()&h["__date"].notna()&h["__side_key"].ne("")].copy()
        h=h.sort_values(["__season","__team","__date","__orig_idx_v231"],kind="stable").drop_duplicates("__side_key",keep="last").copy()
        dedup_rows=int(len(h))

        # Same-game opposite-side raw metrics. If the opposite row is missing,
        # all downstream differentials fail closed instead of being imputed.
        raw=h.set_index("__side_key",drop=False)
        mirror_key=_v214_side_key(h,"Opponent_Norm","Team_Norm")
        for dst,src in (
            ("__opp_yards","__yards"),("__opp_plays","__plays"),("__opp_ypp","__ypp"),
            ("__opp_rush_ypa","__rush_ypa"),("__opp_pass_ypa","__pass_ypa"),("__opp_pass_att","__pass_att"),
            ("__opp_first_downs","__first_downs"),("__opp_turnovers","__turnovers"),("__opp_def_sacks","__def_sacks"),
            ("__opp_nonoff_td","__nonoff_td"),
        ):
            h[dst]=pd.to_numeric(mirror_key.map(raw[src]),errors="coerce")
        h["__net_ypp"]=h["__ypp"]-h["__opp_ypp"]
        h["__yard_margin"]=h["__yards"]-h["__opp_yards"]
        h["__ypp_margin"]=h["__ypp"]-h["__opp_ypp"]
        h["__fd_margin"]=h["__first_downs"]-h["__opp_first_downs"]
        # Positive turnover margin = team benefited by committing fewer turnovers.
        h["__turnover_margin"]=h["__opp_turnovers"]-h["__turnovers"]
        h["__nonoff_td_margin"]=h["__nonoff_td"]-h["__opp_nonoff_td"]
        h["__sacks_allowed"]=h["__opp_def_sacks"]
        h["__off_sack_rate"]=_ratio(h["__sacks_allowed"],h["__pass_att"]+h["__sacks_allowed"])
        h["__def_sack_rate"]=_ratio(h["__def_sacks"],h["__opp_pass_att"]+h["__def_sacks"])
        h["__su_value"]=np.where(h["__margin"].gt(0),1.0,np.where(h["__margin"].lt(0),0.0,np.where(h["__margin"].eq(0),0.5,np.nan)))

        h["__deceptive_win_score"]=0.0; h["__deceptive_loss_score"]=0.0
        h["__win_assist_score"]=0.0; h["__loss_adversity_score"]=0.0
        _win=h["__margin"].gt(0); _loss=h["__margin"].lt(0)
        for c in ("__yard_margin","__ypp_margin","__fd_margin"):
            h.loc[_win&h[c].lt(0),"__deceptive_win_score"] += 1.0
            h.loc[_loss&h[c].gt(0),"__deceptive_loss_score"] += 1.0
        h.loc[_win&h["__turnover_margin"].ge(2),"__win_assist_score"] += 1.0
        h.loc[_win&h["__nonoff_td_margin"].ge(1),"__win_assist_score"] += 1.0
        h.loc[_loss&h["__turnover_margin"].le(-2),"__loss_adversity_score"] += 1.0
        h.loc[_loss&h["__nonoff_td_margin"].le(-1),"__loss_adversity_score"] += 1.0

        metric_src={
            "Off_PPG":"__pf","Def_PPG":"__pa","Off_YPP":"__ypp","Def_YPP_Allowed":"__opp_ypp",
            "Rush_YPA":"__rush_ypa","Def_Rush_YPA_Allowed":"__opp_rush_ypa",
            "Pass_YPA":"__pass_ypa","Def_Pass_YPA_Allowed":"__opp_pass_ypa",
            "Net_YPP":"__net_ypp","Turnover_Margin":"__turnover_margin",
            "Sack_Allowed_Rate":"__off_sack_rate","Def_Sack_Rate":"__def_sack_rate",
        }
        recent3_names=set(metric_src)
        recent5_names={"Off_PPG","Def_PPG","Off_YPP","Def_YPP_Allowed","Net_YPP"}
        pre_cols=["Deep_Game_Count_Prior","Deep_WinPct_Prior"]
        pre_cols += [f"CTX_Season_{k}" for k in metric_src]
        pre_cols += [f"CTX_Recent3_{k}" for k in recent3_names]
        pre_cols += [f"CTX_Recent5_{k}" for k in recent5_names]
        pre_cols += [
            "RQ_Prev_Yardage_Margin","RQ_Prev_YPP_Margin","RQ_Prev_First_Down_Margin","RQ_Prev_Turnover_Margin",
            "RQ_Prev_NonOffensive_TD_Margin","RQ_Prev_Deceptive_Win_Score","RQ_Prev_Deceptive_Loss_Score",
            "RQ_Prev_Win_Assistance_Score","RQ_Prev_Loss_Adversity_Score",
        ]

        # Vectorized, strictly prior-only rolling state. This is equivalent to
        # writing each row before updating with that game's result, but avoids
        # thousands of scalar DataFrame writes on the full 2022-2026 history.
        h=h.sort_values(["__season","__team","__date","__orig_idx_v231"],kind="stable")
        _grp=h.groupby(["__season","__team"],sort=False,dropna=False)
        h["Deep_Game_Count_Prior"]=_grp.cumcount().astype(float)
        h["Deep_WinPct_Prior"]=_grp["__su_value"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        for k,src in metric_src.items():
            h[f"CTX_Season_{k}"]=_grp[src].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
            if k in recent3_names:
                h[f"CTX_Recent3_{k}"]=_grp[src].transform(lambda x: x.shift(1).rolling(3,min_periods=1).mean())
            if k in recent5_names:
                h[f"CTX_Recent5_{k}"]=_grp[src].transform(lambda x: x.shift(1).rolling(5,min_periods=1).mean())
        for dst,src in (
            ("RQ_Prev_Yardage_Margin","__yard_margin"),("RQ_Prev_YPP_Margin","__ypp_margin"),
            ("RQ_Prev_First_Down_Margin","__fd_margin"),("RQ_Prev_Turnover_Margin","__turnover_margin"),
            ("RQ_Prev_NonOffensive_TD_Margin","__nonoff_td_margin"),("RQ_Prev_Deceptive_Win_Score","__deceptive_win_score"),
            ("RQ_Prev_Deceptive_Loss_Score","__deceptive_loss_score"),("RQ_Prev_Win_Assistance_Score","__win_assist_score"),
            ("RQ_Prev_Loss_Adversity_Score","__loss_adversity_score"),
        ):
            h[dst]=_grp[src].shift(1)

        # Opponent pregame quality at the time each historical game was played.
        pre=h.set_index("__side_key",drop=False)
        mirror_key=_v214_side_key(h,"Opponent_Norm","Team_Norm")
        h["__opp_pregame_winpct"]=pd.to_numeric(mirror_key.map(pre["Deep_WinPct_Prior"]),errors="coerce")
        h["__opp_pregame_net_ypp"]=pd.to_numeric(mirror_key.map(pre["CTX_Season_Net_YPP"]),errors="coerce")
        h["__opp_adj_game_net_ypp"]=h["__net_ypp"]+h["__opp_pregame_net_ypp"]
        _grp=h.groupby(["__season","__team"],sort=False,dropna=False)
        h["CTX_SOS_WinPct"]=_grp["__opp_pregame_winpct"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        h["CTX_SOS_NetYPP"]=_grp["__opp_pregame_net_ypp"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        h["CTX_OppAdj_NetYPP"]=_grp["__opp_adj_game_net_ypp"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

        final_lookup=h.set_index("__side_key",drop=False)

        # State after the most recently completed game, used only when evaluating
        # a current/upcoming game that has no exact completed side row yet.
        latest_rows=[]
        for (_sy,_tm),gg in h.groupby(["__season","__team"],sort=False,dropna=False):
            gg=gg.sort_values(["__date","__orig_idx_v231"],kind="stable")
            last=gg.iloc[-1]; row={"__season":float(_sy),"__team":str(_tm),"Deep_Game_Count_Prior":float(len(gg)),"Deep_WinPct_Prior":_v231_finite_mean(gg["__su_value"])}
            for k,src in metric_src.items():
                row[f"CTX_Season_{k}"]=_v231_finite_mean(gg[src])
                if k in recent3_names: row[f"CTX_Recent3_{k}"]=_v231_finite_mean(gg[src].iloc[-3:])
                if k in recent5_names: row[f"CTX_Recent5_{k}"]=_v231_finite_mean(gg[src].iloc[-5:])
            for dst,src in (
                ("RQ_Prev_Yardage_Margin","__yard_margin"),("RQ_Prev_YPP_Margin","__ypp_margin"),
                ("RQ_Prev_First_Down_Margin","__fd_margin"),("RQ_Prev_Turnover_Margin","__turnover_margin"),
                ("RQ_Prev_NonOffensive_TD_Margin","__nonoff_td_margin"),("RQ_Prev_Deceptive_Win_Score","__deceptive_win_score"),
                ("RQ_Prev_Deceptive_Loss_Score","__deceptive_loss_score"),("RQ_Prev_Win_Assistance_Score","__win_assist_score"),
                ("RQ_Prev_Loss_Adversity_Score","__loss_adversity_score"),
            ): row[dst]=last.get(src,np.nan)
            row["CTX_SOS_WinPct"]=_v231_finite_mean(gg["__opp_pregame_winpct"])
            row["CTX_SOS_NetYPP"]=_v231_finite_mean(gg["__opp_pregame_net_ypp"])
            row["CTX_OppAdj_NetYPP"]=_v231_finite_mean(gg["__opp_adj_game_net_ypp"])
            latest_rows.append(row)
        latest=pd.DataFrame(latest_rows)
        if not latest.empty: latest["__latest_key"]=latest["__season"].round().astype("Int64").astype(str)+"|"+latest["__team"].map(_pt_team_key)
        cache={"status":"PASS","lookup":final_lookup,"latest":latest.set_index("__latest_key",drop=False) if not latest.empty else latest,
               "source_rows":raw_rows,"dedup_rows":dedup_rows,"field_count":len(pre_cols)+3,"metric_count":len(metric_src)}
        _V231_DEEP_CONTEXT_CACHE["deep_source"]=cache
        return cache
    except Exception as exc:
        log_func(f"[NCAAF-DEEP-CONTEXT-V2311-SOURCE] status=UNAVAILABLE error={type(exc).__name__}:{exc} authority=0 fail_closed=TRUE")
        return {"status":"UNAVAILABLE","error":f"{type(exc).__name__}:{exc}","lookup":pd.DataFrame(),"latest":pd.DataFrame(),"source_rows":0}


def _attach_deep_stat_context_to_miner(dashboard_module, miner_games: pd.DataFrame, *, log_func=print, for_live: bool=False) -> tuple[pd.DataFrame,dict[str,Any]]:
    if miner_games is None or miner_games.empty: return miner_games,{"status":"EMPTY","selection_influence":0}
    out=miner_games.copy()
    src=_v231_deep_context_source(dashboard_module,log_func=log_func)
    lookup=src.get("lookup"); latest=src.get("latest")
    if not isinstance(lookup,pd.DataFrame) or lookup.empty:
        return out,{"status":src.get("status","UNAVAILABLE"),"error":src.get("error"),"selection_influence":0}
    home_key=_v214_side_key(out,"Team_Norm" if "Team_Norm" in out.columns else "Home_Team_Norm","Opponent_Norm" if "Opponent_Norm" in out.columns else "Away_Team_Norm")
    road_key=_v214_side_key(out,"Opponent_Norm" if "Opponent_Norm" in out.columns else "Away_Team_Norm","Team_Norm" if "Team_Norm" in out.columns else "Home_Team_Norm")
    seasons=pd.to_numeric(out.get("Season"),errors="coerce")
    tnames=out.get("Team_Norm",out.get("Home_Team_Norm",out.get("Home_Team",pd.Series("",index=out.index)))).map(_pt_team_key)
    onames=out.get("Opponent_Norm",out.get("Away_Team_Norm",out.get("Away_Team",pd.Series("",index=out.index)))).map(_pt_team_key)
    tlatest=seasons.round().astype("Int64").astype(str)+"|"+tnames
    olatest=seasons.round().astype("Int64").astype(str)+"|"+onames

    source_fields=[c for c in lookup.columns if c.startswith(("Deep_","RQ_","CTX_"))]
    exact_home=home_key.isin(lookup.index); exact_road=road_key.isin(lookup.index)
    def _map(keys,latest_keys,src_col):
        exact=keys.isin(lookup.index)
        v=pd.to_numeric(keys.map(lookup[src_col]),errors="coerce")
        # Never fill a missing pregame value on an exact historical game with a
        # season-end/latest state; that would leak future games into early rows.
        # Latest-state fallback is permitted only when the physical game itself
        # is absent from completed history (for example an upcoming live game).
        if isinstance(latest,pd.DataFrame) and not latest.empty and src_col in latest.columns:
            lv=pd.to_numeric(latest_keys.map(latest[src_col]),errors="coerce")
            # Fail closed on unmatched historical rows. Latest completed-team state
            # is only valid for a prospective/live game that has no exact completed
            # game record yet; it must never backfill a 2022-2025 historical miss.
            prospective=seasons.ge(PROSPECTIVE_MIN_SEASON).fillna(False)
            allow_latest=(~exact) & prospective
            v=v.where(~allow_latest,lv)
        return pd.Series(v,index=out.index,dtype=float)
    for src_col in source_fields:
        for prefix,keys,lkeys in (("Team_",home_key,tlatest),("Opp_",road_key,olatest)):
            dst=prefix+src_col
            vals=_map(keys,lkeys,src_col)
            if dst in out.columns:
                cur=pd.to_numeric(out[dst],errors="coerce"); out[dst]=cur.where(cur.notna(),vals)
            else: out[dst]=vals

    def _outnum(col):
        return pd.to_numeric(out[col],errors="coerce") if col in out.columns else pd.Series(np.nan,index=out.index,dtype=float)
    season_arr=_outnum("Season").to_numpy(float)
    hist=np.isfinite(season_arr)&(season_arr<=max(CONFIRMATION_SEASONS))
    hist_n=int(hist.sum())
    result_ready=int((_outnum("Team_RQ_Prev_YPP_Margin").notna().to_numpy()&hist).sum()) if hist_n else 0
    recent_ready=int((_outnum("Team_CTX_Recent3_Net_YPP").notna().to_numpy()&hist).sum()) if hist_n else 0
    resume_ready=int((_outnum("Team_CTX_SOS_WinPct").notna().to_numpy()&hist).sum()) if hist_n else 0
    matchup_ready=int((_outnum("Team_CTX_Season_Off_YPP").notna()&_outnum("Opp_CTX_Season_Def_YPP_Allowed").notna()).to_numpy()[hist].sum()) if hist_n else 0
    diag={"status":"PASS","mode":"LIVE_TRIGGER" if for_live else "HISTORICAL_PLUS_PROSPECTIVE","source_rows":int(src.get("source_rows",0)),"source_rows_dedup":int(src.get("dedup_rows",0)),
          "miner_rows":int(len(out)),"historical_rows":hist_n,"exact_home_matches":int((exact_home.to_numpy()&hist).sum()) if hist_n else int(exact_home.sum()),"exact_road_matches":int((exact_road.to_numpy()&hist).sum()) if hist_n else int(exact_road.sum()),
          "result_quality_ready":result_ready,"recent3_ready":recent_ready,"resume_ready":resume_ready,"matchup_ready":matchup_ready,
          "selection_influence":0,"outcomes_2026_selection":False}
    log_func(f"[NCAAF-DEEP-CONTEXT-V2311-BRIDGE] status=PASS mode={diag['mode']} rows={len(out)} historical={hist_n} exact_home={diag['exact_home_matches']}/{hist_n if hist_n else len(out)} exact_road={diag['exact_road_matches']}/{hist_n if hist_n else len(out)} result_quality_ready={result_ready} recent3_ready={recent_ready} resume_ready={resume_ready} matchup_ready={matchup_ready} outcomes_2026_selection=FALSE authority=0")
    return out,diag


# V2.30 deterministic context bridge. Revenge/H2H, rivalry and schedule sequence
# are reconstructed from chronology. Travel is joined from a season/game reference
# table created by bootstrap_ncaaf_travel_reference.py. These fields are pregame
# context only; 2026 outcomes may update LIVE trigger state but never selection.
NCAAF_TRAVEL_REFERENCE_TABLE = os.getenv("NCAAF_TRAVEL_REFERENCE_TABLE", "sharplogger.sharp_data.ncaaf_game_travel_reference")
_V230_TRAVEL_CACHE: dict[str,Any] = {}

# High-confidence named rivalry pairs. This is intentionally conservative; a
# separate H2H-frequency feature captures recurring series that are not listed.
_V230_RIVALRY_PAIRS = {
    tuple(sorted(x)) for x in [
        ("alabama","auburn"),("alabama","tennessee"),("army","navy"),("air force","army"),("air force","navy"),
        ("ohio state","michigan"),("michigan","michigan state"),("minnesota","wisconsin"),("iowa","iowa state"),("iowa","minnesota"),("iowa","wisconsin"),("iowa","nebraska"),
        ("oklahoma","texas"),("oklahoma","oklahoma state"),("texas","texas a&m"),("baylor","tcu"),("tcu","smu"),("kansas","kansas state"),("kansas","missouri"),
        ("georgia","florida"),("georgia","auburn"),("georgia","georgia tech"),("ole miss","mississippi state"),("tennessee","vanderbilt"),("kentucky","louisville"),
        ("clemson","south carolina"),("florida","florida state"),("florida state","miami fl"),("duke","north carolina"),("north carolina","north carolina state"),("virginia","virginia tech"),("pittsburgh","west virginia"),
        ("usc","ucla"),("usc","notre dame"),("california","stanford"),("oregon","oregon state"),("washington","washington state"),("arizona","arizona state"),("colorado","colorado state"),("utah","byu"),
        ("boise state","fresno state"),("colorado state","wyoming"),("cincinnati","miami oh"),("memphis","uab"),("appalachian state","georgia southern"),("louisiana","louisiana monroe"),
        ("houston","rice"),("utep","new mexico state"),("utsa","texas state"),("fresno state","san jose state"),("nevada","unlv"),("hawaii","fresno state"),
        ("notre dame","navy"),("notre dame","purdue"),("purdue","indiana"),("illinois","northwestern"),("maryland","west virginia"),("syracuse","pittsburgh"),
    ]
}
_V230_TEAM_ENDPOINT_ALIASES = {
    "usc":("usc","southern california"),"ucla":("ucla",),"byu":("byu","brigham young"),"tcu":("tcu","texas christian"),
    "smu":("smu","southern methodist"),"uab":("uab","alabama birmingham"),"utep":("utep","texas el paso"),"utsa":("utsa","texas san antonio"),
    "miami fl":("miami hurricanes","miami fl","miami florida"),"miami oh":("miami redhawks","miami ohio","miami oh"),
    "north carolina state":("north carolina state","nc state"),"louisiana monroe":("louisiana monroe","ul monroe","ulm"),
    "appalachian state":("appalachian state","app state"),"texas a&m":("texas a m","texas am"),
}


def _v230_name(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+"," ",str(v or "").lower()).strip()


def _v230_school_match(team_name: Any, endpoint: str) -> bool:
    t=_v230_name(team_name); ep=_v230_name(endpoint)
    if not t or not ep: return False
    opts=_V230_TEAM_ENDPOINT_ALIASES.get(endpoint,(endpoint,))
    for o in opts:
        q=_v230_name(o)
        if t==q or t.startswith(q+" "): return True
    return False


def _v230_is_rivalry(team: Any, opp: Any) -> bool:
    for a,b in _V230_RIVALRY_PAIRS:
        if (_v230_school_match(team,a) and _v230_school_match(opp,b)) or (_v230_school_match(team,b) and _v230_school_match(opp,a)):
            return True
    return False


def _v230_game_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Chronological H2H/revenge + road-sequence state for home-oriented games."""
    if frame is None or frame.empty: return frame
    z=frame.copy().reset_index(drop=True)
    z["__orig_index_v230"]=np.arange(len(z),dtype=int)
    team_col="Team_Norm" if "Team_Norm" in z.columns else "Home_Team_Norm" if "Home_Team_Norm" in z.columns else "Home_Team"
    opp_col="Opponent_Norm" if "Opponent_Norm" in z.columns else "Away_Team_Norm" if "Away_Team_Norm" in z.columns else "Away_Team"
    z["__team_v230"]=z.get(team_col,pd.Series("",index=z.index)).astype(str).str.lower().str.strip()
    z["__opp_v230"]=z.get(opp_col,pd.Series("",index=z.index)).astype(str).str.lower().str.strip()
    z["__date_v230"]=pd.to_datetime(z.get("Game_Date",z.get("Game_Start",pd.Series(pd.NaT,index=z.index))),errors="coerce",utc=True)
    z["__season_v230"]=pd.to_numeric(z.get("Season"),errors="coerce")
    margin=pd.Series(np.nan,index=z.index,dtype=float)
    for c in ("Actual_Margin","Team_Margin","Margin"):
        if c in z.columns:
            margin=margin.where(margin.notna(),pd.to_numeric(z[c],errors="coerce"))
    if not margin.notna().any():
        pf=pd.to_numeric(z.get("Team_Score",z.get("Points_For",pd.Series(np.nan,index=z.index))),errors="coerce")
        pa=pd.to_numeric(z.get("Opponent_Score",z.get("Points_Against",pd.Series(np.nan,index=z.index))),errors="coerce")
        margin=pf-pa
    z["__margin_v230"]=margin
    # Defaults/fail closed.
    num_cols=["Revenge_Flag","Opp_Revenge_Flag","Days_Since_Last_Matchup","Opp_Days_Since_Last_Matchup","Last_Matchup_Margin","Opp_Last_Matchup_Margin",
              "H2H2_Prior_Meetings","Opp_H2H2_Prior_Meetings","Revenge_Depth","Opp_Revenge_Depth","H2H_Meetings_Since_Last_Win","Opp_H2H_Meetings_Since_Last_Win",
              "H2H_Last_Loss_Margin","Opp_H2H_Last_Loss_Margin","Same_Season_Rematch","Prior_H2H_Home_Road_Flip","Opp_Prior_H2H_Home_Road_Flip",
              "Days_Since_Last_Game","Opp_Days_Since_Last_Game","Team_Road_Games_Last3","Opp_Road_Games_Last3","Team_Road_Games_Last4","Opp_Road_Games_Last4",
              "Team_Back_To_Back_Road","Opp_Back_To_Back_Road","Team_Third_Road_In4","Opp_Third_Road_In4","Rivalry_Flag"]
    for c in num_cols:
        if c not in z.columns: z[c]=np.nan

    # Team schedule sequence. No outcome is used.
    events=[]
    for i,r in z.iterrows():
        sy=r["__season_v230"]; dtv=r["__date_v230"]
        if pd.isna(sy) or pd.isna(dtv): continue
        t=str(r["__team_v230"]); o=str(r["__opp_v230"])
        if t: events.append((t,int(sy),dtv,i,0,"TEAM"))
        if o: events.append((o,int(sy),dtv,i,1,"OPP"))
    ev=pd.DataFrame(events,columns=["team","season","date","game_i","is_road","side"])
    seq={}
    if not ev.empty:
        ev=ev.sort_values(["team","season","date","game_i","side"],kind="stable")
        for (_tm,_sy),gg in ev.groupby(["team","season"],sort=False):
            prev_date=None; prior_roles=[]
            for rr in gg.itertuples(index=False):
                days=(rr.date-prev_date).total_seconds()/86400.0 if prev_date is not None else np.nan
                r3=int(sum(prior_roles[-3:])); r4=int(sum(prior_roles[-4:]))
                seq[(int(rr.game_i),str(rr.side))]={
                    "days":days,"r3":r3,"r4":r4,
                    "b2b":int(bool(rr.is_road) and bool(prior_roles[-1] if prior_roles else 0)),
                    "third4":int(bool(rr.is_road) and r3>=2),
                }
                prior_roles.append(int(rr.is_road)); prev_date=rr.date
    for i in z.index:
        a=seq.get((int(i),"TEAM"),{}); b=seq.get((int(i),"OPP"),{})
        for col,key,src in (("Days_Since_Last_Game","days",a),("Team_Road_Games_Last3","r3",a),("Team_Road_Games_Last4","r4",a),("Team_Back_To_Back_Road","b2b",a),("Team_Third_Road_In4","third4",a),
                            ("Opp_Days_Since_Last_Game","days",b),("Opp_Road_Games_Last3","r3",b),("Opp_Road_Games_Last4","r4",b),("Opp_Back_To_Back_Road","b2b",b),("Opp_Third_Road_In4","third4",b)):
            if key in src: z.at[i,col]=src[key]

    # H2H state. Each row is written before current outcome updates pair history.
    pair_state={}
    order=z.loc[z["__date_v230"].notna()&z["__team_v230"].ne("")&z["__opp_v230"].ne("")].sort_values(["__date_v230","__season_v230"],kind="stable").index
    for i in order:
        t=str(z.at[i,"__team_v230"]); o=str(z.at[i,"__opp_v230"]); dtv=z.at[i,"__date_v230"]; sy=z.at[i,"__season_v230"]
        p=tuple(sorted((t,o))); st=pair_state.get(p,{"meetings":0,"last_date":None,"last_season":None,"last_margin":{},"loss_streak":{},"last_loss_margin":{},"last_home":None})
        meetings=int(st["meetings"] or 0)
        if meetings:
            lm_t=st["last_margin"].get(t,np.nan); lm_o=st["last_margin"].get(o,np.nan)
            z.at[i,"Revenge_Flag"]=int(pd.notna(lm_t) and float(lm_t)<0); z.at[i,"Opp_Revenge_Flag"]=int(pd.notna(lm_o) and float(lm_o)<0)
            days=(dtv-st["last_date"]).total_seconds()/86400.0 if st.get("last_date") is not None else np.nan
            z.at[i,"Days_Since_Last_Matchup"]=days; z.at[i,"Opp_Days_Since_Last_Matchup"]=days
            z.at[i,"Last_Matchup_Margin"]=lm_t; z.at[i,"Opp_Last_Matchup_Margin"]=lm_o
            z.at[i,"H2H2_Prior_Meetings"]=meetings; z.at[i,"Opp_H2H2_Prior_Meetings"]=meetings
            z.at[i,"Revenge_Depth"]=int(st["loss_streak"].get(t,0)); z.at[i,"Opp_Revenge_Depth"]=int(st["loss_streak"].get(o,0))
            z.at[i,"H2H_Meetings_Since_Last_Win"]=int(st["loss_streak"].get(t,0)); z.at[i,"Opp_H2H_Meetings_Since_Last_Win"]=int(st["loss_streak"].get(o,0))
            z.at[i,"H2H_Last_Loss_Margin"]=st["last_loss_margin"].get(t,np.nan); z.at[i,"Opp_H2H_Last_Loss_Margin"]=st["last_loss_margin"].get(o,np.nan)
            z.at[i,"Same_Season_Rematch"]=int(pd.notna(sy) and st.get("last_season")==int(sy))
            last_home=st.get("last_home")
            if last_home:
                z.at[i,"Prior_H2H_Home_Road_Flip"]=int(last_home!=t)
                z.at[i,"Opp_Prior_H2H_Home_Road_Flip"]=int(last_home==o)
        else:
            z.at[i,"H2H2_Prior_Meetings"]=0; z.at[i,"Opp_H2H2_Prior_Meetings"]=0
            z.at[i,"Revenge_Flag"]=0; z.at[i,"Opp_Revenge_Flag"]=0; z.at[i,"Same_Season_Rematch"]=0
        z.at[i,"Rivalry_Flag"]=int(_v230_is_rivalry(t,o))
        m=z.at[i,"__margin_v230"]
        if pd.notna(m):
            m=float(m); st["meetings"]=meetings+1; st["last_date"]=dtv; st["last_season"]=int(sy) if pd.notna(sy) else None; st["last_home"]=t
            st["last_margin"][t]=m; st["last_margin"][o]=-m
            for side,sm in ((t,m),(o,-m)):
                if sm<0:
                    st["loss_streak"][side]=int(st["loss_streak"].get(side,0))+1; st["last_loss_margin"][side]=sm
                elif sm>0:
                    st["loss_streak"][side]=0
                else:
                    st["loss_streak"][side]=0
            pair_state[p]=st
    return z.drop(columns=[c for c in ("__orig_index_v230","__team_v230","__opp_v230","__date_v230","__season_v230","__margin_v230") if c in z.columns])


def _v230_attach_travel_reference(out: pd.DataFrame, dashboard_module=None) -> tuple[pd.DataFrame,dict[str,Any]]:
    if out is None or out.empty: return out,{"status":"EMPTY","matched":0}
    bq=getattr(dashboard_module,"bq_client",None) if dashboard_module is not None else None
    if bq is None: return out,{"status":"NO_BQ_CLIENT","matched":0}
    ref=_V230_TRAVEL_CACHE.get("game_ref")
    if not isinstance(ref,pd.DataFrame):
        try:
            q=bq.query(f"SELECT * FROM `{NCAAF_TRAVEL_REFERENCE_TABLE}` WHERE Season BETWEEN 2022 AND 2026")
            try: ref=q.to_dataframe(create_bqstorage_client=False)
            except TypeError: ref=q.to_dataframe()
            if ref is None or ref.empty: raise RuntimeError("travel reference returned zero rows")
            _V230_TRAVEL_CACHE["game_ref"]=ref.copy()
        except Exception as exc:
            return out,{"status":"REFERENCE_UNAVAILABLE","matched":0,"error":f"{type(exc).__name__}:{exc}","table":NCAAF_TRAVEL_REFERENCE_TABLE}
    ref=_V230_TRAVEL_CACHE["game_ref"]
    # Build all alias pair keys. The bootstrap stores pipe-delimited normalized aliases.
    exact={}
    for _,r in ref.iterrows():
        sy=int(pd.to_numeric(r.get("Season"),errors="coerce")) if pd.notna(pd.to_numeric(r.get("Season"),errors="coerce")) else None
        ds=str(r.get("Game_Date") or "")[:10]
        if sy is None or not ds: continue
        ha=set(str(r.get("Home_Alias_Tokens") or "").split("|")); aa=set(str(r.get("Away_Alias_Tokens") or "").split("|"))
        ha.add(str(r.get("Home_Team_Norm") or "")); aa.add(str(r.get("Away_Team_Norm") or ""))
        ht={_v214_occurrence_token(x) for x in ha if x}; at={_v214_occurrence_token(x) for x in aa if x}
        for h in ht:
            for a in at:
                if h and a: exact[(sy,ds,h,a)]=r
    z=out.copy()
    team_col="Team_Norm" if "Team_Norm" in z.columns else "Home_Team_Norm" if "Home_Team_Norm" in z.columns else "Home_Team"
    opp_col="Opponent_Norm" if "Opponent_Norm" in z.columns else "Away_Team_Norm" if "Away_Team_Norm" in z.columns else "Away_Team"
    dates=pd.to_datetime(z.get("Game_Date",z.get("Game_Start",pd.Series(pd.NaT,index=z.index))),errors="coerce",utc=True).dt.strftime("%Y-%m-%d")
    seasons=pd.to_numeric(z.get("Season"),errors="coerce")
    teams=z.get(team_col,pd.Series("",index=z.index)).map(_v214_occurrence_token); opps=z.get(opp_col,pd.Series("",index=z.index)).map(_v214_occurrence_token)
    fields={
        "Team_Travel_Miles":"Home_Travel_Miles","Opp_Travel_Miles":"Away_Travel_Miles",
        "Team_Time_Zones_Crossed":"Home_Time_Zones_Crossed","Opp_Time_Zones_Crossed":"Away_Time_Zones_Crossed",
        "Team_Travel_Eastward":"Home_Travel_Eastward","Opp_Travel_Eastward":"Away_Travel_Eastward",
        "Team_Travel_Westward":"Home_Travel_Westward","Opp_Travel_Westward":"Away_Travel_Westward",
        "Neutral_Site":"Neutral_Site","Travel_Destination_Resolved":"Destination_Resolved",
    }
    for dst in fields:
        if dst not in z.columns: z[dst]=np.nan
    matched=0; unresolved_neutral=0
    for i in z.index:
        if pd.isna(seasons.loc[i]) or not dates.loc[i]: continue
        r=exact.get((int(seasons.loc[i]),str(dates.loc[i]),str(teams.loc[i]),str(opps.loc[i])))
        if r is None: continue
        matched+=1
        for dst,src in fields.items():
            v=r.get(src,np.nan)
            if isinstance(v,(bool,np.bool_)): z.at[i,dst]=int(v)
            else: z.at[i,dst]=pd.to_numeric(pd.Series([v]),errors="coerce").iloc[0]
        if bool(r.get("Neutral_Site")) and not bool(r.get("Destination_Resolved")): unresolved_neutral+=1
    return z,{"status":"PASS","table":NCAAF_TRAVEL_REFERENCE_TABLE,"reference_rows":int(len(ref)),"matched":matched,"miner_rows":int(len(z)),"unresolved_neutral":unresolved_neutral}


def _attach_h2h_rivalry_travel_context(dashboard_module, miner_games: pd.DataFrame, *, log_func=print, for_live: bool=False) -> tuple[pd.DataFrame,dict[str,Any]]:
    if miner_games is None or miner_games.empty: return miner_games,{"status":"EMPTY","selection_influence":0}
    original=miner_games.copy().reset_index(drop=True)
    original["__return_index_v230"]=np.arange(len(original),dtype=int)
    work=original
    history_rows=0
    if for_live:
        bq=getattr(dashboard_module,"bq_client",None) if dashboard_module is not None else None
        view=getattr(dashboard_module,"HISTORICAL_NCAAF_CORE_VIEW","sharplogger.sharp_data.ncaaf_historical_core_training_vw") if dashboard_module is not None else "sharplogger.sharp_data.ncaaf_historical_core_training_vw"
        if bq is not None:
            try:
                q=bq.query(f"SELECT * FROM `{view}` WHERE Historical_Core_Eligible=1 AND Season <= 2026")
                try: h=q.to_dataframe(create_bqstorage_client=False)
                except TypeError: h=q.to_dataframe()
                if h is not None and not h.empty:
                    hh=h.copy(); ih=pd.to_numeric(hh.get("Is_Home"),errors="coerce")
                    if ih.eq(1).any(): hh=hh.loc[ih.eq(1)].copy()
                    hh["Actual_Margin"]=pd.to_numeric(hh.get("Team_Score",hh.get("Points_For")),errors="coerce")-pd.to_numeric(hh.get("Opponent_Score",hh.get("Points_Against")),errors="coerce")
                    hh=hh.loc[pd.to_numeric(hh["Actual_Margin"],errors="coerce").notna()].copy()
                    hh["__return_index_v230"]=np.arange(-len(hh),0,dtype=int)
                    keep=list(dict.fromkeys(list(hh.columns)+list(original.columns)))
                    work=pd.concat([hh.reindex(columns=keep),original.reindex(columns=keep)],ignore_index=True,sort=False)
                    history_rows=int(len(hh))
            except Exception:
                pass
    enriched=_v230_game_context(work)
    # In live mode retain only the current rows; heavy mode retains all rows.
    if for_live:
        want=set(original["__return_index_v230"].tolist())
        enriched=enriched.loc[enriched["__return_index_v230"].isin(want)].copy()
        enriched=enriched.sort_values("__return_index_v230",kind="stable")
    enriched,travel_diag=_v230_attach_travel_reference(enriched,dashboard_module=dashboard_module)
    enriched=enriched.set_index("__return_index_v230",drop=True)
    enriched=enriched.reindex(original["__return_index_v230"].tolist())
    enriched.index=miner_games.index
    h2h_ready=int(pd.to_numeric(enriched.get("H2H2_Prior_Meetings"),errors="coerce").gt(0).sum())
    revenge_ready=int(pd.to_numeric(enriched.get("Revenge_Flag"),errors="coerce").eq(1).sum())+int(pd.to_numeric(enriched.get("Opp_Revenge_Flag"),errors="coerce").eq(1).sum())
    rivalry_ready=int(pd.to_numeric(enriched.get("Rivalry_Flag"),errors="coerce").eq(1).sum())
    rest_ready=int(pd.to_numeric(enriched.get("Days_Since_Last_Game"),errors="coerce").notna().sum())
    travel_ready=int(pd.to_numeric(enriched.get("Opp_Travel_Miles"),errors="coerce").notna().sum())
    diag={"status":"PASS","mode":"LIVE_TRIGGER" if for_live else "HISTORICAL_PLUS_PROSPECTIVE","rows":int(len(enriched)),"history_rows":history_rows,
          "h2h_prior_ready":h2h_ready,"revenge_flags":revenge_ready,"rivalry_rows":rivalry_ready,"rest_ready":rest_ready,"travel_ready":travel_ready,
          "travel_reference":travel_diag,"selection_influence":0,"outcomes_2026_selection":False}
    log_func(f"[NCAAF-CONTEXT-V230-BRIDGE] status=PASS mode={diag['mode']} rows={len(enriched)} h2h_prior_ready={h2h_ready} revenge_flags={revenge_ready} rivalry_rows={rivalry_ready} rest_ready={rest_ready} travel_ready={travel_ready} travel_ref={travel_diag.get('status')} travel_matched={travel_diag.get('matched',0)}/{len(enriched)} outcomes_2026_selection=FALSE authority=0")
    return enriched,diag


# ---------------------------------------------------------------------------
# System Miner V3 — fixed discovery / untouched confirmation / dependence collapse
# ---------------------------------------------------------------------------
def _extended_atoms(g: pd.DataFrame, dashboard_module=None, *, for_live: bool=False, market: str | None=None,
                    enforce_support_floor: bool=True) -> list[dict[str,Any]]:
    """Leak-safe NCAAF System Miner atom catalog.

    Historical mode applies support floors so sparse identities cannot flood the
    search. Live mode materializes the same named atoms without sample-size
    filtering so a frozen/confirmed rule can be evaluated on one current game.
    Unknown/unavailable inputs fail closed (the atom exists only when its source
    field is actually present).
    """
    # Materialize deterministic Pathi/Big Al flags directly on the Miner frame.
    # The dashboard's historical W/L engine uses a richer state frame, while the
    # Miner cache is intentionally lean.  Without this bridge the systems can be
    # graded correctly yet appear as zero Miner atoms.
    _mkt0=str(market or "").lower().strip()
    if _mkt0 in {"", "spreads"} and dashboard_module is not None:
        try:
            _need=not any(str(c).startswith(("Pathi_FB_","BigAl_CF")) for c in g.columns)
            if _need:
                _eg=g.copy()
                if "Sport" not in _eg.columns: _eg["Sport"]="NCAAF"
                if "Market" not in _eg.columns: _eg["Market"]="spreads"
                if "Value" not in _eg.columns:
                    for _c in ("Spread_Value","Current_Spread","Consensus_Open_Spread","Opening_Spread"):
                        if _c in _eg.columns: _eg["Value"]=pd.to_numeric(_eg[_c],errors="coerce"); break
                if "Spread_Value" not in _eg.columns and "Value" in _eg.columns: _eg["Spread_Value"]=pd.to_numeric(_eg["Value"],errors="coerce")
                if "Opening_Spread" not in _eg.columns and "Consensus_Open_Spread" in _eg.columns: _eg["Opening_Spread"]=pd.to_numeric(_eg["Consensus_Open_Spread"],errors="coerce")
                if "Current_Total" not in _eg.columns:
                    for _c in ("Total_Value","Consensus_Open_Total","Opening_Total"):
                        if _c in _eg.columns: _eg["Current_Total"]=pd.to_numeric(_eg[_c],errors="coerce"); break
                if "Is_Regular_Season" not in _eg.columns: _eg["Is_Regular_Season"]=1
                if "Team_Game_Number" not in _eg.columns:
                    for _c in ("Team_Game_Number_Prior","Context_Team_Games_Prior","Game_Number_Prior"):
                        if _c in _eg.columns:
                            _eg["Team_Game_Number"]=pd.to_numeric(_eg[_c],errors="coerce")+1; break
                if "Revenge_Flag_Current" not in _eg.columns:
                    for _c in ("Revenge_Flag","Revenge_Flag_CurrentOrPriorSeason"):
                        if _c in _eg.columns: _eg["Revenge_Flag_Current"]=pd.to_numeric(_eg[_c],errors="coerce"); break
                if hasattr(dashboard_module,"add_pathi_football_key_features"):
                    try:
                        _eg=dashboard_module.add_pathi_football_key_features(_eg)
                    except Exception:
                        pass
                # Preserve successful Pathi materialization even if the broader
                # multi-sport Big Al rule builder cannot run on the lean frame.
                g=_eg
                if hasattr(dashboard_module,"add_pathi_bigal_rule_flags"):
                    try:
                        g=dashboard_module.add_pathi_bigal_rule_flags(g)
                    except Exception:
                        pass
        except Exception:
            # Fail closed: exact side-bridge / legacy Miner atoms remain available.
            pass

    atoms=[]; names=set()

    # Keep the legacy dashboard atom catalog when available for exact continuity.
    # Live evaluation cannot rely on it because the legacy helper intentionally
    # suppresses atoms with fewer than 20 historical matches, so every important
    # atom is also reproduced below in this module.
    if (not for_live) and dashboard_module is not None and hasattr(dashboard_module,"_v1355_system_atoms"):
        try:
            for a in dashboard_module._v1355_system_atoms(g):
                nm=str(a["name"])
                if nm in names: continue
                atoms.append({"name":nm,"family":str(a["family"]),"mask":np.asarray(a["mask"],dtype=bool),"description":str(a.get("description") or nm)})
                names.add(nm)
        except Exception:
            pass

    def has(*cols): return any(c in g.columns for c in cols)
    def nfirst(*cols):
        out=pd.Series(np.nan,index=g.index,dtype=float)
        for c in cols:
            if c not in g.columns: continue
            v=pd.to_numeric(g[c],errors="coerce")
            out=out.where(out.notna(),v)
        return out
    def tfirst(*cols):
        out=pd.Series("",index=g.index,dtype=str)
        for c in cols:
            if c not in g.columns: continue
            v=g[c].astype(str).str.upper().str.strip().replace({"NAN":"","NONE":"","<NA>":""})
            out=out.where(out.ne(""),v)
        return out
    def add(name,family,mask,desc=None,min_n=30,source_ok=True):
        if name in names or not source_ok: return
        mm=pd.Series(mask,index=g.index).fillna(False).astype(bool).to_numpy()
        if for_live or (not enforce_support_floor) or (min_n<=int(mm.sum())<len(g)):
            atoms.append({"name":name,"family":family,"mask":mm,"description":desc or name}); names.add(name)

    sp=nfirst("Consensus_Open_Spread","Opening_Spread")
    tot=nfirst("Consensus_Open_Total","Opening_Total")
    week=nfirst("Context_Week","Week")
    game_no=nfirst("Team_Game_Number_Prior","Game_Number_Prior","Context_Team_Games_Prior")
    is_home=nfirst("Is_Home")

    # V2.26 expert-observation context bridge. These are generic pregame states,
    # not Big Al-specific rules. Historical close/current prices are pre-kickoff
    # market observations; live uses the current Move Master quote when present.
    current_sp=nfirst("Spread_Value","Current_Spread")
    current_tot=nfirst("Current_Total","Total_Value")
    # Never use closing lines to explain a bet graded at an earlier/opening line.
    # Prefer explicit leakage-safe as-of movement fields; otherwise derive from
    # a current snapshot only. If neither exists historically, movement atoms are
    # unavailable and the Miner correctly cannot qualify them yet.
    spread_move=nfirst("Line_Move_From_Open","Spread_Move_From_Open")
    if not spread_move.notna().any() and current_sp.notna().any() and sp.notna().any(): spread_move=current_sp-sp
    total_move=nfirst("Total_Move_From_Open")
    if not total_move.notna().any() and current_tot.notna().any() and tot.notna().any(): total_move=current_tot-tot

    prev_pf=nfirst("Prev_Team_Score","Pregame_Prev_Team_Score","Context_Prev_Team_Score","Prev_Points_For")
    prev_pa=nfirst("Prev_Opponent_Score","Pregame_Prev_Opponent_Score","Context_Prev_Opponent_Score","Prev_Points_Against")
    opp_prev_pf=nfirst("Opp_Prev_Team_Score","Opp_Pregame_Prev_Team_Score","Context_Opp_Prev_Team_Score","Opp_Prev_Points_For")
    opp_prev_pa=nfirst("Opp_Prev_Opponent_Score","Opp_Pregame_Prev_Opponent_Score","Context_Opp_Prev_Opponent_Score","Opp_Prev_Points_Against")
    team_ats_pct_prior=nfirst("ATS_WinPct_Prior","Team_ATS_WinPct_Prior","ATS_CoverPct_Prior")
    opp_ats_pct_prior=nfirst("Opp_ATS_WinPct_Prior","Opponent_ATS_WinPct_Prior","Opp_ATS_CoverPct_Prior")
    derived_prev_dog=pd.Series(np.nan,index=g.index,dtype=float)
    derived_opp_prev_dog=pd.Series(np.nan,index=g.index,dtype=float)

    # If the cache does not already expose prior score/ATS-record fields, derive
    # them by shifting completed game rows inside team-season chronology. Only
    # prior games are used; current outcomes never enter the current row's state.
    if (not for_live) and any(c in g.columns for c in ("Team_Norm","Team")) and "Season" in g.columns:
        _need_prior=not (prev_pf.notna().any() and prev_pa.notna().any() and team_ats_pct_prior.notna().any())
        if _need_prior:
            try:
                _team=tfirst("Team_Norm","Team")
                _opp=tfirst("Opponent_Norm","Opponent")
                _season=nfirst("Season")
                _date=pd.to_datetime(g.get("Game_Date",g.get("Game_Start",pd.Series(pd.NaT,index=g.index))),errors="coerce",utc=True)
                _pf=nfirst("Team_Score","Points_For")
                _pa=nfirst("Opponent_Score","Points_Against")
                _am=nfirst("Actual_Margin")
                if not _am.notna().any(): _am=_pf-_pa
                _ats_margin=_am+sp
                _tmp=pd.DataFrame({"idx":np.arange(len(g)),"team":_team,"opp":_opp,"season":_season,"date":_date,"pf":_pf,"pa":_pa,"dog":sp.gt(0).astype(float),"ats_margin":_ats_margin})
                _tmp=_tmp.sort_values(["team","season","date","idx"],kind="stable")
                _tmp["prev_pf"]=_tmp.groupby(["team","season"],sort=False,dropna=False)["pf"].shift(1)
                _tmp["prev_pa"]=_tmp.groupby(["team","season"],sort=False,dropna=False)["pa"].shift(1)
                _tmp["prev_dog"]=_tmp.groupby(["team","season"],sort=False,dropna=False)["dog"].shift(1)
                _tmp["ats_pct_prior"]=np.nan
                for (_t,_sy),_gg in _tmp.groupby(["team","season"],sort=False,dropna=False):
                    wins=0; graded=0
                    for _ix in _gg.index:
                        _tmp.at[_ix,"ats_pct_prior"]=(wins/graded) if graded else np.nan
                        _m=_tmp.at[_ix,"ats_margin"]
                        if pd.notna(_m) and not np.isclose(float(_m),0.0,atol=1e-9):
                            graded+=1; wins+=int(float(_m)>0)
                _byidx=_tmp.set_index("idx")
                def _restore(col):
                    return pd.Series([_byidx.at[i,col] if i in _byidx.index else np.nan for i in range(len(g))],index=g.index,dtype=float)
                if not prev_pf.notna().any(): prev_pf=_restore("prev_pf")
                if not prev_pa.notna().any(): prev_pa=_restore("prev_pa")
                if not team_ats_pct_prior.notna().any(): team_ats_pct_prior=_restore("ats_pct_prior")
                derived_prev_dog=_restore("prev_dog")
                # Mirror the opponent's prior state from its row in the same physical game.
                _lookup={}
                for _i in range(len(g)):
                    _k=(str(_season.iloc[_i]),str(_date.iloc[_i]),str(_team.iloc[_i]))
                    _lookup[_k]={"pf":prev_pf.iloc[_i],"pa":prev_pa.iloc[_i],"ats":team_ats_pct_prior.iloc[_i],"dog":derived_prev_dog.iloc[_i]}
                _opf=[]; _opa=[]; _oats=[]; _odog=[]
                for _i in range(len(g)):
                    _v=_lookup.get((str(_season.iloc[_i]),str(_date.iloc[_i]),str(_opp.iloc[_i])),{})
                    _opf.append(_v.get("pf",np.nan)); _opa.append(_v.get("pa",np.nan)); _oats.append(_v.get("ats",np.nan)); _odog.append(_v.get("dog",np.nan))
                if not opp_prev_pf.notna().any(): opp_prev_pf=pd.Series(_opf,index=g.index,dtype=float)
                if not opp_prev_pa.notna().any(): opp_prev_pa=pd.Series(_opa,index=g.index,dtype=float)
                if not opp_ats_pct_prior.notna().any(): opp_ats_pct_prior=pd.Series(_oats,index=g.index,dtype=float)
                derived_opp_prev_dog=pd.Series(_odog,index=g.index,dtype=float)
            except Exception:
                pass

    # Core market role / price regimes.
    add("HOME","VENUE",is_home.eq(1),source_ok=has("Is_Home"))
    add("ROAD","VENUE",is_home.eq(0),source_ok=has("Is_Home"))
    add("CURRENT_DOG","MARKET_ROLE",sp.gt(0),source_ok=has("Consensus_Open_Spread","Opening_Spread"))
    add("CURRENT_FAVORITE","MARKET_ROLE",sp.lt(0),source_ok=has("Consensus_Open_Spread","Opening_Spread"))
    for lo,hi in ((0,3),(3,7),(7,10),(10,14),(14,99)):
        add(f"DOG_{lo}_{hi}","MARKET_PRICE",sp.gt(lo)&sp.le(hi),source_ok=has("Consensus_Open_Spread","Opening_Spread"))
        add(f"FAV_{lo}_{hi}","MARKET_PRICE",(-sp).gt(lo)&(-sp).le(hi),source_ok=has("Consensus_Open_Spread","Opening_Spread"))
    # V2.23 season-record/desperation pricing vocabulary. Spread only.
    # All thresholds share MARKET_PRICE so correlated cut points cannot stack
    # inside a single mined rule.
    if market=="spreads":
        for cut,label in ((6.5,"6P5"),(7.0,"7"),(8.0,"8"),(9.0,"9"),(10.0,"10")):
            add(f"DOG_{label}_PLUS","MARKET_PRICE",sp.ge(cut),source_ok=has("Consensus_Open_Spread","Opening_Spread"),min_n=20)
        _spread_move_src=bool(spread_move.notna().any())
        add("SPREAD_MOVED_TOWARD_TEAM_1_PLUS","OBS_SPREAD_MARKET_DIRECTION",spread_move.le(-1.0),desc="current/close spread moved >=1 point toward team from opener",source_ok=_spread_move_src,min_n=20)
        add("SPREAD_MOVED_TOWARD_TEAM_2_PLUS","OBS_SPREAD_MARKET_DIRECTION",spread_move.le(-2.0),desc="current/close spread moved >=2 points toward team from opener",source_ok=_spread_move_src,min_n=20)
        add("SPREAD_MOVED_AWAY_FROM_TEAM_1_PLUS","OBS_SPREAD_MARKET_DIRECTION",spread_move.ge(1.0),desc="current/close spread moved >=1 point away from team from opener",source_ok=_spread_move_src,min_n=20)
        add("SPREAD_MOVED_AWAY_FROM_TEAM_2_PLUS","OBS_SPREAD_MARKET_DIRECTION",spread_move.ge(2.0),desc="current/close spread moved >=2 points away from team from opener",source_ok=_spread_move_src,min_n=20)
        _flip_src=bool(sp.notna().any() and current_sp.notna().any())
        add("OPENED_DOG_NOW_FAVORITE","OBS_SPREAD_MARKET_DIRECTION",sp.gt(0)&current_sp.lt(0),desc="home-oriented team opened underdog and is now favorite across zero",source_ok=_flip_src,min_n=20)
        add("OPENED_FAVORITE_NOW_DOG","OBS_SPREAD_MARKET_DIRECTION",sp.lt(0)&current_sp.gt(0),desc="home-oriented team opened favorite and is now underdog across zero",source_ok=_flip_src,min_n=20)
        add("SPREAD_CROSSED_ZERO","OBS_SPREAD_MARKET_DIRECTION",((sp.gt(0)&current_sp.lt(0))|(sp.lt(0)&current_sp.gt(0))),desc="market crossed pick'em from opener to current quote",source_ok=_flip_src,min_n=20)
    for lo,hi in ((0,45),(45,52),(52,60),(60,99)):
        add(f"TOTAL_{lo}_{hi}","TOTAL_REGIME",tot.ge(lo)&tot.lt(hi),source_ok=has("Consensus_Open_Total","Opening_Total"))
    if market=="totals":
        add("TOTAL_55_PLUS","TOTAL_REGIME",tot.ge(55),source_ok=has("Consensus_Open_Total","Opening_Total"),min_n=20)
        add("TOTAL_58_PLUS","TOTAL_REGIME",tot.ge(58),source_ok=has("Consensus_Open_Total","Opening_Total"),min_n=20)
        _total_move_src=bool(total_move.notna().any())
        for _cut,_lbl in ((1.0,"1"),(2.0,"2"),(3.0,"3")):
            add(f"TOTAL_MOVED_DOWN_{_lbl}_PLUS","OBS_TOTAL_MARKET_DIRECTION",total_move.le(-_cut),desc=f"current/close total moved down >= {_cut:g} from opener",source_ok=_total_move_src,min_n=20)
            add(f"TOTAL_MOVED_UP_{_lbl}_PLUS","OBS_TOTAL_MARKET_DIRECTION",total_move.ge(_cut),desc=f"current/close total moved up >= {_cut:g} from opener",source_ok=_total_move_src,min_n=20)

    # Timing / schedule / rest.
    add("EARLY_SEASON_WK1_4","SEASON_TIMING",week.between(1,4),source_ok=has("Context_Week","Week"))
    add("MID_SEASON_WK5_9","SEASON_TIMING",week.between(5,9),source_ok=has("Context_Week","Week"))
    add("LATE_SEASON_WK10_PLUS","SEASON_TIMING",week.ge(10),source_ok=has("Context_Week","Week"))
    add("FIRST_THREE_TEAM_GAMES","SEASON_TIMING",game_no.le(3)&game_no.notna(),source_ok=has("Team_Game_Number_Prior","Game_Number_Prior","Context_Team_Games_Prior"))
    add("GAME_7_PLUS","SEASON_TIMING",game_no.ge(7),source_ok=has("Team_Game_Number_Prior","Game_Number_Prior","Context_Team_Games_Prior"))
    # Team_Game_Number_Prior is completed games entering this row; +1 is
    # the current game number. Exact slots share SEASON_TIMING to avoid duplicate
    # timing votes (for example TEAM_GAME_5 plus MID_SEASON).
    if market=="spreads":
        _current_game_no=game_no+1
        for _n in (3,4,5,6):
            add(f"TEAM_GAME_{_n}","SEASON_TIMING",_current_game_no.eq(_n),source_ok=has("Team_Game_Number_Prior","Game_Number_Prior","Context_Team_Games_Prior"),min_n=20)
    rest=nfirst("Days_Since_Last_Game_System","Days_Since_Last_Game")
    opprest=nfirst("Opp_Days_Since_Last_Game_System","Opp_Days_Since_Last_Game")
    rest_pair=rest.notna()&opprest.notna()
    if for_live or (int(rest_pair.sum())>=100 and (rest[rest_pair]-opprest[rest_pair]).abs().gt(0).any()):
        add("SHORT_REST_6_OR_LESS","REST",rest_pair&rest.le(6),source_ok=has("Days_Since_Last_Game_System","Days_Since_Last_Game"))
        add("REST_8_PLUS","REST",rest_pair&rest.ge(8),source_ok=has("Days_Since_Last_Game_System","Days_Since_Last_Game"))
        add("REST_ADV_2_PLUS","REST",rest_pair&(rest-opprest).ge(2),source_ok=has("Days_Since_Last_Game_System","Days_Since_Last_Game") and has("Opp_Days_Since_Last_Game_System","Opp_Days_Since_Last_Game"))
        add("REST_DISADV_2_PLUS","REST",rest_pair&(rest-opprest).le(-2),source_ok=has("Days_Since_Last_Game_System","Days_Since_Last_Game") and has("Opp_Days_Since_Last_Game_System","Opp_Days_Since_Last_Game"))

    # V2.30 deterministic travel / road-sequence context. Travel comes from the
    # season/game reference table; sequence is reconstructed from known schedule
    # chronology. Missing reference rows simply make these atoms unavailable.
    tmiles=nfirst("Team_Travel_Miles","Travel_Miles")
    omiles=nfirst("Opp_Travel_Miles","Opponent_Travel_Miles")
    _travel_src=bool(tmiles.notna().any() or omiles.notna().any())
    for _cut in (500,1000,1500):
        add(f"TEAM_TRAVEL_{_cut}_PLUS","TRAVEL_CONTEXT",tmiles.ge(_cut),source_ok=tmiles.notna().any(),min_n=20)
        add(f"OPP_TRAVEL_{_cut}_PLUS","TRAVEL_CONTEXT",omiles.ge(_cut),source_ok=omiles.notna().any(),min_n=20)
    _tdiff=tmiles-omiles
    add("TEAM_TRAVEL_DISADV_500_PLUS","TRAVEL_CONTEXT",_tdiff.ge(500),source_ok=_travel_src,min_n=20)
    add("TEAM_TRAVEL_ADV_500_PLUS","TRAVEL_CONTEXT",_tdiff.le(-500),source_ok=_travel_src,min_n=20)
    ttz=nfirst("Team_Time_Zones_Crossed"); otz=nfirst("Opp_Time_Zones_Crossed")
    add("TEAM_CROSSES_2_TZ_PLUS","TRAVEL_CONTEXT",ttz.ge(2),source_ok=ttz.notna().any(),min_n=20)
    add("OPP_CROSSES_2_TZ_PLUS","TRAVEL_CONTEXT",otz.ge(2),source_ok=otz.notna().any(),min_n=20)
    add("TEAM_TRAVEL_EASTWARD","TRAVEL_CONTEXT",nfirst("Team_Travel_Eastward").eq(1),source_ok=has("Team_Travel_Eastward"),min_n=20)
    add("TEAM_TRAVEL_WESTWARD","TRAVEL_CONTEXT",nfirst("Team_Travel_Westward").eq(1),source_ok=has("Team_Travel_Westward"),min_n=20)
    add("OPP_TRAVEL_EASTWARD","TRAVEL_CONTEXT",nfirst("Opp_Travel_Eastward").eq(1),source_ok=has("Opp_Travel_Eastward"),min_n=20)
    add("OPP_TRAVEL_WESTWARD","TRAVEL_CONTEXT",nfirst("Opp_Travel_Westward").eq(1),source_ok=has("Opp_Travel_Westward"),min_n=20)
    add("TEAM_BACK_TO_BACK_ROAD","ROAD_SEQUENCE",nfirst("Team_Back_To_Back_Road").eq(1),source_ok=has("Team_Back_To_Back_Road"),min_n=20)
    add("OPP_BACK_TO_BACK_ROAD","ROAD_SEQUENCE",nfirst("Opp_Back_To_Back_Road").eq(1),source_ok=has("Opp_Back_To_Back_Road"),min_n=20)
    add("TEAM_2PLUS_ROAD_LAST3","ROAD_SEQUENCE",nfirst("Team_Road_Games_Last3").ge(2),source_ok=has("Team_Road_Games_Last3"),min_n=20)
    add("OPP_2PLUS_ROAD_LAST3","ROAD_SEQUENCE",nfirst("Opp_Road_Games_Last3").ge(2),source_ok=has("Opp_Road_Games_Last3"),min_n=20)
    add("TEAM_THIRD_ROAD_IN4","ROAD_SEQUENCE",nfirst("Team_Third_Road_In4").eq(1),source_ok=has("Team_Third_Road_In4"),min_n=20)
    add("OPP_THIRD_ROAD_IN4","ROAD_SEQUENCE",nfirst("Opp_Third_Road_In4").eq(1),source_ok=has("Opp_Third_Road_In4"),min_n=20)

    # Rivalry / matchup history / revenge.
    riv=tfirst("Rivalry_Flag","Is_Rivalry","Context_Rivalry_Flag")
    if has("Rivalry_Flag","Is_Rivalry","Context_Rivalry_Flag"):
        rv=nfirst("Rivalry_Flag","Is_Rivalry","Context_Rivalry_Flag")
        add("RIVALRY","RIVALRY",rv.eq(1),source_ok=True,min_n=20)
    rev=nfirst("Revenge_Flag","Opp_Revenge_Flag_CurrentOrPriorSeason")
    add("REVENGE","MATCHUP_HISTORY",rev.eq(1),source_ok=has("Revenge_Flag","Opp_Revenge_Flag_CurrentOrPriorSeason"),min_n=20)
    h2hd=nfirst("Days_Since_Last_Matchup","Days_Since_Last_Matchup_System","H2H2_Days_Since")
    add("RECENT_H2H_730D","MATCHUP_HISTORY",h2hd.le(730)&h2hd.notna(),source_ok=has("Days_Since_Last_Matchup","Days_Since_Last_Matchup_System","H2H2_Days_Since"),min_n=20)
    h2hm=nfirst("Last_Matchup_Margin","Last_Matchup_SU_Margin_System","H2H2_Prior_Margin_Current_Orientation")
    add("PRIOR_H2H_LOSS","MATCHUP_HISTORY",h2hm.lt(0),source_ok=has("Last_Matchup_Margin","Last_Matchup_SU_Margin_System","H2H2_Prior_Margin_Current_Orientation"),min_n=20)
    add("PRIOR_H2H_WIN","MATCHUP_HISTORY",h2hm.gt(0),source_ok=has("Last_Matchup_Margin","Last_Matchup_SU_Margin_System","H2H2_Prior_Margin_Current_Orientation"),min_n=20)
    meetings=nfirst("H2H2_Prior_Meetings","H2H_Prior_Meetings_Research")
    add("H2H_2_PLUS_MEETINGS","MATCHUP_HISTORY",meetings.ge(2),source_ok=has("H2H2_Prior_Meetings","H2H_Prior_Meetings_Research"),min_n=20)
    depth=nfirst("Revenge_Depth","H2H_Meetings_Since_Last_Win")
    add("REVENGE_DEPTH_2_PLUS","MATCHUP_HISTORY",depth.ge(2),source_ok=has("Revenge_Depth"),min_n=20)
    add("H2H_3_PLUS_SINCE_WIN","MATCHUP_HISTORY",nfirst("H2H_Meetings_Since_Last_Win").ge(3),source_ok=has("H2H_Meetings_Since_Last_Win"),min_n=20)
    last_loss=nfirst("H2H_Last_Loss_Margin")
    add("H2H_LAST_LOSS_14_PLUS","MATCHUP_HISTORY",last_loss.le(-14),source_ok=has("H2H_Last_Loss_Margin"),min_n=20)
    add("H2H_LAST_LOSS_7_PLUS","MATCHUP_HISTORY",last_loss.le(-7),source_ok=has("H2H_Last_Loss_Margin"),min_n=20)
    # Road-side mirror. Same MATCHUP_HISTORY family, therefore correlated variants
    # cannot become multiple independent votes.
    ore=nfirst("Opp_Revenge_Flag")
    add("OPP_REVENGE","MATCHUP_HISTORY",ore.eq(1),source_ok=has("Opp_Revenge_Flag"),min_n=20)
    oh2hm=nfirst("Opp_Last_Matchup_Margin")
    add("OPP_PRIOR_H2H_LOSS","MATCHUP_HISTORY",oh2hm.lt(0),source_ok=has("Opp_Last_Matchup_Margin"),min_n=20)
    add("OPP_PRIOR_H2H_WIN","MATCHUP_HISTORY",oh2hm.gt(0),source_ok=has("Opp_Last_Matchup_Margin"),min_n=20)
    add("OPP_H2H_LAST_LOSS_7_PLUS","MATCHUP_HISTORY",nfirst("Opp_H2H_Last_Loss_Margin").le(-7),source_ok=has("Opp_H2H_Last_Loss_Margin"),min_n=20)
    add("OPP_REVENGE_DEPTH_2_PLUS","MATCHUP_HISTORY",nfirst("Opp_Revenge_Depth").ge(2),source_ok=has("Opp_Revenge_Depth"),min_n=20)
    add("SAME_SEASON_REMATCH","MATCHUP_HISTORY",nfirst("Same_Season_Rematch").eq(1),source_ok=has("Same_Season_Rematch"),min_n=20)
    add("TEAM_H2H_ROLE_FLIP","MATCHUP_HISTORY",nfirst("Prior_H2H_Home_Road_Flip").eq(1),source_ok=has("Prior_H2H_Home_Road_Flip"),min_n=20)
    add("OPP_H2H_ROLE_FLIP","MATCHUP_HISTORY",nfirst("Opp_Prior_H2H_Home_Road_Flip").eq(1),source_ok=has("Opp_Prior_H2H_Home_Road_Flip"),min_n=20)

    # Model-state / market-relative disagreement regimes. Historical values are OOF.
    stat_edge=nfirst("_V1355_STAT_EDGE_POINTS")
    add("STAT_EDGE_2_PLUS","MODEL_STATE",stat_edge.ge(2),source_ok=has("_V1355_STAT_EDGE_POINTS"))
    add("STAT_EDGE_2_MINUS","MODEL_STATE",stat_edge.le(-2),source_ok=has("_V1355_STAT_EDGE_POINTS"))
    add("STAT_EDGE_ABS_4_PLUS","MODEL_STATE",stat_edge.abs().ge(4),source_ok=has("_V1355_STAT_EDGE_POINTS"))
    h2h_gap=nfirst("_V1355_H2H_STAT_MINUS_MARKET")
    add("H2H_STAT_OVER_MARKET_5P","MODEL_STATE",h2h_gap.ge(.05),source_ok=has("_V1355_H2H_STAT_MINUS_MARKET"))
    add("H2H_STAT_UNDER_MARKET_5P","MODEL_STATE",h2h_gap.le(-.05),source_ok=has("_V1355_H2H_STAT_MINUS_MARKET"))
    total_gap=nfirst("_V1355_TOTAL_EDGE_POINTS")
    add("TOTAL_MODEL_OVER_4","MODEL_STATE",total_gap.ge(4),source_ok=has("_V1355_TOTAL_EDGE_POINTS"))
    add("TOTAL_MODEL_UNDER_4","MODEL_STATE",total_gap.le(-4),source_ok=has("_V1355_TOTAL_EDGE_POINTS"))

    # One-game prior state and magnitude.
    psu=nfirst("Pregame_Prev_SU_Margin","Prev_SU_Margin","Context_Prev_SU_Margin")
    if not psu.notna().any() and prev_pf.notna().any() and prev_pa.notna().any(): psu=prev_pf-prev_pa
    pats=nfirst("Pregame_Prev_ATS_Margin","Prev_ATS_Margin","Prev_ATS_Cover_Margin","Context_Prev_ATS_Margin")
    add("OFF_SU_WIN","PRIOR_RESULT",psu.gt(0),source_ok=has("Pregame_Prev_SU_Margin","Prev_SU_Margin","Context_Prev_SU_Margin"))
    add("OFF_SU_LOSS","PRIOR_RESULT",psu.lt(0),source_ok=has("Pregame_Prev_SU_Margin","Prev_SU_Margin","Context_Prev_SU_Margin"))
    for t in (7,14,21,28):
        add(f"OFF_SU_WIN_{t}_PLUS","PRIOR_MARGIN_MAGNITUDE",psu.ge(t),source_ok=has("Pregame_Prev_SU_Margin","Prev_SU_Margin","Context_Prev_SU_Margin"))
        add(f"OFF_SU_LOSS_{t}_PLUS","PRIOR_MARGIN_MAGNITUDE",psu.le(-t),source_ok=has("Pregame_Prev_SU_Margin","Prev_SU_Margin","Context_Prev_SU_Margin"))
    for t in (7,14):
        add(f"OFF_ATS_COVER_{t}_PLUS","PRIOR_ATS_MAGNITUDE",pats.ge(t),source_ok=has("Pregame_Prev_ATS_Margin","Prev_ATS_Margin","Prev_ATS_Cover_Margin","Context_Prev_ATS_Margin"))
        add(f"OFF_ATS_MISS_{t}_PLUS","PRIOR_ATS_MAGNITUDE",pats.le(-t),source_ok=has("Pregame_Prev_ATS_Margin","Prev_ATS_Margin","Prev_ATS_Cover_Margin","Context_Prev_ATS_Margin"))
    add("ATS_LOSS_STREAK_2","ATS_FORM",nfirst("ATS_Loss_Streak_Prior").ge(2),source_ok=has("ATS_Loss_Streak_Prior"))
    add("ATS_WIN_STREAK_2","ATS_FORM",nfirst("ATS_Win_Streak_Prior").ge(2),source_ok=has("ATS_Win_Streak_Prior"))
    add("SU_WIN_STREAK_2","SU_FORM",nfirst("Current_Win_Streak_Prior","Core_Win_Streak_Prior").ge(2),source_ok=has("Current_Win_Streak_Prior","Core_Win_Streak_Prior"))
    add("SU_LOSS_STREAK_2","SU_FORM",nfirst("Current_Loss_Streak_Prior","Core_Loss_Streak_Prior").ge(2),source_ok=has("Current_Loss_Streak_Prior","Core_Loss_Streak_Prior"))

    # V2.31 objective deep-context completion. Every field below is reconstructed
    # chronologically from completed team-side games and therefore represents
    # pregame information only. These are source-neutral Miner concepts; the
    # observed expert selections only motivated which vocabulary to expose.
    _tprev_yard=nfirst("Team_RQ_Prev_Yardage_Margin"); _oprev_yard=nfirst("Opp_RQ_Prev_Yardage_Margin")
    _tprev_ypp=nfirst("Team_RQ_Prev_YPP_Margin"); _oprev_ypp=nfirst("Opp_RQ_Prev_YPP_Margin")
    _tprev_fd=nfirst("Team_RQ_Prev_First_Down_Margin"); _oprev_fd=nfirst("Opp_RQ_Prev_First_Down_Margin")
    _tprev_to=nfirst("Team_RQ_Prev_Turnover_Margin"); _oprev_to=nfirst("Opp_RQ_Prev_Turnover_Margin")
    _tprev_nonoff=nfirst("Team_RQ_Prev_NonOffensive_TD_Margin"); _oprev_nonoff=nfirst("Opp_RQ_Prev_NonOffensive_TD_Margin")
    _tdw=nfirst("Team_RQ_Prev_Deceptive_Win_Score"); _odw=nfirst("Opp_RQ_Prev_Deceptive_Win_Score")
    _tdl=nfirst("Team_RQ_Prev_Deceptive_Loss_Score"); _odl=nfirst("Opp_RQ_Prev_Deceptive_Loss_Score")
    _twa=nfirst("Team_RQ_Prev_Win_Assistance_Score"); _owa=nfirst("Opp_RQ_Prev_Win_Assistance_Score")
    _tla=nfirst("Team_RQ_Prev_Loss_Adversity_Score"); _ola=nfirst("Opp_RQ_Prev_Loss_Adversity_Score")
    _opsu=nfirst("Opp_Prev_SU_Margin","Opp_Pregame_Prev_SU_Margin","Context_Opp_Prev_SU_Margin")
    _rq_src=_tprev_ypp.notna().any(); _orq_src=_oprev_ypp.notna().any()
    add("TEAM_OFF_WIN_NEG_YARDS","RESULT_QUALITY_REGRESSION",psu.gt(0)&_tprev_yard.lt(0),source_ok=_tprev_yard.notna().any(),min_n=20)
    add("TEAM_OFF_WIN_NEG_YPP","RESULT_QUALITY_REGRESSION",psu.gt(0)&_tprev_ypp.lt(0),source_ok=_rq_src,min_n=20)
    add("TEAM_OFF_WIN_NEG_FIRST_DOWNS","RESULT_QUALITY_REGRESSION",psu.gt(0)&_tprev_fd.lt(0),source_ok=_tprev_fd.notna().any(),min_n=20)
    add("TEAM_OFF_WIN_TURNOVER_EDGE_2PLUS","RESULT_QUALITY_REGRESSION",psu.gt(0)&_tprev_to.ge(2),source_ok=_tprev_to.notna().any(),min_n=20)
    add("TEAM_OFF_WIN_NONOFF_TD_EDGE","RESULT_QUALITY_REGRESSION",psu.gt(0)&_tprev_nonoff.ge(1),source_ok=_tprev_nonoff.notna().any(),min_n=20)
    add("TEAM_OFF_DECEPTIVE_WIN_2PLUS","RESULT_QUALITY_REGRESSION",psu.gt(0)&_tdw.ge(2),source_ok=_tdw.notna().any(),min_n=20)
    add("TEAM_OFF_RESULT_OVERPERFORMANCE","RESULT_QUALITY_REGRESSION",psu.gt(0)&(_tdw.ge(2)|_twa.ge(1)),source_ok=_tdw.notna().any() or _twa.notna().any(),min_n=20)
    add("TEAM_OFF_LOSS_POS_YARDS","RESULT_QUALITY_REGRESSION",psu.lt(0)&_tprev_yard.gt(0),source_ok=_tprev_yard.notna().any(),min_n=20)
    add("TEAM_OFF_LOSS_POS_YPP","RESULT_QUALITY_REGRESSION",psu.lt(0)&_tprev_ypp.gt(0),source_ok=_rq_src,min_n=20)
    add("TEAM_OFF_DECEPTIVE_LOSS_2PLUS","RESULT_QUALITY_REGRESSION",psu.lt(0)&_tdl.ge(2),source_ok=_tdl.notna().any(),min_n=20)
    add("TEAM_OFF_RESULT_UNDERPERFORMANCE","RESULT_QUALITY_REGRESSION",psu.lt(0)&(_tdl.ge(2)|_tla.ge(1)),source_ok=_tdl.notna().any() or _tla.notna().any(),min_n=20)
    add("OPP_OFF_WIN_NEG_YARDS","RESULT_QUALITY_REGRESSION",_opsu.gt(0)&_oprev_yard.lt(0),source_ok=_oprev_yard.notna().any(),min_n=20)
    add("OPP_OFF_WIN_NEG_YPP","RESULT_QUALITY_REGRESSION",_opsu.gt(0)&_oprev_ypp.lt(0),source_ok=_orq_src,min_n=20)
    add("OPP_OFF_WIN_TURNOVER_EDGE_2PLUS","RESULT_QUALITY_REGRESSION",_opsu.gt(0)&_oprev_to.ge(2),source_ok=_oprev_to.notna().any(),min_n=20)
    add("OPP_OFF_DECEPTIVE_WIN_2PLUS","RESULT_QUALITY_REGRESSION",_opsu.gt(0)&_odw.ge(2),source_ok=_odw.notna().any(),min_n=20)
    add("OPP_OFF_RESULT_OVERPERFORMANCE","RESULT_QUALITY_REGRESSION",_opsu.gt(0)&(_odw.ge(2)|_owa.ge(1)),source_ok=_odw.notna().any() or _owa.notna().any(),min_n=20)
    add("OPP_OFF_LOSS_POS_YPP","RESULT_QUALITY_REGRESSION",_opsu.lt(0)&_oprev_ypp.gt(0),source_ok=_orq_src,min_n=20)
    add("OPP_OFF_DECEPTIVE_LOSS_2PLUS","RESULT_QUALITY_REGRESSION",_opsu.lt(0)&_odl.ge(2),source_ok=_odl.notna().any(),min_n=20)
    add("OPP_OFF_RESULT_UNDERPERFORMANCE","RESULT_QUALITY_REGRESSION",_opsu.lt(0)&(_odl.ge(2)|_ola.ge(1)),source_ok=_odl.notna().any() or _ola.notna().any(),min_n=20)

    # Schedule/resume quality. SOS uses only the quality known about each prior
    # opponent at the time that prior game was played. Opp-adjusted Net YPP adds
    # that opponent's pregame Net YPP to the team's game Net YPP, then rolls it.
    _tgames=nfirst("Team_Deep_Game_Count_Prior","Team_Game_Number_Prior"); _ogames=nfirst("Opp_Deep_Game_Count_Prior","Opp_Game_Number_Prior")
    _twp=nfirst("Team_WinPct_Prior","Team_Deep_WinPct_Prior"); _owp=nfirst("Opp_WinPct_Prior","Opp_Deep_WinPct_Prior")
    _tsos=nfirst("Team_CTX_SOS_WinPct"); _osos=nfirst("Opp_CTX_SOS_WinPct")
    _tsos_ypp=nfirst("Team_CTX_SOS_NetYPP"); _osos_ypp=nfirst("Opp_CTX_SOS_NetYPP")
    _tadj=nfirst("Team_CTX_OppAdj_NetYPP"); _oadj=nfirst("Opp_CTX_OppAdj_NetYPP")
    _sos_src=_tsos.notna().any() and _osos.notna().any(); _adj_src=_tadj.notna().any() and _oadj.notna().any()
    add("TEAM_SOS_STRONG_550_PLUS","SCHEDULE_RESUME_QUALITY",_tgames.ge(3)&_tsos.ge(.55),source_ok=_tsos.notna().any(),min_n=20)
    add("TEAM_SOS_WEAK_450_MINUS","SCHEDULE_RESUME_QUALITY",_tgames.ge(3)&_tsos.le(.45),source_ok=_tsos.notna().any(),min_n=20)
    add("OPP_SOS_STRONG_550_PLUS","SCHEDULE_RESUME_QUALITY",_ogames.ge(3)&_osos.ge(.55),source_ok=_osos.notna().any(),min_n=20)
    add("OPP_SOS_WEAK_450_MINUS","SCHEDULE_RESUME_QUALITY",_ogames.ge(3)&_osos.le(.45),source_ok=_osos.notna().any(),min_n=20)
    add("TEAM_SOS_ADV_10P","SCHEDULE_RESUME_QUALITY",_tgames.ge(3)&_ogames.ge(3)&(_tsos-_osos).ge(.10),source_ok=_sos_src,min_n=20)
    add("OPP_SOS_ADV_10P","SCHEDULE_RESUME_QUALITY",_tgames.ge(3)&_ogames.ge(3)&(_osos-_tsos).ge(.10),source_ok=_sos_src,min_n=20)
    add("TEAM_OPPADJ_NETYPP_POS_050","SCHEDULE_RESUME_QUALITY",_tgames.ge(3)&_tadj.ge(.50),source_ok=_tadj.notna().any(),min_n=20)
    add("TEAM_OPPADJ_NETYPP_NEG_050","SCHEDULE_RESUME_QUALITY",_tgames.ge(3)&_tadj.le(-.50),source_ok=_tadj.notna().any(),min_n=20)
    add("OPP_OPPADJ_NETYPP_POS_050","SCHEDULE_RESUME_QUALITY",_ogames.ge(3)&_oadj.ge(.50),source_ok=_oadj.notna().any(),min_n=20)
    add("OPP_OPPADJ_NETYPP_NEG_050","SCHEDULE_RESUME_QUALITY",_ogames.ge(3)&_oadj.le(-.50),source_ok=_oadj.notna().any(),min_n=20)
    add("TEAM_OPPADJ_NETYPP_ADV_075","SCHEDULE_RESUME_QUALITY",_tgames.ge(3)&_ogames.ge(3)&(_tadj-_oadj).ge(.75),source_ok=_adj_src,min_n=20)
    add("OPP_OPPADJ_NETYPP_ADV_075","SCHEDULE_RESUME_QUALITY",_tgames.ge(3)&_ogames.ge(3)&(_oadj-_tadj).ge(.75),source_ok=_adj_src,min_n=20)
    add("TEAM_STRONG_RECORD_WEAK_SOS","SCHEDULE_RESUME_QUALITY",_tgames.ge(4)&_twp.ge(.75)&_tsos.le(.45),source_ok=_twp.notna().any() and _tsos.notna().any(),min_n=20)
    add("OPP_STRONG_RECORD_WEAK_SOS","SCHEDULE_RESUME_QUALITY",_ogames.ge(4)&_owp.ge(.75)&_osos.le(.45),source_ok=_owp.notna().any() and _osos.notna().any(),min_n=20)
    add("TEAM_STRONG_RECORD_WEAK_UNDERLYING","SCHEDULE_RESUME_QUALITY",_tgames.ge(4)&_twp.ge(.75)&_tadj.le(0),source_ok=_twp.notna().any() and _tadj.notna().any(),min_n=20)
    add("OPP_STRONG_RECORD_WEAK_UNDERLYING","SCHEDULE_RESUME_QUALITY",_ogames.ge(4)&_owp.ge(.75)&_oadj.le(0),source_ok=_owp.notna().any() and _oadj.notna().any(),min_n=20)

    # Recent form versus the team's own season-to-date baseline. This is not a
    # generic trend duplicate: it asks whether the last 3/5 materially diverge
    # from the prior season baseline entering the current game.
    def _recent_atoms(prefix,label,games_prior):
        sypp=nfirst(f"{prefix}_CTX_Season_Off_YPP"); rypp=nfirst(f"{prefix}_CTX_Recent3_Off_YPP")
        sdypp=nfirst(f"{prefix}_CTX_Season_Def_YPP_Allowed"); rdypp=nfirst(f"{prefix}_CTX_Recent3_Def_YPP_Allowed")
        snet=nfirst(f"{prefix}_CTX_Season_Net_YPP"); rnet=nfirst(f"{prefix}_CTX_Recent3_Net_YPP"); r5net=nfirst(f"{prefix}_CTX_Recent5_Net_YPP")
        srush=nfirst(f"{prefix}_CTX_Season_Rush_YPA"); rrush=nfirst(f"{prefix}_CTX_Recent3_Rush_YPA")
        spass=nfirst(f"{prefix}_CTX_Season_Pass_YPA"); rpass=nfirst(f"{prefix}_CTX_Recent3_Pass_YPA")
        sppg=nfirst(f"{prefix}_CTX_Season_Off_PPG"); rppg=nfirst(f"{prefix}_CTX_Recent3_Off_PPG")
        ok=games_prior.ge(3)
        add(f"{label}_RECENT3_OFF_YPP_UP_050","RECENT_VS_SEASON",ok&(rypp-sypp).ge(.50),source_ok=rypp.notna().any() and sypp.notna().any(),min_n=20)
        add(f"{label}_RECENT3_OFF_YPP_DOWN_050","RECENT_VS_SEASON",ok&(rypp-sypp).le(-.50),source_ok=rypp.notna().any() and sypp.notna().any(),min_n=20)
        add(f"{label}_RECENT3_NET_YPP_UP_075","RECENT_VS_SEASON",ok&(rnet-snet).ge(.75),source_ok=rnet.notna().any() and snet.notna().any(),min_n=20)
        add(f"{label}_RECENT3_NET_YPP_DOWN_075","RECENT_VS_SEASON",ok&(rnet-snet).le(-.75),source_ok=rnet.notna().any() and snet.notna().any(),min_n=20)
        add(f"{label}_RECENT3_DEF_YPP_IMPROVED_050","RECENT_VS_SEASON",ok&(rdypp-sdypp).le(-.50),source_ok=rdypp.notna().any() and sdypp.notna().any(),min_n=20)
        add(f"{label}_RECENT3_DEF_YPP_WORSE_050","RECENT_VS_SEASON",ok&(rdypp-sdypp).ge(.50),source_ok=rdypp.notna().any() and sdypp.notna().any(),min_n=20)
        add(f"{label}_RECENT3_RUSH_YPA_UP_075","RECENT_VS_SEASON",ok&(rrush-srush).ge(.75),source_ok=rrush.notna().any() and srush.notna().any(),min_n=20)
        add(f"{label}_RECENT3_PASS_YPA_UP_100","RECENT_VS_SEASON",ok&(rpass-spass).ge(1.0),source_ok=rpass.notna().any() and spass.notna().any(),min_n=20)
        add(f"{label}_RECENT3_PPG_UP_7","RECENT_VS_SEASON",ok&(rppg-sppg).ge(7.0),source_ok=rppg.notna().any() and sppg.notna().any(),min_n=20)
        add(f"{label}_RECENT3_AND5_NET_YPP_UP","RECENT_VS_SEASON",games_prior.ge(5)&(rnet-snet).ge(.50)&(r5net-snet).ge(.50),source_ok=r5net.notna().any() and rnet.notna().any() and snet.notna().any(),min_n=20)
        add(f"{label}_RECENT3_AND5_NET_YPP_DOWN","RECENT_VS_SEASON",games_prior.ge(5)&(rnet-snet).le(-.50)&(r5net-snet).le(-.50),source_ok=r5net.notna().any() and rnet.notna().any() and snet.notna().any(),min_n=20)
    _recent_atoms("Team","TEAM",_tgames); _recent_atoms("Opp","OPP",_ogames)

    # Offense-vs-defense matchup differentials. The values are season-to-date
    # priors, so no current-game box-score information can enter the condition.
    _trush=nfirst("Team_CTX_Season_Rush_YPA"); _orush=nfirst("Opp_CTX_Season_Rush_YPA")
    _tdefrush=nfirst("Team_CTX_Season_Def_Rush_YPA_Allowed"); _odefrush=nfirst("Opp_CTX_Season_Def_Rush_YPA_Allowed")
    _tpass=nfirst("Team_CTX_Season_Pass_YPA"); _opass=nfirst("Opp_CTX_Season_Pass_YPA")
    _tdefpass=nfirst("Team_CTX_Season_Def_Pass_YPA_Allowed"); _odefpass=nfirst("Opp_CTX_Season_Def_Pass_YPA_Allowed")
    _typp=nfirst("Team_CTX_Season_Off_YPP"); _oypp=nfirst("Opp_CTX_Season_Off_YPP")
    _tdefypp=nfirst("Team_CTX_Season_Def_YPP_Allowed"); _odefypp=nfirst("Opp_CTX_Season_Def_YPP_Allowed")
    _tppg=nfirst("Team_CTX_Season_Off_PPG"); _oppg=nfirst("Opp_CTX_Season_Off_PPG")
    _tdefppg=nfirst("Team_CTX_Season_Def_PPG"); _odefppg=nfirst("Opp_CTX_Season_Def_PPG")
    _trush_edge=_trush-_odefrush; _orush_edge=_orush-_tdefrush
    _tpass_edge=_tpass-_odefpass; _opass_edge=_opass-_tdefpass
    _typp_edge=_typp-_odefypp; _oypp_edge=_oypp-_tdefypp
    _tscore_edge=_tppg-_odefppg; _oscore_edge=_oppg-_tdefppg
    _tcomp=pd.concat([_trush_edge.ge(.75),_tpass_edge.ge(1.0),_typp_edge.ge(.50),_tscore_edge.ge(5.0)],axis=1).sum(axis=1)
    _ocomp=pd.concat([_orush_edge.ge(.75),_opass_edge.ge(1.0),_oypp_edge.ge(.50),_oscore_edge.ge(5.0)],axis=1).sum(axis=1)
    _match_src=_typp.notna().any() and _odefypp.notna().any(); _omatch_src=_oypp.notna().any() and _tdefypp.notna().any()
    add("TEAM_RUSH_MATCHUP_EDGE_075PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_trush_edge.ge(.75),source_ok=_trush.notna().any() and _odefrush.notna().any(),min_n=20)
    add("TEAM_PASS_MATCHUP_EDGE_100PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_tpass_edge.ge(1.0),source_ok=_tpass.notna().any() and _odefpass.notna().any(),min_n=20)
    add("TEAM_YPP_MATCHUP_EDGE_050PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_typp_edge.ge(.50),source_ok=_match_src,min_n=20)
    add("TEAM_SCORING_MATCHUP_EDGE_5PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_tscore_edge.ge(5.0),source_ok=_tppg.notna().any() and _odefppg.notna().any(),min_n=20)
    add("TEAM_MULTI_MATCHUP_EDGE_2PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_tcomp.ge(2),source_ok=_match_src,min_n=20)
    add("TEAM_MULTI_MATCHUP_EDGE_3PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_tcomp.ge(3),source_ok=_match_src,min_n=20)
    add("OPP_RUSH_MATCHUP_EDGE_075PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_orush_edge.ge(.75),source_ok=_orush.notna().any() and _tdefrush.notna().any(),min_n=20)
    add("OPP_PASS_MATCHUP_EDGE_100PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_opass_edge.ge(1.0),source_ok=_opass.notna().any() and _tdefpass.notna().any(),min_n=20)
    add("OPP_YPP_MATCHUP_EDGE_050PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_oypp_edge.ge(.50),source_ok=_omatch_src,min_n=20)
    add("OPP_SCORING_MATCHUP_EDGE_5PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_oscore_edge.ge(5.0),source_ok=_oppg.notna().any() and _tdefppg.notna().any(),min_n=20)
    add("OPP_MULTI_MATCHUP_EDGE_2PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_ocomp.ge(2),source_ok=_omatch_src,min_n=20)
    add("OPP_MULTI_MATCHUP_EDGE_3PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_ocomp.ge(3),source_ok=_omatch_src,min_n=20)
    add("TEAM_MATCHUP_EDGE_OVER_OPP_2PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&(_tcomp-_ocomp).ge(2),source_ok=_match_src and _omatch_src,min_n=20)
    add("OPP_MATCHUP_EDGE_OVER_TEAM_2PLUS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&(_ocomp-_tcomp).ge(2),source_ok=_match_src and _omatch_src,min_n=20)
    _tsack_allow=nfirst("Team_CTX_Season_Sack_Allowed_Rate"); _osack_allow=nfirst("Opp_CTX_Season_Sack_Allowed_Rate")
    _tdef_sack=nfirst("Team_CTX_Season_Def_Sack_Rate"); _odef_sack=nfirst("Opp_CTX_Season_Def_Sack_Rate")
    add("TEAM_PASS_PROTECTION_STRESS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_tsack_allow.ge(.07)&_odef_sack.ge(.07),source_ok=_tsack_allow.notna().any() and _odef_sack.notna().any(),min_n=20)
    add("OPP_PASS_PROTECTION_STRESS","MATCHUP_DIFFERENTIAL",_tgames.ge(3)&_ogames.ge(3)&_osack_allow.ge(.07)&_tdef_sack.ge(.07),source_ok=_osack_allow.notna().any() and _tdef_sack.notna().any(),min_n=20)

    # V2.26 generic expert-observation vocabulary. These features were added
    # because observed expert cards exposed missing concepts, but the rules are
    # completely source-neutral and must independently validate historically.
    _prev_score_src=bool(prev_pf.notna().any() and prev_pa.notna().any())
    add("OFF_SHUTOUT_LOSS","OBS_PRIOR_SCORING_EXTREME",prev_pf.eq(0)&prev_pa.gt(0),desc="off a straight-up loss while scoring zero",source_ok=_prev_score_src,min_n=15)
    add("OFF_SCORED_7_OR_LESS","OBS_PRIOR_POINTS_FOR",prev_pf.le(7)&prev_pf.notna(),desc="scored <=7 in previous game",source_ok=prev_pf.notna().any(),min_n=20)
    add("OFF_SCORED_35_PLUS","OBS_PRIOR_POINTS_FOR",prev_pf.ge(35),desc="scored >=35 in previous game",source_ok=prev_pf.notna().any(),min_n=20)
    add("OFF_ALLOWED_35_PLUS","OBS_PRIOR_POINTS_ALLOWED",prev_pa.ge(35),desc="allowed >=35 in previous game",source_ok=prev_pa.notna().any(),min_n=20)
    add("OFF_ALLOWED_42_PLUS","OBS_PRIOR_POINTS_ALLOWED",prev_pa.ge(42),desc="allowed >=42 in previous game",source_ok=prev_pa.notna().any(),min_n=20)
    if market=="spreads":
        add("TEAM_ATS_WINPCT_LE_250","OBS_ATS_SEASON_STATE",team_ats_pct_prior.le(.25)&team_ats_pct_prior.notna(),desc="prior-season-to-date ATS cover rate <=25%",source_ok=team_ats_pct_prior.notna().any(),min_n=20)
        add("TEAM_ATS_WINPCT_LE_400","OBS_ATS_SEASON_STATE",team_ats_pct_prior.le(.40)&team_ats_pct_prior.notna(),desc="prior-season-to-date ATS cover rate <=40%",source_ok=team_ats_pct_prior.notna().any(),min_n=20)

    # Horizon-symmetric SU / ATS sequences (oldest -> newest in the atom name).
    def sign_series(kind,lag):
        if kind=="SU":
            if lag==1: return nfirst("Prev_SU_Margin","Pregame_Prev_SU_Margin","Context_Prev_SU_Margin")
            return nfirst(f"Prev{lag}_SU_Margin")
        if lag==1: return nfirst("Prev_ATS_Margin","Prev_ATS_Cover_Margin","Pregame_Prev_ATS_Margin","Context_Prev_ATS_Margin")
        return nfirst(f"Prev{lag}_ATS_Margin",f"Prev{lag}_ATS_Cover_Margin")
    for kind,fam in (("SU","SU_SEQUENCE"),("ATS","ATS_SEQUENCE")):
        v1=sign_series(kind,1); ok1=v1.notna()
        add(f"{kind}_SEQ1_W",fam,ok1&v1.gt(0),source_ok=bool(ok1.any()) or (for_live and any(c in g.columns for c in (["Prev_SU_Margin","Pregame_Prev_SU_Margin"] if kind=="SU" else ["Prev_ATS_Margin","Prev_ATS_Cover_Margin","Pregame_Prev_ATS_Margin"]))))
        add(f"{kind}_SEQ1_L",fam,ok1&v1.lt(0),source_ok=bool(ok1.any()) or (for_live and any(c in g.columns for c in (["Prev_SU_Margin","Pregame_Prev_SU_Margin"] if kind=="SU" else ["Prev_ATS_Margin","Prev_ATS_Cover_Margin","Pregame_Prev_ATS_Margin"]))))
        v2=sign_series(kind,2); ok2=ok1&v2.notna()
        for a in "WL":
            for b in "WL":
                m=ok2 & (v2.gt(0) if a=="W" else v2.lt(0)) & (v1.gt(0) if b=="W" else v1.lt(0))
                add(f"{kind}_SEQ2_{a}{b}",fam,m,source_ok=bool(v2.notna().any()))
        v3=sign_series(kind,3); ok3=ok2&v3.notna()
        for a in "WL":
            for b in "WL":
                for c in "WL":
                    m=ok3 & (v3.gt(0) if a=="W" else v3.lt(0)) & (v2.gt(0) if b=="W" else v2.lt(0)) & (v1.gt(0) if c=="W" else v1.lt(0))
                    add(f"{kind}_SEQ3_{a}{b}{c}",fam,m,source_ok=bool(v3.notna().any()))

    # Trend direction from the same exact 1/2/3-game margins.
    p1=nfirst("Prev_ATS_Margin","Prev_ATS_Cover_Margin","Pregame_Prev_ATS_Margin"); p2=nfirst("Prev2_ATS_Margin","Prev2_ATS_Cover_Margin"); p3=nfirst("Prev3_ATS_Margin","Prev3_ATS_Cover_Margin")
    add("ATS_MARGIN_IMPROVING_2","ATS_TREND",p1.gt(p2)&p1.notna()&p2.notna(),source_ok=has("Prev2_ATS_Margin","Prev2_ATS_Cover_Margin"))
    add("ATS_MARGIN_WORSENING_2","ATS_TREND",p1.lt(p2)&p1.notna()&p2.notna(),source_ok=has("Prev2_ATS_Margin","Prev2_ATS_Cover_Margin"))
    add("ATS_MARGIN_IMPROVING_3","ATS_TREND",p1.gt(p2)&p2.gt(p3)&p3.notna(),source_ok=has("Prev3_ATS_Margin","Prev3_ATS_Cover_Margin"))
    add("ATS_MARGIN_WORSENING_3","ATS_TREND",p1.lt(p2)&p2.lt(p3)&p3.notna(),source_ok=has("Prev3_ATS_Margin","Prev3_ATS_Cover_Margin"))

    # Opponent prior state and quality.
    opsu=nfirst("Context_Opp_Prev_SU_Margin","Opp_Pregame_Prev_SU_Margin","Opp_Prev_SU_Margin")
    if not opsu.notna().any() and opp_prev_pf.notna().any() and opp_prev_pa.notna().any(): opsu=opp_prev_pf-opp_prev_pa
    opats=nfirst("Context_Opp_Prev_ATS_Margin","Opp_Pregame_Prev_ATS_Margin")
    add("OPP_OFF_SU_WIN","OPP_PRIOR_SU1",opsu.gt(0),source_ok=has("Context_Opp_Prev_SU_Margin","Opp_Pregame_Prev_SU_Margin","Opp_Prev_SU_Margin"))
    add("OPP_OFF_SU_LOSS","OPP_PRIOR_SU1",opsu.lt(0),source_ok=has("Context_Opp_Prev_SU_Margin","Opp_Pregame_Prev_SU_Margin","Opp_Prev_SU_Margin"))
    add("OPP_OFF_ATS_WIN","OPP_PRIOR_ATS1",opats.gt(0),source_ok=has("Context_Opp_Prev_ATS_Margin","Opp_Pregame_Prev_ATS_Margin"))
    add("OPP_OFF_ATS_LOSS","OPP_PRIOR_ATS1",opats.lt(0),source_ok=has("Context_Opp_Prev_ATS_Margin","Opp_Pregame_Prev_ATS_Margin"))
    _opp_prev_dog=nfirst("Opp_Prev_Is_ML_Dog")
    _opp_prev_fav=nfirst("Opp_Prev_Is_ML_Favorite")
    if not _opp_prev_dog.notna().any() and _opp_prev_fav.notna().any(): _opp_prev_dog=1-_opp_prev_fav
    if not _opp_prev_dog.notna().any() and derived_opp_prev_dog.notna().any(): _opp_prev_dog=derived_opp_prev_dog
    add("OPP_OFF_UPSET_WIN","OBS_PRIOR_UPSET_STATE",opsu.gt(0)&_opp_prev_dog.eq(1),desc="opponent won previous game straight up as an underdog",source_ok=bool(opsu.notna().any() and _opp_prev_dog.notna().any()),min_n=20)
    if market=="totals":
        add("BOTH_OFF_SU_LOSS","OBS_DUAL_PRIOR_RESULT",psu.lt(0)&opsu.lt(0),desc="both teams enter off straight-up losses",source_ok=bool(psu.notna().any() and opsu.notna().any()),min_n=20)
        add("EITHER_OFF_SCORED_35_PLUS","OBS_PRIOR_POINTS_FOR",prev_pf.ge(35)|opp_prev_pf.ge(35),desc="either team scored >=35 in its prior game",source_ok=bool(prev_pf.notna().any() or opp_prev_pf.notna().any()),min_n=20)
        add("EITHER_OFF_ALLOWED_35_PLUS","OBS_PRIOR_POINTS_ALLOWED",prev_pa.ge(35)|opp_prev_pa.ge(35),desc="either team allowed >=35 in its prior game",source_ok=bool(prev_pa.notna().any() or opp_prev_pa.notna().any()),min_n=20)
        add("BOTH_OFF_ALLOWED_28_PLUS","OBS_PRIOR_POINTS_ALLOWED",prev_pa.ge(28)&opp_prev_pa.ge(28),desc="both teams allowed >=28 in prior game",source_ok=bool(prev_pa.notna().any() and opp_prev_pa.notna().any()),min_n=20)
    twp=nfirst("Team_WinPct_Prior","Core_Team_WinPct_Prior","WinPct_Prior_System","HC_WinPct_Prior")
    owp=nfirst("Opp_WinPct_Prior","Core_Opp_WinPct_Prior","Opp_WinPct_Prior_System","HC_Opp_WinPct_Prior")
    for nm,ser,fam,ok in (("TEAM",twp,"TEAM_STATE",has("Team_WinPct_Prior","Core_Team_WinPct_Prior","WinPct_Prior_System","HC_WinPct_Prior")),("OPP",owp,"OPP_STATE",has("Opp_WinPct_Prior","Core_Opp_WinPct_Prior","Opp_WinPct_Prior_System","HC_Opp_WinPct_Prior"))):
        add(f"{nm}_WINPCT_LE_400",fam,ser.le(.400),source_ok=ok)
        add(f"{nm}_WINPCT_LE_500",fam,ser.le(.500),source_ok=ok)
        if nm=="OPP": add("OPP_WINPCT_GE_500",fam,ser.ge(.500),source_ok=ok,min_n=20)
        add(f"{nm}_WINPCT_GE_600",fam,ser.ge(.600),source_ok=ok)

    # V2.23 SEASON_RECORD_STATE. Every condition is pregame. `game_no` is
    # completed games prior, `twp` is prior win percentage, and ATS loss streak
    # is explicitly prior. ATS_COVERLESS means every prior graded ATS result was
    # a loss; a prior push intentionally fails this exact 0-X ATS definition.
    if market=="spreads":
        _prior_games=game_no
        _record_src=has("Team_Game_Number_Prior","Game_Number_Prior","Context_Team_Games_Prior")
        _su_src=_record_src and has("Team_WinPct_Prior","Core_Team_WinPct_Prior","WinPct_Prior_System","HC_WinPct_Prior")
        _ats_loss=nfirst("ATS_Loss_Streak_Prior")
        _ats_src=_record_src and has("ATS_Loss_Streak_Prior")
        _su_winless=_prior_games.ge(1)&twp.eq(0)
        _ats_coverless=_prior_games.ge(1)&_ats_loss.ge(_prior_games)
        add("SU_WINLESS_PRIOR","SEASON_RECORD_STATE",_su_winless,desc="home-oriented team has zero SU wins entering game",source_ok=_su_src,min_n=20)
        add("ATS_COVERLESS_PRIOR","SEASON_RECORD_STATE",_ats_coverless,desc="home-oriented team has zero ATS covers; every prior graded ATS game was a loss",source_ok=_ats_src,min_n=20)
        add("SU_AND_ATS_WINLESS_PRIOR","SEASON_RECORD_STATE",_su_winless&_ats_coverless,desc="home-oriented team has zero SU wins and zero ATS covers entering game",source_ok=_su_src and _ats_src,min_n=20)
        for _n in (2,3,4):
            add(f"SU_WINLESS_AFTER_{_n}_PLUS","SEASON_RECORD_STATE",_su_winless&_prior_games.ge(_n),desc=f"home-oriented team SU winless after {_n}+ completed games",source_ok=_su_src,min_n=20)
            add(f"ATS_COVERLESS_AFTER_{_n}_PLUS","SEASON_RECORD_STATE",_ats_coverless&_prior_games.ge(_n),desc=f"home-oriented team ATS coverless after {_n}+ completed games",source_ok=_ats_src,min_n=20)
            add(f"SU_AND_ATS_WINLESS_AFTER_{_n}_PLUS","SEASON_RECORD_STATE",_su_winless&_ats_coverless&_prior_games.ge(_n),desc=f"home-oriented team SU and ATS winless after {_n}+ completed games",source_ok=_su_src and _ats_src,min_n=20)

        # The Miner frame is HOME-oriented, so every state concept also needs an
        # opponent mirror to represent a ROAD-side team (for example a winless
        # road underdog). These are the same SEASON_RECORD_STATE family and can
        # never stack as independent votes.
        _opp_prior_games=nfirst("Opp_Game_Number_Prior","Context_Opp_Games_Prior")
        _opp_record_src=has("Opp_Game_Number_Prior","Context_Opp_Games_Prior")
        _opp_su_src=_opp_record_src and has("Opp_WinPct_Prior","Core_Opp_WinPct_Prior","Opp_WinPct_Prior_System","HC_Opp_WinPct_Prior")
        _opp_ats_loss=nfirst("Opp_ATS_Loss_Streak_Prior")
        _opp_ats_src=_opp_record_src and has("Opp_ATS_Loss_Streak_Prior")
        _opp_su_winless=_opp_prior_games.ge(1)&owp.eq(0)
        _opp_ats_coverless=_opp_prior_games.ge(1)&_opp_ats_loss.ge(_opp_prior_games)
        add("OPP_SU_WINLESS_PRIOR","SEASON_RECORD_STATE",_opp_su_winless,desc="road-side opponent has zero SU wins entering game",source_ok=_opp_su_src,min_n=20)
        add("OPP_ATS_COVERLESS_PRIOR","SEASON_RECORD_STATE",_opp_ats_coverless,desc="road-side opponent has zero ATS covers entering game",source_ok=_opp_ats_src,min_n=20)
        add("OPP_SU_AND_ATS_WINLESS_PRIOR","SEASON_RECORD_STATE",_opp_su_winless&_opp_ats_coverless,desc="road-side opponent has zero SU wins and zero ATS covers entering game",source_ok=_opp_su_src and _opp_ats_src,min_n=20)
        for _n in (2,3,4):
            add(f"OPP_SU_WINLESS_AFTER_{_n}_PLUS","SEASON_RECORD_STATE",_opp_su_winless&_opp_prior_games.ge(_n),desc=f"road-side opponent SU winless after {_n}+ completed games",source_ok=_opp_su_src,min_n=20)
            add(f"OPP_ATS_COVERLESS_AFTER_{_n}_PLUS","SEASON_RECORD_STATE",_opp_ats_coverless&_opp_prior_games.ge(_n),desc=f"road-side opponent ATS coverless after {_n}+ completed games",source_ok=_opp_ats_src,min_n=20)
            add(f"OPP_SU_AND_ATS_WINLESS_AFTER_{_n}_PLUS","SEASON_RECORD_STATE",_opp_su_winless&_opp_ats_coverless&_opp_prior_games.ge(_n),desc=f"road-side opponent SU and ATS winless after {_n}+ completed games",source_ok=_opp_su_src and _opp_ats_src,min_n=20)
        add("OPP_HAS_SU_WIN","OPP_STATE",owp.gt(0),desc="opponent has at least one prior SU win",source_ok=has("Opp_WinPct_Prior","Core_Opp_WinPct_Prior","Opp_WinPct_Prior_System","HC_Opp_WinPct_Prior"),min_n=20)

    # Role-change / team-price memory.
    prev_dog=nfirst("Prev_Is_ML_Dog")
    prev_fav=nfirst("Prev_Is_ML_Favorite")
    if not prev_fav.notna().any() and prev_dog.notna().any(): prev_fav=1-prev_dog
    add("PRIOR_DOG","ROLE_CHANGE",prev_dog.eq(1),source_ok=has("Prev_Is_ML_Dog"))
    add("PRIOR_FAVORITE","ROLE_CHANGE",prev_fav.eq(1),source_ok=has("Prev_Is_ML_Favorite","Prev_Is_ML_Dog"))
    add("ROLE_FLIP_FAVORITE_TO_DOG","ROLE_CHANGE",prev_fav.eq(1)&sp.gt(0),source_ok=has("Prev_Is_ML_Favorite","Prev_Is_ML_Dog") and has("Consensus_Open_Spread","Opening_Spread"))
    add("ROLE_FLIP_DOG_TO_FAVORITE","ROLE_CHANGE",prev_dog.eq(1)&sp.lt(0),source_ok=has("Prev_Is_ML_Dog") and has("Consensus_Open_Spread","Opening_Spread"))


    # V2.24 ROLE_TRANSITION_BOUNCEBACK. This is the generic pregame state prompted
    # by the observed high-rated bounceback dog example, not a hard-coded team/pick.
    # Core idea: a team that was HOME FAVORITE, LOST SU, and is now a ROAD DOG.
    # Winning-team / both-winning / +5..+7.5 variants are alternatives inside the
    # SAME family so they can never stack as independent votes.
    _prev_home=nfirst("Prev_Is_Home","Pregame_Prev_Is_Home","Context_Prev_Is_Home")
    if (not _prev_home.notna().any()) and (not for_live) and has("Is_Home") and any(c in g.columns for c in ("Team_Norm","Team")):
        try:
            _team=tfirst("Team_Norm","Team")
            _season=nfirst("Season")
            _date=pd.to_datetime(g.get("Game_Date",g.get("Game_Start",pd.Series(pd.NaT,index=g.index))),errors="coerce",utc=True)
            _tmp=pd.DataFrame({"idx":np.arange(len(g)),"team":_team,"season":_season,"date":_date,"home":pd.to_numeric(g.get("Is_Home"),errors="coerce")})
            _tmp=_tmp.sort_values(["team","season","date","idx"],kind="stable")
            _tmp["prev_home"]=_tmp.groupby(["team","season"],sort=False,dropna=False)["home"].shift(1)
            _ph=np.full(len(g),np.nan,float); _ph[_tmp["idx"].to_numpy(int)]=pd.to_numeric(_tmp["prev_home"],errors="coerce").to_numpy(float)
            _prev_home=pd.Series(_ph,index=g.index,dtype=float)
        except Exception:
            pass
    _opp_prev_home=nfirst("Opp_Prev_Is_Home","Opp_Pregame_Prev_Is_Home","Context_Opp_Prev_Is_Home")
    if (not _opp_prev_home.notna().any()) and (not for_live) and _prev_home.notna().any() and any(c in g.columns for c in ("Opponent_Norm","Opponent")):
        try:
            _team=tfirst("Team_Norm","Team"); _opp=tfirst("Opponent_Norm","Opponent"); _season=nfirst("Season")
            _date=pd.to_datetime(g.get("Game_Date",g.get("Game_Start",pd.Series(pd.NaT,index=g.index))),errors="coerce",utc=True)
            _lookup={}
            for _i in range(len(g)):
                _k=(str(_season.iloc[_i]),str(_date.iloc[_i]),str(_team.iloc[_i]))
                _v=_prev_home.iloc[_i]
                if pd.notna(_v): _lookup[_k]=float(_v)
            _ov=[]
            for _i in range(len(g)):
                _ov.append(_lookup.get((str(_season.iloc[_i]),str(_date.iloc[_i]),str(_opp.iloc[_i])),np.nan))
            _opp_prev_home=pd.Series(_ov,index=g.index,dtype=float)
        except Exception:
            pass
    _opp_prev_fav=nfirst("Opp_Prev_Is_ML_Favorite")
    _opp_prev_dog=nfirst("Opp_Prev_Is_ML_Dog")
    if not _opp_prev_fav.notna().any() and _opp_prev_dog.notna().any(): _opp_prev_fav=1-_opp_prev_dog
    _team_bounce=(is_home.eq(0)&sp.gt(0)&_prev_home.eq(1)&prev_fav.eq(1)&psu.lt(0))
    # On a home-oriented row, the opponent is the road dog iff the home spread is negative.
    _opp_bounce=(is_home.eq(1)&sp.lt(0)&_opp_prev_home.eq(1)&_opp_prev_fav.eq(1)&opsu.lt(0))
    _bh_src=bool((_prev_home.notna().any() or for_live) and has("Prev_Is_ML_Favorite","Prev_Is_ML_Dog") and has("Pregame_Prev_SU_Margin","Prev_SU_Margin","Context_Prev_SU_Margin"))
    _obh_src=bool((_opp_prev_home.notna().any() or for_live) and has("Opp_Prev_Is_ML_Favorite","Opp_Prev_Is_ML_Dog") and has("Context_Opp_Prev_SU_Margin","Opp_Pregame_Prev_SU_Margin","Opp_Prev_SU_Margin"))
    add("BOUNCEBACK_HOME_FAV_UPSET_TO_ROAD_DOG","ROLE_TRANSITION_BOUNCEBACK",_team_bounce,desc="road dog after SU upset loss as home favorite",source_ok=_bh_src,min_n=20)
    add("BOUNCEBACK_WINNING_TEAM","ROLE_TRANSITION_BOUNCEBACK",_team_bounce&twp.ge(.500),desc="winning team road dog after SU upset loss as home favorite",source_ok=_bh_src,min_n=20)
    add("BOUNCEBACK_BOTH_WINNING_TEAMS","ROLE_TRANSITION_BOUNCEBACK",_team_bounce&twp.ge(.500)&owp.ge(.500),desc="winning road dog vs winning opponent after home-favorite upset loss",source_ok=_bh_src,min_n=20)
    add("BOUNCEBACK_WINNING_TEAM_DOG_5_TO_7P5","ROLE_TRANSITION_BOUNCEBACK",_team_bounce&twp.ge(.500)&sp.between(5.0,7.5),desc="winning bounceback road dog from +5 through +7.5",source_ok=_bh_src,min_n=15)
    add("OPP_BOUNCEBACK_HOME_FAV_UPSET_TO_ROAD_DOG","ROLE_TRANSITION_BOUNCEBACK",_opp_bounce,desc="opponent is road dog after SU upset loss as home favorite",source_ok=_obh_src,min_n=20)
    add("OPP_BOUNCEBACK_WINNING_TEAM","ROLE_TRANSITION_BOUNCEBACK",_opp_bounce&owp.ge(.500),desc="opponent is winning road dog after home-favorite upset loss",source_ok=_obh_src,min_n=20)
    add("OPP_BOUNCEBACK_BOTH_WINNING_TEAMS","ROLE_TRANSITION_BOUNCEBACK",_opp_bounce&owp.ge(.500)&twp.ge(.500),desc="opponent winning road dog vs winning team after home-favorite upset loss",source_ok=_obh_src,min_n=20)
    add("OPP_BOUNCEBACK_WINNING_TEAM_DOG_5_TO_7P5","ROLE_TRANSITION_BOUNCEBACK",_opp_bounce&owp.ge(.500)&(-sp).between(5.0,7.5),desc="opponent winning bounceback road dog from +5 through +7.5",source_ok=_obh_src,min_n=15)
    for c,name in (("Pathi_FB_Usually_Dog_Now_Favorite","ROLE_FLIP_DOG_TO_FAV_HISTORY"),("Pathi_FB_Usually_Favorite_Now_Dog","ROLE_FLIP_FAV_TO_DOG_HISTORY")):
        add(name,"ROLE_HISTORY",nfirst(c).eq(1),source_ok=has(c))
    role_shift=nfirst("Role_Price_Shift")
    add("ROLE_PRICE_SHIFT_POS","ROLE_HISTORY",role_shift.ge(2),source_ok=has("Role_Price_Shift"))
    add("ROLE_PRICE_SHIFT_NEG","ROLE_HISTORY",role_shift.le(-2),source_ok=has("Role_Price_Shift"))
    price_z=nfirst("ML_Price_ZScore_vs_TeamHistory")
    add("ML_PRICE_Z_1P5_HIGH","ROLE_HISTORY",price_z.ge(1.5),source_ok=has("ML_Price_ZScore_vs_TeamHistory"))
    add("ML_PRICE_Z_1P5_LOW","ROLE_HISTORY",price_z.le(-1.5),source_ok=has("ML_Price_ZScore_vs_TeamHistory"))

    # Market path / sharp-soft / key crossing.
    for mins in (30,60,120):
        c=f"Line_Move_{mins}m"; mv=nfirst(c)
        add(f"LINE_MOVE_{mins}M_POS","MARKET_PATH",mv.ge(.5),source_ok=has(c))
        add(f"LINE_MOVE_{mins}M_NEG","MARKET_PATH",mv.le(-.5),source_ok=has(c))
    mfo=nfirst("Line_Move_From_Open")
    add("LINE_FROM_OPEN_2_PLUS","MARKET_PATH",mfo.ge(2),source_ok=has("Line_Move_From_Open"))
    add("LINE_FROM_OPEN_2_MINUS","MARKET_PATH",mfo.le(-2),source_ok=has("Line_Move_From_Open"))
    revs=nfirst("Direction_Changes_Count")
    add("MARKET_REVERSAL_2_PLUS","MARKET_PATH",revs.ge(2),source_ok=has("Direction_Changes_Count"))
    sm=nfirst("Sharp_Book_Move_60m")
    add("SHARP_MOVE_60_POS","SHARP_SOFT",sm.ge(.5),source_ok=has("Sharp_Book_Move_60m"))
    add("SHARP_MOVE_60_NEG","SHARP_SOFT",sm.le(-.5),source_ok=has("Sharp_Book_Move_60m"))
    div=nfirst("Sharp_Soft_Divergence")
    add("SHARP_SOFT_DIV_POS","SHARP_SOFT",div.ge(.5),source_ok=has("Sharp_Soft_Divergence"))
    add("SHARP_SOFT_DIV_NEG","SHARP_SOFT",div.le(-.5),source_ok=has("Sharp_Soft_Divergence"))
    cons=nfirst("Sharp_Consensus_Direction")
    add("SHARP_CONSENSUS_POS","SHARP_SOFT",cons.ge(.5),source_ok=has("Sharp_Consensus_Direction"))
    add("SHARP_CONSENSUS_NEG","SHARP_SOFT",cons.le(-.5),source_ok=has("Sharp_Consensus_Direction"))
    for key in (3,7,10,14):
        c=f"Crossed_Key_{key}_Last60m"; add(f"KEY_{key}_CROSSED_60M","KEY_NUMBER",nfirst(c).eq(1),source_ok=has(c),min_n=20)
    for c,name in (("Key_Cross_Confirmed_By_Sharp_Books","KEY_CROSS_SHARP_CONFIRMED"),("Key_Cross_Reversed","KEY_CROSS_REVERSED"),("Key_Cross_Persistence","KEY_CROSS_PERSISTENT")):
        add(name,"KEY_NUMBER",nfirst(c).ge(1),source_ok=has(c),min_n=20)

    # V2.5 Expert/Model Atom Bridge. These are hypotheses for the existing Miner,
    # not a new model. Pathi/Big Al flags are deterministic pregame rules. CORE
    # and specialist states are season-forward OOF fields published by the protected
    # CORE challenger and are research-only until a like-for-like live bridge exists.
    _mkt=str(market or "").lower().strip()
    if _mkt in {"", "spreads"}:
        pathi_cols=[
            "Pathi_FB_Dog_Hook_Above_3","Pathi_FB_Dog_Hook_Above_7","Pathi_FB_Dog_Hook_Above_10",
            "Pathi_FB_Dog_10_Plus","Pathi_FB_Dog_0_to_3","Pathi_FB_Dog_3_to_3_5","Pathi_FB_Dog_3_5_to_6_5",
            "Pathi_FB_Dog_On_7","Pathi_FB_Dog_Above_7","Pathi_FB_Dog_Below_Key_3","Pathi_FB_Dog_Below_Key_7",
            "Pathi_FB_Favorite_Below_Key_3","Pathi_FB_Favorite_Below_Key_7","Pathi_FB_Favorite_Below_Key_10",
            "Pathi_FB_Favorite_Laying_Hook_3","Pathi_FB_Favorite_Laying_Hook_7",
            "Pathi_FB_Dog_TotalSpread_Gap_LE10","Pathi_FB_Dog_Moved_Below_Key_3","Pathi_FB_Dog_Moved_Above_Key_3",
            "Pathi_FB_Dog_Moved_Below_Key_7","Pathi_FB_Dog_Moved_Above_Key_7","Pathi_FB_Dog_Moved_Below_Key_10",
            "Pathi_FB_Dog_Moved_Above_Key_10","Pathi_FB_Crossed_Key_Toward_Team","Pathi_FB_Crossed_Key_Away_From_Team",
            "Pathi_FB_Usually_Dog_Now_Favorite","Pathi_FB_Usually_Favorite_Now_Dog",
        ]
        # Feed base Big Al hypotheses, not hand-tightened children. The Miner can
        # add ROAD/conference/spread/timing atoms itself and lineage can then judge
        # whether the child genuinely improves the published parent.
        bigal_cols=[
            "BigAl_CF1_Week2Home42Win","BigAl_CF2_LateSeasonRevengeDog",
            "BigAl_CF3_Fade19PlusFavoriteUpsetLoss",
        ]
        p_masks=[]; b_masks=[]
        _pathi_candidates=list(pathi_cols)+[c+"__ROAD_SIDE" for c in pathi_cols if has(c+"__ROAD_SIDE")]
        _bigal_candidates=list(bigal_cols)+[c+"__ROAD_SIDE" for c in bigal_cols if has(c+"__ROAD_SIDE")]
        for c in _pathi_candidates:
            if has(c):
                mm=pd.to_numeric(nfirst(c),errors="coerce").fillna(0).eq(1).astype(bool); p_masks.append(mm)
                add("EXPERT_"+re.sub(r"[^A-Z0-9]+","_",c.upper())[:58],"EXPERT_PATHI",mm,desc=f"Pathi atom: {c}",min_n=20,source_ok=True)
        for c in _bigal_candidates:
            if has(c):
                mm=pd.to_numeric(nfirst(c),errors="coerce").fillna(0).eq(1).astype(bool); b_masks.append(mm)
                add("EXPERT_"+re.sub(r"[^A-Z0-9]+","_",c.upper())[:58],"EXPERT_BIGAL",mm,desc=f"Big Al atom: {c}",min_n=10,source_ok=True)
        if p_masks:
            psum=sum(x.fillna(False).astype(bool).astype("int8") for x in p_masks)
            add("EXPERT_PATHI_ANY","EXPERT_PATHI",psum.ge(1),desc="Any directional Pathi atom",min_n=20,source_ok=True)
            add("EXPERT_PATHI_MULTI_2PLUS","EXPERT_PATHI",psum.ge(2),desc="Two or more directional Pathi atoms",min_n=20,source_ok=True)
        if b_masks:
            bsum=sum(x.fillna(False).astype(bool).astype("int8") for x in b_masks)
            add("EXPERT_BIGAL_ANY","EXPERT_BIGAL",bsum.ge(1),desc="Any Big Al NCAAF atom",min_n=10,source_ok=True)

        # Market-journey context is distinct from a directional Pathi recommendation.
        for c,nm in (("Pathi_FB_Moved_Through_Key","KEY_JOURNEY_THROUGH"),("Pathi_FB_Moved_Onto_Key","KEY_JOURNEY_ONTO"),("Pathi_FB_Moved_Off_Key","KEY_JOURNEY_OFF")):
            add(nm,"MARKET_KEY_JOURNEY",nfirst(c).eq(1),source_ok=has(c),min_n=20)
        kval=nfirst("Pathi_FB_Key_Value_Change")
        add("KEY_VALUE_IMPROVED","MARKET_KEY_VALUE",kval.gt(0),source_ok=has("Pathi_FB_Key_Value_Change"),min_n=20)
        add("KEY_VALUE_WORSENED","MARKET_KEY_VALUE",kval.lt(0),source_ok=has("Pathi_FB_Key_Value_Change"),min_n=20)

        core=nfirst("_V29_CORE_INCUMBENT_EDGE_POINTS")
        if has("_V29_CORE_INCUMBENT_EDGE_POINTS"):
            add("CORE_OOF_EDGE_TEAM_2PLUS","RESEARCH_CORE_STATE",core.ge(2),desc="Incumbent CORE OOF edge >= +2",min_n=30)
            add("CORE_OOF_EDGE_TEAM_4PLUS","RESEARCH_CORE_STATE",core.ge(4),desc="Incumbent CORE OOF edge >= +4",min_n=30)
            add("CORE_OOF_EDGE_OPP_2PLUS","RESEARCH_CORE_STATE",core.le(-2),desc="Incumbent CORE OOF edge <= -2",min_n=30)
            add("CORE_OOF_EDGE_ABS_4PLUS","RESEARCH_CORE_STATE",core.abs().ge(4),desc="Incumbent CORE OOF absolute edge >= 4",min_n=30)

        # External fixed-weight Prediction Tracker metamodel. These states are
        # independent external intelligence. A derived Miner mechanism remains
        # zero-authority until it independently clears the same frozen system gate
        # as every other mechanism; source identity alone neither grants nor blocks it.
        meta_edge=nfirst("_V210_PT_META_EDGE_POINTS")
        if has("_V210_PT_META_EDGE_POINTS"):
            add("META_PT_EDGE_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",meta_edge.ge(2),desc="Prediction Tracker five-system metamodel edge >= +2",min_n=30)
            add("META_PT_EDGE_TEAM_3PLUS","EXTERNAL_RATINGS_FAMILY",meta_edge.ge(3),desc="Prediction Tracker five-system metamodel edge >= +3",min_n=30)
            add("META_PT_EDGE_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",meta_edge.le(-2),desc="Prediction Tracker five-system metamodel edge <= -2",min_n=30)
            add("META_PT_EDGE_OPP_3PLUS","EXTERNAL_RATINGS_FAMILY",meta_edge.le(-3),desc="Prediction Tracker five-system metamodel edge <= -3",min_n=30)
            add("META_PT_EDGE_ABS_4PLUS","EXTERNAL_RATINGS_FAMILY",meta_edge.abs().ge(4),desc="Prediction Tracker metamodel absolute edge >= 4",min_n=30)
            if has("_V29_CORE_INCUMBENT_EDGE_POINTS"):
                good=meta_edge.notna()&core.notna()
                add("META_PT_CORE_STRONG_AGREE","EXTERNAL_RATINGS_FAMILY",good&(meta_edge.abs().ge(2))&(core.abs().ge(2))&(np.sign(meta_edge)==np.sign(core)),desc="External metamodel and CORE both >=2 points same direction",min_n=30)
                add("META_PT_CORE_STRONG_CONFLICT","EXTERNAL_RATINGS_FAMILY",good&(meta_edge.abs().ge(2))&(core.abs().ge(2))&(np.sign(meta_edge)!=np.sign(core)),desc="External metamodel and CORE both >=2 points opposite direction",min_n=30)
                add("META_PT_CORE_GAP_4PLUS","EXTERNAL_RATINGS_FAMILY",good&(meta_edge-core).abs().ge(4),desc="External metamodel differs from CORE edge by >=4 points",min_n=30)

        # Preserve the five component ratings as diagnostics/research atoms instead
        # of reducing the external family to one weighted average.  They remain one
        # correlated EXTERNAL family for authority purposes; five agreements are not
        # five independent Bet Authority votes.
        for _k in PT_PUBLISHED_WEIGHTS:
            _ec=f"_V212_PT_{_k}_EDGE_POINTS"
            if has(_ec):
                _ee=nfirst(_ec); _slug=re.sub(r"[^A-Z0-9]+","_",_k.upper())
                add(f"META_PT_{_slug}_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",_ee.ge(2),desc=f"{_k} external edge >= +2",min_n=30)
                add(f"META_PT_{_slug}_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",_ee.le(-2),desc=f"{_k} external edge <= -2",min_n=30)
                if has("_V29_CORE_INCUMBENT_EDGE_POINTS"):
                    _good=_ee.notna()&core.notna()&(_ee.abs().ge(2))&(core.abs().ge(2))
                    add(f"META_PT_{_slug}_CORE_AGREE", "EXTERNAL_RATINGS_FAMILY", _good&(np.sign(_ee)==np.sign(core)), desc=f"{_k} and CORE strong agreement", min_n=30)
                    add(f"META_PT_{_slug}_CORE_CONFLICT", "EXTERNAL_RATINGS_FAMILY", _good&(np.sign(_ee)!=np.sign(core)), desc=f"{_k} and CORE strong conflict", min_n=30)
        _ta=nfirst("_V212_PT_COMPONENT_TEAM_AGREE_COUNT"); _oa=nfirst("_V212_PT_COMPONENT_OPP_AGREE_COUNT"); _ds=nfirst("_V212_PT_COMPONENT_STD")
        _avail=nfirst("_V217_PT_COMPONENT_AVAILABLE_COUNT") if has("_V217_PT_COMPONENT_AVAILABLE_COUNT") else nfirst("_V210_PT_META_SYSTEM_COUNT")
        _all5=_avail.eq(5)
        if has("_V212_PT_COMPONENT_TEAM_AGREE_COUNT"):
            add("META_PT_COMPONENTS_5_OF_5_TEAM","EXTERNAL_RATINGS_FAMILY",_all5&_ta.ge(5),desc="All five available external systems favor team vs market",min_n=30)
            add("META_PT_COMPONENTS_4PLUS_TEAM","EXTERNAL_RATINGS_FAMILY",_all5&_ta.ge(4),desc="At least four of five external systems favor team vs market; full five-system coverage required",min_n=30)
        if has("_V212_PT_COMPONENT_OPP_AGREE_COUNT"):
            add("META_PT_COMPONENTS_5_OF_5_OPP","EXTERNAL_RATINGS_FAMILY",_all5&_oa.ge(5),desc="All five available external systems favor opponent vs market",min_n=30)
            add("META_PT_COMPONENTS_4PLUS_OPP","EXTERNAL_RATINGS_FAMILY",_all5&_oa.ge(4),desc="At least four of five external systems favor opponent vs market; full five-system coverage required",min_n=30)
        if has("_V212_PT_COMPONENT_STD"):
            add("META_PT_LOW_DISPERSION_LE3","EXTERNAL_RATINGS_FAMILY",_all5&_ds.le(3)&_ds.notna(),desc="Five-system margin dispersion <=3 points; full five-system coverage required",min_n=30)
            add("META_PT_HIGH_DISPERSION_GE6","EXTERNAL_RATINGS_FAMILY",_all5&_ds.ge(6),desc="Five-system margin dispersion >=6 points; full five-system coverage required",min_n=30)

        # V2.18: full Prediction Tracker index universe. Every exact source-native
        # predictor gets a directional edge atom, while all indices remain ONE
        # correlated external family. The Miner can therefore combine one external
        # signal with Pathi/Big Al/CORE/specialists, but cannot stack many correlated
        # rating systems as if they were independent authority votes.
        _ptidx_cols=sorted([
            c for c in g.columns
            if str(c).startswith("_V218_PTIDX_") and str(c).endswith("_EDGE_POINTS")
            and str(c) not in {
                "_V218_PTIDX_MEAN_EDGE_POINTS",
                "_V218_PTIDX_MEDIAN_EDGE_POINTS",
                "_V218_PTIDX_CLUSTER_MEDIAN_EDGE_POINTS",
            }
        ])
        for _c in _ptidx_cols:
            _ee=nfirst(_c)
            _slug=str(_c)[len("_V218_PTIDX_"):-len("_EDGE_POINTS")]
            add(f"PTIDX_{_slug}_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",_ee.ge(2),
                desc=f"Prediction Tracker {_slug} edge >= +2",min_n=30)
            add(f"PTIDX_{_slug}_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",_ee.le(-2),
                desc=f"Prediction Tracker {_slug} edge <= -2",min_n=30)

        _idx_n=nfirst("_V218_PTIDX_AVAILABLE_COUNT")
        _idx_med=nfirst("_V218_PTIDX_MEDIAN_EDGE_POINTS")
        _idx_mean=nfirst("_V218_PTIDX_MEAN_EDGE_POINTS")
        _idx_tf=nfirst("_V218_PTIDX_TEAM_AGREE_FRAC")
        _idx_of=nfirst("_V218_PTIDX_OPP_AGREE_FRAC")
        _idx_stf=nfirst("_V218_PTIDX_STRONG_TEAM_FRAC")
        _idx_sof=nfirst("_V218_PTIDX_STRONG_OPP_FRAC")
        _idx_std=nfirst("_V218_PTIDX_STD")
        _idx_iqr=nfirst("_V218_PTIDX_IQR")
        _cl_n=nfirst("_V218_PTIDX_CLUSTER_COUNT")
        _cl_med=nfirst("_V218_PTIDX_CLUSTER_MEDIAN_EDGE_POINTS")
        _cl_tf=nfirst("_V218_PTIDX_CLUSTER_TEAM_AGREE_FRAC")
        _cl_of=nfirst("_V218_PTIDX_CLUSTER_OPP_AGREE_FRAC")
        _enough=_idx_n.ge(8)
        _cl_enough=_cl_n.ge(8)

        if has("_V218_PTIDX_MEDIAN_EDGE_POINTS"):
            add("PT_ALL_MEDIAN_EDGE_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",_enough&_idx_med.ge(2),desc="All-index median edge >= +2 with >=8 available systems",min_n=30)
            add("PT_ALL_MEDIAN_EDGE_TEAM_3PLUS","EXTERNAL_RATINGS_FAMILY",_enough&_idx_med.ge(3),desc="All-index median edge >= +3 with >=8 available systems",min_n=30)
            add("PT_ALL_MEDIAN_EDGE_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",_enough&_idx_med.le(-2),desc="All-index median edge <= -2 with >=8 available systems",min_n=30)
            add("PT_ALL_MEDIAN_EDGE_OPP_3PLUS","EXTERNAL_RATINGS_FAMILY",_enough&_idx_med.le(-3),desc="All-index median edge <= -3 with >=8 available systems",min_n=30)
        if has("_V218_PTIDX_MEAN_EDGE_POINTS"):
            add("PT_ALL_MEAN_EDGE_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",_enough&_idx_mean.ge(2),desc="All-index mean edge >= +2 with >=8 available systems",min_n=30)
            add("PT_ALL_MEAN_EDGE_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",_enough&_idx_mean.le(-2),desc="All-index mean edge <= -2 with >=8 available systems",min_n=30)
        if has("_V218_PTIDX_TEAM_AGREE_FRAC"):
            for _pct,_cut in ((65,.65),(75,.75),(85,.85)):
                add(f"PT_ALL_CONSENSUS_TEAM_{_pct}","EXTERNAL_RATINGS_FAMILY",_enough&_idx_tf.ge(_cut),desc=f">={_pct}% of available external indices favor team vs market",min_n=30)
        if has("_V218_PTIDX_OPP_AGREE_FRAC"):
            for _pct,_cut in ((65,.65),(75,.75),(85,.85)):
                add(f"PT_ALL_CONSENSUS_OPP_{_pct}","EXTERNAL_RATINGS_FAMILY",_enough&_idx_of.ge(_cut),desc=f">={_pct}% of available external indices favor opponent vs market",min_n=30)
        if has("_V218_PTIDX_STRONG_TEAM_FRAC"):
            add("PT_ALL_STRONG_EDGE_TEAM_65","EXTERNAL_RATINGS_FAMILY",_enough&_idx_stf.ge(.65),desc=">=65% of available external indices have team edge >=2",min_n=30)
        if has("_V218_PTIDX_STRONG_OPP_FRAC"):
            add("PT_ALL_STRONG_EDGE_OPP_65","EXTERNAL_RATINGS_FAMILY",_enough&_idx_sof.ge(.65),desc=">=65% of available external indices have opponent edge >=2",min_n=30)
        if has("_V218_PTIDX_STD"):
            add("PT_ALL_LOW_DISPERSION_LE3","EXTERNAL_RATINGS_FAMILY",_enough&_idx_std.le(3)&_idx_std.notna(),desc="All-index prediction dispersion <=3 points",min_n=30)
            add("PT_ALL_HIGH_DISPERSION_GE6","EXTERNAL_RATINGS_FAMILY",_enough&_idx_std.ge(6),desc="All-index prediction dispersion >=6 points",min_n=30)
        if has("_V218_PTIDX_IQR"):
            add("PT_ALL_IQR_LE4","EXTERNAL_RATINGS_FAMILY",_enough&_idx_iqr.le(4)&_idx_iqr.notna(),desc="All-index interquartile range <=4 points",min_n=30)
            add("PT_ALL_IQR_GE8","EXTERNAL_RATINGS_FAMILY",_enough&_idx_iqr.ge(8),desc="All-index interquartile range >=8 points",min_n=30)

        # Source-cluster balanced consensus: Sagarin/Pi/regression/Payne variants
        # receive one cluster vote each, avoiding accidental over-weighting.
        if has("_V218_PTIDX_CLUSTER_MEDIAN_EDGE_POINTS"):
            add("PT_CLUSTER_MEDIAN_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",_cl_enough&_cl_med.ge(2),desc="Source-cluster balanced median edge >= +2",min_n=30)
            add("PT_CLUSTER_MEDIAN_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",_cl_enough&_cl_med.le(-2),desc="Source-cluster balanced median edge <= -2",min_n=30)
        if has("_V218_PTIDX_CLUSTER_TEAM_AGREE_FRAC"):
            add("PT_CLUSTER_CONSENSUS_TEAM_70","EXTERNAL_RATINGS_FAMILY",_cl_enough&_cl_tf.ge(.70),desc=">=70% of external source clusters favor team",min_n=30)
        if has("_V218_PTIDX_CLUSTER_OPP_AGREE_FRAC"):
            add("PT_CLUSTER_CONSENSUS_OPP_70","EXTERNAL_RATINGS_FAMILY",_cl_enough&_cl_of.ge(.70),desc=">=70% of external source clusters favor opponent",min_n=30)

        # V2.18.2: broad consensus with every frozen META constituent source
        # removed. Prefer these when asking whether "the rest of Prediction
        # Tracker" independently agrees with the fixed five-system META.
        _exn=nfirst("_V2182_PT_EXMETA_AVAILABLE_COUNT")
        _exmed=nfirst("_V2182_PT_EXMETA_MEDIAN_EDGE_POINTS")
        _exmean=nfirst("_V2182_PT_EXMETA_MEAN_EDGE_POINTS")
        _extf=nfirst("_V2182_PT_EXMETA_TEAM_AGREE_FRAC")
        _exof=nfirst("_V2182_PT_EXMETA_OPP_AGREE_FRAC")
        _exstf=nfirst("_V2182_PT_EXMETA_STRONG_TEAM_FRAC")
        _exsof=nfirst("_V2182_PT_EXMETA_STRONG_OPP_FRAC")
        _exstd=nfirst("_V2182_PT_EXMETA_STD")
        _exiqr=nfirst("_V2182_PT_EXMETA_IQR")
        _excln=nfirst("_V2182_PT_EXMETA_CLUSTER_COUNT")
        _exclmed=nfirst("_V2182_PT_EXMETA_CLUSTER_MEDIAN_EDGE_POINTS")
        _excltf=nfirst("_V2182_PT_EXMETA_CLUSTER_TEAM_AGREE_FRAC")
        _exclof=nfirst("_V2182_PT_EXMETA_CLUSTER_OPP_AGREE_FRAC")
        _ex_enough=_exn.ge(8)
        _excl_enough=_excln.ge(8)

        if has("_V2182_PT_EXMETA_MEDIAN_EDGE_POINTS"):
            add("PT_EXMETA_MEDIAN_EDGE_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",_ex_enough&_exmed.ge(2),desc="META-source-out median external edge >= +2",min_n=30)
            add("PT_EXMETA_MEDIAN_EDGE_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",_ex_enough&_exmed.le(-2),desc="META-source-out median external edge <= -2",min_n=30)
        if has("_V2182_PT_EXMETA_MEAN_EDGE_POINTS"):
            add("PT_EXMETA_MEAN_EDGE_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",_ex_enough&_exmean.ge(2),desc="META-source-out mean external edge >= +2",min_n=30)
            add("PT_EXMETA_MEAN_EDGE_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",_ex_enough&_exmean.le(-2),desc="META-source-out mean external edge <= -2",min_n=30)
        if has("_V2182_PT_EXMETA_TEAM_AGREE_FRAC"):
            for _pct,_cut in ((65,.65),(75,.75),(85,.85)):
                add(f"PT_EXMETA_CONSENSUS_TEAM_{_pct}","EXTERNAL_RATINGS_FAMILY",_ex_enough&_extf.ge(_cut),desc=f">={_pct}% of META-source-out indices favor team",min_n=30)
        if has("_V2182_PT_EXMETA_OPP_AGREE_FRAC"):
            for _pct,_cut in ((65,.65),(75,.75),(85,.85)):
                add(f"PT_EXMETA_CONSENSUS_OPP_{_pct}","EXTERNAL_RATINGS_FAMILY",_ex_enough&_exof.ge(_cut),desc=f">={_pct}% of META-source-out indices favor opponent",min_n=30)
        if has("_V2182_PT_EXMETA_STRONG_TEAM_FRAC"):
            add("PT_EXMETA_STRONG_EDGE_TEAM_65","EXTERNAL_RATINGS_FAMILY",_ex_enough&_exstf.ge(.65),desc=">=65% of META-source-out indices have team edge >=2",min_n=30)
        if has("_V2182_PT_EXMETA_STRONG_OPP_FRAC"):
            add("PT_EXMETA_STRONG_EDGE_OPP_65","EXTERNAL_RATINGS_FAMILY",_ex_enough&_exsof.ge(.65),desc=">=65% of META-source-out indices have opponent edge >=2",min_n=30)
        if has("_V2182_PT_EXMETA_STD"):
            add("PT_EXMETA_LOW_DISPERSION_LE3","EXTERNAL_RATINGS_FAMILY",_ex_enough&_exstd.le(3)&_exstd.notna(),desc="META-source-out prediction dispersion <=3",min_n=30)
            add("PT_EXMETA_HIGH_DISPERSION_GE6","EXTERNAL_RATINGS_FAMILY",_ex_enough&_exstd.ge(6),desc="META-source-out prediction dispersion >=6",min_n=30)
        if has("_V2182_PT_EXMETA_IQR"):
            add("PT_EXMETA_IQR_LE4","EXTERNAL_RATINGS_FAMILY",_ex_enough&_exiqr.le(4)&_exiqr.notna(),desc="META-source-out prediction IQR <=4",min_n=30)
        if has("_V2182_PT_EXMETA_CLUSTER_MEDIAN_EDGE_POINTS"):
            add("PT_EXMETA_CLUSTER_MEDIAN_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",_excl_enough&_exclmed.ge(2),desc="META-source-out cluster-balanced median edge >= +2",min_n=30)
            add("PT_EXMETA_CLUSTER_MEDIAN_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",_excl_enough&_exclmed.le(-2),desc="META-source-out cluster-balanced median edge <= -2",min_n=30)
        if has("_V2182_PT_EXMETA_CLUSTER_TEAM_AGREE_FRAC"):
            add("PT_EXMETA_CLUSTER_CONSENSUS_TEAM_70","EXTERNAL_RATINGS_FAMILY",_excl_enough&_excltf.ge(.70),desc=">=70% of META-source-out external clusters favor team",min_n=30)
        if has("_V2182_PT_EXMETA_CLUSTER_OPP_AGREE_FRAC"):
            add("PT_EXMETA_CLUSTER_CONSENSUS_OPP_70","EXTERNAL_RATINGS_FAMILY",_excl_enough&_exclof.ge(.70),desc=">=70% of META-source-out external clusters favor opponent",min_n=30)

        # Prediction Tracker's own aggregate lineavg/linemedian are retained as
        # separate diagnostics but still part of the same correlated family.
        _ptavg=nfirst("_V218_PT_TRACKER_AVG_EDGE_POINTS")
        _ptmed=nfirst("_V218_PT_TRACKER_MEDIAN_EDGE_POINTS")
        if has("_V218_PT_TRACKER_AVG_EDGE_POINTS"):
            add("PT_TRACKER_AVG_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",_ptavg.ge(2),desc="Prediction Tracker published average edge >= +2",min_n=30)
            add("PT_TRACKER_AVG_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",_ptavg.le(-2),desc="Prediction Tracker published average edge <= -2",min_n=30)
        if has("_V218_PT_TRACKER_MEDIAN_EDGE_POINTS"):
            add("PT_TRACKER_MEDIAN_TEAM_2PLUS","EXTERNAL_RATINGS_FAMILY",_ptmed.ge(2),desc="Prediction Tracker published median edge >= +2",min_n=30)
            add("PT_TRACKER_MEDIAN_OPP_2PLUS","EXTERNAL_RATINGS_FAMILY",_ptmed.le(-2),desc="Prediction Tracker published median edge <= -2",min_n=30)

        spec_edge_cols=[c for c in g.columns if str(c).startswith("_V29_SPEC_") and str(c).endswith("_EDGE_POINTS")]
        for c in sorted(spec_edge_cols):
            slug=str(c)[len("_V29_SPEC_"):-len("_EDGE_POINTS")]
            se=nfirst(c); divc=f"_V29_SPEC_{slug}_DIVERGENCE_FROM_CORE"; cutc=f"_V29_SPEC_{slug}_DIVERGENCE_CUT"
            fam="RESEARCH_SPECIALIST_"+slug[:28]
            add(f"SPEC_{slug}_EDGE_TEAM_2PLUS",fam,se.ge(2),desc=f"{slug} specialist OOF edge >= +2",min_n=30)
            add(f"SPEC_{slug}_EDGE_OPP_2PLUS",fam,se.le(-2),desc=f"{slug} specialist OOF edge <= -2",min_n=30)
            if has("_V29_CORE_INCUMBENT_EDGE_POINTS"):
                good=core.notna()&se.notna()
                add(f"SPEC_{slug}_CORE_STRONG_AGREE",fam,good&(core.abs().ge(2))&(se.abs().ge(2))&(np.sign(core)==np.sign(se)),desc=f"{slug} and CORE strong agreement",min_n=30)
                add(f"SPEC_{slug}_CORE_STRONG_CONFLICT",fam,good&(core.abs().ge(2))&(se.abs().ge(2))&(np.sign(core)!=np.sign(se)),desc=f"{slug} and CORE strong conflict",min_n=30)
            if has(divc,cutc):
                dv=nfirst(divc); dc=nfirst(cutc)
                add(f"SPEC_{slug}_CORE_DIVERGENCE",fam,dv.ge(dc)&dc.notna(),desc=f"{slug} discovery-frozen divergence from CORE",min_n=30)

    # Conference identity / pairs, rivalry and team-specific memory.
    conf=tfirst("Conference","Team_Conference","Conference_Norm","Context_Conference")
    oppconf=tfirst("Opponent_Conference","Opp_Conference","Opponent_Conference_Norm","Context_Opp_Conference")
    if conf.ne("").any():
        vc=conf.value_counts()
        for v,cnt in vc.items():
            if v and v not in {"UNKNOWN"} and (for_live or cnt>=80):
                add("CONF_"+re.sub(r"[^A-Z0-9]+","_",v)[:28],"CONFERENCE",conf.eq(v),f"Conference={v}",source_ok=True)
    if conf.ne("").any() and oppconf.ne("").any():
        add("SAME_CONFERENCE","CONFERENCE_PAIR",conf.eq(oppconf)&conf.ne(""),source_ok=True)
        pairs=conf+"__VS__"+oppconf; vc=pairs.value_counts()
        for v,cnt in vc.items():
            if "__VS__" in v and "UNKNOWN" not in v and (for_live or cnt>=60):
                add("CONFPAIR_"+re.sub(r"[^A-Z0-9]+","_",v)[:36],"CONFERENCE_PAIR",pairs.eq(v),v.replace("__VS__"," vs "),source_ok=True)
    coach=tfirst("Head_Coach","Coach","Team_Head_Coach")
    if coach.ne("").any():
        for v,cnt in coach.value_counts().items():
            if v and (for_live or cnt>=40): add("COACH_"+re.sub(r"[^A-Z0-9]+","_",v)[:32],"COACH_ERA",coach.eq(v),f"Coach={v}",source_ok=True)
    team=tfirst("Team_Norm","Team","Home_Team_Norm","Home_Team")
    if team.ne("").any():
        for v,cnt in team.value_counts().items():
            if v and (for_live or cnt>=35): add("TEAM_"+re.sub(r"[^A-Z0-9]+","_",v)[:32],"TEAM_SPECIFIC",team.eq(v),f"Team={v}",source_ok=True)

    subdiv=tfirst("Opponent_Subdivision","Opp_Subdivision","Context_Opp_Subdivision")
    add("VS_FCS","OPPONENT_CLASS",subdiv.str.contains("FCS",na=False),source_ok=has("Opponent_Subdivision","Opp_Subdivision","Context_Opp_Subdivision"),min_n=20)
    return atoms

def _market_target(g: pd.DataFrame, market: str):
    market=str(market).lower(); am=_num(g,"Actual_Margin").to_numpy(dtype=float); at=_num(g,"Actual_Total").to_numpy(dtype=float)
    sp=_num(g,"Consensus_Open_Spread").to_numpy(dtype=float); tt=_num(g,"Consensus_Open_Total").to_numpy(dtype=float)
    if market=="spreads":
        raw=am+sp; valid=np.isfinite(raw)&~np.isclose(raw,0,atol=1e-9); y=(raw>0).astype(float); baseline=np.full(len(g),.5)
    elif market=="totals":
        raw=at-tt; valid=np.isfinite(raw)&~np.isclose(raw,0,atol=1e-9); y=(raw>0).astype(float); baseline=np.full(len(g),.5)
    else:
        valid=np.isfinite(am)&~np.isclose(am,0,atol=1e-9); y=(am>0).astype(float); baseline=_num(g,"Market_Open_H2H_Fair").to_numpy(dtype=float)
    return y,valid,baseline


def _one_sided_p(rate: float, n: int) -> float:
    if n<=0 or not np.isfinite(rate): return np.nan
    z=(rate-.5)/max(math.sqrt(.25/n),1e-9)
    return .5*(1-math.erf(z/math.sqrt(2)))


def _first_num_array(g: pd.DataFrame, names: Iterable[str]) -> np.ndarray:
    out=np.full(len(g),np.nan,dtype=float)
    for c in names:
        if c not in g.columns: continue
        v=pd.to_numeric(g[c],errors="coerce").to_numpy(dtype=float)
        take=~np.isfinite(out)&np.isfinite(v); out[take]=v[take]
    return out


def _american_implied(odds: np.ndarray) -> np.ndarray:
    o=np.asarray(odds,dtype=float); p=np.full(len(o),np.nan,dtype=float)
    neg=np.isfinite(o)&(o<0); pos=np.isfinite(o)&(o>0)
    p[neg]=(-o[neg])/((-o[neg])+100.0); p[pos]=100.0/(o[pos]+100.0)
    return p


def _american_unit_return(odds: np.ndarray, won: np.ndarray) -> np.ndarray:
    o=np.asarray(odds,dtype=float); w=np.asarray(won,dtype=float); r=np.full(len(o),np.nan,dtype=float)
    good=np.isfinite(o)&(np.abs(o)>=100)&np.isfinite(w)
    prof=np.full(len(o),np.nan,dtype=float)
    pos=good&(o>0); neg=good&(o<0)
    prof[pos]=o[pos]/100.0; prof[neg]=100.0/np.abs(o[neg])
    r[good]=np.where(w[good]>0.5,prof[good],-1.0)
    return r


def _price_band(o: float) -> str:
    if not np.isfinite(o): return "MISSING"
    if o<=-500: return "<=-500"
    if o<=-300: return "-499..-300"
    if o<=-200: return "-299..-200"
    if o<=-150: return "-199..-150"
    if o<0: return "-149..-100"
    if o<=150: return "+100..+150"
    if o<=250: return "+151..+250"
    return "+251+"


def _market_residual_p(obs: np.ndarray, exp: np.ndarray, mask: np.ndarray) -> float:
    ok=np.asarray(mask,dtype=bool)&np.isfinite(obs)&np.isfinite(exp)&(exp>0)&(exp<1)
    if ok.sum()<20: return np.nan
    diff=float(np.sum(obs[ok]-exp[ok])); var=float(np.sum(exp[ok]*(1-exp[ok])))
    if var<=1e-12: return np.nan
    z=diff/math.sqrt(var)
    return .5*(1-math.erf(z/math.sqrt(2)))


def _h2h_price_metrics(g: pd.DataFrame, mask: np.ndarray, y: np.ndarray, baseline: np.ndarray,
                       seasons: np.ndarray, direction: str, season_scope: Iterable[int]) -> dict[str,Any]:
    team_odds=_first_num_array(g,["Consensus_Open_Moneyline","Opening_ML_Odds","Opening_Moneyline","First_Odds_Price","First_Odds","Open_Odds_Price","Open_Odds"])
    opp_odds=_first_num_array(g,["Opp_Consensus_Open_Moneyline","Opponent_Consensus_Open_Moneyline","Opp_Opening_ML_Odds","Opponent_Opening_ML_Odds"])
    obs=y if direction=="PLAY_ON" else 1-y
    exp=baseline if direction=="PLAY_ON" else 1-baseline
    odds=team_odds if direction=="PLAY_ON" else opp_odds
    scope=np.asarray(mask,dtype=bool)&np.isin(seasons,np.asarray(list(season_scope),dtype=float))
    ret=_american_unit_return(odds,obs)
    good=scope&np.isfinite(ret)&np.isfinite(exp)&np.isfinite(obs)
    resid_good=scope&np.isfinite(exp)&np.isfinite(obs)
    years=[]
    for sy in sorted(set(int(x) for x in seasons[scope&np.isfinite(seasons)])):
        jj=scope&(seasons==sy); pg=jj&np.isfinite(ret); rg=jj&np.isfinite(exp)&np.isfinite(obs)
        years.append({
            "season":sy,"n":int(jj.sum()),"priced_n":int(pg.sum()),
            "rate":float(np.mean(obs[jj&np.isfinite(obs)])) if (jj&np.isfinite(obs)).sum() else np.nan,
            "roi":float(np.mean(ret[pg])) if pg.sum() else np.nan,
            "market_residual":float(np.mean(obs[rg]-exp[rg])) if rg.sum() else np.nan,
        })
    band_rows=[]
    if good.any():
        bands=np.asarray([_price_band(v) for v in odds],dtype=object)
        for b in sorted(set(bands[good])):
            jj=good&(bands==b)
            if jj.sum(): band_rows.append({"band":b,"n":int(jj.sum()),"roi":float(np.mean(ret[jj])),"rate":float(np.mean(obs[jj]))})
    eligible=[x for x in band_rows if x["n"]>=15]
    pos_frac=float(np.mean([x["roi"]>=0 for x in eligible])) if eligible else np.nan
    max_share=max([x["n"] for x in eligible],default=0)/max(sum(x["n"] for x in eligible),1) if eligible else np.nan
    band_robust=bool(len(eligible)>=2 and pos_frac>=.5 and min(x["roi"] for x in eligible)>=-.08 and max_share<=.85)
    return {
        "priced_n":int(good.sum()),
        "roi":float(np.mean(ret[good])) if good.any() else np.nan,
        "market_residual":float(np.mean(obs[resid_good]-exp[resid_good])) if resid_good.any() else np.nan,
        "expected_rate":float(np.mean(exp[resid_good])) if resid_good.any() else np.nan,
        "actual_rate":float(np.mean(obs[resid_good])) if resid_good.any() else np.nan,
        "seasons":years,"price_bands":band_rows,"eligible_price_bands":len(eligible),
        "positive_price_band_fraction":pos_frac,"max_price_band_share":max_share,"price_band_robust":band_robust,
        "observed_price_source":"TEAM_AND_OPPONENT_OPEN_MONEYLINE_ONLY__NO_FAIR_PRICE_PROXY",
    }


def _evaluate_rule(g: pd.DataFrame, mask: np.ndarray, y: np.ndarray, valid: np.ndarray, baseline: np.ndarray, seasons: np.ndarray,
                   names: tuple[str,...], families: tuple[str,...], market: str, *,
                   min_discovery_n_override: int | None=None,
                   min_discovery_years_override: int=2,
                   min_discovery_year_n_override: int=15) -> dict[str,Any] | None:
    disc=mask&valid&np.isfinite(seasons)&(seasons<=DISCOVERY_MAX_SEASON)
    ix=np.flatnonzero(disc)
    team_specific="TEAM_SPECIFIC" in families
    min_n=int(min_discovery_n_override) if min_discovery_n_override is not None else (60 if team_specific else 100)
    if len(ix)<min_n: return None

    h2h_price={}
    if market=="h2h":
        candidates=[]
        for d in ("PLAY_ON","FADE"):
            pm=_h2h_price_metrics(g,mask&valid,y,baseline,seasons,d,[x for x in sorted(set(seasons[np.isfinite(seasons)].astype(int))) if x<=DISCOVERY_MAX_SEASON])
            score=(pm["roi"] if np.isfinite(pm["roi"]) else -9.0)+2.0*(pm["market_residual"] if np.isfinite(pm["market_residual"]) else -9.0)
            candidates.append((score,d,pm))
        _,direction,h2h_price=max(candidates,key=lambda x:x[0])
        obs=y if direction=="PLAY_ON" else 1-y
    else:
        raw=float(np.mean(y[ix])); direction="PLAY_ON" if raw>=.5 else "FADE"; obs=y if direction=="PLAY_ON" else 1-y
    rate=float(np.mean(obs[ix]))

    dyears=[]
    for sy in sorted({int(x) for x in seasons[ix]}):
        jj=disc&(seasons==sy)
        if jj.sum()>=int(min_discovery_year_n_override):
            row={"season":sy,"n":int(jj.sum()),"rate":float(np.mean(obs[jj]))}
            if market=="h2h":
                pm=_h2h_price_metrics(g,jj,y,baseline,seasons,direction,[sy]); row.update({"priced_n":pm["priced_n"],"roi":pm["roi"],"market_residual":pm["market_residual"]})
            dyears.append(row)
    if len(dyears)<int(min_discovery_years_override): return None
    stable=float(np.mean([(x.get("roi",0)>=0 and x.get("market_residual",0)>0) if market=="h2h" else x["rate"]>.5 for x in dyears]))
    best=max(dyears,key=lambda x:(x.get("roi",x["rate"]) if np.isfinite(x.get("roi",np.nan)) else x["rate"]))["season"]
    rem=disc&(seasons!=best)
    remove_best=float(np.mean(obs[rem])) if rem.sum()>=25 else np.nan
    remove_best_roi=remove_best_resid=np.nan
    if market=="h2h" and rem.sum()>=25:
        rpm=_h2h_price_metrics(g,rem,y,baseline,seasons,direction,[int(x) for x in set(seasons[rem].astype(int))])
        remove_best_roi=rpm["roi"]; remove_best_resid=rpm["market_residual"]

    conf=[]
    for sy in CONFIRMATION_SEASONS:
        jj=mask&valid&np.isfinite(seasons)&(seasons==sy)
        if jj.sum():
            row={"season":sy,"n":int(jj.sum()),"rate":float(np.mean(obs[jj]))}
            if market=="h2h":
                pm=_h2h_price_metrics(g,jj,y,baseline,seasons,direction,[sy]); row.update({"priced_n":pm["priced_n"],"roi":pm["roi"],"market_residual":pm["market_residual"]})
            conf.append(row)
    conf_n=sum(x["n"] for x in conf); conf_rate=float(np.average([x["rate"] for x in conf],weights=[x["n"] for x in conf])) if conf_n else np.nan

    # V2.18 explicit discovery/confirmation split audit. Identical N/rate can occur
    # by coincidence, but the same row must never appear in both periods.
    _conf_mask=mask&valid&np.isfinite(seasons)&np.isin(seasons,np.asarray(CONFIRMATION_SEASONS,dtype=float))
    _split_overlap_n=int(np.sum(disc&_conf_mask))
    _split_disjoint=bool(_split_overlap_n==0)
    _disc_years=sorted(set(int(x) for x in seasons[disc&np.isfinite(seasons)]))
    _conf_years=sorted(set(int(x) for x in seasons[_conf_mask&np.isfinite(seasons)]))

    market_resid_disc=market_resid_conf=np.nan; h2h_conf_price={}
    nominal=_one_sided_p(rate,len(ix))
    if market=="h2h":
        market_resid_disc=h2h_price.get("market_residual",np.nan)
        h2h_conf_price=_h2h_price_metrics(g,mask&valid,y,baseline,seasons,direction,CONFIRMATION_SEASONS)
        market_resid_conf=h2h_conf_price.get("market_residual",np.nan)
        exp=baseline if direction=="PLAY_ON" else 1-baseline
        nominal=_market_residual_p(obs,exp,disc)
    return {
        "conditions":list(names),"families":list(families),"direction":direction,"discovery_n":len(ix),"discovery_rate":rate,
        "discovery_seasons":dyears,"stable_discovery_fraction":stable,"remove_best_discovery_rate":remove_best,
        "remove_best_discovery_roi":remove_best_roi,"remove_best_market_residual":remove_best_resid,
        "nominal_pvalue":nominal,"confirmation":conf,"confirmation_n":conf_n,"confirmation_rate":conf_rate,
        "market_residual_discovery":market_resid_disc,"market_residual_confirmation":market_resid_conf,
        "h2h_discovery_price":h2h_price,"h2h_confirmation_price":h2h_conf_price,
        "split_audit":{
            "discovery_years":_disc_years,
            "confirmation_years":_conf_years,
            "row_overlap_n":_split_overlap_n,
            "disjoint":_split_disjoint,
            "identical_n_rate":bool(conf_n==len(ix) and np.isfinite(conf_rate) and abs(conf_rate-rate)<1e-12),
        },
        "mask":mask,
    }


def _concentration(g: pd.DataFrame, mask: np.ndarray, candidates: Iterable[str]) -> dict[str,Any]:
    for c in candidates:
        if c not in g.columns: continue
        z=g.loc[np.asarray(mask,dtype=bool),c].astype(str).replace({"nan":"","None":""})
        z=z[z.str.len()>0]
        if z.empty: continue
        vc=z.value_counts(); return {"field":c,"top":str(vc.index[0]),"top_n":int(vc.iloc[0]),"top_share":float(vc.iloc[0]/vc.sum()),"unique":int(len(vc))}
    return {"field":None,"top":None,"top_n":0,"top_share":np.nan,"unique":0}


def _mechanism_attribution(g: pd.DataFrame, seasons: np.ndarray, rep: dict[str,Any], market: str) -> dict[str,Any]:
    mask=np.asarray(rep.get("_mask_internal"),dtype=bool)
    attr={
        "rule":" AND ".join(rep.get("conditions") or []),
        "discovery_seasons":rep.get("discovery_seasons") or [],
        "confirmation_seasons":rep.get("confirmation") or [],
        "remove_best_discovery_rate":rep.get("remove_best_discovery_rate"),
        "team_concentration":_concentration(g,mask,["Team_Norm","Team","A_Team","Home_Team"]),
        "conference_concentration":_concentration(g,mask,["Conference","Conf","A_Conference","Team_Conference","Home_Conference"]),
    }
    if market in ("spreads","totals"):
        rows=[]
        y,valid,_=_market_target(g,market); obs=y if rep.get("direction")=="PLAY_ON" else 1-y
        for sy in sorted(set(int(x) for x in seasons[mask&valid&np.isfinite(seasons)])):
            jj=mask&valid&(seasons==sy); rate=float(np.mean(obs[jj])) if jj.sum() else np.nan
            roi=rate*(100/110.0)-(1-rate) if np.isfinite(rate) else np.nan
            rows.append({"season":sy,"n":int(jj.sum()),"rate":rate,"roi_at_minus110":roi})
        attr["season_breakdown"]=rows
        # Directional market-error magnitude: positive means the recommended side
        # beat the opening market number. This is diagnostic only and complements W/L.
        if market=="spreads":
            raw=_num(g,"Actual_Margin").to_numpy(dtype=float)+_num(g,"Consensus_Open_Spread").to_numpy(dtype=float)
        else:
            raw=_num(g,"Actual_Total").to_numpy(dtype=float)-_num(g,"Consensus_Open_Total").to_numpy(dtype=float)
        dres=raw if rep.get("direction")=="PLAY_ON" else -raw
        for scope_name,scope_years in (("discovery",[x for x in sorted(set(seasons[np.isfinite(seasons)].astype(int))) if x<=DISCOVERY_MAX_SEASON]),("confirmation",CONFIRMATION_SEASONS)):
            mm=mask&np.isfinite(dres)&np.isin(seasons,np.asarray(scope_years,dtype=float))
            attr[f"{scope_name}_directional_market_residual"]={
                "n":int(mm.sum()),
                "mean":float(np.mean(dres[mm])) if mm.any() else np.nan,
                "median":float(np.median(dres[mm])) if mm.any() else np.nan,
                "positive_fraction":float(np.mean(dres[mm]>0)) if mm.any() else np.nan,
            }
    else:
        attr["discovery_price"]=rep.get("h2h_discovery_price") or {}
        attr["confirmation_price"]=rep.get("h2h_confirmation_price") or {}
    return attr



def _directional_observation(y: np.ndarray, direction: str) -> np.ndarray:
    """Outcome on the system's recommended side."""
    return np.asarray(y,dtype=float) if str(direction).upper()=="PLAY_ON" else 1.0-np.asarray(y,dtype=float)


def _rate_for_mask(obs: np.ndarray, mask: np.ndarray) -> tuple[int,float]:
    m=np.asarray(mask,dtype=bool)&np.isfinite(obs)
    return int(m.sum()), (float(np.mean(obs[m])) if m.any() else np.nan)


def _annotate_system_lineage(g: pd.DataFrame, seasons: np.ndarray, finalists: list[dict[str,Any]], market: str) -> dict[str,Any]:
    """Attach parent/child lineage and incremental-value diagnostics.

    A parent must have the same market/direction and a strict subset of the child
    conditions.  The closest (largest-condition) parent is used.  This is diagnostic
    only: it cannot select a live rule or change confirmation/authority.
    """
    if not finalists:
        return {"status":"NO_SYSTEMS","pairs":[],"nested_total_variants":0,"selection_influence":0,"production_authority":0}
    y,valid,baseline=_market_target(g,market)
    pairs=[]
    by_id={z.get("system_id"):z for z in finalists}
    for child in finalists:
        cc=set(child.get("conditions") or [])
        candidates=[]
        for parent in finalists:
            if parent is child or parent.get("direction")!=child.get("direction"): continue
            pc=set(parent.get("conditions") or [])
            if pc and pc < cc:
                candidates.append(parent)
        if not candidates:
            child.update({"parent_system_id":None,"lineage_depth":0,"lineage_state":"ROOT","nested_total_variant":False,
                          "lineage_incremental":{"status":"NO_PARENT","selection_influence":0}})
            continue
        parent=max(candidates,key=lambda z:(len(z.get("conditions") or []),int(z.get("confirmation_n",0) or 0)))
        pm=np.asarray(parent.get("_mask_internal"),dtype=bool)&valid
        cm=np.asarray(child.get("_mask_internal"),dtype=bool)&valid
        po=pm&~cm
        direction=str(child.get("direction") or "PLAY_ON")
        obs=_directional_observation(y,direction)
        disc=np.isfinite(seasons)&(seasons<=DISCOVERY_MAX_SEASON)
        conf=np.isfinite(seasons)&np.isin(seasons,np.asarray(CONFIRMATION_SEASONS,dtype=float))
        dn,dr=_rate_for_mask(obs,cm&disc); dpn,dpr=_rate_for_mask(obs,po&disc)
        cn,cr=_rate_for_mask(obs,cm&conf); cpn,cpr=_rate_for_mask(obs,po&conf)
        added=sorted(cc-set(parent.get("conditions") or []))
        nested_total=any((str(x).upper().startswith("TOTAL_") or "TOTAL" in str(x).upper()) for x in added)
        inc={
            "status":"DIAGNOSTIC_ONLY","added_conditions":added,
            "discovery_child_n":dn,"discovery_child_rate":dr,"discovery_parent_only_n":dpn,"discovery_parent_only_rate":dpr,
            "discovery_rate_delta_vs_parent_only":(dr-dpr if np.isfinite(dr) and np.isfinite(dpr) else np.nan),
            "confirmation_child_n":cn,"confirmation_child_rate":cr,"confirmation_parent_only_n":cpn,"confirmation_parent_only_rate":cpr,
            "confirmation_rate_delta_vs_parent_only":(cr-cpr if np.isfinite(cr) and np.isfinite(cpr) else np.nan),
            "selection_influence":0,"production_authority":0,
        }
        if market=="h2h":
            child_px=_h2h_price_metrics(g,cm&valid,y,baseline,seasons,direction,CONFIRMATION_SEASONS)
            parent_only_px=_h2h_price_metrics(g,po&valid,y,baseline,seasons,direction,CONFIRMATION_SEASONS)
            inc.update({
                "confirmation_child_price_roi":child_px.get("roi"),
                "confirmation_parent_only_price_roi":parent_only_px.get("roi"),
                "confirmation_child_market_residual":child_px.get("market_residual"),
                "confirmation_parent_only_market_residual":parent_only_px.get("market_residual"),
            })
        child.update({
            "parent_system_id":parent.get("system_id"),
            "lineage_depth":int(parent.get("lineage_depth",0) or 0)+1,
            "lineage_state":"CHILD",
            "nested_total_variant":bool(nested_total),
            "lineage_incremental":inc,
        })
        pairs.append({"parent_system_id":parent.get("system_id"),"child_system_id":child.get("system_id"),
                      "market":market,"direction":direction,"nested_total_variant":bool(nested_total),**inc})
    # Resolve depth in a second pass so ordering cannot truncate ancestry.
    for z in finalists:
        depth=0; seen=set(); pid=z.get("parent_system_id")
        while pid and pid not in seen and pid in by_id:
            seen.add(pid); depth+=1; pid=by_id[pid].get("parent_system_id")
        z["lineage_depth"]=depth
    return {"status":"PASS","pairs":pairs,"pair_count":len(pairs),
            "nested_total_variants":sum(bool(x.get("nested_total_variant")) for x in pairs),
            "selection_influence":0,"production_authority":0}


def _evidence_tier(rep: dict[str,Any], market: str) -> str:
    """Research evidence labels only. Live authority still uses STRONG_VALIDATED gate."""
    if _miner_live_authority_eligible(rep): return "STRONG_VALIDATED"
    if bool(rep.get("confirmation_pass")):
        n=int(rep.get("confirmation_n",0) or 0)
        if market=="h2h":
            hp=rep.get("h2h_confirmation_price") or {}
            if n>=45 and int(hp.get("priced_n",0) or 0)>=30 and float(hp.get("roi",-9) or -9)>=0:
                return "PROMISING_SHADOW"
        else:
            r=float(rep.get("confirmation_rate",np.nan))
            if n>=60 and np.isfinite(r) and r>=.53: return "PROMISING_SHADOW"
        return "VALIDATED_SHADOW"
    if bool(rep.get("discovery_pass")): return "WATCHLIST"
    return "RESEARCH_SHADOW"


_PUBLISHED_NCAAF_DIRECTIONAL_SYSTEMS = [
    # Pathi directional football-side rules. Context-only key events are excluded.
    {"source":"PATHI","system":"Pathi_FB_Dog_Hook_Above_3","label":"Dog >3 to <4","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Hook_Above_7","label":"Dog >7 to <8","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Hook_Above_10","label":"Dog >10 to <11","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_10_Plus","label":"Dog 10+","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_0_to_3","label":"Dog 0 to 3","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_3_to_3_5","label":"Dog 3 to 3.5","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_3_5_to_6_5","label":"Dog 3.5 to 6.5","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_On_7","label":"Dog on 7","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Above_7","label":"Dog above 7","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Below_Key_3","label":"Dog below 3","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Below_Key_7","label":"Dog below 7","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Favorite_Below_Key_3","label":"Favorite below 3","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Favorite_Below_Key_7","label":"Favorite below 7","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Favorite_Below_Key_10","label":"Favorite below 10","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Favorite_Laying_Hook_3","label":"Favorite >3 to <4","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Favorite_Laying_Hook_7","label":"Favorite >7 to <8","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_TotalSpread_Gap_LE10","label":"Dog |total-spread| <= 10","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Moved_Below_Key_3","label":"Dog moved below 3","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Moved_Above_Key_3","label":"Dog moved above 3","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Moved_Below_Key_7","label":"Dog moved below 7","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Moved_Above_Key_7","label":"Dog moved above 7","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Moved_Below_Key_10","label":"Dog moved below 10","market":"spreads"},
    {"source":"PATHI","system":"Pathi_FB_Dog_Moved_Above_Key_10","label":"Dog moved above 10","market":"spreads"},
    # Big Al college-football flags are built on the recommended play-on side,
    # including CF3 (the named favorite is faded, the flag is on its opponent).
    {"source":"BIG_AL","system":"BigAl_CF1_Week2Home42Win","label":"CF1 Week 2 home off 42+ win","market":"spreads"},
    {"source":"BIG_AL","system":"BigAl_CF2_LateSeasonRevengeDog","label":"CF2 late-season revenge dog","market":"spreads"},
    {"source":"BIG_AL","system":"BigAl_CF2_Away_Tightener","label":"CF2 away tightener","market":"spreads","parent":"BigAl_CF2_LateSeasonRevengeDog"},
    {"source":"BIG_AL","system":"BigAl_CF3_Fade19PlusFavoriteUpsetLoss","label":"CF3 fade 19+ favorite after SU loss","market":"spreads"},
]


def _grade_published_ncaaf_systems(g: pd.DataFrame, seasons: np.ndarray) -> dict[str,Any]:
    """Historical W/L for directional Pathi/Big Al NCAAF flags already in the frame.

    Flags are graded as PLAY_ON because dashboard builders put directional flags on
    the recommended side. This table is descriptive/research-only and cannot create
    authority. Symmetric context flags and ambiguous screens are intentionally absent.
    """
    rows=[]
    for spec in _PUBLISHED_NCAAF_DIRECTIONAL_SYSTEMS:
        c=spec["system"]
        if c not in g.columns:
            rows.append({**spec,"direction":"PLAY_ON","status":"MISSING_FIELD","ready_n":0,"trigger_n":0,"production_authority":0})
            continue
        sig=pd.to_numeric(g[c],errors="coerce")
        ready_col=c+"_DataReady"
        ready=(pd.to_numeric(g[ready_col],errors="coerce").fillna(0).eq(1) if ready_col in g.columns else sig.notna())
        fire=ready&sig.fillna(0).eq(1)
        y,valid,baseline=_market_target(g,spec["market"])
        obs=np.asarray(y,dtype=float)
        graded=np.asarray(fire,dtype=bool)&valid
        def scope_stats(years):
            m=graded&np.isin(seasons,np.asarray(list(years),dtype=float))
            n=int(m.sum()); w=int(np.sum(obs[m]>0.5)) if n else 0; rate=float(w/n) if n else np.nan
            roi=(rate*(100/110.0)-(1-rate)) if n and spec["market"] in ("spreads","totals") else np.nan
            return {"n":n,"wins":w,"losses":n-w,"hit_rate":rate,"roi_at_minus110":roi,
                    "shrunk_hit_rate_beta15":_beta_shrunk_rate(w,n),"wilson95_low":_wilson95_low(w,n)}
        discovery=scope_stats([x for x in sorted(set(seasons[np.isfinite(seasons)].astype(int))) if x<=DISCOVERY_MAX_SEASON])
        confirmation=scope_stats(CONFIRMATION_SEASONS)
        year_rows=[]
        for sy in sorted(set(int(x) for x in seasons[graded&np.isfinite(seasons)])):
            st=scope_stats([sy]); year_rows.append({"season":sy,**st})
        cn=int(confirmation.get("n",0) or 0); cr=float(confirmation.get("hit_rate",np.nan))
        assessment=("SOURCE_DIRECTION_SUPPORT" if cn>=50 and np.isfinite(cr) and cr>=.55 else
                    "SOURCE_DIRECTION_WEAK" if cn>=50 and np.isfinite(cr) and cr<=.45 else "MIXED_OR_SMALL_SAMPLE")
        rows.append({**spec,"direction":"PLAY_ON","status":"GRADED" if int(graded.sum()) else "NO_GRADED_TRIGGERS",
                     "ready_n":int(ready.sum()),"trigger_n":int(fire.sum()),"graded_n":int(graded.sum()),
                     "discovery":discovery,"confirmation":confirmation,"by_season":year_rows,
                     "source_direction_assessment":assessment,
                     "confirmation_inverse_hit_rate":(1.0-cr if cn and np.isfinite(cr) else np.nan),
                     "selection_influence":0,"production_authority":0})
    available=[x for x in rows if x.get("status")!="MISSING_FIELD"]
    return {"status":"PASS","systems":rows,"available_systems":len(available),"graded_systems":sum(int(x.get("graded_n",0) or 0)>0 for x in rows),
            "contract":"DIRECTIONAL_NCAAF_FLAGS_ONLY__RECOMMENDED_SIDE_WL__CONTEXT_FLAGS_EXCLUDED__ZERO_AUTHORITY",
            "selection_influence":0,"production_authority":0}

def _v2181_is_expanded_pt_atom(atom: dict[str,Any]) -> bool:
    """V2.18-added all-index PT atoms; fixed-five META_PT atoms remain legacy."""
    if str(atom.get("family") or "") != "EXTERNAL_RATINGS_FAMILY":
        return False
    name=str(atom.get("name") or "")
    return name.startswith(("PTIDX_","PT_ALL_","PT_CLUSTER_","PT_TRACKER_","PT_EXMETA_"))


def _v2182_is_meta_constituent_index_atom_name(name: Any) -> bool:
    n=str(name or "").upper()
    return any(n.startswith(f"PTIDX_{k.upper()}_") for k in PT_META_CONSTITUENT_INDEX_IDS)


def _v2182_meta_overlap_violation(names) -> bool:
    """Future-proof guard: frozen META and one of its own constituent index atoms
    may never coexist in a single Miner rule, even if family labels change later.
    """
    ns=[str(x or "") for x in (names or [])]
    has_meta=any(n.startswith("META_PT_") for n in ns)
    has_constituent=any(_v2182_is_meta_constituent_index_atom_name(n) for n in ns)
    return bool(has_meta and has_constituent)


def _v2181_external_predictor_behavior(g: pd.DataFrame, seasons: np.ndarray, market: str) -> dict[str,Any]:
    """Learn each external predictor's behavior without allowing it to select on 2026.

    Discovery (<=2023) chooses FOLLOW vs FADE for each predictor when |edge|>=2.
    2024/2025 are strictly confirmation diagnostics of that frozen behavior. 2026+
    is never consulted here. This is research metadata only; authority remains zero.
    """
    market=str(market).lower()
    if market!="spreads":
        return {"status":"NOT_APPLICABLE","market":market,"predictors":[],"selection_influence":0,"production_authority":0}
    y,valid,_=_market_target(g,market)
    sy=np.asarray(seasons,dtype=float)
    disc_scope=np.isfinite(sy)&(sy<=DISCOVERY_MAX_SEASON)
    conf_scope=np.isfinite(sy)&np.isin(sy,np.asarray(CONFIRMATION_SEASONS,dtype=float))
    open_spread=_num(g,"Consensus_Open_Spread").to_numpy(dtype=float)
    actual_margin=_num(g,"Actual_Margin").to_numpy(dtype=float)
    edge_cols=sorted([
        c for c in g.columns
        if str(c).startswith("_V218_PTIDX_") and str(c).endswith("_EDGE_POINTS")
        and str(c) not in {
            "_V218_PTIDX_MEAN_EDGE_POINTS","_V218_PTIDX_MEDIAN_EDGE_POINTS","_V218_PTIDX_CLUSTER_MEDIAN_EDGE_POINTS",
        }
    ])

    def scoped(mask, follow_obs, preferred, edge_abs, pred_margin, rec_spread):
        m=np.asarray(mask,dtype=bool)&np.isfinite(follow_obs)
        n=int(m.sum())
        follow=float(np.mean(follow_obs[m])) if n else np.nan
        pref_obs=follow_obs if preferred=="FOLLOW" else 1.0-follow_obs
        pref=float(np.mean(pref_obs[m])) if n else np.nan
        out={
            "n":n,"follow_rate":follow,"preferred_behavior":preferred,"preferred_rate":pref,
            "mean_abs_edge":float(np.nanmean(edge_abs[m])) if n and np.isfinite(edge_abs[m]).any() else np.nan,
        }
        if pred_margin is not None:
            pm=m&np.isfinite(pred_margin)&np.isfinite(actual_margin)
            out["projection_n"]=int(pm.sum())
            out["projection_mae"]=float(np.mean(np.abs(pred_margin[pm]-actual_margin[pm]))) if pm.any() else np.nan
            out["projection_bias_pred_minus_actual"]=float(np.mean(pred_margin[pm]-actual_margin[pm])) if pm.any() else np.nan
        else:
            out.update({"projection_n":0,"projection_mae":np.nan,"projection_bias_pred_minus_actual":np.nan})
        # Learn whether the frozen predictor behavior differs on recommended favorites/dogs.
        role_rows=[]
        for label,rm in (
            ("RECOMMENDED_FAVORITE",m&np.isfinite(rec_spread)&(rec_spread<0)),
            ("RECOMMENDED_DOG",m&np.isfinite(rec_spread)&(rec_spread>0)),
        ):
            rn=int(rm.sum())
            role_rows.append({"role":label,"n":rn,"preferred_rate":float(np.mean(pref_obs[rm])) if rn else np.nan})
        out["recommended_role"]=role_rows
        edge_rows=[]
        for label,lo,hi in (("2_TO_LT3",2,3),("3_TO_LT5",3,5),("5_PLUS",5,np.inf)):
            em=m&np.isfinite(edge_abs)&(edge_abs>=lo)&(edge_abs<hi)
            en=int(em.sum())
            edge_rows.append({"edge_band":label,"n":en,"preferred_rate":float(np.mean(pref_obs[em])) if en else np.nan})
        out["edge_bands"]=edge_rows
        return out

    rows=[]
    for ec in edge_cols:
        slug=str(ec)[len("_V218_PTIDX_"):-len("_EDGE_POINTS")]
        mc=f"_V218_PTIDX_{slug}_MARGIN_TEAM"
        edge=pd.to_numeric(g[ec],errors="coerce").to_numpy(dtype=float)
        pred_margin=pd.to_numeric(g[mc],errors="coerce").to_numpy(dtype=float) if mc in g.columns else None
        active=valid&np.isfinite(edge)&(np.abs(edge)>=2)
        follow_obs=np.full(len(g),np.nan,dtype=float)
        pos=active&(edge>=2); neg=active&(edge<=-2)
        follow_obs[pos]=y[pos]; follow_obs[neg]=1.0-y[neg]
        edge_abs=np.abs(edge)
        rec_spread=np.where(edge>=2,open_spread,np.where(edge<=-2,-open_spread,np.nan))
        dm=active&disc_scope&np.isfinite(follow_obs)
        dn=int(dm.sum()); d_follow=float(np.mean(follow_obs[dm])) if dn else np.nan
        preferred="FOLLOW" if (np.isfinite(d_follow) and d_follow>=.5) else "FADE"
        discovery=scoped(dm,follow_obs,preferred,edge_abs,pred_margin,rec_spread)
        confirmation=scoped(active&conf_scope,follow_obs,preferred,edge_abs,pred_margin,rec_spread)
        by_year=[]
        pref_obs=follow_obs if preferred=="FOLLOW" else 1.0-follow_obs
        for yr in CONFIRMATION_SEASONS:
            ym=active&valid&(sy==yr)&np.isfinite(pref_obs)
            by_year.append({"season":int(yr),"n":int(ym.sum()),"preferred_rate":float(np.mean(pref_obs[ym])) if ym.any() else np.nan})
        confirmation["by_year"]=by_year
        both=bool(len([x for x in by_year if x["n"]>=10])==2 and all(x["preferred_rate"]>=.5 for x in by_year if x["n"]>=10))
        confirmed=bool(discovery["n"]>=100 and confirmation["n"]>=30 and both and np.isfinite(confirmation["preferred_rate"]) and confirmation["preferred_rate"]>=.50)
        strong=bool(confirmed and confirmation["n"]>=60 and confirmation["preferred_rate"]>=.54)
        _canon_slug=str(slug).lower()
        rows.append({
            "predictor":slug,"edge_column":ec,"margin_column":mc if mc in g.columns else None,
            "source_cluster":_pt_index_source_cluster(_canon_slug),
            "meta_constituent":bool(_canon_slug in PT_META_CONSTITUENT_INDEX_IDS),
            "behavior_learned_from":"DISCOVERY_2022_2023_ONLY","preferred_behavior":preferred,
            "discovery":discovery,"confirmation":confirmation,
            "behavior_state":"STRONG_CONFIRMED" if strong else "CONFIRMED" if confirmed else "DISCOVERY_ONLY",
            "2026_selection_influence":0,"selection_influence":0,"production_authority":0,
        })
    ranked=sorted(rows,key=lambda z:(
        z.get("behavior_state")=="STRONG_CONFIRMED",
        z.get("behavior_state")=="CONFIRMED",
        float((z.get("confirmation") or {}).get("preferred_rate") or -1),
        int((z.get("confirmation") or {}).get("n") or 0),
    ),reverse=True)
    return {
        "status":"PASS","market":market,"predictor_count":len(rows),"predictors":rows,
        "strong_confirmed_count":sum(x["behavior_state"]=="STRONG_CONFIRMED" for x in rows),
        "confirmed_count":sum(x["behavior_state"] in {"STRONG_CONFIRMED","CONFIRMED"} for x in rows),
        "top_confirmed":[{
            "predictor":x["predictor"],"behavior":x["preferred_behavior"],"state":x["behavior_state"],
            "meta_constituent":bool(x.get("meta_constituent",False)),
            "source_cluster":x.get("source_cluster"),
            "discovery_n":x["discovery"]["n"],"discovery_rate":x["discovery"]["preferred_rate"],
            "confirmation_n":x["confirmation"]["n"],"confirmation_rate":x["confirmation"]["preferred_rate"],
        } for x in ranked[:12]],
        "contract":"DISCOVERY_LEARNS_FOLLOW_OR_FADE_AND_EDGE_ROLE_BEHAVIOR__META_CONSTITUENTS_TAGGED__2024_2025_CONFIRM__2026_PROSPECTIVE_ONLY__ZERO_AUTHORITY",
        "selection_influence":0,"production_authority":0,
    }


def run_system_miner_v3(games: pd.DataFrame, seasons: np.ndarray, market: str, dashboard_module=None,
                        log_func=print, max_depth: int=4) -> dict[str,Any]:
    market=str(market).lower(); y,valid,baseline=_market_target(games,market); atoms=_extended_atoms(games,dashboard_module,market=market)
    _fam_counts={}
    for _a in atoms: _fam_counts[_a.get("family")]=int(_fam_counts.get(_a.get("family"),0))+1
    _bridge_atoms=sum(v for k,v in _fam_counts.items() if str(k).startswith(("EXPERT_","RESEARCH_CORE_STATE","RESEARCH_SPECIALIST_","MARKET_KEY_","EXTERNAL_")))
    _obs_atoms=[a for a in atoms if str(a.get("family") or "") in BIG_AL_OBSERVATION_ATOM_FAMILIES]
    behavior=_v2181_external_predictor_behavior(games,seasons,market)
    out={"version":"NCAAF-RV2.18.3-SYSTEM-MINER-V9-CURRENT-EXTERNAL-CONSENSUS","market":market,"production_authority":0,"discovery_max_season":DISCOVERY_MAX_SEASON,
         "confirmation_seasons":list(CONFIRMATION_SEASONS),"prospective_min_season":PROSPECTIVE_MIN_SEASON,"atoms":len(atoms),"atom_family_counts":_fam_counts,"expert_model_bridge_atoms":int(_bridge_atoms),"systems":[],"mechanism_families":[],"external_predictor_behavior":behavior}
    log_func(f"[NCAAF-RV25-ATOM-BRIDGE] market={market} atoms={len(atoms)} bridge_atoms={_bridge_atoms} pathi={_fam_counts.get('EXPERT_PATHI',0)} bigal={_fam_counts.get('EXPERT_BIGAL',0)} core={_fam_counts.get('RESEARCH_CORE_STATE',0)} specialist={sum(v for k,v in _fam_counts.items() if str(k).startswith('RESEARCH_SPECIALIST_'))} external={sum(v for k,v in _fam_counts.items() if str(k).startswith('EXTERNAL_'))} authority=0")
    if market in {"spreads","totals"}:
        log_func(f"[NCAAF-RV229-OBS-ATOM-CATALOG] market={market} atoms={len(_obs_atoms)} names={','.join(sorted(a['name'] for a in _obs_atoms))} rating_weight=0 outcomes_2026_used=FALSE authority=0")
    _special_atom_availability={}
    if market=="spreads":
        _special_atom_availability=_special_atom_availability_audit(games,seasons,market,dashboard_module=dashboard_module,admitted_atoms=atoms)
        for _fam,_diag in _special_atom_availability.items():
            log_func(f"[NCAAF-SPECIAL-ATOM-AUDIT] market=spreads category={_fam} expected={_diag.get('expected_atoms')} source_available={_diag.get('source_available_atoms')} admitted={_diag.get('admitted_atoms')} evaluator_eligible={_diag.get('evaluator_eligible_atoms')} below_discovery_min={_diag.get('below_evaluator_min_atoms')} source_unavailable={_diag.get('source_unavailable_atoms')} evaluator_min_discovery_n=100 authority=0")
            for _ar in (_diag.get('atoms') or []):
                log_func(f"[NCAAF-SPECIAL-ATOM] category={_fam} atom={_ar.get('atom')} source_available={str(bool(_ar.get('source_available'))).upper()} admitted={str(bool(_ar.get('admitted_to_catalog'))).upper()} total_n={_ar.get('total_n')} discovery_n={_ar.get('discovery_n')} y2024_n={_ar.get('confirmation_2024_n')} y2025_n={_ar.get('confirmation_2025_n')} evaluator_eligible={str(bool(_ar.get('evaluator_eligible'))).upper()} diag_direction={_ar.get('diagnostic_direction')} diag_discovery={_ar.get('diagnostic_discovery_rate')}/{_ar.get('diagnostic_discovery_graded_n')} diag_2024={_ar.get('diagnostic_2024_rate')}/{_ar.get('diagnostic_2024_graded_n')} diag_2025={_ar.get('diagnostic_2025_rate')}/{_ar.get('diagnostic_2025_graded_n')} diag_confirmation={_ar.get('diagnostic_confirmation_rate')}/{_ar.get('diagnostic_confirmation_graded_n')} diagnostic_only=TRUE authority=0")
    if market=="spreads":
        log_func(f"[NCAAF-RV2182-PREDICTOR-BEHAVIOR] predictors={behavior.get('predictor_count',0)} confirmed={behavior.get('confirmed_count',0)} strong_confirmed={behavior.get('strong_confirmed_count',0)} discovery=2022_2023 confirmation=2024_2025 year_2026_selection=FALSE authority=0")
        for _pb in (behavior.get("top_confirmed") or [])[:8]:
            log_func(f"[NCAAF-RV2182-PREDICTOR-TOP] predictor={_pb.get('predictor')} behavior={_pb.get('behavior')} state={_pb.get('state')} discovery={_pb.get('discovery_rate')}/{_pb.get('discovery_n')} confirmation={_pb.get('confirmation_rate')}/{_pb.get('confirmation_n')} meta_constituent={str(bool(_pb.get('meta_constituent',False))).upper()} source_cluster={_pb.get('source_cluster')} authority=0")
    if valid.sum()<500: out["status"]="INSUFFICIENT_HISTORY"; return out

    expanded_external=[a for a in atoms if _v2181_is_expanded_pt_atom(a)] if market=="spreads" else []
    legacy_atoms=[a for a in atoms if not _v2181_is_expanded_pt_atom(a)]
    if market=="spreads":
        log_func(f"[NCAAF-RV2182-MINER-LANES] market=spreads total_atoms={len(atoms)} legacy_atoms={len(legacy_atoms)} expanded_external_atoms={len(expanded_external)} legacy_beam=64 external_seed_cap=72 external_support_cap=80 external_depth_cap=3 fdr_scope=SEPARATE authority=0")
        log_func(f"[NCAAF-RV2182-META-OVERLAP-GUARD] status=PASS meta_constituents={','.join(sorted(PT_META_CONSTITUENT_INDEX_IDS))} meta_source_clusters={','.join(sorted({_pt_index_source_cluster(k) for k in PT_META_CONSTITUENT_INDEX_IDS}))} rule_same_mechanism=BLOCK authority_family_cap=1 conflict_policy=ABSTAIN exmeta_consensus=SOURCE_CLUSTER_OUT authority=0")

    def ev(atom_mask,names,fams,idx):
        # Explicit guard in addition to same-family search protection.
        if _v2182_meta_overlap_violation(names):
            return None
        r=_evaluate_rule(games,atom_mask,y,valid,baseline,seasons,names,fams,market)
        if r is None: return None
        if market=="h2h":
            hp=r.get("h2h_discovery_price") or {}
            discovery_pass=bool(
                r["stable_discovery_fraction"]>=1.0 and np.isfinite(r["market_residual_discovery"]) and r["market_residual_discovery"]>=.01 and
                int(hp.get("priced_n",0) or 0)>=80 and np.isfinite(hp.get("roi",np.nan)) and hp["roi"]>.01 and
                bool(hp.get("price_band_robust",False)) and np.isfinite(r["remove_best_discovery_roi"]) and r["remove_best_discovery_roi"]>=0 and
                np.isfinite(r["remove_best_market_residual"]) and r["remove_best_market_residual"]>=0
            )
            r["quality"]=4*max(0,r["market_residual_discovery"])+2*max(0,hp.get("roi",0))+0.05*float(hp.get("positive_price_band_fraction",0) or 0)
        else:
            discovery_pass=bool(r["discovery_rate"]>=.54 and r["stable_discovery_fraction"]>=1.0 and np.isfinite(r["remove_best_discovery_rate"]) and r["remove_best_discovery_rate"]>=.515)
            r["quality"]=(r["discovery_rate"]-.5)*3 + max(0,r["remove_best_discovery_rate"]-.5) + .03*r["stable_discovery_fraction"]
        r["discovery_pass"]=discovery_pass; r["idx"]=tuple(idx)
        return r

    def legacy_search(lane_atoms):
        tested=[]; beam=[]; seen=set(); singleton={}
        for i,a in enumerate(lane_atoms):
            r=ev(a["mask"],(a["name"],),(a["family"],),(i,))
            if r:
                tested.append(r); beam.append(r); singleton[a["name"]]=r
        beam=sorted(beam,key=lambda z:z["quality"],reverse=True)[:64]
        for depth in range(2,max_depth+1):
            nxt=[]
            for st in beam:
                used=set(st["families"])
                for i,a in enumerate(lane_atoms):
                    if i in st["idx"] or a["family"] in used: continue
                    names=tuple(sorted(st["conditions"]+[a["name"]]))
                    if names in seen: continue
                    seen.add(names); r=ev(st["mask"]&a["mask"],names,tuple(st["families"]+[a["family"]]),st["idx"]+(i,))
                    if r: tested.append(r); nxt.append(r)
            beam=sorted(nxt,key=lambda z:z["quality"],reverse=True)[:64]
            if not beam: break
        return tested,singleton

    legacy_tested,legacy_single=legacy_search(legacy_atoms)

    # V2.26 constrained expert-observation hypothesis lane. The user-observed
    # card only determines which GENERIC concepts deserve a focused search.
    # No 2026 outcome, Big Al rating, team identity, or current pick result enters
    # discovery/confirmation. The same historical evaluator/FDR/frozen seasons
    # apply, and any surviving rule is still a source-neutral MINER mechanism.
    observation_tested=[]; observation_seed_atoms=[]; observation_support_atoms=[]
    if market in {"spreads","totals"}:
        _legacy_pos={str(a.get("name")):i for i,a in enumerate(legacy_atoms)}
        observation_seed_atoms=[a for a in legacy_atoms if str(a.get("family") or "") in BIG_AL_OBSERVATION_ATOM_FAMILIES]
        observation_support_atoms=[a for a in legacy_atoms if str(a.get("family") or "") in (BIG_AL_OBSERVATION_ATOM_FAMILIES|BIG_AL_OBSERVATION_SUPPORT_FAMILIES)]
        _obs_seen=set(); _obs_beam=[]
        for a in observation_seed_atoms:
            _i=_legacy_pos.get(str(a.get("name")),0)
            r=ev(a["mask"],(a["name"],),(a["family"],),(_i,))
            if r:
                observation_tested.append(r); _obs_beam.append(r); _obs_seen.add(tuple(sorted(r["conditions"])))
        _obs_beam=sorted(_obs_beam,key=lambda z:z["quality"],reverse=True)[:48]
        for _depth in range(2,min(max_depth,4)+1):
            _nxt=[]
            for st in _obs_beam:
                _used=set(st["families"]); _names=set(st["conditions"])
                for a in observation_support_atoms:
                    if a["family"] in _used or a["name"] in _names: continue
                    names=tuple(sorted(st["conditions"]+[a["name"]]))
                    if names in _obs_seen: continue
                    _obs_seen.add(names)
                    _i=_legacy_pos.get(str(a.get("name")),0)
                    r=ev(st["mask"]&a["mask"],names,tuple(st["families"]+[a["family"]]),st["idx"]+(_i,))
                    if r:
                        # Seed construction guarantees every rule contains at least
                        # one observation-vocabulary family.
                        observation_tested.append(r); _nxt.append(r)
            _obs_beam=sorted(_nxt,key=lambda z:z["quality"],reverse=True)[:48]
            if not _obs_beam: break

    external_tested=[]; selected_external=[]; selected_support=[]
    if market=="spreads" and expanded_external:
        external_single=[]
        for i,a in enumerate(expanded_external):
            r=ev(a["mask"],(a["name"],),(a["family"],),(i,))
            if r:
                external_tested.append(r); external_single.append((a,r,i))
        # Always preserve aggregate consensus/dispersion atoms, then fill the bounded
        # external seed set from discovery-only quality. No confirmation data selects seeds.
        agg=[]; indiv=[]
        for item in external_single:
            nm=str(item[0].get("name") or "")
            (agg if nm.startswith(("PT_EXMETA_","PT_ALL_","PT_CLUSTER_","PT_TRACKER_")) else indiv).append(item)
        key=lambda item:(bool(item[1].get("discovery_pass")),float(item[1].get("quality",-9)),int(item[1].get("discovery_n",0)))
        agg=sorted(agg,key=key,reverse=True)
        indiv=sorted(indiv,key=key,reverse=True)
        selected_external=(agg+indiv)[:72]

        # External signals may interact with CORE/Pathi/Big Al/specialists and a
        # bounded number of broader legacy contexts. They never combine with a second
        # external rating family atom.
        support_candidates=[]
        for a in legacy_atoms:
            if str(a.get("family") or "")=="EXTERNAL_RATINGS_FAMILY": continue
            r=legacy_single.get(a.get("name"))
            if r is not None: support_candidates.append((a,r))
        def support_key(item):
            fam=str(item[0].get("family") or "")
            priority=fam.startswith(("EXPERT_PATHI","EXPERT_BIGAL","RESEARCH_CORE_STATE","RESEARCH_SPECIALIST_"))
            return (priority,bool(item[1].get("discovery_pass")),float(item[1].get("quality",-9)),int(item[1].get("discovery_n",0)))
        priority=[x for x in support_candidates if str(x[0].get("family") or "").startswith(("EXPERT_PATHI","EXPERT_BIGAL","RESEARCH_CORE_STATE","RESEARCH_SPECIALIST_"))]
        general=[x for x in support_candidates if not str(x[0].get("family") or "").startswith(("EXPERT_PATHI","EXPERT_BIGAL","RESEARCH_CORE_STATE","RESEARCH_SPECIALIST_"))]
        priority=sorted(priority,key=support_key,reverse=True)
        general=sorted(general,key=support_key,reverse=True)
        selected_support=(priority+general)[:80]

        pair_beam=[]; seen=set()
        for ea,er,ei in selected_external:
            for sa,sr in selected_support:
                if sa["family"]==ea["family"]: continue
                names=tuple(sorted((ea["name"],sa["name"])))
                if names in seen: continue
                seen.add(names)
                r=ev(ea["mask"]&sa["mask"],names,(ea["family"],sa["family"]),(ei,))
                if r:
                    external_tested.append(r); pair_beam.append(r)
        pair_beam=sorted(pair_beam,key=lambda z:z["quality"],reverse=True)[:64]
        # One more support layer is enough to learn conditional predictor behavior
        # without recreating a giant feature factory.
        depth_beam=pair_beam
        for depth in range(3,min(max_depth,3)+1):
            nxt=[]
            for st in depth_beam:
                used=set(st["families"])
                for sa,sr in selected_support:
                    if sa["family"] in used: continue
                    names=tuple(sorted(st["conditions"]+[sa["name"]]))
                    if names in seen: continue
                    seen.add(names)
                    r=ev(st["mask"]&sa["mask"],names,tuple(st["families"]+[sa["family"]]),st["idx"])
                    if r:
                        external_tested.append(r); nxt.append(r)
            depth_beam=sorted(nxt,key=lambda z:z["quality"],reverse=True)[:64]
            if not depth_beam: break

    def finalize_lane(tested,lane,q_cut,max_finalists):
        if not tested: return []
        q=_bh_qvalues([z["nominal_pvalue"] for z in tested])
        for z,qq in zip(tested,q): z["fdr_qvalue"]=float(qq) if np.isfinite(qq) else np.nan
        finalists=[]
        for z in sorted(tested,key=lambda r:(r["discovery_pass"],r["quality"],r["discovery_n"]),reverse=True):
            if not z["discovery_pass"] or not np.isfinite(z["fdr_qvalue"]) or z["fdr_qvalue"]>q_cut: continue
            if market=="h2h":
                hp=z.get("h2h_confirmation_price") or {}; cy=[x for x in z.get("confirmation",[]) if int(x.get("season",0)) in CONFIRMATION_SEASONS]
                both_years=bool(len(cy)==2 and all(int(x.get("priced_n",0) or 0)>=15 and np.isfinite(x.get("roi",np.nan)) and x["roi"]>0 and np.isfinite(x.get("market_residual",np.nan)) and x["market_residual"]>=0 for x in cy))
                conf_ok=bool(
                    (z.get("split_audit") or {}).get("disjoint",False) and
                    z["confirmation_n"]>=30 and both_years and int(hp.get("priced_n",0) or 0)>=30 and
                    np.isfinite(hp.get("roi",np.nan)) and hp["roi"]>0 and bool(hp.get("price_band_robust",False)) and
                    np.isfinite(z["market_residual_confirmation"]) and z["market_residual_confirmation"]>=0 and z["fdr_qvalue"]<=.10
                )
            else:
                conf_ok=bool((z.get("split_audit") or {}).get("disjoint",False) and
                             z["confirmation_n"]>=30 and np.isfinite(z["confirmation_rate"]) and z["confirmation_rate"]>=.50 and
                             len([x for x in z["confirmation"] if x["n"]>=10])==2 and all(x["rate"]>=.50 for x in z["confirmation"] if x["n"]>=10))
            item={k:v for k,v in z.items() if k not in {"mask","idx","quality"}}
            item.update({"system_id":_stable_id("NCAAF-RV2182-"+market.upper()+"-",z["conditions"]),
                         "miner_lane":lane,
                         "authority_state":"CONFIRMED_SHADOW" if conf_ok else "DISCOVERY_FROZEN",
                         "confirmation_pass":conf_ok,"production_authority":0,
                         "admission_note":"2024 AND 2025 must confirm frozen discovery; external lane has separate FDR and one correlated external family; H2H requires observed moneyline ROI/EV and price-band robustness"})
            item["_mask_internal"]=z["mask"]
            finalists.append(item)
            if len(finalists)>=max_finalists: break
        return finalists

    legacy_finalists=finalize_lane(legacy_tested,"LEGACY",.20,160)
    observation_finalists=finalize_lane(observation_tested,"EXPERT_OBSERVATION_HYPOTHESIS",.10,80) if market in {"spreads","totals"} else []
    external_finalists=finalize_lane(external_tested,"EXTERNAL_PREDICTOR_BEHAVIOR",.10,120) if market=="spreads" else []

    # Deduplicate exact rules discovered in more than one lane. A focused lane
    # can rescue a rule the broad beam misses, but it cannot manufacture a second
    # vote for the same rule. Lane provenance is retained for audit.
    _dedup={}
    for _z in legacy_finalists+observation_finalists+external_finalists:
        _key=(str(_z.get("direction")),tuple(sorted(_z.get("conditions") or [])))
        if _key not in _dedup:
            _z=dict(_z); _z["discovery_lanes"]=[str(_z.get("miner_lane") or "")]; _dedup[_key]=_z
        else:
            _old=_dedup[_key]; _lanes=set(_old.get("discovery_lanes") or [str(_old.get("miner_lane") or "")]); _lanes.add(str(_z.get("miner_lane") or "")); _old["discovery_lanes"]=sorted(x for x in _lanes if x)
            if str(_old.get("miner_lane"))=="EXPERT_OBSERVATION_HYPOTHESIS" and str(_z.get("miner_lane"))=="LEGACY":
                _z=dict(_z); _z["discovery_lanes"]=_old["discovery_lanes"]; _dedup[_key]=_z
    finalists=list(_dedup.values())
    out["lanes"]={
        "legacy":{"atoms":len(legacy_atoms),"tested_hypotheses":len(legacy_tested),"systems":len(legacy_finalists),"fdr_q_cut":.20},
        "expert_observation_hypothesis":{"seed_atoms":len(observation_seed_atoms),"support_atoms":len(observation_support_atoms),"tested_hypotheses":len(observation_tested),"systems":len(observation_finalists),"deduplicated_into_total":sum(1 for z in finalists if "EXPERT_OBSERVATION_HYPOTHESIS" in (z.get("discovery_lanes") or [])),"fdr_q_cut":.10,"depth_cap":4,"ratings_used_for_selection":False,"2026_outcomes_used":False,"source_after_validation":"MINER"},
        "external_predictor_behavior":{"atoms":len(expanded_external),"selected_seed_atoms":len(selected_external),"selected_support_atoms":len(selected_support),"tested_hypotheses":len(external_tested),"systems":len(external_finalists),"fdr_q_cut":.10,"depth_cap":3,"selection_uses_2026":False},
    }
    log_func(f"[NCAAF-RV2182-LANE] market={market} lane=LEGACY atoms={len(legacy_atoms)} tested={len(legacy_tested)} systems={len(legacy_finalists)} fdr_q=.20 authority=0")
    if market in {"spreads","totals"}:
        log_func(f"[NCAAF-BIGAL-HYPOTHESIS-LANE] market={market} seed_atoms={len(observation_seed_atoms)} support_atoms={len(observation_support_atoms)} tested={len(observation_tested)} systems={len(observation_finalists)} dedup_total={out['lanes']['expert_observation_hypothesis']['deduplicated_into_total']} fdr_q=.10 ratings_used=FALSE outcomes_2026_used=FALSE source=MINER authority=0")
    if market=="spreads":
        log_func(f"[NCAAF-RV2182-LANE] market=spreads lane=EXTERNAL_PREDICTOR_BEHAVIOR atoms={len(expanded_external)} seeds={len(selected_external)} supports={len(selected_support)} tested={len(external_tested)} systems={len(external_finalists)} fdr_q=.10 depth_cap=3 year_2026_selection=FALSE authority=0")

    if not finalists:
        _tested_total=len(legacy_tested)+len(observation_tested)+len(external_tested)
        out.update({"status":"NO_CANDIDATES","tested_hypotheses":_tested_total,"published_systems":0});
        log_func(f"[NCAAF-RV23-MINER] market={market} atoms={len(atoms)} tested={_tested_total} systems=0 mechanisms=0 confirmed_mechanisms=0 authority=0")
        return out

    lineage=_annotate_system_lineage(games,seasons,finalists,market)
    families=[]; used=set()
    for i,a in enumerate(finalists):
        if i in used: continue
        group=[i]; used.add(i); ma=np.asarray(a["_mask_internal"],dtype=bool)
        for j in range(i+1,len(finalists)):
            if j in used or finalists[j]["direction"]!=a["direction"]: continue
            mb=np.asarray(finalists[j]["_mask_internal"],dtype=bool); union=(ma|mb).sum()
            jac=float((ma&mb).sum()/union) if union else 0.0
            same_fams=set(finalists[j]["families"])==set(a["families"])
            if jac>=.80 or (same_fams and jac>=.65): group.append(j); used.add(j)
        members=[finalists[k] for k in group]
        if market=="h2h":
            def _hkey(z):
                cr=(z.get("h2h_confirmation_price") or {}).get("roi",np.nan); dr=(z.get("h2h_discovery_price") or {}).get("roi",np.nan); mr=z.get("market_residual_confirmation",np.nan)
                cr=float(cr) if cr is not None and np.isfinite(float(cr)) else -9.0
                dr=float(dr) if dr is not None and np.isfinite(float(dr)) else -9.0
                mr=float(mr) if mr is not None and np.isfinite(float(mr)) else -9.0
                return (z["confirmation_pass"],cr,mr,dr)
            rep=max(members,key=_hkey)
        else:
            rep=max(members,key=lambda z:(z["confirmation_pass"],z["confirmation_rate"] if np.isfinite(z["confirmation_rate"]) else -1,z["discovery_rate"],z["discovery_n"]))
        fid=_stable_id("NCAAF-MECH-",[market,rep["direction"],"+".join(sorted(set(rep["families"])))]+[x["system_id"] for x in members])
        _family_set=set(rep.get("families") or [])
        _uses_external=bool("EXTERNAL_RATINGS_FAMILY" in _family_set)
        _uses_season_record=bool("SEASON_RECORD_STATE" in _family_set)
        _uses_bounceback=bool("ROLE_TRANSITION_BOUNCEBACK" in _family_set)
        _uses_h2h=bool("MATCHUP_HISTORY" in _family_set)
        _uses_rivalry=bool("RIVALRY" in _family_set)
        _uses_travel=bool(_family_set & {"TRAVEL_CONTEXT","ROAD_SEQUENCE"})
        _uses_result_quality=bool("RESULT_QUALITY_REGRESSION" in _family_set)
        _uses_resume=bool("SCHEDULE_RESUME_QUALITY" in _family_set)
        _uses_recent=bool("RECENT_VS_SEASON" in _family_set)
        _uses_matchup=bool("MATCHUP_DIFFERENTIAL" in _family_set)
        _uses_observation_hypothesis=bool(_family_set & BIG_AL_OBSERVATION_ATOM_FAMILIES) or any(bool(set(m.get("families") or []) & BIG_AL_OBSERVATION_ATOM_FAMILIES) or "EXPERT_OBSERVATION_HYPOTHESIS" in set(m.get("discovery_lanes") or [m.get("miner_lane")]) for m in members)
        _rep_conditions=list(rep.get("conditions") or [])
        fam={"mechanism_id":fid,"market":market,"direction":rep["direction"],"representative_system_id":rep["system_id"],
             "representative_conditions":_rep_conditions,"member_system_ids":[x["system_id"] for x in members],"member_count":len(members),"families":rep["families"],
             "uses_external_ratings_family":_uses_external,
             "uses_season_record_state_family":_uses_season_record,
             "uses_role_transition_bounceback_family":_uses_bounceback,
             "uses_h2h_context_family":_uses_h2h,
             "uses_rivalry_context_family":_uses_rivalry,
             "uses_travel_context_family":_uses_travel,
             "uses_result_quality_family":_uses_result_quality,
             "uses_schedule_resume_family":_uses_resume,
             "uses_recent_vs_season_family":_uses_recent,
             "uses_matchup_differential_family":_uses_matchup,
             "uses_big_al_observation_hypothesis":_uses_observation_hypothesis,
             "hypothesis_origin":"BIG_AL_OBSERVATION_VOCABULARY" if _uses_observation_hypothesis else None,
             # Observation-inspired rules stay source-neutral. They do NOT collapse
             # to one Big Al vote; independent historical mechanisms retain their
             # own evidence key unless an existing correlation cap (PT/state/bounce) applies.
             "evidence_family_key":"EXTERNAL_RATINGS_FAMILY" if _uses_external else "NCAAF_SEASON_RECORD_STATE_FAMILY" if _uses_season_record else "NCAAF_ROLE_TRANSITION_BOUNCEBACK_FAMILY" if _uses_bounceback else "NCAAF_H2H_CONTEXT_FAMILY" if _uses_h2h else "NCAAF_RIVALRY_CONTEXT_FAMILY" if _uses_rivalry else "NCAAF_TRAVEL_CONTEXT_FAMILY" if _uses_travel else "NCAAF_RESULT_QUALITY_FAMILY" if _uses_result_quality else "NCAAF_SCHEDULE_RESUME_FAMILY" if _uses_resume else "NCAAF_RECENT_VS_SEASON_FAMILY" if _uses_recent else "NCAAF_MATCHUP_DIFFERENTIAL_FAMILY" if _uses_matchup else fid,
             "contains_meta_constituent_index":any(_v2182_is_meta_constituent_index_atom_name(c) for c in _rep_conditions),
             "meta_constituent_overlap_guard":"PASS" if not _v2182_meta_overlap_violation(_rep_conditions) else "FAIL",
             "miner_lane":rep.get("miner_lane"),"member_lanes":sorted(set(str(x.get("miner_lane") or "") for x in members)),
             "discovery_rate":rep["discovery_rate"],"discovery_n":rep["discovery_n"],"confirmation_rate":rep["confirmation_rate"],
             "confirmation_n":rep["confirmation_n"],"confirmation_pass":bool(rep["confirmation_pass"]),
             "split_audit":rep.get("split_audit") or {},
             "authority_state":"CONFIRMED_SHADOW" if rep["confirmation_pass"] else "DISCOVERY_FROZEN","production_authority":0,
             "attribution":_mechanism_attribution(games,seasons,rep,market)}
        # One source-neutral gate for every Miner mechanism. PT-derived systems
        # do not receive a free pass, but are no longer blanket-disqualified.
        strong_validated=_miner_live_authority_eligible(rep)
        fam["current_qualified"]=strong_validated
        fam["live_authority_eligible"]=strong_validated
        fam["live_authority_policy"]=NCAAF_MINER_LIVE_AUTHORITY_POLICY
        fam["evidence_level"]=_evidence_tier(rep,market)
        fam["parent_system_id"]=rep.get("parent_system_id")
        fam["lineage_depth"]=int(rep.get("lineage_depth",0) or 0)
        fam["nested_total_variant"]=bool(rep.get("nested_total_variant",False))
        fam["lineage_incremental"]=rep.get("lineage_incremental") or {}
        if market=="h2h":
            fam["discovery_price_roi"]=(rep.get("h2h_discovery_price") or {}).get("roi")
            fam["confirmation_price_roi"]=(rep.get("h2h_confirmation_price") or {}).get("roi")
            fam["discovery_market_residual"]=rep.get("market_residual_discovery")
            fam["confirmation_market_residual"]=rep.get("market_residual_confirmation")
            fam["price_band_robust_discovery"]=(rep.get("h2h_discovery_price") or {}).get("price_band_robust")
            fam["price_band_robust_confirmation"]=(rep.get("h2h_confirmation_price") or {}).get("price_band_robust")
        if fam.get("meta_constituent_overlap_guard")!="PASS":
            continue
        families.append(fam)
    clean=[]
    for z in finalists:
        z=dict(z); z.pop("_mask_internal",None); clean.append(z)
    _all_tested=list(legacy_tested)+list(observation_tested)+list(external_tested)
    def _special_family_stats(family_name: str, evidence_key: str) -> dict[str,Any]:
        tested=[z for z in _all_tested if family_name in set(z.get("families") or [])]
        final=[z for z in clean if family_name in set(z.get("families") or [])]
        mechs=[z for z in families if family_name in set(z.get("families") or [])]
        confirmed=[z for z in mechs if bool(z.get("confirmation_pass"))]
        live=[z for z in mechs if bool(z.get("live_authority_eligible"))]
        return {
            "family":family_name,
            "evidence_family_key":evidence_key,
            "tested_hypotheses":len(tested),
            "finalist_systems":len(final),
            "mechanism_families":len(mechs),
            "confirmed_mechanisms":len(confirmed),
            "live_authority_mechanisms":len(live),
            "independent_live_family_votes":1 if live else 0,
            "confirmed_rules":[{"mechanism_id":z.get("mechanism_id"),"market":z.get("market"),"direction":z.get("direction"),"confirmation_n":z.get("confirmation_n"),"confirmation_rate":z.get("confirmation_rate"),"live_authority_eligible":bool(z.get("live_authority_eligible")),"rule":" AND ".join(z.get("representative_conditions") or [])} for z in confirmed],
        }
    def _multi_family_stats(family_names: set[str], label: str) -> dict[str,Any]:
        def _uses(z): return bool(set(z.get("families") or []) & family_names)
        tested=[z for z in _all_tested if _uses(z)]
        final=[z for z in clean if _uses(z)]
        mechs=[z for z in families if _uses(z) or (label=="BIG_AL_OBSERVATION_HYPOTHESIS" and bool(z.get("uses_big_al_observation_hypothesis")))]
        confirmed=[z for z in mechs if bool(z.get("confirmation_pass"))]
        live=[z for z in mechs if bool(z.get("live_authority_eligible"))]
        return {
            "label":label,"atom_families":sorted(family_names),"tested_hypotheses":len(tested),
            "finalist_systems":len(final),"mechanism_families":len(mechs),
            "confirmed_mechanisms":len(confirmed),"live_authority_mechanisms":len(live),
            "independent_live_family_votes":len(set((str(z.get("market")),str(z.get("evidence_family_key") or z.get("mechanism_id"))) for z in live)),
            "confirmed_rules":[{"mechanism_id":z.get("mechanism_id"),"market":z.get("market"),"direction":z.get("direction"),"confirmation_n":z.get("confirmation_n"),"confirmation_rate":z.get("confirmation_rate"),"live_authority_eligible":bool(z.get("live_authority_eligible")),"rule":" AND ".join(z.get("representative_conditions") or [])} for z in confirmed],
            "rating_used_for_qualification":False,"2026_outcomes_used":False,"source_after_validation":"MINER",
        }
    special_family_audit={
        "season_record_state":_special_family_stats("SEASON_RECORD_STATE","NCAAF_SEASON_RECORD_STATE_FAMILY"),
        "role_transition_bounceback":_special_family_stats("ROLE_TRANSITION_BOUNCEBACK","NCAAF_ROLE_TRANSITION_BOUNCEBACK_FAMILY"),
        "external_ratings":_special_family_stats("EXTERNAL_RATINGS_FAMILY","EXTERNAL_RATINGS_FAMILY"),
        "result_quality_regression":_special_family_stats("RESULT_QUALITY_REGRESSION","NCAAF_RESULT_QUALITY_FAMILY"),
        "schedule_resume_quality":_special_family_stats("SCHEDULE_RESUME_QUALITY","NCAAF_SCHEDULE_RESUME_FAMILY"),
        "recent_vs_season":_special_family_stats("RECENT_VS_SEASON","NCAAF_RECENT_VS_SEASON_FAMILY"),
        "matchup_differential":_special_family_stats("MATCHUP_DIFFERENTIAL","NCAAF_MATCHUP_DIFFERENTIAL_FAMILY"),
        "big_al_observation_hypothesis":_multi_family_stats(BIG_AL_OBSERVATION_ATOM_FAMILIES,"BIG_AL_OBSERVATION_HYPOTHESIS"),
    }
    if market=="spreads":
        special_family_audit["atom_availability"]=_special_atom_availability
    out.update({"status":"RESEARCH_COMPLETE","tested_hypotheses":len(legacy_tested)+len(observation_tested)+len(external_tested),"systems":clean,"published_systems":len(clean),
                "mechanism_families":families,"mechanism_family_count":len(families),
                "confirmed_mechanism_count":sum(x["confirmation_pass"] for x in families),
                "special_family_audit":special_family_audit,
                "lineage":lineage,
                "admission_contract":"DISCOVERY_2022_2023_ONLY__LEGACY_MINER_LANE_PRESERVED__PERSISTENT_INCUMBENT_EXACT_RULE_REVALIDATION__BIGAL_OBSERVATION_HYPOTHESIS_LANE_SEPARATE_FDR__BIGAL_RATINGS_ZERO_SELECTION_WEIGHT__2026_EXPERT_PICK_OUTCOMES_UNUSED__EXTERNAL_PREDICTOR_BEHAVIOR_LANE_SEPARATE_FDR__EXTERNAL_SEEDS_DISCOVERY_ONLY__ONE_EXTERNAL_FAMILY_PER_RULE__HORIZON_SYMMETRIC_1_2_3__CONFERENCE_RIVALRY_H2H_ROLE_TEAM_MEMORY__MARKET_RELATIVE_FDR_FOR_H2H__OBSERVED_ML_ROI__PRICE_BAND_ROBUSTNESS__BOTH_2024_AND_2025_CONFIRM__DEPENDENCY_COLLAPSE_AFTER_LANE_MERGE__PARENT_CHILD_INCREMENTAL_ATTRIBUTION__FULL_PT_INDEX_UNIVERSE_ONE_CORRELATED_FAMILY__META_CONSTITUENT_OVERLAP_GUARD__META_SOURCE_OUT_CONSENSUS__SPLIT_OVERLAP_AUDIT__2026_PROSPECTIVE_ONLY__ZERO_AUTHORITY"})
    log_func(f"[NCAAF-RV23-MINER] market={market} atoms={len(atoms)} tested={len(legacy_tested)+len(observation_tested)+len(external_tested)} systems={len(clean)} mechanisms={len(families)} confirmed_mechanisms={out['confirmed_mechanism_count']} authority=0")
    _oa=special_family_audit.get("big_al_observation_hypothesis") or {}
    if market in {"spreads","totals"}:
        log_func(f"[NCAAF-BIGAL-HYPOTHESIS-AUDIT] market={market} tested={_oa.get('tested_hypotheses',0)} finalists={_oa.get('finalist_systems',0)} mechanisms={_oa.get('mechanism_families',0)} confirmed={_oa.get('confirmed_mechanisms',0)} live={_oa.get('live_authority_mechanisms',0)} independent_live_families={_oa.get('independent_live_family_votes',0)} rating_weight=0 outcomes_2026_used=FALSE source=MINER")
    for x in families[:20]:
        extra=(f" d_roi={x.get('discovery_price_roi')} c_roi={x.get('confirmation_price_roi')} d_resid={x.get('discovery_market_residual')} c_resid={x.get('confirmation_market_residual')}" if market=="h2h" else "")
        log_func(f"[NCAAF-RV23-MECHANISM] market={market} id={x['mechanism_id']} lane={x.get('miner_lane')} status={x['authority_state']} members={x['member_count']} discovery={x['discovery_rate']:.4f}/{x['discovery_n']} confirmation={x['confirmation_rate']:.4f}/{x['confirmation_n']} rule={' AND '.join(x.get('representative_conditions') or [])}{extra}")
        _sa=x.get("split_audit") or {}
        if _sa.get("identical_n_rate"):
            log_func(
                f"[NCAAF-RV2182-SPLIT-AUDIT] market={market} id={x['mechanism_id']} identical_n_rate=TRUE "
                f"row_overlap_n={_sa.get('row_overlap_n')} disjoint={_sa.get('disjoint')} "
                f"discovery_years={_sa.get('discovery_years')} confirmation_years={_sa.get('confirmation_years')} authority=0"
            )
        if market=="totals" and x.get("confirmation_pass"):
            a=x.get("attribution") or {}
            log_func(f"[NCAAF-RV23-TOTALS-ATTRIBUTION] id={x['mechanism_id']} rule={a.get('rule')} seasons={json.dumps(a.get('season_breakdown') or [],sort_keys=True,default=str)} remove_best={a.get('remove_best_discovery_rate')} team_concentration={json.dumps(a.get('team_concentration') or {},sort_keys=True,default=str)} conference_concentration={json.dumps(a.get('conference_concentration') or {},sort_keys=True,default=str)}")
    return out


def _log_v24_evidence_audit(*, miners: dict[str,Any], system_results: dict[str,Any], published_system_results: dict[str,Any],
                            threshold_neighborhood: dict[str,Any], log_func=print) -> None:
    """Emit concise audit lines for report sections that previously existed only in JSON/UI."""
    for market,mr in (miners or {}).items():
        lin=(mr or {}).get("lineage") or {}; pairs=list(lin.get("pairs") or [])
        roots=sum(1 for x in (mr or {}).get("systems",[]) if not x.get("parent_system_id"))
        children=sum(1 for x in (mr or {}).get("systems",[]) if x.get("parent_system_id"))
        log_func(f"[NCAAF-RV24-LINEAGE] market={market} pairs={len(pairs)} roots={roots} children={children} nested_total_variants={int(lin.get('nested_total_variants',0) or 0)} selection_influence=0 authority=0")
        ranked=sorted([x for x in pairs if int(x.get("confirmation_child_n",0) or 0)>=20 and np.isfinite(float(x.get("confirmation_rate_delta_vs_parent_only",np.nan)))],
                      key=lambda x:(float(x.get("confirmation_rate_delta_vs_parent_only",-9)),int(x.get("confirmation_child_n",0) or 0)),reverse=True)[:8]
        for x in ranked:
            log_func(f"[NCAAF-RV24-LINEAGE-TOP] market={market} parent={x.get('parent_system_id')} child={x.get('child_system_id')} added={'+'.join(map(str,x.get('added_conditions') or []))} nested_total={bool(x.get('nested_total_variant'))} conf_child_n={x.get('confirmation_child_n')} conf_parent_only_n={x.get('confirmation_parent_only_n')} conf_rate_delta={float(x.get('confirmation_rate_delta_vs_parent_only')):+.4f} authority=0")

    for x in (published_system_results or {}).get("systems",[]):
        d=x.get("discovery") or {}; c=x.get("confirmation") or {}; by={int(z.get("season")):z for z in (x.get("by_season") or []) if z.get("season") is not None}
        log_func(f"[NCAAF-RV24-PUBLISHED] source={x.get('source')} system={x.get('system')} label={x.get('label')} status={x.get('status')} assessment={x.get('source_direction_assessment')} d_n={d.get('n',0)} d_rate={d.get('hit_rate')} d_wilson={d.get('wilson95_low')} c_n={c.get('n',0)} c_rate={c.get('hit_rate')} c_shrunk={c.get('shrunk_hit_rate_beta15')} c_wilson={c.get('wilson95_low')} y2024_n={(by.get(2024) or {}).get('n',0)} y2024_rate={(by.get(2024) or {}).get('hit_rate')} y2025_n={(by.get(2025) or {}).get('n',0)} y2025_rate={(by.get(2025) or {}).get('hit_rate')} authority=0")

    for market,z in ((system_results or {}).get("markets") or {}).items():
        anyr=z.get("any_confirmed_system") or {}; multi=z.get("two_plus_independent") or {}
        log_func(f"[NCAAF-RV24-SYSTEM-RESULTS] market={market} confirmed_mechanisms={z.get('confirmed_mechanisms',0)} any_n={anyr.get('n',0)} any_rate={anyr.get('hit_rate')} any_wilson={anyr.get('wilson95_low')} multi_n={multi.get('n',0)} multi_rate={multi.get('hit_rate')} multi_wilson={multi.get('wilson95_low')} conflicts={z.get('conflict_rows',0)} authority=0")

    for x in (threshold_neighborhood or {}).get("grid",[]):
        log_func(f"[NCAAF-RV24-THRESHOLD] min_n={x.get('min_confirmation_n')} min_rate={x.get('min_confirmation_rate')} qualified={x.get('qualified_count')} by_market={json.dumps(x.get('by_market') or {},sort_keys=True)} frozen_gate={bool(x.get('is_frozen_live_gate'))} selection_influence=0 authority=0")


def _prospective_shadow(full_games: pd.DataFrame, full_seasons: np.ndarray, miners: dict[str,Any], dashboard_module=None, log_func=print) -> dict[str,Any]:
    if full_games is None or full_games.empty or len(full_games)!=len(full_seasons): return {"status":"UNAVAILABLE","production_authority":0}
    out={"status":"PASS","season_min":PROSPECTIVE_MIN_SEASON,"mechanisms":[],"production_authority":0,"selection_influence":0}
    for market,mr in (miners or {}).items():
        atoms={a["name"]:np.asarray(a["mask"],dtype=bool) for a in _extended_atoms(full_games,dashboard_module,for_live=False,market=market)}
        y,valid,baseline=_market_target(full_games,market)
        for mech in (mr or {}).get("mechanism_families",[]):
            if not mech.get("confirmation_pass"): continue
            cond=mech.get("representative_conditions") or []; mask=np.ones(len(full_games),dtype=bool)
            _research_bridge=any(str(c).startswith("CORE_OOF_") or str(c).startswith("SPEC_") for c in cond)
            if _research_bridge:
                out["mechanisms"].append({"market":market,"mechanism_id":mech.get("mechanism_id"),"rule":" AND ".join(cond),"trigger_n":None,"settled_n":None,
                                          "prospective_evaluable":False,"reason":"OOF_CORE_SPECIALIST_LIVE_BRIDGE_NOT_WIRED","production_authority":0})
                continue
            for c in cond:
                if c not in atoms: mask[:]=False; break
                mask &= atoms[c]
            pmask=mask&np.isfinite(full_seasons)&(full_seasons>=PROSPECTIVE_MIN_SEASON)
            direction=mech.get("direction"); obs=y if direction=="PLAY_ON" else 1-y
            settled=pmask&valid
            row={"market":market,"mechanism_id":mech.get("mechanism_id"),"rule":" AND ".join(cond),"trigger_n":int(pmask.sum()),"settled_n":int(settled.sum()),"production_authority":0}
            if settled.any(): row["rate"]=float(np.mean(obs[settled]))
            if market in ("spreads","totals") and settled.any():
                rate=row["rate"]; row["roi_at_minus110"]=rate*(100/110.0)-(1-rate)
            if market=="h2h":
                pm=_h2h_price_metrics(full_games,pmask&valid,y,baseline,full_seasons,direction,sorted(set(int(x) for x in full_seasons[pmask&np.isfinite(full_seasons)])))
                row["price_roi"]=pm.get("roi"); row["market_residual"]=pm.get("market_residual"); row["priced_n"]=pm.get("priced_n")
            out["mechanisms"].append(row)
            log_func(f"[NCAAF-RV22-PROSPECTIVE] market={market} mechanism={row['mechanism_id']} triggers={row['trigger_n']} settled={row['settled_n']} rate={row.get('rate')} roi={row.get('roi_at_minus110',row.get('price_roi'))} authority=0")
    return out



def _wilson95_low(wins: int, n: int) -> float:
    n=int(n or 0); wins=int(wins or 0)
    if n<=0: return float("nan")
    z=1.959963984540054; p=wins/n; den=1+z*z/n
    ctr=p+z*z/(2*n); rad=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)
    return float((ctr-rad)/den)


def _minus110_roi(rate: float) -> float:
    return float(rate*(100.0/110.0)-(1.0-rate)) if np.isfinite(rate) else float("nan")


def _beta_shrunk_rate(wins: int, n: int, prior_wins: float=15.0, prior_losses: float=15.0) -> float:
    n=int(n or 0); wins=int(wins or 0)
    return float((wins+prior_wins)/(n+prior_wins+prior_losses)) if n>=0 else float("nan")


def _miner_authority_threshold_neighborhood(miners: dict[str,Any]) -> dict[str,Any]:
    """Show sensitivity near the frozen 60 / 56% live-authority gate.

    This does not choose a threshold and has zero selection influence. It answers
    whether the qualified family count is brittle to small neighboring cutoffs.
    """
    rows=[]
    for min_n in (50,60,70):
        for min_rate in (.55,.56,.57):
            by_market={}; total=0
            for market,mr in (miners or {}).items():
                n=sum(bool(x.get("confirmation_pass")) and int(x.get("confirmation_n",0) or 0)>=min_n and
                      np.isfinite(float(x.get("confirmation_rate",np.nan))) and float(x.get("confirmation_rate",0) or 0)>=min_rate
                      for x in (mr or {}).get("mechanism_families",[]))
                by_market[market]=int(n); total+=int(n)
            rows.append({"min_confirmation_n":min_n,"min_confirmation_rate":min_rate,"qualified_count":total,"by_market":by_market,
                         "is_frozen_live_gate":bool(min_n==NCAAF_MINER_LIVE_MIN_CONFIRMATION_N and abs(min_rate-NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE)<1e-12)})
    return {"status":"PASS","grid":rows,"frozen_gate":{"min_confirmation_n":NCAAF_MINER_LIVE_MIN_CONFIRMATION_N,"min_confirmation_rate":NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE},
            "policy":"DIAGNOSTIC_NEIGHBORHOOD_ONLY__NO_RETUNING__ZERO_SELECTION_INFLUENCE","selection_influence":0,"production_authority":0}


def _build_system_results(games: pd.DataFrame, seasons: np.ndarray, miners: dict[str,Any], dashboard_module=None) -> dict[str,Any]:
    """Direct 2024-25 grading of confirmed Miner mechanisms and confluence.

    Mechanism families are already dependency-collapsed. A multi-system row is
    counted only when 2+ confirmed mechanism families point the same direction
    and no confirmed mechanism points the opposite direction.
    """
    if games is None or games.empty: return {"status":"UNAVAILABLE"}
    out={"status":"PASS","confirmation_seasons":list(CONFIRMATION_SEASONS),"markets":{}}
    for market,mr in (miners or {}).items():
        atoms={a["name"]:np.asarray(a["mask"],dtype=bool) for a in _extended_atoms(games,dashboard_module,for_live=False,market=market)}
        y,valid,baseline=_market_target(games,market)
        scope=valid&np.isfinite(seasons)&np.isin(seasons,np.asarray(CONFIRMATION_SEASONS,dtype=float))
        mechs=[]
        for mech in (mr or {}).get("mechanism_families",[]):
            if not mech.get("confirmation_pass"): continue
            cond=list(mech.get("representative_conditions") or []); mask=np.ones(len(games),dtype=bool)
            evaluable=bool(cond)
            for c in cond:
                if c not in atoms: evaluable=False; mask[:]=False; break
                mask &= atoms[c]
            if not evaluable: continue
            direction=str(mech.get("direction") or "PLAY_ON").upper()
            obs=y if direction=="PLAY_ON" else 1-y
            m=mask&scope; n=int(m.sum()); w=int(np.nansum(obs[m])) if n else 0; l=n-w
            rate=float(w/n) if n else np.nan
            _uses_external=bool("EXTERNAL_RATINGS_FAMILY" in set(mech.get("families") or []))
            mechs.append({"mechanism_id":mech.get("mechanism_id"),"direction":direction,"mask":mask,"obs":obs,"families":list(mech.get("families") or []),
                          "evidence_family_key":str(mech.get("evidence_family_key") or ("EXTERNAL_RATINGS_FAMILY" if _uses_external else mech.get("mechanism_id"))),
                          "uses_external_ratings_family":_uses_external,
                          "rule":" AND ".join(cond),"n":n,"wins":w,"losses":l,"hit_rate":rate,"roi_at_minus110":_minus110_roi(rate) if market in ("spreads","totals") else np.nan,
                          "wilson95_low":_wilson95_low(w,n),"shrunk_hit_rate_beta15":_beta_shrunk_rate(w,n),"evidence_level":mech.get("evidence_level"),"current_qualified":True})
        best=sorted([{k:v for k,v in z.items() if k not in {"mask","obs"}} for z in mechs],key=lambda z:(z.get("wilson95_low",-9),z.get("n",0)),reverse=True)
        # Independence count is by evidence family. Every confirmed external
        # mechanism, regardless of which PT index/consensus atom produced it,
        # contributes at most ONE independent external-family vote per direction.
        play=np.zeros(len(games),dtype=int); fade=np.zeros(len(games),dtype=int)
        _independent_masks={}
        for z in mechs:
            _k=(z["direction"],str(z.get("evidence_family_key") or z["mechanism_id"]))
            if _k not in _independent_masks:
                _independent_masks[_k]=np.zeros(len(games),dtype=bool)
            _independent_masks[_k] |= np.asarray(z["mask"],dtype=bool)
        for (_direction,_family_key),_mask in _independent_masks.items():
            if _direction=="PLAY_ON": play += _mask.astype(int)
            else: fade += _mask.astype(int)
        agreed=((play>0)^(fade>0))&scope
        multi=(((play>=2)&(fade==0))|((fade>=2)&(play==0)))&scope
        conflict=(play>0)&(fade>0)&scope
        def pool(mask):
            idx=np.flatnonzero(mask)
            if not len(idx): return {"n":0,"wins":0,"losses":0,"hit_rate":None,"roi_at_minus110":None,"wilson95_low":None,"shrunk_hit_rate_beta15":None}
            chosen=np.where(play[idx]>0,y[idx],1-y[idx]); w=int(np.nansum(chosen)); n=int(len(idx)); rate=float(w/n)
            return {"n":n,"wins":w,"losses":n-w,"hit_rate":rate,"roi_at_minus110":_minus110_roi(rate) if market in ("spreads","totals") else None,"wilson95_low":_wilson95_low(w,n),"shrunk_hit_rate_beta15":_beta_shrunk_rate(w,n)}
        best_conf=[]
        for z in mechs:
            same_count=play if z["direction"]=="PLAY_ON" else fade; opp_count=fade if z["direction"]=="PLAY_ON" else play
            mm=z["mask"]&scope&(same_count>=2)&(opp_count==0); n=int(mm.sum())
            if not n: continue
            obs=z["obs"]; w=int(np.nansum(obs[mm])); rate=float(w/n)
            best_conf.append({"mechanism_id":z["mechanism_id"],"rule":z["rule"],"confluence_n":n,"wins":w,"losses":n-w,"hit_rate":rate,
                              "roi_at_minus110":_minus110_roi(rate) if market in ("spreads","totals") else np.nan,"wilson95_low":_wilson95_low(w,n),"shrunk_hit_rate_beta15":_beta_shrunk_rate(w,n)})
        best_conf=sorted(best_conf,key=lambda z:(z["wilson95_low"],z["confluence_n"]),reverse=True)
        out["markets"][market]={"confirmed_mechanisms":len(mechs),"any_confirmed_system":pool(agreed),"two_plus_independent":pool(multi),"conflict_rows":int(conflict.sum()),
                                "best_systems":best,"best_in_confluence":best_conf}
    return out


def _norm_team_live(v: Any) -> str:
    return " ".join(str(v or "").strip().lower().split())


def _pt_attach_current_live_frame(cdf: pd.DataFrame, dashboard_module=None) -> tuple[pd.DataFrame,dict[str,Any]]:
    """Attach the latest GCS-cached Prediction Tracker META_MARGIN to live home rows.

    This is display/research context only. Missing/stale/unmatched data simply
    leaves the external fields NaN; no production scoring path depends on it.
    """
    out=cdf.copy()
    try:
        if dashboard_module is None: return out,{"status":"NO_DASHBOARD","matched":0}
        bucket_name=str(getattr(dashboard_module,"GCS_BUCKET","sharp-models") or "sharp-models")
        sc=getattr(dashboard_module,"gcs_client",None)
        if sc is None:
            from google.cloud import storage
            sc=storage.Client()
        raw=sc.bucket(bucket_name).blob(PT_CURRENT_BLOB).download_as_bytes()
        ex=pd.read_csv(io.BytesIO(raw))
        if ex.empty: return out,{"status":"EMPTY","matched":0}
        homes=out.get("Home_Team_Norm",out.get("Home_Team",pd.Series("",index=out.index))).astype(str).map(_pt_team_key)
        aways=out.get("Away_Team_Norm",out.get("Away_Team",pd.Series("",index=out.index))).astype(str).map(_pt_team_key)
        internal=pd.concat([homes,aways],ignore_index=True).tolist()
        emap,_un=_pt_candidate_map(pd.concat([ex.get("home_key",pd.Series(dtype=str)),ex.get("away_key",pd.Series(dtype=str))],ignore_index=True),internal)
        ex=ex.copy(); ex["home_i"]=ex.get("home_key",pd.Series("",index=ex.index)).map(emap); ex["away_i"]=ex.get("away_key",pd.Series("",index=ex.index)).map(emap)
        ex=ex.loc[ex["home_i"].notna()&ex["away_i"].notna()].copy(); ex["pair_key"]=ex["home_i"].astype(str)+"|"+ex["away_i"].astype(str)
        vc=ex["pair_key"].value_counts(); ex=ex.loc[ex["pair_key"].map(vc).eq(1)].set_index("pair_key",drop=False)
        meta=np.full(len(out),np.nan); cnt=np.full(len(out),np.nan); pavg=np.full(len(out),np.nan)
        ext_consensus=np.full(len(out),np.nan); ext_clusters=np.full(len(out),np.nan); ext_indices=np.full(len(out),np.nan)
        ext_std=np.full(len(out),np.nan); ext_iqr=np.full(len(out),np.nan)
        listed=np.zeros(len(out),dtype=float); full_five=np.zeros(len(out),dtype=float)
        comp={k:np.full(len(out),np.nan) for k in PT_PUBLISHED_WEIGHTS}
        for i,(h,a) in enumerate(zip(homes,aways)):
            k=f"{h}|{a}"
            if k not in ex.index: continue
            r=ex.loc[k]
            if isinstance(r,pd.DataFrame): continue
            listed[i]=1.0
            component_count=0
            for _k in PT_PUBLISHED_WEIGHTS:
                _v=pd.to_numeric(pd.Series([r.get(_k,np.nan)]),errors="coerce").iloc[0]
                if pd.notna(_v):
                    comp[_k][i]=float(_v)
                    component_count+=1
            cnt[i]=float(component_count)
            full_five[i]=1.0 if component_count==5 else 0.0
            _mv=pd.to_numeric(pd.Series([r.get("meta_margin_home",np.nan)]),errors="coerce").iloc[0]
            if pd.notna(_mv) and component_count==5: meta[i]=float(_mv)
            _pa=pd.to_numeric(pd.Series([r.get("prediction_avg_home",np.nan)]),errors="coerce").iloc[0]
            if pd.notna(_pa): pavg[i]=float(_pa)
            _ec=pd.to_numeric(pd.Series([r.get("external_consensus_home_margin",np.nan)]),errors="coerce").iloc[0]
            if pd.notna(_ec): ext_consensus[i]=float(_ec)
            _en=pd.to_numeric(pd.Series([r.get("external_consensus_index_count",np.nan)]),errors="coerce").iloc[0]
            if pd.notna(_en): ext_indices[i]=float(_en)
            _ecn=pd.to_numeric(pd.Series([r.get("external_consensus_cluster_count",np.nan)]),errors="coerce").iloc[0]
            if pd.notna(_ecn): ext_clusters[i]=float(_ecn)
            _es=pd.to_numeric(pd.Series([r.get("external_consensus_cluster_std",np.nan)]),errors="coerce").iloc[0]
            if pd.notna(_es): ext_std[i]=float(_es)
            _ei=pd.to_numeric(pd.Series([r.get("external_consensus_cluster_iqr",np.nan)]),errors="coerce").iloc[0]
            if pd.notna(_ei): ext_iqr[i]=float(_ei)
        spread=pd.to_numeric(out.get("Consensus_Open_Spread",out.get("Opening_Spread",pd.Series(np.nan,index=out.index))),errors="coerce").to_numpy(float)
        market=-spread
        out["_V210_PT_META_MARGIN_TEAM"]=meta
        out["_V210_PT_META_EDGE_POINTS"]=meta-market
        out["_V210_PT_META_SYSTEM_COUNT"]=cnt
        out["_V210_PT_PREDICTION_AVG_TEAM"]=pavg
        out["_V2183_PT_EXTERNAL_CONSENSUS_MARGIN_TEAM"]=ext_consensus
        out["_V2183_PT_EXTERNAL_CONSENSUS_EDGE_POINTS"]=ext_consensus-market
        out["_V2183_PT_EXTERNAL_CONSENSUS_CLUSTER_COUNT"]=ext_clusters
        out["_V2183_PT_EXTERNAL_CONSENSUS_INDEX_COUNT"]=ext_indices
        out["_V2183_PT_EXTERNAL_CONSENSUS_STD"]=ext_std
        out["_V2183_PT_EXTERNAL_CONSENSUS_IQR"]=ext_iqr
        _cm=np.column_stack([comp[k] for k in PT_PUBLISHED_WEIGHTS]); _em=np.column_stack([comp[k]-market for k in PT_PUBLISHED_WEIGHTS])
        _cn=np.isfinite(_cm).sum(axis=1); _std=np.full(len(out),np.nan); _ok=_cn>=2
        if _ok.any(): _std[_ok]=np.nanstd(_cm[_ok],axis=1)
        out["_V212_PT_COMPONENT_STD"]=_std
        out["_V212_PT_COMPONENT_TEAM_AGREE_COUNT"]=np.sum(np.isfinite(_em)&(_em>0),axis=1).astype(float)
        out["_V212_PT_COMPONENT_OPP_AGREE_COUNT"]=np.sum(np.isfinite(_em)&(_em<0),axis=1).astype(float)
        out["_V217_PT_GAME_LISTED"]=listed
        out["_V217_PT_COMPONENT_AVAILABLE_COUNT"]=_cn.astype(float)
        out["_V217_PT_FULL_FIVE"]=full_five
        for _k in PT_PUBLISHED_WEIGHTS:
            out[f"_V212_PT_{_k}_MARGIN_TEAM"]=comp[_k]
            out[f"_V212_PT_{_k}_EDGE_POINTS"]=comp[_k]-market
        return out,{
            "status":"PASS","matched":int(listed.sum()),"any_component":int((_cn>0).sum()),
            "partial":int(((_cn>0)&(_cn<5)).sum()),"full_five":int((_cn==5).sum()),
            "external_consensus_ready":int(np.isfinite(ext_consensus).sum()),
            "external_consensus_contract":PT_EXTERNAL_CONSENSUS_CONTRACT,
            "rows":int(len(out)),"sparse":True,"authority":0
        }
    except Exception as exc:
        return out,{"status":"UNAVAILABLE","matched":0,"error":f"{type(exc).__name__}:{exc}","authority":0}


def _miner_live_authority_eligible(mech: dict[str,Any] | None) -> bool:
    """Frozen historical gate for whether a Miner family may influence Bet Authority.

    2026+ outcomes are intentionally absent from this decision. A family must have
    passed the original 2024+2025 confirmation gate AND clear the stronger pooled
    confirmation sample/rate floor. We recompute from immutable report fields so
    the V2.2 report already published before this runtime patch remains usable.
    """
    m=mech or {}
    # A previously validated incumbent whose exact historical inputs are
    # temporarily unavailable remains in the library, but is fail-closed for
    # live authority until exact-rule revalidation succeeds again.
    _inc_status=str(m.get("incumbent_revalidation_status") or "")
    if _inc_status=="HOLD_NOT_EVALUABLE": return False
    if _inc_status=="LEGACY_CONTRACT_HOLD":
        return bool(m.get("incumbent_prior_live_authority_eligible",m.get("live_authority_eligible",False)))
    _conds=[str(x) for x in (m.get("representative_conditions") or m.get("conditions") or [])]
    # CORE OOF / specialist research bridges are not live-authority inputs.
    # Prediction Tracker is intentionally NOT blocked by source: PT-derived
    # mechanisms must clear the same confirmation gate as all other Miner rules.
    if any(x.startswith("CORE_OOF_") or x.startswith("SPEC_") for x in _conds): return False
    if not bool(m.get("confirmation_pass")): return False
    try: n=int(m.get("confirmation_n",0) or 0)
    except Exception: n=0
    try: rate=float(m.get("confirmation_rate",0) or 0)
    except Exception: rate=0.0
    return bool(n>=NCAAF_MINER_LIVE_MIN_CONFIRMATION_N and np.isfinite(rate) and rate>=NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE)


def _attach_role_transition_live_context(cdf: pd.DataFrame, dashboard_module=None, log_func=print) -> tuple[pd.DataFrame,dict[str,Any]]:
    """Fill prior-home/role/result state for live bounceback evaluation.

    Completed 2026 games are trigger context only. They never enter historical
    qualification, selection, or promotion. Existing populated fields win.
    """
    if cdf is None or cdf.empty: return cdf,{"status":"EMPTY","selection_influence":0}
    out=cdf.copy()
    need=("Prev_Is_Home","Opp_Prev_Is_Home","Prev_Team_Score","Prev_Opponent_Score",
          "Opp_Prev_Team_Score","Opp_Prev_Opponent_Score","ATS_WinPct_Prior","Opp_ATS_WinPct_Prior",
          "Prev_Is_ML_Dog","Opp_Prev_Is_ML_Dog","Team_Game_Number_Prior","Opp_Game_Number_Prior",
          "ATS_Loss_Streak_Prior","Opp_ATS_Loss_Streak_Prior")
    if all(c in out.columns and pd.to_numeric(out[c],errors="coerce").notna().any() for c in need):
        return out,{"status":"ALREADY_PRESENT","selection_influence":0}
    bq=getattr(dashboard_module,"bq_client",None) if dashboard_module is not None else None
    view=getattr(dashboard_module,"HISTORICAL_NCAAF_CORE_VIEW","sharplogger.sharp_data.ncaaf_historical_core_training_vw") if dashboard_module is not None else "sharplogger.sharp_data.ncaaf_historical_core_training_vw"
    if bq is None: return out,{"status":"NO_BQ_CLIENT","selection_influence":0}
    try:
        h=bq.query(f"SELECT * FROM `{view}` WHERE Season=2026 AND Team_Score IS NOT NULL AND Opponent_Score IS NOT NULL").to_dataframe(create_bqstorage_client=False)
        if h is None or h.empty: return out,{"status":"NO_COMPLETED_2026_HISTORY","selection_influence":0}
        def _txt(df,*cols):
            z=pd.Series("",index=df.index,dtype="string")
            for c in cols:
                if c in df.columns:
                    v=df[c].astype("string").fillna("").str.lower().str.strip(); z=z.where(z.ne(""),v)
            return z
        def _numc(df,*cols):
            z=pd.Series(np.nan,index=df.index,dtype=float)
            for c in cols:
                if c in df.columns:
                    v=pd.to_numeric(df[c],errors="coerce"); z=z.where(z.notna(),v)
            return z
        h=h.copy(); h["__team"]=_txt(h,"Team_Norm","Team"); h["__opp"]=_txt(h,"Opponent_Norm","Opponent")
        h["__date"]=pd.to_datetime(h.get("Game_Date",h.get("Game_Start")),errors="coerce",utc=True)
        h["__home"]=_numc(h,"Is_Home")
        h["__spread"]=_numc(h,"Opening_Spread","Consensus_Open_Spread")
        h["__pf"]=_numc(h,"Team_Score","Points_For")
        h["__pa"]=_numc(h,"Opponent_Score","Points_Against")
        h["__margin"]=h["__pf"]-h["__pa"]
        h["__ats_margin"]=h["__margin"]+h["__spread"]
        h=h.loc[h["__team"].ne("")&h["__date"].notna()].sort_values(["__team","__date"],kind="stable")
        states={}
        for team,g in h.groupby("__team",sort=False):
            margins=pd.to_numeric(g["__margin"],errors="coerce").dropna()
            wp=float(((margins>0).sum()+.5*(margins==0).sum())/len(margins)) if len(margins) else np.nan
            ats=pd.to_numeric(g["__ats_margin"],errors="coerce")
            graded=ats.notna()&~np.isclose(ats,0.0,atol=1e-9)
            ats_pct=float((ats[graded]>0).mean()) if int(graded.sum()) else np.nan
            ats_loss_streak=ats_win_streak=0
            for _am in ats.to_list():
                if pd.notna(_am) and not np.isclose(float(_am),0.0,atol=1e-9):
                    if float(_am)>0: ats_win_streak+=1; ats_loss_streak=0
                    else: ats_loss_streak+=1; ats_win_streak=0
                else: ats_loss_streak=0; ats_win_streak=0
            su_win_streak=su_loss_streak=0
            for _sm in pd.to_numeric(g["__margin"],errors="coerce").to_list():
                if pd.isna(_sm) or np.isclose(float(_sm),0.0,atol=1e-9): su_win_streak=0; su_loss_streak=0
                elif float(_sm)>0: su_win_streak+=1; su_loss_streak=0
                else: su_loss_streak+=1; su_win_streak=0
            last=g.iloc[-1]; sp=float(last["__spread"]) if pd.notna(last["__spread"]) else np.nan
            states[str(team)]={"games_prior":float(len(margins)),
                               "prev_home":float(last["__home"]) if pd.notna(last["__home"]) else np.nan,
                               "prev_fav":float(sp<0) if np.isfinite(sp) else np.nan,
                               "prev_dog":float(sp>0) if np.isfinite(sp) else np.nan,
                               "prev_margin":float(last["__margin"]) if pd.notna(last["__margin"]) else np.nan,
                               "prev_pf":float(last["__pf"]) if pd.notna(last["__pf"]) else np.nan,
                               "prev_pa":float(last["__pa"]) if pd.notna(last["__pa"]) else np.nan,
                               "winpct":wp,"ats_pct":ats_pct,"ats_loss_streak":float(ats_loss_streak),"ats_win_streak":float(ats_win_streak),
                               "su_win_streak":float(su_win_streak),"su_loss_streak":float(su_loss_streak)}
        team=_txt(out,"Team_Norm","Team","Home_Team_Norm","Home_Team")
        opp=_txt(out,"Opponent_Norm","Opponent","Away_Team_Norm","Away_Team")
        fields={"Prev_Is_Home":("prev_home",team),"Prev_Is_ML_Favorite":("prev_fav",team),"Prev_Is_ML_Dog":("prev_dog",team),
                "Prev_SU_Margin":("prev_margin",team),"Prev_Team_Score":("prev_pf",team),"Prev_Opponent_Score":("prev_pa",team),
                "Team_Game_Number_Prior":("games_prior",team),"Team_WinPct_Prior":("winpct",team),"ATS_WinPct_Prior":("ats_pct",team),
                "ATS_Loss_Streak_Prior":("ats_loss_streak",team),"ATS_Win_Streak_Prior":("ats_win_streak",team),
                "Current_Win_Streak_Prior":("su_win_streak",team),"Current_Loss_Streak_Prior":("su_loss_streak",team),
                "Opp_Prev_Is_Home":("prev_home",opp),"Opp_Prev_Is_ML_Favorite":("prev_fav",opp),"Opp_Prev_Is_ML_Dog":("prev_dog",opp),
                "Opp_Prev_SU_Margin":("prev_margin",opp),"Opp_Prev_Team_Score":("prev_pf",opp),"Opp_Prev_Opponent_Score":("prev_pa",opp),
                "Opp_Game_Number_Prior":("games_prior",opp),"Opp_WinPct_Prior":("winpct",opp),"Opp_ATS_WinPct_Prior":("ats_pct",opp),
                "Opp_ATS_Loss_Streak_Prior":("ats_loss_streak",opp),"Opp_ATS_Win_Streak_Prior":("ats_win_streak",opp),
                "Opp_Current_Win_Streak_Prior":("su_win_streak",opp),"Opp_Current_Loss_Streak_Prior":("su_loss_streak",opp)}
        for col,(k,who) in fields.items():
            vals=pd.Series([states.get(str(x),{}).get(k,np.nan) for x in who],index=out.index,dtype=float)
            if col in out.columns:
                cur=pd.to_numeric(out[col],errors="coerce"); out[col]=cur.where(cur.notna(),vals)
            else: out[col]=vals
        diag={"status":"PASS","history_rows":int(len(h)),"teams":int(len(states)),"selection_influence":0,"year_2026_role":"TRIGGER_CONTEXT_ONLY"}
        log_func(f"[NCAAF-BOUNCEBACK-LIVE-CONTEXT] status=PASS history_rows={len(h)} teams={len(states)} selection_influence=0")
        return out,diag
    except Exception as exc:
        return out,{"status":"ERROR","error":f"{type(exc).__name__}:{exc}","selection_influence":0}


def attach_live_miner_votes(rows: pd.DataFrame, report: dict[str,Any], dashboard_module=None) -> pd.DataFrame:
    """Evaluate frozen confirmed Miner mechanisms against current pregame context.

    Historical Miner semantics are canonical-home oriented. Live evaluation first
    reconstructs one home-side context row per physical game, evaluates exactly
    the frozen representative conditions, then attaches directional votes to every
    market row for Bet Authority. 2026 outcomes never select/qualify a mechanism.
    """
    if rows is None or rows.empty or not isinstance(report,dict): return rows
    out=rows.copy()
    keycol="_prod_game_id" if "_prod_game_id" in out.columns else None
    if keycol is None:
        h=out.get("Home_Team_Norm",out.get("Home_Team",pd.Series("",index=out.index))).astype(str).str.lower().str.strip()
        a=out.get("Away_Team_Norm",out.get("Away_Team",pd.Series("",index=out.index))).astype(str).str.lower().str.strip()
        t=pd.to_datetime(out.get("Game_Start"),errors="coerce",utc=True).dt.floor("h").astype(str)
        out["_rv22_game_id"]=h+"|"+a+"|"+t; keycol="_rv22_game_id"
    ctx=[]; keys=[]
    for gid,g in out.groupby(keycol,sort=False,dropna=False):
        home=_norm_team_live(g.get("Home_Team_Norm",g.get("Home_Team",pd.Series("",index=g.index))).iloc[0])
        spg=g[g.get("Market",pd.Series("",index=g.index)).astype(str).str.lower().eq("spreads")].copy()
        pick=None
        if not spg.empty:
            for ix,r in spg.iterrows():
                outcome=_norm_team_live(r.get("Outcome_Norm",r.get("Outcome","")))
                if outcome==home: pick=r.copy(); break
        if pick is None: pick=g.iloc[0].copy()
        pick["_rv22_game_id_key"]=gid; ctx.append(pick); keys.append(gid)
    if not ctx: return out
    cdf=pd.DataFrame(ctx).reset_index(drop=True)
    cdf,_bounce_live_diag=_attach_role_transition_live_context(cdf,dashboard_module=dashboard_module,log_func=lambda *a,**k:None)
    cdf,_deep_live_diag=_attach_deep_stat_context_to_miner(dashboard_module,cdf,log_func=lambda *a,**k:None,for_live=True)
    cdf,_situational_live_diag=_attach_h2h_rivalry_travel_context(dashboard_module,cdf,log_func=lambda *a,**k:None,for_live=True)
    cdf,_pt_live_diag=_pt_attach_current_live_frame(cdf,dashboard_module=dashboard_module)
    registry=report.get("system_miner_v3") or {}
    authority_votes_by_game={k:[] for k in keys}; research_votes_by_game={k:[] for k in keys}
    confirmed_research=0; authority_qualified=0; authority_evaluable=0; research_evaluable=0
    for market,mr in registry.items():
        atoms={a["name"]:np.asarray(a["mask"],dtype=bool) for a in _extended_atoms(cdf,dashboard_module,for_live=True,market=market)}
        for mech in (mr or {}).get("mechanism_families",[]):
            if not mech.get("confirmation_pass"): continue
            confirmed_research+=1
            live_eligible=_miner_live_authority_eligible(mech)
            if live_eligible: authority_qualified+=1
            cond=list(mech.get("representative_conditions") or [])
            if not cond or any(c not in atoms for c in cond): continue
            research_evaluable+=1
            if live_eligible: authority_evaluable+=1
            mask=np.ones(len(cdf),dtype=bool)
            for c in cond: mask &= atoms[c]
            direction=str(mech.get("direction") or "PLAY_ON").upper()
            for j in np.flatnonzero(mask):
                r=cdf.iloc[j]; home=_norm_team_live(r.get("Home_Team_Norm",r.get("Home_Team",""))); away=_norm_team_live(r.get("Away_Team_Norm",r.get("Away_Team","")))
                if str(market).lower()=="totals": target="over" if direction=="PLAY_ON" else "under"
                else: target=home if direction=="PLAY_ON" else away
                if not target: continue
                _uses_external=bool("EXTERNAL_RATINGS_FAMILY" in set(mech.get("families") or []))
                _uses_season_record=bool("SEASON_RECORD_STATE" in set(mech.get("families") or []))
                _uses_bounceback=bool("ROLE_TRANSITION_BOUNCEBACK" in set(mech.get("families") or []))
                _uses_h2h=bool("MATCHUP_HISTORY" in set(mech.get("families") or [])) or bool(mech.get("uses_h2h_context_family"))
                _uses_rivalry=bool("RIVALRY" in set(mech.get("families") or [])) or bool(mech.get("uses_rivalry_context_family"))
                _uses_travel=bool(set(mech.get("families") or []) & {"TRAVEL","ROAD_SEQUENCE"}) or bool(mech.get("uses_travel_context_family"))
                _uses_result_quality=bool("RESULT_QUALITY_REGRESSION" in set(mech.get("families") or []))
                _uses_resume=bool("SCHEDULE_RESUME_QUALITY" in set(mech.get("families") or []))
                _uses_recent=bool("RECENT_VS_SEASON" in set(mech.get("families") or []))
                _uses_matchup=bool("MATCHUP_DIFFERENTIAL" in set(mech.get("families") or []))
                _evidence_key=str(mech.get("evidence_family_key") or ("EXTERNAL_RATINGS_FAMILY" if _uses_external else "NCAAF_SEASON_RECORD_STATE_FAMILY" if _uses_season_record else "NCAAF_ROLE_TRANSITION_BOUNCEBACK_FAMILY" if _uses_bounceback else "NCAAF_H2H_CONTEXT_FAMILY" if _uses_h2h else "NCAAF_RIVALRY_CONTEXT_FAMILY" if _uses_rivalry else "NCAAF_TRAVEL_CONTEXT_FAMILY" if _uses_travel else "NCAAF_RESULT_QUALITY_FAMILY" if _uses_result_quality else "NCAAF_SCHEDULE_RESUME_FAMILY" if _uses_resume else "NCAAF_RECENT_VS_SEASON_FAMILY" if _uses_recent else "NCAAF_MATCHUP_DIFFERENTIAL_FAMILY" if _uses_matchup else str(mech.get("mechanism_id"))))
                vote={"source_type":"MINER","family_id":str(mech.get("mechanism_id")),"target":target,"market":str(market).lower(),
                      "evidence_family_key":_evidence_key,
                      "uses_external_ratings_family":_uses_external,"uses_season_record_state_family":_uses_season_record,"uses_role_transition_bounceback_family":_uses_bounceback,
                      "uses_h2h_context_family":_uses_h2h,"uses_rivalry_context_family":_uses_rivalry,"uses_travel_context_family":_uses_travel,
                      "uses_result_quality_family":_uses_result_quality,"uses_schedule_resume_family":_uses_resume,"uses_recent_vs_season_family":_uses_recent,"uses_matchup_differential_family":_uses_matchup,
                      "mechanisms":list(mech.get("families") or [str(mech.get("mechanism_id"))]),"rule":" AND ".join(cond),
                      "confirmation_n":mech.get("confirmation_n"),"confirmation_rate":mech.get("confirmation_rate"),
                      "evidence_level":"STRONG_VALIDATED" if live_eligible else "VALIDATED_SHADOW",
                      "live_authority_eligible":live_eligible,"live_authority_policy":NCAAF_MINER_LIVE_AUTHORITY_POLICY}
                research_votes_by_game[keys[j]].append(vote)
                if live_eligible: authority_votes_by_game[keys[j]].append(vote)
    # Production consumes ONLY authority-qualified votes. Collapse by independent
    # evidence family before Bet Authority sees the votes. In particular, every
    # Prediction Tracker mechanism shares EXTERNAL_RATINGS_FAMILY and can contribute
    # at most one vote. If qualified external mechanisms disagree on the target,
    # that correlated family abstains instead of being double-counted both ways.
    def _collapse_independent_authority_votes(votes):
        groups={}
        for v in votes:
            _key=(str(v.get("market") or ""),str(v.get("evidence_family_key") or v.get("family_id") or ""))
            groups.setdefault(_key,[]).append(v)
        kept=[]
        for _key,_vv in groups.items():
            _targets={str(v.get("target") or "") for v in _vv}
            if len(_targets)!=1:
                continue
            def _rank(v):
                try: _r=float(v.get("confirmation_rate") or 0)
                except Exception: _r=0.0
                try: _n=int(v.get("confirmation_n") or 0)
                except Exception: _n=0
                return (_r,_n)
            _best=max(_vv,key=_rank)
            _best=dict(_best)
            _best["correlated_family_candidates"]=len(_vv)
            _best["family_vote_collapsed"]=bool(len(_vv)>1)
            kept.append(_best)
        return kept

    for _gid,_votes in list(authority_votes_by_game.items()):
        authority_votes_by_game[_gid]=_collapse_independent_authority_votes(_votes)

    # The full confirmed set is preserved separately for research display/prospective tracking.
    out["NCAAF_RV22_Miner_Votes"]=[authority_votes_by_game.get(k,[]) for k in out[keycol]]
    out["NCAAF_RV22_Miner_Research_Votes"]=[research_votes_by_game.get(k,[]) for k in out[keycol]]
    out["NCAAF_Miner_Confirmed_Research"]=confirmed_research
    out["NCAAF_Miner_Qualified"]=authority_qualified
    out["NCAAF_Miner_Evaluable"]=authority_evaluable
    out["NCAAF_Miner_Research_Evaluable"]=research_evaluable
    out["NCAAF_Miner_Live_Authority_Policy"]=NCAAF_MINER_LIVE_AUTHORITY_POLICY
    # Expose current external metamodel context on every market row. The raw PT
    # margin fields themselves have no blanket authority and do not rewrite CORE.
    # Only a separately validated PT-derived Miner mechanism may contribute the
    # single collapsed EXTERNAL_RATINGS_FAMILY system vote.
    _pt_meta_by={keys[i]:cdf.iloc[i].get("_V210_PT_META_MARGIN_TEAM",np.nan) for i in range(len(cdf))}
    _pt_edge_by={keys[i]:cdf.iloc[i].get("_V210_PT_META_EDGE_POINTS",np.nan) for i in range(len(cdf))}
    _pt_cnt_by={keys[i]:cdf.iloc[i].get("_V210_PT_META_SYSTEM_COUNT",np.nan) for i in range(len(cdf))}
    _pt_ext_by={keys[i]:cdf.iloc[i].get("_V2183_PT_EXTERNAL_CONSENSUS_MARGIN_TEAM",np.nan) for i in range(len(cdf))}
    _pt_ext_edge_by={keys[i]:cdf.iloc[i].get("_V2183_PT_EXTERNAL_CONSENSUS_EDGE_POINTS",np.nan) for i in range(len(cdf))}
    _pt_ext_cluster_by={keys[i]:cdf.iloc[i].get("_V2183_PT_EXTERNAL_CONSENSUS_CLUSTER_COUNT",np.nan) for i in range(len(cdf))}
    _pt_ext_index_by={keys[i]:cdf.iloc[i].get("_V2183_PT_EXTERNAL_CONSENSUS_INDEX_COUNT",np.nan) for i in range(len(cdf))}
    _pt_ext_std_by={keys[i]:cdf.iloc[i].get("_V2183_PT_EXTERNAL_CONSENSUS_STD",np.nan) for i in range(len(cdf))}
    _pt_ext_iqr_by={keys[i]:cdf.iloc[i].get("_V2183_PT_EXTERNAL_CONSENSUS_IQR",np.nan) for i in range(len(cdf))}
    out["NCAAF_PT_Meta_Margin_Home"]=[_pt_meta_by.get(k,np.nan) for k in out[keycol]]
    out["NCAAF_PT_Meta_Edge_Home"]=[_pt_edge_by.get(k,np.nan) for k in out[keycol]]
    out["NCAAF_PT_Meta_System_Count"]=[_pt_cnt_by.get(k,np.nan) for k in out[keycol]]
    out["NCAAF_PT_Meta_Status"]=_pt_live_diag.get("status","UNAVAILABLE")
    out["NCAAF_PT_External_Consensus_Margin_Home"]=[_pt_ext_by.get(k,np.nan) for k in out[keycol]]
    out["NCAAF_PT_External_Consensus_Edge_Home"]=[_pt_ext_edge_by.get(k,np.nan) for k in out[keycol]]
    out["NCAAF_PT_External_Consensus_Cluster_Count"]=[_pt_ext_cluster_by.get(k,np.nan) for k in out[keycol]]
    out["NCAAF_PT_External_Consensus_Index_Count"]=[_pt_ext_index_by.get(k,np.nan) for k in out[keycol]]
    out["NCAAF_PT_External_Consensus_STD"]=[_pt_ext_std_by.get(k,np.nan) for k in out[keycol]]
    out["NCAAF_PT_External_Consensus_IQR"]=[_pt_ext_iqr_by.get(k,np.nan) for k in out[keycol]]
    out["NCAAF_PT_External_Consensus_Status"]=_pt_live_diag.get("status","UNAVAILABLE")
    out["NCAAF_PT_External_Consensus_Contract"]=PT_EXTERNAL_CONSENSUS_CONTRACT
    live_counts=[]; live_summaries=[]; research_counts=[]; research_summaries=[]
    for _,r in out.iterrows():
        m=str(r.get("Market") or "").lower()
        vv=[v for v in (r.get("NCAAF_RV22_Miner_Votes") or []) if v.get("market")==m]
        rv=[v for v in (r.get("NCAAF_RV22_Miner_Research_Votes") or []) if v.get("market")==m]
        live_counts.append(len(vv)); research_counts.append(len(rv))
        live_summaries.append(" | ".join(f"{v['family_id']} [STRONG]: {v.get('rule') or 'rule'} → {v['target']}" for v in vv) if vv else "—")
        research_summaries.append(" | ".join(f"{v['family_id']} [{'LIVE' if v.get('live_authority_eligible') else 'SHADOW'}]: {v.get('rule') or 'rule'} → {v['target']}" for v in rv) if rv else "—")
    out["NCAAF_Miner_Live_Trigger_Count"]=live_counts
    out["NCAAF_Miner_Research_Trigger_Count"]=research_counts
    out["NCAAF_RV2_System_Count"]=live_counts
    out["NCAAF_RV2_System_Summary"]=live_summaries
    out["NCAAF_RV2_Research_System_Summary"]=research_summaries
    return out



def _confirmation_pass_from_exact_rule(rep: dict[str,Any], market: str) -> bool:
    """Re-apply the frozen confirmation gate to an exact incumbent rule.

    The original multiple-testing/FDR admission is treated as frozen provenance.
    This helper only asks whether the exact rule still passes the historical
    discovery/confirmation evidence contract on the current <=2025 cache.
    """
    if not isinstance(rep,dict): return False
    if not bool((rep.get("split_audit") or {}).get("disjoint",False)): return False
    if str(market).lower()=="h2h":
        hp=rep.get("h2h_confirmation_price") or {}
        cy=[x for x in (rep.get("confirmation") or []) if int(x.get("season",0) or 0) in CONFIRMATION_SEASONS]
        both=bool(len(cy)==2 and all(int(x.get("priced_n",0) or 0)>=15 and np.isfinite(x.get("roi",np.nan)) and float(x.get("roi"))>0 and np.isfinite(x.get("market_residual",np.nan)) and float(x.get("market_residual"))>=0 for x in cy))
        return bool(int(rep.get("confirmation_n",0) or 0)>=30 and both and int(hp.get("priced_n",0) or 0)>=30 and np.isfinite(hp.get("roi",np.nan)) and float(hp.get("roi"))>0 and bool(hp.get("price_band_robust",False)) and np.isfinite(rep.get("market_residual_confirmation",np.nan)) and float(rep.get("market_residual_confirmation"))>=0)
    cy=[x for x in (rep.get("confirmation") or []) if int(x.get("season",0) or 0) in CONFIRMATION_SEASONS]
    graded=[x for x in cy if int(x.get("n",0) or 0)>=10]
    return bool(int(rep.get("confirmation_n",0) or 0)>=30 and np.isfinite(rep.get("confirmation_rate",np.nan)) and float(rep.get("confirmation_rate"))>=.50 and len(graded)==2 and all(np.isfinite(x.get("rate",np.nan)) and float(x.get("rate"))>=.50 for x in graded))


def _discovery_pass_from_exact_rule(rep: dict[str,Any], market: str) -> bool:
    if not isinstance(rep,dict): return False
    if str(market).lower()=="h2h":
        hp=rep.get("h2h_discovery_price") or {}
        return bool(float(rep.get("stable_discovery_fraction",0) or 0)>=1.0 and np.isfinite(rep.get("market_residual_discovery",np.nan)) and float(rep.get("market_residual_discovery"))>=.01 and int(hp.get("priced_n",0) or 0)>=80 and np.isfinite(hp.get("roi",np.nan)) and float(hp.get("roi"))>.01 and bool(hp.get("price_band_robust",False)) and np.isfinite(rep.get("remove_best_discovery_roi",np.nan)) and float(rep.get("remove_best_discovery_roi"))>=0 and np.isfinite(rep.get("remove_best_market_residual",np.nan)) and float(rep.get("remove_best_market_residual"))>=0)
    return bool(np.isfinite(rep.get("discovery_rate",np.nan)) and float(rep.get("discovery_rate"))>=.54 and float(rep.get("stable_discovery_fraction",0) or 0)>=1.0 and np.isfinite(rep.get("remove_best_discovery_rate",np.nan)) and float(rep.get("remove_best_discovery_rate"))>=.515)


def _legacy_contract_evidence_diagnostic(games: pd.DataFrame, seasons: np.ndarray, mask: np.ndarray,
                                         y: np.ndarray, valid: np.ndarray, baseline: np.ndarray,
                                         cond: list[str], fams: list[str], market: str,
                                         prior_direction: str) -> dict[str,Any]:
    """Evaluate an incumbent below the current admission floor without changing Miner gates.

    This is diagnostic-only. It separates a *contract/search eligibility* failure
    from a real historical-performance failure. A legacy incumbent may be frozen
    only when its exact rule is still computable and there is no clear evidence
    that the historical direction/performance has failed.
    """
    rep=_evaluate_rule(
        games,mask,y,valid,baseline,seasons,tuple(cond),tuple(fams),market,
        min_discovery_n_override=1,min_discovery_years_override=1,
        min_discovery_year_n_override=15,
    )
    if rep is None:
        return {"status":"INSUFFICIENT_FOR_DIAGNOSTIC","rep":None,"direction_status":"UNKNOWN",
                "discovery_performance_status":"INSUFFICIENT","confirmation_performance_status":"INSUFFICIENT"}
    market=str(market).lower()
    if market=="h2h":
        dp=rep.get("h2h_discovery_price") or {}
        _direction_sample_sufficient=int(dp.get("priced_n",0) or 0)>=30
        direction_status=("PASS" if str(rep.get("direction"))==str(prior_direction) else "FAIL") if _direction_sample_sufficient else "UNKNOWN"
        d_sample=bool(int(dp.get("priced_n",0) or 0)>=80 and np.isfinite(rep.get("remove_best_discovery_roi",np.nan)) and np.isfinite(rep.get("remove_best_market_residual",np.nan)))
        if not d_sample:
            disc_status="INSUFFICIENT"
        else:
            disc_status="PASS" if _discovery_pass_from_exact_rule(rep,market) else "FAIL"
        cp=rep.get("h2h_confirmation_price") or {}
        cy=[x for x in (rep.get("confirmation") or []) if int(x.get("season",0) or 0) in CONFIRMATION_SEASONS]
        c_sample=bool(len(cy)==2 and all(int(x.get("priced_n",0) or 0)>=15 for x in cy) and int(cp.get("priced_n",0) or 0)>=30)
        if not c_sample:
            conf_status="INSUFFICIENT"
        else:
            conf_status="PASS" if _confirmation_pass_from_exact_rule(rep,market) else "FAIL"
    else:
        _direction_sample_sufficient=int(rep.get("discovery_n",0) or 0)>=30
        direction_status=("PASS" if str(rep.get("direction"))==str(prior_direction) else "FAIL") if _direction_sample_sufficient else "UNKNOWN"
        dyears=list(rep.get("discovery_seasons") or [])
        d_sample=bool(len(dyears)>=2 and np.isfinite(rep.get("remove_best_discovery_rate",np.nan)))
        if not d_sample:
            disc_status="INSUFFICIENT"
        else:
            disc_status="PASS" if _discovery_pass_from_exact_rule(rep,market) else "FAIL"
        cy=[x for x in (rep.get("confirmation") or []) if int(x.get("season",0) or 0) in CONFIRMATION_SEASONS]
        c_sample=bool(int(rep.get("confirmation_n",0) or 0)>=30 and len(cy)==2 and all(int(x.get("n",0) or 0)>=10 for x in cy))
        if not c_sample:
            conf_status="INSUFFICIENT"
        else:
            conf_status="PASS" if _confirmation_pass_from_exact_rule(rep,market) else "FAIL"
    return {
        "status":"PASS","rep":rep,"direction_status":direction_status,
        "discovery_performance_status":disc_status,"confirmation_performance_status":conf_status,
        "discovery_n":rep.get("discovery_n"),"discovery_rate":rep.get("discovery_rate"),
        "confirmation_n":rep.get("confirmation_n"),"confirmation_rate":rep.get("confirmation_rate"),
        "split_audit":rep.get("split_audit") or {},
    }


def _minimal_mechanism_from_inventory_record(row: dict[str,Any], *, prior_live_signatures: set[str] | None=None) -> dict[str,Any]:
    """Reconstruct enough of an older library record to re-test its exact rule."""
    r=dict(row or {})
    rule=str(r.get("rule") or "")
    cond=list(r.get("conditions") or [])
    if not cond and rule:
        cond=[x.strip() for x in rule.split(" AND ") if x.strip()]
    tags=set(str(x) for x in (r.get("tags") or []))
    sig=str(r.get("signature") or "|".join([str(r.get("market") or ""),str(r.get("direction") or ""),"&".join(sorted(cond))]))
    prior_live=bool(r.get("live_authority_eligible")) or bool(prior_live_signatures and sig in prior_live_signatures)
    fams=[]
    if "SEASON_RECORD_STATE" in tags: fams.append("SEASON_RECORD_STATE")
    if "ROLE_TRANSITION_BOUNCEBACK" in tags: fams.append("ROLE_TRANSITION_BOUNCEBACK")
    if "PT_DERIVED" in tags: fams.append("EXTERNAL_RATINGS_FAMILY")
    if "H2H_CONTEXT" in tags: fams.append("MATCHUP_HISTORY")
    if "RIVALRY_CONTEXT" in tags: fams.append("RIVALRY")
    if "TRAVEL_CONTEXT" in tags: fams.append("TRAVEL_CONTEXT")
    if "RESULT_QUALITY_REGRESSION" in tags: fams.append("RESULT_QUALITY_REGRESSION")
    if "SCHEDULE_RESUME_QUALITY" in tags: fams.append("SCHEDULE_RESUME_QUALITY")
    if "RECENT_VS_SEASON" in tags: fams.append("RECENT_VS_SEASON")
    if "MATCHUP_DIFFERENTIAL" in tags: fams.append("MATCHUP_DIFFERENTIAL")
    return {
        "mechanism_id":r.get("mechanism_id") or _stable_id("NCAAF-INCUMBENT-",[sig]),
        "market":str(r.get("market") or "").lower(),"direction":str(r.get("direction") or "PLAY_ON"),
        "representative_conditions":cond,"families":fams,
        "uses_external_ratings_family":"PT_DERIVED" in tags,
        "uses_season_record_state_family":"SEASON_RECORD_STATE" in tags,
        "uses_role_transition_bounceback_family":"ROLE_TRANSITION_BOUNCEBACK" in tags,
        "uses_h2h_context_family":"H2H_CONTEXT" in tags,
        "uses_rivalry_context_family":"RIVALRY_CONTEXT" in tags,
        "uses_travel_context_family":"TRAVEL_CONTEXT" in tags,
        "uses_result_quality_family":"RESULT_QUALITY_REGRESSION" in tags,
        "uses_schedule_resume_family":"SCHEDULE_RESUME_QUALITY" in tags,
        "uses_recent_vs_season_family":"RECENT_VS_SEASON" in tags,
        "uses_matchup_differential_family":"MATCHUP_DIFFERENTIAL" in tags,
        "uses_big_al_observation_hypothesis":"BIG_AL_OBSERVATION_HYPOTHESIS" in tags,
        "evidence_family_key":r.get("evidence_family_key") or r.get("mechanism_id") or _stable_id("NCAAF-INCUMBENT-FAM-",[sig]),
        "confirmation_pass":bool(r.get("confirmation_pass",True)),"confirmation_n":int(r.get("confirmation_n",0) or 0),
        "confirmation_rate":r.get("confirmation_rate"),"live_authority_eligible":prior_live,"current_qualified":prior_live,
        "evidence_level":r.get("evidence_level"),"production_authority":0,
        "incumbent_recovered_from_inventory":True,
    }


def _collect_prior_confirmed_incumbents(previous_report: dict[str,Any] | None) -> list[dict[str,Any]]:
    """Collect the current prior library plus one-generation recovery records.

    V2.26 was the first run that exposed beam/search churn in the delta. Its
    `removed_confirmed` rows contain enough exact rule identity to recover and
    re-test the V2.24 incumbents instead of permanently losing them.
    """
    if not isinstance(previous_report,dict): return []
    out={}
    prior_live_sigs=set()
    d=previous_report.get("system_library_change_audit") or {}
    for x in (d.get("removed_live") or []):
        if x.get("signature"): prior_live_sigs.add(str(x.get("signature")))
    for market,mr in (previous_report.get("system_miner_v3") or {}).items():
        for mech in ((mr or {}).get("mechanism_families") or []):
            if not bool(mech.get("confirmation_pass")): continue
            mm=dict(mech); mm["market"]=str(mm.get("market") or market).lower(); mm["incumbent_source"]="PRIOR_CURRENT_REPORT"
            out[_ncaaf_system_signature(mm)]=mm
    inv=previous_report.get("system_library_inventory") or {}
    for row in (inv.get("records") or []):
        if not bool(row.get("confirmation_pass")): continue
        mm=_minimal_mechanism_from_inventory_record(row,prior_live_signatures=prior_live_sigs); mm["incumbent_source"]="PRIOR_INVENTORY"
        out.setdefault(_ncaaf_system_signature(mm),mm)
    # Recover rules that V2.26 reported as removed from V2.24. They are re-tested
    # below; nothing is restored merely because it once existed.
    for row in (d.get("removed_confirmed") or []):
        mm=_minimal_mechanism_from_inventory_record(row,prior_live_signatures=prior_live_sigs); mm["incumbent_source"]="RECOVERED_FROM_PRIOR_DELTA"
        out.setdefault(_ncaaf_system_signature(mm),mm)
    return list(out.values())


def _refresh_special_family_audit_after_incumbents(mr: dict[str,Any]) -> None:
    fams=list((mr or {}).get("mechanism_families") or [])
    sf=(mr or {}).setdefault("special_family_audit",{})
    defs=(
        ("season_record_state","SEASON_RECORD_STATE","NCAAF_SEASON_RECORD_STATE_FAMILY"),
        ("role_transition_bounceback","ROLE_TRANSITION_BOUNCEBACK","NCAAF_ROLE_TRANSITION_BOUNCEBACK_FAMILY"),
        ("h2h_context","MATCHUP_HISTORY","NCAAF_H2H_CONTEXT_FAMILY"),
        ("rivalry_context","RIVALRY","NCAAF_RIVALRY_CONTEXT_FAMILY"),
        ("travel_context","TRAVEL_CONTEXT","NCAAF_TRAVEL_CONTEXT_FAMILY"),
        ("result_quality_regression","RESULT_QUALITY_REGRESSION","NCAAF_RESULT_QUALITY_FAMILY"),
        ("schedule_resume_quality","SCHEDULE_RESUME_QUALITY","NCAAF_SCHEDULE_RESUME_FAMILY"),
        ("recent_vs_season","RECENT_VS_SEASON","NCAAF_RECENT_VS_SEASON_FAMILY"),
        ("matchup_differential","MATCHUP_DIFFERENTIAL","NCAAF_MATCHUP_DIFFERENTIAL_FAMILY"),
        ("external_ratings","EXTERNAL_RATINGS_FAMILY","EXTERNAL_RATINGS_FAMILY"),
    )
    for key,family_name,evidence_key in defs:
        z=sf.setdefault(key,{})
        mechs=[m for m in fams if family_name in set(m.get("families") or []) or (family_name=="TRAVEL_CONTEXT" and bool(set(m.get("families") or []) & {"TRAVEL_CONTEXT","ROAD_SEQUENCE"})) or (family_name=="EXTERNAL_RATINGS_FAMILY" and bool(m.get("uses_external_ratings_family")))]
        conf=[m for m in mechs if bool(m.get("confirmation_pass"))]; live=[m for m in mechs if bool(m.get("live_authority_eligible"))]
        z.update({"family":family_name,"evidence_family_key":evidence_key,"mechanism_families":len(mechs),"confirmed_mechanisms":len(conf),"live_authority_mechanisms":len(live),"independent_live_family_votes":1 if live and family_name in {"SEASON_RECORD_STATE","ROLE_TRANSITION_BOUNCEBACK","MATCHUP_HISTORY","RIVALRY","TRAVEL_CONTEXT","RESULT_QUALITY_REGRESSION","SCHEDULE_RESUME_QUALITY","RECENT_VS_SEASON","MATCHUP_DIFFERENTIAL","EXTERNAL_RATINGS_FAMILY"} else len(set((str(m.get("market")),str(m.get("evidence_family_key") or m.get("mechanism_id"))) for m in live)),"confirmed_rules":[{"mechanism_id":m.get("mechanism_id"),"market":m.get("market"),"direction":m.get("direction"),"confirmation_n":m.get("confirmation_n"),"confirmation_rate":m.get("confirmation_rate"),"live_authority_eligible":bool(m.get("live_authority_eligible")),"rule":" AND ".join(m.get("representative_conditions") or [])} for m in conf]})
    z=sf.setdefault("big_al_observation_hypothesis",{})
    mechs=[m for m in fams if bool(m.get("uses_big_al_observation_hypothesis"))]
    conf=[m for m in mechs if bool(m.get("confirmation_pass"))]; live=[m for m in mechs if bool(m.get("live_authority_eligible"))]
    z.update({"label":"BIG_AL_OBSERVATION_HYPOTHESIS","mechanism_families":len(mechs),"confirmed_mechanisms":len(conf),"live_authority_mechanisms":len(live),"independent_live_family_votes":len(set((str(m.get("market")),str(m.get("evidence_family_key") or m.get("mechanism_id"))) for m in live)),"confirmed_rules":[{"mechanism_id":m.get("mechanism_id"),"market":m.get("market"),"direction":m.get("direction"),"confirmation_n":m.get("confirmation_n"),"confirmation_rate":m.get("confirmation_rate"),"live_authority_eligible":bool(m.get("live_authority_eligible")),"rule":" AND ".join(m.get("representative_conditions") or [])} for m in conf]})


def _special_atom_availability_audit(games: pd.DataFrame, seasons: np.ndarray, market: str, dashboard_module=None,
                                     admitted_atoms: list[dict[str,Any]] | None=None) -> dict[str,Any]:
    """Explain why State/Bounceback produced zero tested hypotheses.

    `source_available` is computed without the atom support floor. The evaluator
    still keeps its frozen discovery minimum of 100 rows; this function does not
    loosen any qualification gate.
    """
    market=str(market).lower()
    admitted=list(admitted_atoms) if admitted_atoms is not None else _extended_atoms(games,dashboard_module,for_live=False,market=market,enforce_support_floor=True)
    raw=_extended_atoms(games,dashboard_module,for_live=False,market=market,enforce_support_floor=False)
    amap={str(a.get("name")):a for a in admitted}; rmap={str(a.get("name")):a for a in raw}
    expected={
        "SEASON_RECORD_STATE":["SU_WINLESS_PRIOR","ATS_COVERLESS_PRIOR","SU_AND_ATS_WINLESS_PRIOR","SU_WINLESS_AFTER_2_PLUS","ATS_COVERLESS_AFTER_2_PLUS","SU_AND_ATS_WINLESS_AFTER_2_PLUS","SU_WINLESS_AFTER_3_PLUS","ATS_COVERLESS_AFTER_3_PLUS","SU_AND_ATS_WINLESS_AFTER_3_PLUS","SU_WINLESS_AFTER_4_PLUS","ATS_COVERLESS_AFTER_4_PLUS","SU_AND_ATS_WINLESS_AFTER_4_PLUS","OPP_SU_WINLESS_PRIOR","OPP_ATS_COVERLESS_PRIOR","OPP_SU_AND_ATS_WINLESS_PRIOR","OPP_SU_WINLESS_AFTER_2_PLUS","OPP_ATS_COVERLESS_AFTER_2_PLUS","OPP_SU_AND_ATS_WINLESS_AFTER_2_PLUS","OPP_SU_WINLESS_AFTER_3_PLUS","OPP_ATS_COVERLESS_AFTER_3_PLUS","OPP_SU_AND_ATS_WINLESS_AFTER_3_PLUS","OPP_SU_WINLESS_AFTER_4_PLUS","OPP_ATS_COVERLESS_AFTER_4_PLUS","OPP_SU_AND_ATS_WINLESS_AFTER_4_PLUS"],
        "ROLE_TRANSITION_BOUNCEBACK":["BOUNCEBACK_HOME_FAV_UPSET_TO_ROAD_DOG","BOUNCEBACK_WINNING_TEAM","BOUNCEBACK_BOTH_WINNING_TEAMS","BOUNCEBACK_WINNING_TEAM_DOG_5_TO_7P5","OPP_BOUNCEBACK_HOME_FAV_UPSET_TO_ROAD_DOG","OPP_BOUNCEBACK_WINNING_TEAM","OPP_BOUNCEBACK_BOTH_WINNING_TEAMS","OPP_BOUNCEBACK_WINNING_TEAM_DOG_5_TO_7P5"],
        "RESULT_QUALITY_REGRESSION":["TEAM_OFF_WIN_NEG_YARDS","TEAM_OFF_WIN_NEG_YPP","TEAM_OFF_DECEPTIVE_WIN_2PLUS","TEAM_OFF_RESULT_OVERPERFORMANCE","TEAM_OFF_LOSS_POS_YPP","TEAM_OFF_RESULT_UNDERPERFORMANCE","OPP_OFF_WIN_NEG_YARDS","OPP_OFF_WIN_NEG_YPP","OPP_OFF_DECEPTIVE_WIN_2PLUS","OPP_OFF_RESULT_OVERPERFORMANCE","OPP_OFF_LOSS_POS_YPP","OPP_OFF_RESULT_UNDERPERFORMANCE"],
        "SCHEDULE_RESUME_QUALITY":["TEAM_SOS_ADV_10P","OPP_SOS_ADV_10P","TEAM_STRONG_RECORD_WEAK_SOS","OPP_STRONG_RECORD_WEAK_SOS","TEAM_STRONG_RECORD_WEAK_UNDERLYING","OPP_STRONG_RECORD_WEAK_UNDERLYING","TEAM_OPPADJ_NETYPP_ADV_075","OPP_OPPADJ_NETYPP_ADV_075"],
        "RECENT_VS_SEASON":["TEAM_RECENT3_OFF_YPP_UP_050","TEAM_RECENT3_NET_YPP_UP_075","TEAM_RECENT3_DEF_YPP_IMPROVED_050","TEAM_RECENT3_AND5_NET_YPP_UP","OPP_RECENT3_OFF_YPP_UP_050","OPP_RECENT3_NET_YPP_UP_075","OPP_RECENT3_DEF_YPP_IMPROVED_050","OPP_RECENT3_AND5_NET_YPP_UP"],
        "MATCHUP_DIFFERENTIAL":["TEAM_RUSH_MATCHUP_EDGE_075PLUS","TEAM_PASS_MATCHUP_EDGE_100PLUS","TEAM_YPP_MATCHUP_EDGE_050PLUS","TEAM_MULTI_MATCHUP_EDGE_2PLUS","TEAM_MATCHUP_EDGE_OVER_OPP_2PLUS","OPP_RUSH_MATCHUP_EDGE_075PLUS","OPP_PASS_MATCHUP_EDGE_100PLUS","OPP_YPP_MATCHUP_EDGE_050PLUS","OPP_MULTI_MATCHUP_EDGE_2PLUS","OPP_MATCHUP_EDGE_OVER_TEAM_2PLUS"],
    }
    out={}
    disc=np.isfinite(seasons)&(seasons<=DISCOVERY_MAX_SEASON)
    _y,_valid,_baseline=_market_target(games,market)
    for fam,names in expected.items():
        rows=[]
        for nm in names:
            a=rmap.get(nm); mm=np.asarray(a.get("mask"),dtype=bool) if a is not None else np.zeros(len(games),dtype=bool)
            dn=int(np.sum(mm&disc)); total_n=int(np.sum(mm)); y24=int(np.sum(mm&(seasons==2024))); y25=int(np.sum(mm&(seasons==2025)))
            _dd=mm&disc&_valid
            _raw=float(np.mean(_y[_dd])) if _dd.any() else np.nan
            _direction="PLAY_ON" if np.isfinite(_raw) and _raw>=.5 else "FADE"
            _obs=_y if _direction=="PLAY_ON" else 1-_y
            def _diag_year(_sy):
                _jj=mm&_valid&np.isfinite(seasons)&(seasons==float(_sy))
                return int(_jj.sum()),(float(np.mean(_obs[_jj])) if _jj.any() else np.nan)
            _n24,_r24=_diag_year(2024); _n25,_r25=_diag_year(2025)
            _cc=mm&_valid&np.isfinite(seasons)&np.isin(seasons,np.asarray(CONFIRMATION_SEASONS,dtype=float))
            rows.append({"atom":nm,"source_available":a is not None,"admitted_to_catalog":nm in amap,"total_n":total_n,"discovery_n":dn,"confirmation_2024_n":y24,"confirmation_2025_n":y25,"evaluator_min_discovery_n":100,"evaluator_eligible":bool(a is not None and dn>=100),
                         "diagnostic_only":True,"diagnostic_direction":_direction,"diagnostic_discovery_graded_n":int(_dd.sum()),"diagnostic_discovery_rate":float(np.mean(_obs[_dd])) if _dd.any() else np.nan,
                         "diagnostic_2024_graded_n":_n24,"diagnostic_2024_rate":_r24,"diagnostic_2025_graded_n":_n25,"diagnostic_2025_rate":_r25,
                         "diagnostic_confirmation_graded_n":int(_cc.sum()),"diagnostic_confirmation_rate":float(np.mean(_obs[_cc])) if _cc.any() else np.nan,"diagnostic_authority":0})
        out[fam]={"expected_atoms":len(names),"source_available_atoms":sum(x["source_available"] for x in rows),"admitted_atoms":sum(x["admitted_to_catalog"] for x in rows),"evaluator_eligible_atoms":sum(x["evaluator_eligible"] for x in rows),"below_evaluator_min_atoms":sum(x["source_available"] and not x["evaluator_eligible"] for x in rows),"source_unavailable_atoms":sum(not x["source_available"] for x in rows),"diagnostic_only_backtests":sum(x["source_available"] and x["diagnostic_discovery_graded_n"]>0 for x in rows),"atoms":rows}
    return out


def _reconcile_incumbent_system_library(games: pd.DataFrame, seasons: np.ndarray, miners: dict[str,Any], previous_report: dict[str,Any] | None, dashboard_module=None, log_func=print) -> tuple[dict[str,Any],dict[str,Any]]:
    """Re-test prior confirmed exact rules independently of the current search beam.

    V2.29 distinguishes four incumbent states:
      * REVALIDATED_RETAINED: exact rule clears the current frozen evidence gate;
      * LEGACY_CONTRACT_HOLD: exact rule is computable but the current admission/
        evaluator contract no longer admits it; prior confirmed/live status is
        frozen unless current historical evidence clearly fails;
      * HOLD_NOT_EVALUABLE: source atom/input is unavailable; research identity is
        retained but live authority fails closed;
      * DEMOTED: exact historical evidence actually fails or direction reverses.
    """
    if not isinstance(previous_report,dict):
        return miners,{"status":"NO_PRIOR_REPORT","prior_confirmed":0,"revalidated_retained":0,"carried_hold":0,"carried_contract_hold":0,"demoted":0,"rediscovered":0,"restored_missing":0,"production_authority":0,"selection_uses_2026":False}
    prior=_collect_prior_confirmed_incumbents(previous_report)
    _prior_current_count=int(((previous_report.get("system_library_inventory") or {}).get("confirmed_mechanisms") or 0))
    _recovery_candidates=sum(str(x.get("incumbent_source") or "")=="RECOVERED_FROM_PRIOR_DELTA" for x in prior)
    _prior_sigs={_ncaaf_system_signature(x) for x in prior}
    _current_confirmed_sigs={_ncaaf_system_signature(m) for mr in (miners or {}).values() for m in ((mr or {}).get("mechanism_families") or []) if bool(m.get("confirmation_pass"))}
    audit={"status":"PASS","policy":"PERSISTENT_INCUMBENT_EXACT_RULE_REVALIDATION_V2_29_CONTRACT_HOLD","prior_confirmed":len(prior),"prior_current_confirmed":_prior_current_count,"recovery_candidates_from_prior_delta":int(_recovery_candidates),"new_challenger_confirmed":len(_current_confirmed_sigs-_prior_sigs),"revalidated_retained":0,"carried_hold":0,"carried_contract_hold":0,"demoted":0,"rediscovered":0,"restored_missing":0,"details":[],"production_authority":0,"selection_uses_2026":False}
    by_market={}
    for inc in prior: by_market.setdefault(str(inc.get("market") or "").lower(),[]).append(inc)
    for market,incumbents in by_market.items():
        if market not in miners: continue
        mr=miners[market]
        cur=list((mr or {}).get("mechanism_families") or [])
        cur_by_sig={_ncaaf_system_signature(x):x for x in cur}
        atoms=_extended_atoms(games,dashboard_module,for_live=False,market=market,enforce_support_floor=False)
        atom_map={str(a.get("name")):a for a in atoms}
        y,valid,baseline=_market_target(games,market)
        for old in incumbents:
            sig=_ncaaf_system_signature(old); cond=list(old.get("representative_conditions") or [])
            existing=cur_by_sig.get(sig); missing=[c for c in cond if c not in atom_map]
            if missing:
                hold=dict(existing or old)
                hold.update({"market":market,"representative_conditions":cond,"confirmation_pass":True,"current_qualified":False,"live_authority_eligible":False,"library_origin":"INCUMBENT_HOLD_NOT_EVALUABLE","incumbent_revalidation_status":"HOLD_NOT_EVALUABLE","incumbent_missing_atoms":missing,"incumbent_prior_live_authority_eligible":bool(old.get("live_authority_eligible")),"production_authority":0})
                if existing is None:
                    cur.append(hold); cur_by_sig[sig]=hold; audit["restored_missing"]+=1
                else:
                    existing.update(hold)
                audit["carried_hold"]+=1; audit["details"].append({"signature":sig,"market":market,"mechanism_id":hold.get("mechanism_id"),"state":"HOLD_NOT_EVALUABLE","missing_atoms":missing,"prior_live":bool(old.get("live_authority_eligible"))})
                continue
            mask=np.ones(len(games),dtype=bool); fams=[]
            for c in cond:
                a=atom_map[c]; mask &= np.asarray(a.get("mask"),dtype=bool); fams.append(str(a.get("family") or ""))
            rep=_evaluate_rule(games,mask,y,valid,baseline,seasons,tuple(cond),tuple(fams),market)
            if rep is None:
                diag=_legacy_contract_evidence_diagnostic(games,seasons,mask,y,valid,baseline,cond,fams,market,str(old.get("direction")))
                # Contract/search ineligibility is not itself a performance demotion.
                # Demote only on a clear direction reversal or a fully measurable
                # historical performance failure under the frozen evidence tests.
                fail_reason=None
                if diag.get("direction_status")=="FAIL":
                    fail_reason="DIRECTION_CHANGED"
                elif diag.get("discovery_performance_status")=="FAIL" or diag.get("confirmation_performance_status")=="FAIL":
                    fail_reason="EXACT_HISTORICAL_REVALIDATION_FAILED"
                if fail_reason:
                    audit["demoted"]+=1; audit["details"].append({"signature":sig,"market":market,"mechanism_id":old.get("mechanism_id"),"state":"DEMOTED","reason":fail_reason,"discovery_status":diag.get("discovery_performance_status"),"confirmation_status":diag.get("confirmation_performance_status"),"diagnostic_discovery_n":diag.get("discovery_n"),"diagnostic_discovery_rate":diag.get("discovery_rate"),"diagnostic_confirmation_n":diag.get("confirmation_n"),"diagnostic_confirmation_rate":diag.get("confirmation_rate")})
                    if existing is not None:
                        cur.remove(existing); cur_by_sig.pop(sig,None)
                    continue
                prior_live=bool(old.get("live_authority_eligible"))
                hold=dict(existing or old)
                hold.update({
                    "market":market,"representative_conditions":cond,"families":list(dict.fromkeys(fams)),
                    "confirmation_pass":True,"current_qualified":prior_live,"live_authority_eligible":prior_live,
                    "authority_state":"CONFIRMED_FROZEN_INCUMBENT","production_authority":0,
                    "library_origin":"INCUMBENT_LEGACY_CONTRACT_HOLD","incumbent_revalidation_status":"LEGACY_CONTRACT_HOLD",
                    "incumbent_prior_live_authority_eligible":prior_live,"incumbent_missing_atoms":[],
                    "original_search_multiple_testing_status":"FROZEN_PRIOR_PASS",
                    "legacy_contract_hold_reason":"CURRENT_EVALUATOR_OR_ADMISSION_CONTRACT_NO_LONGER_ADMITS_EXACT_RULE",
                    "legacy_contract_diagnostic":{k:v for k,v in diag.items() if k!="rep"},
                })
                if existing is None:
                    cur.append(hold); cur_by_sig[sig]=hold; audit["restored_missing"]+=1
                else:
                    existing.update(hold)
                audit["carried_contract_hold"]+=1
                audit["details"].append({"signature":sig,"market":market,"mechanism_id":hold.get("mechanism_id"),"state":"LEGACY_CONTRACT_HOLD","prior_live":prior_live,"live":prior_live,"reason":hold.get("legacy_contract_hold_reason"),"diagnostic_discovery_n":diag.get("discovery_n"),"diagnostic_discovery_rate":diag.get("discovery_rate"),"diagnostic_confirmation_n":diag.get("confirmation_n"),"diagnostic_confirmation_rate":diag.get("confirmation_rate"),"discovery_status":diag.get("discovery_performance_status"),"confirmation_status":diag.get("confirmation_performance_status")})
                continue
            if str(rep.get("direction"))!=str(old.get("direction")):
                audit["demoted"]+=1; audit["details"].append({"signature":sig,"market":market,"mechanism_id":old.get("mechanism_id"),"state":"DEMOTED","reason":"DIRECTION_CHANGED","old_direction":old.get("direction"),"new_direction":rep.get("direction")})
                if existing is not None:
                    cur.remove(existing); cur_by_sig.pop(sig,None)
                continue
            disc_ok=_discovery_pass_from_exact_rule(rep,market); conf_ok=_confirmation_pass_from_exact_rule(rep,market)
            if not (disc_ok and conf_ok):
                audit["demoted"]+=1; audit["details"].append({"signature":sig,"market":market,"mechanism_id":old.get("mechanism_id"),"state":"DEMOTED","reason":"EXACT_HISTORICAL_REVALIDATION_FAILED","discovery_pass":disc_ok,"confirmation_pass":conf_ok,"discovery_n":rep.get("discovery_n"),"discovery_rate":rep.get("discovery_rate"),"confirmation_n":rep.get("confirmation_n"),"confirmation_rate":rep.get("confirmation_rate")})
                if existing is not None:
                    cur.remove(existing); cur_by_sig.pop(sig,None)
                continue
            target=existing if existing is not None else dict(old)
            if existing is not None: audit["rediscovered"]+=1
            else: audit["restored_missing"]+=1
            target.update({"market":market,"direction":rep.get("direction"),"representative_conditions":cond,"families":list(dict.fromkeys(fams)),"discovery_rate":rep.get("discovery_rate"),"discovery_n":rep.get("discovery_n"),"confirmation_rate":rep.get("confirmation_rate"),"confirmation_n":rep.get("confirmation_n"),"confirmation_pass":True,"split_audit":rep.get("split_audit") or {},"authority_state":"CONFIRMED_SHADOW","production_authority":0,"library_origin":"INCUMBENT_REVALIDATED","incumbent_revalidation_status":"PASS","incumbent_prior_live_authority_eligible":bool(old.get("live_authority_eligible")),"original_search_multiple_testing_status":"FROZEN_PRIOR_PASS"})
            target["uses_external_ratings_family"]=bool(old.get("uses_external_ratings_family")) or "EXTERNAL_RATINGS_FAMILY" in set(fams)
            target["uses_season_record_state_family"]=bool(old.get("uses_season_record_state_family")) or "SEASON_RECORD_STATE" in set(fams)
            target["uses_role_transition_bounceback_family"]=bool(old.get("uses_role_transition_bounceback_family")) or "ROLE_TRANSITION_BOUNCEBACK" in set(fams)
            target["uses_h2h_context_family"]=bool(old.get("uses_h2h_context_family")) or "MATCHUP_HISTORY" in set(fams)
            target["uses_rivalry_context_family"]=bool(old.get("uses_rivalry_context_family")) or "RIVALRY" in set(fams)
            target["uses_travel_context_family"]=bool(old.get("uses_travel_context_family")) or bool(set(fams)&{"TRAVEL_CONTEXT","ROAD_SEQUENCE"})
            target["uses_result_quality_family"]=bool(old.get("uses_result_quality_family")) or "RESULT_QUALITY_REGRESSION" in set(fams)
            target["uses_schedule_resume_family"]=bool(old.get("uses_schedule_resume_family")) or "SCHEDULE_RESUME_QUALITY" in set(fams)
            target["uses_recent_vs_season_family"]=bool(old.get("uses_recent_vs_season_family")) or "RECENT_VS_SEASON" in set(fams)
            target["uses_matchup_differential_family"]=bool(old.get("uses_matchup_differential_family")) or "MATCHUP_DIFFERENTIAL" in set(fams)
            target["uses_big_al_observation_hypothesis"]=bool(old.get("uses_big_al_observation_hypothesis")) or bool(set(fams)&BIG_AL_OBSERVATION_ATOM_FAMILIES)
            target["evidence_family_key"]=old.get("evidence_family_key") or target.get("evidence_family_key") or target.get("mechanism_id")
            target["current_qualified"]=_miner_live_authority_eligible(target); target["live_authority_eligible"]=target["current_qualified"]; target["live_authority_policy"]=NCAAF_MINER_LIVE_AUTHORITY_POLICY; target["evidence_level"]=_evidence_tier(target,market)
            rep2=dict(rep); rep2["_mask_internal"]=mask
            try: target["attribution"]=_mechanism_attribution(games,seasons,rep2,market)
            except Exception: pass
            if existing is None:
                cur.append(target); cur_by_sig[sig]=target
            audit["revalidated_retained"]+=1; audit["details"].append({"signature":sig,"market":market,"mechanism_id":target.get("mechanism_id"),"state":"REVALIDATED_RETAINED","rediscovered":existing is not None,"confirmation_n":target.get("confirmation_n"),"confirmation_rate":target.get("confirmation_rate"),"live":bool(target.get("live_authority_eligible")),"incumbent_source":old.get("incumbent_source")})
        mr["mechanism_families"]=cur; mr["mechanism_family_count"]=len(cur); mr["confirmed_mechanism_count"]=sum(bool(x.get("confirmation_pass")) for x in cur); mr["incumbent_revalidation_applied"]=True
        _refresh_special_family_audit_after_incumbents(mr)
    log_func(f"[NCAAF-INCUMBENT-LIBRARY] status={audit.get('status')} policy={audit.get('policy')} prior_current_confirmed={audit.get('prior_current_confirmed')} recovery_candidates={audit.get('recovery_candidates_from_prior_delta')} incumbent_candidates={audit.get('prior_confirmed')} new_challenger_confirmed={audit.get('new_challenger_confirmed')} revalidated_retained={audit.get('revalidated_retained')} rediscovered={audit.get('rediscovered')} restored_missing={audit.get('restored_missing')} source_hold={audit.get('carried_hold')} contract_hold={audit.get('carried_contract_hold')} demoted={audit.get('demoted')} 2026_selection=FALSE authority=0")
    for x in audit.get("details",[]):
        log_func(f"[NCAAF-INCUMBENT-SYSTEM] state={x.get('state')} market={x.get('market')} id={x.get('mechanism_id')} live={x.get('live')} confirmation={x.get('confirmation_rate')}/{x.get('confirmation_n')} reason={x.get('reason')} discovery_status={x.get('discovery_status')} confirmation_status={x.get('confirmation_status')} missing={','.join(x.get('missing_atoms') or [])}")
    return miners,audit

def _market_rich_audit(games: pd.DataFrame, utils_module=None) -> dict[str,Any]:
    wanted=["Line_Move_30m","Line_Move_60m","Line_Move_120m","Line_Move_From_Open","Direction_Changes_Count","Sharp_Book_Move_60m",
            "Sharp_Soft_Divergence","Sharp_Consensus_Direction","Current_vs_Best_Line","Current_vs_Worst_Line","Key_Cross_Persistence"]
    present=[c for c in wanted if c in games.columns and _num(games,c).notna().sum()>0]
    return {"backend":"UTILS","raw_current_market":"sharp_moves_master","enriched_market_view":"moves_with_features_merged",
            "historical_fields_requested":wanted,"historical_fields_present":present,"historical_field_coverage":len(present)/len(wanted),
            "utils_recent_market_reader":bool(utils_module is not None and hasattr(utils_module,"read_recent_sharp_moves")),
            "production_authority":0}


def _report_without_models(bundle: dict[str,Any]) -> dict[str,Any]:
    return _json_safe({k:v for k,v in bundle.items() if k!="_internal"})



def _apply_v231_context_family_caps(miners: dict[str,Any], *, log_func=print) -> dict[str,Any]:
    """Normalize correlated special-context mechanisms to one family vote each.

    Qualification evidence and mechanism identity are untouched; only the live
    evidence-family key is normalized so variants of the same objective concept
    cannot manufacture multiple independent confirmations.
    """
    changed=0; counts={"h2h":0,"rivalry":0,"travel":0,"result_quality":0,"schedule_resume":0,"recent_vs_season":0,"matchup_differential":0}
    for _market,_mr in (miners or {}).items():
        for m in ((_mr or {}).get("mechanism_families") or []):
            fams=set(str(x) for x in (m.get("families") or []))
            if "EXTERNAL_RATINGS_FAMILY" in fams or bool(m.get("uses_external_ratings_family")):
                continue
            if "SEASON_RECORD_STATE" in fams or bool(m.get("uses_season_record_state_family")):
                continue
            if "ROLE_TRANSITION_BOUNCEBACK" in fams or bool(m.get("uses_role_transition_bounceback_family")):
                continue
            old=str(m.get("evidence_family_key") or m.get("mechanism_id") or "")
            new=old
            if "MATCHUP_HISTORY" in fams:
                m["uses_h2h_context_family"]=True; new="NCAAF_H2H_CONTEXT_FAMILY"; counts["h2h"]+=1
            elif "RIVALRY" in fams:
                m["uses_rivalry_context_family"]=True; new="NCAAF_RIVALRY_CONTEXT_FAMILY"; counts["rivalry"]+=1
            elif fams & {"TRAVEL_CONTEXT","ROAD_SEQUENCE"}:
                m["uses_travel_context_family"]=True; new="NCAAF_TRAVEL_CONTEXT_FAMILY"; counts["travel"]+=1
            elif "RESULT_QUALITY_REGRESSION" in fams:
                m["uses_result_quality_family"]=True; new="NCAAF_RESULT_QUALITY_FAMILY"; counts["result_quality"]+=1
            elif "SCHEDULE_RESUME_QUALITY" in fams:
                m["uses_schedule_resume_family"]=True; new="NCAAF_SCHEDULE_RESUME_FAMILY"; counts["schedule_resume"]+=1
            elif "RECENT_VS_SEASON" in fams:
                m["uses_recent_vs_season_family"]=True; new="NCAAF_RECENT_VS_SEASON_FAMILY"; counts["recent_vs_season"]+=1
            elif "MATCHUP_DIFFERENTIAL" in fams:
                m["uses_matchup_differential_family"]=True; new="NCAAF_MATCHUP_DIFFERENTIAL_FAMILY"; counts["matchup_differential"]+=1
            if new!=old:
                m["evidence_family_key"]=new; changed+=1
    log_func(f"[NCAAF-RV2311-FAMILY-CAPS] status=PASS normalized={changed} h2h={counts['h2h']} rivalry={counts['rivalry']} travel={counts['travel']} result_quality={counts['result_quality']} schedule_resume={counts['schedule_resume']} recent_vs_season={counts['recent_vs_season']} matchup_differential={counts['matchup_differential']} family_vote_cap=1 qualification_mutation=FALSE authority=0")
    return {"status":"PASS","normalized":changed,**counts,"family_vote_cap":1,"qualification_mutation":False}


def _ncaaf_system_signature(mech: dict[str,Any]) -> str:
    cond=sorted(str(x) for x in (mech.get("representative_conditions") or []))
    return "|".join([str(mech.get("market") or ""),str(mech.get("direction") or ""),"&".join(cond)])


def _ncaaf_system_tags(mech: dict[str,Any]) -> list[str]:
    tags=[]
    fams=set(str(x) for x in (mech.get("families") or []))
    if bool(mech.get("uses_season_record_state_family")) or "SEASON_RECORD_STATE" in fams: tags.append("SEASON_RECORD_STATE")
    if bool(mech.get("uses_role_transition_bounceback_family")) or "ROLE_TRANSITION_BOUNCEBACK" in fams: tags.append("ROLE_TRANSITION_BOUNCEBACK")
    if bool(mech.get("uses_external_ratings_family")) or "EXTERNAL_RATINGS_FAMILY" in fams: tags.append("PT_DERIVED")
    if bool(mech.get("uses_h2h_context_family")) or "MATCHUP_HISTORY" in fams: tags.append("H2H_CONTEXT")
    if bool(mech.get("uses_rivalry_context_family")) or "RIVALRY" in fams: tags.append("RIVALRY_CONTEXT")
    if bool(mech.get("uses_travel_context_family")) or bool(fams & {"TRAVEL_CONTEXT","ROAD_SEQUENCE"}): tags.append("TRAVEL_CONTEXT")
    if bool(mech.get("uses_result_quality_family")) or "RESULT_QUALITY_REGRESSION" in fams: tags.append("RESULT_QUALITY_REGRESSION")
    if bool(mech.get("uses_schedule_resume_family")) or "SCHEDULE_RESUME_QUALITY" in fams: tags.append("SCHEDULE_RESUME_QUALITY")
    if bool(mech.get("uses_recent_vs_season_family")) or "RECENT_VS_SEASON" in fams: tags.append("RECENT_VS_SEASON")
    if bool(mech.get("uses_matchup_differential_family")) or "MATCHUP_DIFFERENTIAL" in fams: tags.append("MATCHUP_DIFFERENTIAL")
    if bool(mech.get("uses_big_al_observation_hypothesis")) or bool(fams & BIG_AL_OBSERVATION_ATOM_FAMILIES): tags.append("BIG_AL_OBSERVATION_HYPOTHESIS")
    if not tags: tags.append("STANDARD_MINER")
    return tags


def _ncaaf_system_library_inventory(miners: dict[str,Any]) -> dict[str,Any]:
    rows=[]
    for market,mr in (miners or {}).items():
        for mech in ((mr or {}).get("mechanism_families") or []):
            live=bool(mech.get("live_authority_eligible")) or _miner_live_authority_eligible(mech)
            row={
                "signature":_ncaaf_system_signature(mech),
                "mechanism_id":mech.get("mechanism_id"),
                "market":market,
                "direction":mech.get("direction"),
                "rule":" AND ".join(mech.get("representative_conditions") or []),
                "conditions":list(mech.get("representative_conditions") or []),
                "tags":_ncaaf_system_tags(mech),
                "evidence_family_key":str(mech.get("evidence_family_key") or mech.get("mechanism_id") or ""),
                "confirmation_pass":bool(mech.get("confirmation_pass")),
                "confirmation_n":int(mech.get("confirmation_n",0) or 0),
                "confirmation_rate":mech.get("confirmation_rate"),
                "live_authority_eligible":live,
                "evidence_level":mech.get("evidence_level"),
                "library_origin":mech.get("library_origin") or "CURRENT_CHALLENGER_SEARCH",
                "incumbent_revalidation_status":mech.get("incumbent_revalidation_status"),
                "incumbent_missing_atoms":list(mech.get("incumbent_missing_atoms") or []),
            }
            rows.append(row)
    confirmed=[x for x in rows if x["confirmation_pass"]]
    live=[x for x in rows if x["live_authority_eligible"]]
    independent_live=sorted(set((x["market"],x["evidence_family_key"]) for x in live))
    tag_names=["SEASON_RECORD_STATE","ROLE_TRANSITION_BOUNCEBACK","H2H_CONTEXT","RIVALRY_CONTEXT","TRAVEL_CONTEXT","RESULT_QUALITY_REGRESSION","SCHEDULE_RESUME_QUALITY","RECENT_VS_SEASON","MATCHUP_DIFFERENTIAL","BIG_AL_OBSERVATION_HYPOTHESIS","PT_DERIVED","STANDARD_MINER"]
    by_tag={}
    for tag in tag_names:
        all_t=[x for x in rows if tag in x["tags"]]
        conf_t=[x for x in all_t if x["confirmation_pass"]]
        live_t=[x for x in all_t if x["live_authority_eligible"]]
        by_tag[tag]={
            "mechanisms":len(all_t),"confirmed_mechanisms":len(conf_t),"live_authority_mechanisms":len(live_t),
            "independent_live_family_votes":len(set((x["market"],x["evidence_family_key"]) for x in live_t)),
            "confirmed_rules":[{k:x.get(k) for k in ("signature","mechanism_id","market","direction","rule","confirmation_n","confirmation_rate","live_authority_eligible","evidence_family_key")} for x in conf_t],
        }
    market_counts={}
    for market in sorted(set(x["market"] for x in rows)):
        rr=[x for x in rows if x["market"]==market]
        market_counts[market]={"mechanisms":len(rr),"confirmed_mechanisms":sum(x["confirmation_pass"] for x in rr),"live_authority_mechanisms":sum(x["live_authority_eligible"] for x in rr)}
    return {
        "mechanisms":len(rows),
        "confirmed_mechanisms":len(confirmed),
        "live_authority_mechanisms":len(live),
        "independent_live_family_votes":len(independent_live),
        "by_market":market_counts,
        "by_tag":by_tag,
        "records":rows,
        "family_vote_policy":"ONE_NORMALIZED_INDEPENDENT_FAMILY_ONE_VOTE",
        "selection_uses_2026":False,
        "expert_observation_vocabulary_seeded_from_2026":True,
        "expert_rating_selection_weight":0,
        "expert_pick_outcome_selection_weight":0,
        "production_authority":0,
    }


def _ncaaf_system_library_delta(current: dict[str,Any], previous_report: dict[str,Any] | None) -> dict[str,Any]:
    if not previous_report:
        return {"status":"NO_PRIOR_REPORT","prior_source_tag":None,"added_confirmed":[],"removed_confirmed":[],"restored_confirmed":[],"newly_confirmed":[],"upgraded_to_confirmed":[],"added_live":[],"removed_live":[],"net_confirmed":None,"net_live":None,"net_independent_live_families":None,"reconciliation":{"status":"NO_PRIOR_REPORT"}}
    prev_inv=previous_report.get("system_library_inventory")
    if not isinstance(prev_inv,dict) or "records" not in prev_inv:
        prev_inv=_ncaaf_system_library_inventory(previous_report.get("system_miner_v3") or {})
    cur={x["signature"]:x for x in (current.get("records") or [])}
    prv={x["signature"]:x for x in (prev_inv.get("records") or [])}
    cc={k for k,v in cur.items() if v.get("confirmation_pass")}; pc={k for k,v in prv.items() if v.get("confirmation_pass")}
    cl={k for k,v in cur.items() if v.get("live_authority_eligible")}; pl={k for k,v in prv.items() if v.get("live_authority_eligible")}
    prior_all=set(prv)
    _prior_delta=previous_report.get("system_library_change_audit") or {}
    recovery_sigs={str(x.get("signature")) for x in (_prior_delta.get("removed_confirmed") or []) if x.get("signature")}
    added=cc-pc; removed=pc-cc
    restored=added&recovery_sigs
    upgraded=(added&prior_all)-restored
    newly=added-prior_all-recovery_sigs
    def pack(keys, source):
        return [{k:source[s].get(k) for k in ("signature","mechanism_id","market","direction","rule","conditions","tags","confirmation_n","confirmation_rate","live_authority_eligible","evidence_family_key","library_origin","incumbent_revalidation_status","incumbent_missing_atoms")} for s in sorted(keys)]
    held={k for k,v in cur.items() if str(v.get("incumbent_revalidation_status") or "")=="HOLD_NOT_EVALUABLE"}
    expected_final=len(pc)-len(removed)+len(restored)+len(newly)+len(upgraded)
    actual_final=len(cc)
    recon={
        "status":"PASS" if expected_final==actual_final else "FAIL",
        "prior_confirmed":len(pc),"demoted_confirmed":len(removed),
        "restored_confirmed":len(restored),"newly_confirmed":len(newly),"upgraded_to_confirmed":len(upgraded),
        "expected_final_confirmed":expected_final,"actual_final_confirmed":actual_final,
        "invariant":"prior_confirmed - demoted_confirmed + restored_confirmed + newly_confirmed + upgraded_to_confirmed == final_confirmed",
    }
    return {
        "status":"COMPARED_TO_PRIOR_CURRENT_REPORT",
        "prior_source_tag":previous_report.get("source_tag"),
        "prior_created_utc":previous_report.get("created_utc"),
        "added_confirmed":pack(added,cur),"removed_confirmed":pack(removed,prv),
        "restored_confirmed":pack(restored,cur),"newly_confirmed":pack(newly,cur),"upgraded_to_confirmed":pack(upgraded,cur),
        "added_live":pack(cl-pl,cur),"removed_live":pack((pl-cl)-held,prv),
        "held_live_not_evaluable":pack((pl-cl)&held,cur),
        "net_confirmed":int(current.get("confirmed_mechanisms",0))-int(prev_inv.get("confirmed_mechanisms",0)),
        "net_live":int(current.get("live_authority_mechanisms",0))-int(prev_inv.get("live_authority_mechanisms",0)),
        "net_independent_live_families":int(current.get("independent_live_family_votes",0))-int(prev_inv.get("independent_live_family_votes",0)),
        "prior_counts":{"confirmed_mechanisms":prev_inv.get("confirmed_mechanisms"),"live_authority_mechanisms":prev_inv.get("live_authority_mechanisms"),"independent_live_family_votes":prev_inv.get("independent_live_family_votes")},
        "current_counts":{"confirmed_mechanisms":current.get("confirmed_mechanisms"),"live_authority_mechanisms":current.get("live_authority_mechanisms"),"independent_live_family_votes":current.get("independent_live_family_votes")},
        "reconciliation":recon,
        "selection_uses_2026":False,"production_authority":0,
    }

def _log_ncaaf_system_library_audit(*, inventory: dict[str,Any], delta: dict[str,Any], miners: dict[str,Any], log_func=print) -> None:
    bt=(inventory.get("by_tag") or {}).get("ROLE_TRANSITION_BOUNCEBACK") or {}
    st=(inventory.get("by_tag") or {}).get("SEASON_RECORD_STATE") or {}
    pt=(inventory.get("by_tag") or {}).get("PT_DERIVED") or {}
    obs=(inventory.get("by_tag") or {}).get("BIG_AL_OBSERVATION_HYPOTHESIS") or {}
    h2h=(inventory.get("by_tag") or {}).get("H2H_CONTEXT") or {}
    riv=(inventory.get("by_tag") or {}).get("RIVALRY_CONTEXT") or {}
    trv=(inventory.get("by_tag") or {}).get("TRAVEL_CONTEXT") or {}
    rq=(inventory.get("by_tag") or {}).get("RESULT_QUALITY_REGRESSION") or {}
    sr=(inventory.get("by_tag") or {}).get("SCHEDULE_RESUME_QUALITY") or {}
    rvs=(inventory.get("by_tag") or {}).get("RECENT_VS_SEASON") or {}
    md=(inventory.get("by_tag") or {}).get("MATCHUP_DIFFERENTIAL") or {}
    log_func(
        f"[NCAAF-SYSTEM-LIBRARY-SUMMARY] mechanisms={inventory.get('mechanisms')} confirmed={inventory.get('confirmed_mechanisms')} "
        f"live_authority={inventory.get('live_authority_mechanisms')} independent_live_families={inventory.get('independent_live_family_votes')} "
        f"state_confirmed={st.get('confirmed_mechanisms',0)} state_live={st.get('live_authority_mechanisms',0)} "
        f"bounceback_confirmed={bt.get('confirmed_mechanisms',0)} bounceback_live={bt.get('live_authority_mechanisms',0)} "
        f"h2h_confirmed={h2h.get('confirmed_mechanisms',0)} h2h_live={h2h.get('live_authority_mechanisms',0)} "
        f"rivalry_confirmed={riv.get('confirmed_mechanisms',0)} rivalry_live={riv.get('live_authority_mechanisms',0)} "
        f"travel_confirmed={trv.get('confirmed_mechanisms',0)} travel_live={trv.get('live_authority_mechanisms',0)} "
        f"result_quality_confirmed={rq.get('confirmed_mechanisms',0)} result_quality_live={rq.get('live_authority_mechanisms',0)} "
        f"schedule_resume_confirmed={sr.get('confirmed_mechanisms',0)} schedule_resume_live={sr.get('live_authority_mechanisms',0)} "
        f"recent_vs_season_confirmed={rvs.get('confirmed_mechanisms',0)} recent_vs_season_live={rvs.get('live_authority_mechanisms',0)} "
        f"matchup_diff_confirmed={md.get('confirmed_mechanisms',0)} matchup_diff_live={md.get('live_authority_mechanisms',0)} "
        f"bigal_hypothesis_confirmed={obs.get('confirmed_mechanisms',0)} bigal_hypothesis_live={obs.get('live_authority_mechanisms',0)} "
        f"pt_confirmed={pt.get('confirmed_mechanisms',0)} pt_live={pt.get('live_authority_mechanisms',0)} family_vote_policy=ONE_NORMALIZED_INDEPENDENT_FAMILY_ONE_VOTE 2026_selection=FALSE authority=0"
    )
    for market,mr in (miners or {}).items():
        sf=(mr or {}).get("special_family_audit") or {}
        for key,label in (("season_record_state","SEASON_RECORD_STATE"),("role_transition_bounceback","ROLE_TRANSITION_BOUNCEBACK"),("h2h_context","H2H_CONTEXT"),("rivalry_context","RIVALRY_CONTEXT"),("travel_context","TRAVEL_CONTEXT"),("result_quality_regression","RESULT_QUALITY_REGRESSION"),("schedule_resume_quality","SCHEDULE_RESUME_QUALITY"),("recent_vs_season","RECENT_VS_SEASON"),("matchup_differential","MATCHUP_DIFFERENTIAL"),("big_al_observation_hypothesis","BIG_AL_OBSERVATION_HYPOTHESIS")):
            a=sf.get(key) or {}
            log_func(
                f"[NCAAF-SYSTEM-LIBRARY-CATEGORY] market={market} category={label} tested_hypotheses={a.get('tested_hypotheses',0)} "
                f"finalist_systems={a.get('finalist_systems',0)} mechanisms={a.get('mechanism_families',0)} confirmed={a.get('confirmed_mechanisms',0)} "
                f"live_authority={a.get('live_authority_mechanisms',0)} independent_family_votes={a.get('independent_live_family_votes',0)} authority=0"
            )
    _recon=delta.get("reconciliation") or {}
    log_func(
        f"[NCAAF-SYSTEM-LIBRARY-DELTA] status={delta.get('status')} prior_source={delta.get('prior_source_tag')} "
        f"confirmed_net={delta.get('net_confirmed')} live_net={delta.get('net_live')} independent_live_family_net={delta.get('net_independent_live_families')} "
        f"added_confirmed={len(delta.get('added_confirmed') or [])} restored_confirmed={len(delta.get('restored_confirmed') or [])} newly_confirmed={len(delta.get('newly_confirmed') or [])} upgraded_to_confirmed={len(delta.get('upgraded_to_confirmed') or [])} removed_confirmed={len(delta.get('removed_confirmed') or [])} "
        f"added_live={len(delta.get('added_live') or [])} removed_live={len(delta.get('removed_live') or [])} held_live_not_evaluable={len(delta.get('held_live_not_evaluable') or [])} reconciliation={_recon.get('status')} 2026_selection=FALSE authority=0"
    )
    log_func(
        f"[NCAAF-SYSTEM-LIBRARY-RECONCILIATION] status={_recon.get('status')} prior_confirmed={_recon.get('prior_confirmed')} "
        f"demoted_confirmed={_recon.get('demoted_confirmed')} restored_confirmed={_recon.get('restored_confirmed')} newly_confirmed={_recon.get('newly_confirmed')} upgraded_to_confirmed={_recon.get('upgraded_to_confirmed')} "
        f"expected_final={_recon.get('expected_final_confirmed')} actual_final={_recon.get('actual_final_confirmed')} authority=0"
    )
    for state_key in ("added_confirmed","added_live","removed_confirmed","removed_live","held_live_not_evaluable"):
        for x in (delta.get(state_key) or []):
            log_func(
                f"[NCAAF-SYSTEM-LIBRARY-CHANGE] state={state_key.upper()} market={x.get('market')} id={x.get('mechanism_id')} tags={'+'.join(x.get('tags') or [])} "
                f"confirmation={x.get('confirmation_rate')}/{x.get('confirmation_n')} live={x.get('live_authority_eligible')} family={x.get('evidence_family_key')} rule={x.get('rule')}"
            )
    # Always print confirmed special-family rules, even if the previous run already had them,
    # so a user can see exactly what the new vocabulary contributes today.
    for tag in ("SEASON_RECORD_STATE","ROLE_TRANSITION_BOUNCEBACK","H2H_CONTEXT","RIVALRY_CONTEXT","TRAVEL_CONTEXT","BIG_AL_OBSERVATION_HYPOTHESIS"):
        for x in (((inventory.get("by_tag") or {}).get(tag) or {}).get("confirmed_rules") or [])[:20]:
            log_func(
                f"[NCAAF-SYSTEM-LIBRARY-SPECIAL] category={tag} market={x.get('market')} id={x.get('mechanism_id')} "
                f"confirmation={x.get('confirmation_rate')}/{x.get('confirmation_n')} live={x.get('live_authority_eligible')} family={x.get('evidence_family_key')} rule={x.get('rule')}"
            )

def run_ncaaf_research_v2(*, dashboard_module, utils_module=None, production_module=None, bucket_name="sharp-models", storage_client=None,
                          log_func=print, hard_fail=True) -> dict[str,Any]:
    try:
        # Standalone-safe: ensure the automatic external-rating bridge has had a
        # chance to populate the same Miner frame before research begins.
        external_ratings=refresh_prediction_tracker_external(
            dashboard_module=dashboard_module,storage_client=storage_client,bucket_name=bucket_name,
            include_history=True,include_current=True,force=False,log_func=log_func
        )
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}
        games=cache.get("games"); seasons=np.asarray(cache.get("season_arr"),dtype=float); oof_margin=np.asarray(cache.get("oof_margin"),dtype=float); oof_total=np.asarray(cache.get("oof_total"),dtype=float)
        cols=list(cache.get("candidate_feature_cols") or []); miner_games=cache.get("miner_games")
        if games is None or getattr(games,"empty",True): raise RuntimeError("historical research games cache missing")
        if len(seasons)!=len(games) or len(oof_margin)!=len(games) or len(oof_total)!=len(games): raise RuntimeError("OOF/cache row alignment mismatch")
        if miner_games is None or getattr(miner_games,"empty",True): miner_games=games.copy()
        miner_games,expert_side_bridge=_attach_exact_expert_flags_to_miner(dashboard_module,miner_games,log_func=log_func)
        miner_games,historical_state_context_bridge=_attach_historical_state_context_to_miner(dashboard_module,miner_games,log_func=log_func)
        miner_games,deep_context_bridge=_attach_deep_stat_context_to_miner(dashboard_module,miner_games,log_func=log_func,for_live=False)
        miner_games,situational_context_bridge=_attach_h2h_rivalry_travel_context(dashboard_module,miner_games,log_func=log_func,for_live=False)
        # Persist the enriched frame for prospective/live adapters in this process.
        try:
            cache["miner_games"]=miner_games
            getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})["miner_games"]=miner_games
        except Exception:
            pass
        # Hard seal: 2026+ may exist in source tables, but can never enter discovery/confirmation.
        historic=np.isfinite(seasons)&(seasons<=max(CONFIRMATION_SEASONS))
        if historic.sum()<500: raise RuntimeError(f"insufficient <=2025 history n={int(historic.sum())}")
        g=games.loc[historic].reset_index(drop=True); mg=miner_games.loc[historic].reset_index(drop=True); sy=seasons[historic]; om=oof_margin[historic]; ot=oof_total[historic]
        log_func(f"[NCAAF-RV2311-PREFLIGHT] source={NCAAF_RESEARCH_V2_SOURCE_TAG} rows={len(g)} seasons={sorted(set(sy.astype(int)))} discovery<=2023 confirmation=2024,2025 prospective>=2026 production_authority=0")
        stat=run_orthogonal_stat_research(g,sy,om,ot,cols,log_func=log_func)
        sparse_stat=run_sparse_stat_research(g,sy,om,ot,log_func=log_func)
        miners={m:run_system_miner_v3(mg,sy,m,dashboard_module=dashboard_module,log_func=log_func,max_depth=5) for m in ("spreads","h2h","totals")}
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        b=storage_client.bucket(bucket_name)
        _previous_report=None
        try:
            _prev_blob=b.blob(REPORT_CURRENT_BLOB)
            if _prev_blob.exists():
                _previous_report=json.loads(_prev_blob.download_as_text())
        except Exception as _prev_exc:
            log_func(f"[NCAAF-SYSTEM-LIBRARY-PRIOR] status=UNAVAILABLE error={type(_prev_exc).__name__}:{_prev_exc}")
        miners,incumbent_system_library_audit=_reconcile_incumbent_system_library(mg,sy,miners,_previous_report,dashboard_module=dashboard_module,log_func=log_func)
        context_family_cap_audit=_apply_v231_context_family_caps(miners,log_func=log_func)
        system_results=_build_system_results(mg,sy,miners,dashboard_module=dashboard_module)
        published_system_results=_grade_published_ncaaf_systems(mg,sy)
        miner_threshold_neighborhood=_miner_authority_threshold_neighborhood(miners)
        _log_v24_evidence_audit(miners=miners,system_results=system_results,published_system_results=published_system_results,
                                threshold_neighborhood=miner_threshold_neighborhood,log_func=log_func)
        prospective=_prospective_shadow(miner_games.reset_index(drop=True),seasons,miners,dashboard_module=dashboard_module,log_func=log_func)
        market_audit=_market_rich_audit(g,utils_module)
        pt_incremental=run_pt_incremental_value_research(games=g,miner_games=mg,seasons=sy,dashboard_module=dashboard_module,production_module=production_module,log_func=log_func)
        intelligence_bridge=(getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}).get("miner_intelligence_bridge") or {"status":"NOT_AVAILABLE","production_authority":0,"selection_influence":0}

        # Persist/describe observed Big Al selections. Ratings and 2026 outcomes
        # never enter system qualification; they only tell us which generic
        # vocabulary deserves a constrained historical search lane.
        big_al_observations,big_al_ledger_diag=_load_big_al_observation_ledger(storage_client=storage_client,bucket_name=bucket_name,log_func=log_func)
        _obs_lane_audit={m:((mr or {}).get("lanes") or {}).get("expert_observation_hypothesis",{}) for m,mr in miners.items()}
        big_al_observation_research=_big_al_rating_observation_report(big_al_observations,miner_lane_audit=_obs_lane_audit,context_games=miner_games.reset_index(drop=True),dashboard_module=dashboard_module)
        log_func(f"[NCAAF-BIGAL-RATING-RESEARCH] status={big_al_observation_research.get('status')} observations={big_al_observation_research.get('ncaaf_observations')} rated={big_al_observation_research.get('rated_observations')} ratings={json.dumps(big_al_observation_research.get('rating_counts') or {},sort_keys=True)} pattern_min={BIG_AL_RATING_MIN_FOR_PATTERN_ANALYSIS} rating_qualification_weight=0 rating_authority=0 outcomes_2026_used=FALSE")

        system_library_inventory=_ncaaf_system_library_inventory(miners)
        system_library_change_audit=_ncaaf_system_library_delta(system_library_inventory,_previous_report)
        _log_ncaaf_system_library_audit(inventory=system_library_inventory,delta=system_library_change_audit,miners=miners,log_func=log_func)
        report={"source_tag":NCAAF_RESEARCH_V2_SOURCE_TAG,"version":NCAAF_RESEARCH_V2_VERSION,"created_utc":_now(),
                "status":"NCAAF_RESEARCH_V2_COMPLETE","production_authority":0,"production_contract_mutated":False,
                "benchmark":"FROZEN_NCAAF_PRODUCTION_V1","discovery_max_season":DISCOVERY_MAX_SEASON,"confirmation_seasons":list(CONFIRMATION_SEASONS),"prospective_min_season":PROSPECTIVE_MIN_SEASON,
                "rows":len(g),"seasons":sorted(set(sy.astype(int))),"orthogonal_stat":stat,"sparse_stat_v21":sparse_stat,"system_miner_v3":miners,
                "system_library_inventory":system_library_inventory,"system_library_change_audit":system_library_change_audit,"incumbent_system_library_audit":incumbent_system_library_audit,
                "context_family_cap_audit":context_family_cap_audit,
                "historical_state_context_bridge":historical_state_context_bridge,
                "deep_context_bridge":deep_context_bridge,
                "situational_context_bridge":situational_context_bridge,
                "big_al_observation_ledger":big_al_ledger_diag,"big_al_observation_research":big_al_observation_research,
                "prospective_shadow_2026":prospective,"system_results":system_results,"published_system_results":published_system_results,"miner_threshold_neighborhood":miner_threshold_neighborhood,"market_rich":market_audit,"intelligence_bridge":intelligence_bridge,"expert_side_bridge":expert_side_bridge,"external_rating_metamodel":external_ratings,"pt_incremental_value":pt_incremental,
                "miner_live_authority_policy":{"policy":NCAAF_MINER_LIVE_AUTHORITY_POLICY,"min_confirmation_n":NCAAF_MINER_LIVE_MIN_CONFIRMATION_N,"min_confirmation_rate":NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE,"uses_2026_selection":False,"source_neutral":True,"pt_systems_may_earn_bounded_vote":True,"pt_family_vote_cap":1,"h2h_family_vote_cap":1,"rivalry_family_vote_cap":1,"travel_family_vote_cap":1,"result_quality_family_vote_cap":1,"schedule_resume_family_vote_cap":1,"recent_vs_season_family_vote_cap":1,"matchup_differential_family_vote_cap":1,"pt_model_weight":0},
                "next_step":"KEEP PRODUCTION V1 FROZEN; USE V2.31.1 OBJECTIVE RESULT-QUALITY/RESUME/RECENT/MATCHUP CONTEXT PLUS V2.30 H2H/RIVALRY/TRAVEL AS SOURCE-NEUTRAL MINER INPUTS; PERSIST AND EXACT-RULE REVALIDATE INCUMBENTS; QUALIFY NEW CHALLENGERS ONLY THROUGH 2022-2025 SEALED HISTORY; KEEP 2026 OUTCOMES OUT OF SELECTION"}
        # Preserve a lightweight pickle bundle for future prospective trigger/scoring adapters.
        bundle={"report":report,"system_miner_v3":miners,"sparse_stat_v21":sparse_stat,"prospective_shadow_2026":prospective,"system_results":system_results,"published_system_results":published_system_results,"miner_threshold_neighborhood":miner_threshold_neighborhood,"incumbent_system_library_audit":incumbent_system_library_audit,"deep_context_bridge":deep_context_bridge,"big_al_observation_research":big_al_observation_research,"external_rating_metamodel":external_ratings,"pt_incremental_value":pt_incremental,"stat_family_definitions":STAT_FAMILY_TOKENS,"source_tag":NCAAF_RESEARCH_V2_SOURCE_TAG}
        body=json.dumps(_report_without_models(report),sort_keys=True,separators=(",",":"),default=str).encode()
        sha=hashlib.sha256(body).hexdigest(); hist=f"{REPORT_HISTORY_PREFIX}/{sha[:16]}/report.json"
        b.blob(hist).upload_from_string(body,content_type="application/json"); b.blob(REPORT_CURRENT_BLOB).upload_from_string(body,content_type="application/json")
        bio=io.BytesIO(); pickle.dump(bundle,bio,protocol=pickle.HIGHEST_PROTOCOL); bio.seek(0); pdata=bio.read(); b.blob(BUNDLE_CURRENT_BLOB).upload_from_string(pdata,content_type="application/octet-stream")
        report["artifact"]={"current_report":f"gs://{bucket_name}/{REPORT_CURRENT_BLOB}","current_bundle":f"gs://{bucket_name}/{BUNDLE_CURRENT_BLOB}","history_report":f"gs://{bucket_name}/{hist}","sha256":sha}
        _strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if _miner_live_authority_eligible(_m))
        _pt_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_external_ratings_family")) and _miner_live_authority_eligible(_m))
        _bounce_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_role_transition_bounceback_family")) and _miner_live_authority_eligible(_m))
        _state_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_season_record_state_family")) and _miner_live_authority_eligible(_m))
        _state_confirmed=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_season_record_state_family")) and bool(_m.get("confirmation_pass")))
        _bounce_confirmed=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_role_transition_bounceback_family")) and bool(_m.get("confirmation_pass")))
        _h2h_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_h2h_context_family")) and _miner_live_authority_eligible(_m))
        _h2h_confirmed=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_h2h_context_family")) and bool(_m.get("confirmation_pass")))
        _rivalry_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_rivalry_context_family")) and _miner_live_authority_eligible(_m))
        _rivalry_confirmed=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_rivalry_context_family")) and bool(_m.get("confirmation_pass")))
        _travel_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_travel_context_family")) and _miner_live_authority_eligible(_m))
        _travel_confirmed=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_travel_context_family")) and bool(_m.get("confirmation_pass")))
        _rq_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_result_quality_family")) and _miner_live_authority_eligible(_m))
        _rq_confirmed=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_result_quality_family")) and bool(_m.get("confirmation_pass")))
        _sr_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_schedule_resume_family")) and _miner_live_authority_eligible(_m))
        _sr_confirmed=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_schedule_resume_family")) and bool(_m.get("confirmation_pass")))
        _rvs_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_recent_vs_season_family")) and _miner_live_authority_eligible(_m))
        _rvs_confirmed=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_recent_vs_season_family")) and bool(_m.get("confirmation_pass")))
        _md_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_matchup_differential_family")) and _miner_live_authority_eligible(_m))
        _md_confirmed=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_matchup_differential_family")) and bool(_m.get("confirmation_pass")))
        _obs_strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_big_al_observation_hypothesis")) and _miner_live_authority_eligible(_m))
        _obs_confirmed=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if bool(_m.get("uses_big_al_observation_hypothesis")) and bool(_m.get("confirmation_pass")))
        _bridge_mechs=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if any(str(c).startswith(("EXPERT_PATHI_","EXPERT_BIGAL_","CORE_OOF_","SPEC_","META_PT_","PTIDX_","PT_ALL_","PT_CLUSTER_","PT_TRACKER_")) for c in (_m.get("representative_conditions") or [])))
        log_func(
            f"[NCAAF-RV2311-CONTRACT] status=PASS report=gs://{bucket_name}/{REPORT_CURRENT_BLOB} sha={sha[:16]} "
            f"stat_spread_confirmed={len(stat['confirmed_spread_families'])} stat_totals_confirmed={len(stat['confirmed_totals_families'])} "
            f"sparse_confirmed={len(sparse_stat.get('confirmed_candidates') or [])} miner_confirmed={sum(v.get('confirmed_mechanism_count',0) for v in miners.values())} "
            f"bridge_mechanisms={_bridge_mechs} miner_live_authority={_strong} independent_live_families={system_library_inventory.get('independent_live_family_votes')} "
            f"pt_live={_pt_strong} state_confirmed={_state_confirmed} state_live={_state_strong} bounceback_confirmed={_bounce_confirmed} bounceback_live={_bounce_strong} "
            f"h2h_confirmed={_h2h_confirmed} h2h_live={_h2h_strong} h2h_family_vote_cap=1 rivalry_confirmed={_rivalry_confirmed} rivalry_live={_rivalry_strong} rivalry_family_vote_cap=1 "
            f"travel_confirmed={_travel_confirmed} travel_live={_travel_strong} travel_family_vote_cap=1 travel_bridge={situational_context_bridge.get('travel_reference',{}).get('status')} "
            f"result_quality_confirmed={_rq_confirmed} result_quality_live={_rq_strong} schedule_resume_confirmed={_sr_confirmed} schedule_resume_live={_sr_strong} "
            f"recent_vs_season_confirmed={_rvs_confirmed} recent_vs_season_live={_rvs_strong} matchup_diff_confirmed={_md_confirmed} matchup_diff_live={_md_strong} deep_context_bridge={deep_context_bridge.get('status')} "
            f"deep_result_ready={deep_context_bridge.get('result_quality_ready')} deep_resume_ready={deep_context_bridge.get('resume_ready')} deep_recent_ready={deep_context_bridge.get('recent3_ready')} deep_matchup_ready={deep_context_bridge.get('matchup_ready')} "
            f"travel_ready={situational_context_bridge.get('travel_ready')} h2h_ready={situational_context_bridge.get('h2h_prior_ready')} rivalry_rows={situational_context_bridge.get('rivalry_rows')} "
            f"bigal_hypothesis_confirmed={_obs_confirmed} bigal_hypothesis_live={_obs_strong} bigal_rated_observations={big_al_observation_research.get('rated_observations')} rating_weight=0 "
            f"outcomes_2026_used_for_selection=FALSE incumbent_prior={incumbent_system_library_audit.get('prior_confirmed')} incumbent_retained={incumbent_system_library_audit.get('revalidated_retained')} "
            f"incumbent_contract_hold={incumbent_system_library_audit.get('carried_contract_hold')} incumbent_demoted={incumbent_system_library_audit.get('demoted')} "
            f"added_confirmed={len(system_library_change_audit.get('added_confirmed') or [])} restored_confirmed={len(system_library_change_audit.get('restored_confirmed') or [])} "
            f"library_reconciliation={(system_library_change_audit.get('reconciliation') or {}).get('status')} historical_state_bridge={historical_state_context_bridge.get('status')} "
            f"situational_context_bridge={situational_context_bridge.get('status')} source_neutral_system_gate=TRUE prospective_mechanisms={len((prospective or {}).get('mechanisms') or [])} production_authority=0"
        )
        return report
    except Exception as exc:
        log_func(f"[NCAAF-RV2-FAIL] {type(exc).__name__}: {exc}")
        if hard_fail: raise
        return {"source_tag":NCAAF_RESEARCH_V2_SOURCE_TAG,"status":"FAILED","error":f"{type(exc).__name__}:{exc}","production_authority":0}


def load_current_report(bucket_name="sharp-models", storage_client=None) -> dict[str,Any] | None:
    try:
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        blob=storage_client.bucket(bucket_name).blob(REPORT_CURRENT_BLOB)
        if not blob.exists(): return None
        obj=json.loads(blob.download_as_text())
        return obj if obj.get("source_tag")==NCAAF_RESEARCH_V2_SOURCE_TAG else None
    except Exception:
        return None


def match_live_systems(rows: pd.DataFrame, report: dict[str,Any], dashboard_module=None) -> pd.DataFrame:
    """Backward-compatible UI wrapper for the V2.2 live Miner evaluator."""
    return attach_live_miner_votes(rows,report,dashboard_module=dashboard_module)

def self_test() -> dict[str,Any]:
    _tf=pd.DataFrame({"Consensus_Open_Spread":[6.0],"Current_Spread":[4.0],"Is_Home":[0],"Prev_Is_Home":[1],"Prev_Is_ML_Favorite":[1],"Prev_Is_ML_Dog":[0],"Opp_Prev_Is_Home":[0],"Opp_Prev_Is_ML_Favorite":[0],"Opp_Prev_Is_ML_Dog":[1],
                      "Prev_SU_Margin":[-28.0],"Opp_Prev_SU_Margin":[7.0],"Prev_Team_Score":[0.0],"Prev_Opponent_Score":[28.0],"Opp_Prev_Team_Score":[24.0],"Opp_Prev_Opponent_Score":[21.0],"Prev2_SU_Margin":[-10.0],"Prev3_SU_Margin":[-3.0],"Prev_ATS_Margin":[-8.0],"Prev2_ATS_Margin":[-2.0],"Prev3_ATS_Margin":[-5.0],
                      "Team_Game_Number_Prior":[4],"Opp_Game_Number_Prior":[4],"Current_Loss_Streak_Prior":[4],"ATS_Loss_Streak_Prior":[4],"Opp_ATS_Loss_Streak_Prior":[4],"ATS_WinPct_Prior":[0.20],"Opp_ATS_WinPct_Prior":[0.00],"Team_WinPct_Prior":[0.75],"Opp_WinPct_Prior":[0.00],
                      "Pathi_FB_Dog_Hook_Above_3":[1],"BigAl_CF2_LateSeasonRevengeDog":[1],"_V29_CORE_INCUMBENT_EDGE_POINTS":[3.0],
                      "_V29_SPEC_STRUCTURED_STATS_EDGE_POINTS":[2.5],"_V29_SPEC_STRUCTURED_STATS_DIVERGENCE_FROM_CORE":[1.5],"_V29_SPEC_STRUCTURED_STATS_DIVERGENCE_CUT":[1.0],
                      "_V210_PT_META_MARGIN_TEAM":[6.5],"_V210_PT_META_EDGE_POINTS":[3.0],"_V210_PT_META_SYSTEM_COUNT":[5]})
    live_atoms={a["name"] for a in _extended_atoms(_tf,for_live=True,market="spreads")}
    _tt=_tf.copy(); _tt["Consensus_Open_Total"]=[60.5]; _tt["Current_Total"]=[58.5]; _tt["Opp_Prev_SU_Margin"]=[-3.0]; _tt["Prev_Team_Score"]=[34.0]; _tt["Prev_Opponent_Score"]=[35.0]; _tt["Opp_Prev_Team_Score"]=[17.0]; _tt["Opp_Prev_Opponent_Score"]=[45.0]
    total_live_atoms={a["name"] for a in _extended_atoms(_tt,for_live=True,market="totals")}
    _obs_norm=_normalize_big_al_observation({"sport":"NCAAF","market":"spreads","game_date":"2026-10-09","selection_team":"Wyoming","line":4.5,"rating":2,"outcome":"WIN","final_score":"x"})
    _obs_report=_big_al_rating_observation_report([_obs_norm] if _obs_norm else [])
    q=_bh_qvalues([.01,.04,.20]); fam=_classify_feature_families(["Rush_EPA","Opp_Rush_EPA","Line_Move_60m","Sharp_Soft_Divergence","Actual_Margin"])
    odds=np.asarray([200.0,-200.0]); won=np.asarray([1.0,1.0]); ret=_american_unit_return(odds,won)
    # Name-safe Prediction Tracker contract: system identity must survive an
    # arbitrary column reorder and near-miss/fuzzy headers must not be accepted.
    _pt_df=pd.DataFrame({
        "Home":["Alpha","Gamma"],"Road":["Beta","Delta"],"line":[3.0,-2.0],
        "lineespn":[4.0,-1.0],"linedokter":[5.0,-3.0],"Pi-Ratings Bias":[3.5,-2.5],
        "Keeper":[4.5,-1.5],"Pigskin Index":[2.5,-2.0],
    })
    _pt_a,_pt_da=_pt_parse_csv(_pt_df.to_csv(index=False).encode(),2025,source_context="SELF_TEST_A",log_func=lambda *a,**k:None)
    _pt_b,_pt_db=_pt_parse_csv(_pt_df[["Pigskin Index","Road","linedokter","line","Keeper","Home","Pi-Ratings Bias","lineespn"]].to_csv(index=False).encode(),2025,source_context="SELF_TEST_B",log_func=lambda *a,**k:None)
    _pt_bad=_pt_df.rename(columns={"lineespn":"lineespn_extra"})
    _pt_c,_pt_dc=_pt_parse_csv(_pt_bad.to_csv(index=False).encode(),2025,source_context="SELF_TEST_BAD",log_func=lambda *a,**k:None)
    _pt_name_safe=bool(
        _pt_da.get("metamodel_status")=="FULL_FIVE_VERIFIED" and _pt_db.get("metamodel_status")=="FULL_FIVE_VERIFIED" and
        np.allclose(pd.to_numeric(_pt_a["meta_margin_home"],errors="coerce"),pd.to_numeric(_pt_b["meta_margin_home"],errors="coerce"),equal_nan=True) and
        _pt_da.get("system_columns",{}).get("ESPN_FPI")=="lineespn" and _pt_da.get("system_columns",{}).get("DOKTER")=="linedokter" and
        _pt_dc.get("metamodel_status")=="INCOMPLETE_FAIL_CLOSED" and _pt_dc.get("system_columns",{}).get("ESPN_FPI") is None
    )
    # Relay payload cleanup must preserve the source header exactly and ignore
    # reader metadata/fences.  Named markdown parsing is separately validated.
    _relay_raw=("Title: Prediction Tracker\nURL Source: example\n\n```text\n"+_pt_df.to_csv(index=False)+"```\n").encode()
    _relay_clean=_pt_extract_csv_payload(_relay_raw)
    _relay_df,_relay_errs=_pt_read_csv_frame(_relay_clean)
    _relay_csv_ok=bool(_relay_df is not None and list(_relay_df.columns)==list(_pt_df.columns) and len(_relay_df)==2)
    _md=("home | road | Line | Computer Adj. Line | ESPN FPI | Pi-Ratings Bias | Dokter | Keeper | Pigskin Index\n"
         "--- | --- | --- | --- | --- | --- | --- | --- | ---\n"
         "Alpha | Beta | 3 | 3 | 4 | 3.5 | 5 | 4.5 | 2.5\n").encode()
    _mdf,_mdd=_pt_extract_live_markdown_table(_md)
    _relay_md_ok=bool(_mdf is not None and _mdd.get("status")=="PASS" and all(x in _mdf.columns for x in ("ESPN FPI","Pi-Ratings Bias","Dokter","Keeper","Pigskin Index")))
    _relay_url_ok=(_pt_relay_url(PT_LIVE_PAGE_URL)=="https://r.jina.ai/https://www.thepredictiontracker.com/predncaa.php")
    _challenge_rejected=_pt_is_challenge_payload(b"Title: Just a moment...\nChecking your browser before accessing thepredictiontracker.com")
    # Current-season merge must replace only the newest overlapping occurrence,
    # preserving an older same-home/road rematch in the archive.
    _ma=pd.DataFrame({"home_key":["alpha","alpha","gamma"],"away_key":["beta","beta","delta"],"meta_margin_home":[1.0,2.0,3.0],"source_row":[1,9,4]})
    _ml=pd.DataFrame({"home_key":["alpha"],"away_key":["beta"],"meta_margin_home":[4.0]})
    _mm,_mmd=_pt_merge_current_season(_ma,_ml,log_func=lambda *a,**k:None)
    _merge_rematch_ok=bool(len(_mm)==3 and (_mm["home_key"].eq("alpha")&_mm["away_key"].eq("beta")).sum()==2 and 1.0 in set(pd.to_numeric(_mm["meta_margin_home"],errors="coerce").dropna()))

    # Sparse Prediction Tracker coverage regression: a game listed by PT with
    # only four of five systems must preserve those four components without
    # fabricating/renormalizing META. A game absent from PT remains untouched.
    class _DSP: pass
    _DSP._V1357_SPREAD_RESEARCH_CACHE={
        "games":pd.DataFrame({
            "Season":[2025,2025],"Team_Norm":["alpha","gamma"],"Opponent_Norm":["beta","delta"],
            "Is_Home":[1,1],"Game_Date":["2025-09-01","2025-09-02"],"Consensus_Open_Spread":[-3.0,2.0],
        }),
        "miner_games":pd.DataFrame({
            "Season":[2025,2025],"Team_Norm":["alpha","gamma"],"Opponent_Norm":["beta","delta"],
            "Is_Home":[1,1],"Game_Date":["2025-09-01","2025-09-02"],"Consensus_Open_Spread":[-3.0,2.0],
        }),
    }
    _sp_ext=pd.DataFrame({
        "season":[2025],"home_key":["alpha"],"away_key":["beta"],"game_date":["2025-09-01"],
        "meta_margin_home":[np.nan],"meta_system_count":[4],"prediction_avg_home":[4.0],"tracker_open_home":[3.0],
        "DOKTER":[4.0],"PI_RATE_BIAS":[5.0],"KEEPER":[3.5],"ESPN_FPI":[4.5],"PIGSKIN_INDEX":[np.nan],
    })
    _sp_diag=_pt_attach_history_to_cache(_DSP,_sp_ext,log_func=lambda *a,**k:None)
    _sp_mg=_DSP._V1357_SPREAD_RESEARCH_CACHE["miner_games"]
    _alias_map_test,_alias_unresolved_test=_pt_candidate_map(
        ["Alabama"],["alabama crimson tide"],alias_hints={"alabama":"alabama crimson tide"}
    )
    _pt_alias_hint_ok=bool(
        _alias_map_test.get("alabama")=="alabama crimson tide" and not _alias_unresolved_test
    )
    _hard_valid={
        "miami oh redhawks","miami hurricanes","ole mississippi rebels","utsa roadrunners",
        "troy trojans","louisiana ragin cajuns","ucf knights","fiu panthers"
    }
    _hard_alias_test,_hard_alias_unresolved=_pt_hard_external_aliases(_hard_valid)
    _pt_final_hard_aliases_ok=bool(len(_hard_alias_test)==8 and not _hard_alias_unresolved)
    _sparse_pt_ok=bool(
        _sp_diag.get("matched_rows")==1 and _sp_diag.get("partial_matched_rows")==1 and
        _sp_diag.get("full_five_matched_rows")==0 and _sp_diag.get("no_pt_rows")==1 and
        pd.isna(_sp_mg.loc[0,"_V210_PT_META_MARGIN_TEAM"]) and
        float(_sp_mg.loc[0,"_V217_PT_COMPONENT_AVAILABLE_COUNT"])==4.0 and
        pd.notna(_sp_mg.loc[0,"_V212_PT_DOKTER_MARGIN_TEAM"]) and
        pd.isna(_sp_mg.loc[1,"_V212_PT_DOKTER_MARGIN_TEAM"])
    )

    # Exact expert-side bridge regression: prove that a team-side Pathi trigger
    # and an opponent-side Big Al trigger survive projection into miner_games.
    class _Q:
        def __init__(self,df): self.df=df
        def to_dataframe(self): return self.df.copy()
    class _BQ:
        def __init__(self,df): self.df=df
        def query(self,*a,**k): return _Q(self.df)
    class _D:
        HISTORICAL_NCAAF_CORE_VIEW="proj.ds.view"
        PATHI_DIRECTIONAL_MEMORY_COLS=("Pathi_FB_Dog_Hook_Above_3",)
        def __init__(self,h): self.bq_client=_BQ(h)
        @staticmethod
        def _hc_restore_authoritative_pathi_flags(h,log_func=print):
            z=h.copy(); z["Pathi_FB_Dog_Hook_Above_3"]=[1,0]; return z,True,{"synthetic":True}
        @staticmethod
        def _hc_prepare_bigal_history_state(h,log_func=print): return h.copy()
        @staticmethod
        def add_pathi_bigal_rule_flags(h):
            z=h.copy(); z["BigAl_CF1_Week2Home42Win"]=[0,0]; z["BigAl_CF2_LateSeasonRevengeDog"]=[0,1]; z["BigAl_CF3_Fade19PlusFavoriteUpsetLoss"]=[0,0]; return z
    _hh=pd.DataFrame({"Season":[2023,2023],"Source_Game_ID":["g1","g1"],"Team_Norm":["alpha","beta"],"Opponent_Norm":["beta","alpha"],"Historical_Core_Eligible":[1,1]})
    _mg=pd.DataFrame({"Season":[2023],"Source_Game_ID":["g1"],"Team_Norm":["alpha"],"Opponent_Norm":["beta"],"Consensus_Open_Spread":[3.5],"Actual_Margin":[7.0],"Actual_Total":[45.0],"Consensus_Open_Total":[44.0]})
    _V214_EXPERT_SIDE_CACHE.clear()
    _mg2,_ebd=_attach_exact_expert_flags_to_miner(_D(_hh),_mg,log_func=lambda *a,**k:None)
    _expert_atoms={a["name"] for a in _extended_atoms(_mg2,for_live=True,market="spreads")}
    _expert_bridge_ok=bool(
        _ebd.get("status") in {"PASS","PASS_OCCURRENCE_LEDGER_RECONCILED"} and int(pd.to_numeric(_mg2["Pathi_FB_Dog_Hook_Above_3"],errors="coerce").fillna(0).sum())==1 and
        int(pd.to_numeric(_mg2["BigAl_CF2_LateSeasonRevengeDog__ROAD_SIDE"],errors="coerce").fillna(0).sum())==1 and
        "EXPERT_PATHI_FB_DOG_HOOK_ABOVE_3" in _expert_atoms and
        any("BIGAL_CF2_LATESEASONREVENGEDOG_ROAD_SIDE" in x for x in _expert_atoms)
    )
    _V214_EXPERT_SIDE_CACHE.clear()

    # V2.28 historical state bridge regression. The road-side team enters game
    # five 0-4 SU / 0-4 ATS, having just lost SU at home as a favorite. The
    # current physical game is HOME-oriented in the Miner frame, so the state
    # must appear through OPP_* mirror atoms without using the current result.
    _hist=[]
    _dates=pd.date_range("2023-09-01",periods=5,freq="7D")
    for _i,_dt in enumerate(_dates,1):
        _gid=f"sg{_i}"
        if _i<5:
            _bhome=1 if _i==4 else 0; _bsp=-3.0 if _i==4 else 6.0; _bpf=10.0; _bpa=17.0
            _oppn=f"opp{_i}"
        else:
            _bhome=0; _bsp=6.0; _bpf=20.0; _bpa=17.0; _oppn="alpha"
        _hist.append({"Season":2023,"Source_Game_ID":_gid,"Game_Date":_dt.strftime("%Y-%m-%d"),"Team_Norm":"beta","Opponent_Norm":_oppn,"Historical_Core_Eligible":1,"Is_Home":_bhome,"Consensus_Open_Spread":_bsp,"Team_Score":_bpf,"Opponent_Score":_bpa})
        if _i==5:
            _hist.append({"Season":2023,"Source_Game_ID":_gid,"Game_Date":_dt.strftime("%Y-%m-%d"),"Team_Norm":"alpha","Opponent_Norm":"beta","Historical_Core_Eligible":1,"Is_Home":1,"Consensus_Open_Spread":-6.0,"Team_Score":17.0,"Opponent_Score":20.0})
    class _DState:
        HISTORICAL_NCAAF_CORE_VIEW="proj.ds.view"
        def __init__(self,h): self.bq_client=_BQ(h)
    _state_mg=pd.DataFrame({"Season":[2023],"Source_Game_ID":["sg5"],"Game_Date":[_dates[-1].strftime("%Y-%m-%d")],"Team_Norm":["alpha"],"Opponent_Norm":["beta"],"Is_Home":[1],"Consensus_Open_Spread":[-6.0],"Consensus_Open_Total":[50.0],"Actual_Margin":[-3.0],"Actual_Total":[40.0]})
    _V228_HISTORICAL_STATE_CACHE.clear()
    _state_mg2,_state_diag=_attach_historical_state_context_to_miner(_DState(pd.DataFrame(_hist)),_state_mg,log_func=lambda *a,**k:None)
    _state_atoms={a["name"] for a in _extended_atoms(_state_mg2,for_live=True,market="spreads")}
    _hist_state_bridge_ok=bool(
        _state_diag.get("status")=="PASS" and float(pd.to_numeric(_state_mg2.loc[0,"Opp_Game_Number_Prior"],errors="coerce"))==4.0 and
        float(pd.to_numeric(_state_mg2.loc[0,"Opp_WinPct_Prior"],errors="coerce"))==0.0 and
        float(pd.to_numeric(_state_mg2.loc[0,"Opp_ATS_Loss_Streak_Prior"],errors="coerce"))==4.0 and
        "OPP_SU_AND_ATS_WINLESS_AFTER_4_PLUS" in _state_atoms and "OPP_BOUNCEBACK_HOME_FAV_UPSET_TO_ROAD_DOG" in _state_atoms
    )
    _V228_HISTORICAL_STATE_CACHE.clear()
    # Exact occurrence-ledger reconciliation test: the ledger stores graded
    # occurrences, while `fired` can be larger when ATS target/result is absent.
    class _DOcc:
        _V143_SYSTEM_HISTORY_CACHE={
            "Pathi_FB_Dog_Hook_Above_3":{
                "family":"Pathi","role":"directional","fired":2,"graded":1,
                "occurrences":[{"season":2023,"date":"2023-09-01","team":"alpha","opponent":"beta","ats_win":1.0}],
            },
            "BigAl_CF2_LateSeasonRevengeDog":{
                "family":"BigAl","role":"directional","fired":1,"graded":1,
                "occurrences":[{"season":2023,"date":"2023-09-01","team":"beta","opponent":"alpha","ats_win":1.0}],
            },
        }
    _mg_occ=pd.DataFrame({"Season":[2023],"Game_Date":["2023-09-01"],"Team_Norm":["alpha"],"Opponent_Norm":["beta"]})
    _mg_occ2,_occdiag=_attach_exact_expert_flags_to_miner(_DOcc(),_mg_occ,log_func=lambda *a,**k:None)
    _occ_recon_ok=bool(
        _occdiag.get("status")=="PASS_OCCURRENCE_LEDGER_RECONCILED" and
        int(_occdiag.get("expected_fired",0))==3 and int(_occdiag.get("expected_graded",0))==2 and
        int(_occdiag.get("occurrence_records",0))==2 and int(_occdiag.get("ungraded_fires",0))==1 and
        int(_occdiag.get("projection_delta",999))==0 and
        int(pd.to_numeric(_mg_occ2.get("Pathi_FB_Dog_Hook_Above_3"),errors="coerce").fillna(0).sum())==1 and
        int(pd.to_numeric(_mg_occ2.get("BigAl_CF2_LateSeasonRevengeDog__ROAD_SIDE"),errors="coerce").fillna(0).sum())==1
    )
    # Production-log arithmetic from the V2.14.3 run should reconcile as 19
    # fired-but-ungraded events, not as missing projected occurrences.
    _recon_example={"fired":6384,"graded":6365,"occurrence_records":6365,"projected":6365}
    _occ_recon_ok=bool(_occ_recon_ok and _recon_example["projected"]==_recon_example["occurrence_records"] and (_recon_example["fired"]-_recon_example["graded"])==19)

    # Persistent incumbent regression: a confirmed rule omitted from the current
    # challenger search must be exact-rule revalidated and restored.
    _rows=[]
    for _sy,_n in ((2022,60),(2023,60),(2024,40),(2025,40)):
        for _i in range(_n):
            _win=(_i % 5)!=0  # 80% each season
            _rows.append({"Season":_sy,"Consensus_Open_Spread":3.0,"Consensus_Open_Total":50.0,"Actual_Margin":1.0 if _win else -5.0,"Actual_Total":50.0,"Team_Game_Number_Prior":max(1,_i%10),"Team_WinPct_Prior":.5,"ATS_Loss_Streak_Prior":0})
    _ig=pd.DataFrame(_rows); _isy=pd.to_numeric(_ig["Season"],errors="coerce").to_numpy(float)
    _old_mech={"mechanism_id":"NCAAF-MECH-INCUMBENT-TEST","market":"spreads","direction":"PLAY_ON","representative_conditions":["CURRENT_DOG"],"families":["MARKET_ROLE"],"evidence_family_key":"NCAAF-MECH-INCUMBENT-TEST","confirmation_pass":True,"confirmation_n":80,"confirmation_rate":.80,"live_authority_eligible":True,"current_qualified":True,"production_authority":0}
    _prev={"source_tag":"prior","system_miner_v3":{"spreads":{"mechanism_families":[_old_mech]}},"system_library_inventory":{"records":[]},"system_library_change_audit":{}}
    _cur={"spreads":{"mechanism_families":[],"special_family_audit":{}},"h2h":{"mechanism_families":[],"special_family_audit":{}},"totals":{"mechanism_families":[],"special_family_audit":{}}}
    _cur,_ia=_reconcile_incumbent_system_library(_ig,_isy,_cur,_prev,dashboard_module=None,log_func=lambda *a,**k:None)
    _inc_ok=bool(_ia.get("revalidated_retained")==1 and _ia.get("restored_missing")==1 and len(_cur["spreads"].get("mechanism_families") or [])==1 and bool(_cur["spreads"]["mechanism_families"][0].get("confirmation_pass")))

    # V2.29 contract-hold regression: a previously confirmed/live exact rule with
    # 90 discovery rows is below today's generic >=100 admission floor. It must
    # remain frozen, not be mislabeled as a performance demotion.
    _rows_hold=[]
    for _sy,_n in ((2022,45),(2023,45),(2024,35),(2025,35)):
        for _i in range(_n):
            _win=(_i % 5)!=0
            _rows_hold.append({"Season":_sy,"Consensus_Open_Spread":3.0,"Consensus_Open_Total":50.0,"Actual_Margin":1.0 if _win else -5.0,"Actual_Total":50.0})
    _hg=pd.DataFrame(_rows_hold); _hsy=pd.to_numeric(_hg["Season"],errors="coerce").to_numpy(float)
    _hold_old={"mechanism_id":"NCAAF-MECH-CONTRACT-HOLD-TEST","market":"spreads","direction":"PLAY_ON","representative_conditions":["CURRENT_DOG"],"families":["MARKET_ROLE"],"evidence_family_key":"NCAAF-MECH-CONTRACT-HOLD-TEST","confirmation_pass":True,"confirmation_n":70,"confirmation_rate":.80,"live_authority_eligible":True,"current_qualified":True,"production_authority":0}
    _hold_prev={"source_tag":"prior","system_miner_v3":{"spreads":{"mechanism_families":[_hold_old]}},"system_library_inventory":{"records":[]},"system_library_change_audit":{}}
    _hold_cur={"spreads":{"mechanism_families":[],"special_family_audit":{}},"h2h":{"mechanism_families":[],"special_family_audit":{}},"totals":{"mechanism_families":[],"special_family_audit":{}}}
    _hold_cur,_hold_audit=_reconcile_incumbent_system_library(_hg,_hsy,_hold_cur,_hold_prev,dashboard_module=None,log_func=lambda *a,**k:None)
    _hold_mech=(_hold_cur["spreads"].get("mechanism_families") or [{}])[0]
    _contract_hold_ok=bool(_hold_audit.get("carried_contract_hold")==1 and _hold_audit.get("demoted")==0 and _hold_mech.get("incumbent_revalidation_status")=="LEGACY_CONTRACT_HOLD" and bool(_hold_mech.get("confirmation_pass")) and bool(_miner_live_authority_eligible(_hold_mech)))

    # A true historical performance failure still demotes. Discovery remains
    # strong, but both confirmation seasons fall below 50%.
    _rows_fail=[]
    for _sy,_n in ((2022,60),(2023,60),(2024,40),(2025,40)):
        for _i in range(_n):
            _win=(_i % 5)!=0 if _sy<=2023 else (_i % 5)<2
            _rows_fail.append({"Season":_sy,"Consensus_Open_Spread":3.0,"Consensus_Open_Total":50.0,"Actual_Margin":1.0 if _win else -5.0,"Actual_Total":50.0})
    _fg=pd.DataFrame(_rows_fail); _fsy=pd.to_numeric(_fg["Season"],errors="coerce").to_numpy(float)
    _fail_old={"mechanism_id":"NCAAF-MECH-TRUE-FAIL-TEST","market":"spreads","direction":"PLAY_ON","representative_conditions":["CURRENT_DOG"],"families":["MARKET_ROLE"],"evidence_family_key":"NCAAF-MECH-TRUE-FAIL-TEST","confirmation_pass":True,"confirmation_n":80,"confirmation_rate":.60,"live_authority_eligible":True,"current_qualified":True,"production_authority":0}
    _fail_prev={"source_tag":"prior","system_miner_v3":{"spreads":{"mechanism_families":[_fail_old]}},"system_library_inventory":{"records":[]},"system_library_change_audit":{}}
    _fail_cur={"spreads":{"mechanism_families":[],"special_family_audit":{}},"h2h":{"mechanism_families":[],"special_family_audit":{}},"totals":{"mechanism_families":[],"special_family_audit":{}}}
    _fail_cur,_fail_audit=_reconcile_incumbent_system_library(_fg,_fsy,_fail_cur,_fail_prev,dashboard_module=None,log_func=lambda *a,**k:None)
    _actual_failure_ok=bool(_fail_audit.get("demoted")==1 and _fail_audit.get("carried_contract_hold")==0 and len(_fail_cur["spreads"].get("mechanism_families") or [])==0)

    # Library arithmetic must reconcile exactly and distinguish genuinely new
    # confirmed rules from prior rows upgraded to confirmed.
    _prev_inv_test={"source_tag":"prior","system_library_inventory":{"records":[{"signature":"A","confirmation_pass":True,"live_authority_eligible":False},{"signature":"B","confirmation_pass":False,"live_authority_eligible":False}],"confirmed_mechanisms":1,"live_authority_mechanisms":0,"independent_live_family_votes":0},"system_library_change_audit":{"removed_confirmed":[{"signature":"D"}]}}
    _cur_inv_test={"records":[{"signature":"A","confirmation_pass":True,"live_authority_eligible":False},{"signature":"B","confirmation_pass":True,"live_authority_eligible":False},{"signature":"C","confirmation_pass":True,"live_authority_eligible":False},{"signature":"D","confirmation_pass":True,"live_authority_eligible":False}],"confirmed_mechanisms":4,"live_authority_mechanisms":0,"independent_live_family_votes":0}
    _delta_test=_ncaaf_system_library_delta(_cur_inv_test,_prev_inv_test)
    _reconciliation_ok=bool((_delta_test.get("reconciliation") or {}).get("status")=="PASS" and len(_delta_test.get("restored_confirmed") or [])==1 and len(_delta_test.get("newly_confirmed") or [])==1 and len(_delta_test.get("upgraded_to_confirmed") or [])==1)
    # V2.30 H2H/revenge/rivalry/travel regression. Use a non-RangeIndex to
    # prove context attachment is positional and does not depend on caller index labels.
    _ctx=pd.DataFrame({
        "Season":[2022,2023,2024],"Game_Date":["2022-11-26","2023-11-25","2024-11-30"],
        "Team_Norm":["alabama crimson tide","auburn tigers","alabama crimson tide"],
        "Opponent_Norm":["auburn tigers","alabama crimson tide","auburn tigers"],
        "Actual_Margin":[14.0,-7.0,3.0],"Consensus_Open_Spread":[-8.0,6.5,-10.0],
    },index=[101,205,309])
    _ctx2=_v230_game_context(_ctx)
    _v230_h2h_ok=bool(
        int(pd.to_numeric(_ctx2.iloc[0].get("H2H2_Prior_Meetings"),errors="coerce"))==0 and
        int(pd.to_numeric(_ctx2.iloc[1].get("Revenge_Flag"),errors="coerce"))==1 and
        float(pd.to_numeric(_ctx2.iloc[1].get("Last_Matchup_Margin"),errors="coerce"))==-14.0 and
        int(pd.to_numeric(_ctx2.iloc[2].get("Opp_Revenge_Flag"),errors="coerce"))==1 and
        int(pd.to_numeric(_ctx2.iloc[2].get("H2H2_Prior_Meetings"),errors="coerce"))==2 and
        int(pd.to_numeric(_ctx2.iloc[2].get("Rivalry_Flag"),errors="coerce"))==1 and
        _v230_is_rivalry("memphis tigers","uab blazers") and
        not _v230_is_rivalry("alabama crimson tide","georgia bulldogs")
    )
    _ctx_atom_df=pd.DataFrame({
        "Consensus_Open_Spread":[2.0],"Current_Spread":[-1.5],"Is_Home":[1],
        "Revenge_Flag":[1],"Opp_Revenge_Flag":[0],"Last_Matchup_Margin":[-10.0],
        "Opp_Last_Matchup_Margin":[10.0],"H2H2_Prior_Meetings":[3],"Revenge_Depth":[2],
        "H2H_Meetings_Since_Last_Win":[2],"H2H_Last_Loss_Margin":[-10.0],
        "Rivalry_Flag":[1],"Same_Season_Rematch":[0],"Prior_H2H_Home_Road_Flip":[1],
        "Opp_Prior_H2H_Home_Road_Flip":[1],"Team_Travel_Miles":[0.0],"Opp_Travel_Miles":[1250.0],
        "Team_Time_Zones_Crossed":[0.0],"Opp_Time_Zones_Crossed":[2.0],
        "Team_Travel_Eastward":[0],"Opp_Travel_Eastward":[1],"Team_Travel_Westward":[0],"Opp_Travel_Westward":[0],
        "Team_Back_To_Back_Road":[0],"Opp_Back_To_Back_Road":[1],"Team_Road_Games_Last3":[0],"Opp_Road_Games_Last3":[2],
        "Team_Road_Games_Last4":[0],"Opp_Road_Games_Last4":[3],"Team_Third_Road_In4":[0],"Opp_Third_Road_In4":[1],
        "Days_Since_Last_Game":[7],"Opp_Days_Since_Last_Game":[7],
    })
    _ctx_atoms={a["name"] for a in _extended_atoms(_ctx_atom_df,for_live=True,market="spreads")}
    _v230_atom_ok=all(x in _ctx_atoms for x in (
        "REVENGE","RIVALRY","H2H_2_PLUS_MEETINGS","REVENGE_DEPTH_2_PLUS",
        "OPP_TRAVEL_1000_PLUS","OPP_CROSSES_2_TZ_PLUS","OPP_BACK_TO_BACK_ROAD","OPP_THIRD_ROAD_IN4",
        "OPENED_DOG_NOW_FAVORITE","SPREAD_CROSSED_ZERO"
    ))
    # V2.31 deep-context atom regression. These are synthetic pregame fields
    # only; the test proves the Miner can express all four newly added families.
    _deep_atom_df=pd.DataFrame({
        "Consensus_Open_Spread":[-6.0],"Is_Home":[1],"Prev_SU_Margin":[7.0],"Opp_Prev_SU_Margin":[-3.0],
        "Team_Deep_Game_Count_Prior":[6],"Opp_Deep_Game_Count_Prior":[6],"Team_WinPct_Prior":[.83],"Opp_WinPct_Prior":[.80],
        "Team_RQ_Prev_Yardage_Margin":[-75.0],"Team_RQ_Prev_YPP_Margin":[-.8],"Team_RQ_Prev_First_Down_Margin":[-4],
        "Team_RQ_Prev_Turnover_Margin":[2],"Team_RQ_Prev_NonOffensive_TD_Margin":[1],"Team_RQ_Prev_Deceptive_Win_Score":[3],
        "Team_RQ_Prev_Win_Assistance_Score":[2],"Opp_RQ_Prev_YPP_Margin":[.4],"Opp_RQ_Prev_Deceptive_Loss_Score":[2],"Opp_RQ_Prev_Loss_Adversity_Score":[1],
        "Team_CTX_SOS_WinPct":[.60],"Opp_CTX_SOS_WinPct":[.42],"Team_CTX_OppAdj_NetYPP":[1.2],"Opp_CTX_OppAdj_NetYPP":[.1],
        "Team_CTX_Season_Off_YPP":[6.0],"Team_CTX_Recent3_Off_YPP":[6.7],"Team_CTX_Season_Def_YPP_Allowed":[5.4],"Team_CTX_Recent3_Def_YPP_Allowed":[4.7],
        "Team_CTX_Season_Net_YPP":[.6],"Team_CTX_Recent3_Net_YPP":[1.5],"Team_CTX_Recent5_Net_YPP":[1.2],
        "Team_CTX_Season_Rush_YPA":[5.3],"Team_CTX_Recent3_Rush_YPA":[6.2],"Team_CTX_Season_Pass_YPA":[7.5],"Team_CTX_Recent3_Pass_YPA":[8.7],
        "Team_CTX_Season_Off_PPG":[30.0],"Team_CTX_Recent3_Off_PPG":[39.0],"Team_CTX_Season_Def_PPG":[20.0],
        "Opp_CTX_Season_Off_YPP":[5.2],"Opp_CTX_Recent3_Off_YPP":[5.0],"Opp_CTX_Season_Def_YPP_Allowed":[5.1],"Opp_CTX_Recent3_Def_YPP_Allowed":[5.4],
        "Opp_CTX_Season_Net_YPP":[.1],"Opp_CTX_Recent3_Net_YPP":[-.1],"Opp_CTX_Recent5_Net_YPP":[0.0],
        "Opp_CTX_Season_Rush_YPA":[4.2],"Opp_CTX_Recent3_Rush_YPA":[4.1],"Opp_CTX_Season_Def_Rush_YPA_Allowed":[4.1],
        "Team_CTX_Season_Def_Rush_YPA_Allowed":[4.0],"Opp_CTX_Season_Pass_YPA":[6.8],"Opp_CTX_Recent3_Pass_YPA":[6.7],
        "Opp_CTX_Season_Def_Pass_YPA_Allowed":[6.2],"Team_CTX_Season_Def_Pass_YPA_Allowed":[6.3],
        "Opp_CTX_Season_Off_PPG":[24.0],"Opp_CTX_Season_Def_PPG":[24.0],
        "Team_CTX_Season_Sack_Allowed_Rate":[.04],"Team_CTX_Season_Def_Sack_Rate":[.08],"Opp_CTX_Season_Sack_Allowed_Rate":[.08],"Opp_CTX_Season_Def_Sack_Rate":[.08],
    })
    _deep_atoms={a["name"] for a in _extended_atoms(_deep_atom_df,for_live=True,market="spreads")}
    _v231_deep_atoms_ok=all(x in _deep_atoms for x in (
        "TEAM_OFF_DECEPTIVE_WIN_2PLUS","TEAM_OFF_RESULT_OVERPERFORMANCE",
        "TEAM_SOS_ADV_10P","TEAM_OPPADJ_NETYPP_ADV_075",
        "TEAM_RECENT3_NET_YPP_UP_075","TEAM_RECENT3_AND5_NET_YPP_UP",
        "TEAM_RUSH_MATCHUP_EDGE_075PLUS","TEAM_YPP_MATCHUP_EDGE_050PLUS","TEAM_MULTI_MATCHUP_EDGE_2PLUS"
    ))

    _cap_fake={"spreads":{"mechanism_families":[
        {"mechanism_id":"H1","families":["MATCHUP_HISTORY"]},{"mechanism_id":"H2","families":["MATCHUP_HISTORY"]},
        {"mechanism_id":"R1","families":["RIVALRY"]},{"mechanism_id":"T1","families":["TRAVEL_CONTEXT"]},
        {"mechanism_id":"T2","families":["ROAD_SEQUENCE"]},{"mechanism_id":"Q1","families":["RESULT_QUALITY_REGRESSION"]},
        {"mechanism_id":"S1","families":["SCHEDULE_RESUME_QUALITY"]},{"mechanism_id":"V1","families":["RECENT_VS_SEASON"]},
        {"mechanism_id":"M1","families":["MATCHUP_DIFFERENTIAL"]},
    ]}}
    _cap_diag=_apply_v231_context_family_caps(_cap_fake,log_func=lambda *a,**k:None)
    _cap_rows=_cap_fake["spreads"]["mechanism_families"]
    _v230_caps_ok=bool(
        _cap_diag.get("status")=="PASS" and _cap_rows[0].get("evidence_family_key")=="NCAAF_H2H_CONTEXT_FAMILY" and
        _cap_rows[1].get("evidence_family_key")=="NCAAF_H2H_CONTEXT_FAMILY" and
        _cap_rows[2].get("evidence_family_key")=="NCAAF_RIVALRY_CONTEXT_FAMILY" and
        _cap_rows[3].get("evidence_family_key")=="NCAAF_TRAVEL_CONTEXT_FAMILY" and
        _cap_rows[4].get("evidence_family_key")=="NCAAF_TRAVEL_CONTEXT_FAMILY" and
        _cap_rows[5].get("evidence_family_key")=="NCAAF_RESULT_QUALITY_FAMILY" and
        _cap_rows[6].get("evidence_family_key")=="NCAAF_SCHEDULE_RESUME_FAMILY" and
        _cap_rows[7].get("evidence_family_key")=="NCAAF_RECENT_VS_SEASON_FAMILY" and
        _cap_rows[8].get("evidence_family_key")=="NCAAF_MATCHUP_DIFFERENTIAL_FAMILY"
    )

    _sad=_special_atom_availability_audit(_tf,np.asarray([2025.0]),"spreads",dashboard_module=None)
    _special_diag_ok=bool(
        "SEASON_RECORD_STATE" in _sad and "ROLE_TRANSITION_BOUNCEBACK" in _sad and (_sad["SEASON_RECORD_STATE"].get("expected_atoms") or 0)>=24 and
        all(x in _sad for x in ("RESULT_QUALITY_REGRESSION","SCHEDULE_RESUME_QUALITY","RECENT_VS_SEASON","MATCHUP_DIFFERENTIAL"))
    )
    ok=bool(
        len(q)==3 and "RUN_PASS_MATCHUP" in fam and "MARKET_MICROSTRUCTURE" in fam and
        all("Actual_Margin" not in x for v in fam.values() for x in v) and
        "SU_SEQ3_LLL" in live_atoms and "OFF_ATS_MISS_7_PLUS" in live_atoms and
        "SU_WINLESS_PRIOR" in live_atoms and "ATS_COVERLESS_PRIOR" in live_atoms and "SU_AND_ATS_WINLESS_AFTER_4_PLUS" in live_atoms and
        "OPP_SU_WINLESS_PRIOR" in live_atoms and "OPP_ATS_COVERLESS_PRIOR" in live_atoms and "OPP_SU_AND_ATS_WINLESS_AFTER_4_PLUS" in live_atoms and
        "TEAM_GAME_5" in live_atoms and
        "BOUNCEBACK_HOME_FAV_UPSET_TO_ROAD_DOG" in live_atoms and "BOUNCEBACK_WINNING_TEAM" in live_atoms and "BOUNCEBACK_BOTH_WINNING_TEAMS" in live_atoms and "BOUNCEBACK_WINNING_TEAM_DOG_5_TO_7P5" in live_atoms and
        "OFF_SHUTOUT_LOSS" in live_atoms and "TEAM_ATS_WINPCT_LE_250" in live_atoms and "OPP_OFF_UPSET_WIN" in live_atoms and "SPREAD_MOVED_TOWARD_TEAM_2_PLUS" in live_atoms and
        "BOTH_OFF_SU_LOSS" in total_live_atoms and "TOTAL_MOVED_DOWN_2_PLUS" in total_live_atoms and "TOTAL_58_PLUS" in total_live_atoms and "EITHER_OFF_ALLOWED_35_PLUS" in total_live_atoms and
        bool(_obs_norm and _obs_norm.get("rating")==2 and _obs_norm.get("outcome_used_for_selection") is False and "outcome" not in _obs_norm and _obs_report.get("rating_used_for_system_qualification") is False) and
        "EXPERT_PATHI_FB_DOG_HOOK_ABOVE_3" in live_atoms and "EXPERT_BIGAL_CF2_LATESEASONREVENGEDOG" in live_atoms and
        "CORE_OOF_EDGE_TEAM_2PLUS" in live_atoms and "SPEC_STRUCTURED_STATS_CORE_DIVERGENCE" in live_atoms and "META_PT_EDGE_TEAM_3PLUS" in live_atoms and "META_PT_CORE_STRONG_AGREE" in live_atoms and
        _inc_ok and _contract_hold_ok and _actual_failure_ok and _reconciliation_ok and _special_diag_ok and _v230_h2h_ok and _v230_atom_ok and _v230_caps_ok and _v231_deep_atoms_ok and np.allclose(ret,np.asarray([2.0,.5]),equal_nan=False) and _pt_name_safe and _relay_csv_ok and _relay_md_ok and _relay_url_ok and _challenge_rejected and _merge_rematch_ok and _sparse_pt_ok and _pt_alias_hint_ok and _pt_final_hard_aliases_ok and _expert_bridge_ok and _hist_state_bridge_ok and _occ_recon_ok and
        _miner_live_authority_eligible({"confirmation_pass":True,"confirmation_n":90,"confirmation_rate":0.60,"representative_conditions":["META_PT_EDGE_TEAM_3PLUS"]}) and
        not _miner_live_authority_eligible({"confirmation_pass":True,"confirmation_n":90,"confirmation_rate":0.60,"representative_conditions":["CORE_OOF_EDGE_TEAM_2PLUS"]})
    )
    return {
        "status":"PASS" if ok else "FAIL","source_tag":NCAAF_RESEARCH_V2_SOURCE_TAG,
        "incumbent_library_revalidation":_inc_ok,"incumbent_legacy_contract_hold":_contract_hold_ok,"incumbent_true_failure_demotes":_actual_failure_ok,"library_reconciliation":_reconciliation_ok,"special_family_diagnostics":_special_diag_ok,
        "v230_h2h_rivalry_context":_v230_h2h_ok,"v230_context_atoms":_v230_atom_ok,"v231_family_caps":_v230_caps_ok,"v231_deep_context_atoms":_v231_deep_atoms_ok,
        "families":sorted(fam),"qvalues":q.tolist(),"american_unit_profit_test":ret.tolist(),
        "h2h_price_gate":"OBSERVED_TEAM_AND_OPPONENT_ML_ONLY",
        "confirmation_gate":"BOTH_2024_AND_2025",
        "live_authority_policy":NCAAF_MINER_LIVE_AUTHORITY_POLICY,
        "live_min_confirmation_n":NCAAF_MINER_LIVE_MIN_CONFIRMATION_N,
        "live_min_confirmation_rate":NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE,
        "weak_confirmed_authority":_miner_live_authority_eligible({"confirmation_pass":True,"confirmation_n":109,"confirmation_rate":0.5229}),
        "strong_confirmed_authority":_miner_live_authority_eligible({"confirmation_pass":True,"confirmation_n":90,"confirmation_rate":0.6222}),
        "pt_strong_validated_authority":_miner_live_authority_eligible({"confirmation_pass":True,"confirmation_n":90,"confirmation_rate":0.60,"representative_conditions":["META_PT_EDGE_TEAM_3PLUS"]}),
        "pt_weak_authority":_miner_live_authority_eligible({"confirmation_pass":True,"confirmation_n":90,"confirmation_rate":0.54,"representative_conditions":["META_PT_EDGE_TEAM_3PLUS"]}),
        "core_oof_bridge_authority":_miner_live_authority_eligible({"confirmation_pass":True,"confirmation_n":90,"confirmation_rate":0.60,"representative_conditions":["CORE_OOF_EDGE_TEAM_2PLUS"]}),
        "pt_family_vote_cap":1,
        "season_record_state_family_vote_cap":1,
        "role_transition_bounceback_family_vote_cap":1,
        "h2h_context_family_vote_cap":1,"rivalry_context_family_vote_cap":1,"travel_context_family_vote_cap":1,
        "result_quality_family_vote_cap":1,"schedule_resume_family_vote_cap":1,"recent_vs_season_family_vote_cap":1,"matchup_differential_family_vote_cap":1,
        "role_transition_bounceback_atoms":sorted(x for x in live_atoms if "BOUNCEBACK" in x),
        "season_record_state_atoms":sorted(x for x in live_atoms if x.startswith(("SU_WINLESS","ATS_COVERLESS","SU_AND_ATS_WINLESS","OPP_SU_WINLESS","OPP_ATS_COVERLESS","OPP_SU_AND_ATS_WINLESS","TEAM_GAME_","DOG_6P5_PLUS","DOG_7_PLUS","DOG_8_PLUS","DOG_9_PLUS","DOG_10_PLUS","OPP_HAS_SU_WIN"))),
        "big_al_observation_spread_atoms":sorted(x for x in live_atoms if x.startswith(("OFF_SHUTOUT","TEAM_ATS_WINPCT","OPP_OFF_UPSET","SPREAD_MOVED_"))),
        "big_al_observation_total_atoms":sorted(x for x in total_live_atoms if x.startswith(("BOTH_OFF_SU_LOSS","TOTAL_MOVED_","TOTAL_55_PLUS","TOTAL_58_PLUS","EITHER_OFF_","BOTH_OFF_ALLOWED"))),
        "big_al_rating_min_for_pattern_analysis":BIG_AL_RATING_MIN_FOR_PATTERN_ANALYSIS,
        "big_al_rating_authority":0,
        "pt_name_safe_header_contract":_pt_name_safe,
        "pt_self_test_system_columns":_pt_da.get("system_columns",{}),
        "pt_fuzzy_header_rejected":_pt_dc.get("system_columns",{}).get("ESPN_FPI") is None,
        "pt_relay_csv_payload":_relay_csv_ok,"pt_relay_named_markdown":_relay_md_ok,"pt_relay_https_target":_relay_url_ok,"pt_challenge_rejected":_challenge_rejected,"pt_current_merge_preserves_rematch":_merge_rematch_ok,"pt_sparse_partial_component_preservation":_sparse_pt_ok,"pt_canonical_alias_hint":_pt_alias_hint_ok,"pt_final_hard_aliases":_pt_final_hard_aliases_ok,"expert_side_bridge":_expert_bridge_ok,"historical_state_context_bridge":_hist_state_bridge_ok,"expert_occurrence_reconciliation":_occ_recon_ok,
    }


if __name__ == "__main__":
    print(json.dumps(self_test(),indent=2,default=str))
