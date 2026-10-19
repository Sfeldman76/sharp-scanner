#!/usr/bin/env python3
"""NCAAF CFBD Warehouse Loader V2.

Purpose
-------
Backfill and refresh leakage-safe CollegeFootballData (CFBD) features for the
NCAAF System Miner without changing frozen Production V1 probabilities.

Security
--------
The CFBD API token is read ONLY from the CFBD_API_KEY environment variable.
Never hard-code the token in this file, Cloud Run source, or Git.

Data policy
-----------
* Game-derived features are shifted one completed game before they can appear on
  a pregame row. Current-game box score / advanced stats never appear as inputs.
* Preseason/static team-season context (returning production, roster continuity,
  portal flow, recruiting, talent) may join directly to that season.
* Elo uses the stored pregame Elo carried on each CFBD game record.
* Portal flow is cut off at the season's first kickoff so later portal activity
  cannot leak backward into early-season Miner training.
* Historical observed weather is retained as a context field and tagged as such.
  Use it only if your betting decision horizon is near kickoff; it is not a
  historical forecast snapshot.
* Historical play-by-play is converted to completed-game team metrics and then
  shifted one completed game before use (field position, drive efficiency,
  early/passing-down efficiency, sack/dropback rate, explosive-play rate).
* Season-level SP+, FPI, SRS, and CORE are used only as PRIOR-SEASON static
  context for the next season. Same-season retrospective ratings never qualify
  historical Miner systems, avoiding look-ahead from season-final estimates.

Warehouse outputs
-----------------
Existing/reused:
  sharplogger.sharp_data.ncaaf_historical_game_side_raw
  sharplogger.sharp_data.ncaaf_game_travel_reference
  sharplogger.sharp_data.ncaaf_historical_core_training_vw (read unchanged)
New permanent sidecars:
  sharplogger.sharp_data.ncaaf_cfbd_api_archive
  sharplogger.sharp_data.ncaaf_team_season_personnel
  sharplogger.sharp_data.ncaaf_player_season_history
Build/staging:
  sharplogger.sharp_data.ncaaf_cfbd_miner_features_v1

Typical first backfill
----------------------
  export CFBD_API_KEY='...'
  python ncaaf_cfbd_warehouse_loader_v2.py --seasons 2022-2026 --history-seasons 2013-2021

Weekly refresh
--------------
  python ncaaf_cfbd_warehouse_loader_v2.py --seasons 2026 --history-seasons ''
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.collegefootballdata.com"
DEFAULT_PROJECT = os.getenv("BQ_PROJECT", "sharplogger")
DEFAULT_DATASET = os.getenv("BQ_DATASET", "sharp_data")
SOURCE_TAG = "ncaaf-cfbd-warehouse-v2.0-20261015"

# Existing canonical warehouse objects — deliberately reused.
CANONICAL_RAW_TABLE = os.getenv("NCAAF_HISTORICAL_RAW_TABLE", "ncaaf_historical_game_side_raw")
TRAIN_VIEW = os.getenv("NCAAF_HISTORICAL_TRAIN_VIEW", "ncaaf_historical_core_training_vw")
TRAVEL_TABLE = os.getenv("NCAAF_TRAVEL_REFERENCE_TABLE", "ncaaf_game_travel_reference")

# Only genuinely new source domains get new permanent objects.
API_ARCHIVE_TABLE = os.getenv("CFBD_API_ARCHIVE_TABLE", "ncaaf_cfbd_api_archive")
PERSONNEL_TABLE = os.getenv("CFBD_PERSONNEL_TABLE", "ncaaf_team_season_personnel")
PLAYER_HISTORY_TABLE = os.getenv("CFBD_PLAYER_HISTORY_TABLE", "ncaaf_player_season_history")
CFBD_HISTORY_STAGING_TABLE = os.getenv("CFBD_HISTORY_STAGING_TABLE", "ncaaf_cfbd_historical_game_side_staging")

# This remains a build/staging feature table consumed fail-closed by Research V2.
DEFAULT_TABLE = os.getenv("CFBD_MINER_TABLE", "ncaaf_cfbd_miner_features_v1")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _team_key(x: Any) -> str:
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    replacements = {
        "st": "state", "mich": "michigan", "fla": "florida", "ill": "illinois",
        "ind": "indiana", "mass": "massachusetts", "miss": "mississippi",
        "mo": "missouri", "tenn": "tennessee", "tex": "texas", "va": "virginia",
        "wash": "washington", "wis": "wisconsin", "conn": "connecticut",
        "colo": "colorado", "ark": "arkansas", "cal": "california", "neb": "nebraska",
    }
    toks = [replacements.get(t, t) for t in s.split()]
    s = " ".join(toks)
    aliases = {
        "nc state": "north carolina state",
        "n c state": "north carolina state",
        "sc state": "south carolina state",
        "s c state": "south carolina state",
        "e michigan": "eastern michigan",
        "w michigan": "western michigan",
        "c michigan": "central michigan",
        "n illinois": "northern illinois",
        "s mississippi": "southern mississippi",
        "southern miss": "southern mississippi",
    }
    return aliases.get(s, s)


def _person_key(x: Any) -> str:
    """Conservative player-name key; unlike team keys, do not expand abbreviations."""
    s = "" if x is None else str(x)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _num(s: pd.Series | Iterable[Any]) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _first_col(df: pd.DataFrame, *names: str, default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def _bool_col(df: pd.DataFrame, *names: str) -> pd.Series:
    s = _first_col(df, *names, default=np.nan)
    if s.dtype == bool:
        return s.astype(float)
    txt = s.astype(str).str.lower().str.strip()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[txt.isin({"true", "1", "yes", "y"})] = 1.0
    out.loc[txt.isin({"false", "0", "no", "n"})] = 0.0
    n = pd.to_numeric(s, errors="coerce")
    out = out.where(out.notna(), n)
    return out


def _parse_seasons(spec: str) -> list[int]:
    out: set[int] = set()
    for part in str(spec).split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(p))
    return sorted(out)


def _json_frame(rows: Any) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("results", rows))
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return pd.DataFrame()
    try:
        return pd.json_normalize(rows, sep=".")
    except Exception:
        return pd.DataFrame(rows)


@dataclass
class PullResult:
    data: list[dict[str, Any]]
    path: str
    params: dict[str, Any]
    optional: bool = False


class CFBDClient:
    def __init__(self, api_key: str | None = None, timeout: int = 45, max_attempts: int = 5):
        self.api_key = api_key or os.getenv("CFBD_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("CFBD_API_KEY is not set")
        self.timeout = int(timeout)
        self.max_attempts = int(max_attempts)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "sharp-ncaaf-cfbd-miner-loader/1.0",
        })
        self.call_count = 0
        self.archive_rows: list[dict[str, Any]] = []
        self.request_health: dict[tuple[str, int | None], dict[str, Any]] = {}
        self.api_failures: list[dict[str, Any]] = []

    def _track_request(self, path: str, params: dict[str, Any], status: str, count: int = 0, http_status: int | None = None):
        year = params.get("year")
        try: year = int(year) if year is not None else None
        except (TypeError, ValueError): year = None
        key=(path,year)
        stat=self.request_health.setdefault(key,{"path":path,"season":year,"requests":0,"rows":0,"failures":0,"statuses":{}})
        stat["requests"]+=1
        stat["rows"]+=int(count)
        stat["statuses"][status]=int(stat["statuses"].get(status,0))+1
        if status!="PASS":
            stat["failures"]+=1
            self.api_failures.append({"path":path,"season":year,"status":status,"http_status":http_status})

    def get(self, path: str, params: dict[str, Any] | None = None, *, optional: bool = False) -> list[dict[str, Any]]:
        """Fetch and classify each endpoint/season; absent data != failed API call."""
        url = BASE_URL + path
        params = {k: v for k, v in (params or {}).items() if v is not None}
        last: Exception | None = None
        http_status: int | None = None
        for attempt in range(self.max_attempts):
            try:
                self.call_count += 1
                response = self.session.get(url, params=params, timeout=self.timeout)
                http_status = response.status_code
                # Auth/plan access and invalid endpoint are not transient. Do not
                # spend the Tier 2 request budget retrying them five times.
                if http_status in (400, 401, 403, 404, 422):
                    last = RuntimeError(f"CFBD_HTTP_{http_status}")
                    break
                if http_status == 429:
                    last = RuntimeError("CFBD_HTTP_429_RATE_LIMIT")
                    if attempt+1 < self.max_attempts:
                        retry_after = response.headers.get("Retry-After", "")
                        try: delay = min(60., max(0., float(retry_after)))
                        except (TypeError, ValueError): delay = min(30., float(2**attempt))
                        time.sleep(delay)
                    continue
                if 500 <= http_status < 600:
                    last = RuntimeError(f"CFBD_HTTP_{http_status}")
                    if attempt+1 < self.max_attempts: time.sleep(min(15, 1.5**attempt))
                    continue
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, list): rows=payload
                elif isinstance(payload, dict) and isinstance(payload.get("data"), list): rows=payload["data"]
                elif isinstance(payload, dict): rows=[payload]
                else: rows=[]
                self._archive_success(path,params,rows)
                self._track_request(path,params,"PASS",len(rows),http_status)
                return rows
            except Exception as exc:
                last=exc
                # An HTTP terminal status already classified above must not be
                # retried or put through a second backoff path.
                if attempt+1 < self.max_attempts: time.sleep(min(15, 1.5**attempt))
        status=(f"HTTP_{http_status}" if http_status else type(last).__name__)
        self._track_request(path,params,status,0,http_status)
        if optional:
            print(f"[CFBD-SOURCE-STATUS] path={path} year={params.get('year')} status=UNAVAILABLE category={status} optional=TRUE data_not_verified=TRUE")
            return []
        raise RuntimeError(f"CFBD_REQUIRED_SOURCE_FAILED path={path} year={params.get('year')} category={status}: {last}")

    def _archive_success(self, path: str, params: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        """Retain immutable row-level API snapshots without persisting credentials."""
        retrieved = datetime.now(timezone.utc).isoformat()
        pjson = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
        season = params.get("year") if isinstance(params, dict) else None
        batch_basis = f"{path}|{pjson}|{retrieved}"
        batch_id = hashlib.sha256(batch_basis.encode("utf-8")).hexdigest()
        for i, row in enumerate(rows):
            payload_json = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
            content_hash = hashlib.sha256(f"{path}|{pjson}|{payload_json}".encode("utf-8")).hexdigest()
            self.archive_rows.append({
                "Archive_Row_ID": content_hash,
                "Archive_Batch_ID": batch_id,
                "Endpoint": path,
                "Params_JSON": pjson,
                "Season": int(season) if season is not None and str(season).isdigit() else None,
                "Retrieved_At": retrieved,
                "Row_Index": int(i),
                "Payload_JSON": payload_json,
                "Source_Tag": SOURCE_TAG,
            })


# ---------------------------------------------------------------------------
# Pulls
# ---------------------------------------------------------------------------

def pull_games(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/games", {"year": year, "seasonType": "both", "classification": "fbs"}))


def pull_lines(c: CFBDClient, year: int) -> pd.DataFrame:
    """Historical betting lines. Retain every provider in the API archive; normalize a consensus later."""
    return _json_frame(c.get("/lines", {"year": year, "seasonType": "both"}, optional=True))


def pull_advanced(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/stats/game/advanced", {
        "year": year, "seasonType": "both", "excludeGarbageTime": "true"
    }))


def pull_havoc(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/stats/game/havoc", {"year": year, "seasonType": "both"}, optional=True))


def pull_returning(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/player/returning", {"year": year}, optional=True))


def pull_portal(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/player/portal", {"year": year}, optional=True))


def pull_roster(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/roster", {"year": year, "classification": "fbs"}, optional=True))


def pull_talent(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/talent", {"year": year}, optional=True))


def pull_recruiting_team(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/recruiting/teams", {"year": year}, optional=True))


def pull_player_season_ppa(c: CFBDClient, year: int) -> pd.DataFrame:
    """Prior-season player production used only to value portal movement."""
    return _json_frame(c.get("/ppa/players/season", {
        "year": year, "excludeGarbageTime": "true"
    }, optional=True))


def pull_player_usage(c: CFBDClient, year: int) -> pd.DataFrame:
    """Prior-season player usage used only to value portal movement."""
    return _json_frame(c.get("/player/usage", {
        "year": year, "excludeGarbageTime": "true"
    }, optional=True))


def pull_weather(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/games/weather", {
        "year": year, "seasonType": "both", "classification": "fbs"
    }, optional=True))


def pull_venues(c: CFBDClient) -> pd.DataFrame:
    return _json_frame(c.get("/venues", optional=True))


def pull_fbs_teams(c: CFBDClient, year: int) -> pd.DataFrame:
    """Season-specific FBS team/home-location reference for altitude context."""
    return _json_frame(c.get("/teams/fbs", {"year": year}, optional=True))


def pull_qb_ppa_games(c: CFBDClient, year: int, weeks: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week in sorted({int(w) for w in weeks if pd.notna(w) and int(w) >= 0}):
        part = c.get("/ppa/players/games", {
            "year": year, "week": week, "seasonType": "both", "position": "QB",
            "excludeGarbageTime": "true"
        }, optional=True)
        rows.extend(part)
    return _json_frame(rows)


def pull_qb_success_games(c: CFBDClient, year: int, weeks: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week in sorted({int(w) for w in weeks if pd.notna(w) and int(w) >= 0}):
        part = c.get("/stats/player/success/game", {
            "year": year, "week": week, "seasonType": "both",
            "excludeGarbageTime": "true"
        }, optional=True)
        # Endpoint has no position query parameter; retain QBs locally.
        for z in part:
            if str(z.get("position") or "").upper().strip() == "QB":
                rows.append(z)
    return _json_frame(rows)


def pull_elo_snapshots(c: CFBDClient, year: int, weeks: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Preseason is the only safe source for Week 1.
    pre = c.get("/ratings/elo", {"year": year, "seasonType": "regular", "preseason": "true"}, optional=True)
    for z in pre:
        z = dict(z); z["snapshot_week"] = 0; rows.append(z)
    # A Week N snapshot is used beginning with Week N+1.
    for week in sorted({int(w) for w in weeks if pd.notna(w) and int(w) >= 1}):
        part = c.get("/ratings/elo", {"year": year, "week": week, "seasonType": "regular"}, optional=True)
        for z in part:
            z = dict(z); z["snapshot_week"] = week; rows.append(z)
    return _json_frame(rows)


def pull_plays(c: CFBDClient, year: int, weeks: Iterable[int]) -> pd.DataFrame:
    """Historical CFBD play-by-play, pulled by week to stay within endpoint limits."""
    rows: list[dict[str, Any]] = []
    for week in sorted({int(w) for w in weeks if pd.notna(w) and int(w) >= 0}):
        part = c.get("/plays", {
            "year": year, "week": week, "seasonType": "both", "classification": "fbs"
        }, optional=True)
        rows.extend(part)
    return _json_frame(rows)


def pull_sp_ratings(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/ratings/sp", {"year": year}, optional=True))


def pull_srs_ratings(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/ratings/srs", {"year": year}, optional=True))


def pull_fpi_ratings(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/ratings/fpi", {"year": year}, optional=True))


def pull_core_ratings(c: CFBDClient, year: int) -> pd.DataFrame:
    """Retrospective CORE; only prior-season aggregates are exposed to a target season."""
    return _json_frame(c.get("/ratings/core", {"year": year}, optional=True))


# ---------------------------------------------------------------------------
# Static / preseason feature engineering
# ---------------------------------------------------------------------------

def build_returning_features(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Season", "Team_Key"])
    x = df.copy()
    x["Season"] = _num(_first_col(x, "season", "year")).fillna(year).astype("Int64")
    x["Team"] = _first_col(x, "team", "school").astype(str)
    x["Team_Key"] = x["Team"].map(_team_key)
    mapping = {
        "CFBD_Returning_Total_PPA": ("totalPPA", "total_ppa"),
        "CFBD_Returning_Pass_PPA": ("totalPassingPPA", "total_passing_ppa"),
        "CFBD_Returning_Receive_PPA": ("totalReceivingPPA", "total_receiving_ppa"),
        "CFBD_Returning_Rush_PPA": ("totalRushingPPA", "total_rushing_ppa"),
        "CFBD_Returning_Pct_PPA": ("percentPPA", "percent_ppa"),
        "CFBD_Returning_Pct_Pass_PPA": ("percentPassingPPA", "percent_passing_ppa"),
        "CFBD_Returning_Pct_Receive_PPA": ("percentReceivingPPA", "percent_receiving_ppa"),
        "CFBD_Returning_Pct_Rush_PPA": ("percentRushingPPA", "percent_rushing_ppa"),
        "CFBD_Returning_Usage": ("usage",),
        "CFBD_Returning_Pass_Usage": ("passingUsage", "passing_usage"),
        "CFBD_Returning_Receive_Usage": ("receivingUsage", "receiving_usage"),
        "CFBD_Returning_Rush_Usage": ("rushingUsage", "rushing_usage"),
    }
    out = x[["Season", "Team", "Team_Key"]].copy()
    for dst, srcs in mapping.items():
        out[dst] = _num(_first_col(x, *srcs))
    return out.drop_duplicates(["Season", "Team_Key"], keep="last")


def _player_identity(df: pd.DataFrame) -> pd.Series:
    pid = _first_col(df, "id", "athleteId", "playerId").astype(str).replace({"nan": "", "None": ""})
    name = (_first_col(df, "firstName").astype(str) + " " + _first_col(df, "lastName").astype(str)).str.strip()
    name = name.map(_team_key)
    return pid.where(pid.str.len() >= 2, "NAME:" + name)


def build_roster_continuity(cur: pd.DataFrame, prev: pd.DataFrame, year: int) -> pd.DataFrame:
    if cur.empty:
        return pd.DataFrame(columns=["Season", "Team_Key"])
    c = cur.copy(); p = prev.copy()
    for d in (c, p):
        d["Team"] = _first_col(d, "team", "school").astype(str)
        d["Team_Key"] = d["Team"].map(_team_key)
        d["Player_Key"] = _player_identity(d)
        d["Position"] = _first_col(d, "position").astype(str).str.upper().str.strip()
    psets = p.groupby("Team_Key")["Player_Key"].agg(lambda s: set(v for v in s if v and v != "NAME:")) if not p.empty else {}

    groups = {
        "ALL": set(),
        "QB": {"QB"},
        "OL": {"OL", "OT", "OG", "G", "C", "OC"},
        "SKILL": {"RB", "FB", "WR", "TE"},
        "FRONT7": {"DL", "DE", "DT", "NT", "LB", "ILB", "OLB", "EDGE"},
        "SECONDARY": {"DB", "CB", "S", "FS", "SS"},
    }
    rows = []
    for team_key, g in c.groupby("Team_Key", sort=False):
        prevset = psets.get(team_key, set()) if hasattr(psets, "get") else set()
        rec: dict[str, Any] = {"Season": year, "Team_Key": team_key, "Team": str(g["Team"].iloc[0])}
        for label, positions in groups.items():
            gg = g if label == "ALL" else g[g["Position"].isin(positions)]
            ids = [v for v in gg["Player_Key"] if v and v != "NAME:"]
            n = len(set(ids)); retained = len(set(ids) & set(prevset))
            rec[f"CFBD_Roster_{label}_Count"] = n
            rec[f"CFBD_Roster_{label}_Returning_Count"] = retained
            rec[f"CFBD_Roster_{label}_Retention"] = (retained / n) if n else np.nan
        rec["CFBD_QB_Returning_Flag"] = float(rec.get("CFBD_Roster_QB_Returning_Count", 0) >= 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def _prior_player_value_table(ppa: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    """Normalize previous-season player PPA/usage for portal value matching."""
    parts=[]
    if ppa is not None and not ppa.empty:
        x=ppa.copy(); z=pd.DataFrame(index=x.index)
        z["Team_Key"]=_first_col(x,"team").map(_team_key)
        z["Player_Key"]=_first_col(x,"name").map(_person_key)
        z["Position"]=_first_col(x,"position").astype(str).str.upper().str.strip()
        z["Prior_Total_PPA"]=_num(_first_col(x,"totalPPA.all","totalPPA.total","total_ppa_all"))
        z["Prior_Avg_PPA"]=_num(_first_col(x,"averagePPA.all","averagePPA.total","average_ppa_all"))
        parts.append(z)
    base=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=["Team_Key","Player_Key","Position","Prior_Total_PPA","Prior_Avg_PPA"])
    if usage is not None and not usage.empty:
        u=usage.copy(); uz=pd.DataFrame(index=u.index)
        uz["Team_Key"]=_first_col(u,"team").map(_team_key)
        uz["Player_Key"]=_first_col(u,"name").map(_person_key)
        uz["Prior_Usage"]=_num(_first_col(u,"usage.overall","usage_all","overall"))
        if base.empty:
            base=uz
        else:
            base=base.merge(uz.drop_duplicates(["Team_Key","Player_Key"],keep="last"),on=["Team_Key","Player_Key"],how="outer")
    if "Prior_Usage" not in base.columns: base["Prior_Usage"]=np.nan
    for c in ("Prior_Total_PPA","Prior_Avg_PPA"):
        if c not in base.columns: base[c]=np.nan
    if "Position" not in base.columns: base["Position"]=""
    return base.loc[base["Team_Key"].ne("") & base["Player_Key"].ne("")].drop_duplicates(["Team_Key","Player_Key"],keep="last")


def build_portal_features(df: pd.DataFrame, year: int, prior_player_ppa: pd.DataFrame | None = None,
                          prior_player_usage: pd.DataFrame | None = None, season_start: pd.Timestamp | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Season", "Team_Key"])
    x = df.copy()
    x["origin_key"] = _first_col(x, "origin").map(_team_key)
    x["destination_key"] = _first_col(x, "destination").map(_team_key)
    x["player_key"] = (_first_col(x,"firstName").astype(str).str.strip()+" "+_first_col(x,"lastName").astype(str).str.strip()).map(_person_key)
    x["rating_num"] = _num(_first_col(x, "rating"))
    x["stars_num"] = _num(_first_col(x, "stars"))
    x["position_u"] = _first_col(x, "position").astype(str).str.upper().str.strip()
    x["eligibility_u"] = _first_col(x,"eligibility").astype(str).str.upper().str.strip()
    x["transfer_date"] = pd.to_datetime(_first_col(x,"transferDate"),errors="coerce",utc=True)
    # A withdrawn portal entry is not a roster departure/arrival.  For historical
    # qualification we also require a known transaction date before the season's
    # first kickoff; unknown/later transactions fail closed rather than leak.
    x=x.loc[~x["eligibility_u"].eq("WITHDRAWN")].copy()
    if season_start is not None and pd.notna(season_start):
        cutoff=pd.Timestamp(season_start)
        if cutoff.tzinfo is None: cutoff=cutoff.tz_localize("UTC")
        else: cutoff=cutoff.tz_convert("UTC")
        x=x.loc[x["transfer_date"].notna() & x["transfer_date"].lt(cutoff)].copy()

    pv=_prior_player_value_table(prior_player_ppa if prior_player_ppa is not None else pd.DataFrame(),
                                 prior_player_usage if prior_player_usage is not None else pd.DataFrame())
    if not pv.empty:
        j=pv.rename(columns={"Team_Key":"origin_key","Player_Key":"player_key","Position":"prior_position"})
        x=x.merge(j[["origin_key","player_key","Prior_Total_PPA","Prior_Avg_PPA","Prior_Usage"]],on=["origin_key","player_key"],how="left")
    else:
        x["Prior_Total_PPA"]=np.nan; x["Prior_Avg_PPA"]=np.nan; x["Prior_Usage"]=np.nan

    teams = sorted((set(x["origin_key"]) | set(x["destination_key"])) - {""})
    rows = []
    for tk in teams:
        inc = x[x["destination_key"].eq(tk)]; out = x[x["origin_key"].eq(tk)]
        def avg(s):
            z = _num(s).dropna(); return float(z.mean()) if len(z) else np.nan
        def mx(s):
            z = _num(s).dropna(); return float(z.max()) if len(z) else np.nan
        def sm(s):
            z = _num(s).dropna(); return float(z.sum()) if len(z) else np.nan
        def flow_sum(frame, col):
            if frame is None or len(frame)==0:
                return 0.0
            z=_num(frame[col]).dropna()
            return float(z.sum()) if len(z) else np.nan
        rec = {
            "Season": year, "Team_Key": tk,
            "CFBD_Portal_In_Count": int(len(inc)),
            "CFBD_Portal_Out_Count": int(len(out)),
            "CFBD_Portal_Net_Count": int(len(inc) - len(out)),
            "CFBD_Portal_In_Avg_Rating": avg(inc["rating_num"]),
            "CFBD_Portal_Out_Avg_Rating": avg(out["rating_num"]),
            "CFBD_Portal_In_Max_Rating": mx(inc["rating_num"]),
            "CFBD_Portal_Out_Max_Rating": mx(out["rating_num"]),
            "CFBD_Portal_In_4StarPlus": int((inc["stars_num"] >= 4).sum()),
            "CFBD_Portal_Out_4StarPlus": int((out["stars_num"] >= 4).sum()),
            "CFBD_Portal_In_QB_Count": int(inc["position_u"].eq("QB").sum()),
            "CFBD_Portal_Out_QB_Count": int(out["position_u"].eq("QB").sum()),
            "CFBD_Portal_In_QB_Best_Rating": mx(inc.loc[inc["position_u"].eq("QB"), "rating_num"]),
            "CFBD_Portal_Out_QB_Best_Rating": mx(out.loc[out["position_u"].eq("QB"), "rating_num"]),
            "CFBD_Portal_In_Prior_TotalPPA": flow_sum(inc,"Prior_Total_PPA"),
            "CFBD_Portal_Out_Prior_TotalPPA": flow_sum(out,"Prior_Total_PPA"),
            "CFBD_Portal_In_Prior_Usage": flow_sum(inc,"Prior_Usage"),
            "CFBD_Portal_Out_Prior_Usage": flow_sum(out,"Prior_Usage"),
            "CFBD_Portal_In_QB_Prior_TotalPPA": flow_sum(inc.loc[inc["position_u"].eq("QB")],"Prior_Total_PPA"),
            "CFBD_Portal_Out_QB_Prior_TotalPPA": flow_sum(out.loc[out["position_u"].eq("QB")],"Prior_Total_PPA"),
            "CFBD_Portal_Prior_Value_Matched_In": int(inc["Prior_Total_PPA"].notna().sum()),
            "CFBD_Portal_Prior_Value_Matched_Out": int(out["Prior_Total_PPA"].notna().sum()),
        }
        rec["CFBD_Portal_Net_Rating"] = rec["CFBD_Portal_In_Avg_Rating"] - rec["CFBD_Portal_Out_Avg_Rating"] if np.isfinite(rec["CFBD_Portal_In_Avg_Rating"]) and np.isfinite(rec["CFBD_Portal_Out_Avg_Rating"]) else np.nan
        rec["CFBD_Portal_Net_Prior_TotalPPA"] = (rec["CFBD_Portal_In_Prior_TotalPPA"] - rec["CFBD_Portal_Out_Prior_TotalPPA"]) if np.isfinite(rec["CFBD_Portal_In_Prior_TotalPPA"]) and np.isfinite(rec["CFBD_Portal_Out_Prior_TotalPPA"]) else np.nan
        rec["CFBD_Portal_Net_Prior_Usage"] = (rec["CFBD_Portal_In_Prior_Usage"] - rec["CFBD_Portal_Out_Prior_Usage"]) if np.isfinite(rec["CFBD_Portal_In_Prior_Usage"]) and np.isfinite(rec["CFBD_Portal_Out_Prior_Usage"]) else np.nan
        rows.append(rec)
    outdf=pd.DataFrame(rows)
    if not outdf.empty:
        # Scale-free context for Miner thresholds. Higher percentile = stronger net
        # incoming prior production/usage relative to that season's portal market.
        outdf["CFBD_Portal_Net_PriorPPA_Pct"]=_num(outdf["CFBD_Portal_Net_Prior_TotalPPA"]).rank(pct=True,method="average")
        outdf["CFBD_Portal_Net_PriorUsage_Pct"]=_num(outdf["CFBD_Portal_Net_Prior_Usage"]).rank(pct=True,method="average")
    return outdf


def _roster_name_key(df: pd.DataFrame) -> pd.Series:
    first=_first_col(df,"firstName").astype(str).replace({"nan":"","None":""}).str.strip()
    last=_first_col(df,"lastName").astype(str).replace({"nan":"","None":""}).str.strip()
    combo=(first+" "+last).str.strip()
    single=_first_col(df,"name").astype(str).replace({"nan":"","None":""}).str.strip()
    return combo.where(combo.str.len()>=2,single).map(_person_key)


def build_prior_player_continuity(cur_roster: pd.DataFrame, prior_ppa: pd.DataFrame,
                                  prior_usage: pd.DataFrame, year: int) -> pd.DataFrame:
    """Prior-season production/usage retained on the current preseason roster.

    This is the player-level analogue of returning production.  It uses only
    prior-season CFBD PPA/usage plus the target season roster; no target-season
    game performance enters these fields.
    """
    pv=_prior_player_value_table(prior_ppa if prior_ppa is not None else pd.DataFrame(),
                                 prior_usage if prior_usage is not None else pd.DataFrame())
    if pv.empty or cur_roster is None or cur_roster.empty:
        return pd.DataFrame(columns=["Season","Team_Key"])
    r=cur_roster.copy()
    r["Team_Key"]=_first_col(r,"team","school").map(_team_key)
    r["Player_Key"]=_roster_name_key(r)
    current={(str(t),str(n)) for t,n in zip(r["Team_Key"],r["Player_Key"]) if t and n}
    rows=[]
    for tk,g in pv.groupby("Team_Key",sort=False):
        q=g.copy()
        q["Prior_Total_PPA"]=_num(q["Prior_Total_PPA"])
        q["Prior_Usage"]=_num(q["Prior_Usage"])
        q["Returned"]=[float((str(tk),str(pk)) in current) for pk in q["Player_Key"]]
        ppa=q["Prior_Total_PPA"].where(q["Prior_Total_PPA"].notna(),0.0)
        use=q["Prior_Usage"].where(q["Prior_Usage"].notna(),0.0)
        ppa_pos=ppa.clip(lower=0); use_pos=use.clip(lower=0)
        total_ppa=float(ppa_pos.sum()); total_use=float(use_pos.sum())
        ret_ppa=float((ppa_pos*q["Returned"]).sum()); ret_use=float((use_pos*q["Returned"]).sum())
        top_ppa=q.sort_values("Prior_Total_PPA",ascending=False,na_position="last").head(1)
        top_use=q.sort_values("Prior_Usage",ascending=False,na_position="last").head(1)
        top3_ppa=q.sort_values("Prior_Total_PPA",ascending=False,na_position="last").head(3)["Prior_Total_PPA"].clip(lower=0).sum()
        top3_use=q.sort_values("Prior_Usage",ascending=False,na_position="last").head(3)["Prior_Usage"].clip(lower=0).sum()
        hi_use=q.loc[q["Prior_Usage"].ge(.10)]
        rec={
            "Season":int(year),"Team_Key":str(tk),
            "CFBD_Prior_Player_Count":int(len(q)),
            "CFBD_Prior_Player_Returned_Count":int(q["Returned"].sum()),
            "CFBD_Prior_Player_Return_Rate":float(q["Returned"].mean()) if len(q) else np.nan,
            "CFBD_Prior_PPA_Return_Pct":ret_ppa/total_ppa if total_ppa>0 else np.nan,
            "CFBD_Prior_Usage_Return_Pct":ret_use/total_use if total_use>0 else np.nan,
            "CFBD_Prior_PPA_Lost_Pct":1-ret_ppa/total_ppa if total_ppa>0 else np.nan,
            "CFBD_Prior_Usage_Lost_Pct":1-ret_use/total_use if total_use>0 else np.nan,
            "CFBD_Prior_TopPPA_Returned":float(top_ppa["Returned"].iloc[0]) if len(top_ppa) else np.nan,
            "CFBD_Prior_TopUsage_Returned":float(top_use["Returned"].iloc[0]) if len(top_use) else np.nan,
            "CFBD_Prior_Top3_PPA_Share":float(top3_ppa/total_ppa) if total_ppa>0 else np.nan,
            "CFBD_Prior_Top3_Usage_Share":float(top3_use/total_use) if total_use>0 else np.nan,
            "CFBD_Prior_HighUsage_Count":int(len(hi_use)),
            "CFBD_Prior_HighUsage_Lost_Count":int((1-hi_use["Returned"]).sum()) if len(hi_use) else 0,
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def build_prior_rating_features(sp: pd.DataFrame, srs: pd.DataFrame, fpi: pd.DataFrame,
                                core: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Use only year-1 season ratings as static target-year context."""
    parts=[]
    if sp is not None and not sp.empty:
        x=sp.copy(); z=pd.DataFrame(index=x.index)
        z["Team_Key"]=_first_col(x,"team").map(_team_key)
        for dst,src in {
            "CFBD_PriorYear_SP_Rating":"rating","CFBD_PriorYear_SP_Rank":"ranking",
            "CFBD_PriorYear_SP_SOS":"sos","CFBD_PriorYear_SP_SecondOrderWins":"secondOrderWins",
            "CFBD_PriorYear_SP_Off_Rating":"offense.rating","CFBD_PriorYear_SP_Off_Success":"offense.success",
            "CFBD_PriorYear_SP_Off_Explosiveness":"offense.explosiveness","CFBD_PriorYear_SP_Off_Pace":"offense.pace",
            "CFBD_PriorYear_SP_Def_Rating":"defense.rating","CFBD_PriorYear_SP_Def_Success":"defense.success",
            "CFBD_PriorYear_SP_Def_Explosiveness":"defense.explosiveness","CFBD_PriorYear_SP_Def_Havoc":"defense.havoc.total",
            "CFBD_PriorYear_SP_ST_Rating":"specialTeams.rating",
        }.items(): z[dst]=_num(_first_col(x,src))
        parts.append(z.drop_duplicates("Team_Key",keep="last"))
    if srs is not None and not srs.empty:
        x=srs.copy(); z=pd.DataFrame(index=x.index); z["Team_Key"]=_first_col(x,"team").map(_team_key)
        z["CFBD_PriorYear_SRS_Rating"]=_num(_first_col(x,"rating")); z["CFBD_PriorYear_SRS_Rank"]=_num(_first_col(x,"ranking")); parts.append(z.drop_duplicates("Team_Key",keep="last"))
    if fpi is not None and not fpi.empty:
        x=fpi.copy(); z=pd.DataFrame(index=x.index); z["Team_Key"]=_first_col(x,"team").map(_team_key)
        for dst,src in {
            "CFBD_PriorYear_FPI":"fpi","CFBD_PriorYear_FPI_Eff_Overall":"efficiencies.overall",
            "CFBD_PriorYear_FPI_Eff_Offense":"efficiencies.offense","CFBD_PriorYear_FPI_Eff_Defense":"efficiencies.defense",
            "CFBD_PriorYear_FPI_Eff_ST":"efficiencies.specialTeams","CFBD_PriorYear_FPI_SOS":"resumeRanks.strengthOfSchedule",
            "CFBD_PriorYear_FPI_SOR":"resumeRanks.strengthOfRecord","CFBD_PriorYear_FPI_GameControl":"resumeRanks.gameControl",
        }.items(): z[dst]=_num(_first_col(x,src))
        parts.append(z.drop_duplicates("Team_Key",keep="last"))
    if core is not None and not core.empty:
        x=core.copy(); x["throughWeek_num"]=_num(_first_col(x,"throughWeek"))
        z=pd.DataFrame(index=x.index); z["Team_Key"]=_first_col(x,"team").map(_team_key); z["throughWeek_num"]=_num(_first_col(x,"throughWeek"))
        z["CFBD_PriorYear_CORE_Overall"]=_num(_first_col(x,"overall")); z["CFBD_PriorYear_CORE_Offense"]=_num(_first_col(x,"offense")); z["CFBD_PriorYear_CORE_Defense"]=_num(_first_col(x,"defense")); z["CFBD_PriorYear_CORE_Off_Plays"]=_num(_first_col(x,"offensePlays")); z["CFBD_PriorYear_CORE_Def_Plays"]=_num(_first_col(x,"defensePlays"))
        z=z.sort_values(["Team_Key","throughWeek_num"]).drop_duplicates("Team_Key",keep="last").drop(columns="throughWeek_num")
        parts.append(z)
    if not parts: return pd.DataFrame(columns=["Season","Team_Key"])
    out=parts[0]
    for z in parts[1:]: out=out.merge(z,on="Team_Key",how="outer")
    out["Season"]=int(target_year)
    return out.loc[out["Team_Key"].ne("")]


def build_talent_features(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Season", "Team_Key"])
    x = df.copy()
    x["Season"] = year
    x["Team"] = _first_col(x, "school", "team").astype(str)
    x["Team_Key"] = x["Team"].map(_team_key)
    x["CFBD_Talent"] = _num(_first_col(x, "talent"))
    x["CFBD_Talent_Rank"] = x["CFBD_Talent"].rank(method="min", ascending=False)
    return x[["Season", "Team", "Team_Key", "CFBD_Talent", "CFBD_Talent_Rank"]].drop_duplicates(["Season", "Team_Key"])


def build_recruiting_features(all_years: dict[int, pd.DataFrame], target_year: int) -> pd.DataFrame:
    rows = []
    for y, df in all_years.items():
        if df.empty:
            continue
        x = df.copy()
        x["class_year"] = _num(_first_col(x, "year")).fillna(y)
        x["Team"] = _first_col(x, "team").astype(str)
        x["Team_Key"] = x["Team"].map(_team_key)
        x["rank"] = _num(_first_col(x, "rank"))
        x["points"] = _num(_first_col(x, "points"))
        rows.append(x[["class_year", "Team", "Team_Key", "rank", "points"]])
    if not rows:
        return pd.DataFrame(columns=["Season", "Team_Key"])
    z = pd.concat(rows, ignore_index=True)
    z = z[z["class_year"].between(target_year - 3, target_year)]
    out = []
    for tk, g in z.groupby("Team_Key", sort=False):
        cur = g[g["class_year"].eq(target_year)]
        out.append({
            "Season": target_year, "Team_Key": tk,
            "CFBD_Recruit_Class_Rank": float(cur["rank"].iloc[0]) if len(cur) and pd.notna(cur["rank"].iloc[0]) else np.nan,
            "CFBD_Recruit_Class_Points": float(cur["points"].iloc[0]) if len(cur) and pd.notna(cur["points"].iloc[0]) else np.nan,
            "CFBD_Recruit_4Y_Avg_Rank": float(g["rank"].mean()) if g["rank"].notna().any() else np.nan,
            "CFBD_Recruit_4Y_Avg_Points": float(g["points"].mean()) if g["points"].notna().any() else np.nan,
            "CFBD_Recruit_4Y_Classes": int(g["class_year"].nunique()),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Game / chronology feature engineering
# ---------------------------------------------------------------------------

def standardize_games(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x = df.copy()
    out = pd.DataFrame(index=x.index)
    out["Game_ID"] = _num(_first_col(x, "id", "gameId")).astype("Int64")
    out["Season"] = _num(_first_col(x, "season", "year")).fillna(year).astype("Int64")
    out["Week"] = _num(_first_col(x, "week")).astype("Int64")
    out["Season_Type"] = _first_col(x, "seasonType").astype(str)
    out["Game_Date"] = pd.to_datetime(_first_col(x, "startDate", "startTime"), errors="coerce", utc=True)
    out["Completed"] = _bool_col(x, "completed")
    out["Home_Team"] = _first_col(x, "homeTeam").astype(str)
    out["Away_Team"] = _first_col(x, "awayTeam").astype(str)
    out["Home_Key"] = out["Home_Team"].map(_team_key)
    out["Away_Key"] = out["Away_Team"].map(_team_key)
    out["Home_Conference"] = _first_col(x, "homeConference").astype(str).replace({"nan":"","None":""})
    out["Away_Conference"] = _first_col(x, "awayConference").astype(str).replace({"nan":"","None":""})
    out["Home_Pregame_Elo"] = _num(_first_col(x, "homePregameElo"))
    out["Away_Pregame_Elo"] = _num(_first_col(x, "awayPregameElo"))
    out["Venue_ID"] = _num(_first_col(x, "venueId")).astype("Int64")
    out["Neutral_Site"] = _bool_col(x, "neutralSite")
    return out.dropna(subset=["Game_ID"])


def standardize_advanced(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x = df.copy()
    out = pd.DataFrame(index=x.index)
    out["Game_ID"] = _num(_first_col(x, "gameId", "id")).astype("Int64")
    out["Season"] = _num(_first_col(x, "season")).astype("Int64")
    out["Week"] = _num(_first_col(x, "week")).astype("Int64")
    out["Team"] = _first_col(x, "team").astype(str)
    out["Opponent"] = _first_col(x, "opponent").astype(str)
    out["Team_Key"] = out["Team"].map(_team_key); out["Opponent_Key"] = out["Opponent"].map(_team_key)
    m = {
        "Off_PPA": ("offense.ppa", "off_ppa"),
        "Off_Success": ("offense.successRate", "off_success_rate"),
        "Off_Explosiveness": ("offense.explosiveness", "off_explosiveness"),
        "Off_LineYards": ("offense.lineYards", "off_line_yds"),
        "Off_StuffRate": ("offense.stuffRate", "off_stuff_rate"),
        "Off_PowerSuccess": ("offense.powerSuccess", "off_power_success"),
        "Off_Pass_PPA": ("offense.passingPlays.ppa", "off_pass_ppa"),
        "Off_Pass_Success": ("offense.passingPlays.successRate", "off_pass_success_rate"),
        "Off_Pass_Explosiveness": ("offense.passingPlays.explosiveness", "off_pass_explosiveness"),
        "Off_Rush_PPA": ("offense.rushingPlays.ppa", "off_rush_ppa"),
        "Off_Rush_Success": ("offense.rushingPlays.successRate", "off_rush_success_rate"),
        "Off_Rush_Explosiveness": ("offense.rushingPlays.explosiveness", "off_rush_explosiveness"),
        "Off_PassingDown_PPA": ("offense.passingDowns.ppa",),
        "Off_PassingDown_Success": ("offense.passingDowns.successRate",),
        "Off_PassingDown_Explosiveness": ("offense.passingDowns.explosiveness",),
        "Off_StandardDown_PPA": ("offense.standardDowns.ppa",),
        "Off_StandardDown_Success": ("offense.standardDowns.successRate",),
        "Off_StandardDown_Explosiveness": ("offense.standardDowns.explosiveness",),
        "Off_OpenFieldYards": ("offense.openFieldYards",),
        "Off_SecondLevelYards": ("offense.secondLevelYards",),
        "Off_TotalPPA": ("offense.totalPPA",),
        "Off_Drives": ("offense.drives",),
        "Off_Plays": ("offense.plays",),
        "Def_PPA": ("defense.ppa", "def_ppa"),
        "Def_Success": ("defense.successRate", "def_success_rate"),
        "Def_Explosiveness": ("defense.explosiveness", "def_explosiveness"),
        "Def_LineYards": ("defense.lineYards", "def_line_yds"),
        "Def_StuffRate": ("defense.stuffRate", "def_stuff_rate"),
        "Def_PowerSuccess": ("defense.powerSuccess", "def_power_success"),
        "Def_Pass_PPA": ("defense.passingPlays.ppa", "def_pass_ppa"),
        "Def_Pass_Success": ("defense.passingPlays.successRate", "def_pass_success_rate"),
        "Def_Pass_Explosiveness": ("defense.passingPlays.explosiveness", "def_pass_explosiveness"),
        "Def_Rush_PPA": ("defense.rushingPlays.ppa", "def_rush_ppa"),
        "Def_Rush_Success": ("defense.rushingPlays.successRate", "def_rush_success_rate"),
        "Def_Rush_Explosiveness": ("defense.rushingPlays.explosiveness", "def_rush_explosiveness"),
        "Def_PassingDown_PPA": ("defense.passingDowns.ppa",),
        "Def_PassingDown_Success": ("defense.passingDowns.successRate",),
        "Def_PassingDown_Explosiveness": ("defense.passingDowns.explosiveness",),
        "Def_StandardDown_PPA": ("defense.standardDowns.ppa",),
        "Def_StandardDown_Success": ("defense.standardDowns.successRate",),
        "Def_StandardDown_Explosiveness": ("defense.standardDowns.explosiveness",),
        "Def_OpenFieldYards": ("defense.openFieldYards",),
        "Def_SecondLevelYards": ("defense.secondLevelYards",),
        "Def_TotalPPA": ("defense.totalPPA",),
        "Def_Drives": ("defense.drives",),
        "Def_Plays": ("defense.plays",),
    }
    for dst, src in m.items(): out[dst] = _num(_first_col(x, *src))
    return out


def standardize_havoc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    x = df.copy(); out = pd.DataFrame(index=x.index)
    out["Game_ID"] = _num(_first_col(x, "gameId", "game_id", "id")).astype("Int64")
    out["Team"] = _first_col(x, "team").astype(str); out["Team_Key"] = out["Team"].map(_team_key)
    out["CFBD_Game_Off_Havoc_Allowed"] = _num(_first_col(x, "offense.havocRate", "offense.total", "off_havoc_rate", "offense.havoc_rate"))
    out["CFBD_Game_Def_Havoc"] = _num(_first_col(x, "defense.havocRate", "defense.total", "def_havoc_rate", "defense.havoc_rate"))
    out["CFBD_Game_Def_Front7_Havoc"] = _num(_first_col(x, "defense.frontSevenHavocRate", "defense.frontSeven", "def_front_seven_havoc_rate"))
    out["CFBD_Game_Def_DB_Havoc"] = _num(_first_col(x, "defense.dbHavocRate", "defense.db", "def_db_havoc_rate"))
    return out


def build_play_context_features(plays: pd.DataFrame) -> pd.DataFrame:
    """Completed-game team-side context derived only from CFBD /plays.

    These raw game metrics are never used directly; attach_game_metadata merges
    them and add_prior_advanced_features shifts/rolls them before research use.
    """
    if plays is None or plays.empty:
        return pd.DataFrame(columns=["Game_ID","Team_Key"])
    x=plays.copy()
    x["Game_ID"]=_num(_first_col(x,"gameId")).astype("Int64")
    x["Drive_ID"]=_first_col(x,"driveId").astype(str)
    x["Offense"]=_first_col(x,"offense").astype(str); x["Defense"]=_first_col(x,"defense").astype(str)
    x["Offense_Key"]=x["Offense"].map(_team_key); x["Defense_Key"]=x["Defense"].map(_team_key)
    x["Play_Type"]=_first_col(x,"playType").astype(str).str.lower()
    x["Down"]=_num(_first_col(x,"down")); x["Distance"]=_num(_first_col(x,"distance")); x["YTG"]=_num(_first_col(x,"yardsToGoal")); x["Yards"]=_num(_first_col(x,"yardsGained")); x["PPA"]=_num(_first_col(x,"ppa"))
    non_scrim=x["Play_Type"].str.contains(r"kickoff|punt|field goal|extra point|timeout|end of|penalty|coin toss",regex=True,na=False)
    x=x.loc[~non_scrim & x["Offense_Key"].ne("") & x["Defense_Key"].ne("") & x["Game_ID"].notna()].copy()
    if x.empty: return pd.DataFrame(columns=["Game_ID","Team_Key"])
    # Standard NCAA success-rate approximation when the historical endpoint does
    # not expose a success boolean: 50%/70%/100% of yards-to-go on downs 1/2/3-4.
    need=x["Distance"].replace(0,np.nan)
    req=np.select([x["Down"].eq(1),x["Down"].eq(2),x["Down"].isin([3,4])],[.50,.70,1.0],default=np.nan)
    x["Success"]=(x["Yards"].ge(need*req)).where(np.isfinite(req))
    x["Early"] = x["Down"].isin([1,2])
    x["PassingDown"] = ((x["Down"].eq(2)&x["Distance"].ge(8)) | (x["Down"].isin([3,4])&x["Distance"].ge(5)))
    x["StandardDown"] = x["Down"].isin([1,2,3,4]) & ~x["PassingDown"]
    x["PassDropback"] = x["Play_Type"].str.contains(r"pass|sack",regex=True,na=False)
    x["Sack"] = x["Play_Type"].str.contains("sack",regex=False,na=False)
    x["Explosive20"] = x["Yards"].ge(20)
    first_drive=x.sort_values(["Game_ID","Drive_ID"]).groupby(["Game_ID","Drive_ID"],sort=False).head(1)[["Game_ID","Drive_ID","Offense_Key","YTG"]].copy()
    first_drive["Start_Field_Position"] = 100-first_drive["YTG"]
    # Any play at or inside opponent 40 makes the drive a scoring opportunity.
    opp=x.groupby(["Game_ID","Drive_ID","Offense_Key"],as_index=False).agg(Min_YTG=("YTG","min"),Drive_Yards=("Yards","sum"),Drive_PPA=("PPA","sum"),Drive_Plays=("Yards","size"))
    opp["Scoring_Opportunity"] = opp["Min_YTG"].le(40)
    drv=opp.merge(first_drive[["Game_ID","Drive_ID","Offense_Key","Start_Field_Position"]],on=["Game_ID","Drive_ID","Offense_Key"],how="left")
    rows=[]
    for (gid,offk,defk),g in x.groupby(["Game_ID","Offense_Key","Defense_Key"],sort=False):
        d=drv.loc[(drv["Game_ID"].eq(gid)) & (drv["Offense_Key"].eq(offk))]
        def mean_mask(col,mask):
            z=_num(g.loc[mask,col]).dropna(); return float(z.mean()) if len(z) else np.nan
        drop=g["PassDropback"]
        rec={
            "Game_ID":gid,"Offense_Key":offk,"Defense_Key":defk,
            "Off_Field_Position":float(_num(d["Start_Field_Position"]).mean()) if len(d) else np.nan,
            "Off_Drives_Derived":float(len(d)) if len(d) else np.nan,
            "Off_Plays_Per_Drive":float(_num(d["Drive_Plays"]).mean()) if len(d) else np.nan,
            "Off_Yards_Per_Drive":float(_num(d["Drive_Yards"]).mean()) if len(d) else np.nan,
            "Off_PPA_Per_Drive":float(_num(d["Drive_PPA"]).mean()) if len(d) else np.nan,
            "Off_Scoring_Opportunity_Rate":float(d["Scoring_Opportunity"].mean()) if len(d) else np.nan,
            "Off_EarlyDown_PPA":mean_mask("PPA",g["Early"]),
            "Off_EarlyDown_Success":float(g.loc[g["Early"],"Success"].mean()) if g["Early"].any() else np.nan,
            "Off_PassingDown_PPA_Derived":mean_mask("PPA",g["PassingDown"]),
            "Off_PassingDown_Success_Derived":float(g.loc[g["PassingDown"],"Success"].mean()) if g["PassingDown"].any() else np.nan,
            "Off_StandardDown_PPA_Derived":mean_mask("PPA",g["StandardDown"]),
            "Off_StandardDown_Success_Derived":float(g.loc[g["StandardDown"],"Success"].mean()) if g["StandardDown"].any() else np.nan,
            "Off_Sack_Rate":float(g.loc[drop,"Sack"].mean()) if drop.any() else np.nan,
            "Off_Explosive20_Rate":float(g["Explosive20"].mean()) if len(g) else np.nan,
        }
        rows.append(rec)
    o=pd.DataFrame(rows)
    if o.empty: return pd.DataFrame(columns=["Game_ID","Team_Key"])
    off=o.rename(columns={"Offense_Key":"Team_Key"}).drop(columns="Defense_Key")
    defn=o.rename(columns={"Defense_Key":"Team_Key"})
    def_cols={c:"Def_"+c[len("Off_"):] for c in defn.columns if c.startswith("Off_")}
    defn=defn.rename(columns=def_cols).drop(columns="Offense_Key")
    out=off.merge(defn,on=["Game_ID","Team_Key"],how="outer")
    out=out.rename(columns={c:"CFBD_Game_"+c for c in out.columns if c.startswith(("Off_","Def_"))})
    return out.drop_duplicates(["Game_ID","Team_Key"],keep="last")


def add_conference_schedule_context(df: pd.DataFrame) -> pd.DataFrame:
    """Pregame conference familiarity/streak features from the CFBD schedule."""
    if df.empty: return df
    x=df.sort_values(["Season","Team_Key","Game_Date","Week","Game_ID"],kind="stable").copy()
    same=_num(x.get("CFBD_Same_Conference",pd.Series(np.nan,index=x.index))).eq(1)
    completed=_num(x.get("Completed",pd.Series(np.nan,index=x.index))).eq(1)
    pri=np.full(len(x),np.nan); streak=np.full(len(x),np.nan)
    for _,idx0 in x.groupby(["Season","Team_Key"],sort=False).indices.items():
        idx=np.asarray(idx0,dtype=np.intp); count=0; st=0
        for i in idx:
            pri[i]=count; streak[i]=st
            if completed.iloc[i]:
                if same.iloc[i]: count+=1; st+=1
                else: st=0
    x["CFBD_Conference_Games_Prior"]=pri; x["CFBD_Conference_Game_Streak_Prior"]=st
    return x.sort_index()


def attach_game_metadata(adv: pd.DataFrame, games: pd.DataFrame, havoc: pd.DataFrame, play_context: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one row per scheduled team-side, then attach completed-game stats.

    The schedule is the base so upcoming games exist before kickoff. Completed
    advanced/havoc values are attached only to their own game and are subsequently
    shifted by add_prior_advanced_features before becoming model/Miner inputs.
    """
    if games.empty:
        return pd.DataFrame()
    base_cols=[c for c in ["Game_ID","Season","Week","Season_Type","Game_Date","Completed","Venue_ID","Neutral_Site","Home_Team","Away_Team","Home_Key","Away_Key","Home_Pregame_Elo","Away_Pregame_Elo","Home_Conference","Away_Conference"] if c in games.columns]
    gg=games[base_cols].drop_duplicates("Game_ID").copy()
    home=gg.copy(); away=gg.copy()
    home["Team"]=home["Home_Team"]; home["Opponent"]=home["Away_Team"]
    home["Team_Key"]=home["Home_Key"]; home["Opponent_Key"]=home["Away_Key"]; home["Is_Home"]=1.0
    home["CFBD_Conference"]=home.get("Home_Conference",pd.Series("",index=home.index)); home["CFBD_Opp_Conference"]=home.get("Away_Conference",pd.Series("",index=home.index))
    home["CFBD_Elo_Prior"]=_num(home.get("Home_Pregame_Elo",pd.Series(np.nan,index=home.index)))
    away["Team"]=away["Away_Team"]; away["Opponent"]=away["Home_Team"]
    away["Team_Key"]=away["Away_Key"]; away["Opponent_Key"]=away["Home_Key"]; away["Is_Home"]=0.0
    away["CFBD_Conference"]=away.get("Away_Conference",pd.Series("",index=away.index)); away["CFBD_Opp_Conference"]=away.get("Home_Conference",pd.Series("",index=away.index))
    away["CFBD_Elo_Prior"]=_num(away.get("Away_Pregame_Elo",pd.Series(np.nan,index=away.index)))
    x=pd.concat([home,away],ignore_index=True,sort=False)
    x=x.drop(columns=["Home_Team","Away_Team","Home_Key","Away_Key","Home_Pregame_Elo","Away_Pregame_Elo","Home_Conference","Away_Conference"],errors="ignore")
    x["CFBD_Same_Conference"] = ((x["CFBD_Conference"].astype(str).str.len()>0) & (x["CFBD_Conference"].astype(str)==x["CFBD_Opp_Conference"].astype(str))).astype(float)

    if adv is not None and not adv.empty:
        metric_cols=[c for c in adv.columns if c.startswith(("Off_","Def_"))]
        aa=adv[["Game_ID","Team_Key"]+metric_cols].drop_duplicates(["Game_ID","Team_Key"],keep="last")
        x=x.merge(aa,on=["Game_ID","Team_Key"],how="left")
    if havoc is not None and not havoc.empty:
        hcols=[c for c in havoc.columns if c.startswith("CFBD_Game_")]
        hh=havoc[["Game_ID","Team_Key"]+hcols].drop_duplicates(["Game_ID","Team_Key"],keep="last")
        x=x.merge(hh,on=["Game_ID","Team_Key"],how="left")
    if play_context is not None and not play_context.empty:
        pcols=[c for c in play_context.columns if c.startswith("CFBD_Game_")]
        pp=play_context[["Game_ID","Team_Key"]+pcols].drop_duplicates(["Game_ID","Team_Key"],keep="last")
        x=x.merge(pp,on=["Game_ID","Team_Key"],how="left")
    return x


def add_prior_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe advanced state from completed games only.

    The schedule is the base, so one team can have several future rows. A normal
    row-based rolling window would let those unplayed rows consume Recent-3/5
    slots. History updates only after a completed game's metric is observed.
    """
    if df.empty:
        return df
    x=df.copy().sort_values(["Season","Team_Key","Game_Date","Game_ID"],kind="stable").reset_index(drop=True)
    game_metrics=[c for c in x.columns if c.startswith(("Off_","Def_","CFBD_Game_"))]
    complete=_num(x.get("Completed",pd.Series(np.nan,index=x.index)))
    if not complete.notna().any():
        if game_metrics:
            complete=pd.concat([_num(x[c]).notna() for c in game_metrics],axis=1).any(axis=1).astype(float)
        else:
            complete=pd.Series(0.0,index=x.index)
    complete_arr=complete.fillna(0).eq(1).to_numpy(bool)
    completed_int=pd.Series(complete_arr.astype(int),index=x.index)
    games_prior=completed_int.groupby([x["Season"],x["Team_Key"]],sort=False,dropna=False).cumsum()-completed_int
    newcols: dict[str,pd.Series]={"CFBD_Games_Prior":games_prior.astype(float)}
    group_positions=list(x.groupby(["Season","Team_Key"],sort=False,dropna=False).indices.values())

    for c in game_metrics:
        vals=_num(x[c]).to_numpy(float)
        season_out=np.full(len(x),np.nan,dtype=float)
        r3_out=np.full(len(x),np.nan,dtype=float)
        r5_out=np.full(len(x),np.nan,dtype=float)
        for pos0 in group_positions:
            pos=np.asarray(pos0,dtype=np.intp)
            history=[]; total=0.0; count=0
            for i in pos:
                if count:
                    season_out[i]=total/count
                    t3=history[-3:]; t5=history[-5:]
                    r3_out[i]=float(sum(t3)/len(t3))
                    r5_out[i]=float(sum(t5)/len(t5))
                v=vals[i]
                if complete_arr[i] and np.isfinite(v):
                    fv=float(v); history.append(fv); total+=fv; count+=1
        newcols[f"CFBD_Season_{c}_Prior"]=pd.Series(season_out,index=x.index)
        newcols[f"CFBD_Recent3_{c}_Prior"]=pd.Series(r3_out,index=x.index)
        newcols[f"CFBD_Recent5_{c}_Prior"]=pd.Series(r5_out,index=x.index)

    feat=pd.DataFrame(newcols,index=x.index)
    feat["CFBD_Season_Net_PPA_Prior"] = feat.get("CFBD_Season_Off_PPA_Prior") - feat.get("CFBD_Season_Def_PPA_Prior")
    feat["CFBD_Recent3_Net_PPA_Prior"] = feat.get("CFBD_Recent3_Off_PPA_Prior") - feat.get("CFBD_Recent3_Def_PPA_Prior")
    feat["CFBD_Season_Net_Success_Prior"] = feat.get("CFBD_Season_Off_Success_Prior") - feat.get("CFBD_Season_Def_Success_Prior")
    feat["CFBD_Season_Net_Explosiveness_Prior"] = feat.get("CFBD_Season_Off_Explosiveness_Prior") - feat.get("CFBD_Season_Def_Explosiveness_Prior")
    return pd.concat([x,feat],axis=1)


def build_qb_prior(qb: pd.DataFrame, games: pd.DataFrame, qb_success: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build team-game QB PPA priors.

    CFBD player-game PPA does not expose an attempt count in this endpoint, so
    multiple-QB games are summarized conservatively by the median QB PPA plus a
    QB-count flag. Team passing PPA remains the primary efficiency signal.
    """
    if qb.empty:
        return pd.DataFrame(columns=["Season", "Week", "Team_Key"])
    x = qb.copy()
    x["Season"] = _num(_first_col(x, "season")).astype("Int64")
    x["Week"] = _num(_first_col(x, "week")).astype("Int64")
    x["Team"] = _first_col(x, "team").astype(str); x["Team_Key"] = x["Team"].map(_team_key)
    x["QB_PPA_All"] = _num(_first_col(x, "averagePPA.all", "avg_PPA_all", "averagePPA.total"))
    x["QB_PPA_Pass"] = _num(_first_col(x, "averagePPA.pass", "avg_PPA_pass"))
    qg = x.groupby(["Season", "Week", "Team_Key"], as_index=False).agg(
        CFBD_QB_Game_PPA=("QB_PPA_All", "median"),
        CFBD_QB_Game_Pass_PPA=("QB_PPA_Pass", "median"),
        CFBD_QB_Game_Count=("Team", "size"),
    )
    # Shift by week/team. This is only a prior form signal; roster continuity
    # separately indicates whether the current season's QB room is returning.
    qg = qg.sort_values(["Season", "Team_Key", "Week"], kind="stable")
    grp = [qg["Season"], qg["Team_Key"]]
    # Add QB passing/rushing success rate by game when available. Aggregate by
    # credited plays so multi-QB games do not overweight a low-volume backup.
    if qb_success is not None and not qb_success.empty:
        z=qb_success.copy()
        z["Season"]=_num(_first_col(z,"season")).astype("Int64")
        z["Week"]=_num(_first_col(z,"week")).astype("Int64")
        z["Team_Key"]=_first_col(z,"team").astype(str).map(_team_key)
        z["Pass_Plays"]=_num(_first_col(z,"passing.plays")); z["Pass_Successes"]=_num(_first_col(z,"passing.successes"))
        z["Rush_Plays"]=_num(_first_col(z,"rushing.plays")); z["Rush_Successes"]=_num(_first_col(z,"rushing.successes"))
        zs=z.groupby(["Season","Week","Team_Key"],as_index=False).agg(
            Pass_Plays=("Pass_Plays","sum"),Pass_Successes=("Pass_Successes","sum"),
            Rush_Plays=("Rush_Plays","sum"),Rush_Successes=("Rush_Successes","sum"))
        zs["CFBD_QB_Game_Pass_Success"] = zs["Pass_Successes"] / zs["Pass_Plays"].replace(0,np.nan)
        zs["CFBD_QB_Game_Rush_Success"] = zs["Rush_Successes"] / zs["Rush_Plays"].replace(0,np.nan)
        qg=qg.merge(zs[["Season","Week","Team_Key","CFBD_QB_Game_Pass_Success","CFBD_QB_Game_Rush_Success"]],on=["Season","Week","Team_Key"],how="outer")
        qg=qg.sort_values(["Season","Team_Key","Week"],kind="stable").reset_index(drop=True)
        grp=[qg["Season"],qg["Team_Key"]]

    raw_cols=[c for c in ["CFBD_QB_Game_PPA","CFBD_QB_Game_Pass_PPA","CFBD_QB_Game_Pass_Success","CFBD_QB_Game_Rush_Success"] if c in qg.columns]
    for c in raw_cols:
        vals=_num(qg[c])
        qg[f"{c}_SeasonPrior"] = vals.groupby(grp, sort=False).transform(lambda z: z.shift(1).expanding(min_periods=1).mean())
        qg[f"{c}_Recent3Prior"] = vals.groupby(grp, sort=False).transform(lambda z: z.shift(1).rolling(3, min_periods=1).mean())
    return qg.drop(columns=raw_cols+["Pass_Plays","Pass_Successes","Rush_Plays","Rush_Successes"], errors="ignore")


def build_elo_prior(elo: pd.DataFrame, target_rows: pd.DataFrame) -> pd.DataFrame:
    if elo.empty or target_rows.empty:
        return pd.DataFrame(index=target_rows.index)
    e = elo.copy()
    e["Season"] = _num(_first_col(e, "year", "season")).astype("Int64")
    e["Team_Key"] = _first_col(e, "team").map(_team_key)
    e["snapshot_week"] = _num(_first_col(e, "snapshot_week")).astype("Int64")
    e["elo"] = _num(_first_col(e, "elo"))
    lookup = {(int(r.Season), str(r.Team_Key), int(r.snapshot_week)): r.elo for r in e.itertuples() if pd.notna(r.Season) and pd.notna(r.snapshot_week)}
    vals = []
    for r in target_rows.itertuples():
        if pd.isna(r.Season) or pd.isna(r.Week): vals.append(np.nan); continue
        wk = max(0, int(r.Week) - 1)
        vals.append(lookup.get((int(r.Season), str(r.Team_Key), wk), np.nan))
    return pd.DataFrame({"CFBD_Elo_Prior": vals}, index=target_rows.index)


def build_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    if weather.empty:
        return pd.DataFrame(columns=["Game_ID"])
    x = weather.copy(); out = pd.DataFrame(index=x.index)
    out["Game_ID"] = _num(_first_col(x, "id", "gameId")).astype("Int64")
    out["CFBD_Weather_Observed_Flag"] = 1.0
    out["CFBD_Game_Indoors"] = _bool_col(x, "gameIndoors", "indoor")
    out["CFBD_Temperature"] = _num(_first_col(x, "temperature"))
    out["CFBD_DewPoint"] = _num(_first_col(x, "dewPoint"))
    out["CFBD_Humidity"] = _num(_first_col(x, "humidity"))
    out["CFBD_Precipitation"] = _num(_first_col(x, "precipitation"))
    out["CFBD_Snowfall"] = _num(_first_col(x, "snowfall"))
    out["CFBD_WindSpeed"] = _num(_first_col(x, "windSpeed"))
    out["CFBD_Weather_Condition"] = _first_col(x, "weatherCondition").astype(str)
    return out.drop_duplicates("Game_ID", keep="last")


def build_venue_features(venues: pd.DataFrame) -> pd.DataFrame:
    if venues.empty:
        return pd.DataFrame(columns=["Venue_ID"])
    x = venues.copy(); out = pd.DataFrame(index=x.index)
    out["Venue_ID"] = _num(_first_col(x, "id")).astype("Int64")
    out["CFBD_Venue_Elevation"] = _num(_first_col(x, "elevation"))
    out["CFBD_Venue_Latitude"] = _num(_first_col(x, "latitude"))
    out["CFBD_Venue_Longitude"] = _num(_first_col(x, "longitude"))
    out["CFBD_Venue_Timezone"] = _first_col(x, "timezone").astype(str)
    out["CFBD_Venue_Dome"] = _bool_col(x, "dome")
    return out.drop_duplicates("Venue_ID", keep="last")


def build_home_location_features(teams: pd.DataFrame, year: int) -> pd.DataFrame:
    """Home-campus location known independent of the current game's result."""
    if teams.empty:
        return pd.DataFrame(columns=["Season", "Team_Key"])
    x = teams.copy(); out = pd.DataFrame(index=x.index)
    out["Season"] = int(year)
    out["Team"] = _first_col(x, "school", "team").astype(str)
    out["Team_Key"] = out["Team"].map(_team_key)
    out["CFBD_Home_Elevation"] = _num(_first_col(x, "location.elevation", "elevation"))
    out["CFBD_Home_Latitude"] = _num(_first_col(x, "location.latitude", "latitude"))
    out["CFBD_Home_Longitude"] = _num(_first_col(x, "location.longitude", "longitude"))
    out["CFBD_Home_Timezone"] = _first_col(x, "location.timezone", "timezone").astype(str)
    out["CFBD_Home_Dome"] = _bool_col(x, "location.dome", "dome")
    return out.loc[out["Team_Key"].ne("")].drop_duplicates(["Season", "Team_Key"], keep="last")


def merge_static(base: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for f in frames:
        if f is None or f.empty:
            continue
        keys = [k for k in ["Season", "Team_Key"] if k in f.columns and k in out.columns]
        if keys == ["Season", "Team_Key"]:
            cols = keys + [c for c in f.columns if c.startswith("CFBD_")]
            out = out.merge(f[cols].drop_duplicates(keys), on=keys, how="left")
    return out


def add_opponent_mirrors(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror CFBD team-side pregame columns from the opponent row of same game."""
    if df.empty:
        return df
    out = df.copy()
    feature_cols = [c for c in out.columns if c.startswith("CFBD_") and not c.startswith("CFBD_Opp_") and ("CFBD_Opp_" + c[len("CFBD_"):]) not in out.columns]
    right = out[["Game_ID", "Team_Key"] + feature_cols].copy()
    right = right.rename(columns={"Team_Key": "Opponent_Key", **{c: "CFBD_Opp_" + c[len("CFBD_"):] for c in feature_cols}})
    return out.merge(right.drop_duplicates(["Game_ID", "Opponent_Key"]), on=["Game_ID", "Opponent_Key"], how="left")


def build_season_features(
    year: int,
    games_raw: pd.DataFrame,
    advanced_raw: pd.DataFrame,
    havoc_raw: pd.DataFrame,
    returning_raw: pd.DataFrame,
    portal_raw: pd.DataFrame,
    roster_cur: pd.DataFrame,
    roster_prev: pd.DataFrame,
    talent_raw: pd.DataFrame,
    recruiting_all: dict[int, pd.DataFrame],
    weather_raw: pd.DataFrame,
    venues_raw: pd.DataFrame,
    teams_raw: pd.DataFrame,
    qb_raw: pd.DataFrame,
    qb_success_raw: pd.DataFrame,
    prior_player_ppa_raw: pd.DataFrame,
    prior_player_usage_raw: pd.DataFrame,
    plays_raw: pd.DataFrame | None = None,
    prior_sp_raw: pd.DataFrame | None = None,
    prior_srs_raw: pd.DataFrame | None = None,
    prior_fpi_raw: pd.DataFrame | None = None,
    prior_core_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    games = standardize_games(games_raw, year)
    adv = standardize_advanced(advanced_raw)
    havoc = standardize_havoc(havoc_raw)
    play_context = build_play_context_features(plays_raw if plays_raw is not None else pd.DataFrame())
    x = attach_game_metadata(adv, games, havoc, play_context)
    x = add_prior_advanced_features(x)
    x = add_conference_schedule_context(x)
    if x.empty:
        return x

    x = merge_static(x, [
        build_returning_features(returning_raw, year),
        build_roster_continuity(roster_cur, roster_prev, year),
        build_portal_features(portal_raw, year, prior_player_ppa_raw, prior_player_usage_raw, games["Game_Date"].min() if "Game_Date" in games.columns else None),
        build_talent_features(talent_raw, year),
        build_recruiting_features(recruiting_all, year),
        build_prior_player_continuity(roster_cur, prior_player_ppa_raw, prior_player_usage_raw, year),
        build_prior_rating_features(prior_sp_raw if prior_sp_raw is not None else pd.DataFrame(), prior_srs_raw if prior_srs_raw is not None else pd.DataFrame(), prior_fpi_raw if prior_fpi_raw is not None else pd.DataFrame(), prior_core_raw if prior_core_raw is not None else pd.DataFrame(), year),
        build_home_location_features(teams_raw, year),
    ])

    q = build_qb_prior(qb_raw, games, qb_success_raw)
    if not q.empty:
        x = x.merge(q, on=["Season", "Week", "Team_Key"], how="left")

    w = build_weather_features(weather_raw)
    if not w.empty:
        x = x.merge(w, on="Game_ID", how="left")
    v = build_venue_features(venues_raw)
    if not v.empty:
        x = x.merge(v, on="Venue_ID", how="left")

    # Altitude shock is static/pregame-safe: destination venue elevation minus
    # the team's own home-campus elevation. Positive = team travels uphill.
    if "CFBD_Venue_Elevation" in x.columns and "CFBD_Home_Elevation" in x.columns:
        x["CFBD_Altitude_Gain"] = _num(x["CFBD_Venue_Elevation"]) - _num(x["CFBD_Home_Elevation"])
        x["CFBD_Altitude_Absolute"] = _num(x["CFBD_Venue_Elevation"])

    # Stable identity and audit fields.
    x["CFBD_Source_Tag"] = SOURCE_TAG
    x["CFBD_Leakage_Contract"] = "GAME_STATS_AND_PBP_SHIFT1__PORTAL_PRESEASON_CUTOFF__PRIOR_SEASON_RATINGS_ONLY__PRESEASON_STATIC_DIRECT__GAME_PREGAME_ELO__2026_NOT_USED_FOR_HISTORICAL_SELECTION"
    x["CFBD_Feature_AsOf_Game"] = x["Game_Date"]
    x = add_opponent_mirrors(x)

    # Useful direct matchup deltas. These are all pregame because both sides are
    # already shifted/static.
    pairs = [
        ("CFBD_Season_Net_PPA_Prior", "CFBD_Matchup_Net_PPA_Adv"),
        ("CFBD_Season_Net_Success_Prior", "CFBD_Matchup_Net_Success_Adv"),
        ("CFBD_Season_Net_Explosiveness_Prior", "CFBD_Matchup_Net_Explosiveness_Adv"),
        ("CFBD_Talent", "CFBD_Matchup_Talent_Adv"),
        ("CFBD_Recruit_4Y_Avg_Points", "CFBD_Matchup_Recruit_Adv"),
        ("CFBD_Elo_Prior", "CFBD_Matchup_Elo_Adv"),
        ("CFBD_Returning_Pct_PPA", "CFBD_Matchup_Returning_PPA_Adv"),
        ("CFBD_Prior_PPA_Return_Pct", "CFBD_Matchup_Prior_PPA_Return_Adv"),
        ("CFBD_Prior_Usage_Return_Pct", "CFBD_Matchup_Prior_Usage_Return_Adv"),
        ("CFBD_PriorYear_SP_Rating", "CFBD_Matchup_PriorYear_SP_Adv"),
        ("CFBD_PriorYear_SRS_Rating", "CFBD_Matchup_PriorYear_SRS_Adv"),
        ("CFBD_PriorYear_FPI", "CFBD_Matchup_PriorYear_FPI_Adv"),
        ("CFBD_PriorYear_CORE_Overall", "CFBD_Matchup_PriorYear_CORE_Adv"),
        ("CFBD_Season_CFBD_Game_Off_EarlyDown_PPA_Prior", "CFBD_Matchup_EarlyDown_PPA_Adv"),
        ("CFBD_Season_CFBD_Game_Off_PPA_Per_Drive_Prior", "CFBD_Matchup_PPA_Per_Drive_Adv"),
        ("CFBD_Season_CFBD_Game_Off_Scoring_Opportunity_Rate_Prior", "CFBD_Matchup_Scoring_Opp_Rate_Adv"),
    ]
    for team_col, dst in pairs:
        opp_col = "CFBD_Opp_" + team_col[len("CFBD_"):]
        if team_col in x.columns and opp_col in x.columns:
            x[dst] = _num(x[team_col]) - _num(x[opp_col])

    # For ranks, lower is better. Positive matchup advantage means the team has
    # the better (smaller) rank than its opponent.
    for team_col, dst in [
        ("CFBD_Talent_Rank", "CFBD_Matchup_Talent_Rank_Adv"),
        ("CFBD_Recruit_4Y_Avg_Rank", "CFBD_Matchup_Recruit_Rank_Adv"),
    ]:
        opp_col = "CFBD_Opp_" + team_col[len("CFBD_"):]
        if team_col in x.columns and opp_col in x.columns:
            x[dst] = _num(x[opp_col]) - _num(x[team_col])

    keep_identity = ["Game_ID", "Season", "Week", "Season_Type", "Game_Date", "Team", "Opponent", "Team_Key", "Opponent_Key", "Venue_ID", "Neutral_Site", "Is_Home"]
    keep = keep_identity + [c for c in x.columns if c.startswith("CFBD_")]
    return x[keep].drop_duplicates(["Game_ID", "Team_Key"], keep="last")


# ---------------------------------------------------------------------------
# Offline CFBD archive-v2 adapter. RESEARCH PREVIEW ONLY; never writes BigQuery.
# ---------------------------------------------------------------------------

class CFBDArchiveV2Reader:
    """Verify content-addressed archive.py manifests and load family/year payloads.

    Do not assert that archive variants equal API loader endpoints: in particular,
    advanced_game was retrieved by week without `excludeGarbageTime=true`, and
    several QB, recruiting-team, havoc, weather and player-PPA endpoints were NOT
    downloaded by archive.py. Those fields remain absent in this preview.
    """
    def __init__(self, root: str | Path, years: Iterable[int]):
        self.root = Path(root).expanduser().resolve()
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file():
            raise RuntimeError(f"CFBD_ARCHIVE_MANIFEST_MISSING path={manifest}")
        self.years = set(int(y) for y in years)
        self.requests: dict[tuple[str,int,str], dict] = {}
        self.counts: dict[str,int] = {}
        self.cache: dict[tuple[str,int],list[dict]] = {}
        self.errors: list[str] = []
        with manifest.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream,1):
                try:
                    row = json.loads(line)
                    family = str(row["family"])
                    params = row["params"]
                    year = int(params["year"])
                except (ValueError, TypeError, KeyError) as exc:
                    self.errors.append(f"MALFORMED_MANIFEST_ROW:{lineno}:{exc}")
                    continue
                if year not in self.years:
                    continue
                key = (family, year, json.dumps(params,sort_keys=True))
                # An interrupted retry must not invalidate an earlier verified
                # successful response. Keep last success for the exact request.
                if row.get("success"):
                    self.requests[key] = row
                elif key not in self.requests:
                    self.requests[key] = row
        if self.errors:
            raise RuntimeError("CFBD_ARCHIVE_MANIFEST_ERRORS " + ";".join(self.errors[:5]))

    def read(self, family: str, year: int, *, required: bool=False) -> list[dict]:
        key = (family, int(year))
        if key in self.cache:
            return self.cache[key]
        batches = [(k,r) for k,r in self.requests.items() if k[0]==family and k[1]==int(year)]
        if not batches:
            if required:
                raise RuntimeError(f"CFBD_ARCHIVE_REQUIRED_FAMILY_MISSING year={year} family={family}")
            self.cache[key] = []
            return []
        result=[]
        for (_,_,pjson), record in sorted(batches):
            if not record.get("success") or not record.get("object_path"):
                raise RuntimeError(f"CFBD_ARCHIVE_FAILED_REQUEST family={family} year={year} params={pjson}")
            objfile = (self.root / str(record["object_path"])).resolve()
            if not objfile.is_relative_to(self.root) or not objfile.is_file():
                raise RuntimeError(f"CFBD_ARCHIVE_OBJECT_MISSING_OR_UNSAFE family={family} path={objfile}")
            data = json.loads(objfile.read_text(encoding="utf-8"))
            if not isinstance(data,list):
                raise RuntimeError(f"CFBD_ARCHIVE_INVALID_PAYLOAD family={family} year={year}")
            digest = hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")).hexdigest()
            if digest != record.get("sha256") or len(data)!=int(record.get("row_count",-1)):
                raise RuntimeError(f"CFBD_ARCHIVE_CHECKSUM_OR_COUNT_MISMATCH family={family} year={year} params={pjson}")
            result.extend(data)
        if required and not result:
            raise RuntimeError(f"CFBD_ARCHIVE_REQUIRED_FAMILY_EMPTY family={family} year={year}")
        self.counts[f"{year}.{family}"] = len(result)
        self.cache[key] = result
        return result


def run_archive_research_preview(root: str, seasons: list[int], output_path: str) -> dict[str, Any]:
    """Offline study, reusing this file's SAME build_season_features().

    This deliberately does NOT publish to ncaaf_cfbd_miner_features_v1; the raw
    collector's source parameter set is not parity-equivalent to API_REFRESH.
    """
    years=sorted(set(int(y) for y in seasons))
    if not years or min(years)<2013 or max(years)>2026:  # CFBD_ARCHIVE_RESEARCH_V2_OPT_IN_20261019
        raise RuntimeError("CFBD_ARCHIVE_PREVIEW_SEASONS_MUST_BE_2013_TO_2026")
    outpath=Path(output_path).expanduser().resolve()
    if outpath.suffix.lower()!=".csv":
        raise RuntimeError("CFBD_ARCHIVE_PREVIEW_OUTPUT_MUST_BE_CSV")
    reader=CFBDArchiveV2Reader(root,set(years)|{y-1 for y in years})
    frames=[];season_quality=[]
    for year in years:
        print(f"[CFBD-ARCHIVE-PREVIEW] season={year} begin mode=OFFLINE_ONLY",flush=True)
        all_games=reader.read("games",year,required=True)
        fbs_games=[g for g in all_games if "fbs" in
                   (str(g.get("homeClassification","")).lower(),
                    str(g.get("awayClassification","")).lower())]
        if not fbs_games:
            raise RuntimeError(f"CFBD_ARCHIVE_NO_FBS_GAMES season={year}")
        ids=[str(g.get("id")) for g in fbs_games]
        if len(ids)!=len(set(ids)):
            raise RuntimeError(f"CFBD_ARCHIVE_DUPLICATE_GAME_IDS season={year}")
        adv=reader.read("advanced_game",year,required=True)
        plays=reader.read("plays",year,required=True)
        done={str(g.get("id")) for g in fbs_games if g.get("completed") is True
              and g.get("homePoints") is not None and g.get("awayPoints") is not None}
        def covered(records,key):
            return {str(z.get(key)) for z in records if isinstance(z,dict) and z.get(key) is not None}
        coverage={"advanced_game":len(done-covered(adv,"gameId")),
                  "plays":len(done-covered(plays,"gameId"))}
        # Not strictly needed by this loader's builder; record its availability
        # so downstream research can distinguish unused drive evidence.
        drives=reader.read("drives",year)
        coverage["drives"]=len(done-covered(drives,"gameId")) if drives else len(done)
        if any(coverage[x]>0 for x in ("advanced_game","plays")):
            print(f"[CFBD-ARCHIVE-PREVIEW-COVERAGE] season={year} missing_completed_fbs={coverage}",flush=True)
        games_df=_json_frame(fbs_games)
        advanced_df=_json_frame(adv)
        plays_df=_json_frame(plays)
        returning_df=_json_frame(reader.read("returning",year))
        portal_df=_json_frame(reader.read("transfers",year))
        roster_cur=_json_frame(reader.read("roster",year))
        roster_prev=_json_frame(reader.read("roster",year-1))
        talent_df=_json_frame(reader.read("talent",year))
        prior_sp=_json_frame(reader.read("ratings_sp",year-1))
        prior_fpi=_json_frame(reader.read("ratings_fpi",year-1))
        # These API endpoints are NOT in the raw archive: no fake substitute.
        empty=pd.DataFrame()
        features=build_season_features(year,games_df,advanced_df,empty,
            returning_df,portal_df,roster_cur,roster_prev,talent_df,{},empty,
            empty,empty,empty,empty,empty,empty,plays_df,prior_sp,empty,prior_fpi,empty)
        if features.empty:
            raise RuntimeError(f"CFBD_ARCHIVE_PREVIEW_EMPTY_FEATURES season={year}")
        if features.duplicated(["Season","Game_ID","Team_Key"]).any():
            raise RuntimeError(f"CFBD_ARCHIVE_PREVIEW_DUPLICATE_KEYS season={year}")
        sides=features.groupby(["Season","Game_ID"])["Team_Key"].nunique()
        if (sides!=2).any():
            raise RuntimeError(f"CFBD_ARCHIVE_PREVIEW_SIDE_PARITY_FAIL season={year}")
        features["CFBD_Source_Tag"]="ARCHIVE_V2_PARTIAL_RESEARCH_ONLY"
        features["CFBD_Leakage_Contract"]="SHIFT1_COMPLETED_GAME_STATS__UNVERIFIED_VINTAGE_STATIC_CONTEXT__OFFLINE_RESEARCH_ONLY"
        signals={c:int(features[c].notna().sum()) for c in (
            "CFBD_Season_Net_PPA_Prior","CFBD_Recent3_Off_PPA_Prior",
            "CFBD_Returning_Pct_PPA","CFBD_Talent") if c in features}
        report={"season":year,"schedule_fbs_games":len(fbs_games),
                "completed_scored_fbs_games":len(done),"feature_games":int(sides.size),
                "team_side_rows":len(features),"missing_completed_fbs_family":coverage,
                "non_null_research_signals":signals}
        season_quality.append(report)
        print(f"[CFBD-ARCHIVE-PREVIEW] season={year} status=PASS games={sides.size} rows={len(features)} advanced_prior={signals.get('CFBD_Season_Net_PPA_Prior',0)}",flush=True)
        frames.append(features)
        # Drop large raw payloads promptly; no cross-season live state is reused.
        for cached in list(reader.cache):
            if cached[1] < year-1 or cached[1] == year and cached[0] in ("plays","advanced_game","drives","games"):
                reader.cache.pop(cached,None)
    combined=pd.concat(frames,ignore_index=True,sort=False)
    outpath.parent.mkdir(parents=True,exist_ok=True)
    combined.to_csv(outpath,index=False)
    quality={"status":"OFFLINE_PREVIEW_ONLY_NOT_CERTIFIED_FOR_BQ_PROMOTION",
             "seasons":years,"rows":len(combined),"season_quality":season_quality,
             "archive_root":str(Path(root).expanduser().resolve()),
             "warnings":[
               "Source /stats/game/advanced is weekly archive variant; API loader requests excludeGarbageTime=true. Numerical parity NOT certified.",
               "Unavailable sources remain missing: game havoc, team recruiting ranks, player PPA/usage, QB game PPA, weather, venue and team-location reference, SRS and CORE ratings.",
               "Season roster/talent/returning release timestamps not certified; exclude uncertified static features from sealed pregame selection until vintage audit.",
               "Archive drives recorded for coverage, but this loader derives drive-level aggregates from plays; drives payload not yet used as direct model input.",
               "No BigQuery write, no CFBD API request, no production/model promotion."],
             "output_csv":str(outpath)}
    reportpath=outpath.with_suffix(".quality.json")
    reportpath.write_text(json.dumps(quality,indent=2,default=str)+"\n",encoding="utf-8")
    print(f"[CFBD-ARCHIVE-PREVIEW] status=OFFLINE_PREVIEW_ONLY csv={outpath} report={reportpath} rows={len(combined)}",flush=True)
    return quality


# ---------------------------------------------------------------------------
# BigQuery I/O
# ---------------------------------------------------------------------------

def write_bigquery(df: pd.DataFrame, project: str, dataset: str, table: str, seasons: list[int]) -> None:
    from google.cloud import bigquery
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"
    final = df.copy()
    # Fail closed if an existing warehouse is unreadable: silently truncating
    # all years on a permissions/query error would destroy years of history.
    from google.api_core.exceptions import NotFound
    try:
        client.get_table(table_id)
    except NotFound:
        existing = pd.DataFrame()
    else:
        existing = client.query(f"SELECT * FROM `{table_id}`").to_dataframe(create_bqstorage_client=False)
    if final.duplicated(["Season","Game_ID","Team_Key"]).any():
        raise RuntimeError("CFBD_FEATURE_REFRESH_DUPLICATE_KEYS; refusing destructive replacement")
    side_counts=final.groupby(["Season","Game_ID"],dropna=False)["Team_Key"].nunique(dropna=True)
    if (side_counts != 2).any():
        examples=side_counts[side_counts != 2].head(8).to_dict()
        raise RuntimeError(f"CFBD_FEATURE_SIDE_PARITY_FAILED games={examples}; refusing write")
    if not existing.empty:
        keys=[k for k in ("Season","Game_ID","Team_Key") if k in final.columns and k in existing.columns]
        if len(keys)!=3:
            raise RuntimeError(f"CFBD_FEATURE_REFRESH_KEYS_MISSING keys={keys}; refusing destructive replacement")
        if final.duplicated(keys).any() or existing.duplicated(keys).any():
            raise RuntimeError("CFBD_FEATURE_DUPLICATE_KEYS; refusing destructive replacement")
        old_current=existing.loc[pd.to_numeric(existing["Season"],errors="coerce").isin(seasons)].copy()
        if not old_current.empty:
            for yr in seasons:
                old_n=int(pd.to_numeric(old_current["Season"],errors="coerce").eq(yr).sum())
                fresh_n=int(pd.to_numeric(final["Season"],errors="coerce").eq(yr).sum())
                if fresh_n < old_n:
                    raise RuntimeError(f"CFBD_FEATURE_ROW_REGRESSION season={yr} old={old_n} new={fresh_n}; refusing destructive refresh")
            new=final.set_index(keys); old=old_current.set_index(keys)
            # Only retain past values for missing current cells. Such retained
            # cells are explicitly logged; they must NOT count as a fresh pull.
            old=old.reindex(new.index)
            retained={}
            for col in old.columns:
                if col not in new.columns: new[col]=np.nan
                if col.startswith("CFBD_") and col not in {"CFBD_Source_Tag"}:
                    gap=new[col].isna() & old[col].notna()
                    if gap.any():
                        new.loc[gap,col]=old.loc[gap,col]
                        retained[col]=int(gap.sum())
            final=new.reset_index()
            if retained:
                print(f"[CFBD-PRIOR-VALUE-RETAINED] status=STALE_NOT_FRESH fields={len(retained)} cells={sum(retained.values())} examples={dict(list(retained.items())[:12])}")
        not_refreshed=existing.loc[~pd.to_numeric(existing["Season"],errors="coerce").isin(seasons)]
        if not not_refreshed.empty:
            final=pd.concat([not_refreshed,final],ignore_index=True,sort=False)
    # BigQuery is happier with timezone-aware UTC datetimes and plain strings.
    if "Game_Date" in final.columns:
        final["Game_Date"] = pd.to_datetime(final["Game_Date"], errors="coerce", utc=True)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(final, table_id, job_config=job_config)
    job.result()
    print(f"[CFBD-BQ] status=PASS table={table_id} rows={len(final)} refreshed_seasons={seasons}")


def write_local(df: pd.DataFrame, path: str) -> None:
    if path.lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"[CFBD-LOCAL] status=PASS path={path} rows={len(df)}")



# ---------------------------------------------------------------------------
# Warehouse persistence / reuse of existing NCAA tables
# ---------------------------------------------------------------------------

def _qualified(project: str, dataset: str, table: str) -> str:
    return table if table.count(".") == 2 else f"{project}.{dataset}.{table}"


def _bq_ddl_type(field) -> str:
    """Convert BigQuery SchemaField API types to valid GoogleSQL DDL types.

    The BigQuery Python schema API uses FLOAT/INTEGER/BOOLEAN/RECORD, but
    GoogleSQL ALTER TABLE expects FLOAT64/INT64/BOOL/STRUCT<...>.
    """
    name=str(field.field_type).upper()
    aliases={"FLOAT": "FLOAT64", "INTEGER": "INT64", "BOOLEAN": "BOOL"}
    if name in {"RECORD", "STRUCT"}:
        children=list(getattr(field,"fields",()) or ())
        if not children:
            raise ValueError(f"Cannot safely add a record field without nested schema: {field.name}")
        sql_type="STRUCT<"+", ".join(f"`{child.name}` {_bq_ddl_type(child)}" for child in children)+">"
    else:
        sql_type=aliases.get(name,name)
    if str(getattr(field,"mode","NULLABLE")).upper()=="REPEATED":
        return f"ARRAY<{sql_type}>"
    return sql_type


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _clean_bq_frame(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    for c in out.columns:
        if str(c).endswith(("_At","_UTC")) or c in {"Game_Date","Retrieved_At","Updated_At","Ingested_At"}:
            try: out[c]=pd.to_datetime(out[c],errors="coerce",utc=True)
            except Exception: pass
    return out


def _merge_dataframe(client, table_id: str, df: pd.DataFrame, keys: list[str], *, partition_field: str | None=None) -> dict[str,Any]:
    """Generic schema-evolving BigQuery MERGE for sidecar tables."""
    from google.cloud import bigquery
    if df is None or df.empty:
        return {"status":"EMPTY","table":table_id,"rows":0}
    d=_clean_bq_frame(df).copy()
    tmp=f"{table_id}__stage_{uuid.uuid4().hex[:12]}"
    # Only an actual NotFound means the target may be created. Permission and
    # query errors are never interpreted as an empty warehouse.
    from google.api_core.exceptions import NotFound
    try:
        target=client.get_table(table_id)
    except NotFound:
        cfg=bigquery.LoadJobConfig(write_disposition="WRITE_EMPTY")
        if partition_field and partition_field in d.columns:
            cfg.time_partitioning=bigquery.TimePartitioning(field=partition_field)
        client.load_table_from_dataframe(d,table_id,job_config=cfg).result()
        return {"status":"CREATED","table":table_id,"rows":len(d)}
    if d.duplicated(keys).any():
        raise RuntimeError(f"CFBD_SIDECAR_DUPLICATE_KEYS table={table_id} keys={keys}")
    existing={f.name:f for f in target.schema}
    try:
        # Stage inferred columns, and remove staging even on ALTER/MERGE error.
        client.load_table_from_dataframe(d,tmp,job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")).result()
        for f in client.get_table(tmp).schema:
            if f.name not in existing:
                client.query(f"ALTER TABLE `{table_id}` ADD COLUMN IF NOT EXISTS `{f.name}` {_bq_ddl_type(f)}").result()
        cols=list(d.columns)
        on=" AND ".join([f"T.`{k}` IS NOT DISTINCT FROM S.`{k}`" for k in keys])
        updates=[c for c in cols if c not in keys]
        update_sql=",\n      ".join([f"`{c}`=COALESCE(S.`{c}`, T.`{c}`)" if c.startswith("CFBD_") else f"`{c}`=S.`{c}`" for c in updates]) or f"`{keys[0]}`=S.`{keys[0]}`"
        insert_cols=", ".join([f"`{c}`" for c in cols]); insert_vals=", ".join([f"S.`{c}`" for c in cols])
        client.query(f"""
          MERGE `{table_id}` T USING `{tmp}` S ON {on}
          WHEN MATCHED THEN UPDATE SET {update_sql}
          WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
        """).result()
        return {"status":"PASS","table":table_id,"rows":len(d)}
    finally:
        client.delete_table(tmp,not_found_ok=True)


def replace_player_history_seasons(client, table_id: str, df: pd.DataFrame, *, min_retention: float = 0.80) -> dict[str, Any]:
    """Atomically replace complete player-season snapshots WITHOUT a 3-key MERGE join.

    BigQuery on-demand rejects the prior equality-join MERGE (8,175 slot CPU
    seconds against only 26 MB scanned). An ON FALSE MERGE is a documented
    constant-false partition replacement; it avoids a target/source join.
    Only seasons fully represented in the incoming snapshot can be replaced.
    """
    from google.cloud import bigquery
    from google.api_core.exceptions import NotFound
    keys = ["Season", "Team_Key", "Player_Key"]
    if df is None or df.empty:
        return {"status": "EMPTY", "rows": 0, "table": table_id}
    d = _clean_bq_frame(df).copy()
    if not set(keys).issubset(d.columns):
        raise RuntimeError(f"CFBD_PLAYER_KEYS_MISSING {sorted(set(keys)-set(d.columns))}")
    d["Season"] = pd.to_numeric(d["Season"], errors="raise").astype("Int64")
    if d[keys].isna().any().any() or d.duplicated(keys).any():
        raise RuntimeError("CFBD_PLAYER_NULL_OR_DUPLICATE_KEYS: no writes made")
    for key in ["Team_Key", "Player_Key"]:
        if not d[key].astype(str).str.strip().ne("").all():
            raise RuntimeError(f"CFBD_PLAYER_BLANK_KEY {key}: no writes made")
    season_counts = {int(k):int(v) for k,v in d.groupby("Season").size().items()}
    if any(v < 100 for v in season_counts.values()):
        raise RuntimeError(f"CFBD_PLAYER_SNAPSHOT_TOO_SMALL {season_counts}: no writes made")
    try:
        target = client.get_table(table_id)
    except NotFound:
        client.load_table_from_dataframe(d, table_id, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_EMPTY")).result()
        return {"status":"CREATED","rows":len(d),"seasons":season_counts,"table":table_id}

    columns = list(d.columns)
    extra_cols = sorted(f.name for f in target.schema if f.name not in columns)
    # An unrelated job may have enriched the target. Do not erase those fields.
    if extra_cols:
        raise RuntimeError(f"CFBD_PLAYER_TARGET_HAS_EXTRA_COLUMNS {extra_cols}; refusing season replacement")
    stage = f"{table_id}__player_stage_{uuid.uuid4().hex[:12]}"
    # Retain staging on any failure so a live failure does not discard the work.
    client.load_table_from_dataframe(d, stage, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_EMPTY")).result()
    print(f"[CFBD-PLAYER-STAGED] stage={stage} rows={len(d)} seasons={season_counts}")
    try:
        stage_schema = {f.name: f for f in client.get_table(stage).schema}
        target_schema = {f.name: f for f in target.schema}
        for col in columns:
            if col not in target_schema:
                client.query(f"ALTER TABLE `{table_id}` ADD COLUMN IF NOT EXISTS `{col}` {_bq_ddl_type(stage_schema[col])}").result()
        from google.cloud import bigquery as bq
        years = sorted(season_counts)
        cfg = bq.QueryJobConfig(query_parameters=[bq.ArrayQueryParameter("years", "INT64", years)])
        before = client.query(f"SELECT Season, COUNT(*) n FROM `{table_id}` WHERE Season IN UNNEST(@years) GROUP BY Season",job_config=cfg).result()
        existing_counts={int(r.Season):int(r.n) for r in before}
        decreases={year:{"previous":existing_counts[year],"incoming":size}
                   for year,size in season_counts.items()
                   if year in existing_counts and size < existing_counts[year]*min_retention}
        if decreases:
            raise RuntimeError(f"CFBD_PLAYER_SNAPSHOT_SHRINK_GUARD {decreases}: stage retained for inspection")
        names=", ".join(f"`{c}`" for c in columns)
        values=", ".join(f"S.`{c}`" for c in columns)
        # ON FALSE means there is NO expensive three-key equality join. One
        # atomic DML removes only requested seasons and inserts their snapshot.
        sql=f"""MERGE `{table_id}` AS T USING `{stage}` AS S ON FALSE
          WHEN NOT MATCHED BY SOURCE AND T.Season IN UNNEST(@years) THEN DELETE
          WHEN NOT MATCHED THEN INSERT ({names}) VALUES ({values})"""
        client.query(sql,job_config=cfg).result()
        after = client.query(f"SELECT Season, COUNT(*) n FROM `{table_id}` WHERE Season IN UNNEST(@years) GROUP BY Season",job_config=cfg).result()
        observed={int(r.Season):int(r.n) for r in after}
        if observed != season_counts:
            raise RuntimeError(f"CFBD_PLAYER_READBACK_MISMATCH expected={season_counts} observed={observed}; stage retained")
        client.delete_table(stage,not_found_ok=True)
        print(f"[CFBD-PLAYER-READBACK] status=PASS seasons={season_counts}")
        return {"status":"PASS","table":table_id,"rows":len(d),"seasons":season_counts,
                "strategy":"JOIN_FREE_ATOMIC_SEASON_REPLACE","readback":"PASS"}
    except Exception:
        print(f"[CFBD-PLAYER-RECOVERY-STAGE] preserved={stage} retry_without_CFBD=TRUE")
        raise


def audit_player_history(client, project: str, dataset: str, seasons: list[int]) -> dict[str,Any]:
    """Read-only sidecar coverage; never confuse stored rows with source freshness."""
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery
    tid=_qualified(project,dataset,PLAYER_HISTORY_TABLE)
    try: t=client.get_table(tid)
    except NotFound:
        print(f"[CFBD-PLAYER-AUDIT] status=MISSING_TABLE table={tid} api_calls=0")
        return {"status":"MISSING_TABLE","table":tid}
    colnames={f.name for f in t.schema}
    if not {"Season","Team_Key","Player_Key"}.issubset(colnames):
        raise RuntimeError(f"CFBD_PLAYER_AUDIT_MISSING_KEYS table={tid}")
    expressions=["COUNT(*) AS n", "COUNT(DISTINCT CONCAT(CAST(Team_Key AS STRING), '|', CAST(Player_Key AS STRING))) AS unique_players"]
    for col in ["PPA_Total","Usage_Overall","Player_ID","Player_Name"]:
        if col in colnames: expressions.append(f"COUNTIF(`{col}` IS NOT NULL) AS `{col}_non_null`")
    if "Ingested_At" in colnames: expressions.append("MAX(Ingested_At) AS latest_ingested_at")
    cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("years","INT64",seasons)])
    query=f"SELECT Season, {', '.join(expressions)} FROM `{tid}` WHERE Season IN UNNEST(@years) GROUP BY Season ORDER BY Season"
    results=list(client.query(query,job_config=cfg).result())
    by_year={int(r.Season):int(r.n) for r in results}
    for year in seasons:
        matches=[r for r in results if int(r.Season)==year]
        if matches:
            row=dict(matches[0].items()); row.pop("Season",None)
            print(f"[CFBD-PLAYER-AUDIT-SEASON] year={year} status=PRESENT_UNVERIFIED_FRESHNESS {row}")
        else:
            print(f"[CFBD-PLAYER-AUDIT-SEASON] year={year} status=MISSING rows=0")
    status="PRESENT_UNVERIFIED_FRESHNESS" if len(by_year)==len(seasons) and all(n>=100 for n in by_year.values()) else "PARTIAL"
    print(f"[CFBD-PLAYER-AUDIT] status={status} seasons={seasons} cfbd_api_calls=0")
    return {"status":status,"counts":by_year,"table":tid,"api_calls":0}


def repair_player_history_from_stages(client, project: str, dataset: str) -> dict[str,Any]:
    """Resume a prior failed player write using its BigQuery staging table; no API."""
    table_id=_qualified(project,dataset,PLAYER_HISTORY_TABLE)
    prefix=f"{PLAYER_HISTORY_TABLE}__player_stage_"
    stages=[_qualified(project,dataset,x.table_id) for x in client.list_tables(f"{project}.{dataset}")
            if x.table_id.startswith(prefix) and re.fullmatch(r"[a-zA-Z0-9_]+",x.table_id)]
    print(f"[CFBD-PLAYER-STAGE-PLAN] stages={len(stages)} cfbd_api_calls=0")
    if not stages:
        return {"status":"NO_STAGES_FOUND","cfbd_api_calls":0}
    repaired=[]
    for stage in sorted(stages):
        d=client.query(f"SELECT * FROM `{stage}`").to_dataframe(create_bqstorage_client=False)
        if d.empty: raise RuntimeError(f"CFBD_PLAYER_EMPTY_RECOVERY_STAGE {stage}")
        result=replace_player_history_seasons(client,table_id,d)
        if result['status']!='PASS': raise RuntimeError(f"CFBD_PLAYER_STAGE_REPAIR_FAILED {result}")
        client.delete_table(stage,not_found_ok=True)
        repaired.append({"stage":stage,"rows":result['rows']})
        print(f"[CFBD-PLAYER-STAGE-RECOVERED] stage={stage} rows={result['rows']} api_calls=0")
    return {"status":"PASS","repaired":repaired,"cfbd_api_calls":0}


def repair_player_history_with_minimal_cfbd(client, project: str, dataset: str, seasons: list[int]) -> dict[str,Any]:
    """Refresh only roster/player-PPA/usage endpoints; never rerun full CFBD backfill.

    Archive each season immediately before attempting any later API request or
    player DML. A failed season cannot erase earlier downloaded responses.
    """
    if not seasons: raise RuntimeError("CFBD_PLAYER_REPAIR_REQUIRES_SEASONS")
    c=CFBDClient()
    frames=[]; missing=[]; archive_offset=0
    for year in seasons:
        roster=pull_roster(c,year)
        ppa=pull_player_season_ppa(c,year)
        usage=pull_player_usage(c,year)
        sizes={"roster":len(roster),"ppa":len(ppa),"usage":len(usage)}
        print(f"[CFBD-PLAYER-REPAIR-SOURCE] year={year} counts={sizes} api_calls={c.call_count}")
        # Persist everything fetched for this year before fetching another year.
        new_archive=c.archive_rows[archive_offset:]
        if new_archive:
            ares=write_api_archive(client,project,dataset,new_archive)
            archive_offset=len(c.archive_rows)
            print(f"[CFBD-PLAYER-REPAIR-ARCHIVE] year={year} status={ares['status']} rows={ares.get('rows')} api_calls={c.call_count}")
        if not all(sizes.values()):
            missing.append(year)
            print(f"[CFBD-PLAYER-REPAIR-SKIP] year={year} reason=PARTIAL_SOURCE existing_rows_preserved=TRUE")
            continue
        frames.append(build_player_history_sidecar(year,roster,ppa,usage))
    if not frames:
        raise RuntimeError(f"CFBD_PLAYER_REPAIR_NO_COMPLETE_SEASONS seasons={seasons}; existing rows preserved")
    out=pd.concat(frames,ignore_index=True,sort=False)
    res=replace_player_history_seasons(client,_qualified(project,dataset,PLAYER_HISTORY_TABLE),out)
    print(f"[CFBD-PLAYER-REPAIR-COMPLETE] status={'PARTIAL' if missing else res['status']} seasons={seasons} missing={missing} rows={len(out)} api_calls={c.call_count}")
    if missing:
        raise RuntimeError(f"CFBD_PLAYER_REPAIR_PARTIAL_SOURCE seasons={missing}; good seasons committed, missing seasons retained")
    return res


# Immutable API-archive schema contract. Season may be absent on endpoints that
# are not season-scoped; use nullable INT64, never pandas float inference.
_ARCHIVE_COLUMN_TYPES = {
    "Archive_Row_ID": "STRING",
    "Archive_Batch_ID": "STRING",
    "Endpoint": "STRING",
    "Params_JSON": "STRING",
    "Season": "INT64",
    "Retrieved_At": "TIMESTAMP",
    "Row_Index": "INT64",
    "Payload_JSON": "STRING",
    "Source_Tag": "STRING",
}


def _archive_prepare_frame(archive_rows: list[dict[str, Any]]) -> pd.DataFrame:
    d = pd.DataFrame(archive_rows)
    missing = set(_ARCHIVE_COLUMN_TYPES) - set(d.columns)
    if missing:
        raise RuntimeError(f"CFBD_ARCHIVE_SCHEMA_MISSING columns={sorted(missing)}")
    d = d[list(_ARCHIVE_COLUMN_TYPES)].drop_duplicates("Archive_Row_ID", keep="first").copy()
    if d["Archive_Row_ID"].isna().any():
        raise RuntimeError("CFBD_ARCHIVE_NULL_KEY")
    for name in ("Season", "Row_Index"):
        numeric = pd.to_numeric(d[name], errors="coerce")
        bad = d[name].notna() & (numeric.isna() | (numeric % 1 != 0))
        if bad.any():
            raise RuntimeError(f"CFBD_ARCHIVE_INVALID_INTEGER column={name} rows={int(bad.sum())}")
        d[name] = numeric.astype("Int64")
    return _clean_bq_frame(d)


def _archive_explicit_schema():
    from google.cloud import bigquery
    return [bigquery.SchemaField(name, field_type, mode="NULLABLE")
            for name, field_type in _ARCHIVE_COLUMN_TYPES.items()]


def _archive_assert_stage(client, stage_id: str) -> int:
    """Never silently null invalid seasons or MERGE duplicate archive IDs."""
    actual={f.name for f in client.get_table(stage_id).schema}
    missing=set(_ARCHIVE_COLUMN_TYPES)-actual
    if missing:
        raise RuntimeError(f"CFBD_ARCHIVE_STAGE_MISSING_COLUMNS stage={stage_id} columns={sorted(missing)}")
    result=list(client.query(f"""
        SELECT COUNT(*) AS n, COUNT(DISTINCT Archive_Row_ID) AS unique_keys,
          COUNTIF(Archive_Row_ID IS NULL) AS null_keys,
          COUNTIF(Season IS NOT NULL AND (
            SAFE_CAST(Season AS INT64) IS NULL
            OR SAFE_CAST(Season AS FLOAT64) != SAFE_CAST(SAFE_CAST(Season AS INT64) AS FLOAT64)
          )) AS bad_seasons
        FROM `{stage_id}`
    """).result())[0]
    n=int(result.n)
    if n == 0 or int(result.unique_keys) != n or int(result.null_keys) or int(result.bad_seasons):
        raise RuntimeError(f"CFBD_ARCHIVE_INVALID_STAGE stage={stage_id} rows={n} unique={result.unique_keys} null={result.null_keys} invalid_seasons={result.bad_seasons}")
    return n


def _archive_upsert_stage(client, target_id: str, stage_id: str) -> dict[str, Any]:
    """Explicit, name-matched INSERT; CAST fixes legacy FLOAT64 staging Season."""
    target_schema={f.name: str(f.field_type).upper() for f in client.get_table(target_id).schema}
    missing=set(_ARCHIVE_COLUMN_TYPES)-set(target_schema)
    if missing:
        raise RuntimeError(f"CFBD_ARCHIVE_TARGET_SCHEMA_MISSING columns={sorted(missing)}")
    type_aliases={"INTEGER":"INT64", "FLOAT":"FLOAT64", "BOOLEAN":"BOOL"}
    wrong={name:{"actual":target_schema[name],"expected":type_name}
           for name,type_name in _ARCHIVE_COLUMN_TYPES.items()
           if type_aliases.get(target_schema[name],target_schema[name]) != type_name}
    if wrong:
        raise RuntimeError(f"CFBD_ARCHIVE_TARGET_WRONG_TYPES {wrong}")
    n=_archive_assert_stage(client,stage_id)
    cols=list(_ARCHIVE_COLUMN_TYPES)
    names=", ".join(f"`{x}`" for x in cols)
    values=", ".join(
        f"CAST(S.`{x}` AS INT64)" if x in {"Season", "Row_Index"}
        else f"S.`{x}`" for x in cols
    )
    # Archive is immutable: existing hashes are never updated; retry is idempotent.
    client.query(f"""
        MERGE `{target_id}` T USING `{stage_id}` S
        ON T.Archive_Row_ID=S.Archive_Row_ID
        WHEN NOT MATCHED THEN INSERT ({names}) VALUES ({values})
    """).result()
    counts=list(client.query(f"""
        SELECT COUNT(*) AS present
        FROM `{stage_id}` S
        JOIN `{target_id}` T ON T.Archive_Row_ID=S.Archive_Row_ID
    """).result())[0]
    present=int(counts.present)
    if present != n:
        raise RuntimeError(f"CFBD_ARCHIVE_READBACK_FAILED stage={stage_id} expected={n} found={present}")
    return {"status":"PASS", "target":target_id, "stage":stage_id, "stage_rows":n,
            "verified_rows":present, "api_calls":0}


def write_api_archive(client, project: str, dataset: str, archive_rows: list[dict[str,Any]]) -> dict[str,Any]:
    """Append content-addressed API snapshots; explicit schema and readback."""
    if not archive_rows:
        return {"status":"EMPTY","rows":0}
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery
    d=_archive_prepare_frame(archive_rows)
    tid=_qualified(project,dataset,API_ARCHIVE_TABLE)
    schema=_archive_explicit_schema()
    try:
        client.get_table(tid)
    except NotFound:
        cfg=bigquery.LoadJobConfig(write_disposition="WRITE_EMPTY", schema=schema)
        cfg.time_partitioning=bigquery.TimePartitioning(field="Retrieved_At")
        client.load_table_from_dataframe(d,tid,job_config=cfg).result()
        return {"status":"CREATED","table":tid,"rows":len(d)}
    # Unique staging IDs are left in place on failure so --repair-archive
    # can recover a partially completed run without another API request.
    stage=f"{tid}__stage_{uuid.uuid4().hex[:12]}"
    cfg=bigquery.LoadJobConfig(write_disposition="WRITE_EMPTY", schema=schema)
    client.load_table_from_dataframe(d,stage,job_config=cfg).result()
    print(f"[CFBD-ARCHIVE-STAGED] table={stage} rows={len(d)}")
    res=_archive_upsert_stage(client,tid,stage)
    client.delete_table(stage,not_found_ok=True)
    print(f"[CFBD-ARCHIVE-READBACK] stage_rows={res['stage_rows']} verified={res['verified_rows']}")
    return {"status":"PASS","table":tid,"rows":len(d),"verified_rows":res['verified_rows']}


def _archive_leftover_stage_tables(client, project: str, dataset: str) -> list[str]:
    """Discover only our content-addressed archive's *explicitly named* stages."""
    prefix=f"{API_ARCHIVE_TABLE}__stage_"
    stages=[]
    for item in client.list_tables(f"{project}.{dataset}"):
        name=item.table_id
        if name.startswith(prefix) and re.fullmatch(r"[a-zA-Z0-9_]+",name):
            stages.append(_qualified(project,dataset,name))
    return sorted(stages)


def repair_api_archive_from_stages(client, project: str, dataset: str) -> dict[str,Any]:
    """Recover *original* successful API responses from leftover BigQuery stage tables.

    Reads only explicitly named ncaaf_cfbd_api_archive__stage_* tables. This
    never creates synthetic snapshots or calls CollegeFootballData.
    """
    tid=_qualified(project,dataset,API_ARCHIVE_TABLE)
    stages=_archive_leftover_stage_tables(client,project,dataset)
    print(f"[CFBD-ARCHIVE-RECOVERY-PLAN] stage_tables={len(stages)} cfbd_api_calls=0")
    if not stages:
        return {"status":"NO_STAGES_FOUND","recovered":0,"cfbd_api_calls":0,
                "note":"Previous staging tables may have been deleted; cannot recreate unsaved API responses."}
    # Validate all candidates before ANY writes or deletes.
    counts={stage:_archive_assert_stage(client,stage) for stage in stages}
    recovered=0
    for stage in stages:
        res=_archive_upsert_stage(client,tid,stage)
        recovered+=res["verified_rows"]
        print(f"[CFBD-ARCHIVE-RECOVERED] stage={stage} verified={res['verified_rows']} api_calls=0")
        # Successful stages can be safely cleaned up only after readback.
        client.delete_table(stage,not_found_ok=True)
    print(f"[CFBD-ARCHIVE-RECOVERY] status=PASS stages={len(stages)} verified_rows={recovered} cfbd_api_calls=0")
    return {"status":"PASS","stages":len(stages),"verified_rows":recovered,"cfbd_api_calls":0}


def build_personnel_sidecar(features: pd.DataFrame) -> pd.DataFrame:
    if features is None or features.empty: return pd.DataFrame()
    static_tokens=("Returning_","Roster_","Portal_","Talent","Recruit_")
    cols=[c for c in features.columns if c.startswith("CFBD_") and any(t in c for t in static_tokens) and not c.startswith("CFBD_Opp_") and not c.startswith("CFBD_Matchup_")]
    base=[c for c in ["Season","Team","Team_Key"] if c in features.columns]
    if not cols: return pd.DataFrame()
    d=features[base+cols].sort_values(base).drop_duplicates(["Season","Team_Key"],keep="last").copy()
    d["Source_Name"]="CollegeFootballData"; d["Updated_At"]=_utc_now(); d["Source_Tag"]=SOURCE_TAG
    return d


def build_player_history_sidecar(year: int, roster: pd.DataFrame, ppa: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    """Latest normalized player-season snapshot; raw point-in-time rows remain in API archive."""
    parts=[]
    if roster is not None and not roster.empty:
        r=roster.copy(); z=pd.DataFrame(index=r.index)
        z["Season"]=int(year); z["Team"]=_first_col(r,"team","school").astype(str); z["Team_Key"]=z["Team"].map(_team_key)
        z["Player_ID"]=_first_col(r,"id","athleteId","playerId").astype(str).replace({"nan":"","None":""})
        z["Player_Name"]=(_first_col(r,"firstName").astype(str).str.strip()+" "+_first_col(r,"lastName").astype(str).str.strip()).str.strip()
        # Some roster payloads use a single name field.
        one=_first_col(r,"name").astype(str).str.strip(); z["Player_Name"]=z["Player_Name"].where(z["Player_Name"].str.strip().ne(""),one)
        z["Player_Key"]=z["Player_Name"].map(_person_key)
        z["Position"]=_first_col(r,"position").astype(str).str.upper().str.strip()
        z["Jersey"]=_first_col(r,"jersey").astype(str); z["Height"]=_num(_first_col(r,"height")); z["Weight"]=_num(_first_col(r,"weight"))
        parts.append(z)
    base=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=["Season","Team","Team_Key","Player_ID","Player_Name","Player_Key","Position"])
    if ppa is not None and not ppa.empty:
        x=ppa.copy(); q=pd.DataFrame(index=x.index)
        q["Team_Key"]=_first_col(x,"team").map(_team_key); q["Player_Name"]=_first_col(x,"name").astype(str); q["Player_Key"]=q["Player_Name"].map(_person_key)
        q["PPA_Total"]=_num(_first_col(x,"totalPPA.all","totalPPA.total","total_ppa_all")); q["PPA_Avg"]=_num(_first_col(x,"averagePPA.all","averagePPA.total","average_ppa_all"))
        q["PPA_Pass"]=_num(_first_col(x,"totalPPA.pass","total_passing_ppa")); q["PPA_Rush"]=_num(_first_col(x,"totalPPA.rush","total_rushing_ppa"))
        q=q.drop_duplicates(["Team_Key","Player_Key"],keep="last")
        base=base.merge(q[["Team_Key","Player_Key","PPA_Total","PPA_Avg","PPA_Pass","PPA_Rush"]],on=["Team_Key","Player_Key"],how="outer")
    if usage is not None and not usage.empty:
        u=usage.copy(); q=pd.DataFrame(index=u.index)
        q["Team_Key"]=_first_col(u,"team").map(_team_key); q["Player_Name_U"]=_first_col(u,"name").astype(str); q["Player_Key"]=q["Player_Name_U"].map(_person_key)
        q["Usage_Overall"]=_num(_first_col(u,"usage.overall","overall")); q["Usage_Pass"]=_num(_first_col(u,"usage.pass","passing")); q["Usage_Rush"]=_num(_first_col(u,"usage.rush","rushing"))
        q=q.drop_duplicates(["Team_Key","Player_Key"],keep="last")
        base=base.merge(q.drop(columns=["Player_Name_U"]),on=["Team_Key","Player_Key"],how="outer")
    base["Season"]=pd.to_numeric(base.get("Season"),errors="coerce").fillna(int(year)).astype("Int64")
    if "Team" not in base.columns: base["Team"]=""
    if "Player_Name" not in base.columns: base["Player_Name"]=""
    if "Position" not in base.columns: base["Position"]=""
    base["Source_Name"]="CollegeFootballData"; base["Ingested_At"]=_utc_now(); base["Source_Tag"]=SOURCE_TAG
    return base.loc[base["Team_Key"].astype(str).ne("") & base["Player_Key"].astype(str).ne("")].drop_duplicates(["Season","Team_Key","Player_Key"],keep="last")


def merge_environment_into_travel(client, project: str, dataset: str, features: pd.DataFrame) -> dict[str,Any]:
    """Extend the existing travel reference in place with CFBD environment fields."""
    if features is None or features.empty: return {"status":"EMPTY"}
    tid=_qualified(project,dataset,TRAVEL_TABLE)
    try: client.get_table(tid)
    except Exception as exc:
        return {"status":"TRAVEL_TABLE_UNAVAILABLE","table":tid,"error":f"{type(exc).__name__}:{exc}"}
    d=features.copy(); d["Is_Home"]=_num(d.get("Is_Home",pd.Series(np.nan,index=d.index)))
    home=d.loc[d["Is_Home"].eq(1)].copy(); away=d.loc[d["Is_Home"].eq(0)].copy()
    if home.empty: return {"status":"NO_HOME_ROWS","table":tid}
    keep_game=["Game_ID","Season","CFBD_Venue_Elevation","CFBD_Venue_Latitude","CFBD_Venue_Longitude","CFBD_Venue_Dome","CFBD_Temperature","CFBD_DewPoint","CFBD_Humidity","CFBD_Precipitation","CFBD_Snowfall","CFBD_WindSpeed","CFBD_Weather_Condition","CFBD_Game_Indoors"]
    env=home[[c for c in keep_game if c in home.columns]].drop_duplicates(["Season","Game_ID"],keep="last")
    hp=home[[c for c in ["Season","Game_ID","CFBD_Home_Elevation","CFBD_Altitude_Gain"] if c in home.columns]].rename(columns={"CFBD_Home_Elevation":"CFBD_Home_Team_Elevation","CFBD_Altitude_Gain":"CFBD_Home_Altitude_Gain"})
    ap=away[[c for c in ["Season","Game_ID","CFBD_Home_Elevation","CFBD_Altitude_Gain"] if c in away.columns]].rename(columns={"CFBD_Home_Elevation":"CFBD_Away_Team_Elevation","CFBD_Altitude_Gain":"CFBD_Away_Altitude_Gain"})
    env=env.merge(hp,on=["Season","Game_ID"],how="left").merge(ap,on=["Season","Game_ID"],how="left")
    env["CFBD_Environment_Updated_At"]=_utc_now(); env["CFBD_Environment_Source_Tag"]=SOURCE_TAG
    # Add columns using inferred schema then MERGE onto existing travel rows only.
    from google.cloud import bigquery
    tmp=f"{tid}__cfbd_stage_{uuid.uuid4().hex[:12]}"
    try:
        client.load_table_from_dataframe(_clean_bq_frame(env),tmp,job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")).result()
        target=client.get_table(tid); existing={f.name:f for f in target.schema}
        for f in client.get_table(tmp).schema:
            if f.name not in existing:
                client.query(f"ALTER TABLE `{tid}` ADD COLUMN IF NOT EXISTS `{f.name}` {_bq_ddl_type(f)}").result()
        updates=[c for c in env.columns if c not in {"Season","Game_ID"}]
        us=",\n      ".join([f"`{c}`=COALESCE(S.`{c}`, T.`{c}`)" if c.startswith("CFBD_") else f"`{c}`=S.`{c}`" for c in updates])
        client.query(f"""
          MERGE `{tid}` T USING `{tmp}` S
          ON T.Season=S.Season AND CAST(T.Game_ID AS STRING)=CAST(S.Game_ID AS STRING)
          WHEN MATCHED THEN UPDATE SET {us}
        """).result()
        yr=sorted({int(y) for y in pd.to_numeric(env["Season"],errors="coerce").dropna().tolist()})
        mdf=client.query(f"SELECT COUNT(*) n FROM `{tid}` WHERE CFBD_Environment_Source_Tag=@tag AND Season IN UNNEST(@years)",job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("tag","STRING",SOURCE_TAG),bigquery.ArrayQueryParameter("years","INT64",yr)])).to_dataframe(create_bqstorage_client=False)
        n=int(mdf.iloc[0]["n"]) if len(mdf) else 0
        if n == 0:
            raise RuntimeError(f"CFBD_TRAVEL_NO_MATCHING_ROWS seasons={yr}; check travel.Game_ID mapping")
        return {"status":"PASS","table":tid,"source_rows":len(env),"updated_rows":n}
    finally:
        client.delete_table(tmp,not_found_ok=True)


def _median(values) -> float:
    z=pd.to_numeric(pd.Series(list(values),dtype="object"),errors="coerce").dropna()
    return float(z.median()) if len(z) else np.nan


def _extract_line_records(lines_raw: pd.DataFrame) -> dict[str,dict[str,Any]]:
    """Normalize CFBD's nested/provider line payload to one provider-consensus record per game."""
    if lines_raw is None or lines_raw.empty: return {}
    records={}
    for _,r in lines_raw.iterrows():
        gid=r.get("id",r.get("gameId",r.get("game_id")))
        if pd.isna(gid): continue
        # pd.json_normalize may leave the nested list in `lines`.
        providers=r.get("lines",None)
        if not isinstance(providers,list):
            providers=[]
            # Also tolerate already-flattened single-provider rows.
            if any(k in r.index for k in ["spread","spreadOpen","overUnder","provider"]): providers=[dict(r)]
        good=[p for p in providers if isinstance(p,dict)]
        if not good:
            records[str(int(float(gid))) if str(gid).replace('.0','').isdigit() else str(gid)]={"providers":[]}
            continue
        consensus=[p for p in good if "consensus" in str(p.get("provider") or "").lower()]
        src=consensus if consensus else good
        def vals(*names):
            out=[]
            for p in src:
                for n in names:
                    if p.get(n) is not None: out.append(p.get(n)); break
            return out
        rec={
            "spread":_median(vals("spread")),"spread_open":_median(vals("spreadOpen","openingSpread","openSpread")),
            "total":_median(vals("overUnder","total")),"total_open":_median(vals("overUnderOpen","openingOverUnder","openTotal")),
            "home_ml":_median(vals("homeMoneyline","homeMoneyLine")),"away_ml":_median(vals("awayMoneyline","awayMoneyLine")),
            "providers":good,
        }
        records[str(int(float(gid))) if str(gid).replace('.0','').isdigit() else str(gid)]=rec
    return records


def _alignment_map(client, project: str, dataset: str, season: int) -> dict[str,dict[str,Any]]:
    """Season-aware canonical team mapping from the warehouse's existing alignment history."""
    table=f"{project}.{dataset}.team_sport_alignment_history"
    try:
        from google.cloud import bigquery
        cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("season","INT64",int(season))])
        d=client.query(f"""
          SELECT Team, Team_Norm, Canonical_Team_Name, Canonical_Team_Norm, Conference, Subdivision,
                 ROW_NUMBER() OVER(PARTITION BY LOWER(TRIM(Team_Norm)) ORDER BY Effective_Season_From DESC) rn
          FROM `{table}`
          WHERE UPPER(TRIM(Sport))='NCAAF' AND Effective_Season_From <= @season AND COALESCE(Use_For_Context,TRUE)
        """,job_config=cfg).to_dataframe(create_bqstorage_client=False)
    except Exception:
        return {}
    d=d.loc[d["rn"].eq(1)].copy(); out={}
    for _,r in d.iterrows():
        for v in [r.get("Team"),r.get("Team_Norm"),r.get("Canonical_Team_Name"),r.get("Canonical_Team_Norm")]:
            k=_team_key(v)
            if k: out[k]=dict(r)
    return out


def _result_label(a,b,spread=None,total=None,kind="su"):
    try: a=float(a); b=float(b)
    except Exception: return None
    if kind=="su": return "WIN" if a>b else "LOSS" if a<b else "TIE"
    if kind=="ats":
        try: v=a+float(spread)-b
        except Exception: return None
        return "WIN" if v>1e-9 else "LOSS" if v<-1e-9 else "PUSH"
    if kind=="total":
        try: v=a+b-float(total)
        except Exception: return None
        return "OVER" if v>1e-9 else "UNDER" if v<-1e-9 else "PUSH"
    return None


def build_canonical_history_rows(games_raw: pd.DataFrame, lines_raw: pd.DataFrame, season: int, alignment: dict[str,dict[str,Any]]) -> pd.DataFrame:
    """Map CFBD game/results/lines into the existing fixed raw-table schema conceptually.

    The actual target schema is read from BigQuery at load time, so this function
    only emits fields supported by the existing table and never alters its schema.
    """
    if games_raw is None or games_raw.empty: return pd.DataFrame()
    lm=_extract_line_records(lines_raw)
    rows=[]
    for _,g in games_raw.iterrows():
        gid=g.get("id",g.get("gameId")); start=pd.to_datetime(g.get("startDate",g.get("startTime")),errors="coerce",utc=True)
        if pd.isna(gid) or pd.isna(start): continue
        h=str(g.get("homeTeam") or "").strip(); a=str(g.get("awayTeam") or "").strip(); hk=_team_key(h); ak=_team_key(a)
        hm=alignment.get(hk); am=alignment.get(ak)
        if hm is None or am is None: continue  # fail closed: never invent canonical identities
        hs=pd.to_numeric(pd.Series([g.get("homePoints",g.get("homeScore"))]),errors="coerce").iloc[0]
        aas=pd.to_numeric(pd.Series([g.get("awayPoints",g.get("awayScore"))]),errors="coerce").iloc[0]
        line=lm.get(str(int(float(gid))) if str(gid).replace('.0','').isdigit() else str(gid),{})
        hsp=line.get("spread",np.nan); hop=line.get("spread_open",np.nan); tot=line.get("total",np.nan); top=line.get("total_open",np.nan)
        hml=line.get("home_ml",np.nan); aml=line.get("away_ml",np.nan); raw=json.dumps(line.get("providers",[]),sort_keys=True,default=str)
        neutral=bool(g.get("neutralSite") is True or str(g.get("neutralSite")).lower()=="true")
        st_et=start.tz_convert("America/New_York")
        for is_home,team,opp,tm,om,ts,os_,sp,osp,ml in [
            (1,h,a,hm,am,hs,aas,hsp,hop,hml),(0,a,h,am,hm,aas,hs,(-hsp if pd.notna(hsp) else np.nan),(-hop if pd.notna(hop) else np.nan),aml)
        ]:
            close_sp=float(sp) if pd.notna(sp) else np.nan; open_sp=float(osp) if pd.notna(osp) else np.nan
            row={
              "Sport":"NCAAF","Season":int(season),"Source_Name":"CollegeFootballData","Source_Grain":"GAME_SIDE_CONSENSUS","Source_Game_ID":str(int(float(gid))) if str(gid).replace('.0','').isdigit() else str(gid),
              "Game_Date":start.date().isoformat(),"Week":g.get("week"),"Start_Time_ET":st_et.strftime("%Y-%m-%d %H:%M:%S%z"),
              "Source_Team_Name":team,"Source_Opponent_Name":opp,"Team":tm.get("Team") or team,"Team_Norm":str(tm.get("Team_Norm") or _team_key(team)).strip().lower(),
              "Canonical_Team_Name":tm.get("Canonical_Team_Name") or tm.get("Team") or team,"Canonical_Team_Norm":str(tm.get("Canonical_Team_Norm") or tm.get("Team_Norm") or _team_key(team)).strip().lower(),
              "Opponent":om.get("Team") or opp,"Opponent_Norm":str(om.get("Team_Norm") or _team_key(opp)).strip().lower(),
              "Opponent_Canonical_Team_Name":om.get("Canonical_Team_Name") or om.get("Team") or opp,"Opponent_Canonical_Team_Norm":str(om.get("Canonical_Team_Norm") or om.get("Team_Norm") or _team_key(opp)).strip().lower(),
              "Conference":tm.get("Conference") or (g.get("homeConference") if is_home else g.get("awayConference")),"Opponent_Conference":om.get("Conference") or (g.get("awayConference") if is_home else g.get("homeConference")),
              "Subdivision":tm.get("Subdivision") or "FBS","Opponent_Subdivision":om.get("Subdivision") or "FBS","Venue":"Neutral" if neutral else ("Home" if is_home else "Road"),
              "Is_Home":int(is_home and not neutral),"Is_Away":int((not is_home) and not neutral),"Is_Neutral":int(neutral),"Team_Score":ts,"Opponent_Score":os_,
              "SU_Result":_result_label(ts,os_,kind="su"),"Consensus_Open_Spread":open_sp,"Consensus_Close_Spread":close_sp,
              "Spread_Move_Open_to_Close":close_sp-open_sp if pd.notna(close_sp) and pd.notna(open_sp) else np.nan,
              "Consensus_Open_Total":top,"Consensus_Close_Total":tot,"Total_Move_Open_to_Close":tot-top if pd.notna(tot) and pd.notna(top) else np.nan,
              "Consensus_Open_Moneyline":ml,"Consensus_Close_Moneyline":ml,"ATS_Result_Close":_result_label(ts,os_,close_sp,kind="ats") if pd.notna(close_sp) else None,
              "Total_Result_Close":_result_label(ts,os_,total=tot,kind="total") if pd.notna(tot) else None,"Is_Favorite_Close":int(pd.notna(close_sp) and close_sp<0),"Is_Underdog_Close":int(pd.notna(close_sp) and close_sp>0),"Is_Pickem_Close":int(pd.notna(close_sp) and abs(close_sp)<1e-12),
              "Source_Opening_Odds_Raw":raw,"Source_Closing_Odds_Raw":raw,"Dataset":f"CFBD {season}",
            }
            rows.append(row)
    return pd.DataFrame(rows)


def merge_canonical_history(client, project: str, dataset: str, rows: pd.DataFrame) -> dict[str,Any]:
    """Reuse the fixed existing raw schema; BigDataBall wins on overlap."""
    if rows is None or rows.empty: return {"status":"EMPTY","rows":0}
    raw=_qualified(project,dataset,CANONICAL_RAW_TABLE); stage=_qualified(project,dataset,CFBD_HISTORY_STAGING_TABLE)
    table=client.get_table(raw); raw_cols=[f.name for f in table.schema]
    d=rows.reindex(columns=raw_cols).copy()
    # Load with the target schema exactly; adding CFBD cannot break weekly uploader's schema check.
    client.query(f"CREATE OR REPLACE TABLE `{stage}` AS SELECT * FROM `{raw}` WHERE FALSE").result()
    buf=io.StringIO(); d.to_csv(buf,index=False,na_rep=""); payload=io.BytesIO(buf.getvalue().encode("utf-8"))
    from google.cloud import bigquery
    cfg=bigquery.LoadJobConfig(schema=table.schema,source_format=bigquery.SourceFormat.CSV,skip_leading_rows=1,write_disposition="WRITE_TRUNCATE",allow_quoted_newlines=True)
    client.load_table_from_file(payload,stage,job_config=cfg).result()
    keys={"Season","Source_Name","Source_Game_ID","Team_Norm"}; upd=[c for c in raw_cols if c not in keys]
    us=",\n      ".join([f"`{c}`=S.`{c}`" for c in upd]); ic=", ".join(f"`{c}`" for c in raw_cols); iv=", ".join(f"S.`{c}`" for c in raw_cols)
    # Do not insert where a BigDataBall canonical version already exists.
    client.query(f"""
      MERGE `{raw}` T
      USING (
        SELECT S.* FROM `{stage}` S
        WHERE NOT EXISTS (
          SELECT 1 FROM `{raw}` B
          WHERE B.Source_Name='BigDataBall' AND B.Season=S.Season AND B.Game_Date=S.Game_Date
            AND LOWER(TRIM(B.Team_Norm))=LOWER(TRIM(S.Team_Norm))
            AND LOWER(TRIM(B.Opponent_Norm))=LOWER(TRIM(S.Opponent_Norm))
        )
      ) S
      ON T.Season=S.Season AND T.Source_Name=S.Source_Name AND T.Source_Game_ID=S.Source_Game_ID AND T.Team_Norm=S.Team_Norm
      WHEN MATCHED THEN UPDATE SET {us}
      WHEN NOT MATCHED THEN INSERT ({ic}) VALUES ({iv})
    """).result()
    # If BigDataBall was loaded after CFBD, remove the now-shadowed CFBD duplicate.
    client.query(f"""
      DELETE FROM `{raw}` C
      WHERE C.Source_Name='CollegeFootballData'
        AND EXISTS (
          SELECT 1 FROM `{raw}` B
          WHERE B.Source_Name='BigDataBall' AND B.Season=C.Season AND B.Game_Date=C.Game_Date
            AND LOWER(TRIM(B.Team_Norm))=LOWER(TRIM(C.Team_Norm))
            AND LOWER(TRIM(B.Opponent_Norm))=LOWER(TRIM(C.Opponent_Norm))
        )
    """).result()
    return {"status":"PASS","table":raw,"rows":len(d)}


def rebuild_context_if_available(client, sql_path: str | None) -> dict[str,Any]:
    if not sql_path:
        return {"status":"SKIPPED","reason":"NO_SQL_PATH"}
    from pathlib import Path
    p=Path(sql_path)
    if not p.exists():
        return {"status":"SKIPPED","reason":"SQL_NOT_FOUND","path":str(p)}
    sql=p.read_text(encoding="utf-8")
    client.query(sql).result()
    return {"status":"PASS","path":str(p)}


def warehouse_health(client, project: str, dataset: str) -> dict[str,Any]:
    out={}
    for label,table in {
        "raw":CANONICAL_RAW_TABLE,"travel":TRAVEL_TABLE,"archive":API_ARCHIVE_TABLE,
        "personnel":PERSONNEL_TABLE,"players":PLAYER_HISTORY_TABLE,"features":DEFAULT_TABLE,
    }.items():
        tid=_qualified(project,dataset,table)
        try:
            r=client.query(f"SELECT COUNT(*) n FROM `{tid}`").to_dataframe(create_bqstorage_client=False)
            out[label]=int(r.iloc[0]["n"]) if len(r) else 0
        except Exception as exc: out[label]=f"UNAVAILABLE:{type(exc).__name__}"
    return out


def warehouse_season_verification(client, project: str, dataset: str, table: str,
                                  expected: pd.DataFrame | None, feature_seasons: list[int],
                                  history_seasons: list[int]) -> dict[str,Any]:
    """Read back persisted seasons; never interpret an absent endpoint as success."""
    from google.cloud import bigquery
    tid=_qualified(project,dataset,table)
    report={"status":"PASS","feature_seasons":{},"history_seasons":{},"feature_table":tid}
    if feature_seasons:
        schema={f.name for f in client.get_table(tid).schema}
        probes=(
            "CFBD_Season_Net_PPA_Prior", "CFBD_Season_CFBD_Game_Off_EarlyDown_PPA_Prior",
            "CFBD_Season_CFBD_Game_Off_Field_Position_Prior", "CFBD_Season_CFBD_Game_Off_Sack_Rate_Prior",
            "CFBD_Returning_Pct_PPA", "CFBD_Roster_ALL_Retention", "CFBD_Prior_PPA_Return_Pct",
            "CFBD_QB_Game_Pass_Success_SeasonPrior", "CFBD_Portal_In_Count",
            "CFBD_Matchup_Elo_Adv", "CFBD_Matchup_PriorYear_SP_Adv")
        fields=[x for x in probes if x in schema]
        absent=[x for x in probes if x not in schema]
        if absent: print(f"[CFBD-WAREHOUSE-SCHEMA-GAP] missing_columns={absent} interpretation=VERIFY_ENDPOINT_AND_SEASON_AVAILABILITY")
        selects=["Season","COUNT(*) AS side_rows","COUNT(DISTINCT CAST(Game_ID AS STRING)) AS games"]
        selects += [f"COUNTIF(`{f}` IS NOT NULL) AS n_{i}" for i,f in enumerate(fields)]
        q=f"SELECT {', '.join(selects)} FROM `{tid}` WHERE Season IN UNNEST(@years) GROUP BY Season ORDER BY Season"
        cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("years","INT64",list(map(int,feature_seasons)))])
        got=client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)
        for yr in feature_seasons:
            d=got.loc[got["Season"].eq(yr)]
            observed=int(d.iloc[0]["side_rows"]) if len(d) else 0
            expected_rows=(int(pd.to_numeric(expected.get("Season",pd.Series(dtype=float)),errors="coerce").eq(yr).sum())
                           if expected is not None else None)
            coverage={f:{"n":int(d.iloc[0][f"n_{i}"]) if len(d) else 0,
                          "pct":round(float(d.iloc[0][f"n_{i}"])/observed,4) if observed else 0.,"status":"PRESENT"}
                      for i,f in enumerate(fields)}
            coverage.update({f:{"n":0,"pct":0.0,"status":"MISSING_COLUMN"} for f in absent})
            if expected_rows is None:
                status="PRESENT_UNVERIFIED_COUNT" if observed else "FAIL_MISSING_ROWS"
            else:
                status="PASS" if expected_rows>0 and observed>=expected_rows else "FAIL_MISSING_ROWS"
            if status=="FAIL_MISSING_ROWS": report["status"]="FAIL"
            elif observed and (absent or any(v['n']==0 for v in coverage.values())) and report["status"]=="PASS":
                report["status"]="PARTIAL"
            report["feature_seasons"][str(yr)]={"status":status,"expected_rows":expected_rows,
                "warehouse_rows":observed,"games":int(d.iloc[0]["games"]) if len(d) else 0,"field_coverage":coverage}
            print(f"[CFBD-WAREHOUSE-VERIFY-SEASON] year={yr} status={status} expected_rows={expected_rows} stored_rows={observed} fields={coverage}")
    if history_seasons:
        raw=_qualified(project,dataset,CANONICAL_RAW_TABLE)
        sql=f"""SELECT Season,COUNT(*) AS side_rows,
          COUNTIF(Consensus_Open_Spread IS NOT NULL) AS spread_rows,
          COUNTIF(Consensus_Open_Total IS NOT NULL) AS total_rows,
          COUNTIF(Source_Name='CollegeFootballData') AS cfbd_rows
          FROM `{raw}` WHERE Season IN UNNEST(@years) GROUP BY Season"""
        cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("years","INT64",list(map(int,history_seasons)))])
        got=client.query(sql,job_config=cfg).to_dataframe(create_bqstorage_client=False)
        for yr in history_seasons:
            d=got.loc[got["Season"].eq(yr)]
            if d.empty: stats={"status":"MISSING","side_rows":0,"spread_rows":0,"total_rows":0,"cfbd_rows":0}
            else:
                r=d.iloc[0]
                stats={"status":"PASS" if int(r["spread_rows"])>0 and int(r["total_rows"])>0 else "MARKET_TARGET_GAP",
                       **{k:int(r[k]) for k in ("side_rows","spread_rows","total_rows","cfbd_rows")}}
            report["history_seasons"][str(yr)]=stats
            if stats["status"]!="PASS" and report["status"]=="PASS": report["status"]="PARTIAL"
            print(f"[CFBD-HISTORY-MARKET-VERIFY] year={yr} {stats}")
    return report


# ---------------------------------------------------------------------------
# Operation mode: metadata preflight and warehouse-only repair
# ---------------------------------------------------------------------------

def warehouse_preflight(client, project: str, dataset: str, table: str,
                        require_features: bool = False, require_travel: bool = True,
                        require_archive_clear: bool = False) -> dict[str, Any]:
    """Validate destinations and key columns before API spending/writes.

    This does not consume the CFBD allowance. Missing feature staging is okay
    for a first backfill but forbidden for a warehouse-only repair.
    """
    from google.api_core.exceptions import NotFound
    specs={
        'features': (table, {'Season','Game_ID','Team_Key'},require_features),
        'travel': (TRAVEL_TABLE, {'Season','Game_ID'},require_travel),
        'raw': (CANONICAL_RAW_TABLE, {'Season','Source_Name'},False),
        'archive': (API_ARCHIVE_TABLE, set(_ARCHIVE_COLUMN_TYPES), False),
    }
    result={'status':'PASS','tables':{}}
    for name,(short,required,mandatory) in specs.items():
        tid=_qualified(project,dataset,short)
        try: obj=client.get_table(tid)
        except NotFound:
            state='MISSING_REQUIRED' if mandatory else 'NOT_CREATED'
        else:
            absent=sorted(required-{f.name for f in obj.schema})
            types={f.name:str(f.field_type).upper() for f in obj.schema}
            state='BAD_SCHEMA' if absent else 'PASS'
            if name=='archive' and not absent:
                aliases={"INTEGER":"INT64", "FLOAT":"FLOAT64", "BOOLEAN":"BOOL"}
                bad=[k for k,t in _ARCHIVE_COLUMN_TYPES.items() if aliases.get(types[k],types[k])!=t]
                if bad: state='BAD_ARCHIVE_TYPES'
        result['tables'][name]={'table':tid,'status':state}
        if state not in {'PASS','NOT_CREATED'} or (state=='NOT_CREATED' and mandatory):
            result['status']='FAIL'
        print(f"[CFBD-PREFLIGHT-TABLE] name={name} table={tid} status={state}")
    if require_archive_clear:
        # No repeat API expenditure if an earlier run left recoverable archive
        # snapshots behind. Repair them first, then proceed with new seasons.
        stages=_archive_leftover_stage_tables(client,project,dataset)
        result['pending_archive_stages']=len(stages)
        if stages:
            result['status']='FAIL'
            print(f"[CFBD-PREFLIGHT-ARCHIVE-PENDING] stages={len(stages)} action=RUN_--repair-archive no_api_calls=TRUE")
        # Never run another costly CFBD backfill while a prior player snapshot
        # can be resumed from BigQuery staging without contacting the API.
        player_prefix=f"{PLAYER_HISTORY_TABLE}__player_stage_"
        player_stages=[x.table_id for x in client.list_tables(f"{project}.{dataset}")
                       if x.table_id.startswith(player_prefix)]
        result['pending_player_stages']=len(player_stages)
        if player_stages:
            result['status']='FAIL'
            print(f"[CFBD-PREFLIGHT-PLAYER-PENDING] stages={len(player_stages)} action=RUN_--repair-player-stages no_api_calls=TRUE")
    print(f"[CFBD-PREFLIGHT] status={result['status']} cfbd_api_calls=0")
    return result


def repair_warehouse_from_stored_features(client,project: str,dataset: str,table: str,
                                          seasons: list[int],
                                          *,repair_personnel: bool = False) -> dict[str,Any]:
    """Restart post-load warehouse work without a token or another CFBD pull.

    Does not recreate raw API snapshots or player-level rows. Those are only
    correct when sourced from original endpoint responses; never synthesize them.
    """
    from google.cloud import bigquery
    if not seasons: raise RuntimeError('CFBD_REPAIR_REQUIRES_SEASONS')
    check=warehouse_preflight(client,project,dataset,table,require_features=True)
    if check['status']!='PASS': raise RuntimeError('CFBD_REPAIR_PREFLIGHT_FAILED')
    tid=_qualified(project,dataset,table)
    cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter('years','INT64',list(map(int,seasons)))])
    d=client.query(f"SELECT * FROM `{tid}` WHERE Season IN UNNEST(@years)",job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if d.empty: raise RuntimeError(f'CFBD_REPAIR_FEATURE_SEASONS_NOT_STORED seasons={seasons}')
    for yr in seasons:
        if not pd.to_numeric(d['Season'],errors='coerce').eq(yr).any():
            raise RuntimeError(f'CFBD_REPAIR_SEASON_MISSING season={yr}')
    keys=['Season','Game_ID','Team_Key']
    if not set(keys).issubset(d.columns) or d.duplicated(keys).any():
        raise RuntimeError('CFBD_REPAIR_DUPLICATE_OR_MISSING_FEATURE_KEYS')
    result={'status':'PASS','feature_rows':len(d),'seasons':seasons,'cfbd_api_calls':0}
    if repair_personnel:
        p=build_personnel_sidecar(d)
        result['personnel']=_merge_dataframe(client,_qualified(project,dataset,PERSONNEL_TABLE),p,['Season','Team_Key'])
        print(f"[CFBD-REPAIR-PERSONNEL] {result['personnel']}")
    travel=merge_environment_into_travel(client,project,dataset,d)
    print(f'[CFBD-REPAIR-TRAVEL] {travel}')
    if travel.get('status')!='PASS': raise RuntimeError(f'CFBD_REPAIR_TRAVEL_FAILED {travel}')
    result['travel']=travel
    v=warehouse_season_verification(client,project,dataset,table,d,seasons,[])
    result['feature_verification']=v['status']
    if v['status']=='FAIL': raise RuntimeError('CFBD_REPAIR_READBACK_FAILED')
    print(f"[CFBD-REPAIR-COMPLETE] status=PASS rows={len(d)} seasons={seasons} api_calls=0 archive=UNCHANGED player_sidecar=UNCHANGED feature_verify={v['status']}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    seasons: list[int], project: str, dataset: str, table: str,
    local_output: str | None = None, no_bq: bool = False,
    history_seasons: list[int] | None = None,
    context_rebuild_sql: str | None = None,
    skip_archive: bool = False,
    skip_sidecars: bool = False,
    skip_travel_update: bool = False,
) -> pd.DataFrame:
    history_seasons=sorted(set(history_seasons or []))
    seasons=sorted(set(seasons or []))
    if not seasons and not history_seasons:
        raise RuntimeError("No feature or history seasons selected")
    print(f"[CFBD-RUN-PLAN] features={seasons} history={history_seasons} project={project} dataset={dataset} mode=API_REFRESH")
    client = CFBDClient()
    bq=None
    if not no_bq:
        from google.cloud import bigquery
        bq=bigquery.Client(project=project)

    game_cache: dict[int,pd.DataFrame]={}
    all_features=[]
    personnel=pd.DataFrame(); players=[]

    # Rich research feature warehouse. Pull only if feature seasons were requested.
    if seasons:
        venues = pull_venues(client)
        recruit_years = range(min(seasons) - 3, max(seasons) + 1)
        recruiting_all = {y: pull_recruiting_team(client, y) for y in recruit_years}
        roster_cache = {y: pull_roster(client, y) for y in range(min(seasons) - 1, max(seasons) + 1)}
        # Include current requested seasons so player-season history can be refreshed,
        # while the portal feature builder still uses only prior-year production.
        player_ppa_cache = {y: pull_player_season_ppa(client, y) for y in range(min(seasons) - 1, max(seasons) + 1)}
        player_usage_cache = {y: pull_player_usage(client, y) for y in range(min(seasons) - 1, max(seasons) + 1)}
        # Season-level ratings are exposed only one season later, so historical
        # qualification never sees same-season final/retrospective estimates.
        rating_years=range(min(seasons)-1,max(seasons))
        sp_cache={y:pull_sp_ratings(client,y) for y in rating_years}
        srs_cache={y:pull_srs_ratings(client,y) for y in rating_years}
        fpi_cache={y:pull_fpi_ratings(client,y) for y in rating_years}
        core_cache={y:pull_core_ratings(client,y) for y in rating_years}
        enable_play_context=str(os.getenv("CFBD_ENABLE_PLAY_CONTEXT","1")).strip().lower() not in {"0","false","no","off"}

        for year in seasons:
            print(f"[CFBD-SEASON] year={year} status=START calls={client.call_count}")
            games = pull_games(client, year); game_cache[year]=games
            weeks = sorted(_num(_first_col(games, "week")).dropna().astype(int).unique().tolist()) if not games.empty else []
            advanced = pull_advanced(client, year)
            havoc = pull_havoc(client, year)
            returning = pull_returning(client, year)
            portal = pull_portal(client, year)
            talent = pull_talent(client, year)
            weather = pull_weather(client, year)
            teams = pull_fbs_teams(client, year)
            qb = pull_qb_ppa_games(client, year, weeks)
            qb_success = pull_qb_success_games(client, year, weeks)
            plays = pull_plays(client, year, weeks) if enable_play_context else pd.DataFrame()
            feat = build_season_features(
                year, games, advanced, havoc, returning, portal,
                roster_cache.get(year, pd.DataFrame()), roster_cache.get(year - 1, pd.DataFrame()),
                talent, recruiting_all, weather, venues, teams, qb, qb_success,
                player_ppa_cache.get(year - 1, pd.DataFrame()), player_usage_cache.get(year - 1, pd.DataFrame()),
                plays, sp_cache.get(year-1,pd.DataFrame()), srs_cache.get(year-1,pd.DataFrame()),
                fpi_cache.get(year-1,pd.DataFrame()), core_cache.get(year-1,pd.DataFrame()),
            )
            all_features.append(feat)
            # Never replace stored player seasons from a partial optional-source response.
            player_sources=(roster_cache.get(year,pd.DataFrame()),player_ppa_cache.get(year,pd.DataFrame()),player_usage_cache.get(year,pd.DataFrame()))
            if all(frame is not None and not frame.empty for frame in player_sources):
                players.append(build_player_history_sidecar(year,*player_sources))
            else:
                print(f"[CFBD-PLAYER-SNAPSHOT-SKIPPED] year={year} reason=INCOMPLETE_ROSTER_PPA_OR_USAGE existing_data_preserved=TRUE")
            if feat.empty:
                raise RuntimeError(f"CFBD_REQUIRED_SEASON_EMPTY year={year}; refusing a destructive warehouse refresh")
            print(f"[CFBD-SEASON] year={year} status=PASS feature_rows={len(feat)} calls={client.call_count}")

    out = pd.concat(all_features, ignore_index=True, sort=False) if all_features else pd.DataFrame()
    if seasons and out.empty:
        raise RuntimeError("CFBD loader produced zero feature rows")
    if not out.empty:
        personnel=build_personnel_sidecar(out)
        print(f"[CFBD-QUALITY] rows={len(out)} games={out['Game_ID'].nunique()} seasons={sorted(out['Season'].dropna().astype(int).unique().tolist())} calls={client.call_count}")
        for c in [
            "CFBD_Season_Net_PPA_Prior", "CFBD_Returning_Pct_PPA", "CFBD_Roster_ALL_Retention",
            "CFBD_Talent", "CFBD_Portal_Net_Prior_TotalPPA", "CFBD_QB_Game_Pass_Success_SeasonPrior",
            "CFBD_Matchup_Elo_Adv", "CFBD_Matchup_EarlyDown_PPA_Adv", "CFBD_Matchup_PPA_Per_Drive_Adv",
            "CFBD_Prior_PPA_Return_Pct", "CFBD_Matchup_PriorYear_SP_Adv", "CFBD_Matchup_PriorYear_FPI_Adv",
            "CFBD_WindSpeed", "CFBD_Venue_Elevation", "CFBD_Altitude_Gain",
        ]:
            if c in out.columns:
                print(f"[CFBD-COVERAGE] field={c} n={int(out[c].notna().sum())} pct={float(out[c].notna().mean()):.4f}")
        if local_output:
            write_local(out, local_output)

    # Long-history game/results/market backfill into the EXISTING fixed raw table.
    history_results=[]
    if history_seasons:
        if bq is None:
            print("[CFBD-HISTORY] status=SKIPPED reason=NO_BQ")
        else:
            for year in history_seasons:
                print(f"[CFBD-HISTORY] year={year} status=START calls={client.call_count}")
                games=game_cache.get(year)
                if games is None:
                    games=pull_games(client,year); game_cache[year]=games
                lines=pull_lines(client,year)
                amap=_alignment_map(bq,project,dataset,year)
                if not amap:
                    print(f"[CFBD-HISTORY] year={year} status=SKIP_NO_ALIGNMENT")
                    continue
                rows=build_canonical_history_rows(games,lines,year,amap)
                if not rows.empty:
                    n_open=int(rows["Consensus_Open_Spread"].notna().sum())
                    n_total=int(rows["Consensus_Open_Total"].notna().sum())
                    print(f"[CFBD-HISTORY-OPENING-AVAILABILITY] year={year} mapped_rows={len(rows)} spread_open_rows={n_open} total_open_rows={n_total} status={'MARKET_TARGET_GAP' if not n_open or not n_total else 'PASS'} close_lines_not_substituted=TRUE")
                if rows.empty:
                    print(f"[CFBD-HISTORY] year={year} status=NO_CANONICAL_ROWS")
                    continue
                res=merge_canonical_history(bq,project,dataset,rows); history_results.append({"season":year,**res})
                print(f"[CFBD-HISTORY] year={year} status={res.get('status')} rows={len(rows)} calls={client.call_count}")

    if bq is not None:
        if not out.empty:
            # Build/staging table for Research V2's fail-closed CFBD bridge.
            write_bigquery(out, project, dataset, table, seasons)
        # Archive source payloads BEFORE optional sidecars. If a sidecar DML
        # fails, archive recovery can replay the original CFBD responses.
        if not skip_archive:
            ares=write_api_archive(bq,project,dataset,client.archive_rows); print(f"[CFBD-ARCHIVE] {ares}")
        if not skip_sidecars:
            if not personnel.empty:
                pres=_merge_dataframe(bq,_qualified(project,dataset,PERSONNEL_TABLE),personnel,["Season","Team_Key"])
                print(f"[CFBD-PERSONNEL] {pres}")
            player_df=pd.concat([x for x in players if x is not None and not x.empty],ignore_index=True,sort=False) if any(x is not None and not x.empty for x in players) else pd.DataFrame()
            if not player_df.empty:
                plres=replace_player_history_seasons(bq,_qualified(project,dataset,PLAYER_HISTORY_TABLE),player_df)
                print(f"[CFBD-PLAYERS] {plres}")
        if not skip_travel_update and not out.empty:
            tres=merge_environment_into_travel(bq,project,dataset,out); print(f"[CFBD-TRAVEL-ENV] {tres}")
            if tres.get("status") != "PASS":
                raise RuntimeError(f"CFBD_TRAVEL_STEP_FAILED {tres}")
        if history_results:
            cres=rebuild_context_if_available(bq,context_rebuild_sql); print(f"[CFBD-CONTEXT-REBUILD] {cres}")
        print(f"[CFBD-WAREHOUSE-HEALTH] {warehouse_health(bq,project,dataset)}")
    else:
        print(f"[CFBD-NO-BQ] archive_rows_buffered={len(client.archive_rows)} calls={client.call_count}")
    for (_,yr),stat in sorted(client.request_health.items(),key=lambda x:(str(x[0][1]),x[0][0])):
        print(f"[CFBD-SOURCE-COVERAGE] year={yr} path={stat['path']} status={'PARTIAL' if stat['failures'] else 'PASS'} requests={stat['requests']} rows={stat['rows']} failures={stat['failures']} status_codes={stat['statuses']}")
    print(f"[CFBD-API-SUMMARY] calls={client.call_count} failed_endpoint_batches={len(client.api_failures)} status={'PARTIAL' if client.api_failures else 'PASS'} tier2_access_verification={'INCOMPLETE' if client.api_failures else 'PASS'}")
    if bq is not None:
        verification=warehouse_season_verification(bq,project,dataset,table,out,seasons,history_seasons)
        print(f"[CFBD-WAREHOUSE-VERIFY] status={verification['status']} feature_years={len(verification['feature_seasons'])} history_years={len(verification['history_seasons'])}")
        if verification['status']=="FAIL":
            raise RuntimeError("CFBD_WAREHOUSE_VERIFICATION_FAILED: see CFBD-WAREHOUSE-VERIFY-SEASON")
    return out


def _self_test() -> None:
    # Strict prior-only test plus multi-future-row recent-window behavior.
    adv = pd.DataFrame([
        {"gameId": 1, "season": 2022, "week": 1, "team": "A", "opponent": "B", "offense.ppa": 0.1, "defense.ppa": -0.1},
        {"gameId": 2, "season": 2022, "week": 2, "team": "A", "opponent": "C", "offense.ppa": 0.2, "defense.ppa": 0.0},
        {"gameId": 3, "season": 2022, "week": 3, "team": "A", "opponent": "D", "offense.ppa": 0.3, "defense.ppa": 0.1},
        {"gameId": 4, "season": 2022, "week": 4, "team": "A", "opponent": "E", "offense.ppa": 0.4, "defense.ppa": 0.1},
    ])
    games = pd.DataFrame([
        {"id": 1, "season": 2022, "week": 1, "startDate": "2022-09-01T00:00:00Z", "homeTeam": "A", "awayTeam": "B", "completed": True, "homePregameElo":1500, "awayPregameElo":1450},
        {"id": 2, "season": 2022, "week": 2, "startDate": "2022-09-08T00:00:00Z", "homeTeam": "A", "awayTeam": "C", "completed": True},
        {"id": 3, "season": 2022, "week": 3, "startDate": "2022-09-15T00:00:00Z", "homeTeam": "A", "awayTeam": "D", "completed": True},
        {"id": 4, "season": 2022, "week": 4, "startDate": "2022-09-22T00:00:00Z", "homeTeam": "A", "awayTeam": "E", "completed": True},
        {"id": 5, "season": 2022, "week": 5, "startDate": "2022-09-29T00:00:00Z", "homeTeam": "A", "awayTeam": "F", "completed": False},
        {"id": 6, "season": 2022, "week": 6, "startDate": "2022-10-06T00:00:00Z", "homeTeam": "A", "awayTeam": "G", "completed": False},
    ])
    x = attach_game_metadata(standardize_advanced(adv), standardize_games(games, 2022), pd.DataFrame())
    x = add_prior_advanced_features(x)
    a = x.loc[x["Team_Key"].eq(_team_key("A"))].sort_values("Week")
    assert len(a) == 6
    assert pd.isna(a.iloc[0]["CFBD_Season_Off_PPA_Prior"])
    assert abs(float(a.iloc[1]["CFBD_Season_Off_PPA_Prior"]) - 0.1) < 1e-12
    # Weeks 5 and 6 are both unplayed. Both must see four completed games, and
    # their Recent-3 must remain the same mean of games 2/3/4 = 0.3.
    for pos in (4,5):
        assert float(a.iloc[pos]["CFBD_Games_Prior"]) == 4.0
        assert abs(float(a.iloc[pos]["CFBD_Recent3_Off_PPA_Prior"]) - 0.3) < 1e-12
    assert set(x["Team_Key"]) >= {_team_key(z) for z in "ABCDEFG"}
    assert float(a.iloc[0]["CFBD_Elo_Prior"]) == 1500.0

    # Existing-raw-table mapping: home/away market orientation and grading.
    hg=pd.DataFrame([{"id":99,"season":2020,"week":1,"startDate":"2020-09-01T00:00:00Z","homeTeam":"Alpha","awayTeam":"Beta","homePoints":28,"awayPoints":21,"homeConference":"X","awayConference":"Y","neutralSite":False}])
    hl=pd.DataFrame([{"id":99,"lines":[{"provider":"Consensus","spread":-3.5,"spreadOpen":-3.0,"overUnder":49.5,"overUnderOpen":48.5,"homeMoneyline":-160,"awayMoneyline":140}]}])
    amap={"alpha":{"Team":"Alpha","Team_Norm":"alpha","Canonical_Team_Name":"Alpha","Canonical_Team_Norm":"alpha","Conference":"X","Subdivision":"FBS"},"beta":{"Team":"Beta","Team_Norm":"beta","Canonical_Team_Name":"Beta","Canonical_Team_Norm":"beta","Conference":"Y","Subdivision":"FBS"}}
    hr=build_canonical_history_rows(hg,hl,2020,amap)
    assert len(hr)==2 and float(hr.loc[hr["Is_Home"].eq(1),"Consensus_Close_Spread"].iloc[0])==-3.5 and float(hr.loc[hr["Is_Home"].eq(0),"Consensus_Close_Spread"].iloc[0])==3.5
    assert hr.loc[hr["Is_Home"].eq(1),"ATS_Result_Close"].iloc[0]=="WIN" and hr.loc[hr["Is_Home"].eq(0),"ATS_Result_Close"].iloc[0]=="LOSS"

    pf=pd.DataFrame([{"Season":2025,"Team":"A","Team_Key":"a","CFBD_Returning_Pct_PPA":.70,"CFBD_Talent":800.0,"CFBD_Season_Net_PPA_Prior":.10}])
    ps=build_personnel_sidecar(pf)
    assert "CFBD_Returning_Pct_PPA" in ps.columns and "CFBD_Season_Net_PPA_Prior" not in ps.columns
    pbp=pd.DataFrame([
        {"gameId":99,"driveId":"d1","offense":"A","defense":"B","down":1,"distance":10,"yardsToGoal":75,"yardsGained":6,"playType":"Rush","ppa":.10},
        {"gameId":99,"driveId":"d1","offense":"A","defense":"B","down":2,"distance":4,"yardsToGoal":69,"yardsGained":20,"playType":"Pass Reception","ppa":.35},
        {"gameId":99,"driveId":"d2","offense":"B","defense":"A","down":1,"distance":10,"yardsToGoal":80,"yardsGained":-5,"playType":"Sack","ppa":-.40},
    ])
    pc=build_play_context_features(pbp)
    assert len(pc)==2 and "CFBD_Game_Off_Field_Position" in pc.columns and "CFBD_Game_Off_Sack_Rate" in pc.columns
    print("[CFBD-SELF-TEST] status=PASS leakage_shift1=TRUE completed_game_windows=TRUE schedule_base=TRUE pregame_elo=TRUE raw_history_orientation=TRUE personnel_sidecar=TRUE pbp_context=TRUE prior_ratings_contract=TRUE")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2022-2026", help="Rich CFBD feature seasons, e.g. 2022-2026; empty string disables")
    p.add_argument("--history-seasons", default="2013-2021", help="CFBD game/results/line gap-fill into existing raw table; BigDataBall wins overlaps")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--table", default=DEFAULT_TABLE, help="Build/staging CFBD feature table")
    p.add_argument("--local-output", default=None)
    p.add_argument("--archive-preview", action="store_true", help="Build offline research-only CSV from archive_v2 objects; NEVER invokes CFBD API or BigQuery")
    p.add_argument("--archive-root", default=None, help="Directory containing archive_v2 manifest.jsonl and objects/")
    p.add_argument("--context-rebuild-sql", default=os.getenv("NCAAF_CONTEXT_REBUILD_SQL","ncaaf_context_rebuild.sql"))
    p.add_argument("--skip-archive", action="store_true")
    p.add_argument("--skip-sidecars", action="store_true")
    p.add_argument("--skip-travel-update", action="store_true")
    p.add_argument("--no-bq", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--verify-only", action="store_true",help="BQ warehouse and historical opening-market audit only; no CFBD API calls/writes/token needed")
    p.add_argument("--preflight", action="store_true",help="Check destination schemas and credentials before a live run; no API calls or writes")
    p.add_argument("--repair-warehouse", action="store_true",help="Resume the travel/derived warehouse step using persisted features; zero CFBD API calls")
    p.add_argument("--repair-personnel", action="store_true",help="With --repair-warehouse, also re-merge personnel sidecar")
    p.add_argument("--repair-archive", action="store_true",help="Recover failed prior API archive from leftover BigQuery stage; no CFBD key/calls")
    p.add_argument("--audit-players", action="store_true",help="Read-only player-history coverage/freshness audit; zero CFBD calls")
    p.add_argument("--repair-players", action="store_true",help="Refresh ONLY player roster/PPA/usage endpoints and archive first; avoid full CFBD reload")
    p.add_argument("--repair-player-stages", action="store_true",help="Resume retained failed player snapshots from BigQuery stage; no CFBD calls")
    p.add_argument("--fail-on-partial", action="store_true",help="Make read-only verification return nonzero for PARTIAL coverage")
    args = p.parse_args()
    if args.self_test:
        _self_test(); return
    seasons = _parse_seasons(args.seasons)
    history = _parse_seasons(args.history_seasons)
    if args.archive_preview:
        if not args.archive_root or not args.local_output:
            p.error("--archive-preview requires --archive-root and --local-output")
        if any((args.verify_only,args.preflight,args.repair_warehouse,args.repair_archive,
                args.repair_personnel,args.repair_players,args.repair_player_stages,
                args.audit_players)):
            p.error("--archive-preview cannot be combined with BigQuery/repair modes")
        # Nothing below this line runs for archive previews: NO BQ/CFBD API writes.
        run_archive_research_preview(args.archive_root,seasons,args.local_output)
        return
    if not seasons and not history:
        raise SystemExit("No seasons selected")
    if args.verify_only or args.preflight or args.repair_warehouse or args.repair_archive or args.audit_players or args.repair_players or args.repair_player_stages:
        if args.no_bq: raise SystemExit("This operation requires BigQuery (remove --no-bq)")
        from google.cloud import bigquery
        bq=bigquery.Client(project=args.project)
        if args.audit_players:
            audit_player_history(bq,args.project,args.dataset,seasons)
            return
        if args.repair_player_stages:
            res=repair_player_history_from_stages(bq,args.project,args.dataset)
            if res['status']!='PASS': raise SystemExit('CFBD_PLAYER_NO_RECOVERABLE_STAGE')
            return
        if args.repair_players:
            ck=warehouse_preflight(bq,args.project,args.dataset,args.table,require_archive_clear=True)
            if ck['status']!='PASS': raise SystemExit('CFBD_PLAYER_REPAIR_PREFLIGHT_FAILED')
            audit_player_history(bq,args.project,args.dataset,seasons)
            repair_player_history_with_minimal_cfbd(bq,args.project,args.dataset,seasons)
            audit_player_history(bq,args.project,args.dataset,seasons)
            # Main 4-season run stopped before travel update. Resume travel
            # and feature readback from stored warehouse, with zero more calls.
            repair_warehouse_from_stored_features(bq,args.project,args.dataset,args.table,seasons)
            print(f"[CFBD-PLAYER-AND-TRAVEL-COMPLETE] status=PASS seasons={seasons}")
            return
        if args.repair_archive:
            recovery=repair_api_archive_from_stages(bq,args.project,args.dataset)
            if recovery['status']!='PASS':
                print(f"[CFBD-ARCHIVE-RECOVERY] status={recovery['status']} cfbd_api_calls=0")
                raise SystemExit(2)
            # Run warehouse verification immediately as part of recovery; no
            # separate command or CFBD key is needed. Historical odds gaps
            # report PARTIAL rather than being silently synthesized.
            audit=warehouse_season_verification(bq,args.project,args.dataset,args.table,None,seasons,history)
            print(f"[CFBD-RECOVERY-FINAL] archive=PASS feature_audit={audit['status']} cfbd_api_calls=0")
            if audit['status']=='FAIL' or (args.fail_on_partial and audit['status']!='PASS'):
                raise SystemExit(2)
            return
        if args.preflight:
            ck=warehouse_preflight(bq,args.project,args.dataset,args.table,require_features=False,
                                  require_travel=not args.skip_travel_update,
                                  require_archive_clear=True)
            print(f"[CFBD-PREFLIGHT-KEY] configured={bool(os.getenv('CFBD_API_KEY'))} (not printed)")
            if ck['status']!='PASS': raise SystemExit(2)
            return
        if args.repair_warehouse:
            repair_warehouse_from_stored_features(bq,args.project,args.dataset,args.table,seasons,
                                                  repair_personnel=args.repair_personnel)
            return
        print(f"[CFBD-WAREHOUSE-HEALTH] {warehouse_health(bq,args.project,args.dataset)}")
        result=warehouse_season_verification(bq,args.project,args.dataset,args.table,None,seasons,history)
        print(f"[CFBD-WAREHOUSE-VERIFY] status={result['status']} mode=READ_ONLY tier2_api_calls=0")
        if result['status']=="FAIL" or (args.fail_on_partial and result['status']!='PASS'): raise SystemExit(2)
        return
    if not args.no_bq:
        from google.cloud import bigquery
        ck=warehouse_preflight(bigquery.Client(project=args.project),args.project,args.dataset,args.table,
                               require_features=False,require_travel=not args.skip_travel_update,
                               require_archive_clear=True)
        if ck['status']!='PASS': raise SystemExit('CFBD_PREFLIGHT_FAILED: no CFBD requests were made')
    run(
        seasons,args.project,args.dataset,args.table,args.local_output,args.no_bq,
        history_seasons=history,context_rebuild_sql=args.context_rebuild_sql,
        skip_archive=args.skip_archive,skip_sidecars=args.skip_sidecars,skip_travel_update=args.skip_travel_update,
    )


if __name__ == "__main__":
    main()
