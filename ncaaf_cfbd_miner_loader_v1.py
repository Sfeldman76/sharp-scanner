#!/usr/bin/env python3
"""NCAAF CFBD Miner Feature Loader V1.

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
* Retrospective full-season SP+, FPI, SRS, CORE, WEPA are deliberately NOT used
  for historical Miner qualification because point-in-time availability is not
  guaranteed by those season-level endpoints.

Default BigQuery output
-----------------------
  sharplogger.sharp_data.ncaaf_cfbd_miner_features_v1

Typical first backfill
----------------------
  export CFBD_API_KEY='...'
  python ncaaf_cfbd_miner_loader_v1.py --seasons 2022-2026

Weekly refresh
--------------
  python ncaaf_cfbd_miner_loader_v1.py --seasons 2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.collegefootballdata.com"
DEFAULT_PROJECT = os.getenv("BQ_PROJECT", "sharplogger")
DEFAULT_DATASET = os.getenv("BQ_DATASET", "sharp_data")
DEFAULT_TABLE = os.getenv("CFBD_MINER_TABLE", "ncaaf_cfbd_miner_features_v1")
SOURCE_TAG = "ncaaf-cfbd-miner-features-v1.1-20261015"


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

    def get(self, path: str, params: dict[str, Any] | None = None, *, optional: bool = False) -> list[dict[str, Any]]:
        url = BASE_URL + path
        params = {k: v for k, v in (params or {}).items() if v is not None}
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                self.call_count += 1
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    time.sleep(min(30, 2 ** attempt))
                    continue
                if 500 <= r.status_code < 600:
                    time.sleep(min(15, 1.5 ** attempt))
                    continue
                r.raise_for_status()
                payload = r.json()
                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                    return payload["data"]
                if isinstance(payload, dict):
                    return [payload]
                return []
            except Exception as exc:
                last = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(min(15, 1.5 ** attempt))
        if optional:
            print(f"[CFBD-OPTIONAL] path={path} params={params} status=UNAVAILABLE error={type(last).__name__}:{last}")
            return []
        raise RuntimeError(f"CFBD request failed path={path} params={params}: {type(last).__name__}:{last}")


# ---------------------------------------------------------------------------
# Pulls
# ---------------------------------------------------------------------------

def pull_games(c: CFBDClient, year: int) -> pd.DataFrame:
    return _json_frame(c.get("/games", {"year": year, "seasonType": "both", "classification": "fbs"}))


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


def attach_game_metadata(adv: pd.DataFrame, games: pd.DataFrame, havoc: pd.DataFrame) -> pd.DataFrame:
    """Build one row per scheduled team-side, then attach completed-game stats.

    The schedule is the base so upcoming games exist before kickoff. Completed
    advanced/havoc values are attached only to their own game and are subsequently
    shifted by add_prior_advanced_features before becoming model/Miner inputs.
    """
    if games.empty:
        return pd.DataFrame()
    base_cols=[c for c in ["Game_ID","Season","Week","Season_Type","Game_Date","Completed","Venue_ID","Neutral_Site","Home_Team","Away_Team","Home_Key","Away_Key","Home_Pregame_Elo","Away_Pregame_Elo"] if c in games.columns]
    gg=games[base_cols].drop_duplicates("Game_ID").copy()
    home=gg.copy(); away=gg.copy()
    home["Team"]=home["Home_Team"]; home["Opponent"]=home["Away_Team"]
    home["Team_Key"]=home["Home_Key"]; home["Opponent_Key"]=home["Away_Key"]; home["Is_Home"]=1.0
    home["CFBD_Elo_Prior"]=_num(home.get("Home_Pregame_Elo",pd.Series(np.nan,index=home.index)))
    away["Team"]=away["Away_Team"]; away["Opponent"]=away["Home_Team"]
    away["Team_Key"]=away["Away_Key"]; away["Opponent_Key"]=away["Home_Key"]; away["Is_Home"]=0.0
    away["CFBD_Elo_Prior"]=_num(away.get("Away_Pregame_Elo",pd.Series(np.nan,index=away.index)))
    x=pd.concat([home,away],ignore_index=True,sort=False)
    x=x.drop(columns=["Home_Team","Away_Team","Home_Key","Away_Key","Home_Pregame_Elo","Away_Pregame_Elo"],errors="ignore")

    if adv is not None and not adv.empty:
        metric_cols=[c for c in adv.columns if c.startswith(("Off_","Def_"))]
        aa=adv[["Game_ID","Team_Key"]+metric_cols].drop_duplicates(["Game_ID","Team_Key"],keep="last")
        x=x.merge(aa,on=["Game_ID","Team_Key"],how="left")
    if havoc is not None and not havoc.empty:
        hcols=[c for c in havoc.columns if c.startswith("CFBD_Game_")]
        hh=havoc[["Game_ID","Team_Key"]+hcols].drop_duplicates(["Game_ID","Team_Key"],keep="last")
        x=x.merge(hh,on=["Game_ID","Team_Key"],how="left")
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
    feature_cols = [c for c in out.columns if c.startswith("CFBD_") and not c.startswith("CFBD_Opp_")]
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
) -> pd.DataFrame:
    games = standardize_games(games_raw, year)
    adv = standardize_advanced(advanced_raw)
    havoc = standardize_havoc(havoc_raw)
    x = attach_game_metadata(adv, games, havoc)
    x = add_prior_advanced_features(x)
    if x.empty:
        return x

    x = merge_static(x, [
        build_returning_features(returning_raw, year),
        build_roster_continuity(roster_cur, roster_prev, year),
        build_portal_features(portal_raw, year, prior_player_ppa_raw, prior_player_usage_raw, games["Game_Date"].min() if "Game_Date" in games.columns else None),
        build_talent_features(talent_raw, year),
        build_recruiting_features(recruiting_all, year),
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
    x["CFBD_Leakage_Contract"] = "GAME_STATS_SHIFT1__PORTAL_PRESEASON_CUTOFF__PRESEASON_STATIC_DIRECT__GAME_PREGAME_ELO__2026_NOT_USED_FOR_HISTORICAL_SELECTION"
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

    keep_identity = ["Game_ID", "Season", "Week", "Season_Type", "Game_Date", "Team", "Opponent", "Team_Key", "Opponent_Key", "Venue_ID", "Neutral_Site"]
    keep = keep_identity + [c for c in x.columns if c.startswith("CFBD_")]
    return x[keep].drop_duplicates(["Game_ID", "Team_Key"], keep="last")


# ---------------------------------------------------------------------------
# BigQuery I/O
# ---------------------------------------------------------------------------

def write_bigquery(df: pd.DataFrame, project: str, dataset: str, table: str, seasons: list[int]) -> None:
    from google.cloud import bigquery
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"
    final = df.copy()
    # Preserve seasons not included in this refresh.
    try:
        existing = client.query(f"SELECT * FROM `{table_id}` WHERE Season NOT IN UNNEST(@seasons)", job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("seasons", "INT64", [int(s) for s in seasons])]
        )).to_dataframe(create_bqstorage_client=False)
        if existing is not None and not existing.empty:
            final = pd.concat([existing, final], ignore_index=True, sort=False)
    except Exception:
        pass
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
# Main
# ---------------------------------------------------------------------------

def run(seasons: list[int], project: str, dataset: str, table: str, local_output: str | None = None, no_bq: bool = False) -> pd.DataFrame:
    client = CFBDClient()
    venues = pull_venues(client)

    # Cache recruiting classes needed for rolling four-year context.
    recruit_years = range(min(seasons) - 3, max(seasons) + 1)
    recruiting_all = {y: pull_recruiting_team(client, y) for y in recruit_years}
    roster_cache = {y: pull_roster(client, y) for y in range(min(seasons) - 1, max(seasons) + 1)}
    prior_player_ppa_cache = {y: pull_player_season_ppa(client, y) for y in range(min(seasons) - 1, max(seasons))}
    prior_player_usage_cache = {y: pull_player_usage(client, y) for y in range(min(seasons) - 1, max(seasons))}

    all_features = []
    for year in seasons:
        print(f"[CFBD-SEASON] year={year} status=START calls={client.call_count}")
        games = pull_games(client, year)
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
        feat = build_season_features(
            year, games, advanced, havoc, returning, portal,
            roster_cache.get(year, pd.DataFrame()), roster_cache.get(year - 1, pd.DataFrame()),
            talent, recruiting_all, weather, venues, teams, qb, qb_success,
            prior_player_ppa_cache.get(year - 1, pd.DataFrame()), prior_player_usage_cache.get(year - 1, pd.DataFrame()),
        )
        all_features.append(feat)
        print(f"[CFBD-SEASON] year={year} status=PASS feature_rows={len(feat)} calls={client.call_count}")

    out = pd.concat(all_features, ignore_index=True, sort=False) if all_features else pd.DataFrame()
    if out.empty:
        raise RuntimeError("CFBD loader produced zero feature rows")

    # Quality / leakage diagnostics.
    print(f"[CFBD-QUALITY] rows={len(out)} games={out['Game_ID'].nunique()} seasons={sorted(out['Season'].dropna().astype(int).unique().tolist())} calls={client.call_count}")
    for c in [
        "CFBD_Season_Net_PPA_Prior", "CFBD_Returning_Pct_PPA", "CFBD_Roster_ALL_Retention",
        "CFBD_Talent", "CFBD_Portal_Net_Prior_TotalPPA", "CFBD_QB_Game_Pass_Success_SeasonPrior",
        "CFBD_Matchup_Elo_Adv", "CFBD_WindSpeed", "CFBD_Venue_Elevation", "CFBD_Altitude_Gain",
    ]:
        if c in out.columns:
            print(f"[CFBD-COVERAGE] field={c} n={int(out[c].notna().sum())} pct={float(out[c].notna().mean()):.4f}")

    if local_output:
        write_local(out, local_output)
    if not no_bq:
        write_bigquery(out, project, dataset, table, seasons)
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
    print("[CFBD-SELF-TEST] status=PASS leakage_shift1=TRUE completed_game_windows=TRUE schedule_base=TRUE pregame_elo=TRUE")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2022-2026", help="e.g. 2022-2026 or 2024,2025,2026")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--table", default=DEFAULT_TABLE)
    p.add_argument("--local-output", default=None)
    p.add_argument("--no-bq", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        _self_test(); return
    seasons = _parse_seasons(args.seasons)
    if not seasons:
        raise SystemExit("No seasons selected")
    run(seasons, args.project, args.dataset, args.table, args.local_output, args.no_bq)


if __name__ == "__main__":
    main()
