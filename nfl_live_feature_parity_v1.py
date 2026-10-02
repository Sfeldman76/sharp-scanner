"""NFL Production V1 live-pregame feature parity audit.

Purpose
-------
Prove that the compact, frozen NFL Production V1 feature contract can be
constructed for upcoming games using only information available before kickoff.
The audit has zero production authority and writes no predictions.

The historical replay deliberately recomputes the compact prior-state features
from the authoritative raw team-side history instead of reading them from the
training view.  The recomputed values are then compared field-for-field to the
historical training view.  Upcoming Week_Number uses the uploader-proven Tuesday-to-Monday regular-season calendar on standard live schedule days. Tuesday/Wednesday games are exception-gated and require an authoritative exact-date Week label before live use.
Postseason week numbering is deliberately outside this live contract and remains
fail-closed until a separate stage-aware adapter is proven. Division context comes
from fixed NFL alignment and is replay-validated. Unproven live fields remain
research-only.
"""
from __future__ import annotations

import json
import math
import re
from collections import OrderedDict
from datetime import timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from google.cloud import bigquery as b

from nfl_challenger_v1 import COMPACT_FEATURES

SOURCE_TAG = "nfl-production-v1-live-feature-parity-v1.0.4-standard-week-exception-gate-20261001"
PROJECT = "sharplogger"
DATASET = "sharp_data"
RAW = f"{PROJECT}.{DATASET}.nfl_historical_game_side_raw"
VIEW = f"{PROJECT}.{DATASET}.nfl_historical_core_training_vw"
MARKET_SOURCE = f"{PROJECT}.{DATASET}.sharp_moves_master"
PRODUCTION_AUTHORITY = 0
HISTORICAL_REPLAY_SEASON = 2025
FLOAT_TOL = 1e-6

# Live-production context is deliberately limited to fields we can reproduce
# exactly from the authoritative uploader history plus current matchup identity.
# Is_Neutral and Is_Night_Game remain available to research, but are not part
# of Production V1 until a live source proves exact historical/live parity.
_TEAM_ALIASES = {
    "ari":"ARI","arizona":"ARI","arizona cardinals":"ARI","cardinals":"ARI",
    "atl":"ATL","atlanta":"ATL","atlanta falcons":"ATL","falcons":"ATL",
    "bal":"BAL","baltimore":"BAL","baltimore ravens":"BAL","ravens":"BAL",
    "buf":"BUF","buffalo":"BUF","buffalo bills":"BUF","bills":"BUF",
    "car":"CAR","carolina":"CAR","carolina panthers":"CAR","panthers":"CAR",
    "chi":"CHI","chicago":"CHI","chicago bears":"CHI","bears":"CHI",
    "cin":"CIN","cincinnati":"CIN","cincinnati bengals":"CIN","bengals":"CIN",
    "cle":"CLE","cleveland":"CLE","cleveland browns":"CLE","browns":"CLE",
    "dal":"DAL","dallas":"DAL","dallas cowboys":"DAL","cowboys":"DAL",
    "den":"DEN","denver":"DEN","denver broncos":"DEN","broncos":"DEN",
    "det":"DET","detroit":"DET","detroit lions":"DET","lions":"DET",
    "gb":"GB","green bay":"GB","green bay packers":"GB","packers":"GB",
    "hou":"HOU","houston":"HOU","houston texans":"HOU","texans":"HOU",
    "ind":"IND","indianapolis":"IND","indianapolis colts":"IND","colts":"IND",
    "jax":"JAX","jac":"JAX","jacksonville":"JAX","jacksonville jaguars":"JAX","jaguars":"JAX",
    "kc":"KC","kansas city":"KC","kansas city chiefs":"KC","chiefs":"KC",
    "lv":"LV","las vegas":"LV","las vegas raiders":"LV","raiders":"LV","oakland raiders":"LV","oak":"LV",
    "lac":"LAC","la chargers":"LAC","los angeles chargers":"LAC","chargers":"LAC",
    "lar":"LAR","la rams":"LAR","los angeles rams":"LAR","rams":"LAR","st louis rams":"LAR",
    "mia":"MIA","miami":"MIA","miami dolphins":"MIA","dolphins":"MIA",
    "min":"MIN","minnesota":"MIN","minnesota vikings":"MIN","vikings":"MIN",
    "ne":"NE","new england":"NE","new england patriots":"NE","patriots":"NE",
    "no":"NO","new orleans":"NO","new orleans saints":"NO","saints":"NO",
    "nyg":"NYG","new york giants":"NYG","giants":"NYG",
    "nyj":"NYJ","new york jets":"NYJ","jets":"NYJ",
    "phi":"PHI","philadelphia":"PHI","philadelphia eagles":"PHI","eagles":"PHI",
    "pit":"PIT","pittsburgh":"PIT","pittsburgh steelers":"PIT","steelers":"PIT",
    "sea":"SEA","seattle":"SEA","seattle seahawks":"SEA","seahawks":"SEA",
    "sf":"SF","san francisco":"SF","san francisco 49ers":"SF","49ers":"SF","niners":"SF",
    "tb":"TB","tampa bay":"TB","tampa bay buccaneers":"TB","buccaneers":"TB","bucs":"TB",
    "ten":"TEN","tennessee":"TEN","tennessee titans":"TEN","titans":"TEN",
    "was":"WAS","wsh":"WAS","washington":"WAS","washington commanders":"WAS","commanders":"WAS",
    "washington football team":"WAS","washington redskins":"WAS","redskins":"WAS",
}
_DIVISION = {
    "BUF":"AFC_EAST","MIA":"AFC_EAST","NE":"AFC_EAST","NYJ":"AFC_EAST",
    "BAL":"AFC_NORTH","CIN":"AFC_NORTH","CLE":"AFC_NORTH","PIT":"AFC_NORTH",
    "HOU":"AFC_SOUTH","IND":"AFC_SOUTH","JAX":"AFC_SOUTH","TEN":"AFC_SOUTH",
    "DEN":"AFC_WEST","KC":"AFC_WEST","LV":"AFC_WEST","LAC":"AFC_WEST",
    "DAL":"NFC_EAST","NYG":"NFC_EAST","PHI":"NFC_EAST","WAS":"NFC_EAST",
    "CHI":"NFC_NORTH","DET":"NFC_NORTH","GB":"NFC_NORTH","MIN":"NFC_NORTH",
    "ATL":"NFC_SOUTH","CAR":"NFC_SOUTH","NO":"NFC_SOUTH","TB":"NFC_SOUTH",
    "ARI":"NFC_WEST","LAR":"NFC_WEST","SF":"NFC_WEST","SEA":"NFC_WEST",
}

# Only the compact features intended for the fast frozen Production V1 backbones.
PRODUCTION_FEATURES = tuple(OrderedDict.fromkeys(
    x for market in ("SPREADS", "H2H", "TOTALS") for x in COMPACT_FEATURES[market]
))

SCHEDULE_FEATURES = (
    "Week_Number", "Is_Home", "Is_Division_Game",
)
DERIVED_FEATURES = tuple(x for x in PRODUCTION_FEATURES if x not in SCHEDULE_FEATURES)

RAW_REQUIRED = (
    "Season", "Season_Stage", "Source_Name", "Source_Game_ID", "Game_Date", "Week", "Start_Time_ET",
    "Team_Norm", "Opponent_Norm", "Is_Home", "Is_Away", "Is_Neutral",
    "Team_Score", "Opponent_Score", "ATS_Result_Close",
    "Postgame_Total_Yards", "Postgame_Total_Plays",
)

REF_REQUIRED = tuple(OrderedDict.fromkeys((
    "Season", "Season_Stage", "Source_Name", "Source_Game_ID", "Game_Date",
    "Team_Norm", "Opponent_Norm", "Is_Home", "Is_Away", "Is_Neutral",
    *PRODUCTION_FEATURES,
)))


def _norm_name(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = re.sub(r"[^a-z0-9]+", " ", str(x).strip().lower())
    return re.sub(r"\s+", " ", s).strip()


def _team_code(x):
    s=_norm_name(x)
    return _TEAM_ALIASES.get(s, s.upper() if s else "")


def _division_flag(team, opp):
    a,b=_team_code(team),_team_code(opp)
    if not a or not b or a not in _DIVISION or b not in _DIVISION:
        return np.nan
    return float(_DIVISION[a] == _DIVISION[b])


def _season_from_date(x):
    dt=pd.to_datetime(x, errors="coerce")
    if pd.isna(dt): return np.nan
    return int(dt.year-1 if dt.month <= 3 else dt.year)


def _num(x):
    return pd.to_numeric(x, errors="coerce").astype(float)


def _safe_div(a, c):
    aa, cc = _num(a), _num(c)
    return aa.div(cc.where(cc.abs() > 1e-12))


def _ats_value(s):
    z = s.astype(str).str.strip().str.upper()
    return np.where(z.eq("WIN"), 1.0, np.where(z.eq("LOSS"), 0.0, np.where(z.eq("PUSH"), 0.5, np.nan)))


def _win_value(margin):
    m = _num(margin)
    return np.where(m.gt(0), 1.0, np.where(m.lt(0), 0.0, np.where(m.eq(0), 0.5, np.nan)))


def _prior_roll(d: pd.DataFrame, col: str, window: int) -> pd.Series:
    s = _num(d[col])
    keys = [d["Season"], d["Team_Norm"]]
    return s.groupby(keys, sort=False, dropna=False).transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )


def _prior_exp(d: pd.DataFrame, col: str) -> pd.Series:
    s = _num(d[col])
    keys = [d["Season"], d["Team_Norm"]]
    return s.groupby(keys, sort=False, dropna=False).transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )


def _night_from_start(game_date, start_time_et):
    # Historical source uses ET local scheduled time. Keep this explicit and
    # verify it against the historical training view before permitting live use.
    date = pd.to_datetime(game_date, errors="coerce")
    out = []
    for dt, raw in zip(date, start_time_et):
        if pd.isna(dt) or raw is None or (isinstance(raw, float) and math.isnan(raw)):
            out.append(np.nan); continue
        txt = str(raw).strip().upper()
        parsed = pd.to_datetime(txt, format="%I:%M %p", errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(txt, errors="coerce")
        if pd.isna(parsed):
            out.append(np.nan); continue
        out.append(float(parsed.hour >= 18))
    return pd.Series(out, index=date.index, dtype=float)


def derive_compact_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Recompute the compact production prior-state features from raw outcomes.

    Same-game scores/statistics are first converted to outcome state and are then
    shifted/rolled before becoming predictors. No current-game label enters the
    current row's predictor values.
    """
    missing = sorted(set(RAW_REQUIRED) - set(raw.columns))
    if missing:
        raise RuntimeError("NFL_PROD_V1_RAW_COLUMNS_MISSING " + str(missing))
    d = raw.copy()
    d["Season"] = pd.to_numeric(d["Season"], errors="coerce")
    d = d.loc[d.Season.notna()].copy(); d["Season"] = d.Season.astype(int)
    d["Game_Date"] = pd.to_datetime(d["Game_Date"], errors="coerce")
    d["Team_Norm"] = d["Team_Norm"].map(_norm_name)
    d["Opponent_Norm"] = d["Opponent_Norm"].map(_norm_name)
    d = d.sort_values(["Season","Team_Norm","Game_Date","Source_Name","Source_Game_ID"], kind="mergesort").reset_index(drop=True)

    d["actual_margin"] = _num(d.Team_Score) - _num(d.Opponent_Score)
    d["su_win_value"] = _win_value(d.actual_margin)
    d["ats_win_value"] = _ats_value(d.ATS_Result_Close)
    d["off_ypp"] = _safe_div(d.Postgame_Total_Yards, d.Postgame_Total_Plays)

    # Pair current-game opponent offense to obtain this team's defensive YPP
    # allowed. This current outcome is shifted before any predictor is created.
    keys = ["Season","Source_Name","Source_Game_ID"]
    opp_now = d[keys + ["Team_Norm","off_ypp"]].copy().rename(
        columns={"Team_Norm":"Opponent_Norm","off_ypp":"opp_current_off_ypp"}
    )
    d = d.merge(opp_now, on=keys + ["Opponent_Norm"], how="left", validate="one_to_one")
    d["def_ypp_allowed"] = _num(d.opp_current_off_ypp)

    d["WinPct_Prior_System"] = _prior_exp(d, "su_win_value")
    d["ATS_WinPct_Prior_System"] = _prior_exp(d, "ats_win_value")
    d["Avg_SU_Margin_Last5_Prior"] = _prior_roll(d, "actual_margin", 5)
    d["Avg_Points_For_Last5_Prior"] = _prior_roll(d, "Team_Score", 5)
    d["Avg_Points_Against_Last5_Prior"] = _prior_roll(d, "Opponent_Score", 5)
    d["Avg_Off_YPP_Last3_Prior"] = _prior_roll(d, "off_ypp", 3)
    d["Avg_Def_YPP_Last3_Prior"] = _prior_roll(d, "def_ypp_allowed", 3)

    grp = [d["Season"], d["Team_Norm"]]
    prev_date = d["Game_Date"].groupby(grp, sort=False, dropna=False).shift(1)
    d["Days_Since_Last_Game_System"] = (d.Game_Date - prev_date).dt.days.astype(float)

    # Previous-season record is regular-season-only by contract hypothesis. The
    # historical replay will fail closed if the materialized context uses a
    # different definition.
    reg = d.loc[d.Season_Stage.astype(str).str.upper().eq("REGULAR")].copy()
    prior_map = (reg.groupby(["Season","Team_Norm"], sort=False)["su_win_value"].mean().to_dict())
    d["Prior_Season_WinPct"] = [prior_map.get((int(season)-1, team), np.nan) for season,team in zip(d.Season,d.Team_Norm)]

    # Production schedule/context values independently reconstructable from raw
    # identity or the fixed NFL divisional alignment.
    d["Week_Number_Rebuilt"] = pd.to_numeric(d["Week"], errors="coerce")
    d["Is_Home_Rebuilt"] = _num(d["Is_Home"])
    d["Is_Division_Game_Rebuilt"] = [
        _division_flag(t,o) for t,o in zip(d.Team_Norm,d.Opponent_Norm)
    ]

    # Pair opponent's *prior* state at the same game.
    pair_cols = [
        "WinPct_Prior_System","ATS_WinPct_Prior_System","Avg_SU_Margin_Last5_Prior",
        "Avg_Points_For_Last5_Prior","Avg_Points_Against_Last5_Prior",
        "Avg_Off_YPP_Last3_Prior","Avg_Def_YPP_Last3_Prior",
        "Days_Since_Last_Game_System","Prior_Season_WinPct",
    ]
    opp = d[keys + ["Team_Norm"] + pair_cols].copy().rename(
        columns={"Team_Norm":"Opponent_Norm", **{c:"Opp__"+c for c in pair_cols}}
    )
    d = d.merge(opp, on=keys + ["Opponent_Norm"], how="left", validate="one_to_one")

    d["Rest_Differential_Days"] = d.Days_Since_Last_Game_System - _num(d["Opp__Days_Since_Last_Game_System"])
    d["WinPct_Prior_Diff"] = d.WinPct_Prior_System - _num(d["Opp__WinPct_Prior_System"])
    d["ATS_WinPct_Prior_Diff"] = d.ATS_WinPct_Prior_System - _num(d["Opp__ATS_WinPct_Prior_System"])
    d["Avg_SU_Margin_Last5_Diff"] = d.Avg_SU_Margin_Last5_Prior - _num(d["Opp__Avg_SU_Margin_Last5_Prior"])
    d["Off_YPP_vs_Opp_Def_Last3_Diff"] = d.Avg_Off_YPP_Last3_Prior - _num(d["Opp__Avg_Def_YPP_Last3_Prior"])
    d["Def_YPP_vs_Opp_Off_Last3_Diff"] = d.Avg_Def_YPP_Last3_Prior - _num(d["Opp__Avg_Off_YPP_Last3_Prior"])
    d["Prior_Season_WinPct_Diff"] = d.Prior_Season_WinPct - _num(d["Opp__Prior_Season_WinPct"])

    d["Opp_Avg_Points_For_Last5_Prior"] = _num(d["Opp__Avg_Points_For_Last5_Prior"])
    d["Opp_Avg_Points_Against_Last5_Prior"] = _num(d["Opp__Avg_Points_Against_Last5_Prior"])
    d["Opp_Avg_Off_YPP_Last3_Prior"] = _num(d["Opp__Avg_Off_YPP_Last3_Prior"])
    d["Opp_Avg_Def_YPP_Last3_Prior"] = _num(d["Opp__Avg_Def_YPP_Last3_Prior"])

    keep = list(OrderedDict.fromkeys((
        "Season","Source_Name","Source_Game_ID","Game_Date","Team_Norm","Opponent_Norm",
        "Is_Home","Is_Away","Is_Neutral","Week_Number_Rebuilt","Is_Home_Rebuilt","Is_Division_Game_Rebuilt",
        *DERIVED_FEATURES,
    )))
    return d[keep].copy()


def compare_replay(reference: pd.DataFrame, rebuilt: pd.DataFrame) -> dict:
    ref = reference.copy(); reb = rebuilt.copy()
    for z in (ref, reb):
        z["Team_Norm"] = z.Team_Norm.map(_norm_name); z["Opponent_Norm"] = z.Opponent_Norm.map(_norm_name)
    key = ["Season","Source_Name","Source_Game_ID","Team_Norm"]
    m = ref.merge(reb, on=key, how="left", suffixes=("__ref","__reb"), validate="one_to_one", indicator=True)
    fields = {}
    # schedule values that can be independently replayed from raw
    replay_map = {
        "Week_Number":"Week_Number_Rebuilt",
        "Is_Home":"Is_Home_Rebuilt",
        "Is_Division_Game":"Is_Division_Game_Rebuilt",
        **{c:c for c in DERIVED_FEATURES},
    }
    hard_mismatch = 0
    for target, source in replay_map.items():
        rc = target + "__ref" if (target + "__ref") in m.columns else target
        sc = source + "__reb" if (source + "__reb") in m.columns else source
        if rc not in m.columns or sc not in m.columns:
            fields[target] = {"status":"MISSING_COLUMN","reference":rc,"rebuilt":sc}
            hard_mismatch += 1; continue
        a = pd.to_numeric(m[rc], errors="coerce"); z = pd.to_numeric(m[sc], errors="coerce")
        both_missing = a.isna() & z.isna(); one_missing = a.isna() ^ z.isna()
        finite = a.notna() & z.notna()
        diff = (a-z).abs()
        mism = one_missing | (finite & diff.gt(FLOAT_TOL))
        n = int(len(m)); mm = int(mism.sum())
        sample_cols = [c for c in ("Source_Name","Source_Game_ID","Game_Date","Team_Norm","Opponent_Norm") if c in m.columns]
        samples=[]
        if mm:
            idx=m.index[mism][:12]
            for j in idx:
                rec={c:(None if pd.isna(m.at[j,c]) else str(m.at[j,c])) for c in sample_cols}
                rec["reference_value"]=None if pd.isna(a.at[j]) else float(a.at[j])
                rec["rebuilt_value"]=None if pd.isna(z.at[j]) else float(z.at[j])
                samples.append(rec)
        fields[target] = {
            "n":n,"mismatch_rows":mm,"missing_side_disagreement":int(one_missing.sum()),
            "max_abs_diff":float(diff[finite].max()) if finite.any() else None,
            "mean_abs_diff":float(diff[finite].mean()) if finite.any() else None,
            "status":"PASS" if mm==0 else "MISMATCH",
            "mismatch_samples":samples,
        }
        hard_mismatch += mm
    missing_rebuilt = int(m._merge.ne("both").sum())
    status = "PASS" if hard_mismatch==0 and missing_rebuilt==0 else "HOLD"
    return {
        "status":status,"season":HISTORICAL_REPLAY_SEASON,"rows":int(len(m)),
        "missing_rebuilt_rows":missing_rebuilt,"total_field_mismatch_rows":int(hard_mismatch),
        "float_tolerance":FLOAT_TOL,"fields":fields,
        "note":"Production V1 replays Week_Number, Is_Home, static Is_Division_Game, and every derived compact predictor. Is_Neutral and Is_Night_Game are research-only until exact live parity is proven.",
    }


def _query_df(client, sql, params=None):
    cfg = b.QueryJobConfig(query_parameters=params or [])
    return client.query(sql, job_config=cfg).to_dataframe(create_bqstorage_client=False)


def fetch_raw(client):
    cols = {f.name for f in client.get_table(RAW).schema}
    missing = sorted(set(RAW_REQUIRED)-cols)
    if missing:
        raise RuntimeError("NFL_PROD_V1_RAW_SCHEMA_MISSING "+str(missing))
    q = ", ".join(f"`{c}`" for c in RAW_REQUIRED)
    return _query_df(client, f"SELECT {q} FROM `{RAW}` WHERE UPPER(TRIM(CAST(Sport AS STRING)))='NFL' AND Season BETWEEN 2017 AND 2026 AND Season_Stage IN ('REGULAR','POSTSEASON') ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm")


def fetch_reference(client, season=HISTORICAL_REPLAY_SEASON):
    cols = {f.name for f in client.get_table(VIEW).schema}
    missing = sorted(set(REF_REQUIRED)-cols)
    if missing:
        raise RuntimeError("NFL_PROD_V1_REFERENCE_SCHEMA_MISSING "+str(missing))
    q = ", ".join(f"`{c}`" for c in REF_REQUIRED)
    sql = f"SELECT {q} FROM `{VIEW}` WHERE Season=@season AND Season_Stage IN ('REGULAR','POSTSEASON') AND Historical_Core_Eligible=1 ORDER BY Game_Date, Source_Name, Source_Game_ID, Team_Norm"
    return _query_df(client, sql, [b.ScalarQueryParameter("season","INT64",int(season))])


def _market_schema(client):
    cols={f.name for f in client.get_table(MARKET_SOURCE).schema}
    def first(*xs): return next((x for x in xs if x in cols),None)
    m={
        "sport":first("Sport"),"game_start":first("Game_Start","Commence_Hour","feat_Game_Start"),
        "snapshot":first("Snapshot_Timestamp","snapshot_timestamp","Observed_At","Captured_At"),
        "home":first("Home_Team_Norm","Home_Team","Home"),"away":first("Away_Team_Norm","Away_Team","Away"),
        "game_key":first("Merge_Key_Short","Game_Key","Event_ID","Game_ID"),
    }
    missing=[k for k,v in m.items() if k in ("sport","game_start","snapshot","home","away") and not v]
    if missing: raise RuntimeError("NFL_PROD_V1_MARKET_SCHEMA_MISSING "+str(missing))
    return m


def fetch_upcoming_games(client, now=None, lookahead_days=8):
    now=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)
    m=_market_schema(client)
    key_expr=f"CAST(`{m['game_key']}` AS STRING)" if m.get("game_key") else "CAST(NULL AS STRING)"
    sql=f"""
      SELECT SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) game_start,
             CAST(`{m['home']}` AS STRING) home_team,
             CAST(`{m['away']}` AS STRING) away_team,
             {key_expr} source_game_key,
             MAX(SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP)) latest_snapshot
      FROM `{MARKET_SOURCE}`
      WHERE UPPER(TRIM(CAST(`{m['sport']}` AS STRING)))='NFL'
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) > @now
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) <= TIMESTAMP_ADD(@now, INTERVAL {int(lookahead_days)} DAY)
        AND SAFE_CAST(`{m['snapshot']}` AS TIMESTAMP) < SAFE_CAST(`{m['game_start']}` AS TIMESTAMP)
      GROUP BY game_start,home_team,away_team,source_game_key
      ORDER BY game_start
    """
    d=_query_df(client,sql,[b.ScalarQueryParameter("now","TIMESTAMP",now.to_pydatetime())])
    if d.empty:return d,{"status":"NO_UPCOMING_GAMES","games":0}
    d["game_start"]=pd.to_datetime(d.game_start,utc=True,errors="coerce")
    d["home_key"]=d.home_team.map(_norm_name); d["away_key"]=d.away_team.map(_norm_name)
    # collapse multiple source keys to one physical game
    d=d.sort_values("latest_snapshot").drop_duplicates(["game_start","home_key","away_key"],keep="last")
    return d.reset_index(drop=True),{"status":"READY","games":int(len(d))}


def validate_week_calendar(raw: pd.DataFrame) -> dict:
    """Prove the live-eligible NFL regular-season week adapter.

    Standard NFL schedule days are reconstructed from the Tuesday-to-Monday
    schedule bucket learned from each season's authoritative uploader Week-1
    labels. Historical Tuesday/Wednesday regular-season rows are audited
    separately because rescheduled COVID-era games can retain the prior NFL
    week label after crossing the normal calendar boundary.

    Live Production V1 may therefore infer Week_Number only for standard
    schedule weekdays (Thu-Mon plus Fri/Sat). A future Tuesday/Wednesday game
    fails closed unless an authoritative uploader Week label is available for
    that exact date. POSTSEASON remains outside this adapter.
    """
    d=raw.copy()
    d["Game_Date"]=pd.to_datetime(d.Game_Date, errors="coerce").dt.normalize()
    d["Week_Num"]=pd.to_numeric(d.Week, errors="coerce")
    d["Season_Num"]=pd.to_numeric(d.Season, errors="coerce")
    stage=d.get("Season_Stage", pd.Series("", index=d.index)).astype(str).str.upper().str.strip()
    d["__stage"]=stage
    labeled=d.loc[d.Game_Date.notna() & d.Week_Num.notna() & d.Season_Num.notna()].copy()
    postseason_rows=int(labeled.__stage.eq("POSTSEASON").sum())
    reg=labeled.loc[labeled.__stage.eq("REGULAR")].copy()
    reg["Season_Num"]=reg.Season_Num.astype(int)
    reg["__weekday"]=reg.Game_Date.dt.weekday

    anchors={}
    max_week={}
    date_week_labels={}
    for season,g in reg.groupby("Season_Num",sort=True):
        w1=g.loc[g.Week_Num.eq(1),"Game_Date"]
        if w1.empty:
            continue
        first=pd.Timestamp(w1.min())
        days_since_tuesday=(int(first.weekday())-1) % 7
        anchor=(first-pd.Timedelta(days=days_since_tuesday)).normalize()
        anchors[int(season)]=anchor
        mw=pd.to_numeric(g.Week_Num,errors="coerce").max()
        if pd.notna(mw): max_week[int(season)]=int(mw)
        for gd,gg in g.groupby("Game_Date"):
            vals=sorted(set(int(x) for x in pd.to_numeric(gg.Week_Num,errors="coerce").dropna().tolist()))
            if len(vals)==1:
                date_week_labels[f"{int(season)}|{pd.Timestamp(gd).date()}"]=vals[0]

    # Tue/Wed require authoritative exact-date Week metadata; all other regular
    # schedule days use the standard calendar adapter.
    exceptional=reg.loc[reg.__weekday.isin([1,2])].copy()
    standard=reg.loc[~reg.__weekday.isin([1,2])].copy()

    rows=[]; mismatch=[]; season_mismatch=[]
    for r in standard.itertuples(index=False):
        season=int(r.Season_Num)
        inferred_season=_season_from_date(r.Game_Date)
        if inferred_season != season:
            season_mismatch.append({"season":season,"game_date":str(pd.Timestamp(r.Game_Date).date()),"inferred_season":inferred_season})
        anchor=anchors.get(season)
        if anchor is None:
            continue
        pred=1+int((pd.Timestamp(r.Game_Date)-anchor).days//7)
        rows.append((season,float(r.Week_Num),pred))
        if abs(float(r.Week_Num)-float(pred))>1e-9 and len(mismatch)<20:
            mismatch.append({"season":season,"game_date":str(pd.Timestamp(r.Game_Date).date()),"reference_week":float(r.Week_Num),"rebuilt_week":int(pred)})
    total=len(rows)
    mm=sum(1 for _,a,z in rows if abs(a-z)>1e-9)

    exc_samples=[]; exc_mismatch=0
    for r in exceptional.itertuples(index=False):
        anchor=anchors.get(int(r.Season_Num))
        pred=None if anchor is None else 1+int((pd.Timestamp(r.Game_Date)-anchor).days//7)
        bad=pred is None or abs(float(r.Week_Num)-float(pred))>1e-9
        exc_mismatch += int(bad)
        if len(exc_samples)<20:
            exc_samples.append({
                "season":int(r.Season_Num),
                "game_date":str(pd.Timestamp(r.Game_Date).date()),
                "weekday":pd.Timestamp(r.Game_Date).day_name().upper(),
                "reference_week":float(r.Week_Num),
                "calendar_week":None if pred is None else int(pred),
                "requires_authoritative_week":True,
                "calendar_mismatch":bool(bad),
            })

    status="PASS" if total>0 and mm==0 and not season_mismatch else "HOLD"
    return {
        "status":status,"scope":"REGULAR_SEASON_STANDARD_LIVE_ELIGIBLE_DAYS",
        "rows_checked":int(total),"postseason_rows_excluded":postseason_rows,
        "exception_tue_wed_rows":int(len(exceptional)),
        "exception_tue_wed_calendar_mismatch_rows":int(exc_mismatch),
        "exception_policy":"TUESDAY_WEDNESDAY_REQUIRE_AUTHORITATIVE_EXACT_DATE_WEEK_LABEL",
        "exception_samples":exc_samples,
        "week_mismatch_rows":int(mm),"season_date_mismatch_rows":int(len(season_mismatch)),
        "week_mismatch_samples":mismatch,"season_mismatch_samples":season_mismatch[:20],
        "anchors":{str(k):str(v.date()) for k,v in anchors.items()},
        "max_labeled_regular_week":{str(k):int(v) for k,v in max_week.items()},
        "authoritative_date_week_labels":date_week_labels,
        "rule":"standard live days use Tuesday-to-Monday bucket; Tue/Wed require exact authoritative label",
        "boundary":"TUESDAY_00_ET_THROUGH_MONDAY_23_59_ET",
        "live_inference_weekdays":["THURSDAY","FRIDAY","SATURDAY","SUNDAY","MONDAY"],
        "postseason_policy":"FAIL_CLOSED_UNTIL_STAGE_AWARE_POSTSEASON_ADAPTER_PROVEN",
        "date_or_week_guessing":False,
    }

def _calendar_week_for_game(game_start, calendar:dict):
    ts=pd.to_datetime(game_start,utc=True,errors="coerce")
    if pd.isna(ts): return None,None
    local=ts.tz_convert("America/New_York")
    local_date=pd.Timestamp(local.date())
    season=_season_from_date(local.tz_localize(None))
    if season is None:return None,None
    weekday=int(local_date.weekday())
    # Tuesday/Wednesday are historically exception-prone. Accept them only when
    # this exact season/date already has one unambiguous authoritative Week label.
    if weekday in (1,2):
        exact=(calendar.get("authoritative_date_week_labels") or {}).get(f"{int(season)}|{local_date.date()}")
        if exact is None:return None,None
        week=int(exact)
    else:
        anchor_txt=(calendar.get("anchors") or {}).get(str(int(season)))
        if not anchor_txt:return None,None
        anchor=pd.Timestamp(anchor_txt)
        week=1+int((local_date-anchor).days//7)
    if week < 1 or week > 18:return None,None
    return int(season),int(week)

def build_upcoming_features(raw: pd.DataFrame, upcoming: pd.DataFrame, calendar: dict) -> tuple[pd.DataFrame,dict]:
    if upcoming is None or upcoming.empty:
        return pd.DataFrame(),{"status":"NO_UPCOMING_GAMES"}
    if not isinstance(calendar,dict) or calendar.get("status")!="PASS":
        return pd.DataFrame(),{"status":"HOLD_WEEK_CALENDAR_NOT_PROVEN","upcoming_games":int(len(upcoming))}
    u=upcoming.copy()
    u["game_start"]=pd.to_datetime(u.game_start,utc=True,errors="coerce")
    if "home_key" not in u.columns:u["home_key"]=u.home_team.map(_norm_name)
    if "away_key" not in u.columns:u["away_key"]=u.away_team.map(_norm_name)
    u["__identity"]=[f"{pd.Timestamp(gs).round('s').isoformat()}|{h}|{a}" if pd.notna(gs) else f"|{h}|{a}" for gs,h,a in zip(u.game_start,u.home_key,u.away_key)]

    rows=[]; metadata={}; unresolved=[]
    for _,g in u.iterrows():
        season,week=_calendar_week_for_game(g.game_start,calendar)
        home_code,away_code=_team_code(g.home_team),_team_code(g.away_team)
        div=_division_flag(g.home_team,g.away_team)
        if season is None or week is None or pd.isna(div) or home_code not in _DIVISION or away_code not in _DIVISION:
            unresolved.append(g.__identity); continue
        metadata[g.__identity]={"season":season,"week":week,"is_division_game":float(div)}
        for side in ("home","away"):
            team=g.home_team if side=="home" else g.away_team
            opp=g.away_team if side=="home" else g.home_team
            syn={c:np.nan for c in RAW_REQUIRED}
            syn.update({
                "Season":season,"Season_Stage":"REGULAR","Source_Name":"LIVE_PARITY","Source_Game_ID":g.__identity,
                "Game_Date":pd.Timestamp(g.game_start).tz_convert("America/New_York").date(),"Week":week,
                "Start_Time_ET":pd.Timestamp(g.game_start).tz_convert("America/New_York").strftime("%I:%M %p"),
                "Team_Norm":team,"Opponent_Norm":opp,"Is_Home":1 if side=="home" else 0,
                "Is_Away":0 if side=="home" else 1,"Is_Neutral":0.0,
            })
            rows.append(syn)
    if not rows:
        return pd.DataFrame(),{"status":"HOLD_NO_RESOLVABLE_UPCOMING_GAMES","unresolved_games":int(len(unresolved)),"upcoming_games":int(len(u))}

    synthetic=pd.DataFrame(rows)
    combo=pd.concat([raw[list(RAW_REQUIRED)],synthetic[list(RAW_REQUIRED)]],ignore_index=True,sort=False)
    rebuilt=derive_compact_from_raw(combo)
    live=rebuilt.loc[rebuilt.Source_Name.eq("LIVE_PARITY")].copy()
    if live.empty:return live,{"status":"HOLD_LIVE_FEATURE_BUILD_EMPTY"}
    live["__identity"]=live.Source_Game_ID.astype(str)
    live["Week_Number"]=[float(metadata[i]["week"]) if i in metadata else np.nan for i in live.__identity]
    live["Is_Division_Game"]=[float(metadata[i]["is_division_game"]) if i in metadata else np.nan for i in live.__identity]
    live["Is_Home"]=_num(live.Is_Home)

    missing_by_feature={c:int(pd.to_numeric(live.get(c),errors="coerce").isna().sum()) for c in PRODUCTION_FEATURES}
    # First-game/team prior-state NULLs are legitimate historical behavior.  The
    # model pipeline imputes those fields.  Schedule/context fields must never be NULL.
    schedule_missing={c:missing_by_feature[c] for c in SCHEDULE_FEATURES}
    context_ready=all(v==0 for v in schedule_missing.values())
    return live,{
        "status":"READY" if context_ready and not unresolved else "HOLD_UPCOMING_CONTEXT_INCOMPLETE",
        "rows":int(len(live)),"games":int(live.Source_Game_ID.nunique()),
        "upcoming_games":int(len(u)),"unresolved_games":int(len(unresolved)),
        "unresolved_samples":unresolved[:20],
        "missing_by_feature":missing_by_feature,"schedule_missing":schedule_missing,
        "research_only_excluded_features":["Is_Neutral","Is_Night_Game"],
        "division_source":"STATIC_NFL_ALIGNMENT_VALIDATED_BY_HISTORICAL_REPLAY",
        "week_source":"AUTHORITATIVE_UPLOADER_WEEK1_STANDARD_CALENDAR_WITH_TUE_WED_EXCEPTION_GATE",
    }


def run_nfl_live_feature_parity_v1(*, bq_client=None, log_func=print, now=None):
    c=bq_client or b.Client(project=PROJECT)
    started=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)
    log_func("[NFL-PROD-V1-LIVE-PARITY-PREFLIGHT] "+json.dumps({
        "status":"START","source_tag":SOURCE_TAG,"production_authority":0,
        "historical_replay_season":HISTORICAL_REPLAY_SEASON,"markets":list(COMPACT_FEATURES),
        "production_feature_count":len(PRODUCTION_FEATURES),"production_features":list(PRODUCTION_FEATURES),
    },sort_keys=True))

    raw=fetch_raw(c); ref=fetch_reference(c,HISTORICAL_REPLAY_SEASON)
    rebuilt=derive_compact_from_raw(raw)
    rebuilt25=rebuilt.loc[rebuilt.Season.eq(HISTORICAL_REPLAY_SEASON)].copy()
    replay=compare_replay(ref,rebuilt25)
    log_func("[NFL-PROD-V1-HISTORICAL-REPLAY] "+json.dumps(replay,sort_keys=True,default=str))

    calendar=validate_week_calendar(raw)
    log_func("[NFL-PROD-V1-WEEK-CALENDAR] "+json.dumps(calendar,sort_keys=True,default=str))
    upcoming,umeta=fetch_upcoming_games(c,now=started)
    live,lmeta=build_upcoming_features(raw,upcoming,calendar) if not upcoming.empty else (pd.DataFrame(),{"status":"NO_UPCOMING_GAMES"})
    schedule_contract={
        "market":umeta,"week_calendar":calendar,
        "upcoming_feature_build":lmeta,
        "research_only_excluded_features":["Is_Neutral","Is_Night_Game"],
        "external_future_schedule_dependency":False,
    }
    log_func("[NFL-PROD-V1-UPCOMING-FEATURES] "+json.dumps(schedule_contract,sort_keys=True,default=str))

    ready = replay.get("status")=="PASS" and calendar.get("status")=="PASS" and lmeta.get("status")=="READY"
    status = "NFL_PRODUCTION_V1_LIVE_FEATURE_PARITY_PASS" if ready else "HOLD_LIVE_FEATURE_PARITY"
    summary={
        "status":status,"source_tag":SOURCE_TAG,"production_authority":0,
        "historical_replay":replay,"upcoming":schedule_contract,
        "live_prediction_authority":bool(ready),
        "next_step":"FREEZE_PRODUCTION_V1_BACKBONES" if ready else "FIX_ONLY_REPORTED_PARITY_MISMATCHES",
        "automatic_model_promotion":False,
    }
    log_func("[NFL-PROD-V1-LIVE-PARITY-CONTRACT] "+json.dumps({
        "status":status,"historical_replay_status":replay.get("status"),
        "upcoming_feature_status":lmeta.get("status"),"production_authority":0,
        "live_prediction_authority":bool(ready),"automatic_model_promotion":False,
        "next_step":summary["next_step"],
    },sort_keys=True))
    return summary


def synthetic_self_test():
    # Two seasons, four games/team, deterministic divisional opponent pair.
    rows=[]
    gid=0
    for season in (2024,2025):
        for wk in range(1,5):
            gid+=1
            for team,opp,home,score,oscore,yards,plays,ats in [
                ("buffalo bills","miami dolphins",1,20+wk,17+wk,320+wk*5,60,"WIN" if wk%2 else "LOSS"),
                ("miami dolphins","buffalo bills",0,17+wk,20+wk,300+wk*4,58,"LOSS" if wk%2 else "WIN"),
            ]:
                rows.append({"Season":season,"Season_Stage":"REGULAR","Source_Name":"T","Source_Game_ID":str(gid),
                    "Game_Date":pd.Timestamp(f"{season}-09-{wk*7:02d}"),"Week":wk,"Start_Time_ET":"08:20 PM",
                    "Team_Norm":team,"Opponent_Norm":opp,"Is_Home":home,"Is_Away":1-home,"Is_Neutral":0,
                    "Team_Score":score,"Opponent_Score":oscore,"ATS_Result_Close":ats,
                    "Postgame_Total_Yards":yards,"Postgame_Total_Plays":plays})
    d=derive_compact_from_raw(pd.DataFrame(rows))
    r=d.loc[(d.Season.eq(2025)) & d.Team_Norm.eq("buffalo bills")].sort_values("Game_Date").iloc[-1]
    if not np.isfinite(float(r.WinPct_Prior_Diff)):
        raise AssertionError("WIN_PCT_DIFF_MISSING")
    if float(r.Is_Division_Game_Rebuilt)!=1.0:
        raise AssertionError("DIVISION_FLAG_BAD")
    cal=validate_week_calendar(pd.DataFrame(rows))
    if cal.get("status")!="PASS":
        raise AssertionError("WEEK_CALENDAR_BAD "+str(cal))
    return {"status":"PASS","rows":int(len(d)),"production_feature_count":len(PRODUCTION_FEATURES),"week_calendar":cal.get("status")}
