"""NFL Production V1 live-pregame feature parity audit.

Purpose
-------
Prove that the compact, frozen NFL Production V1 feature contract can be
constructed for upcoming games using only information available before kickoff.
The audit has zero production authority and writes no predictions.

The historical replay deliberately recomputes the compact prior-state features
from the authoritative raw team-side history instead of reading them from the
training view.  The recomputed values are then compared field-for-field to the
historical training view.  Upcoming schedule metadata is resolved from the
existing authoritative Big Al schedule/context view; no week/date inference is
used for division/week metadata.
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

SOURCE_TAG = "nfl-production-v1-live-feature-parity-v1.0-20261001"
PROJECT = "sharplogger"
DATASET = "sharp_data"
RAW = f"{PROJECT}.{DATASET}.nfl_historical_game_side_raw"
VIEW = f"{PROJECT}.{DATASET}.nfl_historical_core_training_vw"
MARKET_SOURCE = f"{PROJECT}.{DATASET}.sharp_moves_master"
SCHEDULE_VIEW = f"{PROJECT}.{DATASET}.bigal_game_context_enriched"
PRODUCTION_AUTHORITY = 0
HISTORICAL_REPLAY_SEASON = 2025
FLOAT_TOL = 1e-6

# Only the compact features intended for the fast frozen Production V1 backbones.
PRODUCTION_FEATURES = tuple(OrderedDict.fromkeys(
    x for market in ("SPREADS", "H2H", "TOTALS") for x in COMPACT_FEATURES[market]
))

SCHEDULE_FEATURES = (
    "Week_Number", "Is_Home", "Is_Neutral", "Is_Night_Game", "Is_Division_Game",
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

    # Historical schedule values that can be independently reconstructed from raw.
    d["Week_Number_Rebuilt"] = pd.to_numeric(d["Week"], errors="coerce")
    d["Is_Night_Game_Rebuilt"] = _night_from_start(d.Game_Date, d.Start_Time_ET)

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
        "Is_Home","Is_Away","Is_Neutral","Week_Number_Rebuilt","Is_Night_Game_Rebuilt",
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
        "Is_Night_Game":"Is_Night_Game_Rebuilt",
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
        fields[target] = {
            "n":n,"mismatch_rows":mm,"missing_side_disagreement":int(one_missing.sum()),
            "max_abs_diff":float(diff[finite].max()) if finite.any() else None,
            "mean_abs_diff":float(diff[finite].mean()) if finite.any() else None,
            "status":"PASS" if mm==0 else "MISMATCH",
        }
        hard_mismatch += mm
    missing_rebuilt = int(m._merge.ne("both").sum())
    status = "PASS" if hard_mismatch==0 and missing_rebuilt==0 else "HOLD"
    return {
        "status":status,"season":HISTORICAL_REPLAY_SEASON,"rows":int(len(m)),
        "missing_rebuilt_rows":missing_rebuilt,"total_field_mismatch_rows":int(hard_mismatch),
        "float_tolerance":FLOAT_TOL,"fields":fields,
        "note":"Static Is_Home/Is_Neutral/Is_Division_Game are separately validated as upcoming schedule-source contract; replay independently checks Week, night flag, and every derived compact predictor.",
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


def _schedule_schema(client):
    cols={f.name for f in client.get_table(SCHEDULE_VIEW).schema}
    def first(*xs): return next((x for x in xs if x in cols),None)
    m={
        "game_key":first("Merge_Key_Short","Game_Key","Event_ID","Game_ID"),
        "game_start":first("Game_Start","feat_Game_Start","Commence_Hour"),
        "home":first("Home_Team_Norm","Home_Context_Team_Norm","Home_Team"),
        "away":first("Away_Team_Norm","Away_Context_Team_Norm","Away_Team"),
        "season":first("Season"),"week":first("Week_Number","Week","Game_Week"),
        "division":first("Is_Division_Game"),"neutral":first("Is_Neutral_Site","Is_Neutral"),
    }
    required=("game_start","home","away","season","week","division")
    miss=[x for x in required if not m.get(x)]
    if miss: raise RuntimeError("NFL_PROD_V1_SCHEDULE_SCHEMA_MISSING "+str(miss)+" resolved="+str(m))
    return m


def fetch_schedule_context(client, upcoming: pd.DataFrame):
    if upcoming is None or upcoming.empty:return pd.DataFrame(),{"status":"NO_UPCOMING_GAMES"}
    m=_schedule_schema(client)
    start=(upcoming.game_start.min()-pd.Timedelta(hours=2)).to_pydatetime(); end=(upcoming.game_start.max()+pd.Timedelta(hours=2)).to_pydatetime()
    key_expr=f"CAST(`{m['game_key']}` AS STRING)" if m.get("game_key") else "CAST(NULL AS STRING)"
    neutral_expr=f"SAFE_CAST(`{m['neutral']}` AS INT64)" if m.get("neutral") else "CAST(NULL AS INT64)"
    sql=f"""
      SELECT {key_expr} source_game_key,
             SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) game_start,
             CAST(`{m['home']}` AS STRING) home_team,
             CAST(`{m['away']}` AS STRING) away_team,
             SAFE_CAST(`{m['season']}` AS INT64) season,
             SAFE_CAST(`{m['week']}` AS INT64) week_number,
             SAFE_CAST(`{m['division']}` AS INT64) is_division_game,
             {neutral_expr} is_neutral
      FROM `{SCHEDULE_VIEW}`
      WHERE UPPER(TRIM(CAST(Sport AS STRING)))='NFL'
        AND SAFE_CAST(`{m['game_start']}` AS TIMESTAMP) BETWEEN @start AND @end
    """
    ctx=_query_df(client,sql,[b.ScalarQueryParameter("start","TIMESTAMP",start),b.ScalarQueryParameter("end","TIMESTAMP",end)])
    if ctx.empty:return ctx,{"status":"HOLD_NO_AUTHORITATIVE_SCHEDULE_ROWS"}
    for z in (ctx,):
        z["game_start"]=pd.to_datetime(z.game_start,utc=True,errors="coerce")
        z["home_key"]=z.home_team.map(_norm_name); z["away_key"]=z.away_team.map(_norm_name)
    ctx=ctx.sort_values("game_start").drop_duplicates(["game_start","home_key","away_key"],keep="last")
    return ctx,{"status":"READY","rows":int(len(ctx))}


def build_upcoming_features(raw: pd.DataFrame, upcoming: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    if upcoming.empty:return pd.DataFrame(),{"status":"NO_UPCOMING_GAMES"}
    u=upcoming.copy(); s=schedule.copy()
    u["__identity"]=[f"{pd.Timestamp(gs).round('s').isoformat()}|{h}|{a}" for gs,h,a in zip(u.game_start,u.home_key,u.away_key)]
    s["__identity"]=[f"{pd.Timestamp(gs).round('s').isoformat()}|{h}|{a}" for gs,h,a in zip(s.game_start,s.home_key,s.away_key)]
    sm=s.drop_duplicates("__identity").set_index("__identity")
    missing_identity=[]; rows=[]

    # Precompute team histories from raw outcomes. Use derive_compact_from_raw
    # plus append-only update from completed games; the last raw row contains the
    # prior-state entering that game, so we recompute all outcomes from raw for
    # an upcoming synthetic row below instead of carrying the prior row forward.
    base=raw.copy()
    for _,g in u.iterrows():
        ident=g.__identity
        if ident not in sm.index:
            missing_identity.append(ident); continue
        sc=sm.loc[ident]
        season=int(sc.season); week=float(sc.week_number) if pd.notna(sc.week_number) else np.nan
        for side in ("home","away"):
            team=g.home_team if side=="home" else g.away_team
            opp=g.away_team if side=="home" else g.home_team
            syn={c:np.nan for c in RAW_REQUIRED}
            syn.update({
                "Season":season,"Season_Stage":"REGULAR","Source_Name":"LIVE_PARITY","Source_Game_ID":ident,
                "Game_Date":pd.Timestamp(g.game_start).tz_convert("America/New_York").date(),"Week":week,
                "Start_Time_ET":pd.Timestamp(g.game_start).tz_convert("America/New_York").strftime("%I:%M %p"),
                "Team_Norm":team,"Opponent_Norm":opp,"Is_Home":1 if side=="home" else 0,
                "Is_Away":0 if side=="home" else 1,"Is_Neutral":float(sc.is_neutral) if pd.notna(sc.is_neutral) else 0.0,
            })
            rows.append(syn)
    if not rows:
        return pd.DataFrame(),{"status":"HOLD_NO_EXACT_SCHEDULE_MATCH","missing_schedule_games":int(len(missing_identity))}
    synthetic=pd.DataFrame(rows)
    combo=pd.concat([base[list(RAW_REQUIRED)],synthetic[list(RAW_REQUIRED)]],ignore_index=True,sort=False)
    rebuilt=derive_compact_from_raw(combo)
    live=rebuilt.loc[rebuilt.Source_Name.eq("LIVE_PARITY")].copy()
    if live.empty:return live,{"status":"HOLD_LIVE_FEATURE_BUILD_EMPTY"}

    # Attach authoritative static schedule fields and opponent aliases.
    sched_lookup={}
    for ident,r in sm.iterrows():sched_lookup[ident]=r
    live["__identity"]=live.Source_Game_ID.astype(str)
    live["Week_Number"]=[float(sched_lookup[i].week_number) if i in sched_lookup and pd.notna(sched_lookup[i].week_number) else np.nan for i in live.__identity]
    live["Is_Division_Game"]=[float(sched_lookup[i].is_division_game) if i in sched_lookup and pd.notna(sched_lookup[i].is_division_game) else np.nan for i in live.__identity]
    live["Is_Neutral"]=[float(sched_lookup[i].is_neutral) if i in sched_lookup and pd.notna(sched_lookup[i].is_neutral) else _num(live.Is_Neutral).iloc[j] for j,i in enumerate(live.__identity)]
    game_start_by_identity = dict(zip(u["__identity"], u["game_start"]))
    live["Is_Night_Game"]=[
        float(pd.Timestamp(game_start_by_identity[i]).tz_convert("America/New_York").hour >= 18)
        if i in game_start_by_identity and pd.notna(game_start_by_identity[i]) else np.nan
        for i in live.__identity
    ]
    live["Is_Home"]=_num(live.Is_Home)

    missing_by_feature={c:int(pd.to_numeric(live.get(c),errors="coerce").isna().sum()) for c in PRODUCTION_FEATURES}
    all_required_ready=all(v==0 for v in missing_by_feature.values())
    return live,{"status":"READY" if all_required_ready else "HOLD_MISSING_UPCOMING_FEATURES","rows":int(len(live)),"games":int(live.Source_Game_ID.nunique()),"missing_schedule_games":int(len(missing_identity)),"missing_by_feature":missing_by_feature}


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

    upcoming,umeta=fetch_upcoming_games(c,now=started)
    sched,smeta=fetch_schedule_context(c,upcoming) if not upcoming.empty else (pd.DataFrame(),{"status":"NO_UPCOMING_GAMES"})
    live,lmeta=build_upcoming_features(raw,upcoming,sched) if not upcoming.empty else (pd.DataFrame(),{"status":"NO_UPCOMING_GAMES"})
    schedule_contract={
        "market":umeta,"authoritative_schedule":smeta,
        "upcoming_feature_build":lmeta,"schedule_view":SCHEDULE_VIEW,
        "date_or_week_inference_for_schedule_metadata":False,
    }
    log_func("[NFL-PROD-V1-UPCOMING-FEATURES] "+json.dumps(schedule_contract,sort_keys=True,default=str))

    ready = replay.get("status")=="PASS" and lmeta.get("status")=="READY"
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
    # Three seasons, four games/team in current season, deterministic opponent pair.
    rows=[]
    teams=("alpha","beta")
    gid=0
    for season in (2024,2025):
        for wk in range(1,5):
            gid+=1
            for team,opp,home,score,oscore,yards,plays,ats in [
                ("alpha","beta",1,20+wk,17+wk,320+wk*5,60,"WIN" if wk%2 else "LOSS"),
                ("beta","alpha",0,17+wk,20+wk,300+wk*4,58,"LOSS" if wk%2 else "WIN"),
            ]:
                rows.append({"Season":season,"Season_Stage":"REGULAR","Source_Name":"T","Source_Game_ID":str(gid),
                    "Game_Date":pd.Timestamp(f"{season}-09-{wk*7:02d}"),"Week":wk,"Start_Time_ET":"08:20 PM",
                    "Team_Norm":team,"Opponent_Norm":opp,"Is_Home":home,"Is_Away":1-home,"Is_Neutral":0,
                    "Team_Score":score,"Opponent_Score":oscore,"ATS_Result_Close":ats,
                    "Postgame_Total_Yards":yards,"Postgame_Total_Plays":plays})
    d=derive_compact_from_raw(pd.DataFrame(rows))
    r=d.loc[(d.Season.eq(2025)) & d.Team_Norm.eq("alpha")].sort_values("Game_Date").iloc[-1]
    if not np.isfinite(float(r.WinPct_Prior_Diff)):
        raise AssertionError("WIN_PCT_DIFF_MISSING")
    if float(r.Is_Night_Game_Rebuilt)!=1.0:
        raise AssertionError("NIGHT_FLAG_BAD")
    return {"status":"PASS","rows":int(len(d)),"production_feature_count":len(PRODUCTION_FEATURES)}
