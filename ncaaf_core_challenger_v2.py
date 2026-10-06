"""NCAAF CORE Challenger V2 — expert/specialist Spread CORE research.

Purpose
-------
Search for incremental NCAAF Spread CORE skill that the frozen two-stat residual
brain may be missing, while preserving the production contract and a strict
chronological validation design.

Research chronology
-------------------
* 2022 = initial fit season.
* 2023 = discovery/recipe selection only.
* 2024 and 2025 = untouched confirmation seasons.
* 2026+ = completely sealed; never queried by this module.

What V2 adds
------------
1. Pregame-only sequential team power ratings.
2. Strength-of-schedule and conference-strength estimates derived only from
   games already completed before kickoff.
3. Rest, home/neutral, same-conference, maturity, and market-regime context.
4. Structured football-stat specialists: opponent-adjusted efficiency, passing,
   rushing, turnover pressure, down-conversion proxies, tempo/play mix, and
   offense-vs-defense matchup features.
5. Recent-vs-season trend features and a small set of preregistered football
   interactions (pass/rush mismatch x usage, efficiency x pace, power x form).
6. A regularized program/conference hierarchy using team/opponent and conference
   identities as shrinkable categorical effects.
7. Incumbent-plus-specialist blends selected in discovery and frozen for both
   confirmation years.
8. Conditional specialist attribution: discovery-only regime selection followed
   by untouched 2024 and 2025 confirmation to determine where Power/SOS,
   structured-stat, matchup, program, or advanced-data specialists add
   incremental value without replacing CORE globally.
9. Explicit advanced-data readiness inventory for EPA/play, success rate,
   explosiveness, havoc, line yards, stuff rate, power success, field position,
   drive efficiency, pressure/sack rate, QB efficiency, roster continuity and
   talent/recruiting inputs when those columns become available.

No Pathi, Big Al, Miner, closing lines, live line movement, 2026 outcomes, or
postgame information from the game being predicted may enter a prediction.
Nothing in this file has production authority or automatic promotion rights.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

SOURCE_TAG = "ncaaf-core-challenger-v2.1-conditional-specialist-attribution-20261006"
VERSION = "2.1.0"
REPORT_CURRENT_BLOB = "research/ncaaf/core_challenger/v2/current_report.json"
REPORT_HISTORY_PREFIX = "research/ncaaf/core_challenger/v2/history"
BUNDLE_CURRENT_BLOB = "research/ncaaf/core_challenger/v2/current_bundle.pkl"

DISCOVERY_TRAIN_SEASONS = (2022,)
DISCOVERY_VALIDATION_SEASON = 2023
CONFIRMATION_SEASONS = (2024, 2025)
PROSPECTIVE_MIN_SEASON = 2026

INCUMBENT_FEATURES = (
    "Context_Intercept",
    "Diff_RawRecent3_Off_YPP",
    "B_RawSeason_GameAdj_Def_Rush_YPA",
)

RIDGE_ALPHAS = (6.0, 12.0, 24.0, 48.0, 96.0)
BLEND_WEIGHTS = (0.50, 0.75, 1.00)   # Ridge share; 1.0 = Ridge only
PROGRAM_ALPHAS = (12.0, 24.0, 48.0, 96.0)
INCUMBENT_BLEND_SPECIALIST_WEIGHTS = (0.25, 0.50, 0.75)
EDGE_BANDS = (1.0, 2.0, 3.0, 4.0, 5.0)
BOOTSTRAP_REPS = 500
MIN_DISCOVERY_TRAIN_ROWS = 500
MIN_DISCOVERY_VALID_ROWS = 400
MIN_CONFIRM_ROWS = 400
MAX_STRUCTURED_STATS = 18
MAX_MATCHUP_STATS = 12
MAX_ADVANCED_STATS = 12
TOP_PER_FAMILY = 3
ATTRIBUTION_BLEND_WEIGHT = 0.25
ATTRIBUTION_DISCOVERY_MIN_N = 60
ATTRIBUTION_CONFIRM_MIN_N = 40
ATTRIBUTION_MAX_REGIMES_PER_SPECIALIST = 2

# Sequential power-state constants are preregistered rather than tuned on 2024/25.
POWER_K = 0.16
POWER_MARGIN_CAP = 35.0
OFFSEASON_SHRINK = 0.68
CONF_K = 0.08
CONF_OFFSEASON_SHRINK = 0.60

_BLOCKED_FEATURE_TOKENS = (
    "actual_", "team_score", "opponent_score", "postgame_win", "final_", "result",
    "market_error", "zero_market", "cover_result", "ats_result", "target",
    "q1_points", "q2_points", "q3_points",
)

_ADVANCED_TOKENS = (
    "epa", "success", "explos", "havoc", "line_yards", "stuff_rate",
    "power_success", "field_position", "points_per_opportunity", "drive_",
    "pressure_rate", "sack_rate", "early_down", "passing_down", "qb_",
    "returning_production", "roster_continuity", "recruit", "talent",
)

_ADVANCED_DATA_FAMILIES = {
    "EPA/play": ("epa", "expected_points_added"),
    "Success rate": ("success_rate", "success"),
    "Explosiveness": ("explosiveness", "explosive", "explos"),
    "Havoc": ("havoc",),
    "Line yards": ("line_yards", "lineyards"),
    "Stuff rate": ("stuff_rate", "stuffrate"),
    "Power success": ("power_success", "powersuccess"),
    "Field position": ("field_position", "starting_field"),
    "Drive efficiency": ("drive_efficiency", "drive_points", "points_per_drive", "points_per_opportunity"),
    "Sack / pressure rate": ("sack_rate", "pressure_rate", "pressure_pct"),
    "Early-down efficiency": ("early_down",),
    "Passing-down efficiency": ("passing_down",),
    "QB efficiency": ("qb_epa", "qb_success", "qb_eff", "quarterback"),
    "Returning production / roster continuity": ("returning_production", "roster_continuity", "returning_starters"),
    "Recruiting / talent": ("recruit", "talent", "blue_chip"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _safe(v, default=float("nan")):
    try:
        z = float(v)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def _rmse(y, p) -> float:
    a=np.asarray(y,dtype=float); b=np.asarray(p,dtype=float); ok=np.isfinite(a)&np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok]-b[ok])**2))) if ok.any() else float("nan")


def _mae(y, p) -> float:
    a=np.asarray(y,dtype=float); b=np.asarray(p,dtype=float); ok=np.isfinite(a)&np.isfinite(b)
    return float(np.mean(np.abs(a[ok]-b[ok]))) if ok.any() else float("nan")


def _json_safe(x):
    if isinstance(x, dict): return {str(k):_json_safe(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, pd.Timestamp): return x.isoformat()
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def _norm_text(v) -> str:
    s=str(v or "").strip().lower()
    return "" if s in {"nan","none","<na>"} else s


def _role_home(row: pd.Series) -> float:
    h=_safe(row.get("Is_Home"),0.0); a=_safe(row.get("Is_Away"),0.0); n=_safe(row.get("Is_Neutral"),0.0)
    if n >= 0.5: return 0.0
    if h >= 0.5: return 1.0
    if a >= 0.5: return -1.0
    return 0.0


def _conference_member_mean(team_rating: dict[str,float], team_conf: dict[str,str], conf: str) -> float:
    if not conf: return float("nan")
    vals=[float(team_rating.get(t,0.0)) for t,c in team_conf.items() if c==conf and np.isfinite(team_rating.get(t,0.0))]
    return float(np.mean(vals)) if vals else float("nan")


def _add_strength_context(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str,Any]]:
    """Create prior-only team/program, SOS, conference and rest state.

    The frame contains one oriented row per physical game. Ratings are recorded
    before each game, then updated from that game's final margin. This makes the
    feature state exactly what would have been knowable at kickoff.
    """
    z=g.copy()
    z["__orig_idx"]=np.arange(len(z))
    z["Game_Date"]=pd.to_datetime(z.get("Game_Date"),errors="coerce",utc=True)
    z=z.sort_values(["Game_Date","Season","Source_Game_ID"],kind="stable").reset_index(drop=True)

    team_rating: dict[str,float]={}
    team_conf: dict[str,str]={}
    # Conference membership is schedule metadata, not an outcome. Preload each
    # season's membership so realignment/new members are correct from Week 0.
    season_conf_maps: dict[int,dict[str,str]]={}
    for _,_r in z.iterrows():
        _sy=int(_safe(_r.get("Season"),-1)); _m=season_conf_maps.setdefault(_sy,{})
        _t=_norm_text(_r.get("Team_Norm")); _o=_norm_text(_r.get("Opponent_Norm")); _tc=_norm_text(_r.get("Conference")); _oc=_norm_text(_r.get("Opponent_Conference"))
        if _t and _tc: _m[_t]=_tc
        if _o and _oc: _m[_o]=_oc
    conf_resid: dict[str,float]={}
    sos_sum: dict[str,float]={}; sos_n: dict[str,int]={}
    last_date: dict[str,pd.Timestamp]={}
    season_games: dict[str,int]={}
    current_season=None
    rows=[]

    for _,r in z.iterrows():
        sy=int(_safe(r.get("Season"),-1))
        if current_season is None:
            current_season=sy
            team_conf.update(season_conf_maps.get(sy,{}))
        elif sy != current_season:
            team_rating={k:OFFSEASON_SHRINK*float(v) for k,v in team_rating.items()}
            conf_resid={k:CONF_OFFSEASON_SHRINK*float(v) for k,v in conf_resid.items()}
            sos_sum={}; sos_n={}; last_date={}; season_games={}
            current_season=sy
            team_conf.update(season_conf_maps.get(sy,{}))

        t=_norm_text(r.get("Team_Norm")); o=_norm_text(r.get("Opponent_Norm"))
        tc=_norm_text(r.get("Conference")); oc=_norm_text(r.get("Opponent_Conference"))
        if t and tc: team_conf[t]=tc
        if o and oc: team_conf[o]=oc
        tr=float(team_rating.get(t,0.0)); orr=float(team_rating.get(o,0.0))
        pwr=tr-orr
        t_sos=(sos_sum.get(t,0.0)/sos_n[t]) if sos_n.get(t,0)>0 else np.nan
        o_sos=(sos_sum.get(o,0.0)/sos_n[o]) if sos_n.get(o,0)>0 else np.nan
        tcm=_conference_member_mean(team_rating,team_conf,tc)
        ocm=_conference_member_mean(team_rating,team_conf,oc)
        cavg=(tcm-ocm) if np.isfinite(tcm) and np.isfinite(ocm) else np.nan
        cres=float(conf_resid.get(tc,0.0)-conf_resid.get(oc,0.0)) if tc and oc else np.nan
        gd=r.get("Game_Date")
        rt=(gd-last_date[t]).total_seconds()/86400.0 if t in last_date and pd.notna(gd) else np.nan
        ro=(gd-last_date[o]).total_seconds()/86400.0 if o in last_date and pd.notna(gd) else np.nan
        rest_diff=float(np.clip(rt-ro,-14.0,14.0)) if np.isfinite(rt) and np.isfinite(ro) else np.nan
        gp_t=float(season_games.get(t,0)); gp_o=float(season_games.get(o,0))
        mkt=_safe(r.get("Market_Open_Margin"))
        home=_role_home(r)
        same_conf=float(tc==oc) if tc and oc else np.nan
        rec={
            "Expert_TeamPower":tr,
            "Expert_OppPower":orr,
            "Expert_PowerDiff":pwr,
            "Expert_TeamSOS":t_sos,
            "Expert_OppSOS":o_sos,
            "Expert_SOSDiff":t_sos-o_sos if np.isfinite(t_sos) and np.isfinite(o_sos) else np.nan,
            "Expert_ConferenceMemberPowerDiff":cavg,
            "Expert_CrossConferenceRatingDiff":cres,
            "Expert_TeamVsConference":tr-tcm if np.isfinite(tcm) else np.nan,
            "Expert_OppVsConference":orr-ocm if np.isfinite(ocm) else np.nan,
            "Expert_HomeRole":home,
            "Expert_RestDiffDays":rest_diff,
            "Expert_SameConference":same_conf,
            "Expert_TeamGamesPrior":gp_t,
            "Expert_OppGamesPrior":gp_o,
            "Expert_MinGamesPrior":min(gp_t,gp_o),
            "Expert_EarlyGame":float(min(gp_t,gp_o)<=1.0),
            "Expert_OpenMargin":mkt,
            "Expert_AbsOpenMargin":abs(mkt) if np.isfinite(mkt) else np.nan,
            "Expert_FavoriteRole":float(mkt>0) if np.isfinite(mkt) else np.nan,
            "Expert_PowerVsMarket":pwr-mkt if np.isfinite(mkt) else np.nan,
            "Expert_ConferenceVsMarket":cavg-mkt if np.isfinite(cavg) and np.isfinite(mkt) else np.nan,
            "Expert_PowerUncertainty":1.0/np.sqrt(1.0+min(gp_t,gp_o)),
        }
        rows.append(rec)

        # Update only after recording the pregame state.
        actual=_safe(r.get("Actual_Margin"))
        if np.isfinite(actual) and t and o:
            err=float(np.clip(actual-pwr,-POWER_MARGIN_CAP,POWER_MARGIN_CAP))
            d=POWER_K*err
            team_rating[t]=tr+d; team_rating[o]=orr-d
            if tc and oc and tc!=oc:
                # Cross-conference residual after team power isolates a group effect.
                cerr=float(np.clip(actual-pwr,-POWER_MARGIN_CAP,POWER_MARGIN_CAP))
                cd=CONF_K*cerr
                conf_resid[tc]=float(conf_resid.get(tc,0.0)+cd)
                conf_resid[oc]=float(conf_resid.get(oc,0.0)-cd)
            sos_sum[t]=float(sos_sum.get(t,0.0)+orr); sos_n[t]=int(sos_n.get(t,0)+1)
            sos_sum[o]=float(sos_sum.get(o,0.0)+tr); sos_n[o]=int(sos_n.get(o,0)+1)
            season_games[t]=int(season_games.get(t,0)+1); season_games[o]=int(season_games.get(o,0)+1)
            if pd.notna(gd): last_date[t]=gd; last_date[o]=gd

    expert=pd.DataFrame(rows,index=z.index)
    for c in expert.columns: z[c]=pd.to_numeric(expert[c],errors="coerce")
    z=z.sort_values("__orig_idx").drop(columns=["__orig_idx"]).reset_index(drop=True)
    cols=list(expert.columns)
    cov={c:float(pd.to_numeric(z[c],errors="coerce").notna().mean()) for c in cols}
    return z,cols,{"feature_count":len(cols),"coverage":cov,"power_k":POWER_K,"offseason_shrink":OFFSEASON_SHRINK,"conference_k":CONF_K}


def _add_trend_interactions(g: pd.DataFrame) -> tuple[pd.DataFrame,list[str]]:
    z=g.copy(); made=[]
    key_metrics=(
        "Off_YPP","Off_Pass_YPA","Off_Rush_YPA","Off_Points_Per_Play","Off_Turnover_Rate","Off_FirstDown_Rate",
        "Def_YPP_Allowed","Def_Pass_YPA_Allowed","Def_Rush_YPA_Allowed","Def_Points_Per_Play_Allowed","Def_Takeaway_Rate",
        "GameAdj_Off_YPP","GameAdj_Def_YPP","GameAdj_Off_Pass_YPA","GameAdj_Def_Pass_YPA",
        "GameAdj_Off_Rush_YPA","GameAdj_Def_Rush_YPA","GameAdj_Off_Points_Per_Play","GameAdj_Def_Points_Per_Play",
    )
    for m in key_metrics:
        a=f"Diff_RawRecent3_{m}"; b=f"Diff_RawSeason_{m}"
        if a in z.columns and b in z.columns:
            nm=f"Expert_Trend_{m}"; z[nm]=_num(z,a)-_num(z,b); made.append(nm)

    def mul(nm,a,b):
        if a in z.columns and b in z.columns:
            z[nm]=_num(z,a)*_num(z,b); made.append(nm)
    mul("Expert_PassMismatch_x_PassRate","Matchup_Diff_RawSeason_Pass_YPA","Diff_RawSeason_Off_Pass_Rate")
    mul("Expert_RushMismatch_x_RushRate","Matchup_Diff_RawSeason_Rush_YPA","Diff_RawSeason_Off_Rush_Rate")
    mul("Expert_PassMismatchRecent_x_PassRate","Matchup_Diff_RawRecent3_Pass_YPA","Diff_RawRecent3_Off_Pass_Rate")
    mul("Expert_RushMismatchRecent_x_RushRate","Matchup_Diff_RawRecent3_Rush_YPA","Diff_RawRecent3_Off_Rush_Rate")
    mul("Expert_EfficiencyMismatch_x_Pace","Matchup_Diff_RawSeason_YPP","Mean_RawSeason_Off_Plays_Per_Game")
    mul("Expert_TurnoverPressure_x_Efficiency","Matchup_Diff_RawSeason_Turnover_Pressure","Matchup_Diff_RawSeason_YPP")
    if "Expert_PowerDiff" in z.columns and "Diff_RawRecent3_Off_YPP" in z.columns:
        z["Expert_Power_x_RecentYPP"]=_num(z,"Expert_PowerDiff")*_num(z,"Diff_RawRecent3_Off_YPP"); made.append("Expert_Power_x_RecentYPP")
    if "Expert_PowerVsMarket" in z.columns and "Expert_PowerUncertainty" in z.columns:
        z["Expert_PowerMarket_x_Certainty"]=_num(z,"Expert_PowerVsMarket")*(1.0-_num(z,"Expert_PowerUncertainty")); made.append("Expert_PowerMarket_x_Certainty")
    return z,made


def _feature_family(c: str, dashboard_module=None) -> str:
    s=str(c)
    if s.startswith("Expert_Trend_"): return "trend"
    if s.startswith("Expert_Power") or s.startswith("Expert_TeamPower") or s.startswith("Expert_OppPower"): return "power"
    if "SOS" in s: return "schedule_strength"
    if "Conference" in s or "SameConference" in s: return "conference_strength"
    if "Rest" in s or "HomeRole" in s or "GamesPrior" in s or "EarlyGame" in s: return "situational"
    if "OpenMargin" in s or "FavoriteRole" in s: return "market_regime"
    if "Mismatch" in s or "_x_" in s: return "interaction"
    fn=getattr(dashboard_module,"_ncaaf_stat_feature_family",None)
    if callable(fn):
        try: return str(fn(s))
        except Exception: pass
    lc=s.lower()
    if any(t in lc for t in _ADVANCED_TOKENS): return "advanced"
    if s.startswith("Matchup_"): return "matchup"
    if "gameadj" in lc: return "opponent_adjusted"
    if "turnover" in lc or "takeaway" in lc: return "turnovers"
    if "pass" in lc or "completion" in lc: return "passing"
    if "rush" in lc: return "rushing"
    if "firstdown" in lc: return "conversion"
    if "play" in lc or "pace" in lc: return "tempo"
    if "ypp" in lc or "points_per_play" in lc: return "efficiency"
    return "other"


def _safe_pool(g: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
    disc=g[_num(g,"Season").isin([2022,2023])]
    out=[]
    for c in list(dict.fromkeys(candidate_cols)):
        if c=="Context_Intercept" or c not in g.columns: continue
        lc=str(c).lower()
        if any(t in lc for t in _BLOCKED_FEATURE_TOKENS): continue
        s=pd.to_numeric(disc[c],errors="coerce")
        if int(s.notna().sum())<500 or s.nunique(dropna=True)<2: continue
        out.append(c)
    return out


def _target_residual(df: pd.DataFrame) -> pd.Series:
    return _num(df,"Actual_Margin")-_num(df,"Market_Open_Margin")


def _new_numeric_models(alpha: float):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    ridge=Pipeline([
        ("imputer",SimpleImputer(strategy="median",add_indicator=True)),
        ("scale",StandardScaler()),
        ("ridge",Ridge(alpha=float(alpha))),
    ])
    hgb=Pipeline([
        ("imputer",SimpleImputer(strategy="median",add_indicator=False)),
        ("hgb",HistGradientBoostingRegressor(
            loss="squared_error",learning_rate=0.03,max_iter=160,max_leaf_nodes=12,max_depth=3,
            min_samples_leaf=35,l2_regularization=6.0,random_state=2605,
        )),
    ])
    return ridge,hgb


def _fit_predict_numeric(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], *, alpha: float, blend: float) -> np.ndarray:
    tt=train.copy(); y=_target_residual(tt); good=y.notna().to_numpy(); feats=[c for c in features if c in train.columns and c in valid.columns]
    if len(feats)<1 or int(good.sum())<100: return np.full(len(valid),np.nan)
    ridge,hgb=_new_numeric_models(alpha); X=tt.loc[:,feats]
    ridge.fit(X.loc[good],y.loc[good].to_numpy(float)); p1=np.asarray(ridge.predict(valid.loc[:,feats]),float)
    w=float(blend)
    if w>=0.999: edge=p1
    else:
        hgb.fit(X.loc[good],y.loc[good].to_numpy(float)); p2=np.asarray(hgb.predict(valid.loc[:,feats]),float); edge=w*p1+(1.0-w)*p2
    return _num(valid,"Market_Open_Margin").to_numpy(float)+edge


def _fit_predict_program(train: pd.DataFrame, valid: pd.DataFrame, numeric_features: list[str], alpha: float) -> np.ndarray:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler,OneHotEncoder
    from sklearn.linear_model import Ridge
    cats=[c for c in ("Team_Norm","Opponent_Norm","Conference","Opponent_Conference") if c in train.columns and c in valid.columns]
    nums=[c for c in numeric_features if c in train.columns and c in valid.columns]
    y=_target_residual(train); good=y.notna().to_numpy()
    if int(good.sum())<100 or not (cats or nums): return np.full(len(valid),np.nan)
    transformers=[]
    if nums:
        transformers.append(("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())]),nums))
    if cats:
        transformers.append(("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cats))
    pre=ColumnTransformer(transformers=transformers,remainder="drop",sparse_threshold=0.3)
    model=Pipeline([("pre",pre),("ridge",Ridge(alpha=float(alpha),solver="lsqr"))])
    model.fit(train.loc[good],y.loc[good].to_numpy(float)); edge=np.asarray(model.predict(valid),float)
    return _num(valid,"Market_Open_Margin").to_numpy(float)+edge


def _incumbent_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
    return _fit_predict_numeric(train,valid,list(INCUMBENT_FEATURES),alpha=24.0,blend=0.75)


def _point_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict[str,Any]:
    actual=_num(df,"Actual_Margin").to_numpy(float); market=_num(df,"Market_Open_Margin").to_numpy(float); p=np.asarray(pred,float)
    ok=np.isfinite(actual)&np.isfinite(market)&np.isfinite(p)
    if not ok.any(): return {"n":0}
    a=actual[ok]; m=market[ok]; pp=p[ok]; imp=np.abs(a-m)-np.abs(a-pp)
    return {"n":int(ok.sum()),"rmse":_rmse(a,pp),"mae":_mae(a,pp),"market_rmse":_rmse(a,m),"market_mae":_mae(a,m),
            "rmse_improvement_vs_market":_rmse(a,m)-_rmse(a,pp),"mae_improvement_vs_market":_mae(a,m)-_mae(a,pp),
            "closer_to_actual_than_market_rate":float(np.mean(imp>0)),"mean_abs_error_improvement_vs_market":float(np.mean(imp))}


def _ats_metrics(df: pd.DataFrame, pred: np.ndarray) -> dict[str,Any]:
    actual=_num(df,"Actual_Margin").to_numpy(float); market=_num(df,"Market_Open_Margin").to_numpy(float); p=np.asarray(pred,float)
    edge=p-market; ats=actual-market; out={}
    for band in EDGE_BANDS:
        sel=np.isfinite(edge)&np.isfinite(ats)&(np.abs(edge)>=band)&(~np.isclose(ats,0.0,atol=1e-9)); n=int(sel.sum())
        if not n: out[str(band)]={"n":0,"hit_rate":None,"roi_at_minus110":None}; continue
        win=np.where(edge[sel]>=0,ats[sel]>0,ats[sel]<0); hit=float(np.mean(win)); roi=hit*(100.0/110.0)-(1.0-hit)
        out[str(band)]={"n":n,"hit_rate":hit,"roi_at_minus110":float(roi)}
    return out


def _rank_stats(g: pd.DataFrame, pool: list[str], dashboard_module=None) -> dict[str,Any]:
    tr=g[_num(g,"Season").eq(2022)].copy(); va=g[_num(g,"Season").eq(2023)].copy()
    base=_fit_predict_numeric(tr,va,["Context_Intercept"],alpha=24.0,blend=1.0); by=_num(va,"Actual_Margin").to_numpy(float)
    br=_rmse(by,base); bm=_mae(by,base); rows=[]
    for c in pool:
        try: p=_fit_predict_numeric(tr,va,["Context_Intercept",c],alpha=24.0,blend=1.0)
        except Exception: continue
        rr=_rmse(by,p); mm=_mae(by,p)
        if np.isfinite(rr): rows.append({"feature":c,"family":_feature_family(c,dashboard_module),"rmse_gain":br-rr,"mae_gain":bm-mm,"rmse":rr,"mae":mm})
    rows.sort(key=lambda x:(x["rmse_gain"],x["mae_gain"]),reverse=True)
    fam_counts={}; selected=[]
    priority={"opponent_adjusted","matchup","efficiency","passing","rushing","turnovers","conversion","tempo","advanced","trend","interaction"}
    for r in rows:
        if r["rmse_gain"]<=0: continue
        fam=r["family"]
        if fam not in priority and len(selected)>=10: continue
        if fam_counts.get(fam,0)>=TOP_PER_FAMILY: continue
        selected.append(r["feature"]); fam_counts[fam]=fam_counts.get(fam,0)+1
        if len(selected)>=MAX_STRUCTURED_STATS: break
    return {"selected":selected,"family_counts":fam_counts,"top_ranked":rows[:40],"baseline_rmse":br,"baseline_mae":bm}


def _rank_subset_features(g: pd.DataFrame, pool: list[str], *, max_features: int) -> dict[str,Any]:
    """Discovery-only single-feature screen for a named specialist subset."""
    tr=g[_num(g,"Season").eq(2022)].copy(); va=g[_num(g,"Season").eq(2023)].copy()
    base=_fit_predict_numeric(tr,va,["Context_Intercept"],alpha=24.0,blend=1.0)
    y=_num(va,"Actual_Margin").to_numpy(float); br=_rmse(y,base); bm=_mae(y,base); rows=[]
    for c in list(dict.fromkeys(pool)):
        if c not in g.columns: continue
        try: p=_fit_predict_numeric(tr,va,["Context_Intercept",c],alpha=24.0,blend=1.0)
        except Exception: continue
        rr=_rmse(y,p); mm=_mae(y,p)
        if np.isfinite(rr): rows.append({"feature":c,"rmse_gain":br-rr,"mae_gain":bm-mm,"rmse":rr,"mae":mm})
    rows.sort(key=lambda x:(x["rmse_gain"],x["mae_gain"]),reverse=True)
    selected=[r["feature"] for r in rows if _safe(r.get("rmse_gain"),-9)>0][:int(max_features)]
    return {"selected":selected,"top_ranked":rows[:40],"baseline_rmse":br,"baseline_mae":bm}


def _advanced_data_readiness(g: pd.DataFrame) -> dict[str,Any]:
    """Inventory whether genuinely new advanced inputs are present historically.

    This never fabricates a field and never promotes a partial/current-season-only
    source.  Coverage is measured only on the <=2025 protected historical frame.
    """
    out=[]; all_cols=[str(c) for c in g.columns]
    for family,tokens in _ADVANCED_DATA_FAMILIES.items():
        cols=[c for c in all_cols if any(tok in c.lower() for tok in tokens)]
        cov=0.0
        if cols:
            cov=max(float(pd.to_numeric(g[c],errors="coerce").notna().mean()) for c in cols)
        state="READY" if cols and cov>=0.60 else "PARTIAL" if cols and cov>=0.10 else "MISSING"
        out.append({"family":family,"state":state,"max_historical_coverage":cov,"columns":cols[:20]})
    return {
        "families":out,
        "ready_families":[x["family"] for x in out if x["state"]=="READY"],
        "partial_families":[x["family"] for x in out if x["state"]=="PARTIAL"],
        "missing_families":[x["family"] for x in out if x["state"]=="MISSING"],
        "policy":"USE_ONLY_LEAK_SAFE_HISTORICAL_FIELDS_WITH_DISCOVERY_COVERAGE; NEVER IMPUTE A MISSING DATA SOURCE",
    }


def _discovery_abs_quantile(g: pd.DataFrame, col: str, q: float=0.75, fallback: float=0.0) -> float:
    d=g[_num(g,"Season").eq(DISCOVERY_VALIDATION_SEASON)]
    x=np.abs(_num(d,col).to_numpy(float)); x=x[np.isfinite(x)]
    if not len(x): return float(fallback)
    return float(np.quantile(x,float(q)))


def _regime_specs(g: pd.DataFrame) -> list[dict[str,Any]]:
    """Predeclare context regimes; numeric cut points come only from 2023 discovery."""
    return [
        {"name":"CROSS_CONFERENCE","kind":"context","column":"Expert_SameConference","op":"lt","value":0.5},
        {"name":"SAME_CONFERENCE","kind":"context","column":"Expert_SameConference","op":"ge","value":0.5},
        {"name":"EARLY_SEASON","kind":"context","column":"Expert_MinGamesPrior","op":"le","value":2.0},
        {"name":"MATURE_SEASON","kind":"context","column":"Expert_MinGamesPrior","op":"ge","value":4.0},
        {"name":"LARGE_SOS_GAP","kind":"abs_context","column":"Expert_SOSDiff","op":"ge","value":_discovery_abs_quantile(g,"Expert_SOSDiff",0.75,4.0)},
        {"name":"POWER_MARKET_DISAGREEMENT","kind":"abs_context","column":"Expert_PowerVsMarket","op":"ge","value":_discovery_abs_quantile(g,"Expert_PowerVsMarket",0.75,4.0)},
        {"name":"CONFERENCE_STRENGTH_GAP","kind":"abs_context","column":"Expert_ConferenceMemberPowerDiff","op":"ge","value":_discovery_abs_quantile(g,"Expert_ConferenceMemberPowerDiff",0.75,2.0)},
        {"name":"BIG_SPREAD","kind":"abs_context","column":"Expert_OpenMargin","op":"ge","value":14.0},
        {"name":"CORE_SPECIALIST_STRONG_AGREE","kind":"prediction","mode":"agree","core_edge":2.0,"specialist_edge":2.0},
        {"name":"CORE_SPECIALIST_STRONG_CONFLICT","kind":"prediction","mode":"conflict","core_edge":2.0,"specialist_edge":2.0},
        {"name":"SPECIALIST_CORE_DIVERGENCE","kind":"prediction","mode":"divergence","value":None},
    ]


def _regime_mask(df: pd.DataFrame, spec: dict[str,Any], inc_pred: np.ndarray, specialist_pred: np.ndarray, *, discovery_divergence_cut: float | None=None) -> np.ndarray:
    n=len(df); kind=spec.get("kind")
    if kind in {"context","abs_context"}:
        x=_num(df,str(spec.get("column"))).to_numpy(float)
        if kind=="abs_context": x=np.abs(x)
        v=float(spec.get("value",0.0)); op=spec.get("op")
        if op=="lt": return np.isfinite(x)&(x<v)
        if op=="le": return np.isfinite(x)&(x<=v)
        if op=="gt": return np.isfinite(x)&(x>v)
        return np.isfinite(x)&(x>=v)
    market=_num(df,"Market_Open_Margin").to_numpy(float)
    ip=np.asarray(inc_pred,float); sp=np.asarray(specialist_pred,float)
    ie=ip-market; se=sp-market
    good=np.isfinite(ie)&np.isfinite(se)
    mode=spec.get("mode")
    if mode=="agree": return good&(np.abs(ie)>=float(spec.get("core_edge",2.0)))&(np.abs(se)>=float(spec.get("specialist_edge",2.0)))&(np.sign(ie)==np.sign(se))
    if mode=="conflict": return good&(np.abs(ie)>=float(spec.get("core_edge",2.0)))&(np.abs(se)>=float(spec.get("specialist_edge",2.0)))&(np.sign(ie)!=np.sign(se))
    if mode=="divergence":
        cut=_safe(discovery_divergence_cut,_safe(spec.get("value"),0.0))
        return good&(np.abs(sp-ip)>=cut)
    return np.zeros(n,dtype=bool)


def _ats_side_metrics(df: pd.DataFrame, pred: np.ndarray, mask: np.ndarray, edge_floor: float=2.0) -> dict[str,Any]:
    actual=_num(df,"Actual_Margin").to_numpy(float); market=_num(df,"Market_Open_Margin").to_numpy(float); p=np.asarray(pred,float)
    edge=p-market; ats=actual-market
    sel=np.asarray(mask,bool)&np.isfinite(edge)&np.isfinite(ats)&(np.abs(edge)>=float(edge_floor))&(~np.isclose(ats,0.0,atol=1e-9))
    n=int(sel.sum())
    if not n: return {"n":0,"hit_rate":None,"roi_at_minus110":None}
    win=np.where(edge[sel]>=0,ats[sel]>0,ats[sel]<0); hit=float(np.mean(win)); roi=hit*(100.0/110.0)-(1.0-hit)
    return {"n":n,"hit_rate":hit,"roi_at_minus110":float(roi)}


def _attribution_metrics(df: pd.DataFrame, inc_pred: np.ndarray, spec_pred: np.ndarray, mask: np.ndarray) -> dict[str,Any]:
    m=np.asarray(mask,bool); a=_num(df,"Actual_Margin").to_numpy(float); market=_num(df,"Market_Open_Margin").to_numpy(float)
    ip=np.asarray(inc_pred,float); sp=np.asarray(spec_pred,float); blend=(1.0-ATTRIBUTION_BLEND_WEIGHT)*ip+ATTRIBUTION_BLEND_WEIGHT*sp
    ok=m&np.isfinite(a)&np.isfinite(market)&np.isfinite(ip)&np.isfinite(sp)&np.isfinite(blend)
    n=int(ok.sum())
    if not n: return {"n":0}
    ii=_point_metrics(df.loc[ok].reset_index(drop=True),ip[ok]); ss=_point_metrics(df.loc[ok].reset_index(drop=True),sp[ok]); bb=_point_metrics(df.loc[ok].reset_index(drop=True),blend[ok])
    ie=ip[ok]-market[ok]; se=sp[ok]-market[ok]
    agreement=float(np.mean(np.sign(ie)==np.sign(se))) if n else None
    return {
        "n":n,
        "incumbent_rmse":ii.get("rmse"),"specialist_rmse":ss.get("rmse"),"blend_rmse":bb.get("rmse"),
        "specialist_rmse_gain":_safe(ii.get("rmse"))-_safe(ss.get("rmse")),
        "blend_rmse_gain":_safe(ii.get("rmse"))-_safe(bb.get("rmse")),
        "blend_mae_gain":_safe(ii.get("mae"))-_safe(bb.get("mae")),
        "agreement_rate":agreement,
        "incumbent_ats":_ats_side_metrics(df,ip,m),"specialist_ats":_ats_side_metrics(df,sp,m),"blend_ats":_ats_side_metrics(df,blend,m),
    }


def _bootstrap_attribution(df: pd.DataFrame, inc_pred: np.ndarray, spec_pred: np.ndarray, mask: np.ndarray, reps: int=BOOTSTRAP_REPS) -> dict[str,Any]:
    a=_num(df,"Actual_Margin").to_numpy(float); ip=np.asarray(inc_pred,float); sp=np.asarray(spec_pred,float); blend=(1.0-ATTRIBUTION_BLEND_WEIGHT)*ip+ATTRIBUTION_BLEND_WEIGHT*sp
    ok=np.asarray(mask,bool)&np.isfinite(a)&np.isfinite(ip)&np.isfinite(sp)&np.isfinite(blend)
    a=a[ok]; i=ip[ok]; b=blend[ok]; n=len(a)
    if n<80: return {"n":n,"blend_rmse_gain_ci95":[None,None]}
    rng=np.random.default_rng(2610); gains=[]
    for _ in range(int(reps)):
        ix=rng.integers(0,n,n); gains.append(_rmse(a[ix],i[ix])-_rmse(a[ix],b[ix]))
    return {"n":n,"blend_rmse_gain_ci95":[float(np.quantile(gains,.025)),float(np.quantile(gains,.975))]}


def _conditional_specialist_attribution(g: pd.DataFrame, tuned: list[dict[str,Any]], log_func=print) -> dict[str,Any]:
    """Find *where* a specialist adds incremental value; never grants authority.

    Regime cut points and which regimes advance are chosen from 2023 only.  The
    exact selected regimes are then evaluated separately in 2024 and 2025.
    """
    tr22=g[_num(g,"Season").eq(2022)].copy(); va23=g[_num(g,"Season").eq(2023)].copy()
    inc23=_incumbent_predict(tr22,va23); base_specs=_regime_specs(g)
    registry=[]; selected_all=[]; qualified=[]
    for t in tuned:
        recipe=t.get("recipe") or {}; name=str(recipe.get("name") or "SPECIALIST")
        sp23=_predict_recipe(tr22,va23,recipe)
        divergence=np.abs(np.asarray(sp23,float)-np.asarray(inc23,float)); divergence=divergence[np.isfinite(divergence)]
        div_cut=float(np.quantile(divergence,.75)) if len(divergence) else 2.0
        disc=[]
        for raw_spec in base_specs:
            spec=dict(raw_spec)
            if spec.get("mode")=="divergence": spec["value"]=div_cut
            mask=_regime_mask(va23,spec,inc23,sp23,discovery_divergence_cut=div_cut)
            met=_attribution_metrics(va23,inc23,sp23,mask)
            rec={"specialist":name,"regime":spec,"discovery":met,"selected_for_confirmation":False}
            disc.append(rec)
        eligible=[x for x in disc if int((x.get("discovery") or {}).get("n",0) or 0)>=ATTRIBUTION_DISCOVERY_MIN_N and _safe((x.get("discovery") or {}).get("blend_rmse_gain"),-9)>0]
        eligible.sort(key=lambda x:(_safe((x.get("discovery") or {}).get("blend_rmse_gain"),-9),_safe((x.get("discovery") or {}).get("blend_mae_gain"),-9)),reverse=True)
        selected=eligible[:ATTRIBUTION_MAX_REGIMES_PER_SPECIALIST]
        selected_names={x["regime"]["name"] for x in selected}
        for rec in disc:
            rec["selected_for_confirmation"]=rec["regime"]["name"] in selected_names
            registry.append(rec)
            if rec["selected_for_confirmation"]:
                d=rec.get("discovery") or {}
                log_func(f"[NCAAF-CORE-V21-ATTR-DISCOVERY] specialist={name} regime={rec['regime']['name']} n={d.get('n')} blend_rmse_gain={_safe(d.get('blend_rmse_gain')):+.4f} agreement={_safe(d.get('agreement_rate')):.3f}")
        for rec in selected:
            spec=rec["regime"]; conf={}; pooled_frames=[]; pooled_i=[]; pooled_s=[]; pooled_masks=[]
            for yr in CONFIRMATION_SEASONS:
                tr=g[_num(g,"Season").lt(yr)].copy(); va=g[_num(g,"Season").eq(yr)].copy(); ip=_incumbent_predict(tr,va); sp=_predict_recipe(tr,va,recipe)
                mask=_regime_mask(va,spec,ip,sp,discovery_divergence_cut=_safe(spec.get("value"),div_cut)); met=_attribution_metrics(va,ip,sp,mask); conf[str(yr)]=met
                pooled_frames.append(va); pooled_i.append(ip); pooled_s.append(sp); pooled_masks.append(mask)
                log_func(f"[NCAAF-CORE-V21-ATTR-CONFIRM] specialist={name} regime={spec['name']} season={yr} n={met.get('n',0)} blend_rmse_gain={_safe(met.get('blend_rmse_gain')):+.4f} blend_mae_gain={_safe(met.get('blend_mae_gain')):+.4f} agreement={_safe(met.get('agreement_rate')):.3f}")
            pdf=pd.concat(pooled_frames,ignore_index=True); pi=np.concatenate(pooled_i); ps=np.concatenate(pooled_s); pm=np.concatenate(pooled_masks)
            pooled=_attribution_metrics(pdf,pi,ps,pm); boot=_bootstrap_attribution(pdf,pi,ps,pm)
            both_n=all(int((conf[str(y)] or {}).get("n",0) or 0)>=ATTRIBUTION_CONFIRM_MIN_N for y in CONFIRMATION_SEASONS)
            both_gain=all(_safe((conf[str(y)] or {}).get("blend_rmse_gain"),-9)>0 for y in CONFIRMATION_SEASONS)
            pooled_gain=_safe(pooled.get("blend_rmse_gain"),-9)>0 and _safe(pooled.get("blend_mae_gain"),-9)>=0
            ci_low=_safe((boot.get("blend_rmse_gain_ci95") or [None,None])[0],-9)
            state="QUALIFIED_PROSPECTIVE_SHADOW" if both_n and both_gain and pooled_gain and ci_low>0 else "MIXED" if pooled_gain else "NO_INCREMENTAL_VALUE"
            agree=_safe(pooled.get("agreement_rate"),0.5)
            role="SUPPORT_SHADOW" if agree>=0.60 else "CAUTION_SHADOW" if agree<=0.40 else "CONTEXT_SHADOW"
            out={"specialist":name,"recipe":recipe,"regime":spec,"role":role,"state":state,"discovery":rec.get("discovery"),"confirmation":conf,"pooled":pooled,"bootstrap":boot,"production_authority":0,"bet_authority_vote":False}
            selected_all.append(out)
            if state=="QUALIFIED_PROSPECTIVE_SHADOW": qualified.append(out)
    log_func(f"[NCAAF-CORE-V21-ATTR-REGISTRY] tested={len(registry)} selected={len(selected_all)} qualified_shadow={len(qualified)} production_authority=0 bet_authority_vote=FALSE year_2026_queried=FALSE")
    return {
        "policy":"DISCOVERY_2023_SELECTS_MAX_2_REGIMES_PER_SPECIALIST__2024_AND_2025_BOTH_CONFIRM__PAIRED_BOOTSTRAP_REQUIRED__SHADOW_ONLY",
        "blend_specialist_weight":ATTRIBUTION_BLEND_WEIGHT,
        "discovery_min_n":ATTRIBUTION_DISCOVERY_MIN_N,
        "confirmation_min_n_each_year":ATTRIBUTION_CONFIRM_MIN_N,
        "tested_regimes":registry,
        "selected_regimes":selected_all,
        "qualified_shadow_regimes":qualified,
        "production_authority":0,"bet_authority_vote":False,"automatic_promotion":False,"year_2026_queried":False,
    }


def _tune_numeric(g: pd.DataFrame, features: list[str], name: str, log_func=print) -> dict[str,Any]:
    tr=g[_num(g,"Season").eq(2022)].copy(); va=g[_num(g,"Season").eq(2023)].copy(); rows=[]
    for a in RIDGE_ALPHAS:
        for w in BLEND_WEIGHTS:
            p=_fit_predict_numeric(tr,va,features,alpha=a,blend=w); m=_point_metrics(va,p)
            rows.append({"alpha":a,"blend":w,"metrics":m})
    rows.sort(key=lambda r:(_safe(r["metrics"].get("rmse"),1e9),_safe(r["metrics"].get("mae"),1e9)))
    b=rows[0]; recipe={"type":"NUMERIC","name":name,"features":list(features),"alpha":float(b["alpha"]),"blend":float(b["blend"])}
    log_func(f"[NCAAF-CORE-V2-RECIPE] candidate={name} alpha={b['alpha']} blend={b['blend']} features={len(features)} discovery_rmse={b['metrics'].get('rmse'):.4f} discovery_mae={b['metrics'].get('mae'):.4f}")
    return {"recipe":recipe,"discovery_metrics":b["metrics"],"grid":rows}


def _tune_program(g: pd.DataFrame, numeric_features: list[str], log_func=print) -> dict[str,Any]:
    tr=g[_num(g,"Season").eq(2022)].copy(); va=g[_num(g,"Season").eq(2023)].copy(); rows=[]
    for a in PROGRAM_ALPHAS:
        p=_fit_predict_program(tr,va,numeric_features,a); m=_point_metrics(va,p); rows.append({"alpha":a,"metrics":m})
    rows.sort(key=lambda r:(_safe(r["metrics"].get("rmse"),1e9),_safe(r["metrics"].get("mae"),1e9)))
    b=rows[0]; recipe={"type":"PROGRAM","name":"PROGRAM_HIERARCHY","numeric_features":list(numeric_features),"alpha":float(b["alpha"])}
    log_func(f"[NCAAF-CORE-V2-RECIPE] candidate=PROGRAM_HIERARCHY alpha={b['alpha']} numeric_features={len(numeric_features)} discovery_rmse={b['metrics'].get('rmse'):.4f} discovery_mae={b['metrics'].get('mae'):.4f}")
    return {"recipe":recipe,"discovery_metrics":b["metrics"],"grid":rows}


def _predict_recipe(train: pd.DataFrame, valid: pd.DataFrame, recipe: dict[str,Any]) -> np.ndarray:
    typ=recipe.get("type")
    if typ=="INCUMBENT": return _incumbent_predict(train,valid)
    if typ=="NUMERIC": return _fit_predict_numeric(train,valid,recipe.get("features") or [],alpha=recipe["alpha"],blend=recipe["blend"])
    if typ=="PROGRAM": return _fit_predict_program(train,valid,recipe.get("numeric_features") or [],recipe["alpha"])
    if typ=="BLEND":
        p1=_incumbent_predict(train,valid); p2=_predict_recipe(train,valid,recipe["specialist_recipe"]); w=float(recipe["specialist_weight"])
        return (1.0-w)*p1+w*p2
    raise ValueError(f"unknown recipe type={typ}")


def _evaluate(g: pd.DataFrame, recipe: dict[str,Any], name: str, log_func=print) -> dict[str,Any]:
    seasons={}; preds={}
    for yr in CONFIRMATION_SEASONS:
        tr=g[_num(g,"Season").lt(yr)].copy(); va=g[_num(g,"Season").eq(yr)].copy()
        if len(va)<MIN_CONFIRM_ROWS: raise RuntimeError(f"confirmation rows insufficient year={yr} n={len(va)}")
        p=_predict_recipe(tr,va,recipe); seasons[str(yr)]={"point":_point_metrics(va,p),"ats":_ats_metrics(va,p)}; preds[yr]=(va,p)
        m=seasons[str(yr)]["point"]
        log_func(f"[NCAAF-CORE-V2-CONFIRM] candidate={name} season={yr} n={m.get('n')} rmse={m.get('rmse'):.4f} mae={m.get('mae'):.4f} vs_market_rmse={m.get('rmse_improvement_vs_market'):+.4f} closer={m.get('closer_to_actual_than_market_rate'):.3f}")
    pdf=pd.concat([preds[y][0] for y in CONFIRMATION_SEASONS],ignore_index=True); pp=np.concatenate([preds[y][1] for y in CONFIRMATION_SEASONS])
    return {"name":name,"recipe":recipe,"confirmation":seasons,"pooled":{"point":_point_metrics(pdf,pp),"ats":_ats_metrics(pdf,pp)},"_pooled_df":pdf,"_pooled_pred":pp}


def _bootstrap_vs_incumbent(df: pd.DataFrame, incumbent: np.ndarray, challenger: np.ndarray, reps=BOOTSTRAP_REPS) -> dict[str,Any]:
    a=_num(df,"Actual_Margin").to_numpy(float); i=np.asarray(incumbent,float); c=np.asarray(challenger,float); ok=np.isfinite(a)&np.isfinite(i)&np.isfinite(c)
    a=a[ok]; i=i[ok]; c=c[ok]
    if len(a)<100: return {"n":len(a),"rmse_gain":None,"mae_gain":None,"rmse_gain_ci95":[None,None],"mae_gain_ci95":[None,None]}
    rg=_rmse(a,i)-_rmse(a,c); mg=_mae(a,i)-_mae(a,c); rng=np.random.default_rng(2606); rs=[]; ms=[]; n=len(a)
    for _ in range(int(reps)):
        ix=rng.integers(0,n,n); aa=a[ix]; ii=i[ix]; cc=c[ix]; rs.append(_rmse(aa,ii)-_rmse(aa,cc)); ms.append(_mae(aa,ii)-_mae(aa,cc))
    return {"n":n,"rmse_gain":rg,"mae_gain":mg,"rmse_gain_ci95":[float(np.quantile(rs,.025)),float(np.quantile(rs,.975))],"mae_gain_ci95":[float(np.quantile(ms,.025)),float(np.quantile(ms,.975))]}


def _score_candidate(c: dict[str,Any], incumbent: dict[str,Any], log_func=print) -> None:
    c["vs_incumbent_pooled"]=_bootstrap_vs_incumbent(c["_pooled_df"],incumbent["_pooled_pred"],c["_pooled_pred"])
    yg={}
    for y in CONFIRMATION_SEASONS:
        im=incumbent["confirmation"][str(y)]["point"]; cm=c["confirmation"][str(y)]["point"]
        yg[str(y)]={"rmse_gain":_safe(im.get("rmse"))-_safe(cm.get("rmse")),"mae_gain":_safe(im.get("mae"))-_safe(cm.get("mae"))}
    c["vs_incumbent_by_season"]=yg; b=c["vs_incumbent_pooled"]
    both=all(_safe(yg[str(y)].get("rmse_gain"),-9)>0 for y in CONFIRMATION_SEASONS)
    pooled=_safe(b.get("rmse_gain"),-9)>0; mae=_safe(b.get("mae_gain"),-9)>=0; ci=_safe((b.get("rmse_gain_ci95") or [None])[0],-9)
    c["state"]=("STRONG_CHALLENGER" if both and pooled and mae and ci>0 else "PROMOTION_ELIGIBLE_RESEARCH" if both and pooled and mae else "MIXED" if pooled else "NO_IMPROVEMENT")
    log_func(f"[NCAAF-CORE-V2-SCORECARD] candidate={c['name']} state={c['state']} pooled_rmse_gain={_safe(b.get('rmse_gain')):+.4f} pooled_mae_gain={_safe(b.get('mae_gain')):+.4f} rmse_ci95={b.get('rmse_gain_ci95')} year_gains={yg} production_authority=0")


def run_ncaaf_core_challenger_v2(*, dashboard_module, bucket_name="sharp-models", storage_client=None, log_func=print, hard_fail=True) -> dict[str,Any]:
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}; games=cache.get("games"); base_candidates=list(cache.get("candidate_feature_cols") or [])
        if games is None or getattr(games,"empty",True): raise RuntimeError("historical research games cache missing")
        g=games.copy(); g=g.loc[_num(g,"Season")<=max(CONFIRMATION_SEASONS)].reset_index(drop=True)
        seasons=sorted({int(x) for x in _num(g,"Season").dropna().unique()})
        if not set((2022,2023,2024,2025)).issubset(seasons): raise RuntimeError(f"required seasons missing seasons={seasons}")
        if int((_num(g,"Season")>=2026).sum())!=0: raise RuntimeError("2026 seal failed")
        missing=[c for c in INCUMBENT_FEATURES if c not in g.columns]
        if missing: raise RuntimeError(f"incumbent features missing {missing}")
        log_func(f"[NCAAF-CORE-V2-PREFLIGHT] status=PASS source={SOURCE_TAG} rows={len(g)} seasons={seasons} discovery_train=2022 discovery_validate=2023 confirmation=2024,2025 year_2026_queried=FALSE production_authority=0")

        g,strength_cols,strength_audit=_add_strength_context(g); g,interaction_cols=_add_trend_interactions(g)
        advanced_candidates=[c for c in g.columns if any(t in str(c).lower() for t in _ADVANCED_TOKENS)]
        safe=_safe_pool(g,base_candidates+interaction_cols+advanced_candidates)
        stat_rank=_rank_stats(g,safe,dashboard_module=dashboard_module)
        structured_stats=list(stat_rank.get("selected") or [])
        matchup_pool=[c for c in safe if str(c).startswith("Matchup_") or str(c).startswith("Expert_Trend_") or "Mismatch" in str(c) or "_x_" in str(c)]
        matchup_rank=_rank_subset_features(g,matchup_pool,max_features=MAX_MATCHUP_STATS)
        matchup_stats=list(matchup_rank.get("selected") or [])
        advanced_pool=[c for c in safe if any(tok in str(c).lower() for tok in _ADVANCED_TOKENS)]
        advanced_rank=_rank_subset_features(g,advanced_pool,max_features=MAX_ADVANCED_STATS) if advanced_pool else {"selected":[],"top_ranked":[]}
        advanced_stats=list(advanced_rank.get("selected") or [])
        power_cols=[c for c in strength_cols if c in g.columns and float(_num(g,c).notna().mean())>=0.20]
        # Do not let market magnitude dominate the power-only branch; disagreement with the market is allowed.
        power_core=[c for c in power_cols if c not in {"Expert_OpenMargin","Expert_FavoriteRole"}]
        combined=list(dict.fromkeys(["Context_Intercept"]+power_core+structured_stats+interaction_cols))
        combined=[c for c in combined if c in g.columns]
        power_features=["Context_Intercept"]+power_core
        stat_features=["Context_Intercept"]+structured_stats
        matchup_features=["Context_Intercept"]+matchup_stats
        advanced_features=["Context_Intercept"]+advanced_stats
        # Program hierarchy gets compact numeric context; identity effects are regularized categorical terms.
        program_nums=[c for c in (power_core+structured_stats[:8]) if c in g.columns]

        advanced_present=[c for c in g.columns if any(t in str(c).lower() for t in _ADVANCED_TOKENS)]
        advanced_readiness=_advanced_data_readiness(g)
        conf_cov=float(g.get("Conference",pd.Series("",index=g.index)).astype(str).str.strip().ne("").mean()) if "Conference" in g.columns else 0.0
        log_func(f"[NCAAF-CORE-V2-EXPERT-DESIGN] power_features={len(power_features)-1} structured_stats={len(structured_stats)} matchup_stats={len(matchup_stats)} interactions={len(interaction_cols)} combined_features={len(combined)-1} conference_coverage={conf_cov:.1%} advanced_named_fields={len(advanced_present)}")
        log_func(f"[NCAAF-CORE-V2-STRUCTURED-STATS] selected={structured_stats}")
        log_func(f"[NCAAF-CORE-V21-MATCHUP-STATS] selected={matchup_stats}")
        log_func(f"[NCAAF-CORE-V21-ADVANCED-READINESS] ready={advanced_readiness.get('ready_families')} partial={advanced_readiness.get('partial_families')} missing={advanced_readiness.get('missing_families')}")

        incumbent_recipe={"type":"INCUMBENT","name":"INCUMBENT_RECIPE","features":list(INCUMBENT_FEATURES)}
        inc=_evaluate(g,incumbent_recipe,"INCUMBENT_RECIPE",log_func=log_func)

        tuned=[]
        if len(power_features)>1: tuned.append(_tune_numeric(g,power_features,"POWER_SOS_CONTEXT",log_func=log_func))
        if len(stat_features)>1: tuned.append(_tune_numeric(g,stat_features,"STRUCTURED_STATS",log_func=log_func))
        if len(matchup_features)>1: tuned.append(_tune_numeric(g,matchup_features,"MATCHUP_CONTEXT",log_func=log_func))
        if len(advanced_features)>1: tuned.append(_tune_numeric(g,advanced_features,"ADVANCED_STATS",log_func=log_func))
        if len(combined)>1: tuned.append(_tune_numeric(g,combined,"EXPERT_COMBINED",log_func=log_func))
        if program_nums: tuned.append(_tune_program(g,program_nums,log_func=log_func))
        if not tuned: raise RuntimeError("no specialist recipes available")

        candidates=[]
        discovery_rank=[]
        for t in tuned:
            r=t["recipe"]; name=r.get("name") or "SPECIALIST"; ev=_evaluate(g,r,name,log_func=log_func); candidates.append(ev)
            discovery_rank.append((float(_safe(t["discovery_metrics"].get("rmse"),1e9)),t))
        discovery_rank.sort(key=lambda x:x[0]); best_disc=discovery_rank[0][1]

        # Freeze one blend of the incumbent and the best specialist using 2023 only.
        tr22=g[_num(g,"Season").eq(2022)].copy(); va23=g[_num(g,"Season").eq(2023)].copy(); pi=_incumbent_predict(tr22,va23); ps=_predict_recipe(tr22,va23,best_disc["recipe"])
        blend_grid=[]
        for w in INCUMBENT_BLEND_SPECIALIST_WEIGHTS:
            p=(1.0-w)*pi+w*ps; blend_grid.append({"specialist_weight":float(w),"metrics":_point_metrics(va23,p)})
        blend_grid.sort(key=lambda x:(_safe(x["metrics"].get("rmse"),1e9),_safe(x["metrics"].get("mae"),1e9)))
        bw=blend_grid[0]["specialist_weight"]
        blend_recipe={"type":"BLEND","name":"INCUMBENT_PLUS_SPECIALIST","specialist_weight":bw,"specialist_recipe":best_disc["recipe"]}
        log_func(f"[NCAAF-CORE-V2-RECIPE] candidate=INCUMBENT_PLUS_SPECIALIST specialist={best_disc['recipe'].get('name')} weight={bw:.2f} discovery_rmse={blend_grid[0]['metrics'].get('rmse'):.4f} discovery_mae={blend_grid[0]['metrics'].get('mae'):.4f}")
        candidates.append(_evaluate(g,blend_recipe,"INCUMBENT_PLUS_SPECIALIST",log_func=log_func))

        for c in candidates: _score_candidate(c,inc,log_func=log_func)
        attribution=_conditional_specialist_attribution(g,tuned,log_func=log_func)
        ranked=sorted(candidates,key=lambda c:(_safe((c.get("vs_incumbent_pooled") or {}).get("rmse_gain"),-9),_safe((c.get("vs_incumbent_pooled") or {}).get("mae_gain"),-9)),reverse=True)
        best=ranked[0]; recommendation="CHALLENGER_DESERVES_PROSPECTIVE_SHADOW" if best.get("state") in {"STRONG_CHALLENGER","PROMOTION_ELIGIBLE_RESEARCH"} else "KEEP_INCUMBENT"

        def clean(e): return {k:v for k,v in e.items() if not str(k).startswith("_")}
        report={
            "source_tag":SOURCE_TAG,"version":VERSION,"created_utc":_now(),"status":"NCAAF_CORE_CHALLENGER_V2_COMPLETE",
            "production_authority":0,"production_mutated":False,"automatic_promotion":False,"year_2026_queried":False,
            "discovery":{"train":[2022],"validation":2023},"confirmation_seasons":[2024,2025],"prospective_min_season":2026,
            "incumbent":{"features":list(INCUMBENT_FEATURES),"recipe":incumbent_recipe,"evaluation":clean(inc)},
            "expert_design":{
                "strength_context":strength_audit,"conference_coverage":conf_cov,"advanced_named_fields_present":advanced_present[:50],
                "power_features":power_features,"structured_stat_selection":stat_rank,"matchup_stat_selection":matchup_rank,"advanced_stat_selection":advanced_rank,"interaction_features":interaction_cols,
                "program_numeric_features":program_nums,
                "principles":["prior-only sequential team power","strength of schedule","conference member strength","cross-conference residual strength","rest/home/maturity","structured matchup statistics","standalone matchup specialist","recent-vs-season trends","regularized program/conference hierarchy","discovery-only blend selection","conditional specialist attribution"],
            },
            "advanced_data_readiness":advanced_readiness,
            "conditional_specialist_attribution":attribution,
            "discovery_recipes":[{"recipe":t["recipe"],"discovery_metrics":t["discovery_metrics"]} for t in tuned],
            "incumbent_specialist_blend_grid":blend_grid,
            "challengers":[clean(c) for c in ranked],
            "best_challenger":{"name":best.get("name"),"state":best.get("state"),"recipe":best.get("recipe"),"vs_incumbent_pooled":best.get("vs_incumbent_pooled"),"vs_incumbent_by_season":best.get("vs_incumbent_by_season")},
            "recommendation":recommendation,
            "promotion_contract":"NO_AUTOMATIC_PROMOTION__2024_AND_2025_BOTH_MUST_BEAT_INCUMBENT_RMSE__POOLED_MAE_NONINFERIOR__PAIRED_BOOTSTRAP_REVIEW__PROSPECTIVE_SHADOW_REQUIRED",
            "next_step":"Keep Production V1 frozen. Use conditional specialist attribution to identify support/caution regimes; any QUALIFIED_PROSPECTIVE_SHADOW regime remains zero-authority until prospectively observed. Enrich historical advanced-data families before testing a larger CORE.",
        }
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        raw=json.dumps(_json_safe(report),sort_keys=True,separators=(",",":"),default=str).encode(); sha=hashlib.sha256(raw).hexdigest(); hist=f"{REPORT_HISTORY_PREFIX}/{sha[:16]}/report.json"; b=storage_client.bucket(bucket_name)
        b.blob(hist).upload_from_string(raw,content_type="application/json"); b.blob(REPORT_CURRENT_BLOB).upload_from_string(raw,content_type="application/json")
        bio=io.BytesIO(); pickle.dump({"report":report,"source_tag":SOURCE_TAG},bio,protocol=pickle.HIGHEST_PROTOCOL); b.blob(BUNDLE_CURRENT_BLOB).upload_from_string(bio.getvalue(),content_type="application/octet-stream")
        report["artifact"]={"current_report":f"gs://{bucket_name}/{REPORT_CURRENT_BLOB}","current_bundle":f"gs://{bucket_name}/{BUNDLE_CURRENT_BLOB}","history_report":f"gs://{bucket_name}/{hist}","sha256":sha}
        log_func(f"[NCAAF-CORE-V2-CONTRACT] status=PASS best={best.get('name')} best_state={best.get('state')} recommendation={recommendation} conditional_shadow_qualified={len((attribution or {}).get('qualified_shadow_regimes') or [])} report=gs://{bucket_name}/{REPORT_CURRENT_BLOB} sha={sha[:16]} year_2026_queried=FALSE production_authority=0 production_mutated=FALSE")
        return report
    except Exception as exc:
        log_func(f"[NCAAF-CORE-V2-FAIL] {type(exc).__name__}: {exc}")
        if hard_fail: raise
        return {"source_tag":SOURCE_TAG,"version":VERSION,"status":"FAILED","error":f"{type(exc).__name__}:{exc}","production_authority":0}


def load_current_report(bucket_name="sharp-models", storage_client=None):
    try:
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        blob=storage_client.bucket(bucket_name).blob(REPORT_CURRENT_BLOB)
        if not blob.exists(): return None
        obj=json.loads(blob.download_as_text())
        return obj if obj.get("source_tag")==SOURCE_TAG else None
    except Exception:
        return None


def self_test() -> dict[str,Any]:
    # Unit-check the sequential feature engine on a tiny synthetic chronology.
    q=pd.DataFrame([
        {"Season":2022,"Game_Date":"2022-09-01","Source_Game_ID":"a","Team_Norm":"a","Opponent_Norm":"b","Conference":"x","Opponent_Conference":"y","Actual_Margin":10,"Market_Open_Margin":3,"Is_Home":1,"Is_Away":0,"Is_Neutral":0},
        {"Season":2022,"Game_Date":"2022-09-08","Source_Game_ID":"b","Team_Norm":"a","Opponent_Norm":"c","Conference":"x","Opponent_Conference":"z","Actual_Margin":-4,"Market_Open_Margin":1,"Is_Home":0,"Is_Away":1,"Is_Neutral":0},
    ])
    e,cols,audit=_add_strength_context(q)
    specs=_regime_specs(pd.concat([e.assign(Season=2023),e.assign(Season=2023)],ignore_index=True))
    ok=bool(len(cols)>=10 and abs(float(e.loc[0,"Expert_PowerDiff"]))<1e-12 and float(e.loc[1,"Expert_TeamPower"])>0 and PROSPECTIVE_MIN_SEASON==2026 and any(x.get("name")=="POWER_MARKET_DISAGREEMENT" for x in specs))
    return {"status":"PASS" if ok else "FAIL","source_tag":SOURCE_TAG,"production_authority":0,"automatic_promotion":False,"year_2026_queried":False,"expert_feature_count":len(cols),"regime_spec_count":len(specs),"audit":audit}


if __name__=="__main__":
    print(json.dumps(_json_safe(self_test()),indent=2,sort_keys=True))
