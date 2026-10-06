"""NCAAF Research V2.5 — expert/model atom bridge + evidence decomposition + self-auditing logs.

Research-only architecture built around the frozen NCAAF Production V1 benchmark.
Nothing in this module can grant or mutate production authority.

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
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

NCAAF_RESEARCH_V2_SOURCE_TAG = "ncaaf-research-v2.5-expert-model-atom-bridge-20261006"
NCAAF_RESEARCH_V2_VERSION = "2.5.0"
NCAAF_MINER_LIVE_AUTHORITY_POLICY = "NCAAF_MINER_LIVE_AUTHORITY_V2_2_1_STRONG_VALIDATED_ONLY_20261005"
NCAAF_MINER_LIVE_MIN_CONFIRMATION_N = 60
NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE = 0.56
DISCOVERY_MAX_SEASON = 2023
CONFIRMATION_SEASONS = (2024, 2025)
PROSPECTIVE_MIN_SEASON = 2026
REPORT_CURRENT_BLOB = "research/ncaaf/v2/current_report.json"
BUNDLE_CURRENT_BLOB = "research/ncaaf/v2/current_bundle.pkl"
REPORT_HISTORY_PREFIX = "research/ncaaf/v2/history"


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


# ---------------------------------------------------------------------------
# System Miner V3 — fixed discovery / untouched confirmation / dependence collapse
# ---------------------------------------------------------------------------
def _extended_atoms(g: pd.DataFrame, dashboard_module=None, *, for_live: bool=False, market: str | None=None) -> list[dict[str,Any]]:
    """Leak-safe NCAAF System Miner atom catalog.

    Historical mode applies support floors so sparse identities cannot flood the
    search. Live mode materializes the same named atoms without sample-size
    filtering so a frozen/confirmed rule can be evaluated on one current game.
    Unknown/unavailable inputs fail closed (the atom exists only when its source
    field is actually present).
    """
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
        if for_live or (min_n<=int(mm.sum())<len(g)):
            atoms.append({"name":name,"family":family,"mask":mm,"description":desc or name}); names.add(name)

    sp=nfirst("Consensus_Open_Spread","Opening_Spread")
    tot=nfirst("Consensus_Open_Total","Opening_Total")
    week=nfirst("Context_Week","Week")
    game_no=nfirst("Team_Game_Number_Prior","Game_Number_Prior","Context_Team_Games_Prior")
    is_home=nfirst("Is_Home")

    # Core market role / price regimes.
    add("HOME","VENUE",is_home.eq(1),source_ok=has("Is_Home"))
    add("ROAD","VENUE",is_home.eq(0),source_ok=has("Is_Home"))
    add("CURRENT_DOG","MARKET_ROLE",sp.gt(0),source_ok=has("Consensus_Open_Spread","Opening_Spread"))
    add("CURRENT_FAVORITE","MARKET_ROLE",sp.lt(0),source_ok=has("Consensus_Open_Spread","Opening_Spread"))
    for lo,hi in ((0,3),(3,7),(7,10),(10,14),(14,99)):
        add(f"DOG_{lo}_{hi}","MARKET_PRICE",sp.gt(lo)&sp.le(hi),source_ok=has("Consensus_Open_Spread","Opening_Spread"))
        add(f"FAV_{lo}_{hi}","MARKET_PRICE",(-sp).gt(lo)&(-sp).le(hi),source_ok=has("Consensus_Open_Spread","Opening_Spread"))
    for lo,hi in ((0,45),(45,52),(52,60),(60,99)):
        add(f"TOTAL_{lo}_{hi}","TOTAL_REGIME",tot.ge(lo)&tot.lt(hi),source_ok=has("Consensus_Open_Total","Opening_Total"))

    # Timing / schedule / rest.
    add("EARLY_SEASON_WK1_4","SEASON_TIMING",week.between(1,4),source_ok=has("Context_Week","Week"))
    add("MID_SEASON_WK5_9","SEASON_TIMING",week.between(5,9),source_ok=has("Context_Week","Week"))
    add("LATE_SEASON_WK10_PLUS","SEASON_TIMING",week.ge(10),source_ok=has("Context_Week","Week"))
    add("FIRST_THREE_TEAM_GAMES","SEASON_TIMING",game_no.le(3)&game_no.notna(),source_ok=has("Team_Game_Number_Prior","Game_Number_Prior","Context_Team_Games_Prior"))
    add("GAME_7_PLUS","SEASON_TIMING",game_no.ge(7),source_ok=has("Team_Game_Number_Prior","Game_Number_Prior","Context_Team_Games_Prior"))
    rest=nfirst("Days_Since_Last_Game_System","Days_Since_Last_Game")
    opprest=nfirst("Opp_Days_Since_Last_Game_System","Opp_Days_Since_Last_Game")
    rest_pair=rest.notna()&opprest.notna()
    if for_live or (int(rest_pair.sum())>=100 and (rest[rest_pair]-opprest[rest_pair]).abs().gt(0).any()):
        add("SHORT_REST_6_OR_LESS","REST",rest_pair&rest.le(6),source_ok=has("Days_Since_Last_Game_System","Days_Since_Last_Game"))
        add("REST_8_PLUS","REST",rest_pair&rest.ge(8),source_ok=has("Days_Since_Last_Game_System","Days_Since_Last_Game"))
        add("REST_ADV_2_PLUS","REST",rest_pair&(rest-opprest).ge(2),source_ok=has("Days_Since_Last_Game_System","Days_Since_Last_Game") and has("Opp_Days_Since_Last_Game_System","Opp_Days_Since_Last_Game"))
        add("REST_DISADV_2_PLUS","REST",rest_pair&(rest-opprest).le(-2),source_ok=has("Days_Since_Last_Game_System","Days_Since_Last_Game") and has("Opp_Days_Since_Last_Game_System","Opp_Days_Since_Last_Game"))

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
    pats=nfirst("Pregame_Prev_ATS_Margin","Prev_ATS_Margin","Prev_ATS_Cover_Margin","Context_Prev_ATS_Margin")
    add("OFF_SU_WIN","PRIOR_RESULT",psu.gt(0),source_ok=has("Pregame_Prev_SU_Margin","Prev_SU_Margin","Context_Prev_SU_Margin"))
    add("OFF_SU_LOSS","PRIOR_RESULT",psu.lt(0),source_ok=has("Pregame_Prev_SU_Margin","Prev_SU_Margin","Context_Prev_SU_Margin"))
    for t in (7,14,21):
        add(f"OFF_SU_WIN_{t}_PLUS","PRIOR_MARGIN_MAGNITUDE",psu.ge(t),source_ok=has("Pregame_Prev_SU_Margin","Prev_SU_Margin","Context_Prev_SU_Margin"))
        add(f"OFF_SU_LOSS_{t}_PLUS","PRIOR_MARGIN_MAGNITUDE",psu.le(-t),source_ok=has("Pregame_Prev_SU_Margin","Prev_SU_Margin","Context_Prev_SU_Margin"))
    for t in (7,14):
        add(f"OFF_ATS_COVER_{t}_PLUS","PRIOR_ATS_MAGNITUDE",pats.ge(t),source_ok=has("Pregame_Prev_ATS_Margin","Prev_ATS_Margin","Prev_ATS_Cover_Margin","Context_Prev_ATS_Margin"))
        add(f"OFF_ATS_MISS_{t}_PLUS","PRIOR_ATS_MAGNITUDE",pats.le(-t),source_ok=has("Pregame_Prev_ATS_Margin","Prev_ATS_Margin","Prev_ATS_Cover_Margin","Context_Prev_ATS_Margin"))
    add("ATS_LOSS_STREAK_2","ATS_FORM",nfirst("ATS_Loss_Streak_Prior").ge(2),source_ok=has("ATS_Loss_Streak_Prior"))
    add("ATS_WIN_STREAK_2","ATS_FORM",nfirst("ATS_Win_Streak_Prior").ge(2),source_ok=has("ATS_Win_Streak_Prior"))
    add("SU_WIN_STREAK_2","SU_FORM",nfirst("Current_Win_Streak_Prior","Core_Win_Streak_Prior").ge(2),source_ok=has("Current_Win_Streak_Prior","Core_Win_Streak_Prior"))
    add("SU_LOSS_STREAK_2","SU_FORM",nfirst("Current_Loss_Streak_Prior","Core_Loss_Streak_Prior").ge(2),source_ok=has("Current_Loss_Streak_Prior","Core_Loss_Streak_Prior"))

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
    opats=nfirst("Context_Opp_Prev_ATS_Margin","Opp_Pregame_Prev_ATS_Margin")
    add("OPP_OFF_SU_WIN","OPP_PRIOR_SU1",opsu.gt(0),source_ok=has("Context_Opp_Prev_SU_Margin","Opp_Pregame_Prev_SU_Margin","Opp_Prev_SU_Margin"))
    add("OPP_OFF_SU_LOSS","OPP_PRIOR_SU1",opsu.lt(0),source_ok=has("Context_Opp_Prev_SU_Margin","Opp_Pregame_Prev_SU_Margin","Opp_Prev_SU_Margin"))
    add("OPP_OFF_ATS_WIN","OPP_PRIOR_ATS1",opats.gt(0),source_ok=has("Context_Opp_Prev_ATS_Margin","Opp_Pregame_Prev_ATS_Margin"))
    add("OPP_OFF_ATS_LOSS","OPP_PRIOR_ATS1",opats.lt(0),source_ok=has("Context_Opp_Prev_ATS_Margin","Opp_Pregame_Prev_ATS_Margin"))
    twp=nfirst("Team_WinPct_Prior","Core_Team_WinPct_Prior","WinPct_Prior_System","HC_WinPct_Prior")
    owp=nfirst("Opp_WinPct_Prior","Core_Opp_WinPct_Prior","Opp_WinPct_Prior_System","HC_Opp_WinPct_Prior")
    for nm,ser,fam,ok in (("TEAM",twp,"TEAM_STATE",has("Team_WinPct_Prior","Core_Team_WinPct_Prior","WinPct_Prior_System","HC_WinPct_Prior")),("OPP",owp,"OPP_STATE",has("Opp_WinPct_Prior","Core_Opp_WinPct_Prior","Opp_WinPct_Prior_System","HC_Opp_WinPct_Prior"))):
        add(f"{nm}_WINPCT_LE_400",fam,ser.le(.400),source_ok=ok)
        add(f"{nm}_WINPCT_LE_500",fam,ser.le(.500),source_ok=ok)
        add(f"{nm}_WINPCT_GE_600",fam,ser.ge(.600),source_ok=ok)

    # Role-change / team-price memory.
    prev_dog=nfirst("Prev_Is_ML_Dog")
    prev_fav=nfirst("Prev_Is_ML_Favorite")
    if not prev_fav.notna().any() and prev_dog.notna().any(): prev_fav=1-prev_dog
    add("PRIOR_DOG","ROLE_CHANGE",prev_dog.eq(1),source_ok=has("Prev_Is_ML_Dog"))
    add("PRIOR_FAVORITE","ROLE_CHANGE",prev_fav.eq(1),source_ok=has("Prev_Is_ML_Favorite","Prev_Is_ML_Dog"))
    add("ROLE_FLIP_FAVORITE_TO_DOG","ROLE_CHANGE",prev_fav.eq(1)&sp.gt(0),source_ok=has("Prev_Is_ML_Favorite","Prev_Is_ML_Dog") and has("Consensus_Open_Spread","Opening_Spread"))
    add("ROLE_FLIP_DOG_TO_FAVORITE","ROLE_CHANGE",prev_dog.eq(1)&sp.lt(0),source_ok=has("Prev_Is_ML_Dog") and has("Consensus_Open_Spread","Opening_Spread"))
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
        for c in pathi_cols:
            if has(c):
                mm=nfirst(c).eq(1); p_masks.append(mm)
                add("EXPERT_"+re.sub(r"[^A-Z0-9]+","_",c.upper())[:58],"EXPERT_PATHI",mm,desc=f"Pathi atom: {c}",min_n=20,source_ok=True)
        for c in bigal_cols:
            if has(c):
                mm=nfirst(c).eq(1); b_masks.append(mm)
                add("EXPERT_"+re.sub(r"[^A-Z0-9]+","_",c.upper())[:58],"EXPERT_BIGAL",mm,desc=f"Big Al atom: {c}",min_n=10,source_ok=True)
        if p_masks:
            psum=sum(x.astype(int) for x in p_masks)
            add("EXPERT_PATHI_ANY","EXPERT_PATHI",psum.ge(1),desc="Any directional Pathi atom",min_n=20,source_ok=True)
            add("EXPERT_PATHI_MULTI_2PLUS","EXPERT_PATHI",psum.ge(2),desc="Two or more directional Pathi atoms",min_n=20,source_ok=True)
        if b_masks:
            bsum=sum(x.astype(int) for x in b_masks)
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
                   names: tuple[str,...], families: tuple[str,...], market: str) -> dict[str,Any] | None:
    disc=mask&valid&np.isfinite(seasons)&(seasons<=DISCOVERY_MAX_SEASON)
    ix=np.flatnonzero(disc)
    team_specific="TEAM_SPECIFIC" in families; min_n=60 if team_specific else 100
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
        if jj.sum()>=15:
            row={"season":sy,"n":int(jj.sum()),"rate":float(np.mean(obs[jj]))}
            if market=="h2h":
                pm=_h2h_price_metrics(g,jj,y,baseline,seasons,direction,[sy]); row.update({"priced_n":pm["priced_n"],"roi":pm["roi"],"market_residual":pm["market_residual"]})
            dyears.append(row)
    if len(dyears)<2: return None
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
        "h2h_discovery_price":h2h_price,"h2h_confirmation_price":h2h_conf_price,"mask":mask,
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

def run_system_miner_v3(games: pd.DataFrame, seasons: np.ndarray, market: str, dashboard_module=None,
                        log_func=print, max_depth: int=4) -> dict[str,Any]:
    market=str(market).lower(); y,valid,baseline=_market_target(games,market); atoms=_extended_atoms(games,dashboard_module,market=market)
    _fam_counts={}
    for _a in atoms: _fam_counts[_a.get("family")]=int(_fam_counts.get(_a.get("family"),0))+1
    _bridge_atoms=sum(v for k,v in _fam_counts.items() if str(k).startswith(("EXPERT_","RESEARCH_CORE_STATE","RESEARCH_SPECIALIST_","MARKET_KEY_")))
    out={"version":"NCAAF-RV2.5-SYSTEM-MINER-V5-EXPERT-MODEL-BRIDGE","market":market,"production_authority":0,"discovery_max_season":DISCOVERY_MAX_SEASON,
         "confirmation_seasons":list(CONFIRMATION_SEASONS),"prospective_min_season":PROSPECTIVE_MIN_SEASON,"atoms":len(atoms),"atom_family_counts":_fam_counts,"expert_model_bridge_atoms":int(_bridge_atoms),"systems":[],"mechanism_families":[]}
    log_func(f"[NCAAF-RV25-ATOM-BRIDGE] market={market} atoms={len(atoms)} bridge_atoms={_bridge_atoms} pathi={_fam_counts.get('EXPERT_PATHI',0)} bigal={_fam_counts.get('EXPERT_BIGAL',0)} core={_fam_counts.get('RESEARCH_CORE_STATE',0)} specialist={sum(v for k,v in _fam_counts.items() if str(k).startswith('RESEARCH_SPECIALIST_'))} authority=0")
    if valid.sum()<500: out["status"]="INSUFFICIENT_HISTORY"; return out
    tested=[]; beam=[]; seen=set()
    def ev(mask,names,fams,idx):
        r=_evaluate_rule(games,mask,y,valid,baseline,seasons,names,fams,market)
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
    for i,a in enumerate(atoms):
        r=ev(a["mask"],(a["name"],),(a["family"],),(i,))
        if r: tested.append(r); beam.append(r)
    beam=sorted(beam,key=lambda z:z["quality"],reverse=True)[:64]
    for depth in range(2,max_depth+1):
        nxt=[]
        for st in beam:
            used=set(st["families"])
            for i,a in enumerate(atoms):
                if i in st["idx"] or a["family"] in used: continue
                names=tuple(sorted(st["conditions"]+[a["name"]]))
                if names in seen: continue
                seen.add(names); r=ev(st["mask"]&a["mask"],names,tuple(st["families"]+[a["family"]]),st["idx"]+(i,))
                if r: tested.append(r); nxt.append(r)
        beam=sorted(nxt,key=lambda z:z["quality"],reverse=True)[:64]
        if not beam: break
    if not tested: out.update({"status":"NO_CANDIDATES","tested_hypotheses":0}); return out
    q=_bh_qvalues([z["nominal_pvalue"] for z in tested])
    for z,qq in zip(tested,q): z["fdr_qvalue"]=float(qq) if np.isfinite(qq) else np.nan
    finalists=[]
    for z in sorted(tested,key=lambda r:(r["discovery_pass"],r["quality"],r["discovery_n"]),reverse=True):
        if not z["discovery_pass"] or not np.isfinite(z["fdr_qvalue"]) or z["fdr_qvalue"]>.20: continue
        if market=="h2h":
            hp=z.get("h2h_confirmation_price") or {}; cy=[x for x in z.get("confirmation",[]) if int(x.get("season",0)) in CONFIRMATION_SEASONS]
            both_years=bool(len(cy)==2 and all(int(x.get("priced_n",0) or 0)>=15 and np.isfinite(x.get("roi",np.nan)) and x["roi"]>0 and np.isfinite(x.get("market_residual",np.nan)) and x["market_residual"]>=0 for x in cy))
            conf_ok=bool(
                z["confirmation_n"]>=30 and both_years and int(hp.get("priced_n",0) or 0)>=30 and
                np.isfinite(hp.get("roi",np.nan)) and hp["roi"]>0 and bool(hp.get("price_band_robust",False)) and
                np.isfinite(z["market_residual_confirmation"]) and z["market_residual_confirmation"]>=0 and z["fdr_qvalue"]<=.10
            )
        else:
            conf_ok=bool(z["confirmation_n"]>=30 and np.isfinite(z["confirmation_rate"]) and z["confirmation_rate"]>=.50 and
                         len([x for x in z["confirmation"] if x["n"]>=10])==2 and all(x["rate"]>=.50 for x in z["confirmation"] if x["n"]>=10))
        item={k:v for k,v in z.items() if k not in {"mask","idx","quality"}}
        item.update({"system_id":_stable_id("NCAAF-RV21-"+market.upper()+"-",z["conditions"]),
                     "authority_state":"CONFIRMED_SHADOW" if conf_ok else "DISCOVERY_FROZEN",
                     "confirmation_pass":conf_ok,"production_authority":0,
                     "admission_note":"2024 AND 2025 must confirm frozen discovery; H2H requires observed moneyline ROI/EV and price-band robustness"})
        item["_mask_internal"]=z["mask"]
        finalists.append(item)
        if len(finalists)>=160: break
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
        fam={"mechanism_id":fid,"market":market,"direction":rep["direction"],"representative_system_id":rep["system_id"],
             "representative_conditions":rep.get("conditions") or [],"member_system_ids":[x["system_id"] for x in members],"member_count":len(members),"families":rep["families"],
             "discovery_rate":rep["discovery_rate"],"discovery_n":rep["discovery_n"],"confirmation_rate":rep["confirmation_rate"],
             "confirmation_n":rep["confirmation_n"],"confirmation_pass":bool(rep["confirmation_pass"]),
             "authority_state":"CONFIRMED_SHADOW" if rep["confirmation_pass"] else "DISCOVERY_FROZEN","production_authority":0,
             "attribution":_mechanism_attribution(games,seasons,rep,market)}
        strong_validated=bool(
            rep["confirmation_pass"] and
            int(rep.get("confirmation_n",0) or 0) >= NCAAF_MINER_LIVE_MIN_CONFIRMATION_N and
            float(rep.get("confirmation_rate",0) or 0) >= NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE
        )
        # Confirmation and live betting authority are deliberately different gates.
        # CONFIRMED_SHADOW remains valuable research/prospective evidence; only the
        # stronger validation tier may cast a live Bet Authority vote.
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
        families.append(fam)
    clean=[]
    for z in finalists:
        z=dict(z); z.pop("_mask_internal",None); clean.append(z)
    out.update({"status":"RESEARCH_COMPLETE","tested_hypotheses":len(tested),"systems":clean,"published_systems":len(clean),
                "mechanism_families":families,"mechanism_family_count":len(families),
                "confirmed_mechanism_count":sum(x["confirmation_pass"] for x in families),
                "lineage":lineage,
                "admission_contract":"DISCOVERY_2022_2023_ONLY__HORIZON_SYMMETRIC_1_2_3__CONFERENCE_RIVALRY_H2H_ROLE_TEAM_MEMORY__MARKET_RELATIVE_FDR_FOR_H2H__OBSERVED_ML_ROI__PRICE_BAND_ROBUSTNESS__BOTH_2024_AND_2025_CONFIRM__DEPENDENCY_COLLAPSE__PARENT_CHILD_INCREMENTAL_ATTRIBUTION__2026_PROSPECTIVE_ONLY__ZERO_AUTHORITY"})
    log_func(f"[NCAAF-RV23-MINER] market={market} atoms={len(atoms)} tested={len(tested)} systems={len(clean)} mechanisms={len(families)} confirmed_mechanisms={out['confirmed_mechanism_count']} authority=0")
    for x in families[:20]:
        extra=(f" d_roi={x.get('discovery_price_roi')} c_roi={x.get('confirmation_price_roi')} d_resid={x.get('discovery_market_residual')} c_resid={x.get('confirmation_market_residual')}" if market=="h2h" else "")
        log_func(f"[NCAAF-RV23-MECHANISM] market={market} id={x['mechanism_id']} status={x['authority_state']} members={x['member_count']} discovery={x['discovery_rate']:.4f}/{x['discovery_n']} confirmation={x['confirmation_rate']:.4f}/{x['confirmation_n']} rule={' AND '.join(x.get('representative_conditions') or [])}{extra}")
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
            mechs.append({"mechanism_id":mech.get("mechanism_id"),"direction":direction,"mask":mask,"obs":obs,"families":list(mech.get("families") or []),
                          "rule":" AND ".join(cond),"n":n,"wins":w,"losses":l,"hit_rate":rate,"roi_at_minus110":_minus110_roi(rate) if market in ("spreads","totals") else np.nan,
                          "wilson95_low":_wilson95_low(w,n),"shrunk_hit_rate_beta15":_beta_shrunk_rate(w,n),"evidence_level":mech.get("evidence_level"),"current_qualified":True})
        best=sorted([{k:v for k,v in z.items() if k not in {"mask","obs"}} for z in mechs],key=lambda z:(z.get("wilson95_low",-9),z.get("n",0)),reverse=True)
        play=np.zeros(len(games),dtype=int); fade=np.zeros(len(games),dtype=int)
        for z in mechs:
            if z["direction"]=="PLAY_ON": play += z["mask"].astype(int)
            else: fade += z["mask"].astype(int)
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


def _miner_live_authority_eligible(mech: dict[str,Any] | None) -> bool:
    """Frozen historical gate for whether a Miner family may influence Bet Authority.

    2026+ outcomes are intentionally absent from this decision. A family must have
    passed the original 2024+2025 confirmation gate AND clear the stronger pooled
    confirmation sample/rate floor. We recompute from immutable report fields so
    the V2.2 report already published before this runtime patch remains usable.
    """
    m=mech or {}
    _conds=[str(x) for x in (m.get("representative_conditions") or m.get("conditions") or [])]
    if any(x.startswith("CORE_OOF_") or x.startswith("SPEC_") for x in _conds): return False
    if not bool(m.get("confirmation_pass")): return False
    try: n=int(m.get("confirmation_n",0) or 0)
    except Exception: n=0
    try: rate=float(m.get("confirmation_rate",0) or 0)
    except Exception: rate=0.0
    return bool(n>=NCAAF_MINER_LIVE_MIN_CONFIRMATION_N and np.isfinite(rate) and rate>=NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE)


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
                vote={"source_type":"MINER","family_id":str(mech.get("mechanism_id")),"target":target,"market":str(market).lower(),
                      "mechanisms":list(mech.get("families") or [str(mech.get("mechanism_id"))]),"rule":" AND ".join(cond),
                      "confirmation_n":mech.get("confirmation_n"),"confirmation_rate":mech.get("confirmation_rate"),
                      "evidence_level":"STRONG_VALIDATED" if live_eligible else "VALIDATED_SHADOW",
                      "live_authority_eligible":live_eligible,"live_authority_policy":NCAAF_MINER_LIVE_AUTHORITY_POLICY}
                research_votes_by_game[keys[j]].append(vote)
                if live_eligible: authority_votes_by_game[keys[j]].append(vote)
    # Production consumes ONLY authority-qualified votes. The full confirmed set is
    # preserved separately for research display/prospective tracking.
    out["NCAAF_RV22_Miner_Votes"]=[authority_votes_by_game.get(k,[]) for k in out[keycol]]
    out["NCAAF_RV22_Miner_Research_Votes"]=[research_votes_by_game.get(k,[]) for k in out[keycol]]
    out["NCAAF_Miner_Confirmed_Research"]=confirmed_research
    out["NCAAF_Miner_Qualified"]=authority_qualified
    out["NCAAF_Miner_Evaluable"]=authority_evaluable
    out["NCAAF_Miner_Research_Evaluable"]=research_evaluable
    out["NCAAF_Miner_Live_Authority_Policy"]=NCAAF_MINER_LIVE_AUTHORITY_POLICY
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


def run_ncaaf_research_v2(*, dashboard_module, utils_module=None, bucket_name="sharp-models", storage_client=None,
                          log_func=print, hard_fail=True) -> dict[str,Any]:
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}
        games=cache.get("games"); seasons=np.asarray(cache.get("season_arr"),dtype=float); oof_margin=np.asarray(cache.get("oof_margin"),dtype=float); oof_total=np.asarray(cache.get("oof_total"),dtype=float)
        cols=list(cache.get("candidate_feature_cols") or []); miner_games=cache.get("miner_games")
        if games is None or getattr(games,"empty",True): raise RuntimeError("historical research games cache missing")
        if len(seasons)!=len(games) or len(oof_margin)!=len(games) or len(oof_total)!=len(games): raise RuntimeError("OOF/cache row alignment mismatch")
        if miner_games is None or getattr(miner_games,"empty",True): miner_games=games.copy()
        # Hard seal: 2026+ may exist in source tables, but can never enter discovery/confirmation.
        historic=np.isfinite(seasons)&(seasons<=max(CONFIRMATION_SEASONS))
        if historic.sum()<500: raise RuntimeError(f"insufficient <=2025 history n={int(historic.sum())}")
        g=games.loc[historic].reset_index(drop=True); mg=miner_games.loc[historic].reset_index(drop=True); sy=seasons[historic]; om=oof_margin[historic]; ot=oof_total[historic]
        log_func(f"[NCAAF-RV23-PREFLIGHT] source={NCAAF_RESEARCH_V2_SOURCE_TAG} rows={len(g)} seasons={sorted(set(sy.astype(int)))} discovery<=2023 confirmation=2024,2025 prospective>=2026 production_authority=0")
        stat=run_orthogonal_stat_research(g,sy,om,ot,cols,log_func=log_func)
        sparse_stat=run_sparse_stat_research(g,sy,om,ot,log_func=log_func)
        miners={m:run_system_miner_v3(mg,sy,m,dashboard_module=dashboard_module,log_func=log_func,max_depth=5) for m in ("spreads","h2h","totals")}
        system_results=_build_system_results(mg,sy,miners,dashboard_module=dashboard_module)
        published_system_results=_grade_published_ncaaf_systems(mg,sy)
        miner_threshold_neighborhood=_miner_authority_threshold_neighborhood(miners)
        _log_v24_evidence_audit(miners=miners,system_results=system_results,published_system_results=published_system_results,
                                threshold_neighborhood=miner_threshold_neighborhood,log_func=log_func)
        prospective=_prospective_shadow(miner_games.reset_index(drop=True),seasons,miners,dashboard_module=dashboard_module,log_func=log_func)
        market_audit=_market_rich_audit(g,utils_module)
        intelligence_bridge=(getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{}) or {}).get("miner_intelligence_bridge") or {"status":"NOT_AVAILABLE","production_authority":0,"selection_influence":0}
        report={"source_tag":NCAAF_RESEARCH_V2_SOURCE_TAG,"version":NCAAF_RESEARCH_V2_VERSION,"created_utc":_now(),
                "status":"NCAAF_RESEARCH_V2_COMPLETE","production_authority":0,"production_contract_mutated":False,
                "benchmark":"FROZEN_NCAAF_PRODUCTION_V1","discovery_max_season":DISCOVERY_MAX_SEASON,"confirmation_seasons":list(CONFIRMATION_SEASONS),"prospective_min_season":PROSPECTIVE_MIN_SEASON,
                "rows":len(g),"seasons":sorted(set(sy.astype(int))),"orthogonal_stat":stat,"sparse_stat_v21":sparse_stat,"system_miner_v3":miners,
                "prospective_shadow_2026":prospective,"system_results":system_results,"published_system_results":published_system_results,"miner_threshold_neighborhood":miner_threshold_neighborhood,"market_rich":market_audit,"intelligence_bridge":intelligence_bridge,
                "miner_live_authority_policy":{"policy":NCAAF_MINER_LIVE_AUTHORITY_POLICY,"min_confirmation_n":NCAAF_MINER_LIVE_MIN_CONFIRMATION_N,"min_confirmation_rate":NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE,"uses_2026_selection":False},
                "next_step":"KEEP PRODUCTION V1 FROZEN; LET THE EXISTING MINER TEST PATHI + BIG AL + OOF CORE/SPECIALIST STATE INTERACTIONS; CORE/SPECIALIST BRIDGE ATOMS REMAIN RESEARCH-ONLY UNTIL A LIKE-FOR-LIKE LIVE BRIDGE EXISTS"}
        # Preserve a lightweight pickle bundle for future prospective trigger/scoring adapters.
        bundle={"report":report,"system_miner_v3":miners,"sparse_stat_v21":sparse_stat,"prospective_shadow_2026":prospective,"system_results":system_results,"published_system_results":published_system_results,"miner_threshold_neighborhood":miner_threshold_neighborhood,"stat_family_definitions":STAT_FAMILY_TOKENS,"source_tag":NCAAF_RESEARCH_V2_SOURCE_TAG}
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        b=storage_client.bucket(bucket_name)
        body=json.dumps(_report_without_models(report),sort_keys=True,separators=(",",":"),default=str).encode()
        sha=hashlib.sha256(body).hexdigest(); hist=f"{REPORT_HISTORY_PREFIX}/{sha[:16]}/report.json"
        b.blob(hist).upload_from_string(body,content_type="application/json"); b.blob(REPORT_CURRENT_BLOB).upload_from_string(body,content_type="application/json")
        bio=io.BytesIO(); pickle.dump(bundle,bio,protocol=pickle.HIGHEST_PROTOCOL); bio.seek(0); pdata=bio.read(); b.blob(BUNDLE_CURRENT_BLOB).upload_from_string(pdata,content_type="application/octet-stream")
        report["artifact"]={"current_report":f"gs://{bucket_name}/{REPORT_CURRENT_BLOB}","current_bundle":f"gs://{bucket_name}/{BUNDLE_CURRENT_BLOB}","history_report":f"gs://{bucket_name}/{hist}","sha256":sha}
        _strong=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if _miner_live_authority_eligible(_m))
        _bridge_mechs=sum(1 for _mr in miners.values() for _m in (_mr.get("mechanism_families") or []) if any(str(c).startswith(("EXPERT_PATHI_","EXPERT_BIGAL_","CORE_OOF_","SPEC_")) for c in (_m.get("representative_conditions") or [])))
        log_func(f"[NCAAF-RV25-CONTRACT] status=PASS report=gs://{bucket_name}/{REPORT_CURRENT_BLOB} sha={sha[:16]} stat_spread_confirmed={len(stat['confirmed_spread_families'])} stat_totals_confirmed={len(stat['confirmed_totals_families'])} sparse_confirmed={len(sparse_stat.get('confirmed_candidates') or [])} miner_confirmed={sum(v.get('confirmed_mechanism_count',0) for v in miners.values())} bridge_mechanisms={_bridge_mechs} miner_live_authority={_strong} prospective_mechanisms={len((prospective or {}).get('mechanisms') or [])} production_authority=0")
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
    _tf=pd.DataFrame({"Consensus_Open_Spread":[3.5],"Prev_SU_Margin":[-7.0],"Prev2_SU_Margin":[10.0],"Prev3_SU_Margin":[-3.0],"Prev_ATS_Margin":[8.0],"Prev2_ATS_Margin":[-2.0],"Prev3_ATS_Margin":[5.0],
                      "Pathi_FB_Dog_Hook_Above_3":[1],"BigAl_CF2_LateSeasonRevengeDog":[1],"_V29_CORE_INCUMBENT_EDGE_POINTS":[3.0],
                      "_V29_SPEC_STRUCTURED_STATS_EDGE_POINTS":[2.5],"_V29_SPEC_STRUCTURED_STATS_DIVERGENCE_FROM_CORE":[1.5],"_V29_SPEC_STRUCTURED_STATS_DIVERGENCE_CUT":[1.0]})
    live_atoms={a["name"] for a in _extended_atoms(_tf,for_live=True,market="spreads")}
    q=_bh_qvalues([.01,.04,.20]); fam=_classify_feature_families(["Rush_EPA","Opp_Rush_EPA","Line_Move_60m","Sharp_Soft_Divergence","Actual_Margin"])
    odds=np.asarray([200.0,-200.0]); won=np.asarray([1.0,1.0]); ret=_american_unit_return(odds,won)
    ok=bool(
        len(q)==3 and "RUN_PASS_MATCHUP" in fam and "MARKET_MICROSTRUCTURE" in fam and
        all("Actual_Margin" not in x for v in fam.values() for x in v) and
        "SU_SEQ3_LWL" in live_atoms and "OFF_ATS_COVER_7_PLUS" in live_atoms and
        "EXPERT_PATHI_FB_DOG_HOOK_ABOVE_3" in live_atoms and "EXPERT_BIGAL_CF2_LATESEASONREVENGEDOG" in live_atoms and
        "CORE_OOF_EDGE_TEAM_2PLUS" in live_atoms and "SPEC_STRUCTURED_STATS_CORE_DIVERGENCE" in live_atoms and
        np.allclose(ret,np.asarray([2.0,.5]),equal_nan=False)
    )
    return {
        "status":"PASS" if ok else "FAIL","source_tag":NCAAF_RESEARCH_V2_SOURCE_TAG,
        "families":sorted(fam),"qvalues":q.tolist(),"american_unit_profit_test":ret.tolist(),
        "h2h_price_gate":"OBSERVED_TEAM_AND_OPPONENT_ML_ONLY",
        "confirmation_gate":"BOTH_2024_AND_2025",
        "live_authority_policy":NCAAF_MINER_LIVE_AUTHORITY_POLICY,
        "live_min_confirmation_n":NCAAF_MINER_LIVE_MIN_CONFIRMATION_N,
        "live_min_confirmation_rate":NCAAF_MINER_LIVE_MIN_CONFIRMATION_RATE,
        "weak_confirmed_authority":_miner_live_authority_eligible({"confirmation_pass":True,"confirmation_n":109,"confirmation_rate":0.5229}),
        "strong_confirmed_authority":_miner_live_authority_eligible({"confirmation_pass":True,"confirmation_n":90,"confirmation_rate":0.6222}),
    }


if __name__ == "__main__":
    print(json.dumps(self_test(),indent=2,default=str))
