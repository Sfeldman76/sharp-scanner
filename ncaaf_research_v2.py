"""NCAAF Research V2 — orthogonal residual STAT + disciplined System Miner V3.

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

NCAAF_RESEARCH_V2_SOURCE_TAG = "ncaaf-research-v2-orthogonal-stat-miner-v3-20261003"
NCAAF_RESEARCH_V2_VERSION = "2.0.0"
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
    return 100.0/o if o>0 else 100.0/abs(o)


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
                           features: list[str], market: str, clip: float) -> dict[str,Any]:
    market=str(market).upper(); n=len(g)
    pred=np.full(n,np.nan,dtype=float); fold_rows=[]
    # Discovery and confirmation are evaluated chronologically, but selection is
    # based ONLY on discovery seasons <= 2023.
    for sy in sorted({int(x) for x in seasons[np.isfinite(seasons)] if int(x)<=max(CONFIRMATION_SEASONS)}):
        tr=np.isfinite(seasons)&(seasons<sy)&np.isfinite(base)&np.isfinite(actual)
        va=np.isfinite(seasons)&(seasons==sy)&np.isfinite(base)&np.isfinite(actual)
        if tr.sum()<250 or va.sum()<50: continue
        Xtr=g.loc[tr,features].apply(pd.to_numeric,errors="coerce")
        Xva=g.loc[va,features].apply(pd.to_numeric,errors="coerce")
        # Family must have real coverage, not just imputer-generated constants.
        usable=[c for c in features if Xtr[c].notna().sum()>=100 and Xtr[c].nunique(dropna=True)>=4]
        if len(usable)<2: continue
        target=(actual-base)[tr]
        ok=np.isfinite(target)
        if ok.sum()<200: continue
        try: corr=_fit_ridge_predict(Xtr.loc[ok,usable],target[ok],Xva[usable])
        except Exception: continue
        corr=np.clip(corr,-clip,clip); pred[va]=base[va]+corr
        bm=_mae(actual[va],base[va]); cm=_mae(actual[va],pred[va]); imp=bm-cm
        fold_rows.append({"season":sy,"n":int(va.sum()),"baseline_mae":bm,"corrected_mae":cm,"mae_improvement":imp,
                          "mean_abs_correction":float(np.nanmean(np.abs(corr))) if len(corr) else np.nan})
    disc=[x for x in fold_rows if x["season"]<=DISCOVERY_MAX_SEASON]
    conf=[x for x in fold_rows if x["season"] in CONFIRMATION_SEASONS]
    d_imps=[x["mae_improvement"] for x in disc if np.isfinite(x["mae_improvement"])]
    c_imps=[x["mae_improvement"] for x in conf if np.isfinite(x["mae_improvement"])]
    # Frozen discovery rule. Confirmation never changes this threshold/rule.
    discovery_pass=bool(len(d_imps)>=2 and np.mean(d_imps)>0 and sum(x>0 for x in d_imps)>=2)
    confirmation_pass=bool(discovery_pass and len(c_imps)>=2 and np.mean(c_imps)>0 and sum(x>=0 for x in c_imps)>=1)
    valid=np.isfinite(pred)&np.isfinite(actual)&np.isfinite(base)
    correction=pred-base
    open_line = _num(g,"Consensus_Open_Spread").to_numpy(dtype=float) if market=="SPREADS" else _num(g,"Consensus_Open_Total").to_numpy(dtype=float)
    incumbent_edge=(base+open_line) if market=="SPREADS" else (base-open_line)
    return {
        "market":market,"feature_count":len(features),"features":features,"folds":fold_rows,
        "discovery_rule":"MEAN_MAE_IMPROVEMENT_GT_0_AND_BOTH_DISCOVERY_SEASONS_POSITIVE",
        "discovery_pass":discovery_pass,"confirmation_pass":confirmation_pass,
        "discovery_mean_improvement":float(np.mean(d_imps)) if d_imps else np.nan,
        "confirmation_mean_improvement":float(np.mean(c_imps)) if c_imps else np.nan,
        "all_oof_n":int(valid.sum()),"all_oof_mae_improvement":_mae(actual[valid],base[valid])-_mae(actual[valid],pred[valid]) if valid.any() else np.nan,
        "correction_vs_incumbent_edge_corr":_corr(correction,incumbent_edge),
        "authority_state":"CONFIRMED_SHADOW" if confirmation_pass else ("DISCOVERY_CANDIDATE" if discovery_pass else "RESEARCH"),
        "production_authority":0,
    }


def run_orthogonal_stat_research(games: pd.DataFrame, seasons: np.ndarray, oof_margin: np.ndarray,
                                 oof_total: np.ndarray, candidate_cols: list[str], log_func=print) -> dict[str,Any]:
    families=_classify_feature_families(candidate_cols)
    actual_margin=_num(games,"Actual_Margin").to_numpy(dtype=float)
    actual_total=_num(games,"Actual_Total").to_numpy(dtype=float)
    out={"version":"NCAAF-RV2-ORTHOGONAL-STAT","selection_freeze":DISCOVERY_MAX_SEASON,"confirmation_seasons":list(CONFIRMATION_SEASONS),
         "prospective_min_season":PROSPECTIVE_MIN_SEASON,"production_authority":0,"families":{}}
    for fam,cols in families.items():
        sp=_season_forward_family(games,seasons,oof_margin,actual_margin,cols,"SPREADS",6.0)
        tot=_season_forward_family(games,seasons,oof_total,actual_total,cols,"TOTALS",8.0)
        out["families"][fam]={"SPREADS":sp,"TOTALS":tot}
        log_func(f"[NCAAF-RV2-STAT] family={fam} features={len(cols)} spread_discovery={sp['discovery_pass']} spread_confirm={sp['confirmation_pass']} spread_conf_imp={sp['confirmation_mean_improvement']:.5f} totals_discovery={tot['discovery_pass']} totals_confirm={tot['confirmation_pass']} totals_conf_imp={tot['confirmation_mean_improvement']:.5f} authority=0")
    out["confirmed_spread_families"]=[k for k,v in out["families"].items() if v["SPREADS"]["confirmation_pass"]]
    out["confirmed_totals_families"]=[k for k,v in out["families"].items() if v["TOTALS"]["confirmation_pass"]]
    return out


# ---------------------------------------------------------------------------
# System Miner V3 — fixed discovery / untouched confirmation / dependence collapse
# ---------------------------------------------------------------------------
def _extended_atoms(g: pd.DataFrame, dashboard_module=None) -> list[dict[str,Any]]:
    atoms=[]
    # Retain all existing leakage-safe Miner V2 atoms first.
    try:
        if dashboard_module is not None and hasattr(dashboard_module,"_v1355_system_atoms"):
            for a in dashboard_module._v1355_system_atoms(g):
                atoms.append({"name":str(a["name"]),"family":str(a["family"]),"mask":np.asarray(a["mask"],dtype=bool),"description":str(a.get("description") or a["name"])})
    except Exception:
        pass
    names={a["name"] for a in atoms}
    def add(name,family,mask,desc=None,min_n=30):
        if name in names: return
        mm=pd.Series(mask,index=g.index).fillna(False).astype(bool).to_numpy()
        if min_n<=int(mm.sum())<len(g):
            atoms.append({"name":name,"family":family,"mask":mm,"description":desc or name}); names.add(name)
    n=lambda c:_num(g,c)
    # Market path / sharp-vs-soft / key-cross context from Utils-derived historical features.
    for mins in (30,60,120):
        c=f"Line_Move_{mins}m"; s=n(c)
        if s.notna().sum()>=80:
            add(f"LINE_MOVE_{mins}M_POS","MARKET_PATH",s.ge(0.5)); add(f"LINE_MOVE_{mins}M_NEG","MARKET_PATH",s.le(-0.5))
    s=n("Line_Move_From_Open")
    if s.notna().sum()>=80:
        add("LINE_FROM_OPEN_2_PLUS","MARKET_PATH",s.ge(2)); add("LINE_FROM_OPEN_2_MINUS","MARKET_PATH",s.le(-2))
    s=n("Direction_Changes_Count")
    if s.notna().sum()>=80: add("MARKET_REVERSAL_2_PLUS","MARKET_PATH",s.ge(2))
    s=n("Sharp_Book_Move_60m")
    if s.notna().sum()>=80:
        add("SHARP_MOVE_60_POS","SHARP_SOFT",s.ge(0.5)); add("SHARP_MOVE_60_NEG","SHARP_SOFT",s.le(-0.5))
    s=n("Sharp_Soft_Divergence")
    if s.notna().sum()>=80:
        add("SHARP_SOFT_DIV_POS","SHARP_SOFT",s.ge(0.5)); add("SHARP_SOFT_DIV_NEG","SHARP_SOFT",s.le(-0.5))
    s=n("Sharp_Consensus_Direction")
    if s.notna().sum()>=80:
        add("SHARP_CONSENSUS_POS","SHARP_SOFT",s.ge(0.5)); add("SHARP_CONSENSUS_NEG","SHARP_SOFT",s.le(-0.5))
    for key in (3,7,10,14):
        c=f"Crossed_Key_{key}_Last60m"; s=n(c)
        if s.notna().sum()>=80: add(f"KEY_{key}_CROSSED_60M","KEY_NUMBER",s.eq(1))
    for c,name in [("Key_Cross_Confirmed_By_Sharp_Books","KEY_CROSS_SHARP_CONFIRMED"),("Key_Cross_Reversed","KEY_CROSS_REVERSED"),("Key_Cross_Persistence","KEY_CROSS_PERSISTENT")]:
        s=n(c)
        if s.notna().sum()>=80: add(name,"KEY_NUMBER",s.ge(1))
    # Deeper H2H / revenge state.
    for c,thr,name in [("Revenge_Depth",2,"REVENGE_DEPTH_2_PLUS"),("H2H_Meetings_Since_Last_Win",3,"H2H_3_PLUS_SINCE_WIN")]:
        s=n(c)
        if s.notna().sum()>=80: add(name,"MATCHUP_HISTORY",s.ge(thr))
    s=n("H2H_Last_Loss_Margin")
    if s.notna().sum()>=80:
        add("H2H_LAST_LOSS_14_PLUS","MATCHUP_HISTORY",s.le(-14)); add("H2H_LAST_LOSS_7_PLUS","MATCHUP_HISTORY",s.le(-7))
    # Prior ATS shape, beyond simple streak counts.
    for c,thr,name,op in [
        ("Prev_ATS_Margin",-7,"OFF_ATS_LOSS_7","le"),("Prev_ATS_Margin",7,"OFF_ATS_WIN_7","ge"),
        ("ATS_Margin_Mean_Last3",-3,"ATS_LAST3_BAD","le"),("ATS_Margin_Mean_Last3",3,"ATS_LAST3_GOOD","ge")]:
        s=n(c)
        if s.notna().sum()>=80: add(name,"ATS_FORM",getattr(s,op)(thr))
    # Role change / historical team price context.
    for c,name in [("Pathi_FB_Usually_Dog_Now_Favorite","ROLE_FLIP_DOG_TO_FAV"),("Pathi_FB_Usually_Favorite_Now_Dog","ROLE_FLIP_FAV_TO_DOG")]:
        s=n(c)
        if s.notna().sum()>=80: add(name,"ROLE_HISTORY",s.eq(1))
    s=n("Role_Price_Shift")
    if s.notna().sum()>=80:
        add("ROLE_PRICE_SHIFT_POS","ROLE_HISTORY",s.ge(2)); add("ROLE_PRICE_SHIFT_NEG","ROLE_HISTORY",s.le(-2))
    s=n("ML_Price_ZScore_vs_TeamHistory")
    if s.notna().sum()>=80:
        add("ML_PRICE_Z_1P5_HIGH","ROLE_HISTORY",s.ge(1.5)); add("ML_PRICE_Z_1P5_LOW","ROLE_HISTORY",s.le(-1.5))
    # Schedule sequencing / rest extremes.
    s=n("Days_Since_Last_Game_System").where(n("Days_Since_Last_Game_System").notna(),n("Days_Since_Last_Game"))
    if s.notna().sum()>=80:
        add("REST_5_OR_LESS","SCHEDULE_SEQUENCE",s.le(5)); add("REST_10_PLUS","SCHEDULE_SEQUENCE",s.ge(10))
    # FBS/FCS or conference-class context if available.
    for c in ("Opponent_Subdivision","Opp_Subdivision","Context_Opp_Subdivision"):
        if c in g.columns:
            t=_txt(g,c); add("VS_FCS","OPPONENT_CLASS",t.str.contains("FCS",na=False),min_n=20); break
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


def _evaluate_rule(mask: np.ndarray, y: np.ndarray, valid: np.ndarray, baseline: np.ndarray, seasons: np.ndarray,
                   names: tuple[str,...], families: tuple[str,...], market: str) -> dict[str,Any] | None:
    disc=mask&valid&np.isfinite(seasons)&(seasons<=DISCOVERY_MAX_SEASON)
    ix=np.flatnonzero(disc)
    team_specific="TEAM_SPECIFIC" in families; min_n=60 if team_specific else 100
    if len(ix)<min_n: return None
    raw=float(np.mean(y[ix])); direction="PLAY_ON" if raw>=.5 else "FADE"; obs=y if direction=="PLAY_ON" else 1-y
    rate=float(np.mean(obs[ix]))
    dyears=[]
    for sy in sorted({int(x) for x in seasons[ix]}):
        jj=disc&(seasons==sy)
        if jj.sum()>=15: dyears.append({"season":sy,"n":int(jj.sum()),"rate":float(np.mean(obs[jj]))})
    if len(dyears)<2: return None
    stable=float(np.mean([x["rate"]>.5 for x in dyears]))
    # Remove best discovery season robustness.
    best=max(dyears,key=lambda x:x["rate"])["season"]
    rem=disc&(seasons!=best); remove_best=float(np.mean(obs[rem])) if rem.sum()>=25 else np.nan
    conf=[]
    for sy in CONFIRMATION_SEASONS:
        jj=mask&valid&np.isfinite(seasons)&(seasons==sy)
        if jj.sum(): conf.append({"season":sy,"n":int(jj.sum()),"rate":float(np.mean(obs[jj]))})
    conf_n=sum(x["n"] for x in conf); conf_rate=float(np.average([x["rate"] for x in conf],weights=[x["n"] for x in conf])) if conf_n else np.nan
    # H2H must also show probability residual vs de-vigged market in discovery and confirmation.
    market_resid_disc=market_resid_conf=np.nan
    if market=="h2h":
        b=baseline
        ok=disc&np.isfinite(b)
        if ok.sum()>=min_n:
            bb=b if direction=="PLAY_ON" else 1-b; market_resid_disc=float(np.nanmean(obs[ok]-bb[ok]))
        okc=mask&valid&np.isin(seasons,np.asarray(CONFIRMATION_SEASONS,dtype=float))&np.isfinite(b)
        if okc.sum()>=20:
            bb=b if direction=="PLAY_ON" else 1-b; market_resid_conf=float(np.nanmean(obs[okc]-bb[okc]))
    return {"conditions":list(names),"families":list(families),"direction":direction,"discovery_n":len(ix),"discovery_rate":rate,
            "discovery_seasons":dyears,"stable_discovery_fraction":stable,"remove_best_discovery_rate":remove_best,
            "nominal_pvalue":_one_sided_p(rate,len(ix)),"confirmation":conf,"confirmation_n":conf_n,"confirmation_rate":conf_rate,
            "market_residual_discovery":market_resid_disc,"market_residual_confirmation":market_resid_conf,"mask":mask}


def run_system_miner_v3(games: pd.DataFrame, seasons: np.ndarray, market: str, dashboard_module=None,
                        log_func=print, max_depth: int=4) -> dict[str,Any]:
    market=str(market).lower(); y,valid,baseline=_market_target(games,market); atoms=_extended_atoms(games,dashboard_module)
    out={"version":"NCAAF-RV2-SYSTEM-MINER-V3","market":market,"production_authority":0,"discovery_max_season":DISCOVERY_MAX_SEASON,
         "confirmation_seasons":list(CONFIRMATION_SEASONS),"prospective_min_season":PROSPECTIVE_MIN_SEASON,"atoms":len(atoms),"systems":[],"mechanism_families":[]}
    if valid.sum()<500: out["status"]="INSUFFICIENT_HISTORY"; return out
    tested=[]; beam=[]; seen=set()
    def ev(mask,names,fams,idx):
        r=_evaluate_rule(mask,y,valid,baseline,seasons,names,fams,market)
        if r is None: return None
        # Discovery qualification only; confirmation cannot rescue a failed discovery rule.
        discovery_pass=bool(r["discovery_rate"]>=.54 and r["stable_discovery_fraction"]>=1.0 and np.isfinite(r["remove_best_discovery_rate"]) and r["remove_best_discovery_rate"]>=.515)
        r["discovery_pass"]=discovery_pass; r["idx"]=tuple(idx)
        r["quality"]=(r["discovery_rate"]-.5)*3 + max(0,r["remove_best_discovery_rate"]-.5) + .03*r["stable_discovery_fraction"]
        return r
    for i,a in enumerate(atoms):
        r=ev(a["mask"],(a["name"],),(a["family"],),(i,))
        if r: tested.append(r); beam.append(r)
    beam=sorted(beam,key=lambda z:z["quality"],reverse=True)[:48]
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
        beam=sorted(nxt,key=lambda z:z["quality"],reverse=True)[:48]
        if not beam: break
    if not tested: out.update({"status":"NO_CANDIDATES","tested_hypotheses":0}); return out
    q=_bh_qvalues([z["nominal_pvalue"] for z in tested])
    for z,qq in zip(tested,q): z["fdr_qvalue"]=float(qq) if np.isfinite(qq) else np.nan
    finalists=[]
    for z in sorted(tested,key=lambda r:(r["discovery_pass"],r["quality"],r["discovery_n"]),reverse=True):
        if not z["discovery_pass"] or not np.isfinite(z["fdr_qvalue"]) or z["fdr_qvalue"]>.20: continue
        conf_ok=bool(z["confirmation_n"]>=30 and np.isfinite(z["confirmation_rate"]) and z["confirmation_rate"]>=.50 and
                     sum(x["rate"]>=.50 for x in z["confirmation"] if x["n"]>=10)>=1)
        if market=="h2h":
            conf_ok=bool(conf_ok and np.isfinite(z["market_residual_discovery"]) and z["market_residual_discovery"]>=.015 and
                         np.isfinite(z["market_residual_confirmation"]) and z["market_residual_confirmation"]>=0)
        item={k:v for k,v in z.items() if k not in {"mask","idx","quality"}}
        item.update({"system_id":_stable_id("NCAAF-RV2-"+market.upper()+"-",z["conditions"]),
                     "authority_state":"CONFIRMED_SHADOW" if conf_ok else "DISCOVERY_FROZEN",
                     "confirmation_pass":conf_ok,"production_authority":0,
                     "admission_note":"2024-2025 confirmation cannot change discovery direction or thresholds"})
        item["_mask_internal"]=z["mask"]
        finalists.append(item)
        if len(finalists)>=120: break
    # Dependence collapse: systems with highly overlapping game masks and the same direction
    # are one mechanism, regardless of how many syntactic variants the miner found.
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
        rep=max(members,key=lambda z:(z["confirmation_pass"],z["confirmation_rate"] if np.isfinite(z["confirmation_rate"]) else -1,z["discovery_rate"],z["discovery_n"]))
        fid=_stable_id("NCAAF-MECH-",[market,rep["direction"],"+".join(sorted(set(rep["families"])))]+[x["system_id"] for x in members])
        families.append({"mechanism_id":fid,"market":market,"direction":rep["direction"],"representative_system_id":rep["system_id"],
                         "member_system_ids":[x["system_id"] for x in members],"member_count":len(members),"families":rep["families"],
                         "discovery_rate":rep["discovery_rate"],"discovery_n":rep["discovery_n"],"confirmation_rate":rep["confirmation_rate"],
                         "confirmation_n":rep["confirmation_n"],"confirmation_pass":bool(rep["confirmation_pass"]),
                         "authority_state":"CONFIRMED_SHADOW" if rep["confirmation_pass"] else "DISCOVERY_FROZEN","production_authority":0})
    clean=[]
    for z in finalists:
        z=dict(z); z.pop("_mask_internal",None); clean.append(z)
    out.update({"status":"RESEARCH_COMPLETE","tested_hypotheses":len(tested),"systems":clean,"published_systems":len(clean),
                "mechanism_families":families,"mechanism_family_count":len(families),
                "confirmed_mechanism_count":sum(x["confirmation_pass"] for x in families),
                "admission_contract":"DISCOVERY_2022_2023_ONLY__FDR__BOTH_DISCOVERY_SEASONS_POSITIVE__REMOVE_BEST__DIRECTION_FROZEN__2024_2025_CONFIRMATION_ONLY__DEPENDENCY_COLLAPSE__2026_PROSPECTIVE_ONLY__ZERO_AUTHORITY"})
    log_func(f"[NCAAF-RV2-MINER] market={market} atoms={len(atoms)} tested={len(tested)} systems={len(clean)} mechanisms={len(families)} confirmed_mechanisms={out['confirmed_mechanism_count']} authority=0")
    for x in families[:20]:
        log_func(f"[NCAAF-RV2-MECHANISM] market={market} id={x['mechanism_id']} status={x['authority_state']} members={x['member_count']} discovery={x['discovery_rate']:.4f}/{x['discovery_n']} confirmation={x['confirmation_rate']:.4f}/{x['confirmation_n']} rep={x['representative_system_id']}")
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
        log_func(f"[NCAAF-RV2-PREFLIGHT] source={NCAAF_RESEARCH_V2_SOURCE_TAG} rows={len(g)} seasons={sorted(set(sy.astype(int)))} discovery<=2023 confirmation=2024,2025 prospective>=2026 production_authority=0")
        stat=run_orthogonal_stat_research(g,sy,om,ot,cols,log_func=log_func)
        miners={m:run_system_miner_v3(mg,sy,m,dashboard_module=dashboard_module,log_func=log_func,max_depth=4) for m in ("spreads","h2h","totals")}
        market_audit=_market_rich_audit(g,utils_module)
        report={"source_tag":NCAAF_RESEARCH_V2_SOURCE_TAG,"version":NCAAF_RESEARCH_V2_VERSION,"created_utc":_now(),
                "status":"NCAAF_RESEARCH_V2_COMPLETE","production_authority":0,"production_contract_mutated":False,
                "benchmark":"FROZEN_NCAAF_PRODUCTION_V1","discovery_max_season":DISCOVERY_MAX_SEASON,"confirmation_seasons":list(CONFIRMATION_SEASONS),"prospective_min_season":PROSPECTIVE_MIN_SEASON,
                "rows":len(g),"seasons":sorted(set(sy.astype(int))),"orthogonal_stat":stat,"system_miner_v3":miners,"market_rich":market_audit,
                "next_step":"FREEZE REPORT AND RUN 2026 PROSPECTIVE SHADOW; NO AUTOMATIC PRODUCTION PROMOTION"}
        # Preserve a lightweight pickle bundle for future prospective trigger/scoring adapters.
        bundle={"report":report,"system_miner_v3":miners,"stat_family_definitions":STAT_FAMILY_TOKENS,"source_tag":NCAAF_RESEARCH_V2_SOURCE_TAG}
        if storage_client is None:
            from google.cloud import storage
            storage_client=storage.Client()
        b=storage_client.bucket(bucket_name)
        body=json.dumps(_report_without_models(report),sort_keys=True,separators=(",",":"),default=str).encode()
        sha=hashlib.sha256(body).hexdigest(); hist=f"{REPORT_HISTORY_PREFIX}/{sha[:16]}/report.json"
        b.blob(hist).upload_from_string(body,content_type="application/json"); b.blob(REPORT_CURRENT_BLOB).upload_from_string(body,content_type="application/json")
        bio=io.BytesIO(); pickle.dump(bundle,bio,protocol=pickle.HIGHEST_PROTOCOL); bio.seek(0); pdata=bio.read(); b.blob(BUNDLE_CURRENT_BLOB).upload_from_string(pdata,content_type="application/octet-stream")
        report["artifact"]={"current_report":f"gs://{bucket_name}/{REPORT_CURRENT_BLOB}","current_bundle":f"gs://{bucket_name}/{BUNDLE_CURRENT_BLOB}","history_report":f"gs://{bucket_name}/{hist}","sha256":sha}
        log_func(f"[NCAAF-RV2-CONTRACT] status=PASS report=gs://{bucket_name}/{REPORT_CURRENT_BLOB} sha={sha[:16]} stat_spread_confirmed={len(stat['confirmed_spread_families'])} stat_totals_confirmed={len(stat['confirmed_totals_families'])} miner_confirmed={sum(v.get('confirmed_mechanism_count',0) for v in miners.values())} production_authority=0")
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
    """Attach research-only V2 system triggers to live rows. Never changes action/probability."""
    if rows is None or rows.empty or not isinstance(report,dict): return rows
    out=rows.copy(); atoms={a["name"]:np.asarray(a["mask"],dtype=bool) for a in _extended_atoms(out,dashboard_module)}
    summaries=[[] for _ in range(len(out))]
    market_col=out.get("Market",pd.Series("",index=out.index)).astype(str).str.lower()
    reg=report.get("system_miner_v3") or {}
    for m,mr in reg.items():
        for sys in (mr or {}).get("systems",[]):
            cond=list(sys.get("conditions") or []); mask=np.ones(len(out),dtype=bool)&market_col.eq(str(m).lower()).to_numpy()
            for c in cond:
                if c not in atoms: mask[:]=False; break
                mask &= atoms[c]
            for j in np.flatnonzero(mask): summaries[j].append(f"{sys.get('system_id')} [{sys.get('authority_state','RESEARCH')}]" )
    out["NCAAF_RV2_System_Summary"]=[" | ".join(x) if x else "—" for x in summaries]
    out["NCAAF_RV2_System_Count"]=[len(x) for x in summaries]
    return out


def self_test() -> dict[str,Any]:
    # Contract-only test: confirms IDs, family classifier, FDR and JSON safety.
    q=_bh_qvalues([.01,.04,.20]); fam=_classify_feature_families(["Rush_EPA","Opp_Rush_EPA","Line_Move_60m","Sharp_Soft_Divergence","Actual_Margin"])
    ok=bool(len(q)==3 and "RUN_PASS_MATCHUP" in fam and "MARKET_MICROSTRUCTURE" in fam and all("Actual_Margin" not in x for v in fam.values() for x in v))
    return {"status":"PASS" if ok else "FAIL","source_tag":NCAAF_RESEARCH_V2_SOURCE_TAG,"families":sorted(fam),"qvalues":q.tolist()}


if __name__ == "__main__":
    print(json.dumps(self_test(),indent=2,default=str))
