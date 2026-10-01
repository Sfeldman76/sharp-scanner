"""NFL Research V2.0 System Lab.

This module is the dedicated situational/system brain.  It deliberately does
NOT use CORE predictions, residual-model outputs, edge-gate probabilities, or
live market-microstructure features during system discovery.  That separation
keeps SYSTEMS genuinely independent from FUNDAMENTAL and MARKET brains.

The lab reuses the research discipline developed in the NCAAF System Miner V3:
logical/domain atoms, bounded beam search, one-side-per-game enforcement,
near-duplicate mask collapse, chronological/LOSO/bootstrap robustness,
remove-best-season and drop-one-condition tests, multiple-testing control, and
frozen later-period validation.

Historical design:
  DISCOVERY       2017-2022  (direction + rule selection only)
  SHADOW          2023       (no tuning)
  CONFIRMATION    2024       (no tuning)
  FINAL CHECK     2025       (untouched by selection)
  SEALED          2026       (never queried)

No discovered system has production authority.  Historical opening/closing
lines do not establish executable historical ROI because timestamped prices and
juice are not available at identical canonical snapshots.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from nfl_feature_audit_v1 import VIEW
from nfl_intelligence_v1 import (
    build_intelligence_query, _prepare_side_state, _bigal_systems, _pathi_engineering,
    _swap_within_game,
)
from nfl_research_v2_contract import assert_contract, contract_hash

SOURCE_TAG = "nfl-system-lab-v1-research-v2.0-ncaaf-miner-v3-methodology-20261001"
PRODUCTION_AUTHORITY = 0
DISCOVERY_SEASONS = (2017,2018,2019,2020,2021,2022)
SHADOW_SEASON = 2023
CONFIRM_SEASON = 2024
FINAL_CHECK_SEASON = 2025
SEALED_SEASON = 2026
BREAK_EVEN_REFERENCE = 0.52381
MAX_DEPTH = 6
BEAM_WIDTH = 24
FINALISTS = 40
JACCARD_CUTOFF = 0.98
MAXSTAT_REPS = 500
BOOTSTRAP_REPS = 800
STATUS = "NFL_RESEARCH_V2_SYSTEM_LAB_FROZEN_FOR_PROSPECTIVE_RETEST"

ACADEMIC_REGISTRY = {
    "ACADEMIC_DIVISION_HOME_ATS": {
        "status":"REPLICATION_HYPOTHESIS",
        "source":"Shank (2019), NFL betting-market efficiency/divisional-rivals study",
        "hypothesis":"Divisional familiarity may reduce home-team ATS performance relative to non-division games.",
        "authority":0,
    },
    "ACADEMIC_DIVISION_TOTAL_UNDER": {
        "status":"REPLICATION_HYPOTHESIS",
        "source":"Shank (2019), NFL betting-market efficiency/divisional-rivals study",
        "hypothesis":"Divisional familiarity may lower OVER probability / favor UNDER relative to non-division games.",
        "authority":0,
    },
    "ACADEMIC_REST_ERA_CHANGE": {
        "status":"REPLICATION_HYPOTHESIS",
        "source":"Lopez/Bliss (2024) NFL rest-differential research",
        "hypothesis":"Traditional rest/bye advantage may be weaker in the modern era; treat rest as an era-sensitive hypothesis rather than folklore.",
        "authority":0,
    },
}


def _num(d,c):
    if c not in d: return pd.Series(np.nan,index=d.index,dtype=float)
    return pd.to_numeric(d[c],errors="coerce")


def _status(v):
    if not np.isfinite(v): return "MISSING"
    if v>1e-9: return "WIN"
    if v<-1e-9: return "LOSS"
    return "PUSH"


def prepare_system_state(side_rows: pd.DataFrame) -> pd.DataFrame:
    """Add opening-line settlement and additional prior-only system state."""
    d=_prepare_side_state(side_rows)
    if d.Season.max()>2025: raise RuntimeError("NFL_SYSTEM_LAB_2026_DATA_LEAK")
    op=_num(d,"Opening_Spread"); ot=_num(d,"Opening_Total")
    d["Opening_ATS_Status"]=[_status(m+s) for m,s in zip(_num(d,"actual_margin"),op)]
    d["Opening_ATS_Win"]=d.Opening_ATS_Status.eq("WIN").astype(float)
    d["Opening_ATS_Loss"]=d.Opening_ATS_Status.eq("LOSS").astype(float)
    total_points=_num(d,"Team_Score")+_num(d,"Opponent_Score")
    d["Opening_Total_Status"]=["OVER" if np.isfinite(t) and np.isfinite(o) and t-o>1e-9 else "UNDER" if np.isfinite(t) and np.isfinite(o) and t-o<-1e-9 else "PUSH" if np.isfinite(t) and np.isfinite(o) else "MISSING" for t,o in zip(total_points,ot)]
    d["Opening_Dog"]=op.gt(0).astype(float); d["Opening_Favorite"]=op.lt(0).astype(float)
    d["Opening_Spread_Abs"]=op.abs()
    d=d.sort_values(["Season","Team_Norm","Game_Date","Source_Name","Source_Game_ID"],kind="mergesort").copy()
    grp=d.groupby(["Season","Team_Norm"],sort=False,dropna=False)
    d["calc_team_game_number"]=grp.cumcount()+1
    d["calc_prev1_opening_spread"]=grp["Opening_Spread"].shift(1)
    d["calc_prev1_open_dog"]=grp["Opening_Dog"].shift(1)
    d["calc_prev1_open_favorite"]=grp["Opening_Favorite"].shift(1)
    d["calc_prev1_home"]=grp["Is_Home"].shift(1)
    d["calc_prev1_away"]=grp["Is_Away"].shift(1)
    prior_games=grp.cumcount().astype(float)
    dog_before=grp["Opening_Dog"].cumsum()-d["Opening_Dog"]
    home_before=grp["Is_Home"].cumsum()-pd.to_numeric(d["Is_Home"],errors="coerce").fillna(0)
    away_before=grp["Is_Away"].cumsum()-pd.to_numeric(d["Is_Away"],errors="coerce").fillna(0)
    d["calc_open_dog_rate_prior"]=np.where(prior_games.gt(0),dog_before/prior_games,np.nan)
    d["calc_first_home_game"]=(pd.to_numeric(d["Is_Home"],errors="coerce").eq(1)&home_before.eq(0)).astype(float)
    d["calc_first_road_game"]=(pd.to_numeric(d["Is_Away"],errors="coerce").eq(1)&away_before.eq(0)).astype(float)
    d["calc_prev1_margin"]=grp["actual_margin"].shift(1)
    d["calc_prev2_margin"]=grp["actual_margin"].shift(2)
    d["calc_prev1_open_ats_loss"]=grp["Opening_ATS_Loss"].shift(1)
    d["calc_prev2_open_ats_loss"]=grp["Opening_ATS_Loss"].shift(2)
    d["calc_prev1_open_ats_win"]=grp["Opening_ATS_Win"].shift(1)
    # Prior game total is computed explicitly below; avoid groupby.apply so the
    # module stays portable across the pandas versions used by Cloud Run images.
    d["calc_total_points_current"]=_num(d,"Team_Score")+_num(d,"Opponent_Score")
    d["calc_prev1_total_points"]=d.groupby(["Season","Team_Norm"],sort=False)["calc_total_points_current"].shift(1)
    d["calc_prev2_total_points"]=d.groupby(["Season","Team_Norm"],sort=False)["calc_total_points_current"].shift(2)
    d["calc_b2b_open_ats_losses"]=(d["calc_prev1_open_ats_loss"].eq(1)&d["calc_prev2_open_ats_loss"].eq(1)).astype(float)
    for c in ("calc_prev1_margin","calc_prev2_margin","calc_prev1_open_ats_loss","calc_prev2_open_ats_loss","calc_prev1_open_ats_win","calc_prev1_total_points","calc_prev2_total_points","calc_team_game_number","calc_prev1_opening_spread","calc_prev1_open_dog","calc_prev1_open_favorite","calc_open_dog_rate_prior","calc_first_home_game","calc_first_road_game"):
        d["Opp_"+c]=_swap_within_game(pd.to_numeric(d[c],errors="coerce").astype("float64").to_numpy(),d)
    return d.sort_values(["Season","Game_Date","Source_Name","Source_Game_ID","Team_Norm"],kind="mergesort").reset_index(drop=True)


def _wilson(w,n,z=1.959963984540054):
    if n<=0:return [None,None]
    p=w/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den; half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [max(0.0,ctr-half),min(1.0,ctr+half)]


def _rec(labels: np.ndarray, idx: np.ndarray, direction: str) -> dict:
    idx=np.asarray(idx,dtype=int)
    if len(idx)==0:return {"n":0,"wins":0,"rate":None,"wilson95":[None,None]}
    y=np.asarray(labels,float)[idx]; m=np.isfinite(y); y=y[m]
    if not len(y):return {"n":0,"wins":0,"rate":None,"wilson95":[None,None]}
    z=y if direction in ("PLAY_ON","OVER") else 1-y
    w=int(z.sum()); n=int(len(z)); ci=_wilson(w,n)
    return {"n":n,"wins":w,"rate":round(w/n,6),"wilson95":[round(ci[0],6),round(ci[1],6)]}


def _candidate_rows_side(state: pd.DataFrame,mask: np.ndarray) -> tuple[np.ndarray,int]:
    mm=np.asarray(mask,bool); tmp=pd.DataFrame({"i":np.arange(len(state)),"game":state.physical_game_id.astype(str),"m":mm.astype(int)})
    cnt=tmp.groupby("game",sort=False)["m"].transform("sum").to_numpy()
    return np.where(mm&(cnt==1))[0],int(np.sum(mm&(cnt>1)))


def _home_or_neutral_one_side(state: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for _,p in state.groupby("physical_game_id",sort=False):
        home=p.loc[_num(p,"Is_Home").eq(1)]
        if len(home)==1: rows.append(home.iloc[0])
        else: rows.append(p.sort_values("Team_Norm",kind="mergesort").iloc[0])
    out=pd.DataFrame(rows).reset_index(drop=True)
    if out.physical_game_id.duplicated().any():raise RuntimeError("NFL_SYSTEM_LAB_DUPLICATE_GAME_ORIENTATION")
    return out


def _spread_atoms(d:pd.DataFrame)->list[dict]:
    op=_num(d,"Opening_Spread"); oa=op.abs(); wk=_num(d,"Week_Number"); game_no=_num(d,"calc_team_game_number"); rest=_num(d,"Rest_Differential_Days")
    atoms=[]
    def add(name,family,mask,desc=None,boundary=None,team_specific=False):
        mm=pd.Series(mask,index=d.index).fillna(False).to_numpy(bool)
        disc=np.isin(_num(d,"Season").to_numpy(),DISCOVERY_SEASONS); n=int((mm&disc).sum())
        minimum=60 if team_specific else 28
        if n>=minimum and n<int(disc.sum()):atoms.append({"name":name,"family":family,"mask":mm,"description":desc or name,"boundary":boundary,"team_specific":bool(team_specific)})
    # Market role/price: required to define the wager, but no line movement or model state.
    add("HOME","LOCATION",_num(d,"Is_Home").eq(1)); add("ROAD","LOCATION",_num(d,"Is_Away").eq(1));
    add("OPEN_DOG","ROLE",op.gt(0)); add("OPEN_FAVORITE","ROLE",op.lt(0)); add("HOME_DOG","ROLE_COMPOSITE",_num(d,"Is_Home").eq(1)&op.gt(0)); add("ROAD_DOG","ROLE_COMPOSITE",_num(d,"Is_Away").eq(1)&op.gt(0));
    for lo,hi in ((0,3),(3,7),(7,10),(10,14),(14,99)):
        add(f"DOG_{lo}_{hi}","PRICE_BAND",op.gt(lo)&op.le(hi),boundary=("dog",lo,hi)); add(f"FAV_{lo}_{hi}","PRICE_BAND",(-op).gt(lo)&(-op).le(hi),boundary=("fav",lo,hi))
    add("DOG_10_PLUS","PRICE_EXTREME",op.ge(10),boundary=("dog_ge",10)); add("FAVORITE_10_PLUS","PRICE_EXTREME",op.le(-10),boundary=("fav_ge",10))
    add("WEEK_1","SEASON_TIMING",wk.eq(1)); add("WEEK_2","SEASON_TIMING",wk.eq(2)); add("EARLY_WK1_4","SEASON_TIMING",wk.between(1,4)); add("LATE_WK11_PLUS","SEASON_TIMING",wk.ge(11)); add("GAME_9_PLUS","SEASON_TIMING",game_no.ge(9))
    add("FIRST_HOME_GAME","SCHEDULE_POSITION",_num(d,"calc_first_home_game").eq(1)); add("FIRST_ROAD_GAME","SCHEDULE_POSITION",_num(d,"calc_first_road_game").eq(1))
    add("POSTSEASON","REGIME",d.Season_Stage.astype(str).eq("POSTSEASON")); add("DIVISION_GAME","MATCHUP",_num(d,"Is_Division_Game").eq(1)); add("CONFERENCE_GAME","MATCHUP",_num(d,"Is_Conference_Game").eq(1)); add("REVENGE","MATCHUP",_num(d,"Revenge_Flag_Current").eq(1)); add("REMATCH_730D","MATCHUP",_num(d,"Days_Since_Last_Matchup_System").between(1,730))
    add("REST_ADV_2_PLUS","REST",rest.ge(2),boundary=("rest_adv",2)); add("REST_DISADV_2_PLUS","REST",rest.le(-2),boundary=("rest_disadv",2)); add("THURSDAY_SHORT_WEEK","REST",_num(d,"Is_Thursday_Short_Week").eq(1)); add("BYE_LIKE_REST","REST",_num(d,"Is_Bye_Like_Rest").eq(1))
    pm=_num(d,"calc_prev1_margin"); add("OFF_SU_WIN","PRIOR_SU",pm.gt(0)); add("OFF_SU_LOSS","PRIOR_SU",pm.lt(0)); add("OFF_BLOWOUT_WIN_14_PLUS","PRIOR_SU",pm.ge(14),boundary=("prev_margin_ge",14)); add("OFF_BLOWOUT_LOSS_14_PLUS","PRIOR_SU",pm.le(-14),boundary=("prev_margin_le",-14)); add("OFF_BLOWOUT_LOSS_21_PLUS","PRIOR_SU",pm.le(-21),boundary=("prev_margin_le",-21))
    add("OFF_UPSET_WIN","PRIOR_ROLE_RESULT",pm.gt(0)&_num(d,"calc_prev1_open_dog").eq(1)); add("OFF_UPSET_LOSS_AS_FAVORITE","PRIOR_ROLE_RESULT",pm.lt(0)&_num(d,"calc_prev1_open_favorite").eq(1)); add("OPP_OFF_UPSET_WIN","OPP_ROLE_RESULT",_num(d,"Opp_calc_prev1_margin").gt(0)&_num(d,"Opp_calc_prev1_open_dog").eq(1)); add("OPP_OFF_UPSET_LOSS_AS_FAVORITE","OPP_ROLE_RESULT",_num(d,"Opp_calc_prev1_margin").lt(0)&_num(d,"Opp_calc_prev1_open_favorite").eq(1))
    add("OFF_ATS_WIN","PRIOR_ATS",_num(d,"calc_prev1_open_ats_win").eq(1)); add("OFF_ATS_LOSS","PRIOR_ATS",_num(d,"calc_prev1_open_ats_loss").eq(1)); add("BACK_TO_BACK_ATS_LOSSES","PRIOR_ATS",_num(d,"calc_b2b_open_ats_losses").eq(1))
    add("OFF_SU_AND_ATS_WIN","PRIOR_COMBO",pm.gt(0)&_num(d,"calc_prev1_open_ats_win").eq(1)); add("OFF_SU_AND_ATS_LOSS","PRIOR_COMBO",pm.lt(0)&_num(d,"calc_prev1_open_ats_loss").eq(1))
    add("ROLE_FLIP_DOG_TO_FAVORITE","ROLE_CHANGE",_num(d,"calc_prev1_open_dog").eq(1)&op.lt(0)); add("ROLE_FLIP_FAVORITE_TO_DOG","ROLE_CHANGE",_num(d,"calc_prev1_open_favorite").eq(1)&op.gt(0))
    wp=_num(d,"calc_win_pct_prior"); owp=_num(d,"Opp_calc_win_pct_prior"); add("TEAM_WINPCT_LE_400","TEAM_STATE",wp.le(.4)&wp.notna()); add("TEAM_WINPCT_GE_600","TEAM_STATE",wp.ge(.6)); add("OPP_WINPCT_LE_500","OPP_STATE",owp.le(.5)&owp.notna()); add("OPP_WINPCT_GE_600","OPP_STATE",owp.ge(.6))
    add("NEGATIVE_LAST5_MARGIN","FORM",_num(d,"Avg_SU_Margin_Last5_Prior").lt(0)); add("POSITIVE_LAST5_MARGIN","FORM",_num(d,"Avg_SU_Margin_Last5_Prior").gt(0)); add("TURNOVER_NEG_LAST3","FORM",_num(d,"Avg_Turnover_Margin_Last3_Prior").lt(-.5)); add("TURNOVER_POS_LAST3","FORM",_num(d,"Avg_Turnover_Margin_Last3_Prior").gt(.5))
    # Team-specific hypotheses are allowed only as a separate high-N family and
    # receive a stricter discovery sample requirement. They never bypass the
    # later shadow/confirmation/final and multiple-testing gates.
    teams=d.Team_Norm.astype(str).fillna("");
    for tm,cnt in teams[np.isin(_num(d,"Season"),DISCOVERY_SEASONS)].value_counts().items():
        if tm and cnt>=60:
            label=re.sub(r"[^A-Za-z0-9]+","_",tm).strip("_").upper()[:40]; add("TEAM_"+label,"TEAM_IDENTITY",teams.eq(tm),team_specific=True)
    return atoms


def _total_atoms(g:pd.DataFrame)->list[dict]:
    ot=_num(g,"Opening_Total"); wk=_num(g,"Week_Number"); rest=_num(g,"Rest_Differential_Days")
    atoms=[]
    def add(name,family,mask,desc=None,boundary=None):
        mm=pd.Series(mask,index=g.index).fillna(False).to_numpy(bool); disc=np.isin(_num(g,"Season").to_numpy(),DISCOVERY_SEASONS); n=int((mm&disc).sum())
        if n>=28 and n<int(disc.sum()):atoms.append({"name":name,"family":family,"mask":mm,"description":desc or name,"boundary":boundary,"team_specific":False})
    for lo,hi in ((0,41.5),(41.5,45.5),(45.5,49.5),(49.5,99)):
        add(f"OPEN_TOTAL_{str(lo).replace('.','P')}_{str(hi).replace('.','P')}","TOTAL_BAND",ot.ge(lo)&ot.lt(hi),boundary=("total",lo,hi))
    add("DIVISION_GAME","MATCHUP",_num(g,"Is_Division_Game").eq(1)); add("CONFERENCE_GAME","MATCHUP",_num(g,"Is_Conference_Game").eq(1)); add("REVENGE","MATCHUP",_num(g,"Revenge_Flag_Current").eq(1)); add("REMATCH_730D","MATCHUP",_num(g,"Days_Since_Last_Matchup_System").between(1,730))
    add("PRIMETIME","SCHEDULE",_num(g,"Is_PrimeTime").eq(1)); add("THURSDAY","SCHEDULE",_num(g,"Is_Thursday_Game").eq(1)); add("POSTSEASON","REGIME",g.Season_Stage.astype(str).eq("POSTSEASON")); add("EARLY_WK1_4","SEASON_TIMING",wk.between(1,4)); add("LATE_WK11_PLUS","SEASON_TIMING",wk.ge(11))
    add("REST_IMBALANCE_2_PLUS","REST",rest.abs().ge(2)); add("HOME_SHORT_REST","REST",_num(g,"Is_Short_Rest").eq(1)); add("OPP_SHORT_REST","REST",_num(g,"Opp_Is_Short_Rest").eq(1))
    p1=_num(g,"calc_prev1_total_points"); op1=_num(g,"Opp_calc_prev1_total_points"); add("BOTH_PRIOR_TOTALS_50_PLUS","PRIOR_SCORING",p1.ge(50)&op1.ge(50)); add("BOTH_PRIOR_TOTALS_UNDER_42","PRIOR_SCORING",p1.lt(42)&op1.lt(42)); add("ONE_PRIOR_TOTAL_55_PLUS","PRIOR_SCORING",p1.ge(55)|op1.ge(55))
    add("BOTH_OFFENSE_LAST5_24_PLUS","OFFENSE_FORM",_num(g,"Avg_Points_For_Last5_Prior").ge(24)&_num(g,"Opp_Avg_Points_For_Last5_Prior").ge(24)); add("BOTH_OFFENSE_LAST5_UNDER_21","OFFENSE_FORM",_num(g,"Avg_Points_For_Last5_Prior").lt(21)&_num(g,"Opp_Avg_Points_For_Last5_Prior").lt(21))
    add("TURNOVER_VOLATILE","FORM",_num(g,"Avg_Turnover_Margin_Last3_Prior").abs().ge(1)|_num(g,"Opp_Avg_Turnover_Margin_Last3_Prior").abs().ge(1))
    return atoms


def _redundant(names:set[str])->bool:
    pairs=(("ROAD_DOG","ROAD"),("ROAD_DOG","OPEN_DOG"),("HOME_DOG","HOME"),("HOME_DOG","OPEN_DOG"),("OFF_BLOWOUT_LOSS_21_PLUS","OFF_BLOWOUT_LOSS_14_PLUS"),("BACK_TO_BACK_ATS_LOSSES","OFF_ATS_LOSS"),("OFF_SU_AND_ATS_WIN","OFF_SU_WIN"),("OFF_SU_AND_ATS_WIN","OFF_ATS_WIN"),("OFF_SU_AND_ATS_LOSS","OFF_SU_LOSS"),("OFF_SU_AND_ATS_LOSS","OFF_ATS_LOSS"),("FIRST_HOME_GAME","HOME"),("FIRST_ROAD_GAME","ROAD"))
    if any(a in names and b in names for a,b in pairs):return True
    if "OPEN_DOG" in names and any(n.startswith("DOG_") for n in names):return True
    if "OPEN_FAVORITE" in names and any(n.startswith("FAV_") for n in names):return True
    return False


def _bh_qvalues(vals):
    pv=np.asarray(vals,float); m=len(pv); q=np.ones(m,float)
    if m==0:return q
    order=np.argsort(pv); prev=1.0
    for pos in range(m-1,-1,-1):
        idx=int(order[pos]); rank=pos+1; cur=min(prev,float(pv[idx])*m/max(rank,1)); q[idx]=min(1.0,cur); prev=q[idx]
    return q


def _bootstrap_ci(labels,idx,direction,reps=BOOTSTRAP_REPS,seed=20261001):
    idx=np.asarray(idx,int); y=np.asarray(labels,float)[idx]; y=y[np.isfinite(y)]
    if len(y)<10:return [None,None]
    z=y if direction in ("PLAY_ON","OVER") else 1-y; rng=np.random.default_rng(seed); means=[]
    for _ in range(reps):means.append(float(np.mean(rng.choice(z,size=len(z),replace=True))))
    return [round(float(np.quantile(means,.025)),6),round(float(np.quantile(means,.975)),6)]


def _search_market(rows:pd.DataFrame,labels:np.ndarray,atoms:list[dict],market:str,side_mode:bool,log_func=print)->dict:
    seasons=pd.to_numeric(rows.Season,errors="coerce").astype(int).to_numpy(); disc=np.isin(seasons,DISCOVERY_SEASONS); shadow=seasons==SHADOW_SEASON; confirm=seasons==CONFIRM_SEASON; final=seasons==FINAL_CHECK_SEASON; valid_label=np.isfinite(np.asarray(labels,float))
    candidate_rows=(lambda m:_candidate_rows_side(rows,m)) if side_mode else (lambda m:(np.where(np.asarray(m,bool))[0],0))
    def rank(mask,depth,team_specific=False):
        ix,amb=candidate_rows(mask); si=ix[disc[ix]&valid_label[ix]]; y=np.asarray(labels,float)[si]
        min_n=50+8*max(depth-1,0)+(20 if team_specific else 0)
        if len(y)<min_n:return None
        raw=float(np.mean(y)); direction=("PLAY_ON" if raw>=.5 else "FADE") if market=="SPREADS" else ("OVER" if raw>=.5 else "UNDER")
        hit=max(raw,1-raw); quality=(hit-.5)*math.sqrt(len(y))*5-.12*depth
        return {"selection_games":int(len(y)),"selection_rate":float(hit),"direction":direction,"quality":float(quality),"ambiguous":int(amb)}
    beam=[]; survivors=[]; seen=set(); tested=0
    for i,a in enumerate(atoms):
        r=rank(a["mask"],1,bool(a.get("team_specific",False))); tested+=1
        if r:beam.append({"idx":(i,),"atoms":(a["name"],),"cats":(a["family"],),"mask":a["mask"],"team_specific":bool(a.get("team_specific",False)),**r})
    beam=sorted(beam,key=lambda x:(x["quality"],x["selection_games"]),reverse=True)[:BEAM_WIDTH]; survivors+=beam
    for depth in range(2,MAX_DEPTH+1):
        nxt=[]
        for st in beam:
            used=set(st["cats"])
            for i,a in enumerate(atoms):
                if i in st["idx"] or a["family"] in used:continue
                names=tuple(sorted(st["atoms"]+(a["name"],))); z=set(names)
                if names in seen or _redundant(z):continue
                seen.add(names); mm=np.asarray(st["mask"],bool)&np.asarray(a["mask"],bool); team_specific=bool(st.get("team_specific",False) or a.get("team_specific",False)); r=rank(mm,depth,team_specific); tested+=1
                if r:nxt.append({"idx":st["idx"]+(i,),"atoms":st["atoms"]+(a["name"],),"cats":st["cats"]+(a["family"],),"mask":mm,"team_specific":team_specific,**r})
        beam=sorted(nxt,key=lambda x:(x["quality"],x["selection_games"]),reverse=True)[:BEAM_WIDTH]
        if not beam:break
        survivors+=beam
    # Near-duplicate hypothesis collapse.
    unique=[]; pruned=0
    for st in sorted(survivors,key=lambda x:(len(x["atoms"]),-x["selection_games"],-x["quality"])):
        ix,_=candidate_rows(st["mask"]); sig=frozenset(str(rows.iloc[i].physical_game_id) for i in ix[disc[ix]])
        if not sig:continue
        duplicate=False
        for k in unique:
            ks=k["_sig"]; u=len(sig|ks); jac=len(sig&ks)/u if u else 1
            if jac>=JACCARD_CUTOFF:duplicate=True;break
        if duplicate:pruned+=1;continue
        q=dict(st);q["_sig"]=sig;unique.append(q)
    finals=sorted(unique,key=lambda x:(x["quality"],x["selection_games"],-len(x["atoms"])),reverse=True)[:FINALISTS]
    atom_by={a["name"]:a for a in atoms}; results=[]
    for rankno,st in enumerate(finals,1):
        ix,amb=candidate_rows(st["mask"]); direction=st["direction"]
        recs={
            "discovery":_rec(labels,ix[disc[ix]],direction),"shadow":_rec(labels,ix[shadow[ix]],direction),"confirmation":_rec(labels,ix[confirm[ix]],direction),"final_2025":_rec(labels,ix[final[ix]],direction),"all_history":_rec(labels,ix,direction)
        }
        # Year-by-year and LOSO are discovery-only robustness tests.
        yby={};
        for sy in DISCOVERY_SEASONS:yby[str(sy)]=_rec(labels,ix[seasons[ix]==sy],direction)
        loso={}
        for sy in DISCOVERY_SEASONS:loso[str(sy)]=_rec(labels,ix[disc[ix]&(seasons[ix]!=sy)],direction)
        valid_loso=[v["rate"] for v in loso.values() if v["rate"] is not None]; min_loso=min(valid_loso) if valid_loso else None
        folds={"2017_2018":_rec(labels,ix[np.isin(seasons[ix],[2017,2018])],direction),"2019_2020":_rec(labels,ix[np.isin(seasons[ix],[2019,2020])],direction),"2021_2022":_rec(labels,ix[np.isin(seasons[ix],[2021,2022])],direction)}
        valid_fold=[v["rate"] for v in folds.values() if v["rate"] is not None]; min_fold=min(valid_fold) if valid_fold else None
        # Remove discovery's best season.
        best=max(DISCOVERY_SEASONS,key=lambda sy:(yby[str(sy)]["rate"] if yby[str(sy)]["rate"] is not None else -1))
        remove_best=_rec(labels,ix[disc[ix]&(seasons[ix]!=best)],direction)
        # Drop-one-condition robustness.
        drops=[]
        if len(st["atoms"])>1:
            for omit in st["atoms"]:
                mm=np.ones(len(rows),bool)
                for nm in st["atoms"]:
                    if nm!=omit:mm &= np.asarray(atom_by[nm]["mask"],bool)
                dix,_=candidate_rows(mm); drops.append({"dropped":omit,**_rec(labels,dix[disc[dix]],direction)})
        structural=min([x["rate"] for x in drops if x["rate"] is not None],default=None)
        si=ix[disc[ix]]; boot=_bootstrap_ci(labels,si,direction,seed=20261001+rankno)
        raw_y=np.asarray(labels,float)[si]; raw_y=raw_y[np.isfinite(raw_y)]; z=raw_y if direction in ("PLAY_ON","OVER") else 1-raw_y
        pnom=float(binomtest(int(z.sum()),len(z),.5,alternative="greater").pvalue) if len(z) else 1.0
        results.append({"rank":rankno,"market":market,"conditions":list(st["atoms"]),"families":list(st["cats"]),"team_specific":bool(st.get("team_specific",False)),"direction":direction,"selection_games":st["selection_games"],"selection_rate":round(st["selection_rate"],6),"records":recs,"discovery_year_by_year":yby,"chronological_folds":folds,"min_fold_rate":min_fold,"leave_one_season_out":loso,"min_loso_rate":min_loso,"remove_best_season":best,"remove_best_season_record":remove_best,"drop_one_condition":drops,"structural_floor":structural,"bootstrap_ci95":boot,"nominal_pvalue":pnom,"ambiguous_both_sides":amb,"quality":st["quality"],"production_authority":0})
    # Multiple testing: max-stat over finalist masks with independent fair-coin null.
    sel_ix=[]; obs=[]
    for r,st in zip(results,finals):
        ix,_=candidate_rows(st["mask"]); si=ix[disc[ix]&valid_label[ix]]; sel_ix.append(si); yy=np.asarray(labels,float)[si]; obs.append(abs(float(np.mean(yy))-.5)*math.sqrt(len(yy)) if len(yy) else math.inf)
    rng=np.random.default_rng(20261002 if market=="SPREADS" else 20261003); nullmax=np.zeros(MAXSTAT_REPS)
    for b in range(MAXSTAT_REPS):
        yb=rng.integers(0,2,size=len(rows))
        mx=0.0
        for si in sel_ix:
            if len(si):mx=max(mx,abs(float(np.mean(yb[si]))-.5)*math.sqrt(len(si)))
        nullmax[b]=mx
    global_q=_bh_qvalues([r["nominal_pvalue"] for r in results]) if results else []
    families={}
    for i,r in enumerate(results):
        fk="|".join(r["families"][:2]) if r["families"] else "UNCLASSIFIED"; r["hypothesis_family"]=fk; families.setdefault(fk,[]).append(i)
    fam_names=list(families); fam_p=[]
    for fk in fam_names:
        pv=sorted(results[i]["nominal_pvalue"] for i in families[fk]); m=len(pv); fam_p.append(min([min(1,p*m/(j+1)) for j,p in enumerate(pv)],default=1))
    fam_q=_bh_qvalues(fam_p) if fam_p else []
    fam_map={fk:float(fam_q[j]) for j,fk in enumerate(fam_names)}
    for fk,ii in families.items():
        wq=_bh_qvalues([results[i]["nominal_pvalue"] for i in ii])
        for j,i in enumerate(ii):results[i]["within_family_qvalue"]=float(wq[j]);results[i]["hierarchical_family_qvalue"]=fam_map[fk]
    for i,r in enumerate(results):
        pmax=float((1+np.sum(nullmax>=obs[i]))/(MAXSTAT_REPS+1)) if np.isfinite(obs[i]) else 1.0
        r["permutation_max_pvalue"]=pmax;r["global_fdr_qvalue"]=float(global_q[i]);r["multiple_testing_pass"]=bool(pmax<=.10 and r.get("hierarchical_family_qvalue",1)<=.10 and r.get("within_family_qvalue",1)<=.10)
        sr=r["records"]["shadow"];cr=r["records"]["confirmation"];fr=r["records"]["final_2025"];dr=r["records"]["discovery"]
        robust=all(x["n"]>=12 and x["rate"] is not None and x["rate"]>BREAK_EVEN_REFERENCE for x in (sr,cr,fr))
        discovery_ok=dr["n"]>=60 and dr["rate"] is not None and dr["rate"]>=.55 and r["bootstrap_ci95"][0] is not None and r["bootstrap_ci95"][0]>.50
        floors_ok=(r["min_fold_rate"] is not None and r["min_fold_rate"]>=.50 and r["min_loso_rate"] is not None and r["min_loso_rate"]>=.51 and r["remove_best_season_record"]["rate"] is not None and r["remove_best_season_record"]["rate"]>=.51)
        team_ok=(not r.get("team_specific",False)) or (dr["n"]>=80 and sr["n"]>=14 and cr["n"]>=14 and fr["n"]>=14)
        if discovery_ok and robust and floors_ok and r["multiple_testing_pass"] and team_ok:
            status="PROMISING_REQUIRES_PROSPECTIVE"
        elif discovery_ok and sr["n"]>=10 and sr["rate"] is not None and sr["rate"]>.50:
            status="HISTORICAL_WATCHLIST_ONLY"
        else:status="EXPLORATORY"
        r["status"]=status
    promising=[r for r in results if r["status"]=="PROMISING_REQUIRES_PROSPECTIVE"]
    log_func(f"[NFL-RESEARCH-V2-SYSTEM-MINER] market={market} atoms={len(atoms)} raw_tested={tested} survivors={len(survivors)} unique_hypotheses={len(unique)} duplicate_masks_pruned={pruned} finalists={len(results)} promising={len(promising)} discovery=2017-2022 shadow=2023 confirm=2024 final=2025 year_2026_queried=FALSE")
    return {"market":market,"atom_count":len(atoms),"raw_tested":tested,"survivor_rules":len(survivors),"unique_hypotheses":len(unique),"duplicate_masks_pruned":pruned,"jaccard_cutoff":JACCARD_CUTOFF,"max_depth":MAX_DEPTH,"beam_width":BEAM_WIDTH,"maxstat_reps":MAXSTAT_REPS,"bootstrap_reps":BOOTSTRAP_REPS,"finalists":results,"promising_rules":promising,"production_authority":0}


def _metrics_status(series:pd.Series,positive_label:str)->dict:
    s=series.astype(str).str.upper(); valid=s.isin([positive_label,"WIN" if positive_label=="LOSS" else "LOSS","OVER" if positive_label=="UNDER" else "UNDER"])
    # specialized helpers below are clearer; keep this generic function unused.
    return {"n":int(valid.sum())}


def _academic_replications(state:pd.DataFrame)->dict:
    # One deterministic physical-game orientation for game-level comparisons.
    g=_home_or_neutral_one_side(state)
    out={k:dict(v) for k,v in ACADEMIC_REGISTRY.items()}
    home=g.loc[_num(g,"Is_Home").eq(1)].copy(); div=_num(home,"Is_Division_Game").eq(1)
    def ats_rate(p):
        q=p.Opening_ATS_Status.astype(str); n=int(q.isin(["WIN","LOSS"]).sum()); w=int((q=="WIN").sum()); return {"n":n,"rate":round(w/n,6) if n else None,"wins":w,"wilson95":[round(x,6) for x in _wilson(w,n)] if n else [None,None]}
    out["ACADEMIC_DIVISION_HOME_ATS"]["our_replication"]={"division_home":ats_rate(home.loc[div]),"nondivision_home":ats_rate(home.loc[~div]),"note":"Descriptive replication; no claim that original paper specification is exactly reproduced."}
    def under_rate(p):
        q=p.Opening_Total_Status.astype(str); n=int(q.isin(["OVER","UNDER"]).sum()); w=int((q=="UNDER").sum()); return {"n":n,"under_rate":round(w/n,6) if n else None,"unders":w,"wilson95":[round(x,6) for x in _wilson(w,n)] if n else [None,None]}
    out["ACADEMIC_DIVISION_TOTAL_UNDER"]["our_replication"]={"division":under_rate(g.loc[_num(g,"Is_Division_Game").eq(1)]),"nondivision":under_rate(g.loc[_num(g,"Is_Division_Game").eq(0)]),"note":"Descriptive replication using opening total; no ROI claim."}
    rest=_num(state,"Rest_Differential_Days"); adv=state.loc[rest.ge(2)].copy()
    out["ACADEMIC_REST_ERA_CHANGE"]["our_replication"]={"rest_advantage_2_plus":ats_rate(adv),"by_season":{str(int(sy)):ats_rate(p) for sy,p in adv.groupby("Season")},"note":"Current dataset begins 2017, so this tests modern-period behavior only; it cannot reproduce pre-2011 comparison."}
    return out


def _opening_retest_from_bigal(state:pd.DataFrame,bigal_plays:pd.DataFrame)->dict:
    out={}
    if bigal_plays.empty:return out
    key=state[["physical_game_id","Team_Norm","Opening_ATS_Status","Season"]].copy(); key["Team_Norm"]=key.Team_Norm.astype(str)
    for sid,p in bigal_plays.groupby("system_id"):
        x=p.merge(key,left_on=["physical_game_id","bet_team"],right_on=["physical_game_id","Team_Norm"],how="left",validate="many_to_one")
        q=x.Opening_ATS_Status.astype(str); n=int(q.isin(["WIN","LOSS"]).sum());w=int((q=="WIN").sum()); out[sid]={"n":n,"wins":w,"rate":round(w/n,6) if n else None,"wilson95":[round(z,6) for z in _wilson(w,n)] if n else [None,None],"by_season":{str(int(sy)):{"n":int(z.Opening_ATS_Status.isin(["WIN","LOSS"]).sum()),"rate":round(float(z.Opening_ATS_Status.eq("WIN").sum()/z.Opening_ATS_Status.isin(["WIN","LOSS"]).sum()),6) if z.Opening_ATS_Status.isin(["WIN","LOSS"]).sum() else None} for sy,z in x.groupby("Season")},"role":"OPENING_LINE_RETROSPECTIVE_RETEST_NO_ROI_CLAIM"}
    return out


def _cross_sport_preregistered_retests(state:pd.DataFrame)->dict:
    """Test externally specified structures that were discovered outside NFL.

    The primary rule below was frozen in the NCAAF System Miner V3 work as
    PLAY_ON: DOG_10_PLUS + OFF_BLOWOUT_LOSS_14_PLUS + BACK_TO_BACK_ATS_LOSSES.
    Because its structure/direction came from a different sport, NFL history is
    treated as a cross-sport transfer/replication test, not as NFL discovery.
    It still has zero production authority and no ROI claim.
    """
    op=_num(state,"Opening_Spread");
    masks={
      "NCAAF_V1344_TRANSFER__DOG10__BLOWOUTLOSS14__B2B_ATS_LOSSES": op.ge(10)&_num(state,"calc_prev1_margin").le(-14)&_num(state,"calc_b2b_open_ats_losses").eq(1),
      "NCAAF_V1344_TRANSFER__DOG10__BLOWOUTLOSS14__OFF_ATS_LOSS": op.ge(10)&_num(state,"calc_prev1_margin").le(-14)&_num(state,"calc_prev1_open_ats_loss").eq(1),
      "NCAAF_V1344_TRANSFER__DOG10__BLOWOUTLOSS14": op.ge(10)&_num(state,"calc_prev1_margin").le(-14),
    }
    out={}
    labels=np.where(state.Opening_ATS_Status.eq("WIN"),1.0,np.where(state.Opening_ATS_Status.eq("LOSS"),0.0,np.nan))
    for name,mask in masks.items():
        ix,amb=_candidate_rows_side(state,pd.Series(mask,index=state.index).fillna(False).to_numpy(bool))
        rec=_rec(labels,ix,"PLAY_ON")
        by={str(int(sy)):_rec(labels,ix[pd.to_numeric(state.Season,errors="coerce").to_numpy()[ix]==sy],"PLAY_ON") for sy in sorted(state.Season.unique())}
        out[name]={"direction":"PLAY_ON","origin":"NCAAF_V13_4_4_SYSTEM_MINER_V3_EXTERNAL_TO_NFL","status":"CROSS_SPORT_REPLICATION_ONLY","all_nfl_history":rec,"by_season":by,"ambiguous_both_sides":amb,"opening_line_target":True,"production_authority":0}
    return out


def _upload_immutable(storage_client,bucket_name,name,data:bytes,content_type:str):
    from google.api_core.exceptions import PreconditionFailed
    blob=storage_client.bucket(bucket_name).blob(name)
    try:
        blob.upload_from_string(data,content_type=content_type,if_generation_match=0);return {"uri":f"gs://{bucket_name}/{name}","created":True}
    except PreconditionFailed:return {"uri":f"gs://{bucket_name}/{name}","created":False,"reason":"ALREADY_EXISTS_IMMUTABLE"}


def run_nfl_system_lab_v1(*,bq_client,storage_client,bucket_name="sharp-models",audit_report=None,log_func=print):
    if not isinstance(audit_report,dict) or audit_report.get("status")!="READY_FOR_OFFLINE_CHALLENGER_SANDBOX":raise RuntimeError("NFL_RESEARCH_V2_SYSTEM_AUDIT_NOT_GREEN")
    contract=assert_contract(); view_cols={f.name for f in bq_client.get_table(VIEW).schema}; side=bq_client.query(build_intelligence_query(view_cols)).to_dataframe(create_bqstorage_client=False)
    if int(pd.to_numeric(side.Season,errors="coerce").max())>2025:raise RuntimeError("NFL_RESEARCH_V2_SYSTEM_2026_QUERY_LEAK")
    state=prepare_system_state(side)
    log_func(f"[NFL-RESEARCH-V2-SYSTEM-PREFLIGHT] status=PASS source_tag={SOURCE_TAG} data_through=2025 year_2026_queried=FALSE model_state_in_discovery=FALSE production_authority=0 contract_sha256={contract_hash()}")
    bigal_report,bigal_plays=_bigal_systems(state); bigal_open=_opening_retest_from_bigal(state,bigal_plays); pathi_report,_=_pathi_engineering(state); academic=_academic_replications(state); cross_sport=_cross_sport_preregistered_retests(state)
    # Spread miner uses opening spread to define target and role; pushes excluded.
    sp_label=np.where(state.Opening_ATS_Status.eq("WIN"),1.0,np.where(state.Opening_ATS_Status.eq("LOSS"),0.0,np.nan))
    spread_miner=_search_market(state,sp_label,_spread_atoms(state),"SPREADS",True,log_func=log_func)
    # Totals miner uses exactly one game orientation.
    games=_home_or_neutral_one_side(state); total_label=np.where(games.Opening_Total_Status.eq("OVER"),1.0,np.where(games.Opening_Total_Status.eq("UNDER"),0.0,np.nan))
    total_miner=_search_market(games,total_label,_total_atoms(games),"TOTALS",False,log_func=log_func)
    registry={
        "source_tag":SOURCE_TAG,"contract_sha256":contract_hash(),"discovery_seasons":list(DISCOVERY_SEASONS),"shadow_season":SHADOW_SEASON,"confirmation_season":CONFIRM_SEASON,"final_historical_check":FINAL_CHECK_SEASON,"year_2026_queried":False,
        "methodology":"NCAAF_SYSTEM_MINER_V3_RESEARCH_DISCIPLINE_PORT","model_state_in_discovery":False,"opening_line_primary_research_target":True,"historical_roi_claim":False,
        "bigal_system_ids":sorted(bigal_report),"pathi_status":"ENGINEERING_TRANSLATION_SHADOW_ONLY","academic_status":"REPLICATION_HYPOTHESES_ONLY","cross_sport_preregistered_ids":sorted(cross_sport),
        "spread_promising_rule_ids":["__".join(r["conditions"]) for r in spread_miner["promising_rules"]],"total_promising_rule_ids":["__".join(r["conditions"]) for r in total_miner["promising_rules"]],
        "production_authority":0,"automatic_promotion":False,"prospective_clock":"NFL_RESEARCH_V2_SYSTEM_POST_DEPLOYMENT_RETEST_CLOCK",
    }
    sha=hashlib.sha256(json.dumps(registry,sort_keys=True,separators=(",",":")).encode()).hexdigest();registry["registry_sha256"]=sha
    report={"status":STATUS,"registry":registry,"bigal":{"documented_close_reference":bigal_report,"opening_line_retest":bigal_open},"pathi_engineering":pathi_report,"academic_replications":academic,"cross_sport_preregistered_retests":cross_sport,"miner":{"spreads":spread_miner,"totals":total_miner},"production_authority":0,"year_2026_queried":False,"ncaaf":"UNCHANGED","legacy_nfl":"UNCHANGED"}
    prefix=f"nfl-research/v2_0/system_lab/{sha[:16]}"
    arts={"registry":_upload_immutable(storage_client,bucket_name,f"{prefix}/system_registry.json",json.dumps(registry,sort_keys=True,indent=2).encode(),"application/json"),"report":_upload_immutable(storage_client,bucket_name,f"{prefix}/system_lab_report.json",json.dumps(report,sort_keys=True,default=str).encode(),"application/json")}
    # Logging-only detail surface: expose every frozen promising rule so Cloud Run
    # logs are sufficient to review the System Lab without separately downloading
    # the GCS report. This does not alter search, ranking, status, registry, or artifacts.
    for _market_name, _miner in (("SPREADS", spread_miner), ("TOTALS", total_miner)):
        for _r in _miner.get("promising_rules", []):
            _detail = {
                "market": _market_name,
                "rule_id": "__".join(_r.get("conditions", [])),
                "rank": _r.get("rank"),
                "conditions": _r.get("conditions", []),
                "families": _r.get("families", []),
                "direction": _r.get("direction"),
                "records": _r.get("records", {}),
                "discovery_year_by_year": _r.get("discovery_year_by_year", {}),
                "chronological_folds": _r.get("chronological_folds", {}),
                "min_fold_rate": _r.get("min_fold_rate"),
                "min_loso_rate": _r.get("min_loso_rate"),
                "remove_best_season": _r.get("remove_best_season"),
                "remove_best_season_record": _r.get("remove_best_season_record", {}),
                "structural_floor": _r.get("structural_floor"),
                "bootstrap_ci95": _r.get("bootstrap_ci95"),
                "nominal_pvalue": _r.get("nominal_pvalue"),
                "permutation_max_pvalue": _r.get("permutation_max_pvalue"),
                "global_fdr_qvalue": _r.get("global_fdr_qvalue"),
                "hierarchical_family_qvalue": _r.get("hierarchical_family_qvalue"),
                "within_family_qvalue": _r.get("within_family_qvalue"),
                "multiple_testing_pass": _r.get("multiple_testing_pass"),
                "team_specific": _r.get("team_specific"),
                "ambiguous_both_sides": _r.get("ambiguous_both_sides"),
                "status": _r.get("status"),
                "production_authority": 0,
            }
            log_func("[NFL-RESEARCH-V2-SYSTEM-PROMISING] "+json.dumps(_detail,sort_keys=True,default=str))

    log_func("[NFL-RESEARCH-V2-SYSTEM-BIGAL] "+json.dumps({"documented_ids":sorted(bigal_report),"opening_retest":bigal_open},sort_keys=True,default=str))
    log_func("[NFL-RESEARCH-V2-SYSTEM-ACADEMIC] "+json.dumps(academic,sort_keys=True,default=str))
    log_func("[NFL-RESEARCH-V2-SYSTEM-CROSS-SPORT] "+json.dumps(cross_sport,sort_keys=True,default=str))
    log_func("[NFL-RESEARCH-V2-SYSTEM-CONTRACT] "+json.dumps({"status":STATUS,"registry_sha256":sha,"spread_promising":len(spread_miner["promising_rules"]),"total_promising":len(total_miner["promising_rules"]),"year_2026_queried":False,"model_state_in_discovery":False,"production_authority":0,"ncaaf":"UNCHANGED","legacy_nfl":"UNCHANGED","artifacts":arts},sort_keys=True,default=str))
    return {**report,"artifacts":arts}


# ------------------------------- synthetic tests -------------------------------
def self_test() -> dict:
    # Generic miner smoke test with deliberately small atom universe and enough
    # rows to exercise frozen-period logic. No assertion that a system must pass.
    rows=[]
    for sy in range(2017,2026):
        for i in range(80):
            rows.append({"Season":sy,"physical_game_id":f"{sy}|{i}","A":i%2==0,"B":i%3==0})
    d=pd.DataFrame(rows); y=np.array([1.0 if (r.A and (r.Season<=2022)) else float((idx*17+3)%2) for idx,r in d.iterrows()])
    atoms=[{"name":"A","family":"F1","mask":d.A.to_numpy(bool),"description":"A","boundary":None},{"name":"B","family":"F2","mask":d.B.to_numpy(bool),"description":"B","boundary":None}]
    rep=_search_market(d,y,atoms,"TOTALS",False,log_func=lambda *_:None)
    if rep["raw_tested"]<=0:raise AssertionError("MINER_DID_NOT_TEST")
    # Selection periods are hard-coded and 2025 cannot choose direction.
    if FINAL_CHECK_SEASON in DISCOVERY_SEASONS:raise AssertionError("FINAL_CHECK_LEAK")
    return {"status":"PASS","raw_tested":rep["raw_tested"],"unique_hypotheses":rep["unique_hypotheses"],"year_2026_queried":False,"model_state_in_discovery":False}
