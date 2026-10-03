"""NFL Edge Authority V2.5 — four-lane multidimensional combination research.

Research-only layer that combines the four interpretable NFL evidence lanes:

    CORE + STAT + SYSTEMS + MARKET

The design follows the NCAAF precedent: prediction stays separate from betting
edge authority, correlated variants collapse before counting mechanisms, and
combinations are tested as interpretable agreement/conflict states rather than
one opaque meta-model.

Historical scope
----------------
* 2021-2023 discovery
* 2024-2025 retrospective confirmation
* 2026 is forbidden

Important market distinction
----------------------------
Historical NFL replay has opening/closing movement, but does not have the full
modern timestamped T-60 book microstructure history. Therefore the historical
MARKET lane below is explicitly a BASIC_MARKET_PROXY using opening->closing
movement relative to the frozen CORE direction. It is useful to test the
combination architecture, but it is NOT a retrospective backtest of the modern
T-60 market microstructure lane. Rich market signals remain prospective/shadow.

No result from this module receives production authority.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any

import numpy as np
import pandas as pd

import nfl_stat_selector_v23 as stat23
import nfl_advanced_stat_research_v24 as stat24
import sports_edge_authority_v1 as shared

SOURCE_TAG = "nfl-multidimensional-edge-v2.5-four-lane-research-20261003"
STATUS = "NFL_MULTIDIMENSIONAL_EDGE_V25_RETROSPECTIVE_RESEARCH_COMPLETE"
DISCOVERY_SEASONS = (2021, 2022, 2023)
CONFIRM_SEASONS = (2024, 2025)
SEALED_SEASON = 2026
LANES = ("CORE", "STAT", "SYSTEM", "MARKET")

# Predeclared historical proxy thresholds. These are architecture tests, not
# optimized rich-market thresholds. The rich T-60 market lane remains prospective.
BASIC_MARKET_THRESHOLD = {"SPREADS": 0.50, "TOTALS": 0.50, "H2H": 0.01}

# Conservative repeatability gate for a combination cell. This is intentionally
# research-only and cannot promote live authority.
MIN_DISCOVERY_N = 20
MIN_CONFIRM_N = 10


def _num(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _sha(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _record_from_core_target(d: pd.DataFrame, direction: pd.Series) -> dict:
    """Grade arbitrary direction using base target encoded in CORE orientation."""
    if d is None or d.empty:
        return {"n":0,"graded_n":0,"wins":0,"losses":0,"pushes":0,"hit_rate":None,"roi_per_unit":None,"wilson95":[None,None]}
    q = d.copy()
    dr = pd.to_numeric(direction.reindex(q.index), errors="coerce")
    core = pd.to_numeric(q.get("core_direction"), errors="coerce")
    ycore = pd.to_numeric(q.get("core_target"), errors="coerce")
    valid = dr.notna() & dr.ne(0) & core.notna() & core.ne(0) & ycore.notna()
    if not valid.any():
        return {"n":0,"graded_n":0,"wins":0,"losses":0,"pushes":0,"hit_rate":None,"roi_per_unit":None,"wilson95":[None,None]}
    same = np.sign(dr.loc[valid].to_numpy(float)) == np.sign(core.loc[valid].to_numpy(float))
    target = np.where(same, ycore.loc[valid].to_numpy(float), 1.0-ycore.loc[valid].to_numpy(float))
    return shared.record(target, np.full(len(target), 100.0/110.0))


def _classification(disc: dict, conf: dict) -> str:
    dn = int((disc or {}).get("n") or 0); cn = int((conf or {}).get("n") or 0)
    dh = _num((disc or {}).get("hit_rate")); ch = _num((conf or {}).get("hit_rate"))
    dr = _num((disc or {}).get("roi_per_unit")); cr = _num((conf or {}).get("roi_per_unit"))
    if dn >= MIN_DISCOVERY_N and cn >= MIN_CONFIRM_N and dh > shared.BREAK_EVEN_110 and ch > shared.BREAK_EVEN_110 and dr > 0 and cr > 0:
        return "RETROSPECTIVE_REPEATABLE_REQUIRES_PROSPECTIVE"
    if dn >= 10 and cn >= 5 and ((math.isfinite(dr) and dr > 0) or (math.isfinite(cr) and cr > 0)):
        return "RESEARCH_WATCH"
    return "NO_REPEATABLE_COMBINATION_EDGE"


def _find_family_config(advanced_report: dict, market: str, family: str, target_mode: str = "MARKET_ORTHOGONAL") -> dict | None:
    for x in (((advanced_report or {}).get("markets") or {}).get(market,{}).get("families") or []):
        if str(x.get("family")) == family and str(x.get("target_mode")) == target_mode:
            return x
    return None


def _lead_stat_config(advanced_report: dict, market: str) -> dict | None:
    """Freeze one lead STAT representative per market using V2.4 discovery ranking.

    We do not count multiple V2.4 families as independent votes. The lead family
    is the first discovery-ranked family that also met V2.4 retrospective
    repeatability. All other repeatable families remain diagnostic siblings.
    """
    block = (((advanced_report or {}).get("markets") or {}).get(market) or {})
    repeatable = set(((block.get("repeatable_research_only_by_target") or {}).get("MARKET_ORTHOGONAL") or []))
    ranking = list(((block.get("ranking_by_target") or {}).get("MARKET_ORTHOGONAL") or []))
    for fam in ranking:
        if fam in repeatable:
            cfg = _find_family_config(advanced_report, market, fam, "MARKET_ORTHOGONAL")
            if cfg:
                return cfg
    return None


def _stat_signal_frame(*, bq_client, games: pd.DataFrame, replay_rows: pd.DataFrame, market: str, cfg: dict) -> pd.DataFrame:
    sg = stat24._add_hidden_interactions(stat23.build_historical_stats_games(bq_client=bq_client, games=games))
    if int(pd.to_numeric(sg.Season, errors="coerce").max()) >= SEALED_SEASON:
        raise RuntimeError("NFL_V25_STAT_2026_LEAK")
    family = str(cfg["family"]); alpha = float(cfg["alpha"]); beta = float(cfg["reliability_beta"])
    threshold = float((cfg.get("selected_threshold") or {}).get("threshold") or 0.0)
    oof = stat24._oof_family(sg, family=family, market=market, target_mode="MARKET_ORTHOGONAL", alpha=alpha)
    score = pd.to_numeric(oof.stat_incremental_score, errors="coerce") * beta
    out = oof[["physical_game_id","season"]].copy()
    out["stat_score"] = score.to_numpy(float)
    out["stat_direction"] = np.where(score.abs().ge(threshold) & score.notna(), np.sign(score), 0.0)
    out["stat_strong"] = out.stat_direction.ne(0)
    return out


def _system_direction(sysctx: dict, gid: str, market: str, legit_system_ids: set[str]) -> tuple[int,int,bool,list[str]]:
    votes=[]; labels=[]
    for v in ((sysctx or {}).get(str(gid),{}).get("votes") or []):
        if str(v.get("market")) != market: continue
        fam = str(v.get("family") or "")
        if fam not in legit_system_ids: continue
        dr = int(v.get("direction") or 0)
        if dr == 0: continue
        votes.append(dr); labels.append(fam)
    if not votes: return 0,0,False,[]
    if len(set(votes)) > 1: return 0,len(votes),True,labels
    return int(votes[0]),len(votes),False,labels


def _historical_lane_frame(*, bq_client, games: pd.DataFrame, replay_rows: pd.DataFrame, base: dict, sysctx: dict, families: list[dict], advanced_report: dict, market: str) -> tuple[pd.DataFrame,dict]:
    if market not in {"SPREADS","TOTALS"}:
        return pd.DataFrame(), {"status":"MARKET_NOT_SUPPORTED"}
    cfg = _lead_stat_config(advanced_report, market)
    if not cfg:
        return pd.DataFrame(), {"status":"NO_REPEATABLE_STAT_REPRESENTATIVE"}
    sf = _stat_signal_frame(bq_client=bq_client,games=games,replay_rows=replay_rows,market=market,cfg=cfg)
    b = (base or {}).get(market)
    if b is None or b.empty:
        return pd.DataFrame(), {"status":"NO_BASE_ROWS"}
    d = b.copy()
    d["physical_game_id"] = d.physical_game_id.astype(str)
    d["season"] = pd.to_numeric(d.season,errors="coerce")
    d = d.loc[d.season.isin((*DISCOVERY_SEASONS,*CONFIRM_SEASONS))].copy()
    d["core_direction"] = np.where(
        (pd.to_numeric(d.get("selected_is_home"),errors="coerce").eq(1) if market=="SPREADS" else pd.to_numeric(d.get("selected_over"),errors="coerce").eq(1)),
        1,-1,
    )
    d["core_target"] = pd.to_numeric(d.target,errors="coerce")
    d["core_edge"] = pd.to_numeric(d.raw_edge,errors="coerce")
    d = d.merge(sf[["physical_game_id","stat_score","stat_direction","stat_strong"]],on="physical_game_id",how="left",validate="one_to_one")
    d["stat_direction"] = pd.to_numeric(d.stat_direction,errors="coerce").fillna(0).astype(int)

    legit_system_ids={str(f.get("mechanism_family_id")) for f in (families or []) if f.get("mechanism_class")=="SYSTEM" and f.get("market")==market and f.get("production_authority")}
    sdir=[]; scount=[]; sconf=[]; slabel=[]
    for gid in d.physical_game_id.astype(str):
        a,bn,c,l=_system_direction(sysctx,gid,market,legit_system_ids)
        sdir.append(a); scount.append(bn); sconf.append(c); slabel.append(l)
    d["system_direction"]=sdir; d["system_vote_count"]=scount; d["system_conflict"]=sconf; d["system_labels"]=slabel

    move=pd.to_numeric(d.get("line_move_toward_selected"),errors="coerce")
    th=float(BASIC_MARKET_THRESHOLD[market])
    rel=np.where(move.ge(th),1,np.where(move.le(-th),-1,0))
    d["market_proxy_relative_to_core"]=rel.astype(int)
    d["market_direction"]=(pd.to_numeric(d.core_direction,errors="coerce").astype(int)*d.market_proxy_relative_to_core).astype(int)

    # States relative to CORE are useful for exact 4D diagnostics.
    def state(v):
        v=pd.to_numeric(v,errors="coerce").fillna(0).astype(int)
        c=pd.to_numeric(d.core_direction,errors="coerce").fillna(0).astype(int)
        return np.where(v.eq(0),"NEUTRAL",np.where(v.eq(c),"AGREE","CONFLICT"))
    d["stat_state"]=state(d.stat_direction)
    d["system_state"]=state(d.system_direction)
    d["market_state"]=state(d.market_direction)
    d["core_edge_bucket"]=pd.cut(d.core_edge,bins=[-np.inf,2,4,6,np.inf],labels=["0_2","2_4","4_6","6_PLUS"],right=False).astype(str)
    meta={
        "status":"READY","market":market,
        "lead_stat_family":cfg.get("family"),"lead_stat_alpha":cfg.get("alpha"),"lead_stat_beta":cfg.get("reliability_beta"),
        "lead_stat_threshold":cfg.get("selected_threshold"),"stat_independence_policy":"ONE_ADVANCED_STAT_LANE_PER_MARKET",
        "legit_system_ids":sorted(legit_system_ids),"market_proxy":"OPEN_TO_CLOSE_MOVEMENT_RELATIVE_TO_CORE",
        "market_proxy_threshold":th,"rich_market_microstructure_backtest":False,
    }
    return d,meta


def _subset_name(subset: tuple[str,...]) -> str:
    return "+".join(subset)


def _subset_consensus_direction(d: pd.DataFrame, subset: tuple[str,...]) -> pd.Series:
    cols={"CORE":"core_direction","STAT":"stat_direction","SYSTEM":"system_direction","MARKET":"market_direction"}
    arr=[pd.to_numeric(d[cols[x]],errors="coerce").fillna(0).astype(int) for x in subset]
    mat=np.column_stack([x.to_numpy(int) for x in arr])
    nonzero=np.all(mat!=0,axis=1)
    same=np.all(mat==mat[:,[0]],axis=1)
    out=np.where(nonzero & same,mat[:,0],0)
    return pd.Series(out,index=d.index,dtype=int)


def _combo_report(d: pd.DataFrame) -> list[dict]:
    rows=[]
    for size in range(1,len(LANES)+1):
        for subset in itertools.combinations(LANES,size):
            direction=_subset_consensus_direction(d,subset)
            rec={"combo":_subset_name(subset),"lanes":list(subset),"lane_count":len(subset)}
            for seasons,label in ((DISCOVERY_SEASONS,"discovery"),(CONFIRM_SEASONS,"confirmation")):
                mask=pd.to_numeric(d.season,errors="coerce").isin(seasons)&direction.ne(0)
                rec[label]=_record_from_core_target(d.loc[mask],direction.loc[mask])
            rec["classification"]=_classification(rec["discovery"],rec["confirmation"])
            rows.append(rec)
    return rows


def _exact_state_report(d: pd.DataFrame) -> list[dict]:
    rows=[]
    for states,g in d.groupby(["stat_state","system_state","market_state"],dropna=False,sort=True):
        rec={"stat_state":states[0],"system_state":states[1],"market_state":states[2],"core_edge_bucket":"ALL"}
        core=pd.to_numeric(g.core_direction,errors="coerce").fillna(0).astype(int)
        for seasons,label in ((DISCOVERY_SEASONS,"discovery"),(CONFIRM_SEASONS,"confirmation")):
            mask=pd.to_numeric(g.season,errors="coerce").isin(seasons)&core.ne(0)
            rec[label]=_record_from_core_target(g.loc[mask],core.loc[mask])
        rec["classification"]=_classification(rec["discovery"],rec["confirmation"])
        if int((rec["discovery"] or {}).get("n") or 0)>0 or int((rec["confirmation"] or {}).get("n") or 0)>0:
            rows.append(rec)
    # Also retain the 6+ CORE bucket because historical replay showed this region
    # behaved differently and it is a pre-existing, not post-hoc, diagnostic.
    z=d.loc[d.core_edge_bucket.eq("6_PLUS")].copy()
    if not z.empty:
        for states,g in z.groupby(["stat_state","system_state","market_state"],dropna=False,sort=True):
            rec={"stat_state":states[0],"system_state":states[1],"market_state":states[2],"core_edge_bucket":"6_PLUS"}
            core=pd.to_numeric(g.core_direction,errors="coerce").fillna(0).astype(int)
            for seasons,label in ((DISCOVERY_SEASONS,"discovery"),(CONFIRM_SEASONS,"confirmation")):
                mask=pd.to_numeric(g.season,errors="coerce").isin(seasons)&core.ne(0)
                rec[label]=_record_from_core_target(g.loc[mask],core.loc[mask])
            rec["classification"]=_classification(rec["discovery"],rec["confirmation"])
            if int((rec["discovery"] or {}).get("n") or 0)>0 or int((rec["confirmation"] or {}).get("n") or 0)>0:
                rows.append(rec)
    return rows


def _prospective_contract(market_meta: dict) -> dict:
    return {
        "core":"FROZEN_PRODUCTION_FAIR_VALUE_DIRECTION",
        "stat":{
            "family":market_meta.get("lead_stat_family"),"alpha":market_meta.get("lead_stat_alpha"),
            "reliability_beta":market_meta.get("lead_stat_beta"),"threshold":market_meta.get("lead_stat_threshold"),
            "independence_key":"NFL_ADVANCED_STAT",
        },
        "systems":{"family_ids":market_meta.get("legit_system_ids") or [],"dependency_policy":"EXISTING_V23_SYSTEM_INDEPENDENCE_KEYS"},
        "market":{
            "historical_proxy":"OPEN_TO_CLOSE_MOVEMENT_RELATIVE_TO_CORE",
            "historical_proxy_threshold":market_meta.get("market_proxy_threshold"),
            "prospective_rich_lane":"T_MINUS_60_TIMESTAMPED_BOOK_MICROSTRUCTURE",
            "rich_families":[
                "CROSS_BOOK_DISAGREEMENT","KEY_CROSSING_PERSISTENCE","LINE_PRICE_DIVERGENCE",
                "LINE_VELOCITY_AND_PERSISTENCE","QUOTE_STALENESS_AND_DISPERSION","SHARP_VS_SOFT_LEAD_LAG",
            ],
            "rich_market_production_authority":0,
        },
        "combination_policy":"INTERPRETABLE_LANE_AGREEMENT_CONFLICT_MATRIX_NOT_META_MODEL",
        "production_authority":0,
    }


def run_multidimensional_research(*, bq_client, games: pd.DataFrame, replay_rows: pd.DataFrame, base: dict, sysctx: dict, families: list[dict], advanced_stat_report: dict, log_func=print) -> dict:
    log_func("[NFL-MULTI-V25-PREFLIGHT] "+json.dumps({
        "status":"START","source_tag":SOURCE_TAG,"lanes":list(LANES),
        "discovery_seasons":list(DISCOVERY_SEASONS),"confirmation_seasons":list(CONFIRM_SEASONS),
        "year_2026_queried":False,"production_authority":0,
        "historical_market_lane":"BASIC_OPEN_CLOSE_PROXY_ONLY",
        "rich_market_lane":"PROSPECTIVE_T60_ONLY",
        "design":"NCAAF_STYLE_INTERPRETABLE_MULTI_MECHANISM_MATRIX",
    },sort_keys=True))
    report={"status":STATUS,"source_tag":SOURCE_TAG,"year_2026_queried":False,"production_authority":0,"markets":{}}
    for market in ("SPREADS","TOTALS"):
        d,meta=_historical_lane_frame(bq_client=bq_client,games=games,replay_rows=replay_rows,base=base,sysctx=sysctx,families=families,advanced_report=advanced_stat_report,market=market)
        if d.empty:
            report["markets"][market]={"status":meta.get("status"),"meta":meta,"combinations":[],"exact_states":[]}
            log_func("[NFL-MULTI-V25-MARKET-HOLD] "+json.dumps({"market":market,**meta},sort_keys=True,default=str))
            continue
        combos=_combo_report(d); states=_exact_state_report(d)
        repeatable=[x for x in combos if x.get("classification")=="RETROSPECTIVE_REPEATABLE_REQUIRES_PROSPECTIVE"]
        # Stable ordering: more lanes first, then confirmation N, then name. This is
        # a report ordering only; it is not a promotion/ranking rule.
        display=sorted(repeatable,key=lambda x:(-int(x.get("lane_count") or 0),-int((x.get("confirmation") or {}).get("n") or 0),str(x.get("combo"))))
        for x in combos:
            log_func("[NFL-MULTI-V25-COMBINATION] "+json.dumps({"market":market,**x},sort_keys=True,default=str))
        for x in states:
            if x.get("classification")!="NO_REPEATABLE_COMBINATION_EDGE":
                log_func("[NFL-MULTI-V25-STATE] "+json.dumps({"market":market,**x},sort_keys=True,default=str))
        pcontract=_prospective_contract(meta)
        report["markets"][market]={
            "status":"READY","meta":meta,"combinations":combos,"exact_states":states,
            "retrospective_repeatable_combinations":[x.get("combo") for x in display],
            "prospective_contract":pcontract,
        }
        log_func("[NFL-MULTI-V25-MARKET] "+json.dumps({
            "market":market,"lead_stat_family":meta.get("lead_stat_family"),"system_family_ids":meta.get("legit_system_ids"),
            "repeatable_combinations":[x.get("combo") for x in display],"historical_market_proxy":meta.get("market_proxy"),
            "rich_market_microstructure_backtest":False,"production_authority":0,
        },sort_keys=True,default=str))
    stable={
        "source_tag":SOURCE_TAG,"lanes":list(LANES),"discovery":list(DISCOVERY_SEASONS),"confirmation":list(CONFIRM_SEASONS),
        "market_proxy_thresholds":BASIC_MARKET_THRESHOLD,
        "markets":{m:{"meta":(v.get("meta") or {}),"repeatable":v.get("retrospective_repeatable_combinations") or [],"prospective_contract":v.get("prospective_contract") or {}} for m,v in report["markets"].items()},
    }
    report["research_contract_sha256"]=_sha(stable)
    log_func("[NFL-MULTI-V25-CONTRACT] "+json.dumps({
        "status":STATUS,"source_tag":SOURCE_TAG,"research_contract_sha256":report["research_contract_sha256"],
        "spread_repeatable_combinations":((report["markets"].get("SPREADS") or {}).get("retrospective_repeatable_combinations") or []),
        "total_repeatable_combinations":((report["markets"].get("TOTALS") or {}).get("retrospective_repeatable_combinations") or []),
        "historical_market_caveat":"BASIC_OPEN_CLOSE_PROXY_NOT_RICH_T60_MICROSTRUCTURE",
        "production_authority":0,"year_2026_queried":False,
        "next_test":"FREEZE FOUR_LANE CONTRACT THEN SCORE 2026 PROSPECTIVELY; DO NOT RETUNE ON 2026",
    },sort_keys=True,default=str))
    return report


def _self_test():
    d=pd.DataFrame({
        "core_direction":[1,1,1,-1],"stat_direction":[1,1,-1,-1],"system_direction":[1,0,-1,-1],"market_direction":[1,-1,-1,-1],
        "core_target":[1,0,1,0],"season":[2021,2022,2024,2025],"core_edge":[3,5,7,1],
    })
    x=_subset_consensus_direction(d,("CORE","STAT"))
    assert list(x)==[1,1,0,-1]
    y=_subset_consensus_direction(d,("STAT","SYSTEM","MARKET"))
    assert list(y)==[1,0,-1,-1]
    assert set(LANES)=={"CORE","STAT","SYSTEM","MARKET"}
    assert 2026 not in DISCOVERY_SEASONS+CONFIRM_SEASONS
    return {"status":"PASS","source_tag":SOURCE_TAG,"lanes":LANES}


if __name__=="__main__":
    print(json.dumps(_self_test(),sort_keys=True))
