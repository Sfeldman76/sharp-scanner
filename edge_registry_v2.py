"""EDGE_REGISTRY_V2 — peer edge-generator research with confirmation states and exclusive Miner support.

Purpose
-------
Do not assume one master model owns the edge.  Register and evaluate independent
edge generators (STAT consensus tails, Big Al, Pathi, System Miner) under a common
zero-authority evidence contract, then inspect agreement/conflict overlaps.

This module does NOT blend source probabilities or grant production authority.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any
import math
import numpy as np
import pandas as pd

EDGE_REGISTRY_V2_SOURCE_TAG = "edge-registry-v2-evidence-states"
BREAK_EVEN = 110.0 / 210.0
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026


def _roi(hit: float) -> float:
    if not np.isfinite(hit):
        return np.nan
    return float(hit * (100.0 / 110.0) - (1.0 - hit))


def _physical_key(g: pd.DataFrame) -> pd.Series:
    if "Source_Game_ID" in g.columns:
        k = g["Source_Game_ID"].astype(str).str.strip().str.lower()
        if k.ne("").sum() >= int(0.9 * len(g)):
            return k
    season = pd.to_numeric(g.get("Season"), errors="coerce").astype("Int64").astype(str)
    date = pd.to_datetime(g.get("Game_Date"), errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("")
    team = g.get("Team_Norm", pd.Series("", index=g.index)).astype(str).str.lower().str.strip()
    opp = g.get("Opponent_Norm", pd.Series("", index=g.index)).astype(str).str.lower().str.strip()
    return season + "|" + date + "|" + team + "|" + opp


def _evidence_from_occurrences(occurrences: List[dict]) -> dict:
    rows=[]
    for o in occurrences or []:
        if not isinstance(o,dict):
            continue
        try:
            sy=int(o.get("season")); y=float(o.get("ats_win"))
        except Exception:
            continue
        if not np.isfinite(y):
            continue
        rows.append((sy,1.0 if y>0.5 else 0.0))
    if not rows:
        return {"n":0,"hit":np.nan,"roi":np.nan,"remove_best":np.nan,"min_loso":np.nan,"positive_seasons":0,"season_count":0,"confirm_n":0,"confirm_hit":np.nan}
    d=pd.DataFrame(rows,columns=["season","win"])
    disc=d[d.season.isin(DISCOVERY_SEASONS)].copy()
    conf=d[d.season.eq(CONFIRM_SEASON)].copy()
    if disc.empty:
        return {"n":0,"hit":np.nan,"roi":np.nan,"remove_best":np.nan,"min_loso":np.nan,"positive_seasons":0,"season_count":0,"confirm_n":int(len(conf)),"confirm_hit":float(conf.win.mean()) if len(conf) else np.nan}
    hit=float(disc.win.mean()); by=[]
    for sy,gg in disc.groupby("season"):
        if len(gg)>=5:
            by.append((int(sy),int(len(gg)),float(gg.win.mean())))
    pos=sum(h>.5 for _,_,h in by)
    rb=np.nan; loso=[]
    if len(by)>=2:
        best=max(by,key=lambda z:z[2])[0]
        rem=disc[disc.season.ne(best)]
        if len(rem): rb=float(rem.win.mean())
        for sy,_,_ in by:
            z=disc[disc.season.ne(sy)]
            if len(z): loso.append(float(z.win.mean()))
    return {
        "n":int(len(disc)),"hit":hit,"roi":_roi(hit),
        "remove_best":rb,"min_loso":float(min(loso)) if loso else np.nan,
        "positive_seasons":int(pos),"season_count":int(len(by)),
        "confirm_n":int(len(conf)),"confirm_hit":float(conf.win.mean()) if len(conf) else np.nan,
        "season_rows":by,
    }


def _strength(ev: dict) -> str:
    """Evidence strength, not expected ATS magnitude.  Research/shadow only."""
    n=int(ev.get("n",0) or 0); hit=float(ev.get("hit",np.nan)); rb=float(ev.get("remove_best",np.nan)); ml=float(ev.get("min_loso",np.nan))
    sc=int(ev.get("season_count",0) or 0); pos=int(ev.get("positive_seasons",0) or 0)
    if n>=100 and sc>=3 and np.isfinite(hit) and hit>=.54 and np.isfinite(rb) and rb>=.525 and np.isfinite(ml) and ml>=.515 and pos>=2:
        return "STRONG_SHADOW"
    if n>=60 and sc>=2 and np.isfinite(hit) and hit>=.53 and np.isfinite(rb) and rb>=.515 and pos>=2:
        return "MODERATE_SHADOW"
    if n>=30 and sc>=2 and np.isfinite(hit) and hit>BREAK_EVEN and pos>=2:
        return "WEAK_SHADOW"
    return "RESEARCH_ONLY"


def _confirmation_state(ev: dict) -> str:
    n=int(ev.get("confirm_n",0) or 0); hit=float(ev.get("confirm_hit",np.nan))
    if n < 5 or not np.isfinite(hit):
        return "NO_OR_TINY_SAMPLE"
    if n < 20:
        return "SMALL_SAMPLE"
    if hit >= 0.55:
        return "CONFIRMING_STRONG"
    if hit >= BREAK_EVEN:
        return "CONFIRMING"
    if hit >= 0.48:
        return "NEUTRAL"
    return "CONTRADICTING"


def _miner_hist_strength(sys: dict) -> str:
    n=int(sys.get("selection_games",0) or 0); hit=float(sys.get("selection_rate",np.nan)); rb=float(sys.get("remove_best_season_rate",np.nan)); ml=float(sys.get("loso_min_rate",np.nan))
    state=str(sys.get("authority_state","SHADOW"))
    if state=="QUALIFIED": return "STRONG_SHADOW"
    if n>=100 and np.isfinite(hit) and hit>=.54 and np.isfinite(rb) and rb>=.515 and np.isfinite(ml) and ml>=.515: return "MODERATE_SHADOW"
    if n>=60 and np.isfinite(hit) and hit>BREAK_EVEN: return "WEAK_SHADOW"
    return "RESEARCH_ONLY"


def _stat_tail_entry(stat_out: dict, games: pd.DataFrame) -> dict:
    p=((stat_out or {}).get("primary") or {})
    leader=p.get("leader") or {}; prior=p.get("prior") or {}; met=prior.get("met") or {}; test=p.get("test") or {}
    n=int(met.get("n",0) or 0); hit=float(met.get("hit",np.nan)); rb=float((prior.get("remove_best") or {}).get("hit",np.nan))
    per=prior.get("per") or []; pos=sum(np.isfinite(x[1].get("hit",np.nan)) and x[1].get("hit",0)>.5 for x in per)
    ev={"n":n,"hit":hit,"roi":float(met.get("roi",np.nan)),"remove_best":rb,
        "min_loso":float(prior.get("min_hit",np.nan)),"positive_seasons":pos,"season_count":len(per),
        "confirm_n":int(test.get("n",0) or 0),"confirm_hit":float(test.get("hit",np.nan))}
    status="MODERATE_SHADOW" if p.get("status")=="TAIL_PASS" else "RESEARCH_ONLY"
    # Do not upgrade a tail beyond its own chronological admission result.
    return {"source":"STAT","edge_id":"STAT_CONSENSUS_TAIL","label":"+".join(leader.get("combo") or []),
            "threshold":leader.get("threshold"),"strength":status,"evidence":ev,
            "signed_market_error":float(met.get("signed",np.nan)),"production_authority":0}


def _system_memory_entries(system_history: dict) -> List[dict]:
    out=[]
    for name,rec in sorted((system_history or {}).items()):
        if not isinstance(rec,dict) or str(rec.get("role","directional")).lower()!="directional":
            continue
        if not bool(rec.get("source_validation_pass",True)):
            continue
        fam=str(rec.get("family","UNKNOWN")).upper()
        if fam not in ("BIGAL","PATHI"):
            continue
        ev=_evidence_from_occurrences(rec.get("occurrences") or [])
        out.append({"source":fam,"edge_id":name,"label":name,"strength":_strength(ev),"evidence":ev,
                    "trust":float(rec.get("trust",0.0) or 0.0),"posterior":float(rec.get("posterior_prob",np.nan)),
                    "production_authority":0})
    return out


def _miner_entries(miner_registry: dict) -> List[dict]:
    out=[]
    spreads=(miner_registry or {}).get("spreads") or {}
    for sys in spreads.get("systems") or []:
        n=int(sys.get("selection_games",0) or 0); hit=float(sys.get("selection_rate",np.nan)); rb=float(sys.get("remove_best_season_rate",np.nan)); ml=float(sys.get("loso_min_rate",np.nan))
        path=str(sys.get("qualification_path","NONE")); q=float(sys.get("fdr_qvalue",np.nan))
        strength=_miner_hist_strength(sys)
        out.append({"source":"MINER","edge_id":str(sys.get("system_id")),"label":" AND ".join(sys.get("conditions") or []),
                    "strength":strength,"evidence":{"n":n,"hit":hit,"roi":float(sys.get("historical_roi",np.nan)),"remove_best":rb,"min_loso":ml,
                    "confirm_n":int(sys.get("shadow_games",0) or 0),"confirm_hit":float(sys.get("shadow_rate",np.nan))},
                    "action":str(sys.get("system_action",sys.get("system_type","BET"))),"direction":str(sys.get("direction","")),
                    "qualification_path":path,"fdr_qvalue":q,"production_authority":0})
    return out

def _entry_sort_key(e):
    order={"STRONG_SHADOW":0,"MODERATE_SHADOW":1,"WEAK_SHADOW":2,"RESEARCH_ONLY":3}
    ev=e.get("evidence") or {}
    h=float(ev.get("hit",np.nan)); n=int(ev.get("n",0) or 0)
    return (order.get(e.get("strength"),9), -(h if np.isfinite(h) else -1), -n, str(e.get("edge_id")))


def _log_entry(e, log_func=print):
    ev=e.get("evidence") or {}; cs=_confirmation_state(ev)
    extra=""
    if e.get("source")=="MINER":
        extra=f" qualification_path={e.get('qualification_path','NONE')} fdr={float(e.get('fdr_qvalue',np.nan)):.4f}"
    log_func(
        f"[EDGE-REGISTRY-V2-SOURCE] source={e.get('source')} edge_id={e.get('edge_id')} historical_strength={e.get('strength')} confirmation_state={cs} "
        f"discovery_n={int(ev.get('n',0) or 0)} discovery_hit={float(ev.get('hit',np.nan)):.4f} discovery_roi={float(ev.get('roi',np.nan)):+.4f} "
        f"remove_best={float(ev.get('remove_best',np.nan)):.4f} min_loso={float(ev.get('min_loso',np.nan)):.4f} "
        f"confirm_2026_n={int(ev.get('confirm_n',0) or 0)} confirm_2026_hit={float(ev.get('confirm_hit',np.nan)):.4f}{extra} production_authority=0"
    )



def _hybrid_from_stat(g: pd.DataFrame, stat_out: dict, dashboard_module, log_func=print) -> List[dict]:
    p=((stat_out or {}).get("primary") or {})
    leader=p.get("leader") or {}
    pred=leader.get("pred"); prior=(p.get("prior") or {}).get("mask"); test=p.get("test_mask")
    if pred is None or prior is None or test is None:
        return []
    pred=np.asarray(pred,float); prior=np.asarray(prior,bool); test=np.asarray(test,bool)
    target=pd.to_numeric(g.get("Market_Error_Margin"),errors="coerce").to_numpy(float)
    out=[]
    try:
        import v14_stat_reliability as rel
        import stat_combination_v2_1 as scv
        hist=getattr(dashboard_module,"_V143_SYSTEM_HISTORY_CACHE",{})
        sm=rel._system_occurrence_masks(g,scv._side_for_prediction(g,pred),hist,log_func=lambda *_:None)
        for state in ("BIGAL_AGREE","PATHI_AGREE","SYSTEM_ANY_AGREE","BIGAL_CONFLICT","PATHI_CONFLICT","SYSTEM_ANY_CONFLICT","SYSTEM_MIXED"):
            if state not in sm: continue
            for label,base in (("DISCOVERY",prior),("CONFIRM_2026",test)):
                m=base&np.asarray(sm[state],bool); met=scv._metrics(target,pred,m)
                if met["n"]>=5:
                    role="SUPPORT" if "AGREE" in state else ("CONFLICT" if "CONFLICT" in state else "MIXED")
                    log_func(f"[EDGE-REGISTRY-V2-HYBRID] edge=STAT_TAIL+{state} sample={label} n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed_market_error={met['signed']:+.3f} support_role={role} production_authority=0")
                    out.append({"edge":f"STAT_TAIL+{state}","sample":label,"role":role,**met})
    except Exception as e:
        log_func(f"[EDGE-REGISTRY-V2-HYBRID] source=BIGAL_PATHI status=UNAVAILABLE error={type(e).__name__}:{e}")

    try:
        import stat_combination_v2_1 as scv
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        miner_registry=cache.get("system_miner_v2") or {}; mg=cache.get("miner_games")
        spreads=(miner_registry.get("spreads") or {}) if isinstance(miner_registry,dict) else {}; systems=spreads.get("systems") or []
        if isinstance(mg,pd.DataFrame) and len(mg)==len(g) and systems:
            atoms={a["name"]:np.asarray(a["mask"],bool) for a in dashboard_module._v1355_system_atoms(mg)}
            independent=[]; excluded=[]
            for sys in systems:
                cond=list(sys.get("conditions") or [])
                if any(str(c).startswith(("STAT_EDGE_","H2H_STAT_","TOTAL_MODEL_")) for c in cond): excluded.append(sys); continue
                independent.append(sys)
            sgn=np.sign(pred)
            def masks_for(group):
                ag=np.zeros(len(g),bool); cf=np.zeros(len(g),bool)
                for sys in group:
                    mm=np.ones(len(g),bool); cond=sys.get("conditions") or []
                    if not cond: continue
                    for c in cond: mm &= atoms.get(c,np.zeros(len(g),bool))
                    direction=str(sys.get("direction","PLAY_ON")); play_sign=1.0 if direction=="PLAY_ON" else -1.0
                    ag |= mm & np.isfinite(sgn) & (sgn==play_sign)
                    cf |= mm & np.isfinite(sgn) & (sgn==-play_sign)
                mixed=ag&cf; return ag&~mixed,cf&~mixed,mixed
            moderate_plus=[sys for sys in independent if _miner_hist_strength(sys) in ("STRONG_SHADOW","MODERATE_SHADOW")]
            groups=(("MINER_ANY",independent),("MINER_MODERATE_PLUS",moderate_plus))
            for prefix,group in groups:
                support,conflict,mixed=masks_for(group)
                log_func(f"[EDGE-REGISTRY-V2-MINER-JOIN] tier={prefix} systems={len(group)} independent_total={len(independent)} excluded_model_state={len(excluded)} support_only_games={int(support.sum())} conflict_only_games={int(conflict.sum())} mixed_games={int(mixed.sum())}")
                for state,mask,role in ((f"{prefix}_SUPPORT_ONLY",support,"SUPPORT"),(f"{prefix}_CONFLICT_ONLY",conflict,"CONFLICT"),(f"{prefix}_MIXED",mixed,"MIXED")):
                    for label,base in (("DISCOVERY",prior),("CONFIRM_2026",test)):
                        met=scv._metrics(target,pred,base&mask)
                        if met["n"]>=5:
                            log_func(f"[EDGE-REGISTRY-V2-HYBRID] edge=STAT_TAIL+{state} sample={label} n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed_market_error={met['signed']:+.3f} support_role={role} production_authority=0")
                            out.append({"edge":f"STAT_TAIL+{state}","sample":label,"role":role,**met})
    except Exception as e:
        log_func(f"[EDGE-REGISTRY-V2-HYBRID] source=MINER status=UNAVAILABLE error={type(e).__name__}:{e}")
    return out

def run_edge_registry_v2(*, dashboard_module, stat_out: dict, reliability_out: dict | None=None, log_func=print, hard_fail=True):
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        g=cache.get("games") if isinstance(cache,dict) else None
        if g is None or not isinstance(g,pd.DataFrame) or g.empty:
            raise RuntimeError("EDGE_REGISTRY_V2 requires NCAAF spread research cache")
        keys=_physical_key(g); dup=int(keys[keys.ne("")&keys.ne("nan")].duplicated().sum())
        if dup: raise RuntimeError(f"physical game duplication rows={dup}")
        system_history=getattr(dashboard_module,"_V143_SYSTEM_HISTORY_CACHE",{})
        miner_registry=cache.get("system_miner_v2") or {}
        entries=[_stat_tail_entry(stat_out,g)]
        entries.extend(_system_memory_entries(system_history)); entries.extend(_miner_entries(miner_registry))
        if len(entries)<3: raise RuntimeError(f"too few edge sources={len(entries)}")
        entries=sorted(entries,key=_entry_sort_key)
        counts={st:sum(e.get("strength")==st for e in entries) for st in ("STRONG_SHADOW","MODERATE_SHADOW","WEAK_SHADOW","RESEARCH_ONLY")}
        source_counts={src:sum(e.get("source")==src for e in entries) for src in ("STAT","BIGAL","PATHI","MINER")}
        confirm_counts={}
        for e in entries:
            cs=_confirmation_state(e.get("evidence") or {}); confirm_counts[cs]=confirm_counts.get(cs,0)+1
        log_func(f"[EDGE-REGISTRY-V2-PREFLIGHT] status=PASS source_tag={EDGE_REGISTRY_V2_SOURCE_TAG} games={len(g)} physical_game_duplicates=0 sources={source_counts} historical_strength_counts={counts} confirmation_counts={confirm_counts} no_master_model=TRUE production_authority=0")
        for e in entries: _log_entry(e,log_func)
        hybrids=_hybrid_from_stat(g,stat_out,dashboard_module,log_func)
        p=((stat_out or {}).get("primary") or {}); prior=((p.get("prior") or {}).get("mask")); test=p.get("test_mask")
        season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
        disc_eligible=np.isin(season,np.asarray(DISCOVERY_SEASONS,float)); conf_eligible=season==float(CONFIRM_SEASON)
        prior_n=int(np.asarray(prior,bool).sum()) if prior is not None else 0; test_n=int(np.asarray(test,bool).sum()) if test is not None else 0
        log_func(f"[EDGE-REGISTRY-V2-RARITY] source=STAT_CONSENSUS_TAIL discovery_flagged={prior_n} discovery_eligible={int(disc_eligible.sum())} discovery_rate={(prior_n/max(1,int(disc_eligible.sum()))):.4f} confirm_2026_flagged={test_n} confirm_2026_eligible={int(conf_eligible.sum())} confirm_rate={(test_n/max(1,int(conf_eligible.sum()))):.4f}")
        wf=(stat_out or {}).get("walkforward") or {}; wfm=wf.get("metrics") or {}; wfv=wf.get("v13") or {}; wfd=wf.get("diag") or {}
        if int(wfm.get("n",0) or 0)>0:
            log_func(f"[EDGE-REGISTRY-V2-STAT-WALKFORWARD] seasons={wf.get('seasons')} n={int(wfm.get('n',0))} hit={float(wfm.get('hit',np.nan)):.4f} roi={float(wfm.get('roi',np.nan)):+.4f} signed_market_error={float(wfm.get('signed',np.nan)):+.3f} v13_hit_same_rows={float(wfv.get('hit',np.nan)):.4f} direction_agree_v13={float(wfd.get('agree_rate',np.nan)):.4f} role={wfd.get('role','UNPROVEN')} chronological=TRUE")
        log_func(f"[EDGE-REGISTRY-V2-CONTRACT] status=PASS source_tag={EDGE_REGISTRY_V2_SOURCE_TAG} registered_edges={len(entries)} hybrids_evaluated={len(hybrids)} peer_generators=STAT,BIGAL,PATHI,MINER historical_strength_separate_from_confirmation=TRUE miner_hybrids_exclusive=TRUE miner_model_state_circularity_blocked=TRUE systems_not_probability_weighted=TRUE zero_production_authority=TRUE")
        return {"status":"PASS","source_tag":EDGE_REGISTRY_V2_SOURCE_TAG,"entries":entries,"hybrids":hybrids,"production_authority":0}
    except Exception as e:
        log_func(f"[EDGE-REGISTRY-V2-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}

