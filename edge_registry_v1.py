"""EDGE_REGISTRY_V1 — peer edge-generator research for NCAAF spreads.

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

EDGE_REGISTRY_SOURCE_TAG = "edge-registry-v1-peer-generators"
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
    for s in spreads.get("systems") or []:
        n=int(s.get("selection_games",0) or 0); hit=float(s.get("selection_rate",np.nan)); rb=float(s.get("remove_best_season_rate",np.nan)); ml=float(s.get("loso_min_rate",np.nan));
        # Miner already has its own multipath gate.  Preserve it instead of replacing
        # it with a second, incompatible qualification formula.
        state=str(s.get("authority_state","SHADOW"))
        path=str(s.get("qualification_path","NONE"))
        if state=="QUALIFIED": strength="STRONG_SHADOW"
        elif n>=100 and np.isfinite(hit) and hit>=.54 and np.isfinite(rb) and rb>=.515: strength="MODERATE_SHADOW"
        elif n>=60 and np.isfinite(hit) and hit>BREAK_EVEN: strength="WEAK_SHADOW"
        else: strength="RESEARCH_ONLY"
        out.append({"source":"MINER","edge_id":str(s.get("system_id")),"label":" AND ".join(s.get("conditions") or []),
                    "strength":strength,"evidence":{"n":n,"hit":hit,"roi":float(s.get("historical_roi",np.nan)),"remove_best":rb,"min_loso":ml,
                    "confirm_n":int(s.get("shadow_games",0) or 0),"confirm_hit":float(s.get("shadow_rate",np.nan))},
                    "action":str(s.get("system_action",s.get("system_type","BET"))),"qualification_path":path,"production_authority":0})
    return out


def _entry_sort_key(e):
    order={"STRONG_SHADOW":0,"MODERATE_SHADOW":1,"WEAK_SHADOW":2,"RESEARCH_ONLY":3}
    ev=e.get("evidence") or {}
    h=float(ev.get("hit",np.nan)); n=int(ev.get("n",0) or 0)
    return (order.get(e.get("strength"),9), -(h if np.isfinite(h) else -1), -n, str(e.get("edge_id")))


def _log_entry(e, log_func=print):
    ev=e.get("evidence") or {}
    log_func(
        f"[EDGE-REGISTRY-SOURCE] source={e.get('source')} edge_id={e.get('edge_id')} strength={e.get('strength')} "
        f"discovery_n={int(ev.get('n',0) or 0)} discovery_hit={float(ev.get('hit',np.nan)):.4f} discovery_roi={float(ev.get('roi',np.nan)):+.4f} "
        f"remove_best={float(ev.get('remove_best',np.nan)):.4f} min_loso={float(ev.get('min_loso',np.nan)):.4f} "
        f"confirm_2026_n={int(ev.get('confirm_n',0) or 0)} confirm_2026_hit={float(ev.get('confirm_hit',np.nan)):.4f} production_authority=0"
    )


def _hybrid_from_stat(g: pd.DataFrame, stat_out: dict, dashboard_module, log_func=print) -> List[dict]:
    """Test support/conflict around the frozen 2026 STAT-tail candidate.

    This does not create a new fair line.  It asks whether independent system
    generators support or reject the exact same STAT-tail selections.
    """
    p=((stat_out or {}).get("primary") or {})
    leader=p.get("leader") or {}
    pred=leader.get("pred")
    prior=(p.get("prior") or {}).get("mask")
    test=p.get("test_mask")
    if pred is None or prior is None or test is None:
        return []
    pred=np.asarray(pred,float); prior=np.asarray(prior,bool); test=np.asarray(test,bool)
    target=pd.to_numeric(g.get("Market_Error_Margin"),errors="coerce").to_numpy(float)
    out=[]
    try:
        import v14_stat_reliability as rel
        # stat_combination_v2 is already loaded in the same process; import only helper.
        import stat_combination_v2 as scv2
        hist=getattr(dashboard_module,"_V143_SYSTEM_HISTORY_CACHE",{})
        sm=rel._system_occurrence_masks(g,scv2._side_for_prediction(g,pred),hist,log_func=lambda *_:None)
        for state in ("BIGAL_AGREE","PATHI_AGREE","SYSTEM_ANY_AGREE","BIGAL_CONFLICT","PATHI_CONFLICT","SYSTEM_ANY_CONFLICT"):
            if state not in sm: continue
            for label,base in (("DISCOVERY",prior),("CONFIRM_2026",test)):
                m=base&np.asarray(sm[state],bool)
                met=scv2._metrics(target,pred,m)
                if met["n"]>=5:
                    log_func(f"[EDGE-REGISTRY-HYBRID] edge=STAT_TAIL+{state} sample={label} n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed_market_error={met['signed']:+.3f} support_role={'SUPPORT' if 'AGREE' in state else 'CONFLICT'} production_authority=0")
                    out.append({"edge":f"STAT_TAIL+{state}","sample":label,**met})
    except Exception as e:
        log_func(f"[EDGE-REGISTRY-HYBRID] source=BIGAL_PATHI status=UNAVAILABLE error={type(e).__name__}:{e}")

    # Miner support/conflict on the same row-oriented STAT-tail selections.
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        miner_registry=cache.get("system_miner_v2") or {}
        mg=cache.get("miner_games")
        spreads=(miner_registry.get("spreads") or {}) if isinstance(miner_registry,dict) else {}
        systems=spreads.get("systems") or []
        if isinstance(mg,pd.DataFrame) and len(mg)==len(g) and systems:
            atoms={a["name"]:np.asarray(a["mask"],bool) for a in dashboard_module._v1355_system_atoms(mg)}
            agree=np.zeros(len(g),bool); conflict=np.zeros(len(g),bool)
            sgn=np.sign(pred)
            for sys in systems:
                mm=np.ones(len(g),bool)
                cond=sys.get("conditions") or []
                if not cond: continue
                for c in cond:
                    mm &= atoms.get(c,np.zeros(len(g),bool))
                direction=str(sys.get("direction","PLAY_ON"))
                play_sign=1.0 if direction=="PLAY_ON" else -1.0
                agree |= mm & np.isfinite(sgn) & (sgn==play_sign)
                conflict |= mm & np.isfinite(sgn) & (sgn==-play_sign)
            import stat_combination_v2 as scv2
            for state,mask in (("MINER_AGREE",agree),("MINER_CONFLICT",conflict)):
                for label,base in (("DISCOVERY",prior),("CONFIRM_2026",test)):
                    met=scv2._metrics(target,pred,base&mask)
                    if met["n"]>=5:
                        log_func(f"[EDGE-REGISTRY-HYBRID] edge=STAT_TAIL+{state} sample={label} n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed_market_error={met['signed']:+.3f} support_role={'SUPPORT' if 'AGREE' in state else 'CONFLICT'} production_authority=0")
                        out.append({"edge":f"STAT_TAIL+{state}","sample":label,**met})
    except Exception as e:
        log_func(f"[EDGE-REGISTRY-HYBRID] source=MINER status=UNAVAILABLE error={type(e).__name__}:{e}")
    return out


def run_edge_registry_v1(*, dashboard_module, stat_out: dict, reliability_out: dict | None=None, log_func=print, hard_fail=True):
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        g=cache.get("games") if isinstance(cache,dict) else None
        if g is None or not isinstance(g,pd.DataFrame) or g.empty:
            raise RuntimeError("EDGE_REGISTRY_V1 requires NCAAF spread research cache")
        keys=_physical_key(g); dup=int(keys[keys.ne("")&keys.ne("nan")].duplicated().sum())
        if dup:
            raise RuntimeError(f"physical game duplication rows={dup}")
        system_history=getattr(dashboard_module,"_V143_SYSTEM_HISTORY_CACHE",{})
        miner_registry=cache.get("system_miner_v2") or {}
        entries=[]
        entries.append(_stat_tail_entry(stat_out,g))
        entries.extend(_system_memory_entries(system_history))
        entries.extend(_miner_entries(miner_registry))
        if len(entries)<3:
            raise RuntimeError(f"too few edge sources={len(entries)}")
        entries=sorted(entries,key=_entry_sort_key)
        counts={s:sum(e.get("strength")==s for e in entries) for s in ("STRONG_SHADOW","MODERATE_SHADOW","WEAK_SHADOW","RESEARCH_ONLY")}
        source_counts={s:sum(e.get("source")==s for e in entries) for s in ("STAT","BIGAL","PATHI","MINER")}
        log_func(f"[EDGE-REGISTRY-PREFLIGHT] status=PASS source_tag={EDGE_REGISTRY_SOURCE_TAG} games={len(g)} physical_game_duplicates=0 sources={source_counts} strength_counts={counts} no_master_model=TRUE production_authority=0")
        for e in entries[:30]:
            _log_entry(e,log_func)
        hybrids=_hybrid_from_stat(g,stat_out,dashboard_module,log_func)
        # Edge rarity is a first-class diagnostic for the STAT tail.
        p=((stat_out or {}).get("primary") or {})
        prior=((p.get("prior") or {}).get("mask"))
        test=p.get("test_mask")
        season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
        disc_eligible=np.isin(season,np.asarray(DISCOVERY_SEASONS,float))
        conf_eligible=season==float(CONFIRM_SEASON)
        prior_n=int(np.asarray(prior,bool).sum()) if prior is not None else 0
        test_n=int(np.asarray(test,bool).sum()) if test is not None else 0
        log_func(f"[EDGE-REGISTRY-RARITY] source=STAT_CONSENSUS_TAIL discovery_flagged={prior_n} discovery_eligible={int(disc_eligible.sum())} discovery_rate={(prior_n/max(1,int(disc_eligible.sum()))):.4f} confirm_2026_flagged={test_n} confirm_2026_eligible={int(conf_eligible.sum())} confirm_rate={(test_n/max(1,int(conf_eligible.sum()))):.4f}")
        log_func(f"[EDGE-REGISTRY-CONTRACT] status=PASS source_tag={EDGE_REGISTRY_SOURCE_TAG} registered_edges={len(entries)} hybrids_evaluated={len(hybrids)} peer_generators=STAT,BIGAL,PATHI,MINER evidence_strength_not_hit_rate=TRUE systems_not_probability_weighted=TRUE zero_production_authority=TRUE")
        return {"status":"PASS","source_tag":EDGE_REGISTRY_SOURCE_TAG,"entries":entries,"hybrids":hybrids,"production_authority":0}
    except Exception as e:
        log_func(f"[EDGE-REGISTRY-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}
