"""Shared Sports Edge Authority V1.

This is the cross-sport decision standard extracted from the NCAAF Production V1
architecture.  Sports provide their own fair-value models, statistical selectors,
systems, and market adapters; this module owns the invariants that should NOT be
re-invented when the sport changes:

1) Prediction/fair-value models are separate from wagering authority.
2) Evidence variants that express the same mechanism collapse to one family/vote.
3) Discovery selects/fixes a representative; confirmation decides whether the
   family can receive authority.  Confirmation never re-optimizes the variant.
4) Independent mechanisms may strengthen a decision; correlated aliases may not.
5) Opposing validated mechanisms force PASS — CONFLICT.
6) A sport/market may remain MODEL ONLY.  The framework never requires a bet.
7) Prospective evidence remains separate from historical confirmation.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

SOURCE_TAG = "sports-edge-authority-v1.0-cross-sport-standard-20261002"
FRAMEWORK_VERSION = "SPORTS_EDGE_AUTHORITY_V1"
BREAK_EVEN_110 = 110.0 / 210.0


def _num(x):
    try:
        z=float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def sha256_json(x: Any) -> str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def wilson95(wins:int,n:int)->list[float|None]:
    if n <= 0:
        return [None,None]
    z=1.959963984540054
    p=wins/n
    den=1+(z*z/n)
    ctr=(p+z*z/(2*n))/den
    half=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return [round(max(0.0,ctr-half),6),round(min(1.0,ctr+half),6)]


def record(results: Iterable[float], profit_if_win: Iterable[float] | None = None) -> dict:
    """Grade 1=win, 0=loss, .5/NaN=push/unknown; ROI is per one unit risked."""
    rr=np.asarray(list(results),float)
    if profit_if_win is None:
        pp=np.full(len(rr),100.0/110.0,float)
    else:
        pp=np.asarray(list(profit_if_win),float)
        if len(pp)!=len(rr):
            raise ValueError("profit_if_win length mismatch")
    valid=np.isfinite(rr)
    rr=rr[valid]; pp=pp[valid]
    wins=int(np.sum(rr==1)); losses=int(np.sum(rr==0)); pushes=int(np.sum((rr!=1)&(rr!=0)))
    graded=wins+losses
    profit=float(np.sum(np.where(rr==1,pp,np.where(rr==0,-1.0,0.0)))) if len(rr) else 0.0
    return {
        "n":int(len(rr)),"graded_n":int(graded),"wins":wins,"losses":losses,"pushes":pushes,
        "hit_rate":round(wins/graded,6) if graded else None,
        "roi_per_unit":round(profit/len(rr),6) if len(rr) else None,
        "wilson95":wilson95(wins,graded) if graded else [None,None],
    }


def family_gate(discovery:dict, confirmation:dict, *, market:str,
                min_discovery_n:int, min_confirmation_n:int,
                min_discovery_positive_seasons:int=2,
                min_confirmation_positive_seasons:int=1,
                discovery_positive_seasons:int=0,
                confirmation_positive_seasons:int=0,
                require_confirm_hit_above_break_even:bool=True) -> dict:
    """Predeclared historical promotion gate.  H2H emphasizes ROI over hit rate."""
    dr=_num(discovery.get("roi_per_unit")); cr=_num(confirmation.get("roi_per_unit"))
    dh=_num(discovery.get("hit_rate")); ch=_num(confirmation.get("hit_rate"))
    reasons=[]
    if int(discovery.get("n") or 0) < int(min_discovery_n): reasons.append("DISCOVERY_N")
    if not math.isfinite(dr) or dr <= 0: reasons.append("DISCOVERY_ROI")
    if int(discovery_positive_seasons) < int(min_discovery_positive_seasons): reasons.append("DISCOVERY_SEASON_STABILITY")
    if int(confirmation.get("n") or 0) < int(min_confirmation_n): reasons.append("CONFIRM_N")
    if not math.isfinite(cr) or cr <= 0: reasons.append("CONFIRM_ROI")
    if int(confirmation_positive_seasons) < int(min_confirmation_positive_seasons): reasons.append("CONFIRM_SEASON_STABILITY")
    if str(market).upper() in {"SPREADS","TOTALS"}:
        if not math.isfinite(dh) or dh <= BREAK_EVEN_110: reasons.append("DISCOVERY_HIT_BELOW_BREAK_EVEN")
        if require_confirm_hit_above_break_even and (not math.isfinite(ch) or ch <= BREAK_EVEN_110): reasons.append("CONFIRM_HIT_BELOW_BREAK_EVEN")
    return {"status":"PASS" if not reasons else "HOLD","reasons":reasons,
            "market":str(market).upper(),"break_even_110":round(BREAK_EVEN_110,6)}


def choose_discovery_representative(candidates:list[dict]) -> dict | None:
    """Choose using discovery only. Prefer robust pass, then larger sample, then simpler/lower threshold."""
    eligible=[c for c in candidates if (c.get("discovery_gate") or {}).get("status")=="PASS"]
    if not eligible:
        return None
    def key(c):
        d=c.get("discovery") or {}
        complexity=float(c.get("complexity",1) or 1)
        threshold=float(c.get("threshold",0) or 0)
        return (int(d.get("n") or 0),-complexity,-threshold)
    return sorted(eligible,key=key,reverse=True)[0]


def collapse_family_variants(variants:list[dict]) -> list[dict]:
    grouped={}
    for v in variants:
        grouped.setdefault(str(v.get("mechanism_family_id")),[]).append(v)
    out=[]
    for fid,vs in sorted(grouped.items()):
        rep=choose_discovery_representative(vs)
        if rep is None:
            # Retain the largest discovery sample for diagnostics only.
            rep=sorted(vs,key=lambda x:int((x.get("discovery") or {}).get("n") or 0),reverse=True)[0]
        z=dict(rep)
        z["family_variant_count"]=len(vs)
        z["family_variant_ids"]=[str(x.get("variant_id")) for x in vs]
        z["correlation_policy"]="ONE_MECHANISM_FAMILY_ONE_VOTE"
        out.append(z)
    return out


def resolve_votes(votes:list[dict], *, strong_play_allowed:bool=True) -> dict:
    """Resolve one vote per independent mechanism family."""
    uniq={}
    for v in votes:
        fid=str(v.get("mechanism_family_id") or "")
        direction=int(np.sign(_num(v.get("direction")))) if math.isfinite(_num(v.get("direction"))) else 0
        if not fid or direction==0:
            continue
        # same family cannot inflate evidence; conflicting aliases invalidate family
        if fid in uniq and int(uniq[fid]["direction"])!=direction:
            uniq[fid]={"mechanism_family_id":fid,"direction":0,"label":fid,"conflicted_aliases":True}
        elif fid not in uniq:
            uniq[fid]={**v,"direction":direction}
    clean=[v for v in uniq.values() if int(v.get("direction") or 0)!=0]
    alias_conflicts=[v for v in uniq.values() if int(v.get("direction") or 0)==0]
    dirs=sorted(set(int(v["direction"]) for v in clean))
    if alias_conflicts or len(dirs)>1:
        return {"decision":"PASS_CONFLICT","action":"PASS — CONFLICT","direction":0,
                "independent_mechanisms":len(clean),"families":sorted(uniq),"votes":clean,
                "reason":"VALIDATED_EDGE_CONFLICT" if len(dirs)>1 else "INTRA_FAMILY_ALIAS_CONFLICT"}
    if not clean:
        return {"decision":"MODEL_ONLY","action":"MODEL ONLY","direction":0,
                "independent_mechanisms":0,"families":[],"votes":[],"reason":"NO_VALIDATED_EDGE_FAMILY"}
    n=len(clean); direction=dirs[0]
    if n>=2 and strong_play_allowed:
        return {"decision":"EDGE_MULTI","action":"STRONG PLAY","direction":direction,
                "independent_mechanisms":n,"families":sorted(uniq),"votes":clean,"reason":"MULTIPLE_INDEPENDENT_MECHANISMS"}
    return {"decision":"EDGE_SINGLE" if n==1 else "EDGE_MULTI_NO_BONUS","action":"PLAY","direction":direction,
            "independent_mechanisms":n,"families":sorted(uniq),"votes":clean,"reason":"VALIDATED_EDGE_FAMILY"}


def framework_contract() -> dict:
    x={
        "framework_version":FRAMEWORK_VERSION,"source_tag":SOURCE_TAG,
        "prediction_and_betting_authority_separate":True,
        "family_collapse":"ONE_MECHANISM_FAMILY_ONE_VOTE",
        "discovery_selects_confirmation_only_validates":True,
        "conflict_rule":"PASS_CONFLICT",
        "multi_rule":"ONLY_INDEPENDENT_MECHANISMS_MAY_ESCALATE",
        "model_only_is_valid_outcome":True,
        "prospective_evidence_separate":True,
    }
    x["contract_sha256"]=sha256_json(x)
    return x


def _self_test():
    assert resolve_votes([])["action"]=="MODEL ONLY"
    assert resolve_votes([{"mechanism_family_id":"A","direction":1}])["action"]=="PLAY"
    assert resolve_votes([{"mechanism_family_id":"A","direction":1},{"mechanism_family_id":"B","direction":1}])["action"]=="STRONG PLAY"
    assert resolve_votes([{"mechanism_family_id":"A","direction":1},{"mechanism_family_id":"B","direction":-1}])["action"]=="PASS — CONFLICT"
    # aliases do not create a second vote
    assert resolve_votes([{"mechanism_family_id":"A","direction":1},{"mechanism_family_id":"A","direction":1}])["independent_mechanisms"]==1
    return {"status":"PASS",**framework_contract()}

if __name__=="__main__":
    print(json.dumps(_self_test(),sort_keys=True))
