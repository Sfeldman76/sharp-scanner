"""EDGE_COMPLEMENTARITY_V1 — test whether peer NCAAF spread edges add independent value.

Research-only; zero production authority.

This module extends Edge Topology V1 in three ways:
1) It tests named source pairs (STAT, Big Al, Pathi, Miner) rather than only
   counting how many sources are active.
2) It decomposes Miner rules into predeclared information signatures so we can
   ask *what kind* of Miner evidence complements another source.
3) It collapses Miner systems with identical game masks and directions before
   analysis so redundant rules cannot masquerade as independent confirmation.

Discovery (2023-2025) is descriptive/hypothesis-generating because the published
Miner inventory/evidence grades were selected from that history. 2026 is
out-of-discovery confirmation, but not post-freeze prospective evidence.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any
import hashlib
import math
import numpy as np
import pandas as pd

import edge_topology_v1 as et

EDGE_COMPLEMENTARITY_V1_SOURCE_TAG = "edge-complementarity-v1-orthogonal-source-signatures"
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026
MIN_SAMPLE_LOG = 5

# Map the System Miner atom families into broader, human-interpretable sources
# of information. MODEL_STATE is intentionally excluded as circular confirmation.
MINER_FAMILY_BUCKET = {
    "MARKET_ROLE": "MARKET",
    "MARKET_PRICE": "MARKET",
    "TOTAL_REGIME": "MARKET",
    "VENUE": "SCHEDULE",
    "SEASON_TIMING": "SCHEDULE",
    "REST": "SCHEDULE",
    "RIVALRY": "HISTORY",
    "MATCHUP_HISTORY": "HISTORY",
    "PRIOR_RESULT": "FORM",
    "ATS_FORM": "FORM",
    "SU_FORM": "FORM",
    "CONFERENCE": "IDENTITY",
    "CONFERENCE_PAIR": "IDENTITY",
    "COACH_ERA": "IDENTITY",
    "TEAM_SPECIFIC": "IDENTITY",
    "MODEL_STATE": "MODEL_STATE",
}


def _wilson(hit: float, n: int, z: float = 1.6448536269514722) -> Tuple[float, float]:
    """Two-sided ~90% Wilson interval, useful for showing small-sample uncertainty."""
    if n <= 0 or not np.isfinite(hit):
        return np.nan, np.nan
    p=float(hit); den=1.0+(z*z/n)
    ctr=(p+z*z/(2*n))/den
    half=(z*math.sqrt(max(0.0,p*(1-p)/n + z*z/(4*n*n))))/den
    return max(0.0,ctr-half), min(1.0,ctr+half)


def _safe_delta(a, b):
    a=float(a); b=float(b)
    return a-b if np.isfinite(a) and np.isfinite(b) else np.nan


def _strength_lookup(registry_out: dict) -> dict:
    return et._registry_strength(registry_out)


def _system_signature(sys: dict) -> str:
    fams=[str(x).upper() for x in (sys.get("families") or [])]
    buckets=[]
    for f in fams:
        b=MINER_FAMILY_BUCKET.get(f, "OTHER")
        if b == "MODEL_STATE":
            return "MODEL_STATE_EXCLUDED"
        if b not in buckets:
            buckets.append(b)
    # deterministic conceptual ordering
    order={"MARKET":0,"SCHEDULE":1,"FORM":2,"HISTORY":3,"IDENTITY":4,"OTHER":5}
    buckets=sorted(buckets,key=lambda x:order.get(x,9))
    return "+".join(buckets) if buckets else "OTHER"


def _miner_signature_votes(g: pd.DataFrame, dashboard_module, strength_lookup: dict, *, min_strength: int = 1):
    """Build one vote per Miner information signature after global exact-mask deduplication."""
    cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
    reg=(cache.get("system_miner_v2") or {}).get("spreads") or {}
    mg=cache.get("miner_games")
    n=len(g)
    if not isinstance(mg,pd.DataFrame) or len(mg)!=n:
        return {}, {"status":"UNAVAILABLE","reason":"MINER_FRAME_UNAVAILABLE"}
    atoms={a["name"]:np.asarray(a["mask"],bool) for a in dashboard_module._v1355_system_atoms(mg)}
    raw=[]; excluded=0; below_strength=0
    for sys in reg.get("systems") or []:
        sid=str(sys.get("system_id","")); cond=list(sys.get("conditions") or [])
        if any(str(c).startswith(("STAT_EDGE_","H2H_STAT_","TOTAL_MODEL_")) for c in cond):
            excluded+=1; continue
        st=strength_lookup.get(("MINER",sid),"RESEARCH_ONLY")
        if et.STRENGTH_RANK.get(st,0)<min_strength:
            below_strength+=1; continue
        sig=_system_signature(sys)
        if sig=="MODEL_STATE_EXCLUDED":
            excluded+=1; continue
        mm=np.ones(n,bool)
        for c in cond:
            mm &= atoms.get(c,np.zeros(n,bool))
        if not mm.any():
            continue
        direction=str(sys.get("direction","PLAY_ON")).upper()
        sign=-1.0 if direction=="FADE" else 1.0
        key=hashlib.sha1(mm.tobytes()+str(sign).encode()).hexdigest()
        raw.append({"id":sid,"mask":mm,"sign":sign,"hash":key,"strength":st,"conditions":cond,"signature":sig})

    # Global collapse: if two nominally different systems fire on exactly the same
    # games in the same direction, they are one piece of evidence for this test.
    # Prefer stronger evidence, then the simpler rule, then stable id order.
    uniq={}
    for r in sorted(raw,key=lambda x:(-et.STRENGTH_RANK.get(x["strength"],0),len(x["conditions"]),x["id"])):
        uniq.setdefault(r["hash"],r)
    unique=list(uniq.values())
    by_sig: Dict[str,List[dict]]={}
    raw_by_sig: Dict[str,int]={}
    for r in raw: raw_by_sig[r["signature"]]=raw_by_sig.get(r["signature"],0)+1
    for r in unique: by_sig.setdefault(r["signature"],[]).append(r)

    out={}
    for sig,systems in sorted(by_sig.items()):
        pos=np.zeros(n,bool); neg=np.zeros(n,bool)
        for r in systems:
            if r["sign"]>0: pos |= r["mask"]
            else: neg |= r["mask"]
        mixed=pos&neg
        vote=np.zeros(n,float); vote[pos&~mixed]=1.0; vote[neg&~mixed]=-1.0; vote[mixed]=np.nan
        out[sig]={
            "vote":vote,"mixed":mixed,"systems":systems,
            "raw_systems":raw_by_sig.get(sig,len(systems)),"unique_systems":len(systems),
            "duplicates_collapsed":raw_by_sig.get(sig,len(systems))-len(systems),
        }
    return out,{"status":"READY","signatures":len(out),"raw_systems":len(raw),"unique_systems":len(unique),
                "duplicates_collapsed":len(raw)-len(unique),"model_state_excluded":excluded,"below_strength":below_strength}


def _source_votes(g: pd.DataFrame, dashboard_module, stat_out: dict, registry_out: dict, sample: str):
    strength=_strength_lookup(registry_out)
    hist=getattr(dashboard_module,"_V143_SYSTEM_HISTORY_CACHE",{})
    bv,bm,_=et._source_vote_from_occurrences(g,hist,"BIGAL",strength,min_strength=1)
    pv,pm,_=et._source_vote_from_occurrences(g,hist,"PATHI",strength,min_strength=1)
    mv,mm,_,_=et._miner_vote(g,dashboard_module,strength,min_strength=1)
    sv,_=et._stat_vote(g,stat_out,"DISCOVERY" if sample=="DISCOVERY" else "CONFIRM_2026")
    return {
        "STAT":(sv,np.zeros(len(g),bool)),
        "BIGAL":(bv,bm),
        "PATHI":(pv,pm),
        "MINER":(mv,mm),
    }


def _active(vote: np.ndarray, mixed: np.ndarray) -> np.ndarray:
    v=np.asarray(vote,float); m=np.asarray(mixed,bool)
    return ~m & np.isfinite(v) & ~np.isclose(v,0.0)


def _pair_masks(av,am,bv,bm):
    aa=_active(av,am); ba=_active(bv,bm)
    asgn=np.sign(np.asarray(av,float)); bsgn=np.sign(np.asarray(bv,float))
    support=aa&ba&(asgn==bsgn)
    conflict=aa&ba&(asgn==-bsgn)
    a_only=aa&~ba
    b_only=ba&~aa
    return {"support":support,"conflict":conflict,"anchor_only":a_only,"partner_only":b_only,"anchor_active":aa,"partner_active":ba}


def _met_for(g,target,sign,mask,dashboard_module):
    clv,_=et._clv_vector(g,dashboard_module,np.asarray(sign,float))
    return et._metrics(target,np.asarray(sign,float),np.asarray(mask,bool),clv)


def _pair_result(g,target,sample_mask,anchor_name,partner_name,av,am,bv,bm,dashboard_module,sample_label,log_func=print):
    masks=_pair_masks(av,am,bv,bm)
    aa=np.asarray(sample_mask,bool)&masks["anchor_only"]
    sup=np.asarray(sample_mask,bool)&masks["support"]
    con=np.asarray(sample_mask,bool)&masks["conflict"]
    anchor_sign=np.sign(np.asarray(av,float))
    base=_met_for(g,target,anchor_sign,aa,dashboard_module)
    smet=_met_for(g,target,anchor_sign,sup,dashboard_module)
    cmet=_met_for(g,target,anchor_sign,con,dashboard_module)
    lo,hi=_wilson(smet.get("hit",np.nan),int(smet.get("n",0)))
    d_hit=_safe_delta(smet.get("hit",np.nan),base.get("hit",np.nan))
    d_signed=_safe_delta(smet.get("signed",np.nan),base.get("signed",np.nan))
    d_clv=_safe_delta(smet.get("clv",np.nan),base.get("clv",np.nan))
    anchor_active_n=int((np.asarray(sample_mask,bool)&masks["anchor_active"]).sum())
    overlap_n=int((np.asarray(sample_mask,bool)&(masks["support"]|masks["conflict"])).sum())
    overlap_rate=(overlap_n/max(1,anchor_active_n))
    agree_rate=(int(sup.sum())/max(1,overlap_n))
    if smet["n"]>=MIN_SAMPLE_LOG or base["n"]>=MIN_SAMPLE_LOG or cmet["n"]>=MIN_SAMPLE_LOG:
        log_func(
            f"[EDGE-COMPLEMENTARITY-V1-PAIR] sample={sample_label} anchor={anchor_name} partner={partner_name} "
            f"anchor_active_n={anchor_active_n} overlap_n={overlap_n} overlap_rate={overlap_rate:.4f} agreement_rate_on_overlap={agree_rate:.4f} "
            f"anchor_only_n={base['n']} anchor_only_hit={base['hit']:.4f} anchor_only_signed={base['signed']:+.3f} anchor_only_clv={base['clv']:+.3f} "
            f"support_n={smet['n']} support_hit={smet['hit']:.4f} support_wilson90=[{lo:.4f},{hi:.4f}] support_roi={smet['roi']:+.4f} "
            f"support_signed={smet['signed']:+.3f} support_clv={smet['clv']:+.3f} delta_hit_vs_anchor_only={d_hit:+.4f} "
            f"delta_signed_vs_anchor_only={d_signed:+.3f} delta_clv_vs_anchor_only={d_clv:+.3f} "
            f"conflict_n={cmet['n']} conflict_hit_anchor_side={cmet['hit']:.4f} conflict_signed_anchor_side={cmet['signed']:+.3f} production_authority=0"
        )
    return {"sample":sample_label,"anchor":anchor_name,"partner":partner_name,"anchor_only":base,"support":smet,"conflict":cmet,
            "delta_hit":d_hit,"delta_signed":d_signed,"delta_clv":d_clv,"wilson90":[lo,hi],
            "anchor_active_n":anchor_active_n,"overlap_n":overlap_n,"overlap_rate":overlap_rate,"agreement_rate_on_overlap":agree_rate}


def _triple_result(g,target,sample_mask,names,votes,dashboard_module,sample_label,log_func=print):
    acts=[]; signs=[]
    for name in names:
        v,m=votes[name]; acts.append(_active(v,m)); signs.append(np.sign(np.asarray(v,float)))
    m=np.asarray(sample_mask,bool)
    for a in acts: m &= a
    agree=m.copy()
    for s in signs[1:]: agree &= (s==signs[0])
    met=_met_for(g,target,signs[0],agree,dashboard_module)
    lo,hi=_wilson(met.get("hit",np.nan),int(met.get("n",0)))
    if met["n"]>=MIN_SAMPLE_LOG:
        log_func(f"[EDGE-COMPLEMENTARITY-V1-TRIPLE] sample={sample_label} sources={'+'.join(names)} n={met['n']} hit={met['hit']:.4f} wilson90=[{lo:.4f},{hi:.4f}] roi={met['roi']:+.4f} signed_market_error={met['signed']:+.3f} avg_clv={met['clv']:+.3f} production_authority=0")
    return {"sample":sample_label,"sources":list(names),**met,"wilson90":[lo,hi]}


def _repeatability_state(disc: dict|None, conf: dict|None) -> str:
    if not disc or not conf: return "INSUFFICIENT"
    ds=disc.get("support") or {}; cs=conf.get("support") or {}
    if int(ds.get("n",0))<20 or int(cs.get("n",0))<10: return "SMALL_SAMPLE"
    dh=float(disc.get("delta_hit",np.nan)); ch=float(conf.get("delta_hit",np.nan))
    dsg=float(ds.get("signed",np.nan)); csg=float(cs.get("signed",np.nan))
    if np.isfinite(dh) and np.isfinite(ch) and dh>0 and ch>0 and dsg>0 and csg>0:
        return "REPEATABLE_COMPLEMENT_CANDIDATE"
    if np.isfinite(dh) and np.isfinite(ch) and dh<0 and ch<0:
        return "REPEATED_NO_INCREMENT"
    return "MIXED_EVIDENCE"


def run_edge_complementarity_v1(*, dashboard_module, stat_out: dict, registry_out: dict, topology_out: dict|None=None, log_func=print, hard_fail=True):
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        g=cache.get("games") if isinstance(cache,dict) else None
        if g is None or not isinstance(g,pd.DataFrame) or g.empty:
            raise RuntimeError("EDGE_COMPLEMENTARITY_V1 requires spread research games")
        g=g.copy(); season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
        target=pd.to_numeric(g.get("Market_Error_Margin"),errors="coerce").to_numpy(float)
        if int(et._physical_key(g).duplicated().sum()):
            raise RuntimeError("physical game duplicates present")
        disc=np.isin(season,np.asarray(DISCOVERY_SEASONS,float)); conf=season==float(CONFIRM_SEASON)
        strength=_strength_lookup(registry_out)
        sigs,sig_diag=_miner_signature_votes(g,dashboard_module,strength,min_strength=1)
        log_func(
            f"[EDGE-COMPLEMENTARITY-V1-PREFLIGHT] status=PASS source_tag={EDGE_COMPLEMENTARITY_V1_SOURCE_TAG} games={len(g)} "
            f"miner_signatures={sig_diag.get('signatures',0)} miner_raw_systems={sig_diag.get('raw_systems',0)} "
            f"miner_unique_masks={sig_diag.get('unique_systems',0)} miner_duplicates_collapsed={sig_diag.get('duplicates_collapsed',0)} "
            f"miner_model_state_excluded={sig_diag.get('model_state_excluded',0)} discovery_role=DESCRIPTIVE_HYPOTHESIS_GENERATION "
            f"confirm_2026_role=OUT_OF_DISCOVERY_CONFIRMATION_NOT_POSTFREEZE_PROSPECTIVE production_authority=0"
        )
        for sig,rec in sorted(sigs.items()):
            ids=','.join(x['id'] for x in rec['systems'])
            log_func(f"[EDGE-COMPLEMENTARITY-V1-MINER-SIGNATURE] signature={sig} raw_systems={rec['raw_systems']} unique_masks={rec['unique_systems']} duplicates_collapsed={rec['duplicates_collapsed']} ids={ids} model_state_free=TRUE production_authority=0")

        samples={"DISCOVERY_2023_2025_DESCRIPTIVE":("DISCOVERY",disc),"CONFIRM_2026":("CONFIRM",conf)}
        pair_rows=[]
        # Predeclared named-source pair tests.
        named_pairs=[("STAT","BIGAL"),("STAT","PATHI"),("STAT","MINER"),("BIGAL","PATHI"),("BIGAL","MINER"),("PATHI","MINER")]
        per_sample_votes={}
        for label,(kind,smask) in samples.items():
            votes=_source_votes(g,dashboard_module,stat_out,registry_out,"DISCOVERY" if kind=="DISCOVERY" else "CONFIRM")
            per_sample_votes[label]=votes
            for a,b in named_pairs:
                av,am=votes[a]; bv,bm=votes[b]
                pair_rows.append(_pair_result(g,target,smask,a,b,av,am,bv,bm,dashboard_module,label,log_func))

            # STAT/BigAl/Pathi against Miner information signatures. Each Miner
            # system belongs to exactly one signature, so this does not double-count
            # an individual rule across conceptual buckets.
            for anchor in ("STAT","BIGAL","PATHI"):
                av,am=votes[anchor]
                for sig,rec in sorted(sigs.items()):
                    pair_rows.append(_pair_result(g,target,smask,anchor,f"MINER[{sig}]",av,am,rec['vote'],rec['mixed'],dashboard_module,label,log_func))

        # Compact repeated-evidence summaries for predeclared pairs/signatures.
        keys=sorted({(r['anchor'],r['partner']) for r in pair_rows})
        repeat=[]
        for key in keys:
            d=next((r for r in pair_rows if (r['anchor'],r['partner'])==key and r['sample'].startswith('DISCOVERY')),None)
            c=next((r for r in pair_rows if (r['anchor'],r['partner'])==key and r['sample']=='CONFIRM_2026'),None)
            state=_repeatability_state(d,c); repeat.append({"anchor":key[0],"partner":key[1],"state":state,"discovery":d,"confirmation":c})
            if state!="INSUFFICIENT":
                dn=int(((d or {}).get('support') or {}).get('n',0)); cn=int(((c or {}).get('support') or {}).get('n',0))
                dh=float((d or {}).get('delta_hit',np.nan)); ch=float((c or {}).get('delta_hit',np.nan))
                log_func(f"[EDGE-COMPLEMENTARITY-V1-REPEAT] anchor={key[0]} partner={key[1]} state={state} discovery_support_n={dn} confirm_support_n={cn} discovery_delta_hit={dh:+.4f} confirm_delta_hit={ch:+.4f} production_authority=0")

        # Predeclared three-source stacks. These are reported, never selected.
        triples=[]
        for label,(kind,smask) in samples.items():
            votes=per_sample_votes[label]
            for names in (("STAT","BIGAL","MINER"),("STAT","PATHI","MINER"),("BIGAL","PATHI","MINER"),("STAT","BIGAL","PATHI")):
                triples.append(_triple_result(g,target,smask,names,votes,dashboard_module,label,log_func))

        n_repeat=sum(r['state']=="REPEATABLE_COMPLEMENT_CANDIDATE" for r in repeat)
        log_func(
            f"[EDGE-COMPLEMENTARITY-V1-CONTRACT] status=PASS source_tag={EDGE_COMPLEMENTARITY_V1_SOURCE_TAG} "
            f"named_pairs={len(named_pairs)} miner_signatures={len(sigs)} pair_tests={len(pair_rows)} three_way_tests={len(triples)} "
            f"repeatable_complement_candidates={n_repeat} identical_miner_masks_collapsed=TRUE model_state_circularity_blocked=TRUE "
            f"tested_hypotheses_predeclared=TRUE confirmation_used_for_evaluation_only=TRUE zero_production_authority=TRUE"
        )
        return {"status":"PASS","source_tag":EDGE_COMPLEMENTARITY_V1_SOURCE_TAG,"pairs":pair_rows,"repeatability":repeat,
                "triples":triples,"miner_signatures":{k:{kk:vv for kk,vv in v.items() if kk not in ('vote','mixed','systems')} for k,v in sigs.items()},
                "production_authority":0}
    except Exception as e:
        log_func(f"[EDGE-COMPLEMENTARITY-V1-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}
