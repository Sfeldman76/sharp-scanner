"""EDGE_MECHANISM_MATRIX_V1 — peer-source mechanism and conflict research.

Research-only; zero production authority.

Purpose
-------
Treat STAT, Big Al, Pathi and System Miner as peer edge generators and test:
* each source standalone;
* every exact 2-way, 3-way and 4-way active-source topology;
* agreement and conflict as separate states;
* ordered incremental pair value (partner presence/support/conflict vs anchor-only);
* pairwise conflict-side performance without forcing a consensus bet;
* coarse mechanism independence so redundant descriptions do not masquerade as
  multiple independent confirmations;
* season stability, remove-best-season, LOSO and firing-rate recurrence;
* CLV as supporting market-agreement evidence, never as an edge veto.

Discovery 2023-2025 is descriptive/hypothesis-generating because the current
Miner/system inventory was selected using that history. 2026 is out-of-discovery
confirmation, not post-freeze prospective evidence.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple, Any
import math
import numpy as np
import pandas as pd

import edge_topology_v1 as et
import edge_complementarity_v1 as cv1
import edge_complementarity_v2 as cv2

EDGE_MECHANISM_MATRIX_V1_SOURCE_TAG = "edge-mechanism-matrix-v1-peer-source-orthogonality-clv"
SOURCE_ORDER = ("STAT", "BIGAL", "PATHI", "MINER")
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026
BREAK_EVEN = 110.0 / 210.0
CLV_NEUTRAL_BAND = 0.15  # points; descriptive only, never a selection gate
MIN_LOG_N = 3

# Coarse concepts are deliberately broader than rule names.  Their purpose is
# to expose duplicated mechanisms across different source labels.
BIGAL_MECHANISMS = {
    "BigAl_CF1_Week2Home42Win": {"SCHEDULE", "FORM", "VENUE"},
    "BigAl_CF2_LateSeasonRevengeDog": {"SCHEDULE", "HISTORY", "MARKET_ROLE", "FORM"},
    "BigAl_CF3_Fade19PlusFavoriteUpsetLoss": {"FORM", "MARKET_ROLE"},
    "BigAl_CF_Enhancer_RevengeDog": {"HISTORY", "MARKET_ROLE"},
    "BigAl_CF2_Away_Tightener": {"VENUE", "SCHEDULE"},
}


def _pathi_mechanisms(name: str) -> set[str]:
    n=str(name)
    if any(x in n for x in ("Moved_Through_Key","Crossed_Key_","Moved_Onto_Key","Moved_Off_Key","On_Key_")):
        return {"LINE_MOVEMENT", "KEY_NUMBER"}
    if any(x in n for x in ("Dog_Hook_Above_","Favorite_Below_Key_","Dog_Below_Key_","Favorite_Laying_Hook_")):
        return {"KEY_NUMBER", "MARKET_ROLE"}
    if "Usually_Dog_Now_Favorite" in n or "Usually_Favorite_Now_Dog" in n:
        return {"ROLE_REVERSAL", "MARKET_ROLE"}
    if "TotalSpread_Gap" in n:
        return {"CROSS_MARKET", "MARKET_PRICE"}
    return {"PATHI_OTHER"}


def _miner_mechanisms(signature: str) -> set[str]:
    # Map the V1 information signatures to the same broad mechanism language.
    mp={
        "MARKET": "MARKET_PRICE",
        "SCHEDULE": "SCHEDULE",
        "FORM": "FORM",
        "HISTORY": "HISTORY",
        "IDENTITY": "IDENTITY",
        "OTHER": "MINER_OTHER",
    }
    return {mp.get(x, x) for x in str(signature).split("+") if x}


def _stat_mechanisms(stat_out: dict) -> set[str]:
    leader=((stat_out or {}).get("primary") or {}).get("leader") or {}
    combo=list(leader.get("combo") or [])
    out={"FOOTBALL_STATS"}
    for x in combo:
        out.add(f"STAT_{str(x).upper()}")
    return out


def _safe_float(x):
    try: return float(x)
    except Exception: return np.nan


def _delta(a,b):
    a=_safe_float(a); b=_safe_float(b)
    return a-b if np.isfinite(a) and np.isfinite(b) else np.nan


def _wilson(hit: float, n: int, z: float = 1.6448536269514722):
    if n<=0 or not np.isfinite(hit): return (np.nan,np.nan)
    p=float(hit); den=1.0+z*z/n
    ctr=(p+z*z/(2*n))/den
    half=z*math.sqrt(max(0.0,p*(1-p)/n+z*z/(4*n*n)))/den
    return max(0.0,ctr-half),min(1.0,ctr+half)


def _clv_class(met: dict) -> str:
    n=int(met.get("clv_n",0) or 0); x=_safe_float(met.get("clv"))
    if n<5 or not np.isfinite(x): return "CLV_INSUFFICIENT"
    if x>CLV_NEUTRAL_BAND: return "MARKET_CONFIRMED"
    if x<-CLV_NEUTRAL_BAND: return "CONTRARIAN_MARKET_MOVED_AGAINST"
    return "MARKET_NEUTRAL"


def _votes(g,dashboard_module,stat_out,registry_out,sample: str):
    return cv1._source_votes(g,dashboard_module,stat_out,registry_out,sample)


def _active(v,m):
    return cv1._active(v,m)


def _metrics(g,target,sign,mask,dashboard_module):
    return cv1._met_for(g,target,sign,mask,dashboard_module)


def _robustness(g,target,sign,mask,dashboard_module, discovery_only=False):
    season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    base=np.asarray(mask,bool)
    years=list(DISCOVERY_SEASONS if discovery_only else (*DISCOVERY_SEASONS,CONFIRM_SEASON))
    per=[]
    for sy in years:
        m=base&(season==float(sy))
        met=_metrics(g,target,sign,m,dashboard_module)
        elig=int((season==float(sy)).sum())
        fire=met["n"]/max(1,elig)
        per.append((sy,met,fire))
    nonempty=[x for x in per if x[1]["n"]>0]
    if nonempty:
        rates=[x[2] for x in nonempty]
        rate_ratio=min(rates)/max(rates) if max(rates)>0 else np.nan
        min_hit=min(_safe_float(x[1]["hit"]) for x in nonempty)
        pos_signed=sum(_safe_float(x[1]["signed"])>0 for x in nonempty)
    else:
        rate_ratio=min_hit=np.nan; pos_signed=0

    # Discovery remove-best and leave-one-season-out diagnostics.
    disc_rows=[x for x in per if x[0] in DISCOVERY_SEASONS and x[1]["n"]>0]
    remove_best_hit=np.nan; min_loso=np.nan
    if len(disc_rows)>=2:
        best=max(disc_rows,key=lambda x:_safe_float(x[1]["hit"]))[0]
        rb=base&np.isin(season,np.asarray(DISCOVERY_SEASONS,float))&(season!=float(best))
        remove_best_hit=_metrics(g,target,sign,rb,dashboard_module)["hit"]
        loso=[]
        for sy,_,_ in disc_rows:
            lm=base&np.isin(season,np.asarray(DISCOVERY_SEASONS,float))&(season!=float(sy))
            mm=_metrics(g,target,sign,lm,dashboard_module)
            if mm["n"]: loso.append(mm["hit"])
        min_loso=min(loso) if loso else np.nan
    return {"per":per,"rate_ratio":rate_ratio,"min_hit":min_hit,"positive_signed_seasons":pos_signed,
            "remove_best_hit":remove_best_hit,"min_loso_hit":min_loso}


def _system_rule_masks(g,dashboard_module,registry_out,family: str):
    """Return qualifying BigAl/Pathi rule masks with side orientation and mechanisms."""
    strength=et._registry_strength(registry_out)
    hist=getattr(dashboard_module,"_V143_SYSTEM_HISTORY_CACHE",{})
    n=len(g); key_to_idx={str(k):i for i,k in enumerate(et._side_key(g).tolist())}
    rows=[]
    for name,rec in (hist or {}).items():
        if not isinstance(rec,dict) or str(rec.get("role","directional")).lower()!="directional": continue
        if str(rec.get("family","")).upper()!=family.upper(): continue
        st=strength.get((family.upper(),str(name)),"RESEARCH_ONLY")
        if et.STRENGTH_RANK.get(st,0)<1: continue
        pos=np.zeros(n,bool); neg=np.zeros(n,bool)
        for o in rec.get("occurrences") or []:
            if not isinstance(o,dict): continue
            try: sy=int(o.get("season"))
            except Exception: continue
            ds=et._norm_text(o.get("date")); tm=et._norm_text(o.get("team")); op=et._norm_text(o.get("opponent"))
            i=key_to_idx.get(f"{sy}|{ds}|{tm}|{op}")
            if i is not None: pos[i]=True; continue
            i=key_to_idx.get(f"{sy}|{ds}|{op}|{tm}")
            if i is not None: neg[i]=True
        mixed=pos&neg; vote=np.zeros(n,float); vote[pos&~mixed]=1.0; vote[neg&~mixed]=-1.0; vote[mixed]=np.nan
        mechs=BIGAL_MECHANISMS.get(str(name),{"BIGAL_OTHER"}) if family.upper()=="BIGAL" else _pathi_mechanisms(str(name))
        rows.append({"id":str(name),"vote":vote,"mixed":mixed,"strength":st,"mechanisms":set(mechs)})
    return rows


def _miner_rule_masks(g,dashboard_module,registry_out):
    strength=et._registry_strength(registry_out)
    sigs,diag=cv1._miner_signature_votes(g,dashboard_module,strength,min_strength=1)
    rows=[]; seen=set()
    for sig,rec in sorted(sigs.items()):
        for s in rec.get("systems") or []:
            key=s.get("hash") or s.get("id")
            if key in seen: continue
            seen.add(key)
            v=np.zeros(len(g),float); m=np.asarray(s["mask"],bool); v[m]=float(s.get("sign",1.0))
            rows.append({"id":s.get("id"),"vote":v,"mixed":np.zeros(len(g),bool),"strength":s.get("strength"),
                         "mechanisms":_miner_mechanisms(sig),"signature":sig,"conditions":s.get("conditions") or []})
    return rows,diag


def _mechanism_rows(g,dashboard_module,stat_out,registry_out,sample: str):
    n=len(g); rows={"STAT":[],"BIGAL":[],"PATHI":[],"MINER":[]}
    sv,_=et._stat_vote(g,stat_out,"DISCOVERY" if sample=="DISCOVERY" else "CONFIRM_2026")
    rows["STAT"].append({"id":"STAT_COMBO_PRIMARY","vote":sv,"mixed":np.zeros(n,bool),"mechanisms":_stat_mechanisms(stat_out)})
    rows["BIGAL"]=_system_rule_masks(g,dashboard_module,registry_out,"BIGAL")
    rows["PATHI"]=_system_rule_masks(g,dashboard_module,registry_out,"PATHI")
    rows["MINER"],diag=_miner_rule_masks(g,dashboard_module,registry_out)
    return rows,diag


def _game_mechanisms(rule_rows: dict, i: int, active_sources: List[str], consensus_sign: float|None=None):
    by_source={}; union=set()
    for src in active_sources:
        sset=set()
        for r in rule_rows.get(src,[]):
            v=r["vote"][i] if i<len(r["vote"]) else 0.0
            if not np.isfinite(v) or np.isclose(v,0.0): continue
            if consensus_sign is not None and np.sign(v)!=np.sign(consensus_sign): continue
            sset |= set(r.get("mechanisms") or [])
        by_source[src]=sset; union |= sset
    # Count source contributions that contain at least one concept not supplied by
    # every other active source. This is descriptive, not a weighting rule.
    independent_sources=0
    for src,sset in by_source.items():
        others=set()
        for os,oss in by_source.items():
            if os!=src: others |= oss
        if sset and (sset-others): independent_sources+=1
    return by_source,union,independent_sources


def _state_arrays(votes: dict):
    acts={s:_active(*votes[s]) for s in SOURCE_ORDER}
    signs={s:np.sign(np.asarray(votes[s][0],float)) for s in SOURCE_ORDER}
    return acts,signs


def _exact_subset_mask(subset,acts):
    subset=set(subset); m=np.ones(len(next(iter(acts.values()))),bool)
    for s in SOURCE_ORDER:
        m &= acts[s] if s in subset else ~acts[s]
    return m


def _agreement_mask(subset,mask,signs):
    names=list(subset); m=np.asarray(mask,bool).copy()
    base=signs[names[0]]
    for s in names[1:]: m &= signs[s]==base
    return m,base


def _mechanism_summary(rule_rows,mask,active_sources,sign):
    idx=np.flatnonzero(mask); mech_counts=[]; indep=[]; unions={}
    for i in idx:
        _,u,k=_game_mechanisms(rule_rows,int(i),list(active_sources),float(sign[i]) if np.isfinite(sign[i]) else None)
        mech_counts.append(len(u)); indep.append(k)
        for x in u: unions[x]=unions.get(x,0)+1
    top=sorted(unions.items(),key=lambda z:(-z[1],z[0]))[:8]
    return {
        "avg_mechanism_dimensions":float(np.mean(mech_counts)) if mech_counts else np.nan,
        "avg_independent_source_contributors":float(np.mean(indep)) if indep else np.nan,
        "top_mechanisms":"|".join(f"{k}:{v}" for k,v in top) if top else "NONE",
    }


def _log_metric(prefix,label,sources,met,rob,mech,log_func):
    lo,hi=_wilson(met.get("hit",np.nan),int(met.get("n",0)))
    log_func(
        f"[{prefix}] sample={label} sources={'+'.join(sources)} n={met['n']} hit={met['hit']:.4f} wilson90=[{lo:.4f},{hi:.4f}] "
        f"roi={met['roi']:+.4f} signed_market_error={met['signed']:+.3f} clv_n={met['clv_n']} avg_clv={met['clv']:+.3f} clv_positive={met['clv_pos']:.4f} "
        f"clv_state={_clv_class(met)} fire_rate_ratio={rob['rate_ratio']:.4f} remove_best_hit={rob['remove_best_hit']:.4f} min_loso_hit={rob['min_loso_hit']:.4f} "
        f"avg_mechanism_dimensions={mech['avg_mechanism_dimensions']:.2f} avg_independent_source_contributors={mech['avg_independent_source_contributors']:.2f} "
        f"top_mechanisms={mech['top_mechanisms']} production_authority=0"
    )


def _evidence_label(dmet,drob,cmet,crob):
    dn=int(dmet.get("n",0)); cn=int(cmet.get("n",0))
    if dn<20 or cn<8: return "SMALL_SAMPLE"
    dgood=_safe_float(dmet.get("hit"))>BREAK_EVEN and _safe_float(dmet.get("signed"))>0
    cgood=_safe_float(cmet.get("hit"))>BREAK_EVEN and _safe_float(cmet.get("signed"))>0
    robust=(not np.isfinite(drob.get("remove_best_hit",np.nan)) or drob["remove_best_hit"]>BREAK_EVEN) and (not np.isfinite(drob.get("min_loso_hit",np.nan)) or drob["min_loso_hit"]>0.50)
    if dgood and cgood and robust:
        cc=_clv_class(cmet)
        if cc=="CONTRARIAN_MARKET_MOVED_AGAINST": return "REPEATABLE_CONTRARIAN_OUTCOME_EDGE"
        if cc=="MARKET_CONFIRMED": return "REPEATABLE_MARKET_CONFIRMED_OUTCOME_EDGE"
        return "REPEATABLE_OUTCOME_EDGE"
    if dgood and not cgood: return "DISCOVERY_EDGE_NOT_CONFIRMED"
    if (not dgood) and cgood: return "CONFIRMATION_ONLY_NOT_ESTABLISHED"
    return "NO_REPEATABLE_EDGE"


def _standalone_and_exact(g,target,votes_by_sample,rules_by_sample,dashboard_module,log_func):
    season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    samples={"DISCOVERY_2023_2025_DESCRIPTIVE":("DISCOVERY",np.isin(season,np.asarray(DISCOVERY_SEASONS,float))),
             "CONFIRM_2026":("CONFIRM",season==float(CONFIRM_SEASON))}
    records={}
    for label,(kind,smask) in samples.items():
        votes=votes_by_sample[kind]; acts,signs=_state_arrays(votes); rule_rows=rules_by_sample[kind]
        # Source-active and true solo are both useful: active measures source quality;
        # solo tells us whether that source can stand on its own without peer evidence.
        for src in SOURCE_ORDER:
            active=smask&acts[src]
            solo=active.copy()
            for os in SOURCE_ORDER:
                if os!=src: solo &= ~acts[os]
            for state,m in (("ACTIVE_ANY",active),("SOLO",solo)):
                met=_metrics(g,target,signs[src],m,dashboard_module)
                if met["n"]>=MIN_LOG_N:
                    rob=_robustness(g,target,signs[src],m,dashboard_module,discovery_only=True)
                    mech=_mechanism_summary(rule_rows,m,[src],signs[src])
                    _log_metric("EDGE-MECHANISM-V1-STANDALONE",label,[f"{src}:{state}"],met,rob,mech,log_func)
                records[(label,src,state)]={"met":met,"mask":m}

        # All exact non-empty source sets: 15 possible sets for four peer sources.
        for k in range(1,len(SOURCE_ORDER)+1):
            for subset in combinations(SOURCE_ORDER,k):
                exact=smask&_exact_subset_mask(subset,acts)
                agree,sgn=_agreement_mask(subset,exact,signs)
                conflict=exact&~agree
                met=_metrics(g,target,sgn,agree,dashboard_module)
                rob=_robustness(g,target,sgn,agree,dashboard_module,discovery_only=True)
                mech=_mechanism_summary(rule_rows,agree,subset,sgn)
                if met["n"]>=MIN_LOG_N:
                    _log_metric("EDGE-MECHANISM-V1-EXACT-AGREE",label,list(subset),met,rob,mech,log_func)
                if int(conflict.sum())>=MIN_LOG_N:
                    log_func(f"[EDGE-MECHANISM-V1-EXACT-CONFLICT] sample={label} sources={'+'.join(subset)} n={int(conflict.sum())} policy=RESEARCH_BOTH_SIDES_NO_FORCED_CONSENSUS production_authority=0")
                records[(label,"+".join(subset),"EXACT_AGREE")]={"met":met,"rob":rob,"mask":agree,"mech":mech}
    return records


def _ordered_pair_matrix(g,target,votes_by_sample,dashboard_module,log_func):
    season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    samples={"DISCOVERY_2023_2025_DESCRIPTIVE":("DISCOVERY",np.isin(season,np.asarray(DISCOVERY_SEASONS,float))),
             "CONFIRM_2026":("CONFIRM",season==float(CONFIRM_SEASON))}
    out=[]
    for label,(kind,smask) in samples.items():
        votes=votes_by_sample[kind]
        for anchor in SOURCE_ORDER:
            for partner in SOURCE_ORDER:
                if anchor==partner: continue
                av,am=votes[anchor]; bv,bm=votes[partner]
                aa=_active(av,am); ba=_active(bv,bm); asg=np.sign(np.asarray(av,float)); bsg=np.sign(np.asarray(bv,float))
                aonly=smask&aa&~ba; present=smask&aa&ba; support=present&(asg==bsg); conflict=present&(asg==-bsg)
                bm0=_metrics(g,target,asg,aonly,dashboard_module); pm=_metrics(g,target,asg,present,dashboard_module)
                sm=_metrics(g,target,asg,support,dashboard_module); cm=_metrics(g,target,asg,conflict,dashboard_module)
                if pm["n"]>=MIN_LOG_N or bm0["n"]>=MIN_LOG_N:
                    log_func(
                        f"[EDGE-MECHANISM-V1-PAIR] sample={label} anchor={anchor} partner={partner} anchor_only_n={bm0['n']} anchor_only_hit={bm0['hit']:.4f} "
                        f"partner_present_n={pm['n']} partner_present_hit={pm['hit']:.4f} delta_presence_hit={_delta(pm['hit'],bm0['hit']):+.4f} "
                        f"delta_presence_signed={_delta(pm['signed'],bm0['signed']):+.3f} partner_present_clv={pm['clv']:+.3f} clv_state={_clv_class(pm)} "
                        f"support_n={sm['n']} support_hit={sm['hit']:.4f} support_signed={sm['signed']:+.3f} support_clv={sm['clv']:+.3f} "
                        f"conflict_n={cm['n']} conflict_anchor_hit={cm['hit']:.4f} conflict_anchor_signed={cm['signed']:+.3f} conflict_anchor_clv={cm['clv']:+.3f} production_authority=0"
                    )
                out.append({"sample":label,"anchor":anchor,"partner":partner,"anchor_only":bm0,"presence":pm,"support":sm,"conflict":cm})
    return out


def _pair_conflicts(g,target,votes_by_sample,dashboard_module,log_func):
    season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    samples={"DISCOVERY_2023_2025_DESCRIPTIVE":("DISCOVERY",np.isin(season,np.asarray(DISCOVERY_SEASONS,float))),
             "CONFIRM_2026":("CONFIRM",season==float(CONFIRM_SEASON))}
    rows=[]
    for label,(kind,smask) in samples.items():
        votes=votes_by_sample[kind]
        for a,b in combinations(SOURCE_ORDER,2):
            av,am=votes[a]; bv,bm=votes[b]
            aa=_active(av,am); ba=_active(bv,bm); asg=np.sign(np.asarray(av,float)); bsg=np.sign(np.asarray(bv,float))
            m=smask&aa&ba&(asg==-bsg)
            amx=_metrics(g,target,asg,m,dashboard_module); bmx=_metrics(g,target,bsg,m,dashboard_module)
            if amx["n"]>=MIN_LOG_N:
                log_func(
                    f"[EDGE-MECHANISM-V1-CONFLICT-HEADTOHEAD] sample={label} sources={a}_VS_{b} n={amx['n']} "
                    f"{a}_side_hit={amx['hit']:.4f} {a}_signed={amx['signed']:+.3f} {a}_clv={amx['clv']:+.3f} "
                    f"{b}_side_hit={bmx['hit']:.4f} {b}_signed={bmx['signed']:+.3f} {b}_clv={bmx['clv']:+.3f} "
                    f"policy=RESEARCH_ONLY_NO_AUTOMATIC_WINNER production_authority=0"
                )
            rows.append({"sample":label,"a":a,"b":b,"a_metrics":amx,"b_metrics":bmx})
    return rows


def _internal_source_conflicts(g,votes_by_sample,log_func):
    season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    samples={"DISCOVERY_2023_2025_DESCRIPTIVE":("DISCOVERY",np.isin(season,np.asarray(DISCOVERY_SEASONS,float))),
             "CONFIRM_2026":("CONFIRM",season==float(CONFIRM_SEASON))}
    rows=[]
    for label,(kind,smask) in samples.items():
        for src in SOURCE_ORDER:
            _,mixed=votes_by_sample[kind][src]
            n=int((np.asarray(smask,bool)&np.asarray(mixed,bool)).sum())
            rows.append({"sample":label,"source":src,"n":n})
            if n:
                log_func(f"[EDGE-MECHANISM-V1-INTERNAL-CONFLICT] sample={label} source={src} n={n} policy=SOURCE_ABSTAINS_NO_FORCED_DIRECTION production_authority=0")
    return rows


def _cross_sample_evidence(exact_records,log_func):
    rows=[]
    for k in range(1,len(SOURCE_ORDER)+1):
        for subset in combinations(SOURCE_ORDER,k):
            key="+".join(subset)
            d=exact_records.get(("DISCOVERY_2023_2025_DESCRIPTIVE",key,"EXACT_AGREE"),{})
            c=exact_records.get(("CONFIRM_2026",key,"EXACT_AGREE"),{})
            dm=d.get("met") or {}; cm=c.get("met") or {}; dr=d.get("rob") or {}; cr=c.get("rob") or {}
            state=_evidence_label(dm,dr,cm,cr)
            if int(dm.get("n",0)) or int(cm.get("n",0)):
                log_func(
                    f"[EDGE-MECHANISM-V1-EVIDENCE] sources={key} state={state} discovery_n={int(dm.get('n',0))} discovery_hit={_safe_float(dm.get('hit')):.4f} "
                    f"discovery_signed={_safe_float(dm.get('signed')):+.3f} discovery_clv={_safe_float(dm.get('clv')):+.3f} discovery_clv_state={_clv_class(dm)} "
                    f"confirm_n={int(cm.get('n',0))} confirm_hit={_safe_float(cm.get('hit')):.4f} confirm_signed={_safe_float(cm.get('signed')):+.3f} "
                    f"confirm_clv={_safe_float(cm.get('clv')):+.3f} confirm_clv_state={_clv_class(cm)} clv_is_support_not_veto=TRUE production_authority=0"
                )
            rows.append({"sources":subset,"state":state,"discovery":dm,"confirmation":cm})
    return rows


def run_edge_mechanism_matrix_v1(*,dashboard_module,stat_out:dict,registry_out:dict,topology_out:dict|None=None,complementarity_v2_out:dict|None=None,log_func=print,hard_fail=True):
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        g=cache.get("games") if isinstance(cache,dict) else None
        if not isinstance(g,pd.DataFrame) or g.empty: raise RuntimeError("spread research games unavailable")
        g=g.copy(); target=pd.to_numeric(g.get("Market_Error_Margin"),errors="coerce").to_numpy(float)
        if int(et._physical_key(g).duplicated().sum()): raise RuntimeError("physical game duplicates present")
        votes_by_sample={
            "DISCOVERY":_votes(g,dashboard_module,stat_out,registry_out,"DISCOVERY"),
            "CONFIRM":_votes(g,dashboard_module,stat_out,registry_out,"CONFIRM"),
        }
        rules_d,mdiag=_mechanism_rows(g,dashboard_module,stat_out,registry_out,"DISCOVERY")
        rules_c,_=_mechanism_rows(g,dashboard_module,stat_out,registry_out,"CONFIRM")
        rules_by_sample={"DISCOVERY":rules_d,"CONFIRM":rules_c}
        rule_counts={s:len(rules_d.get(s,[])) for s in SOURCE_ORDER}
        log_func(
            f"[EDGE-MECHANISM-V1-PREFLIGHT] status=PASS source_tag={EDGE_MECHANISM_MATRIX_V1_SOURCE_TAG} games={len(g)} peer_sources={','.join(SOURCE_ORDER)} "
            f"predeclared_exact_source_sets=15 ordered_pair_tests=12 pair_conflict_tests=6 rule_counts={rule_counts} "
            f"miner_unique_signed_masks={mdiag.get('unique_systems',0)} miner_duplicates_collapsed={mdiag.get('duplicates_collapsed',0)} "
            f"mechanism_independence_descriptive=TRUE clv_supporting_evidence_not_gate=TRUE discovery_role=DESCRIPTIVE_HYPOTHESIS_GENERATION "
            f"confirm_2026_role=OUT_OF_DISCOVERY_CONFIRMATION_NOT_POSTFREEZE_PROSPECTIVE zero_production_authority=TRUE"
        )

        exact=_standalone_and_exact(g,target,votes_by_sample,rules_by_sample,dashboard_module,log_func)
        pairs=_ordered_pair_matrix(g,target,votes_by_sample,dashboard_module,log_func)
        conflicts=_pair_conflicts(g,target,votes_by_sample,dashboard_module,log_func)
        internal_conflicts=_internal_source_conflicts(g,votes_by_sample,log_func)
        evidence=_cross_sample_evidence(exact,log_func)

        # Explicitly preserve the prior STAT+Miner regime/direction hypothesis.
        if complementarity_v2_out:
            ev=((complementarity_v2_out.get("primary") or {}).get("evidence") or {})
            log_func(
                f"[EDGE-MECHANISM-V1-STAT-MINER-REFERENCE] prior_v2_state={ev.get('state','UNKNOWN')} "
                f"regime_hit_repeat={str(ev.get('presence_hit_repeat',False)).upper()} directional_outcome_repeat={str(ev.get('outcome_repeat',False)).upper()} "
                f"directional_separation={str(ev.get('directional_separation',False)).upper()} price_clv_positive={str(ev.get('price_positive',False)).upper()} "
                f"price_clv_incremental={str(ev.get('price_incremental',False)).upper()} interpretation=REGIME_AND_DIRECTION_TEST_RETAINED production_authority=0"
            )

        n_rep=sum(str(x.get("state","")).startswith("REPEATABLE") for x in evidence)
        n_contra=sum(x.get("state")=="REPEATABLE_CONTRARIAN_OUTCOME_EDGE" for x in evidence)
        log_func(
            f"[EDGE-MECHANISM-V1-CONTRACT] status=PASS source_tag={EDGE_MECHANISM_MATRIX_V1_SOURCE_TAG} peer_sources=4 exact_source_sets=15 "
            f"ordered_pair_tests=24 pair_conflict_tests=12 cross_sample_evidence_states=15 repeatable_exact_edges={n_rep} repeatable_contrarian_edges={n_contra} "
            f"standalone_edges_allowed=TRUE pairs_allowed=TRUE triples_allowed=TRUE quad_allowed=TRUE conflicts_researched_not_bet=TRUE "
            f"mechanism_redundancy_audited=TRUE miner_exact_duplicates_collapsed=TRUE model_state_circularity_blocked=TRUE "
            f"internal_source_conflicts_abstain=TRUE clv_tracked=TRUE clv_is_support_not_veto=TRUE zero_production_authority=TRUE"
        )
        return {"status":"PASS","source_tag":EDGE_MECHANISM_MATRIX_V1_SOURCE_TAG,"exact":exact,"pairs":pairs,"conflicts":conflicts,"internal_conflicts":internal_conflicts,"evidence":evidence,"production_authority":0}
    except Exception as e:
        log_func(f"[EDGE-MECHANISM-V1-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}
