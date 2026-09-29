"""ATOMIC_EDGE_GRAPH_V1 — individual rule + mechanism combination research.

Research-only; zero production authority.

Purpose
-------
Preserve each Big Al, Pathi, Miner and STAT rule as an atomic evidence node,
then evaluate combinations only after redundancy is collapsed.  The graph
separates:
* rule identity (the exact system that fired),
* detailed mechanisms (KEY_NUMBER, SCHEDULE, FORM, etc.),
* broad independent mechanism groups,
* source identity (STAT / BIGAL / PATHI / MINER), and
* market confirmation through CLV as supporting evidence, never a veto.

Discovery 2023-2025 is descriptive/hypothesis-generating. 2026 is confirmation
only, not post-freeze prospective evidence. Nothing in this module has
production authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Any, Tuple
import hashlib
import math
import numpy as np
import pandas as pd

import edge_mechanism_matrix_v1 as emm
import edge_topology_v1 as et

ATOMIC_EDGE_GRAPH_V1_SOURCE_TAG = "atomic-edge-graph-v1-individual-rule-mechanism-stacks"
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026
BREAK_EVEN = 110.0 / 210.0
MIN_LOG_N = 3
MIN_ATOMIC_DISCOVERY_N = 8
MIN_ATOMIC_CONFIRM_N = 3
MIN_STACK_DISCOVERY_N = 8
MIN_STACK_CONFIRM_N = 3
MIN_PAIR_DISCOVERY_N = 8
MIN_PAIR_CONFIRM_N = 3

# Detailed mechanisms roll up to a smaller number of independent concepts.  The
# detailed tags are still retained in every atomic node and log line.
MECHANISM_GROUP_MAP = {
    "FOOTBALL_STATS": "STATISTICAL_MATCHUP",
    "STAT_EFFICIENCY": "STATISTICAL_MATCHUP",
    "STAT_PASSING": "STATISTICAL_MATCHUP",
    "STAT_RUSHING": "STATISTICAL_MATCHUP",
    "STAT_OPPONENT_ADJUSTED": "STATISTICAL_MATCHUP",
    "STAT_DOWN_CONVERSION_PROXY": "STATISTICAL_MATCHUP",
    "STAT_TURNOVERS": "STATISTICAL_MATCHUP",
    "STAT_TEMPO_PLAY_MIX": "STATISTICAL_MATCHUP",
    "STAT_MATCHUP": "STATISTICAL_MATCHUP",
    "STAT_CONTEXT": "STATISTICAL_MATCHUP",
    "MARKET_PRICE": "MARKET_STRUCTURE",
    "MARKET_ROLE": "MARKET_STRUCTURE",
    "KEY_NUMBER": "MARKET_STRUCTURE",
    "LINE_MOVEMENT": "MARKET_STRUCTURE",
    "CROSS_MARKET": "MARKET_STRUCTURE",
    "ROLE_REVERSAL": "MARKET_STRUCTURE",
    "SCHEDULE": "SITUATIONAL",
    "VENUE": "SITUATIONAL",
    "FORM": "TEAM_STATE",
    "HISTORY": "TEAM_STATE",
    "IDENTITY": "IDENTITY_CONTEXT",
    "BIGAL_OTHER": "OTHER",
    "PATHI_OTHER": "OTHER",
    "MINER_OTHER": "OTHER",
}


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _groups(mechs) -> set[str]:
    return {MECHANISM_GROUP_MAP.get(str(x), str(x)) for x in (mechs or []) if str(x)}


def _signed_mask_hash(vote: np.ndarray) -> str:
    """Hash exact signed firing pattern. NaN/mixed is distinct from inactive."""
    a = np.asarray(vote, float)
    code = np.zeros(len(a), dtype=np.int8)
    code[np.isfinite(a) & (a > 0)] = 1
    code[np.isfinite(a) & (a < 0)] = -1
    code[~np.isfinite(a)] = 2
    return hashlib.sha256(code.tobytes()).hexdigest()[:16]


def _canonical_rule_rows(rules_by_sample: dict, sample: str) -> List[dict]:
    out=[]
    rows=rules_by_sample[sample]
    for source in emm.SOURCE_ORDER:
        for r in rows.get(source,[]):
            rr=dict(r)
            rr["source"] = source
            rr["rule_key"] = f"{source}:{rr.get('id','UNKNOWN')}"
            rr["mechanisms"] = set(rr.get("mechanisms") or [])
            rr["mechanism_groups"] = _groups(rr["mechanisms"])
            rr["signed_hash"] = _signed_mask_hash(np.asarray(rr.get("vote"), float))
            out.append(rr)
    return out


def _redundancy_clusters(discovery_rules: List[dict], confirm_rules: List[dict]):
    """Collapse exact signed masks across sources/rules before counting evidence.

    Rule identity is preserved; a redundancy cluster is only the unit used when
    counting independent atomic evidence. STAT is included as well: if another
    rule literally has the exact same signed firing vector, that historical
    dependence should be visible rather than counted twice.
    """
    by_key={r["rule_key"]:r for r in discovery_rules}
    by_key_c={r["rule_key"]:r for r in confirm_rules}
    grouped={}
    for key,r in by_key.items():
        # A redundancy cluster must be identical in BOTH discovery and confirmation.
        # Otherwise two rules that happened to coincide historically could be
        # incorrectly collapsed even though they separate in the holdout season.
        ch=by_key_c.get(key, r).get("signed_hash", r["signed_hash"])
        h=f"{r['signed_hash']}:{ch}"
        grouped.setdefault(h,[]).append(key)
    clusters=[]
    rule_to_cluster={}
    for idx,(h,keys) in enumerate(sorted(grouped.items())):
        cid=f"RC{idx+1:03d}-{hashlib.sha256(h.encode()).hexdigest()[:8]}"
        sources=sorted({by_key[k]["source"] for k in keys})
        detailed=set(); groups=set()
        for k in keys:
            detailed |= set(by_key[k].get("mechanisms") or [])
            groups |= set(by_key[k].get("mechanism_groups") or [])
            rule_to_cluster[k]=cid
        # Representative vote is safe because masks/directions are identical.
        rep=by_key[keys[0]]
        repc=by_key_c.get(keys[0],rep)
        clusters.append({
            "cluster_id":cid,
            "signed_hash":h,
            "rule_keys":keys,
            "sources":sources,
            "mechanisms":detailed,
            "mechanism_groups":groups,
            "vote_discovery":np.asarray(rep["vote"],float),
            "vote_confirm":np.asarray(repc["vote"],float),
        })
    return clusters,rule_to_cluster


def _metrics(g,target,sign,mask,dashboard_module):
    return emm._metrics(g,target,sign,mask,dashboard_module)


def _robustness(g,target,sign,mask,dashboard_module):
    return emm._robustness(g,target,sign,mask,dashboard_module,discovery_only=True)


def _evidence_state(dm,dr,cm):
    dn=int(dm.get("n",0) or 0); cn=int(cm.get("n",0) or 0)
    if dn<MIN_ATOMIC_DISCOVERY_N or cn<MIN_ATOMIC_CONFIRM_N:
        return "SMALL_SAMPLE"
    dgood=_safe_float(dm.get("hit"))>BREAK_EVEN and _safe_float(dm.get("signed"))>0
    cgood=_safe_float(cm.get("hit"))>BREAK_EVEN and _safe_float(cm.get("signed"))>0
    rb=_safe_float(dr.get("remove_best_hit")); lo=_safe_float(dr.get("min_loso_hit"))
    robust=(not np.isfinite(rb) or rb>BREAK_EVEN) and (not np.isfinite(lo) or lo>0.50)
    if dgood and cgood and robust:
        cclv=emm._clv_class(cm)
        if cclv=="CONTRARIAN_MARKET_MOVED_AGAINST": return "REPEATABLE_CONTRARIAN_ATOMIC_EDGE"
        if cclv=="MARKET_CONFIRMED": return "REPEATABLE_MARKET_CONFIRMED_ATOMIC_EDGE"
        return "REPEATABLE_ATOMIC_EDGE"
    if dgood and not cgood: return "DISCOVERY_EDGE_NOT_CONFIRMED"
    if (not dgood) and cgood: return "CONFIRMATION_ONLY_NOT_ESTABLISHED"
    return "NO_REPEATABLE_EDGE"


def _rule_evidence(g,target,drules,crules,dashboard_module,log_func):
    season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    dmask=np.isin(season,np.asarray(DISCOVERY_SEASONS,float))
    cmask=season==float(CONFIRM_SEASON)
    cby={r["rule_key"]:r for r in crules}
    rows=[]
    for r in drules:
        key=r["rule_key"]; rr=cby.get(key,r)
        dv=np.asarray(r["vote"],float); cv=np.asarray(rr["vote"],float)
        da=np.isfinite(dv)&(dv!=0)&dmask; ca=np.isfinite(cv)&(cv!=0)&cmask
        dm=_metrics(g,target,np.sign(dv),da,dashboard_module)
        cm=_metrics(g,target,np.sign(cv),ca,dashboard_module)
        dr=_robustness(g,target,np.sign(dv),da,dashboard_module)
        state=_evidence_state(dm,dr,cm)
        detailed=sorted(r.get("mechanisms") or [])
        groups=sorted(r.get("mechanism_groups") or [])
        log_func(
            f"[ATOMIC-EDGE-V1-RULE] source={r['source']} rule={r.get('id')} signed_hash={r['signed_hash']} "
            f"mechanisms={'+'.join(detailed) or 'NONE'} mechanism_groups={'+'.join(groups) or 'NONE'} "
            f"discovery_n={dm['n']} discovery_hit={dm['hit']:.4f} discovery_roi={dm['roi']:+.4f} discovery_signed={dm['signed']:+.3f} discovery_clv={dm['clv']:+.3f} "
            f"remove_best_hit={dr['remove_best_hit']:.4f} min_loso_hit={dr['min_loso_hit']:.4f} fire_rate_ratio={dr['rate_ratio']:.4f} "
            f"confirm_n={cm['n']} confirm_hit={cm['hit']:.4f} confirm_roi={cm['roi']:+.4f} confirm_signed={cm['signed']:+.3f} confirm_clv={cm['clv']:+.3f} "
            f"clv_state={emm._clv_class(cm)} evidence_state={state} production_authority=0"
        )
        rows.append({"rule_key":key,"source":r["source"],"rule_id":r.get("id"),"discovery":dm,"confirm":cm,"robustness":dr,
                     "state":state,"mechanisms":detailed,"mechanism_groups":groups,"signed_hash":r["signed_hash"]})
    return rows


def _cluster_logs(clusters,log_func):
    dup=0; cross=0
    for c in clusters:
        if len(c["rule_keys"])>1:
            dup += len(c["rule_keys"])-1
            if len(c["sources"])>1: cross += 1
            log_func(
                f"[ATOMIC-EDGE-V1-REDUNDANCY] cluster={c['cluster_id']} rules={len(c['rule_keys'])} sources={'+'.join(c['sources'])} "
                f"rule_ids={'|'.join(c['rule_keys'])} mechanism_groups={'+'.join(sorted(c['mechanism_groups'])) or 'NONE'} "
                f"policy=COUNT_AS_ONE_ATOMIC_EVIDENCE_UNIT production_authority=0"
            )
    return dup,cross


def _mechanism_votes(clusters, sample: str, group: str):
    votes=[]
    for c in clusters:
        if group not in c["mechanism_groups"]: continue
        votes.append(c["vote_discovery"] if sample=="DISCOVERY" else c["vote_confirm"])
    if not votes:
        return np.zeros(0,float),np.zeros(0,bool)
    a=np.vstack(votes)
    pos=np.any(np.isfinite(a)&(a>0),axis=0); neg=np.any(np.isfinite(a)&(a<0),axis=0)
    mixed=pos&neg
    v=np.zeros(a.shape[1],float); v[pos&~neg]=1.0; v[neg&~pos]=-1.0; v[mixed]=np.nan
    return v,mixed


def _mechanism_evidence(g,target,clusters,dashboard_module,log_func):
    season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    dmask=np.isin(season,np.asarray(DISCOVERY_SEASONS,float)); cmask=season==float(CONFIRM_SEASON)
    groups=sorted({x for c in clusters for x in c["mechanism_groups"]})
    out=[]
    for group in groups:
        dv,_=_mechanism_votes(clusters,"DISCOVERY",group); cv,_=_mechanism_votes(clusters,"CONFIRM",group)
        if len(dv)==0: continue
        da=dmask&np.isfinite(dv)&(dv!=0); ca=cmask&np.isfinite(cv)&(cv!=0)
        dm=_metrics(g,target,np.sign(dv),da,dashboard_module); cm=_metrics(g,target,np.sign(cv),ca,dashboard_module)
        dr=_robustness(g,target,np.sign(dv),da,dashboard_module)
        state=_evidence_state(dm,dr,cm)
        log_func(
            f"[ATOMIC-EDGE-V1-MECHANISM] mechanism_group={group} discovery_n={dm['n']} discovery_hit={dm['hit']:.4f} "
            f"discovery_signed={dm['signed']:+.3f} discovery_clv={dm['clv']:+.3f} remove_best_hit={dr['remove_best_hit']:.4f} min_loso_hit={dr['min_loso_hit']:.4f} "
            f"confirm_n={cm['n']} confirm_hit={cm['hit']:.4f} confirm_signed={cm['signed']:+.3f} confirm_clv={cm['clv']:+.3f} "
            f"clv_state={emm._clv_class(cm)} evidence_state={state} production_authority=0"
        )
        out.append({"mechanism_group":group,"discovery":dm,"confirm":cm,"robustness":dr,"state":state})
    return out


def _cluster_active(c,sample):
    v=c["vote_discovery"] if sample=="DISCOVERY" else c["vote_confirm"]
    return np.isfinite(v)&(v!=0),np.sign(v)


def _independent_mechanism_matching(active_clusters: List[dict]) -> int:
    """Maximum one-to-one matching between evidence clusters and mechanism groups.

    One atomic rule/cluster can contribute at most ONE independent mechanism even
    when the rule contains several conditions. A mechanism group can likewise be
    counted at most once. This prevents a single multi-condition system from
    masquerading as multiple confirmations.
    """
    match={}
    def dfs(ci,seen):
        for grp in sorted(active_clusters[ci].get("mechanism_groups") or []):
            if grp in seen: continue
            seen.add(grp)
            if grp not in match or dfs(match[grp],seen):
                match[grp]=ci
                return True
        return False
    score=0
    for ci in range(len(active_clusters)):
        if dfs(ci,set()): score+=1
    return score


def _game_signatures(g,clusters,sample: str):
    """Return clean same-side atomic/mechanism signatures and conflict metadata."""
    n=len(g); out=[]
    for i in range(n):
        active=[]
        for c in clusters:
            v=c["vote_discovery"][i] if sample=="DISCOVERY" else c["vote_confirm"][i]
            if np.isfinite(v) and not np.isclose(v,0.0):
                active.append((c,float(np.sign(v))))
        if not active:
            out.append(None); continue
        signs={s for _,s in active}
        active_clusters=[c for c,_ in active]
        source_set=sorted({src for c,_ in active for src in c["sources"]})
        groups=sorted({m for c,_ in active for m in c["mechanism_groups"]})
        rules=sorted({rk for c,_ in active for rk in c["rule_keys"]})
        cluster_ids=sorted(c["cluster_id"] for c,_ in active)
        indep=_independent_mechanism_matching(active_clusters)
        if len(signs)==1:
            s=next(iter(signs))
            out.append({"clean":True,"sign":s,"cluster_ids":cluster_ids,"rules":rules,"sources":source_set,"groups":groups,
                        "atomic_count":len(cluster_ids),"mechanism_count":indep})
        else:
            out.append({"clean":False,"sign":np.nan,"cluster_ids":cluster_ids,"rules":rules,"sources":source_set,"groups":groups,
                        "atomic_count":len(cluster_ids),"mechanism_count":indep})
    return out


def _signature_metrics(g,target,signatures,dashboard_module,label,log_func):
    season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    smask=np.isin(season,np.asarray(DISCOVERY_SEASONS,float)) if label=="DISCOVERY" else season==float(CONFIRM_SEASON)
    # Exact mechanism-stack signature. It intentionally ignores rule names so
    # different rules expressing the same independent concepts aggregate.
    by_mech={}; by_atomic={}; conflicts={}
    for i,s in enumerate(signatures):
        if not smask[i] or not s: continue
        if not s["clean"]:
            k=(tuple(s["sources"]),tuple(s["groups"]))
            conflicts[k]=conflicts.get(k,0)+1
            continue
        mk=(tuple(s["groups"]),int(s["mechanism_count"]))
        ak=tuple(s["cluster_ids"])
        by_mech.setdefault(mk,[]).append((i,s["sign"]))
        by_atomic.setdefault(ak,[]).append((i,s["sign"],tuple(s["rules"])))

    mech_rows=[]
    for sigkey,vals in sorted(by_mech.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        sig,indep_count=sigkey
        n=len(vals)
        if n<MIN_LOG_N: continue
        mask=np.zeros(len(g),bool); sign=np.zeros(len(g),float)
        for i,s in vals: mask[i]=True; sign[i]=s
        met=_metrics(g,target,sign,mask,dashboard_module)
        log_func(
            f"[ATOMIC-EDGE-V1-MECH-STACK] sample={label} mechanisms={'+'.join(sig) or 'NONE'} independent_mechanism_count={indep_count} "
            f"n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed={met['signed']:+.3f} clv={met['clv']:+.3f} clv_state={emm._clv_class(met)} production_authority=0"
        )
        mech_rows.append({"signature":sig,"independent_mechanism_count":indep_count,"metrics":met})

    atomic_rows=[]
    for sig,vals in sorted(by_atomic.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        n=len(vals)
        threshold=MIN_PAIR_DISCOVERY_N if label=="DISCOVERY" else MIN_PAIR_CONFIRM_N
        if n<threshold or len(sig)<2: continue
        mask=np.zeros(len(g),bool); sign=np.zeros(len(g),float); aliases=set()
        for i,sgn,rr in vals:
            mask[i]=True; sign[i]=sgn; aliases.update(rr)
        met=_metrics(g,target,sign,mask,dashboard_module)
        log_func(
            f"[ATOMIC-EDGE-V1-EXACT-RULE-STACK] sample={label} clusters={'|'.join(sig)} rule_aliases={'|'.join(sorted(aliases))} atomic_evidence_count={len(sig)} "
            f"n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed={met['signed']:+.3f} clv={met['clv']:+.3f} clv_state={emm._clv_class(met)} "
            f"role=DESCRIPTIVE_EXACT_COMBINATION production_authority=0"
        )
        atomic_rows.append({"signature":sig,"metrics":met})

    for (sources,groups),n in sorted(conflicts.items(),key=lambda kv:-kv[1]):
        if n>=MIN_LOG_N:
            log_func(
                f"[ATOMIC-EDGE-V1-CONFLICT-STACK] sample={label} sources={'+'.join(sources)} mechanisms={'+'.join(groups)} n={n} "
                f"policy=RESEARCH_ONLY_NO_FORCED_DIRECTION production_authority=0"
            )
    return mech_rows,atomic_rows,conflicts


def _cross_sample_mechanism_stacks(drows,crows,log_func):
    d={(r["signature"],r.get("independent_mechanism_count",len(r["signature"]))):r["metrics"] for r in drows}
    c={(r["signature"],r.get("independent_mechanism_count",len(r["signature"]))):r["metrics"] for r in crows}
    out=[]
    for key in sorted(set(d)|set(c)):
        sig,indep_count=key
        dm=d.get(key,{"n":0,"hit":np.nan,"signed":np.nan,"clv":np.nan}); cm=c.get(key,{"n":0,"hit":np.nan,"signed":np.nan,"clv":np.nan})
        dn=int(dm.get("n",0)); cn=int(cm.get("n",0))
        if dn<MIN_STACK_DISCOVERY_N and cn<MIN_STACK_CONFIRM_N: continue
        dgood=dn>=MIN_STACK_DISCOVERY_N and _safe_float(dm.get("hit"))>BREAK_EVEN and _safe_float(dm.get("signed"))>0
        cgood=cn>=MIN_STACK_CONFIRM_N and _safe_float(cm.get("hit"))>BREAK_EVEN and _safe_float(cm.get("signed"))>0
        if dgood and cgood: state="REPEATS_BUT_SMALL_SAMPLE" if (dn<20 or cn<8) else "REPEATABLE_MECHANISM_STACK"
        elif dgood: state="DISCOVERY_ONLY"
        elif cgood: state="CONFIRMATION_ONLY"
        else: state="NO_REPEATABLE_EDGE"
        log_func(
            f"[ATOMIC-EDGE-V1-MECH-STACK-EVIDENCE] mechanisms={'+'.join(sig)} independent_mechanism_count={indep_count} state={state} "
            f"discovery_n={dn} discovery_hit={dm.get('hit',np.nan):.4f} discovery_signed={dm.get('signed',np.nan):+.3f} discovery_clv={dm.get('clv',np.nan):+.3f} "
            f"confirm_n={cn} confirm_hit={cm.get('hit',np.nan):.4f} confirm_signed={cm.get('signed',np.nan):+.3f} confirm_clv={cm.get('clv',np.nan):+.3f} "
            f"clv_is_support_not_veto=TRUE production_authority=0"
        )
        out.append({"signature":sig,"independent_mechanism_count":indep_count,"state":state,"discovery":dm,"confirm":cm})
    return out


def _pair_atomic_combos(g,target,clusters,dashboard_module,log_func):
    """Predeclared pair scan of non-redundant atomic clusters.

    This is intentionally exhaustive over cluster pairs, with discovery and 2026
    reported separately. It is not a promotion mechanism; it tells us which exact
    systems appear complementary and which merely duplicate/conflict.
    """
    season=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    periods={"DISCOVERY":np.isin(season,np.asarray(DISCOVERY_SEASONS,float)),"CONFIRM_2026":season==float(CONFIRM_SEASON)}
    rows=[]
    for a,b in combinations(clusters,2):
        for label,smask in periods.items():
            av=a["vote_discovery"] if label=="DISCOVERY" else a["vote_confirm"]
            bv=b["vote_discovery"] if label=="DISCOVERY" else b["vote_confirm"]
            aa=np.isfinite(av)&(av!=0); ba=np.isfinite(bv)&(bv!=0)
            overlap=smask&aa&ba; agree=overlap&(np.sign(av)==np.sign(bv)); conflict=overlap&(np.sign(av)==-np.sign(bv))
            threshold=MIN_PAIR_DISCOVERY_N if label=="DISCOVERY" else MIN_PAIR_CONFIRM_N
            if int(agree.sum())>=threshold:
                met=_metrics(g,target,np.sign(av),agree,dashboard_module)
                log_func(
                    f"[ATOMIC-EDGE-V1-RULE-PAIR] sample={label} cluster_a={a['cluster_id']} cluster_b={b['cluster_id']} "
                    f"rules_a={'|'.join(a['rule_keys'])} rules_b={'|'.join(b['rule_keys'])} "
                    f"mechanisms_a={'+'.join(sorted(a['mechanism_groups']))} mechanisms_b={'+'.join(sorted(b['mechanism_groups']))} "
                    f"n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed={met['signed']:+.3f} clv={met['clv']:+.3f} "
                    f"clv_state={emm._clv_class(met)} role=DESCRIPTIVE_PAIR_COMBINATION production_authority=0"
                )
                rows.append({"sample":label,"a":a["cluster_id"],"b":b["cluster_id"],"metrics":met})
            if int(conflict.sum())>=threshold:
                log_func(
                    f"[ATOMIC-EDGE-V1-RULE-PAIR-CONFLICT] sample={label} cluster_a={a['cluster_id']} cluster_b={b['cluster_id']} n={int(conflict.sum())} "
                    f"policy=RESEARCH_ONLY_NO_AUTOMATIC_WINNER production_authority=0"
                )
    return rows


def run_atomic_edge_graph_v1(*,dashboard_module,stat_out:dict,registry_out:dict,mechanism_matrix_out:dict|None=None,log_func=print,hard_fail=True):
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        g=cache.get("games") if isinstance(cache,dict) else None
        if not isinstance(g,pd.DataFrame) or g.empty: raise RuntimeError("spread research games unavailable")
        g=g.copy(); target=pd.to_numeric(g.get("Market_Error_Margin"),errors="coerce").to_numpy(float)
        if int(et._physical_key(g).duplicated().sum()): raise RuntimeError("physical game duplicates present")

        rules_d,_=emm._mechanism_rows(g,dashboard_module,stat_out,registry_out,"DISCOVERY")
        rules_c,_=emm._mechanism_rows(g,dashboard_module,stat_out,registry_out,"CONFIRM")
        drules=_canonical_rule_rows({"DISCOVERY":rules_d},"DISCOVERY")
        crules=_canonical_rule_rows({"CONFIRM":rules_c},"CONFIRM")
        if not drules: raise RuntimeError("no atomic rules available")

        # Strong source-coverage contract: every system source represented in the
        # mechanism matrix must produce atomic rule nodes here as well.
        counts={s:sum(r["source"]==s for r in drules) for s in emm.SOURCE_ORDER}
        if any(counts.get(s,0)==0 for s in ("STAT","BIGAL","PATHI","MINER")):
            raise RuntimeError(f"missing atomic source nodes counts={counts}")

        clusters,rule_to_cluster=_redundancy_clusters(drules,crules)
        duplicates,cross_source_duplicates=_cluster_logs(clusters,log_func)
        log_func(
            f"[ATOMIC-EDGE-V1-PREFLIGHT] status=PASS source_tag={ATOMIC_EDGE_GRAPH_V1_SOURCE_TAG} games={len(g)} atomic_rules={len(drules)} "
            f"rule_counts={counts} redundancy_clusters={len(clusters)} duplicates_collapsed={duplicates} cross_source_duplicate_clusters={cross_source_duplicates} "
            f"mechanism_groups={','.join(sorted({x for c in clusters for x in c['mechanism_groups']}))} discovery_role=DESCRIPTIVE_HYPOTHESIS_GENERATION "
            f"confirm_2026_role=OUT_OF_DISCOVERY_CONFIRMATION_NOT_POSTFREEZE_PROSPECTIVE clv_support_not_veto=TRUE production_authority=0"
        )

        rule_evidence=_rule_evidence(g,target,drules,crules,dashboard_module,log_func)
        mech_evidence=_mechanism_evidence(g,target,clusters,dashboard_module,log_func)
        pair_rows=_pair_atomic_combos(g,target,clusters,dashboard_module,log_func)

        dsig=_game_signatures(g,clusters,"DISCOVERY"); csig=_game_signatures(g,clusters,"CONFIRM")
        dmech,datomic,dconf=_signature_metrics(g,target,dsig,dashboard_module,"DISCOVERY",log_func)
        cmech,catomic,cconf=_signature_metrics(g,target,csig,dashboard_module,"CONFIRM_2026",log_func)
        stack_evidence=_cross_sample_mechanism_stacks(dmech,cmech,log_func)

        n_rule_repeat=sum(str(r["state"]).startswith("REPEATABLE") for r in rule_evidence)
        n_stack_repeat=sum(str(r["state"]).startswith("REPEATABLE") or r["state"]=="REPEATS_BUT_SMALL_SAMPLE" for r in stack_evidence)
        log_func(
            f"[ATOMIC-EDGE-V1-CONTRACT] status=PASS source_tag={ATOMIC_EDGE_GRAPH_V1_SOURCE_TAG} atomic_rules={len(drules)} redundancy_clusters={len(clusters)} "
            f"repeatable_atomic_rules={n_rule_repeat} repeated_mechanism_stacks={n_stack_repeat} rule_pair_records={len(pair_rows)} "
            f"individual_rule_identity_preserved=TRUE exact_duplicates_count_once=TRUE detailed_mechanisms_preserved=TRUE independent_mechanism_groups_preserved=TRUE "
            f"standalone_rules_allowed=TRUE multi_rule_combos_allowed=TRUE conflicts_researched_not_bet=TRUE clv_tracked=TRUE clv_is_support_not_veto=TRUE zero_production_authority=TRUE"
        )
        return {
            "status":"PASS","source_tag":ATOMIC_EDGE_GRAPH_V1_SOURCE_TAG,"rule_evidence":rule_evidence,"mechanism_evidence":mech_evidence,
            "clusters":[{k:v for k,v in c.items() if not str(k).startswith("vote_")} for c in clusters],"rule_to_cluster":rule_to_cluster,
            "pair_rows":pair_rows,"mechanism_stack_evidence":stack_evidence,"production_authority":0,
        }
    except Exception as e:
        log_func(f"[ATOMIC-EDGE-V1-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}
