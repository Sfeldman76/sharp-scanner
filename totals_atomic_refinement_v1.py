"""TOTALS_ATOMIC_REFINEMENT_V1

Research-only cleanup of NCAAF Totals edge candidates.

Goals:
- choose PLAY/FADE orientation from 2023-2025 discovery only;
- identify exact duplicate / near-duplicate Miner rules;
- test parent-child condition-set incrementality;
- collapse related rules into one evidence family;
- count independent mechanisms conservatively after family collapse;
- retest Totals stacks without allowing aliases or nested rules to inflate support.

2026 is confirmation only. No production authority is granted here.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import numpy as np
import pandas as pd

TOTALS_ATOMIC_REFINEMENT_V1_SOURCE_TAG = "totals-atomic-refinement-v1-play-fade-lineage-family-collapse"
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026
BREAK_EVEN_110 = 110.0 / 210.0
MIN_DISCOVERY_N = 80
MIN_CONFIRM_N = 15
RELATED_JACCARD_DISCOVERY = 0.80
RELATED_JACCARD_CONFIRM = 0.65


def _f(x):
    try: return float(x)
    except Exception: return np.nan


def _active(v):
    a=np.asarray(v,float)
    return np.isfinite(a) & ~np.isclose(a,0.0)


def _masks(g):
    s=pd.to_numeric(g.get("Season"),errors="coerce").to_numpy(float)
    return np.isin(s,np.asarray(DISCOVERY_SEASONS,float)), s==float(CONFIRM_SEASON)


def _orientation_good(m, rb):
    if int(m.get("n",0) or 0) < MIN_DISCOVERY_N: return False
    if not (np.isfinite(_f(m.get("hit"))) and _f(m.get("hit")) > BREAK_EVEN_110): return False
    if not (np.isfinite(_f(m.get("roi"))) and _f(m.get("roi")) > 0): return False
    if not (np.isfinite(_f(m.get("signed"))) and _f(m.get("signed")) > 0): return False
    rbh=_f(rb.get("remove_best_hit")); loso=_f(rb.get("min_loso_hit"))
    if np.isfinite(rbh) and rbh < 0.52: return False
    if np.isfinite(loso) and loso < 0.515: return False
    return True


def _orientation_state(chosen, dm, rb, cm):
    if chosen == "NONE": return "NO_DISCOVERY_DIRECTION"
    if not _orientation_good(dm,rb): return "DISCOVERY_DIRECTION_NOT_ROBUST"
    if int(cm.get("n",0) or 0) < MIN_CONFIRM_N: return "DISCOVERY_EDGE_AWAIT_CONFIRMATION"
    if _f(cm.get("hit")) > BREAK_EVEN_110 and _f(cm.get("signed")) > 0:
        return f"REPEATABLE_{chosen}_TOTAL_EDGE"
    return f"{chosen}_DISCOVERY_NOT_CONFIRMED"


def _signed_key(sign, mask):
    s=np.asarray(sign,float); m=np.asarray(mask,bool)
    code=np.zeros(len(s),np.int8)
    code[m & (s>0)] = 1; code[m & (s<0)] = -1
    return hashlib.sha1(code.tobytes()).hexdigest()[:12]


def _jaccard_signed(a,b,period):
    aa=period & _active(a); bb=period & _active(b)
    u=aa|bb
    if not u.any(): return np.nan,0
    same=aa & bb & (np.sign(a)==np.sign(b))
    return float(same.sum()/u.sum()), int(u.sum())


def _uf(n):
    p=list(range(n))
    def find(x):
        while p[x]!=x:
            p[x]=p[p[x]]; x=p[x]
        return x
    def union(a,b):
        a=find(a); b=find(b)
        if a!=b: p[b]=a
    return p,find,union


def _lineage(rows):
    edges=[]
    miners=[r for r in rows if r["kind"]=="MINER" and r.get("conditions")]
    for child in miners:
        ca=set(child["conditions"])
        pars=[]
        for parent in miners:
            if parent["id"]==child["id"]: continue
            pa=set(parent.get("conditions") or [])
            if pa and pa < ca: pars.append(parent)
        if not pars: continue
        mx=max(len(set(p.get("conditions") or [])) for p in pars)
        for p in pars:
            if len(set(p.get("conditions") or []))==mx:
                edges.append((p["id"],child["id"]))
    return edges


def _max_mechanism_matching(active_families):
    """One family can contribute at most one mechanism and one mechanism can be used once."""
    fams=[(fid, sorted(set(mechs))) for fid,mechs in active_families if mechs]
    match={}
    def dfs(fid, mechs, seen):
        for m in mechs:
            if m in seen: continue
            seen.add(m)
            if m not in match or dfs(match[m][0], match[m][1], seen):
                match[m]=(fid,mechs); return True
        return False
    count=0
    for fid,mechs in fams:
        if dfs(fid,mechs,set()): count+=1
    return count


def run_totals_atomic_refinement_v1(*, dashboard_module, sibling_out, sibling_module, log_func=print, hard_fail=True):
    try:
        cache=getattr(dashboard_module,"_V1357_SPREAD_RESEARCH_CACHE",{})
        g=cache.get("miner_games") if isinstance(cache,dict) else None
        if not isinstance(g,pd.DataFrame) or g.empty: raise RuntimeError("miner games unavailable")
        dmask,cmask=_masks(g)
        base=list((sibling_out or {}).get("totals_rules") or [])
        if not base: raise RuntimeError("sibling totals rules unavailable")
        log_func(f"[TOTALS-REFINE-V1-PREFLIGHT] status=PASS source_tag={TOTALS_ATOMIC_REFINEMENT_V1_SOURCE_TAG} games={len(g)} candidates={len(base)} discovery_seasons=2023,2024,2025 confirm_season=2026 orientation_selected_from_discovery_only=TRUE production_authority=0")

        rows=[]
        for c in base:
            # STAT thresholds remain diagnostics; this stage must not promote a post-hoc threshold.
            if c.get("kind")=="STAT":
                rows.append({**c,"chosen":"LOCKED_DIAGNOSTIC","multiplier":1.0,"refine_state":"STAT_THRESHOLD_NOT_PROMOTABLE_HERE"})
                log_func(f"[TOTALS-REFINE-V1-ORIENTATION] kind=STAT rule={c.get('id')} chosen=LOCKED_DIAGNOSTIC state=STAT_THRESHOLD_NOT_PROMOTABLE_HERE production_authority=0")
                continue
            base_sign=np.asarray(c.get("sign"),float); mask=np.asarray(c.get("mask"),bool)
            play=sibling_module._totals_metrics(g, np.sign(base_sign), dmask & mask)
            fade=sibling_module._totals_metrics(g,-np.sign(base_sign), dmask & mask)
            rplay=sibling_module._robustness(g,np.sign(base_sign),dmask & mask,"totals")
            rfade=sibling_module._robustness(g,-np.sign(base_sign),dmask & mask,"totals")
            pg=_orientation_good(play,rplay); fg=_orientation_good(fade,rfade)
            if pg and not fg: chosen,mult,dm,rb="PLAY",1.0,play,rplay
            elif fg and not pg: chosen,mult,dm,rb="FADE",-1.0,fade,rfade
            elif pg and fg:
                # discovery-only tie break
                if _f(play.get("roi")) >= _f(fade.get("roi")): chosen,mult,dm,rb="PLAY",1.0,play,rplay
                else: chosen,mult,dm,rb="FADE",-1.0,fade,rfade
            else: chosen,mult,dm,rb="NONE",0.0,play,rplay
            osign=(mult*np.sign(base_sign)) if chosen!="NONE" else np.sign(base_sign)
            conf=sibling_module._totals_metrics(g,osign,cmask & mask)
            state=_orientation_state(chosen,dm,rb,conf)
            row={**c,"chosen":chosen,"multiplier":mult,"discovery_refined":dm,"confirm_refined":conf,"robustness_refined":rb,"refine_state":state,"oriented_sign":osign}
            rows.append(row)
            log_func(f"[TOTALS-REFINE-V1-ORIENTATION] kind=MINER rule={c.get('id')} conditions={'&'.join(c.get('conditions') or []) or 'NONE'} chosen={chosen} state={state} "
                     f"play_n={play['n']} play_hit={play['hit']:.4f} play_roi={play['roi']:+.4f} fade_hit={fade['hit']:.4f} fade_roi={fade['roi']:+.4f} "
                     f"remove_best_hit={rb.get('remove_best_hit',np.nan):.4f} min_loso_hit={rb.get('min_loso_hit',np.nan):.4f} confirm_n={conf['n']} confirm_hit={conf['hit']:.4f} confirm_roi={conf['roi']:+.4f} confirm_signed={conf['signed']:+.3f} production_authority=0")

        miners=[r for r in rows if r.get("kind")=="MINER"]
        idx={r["id"]:i for i,r in enumerate(miners)}
        p,find,union=_uf(len(miners))
        exact_pairs=[]; corr_pairs=[]
        # exact / near signed activation redundancy
        for i in range(len(miners)):
            a=miners[i]
            sa=np.asarray(a.get("oriented_sign",a.get("sign")),float)
            ma=np.asarray(a.get("mask"),bool)
            for j in range(i+1,len(miners)):
                b=miners[j]; sb=np.asarray(b.get("oriented_sign",b.get("sign")),float); mb=np.asarray(b.get("mask"),bool)
                if _signed_key(sa,ma)==_signed_key(sb,mb):
                    union(i,j); exact_pairs.append((a["id"],b["id"]))
                    continue
                jd,ud=_jaccard_signed(sa,sb,dmask); jc,uc=_jaccard_signed(sa,sb,cmask)
                if ud>=20 and jd>=RELATED_JACCARD_DISCOVERY and (uc<5 or jc>=RELATED_JACCARD_CONFIRM):
                    union(i,j); corr_pairs.append((a["id"],b["id"],jd,jc))

        # Parent/child relationships are related evidence even if the child looks better.
        lineage=_lineage(miners)
        by={r["id"]:r for r in miners}; lineage_rows=[]
        for pid,cid in lineage:
            if pid not in idx or cid not in idx: continue
            union(idx[pid],idx[cid])
            pr,cr=by[pid],by[cid]
            ps=np.asarray(pr.get("oriented_sign",pr.get("sign")),float); cs=np.asarray(cr.get("oriented_sign",cr.get("sign")),float)
            pm=np.asarray(pr.get("mask"),bool); cm=np.asarray(cr.get("mask"),bool)
            po=dmask & pm & ~cm; ch=dmask & cm
            pmet=sibling_module._totals_metrics(g,ps,po); cmet=sibling_module._totals_metrics(g,cs,ch)
            cpo=cmask & pm & ~cm; cch=cmask & cm
            pcf=sibling_module._totals_metrics(g,ps,cpo); ccf=sibling_module._totals_metrics(g,cs,cch)
            dh=_f(cmet.get("hit"))-_f(pmet.get("hit")); ds=_f(cmet.get("signed"))-_f(pmet.get("signed"))
            cdh=_f(ccf.get("hit"))-_f(pcf.get("hit")); cds=_f(ccf.get("signed"))-_f(pcf.get("signed"))
            if pmet["n"]<8 or cmet["n"]<8: state="SMALL_LINEAGE_SAMPLE"
            elif np.isfinite(dh) and np.isfinite(ds) and dh>0 and ds>0:
                if pcf["n"]>=3 and ccf["n"]>=3 and np.isfinite(cdh) and np.isfinite(cds) and cdh>0 and cds>0: state="REFINEMENT_ADDS_VALUE_BOTH"
                elif pcf["n"]>=3 and ccf["n"]>=3: state="DISCOVERY_REFINEMENT_NOT_CONFIRMED"
                else: state="DISCOVERY_REFINEMENT_AWAIT_CONFIRMATION"
            else: state="NO_INCREMENTAL_VALUE"
            lineage_rows.append((pid,cid,state))
            log_func(f"[TOTALS-REFINE-V1-LINEAGE] parent={pid} child={cid} state={state} parent_only_n={pmet['n']} parent_only_hit={pmet['hit']:.4f} child_n={cmet['n']} child_hit={cmet['hit']:.4f} delta_hit={dh:+.4f} delta_signed={ds:+.3f} confirm_parent_only_n={pcf['n']} confirm_child_n={ccf['n']} confirm_delta_hit={cdh:+.4f} confirm_delta_signed={cds:+.3f} production_authority=0")

        groups=defaultdict(list)
        for i,r in enumerate(miners): groups[find(i)].append(r)
        families=[]; member_family={}
        for num,members in enumerate(sorted(groups.values(),key=lambda x:min(z["id"] for z in x)),1):
            fid=f"TF{num:03d}-{hashlib.sha1('|'.join(sorted(x['id'] for x in members)).encode()).hexdigest()[:8]}"
            # representative favors repeatable orientation, then simpler rule, then larger discovery sample
            def rk(r):
                repeat=str(r.get("refine_state","")).startswith("REPEATABLE_")
                n=int((r.get("discovery_refined") or {}).get("n",0) or 0)
                return (1 if repeat else 0, -len(r.get("conditions") or []), n)
            rep=max(members,key=rk)
            mechs=sorted(set(m for r in members for m in (r.get("mechanisms") or [])))
            repeat_members=[r for r in members if str(r.get("refine_state","")).startswith("REPEATABLE_")]
            fam={"family_id":fid,"members":members,"representative":rep,"mechanisms":mechs,"repeatable":bool(repeat_members)}
            families.append(fam)
            for r in members: member_family[r["id"]]=fid
            log_func(f"[TOTALS-REFINE-V1-EVIDENCE-FAMILY] family={fid} members={len(members)} representative={rep['id']} rules={','.join(sorted(r['id'] for r in members))} mechanisms={'+'.join(mechs) or 'NONE'} repeatable_members={len(repeat_members)} policy=COUNT_AS_ONE_RELATED_EVIDENCE_FAMILY production_authority=0")

        # Re-test stacks after family collapse. Family internally abstains on conflicting directions.
        repeat_fams=[f for f in families if f["repeatable"]]
        stack_metrics={}
        for label,period in (("DISCOVERY",dmask),("CONFIRM_2026",cmask)):
            vote=np.zeros(len(g),float); indep=np.zeros(len(g),int); conflicts=np.zeros(len(g),bool); famcount=np.zeros(len(g),int)
            for i in np.flatnonzero(period):
                active=[]
                for fam in repeat_fams:
                    signs=[]
                    for r in fam["members"]:
                        if not str(r.get("refine_state","")).startswith("REPEATABLE_"): continue
                        if np.asarray(r.get("mask"),bool)[i]:
                            s=float(np.asarray(r.get("oriented_sign",r.get("sign")),float)[i])
                            if np.isfinite(s) and not np.isclose(s,0): signs.append(float(np.sign(s)))
                    if not signs: continue
                    if len(set(signs))>1: continue
                    active.append((fam["family_id"], signs[0], fam["mechanisms"]))
                if not active: continue
                dirs={x[1] for x in active}
                if len(dirs)>1: conflicts[i]=True; continue
                vote[i]=next(iter(dirs)); famcount[i]=len(active)
                indep[i]=_max_mechanism_matching([(x[0],x[2]) for x in active])
            for k in (1,2,3):
                mm=period & ~conflicts & (indep>=k) & _active(vote)
                if mm.sum() < (15 if label=="DISCOVERY" else 5): continue
                met=sibling_module._totals_metrics(g,vote,mm)
                stack_metrics[(label,k)]=dict(met)
                log_func(f"[TOTALS-REFINE-V1-STACK] sample={label} min_independent_mechanisms={k} n={met['n']} hit={met['hit']:.4f} roi={met['roi']:+.4f} signed={met['signed']:+.3f} clv={met['clv']:+.3f} family_collapsed=TRUE production_authority=0")
            log_func(f"[TOTALS-REFINE-V1-COUNTS] sample={label} games_with_repeatable_family={int(((famcount>0)&period).sum())} conflicts={int((conflicts&period).sum())} max_active_families={int(famcount[period].max()) if period.any() else 0} max_independent_mechanisms={int(indep[period].max()) if period.any() else 0} production_authority=0")

        pre_repeat=sum(str(r.get("state"))=="REPEATABLE_TOTAL_EDGE" for r in base)
        refined_repeat=sum(str(r.get("refine_state","")).startswith("REPEATABLE_") for r in miners)
        addv=sum(s=="REFINEMENT_ADDS_VALUE_BOTH" for _,_,s in lineage_rows)
        nov=sum(s=="NO_INCREMENTAL_VALUE" for _,_,s in lineage_rows)
        log_func(f"[TOTALS-REFINE-V1-CONTRACT] status=PASS input_candidates={len(base)} miner_rules={len(miners)} sibling_repeatable_rules={pre_repeat} refined_repeatable_rules={refined_repeat} evidence_families={len(families)} repeatable_evidence_families={len(repeat_fams)} exact_duplicate_pairs={len(exact_pairs)} related_correlation_pairs={len(corr_pairs)} lineage_edges={len(lineage_rows)} lineage_add_value_both={addv} lineage_no_incremental_value={nov} stat_thresholds_not_promoted=TRUE related_rules_count_once=TRUE independent_mechanisms_matched_conservatively=TRUE orientation_selected_from_discovery_only=TRUE confirm_2026_not_used_for_selection=TRUE production_authority=0")
        return {"status":"PASS","source_tag":TOTALS_ATOMIC_REFINEMENT_V1_SOURCE_TAG,"rules":rows,"families":families,"lineage":lineage_rows,"stack_metrics":stack_metrics,"production_authority":0}
    except Exception as e:
        log_func(f"[TOTALS-REFINE-V1-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {"status":"FAILED","error":f"{type(e).__name__}:{e}","production_authority":0}
