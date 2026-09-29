"""EDGE_COMPLEMENTARITY_V2 — distinguish directional confirmation from regime detection.

Research-only; zero production authority.

V2 runs the frozen V1 complementarity analysis, then adds:
* partner-presence tests that ignore partner direction and ask whether a source
  identifies games where the anchor is unusually reliable;
* directional-separation tests comparing support versus conflict directly;
* discovery-season stability for STAT+Miner and Miner information signatures;
* rule-level Miner attribution against STAT after exact-mask/direction dedupe;
* separate outcome-edge and price/CLV evidence states.

Discovery 2023-2025 remains descriptive/hypothesis-generating. 2026 is
out-of-discovery confirmation, not post-freeze prospective evidence.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

import edge_topology_v1 as et
import edge_complementarity_v1 as v1

EDGE_COMPLEMENTARITY_V2_SOURCE_TAG = "edge-complementarity-v2-regime-vs-direction"
DISCOVERY_SEASONS = (2023, 2024, 2025)
CONFIRM_SEASON = 2026


def _f(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _delta(a,b):
    a=_f(a); b=_f(b)
    return a-b if np.isfinite(a) and np.isfinite(b) else np.nan


def _presence_result(g,target,sample_mask,anchor_name,partner_name,av,am,bv,bm,dashboard_module,sample_label,log_func=print):
    """Evaluate anchor side when partner is present, regardless of partner direction."""
    aa=v1._active(av,am); pa=v1._active(bv,bm)
    sm=np.asarray(sample_mask,bool)
    anchor_sign=np.sign(np.asarray(av,float))
    anchor_only=sm & aa & ~pa
    present=sm & aa & pa
    pair=v1._pair_masks(av,am,bv,bm)
    support=sm & pair['support']
    conflict=sm & pair['conflict']
    base=v1._met_for(g,target,anchor_sign,anchor_only,dashboard_module)
    pmet=v1._met_for(g,target,anchor_sign,present,dashboard_module)
    smet=v1._met_for(g,target,anchor_sign,support,dashboard_module)
    cmet=v1._met_for(g,target,anchor_sign,conflict,dashboard_module)
    out={
        'sample':sample_label,'anchor':anchor_name,'partner':partner_name,
        'anchor_only':base,'presence':pmet,'support':smet,'conflict':cmet,
        'delta_presence_hit':_delta(pmet.get('hit'),base.get('hit')),
        'delta_presence_signed':_delta(pmet.get('signed'),base.get('signed')),
        'delta_presence_clv':_delta(pmet.get('clv'),base.get('clv')),
        'support_minus_conflict_hit':_delta(smet.get('hit'),cmet.get('hit')),
        'support_minus_conflict_signed':_delta(smet.get('signed'),cmet.get('signed')),
        'support_minus_conflict_clv':_delta(smet.get('clv'),cmet.get('clv')),
    }
    if pmet['n']>=5 or base['n']>=5:
        log_func(
            f"[EDGE-COMPLEMENTARITY-V2-PRESENCE] sample={sample_label} anchor={anchor_name} partner={partner_name} "
            f"anchor_only_n={base['n']} anchor_only_hit={base['hit']:.4f} anchor_only_signed={base['signed']:+.3f} anchor_only_clv={base['clv']:+.3f} "
            f"partner_present_n={pmet['n']} partner_present_hit={pmet['hit']:.4f} partner_present_roi={pmet['roi']:+.4f} "
            f"partner_present_signed={pmet['signed']:+.3f} partner_present_clv={pmet['clv']:+.3f} "
            f"delta_hit_vs_anchor_only={out['delta_presence_hit']:+.4f} delta_signed_vs_anchor_only={out['delta_presence_signed']:+.3f} "
            f"delta_clv_vs_anchor_only={out['delta_presence_clv']:+.3f} "
            f"support_n={smet['n']} support_hit={smet['hit']:.4f} support_signed={smet['signed']:+.3f} support_clv={smet['clv']:+.3f} "
            f"conflict_n={cmet['n']} conflict_hit_anchor_side={cmet['hit']:.4f} conflict_signed_anchor_side={cmet['signed']:+.3f} conflict_clv_anchor_side={cmet['clv']:+.3f} "
            f"support_minus_conflict_hit={out['support_minus_conflict_hit']:+.4f} support_minus_conflict_signed={out['support_minus_conflict_signed']:+.3f} "
            f"support_minus_conflict_clv={out['support_minus_conflict_clv']:+.3f} production_authority=0"
        )
    return out


def _season_breakdown(g,target,anchor_name,partner_name,av_disc,am_disc,av_conf,am_conf,bv,bm,dashboard_module,log_func=print):
    season=pd.to_numeric(g.get('Season'),errors='coerce').to_numpy(float)
    rows=[]
    for sy in (2023,2024,2025,2026):
        m=season==float(sy)
        av,am=(av_conf,am_conf) if sy==2026 else (av_disc,am_disc)
        r=_presence_result(g,target,m,anchor_name,partner_name,av,am,bv,bm,dashboard_module,f"SEASON_{sy}",log_func=lambda *_:None)
        p=r['presence']; s=r['support']; c=r['conflict']; b=r['anchor_only']
        rows.append({'season':sy,**r})
        if p['n']>=3:
            log_func(
                f"[EDGE-COMPLEMENTARITY-V2-SEASON] season={sy} anchor={anchor_name} partner={partner_name} "
                f"anchor_only_n={b['n']} anchor_only_hit={b['hit']:.4f} present_n={p['n']} present_hit={p['hit']:.4f} present_signed={p['signed']:+.3f} present_clv={p['clv']:+.3f} "
                f"support_n={s['n']} support_hit={s['hit']:.4f} conflict_n={c['n']} conflict_hit_anchor_side={c['hit']:.4f} production_authority=0"
            )
    return rows


def _flatten_unique_signature_systems(sigs: dict):
    seen=set(); rows=[]
    for sig,rec in sorted(sigs.items()):
        for s in rec.get('systems') or []:
            key=s.get('hash') or s.get('id')
            if key in seen: continue
            seen.add(key)
            rows.append((sig,s))
    return rows


def _rule_vote(n:int, sysrec:dict):
    v=np.zeros(n,float); m=np.asarray(sysrec['mask'],bool); sign=float(sysrec.get('sign',1.0)); v[m]=sign
    return v,np.zeros(n,bool)


def _evidence_state(disc:dict,conf:dict):
    dp=disc['presence']; cp=conf['presence']; ds=disc['support']; cs=conf['support']; dc=disc['conflict']; cc=conf['conflict']
    presence_hit_repeat=(dp['n']>=40 and cp['n']>=15 and disc['delta_presence_hit']>0 and conf['delta_presence_hit']>0 and dp['signed']>0 and cp['signed']>0)
    presence_signed_incremental=(dp['n']>=40 and cp['n']>=15 and disc['delta_presence_signed']>0 and conf['delta_presence_signed']>0)
    outcome_repeat=(ds['n']>=20 and cs['n']>=10 and _delta(ds['hit'],disc['anchor_only']['hit'])>0 and _delta(cs['hit'],conf['anchor_only']['hit'])>0 and ds['signed']>0 and cs['signed']>0)
    price_positive=(ds['clv_n']>=10 and cs['clv_n']>=5 and ds['clv']>0 and cs['clv']>0)
    price_incremental=(np.isfinite(_delta(ds['clv'],disc['anchor_only']['clv'])) and np.isfinite(_delta(cs['clv'],conf['anchor_only']['clv'])) and _delta(ds['clv'],disc['anchor_only']['clv'])>0 and _delta(cs['clv'],conf['anchor_only']['clv'])>0)
    directional_sep=(ds['n']>=20 and cs['n']>=10 and dc['n']>=10 and cc['n']>=5 and disc['support_minus_conflict_hit']>0 and conf['support_minus_conflict_hit']>0 and disc['support_minus_conflict_signed']>0 and conf['support_minus_conflict_signed']>0)
    return {
        'presence_hit_repeat':bool(presence_hit_repeat),'presence_signed_incremental':bool(presence_signed_incremental),'outcome_repeat':bool(outcome_repeat),'price_positive':bool(price_positive),
        'price_incremental':bool(price_incremental),'directional_separation':bool(directional_sep),
        'state': ('FULL_INCREMENTAL_COMPLEMENT_CANDIDATE' if outcome_repeat and price_incremental else
                  'OUTCOME_COMPLEMENT_PRICE_POSITIVE' if outcome_repeat and price_positive else
                  'OUTCOME_COMPLEMENT_CLV_UNCONFIRMED' if outcome_repeat else
                  'REGIME_SELECTOR_FULL_INCREMENTAL' if presence_hit_repeat and presence_signed_incremental else
                  'REGIME_SELECTOR_HIT_RATE_ONLY' if presence_hit_repeat else 'UNPROVEN')
    }


def run_edge_complementarity_v2(*, dashboard_module, stat_out:dict, registry_out:dict, topology_out:dict|None=None, log_func=print, hard_fail=True):
    try:
        # Retain V1 diagnostics unchanged; V2 is additive and does not retune V1.
        v1_out=v1.run_edge_complementarity_v1(dashboard_module=dashboard_module,stat_out=stat_out,registry_out=registry_out,topology_out=topology_out,log_func=log_func,hard_fail=True)
        cache=getattr(dashboard_module,'_V1357_SPREAD_RESEARCH_CACHE',{})
        g=cache.get('games') if isinstance(cache,dict) else None
        if not isinstance(g,pd.DataFrame) or g.empty: raise RuntimeError('spread research games unavailable')
        g=g.copy(); target=pd.to_numeric(g.get('Market_Error_Margin'),errors='coerce').to_numpy(float)
        season=pd.to_numeric(g.get('Season'),errors='coerce').to_numpy(float)
        if int(et._physical_key(g).duplicated().sum()): raise RuntimeError('physical game duplicates present')
        strength=v1._strength_lookup(registry_out)
        sigs,sig_diag=v1._miner_signature_votes(g,dashboard_module,strength,min_strength=1)
        votes_disc=v1._source_votes(g,dashboard_module,stat_out,registry_out,'DISCOVERY')
        votes_conf=v1._source_votes(g,dashboard_module,stat_out,registry_out,'CONFIRM')
        disc=np.isin(season,np.asarray(DISCOVERY_SEASONS,float)); conf=season==float(CONFIRM_SEASON)
        log_func(
            f"[EDGE-COMPLEMENTARITY-V2-PREFLIGHT] status=PASS source_tag={EDGE_COMPLEMENTARITY_V2_SOURCE_TAG} games={len(g)} "
            f"miner_raw_systems={sig_diag.get('raw_systems',0)} miner_unique_signed_masks={sig_diag.get('unique_systems',0)} miner_duplicates_collapsed={sig_diag.get('duplicates_collapsed',0)} "
            f"discovery_role=DESCRIPTIVE_HYPOTHESIS_GENERATION confirm_2026_role=OUT_OF_DISCOVERY_CONFIRMATION_NOT_POSTFREEZE_PROSPECTIVE production_authority=0"
        )

        # Primary hypothesis: STAT with any Miner activity, then by predeclared signature.
        avd,amd=votes_disc['STAT']; bvd,bmd=votes_disc['MINER']
        avc,amc=votes_conf['STAT']; bvc,bmc=votes_conf['MINER']
        primary_d=_presence_result(g,target,disc,'STAT','MINER',avd,amd,bvd,bmd,dashboard_module,'DISCOVERY_2023_2025_DESCRIPTIVE',log_func)
        primary_c=_presence_result(g,target,conf,'STAT','MINER',avc,amc,bvc,bmc,dashboard_module,'CONFIRM_2026',log_func)
        state=_evidence_state(primary_d,primary_c)
        log_func(
            f"[EDGE-COMPLEMENTARITY-V2-EVIDENCE] anchor=STAT partner=MINER state={state['state']} "
            f"regime_hit_rate_repeat={str(state['presence_hit_repeat']).upper()} regime_signed_error_incremental={str(state['presence_signed_incremental']).upper()} repeatable_directional_outcome={str(state['outcome_repeat']).upper()} "
            f"price_clv_positive={str(state['price_positive']).upper()} price_clv_incremental={str(state['price_incremental']).upper()} "
            f"directional_separation={str(state['directional_separation']).upper()} production_authority=0"
        )
        seasons=_season_breakdown(g,target,'STAT','MINER',avd,amd,avc,amc,bvc,bmc,dashboard_module,log_func)

        sig_results={}
        # Signature votes do not depend on sample; STAT vote does, so pair each sample separately.
        for sig,rec in sorted(sigs.items()):
            rd=_presence_result(g,target,disc,'STAT',f'MINER[{sig}]',avd,amd,rec['vote'],rec['mixed'],dashboard_module,'DISCOVERY_2023_2025_DESCRIPTIVE',log_func)
            rc=_presence_result(g,target,conf,'STAT',f'MINER[{sig}]',avc,amc,rec['vote'],rec['mixed'],dashboard_module,'CONFIRM_2026',log_func)
            sig_results[sig]={'discovery':rd,'confirmation':rc}
            if rd['presence']['n']>=10 or rc['presence']['n']>=5:
                _season_breakdown(g,target,'STAT',f'MINER[{sig}]',avd,amd,avc,amc,rec['vote'],rec['mixed'],dashboard_module,log_func)

        # Rule-level attribution on the exact de-duplicated Miner rules.
        rule_rows=[]
        for sig,s in _flatten_unique_signature_systems(sigs):
            rv,rm=_rule_vote(len(g),s)
            rd=_presence_result(g,target,disc,'STAT',f"MINER_RULE[{s['id']}]",avd,amd,rv,rm,dashboard_module,'DISCOVERY',log_func=lambda *_:None)
            rc=_presence_result(g,target,conf,'STAT',f"MINER_RULE[{s['id']}]",avc,amc,rv,rm,dashboard_module,'CONFIRM_2026',log_func=lambda *_:None)
            rule_rows.append({'id':s['id'],'signature':sig,'conditions':s.get('conditions') or [],'discovery':rd,'confirmation':rc})
            dn=rd['presence']['n']; cn=rc['presence']['n']
            if dn>=8 or cn>=3:
                log_func(
                    f"[EDGE-COMPLEMENTARITY-V2-RULE] id={s['id']} signature={sig} rule={' AND '.join(s.get('conditions') or [])} "
                    f"discovery_present_n={dn} discovery_present_hit={rd['presence']['hit']:.4f} discovery_present_signed={rd['presence']['signed']:+.3f} "
                    f"discovery_support_n={rd['support']['n']} discovery_support_hit={rd['support']['hit']:.4f} discovery_conflict_n={rd['conflict']['n']} discovery_conflict_hit={rd['conflict']['hit']:.4f} "
                    f"confirm_present_n={cn} confirm_present_hit={rc['presence']['hit']:.4f} confirm_present_signed={rc['presence']['signed']:+.3f} "
                    f"confirm_support_n={rc['support']['n']} confirm_support_hit={rc['support']['hit']:.4f} confirm_conflict_n={rc['conflict']['n']} confirm_conflict_hit={rc['conflict']['hit']:.4f} production_authority=0"
                )

        log_func(
            f"[EDGE-COMPLEMENTARITY-V2-CONTRACT] status=PASS source_tag={EDGE_COMPLEMENTARITY_V2_SOURCE_TAG} "
            f"v1_contract={v1_out.get('status')} primary_presence_test=TRUE directional_support_vs_conflict=TRUE season_stability=TRUE "
            f"rule_level_attribution={len(rule_rows)} miner_exact_duplicates_collapsed=TRUE outcome_evidence_separate_from_clv=TRUE "
            f"confirmation_used_for_evaluation_only=TRUE zero_production_authority=TRUE"
        )
        return {'status':'PASS','source_tag':EDGE_COMPLEMENTARITY_V2_SOURCE_TAG,'v1':v1_out,'primary':{'discovery':primary_d,'confirmation':primary_c,'evidence':state,'seasons':seasons},'signatures':sig_results,'rules':rule_rows,'production_authority':0}
    except Exception as e:
        log_func(f"[EDGE-COMPLEMENTARITY-V2-CONTRACT] status=FAILED error={type(e).__name__}:{e} production_authority=0")
        if hard_fail: raise
        return {'status':'FAILED','error':f'{type(e).__name__}:{e}','production_authority':0}
