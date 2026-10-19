"""Research-only parent-child statistical family ablations (2026-10-18).

Candidates selected exclusively from pre-2024 family discovery. Every comparison
uses identical physical games, strictly prior-season training, and the same OOF
score foundation. This is NOT a production model or betting system.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from ncaaf_historical_signal_discovery import (
    HEADS, _num, _dates, _score, _fit_residual, _sign_flip_pvalue,
    reconstruct_historical_score_form,
)

VERSION = "NCAAF_COMBINED_HISTORICAL_FAMILY_ABLATION_20261018_V1"
DISCOVERY_YEARS = tuple(range(2018,2024))
RETROSPECTIVE_YEARS = (2024,2025)
MIN_SELECTION_N = 500


def _fold_predictions(frame, actual, baseline, fields, head, years):
    season=_num(frame,'Season').to_numpy(float)
    X=frame[fields].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    pred=np.full(len(frame),np.nan)
    info={}
    for yr in years:
        tr=(season<yr)&(season>=2014)&np.isfinite(baseline)
        va=season==yr
        delta=_fit_residual(X,actual,baseline,tr,va,head)
        if delta is None:
            info[str(yr)]={'status':'NO_PRIOR_FOLD_OR_FEATURE_COVERAGE'}
            continue
        pred[va]=baseline[va]+.35*delta
        if head=='h2h':pred[va]=np.clip(pred[va],1e-5,1-1e-5)
        info[str(yr)]={'status':'SCORED','n':int(va.sum())}
    return pred,info


def run_combined_historical_ablation(frame, predictions, discovery, *, log_func=print):
    if not isinstance(frame,pd.DataFrame) or 'Season' not in frame:
        raise ValueError('Missing canonical game frame')
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError('Game index not canonical')
    season=_num(frame,'Season').to_numpy(float)
    if np.isfinite(season).any() and np.nanmax(season)>2025:
        raise ValueError('2026+ labels prohibited')
    hist,audit=reconstruct_historical_score_form(frame)
    full=pd.concat([frame,hist],axis=1)
    for head in HEADS:
        if head not in predictions or len(predictions[head])!=len(frame):
            raise ValueError('OOF baseline misaligned')
    results={};dates=_dates(frame)
    years=(*DISCOVERY_YEARS,*RETROSPECTIVE_YEARS)
    for head in HEADS:
        y=_num(frame,'Actual_Total' if head=='total' else 'Actual_Margin').to_numpy(float)
        baseline=np.asarray(predictions[head],float)
        selected=(discovery.get('selection') or {}).get(head,{}).get('selection_passed') or []
        available={fam:rec for fam,rec in (discovery.get('families') or {}).items()
                   if rec.get('lane')=='historical' and fam in selected}
        # Fixed cap to avoid a broad combinatorial feature search.
        names=[fam for fam in selected if fam in available][:4]
        parents={}
        for fam in names:
            cols=[c for c in available[fam].get('features',[]) if c in full]
            if not cols:continue
            parents[fam]=_fold_predictions(full,y,baseline,cols,head,years)[0]
        candidates=[]
        for a,b in combinations(names,2):
            if a not in parents or b not in parents:continue
            cols=list(dict.fromkeys([*available[a]['features'],*available[b]['features']]))
            child,folds=_fold_predictions(full,y,baseline,cols,head,years)
            sel=np.isin(season,DISCOVERY_YEARS)
            conf=np.isin(season,RETROSPECTIVE_YEARS)
            # Pair vs each of its actual parents on EXACTLY matched games.
            ablations={}
            for name,solo in ((a,parents[a]),(b,parents[b])):
                result=_score(y,solo,child,head,sel)
                hold=_score(y,solo,child,head,conf)
                ablations[name]={'selection':result,'retrospective_2024_2025':hold}
            baseline_sel=_score(y,baseline,child,head,sel)
            baseline_conf=_score(y,baseline,child,head,conf)
            # Stronger single-feature-family parent is chosen by pre-2024 score.
            parent_candidates=[]
            for fam,solo in ((a,parents[a]),(b,parents[b])):
                # Choose the stronger single family by its historical improvement
                # over the SAME foundation, not by how weak it is versus child.
                m=_score(y,baseline,solo,head,sel)
                if m.get('gain') is not None:
                    parent_candidates.append((m['gain'],fam))
            if not parent_candidates:continue
            best_parent=max(parent_candidates,key=lambda x:x[0])[1]
            best_parent_oof=parents[best_parent]
            parent_gain=ablations[best_parent]['selection']['gain']
            p=_sign_flip_pvalue(y,best_parent_oof,child,head,sel,dates,repeats=399,seed=1926)
            annual=[]
            for yr in DISCOVERY_YEARS:
                annual.append({'year':yr,**_score(y,best_parent_oof,child,head,season==yr)})
            candidate={'families':[a,b],'best_parent':best_parent,
               'selected_pre2024':True,'holdout_used_for_selection':False,
               'selection_child_vs_foundation':baseline_sel,
               'selection_child_vs_best_parent':ablations[best_parent]['selection'],
               'p_one_sided':p,'positive_selection_years':sum((x.get('gain') or 0)>0 for x in annual),
               'retrospective_child_vs_foundation':baseline_conf,
               'retrospective_child_vs_best_parent':ablations[best_parent]['retrospective_2024_2025'],
               'ablation_vs_each_parent':ablations,'per_year':annual,'folds':folds,
               'features':cols,'authority':0}
            candidates.append(candidate)
        # Conservative Holm adjustment for paired-combination experiments.
        order=sorted([x for x in candidates if x['p_one_sided'] is not None],key=lambda r:r['p_one_sided'])
        prev=0.0
        for j,item in enumerate(order):
            prev=max(prev,min(1.,(len(order)-j)*item['p_one_sided']))
            item['holm_adjusted_p']=round(prev,6)
        # No tuning to 2024–2025 results. Research-shadow nomination only.
        for item in candidates:
            score=item['selection_child_vs_best_parent']
            item['decision']=('PAIR_RESEARCH_SHADOW_CANDIDATE' if
                score['n']>=MIN_SELECTION_N and (score.get('gain') or -999) > {'margin':.05,'total':.05,'h2h':.0015}[head]
                and item['positive_selection_years']>=4 and
                item.get('holm_adjusted_p',1.0)<=.10 else 'NO_INDEPENDENT_PAIR_GAIN')
        results[head]={'tested_pairs':len(candidates), 'candidates':candidates,
                       'shadow_candidates':[c['families'] for c in candidates if c['decision']=='PAIR_RESEARCH_SHADOW_CANDIDATE']}
        log_func('[NCAAF-COMBINED-SIGNAL] '+str({'head':head,'tested_pairs':len(candidates),
                  'research_shadow_pairs':results[head]['shadow_candidates'],'production_authority':0}))
    return {'status':'RESEARCH_ONLY_NO_AUTOMATIC_PROMOTION','version':VERSION,
            'source':'EXISTING_PRE2024_DISCOVERY_ONLY','selection_years':list(DISCOVERY_YEARS),
            'retrospective_years':list(RETROSPECTIVE_YEARS),
            'historical_features_only':True,'2026_outcomes_used':False,
            'families_by_head':results,'authority':0,'production_mutation':False,
            'limitations':'2024/25 extensively viewed; prospective shadow needed'}
