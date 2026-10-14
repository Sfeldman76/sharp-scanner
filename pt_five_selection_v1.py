"""Chronological, provider-diverse selection of five PT ratings (research only).

Selections use ONLY pre-2024 completed games. Later years are reserved for
paired, genuinely untouched testing. 2026 is always excluded from selection.
This deliberately does not grant betting/model authority.
"""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
import pandas as pd

SOURCE_TAG = 'pt-five-selection-v1.0-frozen-development-only-20261013'


def select_five(frame, *, source_cols, group_by_source, actual_col, season_col='Season',
                development_years=(2022, 2023), max_candidates=26, beam_size=16,
                min_full_coverage=.55):
    """Select a source-diverse five-model equal-weight ensemble.

    source_cols maps stable source IDs -> frame column names. Names are never
    inferred from positional order. The provider/family mapping must originate
    from the caller's existing canonicalization. Future-year outcomes NEVER
    influence selection, source eligibility, weights or tie-breaks.
    """
    d=frame.copy()
    years=pd.to_numeric(d[season_col],errors='coerce')
    y=pd.to_numeric(d[actual_col],errors='coerce').to_numpy(float)
    discovery=years.isin(tuple(development_years)).to_numpy()&np.isfinite(y)
    if not discovery.any() or int(discovery.sum())<160:
        return {'status':'INSUFFICIENT_DEVELOPMENT_GAMES','sources':[], 'n':int(discovery.sum()), 'production_authority':0}
    yr=years.to_numpy(float)[discovery]
    target=y[discovery]
    unique_years=tuple(sorted(set(int(t) for t in yr)))
    if len(unique_years)<2 or max(unique_years)>=2024:
        return {'status':'INVALID_OR_LEAKING_DISCOVERY_SPLIT','sources':[], 'production_authority':0}
    m={}; audit=[]; gr={}
    for name,col in sorted(source_cols.items()):
        if col not in d.columns:continue
        group=str(group_by_source.get(name) or '').strip()
        if not group:continue
        raw=pd.to_numeric(d[col],errors='coerce').to_numpy(float)[discovery]
        valid=np.isfinite(raw)
        if valid.sum()<80 or np.nanstd(raw)<.5:continue
        coverage=float(valid.mean())
        annual={str(yy):float(valid[yr==yy].mean()) for yy in unique_years}
        # Do not drop useful sparse data indiscriminately, but require enough
        # observations in EACH development year to prevent one-season winners.
        if coverage<.60 or min(annual.values())<.50:continue
        mae=float(np.mean(np.abs(raw[valid]-target[valid])))
        m[name]=raw;gr[name]=group
        audit.append({'source':name,'provider':group,'development_mae':round(mae,5),
                      'coverage':round(coverage,5),'year_coverage':annual,'observations':int(valid.sum())})
    if len({gr[n] for n in m})<5:
        return {'status':'INSUFFICIENT_DIVERSE_RATINGS','sources':[], 'n':int(discovery.sum()),
                'eligible_sources':len(m),'eligible_providers':len(set(gr.values())),'production_authority':0}
    audit.sort(key=lambda t:(t['development_mae']+3.0*(1-t['coverage']),t['source']))
    pool=[a['source'] for a in audit[:max_candidates]]
    # Keep a distinct candidate from every provider if the top pool is clustered.
    for item in audit:
        if len({gr[p] for p in pool})>=5:break
        if gr[item['source']] not in {gr[p] for p in pool}:pool.append(item['source'])
    pool=sorted(pool)

    def score(names):
        vals=np.column_stack([m[k] for k in names]);count=np.isfinite(vals).sum(axis=1)
        n_req=max(1,int(math.ceil(.60*len(names))))
        usable=count>=n_req
        coverage=float(np.mean(usable))
        if coverage<min_full_coverage:return float('inf'),{'coverage':coverage}
        v=vals[usable]
        pred=np.nansum(v,axis=1)/count[usable]
        err=np.abs(pred-target[usable]);mae=float(np.mean(err))
        by_year=[]
        for year in unique_years:
            xx=usable&(yr==year)
            if xx.sum()<35:return float('inf'),{'coverage':coverage}
            v2=vals[xx];cnt2=count[xx]
            annual_err=float(np.mean(np.abs(np.nansum(v2,axis=1)/cnt2-target[xx])))
            by_year.append(annual_err)
        # penalize missing games, annual instability and duplicate predictions;
        # only proven distinct providers are eligible for the final five.
        corr_penalty=0.0
        for i in range(len(names)):
            for j in range(i+1,len(names)):
                a,b=m[names[i]],m[names[j]]
                pair=np.isfinite(a)&np.isfinite(b)
                if pair.sum()>60:
                    c=float(np.corrcoef(a[pair],b[pair])[0,1])
                    if np.isfinite(c):corr_penalty+=max(0.0,c-.94)*.30
        value=mae+1.5*(1-coverage)+.10*np.std(by_year)+corr_penalty
        return float(value),{'development_mae':mae,'coverage':coverage,'year_maes':dict(zip(map(str,unique_years),by_year)),
                              'rows':int(usable.sum()),'rating_count':len(names),'min_ratings':n_req}

    beams=[((),0.0,{})]
    for depth in range(1,6):
        choices=[]
        for sel,_,_ in beams:
            for name in pool:
                if name in sel or gr[name] in {gr[z] for z in sel}:continue
                candidate=tuple(sorted((*sel,name)))
                val,diag=score(candidate)
                if np.isfinite(val):choices.append((candidate,val,diag))
        # Scores depend on ensemble depth, not on predictive performance in 2024+.
        best={}
        for c,v,info in choices:
            if c not in best or v<best[c][1]:best[c]=(c,v,info)
        beams=sorted(best.values(),key=lambda a:(a[1],a[0]))[:beam_size]
        if not beams:
            return {'status':f'NO_DIVERSE_ENSEMBLE_AT_DEPTH_{depth}','sources':[],
                    'production_authority':0}
    selected,objective,details=beams[0]
    ordered=list(selected)
    selection_contract={'sources':ordered,'providers':[gr[x] for x in ordered],
                        'development_years':unique_years,'objective':'MAE_PLUS_COVERAGE_STABILITY_PROVIDER_PENALTY',
                        'selector_source_tag':SOURCE_TAG}
    sha=hashlib.sha256(json.dumps(selection_contract,sort_keys=True,default=str).encode()).hexdigest()
    return {'status':'FIVE_SELECTED_DEVELOPMENT_ONLY','sources':ordered,
            'source_providers':{x:gr[x] for x in ordered},'selection_sha256':sha,
            'development_years':list(unique_years),'development_metrics':details,
            'eligible_source_count':len(m),'eligible_provider_count':len(set(gr.values())),
            'search_candidates':len(pool),'ranking_audit':audit[:30],
            'holdout_seasons':[2024,2025], 'season_max_for_selection':int(max(unique_years)),
            'production_authority':0,'automatic_promotion':False,
            'early_week_publish_coverage_verified':False}


def _selftest():
    rng=np.random.default_rng(2026)
    d=pd.DataFrame({'Season':np.repeat([2022,2023,2024,2025],200)})
    d['actual']=rng.normal(0,13,len(d));cols={};groups={}
    for k in range(9):
        name=f's{k}'; cols[name]=name;groups[name]=f'p{k}'
        d[name]=d['actual']+rng.normal(0,3+k*.7,len(d))
    a=select_five(d,source_cols=cols,group_by_source=groups,actual_col='actual')
    assert a['status']=='FIVE_SELECTED_DEVELOPMENT_ONLY' and len(a['sources'])==5
    # Held-out season edits must NEVER change the selected sources.
    e=d.copy();e.loc[e.Season>=2024,'actual']=100000
    b=select_five(e,source_cols=cols,group_by_source=groups,actual_col='actual')
    assert a['selection_sha256']==b['selection_sha256']
    assert all(y<2024 for y in a['development_years'])
    return True

if __name__=='__main__':
    print('PASS' if _selftest() else 'FAIL')
