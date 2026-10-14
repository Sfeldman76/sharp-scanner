"""Offline regression tests. No network, GCS, BQ, or production mutation."""
import ast
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import types

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import pt_five_selection_v1 as selector
import nfl_engine

def test_selector():
    assert selector._selftest()
    r=np.random.default_rng(77);n=1600
    years=np.repeat([2022,2023,2024,2025],400)
    df=pd.DataFrame({'Season':pd.Series(years,dtype='Int64'), 'actual':r.normal(0,10,n)})
    cols={};groups={}
    for i in range(14):
        key=f'SOURCE_{i}'
        df[key]=df['actual']+r.normal(0,2+i*.25,n)
        # Missingness is independent of holdout outcomes.
        df.loc[r.random(n)<.12,key]=np.nan
        cols[key]=key;groups[key]=f'PROVIDER_{i}'
    df.index=np.arange(9000,9000+n)
    sel=selector.select_five(df,source_cols=cols,group_by_source=groups,actual_col='actual')
    assert sel['status']=='FIVE_SELECTED_DEVELOPMENT_ONLY',sel
    assert len(sel['sources'])==5
    edits=df.copy()
    edits.loc[edits['Season']>=2024,'actual']+=999.0
    sel2=selector.select_five(edits,source_cols=cols,group_by_source=groups,actual_col='actual')
    assert sel2['selection_sha256']==sel['selection_sha256']
    return {'source_count':5,'history_sha_stable':True}

def test_nfl_live_vote_fallback():
    source=nfl_engine._decode_source('nfl_model_authority_v26')
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_pt_promotion_prebet')
    ns={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<selected-test>','exec'),ns)
    fake=types.SimpleNamespace(promoted_vote=lambda state,margin,line,count,sha:(1 if margin is not None and count is not None and count>=3 and sha=='LOCK' else None))
    prior=sys.modules.get('pt_promotion_v1')
    sys.modules['pt_promotion_v1']=fake
    try:
        reg={ 'PT_LEGACY':{'system_id':'PT_LEGACY','source':'PT','qualified_current':True},'MINER_1':{'system_id':'MINER_1','source':'MINER','qualified_current':True}}
        legacy={'system_id':'PT_LEGACY','family':'OLD_PT','direction':-1}
        miner={'system_id':'MINER_1','family':'MINER_FAM','direction':1}
        base={'market':'SPREADS','system_trigger_votes':[legacy,miner],'edge_votes':[legacy,miner],
              'pt_auto_challenger_margin_home':None,'current_home_spread':-2.,'pt_auto_source_count':2,'pt_auto_selection_sha256':'LOCK'}
        state={'status':'PROMOTED','champion':'FIVE_SOURCE'}
        held,_=ns['apply_pt_promotion_prebet']([base],reg,state)
        assert held[0]['system_trigger_votes']==[legacy,miner]
        assert held[0]['edge_votes']==[legacy,miner]
        assert held[0]['pt_auto_vote_status'].startswith('INCUMBENT_FALLBACK')
        good={**base,'pt_auto_challenger_margin_home':4.0,'pt_auto_source_count':5}
        res,_=ns['apply_pt_promotion_prebet']([good],reg,state)
        votes=res[0]['system_trigger_votes']
        assert len(votes)==2 and votes[0]==miner and votes[1]['system_id']=='NFL_PT_AUTO_FIVE_SOURCE',votes
        assert res[0]['edge_votes']==[miner]
        assert votes[1]['qualification_source']=='LOCKED_CHRONOLOGICAL_HISTORICAL_PROMOTION'
        assert res[0]['pt_auto_vote_status']=='PROMOTED_FIVE_SOURCE_APPLIED'
    finally:
        if prior is None: sys.modules.pop('pt_promotion_v1',None)
        else:sys.modules['pt_promotion_v1']=prior
    return {'fallback_keeps_incumbent':True,'promoted_has_one_pt_vote':True,'non_pt_votes_retained':True}


def test_health_and_publication_audit():
    import ncaaf_research_v2 as nc
    assert hasattr(nc,'NCAAF_PT_DIAGNOSTICS_PATCH')
    lab=nfl_engine._decode_source('nfl_system_lab_v3')
    assert '[NFL-RESEARCH-HEALTH]' in lab
    assert '[NFL-PT-PUBLICATION-AUDIT]' in lab
    assert 'UNVERIFIED_HISTORICAL_SOURCE_TIMING' in lab
    assert '[NCAAF-PT-INCREMENTAL-WATERFALL]' in (HERE/'ncaaf_research_v2.py').read_text()
    assert '[NCAAF-RESEARCH-HEALTH]' in (HERE/'ncaaf_research_v2.py').read_text()
    pf=nfl_engine.preflight()
    assert pf['status']=='PASS' and pf['compiled_and_checked']==33,pf
    return {'nfl_33_components_pass':True,'component_health_present':True,'timing_not_falsely_verified':True}

if __name__=='__main__':
    for name,test in [('selector',test_selector),('nfl_vote_fallback',test_nfl_live_vote_fallback),('research_health',test_health_and_publication_audit)]:
        result=test()
        print(f'{name} PASS {json.dumps(result,sort_keys=True)}')
