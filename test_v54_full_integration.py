"""Mock contract checks; these do not establish live BigQuery/GCS connectivity."""
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
import ncaaf_v54_full_research as mod
import ncaaf_v54_cfbd_truth as truth
from test_v54_cfbd_truth import tiny

class Blob:
 def __init__(self,obj,key):self.obj=obj;self.key=key
 def upload_from_string(self,s,**kw):
  if kw.get('if_generation_match')==0 and self.key in self.obj.data:raise RuntimeError('immutable artifact collision')
  self.obj.data[self.key]=s
 def download_as_text(self):return self.obj.data[self.key]
class Store:
 def __init__(self):self.data={}
 def bucket(self,name):return SimpleNamespace(name=name,blob=lambda key:Blob(self,key))

def _mock_sources(monkeypatch, *, force_fail=False):
    arc,features,mk=tiny()
    scores,_=truth.parse_cfbd_archive_rows(arc)
    sides,_=truth.build_cfbd_canonical_features(features,scores,enforce_full_history=False)
    sides,_=truth.attach_market_only(sides,mk)
    sides=truth.derived_pregame_states(sides)
    home=truth.home_games(sides)
    if force_fail:
        monkeypatch.setattr(truth,'load_from_bigquery',lambda client: (_ for _ in ()).throw(RuntimeError('CFBD_API_ARCHIVE_MISSING_REQUIRED_YEAR_2013')))
    else:
        monkeypatch.setattr(truth,'load_from_bigquery',lambda client:(home,sides,{'status':'CFBD_AUTHORITATIVE','scores_cfbd_only':True}))
    monkeypatch.setattr(mod,'module_contract',lambda m:{k:{'status':'PASS','sha256':'test'} for k in m})
    monkeypatch.setattr(mod,'legacy_source_constants',lambda *a:{'status':'PASS'})
    monkeypatch.setattr(mod,'source_manifest',lambda **kw:{'objects':{},'tables':{}})
    monkeypatch.setenv('NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED','1')
    return home

def test_full_cfbd_firewall_and_weekly_cache_untouched(monkeypatch):
    home=_mock_sources(monkeypatch)
    st=Store(); prefix='research/ncaaf/v51_edge_attribution/runs/test/unique_model_game_edges.csv'
    edge=pd.DataFrame([{'Season':2024,'Game_ID':'100','Game_Date':'2024-09-07','Team':'Alpha',
         'Opponent':'Beta','Model':'NONLINEAR_STATE_HGB','Side':'TEAM','Result':'LOSS',
         'Abs_Edge':3.5,'Selection_Type':'FAVORITE','Period':'examined_2024_2025','Edge_Bin':'3–5'}])
    st.data[prefix]=edge.to_csv(index=False)
    oldcache={'games':pd.DataFrame([{'Team_Score':9999}])}
    dash=SimpleNamespace(V133_DEPLOY_BUILD_ID='test',_V1357_SPREAD_RESEARCH_CACHE=oldcache)
    def v4(**kw):
        x=dash._V1357_SPREAD_RESEARCH_CACHE['games']
        assert x.Historical_Football_Source.eq('CFBD_API_ARCHIVE_GAMES_PLUS_CFBD_RESEARCH_FEATURES').all()
        assert x.Team_Score.tolist()==[21]
        return {'status':'RESEARCH_COMPLETE_NO_PROMOTION','report_uri':'gs://mock/v4'}
    def v42(**kw):
        assert dash._V1357_SPREAD_RESEARCH_CACHE['games'].Team.tolist()==['Alpha']
        return {'status':'EXPLORATORY_RESEARCH__SOURCE_VINTAGE_HOLD','predictions_uri':'gs://mock/v42'}
    def miner(x,seasons,market,**kw):
        assert kw['dashboard_module'] is None
        assert x.Team_Score.tolist()==[21]
        assert 'Postgame_Total_Yards' not in x
        return {'status':'PASS','atoms':0,'mechanism_families':[],'systems':[]}
    rv=SimpleNamespace(_pt_load_canonical_team_aliases=lambda *a,**kw:{},
       refresh_prediction_tracker_external=lambda **kw:{'status':'PASS'},
       run_system_miner_v3=miner,_extended_atoms=lambda *a,**kw:[])
    ut=SimpleNamespace(V133_DEPLOY_BUILD_ID='test',add_pathi_bigal_rule_flags=lambda z:z,
       BIGAL_EXACT_SIGNAL_COLS=(),PATHI_KEY_EVENT_COLS=())
    core=SimpleNamespace(NCAAF_CFBD_MINER_FEATURE_TABLE='original',_CFBD_CORE_CONTEXT_CACHE={})
    mods={k:SimpleNamespace() for k in mod.REQUIRED_LEGACY}
    mods.update(sharp_line_dashboard=dash,ncaaf_core_challenger_v2=core,
        utils=ut,ncaaf_research_v2=rv,
        ncaaf_margin_tournament_v4=SimpleNamespace(run_cloud_research=v4),
        ncaaf_v42_conditional_features=SimpleNamespace(run_cloud_research=v42),
        ncaaf_v5_betting_research=SimpleNamespace(cloud_run=lambda **kw:{'status':'PASS','historical_games':1}),
        ncaaf_v51_edge_research=SimpleNamespace(cloud_run=lambda **kw:{
            'v4_prediction_verified_with_source':True,'dedup_ledger_uri':'gs://sharp-models/'+prefix,
            'physical_games':1,'dedup_model_game_rows':1}),
        ncaaf_v52_brain_research=SimpleNamespace(cloud_run=lambda **kw:{'status':'NO_VERIFIED_ARCHIVE','verified_system_game_records':0}))
    out=mod.run_full(storage_client=st,modules=mods,bq_client=object(),log_func=lambda x:None)
    assert out['status']=='COMPLETE_FULL_RESEARCH__SHADOW_AUTHORITY__VINTAGE_HOLD'
    assert out['bet_authority']==0
    assert out['cfbd_source_truth']['scores_cfbd_only']
    assert out['research_components']['cfbd_only_miner']['spreads']['football_source']=='CFBD'
    assert dash._V1357_SPREAD_RESEARCH_CACHE is oldcache
    assert core.NCAAF_CFBD_MINER_FEATURE_TABLE=='original'
    stages=[x['stage'] for x in out['stages']]
    for item in ('CFBD_CANONICAL_HISTORY_AND_SCORE_PROVENANCE','FULL_V4_FOOTBALL_ONLY_CFBD',
          'FULL_SYSTEM_MINER_ALL_MARKETS_CFBD_ONLY','FULL_CFBD_SYSTEM_INCREMENTAL_ATTRIBUTION'):
        assert item in stages
    for item in ('BUILD_STAT_CACHE','BUILD_HISTORICAL_EXPERT_CACHE','FULL_LEGACY_PATHI_BIGAL_PT_MINER_STATE_RESEARCH'):
        assert item not in stages
    assert 'research/ncaaf/v54_full_research/latest_report.json' in st.data

def test_missing_cfbd_archive_publishes_audit_before_training(monkeypatch):
    _mock_sources(monkeypatch,force_fail=True)
    st=Store();mods={k:SimpleNamespace() for k in mod.REQUIRED_LEGACY}
    mods['sharp_line_dashboard']=SimpleNamespace(V133_DEPLOY_BUILD_ID='test')
    mods['utils']=SimpleNamespace(V133_DEPLOY_BUILD_ID='test')
    with pytest.raises(RuntimeError,match='V54_FULL_RESEARCH_FAILED'):
        mod.run_full(storage_client=st,modules=mods,bq_client=object(),log_func=lambda x:None)
    r=json.loads(st.data['research/ncaaf/v54_full_research/latest_report.json'])
    assert r['status']=='FAILED_FULL_RESEARCH__AUDIT_PUBLISHED'
    assert 'CFBD_API_ARCHIVE_MISSING_REQUIRED_YEAR_2013' in r['research_components']['failure']['error']
    assert 'FULL_V4_FOOTBALL_ONLY_CFBD' not in [x['stage'] for x in r['stages']]
