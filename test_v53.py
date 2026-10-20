import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from ncaaf_v53_full_research import (export_legacy_occurrences,export_legacy_miner_mechanisms,
    exploratory_attribution,policy_shadow,module_contract,REQUIRED_LEGACY)


def sample():
    return pd.DataFrame([
      {'Season':2024,'Game_ID':'a','Game_Date':'2024-09-07','Team':'Ohio State Buckeyes','Opponent':'Michigan Wolverines',
       'Model':'NONLINEAR_STATE_HGB','Side':'TEAM','Result':'WIN','Abs_Edge':3.7,'Selection_Type':'FAVORITE','Period':'examined_2024_2025','Edge_Bin':'3–5'},
      {'Season':2024,'Game_ID':'a','Game_Date':'2024-09-07','Team':'Ohio State Buckeyes','Opponent':'Michigan Wolverines',
       'Model':'A_EFFICIENCY','Side':'OPPONENT','Result':'LOSS','Abs_Edge':5.3,'Selection_Type':'UNDERDOG','Period':'examined_2024_2025','Edge_Bin':'5–7'},
      {'Season':2025,'Game_ID':'b','Game_Date':'2025-10-04','Team':'Alabama Crimson Tide','Opponent':'Auburn Tigers',
       'Model':'NONLINEAR_STATE_HGB','Side':'OPPONENT','Result':'LOSS','Abs_Edge':6.3,'Selection_Type':'UNDERDOG','Period':'examined_2024_2025','Edge_Bin':'5–7'}])

def history():
 return {'Pathi_FB_Dog_On_7':{'family':'Pathi','role':'directional','occurrences':[
     {'season':2024,'date':'2024-09-07','team':'Michigan Wolverines','opponent':'Ohio State Buckeyes'}]},
  'BigAl_CF2_Example':{'family':'BigAl','role':'directional','occurrences':[
     {'season':2024,'date':'2024-09-07','team':'Ohio State Buckeyes','opponent':'Michigan Wolverines'},
     {'season':2025,'date':'2025-10-04','team':'Auburn Tigers','opponent':'Alabama Crimson Tide'}]},
  'NOT_DIRECTIONAL':{'family':'BigAl','role':'context','occurrences':[]}}


def test_occurrences_exact_orientation_and_full_two_models():
 out,a=export_legacy_occurrences(sample(),history())
 assert a['directional_systems']==2 and a['joined']==3 and a['unmatched']==0
 m=out[(out.Game_ID=='a')&(out.Family=='PATHI')]
 assert m.Side.tolist()==['OPPONENT']
 assert not out.Evidence_Class.str.contains('ARCHIVED_PREGAME_VERIFIED').any()
 sys,fam=exploratory_attribution(sample(),out)
 assert len(sys)>0 and set(fam.Relation)=={'CONFIRM','OPPOSE'}
 assert (sys.authority==0).all()


def test_unmatched_team_and_date_fail_closed():
 h=history();h['Pathi_FB_Dog_On_7']['occurrences'][0]['date']='2024-09-08'
 x,a=export_legacy_occurrences(sample(),h)
 assert a['unmatched']==1 and len(x)==2


def test_conflicting_side_same_system_quarantined():
 h=history();h['Pathi_FB_Dog_On_7']['occurrences'].append(
   {'season':2024,'date':'2024-09-07','team':'Ohio State Buckeyes','opponent':'Michigan Wolverines'})
 out,a=export_legacy_occurrences(sample(),h)
 assert a['conflicting_system_game']==1 and not (out.Family=='PATHI').any()


def test_shadow_no_authority_even_for_many_retro_brains():
 out,a=export_legacy_occurrences(sample(),history())
 shadow,policy=policy_shadow(sample(),out,0,False)
 assert policy['bet_authority']==0 and (shadow.V5_Bet_Decision=='NO_BET').all()
 assert not any(col.lower().startswith('p_cover') for col in shadow.columns)
 assert len(shadow)==3 and not shadow.duplicated(['Season','Game_ID','Model']).any()


def test_v4_orientation_reverse_in_miner_mapping():
 mg=pd.DataFrame([{'Season':2024,'Game_Date':'2024-09-07','Team_Norm':'Michigan Wolverines',
      'Opponent_Norm':'Ohio State Buckeyes','Actual_Margin':-3,'Consensus_Open_Spread':7}])
 mech={'confirmation_pass':True,'representative_conditions':['ATOM'],
       'direction':'PLAY_ON','mechanism_id':'M1'}
 rv=SimpleNamespace(_extended_atoms=lambda *a,**kw:[{'name':'ATOM','mask':np.array([True])}])
 a,diag=export_legacy_miner_mechanisms(sample(),mg,{'spreads':{'mechanism_families':[mech]}},rv,None)
 assert len(a)==1 and a.Side.iloc[0]=='OPPONENT'
 assert a.Family.iloc[0]=='SYSTEM_MINER' and diag['matched_games']==1


def test_miner_state_and_pt_mapping_no_duplicate_votes():
 mg=pd.DataFrame([{'Season':2024,'Game_Date':'2024-09-07','Team_Norm':'Ohio State Buckeyes',
      'Opponent_Norm':'Michigan Wolverines','Actual_Margin':3,'Consensus_Open_Spread':-7}])
 m=[{'confirmation_pass':True,'representative_conditions':['ATOM'],'direction':'PLAY_ON',
     'mechanism_id':'m_state','uses_role_transition_bounceback_family':True},
    {'confirmation_pass':True,'representative_conditions':['ATOM'],'direction':'FADE',
     'mechanism_id':'m_pt','uses_external_ratings_family':True}]
 rv=SimpleNamespace(_extended_atoms=lambda *a,**kw:[{'name':'ATOM','mask':np.array([True])}])
 a,diag=export_legacy_miner_mechanisms(sample(),mg,{'spreads':{'mechanism_families':m}},rv,None)
 assert set(a.Family)=={'STATE_BOUNCE_BACK','PREDICTION_TRACKER'} and len(a)==2
 assert (a.Evidence_Class=='CHRONOLOGICAL_MINER_RETROSPECTIVE__INSPECTED_CONFIRMATION').all()


def test_miner_missing_atom_not_faked():
 mg=pd.DataFrame([{'Season':2024,'Game_Date':'2024-09-07','Team_Norm':'Ohio State Buckeyes',
                   'Opponent_Norm':'Michigan Wolverines'}])
 rv=SimpleNamespace(_extended_atoms=lambda *a,**kw:[])
 a,diag=export_legacy_miner_mechanisms(sample(),mg,
       {'spreads':{'mechanism_families':[{'confirmation_pass':True,'representative_conditions':['MISSING'],'direction':'PLAY_ON','mechanism_id':'X'}]}},rv,None)
 assert a.empty and diag['condition_unavailable']==1


def test_required_modules_named_functions_contract():
 mods={k:SimpleNamespace(**{fname:lambda:None for fname in names}) for k,names in REQUIRED_LEGACY.items()}
 c=module_contract(mods)
 assert all(x['status']=='PASS' for x in c.values())
 mods['utils']=SimpleNamespace()
 assert module_contract(mods)['utils']['status']=='FAIL'


def test_real_v51_edge_file_joins_dont_promote():
 path=Path('/mnt/data/v51_edge_engine_work/sample_analysis/unique_model_game_edges.csv')
 if not path.exists():pytest.skip('local fixture not available')
 edges=pd.read_csv(path)
 assert len(edges)==7355
 legacy,a=export_legacy_occurrences(edges,{})
 shadow,pol=policy_shadow(edges,legacy)
 assert len(shadow)==7355 and pol['bet_authority']==0

class _FakeBlob:
    def __init__(self,st,key):self.st=st;self.key=key
    def download_as_text(self):return self.st.data[self.key]
    def upload_from_string(self,v,**kwargs):
        if kwargs.get('if_generation_match')==0 and self.key in self.st.data:
            raise AssertionError('overwriting immutable artifact')
        self.st.data[self.key]=v
class _FakeBucket:
    name='sharp-models'
    def __init__(self,st):self.st=st
    def blob(self,key):return _FakeBlob(self.st,key)
class _FakeStorage:
    def __init__(self):self.data={}
    def bucket(self,name):assert name=='sharp-models';return _FakeBucket(self)


def test_full_job_mocked_one_shot_with_audited_outputs(monkeypatch):
    import ncaaf_v53_full_research as mod
    monkeypatch.setattr(mod,'module_contract',lambda m:{k:{'status':'PASS','sha256':'fake'} for k in m})
    monkeypatch.setattr(mod,'legacy_source_constants',lambda *a:{'status':'PASS'})
    monkeypatch.setattr(mod,'source_manifest',lambda **kw:{'objects':{},'tables':{}})
    st=_FakeStorage();ed=sample()
    uri='gs://sharp-models/research/ncaaf/v51_edge_attribution/runs/mock/unique_model_game_edges.csv'
    st.data[uri.removeprefix('gs://sharp-models/')]=ed.to_csv(index=False)
    h=history()
    cache={'games':pd.DataFrame([{'Season':2024,'Game_Date':'2024-09-07'}]),
           'miner_games':pd.DataFrame([{'Season':2024,'Game_Date':'2024-09-07',
                     'Team_Norm':'Ohio State Buckeyes','Opponent_Norm':'Michigan Wolverines'}])}
    dash=SimpleNamespace(_V1357_SPREAD_RESEARCH_CACHE=cache,
          V133_DEPLOY_BUILD_ID='same',_V143_SYSTEM_HISTORY_CACHE=h,
          fit_historical_ncaaf_core_expert=lambda *a,**kw:True,
          fit_ncaaf_statistical_brain=lambda **kw:True)
    def wrap(**methods):return SimpleNamespace(**methods)
    modules={
      'sharp_line_dashboard':dash,
      'utils':wrap(V133_DEPLOY_BUILD_ID='same'),
      'ncaaf_margin_tournament_v4':wrap(run_cloud_research=lambda **kw:{'status':'RESEARCH_COMPLETE_NO_PROMOTION'}),
      'ncaaf_v42_conditional_features':wrap(run_cloud_research=lambda **kw:{'status':'EXPLORATORY_RESEARCH__SOURCE_VINTAGE_HOLD', 'predictions_uri':'gs://x'}),
      'ncaaf_core_challenger_v2':wrap(run_ncaaf_core_challenger_v2=lambda **kw:{'status':'NCAAF_CORE_CHALLENGER_V2_COMPLETE'}),
      'ncaaf_research_v2':wrap(run_ncaaf_research_v2=lambda **kw:{'status':'NCAAF_RESEARCH_V2_COMPLETE',
            'system_miner_v3':{'spreads':{'mechanism_families':[{'confirmation_pass':True,'direction':'PLAY_ON',
                    'mechanism_id':'MIN1','representative_conditions':['ATOM']}]}}},
            _extended_atoms=lambda *a,**kw:[{'name':'ATOM','mask':np.array([True])}],
            _pt_load_canonical_team_aliases=lambda *a,**kw:{}),
      'ncaaf_v5_betting_research':wrap(cloud_run=lambda **kw:{'status':'HISTORICAL_DIRECTIONAL_RESEARCH__SOURCE_VINTAGE_HOLD'}),
      'ncaaf_v51_edge_research':wrap(cloud_run=lambda **kw:{'v4_prediction_verified_with_source':True,
            'dedup_ledger_uri':uri,'physical_games':2,'dedup_model_game_rows':3}),
      'ncaaf_v52_brain_research':wrap(cloud_run=lambda **kw:{'status':'SOURCE_DISCOVERY_REQUIRED__NO_CONFIGURED_BRAIN_SOURCES',
            'verified_system_game_records':0}),
    }
    messages=[]
    report=mod.run_full(storage_client=st,bq_client=object(),modules=modules,log_func=messages.append)
    assert report['status']=='COMPLETE_FULL_RESEARCH__SHADOW_AUTHORITY__VINTAGE_HOLD'
    assert report['bet_authority']==0
    assert report['miner_occurrence_join']['matched_system_game_rows']==1
    assert 'research/ncaaf/v53_full_research/latest_report.json' in st.data
    assert any('FULL_LEGACY_PATHI_BIGAL_PT_MINER_STATE_RESEARCH' in x for x in messages)
    assert not any('bet_authority=1' in x for x in messages)


def test_full_job_failure_persists_audit_and_raises(monkeypatch):
    import ncaaf_v53_full_research as mod
    monkeypatch.setattr(mod,'module_contract',lambda m:{k:{'status':'PASS','sha256':'fake'} for k in m})
    monkeypatch.setattr(mod,'legacy_source_constants',lambda *a:{'status':'PASS'})
    monkeypatch.setattr(mod,'source_manifest',lambda **kw:{'objects':{},'tables':{}})
    st=_FakeStorage();cache={'games':pd.DataFrame([{'Season':2024}]),'miner_games':pd.DataFrame([{'Season':2024}])}
    dash=SimpleNamespace(_V1357_SPREAD_RESEARCH_CACHE=cache,V133_DEPLOY_BUILD_ID='same',_V143_SYSTEM_HISTORY_CACHE={},
         fit_historical_ncaaf_core_expert=lambda *a,**kw:True,
         fit_ncaaf_statistical_brain=lambda **kw:True)
    modules={k:SimpleNamespace() for k in mod.REQUIRED_LEGACY}
    modules['sharp_line_dashboard']=dash
    modules['utils']=SimpleNamespace(V133_DEPLOY_BUILD_ID='same')
    modules['ncaaf_margin_tournament_v4']=SimpleNamespace(run_cloud_research=lambda **kw: (_ for _ in ()).throw(RuntimeError('EXACT_V4_FAILURE')))
    with pytest.raises(RuntimeError,match='V53_FULL_RESEARCH_FAILED'):
        mod.run_full(storage_client=st,bq_client=object(),modules=modules,log_func=lambda s:None)
    latest=__import__('json').loads(st.data['research/ncaaf/v53_full_research/latest_report.json'])
    assert latest['status']=='FAILED_FULL_RESEARCH__AUDIT_PUBLISHED'
    assert 'EXACT_V4_FAILURE' in latest['research_components']['failure']['error']
    assert any(z['status']=='FAIL' for z in latest['stages'])


def test_pt_legacy_auto_promotion_is_held_and_restored():
    import ncaaf_v53_full_research as mod
    calls=[]
    def real(*a,**kw):calls.append('REAL_PROMOTION');return {'status':'PROMOTED'}
    def run(**kwargs):
        diag=rv.pt_promotion_settle_and_review(allow_promotion=True)
        assert diag['review']['status']=='V53_RESEARCH_ONLY_NO_PROMOTION'
        return {'status':'NCAAF_RESEARCH_V2_COMPLETE'}
    rv=SimpleNamespace(pt_promotion_settle_and_review=real,run_ncaaf_research_v2=run)
    events=[];audit=SimpleNamespace(emit=lambda *a,**kw:events.append((a,kw)))
    assert mod.run_legacy_with_promotion_held(rv,audit=audit)['status']=='NCAAF_RESEARCH_V2_COMPLETE'
    assert rv.pt_promotion_settle_and_review is real and calls==[]
    assert any(z[0][1]=='BLOCKED' for z in events)


def test_legacy_short_school_name_maps_via_original_alias_registry():
    h=history()
    h['Pathi_FB_Dog_On_7']['occurrences'][0]['team']='Michigan'
    h['Pathi_FB_Dog_On_7']['occurrences'][0]['opponent']='Ohio State'
    a,diag=export_legacy_occurrences(sample(),h,aliases={
        'michigan':'michiganwolverines','ohiostate':'ohiostatebuckeyes'})
    assert diag['joined']==3 and diag['unmatched']==0

def test_dashboard_restores_weekly_and_full_v5_and_train_routes():
    w=Path(__file__).parent
    ui=(w/'sharp_line_dashboard.py').read_text()
    job=(w/'train_job.py').read_text()
    assert '"ncaaf_production_weekly",\n            "ncaaf_v53_full_research"' in ui
    assert '"ncaaf_v53_full_research": "NCAAF V5 — FULL Research' in ui
    assert 'market).lower().strip() == "ncaaf_v53_full_research"' in job
    assert 'if _ncaaf_weekly_run:' in job
    assert '"ncaaf_v52_brain_attribution",\n            "ncaaf_research_heavy"' not in ui


def test_full_job_never_claims_live_v5_production():
    import ncaaf_v53_full_research as mod
    edges=sample()
    shadow,pol=mod.policy_shadow(edges,pd.DataFrame(columns=['Season','Game_ID','Family','System_ID','Side','Evidence_Class']))
    assert pol['decision_contract'].startswith('MODEL_POINT_EDGE')
    assert (shadow.V5_Bet_Authority==0).all()
    assert 'P_Cover_Empirical' not in shadow.columns
