import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import hashlib,json
import numpy as np
import pandas as pd
import pytest
import ncaaf_v54_cfbd_truth as m

def tiny():
    game={'id':100,'homeTeam':'Alpha','awayTeam':'Beta','homePoints':21,'awayPoints':17,
          'startDate':'2024-09-07T19:30:00Z','completed':True}
    path='/games'; params=json.dumps({'year':2024,'classification':'fbs'},sort_keys=True,separators=(',',':'))
    payload=json.dumps(game,sort_keys=True,separators=(',',':'))
    h=hashlib.sha256(f'{path}|{params}|{payload}'.encode()).hexdigest()
    archive=pd.DataFrame([{'Archive_Row_ID':h,'Endpoint':path,'Params_JSON':params,
        'Payload_JSON':payload,'Season':2024,'Retrieved_At':'2024-09-08T01:00:00Z'}])
    feat=pd.DataFrame([
        {'Season':2024,'Game_ID':100,'Game_Date':'2024-09-07T19:30:00Z', 'Team':'Alpha', 'Opponent':'Beta',
         'Team_Key':'alpha','Opponent_Key':'beta','Week':2,'CFBD_Season_Net_PPA_Prior':0.4,
         'CFBD_Game_Off_PPA':2.4},
        {'Season':2024,'Game_ID':100,'Game_Date':'2024-09-07T19:30:00Z', 'Team':'Beta', 'Opponent':'Alpha',
         'Team_Key':'beta','Opponent_Key':'alpha','Week':2,'CFBD_Season_Net_PPA_Prior':-0.4,
         'CFBD_Game_Off_PPA':1.4}])
    mk=pd.DataFrame([
        {'Source_Name':'CollegeFootballData','Season':2024,'Source_Game_ID':100,'Source_Team_Name':'Alpha','Is_Home':1,'Consensus_Open_Spread':-7.5,'Consensus_Close_Spread':-7},
        {'Source_Name':'CollegeFootballData','Season':2024,'Source_Game_ID':100,'Source_Team_Name':'Beta','Is_Home':0,'Consensus_Open_Spread':7.5,'Consensus_Close_Spread':7}])
    return archive,feat,mk

def test_hash_games_and_home_orientation():
    archive,feat,mk=tiny();games,a=m.parse_cfbd_archive_rows(archive)
    assert a['bad_hash']==0 and len(games)==1
    sides,diag=m.build_cfbd_canonical_features(feat,games,enforce_full_history=False)
    assert diag['score_match_rate']==1
    assert sides.set_index('Team').loc['Alpha','Actual_Margin']==4
    assert sides.set_index('Team').loc['Beta','Actual_Margin']==-4
    assert 'CFBD_Game_Off_PPA' not in sides
    assert 'CFBD_Season_Net_PPA_Prior' in sides
    j,aud=m.attach_market_only(sides,mk)
    assert set(j.ATS_Win.dropna())=={0.0,1.0}
    assert (j.Market_Open_Margin+j.Consensus_Open_Spread).abs().max()==0
    assert not any(x.endswith('Score') for x in aud['market_fields_imported'])
    assert len(m.home_games(j))==1
    assert m.verify_cfbd_frame(m.home_games(j),require_unique_home=True)['status']=='PASS'

def test_bad_archive_hash_rejected():
    archive,*_=tiny();archive.loc[0,'Archive_Row_ID']='bad'
    with pytest.raises(RuntimeError,match='INTEGRITY_FAILURE'):m.parse_cfbd_archive_rows(archive)

def test_api_conflicting_final_scores_rejected():
    a,_,_=tiny();b=a.copy();obj=json.loads(b.loc[0,'Payload_JSON']);obj['homePoints']=25
    p=json.dumps(obj,sort_keys=True,separators=(',',':'));b.loc[0,'Payload_JSON']=p
    b.loc[0,'Archive_Row_ID']=hashlib.sha256((b.loc[0,'Endpoint']+'|'+b.loc[0,'Params_JSON']+'|'+p).encode()).hexdigest()
    with pytest.raises(RuntimeError,match='CONFLICTING_FINAL_SCORE'):
        m.parse_cfbd_archive_rows(pd.concat([a,b],ignore_index=True))

def test_raw_market_score_must_never_be_read():
    a,f,mkt=tiny();g,_=m.parse_cfbd_archive_rows(a)
    sides,_=m.build_cfbd_canonical_features(f,g,enforce_full_history=False)
    mkt['Team_Score']=[999,999]; mkt['Opponent_Score']=[999,999];mkt['Actual_Margin']=[900,900]
    j,_=m.attach_market_only(sides,mkt)
    assert set(j.Team_Score)=={17,21} and set(j.Actual_Margin)=={-4,4}
    assert not (j.Actual_Margin==900).any()

def test_non_cfbd_market_sources_cannot_enter():
    a,f,mkt=tiny();g,_=m.parse_cfbd_archive_rows(a);s,_=m.build_cfbd_canonical_features(f,g,enforce_full_history=False)
    mkt['Source_Name']='BigDataBall'
    with pytest.raises(RuntimeError,match='CFBD_GAME_ID_SOURCE_EMPTY'):m.attach_market_only(s,mkt)

def test_cfbd_game_id_conflict_or_team_identity_rejected():
    a,f,_=tiny();g,_=m.parse_cfbd_archive_rows(a);f.loc[0,'Team']='Other';f.loc[0,'Team_Key']='other'
    # Feature side reciprocity is deliberately strict.
    with pytest.raises(RuntimeError,match='RECIPROCITY_FAILURE'):
        m.build_cfbd_canonical_features(f,g,enforce_full_history=False)

def test_conflicting_market_lines_rejected():
    a,f,mkt=tiny();g,_=m.parse_cfbd_archive_rows(a);s,_=m.build_cfbd_canonical_features(f,g,enforce_full_history=False)
    mkt.loc[1,'Consensus_Open_Spread']=9
    with pytest.raises(RuntimeError,match='MARKET_SPREAD_SIDES_NOT_OPPOSITES'):
        m.attach_market_only(s,mkt)

def test_prior_state_no_current_day_leakage():
    a,f,mkt=tiny();g,_=m.parse_cfbd_archive_rows(a);s,_=m.build_cfbd_canonical_features(f,g,enforce_full_history=False)
    j,_=m.attach_market_only(s,mkt);h=m.derived_pregame_states(j)
    assert h.Game_Number_Prior.eq(0).all()
    assert h.Prev_SU_Win.isna().all()
    assert h.Opp_Game_Number_Prior.eq(0).all()

def test_no_2026_historical_outcomes():
    a,f,mkt=tiny();g,_=m.parse_cfbd_archive_rows(a);s,_=m.build_cfbd_canonical_features(f,g,enforce_full_history=False)
    s.loc[0,'Season']=2026
    with pytest.raises(RuntimeError,match='2026_LABEL_ACCESS'):
        m.verify_cfbd_frame(s)

def test_no_legacy_bridge_on_authoritative_v4_source():
    src=Path(__file__).with_name('ncaaf_margin_tournament_v4.py').read_text()
    src2=Path(__file__).with_name('ncaaf_v42_conditional_features.py').read_text()
    for s in (src,src2):
        assert "if 'Historical_Football_Source' in history:" in s
        assert 'truth.verify_cfbd_frame(history' in s
        assert "'legacy_bridge_invoked':False" in s

def test_research_dropdown_and_weekly_present():
    dashboard=Path(__file__).with_name('sharp_line_dashboard.py').read_text()
    training=Path(__file__).with_name('train_job.py').read_text()
    assert 'ncaaf_production_weekly' in dashboard
    assert 'ncaaf_v54_full_research' in dashboard
    assert 'ncaaf_v54_full_research' in training
    assert 'import ncaaf_v54_full_research' in training
    assert 'NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED' in dashboard

def test_shadow_no_authority():
    src=Path(__file__).with_name('ncaaf_v54_full_research.py').read_text()
    assert "bet_authority':0" in src and "status='COMPLETE_FULL_RESEARCH__SHADOW_AUTHORITY__VINTAGE_HOLD'" in src
