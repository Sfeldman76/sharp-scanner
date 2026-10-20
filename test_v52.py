from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import pandas as pd
import pytest
from ncaaf_v52_brain_bridge import normalize_records,verify_gcs_originals,attribute
from ncaaf_v52_brain_research import analyze,load_source_map
from ncaaf_v52_source_inventory import _candidate_table


def sample_edges():
    return pd.DataFrame([
     {'Season':2024,'Game_ID':'g1','Game_Date':'2024-09-07','Team':'michigan','Opponent':'ohio state','Model':'NONLINEAR_STATE_HGB',
      'Side':'TEAM','Result':'WIN','Edge_Bin':'3–5','Spread_Bin':'7–14','Selection_Type':'FAVORITE','Period':'examined_2024_2025'},
     {'Season':2024,'Game_ID':'g1','Game_Date':'2024-09-07','Team':'michigan','Opponent':'ohio state','Model':'A_EFFICIENCY',
      'Side':'OPPONENT','Result':'LOSS','Edge_Bin':'5–7','Spread_Bin':'7–14','Selection_Type':'UNDERDOG','Period':'examined_2024_2025'},
     {'Season':2025,'Game_ID':'g2','Game_Date':'2025-09-06','Team':'alabama','Opponent':'auburn','Model':'NONLINEAR_STATE_HGB',
      'Side':'OPPONENT','Result':'WIN','Edge_Bin':'5–7','Spread_Bin':'3–7','Selection_Type':'UNDERDOG','Period':'examined_2024_2025'},
    ])


def spec():
    return {'table':'sharplogger.sharp_data.original_brain_log','format':'long','side_orientation':'V4_CANONICAL',
            'mapping':{'season':'Season','game_id':'Game_ID','family':'Family','system_id':'System_ID',
                       'side':'Side','snapshot_utc':'Snapshot_UTC','kickoff_utc':'Kickoff_UTC',
                       'rule_declared_utc':'Rule_Declared_UTC','source_gcs_uri':'Source_GCS_URI',
                       'source_artifact_sha256':'Source_Artifact_SHA256'}}


def fixture():
    payload=('Season,Game_ID,Family,System_ID,Side,Snapshot_UTC,Kickoff_UTC,Rule_Declared_UTC\n'
       '2024,g1,BIG_AL,RULE1,TEAM,2024-09-06T12:00:00Z,2024-09-07T19:00:00Z,2023-08-01T00:00:00Z\n').encode()
    h=hashlib.sha256(payload).hexdigest()
    df=pd.DataFrame([{'Season':2024,'Game_ID':'g1','Family':'BIG_AL','System_ID':'RULE1','Side':'TEAM',
     'Snapshot_UTC':'2024-09-06T12:00:00Z','Kickoff_UTC':'2024-09-07T19:00:00Z',
     'Rule_Declared_UTC':'2023-08-01T00:00:00Z','Source_GCS_URI':'gs://sharp-models/old/2024.csv',
     'Source_Artifact_SHA256':h}])
    return df,payload


class Blob:
    def __init__(self,payload,created):
        self.payload=payload;self.time_created=pd.Timestamp(created);self.size=len(payload)
    def reload(self):pass
    def download_as_bytes(self):return self.payload
class Bucket:
    def __init__(self,blob):self.blobobj=blob
    def blob(self,key):return self.blobobj
class Storage:
    def __init__(self,blob):self.the_blob=blob
    def bucket(self,key):return Bucket(self.the_blob)


def test_no_config_is_discovery_not_finished_betting_research():
    rep,led,src=analyze(sample_edges())
    assert rep['status']=='SOURCE_DISCOVERY_REQUIRED__NO_CONFIGURED_BRAIN_SOURCES'
    assert rep['bet_authority']==0 and rep['brain_candidates']==0 and rep['roi'] is None
    assert all((led[f'Brain_{family}']=='MISSING').all() for family in ('BIG_AL','PATHI','PREDICTION_TRACKER','SYSTEM_MINER','STATE_BOUNCE_BACK'))


def test_claimed_before_game_cannot_be_counted_as_verified():
    df,payload=fixture()
    rep,ledger,c=analyze(sample_edges(), config={'sources':[spec()]},source_frames=[df])
    assert len(c)==1 and c.preseason_declared.iloc[0]
    assert not c.archive_verified.iloc[0]
    assert rep['status']=='BLOCKED_NO_INDEPENDENTLY_VERIFIED_BRAIN_HISTORY'
    assert (ledger.Brain_BIG_AL=='MISSING').all()


def test_true_original_archive_byte_hash_and_gcs_creation_proof():
    df,payload=fixture()
    c,a=normalize_records(df,spec(),sample_edges())
    assert a['matched_active_records']==1
    verified,audit=verify_gcs_originals(c,Storage(Blob(payload,'2024-09-06T13:00:00Z')))
    assert verified.archive_verified.iloc[0]
    report,ledger=attribute(sample_edges(),verified)
    assert report['verified_system_game_records']==1
    assert ledger.iloc[0]['Brain_BIG_AL']=='CONFIRM'
    assert ledger.iloc[1]['Brain_BIG_AL']=='OPPOSE'
    assert report['bet_authority']==0


def test_uploaded_after_kickoff_is_not_proof():
    df,payload=fixture()
    c,a=normalize_records(df,spec(),sample_edges())
    v,a=verify_gcs_originals(c,Storage(Blob(payload,'2024-09-08T13:00:00Z')))
    assert not v.archive_verified.iloc[0]
    assert attribute(sample_edges(),v)[0]['verified_system_game_records']==0


def test_forged_artifact_sha_rejected():
    df,payload=fixture()
    df['Source_Artifact_SHA256']='0'*64
    c,a=normalize_records(df,spec(),sample_edges())
    v,a=verify_gcs_originals(c,Storage(Blob(payload,'2024-09-06T13:00:00Z')))
    assert not v.archive_verified.iloc[0]
    assert a[0]['status']=='HASH_MISMATCH'


def test_missing_game_never_attaches():
    df,payload=fixture()
    df.loc[0,'Game_ID']='fake'
    c,a=normalize_records(df,spec(),sample_edges())
    assert c.empty
    assert a['status']=='ZERO_GAME_MATCH'


def test_post_kickoff_claim_no_signal_vote():
    df,payload=fixture()
    df['Snapshot_UTC']='2024-09-07T20:00:00Z'
    c,a=normalize_records(df,spec(),sample_edges())
    assert not c.time_claim_valid.iloc[0]
    assert not c.preseason_declared.iloc[0]
    v,a=verify_gcs_originals(c,Storage(Blob(payload,'2024-09-06T13:00:00Z')))
    assert not v.archive_verified.iloc[0]


def test_wide_derived_flags_never_auto_represent_pregame_record():
    mapping={'season':'Season','game_id':'Game_ID','side_team':'SelectedTeam'}
    sp={'table':'sharplogger.sharp_data.rebuilt_context','format':'wide','mapping':mapping,
        'system_columns':{'Pathi_FB_Dog_On_7':'PATHI','BigAl_CF_SecondGame':'BIG_AL'}}
    df=pd.DataFrame([{'Season':2024,'Game_ID':'g1','SelectedTeam':'michigan',
       'Pathi_FB_Dog_On_7':1,'BigAl_CF_SecondGame':1}])
    candidates,stats=normalize_records(df,sp,sample_edges())
    assert len(candidates)==2
    assert not candidates.archive_verified.any()
    assert not candidates.time_claim_valid.any()
    assert attribute(sample_edges(),candidates)[0]['verified_system_game_records']==0


def test_source_config_requires_unambiguous_long_side():
    s=spec();s.pop('side_orientation')
    with pytest.raises(ValueError,match='UNVERIFIED_SIDE_REFERENCE'):
        load_source_map({'sources':[s]})
    s['side_orientation']='V4_CANONICAL'
    assert load_source_map({'sources':[s]})['sources'][0]['table'].endswith('original_brain_log')


def test_inventory_ranks_existing_bigal_and_pathi_context_without_proof():
    x=_candidate_table('scores_with_features',['Sport','Season','Game_Start','Pathi_FB_Dog_On_7','BigAl_CF_SecondGame'])
    assert x['family_column_counts']['PATHI']==1
    assert x['family_column_counts']['BIG_AL']==1
    assert not x['has_all_archive_proof_fields']


def test_real_v51_ledger_no_unsound_bets():
    fixture_path=Path('/mnt/data/v51_edge_engine_work/sample_analysis/unique_model_game_edges.csv')
    if not fixture_path.exists():pytest.skip('real fixture unavailable')
    edges=pd.read_csv(fixture_path)
    rep,led,c=analyze(edges)
    assert rep['games']==4113 and rep['edge_model_rows']==7355
    assert rep['verified_system_game_records']==0
    assert rep['bet_authority']==0
