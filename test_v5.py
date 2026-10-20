import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, pytest
from ncaaf_v5_betting_research import join_frozen_predictions,run_historical
from ncaaf_v5_shadow_lock import build_shadow


def fixture():
    rows=[]; rng=np.random.default_rng(21)
    for year in range(2017,2026):
        for g in range(140):
            dt=pd.Timestamp(year=year,month=9,day=1,tz='UTC')+pd.Timedelta(days=g)
            b=float(rng.normal(0,12));actual=b+float(rng.normal(0,13))
            rows.append(dict(Season=year,Game_ID=f'{year}-{g}',Game_Date=dt.isoformat(),Team=f't{g}',Opponent=f'o{g}',
                Actual_Margin=actual,NONLINEAR_STATE_HGB=b, A_EFFICIENCY=b+0.3,
                AB_EFF_PASS=b+0.15, AC_EFF_DRIVE=b+0.2,
                Verified_Open_Margin_Diagnostic=b+float(rng.normal(0,2)) if year>=2021 else np.nan))
    df=pd.DataFrame(rows)
    v4=df[['Season','Game_ID','Game_Date','Team','Opponent','Actual_Margin',
           'NONLINEAR_STATE_HGB','Verified_Open_Margin_Diagnostic']].copy()
    v42=df[['Season','Game_ID','Game_Date','Actual_Margin','NONLINEAR_STATE_HGB',
           'A_EFFICIENCY','AB_EFF_PASS','AC_EFF_DRIVE']].copy()
    return v4,v42


def test_research_complete_with_all_sources():
    v4,v42=fixture()
    rep,sig,cal=run_historical(v4,v42)
    assert rep['historical_games']==1260
    assert rep['baseline_parity']=='EXACT_V4_TO_V42_PER_GAME'
    assert len(sig)>0
    assert rep['historical_roi'] is None
    assert set(sig.Model.unique())==set(cal['model_calibration'])
    assert sig.loc[sig.Season==2024,'Calibration_N'].min()==140*7
    assert rep['brain_family_status']['status']=='NO_VERIFIABLE_SNAPSHOTS_PROVIDED'
    assert all(sig['Bet_Authority']==0) if 'Bet_Authority' in sig else True


def test_wrong_baseline_fails():
    v4,v42=fixture();v42.loc[1,'NONLINEAR_STATE_HGB']+=1
    with pytest.raises(ValueError,match='FROZEN_V4_BASELINE_PARITY_BROKEN'):run_historical(v4,v42)


def test_duplicate_and_extra_future_game_fails():
    v4,v42=fixture()
    v4=pd.concat([v4,v4.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match='DUPLICATE_GAME_KEYS'):run_historical(v4,v42)
    v4,v42=fixture();v42.loc[0,'Season']=2026
    with pytest.raises(ValueError,match='INVALID_HISTORICAL_SEASONS'):run_historical(v4,v42)


def test_outcome_edit_future_does_not_change_earlier_calibration():
    v4,v42=fixture()
    _,s1,_=run_historical(v4,v42)
    m=v4.Season==2025
    v4.loc[m,'Actual_Margin']+=200
    v42.loc[v42.Season==2025,'Actual_Margin']+=200
    _,s2,_=run_historical(v4,v42)
    j=s1.loc[s1.Season<2025,['Game_ID','Model','Threshold','P_Cover_Empirical']].reset_index(drop=True)
    k=s2.loc[s2.Season<2025,['Game_ID','Model','Threshold','P_Cover_Empirical']].reset_index(drop=True)
    pd.testing.assert_frame_equal(j,k)
    # 2025 forecast residual calibration also excludes 2025 labels.
    x=s1.loc[s1.Season==2025,'P_Cover_Empirical'].to_numpy()
    y=s2.loc[s2.Season==2025,'P_Cover_Empirical'].to_numpy()
    np.testing.assert_allclose(x,y)


def test_brain_requires_historical_vintage_and_rule_predeclared():
    v4,v42=fixture();g='2024-0'
    brain=pd.DataFrame([{'Game_ID':g,'Season':2024,'Family':'BIG_AL','System_ID':'BA_R1','Side':'TEAM',
        'Snapshot_UTC':'2024-09-01T01:00:00Z','Kickoff_UTC':'2024-09-01T20:00:00Z',
        'Rule_Declared_UTC':'2023-01-01T00:00:00Z','Source_Artifact_SHA256':'a'*64}])
    rep,ledger,_=run_historical(v4,v42,brain_raw=brain)
    assert rep['brain_family_status']['pregame_records_kept']==1
    assert ledger.Brain_BIG_AL.isin(['CONFIRM','OPPOSE']).any()
    brain.loc[0,'Snapshot_UTC']='2024-09-01T21:00:00Z'
    with pytest.raises(ValueError,match='BRAIN_POST_KICKOFF_SIGNAL'):run_historical(v4,v42,brain_raw=brain)


def test_uncertified_brain_not_qualified():
    v4,v42=fixture()
    brain=pd.DataFrame([{'Game_ID':'2024-0','Season':2024,'Family':'SYSTEM_MINER','System_ID':'NEW2024',
        'Side':'TEAM','Snapshot_UTC':'2024-09-01T01:00:00Z','Kickoff_UTC':'2024-09-01T20:00:00Z',
        'Rule_Declared_UTC':'2024-08-01T00:00:00Z','Source_Artifact_SHA256':'f'*64}])
    rep,ledger,_=run_historical(v4,v42,brain_raw=brain)
    assert rep['brain_family_status']['preseason_rule_records']==0
    assert ledger.Brain_SYSTEM_MINER.eq('MISSING').all()


def test_no_historical_odds_means_no_roi():
    v4,v42=fixture();v4.loc[:,'Verified_Open_Margin_Diagnostic']=np.nan
    report,ledger,_=run_historical(v4,v42)
    assert report['status']=='NO_VERIFIED_HISTORICAL_OPEN_MARKET_GAMES__RESEARCH_BLOCKED'
    assert ledger.empty and report['historical_roi'] is None


def test_shadow_timestamp_quotes_and_brain_contract():
    pred=pd.DataFrame([{'Game_ID':'game1','Season':2026,'Team':'Georgia','Opponent':'Florida','Model':'A_EFFICIENCY',
        'Model_Margin':7,'Forecast_AsOf_UTC':'2026-10-22T14:00:00Z','Feature_Cutoff_UTC':'2026-10-22T12:00:00Z',
        'Kickoff_UTC':'2026-10-24T18:00:00Z','Model_Artifact_SHA256':'a'*64,
        'Feature_Source_SHA256':'b'*64,'Feature_Vintage_Certified':False}])
    quote=pd.DataFrame([{'Game_ID':'game1','Season':2026,'Team':'Georgia','Opponent':'Florida',
        'Quote_AsOf_UTC':'2026-10-22T14:03:00Z','Kickoff_UTC':'2026-10-24T18:00:00Z',
        'Book':'Book A','Market_Team_Margin':3.5,'Team_American_Odds':-110,
        'Opponent_American_Odds':-110,'Quote_Source_SHA256':'c'*64}])
    calibr={'model_calibration':{'A_EFFICIENCY':{'type':'EMPIRICAL_OOF_MARGIN_ERROR_CDF',
        'residuals':np.linspace(-35,35,600).tolist()}}}
    rep,obs=build_shadow(pred,quote,calibr)
    assert len(obs)==1
    assert obs.iloc[0]['Research_Status']=='SHADOW_HOLD_UNCERTIFIED'
    assert obs.iloc[0]['Bet_Authority']==0
    quote.loc[0,'Quote_AsOf_UTC']='2026-10-24T19:00:00Z'
    with pytest.raises(ValueError,match='QUOTE_POST_KICKOFF'):build_shadow(pred,quote,calibr)


def test_cloud_artifact_io_with_versioned_parity(monkeypatch):
    import json,sys
    sys.path.insert(0,'/mnt/data/v43_focus')
    from ncaaf_v5_betting_research import cloud_run
    v4,v42=fixture()
    prefix='research/ncaaf/conditional_features_v42/'
    latest={'status':'EXPLORATORY_RESEARCH__SOURCE_VINTAGE_HOLD','generated_utc':'2026-10-20T16:00:00Z',
        'experiment_spec_sha256':'abcdef',
        'parity_to_published_v4':{str(y):'EXACT_REPRODUCTION' for y in range(2017,2026)}}
    backing={prefix+'latest_report.json':json.dumps(latest),
       prefix+'runs/test/report.json':json.dumps(latest),
       prefix+'runs/test/game_predictions.csv':v42.to_csv(index=False),
       'research/ncaaf/margin_tournament_v4/game_level_predictions.csv':v4.to_csv(index=False)}
    class Blob:
        def __init__(self,name):self.name=name
        def download_as_text(self):return backing[self.name]
        def upload_from_string(self,value,**kwargs):
            if kwargs.get('if_generation_match')==0 and self.name in backing:
                raise ValueError('WOULD_OVERWRITE_VERSIONED_OBJECT')
            backing[self.name]=value
    class Bucket:
        def __init__(self):self.name='sharp-models'
        def blob(self,name):return Blob(name)
        def list_blobs(self,prefix):return [Blob(name) for name in backing if name.startswith(prefix)]
    class Client:
        def bucket(self,name):assert name=='sharp-models';return Bucket()
    rep=cloud_run(storage_client=Client())
    assert rep['status']=='HISTORICAL_DIRECTIONAL_RESEARCH__SOURCE_VINTAGE_HOLD'
    assert rep['report_uri'].startswith('gs://sharp-models/research/ncaaf/v5_market_value/runs/')
    assert 'research/ncaaf/v5_market_value/latest_report.json' in backing
    assert rep['brain_family_status']['family_records']['BIG_AL']==0
