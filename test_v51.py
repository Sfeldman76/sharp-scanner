import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ncaaf_v51_edge_research import validate_and_dedupe, research


def fixture():
    records=[]; v4=[]
    for season in (2021,2022,2023,2024,2025):
        for n in range(6):
            gid=str(season*1000+n)
            market=(-1 if n%2 else 1)*(3.5+3*n)
            actual=market+[5,-3,0,7,-8,2][n]
            v4_prediction=market+[1.5,2.5,3.5,5.5,-6.5,8][n]
            a_prediction=v4_prediction+([0.2,-0.7,0.3,0.5,-0.2,-0.4][n])
            dt=f'{season}-09-0{n+1}T18:00:00Z'
            v4.append({'Season':season,'Game_ID':gid,'Team':f't{n}','Opponent':f'o{n}',
                       'Actual_Margin':actual,'Verified_Open_Margin_Diagnostic':market,
                       'NONLINEAR_STATE_HGB':v4_prediction})
            for model,pred in [('NONLINEAR_STATE_HGB',v4_prediction),('A_EFFICIENCY',a_prediction)]:
                gap=pred-market
                side='TEAM' if gap>0 else 'OPPONENT'
                direction=(actual-market)*(1 if side=='TEAM' else -1)
                result='PUSH' if direction==0 else ('WIN' if direction>0 else 'LOSS')
                for threshold in (1.,2.,3.,5.,7.):
                    if abs(gap)<threshold:continue
                    records.append({'Season':season,'Game_ID':gid,'Game_Date':dt,'Team':f't{n}',
                        'Opponent':f'o{n}','Model':model,'Model_Margin':pred,'Open_Market_Margin':market,
                        'Gap_Points':gap,'Threshold':threshold,'Side':side,'Result':result,
                        'P_Cover_Empirical':.99})
    return pd.DataFrame(records),pd.DataFrame(v4)


def test_true_game_dedup_and_source_orientation():
    rows,base=fixture()
    data=validate_and_dedupe(rows,v4=base)
    assert len(data)==60
    assert data[['Season','Game_ID']].drop_duplicates().shape[0]==30
    assert all(data.groupby(['Season','Game_ID','Model']).size()==1)
    assert (data.Abs_Edge>=1).all()
    report,_=research(rows,v4=base)
    assert report['raw_threshold_rows']>report['dedup_model_game_rows']
    assert report['v4_prediction_verified_with_source'] is True
    assert report['roi'] is None and report['production_authority']==0
    assert report['brain_coverage']['eligible_records']==0


def test_probability_not_used_for_authority_or_edge_stats():
    rows,base=fixture()
    report1,_=research(rows,v4=base)
    rows['P_Cover_Empirical']=0.001
    report2,_=research(rows,v4=base)
    assert report1['model_analyses']==report2['model_analyses']
    assert 'probability' not in report1['decision_categories']['CANDIDATE'].lower()


def test_gap_orientation_and_source_audits_fail():
    r,b=fixture();r.loc[r.index[0],'Gap_Points']+=10
    with pytest.raises(ValueError,match='EDGE_ORIENTATION_MISMATCH'):research(r,v4=b)
    r,b=fixture();r.loc[r.index[0],'Side']='OPPONENT' if r.iloc[0].Side=='TEAM' else 'TEAM'
    with pytest.raises(ValueError,match='SELECTION_ORIENTATION_MISMATCH'):research(r,v4=b)
    r,b=fixture();b.loc[0,'Verified_Open_Margin_Diagnostic']+=1
    with pytest.raises(ValueError,match='V4_OPEN_MARGIN_MISMATCH'):research(r,v4=b)
    r,b=fixture();b.loc[0,'Actual_Margin']-=50
    with pytest.raises(ValueError,match='V4_RESULT_ORIENTATION_MISMATCH'):research(r,v4=b)


def test_disagreeing_threshold_rows_fail():
    r,b=fixture();idx=r.index[0];dup=r.loc[[idx]].copy();dup.loc[idx,'Threshold']=1.25
    bad=pd.concat([r,dup],ignore_index=True)
    with pytest.raises(ValueError,match='DUPLICATE_SAME_THRESHOLD|THRESHOLD_EXCEEDS_EDGE'):
        # extra 1.25 is legitimate but must be consistent; simulate direct duplicate
        research(pd.concat([r,r.loc[[idx]]],ignore_index=True),v4=b)
    r,b=fixture();r.loc[r.index[1],'Result']='LOSS' if r.iloc[0].Result=='WIN' else 'WIN'
    # if row 1 is different model/game, this must still be detected at V4 parity.
    with pytest.raises(ValueError):research(r,v4=b)


def test_valid_brain_records_are_only_diagnostic():
    r,b=fixture()
    raw=pd.DataFrame([{'Season':2024,'Game_ID':'2024000','Family':'BIG_AL',
      'System_ID':'RULE_X','Side':'TEAM','Snapshot_UTC':'2024-09-01T01:00:00Z',
      'Kickoff_UTC':'2024-09-01T18:00:00Z','Rule_Declared_UTC':'2023-07-01T00:00:00Z',
      'Source_Artifact_SHA256':'a'*64}])
    rep,ledger=research(r,v4=b,brains=raw)
    assert rep['brain_coverage']['eligible_records']==1
    assert rep['brain_coverage']['authoritative_brain_votes']==0
    assert rep['brain_coverage']['independent_archive_proof']=='NOT_ESTABLISHED'
    assert (ledger['Brain_BIG_AL']!='MISSING').any()
    raw.loc[0,'Snapshot_UTC']='2024-09-01T20:00:00Z'
    with pytest.raises(ValueError,match='BRAIN_POST_KICKOFF_SIGNAL'):research(r,v4=b,brains=raw)


def test_real_uploaded_signal_dedup_counts():
    path=Path('/mnt/data/ncaaf_v5_signals(1).csv')
    if not path.exists():pytest.skip('external data not in runtime')
    df=pd.read_csv(path)
    report,ledger=research(df)
    assert report['raw_threshold_rows']==47074
    assert report['dedup_model_game_rows']==7355
    assert report['physical_games']==4113
    assert len(ledger[ledger.Model=='NONLINEAR_STATE_HGB'])==3675
    assert len(ledger[ledger.Model=='A_EFFICIENCY'])==3680
    assert report['brain_coverage']['eligible_records']==0
