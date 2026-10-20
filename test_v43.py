"""Synthetic V4.3 validation / prospective production audit tests."""
import copy
import numpy as np
import pandas as pd
import ncaaf_v43_validation as v43
import ncaaf_prod_v1_robustness as prod

rng=np.random.default_rng(1937)
rows=[];folds={}
for year in range(2017,2026):
    for j in range(60):
        y=float(rng.normal(0,14));base=y+rng.normal(0,16)
        rows.append(dict(Season=year,Game_Date=f'{year}-09-{1+(j%20):02d}',Game_ID=f'{year}-{j}',
            Actual_Margin=y,NONLINEAR_STATE_HGB=base,
            A_EFFICIENCY=y+.96*(base-y)+rng.normal(0,1),
            AB_EFF_PASS=y+.97*(base-y)+rng.normal(0,1),
            AC_EFF_DRIVE=y+.98*(base-y)+rng.normal(0,1)))
raw=pd.DataFrame(rows)
for year in range(2017,2026):
    cut=raw[raw.Season.eq(year)]
    folds[str(year)]={'models':{c:v43._metric(cut.Actual_Margin,cut[c]) for c in (v43.BASE,)+v43.CANDIDATES}}
report={'status':'EXPLORATORY_RESEARCH__SOURCE_VINTAGE_HOLD','source_vintage_certified':False,
    'experiment_spec_sha256':'hash','folds':folds,
    'parity_to_published_v4':{str(y):'EXACT_REPRODUCTION' for y in range(2017,2026)}}
res=v43.validate(report,raw,boot=250)
assert res['status']=='EXPLORATORY_FOCUSED_VALIDATION__SOURCE_VINTAGE_HOLD'
assert len(res['parity_to_published_v42'])==9
assert len(res['paired_2024_2025'])==6
assert len(res['exploratory_holm_three_comparisons'])==3
assert res['prod_v1_mutation'] is False and res['promotion_authority']==0
bad=raw.copy();bad.loc[bad.Season.eq(2024),'A_EFFICIENCY']+=4
try: v43.validate(report,bad,boot=15)
except RuntimeError as e:assert 'PARITY' in str(e)
else:raise AssertionError('corrupt ledger should fail')
bad=raw.copy();bad.loc[0,'Season']=2026
try:v43.validate(report,bad,boot=15)
except RuntimeError as e:assert '2026' in str(e).lower() or 'SEAL' in str(e)
else:raise AssertionError('2026 should fail')

p=[]
for i in range(35):
    kickoff=pd.Timestamp('2026-09-01T18:00:00Z')+pd.Timedelta(days=i)
    win=(i%3!=0);pts=1 if win else -1
    p.append(dict(prediction_event_id=f'event-{i}',game_start=kickoff,
        prediction_timestamp=kickoff-pd.Timedelta(hours=4),quote_timestamp=kickoff-pd.Timedelta(hours=5),
        physical_game_id=f'game-{i}',market='spreads',action='BET',odds=-110.,model_probability=.56,
        quote_executable=True,is_prospective=True,lock_type='FIRST',result='WIN' if win else 'LOSS',
        profit_per_unit=(100/110) if win else -1.,model_version='v1',artifact_sha256='x'*64))
pdframe=pd.DataFrame(p)
r=prod.audit(pdframe)
assert r['status']=='PROSPECTIVE_LEDGER_ANALYSIS_ONLY' and r['graded_executable_bets']==35
assert len(r['by_market'])==1 and r['portfolio']['roi_game_date_bootstrap_ci95']
assert r['historical_replay_validated'] is False and r['production_modified'] is False
mut=pdframe.copy();mut.loc[0,'prediction_timestamp']=mut.loc[0,'game_start']+pd.Timedelta(minutes=1)
assert prod.audit(mut)['graded_executable_bets']==34
print('V43_SELFTEST_PASS',len(raw),'nine folds','three candidates','six pairwise comparisons','production locks',35)
