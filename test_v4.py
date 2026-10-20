"""Deterministic synthetic research, same-day leakage, market exclusion, 2026 seal."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ncaaf_full_history_challenger_v3_1'))
import numpy as np
import pandas as pd
import ncaaf_margin_tournament_v4 as v

rng=np.random.default_rng(101)
rows=[]
nt=84
pow=rng.normal(0,10,nt);off=rng.normal(0,6,nt)
for year in range(2013,2026):
    power=pow+.65*rng.normal(0,3,nt)
    for j in range(355):
        home=(j*37+year*7)%nt;away=(j*19+year*3+17)%nt
        if home==away:away=(away+1)%nt
        # 18 different kickoff dates; no game overlaps matter for this test.
        d=1+j%18
        m=2.5+power[home]-power[away]+rng.normal(0,12)
        total=54+off[home]+off[away]+rng.normal(0,14)
        rows.append({'Season':year,'Game_Date':f'{year}-09-{d:02d}',
            'Source_Game_ID':f'{year}-{j}', 'Team_Norm':f'Team{home}',
            'Opponent_Norm':f'Team{away}', 'Is_Home':1.,
            'Actual_Margin':m,'Actual_Total':total,
            'CFBD_Season_Net_PPA_Prior':power[home]/20+rng.normal(0,.1),
            'CFBD_Opp_Season_Net_PPA_Prior':power[away]/20+rng.normal(0,.1),
            'CFBD_Matchup_Net_PPA_Adv':(power[home]-power[away])/20+rng.normal(0,.1),
            'CFBD_Season_Drive_Quality_Prior':power[home]/15+rng.normal(0,.1),
            'Market_Open_Margin':m+4,'Consensus_Open_Spread':-m-4,
            'CFBD_Game_Final_Score':m+100})
f=pd.DataFrame(rows)
# Confirm locally vendored incumbent is numerically identical to the existing
# V3 scoring function, not a silently altered benchmark.
import ncaaf_full_history_challenger_v3 as reference_v3
tr=f.loc[f.Season<=2021];va=f.loc[f.Season==2022]
expected=reference_v3._program_effects(tr,va)
actual=v._v3_program_strength_benchmark(tr,va)
assert np.allclose(expected,actual,atol=1e-9,rtol=1e-9,equal_nan=True)
a=v.dynamic_features(f,gain=.13,retain=.65)
# Same-day outcome must not influence same-day prediction.
f2=f.copy();f2.loc[0,'Actual_Margin']+=75.;f2.loc[0,'Actual_Total']+=120.
b=v.dynamic_features(f2,gain=.13,retain=.65)
assert np.allclose(a.loc[[x for x in range(len(f)) if f.at[x,'Game_Date']==f.at[0,'Game_Date']]].to_numpy(float),
                   b.loc[[x for x in range(len(f)) if f.at[x,'Game_Date']==f.at[0,'Game_Date']]].to_numpy(float),equal_nan=True)
assert not v.BAD.search('CFBD_Season_Drive_Quality_Prior')
assert v.BAD.search('CFBD_Game_Final_Score')
assert not any('Market' in x or 'Final' in x for x in v._cfbd_cols(f.loc[f.Season<=2022], f.loc[f.Season<=2022]))
# Keep the end-to-end synthetic test quick; model selection still requires >=5 discovery folds.
v.DISCOVERY=(2017,2018,2019,2020,2021)
report,ledger=v.run_on_frame(f,{'table':v.SOURCE},log_func=lambda s:None)
assert report['status']=='RESEARCH_COMPLETE_NO_PROMOTION'
assert report['selected_pre2024']
assert report['production_mutated'] is False and report['production_authority']==0
assert report['training_dependency_on_opening_line'] is False
assert report['2024_2025_status']=='RETROSPECTIVE_PREVIOUSLY_INSPECTED'
assert len(ledger)>1000
assert not (ledger.Season>=2026).any()
assert report['folds']['2024']['train_end']==2023
assert report['folds']['2025']['train_end']==2024
assert report['drive_evidence']['status']=='DRIVE_PROCESS_NOT_YET_CERTIFIED'
try:
    v.run_on_frame(pd.concat([f,f.iloc[:1].assign(Season=2026)]),{'table':v.SOURCE},log_func=lambda s:None)
except RuntimeError as exc:
    assert '2026' in str(exc)
else:raise AssertionError('2026 seal failed')
print('V4_SELFTEST_PASS','selected',report['selected_pre2024'],'ledger_rows',len(ledger),
      '2024_n',report['folds']['2024']['valid'])
