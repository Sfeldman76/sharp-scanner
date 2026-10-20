"""Local offline V4.2 smoke test; no cloud calls, no production mutation."""
import numpy as np
import pandas as pd
import ncaaf_margin_tournament_v4 as v4
import ncaaf_v42_conditional_features as v42

rng=np.random.default_rng(42)
rows=[]
strength=rng.normal(0,11,90)
source_cols=sorted({col for spec in v42.FAMILIES.values() for terms in spec.values()
                    for sign,col in terms})
for year in range(2013,2026):
    for j in range(350):
        a=(j*31+year*17)%90;b=(j*41+year*3+19)%90
        if a==b:b=(b+1)%90
        marg=2.5+strength[a]-strength[b]+rng.normal(0,13)
        total=57+rng.normal(0,13)
        row=dict(Season=year,Game_Date=f'{year}-09-{1+(j%23):02d}',
            Team_Norm=f'Team{a}',Opponent_Norm=f'Team{b}',Source_Game_ID=f'{year}-{j}',
            Is_Home=1.0,Actual_Margin=marg,Actual_Total=total,
            Market_Open_Margin=marg+45,Consensus_Open_Spread=-marg-45)
        for c in source_cols:
            if c.startswith('CFBD_Opp_'): team=b
            else: team=a
            row[c]=0.08*strength[team]+rng.normal(0,.40) if rng.uniform()>.07 else np.nan
        row['CFBD_Season_Net_PPA_Prior']=strength[a]/15+rng.normal(0,.2)
        row['CFBD_Game_Final_Score']=marg+2000
        rows.append(row)
frame=pd.DataFrame(rows)
assert v42.VERSION.startswith('NCAAF_V42')
assert len(v42.FAMILIES)==5 and len(v42.COMBOS)==6
p,meta=v42.feature_frame(frame.head(25))
p2,_=v42.feature_frame(frame.head(25))
assert p.equals(p2)  # no global feature-list mutation
assert not any('Market' in str(x) or 'Final_Score' in str(x) for v in meta.values() for x in v.get('source_columns',[]))
# Adversarial changed market / outcomes must NEVER be used in constructed feature values.
changed=frame.head(25).copy()
changed['Market_Open_Margin']=-999
changed['Actual_Margin']+=3000
changed['CFBD_Game_Final_Score']=-1e9
q,_=v42.feature_frame(changed)
assert p.equals(q)

report,ledger=v42.run_on_frame(frame,{'table':v42.SOURCE},years=(2024,),log_func=lambda _:None)
assert report['status']=='EXPLORATORY_RESEARCH__SOURCE_VINTAGE_HOLD'
assert report['market_data_used_for_training'] is False
assert report['source_vintage_certified'] is False
assert len(report['experiment_spec_sha256'])==64
assert report['selected_new_champion'] is None
assert report['production_mutated'] is False
assert report['year_2026_archive_queried'] is False
assert len(ledger)==350 and ledger.Season.eq(2024).all()
fold=report['folds']['2024']
assert fold['valid']==350
assert len(fold['frozen_base_cols'])==12
assert sum(x=='PASS' for x in fold['experiments'].values())>=5,fold['experiments']
# Numerically compare standalone V4 baseline, same exact prior-year fit and features.
dynamic=v4.dynamic_features(frame,gain=.13,retain=.65)
x=pd.concat([frame.reset_index(drop=True),dynamic],axis=1)
tr=x.loc[x.Season<2024];va=x.loc[x.Season==2024]
baseline=v4._matrix_fit_predict(tr,va,v42.STATE,v4.nums(tr,'Actual_Margin'),kind='hgb')
assert np.allclose(baseline,ledger[v42.BASE].to_numpy(),atol=1e-12)
# Hard seal on all 2026 outcomes.
try:
    v42.run_on_frame(pd.concat([frame,frame.iloc[:1].assign(Season=2026)]),
                     {'table':v42.SOURCE},years=(2024,),log_func=lambda _:None)
except RuntimeError as e:
    assert '2026' in str(e)
else:
    raise AssertionError('2026 sealed data was admitted')
# Absent family fields must be refused, never guessed or populated from current-game outputs.
pruned=frame.drop(columns=[c for c in frame if 'Sack_Rate_Prior' in c or 'StuffRate_Prior' in c or 'PowerSuccess_Prior' in c or 'EarlyDown_Success_Prior' in c])
r2,_=v42.run_on_frame(pruned,{'table':v42.SOURCE},years=(2024,),log_func=lambda _:None)
assert r2['folds']['2024']['experiments']['E_TRENCH_DISRUPTION']=='HOLD_COVERAGE_OR_SOURCE'
print('V42_SELFTEST_PASS', 'rows',len(ledger),'families',len(v42.FAMILIES),
      'combinations',len(v42.COMBOS),'eligible',sum(v=='PASS' for v in fold['experiments'].values()),
      'base_rmse',round(fold['models'][v42.BASE]['rmse'],4))
