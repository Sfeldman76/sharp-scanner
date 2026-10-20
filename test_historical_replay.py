import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor

P=Path(__file__).with_name('ncaaf_prod_v1_historical_replay.py')
spec=importlib.util.spec_from_file_location('hist',P); hist=importlib.util.module_from_spec(spec);spec.loader.exec_module(hist)

def new_models():
    return (Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler()),('ridge',Ridge(alpha=20))]),
            Pipeline([('imputer',SimpleImputer(strategy='median')),('hgb',HistGradientBoostingRegressor(loss='squared_error',learning_rate=.04,max_iter=140,max_leaf_nodes=15,max_depth=3,min_samples_leaf=25,l2_regularization=2.,random_state=42))]))

def blend(m,x,w=.75):return w*m[0].predict(x)+(1-w)*m[1].predict(x)

def h2h():return Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('lr',LogisticRegression(C=.2,max_iter=1500))])

def fixture():
    r=np.random.default_rng(100)
    rows=[]
    for year in range(2013,2027):
        for i in range(120):
            strength=r.normal();rush=r.normal();f=r.normal();x=r.normal();sp=r.uniform(-20,20);total=r.uniform(40,68)
            mr=.6*strength-.35*rush+r.normal(0,10)
            tr=.35*f+r.normal(0,11)
            row={'Season':year,'Game_Date':f'{year}-09-{1+i%28:02d}', 'Source_Game_ID':f'{year}-{i}',
                 'Is_Home':1,'Context_Intercept':0.0,'Diff_RawRecent3_Off_YPP':strength,
                 'B_RawSeason_GameAdj_Def_Rush_YPA':rush,'A_RawRecent3_Def_Rush_YPA_Allowed':f,
                 'Consensus_Open_Spread':sp,'Consensus_Open_Total':total,'Market_Open_Margin':-sp,
                 'Market_Open_Total':total,'Actual_Margin':-sp+mr,'Actual_Total':total+tr}
            row.update({c:x+r.normal(0,1) for c in hist.H2H_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)

def run(d):return hist.run_walkforward(d,new_models=new_models,blend_predict=blend,new_h2h_model=h2h,min_train_games=180,min_test_games=80)

def test():
    d=fixture();r,p=run(d)
    assert r['years_evaluated'][0]==2015
    assert r['years_evaluated'][-1]==2025
    assert 2026 not in r['years_evaluated']
    assert not p.Season.eq(2026).any()
    assert r['production_bet_authority']==0
    assert r['bet_level_roi_supported'] is False
    assert all(t['targets']['spread']['market_rmse']['n']==120 for t in r['folds'])
    assert any('h2h_model_only' in t['targets'] for t in r['folds'])
    tamper=d.copy();tamper.loc[tamper.Season.eq(2025),'Actual_Margin']+=99.
    r2,p2=run(tamper)
    # 2025 outcomes must not influence predictions made for the 2025 games,
    # and cannot influence previous seasons either.
    a=p[p.Market.eq('spreads')][['Season','Source_Game_ID','model_adjustment']]
    b=p2[p2.Market.eq('spreads')][['Season','Source_Game_ID','model_adjustment']]
    pd.testing.assert_frame_equal(a,b)
    missing=d.copy();missing.loc[(missing.Season.eq(2024)) & (missing.Source_Game_ID.str.endswith('-1')),'Consensus_Open_Spread']=np.nan
    r3,_=run(missing)
    assert next(f for f in r3['folds'] if f['season']==2024)['targets']['spread']['n_test']==119
    print('PASS: 2013-2025 chronological folds; 2026 blocked; no future-label leak; missing opener blocked; no false bets')
if __name__=='__main__':test()
