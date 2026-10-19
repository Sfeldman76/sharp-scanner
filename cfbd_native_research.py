"""Standalone CFBD-native learning experiments. Offline, read-only, no bet authority.

1) Identify which *native source family* adds predictive value, not which legacy
   system a CFBD signal resembles.
2) Estimate history-value through equal-feature and equal-evaluation-year ablations.
3) Train and select sequentially: season y may NEVER influence feature/model
   selection for y; 2024-25 are NOT untouched holdouts in this project.
4) Do not feed market odds or final game measurements into pregame X.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from cfbd_native_builder import FAMILY_FEATURES

# Feature families are additive and can be ablated one at a time. The labels
# are NEVER derived from these source measurements on the same game.
FAMILIES=('score_history','team_box','drives','plays','advanced')


def feature_columns(df:pd.DataFrame,groups:tuple[str,...],
                    *, include_matchups:bool=True) -> list[str]:
    featset=set(v for g in groups for v in FAMILY_FEATURES[g])
    cols=['neutral_site','home_prior_games','away_prior_games',
          'home_season_games','away_season_games']
    for z in sorted(featset):
        for prefix in ('home','away','diff'):
            c=f'{prefix}_{z}'
            if c in df.columns: cols.append(c)
            if prefix in ('home','away') and f'{c}_n' in df.columns:
                cols.append(f'{c}_n')
    if include_matchups and 'advanced' in groups:
        cols += [x for x in ('home_off_vs_opp_def_ppa','away_off_vs_opp_def_ppa') if x in df.columns]
    return list(dict.fromkeys(cols))


def _matrix(train:pd.DataFrame,test:pd.DataFrame,columns:list[str],
            min_non_null:int=20) -> tuple[np.ndarray,np.ndarray,list[str]]:
    """Fit all preprocessing ONLY on training partition. No global median leak."""
    keep=[]
    for c in columns:
        if c not in train: continue
        vals=pd.to_numeric(train[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        if int(vals.notna().sum())>=min_non_null and vals.nunique(dropna=True)>=2:
            keep.append(c)
    if not keep:
        raise ValueError('NO_EFFECTIVE_FEATURES: prior training period lacks usable observations')
    a=train[keep].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    b=test[keep].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    # For *partially* missing columns: use training medians, with an explicit
    # missingness indicator. For fully missing fold columns, exclude from fit.
    med=a.median()
    x=np.concatenate([a.fillna(med).to_numpy(dtype=float),a.isna().to_numpy(dtype=float)],axis=1)
    z=np.concatenate([b.fillna(med).to_numpy(dtype=float),b.isna().to_numpy(dtype=float)],axis=1)
    return x,z,keep


def _predict_fold(train:pd.DataFrame,test:pd.DataFrame,target:str,
                  columns:list[str],model_type:str='ridge') -> tuple[np.ndarray,int]:
    x,z,used=_matrix(train,test,columns,min_non_null=max(10,int(0.025*len(train))))
    yy=train[target].to_numpy(dtype=float)
    scaler=StandardScaler().fit(x)
    xt=scaler.transform(x);zt=scaler.transform(z)
    if model_type=='nonlinear' and target!='home_win':
        model=HistGradientBoostingRegressor(max_iter=110,max_leaf_nodes=10,
                                            learning_rate=0.055,l2_regularization=25,
                                            min_samples_leaf=35,random_state=19)
        model.fit(xt,yy)
        return model.predict(zt),len(used)
    if target=='home_win':
        if len(np.unique(yy))<2:
            raise ValueError('Training fold has only one H2H label')
        model=LogisticRegression(C=0.12,max_iter=250,random_state=19)
        model.fit(xt,yy)
        return model.predict_proba(zt)[:,1],len(used)
    model=Ridge(alpha=90.0)
    model.fit(xt,yy)
    return model.predict(zt),len(used)


def _score(a:np.ndarray,p:np.ndarray,target:str)->dict:
    if not len(a):return {'n':0}
    if target=='home_win':
        p=np.clip(p,0.00001,0.99999)
        return {'n':len(a),'brier':round(float(np.mean((a-p)**2)),6),
                'logloss':round(float(-np.mean(a*np.log(p)+(1-a)*np.log(1-p))),6)}
    return {'n':len(a),'rmse':round(float(np.sqrt(np.mean((a-p)**2))),6),
            'mae':round(float(np.mean(abs(a-p))),6)}


def run_research(features:pd.DataFrame,targets:pd.DataFrame,
                 *, earliest_val_year:int=2016,latest_val_year:int=2025,
                 market:pd.DataFrame|None=None,
                 include_nonlinear:bool=False)->tuple[pd.DataFrame,dict]:
    """Year-forward CFBD-only OOF. Genuine opener comparison is optional and separate.

    Model families and windows are *predeclared*. This is NOT an automatic
    search/promoter and doesn't claim discovery on already revisited 2024-25.
    """
    if features.game_id.duplicated().any() or targets.game_id.duplicated().any():
        raise ValueError('Duplicate physical game identity')
    if any(c in features.columns for c in ('home_margin','game_total','home_win',
                                            'home_points','away_points','closing_spread','open_spread')):
        raise ValueError('Leakage contract: outcome/market field in feature table')
    d=features.merge(targets,on=['game_id','season'],how='inner',validate='one_to_one')
    if len(d)!=len(features): raise ValueError('Missing labels for one or more games')
    # Fixed, separated domains: family baseline and each new source layer.
    variants={
      'SCORES_ONLY':('score_history',),
      'PLUS_TEAM_BOX':('score_history','team_box'),
      'PLUS_DRIVES':('score_history','team_box','drives'),
      'PLUS_PLAYS':('score_history','team_box','drives','plays'),
      'ALL_CFBD_NATIVE':FAMILIES,
    }
    folds=[];ledger=[]; seasons=sorted(d.season.dropna().astype(int).unique())
    for yr in seasons:
        if yr<earliest_val_year or yr>latest_val_year:continue
        test=d.loc[d.season==yr].copy()
        for window in (2,5,None):
            prev=d.loc[(d.season<yr)&((d.season>=yr-window) if window is not None else True)]
            if prev.season.nunique()<2 or len(prev)<100: continue
            for variant,families in variants.items():
                cols=feature_columns(d,families)
                for target in ('home_margin','game_total','home_win'):
                    tr=prev.loc[np.isfinite(prev[target])].copy()
                    te=test.loc[np.isfinite(test[target])].copy()
                    if len(tr)<80 or len(te)<25: continue
                    algorithms=['ridge']
                    if include_nonlinear and variant=='ALL_CFBD_NATIVE' and target!='home_win':
                        algorithms.append('nonlinear')
                    for algorithm in algorithms:
                        try:
                            with threadpool_limits(limits=1):
                                pred,active=_predict_fold(tr,te,target,cols,model_type=algorithm)
                        except ValueError as exc:
                            folds.append({'year':yr,'window':window or 'FULL','variant':variant,
                                          'target':target,'algorithm':algorithm,'status':str(exc)})
                            continue
                        metrics=_score(te[target].to_numpy(float),pred,target)
                        folds.append({'year':yr,'window':window or 'FULL','variant':variant,
                                      'target':target,'algorithm':algorithm,
                                      'used_feature_columns':active,'status':'PASS',**metrics})
                        for (_,r), p in zip(te.iterrows(),pred):
                            ledger.append({'game_id':r.game_id,'season':yr,'window':window or 'FULL',
                                           'variant':variant,'target':target,'actual':float(r[target]),
                                           'prediction':float(p),'model':algorithm,
                                           'production_authority':0})
    report={'status':'PASS' if any(r.get('status')=='PASS' for r in folds) else 'INSUFFICIENT_DATA',
            'forecast_type':'FOOTBALL_ONLY_INDEPENDENT',
            'model_selection':'PREDECLARED_CHALLENGERS_NO_AUTO_PROMOTION',
            'seasons':seasons,'folds':folds,'production_authority':0,
            'data_quality_note':'2024 and 2025 are repeatedly analyzed retrospective seasons, not sealed holds',
            'source_metric_vintage':'NOT_CERTIFIED_RETROSPECTIVE_CFBD_DERIVED_METRICS_REQUIRE_SEPARATE_AUDIT',
            'source_lineage':'CFBD_NATIVE_RAW_API_FAMILIES'}
    return pd.DataFrame(ledger),report


def market_diagnostic(oof:pd.DataFrame,quotes:pd.DataFrame)->dict:
    """Exact game/season opening-line comparison; requires a certified vintage.
    Input columns: game_id, season, home_open_margin, quote_timestamp_utc,
    kickoff_timestamp_utc, opener_vintage_verified. Never infer close=open.
    """
    required={'game_id','season','home_open_margin','quote_timestamp_utc',
              'kickoff_timestamp_utc','opener_vintage_verified'}
    if not required.issubset(quotes.columns):
        return {'status':'UNCERTIFIED_PRICE_PROVENANCE','missing':sorted(required-set(quotes))}
    q=quotes.copy()
    if q.duplicated(['game_id','season']).any():
        return {'status':'UNRESOLVED_PROVIDER_GAME_QUOTES'}
    qt=pd.to_datetime(q.quote_timestamp_utc,errors='coerce',utc=True)
    ko=pd.to_datetime(q.kickoff_timestamp_utc,errors='coerce',utc=True)
    q=q.loc[qt.notna() & ko.notna() & (qt<ko) & q.opener_vintage_verified.eq(True)].copy()
    q['home_open_margin']=pd.to_numeric(q.home_open_margin,errors='coerce')
    kinds=oof['model'].eq('ridge') if 'model' in oof.columns else pd.Series(True,index=oof.index)
    paired=oof.loc[oof.target.eq('home_margin') & oof.variant.eq('ALL_CFBD_NATIVE') & kinds].merge(
        q[['game_id','season','home_open_margin']],on=['game_id','season'],validate='many_to_one')
    paired=paired.loc[np.isfinite(paired.home_open_margin)]
    results=[]
    for (yr,window),g in paired.groupby(['season','window']):
        results.append({'season':int(yr),'window':str(window),'n':len(g),
                        'native_rmse':_score(g.actual.to_numpy(),g.prediction.to_numpy(),'home_margin')['rmse'],
                        'market_rmse':_score(g.actual.to_numpy(),g.home_open_margin.to_numpy(),'home_margin')['rmse']})
    return {'status':'PASS' if results else 'NO_CERTIFIED_PAIRED_OPENS',
            'results':results,'production_authority':0}


def save_research(folder:str|Path,output_folder:str|Path,*,min_year:int=2016,
                  market_file:str|Path|None=None,latest_year:int=2025,
                 include_nonlinear:bool=False)->dict:
    folder=Path(folder);out=Path(output_folder);out.mkdir(parents=True,exist_ok=True)
    features=pd.read_csv(folder/'cfbd_native_pregame_features.csv')
    targets=pd.read_csv(folder/'cfbd_native_outcomes_RESEARCH_ONLY.csv')
    ledger,report=run_research(features,targets,earliest_val_year=min_year,
                              latest_val_year=latest_year,include_nonlinear=include_nonlinear)
    ledger.to_csv(out/'cfbd_native_oof_ledger_RESEARCH_ONLY.csv',index=False)
    if market_file:
        report['market_comparison']=market_diagnostic(ledger,pd.read_csv(market_file))
    (out/'cfbd_native_research_report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    return {'status':report['status'],'folds':len(report['folds']),
            'oof_rows':len(ledger),'out':str(out),'production_authority':0}
