"""NCAAF V4 independent football-margin research tournament.

Research ONLY. Fit on seasons 2013-2023 via expanding-origin folds,
retrospective 2024-25 (previously inspected), NO opening-line predictors,
NO 2026 labels, NO mutation of frozen production.

A sequential partial-pooling state estimator learns evolving offensive and
opponent-adjusted defensive power from completed prior games. Two genuinely
competing supervised representations:
 - dynamic offense/defense and power -> margin;
 - forecast home/away scores separately -> margin.
 Plus strictly time-fold HGB nonlinear and optional CatBoost; and the exact
V3 PROGRAM_STRENGTH_RIDGE as the incumbent.

This is not a fully Bayesian state-space estimator. It is a transparent
Kalman-like exponential-gain state filter, which must earn improvements.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from typing import Any
import numpy as np
import pandas as pd

VERSION="NCAAF_V4_FOOTBALL_TOURNAMENT_20261020_V1"
SOURCE="sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1"
DISCOVERY=tuple(range(2017,2024))
RETROSPECTIVE=(2024,2025)
MIN_TRAIN=1300
MIN_TEST=250
SEED=20261020

# All CFBD supplemental columns must be reconstructed PRIOR-only in the attached view.
# Reject uncertainty rather than admitting contemporaneous statistics.
BAD=re.compile(r"(?:market|odds|opening|closing|spread|moneyline|line_|clv|bet_|betting|consensus|_ats_|actual|score|result|final|prediction|"
               r"injur|weather|portal|roster|recruit|returning|talent|current_game|postgame|pt_)",re.I)
DRIVE=re.compile(r"(?:drive|possession|scoring_opportunity|field_position|finishing)",re.I)

def nums(g,c):
    return pd.to_numeric(g[c],errors='coerce').to_numpy(dtype=float) if c in g else np.full(len(g),np.nan)

def metric(y,p):
    a=np.asarray(y,float);b=np.asarray(p,float);keep=np.isfinite(a)&np.isfinite(b)
    if not keep.any():return {'n':0,'rmse':None,'mae':None}
    d=a[keep]-b[keep]
    return {'n':int(keep.sum()),'rmse':float(np.sqrt(np.mean(d*d))),'mae':float(np.mean(abs(d)))}

def paired(y,baseline,pred):
    a=np.asarray(y,float);b=np.asarray(baseline,float);p=np.asarray(pred,float)
    good=np.isfinite(a)&np.isfinite(b)&np.isfinite(p)
    if not good.any():return {'n':0,'rmse_gain':None,'mae_gain':None}
    x=metric(a[good],b[good]);z=metric(a[good],p[good])
    return {'n':int(good.sum()),'rmse_gain':x['rmse']-z['rmse'],'mae_gain':x['mae']-z['mae']}

def key(z):
    if z is None or pd.isna(z):return ''
    return re.sub('[^a-z0-9]','',str(z).lower())

def _id(g, cols):
    ret=pd.Series('',index=g.index,dtype='string')
    for col in cols:
        if col in g:
            candidate=g[col].astype('string').fillna('').map(key)
            ret=ret.mask(ret.eq(''),candidate)
    return ret.astype(str).to_numpy()

def input_audit(g, archive_diag):
    if not len(g):raise RuntimeError('V4_EMPTY_INPUT')
    if (archive_diag or {}).get('table')!=SOURCE:raise RuntimeError('V4_ARCHIVE_SOURCE_MISMATCH')
    required=('Season','Game_Date','Actual_Margin','Actual_Total','Team_Norm','Opponent_Norm')
    missing=[x for x in required if x not in g]
    if missing:raise RuntimeError('V4_MISSING:'+','.join(missing))
    seasons=nums(g,'Season')
    if (np.isfinite(seasons)&(seasons>=2026)).any():raise RuntimeError('V4_2026_SEAL_BROKEN')
    if (np.isfinite(seasons)&(seasons<2013)).any():raise RuntimeError('V4_INVALID_EARLY_SEASON')
    if not (seasons<=2021).sum()>=3000:raise RuntimeError('V4_MISSING_EARLY_HISTORY')
    if 'CFBD_Season_Net_PPA_Prior' not in g:
        raise RuntimeError('V4_MISSING_ARCHIVE_ADVANCED_PREGAME_COLUMN')
    early=(seasons>=2013)&(seasons<=2021)
    advanced=np.isfinite(nums(g,'CFBD_Season_Net_PPA_Prior'))
    early_advanced=float(advanced[early].mean()) if early.any() else 0.
    if early_advanced<.40:
        raise RuntimeError(f'V4_EARLY_ADVANCED_JOIN_TOO_LOW:{early_advanced:.3f}')
    if 'Source_Game_ID' in g:
        ids=g['Source_Game_ID'].astype('string').fillna('').str.strip()
        ok=~ids.str.lower().isin(['','none','nan','<na>'])
        physical=(g.loc[ok,'Season'].astype(str)+'|'+ids[ok]).to_numpy()
        if len(physical)!=len(set(physical)):raise RuntimeError('V4_DUPLICATE_GAME_ORIENTATION')
    d=pd.to_datetime(g['Game_Date'],errors='coerce',utc=True)
    valid_date=float(d.notna().mean())
    if valid_date<.90:raise RuntimeError(f'V4_TOO_FEW_KICKOFF_DATES:{valid_date:.3f}')
    a=_id(g,('Team_Norm',));b=_id(g,('Opponent_Norm',))
    identity=float(np.mean([(bool(x) and bool(y) and x!=y) for x,y in zip(a,b)]))
    if identity<.90:raise RuntimeError(f'V4_IDENTITY_COVERAGE:{identity:.3f}')
    return {'rows':len(g),'seasons':[int(x) for x in sorted(set(seasons[np.isfinite(seasons)]))],
            'date_coverage':valid_date,'identity_coverage':identity,
            'early_advanced_join_rate':early_advanced,
            'archive_source':SOURCE,'2026_labels_accessed':False,
            'pregame_point_in_time_certification':'NOT_COMPLETE_SOURCE_VINTAGE_AUDIT_REQUIRED',
            'same_day_updates':'AFTER_ALL_SAME_DAY_PREDS',
            'production_authority':0}

# Each team has independent offensive and defensive rating (centered around
# league PPG) plus power; those are estimated from *past games only*.
def dynamic_features(g, *,gain=.13,retain=.65):
    g=g.reset_index(drop=True)
    years=nums(g,'Season'); margin=nums(g,'Actual_Margin'); total=nums(g,'Actual_Total')
    dates=pd.to_datetime(g['Game_Date'],errors='coerce',utc=True).dt.normalize()
    team=_id(g,('Team_Norm',));opp=_id(g,('Opponent_Norm',))
    site=nums(g,'Is_Home')
    names=('power_diff','off_diff','def_diff','expected_margin','expected_team_points',
           'expected_opp_points','power_uncertainty','prior_games_min','rest_diff',
           'prior_margin_recent_diff','prior_margin_season_diff','site_adv')
    arr={s:np.full(len(g),np.nan,float) for s in names}
    def fresh():return {'power':0.,'off':0.,'def':0.,'n':0.,'last':None,
                        'recent':0.,'season_margin':0.,'prior_season_power':0.}
    states=defaultdict(fresh)
    good=np.flatnonzero(np.isfinite(years)&dates.notna().to_numpy()&
           np.array([bool(x) and bool(y) and x!=y for x,y in zip(team,opp)]))
    ordered=sorted(good,key=lambda i:(dates.iloc[i],int(years[i]),i))
    current_year=None;league_ppg=27.
    pos=0
    while pos<len(ordered):
        current_date=dates.iloc[ordered[pos]]
        day=[]
        while pos<len(ordered) and dates.iloc[ordered[pos]]==current_date:
            day.append(ordered[pos]);pos+=1
        by_year=defaultdict(list)
        for i in day:by_year[int(years[i])].append(i)
        for season,idxs in sorted(by_year.items()):
            if current_year!=season:
                if current_year is not None:
                    for a in states.values():
                        a['prior_season_power']=a['power']
                        for v in ('power','off','def','recent'):a[v]*=retain
                        a['n']=0;a['season_margin']=0.;a['last']=None
                current_year=season
            # Snapshot all predictions before touching any score from this date.
            pending=[]
            for i in idxs:
                a=states[team[i]];b=states[opp[i]]
                if np.isfinite(site[i]):home=2.5*(2*site[i]-1)
                else:home=0. # missing site -> neutral assumption, explicitly tracked
                power=a['power']-b['power']
                score_a=league_ppg+a['off']-b['def']+home/2
                score_b=league_ppg+b['off']-a['def']-home/2
                vals=dict(power_diff=power,off_diff=a['off']-b['off'],
                    def_diff=b['def']-a['def'],expected_margin=power+home,
                    expected_team_points=score_a,expected_opp_points=score_b,
                    power_uncertainty=1/(1+a['n'])+1/(1+b['n']),
                    prior_games_min=min(a['n'],b['n']),
                    rest_diff=((current_date-a['last']).days-(current_date-b['last']).days)
                        if a['last'] is not None and b['last'] is not None else np.nan,
                    prior_margin_recent_diff=a['recent']-b['recent'],
                    prior_margin_season_diff=(a['season_margin']/a['n'] if a['n'] else 0.)-
                                              (b['season_margin']/b['n'] if b['n'] else 0.),site_adv=home)
                for col,value in vals.items():arr[col][i]=value
                if np.isfinite(margin[i]) and np.isfinite(total[i]):
                    pending.append((i,vals,league_ppg,a['power'],b['power']))
            # Apply previous day's observed data to both opponent-adjusted states.
            observed_ppg=[]
            for i,v,old_league,old_power_a,old_power_b in pending:
                a=states[team[i]];b=states[opp[i]]
                resid=float(np.clip(margin[i]-v['expected_margin'],-45,45))
                k_a=gain/(1+.04*a['n']);k_b=gain/(1+.04*b['n'])
                a['power']+=k_a*resid;b['power']-=k_b*resid
                for_points=(total[i]+margin[i])/2
                opp_points=(total[i]-margin[i])/2
                error_a=float(np.clip(for_points-v['expected_team_points'],-36,36))
                error_b=float(np.clip(opp_points-v['expected_opp_points'],-36,36))
                a['off']+=k_a*error_a/2;b['def']-=k_b*error_a/2
                b['off']+=k_b*error_b/2;a['def']-=k_a*error_b/2
                a['recent']=.65*a['recent']+.35*margin[i]
                b['recent']=.65*b['recent']-.35*margin[i]
                a['season_margin']+=margin[i];b['season_margin']-=margin[i]
                a['n']+=1;b['n']+=1
                a['last']=current_date;b['last']=current_date
                observed_ppg.append(total[i]/2)
            if observed_ppg:league_ppg=.995*league_ppg+.005*float(np.mean(observed_ppg))
    out=pd.DataFrame({'V4_'+k:pd.Series(v,dtype=float) for k,v in arr.items()})
    return out

def _cfbd_cols(frame, train):
    names=[c for c in frame if str(c).startswith('CFBD_') and not BAD.search(c)
          and ('Prior' in str(c) or 'Matchup_' in str(c))]
    # Include drive/possession stats only when they are explicitly reconstructed
    # as prior, not raw current-game drive outcomes.
    order=sorted(names,key=lambda c:(-float(np.isfinite(nums(train,c)).mean()),str(c)))
    cols=[]
    for c in order:
        vals=nums(train,c);valid=vals[np.isfinite(vals)]
        if len(valid)>=max(100,.20*len(train)) and len(np.unique(valid))>=3:cols.append(c)
    return cols

def _matrix_fit_predict(tr,va,cols,target,*,kind='ridge',weights=None):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    if not cols:return np.full(len(va),np.nan)
    x=tr[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    v=va[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    y=np.asarray(target,float);good=np.isfinite(y)
    if good.sum()<MIN_TRAIN:return np.full(len(va),np.nan)
    live=[c for c in cols if x.loc[good,c].notna().sum()>=max(75,.08*good.sum())
          and x.loc[good,c].nunique(dropna=True)>1]
    if not live:return np.full(len(va),np.nan)
    x=x.loc[good,live];v=v[live]
    if kind=='hgb':
        lo=x.quantile(.005);hi=x.quantile(.995)
        x=x.clip(lower=lo,upper=hi,axis=1);v=v.clip(lower=lo,upper=hi,axis=1)
        m=HistGradientBoostingRegressor(max_iter=180,learning_rate=.045,max_leaf_nodes=10,
                min_samples_leaf=60,l2_regularization=25,early_stopping=False,random_state=SEED)
        m.fit(x,y[good]);return np.asarray(m.predict(v),float)
    if kind=='catboost':
        try:from catboost import CatBoostRegressor
        except ImportError:return np.full(len(va),np.nan)
        imp=SimpleImputer(strategy='median');x=imp.fit_transform(x);v=imp.transform(v)
        m=CatBoostRegressor(iterations=260,depth=4,learning_rate=.045,l2_leaf_reg=12,
                loss_function='RMSE',verbose=False,thread_count=2,random_seed=SEED,allow_writing_files=False)
        m.fit(x,y[good]);return np.asarray(m.predict(v),float)
    m=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',RobustScaler()),
                ('ridge',Ridge(alpha=90.,solver='auto'))])
    m.fit(x,y[good]);return np.asarray(m.predict(v),float)

def _v3_program_strength_benchmark(train,valid):
    """Verbatim mathematical V3 benchmark, locally vendored for V4 independence.

    Fixed Ridge(alpha=90) + program contrast + four pregame CFBD fields.
    This must remain identical to the earlier V3 reference implementation.
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import Ridge
    from scipy.sparse import hstack,csr_matrix
    if len(train)<MIN_TRAIN or any(c not in train or c not in valid for c in ('Team_Norm','Opponent_Norm')):
        return np.full(len(valid),np.nan)
    trn=train['Team_Norm'].astype('string').fillna('').str.lower().str.strip()
    opn=train['Opponent_Norm'].astype('string').fillna('').str.lower().str.strip()
    vn=valid['Team_Norm'].astype('string').fillna('').str.lower().str.strip()
    vo=valid['Opponent_Norm'].astype('string').fillna('').str.lower().str.strip()
    if not (trn.ne('').mean()>.85 and opn.ne('').mean()>.85):
        return np.full(len(valid),np.nan)
    def rows(a,b):
        return [({('program:'+str(x)):1.0,('program:'+str(y)):-1.0}
                 if x!=y and x and y else {}) for x,y in zip(a,b)]
    enc=DictVectorizer(sparse=True)
    xtr=enc.fit_transform(rows(trn,opn));xva=enc.transform(rows(vn,vo))
    cols=['CFBD_Season_Net_PPA_Prior','CFBD_Recent3_Net_PPA_Prior',
          'CFBD_Matchup_Net_PPA_Adv','CFBD_Opp_Season_Net_PPA_Prior']
    usable=[c for c in cols if c in train and c in valid and np.isfinite(nums(train,c)).sum()>=max(100,int(len(train)*.12))]
    if usable:
        a=train[usable].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        b=valid[usable].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
        med=a.median().fillna(0);a=a.fillna(med);b=b.fillna(med)
        scale=(a.quantile(.75)-a.quantile(.25)).replace(0,1).fillna(1)
        xtr=hstack([xtr,csr_matrix(((a-med)/scale).to_numpy(dtype=float))],format='csr')
        xva=hstack([xva,csr_matrix(((b-med)/scale).to_numpy(dtype=float))],format='csr')
    yy=nums(train,'Actual_Margin');good=np.isfinite(yy)
    if good.sum()<MIN_TRAIN:return np.full(len(valid),np.nan)
    model=Ridge(alpha=90.,solver='lsqr',fit_intercept=True)
    model.fit(xtr[good],yy[good])
    return np.asarray(model.predict(xva),float)


def _fold(df, year, dyn_variants):
    yy=nums(df,'Season');y=nums(df,'Actual_Margin');tot=nums(df,'Actual_Total')
    tr=(yy<year)&np.isfinite(y)&np.isfinite(tot)
    va=(yy==year)&np.isfinite(y)&np.isfinite(tot)
    if tr.sum()<MIN_TRAIN or va.sum()<MIN_TEST:
        return {'status':'INSUFFICIENT_ROWS','train':int(tr.sum()),'valid':int(va.sum())},None
    train=df.loc[tr];valid=df.loc[va]
    v3_prediction=_v3_program_strength_benchmark(train,valid)
    state_cols=['V4_'+x for x in ('power_diff','off_diff','def_diff','expected_margin','expected_team_points',
        'expected_opp_points','power_uncertainty','prior_games_min','rest_diff','prior_margin_recent_diff',
        'prior_margin_season_diff','site_adv')]
    dyn_results={}
    for label,dyn in dyn_variants.items():
        f=pd.concat([df.reset_index(drop=True),dyn],axis=1)
        xtr=f.loc[tr];xva=f.loc[va]
        # A common structural mapping is learned ONLY on earlier seasons.
        p=_matrix_fit_predict(xtr,xva,state_cols,nums(xtr,'Actual_Margin'))
        # A distinct scoring mechanism: fit separate offense/defense scoring
        # outputs and subtract. This is score-process, not certified drive model.
        team_pts=(nums(xtr,'Actual_Total')+nums(xtr,'Actual_Margin'))/2
        opp_pts=(nums(xtr,'Actual_Total')-nums(xtr,'Actual_Margin'))/2
        a=_matrix_fit_predict(xtr,xva,state_cols,team_pts)
        b=_matrix_fit_predict(xtr,xva,state_cols,opp_pts)
        dyn_results[label]=(p,a-b)
    # Select only two fixed state filters based on discovery folds, with
    # no hyperparameter outcome lookahead from this fold.
    base=dyn_variants['moderate']
    full=pd.concat([df.reset_index(drop=True),base],axis=1)
    xtr=full.loc[tr];xva=full.loc[va]
    available=_cfbd_cols(xtr,xtr)
    chosen=available[:65]
    nonlinear_cols=state_cols+chosen
    state_hgb=_matrix_fit_predict(xtr,xva,state_cols,nums(xtr,'Actual_Margin'),kind='hgb')
    hgb=_matrix_fit_predict(xtr,xva,nonlinear_cols,nums(xtr,'Actual_Margin'),kind='hgb')
    cb=_matrix_fit_predict(xtr,xva,nonlinear_cols,nums(xtr,'Actual_Margin'),kind='catboost')
    pred={'V3_PROGRAM_STRENGTH_RIDGE':v3_prediction,'DYNAMIC_MODERATE':dyn_results['moderate'][0],
          'DYNAMIC_FAST':dyn_results['fast'][0],
          'SCORE_PROCESS_MODERATE':dyn_results['moderate'][1],
          'SCORE_PROCESS_FAST':dyn_results['fast'][1],
          'NONLINEAR_STATE_HGB':state_hgb,'NONLINEAR_HGB':hgb,'CATBOOST_OPTIONAL':cb}
    ys=nums(valid,'Actual_Margin');out={'status':'PASS','train':int(tr.sum()),'valid':int(va.sum()),
        'train_end':year-1,'cfbd_safe_candidates':len(available),'cfbd_nonlinear_features':chosen,
        'models':{m:metric(ys,p) for m,p in pred.items()},
        'paired_to_v3':{m:paired(ys,v3_prediction,p) for m,p in pred.items()}}
    open_line=nums(valid,'Market_Open_Margin');raw=nums(valid,'Consensus_Open_Spread')
    ok=np.isfinite(open_line)&np.isfinite(raw)&np.isclose(open_line,-raw,atol=1e-5)&(yy[va]>=2021)
    out['opening_market_diagnostic']={'n':int(ok.sum()),'status':'NOT_A_MODEL_INPUT',
        'opening':metric(ys[ok],open_line[ok]),
        'v3_on_same_games':metric(ys[ok],v3_prediction[ok]),
        'all_models_same_games':{k:metric(ys[ok],p[ok]) for k,p in pred.items()}}
    ledger=pd.DataFrame({'Season':int(year),'Game_Date':valid['Game_Date'].astype(str).to_numpy(),
        'Game_ID':valid['Source_Game_ID'].astype(str).to_numpy() if 'Source_Game_ID' in valid else np.arange(len(valid)).astype(str),
        'Team':valid['Team_Norm'].astype(str).to_numpy(),
        'Opponent':valid['Opponent_Norm'].astype(str).to_numpy(),
        'Actual_Margin':ys})
    for k,p in pred.items():ledger[k]=p
    ledger['Verified_Open_Margin_Diagnostic']=np.where(ok,open_line,np.nan)
    return out,ledger

def _paired_cluster_boot(ledger,model,anchor,*,boot=350):
    data=ledger.loc[ledger['Season'].isin(RETROSPECTIVE)].copy()
    y=nums(data,'Actual_Margin');p=nums(data,model);b=nums(data,anchor)
    ok=np.isfinite(y)&np.isfinite(p)&np.isfinite(b)
    if ok.sum()<200:return {'status':'INSUFFICIENT_MATCHED','n':int(ok.sum())}
    data=data.loc[ok].copy();y=y[ok];p=p[ok];b=b[ok]
    clusters=(data['Season'].astype(str)+'|'+data['Game_Date'].astype(str)).to_numpy()
    groups=[np.flatnonzero(clusters==c) for c in np.unique(clusters)]
    rng=np.random.default_rng(SEED);diff=[]
    err0=(y-b)**2;err1=(y-p)**2
    for _ in range(boot):
        chosen=rng.integers(0,len(groups),size=len(groups))
        ids=np.concatenate([groups[i] for i in chosen])
        diff.append(np.sqrt(np.mean(err0[ids]))-np.sqrt(np.mean(err1[ids])))
    return {'status':'EXPLORATORY_RETROSPECTIVE','n':int(ok.sum()),
            'paired_rmse_gain':float(np.sqrt(np.mean(err0))-np.sqrt(np.mean(err1))),
            'game_date_cluster_bootstrap_ci95':[float(v) for v in np.quantile(diff,[.025,.975])],
            'clusters':len(groups),'not_a_fresh_holdout':True}

def run_on_frame(frame,archive_diag,*,log_func=print,min_training=None,min_evaluation=None):
    global MIN_TRAIN,MIN_TEST
    # Tests may override, but never modify the caller's configuration.
    old=(MIN_TRAIN,MIN_TEST)
    if min_training is not None:MIN_TRAIN=int(min_training)
    if min_evaluation is not None:MIN_TEST=int(min_evaluation)
    try:return _run(frame,archive_diag,log_func)
    finally:MIN_TRAIN,MIN_TEST=old

def _run(frame,archive_diag,log_func):
    df=frame.reset_index(drop=True).copy()
    audit=input_audit(df,archive_diag)
    log_func(f'[NCAAF-V4-INPUT] rows={len(df)} source={SOURCE} season_min={min(audit["seasons"])} season_max={max(audit["seasons"])}')
    # Only two predetermined state evolutions. Speed/tuning selected ONLY on
    # early chronological discovery folds (2017-2023).
    dynamic={'moderate':dynamic_features(df,gain=.13,retain=.65),
             'fast':dynamic_features(df,gain=.23,retain=.42)}
    years=(*DISCOVERY,*RETROSPECTIVE)
    folds={};ledgers=[]
    for year in years:
        fold,ledger=_fold(df,year,dynamic)
        folds[str(year)]=fold
        if ledger is not None:ledgers.append(ledger)
        log_func(f'[NCAAF-V4-FOLD] year={year} status={fold["status"]} '+
                 str({k:round(v['rmse'],3) if v['rmse'] is not None else None for k,v in fold.get('models',{}).items()}))
    all_ledger=pd.concat(ledgers,ignore_index=True) if ledgers else pd.DataFrame()
    names=('V3_PROGRAM_STRENGTH_RIDGE','DYNAMIC_MODERATE','DYNAMIC_FAST',
        'SCORE_PROCESS_MODERATE','SCORE_PROCESS_FAST','NONLINEAR_STATE_HGB','NONLINEAR_HGB','CATBOOST_OPTIONAL')
    ranking=[]
    for name in names:
        discovery=[folds[str(y)]['models'][name] for y in DISCOVERY if folds[str(y)]['status']=='PASS'
                  and folds[str(y)]['models'][name]['rmse'] is not None
                  and folds[str(y)]['models'][name]['n']>=MIN_TEST]
        if len(discovery)<5:
            ranking.append({'model':name,'status':'INSUFFICIENT_DISCOVERY','folds':len(discovery)})
            continue
        n=sum(x['n'] for x in discovery)
        rmse=float(np.sqrt(sum(x['n']*x['rmse']**2 for x in discovery)/n))
        ranking.append({'model':name,'status':'ELIGIBLE','discovery_folds':len(discovery),
                        'discovery_n':n,'pooled_rmse':rmse})
    eligible=sorted([x for x in ranking if x['status']=='ELIGIBLE'],key=lambda x:(x['pooled_rmse'],x['model']))
    selected=eligible[0]['model'] if eligible else None
    comparisons={}
    retrospective_pooled={}
    if not all_ledger.empty:
        subset=all_ledger.loc[all_ledger.Season.isin(RETROSPECTIVE)]
        for n in names:retrospective_pooled[n]=metric(nums(subset,'Actual_Margin'),nums(subset,n))
    if not all_ledger.empty:
        for name in names:
            if name=='V3_PROGRAM_STRENGTH_RIDGE':continue
            comparisons[name]=_paired_cluster_boot(all_ledger,name,'V3_PROGRAM_STRENGTH_RIDGE')
    # Guard: NO DRIVE conclusion unless genuine, timestamp-audited drive-level
    # independent data exists. Score-process alone is not a drive model.
    drive_cols=[c for c in df if c.startswith('CFBD_') and DRIVE.search(c) and not BAD.search(c) and c.endswith('_Prior')]
    report={'version':VERSION,'status':'RESEARCH_COMPLETE_NO_PROMOTION' if selected else 'INSUFFICIENT_DISCOVERY',
      'generated_utc':datetime.now(timezone.utc).isoformat(),'source':SOURCE,'audit':audit,
      'folds':folds,'discovery_rank':ranking,'selected_pre2024':selected,
      'retrospective_pooled_all_eligible_games':retrospective_pooled,
      'advanced_features_ablation':_paired_cluster_boot(all_ledger,'NONLINEAR_HGB','NONLINEAR_STATE_HGB') if not all_ledger.empty else {'status':'NO_FOLDS'},
      'retrospective_paired_bootstrap':comparisons,
      'drive_evidence':{'status':'DRIVE_PROCESS_NOT_YET_CERTIFIED',
        'candidate_prior_derived_drive_feature_count':len(drive_cols),
        'candidate_examples':sorted(drive_cols)[:15],
        'note':'Scoring-process model predicts two team scores; it is NOT asserted to be possession/drive generative. Genuine drive model requires timestamp/possession provenance.'},
      'optional_catboost_dependency_present':bool(__import__('importlib').util.find_spec('catboost')),
      'head_to_head':'NOT_FITTED_MARGIN_ONLY_FIRST_GATE','totals':'SCORING_HEADS_SUPPORT_MARGIN_ONLY',
      'training_dependency_on_opening_line':False,'selection_uses_2024_2025':False,
      '2024_2025_status':'RETROSPECTIVE_PREVIOUSLY_INSPECTED',
      'current_2026_outcomes_queried':False,'automatic_promotion':False,
      'bet_authority_vote':False,'production_mutated':False,'production_authority':0,
      'limits':['time-vintage of archived pregame columns still requires source-level certification',
                'same dashboard historical-cache preparation as V3 is currently used',
                'no true drive-level generative model until raw drive provenance audited',
                '2024-2025 are not pristine holdouts']}
    log_func(f'[NCAAF-V4-DONE] selected={selected} 2026=SEALED production=UNCHANGED status={report["status"]}')
    return report,all_ledger

def run_cloud_research(*,dashboard_module,storage_client,bucket_name='sharp-models',log_func=print):
    if str(os.getenv('NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED','0')).lower() not in ('1','true','yes','on'):
        raise RuntimeError('V4_REQUIRES_ARCHIVE_OPT_IN')
    import ncaaf_core_challenger_v2 as core
    cache=getattr(dashboard_module,'_V1357_SPREAD_RESEARCH_CACHE',{}) or {}
    games=cache.get('games')
    if games is None or games.empty:raise RuntimeError('V4_HISTORICAL_GAME_CACHE_MISSING')
    # Deliberately remove 2026 BEFORE retrieving supplementary archive data.
    history=games.loc[nums(games,'Season')<=2025].copy().reset_index(drop=True)
    if 'Historical_Football_Source' in history:
        # V5.4 source firewall: direct canonical CFBD source, without a second
        # dashboard/raw/legacy-game join that could overwrite historical facts.
        import ncaaf_v54_cfbd_truth as truth
        truth.verify_cfbd_frame(history, require_unique_home=True)
        diag={'table':SOURCE,'status':'CFBD_AUTHORITATIVE_DIRECT','cfbd_source':truth.CFBD_FEATURE_VIEW,
              'legacy_bridge_invoked':False,'source_vintage_certified':False}
        enriched=history
        log_func('[NCAAF-V54-CFBD-CORE] direct=TRUE legacy_attach=DISABLED football_source=CFBD_ONLY')
    else:
        enriched,diag=core._attach_cfbd_core_context(dashboard_module,history,log_func=log_func)
    report,ledger=run_on_frame(enriched,diag,log_func=log_func)
    if report['status']!='RESEARCH_COMPLETE_NO_PROMOTION':raise RuntimeError('V4_DISCOVERY_INCOMPLETE')
    base='research/ncaaf/margin_tournament_v4'
    payload=json.dumps(report,sort_keys=True,indent=2,allow_nan=False,default=str)
    # Emit complete per-game retrospective and discovery predictions, not only
    # scores. Dedicated research namespace, never production artifacts.
    bucket=storage_client.bucket(bucket_name)
    bucket.blob(f'{base}/current_report.json').upload_from_string(payload,content_type='application/json')
    bucket.blob(f'{base}/game_level_predictions.csv').upload_from_string(ledger.to_csv(index=False),content_type='text/csv')
    report['report_uri']=f'gs://{bucket_name}/{base}/current_report.json'
    report['predictions_uri']=f'gs://{bucket_name}/{base}/game_level_predictions.csv'
    return report
