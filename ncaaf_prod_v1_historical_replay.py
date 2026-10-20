"""NCAAF Production V1 fixed-backbone historical walk-forward (RESEARCH ONLY).

Uses exact frozen Spread and Totals feature sets and fitting implementations;
train year strictly < validation year. H2H is model-only. Historical betting
rules were selected after observing much of this history and do not constitute
prospective wagers. Original executable bookmaker quotes are not present here.
Never publishes/updates any Production V1 artifact, ledger, or betting record.
"""
from __future__ import annotations
import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

VERSION = 'NCAAF_PROD_V1_FIXED_BACKBONES_HISTORY_WALKFORWARD_V1_20261020'
RAW_VIEW = 'sharplogger.sharp_data.ncaaf_historical_core_training_vw'
PREFIX = 'research/ncaaf/production_v1_historical_replay'
SPREAD_FEATURES = ('Context_Intercept','Diff_RawRecent3_Off_YPP','B_RawSeason_GameAdj_Def_Rush_YPA')
TOTAL_FEATURES = ('Context_Intercept','A_RawRecent3_Def_Rush_YPA_Allowed')
H2H_FEATURES = ('H2H2_Prior_Meetings','H2H2_Prior_Margin_Current_Orientation',
                'H2H2_Prior_Total','H2H2_Days_Since','H2H2_Same_Home_Role',
                'H2H2_Recency_Margin','H2H2_Recency_Total')
THRESHOLDS = (1.0, 2.0, 3.0)  # descriptive diagnostics ONLY; not bet authorization
MIN_TRAIN_GAMES = 450
MIN_TEST_GAMES = 60


def _num(df, name):
    return pd.to_numeric(df[name] if name in df else pd.Series(np.nan,index=df.index), errors='coerce').to_numpy(dtype=float)


def _metrics(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return {'n':int(m.sum()),'rmse':float(np.sqrt(np.mean((y[m]-p[m])**2))) if m.any() else None,
            'mae':float(np.mean(np.abs(y[m]-p[m]))) if m.any() else None}


def _directional(y, p, threshold):
    valid=np.isfinite(y)&np.isfinite(p)&(np.abs(p)>=threshold)
    d=y[valid]*np.sign(p[valid]); w=int((d>0).sum()); l=int((d<0).sum()); t=int((d==0).sum())
    return {'signals_not_bets':int(valid.sum()),'wins':w,'losses':l,'pushes':t,
            'win_rate_excluding_pushes':w/(w+l) if w+l else None,
            'source':'MARKET_ERROR_DIRECTIONAL_DIAGNOSTIC','executable_prices':False,'roi':None}


def validate_frame(g):
    if g is None or g.empty: raise RuntimeError('EMPTY_HISTORICAL_GAME_FRAME')
    g=g.copy()
    required=['Season','Game_Date','Source_Game_ID','Actual_Margin','Actual_Total',
              'Consensus_Open_Spread','Consensus_Open_Total','Market_Open_Margin','Market_Open_Total',
              *SPREAD_FEATURES,*TOTAL_FEATURES]
    missing=[c for c in required if c not in g]
    if missing: raise RuntimeError('FRAME_SCHEMA_MISSING:'+','.join(missing))
    g['Season']=pd.to_numeric(g['Season'],errors='coerce')
    if g.Season.gt(2025).any():
        # Absolutely no 2026 row reaches training, validation or reporting.
        g=g[g.Season.le(2025)].copy()
    if g.empty: raise RuntimeError('NO_PRE2026_HISTORY')
    g['Game_Date']=pd.to_datetime(g['Game_Date'],utc=True,errors='coerce')
    if g['Game_Date'].isna().any():raise RuntimeError('INVALID_HISTORICAL_DATES')
    if g[['Season','Source_Game_ID']].duplicated().any():
        raise RuntimeError('DUPLICATE_PHYSICAL_GAME_IDS')
    if g['Source_Game_ID'].isna().any():raise RuntimeError('NULL_PHYSICAL_GAME_IDS')
    if 'Is_Home' in g:
        is_home=_num(g,'Is_Home')
        # Home-oriented margin scoring requires a designated home team.
        g=g[is_home==1].copy()
    if g.empty:raise RuntimeError('NO_HOME_ORIENTED_HISTORY')
    return g.sort_values(['Season','Game_Date','Source_Game_ID'],kind='stable').reset_index(drop=True)


def _fit_pair(train, val, cols, label, *, new_models, blend_predict):
    y=_num(train,label)
    good=np.isfinite(y)
    if int(good.sum())<MIN_TRAIN_GAMES:
        return np.full(len(val),np.nan),0
    for col in cols:
        if col not in train or col not in val:
            raise RuntimeError('MISSING_FIXED_BACKBONE_FEATURE:'+col)
    tr_x=train.loc[good,list(cols)].copy(); te_x=val[list(cols)].copy()
    for x in (tr_x,te_x):
        for col in cols:x[col]=pd.to_numeric(x[col],errors='coerce')
    # Fail closed when the trained predictors contain no actual useful football signal.
    varied=[c for c in cols if c!='Context_Intercept' and tr_x[c].notna().sum()>=100 and tr_x[c].nunique(dropna=True)>5]
    if not varied:return np.full(len(val),np.nan),0
    ms=new_models()
    ms[0].fit(tr_x,y[good]);ms[1].fit(tr_x,y[good])
    return np.asarray(blend_predict(ms,te_x,0.75),dtype=float),int(good.sum())


def _binary_metrics(margins, probabilities):
    valid=np.isfinite(margins)&np.isfinite(probabilities)&(margins!=0)
    if not valid.any():return {'n':0,'brier':None,'log_loss':None,'accuracy':None}
    y=(margins[valid]>0).astype(float)
    p=np.clip(probabilities[valid],1e-6,1-1e-6)
    return {'n':int(len(p)),'brier':float(np.mean((p-y)**2)),
            'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            'accuracy':float(np.mean((p>=.5)==y))}


def run_walkforward(frame, *, new_models=None, blend_predict=None, new_h2h_model=None,
                    min_train_games=MIN_TRAIN_GAMES, min_test_games=MIN_TEST_GAMES):
    """Backtest fixed recipes by expanding training seasons (not final 2026 artifact).

    The default model factories are the EXACT production/dashboard implementations.
    Test fixtures may inject equivalent callable factories.  No tuning on valid years.
    """
    if new_models is None or blend_predict is None or new_h2h_model is None:
        import sharp_line_dashboard as dash
        import ncaaf_production_v1 as prod
        new_models=new_models or dash._ncaaf_stat_new_margin_models
        blend_predict=blend_predict or prod._blend_predict
        new_h2h_model=new_h2h_model or prod._new_h2h_pipe
    global MIN_TRAIN_GAMES,MIN_TEST_GAMES
    original_min_train, original_min_test=MIN_TRAIN_GAMES,MIN_TEST_GAMES
    MIN_TRAIN_GAMES,MIN_TEST_GAMES=int(min_train_games),int(min_test_games)
    try:
        return _walkforward(frame,new_models,blend_predict,new_h2h_model)
    finally:
        MIN_TRAIN_GAMES,MIN_TEST_GAMES=original_min_train,original_min_test


def _walkforward(frame, new_models, blend_predict, new_h2h_model):
    g=validate_frame(frame)
    years=sorted(int(v) for v in g.Season.dropna().unique())
    # Recheck the original home-oriented source market columns, not a derived close.
    open_sp=_num(g,'Consensus_Open_Spread'); open_tot=_num(g,'Consensus_Open_Total')
    original_m=_num(g,'Market_Open_Margin'); original_t=_num(g,'Market_Open_Total')
    good_sp=np.isfinite(open_sp)&np.isfinite(original_m)&np.isclose(-open_sp,original_m,atol=1e-6,rtol=0)&(np.abs(open_sp)<=80)
    good_tot=np.isfinite(open_tot)&np.isfinite(original_t)&np.isclose(open_tot,original_t,atol=1e-6,rtol=0)&(open_tot>15)&(open_tot<110)
    g['__spread_ok']=good_sp
    g['__total_ok']=good_tot
    # Independently drop invalid targets without letting the outcome determine eligibility.
    g['__margin_target']=np.where(good_sp,_num(g,'Actual_Margin')-original_m,np.nan)
    g['__total_target']=np.where(good_tot,_num(g,'Actual_Total')-original_t,np.nan)
    coverage=[]
    for year in years:
        d=g[g.Season.eq(year)]
        coverage.append({'season':year,'physical_games':len(d),'spread_original_and_normalized_agree':int(d.__spread_ok.sum()),
                         'total_original_and_normalized_agree':int(d.__total_ok.sum()),
                         'spread_target_ready':int(np.isfinite(_num(d,'__margin_target')).sum()),
                         'total_target_ready':int(np.isfinite(_num(d,'__total_target')).sum()),
                         'note':'Source-vintage and executable-quote timestamp certification still pending'})
    rows=[];folds=[];prior_h2h_predictions={}
    for year in years:
        tr=g[g.Season.lt(year)];va=g[g.Season.eq(year)].copy()
        if len(tr)<MIN_TRAIN_GAMES or len(va)<MIN_TEST_GAMES:continue
        rec={'season':year,'train_seasons':sorted(int(v) for v in tr.Season.unique()),
             'n_train_physical_games':len(tr),'n_val_physical_games':len(va),'targets':{}}
        # Enforce identical training years for all fixed heads, but each requires its own valid opening source.
        tr_sp=tr[tr.__spread_ok & np.isfinite(_num(tr,'__margin_target'))]
        tr_tt=tr[tr.__total_ok & np.isfinite(_num(tr,'__total_target'))]
        val_sp=va[va.__spread_ok & np.isfinite(_num(va,'__margin_target'))]
        val_tt=va[va.__total_ok & np.isfinite(_num(va,'__total_target'))]
        pm,nm=_fit_pair(tr_sp,val_sp,SPREAD_FEATURES,'__margin_target',new_models=new_models,blend_predict=blend_predict) if len(val_sp) else (np.array([]),0)
        pt,nt=_fit_pair(tr_tt,val_tt,TOTAL_FEATURES,'__total_target',new_models=new_models,blend_predict=blend_predict) if len(val_tt) else (np.array([]),0)
        rec['targets']['spread']={'n_train_with_verified_open_fields':nm,'n_test':len(val_sp),
                                  'market_rmse':_metrics(_num(val_sp,'Actual_Margin'),_num(val_sp,'Market_Open_Margin')),
                                  'model_rmse':_metrics(_num(val_sp,'Actual_Margin'),_num(val_sp,'Market_Open_Margin')+pm),
                                  'directional':{str(t):_directional(_num(val_sp,'__margin_target'),pm,t) for t in THRESHOLDS}}
        rec['targets']['total']={'n_train_with_verified_open_fields':nt,'n_test':len(val_tt),
                                 'market_rmse':_metrics(_num(val_tt,'Actual_Total'),_num(val_tt,'Market_Open_Total')),
                                 'model_rmse':_metrics(_num(val_tt,'Actual_Total'),_num(val_tt,'Market_Open_Total')+pt),
                                 'directional':{str(t):_directional(_num(val_tt,'__total_target'),pt,t) for t in THRESHOLDS}}
        sp_preds=dict(zip(val_sp['Source_Game_ID'].astype(str),pm))
        prior_h2h_predictions.update({(year,k):v for k,v in sp_preds.items() if np.isfinite(v)})
        # H2H is a distinct logistic model, trained ONLY on PREVIOUS YEAR margin OOF predictions.
        # Never substitute fitted in-sample margin estimates.
        if all(c in g for c in H2H_FEATURES) and sp_preds:
            train_h2h=tr_sp.copy()
            oof=np.array([prior_h2h_predictions.get((int(y),str(gid)),np.nan) for y,gid in zip(train_h2h.Season,train_h2h.Source_Game_ID)],float)
            ok=np.isfinite(oof)&np.isfinite(_num(train_h2h,'Actual_Margin'))
            eligible_h2h=int(ok.sum())
            if eligible_h2h>=MIN_TRAIN_GAMES and np.unique((_num(train_h2h,'Actual_Margin')[ok]>0).astype(int)).size>1:
                hx=train_h2h.loc[ok,list(H2H_FEATURES)].copy();hx['STAT_Fair_Margin']=_num(train_h2h,'Market_Open_Margin')[ok]+oof[ok]
                for c in hx.columns:hx[c]=pd.to_numeric(hx[c],errors='coerce')
                yh=(_num(train_h2h,'Actual_Margin')[ok]>0).astype(int)
                tx=val_sp[list(H2H_FEATURES)].copy();tx['STAT_Fair_Margin']=_num(val_sp,'Market_Open_Margin')+pm
                for c in tx.columns:tx[c]=pd.to_numeric(tx[c],errors='coerce')
                valid=np.isfinite(pm)
                probs=np.full(len(tx),np.nan)
                if valid.any():
                    clf=new_h2h_model();clf.fit(hx,yh)
                    probs[valid]=clf.predict_proba(tx.loc[valid])[:,1]
                rec['targets']['h2h_model_only']={'n_training_prior_oof':eligible_h2h,**_binary_metrics(_num(val_sp,'Actual_Margin'),probs)}
            else:rec['targets']['h2h_model_only']={'status':'INSUFFICIENT_PRIOR_OOF','n_training_prior_oof':eligible_h2h}
        else:rec['targets']['h2h_model_only']={'status':'H2H_NOT_EVALUABLE'}
        for idx, r in val_sp.reset_index(drop=True).iterrows():
            rows.append({'Season':year,'Source_Game_ID':str(r.Source_Game_ID),'Game_Date':str(r.Game_Date),
                         'Market':'spreads','opening_anchor':float(r.Market_Open_Margin),'actual':float(r.Actual_Margin),
                         'model_adjustment':float(pm[idx]) if np.isfinite(pm[idx]) else None,
                         'model_forecast':float(r.Market_Open_Margin+pm[idx]) if np.isfinite(pm[idx]) else None,
                         'signal_not_bet':None if not np.isfinite(pm[idx]) else ('HOME_ATS' if pm[idx]>0 else 'AWAY_ATS' if pm[idx]<0 else 'NEUTRAL')})
        for idx, r in val_tt.reset_index(drop=True).iterrows():
            rows.append({'Season':year,'Source_Game_ID':str(r.Source_Game_ID),'Game_Date':str(r.Game_Date),
                         'Market':'totals','opening_anchor':float(r.Market_Open_Total),'actual':float(r.Actual_Total),
                         'model_adjustment':float(pt[idx]) if np.isfinite(pt[idx]) else None,
                         'model_forecast':float(r.Market_Open_Total+pt[idx]) if np.isfinite(pt[idx]) else None,
                         'signal_not_bet':None if not np.isfinite(pt[idx]) else ('OVER' if pt[idx]>0 else 'UNDER' if pt[idx]<0 else 'NEUTRAL')})
        folds.append(rec)
    if not folds:raise RuntimeError('INSUFFICIENT_LONG_HISTORY_FOR_SEASON_FORWARD_BACKTEST')
    report={'version':VERSION,'status':'RESEARCH_FIXED_BACKBONE_REPLAY__HISTORICAL_SELECTION_AND_SOURCE_VINTAGE_HOLD',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'source':RAW_VIEW,
        'train_rule':'ALL_ELIGIBLE_PRIOR_SEASONS_ONLY','fixed_spread_features':list(SPREAD_FEATURES),
        'fixed_total_features':list(TOTAL_FEATURES),'years_evaluated':[f['season'] for f in folds],
        'annual_source_coverage':coverage,'folds':folds,'production_artifact_used':False,
        'production_model_modified':False,'production_bet_authority':0,
        'bet_level_roi_supported':False,'backtest_type':'RESEARCH_CHRONOLOGICAL_FIXED_BACKBONE_RETRAIN',
        'critical_limits':[
            'Not a replay of the already fitted 2026 Production V1 artifact.',
            'Historical selection systems/miners were researched with later outcomes; their betting authority has not been independently reconstructed as of each historical date.',
            'No executable bookmaker prices/timestamps supplied; directional diagnostics are not wagers or ROI.',
            'Market opening columns agree internally; source publication vintage and authenticity remain unverified.',
            'Historical feature engineer uses prior-game values but source as-of timestamps require independent validation.',
            '2024-2025 were repeatedly consulted during model development, so this is not pristine model-selection validation.',
            'All 2026 game outcomes excluded.',
        ]}
    return report,pd.DataFrame(rows)


def run_cloud_research(*,dashboard_module,storage_client,bucket_name='sharp-models',log_func=print):
    from google.cloud import bigquery
    from ncaaf_production_v1 import _blend_predict, _new_h2h_pipe
    bq=bigquery.Client(project='sharplogger')
    sql=f'SELECT * FROM `{RAW_VIEW}` WHERE Historical_Core_Eligible = 1 AND Season BETWEEN 2013 AND 2025'
    log_func('[NCAAF-PROD-V1-BACKTEST] phase=READ_BQ source='+RAW_VIEW+' 2026_queried=FALSE')
    raw=bq.query(sql).to_dataframe(create_bqstorage_client=False)
    if raw.empty:raise RuntimeError('HISTORICAL_CORE_VIEW_EMPTY')
    log_func('[NCAAF-PROD-V1-BACKTEST] phase=BUILD_PRIOR_GAME_FEATURES sides='+str(len(raw)))
    frame,_,_=dashboard_module._ncaaf_stat_build_game_frame(raw)
    log_func('[NCAAF-PROD-V1-BACKTEST] phase=SEASON_FORWARD_TEST games='+str(len(frame)))
    report,preds=run_walkforward(frame,new_models=dashboard_module._ncaaf_stat_new_margin_models,
               blend_predict=_blend_predict,new_h2h_model=_new_h2h_pipe)
    bucket=storage_client.bucket(bucket_name)
    ts=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    txt=json.dumps(report,allow_nan=False,indent=2,default=str)
    for name in (f'{PREFIX}/runs/{ts}/report.json',f'{PREFIX}/latest_report.json'):
        bucket.blob(name).upload_from_string(txt,content_type='application/json')
    buffer=io.StringIO();preds.to_csv(buffer,index=False)
    bucket.blob(f'{PREFIX}/runs/{ts}/predictions.csv').upload_from_string(buffer.getvalue(),content_type='text/csv')
    log_func('[NCAAF-PROD-V1-BACKTEST] status='+report['status']+' folds='+str(len(report['folds']))+
             ' spread='+str(sum(f['targets']['spread']['n_test'] for f in report['folds']))+
             ' totals='+str(sum(f['targets']['total']['n_test'] for f in report['folds']))+' bet_authority=0')
    return report
