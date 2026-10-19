"""NCAAF historical fair-margin -> authentic-opening Core shadow challenger.

Strict two-stage season-forward research, no changes to frozen Production V1:
1) first-stage fair margin is OUT OF FOLD, produced without market prices;
2) second-stage learns actual-minus-authentic-opening residual only from
   PRIOR seasons that themselves have genuine opening quotes and OOF scores;
3) recipe selection uses validation years <= 2023 only; 2024/2025 are
   retrospective diagnostics, 2026 outcomes are never inspected.

The original Core incumbent and market are evaluated on the exact same games.
No model is promoted or passed to Bet Authority. Big Al/Pathi remain separate.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable
import math
import re
import numpy as np
import pandas as pd

VERSION='NCAAF_HISTORICAL_TO_CORE_BRIDGE_V1_20261018'
MARKET_TRAIN_MIN_YEAR=2021  # 2013-20 archives did NOT contain genuine opening quotes
SELECTION_YEARS=(2022,2023)
RETROSPECTIVE_YEARS=(2024,2025)
MIN_META_TRAIN=100
MIN_EVAL=50
ALPHA=35.0
PROVEN=('Diff_RawRecent3_Off_YPP','B_RawSeason_GameAdj_Def_Rush_YPA')
ALLOWED_FAMILIES=('scoring_efficiency','explosiveness','historical_score_form','modern_cfbd_pregame')


def _val(df:pd.DataFrame,c:str)->np.ndarray:
    if c not in df: return np.full(len(df),np.nan,dtype=float)
    return pd.to_numeric(df[c],errors='coerce').to_numpy(dtype=float)


def _metrics(y,p)->dict[str,Any]:
    ok=np.isfinite(y)&np.isfinite(p);n=int(ok.sum())
    if not n:return {'n':0,'rmse':None,'mae':None}
    delta=y[ok]-p[ok]
    return {'n':n,'rmse':float(np.sqrt(np.mean(delta**2))),'mae':float(np.mean(np.abs(delta)))}


_TRUSTED_SOURCE=re.compile(
    r'^(?:(?:A|B|Diff|Mean|Sum)_(?:RawSeason_|RawRecent3_|RawRecent5_|State_|Prior_|Prev_)|'
    r'H2H2_(?:Prior_|Recency_|Days_Since|Same_Home_Role)|'
    r'CFBD_(?:Opp_)?(?:Season|Recent3|Recent5|PrevSeason)_CFBD_Game_.*_Prior$)',
    re.IGNORECASE)
_BLOCKED=re.compile(
    r'(opening|closing|open_|close_|market|moneyline|odds|spread|ats|clv|'
    r'betting|book|line_move|result|actual|final|postgame|target|label|'
    r'team_score|opponent_score|current_game|game_outcome|prediction|pt_meta)',
    re.IGNORECASE)


def _safe_extra(c:str,modern=False)->bool:
    # Keep the same strict source contract as historical-signal discovery.
    if not _TRUSTED_SOURCE.match(c) or _BLOCKED.search(c):return False
    if modern:return bool(c.startswith('CFBD_') and '_cfbd_game_' in c.lower() and c.lower().endswith('_prior'))
    return not c.startswith('CFBD_')


def _source_columns(g:pd.DataFrame,discovery:dict|None)->dict[str,list[str]]:
    """Pre-2024-selected names only; never select using 2024/25 performance."""
    results=(discovery or {}).get('families') or {}
    selected=(discovery or {}).get('selection',{}).get('margin',{}).get('selection_passed') or []
    # Only pre-2024-selected historical families are eligible to be included.
    names=[]
    for f in ('scoring_efficiency','explosiveness','historical_score_form'):
        if f in selected:
            names.append(f)
    group={}
    for family in names:
        item=results.get(family) or {}
        cols=[str(c) for c in (item.get('features') or []) if c in g.columns and _safe_extra(str(c))]
        if cols:group[family]=cols[:3]
    # Modern is a distinct 2022+ lane, never used to drop foundation history.
    mod=(results.get('modern_cfbd_pregame') or {}).get('features') or []
    cols=[str(c) for c in mod if c in g.columns and _safe_extra(str(c),modern=True)]
    if cols:group['modern_cfbd_pregame']=cols[:3]
    return group


def _authentic_market(g:pd.DataFrame)->tuple[np.ndarray,dict[str,Any]]:
    """Fail closed on missing explicit opening source: never borrow closing quotes."""
    y=_val(g,'Season'); op=_val(g,'Market_Open_Margin')
    raw=_val(g,'Consensus_Open_Spread')
    # This is the observed source column from the audited historical market feed.
    # The margin must agree with the sign-transformed original line.
    ok=(np.isfinite(y)&(y>=MARKET_TRAIN_MIN_YEAR)&(y<=2025)&
        np.isfinite(op)&np.isfinite(raw)&np.isclose(op,-raw,atol=1e-5,rtol=0)&
        (np.abs(op)<=80)&(np.abs(raw)<=80))
    audit={'opening_source':'Consensus_Open_Spread', 'market_target':'Market_Open_Margin',
           'eligibility':'EXPLICIT_SOURCE_AND_SIGN_CHECK__NO_CLOSE_PROXY',
           'market_eligible':int(ok.sum()),
           'by_year':{str(int(yr)):{'games':int(np.sum(y==yr)),'authentic_open':int(np.sum(ok&(y==yr)))}
                      for yr in sorted(set(y[np.isfinite(y)].astype(int)))}}
    return ok,audit


def _fit_ridge(train:pd.DataFrame,valid:pd.DataFrame,columns:list[str])->np.ndarray:
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    if not columns or len(train)<MIN_META_TRAIN:return np.full(len(valid),np.nan)
    qualified=[]
    for c in columns:
        a=_val(train,c); n=int(np.isfinite(a).sum())
        if n<max(40,int(len(train)*0.15)):continue
        if len(np.unique(a[np.isfinite(a)]))<3:continue
        qualified.append(c)
    if not qualified:return np.full(len(valid),np.nan)
    xtr=train.loc[:,qualified].apply(pd.to_numeric,errors='coerce')
    xva=valid.loc[:,qualified].apply(pd.to_numeric,errors='coerce')
    yy=_val(train,'_ResidualTarget')
    ok=np.isfinite(yy)
    if ok.sum()<MIN_META_TRAIN:return np.full(len(valid),np.nan)
    model=Pipeline([('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler()),('ridge',Ridge(alpha=ALPHA))])
    model.fit(xtr.loc[ok], yy[ok])
    return _val(valid,'Market_Open_Margin') + model.predict(xva)


def _predict_market_blind(train:pd.DataFrame,valid:pd.DataFrame)->np.ndarray:
    """Diagnostic forecast of the *opening market*, using football-only OOF inputs.

    Training labels are authentic historical openings; an opening price (or any
    later quote) is NEVER a predictor of its own value. Do not use this to claim
    football-outcome accuracy or generate bets.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    cols=['_FairOOF']+[c for c in PROVEN if c in train.columns]
    # Do not let all-null historical optional columns destabilize the fit.
    cols=[c for c in cols if np.isfinite(_val(train,c)).sum()>=max(40,int(len(train)*0.15))]
    if len(train)<MIN_META_TRAIN or '_FairOOF' not in cols:
        return np.full(len(valid),np.nan)
    xtr=train.loc[:,cols].apply(pd.to_numeric,errors='coerce')
    xva=valid.loc[:,cols].apply(pd.to_numeric,errors='coerce')
    y=_val(train,'Market_Open_Margin')
    ok=np.isfinite(y)&np.isfinite(_val(train,'_FairOOF'))
    if ok.sum()<MIN_META_TRAIN:return np.full(len(valid),np.nan)
    model=Pipeline([('impute',SimpleImputer(strategy='median')),
                    ('scale',StandardScaler()),('ridge',Ridge(alpha=ALPHA))])
    model.fit(xtr.loc[ok],y[ok])
    return model.predict(xva)


def _football_oof_by_year(g:pd.DataFrame,market_eligibility:np.ndarray)->dict[str,Any]:
    """Grade independent football forecasts on every observed game, not just
    the modern games with betting quotes. The market matched slice is separate.
    """
    sy=_val(g,'Season');y=_val(g,'Actual_Margin');p=_val(g,'_FairOOF')
    out={}
    for yr in sorted(set(sy[np.isfinite(sy)].astype(int))):
        m=(sy==yr)
        out[str(yr)]={'all_games':_metrics(y[m],p[m]),
                      'market_matched':_metrics(y[m&market_eligibility],p[m&market_eligibility])}
    return out


def _paired(y,p1,p2):
    mask=np.isfinite(y)&np.isfinite(p1)&np.isfinite(p2)
    if mask.sum()<1:return {'n':0,'rmse_gain':None,'mae_gain':None}
    a=_metrics(y[mask],p1[mask]);b=_metrics(y[mask],p2[mask])
    return {'n':int(mask.sum()),'rmse_gain':a['rmse']-b['rmse'], 'mae_gain':a['mae']-b['mae']}


def _prep(g:pd.DataFrame,margin_oof:np.ndarray,discovery:dict|None):
    years=_val(g,'Season')
    if np.isfinite(years).any() and np.nanmax(years)>2025:
        raise ValueError('HISTORICAL_CORE_2026_LABEL_SEAL')
    if len(margin_oof)!=len(g):raise ValueError('HISTORICAL_CORE_OOF_LENGTH_MISMATCH')
    if 'Source_Game_ID' in g.columns:
        ids=g['Source_Game_ID'].astype('string').fillna('').str.strip()
        season_ids=g['Season'].astype('string').fillna('').str.strip()
        meaningful=ids.ne('') & ~ids.str.lower().isin(('nan','none','<na>'))
        if bool((season_ids[meaningful]+'/'+ids[meaningful]).duplicated().any()):
            raise ValueError('HISTORICAL_CORE_DUPLICATED_PHYSICAL_GAMES')
    authentic,market_audit=_authentic_market(g)
    fair=np.asarray(margin_oof,dtype=float).copy()
    if not np.isfinite(fair).any():raise ValueError('HISTORICAL_CORE_MISSING_SCORE_OOF')
    frame=g.copy().reset_index(drop=True)
    frame['_FairOOF']=fair
    frame['_FairVsMarket']=fair-_val(frame,'Market_Open_Margin')
    frame['_ResidualTarget']=_val(frame,'Actual_Margin')-_val(frame,'Market_Open_Margin')
    # Missing ineligible openers remain NaN, and are never synthesized from closes.
    frame.loc[~authentic,'_FairVsMarket']=np.nan
    frame.loc[~authentic,'_ResidualTarget']=np.nan
    family=_source_columns(frame,discovery)
    variants={'FAIR_DISAGREEMENT':['_FairVsMarket'],
              'FAIR_PLUS_PROVEN_CORE':['_FairVsMarket']+[c for c in PROVEN if c in frame.columns]}
    historical=[c for f in ('scoring_efficiency','explosiveness','historical_score_form') for c in family.get(f,[])]
    if historical:
        variants['FAIR_PLUS_VALIDATED_HISTORY']=list(dict.fromkeys(variants['FAIR_PLUS_PROVEN_CORE']+historical))
    if family.get('modern_cfbd_pregame'):
        variants['FAIR_PLUS_MODERN_CFBD']=list(dict.fromkeys(variants['FAIR_PLUS_PROVEN_CORE']+family['modern_cfbd_pregame']))
    return frame,authentic,market_audit,family,variants


def run_historical_to_core_bridge(g:pd.DataFrame,score_oof:dict,
                                  discovery:dict|None,
                                  incumbent_predict:Callable[[pd.DataFrame,pd.DataFrame],np.ndarray],
                                  *,log_func=print)->dict[str,Any]:
    """Build a research-only season-forward market-aware core challenger.

    No 2024/25 labels are used in feature/recipe selection; 2025 model fitting
    is allowed to use 2024 completed outcomes, strictly forward in time.
    """
    margin_oof=score_oof.get('margin')
    if margin_oof is None:raise RuntimeError('HISTORICAL_CORE_OOF_SINK_NOT_POPULATED')
    frame,authentic,audit,family,variants=_prep(g,np.asarray(margin_oof),discovery)
    season=_val(frame,'Season');target=_val(frame,'Actual_Margin')
    years=sorted(set(int(y) for y in season[np.isfinite(season)]))
    # Ensure first-stage OOF is present for training *and* out-of-sample comparisons.
    usable=authentic&np.isfinite(_val(frame,'_FairVsMarket'))&np.isfinite(target)
    rows_by_year={y:int(np.sum(usable&(season==y))) for y in years}
    folds={}
    for year in (*SELECTION_YEARS,*RETROSPECTIVE_YEARS):
        tr=frame.loc[usable&(season<year)].copy()
        va=frame.loc[usable&(season==year)].copy()
        if len(tr)<MIN_META_TRAIN or len(va)<MIN_EVAL:
            folds[str(year)]={'status':'SKIPPED_SOURCE_LIMITED','train':len(tr),'validation':len(va)}
            log_func(f'[NCAAF-HISTORICAL-CORE-BRIDGE-FOLD] val_year={year} status=SOURCE_LIMITED train={len(tr)} eval={len(va)}')
            continue
        yy=_val(va,'Actual_Margin');open_margin=_val(va,'Market_Open_Margin')
        # TWO DISTINCT TARGETS: a football-outcome forecast and a market-price
        # forecast. The latter is a diagnostic; the actionable market-aware
        # football predictor remains the genuine-open residual challenger.
        pure_fair=_val(va,'_FairOOF')
        market_blind_open=_predict_market_blind(tr,va)
        # Same previous-season population as the existing Core _evaluate(),
        # not the narrower OOF subset used to fit this challenger.
        baseline_train=frame.loc[season<year].copy()
        cp=np.asarray(incumbent_predict(baseline_train,va),dtype=float)
        rec={'status':'PASS','train':len(tr),'validation':len(va),
             'baseline':{'market':_metrics(yy,open_margin),'incumbent_core':_metrics(yy,cp),
                         'football_only_fair_margin':_metrics(yy,pure_fair)},
             'market_forecast_diagnostic':{
                 'target':'ORIGINAL_OPENING_MARGIN_NOT_GAME_OUTCOME',
                 'status':'DIAGNOSTIC_ONLY_NO_BETTING_AUTHORITY',
                 'prediction_error_vs_opening':_metrics(open_margin,market_blind_open),
                 'actual_open_vs_game_outcome':_metrics(yy,open_margin),
                 'no_current_market_price_predictor':True,
                 'no_current_game_outcome_predictor':True},
             'variants':{}}
        for key,cols in variants.items():
            modern=(key=='FAIR_PLUS_MODERN_CFBD')
            modern_features=family.get('modern_cfbd_pregame',[])
            supported=(not modern) or (
                year>=2023 and sum(int(np.isfinite(_val(tr,c)).sum()) for c in modern_features)>=150)
            pred=_fit_ridge(tr,va,cols) if supported else np.full(len(va),np.nan)
            rec['variants'][key]={'features':cols,
                                  'modern_training_eligible':bool(supported) if modern else None,
                                  'performance':_metrics(yy,pred),
                                  'vs_market':_paired(yy,open_margin,pred),
                                  'vs_core':_paired(yy,cp,pred)}
        folds[str(year)]=rec
        gain_string={k:round(v['vs_core']['rmse_gain'],4) for k,v in rec['variants'].items()
                     if v['vs_core']['rmse_gain'] is not None}
        log_func(f'[NCAAF-HISTORICAL-CORE-DUAL-TARGET] year={year} '
                 f'football_actual_rmse={rec["baseline"]["football_only_fair_margin"]["rmse"]:.4f} '
                 f'market_open_forecast_rmse={rec["market_forecast_diagnostic"]["prediction_error_vs_opening"]["rmse"]:.4f} '
                 'target_separation=TRUE market_forecast_authority=0')
        log_func(f'[NCAAF-HISTORICAL-CORE-BRIDGE-FOLD] val_year={year} train={len(tr)} '
                 f'eval={len(va)} core_rmse={rec["baseline"]["incumbent_core"]["rmse"]:.4f} '
                 f'market_rmse={rec["baseline"]["market"]["rmse"]:.4f} candidate_gains={gain_string}')

    # Candidate selection is based *only* on 2022/2023 folds, and must be
    # better than Core on both years to earn an exploratory shadow nomination.
    ranks=[]
    for name in variants:
        metrics=[]
        for y in SELECTION_YEARS:
            r=folds.get(str(y),{})
            if r.get('status')=='PASS':
                v=r['variants'][name]['vs_core']
                if v['rmse_gain'] is not None: metrics.append((v['n'],v['rmse_gain'],v['mae_gain']))
        ok=len(metrics)==2 and all(n>=MIN_EVAL and rg>0 and mg>=-0.05 for n,rg,mg in metrics)
        avg=float(sum(n*g for n,g,_ in metrics)/sum(n for n,_,_ in metrics)) if metrics else None
        ranks.append({'candidate':name,'selection_years':[int(y) for y in SELECTION_YEARS if folds.get(str(y),{}).get('status')=='PASS'],
                      'weighted_rmse_gain_vs_core':avg,'both_selection_years_positive':bool(ok),
                      'status':'PRE2024_RESEARCH_NOMINATION' if ok else 'NO_SELECTION_QUALIFICATION'})
    ranked=sorted((x for x in ranks if x['both_selection_years_positive']),key=lambda x:x['weighted_rmse_gain_vs_core'],reverse=True)
    selected=ranked[0]['candidate'] if ranked else None
    # Pooled paired comparison must align games for all three predictors.
    selected_retrospective={}
    if selected:
        for y in RETROSPECTIVE_YEARS:
            r=folds.get(str(y),{})
            if r.get('status')=='PASS':
                selected_retrospective[str(y)]=r['variants'][selected]
    status='PASS' if any(v.get('status')=='PASS' for v in folds.values()) else 'SOURCE_LIMITED'
    report={'version':VERSION,'status':status,'source_years':years,
            'score_oof_rows':int(np.isfinite(_val(frame,'_FairOOF')).sum()),
            'authentic_open_audit':audit,'usable_by_year':rows_by_year,
            'football_only_oof_by_year':_football_oof_by_year(frame,authentic),
            'distinct_targets':{'football_outcome':'ACTUAL_MARGIN__ALL_SCORE_YEARS',
                                'market_expectation':'GENUINE_ORIGINAL_OPENING_MARGIN__SOURCE_ELIGIBLE_ONLY',
                                'market_conditioned_outcome':'ACTUAL_MARGIN_MINUS_GENUINE_ORIGINAL_OPENING',
                                'market_forecast':'DIAGNOSTIC_ONLY',
                                'cover_probability':'NOT_CALIBRATED__NO_BETTING_AUTHORITY'},
            'variants':{k:v for k,v in variants.items()},'family_provenance':family,
            'folds':folds,'pre2024_selection':ranks,'selected_research_candidate':selected,
            'selected_retrospective_2024_2025':selected_retrospective,
            'selection_policy':'2022_2023_ONLY__TWO_POSITIVE_YEARS_REQUIRED',
            'confirmation_policy':'2024_2025_RETROSPECTIVE_REPEATED_INSPECTION_NOT_PRISTINE',
            'prospective_start':2026,'production_authority':0,'automatic_promotion':False,
            'no_betting_actions':True,'market_source':'GENUINE_ORIGINAL_OPENING_ONLY_NO_CLOSE_PROXY',
            'note':'Report includes exact-game Core and opening-market baselines. Research only.'}
    log_func(f'[NCAAF-HISTORICAL-CORE-BRIDGE] status={status} oof={report["score_oof_rows"]} '
             f'open_eligible={audit["market_eligible"]} selected_pre2024={selected} '
             f'retrospective_years={list(selected_retrospective)} production_authority=0')
    return report
