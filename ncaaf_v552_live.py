"""V5.5.2 independent CFBD CORE scorer and quote-aware combination slate.

Unlike V5.5.1, this module never uses Production V1's predictions, selected
picks, or Bet Authority. The *only* optional shared component is the immutable
pregame quote normalization and the existing read-only qualified Miner adapter.
Full-run training is from 2013--2025 completed CFBD football games. A new 2026
CFBD scheduled-game feature frame supplies future inputs and prior completed
2026 games update the existing strictly-prior state machine.
"""
from __future__ import annotations
from datetime import datetime, timezone
from io import StringIO
from collections import defaultdict
import hashlib
import json
import math
import re
import numpy as np
import pandas as pd

VERSION='NCAAF_V552_INDEPENDENT_CORE_WEEKLY_20261021'
ROOT='research/ncaaf/v55_weekly'
HISTORY_SOURCE='sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1'
FUTURE_SOURCE='sharplogger.sharp_data.ncaaf_cfbd_miner_features_v1'
SCORES_SOURCE='sharplogger.sharp_data.ncaaf_cfbd_api_archive'
CHAMPION='NONLINEAR_HGB'
FAMILIES=('STAT','BIG_AL','PATHI','SYSTEM_MINER','PREDICTION_TRACKER')
QUOTE_COLUMNS=('Market','Game_Start','Outcome','Value','Odds_Price','Bookmaker',
               'Snapshot_Timestamp','Home_Team_Norm','Away_Team_Norm','_prod_game_id')


def _now(now=None):
    d=pd.to_datetime(now or datetime.now(timezone.utc),utc=True,errors='coerce')
    if pd.isna(d): raise RuntimeError('V552_INVALID_NOW')
    return d


def _num(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else float('nan')
    except (ValueError,TypeError,OverflowError):return float('nan')


def _school(value):
    from ncaaf_v54_cfbd_truth import cfbd_school_key
    return cfbd_school_key(value)


def _game_id(value):
    from ncaaf_v54_full_research import _canonical_cfbd_research_join_keys
    sample=pd.DataFrame({'Season':[2026], 'Game_ID':[value]})
    return str(_canonical_cfbd_research_join_keys(sample,source='V552_GAME_ID').Game_ID.iloc[0])


def prepare_2026_feature_games(raw,*,now=None):
    """Only the CFBD warehouse's 2026 pregame two-sided schedule is authoritative."""
    utc=_now(now)
    if not isinstance(raw,pd.DataFrame) or raw.empty:
        raise RuntimeError('V552_CFBD_2026_FEATURES_EMPTY__RUN_CFBD_WAREHOUSE_REFRESH')
    required={'Season','Game_ID','Game_Date','Team','Opponent','Is_Home',
              'CFBD_Leakage_Contract','CFBD_Source_Tag','CFBD_Season_Net_PPA_Prior'}
    if required-set(raw):raise RuntimeError('V552_CFBD_2026_COLUMNS_MISSING '+str(sorted(required-set(raw))))
    s=raw.copy()
    yr=pd.to_numeric(s.Season,errors='coerce')
    if not yr.eq(2026).all():raise RuntimeError('V552_CURRENT_FRAME_WRONG_SEASON')
    if not s.CFBD_Leakage_Contract.astype(str).str.contains('SHIFT1',regex=False).all():
        raise RuntimeError('V552_CFBD_PREGAME_SHIFT_CONTRACT_MISSING')
    if not s.CFBD_Source_Tag.astype(str).str.startswith('ncaaf-cfbd-warehouse-').all():
        raise RuntimeError('V552_CFBD_SOURCE_TAG_MISSING')
    s['Game_ID']=s.Game_ID.map(_game_id)
    s['Game_Date']=pd.to_datetime(s.Game_Date,utc=True,errors='coerce')
    if s.Game_Date.isna().any():raise RuntimeError('V552_CFBD_GAME_KICKOFF_UNKNOWN')
    s['Is_Home']=pd.to_numeric(s.Is_Home,errors='coerce')
    if not s.Is_Home.isin((0,1)).all():raise RuntimeError('V552_CFBD_HOME_ORIENTATION_MISSING')
    s['_team_key']=s.Team.map(_school);s['_opponent_key']=s.Opponent.map(_school)
    if s['_team_key'].eq('').any() or (s['_team_key']==s['_opponent_key']).any():
        raise RuntimeError('V552_CFBD_TEAM_IDENTITY_INVALID')
    for gid,g in s.groupby('Game_ID'):
        if len(g)!=2 or set(g.Is_Home)!={0,1} or g._team_key.nunique()!=2 or set(g._team_key)!=set(g._opponent_key):
            raise RuntimeError('V552_CFBD_TWO_SIDED_SCHEDULE_MISMATCH game_id='+gid)
        if g.Game_Date.nunique()!=1:raise RuntimeError('V552_CFBD_MULTIPLE_KICKOFFS game_id='+gid)
    s['Team_Norm']=s.Team.astype(str).str.lower().str.strip()
    s['Opponent_Norm']=s.Opponent.astype(str).str.lower().str.strip()
    s['Source_Game_ID']=s.Game_ID
    # Never take outcomes from the modern feature/market warehouse.
    s['Actual_Margin']=np.nan;s['Actual_Total']=np.nan
    home=s[s.Is_Home.eq(1)].copy().reset_index(drop=True)
    return s,home,{'current_games':len(home),'upcoming_games':int(home.Game_Date.gt(utc).sum()),
                   'history_score_from_modern_features':False,'feature_source':FUTURE_SOURCE}


def attach_2026_past_scores(home,scores,*,now=None,min_past_completion_coverage=.95):
    """Completed previous games update V4 state; never silently ignore 2026 results."""
    utc=_now(now)
    out=home.copy()
    previous=pd.to_datetime(out.Game_Date,utc=True,errors='coerce').lt(utc.normalize())
    if scores is None or scores.empty:
        if previous.any():
            raise RuntimeError('V552_2026_PAST_GAME_SCORES_MISSING__REFRESH_IMMUTABLE_CFBD_ARCHIVE')
        return out
    scored=scores.copy()
    need={'Season','Game_ID','CFBD_Home','CFBD_Away','CFBD_Home_Points','CFBD_Away_Points'}
    if need-set(scored):raise RuntimeError('V552_CFBD_SCORE_ARCHIVE_COLUMNS_MISSING')
    scored=scored.loc[pd.to_numeric(scored.Season,errors='coerce').eq(2026)].copy()
    scored['Game_ID']=scored.Game_ID.map(_game_id)
    if scored.Game_ID.duplicated().any():raise RuntimeError('V552_SCORE_ARCHIVE_DUPLICATE_GAMES')
    out=out.merge(scored[['Game_ID','CFBD_Home','CFBD_Away','CFBD_Home_Points','CFBD_Away_Points']+(['Retrieved_At'] if 'Retrieved_At' in scored else [])],on='Game_ID',how='left',validate='one_to_one')
    h=_school
    has=out.CFBD_Home.notna()
    invalid=has & ((out.Team.map(h)!=out.CFBD_Home.map(h)) | (out.Opponent.map(h)!=out.CFBD_Away.map(h)))
    if invalid.any():raise RuntimeError('V552_CFBD_ARCHIVE_TEAM_CONFLICT')
    past=out.Game_Date.lt(utc.normalize())
    safe=past & has
    if 'Retrieved_At' in out:
        retrieved=pd.to_datetime(out.Retrieved_At,utc=True,errors='coerce')
        safe=safe & retrieved.notna() & retrieved.le(utc) & retrieved.ge(out.Game_Date)

    home_pts=pd.to_numeric(out.CFBD_Home_Points,errors='coerce')
    away_pts=pd.to_numeric(out.CFBD_Away_Points,errors='coerce')
    safe=safe & home_pts.notna() & away_pts.notna()
    historical_count=int(past.sum())
    actual_count=int(safe.sum())
    if historical_count and (actual_count/historical_count)<min_past_completion_coverage:
        raise RuntimeError(f'V552_2026_PRIOR_CFBD_SCORE_COVERAGE_HOLD:{actual_count}/{historical_count}')
    out['Actual_Margin']=np.where(safe,home_pts-away_pts,np.nan)
    out['Actual_Total']=np.where(safe,home_pts+away_pts,np.nan)
    return out.drop(columns=['CFBD_Home','CFBD_Away','CFBD_Home_Points','CFBD_Away_Points','Retrieved_At'],errors='ignore')


def fit_champion_and_score(history, current_home, *, oof, now=None, min_current_feature_coverage=.25):
    """Reproduce V4 winner NONLINEAR_HGB recipe; train only seasons <=2025."""
    import ncaaf_margin_tournament_v4 as v4
    utc=_now(now)
    hist=history.copy();cur=current_home.copy()
    required={'Season','Game_Date','Team_Norm','Opponent_Norm','Actual_Margin','Actual_Total','Is_Home','Source_Game_ID'}
    if required-set(hist) or required-set(cur):raise RuntimeError('V552_V4_CORE_FRAME_MISSING_REQUIRED_FIELDS')
    hyrs=pd.to_numeric(hist.Season,errors='coerce')
    if not hyrs.between(2013,2025).all() or len(hist)<8000:
        raise RuntimeError('V552_TRAINING_HISTORY_INVALID_OR_INCOMPLETE')
    cy=pd.to_numeric(cur.Season,errors='coerce')
    if not cy.eq(2026).all():raise RuntimeError('V552_CURRENT_CANNOT_ENTER_HISTORICAL_FIT')
    if cur.Source_Game_ID.duplicated().any():raise RuntimeError('V552_DUPLICATE_2026_GAME_ID')
    if not cur.Game_Date.gt(utc).any():raise RuntimeError('V552_NO_UPCOMING_CFBD_GAMES')
    # Never mix current outcomes into target fitting. Historical records are
    # scored solely from the canonical CFBD /games archive in V5.4.
    base=pd.concat([hist,cur],ignore_index=True,sort=False)
    base['Game_Date']=pd.to_datetime(base.Game_Date,utc=True,errors='coerce')
    if base.Game_Date.isna().any():raise RuntimeError('V552_MISSING_GAME_DATE')
    dyn=v4.dynamic_features(base,gain=.13,retain=.65)
    combo=pd.concat([base.reset_index(drop=True),dyn],axis=1)
    idx_hist=combo.Season.between(2013,2025) & combo.Actual_Margin.notna() & combo.Actual_Total.notna()
    idx_current=combo.Season.eq(2026) & combo.Game_Date.gt(utc)
    tr=combo.loc[idx_hist].copy();pred=combo.loc[idx_current].copy()
    if len(tr)<8000 or not len(pred):raise RuntimeError('V552_V4_FIT_OR_FUTURE_INSUFFICIENT')
    state=['V4_'+x for x in ('power_diff','off_diff','def_diff','expected_margin','expected_team_points',
        'expected_opp_points','power_uncertainty','prior_games_min','rest_diff','prior_margin_recent_diff',
        'prior_margin_season_diff','site_adv')]
    chosen=v4._cfbd_cols(tr,tr)[:65]
    if len(chosen)<10:raise RuntimeError('V552_V4_CFBD_CHAMPION_FEATURES_MISSING')
    fields=state+chosen
    coverage=float(pred[chosen].apply(pd.to_numeric,errors='coerce').notna().mean(axis=1).median())
    if coverage<min_current_feature_coverage:
        raise RuntimeError(f'V552_2026_V4_FEATURE_COVERAGE_HOLD:{coverage:.3f}')
    projections=v4._matrix_fit_predict(tr,pred,fields,v4.nums(tr,'Actual_Margin'),kind='hgb')
    if not np.isfinite(projections).all():raise RuntimeError('V552_V4_CHAMPION_PREDICTION_NONFINITE')
    # Probability map uses real V4 OOF residuals; NOT a certified cover
    # calibration (normal CDF is a research approximation until validated).
    if oof is None or oof.empty or not {'Season','Actual_Margin','NONLINEAR_HGB'}.issubset(oof):
        raise RuntimeError('V552_V4_OOF_CALIBRATION_INPUT_MISSING')
    fold=oof.loc[pd.to_numeric(oof.Season,errors='coerce').between(2021,2025)].copy()
    residual=pd.to_numeric(fold.Actual_Margin,errors='coerce')-pd.to_numeric(fold.NONLINEAR_HGB,errors='coerce')
    residual=residual.replace([np.inf,-np.inf],np.nan).dropna()
    if len(residual)<1000:raise RuntimeError('V552_RESIDUAL_CALIBRATION_SAMPLE_TOO_SMALL')
    sigma=float(np.sqrt(np.mean(np.square(residual))))
    if not 5<=sigma<=35:raise RuntimeError('V552_RESIDUAL_SCALE_IMPLAUSIBLE')
    out=pred[['Season','Source_Game_ID','Game_Date','Team','Opponent']].copy()
    out=out.rename(columns={'Source_Game_ID':'Game_ID','Game_Date':'Game_Start','Team':'Home','Opponent':'Away'})
    out['CORE_Projected_Home_Margin']=projections
    out['CORE_Residual_SD']=sigma
    out['CORE_Model']=CHAMPION
    out['CORE_Feature_Coverage']=pred[chosen].apply(pd.to_numeric,errors='coerce').notna().mean(axis=1).to_numpy()
    out['CORE_Score_Status']='V552_CFBD_2026_LIVE_SCORED__RESEARCH_PROBABILITY_CALIBRATION'
    audit={'core_model':CHAMPION,'fit_seasons':'2013-2025','training_games':len(tr),
           'current_games':len(out),'feature_count':len(fields),'cfbd_feature_coverage_median':coverage,
           'oof_residual_n':len(residual),'oof_residual_sd':sigma,
           'trained_on_2026_scores':False,'core_prediction_source':'V4_NONLINEAR_HGB_REFIT_CFBD_ONLY',
           'probability_calibration':'RETROSPECTIVE_NORMAL_RESIDUAL_APPROXIMATION',
           'feature_source_vintage_certified':False}
    return out.reset_index(drop=True),audit


def _american_decimal(odds):
    o=_num(odds)
    if not math.isfinite(o) or (abs(o)<100):return float('nan')
    return 1+100/abs(o) if o<0 else 1+o/100


def _cover_probability(points,sigma):
    if not math.isfinite(points) or not (math.isfinite(sigma) and sigma>0):return float('nan')
    return 0.5*(1+math.erf(points/(sigma*math.sqrt(2))))


def prepare_market_quotes(raw,*,now=None):
    """Independent price candidates; do not call the Production V1 scorer/selector."""
    utc=_now(now)
    if not isinstance(raw,pd.DataFrame) or raw.empty:return pd.DataFrame(columns=[*QUOTE_COLUMNS,'_quote_home_key','_quote_away_key'])
    # Shared pure quote/identity normalization is allowed, but not the
    # Production V1 prediction, ranking or Bet Authority functions.
    from ncaaf_production_v1 import prepare_current_market_rows
    z=prepare_current_market_rows(raw,now=utc)
    if z.empty:return pd.DataFrame(columns=[*QUOTE_COLUMNS,'_quote_home_key','_quote_away_key'])
    missing=set(QUOTE_COLUMNS)-set(z)
    if missing:raise RuntimeError('V552_MARKET_QUOTE_COLUMNS_MISSING '+str(sorted(missing)))
    z=z[z.Market.eq('spreads')].copy()
    z['Snapshot_Timestamp']=pd.to_datetime(z.Snapshot_Timestamp,utc=True,errors='coerce')
    z['Game_Start']=pd.to_datetime(z.Game_Start,utc=True,errors='coerce')
    z['Value']=pd.to_numeric(z.Value,errors='coerce')
    z['Odds_Price']=pd.to_numeric(z.Odds_Price,errors='coerce')
    z=z[z.Snapshot_Timestamp.notna() & z.Snapshot_Timestamp.le(utc) &
        z.Game_Start.gt(utc) & z.Value.notna() & z.Odds_Price.notna() & z._prod_game_id.notna()].copy()
    z['_quote_home_key']=z.Home_Team_Norm.map(_school);z['_quote_away_key']=z.Away_Team_Norm.map(_school)
    z['_quote_outcome_key']=z.Outcome.map(_school)
    z=z[z._quote_outcome_key.eq(z._quote_home_key) | z._quote_outcome_key.eq(z._quote_away_key)].copy()
    z=z[z._quote_home_key.ne('') & z._quote_away_key.ne('') & z._quote_home_key.ne(z._quote_away_key)].copy()
    # Only latest quote per exact CFBD-independent physical game, side and book.
    z=z.sort_values('Snapshot_Timestamp',kind='stable').drop_duplicates(
        ['_prod_game_id','_quote_outcome_key','Bookmaker'],keep='last')
    return z


def _family_state(row, selected):
    """Only known sided, independently timestamped observations can confirm."""
    state={f:'NOT_EVALUABLE' for f in FAMILIES}
    selected=_school(selected)
    if not selected:return state
    # Present-day Miner v2 votes have named target, but default to shadow if
    # their actual source time isn't independently certified in this frame.
    votes=row.get('NCAAF_RV22_Miner_Votes')
    if isinstance(votes,list):
        directed=set()
        for v in votes:
            if not isinstance(v,dict) or str(v.get('market','')).lower()!='spreads':continue
            target=_school(v.get('target'))
            if target:
                directed.add('CONFIRM' if target==selected else 'OPPOSE')
        if len(directed)>1:state['SYSTEM_MINER']='CONFLICT'
        elif directed:state['SYSTEM_MINER']='MINER_TRIGGER_UNVERIFIED_TIME'
    names={
      'BIG_AL':('BigAl_Active_Text','BigAl_Trigger_Text'),
      'PATHI':('Pathi_Active_Text','Pathi_System_Summary'),
      'STAT':('NCAAF_Prod_Stat_Selector_Active',),
      'PREDICTION_TRACKER':('NCAAF_PT_Meta_Status',),
    }
    for family,fields in names.items():
        if any(str(row.get(c,'')).strip() not in ('','nan','None','—','0','False','NOT_AVAILABLE') for c in fields):
            state[family]='TRIGGER_ONLY__NO_VERIFIED_DIRECTION'
    # Strict directional provider contract: bounded source identity, timestamp,
    # and side. Family results must be namespaced for V5.5, never inherited
    # from V1 CORE agreement or from V1's Bet Authority labels.
    for family in FAMILIES:
        col='V55_'+family+'_Target';tcol='V55_'+family+'_Observed_At'
        scol='V55_'+family+'_Source_Name'
        if not {col,tcol,scol}.issubset(row.index if isinstance(row,pd.Series) else row):continue
        target=_school(row.get(col));dt=pd.to_datetime(row.get(tcol),utc=True,errors='coerce')
        quote=pd.to_datetime(row.get('Snapshot_Timestamp'),utc=True,errors='coerce')
        kick=pd.to_datetime(row.get('Game_Start'),utc=True,errors='coerce')
        source=str(row.get(scol) or '').strip()
        if (source and source.lower() not in ('nan','none','unknown','unverified')
            and target in {_school(row.get('Home_Team_Norm')),_school(row.get('Away_Team_Norm'))}
            and pd.notna(dt) and pd.notna(quote) and pd.notna(kick)
            and dt<=quote and dt<kick):
            state[family]='CONFIRM' if target==selected else 'OPPOSE'
    return state


def market_and_brains(forecasts,quotes,*,now=None,executable_books=None,max_quote_minutes=240):
    """Every forecast receives a row; no Production V1 candidate prefilter."""
    utc=_now(now)
    if not isinstance(forecasts,pd.DataFrame) or forecasts.empty:
        raise RuntimeError('V552_INDEPENDENT_CORE_FORECASTS_EMPTY')
    needed={'Season','Game_ID','Game_Start','Home','Away','CORE_Projected_Home_Margin','CORE_Residual_SD'}
    if needed-set(forecasts):raise RuntimeError('V552_FORECAST_FIELDS_MISSING')
    if forecasts.Game_ID.duplicated().any():raise RuntimeError('V552_DUPLICATE_CFBD_FORECAST_GAME_ID')
    books={str(b).lower().strip() for b in (executable_books or []) if str(b).strip()}
    entries=[];unmatched=0
    for _,row in forecasts.iterrows():
        kick=pd.to_datetime(row.Game_Start,utc=True,errors='coerce')
        home=_school(row.Home);away=_school(row.Away)
        if pd.isna(kick) or kick<=utc or not home or not away or home==away:
            raise RuntimeError('V552_FORECAST_GAME_IDENTITY_OR_TIME_INVALID')
        relevant=quotes.loc[(quotes._quote_home_key.eq(home) & quotes._quote_away_key.eq(away))].copy() if len(quotes) else pd.DataFrame()
        if len(relevant):
            relevant=relevant.loc[(pd.to_datetime(relevant.Game_Start,utc=True)-kick).abs().le(pd.Timedelta(hours=12))].copy()
            if relevant._prod_game_id.nunique()>1:
                raise RuntimeError('V552_MARKET_MULTIPLE_GAME_IDENTITIES_FOR_CFBD_ID '+str(row.Game_ID))
        candidates=[]
        for _,q in relevant.iterrows():
            snap=pd.to_datetime(q.Snapshot_Timestamp,utc=True,errors='coerce')
            age=(utc-snap).total_seconds()/60
            side=_school(q.Outcome)
            if side not in (home,away) or age<0 or age>max_quote_minutes or snap>=kick:continue
            factor=1 if side==home else -1
            line=_num(q.Value);odds=_num(q.Odds_Price)
            proj=float(row.CORE_Projected_Home_Margin)*factor
            pts=proj+line;p=_cover_probability(pts,float(row.CORE_Residual_SD))
            decimal=_american_decimal(odds)
            if not math.isfinite(decimal):continue
            ev=p*(decimal-1)-(1-p)
            available=bool(books) and str(q.Bookmaker).lower().strip() in books
            candidates.append((bool(available),float(ev),float(p),-age,q,pts,p,ev,side,age))
        if candidates:
            # Preference is executable price first, then EV, never V1's pick.
            best=max(candidates,key=lambda x:x[:4])
            _,_,_,_,q,pts,p,ev,side,age=best
            state=_family_state(q,side)
            effective=bool(best[0]) and ev>0 and p>0 and len(candidates)>0
            # True as-of Market Rich: only same-side original opening quote
            # with a separately recorded observed-at timestamp can be used.
            open_line=_num(q.get('Open_Spread_Selected_Side'))
            open_time=pd.to_datetime(q.get('Open_Snapshot_At'),utc=True,errors='coerce')
            valid_open=(math.isfinite(open_line) and pd.notna(open_time)
                        and open_time<=pd.to_datetime(q.Snapshot_Timestamp,utc=True)
                        and open_time<kick)
            move=float(_num(q.Value)-open_line) if valid_open else float('nan')
            from ncaaf_v55_combined_brain import _crossed_keys
            key_cross=_crossed_keys(open_line,_num(q.Value)) if valid_open else None
            if valid_open and key_cross and state['PATHI']=='NOT_EVALUABLE':
                state['PATHI']='TIMED_KEY_MOVEMENT_TRIGGER__NOT_VALIDATED_VOTE'
            agrees=[k for k,v in state.items() if v=='CONFIRM'];opposes=[k for k,v in state.items() if v in ('OPPOSE','CONFLICT')]
            priority=('REVIEW_OPPOSITION' if effective and opposes else
                      'MULTIFAMILY_CANDIDATE' if effective and len(agrees)>=2 else
                      'SINGLE_FAMILY_CANDIDATE' if effective and len(agrees)==1 else
                      'CORE_PRICE_CANDIDATE' if effective else 'NO_POSITIVE_EXECUTABLE_EDGE')
            r={'Selected_Team':str(q.Outcome),'Book':str(q.Bookmaker),'Current_Spread':_num(q.Value),
               'Odds_American':_num(q.Odds_Price),'CORE_Cover_Probability':p,
               'CORE_Edge':p-1/_american_decimal(q.Odds_Price),'CORE_EV':ev,
               'CORE_Project_Cover_Points':pts,'Quote_Age_Min':age,
               'Snapshot_UTC':pd.to_datetime(q.Snapshot_Timestamp,utc=True).isoformat(),
               'Quote_State':'CURRENT_VERIFIED_QUOTE' if best[0] else 'NON_EXECUTABLE_BOOK',
               'V55_Research_Priority':priority, 'V55_Bet_Decision':'NO_BET__RESEARCH_SHADOW',
               'Confirm_Count':len(agrees),'Oppose_Count':len(opposes),
               'Confirm_Families':' | '.join(agrees),'Oppose_Families':' | '.join(opposes),
               'Market_Rich_Sharp_Move':str(q.get('Sharp_Move_Signal') or ''),
               'Market_Rich_Limit':_num(q.get('Limit')),
               'Market_Rich_Book_Count':_num(q.get('Unique_Sharp_Books')),
               'Market_Rich_Steam_Flag':str(q.get('Late_Game_Steam_Flag') or ''),
               'Market_Rich_Sharp_Bet_Score':_num(q.get('SharpBetScore')),
               'Market_Rich_Open_Line':open_line if valid_open else np.nan,
               'Market_Rich_Open_Observed_At':open_time.isoformat() if valid_open else '',
               'Market_Rich_Move_Points':move,'Market_Rich_Key_Crossings':key_cross,
               'Market_Rich_Source_State':'TIMED_SELECTED_SIDE_OPEN' if valid_open else 'CURRENT_SNAPSHOT__OPEN_VINTAGE_UNVERIFIED',**state}
        else:
            unmatched+=1
            r={'Selected_Team':'','Book':'','Current_Spread':np.nan,'Odds_American':np.nan,
               'CORE_Cover_Probability':np.nan,'CORE_Edge':np.nan,'CORE_EV':np.nan,
               'CORE_Project_Cover_Points':np.nan,'Quote_Age_Min':np.nan,'Snapshot_UTC':'',
               'Quote_State':'NO_VERIFIED_CURRENT_CFBD_MATCHED_QUOTE',
               'V55_Research_Priority':'CORE_FORECAST_ONLY__NO_QUOTE',
               'V55_Bet_Decision':'NO_BET__RESEARCH_SHADOW','Confirm_Count':0,'Oppose_Count':0,
               'Confirm_Families':'','Oppose_Families':'',
               'Market_Rich_Source_State':'NO_QUOTE__NO_EXECUTION',
               'Market_Rich_Open_Line':np.nan,'Market_Rich_Open_Observed_At':'',
               'Market_Rich_Move_Points':np.nan,'Market_Rich_Key_Crossings':np.nan,
               **{f:'NOT_EVALUABLE' for f in FAMILIES}}
        entries.append({'Game_ID':str(row.Game_ID),'Season':int(row.Season),
                        'Game_Start_UTC':kick.isoformat(),'Matchup':f'{row.Away} @ {row.Home}',
                        'Home':str(row.Home),'Away':str(row.Away),
                        'CORE_Model':CHAMPION,'CORE_Projected_Home_Margin':float(row.CORE_Projected_Home_Margin),
                        'CORE_Residual_SD':float(row.CORE_Residual_SD),'CORE_Feature_Coverage':float(row.CORE_Feature_Coverage),
                        'CORE_Provenance':'CFBD_RESEARCH_V4_NONLINEAR_HGB_2026_SCORED',
                        **r,'V55_Bet_Authority':0})
    d=pd.DataFrame(entries).sort_values(['Game_Start_UTC','Matchup'],kind='stable').reset_index(drop=True)
    if d.Game_ID.duplicated().any():raise RuntimeError('V552_DUPLICATE_WEEKLY_GAME')
    audit={'upcoming_core_games':len(d),'no_market_match':unmatched,
           'priced_games':len(d)-unmatched,
           'multi_family':int(d.V55_Research_Priority.eq('MULTIFAMILY_CANDIDATE').sum()),
           'quote_only_refreshable':True,'production_v1_probabilities_used':False,
           'v55_bet_authority':0}
    return d,audit


def publish_weekly(forecasts,board,audit,*,storage_client,bucket_name='sharp-models',log_func=print):
    bucket=storage_client.bucket(bucket_name)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix=f'{ROOT}/runs/{run_id}'
    report={**audit,'version':VERSION,'run_id':run_id,'run_prefix':prefix,
      'core_source':'V55_INDEPENDENT_CFBD_NONLINEAR_HGB','core_model':CHAMPION,
      'production_v1_probabilities_used':False,'production_mutation':False,
      'v55_bet_authority':0,'spread_games':len(board),
      'core_forecast_games':len(forecasts),
      'snapshot_utc':datetime.now(timezone.utc).isoformat(),
      'status':'COMPLETE_V55_INDEPENDENT_WEEKLY_SHADOW'}
    for name,frame in [('weekly_board.csv',board),('core_forecasts.csv',forecasts)]:
        txt=frame.to_csv(index=False)
        report[name+'_sha256']=hashlib.sha256(txt.encode()).hexdigest()
        bucket.blob(prefix+'/'+name).upload_from_string(txt,content_type='text/csv',if_generation_match=0)
    bucket.blob(prefix+'/report.json').upload_from_string(json.dumps(report,default=str,allow_nan=False,indent=2,sort_keys=True),content_type='application/json',if_generation_match=0)
    pointer={'status':report['status'],'run_prefix':prefix,'run_id':run_id,
             'report_sha256':hashlib.sha256(json.dumps(report,default=str,allow_nan=False,indent=2,sort_keys=True).encode()).hexdigest()}
    bucket.blob(ROOT+'/latest.json').upload_from_string(json.dumps(pointer,sort_keys=True),content_type='application/json')
    log_func('[NCAAF-V552-INDEPENDENT-WEEKLY] '+json.dumps({k:report[k] for k in ('status','spread_games','priced_games','core_model','production_v1_probabilities_used') if k in report},sort_keys=True))
    return report


def read_weekly_latest(storage_client,bucket_name='sharp-models'):
    bucket=storage_client.bucket(bucket_name)
    pointer=json.loads(bucket.blob(ROOT+'/latest.json').download_as_text())
    prefix=pointer.get('run_prefix','')
    if not re.fullmatch(re.escape(ROOT)+r'/runs/[0-9]{8}T[0-9]{12}Z',prefix):
        raise RuntimeError('V552_INVALID_LATEST_POINTER')
    report_data=bucket.blob(prefix+'/report.json').download_as_text()
    if hashlib.sha256(report_data.encode()).hexdigest()!=pointer.get('report_sha256'):
        raise RuntimeError('V552_REPORT_DIGEST_MISMATCH')
    report=json.loads(report_data)
    if report.get('status')!='COMPLETE_V55_INDEPENDENT_WEEKLY_SHADOW' or report.get('production_v1_probabilities_used') is not False or report.get('v55_bet_authority')!=0:
        raise RuntimeError('V552_WRONG_OR_LEGACY_WEEKLY_SNAPSHOT')
    csv=bucket.blob(prefix+'/weekly_board.csv').download_as_text()
    if hashlib.sha256(csv.encode()).hexdigest()!=report.get('weekly_board.csv_sha256'):
        raise RuntimeError('V552_WEEKLY_BOARD_DIGEST_MISMATCH')
    frame=pd.read_csv(StringIO(csv),dtype={'Game_ID':'string'})
    if len(frame)!=int(report.get('spread_games',-1)) or frame.Game_ID.duplicated().any():
        raise RuntimeError('V552_WEEKLY_CARDINALITY_MISMATCH')
    return frame,report


def run_weekly(*,storage_client,bq_client,bucket_name='sharp-models',now=None,log_func=print,
               quote_frame=None,history_frame=None,current_frame=None,score_frame=None,
               oof_frame=None):
    """Independent weekly challenger job. No Production V1 model calls."""
    import ncaaf_v54_cfbd_truth as truth
    utc=_now(now)
    if bq_client is None:raise RuntimeError('V552_BIGQUERY_CLIENT_REQUIRED')
    if history_frame is None:
        # CFBD source/score firewall is independently verified; never use the
        # old sportsbook history to manufacture football game outcomes.
        import os
        previous=os.environ.get('NCAAF_CFBD_ARCHIVE_AUTOREPAIR')
        os.environ['NCAAF_CFBD_ARCHIVE_AUTOREPAIR']='0'
        try:history_frame,_,historical_audit=truth.load_from_bigquery(bq_client,log_func=log_func)
        finally:
            if previous is None:os.environ.pop('NCAAF_CFBD_ARCHIVE_AUTOREPAIR',None)
            else:os.environ['NCAAF_CFBD_ARCHIVE_AUTOREPAIR']=previous
    else:historical_audit={'status':'INJECTED_LOCAL_TEST'}
    if current_frame is None:
        current_frame=truth.get_df(bq_client,f'SELECT * FROM `{FUTURE_SOURCE}` WHERE Season = 2026')
    _,cur,feature_audit=prepare_2026_feature_games(current_frame,now=utc)
    if score_frame is None:
        source=truth.get_df(bq_client,f"SELECT * FROM `{SCORES_SOURCE}` WHERE Season = 2026 AND ENDS_WITH(Endpoint, '/games')")
        if source.empty:score_frame=pd.DataFrame()
        else:score_frame,_=truth.parse_cfbd_archive_rows(source)
    cur=attach_2026_past_scores(cur,score_frame,now=utc)
    if oof_frame is None:
        bucket=storage_client.bucket(bucket_name)
        oof_frame=pd.read_csv(StringIO(bucket.blob('research/ncaaf/margin_tournament_v4/game_level_predictions.csv').download_as_text()))
    forecast,core_audit=fit_champion_and_score(history_frame,cur,oof=oof_frame,now=utc)
    if quote_frame is None:
        import utils
        quote_frame=utils.read_recent_sharp_moves(hours=240,table=getattr(utils,'DEFAULT_MOVES_VIEW','sharp_data.moves_with_features_merged'),
                                                   pregame_only=True,sport='NCAAF',use_bq_storage=False)
    quotes=prepare_market_quotes(quote_frame,now=utc)
    # Shared read-only Miner evaluator is permitted. It supplies independent
    # rule triggers, never a probability, price decision or production action.
    if len(quotes):
        try:
            import ncaaf_production_v1 as prod
            quotes=prod._attach_live_research_if_missing(quotes,storage_client=storage_client,bucket_name=bucket_name)
        except Exception as exc:
            log_func('[NCAAF-V552-MINER-RESEARCH-HOLD] '+type(exc).__name__+':'+str(exc)[:200])
    import utils
    board,market_audit=market_and_brains(forecast,quotes,now=utc,executable_books=getattr(utils,'REC_BOOKS',None))
    report=publish_weekly(forecast,board,{**feature_audit,**core_audit,**market_audit,
          'historical_source':HISTORY_SOURCE,'modern_feature_source':FUTURE_SOURCE,
          'modern_scores_used_for_training':False,'production_modified':False,
          'quote_source':'UTILS_MOVE_MASTER_CURRENT',
          'historical_market_vintage_certified':False},storage_client=storage_client,
          bucket_name=bucket_name,log_func=log_func)
    return report


def refresh_market_only(*,storage_client,bucket_name='sharp-models',now=None,
                        quote_frame=None,log_func=print):
    """Reprice latest independent V5.5 forecasts without refitting CORE.

    Runs from its own Cloud Run training dropdown or can be scheduled externally.
    It never calls Production V1 scoring or uses V1 selected-side picks.
    """
    import utils
    bucket=storage_client.bucket(bucket_name)
    pointer=json.loads(bucket.blob(ROOT+'/latest.json').download_as_text())
    prefix=pointer.get('run_prefix','')
    if not re.fullmatch(re.escape(ROOT)+r'/runs/[0-9]{8}T[0-9]{12}Z',prefix):
        raise RuntimeError('V552_INVALID_INDEPENDENT_FORECAST_POINTER')
    report_txt=bucket.blob(prefix+'/report.json').download_as_text()
    if hashlib.sha256(report_txt.encode()).hexdigest()!=pointer.get('report_sha256'):
        raise RuntimeError('V552_INDEPENDENT_FORECAST_REPORT_TAMPERED')
    report=json.loads(report_txt)
    if report.get('production_v1_probabilities_used') is not False or report.get('v55_bet_authority')!=0:
        raise RuntimeError('V552_CANNOT_REFRESH_V1_OVERLAY')
    csv=bucket.blob(prefix+'/core_forecasts.csv').download_as_text()
    if hashlib.sha256(csv.encode()).hexdigest()!=report.get('core_forecasts.csv_sha256'):
        raise RuntimeError('V552_INDEPENDENT_FORECAST_DIGEST_MISMATCH')
    fc=pd.read_csv(StringIO(csv),dtype={'Game_ID':'string'})
    if fc.empty or fc.Game_ID.duplicated().any():
        raise RuntimeError('V552_EMPTY_OR_DUPLICATE_FROZEN_WEEKLY_FORECASTS')
    # Only score games still upcoming. Existing forecast is never overwritten.
    utc=_now(now)
    fc=fc.loc[pd.to_datetime(fc.Game_Start,utc=True,errors='coerce').gt(utc)].copy()
    if fc.empty:raise RuntimeError('V552_NO_UNPLAYED_FORECASTS__RUN_NEXT_WEEKLY')
    if quote_frame is None:
        quote_frame=utils.read_recent_sharp_moves(hours=240,
            table=getattr(utils,'DEFAULT_MOVES_VIEW','sharp_data.moves_with_features_merged'),
            pregame_only=True,sport='NCAAF',use_bq_storage=False)
    quotes=prepare_market_quotes(quote_frame,now=utc)
    market_frame,audit=market_and_brains(fc,quotes,now=utc,executable_books=getattr(utils,'REC_BOOKS',None))
    return publish_weekly(fc,market_frame,{**audit,
        'historical_run':report.get('run_prefix'),
        'core_model':CHAMPION,'core_refit':False,'model_features_reused':True,
        'modern_feature_source':report.get('modern_feature_source'),
        'probability_calibration':report.get('probability_calibration'),
        'historical_market_vintage_certified':False},
        storage_client=storage_client,bucket_name=bucket_name,log_func=log_func)
