"""NCAAF V5.5: independently sourced CORE + expert systems + market-rich research.

Read-only analysis of a *completed* audited V5.4.9 run. Does not refit CORE,
change Production V1, or infer live betting authority from retrospective results.
The latest GCS V4 per-game ledger is verified against the V5.4 source contract.

Live scoring is a separate, explicit adapter. It consumes immutable/frozen CORE
probability and a timestamped sportsbook quote. It never invents missing
STAT/expert votes, market prices, or an executable historical ROI.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import hashlib
import json
import math
import os
import re
from typing import Mapping

import numpy as np
import pandas as pd

VERSION = 'NCAAF_V55_COMBINED_BRAIN_MARKET_RICH_20261021_V1'
ROOT = 'research/ncaaf/v55_combined_brain'
SOURCE_RUN = 'research/ncaaf/v54_full_research/runs/20261021T182810585110Z'
V4_FORECAST = 'research/ncaaf/margin_tournament_v4/game_level_predictions.csv'
V4_REPORT = 'research/ncaaf/margin_tournament_v4/current_report.json'
MARKET_TABLE = 'sharplogger.sharp_data.ncaaf_historical_game_side_raw'
PRIMARY_MODEL = 'NONLINEAR_HGB'
FAMILIES = ('STAT', 'BIG_AL', 'PATHI', 'SYSTEM_MINER', 'STATE_BOUNCE_BACK', 'PREDICTION_TRACKER')
KEY = ['Season', 'Game_ID']
MODEL_KEY = ['Season', 'Game_ID', 'Model']
DIS = 'development_2021_2023'
EXAM = 'examined_2024_2025'


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _num(s):
    return pd.to_numeric(s, errors='coerce')


def _key_frame(df, source, *, unique=None, allow_live=False):
    """Reuse V5.4.9's exact-key guard rather than coercing IDs through floats."""
    from ncaaf_v54_full_research import _canonical_cfbd_research_join_keys
    out = _canonical_cfbd_research_join_keys(df, source=source)
    if not out.empty and (out.Season.gt(2100 if allow_live else 2025) | out.Season.lt(2013)).any():
        raise RuntimeError('V55_2026_OUTCOMES_OR_OUT_OF_RANGE source='+source)
    if unique and out.duplicated(unique).any():
        raise RuntimeError('V55_DUPLICATE_KEY source='+source)
    return out


def _team_token(s):
    # Only exact, canonical, name-safe identity; ambiguous teams are never
    # resolved by their results or by market source guesses.
    from ncaaf_v54_cfbd_truth import cfbd_school_key
    return cfbd_school_key(s)


def _read_csv_blob(bucket, key):
    return pd.read_csv(StringIO(bucket.blob(key).download_as_text()), low_memory=False,
                       dtype={'Game_ID':'string', 'Source_Game_ID':'string'})


def _read_json_blob(bucket, key):
    return json.loads(bucket.blob(key).download_as_text())


def _source_check(report, *, expected_run=None):
    if report.get('status') != 'COMPLETE_FULL_RESEARCH__SHADOW_AUTHORITY__VINTAGE_HOLD':
        raise RuntimeError('V55_PARENT_RESEARCH_NOT_COMPLETE')
    if report.get('historical_football_source') != 'CFBD_ONLY' or report.get('bet_authority') != 0:
        raise RuntimeError('V55_PARENT_CFBD_OR_PRODUCTION_FIREWALL_FAILED')
    if report.get('source_vintage_certified') is not False:
        raise RuntimeError('V55_UNEXPECTED_PARENT_VINTAGE_CONTRACT')
    if expected_run is not None and str(report.get('run_id')) != str(expected_run):
        raise RuntimeError('V55_PARENT_RUN_ID_MISMATCH')
    components = report.get('research_components') or {}
    v42 = components.get('v42') or {}
    if v42.get('selected_v4_research_winner') != PRIMARY_MODEL:
        raise RuntimeError('V55_NOT_SELECTED_V4_CHAMPION')
    return {'parent_run':report.get('run_id'), 'selected_v4_model':PRIMARY_MODEL,
            'football_source':'CFBD_ONLY', 'source_vintage_certified':False}


def champion_candidates(v4_forecasts, report):
    """Use ACTUAL V4 winner predictions and verified original open-market margins.

    Unlike V5.1's previous A_EFFICIENCY/STATE_HGB ledger, this does not
    substitute an unselected proxy for V4's selected NONLINEAR_HGB.
    """
    _source_check(report)
    d=_key_frame(v4_forecasts, 'V4_SELECTED_MODEL_LEDGER', unique=KEY)
    needed={'Game_Date','Team','Opponent','Actual_Margin',PRIMARY_MODEL,
            'Verified_Open_Margin_Diagnostic'}
    if needed-set(d.columns):
        raise RuntimeError('V55_V4_REQUIRED_COLUMNS_MISSING '+str(sorted(needed-set(d.columns))))
    d=d[d.Season.between(2021,2025)].copy()
    for c in (PRIMARY_MODEL, 'Actual_Margin', 'Verified_Open_Margin_Diagnostic'):
        d[c]=_num(d[c])
    d['Game_Date']=pd.to_datetime(d.Game_Date,utc=True,errors='coerce')
    if d.Game_Date.isna().any(): raise RuntimeError('V55_V4_INVALID_DATES')
    d['CORE_Margin']=d[PRIMARY_MODEL]
    d['Market_Open_Margin']=d['Verified_Open_Margin_Diagnostic']
    d['CORE_Edge_Home']=d.CORE_Margin-d.Market_Open_Margin
    d['Abs_Edge']=d.CORE_Edge_Home.abs()
    d['Market_Ready']=np.isfinite(d.Market_Open_Margin) & np.isfinite(d.CORE_Margin)
    d['Side']=np.where(d.CORE_Edge_Home>=0, 'TEAM', 'OPPONENT')
    d['Model']=PRIMARY_MODEL
    d['Period']=np.where(d.Season.isin([2021,2022,2023]), DIS, EXAM)
    d['Selection_Type']=np.select([d.Market_Open_Margin.mul(-1).lt(0) & d.Side.eq('TEAM'),
       d.Market_Open_Margin.mul(-1).gt(0) & d.Side.eq('TEAM'),
       d.Market_Open_Margin.mul(-1).gt(0) & d.Side.eq('OPPONENT'),
       d.Market_Open_Margin.mul(-1).lt(0) & d.Side.eq('OPPONENT')],
       ['FAVORITE','UNDERDOG','FAVORITE','UNDERDOG'], default='PICKEM')
    home_cover=d.Actual_Margin-d.Market_Open_Margin
    signed=np.where(d.Side.eq('TEAM'),home_cover,-home_cover)
    d['Result']=np.select([signed>1e-7,signed< -1e-7],['WIN','LOSS'],default='PUSH')
    d.loc[~d.Market_Ready,'Result']='UNPRICED'
    d['Snapshot_At_UTC']=pd.NaT
    d['Market_Rich_Pregame_Certified']=False
    d['CORE_Prediction_Vintage_Certified']=False
    return d[KEY+['Game_Date','Team','Opponent','Model','CORE_Margin','Market_Open_Margin',
                   'CORE_Edge_Home','Abs_Edge','Side','Selection_Type','Period','Result',
                   'Market_Ready','Snapshot_At_UTC','Market_Rich_Pregame_Certified',
                   'CORE_Prediction_Vintage_Certified']].copy()


def historical_market_context(raw):
    """Strict bookmaker price-only context; no current-game football features.

    CFBD source_game_id + Is_Home must be unique after conflict quarantine.
    Historical closing quotes without publication timestamps are descriptive,
    NOT valid as opening-time predictors.
    """
    if raw is None or len(raw)==0:
        return pd.DataFrame(columns=KEY+['Market_Close_Margin','Market_Rich_Move_Home',
                   'Market_Open_Spread_Truth','Market_Rich_Price_Context_Ready']), {'status':'MISSING_MARKET_CONTEXT'}
    must={'Source_Name','Season','Source_Game_ID','Is_Home','Consensus_Open_Spread','Consensus_Close_Spread'}
    if must-set(raw):raise RuntimeError('V55_MARKET_SCHEMA_MISSING '+str(sorted(must-set(raw))))
    x=raw[raw.Source_Name.astype(str).str.casefold().eq('collegefootballdata')].copy()
    x=x[pd.to_numeric(x.Is_Home,errors='coerce').eq(1)].copy()
    x=x.rename(columns={'Source_Game_ID':'Game_ID'})
    x=_key_frame(x,'CFBD_BOOKMAKER_PRICE_ONLY')
    for f in ['Consensus_Open_Spread','Consensus_Close_Spread']:
        x[f]=_num(x[f])
    # Multiple distinguishable conflicting quotes must NOT be silently averaged
    # or selected based on their subsequent correctness.
    grouped=x.groupby(KEY,sort=False,dropna=False)
    rows=[];conflicts=0
    for (season,gid),g in grouped:
        opens=g.Consensus_Open_Spread.dropna().round(5).unique()
        closes=g.Consensus_Close_Spread.dropna().round(5).unique()
        if len(opens)>1 or len(closes)>1:
            conflicts+=1; continue
        rows.append({'Season':season,'Game_ID':gid,
             'Market_Open_Spread_Truth':float(opens[0]) if len(opens) else np.nan,
             'Market_Close_Margin':-float(closes[0]) if len(closes) else np.nan})
    y=pd.DataFrame(rows,columns=KEY+['Market_Open_Spread_Truth','Market_Close_Margin'])
    if len(y):
        y['Market_Open_Margin_Truth']=-y.Market_Open_Spread_Truth
        y['Market_Rich_Move_Home']=y.Market_Close_Margin-y.Market_Open_Margin_Truth
        y['Market_Rich_Price_Context_Ready']=np.isfinite(y.Market_Rich_Move_Home)
    else:
        y['Market_Open_Margin_Truth']=pd.Series(dtype=float)
        y['Market_Rich_Move_Home']=pd.Series(dtype=float)
        y['Market_Rich_Price_Context_Ready']=pd.Series(dtype=bool)
    return y,{'status':'PRICE_CONTEXT_ONLY','source':MARKET_TABLE,'rows':len(y),
              'conflicting_game_prices_quarantined':conflicts,
              'paired_open_close':int(y.Market_Rich_Price_Context_Ready.sum()),
              'current_or_historical_snapshot_timing_certified':False,
              'bettable_roi_available':False}


def _crossed_keys(open_spread, close_spread):
    """Key-number traversal (3,7,10,14), price context, not a system vote."""
    if not (np.isfinite(open_spread) and np.isfinite(close_spread)):
        return None
    lo,hi=sorted((float(open_spread),float(close_spread)))
    if hi==lo:return 0
    return sum(lo<k<=hi for k in (-14.,-10.,-7.,-3.,3.,7.,10.,14.))


def _market_rich_categories(a):
    """Keep directional market support separate from the PRICE paid.

    A line moving toward the CORE side can confirm its direction but makes
    the bet less attractive at the new quote. No historical closing quote
    qualifies as a pregame input without original publication timestamps.
    """
    move=_num(a['Market_Rich_Move_With_CORE'])
    ready=a['Market_Rich_Price_Context_Ready'].fillna(False).astype(bool)
    a['Market_Rich_Directional_Context']=np.select([ready & move.gt(.25),
        ready & move.lt(-.25),ready], ['MOVE_TOWARD_CORE_SIDE',
        'MOVE_AWAY_FROM_CORE_SIDE','FLAT_MOVE'],default='UNKNOWN_QUOTE_VINTAGE')
    open_spread=np.where(a.Side.eq('TEAM'),-a.Market_Open_Margin,a.Market_Open_Margin)
    # Sign-selected spread: if CORE side is home use negative MarketHomeMargin,
    # otherwise use positive MarketHomeMargin. Closing side symmetric.
    close_home=a.Market_Open_Margin+a.Market_Rich_Move_Home
    close_spread=np.where(a.Side.eq('TEAM'),-close_home,close_home)
    a['Market_Rich_Closing_Spread_For_Core']=np.where(ready,close_spread,np.nan)
    a['Market_Rich_Closing_Edge_Points']=np.where(ready,
         np.where(a.Side.eq('TEAM'),a.CORE_Margin-close_home,close_home-a.CORE_Margin),np.nan)
    a['Market_Rich_Open_Edge_Points']=a.Abs_Edge
    a['Market_Rich_Edge_Delta_Close_Minus_Open']=a.Market_Rich_Closing_Edge_Points-a.Abs_Edge
    a['Market_Rich_Key_Traversals']=pd.Series([
        _crossed_keys(start,end) if enabled else None
        for start,end,enabled in zip(open_spread,close_spread,ready)],index=a.index,dtype='Float64')
    a['Market_Rich_Quote_Vintage_Certified']=False
    return a


def _v51_orientation_audit(core, shadow):
    """Cross-check home/away identity against the earlier source-bound CORE.

    2022 is not in the V5.1 shadow ledger and is NOT silently removed from
    the selected V4 benchmark. Compare only the common physical games.
    """
    if shadow is None or len(shadow)==0:
        return {'status':'NO_V51_ORIENTATION_ANCHOR','common_games':0}
    legacy=_key_frame(shadow,'V51_PROXY_SOURCE',unique=MODEL_KEY)
    keys=legacy.drop_duplicates(KEY)
    x=core.merge(keys[KEY+['Team','Opponent','Game_Date']],on=KEY,how='inner',
                 validate='one_to_one',suffixes=('_V4','_V51'))
    if len(x)==0:raise RuntimeError('V55_NO_V4_V51_SHARED_CFBD_GAMES')
    tm=(x.Team_V4.map(_team_token)==x.Team_V51.map(_team_token))
    op=(x.Opponent_V4.map(_team_token)==x.Opponent_V51.map(_team_token))
    if not (tm & op).all():
        raise RuntimeError('V55_SELECTED_V4_TEAM_ORIENTATION_DIFFERS_FROM_V51')
    date0=pd.to_datetime(x.Game_Date_V4,utc=True,errors='coerce').dt.date
    date1=pd.to_datetime(x.Game_Date_V51,utc=True,errors='coerce').dt.date
    if not date0.eq(date1).all():raise RuntimeError('V55_SELECTED_V4_DATE_DIFFERS_FROM_V51')
    return {'status':'EXACT_HOME_AWAY_DATE_PARITY','common_games':len(x),
            'v51_models_are_comparators_only':True,
            'v4_2022_games_retained_outside_v51':int(core.Season.eq(2022).sum())}


def attach_market_context(champion, market_context):
    a=_key_frame(champion,'V55_CHAMPION_CANDIDATES',unique=MODEL_KEY)
    if market_context is None or market_context.empty:
        a['Market_Rich_Move_Home']=np.nan
        a['Market_Rich_Move_With_CORE']=np.nan
        a['Market_Rich_Price_Context_Ready']=False
        a['Market_Rich_Open_Parity']=False
        return _market_rich_categories(a)
    x=_key_frame(market_context,'CFBD_HISTORICAL_MARKET_ONLY',unique=KEY)
    a=a.merge(x[KEY+['Market_Open_Margin_Truth','Market_Rich_Move_Home',
            'Market_Rich_Price_Context_Ready']],on=KEY,validate='many_to_one',how='left')
    paired=a.Market_Ready & np.isfinite(a.Market_Open_Margin_Truth)
    parity=paired & np.isclose(a.Market_Open_Margin,a.Market_Open_Margin_Truth,atol=0.001)
    if int((paired & ~parity).sum()):
        raise RuntimeError('V55_MARKET_OPEN_SOURCE_DISAGREES_WITH_V4_DIAGNOSTIC')
    a['Market_Rich_Open_Parity']=parity
    a['Market_Rich_Price_Context_Ready']=a.Market_Rich_Price_Context_Ready.fillna(False) & parity
    a['Market_Rich_Move_With_CORE']=np.where(a.Side.eq('TEAM'),a.Market_Rich_Move_Home,-a.Market_Rich_Move_Home)
    a.loc[~a.Market_Rich_Price_Context_Ready,'Market_Rich_Move_With_CORE']=np.nan
    a=a.drop(columns=['Market_Open_Margin_Truth'])
    return _market_rich_categories(a)


def _validate_optional_votes(votes, *, name, kickoff):
    """Require a verifiable per-game timestamp and named source for any optional signal."""
    if votes is None or len(votes)==0:
        return pd.DataFrame(columns=KEY+['Family','System_ID','Side','Evidence_Class']), {'status':'NOT_CONNECTED','records':0}
    needed={'Season','Game_ID','Family','System_ID','Side','Signal_At','Source_Name','Source_Independent'}
    if needed-set(votes):raise RuntimeError('V55_OPTIONAL_EVIDENCE_MISSING_FIELDS '+name)
    d=_key_frame(votes,'V55_'+name)
    if (~d.Family.isin(FAMILIES)).any() or (~d.Side.isin(['TEAM','OPPONENT'])).any():
        raise RuntimeError('V55_INVALID_OPTIONAL_FAMILY_OR_SIDE')
    if not d.Source_Independent.map(lambda x: x is True or (isinstance(x,(bool,np.bool_)) and bool(x))).all():
        raise RuntimeError('V55_OPTIONAL_EVIDENCE_NOT_INDEPENDENT')
    if d.Source_Name.isna().any() or d.Source_Name.astype(str).str.strip().eq('').any():
        raise RuntimeError('V55_OPTIONAL_EVIDENCE_NO_SOURCE')
    d['Signal_At']=pd.to_datetime(d.Signal_At,utc=True,errors='coerce')
    if d.Signal_At.isna().any():raise RuntimeError('V55_OPTIONAL_SIGNAL_TIMESTAMP_INVALID')
    k=_key_frame(kickoff,'V55_KICKOFF_LEDGER',unique=KEY)[KEY+['Game_Date']]
    j=d.merge(k,on=KEY,validate='many_to_one',how='left')
    if j.Game_Date.isna().any() or (j.Signal_At>=pd.to_datetime(j.Game_Date,utc=True)).any():
        raise RuntimeError('V55_OPTIONAL_SIGNAL_NOT_STRICTLY_PREGAME')
    if j.duplicated(KEY+['Family','System_ID','Side']).any():
        raise RuntimeError('V55_DUPLICATE_OPTIONAL_EVIDENCE')
    j['Evidence_Class']='TIMESTAMPED_EXTERNAL__PROVENANCE_METADATA_REVIEW_REQUIRED'
    return j[KEY+['Family','System_ID','Side','Evidence_Class']],{'status':'TIMESTAMP_CHECK_PASS__INDEPENDENCE_CLAIM_NOT_AUDITED','records':len(j)}


def combine_system_evidence(core, reconstructed, optional=None):
    """No model-based system vote multiplication. One relation per family/game.

    Signals on both teams from the same family are a conflict, not two votes.
    Keep all exact system IDs separately for overlap/individual tests.
    """
    a=_key_frame(core,'V55_CORE')
    if a.duplicated(MODEL_KEY).any():raise RuntimeError('V55_DUPLICATE_CORE')
    r=_key_frame(reconstructed,'CFBD_RECONSTRUCTED_SYSTEMS')
    cols=KEY+['Family','System_ID','Side','Evidence_Class']
    if set(cols)-set(r):raise RuntimeError('V55_RECONSTRUCTED_REQUIRED_FIELDS_MISSING')
    if len(r) and (r.Family.isna().any() or ~r.Side.isin(['TEAM','OPPONENT']).all()):
        raise RuntimeError('V55_INVALID_RECONSTRUCTED_DIRECTION')
    if len(r) and not r.Evidence_Class.astype(str).str.contains('RETROSPECTIVE').all():
        raise RuntimeError('V55_RECONSTRUCTED_HISTORICAL_VINTAGE_UNEXPECTED')
    d=pd.concat([r[cols],optional[cols] if optional is not None and len(optional) else r.iloc[0:0][cols]],ignore_index=True)
    d=d.drop_duplicates(KEY+['Family','System_ID','Side'])
    if len(d):
        if not set(zip(d.Season,d.Game_ID)).issubset(set(zip(a.Season,a.Game_ID))):
            # For V5.4 system archive, unmatched games are legitimate because
            # V4 edge pricing covers a strict subset; never force a match.
            d=d.merge(a[KEY].drop_duplicates(),on=KEY,how='inner',validate='many_to_one')
    # A physical game may have two models; assign one relation per model separately.
    z=(a[MODEL_KEY+['Side']].merge(d,on=KEY,how='left',suffixes=('_CORE','_SYSTEM'),validate='one_to_many'))
    z=z[z.Family.notna()].copy()
    if len(z):
        z['Relation']=np.where(z.Side_CORE.eq(z.Side_SYSTEM),'CONFIRM','OPPOSE')
        fam=(z.groupby(MODEL_KEY+['Family'],as_index=False).Relation.agg(
            lambda s:'CONFLICT' if s.nunique()>1 else s.iloc[0]))
        g=fam.pivot(index=MODEL_KEY,columns='Family',values='Relation').reset_index()
        g.columns=[str(x) for x in g.columns]
        joined=a.merge(g,on=MODEL_KEY,how='left',validate='one_to_one')
    else:
        fam=pd.DataFrame(columns=MODEL_KEY+['Family','Relation']);joined=a.copy()
    for name in FAMILIES:
        if name not in joined:joined[name]=pd.NA
        joined[name+'_Relation']=joined[name].fillna('NOT_EVALUABLE')
        joined=joined.drop(columns=name)
    names=[n+'_Relation' for n in FAMILIES]
    joined['Confirmed_Families']=(joined[names]=='CONFIRM').sum(axis=1)
    joined['Opposing_Families']=(joined[names]=='OPPOSE').sum(axis=1)
    joined['Conflicting_Families']=(joined[names]=='CONFLICT').sum(axis=1)
    joined['Observed_Families']=(joined[names]!='NOT_EVALUABLE').sum(axis=1)
    joined['Research_Confluence']=np.select([
       joined.Conflicting_Families.gt(0)|joined.Opposing_Families.gt(0),
       joined.Confirmed_Families.ge(2), joined.Confirmed_Families.eq(1)],
       ['CONFLICT_REVIEW','MULTIFAMILY_CONFIRM','SINGLE_FAMILY_CONFIRM'],default='CORE_ONLY')
    joined['Research_Priority']=np.select([
        joined.Conflicting_Families.gt(0),joined.Opposing_Families.gt(0),
        joined.Confirmed_Families.ge(2),joined.Confirmed_Families.eq(1)],
        ['REVIEW_CONFLICT','REVIEW_OPPOSITION','HIGH_RESEARCH_PRIORITY','RESEARCH_CANDIDATE'],
        default='CORE_ONLY_REVIEW')
    joined.loc[~joined.Market_Ready,'Research_Priority']='UNPRICED_NO_EDGE'
    joined['Bet_Authority']=0
    joined['Bet_Decision']='NO_BET'
    joined['Provisional_Status']='RESEARCH_ONLY__RETROSPECTIVE_INSPECTED'
    return joined,z,fam


def wilson(w,n):
    if not n:return (None,None)
    z=1.95996398454;p=w/n;d=1+z*z/n
    m=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (float(m-h),float(m+h))


def performance(rows,group_cols):
    """Model/game is unit. No multi-threshold or multi-rule duplicate bets."""
    data=rows[rows.Market_Ready & rows.Result.isin(['WIN','LOSS','PUSH'])].copy()
    if data.duplicated(MODEL_KEY).any():raise RuntimeError('V55_PERFORMANCE_DUPLICATED_MODEL_GAME')
    records=[]
    for k,g in data.groupby(group_cols,dropna=False,sort=True):
        k=(k,) if not isinstance(k,tuple) else k
        wins=int(g.Result.eq('WIN').sum());losses=int(g.Result.eq('LOSS').sum());pushes=int(g.Result.eq('PUSH').sum())
        n=wins+losses;lo,hi=wilson(wins,n)
        records.append({**dict(zip(group_cols,k)),'games':len(g),'wins':wins,'losses':losses,'pushes':pushes,
            'decisions':n,'hit_rate':wins/n if n else np.nan,'wilson95_low':lo,'wilson95_high':hi,
            'authority':0,'vintage_certified':False,'after_inspection':True})
    return pd.DataFrame(records)


def system_screening(core,z):
    """Rank actual mechanisms on selected champion, not V5.1 proxy models."""
    if z.empty:return pd.DataFrame()
    # Two opposing signals from same system/game = quarantine. Never group
    # duplicate mechanisms as extra evidence.
    cols=MODEL_KEY+['Family','System_ID']
    conflicts=z.groupby(cols).Relation.nunique()
    bad=conflicts[conflicts>1].index
    z=z.loc[~pd.MultiIndex.from_frame(z[cols]).isin(bad)].copy()
    z=z.drop_duplicates(cols)
    x=z.merge(core[MODEL_KEY+['Period','Result','Market_Ready','Selection_Type']],
              on=MODEL_KEY,validate='many_to_one')
    x=x[x.Market_Ready & x.Result.isin(['WIN','LOSS'])]
    rows=[]
    for (fam,sid,rel,period),g in x.groupby(['Family','System_ID','Relation','Period']):
        w=int(g.Result.eq('WIN').sum());n=len(g);lo,hi=wilson(w,n)
        rows.append({'Family':fam,'System_ID':sid,'Relation':rel,'Period':period,
                     'decisions':n,'wins':w,'losses':n-w,'hit_rate':w/n if n else np.nan,
                     'wilson95_low':lo,'wilson95_high':hi})
    out=pd.DataFrame(rows)
    if out.empty:return out
    # No automatic confidence from 2024-25 inspected data; rankings are
    # exploratory prioritization only, not pre-registered prospective winners.
    piv=out.pivot_table(index=['Family','System_ID','Relation'],columns='Period',
        values=['decisions','hit_rate'],aggfunc='first').reset_index()
    piv.columns=['_'.join([str(k) for k in v if k]) if isinstance(v,tuple) else str(v) for v in piv.columns]
    piv['Promoted_Research_Status']='PROVISIONAL_RESEARCH_ONLY'
    piv['Production_Authority']=0
    return piv.sort_values(['Family','System_ID','Relation']).reset_index(drop=True)


def strategy_tables(combined):
    a=combined[combined.Market_Ready].copy()
    a['Brain_Combination']=a[[f+'_Relation' for f in FAMILIES]].apply(
        lambda row:'|'.join(f'{name}:{row[name+"_Relation"]}' for name in FAMILIES
                            if row[name+'_Relation']!='NOT_EVALUABLE') or 'NONE',axis=1)
    by_combo=performance(a,['Model','Period','Brain_Combination','Selection_Type'])
    by_tier=performance(a,['Model','Period','Research_Confluence','Selection_Type'])
    baseline=performance(a,['Model','Period','Selection_Type'])
    # Retrospective Market Rich split: closing movement is not valid at the opening quote.
    # It is displayed as a research diagnostic, never used for bet authority.
    # Lift is conditional descriptive difference to *all* games in same period,
    # not an independent causal increment or a pristine unseen test.
    if len(by_tier) and len(baseline):
        b=baseline[['Model','Period','Selection_Type','hit_rate']].rename(columns={'hit_rate':'Baseline_Hit_Rate'})
        by_tier=by_tier.merge(b,on=['Model','Period','Selection_Type'],validate='many_to_one')
        by_tier['Observed_PctPt_Above_Period_Baseline']=100*(by_tier.hit_rate-by_tier.Baseline_Hit_Rate)
    market_split=performance(a,['Model','Period','Research_Confluence',
        'Selection_Type','Market_Rich_Directional_Context'])
    return by_combo,by_tier,baseline,market_split


def _moneyline_decimal(american):
    try:x=float(american)
    except (ValueError,TypeError):return None
    if not math.isfinite(x) or (-100<x<100):return None
    return 1+100/abs(x) if x<0 else 1+x/100


def live_research_candidate(frame):
    """Pure adapter for future background market refresh; NEVER issues BET orders.

    Inputs are independently supplied and pregame dated. Missing votes remain
    NOT_EVALUABLE. Real probability / odds / quote are mandatory to compute EV.
    Market Rich can veto stale or mismatched snapshots but cannot rewrite CORE.
    """
    need={'Season','Game_ID','Game_Start','Snapshot_At','Core_Model','Core_Probability',
          'Core_Projected_Margin','Current_Spread','Current_American_Odds',
          'Team','Opponent','CORE_Side'}
    if need-set(frame):raise RuntimeError('V55_LIVE_INPUT_MISSING '+str(sorted(need-set(frame))))
    d=_key_frame(frame,'V55_FUTURE_LIVE_CANDIDATE',unique=KEY,allow_live=True)
    if not set(d.CORE_Side).issubset({'TEAM','OPPONENT'}):raise RuntimeError('V55_LIVE_INVALID_SIDE')
    d['Game_Start']=pd.to_datetime(d.Game_Start,utc=True,errors='coerce')
    d['Snapshot_At']=pd.to_datetime(d.Snapshot_At,utc=True,errors='coerce')
    if d[['Game_Start','Snapshot_At']].isna().any().any() or (d.Snapshot_At>=d.Game_Start).any():
        raise RuntimeError('V55_LIVE_QUOTE_NOT_PREGAME')
    p=_num(d.Core_Probability)
    if p.isna().any() or (~p.between(0,1)).any():raise RuntimeError('V55_LIVE_CORE_PROBABILITY_UNAVAILABLE')
    d['Odds_Decimal']=d.Current_American_Odds.map(_moneyline_decimal)
    s=_num(d.Current_Spread)
    if s.isna().any() or ~np.isfinite(s).all():raise RuntimeError('V55_LIVE_MARKET_SPREAD_UNAVAILABLE')
    d['Current_Spread']=s
    d['Net_EV_Per_Unit']=np.where(d.Odds_Decimal.notna(),p*(d.Odds_Decimal-1)-(1-p),np.nan)
    d['Quoted_Breakeven_Probability']=np.where(d.Odds_Decimal.notna(),1/d.Odds_Decimal,np.nan)
    # Must be the projection oriented TO THE SELECTED SIDE, not necessarily
    # home Team (a home projection is sign-flipped for opponent).
    proj=_num(d.Core_Projected_Margin)
    if proj.isna().any() or ~np.isfinite(proj).all():raise RuntimeError('V55_LIVE_CORE_MARGIN_UNAVAILABLE')
    d['Projected_Cover_Points']=proj+s
    # Live Market Rich extension: explicit opening quote and its timestamp,
    # in the SAME selected-side orientation as Current_Spread. This reads
    # recorded market snapshots; it cannot infer an opening line from prices.
    d['Market_Rich_Open_Observed']=False
    d['Market_Rich_Line_Move_For_Selected']=np.nan
    d['Market_Rich_Key_Cross_Count']=np.nan
    if 'Open_Spread' in d or 'Open_Snapshot_At' in d:
        if 'Open_Spread' not in d or 'Open_Snapshot_At' not in d:
            raise RuntimeError('V55_LIVE_MARKET_RICH_OPEN_QUOTE_TIMESTAMP_PAIR_REQUIRED')
        opened=pd.to_datetime(d.Open_Snapshot_At,utc=True,errors='coerce')
        line0=_num(d.Open_Spread)
        if opened.isna().any() or line0.isna().any() or (~np.isfinite(line0)).any():
            raise RuntimeError('V55_LIVE_MARKET_RICH_OPEN_QUOTE_INVALID')
        if (opened > d.Snapshot_At).any() or (opened >= d.Game_Start).any():
            raise RuntimeError('V55_LIVE_MARKET_RICH_OPEN_NOT_OBSERVED_ASOF')
        d['Market_Rich_Open_Observed']=True
        d['Market_Rich_Line_Move_For_Selected']=d.Current_Spread-line0
        d['Market_Rich_Key_Cross_Count']=[_crossed_keys(x,y) for x,y in zip(line0,s)]
        # A less favorable new spread reduces points versus the open, even if
        # the price/consensus movement signals support for the selected team.
    d['Market_Rich_Book_Count']=pd.to_numeric(d.get('Market_Rich_Book_Count',
        pd.Series(np.nan,index=d.index)),errors='coerce')
    d['Market_Rich_Shadow_Context']='PRICE_SNAPSHOT_ONLY__NOT_INDEPENDENT_MODEL_PROBABILITY'
    d['Research_Quote_Ready']=d.Odds_Decimal.notna()
    d['Research_Quote_Edge_Positive']=d.Net_EV_Per_Unit.gt(0) & d.Research_Quote_Ready
    # Optional independently supplied signed opinions, one per FAMILY.
    # They are never manufactured from retrospective wins or market prices.
    family_cols=[]
    for family in FAMILIES:
        col=family+'_Relation'
        if col not in d:
            d[col]='NOT_EVALUABLE'
            continue
        z=d[col].fillna('NOT_EVALUABLE').astype(str)
        if not set(z).issubset({'NOT_EVALUABLE','CONFIRM','OPPOSE','CONFLICT'}):
            raise RuntimeError('V55_LIVE_UNRECOGNIZED_FAMILY_VOTE '+family)
        present=z.ne('NOT_EVALUABLE')
        if present.any():
            tcol=family+'_Signal_At'; sourcecol=family+'_Source_Name'
            if tcol not in d or sourcecol not in d:
                raise RuntimeError('V55_LIVE_FAMILY_PROVENANCE_MISSING '+family)
            timing=pd.to_datetime(d.loc[present,tcol],utc=True,errors='coerce')
            if timing.isna().any() or (timing > d.loc[present,'Snapshot_At']).any() or (timing >= d.loc[present,'Game_Start']).any():
                raise RuntimeError('V55_LIVE_FAMILY_SIGNAL_NOT_AT_SNAPSHOT '+family)
            src=d.loc[present,sourcecol]
            if src.isna().any() or src.astype(str).str.strip().eq('').any():
                raise RuntimeError('V55_LIVE_FAMILY_SOURCE_MISSING '+family)
            family_cols.append(col)
        d[col]=z
    relation_cols=[family+'_Relation' for family in FAMILIES]
    d['Verified_AsOf_Confirm_Families']=(d[relation_cols]=='CONFIRM').sum(axis=1)
    d['Verified_AsOf_Oppose_Families']=(d[relation_cols]=='OPPOSE').sum(axis=1)
    d['Verified_AsOf_Conflict_Families']=(d[relation_cols]=='CONFLICT').sum(axis=1)
    # A historical validated 2024 mechanism is not automatically a new
    # present-day profitable vote. This is a *research candidate*, not a bet.
    d['Research_Decision']=np.select([
        d.Verified_AsOf_Oppose_Families.gt(0)|d.Verified_AsOf_Conflict_Families.gt(0),
        d.Research_Quote_Edge_Positive & d.Verified_AsOf_Confirm_Families.ge(2),
        d.Research_Quote_Edge_Positive & d.Verified_AsOf_Confirm_Families.eq(1),
        d.Research_Quote_Edge_Positive],
        ['REVIEW_OPPOSING_BRAINS','PROVISIONAL_MULTIFAMILY_RESEARCH_CANDIDATE',
         'PROVISIONAL_SINGLE_BRAIN_RESEARCH_CANDIDATE','CORE_ONLY_RESEARCH_CANDIDATE'],
        default='NO_BET')
    d['Bet_Authority']=0
    d['Bet_Decision']='NO_BET'
    d['Core_Probability_Unchanged']=p
    return d


def analyze(parent_report, v4_forecasts, reconstructed, *, market_raw=None, optional_votes=None,
            v51_shadow=None):
    provenance=_source_check(parent_report)
    champion=champion_candidates(v4_forecasts,parent_report)
    # Source IDs validated on all artifacts, including those that will not
    # match the priced historical season subset.
    systems=_key_frame(reconstructed,'V54_EXPERT_MINER_ARCHIVE')
    if systems.duplicated(KEY+['Family','System_ID','Side']).any():
        raise RuntimeError('V55_DUPLICATE_RECONSTRUCTED_OCCURRENCES')
    m,market_audit=historical_market_context(market_raw)
    champion=attach_market_context(champion,m)
    optional,optional_audit=_validate_optional_votes(optional_votes,name='OPTIONAL_BRAIN',kickoff=champion)
    combined,individual,families=combine_system_evidence(champion,systems,optional)
    combo,tier,baseline,market_split=strategy_tables(combined)
    ranking=system_screening(champion,individual)
    orientation=_v51_orientation_audit(champion,v51_shadow)
    comparable=orientation['common_games']
    audit={**provenance,'version':VERSION,'generated_utc':_utc(),
        'cfbd_champion_games':len(champion),'priced_champion_games':int(champion.Market_Ready.sum()),
        'champion_game_seasons':{str(int(k)):int(v) for k,v in champion.groupby('Season').size().items()},
        'core_model':PRIMARY_MODEL,'v51_proxy_overlap_games':comparable,
        'v51_v4_orientation':orientation,
        'reconstructed_occurrences':len(systems),'matched_individual_rows':len(individual),
        'fired_families':{str(k):int(v) for k,v in individual.Family.value_counts().items()} if len(individual) else {},
        'market_price_context':market_audit,'optional_voices':optional_audit,
        'historical_quotes_bettable':False,'historical_roi':None,
        'historical_research_prospective':False,'examined_2024_2025_not_pristine':True,
        'bet_authority':0,'production_modified':False,'production_v1_frozen':True,
        'market_rich_scope':'BOOKMAKER_OPEN_CLOSE_PRICE_CONTEXT__NO_QUOTE_TIMESTAMP__NO_RETRO_PREGAME_CERTIFICATION',
        'stat_scope':'OPTIONAL_INDEPENDENT_TIMESTAMPED_FEED__NOT_DERIVED_FROM_CORE_LABEL',
        'historical_combination_rows':len(combo),'system_rankings':len(ranking)}
    return {'audit':audit,'v55_champion_candidates.csv':champion,
            'v55_combined_brain_candidates.csv':combined,
            'v55_mechanism_rankings.csv':ranking,'v55_combination_history.csv':combo,
            'v55_confluence_performance.csv':tier,'v55_baseline_performance.csv':baseline,
            'v55_market_rich_conditionals.csv':market_split,
            'v55_individual_evidence.csv':individual,'v55_family_evidence.csv':families}


def _blob_data(bucket,name):
    obj=bucket.blob(name)
    # This is read-only. Do not create a speculative success report.
    return obj.download_as_text()


def run_cloud(*,storage_client,bucket_name='sharp-models',bq_client=None,log_func=print,
              parent_prefix=SOURCE_RUN):
    """Fast, stand-alone, audited job; V5.4 history does NOT rerun."""
    if not re.fullmatch(r'research/ncaaf/v54_full_research/runs/[0-9]{8}T[0-9]{12}Z',parent_prefix):
        raise RuntimeError('V55_PARENT_MUST_BE_IMMUTABLE_CFBD_RUN')
    bucket=storage_client.bucket(bucket_name)
    parent=_read_json_blob(bucket,parent_prefix+'/report.json')
    _source_check(parent, expected_run=parent_prefix.rsplit('/',1)[-1])
    v4_report=_read_json_blob(bucket,V4_REPORT)
    if v4_report.get('selected_pre2024')!=PRIMARY_MODEL:
        raise RuntimeError('V55_V4_REPORT_CHAMPION_CHANGED')
    v4=_read_csv_blob(bucket,V4_FORECAST)
    occurrences=_read_csv_blob(bucket,parent_prefix+'/cfbd_reconstructed_system_occurrences.csv')
    shadow=_read_csv_blob(bucket,parent_prefix+'/v5_shadow_edge_candidates.csv')
    market=None
    market_audit={'status':'BQ_MARKET_CLIENT_NOT_SUPPLIED'}
    if bq_client is not None:
        # Exact, pre-existing, source-limited historical PRICE fields.
        sql=f'''SELECT Source_Name, Season, Source_Game_ID, Is_Home,
             Consensus_Open_Spread, Consensus_Close_Spread
             FROM `{MARKET_TABLE}`
             WHERE Source_Name = 'CollegeFootballData' AND Season BETWEEN 2021 AND 2025'''
        try:
            q=bq_client.query(sql)
            try:market=q.to_dataframe(create_bqstorage_client=False)
            except TypeError:market=q.to_dataframe()
            market_audit={'status':'BQ_PRICE_CONTEXT_LOADED','rows':len(market)}
        except Exception as e:
            # Requested Market Rich is not optional for the V5.5 Cloud job:
            # failing explicitly is safer than a misleading COMPLETE label.
            raise RuntimeError('V55_MARKET_RICH_BQ_QUERY_FAILED '+str(e)[:300]) from e
        if market is None or len(market)==0:
            raise RuntimeError('V55_MARKET_RICH_BQ_EMPTY__NO_COMPLETE_RESEARCH')
    out=analyze(parent,v4,occurrences,market_raw=market,v51_shadow=shadow)
    if bq_client is not None and not out['audit']['market_price_context'].get('paired_open_close'):
        raise RuntimeError('V55_MARKET_RICH_NO_VERIFIABLE_OPEN_CLOSE_PAIRS')
    out['audit']['market_query']=market_audit
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix=f'{ROOT}/runs/{run_id}'
    payloads={}
    for filename,value in out.items():
        if filename=='audit':continue
        payloads[filename]=value.to_csv(index=False)
    out['audit']['status']='COMPLETE_COMBINED_BRAIN_RESEARCH__PROVISIONAL_ONLY'
    out['audit']['run_prefix']=f'gs://{bucket_name}/{prefix}'
    out['audit']['report_uri']=f'gs://{bucket_name}/{prefix}/report.json'
    audit_json=json.dumps(out['audit'],indent=2,sort_keys=True,default=str,allow_nan=False)
    payloads['report.json']=audit_json
    # Publish every artifact into an immutable run directory; no production
    # paths, tables or bet ledgers are written.
    for filename,content in payloads.items():
        blob=bucket.blob(prefix+'/'+filename)
        blob.upload_from_string(content,content_type='application/json' if filename.endswith('.json') else 'text/csv',
                                if_generation_match=0)
    # Small latest pointer is only for a reader/UI; original research run
    # files are immutable and never overwritten.
    latest={'run_prefix':prefix, 'report_uri':f'gs://{bucket_name}/{prefix}/report.json',
            'run_id':run_id,'status':out['audit']['status'],'bet_authority':0,
            'generated_utc':out['audit']['generated_utc']}
    bucket.blob(f'{ROOT}/latest_report.json').upload_from_string(
        json.dumps(latest,sort_keys=True,allow_nan=False),content_type='application/json')
    report={**out['audit'],'status':'COMPLETE_COMBINED_BRAIN_RESEARCH__PROVISIONAL_ONLY',
            'report_uri':f'gs://{bucket_name}/{prefix}/report.json',
            'prefix':f'gs://{bucket_name}/{prefix}',
            'output_files':sorted(payloads),'market_query':market_audit}
    log_func('[NCAAF-V55] '+json.dumps({k:report[k] for k in ('status','core_model',
              'priced_champion_games','reconstructed_occurrences','system_rankings',
              'historical_combination_rows','bet_authority','report_uri')},default=str))
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--local-v549-folder',required=True)
    parser.add_argument('--v4-csv',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    folder=args.local_v549_folder
    parent=json.load(open(os.path.join(folder,'report.json')))
    v4=pd.read_csv(args.v4_csv,dtype={'Game_ID':'string'})
    occ=pd.read_csv(os.path.join(folder,'cfbd_reconstructed_system_occurrences.csv'),dtype={'Game_ID':'string'})
    result=analyze(parent,v4,occ)
    os.makedirs(args.output_dir,exist_ok=True)
    for k,v in result.items():
        if k=='audit':open(os.path.join(args.output_dir,'report.json'),'w').write(json.dumps(v,indent=2,default=str))
        else:v.to_csv(os.path.join(args.output_dir,k),index=False)
