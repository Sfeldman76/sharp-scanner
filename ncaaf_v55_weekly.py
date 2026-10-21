"""V5.5 weekly selected-slate observability overlay (not a new probability model).

Reads the already frozen NCAAF Production V1 model/selector output. Publishes
one immutable, source-labeled research-only snapshot for Streamlit. It NEVER
trains/replaces NONLINEAR_HGB, touches Production V1 locks, or grants new votes.
"""
from __future__ import annotations
from datetime import datetime, timezone
from io import StringIO
import hashlib
import json
import math
import re
import numpy as np
import pandas as pd

VERSION = 'NCAAF_V551_WEEKLY_COMBINED_BRAIN_BOARD_20261021'
ROOT = 'research/ncaaf/v55_weekly'
RESEARCH_ROOT = 'research/ncaaf/v55_combined_brain'
CORE_RESEARCH_WINNER = 'NONLINEAR_HGB'
FAMILY_COLUMNS = ('STAT','BIG_AL','PATHI','SYSTEM_MINER','PREDICTION_TRACKER')
WEEKLY_COLUMNS = (
 'Game_ID','Game_Start_UTC','Matchup','Home','Away','Selected_Team','Snapshot_UTC',
 'Quote_Age_Min','Book','Current_Spread','Odds_American','CORE_Production_Model',
 'CORE_Cover_Probability','CORE_Edge','CORE_EV','CORE_Qualifies','Production_Action',
 'STAT','BIG_AL','PATHI','SYSTEM_MINER','PREDICTION_TRACKER',
 'Confirm_Families','Oppose_Families','Trigger_Only_Families',
 'Confirm_Count','Oppose_Count','Market_Rich_State','Market_Rich_Open_Line',
 'Market_Rich_Move_Points','Market_Rich_Key_Crossings',
 'Market_Rich_Source_State','Market_Rich_Sharp_Move','Market_Rich_Limit',
 'Market_Rich_Sharp_Limit_Total','Market_Rich_Book_Count',
 'Market_Rich_Steam_Flag','Market_Rich_Sharp_Bet_Score',
 'V55_Research_Priority','V55_Bet_Authority','Research_Champion','Champion_Live_Status',
 'Core_Artifact_SHA','Evidence_Time_Status')


def _date(x):
    return pd.to_datetime(x,utc=True,errors='coerce')


def _num(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else float('nan')
    except (TypeError,ValueError,OverflowError):
        return float('nan')


def _str(x):
    if x is None:return ''
    try:
        if pd.isna(x):return ''
    except (ValueError,TypeError):pass
    s=str(x).strip()
    return '' if s.casefold() in ('nan','none','<na>','—','null') else s


def _sources(x):
    return [p.strip() for p in re.split(r'\s*\|\s*',_str(x)) if p.strip()]


def _family(source):
    x=re.sub(r'[^A-Z0-9]+','_',source.upper()).strip('_')
    if x.startswith(('BIG_AL','BIGAL')):return 'BIG_AL'
    if x.startswith('PATHI'):return 'PATHI'
    if x.startswith(('MINER','NCAAF_MECH','NCAAF_RV','SYSTEM_MINER')):return 'SYSTEM_MINER'
    if x.startswith(('STAT','SPREAD_STAT')):return 'STAT'
    if x.startswith(('PT_','PREDICTION_TRACKER')):return 'PREDICTION_TRACKER'
    return None


def _family_evidence(row):
    support=set(filter(None,(_family(t) for t in _sources(row.get('_system_support_sources')))))
    opposition=set(filter(None,(_family(t) for t in _sources(row.get('_system_conflict_sources')))))
    raw_triggers={
       'BIG_AL': _str(row.get('BigAl_Active_Text')),
       'PATHI': _str(row.get('Pathi_Active_Text')),
       'SYSTEM_MINER': _str(row.get('NCAAF_RV2_System_Summary')) or _str(row.get('NCAAF_RV2_Research_System_Summary')),
    }
    states={}; triggers=[]
    for family in FAMILY_COLUMNS:
        if family in support and family in opposition:state='CONFLICT'
        elif family in support:state='CONFIRM'
        elif family in opposition:state='OPPOSE'
        elif raw_triggers.get(family):
            # Named system presence without verified sided vote is not support.
            state='TRIGGER_ONLY';triggers.append(family)
        elif family=='STAT' and _num(row.get('NCAAF_Prod_Stat_Selector_Active'))==1:
            state='ACTIVE_NO_DIRECTION';triggers.append(family)
        else:state='NOT_EVALUABLE'
        states[family]=state
    confirms=sorted(f for f,v in states.items() if v=='CONFIRM')
    opposes=sorted(f for f,v in states.items() if v in ('OPPOSE','CONFLICT'))
    return states,confirms,opposes,triggers


def _key_count(o,c):
    if not (math.isfinite(o) and math.isfinite(c)) or o==c:return float('nan')
    lo,hi=sorted((o,c))
    return int(sum(lo<k<=hi for k in (-14,-10,-7,-3,3,7,10,14)))


def _verified_market_rich(row,snapshot,kickoff):
    """Never treat an untimestamped opening-anchor or postgame close as live data."""
    current=_num(row.get('_line'))
    opener=_num(row.get('Open_Spread_Selected_Side'))
    open_at=_date(row.get('Open_Snapshot_At'))
    if not math.isfinite(current):return ('NO_CURRENT_SPREAD',np.nan,np.nan,np.nan)
    if math.isfinite(opener) and pd.notna(open_at) and open_at<=snapshot and open_at<kickoff:
        return ('TIMED_SELECTED_SIDE_OPEN',opener,current-opener,_key_count(opener,current))
    return ('OPEN_VINTAGE_OR_ORIENTATION_UNVERIFIED',np.nan,np.nan,np.nan)


def weekly_frame_from_picks(picks, *, now=None, contract_sha=''):
    """Format the SAME canonical weekly selector output; never rescore picks.

    No non-CFBD sportsbook key is converted into a CFBD game ID. One row
    per physical game/market is required by the production scorer; spread
    only is shown to avoid implying the V5.5 spread research covers H2H/totals.
    """
    if picks is None or not isinstance(picks,pd.DataFrame):
        raise RuntimeError('V551_CANONICAL_PICKS_FRAME_REQUIRED')
    needed={'_prod_game_id','Market','Game_Start','_prod_outcome','_pred',
            '_line','_odds','_ts','_model_id'}
    missing=needed-set(picks.columns)
    if missing and not picks.empty:
        raise RuntimeError('V551_CANONICAL_FIELDS_MISSING '+str(sorted(missing)))
    utc=_date(now) if now is not None else pd.Timestamp.now(tz='UTC')
    if pd.isna(utc):raise RuntimeError('V551_NOW_INVALID')
    if picks.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS),{'status':'NO_UPCOMING_GAMES','spread_games':0,'production_mutation':False,
                     'v55_bet_authority':0,'champion_research_live_scored':False,'snapshot_utc':utc.isoformat(),
                     'core_qualified':0,'multi_family_research':0,'candidate_confirming_families':0,
                     'timed_open_quotes':0,'excluded_started_games':0}
    if {'_prod_game_id','Market'}.issubset(picks) and picks.duplicated(['_prod_game_id','Market']).any():
        raise RuntimeError('V551_DUPLICATE_CANONICAL_GAME_MARKET')
    sp=picks.loc[picks.Market.astype(str).str.casefold().eq('spreads')].copy()
    output=[];excluded=0
    for _,row in sp.iterrows():
        gameid=_str(row.get('_prod_game_id'))
        kickoff=_date(row.get('Game_Start'))
        snap=_date(row.get('_ts'))
        if not gameid or pd.isna(kickoff) or pd.isna(snap):
            raise RuntimeError('V551_PREGAME_ID_OR_TIME_INVALID')
        # The published weekly snapshot is an observation, never a historical
        # reconstruction. Rows whose kickoff already passed are not current.
        if kickoff<=utc:
            excluded+=1; continue
        if snap>utc+pd.Timedelta(minutes=2) or snap>=kickoff:
            raise RuntimeError('V551_QUOTE_NOT_OBSERVED_PREGAME')
        p=_num(row.get('_pred'));odds=_num(row.get('_odds'));line=_num(row.get('_line'))
        if math.isfinite(p) and not 0<=p<=1:raise RuntimeError('V551_PROBABILITY_OUT_OF_RANGE')
        home=_str(row.get('Home_Team_Norm')) or _str(row.get('Home_Team'))
        away=_str(row.get('Away_Team_Norm')) or _str(row.get('Away_Team'))
        selected=_str(row.get('_prod_outcome'))
        if not home or not away or home.casefold()==away.casefold():
            raise RuntimeError('V551_HOME_AWAY_IDENTITY_MISSING')
        if not selected or selected.casefold() not in {home.casefold(),away.casefold()}:
            raise RuntimeError('V551_SELECTED_SIDE_IDENTITY_MISMATCH')
        states,confirm,oppose,triggers=_family_evidence(row)
        core_gate=row.get('_core_qualifies',False)
        core_gate=bool(core_gate) if not pd.isna(core_gate) else False
        edge=_num(row.get('_edge'));ev=_num(row.get('_ev'))
        mr_state,open_line,movement,keycount=_verified_market_rich(row,snap,kickoff)
        # A provisional research ranking is descriptive and cannot create a bet.
        ready=(core_gate and math.isfinite(edge) and edge>0 and math.isfinite(ev) and ev>0
               and math.isfinite(odds) and math.isfinite(line) and math.isfinite(p))
        priority=('REVIEW_CONFLICT' if ready and oppose else
                  'MULTIFAMILY_RESEARCH' if ready and len(confirm)>=2 else
                  'SINGLE_FAMILY_RESEARCH' if ready and len(confirm)==1 else
                  'CORE_ONLY_RESEARCH' if ready else 'NO_RESEARCH_CANDIDATE')
        output.append({
          'Game_ID':'production:'+gameid,'Game_Start_UTC':kickoff.isoformat(),
          'Matchup':f'{away} @ {home}','Home':home,'Away':away,
          'Selected_Team':selected,'Snapshot_UTC':snap.isoformat(),
          'Quote_Age_Min':round(max((utc-snap).total_seconds()/60,0),1),
          'Book':_str(row.get('_book')) or _str(row.get('Bookmaker')),
          'Current_Spread':line,'Odds_American':odds,
          'CORE_Production_Model':_str(row.get('_model_id')),
          'CORE_Cover_Probability':p,'CORE_Edge':edge,'CORE_EV':ev,
          'CORE_Qualifies':core_gate,'Production_Action':_str(row.get('_prod_action')),
          **states,
          'Confirm_Families':' | '.join(confirm),'Oppose_Families':' | '.join(oppose),
          'Trigger_Only_Families':' | '.join(triggers),
          'Confirm_Count':len(confirm),'Oppose_Count':len(oppose),
          'Market_Rich_State':mr_state,'Market_Rich_Open_Line':open_line,
          'Market_Rich_Move_Points':movement,'Market_Rich_Key_Crossings':keycount,
          # The existing Utils/Move Master features may be present on the
          # canonical selected quote. Export actual observed values ONLY. No
          # missing->zero fill, no inferred steam direction or research vote.
          'Market_Rich_Source_State':'SNAPSHOT_DIAGNOSTICS_ONLY__UNVERIFIED_FEATURE_VINTAGE',
          'Market_Rich_Sharp_Move':_str(row.get('Sharp_Move_Signal')),
          'Market_Rich_Limit':_num(row.get('Limit')),
          'Market_Rich_Sharp_Limit_Total':_num(row.get('Sharp_Limit_Total')),
          'Market_Rich_Book_Count':_num(row.get('Unique_Sharp_Books')),
          'Market_Rich_Steam_Flag':_str(row.get('Late_Game_Steam_Flag')),
          'Market_Rich_Sharp_Bet_Score':_num(row.get('SharpBetScore')),
          'V55_Research_Priority':priority,'V55_Bet_Authority':0,
          'Research_Champion':CORE_RESEARCH_WINNER,
          'Champion_Live_Status':'HISTORICAL_RESEARCH_ONLY__NO_VERIFIED_2026_FORECAST',
          'Core_Artifact_SHA':_str(contract_sha),
          'Evidence_Time_Status':'CURRENT_QUOTE_ASOF__FAMILY_RULE_VINTAGE_NOT_INDEPENDENTLY_VERIFIED'
        })
    d=pd.DataFrame(output,columns=WEEKLY_COLUMNS)
    d=d.sort_values(['Game_Start_UTC','Matchup'],kind='stable').reset_index(drop=True)
    if d.Game_ID.duplicated().any():raise RuntimeError('V551_DUPLICATE_WEEKLY_GAME_ID')
    audit={'status':'PASS' if len(d) else 'NO_UPCOMING_SPREAD_GAMES',
           'spread_games':len(d),'excluded_started_games':excluded,
           'core_qualified':int(d.CORE_Qualifies.sum()) if len(d) else 0,
           'multi_family_research':int(d.V55_Research_Priority.eq('MULTIFAMILY_RESEARCH').sum()) if len(d) else 0,
           'candidate_confirming_families':int(d.Confirm_Count.sum()) if len(d) else 0,
           'timed_open_quotes':int(d.Market_Rich_State.eq('TIMED_SELECTED_SIDE_OPEN').sum()) if len(d) else 0,
           'production_mutation':False,'v55_bet_authority':0,
           'champion_research_live_scored':False,
           'snapshot_utc':utc.isoformat()}
    return d,audit


def publish_weekly(picks, *, storage_client, bucket_name='sharp-models',
                   now=None, contract_sha='', log_func=print):
    frame,audit=weekly_frame_from_picks(picks,now=now,contract_sha=contract_sha)
    bucket=storage_client.bucket(bucket_name)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix=f'{ROOT}/runs/{run_id}'
    csv_data=frame.to_csv(index=False)
    digest=hashlib.sha256(csv_data.encode('utf-8')).hexdigest()
    report={**audit,'version':VERSION,'run_id':run_id,'run_prefix':prefix,
         'csv_sha256':digest,'core_source':'FROZEN_PRODUCTION_V1_CANONICAL_SELECTOR',
         'selected_v4_research_champion':CORE_RESEARCH_WINNER,
         'research_champion_live_forecast':False,'snapshot_type':'OBSERVED_CURRENT_SLATE',
         'historical_system_rankings_prospective_certified':False,
         'market_rich_limited_to_observed_quotes':True,
         'production_authority_unchanged':True,'status':'COMPLETE_WEEKLY_RESEARCH_SNAPSHOT'}
    for name,contents,kind in [('weekly_board.csv',csv_data,'text/csv'),
                               ('report.json',json.dumps(report,sort_keys=True,indent=2,default=str,allow_nan=False),'application/json')]:
        bucket.blob(prefix+'/'+name).upload_from_string(contents,content_type=kind,if_generation_match=0)
    latest={'status':report['status'],'run_id':run_id,'run_prefix':prefix,
            'snapshot_utc':report['snapshot_utc'],'spread_games':report['spread_games'],
            'csv_sha256':digest,'bet_authority':0}
    bucket.blob(ROOT+'/latest.json').upload_from_string(json.dumps(latest,sort_keys=True),content_type='application/json')
    log_func('[NCAAF-V551-WEEKLY] '+json.dumps({k:report[k] for k in ('status','spread_games','core_qualified','multi_family_research','timed_open_quotes','production_mutation')},sort_keys=True))
    return report


def run_weekly(*,storage_client,bucket_name='sharp-models',picks_frame=None,
               contract=None,bq_client=None,log_func=print,now=None):
    """Read-only standalone workflow or reuse SAME picks from production weekly.

    Standalone runs no ledger settlement/locking. No refit. It still uses the
    existing canonical selector, including its original research PT capture.
    """
    if picks_frame is None:
        import ncaaf_production_v1 as prod
        import utils
        contract=prod.load_production_contract(bucket_name=bucket_name,storage_client=storage_client)
        if not isinstance(contract,dict) or not contract.get('_artifact_sha256'):
            raise RuntimeError('V551_PRODUCTION_CONTRACT_UNAVAILABLE__RUN_EXISTING_PRODUCTION_PUBLISH')
        try:
            from ncaaf_research_v2 import load_current_report_diagnostic
            registry,_=load_current_report_diagnostic(bucket_name=bucket_name,storage_client=storage_client)
        except Exception as exc:
            registry=None
            log_func('[NCAAF-V551-REGISTRY] unavailable='+type(exc).__name__)
        prod.set_current_live_research_report(registry if isinstance(registry,dict) else None,
               storage_client=storage_client,bq_client=bq_client,bucket_name=bucket_name)
        raw=utils.read_recent_sharp_moves(hours=240,
             table=getattr(utils,'DEFAULT_MOVES_VIEW','sharp_data.moves_with_features_merged'),
             pregame_only=True,sport='NCAAF',use_bq_storage=False)
        if raw is None or raw.empty:
            picks_frame=pd.DataFrame()
        else:
            prepared=prod.prepare_current_market_rows(raw,now=now)
            scored=prod.score_live_rows(prepared,contract)
            picks_frame=prod.choose_current_production_picks(scored,contract,
                          executable_books=getattr(utils,'REC_BOOKS',None))
        log_func('[NCAAF-V551-WEEKLY-SCORING] source=EXISTING_FROZEN_PROD_SCORER'
                 ' production_write=FALSE probability_refit=FALSE')
    elif contract is not None and not isinstance(contract,dict):
        raise RuntimeError('V551_CONTRACT_FORMAT_INVALID')
    sha=(contract or {}).get('_artifact_sha256','')
    return publish_weekly(picks_frame,storage_client=storage_client,bucket_name=bucket_name,
                          now=now,contract_sha=sha,log_func=log_func)


def read_weekly_latest(storage_client, bucket_name='sharp-models'):
    """Small on-demand dashboard reader; validates the immutable CSV digest."""
    bucket=storage_client.bucket(bucket_name)
    pointer=json.loads(bucket.blob(ROOT+'/latest.json').download_as_text())
    prefix=pointer.get('run_prefix','')
    if not re.fullmatch(re.escape(ROOT)+r'/runs/[0-9]{8}T[0-9]{12}Z',prefix):
        raise RuntimeError('V551_INVALID_WEEKLY_POINTER')
    report=json.loads(bucket.blob(prefix+'/report.json').download_as_text())
    csv_data=bucket.blob(prefix+'/weekly_board.csv').download_as_text()
    h=hashlib.sha256(csv_data.encode('utf-8')).hexdigest()
    if h!=report.get('csv_sha256') or h!=pointer.get('csv_sha256'):
        raise RuntimeError('V551_WEEKLY_SNAPSHOT_CHECKSUM_MISMATCH')
    if report.get('production_mutation') is not False or report.get('v55_bet_authority')!=0:
        raise RuntimeError('V551_WEEKLY_AUTHORITY_INCONSISTENT')
    df=pd.read_csv(StringIO(csv_data),keep_default_na=False)
    if len(df)!=int(report.get('spread_games',-1)):
        raise RuntimeError('V551_WEEKLY_ROW_COUNT_MISMATCH')
    return df,report
