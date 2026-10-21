"""NCAAF V5.5.4: targeted immutable 2026 CFBD /games archive repair.

Only one CFBD endpoint is requested; no other warehouse, market, training,
production, or feature tables are modified.  The archive retains the exact
API payload and original content hash.  Failure to meet the weekly model's
past-game coverage standard leaves the weekly scorer fail closed.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

import ncaaf_cfbd_warehouse_loader_v2 as warehouse
import ncaaf_v54_cfbd_truth as truth

VERSION = 'NCAAF_V554_CFBD_2026_GAMES_ONLY_BACKFILL'
PROJECT = 'sharplogger'
DATASET = 'sharp_data'
FEATURE_TABLE = f'{PROJECT}.{DATASET}.ncaaf_cfbd_miner_features_v1'
ARCHIVE_TABLE = f'{PROJECT}.{DATASET}.ncaaf_cfbd_api_archive'
REPORT_ROOT = 'research/ncaaf/cfbd_2026_games_backfill'


def _utc(now=None):
    x = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    return x.tz_localize('UTC') if x.tzinfo is None else x.tz_convert('UTC')


def _id(value):
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        raise RuntimeError('V554_CFBD_GAME_ID_NULL')
    if isinstance(value, int) and not isinstance(value, bool):
        result = str(value)
    else:
        result = str(value).strip()
    if not re.fullmatch(r'[0-9]+', result):
        raise RuntimeError(f'V554_CFBD_GAME_ID_NOT_EXACT_DIGITS {result[:60]}')
    return result


def prepare_schedule(frame: pd.DataFrame, *, now=None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require 2026 two-sided CFBD schedule identities; no sportsbook joining."""
    needed = {'Game_ID','Season','Game_Date','Team','Opponent','Is_Home'}
    if frame is None or frame.empty or needed - set(frame):
        raise RuntimeError('V554_CFBD_2026_FEATURE_SCHEDULE_MISSING')
    s=frame.copy()
    if not pd.to_numeric(s.Season,errors='coerce').eq(2026).all():
        raise RuntimeError('V554_FEATURE_SCHEDULE_NOT_2026')
    s['Game_ID']=s.Game_ID.map(_id)
    s['Game_Date']=pd.to_datetime(s.Game_Date,errors='coerce',utc=True)
    s['Is_Home']=pd.to_numeric(s.Is_Home,errors='coerce')
    if s.Game_Date.isna().any() or not s.Is_Home.isin([0,1]).all():
        raise RuntimeError('V554_FEATURE_SCHEDULE_DATE_OR_ORIENTATION_INVALID')
    s['_team_key']=s.Team.map(truth.cfbd_school_key)
    s['_opp_key']=s.Opponent.map(truth.cfbd_school_key)
    if s._team_key.eq('').any() or s._opp_key.eq('').any():
        raise RuntimeError('V554_FEATURE_SCHEDULE_TEAM_NAME_INVALID')
    groups=s.groupby('Game_ID',sort=False)
    if (groups.size()!=2).any() or (groups.Game_Date.nunique()!=1).any():
        raise RuntimeError('V554_FEATURE_SCHEDULE_GAME_DUPLICATE_OR_DATE_CONFLICT')
    if (groups.Is_Home.nunique()!=2).any():
        raise RuntimeError('V554_FEATURE_SCHEDULE_GAME_HOME_AWAY_MISSING')
    home=s[s.Is_Home.eq(1)].copy()
    away=s[s.Is_Home.eq(0)].set_index('Game_ID')
    if ((home._team_key.to_numpy()!=away.loc[home.Game_ID,'_opp_key'].to_numpy()) |
        (home._opp_key.to_numpy()!=away.loc[home.Game_ID,'_team_key'].to_numpy())).any():
        raise RuntimeError('V554_FEATURE_SCHEDULE_OPPONENT_MISMATCH')
    prev=home.Game_Date.lt(_utc(now).normalize())  # same cutoff as V5.5.2
    return home[['Game_ID','Game_Date','Team','Opponent']].reset_index(drop=True), {
        'feature_games_2026':len(home),'past_kickoffs_before_utc_day':int(prev.sum()),
        'upcoming_or_same_day_games':int((~prev).sum()), 'game_identity':'CFBD_GAME_ID',
        'score_authority':'ORIGINAL_CFBD_GAMES_API'
    }


def _verify_identity(scheduled: pd.DataFrame, completed: pd.DataFrame) -> tuple[int,int]:
    if completed is None or completed.empty:return 0,0
    g=completed.copy();g['Game_ID']=g.Game_ID.map(_id)
    common=scheduled.merge(g[['Game_ID','CFBD_Home','CFBD_Away']],on='Game_ID',how='inner',validate='one_to_one')
    if ((common.Team.map(truth.cfbd_school_key)!=common.CFBD_Home.map(truth.cfbd_school_key)) |
        (common.Opponent.map(truth.cfbd_school_key)!=common.CFBD_Away.map(truth.cfbd_school_key))).any():
        bad=common.loc[(common.Team.map(truth.cfbd_school_key)!=common.CFBD_Home.map(truth.cfbd_school_key)) |
                       (common.Opponent.map(truth.cfbd_school_key)!=common.CFBD_Away.map(truth.cfbd_school_key)), 'Game_ID'].tolist()
        raise RuntimeError('V554_CFBD_ARCHIVE_TEAM_CONFLICT game_ids='+','.join(map(str,bad[:5])))
    return len(common),len(set(g.Game_ID)-set(scheduled.Game_ID))


def _parse_archive_nonempty(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame(columns=['Season','Game_ID','CFBD_Home','CFBD_Away','CFBD_Home_Points','CFBD_Away_Points','Retrieved_At'])
    try:
        parsed,_=truth.parse_cfbd_archive_rows(rows)
    except RuntimeError as exc:
        if str(exc)=='CFBD_GAME_ARCHIVE_NO_VERIFIABLE_COMPLETED_GAMES':
            return pd.DataFrame(columns=['Season','Game_ID','CFBD_Home','CFBD_Away','CFBD_Home_Points','CFBD_Away_Points','Retrieved_At'])
        raise
    return parsed.loc[parsed.Season.eq(2026)].copy()


def coverage(scheduled: pd.DataFrame, completed: pd.DataFrame, *, now=None) -> dict[str, Any]:
    common_n,unmatched_n=_verify_identity(scheduled,completed)
    past=scheduled.loc[scheduled.Game_Date.lt(_utc(now).normalize())].copy()
    observed=(past.merge(completed[['Game_ID','Retrieved_At','CFBD_Home_Points','CFBD_Away_Points']],
                on='Game_ID',how='left',validate='one_to_one') if len(completed) else past.assign(
                Retrieved_At=pd.NaT,CFBD_Home_Points=float('nan'),CFBD_Away_Points=float('nan')))
    rt=pd.to_datetime(observed.Retrieved_At,errors='coerce',utc=True)
    valid=(rt.ge(observed.Game_Date) & rt.le(_utc(now)) &
           pd.to_numeric(observed.CFBD_Home_Points,errors='coerce').notna() &
           pd.to_numeric(observed.CFBD_Away_Points,errors='coerce').notna())
    count=int(valid.sum());expected=len(past)
    return {
        'past_games_expected':expected,'past_games_verified':count,
        'past_score_coverage':round(count/expected,6) if expected else 1.0,
        'missing_past_game_ids':past.loc[~valid.values,'Game_ID'].astype(str).head(30).tolist(),
        'completed_games_matched_to_feature_schedule':common_n,
        'completed_games_not_in_feature_schedule':unmatched_n,
        'score_observation_after_kickoff_only':True,
        'future_games_not_scored':True,
    }


def validate_api_games(rows:list[dict[str,Any]], schedule:pd.DataFrame, *, season=2026) -> dict[str,Any]:
    if not isinstance(rows,list) or not rows:
        raise RuntimeError('V554_CFBD_2026_GAMES_API_EMPTY')
    names={str(r['Game_ID']):(truth.cfbd_school_key(r['Team']),truth.cfbd_school_key(r['Opponent']))
           for r in schedule.to_dict('records')}
    ids=set(); scored=0; scheduled_match=0; mismatches=[]
    for r in rows:
        if not isinstance(r,dict) or not {'id','homeTeam','awayTeam'}.issubset(r):
            raise RuntimeError('V554_CFBD_GAMES_API_MALFORMED_PAYLOAD')
        gid=_id(r['id'])
        if gid in ids:raise RuntimeError('V554_CFBD_GAMES_API_DUPLICATE_ID '+gid)
        ids.add(gid)
        if r.get('season') is not None and int(r['season'])!=season:
            raise RuntimeError('V554_CFBD_GAMES_API_WRONG_SEASON '+gid)
        home=truth.cfbd_school_key(r['homeTeam']);away=truth.cfbd_school_key(r['awayTeam'])
        if not home or not away or home==away:raise RuntimeError('V554_CFBD_GAMES_API_INVALID_TEAMS '+gid)
        if gid in names:
            scheduled_match+=1
            if names[gid]!=(home,away):mismatches.append(gid)
        hp=r.get('homePoints');ap=r.get('awayPoints')
        if (hp is None) != (ap is None):
            raise RuntimeError('V554_CFBD_GAMES_API_PARTIAL_SCORE '+gid)
        if hp is not None:
            try:
                if not (float(hp)>=0 and float(ap)>=0 and float(hp)<300 and float(ap)<300):raise ValueError()
            except (ValueError,TypeError,OverflowError):
                raise RuntimeError('V554_CFBD_GAMES_API_BAD_SCORE '+gid)
            scored+=1
    if mismatches:
        raise RuntimeError('V554_CFBD_GAMES_API_IDENTITY_CONFLICT game_ids='+','.join(mismatches[:5]))
    if scheduled_match==0:
        raise RuntimeError('V554_CFBD_GAMES_API_NO_GAME_ID_OVERLAP_WITH_FEATURES')
    return {'api_games':len(rows),'api_games_with_both_scores':scored,
            'api_games_matched_feature_ids':scheduled_match,
            'api_games_not_in_feature_table':len(rows)-scheduled_match}


def _query_frame(bq,sql):
    result=bq.query(sql)
    try:return result.to_dataframe(create_bqstorage_client=False)
    except TypeError:return result.to_dataframe()


def run_backfill(*,bq_client,storage_client=None,bucket_name='sharp-models',cfbd_client=None,
                 now=None,project=PROJECT,dataset=DATASET,min_completion_coverage=.95,
                 force_api=False,log_func=print,archive_writer=None) -> dict[str,Any]:
    """Repair only /games 2026 and audit against 2026 two-sided CFBD features."""
    if not 0<min_completion_coverage<=1:raise ValueError('V554_INVALID_COVERAGE_GATE')
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',project) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',dataset):
        raise ValueError('V554_INVALID_BIGQUERY_PROJECT_DATASET')
    feature=f'{project}.{dataset}.ncaaf_cfbd_miner_features_v1'
    archive=f'{project}.{dataset}.{warehouse.API_ARCHIVE_TABLE}'
    # Preflight existing target/schema before issuing any CFBD request.
    # UPDATE permission is verified by the atomic DML writer after API validation.
    expected_schema={'Archive_Row_ID','Endpoint','Params_JSON','Payload_JSON','Season','Retrieved_At'}
    actual={f.name for f in bq_client.get_table(archive).schema}
    if expected_schema-actual:
        raise RuntimeError('V554_CFBD_ARCHIVE_SCHEMA_MISSING '+str(sorted(expected_schema-actual)))
    source=_query_frame(bq_client,f'SELECT Game_ID,Season,Game_Date,Team,Opponent,Is_Home FROM `{feature}` WHERE Season=2026')
    schedule,schedule_audit=prepare_schedule(source,now=now)
    query=f"SELECT Archive_Row_ID, Endpoint, Params_JSON, Payload_JSON, Season, Retrieved_At FROM `{archive}` WHERE Season=2026 AND Endpoint='/games'"
    existing=_query_frame(bq_client,query)
    previous=_parse_archive_nonempty(existing)
    before=coverage(schedule,previous,now=now)
    log_func('[NCAAF-V554-2026-GAMES] '+json.dumps({'stage':'PRECHECK','status':'PASS',**schedule_audit,**before},sort_keys=True))
    if not force_api and before['past_games_expected']>0 and before['past_games_verified']==before['past_games_expected']:
        complete={'status':'PASS_ALREADY_COMPLETE','api_calls':0,'writes':0,
                'production_modified':False,'feature_warehouse_modified':False,
                'market_warehouse_modified':False,'bet_authority':0,
                **schedule_audit,**before}
        log_func('[NCAAF-V554-2026-GAMES] '+json.dumps(complete,sort_keys=True))
        return complete
    if cfbd_client is None:cfbd_client=warehouse.CFBDClient()
    # Exactly one endpoint, classification matches existing CFBD warehouse loader.
    raw=cfbd_client.get('/games',{'year':2026,'seasonType':'both','classification':'fbs'})
    api_audit=validate_api_games(raw,schedule)
    archive_rows=cfbd_client.archive_rows
    if not archive_rows or any(r.get('Endpoint')!='/games' or r.get('Season')!=2026 for r in archive_rows):
        raise RuntimeError('V554_CFBD_CLIENT_ARCHIVE_NOT_GAMES_2026_ONLY')
    # Preview ALL source hashes/score conflicts before DML; no synthetic records.
    combined=_parse_archive_nonempty(pd.DataFrame(archive_rows+existing.to_dict('records')))
    _verify_identity(schedule,combined)
    pending=coverage(schedule,combined,now=now)
    write=archive_writer or warehouse.append_existing_api_archive_without_staging
    written=write(bq_client,project,dataset,archive_rows,log_func=log_func)
    if written.get('status')!='PASS' or int(written.get('verified_rows',-1))!=len({r['Archive_Row_ID'] for r in archive_rows}):
        raise RuntimeError('V554_CFBD_ARCHIVE_WRITE_NOT_VERIFIED')
    after_rows=_query_frame(bq_client,query)
    after=_parse_archive_nonempty(after_rows)
    final=coverage(schedule,after,now=now)
    if final['past_games_verified']<pending['past_games_verified']:
        raise RuntimeError('V554_CFBD_ARCHIVE_READBACK_LOST_VERIFIED_SCORES')
    report={'version':VERSION,'status':'PASS' if final['past_score_coverage']>=min_completion_coverage else 'COVERAGE_HOLD',
            'season':2026,'source':'CFBD_/games_ONLY','api_calls':int(getattr(cfbd_client,'call_count',1)),
            'archive_write':written,'before':before,'after':final,'schedule':schedule_audit,
            'api':api_audit,'min_past_completion_coverage':min_completion_coverage,
            'production_modified':False,'feature_warehouse_modified':False,
            'market_warehouse_modified':False,'bet_authority':0,
            'note':'Past-game cutoff matches V5.5.2 UTC-day guard; no bookmaker scores accepted.'}
    if storage_client is not None:
        stamp=_utc(now).strftime('%Y%m%dT%H%M%S%fZ')
        key=f'{REPORT_ROOT}/runs/{stamp}/report.json'
        blob=storage_client.bucket(bucket_name).blob(key)
        blob.upload_from_string(json.dumps(report,indent=2,sort_keys=True,default=str),content_type='application/json')
        report['report_uri']=f'gs://{bucket_name}/{key}'
    log_func('[NCAAF-V554-2026-GAMES] '+json.dumps(report,sort_keys=True,default=str))
    if report['status']!='PASS':
        raise RuntimeError('V554_CFBD_2026_PAST_GAME_COVERAGE_HOLD '+
                           str(final['past_games_verified'])+'/'+str(final['past_games_expected']))
    return report
