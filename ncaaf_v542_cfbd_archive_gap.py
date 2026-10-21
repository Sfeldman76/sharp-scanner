"""NCAAF V5.4.2: strictly CFBD-only, 2022 /games archive gap recovery.

Only the existing content-addressed API archive is written, and only original
CFBD /games records absent by game ID. Never touches modeled features, markets,
Production V1, or other years. No synthetic score construction.
"""
from __future__ import annotations
import json
import os
from typing import Any
import pandas as pd

YEAR = 2022
FEATURES = 'sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1'
ARCHIVE = 'sharplogger.sharp_data.ncaaf_cfbd_api_archive'


def _frame(bq, sql):
    result = bq.query(sql)
    try: return result.to_dataframe(create_bqstorage_client=False)
    except TypeError: return result.to_dataframe()


def query_2022(bq):
    # Season metadata may be NULL on historical snapshots. Verify and recover the
    # year from original request parameters, whose bytes are included in SHA-256.
    features = _frame(bq, f'''SELECT DISTINCT CAST(Game_ID AS STRING) AS Game_ID
        FROM `{FEATURES}` WHERE Season = 2022''')
    archive = _frame(bq, f'''SELECT Archive_Row_ID, Endpoint, Params_JSON, Season,
        Retrieved_At, Payload_JSON
        FROM `{ARCHIVE}`
        WHERE REGEXP_CONTAINS(LOWER(Endpoint), r'/games/?$')
        AND (Season = 2022 OR SAFE_CAST(JSON_VALUE(Params_JSON, '$.year') AS INT64) = 2022)''')
    return features, archive


def _parse_existing(archive, truth):
    if archive.empty:
        return pd.DataFrame(columns=['Season','Game_ID','CFBD_Home','CFBD_Away',
                                      'CFBD_Home_Points','CFBD_Away_Points']), {'api_rows':0,'unique_games':0}
    # Malformed seasons and hash conflicts fail closed; never hide them by repair.
    for record in archive.to_dict('records'):
        p = json.loads(record.get('Params_JSON') or '{}')
        yr = p.get('year')
        if yr is not None and int(yr) != YEAR:
            raise RuntimeError('CFBD_2022_ARCHIVE_YEAR_CONFLICT_PARAMS')
        if pd.notna(record.get('Season')) and int(record['Season']) != YEAR:
            raise RuntimeError('CFBD_2022_ARCHIVE_YEAR_CONFLICT_METADATA')
    return truth.parse_cfbd_archive_rows(archive)


def _coverage(feature_ids, data, truth):
    expected = {truth.gid(x) for x in feature_ids['Game_ID']}
    expected.discard('')
    matched = expected & set(data.Game_ID.astype(str)) if not data.empty else set()
    return {'year':YEAR, 'expected_feature_games':len(expected),
            'verified_api_games':len(data), 'matched_feature_games':len(matched),
            'missing_feature_games':len(expected-matched),
            'coverage':len(matched)/len(expected) if expected else 0,
            'missing_examples':sorted(expected-matched)[:8]}


def repair_if_needed(bq, *, log_func=print, enabled=False,
                     truth_module=None, loader_module=None, client_factory=None):
    """Audit 2022 early, optionally fetch original /games and append only missing.

    Only runs when the 2022 match coverage falls below 99%. For an already valid
    archive, invokes zero CFBD API calls and performs zero BigQuery writes.
    """
    if truth_module is None:
        import ncaaf_v54_cfbd_truth as truth_module
    truth=truth_module
    feat, old = query_2022(bq)
    if feat.empty:
        raise RuntimeError('CFBD_2022_FEATURE_ARCHIVE_EMPTY__NOT_AN_API_REPAIR')
    old_games, _old_audit = _parse_existing(old,truth)
    before = _coverage(feat,old_games,truth)
    log_func('[NCAAF-V542-ARCHIVE-GAP] '+json.dumps({'stage':'PREGAME_2022_API_SOURCE_AUDIT',
              'status':'PASS' if before['coverage']>=0.99 else 'GAP_DETECTED',
              'archive_rows':len(old),'api_calls':0,**before},sort_keys=True))
    if before['coverage']>=0.99:
        return {'status':'PASS_EXISTING_2022_CFBD','before':before,'after':before,'api_calls':0,'writes':0}
    if not enabled:
        raise RuntimeError('CFBD_2022_API_GAME_ARCHIVE_INCOMPLETE__REPAIR_DISABLED '
                           +json.dumps(before,sort_keys=True))
    if loader_module is None:
        import ncaaf_cfbd_warehouse_loader_v2 as loader_module
    loader=loader_module
    client=client_factory() if client_factory is not None else loader.CFBDClient()
    log_func('[NCAAF-V542-ARCHIVE-GAP] '+json.dumps({'stage':'CFBD_GAMES_2022_ORIGINAL_API_FETCH',
             'status':'START','scope':'/games?year=2022&seasonType=both&classification=fbs',
             'historical_football_source':'CFBD_ONLY'}))
    _ = loader.pull_games(client,YEAR)
    incoming = [r for r in client.archive_rows
                if str(r.get('Endpoint','')).rstrip('/')=='/games'
                and int(r.get('Season') or -1)==YEAR]
    if not incoming:
        raise RuntimeError('CFBD_2022_ORIGINAL_API_RESPONSE_EMPTY')
    new_games, _ = truth.parse_cfbd_archive_rows(pd.DataFrame(incoming))
    after_dry = _coverage(feat,new_games,truth)
    if after_dry['coverage'] < 0.99:
        raise RuntimeError('CFBD_2022_API_SOURCE_INCOMPLETE '
                           +json.dumps(after_dry,sort_keys=True))
    # Verify source continuity before appending: never let a different CFBD
    # snapshot replace an existing game or introduce contradictory scores/teams.
    overlap = old_games.merge(new_games,on=['Season','Game_ID'],suffixes=('_old','_new'))
    cols=['CFBD_Home','CFBD_Away','CFBD_Home_Points','CFBD_Away_Points']
    conflicts=[]
    for c in cols:
        bad=overlap.loc[overlap[c+'_old'].astype(str).ne(overlap[c+'_new'].astype(str))]
        if len(bad): conflicts.extend(bad.Game_ID.astype(str).tolist())
    if conflicts:
        raise RuntimeError('CFBD_2022_EXISTING_VS_CURRENT_API_CONFLICT n='
                           +str(len(set(conflicts)))+' games='+str(sorted(set(conflicts))[:8]))
    seen=set(old_games.Game_ID.astype(str))
    needed = [r for r in incoming if truth.gid(json.loads(r['Payload_JSON']).get('id')) not in seen]
    # In the rare case all incoming games exist but join coverage remains poor,
    # no synthetic or repeated writes can resolve the underlying ID mismatch.
    if not needed:
        raise RuntimeError('CFBD_2022_NO_NEW_GAME_IDS__POSSIBLE_FEATURE_ID_MISMATCH '
                           +json.dumps(before,sort_keys=True))
    log_func('[NCAAF-V542-ARCHIVE-GAP] '+json.dumps({'stage':'APPEND_IMMUTABLE_CFBD_2022_GAMES',
             'status':'START','original_api_rows':len(incoming),'new_game_rows':len(needed),
             'overlap_verified_games':len(overlap),'preflight_coverage':after_dry['coverage']}))
    # Uses the existing warehouse V2 SHA-256 content-addressed, staged, readback
    # writer. This writes only ncaaf_cfbd_api_archive, not training/features.
    archive_result=loader.write_api_archive(bq,'sharplogger','sharp_data',needed)
    if archive_result.get('status') not in ('PASS','CREATED'):
        raise RuntimeError('CFBD_2022_ARCHIVE_WRITE_INCOMPLETE '+str(archive_result))
    _, verify_raw=query_2022(bq)
    verified, _ = _parse_existing(verify_raw,truth)
    after=_coverage(feat,verified,truth)
    if after['coverage']<0.99 or after['matched_feature_games']<before['matched_feature_games']:
        raise RuntimeError('CFBD_2022_POST_WRITE_READBACK_COVERAGE_FAILED '
                           +json.dumps(after,sort_keys=True))
    result={'status':'PASS_CFBD_API_ONLY_2022_REPAIRED', 'before':before,
            'after':after,'api_calls':client.call_count,'writes':len(needed),
            'archive_source':'ORIGINAL_CFBD_GAMES_VERIFIED_SHA256',
            'feature_table_modified':False,'market_table_modified':False,
            'production_modified':False}
    log_func('[NCAAF-V542-ARCHIVE-GAP] '+json.dumps(result,sort_keys=True))
    return result
