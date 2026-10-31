"""NCAAF V5.5: immutable CORE source snapshots and shadow comparisons.

Reads only the committed, SHA-verified V5.5 backend BigQuery board. Does NOT
re-run CORE, source synthetic odds, assert final game status, or grant bets.
The source publication must have preceded kickoff; all new GCS writes are
immutable, in the existing research namespace. A later GCS reconciliation
never impersonates an earlier capture.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone

from ncaaf_v55_settlement_bridge import key_from_row, norm, parse_dt

ROOT = 'research/ncaaf/v55/production_intelligence_v1'
BOARD_DEFAULT = 'sharplogger.sharp_data.ncaaf_v55_live_board'
MAX_BOARD_ROWS = 10000
MAX_BOARD_BYTES = 8_000_000
POLICY = 'V55-ORIGINAL-COMMITTED-CORE-RESEARCH-ONLY-20261031'


def _float(v):
    try:
        if v is None or v == '':
            return None
        n = float(v)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def _str(v):
    return str(v or '').strip()


def _iso(t):
    return t.isoformat() if t else None


def _row(r):
    if isinstance(r, dict):
        return r
    return dict(r.items())


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def fetch_verified_board(client, *, table=BOARD_DEFAULT, now=None):
    """Retrieve and verify matching DATA + COMPLETE rows, never a partial run."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+', table):
        raise ValueError('INVALID_BOARD_TABLE_IDENTIFIER')
    sql = f'''SELECT snapshot_id,published_at,record_kind,payload_json,
                audit_json,row_count,payload_sha256,publication_type
         FROM `{table}`
         WHERE snapshot_id=(SELECT snapshot_id FROM `{table}`
               WHERE record_kind='COMPLETE' ORDER BY published_at DESC LIMIT 1)'''
    result = [_row(r) for r in client.query(sql).result()]
    if len(result) != 2 or {x.get('record_kind') for x in result} != {'DATA', 'COMPLETE'}:
        raise ValueError('INCOMPLETE_BOARD_PUBLICATION')
    data = next(r for r in result if r['record_kind'] == 'DATA')
    complete = next(r for r in result if r['record_kind'] == 'COMPLETE')
    if any(data.get(k) != complete.get(k) for k in ('snapshot_id','row_count','payload_sha256','published_at','publication_type')):
        raise ValueError('BOARD_COMPLETE_DATA_METADATA_MISMATCH')
    published = parse_dt(complete['published_at'])
    t = parse_dt(now or datetime.now(timezone.utc))
    if not published or not t or published > t + timedelta(minutes=2):
        raise ValueError('INVALID_BOARD_PUBLICATION_TIME')
    if t - published > timedelta(hours=3):
        raise ValueError('BOARD_PUBLICATION_TOO_OLD_FOR_CAPTURE')
    raw = data.get('payload_json')
    if not isinstance(raw, str) or len(raw.encode('utf-8')) > MAX_BOARD_BYTES:
        raise ValueError('INVALID_BOARD_PAYLOAD')
    if _sha(raw) != complete.get('payload_sha256'):
        raise ValueError('BOARD_SHA256_MISMATCH')
    payload = json.loads(raw)
    audit = json.loads(complete.get('audit_json') or 'null')
    if not isinstance(payload, list) or not isinstance(audit, dict) or len(payload) != int(complete['row_count']) or len(payload) > MAX_BOARD_ROWS:
        raise ValueError('BOARD_ROWS_AUDIT_INVALID')
    if audit.get('status') != 'COMPLETE_V55_INDEPENDENT_WEEKLY_SHADOW' or audit.get('production_v1_probabilities_used') is not False:
        raise ValueError('NOT_FROZEN_V55_CORE_PUBLICATION')
    if int(audit.get('v55_bet_authority') or 0) != 0:
        raise ValueError('NONZERO_BOARD_AUTHORITY_REQUIRES_SEPARATE_REVIEW')
    return {'snapshot_id':_str(complete['snapshot_id']),
            'published_at_utc':_iso(published), 'payload_sha256':complete['payload_sha256'],
            'publication_type':complete['publication_type'], 'rows':payload,
            'source_table':table, 'row_count':len(payload)}


def build_core_records(board, *, now=None):
    """Use ONLY original CORE research fields and same-book quotes from the board."""
    published = parse_dt(board['published_at_utc'])
    ingest_time = parse_dt(now or board['published_at_utc'])
    candidates = []
    holds = collections.Counter()
    games = set()
    for x in board['rows']:
        if not isinstance(x, dict):
            holds['INVALID_ROW'] += 1
            continue
        home, away = _str(x.get('Home')), _str(x.get('Away'))
        if not home or not away:
            matchup = _str(x.get('Matchup'))
            if ' @ ' in matchup:
                away, home = matchup.split(' @ ', 1)
        kickoff = parse_dt(x.get('Game_Start_UTC'))
        key = key_from_row({'home':home,'away':away,'kickoff_utc':x.get('Game_Start_UTC')})
        margin = _float(x.get('CORE_Projected_Home_Margin'))
        if not key or not kickoff or kickoff <= published or kickoff <= ingest_time or kickoff > published+timedelta(days=15):
            holds['INVALID_IDENTITY_OR_NOT_ORIGINALLY_PREGAME'] += 1
            continue
        if _str(x.get('CORE_Model')) != 'NONLINEAR_HGB' or margin is None:
            holds['UNVERIFIED_CORE_MODEL_OR_PROJECTION'] += 1
            continue
        game_key = '|'.join(key)
        games.add(game_key)
        selected = _str(x.get('V55_Research_Selected_Team'))
        primary = _str(x.get('Selected_Team'))
        quote_stamp = parse_dt(x.get('Backend_Quote_Timestamp'))
        quote_book = _str(x.get('Backend_Quote_Book'))
        bookline = _float(x.get('Current_Spread'))
        odds = _float(x.get('Odds_American'))
        direction = None
        if selected and primary and norm(selected) == norm(primary):
            direction = 'HOME' if norm(selected)==norm(home) else 'AWAY' if norm(selected)==norm(away) else None
        quote_good = bool(direction and quote_book and quote_stamp and
                          published - timedelta(minutes=90) <= quote_stamp <= published and
                          quote_stamp < kickoff and bookline is not None and odds is not None and
                          (odds <= -100 or odds >= 100) and
                          _str(x.get('Quote_State')).startswith('FRESH_') and
                          _str(x.get('V55_Research_Decision')) in (
                              'UNVALIDATED_NORMAL_MODEL_LEAN_ONLY',
                              'CALIBRATED_RESEARCH_ONLY_NO_BET_AUTHORITY'))
        group = 'PRICED_CORE_RESEARCH_LEAN' if quote_good else 'CORE_MODEL_ONLY_NO_PRICE_OR_SELECTION'
        if not quote_good:
            holds['NO_VERIFIED_CORE_DIRECTION_AND_PRICE'] += 1
        doc = {'family':'CORE','signal_id':'CORE-NONLINEAR_HGB-V55',
               'source':'VERIFIED_V55_BIGQUERY_COMPLETE_PUBLICATION',
               'policy':POLICY,'game_key':game_key,
               'original_physical_game_id':_str(x.get('Game_ID')) or None,
               'home_team':home,'away_team':away,'kickoff_utc':_iso(kickoff),
               'home_margin_projection':margin,
               'residual_sd':_float(x.get('CORE_Residual_SD')),
               'core_model':'NONLINEAR_HGB',
               'core_probability_state':_str(x.get('V55_Probability_State')),
               'raw_cover_probability_unvalidated':_float(x.get('V55_Raw_Cover_Prob')),
               'core_research_decision':_str(x.get('V55_Research_Decision')),
               'snapshot_id':board['snapshot_id'],
               'snapshot_payload_sha256':board['payload_sha256'],
               'snapshot_published_at_utc':board['published_at_utc'],
               'ledger_first_ingestion_utc':_iso(ingest_time),
               'snapshot_source_table':board['source_table'],
               'capture_evidence_type':'BQ_PUBLICATION_PREKICKOFF__GCS_INGESTION_TIME_SEPARATE',
               'model_training_and_feature_vintage_certified':False,
               'source_finality_certified':False,
               'fully_qualified_for_betting':False,
               'bet_authority':0, 'authority_status':'SHADOW_ONLY_NO_BET_AUTHORITY',
               'record_kind':group,
               'selected_team':selected if quote_good else None,
               'side_relative':direction if quote_good else None,
               'quote_book':quote_book if quote_good else None,
               'quote_utc':_iso(quote_stamp) if quote_good else None,
               'quote_line':bookline if quote_good else None,
               'quote_odds':odds if quote_good else None,
               'quote_odds_format':'AMERICAN' if quote_good else None,
               'price_verified':bool(quote_good),
               'direction_verified':bool(quote_good),
               'capture_clock_verified':bool(quote_good),
               'settlement_certified':False,'wager_filled':False,
               'independent_bet_count':0,
               'holds':['MODEL_INPUT_VINTAGE_NOT_INDEPENDENTLY_CERTIFIED',
                        'OFFICIAL_FINALITY_NOT_YET_CERTIFIED',
                        'NO_EXECUTED_WAGER']}
        if quote_good:
            doc['decision_id'] = _sha(game_key+'|CORE|NONLINEAR_HGB')[:32]
        candidates.append(doc)
    # Never allow duplicated or contradictory records for the same local-date
    # game within a single signed CORE publication to become a first decision.
    by_game=collections.defaultdict(list)
    for item in candidates:
        by_game[item['game_key']].append(item)
    unique=[]
    for k,items in by_game.items():
        if len(items)==1:
            unique.append(items[0])
            continue
        sides={(x.get('side_relative'),x.get('quote_line'),
                x.get('snapshot_published_at_utc')) for x in items}
        if len(sides)>1:
            holds['AMBIGUOUS_MULTIPLE_CORE_ROWS_PER_GAME']+=1
            continue
        # Identical quote rows are one matchup, not independent opportunities.
        unique.append(items[0])
    return unique, {'model_games':len(games), 'records':len(unique),
                    'holds':dict(holds)}


def _put_once(bucket, path, doc):
    blob = bucket.blob(path)
    try:
        blob.upload_from_string(json.dumps(doc, sort_keys=True, allow_nan=False,
                                            separators=(',',':')).encode('utf-8'),
                                content_type='application/json',if_generation_match=0)
        return 'CREATED'
    except Exception:
        if blob.exists():
            return 'ALREADY_EXISTS'
        raise


def _read_existing(blob):
    return json.loads(blob.download_as_bytes())


def capture_board(*, bq_client, bucket, now=None):
    """Capture one CORE first-priced record per game, no board/research writes."""
    board = fetch_verified_board(bq_client, table=os.getenv('NCAAF_V55_BOARD_TABLE', BOARD_DEFAULT), now=now)
    records, diagnostics = build_core_records(board, now=now)
    created = existing = first_model = changes = 0
    errors = []
    for rec in records:
        # Model-only first-seen is *not* a wager or even a betting candidate.
        digest = _sha(rec['game_key'])[:32]
        if rec['record_kind'] != 'PRICED_CORE_RESEARCH_LEAN':
            path = f'{ROOT}/core_model_first_seen_v1/{digest}.json'
            try:
                first_model += int(_put_once(bucket,path,rec)=='CREATED')
            except Exception as exc:
                errors.append('CORE_MODEL_WRITE:'+type(exc).__name__)
            continue
        path = f'{ROOT}/core_first_priced_v1/{digest}.json'
        try:
            status = _put_once(bucket,path,rec)
            created += int(status=='CREATED')
            existing += int(status=='ALREADY_EXISTS')
            if status == 'ALREADY_EXISTS':
                previous = _read_existing(bucket.blob(path))
                # Keep the very first priced decision. A later opposite side is
                # explicitly reported, never permitted to overwrite the original.
                if norm(previous.get('selected_team')) != norm(rec['selected_team']):
                    changes += 1
                    conflict = {'game_key':rec['game_key'], 'first_decision_id':previous.get('decision_id'),
                                'first_team':previous.get('selected_team'), 'later_team':rec['selected_team'],
                                'later_snapshot_id':rec['snapshot_id'], 'status':'CORE_RESEARCH_SIDE_FLIP',
                                'bet_authority':0}
                    conflict_name = f'{ROOT}/core_side_flips_v1/{digest}/{_sha(rec["snapshot_id"])[:24]}.json'
                    _put_once(bucket,conflict_name,conflict)
        except Exception as exc:
            errors.append('CORE_DECISION_WRITE:'+type(exc).__name__)
    return {'status':'PASS' if not errors else 'PARTIAL_HOLD',
            'source':'V55_VERIFIED_COMPLETE_BIGQUERY_BOARD',
            'source_snapshot_id':board['snapshot_id'],
            'source_published_utc':board['published_at_utc'],
            'core_model_games_seen':diagnostics['model_games'],
            'new_core_first_priced':created,'existing_core_first_priced':existing,
            'new_core_model_only':first_model,
            'core_research_side_flips_observed':changes,
            'missing_selection_and_price':diagnostics['holds'].get('NO_VERIFIED_CORE_DIRECTION_AND_PRICE',0),
            'errors_count':len(errors),'errors':errors[:10],
            'core_input_vintage_certified':False,'bet_authority':0}


def aggregate_core_miner_relationships(decisions, scores, conflict_ids=None):
    """One matchup per game, not one bet per Miner firing or bookmaker."""
    conflicted = set(conflict_ids or ())
    bygame = collections.defaultdict(list)
    for d in decisions:
        if isinstance(d,dict) and d.get('game_key') and d.get('decision_id') not in conflicted:
            bygame[d['game_key']].append(d)
    result=[]
    for game, members in bygame.items():
        core = [d for d in members if d.get('family')=='CORE'
                and d.get('side_relative') in ('HOME','AWAY') and d.get('price_verified')]
        miners = [d for d in members if d.get('family')=='MINER'
                  and d.get('side_relative') in ('HOME','AWAY') and d.get('price_verified')]
        if not core or not miners:
            continue
        if len(core)!=1:
            continue
        c=core[0]
        sides={d['side_relative'] for d in miners}
        relation=('MIXED_MINER_DIRECTIONS' if len(sides)>1 else
                  'MINER_CONFIRMS_CORE' if c['side_relative'] in sides else
                  'MINER_OPPOSES_CORE')
        score=scores.get(game)
        matched=[]
        # Same normalized side and exactly opposite line, no market proxy.
        for m in miners:
            mline, cline = _float(m.get('quote_line')), _float(c.get('quote_line'))
            if mline is None or cline is None:
                continue
            if m['side_relative'] == c['side_relative']:
                same_line = abs(mline - cline) <= .01
            else:
                same_line = abs(mline + cline) <= .01
            if same_line:
                matched.append(m)
        # Score an ATS comparison ONCE per game, only for a single Miner
        # direction and an exactly matched spread. Multiple mechanisms do not
        # contribute multiple independent outcomes. Scores remain provisional.
        core_ats = miner_ats = None
        if score is not None and len(sides)==1 and matched:
            home_score, away_score, _cfbd = score
            core_margin = ((home_score-away_score) if c['side_relative']=='HOME'
                           else (away_score-home_score)) + float(c['quote_line'])
            m0 = sorted(matched,key=lambda r:(_str(r.get('capture_utc')),
                                                 _str(r.get('signal_id'))))[0]
            miner_margin = ((home_score-away_score) if m0['side_relative']=='HOME'
                            else (away_score-home_score)) + float(m0['quote_line'])
            core_ats = 'WIN' if core_margin>0 else 'LOSS' if core_margin<0 else 'PUSH'
            miner_ats = 'WIN' if miner_margin>0 else 'LOSS' if miner_margin<0 else 'PUSH'
        result.append({'game_key':game,'relation':relation,
                       'core_selected_team':c.get('selected_team'),
                       'same_line_core_ats_provisional':core_ats,
                       'same_line_miner_ats_provisional':miner_ats,
                       'same_line_one_result_per_game':True,
                       'core_selected_side':c['side_relative'],
                       'miner_mechanisms_on_game':len(miners),
                       'distinct_miner_named_family_keys_not_certified_independent':len(set(
                           d.get('evidence_family_key') or d.get('signal_id') for d in miners)),
                       'same_line_comparisons':len(matched),
                       'score_review_available':score is not None,
                       'finality_certified':False,'bet_authority':0,
                       'research_only':True})
    return result
