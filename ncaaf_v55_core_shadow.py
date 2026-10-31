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
    value=str(v or '').strip()
    return '' if value.lower() in ('nan','nat','none','null') else value


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
        if selected and primary and canonical_team_strict(selected) == canonical_team_strict(primary):
            direction = ('HOME' if canonical_team_strict(selected)==canonical_team_strict(home) else
                         'AWAY' if canonical_team_strict(selected)==canonical_team_strict(away) else None)
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
            if not selected: holds['MISSING_V55_RESEARCH_SELECTED_TEAM'] += 1
            if not primary: holds['MISSING_PRIMARY_SELECTED_TEAM'] += 1
            if selected and primary and canonical_team_strict(selected)!=canonical_team_strict(primary):
                holds['RESEARCH_VS_PRIMARY_SELECTION_CONFLICT'] += 1
            if selected and canonical_team_strict(selected) not in (canonical_team_strict(home),canonical_team_strict(away)):
                holds['SELECTED_TEAM_NOT_AN_ORIENTED_PARTICIPANT'] += 1
            if not quote_book: holds['MISSING_BACKEND_BOOK'] += 1
            if not quote_stamp: holds['MISSING_QUOTE_TIMESTAMP'] += 1
            if bookline is None: holds['MISSING_BACKEND_SPREAD'] += 1
            if odds is None: holds['MISSING_BACKEND_ODDS'] += 1
            if not _str(x.get('Quote_State')).startswith('FRESH_'):
                holds['NO_FRESH_VERIFIED_QUOTE_STATE'] += 1
            if _str(x.get('V55_Research_Decision')) not in (
                'UNVALIDATED_NORMAL_MODEL_LEAN_ONLY',
                'CALIBRATED_RESEARCH_ONLY_NO_BET_AUTHORITY'):
                holds['RESEARCH_DECISION_NOT_PRICE_ELIGIBLE'] += 1
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
                if canonical_team_strict(previous.get('selected_team')) != canonical_team_strict(rec['selected_team']):
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
            'research_hold_reasons_overlapping':diagnostics['holds'],
            'errors_count':len(errors),'errors':errors[:10],
            'core_input_vintage_certified':False,'bet_authority':0}


# V55-CANONICAL-CORE-MINER-RESEARCH-JOIN-20261031
# Exact, reviewed school / mascot equivalence classes only. Do not use
# substring matching (Florida vs Florida Atlantic) or speculative fuzzy joins.
TEAM_ALIASES = {
    'appalachian state mountaineers':'appalachian state',
    'arizona wildcats':'arizona',
    'boise state broncos':'boise state',
    'colorado state rams':'colorado state',
    'florida gators':'florida',
    'fresno state bulldogs':'fresno state',
    'georgia bulldogs':'georgia',
    'georgia southern eagles':'georgia southern',
    'georgia tech yellow jackets':'georgia tech',
    'indiana hoosiers':'indiana',
    'iowa state cyclones':'iowa state',
    'kansas jayhawks':'kansas',
    'michigan wolverines':'michigan',
    'minnesota golden gophers':'minnesota',
    'northwestern wildcats':'northwestern',
    'oklahoma state cowboys':'oklahoma state',
    'oregon ducks':'oregon',
    'oregon state beavers':'oregon state',
    'pittsburgh panthers':'pittsburgh',
    'rutgers scarlet knights':'rutgers',
    'smu mustangs':'smu',
    'south florida bulls':'south florida',
    'syracuse orange':'syracuse',
    'tcu horned frogs':'tcu',
    'texas state bobcats':'texas state',
    'texas tech red raiders':'texas tech',
    'uab blazers':'uab',
    'utah state aggies':'utah state',
    'florida atlantic owls':'florida atlantic',
    'north texas mean green':'north texas',
    'james madison dukes':'james madison',
    'troy trojans':'troy',
    'middle tennessee blue raiders':'middle tennessee',
    'kennesaw state owls':'kennesaw state',
    'new mexico state aggies':'new mexico state',
    'jacksonville state gamecocks':'jacksonville state',
    'western kentucky hilltoppers':'western kentucky',
    'delaware blue hens':'delaware',
    'utep miners':'utep',
    'southern miss golden eagles':'southern miss',
    'liberty flames':'liberty',
    'north dakota state bison':'north dakota state',
    'florida international panthers':'florida international',
    'washington state cougars':'washington state',
    'san diego state aztecs':'san diego state',
}


def canonical_team_strict(team):
    normalized = norm(team)
    return TEAM_ALIASES.get(normalized,normalized)


def _research_identity(doc):
    """Derived key, never written into the original immutable observation."""
    kickoff = parse_dt(doc.get('kickoff_utc'))
    h = canonical_team_strict(doc.get('home_team'))
    a = canonical_team_strict(doc.get('away_team'))
    if not kickoff or not h or not a or h == a:
        return None
    from zoneinfo import ZoneInfo
    day = kickoff.astimezone(ZoneInfo('America/New_York')).date().isoformat()
    return ('NCAAF',day,h,a)


def _safe_same_fixture(core,miner,max_minutes=120):
    a,b=parse_dt(core.get('kickoff_utc')),parse_dt(miner.get('kickoff_utc'))
    if not a or not b:
        return False, 'MISSING_KICKOFF'
    cid = _str(core.get('original_physical_game_id'))
    mid = _str(miner.get('original_physical_game_id'))
    if cid and mid and cid == mid:
        return True, 'IDENTICAL_ORIGINAL_ID'
    if abs((a-b).total_seconds()) <= max_minutes*60:
        return True, 'EXPLICIT_ALIASES_AND_KICKOFF_WITHIN_2H'
    return False, 'KICKOFF_DIFFERENCE_HOLD'


def _score_for_identity(identity,core,miner,scores):
    """Require an unambiguous identical score for the captured participant pair."""
    choices=[]
    for d in [core]+list(miner):
        score=scores.get(d.get('game_key'))
        if score is not None:
            choices.append(score)
    if not choices:
        return None
    # Score IDs can differ by redundant review record, but score must agree.
    pairs={(s[0],s[1]) for s in choices}
    if len(pairs)!=1:
        return None
    return choices[0]


def _ats_at_original_price(d,score):
    if not score or not d.get('price_verified') or d.get('side_relative') not in ('HOME','AWAY'):
        return None
    ln = _float(d.get('quote_line'))
    if ln is None:
        return None
    margin = (score[0]-score[1] if d['side_relative']=='HOME' else score[1]-score[0]) + ln
    return 'WIN' if margin > .00001 else 'LOSS' if margin < -.00001 else 'PUSH'


def audit_core_miner_linkage(decisions,scores,*,model_context=(),conflict_ids=(),flipped_games=()):
    """Read-only deterministic identity join and status diagnostics.

    Priced comparisons are separate from model-only context. Broad alias
    agreement is *not* independently verified fixture identity: kickoff or
    identical source game id is required. Missing/no-overlap cases are recorded
    rather than fixing unknown pairs by fuzzy matching.
    """
    conflicts=set(conflict_ids or ())
    flipped=set(flipped_games or ())
    base=collections.defaultdict(lambda:{'core':[],'miner':[],'model':[]})
    ignored=collections.Counter()
    for d in list(decisions)+list(model_context):
        if not isinstance(d,dict): continue
        if d.get('decision_id') in conflicts: ignored['CONFLICTED_IMMUTABLE_SYSTEM_DECISION']+=1;continue
        if d.get('game_key') in flipped: ignored['CORE_SIDE_FLIP_HOLD']+=1;continue
        key=_research_identity(d)
        if key is None:
            ignored['INCOMPLETE_IDENTITY']+=1;continue
        typ=('core' if d.get('family')=='CORE' and d.get('price_verified') and
             d.get('side_relative') in ('HOME','AWAY') else
             'model' if d.get('family')=='CORE' and d.get('record_kind')=='CORE_MODEL_ONLY_NO_PRICE_OR_SELECTION' else
             'miner' if d.get('family')=='MINER' and d.get('price_verified') and
             d.get('side_relative') in ('HOME','AWAY') else None)
        if typ:base[key][typ].append(d)
    counters=collections.Counter()
    joined=[]
    examples=[]
    for key,group in sorted(base.items()):
        c=group['core'];m=group['miner'];mo=group['model']
        if c:counters['priced_core_games']+=1
        if m:counters['priced_miner_games']+=1
        if mo:counters['core_model_only_games']+=1
        if not m:
            if c:counters['priced_core_without_miner_same_game']+=1
            continue
        if not c:
            if mo:
                counters['miner_overlaps_core_model_only']+=1
                if len(mo)!=1:
                    counters['ambiguous_core_model_rows']+=1
                elif any(_safe_same_fixture(mo[0],mm)[0] for mm in m):
                    counters['miner_model_only_fixture_verified']+=1
            else:counters['miner_without_core_price_or_model']+=1
            continue
        counters['priced_core_miner_same_alias_date']+=1
        if len(c)!=1:
            counters['ambiguous_multiple_priced_core_rows']+=1;continue
        core=c[0]
        verified=[x for x in m if _safe_same_fixture(core,x)[0]]
        if not verified:
            counters['priced_overlap_kickoff_mismatch_hold']+=1
            if len(examples)<12:examples.append({'reason':'KICKOFF_MISMATCH_HOLD','key':'|'.join(key)})
            continue
        sides={x['side_relative'] for x in verified}
        if len(sides)!=1:
            counters['mixed_miner_sides_vetoed']+=1
            continue
        counters['priced_core_miner_fixture_verified']+=1
        selected_side=next(iter(sides))
        relation='MINER_CONFIRMS_CORE' if selected_side==core['side_relative'] else 'MINER_OPPOSES_CORE'
        ml=[]
        for mm in verified:
            cl, mline=_float(core.get('quote_line')),_float(mm.get('quote_line'))
            if cl is None or mline is None:continue
            if (abs(cl-mline) <= .01 if core['side_relative']==mm['side_relative']
                else abs(cl+mline) <= .01):ml.append(mm)
        score=_score_for_identity(key,core,verified,scores)
        first=sorted(verified,key=lambda d:(str(d.get('capture_utc')),
                                              str(d.get('signal_id'))))[0]
        gradeable=score is not None
        if ml:
            counters['priced_core_miner_same_original_line']+=1
        else:counters['priced_overlap_different_original_line']+=1
        if gradeable:counters['priced_overlap_provisional_score_review']+=1
        joined.append({'game_key':core['game_key'], 'canonical_game_identity':'|'.join(key),
                       'core_source_game_key':core['game_key'],
                       'miner_source_game_keys':sorted({x.get('game_key') for x in verified}),
                       'link_policy':'EXPLICIT_SCHOOL_ALIASES_AND_KICKOFF_2H_OR_IDENTICAL_ORIGINAL_ID',
                       'fixture_identity_original_attestation':False,
                       'relation':relation,
                       'core_selected_team':core.get('selected_team'),
                       'core_selected_side':core['side_relative'],
                       'core_quote_line':core.get('quote_line'),
                       'miner_original_quote_line':first.get('quote_line'),
                       'different_original_price_can_change_ATS':True,
                       'miner_mechanisms_on_game':len(verified),
                       'distinct_miner_named_family_keys_not_certified_independent':len({
                           x.get('evidence_family_key') or x.get('signal_id') for x in verified}),
                       'same_line_comparisons':len(ml),
                       'same_line_one_result_per_game':True,
                       'same_line_core_ats_provisional':_ats_at_original_price(core,score) if ml else None,
                       'same_line_miner_ats_provisional':_ats_at_original_price(ml[0],score) if ml else None,
                       'original_line_core_ats_provisional':_ats_at_original_price(core,score),
                       'original_line_miner_ats_provisional':_ats_at_original_price(first,score),
                       'score_review_available':gradeable,
                       'finality_certified':False,'bet_authority':0,
                       'research_only':True})
    summary={'status':'PASS_IDENTITY_AUDIT_ONLY','alias_policy':'EXPLICIT_KNOWN_SCHOOL_MASCOT_PAIRS',
             'counts':dict(sorted(counters.items())), 'ignored':dict(ignored),
             'held_examples':examples,'results_are_not_approved_bets':True,
             'model_only_does_not_mean_core_pick':True,'bet_authority':0}
    return joined,summary


def aggregate_core_miner_relationships(decisions,scores,conflict_ids=None):
    """Backward-compatible priced-only interface."""
    items,_=audit_core_miner_linkage(decisions,scores,conflict_ids=conflict_ids)
    return items
