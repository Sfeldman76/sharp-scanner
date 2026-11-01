"""NCAAF V5.5 prospective performance and qualification ledger.

Background-only, fail-closed, research/shadow accounting. This module has NO
code path that grants a bet, rewrites capture or outcome objects, or modifies V1.

Evidence distinction: a capture/quote may be genuine before kickoff; a
prospectively registered rule text may be verifiable; a pair of agreeing scores
is NOT proof of final-game status or of a filled wager. All such performance is
explicitly provisional until a separately audited authority implementation.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import math
from datetime import datetime, timedelta, timezone

from ncaaf_v55_settlement_bridge import key_from_row, score_canonical_key, parse_dt, norm, PREFIX

LOG = logging.getLogger('ncaaf_v55_production_ledger')
ROOT = 'research/ncaaf/v55/production_intelligence_v1'
CONTRACT = 'V55-PROSPECTIVE-IMMUTABLE-SHADOW-NO-AUTHORITY-20261031'
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_OBSERVATION_OBJECTS = 300
MAX_DECISIONS = 15000
MAX_REVIEWS = 3000


def _json_bytes(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _utc(now=None):
    t = parse_dt(now or datetime.now(timezone.utc))
    if t is None:
        raise ValueError('timezone-aware clock required')
    return t


def _iso(t):
    return t.isoformat() if t else None


def _str(x):
    return str(x or '').strip()


def _true(x):
    return x is True  # NEVER trust free-text string 'True' as backend attestation


def _float(v):
    try:
        if v is None or v == '':
            return None
        z = float(v)
        return z if math.isfinite(z) else None
    except (ValueError, TypeError):
        return None


def _read(blob, *, max_bytes=MAX_SOURCE_BYTES):
    size = getattr(blob, 'size', None)
    if size is not None and size > max_bytes:
        raise ValueError('oversize blob')
    raw = blob.download_as_bytes()
    if len(raw) > max_bytes:
        raise ValueError('oversize response')
    return raw, json.loads(raw)


def _put_once(bucket, name, document):
    """Immutable write. Existing record is never altered."""
    blob = bucket.blob(name)
    try:
        blob.upload_from_string(_json_bytes(document), content_type='application/json',
                                if_generation_match=0)
        return 'CREATED'
    except Exception:
        if blob.exists():
            return 'ALREADY_EXISTS'
        raise


def _rule_hash(rule):
    if not isinstance(rule, str) or not rule.strip():
        return None
    # Hash the exact live rule text; never substitute a retrospective rule text.
    return hashlib.sha256(rule.encode('utf-8')).hexdigest()


def stamp_capture_rule_versions(doc, bucket, *, now=None):
    """Bind newly observed live rule text and prospective registry to capture.

    Registration begins with first observation; it NEVER backdates first
    appearance. Later identical rule versions can be tracked prospectively.
    Registry only freezes observed text, NOT selection logic, model inputs,
    frozen historical qualifications, or prediction timing of source signals.
    """
    t = _utc(now or doc.get('captured_at_utc'))
    reg_count = bound = unchanged = 0
    cached = {}
    for e in doc.get('observations', []):
        if not isinstance(e, dict) or _str(e.get('family')).upper() != 'MINER':
            continue
        sid = _str(e.get('signal_id'))
        rule = e.get('rule')
        sha = _rule_hash(rule)
        e.update(live_rule_text_sha256=sha,
                 rule_registration_before_capture=False,
                 historical_rule_vintage_certified=False,
                 qualifying_oos_certified=False,
                 model_and_input_vintage_certified=False)
        if not sid or not sha:
            e['rule_version_hold'] = 'MISSING_LIVE_RULE_TEXT_OR_MECHANISM_ID'
            continue
        cachekey = sid, sha
        if cachekey not in cached:
            # Restrict GCS keys to digest-only suffixes to prevent path injection.
            namespace = hashlib.sha256(sid.encode('utf-8')).hexdigest()[:24]
            path = f'{ROOT}/rule_registry/{namespace}/{sha}.json'
            blob = bucket.blob(path)
            exists_before = blob.exists()
            if not exists_before:
                entry = {'signal_id':sid, 'live_rule_text_sha256':sha,
                         'first_registered_utc':_iso(t),
                         'registry_policy':CONTRACT,
                         'observed_live_rule_text':rule,
                         'historical_rule_vintage_certified':False,
                         'qualifying_oos_certified':False,
                         'bet_authority':0}
                outcome = _put_once(bucket, path, entry)
                if outcome == 'CREATED':
                    reg_count += 1
            # GCS immutable object creation time is the durable time anchor.
            blob.reload()
            created = parse_dt(getattr(blob, 'time_created', None))
            cached[cachekey] = (path, created)
        path, created = cached[cachekey]
        pre_registered = bool(created and created < t)
        e.update(rule_registration_uri=f'gs://{bucket.name}/{path}',
                 rule_registration_created_utc=_iso(created),
                 rule_registration_before_capture=pre_registered,
                 rule_version_hold=None if pre_registered else 'FIRST_SEEN_RULE_VERSION_NOT_PRIOR_REGISTERED')
        bound += int(pre_registered)
        unchanged += 1
    return {'registered_new_rule_versions':reg_count, 'miner_events_with_live_rule_hash':unchanged,
            'miner_events_after_prior_registration':bound, 'bet_authority':0}


def _verify_observation_object(name, raw, obj, created):
    # Capture file basename uses the prefix of sha256(payload), not date only.
    digest = hashlib.sha256(raw).hexdigest()[:20]
    name_ok = name.endswith('_' + digest + '.json')
    receipt = parse_dt(obj.get('captured_at_utc'))
    made = parse_dt(created)
    # Strictly forbid records for which GCS creation appears before receipt.
    return bool(name_ok and receipt and made and receipt <= made), receipt, made


def _event_from_capture(event, obj, *, name, i, receipt, created, observed_now):
    if not isinstance(event, dict):
        return None
    family = _str(event.get('family')).upper()
    sid = _str(event.get('signal_id'))
    if family not in ('MINER','BIG_AL','PATHI','PT','STAT','MARKET_RICH','CORE') or not sid:
        return None
    selected = _str(event.get('selection_side'))
    target = _str(event.get('target'))
    home, away = _str(event.get('home_team')), _str(event.get('away_team'))
    kickoff = parse_dt(event.get('kickoff_utc'))
    quoted = parse_dt(event.get('source_snapshot_utc'))
    stamped = parse_dt(event.get('captured_at_utc'))
    key = key_from_row({'home':home, 'away':away, 'kickoff_utc':event.get('kickoff_utc')})
    direction_ok = bool(target and selected and norm(target) == norm(selected) and
                        norm(selected) in (norm(home), norm(away)))
    market = _str(event.get('market')).lower()
    fmt = _str(event.get('quoted_odds_format')).upper()
    line = _float(event.get('original_book_line'))
    odds = _float(event.get('quoted_odds_raw'))
    odds_ok = (fmt == 'AMERICAN' and odds is not None and
               (odds <= -100 or odds >= 100)) or (fmt == 'DECIMAL' and
               odds is not None and 1.01 <= odds <= 99.99)
    quote_ok = bool(market == 'spreads' and _str(event.get('sportsbook')) and
                    line is not None and odds_ok and
                    _true(event.get('original_quote_line_verified')) and
                    _true(event.get('price_roi_eligible')))
    clock_ok = bool(kickoff and quoted and receipt and created and stamped and
                    quoted <= stamped <= receipt <= created < kickoff and
                    created <= observed_now)
    registry_at = parse_dt(event.get('rule_registration_created_utc'))
    version_ok = bool(family == 'MINER' and event.get('live_rule_text_sha256') and
                      _true(event.get('rule_registration_before_capture')) and registry_at and
                      quoted and registry_at < quoted <= receipt)
    reasons = []
    if not key:reasons.append('GAME_IDENTITY_INCOMPLETE')
    if not direction_ok:reasons.append('DIRECTION_NOT_CONFIRMED')
    if not quote_ok:reasons.append('QUOTE_NOT_CONFIRMED')
    if not clock_ok:reasons.append('PREKICKOFF_CLOCK_NOT_CONFIRMED')
    if family == 'PT' and target == '':reasons.append('NON_DIRECTIONAL_FORECAST_NOT_BET_SIGNAL')
    if family == 'MINER' and not version_ok:reasons.append('RULE_VERSION_NOT_PRE_REGISTERED')
    if family != 'MINER':reasons.append('EXTERNAL_BRAIN_RULE_AND_SOURCE_VINTAGE_NOT_CERTIFIED')
    # No blanket assertion that prospective timestamp proves source-input provenance.
    if not _true(event.get('model_and_input_vintage_certified')):
        reasons.append('SOURCE_INPUTS_NOT_CERTIFIED_PRE_KICKOFF')
    return {
        'game_key':'|'.join(key) if key else None,
        'original_physical_game_id':_str(event.get('physical_game_id')) or None,
        'family':family, 'signal_id':sid,
        'evidence_family_key':_str(event.get('evidence_family_key')) or None,
        'home_team':home,'away_team':away,'selected_team':selected,'target':target,
        'side_relative':('HOME' if norm(selected)==norm(home) else
                         'AWAY' if norm(selected)==norm(away) else 'UNKNOWN'),
        'kickoff_utc':_iso(kickoff), 'capture_utc':_iso(receipt),
        'object_created_utc':_iso(created), 'quote_utc':_iso(quoted),
        'quote_book':_str(event.get('sportsbook')),
        'quote_line':line, 'quote_odds':odds, 'quote_odds_format':fmt,
        'source_observation_path':name,'source_observation_index':i,
        'live_rule_text_sha256':event.get('live_rule_text_sha256'),
        'rule_registration_created_utc':event.get('rule_registration_created_utc'),
        'direction_verified':direction_ok, 'price_verified':quote_ok,
        'capture_clock_verified':clock_ok,
        'rule_version_registered_before_capture':version_ok,
        'source_input_vintage_certified':False, # independent requirement, never assumed
        'fully_qualified_for_betting':False,
        'holds':sorted(set(reasons)),
        'bet_authority':0,
    }


def _id(row):
    # One immutable directional ledger record per mechanism and game; books do
    # not create independent bets. Distinct brains are separate evidence families.
    return hashlib.sha256((row['game_key']+'|'+row['family']+'|'+row['signal_id']).encode()).hexdigest()[:32]


def select_first_decisions(events):
    groups = collections.defaultdict(list)
    for x in events:
        if x['game_key'] and x['selected_team']:
            groups[_id(x)].append(x)
    decisions = []
    for ident, rows in sorted(groups.items()):
        rows.sort(key=lambda x: (x['capture_utc'] or 'Z', x['quote_utc'] or 'Z',
                                 x['source_observation_path'], x['source_observation_index']))
        sides = {norm(x['selected_team']) for x in rows if x['direction_verified']}
        # Earliest eligible *quote evidence* only; no look-ahead on score or price.
        acceptable = [x for x in rows if x['price_verified'] and x['capture_clock_verified']
                      and x['direction_verified']]
        preregistered = [x for x in acceptable if x['family'] != 'MINER' or
                         x['rule_version_registered_before_capture']]
        first = (preregistered[0] if preregistered else
                 acceptable[0] if acceptable else rows[0])
        doc = dict(first)
        doc.update(decision_id=ident, initial_capture_utc=rows[0]['capture_utc'],
                   first_qualified_quote_utc=first['capture_utc'] if acceptable else None,
                   quote_count=len(rows),
                   conflicting_directions=len(sides)>1,
                   independent_bet_count=0, wager_filled=False,
                   actual_executable_price_certified=False,
                   finality_certified=False, settlement_certified=False,
                   authority_status='SHADOW_ONLY_NO_BET_AUTHORITY',
                   authority_gates=['PROSPECTIVE_OOS_NOT_CERTIFIED','NO_EXECUTED_WAGER',
                                    'INDEPENDENT_SOURCE_VINTAGE_NOT_CERTIFIED',
                                    'OFFICIAL_FINALITY_NOT_CERTIFIED'],
                   policy=CONTRACT,bet_authority=0)
        if len(sides)>1:
            doc['holds'] = sorted(set(doc['holds']+['CONFLICTING_SIDES_SAME_SYSTEM_GAME']))
        decisions.append(doc)
    return decisions


def _review_index(refs):
    bykey = collections.defaultdict(list)
    for r in refs:
        if r.get('state') != 'DUAL_SOURCE_SCORE_AGREEMENT_REVIEW_ONLY':
            continue
        k = r.get('stable_game_key')
        home, away = _float(r.get('home_score')), _float(r.get('away_score'))
        if not k or home is None or away is None or home < 0 or away < 0:
            continue
        bykey[k].append((home, away, r.get('cfbd_game_id')))
    return {k: v[0] for k, v in bykey.items() if len(set(v)) == 1}


def reconcile_review_scores(reviews, decisions, *, max_kickoff_delta_minutes=120):
    """Read-only alias-aware lookup with independent fixture-clock check.

    Do not change original CORE or Miner decision IDs. A single old/v2 review
    can grade both keys only if its original captured fixture is unambiguous,
    the dual-score review agrees, and the independently captured kickoff is
    within the configured tolerance. This never attests official finality.
    """
    fixture_scores=collections.defaultdict(list)
    rejected=collections.Counter()
    for r in reviews:
        if not isinstance(r,dict) or r.get('state')!='DUAL_SOURCE_SCORE_AGREEMENT_REVIEW_ONLY':
            continue
        if r.get('finality_certified') is True or r.get('settlement_certified') is True:
            # This reader never upgrades certificate status from a shadow review.
            rejected['MISLABELED_OFFICIAL_REVIEW_HOLD']+=1;continue
        if r.get('score_source') != 'ncaaf_cfbd_postgame_finals_v1+game_scores_final':
            rejected['SOURCE_PROVENANCE_HOLD']+=1;continue
        key=score_canonical_key({'home':r.get('captured_home_team'),
             'away':r.get('captured_away_team'),
             'kickoff_utc':r.get('captured_kickoff_utc')})
        original=key_from_row({'home':r.get('captured_home_team'),
             'away':r.get('captured_away_team'),
             'kickoff_utc':r.get('captured_kickoff_utc')})
        if not key or not original or '|'.join(original)!=r.get('stable_game_key'):
            rejected['REVIEW_CAPTURED_IDENTITY_HOLD']+=1;continue
        h=_float(r.get('home_score'));a=_float(r.get('away_score'))
        cf=_str(r.get('cfbd_game_id'))
        if not cf or h is None or a is None or h<0 or a<0 or not h.is_integer() or not a.is_integer():
            rejected['REVIEW_INVALID_SCORE_OR_GAME_ID_HOLD']+=1;continue
        date=parse_dt(r['captured_kickoff_utc'])
        fixture_scores[key].append(((h,a,cf),date,r['stable_game_key']))
    verified={}; held_fixture=set()
    for fixture,items in fixture_scores.items():
        # Agreement of old+new reviews cannot count as two score sources;
        # both reviews already contain their own source attestations.
        vals={row[0] for row in items}
        if len(vals)!=1:
            held_fixture.add(fixture)
            rejected['CONFLICTING_REVIEW_SCORES_OR_GAME_IDS_HOLD']+=1
            continue
        verified[fixture]=(next(iter(vals)),items)
    mapped={}; blocked_keys=set();linked=0;alias_linked=0
    for d in decisions:
        if not isinstance(d,dict) or not d.get('game_key') or not d.get('decision_id'):
            continue
        if d.get('game_key') in blocked_keys or d.get('conflicting_directions'):
            rejected['CONFLICTED_DECISION_HOLD']+=1;continue
        if not d.get('price_verified') or not d.get('capture_clock_verified') or not d.get('direction_verified'):
            rejected['DECISION_NOT_ELIGIBLE_FOR_SHADOW_GRADING']+=1;continue
        fixture=score_canonical_key({'home':d.get('home_team'),
                'away':d.get('away_team'),'kickoff_utc':d.get('kickoff_utc')})
        decision_clock=parse_dt(d.get('kickoff_utc'))
        if not fixture or not decision_clock:
            rejected['DECISION_GAME_IDENTITY_HOLD']+=1;continue
        if fixture in held_fixture:
            rejected['REVIEW_CONFLICT_HOLD']+=1;continue
        if fixture not in verified:
            rejected['NO_DUAL_SOURCE_REVIEW_FOR_FIXTURE']+=1;continue
        score,records=verified[fixture]
        # A date-only or alias-only match is not enough for a wager outcome.
        if not any(abs((decision_clock-r[1]).total_seconds())<=max_kickoff_delta_minutes*60
                   for r in records):
            rejected['KICKOFF_TIME_MISMATCH_HOLD']+=1;continue
        old=mapped.get(d['game_key'])
        if old is not None and old!=score:
            mapped.pop(d['game_key'],None)
            blocked_keys.add(d['game_key'])
            rejected['DECISIONS_SAME_GAME_KEY_DISAGREE_ON_SCORE_HOLD']+=1
            continue
        mapped[d['game_key']]=score
        linked+=1
        if all(x[2]!=d['game_key'] for x in records):alias_linked+=1
    summary={'review_objects_seen':len(reviews), 'unique_verified_score_fixtures':len(verified),
             'review_fixture_conflicts':len(held_fixture),
             'decisions_linked_to_reviews':linked,
             'decisions_linked_by_explicit_alias':alias_linked,
             'scored_original_game_keys':len(mapped),
             'holds':dict(sorted(rejected.items())),
             'source_lineage_independence_certified':False,
             'official_finality_certified':False,'bet_authority':0}
    return mapped,summary


def _net_win(odds, fmt):
    if fmt == 'AMERICAN':
        return odds/100 if odds >= 100 else 100/abs(odds)
    if fmt == 'DECIMAL':
        return odds-1
    raise ValueError('unverified odds format')


def provisional_result(decision, score):
    h, a, cfbd = score
    side = decision['side_relative']
    line = decision['quote_line']
    if side not in ('HOME','AWAY') or line is None or not decision['price_verified']:
        return None
    margin = (h-a if side == 'HOME' else a-h) + line
    result = 'WIN' if margin > 0 else 'LOSS' if margin < 0 else 'PUSH'
    units = (_net_win(decision['quote_odds'],decision['quote_odds_format']) if result=='WIN'
             else -1.0 if result=='LOSS' else 0.0)
    return {'decision_id':decision['decision_id'], 'game_key':decision['game_key'],
            'family':decision['family'], 'signal_id':decision['signal_id'],
            'cfbd_game_id_from_score_review':cfbd,
            'selected_team':decision['selected_team'],
            'review_home_score':h,'review_away_score':a,
            'historical_research_ats_result':result,
            'provisional_unit_pnl':units, 'provisional_roi_only':True,
            'line_and_odds_original_capture':True,
            'finality_certified':False, 'settlement_certified':False,
            'official_final_status_evidence':'NOT_ESTABLISHED',
            'wager_filled':False, 'bet_authority':0,
            'policy':CONTRACT}


def _find_json(storage, bucket, prefix, *, limit, errors):
    results=[]
    for b in storage.list_blobs(bucket, prefix=prefix, max_results=limit+1):
        if len(results)>=limit:
            errors.append('SCAN_CAP_REACHED:'+prefix)
            break
        try:
            _, obj = _read(b, max_bytes=3_000_000)
            if isinstance(obj,dict):results.append(obj)
        except Exception as exc:
            errors.append(f'INVALID_JSON:{b.name}:{type(exc).__name__}')
    return results


def run_cycle(*, storage_client=None, bucket_name='sharp-models', now=None,
              in_streamlit=False, days_back=2):
    """Existing background hook. Never writes V1, production board or bets.

    Score source is the already-existing dual-source review bridge. Outcome
    assessment remains explicitly preliminary until independently certified.
    """
    if in_streamlit:
        return {'status':'SKIPPED_INTERACTIVE_UI','bet_authority':0}
    t = _utc(now)
    if storage_client is None:
        from google.cloud import storage
        storage_client=storage.Client()
    bucket=storage_client.bucket(bucket_name)
    errors=[]; observed=[]; invalid_objects=0
    # CORE-V55-PUBLISHED-JOURNAL: read the COMPLETE+SHA-verified backend board,
    # never infer a CORE pick from other brains or reconstruct after a result.
    # In this deployment the board is published after the research pass, so
    # the latest complete publication is ordinarily the PREVIOUS refresh.
    core_capture={'status':'HOLD_UNAVAILABLE','bet_authority':0}
    try:
        from google.cloud import bigquery as _bigquery
        from ncaaf_v55_core_shadow import capture_board
        core_capture=capture_board(bq_client=_bigquery.Client(project='sharplogger'),
                                   bucket=bucket,now=t)
        if core_capture.get('status') != 'PASS':
            errors.append('CORE_BOARD_CAPTURE_PARTIAL_HOLD')
        LOG.info('[NCAAF-V55-CORE-SOURCE] %s',json.dumps(core_capture,sort_keys=True))
    except Exception as exc:
        errors.append('CORE_BOARD_CAPTURE_HOLD:'+type(exc).__name__)
        LOG.warning('[NCAAF-V55-CORE-SOURCE] HOLD %s',type(exc).__name__)
    days = min(3, max(1,int(days_back)))
    for d in range(days+1):
        day=(t-timedelta(days=d)).strftime('%Y-%m-%d')
        prefix=f'{PREFIX}/observations/{day}/'
        for index, blob in enumerate(storage_client.list_blobs(
                bucket_name,prefix=prefix,max_results=MAX_OBSERVATION_OBJECTS+1)):
            if index >= MAX_OBSERVATION_OBJECTS:
                errors.append('OBSERVATION_OBJECT_SCAN_CAP_REACHED:'+day)
                break
            if len(observed) > MAX_OBSERVATION_OBJECTS*1000:
                errors.append('ROW_SCAN_CAP_REACHED');break
            try:
                raw,obj=_read(blob)
                verified, receipt, made=_verify_observation_object(blob.name,raw,obj,blob.time_created)
                if not verified or not isinstance(obj.get('observations'),list):
                    invalid_objects+=1;continue
                for i,e in enumerate(obj['observations']):
                    if not isinstance(e,dict):continue
                    event=_event_from_capture(e,obj,name=blob.name,i=i,receipt=receipt,
                                              created=made,observed_now=t)
                    if event:observed.append(event)
            except Exception as exc:
                errors.append(f'OBSERVATION_READ_ERROR:{blob.name}:{type(exc).__name__}')
    proposed=select_first_decisions(observed)
    created=existing=0
    for d in proposed:
        # Only register a valid observed priced directional event, never a
        # retrospective replay or an implied PT forecast.
        if not d['price_verified'] or not d['capture_clock_verified'] or not d['direction_verified']:
            continue
        path=f'{ROOT}/shadow_decisions_v1/{d["decision_id"]}.json'
        try:
            state=_put_once(bucket,path,d)
            created+=int(state=='CREATED')
            existing+=int(state=='ALREADY_EXISTS')
        except Exception as exc:
            errors.append(f'DECISION_WRITE_ERROR:{d["decision_id"]}:{type(exc).__name__}')
    decisions=_find_json(storage_client,bucket_name,f'{ROOT}/shadow_decisions_v1/',
                         limit=MAX_DECISIONS,errors=errors)
    # Immutable originally published CORE research leans, locked separately
    # from captured external brain events to avoid any retrospective merging.
    core_decisions=_find_json(storage_client,bucket_name,f'{ROOT}/core_first_priced_v1/',
                              limit=MAX_DECISIONS,errors=errors)
    decisions.extend(d for d in core_decisions if d.get('family')=='CORE'
                     and d.get('price_verified') and d.get('decision_id'))
    core_models=_find_json(storage_client,bucket_name,f'{ROOT}/core_model_first_seen_v1/',
                           limit=MAX_DECISIONS,errors=errors)
    reviews=_find_json(storage_client,bucket_name,f'{PREFIX}/score_reviews_v1/',
                       limit=MAX_REVIEWS,errors=errors)
    reviews.extend(_find_json(storage_client,bucket_name,f'{PREFIX}/score_reviews_v2/',
                       limit=MAX_REVIEWS,errors=errors))
    disputes=_find_json(storage_client,bucket_name,f'{PREFIX}/score_review_disputes_v1/',
                       limit=MAX_REVIEWS,errors=errors)
    disputed_sha={x.get('original_game_index_sha256') for x in disputes
                  if x.get('state')=='DISPUTED_REVIEW_SCORE_HOLD' and x.get('original_game_index_sha256')}
    reviews=[x for x in reviews if x.get('original_game_index_sha256') not in disputed_sha]
    scores,score_link_diagnostics=reconcile_review_scores(reviews,decisions)
    score_link_diagnostics['disputed_original_game_indexes_quarantined']=len(disputed_sha)
    # A previously captured direction cannot be silently replaced if a later
    # refresh produces its opposite. Persist a separate immutable veto record.
    for d in proposed:
        if not d.get('conflicting_directions'):
            continue
        try:
            _put_once(bucket,f'{ROOT}/conflicts_v1/{d["decision_id"]}.json',
                      {'decision_id':d['decision_id'],'game_key':d['game_key'],
                       'family':d['family'],'signal_id':d['signal_id'],
                       'status':'CONFLICT_VETO_REQUIRES_REVIEW',
                       'bet_authority':0,'policy':CONTRACT})
        except Exception as exc:
            errors.append('CONFLICT_VETO_WRITE_ERROR:'+type(exc).__name__)
    conflicts=_find_json(storage_client,bucket_name,f'{ROOT}/conflicts_v1/',
                         limit=MAX_DECISIONS,errors=errors)
    conflict_ids={c.get('decision_id') for c in conflicts}
    # A published CORE side flip is tracked but must not masquerade as an
    # independent, stable comparison against Miner on that game.
    core_flips=_find_json(storage_client,bucket_name,f'{ROOT}/core_side_flips_v1/',
                         limit=MAX_DECISIONS,errors=errors)
    flipped_games={x.get('game_key') for x in core_flips if x.get('game_key')}
    outcome_created=outcome_existing=0
    provisional=[]
    for d in decisions:
        if not isinstance(d,dict) or not d.get('decision_id') or not d.get('game_key'):
            continue
        if d.get('decision_id') in conflict_ids:
            continue
        score=scores.get(d['game_key'])
        if score is None:continue
        result=provisional_result(d,score)
        if not result:continue
        provisional.append(result)
        try:
            state=_put_once(bucket,f'{ROOT}/provisional_outcomes_v2/{d["decision_id"]}.json',result)
            outcome_created+=int(state=='CREATED')
            outcome_existing+=int(state=='ALREADY_EXISTS')
        except Exception as exc:
            errors.append(f'PROVISIONAL_WRITE_ERROR:{d["decision_id"]}:{type(exc).__name__}')
    # CORE vs Miner comparisons are per GAME, not per mechanism firing;
    # historical/review-only scores cannot generate authorized betting proof.
    comparison=[]
    crosswalk_audit={'status':'HOLD_NOT_EVALUATED','counts':{}}
    try:
        from ncaaf_v55_core_shadow import audit_core_miner_linkage
        comparison,crosswalk_audit=audit_core_miner_linkage(
            decisions,scores,model_context=core_models,
            conflict_ids=conflict_ids,flipped_games=flipped_games)
        LOG.info('[NCAAF-V55-CORE-MINER-LINK-AUDIT] %s',
                 json.dumps(crosswalk_audit,sort_keys=True))
        for item in comparison:
            if not item['score_review_available']:
                continue
            identity=hashlib.sha256(item['canonical_game_identity'].encode()).hexdigest()[:32]
            try:
                # Versioned output; never re-write old V1 comparison semantics.
                _put_once(bucket,f'{ROOT}/core_miner_provisional_v3/{identity}.json',item)
            except Exception as exc:
                errors.append('CORE_MINER_COMPARISON_V2_WRITE:'+type(exc).__name__)
    except Exception as exc:
        errors.append('CORE_MINER_COMPARISON_HOLD:'+type(exc).__name__)
        LOG.exception('[NCAAF-V55-CORE-MINER-LINK-AUDIT] ERROR')
    ledger=collections.defaultdict(lambda:{'unique_system_game_decisions':0,'provisional_wins':0,
                                           'provisional_losses':0,'provisional_pushes':0,
                                           'provisional_units':0.0})
    for d in decisions:
        if d.get('decision_id'):
            ledger[_str(d.get('family'))]['unique_system_game_decisions']+=1
    for r in provisional:
        l=ledger[r['family']]
        key={'WIN':'provisional_wins','LOSS':'provisional_losses','PUSH':'provisional_pushes'}[r['historical_research_ats_result']]
        l[key]+=1
        l['provisional_units']+=r['provisional_unit_pnl']
    mechanisms=collections.defaultdict(lambda:{'observed_shadow_decisions':0,
                                                'pre_registered_rule_decisions':0,
                                                'provisional_wins':0,'provisional_losses':0,
                                                'provisional_pushes':0,'provisional_units':0.0,
                                                'home_decisions':0,'away_decisions':0})
    for d in decisions:
        if not d.get('decision_id'):continue
        k=f"{d.get('family')}:{d.get('signal_id')}"
        m=mechanisms[k];m['observed_shadow_decisions']+=1
        m['pre_registered_rule_decisions']+=int(bool(d.get('rule_version_registered_before_capture')))
        if d.get('side_relative')=='HOME':m['home_decisions']+=1
        if d.get('side_relative')=='AWAY':m['away_decisions']+=1
    for r in provisional:
        k=f"{r.get('family')}:{r.get('signal_id')}"
        mechanisms[k][{'WIN':'provisional_wins','LOSS':'provisional_losses','PUSH':'provisional_pushes'}[r['historical_research_ats_result']]]+=1
        mechanisms[k]['provisional_units']+=r['provisional_unit_pnl']
    report={'status':'SHADOW_LEDGER_RUNNING' if not errors else 'PARTIAL_HOLD_WITH_ERRORS',
            'policy':CONTRACT,'generated_utc':_iso(t),'bet_authority':0,
            'authorized_bets':0,'production_bet_instructions':[],
            'observer_events_in_recent_window':len(observed),
            'candidate_system_game_pairs_in_window':len(proposed),
            'new_immutable_shadow_decisions':created,
            'existing_shadow_decisions_in_window':existing,
            'durable_shadow_decisions':len(decisions),
            'conflicted_system_games_vetoed':len(conflict_ids),
            'dual_source_review_game_keys':len(scores),
            'score_review_link_diagnostics':score_link_diagnostics,
            'provisional_outcomes_new':outcome_created,
            'provisional_outcomes_preexisting':outcome_existing,
            'provisional_outcomes_count':len(provisional),
            'performance_by_family_provisional_only':dict(ledger),
            'performance_by_mechanism_provisional_only':dict(mechanisms),
            'core_pregame_decision_ledger_integrated':core_capture.get('status')=='PASS',
            'core_source_capture':core_capture,
            'immutable_core_decisions':len(core_decisions),
            'core_model_only_first_seen':len(core_models),
            'core_miner_identity_audit':crosswalk_audit,
            'core_miner_game_level_research_comparisons':len(comparison),
            'core_miner_games_with_score_reviews':sum(bool(i['score_review_available']) for i in comparison),
            'miner_home_only_diagnostics':[key for key,v in mechanisms.items()
              if key.startswith('MINER:') and v['home_decisions'] and not v['away_decisions']],
            'source_finality_certified':False,
            'qualified_systems':0,'auto_promotion_enabled':False,
            'qualification_state':'HOLD_EVIDENCE_NOT_YET_SUFFICIENT',
            'performance_metrics_are_descriptive_not_authority':True,
            'source_code_cannot_grant_bet_authority':True,
            'errors':errors[:25],'errors_count':len(errors),
            'qualification_holds':['OFFICIAL_FINALITY_AND_SOURCE_TIMING_UNCERTIFIED',
                                   'PROSPECTIVE_INDEPENDENT_OOS_NOT_YET_ESTABLISHED',
                                   'NO_VERIFIED_EXECUTED_WAGER_PRICE_OR_SLIPPAGE',
                                   'MANUAL_APPROVAL_REQUIRED_FOR_ANY_AUTHORITY_CHANGE']}
    # Read-only V5.5 data-flow audit: detects missing/duplicate/misrouted data.
    # Never changes CORE, brains, scores, qualifications or wager authority.
    try:
        from ncaaf_v55_pipeline_health import run_cycle as _audit_pipeline_health
        try:
            from google.cloud import bigquery as _audit_bq
            _audit_client=_audit_bq.Client(project='sharplogger')
        except Exception:
            _audit_client=None
        _health=_audit_pipeline_health(storage_client=storage_client,
            bucket_name=bucket_name,ledger_report=report,bq_client=_audit_client,
            now=t,persist=True)
        report['pipeline_health_audit']={
            'status':_health.get('status'),
            'production_status':_health.get('production_status'),
            'research_status':_health.get('research_status'),
            'production_check_counts':_health.get('production_check_counts',{}),
            'research_check_counts':_health.get('research_check_counts',{}),
            'check_counts':_health.get('check_counts',{}),
            'failure_codes':_health.get('failure_codes',[]),
            'warning_codes':_health.get('warning_codes',[]),
            'unknown_codes':_health.get('unknown_codes',[]),
            'report_uri':_health.get('report_uri')}
    except Exception as exc:
        # The audit is strictly an observer, never a reason to rewrite a ledger
        # or to convert an unavailable audit into a successful verification.
        report['pipeline_health_audit']={
            'status':'FAIL','error_type':type(exc).__name__,
            'failure_codes':['PIPELINE_AUDIT_EXECUTION_FAILED']}
        LOG.exception('[NCAAF-V55-PIPELINE-HEALTH] audit execution failed')
    run_key=t.strftime('%Y%m%dT%H%M%S%fZ')
    # All writes are research namespace only; latest is an observability pointer.
    try:
        _put_once(bucket,f'{ROOT}/reports/{run_key}.json',report)
        bucket.blob(f'{ROOT}/latest.json').upload_from_string(
            _json_bytes(report),content_type='application/json')
    except Exception as exc:
        report['status']='PUBLICATION_ERROR'
        report['errors'].append('REPORT_WRITE:'+type(exc).__name__)
    LOG.info('[NCAAF-V55-PRODUCTION-LEDGER] %s',json.dumps({k:v for k,v in report.items() if k not in ('performance_by_family_provisional_only','performance_by_mechanism_provisional_only')},sort_keys=True))
    return report
