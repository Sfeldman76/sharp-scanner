"""NCAAF V5.5 permanent, read-only data-flow health audit (2026-11-01).

OBSERVABILITY ONLY: Never produces predictions, changes features, relaxes gates,
settles outcomes, trains models, publishes betting instructions or grants authority.
Checks distinguish a demonstrated defect (FAIL), incomplete coverage (WARNING),
and missing verification (UNKNOWN). Absence of a signal is not automatically a bug.

This is a per-refresh evidence audit, not a claim that every historical feature
has been compared row-by-row. Reports disclose exactly what was checked.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
import re

LOG = logging.getLogger('ncaaf_v55_pipeline_health')
PREFIX = 'research/ncaaf/v55'
OUT = f'{PREFIX}/pipeline_health_v1'
CAPTURE = f'{PREFIX}/prospective_brain_evidence_v1/observations/'
REQUIRED_WAREHOUSE = (
    'ncaaf_cfbd_archive_research_v1', 'ncaaf_cfbd_miner_features_v1',
    'ncaaf_game_travel_reference', 'ncaaf_team_season_personnel',
    'ncaaf_player_season_history', 'moves_with_features_merged',
    'bigal_game_context', 'sharp_moves_master', 'ncaaf_v55_live_board',
    'game_scores_final', 'ncaaf_cfbd_postgame_finals_v1',
)
FAMILIES = ('CORE','MINER','PATHI','BIG_AL','PT','STAT','MARKET_RICH')
MAX_CAPTURE_BYTES = 12_000_000


def _dt(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if not value: return None
    try:
        d=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _str(value):
    if value is None:return ''
    s = str(value).strip()
    return '' if s.lower() in ('nan','none','null','<na>') else s


def _num(value):
    try:
        z=float(value)
        return z if math.isfinite(z) else None
    except (ValueError, TypeError):
        return None


def _json(doc):
    return json.dumps(doc,sort_keys=True,separators=(',',':'),allow_nan=False,default=str)


def add_check(report, code, severity, reason, *, metrics=None, evidence=None, domain="production"):
    assert severity in ('PASS','WARNING','FAIL','UNKNOWN')
    entry={'code':code,'status':severity,'domain':domain,'reason':reason,
           'metrics':metrics or {}}
    if evidence is not None: entry['evidence']=evidence
    report['checks'].append(entry)
    return entry


def summarize(report):
    counts=Counter(x['status'] for x in report['checks'])
    report['check_counts']={k:counts.get(k,0) for k in ('PASS','WARNING','FAIL','UNKNOWN')}
    report['status']='FAIL' if counts['FAIL'] else ('WARNING' if counts['WARNING'] else
                      'UNKNOWN' if counts['UNKNOWN'] else 'PASS')
    report['failure_codes']=[x['code'] for x in report['checks'] if x['status']=='FAIL']
    report['warning_codes']=[x['code'] for x in report['checks'] if x['status']=='WARNING']
    report['unknown_codes']=[x['code'] for x in report['checks'] if x['status']=='UNKNOWN']
    for domain in ('production','research'):
        subset=[x for x in report['checks'] if x.get('domain','production')==domain]
        cc=Counter(x['status'] for x in subset)
        report[f'{domain}_check_counts']={k:cc.get(k,0) for k in ('PASS','WARNING','FAIL','UNKNOWN')}
        report[f'{domain}_status']=('FAIL' if cc['FAIL'] else 'WARNING' if cc['WARNING'] else
                                   'UNKNOWN' if cc['UNKNOWN'] or not subset else 'PASS')
        report[f'{domain}_failure_codes']=[x['code'] for x in subset if x['status']=='FAIL']
    return report


def _latest_capture(storage, bucket_name, now):
    """Latest actual immutable snapshot (no new capture; no future observations)."""
    blobs=[]
    for d in (0,1):
        prefix=f'{CAPTURE}{(now-timedelta(days=d)).strftime("%Y-%m-%d")}/'
        blobs.extend(storage.list_blobs(bucket_name,prefix=prefix,max_results=150))
    blobs=[b for b in blobs if _dt(getattr(b,'time_created',None)) and
           _dt(getattr(b,'time_created',None)) <= now]
    if not blobs:return None, 'NO_CAPTURE_OBJECT_ON_CURRENT_OR_PREVIOUS_UTC_DAY'
    b=max(blobs,key=lambda b:_dt(b.time_created))
    if getattr(b,'size',0) and b.size > MAX_CAPTURE_BYTES:
        return None,'LATEST_CAPTURE_EXCEEDS_SIZE_CAP'
    raw=b.download_as_bytes()
    if len(raw)>MAX_CAPTURE_BYTES:return None,'LATEST_CAPTURE_EXCEEDS_SIZE_CAP'
    if not b.name.endswith('_'+hashlib.sha256(raw).hexdigest()[:20]+'.json'):
        return None,'LATEST_CAPTURE_CONTENT_HASH_MISMATCH'
    doc=json.loads(raw)
    if not isinstance(doc,dict) or not isinstance(doc.get('observations'),list):
        return None,'LATEST_CAPTURE_INVALID_SCHEMA'
    return {'object_name':b.name,'time_created_utc':b.time_created.isoformat(),
            'capture':doc},None


def audit_capture(report, latest, now):
    if latest is None:
        add_check(report,'BRAIN_CAPTURE_PRESENT','UNKNOWN','No verified capture available to inspect')
        return
    doc=latest['capture']; events=doc['observations']
    counts=Counter(_str(e.get('family')).upper() for e in events if isinstance(e,dict))
    issues=Counter();unique=set();game_keys=defaultdict(set)
    priced=0; physical=0; source_fields=doc.get('source_signal_fields_present') or {}
    for e in events:
        if not isinstance(e,dict):issues['NON_DICT_EVENT']+=1;continue
        fam=_str(e.get('family')).upper()
        home=_str(e.get('home_team'));away=_str(e.get('away_team'))
        target=_str(e.get('target')); selected=_str(e.get('selection_side'))
        kickoff=_dt(e.get('kickoff_utc')); quote=_dt(e.get('source_snapshot_utc'))
        captured=_dt(e.get('captured_at_utc')); issued=_dt(doc.get('captured_at_utc'))
        if e.get('physical_game_id') and _str(e.get('physical_game_id')):physical+=1
        else:issues['PHYSICAL_GAME_ID_MISSING']+=1
        if e.get('price_roi_eligible') is True:priced+=1
        else:issues['PRICE_NOT_VERIFIED']+=1
        if fam not in FAMILIES:issues['INVALID_FAMILY']+=1
        if not home or not away or not _str(e.get('signal_id')):issues['MISSING_TEAM_OR_SIGNAL_ID']+=1
        if not kickoff or not captured or not issued or captured >= kickoff or issued >= kickoff:
            issues['NOT_VERIFIED_PREGAME']+=1
        if not quote or not captured or quote>captured:
            issues['INVALID_QUOTE_TIME']+=1
        if not target or not selected or target.casefold()!=selected.casefold():
            issues['TARGET_SELECTION_DISAGREEMENT']+=1
        logical=(_str(e.get('physical_game_id')) or f'{away.casefold()}@{home.casefold()}@{kickoff}',
                 fam,_str(e.get('signal_id')),_str(e.get('sportsbook')),
                 selected.casefold(),_str(e.get('original_book_line')),_str(e.get('quoted_odds_raw')),
                 _str(e.get('source_snapshot_utc')))
        if logical in unique:issues['DUPLICATED_EVENT_WITHIN_ONE_CAPTURE']+=1
        unique.add(logical)
        gk=(away.casefold(),home.casefold(),kickoff,fam,_str(e.get('signal_id')))
        game_keys[gk].add(selected.casefold())
    issues['CONFLICTING_DIRECTIONS_WITHIN_ONE_CAPTURE']=sum(len(sides)>1 for sides in game_keys.values())
    declared_counts=doc.get('source_counts')
    if isinstance(declared_counts,dict):
        for family,count in declared_counts.items():
            if int(count) != counts.get(_str(family).upper(),0):
                issues['SOURCE_COUNT_MANIFEST_MISMATCH']+=1
    elif declared_counts is not None:
        issues['INVALID_SOURCE_COUNTS_MANIFEST']+=1
    bad=sum(issues[k] for k in ('NON_DICT_EVENT','INVALID_FAMILY','MISSING_TEAM_OR_SIGNAL_ID',
             'NOT_VERIFIED_PREGAME','INVALID_QUOTE_TIME','TARGET_SELECTION_DISAGREEMENT',
             'DUPLICATED_EVENT_WITHIN_ONE_CAPTURE','CONFLICTING_DIRECTIONS_WITHIN_ONE_CAPTURE',
             'SOURCE_COUNT_MANIFEST_MISMATCH','INVALID_SOURCE_COUNTS_MANIFEST'))
    add_check(report,'BRAIN_CAPTURE_INTEGRITY','FAIL' if bad else 'PASS',
              'Proven malformed/duplicate/conflicting source events' if bad else
              'Latest immutable capture passes per-event structural and timestamp checks',
              metrics={'event_rows':len(events),'unique_event_fingerprints':len(unique),
                       'events_by_family':dict(counts),'issues':dict(issues),
                       'artifact':latest['object_name']})
    if not events:
        add_check(report,'BRAIN_CAPTURE_COVERAGE','WARNING','Capture contains no brain events; may be a legitimate no-signal refresh')
    if 0<physical<len(events):
        add_check(report,'PHYSICAL_GAME_ID_COVERAGE','WARNING',
                  'Some events lack a stable physical game ID; score joins may require explicit fixture crosswalk',
                  metrics={'with_physical_game_id':physical,'events':len(events)})
    elif events and physical==0:
        add_check(report,'PHYSICAL_GAME_ID_COVERAGE','WARNING',
                  'No events carry a stable physical game ID; do not treat this as verified complete settlement linkage',
                  metrics={'with_physical_game_id':0,'events':len(events)})
    else:
        add_check(report,'PHYSICAL_GAME_ID_COVERAGE','PASS' if events else 'UNKNOWN',
                  'All captured events carry physical game IDs' if events else 'No events to assess')
    add_check(report,'BRAIN_PRICE_COVERAGE','PASS' if priced==len(events) else 'WARNING',
              'Captured signal prices are verified' if priced==len(events) else
              'Some captured observations are not priced; model-only evidence must not be treated as bets',
              metrics={'price_verified':priced,'observations':len(events)})
    source_fields_present={f:len(v) for f,v in source_fields.items() if isinstance(v,list)}
    for fam in ('MINER','PATHI','BIG_AL','PT','STAT'):
        n=counts.get(fam,0)
        fields=source_fields_present.get(fam,0)
        severity='PASS' if n else ('WARNING' if fields else 'UNKNOWN')
        why=('Source fields and observed signals present' if n else
             'Source signal fields present but no event captured; inspect gating and whether a valid signal fired' if fields else
             'No source fields or signals in capture; cannot infer whether source is disconnected or legitimately absent')
        add_check(report,f'BRAIN_{fam}_FLOW',severity,why,
                  metrics={'captured_events':n,'source_signal_fields_present':fields})
    report['capture_audit']={'artifact':latest['object_name'],'events':len(events),
                             'count_by_family':dict(counts),'issues':dict(issues)}


def audit_capture_to_ledger_links(report, latest, storage, bucket_name, now):
    """Verify every valid directional first-observation record has a durable ID.

    Checks identity-level existence, not a single quote per bookmaker. No writes.
    A later opposite selection must not replace the first immutable decision.
    """
    from ncaaf_v55_production_ledger import (_event_from_capture,
                                               select_first_decisions)
    from ncaaf_v55_settlement_bridge import parse_dt
    doc=latest['capture']; raw_events=doc.get('observations') or []
    created=parse_dt(latest['time_created_utc']); receipt=parse_dt(doc.get('captured_at_utc'))
    if not created or not receipt:
        add_check(report,'CAPTURE_TO_LEDGER_JOIN','FAIL','Capture creation/receipt timestamp missing')
        return
    normalized=[]
    for idx,item in enumerate(raw_events):
        if not isinstance(item,dict):continue
        e=_event_from_capture(item,doc,name=latest['object_name'],i=idx,
                              receipt=receipt,created=created,observed_now=now)
        if e:normalized.append(e)
    selected=select_first_decisions(normalized)
    eligible=[d for d in selected if d['price_verified'] and
              d['capture_clock_verified'] and d['direction_verified']]
    absent=[]
    b=storage.bucket(bucket_name)
    for d in eligible:
        path=f'research/ncaaf/v55/production_intelligence_v1/shadow_decisions_v1/{d["decision_id"]}.json'
        if not b.blob(path).exists():
            absent.append({'family':d['family'],'game_key':d['game_key'],
                           'signal_id':d['signal_id']})
    add_check(report,'CAPTURE_TO_LEDGER_JOIN',
              'FAIL' if absent else 'PASS' if eligible else 'UNKNOWN',
              'Valid priced brain signals must have corresponding immutable decision IDs; repeated quote rows are not extra decisions',
              metrics={'normalized_capture_events':len(normalized),
                       'distinct_mechanism_game_candidates':len(selected),
                       'eligible_price_direction_clock':len(eligible),
                       'missing_immutable_decisions':len(absent),'missing_examples':absent[:5]})


def audit_ledger(report, ledger):
    if not isinstance(ledger,dict):
        add_check(report,'LEDGER_AVAILABLE','UNKNOWN','No ledger report provided');return
    err=ledger.get('errors_count')
    add_check(report,'LEDGER_PROCESSING','PASS' if err==0 else 'FAIL' if isinstance(err,int) else 'UNKNOWN',
              'Existing performance ledger error accounting',
              metrics={'errors_count':err,'recent_errors':ledger.get('errors',[])[:5]})
    bets=ledger.get('authorized_bets'); authority=ledger.get('bet_authority')
    add_check(report,'BET_AUTHORITY_UNCHANGED','PASS' if bets==0 and authority==0 else 'FAIL',
              'Read-only health audit must not affect bet authorization',
              metrics={'authorized_bets':bets,'bet_authority':authority})
    if ledger.get('observer_events_in_recent_window') is not None:
        obs=ledger['observer_events_in_recent_window']; proposed=ledger.get('candidate_system_game_pairs_in_window')
        add_check(report,'SNAPSHOT_REPETITION_ACCOUNTED_FOR',
                  'PASS' if isinstance(obs,int) and isinstance(proposed,int) and 0<=proposed<=obs else 'FAIL',
                  'Repeated quote observations are distinct from proposed mechanism-game decisions',
                  metrics={'observed_events_recent':obs,'candidate_mechanism_game_pairs':proposed,
                           'durable_shadow_decisions':ledger.get('durable_shadow_decisions')})
    scored=ledger.get('score_review_link_diagnostics') or {}
    total=ledger.get('provisional_outcomes_count')
    add_check(report,'RESULT_LINKAGE','PASS' if isinstance(total,int) and total>=0 else 'UNKNOWN',
              'Score-review linkage runs, but two-source agreement is not official-final certification',
              metrics={'provisional_outcomes':total,
                       'score_review_game_keys':ledger.get('dual_source_review_game_keys'),
                       'score_link_diagnostics':scored})
    core=ledger.get('core_source_capture') or {}
    add_check(report,'CORE_MODEL_AND_CAPTURE','PASS' if core.get('status')=='PASS' and core.get('errors_count')==0
              else 'WARNING' if core else 'UNKNOWN',
              'CORE board capture and priced-or-model-only distinctions',
              metrics={'core_source_state':core.get('status'),
                       'model_games_seen':core.get('core_model_games_seen'),
                       'existing_first_priced':core.get('existing_core_first_priced'),
                       'new_first_priced':core.get('new_core_first_priced'),
                       'model_only_or_missing_price':core.get('missing_selection_and_price'),
                       'capture_errors':core.get('errors_count')})


def audit_verified_board(report, client, now):
    """Read the same complete, SHA-verifiable board as CORE; never infer from UI."""
    if client is None:
        add_check(report,'COMMITTED_BOARD','UNKNOWN','BigQuery client not available');return
    try:
        from ncaaf_v55_core_shadow import fetch_verified_board
        board=fetch_verified_board(client,now=now)
    except Exception as exc:
        add_check(report,'COMMITTED_BOARD','FAIL','Unable to verify last committed V5.5 board',
                  metrics={'error_type':type(exc).__name__,'error':str(exc)[:130]})
        return
    rows=board['rows']; missing=Counter(); ncore=0; fixture=set(); sides=Counter(); examples=[]
    for row in rows:
        if not isinstance(row,dict):missing['INVALID_ROW']+=1;continue
        if _str(row.get('CORE_Model'))=='NONLINEAR_HGB':ncore+=1
        if _num(row.get('CORE_Projected_Home_Margin')) is None:missing['CORE_PROJECTION_MISSING']+=1
        start=_str(row.get('Game_Start_UTC'))
        home=_str(row.get('Home'));away=_str(row.get('Away'))
        if (not home or not away) and ' @ ' in _str(row.get('Matchup')):
            away,home=_str(row.get('Matchup')).split(' @ ',1)
        if not home or not away or not _dt(start):missing['FIXTURE_IDENTITY_INCOMPLETE']+=1
        else:
            fixture.add((away.casefold(),home.casefold(),start))
            sides[(away.casefold(),home.casefold(),start,_str(row.get('Selected_Team')).casefold(),
                   _str(row.get('Current_Spread')),_str(row.get('Backend_Quote_Book')))] += 1
    critical=missing['INVALID_ROW']+missing['CORE_PROJECTION_MISSING']+missing['FIXTURE_IDENTITY_INCOMPLETE']
    duplicates=sum(n-1 for n in sides.values() if n>1)
    # Multiple lines/books/sides for the same game are intentional market quotes;
    # exact repeated row keys inside one committed publication deserve review.
    severity='FAIL' if critical else 'WARNING' if duplicates or ncore<len(rows) else 'PASS'
    add_check(report,'COMMITTED_BOARD',severity,
              'SHA-verified committed publication; checked projection and fixture completeness',
              metrics={'snapshot_id':board['snapshot_id'],'published_utc':board['published_at_utc'],
                       'rows':len(rows),'unique_fixtures':len(fixture),
                       'nonlinear_hgb_rows':ncore,'missing':dict(missing),
                       'same_side_same_line_book_repeats':duplicates})
    report['board_audit']={'snapshot_id':board['snapshot_id'],'rows':len(rows),
                           'unique_fixtures':len(fixture),'duplicate_same_side_price_rows':duplicates}


def audit_warehouse_metadata(report, bq_client):
    """Low-cost schema/availability inventory; NOT a historical row-level proof."""
    if bq_client is None:
        add_check(report,'WAREHOUSE_INVENTORY','UNKNOWN','BigQuery client not available');return
    info={};missing=[]
    for table in REQUIRED_WAREHOUSE:
        path='sharplogger.sharp_data.'+table
        try:
            t=bq_client.get_table(path)
            cols={f.name for f in t.schema}
            info[table]={'available':True,'row_count_metadata':getattr(t,'num_rows',None),
                         'key_fields_available':sorted(cols.intersection({
                             'Sport','Game_ID','Game_Start','Home_Team','Away_Team',
                             'game_id','home_team','away_team','season','Season',
                             'Snapshot_Timestamp','physical_game_id'}))}
            if getattr(t,'num_rows',None)==0:missing.append(table+':EMPTY')
        except Exception as exc:
            info[table]={'available':False,'error_type':type(exc).__name__}
            missing.append(table+':UNAVAILABLE')
    status='WARNING' if missing else 'PASS'
    add_check(report,'WAREHOUSE_INVENTORY',status,
              'Metadata-only availability check; does not prove historical coverage, deduplication, or feature use',
              metrics={'source_tables_checked':len(REQUIRED_WAREHOUSE),
                       'unavailable_or_empty':missing,'tables':info}, domain='research')
    # Never present metadata-only row counts as a full-model feature integrity PASS.
    add_check(report,'HISTORICAL_ROW_LEVEL_JOIN_AND_LEAKAGE','UNKNOWN',
              'Metadata cannot certify research input completeness; read research-input health separately',domain='research')



def audit_research_publications(report, storage_client, bucket_name, now):
    """Independent read of the heavy-research *actual frame* health publication.

    Research is run on a different cadence from market refresh. Stale and missing
    artifacts are NOT mistaken for a failed live production cycle.
    """
    bucket=storage_client.bucket(bucket_name)
    path=f'{OUT}/latest_research.json'
    obj=bucket.blob(path)
    try:
        if not obj.exists():
            add_check(report,'RESEARCH_INPUT_AUDIT','UNKNOWN',
                      'Heavy Research has not published its input-frame audit yet; execute Heavy Research',
                      metrics={'expected_artifact':f'gs://{bucket_name}/{path}'},domain='research')
            return
        size=getattr(obj,'size',None)
        if size and size>MAX_CAPTURE_BYTES:
            add_check(report,'RESEARCH_INPUT_AUDIT','FAIL','Research audit exceeds size cap',domain='research')
            return
        raw=obj.download_as_bytes()
        if len(raw)>MAX_CAPTURE_BYTES:raise ValueError('AUDIT_FILE_TOO_LARGE')
        doc=json.loads(raw)
        if doc.get('schema_version')!='V55_RESEARCH_INPUT_HEALTH_V1_20261101':
            raise ValueError('RESEARCH_INPUT_AUDIT_SCHEMA_MISMATCH')
        generated=_dt(doc.get('created_at_utc'))
        age=(now-generated).total_seconds()/3600 if generated else None
        if age is None or age < -0.0833:
            add_check(report,'RESEARCH_INPUT_AUDIT','FAIL','Invalid or future dated research-input audit',domain='research')
            return
        issues=doc.get('failure_codes') or []
        status=('FAIL' if issues or doc.get('status')=='FAIL' else
                'WARNING' if age>14*24 or doc.get('status')=='WARNING' else
                'UNKNOWN' if doc.get('status')=='UNKNOWN' else 'PASS')
        add_check(report,'RESEARCH_INPUT_AUDIT',status,
                  'Heavy Research actual-frame audit is available; observations are separate from live production',
                  metrics={'input_audit_status':doc.get('status'),
                           'audit_age_hours':round(age,2),
                           'row_summary':doc.get('row_summary',{}),
                           'failure_codes':issues,
                           'warning_codes':doc.get('warning_codes') or [],
                           'source_tag':doc.get('source_tag')},domain='research')
        report['research_input_audit']={'status':doc.get('status'),'age_hours':round(age,2),
                                        'artifact':f'gs://{bucket_name}/{path}',
                                        'row_summary':doc.get('row_summary',{})}
    except Exception as exc:
        add_check(report,'RESEARCH_INPUT_AUDIT','FAIL',
                  'Research audit publication unreadable; do not infer successful research validation',
                  metrics={'error_type':type(exc).__name__},domain='research')
        return

    # Published research report is evidence of pipeline completion; it is not
    # equivalent to verified per-row feature provenance.
    path='research/ncaaf/v2/current_report.json'
    try:
        obj=bucket.blob(path)
        if not obj.exists():
            add_check(report,'RESEARCH_REPORT_PUBLICATION','WARNING',
                      'No current Heavy Research report',domain='research')
            return
        sz=getattr(obj,'size',None)
        if sz and sz>MAX_CAPTURE_BYTES:
            add_check(report,'RESEARCH_REPORT_PUBLICATION','UNKNOWN',
                      'Research report exists but exceeds safe audit read cap; provenance not checked',
                      metrics={'size_bytes':sz},domain='research')
            return
        raw=obj.download_as_bytes()
        if len(raw)>MAX_CAPTURE_BYTES:raise ValueError('RESEARCH_REPORT_TOO_LARGE')
        r=json.loads(raw)
        seasons=r.get('seasons') or []
        prospective=[x for x in seasons if isinstance(x,(float,int)) and x>=2026]
        health=r.get('research_feature_lineage_audit') or {}
        fl=health.get('summary') or {}
        bad=bool(prospective or r.get('production_authority') not in (0,None) or
                 fl.get('status')=='AUDIT_ERROR')
        report_age=(now-_dt(r.get('created_utc'))).total_seconds()/3600 if _dt(r.get('created_utc')) else None
        prior=report.get('research_input_audit') or {}
        mismatch=bool(prior.get('age_hours') is not None and
                      prior.get('row_summary',{}).get('rows') is not None and
                      r.get('rows') is not None and
                      r.get('created_utc') and _dt(r.get('created_utc')) and
                      _dt(r.get('created_utc')) <= now and
                      abs((_dt(r.get('created_utc'))-
                           (now-timedelta(hours=prior['age_hours']))).total_seconds())<3*3600 and
                      r['rows']!=prior['row_summary']['rows'])
        add_check(report,'RESEARCH_REPORT_PUBLICATION',
                  'FAIL' if bad or mismatch else 'WARNING' if report_age is None or report_age>14*24 or r.get('status') not in ('NCAAF_RESEARCH_V2_COMPLETE','PASS') else 'PASS',
                  'Heavy Research report was read without changing systems, model or qualification',
                  metrics={'status':r.get('status'),'source_tag':r.get('source_tag'),
                           'rows':r.get('rows'),'seasons':seasons,
                           'report_age_hours':round(report_age,2) if report_age is not None else None,
                           'feature_lineage_status':fl.get('status'),
                           'research_input_row_count_mismatch':mismatch},domain='research')
        add_check(report,'RESEARCH_FEATURE_LINEAGE_PUBLICATION',
                  'UNKNOWN' if not fl.get('status') else
                  'FAIL' if fl.get('status')=='AUDIT_ERROR' else
                  'PASS' if fl.get('status')=='PASS' else 'WARNING',
                  'Feature-lineage artifact status as reported by Heavy Research; independently checking per-feature vintages requires source timestamps',
                  metrics={'feature_lineage_status':fl.get('status')}, domain='research')
    except Exception as exc:
        add_check(report,'RESEARCH_REPORT_PUBLICATION','FAIL',
                  'Current research report could not be verified',
                  metrics={'exception':type(exc).__name__},domain='research')

def run_cycle(*, storage_client, bucket_name='sharp-models', ledger_report,
              bq_client=None, now=None, persist=True):
    """Call once from existing background ledger after its counters are built.

    Only reads existing source publications and writes its own health namespace.
    Does not mutate the input ledger_report and does not affect the betting gate.
    """
    now=_dt(now or datetime.now(timezone.utc))
    if now is None:raise ValueError('AUDIT_INVALID_CLOCK')
    r={'schema_version':'V55_PIPELINE_HEALTH_V1_20261101',
       'status':'UNKNOWN','created_at_utc':now.isoformat(),
       'source':'V55_COMMITTED_BOARD_AND_PROSPECTIVE_CAPTURE_AND_LEDGER',
       'scope':'NONDESTRUCTIVE_PIPELINE_HEALTH_NOT_MODEL_PERFORMANCE',
       'historical_feature_completeness_certified':False,
       'pre_kickoff_feature_vintage_certified':False,
       'official_final_status_certified':False,
       'model_logic_modified':False,'bet_authority':0,'checks':[]}
    audit_ledger(r,ledger_report)
    try:
        latest,err=_latest_capture(storage_client,bucket_name,now)
        if err:
            add_check(r,'LATEST_IMMUTABLE_CAPTURE','FAIL' if 'HASH_MISMATCH' in err else 'WARNING',err)
        else:
            made=_dt(latest['time_created_utc'])
            add_check(r,'LATEST_IMMUTABLE_CAPTURE','PASS' if made and now-made<=timedelta(hours=2) else 'WARNING',
                      'Content hash and GCS creation time verified',
                      metrics={'artifact':latest['object_name'],'age_minutes':round((now-made).total_seconds()/60,2)})
            audit_capture(r,latest,now)
            try:
                audit_capture_to_ledger_links(r,latest,storage_client,bucket_name,now)
            except Exception as exc:
                add_check(r,'CAPTURE_TO_LEDGER_JOIN','UNKNOWN',
                          'Could not verify capture-to-ledger existence',
                          metrics={'exception':type(exc).__name__})
    except Exception as exc:
        add_check(r,'LATEST_IMMUTABLE_CAPTURE','FAIL','Capture read or verification failed',
                  metrics={'exception':type(exc).__name__})
    audit_verified_board(r,bq_client,now)
    audit_warehouse_metadata(r,bq_client)
    audit_research_publications(r,storage_client,bucket_name,now)
    summarize(r)
    if persist:
        bucket=storage_client.bucket(bucket_name)
        path=f'{OUT}/reports/{now.strftime("%Y%m%dT%H%M%S%fZ")}.json'
        blob=bucket.blob(path)
        try:
            blob.upload_from_string(_json(r),content_type='application/json',if_generation_match=0)
            bucket.blob(f'{OUT}/latest.json').upload_from_string(_json(r),content_type='application/json')
            r['report_uri']=f'gs://{bucket_name}/{path}'
        except Exception as exc:
            add_check(r,'AUDIT_REPORT_PERSISTENCE','FAIL','Health report could not be saved',
                      metrics={'exception':type(exc).__name__})
            summarize(r)
    LOG.info('[NCAAF-V55-PIPELINE-HEALTH] %s',_json({
        'status':r['status'],'check_counts':r['check_counts'],
        'production_status':r['production_status'],
        'research_status':r['research_status'],
        'failure_codes':r['failure_codes'],'warning_codes':r['warning_codes'],
        'unknown_codes':r['unknown_codes'],'report_uri':r.get('report_uri'),
        'bet_authority':0}))
    return r
