"""V5.5 NCAAF immutable *review-only* settlement bridge.

Finds score evidence from both CFBD and game_scores_final, using oriented
America/New_York calendar date plus explicitly allowed school aliases. Does not
claim official finality when neither source supplies an explicit finality field.
It never writes authority-bearing outcomes or authorizes a wager.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PREFIX = 'research/ncaaf/v55/prospective_brain_evidence_v1'
SCORE_POLICY = 'V55-DUAL-SCORE-REVIEW-NO-FINALITY-CERTIFICATE-20261101'
REVIEW_VERSION = 2
ALIASES = {
    'james madison dukes':'james madison', 'troy trojans':'troy',
    'middle tennessee blue raiders':'middle tennessee',
    'kennesaw state owls':'kennesaw state',
    'new mexico state aggies':'new mexico state',
    'jacksonville state gamecocks':'jacksonville state',
    'north texas mean green':'north texas',
    'florida atlantic owls':'florida atlantic',
    'western kentucky hilltoppers':'western kentucky',
    'delaware blue hens':'delaware',
}


def norm(value):
    import re
    v=' '.join(re.sub(r'[^a-z0-9]+', ' ', str(value or '').lower()).split())
    return ALIASES.get(v, v)


def parse_dt(value):
    if isinstance(value, datetime): v=value
    else:
        try:v=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        except (ValueError,TypeError):return None
    if v.tzinfo is None or v.utcoffset() is None:return None
    return v.astimezone(timezone.utc)


def key_from_row(row, *, home='home', away='away', stamp='kickoff_utc'):
    t=parse_dt(row.get(stamp))
    h=norm(row.get(home)); a=norm(row.get(away))
    if not t or not h or not a or h==a:return None
    return ('NCAAF', t.astimezone(ZoneInfo('America/New_York')).date().isoformat(), h, a)


def score_pair(row):
    try:
        h=float(row['home_score']); a=float(row['away_score'])
        if h<0 or a<0 or not h.is_integer() or not a.is_integer():return None
        return int(h),int(a)
    except (TypeError,KeyError,ValueError,OverflowError):return None


# Score-only identity. Never substitute this for the immutable capture/game_key
# identity: doing so would change previously issued decision IDs and hashes.
def score_canonical_key(row):
    t=parse_dt(row.get('kickoff_utc'))
    if t is None:return None
    # Lazy import because the CORE module itself imports the original
    # settlement bridge. This is the SAME explicit alias policy already
    # used for the deployed CORE–Miner linkage; no fuzzy matching.
    from ncaaf_v55_core_shadow import canonical_team_strict
    h=canonical_team_strict(row.get('home'))
    a=canonical_team_strict(row.get('away'))
    if not h or not a or h==a:return None
    return ('NCAAF',t.astimezone(ZoneInfo('America/New_York')).date().isoformat(),h,a)


def source_coverage(rows):
    from collections import Counter
    counts=Counter()
    for row in rows:
        k=score_canonical_key(row)
        if k:counts[k[1]]+=1
    return dict(sorted(counts.items()))


def missing_score_diagnosis(game, cfbd_rows, final_rows):
    """Discriminate source lag from alias/date failures without guessing."""
    key=score_canonical_key(game)
    if not key:return 'CAPTURED_GAME_IDENTITY_INVALID'
    day=key[1]
    cfkeys=[score_canonical_key(r) for r in cfbd_rows]
    fikeys=[score_canonical_key(r) for r in final_rows]
    if key not in cfkeys:
        same_teams_elsewhere=any(x and x[2:]==key[2:] and x[1]!=day for x in cfkeys)
        if same_teams_elsewhere:return 'CFBD_FIXTURE_DIFFERENT_DATE_HOLD'
        reverse=('NCAAF',day,key[3],key[2])
        if reverse in cfkeys:return 'CFBD_REVERSED_ORIENTATION_HOLD'
        if not any(x and x[1]==day for x in cfkeys):return 'CFBD_DATE_NOT_PRESENT_IN_EXTRACT'
        return 'CFBD_FIXTURE_MISSING_ON_PRESENT_DATE'
    if key not in fikeys:
        if not any(x and x[1]==day for x in fikeys):return 'SECOND_SOURCE_DATE_NOT_PRESENT_IN_EXTRACT'
        return 'SECOND_SOURCE_FIXTURE_MISSING_ON_PRESENT_DATE'
    return 'SCORES_PRESENT_BUT_UNVERIFIED_OR_CONFLICTED'


def score_review(game, cfbd_rows, final_rows, now, *, min_maturity_hours=12):
    """Strict dual-table score *review*, NEVER official-final certification."""
    original=key_from_row(game)
    target=score_canonical_key(game)
    if target is None or original is None:return {'state':'INVALID_CAPTURED_IDENTITY'}
    t=parse_dt(game.get('kickoff_utc'))
    if now-t<timedelta(hours=min_maturity_hours):return {'state':'WAIT_FOR_SCORE_MATURITY'}
    cf=[r for r in cfbd_rows if score_canonical_key(r)==target]
    fi=[r for r in final_rows if score_canonical_key(r)==target]
    if not cf:return {'state':'CFBD_SCORE_NOT_FOUND', 'diagnostic':missing_score_diagnosis(game,cfbd_rows,final_rows)}
    if not fi:return {'state':'SECOND_SCORE_SOURCE_NOT_FOUND', 'diagnostic':missing_score_diagnosis(game,cfbd_rows,final_rows)}
    ids={str(r.get('cfbd_game_id') or '').strip() for r in cf}
    if '' in ids or len(ids)!=1:return {'state':'AMBIGUOUS_CFBD_ID'}
    scores={score_pair(r) for r in cf+fi}
    if None in scores or len(scores)!=1:return {'state':'MISSING_OR_CONFLICTING_SCORES'}
    return {'state':'DUAL_SOURCE_SCORE_AGREEMENT_REVIEW_ONLY',
            'cfbd_game_id':next(iter(ids)), 'home_score':next(iter(scores))[0],
            'away_score':next(iter(scores))[1], 'stable_game_key':'|'.join(original),
            'canonical_score_fixture_identity':'|'.join(target),
            'cfbd_rows':len(cf), 'final_rows':len(fi),
            'score_source':'ncaaf_cfbd_postgame_finals_v1+game_scores_final',
            'sources_are_independent_certified':False,
            'finality_certified':False,'settlement_certified':False,
            'bet_authority':0,'source_proof_policy':SCORE_POLICY}


def query_sources(client, start, end):
    """Bounded queries; only known warehouse schemas. Fail closed on BQ errors."""
    from google.cloud import bigquery
    config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter('start','DATE',start),
        bigquery.ScalarQueryParameter('end','DATE',end),
    ],maximum_bytes_billed=2_000_000_000)
    csql='''SELECT CAST(game_id AS STRING) cfbd_game_id,
      CAST(home_team AS STRING) home, CAST(away_team AS STRING) away,
      SAFE_CAST(home_points AS FLOAT64) home_score,
      SAFE_CAST(away_points AS FLOAT64) away_score,
      CAST(game_start_utc AS STRING) kickoff_utc
      FROM `sharplogger.sharp_data.ncaaf_cfbd_postgame_finals_v1`
      WHERE season BETWEEN EXTRACT(YEAR FROM @start)-1 AND EXTRACT(YEAR FROM @end)+1
        AND DATE(SAFE_CAST(game_start_utc AS TIMESTAMP),'America/New_York')
            BETWEEN @start AND @end'''
    fsql='''SELECT CAST(Home_Team AS STRING) home,
      CAST(Away_Team AS STRING) away,
      SAFE_CAST(Score_Home_Score AS FLOAT64) home_score,
      SAFE_CAST(Score_Away_Score AS FLOAT64) away_score,
      CAST(Game_Start AS STRING) kickoff_utc
      FROM `sharplogger.sharp_data.game_scores_final`
      WHERE UPPER(CAST(Sport AS STRING))='NCAAF'
        AND DATE(SAFE_CAST(Game_Start AS TIMESTAMP),'America/New_York')
            BETWEEN @start AND @end'''
    cf=[dict(r.items()) for r in client.query(csql,job_config=config).result()]
    fi=[dict(r.items()) for r in client.query(fsql,job_config=config).result()]
    return cf,fi


def reconcile_prospective_score_reviews(*, dashboard_module=None, storage_client=None,
                                         bucket_name='sharp-models',now=None,
                                         days_back=15, in_streamlit=False):
    """Background-only: immutable review artifacts, NOT certification.

    No writes to existing `outcomes/` or `observations/`; this cannot cause
    Stage1 to start grading bets or change owner authority.
    """
    if in_streamlit:
        return {'status':'SKIPPED_INTERACTIVE_UI_READ_ONLY','bet_authority':0}
    t=parse_dt(now or datetime.now(timezone.utc))
    if t is None:return {'status':'INVALID_CLOCK_HOLD','bet_authority':0}
    if storage_client is None:
        from google.cloud import storage
        storage_client=storage.Client()
    bucket=storage_client.bucket(bucket_name)
    found={}; conflicts=set(); days=min(30,max(1,int(days_back)))
    for delta in range(days+1):
        day=(t-timedelta(days=delta)).strftime('%Y-%m-%d')
        prefix=f'{PREFIX}/games/{day}/'
        for blob in storage_client.list_blobs(bucket_name,prefix=prefix,max_results=500):
            try:
                g=json.loads(blob.download_as_text())
                gid=str(g.get('physical_game_id') or '').strip()
                sha=str(g.get('game_key_sha256') or '')
                if not gid or len(sha)!=32:continue
                if hashlib.sha256(gid.encode('utf-8')).hexdigest()[:32]!=sha:continue
                # Same original ID must not appear with conflicting identities.
                if sha in found and found[sha]!=g:conflicts.add(sha)
                elif sha not in found:found[sha]=g
            except (ValueError,TypeError,KeyError):continue
    targets={sha:g for sha,g in found.items() if sha not in conflicts}
    if not targets:return {'status':'NO_VALID_GAME_INDEXES','review_created':0,'bet_authority':0}
    mature={sha:g for sha,g in targets.items()
            if (parse_dt(g.get('kickoff_utc')) is not None and
                t-parse_dt(g['kickoff_utc'])>=timedelta(hours=12))}
    if not mature:return {'status':'NO_MATURE_GAMES','pending':len(targets),
                          'review_created':0,'bet_authority':0}
    if dashboard_module is not None and getattr(dashboard_module,'bq_client',None) is not None:
        client=dashboard_module.bq_client
    else:
        from google.cloud import bigquery
        client=bigquery.Client(project='sharplogger')
    # Query only the mature game dates; include a one-day pad at each end.
    dates=sorted({key_from_row(g)[1] for g in mature.values() if key_from_row(g)})
    if not dates:return {'status':'INVALID_MATURE_IDENTITIES','review_created':0,'bet_authority':0}
    lo=datetime.fromisoformat(dates[0]).date()-timedelta(days=1)
    hi=datetime.fromisoformat(dates[-1]).date()+timedelta(days=1)
    try: cf,fi=query_sources(client,lo,hi)
    except Exception as exc:
        return {'status':'SCORE_QUERY_FAILED_FAIL_CLOSED',
                'error':f'{type(exc).__name__}:{exc}'[:250],
                'review_created':0,'bet_authority':0}
    counts=Counter();diagnostics=Counter();examples=[];created=0;write_errors=0;score_drift=0;disputes_written=0
    for sha,g in sorted(mature.items()):
        outpath=f'{PREFIX}/score_reviews_v2/{sha}.json'
        blob=bucket.blob(outpath)
        if blob.exists():
            counts['ALREADY_PRESENT']+=1
            # Existing immutable reviews must NOT silently override a later
            # contradictory pair of source scores. An append-only dispute
            # quarantines this fixture in the consuming ledger.
            try:
                old=json.loads(blob.download_as_text())
                current=score_review(g,cf,fi,t)
                if current.get('state')=='DUAL_SOURCE_SCORE_AGREEMENT_REVIEW_ONLY' and (
                    (current.get('home_score'),current.get('away_score'),str(current.get('cfbd_game_id')))
                    != (old.get('home_score'),old.get('away_score'),str(old.get('cfbd_game_id')))):
                    score_drift+=1
                    dispute={'original_game_index_sha256':sha,'original_physical_game_id':g['physical_game_id'],
                             'original_review_scores':[old.get('home_score'),old.get('away_score')],
                             'new_source_scores':[current['home_score'],current['away_score']],
                             'original_cfbd_id':old.get('cfbd_game_id'),
                             'new_cfbd_id':current.get('cfbd_game_id'),
                             'observed_at_utc':t.isoformat(), 'state':'DISPUTED_REVIEW_SCORE_HOLD',
                             'bet_authority':0,'settlement_certified':False}
                    db=bucket.blob(f'{PREFIX}/score_review_disputes_v1/{sha}.json')
                    if not db.exists():
                        db.upload_from_string(json.dumps(dispute,sort_keys=True),
                            content_type='application/json',if_generation_match=0)
                        disputes_written+=1
                elif current.get('state')!='DUAL_SOURCE_SCORE_AGREEMENT_REVIEW_ONLY':
                    diagnostics['EXISTING_REVIEW_RECHECK_SOURCE_NOT_CURRENTLY_AGREEING']+=1
            except Exception:
                # A read error is not evidence that an old review is correct.
                diagnostics['EXISTING_REVIEW_RECHECK_ERROR_HOLD']+=1
                write_errors+=1
            continue
        review=score_review(g,cf,fi,t)
        counts[review['state']]+=1
        if review.get('diagnostic'):
            diagnostics[review['diagnostic']]+=1
            if len(examples)<12:examples.append({'fixture':review.get('diagnostic'), 'home':g.get('home'), 'away':g.get('away'),'kickoff_utc':g.get('kickoff_utc')})
        if review['state']!='DUAL_SOURCE_SCORE_AGREEMENT_REVIEW_ONLY':continue
        artifact={'original_physical_game_id':g['physical_game_id'],
                  'original_game_index_sha256':sha,
                  'captured_home_team':g['home'],
                  'captured_away_team':g['away'],
                  'captured_kickoff_utc':g['kickoff_utc'],
                  'review_created_at_utc':t.isoformat(),
                  'review_version':2,
                  'immutable_source_capture_preserved':True,
                  **review}
        try:
            blob.upload_from_string(json.dumps(artifact,sort_keys=True),
                                    content_type='application/json',if_generation_match=0)
            created+=1
        except Exception:
            if blob.exists():counts['ALREADY_PRESENT_CONCURRENTLY']+=1
            else:write_errors+=1
    return {'status':'PASS_REVIEW_ONLY' if not write_errors else 'PARTIAL_REVIEW_WRITE_FAILURE',
            'mature_game_indexes':len(mature),
            'score_sources':{'cfbd':len(cf),'game_scores_final':len(fi)},
            'review_created':created,'review_states':dict(counts),
            'source_score_drift_detected':score_drift,'review_disputes_written':disputes_written,
            'missing_score_diagnostics':dict(diagnostics),
            'missing_examples':examples,
            'source_coverage_by_et_day':{'cfbd':source_coverage(cf), 'game_scores_final':source_coverage(fi)},
            'write_errors':write_errors,'outcome_objects_modified':0,
            'bet_authority':0,'automatic_wager_execution':False}
