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
SCORE_POLICY = 'V55-DUAL-SCORE-REVIEW-NO-FINALITY-CERTIFICATE-20261030'
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


def score_review(game, cfbd_rows, final_rows, now, *, min_maturity_hours=12):
    """Return a review object, never a finality/authority certificate."""
    target=key_from_row(game)
    if target is None:return {'state':'INVALID_CAPTURED_IDENTITY'}
    t=parse_dt(game.get('kickoff_utc'))
    if now-t<timedelta(hours=min_maturity_hours):return {'state':'WAIT_FOR_SCORE_MATURITY'}
    cf=[r for r in cfbd_rows if key_from_row(r)==target]
    fi=[r for r in final_rows if key_from_row(r)==target]
    if not cf:return {'state':'CFBD_SCORE_NOT_FOUND'}
    if not fi:return {'state':'SECOND_SCORE_SOURCE_NOT_FOUND'}
    ids={str(r.get('cfbd_game_id') or '').strip() for r in cf}
    if '' in ids or len(ids)!=1:return {'state':'AMBIGUOUS_CFBD_ID'}
    scores={score_pair(r) for r in cf+fi}
    if None in scores or len(scores)!=1:return {'state':'MISSING_OR_CONFLICTING_SCORES'}
    return {'state':'DUAL_SOURCE_SCORE_AGREEMENT_REVIEW_ONLY',
            'cfbd_game_id':next(iter(ids)), 'home_score':next(iter(scores))[0],
            'away_score':next(iter(scores))[1], 'stable_game_key':'|'.join(target),
            'cfbd_rows':len(cf), 'final_rows':len(fi),
            'score_source':'ncaaf_cfbd_postgame_finals_v1+game_scores_final',
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
    counts=Counter();created=0;write_errors=0
    for sha,g in sorted(mature.items()):
        outpath=f'{PREFIX}/score_reviews_v1/{sha}.json'
        blob=bucket.blob(outpath)
        if blob.exists():
            counts['ALREADY_PRESENT']+=1
            continue
        review=score_review(g,cf,fi,t)
        counts[review['state']]+=1
        if review['state']!='DUAL_SOURCE_SCORE_AGREEMENT_REVIEW_ONLY':continue
        artifact={'original_physical_game_id':g['physical_game_id'],
                  'original_game_index_sha256':sha,
                  'captured_home_team':g['home'],
                  'captured_away_team':g['away'],
                  'captured_kickoff_utc':g['kickoff_utc'],
                  'review_created_at_utc':t.isoformat(),
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
            'write_errors':write_errors,'outcome_objects_modified':0,
            'bet_authority':0,'automatic_wager_execution':False}
