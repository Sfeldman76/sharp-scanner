#!/usr/bin/env python3
"""NCAAF V5.5 read-only current model-to-live-market scoring integrity audit.

No writes to GCP, no retraining, no source discovery, no authority changes.
Run alongside CURRENT ncaaf_v55_backend.py and ncaaf_v55_board.py.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import traceback

MODEL_KEYS = ('FULL_CORE_margin','FULL_STAT_margin','FULL_COMBINED_margin',
              'FULL_CORE_total','FULL_STAT_total','FULL_COMBINED_total',
              'FULL_STACKED_margin')
FAMILIES = ('CORE','STAT','SYSTEM_MINER','PATHI','BIG_AL','PREDICTION_TRACKER','MARKET_RICH')


def read_blob_json(client, uri):
    if not uri.startswith('gs://'):
        raise ValueError('EXPECTED_GS_URI')
    bucket, key = uri[5:].split('/',1)
    return json.loads(client.bucket(bucket).blob(key).download_as_bytes())


def safe_error(exc):
    return type(exc).__name__ + ': ' + str(exc)[:400]


def run(*, out, manifest_uri, bucket, horizon_days=14, max_games=0,
        live_manifest_uri=None, extra_quotes_uri=None):
    import joblib
    import pandas as pd
    from google.cloud import bigquery, storage
    import ncaaf_v55_backend as backend
    import ncaaf_v55_board as board_mod

    now=datetime.now(timezone.utc)
    bq=bigquery.Client(project='sharplogger')
    gcs=storage.Client(project='sharplogger')
    result={'schema':'NCAAF_V55_MODEL_TO_LIVE_MARKET_INTEGRITY_AUDIT_V1',
            'created_at_utc':now.isoformat(), 'scope':'READ_ONLY_CURRENT_FORWARD_SHADOW',
            'authority_changed':False, 'bet_authority':0,
            'models':{},'stages':{},'holds':{},'by_market':{},'by_brain':{},
            'scoring':{},'status':'INCOMPLETE'}
    errors=[]
    def stage(name, func):
        try:
            x=func();result['stages'][name]={'status':'PASS'};return x
        except Exception as e:
            errors.append(name+':'+safe_error(e));result['stages'][name]={'status':'HOLD','error':safe_error(e)}
            return None

    manifest=stage('manifest_fetch',lambda:read_blob_json(gcs,manifest_uri))
    specs={}
    if manifest is not None:
        result['manifest_schema']=manifest.get('schema')
        all_artifacts=manifest.get('artifacts') or {}
        for name in MODEL_KEYS:
            info=all_artifacts.get(name)
            entry={'present_in_manifest':isinstance(info,dict)}
            if not isinstance(info,dict):
                entry['status']='MISSING_ARTIFACT_CONTRACT';result['models'][name]=entry;continue
            uri=info.get('uri','');sha=info.get('sha256','');features=info.get('features') or []
            entry['feature_count']=len(features);entry['model_uri']=uri
            try:
                if not uri.startswith('gs://') or len(sha)!=64 or not isinstance(features,list):
                    raise ValueError('INVALID_MANIFEST_ARTIFACT_FIELDS')
                b,k=uri[5:].split('/',1)
                payload=gcs.bucket(b).blob(k).download_as_bytes()
                digest=hashlib.sha256(payload).hexdigest()
                if digest!=sha:raise ValueError('SHA256_MISMATCH')
                spec=joblib.load(io.BytesIO(payload))
                if list(spec.get('features',[]))!=features:
                    raise ValueError('FEATURE_SCHEMA_ORDER_OR_CONTENT_MISMATCH')
                if spec.get('bet_authority')!=0 or spec.get('forward_eligible') is not False:
                    raise ValueError('MODEL_NOT_MARKED_RESEARCH_SHADOW')
                if 'model' not in spec or not hasattr(spec['model'],'predict'):
                    raise ValueError('MISSING_PREDICT_METHOD')
                specs[name]=spec;entry['status']='PASS_HASH_SCHEMA_SHADOW'
            except Exception as e:entry['status']='HOLD';entry['error']=safe_error(e)
            result['models'][name]=entry
    result['stages']['model_artifacts']={'status':'PASS' if len(specs)==len(MODEL_KEYS) else 'HOLD',
        'loaded':len(specs),'expected':len(MODEL_KEYS)}

    # Fresh canonical fixtures. Only current 2026 CFBD game-side rows.
    fixtures=stage('cfbd_forward_fixture_join', lambda: backend.read_current_cfbd_fixture_slate(
        bq,now=now,horizon_days=horizon_days))
    fixture_frame=None;fixture_audit={}
    if fixtures is not None:
        fixture_frame,fixture_audit=fixtures
        if max_games>0:fixture_frame=fixture_frame.head(max_games).copy()
        result['stages']['cfbd_forward_fixture_join'].update(fixture_audit)
        result['stages']['cfbd_forward_fixture_join']['selected_games']=len(fixture_frame)

    quote_frame=None;board=None
    if fixture_frame is not None:
        quote_frame=stage('live_scanner_quote_read',lambda:board_mod.read_scanner_quotes(
            bq,minutes=180,horizon_days=horizon_days))
        if quote_frame is not None:
            joined=stage('paired_quote_to_fixture_join', lambda:board_mod.attach_verified_backend_quotes(
                fixture_frame,quote_frame,now=now,max_age_minutes=180))
            if joined is not None:
                board,quote_audit=joined
                result['stages']['paired_quote_to_fixture_join'].update(quote_audit)
                result['stages']['paired_quote_to_fixture_join']['quote_state_counts']=dict(Counter(
                    board.get('Quote_State',pd.Series(dtype='object')).astype(str)))

    inputs=[];feature_audit={}
    if board is not None and manifest is not None:
        needed={f for k,v in (manifest.get('artifacts') or {}).items()
                if k in MODEL_KEYS and isinstance(v,dict) for f in v.get('features',[])}
        result['feature_schema']={'unique_feature_fields':len(needed),
            'cfbd_prior_difference_fields':sum(x.startswith('CFBD_DIFF_') and x.endswith('_Prior') for x in needed),
            'non_cfbd_fields':sorted(x for x in needed if not x.startswith('CFBD_DIFF_'))[:75]}
        source=stage('forward_cfbd_source_columns',lambda:backend.read_live_cfbd_rows_from_bq(
            bq,'sharplogger.sharp_data.ncaaf_cfbd_miner_features_v1',needed,now=now))
        if source is not None:
            feat=stage('feature_parity_and_game_join',lambda:backend.build_live_cfbd_feature_snapshots(
                board.to_dict('records'),source,needed,now=now))
            if feat is not None:
                inputs,feature_audit=feat
                result['stages']['feature_parity_and_game_join'].update(feature_audit)
                result['stages']['feature_parity_and_game_join']['not_scored_games']=len(board)-len(inputs)
                data_by_id={str(r['game_id']):r for r in inputs}
                source_ids={str(r.get('Game_ID')) for r in board.to_dict('records')}
                result['stages']['feature_parity_and_game_join']['missing_game_ids']=sorted(source_ids-set(data_by_id))[:100]
                coverage={}
                for name,spec in specs.items():
                    fields=spec['features'];missing=Counter();finites=0
                    for row in inputs:
                        present=row.get('core_features') or {}
                        # Count only source-field parity, not derived market/evidence features.
                        missing.update(f for f in fields if f.startswith('CFBD_DIFF_') and f not in present)
                        finites+=sum(backend.num(present.get(f)) is not None for f in fields if f.startswith('CFBD_DIFF_'))
                    coverage[name]={'absent_prior_fields':dict(missing.most_common(12)),
                                    'finite_prior_cell_count':finites}
                result['feature_model_coverage']=coverage

    signals=[];signal_audit={'status':'NO_CONFIGURED_LIVE_EVENT_SOURCE'}
    manifest_signals=live_manifest_uri or os.getenv('NCAAF_V55_SEVEN_BRAIN_MANIFEST_URI','')
    if manifest_signals:
        found=stage('original_live_seven_brain_events',lambda:backend.read_seven_brain_signal_sources(
            manifest_signals,bq,now=now))
        if found is not None:signals,signal_audit=found
    result['stages']['original_live_seven_brain_events']={**signal_audit,
        'live_manifest_configured':bool(manifest_signals), 'read_only_no_producer_execution':True}
    if board is not None and signals:
        canonical=stage('live_signal_canonical_game_join',lambda:backend.canonicalize_live_signal_game_ids(signals,board))
        if canonical is not None:
            signals,canonical_audit=canonical
            result['stages']['live_signal_canonical_game_join'].update(canonical_audit)
    event_count=Counter(backend.family(x.get('source_family') or x.get('Source_Family')) for x in signals)
    for fam in FAMILIES:result['by_brain'][fam]={'accepted_current_original_directional_events':event_count[fam],
        'event_source_state':'PRESENT' if event_count[fam] else 'NO_ACCEPTED_ORIGINAL_LIVE_EVENTS',
        'does_not_require_core_origination':True}

    qrows=[]
    if board is not None:
        current=backend.row_quotes(board.to_dict('records'))
        if extra_quotes_uri:
            extra=stage('h2h_totals_quote_ingest',lambda:backend.read_rows(extra_quotes_uri))
            if extra is not None:current.extend(extra)
        raw_normalized=[q for r in current if (q:=backend.quote_normal(r,now,age_limit=180))]
        qrows=[{**q,'kickoff_utc':q['kickoff'].isoformat(),
                'quote_observed_at_utc':q['quote'].isoformat()} for q in raw_normalized]
        pair_ok,pair_holds=backend.v55_validate_shadow_quote_pairs(qrows)
        result['stages']['market_quote_normalization']={'status':'PASS' if pair_ok else 'HOLD_NO_VERIFIED_SAME_BOOK_PAIR',
            'input_quote_rows':len(current),'normalized_rows':len(qrows),'paired_rows':len(pair_ok),
            'pair_rejection_counts':dict(pair_holds),'market_counts':dict(Counter(q['market'] for q in pair_ok))}
    if inputs and qrows and specs and manifest is not None:
        shadow=stage('existing_backend_score_snapshot',lambda:backend.score_snapshot(
            specs,manifest,inputs,qrows,signals,now=now))
        if shadow is not None:
            result['scoring']={k:v for k,v in shadow.items() if k!='records'}
            rows=shadow.get('records',[])
            result['by_market']={market:{'scored_sides':len(rr),
                'scored_games':len({str(x.get('game_id')) for x in rr}),
                'core_model_scored':sum('CORE' in x.get('models',{}) for x in rr),
                'stat_model_scored':sum('STAT' in x.get('models',{}) for x in rr),
                'combined_model_scored':sum('COMBINED' in x.get('models',{}) for x in rr),
                'stack_model_scored':sum('STACKED' in x.get('models',{}) for x in rr),
                'expert_first_without_core_value':sum(x.get('expert_first_without_core_value') is True for x in rr)}
                for market in ('spreads','h2h','totals') for rr in [[r for r in rows if r.get('market')==market]]}
            by_fam=Counter(fam for row in rows for fam in row.get('brain_signal_families',[]))
            for fam in FAMILIES:result['by_brain'][fam]['scored_market_sides_with_signal']=by_fam[fam]
            # Keep compact operational rows; do not leak entire model feature vectors.
            result['scoring']['sample_scored_rows']=[{
                'game_id':r.get('game_id'),'market':r.get('market'),'side':r.get('side'),
                'book':r.get('book'),'model_names':sorted((r.get('models') or {}).keys()),
                'origin_families':r.get('expert_origin_families',[]),
                'expert_first_without_core_value':r.get('expert_first_without_core_value'),
                'status':r.get('status'),'authority':r.get('bet_authority')}
                for r in rows[:35]]
    elif board is not None:
        result['stages']['existing_backend_score_snapshot']={'status':'NOT_RUN_DEPENDENCY_MISSING',
            'feature_rows':len(inputs),'paired_quote_rows':len(qrows),'verified_models':len(specs)}

    # Historical evidence is contextual only; never feed the 2013-25 replay as live observations.
    result['acceptance_checks']={
        'historical_events_not_reused_as_live':True,
        'all_required_model_artifacts_verified':len(specs)==len(MODEL_KEYS),
        'canonical_forward_fixture_available':fixture_frame is not None and not fixture_frame.empty,
        'forward_feature_rows_available':bool(inputs),
        'paired_current_market_quotes_available':bool(qrows) and result['stages'].get('market_quote_normalization',{}).get('paired_rows',0)>0,
        'seven_brain_original_event_manifest_configured':bool(manifest_signals),
        'original_event_handoff_present':bool(signals),
        'backend_scoring_produced_rows':result['scoring'].get('scored_market_sides',0)>0,
        'bet_authority_remained_zero':True}
    result['errors']=errors
    if errors:result['status']='HOLD_AUDIT_STAGE_FAILURE'
    elif result['acceptance_checks']['backend_scoring_produced_rows']:
        result['status']='PASS_SHADOW_SCORING__CHECK_REPORTED_INPUT_HOLDS'
    else:result['status']='HOLD_NO_COMPLETE_LIVE_SCORING_CHAIN'
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(result,indent=2,sort_keys=True,default=str,allow_nan=False)+'\n')
    return result


def self_test():
    import ncaaf_v55_backend as b
    from datetime import timedelta
    now=datetime.now(timezone.utc)
    iso=lambda x:x.isoformat()
    base={'game_id':'G1','market':'spreads','book':'testbook',
          'kickoff_utc':iso(now+timedelta(hours=4)),
          'quote_observed_at_utc':iso(now-timedelta(minutes=3)),
          'odds':-110}
    good=[dict(base,side='HOME',line=-3.5),dict(base,side='AWAY',line=3.5)]
    pair,holds=b.v55_validate_shadow_quote_pairs(good)
    assert len(pair)==2 and not holds,(pair,holds)
    bad=[good[0]]
    pair,holds=b.v55_validate_shadow_quote_pairs(bad)
    assert not pair and holds['NO_UNIQUE_TWO_SIDED_SAME_BOOK_QUOTE']==1
    bad=[good[0],dict(good[1],line=4.5)]
    pair,holds=b.v55_validate_shadow_quote_pairs(bad)
    assert not pair and holds['QUOTE_PAIR_OPPOSITE_SPREAD_INVALID']==2
    # Empty/noncaptured signal is not invented; empty models cannot grant authority.
    ret=b.score_snapshot({}, {'artifacts':{}},[],good,[],now=now)
    assert ret['bet_authority']==0 and ret['scored_market_sides']==0
    print('PASS: paired quotes, missing twin, spread sign and authority-zero regression tests')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',default='v55_model_live_market_audit.json')
    ap.add_argument('--manifest-uri',default=os.getenv('NCAAF_V55_ML_MANIFEST_URI','gs://sharp-models/research/ncaaf/v55/ml/current_manifest.json'))
    ap.add_argument('--bucket',default='sharp-models')
    ap.add_argument('--live-sources-manifest',default=os.getenv('NCAAF_V55_SEVEN_BRAIN_MANIFEST_URI',''))
    ap.add_argument('--extra-quotes-uri',default=os.getenv('NCAAF_V55_OTHER_MARKET_QUOTES_URI',''))
    ap.add_argument('--horizon-days',type=int,default=14)
    ap.add_argument('--max-games',type=int,default=0)
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return 0
    try:
        r=run(out=args.out,manifest_uri=args.manifest_uri,bucket=args.bucket,
              horizon_days=args.horizon_days,max_games=args.max_games,
              live_manifest_uri=args.live_sources_manifest,extra_quotes_uri=args.extra_quotes_uri)
    except Exception as e:
        Path(args.out).write_text(json.dumps({'status':'AUDIT_CRASH','error':safe_error(e),
           'traceback':traceback.format_exc()[-3000:],'bet_authority':0},indent=2)+'\n')
        print('AUDIT_CRASH: '+safe_error(e),file=sys.stderr)
        return 2
    print(json.dumps({'status':r['status'],'scored_games':r['scoring'].get('scored_games',0),
          'scored_market_sides':r['scoring'].get('scored_market_sides',0),
          'model_artifacts':r['stages']['model_artifacts'], 'output':args.out},indent=2))
    return 0 if r['status'].startswith('PASS') else 1


if __name__=='__main__':sys.exit(main())
