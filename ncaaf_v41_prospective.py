"""Research-only, first-write-wins pregame forecast lock (no prediction engine).

V4 currently publishes historical fold predictions but no fitted 2026
inference artifact. Therefore this tool refuses records without a pinned model
artifact and does not fabricate prospective picks. No production/bet writes.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from hashlib import sha256
import json
import re

MODEL = 'NONLINEAR_STATE_HGB'
PREFIX = 'research/ncaaf/v41_integrity/prospective_shadow_2026'


def utc(z):
    d = datetime.fromisoformat(z.replace('Z', '+00:00'))
    if d.tzinfo is None: raise ValueError('V41_LOCK_UNZONED_DATETIME')
    return d.astimezone(timezone.utc)


def _is_sha256(value):
    return isinstance(value, str) and bool(re.fullmatch('[a-f0-9]{64}', value))


def validate_and_prepare(record, *, clock=None):
    now = clock or datetime.now(timezone.utc)
    if now.tzinfo is None: raise ValueError('V41_LOCK_CLOCK_MUST_BE_UTC')
    now = now.astimezone(timezone.utc)
    if record.get('model') != MODEL: raise ValueError('V41_LOCK_WRONG_MODEL')
    if int(record.get('season', 0)) != 2026: raise ValueError('V41_LOCK_INVALID_SEASON')
    game_id = str(record.get('game_id', '')).strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{3,100}', game_id): raise ValueError('V41_LOCK_BAD_GAME_ID')
    team, opp = str(record.get('team', '')).strip(), str(record.get('opponent', '')).strip()
    if not team or not opp or team == opp: raise ValueError('V41_LOCK_BAD_ORIENTATION')
    kickoff = utc(record['kickoff_utc'])
    if kickoff <= now + timedelta(minutes=15): raise ValueError('V41_LOCK_TOO_CLOSE_TO_KICKOFF_OR_LATE')
    forecast = float(record['predicted_team_margin'])
    if not (-150 <= forecast <= 150): raise ValueError('V41_LOCK_INVALID_MARGIN')
    model_uri = str(record.get('immutable_model_artifact_uri', ''))
    feature_uri = str(record.get('immutable_pregame_feature_snapshot_uri', ''))
    if not (model_uri.startswith('gs://') and feature_uri.startswith('gs://')):
        raise ValueError('V41_LOCK_MISSING_PINNED_MODEL_OR_FEATURE_SNAPSHOT')
    if not (_is_sha256(record.get('model_artifact_sha256')) and
            _is_sha256(record.get('feature_snapshot_sha256'))):
        raise ValueError('V41_LOCK_MISSING_ARTIFACT_HASH')
    as_of = utc(record['feature_as_of_utc'])
    if as_of > now or as_of >= kickoff: raise ValueError('V41_LOCK_FUTURE_FEATURE_SNAPSHOT')
    if as_of < now - timedelta(days=14): raise ValueError('V41_LOCK_STALE_FEATURE_SNAPSHOT')
    forbidden = {'actual_margin', 'actual_total', 'result', 'grade', 'outcome', 'final_score',
                 'closing_line', 'bet_authority', 'settlement'}
    if forbidden.intersection(record): raise ValueError('V41_LOCK_OUTCOME_OR_PRODUCTION_FIELDS_PRESENT')
    snapshot = {
        'version': 'NCAAF_V41_SHADOW_LOCK_V1', 'status': 'LOCKED_PREGAME_SOURCE_VINTAGE_PENDING',
        'created_utc': now.isoformat(), 'sport': 'NCAAF', 'season': 2026,
        'game_id': game_id, 'team': team, 'opponent': opp,
        'kickoff_utc': kickoff.isoformat(), 'model': MODEL,
        'predicted_team_margin': forecast,
        'immutable_model_artifact_uri': model_uri,
        'model_artifact_sha256': record['model_artifact_sha256'],
        'immutable_pregame_feature_snapshot_uri': feature_uri,
        'feature_snapshot_sha256': record['feature_snapshot_sha256'],
        'feature_as_of_utc': as_of.isoformat(),
        'source_vintage_certified': False, 'bet_authority': 0,
        'production_mutation': False,
        'disclosure': 'Timestamped pregame shadow prediction; input source-vintage NOT yet certified.',
    }
    return snapshot


def lock_to_gcs(storage_client, record, *, bucket_name='sharp-models', clock=None):
    """No overwrites. A repeated call to an existing exact key fails CLOSED."""
    snapshot = validate_and_prepare(record, clock=clock)
    basename = f"{PREFIX}/{snapshot['game_id']}/{snapshot['model_artifact_sha256'][:20]}.json"
    text = json.dumps(snapshot, sort_keys=True, allow_nan=False, separators=(',', ':'))
    blob = storage_client.bucket(bucket_name).blob(basename)
    blob.upload_from_string(text, content_type='application/json', if_generation_match=0)
    return {'status': snapshot['status'], 'uri': f'gs://{bucket_name}/{basename}',
            'payload_sha256': sha256(text.encode()).hexdigest(),
            'source_vintage_certified': False}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='First-write-wins pregame lock; requires separately fitted model')
    p.add_argument('--record-json', required=True)
    p.add_argument('--bucket', default='sharp-models')
    args = p.parse_args()
    from google.cloud import storage
    with open(args.record_json, encoding='utf-8') as stream: data = json.load(stream)
    print(json.dumps(lock_to_gcs(storage.Client(), data, bucket_name=args.bucket), indent=2))
