#!/usr/bin/env python3
"""Publish the exact prebuilt V143 Pathi/Big Al occurrence ledger in Heavy Research.

Research-only sidecar. No dashboard import, no historical reconstruction,
no CORE/authority changes and no 2026 outcome use in qualification.
"""
from __future__ import annotations
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone

PREFIX = 'research/ncaaf/v55/expert_occurrence_snapshots_v1/runs/'


def _count(v, field):
    try:
        n = int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'EXPERT_BAD_{field}: {v!r}') from exc
    if n < 0: raise ValueError(f'EXPERT_NEGATIVE_{field}: {n}')
    return n


def _json_default(v):
    # Preserve timestamp representation without claiming original capture vintage.
    if hasattr(v, 'isoformat'):
        return v.isoformat()
    if hasattr(v, 'item'):
        return v.item()
    raise TypeError(f'EXPERT_UNSERIALIZABLE_TYPE:{type(v).__name__}')


def exact_cache_snapshot(cache):
    """Return exact directional-system source and reconciliation metadata."""
    if not isinstance(cache, dict) or not cache:
        raise ValueError('EXPERT_V143_CACHE_MISSING')
    out = {}
    counts = {'pathi_systems': 0, 'bigal_systems': 0,
              'pathi_fired': 0, 'pathi_graded': 0,
              'bigal_fired': 0, 'bigal_graded': 0,
              'graded_occurrence_rows': 0, 'ungraded_firings': 0}
    for name, entry in sorted(cache.items()):
        if not isinstance(entry, dict): continue
        fam = str(entry.get('family', ''))
        if str(entry.get('role', '')).lower() != 'directional' or fam not in ('Pathi', 'BigAl'):
            continue
        occurrences = entry.get('occurrences')
        if not isinstance(occurrences, list):
            raise ValueError(f'EXPERT_MISSING_OCCURRENCES:{name}')
        fired = _count(entry.get('fired', 0), 'FIRED')
        graded = _count(entry.get('graded', entry.get('sample', 0)), 'GRADED')
        if fired < graded or len(occurrences) != graded:
            raise ValueError(f'EXPERT_OCCURRENCE_RECONCILIATION_FAIL:{name}:fired={fired}:graded={graded}:rows={len(occurrences)}')
        for i, row in enumerate(occurrences):
            if not isinstance(row, dict) or not row.get('season') or not row.get('date') or not row.get('team') or not row.get('opponent'):
                raise ValueError(f'EXPERT_OCCURRENCE_REQUIRED_FIELDS_MISSING:{name}:{i}')
            if row.get('ats_win') is None:
                raise ValueError(f'EXPERT_GRADED_NO_ATS_RESULT:{name}:{i}')
        # Copy ONLY fields consumed by the established exporter. This is a
        # faithful projection of recorded occurrences, not new pick discovery.
        out[str(name)] = {
            'family': fam, 'role': 'directional',
            'fired': fired, 'graded': graded,
            'occurrences': occurrences,
        }
        fkey = 'pathi' if fam == 'Pathi' else 'bigal'
        counts[f'{fkey}_systems'] += 1
        counts[f'{fkey}_fired'] += fired
        counts[f'{fkey}_graded'] += graded
        counts['graded_occurrence_rows'] += len(occurrences)
        counts['ungraded_firings'] += fired - graded
    if not out or not counts['graded_occurrence_rows']:
        raise ValueError('EXPERT_NO_DIRECTIONAL_GRADED_OCCURRENCES')
    payload = (json.dumps(out, sort_keys=True, default=_json_default, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
    return payload, counts


def publish_existing_expert_cache(*, dashboard_module, storage_client, bucket_name='sharp-models', log_func=print, clock=None):
    """Publish once during *existing* Heavy Research after its full reconciliation.

    Caller owns failures and must leave CORE/authority unchanged. GCS objects
    use generation preconditions and never replace previous archives.
    """
    cache = getattr(dashboard_module, '_V143_SYSTEM_HISTORY_CACHE', None)
    raw, counts = exact_cache_snapshot(cache)
    timestamp = (clock or datetime.now(timezone.utc)).strftime('%Y%m%dT%H%M%S%fZ')
    run = timestamp + '_' + uuid.uuid4().hex[:12]
    prefix = PREFIX + run + '/'
    sha = hashlib.sha256(raw).hexdigest()
    audit = {
        'status': 'PASS_EXACT_V143_EXPERT_CACHE_SNAPSHOT_RESEARCH_ONLY',
        'source': 'sharp_line_dashboard._V143_SYSTEM_HISTORY_CACHE',
        'source_cache_already_constructed': True,
        'no_extra_historical_research_executed': True,
        'evidence_class': 'RETROSPECTIVE_GRADED_SYSTEM_OCCURRENCE_NOT_AUTHENTIC_HISTORICAL_PICK',
        'historical_qualification_certified': False,
        'pregame_capture_time_certified': False,
        'actual_market_quote_certified': False,
        'independent_bet_authority': 0,
        'core_changed': False, 'production_mutation': False,
        'graded_cache_sha256': sha, 'counts': counts,
        'cache_gcs_uri': f'gs://{bucket_name}/{prefix}v143_directional_cache.json',
    }
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(prefix + 'v143_directional_cache.json').upload_from_string(
        raw, content_type='application/json', if_generation_match=0)
    bucket.blob(prefix + 'snapshot_audit.json').upload_from_string(
        json.dumps(audit, sort_keys=True, indent=2).encode() + b'\n',
        content_type='application/json', if_generation_match=0)
    log_func('[NCAAF-V55-EXPERT-CACHE-SNAPSHOT] '
             f'status=PASS graded={counts["graded_occurrence_rows"]} '
             f'pathi={counts["pathi_graded"]} bigal={counts["bigal_graded"]} '
             f'uri={audit["cache_gcs_uri"]} authority=0')
    return audit
