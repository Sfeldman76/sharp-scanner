#!/usr/bin/env python3
"""Add research-only reporting to the existing NCAAF Heavy Research bridge.

Importing this module has no side effects.  The original bridge result is always
returned unchanged: a research-only PASS never changes its certification HOLD,
production board, CORE, or bet authority.  GCS status objects are research-only.
"""
from __future__ import annotations

import functools
import inspect
import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from ncaaf_v55_research_status import ResearchIntegrityError, classify_bridge_result

DEFAULT_PREFIX = (
    'research/ncaaf/v55/combined_brains_shadow/input_staging/runs/'
    '20261103T155254935424Z_b67a2c4d'
)
RESEARCH_AUDIT_PREFIX = 'research/ncaaf/v55/research_execution_status'
MINER_REPLAY_PREFIX = 'research/ncaaf/v2/miner_exact_path_shadow/runs/'
REQUIRED_DISCOVERY_SOURCE = 'ncaaf_unified_history_miner_v5'


def _utcnow():
    return datetime.now(timezone.utc)


def _boolenv(name: str, default: str = '0') -> bool:
    return os.getenv(name, default).strip().lower() in {'1','true','yes','on'}


def _as_utc(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return None


def _get_arg(original, args, kwargs, key, default=None):
    # Works whether bridge is called with positional or named arguments.
    try:
        bound = inspect.signature(original).bind_partial(*args, **kwargs)
        if key in bound.arguments:
            return bound.arguments[key]
    except (TypeError, ValueError):
        pass
    return kwargs.get(key, default)


def _research_prefix(original, args, kwargs):
    provided = _get_arg(original,args,kwargs,'input_prefix')
    if provided:
        return provided
    custom_env = _get_arg(original,args,kwargs,'env')
    if isinstance(custom_env, dict) and custom_env.get('NCAAF_BRAIN_HISTORY_INPUT_PREFIX'):
        return custom_env['NCAAF_BRAIN_HISTORY_INPUT_PREFIX']
    return os.getenv('NCAAF_BRAIN_HISTORY_INPUT_PREFIX', DEFAULT_PREFIX)


def check_discovery_provenance(storage_client, bucket_name: str, *, now=None, max_age_hours=8):
    """Only inspect prior Miner artifact. Do NOT claim this resumes mining.

    Existing V5 research runs in ncaaf_research_v2 ->
    ncaaf_unified_history_miner_v5. This is an independent freshness/lineage
    checkpoint, not new proof of point-in-time validity or a promotion signal.
    """
    now = now or _utcnow()
    result = {
        'status':'NOT_VERIFIED', 'full_history_discovery_executed_this_job':False,
        'legacy_incumbent_retirement_verified':False,
        'new_systems_authorized':False, 'bet_authority':0,
        'note':'Existing Miner research must be verified separately; bridge replay does not rebuild fresh systems.',
    }
    if storage_client is None:
        result['reason']='STORAGE_CLIENT_MISSING'
        return result
    try:
        blobs = (blob for blob in storage_client.list_blobs(bucket_name, prefix=MINER_REPLAY_PREFIX)
                 if blob.name.endswith('/audit.json'))
        candidates = sorted(blobs, key=lambda b: _as_utc(getattr(b,'updated',None)) or
                            datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        if not candidates:
            result['reason']='NO_MINER_EXACT_REPLAY_AUDITS'
            return result
        recent = candidates[0]
        updated = _as_utc(getattr(recent,'updated',None))
        result['latest_audit_uri']=f'gs://{bucket_name}/{recent.name}'
        result['latest_audit_updated_utc']=updated.isoformat() if updated else None
        if updated is None or updated < now - timedelta(hours=max_age_hours):
            result['reason']='MINER_AUDIT_STALE_OR_UNDATED'
            return result
        doc = json.loads(recent.download_as_text())
        # Do not infer a full-history rebuild from a generic Miner audit.
        text = json.dumps(doc, sort_keys=True, default=str)
        if ('unified_full_history_miner_v5' not in text and
                'UNIFIED_HISTORY_MINER_V5' not in text and
                'new_full_history_paths_v5' not in text):
            result['reason']='NO_UNIFIED_FULL_HISTORY_EVIDENCE_IN_AUDIT'
            return result
        result.update({
            'status':'RECENT_FULL_HISTORY_SHADOW_EVIDENCE_OBSERVED',
            'reason':'AUDIT_EXISTS_BUT_THIS_JOB_CAUSALITY_AND_LEGACY_RETIREMENT_UNVERIFIED',
            'historical_research_only':True,
        })
        return result
    except Exception as exc:
        result['reason']='MINER_AUDIT_READ_ERROR'
        result['exception']=type(exc).__name__
        return result


def build_research_audit(bridge_result: dict, *, input_prefix: str,
                         storage_client=None, bucket_name='sharp-models', now=None):
    now = now or _utcnow()
    audit = classify_bridge_result(bridge_result,input_prefix=input_prefix)
    audit['status_source']='DEPLOYED_BRIDGE_VALIDATED_SEVEN_INPUTS'
    audit['scheduled_heavy_research_hook']=True
    audit['discovery_evidence']=check_discovery_provenance(storage_client,bucket_name,now=now)
    audit['no_legacy_promotion']=True
    return audit


def publish_audit(storage_client, bucket_name: str, audit: dict, *, now=None):
    """Write immutable run audit in a separate research-only namespace.

    Deliberately does NOT update production latest pointers, current Miner
    registry, any betting board, or original bridge HOLD.
    """
    now = now or _utcnow()
    if audit.get('research_execution_status')!='PASS_RESEARCH_ONLY':
        raise ResearchIntegrityError('REFUSE_TO_PUBLISH_NON_PASS_RESEARCH_AUDIT')
    if audit.get('bet_authority') != 0 or audit.get('production_mutation') is not False:
        raise ResearchIntegrityError('UNEXPECTED_AUTHORITY_IN_RESEARCH_AUDIT')
    if storage_client is None:
        raise ResearchIntegrityError('MISSING_STORAGE_CLIENT')
    suffix = now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:12]
    name = f'{RESEARCH_AUDIT_PREFIX}/runs/{suffix}/status.json'
    payload = json.dumps(audit, sort_keys=True, indent=2, default=str)
    storage_client.bucket(bucket_name).blob(name).upload_from_string(
        payload, content_type='application/json', if_generation_match=0)
    return f'gs://{bucket_name}/{name}'


def wrap_heavy_brain_shadow(original):
    """Instrument the installed bridge once; all existing calls remain intact.

    Requires NCAAF_V55_RESEARCH_STATUS_ENABLED=1 in the scheduled Cloud Run job.
    Fail-closed reporting: it never turns research status PASS when the original
    result is missing/invalid. The underlying job outcome is not modified.
    """
    if getattr(original,'_ncaaf_v55_research_hook_wrapped',False):
        return original

    @functools.wraps(original)
    def wrapped(*args,**kwargs):
        result = original(*args,**kwargs)
        if not _boolenv('NCAAF_V55_RESEARCH_STATUS_ENABLED'):
            return result
        log = _get_arg(original,args,kwargs,'log_func',print)
        if not callable(log):
            log = print
        client = _get_arg(original,args,kwargs,'storage_client')
        if client is None:
            from google.cloud import storage
            client = storage.Client()
        bucket_name = _get_arg(original,args,kwargs,'bucket_name','sharp-models')
        prefix = _research_prefix(original,args,kwargs)
        try:
            audit = build_research_audit(result,input_prefix=prefix,
                                         storage_client=client,bucket_name=bucket_name)
            # Emit a machine-readable status to Cloud Run logs even when GCS
            # publication is disabled for local validation.
            publish = _boolenv('NCAAF_V55_RESEARCH_STATUS_PUBLISH')
            uri = publish_audit(client,bucket_name,audit) if publish else None
            log('[NCAAF-V55-RESEARCH-EXECUTION] '+json.dumps({
                'research_execution_status':audit['research_execution_status'],
                'historical_certification_status':audit['historical_certification_status'],
                'betting_authority_status':audit['betting_authority_status'],
                'discovery_status':audit['discovery_evidence']['status'],
                'audit_uri':uri, 'original_bridge_status':result.get('status'),
                'bet_authority':0, 'production_mutation':False,
            },sort_keys=True))
        except ResearchIntegrityError as exc:
            log('[NCAAF-V55-RESEARCH-EXECUTION] '+json.dumps({
                'research_execution_status':'BLOCKED',
                'reason':str(exc),'bridge_status':result.get('status') if isinstance(result,dict) else None,
                'bet_authority':0, 'production_mutation':False,
            },sort_keys=True))
        except Exception as exc:
            # Audit failures are visible, never reported as PASS or hidden.
            log('[NCAAF-V55-RESEARCH-EXECUTION] '+json.dumps({
                'research_execution_status':'ERROR',
                'error_type':type(exc).__name__, 'message':str(exc)[:300],
            },sort_keys=True))
        return result

    wrapped._ncaaf_v55_research_hook_wrapped=True
    return wrapped
