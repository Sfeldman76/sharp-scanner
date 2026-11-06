"""Fail-closed Heavy Research handoff for validated (not predicted) betting ROI.

It consumes ORIGINAL timestamped, independently attested betting records. The
existing Stage1/Stage2 retrospectively reconstructed files cannot be promoted.
Does not create blank files, synthesize prices, or assume an OOS PASS.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import logging
from datetime import datetime, timezone

from ncaaf_v55_validated_edge import (MANIFEST, STATUS, _sha_bytes, _canonical,
                                     certify, load_edge_certificates)

PREFIX='research/ncaaf/v55/edge_validation_inputs/'
CERT_LATEST='research/ncaaf/v55/unified_combined_authority/validated_edge_certificates_latest.json'
INPUT_FILES={
    'manifest':PREFIX+'predeclared_manifest.json',
    'audit':PREFIX+'original_capture_audit.json',
    'bets':PREFIX+'attested_historical_bets.csv',
}


def run_attested_edge_validation(storage_client, bucket_name='sharp-models', *, log_func=None, now=None):
    """Run on every Heavy Research; publish only an independently verified PASS.

    Non-PASS does NOT replace a previously valid certificate. Logs a specific
    input/attestation/ROI hold. Return is JSON-serializable and fails closed.
    """
    log=log_func or logging.info
    at=now or datetime.now(timezone.utc)
    bucket=storage_client.bucket(bucket_name)
    parts={}
    for name,path in INPUT_FILES.items():
        try:
            parts[name]=bucket.blob(path).download_as_bytes()
        except Exception as exc:
            result={'status':'HOLD_ORIGINAL_PREGAME_EVIDENCE_INPUT_MISSING',
                'missing':name,'gcs_path':'gs://'+bucket_name+'/'+path,
                'detail':type(exc).__name__, 'bet_authority_changed':False}
            log('[NCAAF-V55-VALIDATED-EDGE] '+json.dumps(result,sort_keys=True))
            return result
    try:
        manifest=json.loads(parts['manifest'])
        audit=json.loads(parts['audit'])
        rows=list(csv.DictReader(io.StringIO(parts['bets'].decode('utf-8-sig'))))
        document=certify(manifest,audit,rows,source_sha256=hashlib.sha256(parts['bets']).hexdigest(),
                         created_utc=at.isoformat())
        accepted=load_edge_certificates(document,now=at)
        result={'status':document['status'],'accepted_edge_certificates':len(accepted),
                'tested_selections':document['tested_selections'],
                'rejected_selections':document.get('rejected_selections'),
                'invalid_source_rows':document.get('invalid_source_rows'),
                'bet_authority_changed':False,
                'price_authority':'EXACT_ORIGINAL_HISTORICAL_BOOK_+_CURRENT_VERIFIED_BACKEND'}
        if document['status']!=STATUS or not accepted or len(accepted)!=document['accepted_selections']:
            result['status']='HOLD_NO_VALIDATED_ROI_CERTIFICATES'
            log('[NCAAF-V55-VALIDATED-EDGE] '+json.dumps(result,sort_keys=True))
            return result
        blob=bucket.blob(CERT_LATEST)
        blob.upload_from_string(json.dumps(document,sort_keys=True,indent=2,allow_nan=False),content_type='application/json')
        result['status']='PASS_CERTIFIED_EDGE_PUBLISHED'
        result['gcs_uri']='gs://'+bucket_name+'/'+CERT_LATEST
        result['bet_authority_changed']=False  # only the subsequent live run can authorize a bet
        log('[NCAAF-V55-VALIDATED-EDGE] '+json.dumps(result,sort_keys=True))
        return result
    except Exception as exc:
        result={'status':'HOLD_EDGE_RESEARCH_VALIDATION_ERROR',
            'error':type(exc).__name__+':'+str(exc)[:240],'bet_authority_changed':False}
        log('[NCAAF-V55-VALIDATED-EDGE] '+json.dumps(result,sort_keys=True))
        return result
