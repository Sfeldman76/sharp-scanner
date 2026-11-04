"""Conservative GCS -> existing V5.5 qualified-event bridge.

Does NOT mine, select, score, or qualify bets. Allows only separately certified
full-history Miner registries into the *existing* live event evaluator. Existing
independent-brain validator is reused unmodified; retrospective occurrence
caches are NEVER treated as pregame-qualified signals.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone

MINER_BLOB = 'research/ncaaf/v55/qualified_full_history_miner_registry_v1.json'
MINER_POLICY = 'V55-FULL-HISTORY-MINER-2013-2025-R1'
MINER_ORIGIN = 'FRESH_FULL_HISTORY_REMINED_2013_2025'
NONMINER_BLOB_ENV = 'NCAAF_V55_CERTIFIED_NONMINER_REGISTRY_BLOB'
SEASONS = list(range(2013, 2026))


def _valid_blob_path(s):
    return bool(s and s.startswith('research/ncaaf/v55/') and
                '..' not in s and s.endswith('.json') and
                re.fullmatch(r'[A-Za-z0-9_./-]+', s))


def _get_json(storage_client, bucket_name, path):
    if not _valid_blob_path(path):
        raise ValueError('UNCERTIFIED_REGISTRY_BLOB_LOCATION')
    if storage_client is None:
        from google.cloud import storage
        storage_client = storage.Client()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', bucket_name):
        raise ValueError('INVALID_BUCKET')
    raw = storage_client.bucket(bucket_name).blob(path).download_as_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _time(s, *, now=None):
    try:
        d=datetime.fromisoformat(str(s).replace('Z', '+00:00'))
        if d.tzinfo is None: return None
        d=d.astimezone(timezone.utc)
        if d> (now or datetime.now(timezone.utc)): return None
        return d
    except (ValueError, TypeError, OverflowError):
        return None


def validate_full_history_miner_registry(doc, *, digest='', now=None):
    """Validate upstream full-history certification without manufacturing it.

    Entries are certificates only when source publisher supplied strict OOS,
    feature-vintage, independently mined, and historical period evidence.
    This does not claim to audit the upstream raw feature vintages itself.
    """
    if not isinstance(doc,dict) or doc.get('status')!='FROZEN_OOS_QUALIFIED':
        return None,'MINER_REGISTRY_NOT_FROZEN_OOS'
    if (doc.get('policy')!=MINER_POLICY or doc.get('operator_approved') is not True
            or doc.get('origin')!=MINER_ORIGIN
            or doc.get('uses_2026_outcomes') is not False):
        return None,'MINER_NEW_FULL_HISTORY_CERTIFICATION_MISSING'
    try:
        first=int(doc['history_first_season'])
        last=int(doc['discovery_last_season'])
        confirm=sorted(int(y) for y in doc['confirmation_seasons'])
        observed=sorted(int(y) for y in doc['source_audit']['observed_seasons'])
    except (KeyError,ValueError,TypeError):
        return None,'MINER_HISTORY_CONTRACT_MISSING'
    if first>2013 or last>2023 or confirm!=[2024,2025] or observed!=SEASONS:
        return None,'MINER_HISTORY_OR_HOLDOUT_CONTRACT_FAIL'
    created=_time(doc.get('created_utc'),now=now)
    if not created:
        return None,'MINER_CERTIFICATE_TIMESTAMP_INVALID'
    found={}; rejected={}
    def reject(code):rejected[code]=rejected.get(code,0)+1
    source=doc.get('mechanisms')
    if not isinstance(source,list):return None,'MINER_CERTIFIED_MECHANISMS_MISSING'
    for rec in source:
        if not isinstance(rec,dict):reject('NON_OBJECT');continue
        mid=str(rec.get('mechanism_id') or '').strip()
        fam=str(rec.get('evidence_family_key') or '').strip()
        conditions=rec.get('representative_conditions')
        if (not re.fullmatch(r'NCAAF-MECH-[A-Za-z0-9_-]+',mid) or mid in found
                or not fam or not isinstance(conditions,list) or not conditions
                or not all(isinstance(c,str) and c.strip() for c in conditions)):
            reject('IDENTITY_OR_CONDITIONS');continue
        if any(re.search(r'(?:2026|POSTGAME|POST_GAME|FINAL_MARGIN|ATS_OUTCOME)',c,re.I)
               for c in conditions):
            reject('FORBIDDEN_POSTGAME_OR_2026_ATOM');continue
        if (rec.get('origin')!=MINER_ORIGIN or rec.get('legacy_incumbent') is not False
                or rec.get('market')!='spreads' or rec.get('role')!='directional'
                or rec.get('direction') not in ('PLAY_ON','FADE')
                or rec.get('uses_2026_outcomes') is not False
                or rec.get('live_authority_eligible') is not True
                or rec.get('oos_pass') is not True
                or rec.get('split_disjoint') is not True
                or rec.get('pregame_feature_vintage_pass') is not True
                or rec.get('source_neutral') is not True
                or rec.get('overlap_audit')!='PASS'
                or rec.get('multiple_testing_audit')!='PASS'):
            reject('CERTIFICATION_FAIL');continue
        try:
            years=rec['by_season']
            a,b=years['2024'],years['2025']
            n1,n2=int(a['n']),int(b['n'])
            r1,r2=float(a['rate']),float(b['rate'])
            n=int(rec['confirmation_n']);rate=float(rec['confirmation_rate'])
            if not (n1>=15 and n2>=15 and n>=60 and n==n1+n2 and rate>=.56
                    and 0<=r1<=1 and 0<=r2<=1
                    and abs((n1*r1+n2*r2)/n-rate)<.001):
                raise ValueError('OOS_THRESHOLD_OR_PARITY')
        except (KeyError,TypeError,ValueError,OverflowError,ZeroDivisionError):
            reject('INVALID_OOS_COUNTS');continue
        found[mid]={'target':rec['direction'], 'evidence_family_key':fam,
                    'rule':' AND '.join(conditions), 'n':n,'rate':rate}
    if not found:
        return None,'MINER_NO_CERTIFIED_FULL_HISTORY_RULES:'+json.dumps(rejected,sort_keys=True)
    return {'mechanisms':found,'created_utc':created,'report_sha256':digest,
            'source_seasons':'2013-2025','rejected_by_reason':rejected,
            'certification_contract':MINER_POLICY},None


def connect_miner_registry(existing_registry, existing_problem, *, storage_client=None,
                           bucket_name=None, blob_path=None, now=None):
    """Replace legacy registry with *certified fresh-mined* one or fail closed.

    Does not accept legacy incumbent fallback. No evidence or votes are made.
    """
    bucket=bucket_name or os.getenv('MODEL_BUCKET','sharp-models')
    path=blob_path or os.getenv('NCAAF_V55_CERTIFIED_MINER_REGISTRY_BLOB',MINER_BLOB)
    status={'source':'FULL_HISTORY_MINER_CERTIFICATE_ONLY','blob':path,
            'legacy_authority_fallback':False,'certified_systems':0,
            'previous_registry_status':existing_problem or 'PASS'}
    try:
        doc,digest=_get_json(storage_client,bucket,path)
        result,problem=validate_full_history_miner_registry(doc,digest=digest,now=now)
    except Exception as exc:
        result,problem=None,'CERTIFIED_MINER_REGISTRY_UNAVAILABLE:'+type(exc).__name__
    if result:
        result['uri']=f'gs://{bucket}/{path}'
        status.update(status='PASS_FRESH_FULL_HISTORY_MINER',certified_systems=len(result['mechanisms']),
                      sha256=digest,uri=result['uri'])
        return result,None,status
    status['status']=problem
    return None,problem,status


def connect_independent_registry(existing_registry, existing_problem, *,
                                 validator, storage_client=None, bucket_name=None):
    """Use original board validator for independent brain qualifications.

    If no certified alternate was explicitly configured, preserve original
    validator's result. Research-only Pathi/Big Al occurrence data is not used.
    """
    path=os.getenv(NONMINER_BLOB_ENV,'').strip()
    status={'source':'EXISTING_FROZEN_OOS_VALIDATOR','status':existing_problem or 'PASS',
            'alternate_path_requested':bool(path),'certified_systems':len((existing_registry or {}).get('rules',{}))}
    if not path:return existing_registry,existing_problem,status
    bucket=bucket_name or os.getenv('MODEL_BUCKET','sharp-models')
    try:
        doc,digest=_get_json(storage_client,bucket,path)
        result,problem=validator(doc,raw_sha=digest)
    except Exception as exc:
        result,problem=None,'CERTIFIED_BRAIN_REGISTRY_UNAVAILABLE:'+type(exc).__name__
    if not result or not result.get('rules'):
        problem=problem or 'NO_INDEPENDENT_FROZEN_OOS_RULES'
        status['status']=problem
        return None,problem,status
    result['uri']=f'gs://{bucket}/{path}'
    status.update(source='EXISTING_FROZEN_OOS_VALIDATOR',status='PASS',
                  certified_systems=len(result['rules']),uri=result['uri'],sha256=digest)
    return result,None,status
