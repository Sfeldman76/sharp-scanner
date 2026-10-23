"""Fail-closed, read-only preflight for a real V5.5 combined-brain installation.

This does NOT deploy, create snapshots, change the live production system, or
certify point-in-time accuracy. Designed for authenticated Cloud Shell / CI.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from apply_to_repository import MODULES, apply, MARKER
from ncaaf_heavy_brain_bridge import INPUTS


def inspect_repo(repo: Path) -> dict:
    p=repo.expanduser().resolve()
    out={"repository":str(p),"present":p.is_dir(),"train_job_present":(p/"train_job.py").is_file(),
         "v55_primary_present":(p/"ncaaf_v55_primary_research.py").is_file(),
         "production_v1_unchanged_by_preflight":True,"patch_status":"UNKNOWN"}
    if not out["train_job_present"]:
        out["patch_status"]="BLOCKED_NO_TRAIN_JOB"
        return out
    raw=(p/"train_job.py").read_text(encoding="utf-8")
    out["entry_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    out["already_integrated"] = raw.count(MARKER)==1
    try:
        assessment=apply(p,dry_run=True)
        out["patch_status"]="APPLIED" if out["already_integrated"] and not assessment["modules_copy"] else "DRY_RUN_READY"
        out["dry_run"] = assessment
    except Exception as e:
        out["patch_status"]="BLOCKED_INCOMPATIBLE_OR_CONFLICTING_SOURCE"
        out["error_type"] = type(e).__name__
        out["error"] = str(e)[:400]
    out["installed_modules"]={m:(p/m).is_file() for m in MODULES}
    return out


def inspect_gcs(bucket: str, prefix: str, client=None) -> dict:
    out={"mode":"gcs", "bucket":bucket,"input_prefix":prefix,"expected_inputs":list(INPUTS),
         "source_schema_certified":False,"point_in_time_certified":False,"history_validation_completed":False,
         "files":{},"missing_inputs":[]}
    if not prefix or prefix.startswith('/') or '..' in prefix.split('/'):
        out["status"]="BLOCKED_INVALID_PREFIX"
        return out
    try:
        if client is None:
            from google.cloud import storage
            client=storage.Client()
        b=client.bucket(bucket)
        for key in INPUTS:
            found=None
            for ext in (".parquet",".csv"):
                path=prefix.rstrip('/')+'/'+key+ext
                blob=b.blob(path)
                if blob.exists():
                    blob.reload()
                    found={"object":path,"generation":str(blob.generation),
                           "size_bytes":getattr(blob,'size',None),"crc32c":getattr(blob,'crc32c',None)}
                    break
            if found is None:
                out["missing_inputs"].append(key)
            else:
                out["files"][key]=found
        out["status"]="ALL_INPUT_OBJECTS_PRESENT_UNVERIFIED" if not out["missing_inputs"] else "BLOCKED_MISSING_INPUT_OBJECTS"
    except Exception as e:
        out["status"]="BLOCKED_CLOUD_ACCESS"
        out["error_type"]=type(e).__name__
        out["error"]=str(e)[:400]
    return out


def inspect_bigquery(env=None, client=None) -> dict:
    env=os.environ if env is None else env
    keys={k:env.get('NCAAF_BRAIN_BQ_'+k.upper()+'_TABLE','').strip() for k in INPUTS}
    out={"mode":"bigquery", "expected_inputs":list(INPUTS),
         "table_names":keys,"missing_inputs":[k for k,v in keys.items() if not v],
         "source_schema_certified":False,"point_in_time_certified":False,
         "history_validation_completed":False,"tables":{}}
    if out['missing_inputs']:
        out['status']='BLOCKED_MISSING_TABLE_BINDINGS'
        return out
    try:
        if client is None:
            from google.cloud import bigquery
            client=bigquery.Client()
        for key,table in keys.items():
            t=client.get_table(table)
            cols={f.name for f in t.schema}
            out['tables'][key]={"table":table, "rows_declared":getattr(t,'num_rows',None),
                                "season_column_present":"season" in cols,
                                "snapshot_table_attested":False}
        out['status']='ALL_TABLES_FOUND_UNVERIFIED' if all(x['season_column_present'] for x in out['tables'].values()) else 'BLOCKED_TABLE_SCHEMA'
    except Exception as e:
        out['status']='BLOCKED_CLOUD_ACCESS_OR_TABLE_MISSING'
        out['error_type']=type(e).__name__
        out['error']=str(e)[:400]
    return out


def evaluate(repo: Path, mode: str, bucket: str, prefix: str, env=None, gcs_client=None, bq_client=None) -> dict:
    source=inspect_repo(repo)
    if mode=='gcs': evidence=inspect_gcs(bucket,prefix,gcs_client)
    elif mode=='bigquery': evidence=inspect_bigquery(env,bq_client)
    else: evidence={"mode":"none","status":"BLOCKED_NO_ARCHIVE_SOURCE","history_validation_completed":False}
    code_ok=source['patch_status'] in ('APPLIED','DRY_RUN_READY') and source['v55_primary_present']
    archival_objects_present=evidence['status'] in ('ALL_INPUT_OBJECTS_PRESENT_UNVERIFIED','ALL_TABLES_FOUND_UNVERIFIED')
    return {"timestamp_utc":datetime.now(timezone.utc).isoformat(),
            "purpose":"RESEARCH_SHADOW_ONLY","bet_authority":0,"production_modified":False,
            "repository":source,"archive_inputs":evidence,
            "source_code_ready_for_local_merge":bool(code_ok),
            "archive_references_resolved":bool(archival_objects_present),
            "point_in_time_validated":False,
            "live_cloud_run_deployment_verified":False,
            "real_historical_system_validation_verified":False,
            "status":"READY_TO_TEST_MERGE_NOT_DEPLOYED" if code_ok and archival_objects_present else "HOLD",
            "reason":"CODE_AND_REFERENCES_PRESENT_BUT_NOT_ATTESTED" if code_ok and archival_objects_present
                    else "REPOSITORY_OR_HISTORICAL_INPUT_ACCESS_MISSING"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--mode',choices=('gcs','bigquery','none'),default='none')
    p.add_argument('--bucket',default='sharp-models')
    p.add_argument('--prefix',default=os.environ.get('NCAAF_BRAIN_HISTORY_INPUT_PREFIX',''))
    p.add_argument('--out',default='')
    args=p.parse_args()
    result=evaluate(args.repo,args.mode,args.bucket,args.prefix)
    payload=json.dumps(result,indent=2,default=str)+'\n'
    if args.out: Path(args.out).write_text(payload,encoding='utf-8')
    else:print(payload,end='')
    return 0 if result['status']=='READY_TO_TEST_MERGE_NOT_DEPLOYED' else 2

if __name__=='__main__':sys.exit(main())
