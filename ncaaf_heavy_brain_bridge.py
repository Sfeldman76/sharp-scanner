"""Fail-closed read-only historical-brain lane for NCAAF V5.5 Heavy Research.

The existing CORE/legacy jobs are never overwritten. Read archived source
snapshots from an explicit immutable GCS prefix (or seven explicit BigQuery
source tables), validate the independent 2013-25 game universe and full
rule-level eligibility, and write research-only reports. Not a betting engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile

import pandas as pd
from ncaaf_brain_full_history import RunConfig, report, write_output
from ncaaf_brain_source_recovery import recover_existing_sources
from ncaaf_historical_brain_replay_v1 import run_heavy_retrospective

INPUTS = ("games", "core", "signals", "coverage", "eligibility", "rule_evaluations", "universe_manifest")
RESEARCH_ROOT = "research/ncaaf/v55/combined_brains_shadow"
TABLE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")


def _serialized(data):
    return json.dumps(data, indent=2, sort_keys=True, default=str, allow_nan=False).encode()


def _read_gcs_inputs(storage_client, bucket_name, prefix):
    if not prefix or prefix.startswith("/") or ".." in prefix or not re.fullmatch(r"[A-Za-z0-9_./-]+", prefix):
        raise ValueError("INVALID_IMMUTABLE_GCS_INPUT_PREFIX")
    bucket = storage_client.bucket(bucket_name)
    frames = {}
    receipts = []
    missing = []
    for name in INPUTS:
        candidates = (prefix.rstrip("/")+"/"+name+".parquet", prefix.rstrip("/")+"/"+name+".csv")
        selected = next((bucket.blob(p) for p in candidates if bucket.blob(p).exists()), None)
        if selected is None:
            missing.append(name)
            continue
        selected.reload()
        payload = selected.download_as_bytes()
        if selected.name.endswith(".parquet"):
            frame = pd.read_parquet(BytesIO(payload))
        else:
            dtype = {"game_id": "string", "system_id": "string"}
            frame = pd.read_csv(BytesIO(payload), dtype=dtype, low_memory=False)
        frames[name] = frame
        receipts.append({"input": name, "gcs": "gs://"+bucket_name+"/"+selected.name,
                         "generation": str(selected.generation), "bytes": len(payload),
                         "sha256": hashlib.sha256(payload).hexdigest(), "rows": len(frame)})
    return frames, receipts, missing


def _read_bigquery_inputs(env):
    names = {key: env.get("NCAAF_BRAIN_BQ_"+key.upper()+"_TABLE", "").strip() for key in INPUTS}
    if not any(names.values()):
        return {}, [], list(INPUTS)
    missing = [key for key, table in names.items() if not table]
    if missing:
        return {}, [], missing
    if not all(TABLE_PATTERN.fullmatch(t) for t in names.values()):
        raise ValueError("INVALID_BIGQUERY_TABLE_ID")
    from google.cloud import bigquery  # lazy import; research-only runtime
    client = bigquery.Client()
    frames, receipts = {}, []
    for name, table in names.items():
        # All seven tables must independently contain a literal 'season' field.
        # Reject a table schema that cannot be chronologically bounded.
        schema = client.get_table(table).schema
        if "season" not in {field.name for field in schema}:
            raise ValueError("BIGQUERY_SOURCE_MISSING_SEASON:"+name)
        sql = f"SELECT * FROM `{table}` WHERE SAFE_CAST(season AS INT64) BETWEEN 2013 AND 2025"
        df = client.query(sql).to_dataframe(create_bqstorage_client=False)
        frames[name] = df
        receipts.append({"input": name, "bigquery_table": table, "rows": len(df),
                         "source_version": "TABLE_SNAPSHOT_NOT_ATTESTED", "sha256": None})
    return frames, receipts, []


def _write_gcs(storage_client, bucket_name, run_prefix, result_path, status, receipts, log_func,
               recovery_frames=None, retrospective=None):
    bucket = storage_client.bucket(bucket_name)
    files = [("status.json", _serialized(status)), ("input_receipts.json", _serialized(receipts))]
    if recovery_frames:
        for source_name, source_frame in sorted(recovery_frames.items()):
            files.append(("recovered_uncertified/" + source_name + ".csv",
                          source_frame.to_csv(index=False).encode()))
    if retrospective:
        audit = retrospective.get("audit", {})
        files.append(("retrospective_replay/audit.json", _serialized(audit)))
        for name, frame in sorted(retrospective.items()):
            if isinstance(frame, pd.DataFrame):
                files.append(("retrospective_replay/" + name + ".csv", frame.to_csv(index=False).encode()))
    if result_path:
        for file in sorted(Path(result_path).glob("*.csv")):
            files.append((file.name, file.read_bytes()))
    for name, payload in files:
        blob = bucket.blob(run_prefix+"/"+name)
        blob.upload_from_string(payload, content_type="text/csv" if name.endswith(".csv") else "application/json",
                                if_generation_match=0)
    # Pointer is contained in isolated research namespace; cannot touch
    # weekly forecasts, Production V1, or any betting authority artifacts.
    bucket.blob(RESEARCH_ROOT+"/latest.json").upload_from_string(
        _serialized({"run_prefix":run_prefix,"status":status["status"],
                     "run_id":run_prefix.rsplit("/",1)[-1],"bet_authority":0}),
        content_type="application/json")
    log_func(f"[NCAAF-V55-HISTORICAL-BRAINS] output=gs://{bucket_name}/{run_prefix} status={status['status']} authority=0")


def run_heavy_brain_shadow(*, storage_client, bucket_name="sharp-models", log_func=print,
                           legacy_cache=None, legacy_research=None, v55_primary_result=None,
                           input_prefix=None, env=None, publish=True, replay_source_frames=None):
    """Invoke *only* after heavy legacy system replay; never use caches as proof.

    If the seven immutable data sources are not accessible, return and publish
    a HOLD rather than inventing historical picks or claiming zero triggers.
    """
    env = os.environ if env is None else env
    prefix = input_prefix if input_prefix is not None else env.get("NCAAF_BRAIN_HISTORY_INPUT_PREFIX", "")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_prefix = f"{RESEARCH_ROOT}/runs/{now}"
    primary = v55_primary_result if isinstance(v55_primary_result, dict) else {}
    cache = legacy_cache if isinstance(legacy_cache, dict) else {}
    secondary = legacy_research if isinstance(legacy_research, dict) else {}
    evidence = {"primary_status": primary.get("status", "UNKNOWN"),
                "legacy_cache_keys": sorted(map(str,cache.keys())),
                "legacy_research_keys": sorted(map(str,secondary.keys())),
                "legacy_outputs_do_not_certify_historical_signal_events": True}
    status = {"status":"HOLD", "reason":"MISSING_IMMUTABLE_HISTORICAL_INPUTS", "bet_authority":0,
              "production_mutation":False, "historical_universe":"2013-2025",
              "2026_selection_influence":0, "v55_primary":evidence}
    frames, receipts = {}, []
    recovery_frames = {}
    retrospective = None
    result_path = None
    try:
        if prefix:
            frames, receipts, missing = _read_gcs_inputs(storage_client,bucket_name,prefix)
        else:
            frames, receipts, missing = _read_bigquery_inputs(env)
        # If no explicit archive prefix or table mapping was supplied, recover
        # source-diagnostic CORE and game rows from work Heavy already completed.
        # These recovered frames never attest historical brain events or book prices.
        if not prefix and not any(env.get("NCAAF_BRAIN_BQ_" + k.upper() + "_TABLE", "") for k in INPUTS):
            recovery_frames, recovered_receipts, recovery_audit = recover_existing_sources(
                storage_client=storage_client, bucket_name=bucket_name,
                legacy_cache=cache, v55_primary_result=primary)
            frames.update(recovery_frames)
            receipts.extend(recovered_receipts)
            status["source_recovery"] = recovery_audit
            missing = [key for key in INPUTS if key not in frames]
            log_func("[NCAAF-V55-HISTORICAL-BRAINS-SOURCE-RECOVERY] recovered=" +
                     ",".join(recovery_audit["recovered_inputs"]) +
                     " still_missing=" + ",".join(missing) + " authority=0")
            # A separate *retrospective* lane now reconstructs explicitly
            # defined research hypotheses from CFBD + historical context + PT.
            # It never fills the five missing certified inputs above: no 2026
            # promotion, no invented Big Al picks, and no 2024-25 OOS claim.
            if str(env.get("NCAAF_BRAIN_RETROSPECTIVE_REPLAY_ENABLED", "1")).lower() not in ("0","false","no"):
                retrospective = run_heavy_retrospective(
                    storage_client=storage_client, bucket_name=bucket_name,
                    core=recovery_frames.get("core"),
                    source_frames=replay_source_frames,
                    project=env.get("GOOGLE_CLOUD_PROJECT", "sharplogger"),
                    dataset=env.get("NCAAF_HISTORY_DATASET", "sharp_data"),
                    log_func=log_func)
                audit = retrospective.get("audit", {})
                status["retrospective_replay"] = {
                    "status": audit.get("status", "HOLD"),
                    "reason": audit.get("reason", "RESEARCH_ONLY_NO_ORIGINAL_VINTAGE_CERTIFICATION"),
                    "covered_cfbd_games": (audit.get("source_coverage") or {}).get("cfbd", {}).get("games", 0),
                    "strict_certified_brain_inputs_replaced": 0,
                    "bet_authority": 0,
                    "independent_holdout_certified": False,
                }
        status["missing_inputs"] = missing
        if missing:
            status["reason"] = "HISTORICAL_BRAIN_EVENT_AND_COVERAGE_ARCHIVES_MISSING" if recovery_frames else "HISTORICAL_INPUTS_NOT_AVAILABLE_FOR_ALL_BRAINS"
            log_func("[NCAAF-V55-HISTORICAL-BRAINS] HOLD missing_inputs="+",".join(missing)+" authority=0")
        else:
            with tempfile.TemporaryDirectory(prefix="ncaaf_brains_shadow_") as td:
                r = report(games=frames["games"], signals=frames["signals"],
                           coverage=frames["coverage"], core=frames["core"], cfg=RunConfig(),
                           eligibility=frames["eligibility"], universe_manifest=frames["universe_manifest"],
                           rule_evaluations=frames["rule_evaluations"])
                result_path = write_output(r,td)
                data_status = r["status"]
                good = (data_status["universe_archive_years_verified"] == 13
                        and data_status["all_brains_full_coverage"]
                        and data_status["rule_years_certified"] > 0
                        and data_status["brain_events_system_vintage_verified"] > 0
                        and not data_status["warnings"])
                status = {**status, **data_status,
                          "status":"HISTORICAL_RESEARCH_COMPLETE_SHADOW_ONLY" if good else "HOLD",
                          "reason":"FULL_HISTORY_RESEARCH_ONLY_NO_BET_AUTHORITY" if good
                                    else "PROVENANCE_OR_HISTORY_COVERAGE_HOLD",
                          "input_receipts":receipts}
                if publish:
                    _write_gcs(storage_client,bucket_name,run_prefix,result_path,status,receipts,log_func)
                return status
    except Exception as exc:
        status["status"] = "HOLD"
        status["reason"] = "INPUT_AUDIT_OR_HISTORICAL_VALIDATION_FAILED"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)[:1000]
        log_func("[NCAAF-V55-HISTORICAL-BRAINS] HOLD error="+type(exc).__name__+" "+str(exc)[:400]+" authority=0")
    if publish:
        try:
            _write_gcs(storage_client,bucket_name,run_prefix,None,status,receipts,log_func,
                       recovery_frames=recovery_frames, retrospective=retrospective)
        except Exception as exc:
            # Failure to publish shadow metadata cannot affect weekly models.
            log_func("[NCAAF-V55-HISTORICAL-BRAINS] SHADOW_PUBLICATION_FAILED "+type(exc).__name__+" "+str(exc)[:400])
    return status
