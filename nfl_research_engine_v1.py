"""NFL V1.9.3 protected challenger research engine.

This is the first modeling run on top of the V1.9.2 research foundation.  It
never queries 2026.  It uses 2017-2025 history, season-forward development,
freezes new challenger specifications, and writes only research artifacts.
"""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone

import joblib
import pandas as pd

from nfl_feature_audit_v1 import VIEW
from nfl_challenger_v1 import REQUIRED_COLUMNS, build_readonly_query, physical_games
from nfl_intelligence_v1 import _generate_oof_core
from nfl_stats_context_v1 import RAW, RAW_STATS_COLUMNS, derive_stats_context, attach_stats_features
from nfl_research_contract_v1 import assert_contract, contract_hash, CONTRACT_VERSION
from nfl_structured_research_v1 import run_structured_research, SOURCE_TAG as STRUCTURED_SOURCE_TAG
from nfl_residual_miner_v2 import run_residual_miner_v2, SOURCE_TAG as MINER_SOURCE_TAG

SOURCE_TAG = "nfl-research-engine-v1.9.3-protected-challengers-20261001"
PRODUCTION_AUTHORITY = 0
DATA_MAX_SEASON = 2025
RESEARCH_CLOCK_ID = "NFL_V1_9_3_POST_DEPLOYMENT_PROSPECTIVE_CLOCK"


def _raw_query(raw_columns):
    miss=set(RAW_STATS_COLUMNS)-set(raw_columns)
    if miss:
        raise RuntimeError("NFL_V1_9_3_RAW_COLUMNS_MISSING "+str(sorted(miss)))
    q=", ".join(f"`{c}`" for c in RAW_STATS_COLUMNS)
    return (f"SELECT {q} FROM `{RAW}` WHERE Season BETWEEN 2017 AND 2025 "
            "AND Season_Stage IN ('REGULAR','POSTSEASON') ORDER BY Season, Game_Date, Source_Name, Source_Game_ID, Team_Norm")


def _upload_immutable(storage_client, bucket_name: str, name: str, data: bytes, content_type: str):
    from google.api_core.exceptions import PreconditionFailed
    blob=storage_client.bucket(bucket_name).blob(name)
    try:
        blob.upload_from_string(data,content_type=content_type,if_generation_match=0)
        return {"uri":f"gs://{bucket_name}/{name}","created":True}
    except PreconditionFailed:
        return {"uri":f"gs://{bucket_name}/{name}","created":False,"reason":"ALREADY_EXISTS_IMMUTABLE"}


def run_nfl_research_engine_v1(*, bq_client, storage_client, bucket_name="sharp-models", audit_report=None, ledger_health=None, log_func=print):
    if not isinstance(audit_report,dict) or audit_report.get("status")!="READY_FOR_OFFLINE_CHALLENGER_SANDBOX":
        raise RuntimeError("NFL_V1_9_3_AUDIT_NOT_GREEN")
    protected=assert_contract()

    view_cols={f.name for f in bq_client.get_table(VIEW).schema}
    query=build_readonly_query(view_cols)  # hard-coded 2017-2025 in audited V1.4 builder
    raw_cols={f.name for f in bq_client.get_table(RAW).schema}
    rquery=_raw_query(raw_cols)

    log_func(f"[NFL-V1.9.3-PREFLIGHT] status=PASS source_tag={SOURCE_TAG} data_through=2025 year_2026_queried=FALSE production_authority=0 protected_contract_sha256={contract_hash()}")
    side=bq_client.query(query).to_dataframe(create_bqstorage_client=False)
    raw=bq_client.query(rquery).to_dataframe(create_bqstorage_client=False)
    if int(pd.to_numeric(side.Season,errors='coerce').max())>2025 or int(pd.to_numeric(raw.Season,errors='coerce').max())>2025:
        raise RuntimeError("NFL_V1_9_3_2026_DATA_LEAK")
    side_counts={int(k):int(v) for k,v in side.groupby("Season").size().items()}
    raw_counts={int(k):int(v) for k,v in raw.groupby("Season").size().items()}
    if side_counts != raw_counts:
        raise RuntimeError(f"NFL_V1_9_3_RAW_VIEW_GRAIN_MISMATCH side={side_counts} raw={raw_counts}")
    if any(v % 2 for v in side_counts.values()):
        raise RuntimeError("NFL_V1_9_3_ODD_SIDE_ROW_COUNT")
    log_func("[NFL-V1.9.3-GRAIN] "+json.dumps({"side_rows":int(len(side)),"physical_games":int(len(side)//2),"games_by_season":{str(k):v//2 for k,v in side_counts.items()},"year_2026_queried":False},sort_keys=True))

    games=physical_games(side)
    games["actual_margin"]=pd.to_numeric(games.Team_Score,errors="coerce")-pd.to_numeric(games.Opponent_Score,errors="coerce")
    games["actual_total"]=pd.to_numeric(games.Team_Score,errors="coerce")+pd.to_numeric(games.Opponent_Score,errors="coerce")
    stats_side=derive_stats_context(raw)
    stats_games=attach_stats_features(games,stats_side)
    core_oof=_generate_oof_core(side)

    structured_report, structured_oof, model_bundle = run_structured_research(stats_games,core_oof,log_func=log_func)
    miner_report=run_residual_miner_v2(structured_oof,log_func=log_func)

    registry={
        "engine_source_tag":SOURCE_TAG,
        "protected_contract_version":CONTRACT_VERSION,
        "protected_contract_sha256":contract_hash(),
        "structured_source_tag":STRUCTURED_SOURCE_TAG,
        "structured_registry_sha256":structured_report["structured_registry_sha256"],
        "miner_source_tag":MINER_SOURCE_TAG,
        "miner_registry_sha256":miner_report["miner_registry_sha256"],
        "selected_residual_families":structured_report["prospective_selected_residual_families"],
        "miner_promising_rule_ids":{
            m:[r["rule_id"] for r in miner_report["markets"][m]["promising_rules"]]
            for m in ("spreads","totals")
        },
        "data_through":2025,
        "year_2026_queried":False,
        "research_clock_id":RESEARCH_CLOCK_ID,
        "automatic_promotion":False,
        "production_authority":0,
    }
    registry_sha=hashlib.sha256(json.dumps(registry,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    registry["challenger_registry_sha256"]=registry_sha

    model_bundle["metadata"].update({
        "engine_source_tag":SOURCE_TAG,
        "protected_contract_sha256":contract_hash(),
        "miner_registry_sha256":miner_report["miner_registry_sha256"],
        "challenger_registry_sha256":registry_sha,
        "research_clock_id":RESEARCH_CLOCK_ID,
        "miner_rules":{m:miner_report["markets"][m]["promising_rules"] for m in ("spreads","totals")},
    })

    report={
        "status":"NFL_V1_9_3_RESEARCH_ENGINE_FROZEN_FOR_PROSPECTIVE_SHADOW",
        "source_tag":SOURCE_TAG,
        "challenger_registry":registry,
        "structured":structured_report,
        "miner":miner_report,
        "ledger_health":ledger_health or {"status":"NOT_CHECKED"},
        "protected_contract":protected,
        "training_games":int(len(stats_games)),
        "oof_games":int(len(structured_oof)),
        "year_2026_queried":False,
        "production_authority":0,
        "ncaaf":"UNCHANGED",
        "legacy_nfl":"UNCHANGED",
    }

    bio=io.BytesIO(); joblib.dump(model_bundle,bio,compress=3); model_bytes=bio.getvalue()
    report_bytes=json.dumps(report,sort_keys=True,default=str,indent=2).encode("utf-8")
    prefix=f"nfl-research/v1_9_3/{registry_sha[:16]}"
    model_art=_upload_immutable(storage_client,bucket_name,f"{prefix}/research_bundle.joblib",model_bytes,"application/octet-stream")
    report_art=_upload_immutable(storage_client,bucket_name,f"{prefix}/research_report.json",report_bytes,"application/json")
    registry_art=_upload_immutable(storage_client,bucket_name,f"{prefix}/challenger_registry.json",json.dumps(registry,sort_keys=True,indent=2).encode(),"application/json")
    report["artifacts"]={"model_bundle":model_art,"report":report_art,"registry":registry_art}
    report["frozen_at_utc"]=datetime.now(timezone.utc).isoformat()

    # Concise final contract log; detailed structured/miner logs are emitted by submodules.
    log_func("[NFL-V1.9.3-CONTRACT] "+json.dumps({
        "status":report["status"],
        "challenger_registry_sha256":registry_sha,
        "selected_residual_families":registry["selected_residual_families"],
        "miner_promising_rule_ids":registry["miner_promising_rule_ids"],
        "research_clock_id":RESEARCH_CLOCK_ID,
        "year_2026_queried":False,
        "artifacts":report["artifacts"],
        "production_authority":0,
        "ncaaf":"UNCHANGED","legacy_nfl":"UNCHANGED",
    },sort_keys=True,default=str))
    return report
