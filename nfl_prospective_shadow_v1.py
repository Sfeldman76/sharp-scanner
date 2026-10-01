"""NFL V1.9.5 prospective shadow orchestrator.

The purpose of V1.9.5 is to stop inventing more offline transforms of the same
box-score history and start collecting genuinely prospective, timestamped market
microstructure evidence.  It is safe to run repeatedly (recommended cadence:
30 minutes); every table is append-only and idempotent by deterministic event id.
"""
from __future__ import annotations

import json
import os
from datetime import timezone

import pandas as pd
from google.cloud import bigquery as b
from google.cloud import storage

import nfl_prospective_ledger_v4 as ledger
import nfl_market_shadow_v1 as market
import nfl_system_shadow_v1 as system_shadow

SOURCE_TAG = "nfl-prospective-shadow-v1.9.5-new-information-clock-20261001"
PRODUCTION_AUTHORITY = 0
RECOMMENDED_CADENCE_MINUTES = 30


def _verify_frozen_bundle(storage_client, bucket_name="sharp-models"):
    expected=ledger.FROZEN_V194_BUNDLE_URI
    prefix=f"gs://{bucket_name}/"
    if not expected.startswith(prefix):
        raise RuntimeError(f"NFL_V1_9_5_BUNDLE_BUCKET_MISMATCH expected={expected} bucket={bucket_name}")
    key=expected[len(prefix):]
    blob=storage_client.bucket(bucket_name).blob(key)
    if not blob.exists():
        raise RuntimeError(f"NFL_V1_9_5_FROZEN_BUNDLE_MISSING {expected}")
    blob.reload()
    return {"uri":expected,"size_bytes":int(blob.size or 0),"generation":str(blob.generation or ""),"exists":True}


def _unsettled_states(client):
    q=f"""
      SELECT s.* FROM `{ledger.STATE_TABLE}` s
      LEFT JOIN `{ledger.STATE_RESULT_TABLE}` r USING(state_event_id)
      WHERE r.state_event_id IS NULL AND s.game_start < CURRENT_TIMESTAMP()
    """
    return client.query(q).to_dataframe(create_bqstorage_client=False)


def run_nfl_prospective_shadow_v1(*, bq_client=None, storage_client=None, bucket_name="sharp-models", log_func=print, now=None):
    c=bq_client or b.Client(project=ledger.PROJECT_ID)
    gcs=storage_client or storage.Client(project=ledger.PROJECT_ID)
    started=pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now,utc=True)

    health=ledger.ledger_health_check(c,service_identity_hint="sharp-train-sa@sharplogger.iam.gserviceaccount.com")
    frozen=_verify_frozen_bundle(gcs,bucket_name)
    clock=ledger.get_or_create_research_clock(c,now=started)
    clock_start=pd.to_datetime(clock["activated_at"],utc=True)

    log_func("[NFL-V1.9.5-PREFLIGHT] "+json.dumps({
        "status":"PASS","source_tag":SOURCE_TAG,"research_clock_id":ledger.RESEARCH_CLOCK_ID,
        "clock_activated_at":clock_start.isoformat(),"clock_created_now":bool(clock.get("created_now",False)),
        "preclock_quotes_admitted":False,"frozen_registry_sha256":ledger.FROZEN_V194_REGISTRY_SHA256,
        "edge_gate_registry_sha256":ledger.FROZEN_V194_EDGE_GATE_SHA256,"frozen_bundle":frozen,
        "production_authority":0,"legacy_nfl":"UNCHANGED","ncaaf":"UNCHANGED",
    },sort_keys=True,default=str))

    source,source_map=market.fetch_market_rows(c,clock_start=clock_start,now=started,lookahead_days=int(os.getenv("NFL_SHADOW_LOOKAHEAD_DAYS","8")))
    quotes,qmeta=market.canonicalize_quotes(source,clock_start=clock_start,captured_at=started)
    qwrite=ledger.append_idempotent(quotes,table=ledger.QUOTE_TABLE,id_col="quote_event_id",time_cols=("captured_at","game_start","snapshot_timestamp"),client=c)

    # State construction uses the complete post-clock source window available in this run.
    states,smeta=market.build_market_states(quotes,clock_start=clock_start,now=started)
    swrite=ledger.append_idempotent(states,table=ledger.STATE_TABLE,id_col="state_event_id",time_cols=("captured_at","game_start","target_asof","quote_cutoff_timestamp"),client=c)

    # Settle prior states without touching prediction/state rows.
    pending=_unsettled_states(c)
    if pending.empty:
        results=pd.DataFrame(); rwrite={"status":"NO_ROWS","inserted":0,"existing":0}
    else:
        scores=market.fetch_final_scores(c,since=clock_start-pd.Timedelta(days=1))
        results=market.settle_states(pending,scores,settled_at=started)
        rwrite=ledger.append_idempotent(results,table=ledger.STATE_RESULT_TABLE,id_col="state_result_event_id",time_cols=("settled_at",),client=c)

    # Freeze/track the newly discovered NFL-native ROLE_FLIP_FADE system family
    # as one correlated SYSTEM family. This lane never mines or retunes rules.
    system_family = system_shadow.run_system_family_shadow(
        bq_client=c, storage_client=gcs, bucket_name=bucket_name, now=started,
        lookahead_days=int(os.getenv("NFL_SHADOW_LOOKAHEAD_DAYS","8")), log_func=log_func,
    )

    # Exact incumbent CORE live prediction capture is intentionally not fabricated.
    # V1.9.5 first creates a clean new-information clock; exact live feature parity
    # will be a separate, auditable contract before CORE predictions enter this ledger.
    live_core={
        "status":"HOLD_EXACT_LIVE_FEATURE_PARITY_NOT_YET_PROVEN",
        "prediction_rows_written":0,
        "reason":"Historical CORE features are audited prior-only, but V1.9.5 will not synthesize live CORE predictions until the upcoming-game feature builder is proven field-for-field against historical rows.",
        "next_contract":"LIVE_PREGAME_FEATURE_PARITY",
    }

    finished=pd.Timestamp.now(tz="UTC")
    summary={
        "status":"NFL_V1_9_5_PROSPECTIVE_MARKET_SHADOW_ACTIVE",
        "source_tag":SOURCE_TAG,
        "research_clock_id":ledger.RESEARCH_CLOCK_ID,
        "clock_activated_at":clock_start.isoformat(),
        "clock_created_now":bool(clock.get("created_now",False)),
        "source_rows":int(len(source)),"eligible_quote_rows":int(len(quotes)),
        "quotes_inserted":int(qwrite.get("inserted",0)),"states_inserted":int(swrite.get("inserted",0)),
        "state_results_inserted":int(rwrite.get("inserted",0)),
        "upcoming_games":int(quotes.physical_game_id.nunique()) if not quotes.empty else 0,
        "canonical_t60_states":int(((states.snapshot_type=="T_MINUS_60") & states.canonical_evaluation).sum()) if not states.empty else 0,
        "quote_write":qwrite,"state_write":swrite,"result_write":rwrite,
        "market_source_contract":source_map,"quote_meta":qmeta,"state_meta":smeta,
        "live_core_capture":live_core,
        "system_family_tracker":system_family,
        "new_research_paths":{
            "SHARP_VS_SOFT_LEAD_LAG":"COLLECTING",
            "CROSS_BOOK_DISAGREEMENT":"COLLECTING",
            "LINE_VELOCITY_AND_PERSISTENCE":"COLLECTING",
            "KEY_CROSSING_PERSISTENCE":"COLLECTING",
            "QUOTE_STALENESS_AND_DISPERSION":"COLLECTING",
            "LINE_PRICE_DIVERGENCE":"COLLECTING",
            "FIXED_ASOF_T_MINUS_60":"PRIMARY_PROSPECTIVE_EVALUATION_SNAPSHOT",
        },
        "recommended_cadence_minutes":RECOMMENDED_CADENCE_MINUTES,
        "preclock_quotes_admitted":False,"production_authority":0,"legacy_nfl":"UNCHANGED","ncaaf":"UNCHANGED",
        "started_at":started,"finished_at":finished,
    }
    ledger.record_run(summary,client=c)
    log_func("[NFL-V1.9.5-MARKET-SHADOW] "+json.dumps({
        k:v for k,v in summary.items() if k not in {"started_at","finished_at"}
    },sort_keys=True,default=str))
    log_func("[NFL-V1.9.5-CONTRACT] "+json.dumps({
        "status":summary["status"],"research_clock_id":ledger.RESEARCH_CLOCK_ID,
        "clock_activated_at":clock_start.isoformat(),"preclock_quotes_admitted":False,
        "primary_snapshot":"T_MINUS_60","event_stream":"APPEND_ONLY",
        "frozen_v1_9_4_registry_sha256":ledger.FROZEN_V194_REGISTRY_SHA256,
        "live_core_capture_status":live_core["status"],
        "system_family_tracker_status":system_family.get("status"),
        "system_family_id":system_shadow.SYSTEM_FAMILY_ID,
        "system_family_clock_id":system_shadow.SYSTEM_CLOCK_ID,
        "automatic_promotion":False,
        "production_authority":0,"legacy_nfl":"UNCHANGED","ncaaf":"UNCHANGED",
    },sort_keys=True,default=str))
    return summary
