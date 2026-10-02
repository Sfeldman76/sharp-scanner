"""NFL Production V2.0 — live scoring, paired ledger, unified Betting Engine V1.

Purpose
-------
This module starts the *prospective* evidence clock for NFL Production V1.
It does not train fair-value models or promote a challenger. Betting actions are produced only by the separately frozen Betting Engine V1 artifact.

For each upcoming physical game it:
1. Rebuilds the already-proven live Production V1 feature frame.
2. Loads the frozen champion and latest weekly challenger from immutable GCS artifacts.
3. Scores both models on the exact same home-oriented feature snapshot.
4. Appends one immutable paired prediction row per physical game/champion contract.
5. Appends settlement rows later when final scores appear in audited NFL history.

The prediction primary key intentionally excludes challenger SHA.  Once a game's
prospective pair is recorded, a later weekly challenger refresh cannot rewrite or
stack a second challenger onto that same game.  This preserves the first truly
prospective challenger available when the game entered the Production V1 ledger.

Promotion remains closed.  A later Promotion Review may use only *settled* rows
from this ledger that were captured after the frozen champion became active.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

import nfl_live_feature_parity_v1 as parity
import nfl_production_v1 as prod
import nfl_betting_engine_v1 as betting

SOURCE_TAG = "nfl-production-v2.0-live-betting-engine-20261002"
EXPECTED_PARITY_TAG = "nfl-production-v1-live-feature-parity-v1.0.5-frozen-local-feature-contract-20261002"
EXPECTED_PROD_TAG = "nfl-production-v1.1.1-publish-receipt-normalization-20261002"

PROJECT = "sharplogger"
DATASET = "sharp_data"
PRED_TABLE = f"{PROJECT}.{DATASET}.nfl_production_v1_paired_predictions"
SETTLE_TABLE = f"{PROJECT}.{DATASET}.nfl_production_v1_paired_settlements"
LEDGER_STATUS_OBJECT = "production/nfl/v1/ledger/current_status.json"

MIN_PROMOTION_SETTLED_GAMES = 60
PROMOTION_REVIEW_CADENCE_DAYS = 28


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_json(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _utc_ts(x):
    t = pd.to_datetime(x, utc=True, errors="coerce")
    return None if pd.isna(t) else t


def _gcs_parts(uri: str):
    u = str(uri or "").strip()
    if not u.startswith("gs://") or "/" not in u[5:]:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] INVALID_GCS_URI " + u)
    b, name = u[5:].split("/", 1)
    return b, name


def _download_gcs(storage_client, uri: str) -> bytes:
    bucket, name = _gcs_parts(uri)
    blob = storage_client.bucket(bucket).blob(name)
    if not blob.exists():
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] GCS_OBJECT_MISSING " + uri)
    return blob.download_as_bytes()


def _load_pointer(storage_client, bucket_name: str, name: str) -> dict:
    blob = storage_client.bucket(bucket_name).blob(name)
    if not blob.exists():
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] POINTER_MISSING " + f"gs://{bucket_name}/{name}")
    try:
        d = json.loads(blob.download_as_text())
    except Exception as exc:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] POINTER_INVALID_JSON " + name) from exc
    if not isinstance(d, dict):
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] POINTER_NOT_OBJECT " + name)
    return d


def _load_model_bundle(storage_client, pointer: dict, *, expected_role: str) -> dict:
    if pointer.get("source_tag") != EXPECTED_PROD_TAG:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] POINTER_SOURCE_TAG_MISMATCH " + expected_role)
    uri = pointer.get("artifact_uri")
    data = _download_gcs(storage_client, uri)
    actual_sha = _sha_bytes(data)
    if actual_sha != pointer.get("artifact_sha256"):
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] ARTIFACT_SHA_MISMATCH " + expected_role)
    try:
        bundle = joblib.load(io.BytesIO(data))
    except Exception as exc:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] ARTIFACT_DESERIALIZE_FAILED " + expected_role) from exc
    meta = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
    if meta.get("source_tag") != EXPECTED_PROD_TAG:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] BUNDLE_SOURCE_TAG_MISMATCH " + expected_role)
    if meta.get("role") != expected_role:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] BUNDLE_ROLE_MISMATCH " + expected_role)
    if meta.get("contract_sha256") != prod.production_contract()["contract_sha256"]:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] BUNDLE_CONTRACT_MISMATCH " + expected_role)
    return bundle


def _prediction_schema():
    S = bigquery.SchemaField
    return [
        S("prediction_pair_id", "STRING", mode="REQUIRED"),
        S("source_tag", "STRING", mode="REQUIRED"),
        S("production_contract_sha256", "STRING", mode="REQUIRED"),
        S("champion_registry_sha256", "STRING", mode="REQUIRED"),
        S("challenger_registry_sha256", "STRING", mode="REQUIRED"),
        S("champion_artifact_sha256", "STRING"),
        S("challenger_artifact_sha256", "STRING"),
        S("captured_at", "TIMESTAMP", mode="REQUIRED"),
        S("champion_frozen_at", "TIMESTAMP", mode="REQUIRED"),
        S("challenger_data_cutoff", "TIMESTAMP"),
        S("game_start", "TIMESTAMP", mode="REQUIRED"),
        S("season", "INTEGER"),
        S("week_number", "INTEGER"),
        S("game_identity", "STRING", mode="REQUIRED"),
        S("home_team", "STRING", mode="REQUIRED"),
        S("away_team", "STRING", mode="REQUIRED"),
        S("feature_snapshot_sha256", "STRING", mode="REQUIRED"),
        S("features_json", "STRING", mode="REQUIRED"),
        S("champion_fair_margin", "FLOAT"),
        S("challenger_fair_margin", "FLOAT"),
        S("champion_home_win_probability", "FLOAT"),
        S("challenger_home_win_probability", "FLOAT"),
        S("champion_fair_total", "FLOAT"),
        S("challenger_fair_total", "FLOAT"),
        S("promotion_evidence_eligible", "BOOLEAN", mode="REQUIRED"),
        S("model_prediction_authority", "BOOLEAN", mode="REQUIRED"),
        S("betting_decision_authority", "BOOLEAN", mode="REQUIRED"),
        S("automatic_promotion", "BOOLEAN", mode="REQUIRED"),
    ]


def _settlement_schema():
    S = bigquery.SchemaField
    return [
        S("settlement_id", "STRING", mode="REQUIRED"),
        S("prediction_pair_id", "STRING", mode="REQUIRED"),
        S("source_tag", "STRING", mode="REQUIRED"),
        S("champion_registry_sha256", "STRING", mode="REQUIRED"),
        S("challenger_registry_sha256", "STRING", mode="REQUIRED"),
        S("prediction_captured_at", "TIMESTAMP", mode="REQUIRED"),
        S("settled_at", "TIMESTAMP", mode="REQUIRED"),
        S("game_start", "TIMESTAMP", mode="REQUIRED"),
        S("season", "INTEGER"),
        S("week_number", "INTEGER"),
        S("home_team", "STRING", mode="REQUIRED"),
        S("away_team", "STRING", mode="REQUIRED"),
        S("result_source_name", "STRING"),
        S("result_source_game_id", "STRING"),
        S("actual_home_score", "FLOAT"),
        S("actual_away_score", "FLOAT"),
        S("actual_margin", "FLOAT"),
        S("actual_total", "FLOAT"),
        S("home_win_label", "FLOAT"),
        S("champion_spread_abs_error", "FLOAT"),
        S("challenger_spread_abs_error", "FLOAT"),
        S("champion_total_abs_error", "FLOAT"),
        S("challenger_total_abs_error", "FLOAT"),
        S("champion_h2h_log_loss", "FLOAT"),
        S("challenger_h2h_log_loss", "FLOAT"),
        S("champion_h2h_brier", "FLOAT"),
        S("challenger_h2h_brier", "FLOAT"),
        S("promotion_evidence_eligible", "BOOLEAN", mode="REQUIRED"),
        S("betting_decision_authority", "BOOLEAN", mode="REQUIRED"),
    ]


def _ensure_table(client, table_id: str, schema, partition_field: str, clustering_fields):
    """Validate a pre-created ledger table; never request schema-creation IAM.

    The production Cloud Run service account intentionally has runtime data access
    but may not have bigquery.tables.create on sharp_data.  Ledger DDL is therefore
    a one-time operator action.  Runtime fails closed if either table is absent.
    """
    try:
        t = client.get_table(table_id)
    except NotFound as exc:
        raise RuntimeError(
            "[NFL-PROD-V1-LIVE-HOLD] LEDGER_TABLE_MISSING_SETUP_REQUIRED "
            + table_id
            + " RUN_SQL=nfl_production_v1_ledger_setup.sql"
        ) from exc
    existing = {f.name: (f.field_type, f.mode) for f in t.schema}
    expected = {f.name: (f.field_type, f.mode) for f in schema}
    missing = sorted(set(expected) - set(existing))
    if missing:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] LEDGER_SCHEMA_MISSING " + table_id + " " + str(missing))
    type_mismatch = {
        k: {"expected": expected[k][0], "actual": existing[k][0]}
        for k in expected
        if k in existing and str(existing[k][0]).upper() != str(expected[k][0]).upper()
    }
    if type_mismatch:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] LEDGER_SCHEMA_TYPE_MISMATCH " + table_id + " " + json.dumps(type_mismatch, sort_keys=True))
    # Partition/clustering are part of the operator-created contract.  Validate
    # them when the API exposes metadata so a subtly wrong table cannot collect
    # prospective evidence.
    tp = getattr(t, "time_partitioning", None)
    actual_partition = getattr(tp, "field", None) if tp is not None else None
    if actual_partition != partition_field:
        raise RuntimeError(
            "[NFL-PROD-V1-LIVE-HOLD] LEDGER_PARTITION_MISMATCH "
            + table_id + " expected=" + str(partition_field) + " actual=" + str(actual_partition)
        )
    actual_cluster = list(getattr(t, "clustering_fields", None) or [])
    if actual_cluster != list(clustering_fields):
        raise RuntimeError(
            "[NFL-PROD-V1-LIVE-HOLD] LEDGER_CLUSTERING_MISMATCH "
            + table_id + " expected=" + str(list(clustering_fields)) + " actual=" + str(actual_cluster)
        )
    return {"table": table_id, "created": False, "rows": int(getattr(t, "num_rows", 0) or 0)}


def _ensure_ledger(client):
    p = _ensure_table(client, PRED_TABLE, _prediction_schema(), "game_start", ["champion_registry_sha256", "challenger_registry_sha256", "season", "week_number"])
    s = _ensure_table(client, SETTLE_TABLE, _settlement_schema(), "settled_at", ["champion_registry_sha256", "challenger_registry_sha256", "season", "week_number"])
    return {"prediction": p, "settlement": s}


def _jsonable_record(rec: dict, time_cols=()):
    out = {}
    for k, v in rec.items():
        if k in time_cols:
            t = _utc_ts(v)
            out[k] = None if t is None else t.isoformat()
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = None if not np.isfinite(float(v)) else float(v)
        elif isinstance(v, (pd.Timestamp, datetime)):
            t = _utc_ts(v)
            out[k] = None if t is None else t.isoformat()
        elif pd.isna(v) if not isinstance(v, (list, dict, tuple, set)) else False:
            out[k] = None
        else:
            out[k] = v
    return out


def _append_idempotent(client, table_id: str, id_col: str, rows: list[dict], time_cols=()):
    if not rows:
        return {"input_rows": 0, "existing_rows": 0, "inserted_rows": 0}
    ids = [str(x[id_col]) for x in rows]
    q = f"SELECT `{id_col}` FROM `{table_id}` WHERE `{id_col}` IN UNNEST(@ids)"
    job = client.query(q, job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", ids)]))
    got = job.to_dataframe(create_bqstorage_client=False)
    existing = set(got[id_col].astype(str).tolist()) if not got.empty else set()
    new = [r for r in rows if str(r[id_col]) not in existing]
    if new:
        payload = [_jsonable_record(r, time_cols=time_cols) for r in new]
        # Use BigQuery streaming inserts rather than a load job.  Load jobs can
        # require bigquery.tables.create even when the destination table already
        # exists.  This runtime is intentionally restricted to writing existing
        # operator-created ledger tables, so insertAll/insert_rows_json is the
        # correct primitive: table must already exist and write authority is
        # bigquery.tables.updateData.  row_ids add a second best-effort duplicate
        # guard on top of the explicit primary-key pre-query above.
        chunk_size = 500
        for start in range(0, len(payload), chunk_size):
            chunk = payload[start:start + chunk_size]
            chunk_ids = [str(new[start + i][id_col]) for i in range(len(chunk))]
            errors = client.insert_rows_json(
                table_id,
                chunk,
                row_ids=chunk_ids,
                skip_invalid_rows=False,
                ignore_unknown_values=False,
            )
            if errors:
                compact = errors[:10] if isinstance(errors, list) else errors
                raise RuntimeError(
                    "[NFL-PROD-V1-LIVE-HOLD] LEDGER_STREAM_INSERT_FAILED "
                    + table_id + " " + json.dumps(compact, sort_keys=True, default=str)
                )
    return {
        "input_rows": len(rows),
        "existing_rows": len(rows) - len(new),
        "inserted_rows": len(new),
        "write_method": "BIGQUERY_STREAMING_INSERT_EXISTING_TABLE_ONLY",
    }


def _feature_payload(row: pd.Series) -> dict:
    d = {}
    for c in prod.PRODUCTION_FEATURES:
        v = pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]
        d[c] = None if pd.isna(v) else float(v)
    return d


def _build_live_frame(client, now):
    raw = parity.fetch_raw(client)
    calendar = parity.validate_week_calendar(raw)
    if calendar.get("status") != "PASS":
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] WEEK_CALENDAR_NOT_GREEN")
    upcoming, umeta = parity.fetch_upcoming_games(client, now=now)
    if upcoming.empty:
        return pd.DataFrame(), upcoming, {"market": umeta, "calendar": calendar, "live": {"status": "NO_UPCOMING_GAMES"}}
    live, lmeta = parity.build_upcoming_features(raw, upcoming, calendar)
    if lmeta.get("status") != "READY":
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] UPCOMING_FEATURES_NOT_READY " + str(lmeta.get("status")))
    h = live.loc[pd.to_numeric(live["Is_Home"], errors="coerce").eq(1)].copy()
    if h.empty or h.Source_Game_ID.duplicated().any() or int(h.Source_Game_ID.nunique()) != int(len(upcoming)):
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] LIVE_HOME_ORIENTATION_GRAIN_INVALID")
    upcoming = upcoming.copy()
    upcoming["__identity"] = [
        f"{pd.Timestamp(gs).round('s').isoformat()}|{hh}|{aa}" if pd.notna(gs) else f"|{hh}|{aa}"
        for gs, hh, aa in zip(upcoming.game_start, upcoming.home_key, upcoming.away_key)
    ]
    meta = upcoming.set_index("__identity")[["game_start", "home_team", "away_team", "source_game_key"]].to_dict("index")
    h["game_start"] = h.Source_Game_ID.map(lambda x: meta.get(str(x), {}).get("game_start"))
    h["home_team_display"] = h.Source_Game_ID.map(lambda x: meta.get(str(x), {}).get("home_team"))
    h["away_team_display"] = h.Source_Game_ID.map(lambda x: meta.get(str(x), {}).get("away_team"))
    h["source_game_key"] = h.Source_Game_ID.map(lambda x: meta.get(str(x), {}).get("source_game_key"))
    return h.reset_index(drop=True), upcoming, {"market": umeta, "calendar": calendar, "live": lmeta}


def _build_prediction_rows(home_rows, champion_scores, challenger_scores, *, champion_ptr, challenger_ptr, now):
    contract = prod.production_contract()
    frozen = _utc_ts(champion_ptr.get("frozen_at_utc"))
    ch_cutoff = _utc_ts(challenger_ptr.get("data_cutoff"))
    if frozen is None:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] CHAMPION_FREEZE_TIME_MISSING")
    if now < frozen:
        raise RuntimeError("[NFL-PROD-V1-LIVE-HOLD] CLOCK_BEFORE_CHAMPION_FREEZE")
    out = []
    for i in range(len(home_rows)):
        r = home_rows.iloc[i]
        c = champion_scores.iloc[i]
        q = challenger_scores.iloc[i]
        gs = _utc_ts(r.get("game_start"))
        if gs is None or gs <= now:
            continue
        features = _feature_payload(r)
        feature_sha = _sha_json(features)
        source_key = str(r.get("source_game_key") or "").strip()
        if source_key and source_key.lower() not in ("nan", "none", "null"):
            game_identity = "SOURCE_KEY|" + source_key
        else:
            local_date = gs.tz_convert("America/New_York").date()
            game_identity = "MATCHUP_DATE|" + str(local_date) + "|" + parity._norm_name(r.get("Team_Norm")) + "|" + parity._norm_name(r.get("Opponent_Norm"))
        pair_id = hashlib.sha256((str(champion_ptr.get("registry_sha256")) + "|" + contract["contract_sha256"] + "|" + game_identity).encode()).hexdigest()
        season = pd.to_numeric(pd.Series([r.get("Season")]), errors="coerce").iloc[0]
        week = pd.to_numeric(pd.Series([r.get("Week_Number")]), errors="coerce").iloc[0]
        eligible = bool(now >= frozen and (ch_cutoff is None or ch_cutoff < gs))
        out.append({
            "prediction_pair_id": pair_id,
            "source_tag": SOURCE_TAG,
            "production_contract_sha256": contract["contract_sha256"],
            "champion_registry_sha256": champion_ptr.get("registry_sha256"),
            "challenger_registry_sha256": challenger_ptr.get("registry_sha256"),
            "champion_artifact_sha256": champion_ptr.get("artifact_sha256"),
            "challenger_artifact_sha256": challenger_ptr.get("artifact_sha256"),
            "captured_at": now,
            "champion_frozen_at": frozen,
            "challenger_data_cutoff": ch_cutoff,
            "game_start": gs,
            "season": None if pd.isna(season) else int(season),
            "week_number": None if pd.isna(week) else int(week),
            "game_identity": game_identity,
            "home_team": str(r.get("Team_Norm") or "").strip().lower(),
            "away_team": str(r.get("Opponent_Norm") or "").strip().lower(),
            "feature_snapshot_sha256": feature_sha,
            "features_json": json.dumps(features, sort_keys=True, separators=(",", ":")),
            "champion_fair_margin": float(c.get("fair_margin")),
            "challenger_fair_margin": float(q.get("fair_margin")),
            "champion_home_win_probability": float(c.get("win_probability")),
            "challenger_home_win_probability": float(q.get("win_probability")),
            "champion_fair_total": float(c.get("fair_total")),
            "challenger_fair_total": float(q.get("fair_total")),
            "promotion_evidence_eligible": eligible,
            "model_prediction_authority": True,
            "betting_decision_authority": False,
            "automatic_promotion": False,
        })
    return out


def _pending_predictions(client, champion_sha: str, now) -> pd.DataFrame:
    sql = f"""
    SELECT p.*
    FROM `{PRED_TABLE}` p
    LEFT JOIN `{SETTLE_TABLE}` s USING(prediction_pair_id)
    WHERE p.champion_registry_sha256=@champion_sha
      AND p.game_start < @now
      AND s.prediction_pair_id IS NULL
    ORDER BY p.game_start
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("champion_sha", "STRING", champion_sha),
        bigquery.ScalarQueryParameter("now", "TIMESTAMP", now.to_pydatetime()),
    ])
    return client.query(sql, job_config=cfg).to_dataframe(create_bqstorage_client=False)


def _completed_result_rows(client, min_date) -> pd.DataFrame:
    # Read both sides.  Settlement chooses the row matching the ledger's home-team
    # orientation, so neutral/venue metadata cannot silently flip the target.
    q = f"""
      SELECT Season, Week_Number, Game_Date, Source_Name, Source_Game_ID,
             Team_Norm, Opponent_Norm, Team_Score, Opponent_Score
      FROM `{prod.VIEW}`
      WHERE Team_Score IS NOT NULL AND Opponent_Score IS NOT NULL
        AND Game_Date >= @min_date
      ORDER BY Game_Date, Source_Name, Source_Game_ID, Team_Norm
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("min_date", "DATE", min_date)])
    d = client.query(q, job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if not d.empty:
        d["Game_Date"] = pd.to_datetime(d.Game_Date, errors="coerce").dt.date
        d["team_key"] = d.Team_Norm.map(parity._norm_name)
        d["opp_key"] = d.Opponent_Norm.map(parity._norm_name)
    return d


def _ll(y, p):
    if y not in (0.0, 1.0):
        return None
    p = float(np.clip(float(p), 0.001, 0.999))
    return float(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))


def _settle_pending(client, champion_sha: str, now):
    pending = _pending_predictions(client, champion_sha, now)
    if pending.empty:
        return [], {"pending_predictions": 0, "matched_results": 0, "unmatched_results": 0}
    min_date = pd.to_datetime(pending.game_start, utc=True, errors="coerce").dt.tz_convert("America/New_York").dt.date.min()
    results = _completed_result_rows(client, min_date)
    rows = []
    unmatched = []
    for _, p in pending.iterrows():
        local_date = pd.to_datetime(p.game_start, utc=True).tz_convert("America/New_York").date()
        hk = parity._norm_name(p.home_team)
        ak = parity._norm_name(p.away_team)
        m = results.loc[(results.Game_Date == local_date) & (results.team_key == hk) & (results.opp_key == ak)] if not results.empty else pd.DataFrame()
        invert = False
        if len(m) != 1:
            rev = results.loc[(results.Game_Date == local_date) & (results.team_key == ak) & (results.opp_key == hk)] if not results.empty else pd.DataFrame()
            if len(m) == 0 and len(rev) == 1:
                m = rev; invert = True
        # Postponement-safe settlement: if the originally recorded kickoff date
        # moved, allow exactly one same-matchup result within +/-3 calendar days.
        # This changes settlement matching only; the prediction row stays immutable.
        if len(m) == 0 and not results.empty:
            delta = results.Game_Date.map(lambda d: abs((d - local_date).days) if d is not None else 999)
            alt = results.loc[(delta <= 3) & (results.team_key == hk) & (results.opp_key == ak)]
            if len(alt) == 1:
                m = alt; invert = False
            else:
                rev_alt = results.loc[(delta <= 3) & (results.team_key == ak) & (results.opp_key == hk)]
                if len(rev_alt) == 1:
                    m = rev_alt; invert = True
        if len(m) != 1:
            unmatched.append(str(p.prediction_pair_id)); continue
        r = m.iloc[0]
        ts = float(r.Team_Score); os = float(r.Opponent_Score)
        if invert:
            hs, aw = os, ts
        else:
            hs, aw = ts, os
        margin = hs - aw; total = hs + aw
        y = 1.0 if margin > 0 else 0.0 if margin < 0 else None
        cp = float(p.champion_home_win_probability); qp = float(p.challenger_home_win_probability)
        rows.append({
            "settlement_id": str(p.prediction_pair_id),
            "prediction_pair_id": str(p.prediction_pair_id),
            "source_tag": SOURCE_TAG,
            "champion_registry_sha256": str(p.champion_registry_sha256),
            "challenger_registry_sha256": str(p.challenger_registry_sha256),
            "prediction_captured_at": p.captured_at,
            "settled_at": now,
            "game_start": p.game_start,
            "season": None if pd.isna(p.season) else int(p.season),
            "week_number": None if pd.isna(p.week_number) else int(p.week_number),
            "home_team": str(p.home_team),
            "away_team": str(p.away_team),
            "result_source_name": str(r.Source_Name),
            "result_source_game_id": str(r.Source_Game_ID),
            "actual_home_score": hs,
            "actual_away_score": aw,
            "actual_margin": margin,
            "actual_total": total,
            "home_win_label": y,
            "champion_spread_abs_error": abs(float(p.champion_fair_margin) - margin),
            "challenger_spread_abs_error": abs(float(p.challenger_fair_margin) - margin),
            "champion_total_abs_error": abs(float(p.champion_fair_total) - total),
            "challenger_total_abs_error": abs(float(p.challenger_fair_total) - total),
            "champion_h2h_log_loss": _ll(y, cp),
            "challenger_h2h_log_loss": _ll(y, qp),
            "champion_h2h_brier": None if y is None else float((cp - y) ** 2),
            "challenger_h2h_brier": None if y is None else float((qp - y) ** 2),
            "promotion_evidence_eligible": bool(p.promotion_evidence_eligible),
            "betting_decision_authority": False,
        })
    return rows, {"pending_predictions": int(len(pending)), "matched_results": int(len(rows)), "unmatched_results": int(len(unmatched)), "unmatched_samples": unmatched[:20]}


def _ledger_counts(client, champion_sha: str):
    sql = f"""
    SELECT
      (SELECT COUNT(*) FROM `{PRED_TABLE}` WHERE champion_registry_sha256=@champion_sha) AS prediction_rows,
      (SELECT COUNT(*) FROM `{SETTLE_TABLE}` WHERE champion_registry_sha256=@champion_sha) AS settled_rows,
      (SELECT COUNT(*) FROM `{SETTLE_TABLE}` WHERE champion_registry_sha256=@champion_sha AND promotion_evidence_eligible=TRUE) AS eligible_settled_rows
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("champion_sha", "STRING", champion_sha)])
    d = client.query(sql, job_config=cfg).to_dataframe(create_bqstorage_client=False)
    if d.empty:
        return {"prediction_rows": 0, "settled_rows": 0, "eligible_settled_rows": 0}
    r = d.iloc[0]
    return {k: int(r[k] or 0) for k in ("prediction_rows", "settled_rows", "eligible_settled_rows")}


def _write_status(storage_client, bucket_name: str, payload: dict):
    storage_client.bucket(bucket_name).blob(LEDGER_STATUS_OBJECT).upload_from_string(
        json.dumps(payload, sort_keys=True, indent=2, default=str), content_type="application/json"
    )
    return f"gs://{bucket_name}/{LEDGER_STATUS_OBJECT}"


def run_nfl_production_live_score(*, bq_client, storage_client, bucket_name="sharp-models", log_func=print, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, utc=True)
    if parity.SOURCE_TAG != EXPECTED_PARITY_TAG or prod.SOURCE_TAG != EXPECTED_PROD_TAG:
        raise RuntimeError("[NFL-PROD-V1-LIVE-PREFLIGHT] STALE_OR_MIXED_DEPENDENCY")
    contract = prod.production_contract()
    champion_ptr = _load_pointer(storage_client, bucket_name, prod.BASELINE_POINTER)
    challenger_ptr = _load_pointer(storage_client, bucket_name, prod.CHALLENGER_POINTER)
    log_func("[NFL-PROD-V1-LIVE-PREFLIGHT] " + json.dumps({
        "status": "START", "source_tag": SOURCE_TAG,
        "production_contract_sha256": contract["contract_sha256"],
        "champion_registry_sha256": champion_ptr.get("registry_sha256"),
        "challenger_registry_sha256": challenger_ptr.get("registry_sha256"),
        "prediction_primary_key_policy": "CHAMPION_CONTRACT_PLUS_PHYSICAL_GAME_FIRST_PAIR_WINS",
        "promotion_evidence": "POSTFREEZE_SETTLED_PAIRED_ONLY",
        "automatic_promotion": False, "betting_decision_authority": False,
    }, sort_keys=True, default=str))

    champion_bundle = _load_model_bundle(storage_client, champion_ptr, expected_role="FROZEN_CHAMPION")
    challenger_bundle = _load_model_bundle(storage_client, challenger_ptr, expected_role="WEEKLY_CHALLENGER")
    ledger_meta = _ensure_ledger(bq_client)

    home, upcoming, live_meta = _build_live_frame(bq_client, now)
    prediction_rows = []
    if not home.empty:
        cscore = prod.score_feature_rows(champion_bundle, home).reset_index(drop=True)
        qscore = prod.score_feature_rows(challenger_bundle, home).reset_index(drop=True)
        prediction_rows = _build_prediction_rows(home.reset_index(drop=True), cscore, qscore,
                                                 champion_ptr=champion_ptr, challenger_ptr=challenger_ptr, now=now)
    pred_write = _append_idempotent(
        bq_client, PRED_TABLE, "prediction_pair_id", prediction_rows,
        time_cols=("captured_at", "champion_frozen_at", "challenger_data_cutoff", "game_start"),
    )
    log_func("[NFL-PROD-V1-LIVE-SCORES] " + json.dumps({
        "status": "READY" if prediction_rows or home.empty else "HOLD",
        "upcoming_games": int(len(upcoming)), "scored_games": int(len(prediction_rows)),
        "champion_registry_sha256": champion_ptr.get("registry_sha256"),
        "challenger_registry_sha256": challenger_ptr.get("registry_sha256"),
        "prediction_write": pred_write,
        "sample": [{
            "game_start": str(r["game_start"]), "home": r["home_team"], "away": r["away_team"],
            "champion_fair_margin": round(r["champion_fair_margin"], 4),
            "challenger_fair_margin": round(r["challenger_fair_margin"], 4),
            "champion_home_win_probability": round(r["champion_home_win_probability"], 4),
            "challenger_home_win_probability": round(r["challenger_home_win_probability"], 4),
            "champion_fair_total": round(r["champion_fair_total"], 4),
            "challenger_fair_total": round(r["challenger_fair_total"], 4),
        } for r in prediction_rows[:5]],
        "model_prediction_authority": True, "betting_decision_authority": False,
    }, sort_keys=True, default=str))

    settlement_rows, settle_meta = _settle_pending(bq_client, champion_ptr.get("registry_sha256"), now)
    settle_write = _append_idempotent(
        bq_client, SETTLE_TABLE, "settlement_id", settlement_rows,
        time_cols=("prediction_captured_at", "settled_at", "game_start"),
    )
    log_func("[NFL-PROD-V1-PAIRED-SETTLEMENT] " + json.dumps({**settle_meta, "write": settle_write, "status": "PASS"}, sort_keys=True, default=str))

    counts = _ledger_counts(bq_client, champion_ptr.get("registry_sha256"))
    frozen = _utc_ts(champion_ptr.get("frozen_at_utc"))
    age_days = None if frozen is None else max(0.0, (now - frozen).total_seconds() / 86400.0)
    clock = {
        **counts,
        "min_settled_games": MIN_PROMOTION_SETTLED_GAMES,
        "min_age_days": PROMOTION_REVIEW_CADENCE_DAYS,
        "days_since_champion_freeze": None if age_days is None else round(age_days, 4),
        "count_gate_met": counts["eligible_settled_rows"] >= MIN_PROMOTION_SETTLED_GAMES,
        "cadence_gate_met": age_days is not None and age_days >= PROMOTION_REVIEW_CADENCE_DAYS,
    }
    clock["promotion_review_ready"] = bool(clock["count_gate_met"] and clock["cadence_gate_met"])
    clock["automatic_promotion"] = False
    log_func("[NFL-PROD-V1-PROMOTION-CLOCK] " + json.dumps(clock, sort_keys=True, default=str))

    try:
        betting_state = betting.update_live_state(
            bq_client=bq_client, storage_client=storage_client, bucket_name=bucket_name,
            prediction_rows=prediction_rows, champion_sha=champion_ptr.get("registry_sha256"),
            now=now, log_func=log_func,
        )
    except Exception as exc:
        betting_state={"status":"HOLD_BETTING_ENGINE_ERROR","error":f"{type(exc).__name__}:{exc}","action_counts":{}}
        log_func("[NFL-BET-ENGINE-V1-LIVE] "+json.dumps(betting_state,sort_keys=True,default=str))

    _live_rows = betting_state.get("live_rows", []) if isinstance(betting_state, dict) else []
    _action_counts = betting_state.get("action_counts", {}) if isinstance(betting_state, dict) else {}

    report = {
        "status": "NFL_PRODUCTION_V1_LIVE_PAIRED_LEDGER_ACTIVE",
        "source_tag": SOURCE_TAG,
        "champion_registry_sha256": champion_ptr.get("registry_sha256"),
        "challenger_registry_sha256": challenger_ptr.get("registry_sha256"),
        "production_contract_sha256": contract["contract_sha256"],
        "ledger_tables": {"predictions": PRED_TABLE, "settlements": SETTLE_TABLE},
        "ledger_table_status": ledger_meta,
        "prediction_write": pred_write,
        "settlement_write": settle_write,
        "promotion_clock": clock,
        "model_prediction_authority": True,
        "betting_decision_authority": bool(((betting_state.get("engine_meta") or {}).get("betting_decision_authority"))) if isinstance(betting_state, dict) else False,
        "automatic_promotion": False,
        "betting_engine": {
            "status": betting_state.get("status"),
            "action_counts": _action_counts,
            "live_bet_performance": betting_state.get("live_bet_performance",{}),
            "prospective_model_performance": betting_state.get("prospective_model_performance",{}),
            "current_uri": betting_state.get("current_uri"),
            "automatic_execution": False,
        },
        "next_step": "CONTINUE_WEEKLY_UPDATE_AND_ACCUMULATE_UNIFIED_BETTING_ENGINE_PERFORMANCE",
    }
    report["status_uri"] = _write_status(storage_client, bucket_name, report)
    log_func("[NFL-PROD-V1-LIVE-CONTRACT] " + json.dumps(report, sort_keys=True, default=str))
    return report


def _self_test():
    assert SOURCE_TAG.startswith("nfl-production-v2.0-")
    assert MIN_PROMOTION_SETTLED_GAMES == 60
    assert PROMOTION_REVIEW_CADENCE_DAYS == 28
    assert prod.SOURCE_TAG == EXPECTED_PROD_TAG
    assert parity.SOURCE_TAG == EXPECTED_PARITY_TAG
    # Primary key must not include challenger SHA so a later refresh cannot
    # overwrite/stack a second challenger on the same physical game.
    champion = "c" * 64; contract = "p" * 64; game = "2026-10-04T17:00:00+00:00|a|b"
    a = hashlib.sha256((champion + "|" + contract + "|" + game).encode()).hexdigest()
    b = hashlib.sha256((champion + "|" + contract + "|" + game).encode()).hexdigest()
    assert a == b
    return {"status": "PASS", "source_tag": SOURCE_TAG, "promotion_min_games": 60, "promotion_min_days": 28}


if __name__ == "__main__":
    print(json.dumps(_self_test(), sort_keys=True))
