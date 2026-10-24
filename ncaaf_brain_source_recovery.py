"""Recover *descriptive* 2013-25 source views from already-run Heavy Research.

This does not attest original rule/signal/market vintages. It is only a data
handoff and source inventory for the separate certified combined-brain study.
Never creates source manifest, signals, rule no-fire rows or certified coverage.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
import re

import pandas as pd

REPLAY_ROOT = "research/ncaaf/v55/ats_primary/replay_runs"


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _years(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "season" not in df:
        return {}
    return {str(int(year)): int(count) for year, count in
            df.groupby("season").size().items()}


def historical_games_from_legacy_cache(cache):
    """Use physical games and outcomes, not Miner-derived system approvals."""
    raw = cache.get("games") if isinstance(cache, dict) else None
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return None, "LEGACY_GAME_FRAME_NOT_FOUND"
    required = {"Season", "Source_Game_ID", "Game_Date", "Actual_Margin", "Is_Home"}
    absent = sorted(required - set(raw.columns))
    if absent:
        return None, "LEGACY_GAME_FIELDS_MISSING:" + ",".join(absent)
    d = raw.copy()
    d["season"] = pd.to_numeric(d.Season, errors="coerce")
    d["kickoff_utc"] = pd.to_datetime(d.Game_Date, errors="coerce", utc=True)
    d["actual_home_margin"] = pd.to_numeric(d.Actual_Margin, errors="coerce")
    d["game_id"] = d.Source_Game_ID.astype("string").str.strip()
    home = d.Is_Home.astype(str).str.strip().str.lower().isin(("1", "1.0", "true"))
    d = d.loc[home & d.season.between(2013, 2025) &
              d.kickoff_utc.notna() & d.actual_home_margin.notna() &
              d.game_id.notna() & d.game_id.ne("")].copy()
    if d.empty:
        return None, "LEGACY_GAME_HOME_ROWS_EMPTY"
    if d.duplicated(["season", "game_id"]).any():
        return None, "LEGACY_GAME_DUPLICATE_PHYSICAL_GAME_ID"
    d["season"] = d.season.astype(int)
    out = d[["season", "game_id", "kickoff_utc", "actual_home_margin"]].copy()
    # Historical quote timing is unknown. Lines are explicitly descriptive.
    if "Consensus_Open_Spread" in d:
        out["home_spread"] = pd.to_numeric(d.Consensus_Open_Spread, errors="coerce")
    else:
        out["home_spread"] = float("nan")
    out["quote_vintage_certified"] = False
    out["source_provenance"] = "LEGACY_HISTORICAL_SCORE_CACHE__INDEPENDENT_CFBD_UNIVERSE_AUDIT_PENDING"
    return out.reset_index(drop=True), None


def retrospective_core_from_immutable_replay(storage_client, bucket_name, primary):
    """Read V5.5's verified *bytes*, not certify its retrospective model selection."""
    uri = primary.get("model_replay_artifacts") if isinstance(primary, dict) else None
    if not isinstance(uri, str):
        return None, "V55_REPLAY_URI_NOT_IN_PRIMARY_RESULT"
    prefix = "gs://" + bucket_name + "/"
    if not uri.startswith(prefix):
        return None, "V55_REPLAY_BUCKET_MISMATCH"
    key = uri[len(prefix):].rstrip("/")
    if not re.fullmatch(re.escape(REPLAY_ROOT) + r"/[0-9]{8}T[0-9]{12}Z", key):
        return None, "V55_REPLAY_PREFIX_NOT_IMMUTABLE"
    bucket = storage_client.bucket(bucket_name)
    audit_blob = bucket.blob(key + "/audit.json")
    prediction_blob = bucket.blob(key + "/game_predictions.csv")
    if not audit_blob.exists() or not prediction_blob.exists():
        return None, "V55_REPLAY_ARTIFACTS_UNAVAILABLE"
    audit = json.loads(audit_blob.download_as_bytes().decode("utf-8"))
    payload = prediction_blob.download_as_bytes()
    if audit.get("status") != "COMPLETE_V55_EXACT_RECIPE_RETROSPECTIVE_MARGINS":
        return None, "V55_REPLAY_STATUS_MISMATCH"
    if audit.get("predictions_sha256") != _sha(payload):
        return None, "V55_REPLAY_SHA256_MISMATCH"
    if audit.get("retrospective_recipe_selected_after_historical_predictions") is not True:
        return None, "V55_REPLAY_RECIPE_DISCLOSURE_MISSING"
    frame = pd.read_csv(BytesIO(payload), dtype={"game_id": "string"})
    needed = {"season", "game_id", "model_home_margin", "prediction_at_utc", "model_id"}
    absent = needed - set(frame.columns)
    if absent:
        return None, "V55_REPLAY_COLUMNS_MISSING:" + ",".join(sorted(absent))
    f = frame[list(needed)].copy()
    f["season"] = pd.to_numeric(f.season, errors="coerce")
    f = f.loc[f.season.between(2013, 2025)].copy()
    if f.empty or f.duplicated(["season", "game_id"]).any():
        return None, "V55_REPLAY_INVALID_YEAR_OR_DUPLICATE_GAME"
    f["season"] = f.season.astype(int)
    f["model_recipe_sha256"] = ""
    f["recipe_selected_at_utc"] = pd.NaT
    f["training_end_utc"] = pd.NaT
    f["pregame_features_verified"] = False
    f["true_oos"] = False
    f["replay_source"] = "V55_EXACT_RETROSPECTIVE_RECIPE__NOT_AN_INDEPENDENT_HOLDOUT"
    return f.reset_index(drop=True), None


def recover_existing_sources(*, storage_client, bucket_name, legacy_cache, v55_primary_result):
    """Produces auditable, uncertified partial datasets. Never invent signals."""
    frames, receipts, errors = {}, [], {}
    for name, func in (
        ("games", lambda: historical_games_from_legacy_cache(legacy_cache)),
        ("core", lambda: retrospective_core_from_immutable_replay(storage_client, bucket_name, v55_primary_result)),
    ):
        try:
            frame, reason = func()
        except Exception as exc:
            frame, reason = None, "SOURCE_READ_FAILED:" + type(exc).__name__ + ":" + str(exc)[:180]
        if frame is None:
            errors[name] = reason
            continue
        frames[name] = frame
        receipts.append({"input": name, "source_type": "EXISTING_RUNTIME_RECONSTRUCTION",
                         "source_vintage_certified": False, "eligible_for_betting": False,
                         "rows": len(frame), "by_season": _years(frame),
                         "sha256": _sha(frame.to_csv(index=False).encode())})
    return frames, receipts, {"status": "PARTIAL_RECOVERY__UNVERIFIED_SOURCE_VINTAGES",
                              "recovered_inputs": sorted(frames),
                              "recovery_failures": errors,
                              "not_reconstructed": ["signals", "coverage", "eligibility", "rule_evaluations", "universe_manifest"],
                              "rule": "Missing signals and nonfiring games must come from actual timestamped archived evaluations; never backfill by assuming no trigger."}
