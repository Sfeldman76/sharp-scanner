"""NFL V1.9.2 append-only prospective result settlement.

Scores are appended to a result table; prediction rows are never updated.
CLV fields remain null unless a separate, validated canonical closing-quote
contract supplies them.
"""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import pandas as pd

from nfl_prospective_ledger_v1 import (
    PROJECT_ID, RESULT_TABLE, ensure_tables, _flt,
)

SOURCE_TAG = "nfl-prospective-settlement-v1.9.2-append-only-20261001"
PRODUCTION_AUTHORITY = 0


def _norm(s) -> str:
    return str(s or "").strip().lower()


def _grade_prediction(r, actual_margin: float, actual_total: float):
    market = _norm(r.get("market"))
    side = _norm(r.get("side"))
    home = _norm(r.get("home_team")); away = _norm(r.get("away_team"))
    line = _flt(r.get("line"))
    result = "UNAVAILABLE"
    if market == "spreads" and line is not None:
        if side in {home, "home"}:
            z = actual_margin + line
        elif side in {away, "away"}:
            z = -actual_margin + line
        else:
            z = None
        if z is not None:
            result = "WIN" if z > 0 else "LOSS" if z < 0 else "PUSH"
    elif market == "totals" and line is not None:
        if side in {"over", "o"}:
            z = actual_total - line
        elif side in {"under", "u"}:
            z = line - actual_total
        else:
            z = None
        if z is not None:
            result = "WIN" if z > 0 else "LOSS" if z < 0 else "PUSH"
    elif market == "h2h":
        if actual_margin == 0:
            result = "PUSH"
        elif side in {home, "home"}:
            result = "WIN" if actual_margin > 0 else "LOSS"
        elif side in {away, "away"}:
            result = "WIN" if actual_margin < 0 else "LOSS"
    return result


def prepare_result_rows(predictions: pd.DataFrame, scores: pd.DataFrame, *, settled_at=None) -> pd.DataFrame:
    """Join immutable prediction events to authoritative final scores.

    ``scores`` requires physical_game_id, home_score and away_score.  Error
    metrics are game-level and use HOME-minus-AWAY margin convention.
    """
    if predictions is None or predictions.empty or scores is None or scores.empty:
        return pd.DataFrame()
    req = {"physical_game_id", "home_score", "away_score"}
    miss = req - set(scores.columns)
    if miss:
        raise ValueError("NFL_SETTLEMENT_SCORES_MISSING " + str(sorted(miss)))
    p = predictions.copy(); s = scores.copy()
    p["physical_game_id"] = p["physical_game_id"].astype(str).str.lower().str.strip()
    s["physical_game_id"] = s["physical_game_id"].astype(str).str.lower().str.strip()
    s = s.drop_duplicates("physical_game_id", keep="last")
    d = p.merge(s[["physical_game_id", "home_score", "away_score"]], on="physical_game_id", how="inner", validate="many_to_one")
    if d.empty:
        return pd.DataFrame()
    when = pd.Timestamp.now(tz="UTC") if settled_at is None else pd.to_datetime(settled_at, utc=True)
    out = []
    for _, r in d.iterrows():
        hs = _flt(r.get("home_score")); aws = _flt(r.get("away_score"))
        if hs is None or aws is None:
            continue
        margin = hs - aws; total = hs + aws; market = _norm(r.get("market"))
        def ae(pred, actual):
            x = _flt(pred)
            return abs(actual - x) if x is not None else None
        market_err = None; direct_err = None; score_err = None; family_err = None; stats_err = None
        brier = None; log_loss = None
        if market == "spreads":
            market_err = ae(r.get("market_margin_reference"), margin)
            direct_err = ae(r.get("core_direct_margin_pred"), margin)
            score_err = ae(r.get("core_score_margin_pred"), margin)
            family_err = ae(r.get("core_margin_pred"), margin)
            stats_err = ae(r.get("stats_corrected_margin_pred"), margin)
        elif market == "totals":
            market_err = ae(r.get("market_total_reference"), total)
            direct_err = ae(r.get("core_direct_total_pred"), total)
            score_err = ae(r.get("core_score_total_pred"), total)
            family_err = ae(r.get("core_total_pred"), total)
            stats_err = ae(r.get("stats_corrected_total_pred"), total)
        elif market == "h2h":
            prob = _flt(r.get("h2h_probability")); side = _norm(r.get("side")); home = _norm(r.get("home_team")); away = _norm(r.get("away_team"))
            if margin != 0 and prob is not None and 0 < prob < 1 and side in {home, away, "home", "away"}:
                y = 1.0 if ((side in {home, "home"} and margin > 0) or (side in {away, "away"} and margin < 0)) else 0.0
                brier = float((y - prob) ** 2)
                pp = min(max(prob, 1e-12), 1 - 1e-12)
                log_loss = float(-(y * math.log(pp) + (1-y) * math.log(1-pp)))
        pid = str(r.get("prediction_event_id"))
        rid = hashlib.sha256(f"{pid}|FINAL_SCORE|{hs}|{aws}".encode()).hexdigest()
        out.append({
            "result_event_id": rid,
            "prediction_event_id": pid,
            "physical_game_id": str(r.get("physical_game_id")),
            "market": market,
            "settled_at": when,
            "home_score": hs,
            "away_score": aws,
            "actual_margin": margin,
            "actual_total": total,
            "result": _grade_prediction(r, margin, total),
            "market_error": market_err,
            "core_direct_error": direct_err,
            "core_score_error": score_err,
            "core_family_error": family_err,
            "stats_corrected_error": stats_err,
            "brier": brier,
            "log_loss": log_loss,
            "clv_points": None,
            "closing_line": None,
            "closing_odds": None,
            "closing_quote_timestamp": None,
            "production_authority": 0,
        })
    return pd.DataFrame(out)


def append_result_rows(rows: pd.DataFrame, *, client=None) -> dict:
    if rows is None or rows.empty:
        return {"status": "NO_ROWS", "inserted": 0}
    from google.cloud import bigquery as b
    c = client or b.Client(project=PROJECT_ID)
    ensure_tables(c)
    ids = rows.result_event_id.astype(str).tolist()
    cfg = b.QueryJobConfig(query_parameters=[b.ArrayQueryParameter("ids", "STRING", ids)])
    got = c.query(
        f"SELECT result_event_id FROM `{RESULT_TABLE}` WHERE result_event_id IN UNNEST(@ids)", job_config=cfg
    ).to_dataframe(create_bqstorage_client=False)
    existing = set(got.result_event_id.astype(str)) if not got.empty else set()
    new = rows.loc[~rows.result_event_id.astype(str).isin(existing)].copy()
    if new.empty:
        return {"status": "NO_NEW_ROWS", "inserted": 0, "existing": len(existing), "production_authority": 0}
    recs = []
    for rec in new.where(pd.notna(new), None).to_dict("records"):
        v = rec.get("settled_at")
        if isinstance(v, pd.Timestamp): rec["settled_at"] = v.isoformat()
        recs.append(rec)
    errors = c.insert_rows_json(RESULT_TABLE, recs)
    if errors:
        raise RuntimeError("NFL_LEDGER_RESULT_INSERT_FAILED " + str(errors[:3]))
    return {"status": "INSERTED", "inserted": int(len(new)), "existing": int(len(existing)), "production_authority": 0}
