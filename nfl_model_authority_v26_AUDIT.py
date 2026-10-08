"""NFL Model Authority V2.6.1 — frozen fair-value model with CORE-first Bet Authority.

The frozen NFL production model remains the only prediction authority. Production
Bet Authority V3.1 starts from the frozen Spread CORE probability benchmark and an
actual executable-price EV gate, which creates a CORE CANDIDATE rather than an
automatic wager. A qualified independent Miner/Pathi/Big Al/PT-derived system family must confirm the
CORE side before the candidate becomes a BET; two or more independent supporters
produce STRONG BET. Qualified systems cannot create a candidate or reverse CORE.
H2H and Totals remain model-only until they earn their own frozen betting gates.

Historical freeze protocol
--------------------------
* 2021-2023: discovery only. A small predeclared edge-threshold grid is tested.
* The FIRST threshold that passes discovery robustness is frozen per market.
* 2024-2025: confirmation only; thresholds are never reselected on confirmation.
* 2026+: prospective only; no retuning.

Spread/Totals historical ROI uses the existing -110 replay proxy. H2H uses the
stored historical closing moneyline. Live execution additionally requires an
actual current price and applies a price-safety gate.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import joblib
try:
    from google.api_core.exceptions import PreconditionFailed
    from google.cloud import bigquery
except ModuleNotFoundError:
    class PreconditionFailed(Exception):
        pass
    bigquery = None

import nfl_production_v1 as prod
import nfl_market_backend_v261 as market_backend

SOURCE_TAG = "nfl-model-authority-v2.6.1-utils-market-backend-20261003"
ENGINE_VERSION = "NFL_MODEL_AUTHORITY_V2_6_1"
PREFIX = "production/nfl/v1/model_authority_v261"
CONTRACT_OBJECT = f"{PREFIX}/current_contract.json"
CURRENT_OBJECT = f"{PREFIX}/current_state.json"
EVENT_PREFIX = f"{PREFIX}/events"
SETTLEMENT_PREFIX = f"{PREFIX}/settlements"
PROJECT = "sharplogger"
DATASET = "sharp_data"
PAIRED_SETTLEMENT_TABLE = f"{PROJECT}.{DATASET}.nfl_production_v1_paired_settlements"
LIVE_MINER_VIEW = f"{PROJECT}.{DATASET}.nfl_historical_core_training_vw"
LIVE_MINER_RAW_SIDE_TABLE = f"{PROJECT}.{DATASET}.nfl_historical_game_side_raw"
FAMILY_POINTER_OBJECT = "nfl-research/v2_0/system_lab/latest_family_registry_pointer_v3.json"

DISCOVERY_SEASONS = (2021, 2022, 2023)
CONFIRMATION_SEASONS = (2024, 2025)
THRESHOLDS = {
    "SPREADS": (2.0, 3.0, 4.0, 5.0, 6.0),
    "H2H": (0.02, 0.05, 0.075, 0.10, 0.15),
    "TOTALS": (2.0, 3.0, 4.0, 5.0, 6.0),
}
MIN_DISCOVERY_N = {"SPREADS": 60, "H2H": 75, "TOTALS": 60}
MIN_CONFIRMATION_N = {"SPREADS": 30, "H2H": 40, "TOTALS": 30}
MIN_DISCOVERY_POSITIVE_SEASONS = 2
MIN_CONFIRMATION_POSITIVE_SEASONS = 1
SPREAD_TOTAL_BREAK_EVEN = 110.0 / 210.0
MAX_SPREAD_TOTAL_LAY = -110.0
MIN_H2H_LIVE_EV = 0.0

# Production Betting V3.1: CORE creates candidates; qualified systems confirm wagers.
# Keep this component SOURCE_TAG unchanged so the frozen fair-value contract still
# loads; the betting policy has its own immutable version tag.
BETTING_POLICY_SOURCE_TAG = "nfl-production-betting-v3.3.2-season-record-state-live-parity-20261008"
LEGACY_EDGE_CONTRACT_SHA256 = "6a1e2e770071725081e29855d47f09aaa797831d62b9b2a8a13968661f8dc20c"
HEAVY_REPORT_OBJECT = "research/nfl/heavy/v211/current_report.json"
FROZEN_CORE_SPREAD_THRESHOLD = 0.575
FROZEN_CORE_SPREAD_RUN_ID = "5313f35b"
FROZEN_CORE_SPREAD_SOURCE_TAG = "nfl-heavy-research-v2.9-residual-distributional-edge-20261003"
MIN_CORE_SPREAD_LIVE_EV = 0.02
STRONG_SUPPORT_FAMILY_COUNT = 2
STRONG_CONFLICT_FAMILY_COUNT = 2

def production_betting_policy_contract():
    return {
        "source_tag": BETTING_POLICY_SOURCE_TAG,
        "prediction_authority": "CORE_ONLY",
        "spread": {
            "authority": "FROZEN_CORE_0575_PLUS_EXECUTABLE_PRICE_EV_CREATES_CANDIDATE",
            "core_probability_threshold": FROZEN_CORE_SPREAD_THRESHOLD,
            "min_live_ev": MIN_CORE_SPREAD_LIVE_EV,
            "frozen_source_run_id": FROZEN_CORE_SPREAD_RUN_ID,
            "frozen_source_tag": FROZEN_CORE_SPREAD_SOURCE_TAG,
            "core_candidate_without_system_support": "CANDIDATE_NO_WAGER",
            "one_independent_system_support": "BET",
            "two_plus_independent_system_support": "STRONG_BET",
            "any_mixed_or_single_system_conflict": "CANDIDATE_CAUTION_NO_WAGER",
            "two_plus_net_independent_system_conflict": "PASS_VETO",
            "systems_can_create_candidate": False,
            "systems_can_confirm_core_candidate": True,
            "systems_can_reverse_core": False,
            "automatic_execution": False,
        },
        "totals": {
            "authority": "MODEL_ONLY_NO_FROZEN_CORE_PROBABILITY_GATE",
            "systems_can_create_bet": False,
            "automatic_execution": False,
        },
        "h2h": {
            "authority": "MODEL_ONLY_NO_FROZEN_BETTING_EV_GATE",
            "systems_can_create_bet": False,
            "automatic_execution": False,
        },
        "overlays": {
            "sources": ["MINER", "PATHI", "BIG AL", "PT"],
            "qualification_source": HEAVY_REPORT_OBJECT,
            "normalization": "ONE_NORMALIZED_INDEPENDENT_FAMILY_ONE_VOTE",
            "role": "CONFIRM_OR_CONFLICT_ONLY_AFTER_CORE_CANDIDATE",
        },
        "stat_market_role": "BOUNDED_DIAGNOSTIC_CONFIDENCE_ONLY",
        "legacy_edge_contract_reference": LEGACY_EDGE_CONTRACT_SHA256,
        "automatic_model_promotion": False,
        "automatic_policy_promotion": False,
    }


def _num(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _norm(x):
    return " ".join(str(x or "").strip().lower().replace(".", " ").replace("_", " ").split())


def _sha(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _write_json(sc, bucket, name, payload):
    sc.bucket(bucket).blob(name).upload_from_string(
        json.dumps(payload, sort_keys=True, indent=2, default=str), content_type="application/json"
    )
    return f"gs://{bucket}/{name}"


def _write_immutable(sc, bucket, name, payload):
    b = sc.bucket(bucket).blob(name)
    try:
        b.upload_from_string(
            json.dumps(payload, sort_keys=True, indent=2, default=str),
            content_type="application/json",
            if_generation_match=0,
        )
        return True
    except PreconditionFailed:
        return False


def _read_json(sc, bucket, name):
    b = sc.bucket(bucket).blob(name)
    if not b.exists():
        return None
    try:
        return json.loads(b.download_as_text())
    except Exception:
        return None


def _american_profit_if_win(odds):
    o = _num(odds)
    if not math.isfinite(o) or o == 0:
        return np.nan
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def _american_ev(prob, odds):
    p = _num(prob)
    win = _american_profit_if_win(odds)
    if not (math.isfinite(p) and 0 <= p <= 1 and math.isfinite(win)):
        return np.nan
    return p * win - (1.0 - p)


def _market_vectors(rows: pd.DataFrame, market: str) -> pd.DataFrame:
    d = rows.copy()
    if market == "SPREADS":
        fair = pd.to_numeric(d["frozen_fair_margin"], errors="coerce")
        line = pd.to_numeric(d["close_spread"], errors="coerce")
        actual = pd.to_numeric(d["actual_margin"], errors="coerce")
        edge = fair + line
        settle = actual + line
        valid = edge.notna() & settle.notna() & edge.abs().gt(1e-12)
        win = ((edge > 0) & (settle > 0)) | ((edge < 0) & (settle < 0))
        push = settle.abs().le(1e-9)
        out = pd.DataFrame({"season": d.season, "edge": edge.abs(), "valid": valid, "push": push, "win": win})
        out["profit"] = np.where(push, 0.0, np.where(win, 100.0 / 110.0, -1.0))
        return out
    if market == "TOTALS":
        fair = pd.to_numeric(d["frozen_fair_total"], errors="coerce")
        line = pd.to_numeric(d["close_total"], errors="coerce")
        actual = pd.to_numeric(d["actual_total"], errors="coerce")
        edge_signed = fair - line
        settle = actual - line
        valid = edge_signed.notna() & settle.notna() & edge_signed.abs().gt(1e-12)
        win = ((edge_signed > 0) & (settle > 0)) | ((edge_signed < 0) & (settle < 0))
        push = settle.abs().le(1e-9)
        out = pd.DataFrame({"season": d.season, "edge": edge_signed.abs(), "valid": valid, "push": push, "win": win})
        out["profit"] = np.where(push, 0.0, np.where(win, 100.0 / 110.0, -1.0))
        return out
    if market == "H2H":
        fair = pd.to_numeric(d["frozen_home_win_probability"], errors="coerce")
        ref = pd.to_numeric(d["close_novig_home_probability"], errors="coerce")
        y = pd.to_numeric(d["home_win_label"], errors="coerce")
        hm = pd.to_numeric(d["home_close_ml"], errors="coerce")
        am = pd.to_numeric(d["away_close_ml"], errors="coerce")
        signed = fair - ref
        home_sel = signed > 0
        p_sel = np.where(home_sel, fair, 1.0 - fair)
        odds = np.where(home_sel, hm, am)
        win = np.where(home_sel, y.eq(1.0), y.eq(0.0))
        profits = []
        evs = []
        for pp, oo, ww in zip(p_sel, odds, win):
            wprof = _american_profit_if_win(oo)
            profits.append((wprof if bool(ww) else -1.0) if math.isfinite(wprof) else np.nan)
            evs.append(_american_ev(pp, oo))
        out = pd.DataFrame({
            "season": d.season,
            "edge": signed.abs(),
            "valid": signed.notna() & y.notna() & pd.Series(odds, index=d.index).notna() & signed.abs().gt(1e-12),
            "push": False,
            "win": win,
            "profit": profits,
            "model_ev": evs,
        })
        return out
    raise ValueError(market)


def _metrics(v: pd.DataFrame, threshold: float, seasons) -> dict:
    z = v.loc[v.season.isin(seasons) & v.valid & v.edge.ge(float(threshold))].copy()
    if z.empty:
        return {"n": 0, "wins": 0, "losses": 0, "pushes": 0, "hit_rate": None, "roi_per_unit": None, "positive_seasons": 0, "by_season": {}}
    dec = z.loc[~z.push].copy()
    wins = int(dec.win.sum())
    losses = int(len(dec) - wins)
    pushes = int(z.push.sum())
    roi = float(pd.to_numeric(z.profit, errors="coerce").mean()) if pd.to_numeric(z.profit, errors="coerce").notna().any() else np.nan
    by = {}
    positive = 0
    for sy in seasons:
        q = z.loc[z.season.eq(sy)]
        qdec = q.loc[~q.push]
        qw = int(qdec.win.sum()) if not qdec.empty else 0
        ql = int(len(qdec) - qw) if not qdec.empty else 0
        qroi = float(pd.to_numeric(q.profit, errors="coerce").mean()) if not q.empty and pd.to_numeric(q.profit, errors="coerce").notna().any() else np.nan
        if math.isfinite(qroi) and qroi > 0:
            positive += 1
        by[str(int(sy))] = {"n": int(len(qdec)), "wins": qw, "losses": ql, "roi_per_unit": round(qroi, 6) if math.isfinite(qroi) else None}
    n = wins + losses
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": round(wins / n, 6) if n else None,
        "roi_per_unit": round(roi, 6) if math.isfinite(roi) else None,
        "positive_seasons": int(positive),
        "by_season": by,
    }


def _discovery_pass(market: str, met: dict) -> bool:
    roi = _num(met.get("roi_per_unit"))
    hit = _num(met.get("hit_rate"))
    if int(met.get("n") or 0) < MIN_DISCOVERY_N[market]:
        return False
    if not math.isfinite(roi) or roi <= 0:
        return False
    if int(met.get("positive_seasons") or 0) < MIN_DISCOVERY_POSITIVE_SEASONS:
        return False
    if market in {"SPREADS", "TOTALS"} and (not math.isfinite(hit) or hit <= SPREAD_TOTAL_BREAK_EVEN):
        return False
    return True


def _confirmation_pass(market: str, met: dict) -> bool:
    roi = _num(met.get("roi_per_unit"))
    hit = _num(met.get("hit_rate"))
    if int(met.get("n") or 0) < MIN_CONFIRMATION_N[market]:
        return False
    if not math.isfinite(roi) or roi <= 0:
        return False
    if int(met.get("positive_seasons") or 0) < MIN_CONFIRMATION_POSITIVE_SEASONS:
        return False
    if market in {"SPREADS", "TOTALS"} and (not math.isfinite(hit) or hit <= SPREAD_TOTAL_BREAK_EVEN):
        return False
    return True


def build_contract(replay_rows: pd.DataFrame, log_func=print) -> dict:
    if replay_rows is None or replay_rows.empty:
        raise RuntimeError("[NFL-MODEL-AUTH-V26-HOLD] EMPTY_REPLAY")
    if 2026 in set(pd.to_numeric(replay_rows.season, errors="coerce").dropna().astype(int).unique()):
        raise RuntimeError("[NFL-MODEL-AUTH-V26-HOLD] 2026_MUST_REMAIN_SEALED")
    markets = {}
    for market in ("SPREADS", "H2H", "TOTALS"):
        v = _market_vectors(replay_rows, market)
        candidates = []
        selected = None
        for th in THRESHOLDS[market]:
            d = _metrics(v, th, DISCOVERY_SEASONS)
            c = _metrics(v, th, CONFIRMATION_SEASONS)
            dp = _discovery_pass(market, d)
            row = {"threshold": float(th), "discovery": d, "confirmation_diagnostic": c, "discovery_gate": "PASS" if dp else "HOLD"}
            candidates.append(row)
            # Freeze the first/smallest discovery threshold that clears the predeclared gate.
            if selected is None and dp:
                selected = float(th)
        if selected is None:
            conf = {"n": 0, "wins": 0, "losses": 0, "pushes": 0, "hit_rate": None, "roi_per_unit": None, "positive_seasons": 0, "by_season": {}}
            status = "MODEL_ONLY_NO_DISCOVERY_THRESHOLD"
            authority = False
        else:
            conf = _metrics(v, selected, CONFIRMATION_SEASONS)
            authority = _confirmation_pass(market, conf)
            status = "MODEL_BET_AUTHORITY_FROZEN" if authority else "MODEL_ONLY_CONFIRMATION_NOT_PASSED"
        disc = _metrics(v, selected, DISCOVERY_SEASONS) if selected is not None else {"n":0}
        markets[market] = {
            "status": status,
            "production_authority": bool(authority),
            "threshold": selected,
            "threshold_unit": "POINTS" if market != "H2H" else "PROBABILITY",
            "historical_reference": "CLOSING_MARKET_PROXY",
            "discovery": disc,
            "confirmation": conf,
            "candidates": candidates,
            "strong_play_allowed": False,
            "live_price_policy": {
                "spread_total_max_lay": MAX_SPREAD_TOTAL_LAY if market in {"SPREADS", "TOTALS"} else None,
                "h2h_min_model_ev": MIN_H2H_LIVE_EV if market == "H2H" else None,
            },
        }
        log_func("[NFL-MODEL-AUTH-V26-MARKET] " + json.dumps({
            "market": market,
            "status": status,
            "threshold": selected,
            "production_authority": bool(authority),
            "discovery": disc,
            "confirmation": conf,
            "selection_rule": "FIRST_SMALLEST_DISCOVERY_THRESHOLD_PASS_THEN_CONFIRMATION_ONLY",
        }, sort_keys=True, default=str))
    core = {
        "source_tag": SOURCE_TAG,
        "engine_version": ENGINE_VERSION,
        "production_model_source_tag": prod.SOURCE_TAG,
        "production_contract_sha256": prod.production_contract()["contract_sha256"],
        "betting_authority": "FROZEN_PRODUCTION_MODEL_ONLY",
        "research_lanes_role": "EVIDENCE_ATTRIBUTION_ONLY_NO_BET_AUTHORITY",
        "discovery_seasons": list(DISCOVERY_SEASONS),
        "confirmation_seasons": list(CONFIRMATION_SEASONS),
        "year_2026_queried": False,
        "automatic_execution": False,
        "automatic_model_promotion": False,
        "markets": markets,
        "policy": {
            "action_source": "MODEL_EDGE_VS_CURRENT_MARKET",
            "threshold_selection": "FIRST_SMALLEST_DISCOVERY_THRESHOLD_PASS",
            "confirmation_reselection": False,
            "strong_play": "DISABLED_UNTIL_SEPARATELY_PROSPECTIVE_VALIDATED",
            "spread_total_historical_price": "STANDARD_MINUS_110_PROXY",
            "h2h_historical_price": "STORED_HISTORICAL_CLOSE_MONEYLINE",
        },
    }
    core["contract_sha256"] = _sha(core)
    return core


def train_publish_model_authority(*, replay_rows: pd.DataFrame, storage_client, bucket_name="sharp-models", log_func=print) -> dict:
    contract = build_contract(replay_rows, log_func=log_func)
    sha = contract["contract_sha256"]
    versioned = f"{PREFIX}/contracts/{sha}.json"
    created = _write_immutable(storage_client, bucket_name, versioned, contract)
    current_uri = _write_json(storage_client, bucket_name, CONTRACT_OBJECT, contract)
    out = {
        "status": "NFL_MODEL_AUTHORITY_V2_6_FROZEN",
        "source_tag": SOURCE_TAG,
        "contract_sha256": sha,
        "contract_uri": f"gs://{bucket_name}/{versioned}",
        "contract_created": created,
        "current_contract_uri": current_uri,
        "markets": {m: {k:v for k,v in b.items() if k in ("status","production_authority","threshold","strong_play_allowed","discovery","confirmation")} for m,b in contract["markets"].items()},
        "betting_authority": contract["betting_authority"],
        "year_2026_queried": False,
    }
    log_func("[NFL-MODEL-AUTH-V26-CONTRACT] " + json.dumps(out, sort_keys=True, default=str))
    return out


def load_contract(*, storage_client, bucket_name="sharp-models") -> dict:
    c = _read_json(storage_client, bucket_name, CONTRACT_OBJECT)
    if not isinstance(c, dict) or c.get("source_tag") != SOURCE_TAG:
        raise RuntimeError("[NFL-MODEL-AUTH-V26-HOLD] CONTRACT_MISSING_OR_STALE")
    expected = c.get("contract_sha256")
    tmp = dict(c); tmp.pop("contract_sha256", None)
    if _sha(tmp) != expected:
        raise RuntimeError("[NFL-MODEL-AUTH-V26-HOLD] CONTRACT_SHA_MISMATCH")
    if c.get("production_contract_sha256") != prod.production_contract()["contract_sha256"]:
        raise RuntimeError("[NFL-MODEL-AUTH-V26-HOLD] PRODUCTION_MODEL_CONTRACT_MISMATCH")
    return c


def _price_ok(market, price):
    p = _num(price)
    if not math.isfinite(p) or p == 0:
        return False
    if market in {"SPREADS", "TOTALS"} and p < MAX_SPREAD_TOTAL_LAY:
        return False
    return True


def _authoritative_market_snapshot(client, now) -> dict:
    """Canonical live market state through Utils only.

    Utils reads both sharp_moves_master (current/raw market) and
    moves_with_features_merged (MARKET-RICH), and also owns the 30/60/120-minute
    timing-feature builder. This authority layer performs no direct market query.
    """
    snap = market_backend.build_market_snapshot(bq_client=client, now=now)
    meta = (snap or {}).get("meta") or {}
    if int(meta.get("direct_bigquery_queries") or 0) != 0:
        raise RuntimeError("[NFL-MODEL-AUTH-V261-HOLD] MARKET_BACKEND_BYPASSED_UTILS")
    return snap

def _authoritative_market_base_row(p: dict, market: str, q: dict) -> dict | None:
    """Build canonical MODEL + current-market rows and retain both executable sides.

    Production Betting V2 may select the opposite side from the fair-value model,
    so both current side prices/books are preserved. Utils remains the canonical
    quote backend.
    """
    home = p.get("home_team")
    away = p.get("away_team")
    if market == "SPREADS":
        line = _num(q.get("current_home_spread"))
        op = _num(q.get("open_home_spread"))
        fair = _num(p.get("champion_fair_margin"))
        if not (math.isfinite(line) and math.isfinite(fair)):
            return None
        signed = fair + line
        direction = 1 if signed > 0 else -1 if signed < 0 else 0
        selected = home if direction > 0 else away if direction < 0 else None
        home_price = _num(q.get("current_home_spread_odds"))
        away_price = _num(q.get("current_away_spread_odds"))
        odds = home_price if direction >= 0 else away_price
        return {
            "model_direction": direction,
            "raw_model_edge": abs(signed),
            "market_move_toward_model": -direction * (line - op) if direction and math.isfinite(op) else np.nan,
            "selected": selected,
            "market_value": line if direction >= 0 else -line,
            "model_value": fair * direction if direction else fair,
            "selected_price": odds,
            "selected_book": q.get("current_home_spread_book" if direction >= 0 else "current_away_spread_book") if direction else None,
            "quote_timestamp": q.get("current_spread_snapshot_ts"),
            "current_home_spread": line,
            "open_home_spread": op,
            "current_total": _num(q.get("current_total")),
            "current_home_spread_odds": home_price,
            "current_away_spread_odds": away_price,
            "current_home_spread_book": q.get("current_home_spread_book"),
            "current_away_spread_book": q.get("current_away_spread_book"),
            "model_home_fair_margin": fair,
        }
    if market == "TOTALS":
        line = _num(q.get("current_total"))
        op = _num(q.get("open_total"))
        fair = _num(p.get("champion_fair_total"))
        if not (math.isfinite(line) and math.isfinite(fair)):
            return None
        signed = fair - line
        direction = 1 if signed > 0 else -1 if signed < 0 else 0
        selected = "OVER" if direction > 0 else "UNDER" if direction < 0 else None
        over_price = _num(q.get("current_over_odds"))
        under_price = _num(q.get("current_under_odds"))
        odds = over_price if direction >= 0 else under_price
        return {
            "model_direction": direction,
            "raw_model_edge": abs(signed),
            "market_move_toward_model": direction * (line - op) if direction and math.isfinite(op) else np.nan,
            "selected": selected,
            "market_value": line,
            "model_value": fair,
            "selected_price": odds,
            "selected_book": q.get("current_over_book" if direction >= 0 else "current_under_book") if direction else None,
            "quote_timestamp": q.get("current_total_snapshot_ts"),
            "current_total": line,
            "current_over_odds": over_price,
            "current_under_odds": under_price,
            "current_over_book": q.get("current_over_book"),
            "current_under_book": q.get("current_under_book"),
            "model_fair_total": fair,
        }
    if market == "H2H":
        ref = _num(q.get("current_home_novig_probability"))
        op = _num(q.get("open_home_novig_probability"))
        fair = _num(p.get("champion_home_win_probability"))
        if not (math.isfinite(ref) and math.isfinite(fair)):
            return None
        signed = fair - ref
        direction = 1 if signed > 0 else -1 if signed < 0 else 0
        selected = home if direction > 0 else away if direction < 0 else None
        home_ml = _num(q.get("current_home_ml"))
        away_ml = _num(q.get("current_away_ml"))
        odds = home_ml if direction >= 0 else away_ml
        return {
            "model_direction": direction,
            "raw_model_edge": abs(signed),
            "market_move_toward_model": direction * (ref - op) if direction and math.isfinite(op) else np.nan,
            "selected": selected,
            "market_value": ref if direction >= 0 else 1.0 - ref,
            "model_value": fair if direction >= 0 else 1.0 - fair,
            "selected_price": odds,
            "selected_book": q.get("current_home_ml_book" if direction >= 0 else "current_away_ml_book") if direction else None,
            "quote_timestamp": q.get("current_h2h_snapshot_ts"),
            "current_home_ml": home_ml,
            "current_away_ml": away_ml,
            "current_home_ml_book": q.get("current_home_ml_book"),
            "current_away_ml_book": q.get("current_away_ml_book"),
            "model_home_win_probability": fair,
        }
    raise ValueError(market)

def build_authoritative_market_rows(*, bq_client, prediction_rows, now) -> list[dict]:
    """Build the live MODEL -> ACTION input rows from production predictions + quotes.

    Crucially, this function does not load Edge Authority V2.5 or any system/stat
    artifact. If shadow research is unavailable, the production model can still
    produce its own gated action.
    """
    snapshot = _authoritative_market_snapshot(bq_client, now)
    quotes = (snapshot or {}).get("quotes") or {}
    backend_meta = (snapshot or {}).get("meta") or {}
    out = []
    for p in prediction_rows or []:
        gs = pd.to_datetime(p.get("game_start"), utc=True, errors="coerce")
        key = (
            gs.round("s").isoformat(),
            _norm(p.get("home_team")),
            _norm(p.get("away_team")),
        ) if pd.notna(gs) else None
        q = quotes.get(key, {}) if key else {}
        for market in ("SPREADS", "H2H", "TOTALS"):
            base = _authoritative_market_base_row(p, market, q)
            common = {
                "prediction_pair_id": p.get("prediction_pair_id"),
                "game_start": p.get("game_start"),
                "home_team": p.get("home_team"),
                "away_team": p.get("away_team"),
                "market": market,
                "edge_votes": [],
                "edge_sources": [],
                "system_labels": [],
                "stat_selector_support": [],
                "stat_selector_live_status": "SHADOW_NOT_MERGED",
                "independent_mechanisms": 0,
                "independence_keys": [],
                "shadow_evidence_status": "NOT_MERGED",
                "shadow_evidence_only": True,
                "betting_decision_authority": False,
                "automatic_execution": False,
            }
            backend_fields = {
                "market_backend": "UTILS",
                "current_market_source": str(backend_meta.get("raw_table") or "sharp_moves_master"),
                "market_rich_source": str(backend_meta.get("enriched_table") or "moves_with_features_merged"),
                "market_backend_source_tag": backend_meta.get("source_tag"),
                "utils_path": backend_meta.get("utils_path"),
            }
            if base is None:
                out.append({**common, **backend_fields, "action": "NO MARKET", "decision": "NO MARKET"})
            else:
                qt = pd.to_datetime(base.get("quote_timestamp"), utc=True, errors="coerce")
                base["quote_age_minutes"] = None if pd.isna(qt) else max(0.0, (pd.to_datetime(now, utc=True) - qt).total_seconds() / 60.0)
                rich = market_backend.rich_for_selection(
                    snapshot, game_start=p.get("game_start"), home_team=p.get("home_team"),
                    away_team=p.get("away_team"), market=market, selected=base.get("selected"),
                )
                out.append({**common, **backend_fields, **base, **rich, "action": "MODEL ONLY", "decision": "MODEL_ONLY_PRE_POLICY"})
    return out


def merge_shadow_evidence(authoritative_rows, shadow_rows) -> list[dict]:
    """Attach diagnostics to authoritative model rows without copying decisions.

    Only an explicit diagnostic whitelist is copied. In particular `action`,
    `decision`, `selected`, prices, model values and model direction remain from
    the independent production MODEL -> ACTION path.
    """
    shadow_map = {
        (str(r.get("prediction_pair_id") or ""), str(r.get("market") or "").upper()): r
        for r in (shadow_rows or [])
    }
    copy_fields = (
        "edge_sources", "edge_votes", "system_labels", "system_trigger_votes", "stat_selector_support",
        "stat_selector_live_status", "independent_mechanisms", "independence_keys",
        "edge_contract_sha256",
    )
    out = []
    for src in authoritative_rows or []:
        r = dict(src)
        sh = shadow_map.get((str(r.get("prediction_pair_id") or ""), str(r.get("market") or "").upper()))
        if sh is not None:
            for k in copy_fields:
                if k in sh:
                    r[k] = sh.get(k)
            r["shadow_edge_action"] = sh.get("action")
            r["shadow_edge_decision"] = sh.get("decision")
            r["shadow_evidence_status"] = "AVAILABLE"
        else:
            r["shadow_edge_action"] = None
            r["shadow_edge_decision"] = None
            r["shadow_evidence_status"] = "UNAVAILABLE"
        out.append(r)
    return out


def _shadow_states(r: dict) -> dict:
    model_dir = int(r.get("model_direction") or 0)
    statq = [x for x in (r.get("stat_selector_support") or []) if x.get("qualifies")]
    stat_state = "SUPPORT" if statq else "NEUTRAL"
    mv = _num(r.get("market_move_toward_model"))
    market_state = "SUPPORT" if math.isfinite(mv) and mv > 1e-9 else "CONFLICT" if math.isfinite(mv) and mv < -1e-9 else "NEUTRAL"
    # system_state is finalized after current qualification + family normalization.
    return {"stat_state": stat_state, "market_state": market_state}


def _parse_gs_uri(uri: str, default_bucket: str):
    u=str(uri or "")
    if not u.startswith("gs://"):
        return default_bucket, u.lstrip("/")
    rest=u[5:]
    if "/" not in rest:
        return rest, ""
    return rest.split("/",1)


def _load_heavy_runtime(storage_client, bucket_name):
    if storage_client is None:
        return None
    report=_read_json(storage_client,bucket_name,HEAVY_REPORT_OBJECT)
    if not isinstance(report,dict) or report.get("year_2026_queried") is not False:
        return None
    spread=((report.get("markets") or {}).get("SPREADS") or {})
    frozen=spread.get("frozen_base_policy") or {}
    if str(frozen.get("model") or "")!="CORE_ONLY":
        return None
    if abs(_num(frozen.get("threshold"))-FROZEN_CORE_SPREAD_THRESHOLD)>1e-12:
        return None
    if str(frozen.get("source_run_id") or "")!=FROZEN_CORE_SPREAD_RUN_ID:
        return None
    art=(report.get("challenger_artifact") or {}).get("uri")
    if not art:
        return None
    try:
        bkt,obj=_parse_gs_uri(art,bucket_name)
        raw=storage_client.bucket(bkt).blob(obj).download_as_bytes()
        bundle=joblib.load(io.BytesIO(raw))
        sm=((bundle.get("markets") or {}).get("SPREADS") or {})
        pool=sm.get("oof_error_pool")
        sf=sm.get("frozen_base_policy") or {}
        if str(sm.get("selected_residual_model") or "")!="CORE_ONLY": return None
        if str(sf.get("model") or "")!="CORE_ONLY": return None
        if abs(_num(sf.get("threshold"))-FROZEN_CORE_SPREAD_THRESHOLD)>1e-12: return None
        if not isinstance(pool,pd.DataFrame) or pool.empty: return None
        return {"report":report,"pool":pool,"artifact_uri":art,"research_contract_sha256":report.get("research_contract_sha256")}
    except Exception:
        return None


def _load_family_registry(storage_client,bucket_name,heavy_report=None):
    """Load the immutable <=2025 System Lab family registry for live trigger definitions.

    The pointer is allowed to identify the frozen family grammar only. It never grants
    qualification by itself; current qualification continues to come from Heavy's
    2023-25 attribution block. Any provenance mismatch fails closed.
    """
    if storage_client is None:
        return {}
    try:
        ptr=_read_json(storage_client,bucket_name,FAMILY_POINTER_OBJECT)
        uri=str((ptr or {}).get("family_registry_uri") or "")
        sha=str((ptr or {}).get("family_registry_sha256") or "")
        prefix=f"gs://{bucket_name}/"
        if not uri.startswith(prefix) or not sha:
            return {}
        fam=_read_json(storage_client,bucket_name,uri[len(prefix):])
        if not isinstance(fam,dict):
            return {}
        if fam.get("year_2026_queried") is not False or int(fam.get("history_through") or 2025)>2025:
            return {}
        if str(fam.get("family_registry_sha256") or "")!=sha:
            return {}
        expected=str((((heavy_report or {}).get("system_provenance") or {}).get("family_registry_sha256")) or "")
        if expected and expected!=sha:
            return {}
        return {str(x.get("system_family_id") or "").strip():x for x in (fam.get("families") or []) if str(x.get("system_family_id") or "").strip()}
    except Exception:
        return {}


def _load_overlay_registry(storage_client,bucket_name):
    if storage_client is None:
        return {}
    report=_read_json(storage_client,bucket_name,HEAVY_REPORT_OBJECT)
    attr=(report or {}).get("research_system_attribution") or {}
    if attr.get("status")!="READY" or (report or {}).get("year_2026_queried") is not False:
        return {}
    family_defs=_load_family_registry(storage_client,bucket_name,report)
    out={}
    for z in attr.get("system_rankings") or []:
        sid=str(z.get("system_id") or "").strip(); fid=str(z.get("family_id") or sid).strip()
        if not fid: continue
        fd=family_defs.get(fid) or {}
        _source=str(z.get("source") or "UNKNOWN").upper()
        _conds=list(z.get("representative_conditions") or fd.get("representative_conditions") or [])
        _season_record=any(str(c).upper().startswith(("SU_WINLESS","ATS_COVERLESS","SU_AND_ATS_WINLESS")) for c in _conds)
        # PT remains its one external-ratings family even if a PT rule later uses
        # season-state context. Legacy Miner season-state variants share one vote.
        _vote_fid=("NFL_SEASON_RECORD_STATE_FAMILY" if _season_record and _source!="PT" else fid)
        e={
            "system_id":sid,"family_id":_vote_fid,"source":_source,
            "qualified_current":bool(z.get("qualified_current")),"name":z.get("name"),
            "validation_rank":z.get("validation_rank"),"graded_n":z.get("graded_n"),
            "hit_rate":z.get("hit_rate"),"roi_per_unit":z.get("roi_per_unit"),
            "representative_conditions":_conds,
            "uses_season_record_state_family":bool(_season_record),
            "direction":str(z.get("direction") or fd.get("direction") or "").upper(),
            "market":str(z.get("market") or fd.get("market") or "").upper(),
            "family_status":z.get("evidence") or fd.get("family_status"),
            "origin_brain":z.get("origin_brain"),
            "family_registry_sha256":str((((report or {}).get("system_provenance") or {}).get("family_registry_sha256")) or ""),
        }
        for k in {sid,fid}:
            if k: out[k]=e
    return out


def _core_spread_probability(r:dict,heavy_runtime=None):
    # Test/diagnostic injection is intentionally supported; production normally computes from frozen pool.
    injected=_num(r.get("core_cover_probability"))
    if math.isfinite(injected) and 0<injected<1:
        return {"probability":float(injected),"push_probability":r.get("core_push_probability"),"source":"ROW_INJECTED"}
    if not heavy_runtime:
        return None
    try:
        import nfl_heavy_research_v31 as heavy
        direction=int(r.get("model_direction") or 0)
        edge=_num(r.get("raw_model_edge")); line=_num(r.get("current_home_spread"))
        if direction not in (-1,1) or not (math.isfinite(edge) and math.isfinite(line)):
            return None
        signed_edge=float(direction)*float(edge)
        z=heavy._distribution_probability(heavy_runtime["pool"],edge=signed_edge,line=line,market="SPREADS",line_ref="open_line")
        p_home=_num(z.get("probability"))
        if not math.isfinite(p_home): return None
        psel=float(p_home if direction>0 else 1.0-p_home)
        return {**z,"probability":psel,"home_probability":float(p_home),"source":"FROZEN_V29_EMPIRICAL_OPENING_DISTRIBUTION"}
    except Exception:
        return None


def _live_pathi_direction(system_id:str, current_home_spread, open_home_spread=None, current_total=None):
    """Evaluate the qualified Pathi key-number leader on the current executable spread.

    Returns +1 for home, -1 for away, 0 for no trigger. This is a live evaluation
    of the already-frozen Pathi rule; it does not use 2026 results or change rule qualification.
    """
    sid=str(system_id or "")
    line=_num(current_home_spread)
    if not math.isfinite(line) or abs(line)<1e-12:
        return 0
    a=abs(float(line))
    fav=1 if line<0 else -1
    dog=-fav
    def between(lo,hi): return a>float(lo) and a<float(hi)
    if sid=="Pathi_FB_Dog_Below_Key_3" and between(2,3): return dog
    if sid=="Pathi_FB_Favorite_Below_Key_3" and between(2,3): return fav
    if sid=="Pathi_FB_Dog_Below_Key_7" and between(6,7): return dog
    if sid=="Pathi_FB_Favorite_Below_Key_7" and between(6,7): return fav
    if sid=="Pathi_FB_Dog_Hook_Above_3" and between(3,4): return dog
    if sid=="Pathi_FB_Favorite_Laying_Hook_3" and between(3,4): return fav
    if sid=="Pathi_FB_Dog_Hook_Above_7" and between(7,8): return dog
    if sid=="Pathi_FB_Favorite_Laying_Hook_7" and between(7,8): return fav
    if sid=="Pathi_FB_Dog_10_Plus" and a>=10: return dog
    if sid=="Pathi_FB_Dog_Hook_Above_10" and between(10,11): return dog
    if sid=="Pathi_FB_Favorite_Below_Key_10" and between(9,10): return fav
    # Current dog total/spread compression rule when a total is available.
    if sid=="Pathi_FB_Dog_TotalSpread_Gap_LE10":
        tot=_num(current_total)
        if math.isfinite(tot) and (tot-a)<=10: return dog
    # Movement rules are evaluated from the perspective of the current dog and
    # require that the same side was also an underdog at the opener.
    op=_num(open_home_spread)
    if math.isfinite(op) and abs(op)>1e-12:
        dog_open=float(op) if dog==1 else -float(op)
        dog_cur=a
        if dog_open>0:
            for key in (3,7,10):
                if sid==f"Pathi_FB_Dog_Moved_Above_Key_{key}" and dog_open<float(key) and dog_cur>float(key): return dog
                if sid==f"Pathi_FB_Dog_Moved_Below_Key_{key}" and dog_open>float(key) and dog_cur<float(key): return dog
    return 0


def _augment_live_pathi_votes(r:dict,registry:dict):
    """Attach current qualified Pathi triggers from frozen research qualification.

    Trigger-table votes are preserved. Qualified Pathi families are derived live from
    the current/opening line so the weekly/background scanner does not depend on a
    separate trigger publisher for key-number rules.
    """
    out=dict(r)
    votes=[dict(x) for x in (out.get("system_trigger_votes") or []) if isinstance(x,dict)]
    labels=[str(x) for x in (out.get("system_labels") or []) if str(x).strip()]
    existing=set(str(x.get("family") or x.get("system_family_id") or "").strip() for x in votes)
    fam_entries={}
    for ent in (registry or {}).values():
        if not isinstance(ent,dict): continue
        if str(ent.get("source") or "").upper()!="PATHI" or not bool(ent.get("qualified_current")): continue
        fid=str(ent.get("family_id") or "").strip()
        if fid and fid not in fam_entries: fam_entries[fid]=ent
    for fid,ent in fam_entries.items():
        if fid in existing: continue
        sid=str(ent.get("system_id") or "").strip()
        d=_live_pathi_direction(sid,out.get("current_home_spread"),out.get("open_home_spread"),out.get("current_total"))
        if d not in (-1,1): continue
        name=str(ent.get("name") or sid or fid)
        votes.append({"market":"SPREADS","family":fid,"system_family_id":fid,"system_id":sid,"source":"PATHI","direction":int(d),"label":f"PATHI: {name}","live_derived":True})
        lbl=f"PATHI: {name}"
        if lbl not in labels: labels.append(lbl)
        existing.add(fid)
    out["system_trigger_votes"]=votes
    out["system_labels"]=labels
    out["live_pathi_trigger_count"]=sum(1 for x in votes if str(x.get("source") or "").upper()=="PATHI" and bool(x.get("live_derived")))
    return out



def _query_live_miner_history(bq_client):
    """Read completed 2026 side-game state needed to evaluate frozen Miner/PT triggers.

    This is prospective trigger context only. It is never joined back into Heavy
    qualification and cannot promote, re-rank, or reselect a family.

    V3.3.1 live-parity hardening also carries the prior-game fields required by
    currently-qualified PT-derived spread systems: division status, points allowed,
    and exact game turnover margin. If Postgame_Turnovers is not present in the
    historical-core view, it is merged from the canonical raw game-side table.
    """
    if bq_client is None or bigquery is None:
        return pd.DataFrame(),{"status":"NO_BIGQUERY_CLIENT"}
    try:
        cols={f.name for f in bq_client.get_table(LIVE_MINER_VIEW).schema}
        required={"Season","Season_Stage","Game_Date","Team_Norm","Opponent_Norm","Team_Score","Opponent_Score","Opening_Spread"}
        missing=sorted(required-cols)
        if missing:
            return pd.DataFrame(),{"status":"NOT_EVALUABLE_SOURCE_COLUMNS_MISSING","missing":missing}

        optional_candidates=(
            "Source_Name","Source_Game_ID","Historical_Core_Eligible",
            "Is_Division_Game","Postgame_Turnovers","Turnover_Margin",
        )
        optional=[c for c in optional_candidates if c in cols]
        select=sorted(required)+optional
        eligible=" AND Historical_Core_Eligible=1" if "Historical_Core_Eligible" in cols else ""
        order_tail=", Source_Name, Source_Game_ID" if "Source_Name" in cols and "Source_Game_ID" in cols else ""
        sql=("SELECT "+", ".join(f"`{c}`" for c in select)+f" FROM `{LIVE_MINER_VIEW}` "
             "WHERE Season=2026 AND Season_Stage IN ('REGULAR','POSTSEASON') "
             "AND Team_Score IS NOT NULL AND Opponent_Score IS NOT NULL "+eligible+
             " ORDER BY Game_Date"+order_tail+", Team_Norm")
        d=bq_client.query(sql).to_dataframe(create_bqstorage_client=False)
        if d is None:
            d=pd.DataFrame()

        raw_turnover_augmented=False
        raw_turnover_error=None
        # The canonical raw table is guaranteed by the NFL stats-context contract to
        # carry Postgame_Turnovers. Use it only when the live training view does not.
        if not d.empty and "Postgame_Turnovers" not in d.columns and {"Source_Name","Source_Game_ID"}.issubset(d.columns):
            try:
                raw_cols={f.name for f in bq_client.get_table(LIVE_MINER_RAW_SIDE_TABLE).schema}
                raw_required={"Season","Source_Name","Source_Game_ID","Team_Norm","Postgame_Turnovers"}
                if raw_required.issubset(raw_cols):
                    rq=(
                        "SELECT `Season`,`Source_Name`,`Source_Game_ID`,`Team_Norm`,`Postgame_Turnovers` "
                        f"FROM `{LIVE_MINER_RAW_SIDE_TABLE}` "
                        "WHERE Season=2026 AND Postgame_Turnovers IS NOT NULL"
                    )
                    rd=bq_client.query(rq).to_dataframe(create_bqstorage_client=False)
                    if rd is not None and not rd.empty:
                        rd=rd.drop_duplicates(["Season","Source_Name","Source_Game_ID","Team_Norm"],keep="last")
                        d=d.merge(
                            rd,
                            on=["Season","Source_Name","Source_Game_ID","Team_Norm"],
                            how="left",
                            validate="many_to_one",
                        )
                        raw_turnover_augmented=True
            except Exception as exc:
                raw_turnover_error=f"{type(exc).__name__}:{exc}"

        meta={
            "status":"READY" if not d.empty else "NO_COMPLETED_2026_HISTORY",
            "rows":int(len(d)),
            "division_context_available":bool("Is_Division_Game" in d.columns),
            "postgame_turnovers_available":bool("Postgame_Turnovers" in d.columns),
            "turnover_margin_available":bool("Turnover_Margin" in d.columns),
            "raw_turnover_augmented":bool(raw_turnover_augmented),
        }
        if raw_turnover_error:
            meta["raw_turnover_error"]=raw_turnover_error
        return d,meta
    except Exception as exc:
        return pd.DataFrame(),{"status":"HOLD_HISTORY_QUERY_FAILED","error":f"{type(exc).__name__}:{exc}"}


def _miner_histories(history_df:pd.DataFrame):
    if not isinstance(history_df,pd.DataFrame) or history_df.empty:
        return {}
    d=history_df.copy()
    d["__team"]=d.get("Team_Norm",pd.Series("",index=d.index)).map(_norm)
    d["__opp"]=d.get("Opponent_Norm",pd.Series("",index=d.index)).map(_norm)
    d["__date"]=pd.to_datetime(d.get("Game_Date"),errors="coerce")
    d["__margin"]=pd.to_numeric(d.get("Team_Score"),errors="coerce")-pd.to_numeric(d.get("Opponent_Score"),errors="coerce")
    d["__points_against"]=pd.to_numeric(d.get("Opponent_Score"),errors="coerce")
    d["__open"]=pd.to_numeric(d.get("Opening_Spread"),errors="coerce")
    d["__ats_margin"]=d["__margin"]+d["__open"]
    d["__division"]=pd.to_numeric(d.get("Is_Division_Game",pd.Series(np.nan,index=d.index)),errors="coerce")

    # Exact same-game turnover margin = opponent giveaways - team giveaways.
    # Prefer a precomputed margin when present; otherwise pair the canonical
    # Postgame_Turnovers values from the two game-side rows.
    d["__turnover_margin"]=pd.to_numeric(
        d.get("Turnover_Margin",pd.Series(np.nan,index=d.index)),errors="coerce"
    )
    own_to=pd.to_numeric(d.get("Postgame_Turnovers",pd.Series(np.nan,index=d.index)),errors="coerce")
    if own_to.notna().any():
        pair_keys=[c for c in ("Season","Source_Name","Source_Game_ID") if c in d.columns]
        if len(pair_keys)==3:
            opp=d[pair_keys+["__team"]].copy()
            opp["__opp_turnovers"]=own_to.values
            opp=opp.rename(columns={"__team":"__opp"})
            opp=opp.drop_duplicates(pair_keys+["__opp"],keep="last")
            d=d.merge(opp,on=pair_keys+["__opp"],how="left",validate="many_to_one")
            calc=pd.to_numeric(d.get("__opp_turnovers"),errors="coerce")-pd.to_numeric(d.get("Postgame_Turnovers"),errors="coerce")
            d["__turnover_margin"]=d["__turnover_margin"].where(d["__turnover_margin"].notna(),calc)
        else:
            # Fail-closed fallback when source game ids are unavailable: pair only
            # exact date/team/opponent mirrors and require a unique opponent row.
            opp=d[["__date","__team","__opp"]].copy()
            opp["__opp_turnovers"]=own_to.values
            opp=opp.rename(columns={"__team":"__opp_side","__opp":"__team_side"})
            counts=opp.groupby(["__date","__opp_side","__team_side"],dropna=False).size()
            if not counts.empty and int(counts.max())==1:
                d=d.merge(
                    opp,
                    left_on=["__date","__team","__opp"],
                    right_on=["__date","__team_side","__opp_side"],
                    how="left",
                    validate="many_to_one",
                )
                calc=pd.to_numeric(d.get("__opp_turnovers"),errors="coerce")-pd.to_numeric(d.get("Postgame_Turnovers"),errors="coerce")
                d["__turnover_margin"]=d["__turnover_margin"].where(d["__turnover_margin"].notna(),calc)

    sort=[c for c in ("__date","Source_Name","Source_Game_ID") if c in d.columns]
    d=d.sort_values(sort or ["__date"],kind="mergesort")
    out={}
    for team,g in d.groupby("__team",sort=False):
        if not team: continue
        recs=[]
        for _,r in g.iterrows():
            m=_num(r.get("__margin")); op=_num(r.get("__open")); am=_num(r.get("__ats_margin"))
            pa=_num(r.get("__points_against")); div=_num(r.get("__division")); tm=_num(r.get("__turnover_margin"))
            if not math.isfinite(m): continue
            recs.append({
                "margin":float(m),
                "opening_spread":float(op) if math.isfinite(op) else None,
                "ats_margin":float(am) if math.isfinite(am) else None,
                "points_against":float(pa) if math.isfinite(pa) else None,
                "is_division":float(div) if math.isfinite(div) else None,
                "turnover_margin":float(tm) if math.isfinite(tm) else None,
                "date":r.get("__date"),
            })
        out[team]=recs
    return out

def _miner_win_pct(hist:list[dict]):
    if not hist: return None
    vals=[]
    for r in hist:
        m=_num(r.get("margin"))
        if not math.isfinite(m): continue
        vals.append(1.0 if m>0 else 0.0 if m<0 else 0.5)
    return (sum(vals)/len(vals)) if vals else None



def _miner_condition(condition:str, *, side_dir:int, side_team:str, opp_team:str, side_open_spread, histories:dict):
    """Return True/False for frozen live atoms, None only when exact state is unavailable."""
    c=str(condition or "").upper().strip()
    th=histories.get(_norm(side_team),[]); oh=histories.get(_norm(opp_team),[])
    op=_num(side_open_spread)
    if c=="ROAD_DOG":
        return bool(side_dir==-1 and math.isfinite(op) and op>0)
    if c=="OPP_OFF_SU_LOSS":
        return None if not oh else bool(_num(oh[-1].get("margin"))<0)
    if c=="OPP_WINPCT_LE_500":
        wp=_miner_win_pct(oh); return None if wp is None else bool(wp<=0.5+1e-12)
    if c=="OFF_ATS_COVER_7_PLUS":
        if not th: return None
        am=_num(th[-1].get("ats_margin")); return None if not math.isfinite(am) else bool(am>=7.0-1e-12)
    if c=="ROLE_FLIP_FAVORITE_TO_DOG":
        if not th or not math.isfinite(op): return None
        prev=_num(th[-1].get("opening_spread")); return None if not math.isfinite(prev) else bool(prev<0 and op>0)
    if c=="SU_SEQ2_LW":
        if len(th)<2: return None
        # System Lab convention is oldest -> newest: calc_prev2, calc_prev1.
        m2=_num(th[-2].get("margin")); m1=_num(th[-1].get("margin"))
        return None if not (math.isfinite(m2) and math.isfinite(m1)) else bool(m2<0 and m1>0)

    # V3.3.1 live atom parity for the six qualified PT-derived spread systems that
    # previously failed closed despite having frozen <=2025 qualification.
    if c=="LAST_GAME_FAVORITE":
        if not th: return None
        prev=_num(th[-1].get("opening_spread"))
        return None if not math.isfinite(prev) else bool(prev<0)
    if c=="OPP_LAST_GAME_DOG":
        if not oh: return None
        prev=_num(oh[-1].get("opening_spread"))
        return None if not math.isfinite(prev) else bool(prev>0)
    if c=="LAST_GAME_DIVISION":
        if not th: return None
        div=_num(th[-1].get("is_division"))
        return None if not math.isfinite(div) else bool(div==1)
    if c=="DEFENSE_IMPROVING_2":
        if len(th)<2: return None
        pa2=_num(th[-2].get("points_against")); pa1=_num(th[-1].get("points_against"))
        return None if not (math.isfinite(pa2) and math.isfinite(pa1)) else bool(pa1<pa2 and (pa2-pa1)>=7.0-1e-12)
    if c=="TURNOVER_POS_LAST3":
        vals=[]
        for z in th[-3:]:
            tm=_num(z.get("turnover_margin"))
            if math.isfinite(tm): vals.append(float(tm))
        # System Lab's rolling feature uses min_periods=1, so one or more exact
        # completed-game margins is enough; use up to the latest three.
        return None if not vals else bool((sum(vals)/len(vals))>0.5+1e-12)

    # V3.3.2 season-record/desperation live parity. Histories contain only
    # completed 2026 games before the current matchup, so these are prior-only.
    if c in {"TEAM_GAME_3","TEAM_GAME_4","TEAM_GAME_5","TEAM_GAME_6"}:
        try: want=int(c.rsplit("_",1)[-1])
        except Exception: return None
        return bool(len(th)+1==want)
    if c in {"DOG_6P5_PLUS","DOG_7_PLUS","DOG_8_PLUS","DOG_9_PLUS","DOG_10_PLUS"}:
        if not math.isfinite(op): return None
        cuts={"DOG_6P5_PLUS":6.5,"DOG_7_PLUS":7.0,"DOG_8_PLUS":8.0,"DOG_9_PLUS":9.0,"DOG_10_PLUS":10.0}
        return bool(op>=cuts[c]-1e-12)
    if c=="OPP_HAS_SU_WIN":
        if not oh: return None
        vals=[_num(z.get("margin")) for z in oh]
        vals=[x for x in vals if math.isfinite(x)]
        return None if not vals else bool(any(x>0 for x in vals))
    if c.startswith(("SU_WINLESS","ATS_COVERLESS","SU_AND_ATS_WINLESS")):
        if not th: return None
        margins=[_num(z.get("margin")) for z in th]
        if any(not math.isfinite(x) for x in margins): return None
        ats=[_num(z.get("ats_margin")) for z in th]
        su_winless=not any(x>0 for x in margins)
        # Exact 0-X ATS contract: every prior ATS result must be gradeable and a loss.
        ats_available=all(math.isfinite(x) for x in ats)
        ats_coverless=all(x<0 for x in ats) if ats_available else None
        min_games=1
        m=re.search(r"_AFTER_(\d+)_PLUS$",c)
        if m: min_games=int(m.group(1))
        if len(th)<min_games: return False
        if c.startswith("SU_AND_ATS_WINLESS"):
            return None if ats_coverless is None else bool(su_winless and ats_coverless)
        if c.startswith("SU_WINLESS"):
            return bool(su_winless)
        return None if ats_coverless is None else bool(ats_coverless)
    return None

def _load_live_pt_spread_context(rows, storage_client, bucket_name="sharp-models", log_func=print):
    """Attach the current Prediction Tracker spread file to live games for trigger context only.

    The current file is never used to qualify/reselect a system.  It only answers
    whether an already-qualified <=2025 PT-derived rule fires on the current slate.
    """
    if storage_client is None:
        return {}, {"status":"NO_STORAGE_CLIENT","year_2026_role":"TRIGGER_CONTEXT_ONLY"}
    try:
        import nfl_system_lab_v3 as lab
        path=f"{lab.PT_RAW_PREFIX}/nflpredictions.csv"
        blob=storage_client.bucket(bucket_name).blob(path)
        if not blob.exists():
            return {}, {"status":"MISSING_CURRENT_PT_FILE","gcs_path":f"gs://{bucket_name}/{path}","year_2026_role":"TRIGGER_CONTEXT_ONLY"}
        raw=blob.download_as_bytes()
        q=pd.read_csv(io.BytesIO(raw))
        home=lab._pt_find_col(q,"home"); road=lab._pt_find_col(q,"road","away","visitor"); week=lab._pt_find_col(q,"week"); line=lab._pt_find_col(q,"line")
        if not home or not road or not line:
            return {}, {"status":"INVALID_CURRENT_PT_SCHEMA","year_2026_role":"TRIGGER_CONTEXT_ONLY"}
        preds=lab._pt_spread_predictor_cols(q)
        if len(preds)<2:
            return {}, {"status":"NO_CURRENT_PT_PREDICTORS","year_2026_role":"TRIGGER_CONTEXT_ONLY"}
        pt=pd.DataFrame(index=q.index)
        pt["Season"]=2026
        pt["Week"]=pd.to_numeric(q[week],errors="coerce") if week else np.nan
        pt["Home_Raw"]=q[home].astype(str); pt["Road_Raw"]=q[road].astype(str)
        pt["Home_Code"]=pt["Home_Raw"].map(lab._pt_team_code); pt["Road_Code"]=pt["Road_Raw"].map(lab._pt_team_code)
        pt["PT_Line"]=pd.to_numeric(q[line],errors="coerce")
        for canon,orig in preds: pt[f"PRED__{canon}"]=pd.to_numeric(q[orig],errors="coerce")
        pt=pt.loc[pt["Home_Code"].ne("")&pt["Road_Code"].ne("")].copy()

        states=[]
        seen=set()
        for r in rows or []:
            if str(r.get("market") or "").upper()!="SPREADS": continue
            gid=str(r.get("prediction_pair_id") or "").strip()
            if not gid or gid in seen: continue
            seen.add(gid)
            h=str(r.get("home_team") or ""); a=str(r.get("away_team") or ""); op=_num(r.get("open_home_spread"))
            if not h or not a or not math.isfinite(op): continue
            base={"Season":2026,"Week_Number":np.nan,"physical_game_id":gid}
            states.append({**base,"Team_Norm":h,"Opponent_Norm":a,"Is_Home":1,"Is_Away":0,"Opening_Spread":float(op)})
            states.append({**base,"Team_Norm":a,"Opponent_Norm":h,"Is_Home":0,"Is_Away":1,"Opening_Spread":float(-op)})
        if not states:
            return {}, {"status":"NO_LIVE_SPREAD_ROWS","year_2026_role":"TRIGGER_CONTEXT_ONLY"}
        attached,predictors,match=lab._pt_attach_spread(pd.DataFrame(states),pt,log_func=lambda *_:None)
        ctx={}
        for _,z in attached.iterrows():
            gid=str(z.get("physical_game_id") or "")
            if not gid: continue
            side="home" if int(_num(z.get("Is_Home")) or 0)==1 else "away"
            ctx.setdefault(gid,{})[side]=z.to_dict()
        meta={"status":"READY","matched_games":int(match.get("matched_games") or 0),"coverage":match.get("coverage"),"predictors":len(predictors),
              "gcs_path":f"gs://{bucket_name}/{path}","qualification_from_2026":False,"year_2026_role":"TRIGGER_CONTEXT_ONLY"}
        return ctx,meta
    except Exception as exc:
        return {}, {"status":"ERROR","error":f"{type(exc).__name__}:{exc}","year_2026_role":"TRIGGER_CONTEXT_ONLY"}


def _pt_live_spread_condition(condition:str, pt_side:dict|None):
    """Evaluate one frozen PT spread atom for one oriented side."""
    if not isinstance(pt_side,dict): return None
    c=str(condition or "").upper().strip()
    try:
        import nfl_system_lab_v3 as lab
    except Exception:
        return None
    suffix="_RECOMMENDS_SIDE_2PLUS"
    if c.startswith("PTSP_") and c.endswith(suffix):
        tok=c[len("PTSP_"):-len(suffix)]
        vals=[]
        for k,v in pt_side.items():
            if not str(k).startswith("PTSP_EDGE__"): continue
            pred=str(k)[len("PTSP_EDGE__"):]
            if str(lab._expert_token(pred)).upper()==tok:
                x=_num(v)
                if math.isfinite(x): vals.append(x)
        return None if not vals else bool(max(vals)>=2.0-1e-12)
    edge=_num(pt_side.get("PTSP_CONSENSUS_EDGE")); agree=_num(pt_side.get("PTSP_AGREE_FRAC")); sd=_num(pt_side.get("PTSP_CLUSTER_STD")); n=_num(pt_side.get("PTSP_CLUSTER_COUNT"))
    if c=="PTSP_CLUSTER_CONSENSUS_EDGE_2PLUS":
        return None if not (math.isfinite(edge) and math.isfinite(n)) else bool(n>=8 and edge>=2.0)
    if c=="PTSP_CLUSTER_CONSENSUS_EDGE_3PLUS":
        return None if not (math.isfinite(edge) and math.isfinite(n)) else bool(n>=8 and edge>=3.0)
    if c=="PTSP_CLUSTER_CONSENSUS_70_EDGE2":
        return None if not all(math.isfinite(x) for x in (edge,agree,n)) else bool(n>=8 and edge>=2.0 and agree>=.70)
    if c=="PTSP_CLUSTER_TIGHT_70_EDGE2":
        return None if not all(math.isfinite(x) for x in (edge,agree,sd,n)) else bool(n>=8 and edge>=2.0 and agree>=.70 and sd<=4.0)
    return None


def augment_live_miner_votes(rows, *, bq_client=None, storage_client=None, bucket_name="sharp-models", overlay_registry=None, history_df=None, pt_context_by_game=None, log_func=print):
    """Evaluate source-neutral qualified Miner/PT systems against live pregame state.

    Qualification is read only from Heavy's <=2025 attribution registry. 2026 completed
    games and current PT forecasts are trigger context only; neither can qualify or
    reselect a system.  Unsupported atoms fail closed as NOT_EVALUABLE.
    """
    reg=overlay_registry if overlay_registry is not None else _load_overlay_registry(storage_client,bucket_name)
    systems={}
    for ent in (reg or {}).values():
        if not isinstance(ent,dict) or str(ent.get("source") or "").upper() not in {"MINER","PT"} or not bool(ent.get("qualified_current")): continue
        if str(ent.get("market") or "SPREADS").upper()!="SPREADS": continue
        sid=str(ent.get("system_id") or ent.get("family_id") or "").strip()
        if sid and sid not in systems: systems[sid]=ent
    if history_df is None:
        history_df,hmeta=_query_live_miner_history(bq_client)
    else:
        hmeta={"status":"INJECTED_TEST_HISTORY","rows":int(len(history_df)) if isinstance(history_df,pd.DataFrame) else 0}
    histories=_miner_histories(history_df)
    if pt_context_by_game is None:
        pt_context_by_game,ptmeta=_load_live_pt_spread_context(rows,storage_client,bucket_name,log_func=log_func)
    else:
        ptmeta={"status":"INJECTED_TEST_CONTEXT","games":len(pt_context_by_game or {}),"year_2026_role":"TRIGGER_CONTEXT_ONLY"}
    qids=sorted(systems)
    supported_atoms={"ROAD_DOG","OPP_OFF_SU_LOSS","OPP_WINPCT_LE_500","OFF_ATS_COVER_7_PLUS","ROLE_FLIP_FAVORITE_TO_DOG","SU_SEQ2_LW","LAST_GAME_FAVORITE","OPP_LAST_GAME_DOG","LAST_GAME_DIVISION","DEFENSE_IMPROVING_2","TURNOVER_POS_LAST3",
                     "TEAM_GAME_3","TEAM_GAME_4","TEAM_GAME_5","TEAM_GAME_6","DOG_6P5_PLUS","DOG_7_PLUS","DOG_8_PLUS","DOG_9_PLUS","DOG_10_PLUS","OPP_HAS_SU_WIN",
                     "SU_WINLESS_PRIOR","ATS_COVERLESS_PRIOR","SU_AND_ATS_WINLESS_PRIOR","SU_WINLESS_AFTER_2_PLUS","SU_WINLESS_AFTER_3_PLUS","SU_WINLESS_AFTER_4_PLUS",
                     "ATS_COVERLESS_AFTER_2_PLUS","ATS_COVERLESS_AFTER_3_PLUS","ATS_COVERLESS_AFTER_4_PLUS","SU_AND_ATS_WINLESS_AFTER_2_PLUS","SU_AND_ATS_WINLESS_AFTER_3_PLUS","SU_AND_ATS_WINLESS_AFTER_4_PLUS"}
    def cond_supported(c):
        c=str(c).upper()
        return c in supported_atoms or c.startswith("PTSP_")
    evaluable=[]; not_eval=[]
    for sid,e in systems.items():
        conds=[str(x).upper() for x in (e.get("representative_conditions") or [])]
        pt_needed=any(c.startswith("PTSP_") for c in conds)
        pt_possible=(ptmeta.get("status") in {"READY","INJECTED_TEST_CONTEXT"}) if pt_needed else True
        if conds and all(cond_supported(c) for c in conds) and pt_possible: evaluable.append(sid)
        else: not_eval.append(sid)
    out=[]; trigger_systems=set(); trigger_families=set(); support_count=0; conflict_count=0
    for src in rows or []:
        r=dict(src)
        if str(r.get("market") or "").upper()!="SPREADS":
            out.append(r); continue
        votes=[dict(x) for x in (r.get("system_trigger_votes") or []) if isinstance(x,dict)]
        labels=[str(x) for x in (r.get("system_labels") or []) if str(x).strip()]
        existing_sids={str(x.get("system_id") or "").strip() for x in votes}
        home=str(r.get("home_team") or ""); away=str(r.get("away_team") or "")
        hopen=_num(r.get("open_home_spread")); gid=str(r.get("prediction_pair_id") or "")
        ptg=(pt_context_by_game or {}).get(gid) or {}
        fired=[]; row_not=[]
        for sid,e in systems.items():
            if sid in existing_sids: continue
            conds=[str(x).upper() for x in (e.get("representative_conditions") or [])]
            if not conds or any(not cond_supported(c) for c in conds):
                row_not.append(sid); continue
            matches=[]; indeterminate=False
            for side_dir,team,opp,side_open,ptside in ((1,home,away,hopen,ptg.get("home")),(-1,away,home,-hopen if math.isfinite(hopen) else np.nan,ptg.get("away"))):
                vals=[]
                for c in conds:
                    if c.startswith("PTSP_"):
                        vals.append(_pt_live_spread_condition(c,ptside))
                    else:
                        vals.append(_miner_condition(c,side_dir=side_dir,side_team=team,opp_team=opp,side_open_spread=side_open,histories=histories))
                if any(v is None for v in vals):
                    indeterminate=True; continue
                if all(bool(v) for v in vals): matches.append(side_dir)
            if len(matches)!=1:
                if indeterminate and not matches: row_not.append(sid)
                continue
            d=matches[0]
            direction=str(e.get("direction") or "PLAY_ON").upper()
            if direction=="FADE": d=-d
            elif direction!="PLAY_ON": row_not.append(sid); continue
            family=str(e.get("family_id") or sid).strip(); source=str(e.get("source") or "MINER").upper(); name=str(e.get("name") or sid)
            votes.append({"market":"SPREADS","family":family,"system_family_id":family,"system_id":sid,
                          "source":source,"direction":int(d),"label":f"{source}: {name}","live_derived":True,
                          "qualification_source":"FROZEN_HEAVY_2023_25_OR_EARLIER","representative_conditions":conds})
            lbl=f"{source}: {name}"
            if lbl not in labels: labels.append(lbl)
            existing_sids.add(sid); fired.append(sid); trigger_systems.add(sid); trigger_families.add(family)
        model_dir=int(r.get("model_direction") or 0)
        # Count resolved independent-family directions, not raw systems.  This is the
        # same family collapse later consumed by Bet Authority and caps PT at one vote.
        live_fam={}
        for v in votes:
            if not v.get("live_derived"): continue
            sid=str(v.get("system_id") or ""); ent=reg.get(sid) or reg.get(str(v.get("family") or ""))
            if not ent or not ent.get("qualified_current"): continue
            fam=str(ent.get("family_id") or v.get("family") or sid)
            try:d=int(v.get("direction") or 0)
            except Exception:d=0
            if d in (-1,1): live_fam.setdefault(fam,set()).add(d)
        for fam,ds in live_fam.items():
            if len(ds)!=1: continue
            d=next(iter(ds))
            if model_dir and d==model_dir:support_count+=1
            elif model_dir and d==-model_dir:conflict_count+=1
        r["system_trigger_votes"]=votes; r["system_labels"]=labels
        r["live_miner_qualified_families"]=qids
        r["live_miner_evaluable_families"]=sorted(evaluable)
        r["live_miner_trigger_families"]=sorted(fired)
        r["live_miner_not_evaluable"]=sorted(set(not_eval+row_not))
        r["live_miner_trigger_count"]=len(fired)
        r["live_miner_status"]=("TRIGGERED" if fired else "READY_NO_TRIGGER" if evaluable and hmeta.get("status") in {"READY","INJECTED_TEST_HISTORY"} else "NOT_EVALUABLE")
        out.append(r)
    diag={"status":"READY" if evaluable and hmeta.get("status") in {"READY","INJECTED_TEST_HISTORY"} else "HOLD_NOT_EVALUABLE",
          "history":hmeta,"prediction_tracker_current":ptmeta,"qualified_families_loaded":len(qids),"qualified_family_ids":qids,
          "evaluable_families":len(evaluable),"evaluable_family_ids":sorted(evaluable),
          "not_evaluable_families":sorted(not_eval),"triggered_systems":sorted(trigger_systems),"triggered_families":sorted(trigger_families),
          "triggers_fired":len(trigger_systems),"supporting_core_votes":support_count,"conflicting_core_votes":conflict_count,
          "qualification_window":"FROZEN_2023_2025_OR_EARLIER","source_neutral":True,"supported_live_atoms":sorted(supported_atoms),"pt_family_vote_cap":1,"pt_model_weight":0.0,
          "year_2026_role":"TRIGGER_CONTEXT_ONLY_NO_QUALIFICATION"}
    try: log_func("[NFL-LIVE-SYSTEM-V33] "+json.dumps(diag,sort_keys=True,default=str))
    except Exception: pass
    return out,diag


def _qualified_system_overlay(r:dict,registry:dict):
    model_dir=int(r.get("model_direction") or 0)
    fam_dirs={}; fam_meta={}
    for raw in (r.get("system_trigger_votes") or []):
        if not isinstance(raw,dict): continue
        sid=str(raw.get("system_id") or raw.get("family") or raw.get("system_family_id") or "").strip()
        rawfam=str(raw.get("family") or raw.get("system_family_id") or sid).strip()
        ent=registry.get(sid) or registry.get(rawfam)
        if not ent or not ent.get("qualified_current"): continue
        fid=str(ent.get("family_id") or rawfam or sid).strip()
        try: d=int(raw.get("direction") or 0)
        except Exception: d=0
        if not fid or d not in (-1,1): continue
        fam_dirs.setdefault(fid,set()).add(d); fam_meta[fid]=ent
    # Compatibility: an older Edge event may already carry a qualified family vote. It is
    # accepted only when that family is present/qualified in the current normalized registry.
    for raw in (r.get("edge_votes") or []):
        if not isinstance(raw,dict): continue
        rawfam=str(raw.get("mechanism_family_id") or "").strip(); ent=registry.get(rawfam)
        if not ent or not ent.get("qualified_current"): continue
        fid=str(ent.get("family_id") or rawfam).strip()
        try: d=int(raw.get("direction") or 0)
        except Exception: d=0
        if fid and d in (-1,1): fam_dirs.setdefault(fid,set()).add(d); fam_meta[fid]=ent
    support=[]; conflict=[]; internal=[]
    for fid,vals in fam_dirs.items():
        if len(vals)!=1:
            internal.append(fid); continue
        d=next(iter(vals)); item=fam_meta.get(fid) or {}
        rec={"family_id":fid,"source":item.get("source") or "UNKNOWN","system_id":item.get("system_id"),"name":item.get("name"),"direction":d}
        if model_dir and d==model_dir: support.append(rec)
        elif model_dir and d==-model_dir: conflict.append(rec)
    return {
        "support":support,"conflict":conflict,"internal_conflict_families":sorted(internal),
        "support_families":sorted(x["family_id"] for x in support),
        "conflict_families":sorted(x["family_id"] for x in conflict),
        "support_sources":sorted(set(str(x.get("source") or "UNKNOWN") for x in support)),
        "conflict_sources":sorted(set(str(x.get("source") or "UNKNOWN") for x in conflict)),
    }


def apply_model_authority_rows(rows, contract, storage_client=None, bucket_name="sharp-models", heavy_runtime=None, overlay_registry=None):
    """Apply Production Betting V3.1 CORE-candidate / system-confirmation policy.

    CORE alone creates Spread candidates from the frozen 0.575 empirical probability
    benchmark plus a +2% executable-price EV gate. A candidate is not a wager. One
    qualified independent supporting family confirms BET; two or more confirm STRONG BET.
    Mixed/single conflict holds the play at CANDIDATE; 2+ net conflicts veto to PASS.
    Systems never create a candidate and never reverse the CORE side.
    """
    if heavy_runtime is None and storage_client is not None:
        heavy_runtime=_load_heavy_runtime(storage_client,bucket_name)
    if overlay_registry is None:
        overlay_registry=_load_overlay_registry(storage_client,bucket_name) if storage_client is not None else {}
    out=[]
    for src in rows or []:
        r=dict(src); market=str(r.get("market") or "").upper(); model_dir=int(r.get("model_direction") or 0)
        if market=="SPREADS": r=_augment_live_pathi_votes(r,overlay_registry or {})
        action="MODEL ONLY"; reason="MODEL_ONLY"; confidence="MODEL_ONLY"; coreq=None; live_ev=np.nan
        ov={"support":[],"conflict":[],"internal_conflict_families":[],"support_families":[],"conflict_families":[],"support_sources":[],"conflict_sources":[]}
        if market=="SPREADS": ov=_qualified_system_overlay(r,overlay_registry or {})
        support_n=len(ov["support_families"]); conflict_n=len(ov["conflict_families"])
        if r.get("action")=="NO MARKET" or model_dir not in (-1,1):
            action="NO MARKET"; reason="CURRENT_MARKET_OR_MODEL_VALUE_UNAVAILABLE"; confidence="NO_MARKET"
        elif market=="SPREADS":
            coreq=_core_spread_probability(r,heavy_runtime)
            p=_num((coreq or {}).get("probability")); price=_num(r.get("selected_price"))
            live_ev=_american_ev(p,price)
            if not math.isfinite(p):
                action="MODEL ONLY"; reason="FROZEN_CORE_0575_RUNTIME_UNAVAILABLE"; confidence="MODEL_ONLY_RUNTIME_HOLD"
            elif p < FROZEN_CORE_SPREAD_THRESHOLD:
                action="PASS"; reason="CORE_PROBABILITY_BELOW_0575"; confidence="CORE_BELOW_GATE"
            elif not math.isfinite(price):
                action="EDGE — NO EXEC QUOTE"; reason="CORE_0575_PASS_BUT_EXECUTABLE_PRICE_UNAVAILABLE"; confidence="CORE_QUALIFIED_NO_PRICE"
            elif not math.isfinite(live_ev) or live_ev < MIN_CORE_SPREAD_LIVE_EV:
                action="PASS"; reason="CORE_0575_PASS_BUT_LIVE_EV_BELOW_2PCT"; confidence="CORE_PRICE_NOT_GOOD_ENOUGH"
            elif conflict_n>=STRONG_CONFLICT_FAMILY_COUNT and conflict_n>support_n:
                action="PASS"; reason="CORE_CANDIDATE_VETOED_BY_2PLUS_NET_QUALIFIED_SYSTEM_CONFLICT"; confidence="CORE_CANDIDATE_STRONG_SYSTEM_CONFLICT"
            elif support_n>=STRONG_SUPPORT_FAMILY_COUNT and conflict_n==0:
                action="STRONG BET"; reason="CORE_CANDIDATE_CONFIRMED_BY_2PLUS_INDEPENDENT_SYSTEM_FAMILIES"; confidence="STRONG_CONFLUENCE"
            elif support_n>=1 and conflict_n==0:
                action="BET"; reason="CORE_CANDIDATE_CONFIRMED_BY_1_INDEPENDENT_SYSTEM_FAMILY"; confidence="SYSTEM_CONFIRMED"
            elif conflict_n>0:
                action="CANDIDATE"; reason="CORE_CANDIDATE_HELD_NO_WAGER_DUE_TO_SYSTEM_CONFLICT_OR_MIXED_EVIDENCE"; confidence="CANDIDATE_CAUTION"
            else:
                action="CANDIDATE"; reason="CORE_CANDIDATE_AWAITING_INDEPENDENT_SYSTEM_CONFIRMATION"; confidence="CANDIDATE_UNCONFIRMED"
        elif market=="H2H":
            action="MODEL ONLY"; reason="H2H_HAS_NO_FROZEN_BETTING_EV_GATE"; confidence="MODEL_ONLY"
        elif market=="TOTALS":
            action="MODEL ONLY"; reason="TOTALS_CORE_HAS_NO_FROZEN_PROBABILITY_GATE"; confidence="MODEL_ONLY"
        else:
            action="MODEL ONLY"; reason="UNSUPPORTED_MARKET"; confidence="MODEL_ONLY"

        shadow=_shadow_states(r)
        system_state="NEUTRAL"
        if support_n and conflict_n: system_state="MIXED"
        elif support_n: system_state="SUPPORT"
        elif conflict_n: system_state="CONFLICT"
        core_p=_num((coreq or {}).get("probability")); push_p=_num((coreq or {}).get("push_probability"))
        r.update({
            "action":action,
            "decision":"PRODUCTION_STRONG_BET" if action=="STRONG BET" else "PRODUCTION_BET" if action=="BET" else "CORE_CANDIDATE_NO_WAGER" if action=="CANDIDATE" else "PRODUCTION_PASS" if action=="PASS" else action,
            "decision_reason":reason,
            "betting_authority":"NFL_PRODUCTION_BETTING_V3_2_CORE_CANDIDATE_CONFIRMED" if market=="SPREADS" else "FROZEN_PRODUCTION_MODEL_ONLY",
            "authority_source_tag":BETTING_POLICY_SOURCE_TAG,
            "production_betting_policy":BETTING_POLICY_SOURCE_TAG,
            "legacy_edge_contract_sha256":LEGACY_EDGE_CONTRACT_SHA256,
            "legacy_edge_contract_match":str(r.get("edge_contract_sha256") or "")==LEGACY_EDGE_CONTRACT_SHA256,
            "core_cover_probability":None if not math.isfinite(core_p) else float(core_p),
            "core_home_cover_probability":None if not math.isfinite(_num((coreq or {}).get("home_probability"))) else float(_num((coreq or {}).get("home_probability"))),
            "core_push_probability":None if not math.isfinite(push_p) else float(push_p),
            "core_probability_source":(coreq or {}).get("source"),
            "core_probability_threshold":FROZEN_CORE_SPREAD_THRESHOLD if market=="SPREADS" else None,
            "core_live_ev":None if not math.isfinite(live_ev) else float(live_ev),
            "core_min_live_ev":MIN_CORE_SPREAD_LIVE_EV if market=="SPREADS" else None,
            "core_candidate_qualifies":bool(market=="SPREADS" and math.isfinite(core_p) and core_p>=FROZEN_CORE_SPREAD_THRESHOLD and math.isfinite(live_ev) and live_ev>=MIN_CORE_SPREAD_LIVE_EV),
            "qualified_system_support_count":support_n,
            "qualified_system_conflict_count":conflict_n,
            "qualified_system_support_families":ov["support_families"],
            "qualified_system_conflict_families":ov["conflict_families"],
            "qualified_system_support_sources":ov["support_sources"],
            "qualified_system_conflict_sources":ov["conflict_sources"],
            "qualified_system_internal_conflicts":ov["internal_conflict_families"],
            "production_edge_direction":model_dir if action in {"BET","STRONG BET"} else 0,
            "production_edge_families":ov["support_families"],
            "production_edge_mechanism_count":support_n,
            "model_policy_threshold":FROZEN_CORE_SPREAD_THRESHOLD if market=="SPREADS" else None,
            "model_policy_status":"FROZEN_CORE_0575" if market=="SPREADS" else "MODEL_ONLY_NO_FROZEN_BETTING_GATE",
            "model_production_authority":bool(market=="SPREADS"),
            "model_live_ev":None if not math.isfinite(live_ev) else float(live_ev),
            "model_confidence":confidence,
            "recommendation_confidence":confidence,
            "system_state":system_state,
            **shadow,
            "shadow_evidence_only":False if market=="SPREADS" else True,
            "betting_decision_authority":bool(action in {"BET","STRONG BET"}),
            "automatic_execution":False,
        })
        out.append(r)
    return out

def _list_json(sc, bucket, prefix):
    out = []
    for b in sc.list_blobs(bucket, prefix=prefix):
        if str(b.name).endswith(".json"):
            try:
                out.append(json.loads(b.download_as_text()))
            except Exception:
                pass
    return out


def _capture(sc, bucket, rows, now, contract_sha):
    inserted=0; existing=0
    for r in rows:
        if r.get("action") not in {"BET","STRONG BET"}: continue
        rid=_sha({"contract":contract_sha,"betting_policy":BETTING_POLICY_SOURCE_TAG,"prediction_pair_id":r.get("prediction_pair_id"),"market":r.get("market")})
        event={**r,"model_bet_event_id":rid,"captured_at":pd.to_datetime(now,utc=True).isoformat(),"source_tag":SOURCE_TAG,"betting_policy_source_tag":BETTING_POLICY_SOURCE_TAG}
        if _write_immutable(sc,bucket,f"{EVENT_PREFIX}/{rid}.json",event): inserted+=1
        else: existing+=1
    return {"inserted":inserted,"existing":existing}


def _settlement_rows(client, pair_ids):
    if not pair_ids: return pd.DataFrame()
    q=f"SELECT * FROM `{PAIRED_SETTLEMENT_TABLE}` WHERE prediction_pair_id IN UNNEST(@ids)"
    cfg=bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids","STRING",list(pair_ids))])
    return client.query(q,job_config=cfg).to_dataframe(create_bqstorage_client=False)


def _settle(client, sc, bucket, now):
    events=_list_json(sc,bucket,EVENT_PREFIX); settled={str(x.get("model_bet_event_id")) for x in _list_json(sc,bucket,SETTLEMENT_PREFIX)}
    pending=[e for e in events if str(e.get("model_bet_event_id")) not in settled]
    if not pending:return {"pending_before":0,"inserted":0}
    s=_settlement_rows(client,{str(e.get("prediction_pair_id")) for e in pending}); by={str(r.prediction_pair_id):r for _,r in s.iterrows()}; n=0
    for e in pending:
        r=by.get(str(e.get("prediction_pair_id")))
        if r is None: continue
        market=str(e.get("market") or "").upper(); sel=str(e.get("selected") or ""); result="UNRESOLVED"; profit=np.nan; price=_num(e.get("selected_price")); win_profit=_american_profit_if_win(price)
        if market=="SPREADS":
            line=_num(e.get("market_value")); margin=_num(r.actual_margin); home_sel=_norm(sel)==_norm(e.get("home_team")); v=(margin+line) if home_sel else -(margin+line)
            result="WIN" if v>0 else "LOSS" if v<0 else "PUSH"; profit=win_profit if result=="WIN" else -1.0 if result=="LOSS" else 0.0
        elif market=="TOTALS":
            line=_num(e.get("market_value")); actual=_num(r.actual_total); v=actual-line; v=-v if sel.upper()=="UNDER" else v
            result="WIN" if v>0 else "LOSS" if v<0 else "PUSH"; profit=win_profit if result=="WIN" else -1.0 if result=="LOSS" else 0.0
        elif market=="H2H":
            home_win=_num(r.home_win_label)==1; won=home_win if _norm(sel)==_norm(e.get("home_team")) else not home_win; result="WIN" if won else "LOSS"; profit=win_profit if won else -1.0
        payload={"model_bet_event_id":e.get("model_bet_event_id"),"prediction_pair_id":e.get("prediction_pair_id"),"market":market,"selected":sel,"result":result,"profit_per_unit":None if not math.isfinite(_num(profit)) else float(profit),"settled_at":pd.to_datetime(now,utc=True).isoformat(),"source_tag":SOURCE_TAG,"betting_policy_source_tag":e.get("betting_policy_source_tag") or e.get("production_betting_policy")}
        if _write_immutable(sc,bucket,f"{SETTLEMENT_PREFIX}/{e['model_bet_event_id']}.json",payload): n+=1
    return {"pending_before":len(pending),"inserted":n}


def _performance(sc,bucket):
    all_s=_list_json(sc,bucket,SETTLEMENT_PREFIX)
    s=[r for r in all_s if str(r.get("betting_policy_source_tag") or "")==BETTING_POLICY_SOURCE_TAG]
    def agg(x):
        w=sum(1 for r in x if r.get("result")=="WIN");l=sum(1 for r in x if r.get("result")=="LOSS");p=sum(1 for r in x if r.get("result")=="PUSH");profits=[_num(r.get("profit_per_unit")) for r in x if math.isfinite(_num(r.get("profit_per_unit")))]
        return {"n":len(x),"wins":w,"losses":l,"pushes":p,"hit_rate":round(w/(w+l),6) if w+l else None,"roi_per_unit":round(float(np.mean(profits)),6) if profits else None}
    out={"ALL":agg(s),"policy_source_tag":BETTING_POLICY_SOURCE_TAG,"all_policy_settlements":len(all_s)}
    for m in ("SPREADS","H2H","TOTALS"):out[m]=agg([r for r in s if r.get("market")==m])
    return out


def update_live_state(*, bq_client, storage_client, bucket_name, evidence_rows, now, log_func=print):
    contract=load_contract(storage_client=storage_client,bucket_name=bucket_name)
    overlay_registry=_load_overlay_registry(storage_client,bucket_name)
    enriched,miner_diag=augment_live_miner_votes(evidence_rows,bq_client=bq_client,storage_client=storage_client,bucket_name=bucket_name,overlay_registry=overlay_registry,log_func=log_func)
    rows=apply_model_authority_rows(enriched,contract,storage_client=storage_client,bucket_name=bucket_name,overlay_registry=overlay_registry)
    capture=_capture(storage_client,bucket_name,rows,now,contract.get("contract_sha256")); settlement=_settle(bq_client,storage_client,bucket_name,now); perf=_performance(storage_client,bucket_name)
    actions=("STRONG BET","BET","CANDIDATE","PASS","MODEL ONLY","EDGE — NO EXEC QUOTE","NO MARKET"); counts={a:sum(1 for r in rows if r.get("action")==a) for a in actions}
    state={
        "status":"NFL_PRODUCTION_BETTING_V3_3_SOURCE_NEUTRAL_SYSTEM_CONFIRMATION_ACTIVE","source_tag":SOURCE_TAG,"betting_policy_source_tag":BETTING_POLICY_SOURCE_TAG,
        "generated_at_utc":pd.to_datetime(now,utc=True).isoformat(),"contract":contract,"production_betting_policy":production_betting_policy_contract(),"live_rows":rows,
        "action_counts":counts,"capture":capture,"settlement":settlement,"live_performance":perf,
        "betting_authority":"NFL_PRODUCTION_BETTING_V3_2_CORE_CANDIDATE_CONFIRMED",
        "shadow_evidence_role":"CORE_CREATES_CANDIDATE__QUALIFIED_SYSTEMS_CONFIRM_OR_CONFLICT__STAT_MARKET_DIAGNOSTIC",
        "live_miner_diagnostics":miner_diag,
        "automatic_execution":False,
    }
    state["current_uri"]=_write_json(storage_client,bucket_name,CURRENT_OBJECT,state)
    log_func("[NFL-PROD-BET-V3-LIVE] "+json.dumps({"status":state["status"],"action_counts":counts,"capture":capture,"settlement":settlement,"live_performance":perf,"model_contract_sha256":contract.get("contract_sha256"),"betting_policy_source_tag":BETTING_POLICY_SOURCE_TAG},sort_keys=True,default=str))
    return state

def read_dashboard_state(*, storage_client, bucket_name="sharp-models"):
    return {"meta": _read_json(storage_client, bucket_name, CONTRACT_OBJECT), "current": _read_json(storage_client, bucket_name, CURRENT_OBJECT), "status": "READY"}


def _self_test():
    assert set(DISCOVERY_SEASONS).isdisjoint(CONFIRMATION_SEASONS)
    assert 2026 not in DISCOVERY_SEASONS + CONFIRMATION_SEASONS
    c={"markets":{"SPREADS":{},"H2H":{},"TOTALS":{}}}
    base={"market":"SPREADS","action":"MODEL ONLY","raw_model_edge":4.0,"model_direction":1,"home_team":"HOME","away_team":"AWAY","selected":"HOME","selected_price":-110,"current_home_spread":-2.0,"core_cover_probability":0.60,"edge_votes":[],"system_trigger_votes":[],"stat_selector_support":[],"market_move_toward_model":0.0}
    rr=apply_model_authority_rows([base],c,overlay_registry={})
    assert rr[0]["action"]=="CANDIDATE" and rr[0]["selected"]=="HOME" and rr[0]["core_candidate_qualifies"]
    reg={"S1":{"system_id":"S1","family_id":"F1","source":"MINER","qualified_current":True},"F1":{"system_id":"S1","family_id":"F1","source":"MINER","qualified_current":True},"S2":{"system_id":"S2","family_id":"F2","source":"PATHI","qualified_current":True},"F2":{"system_id":"S2","family_id":"F2","source":"PATHI","qualified_current":True}}
    one={**base,"system_trigger_votes":[{"system_id":"S1","family":"F1","direction":1}]}
    ro=apply_model_authority_rows([one],c,overlay_registry=reg); assert ro[0]["action"]=="BET" and ro[0]["qualified_system_support_count"]==1
    one_conflict={**base,"system_trigger_votes":[{"system_id":"S1","family":"F1","direction":-1}]}
    rc=apply_model_authority_rows([one_conflict],c,overlay_registry=reg); assert rc[0]["action"]=="CANDIDATE" and rc[0]["qualified_system_conflict_count"]==1
    strong={**base,"system_trigger_votes":[{"system_id":"S1","family":"F1","direction":1},{"system_id":"S2","family":"F2","direction":1}]}
    rs=apply_model_authority_rows([strong],c,overlay_registry=reg); assert rs[0]["action"]=="STRONG BET" and rs[0]["qualified_system_support_count"]==2
    veto={**base,"system_trigger_votes":[{"system_id":"S1","family":"F1","direction":-1},{"system_id":"S2","family":"F2","direction":-1}]}
    rv=apply_model_authority_rows([veto],c,overlay_registry=reg); assert rv[0]["action"]=="PASS" and rv[0]["qualified_system_conflict_count"]==2
    preg={"PATHI_KEY_7_HOOK":{"system_id":"Pathi_FB_Dog_Hook_Above_7","family_id":"PATHI_KEY_7_HOOK","source":"PATHI","qualified_current":True,"name":"Dog Hook Above 7"}}
    pr=apply_model_authority_rows([{**base,"current_home_spread":-7.5,"open_home_spread":-7.0,"selected":"AWAY","model_direction":-1}],c,overlay_registry=preg); assert pr[0]["action"]=="BET" and pr[0]["qualified_system_support_count"]==1
    # Live Miner: frozen qualification + 2026 trigger context only.
    mreg={
      "NFL_SPREAD_ROAD_DOG_VS_WEAK_OPP":{"system_id":"NFL_SPREAD_ROAD_DOG_VS_WEAK_OPP","family_id":"NFL_SPREAD_ROAD_DOG_VS_WEAK_OPP","source":"MINER","qualified_current":True,"name":"Road Dog vs Weak Opponent","market":"SPREADS","direction":"PLAY_ON","representative_conditions":["OPP_OFF_SU_LOSS","ROAD_DOG","OPP_WINPCT_LE_500"]},
      "NFL_SPREADS_PLAY_ON_PRIOR_ATS_MAGNITUDE__ROLE_CHANGE":{"system_id":"NFL_SPREADS_PLAY_ON_PRIOR_ATS_MAGNITUDE__ROLE_CHANGE","family_id":"NFL_SPREADS_PLAY_ON_PRIOR_ATS_MAGNITUDE__ROLE_CHANGE","source":"MINER","qualified_current":True,"name":"Prior ATS Magnitude Role Change","market":"SPREADS","direction":"PLAY_ON","representative_conditions":["OFF_ATS_COVER_7_PLUS","ROLE_FLIP_FAVORITE_TO_DOG","SU_SEQ2_LW"]},
    }
    hist=pd.DataFrame([
      {"Season":2026,"Season_Stage":"REGULAR","Game_Date":"2026-09-03","Source_Name":"T","Source_Game_ID":"1A","Team_Norm":"AWAY","Opponent_Norm":"X","Team_Score":14,"Opponent_Score":28,"Opening_Spread":3.0,"Is_Division_Game":0,"Turnover_Margin":1.0},
      {"Season":2026,"Season_Stage":"REGULAR","Game_Date":"2026-09-10","Source_Name":"T","Source_Game_ID":"1B","Team_Norm":"AWAY","Opponent_Norm":"X2","Team_Score":21,"Opponent_Score":24,"Opening_Spread":2.0,"Is_Division_Game":0,"Turnover_Margin":1.0},
      {"Season":2026,"Season_Stage":"REGULAR","Game_Date":"2026-09-17","Source_Name":"T","Source_Game_ID":"1C","Team_Norm":"AWAY","Opponent_Norm":"Y","Team_Score":31,"Opponent_Score":17,"Opening_Spread":-2.0,"Is_Division_Game":1,"Turnover_Margin":1.0},
      {"Season":2026,"Season_Stage":"REGULAR","Game_Date":"2026-09-17","Source_Name":"T","Source_Game_ID":"2A","Team_Norm":"HOME","Opponent_Norm":"Z","Team_Score":17,"Opponent_Score":24,"Opening_Spread":-3.0,"Is_Division_Game":1,"Turnover_Margin":-1.0},
      {"Season":2026,"Season_Stage":"REGULAR","Game_Date":"2026-09-17","Source_Name":"T","Source_Game_ID":"3A","Team_Norm":"OPPDOG","Opponent_Norm":"Q","Team_Score":20,"Opponent_Score":17,"Opening_Spread":4.0,"Is_Division_Game":0,"Turnover_Margin":0.0},
    ])
    mr={**base,"home_team":"HOME","away_team":"AWAY","open_home_spread":-3.5,"current_home_spread":-3.5,"selected":"AWAY","model_direction":-1}
    ma,md=augment_live_miner_votes([mr],overlay_registry=mreg,history_df=hist,log_func=lambda *_:None)
    # AWAY is road dog; HOME lost prior; HOME is <=.500 => Road Dog family fires.
    assert "NFL_SPREAD_ROAD_DOG_VS_WEAK_OPP" in ma[0]["live_miner_trigger_families"] and md["qualified_families_loaded"]==2
    # Live atom parity: exact System Lab semantics for the six qualified PT systems.
    hh=_miner_histories(hist)
    assert _miner_condition("LAST_GAME_FAVORITE",side_dir=-1,side_team="AWAY",opp_team="HOME",side_open_spread=3.5,histories=hh) is True
    assert _miner_condition("LAST_GAME_DIVISION",side_dir=-1,side_team="AWAY",opp_team="HOME",side_open_spread=3.5,histories=hh) is True
    assert _miner_condition("DEFENSE_IMPROVING_2",side_dir=-1,side_team="AWAY",opp_team="HOME",side_open_spread=3.5,histories=hh) is True
    assert _miner_condition("TURNOVER_POS_LAST3",side_dir=-1,side_team="AWAY",opp_team="HOME",side_open_spread=3.5,histories=hh) is True
    assert _miner_condition("OPP_LAST_GAME_DOG",side_dir=1,side_team="HOME",opp_team="OPPDOG",side_open_spread=-3.5,histories=hh) is True
    ss_hist={"winless":[{"margin":-7.0,"opening_spread":3.0,"ats_margin":-4.0},{"margin":-3.0,"opening_spread":7.0,"ats_margin":-1.0},{"margin":-10.0,"opening_spread":4.0,"ats_margin":-6.0},{"margin":-2.0,"opening_spread":6.0,"ats_margin":-1.0}],"oppwin":[{"margin":7.0,"opening_spread":-3.0,"ats_margin":4.0}]}
    assert _miner_condition("SU_AND_ATS_WINLESS_AFTER_4_PLUS",side_dir=-1,side_team="WINLESS",opp_team="OPPWIN",side_open_spread=9.0,histories=ss_hist) is True
    assert _miner_condition("TEAM_GAME_5",side_dir=-1,side_team="WINLESS",opp_team="OPPWIN",side_open_spread=9.0,histories=ss_hist) is True
    assert _miner_condition("DOG_9_PLUS",side_dir=-1,side_team="WINLESS",opp_team="OPPWIN",side_open_spread=9.0,histories=ss_hist) is True
    assert _miner_condition("OPP_HAS_SU_WIN",side_dir=-1,side_team="WINLESS",opp_team="OPPWIN",side_open_spread=9.0,histories=ss_hist) is True
    mfinal=apply_model_authority_rows(ma,c,overlay_registry=mreg); assert mfinal[0]["action"] in {"BET","STRONG BET"}
    # Qualified PT rule using the previously unsupported live atoms must now be evaluable.
    parity_reg={
      "PTPAR":{"system_id":"PTPAR","family_id":"NFL_SPREAD_EXTERNAL_RATINGS_FAMILY","source":"PT","qualified_current":True,
               "name":"PT parity","market":"SPREADS","direction":"PLAY_ON",
               "representative_conditions":["PTSP_LINEELO_RECOMMENDS_SIDE_2PLUS","OFF_ATS_COVER_7_PLUS","LAST_GAME_FAVORITE"]}
    }
    parity_ctx={"PAIRP":{"home":{"PTSP_EDGE__LINEELO":-3.0},"away":{"PTSP_EDGE__LINEELO":3.0}}}
    parity_row={**base,"prediction_pair_id":"PAIRP","home_team":"HOME","away_team":"AWAY","open_home_spread":-3.5,"selected":"AWAY","model_direction":-1}
    px,parity_diag=augment_live_miner_votes([parity_row],overlay_registry=parity_reg,history_df=hist,pt_context_by_game=parity_ctx,log_func=lambda *_:None)
    assert "PTPAR" not in parity_diag["not_evaluable_families"] and "PTPAR" in px[0]["live_miner_trigger_families"]
    # Unknown atoms must fail closed, never fabricate a vote.
    badreg={"F":{"system_id":"F","family_id":"F","source":"MINER","qualified_current":True,"name":"F","market":"SPREADS","direction":"PLAY_ON","representative_conditions":["UNSUPPORTED_ATOM"]}}
    bad,bd=augment_live_miner_votes([mr],overlay_registry=badreg,history_df=hist,log_func=lambda *_:None); assert not bad[0].get("live_miner_trigger_families") and "F" in bd["not_evaluable_families"]
    # PT-derived systems are source-neutral at the system gate, while all PT variants
    # collapse to one external-ratings family vote. Two agreeing PT systems = one vote;
    # opposing PT systems in that family abstain as an internal conflict.
    ptreg={
      "PT1":{"system_id":"PT1","family_id":"NFL_SPREAD_EXTERNAL_RATINGS_FAMILY","source":"PT","qualified_current":True,"name":"PT one","market":"SPREADS","direction":"PLAY_ON","representative_conditions":["PTSP_CLUSTER_CONSENSUS_EDGE_2PLUS"]},
      "PT2":{"system_id":"PT2","family_id":"NFL_SPREAD_EXTERNAL_RATINGS_FAMILY","source":"PT","qualified_current":True,"name":"PT two","market":"SPREADS","direction":"PLAY_ON","representative_conditions":["PTSP_CLUSTER_CONSENSUS_70_EDGE2"]},
    }
    ptctx={"PAIR":{"home":{"PTSP_CONSENSUS_EDGE":3.0,"PTSP_AGREE_FRAC":.80,"PTSP_CLUSTER_STD":2.0,"PTSP_CLUSTER_COUNT":10},"away":{"PTSP_CONSENSUS_EDGE":-3.0,"PTSP_AGREE_FRAC":.20,"PTSP_CLUSTER_STD":2.0,"PTSP_CLUSTER_COUNT":10}}}
    ptr={**base,"prediction_pair_id":"PAIR","open_home_spread":-3.0,"selected":"HOME","model_direction":1}
    pa,pdmeta=augment_live_miner_votes([ptr],overlay_registry=ptreg,history_df=hist,pt_context_by_game=ptctx,log_func=lambda *_:None)
    po=_qualified_system_overlay(pa[0],ptreg); assert len(po["support_families"])==1 and po["support_families"]==["NFL_SPREAD_EXTERNAL_RATINGS_FAMILY"]
    assert pdmeta.get("pt_family_vote_cap")==1 and pdmeta.get("source_neutral") is True
    ptreg_conf={**ptreg,"PT2":{**ptreg["PT2"],"direction":"FADE"}}
    pc,_=augment_live_miner_votes([ptr],overlay_registry=ptreg_conf,history_df=hist,pt_context_by_game=ptctx,log_func=lambda *_:None)
    pco=_qualified_system_overlay(pc[0],ptreg_conf); assert not pco["support_families"] and "NFL_SPREAD_EXTERNAL_RATINGS_FAMILY" in pco["internal_conflict_families"]
    low={**base,"core_cover_probability":0.56}; rl=apply_model_authority_rows([low],c,overlay_registry={}); assert rl[0]["action"]=="PASS"
    h=apply_model_authority_rows([{"market":"H2H","action":"MODEL ONLY","raw_model_edge":0.08,"model_direction":1}],c,overlay_registry={}); assert h[0]["action"]=="MODEL ONLY"
    t=apply_model_authority_rows([{"market":"TOTALS","action":"MODEL ONLY","raw_model_edge":4.0,"model_direction":-1}],c,overlay_registry={}); assert t[0]["action"]=="MODEL ONLY"
    return {"status":"PASS","source_tag":SOURCE_TAG,"betting_policy_source_tag":BETTING_POLICY_SOURCE_TAG,"spread":"CORE_0575_PLUS_2PCT_EV_CREATES_CANDIDATE","bet":"1_QUALIFIED_INDEPENDENT_SYSTEM_SUPPORT","strong_bet":"2PLUS_QUALIFIED_INDEPENDENT_SYSTEM_SUPPORT","single_conflict":"CANDIDATE_NO_WAGER","veto":"2PLUS_NET_QUALIFIED_INDEPENDENT_SYSTEM_CONFLICT","totals":"MODEL_ONLY","h2h":"MODEL_ONLY","automatic_execution":False,"source_neutral_system_authority":True,"pt_family_vote_cap":1,"pt_model_weight":0.0,"live_system_atom_parity_v331":True,"season_record_state_live_parity":True,"season_record_state_family_vote_cap":1}


if __name__ == "__main__":
    print(json.dumps(_self_test(), sort_keys=True))
