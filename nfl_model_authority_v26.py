"""NFL Model Authority V2.6 — frozen production-model betting policy.

The frozen NFL production model is the only betting authority. CORE/STAT/SYSTEM/
MARKET research remains evidence/attribution and cannot create, reverse, or
escalate a wager.

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
import json
import math
from typing import Any

import numpy as np
import pandas as pd
try:
    from google.api_core.exceptions import PreconditionFailed
    from google.cloud import bigquery
except ModuleNotFoundError:
    class PreconditionFailed(Exception):
        pass
    bigquery = None

import nfl_production_v1 as prod
import nfl_betting_engine_v1 as benchmark

SOURCE_TAG = "nfl-model-authority-v2.6-frozen-model-bet-policy-20261003"
ENGINE_VERSION = "NFL_MODEL_AUTHORITY_V2_6"
EXPECTED_MARKET_ADAPTER_TAG = "nfl-betting-engine-v1.0-unified-decision-20261002"
PREFIX = "production/nfl/v1/model_authority_v26"
CONTRACT_OBJECT = f"{PREFIX}/current_contract.json"
CURRENT_OBJECT = f"{PREFIX}/current_state.json"
EVENT_PREFIX = f"{PREFIX}/events"
SETTLEMENT_PREFIX = f"{PREFIX}/settlements"
PROJECT = "sharplogger"
DATASET = "sharp_data"
PAIRED_SETTLEMENT_TABLE = f"{PROJECT}.{DATASET}.nfl_production_v1_paired_settlements"

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


def _num(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _norm(x):
    # Use the exact live market/team normalization contract already proven by the
    # production quote join. This keeps the authoritative MODEL -> ACTION path
    # independent from the V2.5 shadow resolver while preserving quote parity.
    return benchmark._norm(x)


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


def _authoritative_consensus_market(client, now) -> dict:
    """Production quote consensus with no fabricated execution prices.

    The legacy benchmark helper intentionally substitutes -110 when spread/total
    odds are missing. That is useful for retrospective modeling but is not an
    acceptable live execution gate. This V2.6 path preserves the same line/value
    consensus while leaving an unobserved price as NaN so action fails closed as
    EDGE — NO EXEC QUOTE.
    """
    if getattr(benchmark, "SOURCE_TAG", "") != EXPECTED_MARKET_ADAPTER_TAG:
        raise RuntimeError("[NFL-MODEL-AUTH-V26-HOLD] MARKET_ADAPTER_STALE_OR_MIXED")
    d = benchmark._fetch_market_rows(client, now)
    out = {}
    if d is None or d.empty:
        return out
    d = d.sort_values("snapshot_ts")
    first = d.drop_duplicates(["game_start","home_key","away_key","market_norm","outcome_key","bookmaker"], keep="first")
    last = d.drop_duplicates(["game_start","home_key","away_key","market_norm","outcome_key","bookmaker"], keep="last")
    for (gs, hk, ak), _ in d.groupby(["game_start","home_key","away_key"], sort=False):
        rec = {"game_start": gs, "home_key": hk, "away_key": ak}
        for label, frame in (("open", first), ("current", last)):
            g = frame[(frame.game_start.eq(gs)) & (frame.home_key.eq(hk)) & (frame.away_key.eq(ak))]

            sp = g[g.market_norm.eq("spreads")].copy()
            vals = []
            for _, r in sp.iterrows():
                v = _num(r.value)
                if not math.isfinite(v):
                    continue
                if r.outcome_key == hk:
                    vals.append(v)
                elif r.outcome_key == ak:
                    vals.append(-v)
            if vals:
                line = float(np.median(vals))
                rec[label + "_home_spread"] = line
                hodd = pd.to_numeric(sp.loc[sp.outcome_key.eq(hk) & pd.to_numeric(sp.value, errors="coerce").sub(line).abs().lt(.011), "odds"], errors="coerce").dropna()
                aodd = pd.to_numeric(sp.loc[sp.outcome_key.eq(ak) & pd.to_numeric(sp.value, errors="coerce").add(line).abs().lt(.011), "odds"], errors="coerce").dropna()
                rec[label + "_home_spread_odds"] = float(hodd.max()) if len(hodd) else np.nan
                rec[label + "_away_spread_odds"] = float(aodd.max()) if len(aodd) else np.nan
                ts = pd.to_datetime(sp.snapshot_ts, utc=True, errors="coerce").dropna()
                rec[label + "_spread_snapshot_ts"] = ts.max().isoformat() if len(ts) else None

            to = g[g.market_norm.eq("totals")].copy()
            ov = to[to.outcome.astype(str).str.lower().str.contains("over", na=False)]
            uv = to[to.outcome.astype(str).str.lower().str.contains("under", na=False)]
            tv = pd.to_numeric(ov.value, errors="coerce").dropna()
            if len(tv):
                line = float(np.median(tv))
                rec[label + "_total"] = line
                oo = pd.to_numeric(ov.loc[pd.to_numeric(ov.value, errors="coerce").sub(line).abs().lt(.011), "odds"], errors="coerce").dropna()
                uo = pd.to_numeric(uv.loc[pd.to_numeric(uv.value, errors="coerce").sub(line).abs().lt(.011), "odds"], errors="coerce").dropna()
                rec[label + "_over_odds"] = float(oo.max()) if len(oo) else np.nan
                rec[label + "_under_odds"] = float(uo.max()) if len(uo) else np.nan
                ts = pd.to_datetime(to.snapshot_ts, utc=True, errors="coerce").dropna()
                rec[label + "_total_snapshot_ts"] = ts.max().isoformat() if len(ts) else None

            h2 = g[g.market_norm.eq("h2h")].copy()
            probs, hos, aos = [], [], []
            for _, bg in h2.groupby("bookmaker", sort=False):
                ho = pd.to_numeric(bg.loc[bg.outcome_key.eq(hk), "odds"], errors="coerce").dropna()
                ao = pd.to_numeric(bg.loc[bg.outcome_key.eq(ak), "odds"], errors="coerce").dropna()
                if len(ho):
                    hos.extend(ho.tolist())
                if len(ao):
                    aos.extend(ao.tolist())
                if len(ho) and len(ao):
                    ih = benchmark._american_implied(ho.iloc[-1])
                    ia = benchmark._american_implied(ao.iloc[-1])
                    if math.isfinite(ih) and math.isfinite(ia) and ih + ia > 0:
                        probs.append(ih / (ih + ia))
            if probs:
                rec[label + "_home_novig_probability"] = float(np.median(probs))
            if hos:
                rec[label + "_home_ml"] = float(max(hos))
            if aos:
                rec[label + "_away_ml"] = float(max(aos))
            if not h2.empty:
                ts = pd.to_datetime(h2.snapshot_ts, utc=True, errors="coerce").dropna()
                rec[label + "_h2h_snapshot_ts"] = ts.max().isoformat() if len(ts) else None
        out[(pd.Timestamp(gs).round("s").isoformat(), hk, ak)] = rec
    return out


def _authoritative_market_base_row(p: dict, market: str, q: dict) -> dict | None:
    """Build MODEL-vs-current-market rows without loading any research resolver.

    This intentionally duplicates only the minimal quote-orientation logic needed
    for the frozen production model. No SYSTEM/STAT/V2.5 state is consulted here.
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
        odds = _num(q.get("current_home_spread_odds" if direction >= 0 else "current_away_spread_odds")) if direction else np.nan
        return {
            "model_direction": direction,
            "raw_model_edge": abs(signed),
            "market_move_toward_model": -direction * (line - op) if direction and math.isfinite(op) else np.nan,
            "selected": selected,
            "market_value": line if direction >= 0 else -line,
            "model_value": fair * direction if direction else fair,
            "selected_price": odds,
            "quote_timestamp": q.get("current_spread_snapshot_ts"),
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
        odds = _num(q.get("current_over_odds" if direction >= 0 else "current_under_odds")) if direction else np.nan
        return {
            "model_direction": direction,
            "raw_model_edge": abs(signed),
            "market_move_toward_model": direction * (line - op) if direction and math.isfinite(op) else np.nan,
            "selected": selected,
            "market_value": line,
            "model_value": fair,
            "selected_price": odds,
            "quote_timestamp": q.get("current_total_snapshot_ts"),
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
        odds = _num(q.get("current_home_ml" if direction >= 0 else "current_away_ml")) if direction else np.nan
        return {
            "model_direction": direction,
            "raw_model_edge": abs(signed),
            "market_move_toward_model": direction * (ref - op) if direction and math.isfinite(op) else np.nan,
            "selected": selected,
            "market_value": ref if direction >= 0 else 1.0 - ref,
            "model_value": fair if direction >= 0 else 1.0 - fair,
            "selected_price": odds,
            "quote_timestamp": q.get("current_h2h_snapshot_ts"),
        }
    raise ValueError(market)


def build_authoritative_market_rows(*, bq_client, prediction_rows, now) -> list[dict]:
    """Build the live MODEL -> ACTION input rows from production predictions + quotes.

    Crucially, this function does not load Edge Authority V2.5 or any system/stat
    artifact. If shadow research is unavailable, the production model can still
    produce its own gated action.
    """
    quotes = _authoritative_consensus_market(bq_client, now)
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
            if base is None:
                out.append({**common, "action": "NO MARKET", "decision": "NO MARKET"})
            else:
                qt = pd.to_datetime(base.get("quote_timestamp"), utc=True, errors="coerce")
                base["quote_age_minutes"] = None if pd.isna(qt) else max(0.0, (pd.to_datetime(now, utc=True) - qt).total_seconds() / 60.0)
                out.append({**common, **base, "action": "MODEL ONLY", "decision": "MODEL_ONLY_PRE_POLICY"})
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
        "edge_sources", "edge_votes", "system_labels", "stat_selector_support",
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
    votes = r.get("edge_votes") or []
    system_dirs = [int(v.get("direction") or 0) for v in votes if str(v.get("label") or "").upper() not in {"FAIR_VALUE", "MARKET_CONFIRMATION"} and not str(v.get("label") or "").upper().startswith("STAT ")]
    if not system_dirs:
        system_state = "NEUTRAL"
    elif all(v == model_dir for v in system_dirs):
        system_state = "AGREE"
    elif all(v == -model_dir for v in system_dirs):
        system_state = "CONFLICT"
    else:
        system_state = "MIXED"
    statq = [x for x in (r.get("stat_selector_support") or []) if x.get("qualifies")]
    stat_state = "AGREE" if statq else "NEUTRAL"
    mv = _num(r.get("market_move_toward_model"))
    market_state = "AGREE" if math.isfinite(mv) and mv > 1e-9 else "CONFLICT" if math.isfinite(mv) and mv < -1e-9 else "NEUTRAL"
    return {"stat_state": stat_state, "system_state": system_state, "market_state": market_state}


def apply_model_authority_rows(rows, contract):
    out = []
    for src in rows or []:
        r = dict(src)
        market = str(r.get("market") or "").upper()
        mc = (contract.get("markets") or {}).get(market) or {}
        edge = _num(r.get("raw_model_edge"))
        th = _num(mc.get("threshold"))
        price = _num(r.get("selected_price"))
        action = "MODEL ONLY"
        reason = str(mc.get("status") or "MODEL_AUTHORITY_CLOSED")
        live_ev = np.nan
        if r.get("action") == "NO MARKET" or not math.isfinite(edge):
            action = "NO MARKET"
            reason = "CURRENT_MARKET_OR_MODEL_VALUE_UNAVAILABLE"
        elif not bool(mc.get("production_authority")) or not math.isfinite(th):
            action = "MODEL ONLY"
            reason = str(mc.get("status") or "MODEL_AUTHORITY_NOT_VALIDATED")
        elif edge < th:
            action = "PASS"
            reason = "MODEL_EDGE_BELOW_FROZEN_THRESHOLD"
        elif not _price_ok(market, price):
            action = "EDGE — NO EXEC QUOTE"
            reason = "MODEL_EDGE_QUALIFIES_BUT_EXECUTION_PRICE_UNAVAILABLE_OR_TOO_EXPENSIVE"
        else:
            if market == "H2H":
                live_ev = _american_ev(r.get("model_value"), price)
                if not math.isfinite(live_ev) or live_ev <= MIN_H2H_LIVE_EV:
                    action = "PASS"
                    reason = "H2H_MODEL_EV_NOT_POSITIVE_AT_CURRENT_PRICE"
                else:
                    action = "BET"
                    reason = "FROZEN_PRODUCTION_MODEL_EDGE_AND_PRICE_GATE_PASS"
            else:
                action = "BET"
                reason = "FROZEN_PRODUCTION_MODEL_EDGE_AND_PRICE_GATE_PASS"
        shadow = _shadow_states(r)
        shadow_action = r.get("shadow_edge_action")
        shadow_decision = r.get("shadow_edge_decision")
        r.update({
            "action": action,
            "decision": "MODEL_BET" if action == "BET" else "MODEL_PASS" if action == "PASS" else action,
            "decision_reason": reason,
            "betting_authority": "FROZEN_PRODUCTION_MODEL",
            "authority_source_tag": SOURCE_TAG,
            "model_policy_threshold": None if not math.isfinite(th) else float(th),
            "model_policy_status": mc.get("status"),
            "model_production_authority": bool(mc.get("production_authority")),
            "model_live_ev": None if not math.isfinite(live_ev) else float(live_ev),
            "model_confidence": "HIGH_MODEL_EDGE" if math.isfinite(th) and th > 0 and edge >= 1.5 * th else "QUALIFIED_MODEL_EDGE" if action == "BET" else "BELOW_BET_GATE",
            "shadow_edge_action": shadow_action,
            "shadow_edge_decision": shadow_decision,
            "shadow_evidence_only": True,
            **shadow,
            "betting_decision_authority": bool(action == "BET"),
            "automatic_execution": False,
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
    inserted = 0; existing = 0
    for r in rows:
        if r.get("action") != "BET":
            continue
        rid = _sha({"contract": contract_sha, "prediction_pair_id": r.get("prediction_pair_id"), "market": r.get("market")})
        event = {**r, "model_bet_event_id": rid, "captured_at": pd.to_datetime(now, utc=True).isoformat(), "source_tag": SOURCE_TAG}
        if _write_immutable(sc, bucket, f"{EVENT_PREFIX}/{rid}.json", event):
            inserted += 1
        else:
            existing += 1
    return {"inserted": inserted, "existing": existing}


def _settlement_rows(client, pair_ids):
    if not pair_ids:
        return pd.DataFrame()
    q = f"SELECT * FROM `{PAIRED_SETTLEMENT_TABLE}` WHERE prediction_pair_id IN UNNEST(@ids)"
    cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", list(pair_ids))])
    return client.query(q, job_config=cfg).to_dataframe(create_bqstorage_client=False)


def _settle(client, sc, bucket, now):
    events = _list_json(sc, bucket, EVENT_PREFIX)
    settled = {str(x.get("model_bet_event_id")) for x in _list_json(sc, bucket, SETTLEMENT_PREFIX)}
    pending = [e for e in events if str(e.get("model_bet_event_id")) not in settled]
    if not pending:
        return {"pending_before": 0, "inserted": 0}
    s = _settlement_rows(client, {str(e.get("prediction_pair_id")) for e in pending})
    by = {str(r.prediction_pair_id): r for _, r in s.iterrows()}
    n = 0
    for e in pending:
        r = by.get(str(e.get("prediction_pair_id")))
        if r is None:
            continue
        market = str(e.get("market") or "").upper(); sel = str(e.get("selected") or "")
        result = "UNRESOLVED"; profit = np.nan; price = _num(e.get("selected_price")); win_profit = _american_profit_if_win(price)
        if market == "SPREADS":
            line = _num(e.get("market_value")); margin = _num(r.actual_margin); home_sel = _norm(sel) == _norm(e.get("home_team"))
            v = (margin + line) if home_sel else -(margin + line)
            result = "WIN" if v > 0 else "LOSS" if v < 0 else "PUSH"; profit = win_profit if result == "WIN" else -1.0 if result == "LOSS" else 0.0
        elif market == "TOTALS":
            line = _num(e.get("market_value")); actual = _num(r.actual_total); v = actual - line; v = -v if sel.upper() == "UNDER" else v
            result = "WIN" if v > 0 else "LOSS" if v < 0 else "PUSH"; profit = win_profit if result == "WIN" else -1.0 if result == "LOSS" else 0.0
        elif market == "H2H":
            home_win = _num(r.home_win_label) == 1; won = home_win if _norm(sel) == _norm(e.get("home_team")) else not home_win
            result = "WIN" if won else "LOSS"; profit = win_profit if won else -1.0
        payload = {"model_bet_event_id": e.get("model_bet_event_id"), "prediction_pair_id": e.get("prediction_pair_id"), "market": market, "selected": sel, "result": result, "profit_per_unit": None if not math.isfinite(_num(profit)) else float(profit), "settled_at": pd.to_datetime(now, utc=True).isoformat(), "source_tag": SOURCE_TAG}
        if _write_immutable(sc, bucket, f"{SETTLEMENT_PREFIX}/{e['model_bet_event_id']}.json", payload):
            n += 1
    return {"pending_before": len(pending), "inserted": n}


def _performance(sc, bucket):
    s = _list_json(sc, bucket, SETTLEMENT_PREFIX)
    def agg(x):
        w = sum(1 for r in x if r.get("result") == "WIN"); l = sum(1 for r in x if r.get("result") == "LOSS"); p = sum(1 for r in x if r.get("result") == "PUSH")
        profits = [_num(r.get("profit_per_unit")) for r in x if math.isfinite(_num(r.get("profit_per_unit")))]
        return {"n": len(x), "wins": w, "losses": l, "pushes": p, "hit_rate": round(w/(w+l),6) if w+l else None, "roi_per_unit": round(float(np.mean(profits)),6) if profits else None}
    out = {"ALL": agg(s)}
    for m in ("SPREADS", "H2H", "TOTALS"):
        out[m] = agg([r for r in s if r.get("market") == m])
    return out


def update_live_state(*, bq_client, storage_client, bucket_name, evidence_rows, now, log_func=print):
    contract = load_contract(storage_client=storage_client, bucket_name=bucket_name)
    rows = apply_model_authority_rows(evidence_rows, contract)
    capture = _capture(storage_client, bucket_name, rows, now, contract.get("contract_sha256"))
    settlement = _settle(bq_client, storage_client, bucket_name, now)
    perf = _performance(storage_client, bucket_name)
    actions = ("BET", "PASS", "MODEL ONLY", "EDGE — NO EXEC QUOTE", "NO MARKET")
    counts = {a: sum(1 for r in rows if r.get("action") == a) for a in actions}
    state = {
        "status": "NFL_MODEL_AUTHORITY_V2_6_LIVE_ACTIVE",
        "source_tag": SOURCE_TAG,
        "generated_at_utc": pd.to_datetime(now, utc=True).isoformat(),
        "contract": contract,
        "live_rows": rows,
        "action_counts": counts,
        "capture": capture,
        "settlement": settlement,
        "live_performance": perf,
        "betting_authority": "FROZEN_PRODUCTION_MODEL_ONLY",
        "shadow_evidence_role": "CORE_STAT_SYSTEM_MARKET_ATTRIBUTION_ONLY",
        "automatic_execution": False,
    }
    state["current_uri"] = _write_json(storage_client, bucket_name, CURRENT_OBJECT, state)
    log_func("[NFL-MODEL-AUTH-V26-LIVE] " + json.dumps({"status": state["status"], "action_counts": counts, "capture": capture, "settlement": settlement, "live_performance": perf, "contract_sha256": contract.get("contract_sha256")}, sort_keys=True, default=str))
    return state


def read_dashboard_state(*, storage_client, bucket_name="sharp-models"):
    return {"meta": _read_json(storage_client, bucket_name, CONTRACT_OBJECT), "current": _read_json(storage_client, bucket_name, CURRENT_OBJECT), "status": "READY"}


def _self_test():
    assert set(DISCOVERY_SEASONS).isdisjoint(CONFIRMATION_SEASONS)
    assert 2026 not in DISCOVERY_SEASONS + CONFIRMATION_SEASONS
    assert abs(SPREAD_TOTAL_BREAK_EVEN - 0.5238095238) < 1e-6
    c = {
        "markets": {"SPREADS": {"production_authority": True, "threshold": 3.0, "status":"MODEL_BET_AUTHORITY_FROZEN"}, "H2H": {"production_authority": False, "threshold": None, "status":"MODEL_ONLY"}, "TOTALS": {"production_authority": False, "threshold": None, "status":"MODEL_ONLY"}}
    }
    rr = apply_model_authority_rows([{"market":"SPREADS","action":"MODEL ONLY","raw_model_edge":4.0,"selected_price":-110,"model_direction":1,"edge_votes":[],"stat_selector_support":[],"market_move_toward_model":0.0}], c)
    assert rr[0]["action"] == "BET" and rr[0]["betting_authority"] == "FROZEN_PRODUCTION_MODEL"
    return {"status":"PASS","source_tag":SOURCE_TAG,"authority":"FROZEN_PRODUCTION_MODEL_ONLY"}


if __name__ == "__main__":
    print(json.dumps(_self_test(), sort_keys=True))
