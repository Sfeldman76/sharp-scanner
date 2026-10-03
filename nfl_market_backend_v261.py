"""NFL V2.6.1 canonical live-market backend adapter.

Contract
--------
* `utils.py` remains the only BigQuery market-reader/backend entry point.
* `sharp_moves_master` is the canonical current/raw market source.
* `moves_with_features_merged` plus Utils' timing builder is the canonical
  MARKET-RICH attribution source.
* This module does not issue betting decisions. It only normalizes the Utils
  outputs for the frozen NFL production model and dashboard.

There are deliberately no direct BigQuery queries in this module.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE_TAG = "nfl-market-backend-v2.6.1-utils-canonical-20261003"
RAW_MARKET_FALLBACK = "sharp_data.sharp_moves_master"
ENRICHED_MARKET_FALLBACK = "sharp_data.moves_with_features_merged"
DEFAULT_LOOKBACK_HOURS = 336
DEFAULT_LOOKAHEAD_DAYS = 8

RICH_FIELDS = (
    "Line_Move_30m", "Line_Move_60m", "Line_Move_120m",
    "Line_Move_Last30m", "Line_Move_Last60m", "Line_Move_From_Open",
    "Line_Move_Velocity_60m", "Line_Move_Velocity_120m",
    "Direction_Changes_Count", "Max_Line_Seen", "Min_Line_Seen",
    "Current_vs_Best_Line", "Current_vs_Worst_Line",
    "Sharp_Book_Move_30m", "Sharp_Book_Move_60m",
    "Sharp_Move_Before_Market", "Sharp_Lead_Time_Minutes",
    "Sharp_Soft_Divergence", "Sharp_Consensus_Direction",
    "Crossed_Key_3_Last60m", "Crossed_Key_7_Last60m",
    "Crossed_Key_10_Last60m", "Crossed_Key_14_Last60m",
    "Minutes_Since_Key_Cross", "Key_Cross_Confirmed_By_Sharp_Books",
    "Key_Cross_Reversed", "Key_Cross_Persistence",
)


def _num(x):
    try:
        z = float(x)
        return z if math.isfinite(z) else np.nan
    except Exception:
        return np.nan


def _norm(x):
    return " ".join(str(x or "").strip().lower().replace(".", " ").replace("_", " ").split())


def _first_col(df: pd.DataFrame, *names: str):
    cmap = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        hit = cmap.get(str(name).lower())
        if hit is not None:
            return hit
    return None


def _require_utils(bq_client):
    u = importlib.import_module("utils")
    module_path = Path(getattr(u, "__file__", "")).resolve()
    if not module_path.is_file():
        raise RuntimeError("[NFL-MARKET-BACKEND-V261-HOLD] LOCAL_UTILS_NOT_FOUND")
    required = ("read_recent_sharp_moves", "build_30min_line_timing_features")
    missing = [name for name in required if not hasattr(u, name)]
    if missing:
        raise RuntimeError("[NFL-MARKET-BACKEND-V261-HOLD] UTILS_API_MISSING " + str(missing))
    # Force the same authenticated client used by the production job. Utils is
    # the backend, while the caller owns credentials/lifecycle.
    if hasattr(u, "bq_client"):
        u.bq_client = bq_client
    return u, module_path


def _canonicalize(df: pd.DataFrame, now, lookahead_days=DEFAULT_LOOKAHEAD_DAYS) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    m = {
        "sport": _first_col(df, "Sport"),
        "market": _first_col(df, "Market"),
        "outcome": _first_col(df, "Outcome"),
        "value": _first_col(df, "Value"),
        "odds": _first_col(df, "Odds_Price", "Odds", "Price"),
        "book": _first_col(df, "Bookmaker", "Book", "Sportsbook"),
        "game_start": _first_col(df, "Game_Start", "Commence_Hour", "feat_Game_Start"),
        "snapshot": _first_col(df, "Snapshot_Timestamp", "snapshot_timestamp", "Observed_At", "Captured_At", "Time"),
        "home": _first_col(df, "Home_Team_Norm", "Home_Team", "Home"),
        "away": _first_col(df, "Away_Team_Norm", "Away_Team", "Away"),
        "game_key": _first_col(df, "Game_Key", "game_key"),
    }
    missing = [k for k in ("sport","market","outcome","value","odds","book","game_start","snapshot","home","away") if not m[k]]
    if missing:
        raise RuntimeError("[NFL-MARKET-BACKEND-V261-HOLD] UTILS_MARKET_SCHEMA_MISSING " + str(missing))
    d = pd.DataFrame({
        "sport": df[m["sport"]].astype(str),
        "market": df[m["market"]].astype(str),
        "outcome": df[m["outcome"]].astype(str),
        "value": pd.to_numeric(df[m["value"]], errors="coerce"),
        "odds": pd.to_numeric(df[m["odds"]], errors="coerce"),
        "bookmaker": df[m["book"]].astype(str),
        "game_start": pd.to_datetime(df[m["game_start"]], utc=True, errors="coerce"),
        "snapshot_ts": pd.to_datetime(df[m["snapshot"]], utc=True, errors="coerce"),
        "home_team": df[m["home"]].astype(str),
        "away_team": df[m["away"]].astype(str),
        "game_key": df[m["game_key"]].astype(str) if m["game_key"] else "",
    }, index=df.index)
    now_ts = pd.to_datetime(now, utc=True)
    d = d.loc[
        d.sport.str.upper().str.strip().eq("NFL")
        & d.game_start.notna() & d.snapshot_ts.notna()
        & d.game_start.gt(now_ts)
        & d.game_start.le(now_ts + pd.Timedelta(days=int(lookahead_days)))
        & d.snapshot_ts.lt(d.game_start)
    ].copy()
    if d.empty:
        return d
    d["source_index"] = d.index
    d["home_key"] = d.home_team.map(_norm)
    d["away_key"] = d.away_team.map(_norm)
    d["outcome_key"] = d.outcome.map(_norm)
    d["market_norm"] = d.market.astype(str).str.lower().str.strip()
    d["book_key"] = d.bookmaker.map(_norm)
    d["game_key_norm"] = d.game_key.map(_norm)
    return d.reset_index(drop=True)


def _load_frames(bq_client, now, lookback_hours=DEFAULT_LOOKBACK_HOURS):
    u, upath = _require_utils(bq_client)
    raw_table = str(getattr(u, "BQ_FULL_TABLE", RAW_MARKET_FALLBACK) or RAW_MARKET_FALLBACK)
    enriched_table = str(getattr(u, "DEFAULT_MOVES_VIEW", ENRICHED_MARKET_FALLBACK) or ENRICHED_MARKET_FALLBACK)
    raw = u.read_recent_sharp_moves(hours=int(lookback_hours), table=raw_table, pregame_only=True)
    enriched = u.read_recent_sharp_moves(hours=int(lookback_hours), table=enriched_table, pregame_only=True)
    if raw is None:
        raw = pd.DataFrame()
    if enriched is None:
        enriched = pd.DataFrame()
    return u, upath, raw, enriched, raw_table, enriched_table


def _best_price(df: pd.DataFrame):
    if df is None or df.empty:
        return np.nan, None
    z = df.copy()
    z["odds"] = pd.to_numeric(z.odds, errors="coerce")
    z = z.loc[z.odds.notna()]
    if z.empty:
        return np.nan, None
    i = z.odds.idxmax()
    return float(z.loc[i, "odds"]), str(z.loc[i, "bookmaker"])


def _american_implied(odds):
    o = _num(odds)
    if not math.isfinite(o) or o == 0:
        return np.nan
    return (-o) / ((-o) + 100.0) if o < 0 else 100.0 / (o + 100.0)


def _build_rich_map(u, enriched_raw: pd.DataFrame, enriched: pd.DataFrame) -> dict:
    if enriched.empty:
        return {}
    timing = pd.DataFrame()
    try:
        timing = u.build_30min_line_timing_features(
            enriched_raw,
            sharp_books=getattr(u, "SHARP_BOOKS", None),
        )
    except Exception:
        timing = pd.DataFrame()

    # Start from the latest enriched row per book/outcome so Utils-enriched fields
    # and Utils timing fields share one canonical game/market/outcome grain.
    latest = enriched.sort_values("snapshot_ts").drop_duplicates(
        ["game_start","home_key","away_key","market_norm","outcome_key","book_key"], keep="last"
    ).copy()

    # Pull already-materialized rich fields directly from moves_with_features_merged.
    source_cols = {}
    for c in RICH_FIELDS:
        hit = _first_col(enriched_raw, c)
        if hit:
            source_cols[c] = hit
    if source_cols:
        src_idx = latest["source_index"].tolist()
        copy = enriched_raw.loc[src_idx, list(source_cols.values())].copy().reset_index(drop=True)
        copy.columns = list(source_cols.keys())
        for c in copy.columns:
            latest[c] = pd.to_numeric(copy[c], errors="coerce").to_numpy()

    if timing is not None and not timing.empty:
        t = timing.copy()
        for c in ("Game_Key","Market","Outcome","Bookmaker"):
            if c not in t.columns:
                t = pd.DataFrame(); break
        if not t.empty:
            t["game_key_norm"] = t["Game_Key"].map(_norm)
            t["market_norm"] = t["Market"].astype(str).str.lower().str.strip()
            t["outcome_key"] = t["Outcome"].map(_norm)
            t["book_key"] = t["Bookmaker"].map(_norm)
            keep = ["game_key_norm","market_norm","outcome_key","book_key"] + [c for c in RICH_FIELDS if c in t.columns]
            t = t[keep].drop_duplicates(["game_key_norm","market_norm","outcome_key","book_key"], keep="last")
            latest = latest.merge(t, on=["game_key_norm","market_norm","outcome_key","book_key"], how="left", suffixes=("", "__utils_timing"))
            for c in RICH_FIELDS:
                tc = c + "__utils_timing"
                if tc in latest.columns:
                    base = pd.to_numeric(latest[c], errors="coerce") if c in latest.columns else pd.Series(np.nan,index=latest.index)
                    latest[c] = base.where(base.notna(), pd.to_numeric(latest[tc], errors="coerce"))

    rich = {}
    keys = ["game_start","home_key","away_key","market_norm","outcome_key"]
    for key, g in latest.groupby(keys, sort=False):
        rec = {
            "market_rich_source": "UTILS:moves_with_features_merged+build_30min_line_timing_features",
            "market_rich_books": int(g.book_key.nunique()),
        }
        current_value = pd.to_numeric(g.value, errors="coerce")
        rec["market_rich_current_value"] = float(current_value.median()) if current_value.notna().any() else np.nan
        for c in RICH_FIELDS:
            if c in g.columns:
                s = pd.to_numeric(g[c], errors="coerce")
                rec["market_rich_" + c] = float(s.median()) if s.notna().any() else np.nan
        mv60 = _num(rec.get("market_rich_Line_Move_60m"))
        cv = _num(rec.get("market_rich_current_value"))
        rec["market_rich_t60_value"] = cv - mv60 if math.isfinite(cv) and math.isfinite(mv60) else np.nan
        rich[(pd.Timestamp(key[0]).round("s").isoformat(), key[1], key[2], key[3], key[4])] = rec
    return rich


def build_market_snapshot(*, bq_client, now, lookback_hours=DEFAULT_LOOKBACK_HOURS, lookahead_days=DEFAULT_LOOKAHEAD_DAYS) -> dict:
    """Return canonical current quotes and MARKET-RICH evidence through Utils only."""
    u, upath, raw0, enriched0, raw_table, enriched_table = _load_frames(bq_client, now, lookback_hours)
    raw = _canonicalize(raw0, now, lookahead_days)
    enriched = _canonicalize(enriched0, now, lookahead_days)
    rich_map = _build_rich_map(u, enriched0, enriched)
    out = {}
    if raw.empty:
        return {
            "quotes": {}, "rich": rich_map,
            "meta": {
                "source_tag": SOURCE_TAG, "utils_path": str(upath),
                "raw_table": raw_table, "enriched_table": enriched_table,
                "raw_rows": 0, "enriched_rows": int(len(enriched)),
                "direct_bigquery_queries": 0,
            },
        }
    raw = raw.sort_values("snapshot_ts")
    first = raw.drop_duplicates(["game_start","home_key","away_key","market_norm","outcome_key","book_key"], keep="first")
    last = raw.drop_duplicates(["game_start","home_key","away_key","market_norm","outcome_key","book_key"], keep="last")
    for (gs, hk, ak), _ in raw.groupby(["game_start","home_key","away_key"], sort=False):
        rec = {
            "game_start": gs, "home_key": hk, "away_key": ak,
            "market_backend": "UTILS",
            "current_market_source": "sharp_moves_master",
            "market_rich_source": "moves_with_features_merged",
        }
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
                hs = sp.loc[sp.outcome_key.eq(hk) & pd.to_numeric(sp.value, errors="coerce").sub(line).abs().lt(.011)]
                aw = sp.loc[sp.outcome_key.eq(ak) & pd.to_numeric(sp.value, errors="coerce").add(line).abs().lt(.011)]
                hp, hb = _best_price(hs); ap, ab = _best_price(aw)
                rec[label + "_home_spread_odds"] = hp; rec[label + "_home_spread_book"] = hb
                rec[label + "_away_spread_odds"] = ap; rec[label + "_away_spread_book"] = ab
                ts = pd.to_datetime(sp.snapshot_ts, utc=True, errors="coerce").dropna()
                rec[label + "_spread_snapshot_ts"] = ts.max().isoformat() if len(ts) else None

            to = g[g.market_norm.eq("totals")].copy()
            ov = to[to.outcome.astype(str).str.lower().str.contains("over", na=False)]
            uv = to[to.outcome.astype(str).str.lower().str.contains("under", na=False)]
            tv = pd.to_numeric(ov.value, errors="coerce").dropna()
            if len(tv):
                line = float(np.median(tv)); rec[label + "_total"] = line
                oo = ov.loc[pd.to_numeric(ov.value, errors="coerce").sub(line).abs().lt(.011)]
                uu = uv.loc[pd.to_numeric(uv.value, errors="coerce").sub(line).abs().lt(.011)]
                op, ob = _best_price(oo); up, ub = _best_price(uu)
                rec[label + "_over_odds"] = op; rec[label + "_over_book"] = ob
                rec[label + "_under_odds"] = up; rec[label + "_under_book"] = ub
                ts = pd.to_datetime(to.snapshot_ts, utc=True, errors="coerce").dropna()
                rec[label + "_total_snapshot_ts"] = ts.max().isoformat() if len(ts) else None

            h2 = g[g.market_norm.eq("h2h")].copy(); probs = []
            for _, bg in h2.groupby("book_key", sort=False):
                ho = pd.to_numeric(bg.loc[bg.outcome_key.eq(hk), "odds"], errors="coerce").dropna()
                ao = pd.to_numeric(bg.loc[bg.outcome_key.eq(ak), "odds"], errors="coerce").dropna()
                if len(ho) and len(ao):
                    ih, ia = _american_implied(ho.iloc[-1]), _american_implied(ao.iloc[-1])
                    if math.isfinite(ih) and math.isfinite(ia) and ih + ia > 0:
                        probs.append(ih / (ih + ia))
            if probs:
                rec[label + "_home_novig_probability"] = float(np.median(probs))
            hp, hb = _best_price(h2.loc[h2.outcome_key.eq(hk)])
            ap, ab = _best_price(h2.loc[h2.outcome_key.eq(ak)])
            rec[label + "_home_ml"] = hp; rec[label + "_home_ml_book"] = hb
            rec[label + "_away_ml"] = ap; rec[label + "_away_ml_book"] = ab
            if not h2.empty:
                ts = pd.to_datetime(h2.snapshot_ts, utc=True, errors="coerce").dropna()
                rec[label + "_h2h_snapshot_ts"] = ts.max().isoformat() if len(ts) else None
        out[(pd.Timestamp(gs).round("s").isoformat(), hk, ak)] = rec

    return {
        "quotes": out,
        "rich": rich_map,
        "meta": {
            "source_tag": SOURCE_TAG,
            "utils_path": str(upath),
            "raw_table": raw_table,
            "enriched_table": enriched_table,
            "raw_rows": int(len(raw)),
            "enriched_rows": int(len(enriched)),
            "direct_bigquery_queries": 0,
        },
    }


def rich_for_selection(snapshot: dict, *, game_start, home_team, away_team, market: str, selected: str | None) -> dict:
    if not snapshot or not selected:
        return {}
    gs = pd.to_datetime(game_start, utc=True, errors="coerce")
    if pd.isna(gs):
        return {}
    hk, ak = _norm(home_team), _norm(away_team)
    m = str(market or "").lower().strip()
    if m == "spreads":
        outcome = hk if _norm(selected) == hk else ak
    elif m == "totals":
        outcome = _norm(selected)
    elif m == "h2h":
        outcome = hk if _norm(selected) == hk else ak
    else:
        return {}
    return dict((snapshot.get("rich") or {}).get((gs.round("s").isoformat(), hk, ak, m, outcome), {}) or {})


def preflight(*, bq_client) -> dict:
    u, upath = _require_utils(bq_client)
    return {
        "status": "NFL_MARKET_BACKEND_V261_PREFLIGHT_PASS",
        "source_tag": SOURCE_TAG,
        "utils_path": str(upath),
        "raw_table": str(getattr(u, "BQ_FULL_TABLE", RAW_MARKET_FALLBACK) or RAW_MARKET_FALLBACK),
        "enriched_table": str(getattr(u, "DEFAULT_MOVES_VIEW", ENRICHED_MARKET_FALLBACK) or ENRICHED_MARKET_FALLBACK),
        "read_api": "utils.read_recent_sharp_moves",
        "timing_api": "utils.build_30min_line_timing_features",
        "direct_bigquery_queries": 0,
    }


def self_test():
    assert SOURCE_TAG.startswith("nfl-market-backend-v2.6.1-")
    assert "sharp_moves_master" in RAW_MARKET_FALLBACK
    assert "moves_with_features_merged" in ENRICHED_MARKET_FALLBACK
    return {"status":"PASS","source_tag":SOURCE_TAG,"direct_bigquery_queries":0}


if __name__ == "__main__":
    print(self_test())
