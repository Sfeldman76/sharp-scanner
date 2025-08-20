import logging
import pandas as pd
import json
import psutil
import os
from google.cloud import bigquery 
from datetime import datetime
from config import SPORTS, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, API_KEY
from utils import (
    fetch_live_odds,
    #read_latest_snapshot_from_bigquery,
    #read_market_weights_from_bigquery,
    detect_sharp_moves,
    write_sharp_moves_to_master,
    #write_line_history_to_bigquery,
    write_snapshot_to_gcs_parquet,
    detect_market_leaders,
    apply_blended_sharp_score,
    load_model_from_gcs,
    write_to_bigquery,
    build_game_key,
    fetch_scores_and_backtest,
    read_recent_sharp_moves, 
    detect_cross_market_sharp_support, 
    assign_confidence_scores,
    summarize_consensus,
    compute_weighted_signal,
    compute_confidence,
    compute_line_hash,
    compute_and_write_market_weights,
    build_game_key,
    normalize_book_key, 
    #apply_sharp_scoring,
    compute_sharp_metrics,
    log_memory,
    calc_implied_prob,
    load_market_weights_from_bq,
    compute_sharp_magnitude_by_time_bucket,
    #apply_compute_sharp_metrics_rowwise,
    compute_all_sharp_metrics,
    implied_prob_to_point_move,
    get_trained_models,
    was_line_resistance_broken,
    compute_line_resistance_flag,
    add_line_and_crossmarket_features,
    compute_small_book_liquidity_features,
    normalize_book_name,
    hydrate_inverse_rows_from_snapshot,
    fallback_flip_inverse_rows,
    add_time_context_flags,
    update_power_ratings,
    normalize_sport,
    enrich_power_from_current_inplace
)

def detect_and_save_all_sports():
    """
    Orchestrates sharp detection per sport with:
      - pre-run power ratings update (fresh inputs),
      - single load of market weights (cached),
      - per-sport optional model loading (non-fatal if missing),
      - in-batch + cross-batch dedup handled by writer,
      - lease guard to avoid duplicate runs on restarts,
      - backtest write + conditional power-rating update (post-pass).
    """
    ratings_need_update = False
    run_started_utc = pd.Timestamp.utcnow().floor('min')

    # ---- 0) Global lease: if something restarted seconds later, bail quickly
    try:
        if not take_run_lease(resource="detect_all_sports", lease_ts=run_started_utc, ttl_minutes=5):
            logging.info("🔒 Another run is active (lease). Exiting.")
            return
    except Exception as e:
        logging.warning(f"⚠️ Lease check failed (continuing defensively): {e}")

    # ---- 0.1) Pre-run ratings update (moved to the beginning)
    try:
        logging.info("🟢 Pre-pass: updating power ratings BEFORE detection …")
        bq = bigquery.Client()
        pre_summary = update_power_ratings(bq)   # should be NaT-safe / idempotent
        logging.info(f"📈 Pre-pass ratings summary: {pre_summary}")
    except Exception as e:
        logging.error(f"❌ Pre-pass ratings update failed (continuing with best-known ratings): {e}", exc_info=True)

    # ---- 1) Cache shared resources
    try:
        market_weights = load_market_weights_from_bq()
        logging.info("✅ Loaded market weights")
    except Exception as e:
        logging.error(f"❌ Could not load market weights — aborting all: {e}", exc_info=True)
        return

    sports_order = ["NBA", "MLB", "WNBA", "CFL", "NFL", "NCAAF"]

    for sport_label in sports_order:
        try:
            sport_key = SPORTS[sport_label]  # e.g. "basketball_nba"
        except KeyError:
            logging.error(f"❌ Unknown sport label '{sport_label}' in SPORTS mapping; skipping.")
            continue

        try:
            logging.info(f"🔍 Running sharp detection for {sport_label}…")
            timestamp = pd.Timestamp.utcnow()

            # ---- 2) Fetch live odds
            try:
                current = fetch_live_odds(sport_key, API_KEY)
            except Exception as e:
                logging.error(f"❌ fetch_live_odds failed for {sport_label}: {e}", exc_info=True)
                continue

            n_games = len(current) if hasattr(current, "__len__") else (current.shape[0] if hasattr(current, "shape") else 0)
            logging.info(f"📥 Odds pulled: {n_games} games")
            if not current or n_games == 0:
                logging.warning(f"⚠️ No odds for {sport_label}, skipping…")
                continue

            # ---- 3) Load models (optional per market)
            models_to_try = ["spreads", "totals", "h2h"]
            trained_models = {}
            for m in models_to_try:
                try:
                    model = load_model_from_gcs(sport_label, m)
                    if model:
                        trained_models[m] = model
                except FileNotFoundError:
                    logging.warning(f"⚠️ Model missing in GCS for {sport_label}-{m}; will skip model scoring for that market.")
                except Exception as e:
                    logging.warning(f"⚠️ Model load error for {sport_label}-{m}: {e}")

            logging.info(f"🧠 Models loaded for {sport_label}: {list(trained_models.keys()) or 'none'}")

            # ---- 4) Detect sharp moves (detect computes Line_Hash)
            try:
                df_moves, df_snap_unused, df_audit = detect_sharp_moves(
                    current=current,
                    previous=None,
                    sport_key=sport_key,
                    sport_label=sport_label,
                    SHARP_BOOKS=SHARP_BOOKS,
                    REC_BOOKS=REC_BOOKS,
                    BOOKMAKER_REGIONS=BOOKMAKER_REGIONS,
                    trained_models=trained_models,   # may be partial or empty
                    weights=market_weights,
                )
            except Exception as e:
                logging.error(f"❌ detect_sharp_moves failed for {sport_label}: {e}", exc_info=True)
                continue

            # ---- 5) Persist sharp moves (trust the hash; writer dedups)
            try:
                if df_moves is not None and not df_moves.empty:
                    if 'Line_Hash' not in df_moves.columns:
                        logging.error(f"❌ Line_Hash missing for {sport_label}; not writing.")
                    else:
                        before = len(df_moves)
                        df_moves = df_moves.drop_duplicates(subset=['Line_Hash'], keep='last')
                        if len(df_moves) < before:
                            logging.info(f"🧽 In-batch dedup: removed {before - len(df_moves)} dupe rows for {sport_label}")
                        write_sharp_moves_to_master(df_moves, table='sharp_data.sharp_moves_master')
                else:
                    logging.info(f"ℹ️ No sharp moves produced for {sport_label}.")
            except Exception as e:
                logging.error(f"❌ Failed writing sharp moves for {sport_label}: {e}", exc_info=True)

            # ---- 6) Backtest + write finals
            try:
                backtest_days = 3
                df_backtest = fetch_scores_and_backtest(
                    sport_key=sport_key,
                    df_moves=df_moves,
                    days_back=backtest_days,
                    api_key=API_KEY,
                    trained_models=trained_models
                )
                if isinstance(df_backtest, tuple):
                    df_backtest = df_backtest[0]

                if df_backtest is not None and not df_backtest.empty:
                    write_to_bigquery(df_backtest, table="sharp_data.sharp_scores_full")
                    ratings_need_update = True
                else:
                    logging.warning(f"⚠️ No backtest rows for {sport_label} — skip write.")
            except Exception as e:
                logging.error(f"❌ Backtest failed for {sport_label}: {e}", exc_info=True)

        except Exception as e:
            logging.error(f"❌ Unhandled error during {sport_label} detection: {e}", exc_info=True)
    else:
        logging.info("ℹ️ No new finals written; skipping post-pass ratings update.")

