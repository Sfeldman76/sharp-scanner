import logging
import pandas as pd
import json
import psutil
import os
from datetime import datetime
from config import SPORTS, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, API_KEY
from utils import (
    fetch_live_odds,
    read_latest_snapshot_from_bigquery,
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
    compute_sharp_prob_shift,
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
    get_opening_snapshot,
    add_time_context_flags,
    update_power_ratings,
    normalize_sport
)

def detect_and_save_all_sports():
    # 1) ✅ Load ratings first (your preference)
    try:
        logging.info("🟢 Pre-pass: updating power ratings BEFORE detection …")
        pre_summary = update_power_ratings()
        logging.info(f"📊 Pre-pass ratings summary: {pre_summary}")
    except Exception as e:
        logging.error(f"❌ Pre-pass ratings update failed: {e}", exc_info=True)

    ratings_need_update = False

    for sport_label in ["NBA", "MLB", "WNBA", "CFL", "NFL", "NCAAF"]:
        try:
            sport_key = SPORTS[sport_label]  # e.g., "basketball_nba" (⚠️ not the label)
            logging.info(f"🔍 Running sharp detection for {sport_label}…")

            timestamp = pd.Timestamp.utcnow()
            current = fetch_live_odds(sport_key, API_KEY)
            logging.info(f"📥 Odds pulled: {len(current)} games")

            if not current:
                logging.warning(f"⚠️ No odds for {sport_label}, skipping…")
                continue

            previous = read_latest_snapshot_from_bigquery()
            logging.info(f"📦 Previous snapshot loaded: {len(previous)} games")

            market_weights = load_market_weights_from_bq()
            trained_models = {
                m: load_model_from_gcs(sport_label, m) for m in ["spreads", "totals", "h2h"]
            }
            trained_models = {k: v for k, v in trained_models.items() if v}
            logging.info(f"🧠 Models loaded for {sport_label}: {list(trained_models.keys())}")

            # Use the API sport_key here
            df_moves, df_snap_unused, df_audit = detect_sharp_moves(
                current=current,
                previous=previous,
                sport_key=sport_key,
                SHARP_BOOKS=SHARP_BOOKS,
                REC_BOOKS=REC_BOOKS,
                BOOKMAKER_REGIONS=BOOKMAKER_REGIONS,
                trained_models=trained_models,
                weights=market_weights
            )

            # --- Backtest (writes to sharp_scores_full) ---
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

            # --- Save completed game scores to game_scores_final (ensure Sport present) ---
            try:
                completed_rows = build_completed_game_rows(current, sport_key)  # includes 'Sport'
                if completed_rows is not None and not completed_rows.empty:
                    write_to_bigquery(completed_rows, table="sharp_data.game_scores_final")
                    ratings_need_update = True
            except Exception as e:
                logging.error(f"❌ Writing game_scores_final failed for {sport_label}: {e}", exc_info=True)

            # … any snapshot/build_game_key work …

        except Exception as e:
            logging.error(f"❌ Unhandled error during {sport_label} detection: {e}", exc_info=True)

    # 2) ✅ Post-pass: only if we actually saved new finals anywhere
    if ratings_need_update:
        try:
            logging.info("🟢 Post-pass: updating power ratings AFTER detection …")
            post_summary = update_power_ratings()  # NaT-safe version
            logging.info(f"📈 Post-pass ratings summary: {post_summary}")
        except Exception as e:
            logging.error(f"❌ Post-pass ratings update failed: {e}", exc_info=True)
    else:
        logging.info("ℹ️ No new finals written; skipping post-pass ratings update.")

