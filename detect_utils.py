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
    apply_sharp_scoring,
    compute_sharp_metrics,
    compute_sharp_prob_shift,
    log_memory,
    calc_implied_prob,
    load_market_weights_from_bq,
    compute_sharp_magnitude_by_time_bucket,
    apply_compute_sharp_metrics_rowwise
)

def detect_and_save_all_sports():
    for sport_label in ["NBA", "MLB", "WNBA", "CFL"]:
        try:
            sport_key = SPORTS[sport_label]
            logging.info(f"🔍 Running sharp detection for {sport_label}...")

            timestamp = pd.Timestamp.utcnow()
            current = fetch_live_odds(sport_key, API_KEY)
            logging.info(f"📥 Odds pulled: {len(current)} games")

            previous = read_latest_snapshot_from_bigquery()
            logging.info(f"📦 Previous snapshot loaded: {len(previous)} games")

            market_weights = load_market_weights_from_bq()

            trained_models = {
                market: load_model_from_gcs(sport_label, market)
                for market in ['spreads', 'totals', 'h2h']
            }
            
            trained_models = {k: v for k, v in trained_models.items() if v}
            logging.info(f"🧠 Models loaded for {sport_label}: {list(trained_models.keys())}")
            
            df_moves, df_snap_unused, df_audit = detect_sharp_moves(
                current=current,
                previous=previous,
                sport_key=sport_label,
                SHARP_BOOKS=SHARP_BOOKS,
                REC_BOOKS=REC_BOOKS,
                BOOKMAKER_REGIONS=BOOKMAKER_REGIONS,
                trained_models=trained_models,
                weights=market_weights  # ✅ this line is key
            )
            
           
            backtest_days = 3
            try:
                df_backtest = fetch_scores_and_backtest(
                    sport_key=sport_key,
                    df_moves=df_moves,
                    days_back=backtest_days,
                    api_key=API_KEY,
                    trained_models=trained_models
                )
                
                # ✅ Prevent failure if function ever accidentally returns a tuple
                if isinstance(df_backtest, tuple):
                    df_backtest = df_backtest[0]
                
                if df_backtest is not None and not df_backtest.empty:
                    write_to_bigquery(df_backtest, table="sharp_data.sharp_scores_full")
                else:
                    logging.warning(f"⚠️ No backtest rows returned for {sport_label} — skipping BigQuery write.")
            except Exception as e:
                logging.error(f"❌ Backtest failed for {sport_label}: {e}", exc_info=True)

            #if trained_models:
                #try:
                    #df_scored = apply_blended_sharp_score(df_moves.copy(), trained_models)
                    #if not df_scored.empty:
                        #df_moves = df_scored.copy()
                        #logging.info(f"✅ Scored {len(df_moves)} rows, now writing to master.")
                    #else:
                        #logging.info("ℹ️ No scored rows — model returned empty.")
                    # 🔁 Recompute Sharp_Prob_Shift with historical context
                    #recent_history = read_recent_sharp_moves(hours=72)
                    #recent_history = recent_history[[
                        #'Team_Key', 'Bookmaker', 'Snapshot_Timestamp', 'Model_Sharp_Win_Prob'
                    #]].dropna()
        
                    #combined = pd.concat([recent_history, df_moves], ignore_index=True)
                    #combined = compute_sharp_prob_shift(combined)
        
                    # ✅ Only keep current snapshot's rows
                    #current_ts = df_moves['Snapshot_Timestamp'].max()
                    #df_moves = combined[combined['Snapshot_Timestamp'] == current_ts].copy()                
                #except Exception as e:
                    #logging.error(f"❌ Model scoring failed for {sport_label}: {e}", exc_info=True)
            #else:
                #logging.info(f"ℹ️ No trained models found for {sport_label} — skipping scoring.")

            try:
                df_snap = pd.DataFrame([
                    {
                        'Game_ID': game.get('id'),
                        'Game': f"{game.get('home_team')} vs {game.get('away_team')}",
                        'Game_Start': pd.to_datetime(game.get("commence_time"), utc=True),
                        'Bookmaker': book.get('key'),
                        'Market': market.get('key'),
                        'Outcome': outcome.get('name'),
                        'Value': outcome.get('point') if market.get('key') != 'h2h' else outcome.get('price'),
                        'Limit': outcome.get('bet_limit'),
                        'Snapshot_Timestamp': timestamp
                    }
                    for game in current
                    for book in game.get('bookmakers', [])
                    for market in book.get('markets', [])
                    for outcome in market.get('outcomes', [])
                ])
                df_snap = build_game_key(df_snap)

                
                # Before writing sharp_moves
                if not df_moves.empty and df_moves.columns.size > 0:
                    df_moves["Line_Hash"] = df_moves.apply(compute_line_hash, axis=1)
                    df_moves = df_moves.drop_duplicates(subset=["Line_Hash"])
                else:
                    logging.warning(f"⚠️ df_moves is empty or has no columns — skipping Line_Hash computation for {sport_label}")
                    continue
                df_moves = df_moves.drop_duplicates(subset=["Line_Hash"])
                write_sharp_moves_to_master(df_moves)
               
                write_snapshot_to_gcs_parquet(current)

            except Exception as e:
                logging.error(f"❌ Failed to write snapshot or move data for {sport_label}: {e}", exc_info=True)

        except Exception as e:
            logging.error(f"❌ Unhandled error during {sport_label} detection: {e}", exc_info=True)
