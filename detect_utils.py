import logging
import pandas as pd
from datetime import datetime
from config import SPORTS, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, API_KEY
from utils import (
    fetch_live_odds,
    read_latest_snapshot_from_bigquery,
    read_market_weights_from_bigquery,
    detect_sharp_moves,
    write_sharp_moves_to_master,
    write_line_history_to_bigquery,
    write_snapshot_to_gcs_parquet,
    detect_market_leaders,
    apply_blended_sharp_score,
    load_model_from_gcs,
    write_to_bigquery,
    build_game_key
)

def detect_and_save_all_sports():
    for sport_label in ["NBA", "MLB", "WNBA", "CFL"]:
        sport_key = SPORTS[sport_label]
        logging.info(f"🔍 Running sharp detection for {sport_label}...")

        try:
            timestamp = pd.Timestamp.utcnow()
            live = fetch_live_odds(sport_key, API_KEY)
            logging.info(f"📥 Odds pulled: {len(live)} games")

            previous = read_latest_snapshot_from_bigquery()
            logging.info(f"📦 Previous snapshot loaded: {len(previous)} games")

            market_weights = read_market_weights_from_bigquery()

            df_moves, df_snap_unused, df_audit = detect_sharp_moves(
                current=live,
                previous=previous,
                sport_key=sport_key,
                SHARP_BOOKS=SHARP_BOOKS,
                REC_BOOKS=REC_BOOKS,
                BOOKMAKER_REGIONS=BOOKMAKER_REGIONS,
                weights=market_weights
            )
            logging.info(f"🔎 Detected sharp moves: {len(df_moves)} rows")

            # Rebuild df_snap manually for snapshot logging
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
                for game in live
                for book in game.get('bookmakers', [])
                for market in book.get('markets', [])
                for outcome in market.get('outcomes', [])
            ])
            df_snap = build_game_key(df_snap)

            # Save all output
            write_sharp_moves_to_master(df_moves)
            write_line_history_to_bigquery(df_audit)
            write_snapshot_to_gcs_parquet(current)  # where `current` is the JSON list from `fetch_live_odds()`

            # 🔍 Load models and apply scoring
            trained_models = {
                market: load_model_from_gcs(sport_label, market)
                for market in ['spreads', 'totals', 'h2h']
            }
            trained_models = {k: v for k, v in trained_models.items() if v}  # remove missing

            if trained_models:
                df_scored = apply_blended_sharp_score(df_moves.copy(), trained_models)
                if not df_scored.empty:
                    write_to_bigquery(df_scored)
                    logging.info(f"✅ Scored and saved {len(df_scored)} rows to sharp_scores_full.")
                else:
                    logging.info("ℹ️ No scored rows — model returned empty.")
            else:
                logging.info(f"ℹ️ No trained models found for {sport_label} — skipping scoring.")

            logging.info(f"✅ Completed: {sport_label} — Moves: {len(df_moves)}")

        except Exception as e:
            logging.error(f"❌ Error during {sport_label} detection: {e}", exc_info=True)