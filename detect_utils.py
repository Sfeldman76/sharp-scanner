import logging
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
    write_to_bigquery
)

def detect_and_save_all_sports():
    for sport_label in ["NBA", "MLB", "WNBA", "CFL"]: 
        sport_key = SPORTS[sport_label]
        logging.info(f"🔍 Running sharp detection for {sport_label}...")

        try:
            current = fetch_live_odds(sport_key, API_KEY)
            logging.info(f"📥 Odds pulled: {len(current)} games")

            previous = read_latest_snapshot_from_bigquery()
            logging.info(f"📦 Previous snapshot loaded: {len(previous)} games")

            market_weights = read_market_weights_from_bigquery()
            df_moves, df_snap, df_audit = detect_sharp_moves(
                current=current,
                previous=previous,
                sport_key=sport_key,
                SHARP_BOOKS=SHARP_BOOKS,
                REC_BOOKS=REC_BOOKS,
                BOOKMAKER_REGIONS=BOOKMAKER_REGIONS,
                weights=market_weights
            )
            logging.info(f"🔎 Detected sharp moves: {len(df_moves)} rows")

            # Save snapshot + raw sharp moves
            write_sharp_moves_to_master(df_moves)
            write_line_history_to_bigquery(df_audit)
            write_snapshot_to_gcs_parquet(current)

            # 🔍 Load models and apply scoring
            # ✅ Load models for this sport
            trained_models = {
                market: load_model_from_gcs(sport_label, market)
                for market in ['spreads', 'totals', 'h2h']
            }
            trained_models = {k: v for k, v in trained_models.items() if v}  # filter out missing models
            
            # 🧠 Apply model scoring if available
            if trained_models:
                df_scored = apply_blended_sharp_score(df_moves.copy(), trained_models)
                write_to_bigquery(df_scored)
                logging.info(f"✅ Scored and saved {len(df_scored)} rows to sharp_scores_full.")
            else:
                logging.warning("⚠️ No trained models found — skipping scoring.")


            logging.info(f"✅ Completed: {sport_label} — Moves: {len(df_moves)}")

        except Exception as e:
            logging.warning(f"⚠️ No model found for {sport}-{market}: {e}")
            return None
