from config import SPORTS, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, API_KEY
from utils import (
    fetch_live_odds,
    read_latest_snapshot_from_bigquery,
    read_market_weights_from_bigquery,
    detect_sharp_moves,
    write_sharp_moves_to_master,
    write_line_history_to_bigquery,
    write_snapshot_to_gcs_parquet,
    detect_market_leaders
)

def detect_and_save_all_sports():
    for sport_label in ["NBA", "MLB", "WNBA", "CFL"]: 
        print(f"🔍 Running sharp detection for {sport_label}...")
        sport_key = SPORTS[sport_label]

        try:
            # Fetch current odds
            current = fetch_live_odds(sport_key, API_KEY)

            # Historical snapshot + weights
            previous = read_latest_snapshot_from_bigquery()
            market_weights = read_market_weights_from_bigquery()

            # Detection logic
            df_moves_raw, df_audit, _ = detect_sharp_moves(
                current=current,
                previous=previous,
                sport_key=sport_key,
                SHARP_BOOKS=SHARP_BOOKS,
                REC_BOOKS=REC_BOOKS,
                BOOKMAKER_REGIONS=BOOKMAKER_REGIONS,
                weights=market_weights,
            )

            # Save results
            write_sharp_moves_to_master(df_moves_raw)
            write_line_history_to_bigquery(df_audit)
            write_snapshot_to_gcs_parquet(current)

            print(f"✅ Completed: {sport_label} — Moves: {len(df_moves_raw)}")

        except Exception as e:
            print(f"❌ Error during detection for {sport_label}: {e}")
