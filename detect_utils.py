from config import SPORTS, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS
from utils import (
    fetch_live_odds,
    read_latest_snapshot_from_bigquery,
    read_market_weights_from_bigquery,
    detect_sharp_moves,
    write_sharp_moves_to_master,
    write_line_history_to_bigquery,
    write_snapshot_to_gcs_parquet,
)

def detect_and_save_all_sports():
    for sport_label in ["NBA", "MLB","WNBA","CFL"]: 
        print(f"🔍 Running sharp detection for {sport_label}...")
        sport_key = SPORTS[sport_label]

        try:
            current = fetch_live_odds(sport_key)
            previous = read_latest_snapshot_from_bigquery()
            market_weights = read_market_weights_from_bigquery()

            df_moves, df_snap, df_audit = detect_sharp_moves(
                current=current,
                previous=previous,
                sport_key=sport_key,
                SHARP_BOOKS=SHARP_BOOKS,
                REC_BOOKS=REC_BOOKS,
                BOOKMAKER_REGIONS=BOOKMAKER_REGIONS,
                weights=market_weights,
            )

            write_sharp_moves_to_master(df_moves)
            write_line_history_to_bigquery(df_audit)
            upload_snapshot_to_gcs(df_snap)

            print(f"✅ Completed: {sport_label} — Moves: {len(df_moves)}")

        except Exception as e:
            print(f"❌ Error during detection for {sport_label}: {e}")
