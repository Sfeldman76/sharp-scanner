import pandas as pd
from datetime import datetime
from collections import defaultdict
from google.cloud import bigquery, storage
from pandas_gbq import to_gbq

# === Project Config ===
GCP_PROJECT_ID = "sharplogger"
BQ_DATASET = "sharp_data"
BQ_TABLE = "sharp_moves_master"
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
MARKET_WEIGHTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.market_weights"
LINE_HISTORY_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.line_history_master"
SNAPSHOTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.odds_snapshot_log"

bq_client = bigquery.Client(project=GCP_PROJECT_ID)
gcs_client = storage.Client(project=GCP_PROJECT_ID)

# === Placeholder Function Implementations ===

def fetch_live_odds(sport_key):
    print(f"🧪 [utils] Mock fetch_live_odds for {sport_key}")
    return []

def read_latest_snapshot_from_bigquery(hours=2):
    print(f"🧪 [utils] Mock read_latest_snapshot_from_bigquery for last {hours} hours")
    return {}

def read_market_weights_from_bigquery():
    print("🧪 [utils] Mock read_market_weights_from_bigquery")
    return {}

def detect_sharp_moves(current, previous, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, weights={}):
    print(f"🧪 [utils] Mock detect_sharp_moves for {sport_key}")
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def write_sharp_moves_to_master(df, table=BQ_FULL_TABLE):
    if df is not None and not df.empty:
        print(f"✅ [utils] Would write {len(df)} rows to {table}")
    else:
        print("⚠️ [utils] No sharp moves to write.")

def write_line_history_to_bigquery(df):
    if df is not None and not df.empty:
        print(f"✅ [utils] Would write {len(df)} rows to line history table")
    else:
        print("⚠️ [utils] No line history to write.")

def upload_snapshot_to_gcs(df):
    print("✅ [utils] Would upload snapshot to GCS")