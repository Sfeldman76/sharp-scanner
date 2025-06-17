import pandas as pd
from datetime import datetime
from collections import defaultdict
from google.cloud import bigquery, storage
from pandas_gbq import to_gbq
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import requests
import numpy as np

# === Config ===
GCP_PROJECT_ID = "sharplogger"
BQ_DATASET = "sharp_data"
BQ_TABLE = "sharp_moves_master"
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
MARKET_WEIGHTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.market_weights"
LINE_HISTORY_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.line_history_master"
SNAPSHOTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.odds_snapshot_log"
GCS_BUCKET = "sharp-models"
API_KEY = "3879659fe861d68dfa2866c211294684"
bq_client = bigquery.Client(project=GCP_PROJECT_ID)
gcs_client = storage.Client(project=GCP_PROJECT_ID)

SPORTS = {
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "CFL": "americanfootball_cfl",
    "WNBA": "basketball_wnba",
}

SHARP_BOOKS_FOR_LIMITS = ['pinnacle']
SHARP_BOOKS = SHARP_BOOKS_FOR_LIMITS + ['betfair_ex_eu','betfair_ex_uk', 'smarkets','betonlineag','lowvig']

REC_BOOKS = [
    'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers',
    'fanatics', 'espnbet', 'hardrockbet', 'bovada', 'betus'
]

BOOKMAKER_REGIONS = {
    'pinnacle': 'eu', 'betfair_ex_eu': 'eu', 'betfair_ex_uk': 'uk', 'smarkets': 'uk',
    'matchbook': 'uk', 'betonlineag': 'us', 'lowvig': 'us', 'betanysports': 'us2', 'betus': 'us',
    'betmgm': 'us', 'draftkings': 'us', 'fanduel': 'us', 'betrivers': 'us', 'espnbet': 'us2',
    'hardrockbet': 'us2', 'fanatics': 'us', 'mybookieag': 'us', 'bovada': 'us', 'rebet': 'us2',
    'windcreek': 'us2', 'bet365': 'uk', 'williamhill': 'uk', 'ladbrokes': 'uk', 'unibet': 'eu',
    'bwin': 'eu', 'sportsbet': 'au', 'ladbrokesau': 'au', 'neds': 'au'
}

MARKETS = ['spreads', 'totals', 'h2h']

# === Utility Functions ===

def implied_prob(odds):
    try:
        if odds < 0:
            return -odds / (-odds + 100)
        else:
            return 100 / (odds + 100)
    except:
        return None

def ensure_columns(df, required_cols, fill_value=None):
    for col in required_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df

def normalize_team(t):
    return str(t).strip().lower().replace('.', '').replace('&', 'and')

def build_merge_key(home, away, game_start):
    return f"{normalize_team(home)}_{normalize_team(away)}_{game_start.floor('h').strftime('%Y-%m-%d %H:%M:%S')}"

def build_game_key(df):
    required = ['Game', 'Game_Start', 'Market', 'Outcome']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"⚠️ Missing columns in build_game_key: {missing}")
        return df

    df = df.copy()
    df['Home_Team_Norm'] = df['Game'].str.extract(r'^(.*?) vs')[0].str.strip().str.lower()
    df['Away_Team_Norm'] = df['Game'].str.extract(r'vs (.*)$')[0].str.strip().str.lower()
    df['Commence_Hour'] = pd.to_datetime(df['Game_Start'], errors='coerce', utc=True).dt.floor('h')
    df['Market_Norm'] = df['Market'].str.strip().str.lower()
    df['Outcome_Norm'] = df['Outcome'].str.strip().str.lower()
    df['Game_Key'] = (
        df['Home_Team_Norm'] + "_" + df['Away_Team_Norm'] + "_" +
        df['Commence_Hour'].astype(str) + "_" + df['Market_Norm'] + "_" + df['Outcome_Norm']
    )
    df['Merge_Key_Short'] = df.apply(
        lambda row: build_merge_key(row['Home_Team_Norm'], row['Away_Team_Norm'], row['Commence_Hour']),
        axis=1
    )
    return df

def fetch_live_odds(sport_key, api_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us,us2,uk,eu,au',
        'markets': ','.join(MARKETS),
        'oddsFormat': 'american',
        'includeBetLimits': 'true'
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def safe_to_gbq(df, table, replace=False):
    mode = 'replace' if replace else 'append'
    try:
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists=mode)
        return True
    except Exception as e:
        print(f"❌ Failed to upload to {table}: {e}")
        return False

def write_parquet_to_gcs(df, filename, bucket_name=GCS_BUCKET, folder="snapshots/"):
    if df.empty:
        print("⚠️ No data to write.")
        return
    table = pa.Table.from_pandas(df)
    buffer = BytesIO()
    pq.write_table(table, buffer, compression='snappy')
    blob_path = f"{folder}{filename}"
    blob = gcs_client.bucket(bucket_name).blob(blob_path)
    blob.upload_from_string(buffer.getvalue(), content_type="application/octet-stream")
    print(f"✅ Uploaded Parquet to gs://{bucket_name}/{blob_path}")

def read_latest_snapshot_from_bigquery(hours=2):
    try:
        query = f"""
            SELECT * FROM `{SNAPSHOTS_TABLE}`
            WHERE Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
        """
        df = bq_client.query(query).to_dataframe()

        # Restructure into grouped dict format expected by detect_sharp_moves
        grouped = defaultdict(lambda: {"bookmakers": []})
        for _, row in df.iterrows():
            gid = row["Game_ID"]
            entry = grouped[gid]
            found_book = next((b for b in entry["bookmakers"] if b["key"] == row["Bookmaker"]), None)
            if not found_book:
                found_book = {"key": row["Bookmaker"], "markets": []}
                entry["bookmakers"].append(found_book)
            found_market = next((m for m in found_book["markets"] if m["key"] == row["Market"]), None)
            if not found_market:
                found_market = {"key": row["Market"], "outcomes": []}
                found_book["markets"].append(found_market)
            found_market["outcomes"].append({
                "name": row["Outcome"],
                "point": row["Value"],
                "price": row["Value"] if row["Market"] == "h2h" else None,
                "bet_limit": row["Limit"]
            })

        return dict(grouped)
    except Exception as e:
        print(f"❌ Failed to load snapshot from BigQuery: {e}")
        return {}

