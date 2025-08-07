import pandas as pd
from datetime import datetime
from collections import defaultdict
from google.cloud import bigquery, storage
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import requests
import numpy as np
import logging
import hashlib
import time
import json
import psutil
import os
import gc
import psutil
import logging
from pandas_gbq import to_gbq
import traceback
import pickle  # ✅ Add this at the top of your script
          
import sys
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from google.cloud import bigquery, storage
import logging
logging.basicConfig(level=logging.INFO)  # <- Must be INFO or DEBUG to show .info() logs
logger = logging.getLogger(__name__)

import warnings
from sklearn.exceptions import InconsistentVersionWarning

# 🔇 Suppress scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# 🔇 Suppress XGBoost C++ backend warnings
os.environ["XGBOOST_VERBOSITY"] = "0"

# === Config ===
GCP_PROJECT_ID = "sharplogger"
BQ_DATASET = "sharp_data"
BQ_TABLE = "sharp_moves_master"
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
MARKET_WEIGHTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.market_weights"
#LINE_HISTORY_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.line_history_master"
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
    "NFL": "americanfootball_nfl",     # National Football League
    "NCAAF": "americanfootball_ncaaf", # NCAA College Football
}

SHARP_BOOKS_FOR_LIMITS = ['pinnacle']
SHARP_BOOKS = SHARP_BOOKS_FOR_LIMITS + ['betus','mybookieag','smarkets','betfair_ex_eu','betfair_ex_uk','betfair_ex_au','lowvig','betonlineag','matchbook']

REC_BOOKS = [
    'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers',
    'fanatics', 'espnbet', 'hardrockbet','sport888']


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
        odds = float(odds)
        if pd.isna(odds):
            return None
        if odds < 0:
            return -odds / (-odds + 100)
        else:
            return 100 / (odds + 100)
    except:
        return None

def calc_implied_prob(odds):
    try:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    except Exception:
        return None

def ensure_columns(df, required_cols, fill_value=None):
    for col in required_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df

def normalize_book_name(bookmaker: str, book: str) -> str:
    book = book.lower().strip() if isinstance(book, str) else ""
    bookmaker = bookmaker.lower().strip() if isinstance(bookmaker, str) else ""

    if bookmaker == "betfair" and book.startswith("betfair_ex_"):
        return book  # e.g., betfair_ex_uk

    return bookmaker




def normalize_team(t):
    return str(t).strip().lower().replace('.', '').replace('&', 'and')

def build_merge_key(home, away, game_start):
    return f"{normalize_team(home)}_{normalize_team(away)}_{game_start.floor('h').strftime('%Y-%m-%d %H:%M:%S')}"


def compute_line_hash(row):
    try:
        key_fields = [
            str(row.get('Game_Key', '')).strip().lower(),
            str(row.get('Bookmaker', '')).strip().lower(),
            str(row.get('Market', '')).strip().lower(),
            str(row.get('Outcome', '')).strip().lower(),
            str(row.get('Value', '')).strip(),
            str(row.get('Odds_Price', '')).strip(),
            str(row.get('Limit', '')).strip(),
        ]
        key = "|".join(key_fields)
        return hashlib.md5(key.encode()).hexdigest()
    except Exception as e:
        return f"ERROR_HASH_{hashlib.md5(str(e).encode()).hexdigest()[:8]}"


def log_memory(msg=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    logging.info(f"🔍 Memory {msg}: RSS={mem.rss / 1024 / 1024:.2f} MB, VMS={mem.vms / 1024 / 1024:.2f} MB")

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


model_cache = {}


def get_trained_models(sport_key):
    if sport_key not in model_cache:
        model_cache[sport_key] = {
            market: load_model_from_gcs(sport_key, market)
            for market in ['spreads', 'totals', 'h2h']
        }
    return model_cache[sport_key]
    
    
sharp_moves_cache = {}

def read_recent_sharp_master_cached(hours=120):
    cache_key = f"sharp_master_{hours}h"
    
    if cache_key in sharp_moves_cache:
        return sharp_moves_cache[cache_key]
    
    df = read_recent_sharp_moves(hours=hours)  # this fetches from BigQuery
    sharp_moves_cache[cache_key] = df
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



def write_snapshot_to_gcs_parquet(snapshot_list, bucket_name="sharp-models", folder="snapshots/"):
    rows = []
    snapshot_time = pd.Timestamp.utcnow()

    for game in snapshot_list:
        gid = game.get('id')
        if not gid:
            continue

        home = game.get("home_team", "").strip().lower()
        away = game.get("away_team", "").strip().lower()
        game_name = f"{home.title()} vs {away.title()}"
        event_time = pd.to_datetime(game.get("commence_time"), utc=True, errors='coerce')

        for book in game.get('bookmakers', []):
            book_key = book.get('key')
            for market in book.get('markets', []):
                market_key = market.get('key')
                for outcome in market.get('outcomes', []):
                    rows.append({
                        'Game_ID': gid,
                        'Game': game_name,
                        'Game_Start': event_time,
                        'Bookmaker': book_key,
                        'Market': market_key,
                        'Outcome': outcome.get('name'),
                        'Value': outcome.get('point') if market_key != 'h2h' else outcome.get('price'),
                        'Odds_Price': outcome.get('price'),  # ✅ This adds the odds for spread/total
                        'Limit': outcome.get('bet_limit'),
                        'Snapshot_Timestamp': snapshot_time
                    })

 

    # Build Game_Key in df_snap using the same function as df_moves_raw
    df_snap = pd.DataFrame(rows)

    # ✅ Only run build_game_key if required fields exist
    required_fields = {'Game', 'Game_Start', 'Market', 'Outcome'}
    if required_fields.issubset(df_snap.columns):
        df_snap = build_game_key(df_snap)
    else:
        missing = required_fields - set(df_snap.columns)
        logging.warning(f"⚠️ Skipping build_game_key — missing columns: {missing}")

    if df_snap.empty:
        logging.warning("⚠️ No snapshot data to upload to GCS.")
        return

    filename = f"{folder}{snapshot_time.strftime('%Y%m%d_%H%M%S')}_snapshot.parquet"
    buffer = BytesIO()

    try:
        table = pa.Table.from_pandas(df_snap)
        pq.write_table(table, buffer, compression='snappy')
    except Exception as e:
        logging.exception("❌ Failed to write snapshot DataFrame to Parquet.")
        return

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_string(buffer.getvalue(), content_type='application/octet-stream')
        logging.info(f"✅ Snapshot uploaded to GCS: gs://{bucket_name}/{filename}")
    except Exception as e:
        logging.exception("❌ Failed to upload snapshot to GCS.")

def read_latest_snapshot_from_bigquery(hours=24):
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
            price = row["Value"] if row["Market"].lower() == "h2h" else row.get("Odds_Price", None)

            found_market["outcomes"].append({
                "name": row["Outcome"],
                "point": row["Value"],
                "price": price,
                "bet_limit": row["Limit"]
            })
        return dict(grouped)
    except Exception as e:
        print(f"❌ Failed to load snapshot from BigQuery: {e}")
        return {}



def write_sharp_moves_to_master(df, table='sharp_data.sharp_moves_master'):
    if df is None or df.empty:
        logging.warning("⚠️ No sharp moves to write.")
        return

    df = df.copy()
    df = build_game_key(df)

    allowed_books = SHARP_BOOKS + REC_BOOKS
    # Apply normalization early
    # ✅ Normalize book names before any filtering
    df['Book'] = df['Book'].str.lower()
    

    
    # ✅ Now apply the filtering with normalized names
    df = df[df['Book'].isin(allowed_books)]


    if 'Game_Key' not in df.columns or df['Game_Key'].isnull().all():
        logging.warning("❌ No valid Game_Key present — skipping upload.")
        logging.debug(df[['Game', 'Game_Start', 'Market', 'Outcome']].head().to_string())
        return
    # ❌ Don't write any rows that are already past game start
    if 'Post_Game' in df.columns:
        pre_filter = len(df)
        df = df[df['Post_Game'] == False]
        logging.info(f"🧹 Removed {pre_filter - len(df)} post-game rows before writing to sharp_moves_master")

      

    logging.info(f"🧪 Sharp moves ready to write: {len(df)}")

    # Clean column names
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    df = df.drop(columns=[col for col in df.columns if col.endswith('_x') or col.endswith('_y')], errors='ignore')

    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce', utc=True)
   

   
    # Convert object columns safely
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].where(df[col].notna(), None)
    # Preview model columns
    model_cols = [
        'Model_Sharp_Win_Prob', 'Model_Confidence_Tier', 'SharpBetScore',
        'Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score',
        'Final_Confidence_Score', 'Model_Confidence'
    ]
    # Define the full expected schema
    ALLOWED_COLS = [
        # Metadata
        'Game_Key', 'Time', 'Game', 'Game_Start', 'Sport', 'Market', 'Outcome',
        'Bookmaker', 'Book', 'Value', 'Limit', 'Delta', 'Old_Value',
        'Event_Date', 'Home_Team_Norm', 'Away_Team_Norm', 'Commence_Hour',
        'Limit_Max', 'Delta_vs_Sharp','Team_Key',

        # Sharp logic fields
        'SHARP_SIDE_TO_BET', 'Sharp_Move_Signal', 'Sharp_Limit_Jump',
        'Sharp_Prob_Shift', 'Sharp_Time_Score', 'Sharp_Limit_Total', 'SharpBetScore',
        'Open_Value', 'Open_Book_Value', 'Opening_Limit', 'Limit_Jump',
        'Sharp_Timing', 'Limit_NonZero', 'Limit_Min', 'Market_Leader',
        'Is_Pinnacle', 'LimitUp_NoMove_Flag', 'SupportKey', 'CrossMarketSharpSupport',
        'Is_Reinforced_MultiMarket','Is_Sharp_Book',

        # Scoring / diagnostics
        'True_Sharp_Confidence_Score', 'Enhanced_Sharp_Confidence_Score',
        'Sharp_Confidence_Tier', 'Model_Sharp_Win_Prob', 'Model_Confidence_Tier',
        'Final_Confidence_Score', 'Model_Confidence', 'Snapshot_Timestamp',
        'Market_Norm', 'Outcome_Norm', 'Merge_Key_Short', 'Line_Hash',
        'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT', 'Scored', 'Pre_Game', 'Post_Game',
        'Unique_Sharp_Books', 'Sharp_Move_Magnitude_Score', 'Was_Canonical',
        'Scored_By_Model', 'Scoring_Market',
        'Line_Move_Magnitude',
        'Is_Home_Team_Bet',
        'Is_Favorite_Bet',
        'High_Limit_Flag',
        'Line_Delta',               # ✅ Add this
        'Line_Magnitude_Abs',       # Already present
        'Direction_Aligned','Odds_Price', 'Implied_Prob', 
        'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds',
        'Value_Reversal_Flag', 'Odds_Reversal_Flag','Open_Odds', 'Was_Line_Resistance_Broken',
        'Line_Resistance_Crossed_Levels',
        'Line_Resistance_Crossed_Count', 'Late_Game_Steam_Flag', 'Sharp_Line_Magnitude',
        'Rec_Line_Magnitude',
        'SharpMove_Odds_Up',
        'SharpMove_Odds_Down',
        'SharpMove_Odds_Mag',
        'SharpMove_Resistance_Break',
        'Active_Signal_Count', 
        'SharpMove_Magnitude_Overnight_VeryEarly', 'SharpMove_Magnitude_Overnight_MidRange',
        'SharpMove_Magnitude_Overnight_LateGame', 'SharpMove_Magnitude_Overnight_Urgent',
        'SharpMove_Magnitude_Early_VeryEarly', 'SharpMove_Magnitude_Early_MidRange',
        'SharpMove_Magnitude_Early_LateGame', 'SharpMove_Magnitude_Early_Urgent',
        'SharpMove_Magnitude_Midday_VeryEarly', 'SharpMove_Magnitude_Midday_MidRange',
        'SharpMove_Magnitude_Midday_LateGame', 'SharpMove_Magnitude_Midday_Urgent',
        'SharpMove_Magnitude_Late_VeryEarly', 'SharpMove_Magnitude_Late_MidRange',
        'SharpMove_Magnitude_Late_LateGame', 'SharpMove_Magnitude_Late_Urgent',
        'SharpMove_Timing_Dominant',
        'SharpMove_Timing_Magnitude',# === Net movement columns
        'Net_Line_Move_From_Opening',
        'Abs_Line_Move_From_Opening',
        'Net_Odds_Move_From_Opening',
        'Abs_Odds_Move_From_Opening',
    ]
    
    # 🧩 Add schema-consistent consensus fields from summarize_consensus()
    ALLOWED_ODDS_MOVE_COLUMNS = [
        f'OddsMove_Magnitude_{b}' for b in [
            'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
            'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
            'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
            'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
        ]
    ] + ['Odds_Move_Magnitude']  # ✅ Add this total field
    
    ALLOWED_COLS += ALLOWED_ODDS_MOVE_COLUMNS
    # Ensure all required columns exist
    df = ensure_columns(df, ALLOWED_COLS, fill_value=None)
    df['Odds_Price'] = pd.to_numeric(df.get('Odds_Price'), errors='coerce')
    df['Implied_Prob'] = pd.to_numeric(df.get('Implied_Prob'), errors='coerce')

    # 🔍 Add required column check before filtering
    required_cols = ['Snapshot_Timestamp', 'Model_Sharp_Win_Prob']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logging.error(f"❌ Missing required columns before upload: {missing}")
        return

    # Log any remaining mismatches
    missing_cols = [col for col in ALLOWED_COLS if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ Missing required columns for BigQuery upload: {missing_cols}")
        return
    # === Deduplicate using Line_Hash from previous uploads ===
    try:
        client = bigquery.Client(project=GCP_PROJECT_ID, location="us")
        df_prev = client.query("""
            SELECT DISTINCT Line_Hash
            FROM `sharplogger.sharp_data.sharp_moves_master`
            WHERE DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
        """).to_dataframe()
    
        if not df_prev.empty and 'Line_Hash' in df.columns:
            before = len(df)
            df = df[~df['Line_Hash'].isin(df_prev['Line_Hash'])]
            logging.info(f"🧼 Line_Hash dedup: removed {before - len(df)} duplicate rows")
    except Exception as e:
        logging.warning(f"⚠️ Failed to deduplicate using Line_Hash: {e}")
    
    # ⛔ Exit early if nothing to write
    if df.empty:
        logging.info("🛑 No new rows after Line_Hash deduplication — exiting.")
        return
    logging.info(f"📦 Final row count to upload after filtering and dedup: {len(df)}")
    # Filter to allowed schema
    # === Force float type on columns that might fail if BigQuery expects INT64
    # === Cast known float columns safely
    known_float_cols = [
        'SharpMove_Odds_Mag', 'HomeRecLineMag',
        'Rec_Line_Delta', 'Sharp_Line_Delta',
        'Odds_Shift', 'Implied_Prob_Shift', 'Line_Delta',
        'Sharp_Line_Magnitude', 'Rec_Line_Magnitude',
        'Delta_Sharp_vs_Rec','Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds'
    ]
    for col in known_float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    # === Cast known INT64 columns to nullable Int64
    int_cols = [
        'Sharp_Limit_Total', 'Limit_Jump', 'LimitUp_NoMove_Flag',
        'SHARP_SIDE_TO_BET', 'Sharp_Move_Signal', 'Sharp_Limit_Jump',
        'Scored', 'Scored_By_Model', 'Is_Home_Team_Bet',
        'Is_Favorite_Bet', 'High_Limit_Flag', 'CrossMarketSharpSupport','Value_Reversal_Flag', 'Odds_Reversal_Flag'
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
    df = df[ALLOWED_COLS]
    
    logging.info("🧪 Preview of model columns being written:")
    logging.info(df[model_cols].dropna(how='all').head(5).to_string())
    # 🔍 Preview Odds_Price and Implied_Prob distribution
    if 'Odds_Price' in df.columns and 'Implied_Prob' in df.columns:
        logging.info("🎯 Odds_Price sample:\n" + df['Odds_Price'].dropna().astype(str).head().to_string(index=False))
        logging.info("🎯 Implied_Prob sample:\n" + df['Implied_Prob'].dropna().round(4).astype(str).head().to_string(index=False))
    else:
        logging.warning("⚠️ Odds_Price or Implied_Prob missing from DataFrame before upload")
    logging.info(f"📦 Final row count to upload after filtering and dedup: {len(df)}")
    # Write to BigQuery
    try:
        logging.info(f"📤 Uploading to `{table}`...")
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='append')
        logging.info(f"✅ Wrote {len(df)} new rows to `{table}`")
    except Exception as e:
        logging.exception(f"❌ Upload to `{table}` failed.")
        logging.debug("Schema:\n" + df.dtypes.to_string())
        logging.debug("Preview:\n" + df.head(5).to_string())
        


def normalize_label(label):
     return str(label).strip().lower().replace('.0', '')

def normalize_book_key(raw_key, sharp_books, rec_books):
    raw_key = raw_key.lower()

    # First: Exact matches for sharp books (preserve unique identifiers like betfair_uk/eu)
    for sharp in sharp_books:
        if raw_key == sharp:
            return sharp

    # Second: Exact matches for rec books
    for rec in rec_books:
        if raw_key == rec:
            return rec

    # Third: Fallback to partial match — but only if no exact match occurred
    for rec in rec_books:
        if rec.replace(" ", "") in raw_key:
            return rec.replace(" ", "")
    for sharp in sharp_books:
        if sharp in raw_key:
            return sharp

    # ✅ Final fallback — return normalized key as-is
    return raw_key


#def implied_prob(odds):
    #try:
        #odds = float(odds)
        #if odds > 0:
            #return 100 / (odds + 100)
        #else:
            #return abs(odds) / (abs(odds) + 100)
    #except:
        #return None

def load_market_weights_from_bq():
    
    client = bigquery.Client(project="sharplogger", location="us")

    query = """
        SELECT *
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE SHARP_HIT_BOOL IS NOT NULL
          AND Scored = TRUE
    """
    df = client.query(query).to_dataframe()

    market_weights = compute_and_write_market_weights(df)  # Returns a dict
    logging.info(f"✅ Loaded market weights for {len(market_weights)} markets")
    return market_weights
    
def implied_prob_to_point_move(prob_delta, base_odds=-110):
    """
    Convert a probability delta into an approximate odds point movement
    assuming the slope near a base odds value (default -110).
    """
    # Use slope near -110 (standard point spread odds)
    # At -110, implied prob ≈ 0.524, slope ≈ ~0.0045 per point
    approx_slope = 0.0045  # ← 1 point ≈ 0.0045 prob shift near -110
    
    point_equiv = prob_delta / approx_slope
    return point_equiv    


def compute_sharp_metrics(entries, open_val, mtype, label, gk=None, book=None, open_odds=None, opening_limit=None):
    logging.debug(f"🔍 Running compute_sharp_metrics for Outcome: {label}, Market: {mtype}")
    logging.debug(f"📥 Open value: {open_val}, Open odds: {open_odds}")
    logging.debug(f"📦 Received {len(entries)} entries")

    move_signal = 0.0
    move_magnitude_score = 0.0
    limit_score = 0.0
    total_limit = 0.0
    hybrid_timing_mags = defaultdict(float)
    odds_move_magnitude_score = 0.0
    hybrid_timing_odds_mags = defaultdict(float)
        # === Net movement from opening
    net_line_move = None
    abs_net_line_move = None
    net_odds_move = None
    abs_net_odds_move = None
   
    
    def get_hybrid_bucket(ts, game_start):
        hour = pd.to_datetime(ts).hour
        minutes_to_game = (
            (pd.to_datetime(game_start) - pd.to_datetime(ts)).total_seconds() / 60
            if pd.notnull(game_start) else None
        )
        tod = (
            'Overnight' if 0 <= hour <= 5 else
            'Early'     if 6 <= hour <= 11 else
            'Midday'    if 12 <= hour <= 15 else
            'Late'
        )
        mtg = (
            'VeryEarly' if minutes_to_game is None else
            'VeryEarly' if minutes_to_game > 720 else
            'MidRange'  if 180 < minutes_to_game <= 720 else
            'LateGame'  if 60 < minutes_to_game <= 180 else
            'Urgent'
        )
        return f"{tod}_{mtg}"

    # === NEW: track previous values for proper delta logic
    entries = sorted(entries, key=lambda x: x[2])  # Sort by timestamp
    prev_val = open_val
    prev_odds = open_odds

    for i, entry in enumerate(entries):
        if len(entry) != 5:
            logging.warning(f"⚠️ Malformed entry {i+1}: {entry}")
            continue
        
        limit, curr_val, ts, game_start, curr_odds = entry
        logging.debug(f"🧾 Entry {i+1} → Limit={limit}, Value={curr_val}, Time={ts}, Odds={curr_odds}")

        try:
            # === Line movement
            if pd.notna(prev_val) and pd.notna(curr_val):
                delta = curr_val - prev_val  # 🔁 Signed delta
                sharp_move_delta = delta     # No abs()
            
                # ✅ Always count signed movement magnitude
                move_magnitude_score += sharp_move_delta
            
                # === Direction-aware sharp signal scoring
                if mtype == 'totals':
                    if 'under' in label and sharp_move_delta < 0:
                        move_signal += sharp_move_delta
                    elif 'over' in label and sharp_move_delta > 0:
                        move_signal += sharp_move_delta
                elif mtype == 'spreads':
                    if prev_val < 0 and curr_val < prev_val:  # Favorite getting stronger
                        move_signal += sharp_move_delta
                    elif prev_val > 0 and curr_val > prev_val:  # Dog getting more dog
                        move_signal += sharp_move_delta
            
                # ✅ Signed per-timing contribution
                timing_label = get_hybrid_bucket(ts, game_start)
                hybrid_timing_mags[timing_label] += sharp_move_delta


                prev_val = curr_val

            # === Odds movement
            if pd.notna(prev_odds) and pd.notna(curr_odds):
                # ✅ Convert to implied probability FIRST
                prev_prob = implied_prob(prev_odds)
                curr_prob = implied_prob(curr_odds)
            
                # ✅ Then compute delta (signed)
                odds_delta = curr_prob - prev_prob
                point_equiv = implied_prob_to_point_move(odds_delta)
            
                logging.debug(f"🧾 Odds Δ: {odds_delta:+.3f} (~{point_equiv:+.1f} pts), From {prev_odds} → {curr_odds}")
            
                # ✅ Always include signed magnitude
                odds_move_magnitude_score += point_equiv  # preserve sign
                timing_label = get_hybrid_bucket(ts, game_start)
                hybrid_timing_odds_mags[timing_label] += odds_delta  # preserve sign
            
                prev_odds = curr_odds  # ✅ Important: update for next iteration

                
                
            # === Net line movement from open to final value
            if pd.notna(open_val) and pd.notna(curr_val):
                net_line_move = curr_val - open_val
                abs_net_line_move = abs(net_line_move)
        
            # === Net odds movement from open to final odds
            if pd.notna(open_odds) and pd.notna(curr_odds):
                net_odds_move = curr_odds - open_odds
                abs_net_odds_move = abs(net_odds_move)
                
                
            # === Limit tracking
            if limit is not None:
                total_limit += limit
                if limit >= 100:
                    limit_score += limit

        except Exception as e:
            logging.warning(f"⚠️ Error in entry {i+1}: {e}")

    # === Final bucket flattening
    all_possible_buckets = [
        f"{tod}_{mtg}"
        for tod in ['Overnight', 'Early', 'Midday', 'Late']
        for mtg in ['VeryEarly', 'MidRange', 'LateGame', 'Urgent']
    ]
    flattened_buckets = {
        f'SharpMove_Magnitude_{b}': round(hybrid_timing_mags.get(b, 0.0), 3)
        for b in all_possible_buckets
    }
    flattened_odds_buckets = {
        f'OddsMove_Magnitude_{b}': round(hybrid_timing_odds_mags.get(b, 0.0), 3)
        for b in all_possible_buckets
    }

    dominant_label, dominant_mag = max(
        hybrid_timing_mags.items(), key=lambda x: x[1], default=("unknown", 0.0)
    )
    logging.debug(f"📊 Final hybrid_timing_mags: {dict(hybrid_timing_mags)}")
    logging.debug(f"📊 Final hybrid_timing_odds_mags: {dict(hybrid_timing_odds_mags)}")
    return {
        'Sharp_Move_Signal': int(move_signal > 0),
        'Opening_Limit': opening_limit,
        'Sharp_Line_Magnitude': round(move_magnitude_score, 2),
        'Sharp_Limit_Jump': int(limit_score >= 10000),
        'Sharp_Limit_Total': round(total_limit, 1),
        'SharpMove_Timing_Dominant': dominant_label,
        'Net_Line_Move_From_Opening': round(net_line_move, 3) if net_line_move is not None else None,
        'Abs_Line_Move_From_Opening': round(abs_net_line_move, 3) if abs_net_line_move is not None else None,
        'Net_Odds_Move_From_Opening': round(net_odds_move, 3) if net_odds_move is not None else None,
        'Abs_Odds_Move_From_Opening': round(abs_net_odds_move, 3) if abs_net_odds_move is not None else None,
        'SharpMove_Timing_Magnitude': round(move_magnitude_score, 3),
        **flattened_buckets,
        'Odds_Move_Magnitude': round(odds_move_magnitude_score, 2),
        **flattened_odds_buckets,
        'SharpBetScore': 0.0
    }
    
def apply_compute_sharp_metrics_rowwise(df: pd.DataFrame, df_all_snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Applies compute_sharp_metrics() to each row in df using historical snapshots.
    Enriches df with sharp movement features like Sharp_Move_Signal, timing buckets, etc.
    """
    if df.empty or df_all_snapshots.empty:
        return df

    df = df.copy()
    
    # Ensure required fields exist
    for col in ['Game_Key', 'Market', 'Outcome', 'Bookmaker']:
        if col not in df.columns:
            raise ValueError(f"Missing required column in df: {col}")
    
    # Normalize keys for consistency
    df_all_snapshots = df_all_snapshots.copy()
    df_all_snapshots['Snapshot_Timestamp'] = pd.to_datetime(df_all_snapshots['Snapshot_Timestamp'], errors='coerce', utc=True)
    df_all_snapshots['Commence_Hour'] = pd.to_datetime(df_all_snapshots['Game_Start'], errors='coerce', utc=True).dt.floor('h')
    df_all_snapshots['Team_Key'] = (
        df_all_snapshots['Home_Team_Norm'] + "_" +
        df_all_snapshots['Away_Team_Norm'] + "_" +
        df_all_snapshots['Commence_Hour'].astype(str) + "_" +
        df_all_snapshots['Market'] + "_" +
        df_all_snapshots['Outcome']
    )

    # Index for faster access
    snapshots_grouped = df_all_snapshots.groupby(['Game_Key', 'Market', 'Outcome', 'Bookmaker'])

    enriched_rows = []

    for idx, row in df.iterrows():
        gk = row['Game_Key']
        market = row['Market']
        outcome = row['Outcome']
        book = row['Bookmaker']

        try:
            group = snapshots_grouped.get_group((gk, market, outcome, book))

        except KeyError:
            default_metrics = {
                'Sharp_Move_Signal': 0,
                'Opening_Limit': None,
                'Sharp_Line_Magnitude': 0.0,
                'Sharp_Limit_Jump': 0,
                'Sharp_Limit_Total': 0.0,
                'SharpMove_Timing_Dominant': 'unknown',
                'Net_Line_Move_From_Opening': None,
                'Abs_Line_Move_From_Opening': None,
                'Net_Odds_Move_From_Opening': None,
                'Abs_Odds_Move_From_Opening': None,
                'SharpMove_Timing_Magnitude': 0.0,
                'Odds_Move_Magnitude': 0.0,
                'SharpBetScore': 0.0,
                **{f'SharpMove_Magnitude_{b}': 0.0 for b in [
                    f'{tod}_{mtg}' for tod in ['Overnight', 'Early', 'Midday', 'Late']
                                    for mtg in ['VeryEarly', 'MidRange', 'LateGame', 'Urgent']
                ]},
                **{f'OddsMove_Magnitude_{b}': 0.0 for b in [
                    f'{tod}_{mtg}' for tod in ['Overnight', 'Early', 'Midday', 'Late']
                                    for mtg in ['VeryEarly', 'MidRange', 'LateGame', 'Urgent']
                ]}
            }
            enriched_rows.append({**row, **default_metrics})
            continue

          

        group = group.sort_values('Snapshot_Timestamp')

        game_start = (
            group['Game_Start'].dropna().iloc[0]
            if 'Game_Start' in group and not group['Game_Start'].isnull().all()
            else None
        )
        open_val = (
            group['Value'].dropna().iloc[0]
            if not group['Value'].isnull().all()
            else None
        )
        open_odds = (
            group['Odds_Price'].dropna().iloc[0]
            if not group['Odds_Price'].isnull().all()
            else None
        )
        opening_limit = (
            group['Limit'].dropna().iloc[0]
            if not group['Limit'].isnull().all()
            else None
        )

        entries = list(zip(
            group['Limit'],
            group['Value'],
            group['Snapshot_Timestamp'],
            [game_start] * len(group),
            group['Odds_Price']
        ))

        metrics = compute_sharp_metrics(
            entries=entries,
            open_val=open_val,
            mtype=market,
            label=outcome,
            gk=gk,
            book=book,
            open_odds=open_odds,
            opening_limit=opening_limit
        )

        enriched_rows.append({**row, **metrics})

    df_enriched = pd.DataFrame(enriched_rows)

    return df_enriched
    
def compute_all_sharp_metrics(df_all_snapshots):
    results = []

    grouped = df_all_snapshots.groupby(['Game_Key', 'Market', 'Outcome', 'Bookmaker'])

    for (gk, market, outcome, book), group in grouped:
        game_start = (
            group['Game_Start'].dropna().iloc[0]
            if 'Game_Start' in group and not group['Game_Start'].isnull().all()
            else None
        )
        open_val = (
            group.sort_values('Snapshot_Timestamp')['Value']
            .dropna().iloc[0]
            if not group['Value'].isnull().all()
            else None
        )
        open_odds = (
            group.sort_values('Snapshot_Timestamp')['Odds_Price']
            .dropna().iloc[0]
            if not group['Odds_Price'].isnull().all()
            else None
        )

        entries = list(zip(
            group['Limit'],
            group['Value'],
            group['Snapshot_Timestamp'],
            [game_start] * len(group),
            group['Odds_Price']
        ))

        metrics = compute_sharp_metrics(
            entries, open_val=open_val, mtype=market, label=outcome,
            gk=gk, book=book, open_odds=open_odds
        )
        metrics.update({
            'Game_Key': gk,
            'Market': market,
            'Outcome': outcome,
            'Bookmaker': book
        })
        results.append(metrics)

    return pd.DataFrame(results)
    
    
SPORT_ALIAS = {
    'AMERICANFOOTBALL_CFL': 'CFL',
    'BASEBALL_MLB': 'MLB',
    'BASKETBALL_WNBA': 'WNBA',
    'AMERICANFOOTBALL_NFL': 'NFL',
    'AMERICANFOOTBALL_NCAAF': 'NCAAF',
    'BASKETBALL_NBA': 'NBA',
    'BASKETBALL_NCAAB': 'NCAAB',
    'CFL': 'CFL',
    'MLB': 'MLB',
    'WNBA': 'WNBA',
    'NFL': 'NFL',
    'NCAAF': 'NCAAF',
    'NBA': 'NBA',
    'NCAAB': 'NCAAB'
}

KEY_LINE_RESISTANCE = {
    'NFL': {'spread': [3, 7, 10, 14], 'total': [41, 44, 47, 51]},
    'NBA': {'spread': [2.5, 5, 7, 10], 'total': [210, 220, 225, 230]},
    'WNBA': {
        'spread': [1.5, 3.5, 6.5, 9.5],     # updated with common clustering points
        'total': [157.5, 162.5, 167.5, 172.5]  # reflects key ranges in WNBA totals
    },
    'CFL': {
        'spread': [2.5, 4.5, 6.5, 9.5],      # narrower CFL games; 6.5 is key for TDs
        'total': [48.5, 52.5, 55.5, 58.5]    # aligns with common key total zones
    },
    'NCAAF': {'spread': [3, 7, 10, 14, 17], 'total': [45, 52, 59, 66]},
    'NCAAB': {'spread': [2, 5, 7, 10], 'total': [125, 135, 145, 150]},
    'MLB': {
        'spread': [],                        # runline is fixed at -1.5 / +1.5
        'total': [6.5, 7, 7.5, 8, 8.5, 9]     # updated with common clustering points
    },
    'NHL': {'spread': [], 'total': [5.5, 6, 6.5, 7]},
}


def was_line_resistance_broken(open_val, close_val, key_levels, market_type):
    if pd.isna(open_val) or pd.isna(close_val):
        return 0, []

    # Use absolute values for spread resistance checks
    if market_type == 'spread':
        open_val, close_val = abs(open_val), abs(close_val)
        key_levels = [abs(k) for k in key_levels]

    crossed = [
        key for key in key_levels
        if (open_val < key < close_val) or (close_val < key < open_val)
    ]

    return int(bool(crossed)), crossed



def compute_line_resistance_flag(df, source='moves'):
    # Normalize sport keys using SPORT_ALIAS
    df['Sport'] = df['Sport'].str.upper().map(SPORT_ALIAS).fillna(df['Sport'].str.upper())

    def get_key_levels(sport, market):
        if not sport or not market:
            return []
        return KEY_LINE_RESISTANCE.get(sport, {}).get(market.lower(), [])

    def get_opening_line(row):
        return row.get('Open_Value') if source == 'moves' else row.get('First_Line_Value')

    def apply_resistance_logic(row):
        open_val = get_opening_line(row)
        close_val = row.get('Value')
        market_type = row.get('Market', '').lower()
        key_levels = get_key_levels(row.get('Sport', ''), market_type)

        flag, levels_crossed = was_line_resistance_broken(open_val, close_val, key_levels, market_type)
        return pd.Series({
            'Was_Line_Resistance_Broken': flag,
            'Line_Resistance_Crossed_Levels': levels_crossed,
            'Line_Resistance_Crossed_Count': len(levels_crossed)
        })

    resistance_flags = df.apply(apply_resistance_logic, axis=1)
    df = pd.concat([df, resistance_flags], axis=1)
    return df


def compute_sharp_magnitude_by_time_bucket(df_all_snapshots):
    results = []

    grouped = df_all_snapshots.groupby(['Game_Key', 'Market', 'Outcome', 'Bookmaker'])

    for (gk, market, outcome, book), group in grouped:
        logging.debug(f"📊 Group: Game={gk}, Market={market}, Outcome={outcome}, Book={book}")
        logging.debug(f"📋 Columns in group: {list(group.columns)}")

        game_start = (
            group['Game_Start'].dropna().iloc[0]
            if 'Game_Start' in group and not group['Game_Start'].isnull().all()
            else None
        )

        entries = list(zip(
            group['Limit'],
            group['Value'],
            group['Snapshot_Timestamp'],
            [game_start] * len(group)  # Apply game_start to all entries
        ))

        open_val = (
            group.sort_values('Snapshot_Timestamp')['Value']
            .dropna().iloc[0]
            if not group['Value'].isnull().all()
            else None
        )

        metrics = compute_sharp_metrics(entries, open_val, market, outcome)
        metrics.update({
            'Game_Key': gk,
            'Market': market,
            'Outcome': outcome,
            'Bookmaker': book
        })
        results.append(metrics)

    return pd.DataFrame(results)
        
def add_minutes_to_game(df):
    df = df.copy()

    if 'Commence_Hour' not in df.columns or 'Snapshot_Timestamp' not in df.columns:
        logger.warning("⏳ Skipping time-to-game calculation — missing 'Commence_Hour' or 'Snapshot_Timestamp'")
        df['Minutes_To_Game'] = None
        df['Timing_Tier'] = None
        return df

    df['Commence_Hour'] = pd.to_datetime(df['Commence_Hour'], utc=True, errors='coerce')
    df['Snapshot_Timestamp'] = pd.to_datetime(df['Snapshot_Timestamp'], utc=True, errors='coerce')

    df['Minutes_To_Game'] = (
        (df['Commence_Hour'] - df['Snapshot_Timestamp'])
        .dt.total_seconds() / 60
    ).clip(lower=0)

    df['Timing_Tier'] = pd.cut(
        df['Minutes_To_Game'],
        bins=[0, 60, 360, 1440, float('inf')],  # <1h, 1–6h, 6–24h, >24h
        labels=[
            '🔥 Late (<1h)',
            '⚠️ Mid (1–6h)',
            '⏳ Early (6–24h)',
            '🧊 Very Early (>24h)',
        ],
        right=False
    )

    return df

from scipy.stats import zscore

def add_line_and_crossmarket_features(df):
    # === Normalize sport/market
    SPORT_ALIAS = {
        'MLB': 'MLB', 'NFL': 'NFL', 'CFL': 'CFL',
        'WNBA': 'WNBA', 'NBA': 'NBA', 'NCAAF': 'NCAAF', 'NCAAB': 'NCAAB'
    }
    df['Sport_Norm'] = df['Sport'].map(SPORT_ALIAS).fillna(df['Sport'])
    df['Market_Norm'] = df['Market'].str.lower()

    # === Absolute line + odds movement
    df['Abs_Line_Move_From_Opening'] = (df['Value'] - df['Open_Value']).abs()
    df['Odds_Shift'] = df['Odds_Price'] - df['Open_Odds']
    df['Implied_Prob'] = df['Odds_Price'].apply(implied_prob)
    df['First_Imp_Prob'] = df['Open_Odds'].apply(implied_prob)
    df['Implied_Prob_Shift'] = df['Implied_Prob'] - df['First_Imp_Prob']

    # === Directional movement
    df['Line_Moved_Toward_Team'] = np.where(
        ((df['Value'] > df['Open_Value']) & (df['Is_Favorite_Bet'] == 1)) |
        ((df['Value'] < df['Open_Value']) & (df['Is_Favorite_Bet'] == 0)),
        1, 0
    )

    df['Line_Moved_Away_From_Team'] = np.where(
        ((df['Value'] < df['Open_Value']) & (df['Is_Favorite_Bet'] == 1)) |
        ((df['Value'] > df['Open_Value']) & (df['Is_Favorite_Bet'] == 0)),
        1, 0
    )

    # === Percent line move (totals only)
    df['Pct_Line_Move_From_Opening'] = np.where(
        (df['Market_Norm'] == 'total') & (df['Open_Value'].abs() > 0),
        df['Abs_Line_Move_From_Opening'] / df['Open_Value'].abs(),
        np.nan
    )

    df['Pct_Line_Move_Bin'] = pd.cut(
        df['Pct_Line_Move_From_Opening'],
        bins=[-np.inf, 0.0025, 0.005, 0.01, 0.02, np.inf],
        labels=['<0.25%', '0.5%', '1%', '2%', '2%+']
    )

    # === Disable flags for unsupported markets
    df['Disable_Line_Move_Features'] = np.where(
        ((df['Sport_Norm'] == 'MLB') & (df['Market_Norm'] == 'spread')) |
        (df['Market_Norm'].isin(['h2h', 'moneyline'])),
        1, 0
    )

    # === Z-score based overmove detection
    df['Abs_Line_Move_Z'] = (
        df.groupby(['Sport_Norm', 'Market_Norm'])['Abs_Line_Move_From_Opening']
        .transform(lambda x: zscore(x.fillna(0), ddof=0))
    ).clip(-5, 5)

    df['Pct_Line_Move_Z'] = (
        df.groupby(['Sport_Norm'])['Pct_Line_Move_From_Opening']
        .transform(lambda x: zscore(x.fillna(0), ddof=0))
    ).clip(-5, 5)

    df['Implied_Prob_Shift_Z'] = (
        df.groupby(['Sport_Norm', 'Market_Norm'])['Implied_Prob_Shift']
        .transform(lambda x: zscore(x.fillna(0), ddof=0))
    ).clip(-5, 5)

    df['Potential_Overmove_Flag'] = np.where(
        (df['Market_Norm'] == 'spread') &
        (df['Line_Moved_Toward_Team'] == 1) &
        (df['Abs_Line_Move_Z'] >= 2) &
        (df['Disable_Line_Move_Features'] == 0),
        1, 0
    )

    df['Potential_Overmove_Total_Pct_Flag'] = np.where(
        (df['Market_Norm'] == 'total') &
        (df['Line_Moved_Toward_Team'] == 1) &
        (df['Pct_Line_Move_Z'] >= 2) &
        (df['Disable_Line_Move_Features'] == 0),
        1, 0
    )

    df['Potential_Odds_Overmove_Flag'] = np.where(
        (df['Implied_Prob_Shift_Z'] >= 2),
        1, 0
    )

    # === Cross-market alignment
    df['Spread_Implied_Prob'] = df['Spread_Odds'].apply(implied_prob)
    df['H2H_Implied_Prob'] = df['H2H_Odds'].apply(implied_prob)
    df['Total_Implied_Prob'] = df['Total_Odds'].apply(implied_prob)

    df['Spread_vs_H2H_Aligned'] = np.where(
        (df['Market_Norm'] == 'spread') &
        (df['Value'] < 0) &  # favorite
        (df.get('H2H_Implied_Prob', 0) > 0.5),
        1, 0
    )
    df['Total_vs_Spread_Contradiction'] = (
        (df['Spread_Implied_Prob'] > 0.55) &
        (df['Total_Implied_Prob'] < 0.48)
    ).astype(int)

    df['Spread_vs_H2H_ProbGap'] = df['Spread_Implied_Prob'] - df['H2H_Implied_Prob']
    df['Total_vs_H2H_ProbGap'] = df['Total_Implied_Prob'] - df['H2H_Implied_Prob']
    df['Total_vs_Spread_ProbGap'] = df['Total_Implied_Prob'] - df['Spread_Implied_Prob']

    df['CrossMarket_Prob_Gap_Exists'] = (
        (df['Spread_vs_H2H_ProbGap'].abs() > 0.05) |
        (df['Total_vs_Spread_ProbGap'].abs() > 0.05)
    ).astype(int)

    return df
def compute_small_book_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    SMALL_LIMIT_BOOKS = ['betus', 'mybookie', 'betfair_eu', 'betfair_uk', 'lowvig', 'betonline', 'matchbook']
    df = df.copy()

    df['Bookmaker_Norm'] = df['Book']


    df['Is_Small_Limit_Book'] = df['Bookmaker_Norm'].isin(SMALL_LIMIT_BOOKS).astype(int)

    if 'Limit' not in df.columns:
        df['Limit'] = 0
    else:
        df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)

    try:
        agg = (
            df[df['Is_Small_Limit_Book'] == 1]
            .groupby(['Game_Key', 'Outcome'])
            .agg(
                SmallBook_Total_Limit=('Limit', 'sum'),
                SmallBook_Max_Limit=('Limit', 'max'),
                SmallBook_Min_Limit=('Limit', 'min'),
                SmallBook_Limit_Count=('Limit', 'count')
            )
            .reset_index()
        )
    except Exception:
        agg = pd.DataFrame(columns=[
            'Game_Key', 'Outcome', 'SmallBook_Total_Limit',
            'SmallBook_Max_Limit', 'SmallBook_Min_Limit', 'SmallBook_Limit_Count'
        ])

    df = df.merge(agg, on=['Game_Key', 'Outcome'], how='left')

    for col in ['SmallBook_Total_Limit', 'SmallBook_Max_Limit', 'SmallBook_Min_Limit']:
        df[col] = pd.to_numeric(
            df[col] if col in df.columns else pd.Series(0, index=df.index),
            errors='coerce'
        ).fillna(0)


    df['SmallBook_Limit_Skew'] = df['SmallBook_Max_Limit'] - df['SmallBook_Min_Limit']
    df['SmallBook_Heavy_Liquidity_Flag'] = (df['SmallBook_Total_Limit'] > 1000).astype(int)
    df['SmallBook_Limit_Skew_Flag'] = (df['SmallBook_Limit_Skew'] > 500).astype(int)

    return df
          
def hydrate_inverse_rows_from_snapshot(df_inverse: pd.DataFrame, df_all_snapshots: pd.DataFrame) -> pd.DataFrame:
    df = df_inverse.copy()

    # Normalize bookmaker names
    df['Bookmaker'] = df.apply(lambda row: normalize_book_name(row.get('Bookmaker'), row.get('Book')), axis=1)
    df_all_snapshots['Bookmaker'] = df_all_snapshots.apply(lambda row: normalize_book_name(row.get('Bookmaker'), row.get('Book')), axis=1)

    # ⛔️ DO NOT FLIP OUTCOMES — inverse rows are already constructed correctly
    # df['Outcome'] = df.apply(get_inverse_label, axis=1)
    # df['Outcome_Norm'] = df['Outcome']

    # Ensure Commence_Hour is normalized
    df['Commence_Hour'] = pd.to_datetime(df['Commence_Hour'], utc=True, errors='coerce').dt.floor('h')
    df['Team_Key'] = (
        df['Home_Team_Norm'] + "_" +
        df['Away_Team_Norm'] + "_" +
        df['Commence_Hour'].astype(str) + "_" +
        df['Market'] + "_" +
        df['Outcome']
    )

    # Build snapshot Team_Key
    df_all_snapshots['Commence_Hour'] = pd.to_datetime(df_all_snapshots['Game_Start'], utc=True, errors='coerce').dt.floor('h')
    df_all_snapshots['Team_Key'] = (
        df_all_snapshots['Home_Team_Norm'] + "_" +
        df_all_snapshots['Away_Team_Norm'] + "_" +
        df_all_snapshots['Commence_Hour'].astype(str) + "_" +
        df_all_snapshots['Market'] + "_" +
        df_all_snapshots['Outcome']
    )

    # Merge latest snapshot per (Team_Key, Bookmaker)
    df_latest = (
        df_all_snapshots
        .sort_values('Snapshot_Timestamp', ascending=False)
        .drop_duplicates(subset=['Team_Key', 'Bookmaker'])
        [['Team_Key', 'Bookmaker', 'Value', 'Odds_Price', 'Limit']]
        .rename(columns={
            'Value': 'Value_opponent',
            'Odds_Price': 'Odds_Price_opponent',
            'Limit': 'Limit_opponent'
        })
    )

    df = df.merge(df_latest, on=['Team_Key', 'Bookmaker'], how='left')

    # Only overwrite if snapshot value is available
    for col in ['Value', 'Odds_Price', 'Limit']:
        opp_col = f"{col}_opponent"
        if opp_col in df.columns:
            df[col] = np.where(df[opp_col].notna(), df[opp_col], df[col])

    df.drop(columns=['Value_opponent', 'Odds_Price_opponent', 'Limit_opponent'], errors='ignore', inplace=True)
    return df
def fallback_flip_inverse_rows(df_inverse: pd.DataFrame) -> pd.DataFrame:
    # Only flip if Value is missing
    missing_value_mask = df_inverse['Value'].isnull()

    df_to_flip = df_inverse[missing_value_mask].copy()
    if df_to_flip.empty:
        return df_inverse  # Nothing to flip

    logger.info(f"🔁 Fallback flipping {len(df_to_flip)} inverse rows missing value...")

    if 'Market' in df_to_flip.columns:
        df_to_flip.loc[df_to_flip['Market'] == 'spreads', 'Value'] *= -1
        df_to_flip.loc[df_to_flip['Market'] == 'totals', 'Outcome_Norm'] = df_to_flip['Outcome_Norm'].map(
            {'over': 'under', 'under': 'over'}
        )
        df_to_flip.loc[df_to_flip['Market'] == 'totals', 'Outcome'] = df_to_flip['Outcome_Norm']
    
    df_inverse.update(df_to_flip)
    return df_inverse

def get_opening_snapshot(df_all_snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the first valid snapshot per (Game_Key, Market, Outcome, Bookmaker) with:
    - Open_Value
    - Open_Odds
    - First_Imp_Prob
    - Opening_Limit (optional, only if 'Limit' exists and is not null)
    """
    if df_all_snapshots.empty:
        logging.warning("⚠️ df_all_snapshots is empty — returning empty opening snapshot")
        return pd.DataFrame(columns=['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Open_Value', 'Open_Odds', 'First_Imp_Prob', 'Opening_Limit'])

    required_cols = ['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Snapshot_Timestamp', 'Value', 'Odds_Price']
    for col in required_cols:
        if col not in df_all_snapshots.columns:
            raise ValueError(f"Missing required column in df_all_snapshots: {col}")

    merge_keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
    df = df_all_snapshots.copy()
    for col in merge_keys:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df['Snapshot_Timestamp'] = pd.to_datetime(df['Snapshot_Timestamp'], errors='coerce', utc=True)
    df['Odds_Price'] = pd.to_numeric(df['Odds_Price'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    if 'Limit' in df.columns:
        df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce')

    # Compute Implied_Prob if missing
    if 'Implied_Prob' not in df.columns:
        df['Implied_Prob'] = np.nan
    df['Implied_Prob'] = df['Implied_Prob'].fillna(df['Odds_Price'].apply(implied_prob))

    # Drop bad rows
    df = df.dropna(subset=['Snapshot_Timestamp', 'Value', 'Odds_Price'])

    # === Get first snapshot per outcome
    df_first = (
        df.sort_values('Snapshot_Timestamp')
        .drop_duplicates(subset=merge_keys, keep='first')
        [merge_keys + ['Value', 'Odds_Price', 'Implied_Prob'] + (['Limit'] if 'Limit' in df.columns else [])]
    )

    df_first = df_first.rename(columns={
        'Value': 'Open_Value',
        'Odds_Price': 'Open_Odds',
        'Implied_Prob': 'First_Imp_Prob',
        'Limit': 'Opening_Limit' if 'Limit' in df_first.columns else None
    })

    return df_first


          
def apply_blended_sharp_score(df, trained_models, df_all_snapshots=None, weights=None):
    logger.info("🛠️ Running `apply_blended_sharp_score()`")
    scored_all = []
    total_start = time.time()
    df = df.copy()
    df['Market'] = df['Market'].astype(str).str.lower().str.strip()
    df['Is_Sharp_Book'] = df['Bookmaker'].astype(str).str.lower().str.strip().isin(SHARP_BOOKS).astype(int)
    df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()

    if 'Game_Start' in df.columns:
        df['Event_Date'] = pd.to_datetime(df['Game_Start'], errors='coerce').dt.date
    else:
        df['Event_Date'] = pd.NaT

    # === Normalize relevant fields
    merge_keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
    for col in merge_keys:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # === Load full snapshot history if needed
    if df_all_snapshots is None:
        logger.warning("⚠️ df_all_snapshots not passed — loading fallback from BigQuery")
        df_all_snapshots = read_recent_sharp_master_cached(hours=120)
    else:
        logger.info(f"🧪 Using df_all_snapshots from caller — {len(df_all_snapshots)} rows")

    # ✅ Only concat rows where Value is not null and Outcome is non-canonical
    # ✅ Normalize snapshot history
    for col in merge_keys:
        df_all_snapshots[col] = df_all_snapshots[col].astype(str).str.strip().str.lower()
    
    # ✅ Add inverse rows back into snapshots (so both sides can be hydrated)
    inverse_snapshot_rows = df[df['Was_Canonical'] == False][
        merge_keys + ['Value', 'Odds_Price', 'Snapshot_Timestamp']
    ]
    
    df_all_snapshots = pd.concat([
        df_all_snapshots,
        inverse_snapshot_rows
    ], ignore_index=True).drop_duplicates(subset=merge_keys + ['Snapshot_Timestamp'])
    
    # ✅ Clean and compute Implied_Prob
    df_all_snapshots['Odds_Price'] = pd.to_numeric(df_all_snapshots['Odds_Price'], errors='coerce')
    df_all_snapshots['Value'] = pd.to_numeric(df_all_snapshots['Value'], errors='coerce')
    df_all_snapshots['Implied_Prob'] = df_all_snapshots.get('Implied_Prob')
    df_all_snapshots['Implied_Prob'] = df_all_snapshots['Implied_Prob'].fillna(
        df_all_snapshots['Odds_Price'].apply(implied_prob)
    )
    logger.info(f"🔍 Columns in df_all_snapshots before opening snapshot: {df_all_snapshots.columns.tolist()}")
    logger.info(f"📌 Sample snapshot rows:\n{df_all_snapshots[merge_keys + ['Value', 'Odds_Price']].dropna().head(20)}")
    df_open = get_opening_snapshot(df_all_snapshots)
    

    logger.info(f"📦 get_opening_snapshot() returned {len(df_open)} rows")
    logger.info(f"🧾 df_open columns: {df_open.columns.tolist()}")
    logger.info(f"🔍 df_open sample:\n{df_open.head(20)}")
    merge_keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
    for col in merge_keys:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df_open[col] = df_open[col].astype(str).str.strip().str.lower()
    logger.info(f"🔍 Columns BEFORE merge: {df.columns.tolist()}")
    logger.info(f"🔍 df_open Columns: {df_open.columns.tolist()}")
    logger.info(f"📌 Sample df rows BEFORE merge:\n{df[merge_keys + ['Value', 'Odds_Price']].head(5)}")
    logger.info(f"📌 Sample df_open rows:\n{df_open.head(5)}")
    
      # --- 1) Normalize merge keys ---
    merge_keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
    for c in merge_keys:
        df[c] = df[c].astype(str).str.strip().str.lower()
    
    # --- 2) Build a clean opening snapshot with final names (no suffixes anywhere) ---
    df_open_raw = get_opening_snapshot(df_all_snapshots)  # must include the same merge keys
    for c in merge_keys:
        df_open_raw[c] = df_open_raw[c].astype(str).str.strip().str.lower()
    
    # Whatever get_opening_snapshot returns, force the final column names here
    rename_map = {
        'Value': 'Open_Value',
        'Odds_Price': 'Open_Odds',
        'Implied_Prob': 'First_Imp_Prob',
        'Opening_Limit': 'Opening_Limit',
    }
    df_open = df_open_raw.rename(columns={k: v for k, v in rename_map.items() if k in df_open_raw.columns})
    
    # keep ONLY the columns we need to merge
    needed_open_cols = merge_keys + ['Open_Value', 'Open_Odds', 'First_Imp_Prob', 'Opening_Limit']
    df_open = df_open[[c for c in needed_open_cols if c in df_open.columns]].copy()
    
    # --- 3) Merge (no suffixes expected now) ---
    df = df.merge(df_open, how='left', on=merge_keys)
    
    # --- 4) Defensive guarantees BEFORE any use downstream ---
    # If Open_Value is absent (column missing) create it; if present but NaN, fill from current Value.
    if 'Open_Value' not in df.columns:
        df['Open_Value'] = df['Value']
    else:
        df['Open_Value'] = df['Open_Value'].fillna(df['Value'])
    
    if 'Open_Odds' not in df.columns:
        df['Open_Odds'] = df.get('Odds_Price')
    else:
        df['Open_Odds'] = df['Open_Odds'].fillna(df.get('Odds_Price'))
    
    # First_Imp_Prob final fallback
    if 'First_Imp_Prob' not in df.columns:
        df['First_Imp_Prob'] = np.nan
    if df['First_Imp_Prob'].isna().any():
        if 'Odds_Price' in df.columns:
            df['First_Imp_Prob'] = df['First_Imp_Prob'].fillna(df['Odds_Price'].apply(implied_prob))
        else:
            df['First_Imp_Prob'] = df['First_Imp_Prob'].fillna(0.5)
    
    # Opening_Limit can be legitimately missing for some books; ensure the column exists
    if 'Opening_Limit' not in df.columns:
        df['Opening_Limit'] = np.nan
    
    # --- 5) Hard assertions with context (better than mystery KeyError later) ---
    missing_cols = [c for c in ['Open_Value','Open_Odds','First_Imp_Prob','Opening_Limit'] if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Opening merge failed to create columns: {missing_cols}. "
                           f"df_open columns: {list(df_open.columns)}")
    
    # Optional diagnostics
    logger.info("📊 Missing Open_Value: %.2f%%", 100*df['Open_Value'].isna().mean())
    logger.info("📊 Missing Open_Odds: %.2f%%", 100*df['Open_Odds'].isna().mean())
    logger.info("📊 Missing First_Imp_Prob: %.2f%%", 100*df['First_Imp_Prob'].isna().mean())
    logger.info("📊 Missing Opening_Limit: %.2f%%", 100*df['Opening_Limit'].isna().mean())
        
   f.columns]
    if missing_cols:
        logger.warning(f"⚠️ Missing columns after merge: {missing_cols}")
    
    # ✅ Optional: Open_Book_Value (if still needed for reference or diagnostics)
    df_open_book = (
        df_open
        .dropna(subset=['Open_Value'])
        .drop_duplicates(subset=merge_keys)
        [merge_keys + ['Open_Value']]
        .rename(columns={'Open_Value': 'Open_Book_Value'})
    )


    # === Step 4: Compute extremes even with one row
    df_extremes = (
        df_all_snapshots
        .dropna(subset=['Value', 'Odds_Price'], how='all')
        .groupby(merge_keys)[['Value', 'Odds_Price']]
        .agg(
            Max_Value=('Value', 'max'),
            Min_Value=('Value', 'min'),
            Max_Odds=('Odds_Price', 'max'),
            Min_Odds=('Odds_Price', 'min')
        )
        .reset_index()
    )

    for col in merge_keys:
        df_extremes[col] = df_extremes[col].astype(str).str.strip().str.lower()

    # === Final merges
    df = df.merge(df_open, on=merge_keys, how='left')
    df = df.merge(df_open_book, on=merge_keys, how='left')
    df = df.merge(df_extremes, on=merge_keys, how='left')
    # === Fill fallback values where open/extreme data is missing

    # 🛡️ Ensure Implied_Prob exists in df for fallback to work
    if 'Implied_Prob' not in df.columns:
        df['Odds_Price'] = pd.to_numeric(df['Odds_Price'], errors='coerce')
        df['Implied_Prob'] = df['Odds_Price'].apply(implied_prob)
      
    df['Open_Value'] = df['Open_Value'].fillna(df['Value'])
    df['Open_Odds'] = df['Open_Odds'].fillna(df['Odds_Price'])
    df['First_Imp_Prob'] = df['First_Imp_Prob'].fillna(df['Implied_Prob'])
    
    df['Open_Book_Value'] = df['Open_Book_Value'].fillna(df['Value'])
    
    df['Max_Value'] = df['Max_Value'].fillna(df['Value'])
    df['Min_Value'] = df['Min_Value'].fillna(df['Value'])
    df['Max_Odds'] = df['Max_Odds'].fillna(df['Odds_Price'])
    df['Min_Odds'] = df['Min_Odds'].fillna(df['Odds_Price'])

    # === Diagnostics
    logger.info("🧪 Merge keys sample from df:")
    logger.info(df[merge_keys].drop_duplicates().head().to_string(index=False))

    logger.info("🧪 Merge keys sample from df_open_rows:")
    logger.info(df_open_rows[merge_keys].drop_duplicates().head().to_string(index=False))

    try:
        logger.info("🧪 Sample of enriched df after merge:")
        logger.info(df[[
            'Game_Key', 'Market', 'Outcome', 'Bookmaker',
            'Odds_Price', 'Value',
            'Open_Odds', 'Open_Value', 'First_Imp_Prob', 'Open_Book_Value',
            'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds'
        ]].drop_duplicates().sort_values('Game_Key').head().to_string(index=False))
    except Exception as e:
        logger.warning(f"⚠️ Failed to print preview: {e}")



    # === Compute shifts
    # === Compute Odds_Shift as change in implied probability
    df['Odds_Shift'] = (
        df['Odds_Price'].apply(implied_prob) -
        df['Open_Odds'].apply(implied_prob)
    ) * 100  # Optional: express as perce
    
    df['Implied_Prob_Shift'] = pd.to_numeric(df['Implied_Prob'], errors='coerce') - pd.to_numeric(df['First_Imp_Prob'], errors='coerce')
    # Compute deltas
    df['Line_Delta'] = pd.to_numeric(df['Value'], errors='coerce') - pd.to_numeric(df['Open_Value'], errors='coerce')
    
     # === Enrich with resistance, timing, etc.
    df = compute_line_resistance_flag(df, source='moves')
    df = add_minutes_to_game(df)
    # === Clean columns
    df.drop(columns=['First_Imp_Prob'], inplace=True, errors='ignore')

    # === Delta and reversal flags
    df['Delta'] = df['Value'] - df['Open_Value']
    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)
    df['Line_Magnitude_Abs'] = df['Line_Delta'].abs()
    df['Line_Move_Magnitude'] = df['Line_Delta'].abs()
    def compute_value_reversal(df, market_col='Market'):
        is_spread = df[market_col].str.lower().str.contains('spread')
        is_total = df[market_col].str.lower().str.contains('total')
        is_h2h = df[market_col].str.lower().str.contains('h2h')
    
        df['Value_Reversal_Flag'] = np.where(
            is_spread,
            (
                ((df['Open_Value'] < 0) & (df['Value'] > df['Open_Value']) & (df['Value'] == df['Max_Value'])) |
                ((df['Open_Value'] > 0) & (df['Value'] < df['Open_Value']) & (df['Value'] == df['Min_Value']))
            ).astype(int),
            np.where(
                is_total,
                (
                    ((df['Outcome_Norm'] == 'over') & (df['Value'] > df['Open_Value']) & (df['Value'] == df['Max_Value'])) |
                    ((df['Outcome_Norm'] == 'under') & (df['Value'] < df['Open_Value']) & (df['Value'] == df['Min_Value']))
                ).astype(int),
                np.where(
                    is_h2h,
                    (
                        ((df['Value'] > df['Open_Value']) & (df['Value'] == df['Max_Value'])) |
                        ((df['Value'] < df['Open_Value']) & (df['Value'] == df['Min_Value']))
                    ).astype(int),
                    0
                )
            )
        )
        return df


    
 
  
    def compute_odds_reversal(df, prob_threshold=0.05):
        df = df.copy()
    
        print("✅ compute_odds_reversal: reusing existing implied prob columns")
    
        # Market flags
        is_spread = df['Market'].str.lower().str.contains('spread', na=False)
        is_total = df['Market'].str.lower().str.contains('total', na=False)
        is_h2h = df['Market'].str.lower().str.contains('h2h', na=False)
    
        # Ensure all required columns exist
        if 'Implied_Prob' not in df.columns:
            df['Implied_Prob'] = df['Odds_Price'].apply(implied_prob)
        if 'First_Imp_Prob' not in df.columns:
            df['First_Imp_Prob'] = df['Implied_Prob']
        if 'Min_Odds' not in df.columns or 'Max_Odds' not in df.columns:
            df['Min_Odds'] = df['Odds_Price']
            df['Max_Odds'] = df['Odds_Price']
    
        # H2H reversal using stored implied probs
        min_prob = df['Min_Odds'].apply(implied_prob)
        max_prob = df['Max_Odds'].apply(implied_prob)
    
        valid_mask = (
            pd.notna(df['First_Imp_Prob']) &
            pd.notna(df['Implied_Prob']) &
            pd.notna(min_prob) &
            pd.notna(max_prob)
        )
    
        h2h_flag = np.zeros(len(df), dtype=int)
        h2h_flag[valid_mask] = (
            ((df['First_Imp_Prob'][valid_mask] > min_prob[valid_mask]) &
             (df['Implied_Prob'][valid_mask] <= min_prob[valid_mask] + 1e-5)) |
            ((df['First_Imp_Prob'][valid_mask] < max_prob[valid_mask]) &
             (df['Implied_Prob'][valid_mask] >= max_prob[valid_mask] - 1e-5))
        ).astype(int)
    
        # Reuse Implied_Prob_Shift or compute if missing
        if 'Implied_Prob_Shift' not in df.columns:
            df['Implied_Prob_Shift'] = df['Implied_Prob'] - df['First_Imp_Prob']
    
        abs_shift = df['Implied_Prob_Shift'].abs()
        spread_total_flag = (abs_shift >= prob_threshold).astype(int)
    
        # Final flag
        df['Odds_Reversal_Flag'] = np.where(
            is_h2h,
            h2h_flag,
            np.where(is_spread | is_total, spread_total_flag, 0)
        )
    
        # Optional diagnostics
        df['Abs_Odds_Prob_Move'] = abs_shift
    
        return df

        
    # === Compute Openers BEFORE creating df_history_sorted

    # === Additional sharp flags
    df['Limit_Jump'] = (df['Limit'] >= 2500).astype(int)
    df['Sharp_Timing'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour.apply(
        lambda h: (
            1.0 if 0 <= h <= 5 else
            0.9 if 6 <= h <= 11 else
            0.5 if 12 <= h <= 15 else
            0.2
        ) if pd.notnull(h) else 0.0
    )
    df['Limit_NonZero'] = df['Limit'].where(df['Limit'] > 0)
    df['Limit_Max'] = df.groupby(['Game', 'Market'])['Limit_NonZero'].transform('max')
    df['Limit_Min'] = df.groupby(['Game', 'Market'])['Limit_NonZero'].transform('min')
  
    
    # ✅ FIXED
    market_leader_flags = detect_market_leaders(df_all_snapshots, SHARP_BOOKS, REC_BOOKS)
  
    df = df.merge(
        market_leader_flags[['Game', 'Market', 'Outcome', 'Book', 'Market_Leader']],
        on=['Game', 'Market', 'Outcome', 'Book'],
        how='left'
    )
    
    # === Flag Pinnacle no-move behavior
    df['Is_Pinnacle'] = df['Book'] == 'pinnacle'
    df['LimitUp_NoMove_Flag'] = (
        (df['Is_Pinnacle']) &
        (df['Limit'] >= 2500) &
        (df['Value'] == df['Open_Value'])
    ).astype(int)
    
    # === Cross-market support (optional)
    df = detect_cross_market_sharp_support(df, SHARP_BOOKS)
    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Unique_Sharp_Books'] = df['Unique_Sharp_Books'].fillna(0).astype(int)
    df['LimitUp_NoMove_Flag'] = df['LimitUp_NoMove_Flag'].fillna(False).astype(int)
    df['Market_Leader'] = df['Market_Leader'].fillna(False).astype(int)
    

    # === Confidence scores and tiers
    try:
        # Only assign confidence scores if weights are provided
        if weights:
            df = assign_confidence_scores(df, weights)
        else:
            logging.warning("⚠️ Skipping confidence scoring — 'weights' not defined.")
    except Exception as e:
        logging.warning(f"⚠️ Failed to assign confidence scores: {e}")
    # === Patch derived fields before BigQuery write ===
   
    try:
        # Line_Delta: Value - Open_Value
        # ✅ corrected (aligned with directional logic)
        
        # Line magnitudes
        
    
        # High Limit flag
        df['High_Limit_Flag'] = (pd.to_numeric(df['Sharp_Limit_Total'], errors='coerce') >= 10000).astype(float)
    
        # Home team indicator
        df['Is_Home_Team_Bet'] = (df['Outcome'].str.lower() == df['Home_Team_Norm'].str.lower()).astype(float)
    
        # Favorite indicator
        df['Is_Favorite_Bet'] = (pd.to_numeric(df['Value'], errors='coerce') < 0).astype(float)
    
        # Direction alignment: market-based only
        df['Direction_Aligned'] = np.where(
            df['Line_Delta'] > 0, 1,
            np.where(df['Line_Delta'] < 0, 0, np.nan)
        ).astype(float)
    except Exception as e:
        logging.warning(f"⚠️ Failed to compute sharp move diagnostic columns: {e}")
    team_feature_map = None
    for bundle in trained_models.values():
        team_feature_map = bundle.get('team_feature_map')
        if team_feature_map is not None:
            break  # Use the first one found
    logger.info(f"✅ Snapshot enrichment complete — rows: {len(df)}")
    logger.info(f"📊 Columns present after enrichment: {df.columns.tolist()}")
    # === Cross-Market Odds Pivot
    odds_pivot = (
        df
        .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome'])
        .pivot_table(index='Game_Key', columns='Market', values='Odds_Price')
        .rename(columns={
            'spreads': 'Spread_Odds',
            'totals': 'Total_Odds',
            'h2h': 'H2H_Odds'
        })
        .reset_index()
    )
    df = df.merge(odds_pivot, on='Game_Key', how='left')

   
    
    if team_feature_map is not None and not team_feature_map.empty:
        logger.info("📊 Team Historical Performance Metrics (Hit Rate and Avg Model Prob):")
        sample_log = team_feature_map.head(40).to_string(index=False)
        logger.info(f"\n{sample_log}")
    else:
        logger.warning("⚠️ team_feature_map is empty or missing.")
    
    for market_type, bundle in trained_models.items():
        try:
            model = bundle.get('model')
            iso = bundle.get('calibrator')
            team_feature_map = bundle.get('team_feature_map')

            if model is None or iso is None:
                logger.warning(f"⚠️ Skipping {market_type.upper()} — model or calibrator missing")
                continue

            df_market = df[df['Market'] == market_type].copy()
            if df_market.empty:
                logger.warning(f"⚠️ No rows to score for {market_type.upper()}")
                continue

            df_market['Outcome'] = df_market['Outcome'].astype(str).str.lower().str.strip()
            df_market['Outcome_Norm'] = df_market['Outcome']
            df_market['Value'] = pd.to_numeric(df_market['Value'], errors='coerce')
            df_market['Commence_Hour'] = pd.to_datetime(df_market['Game_Start'], utc=True, errors='coerce').dt.floor('h')
            df_market['Odds_Price'] = pd.to_numeric(df_market.get('Odds_Price'), errors='coerce')
            # === Compute implied probabilities and shifts using global `calc_implied_prob`
            df_market['Implied_Prob'] = df_market['Odds_Price'].apply(calc_implied_prob)
           
            df_market['Game_Key'] = (
                df_market['Home_Team_Norm'] + "_" +
                df_market['Away_Team_Norm'] + "_" +
                df_market['Commence_Hour'].astype(str) + "_" +
                df_market['Market'] + "_" +
                df_market['Outcome_Norm']
            )

            df_market['Game_Key_Base'] = (
                df_market['Home_Team_Norm'] + "_" +
                df_market['Away_Team_Norm'] + "_" +
                df_market['Commence_Hour'].astype(str) + "_" +
                df_market['Market']
            )

            sided_games_check = (
                df_market.groupby(['Game_Key_Base'])['Outcome']
                .nunique()
                .reset_index(name='Num_Sides')
            )

            valid_games = sided_games_check[sided_games_check['Num_Sides'] >= 2]['Game_Key_Base']
            df_market = df_market[df_market['Game_Key_Base'].isin(valid_games)].copy()
    
            df_market = df_market[df_market['Game_Key_Base'].isin(valid_games)].copy()
            # ✅ NOW apply canonical filtering based on market_type
            if market_type == "spreads":
                
                # ✅ Pick canonical row with most negative value per Game_Key_Base
                df_market = df_market[df_market['Value'].notna()]
                df_canon = df_market[df_market['Value'] < 0].copy()
                df_full_market = df_market.copy()
                missing_over_books = set(df_market['Bookmaker'].unique()) - set(df_canon['Bookmaker'].unique())
                if missing_over_books:
                    logger.warning(f"⚠️ These books had NO over rows and were skipped from canonical totals scoring: {missing_over_books}")

            elif market_type == "h2h":
                
                df_market = df_market[df_market['Value'].notna()]
                df_canon = df_market[df_market['Value'] < 0].copy()
                df_full_market = df_market.copy()
               
    
            elif market_type == "totals":
                df_canon = df_market[df_market['Outcome_Norm'] == 'over'].copy()
                df_full_market = df_market.copy()

            dedup_keys = ['Game_Key', 'Market', 'Bookmaker', 'Outcome']
            pre_dedup_canon = len(df_canon)
            df_canon = df_canon.drop_duplicates(subset=dedup_keys)
            post_dedup_canon = len(df_canon)
            
            if team_feature_map is not None and not team_feature_map.empty:
                df_canon['Team'] = df_canon['Outcome_Norm'].str.lower().str.strip()
                df_canon = df_canon.merge(team_feature_map, on='Team', how='left')
                df_canon.drop(columns=['Team'], inplace=True, errors='ignore')
            
            # === Core deltas and magnitude features
            df_canon['Line_Move_Magnitude'] = pd.to_numeric(df_canon['Line_Delta'], errors='coerce').abs()
            df_canon['Line_Magnitude_Abs'] = df_canon['Line_Move_Magnitude']  # Alias
            
            # === Sharp vs. Rec Line Delta (must come before any delta-based features)
            df_canon['Sharp_Line_Delta'] = np.where(
                df_canon['Is_Sharp_Book'] == 1,
                df_canon['Line_Delta'],
                0
            )
            
            df_canon['Rec_Line_Delta'] = np.where(
                df_canon['Is_Sharp_Book'] == 0,
                df_canon['Line_Delta'],
                0
            )
            
            df_canon['Sharp_Line_Magnitude'] = df_canon['Sharp_Line_Delta'].abs()
            df_canon['Rec_Line_Magnitude'] = df_canon['Rec_Line_Delta'].abs()
            
            # === Pricing features
            df_canon['Odds_Price'] = pd.to_numeric(df_canon.get('Odds_Price'), errors='coerce')
            df_canon['Implied_Prob'] = df_canon['Odds_Price'].apply(implied_prob)
            
            # === Limit indicators
            df_canon['High_Limit_Flag'] = (df_canon['Sharp_Limit_Total'] >= 10000).astype(int)
            
            # === Team & bet side indicators
            df_canon['Is_Home_Team_Bet'] = (df_canon['Outcome'] == df_canon['Home_Team_Norm']).astype(int)
            df_canon['Is_Favorite_Bet'] = (df_canon['Value'] < 0).astype(int)
            
            # === Interaction Features (conditionally built)
            if 'Odds_Shift' in df_canon.columns:
                df_canon['SharpMove_OddsShift'] = df_canon['Sharp_Move_Signal'] * df_canon['Odds_Shift']
            
            if 'Implied_Prob_Shift' in df_canon.columns:
                df_canon['MarketLeader_ImpProbShift'] = df_canon['Market_Leader'] * df_canon['Implied_Prob_Shift']
            
            df_canon['SharpLimit_SharpBook'] = df_canon['Is_Sharp_Book'] * df_canon['Sharp_Limit_Total']
            df_canon['LimitProtect_SharpMag'] = df_canon['LimitUp_NoMove_Flag'] * df_canon['Sharp_Line_Magnitude']
            df_canon['HomeRecLineMag'] = df_canon['Is_Home_Team_Bet'] * df_canon['Rec_Line_Magnitude']
            
            # === Comparative & diagnostic features
            df_canon['Delta_Sharp_vs_Rec'] = df_canon['Sharp_Line_Delta'] - df_canon['Rec_Line_Delta']
            df_canon['Sharp_Leads'] = (df_canon['Sharp_Line_Magnitude'] > df_canon['Rec_Line_Magnitude']).astype(int)
            
            df_canon['Same_Direction_Move'] = (
                (np.sign(df_canon['Sharp_Line_Delta']) == np.sign(df_canon['Rec_Line_Delta']))
                & (df_canon['Sharp_Line_Delta'].abs() > 0)
                & (df_canon['Rec_Line_Delta'].abs() > 0)
            ).astype(int)
            
            df_canon['Opposite_Direction_Move'] = (
                (np.sign(df_canon['Sharp_Line_Delta']) != np.sign(df_canon['Rec_Line_Delta']))
                & (df_canon['Sharp_Line_Delta'].abs() > 0)
                & (df_canon['Rec_Line_Delta'].abs() > 0)
            ).astype(int)
            
            df_canon['Sharp_Move_No_Rec'] = (
                (df_canon['Sharp_Line_Delta'].abs() > 0) & (df_canon['Rec_Line_Delta'].abs() == 0)
            ).astype(int)
            
            df_canon['Rec_Move_No_Sharp'] = (
                (df_canon['Rec_Line_Delta'].abs() > 0) & (df_canon['Sharp_Line_Delta'].abs() == 0)
            ).astype(int)
            df_canon['SharpMove_Odds_Up'] = ((df_canon['Sharp_Move_Signal'] == 1) & (df_canon['Odds_Shift'] > 0)).astype(int)
            df_canon['SharpMove_Odds_Down'] = ((df_canon['Sharp_Move_Signal'] == 1) & (df_canon['Odds_Shift'] < 0)).astype(int)
            df_canon['SharpMove_Odds_Mag'] = df_canon['Odds_Shift'].abs() * df_canon['Sharp_Move_Signal']
            df_canon['Was_Line_Resistance_Broken'] = df_canon.get('Was_Line_Resistance_Broken', 0).fillna(0).astype(int)
            df_canon['SharpMove_Resistance_Break'] = (
                df_canon['Sharp_Move_Signal'] * df_canon['Was_Line_Resistance_Broken']
            )
            df_canon['Net_Line_Move_From_Opening'] = df_canon['Value'] - df_canon['Open_Value']
            df_canon['Abs_Line_Move_From_Opening'] = df_canon['Net_Line_Move_From_Opening'].abs()
            df_canon['Net_Odds_Move_From_Opening'] = (
                df_canon['Odds_Price'].apply(implied_prob) -
                df_canon['Open_Odds'].apply(implied_prob)
            ) * 100  # Optional: express in percentage points
            
            df_canon['Abs_Odds_Move_From_Opening'] = df_canon['Net_Odds_Move_From_Opening'].abs()
            df_canon['Line_Resistance_Crossed_Levels'] = df_canon.get('Line_Resistance_Crossed_Levels', '[]')
            df_canon['Line_Resistance_Crossed_Count'] = df_canon.get('Line_Resistance_Crossed_Count', 0)
            df_canon['Line_Resistance_Crossed_Levels'] = df_canon['Line_Resistance_Crossed_Levels'].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else str(x) if x else "[]"
            )

            df_canon['Minutes_To_Game'] = (
                pd.to_datetime(df_canon['Game_Start'], utc=True) - pd.to_datetime(df_canon['Snapshot_Timestamp'], utc=True)
            ).dt.total_seconds() / 60
            
            df_canon['Late_Game_Steam_Flag'] = (df_canon['Minutes_To_Game'] <= 60).astype(int)
            
            df_canon['Minutes_To_Game_Tier'] = pd.cut(
                df_canon['Minutes_To_Game'],
                bins=[-1, 30, 60, 180, 360, 720, np.inf],
                labels=['🚨 ≤30m', '🔥 ≤1h', '⚠️ ≤3h', '⏳ ≤6h', '📅 ≤12h', '🕓 >12h']
            )
            if 'Value_Reversal_Flag' not in df_canon.columns:
                df_canon['Value_Reversal_Flag'] = 0
            df_canon['Value_Reversal_Flag'] = df_canon['Value_Reversal_Flag'].fillna(0).astype(int)
            if 'Odds_Reversal_Flag' not in df_canon.columns:
                df_canon['Odds_Reversal_Flag'] = 0
            df_canon['Odds_Reversal_Flag'] = df_canon['Odds_Reversal_Flag'].fillna(0).astype(int)
            # Absolute deviation from 50% implied fair odds
            df_canon['Market_Implied_Prob'] = df_canon['Odds_Price'].apply(implied_prob)

            df_canon['Mispricing_Gap'] = df_canon['Team_Past_Avg_Model_Prob'] - df_canon['Market_Implied_Prob']
            df_canon['Abs_Mispricing_Gap'] = df_canon['Mispricing_Gap'].abs()
            
            df_canon['Team_Implied_Prob_Gap_Home'] = df_canon['Team_Past_Avg_Model_Prob_Home'] - df_canon['Market_Implied_Prob']
            df_canon['Team_Implied_Prob_Gap_Away'] = df_canon['Team_Past_Avg_Model_Prob_Away'] - df_canon['Market_Implied_Prob']
            
            df_canon['Abs_Team_Implied_Prob_Gap'] = np.where(
                df_canon['Is_Home_Team_Bet'] == 1,
                df_canon['Team_Implied_Prob_Gap_Home'].abs(),
                df_canon['Team_Implied_Prob_Gap_Away'].abs()
            )
            
            df_canon['Team_Mispriced_Flag'] = (df_canon['Abs_Team_Implied_Prob_Gap'] > 0.05).astype(int)
            # === Cross-Market Alignment Features
            df_canon['Spread_Implied_Prob'] = df_canon['Spread_Odds'].apply(implied_prob)
            df_canon['H2H_Implied_Prob'] = df_canon['H2H_Odds'].apply(implied_prob)
            df_canon['Total_Implied_Prob'] = df_canon['Total_Odds'].apply(implied_prob)
            
            df_canon['Spread_vs_H2H_Aligned'] = (
                (df_canon['Value'] < 0) &  # Spread_Value
                (df_canon['H2H_Implied_Prob'] > 0.5)
            ).astype(int)
            
            df_canon['Total_vs_Spread_Contradiction'] = (
                (df_canon['Spread_Implied_Prob'] > 0.55) &
                (df_canon['Total_Implied_Prob'] < 0.48)
            ).astype(int)
            
            df_canon['Spread_vs_H2H_ProbGap'] = df_canon['Spread_Implied_Prob'] - df_canon['H2H_Implied_Prob']
            df_canon['Total_vs_H2H_ProbGap'] = df_canon['Total_Implied_Prob'] - df_canon['H2H_Implied_Prob']
            df_canon['Total_vs_Spread_ProbGap'] = df_canon['Total_Implied_Prob'] - df_canon['Spread_Implied_Prob']
            
            df_canon['CrossMarket_Prob_Gap_Exists'] = (
                (df_canon['Spread_vs_H2H_ProbGap'].abs() > 0.05) |
                (df_canon['Total_vs_Spread_ProbGap'].abs() > 0.05)
            ).astype(int)

            df_canon = add_line_and_crossmarket_features(df_canon)
            df_canon = compute_small_book_liquidity_features(df_canon)

            # Flattened hybrid timing buckets
            # 🔄 Flattened hybrid timing columns (NUMERIC only)
            hybrid_timing_cols = [
                'SharpMove_Magnitude_Overnight_VeryEarly', 'SharpMove_Magnitude_Overnight_MidRange',
                'SharpMove_Magnitude_Overnight_LateGame', 'SharpMove_Magnitude_Overnight_Urgent',
                'SharpMove_Magnitude_Early_VeryEarly', 'SharpMove_Magnitude_Early_MidRange',
                'SharpMove_Magnitude_Early_LateGame', 'SharpMove_Magnitude_Early_Urgent',
                'SharpMove_Magnitude_Midday_VeryEarly', 'SharpMove_Magnitude_Midday_MidRange',
                'SharpMove_Magnitude_Midday_LateGame', 'SharpMove_Magnitude_Midday_Urgent',
                'SharpMove_Magnitude_Late_VeryEarly', 'SharpMove_Magnitude_Late_MidRange',
                'SharpMove_Magnitude_Late_LateGame', 'SharpMove_Magnitude_Late_Urgent',
                'SharpMove_Timing_Magnitude'  # ✅ numeric
            ]
            
            # ✅ Process numeric timing features
            for col in hybrid_timing_cols:
                if col in df_canon.columns:
                    df_canon[col] = pd.to_numeric(df_canon[col], errors='coerce').fillna(0.0)
                else:
                    df_canon[col] = 0.0
            
            # ✅ Handle string column separately
            if 'SharpMove_Timing_Dominant' not in df_canon.columns:
                df_canon['SharpMove_Timing_Dominant'] = 'unknown'
                
            hybrid_odds_timing_cols = [
                'Odds_Move_Magnitude',  # total
            ] + [
                f'OddsMove_Magnitude_{b}' for b in [
                    'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
                    'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
                    'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
                    'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
                ]
            ]
            # ✅ Process numeric odds timing features
            for col in hybrid_odds_timing_cols:
                if col in df_canon.columns:
                    df_canon[col] = pd.to_numeric(df_canon[col], errors='coerce').fillna(0.0)
                else:
                    df_canon[col] = 0.0  
            # === Ensure required features exist ===
            # === Ensure required features exist ===
            model_features = trained_models[market_type]['model'].get_booster().feature_names
            
            missing_cols = [col for col in model_features if col not in df_full_market.columns]
            df_full_market[missing_cols] = 0  # Fill missing model features
            
            # Normalize booleans and ensure types on canonical rows only
            df_full_market.loc[df_full_market['Was_Canonical'], model_features] = (
                df_full_market.loc[df_full_market['Was_Canonical'], model_features]
                .replace({'True': 1, 'False': 0})
                .infer_objects(copy=False)
            )
            
            # Score on full_market canonical rows directly
            X_full = df_full_market.loc[df_full_market['Was_Canonical'], model_features]
            preds = trained_models[market_type]['calibrator'].predict_proba(X_full)[:, 1]
            
            df_full_market.loc[df_full_market['Was_Canonical'], 'Model_Sharp_Win_Prob'] = preds
            df_full_market.loc[df_full_market['Was_Canonical'], 'Model_Confidence'] = preds
            df_full_market.loc[df_full_market['Was_Canonical'], 'Scored_By_Model'] = True
            df_full_market.loc[df_full_market['Was_Canonical'], 'Was_Canonical'] = True
            df_full_market.loc[df_full_market['Was_Canonical'], 'Scoring_Market'] = market_type                       
            # === Batch assign all new columns at once to avoid fragmentation
            
        
            
            # Optional: trigger defragmentation  df_canon = df_canon.copy()

            logger.info(f"📋 canon after all processes row columns after enrichment: {sorted(df_canon.columns.tolist())}")
            df_canon = df_full_market[df_full_market['Was_Canonical'] == True].copy()

            df_inverse = df_full_market[df_full_market['Was_Canonical'] == False].copy()

            logger.info(f"🧪 Inverse rows found for {market_type.upper()}: {len(df_inverse)}")
            df_canon['Bookmaker'] = df_canon['Bookmaker'].str.lower().str.strip()
            df_inverse['Bookmaker'] = df_inverse['Bookmaker'].str.lower().str.strip()
            
            # ✅ Step 2: Only now is it safe to inspect df_canon
            # Better: Keep keys simple and merge on both columns
            df_canon['Team_Key_Base'] = (
                df_canon['Home_Team_Norm'].str.lower().str.strip() + "_" +
                df_canon['Away_Team_Norm'].str.lower().str.strip() + "_" +
                df_canon['Commence_Hour'].astype(str) + "_" +
                df_canon['Market'].str.lower().str.strip()
            )
            
            df_inverse['Team_Key_Base'] = (
                df_inverse['Home_Team_Norm'].str.lower().str.strip() + "_" +
                df_inverse['Away_Team_Norm'].str.lower().str.strip() + "_" +
                df_inverse['Commence_Hour'].astype(str) + "_" +
                df_inverse['Market'].str.lower().str.strip()
            )
            
            df_canon_preds = (
                df_canon[['Team_Key_Base', 'Bookmaker', 'Model_Sharp_Win_Prob', 'Model_Confidence']]
                .drop_duplicates(subset=['Team_Key_Base', 'Bookmaker'])
                .rename(columns={
                    'Model_Sharp_Win_Prob': 'Model_Sharp_Win_Prob_opponent',
                    'Model_Confidence': 'Model_Confidence_opponent'
                })
            )
            
            df_inverse = df_inverse.merge(df_canon_preds, on=['Team_Key_Base', 'Bookmaker'], how='left')
            df_inverse['Model_Sharp_Win_Prob'] = 1 - df_inverse['Model_Sharp_Win_Prob_opponent']
            df_inverse['Model_Confidence'] = 1 - df_inverse['Model_Confidence_opponent']
            df_inverse.drop(columns=['Model_Sharp_Win_Prob_opponent', 'Model_Confidence_opponent'], inplace=True)
            
            df_inverse['Was_Canonical'] = False
            df_inverse['Scored_By_Model'] = True
            df_inverse['Scoring_Market'] = market_type

            df_inverse.drop(columns=['Model_Sharp_Win_Prob_opponent', 'Model_Confidence_opponent'], inplace=True, errors='ignore')


            logger.info(f"✅ Canonical rows with non-null model prob: {df_canon['Model_Sharp_Win_Prob'].notnull().sum()} / {len(df_canon)}")
            
            # ✅ Step 3: Build canonical prediction map
            
            # === Merge cross-market odds into inverse rows
                
            # === Recompute implied probabilities from merged odds
            df_inverse['Spread_Implied_Prob'] = df_inverse['Spread_Odds'].apply(implied_prob)
            df_inverse['H2H_Implied_Prob'] = df_inverse['H2H_Odds'].apply(implied_prob)
            df_inverse['Total_Implied_Prob'] = df_inverse['Total_Odds'].apply(implied_prob)
            
            # === Cross-market alignment and gaps
            df_inverse['Spread_vs_H2H_Aligned'] = (
                (df_inverse['Value'] < 0) &
                (df_inverse['H2H_Implied_Prob'] > 0.5)
            ).astype(int)
            
            df_inverse['Total_vs_Spread_Contradiction'] = (
                (df_inverse['Spread_Implied_Prob'] > 0.55) &
                (df_inverse['Total_Implied_Prob'] < 0.48)
            ).astype(int)
            
            df_inverse['Spread_vs_H2H_ProbGap'] = df_inverse['Spread_Implied_Prob'] - df_inverse['H2H_Implied_Prob']
            df_inverse['Total_vs_H2H_ProbGap'] = df_inverse['Total_Implied_Prob'] - df_inverse['H2H_Implied_Prob']
            df_inverse['Total_vs_Spread_ProbGap'] = df_inverse['Total_Implied_Prob'] - df_inverse['Spread_Implied_Prob']
            
            df_inverse['CrossMarket_Prob_Gap_Exists'] = (
                (df_inverse['Spread_vs_H2H_ProbGap'].abs() > 0.05) |
                (df_inverse['Total_vs_Spread_ProbGap'].abs() > 0.05)
            ).astype(int)

          
            # Step 1: Define the columns FIRST
            team_stat_cols = [col for col in df_canon.columns if col.startswith('Team_Past_')]
            
            # Step 2: Merge if applicable
            if team_stat_cols:
                df_inverse = df_inverse.drop(columns=team_stat_cols, errors='ignore')
                df_inverse = df_inverse.merge(
                    df_canon[['Outcome_Norm'] + team_stat_cols].drop_duplicates(subset=['Outcome_Norm']),
                    on='Outcome_Norm', how='left'
                )
            logger.info(f"📋 Inverse1 row columns after enrichment: {sorted(df_inverse.columns.tolist())}")
            logger.info(f"🧪 Inverse rows with Open_Value: {df_inverse['Open_Value'].notnull().sum()} / {len(df_inverse)}")
            # Merge canonical model predictions into inverse rows by Outcome_Norm
            # Ensure the merge has correct source columns
            # ✅ Step 0: Extract canonical rows first — before logging anything
            
            
            
            # ✅ Step 1: Normalize for safe key matching

            df_inverse['Was_Canonical'] = False
            df_inverse['Scored_By_Model'] = True
            logger.info(f"📋 Inverse2 row columns after enrichment: {sorted(df_inverse.columns.tolist())}")
            # === Core deltas and magnitude features
            df_inverse['Line_Move_Magnitude'] = pd.to_numeric(df_inverse['Line_Delta'], errors='coerce').abs()
            df_inverse['Line_Magnitude_Abs'] = df_inverse['Line_Move_Magnitude']  # Alias
            df_inverse['Sharp_Line_Delta'] = np.where(
                df_inverse['Is_Sharp_Book'] == 1,
                df_inverse['Line_Delta'],
                0
            )
            
            df_inverse['Rec_Line_Delta'] = np.where(
                df_inverse['Is_Sharp_Book'] == 0,
                df_inverse['Line_Delta'],
                0
            )
            # === Sharp vs. Rec Line Deltas (ensure computed upstream or now)
            df_inverse['Sharp_Line_Magnitude'] = df_inverse['Sharp_Line_Delta'].abs()
            df_inverse['Rec_Line_Magnitude'] = df_inverse['Rec_Line_Delta'].abs()
            
            # === Limit flag
            df_inverse['High_Limit_Flag'] = (df_inverse['Sharp_Limit_Total'] >= 10000).astype(int)
            
            # === Interaction Features
            if 'Odds_Shift' in df_inverse.columns:
                df_inverse['SharpMove_OddsShift'] = df_inverse['Sharp_Move_Signal'] * df_inverse['Odds_Shift']
            
            if 'Implied_Prob_Shift' in df_inverse.columns:
                df_inverse['MarketLeader_ImpProbShift'] = df_inverse['Market_Leader'] * df_inverse['Implied_Prob_Shift']
            
            df_inverse['SharpLimit_SharpBook'] = df_inverse['Is_Sharp_Book'] * df_inverse['Sharp_Limit_Total']
            df_inverse['LimitProtect_SharpMag'] = df_inverse['LimitUp_NoMove_Flag'] * df_inverse['Sharp_Line_Magnitude']
            df_inverse['HomeRecLineMag'] = df_inverse['Is_Home_Team_Bet'] * df_inverse['Rec_Line_Magnitude']
            
            # === Comparative & diagnostic features
            df_inverse['Delta_Sharp_vs_Rec'] = df_inverse['Sharp_Line_Delta'] - df_inverse['Rec_Line_Delta']
            df_inverse['Sharp_Leads'] = (df_inverse['Sharp_Line_Magnitude'] > df_inverse['Rec_Line_Magnitude']).astype(int)
            
            df_inverse['Same_Direction_Move'] = (
                (np.sign(df_inverse['Sharp_Line_Delta']) == np.sign(df_inverse['Rec_Line_Delta']))
                & (df_inverse['Sharp_Line_Delta'].abs() > 0)
                & (df_inverse['Rec_Line_Delta'].abs() > 0)
            ).astype(int)
            
            df_inverse['Opposite_Direction_Move'] = (
                (np.sign(df_inverse['Sharp_Line_Delta']) != np.sign(df_inverse['Rec_Line_Delta']))
                & (df_inverse['Sharp_Line_Delta'].abs() > 0)
                & (df_inverse['Rec_Line_Delta'].abs() > 0)
            ).astype(int)
            
            df_inverse['Sharp_Move_No_Rec'] = (
                (df_inverse['Sharp_Line_Delta'].abs() > 0) & (df_inverse['Rec_Line_Delta'].abs() == 0)
            ).astype(int)
            
            df_inverse['Rec_Move_No_Sharp'] = (
                (df_inverse['Rec_Line_Delta'].abs() > 0) & (df_inverse['Sharp_Line_Delta'].abs() == 0)
            ).astype(int)
            
           
            df_inverse['SharpMove_Odds_Up'] = ((df_inverse['Sharp_Move_Signal'] == 1) & (df_inverse['Odds_Shift'] > 0)).astype(int)
            df_inverse['SharpMove_Odds_Down'] = ((df_inverse['Sharp_Move_Signal'] == 1) & (df_inverse['Odds_Shift'] < 0)).astype(int)
            df_inverse['SharpMove_Odds_Mag'] = df_inverse['Odds_Shift'].abs() * df_inverse['Sharp_Move_Signal']
            df_inverse['Net_Line_Move_From_Opening'] = df_inverse['Value'] - df_inverse['Open_Value']
            df_inverse['Abs_Line_Move_From_Opening'] = df_inverse['Net_Line_Move_From_Opening'].abs()
            # Helper function (ensure it's defined)

            # Apply corrected logic to inverse rows
            df_inverse['Net_Odds_Move_From_Opening'] = (
                df_inverse['Odds_Price'].apply(implied_prob) -
                df_inverse['Open_Odds'].apply(implied_prob)
            ) * 100  # Optional: percent points
            
            df_inverse['Abs_Odds_Move_From_Opening'] = df_inverse['Net_Odds_Move_From_Opening'].abs()

                            
            df_inverse['Was_Line_Resistance_Broken'] = df_inverse.get('Was_Line_Resistance_Broken', 0).fillna(0).astype(int)
            df_inverse['SharpMove_Resistance_Break'] = (
                df_inverse['Sharp_Move_Signal'] * df_inverse['Was_Line_Resistance_Broken']
            )
            df_inverse['Line_Resistance_Crossed_Levels'] = df_inverse.get('Line_Resistance_Crossed_Levels', '[]')
            df_inverse['Line_Resistance_Crossed_Count'] = df_inverse.get('Line_Resistance_Crossed_Count', 0)
            df_inverse['Line_Resistance_Crossed_Levels'] = df_inverse['Line_Resistance_Crossed_Levels'].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else str(x) if x else "[]"
            )

            # Compute minutes until game start
            df_inverse['Minutes_To_Game'] = (
                pd.to_datetime(df_inverse['Game_Start'], utc=True) -
                pd.to_datetime(df_inverse['Snapshot_Timestamp'], utc=True)
            ).dt.total_seconds() / 60
            
            # Flag: Is this a late steam move?
            df_inverse['Late_Game_Steam_Flag'] = (df_inverse['Minutes_To_Game'] <= 60).astype(int)
            df_inverse['Market_Implied_Prob'] = df_inverse['Odds_Price'].apply(implied_prob)

            df_inverse['Mispricing_Gap'] = df_inverse['Team_Past_Avg_Model_Prob'] - df_inverse['Market_Implied_Prob']
            df_inverse['Abs_Mispricing_Gap'] = df_inverse['Mispricing_Gap'].abs()
            df_inverse['Mispricing_Flag'] = (df_inverse['Abs_Mispricing_Gap'] > 0.05).astype(int)
            
            df_inverse['Team_Implied_Prob_Gap_Home'] = (
                df_inverse['Team_Past_Avg_Model_Prob_Home'] - df_inverse['Market_Implied_Prob']
            )
            
            df_inverse['Team_Implied_Prob_Gap_Away'] = (
                df_inverse['Team_Past_Avg_Model_Prob_Away'] - df_inverse['Market_Implied_Prob']
            )
            
            # ✅ Context-aware gap (if Is_Home_Team_Bet exists)
            df_inverse['Abs_Team_Implied_Prob_Gap'] = np.where(
                df_inverse['Is_Home_Team_Bet'] == 1,
                df_inverse['Team_Implied_Prob_Gap_Home'].abs(),
                df_inverse['Team_Implied_Prob_Gap_Away'].abs()
            )
            
            df_inverse['Team_Mispriced_Flag'] = (df_inverse['Abs_Team_Implied_Prob_Gap'] > 0.05).astype(int)
            df_inverse = add_line_and_crossmarket_features(df_inverse)
            #df_inverse = compute_small_book_liquidity_features(df_inverse)
            # Bucketed tier for diagnostics or categorical modeling
            df_inverse['Minutes_To_Game_Tier'] = pd.cut(
                df_inverse['Minutes_To_Game'],
                bins=[-1, 30, 60, 180, 360, 720, np.inf],
                labels=['🚨 ≤30m', '🔥 ≤1h', '⚠️ ≤3h', '⏳ ≤6h', '📅 ≤12h', '🕓 >12h']
            )
            logger.info(f"📋 Inverse3 row columns after enrichment: {sorted(df_inverse.columns.tolist())}")
            
            # ✅ Only numeric hybrid timing columns
            hybrid_timing_cols = [
                'SharpMove_Magnitude_Overnight_VeryEarly', 'SharpMove_Magnitude_Overnight_MidRange',
                'SharpMove_Magnitude_Overnight_LateGame', 'SharpMove_Magnitude_Overnight_Urgent',
                'SharpMove_Magnitude_Early_VeryEarly', 'SharpMove_Magnitude_Early_MidRange',
                'SharpMove_Magnitude_Early_LateGame', 'SharpMove_Magnitude_Early_Urgent',
                'SharpMove_Magnitude_Midday_VeryEarly', 'SharpMove_Magnitude_Midday_MidRange',
                'SharpMove_Magnitude_Midday_LateGame', 'SharpMove_Magnitude_Midday_Urgent',
                'SharpMove_Magnitude_Late_VeryEarly', 'SharpMove_Magnitude_Late_MidRange',
                'SharpMove_Magnitude_Late_LateGame', 'SharpMove_Magnitude_Late_Urgent',
                'SharpMove_Timing_Magnitude'  # ✅ numeric
            ]
            
            # ✅ Convert numeric timing columns
            for col in hybrid_timing_cols:
                if col in df_inverse.columns:
                    df_inverse[col] = pd.to_numeric(df_inverse[col], errors='coerce').fillna(0.0)
                else:
                    df_inverse[col] = 0.0
            if 'SharpMove_Timing_Dominant' not in df_inverse.columns:
                df_inverse['SharpMove_Timing_Dominant'] = 'unknown'
            else:
                df_inverse['SharpMove_Timing_Dominant'] = df_inverse['SharpMove_Timing_Dominant'].fillna('unknown').astype(str)
                
                           
            hybrid_odds_timing_cols = [
                'Odds_Move_Magnitude',
            ] + [
                f'OddsMove_Magnitude_{b}' for b in [
                    'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
                    'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
                    'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
                    'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
                ]
            ]
            # ✅ Convert numeric odds timing columns for inverse rows
            for col in hybrid_odds_timing_cols:
                if col in df_inverse.columns:
                    df_inverse[col] = pd.to_numeric(df_inverse[col], errors='coerce').fillna(0.0)
                else:
                    df_inverse[col] = 0.0
            # ✅ Handle string timing label separately
            
            if 'Value_Reversal_Flag' not in df_inverse.columns:
                df_inverse['Value_Reversal_Flag'] = 0
            df_inverse['Value_Reversal_Flag'] = df_inverse['Value_Reversal_Flag'].fillna(0).astype(int)
            
            if 'Odds_Reversal_Flag' not in df_inverse.columns:
                df_inverse['Odds_Reversal_Flag'] = 0
            df_inverse['Odds_Reversal_Flag'] = df_inverse['Odds_Reversal_Flag'].fillna(0).astype(int)

           
                # ✅ Step 4: Deduplicate snapshot to keep latest valid lines
                
          
                               
             
            
            # === 🔁 Re-merge opening/extremes onto inverse rows
            try:
                # Drop existing versions to avoid _x/_y suffixes
                cols_to_refresh = [
                    'Open_Value', 'Open_Odds', 'First_Imp_Prob',
                    'Open_Book_Value',  # ✅ ADD THESE
                    'Min_Value', 'Max_Value', 'Min_Odds', 'Max_Odds',
                    'Odds_Shift', 'Line_Delta', 'Implied_Prob_Shift',
                    'Value_Reversal_Flag', 'Odds_Reversal_Flag',
                    'Is_Home_Team_Bet', 'Is_Favorite_Bet',
                    'Delta', 'Direction_Aligned', 'Line_Move_Magnitude', 'Line_Magnitude_Abs',
                    # ✅ NEW — Net movement from open (line & odds)
                    'Net_Line_Move_From_Opening', 'Abs_Line_Move_From_Opening',
                    'Net_Odds_Move_From_Opening', 'Abs_Odds_Move_From_Opening',
                    'Team_Past_Hit_Rate', 'Team_Past_Hit_Rate_Home', 'Team_Past_Hit_Rate_Away',
                    'Team_Past_Avg_Model_Prob', 'Team_Past_Avg_Model_Prob_Home', 'Team_Past_Avg_Model_Prob_Away',
                    'Market_Implied_Prob', 'Mispricing_Gap','Abs_Mispricing_Gap','Mispricing_Flag',
                    'Team_Implied_Prob_Gap_Home','Team_Implied_Prob_Gap_Away', 
                    # 🔥 Recent cover streak counts (0–3 range)
                    'Avg_Recent_Cover_Streak',
                    'Avg_Recent_Cover_Streak_Home',
                    'Avg_Recent_Cover_Streak_Away',
                    
                    # ✅ Binary flag rate (percentage of rows on a streak)
                    'Rate_On_Cover_Streak',
                    'Rate_On_Cover_Streak_Home',
                    'Rate_On_Cover_Streak_Away',
                    'Spread_vs_H2H_Aligned','Total_vs_Spread_Contradiction','Spread_vs_H2H_ProbGap','Total_vs_H2H_ProbGap','Total_vs_Spread_ProbGap','CrossMarket_Prob_Gap_Exists',
                    'Spread_Implied_Prob','H2H_Implied_Prob','Total_Implied_Prob',
                    'Abs_Line_Move_From_Opening',
                    'Pct_Line_Move_From_Opening',
                    'Pct_Line_Move_Bin',
                    'Abs_Line_Move_Z',
                    'Pct_Line_Move_Z',
                
                    # 🧠 Odds movement and Z-score
                    'Implied_Prob',
                    'First_Imp_Prob',
                    'Implied_Prob_Shift',
                    'Implied_Prob_Shift_Z',
                
                    # 🚨 Overmove flags
                    'Potential_Overmove_Flag',
                    'Potential_Overmove_Total_Pct_Flag',
                    #'Potential_Odds_Overmove_Flag',
                
                    # 🔁 Directional movement
                    #'Line_Moved_Toward_Team',
                    'Line_Moved_Away_From_Team',
                
                    # 🔁 Cross-market alignment and contradiction
                    'Spread_Implied_Prob',
                    'H2H_Implied_Prob',
                    'Total_Implied_Prob',
                    'Spread_vs_H2H_Aligned',
                    'Spread_vs_H2H_ProbGap',
                    'Total_vs_Spread_Contradiction',
                    'Total_vs_Spread_ProbGap',
                    'Total_vs_H2H_ProbGap',
                    'CrossMarket_Prob_Gap_Exists',
                
                    # 🔁 Defensive drop
                    'Disable_Line_Move_Features',
                    'Pct_On_Recent_Cover_Streak_Home',
                    'Pct_On_Recent_Cover_Streak_Away',
                    'Pct_On_Recent_Cover_Streak',
                    'SmallBook_Total_Limit',
                    'SmallBook_Max_Limit',
                    'SmallBook_Min_Limit',
                    'SmallBook_Limit_Skew',
                    'SmallBook_Heavy_Liquidity_Flag',
                    'SmallBook_Limit_Skew_Flag',
                    
                    
                    

                ]
        
              
                  
                # Drop old enriched columns
                df_inverse = df_inverse.drop(columns=[col for col in cols_to_refresh if col in df_inverse.columns], errors='ignore')
                
            
                
                # 🔁 Merge openers/extremes
                df_inverse = df_inverse.merge(df_open, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left')
                df_inverse = df_inverse.merge(df_open_book, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left')
                df_inverse = df_inverse.merge(df_extremes, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left')
            
                
                # 🔁 Re-merge team-level features
                try:
                    if team_feature_map is not None and not team_feature_map.empty:
                        df_inverse['Team'] = df_inverse['Outcome_Norm'].str.lower().str.strip()
                        df_inverse = df_inverse.merge(team_feature_map, on='Team', how='left')
                        logger.info(f"🔁 Re-merged team-level features for {len(df_inverse)} inverse rows.")
                except Exception as e:
                    logger.error(f"❌ Failed to re-merge team-level features for inverse rows: {e}")
                
                df_inverse.drop(columns=['Team'], inplace=True, errors='ignore')
                  
                    
                # === 🔁 Recompute outcome-sensitive fields
                df_inverse['Implied_Prob'] = df_inverse['Odds_Price'].apply(implied_prob)
                df_inverse['Odds_Shift'] = pd.to_numeric(df_inverse['Odds_Price'], errors='coerce') - pd.to_numeric(df_inverse['Open_Odds'], errors='coerce')
                df_inverse['Implied_Prob_Shift'] = df_inverse['Implied_Prob'] - pd.to_numeric(df_inverse['First_Imp_Prob'], errors='coerce')
                df_inverse['Line_Delta'] = pd.to_numeric(df_inverse['Value'], errors='coerce') - pd.to_numeric(df_inverse['Open_Value'], errors='coerce')
                df_inverse['Delta'] = pd.to_numeric(df_inverse['Value'], errors='coerce') - pd.to_numeric(df_inverse['Open_Value'], errors='coerce')
                # === Merge cross-market odds into inverse rows
             
                
                # === Recompute implied probabilities from merged odds
                df_inverse['Spread_Implied_Prob'] = df_inverse['Spread_Odds'].apply(implied_prob)
                df_inverse['H2H_Implied_Prob'] = df_inverse['H2H_Odds'].apply(implied_prob)
                df_inverse['Total_Implied_Prob'] = df_inverse['Total_Odds'].apply(implied_prob)
                
                # === Cross-market alignment and gaps
                df_inverse['Spread_vs_H2H_Aligned'] = (
                    (df_inverse['Value'] < 0) &
                    (df_inverse['H2H_Implied_Prob'] > 0.5)
                ).astype(int)
                
                df_inverse['Total_vs_Spread_Contradiction'] = (
                    (df_inverse['Spread_Implied_Prob'] > 0.55) &
                    (df_inverse['Total_Implied_Prob'] < 0.48)
                ).astype(int)
                
                df_inverse['Spread_vs_H2H_ProbGap'] = df_inverse['Spread_Implied_Prob'] - df_inverse['H2H_Implied_Prob']
                df_inverse['Total_vs_H2H_ProbGap'] = df_inverse['Total_Implied_Prob'] - df_inverse['H2H_Implied_Prob']
                df_inverse['Total_vs_Spread_ProbGap'] = df_inverse['Total_Implied_Prob'] - df_inverse['Spread_Implied_Prob']
                
                df_inverse['CrossMarket_Prob_Gap_Exists'] = (
                    (df_inverse['Spread_vs_H2H_ProbGap'].abs() > 0.05) |
                    (df_inverse['Total_vs_Spread_ProbGap'].abs() > 0.05)
                ).astype(int)

                

         
                df_inverse['Is_Home_Team_Bet'] = (df_inverse['Outcome'].str.lower() == df_inverse['Home_Team_Norm'].str.lower()).astype(float)
                df_inverse['Is_Favorite_Bet'] = (pd.to_numeric(df_inverse['Value'], errors='coerce') < 0).astype(float)
                df_inverse['Direction_Aligned'] = np.where(
                    df_inverse['Line_Delta'] > 0, 1,
                    np.where(df_inverse['Line_Delta'] < 0, 0, np.nan)
                ).astype(float)
                df_inverse['Net_Line_Move_From_Opening'] = df_inverse['Value'] - df_inverse['Open_Value']
                df_inverse['Abs_Line_Move_From_Opening'] = df_inverse['Net_Line_Move_From_Opening'].abs()
                df_inverse['Net_Odds_Move_From_Opening'] = df_inverse['Odds_Price'] - df_inverse['Open_Odds']
                df_inverse['Abs_Odds_Move_From_Opening'] = df_inverse['Net_Odds_Move_From_Opening'].abs()
                
                    
                df_inverse['Line_Move_Magnitude'] = df_inverse['Line_Delta'].abs()
                df_inverse['Line_Magnitude_Abs'] = df_inverse['Line_Move_Magnitude']
                df_inverse['Market_Implied_Prob'] = df_inverse['Odds_Price'].apply(implied_prob)
                df_inverse['Market_Implied_Prob'] = df_inverse['Odds_Price'].apply(implied_prob)

                df_inverse['Mispricing_Gap'] = df_inverse['Team_Past_Avg_Model_Prob'] - df_inverse['Market_Implied_Prob']
                df_inverse['Abs_Mispricing_Gap'] = df_inverse['Mispricing_Gap'].abs()
                df_inverse['Mispricing_Flag'] = (df_inverse['Abs_Mispricing_Gap'] > 0.05).astype(int)
                
                df_inverse['Team_Implied_Prob_Gap_Home'] = (
                    df_inverse['Team_Past_Avg_Model_Prob_Home'] - df_inverse['Market_Implied_Prob']
                )
                
                df_inverse['Team_Implied_Prob_Gap_Away'] = (
                    df_inverse['Team_Past_Avg_Model_Prob_Away'] - df_inverse['Market_Implied_Prob']
                )
                
                # ✅ Context-aware gap (if Is_Home_Team_Bet exists)
                df_inverse['Abs_Team_Implied_Prob_Gap'] = np.where(
                    df_inverse['Is_Home_Team_Bet'] == 1,
                    df_inverse['Team_Implied_Prob_Gap_Home'].abs(),
                    df_inverse['Team_Implied_Prob_Gap_Away'].abs()
                )
                
                df_inverse['Team_Mispriced_Flag'] = (df_inverse['Abs_Team_Implied_Prob_Gap'] > 0.05).astype(int)
                df_inverse = add_line_and_crossmarket_features(df_inverse)
                df_inverse = compute_small_book_liquidity_features(df_inverse)
                # Propagate cover streak from canonical rows
                
                df_inverse = compute_value_reversal(df_inverse)
                df_inverse = compute_odds_reversal(df_inverse)
                logger.info(f"🔁 Refreshed Open/Extreme alignment for {len(df_inverse)} inverse rows.")
            

            
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                tb_str = ''.join(tb_lines)
            
                logger.error("❌ Failed to refresh inverse rows after re-merge.")
                logger.error(f"🛠 Exception Type: {exc_type.__name__}")
                logger.error(f"📍 Exception Message: {e}")
                logger.error(f"🧵 Full Traceback:\n{tb_str}")

            # ✅ Combine canonical and inverse into one scored DataFrame
            
            logger.info(f"📋 Inverse2 row columns after enrichment: {sorted(df_inverse.columns.tolist())}")
            logger.info(f"📋 canon row columns after enrichment: {sorted(df_canon.columns.tolist())}")
           
            
            def dedupe_columns(df):
                return df.loc[:, ~df.columns.duplicated()]
            
            # Dedupe columns
            df_canon = dedupe_columns(df_canon)
            df_inverse = dedupe_columns(df_inverse)
            
            # Align columns
            all_cols = sorted(set(df_canon.columns).union(set(df_inverse.columns)))
            
            for col in all_cols:
                if col not in df_canon.columns:
                    df_canon[col] = np.nan
                if col not in df_inverse.columns:
                    df_inverse[col] = np.nan
            
            # Reorder for consistency
            df_canon = df_canon[all_cols]
            df_inverse = df_inverse[all_cols]
            
            # Reset index and concat
            df_canon = df_canon.reset_index(drop=True)
            df_inverse = df_inverse.reset_index(drop=True)
            df_canon.index.name = None
            df_inverse.index.name = None
            # ✅ Always generate df_scored
            df_scored = pd.concat([df_canon, df_inverse], ignore_index=True)
          
            # ✅ Safe logging
            try:
                logger.info(f"📋 scored row columns after enrichment: {sorted(df_scored.columns.tolist())}")
                logger.info("🧩 df_scored — Columns: %s", df_scored.columns.tolist())
                logger.info("🔍 df_scored — Sample Rows:\n%s", df_scored[[
                    'Game_Key', 'Market', 'Outcome', 'Model_Sharp_Win_Prob', 
                    'Team_Past_Hit_Rate', 'Team_Past_Avg_Model_Prob'
                ]].head(5).to_string(index=False))
            except Exception as log_error:
                logger.warning(f"⚠️ Could not log scored row preview: {log_error}")
          
    

            df_scored['Model_Confidence_Tier'] = pd.cut(
                df_scored['Model_Sharp_Win_Prob'],
                bins=[0, 0.4, 0.6, 0.8, 1],
                labels=["✅ Coinflip", "⭐ Lean", "🔥 Strong Indication", "🔥 Steam"]
            )

            scored_all.append(df_scored)

        except Exception as e:
            logger.error(f"❌ Failed scoring {market_type.upper()}")
            logger.error(traceback.format_exc())

            
         
    try:
        df_final = pd.DataFrame()
        hybrid_line_cols = [
            f'SharpMove_Magnitude_{b}' for b in [
                'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
                'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
                'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
                'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
            ]
        ]
    
        hybrid_odds_cols = [
            f'OddsMove_Magnitude_{b}' for b in [
                'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
                'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
                'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
                'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
            ]
        ]
    
        if scored_all:
            # ✅ Concatenate first!
            df_final = pd.concat(scored_all, ignore_index=True)
          
            logger.info(f"🧮 Final scored breakdown — total={len(df_final)}, canonical={df_final['Was_Canonical'].sum()}, inverse={(~df_final['Was_Canonical']).sum()}")
  
        
    
            # ✅ Then create hybrid timing flags
            df_final['Hybrid_Line_Timing_Flag'] = df_final[hybrid_line_cols].gt(1.0).any(axis=1).astype(int)
            df_final['Hybrid_Odds_Timing_Flag'] = df_final[hybrid_odds_cols].gt(1.0).any(axis=1).astype(int)
    
            # ✅ Now compute active signals including hybrid flags
            df_final['Active_Signal_Count'] = (
                (df_final['Sharp_Move_Signal'] == 1).astype(int) +
                (df_final['Sharp_Limit_Jump'] == 1).astype(int) +
                (df_final['Sharp_Limit_Total'] > 10000).astype(int) +
                (df_final['LimitUp_NoMove_Flag'] == 1).astype(int) +
                (df_final['Market_Leader'] == 1).astype(int) +
                (df_final['Is_Reinforced_MultiMarket'] == 1).astype(int) +
               
                (df_final['Sharp_Line_Magnitude'] > 0.5).astype(int) +
               
               
                (df_final['SharpMove_Odds_Up'] == 1).astype(int) +
                (df_final['SharpMove_Odds_Down'] == 1).astype(int) +
                (df_final['SharpMove_Odds_Mag'] > 5).astype(int) +
                (df_final['SharpMove_Resistance_Break'] == 1).astype(int) +
                (df_final['Late_Game_Steam_Flag'] == 1).astype(int) +
                (df_final['Value_Reversal_Flag'] == 1).astype(int) +
                (df_final['Odds_Reversal_Flag'] == 1).astype(int) +
                (df_final['Abs_Line_Move_From_Opening'] > 1.0).astype(int) +
                (df_final['Abs_Odds_Move_From_Opening'] > 5).astype(int) +
                df_final['Hybrid_Line_Timing_Flag'].fillna(0).astype(int) +
                df_final['Hybrid_Odds_Timing_Flag'].fillna(0).astype(int) +
                (df_final['Team_Past_Hit_Rate'].fillna(0) > 0.6).astype(int) +
                df_final['Mispricing_Flag'].fillna(0).astype(int) +
                (df_final['Team_Implied_Prob_Gap_Home'].fillna(0) > 0.05).astype(int) +
                (df_final['Team_Implied_Prob_Gap_Away'].fillna(0) > 0.05).astype(int) +
                (df_final['Avg_Recent_Cover_Streak'].fillna(0) >= 2).astype(int) +
                (df_final['Avg_Recent_Cover_Streak_Home'].fillna(0) >= 2).astype(int) +
                (df_final['Avg_Recent_Cover_Streak_Away'].fillna(0) >= 2).astype(int)+
                df_final['Spread_vs_H2H_Aligned'].fillna(0).astype(int) +
                df_final['Total_vs_Spread_Contradiction'].fillna(0).astype(int) +
                df_final['CrossMarket_Prob_Gap_Exists'].fillna(0).astype(int)+
                (df_final['Potential_Overmove_Flag'] == 1).astype(int) +
                (df_final['Potential_Overmove_Total_Pct_Flag'] == 1).astype(int) +
                (df_final['Potential_Odds_Overmove_Flag'] == 1).astype(int) +
                (df_final['Line_Moved_Toward_Team'] == 1).astype(int) +
                (df_final['Line_Moved_Away_From_Team'] == 1).astype(int) +
                (df_final['Line_Resistance_Crossed_Count'].fillna(0) >= 1).astype(int) +
                (df_final['Abs_Line_Move_Z'].fillna(0) > 1).astype(int) +
                (df_final['Pct_Line_Move_Z'].fillna(0) > 1).astype(int)+
                (df_final['SmallBook_Heavy_Liquidity_Flag'] == 1).astype(int) +
                (df_final['SmallBook_Limit_Skew_Flag'] == 1).astype(int) +
                (df_final['SmallBook_Total_Limit'].fillna(0) > 500).astype(int)


            ).fillna(0).astype(int)


            
                
        

            # === 🔍 Diagnostic for unscored rows
            try:
                unscored_rows = df_final[df_final['Model_Sharp_Win_Prob'].isnull()]

                if not unscored_rows.empty:
                    logger.warning(f"⚠️ {len(unscored_rows)} rows were not scored (Model_Sharp_Win_Prob is null).")
                    logger.warning(unscored_rows[['Game', 'Bookmaker', 'Market', 'Outcome', 'Was_Canonical']].head(40).to_string(index=False))

        
            except Exception as e:
                logger.error(f"❌ Failed to log unscored rows by market: {e}")

            df_final['Team_Key'] = (
                df_final['Home_Team_Norm'] + "_" +
                df_final['Away_Team_Norm'] + "_" +
                df_final['Commence_Hour'].astype(str) + "_" +
                df_final['Market'] + "_" +
                df_final['Outcome_Norm']
            )
    
            # Leave Sharp_Prob_Shift to be computed with historical context later
            df_final['Sharp_Prob_Shift'] = 0.0
            #df_final = df_final[df_final['Model_Sharp_Win_Prob'].notna()]
    
            # === 🔍 Debug unscored rows
            try:
                original_keys = set(df_final['Game_Key']) | set(df['Game_Key'])  # defensive

                scored_keys = set(df_final['Game_Key'])
                unscored_keys = original_keys - scored_keys
    
                if unscored_keys:
                    logger.warning(f"⚠️ {len(unscored_keys)} Game_Keys were not scored by model")
                    unscored_df = df[df['Game_Key'].isin(unscored_keys)]
                    logger.warning("🧪 Sample unscored rows:")
                    logger.warning(unscored_df[['Game', 'Bookmaker', 'Market', 'Outcome', 'Value']].head(5).to_string(index=False))
            except Exception as debug_error:
                logger.error(f"❌ Failed to log unscored rows: {debug_error}")
            # === Remove unscored "under" rows that were never created from "over"
            pre_filter = len(df_final)
            df_final = df_final[~(
                (df_final['Market'] == 'totals') &
                (df_final['Outcome_Norm'] == 'under') &
                (df_final['Was_Canonical'] == False) &
                (df_final['Model_Sharp_Win_Prob'].isna())
            )]
            logger.info(f"🧹 Removed {pre_filter - len(df_final)} unscored UNDER rows (no OVER available)")

            logger.info(f"✅ Scoring completed in {time.time() - total_start:.2f} seconds")
            return df_final
    
        else:
            logger.warning("⚠️ No market types scored — returning empty DataFrame.")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error("❌ Exception during final aggregation")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


        
def detect_sharp_moves(current, previous, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, trained_models=None, weights=None):   
    if not current:
        logging.warning("⚠️ No current odds data provided.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}

    df_history = read_recent_sharp_master_cached(hours=120)
    df_history = df_history.dropna(subset=['Game', 'Market', 'Outcome', 'Book', 'Value'])
    df_history = df_history.sort_values('Snapshot_Timestamp')
    df_history['Book'] = df_history['Book'].str.lower().str.strip()
    df_history['Bookmaker'] = df_history['Bookmaker'].str.lower().str.strip()
    df_history['Game'] = df_history['Game'].str.strip().str.lower()

    # Build old value/odds maps
    old_val_map = (
        df_history
        .drop_duplicates(subset=['Game', 'Market', 'Outcome', 'Book'], keep='first')
        .set_index(['Game', 'Market', 'Outcome', 'Book'])['Value']
        .to_dict()
    )
    old_odds_map = (
        df_history
        .dropna(subset=['Odds_Price'])
        .drop_duplicates(subset=['Game', 'Market', 'Outcome', 'Book'], keep='first')
        .set_index(['Game', 'Market', 'Outcome', 'Book'])['Odds_Price']
        .to_dict()
    )

    rows = []
    previous_odds_map = {}
    for g in previous_map.values():
        for book in g.get('bookmakers', []):
            book_key_raw = book.get('key', '').lower()
            book_key = normalize_book_name(book_key_raw, book_key_raw)
            for market in book.get('markets', []):
                mtype = market.get('key')
                for outcome in market.get('outcomes', []):
                    label = normalize_label(outcome.get('name', ''))
                    price = outcome.get('point') if mtype != 'h2h' else outcome.get('price')
                    previous_odds_map[(g.get('home_team'), g.get('away_team'), mtype, label, book_key)] = price

    for game in current:
        home_team = game.get('home_team', '').strip().lower()
        away_team = game.get('away_team', '').strip().lower()
        

        if not home_team or not away_team:
            continue

        game_name = f"{home_team.title()} vs {away_team.title()}".strip().lower()
        event_time = pd.to_datetime(game.get("commence_time"), utc=True, errors='coerce')
        game_hour = event_time.floor('h') if pd.notnull(event_time) else pd.NaT

        for book in game.get('bookmakers', []):
            book_key_raw = book.get('key', '').lower()
            book_key = normalize_book_name(book_key_raw, book_key_raw)
            if book_key not in SHARP_BOOKS + REC_BOOKS:
                continue

            for market in book.get('markets', []):
                mtype = market.get('key', '').strip().lower()
                if mtype not in ['spreads', 'totals', 'h2h']:
                    continue
                for o in market.get('outcomes', []):
                    label = normalize_label(o.get('name', ''))
                    point = o.get('point')
                    odds_price = o.get('price')
                    value = odds_price if mtype == 'h2h' else point
                    limit = o.get('bet_limit')
                    #logger.info(f"🧪 API ROW [{mtype.upper()}] | Book: {book_key} | Outcome: {label} | Point: {point} | Odds: {odds_price} | Limit: {limit}")

                    game_key = f"{home_team}_{away_team}_{str(game_hour)}_{mtype}_{label}"
                    team_key = game_key

                    rows.append({
                        'Sport': sport_key.upper(),
                        'Game_Key': game_key,
                        'Time': snapshot_time,
                        'Game': game_name,
                        'Game_Start': event_time,
                        'Market': mtype,
                        'Outcome': label,
                        'Outcome_Norm': label,
                        'Bookmaker': book_key,
                        'Book': book_key,
                        'Value': value,
                        'Odds_Price': odds_price,
                        'Limit': limit,
                        'Old Value': None,
                        'Home_Team_Norm': home_team,
                        'Away_Team_Norm': away_team,
                        'Commence_Hour': game_hour,
                        'Was_Canonical': None,
                        'Team_Key': team_key,
                    })

                    if mtype == "spreads":
                        logging.info(f"✅ Final Spread Row for {label} @ {book_key} = {value}")

    

    df = pd.DataFrame(rows)
    if df.empty:
        logging.warning("⚠️ No sharp rows built.")
        return df, df_history, pd.DataFrame()
    df['Market'] = df['Market'].astype(str).str.lower().str.strip()
    df['Outcome_Norm'] = df['Outcome_Norm'].astype(str).str.lower().str.strip()

    df['Was_Canonical'] = False
    df.loc[(df['Market'] == 'totals') & (df['Outcome_Norm'] == 'over'), 'Was_Canonical'] = True
    df.loc[(df['Market'] == 'spreads') & (df['Value'] < 0), 'Was_Canonical'] = True
    df.loc[(df['Market'] == 'h2h') & (df['Value'] < 0), 'Was_Canonical'] = True
    
    if trained_models is None:
        trained_models = get_trained_models(sport_key)

    
    try:
        df_all_snapshots = read_recent_sharp_master_cached(hours=120)
        df_all_snapshots['Bookmaker'] = df_all_snapshots['Bookmaker'].astype(str).str.strip().str.lower()
        df_all_snapshots['Book'] = df_all_snapshots['Book'].astype(str).str.strip().str.lower()
        df_all_snapshots['Game'] = df_all_snapshots['Game'].astype(str).str.strip().str.lower()
        df_all_snapshots['Market'] = df_all_snapshots['Market'].astype(str).str.strip().str.lower()
        df_all_snapshots['Outcome'] = df_all_snapshots['Outcome'].astype(str).str.strip().str.lower()
        df_all_snapshots['Snapshot_Timestamp'] = pd.to_datetime(df_all_snapshots['Snapshot_Timestamp'], errors='coerce', utc=True)

        # ➕ Define inverse rows BEFORE using them
        df_inverse = df[df['Was_Canonical'] == False].copy()
        logger.info(f"📦 Built {len(df_inverse)} inverse rows")
    
        # 🔍 Check for misclassified canonical outcomes in inverse rows
        bad_inverse_rows = df_inverse[
            ((df_inverse['Market'] == 'totals') & (df_inverse['Outcome_Norm'] == 'over')) |
            ((df_inverse['Market'].isin(['spreads', 'h2h'])) & (df_inverse['Value'] < 0))
        ]
    
        if not bad_inverse_rows.empty:
            logger.warning(f"🚨 {len(bad_inverse_rows)} rows in df_inverse should have been canonical!")
            logger.warning(bad_inverse_rows[['Team_Key', 'Market', 'Outcome_Norm', 'Value']].head(10).to_string(index=False))
    
        # 🧠 Preserve original outcome before hydration flip
        df_inverse['Original_Outcome_Norm'] = df_inverse['Outcome_Norm']
    
       # Only hydrate if Value or Odds_Price is missing
        needs_hydration = df_inverse[
            df_inverse['Value'].isna() | df_inverse['Odds_Price'].isna()
        ].copy()
        
        logger.info(f"💧 {len(needs_hydration)} inverse rows need hydration (missing Value or Odds)")
        
        if not needs_hydration.empty:
            needs_hydration = hydrate_inverse_rows_from_snapshot(needs_hydration, df_all_snapshots)
            needs_hydration = fallback_flip_inverse_rows(needs_hydration)
        
            # Update only those rows in the main df
            df.set_index('Team_Key', inplace=True)
            needs_hydration.set_index('Team_Key', inplace=True)
            df.update(needs_hydration)
            df.reset_index(inplace=True)
        else:
            logger.info("✅ All inverse rows already had Value and Odds_Price — no hydration needed.")


    
        # ✅ Deduplicate before computing sharp metrics
        df = df.drop_duplicates(subset=["Game_Key", "Market", "Outcome", "Bookmaker"])
    
        df = apply_compute_sharp_metrics_rowwise(df, df_all_snapshots)
    
        market_weights = load_market_weights_from_bq()
        now = pd.Timestamp.utcnow()  # ✅ Add this line BEFORE using `now`



        df['Snapshot_Timestamp'] = now
        df['Event_Date'] = df['Game_Start'].dt.date
        df['Game_Start'] = pd.to_datetime(df['Game_Start'], errors='coerce', utc=True)
        logging.info("🧪 Sample final spread rows before scoring:")
        logging.info(df[df['Market'] == 'spreads'][['Game', 'Outcome', 'Bookmaker', 'Value', 'Odds_Price']].head(50).to_string(index=False))
   
        df['Line_Hash'] = df.apply(compute_line_hash, axis=1)  # ✅ Move this BEFORE scoring

        df_scored = apply_blended_sharp_score(df.copy(), trained_models, df_all_snapshots, market_weights)
    
        if not df_scored.empty:
         
           
            df_scored['Pre_Game'] = df_scored['Game_Start'] > now
            df_scored['Post_Game'] = ~df_scored['Pre_Game']
         
          
            logger.info(f"✅ Final scored row count: total={len(df_scored)}, canonical={df_scored['Was_Canonical'].sum()}, inverse={(~df_scored['Was_Canonical']).sum()}")

            df_scored = df_scored.drop_duplicates(subset=["Game_Key", "Market", "Outcome", "Bookmaker"])

            logger.info("🔎 Sample scored spread rows:")
            logger.info(df_scored[df_scored['Market'] == 'spreads'][['Game', 'Outcome', 'Bookmaker', 'Value', 'Odds_Price', 'Was_Canonical']].head(30).to_string(index=False))

            try:
                write_sharp_moves_to_master(df_scored)
                logging.info(f"✅ Wrote {len(df_scored)} rows to sharp_moves_master")
            except Exception as e:
                logging.error(f"❌ Failed to write sharp moves to BigQuery: {e}", exc_info=True)
    
            df = df_scored.copy()
            summary_df = summarize_consensus(df, SHARP_BOOKS, REC_BOOKS)
        else:
            logging.warning("⚠️ apply_blended_sharp_score() returned no rows")
            df = pd.DataFrame()
            summary_df = pd.DataFrame()
    
    except Exception as e:
        logging.error(f"❌ Error applying model scoring: {e}", exc_info=True)
        df = pd.DataFrame()
        summary_df = pd.DataFrame()
    
    # ✅ Return same snapshot history as df_history for consistency
    return df, df_all_snapshots, summary_df


def compute_weighted_signal(row, market_weights):
    market = str(row.get('Market', '')).lower()
    total_score = 0
    max_possible = 0

    component_importance = {
        'Sharp_Move_Signal': 2.0,
        'Sharp_Limit_Jump': 2.0,
        'Sharp_Time_Score': 1.5,
        'Sharp_Limit_Total': 0.001  # assumes large values (scale to 0/1 range implicitly)
    }

    for comp, importance in component_importance.items():
        val = row.get(comp)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue

        try:
            if comp == 'Sharp_Limit_Total':
                val_key = str(int(float(val) // 1000 * 1000))  # bucket to nearest 1000
            elif isinstance(val, float) and val.is_integer():
                val_key = str(int(val))
            else:
                val_key = str(val).strip().lower()
            
            weight = market_weights.get(market, {}).get(comp, {}).get(val_key, 0.5)
        except Exception:
            weight = 0.5

        total_score += weight * importance
        max_possible += importance

    return round((total_score / max_possible) * 100, 2) if max_possible else 50.0
    
    
def compute_confidence(row, market_weights):
    base_score = min(row.get('SharpBetScore', 0) / 50, 1.0) * 50
    weight_score = compute_weighted_signal(row, market_weights)

    limit_position_bonus = 0
    if row.get('LimitUp_NoMove_Flag') == 1:
        limit_position_bonus = 15
    elif row.get('Limit_Jump') == 1 and abs(row.get('Delta vs Sharp', 0)) > 0.25:
        limit_position_bonus = 5

    market_lead_bonus = 5 if row.get('Market_Leader') else 0

    final_conf = base_score + weight_score + limit_position_bonus + market_lead_bonus
    return round(min(final_conf, 100), 2)


# === Outside of detect_sharp_moves ===
def assign_confidence_scores(df, market_weights):
    df['True_Sharp_Confidence_Score'] = df.apply(
        lambda r: compute_weighted_signal(r, market_weights), axis=1
    )
    df['Enhanced_Sharp_Confidence_Score'] = df.apply(
        lambda r: compute_confidence(r, market_weights), axis=1
    )
    df['Sharp_Confidence_Tier'] = pd.cut(
        df['Enhanced_Sharp_Confidence_Score'],
        bins=[-1, 25, 50, 75, float('inf')],
        labels=['⚠️ Low', '✅ Medium', '⭐ High', '🔥 Steam']
    )
    return df

def summarize_consensus(df, SHARP_BOOKS, REC_BOOKS):
    def summarize_group(g):
        return pd.Series({
            'Rec_Book_Consensus': g[g['Book'].isin(REC_BOOKS)]['Value'].mean(),
            'Sharp_Book_Consensus': g[g['Book'].isin(SHARP_BOOKS)]['Value'].mean(),
            'Rec_Open': g[g['Book'].isin(REC_BOOKS)]['Open_Value'].mean(),
            'Sharp_Open': g[g['Book'].isin(SHARP_BOOKS)]['Open_Value'].mean()
        })

    summary_df = (
        df.groupby(['Event_Date', 'Game', 'Market', 'Outcome'])
        .apply(summarize_group)
        .reset_index()
    )

    summary_df['Recommended_Outcome'] = summary_df['Outcome']
    summary_df['Move_From_Open_Rec'] = (summary_df['Rec_Book_Consensus'] - summary_df['Rec_Open']).fillna(0)
    summary_df['Move_From_Open_Sharp'] = (summary_df['Sharp_Book_Consensus'] - summary_df['Sharp_Open']).fillna(0)

    sharp_scores = df[df['SharpBetScore'].notnull()][[
        'Event_Date', 'Game', 'Market', 'Outcome',
        'SharpBetScore', 'Enhanced_Sharp_Confidence_Score', 'Sharp_Confidence_Tier'
    ]].drop_duplicates()

    summary_df = summary_df.merge(
        sharp_scores,
        on=['Event_Date', 'Game', 'Market', 'Outcome'],
        how='left'
    )

    summary_df[['SharpBetScore', 'Enhanced_Sharp_Confidence_Score']] = summary_df[[
        'SharpBetScore', 'Enhanced_Sharp_Confidence_Score'
    ]].fillna(0)
    summary_df['Sharp_Confidence_Tier'] = summary_df['Sharp_Confidence_Tier'].fillna('⚠️ Low')

    return summary_df




        
def detect_market_leaders(df_history, sharp_books, rec_books):
    df_history = df_history.copy()
    df_history['Time'] = pd.to_datetime(df_history['Time'])
    df_history['Book'] = df_history['Book'].str.lower()
    df_history['Book'] = df_history.apply(
        lambda row: normalize_book_name(row['Book'], row.get('Bookmaker')), axis=1
    )


    # === LINE MOVE DETECTION ===
    df_open_line = (
        df_history
        .sort_values('Time')
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Line_Value'})
    )

    df_history = df_history.merge(df_open_line, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df_history['Line_Has_Moved'] = (df_history['Value'] != df_history['Open_Line_Value']) & df_history['Value'].notna()

    first_line_moves = (
        df_history[df_history['Line_Has_Moved']]
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Time']
        .min()
        .reset_index()
        .rename(columns={'Time': 'First_Line_Move_Time'})
    )

    # === ODDS MOVE DETECTION ===
    if 'Odds_Price' in df_history.columns:
        df_open_odds = (
            df_history
            .sort_values('Time')
            .groupby(['Game', 'Market', 'Outcome', 'Book'])['Odds_Price']
            .first()
            .reset_index()
            .rename(columns={'Odds_Price': 'Open_Odds_Price'})
        )
        df_history = df_history.merge(df_open_odds, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
        df_history['Odds_Has_Moved'] = (df_history['Odds_Price'] != df_history['Open_Odds_Price']) & df_history['Odds_Price'].notna()

        first_odds_moves = (
            df_history[df_history['Odds_Has_Moved']]
            .groupby(['Game', 'Market', 'Outcome', 'Book'])['Time']
            .min()
            .reset_index()
            .rename(columns={'Time': 'First_Odds_Move_Time'})
        )
    else:
        first_odds_moves = pd.DataFrame(columns=['Game', 'Market', 'Outcome', 'Book', 'First_Odds_Move_Time'])

    # === Merge both move types ===
    first_moves = pd.merge(first_line_moves, first_odds_moves, on=['Game', 'Market', 'Outcome', 'Book'], how='outer')

    # Identify sharp vs. rec books
    first_moves['Book_Type'] = first_moves['Book'].map(
        lambda b: 'Sharp' if b in sharp_books else ('Rec' if b in rec_books else 'Other')
    )

    # Rank order
    first_moves['Line_Move_Rank'] = first_moves.groupby(
        ['Game', 'Market', 'Outcome']
    )['First_Line_Move_Time'].rank(method='first')

    first_moves['Odds_Move_Rank'] = first_moves.groupby(
        ['Game', 'Market', 'Outcome']
    )['First_Odds_Move_Time'].rank(method='first')

    # === Market Leader Flags ===
    first_moves['Market_Leader_Line'] = (
        (first_moves['Book_Type'] == 'Sharp') & (first_moves['Line_Move_Rank'] == 1)
    )

    first_moves['Market_Leader_Odds'] = (
        (first_moves['Book_Type'] == 'Sharp') & (first_moves['Odds_Move_Rank'] == 1)
    )

    # ✅ Final combined flag — fallback-compatible
    first_moves['Market_Leader'] = (
        (first_moves['Book_Type'] == 'Sharp') &
        (
            (first_moves['Line_Move_Rank'] == 1) |
            (first_moves['Odds_Move_Rank'] == 1)
        )
    ) # change logic here if you want a blend

    return first_moves


def detect_cross_market_sharp_support(df_moves, SHARP_BOOKS):
    df = df_moves.copy()
    df['Market'] = df['Market'].astype(str).str.lower().str.strip()
    df['SupportKey'] = df['Game'].astype(str).str.strip() + " | " + df['Outcome'].astype(str).str.strip()

    # ✅ Step 1: Define a "sharp" row: book is sharp and sharp signal exists
    df['Is_Sharp_Row'] = (
        df['Book'].isin(SHARP_BOOKS) & 
        (df['Sharp_Move_Signal'].astype(bool))  # could also include other flags like Limit_Jump or Prob_Shift
    )

    df_sharp = df[df['Is_Sharp_Row']].copy()

    # ✅ Step 2: Count cross-market sharp support per (Game + Outcome)
    market_counts = (
        df_sharp.groupby('SupportKey')['Market']
        .nunique()
        .reset_index()
        .rename(columns={'Market': 'CrossMarketSharpSupport'})
    )

    # ✅ Step 3: Count how many sharp books have that outcome
    sharp_book_counts = (
        df_sharp.groupby('SupportKey')['Book']
        .nunique()
        .reset_index()
        .rename(columns={'Book': 'Unique_Sharp_Books'})
    )

    # ✅ Step 4: Merge back to original dataframe
    df = df.merge(market_counts, on='SupportKey', how='left')
    df = df.merge(sharp_book_counts, on='SupportKey', how='left')

    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Unique_Sharp_Books'] = df['Unique_Sharp_Books'].fillna(0).astype(int)

    # ✅ Step 5: Final flag
    df['Is_Reinforced_MultiMarket'] = (
        (df['CrossMarketSharpSupport'] >= 2) | (df['Unique_Sharp_Books'] >= 2)
    )

    return df

def load_model_from_gcs(sport, market, bucket_name="sharp-models"):
    filename = f"sharp_win_model_{sport.lower()}_{market.lower()}.pkl"
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        content = blob.download_as_bytes()
        data = pickle.loads(content)

        print(f"✅ Loaded model + calibrator from GCS: gs://{bucket_name}/{filename}")
        return {
            "model": data.get("model"),
            "calibrator": data.get("calibrator"),
            "team_feature_map": data.get("team_feature_map", pd.DataFrame())  # ✅ Optional, default empty
        }
    except Exception as e:
        logging.warning(f"⚠️ Could not load model from GCS for {sport}-{market}: {e}")
        return None

def read_recent_sharp_moves(hours=120, table=BQ_FULL_TABLE):
    try:
        client = bq_client
        query = f"""
            SELECT * FROM `{table}`
            WHERE Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
        """
        df = client.query(query).to_dataframe()
        df['Commence_Hour'] = pd.to_datetime(df['Commence_Hour'], errors='coerce', utc=True)

        print(f"✅ Loaded {len(df)} rows from BigQuery (last {hours}h)")
        return df
    except Exception as e:
        print(f"❌ Failed to read from BigQuery: {e}")
        return pd.DataFrame()

def force_bool(series):
    return series.map(lambda x: bool(int(x)) if str(x).strip() in ['0', '1'] else bool(x)).fillna(False).astype(bool)
       

def write_to_bigquery(df, table='sharp_data.sharp_scores_full', force_replace=False):
    if df.empty:
        logging.info("ℹ️ No data to write to BigQuery.")
        return

    df = df.copy()
    df.columns = [col.replace(" ", "_") for col in df.columns]

    allowed_cols = {
        'sharp_data.sharp_scores_full': [
            'Game_Key', 'Bookmaker','Book', 'Market', 'Outcome', 'Limit', 'Ref_Sharp_Value',
            'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
            'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
            'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
            'Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score',
            'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT', 'Scored', 'Snapshot_Timestamp',
            'Sport', 'Value', 'First_Line_Value', 'First_Sharp_Prob',
            'Line_Delta', 'Model_Prob_Diff', 'Direction_Aligned',
            'Unique_Sharp_Books', 'Merge_Key_Short',
            'Line_Magnitude_Abs', 'Line_Move_Magnitude',
            'Is_Home_Team_Bet', 'Is_Favorite_Bet', 'High_Limit_Flag',
            'Home_Team_Norm', 'Away_Team_Norm', 'Commence_Hour',
            'Model_Sharp_Win_Prob', 'Odds_Price', 'Implied_Prob',
            'First_Odds', 'First_Imp_Prob', 'Odds_Shift', 'Implied_Prob_Shift',
            'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds','Value_Reversal_Flag','Odds_Reversal_Flag', 
    
            # ✅ Resistance logic additions
            'Was_Line_Resistance_Broken',
            'SharpMove_Resistance_Break',
            'Line_Resistance_Crossed_Levels',
            'Line_Resistance_Crossed_Count', 'Late_Game_Steam_Flag', 
            'SharpMove_Magnitude_Overnight_VeryEarly', 'SharpMove_Magnitude_Overnight_MidRange',
            'SharpMove_Magnitude_Overnight_LateGame', 'SharpMove_Magnitude_Overnight_Urgent',
            'SharpMove_Magnitude_Early_VeryEarly', 'SharpMove_Magnitude_Early_MidRange',
            'SharpMove_Magnitude_Early_LateGame', 'SharpMove_Magnitude_Early_Urgent',
            'SharpMove_Magnitude_Midday_VeryEarly', 'SharpMove_Magnitude_Midday_MidRange',
            'SharpMove_Magnitude_Midday_LateGame', 'SharpMove_Magnitude_Midday_Urgent',
            'SharpMove_Magnitude_Late_VeryEarly', 'SharpMove_Magnitude_Late_MidRange',
            'SharpMove_Magnitude_Late_LateGame', 'SharpMove_Magnitude_Late_Urgent','SharpMove_Timing_Dominant',
            'SharpMove_Timing_Magnitude', 
                # 🎯 Odds timing magnitude (odds) — ✅ NEW ADDITIONS
            'Odds_Move_Magnitude',
            'OddsMove_Magnitude_Overnight_VeryEarly', 'OddsMove_Magnitude_Overnight_MidRange',
            'OddsMove_Magnitude_Overnight_LateGame', 'OddsMove_Magnitude_Overnight_Urgent',
            'OddsMove_Magnitude_Early_VeryEarly', 'OddsMove_Magnitude_Early_MidRange',
            'OddsMove_Magnitude_Early_LateGame', 'OddsMove_Magnitude_Early_Urgent',
            'OddsMove_Magnitude_Midday_VeryEarly', 'OddsMove_Magnitude_Midday_MidRange',
            'OddsMove_Magnitude_Midday_LateGame', 'OddsMove_Magnitude_Midday_Urgent',
            'OddsMove_Magnitude_Late_VeryEarly', 'OddsMove_Magnitude_Late_MidRange',
            'OddsMove_Magnitude_Late_LateGame', 'OddsMove_Magnitude_Late_Urgent',
            'Net_Line_Move_From_Opening',
            'Abs_Line_Move_From_Opening',
            'Net_Odds_Move_From_Opening',
            'Abs_Odds_Move_From_Opening',
         
    
        ]
    }



    if table in allowed_cols:
        # Fill missing expected columns with None
        for col in allowed_cols[table]:
            if col not in df.columns:
                df[col] = None

        df = df[[col for col in allowed_cols[table]]]
        logging.info(f"🧪 Final columns in df before upload: {df.columns.tolist()}")

        missing_cols = [col for col in allowed_cols[table] if col not in df.columns]
        if missing_cols:
            logging.warning(f"⚠️ Missing expected columns in df: {missing_cols}")

    try:
        df['Odds_Price'] = pd.to_numeric(df['Odds_Price'], errors='coerce')
        df['Implied_Prob'] = pd.to_numeric(df['Implied_Prob'], errors='coerce')
        df['Odds_Shift'] = pd.to_numeric(df.get('Odds_Shift'), errors='coerce')
        df['Implied_Prob_Shift'] = pd.to_numeric(df.get('Implied_Prob_Shift'), errors='coerce')

        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='replace' if force_replace else 'append')
        logging.info(f"✅ Uploaded {len(df)} rows to {table}")
    except Exception as e:
        logging.exception(f"❌ Failed to upload to {table}")

  
def fetch_scores_and_backtest(sport_key, df_moves=None, days_back=3, api_key=API_KEY, trained_models=None):
    expected_label = [k for k, v in SPORTS.items() if v == sport_key]
    sport_label = expected_label[0].upper() if expected_label else "NBA"
    track_feature_evolution = False

    # === 1. Fetch completed games ===
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
        response = requests.get(url, params={'apiKey': api_key, 'daysFrom': int(days_back)}, timeout=10)
        response.raise_for_status()

        try:
            games = response.json()
            if not isinstance(games, list):
                raise ValueError("Unexpected API response: not a list of games")
        except Exception as e:
            logging.error(f"❌ Failed to parse JSON response: {e}")
            return pd.DataFrame()

    except Exception as e:
        logging.error(f"❌ Failed to fetch scores: {e}")
        return pd.DataFrame()

    # === Normalize completion logic to include score-present games ===
    def is_completed(game):
        scores = game.get("scores")
        if not isinstance(scores, list):
            logging.warning(f"⚠️ Game missing or invalid 'scores': {game}")
            return False  # definitely incomplete
        return game.get("completed", False) and all(s.get("score") is not None for s in scores)

    completed_games = [g for g in games if is_completed(g)]
    logging.info(f"✅ Completed games: {len(completed_games)}")

    # === 2. Extract valid score rows ===
    score_rows = []
    for game in completed_games:
        raw_home = game.get("home_team", "")
        raw_away = game.get("away_team", "")
        home = normalize_team(raw_home)
        away = normalize_team(raw_away)

        game_start_raw = pd.to_datetime(game.get("commence_time"), utc=True)
        game_start = game_start_raw.floor("h")  # ✅ Use rounded time consistently
        merge_key = build_merge_key(home, away, game_start)

        # ✅ Normalize score names too
        scores = {normalize_team(s.get("name", "")): s.get("score") for s in game.get("scores", [])}
        if home in scores and away in scores:
            score_rows.append({
                'Merge_Key_Short': merge_key,
                'Home_Team': home,
                'Away_Team': away,
                'Game_Start': game_start,
                'Score_Home_Score': scores[home],
                'Score_Away_Score': scores[away],
                'Source': 'oddsapi',
                'Inserted_Timestamp': pd.Timestamp.utcnow()
            })
        else:
            logging.warning(f"⚠️ Skipped due to missing scores: {raw_home} vs {raw_away} | "
                            f"Home in scores: {home in scores}, Away in scores: {away in scores}")


    df_scores = pd.DataFrame(score_rows).dropna(subset=['Merge_Key_Short', 'Game_Start'])
    df_scores = df_scores.drop_duplicates(subset=['Merge_Key_Short'])
    df_scores['Score_Home_Score'] = pd.to_numeric(df_scores['Score_Home_Score'], errors='coerce')
    df_scores['Score_Away_Score'] = pd.to_numeric(df_scores['Score_Away_Score'], errors='coerce')
    df_scores = df_scores.dropna(subset=['Score_Home_Score', 'Score_Away_Score'])

    # === 3. Upload scores to `game_scores_final` ===
    # === 3. Upload scores to `game_scores_final` ===
    try:
        # Safe fallback to avoid UnboundLocalError or bad structure
        if 'df_scores_needed' not in locals() or not isinstance(df_scores_needed, pd.DataFrame):
            df_scores_needed = pd.DataFrame()
    
        existing_keys = bq_client.query("""
            SELECT DISTINCT Merge_Key_Short FROM `sharp_data.game_scores_final`
        """).to_dataframe()
        existing_keys = set(existing_keys['Merge_Key_Short'].dropna())
    
        new_scores = df_scores[~df_scores['Merge_Key_Short'].isin(existing_keys)].copy()
        blocked = df_scores[df_scores['Merge_Key_Short'].isin(existing_keys)]
    
        logging.info(f"⛔ Skipped (already in table): {len(blocked)}")
        logging.info(f"🧪 Sample skipped keys:\n{blocked[['Merge_Key_Short', 'Home_Team', 'Away_Team', 'Game_Start']].head().to_string(index=False)}")
    
        # === Safe diagnostics for df_scores_needed
        if df_scores_needed.empty or df_scores_needed.shape[1] == 0:
            logging.warning("⚠️ df_scores_needed is empty or has no columns.")
        else:
            logging.info(f"🧪 df_scores_needed shape: {df_scores_needed.shape}")
            logging.info(f"🧪 df_scores_needed head:\n{df_scores_needed.head().to_string(index=False)}")
    
            if 'Merge_Key_Short' in df_scores_needed.columns:
                if df_scores_needed['Merge_Key_Short'].isnull().any():
                    logging.warning("⚠️ At least one row in df_scores_needed has a NULL Merge_Key_Short")
            else:
                logging.warning("⚠️ 'Merge_Key_Short' column missing from df_scores_needed")
    
            if {'Home_Team', 'Away_Team'}.issubset(df_scores_needed.columns):
                if df_scores_needed[['Home_Team', 'Away_Team']].isnull().any().any():
                    logging.warning("⚠️ At least one row has missing team names")
            else:
                logging.warning("⚠️ Missing 'Home_Team' or 'Away_Team' columns in df_scores_needed")
    
            empty_rows = df_scores_needed[df_scores_needed.isnull().all(axis=1)]
            if not empty_rows.empty:
                logging.warning(f"⚠️ Found completely empty rows in df_scores_needed:\n{empty_rows}")
    
            try:
                sample = df_scores_needed[['Merge_Key_Short', 'Home_Team', 'Away_Team', 'Game_Start']].head(5)
            except KeyError as e:
                logging.warning(f"⚠️ Could not extract sample unscored game(s): {e}")
                sample = df_scores_needed.head(5)
            logging.info("🕵️ Sample unscored game(s):\n" + sample.to_string(index=False))
    
        # Detect keys that are neither new nor already in BigQuery
        all_found_keys = set(new_scores['Merge_Key_Short']) | existing_keys
        missing_keys = set(df_scores['Merge_Key_Short']) - all_found_keys
        if missing_keys:
            logging.warning(f"⚠️ These keys were neither uploaded nor matched in BigQuery:")
            logging.warning(list(missing_keys)[:5])
    
        # === Upload if valid
        if new_scores.empty:
            logging.info("ℹ️ No new scores to upload to game_scores_final")
        else:
            to_gbq(new_scores, 'sharp_data.game_scores_final', project_id=GCP_PROJECT_ID, if_exists='append')
            logging.info(f"✅ Uploaded {len(new_scores)} new game scores")
    
    except Exception as e:
        logging.exception("❌ Failed to upload game scores")
    
        # Dump a preview of the DataFrame
        try:
            logging.error("📋 Sample of new_scores DataFrame:")
            logging.error(new_scores.head(5).to_string(index=False))
        except Exception as preview_error:
            logging.error(f"❌ Failed to log DataFrame preview: {preview_error}")
    
        # Dump column dtypes
        try:
            logging.error("🧪 DataFrame dtypes:")
            logging.error(new_scores.dtypes.to_string())
        except Exception as dtypes_error:
            logging.error(f"❌ Failed to log dtypes: {dtypes_error}")
    
        # Check for suspicious column content
        for col in new_scores.columns:
            try:
                if new_scores[col].apply(lambda x: isinstance(x, dict)).any():
                    logging.error(f"⚠️ Column {col} contains dicts")
                elif new_scores[col].apply(lambda x: isinstance(x, list)).any():
                    logging.error(f"⚠️ Column {col} contains lists")
            except Exception as content_error:
                logging.error(f"❌ Failed to inspect column '{col}': {content_error}")

        return pd.DataFrame()  # Return empty DataFrame if missing


    # Function to optimize and process in chunks


    # Function to process chunks of data (sorting and deduplication)
    # Function to process chunks of data (sorting, deduplication, and memory management)
    def process_chunk(df_chunk):
        # ✅ Normalize string columns first
        for col in ['Game_Key', 'Market', 'Outcome', 'Bookmaker']:
            df_chunk.loc[:, col] = df_chunk[col].astype(str).str.strip().str.lower()
    
        # ✅ Convert to categorical for memory efficiency (after normalization)
        for col in ['Game_Key', 'Market', 'Outcome', 'Bookmaker']:
            df_chunk.loc[:, col] = df_chunk[col].astype('category')
    
        # ✅ Sort by latest snapshot to keep most recent
        df_chunk = df_chunk.sort_values(by='Snapshot_Timestamp', ascending=False)
    
        # ✅ Deduplicate per game/market/outcome/bookmaker
        df_chunk = df_chunk.drop_duplicates(
            subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'],
            keep='first'
        )
    
        gc.collect()
        return df_chunk
        

    
    def calc_cover(df):
        """
        Calculate SHARP_HIT_BOOL and SHARP_COVER_RESULT for spreads, totals, and h2h.
        Pushes are excluded from scoring.
    
        Returns:
            DataFrame with columns:
                - SHARP_COVER_RESULT: 'Win' or 'Loss'
                - SHARP_HIT_BOOL: 1 or 0
        """
        df = df.copy()
    
        # Defensive normalization
        df['Market_Lower'] = df['Market'].str.lower().fillna('')
        df['Outcome_Lower'] = df['Outcome'].str.lower().fillna('')
    
        # Scores and line
        home = pd.to_numeric(df['Score_Home_Score'], errors='coerce')
        away = pd.to_numeric(df['Score_Away_Score'], errors='coerce')
        value = pd.to_numeric(df['Value'], errors='coerce')
    
        # Booleans
        is_totals = df['Market_Lower'] == 'totals'
        is_spreads = df['Market_Lower'] == 'spreads'
        is_h2h = df['Market_Lower'] == 'h2h'
        is_under = df['Outcome_Lower'] == 'under'
        is_over = df['Outcome_Lower'] == 'over'
        is_home_side = df['Outcome'] == df['Home_Team']
        is_away_side = df['Outcome'] == df['Away_Team']
    
        # Totals logic
        total_score = home + away
        totals_hit = np.where(
            is_under, total_score < value,
            np.where(is_over, total_score > value, False)
        )
        push_totals = total_score == value
    
        # Spreads logic
        spread_margin = np.where(is_home_side, home - away, away - home)
        spread_hit = spread_margin > -value
        push_spreads = spread_margin == -value
    
        # H2H logic
        winner = np.where(home > away, df['Home_Team'],
                 np.where(away > home, df['Away_Team'], 'tie'))
        h2h_hit = df['Outcome'] == winner
    
        # Push filter
        is_push = (is_totals & push_totals) | (is_spreads & push_spreads)
    
        # Final coverage logic
        hit = np.where(
            is_totals, totals_hit,
            np.where(is_spreads, spread_hit,
                     np.where(is_h2h, h2h_hit, False))
        )
    
        # Assign final result (excluding pushes)
        result_df = pd.DataFrame({
            'SHARP_COVER_RESULT': np.where(hit, 'Win', 'Loss'),
            'SHARP_HIT_BOOL': hit.astype(int)
        })
    
        # Remove push rows
        result_df.loc[is_push, 'SHARP_COVER_RESULT'] = None
        result_df.loc[is_push, 'SHARP_HIT_BOOL'] = None
    
        return result_df
    # === 4. Load recent sharp picks
    df_master = read_recent_sharp_master_cached(hours=120)
    df_master = build_game_key(df_master)  # Ensure Merge_Key_Short is created
    
    # === Filter out games already scored in sharp_scores_full
   

    if not track_feature_evolution:
        scored_keys = bq_client.query("""
            SELECT DISTINCT Merge_Key_Short
            FROM `sharplogger.sharp_data.sharp_scores_full`
            WHERE DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
        """).to_dataframe()
    
        already_scored = set(scored_keys['Merge_Key_Short'].dropna())
        df_scores_needed = df_scores[~df_scores['Merge_Key_Short'].isin(already_scored)]
        logging.info(f"✅ Remaining unscored completed games: {len(df_scores_needed)}")
        
        if not df_scores_needed.empty:
            sample = df_scores_needed[['Merge_Key_Short', 'Home_Team', 'Away_Team', 'Game_Start']].head(5)
            logging.info("🕵️ Sample unscored game(s):\n" + sample.to_string(index=False))
    else:
        df_scores_needed = df_scores.copy()
        logging.info("📈 Time-series mode enabled: Skipping scored-key filter to allow resnapshots")
        
    # Ensure Merge_Key_Short exists AFTER loading
    if 'Merge_Key_Short' not in df_master.columns:
        df_master = build_game_key(df_master)
    if 'Merge_Key_Short' not in df_scores_needed.columns:
        df_scores_needed = build_game_key(df_scores_needed)
    
    # Debugging: Log the columns of the DataFrames after build_game_key
    # === Log schemas for debug
    logging.info(f"After build_game_key - df_scores_needed columns: {df_scores_needed.columns.tolist()}")
    logging.info(f"After build_game_key - df_master columns: {df_master.columns.tolist()}")
    
    # === Track memory usage
    process = psutil.Process(os.getpid())
    logging.info(f"Memory before snapshot load: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
    # === 1. Load full snapshots
    df_all_snapshots = read_recent_sharp_master_cached(hours=120)
    
    # === 2. Build df_first from raw history for opening lines
    required_cols = ['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Value', 'Model_Sharp_Win_Prob']
    missing = [col for col in required_cols if col not in df_all_snapshots.columns]
    
    if missing:
        logging.warning(f"⚠️ Cannot compute df_first — missing columns: {missing}")
        df_first = pd.DataFrame(columns=[
            'Game_Key', 'Market', 'Outcome', 'Bookmaker',
            'First_Line_Value', 'First_Sharp_Prob',
            'First_Odds', 'First_Imp_Prob'
        ])
    else:
        df_first = (
            df_all_snapshots
            .sort_values(by='Snapshot_Timestamp')
            .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='first')
            .loc[:, required_cols + ['Odds_Price', 'Implied_Prob']]
            .rename(columns={
                'Value': 'First_Line_Value',
                'Model_Sharp_Win_Prob': 'First_Sharp_Prob',
                'Odds_Price': 'First_Odds',
                'Implied_Prob': 'First_Imp_Prob',
            })
        )
        for col in ['Game_Key', 'Market', 'Outcome', 'Bookmaker']:
            df_first[col] = df_first[col].astype('category')
    
        df_first = df_first.reset_index(drop=True).copy()
    
        logging.info("📋 Sample df_first values:\n" +
            df_first[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'First_Line_Value']].head(10).to_string(index=False))
        logging.info("💰 Sample odds + implied:\n" +
            df_first[['First_Odds', 'First_Imp_Prob']].dropna().head(5).to_string(index=False))
        logging.info(f"🧪 df_first created with {len(df_first)} rows and columns: {df_first.columns.tolist()}")
        logging.info(f"🧪 Unique Game_Key+Market+Outcome+Bookmaker combos: {df_first[['Game_Key', 'Market', 'Outcome', 'Bookmaker']].drop_duplicates().shape[0]}")
        logging.info(f"📉 Null rates in df_first:\n{df_first[['First_Line_Value', 'First_Sharp_Prob']].isnull().mean().to_string()}")
    
        if df_first.empty:
            raise RuntimeError("❌ df_first is empty — stopping early to avoid downstream issues.")
    
    # === 3. Build df_all_snapshots_filtered from process_chunk() → latest lines
    df_all_snapshots_filtered = pd.concat([
        process_chunk(df_all_snapshots.iloc[start:start + 10000])
        for start in range(0, len(df_all_snapshots), 10000)
    ], ignore_index=True)
            
    # === Prepare df_first: reduce + convert
    df_first = df_first[[
        'Game_Key', 'Market', 'Outcome', 'Bookmaker',
        'First_Line_Value', 'First_Sharp_Prob',
        'First_Odds', 'First_Imp_Prob'
    ]].copy()

    
    # Normalize key columns in both df_first and df_master
    key_cols = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
    for df in [df_master, df_first]:
        for col in key_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()
    
    # Optional: convert to category for memory savings
    for col in key_cols:
        df_first[col] = df_first[col].astype('category')
        df_master[col] = df_master[col].astype('category')
    
    # === Prepare df_scores: reduce + deduplicate
    df_scores = df_scores[['Merge_Key_Short', 'Score_Home_Score', 'Score_Away_Score']].copy()
    df_scores['Merge_Key_Short'] = df_scores['Merge_Key_Short'].astype('category')
  
    if 'Inserted_Timestamp' in df_scores.columns:
        df_scores = (
            df_scores
            .sort_values('Inserted_Timestamp')
            .drop_duplicates(subset='Merge_Key_Short', keep='last')
        )
    else:
        df_scores = df_scores.drop_duplicates(subset='Merge_Key_Short', keep='last')
    
    # === Check join key overlap BEFORE merge
    common_keys = df_master[key_cols].drop_duplicates()
    first_keys = df_first[key_cols].drop_duplicates()
    overlap = common_keys.merge(first_keys, on=key_cols, how='inner')
    logging.info(f"🧪 Join key overlap: {len(overlap)} / {len(common_keys)} df_master rows match df_first")
    
    # === 1. Merge df_first into df_master
    df_master = df_master.merge(
        df_first,
        on=key_cols,
        how='left'
    )
    log_memory("AFTER merge with df_first")
    logging.info("🧪 Sample First_Sharp_Prob before scores:\n" + df_master[['First_Sharp_Prob']].dropna().head().to_string(index=False))
    # === 2. Save First_* columns
   
    
    # === 3. Drop Score_* to prevent conflict
    df_master.drop(columns=['Score_Home_Score', 'Score_Away_Score'], errors='ignore', inplace=True)
    
    # === 4. Merge in game scores
    df_master = df_master.merge(
        df_scores,
        on='Merge_Key_Short',
        how='inner'
    )
    # === Log resulting columns after merging df_first into df_master
    logging.info("🧩 Columns after merging df_first into df_master:")
    logging.info("🧩 df_master columns:\n" + ", ".join(df_master.columns.tolist()))
    
    # Optional: log sample of First_* fields for verification
    first_fields = ['First_Line_Value', 'First_Sharp_Prob', 'First_Odds', 'First_Imp_Prob']
    sample_first = df_master[first_fields].dropna().head(5)
    if not sample_first.empty:
        logging.info("🔍 Sample First_* fields after merge:\n" + sample_first.to_string(index=False))
    else:
        logging.warning("⚠️ No non-null First_* values found after merge — check join keys or source data.")

    # === 5. Restore First_* if dropped during merge
    if 'First_Sharp_Prob' not in df_master.columns or 'First_Line_Value' not in df_master.columns:
        df_master = df_master.merge(first_cols, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left')
    
    # === 6. Clean up suffixes
    for col in ['First_Sharp_Prob', 'First_Line_Value']:
        x_col, y_col = f"{col}_x", f"{col}_y"
        if x_col in df_master.columns or y_col in df_master.columns:
            df_master[col] = df_master.get(x_col).combine_first(df_master.get(y_col))
    df_master.drop(columns=[col for col in df_master.columns if col.endswith('_x') or col.endswith('_y')], inplace=True)
    
    # === 7. Final safety + diagnostics
    if 'First_Sharp_Prob' not in df_master.columns:
        logging.error("❌ First_Sharp_Prob missing after final merges.")
    else:
        logging.info("✅ First_Sharp_Prob successfully preserved:")
        logging.info(df_master[['First_Sharp_Prob']].dropna().head().to_string(index=False))
    
    # === 8. Final cleanup and return
    df = df_master
    logging.info(f"✅ Final df columns before scoring: {df.columns.tolist()}")
    log_memory("AFTER merge with df_scores_needed")
    logging.info(f"df shape after merge: {df.shape}")
    # === Reassign Merge_Key_Short from df_master using Game_Key
    if 'Merge_Key_Short' in df_master.columns:
        logging.info("🧩 Reassigning Merge_Key_Short from df_master via Game_Key (optimized)")
        # Build mapping dictionary (Game_Key → Merge_Key_Short)
        key_map = df_master.drop_duplicates(subset=['Game_Key'])[['Game_Key', 'Merge_Key_Short']].set_index('Game_Key')['Merge_Key_Short'].to_dict()
        # Reassign inplace without merge
        df['Merge_Key_Short'] = df['Game_Key'].map(key_map)
    
    
    # Final logging
    null_count = df['Merge_Key_Short'].isnull().sum()
    logging.info(f"🧪 Final Merge_Key_Short nulls: {null_count}")
    df['Sport'] = sport_label.upper()
    
 
    # === Track memory usage before the operation
    process = psutil.Process(os.getpid())
    logging.info(f"Memory before operation: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Function to process DataFrame in smaller chunks
    def process_chunk_logic(df_chunk):
        df_chunk = df_chunk.copy()
    
        # === Clean Line_Delta
        # === Calculate Line_Delta safely
        df_chunk['Line_Delta'] = (
            pd.to_numeric(df_chunk['Value'], errors='coerce') -
            pd.to_numeric(df_chunk['First_Line_Value'], errors='coerce')
        )
        
        # === Direction alignment (nullable-safe)
        # Set Direction_Aligned based on Line_Delta only (market-only)
        df_chunk['Direction_Aligned'] = np.where(
            df_chunk['Line_Delta'] > 0, 1,
            np.where(df_chunk['Line_Delta'] < 0, 0, np.nan)
        ).astype(float)
        df_chunk['High_Limit_Flag'] = (df_chunk['Sharp_Limit_Total'] >= 10000).astype('Int64')
        # === Direction_Aligned: purely market-based logic
        
        return df_chunk
        
    
    def process_in_chunks(df, chunk_size=10000):
        chunks = []
        for start in range(0, len(df), chunk_size):
            df_chunk = df.iloc[start:start + chunk_size]
            processed_chunk = process_chunk_logic(df_chunk)
            chunks.append(processed_chunk)
            del df_chunk, processed_chunk
            gc.collect()
        return pd.concat(chunks, ignore_index=True)
    
    
    # === ✅ Apply the chunk processing
    df = process_in_chunks(df, chunk_size=10000)
    
    # === Track memory usage after the operation
    logging.info(f"Memory after operation: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # === Clean up temporary columns and other resources if necessary
    gc.collect()
    
    # Final logging
    logging.info("🧭 Direction_Aligned counts:\n" + df['Direction_Aligned'].value_counts(dropna=False).to_string())
    
    # === 6. Calculate result
    df_valid = df.dropna(subset=['Score_Home_Score', 'Score_Away_Score'])
    if df_valid.empty:
        logging.warning("ℹ️ No valid sharp picks with scores to evaluate")
        return pd.DataFrame()
    
    logging.info("🔍 df_master columns: %s", df_master.columns.tolist())
    # === Ensure Home_Team and Away_Team are attached before cover calc
    if 'Home_Team' not in df_valid.columns or 'Away_Team' not in df_valid.columns:
        logging.info("🔗 Attaching Home_Team and Away_Team to df_valid from df_master")
        team_cols = df_master[['Game_Key', 'Home_Team_Norm', 'Away_Team_Norm']].drop_duplicates()
        team_cols = team_cols.rename(columns={'Home_Team_Norm': 'Home_Team', 'Away_Team_Norm': 'Away_Team'})
        df_valid = df_valid.merge(team_cols, on='Game_Key', how='left')
    # ✅ Vectorized calculation
    result = calc_cover(df_valid)  # must return a DataFrame with 2 columns
    
    if result.shape[1] != 2:
        logging.error("❌ calc_cover output shape mismatch — expected 2 columns.")
        return pd.DataFrame()
    
    result.columns = ['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']
    df['SHARP_COVER_RESULT'] = None
    df['SHARP_HIT_BOOL'] = None
    df['Scored'] = False
    
    df.loc[df_valid.index, ['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']] = result
    df.loc[df_valid.index, 'Scored'] = result['SHARP_COVER_RESULT'].notna()
    
    # Ensure 'Unique_Sharp_Books' is present and numeric
    if 'Unique_Sharp_Books' not in df.columns:
        df['Unique_Sharp_Books'] = 0
    df['Unique_Sharp_Books'] = pd.to_numeric(df['Unique_Sharp_Books'], errors='coerce').fillna(0).astype(int)
       
        # === Merge Home_Team_Norm and Away_Team_Norm from df_scores (which has them)
    # Skip fallback if the first merge created normalized team columns
    has_team_norm_clean = 'Home_Team_Norm' in df_master.columns and 'Away_Team_Norm' in df_master.columns
    has_team_norm_merge_x = 'Home_Team_Norm_x' in df_master.columns and 'Away_Team_Norm_x' in df_master.columns
    has_team_norm_merge_y = 'Home_Team_Norm_y' in df_master.columns and 'Away_Team_Norm_y' in df_master.columns
    
    if not (has_team_norm_clean or has_team_norm_merge_x or has_team_norm_merge_y):
        
    
        if (
            'Game_Key' in df_scores.columns and
            'Home_Team' in df_scores.columns and
            'Away_Team' in df_scores.columns
        ):
            logging.info("🧩 Fallback: merging teams via Game_Key from df_scores")
            team_cols = df_scores[['Game_Key', 'Home_Team', 'Away_Team']].drop_duplicates()
            team_cols['Home_Team_Norm'] = team_cols['Home_Team'].astype(str).str.lower().str.strip()
            team_cols['Away_Team_Norm'] = team_cols['Away_Team'].astype(str).str.lower().str.strip()
            df_master = df_master.merge(team_cols, on='Game_Key', how='left')
            logging.info("✅ Fallback team merge complete")
        else:
            logging.warning("⚠️ df_scores is missing Game_Key or team columns — skipping fallback merge")
    else:
        logging.info("⏭️ Skipping fallback — team normalization columns already exist from Merge_Key_Short merge")
    
    logging.info("📦 df_master columns AFTER team merge: %s", df_master.columns.tolist())
    
    # === 🔬 Optional: count missing normalized names
    missing_home_norm = df_master['Home_Team_Norm'].isna().sum()
    missing_away_norm = df_master['Away_Team_Norm'].isna().sum()
    logging.info(f"🔎 Missing Home_Team_Norm: {missing_home_norm}, Away_Team_Norm: {missing_away_norm}")
    # === Compute engineered features
    df_master['Line_Magnitude_Abs'] = df_master['Value'].abs()
    df_master['Is_Home_Team_Bet'] = (df_master['Outcome'].str.lower() == df_master['Home_Team_Norm'].str.lower())
    df_master['Is_Favorite_Bet'] = df_master['Value'] < 0
    
    # Optional: also calculate if not already done
    if 'Line_Move_Magnitude' not in df_master.columns:
        df_master['Line_Move_Magnitude'] = (df_master['Value'] - df_master['First_Line_Value']).abs()
    # === Final Output DataFrame ===
    score_cols = [
        'Game_Key', 'Bookmaker','Book', 'Market', 'Limit', 'Outcome', 'Ref_Sharp_Value',
        'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
        'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
        'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
        'Unique_Sharp_Books', 'Enhanced_Sharp_Confidence_Score',
        'True_Sharp_Confidence_Score', 'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT',
        'Scored', 'Sport', 'Value', 'Merge_Key_Short',
        'First_Line_Value', 'First_Sharp_Prob',
        'Line_Delta', 'Model_Prob_Diff', 'Direction_Aligned',
        'Home_Team_Norm', 'Away_Team_Norm', 'Commence_Hour',
        'Line_Magnitude_Abs', 'High_Limit_Flag',
        'Line_Move_Magnitude', 'Is_Home_Team_Bet', 'Is_Favorite_Bet','Model_Sharp_Win_Prob', 'Odds_Price', 'Implied_Prob','First_Odds', 'First_Imp_Prob','Odds_Shift','Implied_Prob_Shift',
        'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds', 'Value_Reversal_Flag','Odds_Reversal_Flag',      
        'Was_Line_Resistance_Broken',
        'SharpMove_Resistance_Break',
        'Line_Resistance_Crossed_Levels',
        'Line_Resistance_Crossed_Count', 'Late_Game_Steam_Flag', 
        'SharpMove_Magnitude_Overnight_VeryEarly', 'SharpMove_Magnitude_Overnight_MidRange',
        'SharpMove_Magnitude_Overnight_LateGame', 'SharpMove_Magnitude_Overnight_Urgent',
        'SharpMove_Magnitude_Early_VeryEarly', 'SharpMove_Magnitude_Early_MidRange',
        'SharpMove_Magnitude_Early_LateGame', 'SharpMove_Magnitude_Early_Urgent',
        'SharpMove_Magnitude_Midday_VeryEarly', 'SharpMove_Magnitude_Midday_MidRange',
        'SharpMove_Magnitude_Midday_LateGame', 'SharpMove_Magnitude_Midday_Urgent',
        'SharpMove_Magnitude_Late_VeryEarly', 'SharpMove_Magnitude_Late_MidRange',
        'SharpMove_Magnitude_Late_LateGame', 'SharpMove_Magnitude_Late_Urgent','SharpMove_Timing_Dominant',
        'SharpMove_Timing_Magnitude', 
            # 🎯 Odds timing magnitude (odds) — ✅ NEW ADDITIONS
        'Odds_Move_Magnitude',
        'OddsMove_Magnitude_Overnight_VeryEarly', 'OddsMove_Magnitude_Overnight_MidRange',
        'OddsMove_Magnitude_Overnight_LateGame', 'OddsMove_Magnitude_Overnight_Urgent',
        'OddsMove_Magnitude_Early_VeryEarly', 'OddsMove_Magnitude_Early_MidRange',
        'OddsMove_Magnitude_Early_LateGame', 'OddsMove_Magnitude_Early_Urgent',
        'OddsMove_Magnitude_Midday_VeryEarly', 'OddsMove_Magnitude_Midday_MidRange',
        'OddsMove_Magnitude_Midday_LateGame', 'OddsMove_Magnitude_Midday_Urgent',
        'OddsMove_Magnitude_Late_VeryEarly', 'OddsMove_Magnitude_Late_MidRange',
        'OddsMove_Magnitude_Late_LateGame', 'OddsMove_Magnitude_Late_Urgent',
        'Net_Line_Move_From_Opening',
        'Abs_Line_Move_From_Opening',
        'Net_Odds_Move_From_Opening',
        'Abs_Odds_Move_From_Opening',

    ]
    
    
    # === Final output
    df_scores_out = ensure_columns(df, score_cols)[score_cols].copy()
  
    logging.info("📦 df Scores out: %s", df_scores_out.columns.tolist())
    # Function to coerce boolean columns to proper format
    def coerce_bool_series(series):
        return series.map(lambda x: str(x).strip().lower() in ['true', '1', '1.0', 'yes']).astype(bool)
    
    # ✅ Coerce all BigQuery BOOL fields
    bool_cols = ['Is_Reinforced_MultiMarket', 'Market_Leader', 'LimitUp_NoMove_Flag', 'Scored']
    for col in bool_cols:
        if col in df_scores_out.columns:
            # Log before coercion
            logging.info(f"🔍 Coercing column '{col}' to bool — unique values before: {df_scores_out[col].dropna().unique()[:5]}")
            df_scores_out[col] = coerce_bool_series(df_scores_out[col])
    
            # Post-coercion validation
            if df_scores_out[col].isnull().any():
                logging.warning(f"⚠️ Column '{col}' still contains nulls after coercion!")
    
    df_scores_out['Sport'] = sport_label.upper()
    
    # === Normalize and unify sport labels
    df_scores_out['Sport'] = df_scores_out['Sport'].replace({
        'BASEBALL_MLB': 'MLB',
        'BASKETBALL_NBA': 'NBA',
        'BASKETBALL_WNBA': 'WNBA',
        'FOOTBALL_CFL': 'CFL',
        'MLB': 'MLB',  # handles redundancy safely
        'NBA': 'NBA',
        'WNBA': 'WNBA',
        'CFL': 'CFL',
        'NFL': 'NFL',
        'NCAAF': 'NCAAF',
    }).str.upper()
    
    if 'Snapshot_Timestamp' in df.columns:
        df_scores_out['Snapshot_Timestamp'] = df['Snapshot_Timestamp']
    else:
        df_scores_out['Snapshot_Timestamp'] = pd.Timestamp.utcnow()
    
    # === Coerce and clean all fields BEFORE dedup and upload
    df_scores_out['Sharp_Move_Signal'] = pd.to_numeric(df_scores_out['Sharp_Move_Signal'], errors='coerce').astype('Int64')
    df_scores_out['Sharp_Limit_Jump'] = pd.to_numeric(df_scores_out['Sharp_Limit_Jump'], errors='coerce').astype('Int64')
    df_scores_out['Sharp_Prob_Shift'] = pd.to_numeric(df_scores_out['Sharp_Prob_Shift'], errors='coerce').fillna(0.0).astype(float)
    
    # === Debug unexpected boolean coercion errors before Parquet conversion
    for col in ['Is_Reinforced_MultiMarket', 'Market_Leader', 'LimitUp_NoMove_Flag', 'Scored']:
        if col in df_scores_out.columns:
            unique_vals = df_scores_out[col].unique()
            invalid = df_scores_out[~df_scores_out[col].isin([True, False])]
            logging.info(f"🧪 {col} unique values: {unique_vals}")
            if not invalid.empty:
                logging.warning(f"⚠️ Invalid boolean values in {col}:\n{invalid[[col]].drop_duplicates().to_string(index=False)}")
    
    # === Final upload to BigQuery
    try:
        pa.Table.from_pandas(df_scores_out)
    except Exception as e:
        logging.error("❌ Parquet conversion failure before upload:")
        logging.error(str(e))
        for col in df_scores_out.columns:
            logging.info(f"🔍 {col} → {df_scores_out[col].dtype}, sample: {df_scores_out[col].dropna().unique()[:5].tolist()}")
    
    # === Final logging and clean up before upload
    df_scores_out['Sharp_Time_Score'] = pd.to_numeric(df_scores_out['Sharp_Time_Score'], errors='coerce')
    df_scores_out['Sharp_Limit_Total'] = pd.to_numeric(df_scores_out['Sharp_Limit_Total'], errors='coerce')
    df_scores_out['Value'] = pd.to_numeric(df_scores_out['Value'], errors='coerce')
    
    df_scores_out['SharpBetScore'] = pd.to_numeric(df_scores_out['SharpBetScore'], errors='coerce')
    df_scores_out['Enhanced_Sharp_Confidence_Score'] = pd.to_numeric(df_scores_out['Enhanced_Sharp_Confidence_Score'], errors='coerce')
    df_scores_out['True_Sharp_Confidence_Score'] = pd.to_numeric(df_scores_out['True_Sharp_Confidence_Score'], errors='coerce')
    df_scores_out['SHARP_HIT_BOOL'] = pd.to_numeric(df_scores_out['SHARP_HIT_BOOL'], errors='coerce').astype('Int64')
    df_scores_out['SHARP_COVER_RESULT'] = df_scores_out['SHARP_COVER_RESULT'].fillna('').astype(str)
    
    df_scores_out['Sport'] = df_scores_out['Sport'].fillna('').astype(str)
    df_scores_out['Unique_Sharp_Books'] = pd.to_numeric(df_scores_out['Unique_Sharp_Books'], errors='coerce').fillna(0).astype(int)
    df_scores_out['First_Line_Value'] = pd.to_numeric(df_scores_out['First_Line_Value'], errors='coerce')
    df_scores_out['First_Sharp_Prob'] = pd.to_numeric(df_scores_out['First_Sharp_Prob'], errors='coerce')
    df_scores_out['Line_Delta'] = pd.to_numeric(df_scores_out['Line_Delta'], errors='coerce')
    df_scores_out['Model_Prob_Diff'] = pd.to_numeric(df_scores_out['Model_Prob_Diff'], errors='coerce')
    df_scores_out['Direction_Aligned'] = pd.to_numeric(df_scores_out['Direction_Aligned'], errors='coerce').fillna(0).round().astype('Int64')
    df_scores_out['First_Odds'] = pd.to_numeric(df_scores_out['First_Odds'], errors='coerce')
    df_scores_out['First_Imp_Prob'] = pd.to_numeric(df_scores_out['First_Imp_Prob'], errors='coerce')
    df_scores_out['Odds_Shift'] = df_scores_out['Odds_Price'] - df_scores_out['First_Odds']
    df_scores_out['Implied_Prob_Shift'] = df_scores_out['Implied_Prob'] - df_scores_out['First_Imp_Prob']
    df_scores_out['Was_Line_Resistance_Broken'] = pd.to_numeric(df_scores_out['Was_Line_Resistance_Broken'], errors='coerce').astype('Int64')
    df_scores_out['SharpMove_Resistance_Break'] = pd.to_numeric(df_scores_out['SharpMove_Resistance_Break'], errors='coerce').astype('Int64')
    df_scores_out['Line_Resistance_Crossed_Count'] = pd.to_numeric(df_scores_out['Line_Resistance_Crossed_Count'], errors='coerce').astype('Int64')
    df_scores_out['Odds_Reversal_Flag'] = pd.to_numeric(df_scores_out['Odds_Reversal_Flag'], errors='coerce').astype('Int64')

    # === Final upload
    try:
        df_weights = compute_and_write_market_weights(df_scores_out[df_scores_out['Scored']])
        logging.info(f"✅ Computed updated market weights for {sport_label.upper()}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to compute market weights: {e}")
    
    # Final deduplication and uploading to BigQuery
    pre_dedup_count = len(df_scores_out)
    logging.info(f"🧪 Before dedup: {pre_dedup_count} rows in df_scores_out")
    logging.info(f"🧪 Sports in df_scores_out: {df_scores_out['Sport'].unique().tolist()}")
    logging.info(f"🧪 Snapshot_Timestamp range: {df_scores_out['Snapshot_Timestamp'].min()} to {df_scores_out['Snapshot_Timestamp'].max()}")
    
    # Deduplication fingerprint columns
    dedup_fingerprint_cols = [
        'Game_Key', 'Bookmaker', 'Market', 'Outcome', 'Ref_Sharp_Value',
        'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
        'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
        'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
        'Unique_Sharp_Books', 'Enhanced_Sharp_Confidence_Score',
        'True_Sharp_Confidence_Score', 'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT',
        'Scored', 'Sport', 'Value', 'Merge_Key_Short',
        'First_Line_Value', 'First_Sharp_Prob',
        'Line_Delta', 'Model_Prob_Diff', 'Direction_Aligned',
        'Home_Team_Norm', 'Away_Team_Norm', 'Commence_Hour',
        'Line_Magnitude_Abs', 'High_Limit_Flag','Model_Sharp_Win_Prob',
        'Line_Move_Magnitude', 'Is_Home_Team_Bet', 'Is_Favorite_Bet','Model_Sharp_Win_Prob']
    
    logging.info(f"🧪 Fingerprint dedup keys: {dedup_fingerprint_cols}")
    float_cols_to_round = [
        'Sharp_Prob_Shift', 'Sharp_Time_Score', 'Sharp_Limit_Total', 'Value',
        'First_Line_Value', 'First_Sharp_Prob', 'Line_Delta', 'Model_Prob_Diff'
    ]

  
    for col in float_cols_to_round:
        if col in df_scores_out.columns:
            df_scores_out[col] = pd.to_numeric(df_scores_out[col], errors='coerce').round(4)
    # === 🧹 Canonical row filtering BEFORE fingerprint dedup
    canonical_sort_order = [
       'Scored',  # Prefer scored
       'Snapshot_Timestamp',  # Most recent
       'Model_Sharp_Win_Prob',  # Strongest signal
    ]
    
    df_scores_out = df_scores_out.sort_values(
       by=[col for col in canonical_sort_order if col in df_scores_out.columns],
       ascending=[False] * len(canonical_sort_order)
    )
    
    # Keep best row per game/book/market/outcome
    df_scores_out = df_scores_out.drop_duplicates(
       subset=['Game_Key', 'Bookmaker', 'Market', 'Outcome'], keep='first'
    )
    
    logging.info(f"🧪 Local rows before dedup: {len(df_scores_out)}")

    # === Query BigQuery for existing line-level deduplication
    existing = bq_client.query(f"""
        SELECT DISTINCT {', '.join(dedup_fingerprint_cols)}
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    """).to_dataframe()
    logging.info(f"🧪 Existing fingerprinted rows in BigQuery: {len(existing)}")
    
    # === Fetch all games already scored (to block re-inserts)
    scored_keys_df = bq_client.query("""
        SELECT DISTINCT Merge_Key_Short
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE LOWER(CAST(Scored AS STRING)) = 'true'
    """).to_dataframe()
    already_scored_keys = set(scored_keys_df['Merge_Key_Short'].dropna())
    
    # === Filter out already-scored games
    pre_score_filter = len(df_scores_out)
    # Allow rescore for tracking time-based line evolution
    if track_feature_evolution:
        logging.info("⏱️ Allowing rescore for time-series tracking mode")
    else:
        df_scores_out = df_scores_out[~df_scores_out['Merge_Key_Short'].isin(already_scored_keys)]

    logging.info(f"🧹 Removed {pre_score_filter - len(df_scores_out)} rows from already-scored games")
    
    # === Deduplicate against BigQuery fingerprinted rows
    pre_dedup = len(df_scores_out)
    df_scores_out = df_scores_out.merge(
        existing,
        on=dedup_fingerprint_cols,
        how='left',
        indicator=True
    )
    df_scores_out = df_scores_out[df_scores_out['_merge'] == 'left_only'].drop(columns=['_merge'])
    logging.info(f"🧹 Removed {pre_dedup - len(df_scores_out)} duplicate line-level rows based on fingerprint keys")
    
    # === Final logs and early exit
    if df_scores_out.empty:
        logging.info("ℹ️ No new scores to upload — all rows were already scored or duplicate line states.")
        return pd.DataFrame()
    logging.info("🔎 Sample value/odds extremes:\n" +
        df_scores_out[['Game_Key', 'Value', 'Max_Value', 'Min_Value', 'Odds_Price', 'Max_Odds', 'Min_Odds']].dropna().head(10).to_string(index=False))

    # === Log preview before upload
    logging.info(f"✅ Final rows to upload: {len(df_scores_out)}")
    logging.info("🧪 Sample rows to upload:\n" +
                 df_scores_out[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Snapshot_Timestamp']].head(5).to_string(index=False))
    
    # === Parquet validation (debug aid)
    try:
        pa.Table.from_pandas(df_scores_out)
    except Exception as e:
        logging.error("❌ Parquet conversion failure before upload:")
        logging.error(str(e))
        for col in df_scores_out.columns:
            logging.info(f"🔍 {col} → {df_scores_out[col].dtype}, sample: {df_scores_out[col].dropna().unique()[:5].tolist()}")
    
    # === Return final deduplicated and filtered DataFrame
    return df_scores_out


def compute_and_write_market_weights(df):
    component_cols = [
        'Sharp_Move_Signal',
        'Sharp_Limit_Jump',
        'Sharp_Prob_Shift',
        'Sharp_Time_Score',
        'Sharp_Limit_Total'
    ]

    rows = []

    for comp in component_cols:
        if comp not in df.columns:
            continue

        temp = df[['Market', comp, 'SHARP_HIT_BOOL']].dropna().copy()
        temp['Market'] = temp['Market'].astype(str).str.lower()
        temp['Component'] = comp

        # Convert values to bucketed strings
        if comp == 'Sharp_Limit_Total':
            temp['Value'] = temp[comp].astype(float).apply(lambda x: str(int(x // 1000 * 1000)))
        else:
            temp['Value'] = temp[comp].apply(
                lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x).strip().lower()
            )

        # Group and average win rates
        grouped = (
            temp.groupby(['Market', 'Component', 'Value'])['SHARP_HIT_BOOL']
            .mean()
            .reset_index()
            .rename(columns={'SHARP_HIT_BOOL': 'Win_Rate'})
        )

        for _, row in grouped.iterrows():
            rows.append({
                'Market': row['Market'],
                'Component': row['Component'],
                'Value': row['Value'],
                'Win_Rate': round(row['Win_Rate'], 4)
            })

    if not rows:
        print("⚠️ No valid market weights to upload.")
        return {}

    df_weights = pd.DataFrame(rows)
    print(f"✅ Prepared {len(df_weights)} market weight rows. Sample:")
    print(df_weights.head(10).to_string(index=False))

    # Return as nested dict for use in scoring
    market_weights = {}
    for _, row in df_weights.iterrows():
        market = row['Market']
        comp = row['Component']
        val = row['Value']
        rate = row['Win_Rate']
        market_weights.setdefault(market, {}).setdefault(comp, {})[val] = rate

    return market_weights

def compute_sharp_prob_shift(df):
    """
    Computes the change in Model_Sharp_Win_Prob per Team_Key + Bookmaker across snapshots.
    Adds column: Sharp_Prob_Shift
    """
    df = df.copy()

    required = ['Team_Key', 'Bookmaker', 'Snapshot_Timestamp', 'Model_Sharp_Win_Prob']
    if not all(col in df.columns for col in required):
        df['Sharp_Prob_Shift'] = 0.0
        return df

    df['Snapshot_Timestamp'] = pd.to_datetime(df['Snapshot_Timestamp'], utc=True, errors='coerce')
    df = df.sort_values(['Team_Key', 'Bookmaker', 'Snapshot_Timestamp'])

    df['Sharp_Prob_Shift'] = (
        df.groupby(['Team_Key', 'Bookmaker'])['Model_Sharp_Win_Prob']
        .transform(lambda x: x.diff().fillna(0))
    )

    return df

