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

from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from google.cloud import bigquery, storage
import logging
logging.basicConfig(level=logging.INFO)  # <- Must be INFO or DEBUG to show .info() logs
logger = logging.getLogger(__name__)
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
SHARP_BOOKS = SHARP_BOOKS_FOR_LIMITS + ['betus','mybookieag','betfair_ex_eu','betfair_ex_uk','lowvig']

REC_BOOKS = [
    'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers',
    'fanatics', 'espnbet', 'hardrockbet']


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
            str(row.get('Sharp_Move_Signal', '')).strip(),
            str(row.get('Sharp_Limit_Total', '')).strip(),
            str(row.get('Open_Value', '')).strip(),
            str(row.get('Snapshot_Timestamp', '')).strip(),  # ✅ NEW LINE
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
        'Is_Reinforced_MultiMarket',

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
        'Value_Reversal_Flag', 'Odds_Reversal_Flag','Open_Odds'      # ✅ Add this
    ]
    # 🧩 Add schema-consistent consensus fields from summarize_consensus()
     
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
    for rec in rec_books:
        if rec.replace(" ", "") in raw_key:
            return rec.replace(" ", "")
    for sharp in sharp_books:
        if sharp in raw_key:
            return sharp
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
    
    
def compute_sharp_metrics(entries, open_val, mtype, label):
    move_signal = 0
    move_magnitude_score = 0.0
    limit_score = 0.0
    time_score = 0.0
    total_limit = 0.0
    entry_count = 0

    for limit, curr, ts in entries:
        if open_val is not None and curr is not None:
            try:
                delta = curr - open_val
                sharp_move_delta = abs(delta)
        
                # Total movement magnitude
                if sharp_move_delta >= 0.01:
                    move_magnitude_score += sharp_move_delta
        
                # Directional sharp move scoring (weighted)
                if mtype == 'totals':
                    if 'under' in label and curr < open_val:
                        move_signal += sharp_move_delta
                    elif 'over' in label and curr > open_val:
                        move_signal += sharp_move_delta
                elif mtype == 'spreads':
                    # Case 1: Favorite getting more favored (e.g. -3.5 → -5)
                    if open_val < 0 and curr < open_val:
                        move_signal += sharp_move_delta
                
                    # Case 2: Underdog getting more underdog (e.g. +3.5 → +5.5)
                    elif open_val > 0 and curr > open_val:
                        move_signal += sharp_move_delta
        
            except Exception:
                continue



        # Weight limit jumps by size
        if limit is not None and limit >= 100:
            limit_score += limit
            total_limit += limit
        elif limit is not None:
            total_limit += limit

        # Weighted timing
        try:
            hour = pd.to_datetime(ts).hour
            timing_weight = (
                1.0 if 0 <= hour <= 5 else
                0.9 if 6 <= hour <= 11 else
                0.5 if 12 <= hour <= 15 else
                0.2
            )
            time_score += timing_weight
        except:
            time_score += 0.5

        entry_count += 1

    # Normalize to avoid overweighting if many entries
    move_magnitude_score = min(move_magnitude_score, 5.0)
    time_score = round(time_score / max(1, entry_count), 2)
    limit_score = round(limit_score / max(1, entry_count), 2)

    return {
        'Sharp_Move_Signal': round(move_signal, 2),
        'Sharp_Limit_Jump': round(limit_score, 2),
        'Sharp_Time_Score': time_score,
        'Sharp_Limit_Total': total_limit,
        'Sharp_Move_Magnitude_Score': round(move_magnitude_score, 2),
        'SharpBetScore': round(
            2.0 * move_signal +
            2.0 * limit_score +
            1.5 * time_score +
            0.001 * total_limit +
            3.0 * move_magnitude_score, 2
        )
    }

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def apply_sharp_scoring(rows, sharp_limit_map, line_open_map, sharp_total_limit_map):
    for r in rows:
        # Identify the key for this market
        game_key = (r['Game'], r['Market'], r['Outcome'])

        # Get all relevant line movements for this label
        entry_group = sharp_limit_map.get((r['Game'], r['Market']), {}).get(r['Outcome'], [])

        # Get the original opening line from the stored line_open_map
        open_val = line_open_map.get(game_key, (None,))[0]

        # Compute sharp signals and update the row dict
        r.update(compute_sharp_metrics(entry_group, open_val, r['Market'], r['Outcome']))

    return rows

    

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SPORT_ALIAS = {
    'AMERICANFOOTBALL_CFL': 'CFL',
    'BASEBALL_MLB': 'MLB',
    'BASKETBALL_WNBA': 'WNBA',
    'AMERICANFOOTBALL_NFL': 'NFL',
    'AMERICANFOOTBALL_NCAAF': 'NCAAF',
    'BASKETBALL_NBA': 'NBA',
    'BASKETBALL_NCAAB': 'NCAAB',
}

KEY_LINE_RESISTANCE = {
    'NFL': {'spread': [3, 7, 10, 14], 'total': [41, 44, 47, 51]},
    'NBA': {'spread': [2.5, 5, 7, 10], 'total': [210, 220, 225, 230]},
    'WNBA': {'spread': [2, 4.5, 6.5], 'total': [155, 160, 165, 170]},
    'CFL': {'spread': [3, 6.5, 9.5], 'total': [48, 52, 55, 58]},
    'NCAAF': {'spread': [3, 7, 10, 14, 17], 'total': [45, 52, 59, 66]},
    'NCAAB': {'spread': [2, 5, 7, 10], 'total': [125, 135, 145, 150]},
    'MLB': {'spread': [], 'total': [7, 7.5, 8.5, 9]},
    'NHL': {'spread': [], 'total': [5.5, 6, 6.5, 7]},
}


def was_line_resistance_broken(open_val, close_val, key_levels):
    if open_val is None or close_val is None:
        return 0
    for key in key_levels:
        if (open_val < key < close_val) or (close_val < key < open_val):
            return 1
    return 0




def compute_line_resistance_flag(df, source='moves'):
    # Normalize Sport using alias map
    df['Sport'] = df['Sport'].str.upper().map(SPORT_ALIAS).fillna(df['Sport'].str.upper())

    def get_key_levels(sport, market):
        if not sport or not market:
            return []
        sport_key = sport.upper()
        market_key = market.lower()
        return KEY_LINE_RESISTANCE.get(sport_key, {}).get(market_key, [])

    def get_opening_line(row):
        if source == 'moves':
            return row.get('Old_Value')
        elif source == 'scores':
            return row.get('First_Line_Value') if 'First_Line_Value' in row else None
        return None

    df['Was_Line_Resistance_Broken'] = df.apply(
        lambda row: was_line_resistance_broken(
            get_opening_line(row),
            row.get('Value'),
            get_key_levels(row.get('Sport', ''), row.get('Market', ''))
        ),
        axis=1
    )

    return df
    
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

def apply_blended_sharp_score(df, trained_models, df_all_snapshots=None, weights=None):
    logger.info("🛠️ Running `apply_blended_sharp_score()`")

    df = df.copy()
    scored_all = []
    total_start = time.time()

    df['Market'] = df['Market'].astype(str).str.lower().str.strip()
    df['Is_Sharp_Book'] = df['Bookmaker'].isin(SHARP_BOOKS).astype(int)

    # Drop merge artifacts
    try:
        df = df.drop(columns=[col for col in df.columns if col.endswith(('_x', '_y'))], errors='ignore')
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return pd.DataFrame()

    # ✅ Use passed history or fallback
    if df_all_snapshots is None:
        df_all_snapshots = read_recent_sharp_moves(hours=72)
    # Drop leftover merge artifacts
    # ✅ Sanity check: Unique outcomes and books
    logger.info(f"🧪 Unique outcomes in snapshot: {df_all_snapshots['Outcome'].nunique()} — {df_all_snapshots['Outcome'].unique().tolist()}")
    logger.info(f"🧪 Unique books in snapshot: {df_all_snapshots['Bookmaker'].nunique()} — {df_all_snapshots['Bookmaker'].unique().tolist()}")
    
    # ✅ Count outcomes per Game + Market + Bookmaker
    outcome_counts = (
        df_all_snapshots
        .groupby(['Game_Key', 'Market', 'Bookmaker'])['Outcome']
        .nunique()
        .reset_index()
        .rename(columns={'Outcome': 'Num_Outcomes'})
    )
    
    # ✅ Log any books missing one side of market
    missing_outcomes = outcome_counts[outcome_counts['Num_Outcomes'] < 2]
    if not missing_outcomes.empty:
        logger.warning(f"⚠️ Some (Game, Market, Bookmaker) combos are missing one side — {len(missing_outcomes)} rows")
        logger.debug(f"{missing_outcomes.head(10).to_string(index=False)}")
    else:
        logger.info("✅ All Game + Market + Bookmaker combinations have both outcomes present.")
    
    # === Load full sharp move history for enrichment
   
    df = compute_line_resistance_flag(df, source='moves')
    df = add_minutes_to_game(df)

    # Normalize merge keys
    for col in ['Game_Key', 'Market', 'Outcome', 'Bookmaker']:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df_all_snapshots[col] = df_all_snapshots[col].astype(str).str.strip().str.lower()

   # Opening values per outcome + book
    df_open = (
        df_all_snapshots
        .sort_values('Snapshot_Timestamp')
        .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='first')
        .loc[:, ['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Value', 'Odds_Price', 'Implied_Prob', 'Limit']]
        .rename(columns={
            'Value': 'Open_Value',
            'Odds_Price': 'Open_Odds',
            'Implied_Prob': 'First_Imp_Prob',
            'Limit': 'Opening_Limit'
        })
    )
    
    # Extremes per outcome + book
    df_extremes = (
        df_all_snapshots
        .groupby(['Game_Key', 'Market', 'Outcome', 'Bookmaker'])[['Value', 'Odds_Price']]
        .agg(
            Max_Value=('Value', 'max'),
            Min_Value=('Value', 'min'),
            Max_Odds=('Odds_Price', 'max'),
            Min_Odds=('Odds_Price', 'min')
        )
        .reset_index()
    )
    
    # Merge
    line_enrichment = df_open.merge(df_extremes, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left')
    df = df.merge(line_enrichment, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left')
    

    # === Compute shifts
    df['Odds_Shift'] = pd.to_numeric(df['Odds_Price'], errors='coerce') - pd.to_numeric(df['Open_Odds'], errors='coerce')
    df['Implied_Prob_Shift'] = pd.to_numeric(df['Implied_Prob'], errors='coerce') - pd.to_numeric(df['First_Imp_Prob'], errors='coerce')
    # Compute deltas
    df['Line_Delta'] = pd.to_numeric(df['Value'], errors='coerce') - pd.to_numeric(df['Open_Value'], errors='coerce')
    
    
    # === Clean columns
    df.drop(columns=['First_Imp_Prob'], inplace=True, errors='ignore')

    # === Delta and reversal flags
    df['Delta'] = df['Value'] - df['Open_Value']
    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)
    df['Line_Magnitude_Abs'] = df['Line_Delta'].abs()
    df['Line_Move_Magnitude'] = df['Line_Delta'].abs()
    df['Value_Reversal_Flag'] = (
        ((df['Value'] < df['Open_Value']) & (df['Value'] == df['Min_Value'])) |
        ((df['Value'] > df['Open_Value']) & (df['Value'] == df['Max_Value']))
    ).astype(int)

    df['Odds_Reversal_Flag'] = (
        ((df['Odds_Price'] < df['Open_Odds']) & (df['Odds_Price'] == df['Min_Odds'])) |
        ((df['Odds_Price'] > df['Open_Odds']) & (df['Odds_Price'] == df['Max_Odds']))
    ).astype(int)
    df['LimitUp_NoMove_Flag'] = (
        (df['Sharp_Limit_Total'] > 5000) &
        (df['Sharp_Move_Signal'] < 0.05) &
        (df['Line_Delta'].abs() < 0.1)
    ).astype(int)
    
        
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
    
    logger.info(f"✅ Snapshot enrichment complete — rows: {len(df)}")
    logger.info(f"📊 Columns present after enrichment: {df.columns.tolist()}")

   

    for market_type, bundle in trained_models.items():
        try:
            model = bundle.get('model')
            iso = bundle.get('calibrator')
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
            df_canon['Minutes_To_Game'] = (
                pd.to_datetime(df_canon['Game_Start'], utc=True) - pd.to_datetime(df_canon['Snapshot_Timestamp'], utc=True)
            ).dt.total_seconds() / 60
            
            df_canon['Late_Game_Steam_Flag'] = (df_canon['Minutes_To_Game'] <= 60).astype(int)
            
            df_canon['Minutes_To_Game_Tier'] = pd.cut(
                df_canon['Minutes_To_Game'],
                bins=[-1, 30, 60, 180, 360, 720, np.inf],
                labels=['🚨 ≤30m', '🔥 ≤1h', '⚠️ ≤3h', '⏳ ≤6h', '📅 ≤12h', '🕓 >12h']
            )
            df_canon['Value_Reversal_Flag'] = df_canon.get('Value_Reversal_Flag', 0).astype(int)
            df_canon['Odds_Reversal_Flag'] = df_canon.get('Odds_Reversal_Flag', 0).astype(int)
            # === Ensure required features exist ===
            model_features = model.get_booster().feature_names
            missing_cols = [col for col in model_features if col not in df_canon.columns]
            df_canon[missing_cols] = 0
            
            # === Align features to model input
            X_canon = df_canon[model_features].replace({'True': 1, 'False': 0}).apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            
            ## ✅ USE THE CALIBRATED PROBABILITIES FOR MAIN MODEL OUTPUT
            df_canon['Model_Sharp_Win_Prob'] = trained_models[market_type]['calibrator'].predict_proba(X_canon)[:, 1]
            
            # Optional: you can drop Model_Confidence entirely or alias it to the same
            df_canon['Model_Confidence'] = df_canon['Model_Sharp_Win_Prob']

            # === Tag for downstream usage
            df_canon['Was_Canonical'] = True
            df_canon['Scoring_Market'] = market_type
            df_canon['Scored_By_Model'] = True
            
            

            df_inverse = df_canon.copy(deep=True)
            df_inverse['Model_Sharp_Win_Prob'] = 1 - df_inverse['Model_Sharp_Win_Prob']
            df_inverse['Model_Confidence'] = 1 - df_inverse['Model_Confidence']
            df_inverse['Was_Canonical'] = False
            df_inverse['Scored_By_Model'] = True
            # === Core deltas and magnitude features
            df_inverse['Line_Move_Magnitude'] = pd.to_numeric(df_inverse['Line_Delta'], errors='coerce').abs()
            df_inverse['Line_Magnitude_Abs'] = df_inverse['Line_Move_Magnitude']  # Alias
            
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
            df_inverse['Was_Line_Resistance_Broken'] = df_inverse.get('Was_Line_Resistance_Broken', 0).fillna(0).astype(int)
            df_inverse['SharpMove_Resistance_Break'] = (
                df_inverse['Sharp_Move_Signal'] * df_inverse['Was_Line_Resistance_Broken']
            )
            # Compute minutes until game start
            df_inverse['Minutes_To_Game'] = (
                pd.to_datetime(df_inverse['Game_Start'], utc=True) -
                pd.to_datetime(df_inverse['Snapshot_Timestamp'], utc=True)
            ).dt.total_seconds() / 60
            
            # Flag: Is this a late steam move?
            df_inverse['Late_Game_Steam_Flag'] = (df_inverse['Minutes_To_Game'] <= 60).astype(int)
            
            # Bucketed tier for diagnostics or categorical modeling
            df_inverse['Minutes_To_Game_Tier'] = pd.cut(
                df_inverse['Minutes_To_Game'],
                bins=[-1, 30, 60, 180, 360, 720, np.inf],
                labels=['🚨 ≤30m', '🔥 ≤1h', '⚠️ ≤3h', '⏳ ≤6h', '📅 ≤12h', '🕓 >12h']
            )
            df_inverse['Value_Reversal_Flag'] = df_canon['Value_Reversal_Flag'].values
            df_inverse['Odds_Reversal_Flag'] = df_canon['Odds_Reversal_Flag'].values
            if market_type == "totals":
                df_inverse = df_inverse[df_inverse['Outcome'] == 'over'].copy()
                df_inverse['Outcome'] = 'under'
                df_inverse['Outcome_Norm'] = 'under'
            
                logger.info(f"🔁 Attempting to create {len(df_inverse)} inverse UNDER rows from OVER rows.")
            
                original_books = set(df_market['Bookmaker'].unique())
                inverse_books = set(df_inverse['Bookmaker'].unique())
                missing_books = original_books - inverse_books
                if missing_books:
                    logger.warning(f"⚠️ These books had 'under' rows but no inverse created: {missing_books}")
            
                df_inverse['Team_Key'] = df_inverse['Game_Key_Base'] + "_" + df_inverse['Outcome']
                df_full_market['Team_Key'] = df_full_market['Game_Key_Base'] + "_" + df_full_market['Outcome']
            
                df_full_market = (
                    df_full_market
                    .dropna(subset=['Value', 'Odds_Price'])
                    .sort_values(['Snapshot_Timestamp', 'Bookmaker'])
                    .drop_duplicates(subset=['Team_Key'], keep='last')
                )
            
                df_inverse = df_inverse.merge(
                    df_full_market[['Team_Key', 'Value', 'Odds_Price']],
                    on='Team_Key',
                    how='left',
                    suffixes=('', '_opponent')
                )
            
                df_inverse['Value'] = df_inverse['Value_opponent']
                df_inverse['Odds_Price'] = df_inverse['Odds_Price_opponent']
                df_inverse.drop(columns=['Value_opponent', 'Odds_Price_opponent'], inplace=True, errors='ignore')
            
                # Diagnostic
                missing_val = df_inverse['Value'].isna().sum()
                missing_odds = df_inverse['Odds_Price'].isna().sum()
                logger.warning(f"⚠️ totals inverse: {missing_val} missing Value, {missing_odds} missing Odds_Price")

            elif market_type == "h2h":
                df_inverse['Canonical_Team'] = df_inverse['Outcome'].str.lower().str.strip()
                df_full_market['Outcome'] = df_full_market['Outcome'].str.lower().str.strip()
            
                df_inverse['Outcome'] = np.where(
                    df_inverse['Canonical_Team'] == df_inverse['Home_Team_Norm'],
                    df_inverse['Away_Team_Norm'],
                    df_inverse['Home_Team_Norm']
                )
                df_inverse['Outcome'] = df_inverse['Outcome'].str.lower().str.strip()
                df_inverse['Outcome_Norm'] = df_inverse['Outcome']
            
                df_inverse['Commence_Hour'] = pd.to_datetime(df_inverse['Game_Start'], utc=True, errors='coerce').dt.floor('h')
                df_inverse['Game_Key'] = (
                    df_inverse['Home_Team_Norm'] + "_" +
                    df_inverse['Away_Team_Norm'] + "_" +
                    df_inverse['Commence_Hour'].astype(str) + "_" +
                    df_inverse['Market'] + "_" +
                    df_inverse['Outcome']
                )
                df_inverse['Game_Key_Base'] = (
                    df_inverse['Home_Team_Norm'] + "_" +
                    df_inverse['Away_Team_Norm'] + "_" +
                    df_inverse['Commence_Hour'].astype(str) + "_" +
                    df_inverse['Market']
                )
            
                df_inverse['Team_Key'] = df_inverse['Game_Key_Base'] + "_" + df_inverse['Outcome']
                df_full_market['Team_Key'] = df_full_market['Game_Key_Base'] + "_" + df_full_market['Outcome']
            
                df_full_market = (
                    df_full_market
                    .dropna(subset=['Value'])
                    .sort_values(['Snapshot_Timestamp', 'Bookmaker'])
                    .drop_duplicates(subset=['Team_Key'], keep='last')
                )
            
                df_inverse = df_inverse.merge(
                    df_full_market[['Team_Key', 'Value', 'Odds_Price']],
                    on='Team_Key',
                    how='left',
                    suffixes=('', '_opponent')
                )
            
                df_inverse['Value'] = df_inverse['Value_opponent']
                df_inverse['Odds_Price'] = df_inverse['Odds_Price_opponent']
                df_inverse.drop(columns=['Value_opponent', 'Odds_Price_opponent'], inplace=True, errors='ignore')
            
                missing_val = df_inverse['Value'].isna().sum()
                missing_odds = df_inverse['Odds_Price'].isna().sum()
                logger.warning(f"⚠️ {missing_val} h2h inverse rows missing Value, {missing_odds} missing Odds_Price after merge.")
            
                df_inverse = df_inverse.drop_duplicates(subset=['Game_Key', 'Market', 'Bookmaker', 'Outcome'])



            elif market_type == "spreads":
                
                df_inverse['Canonical_Team'] = df_inverse['Outcome'].str.lower().str.strip()
                df_full_market['Outcome'] = df_full_market['Outcome'].str.lower().str.strip()
            
                df_inverse['Outcome'] = np.where(
                    df_inverse['Canonical_Team'] == df_inverse['Home_Team_Norm'],
                    df_inverse['Away_Team_Norm'],
                    df_inverse['Home_Team_Norm']
                )
                df_inverse['Outcome'] = df_inverse['Outcome'].str.lower().str.strip()
                df_inverse['Outcome_Norm'] = df_inverse['Outcome']
            
                # Rebuild Game_Key and Game_Key_Base using flipped outcome
                df_inverse['Commence_Hour'] = pd.to_datetime(df_inverse['Game_Start'], utc=True, errors='coerce').dt.floor('h')
                df_inverse['Game_Key'] = (
                    df_inverse['Home_Team_Norm'] + "_" +
                    df_inverse['Away_Team_Norm'] + "_" +
                    df_inverse['Commence_Hour'].astype(str) + "_" +
                    df_inverse['Market'] + "_" +
                    df_inverse['Outcome']
                )
                df_inverse['Game_Key_Base'] = (
                    df_inverse['Home_Team_Norm'] + "_" +
                    df_inverse['Away_Team_Norm'] + "_" +
                    df_inverse['Commence_Hour'].astype(str) + "_" +
                    df_inverse['Market']
                )
                  # ✅ Build Team_Key for safe merge
                df_inverse['Team_Key'] = df_inverse['Game_Key_Base'] + "_" + df_inverse['Outcome']
                df_full_market['Team_Key'] = df_full_market['Game_Key_Base'] + "_" + df_full_market['Outcome']
            
                # ✅ Merge opponent Value cleanly
                # Before merge, deduplicate df_full_market so each Team_Key has only one value
                df_full_market = (
                    df_full_market
                    .dropna(subset=['Value'])
                    .sort_values(['Snapshot_Timestamp', 'Bookmaker'])  # You can customize sorting if needed
                    .drop_duplicates(subset=['Team_Key'], keep='last')  # ⬅️ Keep only the most recent line
                )
                
                df_inverse = df_inverse.merge(
                    df_full_market[['Team_Key', 'Value', 'Odds_Price']],
                    on='Team_Key',
                    how='left',
                    suffixes=('', '_opponent')
                )
                
                df_inverse['Value'] = df_inverse['Value_opponent']
                df_inverse['Odds_Price'] = df_inverse['Odds_Price_opponent']
                df_inverse.drop(columns=['Value_opponent', 'Odds_Price_opponent'], inplace=True, errors='ignore')

            
                # Final deduplication
                df_inverse = df_inverse.drop_duplicates(subset=['Game_Key', 'Market', 'Bookmaker', 'Outcome'])

            if df_inverse.empty:
                logger.warning("⚠️ No inverse rows generated — check canonical filtering or flip logic.")
                continue  # optional: skip this scoring loop if inverse fails

           
            # ✅ Combine canonical and inverse into one scored DataFrame
            df_scored = pd.concat([df_canon, df_inverse], ignore_index=True)
            
            
            

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
    
        if scored_all:
       
            df_final = pd.concat(scored_all, ignore_index=True)
            df_final = df_final[df_final['Model_Sharp_Win_Prob'].notna()]
        
            # === 🔍 Diagnostic for unscored rows
            try:
                unscored_df = df[df['Game_Key'].isin(set(df['Game_Key']) - set(df_final['Game_Key']))]
                if not unscored_df.empty:
                    logger.warning(f"⚠️ {len(unscored_df)} rows were not scored.")
                    logger.warning("🔍 Breakdown by market type:")
                    logger.warning(unscored_df['Market'].value_counts().to_string())
        
                    logger.warning("🧪 Sample of unscored rows:")
                    logger.warning(unscored_df[['Game', 'Bookmaker', 'Market', 'Outcome', 'Value']].head(5).to_string(index=False))
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

    
            # === 🔍 Debug unscored rows
            try:
                original_keys = set(df['Game_Key'])
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


        


def detect_sharp_moves(current, previous, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, trained_models, weights=None):   
    
    if not current:
        logging.warning("⚠️ No current odds data provided.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}
    df_history = read_recent_sharp_moves(hours=72)
    df_history = df_history.dropna(subset=['Game', 'Market', 'Outcome', 'Book', 'Value'])
    df_history = df_history.sort_values('Snapshot_Timestamp')
    
    # ✅ Build old value map using first recorded value per outcome
    old_val_map = (
        df_history
        .drop_duplicates(subset=['Game', 'Market', 'Outcome', 'Book'], keep='first')
        .set_index(['Game', 'Market', 'Outcome', 'Book'])['Value']
        .to_dict()
    )
    rows = []
    sharp_limit_map = defaultdict(lambda: defaultdict(list))
    sharp_total_limit_map = defaultdict(int)
    sharp_lines = {}
    line_open_map = {}
    
    previous_odds_map = {}
    for g in previous_map.values():
        for book in g.get('bookmakers', []):
            book_key = book.get('key')
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

        game_name = f"{home_team.title()} vs {away_team.title()}"
        event_time = pd.to_datetime(game.get("commence_time"), utc=True, errors='coerce')
        game_hour = event_time.floor('h') if pd.notnull(event_time) else pd.NaT
        gid = game.get('id')

        for book in game.get('bookmakers', []):
            book_key_raw = book.get('key', '').lower()
            book_key = normalize_book_key(book_key_raw, SHARP_BOOKS, REC_BOOKS)

            if book_key not in SHARP_BOOKS + REC_BOOKS:
                continue

            book_title = book.get('title', book_key)

            for market in book.get('markets', []):
                mtype = market.get('key', '').strip().lower()
                if mtype not in ['spreads', 'totals', 'h2h']:
                    continue

                
                seen = {}
                canonical_outcomes = []
                odds_map = {}
                #logger.debug(f"Game: {game['home_team']} vs {game['away_team']} | Market: {mtype}")
                # === First pass to deduplicate and store odds
                for o in market.get('outcomes', []):
                    logging.debug(f"  Outcome: {o.get('name')} | Point: {o.get('point')} | Price: {o.get('price')}")
                    label = normalize_label(o.get('name', ''))
                    point = o.get('point')
                    price = o.get('price')
                    #logging.debug(f"[{mtype}] Outcome: {label} | Point: {point} | Price: {price}")
                
                    key = (label, point)
                    # Keep all outcomes — no deduping
                    canonical_outcomes.append(o)
                    odds_map[(normalize_label(o['name']), o.get('point'))] = o.get('price')

                
                # === Second pass to build entries
                for o in canonical_outcomes:
                    label = normalize_label(o.get('name', ''))  # ✅ FIXED: per-outcome label
                    raw_label = o.get('name', '').strip().lower()
                    point = o.get('point')
                    price = o.get('price')
                
                    if mtype == 'h2h':
                        value = price
                        odds_price = price
                    else:
                        value = point
                        odds_price = odds_map.get((label, point))  # ✅ ensure odds match label+point
                    #logging.debug(f"{label} {point}: odds_price = {odds_price}")

                
                    limit = o.get('bet_limit')
                    prev_key = (game.get('home_team'), game.get('away_team'), mtype, label, book_key)
                    # ⚠️ Read the open value *before* possibly writing it
                    open_val = old_val_map.get((game_name, mtype, label, book_key))
                    
                    # ✅ Only set the open value if it's not already set
                    if (game_name, mtype, label) not in line_open_map and value is not None:
                        line_open_map[(game_name, mtype, label)] = (value, snapshot_time)
                    
                   
                    game_key = f"{home_team}_{away_team}_{str(game_hour)}_{mtype}_{label}"
                    
                    entry = {
                        'Sport': sport_key.upper(),
                        'Game_Key': game_key,
                        'Time': snapshot_time,
                        'Game': game_name,
                        'Game_Start': event_time,
                        'Market': mtype,
                        'Outcome': label,
                        'Bookmaker': book_title,
                        'Book': book_key,
                        'Value': value,
                        'Odds_Price': odds_price,
                        'Limit': limit,
                        'Old Value': None,
                        #'Delta': round(value - open_val, 2) if open_val is not None and value is not None else None,
                        'Home_Team_Norm': home_team,
                        'Away_Team_Norm': away_team,
                        'Commence_Hour': game_hour
                    }
                
                   
                    # ✅ Defensive check before adding the entry
                    required_fields = ['Game_Key', 'Game', 'Book', 'Market', 'Outcome', 'Value', 'Odds_Price']
                    if any(entry.get(f) is None for f in required_fields):
                        logging.warning(
                            f"⚠️ Skipping incomplete row: "
                            f"{game_name} | {mtype} | {label} | Book: {book_key} | Value: {value} | Odds: {odds_price}"
                        )
                        continue
                    
                    rows.append(entry)

       
                    logging.debug(
                        f"[{mtype.upper()}] {label} | Book: {book_key} | Value: {value} | Odds_Price: {odds_price} | "
                        f"Limit: {limit} | Game: {game_name}"
                    )

                    if value is not None:
                        sharp_lines[(game_name, mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit, value, open_val))
                        if book_key in SHARP_BOOKS:
                            sharp_total_limit_map[(game_name, mtype, label)] += limit or 0
                        if (game_name, mtype, label) not in line_open_map:
                            line_open_map[(game_name, mtype, label)] = (value, snapshot_time)

    pre_dedup = len(rows)
    rows_df = pd.DataFrame(rows).drop_duplicates()
    rows_df = (
        rows_df.sort_values('Time').drop_duplicates(subset=['Game', 'Bookmaker', 'Market', 'Outcome', 'Value', 'Odds_Price'], keep='last')

    )
    rows = rows_df.to_dict(orient='records')

    for r in rows:
        game_key = (r['Game'], r['Market'], r['Outcome'])
        entry_group = sharp_limit_map.get((r['Game'], r['Market']), {}).get(r['Outcome'], [])
        open_val = line_open_map.get(game_key, (None,))[0]
        sharp_scores = compute_sharp_metrics(entry_group, open_val, r['Market'], r['Outcome'])
        r.update(sharp_scores)

    logging.info(f"🧹 Deduplicated rows: {pre_dedup - len(rows)} duplicates removed")
    df = pd.DataFrame(rows)
    # === Compute Market Leader (if not already done)
    
    # ✅ Add required fields BEFORE scoring
    df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()
    df['Odds_Price'] = pd.to_numeric(df.get('Odds_Price'), errors='coerce')
    df['Implied_Prob'] = df['Odds_Price'].apply(implied_prob)
    df['Book'] = df['Book'].str.lower()
    # === Compute Market Leader (if not already done)
    
    # Protect against zero-move but high-limit situations
    
    
    # Only now run model scoring

    if trained_models:
        try:
            df_all_snapshots = read_recent_sharp_moves(hours=72)
            market_weights = load_market_weights_from_bq()
            df_scored = apply_blended_sharp_score(df.copy(), trained_models, df_all_snapshots, market_weights)
            if not df_scored.empty:
                df_scored['Game_Start'] = pd.to_datetime(df_scored['Game_Start'], errors='coerce', utc=True)
                now = pd.Timestamp.utcnow()
                df_scored['Pre_Game'] = df_scored['Game_Start'] > now
                df_scored['Post_Game'] = ~df_scored['Pre_Game']
                df_scored['Event_Date'] = df_scored['Game_Start'].dt.strftime('%Y-%m-%d')
                df_scored['Line_Hash'] = df_scored.apply(compute_line_hash, axis=1)
            
                df = df_scored.copy()
                logging.info(f"✅ Scored {len(df)} rows using apply_blended_sharp_score()")
                
            else:
                logging.warning("⚠️ apply_blended_sharp_score() returned no rows")
        except Exception as e:
            logging.error(f"❌ Error applying model scoring: {e}", exc_info=True)
    
 
    # === Build main DataFrame
    logging.info(f"📊 Columns after sharp scoring: {df.columns.tolist()}")

   
    

    # === Summary consensus metrics
    summary_df = summarize_consensus(df_scored, SHARP_BOOKS, REC_BOOKS)
   
    # ✅ Final return (no field names changed)
    return df, df_history, summary_df

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


def detect_cross_market_sharp_support(df_moves, SHARP_BOOKS, score_threshold=15):
    df = df_moves.copy()
    df['Market'] = df['Market'].astype(str).str.lower().str.strip()
    df['SupportKey'] = df['Game'].astype(str) + " | " + df['Outcome'].astype(str)

    df_sharp = df[df['SharpBetScore'] >= score_threshold].copy()

    # Cross-market count
    market_counts = (
        df_sharp.groupby('SupportKey')['Market']
        .nunique()
        .reset_index()
        .rename(columns={'Market': 'CrossMarketSharpSupport'})
    )

    # Sharp books per outcome
    sharp_book_counts = (
        df_sharp[df_sharp['Book'].isin(SHARP_BOOKS)]
        .groupby('SupportKey')['Book']
        .nunique()
        .reset_index()
        .rename(columns={'Book': 'Unique_Sharp_Books'})
    )

    # Merge and flag
    df = df.merge(market_counts, on='SupportKey', how='left')
    df = df.merge(sharp_book_counts, on='SupportKey', how='left')
    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Unique_Sharp_Books'] = df['Unique_Sharp_Books'].fillna(0).astype(int)

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
            "model": data["model"],
            "calibrator": data["calibrator"]
        }
    except Exception as e:
        logging.warning(f"⚠️ Could not load model from GCS for {sport}-{market}: {e}")
        return None

def read_recent_sharp_moves(hours=140, table=BQ_FULL_TABLE):
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
            'Game_Key', 'Bookmaker', 'Market', 'Outcome', 'Ref_Sharp_Value',
            'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
            'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
            'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
            'Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score',
            'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT', 'Scored', 'Snapshot_Timestamp',
            'Sport', 'Value', 'First_Line_Value', 'First_Sharp_Prob',
            'Line_Delta', 'Model_Prob_Diff', 'Direction_Aligned',
            'Unique_Sharp_Books', 'Merge_Key_Short',
            'Line_Magnitude_Abs',
            'Line_Move_Magnitude',
            'Is_Home_Team_Bet',
            'Is_Favorite_Bet',
            'High_Limit_Flag',
            'Home_Team_Norm',
            'Away_Team_Norm',
            'Commence_Hour','Model_Sharp_Win_Prob','Odds_Price','Implied_Prob', 'First_Odds', 'First_Imp_Prob', 'Odds_Shift','Implied_Prob_Shift',
            'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds' # ✅ Add this line
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
    df_master = read_recent_sharp_moves(hours=days_back * 72)
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
    df_all_snapshots = read_recent_sharp_moves(hours=days_back * 72)
    
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
        'Line_Magnitude_Abs', 'High_Limit_Flag',
        'Line_Move_Magnitude', 'Is_Home_Team_Bet', 'Is_Favorite_Bet','Model_Sharp_Win_Prob', 'Odds_Price', 'Implied_Prob','First_Odds', 'First_Imp_Prob','Odds_Shift','Implied_Prob_Shift',
        'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds'  # ✅ ADD THESE
    ]
    
    
    # === Final output
    df_scores_out = ensure_columns(df, score_cols)[score_cols].copy()
  
    
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
        'CFL': 'CFL'
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

