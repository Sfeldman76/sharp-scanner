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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

def compute_line_hash(row):
    try:
        key = "|".join([
            str(row.get('Game_Key', '')),
            str(row.get('Bookmaker', '')),
            str(row.get('Market', '')),
            str(row.get('Outcome', '')),
            str(row.get('Value', '')),
            str(row.get('Limit', '')),
            str(row.get('Sharp_Move_Signal', ''))
        ])
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

    df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()
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
        'Scored_By_Model', 'Scoring_Market'
    ]
    # 🧩 Add schema-consistent consensus fields from summarize_consensus()
    
    # Ensure all required columns exist
    df = ensure_columns(df, ALLOWED_COLS, fill_value=None)

    # Log any remaining mismatches
    missing_cols = [col for col in ALLOWED_COLS if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ Missing required columns for BigQuery upload: {missing_cols}")
        return

    # Filter to allowed schema
    df = df[ALLOWED_COLS]

    
    logging.info("🧪 Preview of model columns being written:")
    logging.info(df[model_cols].dropna(how='all').head(5).to_string())

    # Write to BigQuery
    try:
        logging.info(f"📤 Uploading to `{table}`...")
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='append')
        logging.info(f"✅ Wrote {len(df)} new rows to `{table}`")
    except Exception as e:
        logging.exception(f"❌ Upload to `{table}` failed.")
        logging.debug("Schema:\n" + df.dtypes.to_string())
        logging.debug("Preview:\n" + df.head(5).to_string())
        
def read_market_weights_from_bigquery():
    try:
        query = f"SELECT * FROM `{MARKET_WEIGHTS_TABLE}`"
        df = bq_client.query(query).to_dataframe()
        weights = {row["Feature"]: row["Weight"] for _, row in df.iterrows()}
        print(f"✅ Loaded {len(weights)} market weights from BigQuery")
        return weights
    except Exception as e:
        print(f"❌ Failed to load market weights from BigQuery: {e}")
        return {}

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

    def implied_prob(odds):
        try:
            odds = float(odds)
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        except:
            return None

def compute_sharp_metrics(entries, open_val, mtype, label):

    move_signal = limit_jump = prob_shift = time_score = 0
    move_magnitude_score = 0.0
    for limit, curr, _ in entries:
        if open_val is not None and curr is not None:
            try:
                sharp_move_delta = abs(curr - open_val)
                if sharp_move_delta >= 0.01:
                    move_signal += 1
                    move_magnitude_score += sharp_move_delta
                if mtype == 'totals':
                    if 'under' in label and curr < open_val: move_signal += 1
                    elif 'over' in label and curr > open_val: move_signal += 1
                elif mtype == 'spreads' and abs(curr) > abs(open_val): move_signal += 1
                elif mtype == 'h2h':
                    imp_now, imp_open = implied_prob(curr), implied_prob(open_val)
                    
            except:
                continue

        if limit is not None and limit >= 100:
            limit_jump += 1

        try:
            hour = datetime.now().hour
            time_score += 1.0 if 6 <= hour <= 11 else 0.5 if hour <= 15 else 0.2
        except:
            time_score += 0.5

    move_magnitude_score = min(move_magnitude_score, 5.0)
    total_limit = sum([l or 0 for l, _, _ in entries])

    return {
        'Sharp_Move_Signal': move_signal,
        'Sharp_Limit_Jump': limit_jump,
        
        'Sharp_Time_Score': time_score,
        'Sharp_Limit_Total': total_limit,
        'Sharp_Move_Magnitude_Score': round(move_magnitude_score, 2),
        'SharpBetScore': round(
            2 * move_signal +
            2 * limit_jump +
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

def apply_blended_sharp_score(df, trained_models):
    logger.info("🛠️ Running `apply_blended_sharp_score()`")

    df = df.copy()
    df['Market'] = df['Market'].astype(str).str.lower().str.strip()

    try:
        df = df.drop(columns=[col for col in df.columns if col.endswith(('_x', '_y'))], errors='ignore')
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return pd.DataFrame()

    total_start = time.time()
    scored_all = []
    if 'Snapshot_Timestamp' not in df.columns:
        df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()
        logger.info("✅ 'Snapshot_Timestamp' column added.")

   
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
            
            

            
            # === Ensure required features exist ===
            model_features = model.get_booster().feature_names
            missing_cols = [col for col in model_features if col not in df_canon.columns]
            df_canon[missing_cols] = 0
            
            # === Align features exactly to model input ===
            X_canon = df_canon[model_features].replace({'True': 1, 'False': 0}).apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            
            # === Raw model output (optional)
            df_canon['Model_Sharp_Win_Prob'] = trained_models[market_type]['model'].predict_proba(X_canon)[:, 1]
            
            df_canon['Model_Confidence'] = trained_models[market_type]['calibrator'].predict_proba(X_canon)[:, 1]
            
            # === Tag for downstream usage
            df_canon['Was_Canonical'] = True
            df_canon['Scoring_Market'] = market_type
            df_canon['Scored_By_Model'] = True
            

            df_inverse = df_canon.copy(deep=True)
            df_inverse['Model_Sharp_Win_Prob'] = 1 - df_inverse['Model_Sharp_Win_Prob']
            df_inverse['Model_Confidence'] = 1 - df_inverse['Model_Confidence']
            df_inverse['Was_Canonical'] = False
            df_inverse['Scored_By_Model'] = True

            if market_type == "totals":
                df_inverse = df_inverse[df_inverse['Outcome'] == 'over'].copy()
                df_inverse['Outcome'] = 'under'
                df_inverse['Outcome_Norm'] = 'under'
            
                logger.info(f"🔁 Attempting to create {len(df_inverse)} inverse UNDER rows from OVER rows.")
            
                # Optional: log books missing after inverse
                original_books = set(df_market['Bookmaker'].unique())
                inverse_books = set(df_inverse['Bookmaker'].unique())
                missing_books = original_books - inverse_books
                if missing_books:
                    logger.warning(f"⚠️ These books had 'under' rows but no inverse created: {missing_books}")
            
                # ✅ Merge opponent Value (as usual)
                df_inverse['Team_Key'] = df_inverse['Game_Key_Base'] + "_" + df_inverse['Outcome']
                df_full_market['Team_Key'] = df_full_market['Game_Key_Base'] + "_" + df_full_market['Outcome']
            
                df_inverse = df_inverse.merge(
                    df_full_market[['Team_Key', 'Value']],
                    on='Team_Key',
                    how='left',
                    suffixes=('', '_opponent')
                )
            
                df_inverse['Value'] = df_inverse['Value_opponent']
                df_inverse.drop(columns=['Value_opponent'], inplace=True, errors='ignore')
            
                # ✅ Now check for merge failures
                missing_value_count = df_inverse['Value'].isna().sum()
                total_inverse = len(df_inverse)
                logger.warning(f"⚠️ {missing_value_count}/{total_inverse} totals inverse rows failed to match 'under' Value (likely missing from df_full_market)")

            elif market_type == "h2h":
                # Flip outcome to opposing team
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
                df_inverse = df_inverse.merge(
                    df_full_market[['Team_Key', 'Value']],
                    on='Team_Key',
                    how='left',
                    suffixes=('', '_opponent')
                )
                df_inverse['Value'] = df_inverse['Value_opponent']
                df_inverse.drop(columns=['Value_opponent'], inplace=True, errors='ignore')
            
                # Final deduplication
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
                df_inverse = df_inverse.merge(
                    df_full_market[['Team_Key', 'Value']],
                    on='Team_Key',
                    how='left',
                    suffixes=('', '_opponent')
                )
                df_inverse['Value'] = df_inverse['Value_opponent']
                df_inverse.drop(columns=['Value_opponent'], inplace=True, errors='ignore')
            
                # Final deduplication
                df_inverse = df_inverse.drop_duplicates(subset=['Game_Key', 'Market', 'Bookmaker', 'Outcome'])

            if df_inverse.empty:
                st.warning("⚠️ No inverse rows generated — check canonical filtering or flip logic.")
                continue  # optional: skip this scoring loop if inverse fails

           
            # ✅ Combine canonical and inverse into one scored DataFrame
            df_scored = pd.concat([df_canon, df_inverse], ignore_index=True)
            
            
            

            df_scored['Model_Confidence_Tier'] = pd.cut(
                df_scored['Model_Sharp_Win_Prob'],
                bins=[0.0, 0.4, 0.5, 0.6, 1.0],
                labels=["⚠️ Weak Indication", "✅ Coinflip", "⭐ Lean", "🔥 Strong Indication"]
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


        

def detect_sharp_moves(current, previous, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, weights={}):

    if not current:
        logger.warning("⚠️ No current odds data provided.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
  

    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}

    rows = []
    sharp_limit_map = defaultdict(lambda: defaultdict(list))
    sharp_total_limit_map = defaultdict(int)
    sharp_lines = {}
    line_history_log = {}
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
        prev_game = previous_map.get(gid, {})

        for book in game.get('bookmakers', []):
            book_key_raw = book.get('key', '').lower()
            book_key = normalize_book_key(book_key_raw, SHARP_BOOKS, REC_BOOKS)

            # 🚫 Skip if book is not allowed
            if book_key not in SHARP_BOOKS + REC_BOOKS:
                continue
            
            book_title = book.get('title', book_key)

            for market in book.get('markets', []):
                mtype = market.get('key', '').strip().lower()
                if mtype not in ['spreads', 'totals', 'h2h']:
                    continue

                for o in market.get('outcomes', []):
                    label = normalize_label(o.get('name', ''))
                    val = o.get('point') if mtype != 'h2h' else o.get('price')
                    limit = o.get('bet_limit')
                    prev_key = (game.get('home_team'), game.get('away_team'), mtype, label, book_key)
                    old_val = previous_odds_map.get(prev_key)

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
                        'Value': val,
                        'Limit': limit,
                        'Old Value': old_val,
                        'Delta': round(val - old_val, 2) if old_val is not None and val is not None else None,
                        'Home_Team_Norm': home_team,
                        'Away_Team_Norm': away_team,
                        'Commence_Hour': game_hour
                    }

                    rows.append(entry)
                    line_history_log.setdefault(gid, []).append(entry.copy())

                    if val is not None:
                        sharp_lines[(game_name, mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit, val, old_val))
                        if book_key in SHARP_BOOKS:
                            sharp_total_limit_map[(game_name, mtype, label)] += limit or 0
                        if (game_name, mtype, label) not in line_open_map:
                            line_open_map[(game_name, mtype, label)] = (val, snapshot_time)

    # 🔁 REFACTORED SHARP SCORING
    rows = apply_sharp_scoring(rows, sharp_limit_map, line_open_map, sharp_total_limit_map)


    df_sharp_lines = pd.DataFrame(sharp_lines.values())



    
       # === Build main DataFrame
    df = pd.DataFrame(rows)
    df['Book'] = df['Book'].str.lower()
    df['Event_Date'] = pd.to_datetime(df['Game_Start'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # === Historical sorting for open-line extraction
    df_history = df.copy()
    df_history_sorted = df_history.sort_values('Time')
    
    # === Extract open lines globally and per book
    line_open_df = (
        df_history_sorted.dropna(subset=['Value'])
        .groupby(['Game', 'Market', 'Outcome'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Value'})
    )
    
    line_open_per_book = (
        df_history_sorted.dropna(subset=['Value'])
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Book_Value'})
    )
    
    open_limit_df = (
        df_history_sorted
        .dropna(subset=['Limit'])
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Limit']
        .first()
        .reset_index()
        .rename(columns={'Limit': 'Opening_Limit'})
    )
    
    # === Merge open lines into df
    df = df.merge(line_open_df, on=['Game', 'Market', 'Outcome'], how='left')
    df = df.merge(line_open_per_book, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df = df.merge(open_limit_df, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df['Delta vs Sharp'] = df['Value'] - df['Open_Value']
    df['Delta'] = pd.to_numeric(df['Delta vs Sharp'], errors='coerce')
    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)
    
    # === Additional sharp flags
    df['Limit_Jump'] = (df['Limit'] >= 2500).astype(int)
    df['Sharp_Timing'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour.apply(
        lambda h: 1.0 if 6 <= h <= 11 else 0.5 if h <= 15 else 0.2
    )
    df['Limit_NonZero'] = df['Limit'].where(df['Limit'] > 0)
    df['Limit_Max'] = df.groupby(['Game', 'Market'])['Limit_NonZero'].transform('max')
    df['Limit_Min'] = df.groupby(['Game', 'Market'])['Limit_NonZero'].transform('min')

  

    
   # === Detect market leaders
    market_leader_flags = detect_market_leaders(df_history, SHARP_BOOKS, REC_BOOKS)
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
    df = detect_cross_market_sharp_support(df)
    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Unique_Sharp_Books'] = df['Unique_Sharp_Books'].fillna(0).astype(int)
    df['LimitUp_NoMove_Flag'] = df['LimitUp_NoMove_Flag'].fillna(False).astype(int)
    df['Market_Leader'] = df['Market_Leader'].fillna(False).astype(int)
    # === Recompute Pre_Game / Post_Game before saving
    df['Game_Start'] = pd.to_datetime(df['Game_Start'], errors='coerce', utc=True)
    now = pd.Timestamp.utcnow()
    df['Pre_Game'] = df['Game_Start'] > now
    df['Post_Game'] = ~df['Pre_Game']
    # === Confidence scores and tiers
    df = assign_confidence_scores(df, weights)
  

    
    # === Summary consensus metrics
    summary_df = summarize_consensus(df, SHARP_BOOKS, REC_BOOKS)
   
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
        
        'Sharp_Limit_Total': 0.001
    }

    for comp, importance in component_importance.items():
        val = row.get(comp)
        if val is None:
            continue

        try:
            val_key = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val).lower()
            weight = market_weights.get(market, {}).get(comp, {}).get(val_key, 0.5)
        except:
            weight = 0.5

        total_score += weight * importance
        max_possible += importance

    return round((total_score / max_possible) * 100 if max_possible else 50, 2)

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


def write_line_history_to_bigquery(df):
    if df is None or df.empty:
        logging.warning("⚠️ No line history data to upload.")
        return

    df = df.copy()

    # Convert Time to UTC datetime
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce', utc=True)

    df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()

    # Clean merge artifacts
    df = df.rename(columns=lambda x: x.rstrip('_x'))
    df = df.drop(columns=[col for col in df.columns if col.endswith('_y')], errors='ignore')

    # Define the allowed schema columns
    LINE_HISTORY_ALLOWED_COLS = [
        "Sport", "Game_Key", "Time", "Game", "Game_Start", "Event_Date",
        "Market", "Outcome", "Bookmaker", "Book", "Value", "Limit",
        "Old Value", "Delta", "Home_Team_Norm", "Away_Team_Norm", "Commence_Hour",
        "Ref Sharp Value", "Ref Sharp Old Value", "Delta vs Sharp",
        "SHARP_SIDE_TO_BET", "Sharp_Move_Signal", "Sharp_Limit_Jump",
        "Sharp_Time_Score", "Sharp_Prob_Shift", "Sharp_Limit_Total",
        "SharpBetScore", "Snapshot_Timestamp"
    ]

    # ✅ Remove any unexpected columns before upload
    allowed = set(LINE_HISTORY_ALLOWED_COLS)
    actual = set(df.columns)
    extra = actual - allowed
    if extra:
        logging.warning(f"⚠️ Dropping unexpected columns before upload: {extra}")
    df = df[[col for col in LINE_HISTORY_ALLOWED_COLS if col in df.columns]]

    # Log preview
    logging.debug("🧪 Line history dtypes:\n" + str(df.dtypes.to_dict()))
    logging.debug("Sample rows:\n" + df.head(2).to_string())

    # Upload
    if not safe_to_gbq(df, LINE_HISTORY_TABLE):
        logging.error(f"❌ Failed to upload line history to {LINE_HISTORY_TABLE}")
    else:
        logging.info(f"✅ Uploaded {len(df)} line history rows to {LINE_HISTORY_TABLE}.")



        
def detect_market_leaders(df_history, sharp_books, rec_books):
    df_history = df_history.copy()
    df_history['Time'] = pd.to_datetime(df_history['Time'])
    df_history['Book'] = df_history['Book'].str.lower()

    # Detect open value per Book
    df_open = (
        df_history
        .sort_values('Time')
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Value'})
    )

    df_history = df_history.merge(df_open, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df_history['Has_Moved'] = (df_history['Value'] != df_history['Open_Value']) & df_history['Value'].notna()

    first_moves = (
        df_history[df_history['Has_Moved']]
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Time']
        .min()
        .reset_index()
        .rename(columns={'Time': 'First_Move_Time'})
    )

    first_moves['Book_Type'] = first_moves['Book'].map(
        lambda b: 'Sharp' if b in sharp_books else ('Rec' if b in rec_books else 'Other')
    )
    first_moves['Move_Rank'] = first_moves.groupby(
        ['Game', 'Market', 'Outcome']
    )['First_Move_Time'].rank(method='first')

    first_moves['Market_Leader'] = (
        (first_moves['Book_Type'] == 'Sharp') & (first_moves['Move_Rank'] == 1)
    )

    return first_moves

def detect_cross_market_sharp_support(df_moves, score_threshold=15):
    df = df_moves.copy()
    df['SupportKey'] = df['Game'].astype(str) + " | " + df['Outcome'].astype(str)

    df_sharp = df[df['SharpBetScore'] >= score_threshold].copy()

    # Count unique markets
    market_counts = (
        df_sharp.groupby('SupportKey')['Market']
        .nunique()
        .reset_index()
        .rename(columns={'Market': 'CrossMarketSharpSupport'})
    )

    # Count unique sharp bookmakers
    sharp_book_counts = (
        df_sharp[df_sharp['Book'].isin(SHARP_BOOKS)]
        .groupby('SupportKey')['Book']
        .nunique()
        .reset_index()
        .rename(columns={'Book': 'Unique_Sharp_Books'})
    )

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
            'Unique_Sharp_Books', 'Merge_Key_Short'  # ✅ Add this line
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
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='replace' if force_replace else 'append')
        logging.info(f"✅ Uploaded {len(df)} rows to {table}")
    except Exception as e:
        logging.exception(f"❌ Failed to upload to {table}")

  
def fetch_scores_and_backtest(sport_key, df_moves=None, days_back=3, api_key=API_KEY, trained_models=None):
    expected_label = [k for k, v in SPORTS.items() if v == sport_key]
    sport_label = expected_label[0].upper() if expected_label else "NBA"
  

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
    try:
        existing_keys = bq_client.query("""
            SELECT DISTINCT Merge_Key_Short FROM `sharp_data.game_scores_final`
        """).to_dataframe()
        existing_keys = set(existing_keys['Merge_Key_Short'].dropna())
    
        new_scores = df_scores[~df_scores['Merge_Key_Short'].isin(existing_keys)].copy()
        blocked = df_scores[df_scores['Merge_Key_Short'].isin(existing_keys)]
    
        logging.info(f"⛔ Skipped (already in table): {len(blocked)}")
        logging.info(f"🧪 Sample skipped keys:\n{blocked[['Merge_Key_Short', 'Home_Team', 'Away_Team', 'Game_Start']].head().to_string(index=False)}")
    
        if not new_scores.empty:
            logging.info(f"✅ Uploading {len(new_scores)} games to BigQuery")
            logging.info(f"🆕 Sample uploaded:\n{new_scores[['Merge_Key_Short', 'Home_Team', 'Away_Team', 'Game_Start']].head().to_string(index=False)}")
    
        # Detect keys that are neither new nor in existing table (missing entirely?)
        all_found_keys = set(new_scores['Merge_Key_Short']) | existing_keys
        missing_keys = set(df_scores['Merge_Key_Short']) - all_found_keys
        if missing_keys:
            logging.warning(f"⚠️ These keys were neither uploaded nor matched in BigQuery:")
            logging.warning(list(missing_keys)[:5])
    
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
        # Convert string columns to categorical for memory efficiency
        for col in ['Game_Key', 'Market', 'Outcome', 'Bookmaker']:
            df_chunk.loc[:, col] = df_chunk[col].astype('category')  # Use .loc for correct assignment
    
        # Normalize the string columns (strip whitespace and lowercase)
        df_chunk.loc[:, 'Game_Key'] = df_chunk['Game_Key'].str.strip().str.lower()  # Use .loc
        df_chunk.loc[:, 'Market'] = df_chunk['Market'].str.strip().str.lower()      # Use .loc
        df_chunk.loc[:, 'Outcome'] = df_chunk['Outcome'].str.strip().str.lower()    # Use .loc
        df_chunk.loc[:, 'Bookmaker'] = df_chunk['Bookmaker'].str.strip().str.lower()  # Use .loc
    
        # Deduplicate based on necessary columns
        df_chunk = df_chunk.drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='first')
    
        # Free memory after processing chunk
        gc.collect()
    
        return df_chunk



    # === Vectorized version of calc_cover ===
    def calc_cover(df):
        # Calculate total score (only needed for 'totals' market)
        df['Total_Score'] = df['Score_Home_Score'] + df['Score_Away_Score']
        
        # Create a mask for 'totals' market
        df['Is_Totals'] = df['Market'] == 'totals'
    
        # Vectorized computation for 'totals' market
        df['Tot_Outcome'] = np.where(df['Outcome'] == 'under', -1, 1)
        
        # Vectorized computation for 'spreads' and 'h2h' markets
        df['Spread_Outcome'] = np.where(df['Outcome'] == df['Home_Team_Norm'], df['Score_Home_Score'], df['Score_Away_Score'])
        
        # Calculate the covered bet based on market type
        df['Covered'] = np.where(
            df['Is_Totals'],  # For 'totals' market
            np.where(df['Outcome'] == 'under', df['Total_Score'] < df['Value'], df['Total_Score'] > df['Value']),
            (df['Spread_Outcome'] + df['Value']) > df['Score_Away_Score']  # For other markets
        )

        
        # Assign the final results based on conditions
        df['SHARP_COVER_RESULT'] = np.where(df['Covered'], 'Win', 'Loss')
        df['SHARP_HIT_BOOL'] = np.where(df['Covered'], 1, 0)
        
        return df
    

    
 
    # === 4. Load recent sharp picks
    df_master = read_recent_sharp_moves(hours=days_back * 24)
    df_master = build_game_key(df_master)  # Ensure Merge_Key_Short is created
    
    # === Filter out games already scored in sharp_scores_full
    scored_keys = bq_client.query("""
        SELECT DISTINCT Merge_Key_Short
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
    """).to_dataframe()
    
    already_scored = set(scored_keys['Merge_Key_Short'].dropna())
    
    # Filter out already scored keys in df_scores
    df_scores_needed = df_scores[~df_scores['Merge_Key_Short'].isin(already_scored)]
    logging.info(f"✅ Remaining unscored completed games: {len(df_scores_needed)}")
    
    # Ensure Merge_Key_Short exists AFTER loading
    if 'Merge_Key_Short' not in df_master.columns:
        df_master = build_game_key(df_master)
    if 'Merge_Key_Short' not in df_scores_needed.columns:
        df_scores_needed = build_game_key(df_scores_needed)
    
    # Debugging: Log the columns of the DataFrames after build_game_key
    logging.info(f"After build_game_key - df_scores_needed columns: {df_scores_needed.columns.tolist()}")
    logging.info(f"After build_game_key - df_master columns: {df_master.columns.tolist()}")
    
    # Track memory usage before the operation
    process = psutil.Process(os.getpid())
    logging.info(f"Memory before operation: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # === Ensure df_all_snapshots is loaded and processed correctly
    df_all_snapshots = read_recent_sharp_moves(hours=days_back * 24)
    # Process df_all_snapshots in chunks to avoid memory overload
    df_all_snapshots_filtered = pd.concat([
        process_chunk(df_all_snapshots.iloc[start:start + 200])  # Reduced chunk size to 1000 for memory optimization
        for start in range(0, len(df_all_snapshots), 200)
    ], ignore_index=True)
     # Optionally log the shape of df_all_snapshots after filtering
    logging.info(f"After filtering, df_all_snapshots_filtered shape: {df_all_snapshots_filtered.shape}")
    # === Ensure 'df_first' is created correctly
    # Check if 'Model_Sharp_Win_Prob' is available before renaming it
    
    # === Ensure 'df_first' is created correctly
    # Check if 'df_first' exists, otherwise create it
    if 'df_first' not in locals():
        # === Ensure 'df_first' is created correctly
        # Check if 'Model_Sharp_Win_Prob' is available before renaming it
        snapshot_cols = ['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Value']
        if 'Model_Sharp_Win_Prob' in df_all_snapshots_filtered.columns:
            snapshot_cols.append('Model_Sharp_Win_Prob')
        
        df_first = (
            df_all_snapshots_filtered
            .sort_values('Snapshot_Timestamp')
            .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='first')
            .loc[:, snapshot_cols]
            .rename(columns={
                'Value': 'First_Line_Value',
                'Model_Sharp_Win_Prob': 'First_Sharp_Prob'
            })
        )
        
        # Ensure First_Sharp_Prob is created even if it was missing
        if 'First_Sharp_Prob' not in df_first.columns:
            df_first['First_Sharp_Prob'] = np.nan
        
        # Convert relevant columns to category type before merging to save memory
        for col in ['Game_Key', 'Market', 'Outcome', 'Bookmaker']:
            df_first[col] = df_first[col].astype('category')
    
        # Debug: validate uniqueness
        num_unique_keys = df_first[['Game_Key', 'Market', 'Outcome', 'Bookmaker']].drop_duplicates().shape[0]
        logging.info(f"🧪 df_first keys: {num_unique_keys} unique Game_Key + Market + Outcome + Bookmaker combos")
    
        # 🚨 Warn if duplicates still slipped through
        if df_first.duplicated(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker']).any():
            logging.warning("⚠️ df_first has duplicates — this will corrupt snapshot merge")
    
    
    # Function to process DataFrames in smaller batches
    def batch_merge(df_master, df_first, batch_size=200):
        num_chunks = len(df_first) // batch_size + 1
        merged_df_list = []
    
        for i in range(num_chunks):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df_first))
            df_first_batch = df_first.iloc[start_idx:end_idx]
    
            # Merge the batch with df_master
            df_batch = df_master.merge(df_first_batch, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left')
            merged_df_list.append(df_batch)
    
            # Free memory after processing each batch
            del df_first_batch
            gc.collect()
    
        # Concatenate all merged DataFrames into one
        df_master = pd.concat(merged_df_list, ignore_index=True)
        del merged_df_list
        gc.collect()
    
        return df_master
    
    
    # Function to batch merge df_scores
    def batch_merge_scores(df_master, df_scores, batch_size=200):
        num_chunks = len(df_scores) // batch_size + 1
        merged_df_list = []
    
        for i in range(num_chunks):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df_scores))
            df_scores_batch = df_scores.iloc[start_idx:end_idx]
    
            # Merge the batch with df_master
            df_batch = df_master.merge(df_scores_batch[['Merge_Key_Short', 'Score_Home_Score', 'Score_Away_Score']], on='Merge_Key_Short', how='inner')
            merged_df_list.append(df_batch)
    
            # Free memory after processing each batch
            del df_scores_batch
            gc.collect()
    
        # Concatenate all merged DataFrames into one
        df_master = pd.concat(merged_df_list, ignore_index=True)
        del merged_df_list
        gc.collect()
    
        return df_master
    
    
    # Main function to apply batch processing
    def process_in_batches(df_master, df_first, df_scores, batch_size=500):
        # Reduce df_first to only necessary columns before the merge to save memory
        df_first = df_first[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'First_Line_Value', 'First_Sharp_Prob']]
    
        # Convert relevant columns to category type before merging to save memory
        df_first['Game_Key'] = df_first['Game_Key'].astype('category')
        df_first['Market'] = df_first['Market'].astype('category')
        df_first['Outcome'] = df_first['Outcome'].astype('category')
        df_first['Bookmaker'] = df_first['Bookmaker'].astype('category')
    
        # Reduce df_scores to only necessary columns before the merge to save memory
        df_scores = df_scores[['Merge_Key_Short', 'Score_Home_Score', 'Score_Away_Score']]
    
        # Convert Merge_Key_Short to category in df_scores to save memory
       
        
        # Use .loc[] to ensure you're modifying the original DataFrame:
        df_scores.loc[:, 'Merge_Key_Short'] = df_scores['Merge_Key_Short'].astype('category')
    
        # Process df_first in smaller batches
        df_master = batch_merge(df_master, df_first, batch_size)
    
        # Process df_scores in smaller batches
        df_master = batch_merge_scores(df_master, df_scores, batch_size)
    
        # Return final processed DataFrame
        return df_master
    
    
    # === Process Data
    # Apply the batch processing function to your data
    df_master = process_in_batches(df_master, df_first, df_scores_needed, batch_size=200)
    
    # Track memory usage after the operation
    logging.info(f"Memory after operation: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # === Continue with further operations (e.g., reassigning Merge_Key_Short, computing model probabilities, etc.)
    
    # Ensure 'Merge_Key_Short' is present in df_all_snapshots
    if 'Merge_Key_Short' not in df_all_snapshots.columns:
        logging.error("❌ 'Merge_Key_Short' is missing in df_all_snapshots.")
        return pd.DataFrame()  # Return empty DataFrame if missing

   
    
    # Track memory usage after the operation
    logging.info(f"Memory after operation: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Continue with further operations (e.g., computing model probabilities, etc.)
    
    
    # === Convert df_master columns to category type to reduce memory usage before merge
    df_master['Game_Key'] = df_master['Game_Key'].astype('category')
    df_master['Market'] = df_master['Market'].astype('category')
    df_master['Outcome'] = df_master['Outcome'].astype('category')
    df_master['Bookmaker'] = df_master['Bookmaker'].astype('category')
    
    # === Merge first snapshot into master before scoring
    df_master = df_master.merge(df_first, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left')
    
    # Free memory after merge
    del df_first
    gc.collect()
    
   # === Merge scores and filter (with deduplication and memory safety)
    df_scores_needed = df_scores[['Merge_Key_Short', 'Score_Home_Score', 'Score_Away_Score']].copy()
    
    # Convert to category to reduce memory
    df_scores_needed.loc[:, 'Merge_Key_Short'] = df_scores_needed['Merge_Key_Short'].astype('category')
    
    # Deduplicate to avoid row explosion
    # Deduplicate to avoid row explosion (optional sort if timestamp is available)
    if 'Inserted_Timestamp' in df_scores_needed.columns:
        df_scores_needed = (
            df_scores_needed
            .sort_values('Inserted_Timestamp', ascending=True)
            .drop_duplicates(subset='Merge_Key_Short', keep='last')
        )
    else:
        df_scores_needed = df_scores_needed.drop_duplicates(subset='Merge_Key_Short', keep='last')
        
    # Log shape before merge
    log_memory("BEFORE merge with df_scores_needed")
    logging.info(f"df_master shape: {df_master.shape}, df_scores_needed shape: {df_scores_needed.shape}")
    
    logging.info(f"✅ Before final score merge: df_master columns: {df_master.columns.tolist()}")
    # Merge only required columns
    df = df_master.merge(
        df_scores_needed[['Merge_Key_Short', 'Score_Home_Score', 'Score_Away_Score']],
        on='Merge_Key_Short',
        how='inner'
    )
    logging.info(f"✅ After merge: df columns: {df.columns.tolist()}")
    # Log shape after merge
    log_memory("AFTER merge with df_scores_needed")
    logging.info(f"df shape after merge: {df.shape}")
    
    # Free memory
    
    # === Reassign Merge_Key_Short from df_master using Game_Key
    if 'Merge_Key_Short' in df_master.columns:
        logging.info("🧩 Reassigning Merge_Key_Short from df_master via Game_Key (optimized)")
        # Build mapping dictionary (Game_Key → Merge_Key_Short)
        key_map = df_master.drop_duplicates(subset=['Game_Key'])[['Game_Key', 'Merge_Key_Short']].set_index('Game_Key')['Merge_Key_Short'].to_dict()
        # Reassign inplace without merge
        df['Merge_Key_Short'] = df['Game_Key'].map(key_map)
    del df_master
    del df_scores_needed
    gc.collect()
    
    
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
    
        # Compute Model_Prob_Diff and Line_Delta
        df_chunk['Model_Prob_Diff'] = pd.to_numeric(df_chunk['Model_Sharp_Win_Prob'], errors='coerce') - pd.to_numeric(df_chunk['First_Sharp_Prob'], errors='coerce')
        df_chunk['Line_Delta'] = pd.to_numeric(df_chunk['Value'], errors='coerce') - pd.to_numeric(df_chunk['First_Line_Value'], errors='coerce')
    
        # Compute adjusted support direction
        market_totals = df_chunk['Market'].str.lower() == 'totals'
        outcome_under = df_chunk['Outcome'].str.lower() == 'under'
        first_line_negative = df_chunk['First_Line_Value'] < 0
    
        df_chunk['Line_Support_Sign'] = np.where(market_totals & outcome_under, -1, 1)
        df_chunk['Line_Support_Sign'] = np.where(~market_totals & first_line_negative, -1, df_chunk['Line_Support_Sign'])
    
        # Compute Adjusted_Line_Delta
        df_chunk['Adjusted_Line_Delta'] = df_chunk['Line_Delta'] * df_chunk['Line_Support_Sign']
    
        # Assign Direction_Aligned
        mask_aligned = (df_chunk['Model_Prob_Diff'] > 0) & (df_chunk['Adjusted_Line_Delta'] > 0) | \
                       (df_chunk['Model_Prob_Diff'] < 0) & (df_chunk['Adjusted_Line_Delta'] < 0)
        mask_conflict = (df_chunk['Model_Prob_Diff'] > 0) & (df_chunk['Adjusted_Line_Delta'] < 0) | \
                        (df_chunk['Model_Prob_Diff'] < 0) & (df_chunk['Adjusted_Line_Delta'] > 0)
    
        df_chunk['Direction_Aligned'] = np.where(mask_aligned, 1, np.where(mask_conflict, 0, pd.NA))
        df_chunk['Direction_Aligned'] = df_chunk['Direction_Aligned'].astype('Int64')
    
        # Clean up temp columns
        df_chunk.drop(columns=['Line_Support_Sign', 'Adjusted_Line_Delta'], inplace=True, errors='ignore')
    
        return df_chunk
    
    
    def process_in_chunks(df, chunk_size=200):
        chunks = []
        for start in range(0, len(df), chunk_size):
            df_chunk = df.iloc[start:start + chunk_size]
            processed_chunk = process_chunk_logic(df_chunk)
            chunks.append(processed_chunk)
            del df_chunk, processed_chunk
            gc.collect()
        return pd.concat(chunks, ignore_index=True)
    
    
    # === ✅ Apply the chunk processing
    df = process_in_chunks(df, chunk_size=200)
    
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
    
    # Calculate result using `calc_cover` function
    result = df_valid.apply(calc_cover, axis=1, result_type='expand')
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
    
    # === Final Output DataFrame ===
    score_cols = [
        'Game_Key', 'Bookmaker', 'Market', 'Outcome', 'Ref_Sharp_Value',
        'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
        'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
        'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
        'Unique_Sharp_Books', 'Enhanced_Sharp_Confidence_Score',
        'True_Sharp_Confidence_Score', 'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT',
        'Scored', 'Sport', 'Value','Merge_Key_Short',
        'First_Line_Value', 'First_Sharp_Prob',         # ✅ new
        'Line_Delta', 'Model_Prob_Diff', 'Direction_Aligned',  'Home_Team_Norm',
        'Away_Team_Norm',
        'Commence_Hour',   # ✅ new
    ]
    
    # === Final output
    df_scores_out = ensure_columns(df, score_cols)[score_cols].copy()
    logging.info(f"✅ Uploaded {len(df_scores_out)} new scored picks to sharp_scores_full")
    
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
    
    df_scores_out['Snapshot_Timestamp'] = pd.Timestamp.utcnow()  # ✅ Only do this once here
    
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
        'Game_Key', 'Bookmaker', 'Market', 'Outcome',
        'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
        'Sharp_Time_Score', 'Sharp_Limit_Total', 'Value',
        'First_Line_Value', 'First_Sharp_Prob', 'Line_Delta',
        'Model_Prob_Diff', 'Direction_Aligned'
    ]
    
    logging.info(f"🧪 Fingerprint dedup keys: {dedup_fingerprint_cols}")
    float_cols_to_round = [
        'Sharp_Prob_Shift', 'Sharp_Time_Score', 'Sharp_Limit_Total', 'Value',
        'First_Line_Value', 'First_Sharp_Prob', 'Line_Delta', 'Model_Prob_Diff'
    ]
    
    for col in float_cols_to_round:
        if col in df_scores_out.columns:
            df_scores_out[col] = pd.to_numeric(df_scores_out[col], errors='coerce').round(4)
    df_scores_out = df_scores_out.drop_duplicates(subset=dedup_fingerprint_cols)
    logging.info(f"🧪 Local rows before dedup: {len(df_scores_out)}")
    
    # Query BigQuery for existing rows
    existing = bq_client.query(f"""
        SELECT DISTINCT {', '.join(dedup_fingerprint_cols)}
        FROM `sharp_data.sharp_scores_full`
        WHERE DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    """).to_dataframe()
    logging.info(f"🧪 Existing rows in BigQuery for dedup: {len(existing)}")
    
    # Dedup against BigQuery
    df_scores_out = df_scores_out.merge(
        existing,
        on=dedup_fingerprint_cols,
        how='left',
        indicator=True
    )
    df_scores_out = df_scores_out[df_scores_out['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    # Final logs before upload
    logging.info(f"🧪 Remaining new rows after dedup merge: {len(df_scores_out)}")
    
    # If no new rows left, return
    if df_scores_out.empty:
        logging.info("ℹ️ No new scores to upload — all identical line states already in BigQuery.")
        return pd.DataFrame()
    
    # Convert to Parquet format
    try:
        pa.Table.from_pandas(df_scores_out)
    except Exception as e:
        logging.error("❌ Parquet conversion failure before upload:")
        logging.error(str(e))
        for col in df_scores_out.columns:
            logging.info(f"🔍 {col} → {df_scores_out[col].dtype}, sample: {df_scores_out[col].dropna().unique()[:5].tolist()}")
    
    logging.info(f"🧪 df_scores_out ready for return: {len(df_scores_out)} rows")
    logging.info(f"🧪 Final sample:\n{df_scores_out[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Snapshot_Timestamp']].head(3).to_string(index=False)}")
    
    return df_scores_out



def compute_and_write_market_weights(df):
    import pandas as pd

    component_cols = [
        'Sharp_Move_Signal',
        'Sharp_Limit_Jump',
        'Sharp_Prob_Shift',
        'Sharp_Time_Score',
        'Sharp_Limit_Total'
    ]

    rows = []

    for comp in component_cols:
        df_temp = df[['Market', comp, 'SHARP_HIT_BOOL']].copy()
        df_temp = df_temp.rename(columns={comp: 'Value'})
        df_temp['Component'] = comp
        df_temp['Value'] = df_temp['Value'].astype(str).str.lower()
        df_temp = df_temp.dropna()

        # Group by Market, Component, Value
        grouped = df_temp.groupby(['Market', 'Component', 'Value'])['SHARP_HIT_BOOL'].mean().reset_index()
        grouped.rename(columns={'SHARP_HIT_BOOL': 'Win_Rate'}, inplace=True)

        for _, row in grouped.iterrows():
            rows.append({
                'Market': str(row['Market']).lower(),
                'Component': row['Component'],
                'Value': row['Value'],
                'Win_Rate': round(row['Win_Rate'], 4)
            })

    if not rows:
        print("⚠️ No valid market weights to upload.")
        return

    df_weights = pd.DataFrame(rows)
    print(f"✅ Prepared {len(df_weights)} market weight rows. Sample:")
    print(df_weights.head(10).to_string(index=False))

    # Optional: return df_weights for upload or inspection
    return df_weights

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

