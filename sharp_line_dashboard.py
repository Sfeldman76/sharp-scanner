import streamlit as st
import time
from streamlit_autorefresh import st_autorefresh


# === Page Config ===
st.set_page_config(layout="wide")
st.title("Betting Line Scanner")
st.markdown("""
<style>
.scrollable-dataframe-container {
    max-height: 600px;
    overflow-y: auto;
    overflow-x: auto;
    border: 1px solid #444;
    padding: 0.5rem;
    margin-bottom: 1rem;
}
div[data-testid="stDataFrame"] > div {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# === Auto-refresh every 380 seconds ===
st_autorefresh(interval=380 * 1000, key="data_refresh")


st.markdown("""
<style>
.scrollable-table-container {
    max-height: 600px;
    overflow-y: auto;
    border: 1px solid #444;
    margin-bottom: 1rem;
}
.custom-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 14px;
}
.custom-table th, .custom-table td {
    border: 1px solid #444;
    padding: 8px;
    text-align: center;
}
.custom-table th {
    background-color: #1f2937;
    color: white;
}
.custom-table tr:nth-child(even) {
    background-color: #2d3748;
}
.custom-table tr:hover {
    background-color: #4b5563;
}
</style>
""", unsafe_allow_html=True)




# === Standard Imports ===
import os
import json
import pickle
import pytz
import time
import requests
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pytz import timezone as pytz_timezone
from google.oauth2 import service_account
import pandas_gbq
import pandas as pd
import pandas_gbq  # ✅ Required for setting .context.project / .context.credentials
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import bigquery
from pandas_gbq import to_gbq
import pandas as pd
import google.api_core.exceptions
from google.cloud import bigquery_storage_v1
import pyarrow as pa
import pyarrow.parquet as pq
#from detect_utils import detect_and_save_all_sports
import numpy as np


GCP_PROJECT_ID = "sharplogger"  # ✅ confirmed project ID
BQ_DATASET = "sharp_data"       # ✅ your dataset name
BQ_TABLE = "sharp_moves_master" # ✅ your table name
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
MARKET_WEIGHTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.market_weights"
LINE_HISTORY_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.line_history_master"
SNAPSHOTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.odds_snapshot_log"
GCS_BUCKET = "sharp-models"
import os, json

pandas_gbq.context.project = GCP_PROJECT_ID  # credentials will be inferred

bq_client = bigquery.Client(project=GCP_PROJECT_ID)  # uses env var
gcs_client = storage.Client(project=GCP_PROJECT_ID)



# === Constants and Config ===
API_KEY = "3879659fe861d68dfa2866c211294684"
#FOLDER_ID = "1v6WB0jRX_yJT2JSdXRvQOLQNfOZ97iGA"
REDIRECT_URI = "https://sharp-scanner-723770381669.us-east4.run.app/"  # no longer used for login, just metadata

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
    'fanatics', 'espnbet', 'hardrockbet','bovada','betus' ]

BOOKMAKER_REGIONS = {
    # 🔹 Sharp Books
    'pinnacle': 'eu',
    'betfair_ex_eu': 'eu',
    'betfair_ex_uk': 'uk',
    'smarkets': 'uk',
    'matchbook': 'uk',
    'betonlineag': 'us',
    'lowvig': 'us',
    'betanysports': 'us2',
    'betus': 'us',

    # 🔸 Rec Books
    'betmgm': 'us',
    'draftkings': 'us',
    'fanduel': 'us',
    'betrivers': 'us',
    'espnbet': 'us2',
    'hardrockbet': 'us2',
    'fanatics': 'us',
    'mybookieag': 'us',
    'bovada': 'us',
    'rebet': 'us2',
    'windcreek': 'us2',

    # Optional extras (if needed later)
    'bet365': 'uk',
    'williamhill': 'uk',
    'ladbrokes': 'uk',
    'unibet': 'eu',
    'bwin': 'eu',
    'sportsbet': 'au',
    'ladbrokesau': 'au',
    'neds': 'au'
}


MARKETS = ['spreads', 'totals', 'h2h']



# === Component fields used in sharp scoring ===
component_fields = OrderedDict({
    'Sharp_Move_Signal': 'Win Rate by Move Signal',
    'Sharp_Time_Score': 'Win Rate by Time Score',
    'Sharp_Limit_Jump': 'Win Rate by Limit Jump',
    'Sharp_Prob_Shift': 'Win Rate by Prob Shift',
    'Is_Reinforced_MultiMarket': 'Win Rate by Cross-Market Reinforcement',
    'Market_Leader': 'Win Rate by Market Leader',
    'LimitUp_NoMove_Flag': 'Win Rate by Limit↑ No Move'
})



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


@st.cache_data(ttl=380)
def fetch_live_odds(sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us,us2,uk,eu,au',
        'markets': ','.join(MARKETS),
        'oddsFormat': 'american',
        'includeBetLimits': 'true'
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Odds API Error: {e}")

        return []

def safe_to_gbq(df, table, replace=False):
    mode = 'replace' if replace else 'append'
    for attempt in range(3):
        try:
            to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists=mode)
            return True
        except google.api_core.exceptions.BadRequest as e:
            print(f"❌ BadRequest during BigQuery write: {e}")
            if "Cannot add fields" in str(e):
                print("⚠️ Retrying with schema replace...")
                to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='replace')
                return True
            else:
                return False
        except Exception as e:
            print(f"❌ Retry {attempt + 1}/3 failed: {e}")
    return False




def write_snapshot_to_gcs_parquet(snapshot_list, bucket_name="sharp-models", folder="snapshots/"):
    rows = []
    snapshot_time = pd.Timestamp.utcnow()

    for game in snapshot_list:
        gid = game.get('id')
        if not gid:
            continue
        for book in game.get('bookmakers', []):
            book_key = book.get('key')
            for market in book.get('markets', []):
                market_key = market.get('key')
                for outcome in market.get('outcomes', []):
                    rows.append({
                        'Game_ID': gid,
                        'Bookmaker': book_key,
                        'Market': market_key,
                        'Outcome': outcome.get('name'),
                        'Value': outcome.get('point') if market_key != 'h2h' else outcome.get('price'),
                        'Limit': outcome.get('bet_limit'),
                        'Snapshot_Timestamp': snapshot_time
                    })

    df_snap = pd.DataFrame(rows)
    # Build Game_Key in df_snap using the same function as df_moves_raw
    df_snap = build_game_key(df_snap)
    if df_snap.empty:
        print("⚠️ No snapshot data to upload.")
        return

    filename = f"{folder}{snapshot_time.strftime('%Y%m%d_%H%M%S')}_snapshot.parquet"
    table = pa.Table.from_pandas(df_snap)
    buffer = BytesIO()
    pq.write_table(table, buffer, compression='snappy')

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_string(buffer.getvalue(), content_type='application/octet-stream')
        print(f"✅ Snapshot uploaded to GCS as Parquet: gs://{bucket_name}/{filename}")
    except Exception as e:
        print(f"❌ Failed to upload snapshot to GCS: {e}")

def write_sharp_moves_to_master(df, table='sharp_data.sharp_moves_master'):
    if df is None or df.empty:
        st.warning("⚠️ No sharp moves to write.")
        return

    df = df.copy()

    # Safety: Ensure Game_Key exists
    if 'Game_Key' not in df.columns or df['Game_Key'].isnull().all():
        st.warning("❌ No valid Game_Key present — skipping upload.")
        st.dataframe(df[['Game', 'Game_Start', 'Market', 'Outcome']].head())
        return

    df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()
    st.info(f"🧪 Sharp moves ready to write: {len(df)}")

    # Clean column names
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    df = df.drop(columns=[col for col in df.columns if col.endswith('_x') or col.endswith('_y')], errors='ignore')

    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce', utc=True)

    # Convert object columns safely
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].where(df[col].notna(), None)

    try:
        st.info(f"📤 Uploading to `{table}`...")
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='append')
        st.success(f"✅ Wrote {len(df)} new rows to `{table}`")
    except Exception as e:
        st.error(f"❌ Upload to `{table}` failed: {e}")
        st.code(df.dtypes.to_string(), language="python")
        st.dataframe(df.head(5))





def write_to_bigquery(df, table='sharp_data.sharp_scores_full', force_replace=False):
    from pandas_gbq import to_gbq

    if df.empty:
        st.info("ℹ️ No data to write to BigQuery.")
        return

    df = df.copy()
    df.columns = [col.replace(" ", "_") for col in df.columns]

    # Drop unapproved fields (BigQuery strict schema match)
    allowed_cols = {
        'sharp_data.sharp_scores_full': [
            'Game_Key', 'Bookmaker', 'Market', 'Outcome', 'Ref_Sharp_Value',
            'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
            'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
            'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
            'Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score',
            'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT', 'Scored', 'Snapshot_Timestamp'
        ],
        'sharp_data.sharp_moves_master': None  # Add allowed list here if needed
    }
    if table in allowed_cols and allowed_cols[table] is not None:
        df = df[[col for col in df.columns if col in allowed_cols[table]]]

    try:
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='replace' if force_replace else 'append')
        
        #st.success(f"✅ Uploaded {len(df)} rows to {table}")
    except Exception as e:
        st.error(f"❌ Failed to upload to {table}: {e}")

    

        
def build_game_key(df):
    required = ['Game', 'Game_Start', 'Market', 'Outcome']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"⚠️ Skipping build_game_key — missing columns: {missing}")
        return df

    df = df.copy()
    df['Home_Team_Norm'] = df['Game'].str.extract(r'^(.*?) vs')[0].str.strip().str.lower()
    df['Away_Team_Norm'] = df['Game'].str.extract(r'vs (.*)$')[0].str.strip().str.lower()
    df['Commence_Hour'] = pd.to_datetime(df['Game_Start'], errors='coerce', utc=True).dt.floor('h')
    df['Market_Norm'] = df['Market'].str.strip().str.lower()
    df['Outcome_Norm'] = df['Outcome'].str.strip().str.lower()

    df['Game_Key'] = (
        df['Home_Team_Norm'] + "_" +
        df['Away_Team_Norm'] + "_" +
        df['Commence_Hour'].astype(str) + "_" +
        df['Market_Norm'] + "_" +
        df['Outcome_Norm']
    )

    df['Merge_Key_Short'] = df.apply(
        lambda row: build_merge_key(row['Home_Team_Norm'], row['Away_Team_Norm'], row['Commence_Hour']),
        axis=1
    )

    return df
    
def normalize_team(t):
    return str(t).strip().lower().replace('.', '').replace('&', 'and')

def build_merge_key(home, away, game_start):
    return f"{normalize_team(home)}_{normalize_team(away)}_{game_start.floor('h').strftime('%Y-%m-%d %H:%M:%S')}"


@st.cache_data(ttl=380)
def read_recent_sharp_moves(hours=500, table=BQ_FULL_TABLE):
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


       
def read_latest_snapshot_from_bigquery(hours=2):
    try:
        client = bq_client
        query = f"""
            SELECT * FROM `{SNAPSHOTS_TABLE}`
            WHERE Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
        """
        df = client.query(query).to_dataframe()
        # Group back into the same format as before if needed
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
        print(f"✅ Reconstructed {len(grouped)} snapshot games from BigQuery")
        return dict(grouped)
    except Exception as e:
        print(f"❌ Failed to load snapshot from BigQuery: {e}")
        return {}

def write_market_weights_to_bigquery(weights_dict):
    rows = []

    for market, components in weights_dict.items():
        for component, values in components.items():
            for val_key, win_rate in values.items():
                try:
                    # === Debug: Log raw input
                    print(f"🧪 Market={market}, Component={component}, Value={val_key}, Raw WinRate={win_rate}")
                    
                    # === Flatten if nested dict
                    if isinstance(win_rate, dict) and 'value' in win_rate:
                        win_rate = win_rate['value']
                    if isinstance(win_rate, dict):
                        raise ValueError("Nested dict still present")

                    # === Add row
                    rows.append({
                        'Market': market,
                        'Component': component,
                        'Value': str(val_key).lower(),
                        'Win_Rate': float(win_rate)
                    })
                except Exception as e:
                    print(f"⚠️ Skipped invalid win_rate for {market}/{component}/{val_key}: {e}")

    if not rows:
        print("⚠️ No valid market weights to upload.")
        return

    df = pd.DataFrame(rows)
    print(f"✅ Prepared {len(df)} rows for upload. Preview:")
    print(df.head(5).to_string(index=False))

    # === Upload to BigQuery
    try:
        to_gbq(df, MARKET_WEIGHTS_TABLE, project_id=GCP_PROJECT_ID, if_exists='replace')
        print(f"✅ Uploaded to {MARKET_WEIGHTS_TABLE}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(df.dtypes)
        
        
def write_line_history_to_bigquery(df):
    if df is None or df.empty:
        print("⚠️ No line history data to upload.")
        return

    df = df.copy()

    # ✅ Force conversion of 'Time' to datetime
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce', utc=True)

    df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()

    # ✅ Clean merge artifacts
    df = df.rename(columns=lambda x: x.rstrip('_x'))
    df = df.drop(columns=[col for col in df.columns if col.endswith('_y')], errors='ignore')

    print("🧪 Line history dtypes:\n", df.dtypes.to_dict())
    print(df.head(2))

    if not safe_to_gbq(df, LINE_HISTORY_TABLE):
        print(f"❌ Failed to upload line history to {LINE_HISTORY_TABLE}")
    else:
        print(f"✅ Uploaded {len(df)} line history rows to {LINE_HISTORY_TABLE}.")

def initialize_all_tables(df_snap, df_audit, market_weights_dict):
    from google.cloud import bigquery

    def table_needs_replacement(table_name):
        try:
            query = f"SELECT * FROM `{table_name}` LIMIT 1"
            _ = bq_client.query(query).to_dataframe()
            return False  # Table exists and has schema
        except Exception as e:
            print(f"⚠️ Table {table_name} likely missing or misconfigured: {e}")
            return True

    # === 1. Initialize line_history_master
    if table_needs_replacement(LINE_HISTORY_TABLE):
        if df_audit is not None and not df_audit.empty:
            df = df_audit.copy()
            df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'], errors='coerce', utc=True)
            df = df.rename(columns=lambda x: x.rstrip('_x'))
            df = df.drop(columns=[col for col in df.columns if col.endswith('_y')], errors='ignore')
            to_gbq(df, LINE_HISTORY_TABLE, project_id=GCP_PROJECT_ID, if_exists='replace')
            print(f"✅ Initialized {LINE_HISTORY_TABLE} with {len(df)} rows")
        else:
            print(f"⚠️ Skipping {LINE_HISTORY_TABLE} initialization — df_audit is empty")

    # === 2. Initialize odds_snapshot_log
    if table_needs_replacement(SNAPSHOTS_TABLE):
        if df_snap is not None and not df_snap.empty:
            df = df_snap.copy()
            df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'], errors='coerce', utc=True)
            df = df.rename(columns=lambda x: x.rstrip('_x'))
            df = df.drop(columns=[col for col in df.columns if col.endswith('_y')], errors='ignore')
            to_gbq(df, SNAPSHOTS_TABLE, project_id=GCP_PROJECT_ID, if_exists='replace')
            print(f"✅ Initialized {SNAPSHOTS_TABLE} with {len(df)} rows")
        else:
            print(f"⚠️ Skipping {SNAPSHOTS_TABLE} initialization — df_snap is empty")

    # === 3. Initialize market_weights
    if table_needs_replacement(MARKET_WEIGHTS_TABLE):
        rows = []
        for market, components in market_weights_dict.items():
            for component, values in components.items():
                for val_key, win_rate in values.items():
                    rows.append({
                        'Market': market,
                        'Component': component,
                        'Value': val_key,
                        'Win_Rate': float(win_rate)
                    })
        df = pd.DataFrame(rows)
        if not df.empty:
            to_gbq(df, MARKET_WEIGHTS_TABLE, project_id=GCP_PROJECT_ID, if_exists='replace')
            print(f"✅ Initialized {MARKET_WEIGHTS_TABLE} with {len(df)} rows")
        else:
            print(f"⚠️ Skipping {MARKET_WEIGHTS_TABLE} initialization — no weight rows available")


            
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



def read_market_weights_from_bigquery():
    try:
        client = bq_client
        query = f"SELECT * FROM `{MARKET_WEIGHTS_TABLE}`"
        df = client.query(query).to_dataframe()
        weights = defaultdict(lambda: defaultdict(dict))
        for _, row in df.iterrows():
            market = row['Market']
            component = row['Component']
            value = str(row['Value']).lower()
            win_rate = float(row['Win_Rate'])
            weights[market][component][value] = win_rate
        print(f"✅ Loaded {len(df)} market weight rows from BigQuery.")
        return dict(weights)
    except Exception as e:
        print(f"❌ Failed to load market weights from BigQuery: {e}")
        return {}



def detect_cross_market_sharp_support(df_moves, score_threshold=25):
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
    
    
@st.cache_data(ttl=380)
def detect_sharp_moves(current, previous, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, weights={}):
    from collections import defaultdict
    import pandas as pd
    from datetime import datetime

    def normalize_label(label):
        return str(label).strip().lower().replace('.0', '')

    if not current:
        print("⚠️ No current odds data provided.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}
    
    rows = []
    sharp_limit_map = defaultdict(lambda: defaultdict(list))
    sharp_total_limit_map = defaultdict(int)
    sharp_lines = {}
    line_history_log = {}
    line_open_map = {}
    sharp_side_flags = {}
    sharp_metrics_map = {}

    sport_scope_key = {
        'MLB': 'baseball_mlb',
        'NBA': 'basketball_nba'
    }.get(sport_key.upper(), sport_key.lower())
    confidence_weights = weights.get(sport_scope_key, {})

    # === Safely flatten previous odds
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

    # === Parse current odds
    for game in current:
        home_team = game.get('home_team', '').strip().lower()
        away_team = game.get('away_team', '').strip().lower()
        if not home_team or not away_team:
            continue

        game_name = f"{home_team.title()} vs {away_team.title()}"
        event_time = pd.to_datetime(game.get("commence_time"), utc=True, errors='coerce')
        event_date = event_time.strftime("%Y-%m-%d") if pd.notnull(event_time) else ""
        game_hour = event_time.floor('h') if pd.notnull(event_time) else pd.NaT
        gid = game.get('id')
        prev_game = previous_map.get(gid, {})
        
        for book in game.get('bookmakers', []):
            book_key_raw = book.get('key', '').lower()
            book_key = book_key_raw  # default
        
            # 🔄 Normalize to known sharp/rec books
            for rec in REC_BOOKS:
                if rec.replace(" ", "") in book_key_raw:
                    book_key = rec.replace(" ", "")
                    break
            for sharp in SHARP_BOOKS:
                if sharp in book_key_raw:
                    book_key = sharp
                    break
        
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
                        'Event_Date': event_date,
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

                    # Fallback to previous game override if available
                    if prev_game:
                        for prev_b in prev_game.get('bookmakers', []):
                            if prev_b.get('key', '').lower() == book_key_raw:
                                for prev_m in prev_b.get('markets', []):
                                    if prev_m.get('key') == mtype:
                                        for prev_o in prev_m.get('outcomes', []):
                                            if normalize_label(prev_o.get('name')) == label:
                                                prev_val = prev_o.get('point') if mtype != 'h2h' else prev_o.get('price')
                                                if prev_val is not None:
                                                    entry['Old Value'] = prev_val
                                                    entry['Delta'] = round(val - prev_val, 2) if val is not None else None

                    rows.append(entry)
                    if len(rows) % 100 == 0:
                        print(f"📥 Processed {len(rows)} odds entries so far...")

                    line_history_log.setdefault(gid, []).append(entry.copy())

                    if val is not None:
                        sharp_lines[(game_name, mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit, val, old_val))
                        if book_key in SHARP_BOOKS:
                            sharp_total_limit_map[(game_name, mtype, label)] += limit or 0
                        if (game_name, mtype, label) not in line_open_map:
                            line_open_map[(game_name, mtype, label)] = (val, snapshot_time)

    df_moves = pd.DataFrame(rows)
    df_audit = pd.DataFrame([item for sublist in line_history_log.values() for item in sublist])
    df_sharp_lines = pd.DataFrame(sharp_lines.values())




    
    # === Sharp scoring logic
    # === Sharp scoring logic (safe version)
    for (game_name, mtype), label_map in sharp_limit_map.items():
        scores = {}
        label_signals = {}
        for label, entries in label_map.items():
            move_signal = limit_jump = prob_shift = time_score = 0
            move_magnitude_score = 0.0
    
            for limit, curr, _ in entries:
                open_val, _ = line_open_map.get((game_name, mtype, label), (None, None))
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
                            if imp_now and imp_open and imp_now > imp_open:
                                prob_shift += 1
                    except:
                        continue
    
                if limit is not None and limit >= 100:
                    limit_jump += 1
    
                try:
                    hour = datetime.now().hour
                    time_score += 1.0 if 6 <= hour <= 11 else 0.5 if hour <= 15 else 0.2
                except:
                    time_score += 0.5  # fallback
    
            move_magnitude_score = min(move_magnitude_score, 5.0)
            total_limit = sharp_total_limit_map.get((game_name, mtype, label), 0)
    
            scores[label] = (
                2 * move_signal +
                2 * limit_jump +
                1.5 * time_score +
                1.0 * prob_shift +
                0.001 * total_limit +
                3.0 * move_magnitude_score
            )
    
            label_signals[label] = {
                'Sharp_Move_Signal': move_signal,
                'Sharp_Limit_Jump': limit_jump,
                'Sharp_Time_Score': time_score,
                'Sharp_Prob_Shift': prob_shift,
                'Sharp_Limit_Total': total_limit,
                'Sharp_Move_Magnitude_Score': round(move_magnitude_score, 2)
            }
    
        if scores:
            best_label = max(scores, key=scores.get)
            sharp_side_flags[(game_name, mtype, best_label)] = 1
            sharp_metrics_map[(game_name, mtype, best_label)] = label_signals.get(best_label, {})

    # === Assign sharp-side logic + metric scores to all rows
    # === Assign sharp-side logic + metric scores
    for row in rows:
        game_name = row.get('Game', '')
        mtype = row.get('Market', '')
        label = row.get('Outcome', '')
        metrics = sharp_metrics_map.get((game_name, mtype, label), {})
        is_sharp_side = int(sharp_side_flags.get((game_name, mtype, label), 0))
    
        row.update({
            'Ref Sharp Value': row.get('Value'),
            'Ref Sharp Old Value': row.get('Old Value'),
            'Delta vs Sharp': 0.0,
            'SHARP_SIDE_TO_BET': is_sharp_side,
            'Sharp_Move_Signal': metrics.get('Sharp_Move_Signal', 0),
            'Sharp_Limit_Jump': metrics.get('Sharp_Limit_Jump', 0),
            'Sharp_Time_Score': metrics.get('Sharp_Time_Score', 0),
            'Sharp_Prob_Shift': metrics.get('Sharp_Prob_Shift', 0),
            'Sharp_Limit_Total': metrics.get('Sharp_Limit_Total', 0)
        })
    
        try:
            row['SharpBetScore'] = round(
                2.0 * row['Sharp_Move_Signal'] +
                2.0 * row['Sharp_Limit_Jump'] +
                1.5 * row['Sharp_Time_Score'] +
                1.0 * row['Sharp_Prob_Shift'] +
                0.001 * row['Sharp_Limit_Total'], 2
            )
        except:
            row['SharpBetScore'] = 0.0

    
    df = pd.DataFrame(rows)
    df_history = df.copy()  # needed for downstream sort on 'Time'

    # === Intelligence scoring
    def compute_weighted_signal(row, market_weights):
        market = str(row.get('Market', '')).lower()
        total_score = 0
        max_possible = 0
    
        component_importance = {
            'Sharp_Move_Signal': 2.0,
            'Sharp_Limit_Jump': 2.0,
            'Sharp_Time_Score': 1.5,
            'Sharp_Prob_Shift': 1.0,
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
        try:
            # Base scaled sharp signal
            base_score = min(row.get('SharpBetScore', 0) / 50, 1.0) * 50
    
            # Empirical weight signal
            weight_score = compute_weighted_signal(row, market_weights)
    
            # Limit positioning bonus
            limit_position_bonus = 0
            if row.get('LimitUp_NoMove_Flag') == 1:
                limit_position_bonus = 15
            elif row.get('Limit_Jump') == 1 and abs(row.get('Delta vs Sharp', 0)) > 0.25:
                limit_position_bonus = 5
    
            # Market leadership bonus
            market_lead_bonus = 5 if row.get('Market_Leader') else 0
    
            # Final blended score
            final_conf = base_score + weight_score + limit_position_bonus + market_lead_bonus
            return round(min(final_conf, 100), 2)
    
        except Exception as e:
            print(f"⚠️ Confidence scoring error: {e}")
            return 50.0  # fallback neutral score
    
 
    sharp_count = df[df['SHARP_SIDE_TO_BET'] == 1].shape[0]
   
    # === Sort by timestamp and extract open lines
    df_history_sorted = df_history.sort_values('Time')
    
    # Global open per market/outcome (first ever seen)
    line_open_df = (
        df_history_sorted.dropna(subset=['Value'])
        .groupby(['Game', 'Market', 'Outcome'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Value'})
    )
    
    # Per-book open value (first seen per book)
    line_open_per_book = (
        df_history_sorted.dropna(subset=['Value'])
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Book_Value'})
    )
    # === Extract opening limit for each (Game, Market, Outcome, Book)
    open_limit_df = (
        df_history_sorted
        .dropna(subset=['Limit'])
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Limit']
        .first()
        .reset_index()
        .rename(columns={'Limit': 'Opening_Limit'})
    )

    # === Early exit if empty
    if df.empty:
        return df, df_history
    
    # === Merge opening lines into live odds
    df = df.merge(line_open_df, on=['Game', 'Market', 'Outcome'], how='left')
    df = df.merge(line_open_per_book, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df = df.merge(open_limit_df, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df['Delta vs Sharp'] = df['Value'] - df['Open_Value']
    df['Delta'] = pd.to_numeric(df['Delta vs Sharp'], errors='coerce')
    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)
    df['Limit_Jump'] = (df['Limit'] >= 2500).astype(int)
    df['Sharp_Timing'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour.apply(
        lambda h: 1.0 if 6 <= h <= 11 else 0.5 if h <= 15 else 0.2
    )
    df['Limit_NonZero'] = df['Limit'].where(df['Limit'] > 0)
    df['Limit_Max'] = df.groupby(['Game', 'Market'])['Limit_NonZero'].transform('max')
    df['Limit_Min'] = df.groupby(['Game', 'Market'])['Limit_NonZero'].transform('min')
    
  
    df['Book'] = df['Book'].str.lower()
    
    market_leader_flags = detect_market_leaders(df_history, SHARP_BOOKS, REC_BOOKS)
    df = df.merge(
        market_leader_flags[['Game', 'Market', 'Outcome', 'Book', 'Market_Leader']],
        on=['Game', 'Market', 'Outcome', 'Book'],
        how='left'
    )
    
    df['Is_Pinnacle'] = df['Book'] == 'pinnacle'
    df['LimitUp_NoMove_Flag'] = (
        (df['Is_Pinnacle']) &
        (df['Limit'] >= 2500) &
        (df['Value'] == df['Open_Value'])
    ).astype(int)
    
    df = detect_cross_market_sharp_support(df)
    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Unique_Sharp_Books'] = df['Unique_Sharp_Books'].fillna(0).astype(int)
    df['LimitUp_NoMove_Flag'] = df['LimitUp_NoMove_Flag'].fillna(False).astype(int)
    df['Market_Leader'] = df['Market_Leader'].fillna(False).astype(int)
    # ✅ Compute dynamic, market-calibrated confidence
    df['True_Sharp_Confidence_Score'] = df.apply(
        lambda r: compute_weighted_signal(r, confidence_weights), axis=1
    )
    df['Enhanced_Sharp_Confidence_Score'] = df.apply(
        lambda r: compute_confidence(r, confidence_weights), axis=1
    )
    
    # ✅ Tiering based on enhanced score
    df['Sharp_Confidence_Tier'] = pd.cut(
        df['Enhanced_Sharp_Confidence_Score'],
        bins=[-1, 25, 50, 75, float('inf')],
        labels=['⚠️ Low', '✅ Medium', '⭐ High', '🔥 Steam']
    )
    
    # === Sharp vs Rec Book Consensus Summary
    rec_df = df[df['Book'].isin(REC_BOOKS)].copy()
    sharp_df = df[df['Book'].isin(SHARP_BOOKS)].copy()
    
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
    
    # Restore 'Recommended_Outcome'
    summary_df['Recommended_Outcome'] = summary_df['Outcome']
    
    # Compute move deltas
    summary_df['Move_From_Open_Rec'] = (
        summary_df['Rec_Book_Consensus'] - summary_df['Rec_Open']
    ).fillna(0)
    
    summary_df['Move_From_Open_Sharp'] = (
        summary_df['Sharp_Book_Consensus'] - summary_df['Sharp_Open']
    ).fillna(0)
    
    # Merge sharp scoring values
    sharp_scores = df[df['SharpBetScore'].notnull()][[
        'Event_Date', 'Game', 'Market', 'Outcome',
        'SharpBetScore',
        'Enhanced_Sharp_Confidence_Score',
        'Sharp_Confidence_Tier'
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


    required_sharp_cols = [
        'SharpBetScore', 'Enhanced_Sharp_Confidence_Score',
        'Sharp_Confidence_Tier', 'True_Sharp_Confidence_Score'
    ]
    for col in required_sharp_cols:
        if col not in df.columns:
            df[col] = None

    conf_df = df[df['Enhanced_Sharp_Confidence_Score'].notna()]
   
    return df, df_history, summary_df


from sklearn.isotonic import IsotonicRegression
import joblib

def train_sharp_model_from_bq(sport: str = "NBA", hours: int = 336, save_to_gcs: bool = True):
    from google.cloud import bigquery
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    import streamlit as st

    EXPECTED_FEATURES = [
        'Sharp_Move_Signal',
        'Sharp_Limit_Jump',
        'Sharp_Prob_Shift',
        'Sharp_Time_Score',
        'Sharp_Limit_Total',
        'Is_Reinforced_MultiMarket',
        'Market_Leader',
        'LimitUp_NoMove_Flag',
        'SharpBetScore',
        'Unique_Sharp_Books'
    ]

    client = bigquery.Client()
    sport = sport.upper()

    query = """
        SELECT *
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE SHARP_HIT_BOOL IS NOT NULL
          AND Scored = TRUE
          AND Sport = @sport
          AND Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @hours HOUR)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("sport", "STRING", sport),
            bigquery.ScalarQueryParameter("hours", "INT64", hours)
        ]
    )

    df_all = client.query(query, job_config=job_config).to_dataframe()
    if df_all.empty:
        st.warning(f"⚠️ No data found for {sport}")
        return {}

    df_all['target'] = df_all['SHARP_HIT_BOOL'].astype(int)
    df_all['Market'] = df_all['Market'].astype(str).str.lower()

    trained_models = {}

    for market_type in ['spreads', 'totals', 'h2h']:
        df = df_all[df_all['Market'] == market_type].copy()
        if len(df) < 10:
            st.info(f"ℹ️ Skipping {market_type} – not enough data.")
            continue

        # Ensure all features are present and numeric
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        X = df[EXPECTED_FEATURES].astype(float)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)

        # === Isotonic Calibration
        X_proba_train = model.predict_proba(X_train)[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(X_proba_train, y_train)

        # Save model and iso per market
        trained_models[market_type] = {
            "model": model,
            "calibrator": iso
        }

        # Save both if needed
        if save_to_gcs:
            save_model_to_gcs(model, calibrator=iso, sport=sport, market=market_type)

           

        # AUC
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        st.success(f"✅ Trained {sport} {market_type.upper()} – AUC: {auc:.3f} on {len(df)} samples")

    return trained_models



def apply_blended_sharp_score(df, trained_models):
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.info("🔍 Entered apply_blended_sharp_score()")
    df = df.copy()

    try:
        df = df.drop(columns=[col for col in df.columns if col.endswith(('_x', '_y'))], errors='ignore')
    except Exception as e:
        st.error(f"❌ Cleanup failed: {e}")
        return pd.DataFrame()

    try:
        if 'Enhanced_Sharp_Confidence_Score' in df.columns:
            df['Final_Confidence_Score'] = pd.to_numeric(df['Enhanced_Sharp_Confidence_Score'], errors='coerce') / 100
            df['Final_Confidence_Score'] = df['Final_Confidence_Score'].clip(0, 1)
    except Exception as e:
        st.warning(f"⚠️ Could not compute fallback confidence score: {e}")

    df['Market'] = df['Market'].astype(str).str.lower()

    for market_type, bundle in trained_models.items():
        model = bundle['model']
        iso = bundle['calibrator']
        df_market = df[df['Market'] == market_type].copy()

        if df_market.empty:
            continue

        try:
            model_features = model.get_booster().feature_names

            # Ensure all required features exist and are numeric
            for col in model_features:
                if col not in df_market.columns:
                    df_market[col] = 0
                df_market[col] = (
                    df_market[col]
                    .astype(str)
                    .replace({'True': 1, 'False': 0, 'true': 1, 'false': 0})
                )
                df_market[col] = pd.to_numeric(df_market[col], errors='coerce').fillna(0)

            df_market = df_market[model_features].astype(float)

            # Predict with calibration
            raw_probs = model.predict_proba(df_market)[:, 1]
            calibrated_probs = iso.predict(raw_probs)

            df.loc[df['Market'] == market_type, 'Model_Sharp_Win_Prob'] = raw_probs
            df.loc[df['Market'] == market_type, 'Model_Confidence'] = calibrated_probs

        except Exception as e:
            st.error(f"❌ Failed to apply model for {market_type}: {e}")

    # Apply tiering only if Model_Sharp_Win_Prob exists
    if 'Model_Sharp_Win_Prob' in df.columns:
        df['Model_Confidence'] = df['Model_Confidence'].fillna(0).clip(0, 1)
        df['Model_Confidence_Tier'] = pd.cut(
            df['Model_Sharp_Win_Prob'],
            bins=[0.0, 0.4, 0.5, 0.6, 1.0],
            labels=["⚠️ Weak Indication", "✅ Coinflip", "⭐ Lean", "🔥 Strong Indication"]
        )

    #st.success("✅ Model scoring complete (per-market)")
    return df

        
        
from io import BytesIO
import pickle
from google.cloud import storage

def save_model_to_gcs(model, calibrator, sport, market, bucket_name="sharp-models"):
    filename = f"sharp_win_model_{sport.lower()}_{market.lower()}.pkl"
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)

        # Save both model and calibrator together
        buffer = BytesIO()
        pickle.dump({"model": model, "calibrator": calibrator}, buffer)
        blob.upload_from_string(buffer.getvalue(), content_type='application/octet-stream')

        print(f"✅ Model + calibrator saved to GCS: gs://{bucket_name}/{filename}")
    except Exception as e:
        print(f"❌ Failed to save model to GCS: {e}")

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
        print(f"❌ Failed to load model from GCS: {e}")
        return None

        
        
        
@st.cache_data(ttl=300)
def fetch_scored_picks_from_bigquery(limit=50000):
    query = f"""
        SELECT *
        FROM `sharp_data.sharp_scores_full`
        WHERE SHARP_HIT_BOOL IS NOT NULL
        ORDER BY Snapshot_Timestamp DESC
        LIMIT {limit}
    """
    try:
        df = bq_client.query(query).to_dataframe()
        df['Snapshot_Timestamp'] = pd.to_datetime(df['Snapshot_Timestamp'], utc=True, errors='coerce')
        return df
    except Exception as e:
        st.error(f"❌ Failed to fetch scored picks: {e}")
        return pd.DataFrame()
        

def render_scanner_tab(label, sport_key, container):
    market_component_win_rates = read_market_weights_from_bigquery()
    timestamp = pd.Timestamp.utcnow()
    sport_key_lower = sport_key.lower()

    with container:
        st.subheader(f"📡 Scanning {label} Sharp Signals")

        live = fetch_live_odds(sport_key)
        if not live:
            st.warning("⚠️ No live odds returned.")
            return pd.DataFrame()

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
        
        #write_snapshot_to_gcs_parquet(live)
        prev = read_latest_snapshot_from_bigquery(hours=2) or {}
        # === Build df_prev_raw for audit
        df_prev_raw = pd.DataFrame([
            {
                "Game": f"{game.get('home_team')} vs {game.get('away_team')}",
                "Market": market['key'],
                "Outcome": o["name"],
                "Bookmaker": book.get("title", book.get("key", "Unknown Book")),
                "Value": o.get('point') if market['key'] != 'h2h' else o.get('price'),
                "Limit": o.get("bet_limit", 0),
                "Game_Start": pd.to_datetime(game.get("commence_time"), utc=True)
            }
            for game in prev.values()
            for book in game.get("bookmakers", [])
            for market in book.get("markets", [])
            for o in market.get("outcomes", [])
        ])
        
        df_prev_display = pd.DataFrame()
        if not df_prev_raw.empty:
            df_prev_raw['Value_Limit'] = df_prev_raw.apply(
                lambda r: f"{round(r['Value'], 1)} ({int(r['Limit'])})" if pd.notnull(r['Limit']) and pd.notnull(r['Value']) else "", axis=1
            )
            df_prev_raw['Date + Time (EST)'] = pd.to_datetime(df_prev_raw['Game_Start'], errors='coerce').dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.strftime('%Y-%m-%d %I:%M %p')
            df_prev_display = df_prev_raw.pivot_table(
                index=["Date + Time (EST)", "Game", "Market", "Outcome"],
                columns="Bookmaker",
                values="Value_Limit",
                aggfunc="first"
            ).reset_index()
        
        # === Fetch/calculate sharp signals
        detection_key = f"sharp_moves_{sport_key_lower}"
        if detection_key in st.session_state:
            df_moves_raw, df_audit, summary_df = st.session_state[detection_key]
          
            st.info(f"✅ Using cached sharp detection results for {label}")
        else:
            df_moves_raw, df_audit, summary_df = detect_sharp_moves(
                live, prev, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, weights=market_component_win_rates
            )
            st.session_state[detection_key] = (df_moves_raw, df_audit, summary_df)
            #st.success("🧠 Sharp detection run completed and cached.")
       # === Exit early if no data
        if df_moves_raw.empty or 'Enhanced_Sharp_Confidence_Score' not in df_moves_raw.columns:
            st.warning("⚠️ No sharp signals detected.")
            return pd.DataFrame()
        
        # === Enrich raw frame
        df_moves_raw['Game_Start'] = pd.to_datetime(df_moves_raw['Game_Start'], errors='coerce', utc=True)
        df_moves_raw['Snapshot_Timestamp'] = timestamp
        now = pd.Timestamp.utcnow()
        df_moves_raw['Pre_Game'] = df_moves_raw['Game_Start'] > now
        df_moves_raw['Post_Game'] = ~df_moves_raw['Pre_Game']
        df_moves_raw['Sport'] = label.upper()
        df_moves_raw = build_game_key(df_moves_raw)
        # === Diagnose Game_Key build failure
        required_fields = ['Game', 'Game_Start', 'Market', 'Outcome']
        missing_cols = [col for col in required_fields if col not in df_moves_raw.columns]
        if missing_cols:
            st.warning(f"❌ Missing columns before build_game_key: {missing_cols}")
        else:
            null_counts = df_moves_raw[required_fields].isnull().sum()
            st.info(f"🧪 Nulls in required Game_Key fields: {null_counts.to_dict()}")


        #write_sharp_moves_to_master(df_moves_raw)
        # === Load model
        # === Load per-market models
        model_key = f'sharp_models_{label.lower()}'
        trained_models = st.session_state.get(model_key)
        
        if trained_models is None:
            trained_models = {}
            for market_type in ['spreads', 'totals', 'h2h']:
                model_bundle = load_model_from_gcs(sport=label, market=market_type)
                if model_bundle:
                    trained_models[market_type] = model_bundle

                    
            st.session_state[model_key] = trained_models

        
        # === Apply model scoring
        # === Apply model scoring
        if trained_models:
            try:
                df_pre_game = df_moves_raw[df_moves_raw['Pre_Game']].copy()
             
        
                if not df_pre_game.empty:
                    df_scored = apply_blended_sharp_score(df_pre_game, trained_models)
                   
        
                    if not df_scored.empty:
                        merge_keys = ['Game_Key', 'Bookmaker', 'Market', 'Outcome']
                        score_cols = ['Model_Sharp_Win_Prob', 'Model_Confidence_Tier']
        
                        # Check for missing columns
                        missing = [col for col in merge_keys + score_cols if col not in df_scored.columns]
                        if missing:
                            st.warning(f"⚠️ Cannot merge scores — missing columns in df_scored: {missing}")
                            st.dataframe(df_scored.head(3))
                        else:
                            try:
                                df_moves_raw = df_moves_raw.merge(
                                    df_scored[merge_keys + score_cols],
                                    on=merge_keys,
                                    how='left',
                                    suffixes=('', '_scored')
                                )
                                for col in score_cols:
                                    scored_col = f"{col}_scored" if f"{col}_scored" in df_moves_raw.columns else col
                                    if scored_col in df_moves_raw.columns:
                                        df_moves_raw[col] = df_moves_raw[scored_col]
                                        if scored_col != col:
                                            df_moves_raw.drop(columns=[scored_col], inplace=True)
                                    else:
                                        st.warning(f"⚠️ Missing expected scored column: {scored_col}")
                            
                            except Exception as merge_error:
                                st.error(f"❌ Merge failed: {merge_error}")
                                st.dataframe(df_scored[merge_keys + score_cols].head(5))
                    else:
                        st.warning("⚠️ Model returned empty DataFrame during live scoring.")
                else:
                    st.info("ℹ️ No pre-game rows to score")
            except Exception as e:
                st.error(f"⚠️ Failed to apply model scoring: {e}")
        else:
            st.warning("⚠️ No per-market models available — skipping scoring.")

        
        # === Clean up and fill required columns
        df_moves_raw = df_moves_raw.rename(columns=lambda x: x.rstrip('_x'))
        df_moves_raw = df_moves_raw.drop(columns=[col for col in df_moves_raw.columns if col.endswith('_y')], errors='ignore')
        
        # Add fallback for Ref_Sharp_Value
        if 'Ref_Sharp_Value' not in df_moves_raw.columns and 'Value' in df_moves_raw.columns:
            df_moves_raw['Ref_Sharp_Value'] = df_moves_raw['Value']
        
        # Ensure scoring result placeholders
        for col in ['SHARP_HIT_BOOL', 'SHARP_COVER_RESULT', 'Scored']:
            if col not in df_moves_raw.columns:
                df_moves_raw[col] = None
        
        # Ensure model scoring outputs exist
        model_cols = ['Model_Sharp_Win_Prob', 'Model_Confidence_Tier']
        for col in model_cols:
            if col not in df_moves_raw.columns:
                df_moves_raw[col] = None
        
        # Prepare final deduplicated scoring view
        df_moves = df_moves_raw.drop_duplicates(subset=['Game_Key', 'Bookmaker'], keep='first')[['Game', 'Market', 'Outcome'] + model_cols]
                
        # === Enhance df_moves_raw with Model Reasoning and Confidence Trend
        
        # 1. First snapshot values per line
        df_first = df_moves_raw.sort_values('Snapshot_Timestamp') \
            .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='first') \
            .rename(columns={
                'Value': 'First_Line_Value',
                'Sharp_Confidence_Tier': 'First_Tier',
                'Enhanced_Sharp_Confidence_Score': 'First_Sharp_Prob'
            })[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'First_Line_Value', 'First_Tier', 'First_Sharp_Prob']]
        
                
        # 2. Merge into df_moves_raw
        df_moves_raw = df_moves_raw.merge(df_first, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left')
        
        # 3. Tier Change
        tier_rank = {"⚠️ Low": 1, "✅ Medium": 2, "⭐ High": 3, "🔥 Steam": 4}
        df_moves_raw['Tier_Change'] = df_moves_raw.apply(lambda row: (
            f"↑ {row['First_Tier']} → {row['Sharp_Confidence_Tier']}" if tier_rank.get(row['Sharp_Confidence_Tier'], 0) > tier_rank.get(row['First_Tier'], 0) else
            f"↓ {row['First_Tier']} → {row['Sharp_Confidence_Tier']}" if tier_rank.get(row['Sharp_Confidence_Tier'], 0) < tier_rank.get(row['First_Tier'], 0) else
            "↔ No Change"
        ), axis=1)
        
        # 4. Direction
        df_moves_raw['Prob_Delta'] = df_moves_raw['Model_Sharp_Win_Prob'] - df_moves_raw['First_Sharp_Prob']
        df_moves_raw['Line_Delta'] = df_moves_raw['Value'] - df_moves_raw['First_Line_Value']
        df_moves_raw['Direction'] = df_moves_raw.apply(lambda row: (
            "🟢 Model ↑ / Line ↓" if row['Prob_Delta'] > 0 and row['Line_Delta'] < 0 else
            "🔴 Model ↓ / Line ↑" if row['Prob_Delta'] < 0 and row['Line_Delta'] > 0 else
            "⚪ Mixed"
        ), axis=1)
        
        def build_model_reason(row):
            reasons = []
        
            try:
                win_prob = row.get('Model_Sharp_Win_Prob')
                if win_prob is not None and pd.notnull(win_prob) and win_prob > 0.55:
                    reasons.append("Model ↑")
            except:
                pass
        
            if row.get('Sharp_Prob_Shift', 0) > 0:
                reasons.append("Confidence ↑")
            if row.get('Sharp_Limit_Jump', 0):
                reasons.append("Limit Jump")
            if row.get('Market_Leader', 0):
                reasons.append("Led Move")
            if row.get('Is_Reinforced_MultiMarket', 0):
                reasons.append("Cross-Market Support")
            if row.get('LimitUp_NoMove_Flag', 0):
                reasons.append("Limit ↑ w/o Price Move")
        
            return " | ".join(reasons)

        
        df_moves_raw['📌 Model Reasoning'] = df_moves_raw.apply(build_model_reason, axis=1)
        
        # 6. Confidence Evolution
        def build_trend_explanation(row):
            start = row.get('First_Sharp_Prob', None)
            now = row.get('Model_Sharp_Win_Prob', None)
            if start is None or now is None:
                return ""
            delta = now - start
            trend = "↔ Stable"
            reason = []
        
            if delta >= 0.04:
                trend = "📈 Trending Up"
                if row.get('Sharp_Prob_Shift', 0) > 0:
                    reason.append("confidence ↑")
                if row.get('Sharp_Limit_Jump'):
                    reason.append("limit ↑")
                if row.get('Is_Reinforced_MultiMarket'):
                    reason.append("multi-market support")
            elif delta <= -0.04:
                trend = "📉 Trending Down"
                if row.get('Sharp_Prob_Shift', 0) < 0:
                    reason.append("confidence ↓")
                if not row.get('Sharp_Limit_Jump'):
                    reason.append("no limit activity")
                if not row.get('Market_Leader'):
                    reason.append("market resistance")
            else:
                trend = "↔ Stable"
                if abs(delta) > 0.01:
                    reason.append("minor shift")
        
            return f"{trend}: {start:.2f} → {now:.2f}" + (f" due to {', '.join(reason)}" if reason else "")
        
        df_moves_raw['📊 Confidence Evolution'] = df_moves_raw.apply(build_trend_explanation, axis=1)


        # === Run backtest (if not already done this session)


        backtest_date_key = f"last_backtest_date_{sport_key_lower}"
        today = datetime.utcnow().date()
        
        if st.session_state.get(backtest_date_key) != today:
            fetch_scores_and_backtest(
                sport_key, df_moves=None, api_key=API_KEY, trained_models=trained_models
            )
            st.session_state[backtest_date_key] = today
            st.success("✅ Backtesting and scoring completed for today.")
        else:
            st.info(f"⏭ Backtest already run today for {label.upper()} — skipping.")

                  
                        
            
            df_moves = df_moves_raw.copy()
        
        
        # === Upload line history
        if not df_audit.empty:
            df_audit['Snapshot_Timestamp'] = timestamp
            pass#write_line_history_to_bigquery(df_audit)
           

        # === 6. Summary Table ===
        if summary_df.empty:
            st.info("ℹ️ No summary data available.")
            return df_moves
        
        st.subheader(f"Sharp vs Rec Book Consensus Summary – {label}")
        
        # === Normalize keys
        for col in ['Game', 'Outcome']:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].str.strip().str.lower()
            if col in df_moves.columns:
                df_moves[col] = df_moves[col].str.strip().str.lower()
        
        # === Ensure required model columns exist
        model_cols = ['Model_Sharp_Win_Prob', 'Model_Confidence_Tier']
        for col in model_cols:
            if col not in df_moves.columns:
                df_moves[col] = None
        
        # === Merge model score + confidence tier
        model_cols_to_merge = ['Game', 'Market', 'Outcome'] + model_cols
        df_merge_scores = df_moves[model_cols_to_merge].drop_duplicates()
        summary_df = summary_df.merge(df_merge_scores, on=['Game', 'Market', 'Outcome'], how='left')
        
        # === Ensure extra diagnostic columns exist before merging
        for col in ['📌 Model Reasoning', '📊 Confidence Evolution', 'Tier_Change', 'Direction']:
            if col not in df_moves_raw.columns:
                df_moves_raw[col] = ""
        
        extra_cols = ['Game', 'Market', 'Outcome', '📌 Model Reasoning', '📊 Confidence Evolution', 'Tier_Change', 'Direction']
        df_extras = df_moves_raw[extra_cols].drop_duplicates()
        summary_df = summary_df.merge(df_extras, on=['Game', 'Market', 'Outcome'], how='left')
        
        # === Merge Game_Start for EST display
        if {'Event_Date', 'Market', 'Game'}.issubset(df_moves_raw.columns):
            df_game_start = df_moves_raw[['Game', 'Market', 'Event_Date', 'Game_Start', 'Model_Confidence_Tier']].dropna().drop_duplicates()
            df_game_start['MergeKey'] = (
                df_game_start['Game'].str.strip().str.lower() + "_" +
                df_game_start['Market'].str.strip().str.lower() + "_" +
                df_game_start['Event_Date'].astype(str)
            )
            summary_df['MergeKey'] = (
                summary_df['Game'].str.strip().str.lower() + "_" +
                summary_df['Market'].str.strip().str.lower() + "_" +
                summary_df['Event_Date'].astype(str)
            )
            summary_df = summary_df.merge(
                df_game_start[['MergeKey', 'Game_Start']],
                on='MergeKey',
                how='left'
            )
        
        # === Format to EST
        def safe_to_est(dt):
            if pd.isna(dt): return ""
            try:
                dt = pd.to_datetime(dt, errors='coerce')
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                return dt.tz_convert('US/Eastern').strftime('%Y-%m-%d %I:%M %p')
            except:
                return ""
        
        summary_df['Game_Start'] = pd.to_datetime(summary_df['Game_Start'], errors='coerce', utc=True)
        summary_df['Date + Time (EST)'] = summary_df['Game_Start'].apply(safe_to_est)
        summary_df = summary_df[summary_df['Date + Time (EST)'] != ""]
        
        # === Rename columns for display
        summary_df.rename(columns={
            'Date + Time (EST)': 'Date\n+ Time (EST)',
            'Game': 'Matchup',
            'Recommended_Outcome': 'Pick\nSide',
            'Rec_Book_Consensus': 'Rec\nConsensus',
            'Sharp_Book_Consensus': 'Sharp\nConsensus',
            'Move_From_Open_Rec': 'Rec\nMove',
            'Move_From_Open_Sharp': 'Sharp\nMove',
            'Model_Confidence_Tier': 'Confidence\nTier',
            '📌 Model Reasoning': 'Why Model Prefers',
            '📊 Confidence Evolution': 'Confidence Trend',
            'Tier_Change': 'Tier Δ',
            'Direction': 'Line/Model Direction'
        }, inplace=True)
        
        # === Drop duplicate rows
        summary_df = summary_df.drop_duplicates(subset=["Matchup", "Market", "Pick\nSide", "Date\n+ Time (EST)"])
        
        # === Market filter
        market_options = ["All"] + sorted(summary_df['Market'].dropna().unique())
        market = st.selectbox(f"📊 Filter {label} by Market", market_options, key=f"{label}_market_summary")
        filtered_df = summary_df if market == "All" else summary_df[summary_df['Market'] == market]

        
        # === Columns to show
        # === Columns to show
         # === Columns to show
        
        view_cols = [
            'Date\n+ Time (EST)', 'Matchup', 'Market', 'Pick\nSide',
            'Rec\nConsensus', 'Sharp\nConsensus', 'Rec\nMove', 'Sharp\nMove',
            'Model_Sharp_Win_Prob', 'Confidence\nTier',
            'Why Model Prefers', 'Confidence Trend', 'Tier Δ', 'Line/Model Direction'
        ]
        
        # === Filtered subset
        table_df = filtered_df[view_cols].copy()
        table_df.columns = [col.replace('\n', ' ') for col in table_df.columns]
        
        # === CSS Styling for All Tables (keep this once)
        st.markdown("""
        <style>
        .scrollable-table-container {
            max-height: 600px;
            overflow: auto;
            border: 1px solid #444;
            margin-bottom: 1rem;
            position: relative;
        }
        
        .custom-table {
            border-collapse: collapse;
            width: max-content;
            table-layout: fixed;
            font-size: 14px;
        }
        
        .custom-table th, .custom-table td {
            border: 1px solid #444;
            padding: 8px;
            text-align: center;
            white-space: nowrap;
        }
        
        .custom-table th {
            background-color: #1f2937;
            color: white;
            position: sticky;
            top: 0;
            z-index: 2;
        }
        
        /* Freeze first 4 columns */
        .custom-table th:nth-child(1),
        .custom-table td:nth-child(1) {
            position: sticky;
            left: 0;
            background-color: #2d3748;
            z-index: 3;
        }
        .custom-table th:nth-child(2),
        .custom-table td:nth-child(2) {
            position: sticky;
            left: 120px;
            background-color: #2d3748;
            z-index: 3;
        }
        .custom-table th:nth-child(3),
        .custom-table td:nth-child(3) {
            position: sticky;
            left: 240px;
            background-color: #2d3748;
            z-index: 3;
        }
        .custom-table th:nth-child(4),
        .custom-table td:nth-child(4) {
            position: sticky;
            left: 360px;
            background-color: #2d3748;
            z-index: 3;
        }
        
        .custom-table tr:nth-child(even) {
            background-color: #2d3748;
        }
        .custom-table tr:hover {
            background-color: #4b5563;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # === Render HTML Table
        table_html = table_df.to_html(classes="custom-table", index=False, escape=False)
        st.markdown(f"<div class='scrollable-table-container'>{table_html}</div>", unsafe_allow_html=True)
        
        # === Optional footer
        st.caption(f"Showing {len(table_df)} rows")
        # === Live Odds Snapshot Table ===
        st.subheader(f" Live Odds Snapshot – {label} (Odds + Limit)")
        odds_rows = []
        for game in live:
            game_name = f"{game['home_team']} vs {game['away_team']}"
            game_start = pd.to_datetime(game.get("commence_time")) if game.get("commence_time") else pd.NaT
        
            for book in game.get("bookmakers", []):
                if book.get("key") not in SHARP_BOOKS + REC_BOOKS:
                    continue  # ✅ Skip non-sharp/rec books
        
                for market in book.get("markets", []):
                    if market.get('key') not in ['h2h', 'spreads', 'totals']:
                        continue  # ✅ Skip non-target markets
        
                    for o in market.get("outcomes", []):
                        price = o.get('point') if market['key'] != 'h2h' else o.get('price')
                        odds_rows.append({
                            "Game": game_name,
                            "Market": market['key'],
                            "Outcome": o["name"],
                            "Bookmaker": book["title"],
                            "Value": price,
                            "Limit": o.get("bet_limit", 0),
                            "Game_Start": game_start
                        })

                
        df_odds_raw = pd.DataFrame(odds_rows)
        if not df_odds_raw.empty:
            df_odds_raw['Value_Limit'] = df_odds_raw.apply(
                lambda r: f"{round(r['Value'], 1)} ({int(r['Limit'])})" if pd.notnull(r['Limit']) and pd.notnull(r['Value'])
                else "" if pd.isnull(r['Value']) else f"{round(r['Value'], 1)}",
                axis=1
            )
        
            eastern = pytz_timezone('US/Eastern')
            df_odds_raw['Date + Time (EST)'] = pd.to_datetime(df_odds_raw['Game_Start'], errors='coerce').apply(
                lambda x: x.tz_convert(eastern).strftime('%Y-%m-%d %I:%M %p') if pd.notnull(x) and x.tzinfo
                else pd.to_datetime(x).tz_localize('UTC').tz_convert(eastern).strftime('%Y-%m-%d %I:%M %p') if pd.notnull(x)
                else ""
            )
        
            # === Pivot Current Snapshot ===
            df_display = df_odds_raw.pivot_table(
                index=["Date + Time (EST)", "Game", "Market", "Outcome"],
                columns="Bookmaker",
                values="Value_Limit",
                aggfunc="first"
            ).reset_index()
            
            # === Compare to Previous Snapshot to Highlight Changes ===
            if not df_prev_display.empty:
                df_compare = df_display.merge(
                    df_prev_display,
                    on=["Date + Time (EST)", "Game", "Market", "Outcome"],
                    suffixes=("", "_old"),
                    how="left"
                )
                change_mask = pd.DataFrame(False, index=df_compare.index, columns=df_display.columns)
                for col in df_display.columns:
                    old_col = f"{col}_old"
                    if old_col in df_compare.columns:
                        change_mask[col] = df_compare[col] != df_compare[old_col]
            else:
                df_compare = df_display.copy()
                change_mask = pd.DataFrame(False, index=df_display.index, columns=df_display.columns)
            # === Live Odds Table: Skip pagination and use full scrollable display ===
            paginated_df_2 = df_display.copy()
            compare_slice = df_compare.copy()
            mask_slice = change_mask.copy()
            # === Render Custom HTML Table with Change Highlighting ===
            def render_custom_html_table(df, highlight_cols, change_mask=None, df_compare=None):
                def style_cell(val, col, row_idx):
                    base = ""
                    arrow = ""
                    # Highlight sharp books
                    if col in highlight_cols and val != "":
                        base += "background-color:#d0f0c0; color:black; font-weight:600;"
                    # Highlight changed values
                    if change_mask is not None and col in change_mask.columns and change_mask.loc[row_idx, col]:
                        old_val = df_compare.iloc[row_idx].get(f"{col}_old", "")
                        new_val = df_compare.iloc[row_idx].get(col, "")
                        try:
                            val_float = float(str(new_val).split(" ")[0])
                            old_float = float(str(old_val).split(" ")[0])
                            if val_float > old_float:
                                arrow = " 🔺"
                                base += "background-color:#e6ffe6;"  # subtle green
                            elif val_float < old_float:
                                arrow = " 🔻"
                                base += "background-color:#ffe6e6;"  # subtle red
                        except:
                            pass
                    return base, arrow
            
                header_html = ''.join(f'<th>{col}</th>' for col in df.columns)
                rows_html = ''
                for i, row in df.iterrows():
                    row_html = '<tr>'
                    for col in df.columns:
                        val = row[col]
                        style, arrow = style_cell(val, col, i)
                        display_val = f"{val}{arrow}" if pd.notnull(val) else ""
                        row_html += f'<td style="{style}">{display_val}</td>'
                    row_html += '</tr>'
                    rows_html += row_html
            
                html = f"""
                <table class="custom-table">
                    <thead><tr>{header_html}</tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """
                return html
            
            html_table_2 = render_custom_html_table(
                paginated_df_2,
                highlight_cols=['Pinnacle', 'Bookmaker', 'BetOnline'],
                change_mask=mask_slice,
                df_compare=compare_slice
            )
            
            st.markdown(f"<div class='scrollable-table-container'>{html_table_2}</div>", unsafe_allow_html=True)

            st.caption(f"Showing {len(paginated_df_2)} rows")


        return df_moves

def fetch_scores_and_backtest(sport_key, df_moves=None, days_back=3, api_key=API_KEY, trained_models=None):

    expected_label = [k for k, v in SPORTS.items() if v == sport_key]
    sport_label = expected_label[0].upper() if expected_label else "NBA"

    # === 1. Fetch completed games ===
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
        response = requests.get(url, params={'apiKey': api_key, 'daysFrom': int(days_back)}, timeout=10)
        response.raise_for_status()
        games = response.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch scores: {e}")
        return pd.DataFrame()

    completed_games = [g for g in games if g.get("completed")]
    st.info(f"✅ Completed games: {len(completed_games)}")

    # === 2. Extract valid score rows ===
    score_rows = []
    for game in completed_games:
        home = normalize_team(game.get("home_team", ""))
        away = normalize_team(game.get("away_team", ""))
        game_start = pd.to_datetime(game.get("commence_time"), utc=True)
        merge_key = build_merge_key(home, away, game_start)
        scores = {s.get("name", "").strip().lower(): s.get("score") for s in game.get("scores", [])}
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
       
 

        pass#to_gbq(new_scores, 'sharp_data.game_scores_final', project_id=GCP_PROJECT_ID, if_exists='append')
        pass#st.success(f"✅ Uploaded {len(new_scores)} new game scores")
    except Exception as e:
        pass#st.error(f"❌ Failed to upload game scores: {e}")
        pass#st.code(new_scores.dtypes.to_string())

    # === 4. Load recent sharp picks
    df_master = read_recent_sharp_moves(hours=days_back * 72)
    df_master = build_game_key(df_master)
    df_master = ensure_columns(df_master, ['Game_Start'])
    df_master = df_master[df_master['Merge_Key_Short'].isin(df_scores['Merge_Key_Short'])]

    if df_master.empty:
        st.warning("⚠️ No sharp picks to backtest")
        return pd.DataFrame()

    # === 5. Merge scores and filter
    df = df_master.merge(
        df_scores[['Merge_Key_Short', 'Score_Home_Score', 'Score_Away_Score']],
        on='Merge_Key_Short', how='inner'
    )
    df = df[df['Book'].isin(SHARP_BOOKS + REC_BOOKS)]
    df = df[pd.to_datetime(df['Game_Start'], utc=True, errors='coerce') < pd.Timestamp.utcnow()]

    if 'Ref_Sharp_Value' not in df.columns:
        df['Ref_Sharp_Value'] = df.get('Value')
    else:
        df['Ref_Sharp_Value'] = df['Ref_Sharp_Value'].combine_first(df.get('Value'))

    # === 6. Calculate result
    df_valid = df.dropna(subset=['Score_Home_Score', 'Score_Away_Score', 'Ref_Sharp_Value'])
    if df_valid.empty:
        st.warning("ℹ️ No valid sharp picks with scores to evaluate")
        return pd.DataFrame()

    # === 6. Calculate result
    def calc_cover(row):
        try:
            h = float(row['Score_Home_Score'])
            a = float(row['Score_Away_Score'])
            val = float(row['Ref_Sharp_Value'])
            market = str(row.get('Market', '')).lower()
            outcome = str(row.get('Outcome', '')).lower()
    
            if market == 'totals':
                if 'under' in outcome and h + a < val:
                    return ['Win', 1]
                elif 'over' in outcome and h + a > val:
                    return ['Win', 1]
                else:
                    return ['Loss', 0]
    
            if market == 'spreads':
                margin = h - a if row['Home_Team_Norm'] in outcome else a - h
                hit = (margin > abs(val)) if val < 0 else (margin + val > 0)
                return ['Win', 1] if hit else ['Loss', 0]
    
            if market == 'h2h':
                home_win = row['Home_Team_Norm'] in outcome and h > a
                away_win = row['Away_Team_Norm'] in outcome and a > h
                return ['Win', 1] if home_win or away_win else ['Loss', 0]
    
            return [None, 0]
        except Exception:
            return [None, 0]
    
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

    # === 7. Apply model scoring if available
    if trained_models:
        try:
            df = apply_blended_sharp_score(df, trained_models)


        except Exception as e:
            st.warning(f"⚠️ Model scoring failed: {e}")
    
   # === 8. Final output DataFrame ===
    score_cols = [
        'Game_Key', 'Bookmaker', 'Market', 'Outcome', 'Ref_Sharp_Value',
        'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
        'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
        'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
        'Unique_Sharp_Books',
        'Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score',
        'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT', 'Scored', 'Sport'
    ]

    
    # Build full output
    df_scores_out = ensure_columns(df, score_cols)[score_cols].copy()
    df_scores_out['Sport'] = sport_label.upper()
    df_scores_out['Snapshot_Timestamp'] = pd.Timestamp.utcnow()  # ✅ Only do this once here
    # === Coerce and clean all fields BEFORE dedup and upload
    df_scores_out['Sharp_Move_Signal'] = pd.to_numeric(df_scores_out['Sharp_Move_Signal'], errors='coerce').astype('Int64')
    df_scores_out['Sharp_Limit_Jump'] = pd.to_numeric(df_scores_out['Sharp_Limit_Jump'], errors='coerce').astype('Int64')
    df_scores_out['Sharp_Prob_Shift'] = pd.to_numeric(df_scores_out['Sharp_Prob_Shift'], errors='coerce').astype('Int64')
    df_scores_out['Sharp_Time_Score'] = pd.to_numeric(df_scores_out['Sharp_Time_Score'], errors='coerce')
    df_scores_out['Sharp_Limit_Total'] = pd.to_numeric(df_scores_out['Sharp_Limit_Total'], errors='coerce')
    df_scores_out['Is_Reinforced_MultiMarket'] = df_scores_out['Is_Reinforced_MultiMarket'].fillna(False).astype(bool)
    df_scores_out['Market_Leader'] = df_scores_out['Market_Leader'].fillna(False).astype(bool)
    df_scores_out['LimitUp_NoMove_Flag'] = df_scores_out['LimitUp_NoMove_Flag'].fillna(False).astype(bool)
    df_scores_out['SharpBetScore'] = pd.to_numeric(df_scores_out['SharpBetScore'], errors='coerce')
    df_scores_out['Enhanced_Sharp_Confidence_Score'] = pd.to_numeric(df_scores_out['Enhanced_Sharp_Confidence_Score'], errors='coerce')
    df_scores_out['True_Sharp_Confidence_Score'] = pd.to_numeric(df_scores_out['True_Sharp_Confidence_Score'], errors='coerce')
    df_scores_out['SHARP_HIT_BOOL'] = pd.to_numeric(df_scores_out['SHARP_HIT_BOOL'], errors='coerce').astype('Int64')
    df_scores_out['SHARP_COVER_RESULT'] = df_scores_out['SHARP_COVER_RESULT'].fillna('').astype(str)
    df_scores_out['Scored'] = df_scores_out['Scored'].fillna(False).astype(bool)
    df_scores_out['Sport'] = df_scores_out['Sport'].fillna('').astype(str)
    df_scores_out['Unique_Sharp_Books'] = pd.to_numeric(df_scores_out['Unique_Sharp_Books'], errors='coerce').fillna(0).astype(int)

  
 
        # Debug: ensure schema matches
    try:
        import pyarrow as pa
        pa.Table.from_pandas(df_scores_out)
    except Exception as e:
        st.error("❌ Parquet validation failed before upload")
        st.code(str(e))
        st.write(df_scores_out.dtypes)
        st.stop()
    # ✅ Define full deduplication fingerprint (ignore timestamp)
    dedup_fingerprint_cols = score_cols.copy()  # includes all except timestamp
    
    # ✅ Remove local exact duplicates before querying BigQuery
    df_scores_out = df_scores_out.drop_duplicates(subset=dedup_fingerprint_cols)
    
    # ✅ Query BigQuery for existing fingerprints (same line state, any time)
    existing = bq_client.query(f"""
        SELECT DISTINCT {', '.join(dedup_fingerprint_cols)}
        FROM `sharp_data.sharp_scores_full`
        WHERE DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    """).to_dataframe()
    
    # ✅ Remove already-existing line states (not new even if timestamp is different)
    df_scores_out = df_scores_out.merge(
        existing,
        on=dedup_fingerprint_cols,
        how='left',
        indicator=True
    )
    df_scores_out = df_scores_out[df_scores_out['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    # ✅ Final upload
    if df_scores_out.empty:
        st.info("ℹ️ No new scored picks to upload — all identical line states already in BigQuery.")
        return df, pd.DataFrame()
    
 
    
    return df
    
    
# Safe predefinition
df_nba_bt = pd.DataFrame()
df_mlb_bt = pd.DataFrame()

def render_sharp_signal_analysis_tab(tab, sport_label, sport_key_api):
    with tab:
        st.subheader(f"📈 Model Calibration – {sport_label}")

        trained_models = {}
        for market_type in ['spreads', 'totals', 'h2h']:
            model_bundle = load_model_from_gcs(sport=sport_label, market=market_type)

            if model_bundle:
                trained_models[market_type] = model_bundle



        df_master = read_recent_sharp_moves(hours=168)
        if df_master.empty:
            st.warning(f"⚠️ No sharp picks found.")
            return

        df_master = df_master[df_master['Sport'] == sport_label.upper()]
        if df_master.empty:
            st.warning(f"⚠️ No data for {sport_label}.")
            return

        df_bt, _ = fetch_scores_and_backtest(
            sport_key_api, df_master.copy(), api_key=API_KEY, trained_models=trained_models
        )
        if df_bt.empty or 'Model_Sharp_Win_Prob' not in df_bt.columns:
            st.info("ℹ️ No scored predictions available for calibration analysis.")
            return

        df_bt = df_bt[df_bt['Scored'] == True].copy()

        # Clean & coerce
        df_bt['Model_Sharp_Win_Prob'] = pd.to_numeric(df_bt['Model_Sharp_Win_Prob'], errors='coerce')
        df_bt['Model_Confidence'] = pd.to_numeric(df_bt['Model_Confidence'], errors='coerce')

        # Define bins
        prob_bins = np.linspace(0, 1, 11)

        # Bin probabilities
        df_bt['Prob_Bin'] = pd.cut(df_bt['Model_Sharp_Win_Prob'], bins=prob_bins, labels=[f"{int(p*100)}–{int(prob_bins[i+1]*100)}%" for i, p in enumerate(prob_bins[:-1])])
        df_bt['Conf_Bin'] = pd.cut(df_bt['Model_Confidence'], bins=prob_bins, labels=[f"{int(p*100)}–{int(prob_bins[i+1]*100)}%" for i, p in enumerate(prob_bins[:-1])])

        # Group and summarize
        prob_summary = (
            df_bt.groupby(['Market', 'Prob_Bin'])['SHARP_HIT_BOOL']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'Picks', 'mean': 'Win_Rate'})
            .reset_index()
        )
        
        conf_summary = (
            df_bt.groupby(['Market', 'Conf_Bin'])['SHARP_HIT_BOOL']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'Picks', 'mean': 'Win_Rate'})
            .reset_index()
        )


        st.markdown("#### 🔢 Model Probability Calibration by Market")
        for market in prob_summary['Market'].unique():
            st.markdown(f"**📊 {market.upper()}**")
            st.dataframe(
                prob_summary[prob_summary['Market'] == market]
                .drop(columns='Market')
                .style.format({'Win_Rate': '{:.1%}'})
            )
        
        st.markdown("#### 🎯 Model Confidence Calibration by Market")
        for market in conf_summary['Market'].unique():
            st.markdown(f"**📊 {market.upper()}**")
            st.dataframe(
                conf_summary[conf_summary['Market'] == market]
                .drop(columns='Market')
                .style.format({'Win_Rate': '{:.1%}'})
            )

tab_nba, tab_mlb, tab_cfl, tab_wnba = st.tabs(["🏀 NBA", "⚾ MLB", "🏈 CFL", "🏀 WNBA"])

# --- NBA Tab Block
with tab_nba:
    st.subheader("🏀 NBA Sharp Scanner")
    run_nba = st.checkbox("Run NBA Scanner", value=True, key="run_nba_scanner")
    if st.button("📈 Train NBA Sharp Model"):
        train_sharp_model_from_bq(sport="NBA")

    if run_nba:
        if any(st.session_state.get(k) for k in ["run_mlb_scanner", "run_cfl_scanner", "run_wnba_scanner"]):
            st.warning("⚠️ Please disable other scanners to run NBA.")
        else:
            scan_tab, analysis_tab = st.tabs(["📡 Live Scanner", "📈 Backtest Analysis"])
            with scan_tab:
                df_nba_live = render_scanner_tab("NBA", SPORTS["NBA"], scan_tab)
            with analysis_tab:
                render_sharp_signal_analysis_tab(analysis_tab, "NBA", SPORTS["NBA"])

# --- MLB Tab Block
with tab_mlb:
    st.subheader("⚾ MLB Sharp Scanner")
    run_mlb = st.checkbox("Run MLB Scanner", value=False, key="run_mlb_scanner")
    if st.button("⚾ Train MLB Sharp Model"):
        train_sharp_model_from_bq(sport="MLB")
    if run_mlb:
        if any(st.session_state.get(k) for k in ["run_nba_scanner", "run_cfl_scanner", "run_wnba_scanner"]):
            st.warning("⚠️ Please disable other scanners to run MLB.")
        else:
            scan_tab, analysis_tab = st.tabs(["📡 Live Scanner", "📈 Backtest Analysis"])
            with scan_tab:
                df_mlb_live = render_scanner_tab("MLB", SPORTS["MLB"], scan_tab)
            with analysis_tab:
                render_sharp_signal_analysis_tab(analysis_tab, "MLB", SPORTS["MLB"])

# --- CFL Tab Block
with tab_cfl:
    st.subheader("🏈 CFL Sharp Scanner")
    run_cfl = st.checkbox("Run CFL Scanner", value=False, key="run_cfl_scanner")
    if st.button("🏈 Train CFL Sharp Model"):
        train_sharp_model_from_bq(sport="CFL")
    if run_cfl:
        if any(st.session_state.get(k) for k in ["run_nba_scanner", "run_mlb_scanner", "run_wnba_scanner"]):
            st.warning("⚠️ Please disable other scanners to run CFL.")
        else:
            scan_tab, analysis_tab = st.tabs(["📡 Live Scanner", "📈 Backtest Analysis"])
            with scan_tab:
                df_cfl_live = render_scanner_tab("CFL", SPORTS["CFL"], scan_tab)
            with analysis_tab:
                render_sharp_signal_analysis_tab(analysis_tab, "CFL", SPORTS["CFL"])

# --- WNBA Tab Block
with tab_wnba:
    st.subheader("🏀 WNBA Sharp Scanner")
    run_wnba = st.checkbox("Run WNBA Scanner", value=False, key="run_wnba_scanner")
    if st.button("🏀 Train WNBA Sharp Model"):
        train_sharp_model_from_bq(sport="WNBA")
    if run_wnba:
        if any(st.session_state.get(k) for k in ["run_nba_scanner", "run_mlb_scanner", "run_cfl_scanner"]):
            st.warning("⚠️ Please disable other scanners to run WNBA.")
        else:
            scan_tab, analysis_tab = st.tabs(["📡 Live Scanner", "📈 Backtest Analysis"])
            with scan_tab:
                df_wnba_live = render_scanner_tab("WNBA", SPORTS["WNBA"], scan_tab)
            with analysis_tab:
                render_sharp_signal_analysis_tab(analysis_tab, "WNBA", SPORTS["WNBA"])

