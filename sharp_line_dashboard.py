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
    text-align: left;
    word-break: break-word;
    white-space: normal;
    max-width: 220px;  /* or whatever max width fits your layout */
}
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
from datetime import datetime
from datetime import date, timedelta
from collections import defaultdict, OrderedDict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pytz import timezone as pytz_timezone
from google.oauth2 import service_account
import pandas_gbq
import pandas as pd
import pandas_gbq  # ✅ Required for setting .context.project / .context.credentials
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss
import requests
import traceback
from io import BytesIO
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.stats import entropy
from sklearn.model_selection import train_test_split

import logging
GCP_PROJECT_ID = "sharplogger"  # ✅ confirmed project ID
BQ_DATASET = "sharp_data"       # ✅ your dataset name
BQ_TABLE = "sharp_moves_master" # ✅ your table name
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
MARKET_WEIGHTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.market_weights"
LINE_HISTORY_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.line_history_master"
SNAPSHOTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.odds_snapshot_log"
GCS_BUCKET = "sharp-models"
import os, json



import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

pandas_gbq.context.project = GCP_PROJECT_ID  # credentials will be inferred

bq_client = bigquery.Client(project=GCP_PROJECT_ID)  # uses env var
gcs_client = storage.Client(project=GCP_PROJECT_ID)



# === Constants and Config ===
API_KEY = "3879659fe861d68dfa2866c211294684"
#FOLDER_ID = "1v6WB0jRX_yJT2JSdXRvQOLQNfOZ97iGA"
REDIRECT_URI = "https://sharp-scanner-723770381669.us-east4.run.app/"  # no longer used for login, just metadata

SPORTS = {
    "NBA": "basketball_nba",
    "WNBA": "basketball_wnba",
    "MLB": "baseball_mlb",
    "CFL": "americanfootball_cfl",
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "NCAAB": "basketball_ncaab"
}
SPORT_ALIAS_MAP = {
    "NBA": "basketball_nba",
    "WNBA": "basketball_wnba",
    "MLB": "baseball_mlb",
    "CFL": "americanfootball_cfl",
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "NCAAB": "basketball_ncaab",
}
SHARP_BOOKS_FOR_LIMITS = ['pinnacle']
SHARP_BOOKS = SHARP_BOOKS_FOR_LIMITS + ['betus','mybookieag','betfair_ex_eu','betfair_ex_uk','lowvig']

REC_BOOKS = [
    'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers',
    'fanatics', 'espnbet', 'hardrockbet']

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
        st.info(f"📡 Fetching odds for `{sport_key}`...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        odds_data = response.json()
        if not odds_data:
            st.warning("⚠️ No odds returned from API.")
        return odds_data

    except requests.exceptions.HTTPError as http_err:
        st.error(f"❌ HTTP error: {http_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"❌ Request error: {req_err}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
    
    return []  # Return empty list on failure


   
def read_from_bigquery(table_name):
    from google.cloud import bigquery
    try:
        client = bigquery.Client()
        return client.query(f"SELECT * FROM `{table_name}`").to_dataframe()
    except Exception as e:
        st.error(f"❌ Failed to load `{table_name}`: {e}")
        return pd.DataFrame()
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



def read_recent_sharp_moves(hours=72, table=BQ_FULL_TABLE):
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

# ✅ Cached wrapper for diagnostics and line movement history
@st.cache_data(ttl=600)
def get_recent_history():
    return read_recent_sharp_moves(hours=72)
       
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


from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, brier_score_loss



    
    
def train_sharp_model_from_bq(sport: str = "NBA", days_back: int = 14):
    st.info(f"🎯 Training sharp model for {sport.upper()}...")

    # ✅ Load from sharp_scores_full with all necessary columns up front
    query = f"""
        SELECT *
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE Sport = '{sport.upper()}'
          AND Scored = TRUE
          AND SHARP_HIT_BOOL IS NOT NULL
          AND DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
    """

    df_bt = bq_client.query(query).to_dataframe()

    if df_bt.empty:
        st.warning("⚠️ No historical sharp picks available to train model.")
        return

    df_bt = df_bt.copy()
    df_bt['SHARP_HIT_BOOL'] = pd.to_numeric(df_bt['SHARP_HIT_BOOL'], errors='coerce')
 
    dedup_cols = [
        'Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Value',
        'Sharp_Move_Signal', 'Sharp_Limit_Jump',
        'Sharp_Time_Score', 'Sharp_Limit_Total',
        'Is_Reinforced_MultiMarket', 'Market_Leader', 'LimitUp_NoMove_Flag'
    ]

    before = len(df_bt)
    df_bt = df_bt.drop_duplicates(subset=dedup_cols, keep='last')
    after = len(df_bt)

    trained_models = {}
    progress = st.progress(0)
    status = st.status("🔄 Training in progress...", expanded=True)

    for idx, market in enumerate(['spreads', 'totals', 'h2h'], start=1):
        status.write(f"🚧 Training model for `{market.upper()}`...")
        df_market = df_bt[df_bt['Market'] == market].copy()

        if df_market.empty:
            status.warning(f"⚠️ No data for {market.upper()} — skipping.")
            progress.progress(idx / 3)
            continue

        # ✅ You now safely have Home_Team_Norm and Away_Team_Norm here
        # You can continue with canonical filtering, label validation, feature engineering, etc.

        # Normalize team columns
        df_market['Outcome_Norm'] = df_market['Outcome'].astype(str).str.lower().str.strip()
        df_market['Home_Team_Norm'] = df_market['Home_Team_Norm'].astype(str).str.lower().str.strip()
        df_market['Away_Team_Norm'] = df_market['Away_Team_Norm'].astype(str).str.lower().str.strip()
        # Only train on labeled rows
        
        # === Canonical side filtering ONLY ===
        if market == "totals":
            df_market = df_market[df_market['Outcome_Norm'] == 'over']
        
        elif market == "spreads":
            df_market = df_market[df_market['Value'] < 0]  # Favorite only
        
        elif market == "h2h":
            df_market = df_market[df_market['Value'] < 0]  # Favorite only
        
        # ✅ Use existing SHARP_HIT_BOOL as-is (already precomputed)
        df_market = df_market[df_market['SHARP_HIT_BOOL'].isin([0, 1])]
        def label_team_role(row):
            market = row['Market']
            value = row['Value']
            outcome = row['Outcome_Norm']
            home_team = row['Home_Team_Norm']
            
            if market in ['spreads', 'h2h']:
                if value < 0:
                    return 'favorite'
                elif value > 0:
                    return 'underdog'
                else:
                    return 'even'  # fallback (rare)
            elif market == 'totals':
                if outcome == 'over':
                    return 'over'
                elif outcome == 'under':
                    return 'under'
                else:
                    return 'unknown'
            else:
                return 'unknown'
            
        df_market['Team_Bet_Role'] = df_market.apply(label_team_role, axis=1)
        # Normalize team identifiers
        df_market['Team'] = df_market['Outcome_Norm']
        df_market['Is_Home'] = (df_market['Team'] == df_market['Home_Team_Norm']).astype(int)
        
        # === Overall team stats
        team_stats = (
            df_market.groupby('Team')
            .agg({
                'Model_Sharp_Win_Prob': 'mean',
                'SHARP_HIT_BOOL': 'mean'
            })
            .rename(columns={
                'Model_Sharp_Win_Prob': 'Team_Past_Avg_Model_Prob',
                'SHARP_HIT_BOOL': 'Team_Past_Hit_Rate'
            })
            .reset_index()
        )
        
        # === Home-only stats
        home_stats = (
            df_market[df_market['Is_Home'] == 1]
            .groupby('Team')
            .agg({
                'Model_Sharp_Win_Prob': 'mean',
                'SHARP_HIT_BOOL': 'mean'
            })
            .rename(columns={
                'Model_Sharp_Win_Prob': 'Team_Past_Avg_Model_Prob_Home',
                'SHARP_HIT_BOOL': 'Team_Past_Hit_Rate_Home'
            })
            .reset_index()
        )
        
        # === Away-only stats
        away_stats = (
            df_market[df_market['Is_Home'] == 0]
            .groupby('Team')
            .agg({
                'Model_Sharp_Win_Prob': 'mean',
                'SHARP_HIT_BOOL': 'mean'
            })
            .rename(columns={
                'Model_Sharp_Win_Prob': 'Team_Past_Avg_Model_Prob_Away',
                'SHARP_HIT_BOOL': 'Team_Past_Hit_Rate_Away'
            })
            .reset_index()
        )
        
        # === Merge all stats into training data
        df_market = df_market.merge(team_stats, on='Team', how='left')
        df_market = df_market.merge(home_stats, on='Team', how='left')
        df_market = df_market.merge(away_stats, on='Team', how='left')

        if df_market.empty or df_market['SHARP_HIT_BOOL'].nunique() < 2:
            status.warning(f"⚠️ Not enough label variety for {market.upper()} — skipping.")
            progress.progress(idx / 3)
            continue
        # === Directional agreement (for spreads/h2h invert line logic)
        df_market['Line_Delta'] = pd.to_numeric(df_market['Line_Delta'], errors='coerce')
       
        
        df_market['Direction_Aligned'] = np.where(
            df_market['Line_Delta'] > 0, 1,
            np.where(df_market['Line_Delta'] < 0, 0, -1)
        ).astype(int)
        df_market['Line_Value_Abs'] = df_market['Value'].abs()
        df_market['Prob_Shift_Signed'] = df_market['Sharp_Prob_Shift'] * np.sign(df_market['Value'])
        df_market['Line_Delta_Signed'] = df_market['Line_Delta'] * np.sign(df_market['Value'])
        
        
        df_market['Book_Norm'] = df_market['Bookmaker'].str.lower().str.strip()
        df_market['Is_Sharp_Book'] = df_market['Book_Norm'].isin(SHARP_BOOKS).astype(int)
        # === Sharp vs. Recreational Line Movement
        df_market['Sharp_Line_Delta'] = np.where(
            df_market['Is_Sharp_Book'] == 1,
            df_market['Line_Delta'],
            0
        )
        
        df_market['Rec_Line_Delta'] = np.where(
            df_market['Is_Sharp_Book'] == 0,
            df_market['Line_Delta'],
            0
        )
        
        # Optional: absolute versions if you're using magnitude
       # === Magnitude & Directional Features (retain only de-correlated ones)
        df_market['Sharp_Line_Magnitude'] = df_market['Sharp_Line_Delta'].abs()
        df_market['Delta_Sharp_vs_Rec'] =  df_market['Rec_Line_Delta'] - df_market['Sharp_Line_Delta']
        df_market['Sharp_Leads'] = (df_market['Sharp_Line_Magnitude'] > df_market['Rec_Line_Delta'].abs()).astype('int')
        
        # Optional: Keep for diagnostics only, not training
        # df_market['Line_Move_Magnitude'] = df_market['Line_Delta'].abs()
        
        # === Contextual Flags
        df_market['Is_Home_Team_Bet'] = (df_market['Outcome'] == df_market['Home_Team_Norm']).astype(int)
        df_market['Is_Favorite_Bet'] = (df_market['Value'] < 0).astype(int)
        # Optional: Keep as debug flag, not as model feature
        # df_market['High_Limit_Flag'] = (df_market['Sharp_Limit_Total'] > 10000).astype(int)
        
        # Ensure NA-safe boolean logic and conversion
        df_market['SharpMove_Odds_Up'] = (
            ((df_market['Sharp_Move_Signal'] == 1) & (df_market['Odds_Shift'] > 0))
            .fillna(False)
            .astype(int)
        )
        
        df_market['SharpMove_Odds_Down'] = (
            ((df_market['Sharp_Move_Signal'] == 1) & (df_market['Odds_Shift'] < 0))
            .fillna(False)
            .astype(int)
        )
        
        df_market['SharpMove_Odds_Mag'] = (
            df_market['Odds_Shift'].abs().fillna(0) * df_market['Sharp_Move_Signal'].fillna(0)
        )

        # === Interaction Features (filtered for value)
        #if 'Odds_Shift' in df_market.columns:
            #df_market['SharpMove_OddsShift'] = df_market['Sharp_Move_Signal'] * df_market['Odds_Shift']
        
        if 'Implied_Prob_Shift' in df_market.columns:
            df_market['MarketLeader_ImpProbShift'] = df_market['Market_Leader'] * df_market['Implied_Prob_Shift']
        
        df_market['SharpLimit_SharpBook'] = df_market['Is_Sharp_Book'] * df_market['Sharp_Limit_Total']
        df_market['LimitProtect_SharpMag'] = df_market['LimitUp_NoMove_Flag'] * df_market['Sharp_Line_Magnitude']
        # Example for engineered features

        df_market['High_Limit_Flag'] = (df_market['Sharp_Limit_Total'] >= 7000).astype(int)
        df_market['Was_Line_Resistance_Broken'] = df_market.get('Was_Line_Resistance_Broken', 0).fillna(0).astype(int)
        df_market['SharpMove_Resistance_Break'] = (
            df_market['Sharp_Move_Signal'].fillna(0).astype(int) *
            df_market['Was_Line_Resistance_Broken'].fillna(0).astype(int)
        )

        # === Normalize Resistance Break Count
        df_market['Line_Resistance_Crossed_Count'] = (
            pd.to_numeric(df_market.get('Line_Resistance_Crossed_Count'), errors='coerce')
            .fillna(0)
            .astype(int)
        )
        
        # === Optional: store decoded JSON for preview/debug only (not as model input)
        df_market['Line_Resistance_Crossed_Levels'] = df_market.get('Line_Resistance_Crossed_Levels', '[]')

        df_market['Value_Reversal_Flag'] = df_market.get('Value_Reversal_Flag', 0).fillna(0).astype(int)
        df_market['Odds_Reversal_Flag'] = df_market.get('Odds_Reversal_Flag', 0).fillna(0).astype(int)
        #df_market['Is_Team_Favorite'] = (df_market['Team_Bet_Role'] == 'favorite').astype(int)
        #df_market['Is_Team_Underdog'] = (df_market['Team_Bet_Role'] == 'underdog').astype(int)
        #df_market['Is_Team_Over'] = (df_market['Team_Bet_Role'] == 'over').astype(int)
        #df_market['Is_Team_Under'] = (df_market['Team_Bet_Role'] == 'under').astype(int)


        
            # === Resistance Flag Debug
        with st.expander(f"📊 Resistance Flag Debug – {market.upper()}"):
            st.write("Value Counts:")
            st.write(df_market['Was_Line_Resistance_Broken'].value_counts(dropna=False))
    
            st.write("Sample Resistance Breaks:")
            st.dataframe(
                df_market[df_market['Was_Line_Resistance_Broken'] == 1][
                    ['Game_Key', 'Market', 'Outcome', 'First_Line_Value', 'Value', 'Was_Line_Resistance_Broken']
                ].head(10)
            )
    
            if df_market['Was_Line_Resistance_Broken'].sum() == 0:
                st.warning("⚠️ No line resistance breaks detected.")
            else:
                st.success("✅ Resistance break logic is populating correctly.")
        # === 🧠 Add new features to training
        features = [
            # 🔹 Core sharp signals
            'Sharp_Move_Signal', 'Sharp_Limit_Jump', #'Sharp_Time_Score', 
            'Sharp_Limit_Total',
            'Is_Reinforced_MultiMarket', 'Market_Leader', 'LimitUp_NoMove_Flag',
        
            # 🔹 Market response
            'Sharp_Line_Magnitude', 'Is_Home_Team_Bet',
        
            # 🔹 Engineered odds shift decomposition
            'SharpMove_Odds_Up', 'SharpMove_Odds_Down', 'SharpMove_Odds_Mag',
        
            # 🔹 Engineered interactions
            'MarketLeader_ImpProbShift', 'LimitProtect_SharpMag', 'Delta_Sharp_vs_Rec',
            'Sharp_Leads', 'SharpMove_Resistance_Break',
        
            # 🔹 Resistance feature
            'Line_Resistance_Crossed_Count',  # ✅ newly added here
        
            # 🔁 Reversal logic
            'Value_Reversal_Flag', 'Odds_Reversal_Flag',
        
            # 🔥 Timing flags
            'Late_Game_Steam_Flag',
            
            'Abs_Line_Move_From_Opening',
            'Abs_Odds_Move_From_Opening', 
        ]
        hybrid_timing_features = [
            
            f'SharpMove_Magnitude_{b}' for b in [
                'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
                'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
                'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
                'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
            ]
        ]
        hybrid_odds_timing_features = [

            f'OddsMove_Magnitude_{b}' for b in [
                'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
                'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
                'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
                'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
            ]
        ]   
        features += hybrid_timing_features
        features += hybrid_odds_timing_features
    
        features += [
            'Team_Past_Avg_Model_Prob',
            'Team_Past_Hit_Rate',
            'Team_Past_Avg_Model_Prob_Home',
            'Team_Past_Hit_Rate_Home',
            'Team_Past_Avg_Model_Prob_Away',
            'Team_Past_Hit_Rate_Away',
            
            # Optional betting context flags
            #'Is_Team_Favorite', 'Is_Team_Underdog',
            #'Is_Team_Over', 'Is_Team_Under'
        ]

        st.markdown(f"### 📈 Features Used: `{len(features)}`")
        df_market = ensure_columns(df_market, features, 0)

        X = df_market[features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        # Step: Check for multicollinearity in features
        corr_matrix = X.corr().abs()
        
        # Threshold for flagging redundancy
        threshold = 0.85
        
        # Collect highly correlated feature pairs (excluding self-pairs)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                f1 = corr_matrix.columns[i]
                f2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                if corr > threshold:
                    high_corr_pairs.append((f1, f2, corr))
        
        # Display as DataFrame
        if high_corr_pairs:
            df_corr = pd.DataFrame(high_corr_pairs, columns=['Feature_1', 'Feature_2', 'Correlation'])
            df_corr = df_corr.sort_values(by='Correlation', ascending=False)
            
            st.subheader(f"🔁 Highly Correlated Features — {market.upper()}")
            st.dataframe(df_corr)
        else:
            st.success("✅ No highly correlated feature pairs found")
        y = df_market['SHARP_HIT_BOOL'].astype(int)
        
        # === Abort early if label has only one class
        if y.nunique() < 2:
            st.warning(f"⚠️ Skipping {market.upper()} — only one label class.")
            progress.progress(idx / 3)
            continue
      
        # === Check each fold for label balance
        
        
        
        # === Param grid (expanded)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
            'min_child_weight': [3, 5, 7, 10],
            'gamma': [0, 0.1, 0.3],
            'reg_alpha': [0.1, 0.5, 1.0, 5.0],  # L1
            'reg_lambda': [1.0, 5.0, 10.0],     # L2
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        
   
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        
    
        
        
        # === LogLoss model
        # ✅ Calculate scale_pos_weight for class imbalance
        scale_pos_weight = (len(y) - y.sum()) / y.sum()
        
        # ✅ LogLoss Model Grid Search
        grid_logloss = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(
                eval_metric='logloss',
                tree_method='hist',
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight  # ✅ Proper location
            ),
            param_distributions=param_grid,
            scoring='neg_log_loss',
            cv=cv,
            n_iter=50,
            verbose=1,
            random_state=42
        )
        grid_logloss.fit(X_train, y_train)
        model_logloss = grid_logloss.best_estimator_
        
        # ✅ AUC Model Grid Search
        grid_auc = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(
                eval_metric='logloss',
                tree_method='hist',
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight  # ✅ Proper location
            ),
            param_distributions=param_grid,
            scoring='roc_auc',
            cv=cv,
            n_iter=50,
            verbose=1,
            random_state=42
        )
        grid_auc.fit(X_train, y_train)
        model_auc = grid_auc.best_estimator_
        from collections import Counter

        # Count class distribution in training set
        class_counts = Counter(y_train)
        min_class_count = min(class_counts.values())
        
        if min_class_count < 5:
            st.warning(f"⚠️ Not enough samples per class for isotonic calibration in {market.upper()} — skipping calibration.")
            cal_logloss = model_logloss
            cal_auc = model_auc
        else:
            cal_logloss = CalibratedClassifierCV(model_logloss, method='isotonic', cv=cv).fit(X_train, y_train)
            cal_auc = CalibratedClassifierCV(model_auc, method='isotonic', cv=cv).fit(X_train, y_train)
        # ✅ Use isotonic calibration (more stable for reducing std dev)
        
        
        # === Predict calibrated probabilities
        prob_logloss = cal_logloss.predict_proba(X)[:, 1]
        prob_auc = cal_auc.predict_proba(X)[:, 1]
        # === Predict calibrated probabilities on validation set
        val_prob_logloss = cal_logloss.predict_proba(X_val)[:, 1]
        val_prob_auc = cal_auc.predict_proba(X_val)[:, 1]
        
        # === Evaluate on holdout
        val_prob_auc = np.clip(val_prob_auc, 0.05, 0.95)
        val_prob_logloss = np.clip(val_prob_logloss, 0.05, 0.95)

        val_auc_logloss = roc_auc_score(y_val, val_prob_logloss)
        val_auc_auc = roc_auc_score(y_val, val_prob_auc)
        # === Log holdout performance
        st.markdown(f"### 🧪 Holdout Validation – `{market.upper()}`")
        st.write(f"- LogLoss Model AUC: `{val_auc_logloss:.4f}`")
        st.write(f"- AUC Model AUC: `{val_auc_auc:.4f}`")
        
        if y_val.nunique() < 2:
            st.warning("⚠️ Cannot compute log loss or Brier score — only one label class in validation set.")
            val_logloss = np.nan
            val_brier = np.nan
        else:
            val_logloss = log_loss(y_val, val_prob_auc, labels=[0, 1])
            val_brier = brier_score_loss(y_val, val_prob_auc)
        
            st.write(f"- Holdout LogLoss: `{val_logloss:.4f}`")
            st.write(f"- Holdout Brier Score: `{val_brier:.4f}`")
        
        # === Compute AUCs for weighting
        auc_logloss = roc_auc_score(y, prob_logloss)
        auc_auc = roc_auc_score(y, prob_auc)
        
        # === Normalize AUCs to get ensemble weights
        total_auc = auc_logloss + auc_auc
        w_logloss = auc_logloss / total_auc
        w_auc = auc_auc / total_auc
        
        # === Weighted ensemble
        ensemble_prob = w_logloss * prob_logloss + w_auc * prob_auc

       # === Std deviation of ensemble probabilities (measures confidence tightness)
        std_dev_pred = np.std(ensemble_prob)
        
        # === Spread of calibrated win probabilities
        prob_range = np.max(ensemble_prob) - np.min(ensemble_prob)
        
        # === Sharpeness score (low entropy = sharper predictions)
        entropy = -np.mean([
            p * np.log(p + 1e-10) + (1 - p) * np.log(1 - p + 1e-10)
            for p in ensemble_prob
        ])
        
        st.markdown(f"### 🔍 Prediction Confidence Analysis – `{market.upper()}`")
        st.write(f"- Std Dev of Predictions: `{std_dev_pred:.4f}`")
        st.write(f"- Probability Range: `{prob_range:.4f}`")
        st.write(f"- Avg Prediction Entropy: `{entropy:.4f}`")


        # === Threshold sweep
        thresholds = np.arange(0.1, 0.96, 0.05)
        threshold_metrics = []
        
        for thresh in thresholds:
            preds = (ensemble_prob >= thresh).astype(int)
            threshold_metrics.append({
                'Threshold': round(thresh, 2),
                'Accuracy': accuracy_score(y, preds),
                'Precision': precision_score(y, preds, zero_division=0),
                'Recall': recall_score(y, preds, zero_division=0),
                'F1': f1_score(y, preds, zero_division=0)
            })
        
        # Create DataFrame for display
        df_thresh = pd.DataFrame(threshold_metrics)
        df_thresh = df_thresh[['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1']]
        
        # Display in Streamlit
        st.markdown(f"#### 📈 Performance by Threshold – `{market.upper()}`")
        st.dataframe(df_thresh.style.format({c: "{:.3f}" for c in df_thresh.columns if c != 'Threshold'}))
        
        y_pred = (ensemble_prob >= 0.6).astype(int)
        # === Log weight contributions
        dominant_model = "AUC" if w_auc > w_logloss else "LogLoss"
        st.markdown(f"🧠 **Ensemble Weighting:**")
        st.write(f"- AUC Model Weight: `{w_auc:.2f}`")
        st.write(f"- LogLoss Model Weight: `{w_logloss:.2f}`")
        st.success(f"📌 Dominant Model in Ensemble: **{dominant_model} Model**")
       
        # === Metrics
        auc = roc_auc_score(y, ensemble_prob)
        acc = accuracy_score(y, y_pred)
        logloss = log_loss(y, ensemble_prob)
        brier = brier_score_loss(y, ensemble_prob)
        st.markdown("### 📉 Overfitting Check – Gap Analysis")
        st.write(f"- AUC Gap (Train - Holdout): `{auc - val_auc_auc:.4f}`")
        st.write(f"- LogLoss Gap (Train - Holdout): `{logloss - val_logloss:.4f}`")
        st.write(f"- Brier Gap (Train - Holdout): `{brier - val_brier:.4f}`")

        importances = model_auc.feature_importances_
        feature_names = features[:len(importances)]
        
        # === Estimate directional impact via correlation with model output
        # This assumes you have access to the original training data X and predictions
        X_features = X[feature_names]  # original training feature matrix
        preds = model_auc.predict_proba(X_features)[:, 1]  # class 1 probabilities
        
        # Estimate sign via correlation
        correlations = np.array([
            np.corrcoef(X_features[col], preds)[0, 1] if np.std(X_features[col]) > 0 else 0
            for col in feature_names
        ])
        
        impact_directions = np.where(correlations > 0, '↑ Increases', '↓ Decreases')
        
        # === Combine into one table
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Impact': impact_directions
        }).sort_values(by='Importance', ascending=False)
        
        st.markdown(f"#### 📊 Feature Importance & Impact for `{market.upper()}`")
        st.table(importance_df.head(10))
        
        # === Calibration
        prob_true, prob_pred = calibration_curve(y, ensemble_prob, n_bins=10)
        calib_df = pd.DataFrame({
            "Predicted Bin Center": prob_pred,
            "Actual Hit Rate": prob_true
        })
        st.markdown(f"#### 🎯 Calibration Bins – {market.upper()}")
        st.dataframe(calib_df)
        
        team_feature_map = (
            df_market.groupby('Team')
            .agg({
                'Model_Sharp_Win_Prob': 'mean',
                'SHARP_HIT_BOOL': 'mean',                
                'Team_Past_Avg_Model_Prob_Home': 'mean',
                'Team_Past_Hit_Rate_Home': 'mean',
                'Team_Past_Avg_Model_Prob_Away': 'mean',
                'Team_Past_Hit_Rate_Away': 'mean'
            })
            .rename(columns={
                'Model_Sharp_Win_Prob': 'Team_Past_Avg_Model_Prob',
                'SHARP_HIT_BOOL': 'Team_Past_Hit_Rate',               
            })
            .reset_index()
        )



        # === Save ensemble (choose one or both)
        trained_models[market] = {
            "model": model_auc,
            "calibrator": cal_auc,
            "team_feature_map": team_feature_map  # ✅ include this here for later use
        }
        save_model_to_gcs(model_auc, cal_auc, sport, market, bucket_name=GCS_BUCKET, team_feature_map=team_feature_map)
        from scipy.stats import entropy
        
        # === Base features already known to work well
        base_features = [
            'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Time_Score',
            'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket', 'Market_Leader',
            'LimitUp_NoMove_Flag', 'Sharp_Line_Magnitude', 'Is_Home_Team_Bet'
        ]
        
        # === Optional new features to test
        optional_feature_sets = [
            ['Rec_Line_Magnitude'],
            ['SharpMove_OddsShift'],
            ['MarketLeader_ImpProbShift'],
            ['Delta_Sharp_vs_Rec'],
            ['Sharp_Leads'],
            ['SharpLimit_SharpBook'],
            ['LimitProtect_SharpMag'],
            ['High_Limit_Flag'],
            ['Is_Favorite_Bet']
        ]
        
        results = []
        
        for opt_feats in optional_feature_sets:
            current_features = base_features + opt_feats
            df_market = ensure_columns(df_market, current_features, 0)
            
            X = df_market[current_features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            y = df_market['SHARP_HIT_BOOL'].astype(int)
        
            X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
        
            # === Simple model
            model = xgb.XGBClassifier(eval_metric='logloss', tree_method='hist', n_jobs=-1)
            model.fit(X_train, y_train)
            proba_val = model.predict_proba(X_val)[:, 1]
        
            # === Metrics
            val_logloss = log_loss(y_val, proba_val)
            val_brier = brier_score_loss(y_val, proba_val)
            val_std = np.std(proba_val)
            val_entropy = entropy(np.histogram(proba_val, bins=10, range=(0, 1), density=True)[0])
        
            results.append({
                'Feature_Added': opt_feats[0],
                'LogLoss': val_logloss,
                'Brier': val_brier,
                'Prob Std Dev': val_std,
                'Prediction Entropy': val_entropy
            })
        
        # Show results
        results_df = pd.DataFrame(results).sort_values(by='LogLoss')
        st.markdown("### 🧪 Optional Feature Comparison")
        st.dataframe(results_df.style.format("{:.4f}", subset=['LogLoss', 'Brier', 'Prob Std Dev', 'Prediction Entropy']))

        st.success(f"""✅ Trained + saved ensemble model for {market.upper()}
        - AUC: {auc:.4f}
        - Accuracy: {acc:.4f}
        - Log Loss: {logloss:.4f}
        - Brier Score: {brier:.4f}
        """)
        progress.progress(idx / 3)

    status.update(label="✅ All models trained", state="complete", expanded=False)
    if not trained_models:
        st.error("❌ No models were trained.")
    return trained_models

from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.stats import entropy
import numpy as np

def evaluate_model_confidence_and_performance(X_train, y_train, X_val, y_val, model_label="Base"):
    model = xgb.XGBClassifier(eval_metric='logloss', tree_method='hist', n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict probabilities
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_val = model.predict_proba(X_val)[:, 1]

    # Confidence metrics
    std_dev = np.std(prob_val)
    prob_range = round(prob_val.max() - prob_val.min(), 4)
    avg_entropy = np.mean([
        entropy([p, 1 - p], base=2) if 0 < p < 1 else 0 for p in prob_val
    ])

    # Performance metrics
    auc_val = roc_auc_score(y_val, prob_val)
    brier_val = brier_score_loss(y_val, prob_val)
    logloss_val = log_loss(y_val, prob_val)

    st.markdown(f"### 🧪 Confidence & Performance – `{model_label}`")
    st.write(f"- Std Dev of Predictions: `{std_dev:.4f}`")
    st.write(f"- Probability Range: `{prob_range:.4f}`")
    st.write(f"- Avg Prediction Entropy: `{avg_entropy:.4f}`")
    st.write(f"- Holdout AUC: `{auc_val:.4f}`")
    st.write(f"- Holdout Log Loss: `{logloss_val:.4f}`")
    st.write(f"- Holdout Brier Score: `{brier_val:.4f}`")

    return {
        "model": model,
        "std_dev": std_dev,
        "prob_range": prob_range,
        "entropy": avg_entropy,
        "auc": auc_val,
        "brier": brier_val,
        "logloss": logloss_val
    }


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
def compute_diagnostics_vectorized(df):
    df = df.copy()

    # === Ensure only latest snapshot per bookmaker is used (to match df_summary_base logic)
    df = (
        df.sort_values('Snapshot_Timestamp')
        .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='last')
    )

    # === Tier ordering for change tracking
    TIER_ORDER = {
        '🪙 Low Probability': 1,
        '🤏 Lean': 2,
        '🔥 Strong Indication': 3,
        '🌋 Steam': 4
    }

    # === Normalize tier columns
    for col in ['Confidence Tier', 'First_Tier']:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()

    # === Tier Δ logic
    tier_now = df['Confidence Tier'].map(TIER_ORDER).fillna(0).astype(int)
    tier_first = df['First_Tier'].map(TIER_ORDER).fillna(0).astype(int)
    df['Tier_Change'] = np.where(
        df['First_Tier'] != "",
        np.where(
            tier_now > tier_first,
            "↑ " + df['First_Tier'] + " → " + df['Confidence Tier'],
            np.where(
                tier_now < tier_first,
                "↓ " + df['First_Tier'] + " → " + df['Confidence Tier'],
                "↔ No Change"
            )
        ),
        "⚠️ Missing"
    )
            
    # === Confidence Trend
    prob_now = pd.to_numeric(df['Model Prob'], errors='coerce')
    prob_start = pd.to_numeric(df['First_Sharp_Prob'], errors='coerce')
    
    df['Model Prob Snapshot'] = prob_now
    df['First Prob Snapshot'] = prob_start


    df['Confidence Trend'] = np.where(
        prob_now.isna() | prob_start.isna(),
        "⚠️ Missing",
        np.where(
            prob_now - prob_start >= 0.04,
            "📈 Trending Up: " + (prob_start * 100).round(1).astype(str) + "% → " + (prob_now * 100).round(1).astype(str) + "%",
            np.where(
                prob_now - prob_start <= -0.04,
                "📉 Trending Down: " + (prob_start * 100).round(1).astype(str) + "% → " + (prob_now * 100).round(1).astype(str) + "%",
                "↔ Stable: " + (prob_start * 100).round(1).astype(str) + "% → " + (prob_now * 100).round(1).astype(str) + "%"
            )
        )
    )

    # === Line/Model Direction Alignment
    df['Line_Delta'] = pd.to_numeric(df.get('Line_Delta'), errors='coerce')
    df['Line_Support_Sign'] = df.apply(
        lambda row: -1 if row.get('Market', '').lower() == 'totals' and row.get('Outcome', '').lower() == 'under' else 1,
        axis=1
    )
    df['Line_Support_Direction'] = df['Line_Delta'] * df['Line_Support_Sign']
    prob_trend = prob_now - prob_start
    df['Line/Model Direction'] = np.select(
        [
            (prob_trend > 0) & (df['Line_Support_Direction'] > 0),
            (prob_trend < 0) & (df['Line_Support_Direction'] < 0),
            (prob_trend > 0) & (df['Line_Support_Direction'] < 0),
            (prob_trend < 0) & (df['Line_Support_Direction'] > 0),
        ],
        [
            "🟢 Aligned ↑",
            "🔻 Aligned ↓",
            "🔴 Model ↑ / Line ↓",
            "🔴 Model ↓ / Line ↑"
        ],
        default="⚪ Mixed"
    )

    # ✅ Cast all diagnostic flags to numeric

    # ✅ Cast all diagnostic flags to numeric
    flag_cols = [
        'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Market_Leader',
        'Is_Reinforced_MultiMarket', 'LimitUp_NoMove_Flag', 'Is_Sharp_Book',
        'SharpMove_Odds_Up', 'SharpMove_Odds_Down', 'Is_Home_Team_Bet',
        'SharpMove_Resistance_Break', 'Late_Game_Steam_Flag',
        'Value_Reversal_Flag', 'Odds_Reversal_Flag',
        'Hybrid_Line_Timing_Flag', 'Hybrid_Odds_Timing_Flag'
    ]
    
    magnitude_cols = [
        'Sharp_Line_Magnitude', 'Sharp_Time_Score', 'Rec_Line_Magnitude',
        'Sharp_Limit_Total', 'SharpMove_Odds_Mag', 'SharpMove_Timing_Magnitude',
        'Abs_Line_Move_From_Opening', 'Abs_Odds_Move_From_Opening',
        'Team_Past_Hit_Rate', 'Team_Past_Avg_Model_Prob',
        'Team_Past_Hit_Rate_Home', 'Team_Past_Hit_Rate_Away',
        'Team_Past_Avg_Model_Prob_Home', 'Team_Past_Avg_Model_Prob_Away'
    ]

    for col in flag_cols + magnitude_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    HYBRID_LINE_COLS = [
        f'SharpMove_Magnitude_{b}' for b in [
            'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
            'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
            'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
            'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
        ]
    ]
    
    HYBRID_ODDS_COLS = [
        f'OddsMove_Magnitude_{b}' for b in [
            'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
            'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
            'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
            'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
        ]
    ]
    # === Compute Hybrid Line/Odds Timing Flags if missing
    if 'Hybrid_Line_Timing_Flag' not in df.columns:
        df['Hybrid_Line_Timing_Flag'] = (df[HYBRID_LINE_COLS].sum(axis=1) > 0).astype(int)
    
    if 'Hybrid_Odds_Timing_Flag' not in df.columns:
        df['Hybrid_Odds_Timing_Flag'] = (df[HYBRID_ODDS_COLS].sum(axis=1) > 0).astype(int)


    # === Passes Gate
    df['Passes_Gate'] = (
        pd.to_numeric(df['Model Prob'], errors='coerce') >= 0.0
    ) & (df['Active_Signal_Count'] > 1)  # You can adjust the threshold if needed
    
    # === Confidence Tier from Model
    model_prob = pd.to_numeric(df['Model Prob'], errors='coerce')

    # Assign base tiers
    df['Confidence Tier'] = pd.cut(
        model_prob,
        bins=[0, 0.4, 0.6, 0.8, 1.0],
        labels=["🪙Low Probability", "🤏 Lean", "🔥 Strong Indication", "🌋 Steam"]
    ).astype(str)
    
    # Override with "zero" if probability is exactly 0
    df.loc[model_prob == 0.0, 'Confidence Tier'] = "None"

    
    # === Why Model Likes It
    def build_why(row):
        model_prob = row.get('Model Prob')
    
        if pd.isna(model_prob):
            return "⚠️ Missing — run apply_blended_sharp_score() first"
    
        if not row.get('Passes_Gate', False):
            return "🕓 Still Calculating Signal"
    
        parts = []

    
        # === Core sharp move reasons
        parts = []

   

        if row.get('Sharp_Move_Signal'): parts.append("📈 Sharp Move Detected")
        if row.get('Sharp_Limit_Jump'): parts.append("💰 Limit Jumped")
        if row.get('Market_Leader'): parts.append("🏆 Market Leader")
        if row.get('Is_Reinforced_MultiMarket'): parts.append("📊 Multi-Market Consensus")
        if row.get('LimitUp_NoMove_Flag'): parts.append("🛡️ Limit Up + No Line Move")
        if row.get('Is_Sharp_Book'): parts.append("🎯 Sharp Book Signal")
    
        # === Odds & line movement
        if row.get('SharpMove_Odds_Up'): parts.append("🟢 Odds Moved Up (Steam)")
        if row.get('SharpMove_Odds_Down'): parts.append("🔻 Odds Moved Down (Buyback)")
        if row.get('Is_Home_Team_Bet'): parts.append("🏠 Home Team Bet")
        if row.get('Sharp_Line_Magnitude', 0) > 0.5: parts.append("📏 Big Line Move")
        if row.get('Rec_Line_Magnitude', 0) > 0.5: parts.append("📉 Rec Book Move")
        if row.get('Sharp_Limit_Total', 0) > 10000: parts.append("💼 Sharp High Limit")
        if row.get('SharpMove_Odds_Mag', 0) > 5: parts.append("💥 Sharp Odds Steam")
    
        # === Resistance and timing
        if row.get('SharpMove_Resistance_Break'): parts.append("🧱 Broke Key Resistance")
        if row.get('Late_Game_Steam_Flag'): parts.append("⏰ Late Game Steam")
        if row.get('Value_Reversal_Flag'): parts.append("🔄 Value Reversal")
        if row.get('Odds_Reversal_Flag'): parts.append("📉 Odds Reversal")
        if row.get('Sharp_Time_Score', 0) > 0.5: parts.append("⏱️ Timing Edge")
        # === Team-level diagnostics
        if row.get('Team_Past_Hit_Rate', 0) > 0.6:
            parts.append("⚔️📊 Team Historically Sharp")
        if row.get('Team_Past_Avg_Model_Prob', 0) > 0.6:
            parts.append("🔮 Model Favored This Team Historically")
        # === Line/odds movement from open
        if row.get('Abs_Line_Move_From_Opening', 0) > 1.0:
            parts.append("📈 Line Moved from Open")
        if row.get('Abs_Odds_Move_From_Opening', 0) > 5.0:
            parts.append("💹 Odds Moved from Open")
        
    
        # === Hybrid timing buckets — LINE
        HYBRID_LINE_COLS = [
            'SharpMove_Magnitude_Overnight_VeryEarly', 'SharpMove_Magnitude_Overnight_MidRange',
            'SharpMove_Magnitude_Overnight_LateGame', 'SharpMove_Magnitude_Overnight_Urgent',
            'SharpMove_Magnitude_Early_VeryEarly', 'SharpMove_Magnitude_Early_MidRange',
            'SharpMove_Magnitude_Early_LateGame', 'SharpMove_Magnitude_Early_Urgent',
            'SharpMove_Magnitude_Midday_VeryEarly', 'SharpMove_Magnitude_Midday_MidRange',
            'SharpMove_Magnitude_Midday_LateGame', 'SharpMove_Magnitude_Midday_Urgent',
            'SharpMove_Magnitude_Late_VeryEarly', 'SharpMove_Magnitude_Late_MidRange',
            'SharpMove_Magnitude_Late_LateGame', 'SharpMove_Magnitude_Late_Urgent'
        ]
    
        # === Hybrid timing buckets — ODDS
        HYBRID_ODDS_COLS = [
            'OddsMove_Magnitude_Overnight_VeryEarly', 'OddsMove_Magnitude_Overnight_MidRange',
            'OddsMove_Magnitude_Overnight_LateGame', 'OddsMove_Magnitude_Overnight_Urgent',
            'OddsMove_Magnitude_Early_VeryEarly', 'OddsMove_Magnitude_Early_MidRange',
            'OddsMove_Magnitude_Early_LateGame', 'OddsMove_Magnitude_Early_Urgent',
            'OddsMove_Magnitude_Midday_VeryEarly', 'OddsMove_Magnitude_Midday_MidRange',
            'OddsMove_Magnitude_Midday_LateGame', 'OddsMove_Magnitude_Midday_Urgent',
            'OddsMove_Magnitude_Late_VeryEarly', 'OddsMove_Magnitude_Late_MidRange',
            'OddsMove_Magnitude_Late_LateGame', 'OddsMove_Magnitude_Late_Urgent'
        ]
    
        EMOJI_MAP = {
            'Overnight': '🌙', 'Early': '🌅',
            'Midday': '🌞', 'Late': '🌆'
        }
    
        for col in HYBRID_LINE_COLS:
            if row.get(col, 0) > 0.25:
                bucket = col.replace('SharpMove_Magnitude_', '').replace('_', ' ')
                emoji = EMOJI_MAP.get(col.split('_')[2], '⏱️')
                parts.append(f"{emoji} {bucket} Sharp Move")
    
        for col in HYBRID_ODDS_COLS:
            if row.get(col, 0) > 0.5:
                bucket = col.replace('OddsMove_Magnitude_', '').replace('_', ' ')
                emoji = EMOJI_MAP.get(col.split('_')[2], '⏱️')
                parts.append(f"{emoji} {bucket} Odds Steam")
    
        return " + ".join(parts) if parts else "🤷‍♂️ Still Calculating"
    
    # Apply to DataFrame
   
    df['Why Model Likes It'] = df.apply(build_why, axis=1)
    df['Model_Confidence_Tier'] = df['Confidence Tier']  # ✅ snapshot tier for summary view

    # === Final Output
    diagnostics_df = df[[
        'Game_Key', 'Market', 'Outcome', 'Bookmaker',
        'Tier_Change', 'Confidence Trend', 'Line/Model Direction',
        'Why Model Likes It', 'Passes_Gate', 'Confidence Tier',
        'Model Prob Snapshot', 'First Prob Snapshot',
        'Model_Confidence_Tier'   # ✅ Add this
    ]].rename(columns={
        'Tier_Change': 'Tier Δ'
    })


    
    return diagnostics_df



from google.cloud import storage
from io import BytesIO
import pickle
import logging

def save_model_to_gcs(model, calibrator, sport, market, team_feature_map=None, bucket_name="sharp-models"):
    """
    Save a trained model, calibrator, and optional team_feature_map to Google Cloud Storage.

    Parameters:
    - model: Trained XGBoost or sklearn model
    - calibrator: CalibratedClassifierCV or similar
    - sport (str): e.g., "nfl"
    - market (str): e.g., "spreads"
    - team_feature_map (pd.DataFrame or None): Optional team-level stats used in scoring
    - bucket_name (str): GCS bucket name
    """
    filename = f"sharp_win_model_{sport.lower()}_{market.lower()}.pkl"

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)

        # Prepare payload
        payload = {
            "model": model,
            "calibrator": calibrator
        }

        if team_feature_map is not None:
            payload["team_feature_map"] = team_feature_map

        # Serialize to bytes
        buffer = BytesIO()
        pickle.dump(payload, buffer)
        buffer.seek(0)

        # Upload to GCS
        blob.upload_from_file(buffer, content_type='application/octet-stream')

        print(f"✅ Model + calibrator{' + team features' if team_feature_map is not None else ''} saved to GCS: gs://{bucket_name}/{filename}")

    except Exception as e:
        logging.error(f"❌ Failed to save model to GCS: {e}", exc_info=True)

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

        
        
        

def fetch_scored_picks_from_bigquery(limit=1000000):
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
        
# ✅ Step: Fill missing opposite picks (mirror rows that didn’t exist pre-merge)
def ensure_opposite_side_rows(df, scored_df):
    
    #Ensure both sides of each market are represented by injecting the mirrored side if missing.
    
    merge_keys = ['Game_Key', 'Market', 'Bookmaker', 'Outcome']

    # Identify which scored rows are not already in df
    scored_keys = scored_df[merge_keys].drop_duplicates()
    existing_keys = df[merge_keys].drop_duplicates()
    missing = scored_keys.merge(existing_keys, how='left', on=merge_keys, indicator=True)
    missing = missing[missing['_merge'] == 'left_only'].drop(columns=['_merge'])

    if not missing.empty:
        injected = scored_df.merge(missing, on=merge_keys, how='inner')
        st.info(f"➕ Injected {len(injected)} mirrored rows not present in original data.")
        # Fill missing fields with NaNs
        for col in df.columns:
            if col not in injected.columns:
                injected[col] = np.nan
        df = pd.concat([df, injected[df.columns]], ignore_index=True)
    else:
        st.info("✅ No mirrored rows needed — both sides already present.")

    return df




def render_scanner_tab(label, sport_key, container):
    # Optional: Load market weights, skip if not found
    # market_weights = read_market_weights_from_bigquery()
    # if not market_weights:
    #     st.error("❌ No market weights found. Cannot compute confidence scores.")
    #     return

    # 🔄 Auto-refresh only if not paused
    if st.session_state.get("pause_refresh", False):
        st.info("⏸️ Auto-refresh paused")
        return  # ❌ exit early — no fetching, no rendering
    
    timestamp = pd.Timestamp.utcnow()
    sport_key_lower = sport_key.lower()

    with container:
        st.subheader(f"📡 Scanning {label} Sharp Signals")

        # === Fetch live odds for display table
        live = fetch_live_odds(sport_key)
        if not live:
            st.warning("⚠️ No live odds returned.")
            return pd.DataFrame()

        # === Build current snapshot table (live odds view)
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

        # === Previous snapshot for live odds comparison
        # === Previous snapshot for live odds comparison
        prev = read_latest_snapshot_from_bigquery(hours=48) or {}
        
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
        
        # === Format and display snapshot table
        df_prev_display = pd.DataFrame()
        if not df_prev_raw.empty:
            df_prev_raw['Value_Limit'] = df_prev_raw.apply(
                lambda r: f"{round(r['Value'], 1)} ({int(r['Limit'])})"
                if pd.notnull(r['Limit']) and pd.notnull(r['Value']) else "", axis=1
            )
            df_prev_raw['Date + Time (EST)'] = pd.to_datetime(df_prev_raw['Game_Start'], errors='coerce')\
                .dt.tz_localize('UTC').dt.tz_convert('US/Eastern').dt.strftime('%Y-%m-%d %I:%M %p')
            df_prev_display = df_prev_raw.pivot_table(
                index=["Date + Time (EST)", "Game", "Market", "Outcome"],
                columns="Bookmaker",
                values="Value_Limit",
                aggfunc="first"
            ).reset_index()
        
        # === Load sharp moves from BigQuery (from Cloud Scheduler)
        detection_key = f"sharp_moves_{sport_key_lower}"
        
        if detection_key in st.session_state:
            df_moves_raw = st.session_state[detection_key]
            st.info(f"✅ Using cached sharp moves for {label}")
        else:
            df_moves_raw = read_recent_sharp_moves(hours=48)
            st.session_state[detection_key] = df_moves_raw
            st.success(f"📥 Loaded sharp moves from BigQuery")
        # === Filter to current tab's sport
        # Normalize and match directly
        df_moves_raw['Sport_Norm'] = df_moves_raw['Sport'].astype(str).str.upper().str.strip()
        df_moves_raw = df_moves_raw[df_moves_raw['Sport_Norm'] == label.upper()]
        
       
        
       
        # ✅ Snapshot log
        #st.write("📦 Total raw rows loaded from BigQuery:", len(df_moves_raw))
        #st.dataframe(df_moves_raw.head(3))
        
        # === Defensive check before build_game_key
        required_cols = ['Game', 'Game_Start', 'Market', 'Outcome']
        missing = [col for col in required_cols if col not in df_moves_raw.columns]
        
        if df_moves_raw.empty:
            st.warning("⚠️ No picks returned — df_moves_raw is empty before build_game_key")
            return pd.DataFrame()
        
        if missing:
            st.error(f"❌ Required columns missing before build_game_key: {missing}")
            st.dataframe(df_moves_raw.head())
            return pd.DataFrame()
        
        # ✅ Deduplicate snapshot duplicates (exact matches except timestamp)
        if 'Snapshot_Timestamp' in df_moves_raw.columns:
            dedup_cols = [col for col in df_moves_raw.columns if col != 'Snapshot_Timestamp']
            before = len(df_moves_raw)
            df_moves_raw = df_moves_raw.sort_values('Snapshot_Timestamp', ascending=False)
            df_moves_raw = df_moves_raw.drop_duplicates(subset=dedup_cols, keep='first')
            after = len(df_moves_raw)
            st.info(f"🧹 Deduplicated {before - after} snapshot rows (kept latest per unique pick).")
        
        # === Build game keys (for merging)
        df_moves_raw = build_game_key(df_moves_raw)
        # === Keep only sharp picks for upcoming games (filter by Game_Start, not Pre_Game)
        #st.info("🕒 Filtering to truly live (upcoming) picks based on Game_Start...")
        
        now = pd.Timestamp.utcnow()
        df_moves_raw['Game_Start'] = pd.to_datetime(df_moves_raw['Game_Start'], errors='coerce', utc=True)
        
        before = len(df_moves_raw)
        df_moves_raw = df_moves_raw[df_moves_raw['Game_Start'] > now]
        after = len(df_moves_raw)
        
        #st.info(f"✅ Game_Start > now: filtered {before} → {after} rows")
        # === Load per-market models from GCS (once per session)
        model_key = f'sharp_models_{label.lower()}'
        # === Load models and team_feature_map from GCS
        market_list = ['spreads', 'totals', 'h2h']
        trained_models = {}
        
        for market in market_list:
            model_bundle = load_model_from_gcs(label, market)
            if model_bundle:
                trained_models[market] = model_bundle
            else:
                st.warning(f"⚠️ No model found for {market.upper()} — skipping.")
        
        # Optional: extract a unified team_feature_map if needed
        team_feature_map = None
        for bundle in trained_models.values():
            tfm = bundle.get('team_feature_map')
            if tfm is not None and not tfm.empty:
                team_feature_map = tfm
                break

      
        
                

        # === Final cleanup
       
        # === 1. Load df_history and compute df_first
        # === Load broader trend history for open line / tier comparison
        start = time.time()
       # Keep all rows for proper line open capture
        df_history_all = get_recent_history()
        merge_keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']

        # === Build First Snapshot: keep *first* rows even if missing model prob
        df_first = (
            df_history_all
            .sort_values('Snapshot_Timestamp')
            .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='first')
            .rename(columns={'Value': 'First_Line_Value'})
            [['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'First_Line_Value']]
        )
        
        df_first_model = (
            df_history_all[df_history_all['Model_Sharp_Win_Prob'].notna()]
            .sort_values('Snapshot_Timestamp')
            .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='first')
            .rename(columns={
                'Model_Sharp_Win_Prob': 'First_Sharp_Prob',
                'Model_Confidence_Tier': 'First_Tier'
            })
            [['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'First_Sharp_Prob', 'First_Tier']]
        )
        
        df_first_full = df_first.merge(
            df_first_model,
            on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'],
            how='left'
        )

        df_moves_raw = df_moves_raw.merge(df_first_full, on=merge_keys, how='left')



                # === Normalize and merge first snapshot into df_moves_raw
    
        alias_map = {
            'Model_Sharp_Win_Prob': 'Model Prob',
            'Model_Confidence_Tier': 'Confidence Tier',
        }
        for orig, alias in alias_map.items():
            if orig in df_moves_raw.columns and alias not in df_moves_raw.columns:
                df_moves_raw[alias] = df_moves_raw[orig]

       
       
        # Alias for clarity in trend logic
        if 'First_Sharp_Prob' in df_moves_raw.columns and 'First_Model_Prob' not in df_moves_raw.columns:
            df_moves_raw['First_Model_Prob'] = df_moves_raw['First_Sharp_Prob']
        
                # === Deduplicate before filtering and diagnostics
        #before = len(df_moves_raw)
        #df_moves_raw = df_moves_raw.drop_duplicates(subset=['Game_Key', 'Market', 'Bookmaker', 'Outcome'], keep='last')
        #after = len(df_moves_raw)
        #st.info(f"🧹 Deduplicated df_moves_raw: {before:,} → {after:,}")
        
        # === Filter upcoming pre-game picks
        now = pd.Timestamp.utcnow()
        
      
        
        # === Step 0: Define keys and snapshot time
        merge_keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
       
        first_cols = ['First_Model_Prob', 'First_Line_Value', 'First_Tier']
        
        # === Step 1: Normalize df_moves_raw BEFORE filtering or extracting
        # === Step 1: Normalize df_moves_raw BEFORE filtering or extracting
        for col in merge_keys:
            df_moves_raw[col] = df_moves_raw[col].astype(str).str.strip().str.lower()
        
        # === Step 2: Build df_first_cols BEFORE slicing
        df_first_cols = df_first_full.copy()
        
        # === Step 3: Filter pre-game picks AFTER normalization
        df_pre = df_moves_raw[
            (df_moves_raw['Pre_Game'] == True) &
            (df_moves_raw['Model_Sharp_Win_Prob'].notna()) &
            (pd.to_datetime(df_moves_raw['Game_Start'], errors='coerce') > now)
        ].copy()
        
        # ✅ Ensure only latest snapshot per bookmaker/outcome is kept
        df_pre = (
            df_pre
            .sort_values('Snapshot_Timestamp')
            .drop_duplicates(subset=merge_keys, keep='last')
        )
        
        # === Step 4: Normalize both sides before merge (again, to be safe)
        for col in merge_keys:
            df_pre[col] = df_pre[col].astype(str).str.strip().str.lower()
            df_first_cols[col] = df_first_cols[col].astype(str).str.strip().str.lower()
        
        # === Step 5: Merge firsts into pre-game picks
        df_pre = df_pre.merge(df_first_cols, on=merge_keys, how='left')
        
        # === Step 6: Resolve _x/_y conflicts
        for col in ['First_Model_Prob', 'First_Line_Value', 'First_Tier']:
            y_col = f"{col}_y"
            x_col = f"{col}_x"
            if y_col in df_pre.columns:
                df_pre[col] = df_pre[y_col]
                df_pre.drop(columns=[y_col], inplace=True)
            if x_col in df_pre.columns:
                df_pre.drop(columns=[x_col], inplace=True)
        
        df_pre.drop(columns=['_merge'], errors='ignore', inplace=True)

        
        # === Optional: Normalize again (safety for downstream groupby)
        df_pre['Bookmaker'] = df_pre['Bookmaker'].str.lower()
        df_pre['Outcome'] = df_pre['Outcome'].astype(str).str.strip().str.lower()
        
        # === Debug/Preview Other Tables
        
        
        # === Step 8: Rename columns for display
        df_pre.rename(columns={
                'Game': 'Matchup',
                'Model_Sharp_Win_Prob': 'Model Prob',
                'Model_Confidence_Tier': 'Confidence Tier'
        }, inplace=True)
        
       

        # === Compute consensus lines
        sharp_consensus = (
            df_pre[df_pre['Bookmaker'].isin(SHARP_BOOKS)]
            .groupby(['Game_Key', 'Market', 'Outcome'])['Value']
            .mean().reset_index(name='Sharp Line')
        )
        rec_consensus = (
            df_pre[df_pre['Bookmaker'].isin(REC_BOOKS)]
            .groupby(['Game_Key', 'Market', 'Outcome'])['Value']
            .mean().reset_index(name='Rec Line')
        )
        
        df_pre = df_pre.merge(sharp_consensus, on=['Game_Key', 'Market', 'Outcome'], how='left')
        df_pre = df_pre.merge(rec_consensus, on=['Game_Key', 'Market', 'Outcome'], how='left')
        # Fill Sharp/Rec Line for missing rows (consensus lines)
        
        # Define summary columns
        summary_cols = [
            'Matchup', 'Market', 'Game_Start', 'Outcome',
            'Rec Line', 'Sharp Line', 'Rec Move', 'Sharp Move',
            'Model Prob', 'Confidence Tier',
            'Confidence Trend', 'Line/Model Direction', 'Tier Δ', 'Why Model Likes It',
            'Game_Key',  # ✅ already there
            'Snapshot_Timestamp'  # ✅ add this line
        ]
        
        # Create df_summary_base
        df_summary_base = df_pre.drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='last')
        # === Use sharp books only to compute final Model Prob per Game × Market × Outcome
        group_cols = ['Game_Key', 'Market', 'Outcome']

        df_sharp = df_summary_base[df_summary_base['Bookmaker'].str.lower().isin(SHARP_BOOKS)]
        model_prob_map = (
            df_sharp
            .groupby(['Game_Key', 'Market', 'Outcome'])['Model Prob']
            .mean()
            .reset_index()
        )

        
        df_summary_base = df_summary_base.drop(columns=['Model Prob'], errors='ignore')
        df_summary_base = df_summary_base.merge(model_prob_map, on=['Game_Key', 'Market', 'Outcome'], how='left')

        df_summary_base['Sharp Line'] = df_summary_base['Sharp Line'].fillna(df_pre['Sharp Line'])
        df_summary_base['Rec Line'] = df_summary_base['Rec Line'].fillna(df_pre['Rec Line'])
        # Ensure required columns exist
        for col in ['Sharp Line', 'Rec Line', 'First_Line_Value']:
            if col not in df_summary_base.columns:
                df_summary_base[col] = np.nan
                     
        # Line movement calculations
        # === Calculate line movement on df_summary_base
        move_start = time.time()
        if 'First_Line_Value' in df_summary_base.columns:
            if 'Sharp Line' in df_summary_base.columns:
                df_summary_base['Sharp Move'] = (df_summary_base['Sharp Line'] - df_summary_base['First_Line_Value']).round(2)
            if 'Rec Line' in df_summary_base.columns:
                df_summary_base['Rec Move'] = (df_summary_base['Rec Line'] - df_summary_base['First_Line_Value']).round(2)
        
        # 🧪 Force numeric types
        for col in ['Sharp Line', 'Rec Line', 'First_Line_Value']:
            if col in df_summary_base.columns:
                df_summary_base[col] = pd.to_numeric(df_summary_base[col], errors='coerce')
        
        # Recalculate Sharp/Rec Move just to be safe
        df_summary_base['Sharp Move'] = (df_summary_base['Sharp Line'] - df_summary_base['First_Line_Value']).round(2)
        df_summary_base['Rec Move'] = (df_summary_base['Rec Line'] - df_summary_base['First_Line_Value']).round(2)
        
        st.info(f"📊 Movement calculations completed in {time.time() - move_start:.2f}s")

        if 'Model_Sharp_Win_Prob' in df_summary_base.columns and 'Model Prob' not in df_summary_base.columns:
            df_summary_base['Model Prob'] = df_summary_base['Model_Sharp_Win_Prob']
        
        if 'Model_Confidence_Tier' in df_summary_base.columns and 'Confidence Tier' not in df_summary_base.columns:
            df_summary_base['Confidence Tier'] = df_summary_base['Model_Confidence_Tier']

        df_summary_base.drop(columns=[col for col in df_summary_base.columns if col.endswith('_x')], inplace=True)
        df_summary_base.columns = [col.replace('_y', '') if col.endswith('_y') else col for col in df_summary_base.columns]
        # Remove true duplicate column names (keep the first occurrence)
        df_summary_base = df_summary_base.loc[:, ~df_summary_base.columns.duplicated()]
        # === Merge team features into df_summary_base
        if team_feature_map is not None and not team_feature_map.empty:
            df_summary_base['Team'] = df_summary_base['Outcome'].str.lower().str.strip()
            df_summary_base = df_summary_base.merge(team_feature_map, on='Team', how='left')

     
       

       

        st.subheader("🧪 Debug: `df_summary_base` Columns + Sample")
        #st.write(f"🔢 Rows: {len(df_summary_base)}")
        st.write("📋 Columns:", df_summary_base.columns.tolist())
        
        #st.dataframe(df_summary_base.head(10))
       
        # === Compute diagnostics from df_pre (upcoming + scored)
        # === Step 1: Use only SHARP_BOOKS for diagnostics
        diag_source = df_summary_base[df_summary_base['Bookmaker'].str.lower().isin(SHARP_BOOKS)].copy()
        
        # === Step 2: Compute diagnostics across all sharp book rows (NO deduplication yet)
        if diag_source.empty:
            st.warning("⚠️ No sharp book rows available for diagnostics.")
            for col in ['Confidence Trend', 'Tier Δ', 'Line/Model Direction', 'Why Model Likes It']:
                df_summary_base[col] = "⚠️ Missing"
        else:
            diagnostics_df = compute_diagnostics_vectorized(diag_source)
        
            # === Step 3: Deduplicate AFTER computing diagnostics
            # Keep the *best* row per outcome — prefer sharp books + latest timestamp
            df_summary_base['Book_Is_Sharp'] = df_summary_base['Bookmaker'].str.lower().isin(SHARP_BOOKS).astype(int)
            df_summary_base = (
                df_summary_base
                .sort_values(['Book_Is_Sharp', 'Snapshot_Timestamp'], ascending=[False, False])
                .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome'], keep='first')
            )
          
                    
            # === Step 4: Merge diagnostics back to deduped summary
            df_summary_base = df_summary_base.merge(
                diagnostics_df,
                on=['Game_Key', 'Market', 'Outcome'],
                how='left'
            )
            st.write("🧪 Columns in filtered_df after diagnostics merge:")
            st.write(df_summary_base.columns.tolist())
                    
            # Fallback fill for missing
            for col in ['Confidence Trend', 'Tier Δ', 'Line/Model Direction', 'Why Model Likes It']:
                df_summary_base[col] = df_summary_base[col].fillna("⚠️ Missing")
        
        #st.markdown("### 🧪 Summary Grouped Debug View")
        
        # Print column list
        #st.code(f"🧩 Columns in summary_after sharp diag:\n{summary_grouped.columns.tolist()}")
        # === 6. Final Summary Table ===

        # Define the core columns we want to extract

        # Define the core columns we want to extract — match these to the renamed names
        
        summary_df = df_summary_base[[col for col in summary_cols if col in df_summary_base.columns]].copy()

        
        # Convert and format datetime columns
        summary_df['Game_Start'] = pd.to_datetime(summary_df['Game_Start'], errors='coerce', utc=True)
        summary_df = summary_df[summary_df['Game_Start'].notna()]
        summary_df['Date + Time (EST)'] = summary_df['Game_Start'].dt.tz_convert('US/Eastern').dt.strftime('%Y-%m-%d %I:%M %p')
        summary_df['Event_Date_Only'] = summary_df['Game_Start'].dt.date.astype(str)
        
        # Clean column suffixes and duplicates if any remain
        summary_df.columns = summary_df.columns.str.replace(r'_x$|_y$|_scored$', '', regex=True)
        summary_df = summary_df.loc[:, ~summary_df.columns.duplicated()]
        
        # === 🔍 Diagnostic: Check for duplicate Game × Market × Outcome
        # === 🔍 Diagnostic: Check for duplicate Game × Market × Outcome
        # === 🔍 Diagnostic: Check for duplicate Game × Market × Outcome
        


        # === Preview & column check
        #st.write("📋 Columns in summary_df:", summary_df.columns.tolist())
        
        # Optional: final sort if needed
        #summary_df.sort_values(by=['Game_Start', 'Matchup', 'Market'], inplace=True)
        
   
        # === Build Market + Date Filters
        market_options = ["All"] + sorted(summary_df['Market'].dropna().unique())
        selected_market = st.selectbox(f"📊 Filter {label} by Market", market_options, key=f"{label}_market_summary")
        
        date_only_options = ["All"] + sorted(summary_df['Event_Date_Only'].dropna().unique())
        selected_date = st.selectbox(f"📅 Filter {label} by Date", date_only_options, key=f"{label}_date_filter")
        
        
        filtered_df = summary_df.copy()

        # ✅ Apply UI filters
        if selected_market != "All":
            filtered_df = filtered_df[filtered_df['Market'] == selected_market]
        if selected_date != "All":
            filtered_df = filtered_df[filtered_df['Event_Date_Only'] == selected_date]
      

        ## Step 1: Normalize keys
        for col in ['Game_Key', 'Market', 'Outcome']:
            filtered_df[col] = filtered_df[col].astype(str).str.strip().str.lower()
        
        # Step 2: Deduplicate filtered_df for last snapshot per outcome
        filtered_df = (
            filtered_df
            .sort_values('Snapshot_Timestamp', ascending=False)
            .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome'], keep='first')
        )
        st.write("🧪 Columns in filtered after diagnostics merge:")
        st.write(filtered_df.columns.tolist())
        # Step 3: Pull diagnostics from earlier
        # Step 3: Pull diagnostics and rename snapshot → Model Prob
        diagnostics_dedup = diagnostics_df.drop_duplicates(
            subset=['Game_Key', 'Market', 'Outcome']
        )[[
            'Game_Key', 'Market', 'Outcome',
            'Confidence Trend', 'Tier Δ', 'Line/Model Direction',
            'Why Model Likes It', 'Model Prob Snapshot', 'Model_Confidence_Tier'
        ]].rename(columns={
            'Model Prob Snapshot': 'Model Prob',
            'Model_Confidence_Tier': 'Confidence Tier'  # ✅ Now this will work
        })


        # ✅ Drop stale version *before* merge to avoid suffixes
        # ✅ Drop stale versions to prevent _x/_y suffixes
        # ✅ Drop all diagnostics to prevent _x/_y suffixes
        diagnostic_cols = [
            'Model Prob', 'Confidence Tier',
            'Confidence Trend', 'Tier Δ', 'Line/Model Direction', 'Why Model Likes It'
        ]
        filtered_df = filtered_df.drop(columns=[col for col in diagnostic_cols if col in filtered_df.columns], errors='ignore')


        
        # Step 4: Merge snapshot version cleanly
        filtered_df = filtered_df.merge(
            diagnostics_dedup,
            on=['Game_Key', 'Market', 'Outcome'],
            how='left'
        )
        
        st.write("🧪 Columns ibefore soummary group:")
        st.write(filtered_df.columns.tolist())
            
        
        # Step 5: Group from merged filtered_df to produce summary
        summary_grouped = (
            filtered_df
            .groupby(['Game_Key', 'Matchup', 'Market', 'Outcome'], as_index=False)
            .agg({
                'Rec Line': 'mean',
                'Sharp Line': 'mean',
                'Rec Move': 'mean',
                'Sharp Move': 'mean',
                'Model Prob': 'mean',
                'Confidence Tier': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
                'Confidence Trend': 'first',
                'Tier Δ': 'first',
                'Line/Model Direction': 'first',
                'Why Model Likes It': 'first'
            })
        )

        #st.markdown("### 🧪 Summary Grouped Debug View")
        
        # Print column list
        #st.code(f"🧩 Columns in summary_grouped:\n{summary_grouped.columns.tolist()}")

        # Step 6: Add back timestamp if available
        if 'Date + Time (EST)' in summary_df.columns:
            summary_grouped = summary_grouped.merge(
                summary_df[['Game_Key', 'Date + Time (EST)']].drop_duplicates(),
                on='Game_Key',
                how='left'
            )
        
    
        required_cols = ['Model Prob', 'Confidence Tier']
                   
        # === Re-merge diagnostics AFTER groupby
       

        
        # ✅ Resolve _y suffixes (only if collision occurred)
        for col in ['Confidence Trend', 'Tier Δ', 'Line/Model Direction', 'Why Model Likes It']:
            if f"{col}_y" in summary_grouped.columns:
                summary_grouped[col] = summary_grouped[f"{col}_y"]
                summary_grouped.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True, errors='ignore')
        
        # Fill empty diagnostics with ⚠️ Missing
        diagnostic_fields = ['Confidence Trend', 'Tier Δ', 'Line/Model Direction', 'Why Model Likes It']
        for col in diagnostic_fields:
            summary_grouped[col] = summary_grouped[col].fillna("⚠️ Missing")

        # === Final Column Order for Display
        view_cols = [
            'Date + Time (EST)', 'Matchup', 'Market', 'Outcome',
            'Rec Line', 'Sharp Line', 'Rec Move', 'Sharp Move',
            'Model Prob', 'Confidence Tier',
            'Why Model Likes It', 'Confidence Trend', 'Tier Δ', 'Line/Model Direction'
        ]
        summary_grouped = summary_grouped.sort_values(
            by=['Date + Time (EST)', 'Matchup', 'Market'],
            ascending=[True, True, True]
        )
        summary_grouped['Model Prob'] = summary_grouped['Model Prob'].apply(lambda x: f"{round(x * 100, 1)}%" if pd.notna(x) else "—")

        summary_grouped = summary_grouped[view_cols]

        
        # === Final Output
        st.subheader(f"📊 Sharp vs Rec Book Summary Table – {label}")
        st.info(f"✅ Summary table shape: {summary_grouped.shape}")
        

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
            text-align: left;
            white-space: normal;
            word-break: break-word;
            max-width: 300px; /* Adjust this to prevent overflow */
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
        

        
        # === Render Sharp Picks Table (HTML Version)
        table_df = summary_grouped[view_cols].copy()
        table_html = table_df.to_html(classes="custom-table", index=False, escape=False)
        st.markdown(f"<div class='scrollable-table-container'>{table_html}</div>", unsafe_allow_html=True)
        st.success("✅ Finished rendering sharp picks table.")
        st.caption(f"Showing {len(table_df)} rows")

    # === 2. Render Live Odds Snapshot Table
    with st.container():  # or a dedicated tab/expander if you want
        st.subheader(f"📊 Live Odds Snapshot – {label} (Odds + Limit)")
    
        # ✅ Only this block will autorefresh
        #st_autorefresh(interval=180 * 1000, key=f"{label}_odds_refresh")  # every 3 minutes
    
        # === Live odds fetch + display logic
        live = fetch_live_odds(sport_key)
        odds_rows = []
   
    
    
        for game in live:
            game_name = f"{game['home_team']} vs {game['away_team']}"
            game_start = pd.to_datetime(game.get("commence_time"), utc=True) if game.get("commence_time") else pd.NaT
        
            for book in game.get("bookmakers", []):
                if book.get("key") not in SHARP_BOOKS + REC_BOOKS:
                    continue
        
                for market in book.get("markets", []):
                    if market.get("key") not in ['h2h', 'spreads', 'totals']:
                        continue
        
                    for outcome in market.get("outcomes", []):
                        price = outcome.get('point') if market['key'] != 'h2h' else outcome.get('price')
                        odds_rows.append({
                            "Game": game_name,
                            "Market": market["key"],
                            "Outcome": outcome["name"],
                            "Bookmaker": book["title"],
                            "Value": price,
                            "Limit": outcome.get("bet_limit", 0),
                            "Game_Start": game_start
                        })
        
        df_odds_raw = pd.DataFrame(odds_rows)
        
        if not df_odds_raw.empty:
            # Combine Value + Limit
            df_odds_raw['Value_Limit'] = df_odds_raw.apply(
                lambda r: f"{round(r['Value'], 1)} ({int(r['Limit'])})" if pd.notnull(r['Limit']) and pd.notnull(r['Value'])
                else "" if pd.isnull(r['Value']) else f"{round(r['Value'], 1)}",
                axis=1
            )
        
            # Localize to EST
            eastern = pytz_timezone('US/Eastern')
            df_odds_raw['Date + Time (EST)'] = df_odds_raw['Game_Start'].apply(
                lambda x: x.tz_convert(eastern).strftime('%Y-%m-%d %I:%M %p') if pd.notnull(x) and x.tzinfo
                else pd.to_datetime(x).tz_localize('UTC').tz_convert(eastern).strftime('%Y-%m-%d %I:%M %p') if pd.notnull(x)
                else ""
            )
        
            # Pivot into Bookmaker columns
            df_display = (
                df_odds_raw.pivot_table(
                    index=["Date + Time (EST)", "Game", "Market", "Outcome"],
                    columns="Bookmaker",
                    values="Value_Limit",
                    aggfunc="first"
                )
                .rename_axis(columns=None)  # Removes the "Bookmaker" column level name
                .reset_index()
            )
        
            # Render as HTML
            table_html_2 = df_display.to_html(classes="custom-table", index=False, escape=False)
            st.markdown(f"<div class='scrollable-table-container'>{table_html_2}</div>", unsafe_allow_html=True)
            st.success(f"✅ Live odds snapshot rendered — {len(df_display)} rows.")
    
def fetch_scores_and_backtest(*args, **kwargs):
    print("⚠️ fetch_scores_and_backtest() is deprecated in UI and will be handled by Cloud Scheduler.")
    return pd.DataFrame()

    
def load_backtested_predictions(sport_label: str, days_back: int = 30) -> pd.DataFrame:
    client = bigquery.Client(location="us")
    query = f"""
        SELECT 
            Game_Key, Bookmaker, Market, Outcome,
            Value,
            Sharp_Move_Signal, Sharp_Limit_Jump, Sharp_Prob_Shift,
            Sharp_Time_Score, Sharp_Limit_Total,
            Is_Reinforced_MultiMarket, Market_Leader, LimitUp_NoMove_Flag,
            SharpBetScore, Enhanced_Sharp_Confidence_Score, True_Sharp_Confidence_Score,
            SHARP_HIT_BOOL, SHARP_COVER_RESULT, Scored, Snapshot_Timestamp, Sport,
            First_Line_Value, First_Sharp_Prob, Line_Delta, Model_Prob_Diff, Direction_Aligned

        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE 
            Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
            AND SHARP_HIT_BOOL IS NOT NULL
            AND Sport = '{sport_label}'
    """
    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"❌ Failed to load predictions: {e}")
        return pd.DataFrame()

def render_sharp_signal_analysis_tab(tab, sport_label, sport_key_api, start_date=None, end_date=None):
    from google.cloud import bigquery
    client = bigquery.Client(project="sharplogger", location="us")

    with tab:
        st.subheader(f"📈 Model Confidence Calibration – {sport_label}")
    
        # === Date Filters UI ===
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date.today() - timedelta(days=14))
        with col2:
            end_date = st.date_input("End Date", value=date.today())
    
        # === Build WHERE clause
        date_filter = ""
        if start_date and end_date:
            date_filter = f"AND DATE(Snapshot_Timestamp) BETWEEN '{start_date}' AND '{end_date}'"


        try:
            df = client.query(f"""
                SELECT *
                FROM `sharplogger.sharp_data.sharp_scores_full`
                WHERE Sport = '{sport_label.upper()}' {date_filter}
            """).to_dataframe()
        except Exception as e:
            st.error(f"❌ Failed to load data: {e}")
            return

        st.info(f"✅ Loaded rows: {len(df)}")

        # === Filter valid rows
        df = df[df['SHARP_HIT_BOOL'].notna() & df['Model_Sharp_Win_Prob'].notna()].copy()
        df['SHARP_HIT_BOOL'] = pd.to_numeric(df['SHARP_HIT_BOOL'], errors='coerce').astype('Int64')
        df['Model_Sharp_Win_Prob'] = pd.to_numeric(df['Model_Sharp_Win_Prob'], errors='coerce')

        # === Bin probabilities
        prob_bins = [0, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ["✅ Coinflip", "⭐ Lean", "🔥 Strong Indication", "🔥 Steam"]

        
        df['Confidence_Bin'] = pd.cut(df['Model_Sharp_Win_Prob'], bins=prob_bins, labels=bin_labels)


        # === Overall Summary
        st.subheader("📊 Model Win Rate by Confidence Bin (Overall)")
        overall = (
            df.groupby('Confidence_Bin')['SHARP_HIT_BOOL']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'Picks', 'mean': 'Win_Rate'})
            .reset_index()
        )
        st.dataframe(overall.style.format({'Win_Rate': '{:.1%}'}))

        # === By Market
        st.markdown("#### 📉 Confidence Calibration by Market")
        conf_summary = (
            df.groupby(['Market', 'Confidence_Bin'])['SHARP_HIT_BOOL']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'Picks', 'mean': 'Win_Rate'})
            .reset_index()
        )

        for market in conf_summary['Market'].dropna().unique():
            st.markdown(f"**📊 {market.upper()}**")
            st.dataframe(
                conf_summary[conf_summary['Market'] == market]
                .drop(columns='Market')
                .style.format({'Win_Rate': '{:.1%}'})
            )


# --- Sidebar navigation
sport = st.sidebar.radio("🏈 Select a League", ["General", "NBA", "MLB", "CFL", "WNBA"])
st.sidebar.markdown("### ⚙️ Controls")
st.sidebar.checkbox("⏸️ Pause Auto Refresh", key="pause_refresh")
# --- Optional: Track scanner checkboxes by sport
scanner_flags = {
    "NBA": "run_nba_scanner",
    "MLB": "run_mlb_scanner",
    "CFL": "run_cfl_scanner",
    "WNBA": "run_wnba_scanner"
}

# === GENERAL PAGE ===
if sport == "General":
    st.title("🎯 Sharp Scanner Dashboard")
    st.write("Use the sidebar to select a league and begin scanning or training models.")

# === LEAGUE PAGES ===
# === LEAGUE PAGES ===
else:
    st.title(f"🏟️ {sport} Sharp Scanner")

    scanner_key = scanner_flags.get(sport)
    run_scanner = st.checkbox(f"Run {sport} Scanner", value=True, key=scanner_key)

    label = sport  # e.g. "WNBA"
    sport_key = SPORTS[sport]  # e.g. "basketball_wnba"

    if st.button(f"📈 Train {sport} Sharp Model"):
        train_sharp_model_from_bq(sport=label)  # label matches BigQuery Sport column

    # Prevent multiple scanners from running
    conflicting = [
        k for k, v in scanner_flags.items()
        if k != sport and st.session_state.get(v, False)
    ]

    if conflicting:
        st.warning(f"⚠️ Please disable other scanners before running {sport}: {conflicting}")
    elif run_scanner:
        scan_tab, analysis_tab = st.tabs(["📡 Live Scanner", "📈 Backtest Analysis"])
        
        with scan_tab:
            render_scanner_tab(label=label, sport_key=sport_key, container=scan_tab)

        with analysis_tab:
            render_sharp_signal_analysis_tab(tab=analysis_tab, sport_label=label, sport_key_api=sport_key)
        
        
