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
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
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



def read_recent_sharp_moves(hours=96, table=BQ_FULL_TABLE):
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

def train_sharp_model_from_bq(sport: str = "NBA", days_back: int = 30):
    st.info(f"🎯 Training sharp model for {sport.upper()}...")

    # ✅ Load from sharp_scores_full with all necessary columns up front
    query = f"""
        SELECT *
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE Sport = '{sport.upper()}'
          AND Scored = TRUE
          AND SHARP_HIT_BOOL IS NOT NULL
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

        df_market['Outcome_Norm'] = df_market['Outcome'].astype(str).str.lower().str.strip()
        
       
     
        # === Canonical side filtering ===
        if market == "totals":
            df_market = df_market[df_market['Outcome_Norm'] == 'over']

        elif market == 'spreads':
            # === 🔍 Debug label correctness for spreads
            if {'Score_Home_Score', 'Score_Away_Score', 'Home_Team'}.issubset(df_market.columns):
                df_market['home_margin'] = df_market['Score_Home_Score'] - df_market['Score_Away_Score']
                df_market['side'] = np.where(df_market['Outcome'] == df_market['Home_Team'], 'home', 'away')
        
                df_market['should_cover'] = np.where(
                    df_market['side'] == 'home',
                    df_market['home_margin'] > -df_market['Value'],
                    -df_market['home_margin'] > -df_market['Value']
                )
        
                df_market['mismatch'] = df_market['SHARP_HIT_BOOL'] != df_market['should_cover'].astype(int)
                num_wrong = df_market['mismatch'].sum()
        
                if num_wrong > 0:
                    st.warning(f"❗ {num_wrong} spread rows have incorrect SHARP_HIT_BOOL values")
                    logging.warning(f"❌ Label mismatch in {num_wrong} SPREAD rows — check spread cover logic")
                    logging.info(df_market[df_market['mismatch']][
                        ['Game_Key', 'Outcome', 'Value', 'Score_Home_Score', 'Score_Away_Score', 'SHARP_HIT_BOOL', 'should_cover']
                    ].head(5).to_string(index=False))
    
        elif market == "h2h":
            df_market['Value'] = pd.to_numeric(df_market['Value'], errors='coerce')
            df_market = df_market[df_market['Value'].notna()]
            df_market['Side_Label'] = np.where(df_market['Value'] < 0, 'favorite', 'underdog')
            df_market = df_market[df_market['Side_Label'] == 'favorite']

        if df_market.empty or df_market['SHARP_HIT_BOOL'].nunique() < 2:
            status.warning(f"⚠️ Not enough label variety for {market.upper()} — skipping.")
            progress.progress(idx / 3)
            continue
        # === Directional agreement (for spreads/h2h invert line logic)
        df_market['Line_Delta'] = pd.to_numeric(df_market['Line_Delta'], errors='coerce')
       
        
        df_market['Direction_Aligned'] = np.where(
            (df_market['Line_Delta'] > 0) & (df_market['Sharp_Limit_Jump'] == 1), 1,
            np.where((df_market['Line_Delta'] < 0) & (df_market['Sharp_Limit_Jump'] == 1), 0, np.nan)
        )
        df_market['Direction_Aligned'] = df_market['Direction_Aligned'].fillna(0).astype(int)
        df_market['Line_Value_Abs'] = df_market['Value'].abs()
        df_market['Prob_Shift_Signed'] = df_market['Sharp_Prob_Shift'] * np.sign(df_market['Value'])
        df_market['Line_Delta_Signed'] = df_market['Line_Delta'] * np.sign(df_market['Value'])
        
        sharp_books = ["pinnacle", "betfair", "circa", "bookmaker"]
        df_market['Book_Norm'] = df_market['Bookmaker'].str.lower().str.strip()
        df_market['Is_Sharp_Book'] = df_market['Book_Norm'].isin(sharp_books).astype(int) 
        df_market['Line_Move_Magnitude'] = df_market['Line_Delta'].abs()
        df_market['Is_Home_Team_Bet'] = (df_market['Outcome'] == df_market['Home_Team_Norm']).astype(int)
        df_market['Is_Favorite_Bet'] = (df_market['Value'] < 0).astype(int)
        df_market['High_Limit_Flag'] = (df_market['Sharp_Limit_Total'] > 10000).astype(int)       
        # === 🧠 Add new features to training
        features = [
            'Sharp_Move_Signal',
            'Sharp_Limit_Jump',
            'Sharp_Time_Score',
            'Sharp_Limit_Total',
            'Is_Reinforced_MultiMarket',
            'Market_Leader',
            'LimitUp_NoMove_Flag',
            'Is_Sharp_Book',
            'Line_Value_Abs',
            'Line_Delta',
            'Direction_Aligned',
            'Line_Move_Magnitude',
            'Is_Home_Team_Bet',
            'Is_Favorite_Bet',
            'High_Limit_Flag'
        ]
        
       
       
        
    
                
        df_market = ensure_columns(df_market, features, 0)
        
        X = df_market[features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        y = df_market['SHARP_HIT_BOOL'].astype(int)
        
        # === Abort early if label has only one class
        if y.nunique() < 2:
            st.warning(f"⚠️ Skipping {market.upper()} — only one label class.")
            progress.progress(idx / 3)
            continue
        
        # === Stratified K-Fold setup
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # === Check each fold for label balance
        bad_folds = []
        for i, (_, val_idx) in enumerate(cv.split(X, y)):
            val_counts = y.iloc[val_idx].value_counts()
            logging.info(f"🧪 Fold {i+1} label distribution: {val_counts.to_dict()}")
            if len(val_counts) < 2:
                bad_folds.append(i+1)
        
        if bad_folds:
            st.warning(f"⚠️ Skipping {market.upper()} — folds with only one class: {bad_folds}")
            logging.warning(f"❌ Cannot train model for {market.upper()} — CV folds are imbalanced.")
            progress.progress(idx / 3)
            continue
        
        # === Param grid (simplified to avoid over-regularization)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'gamma': [0],  # reduced to prevent pruning
            'reg_alpha': [0, 0.01],  # minimal regularization
            'reg_lambda': [1]
        }
        
        grid = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(eval_metric='logloss', tree_method='hist', use_label_encoder=False, n_jobs=-1),
            param_distributions=param_grid,
            scoring='neg_log_loss',
            cv=cv,
            n_iter=20,
            verbose=1,
            random_state=42
        )
        
        # === Train WITHOUT weights (simplified)
        grid.fit(X, y)
        best_model = grid.best_estimator_
        
        # === Calibrate
        calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
        calibrated_model.fit(X, y)
        
        # === Feature importances
        try:
            importances = best_model.feature_importances_
            if len(importances) != len(features):
                st.warning(f"⚠️ Feature mismatch: {len(importances)} vs {len(features)}")
            if len(importances) < len(features):
                dropped = features[len(importances):]
                logging.info(f"⚠️ Dropped features: {dropped}")
            n = min(len(importances), len(features))
            importance_df = pd.DataFrame({
                'Feature': features[:n],
                'Importance': importances[:n]
            }).sort_values(by='Importance', ascending=False)
            st.markdown(f"#### 📊 Feature Importance for `{market.upper()}`")
            st.table(importance_df.head(10))  # safer display
        except Exception as e:
            st.error("❌ Feature importance failed")
            logging.exception(e)
        
        # === Predictions and metrics
        # === Predict raw probabilities
        # === Predict raw probabilities
        raw_probs = calibrated_model.predict_proba(X)[:, 1]
        y_pred = (raw_probs >= 0.5).astype(int)
        
        # === Compute both normal and flipped AUC
        auc = roc_auc_score(y, raw_probs)
        flipped_auc = roc_auc_score(y, 1 - raw_probs)
        
        # === Alert if this market appears inverted
        if flipped_auc > auc + 0.05:
            st.warning(f"⚠️ Model for `{market.upper()}` may be learning inverted signal\nAUC: {auc:.4f} vs Flipped AUC: {flipped_auc:.4f}")
            logging.warning(f"🛑 Flipped AUC alert for {market.upper()}: AUC = {auc:.4f}, Flipped = {flipped_auc:.4f}")
        else:
            st.info(f"✅ AUC for `{market.upper()}`: {auc:.4f}")
        
        # === Continue using original probabilities
        acc = accuracy_score(y, y_pred)
        logloss = log_loss(y, raw_probs)
        brier = brier_score_loss(y, raw_probs)
        
        # === Save model
        save_model_to_gcs(best_model, calibrated_model, sport, market, bucket_name=GCS_BUCKET)
        trained_models[market] = {"model": best_model, "calibrator": calibrated_model}
        
        # === Confusion matrix
        conf = confusion_matrix(y, y_pred)
        conf_labels = pd.DataFrame(conf, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        st.markdown(f"#### 📊 Confusion Matrix — `{market.upper()}`")
        st.dataframe(conf_labels)
        
        st.markdown(f"#### ✅ Class Distribution — `{market.upper()}`")
        st.write(y.value_counts())
        prob_true, prob_pred = calibration_curve(y, raw_probs, n_bins=10)
        calib_df = pd.DataFrame({
            "Predicted Bin Center": prob_pred,
            "Actual Hit Rate": prob_true
        })
        st.markdown(f"#### 🎯 Calibration Bins – {market.upper()}")
        st.dataframe(calib_df)
        
        st.success(f"""✅ Trained + saved model for {market.upper()}
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
    import numpy as np
    import pandas as pd

    TIER_ORDER = {
        '⚠️ Weak Indication': 1,
        '✅ Coinflip': 2,
        '⭐ Lean': 3,
        '🔥 Strong Indication': 4
    }

    try:
        df = df.copy()

        # === Step 1: Normalize Tier Columns
        for col in ['Confidence Tier', 'First_Tier']:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].astype(str).str.strip()

        # === Step 2: Tier Change
        tier_current = df['Confidence Tier'].map(TIER_ORDER).fillna(0).astype(int)
        tier_open = df['First_Tier'].map(TIER_ORDER).fillna(0).astype(int)

        df['Tier_Change'] = np.where(
            df['First_Tier'] != "",
            np.where(
                tier_current > tier_open,
                "↑ " + df['First_Tier'] + " → " + df['Confidence Tier'],
                np.where(
                    tier_current < tier_open,
                    "↓ " + df['First_Tier'] + " → " + df['Confidence Tier'],
                    "↔ No Change"
                )
            ),
            "⚠️ Missing"
        )
        
        # Safe formatting of trend strings
        
        # Safe formatting of trend strings
        prob_now = pd.to_numeric(df.get('Model_Sharp_Win_Prob'), errors='coerce')
        prob_start = pd.to_numeric(df.get('First_Sharp_Prob'), errors='coerce')
        delta = prob_now - prob_start
        
        # Build the trend strings using a vectorized loop
        trend_strs = []
        for s, n in zip(prob_start, prob_now):
            if pd.isna(s) or pd.isna(n):
                trend_strs.append("⚠️ Missing")
            elif n - s >= 0.04:
                trend_strs.append(f"📈 Trending Up: {s:.2%} → {n:.2%}")
            elif n - s <= -0.04:
                trend_strs.append(f"📉 Trending Down: {s:.2%} → {n:.2%}")
            else:
                trend_strs.append(f"↔ Stable: {s:.2%} → {n:.2%}")
        
        df['Confidence Trend'] = trend_strs
        
        # === Step 4: Line/Model Direction Alignment
        # === Step 4: Line/Model Direction Alignment
        df['Line_Delta'] = pd.to_numeric(df.get('Line_Delta'), errors='coerce')

        def get_line_support_sign(row):
            try:
                market = str(row.get('Market', '')).lower()
                outcome = str(row.get('Outcome', '')).lower()
                first_line = pd.to_numeric(row.get('First_Line_Value'), errors='coerce')
                if market == 'totals':
                    return -1 if outcome == 'under' else 1
                else:
                    return -1 if first_line < 0 else 1
            except:
                return 1  # fallback
        df['Line_Support_Sign'] = df.apply(get_line_support_sign, axis=1)
        df['Line_Support_Direction'] = df['Line_Delta'] * df['Line_Support_Sign']

        # Use Model_Prob_Trend instead of deprecated Model_Prob_Diff
        model_prob = pd.to_numeric(df.get('Model_Sharp_Win_Prob'), errors='coerce')
        first_prob = pd.to_numeric(df.get('First_Sharp_Prob'), errors='coerce')
        prob_trend = model_prob - first_prob

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

        # === Step 5: Why Model Likes It
        prob = pd.to_numeric(df.get('Model_Sharp_Win_Prob'), errors='coerce').fillna(0)

        model_reason = np.select(
            [
                prob >= 0.58,
                prob >= 0.52,
                prob <= 0.48,
                prob <= 0.42
            ],
            [
                "🔼 Strong Model Edge",
                "↗️ Slight Model Lean",
                "↘️ Slight Model Fade",
                "🔽 Strong Model Fade"
            ],
            default="🪙 Coinflip"
        )

        reasoning_parts = [pd.Series(model_reason, index=df.index)]

        def append_reason(condition, label):
            mask = pd.Series(condition).fillna(False)
            reasoning_parts.append(mask.map(lambda x: label if x else ""))

        append_reason(df.get('Sharp_Prob_Shift', 0) > 0, "Confidence ↑")
        append_reason(df.get('Sharp_Prob_Shift', 0) < 0, "Confidence ↓")
        append_reason(df.get('Sharp_Limit_Jump', 0), "Limit Jump")
        append_reason(df.get('Market_Leader', 0), "Led Market Move")
        append_reason(df.get('Is_Reinforced_MultiMarket', 0), "Cross-Market Signal")
        append_reason(df.get('LimitUp_NoMove_Flag', 0), "Limit ↑ w/o Price Move")

        try:
            df['Why Model Likes It'] = [
                " | ".join(dict.fromkeys([tag for tag in parts if tag]))  # dedupe and preserve order
                for parts in zip(*reasoning_parts)
            ]
        except Exception:
            df['Why Model Likes It'] = "🪙 Unavailable"

        # === Final output table
        diagnostics_df = df[[
            'Game_Key', 'Market', 'Outcome', 'Bookmaker',
            'Tier_Change', 'Confidence Trend', 'Line/Model Direction', 'Why Model Likes It'
        ]].rename(columns={'Tier_Change': 'Tier Δ'})

        return diagnostics_df

    except Exception as e:
        import streamlit as st
        st.error("❌ Error computing diagnostics")
        st.exception(e)
        return pd.DataFrame(columns=[
            'Game_Key', 'Market', 'Outcome', 'Bookmaker',
            'Tier Δ', 'Confidence Trend', 'Line/Model Direction', 'Why Model Likes It'
        ])
        
        
def apply_blended_sharp_score(df, trained_models):


    #st.markdown("### 🛠️ Running `apply_blended_sharp_score()`")

    df = df.copy()
    df['Market'] = df['Market'].astype(str).str.lower().str.strip()

    try:
        df = df.drop(columns=[col for col in df.columns if col.endswith(('_x', '_y'))], errors='ignore')
        #st.success("🧹 Cleaned up duplicate suffix columns (_x, _y)")
    except Exception as e:
        st.error(f"❌ Cleanup failed: {e}")
        return pd.DataFrame()
    total_start = time.time()
    scored_all = []
    
    for market_type, bundle in trained_models.items():
        try:
            model = bundle.get('model')
            iso = bundle.get('calibrator')
            if model is None or iso is None:
                st.warning(f"⚠️ Skipping {market_type.upper()} — model or calibrator missing")
                continue
    
            # ✅ Must assign this first!
            df_market = df[df['Market'] == market_type].copy()
            if df_market.empty:
                st.warning(f"⚠️ No rows to score for {market_type.upper()}")
                continue
    
            # Normalize fields
            df_market['Outcome'] = df_market['Outcome'].astype(str).str.lower().str.strip()
            df_market['Outcome_Norm'] = df_market['Outcome']
            df_market['Value'] = pd.to_numeric(df_market['Value'], errors='coerce')
               
            # Normalize
            df_market['Outcome'] = df_market['Outcome'].astype(str).str.lower().str.strip()
            df_market['Outcome_Norm'] = df_market['Outcome']
            df_market['Value'] = pd.to_numeric(df_market['Value'], errors='coerce')
            df_market['Commence_Hour'] = pd.to_datetime(df_market['Game_Start'], utc=True, errors='coerce').dt.floor('h')
            
            # ✅ Build a clean Game_Key and Game_Key_Base
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

            # ✅ Two-sided check block (REQUIRED for valid_games to exist)
            sided_games_check = (
                df_market.groupby(['Game_Key_Base'])['Outcome']
                .nunique()
                .reset_index(name='Num_Sides')
            )
            
            valid_games = sided_games_check[sided_games_check['Num_Sides'] >= 2]['Game_Key_Base']

            # Dynamically strip the outcome-specific suffix from Game_Key
            df_market = df_market[df_market['Game_Key_Base'].isin(valid_games)].copy()
            # ✅ NOW apply canonical filtering based on market_type
            if market_type == "spreads":
                
                # ✅ Pick canonical row with most negative value per Game_Key_Base
                df_market = df_market[df_market['Value'].notna()]
                df_canon = df_market[df_market['Value'] < 0].copy()
                df_full_market = df_market.copy()
    
            elif market_type == "h2h":
                
                df_market = df_market[df_market['Value'].notna()]
                df_canon = df_market[df_market['Value'] < 0].copy()
                df_full_market = df_market.copy()
               
    
            elif market_type == "totals":
                df_canon = df_market[df_market['Outcome_Norm'] == 'over'].copy()
                df_full_market = df_market.copy()
    
            else:
                df_canon = df_market.copy()
                df_full_market = df_market.copy()
    
            if df_canon.empty:
                st.warning(f"⚠️ No canonical rows for {market_type.upper()}")
                continue
            # === Deduplicate canonical rows
            dedup_keys = ['Game_Key', 'Market', 'Bookmaker', 'Outcome', 'Snapshot_Timestamp']

            pre_dedup_canon = len(df_canon)
            df_canon = df_canon.drop_duplicates(subset=dedup_keys)
            post_dedup_canon = len(df_canon)
            
            #st.success(f"✅ Canonical rows deduplicated: {pre_dedup_canon:,} → {post_dedup_canon:,}")

            
            # === Ensure required features exist ===
            model_features = model.get_booster().feature_names
            missing_cols = [col for col in model_features if col not in df_canon.columns]
            df_canon[missing_cols] = 0
            # === Feature engineering to match training
            df_canon['Line_Move_Magnitude'] = df_canon['Line_Delta'].abs()
            df_canon['Is_Home_Team_Bet'] = (df_canon['Outcome'] == df_canon['Home_Team_Norm']).astype(int)
            df_canon['Is_Favorite_Bet'] = (df_canon['Value'] < 0).astype(int)
            df_canon['High_Limit_Flag'] = (df_canon['Sharp_Limit_Total'] > 10000).astype(int)
            # === Align features exactly to model input ===
            X_canon = df_canon[model_features].replace({'True': 1, 'False': 0}).apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            
            # === Raw model output (optional)
            df_canon['Model_Sharp_Win_Prob'] = trained_models[market_type]['model'].predict_proba(X_canon)[:, 1]
            
            df_canon['Model_Confidence'] = trained_models[market_type]['calibrator'].predict_proba(X_canon)[:, 1]
            
            # === Tag for downstream usage
            df_canon['Was_Canonical'] = True
            df_canon['Scoring_Market'] = market_type
            df_canon['Scored_By_Model'] = True
            
           
           # === Build Inverse from already-scored canonical rows
            df_inverse = df_canon.copy(deep=True)
            df_inverse['Model_Sharp_Win_Prob'] = 1 - df_inverse['Model_Sharp_Win_Prob']
            df_inverse['Model_Confidence'] = 1 - df_inverse['Model_Confidence']
            df_inverse['Was_Canonical'] = False
            df_inverse['Scored_By_Model'] = True
            
            # Normalize outcomes once
            df_inverse['Outcome'] = df_inverse['Outcome'].astype(str).str.lower().str.strip()
            df_inverse['Outcome_Norm'] = df_inverse['Outcome']
            df_full_market['Outcome'] = df_full_market['Outcome'].astype(str).str.lower().str.strip()
            
            # === Flip outcome depending on market
            if market_type == "totals":
                df_inverse['Outcome'] = df_inverse['Outcome'].map(lambda x: 'under' if x == 'over' else 'over')
            
            elif market_type in ["h2h", "spreads"]:
                df_inverse['Canonical_Team'] = df_inverse['Outcome']  # Already normalized
                df_inverse['Outcome'] = (
                    pd.Series(np.where(
                        df_inverse['Canonical_Team'] == df_inverse['Home_Team_Norm'],
                        df_inverse['Away_Team_Norm'],
                        df_inverse['Home_Team_Norm']
                    ), index=df_inverse.index)
                    .astype(str).str.lower().str.strip()
                )
            
            # Update Outcome_Norm after flipping
            df_inverse['Outcome_Norm'] = df_inverse['Outcome']
            
            # Rebuild Game_Key and Game_Key_Base after flip
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
            
            # Merge in opponent value using Team_Key
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
            
            # === Shared feature engineering for all market types
            df_inverse['Line_Move_Magnitude'] = df_inverse['Line_Delta'].abs()
            df_inverse['Is_Home_Team_Bet'] = (df_inverse['Outcome'] == df_inverse['Home_Team_Norm']).astype(int)
            df_inverse['Is_Favorite_Bet'] = (df_inverse['Value'] < 0).astype(int)
            df_inverse['High_Limit_Flag'] = (df_inverse['Sharp_Limit_Total'] > 10000).astype(int)
            
            # === Final deduplication
            df_inverse = df_inverse.drop_duplicates(subset=['Game_Key', 'Market', 'Bookmaker', 'Outcome', 'Snapshot_Timestamp'])
                        # After generating df_inverse
            if df_inverse.empty:
                st.warning("⚠️ No inverse rows generated — check canonical filtering or flip logic.")
                continue  # optional: skip this scoring loop if inverse fails

           
            # ✅ Combine canonical and inverse into one scored DataFrame
            df_scored = pd.concat([df_canon, df_inverse], ignore_index=True)
            
            # ✅ Assign tier labels safely
            df_scored['Model_Confidence_Tier'] = pd.cut(
                df_scored['Model_Sharp_Win_Prob'],
                bins=[0.0, 0.4, 0.5, 0.6, 1.0],
                labels=["⚠️ Weak Indication", "✅ Coinflip", "⭐ Lean", "🔥 Strong Indication"]
            )
            
            #st.info(f"✅ Canonical: {df_canon.shape[0]} | Inverse: {df_inverse.shape[0]} | Combined: {df_scored.shape[0]}")
            #st.dataframe(df_scored[['Game_Key', 'Outcome', 'Model_Sharp_Win_Prob', 'Model_Confidence', 'Model_Confidence_Tier']].head())
            
            scored_all.append(df_scored)

            

        except Exception as e:
            st.error(f"❌ Failed scoring {market_type.upper()}")
            st.code(traceback.format_exc())

    try:
        if scored_all:
            df_final = pd.concat(scored_all, ignore_index=True)
            df_final = df_final[df_final['Model_Sharp_Win_Prob'].notna()]
            
            return df_final
        else:
            st.warning("⚠️ No market types scored — returning empty DataFrame.")
            return pd.DataFrame()
    except Exception as e:
        st.error("❌ Exception during final aggregation")
        st.code(traceback.format_exc())
        return pd.DataFrame()
        

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
    #market_weights = read_market_weights_from_bigquery()
    #if not market_weights:
        #st.error("❌ No market weights found. Cannot compute confidence scores.")
        #return
    st_autorefresh(interval=180 * 1000, key=f"{label}_scanner_refresh")
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
        SPORT_BQ_MAP = {
            "NBA": "BASKETBALL_NBA",
            "WNBA": "BASKETBALL_WNBA",
            "MLB": "BASEBALL_MLB",
            "CFL": "AMERICANFOOTBALL_CFL"
        }
        
        bq_sport = SPORT_BQ_MAP.get(label.upper())
        
        if 'Sport' in df_moves_raw.columns and bq_sport:
            before = len(df_moves_raw)
            df_moves_raw = df_moves_raw[df_moves_raw['Sport'] == bq_sport]
            after = len(df_moves_raw)
            #st.info(f"🏷️ Filtered to sport = {bq_sport}: {before} → {after} rows")
        else:
            st.warning("⚠️ Could not filter by Sport — missing column or mapping.")
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
        trained_models = st.session_state.get(model_key)
        
        if trained_models is None:
            trained_models = {}
            for market_type in ['spreads', 'totals', 'h2h']:
                model_bundle = load_model_from_gcs(sport=label, market=market_type)
                if model_bundle:
                    trained_models[market_type] = model_bundle
            st.session_state[model_key] = trained_models
        
        # === Apply model scoring if models are loaded
        # === Apply model scoring if models are loaded
        if trained_models:
            try:
                df_pre_game_picks = df_moves_raw.copy()
                merge_keys = ['Game_Key', 'Market', 'Bookmaker', 'Outcome']
        
                # ✅ Score everything
                df_scored = apply_blended_sharp_score(df_pre_game_picks, trained_models)
                #st.write("📋 df_scored.columns BEFORE normalization:", df_scored.columns.tolist())
        
                if df_scored.empty:
                    st.warning("⚠️ No rows successfully scored — possibly model failure or input issues.")
                    st.dataframe(df_pre_game_picks.head(5))
                    return pd.DataFrame()
                for col in merge_keys:
                    if col in df_scored.columns:
                        df_scored[col] = df_scored[col].astype(str).str.strip().str.lower()
                    if col in df_pre_game_picks.columns:
                        df_pre_game_picks[col] = df_pre_game_picks[col].astype(str).str.strip().str.lower()
                #st.write("🧪 Model_Sharp_Win_Prob summary:")
                #st.dataframe(df_scored[['Game_Key', 'Market', 'Outcome', 'Model_Sharp_Win_Prob', 'Model_Confidence']].head())
                
                # Count nulls
                num_scored = df_scored['Model_Sharp_Win_Prob'].notna().sum()
                #st.write(f"✅ Non-null Model_Sharp_Win_Prob rows: {num_scored:,} / {len(df_scored):,}")

                #st.write("✅ Merge keys normalized.")
                #st.write("📋 df_scored head:", df_scored[merge_keys].head())
        
                # ✅ Deduplicate and finalize scored output
                df_scored = df_scored.sort_values('Snapshot_Timestamp', ascending=False)
                df_scored = df_scored.drop_duplicates(subset=merge_keys, keep='first')
        
                # ✅ Ensure all necessary columns exist
                required_score_cols = ['Model_Sharp_Win_Prob', 'Model_Confidence', 'Model_Confidence_Tier', 'Scored_By_Model']
                for col in required_score_cols:
                    if col not in df_scored.columns:
                        df_scored[col] = np.nan
        
                # ✅ Defensive check
                # ✅ Defensive check
                if 'Model_Sharp_Win_Prob' not in df_scored.columns:
                    st.error("❌ Model_Sharp_Win_Prob missing from df_scored before merge!")
                    st.dataframe(df_scored.head())
                    raise ValueError("Model_Sharp_Win_Prob missing — merge will fail.")
                # 🔒 Save Pre_Game for restoration
                pre_game_map = df_moves_raw[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Pre_Game']].drop_duplicates()
                for col in merge_keys:
                    pre_game_map[col] = pre_game_map[col].astype(str).str.strip().str.lower()
                              
                # ✅ Prepare scored data for merge
                merge_columns = merge_keys + required_score_cols + ['Snapshot_Timestamp']
                df_scored = df_scored[merge_columns].copy()
                df_scored['Snapshot_Timestamp'] = pd.to_datetime(df_scored['Snapshot_Timestamp'], errors='coerce', utc=True)
                
                df_scored_clean = df_scored[merge_keys + required_score_cols].copy()
                # Normalize keys on both sides
                for col in merge_keys:
                    df_scored_clean[col] = df_scored_clean[col].astype(str).str.strip().str.lower()
                    df_moves_raw[col] = df_moves_raw[col].astype(str).str.strip().str.lower()
                # Only drop conflicting columns NOT used in merge
                # 🛡️ Merge-safe columns we want to preserve
                protected_cols = merge_keys + ['Pre_Game', 'Post_Game']
                
                # ✅ Only drop non-protected, conflicting columns
                cols_to_drop = [
                    col for col in df_scored_clean.columns
                    if col in df_moves_raw.columns and col not in protected_cols
                ]
                
                df_moves_raw = df_moves_raw.drop(columns=cols_to_drop, errors='ignore')
                #st.info(f"🧹 Dropped {len(cols_to_drop)} conflicting non-key, non-protected columns before merge.")

           
                
                # Step 2: Now do the merge safely
                df_moves_raw = df_moves_raw.merge(
                    df_scored_clean,
                    on=merge_keys,
                    how='left',
                    validate='many_to_one'
                )
                
                # Step 3: Cleanup — now this is safe
                # ✅ Restore Pre_Game from saved map
                df_moves_raw = df_moves_raw.merge(
                    pre_game_map,
                    on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'],
                    how='left',
                    suffixes=('', '_pre_game')
                )
                
                # Defensive cleanup in case of name collision
                if 'Pre_Game_x' in df_moves_raw.columns and 'Pre_Game_y' in df_moves_raw.columns:
                    df_moves_raw['Pre_Game'] = df_moves_raw['Pre_Game_x'].combine_first(df_moves_raw['Pre_Game_y'])
                    df_moves_raw.drop(columns=['Pre_Game_x', 'Pre_Game_y'], inplace=True)
                elif 'Pre_Game_x' in df_moves_raw.columns:
                    df_moves_raw.rename(columns={'Pre_Game_x': 'Pre_Game'}, inplace=True)
                elif 'Pre_Game_y' in df_moves_raw.columns:
                    df_moves_raw.rename(columns={'Pre_Game_y': 'Pre_Game'}, inplace=True)
                restored = df_moves_raw['Pre_Game'].notna().sum()
                total = len(df_moves_raw)
                #st.info(f"🧠 Pre_Game restored: {restored:,} / {total:,} rows have non-null values")

                
                df_moves_raw.columns = df_moves_raw.columns.str.replace(r'_x$|_y$|_scored$', '', regex=True)
                df_moves_raw = df_moves_raw.loc[:, ~df_moves_raw.columns.duplicated()]

                # Sample mismatch debugging
                merged_keys = df_scored_clean[merge_keys].drop_duplicates()
              
                raw_keys = df_moves_raw[merge_keys].drop_duplicates()
                
               
                merge_check = merged_keys.merge(
                    raw_keys,
                    on=merge_keys,
                    how='outer',
                    indicator=True
                )
              

                # ✅ Clean suffixes
                df_moves_raw.columns = df_moves_raw.columns.str.replace(r'_x$|_y$|_scored$', '', regex=True)
                df_moves_raw = df_moves_raw.loc[:, ~df_moves_raw.columns.duplicated()]
                
                # ✅ Final check
                if 'Model_Sharp_Win_Prob' not in df_moves_raw.columns:
                    st.error("❌ Post-merge: Model_Sharp_Win_Prob missing entirely from df_moves_raw!")
                else:
                    pass#st.success("✅ All rows successfully scored.")

        
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                full_trace = traceback.format_exc()
        
                st.error(f"❌ Model scoring failed — {error_type}: {error_msg}")
                st.code(full_trace, language='python')
                st.warning("📛 Check the traceback above for where the failure occurred.")
        else:
            st.warning("⚠️ No trained models available for scoring.")
        
                

        # === Final cleanup
       
        # === 1. Load df_history and compute df_first
        # === Load broader trend history for open line / tier comparison
        start = time.time()
        df_history = get_recent_history()
        
                
        hist_start = time.time()

        # === Filter to only current live games
        df_history = df_history[df_history['Game_Key'].isin(df_moves_raw['Game_Key'])]
        df_history = df_history[df_history['Model_Sharp_Win_Prob'].notna()]
        
        # === Build First Snapshot
        # Build First Snapshot (from historical values)
        df_first = (
            df_history.sort_values('Snapshot_Timestamp')
            .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='first')
            .rename(columns={
                'Value': 'First_Line_Value',
                'Model_Confidence_Tier': 'First_Tier',
                'Model_Sharp_Win_Prob': 'First_Sharp_Prob'
            })[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'First_Line_Value', 'First_Tier', 'First_Sharp_Prob']]
        )
        

                # === Normalize and merge first snapshot into df_moves_raw
        df_first['Bookmaker'] = df_first['Bookmaker'].astype(str).str.strip().str.lower()
        df_moves_raw['Bookmaker'] = df_moves_raw['Bookmaker'].astype(str).str.strip().str.lower()
        
        df_moves_raw = df_moves_raw.merge(
            df_first, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left'
        )
       
        # Alias for clarity in trend logic
        if 'First_Sharp_Prob' in df_moves_raw.columns and 'First_Model_Prob' not in df_moves_raw.columns:
            df_moves_raw['First_Model_Prob'] = df_moves_raw['First_Sharp_Prob']
        
                # === Deduplicate before filtering and diagnostics
        before = len(df_moves_raw)
        df_moves_raw = df_moves_raw.drop_duplicates(subset=['Game_Key', 'Market', 'Bookmaker', 'Outcome'], keep='last')
        after = len(df_moves_raw)
        #st.info(f"🧹 Deduplicated df_moves_raw: {before:,} → {after:,}")
        
        # === Filter upcoming pre-game picks
        now = pd.Timestamp.utcnow()
        
        # === Now apply actual filter
        df_pre = df_moves_raw[
            (df_moves_raw['Pre_Game'] == True) &
            (df_moves_raw['Model_Sharp_Win_Prob'].notna()) &
            (pd.to_datetime(df_moves_raw['Game_Start'], errors='coerce') > now)
        ].copy()

        # === Step 0: Define keys and snapshot time
        merge_keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
        now = pd.Timestamp.utcnow()
        first_cols = ['First_Model_Prob', 'First_Line_Value', 'First_Tier']
        
        # === Step 1: Normalize df_moves_raw BEFORE filtering or extracting
        for col in merge_keys:
            df_moves_raw[col] = df_moves_raw[col].astype(str).str.strip().str.lower()
        
        # === Step 2: Build df_first_cols BEFORE slicing
        df_first_cols = df_moves_raw[merge_keys + first_cols].drop_duplicates()
        
        # === Step 3: Filter pre-game picks AFTER normalization
        df_pre = df_moves_raw[
            (df_moves_raw['Pre_Game'] == True) &
            (df_moves_raw['Model_Sharp_Win_Prob'].notna()) &
            (pd.to_datetime(df_moves_raw['Game_Start'], errors='coerce') > now)
        ].copy()
        
        # === Step 4: Normalize df_pre again (redundant but safe)
        merge_keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']

        # === Normalize both sides (ensure string equality)
        for col in merge_keys:
            df_pre[col] = df_pre[col].astype(str).str.strip().str.lower()
            df_first_cols[col] = df_first_cols[col].astype(str).str.strip().str.lower()
        
        # === Ensure expected columns exist before merge
        for col in ['First_Model_Prob', 'First_Line_Value', 'First_Tier']:
            if col not in df_pre.columns:
                df_pre[col] = None
        
        # === Check pre-merge matches vs misses
        merge_debug = df_pre[merge_keys].drop_duplicates().merge(
            df_first_cols[merge_keys].drop_duplicates(),
            on=merge_keys,
            how='left',
            indicator=True
        )
        
        # === Show unmatched keys
        
        pre_keys = df_pre[merge_keys].drop_duplicates()
        first_keys = df_first_cols[merge_keys].drop_duplicates()
        # === Optional: show matching rows as well
        
        # Safe merge
        df_pre = df_pre.merge(df_first_cols, on=merge_keys, how='left', indicator=True)
        
        for col in ['First_Model_Prob', 'First_Line_Value', 'First_Tier']:
            y_col = f"{col}_y"
            x_col = f"{col}_x"
        
            if y_col in df_pre.columns:
                df_pre[col] = df_pre[y_col]  # take the good values
                df_pre.drop(columns=[y_col], inplace=True)
        
            if x_col in df_pre.columns:
                df_pre.drop(columns=[x_col], inplace=True)
        
        df_pre.drop(columns=['_merge'], inplace=True)
        # === Defensive patch for missing First_Tier ===
        if 'First_Tier' not in df_pre.columns or df_pre['First_Tier'].isnull().all():
            st.warning("⚠️ First_Tier is missing or null — filling with fallback to prevent Tier Δ defaulting to Steam.")
            df_pre['First_Tier'] = "✅ Coinflip"  # or your true baseline tier
        
        # === Step 6: Confirm merge success
        
        
        # === Step 7: Deduplicate post-merge
        df_pre = df_pre.drop_duplicates(subset=merge_keys, keep='last')
        
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


        #st.subheader("🧪 Debug: `df_summary_base` Columns + Sample")
        #.write(f"🔢 Rows: {len(df_summary_base)}")
        #st.write("📋 Columns:", df_summary_base.columns.tolist())
        #st.dataframe(df_summary_base.head(10))

        # === Compute diagnostics from df_pre (upcoming + scored)
        if df_summary_base.empty:
            st.warning("⚠️ No valid *upcoming* scored picks for diagnostics.")
            for col in ['Confidence Trend', 'Tier Δ', 'Line/Model Direction', 'Why Model Likes It']:
                df_moves_raw[col] = "⚠️ Missing"
        else:
            diagnostics_df = compute_diagnostics_vectorized(df_summary_base)
            diag_keys = ['Game_Key', 'Market', 'Outcome']
        
            df_moves_raw = df_moves_raw.merge(
                diagnostics_df,
                on=diag_keys,
                how='left',
                suffixes=('', '_diagnostics')
            )
        
            # Cleanly attach diagnostic columns with fallback
            diagnostic_cols = {
                'Confidence Trend': 'Confidence Trend_diagnostics',
                'Why Model Likes It': 'Why Model Likes It_diagnostics',
                'Tier Δ': 'Tier Δ_diagnostics',
                'Line/Model Direction': 'Line/Model Direction_diagnostics',
            }
            for final_col, diag_col in diagnostic_cols.items():
                if diag_col in df_moves_raw.columns:
                    df_moves_raw[final_col] = df_moves_raw[diag_col].fillna("⚠️ Missing")
                else:
                    df_moves_raw[final_col] = "⚠️ Missing"

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
      

        # ✅ Normalize keys
        for col in ['Game_Key', 'Market', 'Outcome']:
            filtered_df[col] = filtered_df[col].astype(str).str.strip().str.lower()
        
               
        filtered_df = (
            filtered_df
            .sort_values('Snapshot_Timestamp', ascending=False)
            .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome'], keep='first')
        )

        # === Group numeric + categorical fields ONLY
        summary_grouped = (
            filtered_df
            .groupby(['Game_Key', 'Matchup', 'Market', 'Outcome'], as_index=False)
            .agg({
                'Rec Line': 'mean',
                'Sharp Line': 'mean',
                'Rec Move': 'mean',
                'Sharp Move': 'mean',
                'Model Prob': 'mean',
                'Confidence Tier': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
            })
        )
        if 'Date + Time (EST)' in summary_df.columns:
            summary_grouped = summary_grouped.merge(
                summary_df[['Game_Key', 'Date + Time (EST)']].drop_duplicates(),
                on='Game_Key',
                how='left'
            )
        else:
            st.warning("⚠️ 'Date + Time (EST)' not found in summary_df — timestamp merge skipped.")
        
        
        
        required_cols = ['Model Prob', 'Confidence Tier']
        for col in required_cols:
            if col not in summary_grouped.columns:
                summary_grouped[col] = np.nan if 'Prob' in col else ""
        if 'Model_Sharp_Win_Prob' in filtered_df.columns:
            filtered_df['Model Prob'] = filtered_df['Model_Sharp_Win_Prob']
        else:          
            filtered_df['Model Prob'] = 0
        if 'Model_Confidence_Tier' in filtered_df.columns:
            filtered_df['Confidence Tier'] = filtered_df['Model_Confidence_Tier']
        else:
            filtered_df['Confidence Tier'] = ""

            
        # === Re-merge diagnostics AFTER groupby
        diagnostic_cols = ['Game_Key', 'Market', 'Outcome',
                           'Confidence Trend', 'Tier Δ', 'Line/Model Direction', 'Why Model Likes It']
        
        # ✅ SAFEGUARD: assign diagnostics_df if missing
        if 'diagnostics_df' not in locals():
            st.warning("⚠️ diagnostics_df missing — assigning empty DataFrame fallback")
            diagnostics_df = pd.DataFrame(columns=diagnostic_cols)
        
        # === Re-merge diagnostics AFTER groupby using clean diagnostics_df
        diagnostics_df_clean = diagnostics_df.drop_duplicates(subset=['Game_Key', 'Market', 'Outcome'])
        
        # ✅ Normalize merge keys on both sides
        for col in ['Game_Key', 'Market', 'Outcome']:
            summary_grouped[col] = summary_grouped[col].astype(str).str.strip().str.lower()
            diagnostics_df_clean[col] = diagnostics_df_clean[col].astype(str).str.strip().str.lower()
        
        
        # 🧪 Add these diagnostics RIGHT HERE before merging
        #st.write("🧪 Unique Game_Keys in summary:", summary_grouped['Game_Key'].unique()[:5])
        #st.write("🧪 Unique Game_Keys in diagnostics:", diagnostics_df_clean['Game_Key'].unique()[:5])
        merge_keys = ['Game_Key', 'Market', 'Outcome']
        
        merged_check = summary_grouped[merge_keys].merge(
            diagnostics_df_clean[merge_keys],
            on=merge_keys,
            how='outer',
            indicator=True
        )
        
        only_in_summary = merged_check[merged_check['_merge'] == 'left_only']
        only_in_diagnostics = merged_check[merged_check['_merge'] == 'right_only']
        
        #st.warning(f"🚫 {len(only_in_summary)} keys in summary not matched in diagnostics")
        #st.warning(f"🚫 {len(only_in_diagnostics)} keys in diagnostics not matched in summary")
        
        if not only_in_summary.empty:
            st.dataframe(only_in_summary.head())
        
        # === Merge diagnostics back into grouped summary
        diagnostic_fields = ['Game_Key', 'Market', 'Outcome', 'Confidence Trend', 'Tier Δ', 'Line/Model Direction', 'Why Model Likes It']
        summary_grouped = summary_grouped.merge(
            diagnostics_df_clean[diagnostic_fields],
            on=['Game_Key', 'Market', 'Outcome'],
            how='left'
        )

        
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

def render_sharp_signal_analysis_tab(tab, sport_label, sport_key_api):
    from google.cloud import bigquery
    client = bigquery.Client(project="sharplogger", location="us")

    # Map for sharp_moves_master only
    sport_label_upper = {
        "MLB": "BASEBALL_MLB",
        "NBA": "BASKETBALL_NBA",
        "WNBA": "BASKETBALL_WNBA",
        "CFL": "AMERICANFOOTBALL_CFL"
    }.get(sport_label.upper(), sport_label.upper())

    with tab:
        st.subheader(f"📈 Model Confidence Calibration – {sport_label}")

        try:
            # ✅ sharp_moves_master uses verbose sport label
            df_master = client.query(f"""
                SELECT * FROM `sharplogger.sharp_data.sharp_moves_master`
                WHERE Sport = '{sport_label_upper}'
            """).to_dataframe()

            # ✅ sharp_scores_full uses short sport label like 'MLB'
            df_scores = client.query(f"""
                SELECT * FROM `sharplogger.sharp_data.sharp_scores_full`
                WHERE Sport = '{sport_label.upper()}'
            """).to_dataframe()

        except Exception as e:
            st.error(f"❌ Failed to load BigQuery data: {e}")
            return

        st.info(f"✅ Loaded Sport Filter: sport_label = '{sport_label}', sport_label_upper = '{sport_label_upper}'")
        st.info(f"📦 df_master rows: {len(df_master)}")
        st.info(f"📦 df_scores rows: {len(df_scores)}")
        
        if df_master.empty:
            st.error("❌ df_master is EMPTY!")
        if df_scores.empty:
            st.error("❌ df_scores is EMPTY!")

        if df_master.empty or df_scores.empty:
            st.warning(f"⚠️ No sharp picks found for {sport_label}")
            return

        merge_keys = ['Game_Key', 'Bookmaker', 'Market', 'Outcome']

        # === Normalize keys
        for df_ in [df_master, df_scores]:
            for col in merge_keys:
                if df_[col].dtype == "object":
                    df_[col] = df_[col].str.strip().str.lower()

        # === Filter to valid scored rows
        df_scores_filtered = (
            df_scores[df_scores['SHARP_HIT_BOOL'].notna()]
            [['Game_Key', 'Bookmaker', 'Market', 'Outcome', 'SHARP_HIT_BOOL']]
            .drop_duplicates(subset=merge_keys)
        )
       # Drop SHARP_HIT_BOOL from master to avoid suffixes
        df_master = df_master.drop(columns=['SHARP_HIT_BOOL'], errors='ignore')
        
        # === Filter to valid scored rows
        df_scores_filtered = (
            df_scores[df_scores['SHARP_HIT_BOOL'].notna()]
            [['Game_Key', 'Bookmaker', 'Market', 'Outcome', 'SHARP_HIT_BOOL']]
            .drop_duplicates(subset=merge_keys)
        )
        df_master = df_master.drop_duplicates(subset=merge_keys)

        merge_keys = ['Game_Key', 'Bookmaker', 'Market', 'Outcome']

        st.info(f"🔍 df_master shape: {df_master.shape}")
        st.info(f"🔍 df_scores shape: {df_scores.shape}")
        
        st.info("🔍 Checking unique keys in each DataFrame...")
        st.write("df_master keys:", df_master[merge_keys].drop_duplicates().head())
        st.write("df_scores keys:", df_scores[merge_keys].drop_duplicates().head())
        df_master_keys = df_master[merge_keys].drop_duplicates()
        df_scores_keys = df_scores_filtered[merge_keys].drop_duplicates()
        
        merge_check = df_master_keys.merge(
            df_scores_keys,
            on=merge_keys,
            how='outer',
            indicator=True
        )
        
        only_in_master = merge_check[merge_check['_merge'] == 'left_only']
        only_in_scores = merge_check[merge_check['_merge'] == 'right_only']
        
        st.warning(f"🔍 Unmatched keys — only in sharp_moves_master: {len(only_in_master)}")
        st.warning(f"🔍 Unmatched keys — only in sharp_scores_full: {len(only_in_scores)}")
        
        if not only_in_master.empty:
            st.dataframe(only_in_master.head(5))
        if not only_in_scores.empty:
            st.dataframe(only_in_scores.head(5))
        # === Merge
        df = df_master.merge(df_scores_filtered, on=merge_keys, how='inner')
        
        # Defensive check
        if 'SHARP_HIT_BOOL' not in df.columns:
            st.error("❌ SHARP_HIT_BOOL missing in merged data.")
            return
        
        if df.empty:
            st.error("❌ No matched rows between sharp moves and scores.")
            return
        
        # === Required columns
        required_cols = ['Model_Confidence', 'SHARP_HIT_BOOL']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"❌ Missing required column: {col}")
                return


        df['Model_Confidence'] = pd.to_numeric(df['Model_Confidence'], errors='coerce')
        df['SHARP_HIT_BOOL'] = pd.to_numeric(df['SHARP_HIT_BOOL'], errors='coerce')
        df = df[df['SHARP_HIT_BOOL'].notna() & df['Model_Confidence'].notna()]

        # === Bin confidence
        prob_bins = np.linspace(0, 1, 11)
        bin_labels = [f"{int(p*100)}–{int(prob_bins[i+1]*100)}%" for i, p in enumerate(prob_bins[:-1])]
        df['Confidence_Bin'] = pd.cut(df['Model_Confidence'], bins=prob_bins, labels=bin_labels)

        # === Summary by Market + Bin
        conf_summary = (
            df.groupby(['Market', 'Confidence_Bin'])['SHARP_HIT_BOOL']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'Picks', 'mean': 'Win_Rate'})
            .reset_index()
        )

        # === Overall summary
        st.subheader("📊 Model Win Rate by Confidence Bin (Overall)")
        overall = (
            df.groupby('Confidence_Bin')['SHARP_HIT_BOOL']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'Picks', 'mean': 'Win_Rate'})
            .reset_index()
        )
        st.dataframe(overall.style.format({'Win_Rate': '{:.1%}'}))

        st.markdown("#### 📉 Confidence Calibration by Market")
        for market in conf_summary['Market'].dropna().unique():
            st.markdown(f"**📊 {market.upper()}**")
            st.dataframe(
                conf_summary[conf_summary['Market'] == market]
                .drop(columns='Market')
                .style.format({'Win_Rate': '{:.1%}'})
            )


            
import streamlit as st

# --- Sidebar navigation
sport = st.sidebar.radio("🏈 Select a League", ["General", "NBA", "MLB", "CFL", "WNBA"])

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
else:
    st.title(f"🏟️ {sport} Sharp Scanner")

    scanner_key = scanner_flags.get(sport)
    run_scanner = st.checkbox(f"Run {sport} Scanner", value=True, key=scanner_key)

    if st.button(f"📈 Train {sport} Sharp Model"):
        train_sharp_model_from_bq(sport=sport)

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
            df_live = render_scanner_tab(sport, SPORTS[sport], scan_tab)
        with analysis_tab:
            render_sharp_signal_analysis_tab(analysis_tab, sport, SPORTS[sport])
