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

    df_bt = load_backtested_predictions(sport, days_back)
    if df_bt.empty:
        st.warning("⚠️ No historical sharp picks available to train model.")
        return

    df_bt = df_bt.copy()
    df_bt['SHARP_HIT_BOOL'] = pd.to_numeric(df_bt['SHARP_HIT_BOOL'], errors='coerce')
    # Smart deduplication before training
    # Smart deduplication: Keep unique signals (not just by Game/Market/Bookmaker)
    dedup_cols = [
        'Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Value',
        'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
        'Sharp_Time_Score', 'Sharp_Limit_Total',
        'Is_Reinforced_MultiMarket', 'Market_Leader', 'LimitUp_NoMove_Flag'
    ]
    
    before = len(df_bt)
    df_bt = df_bt.drop_duplicates(subset=dedup_cols, keep='last')
    after = len(df_bt)
    st.info(f"🧹 Deduplicated exact signal rows: {before} → {after}")
    trained_models = {}

    for market in ['spreads', 'totals', 'h2h']:
        df_market = df_bt[df_bt['Market'] == market].copy()
        if df_market.empty:
            continue

        df_market['Outcome_Norm'] = df_market['Outcome'].astype(str).str.lower().str.strip()

        # === Canonical side filtering ===
        if market == "totals":
            df_market = df_market[df_market['Outcome_Norm'] == 'over']
            st.info(f"🧪 {sport.upper()} TOTALS rows after 'over' filter: {df_market.shape[0]}")
            st.info(f"🧪 Class distribution: {df_market['SHARP_HIT_BOOL'].value_counts().to_dict()}")
        
        elif market == "spreads":
            df_market = df_market[df_market['Value'].notna()]
            df_market['Side_Label'] = np.where(df_market['Value'] < 0, 'favorite', 'underdog')
            df_market = df_market[df_market['Side_Label'] == 'favorite']
            st.info(f"🧪 {sport.upper()} SPREADS favorite-side rows: {df_market.shape[0]}")
            st.info(f"🧪 Class distribution: {df_market['SHARP_HIT_BOOL'].value_counts().to_dict()}")
        
        elif market == "h2h":
            # Ensure Value is numeric and not null
            df_market = df_market.copy()
            df_market['Value'] = pd.to_numeric(df_market['Value'], errors='coerce')
            df_market = df_market[df_market['Value'].notna()]  # drop rows with missing odds
        
            # Canonical H2H: favorite = team with more negative moneyline
            df_market['Side_Label'] = np.where(df_market['Value'] < 0, 'favorite', 'underdog')
            df_market = df_market[df_market['Side_Label'] == 'favorite']
        
            # Make SHARP_HIT_BOOL safe for counting
            if 'SHARP_HIT_BOOL' in df_market.columns:
                df_market['SHARP_HIT_BOOL'] = pd.to_numeric(df_market['SHARP_HIT_BOOL'], errors='coerce')
        
            st.info(f"🧪 {sport.upper()} H2H favorite-side rows: {df_market.shape[0]}")
            class_dist = df_market['SHARP_HIT_BOOL'].value_counts(dropna=True).to_dict()
            st.info(f"🧪 Class distribution: {class_dist}")
                
            

            
            
        if df_market.empty or df_market['SHARP_HIT_BOOL'].nunique() < 2:
            st.warning(f"⚠️ Not enough data to train {market.upper()} model.")
            continue

        features = [
            'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
            'Sharp_Time_Score', 'Sharp_Limit_Total',
            'Is_Reinforced_MultiMarket', 'Market_Leader', 'LimitUp_NoMove_Flag'
        ]
        df_market = ensure_columns(df_market, features, 0)

        X = df_market[features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        y = df_market['SHARP_HIT_BOOL'].astype(int)

        model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, eval_metric='logloss')
        model.fit(X, y)

        iso = IsotonicRegression(out_of_bounds='clip')
        raw_probs = model.predict_proba(X)[:, 1]
        iso.fit(raw_probs, y)

        # === Evaluation Metrics ===
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, log_loss, brier_score_loss
        )
        y_pred = (raw_probs >= 0.5).astype(int)

        auc = roc_auc_score(y, raw_probs)
        acc = accuracy_score(y, y_pred)
        logloss = log_loss(y, raw_probs)
        brier = brier_score_loss(y, raw_probs)

        # === Save model ===
        save_model_to_gcs(model, iso, sport, market, bucket_name=GCS_BUCKET)
        trained_models[market] = {"model": model, "calibrator": iso}

        st.success(f"""✅ Trained + saved model for {market.upper()}
- AUC: {auc:.4f}
- Accuracy: {acc:.4f}
- Log Loss: {logloss:.4f}
- Brier Score: {brier:.4f}
""")

    if not trained_models:
        st.error("❌ No models trained.")



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
    TIER_ORDER = {
        '⚠️ Weak Indication': 1,
        '✅ Coinflip': 2,
        '⭐ Lean': 3,
        '🔥 Strong Indication': 4
    }

    try:
        # === Step 1: Strip Tier Columns
        for col in ['Confidence Tier', 'First_Tier']:
            df[col] = df[col].astype(str).str.strip()

        # === Step 2: Tier Change
        tier_current = df['Confidence Tier'].map(TIER_ORDER).fillna(0).astype(int)
        tier_open = df['First_Tier'].map(TIER_ORDER).fillna(0).astype(int)
        tier_change = np.where(
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
        df['Tier_Change'] = tier_change

        # === Step 3: Confidence Trend
        if 'Model Prob' in df.columns and 'First_Model_Prob' in df.columns:
            prob_now = pd.to_numeric(df['Model Prob'], errors='coerce')
            prob_start = pd.to_numeric(df['First_Model_Prob'], errors='coerce')
            delta = prob_now - prob_start

            confidence_trend = np.where(
                prob_start.isna() | prob_now.isna(),
                "⚠️ Missing",
                np.where(
                    delta >= 0.04,
                    ["📈 Trending Up: {:.2%} → {:.2%}".format(s, n) for s, n in zip(prob_start, prob_now)],
                    np.where(
                        delta <= -0.04,
                        ["📉 Trending Down: {:.2%} → {:.2%}".format(s, n) for s, n in zip(prob_start, prob_now)],
                        ["↔ Stable: {:.2%} → {:.2%}".format(s, n) for s, n in zip(prob_start, prob_now)]
                    )
                )
            )

            # === Step 4: Line/Model Direction
            line_delta = pd.to_numeric(df.get('Value'), errors='coerce') - pd.to_numeric(df.get('First_Line_Value'), errors='coerce')
            direction = np.where(
                (delta > 0.04) & (line_delta < 0), "🟢 Model ↑ / Line ↓",
                np.where(
                    (delta < -0.04) & (line_delta > 0), "🔴 Model ↓ / Line ↑",
                    np.where(
                        (delta > 0.04) & (line_delta > 0), "🟢 Aligned ↑",
                        np.where(
                            (delta < -0.04) & (line_delta < 0), "🔻 Aligned ↓",
                            "⚪ Mixed"
                        )
                    )
                )
            )
        else:
            confidence_trend = ["⚠️ Missing"] * len(df)
            direction = ["⚠️ Missing"] * len(df)
            st.warning("⚠️ Missing probability columns for trend/direction.")

        # === Step 5: Why Model Likes It
        prob = (
            pd.to_numeric(df.get('Model_Sharp_Win_Prob'), errors='coerce')
            if 'Model_Sharp_Win_Prob' in df.columns
            else pd.to_numeric(df.get('Model Prob'), errors='coerce').fillna(0)
        )

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
            model_reasoning = pd.Series([
                " | ".join(dict.fromkeys([tag for tag in parts if tag]))  # Dedup and preserve order
                for parts in zip(*reasoning_parts)
            ], index=df.index)
        except Exception as e:
            st.warning("⚠️ Failed to compute model_reasoning — using fallback.")
            st.exception(e)
            model_reasoning = pd.Series(["🪙 Unavailable"] * len(df), index=df.index)

        # === Final Output
        diagnostics_df = pd.DataFrame({
            'Game_Key': df['Game_Key'],
            'Market': df['Market'],
            'Outcome': df['Outcome'],
            'Bookmaker': df['Bookmaker'],
            'Tier Δ': df['Tier_Change'],
            'Confidence Trend': confidence_trend,
            'Line/Model Direction': direction,
            'Why Model Likes It': model_reasoning
        })

        st.info(f"✅ Diagnostics computed for {len(diagnostics_df)} rows.")
        return diagnostics_df

    except Exception as e:
        st.error("❌ Error computing diagnostics")
        st.exception(e)
        return None


def apply_blended_sharp_score(df, trained_models):


    st.markdown("### 🛠️ Running `apply_blended_sharp_score()`")

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

            model_features = model.get_booster().feature_names
            missing_cols = [col for col in model_features if col not in df_canon.columns]
            df_canon[missing_cols] = 0

            X = df_canon[model_features].replace({'True': 1, 'False': 0}).apply(pd.to_numeric, errors='coerce').fillna(0)
            df_canon['Model_Sharp_Win_Prob'] = model.predict_proba(X)[:, 1]
            df_canon['Model_Confidence'] = iso.predict(df_canon['Model_Sharp_Win_Prob'])
            df_canon['Was_Canonical'] = True
            df_canon['Scoring_Market'] = market_type
            df_canon['Scored_By_Model'] = True

           
            # === Build Inverse from already-scored df_canon
           # === Build inverse from scored canonical
            df_inverse = df_canon.copy(deep=True)
            df_inverse['Model_Sharp_Win_Prob'] = 1 - df_inverse['Model_Sharp_Win_Prob']
            df_inverse['Model_Confidence'] = 1 - df_inverse['Model_Confidence']
            df_inverse['Was_Canonical'] = False
            df_inverse['Scored_By_Model'] = True
            
            if market_type == "totals":
                # ✅ Invert outcome cleanly: over → under, under → over
                df_inverse['Outcome'] = df_inverse['Outcome'].map(lambda x: 'under' if x == 'over' else 'over')
                df_inverse['Outcome_Norm'] = df_inverse['Outcome']
            
                # Rebuild Game_Key and Game_Key_Base for the flipped side
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
            
                # Safe merge to get the opposing line
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
            
                df_inverse = df_inverse.drop_duplicates(subset=['Game_Key', 'Market', 'Bookmaker', 'Outcome', 'Snapshot_Timestamp'])
                        
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
                df_inverse = df_inverse.drop_duplicates(subset=['Game_Key', 'Market', 'Bookmaker', 'Outcome', 'Snapshot_Timestamp'])

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
                df_inverse = df_inverse.drop_duplicates(subset=['Game_Key', 'Market', 'Bookmaker', 'Outcome', 'Snapshot_Timestamp'])

            
            #st.subheader(f"🧪 {market_type.upper()} — Inverse Preview (Before Dedup)")
            #st.info(f"🔄 Inverse rows generated pre-dedup: {len(df_inverse)}")
            
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
            st.info(f"🏷️ Filtered to sport = {bq_sport}: {before} → {after} rows")
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
        st.info("🕒 Filtering to truly live (upcoming) picks based on Game_Start...")
        
        now = pd.Timestamp.utcnow()
        df_moves_raw['Game_Start'] = pd.to_datetime(df_moves_raw['Game_Start'], errors='coerce', utc=True)
        
        before = len(df_moves_raw)
        df_moves_raw = df_moves_raw[df_moves_raw['Game_Start'] > now]
        after = len(df_moves_raw)
        
        st.info(f"✅ Game_Start > now: filtered {before} → {after} rows")
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
                st.info(f"🧹 Dropped {len(cols_to_drop)} conflicting non-key, non-protected columns before merge.")

           
                
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
                st.info(f"🧠 Pre_Game restored: {restored:,} / {total:,} rows have non-null values")

                
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
                    st.success("✅ All rows successfully scored.")

        
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
                'Sharp_Confidence_Tier': 'First_Tier',
                'Model_Sharp_Win_Prob': 'First_Sharp_Prob'
            })[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'First_Line_Value', 'First_Tier', 'First_Sharp_Prob']]
        )
        st.write("🧪 Step 2: df_first preview after renaming")
        st.dataframe(df_first[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'First_Line_Value']].head(10))
        st.write("Non-null First_Line_Value rows in df_first:", df_first['First_Line_Value'].notna().sum())

                # === Normalize and merge first snapshot into df_moves_raw
        df_first['Bookmaker'] = df_first['Bookmaker'].astype(str).str.strip().str.lower()
        df_moves_raw['Bookmaker'] = df_moves_raw['Bookmaker'].astype(str).str.strip().str.lower()
        
        df_moves_raw = df_moves_raw.merge(
            df_first, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left'
        )
        st.write("🧪 Post-merge: is First_Line_Value in df_moves_raw?")
        st.write('First_Line_Value' in df_moves_raw.columns)
        st.write("✅ Step 3: Post-merge — First_Line_Value null count:", df_moves_raw['First_Line_Value'].isnull().sum())
        st.dataframe(
            df_moves_raw[df_moves_raw['First_Line_Value'].isnull()][
                ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
            ].head(10)
        )
        st.write("🧪 Step 3: First_Line_Value after merging into df_moves_raw")
        st.dataframe(df_moves_raw[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'First_Line_Value']].head(10))
        st.write("Non-null First_Line_Value rows in df_moves_raw:", df_moves_raw['First_Line_Value'].notna().sum())

        # Alias for clarity in trend logic
        if 'First_Sharp_Prob' in df_moves_raw.columns and 'First_Model_Prob' not in df_moves_raw.columns:
            df_moves_raw['First_Model_Prob'] = df_moves_raw['First_Sharp_Prob']
        
                # === Deduplicate before filtering and diagnostics
        before = len(df_moves_raw)
        df_moves_raw = df_moves_raw.drop_duplicates(subset=['Game_Key', 'Market', 'Bookmaker', 'Outcome'], keep='last')
        after = len(df_moves_raw)
        st.info(f"🧹 Deduplicated df_moves_raw: {before:,} → {after:,}")
        
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
        st.write("📋 Columns in df_first_cols:", df_first_cols.columns.tolist())
        
        # === Step 3: Filter pre-game picks AFTER normalization
        df_pre = df_moves_raw[
            (df_moves_raw['Pre_Game'] == True) &
            (df_moves_raw['Model_Sharp_Win_Prob'].notna()) &
            (pd.to_datetime(df_moves_raw['Game_Start'], errors='coerce') > now)
        ].copy()
        
        # === Step 4: Normalize df_pre again (redundant but safe)
        for col in merge_keys:
            df_pre[col] = df_pre[col].astype(str).str.strip().str.lower()
            df_first_cols[col] = df_first_cols[col].astype(str).str.strip().str.lower()
        
        # Step 5: Prepare to avoid column loss
        for col in ['First_Model_Prob', 'First_Line_Value', 'First_Tier']:
            if col not in df_pre.columns:
                df_pre[col] = None  # ensures column exists even if merge fails
        
        # Step 5+1: Merge
        df_pre = df_pre.merge(df_first_cols, on=merge_keys, how='left', indicator=True)
        st.write("🧪 Merge indicator counts:", df_pre['_merge'].value_counts())
        df_pre.drop(columns=['_merge'], inplace=True)
   
    
        
        # === Step 6: Confirm merge success
        if 'First_Line_Value' in df_pre.columns:
            st.write("✅ First_Line_Value notna after merge:", df_pre['First_Line_Value'].notna().sum())
            st.dataframe(df_pre[['Game_Key', 'Outcome', 'Bookmaker', 'First_Line_Value']].head(10))
        else:
            st.warning("⚠️ 'First_Line_Value' column is missing after merge!")
        
        # === Step 7: Deduplicate post-merge
        df_pre = df_pre.drop_duplicates(subset=merge_keys, keep='last')
        
        # === Optional: Normalize again (safety for downstream groupby)
        df_pre['Bookmaker'] = df_pre['Bookmaker'].str.lower()
        df_pre['Outcome'] = df_pre['Outcome'].astype(str).str.strip().str.lower()
        
        # === Debug/Preview Other Tables
        st.dataframe(df_history[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Value']].head(10))
        st.dataframe(df_first[['Game_Key', 'First_Line_Value']].head(10))
        
        # === Step 8: Rename columns for display
        df_pre.rename(columns={
                'Game': 'Matchup',
                'Model_Sharp_Win_Prob': 'Model Prob',
                'Model_Confidence_Tier': 'Confidence Tier'
        }, inplace=True)
        st.dataframe(df_history[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Value']].head(10))
        st.dataframe(df_first[['Game_Key', 'First_Line_Value']].head(10))
        
       

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
        st.write("🔍 Before df_summary_base dedup: First_Line_Value count")
        st.write(df_pre['First_Line_Value'].notna().sum())
        # Create df_summary_base
        df_summary_base = df_pre.drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='last')
        df_summary_base['Sharp Line'] = df_summary_base['Sharp Line'].fillna(df_pre['Sharp Line'])
        df_summary_base['Rec Line'] = df_summary_base['Rec Line'].fillna(df_pre['Rec Line'])
        # Ensure required columns exist
        for col in ['Sharp Line', 'Rec Line', 'First_Line_Value']:
            if col not in df_summary_base.columns:
                df_summary_base[col] = np.nan
        st.write("🚨 In df_summary_base: First_Line_Value")
        st.write(df_summary_base['First_Line_Value'].notna().sum())              
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
        st.write("✅ Movement check — Nulls:")
        st.write(df_summary_base[['Sharp Line', 'Rec Line', 'First_Line_Value']].isnull().sum())
        st.dataframe(df_summary_base[['Game_Key', 'Bookmaker', 'First_Line_Value', 'Sharp Line', 'Rec Line', 'Sharp Move']].head(10))
        # Preview after calc
        st.write("✅ Movement check:")
        st.dataframe(df_summary_base[['Sharp Line', 'Rec Line', 'First_Line_Value', 'Sharp Move', 'Rec Move']].head())

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
        dupes = (
            summary_df.groupby(['Game_Key', 'Market', 'Outcome'])
            .size()
            .reset_index(name='count')
            .query("count > 1")
        )
        
        if not dupes.empty:
            st.warning(f"🚨 Found {len(dupes)} duplicated Game_Key-Market-Outcome combinations")
        
            # Merge back to see actual duplicated rows
            duped_rows = summary_df.merge(
                dupes[['Game_Key', 'Market', 'Outcome']],
                on=['Game_Key', 'Market', 'Outcome'],
                how='inner'
            ).sort_values(['Game_Key', 'Market', 'Outcome'])
        
            st.dataframe(duped_rows.head(10))
        
            # 🧪 Inspect what’s actually varying across the duplicates
            diff_cols = summary_df.columns.difference(['Game_Key', 'Market', 'Outcome'])
            st.write("🧪 Sample differences across duplicate rows:")
            st.dataframe(duped_rows[diff_cols].drop_duplicates())


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
                'Rec Line': 'first',
                'Sharp Line': 'first',
                'Rec Move': 'first',
                'Sharp Move': 'first',
                'Model Prob': 'first',
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
        
        st.subheader("📊 Debug — Grouped DF Columns")
        st.write(summary_grouped.columns.tolist())
        st.write(f"✅ Filtered rows: {len(filtered_df)}")
        st.dataframe(filtered_df[['Game_Key', 'Market', 'Outcome', 'Model Prob']].head(10))

        
        required_cols = ['Model Prob', 'Confidence Tier']
        for col in required_cols:
            if col not in summary_grouped.columns:
                summary_grouped[col] = np.nan if 'Prob' in col else ""
        if 'Model_Sharp_Win_Prob' in filtered_df.columns:
            filtered_df['Model Prob'] = filtered_df['Model_Sharp_Win_Prob']
        else:
            st.warning("⚠️ Model_Sharp_Win_Prob missing — assigning Model Prob = 0")
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
        st_autorefresh(interval=180 * 1000, key=f"{label}_odds_refresh")  # every 3 minutes
    
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

    


from google.cloud import bigquery
import pandas as pd

def load_backtested_predictions(sport_label: str, days_back: int = 30) -> pd.DataFrame:
    client = bigquery.Client(location="us")  # Correct location
    query = f"""
      
        SELECT 
            Game_Key, Bookmaker, Market, Outcome,
            Value,  -- ✅ Needed for spreads/totals canonical side filtering
            Sharp_Move_Signal, Sharp_Limit_Jump, Sharp_Prob_Shift,
            Sharp_Time_Score, Sharp_Limit_Total,
            Is_Reinforced_MultiMarket, Market_Leader, LimitUp_NoMove_Flag,
            SharpBetScore, Enhanced_Sharp_Confidence_Score, True_Sharp_Confidence_Score,
            SHARP_HIT_BOOL, SHARP_COVER_RESULT, Scored, Snapshot_Timestamp, Sport
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
        import streamlit as st
        st.error(f"❌ Failed to load predictions: {e}")
        return pd.DataFrame()



def render_sharp_signal_analysis_tab(tab, sport_label, sport_key_api):
    from google.cloud import bigquery
    client = bigquery.Client(project="sharplogger", location="us")

    with tab:
        st.subheader(f"📈 Model Calibration – {sport_label}")
        sport_label_upper = sport_label.upper()

        # === Load data
        try:
            df_master = client.query(f"""
                SELECT * FROM `sharplogger.sharp_data.sharp_moves_master`
                WHERE Sport = '{sport_label_upper}'
            """).to_dataframe()

            df_scores = client.query(f"""
                SELECT * FROM `sharplogger.sharp_data.sharp_scores_full`
                WHERE Sport = '{sport_label_upper}'
            """).to_dataframe()
        except Exception as e:
            st.error(f"❌ Failed to load BigQuery data: {e}")
            return

        # === Empty check
        if df_master.empty or df_scores.empty:
            st.warning(f"⚠️ No sharp picks found for {sport_label}")
            available_sports = client.query("""
                SELECT DISTINCT Sport FROM `sharplogger.sharp_data.sharp_scores_full`
            """).to_dataframe()
            st.info("📦 Available sports in sharp_scores_full:")
            st.dataframe(available_sports)
            return

        merge_keys = ['Game_Key', 'Bookmaker', 'Market', 'Outcome']

        # === Normalize text columns
        for df_ in [df_master, df_scores]:
            for col in merge_keys:
                if df_[col].dtype == "object":
                    df_[col] = df_[col].str.strip().str.lower()

        # === Filter scores to rows that have valid SHARP_HIT_BOOL
        df_scores_filtered = (
            df_scores[df_scores['SHARP_HIT_BOOL'].notna()]
            [['Game_Key', 'Bookmaker', 'Market', 'Outcome', 'SHARP_HIT_BOOL']]
            .drop_duplicates(subset=['Game_Key', 'Bookmaker', 'Market', 'Outcome'])
            .copy()
        )
        df_master = df_master.drop_duplicates(subset=['Game_Key', 'Bookmaker', 'Market', 'Outcome'])
        # === Early diagnostics
        st.markdown("🔍 Market breakdown in df_scores with SHARP_HIT_BOOL:")
        st.dataframe(
            df_scores_filtered
            .groupby(['Market', 'Outcome'])
            .size()
            .reset_index(name='Rows with SHARP_HIT_BOOL')
            .sort_values('Rows with SHARP_HIT_BOOL', ascending=False)
        )

        # === Show pre-merge sample matches
        preview_merge = df_master.merge(df_scores_filtered, on=merge_keys, how='inner')
        st.success(f"✅ Pre-merge matches with SHARP_HIT_BOOL: {len(preview_merge)}")

        # ✅ Confirm columns in merged result
        st.write("✅ preview_merge columns:", preview_merge.columns.tolist())
        
        # Prefer the SHARP_HIT_BOOL from scores (likely '_y' after merge)
        score_col = None
        if 'SHARP_HIT_BOOL_y' in preview_merge.columns:
            score_col = 'SHARP_HIT_BOOL_y'
        elif 'SHARP_HIT_BOOL' in preview_merge.columns:
            score_col = 'SHARP_HIT_BOOL'
        
        # Build display columns
        cols_to_display = merge_keys + ([score_col] if score_col else [])
        
        if score_col:
            st.success(f"✅ Using '{score_col}' as SHARP_HIT_BOOL column")
            st.dataframe(preview_merge[cols_to_display].head(10))
        else:
            st.error("❌ SHARP_HIT_BOOL not found in preview_merge after merge.")
            st.dataframe(preview_merge.head(10))

        # === Required columns check
        required_master_cols = merge_keys + ['Model_Sharp_Win_Prob', 'Model_Confidence']
        missing_in_master = [col for col in required_master_cols if col not in df_master.columns]
        if missing_in_master:
            st.error(f"❌ Missing columns in df_master: {missing_in_master}")
            return

        # === Final merge for analysis
        df = df_master.merge(df_scores_filtered, on=merge_keys, how='inner')
        st.info(f"🔗 Rows after merge: {len(df)}")
        # Normalize SHARP_HIT_BOOL column
        if 'SHARP_HIT_BOOL_y' in df.columns:
            df['SHARP_HIT_BOOL'] = df['SHARP_HIT_BOOL_y']
        elif 'SHARP_HIT_BOOL' not in df.columns:
            st.error("❌ SHARP_HIT_BOOL missing in merged dataframe.")
            st.dataframe(df.head())
            return
        if df.empty:
            st.error("❌ Merge returned 0 rows — likely due to mismatched keys.")
            st.markdown("🧩 **Mismatch diagnostics:**")
            mismatch_debug = df_master[merge_keys].merge(
                df_scores_filtered[merge_keys],
                on=merge_keys,
                how='outer',
                indicator=True
            )
            st.dataframe(mismatch_debug[mismatch_debug['_merge'] != 'both'].head(100))
            return

        # === Defensive column check
        if 'SHARP_HIT_BOOL' not in df.columns:
            st.error("❌ 'SHARP_HIT_BOOL' missing after merge — cannot continue.")
            st.dataframe(df.head())
            return

        # === Clean and bin
        df['Model_Sharp_Win_Prob'] = pd.to_numeric(df['Model_Sharp_Win_Prob'], errors='coerce')
        df['Model_Confidence'] = pd.to_numeric(df['Model_Confidence'], errors='coerce')
        df['SHARP_HIT_BOOL'] = pd.to_numeric(df['SHARP_HIT_BOOL'], errors='coerce')
        df = df[df['SHARP_HIT_BOOL'].notna()]

        prob_bins = np.linspace(0, 1, 11)
        bin_labels = [f"{int(p*100)}–{int(prob_bins[i+1]*100)}%" for i, p in enumerate(prob_bins[:-1])]
        df['Prob_Bin'] = pd.cut(df['Model_Sharp_Win_Prob'], bins=prob_bins, labels=bin_labels)
        df['Conf_Bin'] = pd.cut(df['Model_Confidence'], bins=prob_bins, labels=bin_labels)

        # === Summary tables
        prob_summary = (
            df.groupby(['Market', 'Prob_Bin'])['SHARP_HIT_BOOL']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'Picks', 'mean': 'Win_Rate'})
            .reset_index()
        )

        conf_summary = (
            df.groupby(['Market', 'Conf_Bin'])['SHARP_HIT_BOOL']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'Picks', 'mean': 'Win_Rate'})
            .reset_index()
        )

        # === Output tables
        st.subheader("📊 Overall Model Win Rate by Probability Bin")
        bin_summary = (
            df.groupby('Prob_Bin')['SHARP_HIT_BOOL']
            .agg(['count', 'mean'])
            .rename(columns={'count': 'Picks', 'mean': 'Win_Rate'})
            .reset_index()
        )
        st.dataframe(bin_summary.style.format({'Win_Rate': '{:.1%}'}))

        st.markdown("#### 📉 Probability Calibration by Market")
        for market in prob_summary['Market'].dropna().unique():
            st.markdown(f"**📊 {market.upper()}**")
            st.dataframe(
                prob_summary[prob_summary['Market'] == market]
                .drop(columns='Market')
                .style.format({'Win_Rate': '{:.1%}'})
            )

        st.markdown("#### 🎯 Confidence Score Calibration by Market")
        for market in conf_summary['Market'].dropna().unique():
            st.markdown(f"**📊 {market.upper()}**")
            st.dataframe(
                conf_summary[conf_summary['Market'] == market]
                .drop(columns='Market')
                .style.format({'Win_Rate': '{:.1%}'})
            )
            
# --- Set up tab selection state
if "active_sport_tab" not in st.session_state:
    st.session_state["active_sport_tab"] = "General"

# --- Top-level tabs: General + others
main_tabs = st.tabs(["🏠 General", "🏀 NBA", "⚾ MLB", "🏈 CFL", "🏀 WNBA"])
tab_general, tab_nba, tab_mlb, tab_cfl, tab_wnba = main_tabs

# --- General Tab
with tab_general:
    st.header("🎯 Sharp Scanner Dashboard")
    st.write("Welcome! Choose a league below to begin scanning or training models.")

    if st.button("🏀 Go to NBA"):
        st.session_state["active_sport_tab"] = "NBA"
    if st.button("⚾ Go to MLB"):
        st.session_state["active_sport_tab"] = "MLB"
    if st.button("🏈 Go to CFL"):
        st.session_state["active_sport_tab"] = "CFL"
    if st.button("🏀 Go to WNBA"):
        st.session_state["active_sport_tab"] = "WNBA"

# --- NBA Tab
with tab_nba:
    if st.session_state["active_sport_tab"] == "NBA":
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

# --- MLB Tab
with tab_mlb:
    if st.session_state["active_sport_tab"] == "MLB":
        st.subheader("⚾ MLB Sharp Scanner")
        run_mlb = st.checkbox("Run MLB Scanner", value=True, key="run_mlb_scanner")
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

# --- CFL Tab
with tab_cfl:
    if st.session_state["active_sport_tab"] == "CFL":
        st.subheader("🏈 CFL Sharp Scanner")
        run_cfl = st.checkbox("Run CFL Scanner", value=True, key="run_cfl_scanner")
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

# --- WNBA Tab
with tab_wnba:
    if st.session_state["active_sport_tab"] == "WNBA":
        st.subheader("🏀 WNBA Sharp Scanner")
        run_wnba = st.checkbox("Run WNBA Scanner", value=True, key="run_wnba_scanner")
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

