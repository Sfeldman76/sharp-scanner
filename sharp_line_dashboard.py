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


def train_sharp_model_from_bq(sport: str = "NBA", days_back: int = 7):
    st.info(f"🎯 Training sharp model for {sport.upper()}...")

    df_bt = load_backtested_predictions(sport, days_back)

    if df_bt.empty:
        st.warning("⚠️ No historical sharp picks available to train model.")
        return

    df_bt = df_bt.copy()
    df_bt['Model_Sharp_Win_Prob'] = pd.to_numeric(df_bt['Model_Sharp_Win_Prob'], errors='coerce')
    df_bt['SharpBetScore'] = pd.to_numeric(df_bt['SharpBetScore'], errors='coerce')
    df_bt['SHARP_HIT_BOOL'] = pd.to_numeric(df_bt['SHARP_HIT_BOOL'], errors='coerce')

    trained_models = {}

    for market in ['spreads', 'totals', 'h2h']:
        df_market = df_bt[df_bt['Market'] == market].copy()
        if df_market.empty or df_market['SHARP_HIT_BOOL'].nunique() < 2:
            st.warning(f"⚠️ Not enough data to train {market.upper()} model.")
            continue

        # Use basic features — expand if needed
        features = [
            'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
            'Sharp_Time_Score', 'Sharp_Limit_Total',
            'Is_Reinforced_MultiMarket', 'Market_Leader', 'LimitUp_NoMove_Flag'
        ]
        df_market = ensure_columns(df_market, features, 0)

        X = df_market[features]
        y = df_market['SHARP_HIT_BOOL']

        # Clean and convert
        for col in features:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        X = X.astype(float)
        y = y.astype(int)

        # Train model
        model = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, eval_metric='logloss')

        model.fit(X, y)

        # Calibrate model
        iso = IsotonicRegression(out_of_bounds='clip')
        raw_probs = model.predict_proba(X)[:, 1]
        iso.fit(raw_probs, y)

        # Save to GCS
        save_model_to_gcs(model, iso, sport, market, bucket_name=GCS_BUCKET)
        trained_models[market] = {"model": model, "calibrator": iso}

        st.success(f"✅ Trained + saved model for {market.upper()}")

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
        start = time.time()
        model = bundle['model']
        iso = bundle['calibrator']
        df_market = df[df['Market'] == market_type].copy()

        if df_market.empty:
            continue

        try:
            model_features = model.get_booster().feature_names

            # Ensure all required features exist and are numeric
            missing_cols = [col for col in model_features if col not in df_market.columns]
            for col in missing_cols:
                df_market[col] = 0
            
            df_market = df_market[model_features].copy()
            df_market = df_market.replace({'True': 1, 'False': 0, 'true': 1, 'false': 0})
            df_market = df_market.apply(pd.to_numeric, errors='coerce').fillna(0)


            df_market = df_market[model_features].astype(float)

            # Predict with calibration
            raw_probs = model.predict_proba(df_market)[:, 1]
            calibrated_probs = iso.predict(raw_probs)

            df.loc[df_market.index, 'Model_Sharp_Win_Prob'] = raw_probs
            df.loc[df_market.index, 'Model_Confidence'] = calibrated_probs


        except Exception as e:
            st.error(f"❌ Failed to apply model for {market_type}: {e}")
        st.info(f"⏱️ Scored {market_type} in {time.time() - start:.2f}s")
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
        prev = read_latest_snapshot_from_bigquery(hours=24) or {}
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
            df_moves_raw = read_recent_sharp_moves(hours=12)
            st.session_state[detection_key] = df_moves_raw
            st.success(f"📥 Loaded sharp moves from BigQuery")
       

        df_moves_raw['Sport'] = df_moves_raw['Sport'].astype(str)
        label_lower = label.lower()
        
        # Normalize and match the sport field (e.g., 'basketball_nba' should match 'nba')
        df_moves_raw = df_moves_raw[df_moves_raw['Sport'].str.lower() == sport_key_lower]



        # ✅ DEBUG: Show sport filtering
    

        # === Filter to only live (upcoming) picks
        df_moves_raw['Game_Start'] = pd.to_datetime(df_moves_raw['Game_Start'], errors='coerce', utc=True)
        df_moves_raw = df_moves_raw[df_moves_raw['Game_Start'] >= pd.Timestamp.utcnow()]
    

              # === Exit early if none remain
        if df_moves_raw.empty:
            st.warning("⚠️ No live sharp picks available at this time.")
            return pd.DataFrame()

        # Continue to enrichment + scoring...
        df_moves_raw = df_moves_raw[df_moves_raw['Book'].isin(SHARP_BOOKS + REC_BOOKS)]
        
        # === Enrich and Score ===
        df_moves_raw['Game_Start'] = pd.to_datetime(df_moves_raw['Game_Start'], errors='coerce', utc=True)
        df_moves_raw['Snapshot_Timestamp'] = timestamp
        now = pd.Timestamp.utcnow()
        df_moves_raw['Pre_Game'] = df_moves_raw['Game_Start'] > now
        df_moves_raw['Post_Game'] = ~df_moves_raw['Pre_Game']
        df_moves_raw['Sport'] = label.upper()
        df_moves_raw = build_game_key(df_moves_raw)
        
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
        
        # === Apply model scoring
        if trained_models:
            try:
                df_pre_game = df_moves_raw[df_moves_raw['Pre_Game']].copy()
        
                if not df_pre_game.empty:
                    df_scored = apply_blended_sharp_score(df_pre_game, trained_models)
        
                    if not df_scored.empty:
                        merge_keys = ['Game_Key', 'Bookmaker', 'Market', 'Outcome']
                        score_cols = ['Model_Sharp_Win_Prob', 'Model_Confidence_Tier']
        
                        missing_cols = [col for col in merge_keys + score_cols if col not in df_scored.columns]
                        if missing_cols:
                            st.warning(f"⚠️ Cannot merge — missing columns: {missing_cols}")
                            st.dataframe(df_scored.head(3))
                        else:
                            df_moves_raw = df_moves_raw.merge(
                                df_scored[merge_keys + score_cols],
                                on=merge_keys,
                                how='left',
                                suffixes=('', '_scored')
                            )
                            
                            # Clean up any residual suffixes (e.g., _scored, _x, _y)
                            df_moves_raw.columns = df_moves_raw.columns.str.replace(r'_x$|_y$|_scored$', '', regex=True)

        
                            for col in score_cols:
                                scored_col = f"{col}_scored"
                                if scored_col in df_moves_raw.columns:
                                    df_moves_raw[col] = df_moves_raw[scored_col]
                                    df_moves_raw.drop(columns=[scored_col], inplace=True)
                    else:
                        st.warning("⚠️ Scoring produced no output.")
                else:
                    st.info("ℹ️ No pre-game rows available for scoring.")
            except Exception as e:
                st.error(f"❌ Model scoring failed: {e}")
        else:
            st.warning("⚠️ No trained models available for scoring.")

        
               
        if df_moves_raw.empty:
            st.warning("⚠️ No live sharp picks available in the last 12 hours.")
            return pd.DataFrame()
        # === Final Cleanup + Placeholders
        df_moves_raw = df_moves_raw.rename(columns=lambda x: x.rstrip('_x'))
        # === Consensus line calculation ===
        sharp_books = ['pinnacle', 'bookmaker', 'betonline']  # customize as needed
        rec_books = ['fanduel', 'draftkings', 'pointsbet', 'betmgm']
        
        # Normalize bookmaker column
        df_moves_raw['Bookmaker'] = df_moves_raw['Bookmaker'].str.lower()
        
        # Sharp consensus
        df_sharp = df_moves_raw[df_moves_raw['Bookmaker'].isin(sharp_books)]
        sharp_consensus = df_sharp.groupby(['Game_Key', 'Market', 'Outcome'])['Value'].mean().reset_index()
        sharp_consensus.rename(columns={'Value': 'Sharp_Book_Consensus'}, inplace=True)
        
        # Rec consensus
        df_rec = df_moves_raw[df_moves_raw['Bookmaker'].isin(rec_books)]
        rec_consensus = df_rec.groupby(['Game_Key', 'Market', 'Outcome'])['Value'].mean().reset_index()
        rec_consensus.rename(columns={'Value': 'Rec_Book_Consensus'}, inplace=True)
        
        # Merge into df_moves_raw
        df_moves_raw = df_moves_raw.merge(sharp_consensus, on=['Game_Key', 'Market', 'Outcome'], how='left')
        df_moves_raw = df_moves_raw.merge(rec_consensus, on=['Game_Key', 'Market', 'Outcome'], how='left')
                # === 1. Load df_history and compute df_first
        # === Load broader trend history for open line / tier comparison
        df_history = read_recent_sharp_moves(hours=72)
        
        # ✅ Filter to only rows relevant to current live games
        df_history = df_history[df_history['Game_Key'].isin(df_moves_raw['Game_Key'])]
        df_history = df_history[df_history['Model_Sharp_Win_Prob'].notna()]

        
        # === 1. Build First Snapshot Table
        df_first = (
            df_history.sort_values('Snapshot_Timestamp')
            .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='first')
            .rename(columns={
                'Value': 'First_Line_Value',
                'Sharp_Confidence_Tier': 'First_Tier',
                'Model_Sharp_Win_Prob': 'First_Sharp_Prob'
            })[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'First_Line_Value', 'First_Tier', 'First_Sharp_Prob']]
        )
        
        df_first['Bookmaker'] = df_first['Bookmaker'].astype(str).str.strip().str.lower()
        df_moves_raw['Bookmaker'] = df_moves_raw['Bookmaker'].astype(str).str.strip().str.lower()
        
        # === 2. Merge First Snapshot into Raw Moves
        df_moves_raw = df_moves_raw.merge(df_first, on=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], how='left')
        
        # === 3. Movement Calculations
        for col_name, col_calc in {
            'Move_From_Open_Sharp': lambda df: df['Sharp_Book_Consensus'] - df['First_Line_Value'],
            'Move_From_Open_Rec': lambda df: df['Rec_Book_Consensus'] - df['First_Line_Value']
        }.items():
            if 'First_Line_Value' in df_moves_raw.columns:
                df_moves_raw[col_name] = col_calc(df_moves_raw).round(2)
        
        # === 4. Ensure Required Fields
        fallbacks = {
            'Ref_Sharp_Value': df_moves_raw.get('Value'),
            'SHARP_HIT_BOOL': None,
            'SHARP_COVER_RESULT': None,
            'Scored': None,
            'Model_Sharp_Win_Prob': None,
            'Model_Confidence_Tier': None,
        }
        for col, fallback in fallbacks.items():
            if col not in df_moves_raw.columns:
                df_moves_raw[col] = fallback
        
        # === 5. Trend + Diagnostic Columns
        TIER_ORDER = {'⚠️ Low': 1, '✅ Medium': 2, '⭐ High': 3, '🔥 Steam': 4}
        
        df_moves_raw['Tier_Change'] = df_moves_raw.apply(
            lambda row: (
                "↔ No Change" if TIER_ORDER.get(str(row['Model_Confidence_Tier']).strip(), 0) == TIER_ORDER.get(str(row['First_Tier']).strip(), 0)
                else f"{'↑' if TIER_ORDER.get(str(row['Model_Confidence_Tier']).strip(), 0) > TIER_ORDER.get(str(row['First_Tier']).strip(), 0) else '↓'} {row['First_Tier']} → {row['Model_Confidence_Tier']}"
            ) if pd.notna(row['First_Tier']) else "⚠️ Missing",
            axis=1
        )
        
        df_moves_raw['📊 Confidence Evolution'] = df_moves_raw.apply(
            lambda row: (
                "⚠️ Missing" if pd.isna(row['First_Sharp_Prob']) or pd.isna(row['Model_Sharp_Win_Prob']) else
                f"{'📈 Trending Up' if row['Model_Sharp_Win_Prob'] - row['First_Sharp_Prob'] >= 0.04 else '📉 Trending Down' if row['Model_Sharp_Win_Prob'] - row['First_Sharp_Prob'] <= -0.04 else '↔ Stable'}: {row['First_Sharp_Prob']:.2%} → {row['Model_Sharp_Win_Prob']:.2%}"
            ),
            axis=1
        )
        
        df_moves_raw['Direction'] = df_moves_raw.apply(
            lambda row: (
                "⚪ Mixed" if pd.isna(row['First_Sharp_Prob']) or pd.isna(row['Model_Sharp_Win_Prob']) or pd.isna(row['First_Line_Value']) or pd.isna(row['Value']) else
                "🟢 Model ↑ / Line ↓" if row['Model_Sharp_Win_Prob'] - row['First_Sharp_Prob'] > 0.04 and row['Value'] - row['First_Line_Value'] < 0 else
                "🔴 Model ↓ / Line ↑" if row['Model_Sharp_Win_Prob'] - row['First_Sharp_Prob'] < -0.04 and row['Value'] - row['First_Line_Value'] > 0 else
                "🟢 Aligned ↑" if row['Model_Sharp_Win_Prob'] - row['First_Sharp_Prob'] > 0.04 and row['Value'] - row['First_Line_Value'] > 0 else
                "🔻 Aligned ↓" if row['Model_Sharp_Win_Prob'] - row['First_Sharp_Prob'] < -0.04 and row['Value'] - row['First_Line_Value'] < 0 else
                "⚪ Mixed"
            ),
            axis=1
        )
        
        def build_model_reason(row):
            reasons = []
            try:
                prob = float(row.get('Model_Sharp_Win_Prob', 0))
                if prob >= 0.58: reasons.append("🔼 Strong Model Edge")
                elif prob >= 0.52: reasons.append("↗️ Slight Model Lean")
                elif prob <= 0.48: reasons.append("↘️ Slight Model Fade")
                elif prob <= 0.42: reasons.append("🔽 Strong Model Fade")
                else: reasons.append("🪙 Coinflip")
            except:
                reasons.append("⚠️ No model confidence")
        
            if row.get('Sharp_Prob_Shift', 0) > 0: reasons.append("Confidence ↑")
            if row.get('Sharp_Prob_Shift', 0) < 0: reasons.append("Confidence ↓")
            if row.get('Sharp_Limit_Jump', 0): reasons.append("Limit Jump")
            if row.get('Market_Leader', 0): reasons.append("Led Market Move")
            if row.get('Is_Reinforced_MultiMarket', 0): reasons.append("Cross-Market Signal")
            if row.get('LimitUp_NoMove_Flag', 0): reasons.append("Limit ↑ w/o Price Move")
            return " | ".join(reasons)
        
        df_moves_raw['📌 Model Reasoning'] = df_moves_raw.apply(build_model_reason, axis=1)
        
        
        # === 6. Final Summary Table
        summary_cols = [
            'Game', 'Market', 'Game_Start', 'Outcome',
            'Rec_Book_Consensus', 'Sharp_Book_Consensus',
            'Move_From_Open_Rec', 'Move_From_Open_Sharp',
            'Model_Sharp_Win_Prob', 'Model_Confidence_Tier',
            '📌 Model Reasoning', '📊 Confidence Evolution',
            'Tier_Change', 'Direction'
        ]
        summary_df = df_moves_raw[[col for col in summary_cols if col in df_moves_raw.columns]].copy()
        summary_df['Game_Start'] = pd.to_datetime(summary_df['Game_Start'], errors='coerce', utc=True)
        summary_df = summary_df[summary_df['Game_Start'].notna()]
        summary_df['Date + Time (EST)'] = summary_df['Game_Start'].dt.tz_convert('US/Eastern').dt.strftime('%Y-%m-%d %I:%M %p')
        summary_df['Event_Date_Only'] = summary_df['Game_Start'].dt.date.astype(str)
        summary_df.columns = summary_df.columns.str.replace(r'_x$|_y$|_scored$', '', regex=True)
        summary_df = summary_df.loc[:, ~summary_df.columns.duplicated()]

        summary_df.rename(columns={
            'Game': 'Matchup',
            'Outcome': 'Pick',
            'Rec_Book_Consensus': 'Rec Line',
            'Sharp_Book_Consensus': 'Sharp Line',
            'Move_From_Open_Rec': 'Rec Move',
            'Move_From_Open_Sharp': 'Sharp Move',
            'Model_Sharp_Win_Prob': 'Model Prob',
            'Model_Confidence_Tier': 'Confidence Tier',
            '📌 Model Reasoning': 'Why Model Likes It',
            '📊 Confidence Evolution': 'Confidence Trend',
            'Tier_Change': 'Tier Δ',
            'Direction': 'Line/Model Direction'
        }, inplace=True)
        # === Build Market + Date Filters
        market_options = ["All"] + sorted(summary_df['Market'].dropna().unique())
        selected_market = st.selectbox(f"📊 Filter {label} by Market", market_options, key=f"{label}_market_summary")
        
        date_only_options = ["All"] + sorted(summary_df['Event_Date_Only'].dropna().unique())
        selected_date = st.selectbox(f"📅 Filter {label} by Date", date_only_options, key=f"{label}_date_filter")
        
        # === Apply Filters
        filtered_df = summary_df.copy()
        if selected_market != "All":
            filtered_df = filtered_df[filtered_df['Market'] == selected_market]
        if selected_date != "All":
            filtered_df = filtered_df[filtered_df['Event_Date_Only'] == selected_date]
        
        # === Group by Matchup + Side + Timestamp
        summary_grouped = (
            filtered_df
            .groupby(['Matchup', 'Market', 'Pick', 'Date + Time (EST)'], as_index=False)
            .agg({
                'Rec Line': 'mean',
                'Sharp Line': 'mean',
                'Rec Move': 'mean',
                'Sharp Move': 'mean',
                'Model Prob': 'mean',
                'Confidence Tier': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
                    'Why Model Likes It': lambda x: ' | '.join(sorted(set(x.dropna()))),
                    'Confidence Trend': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
                    'Tier Δ': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
                    'Line/Model Direction': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
                })
            )
        
        # === Final Column Order for Display
        view_cols = [
            'Date + Time (EST)', 'Matchup', 'Market', 'Pick',
            'Rec Line', 'Sharp Line', 'Rec Move', 'Sharp Move',
            'Model Prob', 'Confidence Tier',
            'Why Model Likes It', 'Confidence Trend', 'Tier Δ', 'Line/Model Direction'
        ]
        
        # === Final Output
        st.subheader(f"📊 Sharp vs Rec Book Summary Table – {label}")
        st.info(f"✅ Summary table shape: {summary_grouped.shape}")
        st.dataframe(summary_grouped[view_cols], use_container_width=True)

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
        

        
        # === Render Sharp Picks Table (HTML Version)
        table_df = summary_grouped[view_cols].copy()
        table_html = table_df.to_html(classes="custom-table", index=False, escape=False)
        st.markdown(f"<div class='scrollable-table-container'>{table_html}</div>", unsafe_allow_html=True)
        st.success("✅ Finished rendering sharp picks table.")
        st.caption(f"Showing {len(table_df)} rows")

    # === 2. Render Live Odds Snapshot Table
    st.subheader(f"📊 Live Odds Snapshot – {label} (Odds + Limit)")
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
        df_display = df_odds_raw.pivot_table(
            index=["Date + Time (EST)", "Game", "Market", "Outcome"],
            columns="Bookmaker",
            values="Value_Limit",
            aggfunc="first"
        ).reset_index()
    
        # Render as HTML
        table_html_2 = df_display.to_html(classes="custom-table", index=False, escape=False)
        st.markdown(f"<div class='scrollable-table-container'>{table_html_2}</div>", unsafe_allow_html=True)
        st.success(f"✅ Live odds snapshot rendered — {len(df_display)} rows.")

def fetch_scores_and_backtest(*args, **kwargs):
    print("⚠️ fetch_scores_and_backtest() is deprecated in UI and will be handled by Cloud Scheduler.")
    return pd.DataFrame()

    


def load_backtested_predictions(sport_label: str, days_back: int = 3) -> pd.DataFrame:
    sport_label = sport_label.upper()

    query_scores = f"""
        SELECT 
            Game_Key,
            SHARP_HIT_BOOL,
            SHARP_COVER_RESULT,
            Snapshot_Timestamp AS Score_Timestamp
        FROM `sharp_data.sharp_scores_full`
        WHERE Sport = '{sport_label}'
          AND DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
          AND Scored = TRUE
    """

    query_probs = f"""
        SELECT 
            Game_Key,
            Bookmaker,
            Market,
            Outcome,
            Model_Sharp_Win_Prob,
            Model_Confidence_Tier,
            SharpBetScore,
            Enhanced_Sharp_Confidence_Score,
            True_Sharp_Confidence_Score,
            Snapshot_Timestamp AS Prob_Timestamp
        FROM `sharp_data.sharp_moves_master`
        WHERE Sport = '{sport_label}'
          AND DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
    """

    df_scores = bq_client.query(query_scores).to_dataframe()
    df_probs = bq_client.query(query_probs).to_dataframe()

    # Merge on Game_Key — inner join to ensure we have both outcome + model prediction
    df_bt = df_scores.merge(df_probs, on="Game_Key", how="inner")
    return df_bt


def render_sharp_signal_analysis_tab(tab, sport_label, sport_key_api):
    with tab:
        st.subheader(f"📈 Model Calibration – {sport_label}")

        df_bt = load_backtested_predictions(sport_label)

        if df_bt.empty:
            st.warning(f"⚠️ No matched sharp picks with scored outcomes for {sport_label}")
            return

        # Coerce to numeric
        df_bt['Model_Sharp_Win_Prob'] = pd.to_numeric(df_bt['Model_Sharp_Win_Prob'], errors='coerce')
        df_bt['SharpBetScore'] = pd.to_numeric(df_bt['SharpBetScore'], errors='coerce')

        # Quick summary view
        bins = [0, 0.5, 0.55, 0.6, 0.65, 1.0]
        df_bt['ProbBin'] = pd.cut(df_bt['Model_Sharp_Win_Prob'], bins)
        win_rates = df_bt.groupby('ProbBin')['SHARP_HIT_BOOL'].mean().rename("Win %")
        st.subheader("📊 Model Win Rate by Probability Bin")
        st.dataframe(win_rates.reset_index())

        # === Full calibration by Market
        prob_bins = np.linspace(0, 1, 11)
        df_bt['Prob_Bin'] = pd.cut(df_bt['Model_Sharp_Win_Prob'], bins=prob_bins,
                                   labels=[f"{int(p*100)}–{int(prob_bins[i+1]*100)}%" for i, p in enumerate(prob_bins[:-1])])
        df_bt['Conf_Bin'] = pd.cut(df_bt['SharpBetScore'], bins=prob_bins,
                                   labels=[f"{int(p*100)}–{int(prob_bins[i+1]*100)}%" for i, p in enumerate(prob_bins[:-1])])

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
        for market in prob_summary['Market'].dropna().unique():
            st.markdown(f"**📊 {market.upper()}**")
            st.dataframe(
                prob_summary[prob_summary['Market'] == market]
                .drop(columns='Market')
                .style.format({'Win_Rate': '{:.1%}'})
            )

        st.markdown("#### 🎯 Model Confidence Calibration by Market")
        for market in conf_summary['Market'].dropna().unique():
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

