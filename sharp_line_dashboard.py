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
import time, contextlib
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
from scipy.stats import zscore
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from html import escape

import re
import logging


GCP_PROJECT_ID = "sharplogger"  # ✅ confirmed project ID
BQ_DATASET = "sharp_data"       # ✅ your dataset name
BQ_TABLE = "sharp_moves_master" # ✅ your table name
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
MARKET_WEIGHTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.market_weights"
LINE_HISTORY_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.line_history_master"
SNAPSHOTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.odds_snapshot_log"

RATINGS_HISTORY_TABLE = "sharplogger.sharp_data.ratings_history"  # <- fully qualified

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
SHARP_BOOKS = SHARP_BOOKS_FOR_LIMITS + ['betus','mybookieag','smarkets','betfair_ex_eu','"betfair_ex_uk','betfair_ex_au','lowvig','betonlineag','matchbook']


REC_BOOKS = [
    'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers',
    'fanatics', 'espnbet', 'hardrockbet','sport888', 'fanatics', 'bovada', 'bet365', 'williamhillus', 'ballybet', 'bet365_au','betopenly']

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


def calc_implied_prob(series):
    series = pd.to_numeric(series, errors='coerce').clip(lower=-10000, upper=10000)
    return np.where(
        series < 0,
        -series / (-series + 100),
        100 / (series + 100)
    )


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

        
def build_game_key(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()

    # --- Ensure columns exist (best-effort, no early return) ---
    for c in ['Game', 'Home_Team_Norm', 'Away_Team_Norm', 'Game_Start', 'Market', 'Outcome']:
        if c not in df.columns:
            df[c] = ''

    # --- Derive home/away from "Game" if needed ---
    need_home = df['Home_Team_Norm'].astype(str).str.strip().eq('')
    need_away = df['Away_Team_Norm'].astype(str).str.strip().eq('')
    if need_home.any() or need_away.any():
        home_from_game = df['Game'].astype(str).str.extract(r'^(.*?)\s+vs', expand=False)
        away_from_game = df['Game'].astype(str).str.extract(r'vs\s+(.*)$', expand=False)
        df.loc[need_home, 'Home_Team_Norm'] = home_from_game.where(need_home, df['Home_Team_Norm'])
        df.loc[need_away, 'Away_Team_Norm'] = away_from_game.where(need_away, df['Away_Team_Norm'])

    # --- Normalize fields ---
    df['Home_Team_Norm'] = df['Home_Team_Norm'].astype(str).str.lower().str.strip()
    df['Away_Team_Norm'] = df['Away_Team_Norm'].astype(str).str.lower().str.strip()
    df['Market_Norm']    = df['Market'].astype(str).str.lower().str.strip()
    df['Outcome_Norm']   = df['Outcome'].astype(str).str.lower().str.strip()

    # --- Commence hour (floored) ---
    start = pd.to_datetime(df['Game_Start'], utc=True, errors='coerce')
    df['Commence_Hour'] = start.dt.floor('h')
    hour_str = df['Commence_Hour'].dt.strftime('%Y-%m-%dT%H:00Z').fillna('')

    # --- Canonicalize team pair (stable regardless of home/away ordering) ---
    left  = df['Home_Team_Norm'].where(df['Home_Team_Norm'] <= df['Away_Team_Norm'], df['Away_Team_Norm'])
    right = df['Away_Team_Norm'].where(df['Home_Team_Norm'] <= df['Away_Team_Norm'], df['Home_Team_Norm'])

    # --- Always-on Merge_Key_Short (no apply, always scalar) ---
    fallback_merge = left.fillna('') + '_' + right.fillna('') + '_' + hour_str
    df['Merge_Key_Short'] = fallback_merge

    # --- Synthetic Game_Key (keeps your original shape when Market/Outcome exist) ---
    synthetic_game_key = (
        left.fillna('') + '_' + right.fillna('') + '_' + hour_str + '_' +
        df['Market_Norm'].fillna('') + '_' + df['Outcome_Norm'].fillna('')
    )

    # If a Game_Key column exists, only fill blanks; otherwise create it
    if 'Game_Key' in df.columns:
        df['Game_Key'] = df['Game_Key'].astype(str)
        blank = df['Game_Key'].eq('') | df['Game_Key'].isna()
        df.loc[blank, 'Game_Key'] = synthetic_game_key
    else:
        df['Game_Key'] = synthetic_game_key

    return df
# === Tiers from probability (unified) ===
def tier_from_prob(p: float) -> str:
    if pd.isna(p): return "⚠️ Missing"
    if p >= 0.90: return "🔥 Steam"
    if p >= 0.75: return "🔥 Strong Indication"
    if p >= 0.60: return "⭐ Lean"
    return "✅ Coinflip"


    
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
# Smart getter — use cache unless forced to reload


@st.cache_data(ttl=600)
def read_recent_sharp_moves_cached(hours=72, table=BQ_FULL_TABLE):
    return read_recent_sharp_moves(hours=hours, table=table)

def read_recent_sharp_moves_conditional(force_reload=False, hours=72, table=BQ_FULL_TABLE):
    if force_reload:
        st.info("🔁 Reloading sharp moves from BigQuery...")
        return read_recent_sharp_moves(hours=hours, table=table)  # Uncached
    else:
        return read_recent_sharp_moves_cached(hours=hours, table=table)  # Cached

@st.cache_data(ttl=600)
def get_recent_history():
    st.write("📦 Using cached sharp history (get_recent_history)")
    return read_recent_sharp_moves_cached(hours=72)


def _prep_for_asof_left(df: pd.DataFrame, by_keys: list[str], on_key: str) -> pd.DataFrame:
    # drop rows where any join keys are missing
    df = df.dropna(subset=by_keys + [on_key]).copy()
    # enforce dtypes
    for bk in by_keys:
        df[bk] = df[bk].astype(str)
    df[on_key] = pd.to_datetime(df[on_key], utc=True, errors='coerce')
    df = df.dropna(subset=[on_key])
    # strict group-wise sort by keys
    df = df.sort_values(by_keys + [on_key], kind='mergesort').reset_index(drop=True)
    # sanity: ensure sorted within each group
    if not df.groupby(by_keys, sort=False)[on_key].apply(lambda s: s.is_monotonic_increasing).all():
        # enforce again group-wise just in case
        df = (df.groupby(by_keys, sort=False, group_keys=True)
                .apply(lambda g: g.sort_values(on_key, kind='mergesort'))
                .reset_index(drop=True))
    return df

def _prep_for_asof_right(df: pd.DataFrame, by_keys: list[str], on_key: str) -> pd.DataFrame:
    df = df.dropna(subset=by_keys + [on_key]).copy()
    for bk in by_keys:
        df[bk] = df[bk].astype(str)
    df[on_key] = pd.to_datetime(df[on_key], utc=True, errors='coerce')
    df = df.dropna(subset=[on_key])
    df = df.sort_values(by_keys + [on_key], kind='mergesort').reset_index(drop=True)
    return df
    
def _groupwise_asof(left: pd.DataFrame, right: pd.DataFrame,
                    by: list[str], left_on: str, right_on: str) -> pd.DataFrame:
    """Run merge_asof per-group to avoid global-sort pitfalls."""
    out = []
    # Pre-index right by groups for fast slicing
    right_groups = {k: v.sort_values(right_on, kind='mergesort')
                    for k, v in right.groupby(by, sort=False)}
    for k, g in left.groupby(by, sort=False):
        # k is a tuple when len(by)>1; normalize to tuple
        key = k if isinstance(k, tuple) else (k,)
        r = right_groups.get(key)
        gl = g.sort_values(left_on, kind='mergesort')
        if r is None or r.empty:
            # No ratings for this group → return NaNs for PR columns
            merged = gl.copy()
            for col in ['Power_Rating','PR_Off','PR_Def', right_on]:
                if col not in merged.columns:
                    merged[col] = np.nan
        else:
            merged = pd.merge_asof(
                gl, r,
                left_on=left_on, right_on=right_on,
                direction='backward', allow_exact_matches=True
            )
        out.append(merged)
    return pd.concat(out, ignore_index=True)

@st.cache_resource
def get_bq_client() -> bigquery.Client:
    return bigquery.Client()  

def norm_team(x):
    import pandas as pd
    if isinstance(x, pd.Series):
        s = x
        ret_series = True
    else:
        s = pd.Series(x)
        ret_series = False

    out = (
        s.astype(str)
         .str.lower()
         .str.strip()
         .str.replace(r'\s+', ' ', regex=True)
         .str.replace('.', '', regex=False)
         .str.replace('&', 'and', regex=False)
         .str.replace('-', ' ', regex=False)
    )

    return out if ret_series else out.iloc[0]



@contextlib.contextmanager
def tmr(label):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    st.write(f"⏱ {label}: {dt:.2f}s")


@st.cache_data(ttl=900, max_entries=64, show_spinner=False)
def fetch_power_ratings_from_bq_cached(
    sport: str,
    lookback_days: int = 400,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    source: str = "history",
    **_kwargs, 
) -> pd.DataFrame:
    """
    History only: last row per (Sport, Team, DATE) within [start_ts-3d, end_ts+3d].
    Falls back to current if empty.
    """
    bq = get_bq_client()

    # pad the window a bit so asof has something to match
    pad_start = (pd.to_datetime(start_ts, utc=True) - pd.Timedelta(days=3)).isoformat()
    pad_end   = (pd.to_datetime(end_ts,   utc=True) + pd.Timedelta(days=3)).isoformat()

    bq = get_bq_client()

    if source == "history":
        sql = """
            SELECT
              UPPER(Sport) AS Sport,
              CAST(Team AS STRING) AS Team_Raw,
              TIMESTAMP(Updated_At) AS AsOfTS,
              CAST(Rating AS FLOAT64) AS Power_Rating,
              CAST(NULL AS FLOAT64) AS PR_Off,
              CAST(NULL AS FLOAT64) AS PR_Def
            FROM `sharplogger.sharp_data.ratings_history`
            WHERE UPPER(Sport) = @sport
              AND Updated_At >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
        """
        cfg = bigquery.QueryJobConfig(
            use_query_cache=True,
            query_parameters=[
                bigquery.ScalarQueryParameter("sport", "STRING", sport.upper()),
                bigquery.ScalarQueryParameter("lookback_days", "INT64", lookback_days),
            ],
        )
    else:
        sql = """
            SELECT
              UPPER(Sport) AS Sport,
              CAST(Team AS STRING) AS Team_Raw,
              CAST(Rating AS FLOAT64) AS Power_Rating,
              CAST(NULL AS FLOAT64) AS PR_Off,
              CAST(NULL AS FLOAT64) AS PR_Def,
              TIMESTAMP(Updated_At) AS AsOfTS
            FROM `sharplogger.sharp_data.ratings_current`
            WHERE UPPER(Sport) = @sport
        """
        cfg = bigquery.QueryJobConfig(
            use_query_cache=True,
            query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport.upper())],
        )

    df = bq.query(sql, job_config=cfg).to_dataframe()
    if df.empty:
        return df


    # normalize once
    df["Sport"] = df["Sport"].astype(str).str.upper()
    df["Team_Norm"] = norm_team(df["Team_Raw"])
    df["AsOfTS"] = pd.to_datetime(df["AsOfTS"], utc=True, errors="coerce")
    df.rename(columns={"AsOfTS":"AsOf"}, inplace=True)
    # keep just what we need
    return df[["Sport","Team_Norm","AsOf","Power_Rating"]]

    # normalize exactly once here
# --- ONE canonical normalizer (handles Series OR scalar) ---



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

def compute_small_book_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    UI-only: Adds small-book liquidity features.
    Market-aware and safe if no small-limit books or missing Limit values.
    Run on df_pre (latest snapshot per book/outcome) BEFORE building df_summary_base.
    """
    SMALL_LIMIT_BOOKS = ['betfair_ex_uk','betfair_ex_eu','betfair_ex_au','matchbook','smarkets']

    out = df.copy()

    # Normalize
    out['Bookmaker_Norm'] = out.get('Bookmaker_Norm', out.get('Bookmaker', '')).astype(str).str.lower().str.strip()
    out['Market'] = out['Market'].astype(str).str.lower().str.strip()
    out['Outcome'] = out['Outcome'].astype(str).str.lower().str.strip()

    # Numeric limit
    out['Limit'] = pd.to_numeric(out.get('Limit', 0), errors='coerce').fillna(0)

    # Filter small-limit books
    sb = out[out['Bookmaker_Norm'].isin(SMALL_LIMIT_BOOKS)].copy()
    if sb.empty:
        # Create empty columns and return (UI-safe)
        for c in [
            'SmallBook_Total_Limit','SmallBook_Max_Limit','SmallBook_Min_Limit',
            'SmallBook_Limit_Skew','SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag'
        ]:
            out[c] = np.nan if c == 'SmallBook_Limit_Skew' else 0
        return out

    # Aggregate per Game_Key × Market × Outcome
    agg = (
        sb.groupby(['Game_Key','Market','Outcome'])
          .agg(
              SmallBook_Total_Limit=('Limit','sum'),
              SmallBook_Max_Limit=('Limit','max'),
              SmallBook_Min_Limit=('Limit','min'),
              SmallBook_Count=('Limit','size')
          )
          .reset_index()
    )

    # Compute skew (avoid /0)
    agg['SmallBook_Limit_Skew'] = np.where(
        agg['SmallBook_Min_Limit'] > 0,
        agg['SmallBook_Max_Limit'] / agg['SmallBook_Min_Limit'],
        np.nan
    )

    # Flags (tune thresholds as you wish)
    HEAVY_TOTAL = 700    # you used 700 in your current skew/heavy logic
    SKEW_RATIO  = 1.5

    agg['SmallBook_Heavy_Liquidity_Flag'] = (agg['SmallBook_Total_Limit'] >= HEAVY_TOTAL).astype(int)
    agg['SmallBook_Limit_Skew_Flag']      = (agg['SmallBook_Limit_Skew'] >= SKEW_RATIO).astype(int)

    # Merge skinny back (left)
    out = out.merge(
        agg[['Game_Key','Market','Outcome',
             'SmallBook_Total_Limit','SmallBook_Max_Limit','SmallBook_Min_Limit',
             'SmallBook_Limit_Skew','SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag']],
        on=['Game_Key','Market','Outcome'],
        how='left'
    )

    return out

def add_time_context_flags(df: pd.DataFrame, sport: str, local_tz: str = "America/New_York") -> pd.DataFrame:
    out = df.copy()

    # 1) Pick a timestamp source (prefer Commence_Hour, else Game_Start)
    if 'Commence_Hour' in out.columns:
        ts = pd.to_datetime(out['Commence_Hour'], errors='coerce', utc=True)
    elif 'Game_Start' in out.columns:
        ts = pd.to_datetime(out['Game_Start'], errors='coerce', utc=True)
    else:
        # if neither exist, create dummies and return
        out['Is_Weekend'] = 0
        out['Is_Night_Game'] = 0
        out['Game_Local_Hour'] = np.nan
        out['Game_DOW'] = np.nan
        return out

    # 2) Convert to local for day/night & weekend logic
    ts_local = ts.dt.tz_convert(local_tz)
    out['Game_Local_Hour'] = ts_local.dt.hour
    out['Game_DOW'] = ts_local.dt.dayofweek  # Mon=0 ... Sun=6
    out['Is_Weekend'] = out['Game_DOW'].isin([5, 6]).astype(int)

    # 3) Night cutoffs by sport (tweak to taste)
    SPORT_NIGHT_CUTOFF = {
        'MLB': 18, 'NFL': 18, 'CFL': 18, 'NBA': 18, 'WNBA': 18, 'NCAAF': 18, 'NCAAB': 18
    }
    night_cutoff = SPORT_NIGHT_CUTOFF.get(str(sport).upper(), 18)
    out['Is_Night_Game'] = (out['Game_Local_Hour'] >= night_cutoff).astype(int)

    # (Optional) primetime flag (example tuned for NFL)
    if str(sport).upper() in {'NFL', 'CFL'}:
        # Thu(3), Sun(6), Mon(0) and 7–11pm local
        out['Is_PrimeTime'] = ((out['Game_DOW'].isin([3, 6, 0])) &
                               (out['Game_Local_Hour'].between(19, 23))).astype(int)
    else:
        out['Is_PrimeTime'] = 0

    #(Optional) cyclical DOW encodings
    out['DOW_Sin'] = np.sin(2*np.pi*(out['Game_DOW'] / 7.0))
    out['DOW_Cos'] = np.cos(2*np.pi*(out['Game_DOW'] / 7.0))

    return out

def add_book_reliability_features(
    df: pd.DataFrame,
    label_col: str = "SHARP_HIT_BOOL",
    prior_strength: float = 200.0
) -> pd.DataFrame:
    """
    Leak-safe per-row features:
      Book_Reliability_Score: Beta-Binomial posterior mean for this (Sport, Market, Bookmaker)
      Book_Reliability_Lift : log-odds of that mean vs 50/50

    Uses ONLY prior rows ("as of" via cumulative counts shifted by 1).
    Requires: Sport (UPPER), Market (lower), Bookmaker (lower), time col (Game_Start or Snapshot_Timestamp), label_col.
    """
    out = df.copy()

    # --- Normalize keys & types
    out['Sport'] = out['Sport'].astype(str).str.upper()
    out['Market'] = out['Market'].astype(str).str.lower().str.strip()
    if 'Bookmaker' not in out.columns and 'Book' in out.columns:
        out['Bookmaker'] = out['Book']
    out['Bookmaker'] = out['Bookmaker'].astype(str).str.lower().str.strip()

    # Time column with safe fallback
    if 'Game_Start' in out.columns and pd.api.types.is_datetime64_any_dtype(out['Game_Start']):
        out['Game_Start'] = pd.to_datetime(out['Game_Start'], errors='coerce', utc=True)
    else:
        # Fallback to snapshot if Game_Start missing
        out['Game_Start'] = pd.to_datetime(out.get('Snapshot_Timestamp'), errors='coerce', utc=True)

    # Label
    out[label_col] = pd.to_numeric(out[label_col], errors='coerce').fillna(0).astype(int)

    # --- Sort for "as-of" math
    out = out.sort_values(['Sport', 'Market', 'Bookmaker', 'Game_Start'], kind='mergesort')

    # Book-level cumulative (exclude current row)
    g_smb = out.groupby(['Sport', 'Market', 'Bookmaker'], sort=False)
    cum_trials = g_smb.cumcount()
    cum_hits   = g_smb[label_col].cumsum() - out[label_col]

    # Sport/market global prior (exclude current row)
    g_sm = out.groupby(['Sport', 'Market'], sort=False)
    cum_trials_sm = g_sm.cumcount()
    cum_hits_sm   = g_sm[label_col].cumsum() - out[label_col]

    p_global_asof = np.where(
        cum_trials_sm > 0,
        (cum_hits_sm / np.maximum(cum_trials_sm, 1)).astype(float),
        0.5
    )

    alpha0 = prior_strength * p_global_asof
    beta0  = prior_strength * (1.0 - p_global_asof)

    post_mean = (cum_hits + alpha0) / (np.maximum(cum_trials, 0) + alpha0 + beta0)
    post_mean = np.clip(post_mean, 0.01, 0.99)
    post_lift = np.log(post_mean / (1 - post_mean))

    out['Book_Reliability_Score'] = post_mean
    out['Book_Reliability_Lift']  = post_lift

    return out


def build_book_reliability_map(df: pd.DataFrame, prior_strength: float = 200.0) -> pd.DataFrame:
    """
    Returns a slim mapping keyed by (Sport, Market, Bookmaker):
      - Book_Reliability_Score
      - Book_Reliability_Lift

    Compute row-level interactions later, during live apply.
    """
    df_rel = add_book_reliability_features(
        df, label_col="SHARP_HIT_BOOL", prior_strength=prior_strength
    )

    mapping = (
        df_rel
        .groupby(['Sport', 'Market', 'Bookmaker'], as_index=False)
        .agg({
            'Book_Reliability_Score': 'mean',
            'Book_Reliability_Lift':  'mean',
            # Optional diagnostics you can keep if useful:
            # 'Sharp_Line_Magnitude': 'mean'
        })
    )

    # Keep keys normalized
    mapping['Sport'] = mapping['Sport'].astype(str).str.upper()
    mapping['Market'] = mapping['Market'].astype(str).str.lower().str.strip()
    mapping['Bookmaker'] = mapping['Bookmaker'].astype(str).str.lower().str.strip()


    
    return mapping


@st.cache_data(ttl=900, max_entries=32, show_spinner=False)
@st.cache_data(ttl=900, max_entries=32, show_spinner=False)
@st.cache_data(ttl=900, max_entries=32, show_spinner=False)
def build_per_game_power(sport: str, df_bt: pd.DataFrame, df_power: pd.DataFrame) -> pd.DataFrame:
    # latest snapshot per game (to get teams + a timestamp)
    games = (
        df_bt.sort_values('Snapshot_Timestamp')
             .drop_duplicates(subset=['Game_Key'], keep='last')
             [['Game_Key','Sport','Home_Team_Norm','Away_Team_Norm','Game_Start','Snapshot_Timestamp']]
             .copy()
    )

    # normalize games
    games['Sport'] = games['Sport'].astype(str).str.upper()
    games['Home_Team_Norm'] = norm_team(games['Home_Team_Norm'])
    games['Away_Team_Norm'] = norm_team(games['Away_Team_Norm'])
    games['Game_Start'] = pd.to_datetime(games['Game_Start'], utc=True, errors='coerce')
    games['ts'] = games['Game_Start'].fillna(pd.to_datetime(games['Snapshot_Timestamp'], utc=True, errors='coerce'))

    # long-form sides
    home = games[['Game_Key','Sport','Home_Team_Norm','ts']].rename(columns={'Home_Team_Norm':'Team_Norm'})
    home['Side'] = 'home'
    away = games[['Game_Key','Sport','Away_Team_Norm','ts']].rename(columns={'Away_Team_Norm':'Team_Norm'})
    away['Side'] = 'away'
    teams = pd.concat([home, away], ignore_index=True)

    # prep ratings (df_power produced by your cached fetch; it has 'AsOf')
    pr = df_power.copy()
    pr['Sport'] = pr['Sport'].astype(str).str.upper()
    pr['Team_Norm'] = norm_team(pr['Team_Norm'])
    # unify time column to AsOfTS
    if 'AsOfTS' not in pr.columns and 'AsOf' in pr.columns:
        pr = pr.rename(columns={'AsOf':'AsOfTS'})
    pr['AsOfTS'] = pd.to_datetime(pr['AsOfTS'], utc=True, errors='coerce')
    pr = pr.dropna(subset=['AsOfTS'])

    # PERF: keep only teams we actually need
    # inside build_per_game_power
    needed = teams['Team_Norm'].unique()
    pr = pr[pr['Team_Norm'].isin(needed)]


    # sort once for asof
    teams = teams.sort_values(['Sport','Team_Norm','ts'], kind='mergesort')
    pr    = pr.sort_values(['Sport','Team_Norm','AsOfTS'], kind='mergesort')

    # single PIT-safe asof join (backward ⇒ AsOfTS <= ts)
    enriched = pd.merge_asof(
        teams, pr,
        by=['Sport','Team_Norm'],
        left_on='ts', right_on='AsOfTS',
        direction='backward', allow_exact_matches=True
    )[['Game_Key','Side','Power_Rating','PR_Off','PR_Def']]

    # pivot + flatten
    per_game = enriched.pivot(index='Game_Key', columns='Side', values=['Power_Rating','PR_Off','PR_Def'])
    # ensure missing blocks exist (e.g., if PR_Off/PR_Def absent)
    for top in ['Power_Rating','PR_Off','PR_Def']:
        for side in ['home','away']:
            if (top, side) not in per_game.columns:
                per_game[(top, side)] = pd.NA

    per_game = per_game.rename(columns={
        ('Power_Rating','home'): 'Home_Power_Rating',
        ('Power_Rating','away'): 'Away_Power_Rating',
        ('PR_Off','home'):        'Home_PR_Off',
        ('PR_Off','away'):        'Away_PR_Off',
        ('PR_Def','home'):        'Home_PR_Def',
        ('PR_Def','away'):        'Away_PR_Def',
    }).reset_index()

    # handy diffs
    per_game['PR_Rating_Diff_game'] = per_game['Home_Power_Rating'] - per_game['Away_Power_Rating']
    per_game['PR_Off_Diff_game']    = per_game['Home_PR_Off'] - per_game['Away_PR_Off']
    per_game['PR_Def_Diff_game']    = per_game['Home_PR_Def'] - per_game['Away_PR_Def']

    return per_game


    
def train_sharp_model_from_bq(sport: str = "NBA", days_back: int = 35):
    # Dictionary specifying days_back for each sport
    SPORT_DAYS_BACK = {
        'NBA': 35,      # 35 days for NBA
        'NFL': 60,      # 20 days for NFL
        'CFL': 60,      # 20 days for NFL
        'WNBA': 30,     # 30 days for WNBA
        'MLB': 50,      # 50 days for MLB
        'NCAAF': 30,    # 20 days for NCAAF
        'NCAAB': 30,    # 30 days for NCAAB
        # Add more sports as needed
    }

    # Get the days_back from the dictionary, or use the default if sport is not in the dictionary
    days_back = SPORT_DAYS_BACK.get(sport.upper(), days_back)
    
    st.info(f"🎯 Training sharp model for {sport.upper()} with {days_back} days of historical data...")

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
    # Normalize keys
    df_bt['Game_Key'] = df_bt['Game_Key'].astype(str).str.strip().str.lower()
    df_bt['Market'] = df_bt['Market'].astype(str).str.lower().str.strip()
    df_bt['Sport'] = df_bt['Sport'].astype(str).str.upper()
    
    df_bt['Bookmaker'] = df_bt['Bookmaker'].astype(str).str.lower().str.strip()
    
    # ✅ Timestamps (UTC)
    df_bt['Snapshot_Timestamp'] = pd.to_datetime(df_bt['Snapshot_Timestamp'], errors='coerce', utc=True)
    # Use true Game_Start if present; else fall back to Snapshot_Timestamp for ordering
    if 'Game_Start' in df_bt.columns:
        df_bt['Game_Start'] = pd.to_datetime(df_bt['Game_Start'], errors='coerce', utc=True)
    else:
        df_bt['Game_Start'] = df_bt['Snapshot_Timestamp']
    for c in ['Sport', 'Market', 'Bookmaker', 'Outcome', 'Game_Key']:
        if c in df_bt.columns:
            df_bt[c] = df_bt[c].astype('category')

    
    # ✅ Make sure helper won't choke if these are missing
    if 'Is_Sharp_Book' not in df_bt.columns:
        df_bt['Is_Sharp_Book'] = df_bt['Bookmaker'].isin(SHARP_BOOKS).astype(int)
    if 'Sharp_Line_Magnitude' not in df_bt.columns:
        df_bt['Sharp_Line_Magnitude'] = pd.to_numeric(df_bt.get('Line_Delta', 0), errors='coerce').abs().fillna(0)
    # === Get latest snapshot per Game_Key + Market + Outcome (avoid multi-snapshot double counting) ===
    dedup_cols = [
        'Game_Key','Market','Outcome','Bookmaker','Value',
        'Sharp_Move_Signal','Sharp_Limit_Jump','Sharp_Time_Score','Sharp_Limit_Total',
        'Is_Reinforced_MultiMarket','Market_Leader','LimitUp_NoMove_Flag'
    ]
    df_bt = (
        df_bt.sort_values('Snapshot_Timestamp')
             .drop_duplicates(subset=dedup_cols, keep='last')
    )

   
   
    with tmr("reliability features"):
        df_bt = add_book_reliability_features(df_bt, label_col="SHARP_HIT_BOOL", prior_strength=200.0)


    with tmr("pre-agg reliability"):
        rel = (df_bt
               .groupby(['Sport','Market','Bookmaker'], observed=True)['Book_Reliability_Score']
               .mean()
               .reset_index())
    # Optionally only keep top-N per (Sport, Market)
    rel = (rel
           .sort_values(['Sport','Market','Book_Reliability_Score'], ascending=[True, True, False])
           .groupby(['Sport','Market'], observed=True)
           .head(10)
           .reset_index(drop=True))

    # Latest snapshot per market/game/outcome
    # === Get latest snapshot per Game_Key + Market + Outcome ===
    df_latest = (
        df_bt
        .sort_values('Snapshot_Timestamp')
        .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome'], keep='last')
    )
    # ratings once per sport (cached fetch you already have)
  
    # after df_bt is loaded/filtered
    min_ts = pd.to_datetime(df_bt['Game_Start'], utc=True, errors='coerce').min()
    if pd.isna(min_ts):
        min_ts = pd.to_datetime(df_bt['Snapshot_Timestamp'], utc=True, errors='coerce').min()
    max_ts = pd.to_datetime(df_bt['Game_Start'], utc=True, errors='coerce').max()
    if pd.isna(max_ts):
        max_ts = pd.to_datetime(df_bt['Snapshot_Timestamp'], utc=True, errors='coerce').max()
    with tmr("fetch power (cached)"):
        df_power = fetch_power_ratings_from_bq_cached(sport, lookback_days=400, source="history")
        if df_power.empty:
            df_power = fetch_power_ratings_from_bq_cached(sport, source="current")
    with tmr("build per_game power"):
        per_game = build_per_game_power(sport, df_bt, df_power)

    ts_min = pd.to_datetime(df_bt['Game_Start'].min(), utc=True, errors='coerce')
    ts_max = pd.to_datetime(df_bt['Game_Start'].max(), utc=True, errors='coerce')
    
    # Trim to [ts_min - 31d, ts_max] to guarantee a prior rating exists
    pad = pd.Timedelta(days=31)
    df_power = df_power[(df_power['AsOf'] >= ts_min - pad) & (df_power['AsOf'] <= ts_max)]




    # === Pivot line values (e.g., -3.5, 210.5, etc.)
    value_pivot = df_latest.pivot_table(
        index='Game_Key',
        columns='Market',
        values='Value'
    ).rename(columns={
        'spreads': 'Spread_Value',
        'totals': 'Total_Value',
        'h2h': 'H2H_Value'
    })
    
    # === Pivot odds prices (e.g., -110, +100, etc.)
    odds_pivot = df_latest.pivot_table(
        index='Game_Key',
        columns='Market',
        values='Odds_Price'
    ).rename(columns={
        'spreads': 'Spread_Odds',
        'totals': 'Total_Odds',
        'h2h': 'H2H_Odds'
    })
    
    # === Merge line values and odds into one cross-market frame
    df_cross_market = (
        value_pivot
        .join(odds_pivot, how='outer')
        .reset_index()
    )

    

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
    markets = ['spreads', 'totals', 'h2h']
    n_markets = len(markets)
    pb = st.progress(0)  # use int 0–100
    status = st.status("🔄 Training in progress...", expanded=True)

    for idx, market in enumerate(markets, start=1):
        status.write(f"🚧 Training model for `{market.upper()}`...")
        df_market = df_bt[df_bt['Market'] == market].copy()
        df_market = compute_small_book_liquidity_features(df_market)  
        df_market = df_market.merge(df_cross_market, on='Game_Key', how='left')
        if df_market.empty:
            status.warning(f"⚠️ No data for {market.upper()} — skipping.")
            pb.progress(int(round(idx / n_markets * 100)))
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
    
        
       
        
        # === Step 0: Sort once up front
        df_market = df_market.sort_values(['Team', 'Game_Key'])
        
        # === Step 1: Compute LOO Stats with Game_Key
        def compute_loo_stats_by_game(df, home_filter=None):
            df = df.copy()
            if home_filter is not None:
                df = df[df['Is_Home'] == home_filter]
            
            # Deduplicate: one row per (Game_Key, Team) to avoid explosion
            df_dedup = (
                df.sort_values('Snapshot_Timestamp')  # keep earliest per game
                .drop_duplicates(subset=['Game_Key', 'Team'], keep='first')
            )
        
            # Sort to ensure shift aligns chronologically
            df_dedup = df_dedup.sort_values(['Team', 'Game_Key'])
        
            # Shifted cumulative stats for leave-one-out
            df_dedup['cum_model_prob'] = df_dedup.groupby('Team')['Model_Sharp_Win_Prob'].cumsum().shift(1)
            df_dedup['cum_hit'] = df_dedup.groupby('Team')['SHARP_HIT_BOOL'].cumsum().shift(1)
            df_dedup['cum_count'] = df_dedup.groupby('Team').cumcount()
        
            df_dedup['Team_Past_Avg_Model_Prob'] = df_dedup['cum_model_prob'] / df_dedup['cum_count'].replace(0, np.nan)
            df_dedup['Team_Past_Hit_Rate'] = df_dedup['cum_hit'] / df_dedup['cum_count'].replace(0, np.nan)
        
            return df_dedup[['Game_Key', 'Team', 'Team_Past_Avg_Model_Prob', 'Team_Past_Hit_Rate']]


        # === Compute all 3 sets
        overall_stats = compute_loo_stats_by_game(df_market)
        
        # Home-only
        home_stats = compute_loo_stats_by_game(df_market, home_filter=1).rename(columns={
            'Team_Past_Avg_Model_Prob': 'Team_Past_Avg_Model_Prob_Home',
            'Team_Past_Hit_Rate': 'Team_Past_Hit_Rate_Home'
        })
        
        # Away-only
        away_stats = compute_loo_stats_by_game(df_market, home_filter=0).rename(columns={
            'Team_Past_Avg_Model_Prob': 'Team_Past_Avg_Model_Prob_Away',
            'Team_Past_Hit_Rate': 'Team_Past_Hit_Rate_Away'
        })
        
        # === Merge back in
        df_market = df_market.merge(overall_stats, on=['Game_Key', 'Team'], how='left')
        df_market = df_market.merge(home_stats, on=['Game_Key', 'Team'], how='left')
        df_market = df_market.merge(away_stats, on=['Game_Key', 'Team'], how='left')

        # Sort chronologically per team
        # === Ensure data is sorted chronologically
        df_market = df_market.sort_values(['Team', 'Snapshot_Timestamp'])
        
        # Define a dictionary to specify rolling window lengths per sport
        SPORT_COVER_WINDOW = {
            'NBA': 5,  # Example: For NBA, 5 games
            'NFL': 4,  # Example: For NFL, 4 games
            'WNBA': 3,  # Example: For WNBA, 3 games
            'MLB': 7,  # Example: For MLB, 7 games
            'CFL': 4,  # Example: For NFL, 4 games
        }
        
        # === Ensure data is sorted chronologically
        df_market = df_market.sort_values(['Team', 'Snapshot_Timestamp'])
        
        # === Cover Streak (Overall) - Sport-Specific Window Length
        window_length = SPORT_COVER_WINDOW.get(df_market['Sport'].iloc[0], 4)  # Default to 4 if sport is not in the dict
        
        df_market['Team_Recent_Cover_Streak'] = (
            df_market.groupby('Team')['SHARP_HIT_BOOL']
            .transform(lambda x: x.shift().rolling(window=window_length, min_periods=1).sum())
        )
        df_market['On_Cover_Streak'] = (df_market['Team_Recent_Cover_Streak'] >= 2).astype(int)
        
        # === Cover Streak (Home Only) - Sport-Specific Window Length
        df_market['Cover_Home_Only'] = df_market['SHARP_HIT_BOOL'].where(df_market['Is_Home'] == 1)
        
        df_market['Team_Recent_Cover_Streak_Home'] = (
            df_market.groupby('Team')['Cover_Home_Only']
            .transform(lambda x: x.shift().rolling(window=window_length, min_periods=1).sum())
        )
        df_market['On_Cover_Streak_Home'] = (df_market['Team_Recent_Cover_Streak_Home'] >= 2).astype(int)
        
        # === Cover Streak (Away Only) - Sport-Specific Window Length
        df_market['Cover_Away_Only'] = df_market['SHARP_HIT_BOOL'].where(df_market['Is_Home'] == 0)
        
        df_market['Team_Recent_Cover_Streak_Away'] = (
            df_market.groupby('Team')['Cover_Away_Only']
            .transform(lambda x: x.shift().rolling(window=window_length, min_periods=1).sum())
        )
        df_market['On_Cover_Streak_Away'] = (df_market['Team_Recent_Cover_Streak_Away'] >= 2).astype(int)


        if df_market.empty or df_market['SHARP_HIT_BOOL'].nunique() < 2:
            status.warning(f"⚠️ Not enough label variety for {market.upper()} — skipping.")
            pb.progress(int(round(idx / n_markets * 100)))
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
        df_market['Market_Implied_Prob'] = df_market['Odds_Price'].apply(implied_prob)

        df_market['Market_Mispricing'] = df_market['Team_Past_Avg_Model_Prob'] - df_market['Market_Implied_Prob']
        df_market['Abs_Market_Mispricing'] = df_market['Market_Mispricing'].abs()
        df_market['Mispricing_Flag'] = (df_market['Abs_Market_Mispricing'] > 0.05).astype(int)
        
        df_market['Team_Implied_Prob_Gap_Home'] = (
            df_market['Team_Past_Avg_Model_Prob_Home'] - df_market['Market_Implied_Prob']
        )
        
        df_market['Team_Implied_Prob_Gap_Away'] = (
            df_market['Team_Past_Avg_Model_Prob_Away'] - df_market['Market_Implied_Prob']
        )

        df_market['Value_Reversal_Flag'] = df_market.get('Value_Reversal_Flag', 0).fillna(0).astype(int)
        df_market['Odds_Reversal_Flag'] = df_market.get('Odds_Reversal_Flag', 0).fillna(0).astype(int)
        #df_market['Is_Team_Favorite'] = (df_market['Team_Bet_Role'] == 'favorite').astype(int)
        #df_market['Is_Team_Underdog'] = (df_market['Team_Bet_Role'] == 'underdog').astype(int)
        #df_market['Is_Team_Over'] = (df_market['Team_Bet_Role'] == 'over').astype(int)
        #df_market['Is_Team_Under'] = (df_market['Team_Bet_Role'] == 'under').astype(int)
        # === Cross-Market Alignment and Gaps ===
        df_market['Spread_Implied_Prob'] = df_market['Spread_Odds'].apply(implied_prob)
        df_market['H2H_Implied_Prob'] = df_market['H2H_Odds'].apply(implied_prob)
        df_market['Total_Implied_Prob'] = df_market['Total_Odds'].apply(implied_prob)

        # 1. Spread and H2H line alignment (are both favoring same side)
        df_market['Spread_vs_H2H_Aligned'] = (
            (df_market['Spread_Value'] < 0)  &
            (df_market['H2H_Implied_Prob'] > 0.5)
        ).astype(int)

        
        # 2. Total vs Spread directional contradiction
        df_market['Total_vs_Spread_ProbGap'] = df_market['Total_Implied_Prob'] - df_market['Spread_Implied_Prob']
        df_market['Total_vs_Spread_Contradiction'] = (
            (df_market['Spread_Implied_Prob'] > 0.55) &
            (df_market['Total_Implied_Prob'] < 0.48)
        ).astype(int)
        df_market['Spread_vs_H2H_ProbGap'] = df_market['Spread_Implied_Prob'] - df_market['H2H_Implied_Prob']

               
        # 4. Prob gaps
        df_market['Spread_vs_H2H_ProbGap'] = df_market['Spread_Implied_Prob'] - df_market['H2H_Implied_Prob']
        df_market['Total_vs_H2H_ProbGap'] = df_market['Total_Implied_Prob'] - df_market['H2H_Implied_Prob']
        df_market['Total_vs_Spread_ProbGap'] = df_market['Total_Implied_Prob'] - df_market['Spread_Implied_Prob']
        
        # 5. Prob dislocation signal
        df_market['CrossMarket_Prob_Gap_Exists'] = (
            (df_market['Spread_vs_H2H_ProbGap'].abs() > 0.05) |
            (df_market['Total_vs_Spread_ProbGap'].abs() > 0.05)
        ).astype(int)

        
         # Absolute line move
        SPORT_ALIAS = {
            'MLB': 'MLB',
            'NFL': 'NFL',
            'CFL': 'CFL',
            'WNBA': 'WNBA',
            'NBA': 'NBA',
            'NCAAF': 'NCAAF',
            'NCAAB': 'NCAAB',
        }
                # === Sport and Market Normalization (if not already present)
        df_market['Sport_Norm'] = df_market['Sport'].map(SPORT_ALIAS).fillna(df_market['Sport'])
        df_market['Market_Norm'] = df_market['Market'].str.lower()
        
        # === Absolute Line and Odds Movement
        df_market['Abs_Line_Move_From_Opening'] = (df_market['Value'] - df_market['First_Line_Value']).abs()
        df_market['Odds_Shift'] = df_market['Odds_Price'] - df_market['First_Odds']
        df_market['Implied_Prob_Shift'] = (
            calc_implied_prob(df_market['Odds_Price']) - calc_implied_prob(df_market['First_Odds'])
        )
        
        # === Directional Movement Flags
        df_market['Line_Moved_Toward_Team'] = np.where(
            ((df_market['Value'] > df_market['First_Line_Value']) & (df_market['Is_Favorite_Bet'] == 1)) |
            ((df_market['Value'] < df_market['First_Line_Value']) & (df_market['Is_Favorite_Bet'] == 0)),
            1, 0
        )
        
        df_market['Line_Moved_Away_From_Team'] = np.where(
            ((df_market['Value'] < df_market['First_Line_Value']) & (df_market['Is_Favorite_Bet'] == 1)) |
            ((df_market['Value'] > df_market['First_Line_Value']) & (df_market['Is_Favorite_Bet'] == 0)),
            1, 0
        )
        
        # === Percent Move (Totals only)
        df_market['Pct_Line_Move_From_Opening'] = np.where(
            (df_market['Market_Norm'] == 'totals') & (df_market['First_Line_Value'].abs() > 0),
            df_market['Abs_Line_Move_From_Opening'] / df_market['First_Line_Value'].abs(),
            np.nan
        )
        
        df_market['Pct_Line_Move_Bin'] = pd.cut(
            df_market['Pct_Line_Move_From_Opening'],
            bins=[-np.inf, 0.0025, 0.005, 0.01, 0.02, np.inf],
            labels=['<0.25%', '0.5%', '1%', '2%', '2%+']
        )
        
       

        # 🧼 Add this after:
        df_market['Pct_Line_Move_Bin'] = df_market['Pct_Line_Move_Bin'].astype(str)
        
        
        # === Disable line-move-based features in moneyline and MLB spread
        df_market['Disable_Line_Move_Features'] = np.where(
            ((df_market['Sport_Norm'] == 'MLB') & (df_market['Market_Norm'] == 'spread')) |
            (df_market['Market_Norm'].isin(['h2h'])),
            1, 0
        )
        

        # Z-scores for line movement magnitude by sport and market
        df_market['Abs_Line_Move_Z'] = (
            df_market
            .groupby(['Sport_Norm', 'Market_Norm'])['Abs_Line_Move_From_Opening']
            .transform(lambda x: zscore(x.fillna(0), ddof=0))
        )
        
        # For totals: Z-score of percentage movement
        df_market['Pct_Line_Move_Z'] = (
            df_market
            .groupby(['Sport_Norm'])['Pct_Line_Move_From_Opening']
            .transform(lambda x: zscore(x.fillna(0), ddof=0))
        )
        df_market['Abs_Line_Move_Z'] = df_market['Abs_Line_Move_Z'].clip(-5, 5)
        df_market['Pct_Line_Move_Z'] = df_market['Pct_Line_Move_Z'].clip(-5, 5)
        # Spread: Z ≥ 2 (extreme)
        df_market['Potential_Overmove_Flag'] = np.where(
            (df_market['Market_Norm'] == 'spread') &
            (df_market['Line_Moved_Toward_Team'] == 1) &
            (df_market['Abs_Line_Move_Z'] >= 2) &
            (df_market['Disable_Line_Move_Features'] == 0),
            1, 0
        )
        
        # Totals: Z ≥ 2 for % move
        df_market['Potential_Overmove_Total_Pct_Flag'] = np.where(
            (df_market['Market_Norm'] == 'totals') &
            (df_market['Line_Moved_Toward_Team'] == 1) &
            (df_market['Pct_Line_Move_Z'] >= 2) &
            (df_market['Disable_Line_Move_Features'] == 0),
            1, 0
        )
       
        df_market['Implied_Prob_Shift_Z'] = df_market.groupby(['Sport_Norm', 'Market_Norm'])['Implied_Prob_Shift']\
        .transform(lambda x: zscore(x.fillna(0), ddof=0))
        
        df_market['Potential_Odds_Overmove_Flag'] = np.where(
            df_market['Implied_Prob_Shift_Z'] >= 2,
            1, 0
        )
        # === Postgame diagnostics (only used for understanding missed hits — not features)
        df_market['Line_Moved_Toward_Team_And_Missed'] = (
            (df_market['Line_Moved_Toward_Team'] == 1) & (df_market['SHARP_HIT_BOOL'] == 0)
        ).astype(int)
        
        df_market['Line_Moved_Toward_Team_And_Hit'] = (
            (df_market['Line_Moved_Toward_Team'] == 1) & (df_market['SHARP_HIT_BOOL'] == 1)
        ).astype(int)
        
        df_market['Line_Moved_Away_And_Hit'] = (
            (df_market['Line_Moved_Away_From_Team'] == 1) & (df_market['SHARP_HIT_BOOL'] == 1)
        ).astype(int)
        df_market['Book_Reliability_x_Sharp']     = df_market['Book_Reliability_Score'] * df_market['Is_Sharp_Book']
        df_market['Book_Reliability_x_Magnitude'] = df_market['Book_Reliability_Score'] * df_market['Sharp_Line_Magnitude']
    
        df_market['Book_Reliability_x_PROB_SHIFT'] = df_market['Book_Reliability_Score'] * df_market['Is_Sharp_Book'] * df_market['Implied_Prob_Shift']    
        df_market['Book_lift_x_Sharp']     = df_market['Book_Reliability_Lift'] * df_market['Is_Sharp_Book']
        df_market['Book_lift_x_Magnitude'] = df_market['Book_Reliability_Lift'] * df_market['Sharp_Line_Magnitude']
    
        df_market['Book_lift_x_PROB_SHIFT'] = df_market['Book_Reliability_Lift'] * df_market['Is_Sharp_Book'] * df_market['Implied_Prob_Shift'] 
        
        # 0) Fetch once per sport
        df_market = df_market.merge(per_game, on='Game_Key', how='left')

                   
        # ensure normalized for comparison
        df_market['Outcome_Norm']   = norm_team(df_market['Outcome_Norm'])
        df_market['Home_Team_Norm'] = norm_team(df_market['Home_Team_Norm'])
        df_market['Away_Team_Norm'] = norm_team(df_market['Away_Team_Norm'])
        
        is_home_bet = (df_market['Outcome_Norm'] == df_market['Home_Team_Norm'])
        
        df_market['PR_Team_Rating'] = np.where(is_home_bet, df_market['Home_Power_Rating'], df_market['Away_Power_Rating'])
        df_market['PR_Opp_Rating']  = np.where(is_home_bet, df_market['Away_Power_Rating'], df_market['Home_Power_Rating'])
        df_market['PR_Team_Off']    = np.where(is_home_bet, df_market['Home_PR_Off'], df_market['Away_PR_Off'])
        df_market['PR_Team_Def']    = np.where(is_home_bet, df_market['Home_PR_Def'], df_market['Away_PR_Def'])
        df_market['PR_Opp_Off']     = np.where(is_home_bet, df_market['Away_PR_Off'], df_market['Home_PR_Off'])
        df_market['PR_Opp_Def']     = np.where(is_home_bet, df_market['Away_PR_Def'], df_market['Home_PR_Def'])
        
        df_market['PR_Rating_Diff']     = df_market['PR_Team_Rating'] - df_market['PR_Opp_Rating']
        df_market['PR_Abs_Rating_Diff'] = df_market['PR_Rating_Diff'].abs()
        df_market['PR_Total_Est']       = (df_market['PR_Team_Off'] + df_market['PR_Opp_Off']
                                           - df_market['PR_Team_Def'] - df_market['PR_Opp_Def'])

 
                
        features = [
        
            # 🔹 Core sharp signals
            'Sharp_Move_Signal', 'Sharp_Limit_Jump', #'Sharp_Time_Score', 'Book_lift_x_Sharp', 'Book_lift_x_Magnitude', 'Book_lift_x_PROB_SHIFT',
            'Sharp_Limit_Total',
            'Is_Reinforced_MultiMarket', 'Market_Leader', 'LimitUp_NoMove_Flag',
        
            # 🔹 Market response
            'Sharp_Line_Magnitude', #'Is_Home_Team_Bet',
            'Team_Implied_Prob_Gap_Home', 'Team_Implied_Prob_Gap_Away',
        
            # 🔹 Engineered odds shift decomposition
            'SharpMove_Odds_Up', 'SharpMove_Odds_Down', 'SharpMove_Odds_Mag',
        
            # 🔹 Engineered interactions
            'MarketLeader_ImpProbShift', 'LimitProtect_SharpMag', 'Delta_Sharp_vs_Rec',
            #'Sharp_Leads',
            'SharpMove_Resistance_Break',
        
            # 🔹 Resistance feature
            'Line_Resistance_Crossed_Count',  # ✅ newly added here
        
            # 🔁 Reversal logic
            'Value_Reversal_Flag', 'Odds_Reversal_Flag',
        
            # 🔥 Timing flags
            #'Late_Game_Steam_Flag',
            
            #'Abs_Line_Move_From_Opening',
            #'Abs_Odds_Move_From_Opening', 
            'Market_Mispricing', 'Abs_Market_Mispricing',
            'Spread_vs_H2H_Aligned',
            'Total_vs_Spread_Contradiction',
            'Spread_vs_H2H_ProbGap',
            'Total_vs_H2H_ProbGap',
            'Total_vs_Spread_ProbGap',
            'CrossMarket_Prob_Gap_Exists',
            
            'Line_Moved_Away_From_Team',            
            'Pct_Line_Move_From_Opening', 
            'Pct_Line_Move_Bin',
            'Potential_Overmove_Flag', 
            'Potential_Overmove_Total_Pct_Flag', 'Mispricing_Flag',
        
            # 🧠 Cross-market alignment                       
            'Potential_Odds_Overmove_Flag',
            'Line_Moved_Toward_Team',
            'Abs_Line_Move_Z',
            'Pct_Line_Move_Z', 
            'SmallBook_Limit_Skew',
            'SmallBook_Heavy_Liquidity_Flag',
            'SmallBook_Limit_Skew_Flag',
            'Book_Reliability_Score',
            'Book_Reliability_Lift',
            'Book_Reliability_x_Sharp',
            'Book_Reliability_x_Magnitude','Book_Reliability_x_PROB_SHIFT',
            'PR_Team_Rating','PR_Opp_Rating',
            'PR_Rating_Diff','PR_Abs_Rating_Diff',
            'PR_Spread_Est','PR_Spread_Residual',
            'PR_Agrees_With_Favorite',
            'PR_Prob_From_Rating','PR_Prob_Gap_vs_Market'
            
    
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
            # 🔮 Historical team model performance
            'Team_Past_Avg_Model_Prob',
            'Team_Past_Hit_Rate',
            'Team_Past_Avg_Model_Prob_Home',
            'Team_Past_Hit_Rate_Home',
            'Team_Past_Avg_Model_Prob_Away',
            'Team_Past_Hit_Rate_Away',
        
            # 🔥 Recent cover streak stats (overall + home/away)
            'Avg_Recent_Cover_Streak',
            'Avg_Recent_Cover_Streak_Home',
            'Avg_Recent_Cover_Streak_Away',
           
        ]
        df_market = add_time_context_flags(df_market, sport=sport)
        
        # add to features
        features += [
            'Is_Weekend',
            'Is_Night_Game',
            'Is_PrimeTime',      # if you kept it
            'DOW_Sin','DOW_Cos' # if you enabled cyclical
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
            pb.progress(int(round(idx / n_markets * 100)))
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
        # 🚫 Detect and drop nested/bad columns from X_train
        
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
                # ================================
        # === 📊 MODEL STRESS TESTS  ====
        # ================================
       
        st.subheader(f"🧪 Holdout Validation – {market.upper()}")

        # --- Helpers
        def _american_to_roi(odds_series, outcome_bool):
            """
            Unit stake ROI given American odds and binary outcomes.
            ROI = (return - stake)/stake. For a win:
              - If odds > 0: profit = odds/100
              - If odds < 0: profit = 100/|odds|
            For a loss: -1
            """
            odds = pd.to_numeric(odds_series, errors='coerce')
            win = pd.Series(outcome_bool).astype(int)
            pos = (odds > 0).astype(int)

            profit_on_win_pos = odds.where(odds > 0, np.nan) / 100.0
            profit_on_win_neg = 100.0 / odds.abs()
            profit_on_win = np.where(pos == 1, profit_on_win_pos, profit_on_win_neg)
            profit_on_win = pd.Series(profit_on_win).fillna(0.0)

            roi = win * profit_on_win - (1 - win) * 1.0
            return roi

        def _psi(expected, actual, bins=10):
            """
            Population Stability Index for numeric arrays.
            Bins are derived from expected distribution.
            """
            exp = pd.to_numeric(pd.Series(expected).dropna(), errors='coerce')
            act = pd.to_numeric(pd.Series(actual).dropna(), errors='coerce')
            if exp.empty or act.empty:
                return np.nan

            # use quantiles from expected to define bins
            qs = np.linspace(0, 1, bins + 1)
            try:
                cuts = np.unique(np.quantile(exp, qs))
                if len(cuts) < 3:  # not enough spread
                    return 0.0
                exp_bins = pd.cut(exp, cuts, include_lowest=True)
                act_bins = pd.cut(act, cuts, include_lowest=True)
            except Exception:
                return np.nan

            exp_dist = exp_bins.value_counts(normalize=True).reindex(exp_bins.cat.categories, fill_value=0)
            act_dist = act_bins.value_counts(normalize=True).reindex(exp_bins.cat.categories, fill_value=0)

            # avoid log(0)
            exp_dist = exp_dist.replace(0, 1e-6)
            act_dist = act_dist.replace(0, 1e-6)

            return float(((act_dist - exp_dist) * np.log(act_dist / exp_dist)).sum())

        def _grade_psi(v):
            if pd.isna(v):
                return "n/a"
            if v > 0.3:
                return "🚨 heavy drift"
            if v > 0.2:
                return "⚠️ medium drift"
            return "✅ stable"

        # === 1) DRIFT & STABILITY (PSI) =======================================
        st.markdown("### 🔁 Drift & Stability (PSI)")
        # choose most recent 7 days in df_market as "live" window
        if pd.api.types.is_datetime64_any_dtype(df_market['Snapshot_Timestamp']) and not df_market['Snapshot_Timestamp'].isna().all():
            recent_end = df_market['Snapshot_Timestamp'].max()
            recent_start = recent_end - pd.Timedelta(days=7)
            df_recent = df_market[(df_market['Snapshot_Timestamp'] >= recent_start) & (df_market['Snapshot_Timestamp'] <= recent_end)]
        else:
            df_recent = pd.DataFrame()

        psi_rows = []
        if not df_recent.empty:
            X_recent = df_recent.reindex(columns=features)
            # numeric-only for PSI
            X_base = X[features].apply(pd.to_numeric, errors='coerce')
            X_recent_num = X_recent.apply(pd.to_numeric, errors='coerce')
            shared_cols = [c for c in X_base.columns if c in X_recent_num.columns]

            for col in shared_cols:
                try:
                    v = _psi(X_base[col], X_recent_num[col])
                    psi_rows.append((col, v, _grade_psi(v)))
                except Exception:
                    psi_rows.append((col, np.nan, "n/a"))

            df_psi = pd.DataFrame(psi_rows, columns=["Feature", "PSI", "Assessment"]).sort_values("PSI", ascending=False)
            st.dataframe(df_psi.head(30))
        else:
            st.info("No recent window found for PSI (missing or non-datetime Snapshot_Timestamp).")

        # Ensure val_proba is a Series with the X_val index
        try:
            val_proba = pd.Series(val_proba, index=X_val.index)
        except Exception:
            val_proba = pd.Series(cal_logloss.predict_proba(X_val)[:, 1], index=X_val.index)
        
        # Build one aligned frame
        df_eval = pd.DataFrame({
            "p": val_proba,
            "y": pd.Series(y_val, index=val_proba.index),
            "odds": pd.to_numeric(df_market.loc[val_proba.index, "Odds_Price"], errors="coerce")
        })
        
        # Define bins
        bins = np.linspace(0, 1, 11)
        labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
        cuts = pd.cut(df_eval["p"], bins=bins, include_lowest=True, labels=labels)
        
        # Compute per-bin metrics safely
        out_rows = []
        for lb in labels:
            sub = df_eval[cuts == lb]
            n = int(len(sub))
            if n == 0:
                out_rows.append((lb, 0, np.nan, np.nan, np.nan))
                continue
        
            hr = float(sub["y"].mean())
            roi = float(_american_to_roi(sub["odds"], sub["y"]).mean()) if sub["odds"].notna().any() else np.nan
            avg_p = float(sub["p"].mean())
            out_rows.append((lb, n, hr, roi, avg_p))
        
        df_bins = pd.DataFrame(out_rows, columns=["Prob Bin", "N", "Hit Rate", "Avg ROI (unit)", "Avg Pred P"])
        df_bins["N"] = df_bins["N"].astype(int)

        # quick extreme-bucket snapshot
        hi = df_bins.iloc[-1]
        lo = df_bins.iloc[0]
        st.write(f"**High bin ({hi['Prob Bin']}):** N={hi['N']}, Hit={hi['Hit Rate']:.3f}, ROI={hi['Avg ROI (unit)']:.3f}")
        st.write(f"**Low bin  ({lo['Prob Bin']}):** N={lo['N']}, Hit={lo['Hit Rate']:.3f}, ROI={lo['Avg ROI (unit)']:.3f}")

        # === 3) ADVERSE SCENARIO REPLAY ======================================
        st.markdown("### 🌪️ Adverse Scenario Replay (reversal-heavy days)")
        if "Value_Reversal_Flag" in df_market.columns or "Odds_Reversal_Flag" in df_market.columns:
            df_market["_rev_flag"] = (
                df_market.get("Value_Reversal_Flag", 0).fillna(0).astype(int) |
                df_market.get("Odds_Reversal_Flag", 0).fillna(0).astype(int)
            )
            # day-level reversal rate
            if "Snapshot_Timestamp" in df_market.columns and pd.api.types.is_datetime64_any_dtype(df_market["Snapshot_Timestamp"]):
                df_market["_day"] = pd.to_datetime(df_market["Snapshot_Timestamp"], utc=True, errors='coerce').dt.date
                daily = df_market.groupby("_day")["_rev_flag"].mean().sort_values(ascending=False).head(5).index.tolist()
                if len(daily) == 0:
                    st.info("No reversal-heavy days found.")
                else:
                    rows = []
                    for d in daily:
                        mask = df_market["_day"] == d
                        Xd = df_market.loc[mask, features].apply(pd.to_numeric, errors='coerce').fillna(0)
                        yd = df_market.loc[mask, "SHARP_HIT_BOOL"].astype(int)
                        if len(yd.unique()) < 2:
                            continue
                        try:
                            pd_pred = pd.Series(cal_logloss.predict_proba(Xd)[:, 1], index=Xd.index)
                        except Exception:
                            pd_pred = pd.Series(model_logloss.predict_proba(Xd)[:, 1], index=Xd.index)
                        auc_d = roc_auc_score(yd, pd_pred)
                        ll_d = log_loss(yd, pd_pred, labels=[0,1])
                        conf = pd_pred.apply(lambda p: max(p, 1-p)).mean()
                        rows.append((str(d), len(yd), auc_d, ll_d, conf))
                    if rows:
                        df_bad = pd.DataFrame(rows, columns=["Day", "N", "AUC", "LogLoss", "Avg Confidence"])
                        st.dataframe(df_bad)
                    else:
                        st.info("Reversal-heavy days did not have enough label variety for evaluation.")
            else:
                st.info("Missing/invalid Snapshot_Timestamp for adverse replay.")
        else:
            st.info("Reversal flags not present — skipping adverse replay.")

        # === Summary Card =====================================================
        st.markdown("### ✅ Deployment Readiness Snapshot")
        # PSI summary
        psi_flag = "Unknown"
        if 'df_psi' in locals():
            psi_worst = df_psi['PSI'].replace(np.nan, 0).max()
            if psi_worst > 0.3:
                psi_flag = "🚨 heavy drift"
            elif psi_worst > 0.2:
                psi_flag = "⚠️ medium drift"
            else:
                psi_flag = "✅ stable"
        # extreme bin checks
        hi_ok = (hi["N"] > 0) and (hi["Hit Rate"] >= 0.60 or hi["Avg ROI (unit)"] > 0)
        lo_ok = (lo["N"] > 0) and ((1 - lo["Hit Rate"]) >= 0.60 or lo["Avg ROI (unit)"] > 0)

        st.write(f"- **PSI status:** {psi_flag}")
        st.write(f"- **High-confidence bucket profitable/accurate?** {'✅' if hi_ok else '⚠️'}")
        st.write(f"- **Low-confidence bucket fade profitable/accurate?** {'✅' if lo_ok else '⚠️'}")
        # 🕵️‍♂️ Debug: Inspect problematic features in X
        for col in X.columns:
            try:
                sample_val = X[col].dropna().iloc[0]
            except IndexError:
                continue  # Skip empty columns
        
            if isinstance(sample_val, (pd.DataFrame, pd.Series, list, dict)):
                print(f"❌ Feature '{col}' has bad type: {type(sample_val)} — value: {sample_val}")
            elif not hasattr(sample_val, 'dtype'):
                print(f"⚠️ Feature '{col}' has unknown/non-numeric type: {type(sample_val)} — value: {sample_val}")
        # 🚫 Detect and drop nested/bad columns from X
        # 🧹 Clean X BEFORE any predict_proba calls or calibrator usage
        bad_cols = []
        for col in X.columns:
            sample_val = X[col].dropna().iloc[0] if not X[col].dropna().empty else None
            if isinstance(sample_val, (pd.DataFrame, pd.Series, list, dict)):
                print(f"❌ Feature '{col}' has bad type: {type(sample_val)} — value: {sample_val}")
                bad_cols.append(col)
        
        if bad_cols:
            print(f"🧹 Dropping from X: {bad_cols}")
            X = X.drop(columns=bad_cols)
        
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
        # Filter only active features (Importance > 0)
        active_features = importance_df[importance_df['Importance'] > 0]
        
        # Use dataframe for scrollable view if there are many features
        if len(active_features) > 30:
            st.dataframe(active_features.reset_index(drop=True))
        else:
            st.table(active_features)
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
                # === Base performance features
                'Model_Sharp_Win_Prob': 'mean',
                'SHARP_HIT_BOOL': 'mean',
        
                # === Home/Away past hit rate and prob
                'Team_Past_Avg_Model_Prob_Home': 'mean',
                'Team_Past_Hit_Rate_Home': 'mean',
                'Team_Past_Avg_Model_Prob_Away': 'mean',
                'Team_Past_Hit_Rate_Away': 'mean',
        
                # === Recent cover streak features (overall + home/away)
                'Team_Recent_Cover_Streak': 'mean',
                'On_Cover_Streak': 'mean',
                'Team_Recent_Cover_Streak_Home': 'mean',
                'On_Cover_Streak_Home': 'mean',
                'Team_Recent_Cover_Streak_Away': 'mean',
                'On_Cover_Streak_Away': 'mean'
            })
            .rename(columns={
                'Model_Sharp_Win_Prob': 'Team_Past_Avg_Model_Prob',
                'SHARP_HIT_BOOL': 'Team_Past_Hit_Rate',
                'Team_Recent_Cover_Streak': 'Avg_Recent_Cover_Streak',
                'On_Cover_Streak': 'Pct_On_Recent_Cover_Streak',
                'Team_Recent_Cover_Streak_Home': 'Avg_Recent_Cover_Streak_Home',
                'On_Cover_Streak_Home': 'Pct_On_Recent_Cover_Streak_Home',
                'Team_Recent_Cover_Streak_Away': 'Avg_Recent_Cover_Streak_Away',
                'On_Cover_Streak_Away': 'Pct_On_Recent_Cover_Streak_Away'
            })
            .reset_index()
        )


        book_reliability_map = build_book_reliability_map(df_bt, prior_strength=200.0)


        # === Save ensemble (choose one or both)
        trained_models[market] = {
            "model": model_auc,
            "calibrator": cal_auc,
            "team_feature_map": team_feature_map,
            "book_reliability_map": book_reliability_map  # ✅ include for in-memory use
        }

        save_model_to_gcs(model_auc, cal_auc, sport, market, bucket_name=GCS_BUCKET, team_feature_map=team_feature_map,book_reliability_map=book_reliability_map)
        from scipy.stats import entropy
        
       
        st.success(f"""✅ Trained + saved ensemble model for {market.upper()}
        - AUC: {auc:.4f}
        - Accuracy: {acc:.4f}
        - Log Loss: {logloss:.4f}
        - Brier Score: {brier:.4f}
        """)
        pb.progress(int(round(idx / n_markets * 100)))

    status.update(label="✅ All models trained", state="complete", expanded=False)
    if not trained_models:
        st.error("❌ No models were trained.")
    return trained_models



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


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd

def train_timing_opportunity_model(sport: str = "NBA", days_back: int = 35):
    st.info(f"🧠 Training timing opportunity models for {sport.upper()}...")

    # === Load historical scored data
    query = f"""
        SELECT *
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE UPPER(Sport) = '{sport.upper()}'
          AND Scored = TRUE
          AND SHARP_HIT_BOOL IS NOT NULL
          AND DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
    """
    df = bq_client.query(query).to_dataframe()
    if df.empty:
        st.warning("⚠️ No historical sharp picks available.")
        return

    # Normalize types/casing used below
    df = df.copy()
    df['Market'] = df['Market'].astype(str).str.lower().str.strip()
    df['SHARP_HIT_BOOL'] = pd.to_numeric(df['SHARP_HIT_BOOL'], errors='coerce').fillna(0).astype(int)

    # If Model_Sharp_Win_Prob is missing, create a neutral column
    if 'Model_Sharp_Win_Prob' not in df.columns:
        df['Model_Sharp_Win_Prob'] = 0.0
    else:
        df['Model_Sharp_Win_Prob'] = pd.to_numeric(df['Model_Sharp_Win_Prob'], errors='coerce').fillna(0.0)

    markets = ['spreads', 'totals', 'h2h']
    for market in markets:
        df_market = df[df['Market'] == market].copy()
        if df_market.empty:
            st.warning(f"⚠️ No rows for market: {market}")
            continue

        # Ensure numeric for movement columns (they exist in your schema)
        for c in ['Abs_Line_Move_From_Opening', 'Abs_Odds_Move_From_Opening']:
            if c not in df_market.columns:
                df_market[c] = 0.0
            df_market[c] = pd.to_numeric(df_market[c], errors='coerce').fillna(0.0)

        # Label: “good timing” when we hit + (both moves not already extreme) OR the model was strong
        mv_thresh  = df_market['Abs_Line_Move_From_Opening'].quantile(0.80)
        odd_thresh = df_market['Abs_Odds_Move_From_Opening'].quantile(0.80)
        df_market['TIMING_OPPORTUNITY_LABEL'] = (
            (df_market['SHARP_HIT_BOOL'] == 1) &
            (
                ((df_market['Abs_Line_Move_From_Opening'] < mv_thresh) &
                 (df_market['Abs_Odds_Move_From_Opening'] < odd_thresh))
                |
                (df_market['Model_Sharp_Win_Prob'] > 0.60)
            )
        ).astype(int)

        # Feature list (present in your schema)
        base_feats = [
            'Abs_Line_Move_From_Opening',
            'Abs_Odds_Move_From_Opening',
            'Late_Game_Steam_Flag'
        ]
        sharp_blocks = [
            'Overnight_VeryEarly','Overnight_MidRange','Overnight_LateGame','Overnight_Urgent',
            'Early_VeryEarly','Early_MidRange','Early_LateGame','Early_Urgent',
            'Midday_VeryEarly','Midday_MidRange','Midday_LateGame','Midday_Urgent',
            'Late_VeryEarly','Late_MidRange','Late_LateGame','Late_Urgent'
        ]
        timing_feats = [f'SharpMove_Magnitude_{b}' for b in sharp_blocks]
        odds_feats   = [f'OddsMove_Magnitude_{b}' for b in sharp_blocks]

        # Keep only features that actually exist
        feature_cols = [c for c in base_feats + timing_feats + odds_feats if c in df_market.columns]
        if not feature_cols:
            st.warning(f"⚠️ No usable features for {market} — skipping.")
            continue

        X = df_market[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        y = df_market['TIMING_OPPORTUNITY_LABEL']

        # Safety: enough samples & both classes present & enough per-class for cv=5
        if y.nunique() < 2:
            st.warning(f"⚠️ Not enough variation in label for {market} — skipping.")
            continue
        n_min_class = min((y == 0).sum(), (y == 1).sum())
        if len(y) < 50 or n_min_class < 5:
            st.warning(f"⚠️ Not enough samples for robust CV (n={len(y)}, min_class={n_min_class}) — skipping {market}.")
            continue

        # Train calibrated model
        base = GradientBoostingClassifier()
        calibrated = CalibratedClassifierCV(base, method='isotonic', cv=5)
        calibrated.fit(X, y)

        # Save the timing model; it does NOT need book_reliability_map
        save_model_to_gcs(
            model=calibrated,           # store under "model" slot
            calibrator=None,            # no separate calibrator object
            sport=sport,
            market=f"timing_{market}",  # namespaced market
            bucket_name=GCS_BUCKET,
            team_feature_map=None,      # not used here
            book_reliability_map=None   # keep explicit to avoid NameError
        )
        st.success(f"✅ Timing model saved for {market.upper()}")

    
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

    # Group and collect snapshot history of model probabilities
    
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


        # --- make sure aliases/defaults exist *before* build_why runs
    if 'Model Prob' not in df.columns and 'Model_Sharp_Win_Prob' in df.columns:
        df['Model Prob'] = df['Model_Sharp_Win_Prob']
    
    # sensible default for Passes_Gate if it's missing or NaN (prevents one side showing "Still Calculating")
    core_flags = [
        'Sharp_Move_Signal','Sharp_Limit_Jump','Market_Leader',
        'Is_Reinforced_MultiMarket','LimitUp_NoMove_Flag','Is_Sharp_Book'
    ]
    for c in core_flags:
        if c not in df.columns:
            df[c] = 0
    df['Passes_Gate'] = (
        df.get('Passes_Gate')
        .fillna((df[core_flags].fillna(0).astype(int).sum(axis=1) > 0))
        if 'Passes_Gate' in df.columns else
        (df[core_flags].fillna(0).astype(int).sum(axis=1) > 0)
    )
    
    def build_why(row):
        model_prob = row.get('Model Prob')
        if pd.isna(model_prob):
            return "⚠️ Missing — run apply_blended_sharp_score() first"
    
        parts = []
    
        # --- Core sharp move reasons (cast to bool so 0/1 ints work)
        if bool(row.get('Sharp_Move_Signal', 0)): parts.append("📈 Sharp Move Detected")
        if bool(row.get('Sharp_Limit_Jump', 0)): parts.append("💰 Limit Jumped")
        if bool(row.get('Market_Leader', 0)): parts.append("🏆 Market Leader")
        if bool(row.get('Is_Reinforced_MultiMarket', 0)): parts.append("📊 Multi-Market Consensus")
        if bool(row.get('LimitUp_NoMove_Flag', 0)): parts.append("🛡️ Limit Up + No Line Move")
        if bool(row.get('Is_Sharp_Book', 0)): parts.append("🎯 Sharp Book Signal")  # ← typo fixed
    
        # --- Odds & line movement
        if bool(row.get('SharpMove_Odds_Up', 0)): parts.append("🟢 Odds Moved Up (Steam)")
        if bool(row.get('SharpMove_Odds_Down', 0)): parts.append("🔻 Odds Moved Down (Buyback)")
        if float(row.get('Sharp_Line_Magnitude', 0)) > 0.5: parts.append("📏 Big Line Move")
        if float(row.get('Rec_Line_Magnitude', 0)) > 0.5: parts.append("📉 Rec Book Move")
        if float(row.get('Sharp_Limit_Total', 0)) > 10000: parts.append("💼 Sharp High Limit")
        if float(row.get('SharpMove_Odds_Mag', 0)) > 5: parts.append("💥 Sharp Odds Steam")
    
        # --- Resistance & timing
        if bool(row.get('SharpMove_Resistance_Break', 0)): parts.append("🧱 Broke Key Resistance")
        if bool(row.get('Late_Game_Steam_Flag', 0)): parts.append("⏰ Late Game Steam")
        if bool(row.get('Value_Reversal_Flag', 0)): parts.append("🔄 Value Reversal")
        if bool(row.get('Odds_Reversal_Flag', 0)): parts.append("📉 Odds Reversal")
        if float(row.get('Sharp_Time_Score', 0)) > 0.5: parts.append("⏱️ Timing Edge")
    
        # --- Team-level
        if float(row.get('Team_Past_Hit_Rate', 0)) > 0.5: parts.append("⚔️📊 Team Historically Sharp")
        if float(row.get('Team_Past_Avg_Model_Prob', 0)) > 0.5: parts.append("🔮 Model Favored This Team Historically")
        if float(row.get('Avg_Recent_Cover_Streak', 0)) >= 2: parts.append("🔥 Recent Hot Streak")
        if float(row.get('Avg_Recent_Cover_Streak_Home', 0)) >= 2: parts.append("🏠🔥 Home Streaking")
        if float(row.get('Avg_Recent_Cover_Streak_Away', 0)) >= 2: parts.append("✈️🔥 Road Streaking")
    
        # --- From open
        if float(row.get('Abs_Line_Move_From_Opening', 0)) > 1.0: parts.append("📈 Line Moved from Open")
        if float(row.get('Abs_Odds_Move_From_Opening', 0)) > 5.0: parts.append("💹 Odds Moved from Open")
    
        # --- Cross-market + diagnostics
        if bool(row.get('Spread_vs_H2H_Aligned', 0)): parts.append("🧩 Spread and H2H Align")
        if bool(row.get('Total_vs_Spread_Contradiction', 0)): parts.append("⚠️ Total Contradicts Spread")
        if bool(row.get('CrossMarket_Prob_Gap_Exists', 0)): parts.append("🔀 Cross-Market Probability Gap")
        if bool(row.get('Potential_Overmove_Flag', 0)): parts.append("📊 Line Possibly Overmoved")
        if bool(row.get('Potential_Overmove_Total_Pct_Flag', 0)): parts.append("📉 Total Possibly Overmoved")
        if bool(row.get('Potential_Odds_Overmove_Flag', 0)): parts.append("🎯 Odds Possibly Overmoved")
        if bool(row.get('Line_Moved_Toward_Team', 0)): parts.append("🧲 Line Moved Toward This Team")
        if bool(row.get('Line_Moved_Away_From_Team', 0)): parts.append("🚫 Line Moved Away From This Team")
        if float(row.get('Line_Resistance_Crossed_Count', 0)) >= 1: parts.append("🧱 Crossed Resistance Levels")
        if float(row.get('Abs_Line_Move_Z', 0)) > 1: parts.append("📊 Unusual Line Z-Move")
        if float(row.get('Pct_Line_Move_Z', 0)) > 1: parts.append("📈 Abnormal % Line Z-Score")
        if bool(row.get('Mispricing_Flag', 0)): parts.append("💸 Market Mispricing Detected")
        if bool(row.get('Hybrid_Line_Timing_Flag', 0)): parts.append("⏱️ Sharp Line Timing Bucket")
        if bool(row.get('Hybrid_Odds_Timing_Flag', 0)): parts.append("🕰️ Sharp Odds Timing Bucket")
        if bool(row.get('Is_Weekend', 0)): parts.append("📅 Weekend Game")
        if bool(row.get('Is_Night_Game', 0)): parts.append("🌙 Night Game")
        if bool(row.get('Is_PrimeTime', 0)): parts.append("⭐ Prime Time Matchup")
    
        # --- Hybrid timing buckets
        HYBRID_LINE_COLS = [
            'SharpMove_Magnitude_Overnight_VeryEarly','SharpMove_Magnitude_Overnight_MidRange',
            'SharpMove_Magnitude_Overnight_LateGame','SharpMove_Magnitude_Overnight_Urgent',
            'SharpMove_Magnitude_Early_VeryEarly','SharpMove_Magnitude_Early_MidRange',
            'SharpMove_Magnitude_Early_LateGame','SharpMove_Magnitude_Early_Urgent',
            'SharpMove_Magnitude_Midday_VeryEarly','SharpMove_Magnitude_Midday_MidRange',
            'SharpMove_Magnitude_Midday_LateGame','SharpMove_Magnitude_Midday_Urgent',
            'SharpMove_Magnitude_Late_VeryEarly','SharpMove_Magnitude_Late_MidRange',
            'SharpMove_Magnitude_Late_LateGame','SharpMove_Magnitude_Late_Urgent'
        ]
        HYBRID_ODDS_COLS = [
            'OddsMove_Magnitude_Overnight_VeryEarly','OddsMove_Magnitude_Overnight_MidRange',
            'OddsMove_Magnitude_Overnight_LateGame','OddsMove_Magnitude_Overnight_Urgent',
            'OddsMove_Magnitude_Early_VeryEarly','OddsMove_Magnitude_Early_MidRange',
            'OddsMove_Magnitude_Early_LateGame','OddsMove_Magnitude_Early_Urgent',
            'OddsMove_Magnitude_Midday_VeryEarly','OddsMove_Magnitude_Midday_MidRange',
            'OddsMove_Magnitude_Midday_LateGame','OddsMove_Magnitude_Midday_Urgent',
            'OddsMove_Magnitude_Late_VeryEarly','OddsMove_Magnitude_Late_MidRange',
            'OddsMove_Magnitude_Late_LateGame','OddsMove_Magnitude_Late_Urgent'
        ]
        EMOJI_MAP = {'Overnight':'🌙','Early':'🌅','Midday':'🌞','Late':'🌆'}
    
        for col in HYBRID_LINE_COLS:
            if float(row.get(col, 0)) > 0.25:
                bucket = col.replace('SharpMove_Magnitude_', '').replace('_', ' ')
                epoch = col.split('_')[2]  # Overnight/Early/Midday/Late
                parts.append(f"{EMOJI_MAP.get(epoch, '⏱️')} {bucket} Sharp Move")
    
        for col in HYBRID_ODDS_COLS:
            if float(row.get(col, 0)) > 0.5:
                bucket = col.replace('OddsMove_Magnitude_', '').replace('_', ' ')
                epoch = col.split('_')[2]
                parts.append(f"{EMOJI_MAP.get(epoch, '⏱️')} {bucket} Odds Steam")
    
        # If we truly have nothing and gate is false, show the waiting message
        if not parts and not bool(row.get('Passes_Gate', False)):
            return "🕓 Still Calculating Signal"
    
        return " + ".join(parts) if parts else "🤷‍♂️ Still Calculating"

# keep this AFTER the aliasing above
    df['Why Model Likes It'] = df.apply(build_why, axis=1)

    # Apply to DataFrame
   # === Model_Confidence_Tier for summary compatibility
    df['Model_Confidence_Tier'] = df['Confidence Tier']
    
 
    # === Load Timing Opportunity Model
    # === Load All Timing Models by Market
    # === Load Timing Opportunity Models by Market
    timing_models = {}
    for m in ['spreads', 'totals', 'h2h']:
        try:
            model_data = load_model_from_gcs(sport=sport, market=f"timing_{m}", bucket_name=GCS_BUCKET)
            timing_models[m] = model_data.get("calibrator") or model_data.get("model")
        except Exception:
            timing_models[m] = None  # fallback if missing
        
    # === Timing Features
    timing_feature_cols = [
        'Abs_Line_Move_From_Opening', 'Abs_Odds_Move_From_Opening', 'Late_Game_Steam_Flag'
    ] + [
        f'SharpMove_Magnitude_{b}' for b in [
            'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
            'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
            'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
            'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
        ]
    ] + [
        f'OddsMove_Magnitude_{b}' for b in [
            'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
            'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
            'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
            'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
        ]
    ]
    
    # Ensure all features exist
    for col in timing_feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    from sklearn.cluster import KMeans
   # Ensure all required features exist
    for col in timing_feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    df['Timing_Opportunity_Score'] = np.nan
    for m in ['spreads', 'totals', 'h2h']:
        model = timing_models.get(m)
        if model is None:
            continue
        mask = df['Market'].str.lower() == m
        if mask.any():
            X = df.loc[mask, timing_feature_cols].fillna(0)
            df.loc[mask, 'Timing_Opportunity_Score'] = model.predict_proba(X)[:, 1]
        
    if 'Timing_Opportunity_Score' in df.columns:
        scores = df[['Timing_Opportunity_Score']].copy()
        scores['Timing_Opportunity_Score'] = scores['Timing_Opportunity_Score'].clip(0, 1).fillna(0)
    
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        kmeans.fit(scores)
    
        df['Timing_Cluster'] = kmeans.labels_
    
        cluster_order = np.argsort(kmeans.cluster_centers_.ravel())
        cluster_map = {
            cluster_order[0]: "🔴 Late / Overmoved",
            cluster_order[1]: "🟡 Developing",
            cluster_order[2]: "🟢 Smart Timing"
        }
        df['Timing_Stage'] = df['Timing_Cluster'].map(cluster_map)
    else:
        df['Timing_Stage'] = "⚠️ Unavailable"
        
    # === Final Output
    diagnostics_df = df[[
        'Game_Key', 'Market', 'Outcome', 'Bookmaker',
        'Tier_Change', 'Confidence Trend', 'Line/Model Direction',
        'Why Model Likes It', 'Passes_Gate', 'Confidence Tier',
        'Model Prob Snapshot', 'First Prob Snapshot',
        'Model_Confidence_Tier',
        'Timing_Opportunity_Score',  # ✅
        'Timing_Stage'   # ✅ Add this
    ]].rename(columns={
        'Tier_Change': 'Tier Δ'
    })


    
    return diagnostics_df
def _as_num_seq(x):
    """Coerce x into a clean numeric list (drop NaNs). Handles scalars, lists, arrays, and stringified lists/CSV."""
    if x is None:
        return []
    # If it's already a non-string list-like, use it directly
    if is_list_like(x) and not isinstance(x, (str, bytes, dict)):
        seq = list(x)
    else:
        # Try to parse strings like "[0.1, 0.2]" or "0.1,0.2"
        if isinstance(x, str):
            s = x.strip()
            parsed = None
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = json.loads(s)
                except Exception:
                    parsed = None
            if parsed is None:
                try:
                    parsed = [float(tok) for tok in re.split(r'[,\s]+', s) if tok]
                except Exception:
                    parsed = []
            seq = parsed
        else:
            # Scalar (float/int/np.float)
            seq = [x]
    # Coerce to numeric and drop NaNs
    return pd.to_numeric(pd.Series(seq), errors='coerce').dropna().tolist()
# === Global utility for creating a sparkline with the full trend history
# === Global utility for creating a sparkline with the full trend history

def create_sparkline_html_safe(probs):
    # normalize to numeric list
    if probs is None or (isinstance(probs, (float, int, np.floating, np.integer)) and pd.isna(probs)):
        vals = []
    elif isinstance(probs, (list, tuple, np.ndarray, pd.Series)):
        vals = pd.to_numeric(pd.Series(list(probs)), errors='coerce').dropna().tolist()
    elif isinstance(probs, str):
        s = probs.strip()
        parsed = None
        if s.startswith('[') and s.endswith(']'):
            try:
                parsed = json.loads(s)
            except Exception:
                parsed = None
        if parsed is None:
            try:
                parsed = [float(tok) for tok in re.split(r'[,\s]+', s) if tok]
            except Exception:
                parsed = []
        vals = pd.to_numeric(pd.Series(parsed), errors='coerce').dropna().tolist()
    else:
        try:
            vals = [float(probs)]
        except Exception:
            vals = []

    if len(vals) < 2:
        return "—"

    # tooltip + spark
    labels = [f"{round(p * 100, 1)}%" for p in vals]
    tooltip = escape(" → ".join(labels), quote=True)
    chars = "⎽⎼⎻⎺"
    lo, hi = min(vals), max(vals)
    if lo == hi:
        spark = chars[len(chars)//2] * len(vals)
    else:
        scaled = np.interp(vals, (lo, hi), (0, len(chars) - 1))
        spark = ''.join(chars[int(round(i))] for i in scaled)
    return f'<span title="{tooltip}" style="cursor: help;">{spark}</span>'


# --- helper to normalize a history cell into a numeric list
def _normalize_history(x):
    if x is None or (isinstance(x, (float, int, np.floating, np.integer)) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return pd.to_numeric(pd.Series(list(x)), errors='coerce').dropna().tolist()
    if isinstance(x, str):
        s = x.strip()
        try:
            if s.startswith('[') and s.endswith(']'):
                parsed = json.loads(s)
            else:
                parsed = [float(tok) for tok in re.split(r'[,\s]+', s) if tok]
        except Exception:
            parsed = []
        return pd.to_numeric(pd.Series(parsed), errors='coerce').dropna().tolist()
    try:
        return [float(x)]
    except Exception:
        return []

   

from google.cloud import storage
from io import BytesIO
import pickle
import logging

def save_model_to_gcs(model, calibrator, sport, market, 
                      team_feature_map=None, 
                      book_reliability_map=None, 
                      bucket_name="sharp-models"):
    """
    Save a trained model, calibrator, and optional feature maps to Google Cloud Storage.

    Parameters:
    - model: Trained XGBoost or sklearn model
    - calibrator: CalibratedClassifierCV or similar
    - sport (str): e.g., "nfl"
    - market (str): e.g., "spreads"
    - team_feature_map (pd.DataFrame or None): Optional team-level stats used in scoring
    - book_reliability_map (dict or pd.DataFrame or None): Optional bookmaker reliability scores
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

        if book_reliability_map is not None:
            payload["book_reliability_map"] = book_reliability_map

        # Serialize to bytes
        buffer = BytesIO()
        pickle.dump(payload, buffer)
        buffer.seek(0)

        # Upload to GCS
        blob.upload_from_file(buffer, content_type='application/octet-stream')

        print(
            f"✅ Model + calibrator"
            f"{' + team features' if team_feature_map is not None else ''}"
            f"{' + book reliability map' if book_reliability_map is not None else ''} "
            f"saved to GCS: gs://{bucket_name}/{filename}"
        )

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





def render_scanner_tab(label, sport_key, container, force_reload=False):

    if st.session_state.get("pause_refresh", False):
        st.info("⏸️ Auto-refresh paused")
        return


    timestamp = pd.Timestamp.utcnow()

    with container:
        st.subheader(f"📡 Scanning {label} Sharp Signals")

        # === Live odds snapshot
        live = fetch_live_odds(sport_key)
        if not live:
            st.warning("⚠️ No live odds returned.")
            return pd.DataFrame()

       
    

        from utils import normalize_book_name

        df_snap = pd.DataFrame([
            {
                'Game_ID': game.get('id'),
                'Game': f"{game.get('home_team')} vs {game.get('away_team')}",
                'Game_Start': pd.to_datetime(game.get("commence_time"), utc=True),
                
                # ✅ Normalized key used for Bookmaker column
                'Bookmaker': normalize_book_name(book.get('key', ''), book.get('key', '')),
                'Book': normalize_book_name(book.get('key', ''), book.get('key', '')),  # same as Bookmaker
        
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

        df_all_snapshots = get_recent_history()

        # === Load sharp moves from BigQuery (from Cloud Scheduler or live)
       
        detection_key = f"sharp_moves_{sport_key.lower()}"
        if not force_reload and detection_key in st.session_state:
            df_moves_raw = st.session_state[detection_key]
            st.info(f"✅ Using cached sharp moves for {label}")
        else:
            with st.spinner(f"📥 Loading sharp moves for {label} from BigQuery..."):
                df_moves_raw = read_recent_sharp_moves_conditional(force_reload=force_reload, hours=48)
                st.session_state[detection_key] = df_moves_raw
                st.success(f"✅ Loaded {len(df_moves_raw)} sharp move rows from BigQuery")

        # === Filter to current tab's sport
        df_moves_raw['Sport_Norm'] = df_moves_raw['Sport'].astype(str).str.upper().str.strip()
        df_moves_raw = df_moves_raw[df_moves_raw['Sport_Norm'] == label.upper()]
       
        
       
        # ✅ Snapshot log
        #st.write("📦 Total raw rows loaded from BigQuery:", len(df_moves_raw))
        #st.dataframe(df_moves_raw.head(3))
        
        # === Defensive check before build_game_key
        required_cols = ['Game', 'Game_Start', 'Market', 'Outcome']
        missing = [col for col in required_cols if col not in df_moves_raw.columns]
        
        if df_moves_raw.empty:
            st.warning("⚠️ No graded picks available yet. I’ll still show live odds below.")
            skip_grading = True
        
        if missing:
            st.error(f"❌ Required columns missing before build_game_key: {missing}")
            st.dataframe(df_moves_raw.head())
            skip_grading = True

                
        if 'Snapshot_Timestamp' in df_moves_raw.columns and not df_moves_raw.empty:
            # Use a tighter subset to avoid shape/nullable weirdness
            dedup_cols = [c for c in [
                'Home_Team_Norm','Away_Team_Norm','Game_Start','Market','Outcome',
                'Bookmaker','Value','Odds_Price','Limit'
            ] if c in df_moves_raw.columns]
        
            before = len(df_moves_raw)
            df_moves_raw = df_moves_raw.sort_values('Snapshot_Timestamp', ascending=False)
            df_moves_raw = df_moves_raw.drop_duplicates(subset=dedup_cols, keep='first')
            after = len(df_moves_raw)
            st.info(f"🧹 Deduplicated {before - after} snapshot rows (kept latest per unique pick).")
        
        # === Build game keys (for merging)
        df_moves_raw = build_game_key(df_moves_raw)
                
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
    
        #alias_map = {
            #'Model_Sharp_Win_Prob': 'Model Prob',
            #'Model_Confidence_Tier': 'Confidence Tier',
        #}
        #for orig, alias in alias_map.items():
            #if orig in df_moves_raw.columns and alias not in df_moves_raw.columns:
                #df_moves_raw[alias] = df_moves_raw[orig]

       
       
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
        
       # === Step 6: Resolve _x/_y conflicts (include First_Sharp_Prob)
        for col in ['First_Sharp_Prob', 'First_Model_Prob', 'First_Line_Value', 'First_Tier']:
            y_col = f"{col}_y"
            x_col = f"{col}_x"
            if y_col in df_pre.columns:
                # prefer the merged (right) side
                df_pre[col] = df_pre[y_col]
                df_pre.drop(columns=[y_col], inplace=True)
            if x_col in df_pre.columns:
                df_pre.drop(columns=[x_col], inplace=True)
        
        df_pre.drop(columns=['_merge'], errors='ignore', inplace=True)

        # Backfill First_Sharp_Prob from First_Model_Prob if missing
        first_sharp = pd.to_numeric(df_pre.get('First_Sharp_Prob', np.nan), errors='coerce')
        first_model = pd.to_numeric(df_pre.get('First_Model_Prob', np.nan), errors='coerce')
        df_pre['First_Sharp_Prob'] = first_sharp.combine_first(first_model)


        
        # === Optional: Normalize again (safety for downstream groupby)
        df_pre['Bookmaker'] = df_pre['Bookmaker'].str.lower()
        df_pre['Outcome'] = df_pre['Outcome'].astype(str).str.strip().str.lower()
        
        # === Debug/Preview Other Tables
        
        
        # === Step 8: Rename columns for display
        df_pre.rename(columns={
                'Game': 'Matchup',
              
        }, inplace=True)
     # === Step 8: Rename columns for display (safe, duplicate-proof)
        # If 'Model Prob' already exists, drop the source instead of renaming into it
        if 'Model Prob' in df_pre.columns and 'Model_Sharp_Win_Prob' in df_pre.columns:
            df_pre.drop(columns=['Model_Sharp_Win_Prob'], inplace=True)
        else:
            if 'Model_Sharp_Win_Prob' in df_pre.columns:
                df_pre.rename(columns={'Model_Sharp_Win_Prob': 'Model Prob'}, inplace=True)
        
        if 'Confidence Tier' in df_pre.columns and 'Model_Confidence_Tier' in df_pre.columns:
            df_pre.drop(columns=['Model_Confidence_Tier'], inplace=True)
        else:
            if 'Model_Confidence_Tier' in df_pre.columns:
                df_pre.rename(columns={'Model_Confidence_Tier': 'Confidence Tier'}, inplace=True)
           
       

        # === Preview & column check
        st.write("📋 Columns in df_pre:", df_pre.columns.tolist())
        
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
        
              
           
        # === 2) Create df_summary_base as latest-per-book rows ===
        df_summary_base = (
            df_pre
              .sort_values('Snapshot_Timestamp')
              .drop_duplicates(subset=['Game_Key','Market','Outcome','Bookmaker'], keep='last')
              .copy()
        )
        

        # === Sharp-book average Model Prob & Tier (safe) ===
        
        # 0) Make sure a base prob exists (row-level)
        if 'Model Prob' not in df_summary_base.columns:
            if 'Model_Sharp_Win_Prob' in df_summary_base.columns:
                df_summary_base['Model Prob'] = pd.to_numeric(df_summary_base['Model_Sharp_Win_Prob'], errors='coerce')
            else:
                df_summary_base['Model Prob'] = np.nan
        
        # 1) Normalize keys
        for k in ['Bookmaker','Market','Outcome']:
            df_summary_base[k] = df_summary_base[k].astype(str).str.strip().str.lower()
        
        # 2) Build sharp-only average map
        df_sharp = df_summary_base[df_summary_base['Bookmaker'].isin(SHARP_BOOKS)].copy()
        if not df_sharp.empty:
            model_prob_map = (
                df_sharp
                .groupby(['Game_Key','Market','Outcome'], as_index=False)['Model Prob']
                .mean()
                .rename(columns={'Model Prob':'Model_Prob_SharpAvg'})
            )
        else:
            model_prob_map = pd.DataFrame(columns=['Game_Key','Market','Outcome','Model_Prob_SharpAvg'])
        st.write("📋 Columns in df_summary_base_mid:", df_summary_base.columns.tolist())
        # 3) Merge sharp-avg (or create empty column)
        df_summary_base = df_summary_base.drop(columns=['Model_Prob_SharpAvg'], errors='ignore')
        if not model_prob_map.empty:
            df_summary_base = df_summary_base.merge(
                model_prob_map, on=['Game_Key','Market','Outcome'],
                how='left', validate='many_to_one', suffixes=('','_dup')
            )
            if 'Model_Prob_SharpAvg_dup' in df_summary_base.columns:
                a = pd.to_numeric(df_summary_base['Model_Prob_SharpAvg'], errors='coerce')
                b = pd.to_numeric(df_summary_base['Model_Prob_SharpAvg_dup'], errors='coerce')
                df_summary_base['Model_Prob_SharpAvg'] = a.where(a.notna(), b)
                df_summary_base.drop(columns=['Model_Prob_SharpAvg_dup'], inplace=True)
        else:
                    df_summary_base['Model_Prob_SharpAvg'] = np.nan
        # --- Helper: safely fetch a column as a 1-D Series, even if duplicated or missing
        def _safe_series(df, col, default=np.nan):
            if col in df.columns:
                idxs = [i for i, c in enumerate(df.columns) if c == col]
                if len(idxs) == 1:
                    s = df.iloc[:, idxs[0]]
                else:
                    # multiple columns with the same name → coerce numeric then take first non-null across dups
                    s = (
                        df.iloc[:, idxs]
                          .apply(pd.to_numeric, errors='coerce')
                          .bfill(axis=1)
                          .iloc[:, 0]
                    )
            else:
                s = pd.Series(default, index=df.index)
            return pd.to_numeric(s, errors='coerce')
        
        # --- Build "sharp avg" series and base series safely
        sharp_avg = _safe_series(df_summary_base, 'Model_Prob_SharpAvg', default=np.nan)
        base_prob = _safe_series(df_summary_base, 'Model Prob', default=np.nan)
        
        # Prefer sharp_avg when available; otherwise keep existing/base
        df_summary_base['Model Prob'] = sharp_avg.combine_first(base_prob)
        
        # Tiering stays the same
        df_summary_base['Confidence Tier'] = df_summary_base['Model Prob'].apply(tier_from_prob)
        df_summary_base['Model_Confidence_Tier'] = df_summary_base['Confidence Tier']
     


        st.write("📋 Columns in df_summary_base_end:", df_summary_base.columns.tolist())
        # === 4) STEP 3: safely hydrate Sharp/Rec/First_Line_Value via skinny merge (avoid row-order fillna) ===
        skinny_lines = (
            df_pre[['Game_Key','Market','Outcome','Sharp Line','Rec Line','First_Line_Value']]
            .drop_duplicates(subset=['Game_Key','Market','Outcome'], keep='last')
        )
        df_summary_base = df_summary_base.merge(
            skinny_lines, on=['Game_Key','Market','Outcome'], how='left', suffixes=('','_src')
        )
        for c in ['Sharp Line','Rec Line','First_Line_Value']:
            src = f'{c}_src'
            if src in df_summary_base.columns:
                df_summary_base[c] = df_summary_base[c].where(df_summary_base[c].notna(), df_summary_base[src])
        df_summary_base.drop(columns=[c for c in df_summary_base.columns if c.endswith('_src')], inplace=True)
        
        # === 5) Movement calcs (use numeric) ===
        for col in ['Sharp Line','Rec Line','First_Line_Value']:
            if col in df_summary_base.columns:
                df_summary_base[col] = pd.to_numeric(df_summary_base[col], errors='coerce')
        df_summary_base['Sharp Move'] = (df_summary_base['Sharp Line'] - df_summary_base['First_Line_Value']).round(2)
        df_summary_base['Rec Move']   = (df_summary_base['Rec Line']  - df_summary_base['First_Line_Value']).round(2)
        
        # === 6) Aggregated sharp-only trend + spark (from df_all_snapshots) ===
        # === 6) Aggregated sharp-only trend + spark (from df_all_snapshots) ===
        # Ensure we have a consistent prob column in snapshots
        if 'Model Prob' not in df_all_snapshots.columns and 'Model_Sharp_Win_Prob' in df_all_snapshots.columns:
            df_all_snapshots['Model Prob'] = pd.to_numeric(df_all_snapshots['Model_Sharp_Win_Prob'], errors='coerce')
        else:
            df_all_snapshots['Model Prob'] = pd.to_numeric(df_all_snapshots.get('Model Prob'), errors='coerce')
        
        # Sharp books only
        snap_sharp = df_all_snapshots[df_all_snapshots['Bookmaker'].str.lower().isin(SHARP_BOOKS)].copy()
        snap_sharp['Snapshot_Timestamp'] = pd.to_datetime(snap_sharp['Snapshot_Timestamp'], errors='coerce', utc=True)
        
        # --- Single source of truth: per-timestamp sharp-book AVERAGE ---
        sharp_ts_avg = (
            snap_sharp
              .dropna(subset=['Snapshot_Timestamp'])
              .sort_values('Snapshot_Timestamp')
              .groupby(['Game_Key','Market','Outcome','Snapshot_Timestamp'], as_index=False)['Model Prob']
              .mean()  # avg across sharp books at each timestamp
        )
        
        # First sharp prob (from the same averaged series)
        first_prob_map = (
            sharp_ts_avg
              .groupby(['Game_Key','Market','Outcome'], as_index=False)['Model Prob']
              .first()
              .rename(columns={'Model Prob':'First_Sharp_Prob_Agg'})
        )
        
        # Full trend list (from the same averaged series)
        trend_history_sharp = (
            sharp_ts_avg
              .groupby(['Game_Key','Market','Outcome'])['Model Prob']
              .apply(list)
              .reset_index(name='Prob_Trend_List_Agg')
        )
        
        # Merge onto summary
        df_summary_base = df_summary_base.merge(first_prob_map, on=['Game_Key','Market','Outcome'], how='left')
        df_summary_base = df_summary_base.merge(trend_history_sharp, on=['Game_Key','Market','Outcome'], how='left')
        
        # Confidence Trend from the SAME series as the sparkline (first→current)
        # --- trend text helper (expects a list already)
        def _trend_text_from_list(lst, current):
            if not isinstance(lst, list) or len(lst) == 0 or pd.isna(current):
                return "⚠️ Missing"
            start = lst[0]
            if pd.isna(start):
                return "⚠️ Missing"
            if current == start:
                return f"↔ Stable: {start:.1%} → {current:.1%}"
            arrow = "📈 Trending Up" if current > start else "📉 Trending Down"
            return f"{arrow}: {start:.1%} → {current:.1%}"
        
        # ===== Use ONE dataframe consistently (df_summary_base) =====
        # Normalize the history column USED BELOW
        hist_col = "Prob_Trend_List_Agg"   # <- this is the column you actually use for the spark and trend
        df_summary_base[hist_col] = df_summary_base[hist_col].apply(_normalize_history)
        
        # Current prob (already set earlier)
        _prob_now = pd.to_numeric(df_summary_base['Model Prob'], errors='coerce')
        
        # Confidence Trend (no len() on floats anywhere)
        df_summary_base['Confidence Trend'] = [
            _trend_text_from_list(lst, cur) for lst, cur in zip(df_summary_base[hist_col], _prob_now)
        ]
        
        # Confidence Spark using the SAME normalized history + safe sparkline
        MAX_SPARK_POINTS = 36
        df_summary_base['Confidence Spark'] = (
            df_summary_base[hist_col]
              .apply(lambda lst: lst[-MAX_SPARK_POINTS:] if isinstance(lst, list) else [])
              .apply(create_sparkline_html_safe)
        )
        
        # === 7) STEP 4: select representative book row (sharp-first, latest) ===
        df_summary_base['Book_Is_Sharp'] = df_summary_base['Bookmaker'].str.lower().isin(SHARP_BOOKS).astype(int)
        df_summary_base = (
            df_summary_base
            .sort_values(['Book_Is_Sharp','Snapshot_Timestamp'], ascending=[False, False])
            .drop_duplicates(subset=['Game_Key','Market','Outcome'], keep='first')
            .drop(columns=['Book_Is_Sharp'])
        )
        
        # === 8) STEP 5: per-book diagnostics built on latest-per-book sharp rows from df_pre, then attach ===
        diag_source = (
            df_pre
              .sort_values('Snapshot_Timestamp')
              .drop_duplicates(subset=['Game_Key','Market','Outcome','Bookmaker'], keep='last')
        )
        diag_source = diag_source[diag_source['Bookmaker'].str.lower().isin(SHARP_BOOKS)].copy()
        
        if diag_source.empty:
            st.warning("⚠️ No sharp book rows available for diagnostics.")
            for col in ['Tier Δ','Line/Model Direction','Why Model Likes It']:
                df_summary_base[col] = "⚠️ Missing"
        else:
            # normalize keys before merge
            for df_ in (df_summary_base, diag_source):
                for k in ['Game_Key','Market','Outcome','Bookmaker']:
                    df_[k] = df_[k].astype(str).str.strip().str.lower()
        
            diagnostics_df = compute_diagnostics_vectorized(diag_source)  # per-book diagnostics
        
            rep = df_summary_base[['Game_Key','Market','Outcome','Bookmaker']].drop_duplicates()
            diagnostics_pick = diagnostics_df.merge(
                rep, on=['Game_Key','Market','Outcome','Bookmaker'], how='inner'
            )[[
                'Game_Key','Market','Outcome','Bookmaker',
                'Tier Δ','Line/Model Direction','Why Model Likes It',
                'Timing_Opportunity_Score','Timing_Stage'  # <-- add these
            ]]
        
            df_summary_base = df_summary_base.merge(
                diagnostics_pick,
                on=['Game_Key','Market','Outcome','Bookmaker'],
                how='left'
            )
            for col in ['Tier Δ','Line/Model Direction','Why Model Likes It']:
                df_summary_base[col] = df_summary_base[col].fillna("⚠️ Missing")
        
        # === 9) (Optional) merge team features and small-book liquidity AFTER the dedupe ===
        if team_feature_map is not None and not team_feature_map.empty:
            df_summary_base['Team'] = df_summary_base['Outcome'].astype(str).str.strip().str.lower()
            df_summary_base = df_summary_base.merge(team_feature_map, on='Team', how='left')
        
        df_pre = compute_small_book_liquidity_features(df_pre)
        sb_cols = [
            'Game_Key','Market','Outcome',
            'SmallBook_Total_Limit','SmallBook_Max_Limit','SmallBook_Min_Limit',
            'SmallBook_Limit_Skew','SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag'
        ]
        sb_skinny = (
            df_pre[sb_cols]
            .drop_duplicates(subset=['Game_Key','Market','Outcome'], keep='last')
        )
        df_summary_base = df_summary_base.merge(sb_skinny, on=['Game_Key','Market','Outcome'], how='left')
        #st.write( df_summary_base.columns.tolist())
        
        # === 10) Build summary_df with selected columns ===
        summary_cols = [
            'Matchup','Market','Game_Start','Outcome',
            'Rec Line','Sharp Line','Rec Move','Sharp Move',
            'Model Prob','Confidence Tier',
            'Confidence Trend','Confidence Spark','Line/Model Direction','Tier Δ','Why Model Likes It',
            'Game_Key','Snapshot_Timestamp','Timing_Stage','Timing_Opportunity_Score'
        ]
        summary_df = df_summary_base[[c for c in summary_cols if c in df_summary_base.columns]].copy()
        
        # time/formatting
        summary_df['Game_Start'] = pd.to_datetime(summary_df['Game_Start'], errors='coerce', utc=True)
        summary_df = summary_df[summary_df['Game_Start'].notna()]
        summary_df['Date + Time (EST)'] = summary_df['Game_Start'].dt.tz_convert('US/Eastern').dt.strftime('%Y-%m-%d %I:%M %p')
        summary_df['Event_Date_Only'] = summary_df['Game_Start'].dt.date.astype(str)
        
        # cleanup
        summary_df.columns = summary_df.columns.str.replace(r'_x$|_y$|_scored$', '', regex=True)
        summary_df = summary_df.loc[:, ~summary_df.columns.duplicated()]

        
        # === 🔍 Diagnostic: Check for duplicate Game × Market × Outcome
        # === 🔍 Diagnostic: Check for duplicate Game × Market × Outcome
        # === 🔍 Diagnostic: Check for duplicate Game × Market × Outcome
        


        # Optional: final sort if needed
        #summary_df.sort_values(by=['Game_Start', 'Matchup', 'Market'], inplace=True)
        
   
        # === Build Market + Date Filters
        market_options = ["All"] + sorted(summary_df['Market'].dropna().unique())
        selected_market = st.selectbox(f"📊 Filter {label} by Market", market_options, key=f"{label}_market_summary")
        
        date_only_options = ["All"] + sorted(summary_df['Event_Date_Only'].dropna().unique())
        selected_date = st.selectbox(f"📅 Filter {label} by Date", date_only_options, key=f"{label}_date_filter")
        
        filtered_df = summary_df.copy()
        
        # ✅ UI filters (run before any normalization)
        if selected_market != "All":
            filtered_df = filtered_df[filtered_df['Market'] == selected_market]
        if selected_date != "All":
            filtered_df = filtered_df[filtered_df['Event_Date_Only'] == selected_date]
        


      
        #st.write("🧪 Columns ibefore soummary group:")
        #st.write(filtered_df.columns.tolist())
            
        
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
                'Timing_Opportunity_Score': 'first',
                'Timing_Stage': 'first', 
                'Confidence Tier': lambda x: x.mode().iloc[0] if not x.mode().empty else (x.iloc[0] if not x.empty else "⚠️ Missing"),
                'Confidence Trend': 'first',
                'Tier Δ': 'first',
                'Line/Model Direction': 'first',
                'Why Model Likes It': 'first',
                'Confidence Spark':'first'
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
            'Model Prob', 'Confidence Tier', 'Timing_Stage',
            'Why Model Likes It', 'Confidence Trend','Confidence Spark', 'Tier Δ', 
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
                            "Bookmaker": normalize_book_name(book.get("key", ""), book.get("key", "")),  # ✅ normalized key only
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
sport = st.sidebar.radio("Select a League", ["General", "NFL", "NCAAF", "NBA", "MLB", "CFL", "WNBA"])

st.sidebar.markdown("### ⚙️ Controls")
st.sidebar.checkbox("⏸️ Pause Auto Refresh", key="pause_refresh")
force_reload = st.sidebar.button("🔁 Force Reload")

# --- Optional: Track scanner checkboxes by sport
scanner_flags = {
    "NFL": "run_nfl_scanner",
    "NCAAF": "run_ncaaf_scanner",
    "NBA": "run_nba_scanner",
    "MLB": "run_mlb_scanner",
    "CFL": "run_cfl_scanner",
    "WNBA": "run_wnba_scanner",
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
        train_timing_opportunity_model(sport=label)
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
        
        
