
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
from pandas.util import hash_pandas_object

from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import xgboost as xgb  # make sure this import exists
import re
import logging
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import RandomizedSearchCV
from itertools import product
from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV, TimeSeriesSplit
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint, loguniform, uniform
from sklearn.metrics import log_loss, make_scorer


              
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
# Sharp limit anchor(s)
SHARP_BOOKS_FOR_LIMITS = ['pinnacle']

# Full sharp set (includes Betfair Exchange regions)
SHARP_BOOKS = [
    'pinnacle', 'betus', 'mybookieag', 'smarkets',
    'betfair_ex_eu', 'betfair_ex_uk', 'betfair_ex_au',
    'lowvig', 'betonlineag', 'matchbook'
]

# Recreational set (deduped)
REC_BOOKS = [
    'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers',
    'fanatics', 'espnbet', 'hardrockbet', '888sport',
    'williamhillus', 'ballybet', 'bet365_au', 'betopenly'
]

_BOOK_ALIASES = {
    # canonical : accepted raw keys
    "pinnacle":     {"pinnacle", "pinny", "pinn"},
    "betus":        {"betus"},
    "mybookieag":   {"mybookieag", "mybookie"},
    "smarkets":     {"smarkets"},
    "lowvig":       {"lowvig"},
    "betonlineag":  {"betonlineag", "betonline"},
    "matchbook":    {"matchbook"},

    # betfair exchange + regions
    "betfair_ex_eu": {"betfair_ex_eu"},
    "betfair_ex_uk": {"betfair_ex_uk"},
    "betfair_ex_au": {"betfair_ex_au"},
    "betfair":       {"betfair", "betfair_exchange", "betfair-exchange", "betfair exchange", "betfair_ex"},

    # rec aliases
    "betmgm":        {"betmgm", "mgm"},
    "bet365":        {"bet365"},
    "bet365_au":     {"bet365_au"},
    "draftkings":    {"draftkings", "dk"},
    "fanduel":       {"fanduel", "fd"},
    "betrivers":     {"betrivers"},
    "fanatics":      {"fanatics"},
    "espnbet":       {"espnbet"},
    "hardrockbet":   {"hardrockbet", "hardrock"},
    "888sport":      {"888sport", "sport888", "888-sport", "888 sport"},
    "williamhillus": {"williamhillus", "william_hill_us", "williamhill_us"},
    "ballybet":      {"ballybet"},
    "betopenly":     {"betopenly"},
}

def _canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    return s

def _alias_lookup(s: str) -> str:
    s = _canon(s)
    for canon, alts in _BOOK_ALIASES.items():
        if s in alts or s == canon:
            return canon
    return s
def normalize_book_name(bookmaker: str, book: str) -> str:
    book = book.lower().strip() if isinstance(book, str) else ""
    bookmaker = bookmaker.lower().strip() if isinstance(bookmaker, str) else ""

    if bookmaker == "betfair" and book.startswith("betfair_ex_"):
        return book  # e.g., betfair_ex_uk

    return bookmaker
def normalize_book_and_bookmaker(book_key: str, bookmaker_key: str | None = None) -> tuple[str, str]:
    """
    Returns (book_norm, bookmaker_norm).
    - Preserve regional Betfair Exchange keys exactly for BOTH (e.g., 'betfair_ex_uk').
    - Otherwise map via aliases.
    """
    book_raw = _canon(book_key)
    bm_raw   = _canon(bookmaker_key if bookmaker_key is not None else book_key)

    if book_raw.startswith("betfair_ex_"):
        return book_raw, book_raw

    return _alias_lookup(book_raw), _alias_lookup(bm_raw)
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
   
    'Is_Reinforced_MultiMarket': 'Win Rate by Cross-Market Reinforcement',
    'Market_Leader': 'Win Rate by Market Leader',
    'LimitUp_NoMove_Flag': 'Win Rate by Limit↑ No Move'
})


SPORT_ALIASES = {
    "MLB":   ["MLB", "BASEBALL_MLB", "BASEBALL-MLB", "BASEBALL"],
    "NFL":   ["NFL", "AMERICANFOOTBALL_NFL", "FOOTBALL_NFL"],
    "NCAAF": ["NCAAF", "AMERICANFOOTBALL_NCAAF", "CFB"],
    "NBA":   ["NBA", "BASKETBALL_NBA", "BASKETBALL-NBA"],
    "WNBA":  ["WNBA", "BASKETBALL_WNBA"],
    "CFL":   ["CFL", "CANADIANFOOTBALL", "CANADIANFOOTBALL_CFL"],
    # extend as needed
}

def _is_num(x):
    x = pd.to_numeric(x, errors="coerce")
    return pd.notna(x) and np.isfinite(x)

def _fmt_signed(x, nd=1, suffix=""):
    x = float(x)
    return f"{x:+.{nd}f}{suffix}"

def _fmt_pct(x, nd=0):
    x = float(x)
    return f"{x:.{nd}%}"

def _append_metric(parts, label, value, fmt="signed", nd=1, suffix=""):
    if _is_num(value):
        if fmt == "signed":
            parts.append(f"{label} {_fmt_signed(value, nd, suffix)}")
        elif fmt == "pct":
            parts.append(f"{label} {_fmt_pct(value, nd)}")
        else:
            parts.append(f"{label} {value}")

def _aliases_for(s: str | None) -> list[str]:
    if not s:
        return []
    canon = str(s).upper().strip()
    return SPORT_ALIASES.get(canon, [canon])

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


   
#def read_from_bigquery(table_name):
    #try:
        #client = bigquery.Client()
        #return client.query(f"SELECT * FROM `{table_name}`").to_dataframe()
    #except Exception as e:
        #st.error(f"❌ Failed to load `{table_name}`: {e}")
        #return pd.DataFrame()
        
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


def read_recent_sharp_moves(
    hours=24,
    table = "sharplogger.sharp_data.sharp_moves_master",
    sport: str | None = None
):
    try:
        client = bq_client
        hours_int = int(hours)

        time_where = f"Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours_int} HOUR)"
        aliases = _aliases_for(sport)
        if aliases:
            sports_in = ", ".join([f"'{a}'" for a in aliases])
            sport_where = f"UPPER(TRIM(Sport)) IN ({sports_in})"
            where_clause = f"{time_where} AND {sport_where}"
        else:
            where_clause = time_where

        query = f"""
            SELECT *
            FROM `{table}`
            WHERE {where_clause}
        """
        df = client.query(query).to_dataframe()

        if "Commence_Hour" in df.columns:
            df["Commence_Hour"] = pd.to_datetime(df["Commence_Hour"], errors="coerce", utc=True)
        if "Sport" in df.columns:
            df["Sport_Norm"] = df["Sport"].astype(str).str.upper().str.strip()

        print(f"✅ Loaded {len(df)} rows from BigQuery (last {hours_int}h"
              + (f", sport={sport}" if sport else ", all sports") + ")")
        return df
    except Exception as e:
        print(f"❌ Failed to read from BigQuery: {e}")
        return pd.DataFrame()



# ✅ Cached wrapper for diagnostics and line movement history
# Smart getter — use cache unless forced to reload

# Optional: restrict allowed tables if you truly want this configurable.
DEFAULT_TABLE = "sharplogger.sharp_data.sharp_moves_master"
ALLOWED_TABLES = {DEFAULT_TABLE}  # add more if needed

bq_client = bigquery.Client()

def _validate_table(table: str) -> str:
    tbl = table or DEFAULT_TABLE
    if tbl not in ALLOWED_TABLES:
        # Fallback to default to avoid SQL injection via table name
        tbl = DEFAULT_TABLE
    return tbl

@st.cache_data(ttl=600)
def read_recent_sharp_moves_cached(
    hours: int = 24,
    sport: str | None = None,
    table: str = "sharplogger.sharp_data.sharp_moves_master",
):
    # Validate/whitelist the table name at call time
    tbl = _validate_table(table)  # must return a backticked identifier or a known-safe string

    # Build query with optional sport filter
    sport_filter = "AND Sport = @sport" if sport else ""
    query = f"""
        SELECT *
        FROM {tbl}
        WHERE Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @hours HOUR)
        {sport_filter}
        ORDER BY Snapshot_Timestamp DESC
    """

    params = [bigquery.ScalarQueryParameter("hours", "INT64", int(hours))]
    if sport:
        params.append(bigquery.ScalarQueryParameter("sport", "STRING", sport))

    job = bq_client.query(query, job_config=bigquery.QueryJobConfig(query_parameters=params))
    df = job.to_dataframe()
    return df

@st.cache_data(ttl=600)
def get_recent_history(hours: int = 24, sport: str | None = None):
    st.write("📦 Using cached sharp history (get_recent_history)")
    return read_recent_sharp_moves_cached(hours=hours, sport=sport, table=DEFAULT_TABLE)

def read_recent_sharp_moves_conditional(
    hours: int = 24,
    sport: str | None = None,
    force_reload: bool = False,
    table: str = DEFAULT_TABLE,
):
    if force_reload:
        read_recent_sharp_moves_cached.cache_clear()
    return read_recent_sharp_moves_cached(hours=hours, sport=sport, table=table)
    

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

from sklearn.model_selection import BaseCrossValidator, RandomizedSearchCV



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

def sanitize_features(X: pd.DataFrame) -> pd.DataFrame:

    from pandas.api.types import (
        is_bool_dtype, is_numeric_dtype, is_categorical_dtype
    )

    out = X.copy()

    # Flatten MultiIndex cols if any
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ['__'.join(map(str, t)).strip() for t in out.columns.to_list()]

    # Drop duplicate columns (keep first)
    dup_mask = out.columns.duplicated(keep='first')
    if dup_mask.any():
        # Optional: log which ones
        # print("Dropping duplicate feature columns:", out.columns[dup_mask].tolist())
        out = out.loc[:, ~dup_mask]

    # Coerce types column-wise
    for c in out.columns:
        s = out[c]
        if is_bool_dtype(s):
            out[c] = s.astype('int8')
        elif is_categorical_dtype(s):
            out[c] = s.cat.codes.astype('int32')
        elif is_numeric_dtype(s):
            out[c] = pd.to_numeric(s, errors='coerce')
        else:
            # Try best-effort numeric; otherwise drop
            out[c] = pd.to_numeric(s, errors='coerce')

    # Replace infs, fill NaN, downcast
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.astype('float32', copy=False)
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
# --- IMPORTS (must be before helpers) ---
import streamlit as st

from pandas.api.types import is_categorical_dtype
from google.cloud import bigquery
import gc

# --- CACHED CLIENT (resource-level) ---
@st.cache_resource
def get_bq() -> bigquery.Client:
    return bigquery.Client(project="sharplogger")

                     
def _iso(ts) -> str:
    return pd.to_datetime(ts, utc=True).isoformat()




# ---------- Tiny helper to parse table id safely ----------
def _parse_table_id(project_default: str, table_id: str):
    """
    Accepts:
      - project.dataset.table
      - `project.dataset.table`
      - dataset.table  (fills project from project_default)
    Returns: (project, dataset, table)
    """
    tid = str(table_id).strip().strip("`").strip()
    parts = [p for p in tid.split(".") if p]

    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        if not project_default:
            raise ValueError(f"Missing project for '{table_id}'. Provide project or use a 3-part id.")
        return project_default, parts[0], parts[1]
    else:
        raise ValueError(f"Invalid table id '{table_id}'. Expected dataset.table or project.dataset.table.")

# --- 1) Cached helper: resolve rating column (trivial for this schema) ---
@st.cache_data(show_spinner=False)
def _resolve_rating_col(table_fq: str, project: str | None = None) -> str:
    """
    Your ratings_history table uses 'Rating' (FLOAT).
    We still confirm via INFORMATION_SCHEMA in case the casing changes.
    """
    bq = bigquery.Client(project=project) if project else bigquery.Client()
    proj, ds, tbl = _parse_table_id(bq.project, table_fq)

    sql = f"""
        SELECT column_name
        FROM `{proj}.{ds}.INFORMATION_SCHEMA.COLUMNS`
        WHERE LOWER(table_name) = LOWER(@tname)
    """
    cols = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("tname", "STRING", tbl)]
        ),
    ).to_dataframe()["column_name"].str.lower().tolist()

    if "rating" in cols:
        return "Rating"  # return actual casing you store

    raise RuntimeError(
        f"'Rating' column not found in {proj}.{ds}.{tbl}. "
        f"Available (lowercased): {cols}"
    )

# --- 2) Cached main fetch: parameterized, dedup per day, schema-safe ---
@st.cache_data(show_spinner=True, ttl=600)
def fetch_training_ratings_window_cached(
    sport: str,
    start_iso: str,
    end_iso: str,
    table_history: str = "sharplogger.sharp_data.ratings_history",
    project: str = "sharplogger",
    method_filter: str | None = None,   # optional
    source_filter: str | None = None,   # optional
):
    """
    Returns one row per (Sport, Team_Norm, day) with the latest rating as-of that day.
    Columns: Sport, Team_Norm, AsOfTS, Power_Rating
    """
    bq = bigquery.Client(project=project)
    rating_col = _resolve_rating_col(table_history, project)

    proj, ds, tbl = _parse_table_id(bq.project, table_history)
    table_fq = f"{proj}.{ds}.{tbl}"

    # Optional filters (Method/Source) as SQL fragments + parameters
    where_extra = []
    params = [
        bigquery.ScalarQueryParameter("sport",    "STRING", sport.upper()),
        bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", pd.to_datetime(start_iso).to_pydatetime()),
        bigquery.ScalarQueryParameter("end_ts",   "TIMESTAMP", pd.to_datetime(end_iso).to_pydatetime()),
    ]
    if method_filter:
        where_extra.append("AND Method = @method")
        params.append(bigquery.ScalarQueryParameter("method", "STRING", method_filter))
    if source_filter:
        where_extra.append("AND Source = @source")
        params.append(bigquery.ScalarQueryParameter("source", "STRING", source_filter))

    sql = f"""
      WITH base AS (
        SELECT
          UPPER(Sport) AS Sport,
          LOWER(TRIM(Team)) AS Team_Norm,
          SAFE_CAST(Updated_At AS TIMESTAMP) AS AsOfTS,
          CAST(`{rating_col}` AS FLOAT64) AS Power_Rating
        FROM `{table_fq}`
        WHERE UPPER(Sport) = @sport
          AND SAFE_CAST(Updated_At AS TIMESTAMP) BETWEEN @start_ts AND @end_ts
          {' '.join(where_extra)}
      ),
      dedup AS (
        SELECT *,
               ROW_NUMBER() OVER (
                 PARTITION BY Sport, Team_Norm, DATE(AsOfTS)
                 ORDER BY AsOfTS DESC
               ) AS rn
        FROM base
      )
      SELECT Sport, Team_Norm, AsOfTS, Power_Rating
      FROM dedup
      WHERE rn = 1
    """

    # Optional dry run
    bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False, query_parameters=params),
    ).result()

    df = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(use_query_cache=True, query_parameters=params),
    ).result().to_dataframe()

    if df.empty:
        return df

    df = df[["Sport", "Team_Norm", "AsOfTS", "Power_Rating"]].copy()
    df["AsOfTS"] = pd.to_datetime(df["AsOfTS"], utc=True, errors="coerce")
    df["Power_Rating"] = pd.to_numeric(df["Power_Rating"], errors="coerce").astype("float32")
    df["Sport"] = df["Sport"].astype(str)
    df["Team_Norm"] = df["Team_Norm"].astype(str)
    return df




# --- TRAINING ENRICHMENT (LEAK-SAFE) ---


def enrich_power_for_training_lowmem(
    df: pd.DataFrame,
    bq,                           # not used inside, kept for signature parity
    sport_aliases: dict,
    table_history: str = "sharplogger.sharp_data.ratings_history",
    pad_days: int = 10,
    allow_forward_hours: float = 0.0,  # 0 = strict backward-only
    project: str = None,
) -> pd.DataFrame:
    
    from pandas.api.types import is_categorical_dtype

    if df.empty:
        return df

    out = df.copy()

    # --- normalize inputs (strings + time) ---
    out['Sport'] = out['Sport'].astype(str).str.upper()
    out['Home_Team_Norm'] = out['Home_Team_Norm'].astype(str).str.lower().str.strip()
    out['Away_Team_Norm'] = out['Away_Team_Norm'].astype(str).str.lower().str.strip()
    out['Game_Start'] = pd.to_datetime(out['Game_Start'], utc=True, errors='coerce')

    # sport alias → canon
    canon = {}
    for k, v in sport_aliases.items():
        if isinstance(v, list):
            for a in v: canon[str(a).upper()] = str(k).upper()
        else:
            canon[str(k).upper()] = str(v).upper()
    out['Sport'] = out['Sport'].map(lambda s: canon.get(s, s))

    # assume one sport per call
    sport_canon = out['Sport'].iloc[0]
    out = out[out['Sport'] == sport_canon].copy()
    if out.empty:
        return df

    # teams+window for fetch
    teams = pd.Index(out['Home_Team_Norm']).union(out['Away_Team_Norm']).unique().tolist()
    gmin, gmax = out['Game_Start'].min(), out['Game_Start'].max()
    pad = pd.Timedelta(days=pad_days)
    start_iso = pd.to_datetime(gmin - pad, utc=True).isoformat()
    end_iso   = pd.to_datetime(gmax + pad, utc=True).isoformat()

    # === Pull history (cached) and filter to these teams ===
    ratings = fetch_training_ratings_window_cached(
        sport=sport_canon,
        start_iso=start_iso,
        end_iso=end_iso,
        table_history=table_history,
        project=project,
    )
    if ratings.empty:
        base = np.float32(1500.0)
        out['Home_Power_Rating'] = base
        out['Away_Power_Rating'] = base
        out['Power_Rating_Diff'] = np.float32(0.0)
        return out

    # normalize team names again (same way on both sides)
    def norm_team(s: pd.Series) -> pd.Series:
        return (s.astype(str).str.lower().str.strip()
                .str.replace(r'\s+', ' ', regex=True)
                .str.replace('.', '', regex=False)
                .str.replace('&', 'and', regex=False)
                .str.replace('-', ' ', regex=False))

    ratings = ratings.copy()
    ratings['Team_Norm'] = norm_team(ratings['Team_Norm'])
    ratings = ratings[ratings['Team_Norm'].isin(teams)]
    if ratings.empty:
        base = np.float32(1500.0)
        out['Home_Power_Rating'] = base
        out['Away_Power_Rating'] = base
        out['Power_Rating_Diff'] = np.float32(0.0)
        return out

    # compact arrays per team: times + values (sorted)
    ratings['AsOfTS'] = pd.to_datetime(ratings['AsOfTS'], utc=True, errors='coerce')
    ratings = ratings.dropna(subset=['AsOfTS', 'Power_Rating', 'Team_Norm'])
    team_series = {}
    for team, g in ratings.groupby('Team_Norm', sort=False):
        g = g.sort_values('AsOfTS')
        team_series[team] = (
            g['AsOfTS'].to_numpy(dtype='datetime64[ns]'),
            g['Power_Rating'].to_numpy(dtype=np.float32),
        )

    # helper: vectorized backward asof via searchsorted
    def asof_fill(team_col: pd.Series) -> np.ndarray:
        # returns an array aligned to out.index with ratings for the provided team column
        vals = np.full(len(out), np.nan, dtype=np.float32)
        grp = out.groupby(team_col, sort=False)
        for team, idx in grp.groups.items():
            if team not in team_series:
                continue
            t_times, t_vals = team_series[team]
            if t_times.size == 0:
                continue
            game_times = out.loc[idx, 'Game_Start'].to_numpy(dtype='datetime64[ns]')
            pos = np.searchsorted(t_times, game_times, side='right') - 1
            valid = pos >= 0
            if valid.any():
                vals_idx = np.asarray(idx)[valid]
                vals[vals_idx] = t_vals[pos[valid]]
                if allow_forward_hours > 0:
                    # optional tiny forward grace for gaps only
                    need = ~valid
                    if need.any():
                        pos_fwd = np.searchsorted(t_times, game_times[need], side='left')
                        ok = (pos_fwd < t_times.size)
                        if ok.any():
                            # within grace?
                            fwd_dt = (t_times[pos_fwd[ok]] - game_times[need][ok])
                            ok2 = fwd_dt <= np.timedelta64(int(allow_forward_hours * 3600), 's')
                            if ok2.any():
                                dst_idx = np.asarray(idx)[need][ok][ok2]
                                vals[dst_idx] = t_vals[pos_fwd[ok][ok2]]
        return vals

    # compute ratings with no giant merges
    out = out.sort_values('Game_Start', kind='mergesort').reset_index(drop=False)  # keep original index in col 'index'
    home_pr = asof_fill(out['Home_Team_Norm'])
    away_pr = asof_fill(out['Away_Team_Norm'])

    out['Home_Power_Rating'] = home_pr
    out['Away_Power_Rating'] = away_pr

    # baseline fill for any remaining NaNs
    sport_mean = np.float32(pd.to_numeric(ratings['Power_Rating'], errors='coerce').mean() if not ratings.empty else 1500.0)
    out['Home_Power_Rating'] = out['Home_Power_Rating'].fillna(sport_mean).astype('float32')
    out['Away_Power_Rating'] = out['Away_Power_Rating'].fillna(sport_mean).astype('float32')
    out['Power_Rating_Diff'] = (out['Home_Power_Rating'] - out['Away_Power_Rating']).astype('float32')

    # restore original row order
    out = out.sort_values('index').drop(columns=['index']).reset_index(drop=True)
    return out   



# --- fast Normal CDF (SciPy-free), vectorized, low-temp ---
def _phi(x):
    """
    Approx to Φ(x) using Abramowitz–Stegun 7.1.26.
    Vectorized, float32, stable for |x| up to ~8.
    """
  
    x = np.asarray(x, dtype=np.float32)
    # constants as float32
    p = np.float32(0.2316419)
    b1, b2, b3, b4, b5 = (np.float32(0.319381530),
                          np.float32(-0.356563782),
                          np.float32(1.781477937),
                          np.float32(-1.821255978),
                          np.float32(1.330274429))
    # work on absolute x
    ax = np.abs(x, dtype=np.float32)
    t = 1.0 / (1.0 + p * ax)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    # standard normal pdf
    pdf = np.exp(-0.5 * ax * ax, dtype=np.float32) * np.float32(0.3989422804014327)  # 1/sqrt(2π)
    poly = b1*t + b2*t2 + b3*t3 + b4*t4 + b5*t5
    cdf_ax = 1.0 - (pdf * poly).astype(np.float32)
    # reflect for negative x
    return np.where(x >= 0, cdf_ax, 1.0 - cdf_ax).astype(np.float32)

SPORT_SPREAD_CFG = {
    "NFL":   dict(points_per_elo=np.float32(25.0), HFA=np.float32(1.6), sigma_pts=np.float32(13.2)),
    "NCAAF": dict(points_per_elo=np.float32(28.0), HFA=np.float32(2.4), sigma_pts=np.float32(16.0)),
    "NBA":   dict(points_per_elo=np.float32(28.0), HFA=np.float32(2.2), sigma_pts=np.float32(11.5)),
    "WNBA":  dict(points_per_elo=np.float32(28.0), HFA=np.float32(2.0), sigma_pts=np.float32(10.5)),
    "CFL":   dict(points_per_elo=np.float32(26.0), HFA=np.float32(1.8), sigma_pts=np.float32(14.0)),
}

def prep_consensus_market_spread_lowmem(
    df_spreads: pd.DataFrame,
    value_col: str = "Value",
    outcome_col: str = "Outcome_Norm",
) -> pd.DataFrame:
   
    cols = ['Sport','Home_Team_Norm','Away_Team_Norm', outcome_col, value_col]
    d = df_spreads[cols].copy()

    d['Sport'] = d['Sport'].astype(str).str.upper()
    for c in ['Home_Team_Norm','Away_Team_Norm', outcome_col]:
        d[c] = d[c].astype(str).str.lower().str.strip()
    d[value_col] = pd.to_numeric(d[value_col], errors='coerce').astype('float32')

    keys = ['Sport','Home_Team_Norm','Away_Team_Norm']

    m = (
        d.groupby(keys + [outcome_col], observed=True)[value_col]
         .median()
         .astype('float32')
         .reset_index(name='med')
    )
    # favorite = more negative median
    idx = m.groupby(keys, observed=True)['med'].idxmin()
    fav = m.loc[idx, keys + [outcome_col]].rename(columns={outcome_col: 'Market_Favorite_Team'})

    # k = median absolute spread per game
    k = (
        d.assign(_abs=d[value_col].abs().astype('float32'))
         .groupby(keys, observed=True)['_abs']
         .median()
         .astype('float32')
         .reset_index()
         .rename(columns={'_abs':'k'})
    )

    base = d.drop_duplicates(subset=keys)[keys].copy()
    g = fav.merge(base, on=keys, how='left', copy=False)
    g['Market_Underdog_Team'] = np.where(
        g['Market_Favorite_Team'].values == g['Home_Team_Norm'].values,
        g['Away_Team_Norm'].values,
        g['Home_Team_Norm'].values
    )
    g = g.merge(k, on=keys, how='left', copy=False)
    g['Favorite_Market_Spread']  = -g['k'].astype('float32')
    g['Underdog_Market_Spread']  =  g['k'].astype('float32')
    return g

def favorite_centric_from_powerdiff_lowmem(df_games: pd.DataFrame) -> pd.DataFrame:
   
    g = df_games.copy()
    g['Sport'] = g['Sport'].astype(str).str.upper()

    # pull sport params row-wise
    n = len(g)
    ppe   = np.full(n, np.float32(27.0), dtype=np.float32)  # default
    hfa   = np.zeros(n, dtype=np.float32)
    sigma = np.full(n, np.float32(12.0), dtype=np.float32)

    sp = g['Sport'].values
    for s, cfg in SPORT_SPREAD_CFG.items():
        mask = (sp == s)
        if mask.any():
            ppe[mask]   = cfg['points_per_elo']
            hfa[mask]   = cfg['HFA']
            sigma[mask] = cfg['sigma_pts']

    pr_diff = pd.to_numeric(g['Power_Rating_Diff'], errors='coerce').fillna(0).astype('float32').values
    # expected home - away margin in points
    mu = (pr_diff + hfa) / ppe
    mu = mu.astype(np.float32)
    mu_abs = np.abs(mu, dtype=np.float32)

    # k (market absolute spread) must be present from consensus merge
    k = pd.to_numeric(g.get('k', np.nan), errors='coerce').astype('float32').values

    # edges at game level (favorite vs dog)
    fav_edge = (mu_abs - k).astype('float32')
    dog_edge = (k - mu_abs).astype('float32')

    # cover probs for favorite side: P(margin > k)
    denom = sigma.copy()
    denom[denom == 0] = np.nan
    z_cov = (k - mu_abs) / denom
    fav_cover = (1.0 - _phi(z_cov)).astype('float32')
    dog_cover = (1.0 - fav_cover).astype('float32')

    g_out = pd.DataFrame({
        'Sport': g['Sport'].astype(str).values,
        'Home_Team_Norm': g['Home_Team_Norm'].astype(str).values,
        'Away_Team_Norm': g['Away_Team_Norm'].astype(str).values,
        'Model_Expected_Margin': mu,
        'Model_Expected_Margin_Abs': mu_abs,
        'Sigma_Pts': sigma.astype('float32'),
        'Model_Fav_Spread': (-mu_abs).astype('float32'),
        'Model_Dog_Spread': ( mu_abs).astype('float32'),
        'Model_Favorite_Team': np.where(mu >= 0, g['Home_Team_Norm'].values, g['Away_Team_Norm'].values),
        'Model_Underdog_Team': np.where(mu >= 0, g['Away_Team_Norm'].values, g['Home_Team_Norm'].values),

        # market bits (already merged prior step)
        'Market_Favorite_Team': g['Market_Favorite_Team'].values.astype(str),
        'Market_Underdog_Team': g['Market_Underdog_Team'].values.astype(str),
        'Favorite_Market_Spread': g['Favorite_Market_Spread'].astype('float32').values,
        'Underdog_Market_Spread': g['Underdog_Market_Spread'].astype('float32').values,

        # NEW: edges + cover probs at game level
        'Fav_Edge_Pts': fav_edge,
        'Dog_Edge_Pts': dog_edge,
        'Fav_Cover_Prob': fav_cover,
        'Dog_Cover_Prob': dog_cover,
    })
    return g_out

def enrich_and_grade_for_training(
    df_spread_rows: pd.DataFrame,
    bq,
    sport_aliases: dict,
    value_col: str = "Value",
    outcome_col: str = "Outcome_Norm",
    pad_days: int =30,
    allow_forward_hours: float = 0.0,
    table_history: str = "sharplogger.sharp_data.ratings_history",
    project: str = None,
) -> pd.DataFrame:
   

    if df_spread_rows.empty:
        return df_spread_rows

    # 1) leakage-safe ratings (your function)
    base = enrich_power_for_training_lowmem(
        df_spread_rows[['Sport','Home_Team_Norm','Away_Team_Norm','Game_Start']].drop_duplicates(),
        bq=bq,
        sport_aliases=sport_aliases,
        table_history=table_history,
        pad_days=pad_days,
        allow_forward_hours=allow_forward_hours,
        project=project,
    )

    # 2) consensus market spread (k) and favorite
    g_cons = prep_consensus_market_spread_lowmem(df_spread_rows, value_col=value_col, outcome_col=outcome_col)

    # 3) join → game-level favorite-centric grading (also computes edges & probs)
    game_key = ['Sport','Home_Team_Norm','Away_Team_Norm']
    g_full = base.merge(g_cons, on=game_key, how='left')
    g_fc   = favorite_centric_from_powerdiff_lowmem(g_full)

    # ensure power rating cols exist even if base missed (avoid KeyError later)
    for c in ['Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff']:
        if c not in g_fc.columns and c in g_full.columns:
            g_fc[c] = g_full[c].values
        elif c not in g_fc.columns:
            g_fc[c] = np.nan

    keep_cols = game_key + [
        'Market_Favorite_Team','Market_Underdog_Team',
        'Favorite_Market_Spread','Underdog_Market_Spread',
        'Model_Favorite_Team','Model_Underdog_Team',
        'Model_Fav_Spread','Model_Dog_Spread',
        'Fav_Edge_Pts','Dog_Edge_Pts',
        'Fav_Cover_Prob','Dog_Cover_Prob',
        'Model_Expected_Margin','Model_Expected_Margin_Abs','Sigma_Pts',
        'Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff'
    ]
    # backfill any missing keep cols with NaN to avoid KeyError
    for c in keep_cols:
        if c not in g_fc.columns:
            g_fc[c] = np.nan
    g_fc = g_fc[keep_cols].copy()

    out = df_spread_rows.merge(g_fc, on=game_key, how='left')

    # per-outcome mapping (vectorized, robust to dtype)
    is_fav = (out[outcome_col].astype(str).values == out['Market_Favorite_Team'].astype(str).values)
    out['Outcome_Model_Spread']  = np.where(is_fav, out['Model_Fav_Spread'].values, out['Model_Dog_Spread'].values).astype('float32')
    out['Outcome_Market_Spread'] = np.where(is_fav, out['Favorite_Market_Spread'].values, out['Underdog_Market_Spread'].values).astype('float32')
    out['Outcome_Spread_Edge']   = np.where(is_fav, out['Fav_Edge_Pts'].values, out['Dog_Edge_Pts'].values).astype('float32')
    out['Outcome_Cover_Prob']    = np.where(is_fav, out['Fav_Cover_Prob'].values, out['Dog_Cover_Prob'].values).astype('float32')

    return out

# --- 0) Build a unified favorite flag where totals use OVER as "favorite"
def add_favorite_context_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mkt = out['Market'].astype(str).str.lower()

    # Try Outcome_Norm first, fall back to Outcome/Label text if needed
    # We consider any token 'over' as OVER, 'under' as UNDER
    txt = (
        out.get('Outcome_Norm', out.get('Outcome', out.get('Label', '')))
           .astype(str).str.lower()
    )
    is_over = txt.str.contains(r'\bover\b', regex=True)

    # Default to existing Is_Favorite_Bet except for totals
    fav_flag = np.where(mkt.eq('totals'),
                        is_over.astype(int),                 # totals: OVER=1, UNDER=0
                        out.get('Is_Favorite_Bet', np.nan))  # spreads/h2h: keep your flag

    out['Is_Favorite_Context'] = fav_flag
    return out

def build_totals_training_from_scores(df_scores: pd.DataFrame,
                                      sport: str | None = None,
                                      window_games: int = 10,
                                      shrink: float = 0.30,
                                      key_col: str = "Merge_Key_Short"  # 👈 NEW
                                      ) -> pd.DataFrame:
    """
    Returns one row per historical game with:
      {key_col}, TOT_Proj_Total_Baseline, TOT_Off_H, TOT_Def_H, TOT_Off_A, TOT_Def_A,
      TOT_GT_H, TOT_GT_A, TOT_LgAvg_Total, TOT_Actual_Total
    Uses only prior games (shifted) => leakage-safe.
    """
    if key_col not in df_scores.columns:
        raise ValueError(f"Expected key_col '{key_col}' in df_scores")

    df = df_scores.copy()
    # normalize the join key and rename to internal 'Game_Key'
    df[key_col] = df[key_col].astype(str).str.strip().str.lower()
    df = df.rename(columns={key_col: "Game_Key"})  # 👈 internal key

    df["Game_Start"] = pd.to_datetime(df["Game_Start"])
    if "Season" not in df.columns:
        df["Season"] = df["Game_Start"].dt.year
    if sport:
        df = df[df["Sport"].str.upper() == sport.upper()].copy()

    home = df.assign(team=df["Home_Team"], opp=df["Away_Team"],
                     pts_for=df["Score_Home_Score"], pts_against=df["Score_Away_Score"])
    away = df.assign(team=df["Away_Team"], opp=df["Home_Team"],
                     pts_for=df["Score_Away_Score"], pts_against=df["Score_Home_Score"])
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["Season","team","Game_Start"]).reset_index(drop=True)
    long["game_total"] = long["pts_for"] + long["pts_against"]

    def _roll(x, col, w):
        return x[col].rolling(w, min_periods=1).mean().shift(1)  # exclude current game

    long["pf_roll"] = long.groupby(["Season","team"], group_keys=False).apply(_roll, "pts_for", window_games)
    long["pa_roll"] = long.groupby(["Season","team"], group_keys=False).apply(_roll, "pts_against", window_games)
    long["gt_roll"] = long.groupby(["Season","team"], group_keys=False).apply(_roll, "game_total", window_games)

    long["league_avg_total_prior"] = (long.groupby("Season")["game_total"]
                                         .expanding().mean().reset_index(level=0, drop=True).shift(1))
    long["league_avg_total_prior"] = long["league_avg_total_prior"].fillna(
        long.groupby("Season")["game_total"].transform("mean")
    )

    half = long["league_avg_total_prior"] / 2.0
    long["Off_Rating"] = (1-shrink)*(long["pf_roll"] - half)
    long["Def_Rating"] = (1-shrink)*(long["pa_roll"] - half)

    ratings = long[["Sport","Season","Game_Key","Game_Start","team","opp",
                    "Off_Rating","Def_Rating","gt_roll","league_avg_total_prior"]]

    home_r = ratings.rename(columns={"team":"Home_Team","opp":"Away_Team",
                                     "Off_Rating":"Off_H","Def_Rating":"Def_H",
                                     "gt_roll":"GT_H","league_avg_total_prior":"LgAvg_H"})
    away_r = ratings.rename(columns={"team":"Away_Team","opp":"Home_Team",
                                     "Off_Rating":"Off_A","Def_Rating":"Def_A",
                                     "gt_roll":"GT_A","league_avg_total_prior":"LgAvg_A"})

    g = df[["Sport","Season","Game_Key","Game_Start","Home_Team","Away_Team",
            "Score_Home_Score","Score_Away_Score"]].drop_duplicates()
    g = (g.merge(home_r, on=["Sport","Season","Game_Key","Game_Start","Home_Team","Away_Team"], how="left")
          .merge(away_r, on=["Sport","Season","Game_Key","Game_Start","Home_Team","Away_Team"], how="left"))

    g["LgAvg_Total"] = g["LgAvg_H"].combine_first(g["LgAvg_A"])
    base = (g["LgAvg_Total"]
            + 0.5*(g["Off_H"].fillna(0)+g["Def_A"].fillna(0))
            + 0.5*(g["Off_A"].fillna(0)+g["Def_H"].fillna(0)))
    pace_mult = ((g["GT_H"].fillna(g["LgAvg_Total"]) + g["GT_A"].fillna(g["LgAvg_Total"])) /
                 (2.0 * g["LgAvg_Total"].replace(0, np.nan))).clip(0.8, 1.2).fillna(1.0)
    g["Proj_Total_Baseline"] = base * pace_mult

    g["Actual_Total"] = g["Score_Home_Score"].astype(float) + g["Score_Away_Score"].astype(float)

    # rename internal key back to your key_col and prefix outputs
    g = g.rename(columns={"Game_Key": key_col})
    cols = ["Proj_Total_Baseline","Off_H","Def_H","Off_A","Def_A","GT_H","GT_A","LgAvg_Total","Actual_Total"]
    g = g[[key_col] + cols].rename(columns={c: f"TOT_{c}" for c in cols})
    return g


def totals_features_for_upcoming(df_scores: pd.DataFrame,
                                 df_schedule_like: pd.DataFrame,
                                 sport: str | None = None,
                                 window_games: int = 10,
                                 shrink: float = 0.30,
                                 key_col: str = "Merge_Key_Short"  # 👈 NEW
                                 ) -> pd.DataFrame:
    if key_col not in df_scores.columns or key_col not in df_schedule_like.columns:
        raise ValueError(f"Expected key_col '{key_col}' in both dataframes")

    # normalize join keys
    hist = build_totals_training_from_scores(df_scores, sport=sport,
                                             window_games=window_games, shrink=shrink,
                                             key_col=key_col)
    want = df_schedule_like[[key_col]].drop_duplicates()
    cols = [c for c in hist.columns if c.startswith("TOT_") and c != "TOT_Actual_Total"]
    return hist.merge(want, on=key_col, how="right")[[key_col] + cols]


from scipy.stats import randint, uniform, loguniform

SMALL_LEAGUES = {"WNBA", "CFL"}
BIG_LEAGUES   = {"MLB", "NBA", "NFL", "NCAAF", "NCAAB"}  # extend as needed

def get_xgb_search_space(
    sport: str,
    *,
    X_rows: int,
    n_jobs: int = 1,
    scale_pos_weight: float = 1.0,
    features: list[str] | None = None,
) -> tuple[dict, dict]:
    s = sport.upper()

    # ---- base kwargs (same skeleton for all; size-aware max_bin & CPU cores) ----
    base_kwargs = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        grow_policy="lossguide",
        max_leaves=32,                                 # bump to 48 if you see underfitting on BIG leagues
        max_bin=128 if X_rows < 150_000 else 64,
        sampling_method="uniform",
        single_precision_histogram=True,
        colsample_bynode=0.8,
        max_delta_step=1,
        n_jobs=n_jobs,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        importance_type="total_gain",
    )

    # ---- sport-size specific search spaces ----
    if s in SMALL_LEAGUES:
        # Small, noisy leagues → tighter, more regularized to keep generalization
        param_distributions = {
            "max_depth":        randint(2, 4),            # {2,3}
            "learning_rate":    loguniform(1e-2, 3e-2),   # 0.01–0.03
            "subsample":        uniform(0.70, 0.25),      # 0.70–0.95
            "colsample_bytree": uniform(0.60, 0.30),      # 0.60–0.90
            "min_child_weight": randint(20, 36),          # 20–35
            "gamma":            uniform(0.10, 0.40),      # 0.10–0.50 is okay too
            "reg_alpha":        loguniform(1.0, 2.0e1),   # 1–20
            "reg_lambda":       loguniform(5.0, 6.0e1),   # 5–60
        }
    else:
        # Bigger leagues → allow more expressiveness to expand std dev & AUC
        param_distributions = {
            "max_depth":        randint(2, 5),            # {2,3,4}
            "learning_rate":    loguniform(8e-3, 5e-2),   # 0.008–0.05
            "subsample":        uniform(0.75, 0.20),      # 0.75–0.95
            "colsample_bytree": uniform(0.65, 0.25),      # 0.65–0.90
            "min_child_weight": randint(10, 26),          # 10–25
            "gamma":            uniform(0.00, 0.40),      # 0–0.40
            "reg_alpha":        loguniform(1e-2, 1.5e1),  # 0.01–15
            "reg_lambda":       loguniform(2.0, 3.5e1),   # 2–35
        }

    # ---- optional: monotone constraints (no small-book creation) ----
    if features is not None:
        mono = {c: 0 for c in features}
        plus_1 = [
            "Abs_Odds_Move_From_Opening",
            "Pct_Line_Move_From_Opening", "Pct_Line_Move_Bin",
            "Abs_Line_Move_Z", "Pct_Line_Move_Z",
            "Line_Moved_Toward_Team",
            *[c for c in features if c.startswith("SharpMove_Magnitude_")],
            *[c for c in features if c.startswith("OddsMove_Magnitude_")],
            "Spread_vs_H2H_Aligned",
            "MarketLeader_ImpProbShift", "LimitProtect_SharpMag",
            "Delta_Sharp_vs_Rec", "Sharp_Leads",
            "Book_Reliability_Lift", "Book_Reliability_x_Sharp", "Book_Reliability_x_PROB_SHIFT",
            "Outcome_Cover_Prob",
        ]
        minus_1 = [
            "Value_Reversal_Flag", "Odds_Reversal_Flag",
            "Potential_Overmove_Flag", "Potential_Odds_Overmove_Flag",
            "Total_vs_Spread_Contradiction",
        ]
        for c in plus_1:
            if c in mono: mono[c] = 1
        for c in minus_1:
            if c in mono: mono[c] = -1
        base_kwargs["monotone_constraints"] = "(" + ",".join(str(mono.get(c, 0)) for c in features) + ")"

    return base_kwargs, param_distributions

def resolve_groups(df: pd.DataFrame) -> np.ndarray:
    # 1) Prefer the per-game key if present
    if "Merge_Key_Short" in df.columns and df["Merge_Key_Short"].notna().any():
        return df["Merge_Key_Short"].astype(str).to_numpy()

    # 2) If Merge_Key_Short missing, derive a per-game key
    #    Build from normalized teams + commence time (already in your table)
    cols_ok = all(c in df.columns for c in ["Home_Team_Norm","Away_Team_Norm","Commence_Hour"])
    if cols_ok:
        return (
            df["Home_Team_Norm"].astype(str)
            + "|" + df["Away_Team_Norm"].astype(str)
            + "|" + pd.to_datetime(df["Commence_Hour"], errors="coerce", utc=True).astype(str)
        ).to_numpy()

    # 3) Last-resort: strip market/outcome from Game_Key if possible
    #    Keep the substring up to the commence timestamp (first 19 chars of ISO datetime)
    if "Game_Key" in df.columns:
        # Example Game_Key pattern: "..._2025-07-20 01:00:00+00:00_h2h_los angeles dodgers"
        base = df["Game_Key"].astype(str).str.replace(r"_\d{4}-\d{2}-\d{2} .*", "", regex=True)
        # Reattach a normalized commence time if available for stability
        if "Commence_Hour" in df.columns:
            ch = pd.to_datetime(df["Commence_Hour"], errors="coerce", utc=True).astype(str)
            return (base + "|" + ch).to_numpy()
        return base.to_numpy()

    raise ValueError("No columns available to form per-game groups")

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
        FROM `sharplogger.sharp_data.scores_with_features`
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
    for c in ['Outcome']:
        if c in df_bt.columns:
            df_bt[c] = df_bt[c].astype(str).str.lower().str.strip()

    # normalize Merge_Key_Short on df_bt (sharp_scores_full)
    df_bt['Merge_Key_Short'] = df_bt['Merge_Key_Short'].astype(str).str.strip().str.lower()
    
    # pull completed scores from game_scores_final
    df_results = bq_client.query("""
      SELECT
        Merge_Key_Short,
        Home_Team, Away_Team, Game_Start,
        SAFE_CAST(Score_Home_Score AS FLOAT64) AS Score_Home_Score,
        SAFE_CAST(Score_Away_Score AS FLOAT64) AS Score_Away_Score,
        Sport
      FROM `sharplogger.sharp_data.game_scores_final`
      WHERE Score_Home_Score IS NOT NULL AND Score_Away_Score IS NOT NULL
    """).to_dataframe()
    
    df_results['Merge_Key_Short'] = df_results['Merge_Key_Short'].astype(str).str.strip().str.lower()
    # Pull completed games from game_scores_final
    df_totals = bq_client.query("""
      SELECT
        Merge_Key_Short,
        Home_Team, Away_Team, Game_Start,
        SAFE_CAST(Score_Home_Score AS FLOAT64) AS Score_Home_Score,
        SAFE_CAST(Score_Away_Score AS FLOAT64) AS Score_Away_Score,
        Sport
      FROM `sharplogger.sharp_data.game_scores_final`
      WHERE Score_Home_Score IS NOT NULL AND Score_Away_Score IS NOT NULL
    """).to_dataframe()
    
    # Normalize join key
    df_totals["Merge_Key_Short"] = df_totals["Merge_Key_Short"].astype(str).str.strip().str.lower()
    
    # Build totals features
    df_tot_train = build_totals_training_from_scores(
        df_totals, sport=sport, window_games=10, shrink=0.30, key_col="Merge_Key_Short"
    )
    
    # Merge into sharp_scores_full
    df_bt = df_bt.merge(
        df_tot_train.drop(columns=["TOT_Actual_Total"]),
        on="Merge_Key_Short",
        how="left"
    )
   
    # add mispricing only for totals markets
    if 'Value' not in df_bt.columns:
        df_bt['Value'] = np.nan
    is_totals = df_bt['Market'].str.lower().eq('totals')
    
    df_bt['TOT_Mispricing'] = np.where(
        is_totals & df_bt['TOT_Proj_Total_Baseline'].notna(),
        df_bt['Value'] - df_bt['TOT_Proj_Total_Baseline'],
        np.nan
    )

    
            
   # === Existing "as-of" history features (unchanged) ===
    history_cols = [
        #"After_Win_Flag","Revenge_Flag",
        "Current_Win_Streak_Prior","Current_Loss_Streak_Prior",
        "H2H_Win_Pct_Prior","Opp_WinPct_Prior",
        #"Last_Matchup_Result","Last_Matchup_Margin","Days_Since_Last_Matchup",
        "Wins_Last5_Prior","Margin_Last5_Prior",
        "Days_Since_Last_Game",
        "Close_Game_Rate_Prior","Blowout_Game_Rate_Prior",
        "Avg_Home_Margin_Prior","Avg_Away_Margin_Prior",
        
        
    ]
    
    # === New team ATS cover / margin stats (prior-only, last-5) ===
    team_cover_cols = [
        # overall cover signal
        "Cover_Rate_Last5",
        "Cover_Rate_After_Win_Last5",
        "Cover_Rate_After_Loss_Last5",
        # 8 situational cover rates (last-5)
        #"Cover_Rate_Home_After_Home_Win_Last5",
        #"Cover_Rate_Home_After_Home_Loss_Last5",
        #"Cover_Rate_Home_After_Away_Win_Last5",
        #"Cover_Rate_Home_After_Away_Loss_Last5",
        #"Cover_Rate_Away_After_Home_Win_Last5",
        #"Cover_Rate_Away_After_Home_Loss_Last5",
        #"Cover_Rate_Away_After_Away_Win_Last5",
        #"Cover_Rate_Away_After_Away_Loss_Last5",
        # margin distribution
        "ATS_Cover_Margin_Last5_Prior_Mean",
        "ATS_Cover_Margin_Last5_Prior_Std",
        # (optional per-game instantaneous margin if you want it)
       
    ]
    
    # === Opponent mirrors (prior-only, last-5) ===
    opp_cover_cols = [
        "Opp_Cover_Rate_Last5",
        "Opp_ATS_Cover_Margin_Last5_Prior_Mean",
        "Opp_ATS_Cover_Margin_Last5_Prior_Std",
        "Opp_Cover_Rate_After_Win_Last5",
        "Opp_Cover_Rate_After_Loss_Last5",
        #"Opp_Cover_Rate_Home_After_Home_Win_Last5",
        #"Opp_Cover_Rate_Home_After_Home_Loss_Last5",
        #"Opp_Cover_Rate_Home_After_Away_Win_Last5",
        #"Opp_Cover_Rate_Home_After_Away_Loss_Last5",
        #"Opp_Cover_Rate_Away_After_Home_Win_Last5",
        #"Opp_Cover_Rate_Away_After_Home_Loss_Last5",
        #"Opp_Cover_Rate_Away_After_Away_Win_Last5",
        #"Opp_Cover_Rate_Away_After_Away_Loss_Last5",
    ]
    
    # === (Optional but recommended) Team-vs-Opp diffs ===
    # These often help the model; they’ll be created only if both sides exist.
   
    # =============== Schema-safe selection ===============
    all_feature_cols = (
        history_cols
        + team_cover_cols
        + opp_cover_cols
    )
    
    # Keep only columns that actually exist
    history_present = [c for c in history_cols if c in df_bt.columns]
    
    team_cover_present = [c for c in team_cover_cols if c in df_bt.columns]
    opp_cover_present = [c for c in opp_cover_cols if c in df_bt.columns]
    all_present = history_present + team_cover_present + opp_cover_present
    
    # Optional: build diffs where both inputs exist
    for left, right, out in diff_specs:
        if left in df_bt.columns and right in df_bt.columns:
            df_bt[out] = df_bt[left] - df_bt[right]
            all_present.append(out)
    
    # (Optional) enforce numeric dtypes for model features
    for col in all_present:
        # Leave booleans as is; cast others to numeric where possible
        if df_bt[col].dtype == "bool":
            continue
        df_bt[col] = pd.to_numeric(df_bt[col], errors="coerce")
    
    # Handle NaNs if your model can’t
    df_bt[all_present] = df_bt[all_present].fillna(0.0)
    

   
        
    # ✅ Make sure helper won't choke if these are missing
    if 'Is_Sharp_Book' not in df_bt.columns:
        df_bt['Is_Sharp_Book'] = df_bt['Bookmaker'].isin(SHARP_BOOKS).astype(int)
    if 'Sharp_Line_Magnitude' not in df_bt.columns:
        df_bt['Sharp_Line_Magnitude'] = pd.to_numeric(df_bt.get('Line_Delta', 0), errors='coerce').abs().fillna(0)
   
    df_bt['Market'] = df_bt['Market'].astype(str).str.lower().str.strip()
    df_bt['Sport']  = df_bt['Sport'].astype(str).str.upper()
    
   
   
    
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
     
    with st.spinner("Training…"):
        try:
            df_bt = enrich_power_for_training_lowmem(
                df=df_bt,
                bq=bq_client,
                sport_aliases=SPORT_ALIASES,
                pad_days=10,
                allow_forward_hours=0.0,
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    # df_spreads is your historical spread rows (per outcome/book/snapshot)
    # must include: ['Sport','Game_Start','Home_Team_Norm','Away_Team_Norm','Outcome_Norm','Value']
    # Build the spread-rows input from your working frame
    # Build the spread-rows input from your working frame
    # === Build training input for spread rows from df_bt (deduped base) ===
    df_spreads = (
        df_bt[df_bt['Market'] == 'spreads']  # training focuses on spreads here
        [[
            'Sport','Game_Start','Home_Team_Norm','Away_Team_Norm',
            'Outcome'  # will normalize to Outcome_Norm next
           ,'Value'
        ]]
        .copy()
    )
    df_spreads['Outcome_Norm'] = df_spreads['Outcome'].astype(str).str.lower().str.strip()
    df_spreads.drop(columns=['Outcome'], inplace=True)
    df_spreads = df_spreads.dropna(
        subset=['Sport','Home_Team_Norm','Away_Team_Norm','Outcome_Norm','Value']
    )
    df_spreads['Sport'] = df_spreads['Sport'].astype(str).str.upper()
    
    # === TRAINING-SAFE enrichment & grading (historical/as-of) ===
    df_train = enrich_and_grade_for_training(
        df_spread_rows=df_spreads,
        bq=bq_client,
        sport_aliases=SPORT_ALIASES,
        value_col="Value",
        outcome_col="Outcome_Norm",
        pad_days=10,
        allow_forward_hours=0.0,  # strict backward-only
        table_history="sharplogger.sharp_data.ratings_history",
        project="sharplogger",
    )
    

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
    
    def _prep_team_context(df):
        out = df.copy()
        out['Market']         = out['Market'].astype(str).str.lower().str.strip()
        out['Outcome_Norm']   = out['Outcome'].astype(str).str.lower().str.strip()
        out['Home_Team_Norm'] = out['Home_Team_Norm'].astype(str).str.lower().str.strip()
        out['Away_Team_Norm'] = out['Away_Team_Norm'].astype(str).str.lower().str.strip()
    
        is_totals = out['Market'].eq('totals')
        # Team identity (consistent with your training logic)
        out['Team'] = out['Outcome_Norm'].where(~is_totals, out['Home_Team_Norm']).astype(str).str.strip()
        out['Is_Home'] = np.where(is_totals, 1, (out['Team'] == out['Home_Team_Norm']).astype(int)).astype(int)
    
        # Favorite context: spreads/h2h -> Value<0 ; totals -> OVER
        out['Is_Favorite_Context'] = np.where(
            is_totals, (out['Outcome_Norm'] == 'over').astype(int),
            (pd.to_numeric(out['Value'], errors='coerce') < 0).astype(int)
        ).astype(int)
        return out
    
    df_bt_prepped = _prep_team_context(df_bt)
    _prob_candidates = [
        'Model_Sharp_Win_Prob',
        'Model_Outcome_Prob',
        'Model_Prob',
        'Pred_Prob',
    ]
    prob_col = next((c for c in _prob_candidates if c in df_bt_prepped.columns), None)
    
    if prob_col is None:
        # keep pipeline alive; will produce NaNs in LOO averages (no leakage)
        df_bt_prepped['Model_Sharp_Win_Prob'] = np.nan
    else:
        df_bt_prepped['Model_Sharp_Win_Prob'] = pd.to_numeric(df_bt_prepped[prob_col], errors='coerce')


    def compute_loo_stats_by_game(df, home_filter=None, favorite_filter=None, col_suffix=""):
        df = df.copy()
        if home_filter is not None:
            df = df[df['Is_Home'] == home_filter]
        if favorite_filter is not None:
            df = df[df['Is_Favorite_Context'] == favorite_filter]
        if df.empty:
            return pd.DataFrame(columns=['Game_Key','Team',
                                         f'Team_Past_Avg_Model_Prob{col_suffix}',
                                         f'Team_Past_Hit_Rate{col_suffix}'])
        # earliest per (Game_Key, Team)
        sort_cols = [c for c in ['Team','Game_Start','Snapshot_Timestamp'] if c in df.columns] or ['Team','Game_Key']
        df_dedup = (df.sort_values(sort_cols)
                      .drop_duplicates(subset=['Game_Key','Team'], keep='first')
                      .sort_values(['Team'] + ([c for c in ['Game_Start','Snapshot_Timestamp'] if c in df.columns] or ['Game_Key'])))
    
        # Leave-one-out
        df_dedup['cum_model_prob'] = df_dedup.groupby('Team')['Model_Sharp_Win_Prob'].cumsum().shift(1)
        df_dedup['cum_hit']        = df_dedup.groupby('Team')['SHARP_HIT_BOOL'].cumsum().shift(1)
        df_dedup['cum_count']      = df_dedup.groupby('Team').cumcount()
    
        avg_col = f'Team_Past_Avg_Model_Prob{col_suffix}'
        hit_col = f'Team_Past_Hit_Rate{col_suffix}'
        df_dedup[avg_col] = df_dedup['cum_model_prob'] / df_dedup['cum_count'].replace(0, np.nan)
        df_dedup[hit_col] = df_dedup['cum_hit'] / df_dedup['cum_count'].replace(0, np.nan)
        return df_dedup[['Game_Key','Team', avg_col, hit_col]]
    
    # LOO sets — computed from the FULL base
    overall_stats = compute_loo_stats_by_game(df_bt_prepped)
    home_stats    = compute_loo_stats_by_game(df_bt_prepped, home_filter=1, col_suffix="_Home")
    away_stats    = compute_loo_stats_by_game(df_bt_prepped, home_filter=0, col_suffix="_Away")
    fav_overall   = compute_loo_stats_by_game(df_bt_prepped, favorite_filter=1, col_suffix="_Fav")
    fav_home      = compute_loo_stats_by_game(df_bt_prepped, home_filter=1, favorite_filter=1, col_suffix="_Home_Fav")
    fav_away      = compute_loo_stats_by_game(df_bt_prepped, home_filter=0, favorite_filter=1, col_suffix="_Away_Fav")
    
    # Merge these LOO features back onto df_bt (so you can later merge into df_market)
    df_bt_loostats = (df_bt_prepped[['Game_Key','Team']]
                      .drop_duplicates()
                      .merge(overall_stats, on=['Game_Key','Team'], how='left')
                      .merge(home_stats,    on=['Game_Key','Team'], how='left')
                      .merge(away_stats,    on=['Game_Key','Team'], how='left')
                      .merge(fav_overall,   on=['Game_Key','Team'], how='left')
                      .merge(fav_home,      on=['Game_Key','Team'], how='left')
                      .merge(fav_away,      on=['Game_Key','Team'], how='left'))
    
    # Rolling/streak window by sport
    SPORT_COVER_WINDOW = {'NBA':5, 'NFL':4, 'WNBA':3, 'MLB':7, 'CFL':4}
    sport0 = (df_bt_prepped['Sport'].dropna().astype(str).iloc[0] if df_bt_prepped['Sport'].notna().any() else 'NFL')
    window_length = SPORT_COVER_WINDOW.get(sport0, 4)
    
    # Rolling/streaks — FULL base
    dfb = df_bt_prepped.sort_values(['Team','Snapshot_Timestamp'])
    dfb['Cover_Home_Only'] = dfb['SHARP_HIT_BOOL'].where(dfb['Is_Home'] == 1)
    dfb['Cover_Away_Only'] = dfb['SHARP_HIT_BOOL'].where(dfb['Is_Home'] == 0)
    dfb['Cover_Fav_Only']  = dfb['SHARP_HIT_BOOL'].where(dfb['Is_Favorite_Context'] == 1)
    dfb['Cover_Dog_Only']  = dfb['SHARP_HIT_BOOL'].where(dfb['Is_Favorite_Context'] == 0)
    
    def _roll_sum_shift(s, g, w):
        s1 = s.groupby(g).shift(1)
        return s1.groupby(g).rolling(window=w, min_periods=1).sum().reset_index(level=0, drop=True)
    
    dfb['Team_Recent_Cover_Streak']          = _roll_sum_shift(dfb['SHARP_HIT_BOOL'],     dfb['Team'], window_length)
    dfb['Team_Recent_Cover_Streak_Home']     = _roll_sum_shift(dfb['Cover_Home_Only'],    dfb['Team'], window_length)
    dfb['Team_Recent_Cover_Streak_Away']     = _roll_sum_shift(dfb['Cover_Away_Only'],    dfb['Team'], window_length)
    dfb['Team_Recent_Cover_Streak_Fav']      = _roll_sum_shift(dfb['Cover_Fav_Only'],     dfb['Team'], window_length)
    dfb['Team_Recent_Cover_Streak_Home_Fav'] = _roll_sum_shift(dfb['Cover_Home_Only'].where(dfb['Is_Favorite_Context']==1), dfb['Team'], window_length)
    dfb['Team_Recent_Cover_Streak_Away_Fav'] = _roll_sum_shift(dfb['Cover_Away_Only'].where(dfb['Is_Favorite_Context']==1), dfb['Team'], window_length)
    
    dfb['On_Cover_Streak']          = (dfb['Team_Recent_Cover_Streak']          >= 2).astype(int)
    dfb['On_Cover_Streak_Home']     = (dfb['Team_Recent_Cover_Streak_Home']     >= 2).astype(int)
    dfb['On_Cover_Streak_Away']     = (dfb['Team_Recent_Cover_Streak_Away']     >= 2).astype(int)
    dfb['On_Cover_Streak_Fav']      = (dfb['Team_Recent_Cover_Streak_Fav']      >= 2).astype(int)
    dfb['On_Cover_Streak_Home_Fav'] = (dfb['Team_Recent_Cover_Streak_Home_Fav'] >= 2).astype(int)
    dfb['On_Cover_Streak_Away_Fav'] = (dfb['Team_Recent_Cover_Streak_Away_Fav'] >= 2).astype(int)
    
    streak_cols = [
        'Team_Recent_Cover_Streak','Team_Recent_Cover_Streak_Home','Team_Recent_Cover_Streak_Away',
        'Team_Recent_Cover_Streak_Fav','Team_Recent_Cover_Streak_Home_Fav','Team_Recent_Cover_Streak_Away_Fav',
        'On_Cover_Streak','On_Cover_Streak_Home','On_Cover_Streak_Away',
        'On_Cover_Streak_Fav','On_Cover_Streak_Home_Fav','On_Cover_Streak_Away_Fav'
    ]
    
    df_bt_streaks = (dfb[['Game_Key','Team'] + streak_cols]
                     .drop_duplicates(subset=['Game_Key','Team'], keep='last'))
    # Combine LOO stats + streaks into one context frame
    df_bt_context = (
        pd.concat([df_bt_loostats, df_bt_streaks], axis=0, ignore_index=True)
          .groupby(['Game_Key','Team'], as_index=False).last()
    )

    context_cols = [c for c in df_bt_context.columns if c not in ['Game_Key','Team']]

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
    
        # 🔧 Totals often duplicate (over/under, multiple snapshots) — dedup here
        if market == "totals":
            df_market = (
                df_market
                .sort_values(['Snapshot_Timestamp'])
                .drop_duplicates(subset=['Game_Key','Bookmaker','Market','Outcome'], keep='last')
            )
    
        # Make alignment issues impossible downstream
        df_market.reset_index(drop=True, inplace=True)
    
        df_market = compute_small_book_liquidity_features(df_market)
        df_market = add_favorite_context_flag(df_market)  # adds Is_Favorite_Context
        df_market = df_market.merge(df_cross_market, on='Game_Key', how='left')
    
        if df_market.empty:
            status.warning(f"⚠️ No data for {market.upper()} — skipping.")
            pb.progress(int(round(idx / n_markets * 100)))
            continue
    
        # Normalize...
        df_market['Outcome_Norm']   = df_market['Outcome'].astype(str).str.lower().str.strip()
        df_market['Home_Team_Norm'] = df_market['Home_Team_Norm'].astype(str).str.lower().str.strip()
        df_market['Away_Team_Norm'] = df_market['Away_Team_Norm'].astype(str).str.lower().str.strip()
        df_market['Market']         = df_market['Market'].astype(str).str.lower().str.strip()
        df_market['Game_Key']       = df_market['Game_Key'].astype(str).str.lower().str.strip()

        
        # --- Team / Is_Home (totals anchor to home)
        is_totals = df_market['Market'].eq('totals')
        df_market['Team'] = (
            df_market['Outcome_Norm'].where(~is_totals, df_market['Home_Team_Norm'])
        ).astype(str).str.strip().str.lower()
        df_market['Is_Home'] = np.where(is_totals, 1, (df_market['Team'] == df_market['Home_Team_Norm']).astype(int)).astype(int)
        
        # --- make sure context keys are normalized the same way
        df_bt_context['Game_Key'] = df_bt_context['Game_Key'].astype(str).str.lower().str.strip()
        df_bt_context['Team']     = df_bt_context['Team'].astype(str).str.lower().str.strip()
        
        # --- guard selected columns
        keep_ctx = ['Game_Key','Team'] + [c for c in context_cols if c in df_bt_context.columns]
        
        # --- merge
        before = len(df_market)
        df_market = df_market.merge(df_bt_context[keep_ctx], on=['Game_Key','Team'], how='left')
        hit = df_market['Team'].notna() & df_market[keep_ctx[2]].notna() if len(keep_ctx) > 2 else df_market['Team'].notna()
        miss_rate = 1 - (hit.sum() / max(len(df_market), 1))

        # === Canonical side filtering ONLY (training subset)
        if market == "totals":
            df_market = df_market[df_market['Outcome_Norm'] == 'over']
        elif market in ("spreads", "h2h"):
            df_market = df_market[pd.to_numeric(df_market['Value'], errors='coerce') < 0]
    
        # labels
        df_market = df_market[df_market['SHARP_HIT_BOOL'].isin([0, 1])]
        if df_market.empty or df_market['SHARP_HIT_BOOL'].nunique() < 2:
            status.warning(f"⚠️ Not enough label variety for {market.upper()} — skipping.")
            pb.progress(int(round(idx / n_markets * 100)))
            continue
        if market == "spreads":
            # ---- SPREADS training block (leakage-safe) ----
            game_keys = ['Sport', 'Home_Team_Norm', 'Away_Team_Norm']
        
            # Normalize merge keys on BOTH sides (Sport=UPPER, teams=lower)
            df_market['Sport'] = df_market['Sport'].astype(str).str.upper().str.strip()
            df_market['Home_Team_Norm'] = df_market['Home_Team_Norm'].astype(str).str.lower().str.strip()
            df_market['Away_Team_Norm'] = df_market['Away_Team_Norm'].astype(str).str.lower().str.strip()
            if 'Sport' in df_train.columns:
                df_train['Sport'] = df_train['Sport'].astype(str).str.upper().str.strip()
            if 'Home_Team_Norm' in df_train.columns:
                df_train['Home_Team_Norm'] = df_train['Home_Team_Norm'].astype(str).str.lower().str.strip()
            if 'Away_Team_Norm' in df_train.columns:
                df_train['Away_Team_Norm'] = df_train['Away_Team_Norm'].astype(str).str.lower().str.strip()
        
            # 1) Skinny slice for this snapshot (only needed cols)
            slice_cols = ['Sport','Game_Start','Home_Team_Norm','Away_Team_Norm','Outcome_Norm','Value']
            df_slice = (
                df_market[slice_cols]
                .dropna(subset=['Sport','Home_Team_Norm','Away_Team_Norm','Outcome_Norm','Value'])
                .copy()
            )
            df_slice['Value'] = pd.to_numeric(df_slice['Value'], errors='coerce').astype('float32')
        
            # 2) Build maps from df_train (training-safe enrichment)
            pr_map = df_train[game_keys + ['Power_Rating_Diff']].drop_duplicates(subset=game_keys)
        
            cons_cols = game_keys + [
                'Market_Favorite_Team','Market_Underdog_Team',
                'Favorite_Market_Spread','Underdog_Market_Spread'
            ]
            missing_cons_cols = [c for c in cons_cols if c not in df_train.columns]
            if missing_cons_cols:
                # ✅ FIX: proper fallback helper
                g_cons_fallback = prep_consensus_market_spread_lowmem(df_slice, value_col='Value', outcome_col='Outcome_Norm')
                cons_map = g_cons_fallback[cons_cols].drop_duplicates(subset=game_keys)
            else:
                cons_map = df_train[cons_cols].drop_duplicates(subset=game_keys)
        
            # Reconstruct k = |spread|
            k1 = pd.to_numeric(cons_map['Favorite_Market_Spread'], errors='coerce').abs()
            k2 = pd.to_numeric(cons_map['Underdog_Market_Spread'], errors='coerce').abs()
            cons_map['k'] = k1.fillna(k2).astype('float32')
        
            # 3) Join maps to make game-level frame
            g_full = (
                df_slice[game_keys].drop_duplicates(subset=game_keys)
                .merge(pr_map,  on=game_keys, how='left')
                .merge(cons_map, on=game_keys, how='left')
            )
        
            # 4) Model margin & spreads at game level
            g_fc = favorite_centric_from_powerdiff_lowmem(g_full)
        
            # Ensure market fields & k exist in g_fc (robust)
            need_market = ['Market_Favorite_Team','Market_Underdog_Team',
                           'Favorite_Market_Spread','Underdog_Market_Spread']
            missing_mkt = [c for c in need_market if c not in g_fc.columns]
            if missing_mkt:
                g_fc = g_fc.merge(
                    cons_map[game_keys + need_market].drop_duplicates(subset=game_keys),
                    on=game_keys, how='left', copy=False
                )
        
            if 'k' not in g_fc.columns:
                fav_abs = pd.to_numeric(g_fc.get('Favorite_Market_Spread'), errors='coerce').abs()
                dog_abs = pd.to_numeric(g_fc.get('Underdog_Market_Spread'), errors='coerce').abs()
                g_fc['k'] = fav_abs.fillna(dog_abs).astype('float32')
        
            # ✅ Guarantee the model spreads exist even if helper didn’t attach them
            if 'Model_Fav_Spread' not in g_fc.columns or 'Model_Dog_Spread' not in g_fc.columns:
                ema = pd.to_numeric(g_fc.get('Model_Expected_Margin_Abs'), errors='coerce').astype('float32')
                g_fc['Model_Fav_Spread'] = (-ema).astype('float32')
                g_fc['Model_Dog_Spread'] = ( ema).astype('float32')
        
            # Only the columns we need from g_fc
            proj_cols = [
                'Model_Fav_Spread','Model_Dog_Spread',
                'Market_Favorite_Team','Market_Underdog_Team',
                'Favorite_Market_Spread','Underdog_Market_Spread',
                'Model_Expected_Margin_Abs','Sigma_Pts','k'
            ]
            g_map = g_fc[game_keys + proj_cols].drop_duplicates(subset=game_keys)
        
            # Drop overlaps to avoid dup labels, then merge
            overlap = [c for c in proj_cols if c in df_market.columns]
            if overlap:
                df_market.drop(columns=overlap, inplace=True, errors='ignore')
        
            df_market = df_market.merge(g_map, on=game_keys, how='left', copy=False)
            df_market = df_market.loc[:, ~df_market.columns.duplicated(keep='first')]
        
            # Streamlit KPIs (guarded)
            if 'Model_Fav_Spread' in df_market.columns:
                have_spreads = float(df_market['Model_Fav_Spread'].notna().mean())
                have_k = float(df_market['k'].notna().mean()) if 'k' in df_market.columns else 0.0
                status.write(
                    f"🧪 SPREADS merge health — rows: {len(df_market):,} | "
                    f"have Model_Fav_Spread: **{have_spreads:.1%}** | "
                    f"have k: **{have_k:.1%}**"
                )
            else:
                status.error("❌ SPREADS: `Model_Fav_Spread` missing after merge — showing debug keys below.")
                with st.expander("Debug: spreads merge inputs"):
                    st.write("Expected from g_map: Model_Fav_Spread, Model_Dog_Spread, Favorite/Underdog_Market_Spread, k")
                    st.dataframe(
                        df_market[['Sport','Home_Team_Norm','Away_Team_Norm']]
                        .drop_duplicates().head(30),
                        use_container_width=True
                    )
        
            # Compute per-outcome spreads from model/market ONLY (no engineered fields yet)
            for c in ['Outcome_Norm','Market_Favorite_Team']:
                df_market[c] = df_market[c].astype(str).str.lower().str.strip()
        
            is_fav = (df_market['Outcome_Norm'].values == df_market['Market_Favorite_Team'].values)
        
            df_market['Outcome_Model_Spread']  = np.where(
                is_fav, df_market['Model_Fav_Spread'].values, df_market['Model_Dog_Spread'].values
            ).astype('float32')
        
            df_market['Outcome_Market_Spread'] = np.where(
                is_fav, df_market['Favorite_Market_Spread'].values, df_market['Underdog_Market_Spread'].values
            ).astype('float32')
        
            # --- Per-outcome engineered features (requires: k, Sigma_Pts, fav/dog spreads) ---
            df_market['Outcome_Spread_Edge'] = (
                pd.to_numeric(df_market['Outcome_Model_Spread'],  errors='coerce') -
                pd.to_numeric(df_market['Outcome_Market_Spread'], errors='coerce')
            ).astype('float32')
            
            # z = edge / k  (guard against zero/NaN k)
            k = pd.to_numeric(df_market.get('k'), errors='coerce').astype('float32')
            k = k.where(k > 0, np.nan)              # avoid div-by-0
            df_market['z'] = (df_market['Outcome_Spread_Edge'] / k).astype('float32')
            
            # cover prob: Φ(z)
            df_market['Outcome_Cover_Prob'] = _phi(df_market['z']).astype('float32')
            
            # magnitudes + interactions
            df_market['edge_pts'] = df_market['Outcome_Spread_Edge'].astype('float32')
            df_market['mu_abs']   = df_market['Outcome_Spread_Edge'].abs().astype('float32')
            df_market['edge_x_k'] = (df_market['Outcome_Spread_Edge'] * k).astype('float32')
            df_market['mu_x_k']   = (df_market['mu_abs'] * k).astype('float32')
            
            # agreement flag: does model mark THIS team as fav the same as market?
            model_this_is_fav  = pd.to_numeric(df_market['Outcome_Model_Spread'],  errors='coerce') < 0
            market_this_is_fav = df_market['Outcome_Norm'].astype(str).str.lower().eq(
                df_market['Market_Favorite_Team'].astype(str).str.lower()
            )
            df_market['model_fav_vs_market_fav_agree'] = (model_this_is_fav == market_this_is_fav).astype('int8')
            
            # (Optional) quick health checks to catch why features “disappear”
            # --- Streamlit KPIs / Debug ---
            have_k   = float(df_market['k'].notna().mean()) if 'k' in df_market.columns else 0.0
            have_sig = float(pd.to_numeric(df_market.get('Sigma_Pts'), errors='coerce').notna().mean()) if 'Sigma_Pts' in df_market.columns else 0.0
            have_oms = float(df_market['Outcome_Model_Spread'].notna().mean()) if 'Outcome_Model_Spread' in df_market.columns else 0.0
            have_oms_mkt = float(df_market['Outcome_Market_Spread'].notna().mean()) if 'Outcome_Market_Spread' in df_market.columns else 0.0
            
            st.write(
                f"🧪 SPREADS enrich — rows: {len(df_market):,} | "
                f"Outcome_Model_Spread: **{have_oms:.1%}** | "
                f"Outcome_Market_Spread: **{have_oms_mkt:.1%}** | "
                f"k: **{have_k:.1%}** | Sigma: **{have_sig:.1%}**"
            )
            
            if have_oms < 1.0 or have_oms_mkt < 1.0:
                with st.expander("⚠️ Debug: missing Outcome spreads"):
                    st.dataframe(
                        df_market[['Sport','Home_Team_Norm','Away_Team_Norm','Outcome_Norm']]
                        .drop_duplicates().head(30),
                        use_container_width=True
        )



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
        # before any LOO/streak calcs
        # --- Normalize inputs first
        # -- keep only one Outcome_Norm normalization (this one)
        # Normalize key text fields
        # Normalize key text fields
        df_market['Market']         = df_market['Market'].astype(str).str.lower().str.strip()
        df_market['Outcome_Norm']   = df_market['Outcome'].astype(str).str.lower().str.strip()
        df_market['Home_Team_Norm'] = df_market['Home_Team_Norm'].astype(str).str.lower().str.strip()
        df_market['Away_Team_Norm'] = df_market['Away_Team_Norm'].astype(str).str.lower().str.strip()
        
        # Totals flag
        is_totals = df_market['Market'].eq('totals')
        
        # Define Team exactly once
        df_market['Team'] = (
            df_market['Outcome_Norm']
              .where(~is_totals, df_market['Home_Team_Norm'])  # totals -> anchor to home
              .astype(str).str.lower().str.strip()
        )
        
        # Define Is_Home exactly once
        df_market['Is_Home'] = np.where(
            is_totals,
            1,
            (df_market['Team'] == df_market['Home_Team_Norm']).astype(int)
        ).astype(int)
        
        # Role labeler AFTER Team/Is_Home
       
                
        
        # (optional) Keep your role labeler, but call it AFTER the fields above are set
        df_market['Team_Bet_Role'] = df_market.apply(label_team_role, axis=1)
        
        # === Step 0: Sort once up front
        df_market = df_market.sort_values(['Team', 'Game_Key'])

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
        
        # === Interaction Features (filtered for value)
        #if 'Odds_Shift' in df_market.columns:
            #df_market['SharpMove_OddsShift'] = df_market['Sharp_Move_Signal'] * df_market['Odds_Shift']
        
        if 'Implied_Prob_Shift' in df_market.columns:
            df_market['MarketLeader_ImpProbShift'] = df_market['Market_Leader'] * df_market['Implied_Prob_Shift']
        
        df_market['SharpLimit_SharpBook'] = df_market['Is_Sharp_Book'] * df_market['Sharp_Limit_Total']
        df_market['LimitProtect_SharpMag'] = df_market['LimitUp_NoMove_Flag'] * df_market['Sharp_Line_Magnitude']
        # Example for engineered features

        df_market['High_Limit_Flag'] = (df_market['Sharp_Limit_Total'] >= 7000).astype(int)
       
        df_market = df_market.reset_index(drop=True)

        # Odds magnitude * sharp flag
        df_market['SharpMove_Odds_Mag'] = (
            df_market['Odds_Shift'].abs().fillna(0).to_numpy()
            * df_market['Sharp_Move_Signal'].fillna(0).astype(int).to_numpy()
        )
       

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
        
       

                   
        # ensure normalized for comparison
        df_market['Outcome_Norm']   = norm_team(df_market['Outcome_Norm'])
        df_market['Home_Team_Norm'] = norm_team(df_market['Home_Team_Norm'])
        df_market['Away_Team_Norm'] = norm_team(df_market['Away_Team_Norm'])
        
        is_home_bet = (df_market['Outcome_Norm'] == df_market['Home_Team_Norm'])
        
        df_market['PR_Team_Rating'] = np.where(is_home_bet, df_market['Home_Power_Rating'], df_market['Away_Power_Rating'])
        df_market['PR_Opp_Rating']  = np.where(is_home_bet, df_market['Away_Power_Rating'], df_market['Home_Power_Rating'])
        #df_market['PR_Team_Off']    = np.where(is_home_bet, df_market['Home_PR_Off'], df_market['Away_PR_Off'])
        #df_market['PR_Team_Def']    = np.where(is_home_bet, df_market['Home_PR_Def'], df_market['Away_PR_Def'])
        #df_market['PR_Opp_Off']     = np.where(is_home_bet, df_market['Away_PR_Off'], df_market['Home_PR_Off'])
        #df_market['PR_Opp_Def']     = np.where(is_home_bet, df_market['Away_PR_Def'], df_market['Home_PR_Def'])
        
        df_market['PR_Rating_Diff']     = df_market['PR_Team_Rating'] - df_market['PR_Opp_Rating']
        df_market['PR_Abs_Rating_Diff'] = df_market['PR_Rating_Diff'].abs()
        #df_market['PR_Total_Est']       = (df_market['PR_Team_Off'] + df_market['PR_Opp_Off'] - df_market['PR_Team_Def'] - df_market['PR_Opp_Def'])

        # helper to extend without dupes (order-preserving)
        def extend_unique(base, items):
            for c in items:
                if c not in base:
                    base.append(c)
        
        # --- start with your manual core list ---
        features = [
            # 🔹 Core sharp signals
            #'Sharp_Move_Signal',
            'Sharp_Limit_Jump',#'Sharp_Time_Score','Book_lift_x_Sharp',
            'Book_lift_x_Magnitude','Book_lift_x_PROB_SHIFT','Sharp_Limit_Total',
            'Is_Reinforced_MultiMarket',#'Market_Leader','LimitUp_NoMove_Flag',
        
            # 🔹 Market response
            #'Sharp_Line_Magnitude','Is_Home_Team_Bet',
            #'Team_Implied_Prob_Gap_Home','Team_Implied_Prob_Gap_Away',
        
            # 🔹 Engineered odds shift decomposition
            #'SharpMove_Odds_Up','SharpMove_Odds_Down','SharpMove_Odds_Mag',
        
            # 🔹 Engineered interactions
            'MarketLeader_ImpProbShift','LimitProtect_SharpMag','Delta_Sharp_vs_Rec','Sharp_Leads',
        
            # 🔁 Reversal logic
            'Value_Reversal_Flag','Odds_Reversal_Flag',
        
            # 🔥 Timing flags
            #'Late_Game_Steam_Flag',
        
            #'Abs_Line_Move_From_Opening',
            'Abs_Odds_Move_From_Opening',
            'Market_Mispricing','Spread_vs_H2H_Aligned','Total_vs_Spread_Contradiction',
            'Spread_vs_H2H_ProbGap','Total_vs_H2H_ProbGap','Total_vs_Spread_ProbGap',
            'CrossMarket_Prob_Gap_Exists',#'Line_Moved_Away_From_Team',
            'Pct_Line_Move_From_Opening','Pct_Line_Move_Bin','Potential_Overmove_Flag',
            'Potential_Overmove_Total_Pct_Flag','Mispricing_Flag',
        
            # 🧠 Cross-market alignment
            'Potential_Odds_Overmove_Flag','Line_Moved_Toward_Team',
            'Abs_Line_Move_Z','Pct_Line_Move_Z','SmallBook_Limit_Skew',
            'SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag',
            #'Book_Reliability_Score',
            'Book_Reliability_Lift','Book_Reliability_x_Sharp',
            #'Book_Reliability_x_Magnitude',
            'Book_Reliability_x_PROB_SHIFT',
        
            # Power ratings / edges
            'PR_Team_Rating','PR_Opp_Rating','PR_Rating_Diff','PR_Abs_Rating_Diff',
            'Outcome_Model_Spread','Outcome_Market_Spread',#'Outcome_Spread_Edge',
            'Outcome_Cover_Prob','model_fav_vs_market_fav_agree','edge_pts',
            'TOT_Proj_Total_Baseline','TOT_Off_H','TOT_Def_H','TOT_Off_A','TOT_Def_A',
            'TOT_GT_H','TOT_GT_A','TOT_LgAvg_Total','TOT_Mispricing'
        ]
        
        # ensure uniqueness (order-preserving)
        _seen = set()
        features = [f for f in features if not (f in _seen or _seen.add(f))]
        
        # build generated groups
        hybrid_timing_features = [
            f'SharpMove_Magnitude_{b}'
            for b in [
                'Overnight_VeryEarly','Overnight_MidRange','Overnight_LateGame','Overnight_Urgent',
                'Early_VeryEarly','Early_MidRange','Early_LateGame','Early_Urgent',
                'Midday_VeryEarly','Midday_MidRange','Midday_LateGame','Midday_Urgent',
                'Late_VeryEarly','Late_MidRange','Late_LateGame','Late_Urgent'
            ]
        ]
        hybrid_odds_timing_features = [
            f'OddsMove_Magnitude_{b}'
            for b in [
                'Overnight_VeryEarly','Overnight_MidRange','Overnight_LateGame','Overnight_Urgent',
                'Early_VeryEarly','Early_MidRange','Early_LateGame','Early_Urgent',
                'Midday_VeryEarly','Midday_MidRange','Midday_LateGame','Midday_Urgent',
                'Late_VeryEarly','Late_MidRange','Late_LateGame','Late_Urgent'
            ]
        ]
        extend_unique(features, hybrid_timing_features)
        extend_unique(features, hybrid_odds_timing_features)
        
        # add historical/streak features you computed earlier (schema-safe list)
        extend_unique(features, [c for c in history_present if c not in features])
        
        # add recent team model performance stats
        extend_unique(features, [
            'Team_Past_Avg_Model_Prob','Team_Past_Hit_Rate',
            'Team_Past_Avg_Model_Prob_Home','Team_Past_Hit_Rate_Home',
            'Team_Past_Avg_Model_Prob_Away','Team_Past_Hit_Rate_Away',
            
            'Avg_Recent_Cover_Streak','Avg_Recent_Cover_Streak_Home',
            'Avg_Recent_Cover_Streak_Away'
        ])
        
        # add time-context flags
        extend_unique(features, ['Is_Weekend','Is_Night_Game','Is_PrimeTime','DOW_Sin','DOW_Cos'])
        
        
        
        # merge view-driven features (all_present) without losing order
        if 'all_present' in locals():
            extend_unique(features, all_present)
        
        # ======= IMPORTANT: work with df_market from here on =======
        
        # Make sure df_market has placeholders for requested feature columns (0 default)
        df_market = ensure_columns(df_market, features, 0)
        
        # Now prune to columns that actually exist (or were just ensured)
        missing_in_market = [c for c in features if c not in df_market.columns]
        if missing_in_market:
            st.write(f"ℹ️ Dropping {len(missing_in_market)} missing feature(s): "
                     f"{sorted(missing_in_market)[:20]}{'...' if len(missing_in_market)>20 else ''}")
        
        features = [c for c in features if c in df_market.columns]
        
        st.markdown(f"### 📈 Features Used: `{len(features)}`")
        
        # final dataset for modeling
        X = df_market[features].apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)
        
        # Correlation check (robust to NaNs/constant columns)
        try:
            corr_matrix = X.corr(numeric_only=True).abs()
            threshold = 0.85
            high_corr_pairs = []
            cols = corr_matrix.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    corr = corr_matrix.iat[i, j]
                    if pd.notna(corr) and corr > threshold:
                        high_corr_pairs.append((cols[i], cols[j], corr))
            if high_corr_pairs:
                df_corr = (pd.DataFrame(high_corr_pairs, columns=['Feature_1','Feature_2','Correlation'])
                             .sort_values('Correlation', ascending=False))
                title_market = market.upper() if 'market' in locals() else 'MARKET'
                st.subheader(f"🔁 Highly Correlated Features — {title_market}")
                st.dataframe(df_corr)
            else:
                st.success("✅ No highly correlated feature pairs found")
        except Exception as e:
            st.write(f"Correlation check skipped: {e}")
        
        # target
        if 'SHARP_HIT_BOOL' not in df_market.columns:
            st.warning("⚠️ Missing SHARP_HIT_BOOL in df_market — skipping.")
            # handle skip/continue in your loop as appropriate
        else:
            y = pd.to_numeric(df_market['SHARP_HIT_BOOL'], errors='coerce').fillna(0).astype(int)
            if y.nunique() < 2:
                title_market = market.upper() if 'market' in locals() else 'MARKET'
                st.warning(f"⚠️ Skipping {title_market} — only one label class.")
        # continue / return in your loop
      
        # ===============================
        # Purged Group Time-Series CV (PGTSCV) + Embargo
        # ===============================
        # Normalize the sport string for lookup
        # Use the passed-in `sport` exactly as-is (case-sensitive mapping + default)
       
       
        
        # --- sport → embargo (top of file, once) ---
        SPORT_EMBARGO = {
            "MLB":  pd.Timedelta("12 hours"),
            "NBA":  pd.Timedelta("12 hours"),
            "NHL":  pd.Timedelta("12 hours"),
            "NCAAB": pd.Timedelta("12 hours"),
            "NFL":  pd.Timedelta("3 days"),
            "NCAAF": pd.Timedelta("2 days"),
            "WNBA":  pd.Timedelta("24 hours"),
            "MLS":   pd.Timedelta("24 hours"),
            "default": pd.Timedelta("12 hours"),
        }
        def get_embargo_for_sport(sport: str) -> pd.Timedelta:
            return SPORT_EMBARGO.get(sport, SPORT_EMBARGO["default"])
        


        class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
            """
            Time-ordered, group-based CV with purge + time embargo.
        
            - Groups (e.g., Game_Key) never straddle train/val.
            - Any group overlapping the validation time window is *purged* from train.
            - Any group starting within [val_start - embargo, val_end + embargo] is embargoed from train.
            - Folds are contiguous in time at the group level.
            """
        
            def __init__(self, n_splits=5, embargo=pd.Timedelta("0 hours"), time_values=None, min_val_size=20):
                """
                Parameters
                ----------
                n_splits : int
                    Number of time-ordered folds over *groups*.
                embargo : pd.Timedelta
                    Symmetric embargo applied around the validation window.
                time_values : 1D datetime-like aligned to X rows
                    Timestamps for each row (e.g., Game_Start or Snapshot_Timestamp).
                min_val_size : int
                    Minimum number of validation rows required for a fold to be yielded.
                """
                if n_splits < 2:
                    raise ValueError("n_splits must be at least 2")
                self.n_splits = int(n_splits)
                self.embargo = pd.Timedelta(embargo)
                self.time_values = time_values
                self.min_val_size = int(min_val_size)
        
            def get_n_splits(self, X=None, y=None, groups=None):
                return self.n_splits
        
            def split(self, X, y=None, groups=None):
                if groups is None:
                    raise ValueError("groups must be provided to split()")
                if self.time_values is None:
                    raise ValueError("time_values must be set on the splitter")
        
                # Align & normalize inputs
                if len(groups) != len(self.time_values):
                    raise ValueError("groups and time_values must be aligned to X rows")
        
                meta = pd.DataFrame({
                    "group": np.asarray(groups),
                    "time":  pd.to_datetime(self.time_values, errors="coerce", utc=True)
                })
                if meta["time"].isna().any():
                    raise ValueError("time_values contain NaT after to_datetime; check your inputs")
        
                # One row per group with start/end times, ordered by start time
                gmeta = (meta.groupby("group", as_index=False)["time"]
                             .agg(start="min", end="max")
                             .sort_values("start")
                             .reset_index(drop=True))
        
                n_groups = len(gmeta)
                if n_groups < self.n_splits:
                    # relax to feasible number of splits, minimum 2
                    self.n_splits = max(2, n_groups)
        
                # contiguous group folds
                fold_sizes = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
                fold_sizes[: n_groups % self.n_splits] += 1
                edges = np.cumsum(fold_sizes)
        
                start = 0
                for k, stop in enumerate(edges):
                    val_slice = gmeta.iloc[start:stop]
                    start = stop
        
                    if val_slice.empty:
                        continue
        
                    val_groups = val_slice["group"].to_numpy()
                    val_start  = val_slice["start"].iloc[0]
                    val_end    = val_slice["end"].iloc[-1]
        
                    # 1) PURGE: remove any group that overlaps validation window
                    overlap_mask = ~((gmeta["end"] < val_start) | (gmeta["start"] > val_end))
                    purged_groups = set(gmeta.loc[overlap_mask, "group"])
        
                    # 2) EMBARGO: drop groups whose *start* falls inside the embargo window around validation
                    emb_lo = val_start - self.embargo
                    emb_hi = val_end + self.embargo
                    embargo_mask = (gmeta["start"] >= emb_lo) & (gmeta["start"] <= emb_hi)
                    embargo_groups = set(gmeta.loc[embargo_mask, "group"])
        
                    # Training groups = not in val, not purged, not embargoed
                    train_groups = gmeta.loc[
                        ~gmeta["group"].isin(val_groups)
                        & ~gmeta["group"].isin(purged_groups)
                        & ~gmeta["group"].isin(embargo_groups),
                        "group"
                    ].to_numpy()
        
                    # Map back to row indices
                    all_groups = meta["group"].to_numpy()
                    val_idx   = np.flatnonzero(np.isin(all_groups, val_groups))
                    train_idx = np.flatnonzero(np.isin(all_groups, train_groups))
        
                    # HARDENING
                    if len(val_idx) == 0 or len(train_idx) == 0:
                        continue
                    if len(val_idx) < self.min_val_size:
                        continue
                    if y is not None:
                        y_arr = np.asarray(y)
                        # need both classes in validation to drive stable early stopping / metrics
                        if np.unique(y_arr[val_idx]).size < 2:
                            continue
        
                    yield train_idx, val_idx
       
        # 1) Make sure all model features are numeric
        df_market[features] = df_market[features].apply(pd.to_numeric, errors="coerce")
        
        # 2) Convert pandas NA to np.nan, then remove infs and fill
        X_full = (df_market[features]
                  .replace({pd.NA: np.nan})
                  .replace([np.inf, -np.inf], np.nan)
                  .fillna(0.0)
                  .to_numpy(dtype=np.float32))
        
        # 3) Labels: coerce and drop NA rows if any slipped through
        y_series = pd.to_numeric(df_market["SHARP_HIT_BOOL"], errors="coerce")
        valid_mask = ~y_series.isna()
        
        # (optional) Keep rows where label is valid; also ensure features already aligned
        if not valid_mask.all():
            X_full = X_full[valid_mask.to_numpy()]
        y_full = y_series.loc[valid_mask].astype(int).to_numpy()
        
        # 4) Groups & times: ensure no NAType
        groups = resolve_groups(df_market.loc[valid_mask]).astype(str)  # your helper returns ndarray
        times  = pd.to_datetime(
            df_market.loc[valid_mask, "Snapshot_Timestamp"], errors="coerce", utc=True
        ).to_numpy()
        
        # 5) Quick guard: no NaT in times
        if pd.isna(times).any():
            bad = np.flatnonzero(pd.isna(times))
            # drop bad rows
            keep = np.setdiff1d(np.arange(len(times)), bad)
            X_full = X_full[keep]
            y_full = y_full[keep]
            groups = groups[keep]
            times  = times[keep]
        # 0) Build matrices and folds FIRST
   
        # class balance for scale_pos_weight
        pos = y_full.sum()
        neg = len(y_full) - pos
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        
        # sport can be "WNBA", "MLB", etc.
        sport_key  = str(sport).upper()
        embargo_td = SPORT_EMBARGO.get(sport_key, SPORT_EMBARGO["default"])
        
        # CV splitter (after building X_full, y_full, groups, times)
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=3,
            embargo=embargo_td,          # e.g., pd.Timedelta("8h")
            time_values=times,
            min_val_size=20,
        )
        
        folds = list(cv.split(X_full, y_full, groups=groups))
        assert folds, "No usable folds"
        
        # sport-aware kwargs & search space
        base_kwargs, param_distributions = get_xgb_search_space(
            sport=sport,
            X_rows=X_full.shape[0],
            n_jobs=2,  # CV-level parallelism; keep base_kwargs['n_jobs']=1 inside the helper
            scale_pos_weight=scale_pos_weight,
            features=features,
        )
        
        
        DEBUG_ONCE = True
        
        # lock classifier objective/metric
        base_kwargs["objective"]   = "binary:logistic"
        base_kwargs["eval_metric"] = "logloss"
        base_kwargs["n_jobs"]      = 1  # avoid core oversubscription
        
        # sanity: keep unsafe keys out of the search space
        assert "objective"   not in param_distributions
        assert "eval_metric" not in param_distributions
        
        search_estimators = 600
        search_base = XGBClassifier(**{**base_kwargs, "n_estimators": search_estimators})
        
        # assert classifier
        assert isinstance(search_base, XGBClassifier)
        assert getattr(search_base, "_estimator_type", "") == "classifier"
        
        # --- one-time debug (optional) ---
      
        
        def neg_logloss_scorer(est, X, y):
            proba = est.predict_proba(X)[:, 1]   # will raise if regressor
            return -log_loss(y, proba)
        
        def roc_auc_proba_scorer(est, X, y):
            proba = est.predict_proba(X)[:, 1]
            return roc_auc_score(y, proba)
        
        # 3) debug scorer to *show* the bad candidate in Streamlit
        def neg_logloss_scorer_debug(est, X, y):
            try:
                proba = est.predict_proba(X)[:, 1]
                return -log_loss(y, proba)
            except Exception as e:
                # Surface details in the Streamlit UI
                try:
                    xgb_params = est.get_xgb_params()
                except Exception:
                    xgb_params = {}
                st.error("Offending candidate encountered during CV:")
                st.json({
                    "estimator_type": type(est).__name__,
                    "_estimator_type": getattr(est, "_estimator_type", None),
                    "objective": getattr(est, "objective", None),
                    "xgb_params": xgb_params,
                })
                st.exception(e)
                # Re-raise so the run stops here and you can fix it
                raise
        
        DEBUG_ONCE = True  # flip to False after one run
        
        if DEBUG_ONCE:
            try:
                rs_dbg = RandomizedSearchCV(
                    estimator=search_base,
                    param_distributions=param_distributions,
                    scoring=neg_logloss_scorer_debug,  # <— debug scorer
                    cv=folds,
                    n_iter=5,      # small, fast
                    n_jobs=1,      # simpler trace
                    verbose=2,
                    random_state=7,
                    refit=True,
                    error_score="raise",
                )
                rs_dbg.fit(X_full, y_full, groups=groups)
                st.success("DEBUG: All candidates were classifiers with predict_proba ✅")
            except Exception:
                st.stop()  # stop Streamlit run after showing details

       
        try:
            rs_ll = RandomizedSearchCV(
                estimator=search_base,
                param_distributions=param_distributions,
                scoring=neg_logloss_scorer,   # your callable scorer
                cv=folds,
                n_iter=40,
                n_jobs=3,
                verbose=1,
                random_state=42,
                refit=True,
                error_score="raise",          # <-- force raise instead of silent -inf
            )
            rs_ll.fit(X_full, y_full, groups=groups)
        except Exception as e:
            st.error("❌ Error during RandomizedSearchCV (logloss search)")
            st.exception(e)
            st.stop()   # stop Streaml
        
        
        try:
            rs_auc = RandomizedSearchCV(
                estimator=search_base,
                param_distributions=param_distributions,
                scoring="roc_auc",
                cv=folds,
                n_iter=40,
                n_jobs=3,
                verbose=1,
                random_state=4242,
                refit=True,
                error_score="raise",
            )
            rs_auc.fit(X_full, y_full, groups=groups)
        except Exception as e:
            st.error("❌ Error during RandomizedSearchCV (AUC search)")
            st.exception(e)
            st.stop()   # stop Streaml
        
        
        # clean best params (strip any unsafe keys)
        best_ll_params  = rs_ll.best_params_.copy()
        best_auc_params = rs_auc.best_params_.copy()
        for k in ("objective", "eval_metric", "_estimator_type"):
            best_ll_params.pop(k, None)
            best_auc_params.pop(k, None)
        
        # --- final refit with early stopping on forward holdout (last fold) ---
        (train_idx, val_idx) = folds[-1]
        final_estimators_cap  = 3000
        early_stopping_rounds = 100
        
        model_logloss = XGBClassifier(**base_kwargs, **best_ll_params, n_estimators=final_estimators_cap)
        model_logloss.fit(
            X_full[train_idx], y_full[train_idx],
            eval_set=[(X_full[val_idx], y_full[val_idx])],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        
        model_auc = XGBClassifier(**base_kwargs, **best_auc_params, n_estimators=final_estimators_cap)
        model_auc.fit(
            X_full[train_idx], y_full[train_idx],
            eval_set=[(X_full[val_idx], y_full[val_idx])],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        # ---- helper: safest best-round extraction across xgboost versions ----
        def best_rounds(clf: XGBClassifier) -> int:
            # try sklearn attr
            br = getattr(clf, "best_iteration", None)
            if br is not None and br >= 0:
                return int(br + 1)  # iterations are 0-based; n_estimators is count
            # try booster attrs
            try:
                booster = clf.get_booster()
                if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
                    return int(booster.best_iteration + 1)
                if hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit:
                    return int(booster.best_ntree_limit)
            except Exception:
                pass
            # fallback to set n_estimators if early stopping wasn’t used
            return int(getattr(clf, "n_estimators", 200))
        
        n_trees_ll  = best_rounds(model_logloss)
        n_trees_auc = best_rounds(model_auc)
        
        # ---- OOF predictions (single pass over 'folds') ----
        oof_pred_logloss = np.full(len(y_full), np.nan, dtype=float)
        oof_pred_auc     = np.full(len(y_full), np.nan, dtype=float)
        
        for tr_idx, va_idx in folds:
            m_ll  = XGBClassifier(**base_kwargs, **best_ll_params,  n_estimators=n_trees_ll)
            m_auc = XGBClassifier(**base_kwargs, **best_auc_params, n_estimators=n_trees_auc)
            m_ll.fit(X_full[tr_idx],  y_full[tr_idx],  verbose=False)
            m_auc.fit(X_full[tr_idx], y_full[tr_idx], verbose=False)
            oof_pred_logloss[va_idx] = m_ll.predict_proba(X_full[va_idx])[:, 1]
            oof_pred_auc[va_idx]     = m_auc.predict_proba(X_full[va_idx])[:, 1]
        
        # ---- Isotonic on the blended OOF predictions ----
        mask_oof = ~np.isnan(oof_pred_logloss) & ~np.isnan(oof_pred_auc)
        if mask_oof.sum() < 50:
            raise RuntimeError("Too few OOF predictions to fit isotonic calibration.")
        
        oof_blend = 0.5 * oof_pred_logloss[mask_oof] + 0.5 * oof_pred_auc[mask_oof]
        y_oof     = y_full[mask_oof].astype(int)
        
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof_blend, y_oof)
        
        # Optional: wrapper so we can use predict_proba later
        
        
        # ----- final refit with early stopping on most-recent 15% -----
        # --- final time-forward holdout split (15% tail) ---
        n = len(X_full)
        hold = max(1, int(round(n * 0.15)))
        
        # indices for validation rows in the ORIGINAL order
        val_idx = np.arange(n - hold, n)
        
        # split arrays
        X_tr = X_full[: n - hold]
        y_tr = y_full[: n - hold].astype(int)
        
        X_val = X_full[val_idx]                                  # NumPy array (fine for model.predict_proba)
        y_val = pd.Series(y_full[val_idx].astype(int), index=val_idx)  # Pandas Series so .nunique(), alignment, merges work
                # --- helper for robust early stopping ---
        def fit_with_es(model, X_tr, y_tr, X_va, y_va, es_rounds):
            """Try native early_stopping_rounds; fallback to callback API if needed."""
            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    early_stopping_rounds=es_rounds,
                    verbose=False,
                )
            except TypeError:
                es = xgb.callback.EarlyStopping(
                    rounds=es_rounds,
                    save_best=True,
                    maximize=False
                )
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[es],
                    verbose=False,
                )
            return model
        es = xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True, maximize=False)
        
        final_log = xgb.XGBClassifier(
            **{**base_kwargs, **model_logloss.get_params(), "n_estimators": final_estimators_cap}
        )
        final_auc = xgb.XGBClassifier(
            **{**base_kwargs, **model_auc.get_params(), "n_estimators": final_estimators_cap}
        )
    
        final_log = fit_with_es(final_log, X_tr, y_tr, X_val, y_val.values, early_stopping_rounds)
        final_auc = fit_with_es(final_auc, X_tr, y_tr, X_val, y_val.values, early_stopping_rounds)

        
        class IsoWrapper:
            def __init__(self, base, iso):
                self.base = base
                self.iso = iso
            def predict_proba(self, X):
                p = self.base.predict_proba(X)[:, 1]
                p_cal = np.clip(self.iso.transform(p), 1e-6, 1-1e-6)
                return np.vstack([1 - p_cal, p_cal]).T
        
        cal_logloss = IsoWrapper(final_log, iso)
        cal_auc     = IsoWrapper(final_auc, iso)
        
        # ----- holdout metrics (blended + calibrated) -----
        # Use X_val / y_val (new names) instead of X_va / y_va
        p_ll    = cal_logloss.predict_proba(X_val)[:, 1]
        p_au    = cal_auc.predict_proba(X_val)[:, 1]
        p_blend = 0.5 * p_ll + 0.5 * p_au
        p_cal   = np.clip(p_blend, 1e-6, 1-1e-6)
        
        def safe_auc(y, p):   return roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan
        def safe_ll(y, p):    return log_loss(y, p, labels=[0,1]) if np.unique(y).size == 2 else np.nan
        def safe_brier(y, p): return brier_score_loss(y, p) if np.unique(y).size == 2 else np.nan
        
        auc = safe_auc(y_val.values, p_cal)
        ll  = safe_ll(y_val.values, p_cal)
        bri = safe_brier(y_val.values, p_cal)

        
        st.markdown(f"### 🧪 Holdout Validation — `{market.upper()}` (purged-CV tuned, time-forward holdout)")
        st.write(f"- Blended+Calibrated AUC: `{auc:.4f}`")
        st.write(f"- Holdout LogLoss:        `{ll:.4f}`")
        st.write(f"- Holdout Brier:          `{bri:.4f}`")


        
        # --- blended + calibrated validation probabilities aligned to original rows ---
        # assumes you already defined: val_idx, X_val (np.array), y_val (pd.Series with index=val_idx)
        p_ll    = cal_logloss.predict_proba(X_val)[:, 1]
        p_au    = cal_auc.predict_proba(X_val)[:, 1]
        p_blend = 0.5 * p_ll + 0.5 * p_au
        p_cal   = np.clip(p_blend, 1e-6, 1-1e-6)
        
        # Build aligned evaluation frame (index = original df rows for the val slice)
        df_eval = pd.DataFrame({
            "p":    pd.Series(p_cal, index=val_idx),
            "y":    y_val.astype(int),  # Series with val_idx
            "odds": pd.to_numeric(df_market.loc[val_idx, "Odds_Price"], errors="coerce"),
        }).dropna(subset=["p", "y"])   # keep rows with both prob and label
        
        # --- probability bins (0.0–1.0 by 0.1) ---
        bins   = np.linspace(0, 1, 11)
        labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
        cuts   = pd.cut(df_eval["p"], bins=bins, include_lowest=True, labels=labels)
        
        # --- inline ROI calc: American odds to unit ROI ---
        def _roi_mean_inline(sub: pd.DataFrame) -> float:
            if not sub["odds"].notna().any():
                return float("nan")
            odds = pd.to_numeric(sub["odds"], errors="coerce")
            win  = sub["y"].astype(int)
        
            # profit if win: +odds/100 for positive odds, +100/|odds| for negative odds
            profit_pos = odds.where(odds > 0, np.nan) / 100.0
            profit_neg = 100.0 / odds.abs()
            profit_on_win = np.where(odds > 0, profit_pos, profit_neg)
            profit_on_win = pd.Series(profit_on_win, index=odds.index).fillna(0.0)
        
            roi = win * profit_on_win - (1 - win) * 1.0   # lose stake if loss
            return float(roi.mean())
        
        # --- aggregate per bin
        rows = []
        for lb in labels:
            sub = df_eval[cuts == lb]
            n   = int(len(sub))
            if n == 0:
                rows.append((lb, 0, np.nan, np.nan, np.nan))
                continue
            hit   = float(sub["y"].mean())
            roi   = _roi_mean_inline(sub)
            avg_p = float(sub["p"].mean())
            rows.append((lb, n, hit, roi, avg_p))
        
        df_bins = pd.DataFrame(rows, columns=["Prob Bin", "N", "Hit Rate", "Avg ROI (unit)", "Avg Pred P"])
        df_bins["N"] = df_bins["N"].astype(int)
        
        st.markdown("#### 🎯 Calibration Bins (blended + calibrated)")
        st.dataframe(df_bins)
        
        # quick extreme-bucket snapshot (only if non-empty)
        hi = df_bins.iloc[-1]
        lo = df_bins.iloc[0]
        st.write(f"**High bin ({hi['Prob Bin']}):** " + (f"N={hi['N']}, Hit={hi['Hit Rate']:.3f}, ROI={hi['Avg ROI (unit)']:.3f}" if hi["N"] > 0 else "N=0"))
        st.write(f"**Low bin  ({lo['Prob Bin']}):** " + (f"N={lo['N']}, Hit={lo['Hit Rate']:.3f}, ROI={lo['Avg ROI (unit)']:.3f}" if lo["N"] > 0 else "N=0"))
        
        # === Summary Card =====================================================
        st.markdown("### ✅ Deployment Readiness Snapshot")
        psi_flag = "Unknown"
        if 'df_psi' in locals():
            psi_worst = df_psi['PSI'].replace(np.nan, 0).max()
            psi_flag = "🚨 heavy drift" if psi_worst > 0.3 else ("⚠️ medium drift" if psi_worst > 0.2 else "✅ stable")
        
        hi_ok = (hi["N"] > 0) and ( (hi["Hit Rate"] >= 0.60) or (pd.notna(hi["Avg ROI (unit)"]) and hi["Avg ROI (unit)"] > 0) )
        lo_ok = (lo["N"] > 0) and ( ((1 - lo["Hit Rate"]) >= 0.60) or (pd.notna(lo["Avg ROI (unit)"]) and lo["Avg ROI (unit)"] > 0) )
        
        st.write(f"- **PSI status:** {psi_flag}")
        st.write(f"- **High-confidence bucket profitable/accurate?** {'✅' if hi_ok else '⚠️'}")
        st.write(f"- **Low-confidence bucket fade profitable/accurate?** {'✅' if lo_ok else '⚠️'}")
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
        # Start with Game_Key/Team unique combos
        # --- 0) one row per (Game_Key, Team) to merge everything onto
        df_team_base = df_bt_prepped[['Game_Key','Team']].drop_duplicates()
        
        # --- 1) LOO stats (already built above as df_bt_loostats)
        df_team_base = df_team_base.merge(df_bt_loostats, on=['Game_Key','Team'], how='left')
        
        # --- 2) Streaks: ensure dfb has Game_Key, then merge
        # If dfb doesn't already have Game_Key, add it from df_bt_prepped (safe merge)
        if 'Game_Key' not in dfb.columns:
            dfb = dfb.merge(
                df_bt_prepped[['Game_Key','Team','Snapshot_Timestamp']].drop_duplicates(),
                on=['Team','Snapshot_Timestamp'],
                how='left'
            )
        
        streak_cols = [
            'Team_Recent_Cover_Streak','Team_Recent_Cover_Streak_Home','Team_Recent_Cover_Streak_Away',
            'Team_Recent_Cover_Streak_Fav','Team_Recent_Cover_Streak_Home_Fav','Team_Recent_Cover_Streak_Away_Fav',
            'On_Cover_Streak','On_Cover_Streak_Home','On_Cover_Streak_Away',
            'On_Cover_Streak_Fav','On_Cover_Streak_Home_Fav','On_Cover_Streak_Away_Fav'
        ]
        
        df_team_base = df_team_base.merge(
            dfb[['Game_Key','Team'] + streak_cols].drop_duplicates(),
            on=['Game_Key','Team'],
            how='left'
        )
        
        # --- 3) collapse to one row per Team (final team_feature_map)
        team_feature_map = (
            df_team_base.groupby('Team', as_index=False)
            .agg({
                # LOO stats
                'Team_Past_Avg_Model_Prob': 'mean',
                'Team_Past_Hit_Rate': 'mean',
                'Team_Past_Avg_Model_Prob_Home': 'mean',
                'Team_Past_Hit_Rate_Home': 'mean',
                'Team_Past_Avg_Model_Prob_Away': 'mean',
                'Team_Past_Hit_Rate_Away': 'mean',
                'Team_Past_Avg_Model_Prob_Fav': 'mean',
                'Team_Past_Hit_Rate_Fav': 'mean',
                'Team_Past_Avg_Model_Prob_Home_Fav': 'mean',
                'Team_Past_Hit_Rate_Home_Fav': 'mean',
                'Team_Past_Avg_Model_Prob_Away_Fav': 'mean',
                'Team_Past_Hit_Rate_Away_Fav': 'mean',
        
                # Streak metrics
                'Team_Recent_Cover_Streak': 'mean',
                'Team_Recent_Cover_Streak_Home': 'mean',
                'Team_Recent_Cover_Streak_Away': 'mean',
                'Team_Recent_Cover_Streak_Fav': 'mean',
                'Team_Recent_Cover_Streak_Home_Fav': 'mean',
                'Team_Recent_Cover_Streak_Away_Fav': 'mean',
                'On_Cover_Streak': 'mean',
                'On_Cover_Streak_Home': 'mean',
                'On_Cover_Streak_Away': 'mean',
                'On_Cover_Streak_Fav': 'mean',
                'On_Cover_Streak_Home_Fav': 'mean',
                'On_Cover_Streak_Away_Fav': 'mean',
            })
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


def train_timing_opportunity_model(sport: str = "NBA", days_back: int = 35):
    st.info(f"🧠 Training timing opportunity models for {sport.upper()}...")

    # === Load historical scored data
    query = f"""
        SELECT *
        FROM `sharplogger.sharp_data.scores_with_features`
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
        
def attach_ratings_and_edges_for_diagnostics(
    df: pd.DataFrame,
    sport_aliases: dict,
    table_history: str = "sharplogger.sharp_data.ratings_current",  # ← current table by default
    project: str = "sharplogger",
    pad_days: int = 30,     # normal pad for history tables
    allow_forward_hours: float = 0.0,
) -> pd.DataFrame:
    
    UI_EDGE_COLS = [
        'PR_Team_Rating','PR_Opp_Rating','PR_Rating_Diff',
        'Outcome_Model_Spread','Outcome_Market_Spread','Outcome_Spread_Edge',
        'Outcome_Cover_Prob','model_fav_vs_market_fav_agree','edge_x_k','mu_x_k'
    ]

    # Fast exit on empty input — keep UI cols present
    if df.empty:
        out = df.copy()
        for c in UI_EDGE_COLS:
            out[c] = np.nan
        return out

    out = df.copy()

    # ---- Safe normalizations (Series-safe defaults)
    def _series(col, default=""):
        return out[col] if col in out.columns else pd.Series(default, index=out.index)

    out['Sport']          = _series('Sport').astype(str).str.upper().str.strip()
    out['Market']         = _series('Market').astype(str).str.lower().str.strip()
    out['Home_Team_Norm'] = _series('Home_Team_Norm').astype(str).str.lower().str.strip()
    out['Away_Team_Norm'] = _series('Away_Team_Norm').astype(str).str.lower().str.strip()
    out['Outcome_Norm']   = (
        _series('Outcome_Norm', None)
        .where((_series('Outcome_Norm', None).notna()) if 'Outcome_Norm' in out else False,
               _series('Outcome'))
        .astype(str).str.lower().str.strip()
    )

    # Game_Start fallback to Snapshot_Timestamp
    if 'Game_Start' not in out.columns or out['Game_Start'].isna().all():
        out['Game_Start'] = pd.to_datetime(_series('Snapshot_Timestamp', pd.NaT),
                                           utc=True, errors='coerce')

    # We only compute edges for spreads
    mask = out['Market'].eq('spreads') & out['Home_Team_Norm'].ne('') & out['Away_Team_Norm'].ne('')
    if not mask.any():
        for c in UI_EDGE_COLS:
            if c not in out.columns:
                out[c] = np.nan
        return out

    # Minimal join key back into the original rows
    need_cols = ['Sport','Game_Start','Home_Team_Norm','Away_Team_Norm','Outcome_Norm','Value']
    d_sp = (
        out.loc[mask, need_cols]
           .dropna(subset=['Sport','Home_Team_Norm','Away_Team_Norm','Outcome_Norm','Value'])
           .copy()
    )
    d_sp['Value'] = pd.to_numeric(d_sp['Value'], errors='coerce').astype('float32')

    # ---------- KEY FIX: widen the ratings window when using *current* table ----------
    is_current_table = 'ratings_current' in str(table_history).lower()
    pad_days_eff = (365 if is_current_table else pad_days)  # wide window so older “Updated_At” is included

    # 1) Ratings (as-of, low-mem)
    base = enrich_power_for_training_lowmem(
        df=d_sp[['Sport','Home_Team_Norm','Away_Team_Norm','Game_Start']].drop_duplicates(),
        bq=None,
        sport_aliases=sport_aliases,
        table_history=table_history,           # ← points at ratings_current by default
        pad_days=pad_days_eff,                 # ← wide pad for current table
        allow_forward_hours=allow_forward_hours,
        project=project,
    )

    # 2) Consensus favorite & k
    cons = prep_consensus_market_spread_lowmem(d_sp, value_col='Value', outcome_col='Outcome_Norm')

    # 3) Game-level model spreads/edges
    game_keys = ['Sport','Home_Team_Norm','Away_Team_Norm']
    g_full = base.merge(cons, on=game_keys, how='left')
    g_fc   = favorite_centric_from_powerdiff_lowmem(g_full)

    # Ensure PR columns exist (carry from g_full or set NaN)
    for c in ['Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff']:
        if c not in g_fc.columns:
            g_fc[c] = g_full[c] if c in g_full.columns else np.nan

    # Map to the row’s bet side
    d_map = d_sp.merge(g_fc, on=game_keys, how='left')

    is_fav_row = d_map['Outcome_Norm'].eq(d_map['Market_Favorite_Team'])
    d_map['Outcome_Model_Spread']  = np.where(is_fav_row, d_map['Model_Fav_Spread'], d_map['Model_Dog_Spread']).astype('float32')
    d_map['Outcome_Market_Spread'] = np.where(is_fav_row, d_map['Favorite_Market_Spread'], d_map['Underdog_Market_Spread']).astype('float32')
    d_map['Outcome_Spread_Edge']   = np.where(is_fav_row, d_map['Fav_Edge_Pts'], d_map['Dog_Edge_Pts']).astype('float32')
    d_map['Outcome_Cover_Prob']    = np.where(is_fav_row, d_map['Fav_Cover_Prob'], d_map['Dog_Cover_Prob']).astype('float32')

    # Ratings shown for the bet side (home vs away)
    is_home_bet = d_map['Outcome_Norm'].eq(d_map['Home_Team_Norm'])
    d_map['PR_Team_Rating'] = np.where(is_home_bet, d_map['Home_Power_Rating'], d_map['Away_Power_Rating']).astype('float32')
    d_map['PR_Opp_Rating']  = np.where(is_home_bet, d_map['Away_Power_Rating'], d_map['Home_Power_Rating']).astype('float32')
    d_map['PR_Rating_Diff'] = (
        pd.to_numeric(d_map['PR_Team_Rating'], errors='coerce') -
        pd.to_numeric(d_map['PR_Opp_Rating'],  errors='coerce')
    ).astype('float32')

    # Agreement & scaled edges
    k_abs = (
        pd.to_numeric(d_map.get('Favorite_Market_Spread'),  errors='coerce').abs()
          .combine_first(pd.to_numeric(d_map.get('Underdog_Market_Spread'), errors='coerce').abs())
    ).astype('float32').where(lambda s: s > 0, np.nan)

    d_map['model_fav_vs_market_fav_agree'] = (
        (pd.to_numeric(d_map['Outcome_Model_Spread'], errors='coerce') < 0) ==
        d_map['Outcome_Norm'].eq(d_map['Market_Favorite_Team'])
    ).astype('int8')

    d_map['edge_x_k'] = (pd.to_numeric(d_map['Outcome_Spread_Edge'], errors='coerce') * k_abs).astype('float32')
    d_map['mu_x_k']   = (pd.to_numeric(d_map['Outcome_Spread_Edge'], errors='coerce').abs() * k_abs).astype('float32')

    # Clean merge back (avoid suffixes)
    out.drop(columns=UI_EDGE_COLS, inplace=True, errors='ignore')
    out = out.merge(d_map[need_cols + UI_EDGE_COLS], on=need_cols, how='left')

    # Ensure UI cols always exist
    for c in UI_EDGE_COLS:
        if c not in out.columns:
            out[c] = np.nan

    return out
def compute_diagnostics_vectorized(df):
   
    df = df.copy()
   
    df = (
        df.sort_values('Snapshot_Timestamp')
        .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='last')
    )
    df = attach_ratings_and_edges_for_diagnostics(
        df=df,
        sport_aliases=SPORT_ALIASES,           # your existing global/map
        table_history="sharplogger.sharp_data.ratings_current",
        project="sharplogger",
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
    prob_now = pd.to_numeric(df.get('Model Prob'), errors='coerce')
    prob_start = pd.to_numeric(df.get('First_Sharp_Prob'), errors='coerce')

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
        lambda row: -1 if str(row.get('Market','')).lower() == 'totals' and str(row.get('Outcome','')).lower() == 'under' else 1,
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

    # ----------------------------------------------------------------
    # 💡 MODEL FEATURES: ensure presence, casting, and hybrid flags
    # ----------------------------------------------------------------
    # Full modeling feature list you provided (order-preserving)
    features = [
        # Core sharp signals
        'Sharp_Move_Signal','Sharp_Limit_Jump','Sharp_Limit_Total',
        'Is_Reinforced_MultiMarket','Market_Leader','LimitUp_NoMove_Flag',

        # Market response
        'Sharp_Line_Magnitude',
        'Team_Implied_Prob_Gap_Home','Team_Implied_Prob_Gap_Away',

        # Engineered odds shift decomposition
        'SharpMove_Odds_Up','SharpMove_Odds_Down','SharpMove_Odds_Mag',

        # Engineered interactions
        'MarketLeader_ImpProbShift','LimitProtect_SharpMag','Delta_Sharp_vs_Rec',
    
        # Reversal logic
        'Value_Reversal_Flag','Odds_Reversal_Flag',

        # Mispricing + cross-market alignment
        'Market_Mispricing','Abs_Market_Mispricing',
        'Spread_vs_H2H_Aligned','Total_vs_Spread_Contradiction',
        'Spread_vs_H2H_ProbGap','Total_vs_H2H_ProbGap',
        'Total_vs_Spread_ProbGap','CrossMarket_Prob_Gap_Exists',

        # Movement diagnostics
        'Line_Moved_Away_From_Team','Pct_Line_Move_From_Opening','Pct_Line_Move_Bin',
        'Potential_Overmove_Flag','Potential_Overmove_Total_Pct_Flag','Mispricing_Flag',
        'Potential_Odds_Overmove_Flag','Line_Moved_Toward_Team',
        # 'Abs_Line_Move_Z',  # (optional in display)
        'Pct_Line_Move_Z',

        # Small-book reliability / liquidity
        'SmallBook_Limit_Skew','SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag',
        'Book_Reliability_Lift','Book_Reliability_x_Sharp',
        'Book_Reliability_x_Magnitude','Book_Reliability_x_PROB_SHIFT',

        # Power ratings & model/market edges
        'PR_Team_Rating','PR_Opp_Rating','PR_Rating_Diff',
        'Outcome_Model_Spread','Outcome_Market_Spread','Outcome_Spread_Edge',
        'Outcome_Cover_Prob','model_fav_vs_market_fav_agree','edge_x_k','mu_x_k',
    ]

    # Hybrid timing features (line + odds)
    hybrid_timing_features = [
        f'SharpMove_Magnitude_{b}' for b in [
            'Overnight_VeryEarly','Overnight_MidRange','Overnight_LateGame','Overnight_Urgent',
            'Early_VeryEarly','Early_MidRange','Early_LateGame','Early_Urgent',
            'Midday_VeryEarly','Midday_MidRange','Midday_LateGame','Midday_Urgent',
            'Late_VeryEarly','Late_MidRange','Late_LateGame','Late_Urgent'
        ]
    ]
    hybrid_odds_timing_features = [
        f'OddsMove_Magnitude_{b}' for b in [
            'Overnight_VeryEarly','Overnight_MidRange','Overnight_LateGame','Overnight_Urgent',
            'Early_VeryEarly','Early_MidRange','Early_LateGame','Early_Urgent',
            'Midday_VeryEarly','Midday_MidRange','Midday_LateGame','Midday_Urgent',
            'Late_VeryEarly','Late_MidRange','Late_LateGame','Late_Urgent'
        ]
    ]
    features += hybrid_timing_features + hybrid_odds_timing_features

    # Team history & streaks
    features += [
        'Team_Past_Avg_Model_Prob','Team_Past_Hit_Rate',
        'Team_Past_Avg_Model_Prob_Home','Team_Past_Hit_Rate_Home',
        'Team_Past_Avg_Model_Prob_Away','Team_Past_Hit_Rate_Away',
        'Team_Past_Avg_Model_Prob_Fav','Team_Past_Hit_Rate_Fav',
        'Team_Past_Avg_Model_Prob_Home_Fav','Team_Past_Hit_Rate_Home_Fav',
        'Team_Past_Avg_Model_Prob_Away_Fav','Team_Past_Hit_Rate_Away_Fav',
        'Avg_Recent_Cover_Streak','Avg_Recent_Cover_Streak_Home',
        'Avg_Recent_Cover_Streak_Away','Avg_Recent_Cover_Streak_Fav',
        'Avg_Recent_Cover_Streak_Home_Fav','Avg_Recent_Cover_Streak_Away_Fav',
    ]

    # Context flags (UI)
    features += ['Is_Night_Game','Is_PrimeTime','DOW_Sin','DOW_Cos']
    # Define once
    RATINGS_EDGE_UI_COLS = [
        'PR_Team_Rating','PR_Opp_Rating','PR_Rating_Diff',
        'Outcome_Model_Spread','Outcome_Market_Spread','Outcome_Spread_Edge',
        'Outcome_Cover_Prob','model_fav_vs_market_fav_agree',
        'edge_x_k','mu_x_k'
    ]
    
    # Guarantee presence in df so UI shows them (default NaN is best for visibility)
    for c in RATINGS_EDGE_UI_COLS:
        if c not in df.columns:
            df[c] = np.nan
    _seen = set()
    features = [f for f in features if not (f in _seen or _seen.add(f))]

  
    # Casting helpers
    def _ensure_cols(frame, cols, fill=0.0):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            frame[missing] = fill
        return frame

    # Flag & magnitude groups used by your build_why and display
    flag_cols = [
        'Sharp_Move_Signal','Sharp_Limit_Jump','Market_Leader',
        'Is_Reinforced_MultiMarket','LimitUp_NoMove_Flag','Is_Sharp_Book',
        'SharpMove_Odds_Up','SharpMove_Odds_Down','Is_Home_Team_Bet',
       
        'Late_Game_Steam_Flag',
        'Value_Reversal_Flag','Odds_Reversal_Flag',
        'Hybrid_Line_Timing_Flag','Hybrid_Odds_Timing_Flag'
    ]
    magnitude_cols = [
        'Sharp_Line_Magnitude','Sharp_Time_Score','Rec_Line_Magnitude',
        'Sharp_Limit_Total','SharpMove_Odds_Mag','SharpMove_Timing_Magnitude',
        'Abs_Line_Move_From_Opening','Abs_Odds_Move_From_Opening',
        'Team_Past_Hit_Rate','Team_Past_Avg_Model_Prob',
        'Team_Past_Hit_Rate_Home','Team_Past_Hit_Rate_Away',
        'Team_Past_Avg_Model_Prob_Home','Team_Past_Avg_Model_Prob_Away'
    ]

    # Ensure all model features exist (numeric default 0)
    df = _ensure_cols(df, features, 0.0)

    # Compute Hybrid Line/Odds Timing Flags if missing
    HYBRID_LINE_COLS = hybrid_timing_features
    HYBRID_ODDS_COLS = hybrid_odds_timing_features
    if 'Hybrid_Line_Timing_Flag' not in df.columns:
        df['Hybrid_Line_Timing_Flag'] = (df[HYBRID_LINE_COLS].sum(axis=1) > 0).astype(int)
    if 'Hybrid_Odds_Timing_Flag' not in df.columns:
        df['Hybrid_Odds_Timing_Flag'] = (df[HYBRID_ODDS_COLS].sum(axis=1) > 0).astype(int)

    # Cast flags & magnitudes to numeric
    for col in flag_cols + magnitude_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- make sure aliases/defaults exist *before* build_why runs
    if 'Model Prob' not in df.columns and 'Model_Sharp_Win_Prob' in df.columns:
        df['Model Prob'] = df['Model_Sharp_Win_Prob']

    # sensible default for Passes_Gate if it's missing or NaN
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

    # === build_why (unchanged, uses the casted columns above)
    def build_why(row):
        model_prob = row.get('Model Prob')
        if pd.isna(model_prob):
            return "⚠️ Missing — run apply_blended_sharp_score() first"

        parts = []

        # Core sharp move reasons
        if bool(row.get('Sharp_Move_Signal', 0)): parts.append("📈 Sharp Move Detected")
        if bool(row.get('Sharp_Limit_Jump', 0)): parts.append("💰 Limit Jumped")
        if bool(row.get('Market_Leader', 0)): parts.append("🏆 Market Leader")
        if bool(row.get('Is_Reinforced_MultiMarket', 0)): parts.append("📊 Multi-Market Consensus")
        if bool(row.get('LimitUp_NoMove_Flag', 0)): parts.append("🛡️ Limit Up + No Line Move")
        if bool(row.get('Is_Sharp_Book', 0)): parts.append("🎯 Sharp Book Signal")

        # Odds & line movement
        if bool(row.get('SharpMove_Odds_Up', 0)): parts.append("🟢 Odds Moved Up (Steam)")
        if bool(row.get('SharpMove_Odds_Down', 0)): parts.append("🔻 Odds Moved Down (Buyback)")
        if float(row.get('Sharp_Line_Magnitude', 0)) > 0.5: parts.append("📏 Big Line Move")
        if float(row.get('Rec_Line_Magnitude', 0)) > 0.5: parts.append("📉 Rec Book Move")
        if float(row.get('Sharp_Limit_Total', 0)) > 10000: parts.append("💼 Sharp High Limit")
        if float(row.get('SharpMove_Odds_Mag', 0)) > 5: parts.append("💥 Sharp Odds Steam")

        # Resistance & timing
        
        if bool(row.get('Late_Game_Steam_Flag', 0)): parts.append("⏰ Late Game Steam")
        if bool(row.get('Value_Reversal_Flag', 0)): parts.append("🔄 Value Reversal")
        if bool(row.get('Odds_Reversal_Flag', 0)): parts.append("📉 Odds Reversal")
        if float(row.get('Sharp_Time_Score', 0)) > 0.5: parts.append("⏱️ Timing Edge")

        # Team-level
        if float(row.get('Team_Past_Hit_Rate', 0)) > 0.5: parts.append("⚔️📊 Team Historically Sharp")
        if float(row.get('Team_Past_Avg_Model_Prob', 0)) > 0.5: parts.append("🔮 Model Favored This Team Historically")
        if float(row.get('Avg_Recent_Cover_Streak', 0)) >= 2: parts.append("🔥 Recent Hot Streak")
        if float(row.get('Avg_Recent_Cover_Streak_Home', 0)) >= 2: parts.append("🏠🔥 Home Streaking")
        if float(row.get('Avg_Recent_Cover_Streak_Away', 0)) >= 2: parts.append("✈️🔥 Road Streaking")

        # From open
        if float(row.get('Abs_Line_Move_From_Opening', 0)) > 1.0: parts.append("📈 Line Moved from Open")
        if float(row.get('Abs_Odds_Move_From_Opening', 0)) > 5.0: parts.append("💹 Odds Moved from Open")

        # Cross-market + diagnostics
        if bool(row.get('Spread_vs_H2H_Aligned', 0)): parts.append("🧩 Spread and H2H Align")
        if bool(row.get('Total_vs_Spread_Contradiction', 0)): parts.append("⚠️ Total Contradicts Spread")
        if bool(row.get('CrossMarket_Prob_Gap_Exists', 0)): parts.append("🔀 Cross-Market Probability Gap")
        if bool(row.get('Potential_Overmove_Flag', 0)): parts.append("📊 Line Possibly Overmoved")
        if bool(row.get('Potential_Overmove_Total_Pct_Flag', 0)): parts.append("📉 Total Possibly Overmoved")
        if bool(row.get('Potential_Odds_Overmove_Flag', 0)): parts.append("🎯 Odds Possibly Overmoved")
        if bool(row.get('Line_Moved_Toward_Team', 0)): parts.append("🧲 Line Moved Toward This Team")
        if bool(row.get('Line_Moved_Away_From_Team', 0)): parts.append("🚫 Line Moved Away From This Team")
     
        if float(row.get('Abs_Line_Move_Z', 0)) > 1: parts.append("📊 Unusual Line Z-Move")
        if float(row.get('Pct_Line_Move_Z', 0)) > 1: parts.append("📈 Abnormal % Line Z-Score")
        if bool(row.get('Mispricing_Flag', 0)): parts.append("💸 Market Mispricing Detected")
        if bool(row.get('Hybrid_Line_Timing_Flag', 0)): parts.append("⏱️ Sharp Line Timing Bucket")
        if bool(row.get('Hybrid_Odds_Timing_Flag', 0)): parts.append("🕰️ Sharp Odds Timing Bucket")
        if bool(row.get('Is_Weekend', 0)): parts.append("📅 Weekend Game")
        if bool(row.get('Is_Night_Game', 0)): parts.append("🌙 Night Game")
        if bool(row.get('Is_PrimeTime', 0)): parts.append("⭐ Prime Time Matchup")
        # --- Power ratings & outcome model edges (robust & tolerant) ---
        def _num(row, col):
            v = row.get(col)
            # normalize common formatting issues
            if isinstance(v, str):
                v = v.replace('\u2212','-').replace(',','').strip()
                if v in ('', '—', '–'):
                    return np.nan
            return pd.to_numeric(v, errors='coerce')
        
        
        
        _mod_spread, _mkt_spread = _num(row,'Outcome_Model_Spread'), _num(row,'Outcome_Market_Spread')
        if pd.notna(_mod_spread) and pd.notna(_mkt_spread):
            parts.append(f"📐 Model Spread {_mod_spread:+.1f} vs Market {_mkt_spread:+.1f}")
        
        _edge = _num(row,'Outcome_Spread_Edge')
        if pd.notna(_edge):     # or abs(_edge) >= 0.1 if you want a threshold
            parts.append(f"🎯 Spread Edge {_edge:+.1f}")
        
        _cov = _num(row,'Outcome_Cover_Prob')
        if pd.notna(_cov):
            parts.append(f"🛡️ Cover Prob {_cov:.0%}" + (" ✅" if _cov >= 0.55 else ""))
        
        # robust agree check
        agree_val = row.get('model_fav_vs_market_fav_agree', 0)
        agree_num = pd.to_numeric(agree_val, errors='coerce')
        if (pd.notna(agree_num) and int(round(float(agree_num))) == 1) or \
           (str(agree_val).strip().lower() in ('1','true','yes')):
            parts.append("🤝 Model & Market Favor Same Team")
        
     
        # Hybrid timing buckets → nice human labels
        HYBRID_LINE_COLS_LOCAL = [
            'SharpMove_Magnitude_Overnight_VeryEarly','SharpMove_Magnitude_Overnight_MidRange',
            'SharpMove_Magnitude_Overnight_LateGame','SharpMove_Magnitude_Overnight_Urgent',
            'SharpMove_Magnitude_Early_VeryEarly','SharpMove_Magnitude_Early_MidRange',
            'SharpMove_Magnitude_Early_LateGame','SharpMove_Magnitude_Early_Urgent',
            'SharpMove_Magnitude_Midday_VeryEarly','SharpMove_Magnitude_Midday_MidRange',
            'SharpMove_Magnitude_Midday_LateGame','SharpMove_Magnitude_Midday_Urgent',
            'SharpMove_Magnitude_Late_VeryEarly','SharpMove_Magnitude_Late_MidRange',
            'SharpMove_Magnitude_Late_LateGame','SharpMove_Magnitude_Late_Urgent'
        ]
        HYBRID_ODDS_COLS_LOCAL = [
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

        for col in HYBRID_LINE_COLS_LOCAL:
            if float(row.get(col, 0)) > 0.25:
                bucket = col.replace('SharpMove_Magnitude_', '').replace('_', ' ')
                epoch = col.split('_')[2]  # Overnight/Early/Midday/Late
                parts.append(f"{EMOJI_MAP.get(epoch, '⏱️')} {bucket} Sharp Move")

        for col in HYBRID_ODDS_COLS_LOCAL:
            if float(row.get(col, 0)) > 0.5:
                bucket = col.replace('OddsMove_Magnitude_', '').replace('_', ' ')
                epoch = col.split('_')[2]
                parts.append(f"{EMOJI_MAP.get(epoch, '⏱️')} {bucket} Odds Steam")

        if not parts and not bool(row.get('Passes_Gate', False)):
            return "🕓 Still Calculating Signal"
        return " + ".join(parts) if parts else "🤷‍♂️ Still Calculating"

    # keep this AFTER the aliasing above
    df['Why Model Likes It'] = df.apply(build_why, axis=1)

    # === Model_Confidence_Tier for summary compatibility
    df['Model_Confidence_Tier'] = df['Confidence Tier']

    # === Timing Opportunity Models (unchanged scaffold)
    timing_models = {}
    for m in ['spreads', 'totals', 'h2h']:
        try:
            model_data = load_model_from_gcs(sport=sport, market=f"timing_{m}", bucket_name=GCS_BUCKET)
            timing_models[m] = model_data.get("calibrator") or model_data.get("model")
        except Exception:
            timing_models[m] = None

    timing_feature_cols = [
        'Abs_Line_Move_From_Opening','Abs_Odds_Move_From_Opening','Late_Game_Steam_Flag'
    ] + hybrid_timing_features + hybrid_odds_timing_features

    for col in timing_feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df['Timing_Opportunity_Score'] = np.nan
    for m in ['spreads','totals','h2h']:
        model = timing_models.get(m)
        if model is None:
            continue
        mask = df['Market'].str.lower() == m
        if mask.any():
            X = df.loc[mask, timing_feature_cols].fillna(0)
            # handle both calibrated and raw models
            if hasattr(model, "predict_proba"):
                df.loc[mask, 'Timing_Opportunity_Score'] = model.predict_proba(X)[:, 1]
            else:
                df.loc[mask, 'Timing_Opportunity_Score'] = np.clip(model.predict(X), 0, 1)

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
    # Base diagnostic columns
    base_cols = [
        'Game_Key','Market','Outcome','Bookmaker',
        'Tier_Change','Confidence Trend','Line/Model Direction',
        'Why Model Likes It','Passes_Gate','Confidence Tier',
        'Model Prob Snapshot','First Prob Snapshot',
        'Model_Confidence_Tier',
        'Timing_Opportunity_Score','Timing_Stage'
    ]

    # Only include features that actually exist (we ensured most above)
    feature_cols_present = [c for c in features if c in df.columns]

    diagnostics_df = df[base_cols + feature_cols_present].rename(columns={'Tier_Change': 'Tier Δ'})

    return diagnostics_df



    
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
        FROM `sharplogger.sharp_data.scores_with_features`
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


from math import erf, sqrt



def render_scanner_tab(label, sport_key, container, force_reload=False):
    if st.session_state.get("pause_refresh", False):
        st.info("⏸️ Auto-refresh paused")
        return

    with container:
        st.subheader(f"📡 Scanning {label} Sharp Signals")

        # === 0) History (used later for trends)
        HOURS = 24
        df_all_snapshots = get_recent_history(hours=HOURS, sport=label)

        # === 1) Load/cached sharp moves
        detection_key = f"sharp_moves:{label.upper()}:{HOURS}"
        if not force_reload and detection_key in st.session_state:
            df_moves_raw = st.session_state[detection_key]
            st.info(f"✅ Using cached {label} sharp moves")
        else:
            with st.spinner(f"📥 Loading {label} sharp moves from BigQuery..."):
                df_moves_raw = read_recent_sharp_moves_conditional(
                    force_reload=force_reload, hours=HOURS, sport=label
                )
                st.session_state[detection_key] = df_moves_raw
                st.success(f"✅ Loaded {0 if df_moves_raw is None else len(df_moves_raw)} {label} sharp-move rows from BigQuery")

        # === 2) Base guards BEFORE touching columns
        skip_grading = False
        if df_moves_raw is None or df_moves_raw.empty:
            st.warning(f"⚠️ No recent sharp moves for {label}.")
            skip_grading = True
        else:
            # Safe sport filter
            if 'Sport' in df_moves_raw.columns:
                df_moves_raw['Sport_Norm'] = (
                    df_moves_raw['Sport'].astype(str).str.upper().str.strip()
                )
                df_moves_raw = df_moves_raw[df_moves_raw['Sport_Norm'] == label.upper()]

            # Required columns check
            required_cols = ['Game', 'Game_Start', 'Market', 'Outcome']
            missing = [c for c in required_cols if c not in df_moves_raw.columns]
            if missing:
                st.error(f"❌ Required columns missing before build_game_key: {missing}")
                st.dataframe(df_moves_raw.head())
                skip_grading = True

            # Dedup snapshots (only if we still have data)
            if not skip_grading and 'Snapshot_Timestamp' in df_moves_raw.columns and not df_moves_raw.empty:
                dedup_cols = [c for c in [
                    'Home_Team_Norm','Away_Team_Norm','Game_Start','Market','Outcome',
                    'Bookmaker','Value','Odds_Price','Limit'
                ] if c in df_moves_raw.columns]
                before = len(df_moves_raw)
                df_moves_raw = df_moves_raw.sort_values('Snapshot_Timestamp', ascending=False)
                df_moves_raw = df_moves_raw.drop_duplicates(subset=dedup_cols, keep='first')
                after = len(df_moves_raw)
                st.info(f"🧹 Deduplicated {before - after} snapshot rows (kept latest per unique pick).")

            # Build game keys + future filter
            if not skip_grading:
                df_moves_raw = build_game_key(df_moves_raw)
                df_moves_raw['Game_Start'] = pd.to_datetime(
                    df_moves_raw['Game_Start'], errors='coerce', utc=True
                )
                now = pd.Timestamp.utcnow()
                before = len(df_moves_raw)
                df_moves_raw = df_moves_raw[df_moves_raw['Game_Start'] > now]
                after = len(df_moves_raw)
                # st.info(f"✅ Game_Start > now: filtered {before} → {after} rows")

                if df_moves_raw.empty:
                    st.warning("⚠️ No graded picks available yet. I’ll still show live odds below.")
                    skip_grading = True

        # Keep a pristine copy for "first snapshot" work later (or empty if skipping)
        df_raw_for_history = df_moves_raw.copy() if not skip_grading else pd.DataFrame()

        # === 3) Load per-market models (BEFORE any sharp-scoring)
        market_list = ['spreads', 'totals', 'h2h']
        trained_models = {}
        for market in market_list:
            model_bundle = load_model_from_gcs(label, market)
            if model_bundle:
                trained_models[market] = model_bundle
            else:
                st.warning(f"⚠️ No model found for {market.upper()} — skipping.")

        if not trained_models:
            st.warning(f"⚠️ No models available for {label}. Live odds are shown below.")

        # Optional: unified team_feature_map
        team_feature_map = None
        if trained_models:
            for bundle in trained_models.values():
                tfm = bundle.get('team_feature_map')
                if tfm is not None and not tfm.empty:
                    team_feature_map = tfm
                    break

        # === 4) One gate you can use below to run the sharp section
        can_render_sharp = bool(trained_models) and not skip_grading

        if can_render_sharp:
  
            # === Load broader trend history for open line / tier comparison
            start = time.time()
           # Keep all rows for proper line open capture
            df_history_all =  df_raw_for_history
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
            #st.write("📋 Columns in df_pre:", df_pre.columns.tolist())
            
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
            #st.write("📋 Columns in df_summary_base_mid:", df_summary_base.columns.tolist())
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
         
    
    
            #st.write("📋 Columns in df_summary_base_end:", df_summary_base.columns.tolist())
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
                # Optionally build a ratings_map (if you have a simple table of team → rating)
                # ratings_map = pd.DataFrame({'Team': [...], 'PRating': [...]})
  
                diagnostics_df = compute_diagnostics_vectorized(diag_source)  # per-book diagnostics
            
                rep = df_summary_base[['Game_Key','Market','Outcome','Bookmaker']].drop_duplicates()
                diagnostics_pick = diagnostics_df.merge(
                    rep, on=['Game_Key','Market','Outcome','Bookmaker'], how='inner'
                )[[
                    'Game_Key','Market','Outcome','Bookmaker',
                    'Tier Δ','Line/Model Direction','Why Model Likes It',
                    'Timing_Opportunity_Score','Timing_Stage',  # <-- add these
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
            pass
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

        FROM `sharplogger.sharp_data.scores_with_features`
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
                FROM `sharplogger.sharp_data.scores_with_features`
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
        
        
