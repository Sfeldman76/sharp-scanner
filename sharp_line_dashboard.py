
import streamlit as st
import time  # keep only if you use it elsewhere

# === Page Config ===
st.set_page_config(layout="wide")

# =========================
# Session-state initialization
# =========================
if "is_training" not in st.session_state:
    st.session_state["is_training"] = False
if "pause_refresh_lock" not in st.session_state:
    st.session_state["pause_refresh_lock"] = False

# ✅ No auto-refresh. No component import. No warnings.


st.title("Betting Line Scanner")
# --- Custom CSS for scrollable DataFrames ---
st.markdown(
    """
    <style>
    .scrollable-dataframe-container {
        max-height: 600px;
        overflow-y: auto;
        overflow-x: auto;
        border: 1px solid #444;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Force DataFrame elements to full width */
    div[data-testid="stDataFrame"] > div {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



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
import re
import time
import json
import pickle
import logging
import traceback
import contextlib
from io import StringIO, BytesIO
from datetime import datetime, date, timedelta
from collections import defaultdict, OrderedDict
from itertools import product
from html import escape
import io, pickle, logging
import traceback
from google.cloud import storage
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype
import re, hashlib  # ← for safe DOM keys
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import requests
import pytz
from pytz import timezone as pytz_timezone
from sklearn.feature_selection import VarianceThreshold
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from sklearn.base import clone
import os
os.environ.setdefault("OMP_NUM_THREADS",  "1")
os.environ.setdefault("MKL_NUM_THREADS",  "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
from sklearn.calibration import calibration_curve  # (kept import local to avoid global clutter)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from scipy.stats import zscore, entropy, randint, loguniform, uniform
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    TimeSeriesSplit, HalvingRandomSearchCV, BaseCrossValidator
)
from sklearn.model_selection import RandomizedSearchCV as _SearchCV
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, make_scorer
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import xgboost as xgb
from xgboost import XGBClassifier
import shap, numpy as np, pandas as pd
from google.oauth2 import service_account
from google.cloud import storage, bigquery, bigquery_storage_v1
import pandas_gbq
from pandas_gbq import to_gbq
import google.api_core.exceptions
from google.cloud import bigquery, bigquery_storage
from sklearn.inspection import PartialDependenceDisplay
from pandas.api.types import (
    is_bool_dtype, is_numeric_dtype, is_categorical_dtype, is_datetime64_any_dtype,
    is_period_dtype, is_interval_dtype, is_object_dtype
)
from copy import deepcopy

from sklearn.model_selection import RandomizedSearchCV

from sklearn.base import clone
             
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
import gc
from google.cloud import bigquery
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date, timedelta 
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.special import ndtr
from math import erf, sqrt
from xgboost import XGBClassifier
# put near your imports (only once)
from sklearn.base import is_classifier as sk_is_classifier
import sys, inspect, xgboost, sklearn
from pandas.api.types import is_numeric_dtype
import numpy as np
from xgboost import XGBClassifier
from dataclasses import dataclass, asdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import log_loss, make_scorer
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np, pandas as pd
from math import erf as _erf
import html, unicodedata, re
import shap
# ---- HTML sanitizers (define once, top-level) ----
_ALLOW_HTML_COLS = {"Confidence Spark"}  # whitelist columns that intentionally contain HTML

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

class StreamlitLogger:
    def _fmt(self, msg, args):
        return msg % args if args else msg

    def info(self, msg, *args, **kwargs):
        st.info(self._fmt(msg, args))

    def warning(self, msg, *args, **kwargs):
        st.warning(self._fmt(msg, args))

    def error(self, msg, *args, **kwargs):
        st.error(self._fmt(msg, args))

logger = StreamlitLogger()

def _clean_text(x: str) -> str:
    # Normalize, strip control chars, then escape HTML
    x = unicodedata.normalize("NFC", x)
    x = _CONTROL_CHARS.sub("", x)
    return html.escape(x, quote=True)

def _clean_cell(val, colname: str):
    if val is None or (hasattr(val, "__class__") and str(val) == "nan"):
        return ""
    s = str(val)
    if colname in _ALLOW_HTML_COLS:
        # Keep HTML but normalize & remove control chars
        s = unicodedata.normalize("NFC", s)
        s = _CONTROL_CHARS.sub("", s)
        return s
    return _clean_text(s)


from sklearn.feature_selection import VarianceThreshold
pandas_gbq.context.project = GCP_PROJECT_ID  # credentials will be inferred

bq_client = bigquery.Client(project=GCP_PROJECT_ID)  # uses env var
gcs_client = storage.Client(project=GCP_PROJECT_ID)
import os, random
GLOBAL_SEED = 1337
os.environ["PYTHONHASHSEED"] = "0"
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


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
    'lowvig', 'betonlineag', 'matchbook', 'sport888'
]

# Recreational set (deduped)
REC_BOOKS = [
    'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers',
    'fanatics', 'espnbet', 'hardrockbet', 'williamhillus', 'ballybet', 'bet365_au', 'betopenly'
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
    "sport888":     {"sports888"},
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

import os

@st.cache_resource
def get_clients():
    from google.cloud import bigquery, storage
    return bigquery.Client(), storage.Client()

bq, gcs = get_clients()


def get_vcpus(env_var: str = "XGB_SEARCH_NJOBS") -> int:
    """Prefer ENV override; else fall back to container vCPU count."""
    val = os.getenv(env_var, "")
    try:
        n = int(val) if val != "" else (os.cpu_count() or 1)
    except Exception:
        n = os.cpu_count() or 1
    return max(1, n)

VCPUS = get_vcpus()  # e.g., 8 if XGB_SEARCH_NJOBS=8, else os.cpu_count()

# put this with your imports
try:
    from threadpoolctl import threadpool_limits
except Exception:
    # fallback: no-op context manager so your code still runs
    from contextlib import contextmanager
    @contextmanager
    def threadpool_limits(**kwargs):
        yield

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



def safe_row_entropy(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise Shannon entropy for nonnegative weights matrix W (n_rows x n_cols).
    - Handles rows with zero total mass safely (entropy -> 0).
    - Avoids NaN/inf and broadcasting issues.
    """
    W = np.asarray(W, dtype=np.float32)
    # sanitize inputs
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    W = np.maximum(W, 0.0)

    row_sum = W.sum(axis=1, keepdims=True)             # shape (n, 1)
    # probability rows; where sum==0 produce zeros
    P = np.divide(W, np.maximum(row_sum, eps), out=np.zeros_like(W), where=row_sum > 0)

    # entropy per row: -sum(p*log p)
    # compute only where p>0 to avoid log(0)
    ent = -(np.where(P > 0, P * np.log(P), 0.0)).sum(axis=1).astype(np.float32)
    return ent

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
    "NCAAB":   ["NCAAB", "BASKETBALL_NCAAB", "BASKETBALL-NCAAB"],
    # extend as needed
}


sport_aliases = SPORT_ALIASES
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
    if p >= 0.70: return "🔥 Steam"
    if p >= 0.55: return "🔥 Strong Indication"
    if p >= 0.51: return "⭐ Lean"
    return "✅ Low"

# --- helpers (put these near your other utils) ---
class _IdentityIsoCal:
    def __init__(self, eps=1e-6):
        self.eps = eps
    def transform(self, p):
        p = np.asarray(p, float).reshape(-1)
        return np.clip(p, self.eps, 1.0 - self.eps)

class _IdentityProbCal:
    def __init__(self, eps=1e-6):
        self.eps = eps
    def predict_proba(self, X):
        p = np.asarray(X, float).reshape(-1)
        p = np.clip(p, self.eps, 1.0 - self.eps)
        return np.c_[1.0 - p, p]

def _ensure_transform_for_iso(iso_obj):
    if iso_obj is None:
        return None
    if hasattr(iso_obj, "transform"):
        return iso_obj
    if hasattr(iso_obj, "predict"):
        class _IsoAsTransform:
            def __init__(self, iso): self.iso = iso
            def transform(self, p):  return self.iso.predict(p)
        return _IsoAsTransform(iso_obj)
    return iso_obj

def _ensure_predict_proba_for_prob_cal(cal_obj, eps=1e-6):
    if cal_obj is None or hasattr(cal_obj, "predict_proba"):
        return cal_obj
    class _AsPredictProba:
        def __init__(self, base, eps=1e-6): self.base, self.eps = base, eps
        def predict_proba(self, X):
            p = np.asarray(X, float).reshape(-1)
            if hasattr(self.base, "__call__"):
                p = np.asarray(self.base(p), float).reshape(-1)
            p = np.clip(p, self.eps, 1.0 - self.eps)
            return np.c_[1.0 - p, p]
    return _AsPredictProba(cal_obj, eps=eps)

def _normalize_cals(cals_tuple_or_dict):
    if isinstance(cals_tuple_or_dict, tuple) and len(cals_tuple_or_dict) == 2:
        kind, model = cals_tuple_or_dict
        return {
            "iso":   model if kind == "iso"   else None,
            "platt": model if kind == "platt" else None,
            "beta":  model if kind == "beta"  else None,
        }
    return cals_tuple_or_dict

class _IdentityCal:
    def __init__(self, eps=1e-6):
        self.eps = eps
    # used when select_blend calls .transform for iso
    def transform(self, p):
        p = np.asarray(p, float).reshape(-1)
        return np.clip(p, self.eps, 1.0 - self.eps)
    # used if select_blend ever calls predict_proba for platt/beta
    def predict_proba(self, X):
        p = np.asarray(X, float).reshape(-1)
        p = np.clip(p, self.eps, 1.0 - self.eps)
        return np.c_[1.0 - p, p]






class _BetaCalibrator:
    """Logistic on [logit(p), logit(p)^2]; exposes .predict(probs)->calibrated_probs."""
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
    @staticmethod
    def _logit(x, eps):
        x = np.clip(np.asarray(x, float).reshape(-1), eps, 1.0 - eps)
        return np.log(x / (1.0 - x))
    def fit(self, p, y):
        z = self._logit(p, self.eps)
        X = np.c_[z, z*z]
        self.model.fit(X, np.asarray(y, int))
        return self
    def predict(self, p):
        z = self._logit(p, self.eps)
        X = np.c_[z, z*z]
        return self.model.predict_proba(X)[:, 1]
import numpy as np
import math
from sklearn.metrics import roc_auc_score

def pos_proba_safe(clf, X, positive=1):
    """Return P(y==positive) using the correct column from predict_proba."""
    proba = clf.predict_proba(X)
    classes = np.asarray(getattr(clf, "classes_", []))
    if classes.size == 0:
        raise RuntimeError("Classifier has no classes_.")
    where = np.where(classes == positive)[0]
    if where.size == 0:
        raise RuntimeError(f"Positive label {positive} not in classes_: {classes!r}")
    return proba[:, int(where[0])].astype(np.float64), classes

def auc_with_flip(y, p, w=None):
    """Compute AUC and flip scores if AUC<0.5 (returns auc, possibly flipped p, flipped?)."""
    a = roc_auc_score(y, p, sample_weight=w)
    if a < 0.5:
        return 1.0 - a, 1.0 - p, True
    return a, p, False

def best_round_plus_one(clf) -> int:
    """Robust number-of-trees getter after early stopping."""
    bi = getattr(clf, "best_iteration", None)
    if isinstance(bi, (int, np.integer)) and bi >= 0: return int(bi + 1)
    if isinstance(bi, float) and not math.isnan(bi) and bi >= 0: return int(bi + 1)
    try:
        bst = clf.get_booster()
        bi2 = getattr(bst, "best_iteration", None)
        if isinstance(bi2, (int, np.integer)) and bi2 >= 0: return int(bi2 + 1)
        bntl = getattr(bst, "best_ntree_limit", None)
        if isinstance(bntl, (int, np.integer)) and bntl > 0: return int(bntl)
    except Exception:
        pass
    ev = getattr(clf, "evals_result_", {}) or {}
    ds = next((k for k in ("validation_0","eval","valid_0") if k in ev), None)
    if ds:
        metrics = ev[ds]
        if metrics:
            arr = next(iter(metrics.values()))
            if isinstance(arr, list) and len(arr) > 0:
                return int(len(arr))
    return int(getattr(clf, "n_estimators", 0))

def weight_report(w, y):
    
    df = pd.DataFrame({"w": w, "y": y})
    return df.groupby("y")["w"].agg(["count","sum","mean","min","max"])

def fit_robust_calibrator(p_oof, y_oof, *, eps=1e-6, min_unique=200, prefer_beta=True):
    """
    Returns (kind, model) where kind ∈ {'iso','platt','beta'} compatible with your _CalAdapter.
    - If enough unique probabilities & both classes → IsotonicRegression
    - Else → Beta calibrator (if prefer_beta) or Platt (logistic) as fallback
    """
    p = np.clip(np.asarray(p_oof, float).reshape(-1), eps, 1.0 - eps)
    y = np.asarray(y_oof, int).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(y) & ((y == 0) | (y == 1))
    p, y = p[mask], y[mask]

    # Need both classes
    if np.unique(y).size < 2 or p.size < 2:
        return ("iso", _IdentityIsoCal(eps=eps))

    # Prefer isotonic when we have enough support
    if np.unique(p).size >= int(min_unique):
        iso = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        # sklearn's iso uses .predict; your code calls .transform → wrap isn’t needed if upstream handled it
        # but return raw; you already have an ensure-transform shim, otherwise add:
        return ("iso", iso)

    # Otherwise, parametric fallback
    if prefer_beta:
        beta = _BetaCalibrator(eps=eps).fit(p, y)
        return ("beta", beta)
    else:
        platt = LogisticRegression(solver="lbfgs", max_iter=1000)
        platt.fit(p.reshape(-1, 1), y)
        return ("platt", platt)

def pos_col_index(est, positive=1):
    cls = getattr(est, "classes_", None)
    if cls is None:
        raise RuntimeError("Estimator has no classes_. Was it fitted?")
    hits = np.where(cls == positive)[0]
    if len(hits) == 0:
        raise RuntimeError(f"Positive class {positive!r} not in classes_={cls}.")
    return int(hits[0])

def pos_proba(est, X, positive=1):
    return est.predict_proba(X)[:, pos_col_index(est, positive=positive)]

def _auc_safe(y, p):
    # returns (auc, has_both_classes)
    y = np.asarray(y, int); p = np.asarray(p, float)
    ok = (np.unique(y).size == 2) and np.isfinite(p).all() and (p.min() >= 0) and (p.max() <= 1)
    if not ok: return (np.nan, False)
    return (roc_auc_score(y, p), True)
class _CalAdapter:
    """
    Wrap a calibrator tuple ('iso'|'beta'|'platt', model) so you can call .predict(p)
    and get clipped probabilities. Compatible with your fit_robust_calibrator output.
    """
    def __init__(self, cal_tuple, clip=(0.001, 0.999)):
        if cal_tuple is None or len(cal_tuple) != 2:
            raise ValueError("cal_tuple must be ('iso'|'beta'|'platt', model)")
        self.kind, self.model = cal_tuple
        self.clip = clip

    def predict(self, p):
        
        p = np.asarray(p, float)
        if self.kind == "iso":
            # your helpers already normalize iso to have .transform if needed
            out = self.model.transform(p) if hasattr(self.model, "transform") else self.model.predict(p)
        elif self.kind == "beta":
            # your _BetaCalibrator exposes .predict(p)
            out = self.model.predict(p)
        else:  # 'platt'
            out = self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        lo, hi = self.clip
        return np.clip(out, lo, hi)

def pick_blend_weight_on_oof(
    y_oof,
    p_oof_auc,
    p_oof_log=None,           # pass None to do AUC-only
    grid=None,
    eps=1e-4,
    metric="logloss",         # or "hybrid"
    hybrid_alpha=0.9,
):
    y = np.asarray(y_oof, int)

    # AUC-only path
    if p_oof_log is None:
        b = np.asarray(p_oof_auc, float)
        b = np.clip(b, eps, 1-eps)
        valid = np.isfinite(b) & np.isfinite(y)
        yv, bv = y[valid], b[valid]
        # polarity lock
        auc_raw, ok = _auc_safe(yv, bv)
        auc_flip, _ = _auc_safe(yv, 1.0 - bv)
        flipped = ok and (np.nan_to_num(auc_flip) > np.nan_to_num(auc_raw))
        if flipped:
            bv = 1.0 - bv
        return 0.0, bv, flipped

    # Two-model path
    a = np.asarray(p_oof_log, float)
    b = np.asarray(p_oof_auc, float)
    a = np.clip(a, eps, 1-eps)
    b = np.clip(b, eps, 1-eps)
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    y, a, b = y[valid], a[valid], b[valid]

    # polarity lock on mean stream
    mix = 0.5 * (a + b)
    auc_raw, ok = _auc_safe(y, mix)
    auc_flip, _ = _auc_safe(y, 1.0 - mix)
    flipped = ok and (np.nan_to_num(auc_flip) > np.nan_to_num(auc_raw))
    if flipped:
        a, b = 1.0 - a, 1.0 - b

    if grid is None:
        grid = np.round(np.linspace(0.20, 0.80, 13), 2)

    best_w, best_score = None, +1e18
    for w in grid:
        p = np.clip(w*a + (1-w)*b, eps, 1-eps)
        if metric == "logloss":
            score = log_loss(y, p, labels=[0, 1])
        else:
            auc, _ = _auc_safe(y, p)
            score = hybrid_alpha*log_loss(y, p, labels=[0,1]) + (1-hybrid_alpha)*(1.0 - np.nan_to_num(auc))
        if score < best_score:
            best_score, best_w = score, float(w)

    p_oof_blend = np.clip(best_w*a + (1-best_w)*b, eps, 1-eps)
    return best_w, p_oof_blend, flipped


# --- Calibrator normalization & safety shims (put right before select_blend) ---
def _ensure_transform_for_iso(iso_obj):
    """Wrap sklearn IsotonicRegression (predict-only) to expose .transform."""
    if iso_obj is None:
        return None
    if hasattr(iso_obj, "transform"):
        return iso_obj
    if hasattr(iso_obj, "predict"):
        class _IsoAsTransform:
            def __init__(self, iso): self.iso = iso
            def transform(self, p): return self.iso.predict(p)
        return _IsoAsTransform(iso_obj)
    return iso_obj  # unknown type, best effort

def _ensure_predict_proba_for_prob_cal(cal_obj, eps=1e-6):
    """
    Ensure platt/beta calibrators expose .predict_proba(*).
    If cal_obj already has predict_proba → return as-is.
    If it's a bare function/obj returning probs, wrap it.
    """
    if cal_obj is None:
        return None
    if hasattr(cal_obj, "predict_proba"):
        return cal_obj
    # Fallback: treat input as raw probabilities and build a 2-col proba
    class _AsPredictProba:
        def __init__(self, base, eps=1e-6):
            self.base = base; self.eps = eps
        def predict_proba(self, X):
            p = np.asarray(X, float).reshape(-1)
            if hasattr(self.base, "__call__"):
                p = np.asarray(self.base(p), float).reshape(-1)
            p = np.clip(p, self.eps, 1.0 - self.eps)
            return np.c_[1.0 - p, p]
    return _AsPredictProba(cal_obj, eps=eps)



from scipy.special import ndtr as _phi      # Φ
from scipy.special import ndtri as _ppf     # Φ^{-1}

def _col_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present; else an all-NaN numeric Series aligned to df.index."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")

def _ensure_series(x, index) -> pd.Series:
    """Broadcast scalars/arrays to a Series aligned to index."""
    if isinstance(x, pd.Series):
        return x
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = np.repeat(arr, len(index))
    elif arr.shape[0] != len(index):
        arr = np.resize(arr, len(index))
    return pd.Series(arr, index=index)

def compute_ev_features_sharp_vs_rec(
    df_market: pd.DataFrame,
    *,
    sharp_books: list[str] | None = None,
    reliability_col: str = "Book_Reliability_Score",
    limit_col: str = "Sharp_Limit_Total",
    sigma_col: str = "Sigma_Pts",
) -> pd.DataFrame:
    """
    Compare a sharp 'truth' vs each book's offer (line + odds) and add EV features.
    Always returns same row count. Safe when sharp data or sigma are missing.
    """
    def _amer_to_prob(odds):
        o = pd.to_numeric(odds, errors="coerce")
        return np.where(o >= 0, 100.0/(o+100.0), (-o)/((-o)+100.0))

    out_cols = [
        "Truth_Fair_Prob_at_SharpLine","Truth_Margin_Mu","Truth_Sigma",
        "Truth_Fair_Prob_at_RecLine","Rec_Implied_Prob",
        "EV_Sh_vs_Rec_Prob","EV_Sh_vs_Rec_Dollar","Kelly_Fraction"
    ]

    if df_market is None or df_market.empty:
        dm = (df_market.copy() if df_market is not None else pd.DataFrame())
        for c in out_cols: dm[c] = np.nan
        return dm

    dm = df_market.copy()
    dm["Market"] = dm["Market"].astype(str).str.lower().str.strip()
    dm["Outcome_Norm"] = dm.get("Outcome_Norm", dm.get("Outcome", "")).astype(str).str.lower().str.strip()
    dm["Bookmaker"] = dm["Bookmaker"].astype(str).str.lower().str.strip()
    for c in out_cols:
        if c not in dm.columns:
            dm[c] = np.nan

    SHARP_SET = set(sharp_books or SHARP_BOOKS)
    sharp_mask = dm["Bookmaker"].isin(SHARP_SET)

    # No sharp rows → still compute Rec_Implied_Prob and return
    if not sharp_mask.any():
        dm["Rec_Implied_Prob"] = _amer_to_prob(_col_or_nan(dm, "Odds_Price")).astype("float32")
        return dm

    keep = ["Game_Key","Market","Outcome_Norm","Bookmaker","Value","Odds_Price"]
    if reliability_col in dm.columns: keep.append(reliability_col)
    if limit_col in dm.columns:       keep.append(limit_col)
    sharp_rows = dm.loc[sharp_mask, keep].copy()

    if sharp_rows.empty:
        dm["Rec_Implied_Prob"] = _amer_to_prob(_col_or_nan(dm, "Odds_Price")).astype("float32")
        return dm

    # Rank sharp by reliability then limit
    if reliability_col not in sharp_rows.columns:
        sharp_rows[reliability_col] = 0.0
    sharp_rows["_rel_rank"] = sharp_rows.groupby(["Game_Key","Market","Outcome_Norm"])[reliability_col] \
        .rank(ascending=False, method="first")
    if limit_col in sharp_rows.columns:
        sharp_rows["_lim_rank"] = sharp_rows.groupby(["Game_Key","Market","Outcome_Norm"])[limit_col] \
            .rank(ascending=False, method="first")
    else:
        sharp_rows["_lim_rank"] = 1.0

    sharp_ref = (sharp_rows
        .sort_values(["Game_Key","Market","Outcome_Norm","_rel_rank","_lim_rank"])
        .groupby(["Game_Key","Market","Outcome_Norm"], as_index=False)
        .head(1)
        .drop(columns=["_rel_rank","_lim_rank"])
        .rename(columns={"Value":"Sharp_Line","Odds_Price":"Sharp_Odds"})
    )

    # Take top-2 sharp outcomes per (Game_Key, Market) → de‑vig pair
    pairs_base = (sharp_rows
        .sort_values(["Game_Key","Market", reliability_col], ascending=[True,True,False])
        .drop_duplicates(subset=["Game_Key","Market","Outcome_Norm"], keep="first"))

    top2 = (pairs_base.groupby(["Game_Key","Market"], as_index=False)
        .apply(lambda g: g.head(2))
        .reset_index(drop=True))

    def _to_pair(df):
        if len(df) < 2:
            return pd.Series({"Outcome_A":np.nan,"Line_A":np.nan,"Odds_A":np.nan,
                              "Outcome_B":np.nan,"Line_B":np.nan,"Odds_B":np.nan})
        a, b = df.iloc[0], df.iloc[1]
        return pd.Series({"Outcome_A":a["Outcome_Norm"], "Line_A":a["Value"], "Odds_A":a["Odds_Price"],
                          "Outcome_B":b["Outcome_Norm"], "Line_B":b["Value"], "Odds_B":b["Odds_Price"]})
    sharp_pairs = top2.groupby(["Game_Key","Market"]).apply(_to_pair).reset_index()

    truth = sharp_ref.merge(sharp_pairs, on=["Game_Key","Market"], how="left")

    # De‑vig at the sharp line (if pair available)
    same_as_A = truth["Outcome_Norm"].astype(str).eq(truth["Outcome_A"].astype(str))
    odds_this = np.where(same_as_A, truth["Odds_A"], truth["Odds_B"])
    odds_opp  = np.where(same_as_A, truth["Odds_B"], truth["Odds_A"])
    p_this_raw = _amer_to_prob(odds_this)
    p_opp_raw  = _amer_to_prob(odds_opp)
    s = p_this_raw + p_opp_raw
    good = s > 0
    p_fair_this = np.where(good, p_this_raw / s, np.nan)
    truth["Truth_Fair_Prob_at_SharpLine"] = p_fair_this

    # Sigma: prefer row sigma; else sport defaults
    if sigma_col not in dm.columns:
        dm[sigma_col] = np.nan
    sig_map = dm[["Game_Key","Market","Outcome_Norm",sigma_col,"Sport"]].drop_duplicates(["Game_Key","Market","Outcome_Norm"])
    truth = truth.merge(sig_map, on=["Game_Key","Market","Outcome_Norm"], how="left")

    SPORT_SIGMA_DEFAULT = {'NFL':13.0,'NCAAF':14.0,'NBA':12.0,'WNBA':11.0,'MLB':5.5,'CFL':13.5,'NCAAB':11.5}
    def _pick_sigma(row):
        s = pd.to_numeric(row.get(sigma_col), errors="coerce")
        if pd.notna(s) and s > 0: return float(s)
        return float(SPORT_SIGMA_DEFAULT.get(str(row.get("Sport","NFL")).upper(), 12.0))

    s_sharp = pd.to_numeric(truth["Sharp_Line"], errors="coerce")
    p_sharp = pd.to_numeric(truth["Truth_Fair_Prob_at_SharpLine"], errors="coerce")
    sigma   = truth.apply(_pick_sigma, axis=1).astype(float)

    # Solve μ only where both pieces exist
    mask_mu = p_sharp.notna() & s_sharp.notna()
    mu = np.full(len(truth), np.nan, dtype=float)
    mu[mask_mu.values] = (s_sharp[mask_mu] + sigma[mask_mu] * _ppf(p_sharp[mask_mu])).astype(float)

    truth["Truth_Margin_Mu"] = mu
    truth["Truth_Sigma"]     = sigma

    # Merge μ,σ back to every book row (merge can drop pre-created cols)
    keys = ["Game_Key","Market","Outcome_Norm"]
    truth_cols = ["Truth_Fair_Prob_at_SharpLine", "Truth_Margin_Mu", "Truth_Sigma"]
    dm = merge_drop_overlap(
        dm,
        truth[keys + truth_cols],
        on=keys,
        how="left",
        keep_right=True,
    )

    # --- Price each row at its own line (always Series, aligned to dm.index) ---
    mu_series  = _ensure_series(_col_or_nan(dm, "Truth_Margin_Mu"), dm.index)
    sig_series = _ensure_series(_col_or_nan(dm, "Truth_Sigma"), dm.index)
    line_rec   = _ensure_series(_col_or_nan(dm, "Value"), dm.index)

    ok = mu_series.notna() & sig_series.notna() & (sig_series > 0) & line_rec.notna()
    dm.loc[ok, "Truth_Fair_Prob_at_RecLine"] = _phi((mu_series[ok] - line_rec[ok]) / sig_series[ok])

    # --- Moneyline fallback (no line shift) ---
    is_ml   = dm["Market"].isin(["h2h","ml","moneyline","headtohead"])
    need_ml = is_ml & dm["Truth_Fair_Prob_at_RecLine"].isna()

    sharp_base = _ensure_series(_col_or_nan(dm, "Truth_Fair_Prob_at_SharpLine"), dm.index)
    dm.loc[need_ml, "Truth_Fair_Prob_at_RecLine"] = sharp_base[need_ml].values

    # Implied prob of offered odds
    dm["Rec_Implied_Prob"] = _amer_to_prob(_col_or_nan(dm, "Odds_Price"))

    # EV per $1 stake & probability edge
    p_truth = pd.to_numeric(dm["Truth_Fair_Prob_at_RecLine"], errors="coerce")
    odds = pd.to_numeric(_col_or_nan(dm, "Odds_Price"), errors="coerce")
    payout = np.where(odds >= 0, odds/100.0, 100.0/(-odds))
    ok_ev = p_truth.notna() & np.isfinite(payout)
    dm.loc[ok_ev, "EV_Sh_vs_Rec_Dollar"] = (p_truth[ok_ev] * payout[ok_ev]) - (1.0 - p_truth[ok_ev])
    dm.loc[ok_ev, "EV_Sh_vs_Rec_Prob"]   = p_truth[ok_ev] - dm.loc[ok_ev, "Rec_Implied_Prob"]
    dm.loc[ok_ev, "Kelly_Fraction"]      = np.maximum(0.0, (p_truth[ok_ev] * payout[ok_ev] - (1.0 - p_truth[ok_ev])) / payout[ok_ev])

    # --- Final: ensure outputs exist and are numeric (column-safe)
    for c in out_cols:
        if c not in dm.columns:
            dm[c] = np.nan
        dm[c] = pd.to_numeric(dm[c], errors="coerce").astype("float32")

    return dm


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


class IsoWrapper:
    def __init__(self, base, iso):
        self.base = base              # fitted base model (has predict_proba)
        self.iso  = iso               # fitted IsotonicRegression

    def predict_proba(self, X):
        # base probs → calibrate with isotonic → return 2-col proba
        p = self.base.predict_proba(X)
        p = p[:, 1] if isinstance(p, np.ndarray) and p.ndim > 1 else np.asarray(p, float).ravel()
        p_cal = self.iso.predict(p)   # <- use predict(), not transform()
        p_cal = np.clip(np.asarray(p_cal, float).ravel(), 1e-6, 1-1e-6)
        return np.column_stack([1.0 - p_cal, p_cal])


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
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

def add_book_path_reliability_features(
    df: pd.DataFrame,
    closers: pd.DataFrame,
    *,
    eps_open: float = 0.5,
    prior_strength_open: float = 200.0,
    prior_strength_speed: float = 200.0,
    min_trials_for_stats: int = 5,
    early_tiers: set[str] | None = None,
    assume_normalized: bool = True,     # skip repeated string normalization
    canon_mask=None,                    # compute stats only on canonical rows (big win for spreads)
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()

    if early_tiers is None:
        early_tiers = {
            "Overnight_VeryEarly",
            "Overnight_MidRange",
            "Early_VeryEarly",
            "Early_MidRange",
            "Midday_VeryEarly",
        }

    # Optional: compute stats only on canonical rows
    base = df if canon_mask is None else df.loc[canon_mask]

    # ---- thin slice ----
    cols_needed = [
        "Sport", "Market", "Bookmaker", "Game_Key",
        "Snapshot_Timestamp", "Value", "Odds_Price",
    ]
    tier_col = "SharpMove_Timing_Dominant" if "SharpMove_Timing_Dominant" in base.columns else None
    if tier_col is not None:
        cols_needed.append(tier_col)
    if "Game_Start" in base.columns:
        cols_needed.append("Game_Start")

    out = base.loc[:, [c for c in cols_needed if c in base.columns]]

    # ---- normalize keys (ONLY if not already normalized upstream) ----
    if not assume_normalized:
        out = out.copy()
        out["Game_Key"]  = out["Game_Key"].astype(str).str.lower().str.strip()
        out["Bookmaker"] = out["Bookmaker"].astype(str).str.lower().str.strip()
        out["Market"]    = out["Market"].astype(str).str.lower().str.strip()
        if "Sport" in out.columns:
            out["Sport"] = out["Sport"].astype(str).str.upper().str.strip()

    # ---- closers lookup: .map (avoid merge) ----
    clos = closers.loc[:, ["Game_Key", "close_spread", "close_total", "p_ml_fav"]]
    if not assume_normalized:
        clos = clos.copy()
        clos["Game_Key"] = clos["Game_Key"].astype(str).str.lower().str.strip()

    # Ensure unique Game_Key → faster + avoids weirdness
    clos_ix = clos.drop_duplicates("Game_Key").set_index("Game_Key")

    gk = out["Game_Key"]
    close_spread = gk.map(clos_ix["close_spread"]) if "close_spread" in clos_ix.columns else np.nan
    close_total  = gk.map(clos_ix["close_total"])  if "close_total"  in clos_ix.columns else np.nan
    p_ml_fav     = gk.map(clos_ix["p_ml_fav"])     if "p_ml_fav"     in clos_ix.columns else np.nan

    # ---- per-row errors (vectorized numpy) ----
    mkt = out["Market"].astype(str) if getattr(out["Market"].dtype, "name", "") == "category" else out["Market"]
    is_spread = (mkt == "spreads").to_numpy()
    is_total  = (mkt == "totals").to_numpy()
    is_ml     = (mkt == "h2h").to_numpy()

    val = pd.to_numeric(out["Value"], errors="coerce").to_numpy()

    close_line = np.where(is_spread, close_spread.to_numpy(),
                 np.where(is_total,  close_total.to_numpy(), np.nan))
    path_line_err = np.abs(val - close_line)

    x = pd.to_numeric(out["Odds_Price"], errors="coerce").to_numpy()
    imp_prob = np.where(x >= 0, 100.0 / (x + 100.0), (-x) / ((-x) + 100.0))
    # keep for completeness; not used in downstream here, but cheap
    _ = np.where(is_ml, np.abs(imp_prob - p_ml_fav.to_numpy()), np.nan)

    # ---- timestamps (tz-safe; avoids numpy dtype crash) ----
    ts = out["Snapshot_Timestamp"]
    if not (is_datetime64_any_dtype(ts) or is_datetime64tz_dtype(ts)):
        ts = pd.to_datetime(ts, errors="coerce", utc=True)
    else:
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")

    if "Game_Start" in out.columns:
        gs = out["Game_Start"]
        if not (is_datetime64_any_dtype(gs) or is_datetime64tz_dtype(gs)):
            gs = pd.to_datetime(gs, errors="coerce", utc=True)
        else:
            if getattr(gs.dt, "tz", None) is None:
                gs = gs.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    else:
        gs = ts

    # ---- opener rows WITHOUT sorting the full frame ----
    keys = ["Sport", "Market", "Bookmaker", "Game_Key"]

    # idx of earliest snapshot per (Sport,Market,Bookmaker,Game_Key)
    grp = [out[k] for k in keys]
    idx_open = (
        pd.Series(ts.to_numpy(), index=out.index)
          .groupby(grp, sort=False, observed=True)
          .idxmin()
    )
    # drop groups with all-NaT timestamps (rare but possible)
    idx_open = idx_open.dropna()

    # opener line error and open_good
    open_err = pd.Series(path_line_err, index=out.index).reindex(idx_open.to_numpy())
    open_good = (open_err.notna() & (open_err.to_numpy() <= eps_open)).astype("int8")

    # fast “good” (any early-tier snapshot in band)
    if tier_col is not None:
        tier = out[tier_col]
        band_early = pd.Series(
            (path_line_err <= eps_open) & tier.isin(early_tiers).to_numpy(),
            index=out.index,
        )

        grouped_any = (
            band_early.groupby(grp, sort=False, observed=True)
                      .any()
                      .astype("int8")
        )

        # ✅ align WITHOUT MultiIndex reindex() issues:
        # use the exact group index object from idx_open (same grouping)
        fast_good = grouped_any.loc[idx_open.index].fillna(0).astype("int8")
        fast_good_np = fast_good.to_numpy()
    else:
        fast_good_np = np.zeros(len(idx_open), dtype=np.int8)

    # game start at opener idx
    game_start_open = pd.Series(gs.to_numpy(), index=out.index).reindex(idx_open.to_numpy())

    # Build game_stats (one row per group)
    game_stats = pd.DataFrame({
        "Sport": out.loc[idx_open.to_numpy(), "Sport"].to_numpy(),
        "Market": out.loc[idx_open.to_numpy(), "Market"].to_numpy(),
        "Bookmaker": out.loc[idx_open.to_numpy(), "Bookmaker"].to_numpy(),
        "Game_Key": out.loc[idx_open.to_numpy(), "Game_Key"].to_numpy(),
        "Game_Start": game_start_open.to_numpy(),
        "Open_Good_Game": open_good.to_numpy(),
        "Fast_Good_Game": fast_good_np,
    })

    # ---- cumulative, leak-safe per (Sport,Market,Bookmaker) ----
    game_stats = game_stats.sort_values(["Sport", "Market", "Bookmaker", "Game_Start"], kind="mergesort")
    g = game_stats.groupby(["Sport", "Market", "Bookmaker"], sort=False, observed=True)

    prior_n  = g.cumcount()
    cum_open = g["Open_Good_Game"].cumsum() - game_stats["Open_Good_Game"]
    cum_fast = g["Fast_Good_Game"].cumsum() - game_stats["Fast_Good_Game"]

    a0_open = prior_strength_open * 0.5
    b0_open = prior_strength_open * 0.5
    post_open = (cum_open + a0_open) / (prior_n + a0_open + b0_open)
    post_open = np.where(prior_n >= min_trials_for_stats, post_open, 0.5)

    a0_spd = prior_strength_speed * 0.5
    b0_spd = prior_strength_speed * 0.5
    post_spd = (cum_fast + a0_spd) / (prior_n + a0_spd + b0_spd)
    post_spd = np.where(prior_n >= min_trials_for_stats, post_spd, 0.5)

    game_stats["Book_Path_Open_Score"]  = post_open
    game_stats["Book_Path_Speed_Score"] = post_spd

    p1 = np.clip(post_open, 0.01, 0.99)
    p2 = np.clip(post_spd, 0.01, 0.99)
    game_stats["Book_Path_Open_Lift"]  = np.log(p1 / (1.0 - p1))
    game_stats["Book_Path_Speed_Lift"] = np.log(p2 / (1.0 - p2))

    # ---- merge back ----
    # Many rows per key → one row in game_stats (validate many_to_one)
    merged = df.merge(
        game_stats[[
            "Sport", "Market", "Bookmaker", "Game_Key",
            "Book_Path_Open_Score", "Book_Path_Speed_Score",
            "Book_Path_Open_Lift",  "Book_Path_Speed_Lift",
        ]],
        on=["Sport", "Market", "Bookmaker", "Game_Key"],
        how="left",
        validate="many_to_one",
    )

    # If df wasn't normalized but assume_normalized=True was passed incorrectly,
    # this will produce lots of NaNs. That's fine; you’ll notice quickly.

    return merged


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



def _amer_to_prob_one(odds) -> float:
    if pd.isna(odds): return np.nan
    o = float(odds)
    return 100.0/(o+100.0) if o>=0 else (-o)/((-o)+100.0)

def _devig_pair(p_a: float, p_b: float) -> float:
    if np.isnan(p_a) or np.isnan(p_b): return np.nan
    s = p_a + p_b
    return np.nan if s <= 0 else p_a / s  # returns prob for the "A" leg



def build_cross_market_pivots_for_training(df: pd.DataFrame) -> pd.DataFrame:
    need = [c for c in ["Game_Key","Bookmaker","Market","Outcome","Value","Odds_Price","Snapshot_Timestamp"] if c in df.columns]
    g = (df.loc[:, need]
           .sort_values("Snapshot_Timestamp")
           .groupby(["Game_Key","Bookmaker","Market","Outcome"], as_index=False)
           .tail(1))

    # Spread magnitude
    spread = g[g.Market.str.lower().eq("spreads")].copy()
    spread["Spread_Value"] = pd.to_numeric(spread["Value"], errors="coerce").abs()
    spread = spread.groupby(["Game_Key","Bookmaker"], as_index=False).agg(Spread_Value=("Spread_Value","max"))

    # Totals p(Over) de-vig
    tots = g[g.Market.str.lower().eq("totals")].copy()
    tots["Outcome_l"] = tots["Outcome"].astype(str).str.lower()
    tots["Total_Value"] = pd.to_numeric(tots["Value"], errors="coerce")
    tots["p_raw"] = tots["Odds_Price"].map(_amer_to_prob_one)
    over  = tots[tots["Outcome_l"].eq("over") ][["Game_Key","Bookmaker","Total_Value","p_raw"]].rename(columns={"p_raw":"p_over_raw"})
    under = tots[tots["Outcome_l"].eq("under")][["Game_Key","Bookmaker","p_raw"]].rename(columns={"p_raw":"p_under_raw"})
    tot_p = over.merge(under, on=["Game_Key","Bookmaker"], how="left")
    tot_p["p_over"] = tot_p.apply(lambda r: _devig_pair(r["p_over_raw"], r["p_under_raw"]), axis=1)

    # ML favorite prob de-vig
    h2h = g[g.Market.str.lower().eq("h2h")].copy()
    h2h["p_raw"] = h2h["Odds_Price"].map(_amer_to_prob_one)
    favdog = (h2h.sort_values(["Game_Key","Bookmaker","p_raw"], ascending=[True,True,False])
                .groupby(["Game_Key","Bookmaker"], as_index=False)
                .agg(p_fav_raw=("p_raw","first"), p_dog_raw=("p_raw","last")))
    favdog["p_ml_fav"] = favdog.apply(lambda r: _devig_pair(r["p_fav_raw"], r["p_dog_raw"]), axis=1)

    return (spread
            .merge(tot_p[["Game_Key","Bookmaker","Total_Value","p_over"]], on=["Game_Key","Bookmaker"], how="outer")
            .merge(favdog[["Game_Key","Bookmaker","p_ml_fav"]],             on=["Game_Key","Bookmaker"], how="outer"))


def _bin_spread(x):
    if pd.isna(x): return np.nan
    x = abs(float(x))
    return "≥10" if x>=10 else ("7–9.5" if x>=7 else ("3–6.5" if x>=3 else "<3"))

def _bin_total(x):
    if pd.isna(x): return np.nan
    x = float(x)
    return "high" if x>=52 else ("mid" if x>=46 else "low")

def _bin_ml(p):
    if pd.isna(p): return np.nan
    p = float(p)
    if p>=0.80: return "≥0.80"
    if p>=0.70: return "0.70–0.79"
    if p>=0.60: return "0.60–0.69"
    return "0.50–0.59"

def _rho_xy(x_bool: pd.Series, y_bool: pd.Series) -> float:
    x = x_bool.astype(float).to_numpy()
    y = y_bool.astype(float).to_numpy()
    if len(x) < 150 or x.std()==0 or y.std()==0:  # stability guard
        return np.nan
    return float(np.corrcoef(x, y)[0,1])

def build_corr_lookup_ST_SM_TM(hist: pd.DataFrame, sport: str = "NFL"):
    H = hist.loc[hist["Sport"].astype(str).str.upper().eq(str(sport).upper())].copy()

    def _bin_spread(x):
        if pd.isna(x): return np.nan
        x = abs(float(x))
        return "≥10" if x>=10 else ("7–9.5" if x>=7 else ("3–6.5" if x>=3 else "<3"))

    def _bin_total(x):
        if pd.isna(x): return np.nan
        x = float(x)
        return "high" if x>=52 else ("mid" if x>=46 else "low")

    def _bin_ml(p):
        if pd.isna(p): return np.nan
        p = float(p)
        if p>=0.80: return "≥0.80"
        if p>=0.70: return "0.70–0.79"
        if p>=0.60: return "0.60–0.69"
        return "0.50–0.59"

    if "spread_bin" not in H.columns: H["spread_bin"] = H["close_spread"].apply(_bin_spread)
    if "total_bin"  not in H.columns: H["total_bin"]  = H["close_total"].apply(_bin_total)
    if "ml_bin"     not in H.columns: H["ml_bin"]     = H["p_ml_fav"].apply(_bin_ml)

    def _rho_xy(xb, yb):
        x = xb.astype(float).to_numpy(); y = yb.astype(float).to_numpy()
        if len(x) < 150 or np.nanstd(x)==0 or np.nanstd(y)==0: return np.nan
        return float(np.corrcoef(x, y)[0,1])

    # Build rows
    ST_rows = [{'spread_bin':sb,'total_bin':tb,'rho_ST':_rho_xy(g['fav_covered'],g['went_over']),'n':len(g)}
               for (sb,tb), g in H.groupby(['spread_bin','total_bin'], dropna=True)]
    SM_rows = [{'spread_bin':sb,'ml_bin':mb,'rho_SM':_rho_xy(g['fav_covered'],g['fav_won']),'n':len(g)}
               for (sb,mb), g in H.groupby(['spread_bin','ml_bin'], dropna=True)]
    TM_rows = [{'total_bin':tb,'ml_bin':mb,'rho_TM':_rho_xy(g['went_over'],g['fav_won']),'n':len(g)}
               for (tb,mb), g in H.groupby(['total_bin','ml_bin'], dropna=True)]

    # Guarantee columns even if empty
    ST_lookup = pd.DataFrame(ST_rows, columns=["spread_bin","total_bin","rho_ST","n"])
    SM_lookup = pd.DataFrame(SM_rows, columns=["spread_bin","ml_bin","rho_SM","n"])
    TM_lookup = pd.DataFrame(TM_rows, columns=["total_bin","ml_bin","rho_TM","n"])

    return ST_lookup, SM_lookup, TM_lookup





def attach_pairwise_correlation_features(
    df: pd.DataFrame,
    cross_pivots: pd.DataFrame,
    ST_lookup: pd.DataFrame,
    SM_lookup: pd.DataFrame,
    TM_lookup: pd.DataFrame,
    sport_default: str="NFL",
) -> pd.DataFrame:
    if df.empty:
        return df

    out = merge_drop_overlap(
        df,
        cross_pivots,
        on=["Game_Key", "Bookmaker"],
        how="left",
        keep_right=True,      # cross_pivots should win if it has the same fields
        validate="m:1",
    )

    # ---- GUARANTEE Spread_Value / Total_Value exist (avoid KeyError) ----
    # If pivots didn't bring them, derive from per-row (Market, Value) and broadcast per (Game_Key, Bookmaker).
    if "Spread_Value" not in out.columns:
        out["Spread_Value"] = np.nan
        if "Market" in out.columns and "Value" in out.columns:
            m = out["Market"].astype(str).str.lower().str.strip()
            v = pd.to_numeric(out["Value"], errors="coerce")
            out.loc[m == "spreads", "Spread_Value"] = v
            out["Spread_Value"] = out.groupby(["Game_Key","Bookmaker"])["Spread_Value"].transform(
                lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
            )

    if "Total_Value" not in out.columns:
        out["Total_Value"] = np.nan
        if "Market" in out.columns and "Value" in out.columns:
            m = out["Market"].astype(str).str.lower().str.strip()
            v = pd.to_numeric(out["Value"], errors="coerce")
            out.loc[m == "totals", "Total_Value"] = v
            out["Total_Value"] = out.groupby(["Game_Key","Bookmaker"])["Total_Value"].transform(
                lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan
            )

    # ---- Core probs (now safe) ----
    sport = out.get("Sport", sport_default).astype(str).str.upper().to_numpy()
    spread_abs = pd.to_numeric(out["Spread_Value"], errors="coerce").to_numpy(dtype="float64")
    p_spread_fav = _spread_to_winprob(spread_abs, sport)

    p_ml_fav = pd.to_numeric(out.get("p_ml_fav"), errors="coerce").to_numpy(dtype="float64")
    p_over   = pd.to_numeric(out.get("p_over"), errors="coerce").to_numpy(dtype="float64")

    # Side-awareness ...
    if "Is_Favorite_Bet" in out.columns:
        is_fav = out["Is_Favorite_Bet"].astype("float32").to_numpy() == 1.0
    else:
        v = pd.to_numeric(out.get("Value"), errors="coerce").to_numpy(dtype="float64")
        is_fav = np.where(np.isnan(v), False, v < 0)

    p_side = np.where(is_fav, p_spread_fav, 1.0 - p_spread_fav)

    # Bins for lookups
    out["spread_bin"] = pd.Series(out.get("Spread_Value")).apply(_bin_spread)
    out["total_bin"]  = pd.Series(out.get("Total_Value")).apply(_bin_total)
    out["ml_bin"]     = pd.Series(out.get("p_ml_fav")).apply(_bin_ml)

    out = out.merge(ST_lookup[["spread_bin","total_bin","rho_ST"]], on=["spread_bin","total_bin"], how="left")
    out = out.merge(SM_lookup[["spread_bin","ml_bin","rho_SM"]],     on=["spread_bin","ml_bin"],    how="left")
    out = out.merge(TM_lookup[["total_bin","ml_bin","rho_TM"]],     on=["total_bin","ml_bin"],     how="left")

    # Synergy = rho * sqrt(p(1-p) q(1-q))  (NaN-safe)
    def _synergy(p, q, rho):
        mult = np.sqrt(np.clip(p*(1-p),0,1) * np.clip(q*(1-q),0,1))
        return rho * mult

    # Spread↔Total (p_side vs p_over)
    ST_syn = _synergy(p_side, p_over, pd.to_numeric(out["rho_ST"], errors="coerce").to_numpy(float))
    out["SpreadTotal_Rho"]     = pd.to_numeric(out["rho_ST"]).astype("float32")
    out["SpreadTotal_Synergy"] = pd.Series(ST_syn).astype("float32")
    out["SpreadTotal_Sign"]    = ((p_side - 0.5)*(p_over - 0.5)).astype("float32")

    # Spread↔ML (p_spread_fav vs p_ml_fav)
    SM_syn = _synergy(p_spread_fav, p_ml_fav, pd.to_numeric(out["rho_SM"], errors="coerce").to_numpy(float))
    out["SpreadML_Rho"]        = pd.to_numeric(out["rho_SM"]).astype("float32")
    out["SpreadML_Synergy"]    = pd.Series(SM_syn).astype("float32")
    out["SpreadML_Sign"]       = ((p_spread_fav - 0.5)*(p_ml_fav - 0.5)).astype("float32")

    # Total↔ML (p_over vs p_ml_fav)
    TM_syn = _synergy(p_over, p_ml_fav, pd.to_numeric(out["rho_TM"], errors="coerce").to_numpy(float))
    out["TotalML_Rho"]         = pd.to_numeric(out["rho_TM"]).astype("float32")
    out["TotalML_Synergy"]     = pd.Series(TM_syn).astype("float32")
    out["TotalML_Sign"]        = ((p_over - 0.5)*(p_ml_fav - 0.5)).astype("float32")

    # (Optional) keep your existing consistency gap too
    if "Spread_ML_ProbGap" not in out.columns:
        gap = np.abs(p_spread_fav - p_ml_fav)
        out["Spread_ML_ProbGap"] = pd.Series(gap).astype("float32")

    return out



# --- CACHED CLIENT (resource-level) ---
@st.cache_resource
def get_bq() -> bigquery.Client:
    return bigquery.Client(project="sharplogger")

                     
def _iso(ts) -> str:
    return pd.to_datetime(ts, utc=True).isoformat()


@st.cache_data(show_spinner=False)
# ===========================

# ===========================
def _resolve_feature_cols_like_training(bundle, model=None, df_like=None, market: str | None = None) -> list[str]:
    """
    Return the exact feature columns used at training time.

    Priority:
      1) bundle[market]['feature_cols'] or bundle['feature_cols']
      2) Other common bundle keys: 'feature_names', 'features', 'training_features', etc.
      3) Names exposed by the model (feature_names_in_, booster.feature_names, etc.)
      4) Heuristic fallback from df_like (numeric/boo columns minus obvious IDs/targets)

    Always de-duplicates and (if df_like is provided) filters to columns present in df_like.
    """
   

    # --- unwrap per-market sub-bundle if provided ---
    sub = bundle
    if isinstance(bundle, dict) and market and market in bundle and isinstance(bundle[market], dict):
        sub = bundle[market]

    def _first_nonempty_list(d: dict, keys: tuple) -> list[str] | None:
        if not isinstance(d, dict):
            return None
        for k in keys:
            v = d.get(k)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return [str(c) for c in v if c is not None]
        return None

    # 1) Directly from bundle (you save 'feature_cols' explicitly)
    names = None
    if isinstance(sub, dict):
        names = _first_nonempty_list(
            sub,
            ("feature_cols", "feature_names", "features", "training_features",
             "feature_list", "X_cols", "input_cols", "training_columns", "columns")
        )
        # look in nested metadata if needed
        if not names:
            for meta_key in ("metadata", "meta", "info"):
                md = sub.get(meta_key)
                if isinstance(md, dict):
                    names = _first_nonempty_list(
                        md,
                        ("feature_cols", "feature_names", "features", "training_features",
                         "feature_list", "X_cols", "input_cols")
                    )
                    if names:
                        break
        # sometimes models live inside the bundle
        if not names:
            for mkey in ("model_logloss", "model_auc", "model"):
                if mkey in sub:
                    names = None
                    try:
                        m = sub[mkey]
                        # sklearn ≥1.0
                        fn = getattr(m, "feature_names_in_", None)
                        if fn is not None and len(fn) > 0:
                            names = [str(c) for c in list(fn)]
                        # ColumnTransformer / Pipeline
                        if not names and hasattr(m, "get_feature_names_out"):
                            out = m.get_feature_names_out()
                            if out is not None and len(out) > 0:
                                names = [str(c) for c in list(out)]
                        # XGBoost booster
                        if not names and hasattr(m, "get_booster"):
                            b = m.get_booster()
                            if b is not None and getattr(b, "feature_names", None):
                                names = [str(c) for c in list(b.feature_names)]
                    except Exception:
                        pass
                    if names:
                        break

    # 2) If still nothing, try the separate model arg
    if not names and model is not None:
        try:
            fn = getattr(model, "feature_names_in_", None)
            if fn is not None and len(fn) > 0:
                names = [str(c) for c in list(fn)]
        except Exception:
            pass
        if not names:
            try:
                if hasattr(model, "get_feature_names_out"):
                    out = model.get_feature_names_out()
                    if out is not None and len(out) > 0:
                        names = [str(c) for c in list(out)]
            except Exception:
                pass
        if not names:
            try:
                booster = getattr(model, "get_booster", None)
                if callable(booster):
                    b = booster()
                    if b is not None and getattr(b, "feature_names", None):
                        names = [str(c) for c in list(b.feature_names)]
            except Exception:
                pass

    # 3) Heuristic fallback from df_like
    if not names:
        names = []
        if isinstance(df_like, pd.DataFrame):
            bad_exact = {
                "y","label","target",
                "Game","Game_Key","Game_Key_Base","Team_Key","Team_Key_Base",
                "Home_Team","Away_Team","Home_Team_Norm","Away_Team_Norm","Team",
                "Outcome","Outcome_Norm","Sport","League","Book","Bookmaker",
                "Snapshot_Timestamp","Insert_Timestamp","Time","Game_Start","Commence_Hour",
                "Was_Canonical","Scored_By_Model","Scoring_Market",
                "Model_Sharp_Win_Prob","Model_Confidence","Model_Confidence_Tier","Why_Model_Likes_It",
            }
            bad_prefix = ("id","idx","__","Model_","Why_","Team_Key","Game_Key","hash_","raw_")
            def _is_numeric_like(s: pd.Series) -> bool:
                from pandas.api.types import is_numeric_dtype, is_bool_dtype
                try:
                    return bool(is_numeric_dtype(s) or is_bool_dtype(s))
                except Exception:
                    return False
            cols_numeric = [
                c for c in df_like.columns
                if (c not in bad_exact)
                and (not any(str(c).startswith(p) for p in bad_prefix))
                and _is_numeric_like(df_like[c])
            ]
            # prefer common feature-y prefixes
            good_prefixes = (
                "Sharp_","Rec_","Line_","Odds_","Implied_","Limit","Market_",
                "Hybrid_","Timing_","Spread_","Total_","H2H_",
                "Book_Reliability","SmallBook_","CrossMarket_",
                "Team_Past_","Team_","Power_Rating","Net_","Abs_","Pct_","Avg_","Rate_",
                "Is_","Same_","Opposite_","Potential_","Mispricing_","Value_","Minutes_",
            )
            preferred = [c for c in cols_numeric if any(str(c).startswith(p) for p in good_prefixes)]
            others    = [c for c in cols_numeric if c not in preferred]
            names = preferred + others

    # 4) Cup: drop dupes, keep order, and (if df_like is given) filter to existing cols
    seen, final = set(), []
    for c in names or []:
        c = str(c)
        if df_like is not None and c not in df_like.columns:
            continue
        if c not in seen:
            seen.add(c)
            final.append(c)
    return final



def holdout_by_percent_groups(
    *,
    sport: str | None = None,
    groups: np.ndarray,
    times: np.ndarray | None,
    y: np.ndarray,
    pct_holdout: float | None = None,
    min_train_games: int = 25,
    min_hold_games: int = 8,
    ensure_label_diversity: bool = True,
):
    """
    Time-forward holdout by last % of GROUPS (e.g., Game_Key).
    If pct_holdout is None, choose per sport via SPORT_HOLDOUT_PCT.
    Returns (train_idx, hold_idx) as **row indices** (int arrays), sorted ascending.
    """
    import numpy as np
    import pandas as pd

    # ---- sport-defaults ----
    SPORT_HOLDOUT_PCT = {
        "NFL": 0.12, "NCAAF": 0.12, "NBA": 0.18, "WNBA": 0.12,
        "NHL": 0.18, "MLB": 0.20, "MLS": 0.18, "CFL": 0.12, "DEFAULT": 0.18,"NCCAM": 0.18,
    }
    if pct_holdout is None:
        key = (sport or "DEFAULT").upper()
        pct_holdout = float(SPORT_HOLDOUT_PCT.get(key, SPORT_HOLDOUT_PCT["DEFAULT"]))
    pct_holdout = float(np.clip(pct_holdout, 0.05, 0.50))  # keep reasonable bounds

    # ---- align lengths safely (no crashes on mismatches) ----
    n = int(min(len(groups), len(y), len(times) if times is not None else len(groups)))
    groups = np.asarray(groups)[:n]
    y = np.asarray(y).astype(int)[:n]
    if times is None:
        # Use an increasing counter as a last-resort "time"
        times = np.arange(n)
        times_is_datetime = False
    else:
        times = np.asarray(times)[:n]
        times_is_datetime = True

    # ---- row-level frame (we keep all rows; do not drop NaT rows) ----
    df_rows = pd.DataFrame({
        "row_idx": np.arange(n, dtype=int),
        "group":   pd.Series(groups).astype(str),
        "y":       y,
    })
    # Parse times; may produce NaT
    if times_is_datetime:
        t_ser = pd.to_datetime(times, utc=True, errors="coerce")
    else:
        # fallback: monotonic increasing pseudo-time
        t_ser = pd.to_datetime(pd.Series(times, dtype="int64"), unit="s", utc=True, errors="ignore")
    df_rows["time"] = t_ser

    # ---- group-level meta (order by real time; fallback to first row order) ----
    gfirst = df_rows.groupby("group", sort=False)["row_idx"].min()
    gstart = df_rows.dropna(subset=["time"]).groupby("group")["time"].min()
    gend   = df_rows.dropna(subset=["time"]).groupby("group")["time"].max()

    gmeta = pd.DataFrame({
        "group": gfirst.index,
        "first_row": gfirst.values,                  # fallback order
        "start": gstart.reindex(gfirst.index),       # may be NaT
        "end":   gend.reindex(gfirst.index),         # may be NaT
    })
    # Primary sort: by start time; Secondary: by first occurrence (stable for NaT)
    gmeta = gmeta.sort_values(by=["start", "first_row"], na_position="last").reset_index(drop=True)

    n_groups = len(gmeta)
    if n_groups == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # ---- choose #hold groups with bounds ----
    # ensure at least 1 train group exists
    min_train_groups = max(1, int(min_train_games))
    min_hold_groups  = max(1, int(min_hold_games))

    # requested hold groups by pct
    wanted_hold = int(np.ceil(n_groups * pct_holdout))
    # clamp to feasible window
    max_hold_allowed = max(1, n_groups - min_train_groups)
    n_hold_groups = int(np.clip(wanted_hold, min_hold_groups, max_hold_allowed))

    # If dataset is too small, shrink gracefully
    if n_groups < (min_train_groups + min_hold_groups):
        # leave at least 1 train group
        n_hold_groups = int(np.clip(n_groups - 1, 1, n_groups))

    # ---- select train/hold groups (last % by time) ----
    hold_groups = gmeta["group"].iloc[-n_hold_groups:].to_numpy()
    train_groups = gmeta["group"].iloc[: n_groups - n_hold_groups].to_numpy()

    # ---- map groups back to rows (non-overlapping; sorted) ----
    all_groups = df_rows["group"].to_numpy()
    hold_mask = np.isin(all_groups, hold_groups)
    train_mask = np.isin(all_groups, train_groups)

    # enforce disjointness (just in case)
    both = train_mask & hold_mask
    if np.any(both):
        hold_mask[both] = False  # keep in train if ever ambiguous

    hold_idx = np.sort(np.flatnonzero(hold_mask).astype(int))
    train_idx = np.sort(np.flatnonzero(train_mask).astype(int))

    # ---- optional: ensure label diversity in holdout (expand boundary if needed) ----
    if ensure_label_diversity and hold_idx.size > 0:
        def _has_both(idx_rows: np.ndarray) -> bool:
            if idx_rows.size == 0:
                return False
            return np.unique(df_rows.loc[idx_rows, "y"]).size >= 2

        if not _has_both(hold_idx):
            k = n_hold_groups
            # expand hold boundary forward as much as possible while leaving min_train_groups
            while (not _has_both(hold_idx)) and ((n_groups - k) >= min_train_groups) and (k < n_groups):
                k += 1
                hold_groups = gmeta["group"].iloc[-k:].to_numpy()
                train_groups = gmeta["group"].iloc[: n_groups - k].to_numpy()
                hold_mask = np.isin(all_groups, hold_groups)
                train_mask = np.isin(all_groups, train_groups)
                both = train_mask & hold_mask
                if np.any(both):
                    hold_mask[both] = False
                hold_idx = np.sort(np.flatnonzero(hold_mask).astype(int))
                train_idx = np.sort(np.flatnonzero(train_mask).astype(int))

    return train_idx, hold_idx



def sharp_row_weights(df, a_sharp=0.8, b_limit=0.15, c_liq=0.10, d_steam=0.10):
  
    w = np.ones(len(df), dtype=np.float32)

    # sharp‑ish signals available in your schema
    if "Market_Leader" in df.columns:
        # if your Market_Leader is boolean (0/1 or True/False)
        try:
            w += 0.25 * df["Market_Leader"].fillna(False).astype(float)
        except Exception:
            pass

    if "High_Limit_Flag" in df.columns:
        w += 0.10 * df["High_Limit_Flag"].fillna(False).astype(float)

    if "Late_Game_Steam_Flag" in df.columns:
        w += d_steam * df["Late_Game_Steam_Flag"].fillna(False).astype(float)

    if "Limit" in df.columns:
        lim = pd.to_numeric(df["Limit"], errors="coerce").fillna(0.0)
        med = lim.median(); mad = (np.abs(lim - med).median() + 1e-9)
        z = np.clip((lim - med) / (1.4826 * mad + 1e-9), -3, 3)
        w += b_limit * z.astype(np.float32)

    # if you later add a boolean "Is_Sharp_Book", include it:
    if "Is_Sharp_Book" in df.columns:
        w += a_sharp * df["Is_Sharp_Book"].fillna(False).astype(float)

    return w.astype(np.float32)


def build_game_market_sharpaware_schema(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    key_col: str = "Merge_Key_Short",
    book_col_candidates: tuple[str, ...] = ("Bookmaker","Bookmaker_Norm"),
    outcome_col: str = "Outcome",
    time_col: str = "Snapshot_Timestamp",
    start_col: str = "Game_Start",
    label_col: str = "SHARP_HIT_BOOL",
    reliability_col: str = "Book_Reliability_x_Magnitude",
    sharp_flag: str = "Is_Sharp_Book",
    sharp_tilt: float = 0.5,
    normalize_within_game: bool = True,
    fill_missing_feature: float = 0.0,
) -> pd.DataFrame:
    """
    Returns one row per (Game × Bookmaker × Outcome) for training.
    - Dedups to last snapshot per (game, book, outcome)
    - Keeps both outcomes (no canonical filtering)
    - Returns features + y + sample_weight
    - Uses the same function name as before so downstream doesn’t break
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    # Resolve book column
    book_col = next((c for c in book_col_candidates if c in d.columns), None)
    if book_col is None:
        book_col = "_ONE_BOOK"
        d[book_col] = "ONE"

    # Basic checks
    for c in (key_col, outcome_col, start_col, label_col):
        if c not in d.columns:
            raise KeyError(f"Missing required column: {c}")
    d = d[d[label_col].isin([0, 1])]
    if d.empty:
        return pd.DataFrame()

    # Fill defaults
    if reliability_col not in d.columns:
        d[reliability_col] = 1.0
    if sharp_flag not in d.columns:
        d[sharp_flag] = 0

    # Dedup: last row per (game, book, outcome)
    sort_cols = [key_col, book_col, outcome_col]
    if time_col in d.columns:
        d = d.sort_values(sort_cols + [time_col])
    elif "Inserted_Timestamp" in d.columns:
        d = d.sort_values(sort_cols + ["Inserted_Timestamp"])
    else:
        d = d.sort_values(sort_cols)
    d = d.groupby([key_col, book_col, outcome_col], as_index=False).tail(1)

    # Sample weight = reliability × (1 + tilt × sharp_flag)
    base_w = d[reliability_col].astype(float).clip(0, 1) * (1.0 + sharp_tilt * d[sharp_flag].astype(int))
    d["_w_base"] = base_w.astype(float)

    # Vectorized per-game normalization (no apply, no index issues)
    sum_w = d.groupby(key_col)["_w_base"].transform("sum")
    cnt_w = d.groupby(key_col)["_w_base"].transform("size").clip(lower=1)
    
    d["_w"] = np.where(sum_w > 0, d["_w_base"] / sum_w, 1.0 / cnt_w)
    d["_w"] = d["_w"].astype(float)

    # Feature matrix construction with padding
    X = d.reindex(columns=feature_cols, copy=False)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = fill_missing_feature
        else:
            s = X[c]
            if s.dtype == object:
                X[c] = pd.to_numeric(
                    s.replace({
                        'True': 1, 'False': 0, True: 1, False: 0,
                        '': np.nan, 'none': np.nan, 'None': np.nan,
                        'NA': np.nan, 'NaN': np.nan, 'unknown': np.nan
                    }),
                    errors="coerce"
                )
            elif s.dtype == bool:
                X[c] = s.astype(float)
    X = X.fillna(fill_missing_feature).astype(np.float32, copy=False)
    X = X[feature_cols]

    # Final output frame with label and weight
    out = X.copy()
    out["y"] = d[label_col].astype(int).to_numpy()
    out["sample_weight"] = d["_w"].astype(float).to_numpy()
    out["Merge_Key_Short"] = d[key_col].values
    out["Bookmaker"] = d[book_col].values
    out["Outcome"] = d[outcome_col].values
    out["Game_Start"] = pd.to_datetime(d[start_col], errors="coerce", utc=True)

    return out.reset_index(drop=True)


import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

# ------- lightweight corr-pruner (keeps highest-ranked, drops high-corr followers) -------
def greedy_corr_prune(X: pd.DataFrame, candidates, rank_df: pd.DataFrame,
                      corr_thresh=0.92, must_keep=None):
    must_keep = set(must_keep or [])
    C = X.loc[:, candidates].astype(float)
    # order by rank_df (avg_abs_shap desc, then presence desc, then avg_rank asc)
    score = pd.Series(0.0, index=C.columns)
    if "avg_abs_shap" in rank_df.columns:
        score = score.add(rank_df["avg_abs_shap"].reindex(C.columns).fillna(0.0))
    if "presence" in rank_df.columns:
        score = score.add(rank_df["presence"].reindex(C.columns).fillna(0.0) * 0.01)
    if "avg_rank" in rank_df.columns:
        score = score.sub(rank_df["avg_rank"].reindex(C.columns).fillna(score.max()+1) * 1e-4)
    order = list(pd.Series(score).sort_values(ascending=False).index)
    keep, dropped = [], set()
    corr = C.corr().abs()
    for f in order:
        if f in dropped:
            continue
        keep.append(f)
        # drop those too correlated with f (unless must_keep)
        too_cor = corr.index[(corr[f] > corr_thresh) & (corr.index != f)]
        for g in too_cor:
            if g not in must_keep:
                dropped.add(g)
    # always ensure must_keep survive
    for m in must_keep:
        if m in C.columns and m not in keep:
            keep.append(m)
    return keep

from sklearn.base import clone
from sklearn.metrics import roc_auc_score


# ------- permutation AUC importance (fallback) -------
# --- Drop-in shim: make perm AUC robust to mixed call styles and return (base_auc, df)
def perm_auc_importance(model, X, y, *args, **kwargs):
    """
    Compatible with calls like:
        base_auc, perm_df = perm_auc_importance(model, X_df, y, n_repeats=5, rnd=42)
        base_auc, perm_df = perm_auc_importance(model, X_df, y, 5)  # positional repeats
        base_auc, perm_df = perm_auc_importance(model, X_df, y, repeats=10, random_state=123)
    Returns:
        base_auc: float
        perm_df : DataFrame with [feature, perm_auc_drop_mean, perm_auc_drop_std] indexed by feature
    """
    # Resolve repeats / n_repeats from args/kwargs
    n_repeats_kw = kwargs.pop("n_repeats", None)
    repeats_kw   = kwargs.pop("repeats", None)
    repeats_pos  = args[0] if len(args) >= 1 else None
    repeats = (
        repeats_kw
        if repeats_kw is not None
        else (
            n_repeats_kw
            if n_repeats_kw is not None
            else (repeats_pos if repeats_pos is not None else 5)
        )
    )

    # Resolve RNG seed (rnd OR random_state)
    rnd_kw          = kwargs.pop("rnd", None)
    random_state_kw = kwargs.pop("random_state", None)
    seed = random_state_kw if random_state_kw is not None else (rnd_kw if rnd_kw is not None else 42)
    rng  = np.random.RandomState(seed)

    # Compute base AUC once
    proba = model.predict_proba(X)[:, 1]
    base_auc = roc_auc_score(y, proba)

    # Permutation drops
    drops = []
    X = X.copy()
    for col in X.columns:
        vals = []
        for _ in range(int(repeats)):
            Xp = X.copy()
            Xp[col] = Xp[col].sample(
                frac=1.0, replace=False,
                random_state=rng.randint(1_000_000_000)
            ).values
            vals.append(base_auc - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        drops.append((col, float(np.mean(vals)), float(np.std(vals))))
    perm_df = (
        pd.DataFrame(drops, columns=["feature", "perm_auc_drop_mean", "perm_auc_drop_std"])
        .set_index("feature")
    )
    return base_auc, perm_df


# ------- SHAP stability (your function) -------
def shap_stability_select(
    model_proto,
    X: pd.DataFrame,
    y: np.ndarray,
    folds, *,
    topk_per_fold: int = 150,
    min_presence: float = 0.8,
    max_keep: int | None = None,
    sample_per_fold: int = 4000,
    random_state: int = 42,
    must_keep: list[str] = None
):
    try:
        import shap
    except Exception as e:
        raise RuntimeError("Install shap to use shap_stability_select") from e

    feat = list(X.columns)
    if not feat:
        return [], pd.DataFrame()

    rng = np.random.RandomState(random_state)
    counts_in_topk   = pd.Series(0, index=feat, dtype=int)
    shap_sum_abs     = pd.Series(0.0, index=feat, dtype=float)
    per_fold_abs, per_fold_sign, fold_rank_frames = [], [], []

    for fold in folds:
        tr_idx, _ = fold[:2]
        Xtr, ytr = X.iloc[tr_idx], y[tr_idx]
        mdl = clone(model_proto).fit(Xtr, ytr)

        if len(Xtr) <= sample_per_fold:
            Xs = Xtr
        else:
            Xs = Xtr.iloc[rng.choice(len(Xtr), size=sample_per_fold, replace=False)]

        explainer = shap.TreeExplainer(
            mdl, feature_perturbation="tree_path_dependent"
        )
        sval = explainer.shap_values(Xs)
        if isinstance(sval, list):
            sval = sval[1]
        sval = np.asarray(sval)

        abs_mean  = pd.Series(np.abs(sval).mean(axis=0), index=feat)
        sign_mean = pd.Series(np.sign(sval).mean(axis=0), index=feat)

        shap_sum_abs = shap_sum_abs.add(abs_mean, fill_value=0.0)
        per_fold_abs.append(abs_mean)
        per_fold_sign.append(sign_mean)
        fold_rank_frames.append(abs_mean.rank(ascending=False, method="average"))

        topk = abs_mean.sort_values(ascending=False).head(
            min(topk_per_fold, len(feat))
        ).index
        counts_in_topk.loc[topk] += 1

    n_folds  = max(len(fold_rank_frames), 1)
    avg_rank = pd.concat(fold_rank_frames, axis=1).mean(axis=1)
    presence = counts_in_topk / n_folds
    avg_abs  = shap_sum_abs / n_folds

    S = pd.concat(per_fold_sign, axis=1) if per_fold_sign else pd.DataFrame(index=feat)
    if not S.empty:
        modal_sign = np.sign(S.mean(axis=1)).replace(0, 1)
        disagree   = (np.sign(S).replace(0, 1).ne(modal_sign, axis=0)).sum(axis=1)
        sign_flip_rate = disagree / n_folds
    else:
        sign_flip_rate = pd.Series(0.0, index=feat)

    A = pd.concat(per_fold_abs, axis=1) if per_fold_abs else pd.DataFrame(index=feat)
    if not A.empty:
        shap_cv = (A.std(axis=1) / (A.mean(axis=1).replace(0, np.nan))).fillna(0.0)
    else:
        shap_cv = pd.Series(0.0, index=feat)

    keep_presence   = presence.index[presence >= float(min_presence)].tolist()
    keep_top_global = avg_abs.sort_values(ascending=False).head(topk_per_fold).index.tolist()
    selected = list(dict.fromkeys(keep_top_global + keep_presence))
    selected = sorted(
        selected,
        key=lambda c: (
            -float(avg_abs[c]),
            float(avg_rank[c]),
            float(sign_flip_rate[c]),
            float(shap_cv[c]),
        ),
    )

    if must_keep:
        for c in must_keep:
            if c in feat and c not in selected:
                selected.append(c)

    if max_keep is not None and len(selected) > max_keep:
        base = [c for c in selected if (not must_keep or c not in must_keep)]
        head = base[: max_keep - (len(must_keep or []))]
        selected = head + [
            c for c in (must_keep or []) if c in feat and c not in head
        ]

    summary = (
        pd.DataFrame(
            {
                "avg_rank":       avg_rank,
                "presence":       presence,
                "avg_abs_shap":   avg_abs,
                "sign_flip_rate": sign_flip_rate,
                "shap_cv":        shap_cv,
            }
        )
        .loc[selected]
        .sort_values(["avg_abs_shap", "presence"], ascending=[False, False])
    )
    summary.index.name = "feature"
    return selected, summary


from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import numpy as np

# =========================
# FAST, DIRECTED FEATURE ORIENTATION DROP-IN
# (Adds quick screen + early-abort full-CV for flips)
# =========================

def _safe_predict_proba_pos(mdl, X_val):
    """
    Robustly return P(class==1) for classifiers.
    Falls back to predict() if predict_proba not available.
    """
    if hasattr(mdl, "predict_proba"):
        proba2 = mdl.predict_proba(X_val)
        classes = getattr(mdl, "classes_", None)
        if classes is None:
            return np.asarray(proba2[:, 1], float)
        classes = np.asarray(classes)
        if np.any(classes == 1):
            pos_idx = int(np.where(classes == 1)[0][0])
        else:
            pos_idx = int(len(classes) - 1)
        return np.asarray(proba2[:, pos_idx], float)
    return np.asarray(mdl.predict(X_val), float)

def _auto_flip_mode(x):
    """
    Decide flip mode without allocating a flipped array.
    """
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).any():
        return "none"

    x_nan = np.where(np.isfinite(x), x, np.nan)
    mn = np.nanmin(x_nan)
    mx = np.nanmax(x_nan)

    return "one_minus" if (mn >= -1e-3 and mx <= 1.0 + 1e-3) else "negate"


def _apply_flip_inplace(col, mode: str):
    """
    Apply flip directly to a column view (IN PLACE).
    """
    import numpy as np
    if mode == "one_minus":
        np.subtract(np.float32(1.0), col, out=col)   # col = 1 - col
    elif mode == "negate":
        np.negative(col, out=col)                    # col = -col


from sklearn.base import clone
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, log_loss

# 1) CV EVAL BLOCK (unchanged interface)
# =========================
def _cv_auc_for_feature_set(
    model_proto, X, y, folds, feature_list, *,
    log_func=print,
    debug=False,
    return_oof=True,

    enable_feature_flips=False,
    max_feature_flips=0,
    orient_features=False,
    orient_passes=1,

    auc_weight=1.0,
    ll_weight=0.15,
    brier_weight=0.10,
    eps=1e-6,

    cv_mode: str = "full",
    quick_folds_n: int = 1,
    compute_ll_brier: bool = True,
    max_folds: int | None = None,

    abort_if_cannot_beat_score=None,
    abort_margin=0.0,

    game_keys=None,
    enforce_no_game_overlap=True,
    leak_col_patterns=None,
    drop_forbidden_features=True,
    fallback_to_all_available_features=True,
    require_numeric_features=True,
    min_non_nan_frac = 0.001,
):


    folds = list(folds) if folds is not None else []
    if max_folds is not None and folds:
        folds = folds[: int(max_folds)]

    y = np.asarray(y, int).reshape(-1)
    n = len(y)
    if len(y) != len(X):
        raise ValueError(f"X and y length mismatch: len(X)={len(X)} len(y)={len(y)}")

    if leak_col_patterns is None:
        leak_col_patterns = ["SHARP_COVER_RESULT"]

    def _is_forbidden(col: str) -> bool:
        s = str(col)
        return any(p in s for p in leak_col_patterns)

    def _all_usable_features_from_X() -> list[str]:
        cols = [c for c in list(X.columns) if not _is_forbidden(c)]
        if not cols:
            return []
        keep = []
        for c in cols:
            s = pd.to_numeric(X[c], errors="coerce")
            arr = np.asarray(s, float)
            frac = float(np.isfinite(arr).mean()) if arr.size else 0.0
            if frac >= float(min_non_nan_frac):
                keep.append(c)
        return keep

    feature_list = [f for f in feature_list if f in X.columns]

    forbidden = [f for f in feature_list if _is_forbidden(f)]
    if forbidden:
        msg = f"[LEAK-GUARD] Forbidden feature(s) present: {forbidden[:50]}{'...' if len(forbidden)>50 else ''}"
        if drop_forbidden_features:
            if debug:
                log_func(msg + " (dropping)")
            feature_list = [f for f in feature_list if f not in forbidden]
        else:
            raise RuntimeError(msg)

    if len(feature_list) == 0 and fallback_to_all_available_features:
        feature_list = _all_usable_features_from_X()
        if debug:
            log_func(f"[FEATS] Requested list collapsed; fallback -> using {len(feature_list)} cols")

    if len(feature_list) == 0:
        return {
            "feature_list": [],
            "oof_proba": np.full(n, np.nan, dtype=float) if return_oof else None,
            "auc": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
            "score": float("-inf"),
            "aborted": False,
        }

    folds_np = [(np.asarray(tr, np.int64), np.asarray(va, np.int64)) for tr, va in (folds or [])]
    folds = folds_np

    if cv_mode == "quick" and folds:
        folds = folds[: max(1, int(quick_folds_n))]

    def _assert_no_game_overlap(tr_idx, val_idx):
        if not enforce_no_game_overlap or game_keys is None:
            return
        g = np.asarray(game_keys)
        inter = set(map(str, g[tr_idx])) & set(map(str, g[val_idx]))
        if inter:
            raise RuntimeError(f"[LEAK] Game overlap between train/val: {len(inter)} games. Sample={list(inter)[:10]}")

    # numeric cache
    _MAT_CACHE = getattr(_cv_auc_for_feature_set, "_MAT_CACHE", {})
    sig = (id(X), tuple(X.columns), bool(require_numeric_features))
    if sig in _MAT_CACHE:
        X_all_mat, col_ix = _MAT_CACHE[sig]
    else:
        X_num_all = X.apply(pd.to_numeric, errors="coerce") if require_numeric_features else X
        X_all_mat = X_num_all.to_numpy(dtype=np.float32, copy=False)
        col_ix = {c: i for i, c in enumerate(X.columns)}
        _MAT_CACHE[sig] = (X_all_mat, col_ix)
        _cv_auc_for_feature_set._MAT_CACHE = _MAT_CACHE

    feats = list(feature_list)
    cols_idx = [col_ix[f] for f in feats if f in col_ix]

    if not cols_idx and fallback_to_all_available_features:
        feats = _all_usable_features_from_X()
        cols_idx = [col_ix[f] for f in feats if f in col_ix]

    if not cols_idx:
        return {
            "feature_list": [],
            "oof_proba": np.full(n, np.nan, dtype=float) if return_oof else None,
            "auc": float("nan"),
            "logloss": float("nan"),
            "brier": float("nan"),
            "score": float("-inf"),
            "aborted": False,
        }

    X_mat = X_all_mat[:, cols_idx]

    # xgb native fast path (helpers must exist)
    use_xgb = False
    try:
        use_xgb = bool(_is_xgb_classifier(model_proto))
    except Exception:
        use_xgb = False

    xgb_params = None
    xgb_num_round = None
    if use_xgb:
        import xgboost as xgb  # noqa
        xgb_params, xgb_num_round = _xgb_params_from_proto(model_proto)
        if isinstance(xgb_params, dict) and "nthread" not in xgb_params and "n_jobs" not in xgb_params:
            xgb_params["nthread"] = 0
   

    def _fit_predict_proba(X_tr, y_tr, X_va):
        if use_xgb:
            import xgboost as xgb  # noqa
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dva = xgb.DMatrix(X_va)
            booster = xgb.train(xgb_params, dtr, num_boost_round=int(xgb_num_round))
            return booster.predict(dva)
        mdl = clone(model_proto)
        mdl.fit(X_tr, y_tr)
        return _safe_predict_proba_pos(mdl, X_va)

   
    def _oof_metrics_for_mat(X_local_mat):
        oof = np.full(n, np.nan, dtype=np.float32) if return_oof else None
    
        do_abort = abort_if_cannot_beat_score is not None and np.isfinite(abort_if_cannot_beat_score)
        auc_fold_vals = []
        n_folds = len(folds)
    
        # ---------------------------
        # ✅ XGB no-copy CV fast path
        # ---------------------------
        if use_xgb:
            import xgboost as xgb  # noqa
    
            # build once (float32 already)
            d_all = xgb.DMatrix(X_local_mat, label=y)
    
            # small helper
            def _fit_predict_xgb(tr_idx, val_idx):
                d_tr = d_all.slice(tr_idx)
                d_va = d_all.slice(val_idx)
                booster = xgb.train(xgb_params, d_tr, num_boost_round=int(xgb_num_round))
                return booster.predict(d_va)
    
            for (tr_idx, val_idx) in folds:
                _assert_no_game_overlap(tr_idx, val_idx)
    
                proba = np.asarray(_fit_predict_xgb(tr_idx, val_idx), float).ravel()
                if oof is not None:
                    oof[val_idx] = proba.astype(np.float32, copy=False)
    
                if do_abort:
                    try:
                        auc_fold = roc_auc_score(y[val_idx], np.clip(proba, eps, 1 - eps))
                        if not np.isfinite(auc_fold):
                            auc_fold = 0.5
                    except Exception:
                        auc_fold = 0.5
    
                    auc_fold_vals.append(float(auc_fold))
                    left = n_folds - len(auc_fold_vals)
                    mean_auc_best_possible = (sum(auc_fold_vals) + left * 1.0) / float(n_folds)
                    best_possible_score = (auc_weight * mean_auc_best_possible)
                    if best_possible_score < float(abort_if_cannot_beat_score) + float(abort_margin):
                        return {
                            "oof_proba": oof.astype(float) if oof is not None else None,
                            "auc": float("nan"),
                            "logloss": float("nan"),
                            "brier": float("nan"),
                            "score": float("-inf"),
                            "aborted": True,
                        }
    
            if oof is None:
                # still need oof for final metrics
                oof2 = np.full(n, np.nan, dtype=np.float32)
                for (tr_idx, val_idx) in folds:
                    _assert_no_game_overlap(tr_idx, val_idx)
                    p = np.asarray(_fit_predict_xgb(tr_idx, val_idx), float).ravel()
                    oof2[val_idx] = p.astype(np.float32, copy=False)
                oof = oof2
    
        # --------------------------------------
        # sklearn / generic path (still copies)
        # --------------------------------------
        else:
            def _fit_predict_proba_sklearn(tr_idx, val_idx):
                X_tr = X_local_mat[tr_idx, :]
                X_va = X_local_mat[val_idx, :]
                y_tr = y[tr_idx]
                mdl = clone(model_proto)
                mdl.fit(X_tr, y_tr)
                return _safe_predict_proba_pos(mdl, X_va)
    
            for (tr_idx, val_idx) in folds:
                _assert_no_game_overlap(tr_idx, val_idx)
    
                proba = np.asarray(_fit_predict_proba_sklearn(tr_idx, val_idx), float).ravel()
                if oof is not None:
                    oof[val_idx] = proba.astype(np.float32, copy=False)
    
                if do_abort:
                    try:
                        auc_fold = roc_auc_score(y[val_idx], np.clip(proba, eps, 1 - eps))
                        if not np.isfinite(auc_fold):
                            auc_fold = 0.5
                    except Exception:
                        auc_fold = 0.5
                    auc_fold_vals.append(float(auc_fold))
    
                    left = n_folds - len(auc_fold_vals)
                    mean_auc_best_possible = (sum(auc_fold_vals) + left * 1.0) / float(n_folds)
                    best_possible_score = (auc_weight * mean_auc_best_possible)
                    if best_possible_score < float(abort_if_cannot_beat_score) + float(abort_margin):
                        return {
                            "oof_proba": oof.astype(float) if oof is not None else None,
                            "auc": float("nan"),
                            "logloss": float("nan"),
                            "brier": float("nan"),
                            "score": float("-inf"),
                            "aborted": True,
                        }
    
            if oof is None:
                oof2 = np.full(n, np.nan, dtype=np.float32)
                for (tr_idx, val_idx) in folds:
                    _assert_no_game_overlap(tr_idx, val_idx)
                    p = np.asarray(_fit_predict_proba_sklearn(tr_idx, val_idx), float).ravel()
                    oof2[val_idx] = p.astype(np.float32, copy=False)
                oof = oof2
    
        # ---------------------------
        # metrics (vectorized already)
        # ---------------------------
        valid = np.isfinite(oof)
        y_v = y[valid]
        p_v = np.clip(oof[valid].astype(float), eps, 1 - eps)
    
        auc = roc_auc_score(y_v, p_v) if (len(y_v) >= 2 and len(np.unique(y_v)) >= 2) else np.nan
    
        if compute_ll_brier:
            ll = log_loss(y_v, p_v, labels=[0, 1])
            br = float(np.mean((p_v - y_v) ** 2))
        else:
            ll = float("nan")
            br = float("nan")
    
        score = (auc_weight * auc)
        if compute_ll_brier:
            score = score - (ll_weight * ll) - (brier_weight * br)
    
        return {
            "oof_proba": oof.astype(float) if return_oof else None,
            "auc": float(auc) if np.isfinite(auc) else float("nan"),
            "logloss": float(ll) if compute_ll_brier else float("nan"),
            "brier": float(br) if compute_ll_brier else float("nan"),
            "score": float(score) if np.isfinite(score) else float("-inf"),
            "aborted": False,
        }

    base = _oof_metrics_for_mat(X_mat)

    out = dict(base)
    out["feature_list"] = list(feats)
    out["oof_auc_best"] = float(out.get("auc", float("nan")))
    out["oof_logloss_best"] = float(out.get("logloss", float("nan")))
    out["oof_brier_best"] = float(out.get("brier", float("nan")))
    out["oof_score_best"] = float(out.get("score", float("-inf")))

    out["oof_auc"] = out["oof_auc_best"]
    out["oof_logloss"] = out["oof_logloss_best"]
    out["oof_brier"] = out["oof_brier_best"]
    out["oof_score"] = out["oof_score_best"]

    if not return_oof:
        out["oof_proba"] = None
    return out




import numpy as np
import pandas as pd
import heapq
from sklearn.base import clone
from sklearn.metrics import roc_auc_score



# =========================
# 2) AUTO SELECT BLOCK (evaluate-all first, then forward-select, NO quick gate)
#    + optional per-candidate flip test (reverse AUC) that can be accepted
# =========================
def _auto_select_k_by_auc(
    model_proto, X, y, folds, ordered_features, *,
    min_k=None,
    max_k=None,
    patience=100,
    min_improve=1e-6,
    verbose=True,
    log_func=print,
    debug=True,
    debug_every=20,

    max_ll_increase=0.20,
    max_brier_increase=0.06,

    orient_features=False,
    enable_feature_flips=False,
    max_feature_flips=0,
    orient_passes=1,

    force_full_scan=True,

    fallback_to_all_available_features=True,
    require_present_in_X=True,

    accept_metric: str = "auc",
    flips_after_selection: bool = True,

    allow_candidate_flip: bool = True,
    flip_gain_min: float = 0.0,

    must_keep: list[str] | None = None,
    baseline_feats: list[str] | None = None,

    # --- speed knobs (were "compat ignored" before, now used) ---
    quick_screen: bool = True,     # if True, do quick-fold gate
    quick_folds: int = 2,          # how many folds for quick gate
    quick_accept: float = 0.0,     # (unused but kept)
    quick_drop: float = 0.0,       # (unused but kept)
    abort_margin_cv: float = 0.0,  # used in early abort upper-bound

    time_budget_s: float = 1e18,
    resume_state: dict | None = None,
    max_total_evals: int | None = None,

    psi_fn=None,
    psi_max: float | None = None,
):
    """
    Faster + lower-memory forward selection.

    Speed wins:
      ✅ quick-fold gate (skip most candidates)
      ✅ XGB early stopping during selection
      ✅ AUC-only scan (LL/Brier only when candidate is about to be accepted)
      ✅ flip test only when normal is close / promising
      ✅ no X.copy(), no repeated dataframe slicing (matrix-backed work buffer)

    Keeps original behavior for:
      - ALL-FEATS CV print
      - baseline check print
      - final orient/feature flip pass via _cv_auc_for_feature_set
    """
    import time
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score, log_loss

    t0 = time.time()
    folds = list(folds) if folds is not None else []
    y_arr = np.asarray(y, int).reshape(-1)
    n = len(y_arr)

    # ---- sanitize ordered_features ----
    ordered = list(ordered_features) if ordered_features is not None else []
    if require_present_in_X:
        ordered = [f for f in ordered if f in X.columns]
    if (not ordered) and fallback_to_all_available_features:
        ordered = list(X.columns)
    if not ordered:
        return [], None, {"done": True, "reason": "no_candidates"}

    must_keep = must_keep or []
    mk = [m for m in must_keep if m in X.columns]
    ordered = list(dict.fromkeys(mk + ordered))

    if max_k is None:
        max_k = len(ordered)
    max_k = int(min(max_k, len(ordered)))

    if min_k is None:
        min_k = len(mk)
    min_k = int(max(len(mk), min(int(min_k), max_k)))

    if verbose:
        log_func(f"[AUTO-FEAT] seed(min_k)={min_k} must_keep={len(mk)} max_k={max_k} candidates={len(ordered)}")

    # ---------- evaluate ALL first (unchanged) ----------
    all_feats = list(ordered)
    all_res = _cv_auc_for_feature_set(
        model_proto, X, y_arr, folds, all_feats,
        log_func=log_func,
        debug=False,
        return_oof=False,
        orient_features=False,
        enable_feature_flips=False,
        cv_mode="full",
    )
    if verbose and isinstance(all_res, dict):
        log_func(
            f"[ALL-FEATS CV] auc={all_res.get('auc', float('nan')):.4f} "
            f"ll={all_res.get('logloss', float('nan')):.4f} "
            f"brier={all_res.get('brier', float('nan')):.4f} "
            f"score={all_res.get('score', float('nan')):.4f} "
            f"n_feats={len(all_feats)}"
        )

    # ---------- baseline check (unchanged) ----------
    if baseline_feats is None:
        baseline_feats = mk
    base_feats = [f for f in (baseline_feats or []) if f in X.columns]
    if base_feats:
        base_res = _cv_auc_for_feature_set(
            model_proto, X, y_arr, folds, base_feats,
            log_func=log_func,
            debug=False,
            return_oof=False,
            cv_mode="full",
        )
        if verbose and isinstance(base_res, dict):
            lift_auc = float(all_res.get("auc", 0.5)) - float(base_res.get("auc", 0.5))
            lift_ll  = float(base_res.get("logloss", float("nan"))) - float(all_res.get("logloss", float("nan")))
            log_func(
                f"[SIGNAL CHECK] baseline_auc={base_res.get('auc', float('nan')):.4f} "
                f"all_auc={all_res.get('auc', float('nan')):.4f} "
                f"Δauc={lift_auc:+.4f} | "
                f"baseline_ll={base_res.get('logloss', float('nan')):.4f} "
                f"all_ll={all_res.get('logloss', float('nan')):.4f} "
                f"Δll={lift_ll:+.4f}"
            )

    # ---------- helpers ----------
    def _metric(res):
        if res is None:
            return float("-inf") if accept_metric == "score" else float("nan")
        if accept_metric == "score":
            return float(res.get("score", float("-inf")))
        return float(res.get("auc", float("nan")))

    def _accept_ok(val_try, best_val, ll_try, best_ll, br_try, best_br):
        ll_ok = (not np.isfinite(ll_try)) or (not np.isfinite(best_ll)) or (ll_try <= float(best_ll) + float(max_ll_increase))
        br_ok = (not np.isfinite(br_try)) or (not np.isfinite(best_br)) or (br_try <= float(best_br) + float(max_brier_increase))
        if not (ll_ok and br_ok):
            return False
        if not np.isfinite(val_try):
            return False
        if not np.isfinite(best_val):
            return True
        return bool(val_try >= float(best_val) + float(min_improve))

    # ---- fold arrays once ----
    folds_np = [(np.asarray(tr, np.int64), np.asarray(va, np.int64)) for (tr, va) in (folds or [])]
    folds = folds_np

    # quick folds subset
    qn = int(max(1, quick_folds)) if (quick_screen and len(folds) > 1) else len(folds)
    folds_quick = folds[:qn]
    folds_full = folds

    # ---- numeric matrix once (float32) ----
    X_num_all = X.apply(pd.to_numeric, errors="coerce")
    X_all_mat = X_num_all.to_numpy(dtype=np.float32, copy=False)
    col_ix = {c: i for i, c in enumerate(X.columns)}

    # ---- XGB config ----
    use_xgb = False
    try:
        use_xgb = bool(_is_xgb_classifier(model_proto))
    except Exception:
        use_xgb = False

    xgb_params = None
    xgb_num_round = None
    if use_xgb:
        import xgboost as xgb  # noqa
        xgb_params, xgb_num_round = _xgb_params_from_proto(model_proto)
        if isinstance(xgb_params, dict) and "nthread" not in xgb_params and "n_jobs" not in xgb_params:
            xgb_params["nthread"] = 0
    # ✅ Slightly increase regularization (Option A)
  

    # ---- fast flip helpers (mode-only + inplace) ----
    def _auto_flip_mode(arr1d):
        arr1d = np.asarray(arr1d, dtype=np.float32)
        if not np.isfinite(arr1d).any():
            return "none"
        x_nan = np.where(np.isfinite(arr1d), arr1d, np.nan)
        mn = np.nanmin(x_nan)
        mx = np.nanmax(x_nan)
        return "one_minus" if (mn >= -1e-3 and mx <= 1.0 + 1e-3) else "negate"

    def _apply_flip_inplace(col, mode: str):
        if mode == "one_minus":
            np.subtract(np.float32(1.0), col, out=col)  # col = 1 - col
        elif mode == "negate":
            np.negative(col, out=col)                   # col = -col

    # cache flip mode once per feature (tiny win, free)
    flip_mode_cache = {}
    if allow_candidate_flip:
        for f in ordered:
            if f in col_ix:
                flip_mode_cache[f] = _auto_flip_mode(X_all_mat[:, col_ix[f]])

    # ---- matrix-backed CV evaluator ----
    def _cv_eval_auc_only(X_mat, folds_use, *, abort_best_auc=None, early_stop_rounds=25):
        """
        Returns dict with auc (no ll/brier).
        Adds early stopping for XGB during selection.
        """
        eps = 1e-6
        oof = np.full(n, np.nan, dtype=np.float32)

        do_abort = (abort_best_auc is not None) and np.isfinite(abort_best_auc)
        auc_fold_vals = []
        n_folds = len(folds_use)

        if use_xgb:
            import xgboost as xgb
            # build once per eval (still cheaper than pandas slicing; X_mat is contiguous slice)
            d_all = xgb.DMatrix(np.ascontiguousarray(X_mat, dtype=np.float32), label=y_arr)

            def _fit_predict(tr_idx, va_idx):
                d_tr = d_all.slice(tr_idx)
                d_va = d_all.slice(va_idx)
                booster = xgb.train(
                    xgb_params,
                    d_tr,
                    num_boost_round=int(xgb_num_round),
                    evals=[(d_va, "val")],
                    early_stopping_rounds=int(early_stop_rounds),
                    verbose_eval=False,
                )
                return booster.predict(d_va)
        else:
            def _fit_predict(tr_idx, va_idx):
                mdl = clone(model_proto)
                mdl.fit(X_mat[tr_idx, :], y_arr[tr_idx])
                p = mdl.predict_proba(X_mat[va_idx, :])
                return p[:, 1] if p.ndim == 2 else p

        for tr_idx, va_idx in folds_use:
            proba = np.asarray(_fit_predict(tr_idx, va_idx), float).ravel()
            oof[va_idx] = proba.astype(np.float32, copy=False)

            if do_abort:
                try:
                    auc_fold = roc_auc_score(y_arr[va_idx], np.clip(proba, eps, 1 - eps))
                    if not np.isfinite(auc_fold):
                        auc_fold = 0.5
                except Exception:
                    auc_fold = 0.5
                auc_fold_vals.append(float(auc_fold))

                left = n_folds - len(auc_fold_vals)
                best_possible_auc = (sum(auc_fold_vals) + left * 1.0) / float(n_folds)
                if best_possible_auc < float(abort_best_auc) + float(abort_margin_cv):
                    return {"auc": float("nan"), "aborted": True}

        valid = np.isfinite(oof)
        y_v = y_arr[valid]
        p_v = np.clip(oof[valid].astype(float), eps, 1 - eps)
        auc = roc_auc_score(y_v, p_v) if (len(y_v) >= 2 and len(np.unique(y_v)) >= 2) else np.nan
        return {"auc": float(auc) if np.isfinite(auc) else float("nan"), "aborted": False}

    def _cv_eval_full_metrics(X_mat, folds_use, *, early_stop_rounds=25):
        """
        Full metrics (AUC + LL + Brier) — call only when a candidate is about to be accepted.
        """
        eps = 1e-6
        oof = np.full(n, np.nan, dtype=np.float32)

        if use_xgb:
            import xgboost as xgb
            d_all = xgb.DMatrix(np.ascontiguousarray(X_mat, dtype=np.float32), label=y_arr)

            def _fit_predict(tr_idx, va_idx):
                d_tr = d_all.slice(tr_idx)
                d_va = d_all.slice(va_idx)
                booster = xgb.train(
                    xgb_params,
                    d_tr,
                    num_boost_round=int(xgb_num_round),
                    evals=[(d_va, "val")],
                    early_stopping_rounds=int(early_stop_rounds),
                    verbose_eval=False,
                )
                return booster.predict(d_va)
        else:
            def _fit_predict(tr_idx, va_idx):
                mdl = clone(model_proto)
                mdl.fit(X_mat[tr_idx, :], y_arr[tr_idx])
                p = mdl.predict_proba(X_mat[va_idx, :])
                return p[:, 1] if p.ndim == 2 else p

        for tr_idx, va_idx in folds_use:
            proba = np.asarray(_fit_predict(tr_idx, va_idx), float).ravel()
            oof[va_idx] = proba.astype(np.float32, copy=False)

        valid = np.isfinite(oof)
        y_v = y_arr[valid]
        p_v = np.clip(oof[valid].astype(float), eps, 1 - eps)

        auc = roc_auc_score(y_v, p_v) if (len(y_v) >= 2 and len(np.unique(y_v)) >= 2) else np.nan
        ll = log_loss(y_v, p_v, labels=[0, 1])
        br = float(np.mean((p_v - y_v) ** 2))

        # match your _cv_auc_for_feature_set "score" weights
        score = float(auc) - (0.15 * float(ll)) - (0.10 * float(br))
        return {"auc": float(auc), "logloss": float(ll), "brier": float(br), "score": float(score), "aborted": False}

    # ---- selection state ----
    accepted = list(mk)
    accepted_set = set(accepted)
    flip_map = {}

    # Work matrix: allocate once and only grow when accepted
    X_work = np.empty((n, max_k), dtype=np.float32)
    k = 0

    def _put_feat_into_col(feat: str, j: int):
        X_work[:, j] = X_all_mat[:, col_ix[feat]]

    for f in accepted:
        _put_feat_into_col(f, k)
        k += 1

    # initial best (full metrics once)
    if k > 0:
        best_res = _cv_eval_full_metrics(X_work[:, :k], folds_full)
    else:
        best_res = None

    best_val = _metric(best_res)
    best_ll  = float(best_res.get("logloss", np.nan)) if best_res is not None else float("nan")
    best_br  = float(best_res.get("brier", np.nan)) if best_res is not None else float("nan")

    rejects_in_row = 0
    if verbose:
        log_func(f"[AUTO-FEAT] start scan: must_keep={len(mk)} target_seed={min_k} metric={accept_metric} quick_folds={len(folds_quick)}")

    # --- gating margins (tune if you want) ---
    quick_margin_auc = 0.0005   # if quick AUC is worse than best-0.002, skip full CV
    flip_close_margin = 0.0005  # only try flip if normal is at least (best - 0.001) in quick/full

    for i, feat in enumerate(ordered):
        if (time.time() - t0) >= float(time_budget_s):
            break
        if feat in accepted_set:
            continue
        if max_k is not None and len(accepted) >= int(max_k):
            break
        if (not force_full_scan) and (len(accepted) >= min_k) and (rejects_in_row >= int(patience)):
            break

        # overwrite candidate into next free column
        _put_feat_into_col(feat, k)

        # -----------------------------
        # 1) QUICK gate (AUC-only)
        # -----------------------------
        res_q = None
        if accept_metric == "auc" and quick_screen and len(folds_quick) < len(folds_full):
            res_q = _cv_eval_auc_only(
                X_work[:, :k+1], folds_quick,
                abort_best_auc=(best_val if np.isfinite(best_val) else None),
                early_stop_rounds=15,
            )
            m_q = float(res_q.get("auc", np.nan))
            if np.isfinite(best_val) and (not np.isfinite(m_q) or (m_q < float(best_val) - float(quick_margin_auc))):
                rejects_in_row += 1
                continue

        # -----------------------------
        # 2) FULL AUC-only (still cheap)
        # -----------------------------
        if accept_metric == "auc":
            res_norm_auc = _cv_eval_auc_only(
                X_work[:, :k+1], folds_full,
                abort_best_auc=(best_val if np.isfinite(best_val) else None),
                early_stop_rounds=25,
            )
            m_norm = float(res_norm_auc.get("auc", np.nan))
            if (not np.isfinite(m_norm)) or (np.isfinite(best_val) and (m_norm < float(best_val) - 0.0015)):
                rejects_in_row += 1
                continue
        else:
            # score-mode: must compute full metrics
            res_norm_full = _cv_eval_full_metrics(X_work[:, :k+1], folds_full, early_stop_rounds=25)
            m_norm = _metric(res_norm_full)

        best_trial_flip = False
        best_trial_flip_mode = None
        best_trial_auc = float(m_norm)

        # -----------------------------
        # 3) Flip test only if close/promising
        # -----------------------------
        if allow_candidate_flip and accept_metric == "auc" and np.isfinite(m_norm):
            if (not np.isfinite(best_val)) or (m_norm >= float(best_val) - float(flip_close_margin)):
                col = X_work[:, k]
                orig = col.copy()

                fm = flip_mode_cache.get(feat) or _auto_flip_mode(orig)
                _apply_flip_inplace(col, fm)

                # quick flip gate (optional)
                ok_to_full_flip = True
                if quick_screen and len(folds_quick) < len(folds_full):
                    res_fq = _cv_eval_auc_only(X_work[:, :k+1], folds_quick, abort_best_auc=None, early_stop_rounds=15)
                    mfq = float(res_fq.get("auc", np.nan))
                    ok_to_full_flip = np.isfinite(mfq) and (mfq >= m_norm + float(flip_gain_min))

                if ok_to_full_flip:
                    res_flip_auc = _cv_eval_auc_only(X_work[:, :k+1], folds_full, abort_best_auc=None, early_stop_rounds=25)
                    m_flip = float(res_flip_auc.get("auc", np.nan))
                    if np.isfinite(m_flip) and (m_flip > m_norm + float(flip_gain_min)):
                        best_trial_flip = True
                        best_trial_flip_mode = fm
                        best_trial_auc = float(m_flip)

                # restore
                col[:] = orig

        # -----------------------------
        # 4) If AUC suggests "maybe accept", compute full metrics ONCE
        # -----------------------------
        if accept_metric == "auc":
            # AUC must clear improvement threshold before paying LL/Brier cost
            if np.isfinite(best_trial_auc) and (not np.isfinite(best_val) or (best_trial_auc >= float(best_val) + float(min_improve))):
                # if flip won, apply flip again temporarily to compute full metrics
                col = X_work[:, k]
                orig = col.copy()
                if best_trial_flip:
                    _apply_flip_inplace(col, best_trial_flip_mode)

                res_try = _cv_eval_full_metrics(X_work[:, :k+1], folds_full, early_stop_rounds=25)

                # restore
                col[:] = orig
            else:
                rejects_in_row += 1
                continue
        else:
            # score-mode already computed
            res_try = res_norm_full
            best_trial_auc = float(res_try.get("auc", np.nan))

        val_try = _metric(res_try)
        ll_try  = float(res_try.get("logloss", np.nan))
        br_try  = float(res_try.get("brier", np.nan))

        if _accept_ok(val_try, best_val, ll_try, best_ll, br_try, best_br):
            accepted.append(feat)
            accepted_set.add(feat)
            k += 1  # commit the column

            best_res = res_try
            best_val = float(val_try)
            if np.isfinite(ll_try): best_ll = float(ll_try)
            if np.isfinite(br_try): best_br = float(br_try)
            rejects_in_row = 0

            if best_trial_flip:
                flip_map[feat] = {"flipped": True, "mode": best_trial_flip_mode}
                if verbose:
                    log_func(f"[AUTO-FEAT] 🔁 flip accepted for {feat} mode={best_trial_flip_mode}")

            if verbose:
                tag = " (flipped)" if best_trial_flip else ""
                log_func(f"[AUTO-FEAT] ✅ accept +{feat}{tag} -> {accept_metric}={best_val:.6f} k={len(accepted)}")
        else:
            rejects_in_row += 1

        if debug and (i % int(max(1, debug_every)) == 0):
            log_func(f"[AUTO-FEAT][DBG] i={i} k={len(accepted)} best_{accept_metric}={best_val:.6f}")

    # ---------- final orient pass (unchanged) ----------
    if flips_after_selection and accepted:
        best_res = _cv_auc_for_feature_set(
            model_proto, X, y_arr, folds_full, list(accepted),
            log_func=log_func,
            debug=False,
            return_oof=True,
            orient_features=bool(orient_features),
            enable_feature_flips=bool(enable_feature_flips),
            max_feature_flips=int(max_feature_flips),
            orient_passes=int(orient_passes),
            cv_mode="full",
        )
        if verbose and isinstance(best_res, dict):
            log_func(f"[AUTO-FEAT] Final accepted set: k={len(accepted)} {accept_metric}={_metric(best_res):.6f}")

    state = {"done": True, "accepted": list(accepted), "best_res": best_res, "flip_map": flip_map}
    return list(accepted), best_res, state

# =========================
# 3) SELECT FEATURES AUTO (backward compatible signature)
#    - seeds with must_keep only (earned thereafter)
#    - evaluates ALL features + baseline fade check (inside _auto_select_k_by_auc)
#    - supports reverse-AUC candidate flips and returns flip_map in summary
# =========================
def select_features_auto(
    model_proto,
    X_df_train: "pd.DataFrame",
    y_train: "np.ndarray",
    folds,
    *,
    sport_key: str = "NFL",
    must_keep: list[str] | None = None,

    # keep old knobs but they won't break your call
    corr_within: float = 0.90,
    corr_global: float = 0.92,
    max_feats_major: int = 160,
    max_feats_small: int = 160,
    topk_per_fold: int = 80,
    min_presence: float = 0.40,
    sign_flip_max: float = 0.35,
    shap_cv_max: float = 1.00,

    # selection knobs
    use_auc_auto: bool = True,
    auc_min_k: int | None = None,
    auc_patience: int = 100,
    auc_min_improve: float = 5e-6,
    accept_metric: str = "auc",
    auc_verbose: bool = True,

    # fade/baseline check features
    baseline_feats: list[str] | None = None,

    # reverse-AUC / candidate flip knobs
    allow_candidate_flip: bool = True,
    flip_gain_min: float = 0.0,

    final_orient: bool = True,
    log_func=print,

    # ✅ backward compat: swallow extra kwargs safely
    **_ignored_kwargs,
):
 

    if X_df_train is None or X_df_train.empty:
        return [], pd.DataFrame(columns=["selected", "flipped", "flip_mode"])

    # ✅ Default must_keep
    must_keep = must_keep or []

    y_arr = np.asarray(y_train, dtype=int).reshape(-1)

    # ----------------------------
    # 1) Candidate prefilter (big win)
    # ----------------------------
    # Keep must_keep first, but don't blindly include ALL columns.
    mk_in = [m for m in must_keep if m in X_df_train.columns]

    # Lightweight leak/forbidden guard here too (optional but cheap)
    # (You already have a leak guard in _cv_auc_for_feature_set; this just shrinks candidates early)
    leak_patterns = ("SHARP_COVER_RESULT",)
    def _is_forbidden(c: str) -> bool:
        s = str(c)
        return any(p in s for p in leak_patterns)

    # Start from non-forbidden cols
    cols = [c for c in X_df_train.columns if (c in mk_in) or (not _is_forbidden(c))]

    # Convert to numeric ONCE (float32). This is the only “big” operation here,
    # but it avoids tons of repeated pd.to_numeric work later.
    X_num = X_df_train[cols].apply(pd.to_numeric, errors="coerce")
    X_mat = X_num.to_numpy(dtype=np.float32, copy=False)

    # Compute non-nan fraction (vectorized)
    finite = np.isfinite(X_mat)
    nn_frac = finite.mean(axis=0)

    # Compute “usable” mask:
    # - enough non-nan
    # - not constant (variance > 0, nan-safe)
    # NOTE: nanvar is fine here because we already cast to float32.
    with np.errstate(invalid="ignore"):
        var = np.nanvar(X_mat, axis=0)

    # Tune these if you want; these are safe defaults
    min_non_nan_frac = 0.001
    usable = (nn_frac >= float(min_non_nan_frac)) & (var > 0.0)

    # Force must_keep to be usable if present (even if constant) so earned logic can decide later
    if mk_in:
        mk_idx = [i for i, c in enumerate(cols) if c in set(mk_in)]
        usable[np.asarray(mk_idx, dtype=int)] = True

    candidate_cols = [c for c, ok in zip(cols, usable) if bool(ok)]

    # Rebuild keep_order: must_keep first, then usable candidates (no duplicates)
    keep_order = list(dict.fromkeys(mk_in + candidate_cols))

    if not keep_order:
        return [], pd.DataFrame(columns=["selected", "flipped", "flip_mode"])

    # min_k should NEVER exceed earned logic; default is exactly must_keep length
    min_k_eff = len(mk_in) if (auc_min_k is None) else int(max(len(mk_in), auc_min_k))

    flip_map = {}

    # ----------------------------
    # 2) Auto-select (your optimized _auto_select_k_by_auc does the heavy lifting)
    # ----------------------------
    if use_auc_auto and keep_order:
        accepted_feats, best_res, state = _auto_select_k_by_auc(
            model_proto, X_df_train, y_arr, folds, keep_order,
            min_k=min_k_eff,
            max_k=len(keep_order),
            patience=int(auc_patience),
            min_improve=float(auc_min_improve),
            verbose=bool(auc_verbose),
            log_func=log_func,
            accept_metric=str(accept_metric),
            must_keep=mk_in,
            baseline_feats=baseline_feats if baseline_feats is not None else mk_in,

            allow_candidate_flip=bool(allow_candidate_flip),
            flip_gain_min=float(flip_gain_min),

            flips_after_selection=bool(final_orient),
            orient_features=True if final_orient else False,
            enable_feature_flips=True if final_orient else False,
            max_feature_flips=10,
            orient_passes=1,

            quick_screen=False,
            abort_margin_cv=0.0,
            force_full_scan=True,
            time_budget_s=1e18,
            resume_state=None,
            max_total_evals=None,
        )
        feature_cols = list(accepted_feats or [])
        flip_map = (state or {}).get("flip_map", {}) or {}
    else:
        feature_cols = keep_order

    # ----------------------------
    # 3) Summary (vectorized, no Python loops)
    # ----------------------------
    # Build arrays once
    f_arr = np.asarray(feature_cols, dtype=object)

    flipped = np.fromiter(
        (bool(flip_map.get(f, {}).get("flipped", False)) for f in feature_cols),
        dtype=bool,
        count=len(feature_cols),
    )
    flip_mode = np.fromiter(
        ((flip_map.get(f, {}).get("mode", "") or "") for f in feature_cols),
        dtype=object,
        count=len(feature_cols),
    )

    summary = pd.DataFrame(
        {
            "selected": np.ones(len(feature_cols), dtype=bool),
            "flipped": flipped,
            "flip_mode": flip_mode,
        },
        index=f_arr,
    )
    summary.index.name = "feature"

    # Optional: log candidate shrink (nice sanity check)
    if auc_verbose:
        log_func(f"[AUTO-FEAT] candidates after prefilter: {len(keep_order)} (from {X_df_train.shape[1]})")

    return feature_cols, summary




from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np, pandas as pd

from sklearn.metrics import roc_auc_score
import numpy as np, pandas as pd

def perm_auc_importance_ci(
    model, X_df: pd.DataFrame, y: np.ndarray, *,
    n_repeats: int = 20,
    random_state: int = 42,
    stratify: bool = True
):
    """
    Returns:
      base_auc: float
      df: DataFrame indexed by feature with columns:
          perm_auc_drop_mean, perm_auc_drop_std, ci_lo, ci_hi
    CI uses a normal approximation over repeated permutations.
    If stratify=True, permutes each feature within class labels to preserve class balance.
    """
    rng = np.random.RandomState(random_state)
    proba = model.predict_proba(X_df)[:, 1]
    base_auc = roc_auc_score(y, proba)

    drops = []
    cols = list(X_df.columns)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    for col in cols:
        vals = []
        c = X_df.columns.get_loc(col)
        for _ in range(int(n_repeats)):
            Xp = X_df.copy()
            if stratify:
                Xp.iloc[idx0, c] = Xp.iloc[idx0, c].sample(
                    frac=1.0, replace=False, random_state=rng.randint(1_000_000_000)
                ).values
                Xp.iloc[idx1, c] = Xp.iloc[idx1, c].sample(
                    frac=1.0, replace=False, random_state=rng.randint(1_000_000_000)
                ).values
            else:
                Xp.iloc[:, c] = Xp.iloc[:, c].sample(
                    frac=1.0, replace=False, random_state=rng.randint(1_000_000_000)
                ).values

            vals.append(base_auc - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))

        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        half = 1.96 * sd / np.sqrt(n_repeats) if n_repeats > 1 else 0.0
        drops.append((col, mu, sd, mu - half, mu + half))

    df = (pd.DataFrame(drops, columns=["feature","perm_auc_drop_mean","perm_auc_drop_std","ci_lo","ci_hi"])
          .set_index("feature"))
    return base_auc, df



def _to_float_1d(x):
    
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.values
    x = np.asarray(x).astype(np.float64)
    return x.ravel()

def _clean_probs(y, *probs, eps=1e-6):
    """Cast to float64, drop non-finite entries across all arrays, then clip."""
   
    arrs = [ _to_float_1d(y) ] + [ _to_float_1d(p) for p in probs ]
    mask = np.isfinite(arrs[0])
    for a in arrs[1:]:
        mask &= np.isfinite(a)
    cleaned = [ a[mask] for a in arrs ]
    # clip probs (skip y at index 0)
    for i in range(1, len(cleaned)):
        cleaned[i] = np.clip(cleaned[i], eps, 1 - eps)
    return cleaned, mask  # ([y, p1, p2, ...], keep_mask)

# ---------------------------
# WHY: all-features + metadata (V3, corrected)
# ---------------------------
from collections import defaultdict
import numpy as np
import pandas as pd

# --- robust value getter ---
def _rv(row, *names, default=0.0):
    for n in names:
        if n in row and pd.notna(row[n]):
            try:
                return float(row[n])
            except Exception:
                return default
    return default

# --- thresholds (as you had) ---
THR = dict(
    line_mag_big=0.01,
    late_share_high=0.50,
    urgent_share_high=0.20,
    entropy_concentrated=1.20,
    corr_confirm=0.35,
    odds_overmove_ratio=1.10,
    pct_from_open_big=0.10,
    pr_diff_meaningful=4.0,
    cover_prob_conf=0.51,
    ats_rate_strong=0.55,
    ats_margin_meaningful=2,
    ats_roll_decay_hot=0.4,
)

# --- discover ALL usable features (numeric/bool or numeric-coercible) ---
_EXCLUDE_KEYS = {"y","label","target"}
def _is_usable_feature(s: pd.Series) -> bool:
    dt = s.dtype
    if pd.api.types.is_bool_dtype(dt) or pd.api.types.is_integer_dtype(dt) or pd.api.types.is_float_dtype(dt):
        return True
    if dt == object:
        try:
            return pd.to_numeric(s, errors="coerce").notna().any()
        except Exception:
            return False
    return False

def _resolve_active_features_union(bundle, model, df_like: pd.DataFrame, why_rules) -> list[str]:
    """Bundle + booster + helper + why_rules refs + ALL usable df_like columns (order-preserving)."""
    def _norm(xs):
        return [str(c) for c in xs if c is not None]

    out = []
    # 1) bundle-declared
    if isinstance(bundle, dict):
        for k in ("feature_names", "features", "training_features"):
            if k in bundle and bundle[k]:
                out += _norm(bundle[k])
                break
    # 2) model booster feature_names
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            if booster is not None and getattr(booster, "feature_names", None):
                out += _norm(booster.feature_names)
        except Exception:
            pass
    # 3) optional helper
    try:
        cols = _resolve_feature_cols_like_training(bundle, model, df_like)  # if defined elsewhere
        if cols:
            out += _norm(cols)
    except Exception:
        pass
    # 4) all rule-referenced columns (from the provided rules set)
    rule_cols = []
    for rule in why_rules:
        rule_cols += rule.get("requires", [])
        rule_cols += rule.get("requires_any", [])
    out += _norm(rule_cols)
    # 5) ALL usable df_like columns
    all_usable = [c for c in df_like.columns if c not in _EXCLUDE_KEYS and _is_usable_feature(df_like[c])]
    out += _norm(all_usable)

    # dedupe preserve order
    seen, ordered = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered

# --- numeric coercion for rule inputs (single, non-duplicated version) ---
def _coerce_numeric_inplace(df: pd.DataFrame, cols, fill=0.0, ui_cols=frozenset()):
    for c in cols:
        if c not in df.columns:
            df[c] = (np.nan if c in ui_cols else fill)
            continue
        s = df[c]
        if pd.api.types.is_categorical_dtype(s.dtype):
            s = s.astype(object)
        vals = pd.to_numeric(s, errors="coerce")
        df[c] = vals if c in ui_cols else vals.fillna(fill)

# ---------- WHY_RULES_V3 ----------
def build_hybrid_bin_rules():
    phases = ["Overnight", "Early", "Midday", "Late"]
    urg = ["VeryEarly", "MidRange", "LateGame", "Urgent"]
    rules = []
    for p in phases:
        for u in urg:
            col1 = f"SharpMove_Magnitude_{p}_{u}"
            col2 = f"OddsMove_Magnitude_{p}_{u}"
            rules.append(dict(requires_any=[col1], check=lambda r, c=col1: _rv(r, c) > 0.0,
                              msg=f"💥 Sharp Timing Spike ({p}/{u})"))
            rules.append(dict(requires_any=[col2], check=lambda r, c=col2: _rv(r, c) > 0.0,
                              msg=f"💥 Odds Timing Spike ({p}/{u})"))
    return rules

# ---------- WHY_RULES_V3 ----------
def build_hybrid_bin_rules():
    phases = ["Overnight", "Early", "Midday", "Late"]
    urg = ["VeryEarly", "MidRange", "LateGame", "Urgent"]
    rules = []
    for p in phases:
        for u in urg:
            col1 = f"SharpMove_Magnitude_{p}_{u}"
            col2 = f"OddsMove_Magnitude_{p}_{u}"
            rules.append(dict(requires_any=[col1], check=lambda r, c=col1: _rv(r, c) > 0.0,
                              msg=f"💥 Sharp Timing Spike ({p}/{u})"))
            rules.append(dict(requires_any=[col2], check=lambda r, c=col2: _rv(r, c) > 0.0,
                              msg=f"💥 Odds Timing Spike ({p}/{u})"))
    return rules

WHY_RULES_V3 = [
    # Sharp / book pressure
    dict(requires_any=["Book_Reliability_Lift"],
         check=lambda r: _rv(r,"Book_Reliability_Lift") > 0.0,
         msg="✅ Reliable Book Confirms"),
    dict(requires_any=["Is_Sharp_Book","Sharp_Move_Signal"],
         check=lambda r: (_rv(r,"Is_Sharp_Book") > 0.0) and (_rv(r,"Sharp_Move_Signal") > 0.0),
         msg="🎯 Sharp Book Triggered Move"),
    dict(requires_any=["Sharp_Limit_Total"],
         check=lambda r: _rv(r,"Sharp_Limit_Total") >= 5000,
         msg="💼 High Sharp Limits"),
    dict(requires_any=["Book_lift_x_Magnitude"],
         check=lambda r: _rv(r,"Book_lift_x_Magnitude") > 0.0,
         msg="🏦 Book Lift Supports Move"),
    dict(requires_any=["Book_lift_x_PROB_SHIFT"],
         check=lambda r: _rv(r,"Book_lift_x_PROB_SHIFT") > 0.0,
         msg="📈 Book Lift Aligned with Prob Shift"),
    dict(requires_any=["Is_Reinforced_MultiMarket"],
         check=lambda r: _rv(r,"Is_Reinforced_MultiMarket") > 0.0,
         msg="📊 Multi‑Market Reinforcement"),
    dict(requires_any=["Market_Leader"],
         check=lambda r: _rv(r,"Market_Leader") > 0.0,
         msg="🏆 Market Leader Led the Move"),
    dict(requires_any=["High_Limit_Flag"],
         check=lambda r: _rv(r,"High_Limit_Flag") > 0.0,
         msg="🧱 High Limit Context"),

    # Market response / mispricing
    dict(requires_any=["Line_Moved_Toward_Team"],
         check=lambda r: _rv(r,"Line_Moved_Toward_Team") > 0.0,
         msg="🧲 Line Moved Toward This Side"),
    dict(requires_any=["Market_Mispricing","Abs_Market_Mispricing"],
         check=lambda r: max(_rv(r,"Market_Mispricing"), _rv(r,"Abs_Market_Mispricing")) > 0.05,
         msg="💸 Market Mispricing Detected"),
    dict(requires_any=["Pct_Line_Move_From_Opening"],
         check=lambda r: _rv(r,"Pct_Line_Move_From_Opening") >= THR["pct_from_open_big"],
         msg="📈 Significant Move From Open"),

    # Reversal / overmove
    dict(requires_any=["Value_Reversal_Flag"],
         check=lambda r: _rv(r,"Value_Reversal_Flag") > 0.0,
         msg="🔄 Value Reversal"),
    dict(requires_any=["Odds_Reversal_Flag"],
         check=lambda r: _rv(r,"Odds_Reversal_Flag") > 0.0,
         msg="📉 Odds Reversal"),
    dict(requires_any=["Potential_Overmove_Flag","Abs_Line_Move_Z"],
         check=lambda r: (_rv(r,"Potential_Overmove_Flag") > 0.0) or (_rv(r,"Abs_Line_Move_Z") >= 2.0),
         msg="📊 Possible Line Overmove"),
    dict(requires_any=["Potential_Odds_Overmove_Flag","Implied_Prob_Shift_Z"],
         check=lambda r: (_rv(r,"Potential_Odds_Overmove_Flag") > 0.0) or (_rv(r,"Implied_Prob_Shift_Z") >= 2.0),
         msg="🎯 Possible Odds Overmove"),

    # Resistance / levels
    dict(requires_any=["Line_Resistance_Crossed_Count"],
         check=lambda r: _rv(r,"Line_Resistance_Crossed_Count") >= 1,
         msg="🪵 Crossed Key Resistance Levels"),
    dict(requires_any=["SharpMove_Resistance_Break","Was_Line_Resistance_Broken"],
         check=lambda r: (_rv(r,"SharpMove_Resistance_Break") > 0.0) or (_rv(r,"Was_Line_Resistance_Broken") > 0.0),
         msg="🪓 Resistance Broken by Sharp Move"),

    # Timing aggregates (hybrid)
    dict(requires_any=["Line_TotalMag","Sharp_Line_Magnitude","Hybrid_Line_LateMag"],
         check=lambda r: max(_rv(r,"Line_TotalMag"), _rv(r,"Sharp_Line_Magnitude"), _rv(r,"Hybrid_Line_LateMag")) >= THR["line_mag_big"],
         msg="📏 Strong Timing Magnitude"),
    dict(requires_any=["Hybrid_Line_LateShare","Line_LateShare"],
         check=lambda r: max(_rv(r,"Hybrid_Line_LateShare"), _rv(r,"Line_LateShare")) >= THR["late_share_high"],
         msg="🌙 Late‑Phase Dominant"),
    dict(requires_any=["Hybrid_Line_EarlyShare","Hybrid_Line_Imbalance_LateVsEarly","Line_UrgentShare"],
         check=lambda r: (_rv(r,"Line_UrgentShare") >= THR["urgent_share_high"]) or (_rv(r,"Hybrid_Line_Imbalance_LateVsEarly") > 0.0),
         msg="⏱️ Urgent Push / Late Imbalance"),
    dict(requires_any=["Line_MaxBinMag"],
         check=lambda r: _rv(r,"Line_MaxBinMag") > 0.0,
         msg="💥 Sharp Timing Spike"),
    dict(requires_any=["Line_Entropy","Hybrid_Timing_Entropy_Line"],
         check=lambda r: 0.0 < _rv(r,"Line_Entropy","Hybrid_Timing_Entropy_Line") <= THR["entropy_concentrated"],
         msg="🎯 Concentrated Timing"),
    dict(requires_any=["Timing_Corr_Line_Odds"],
         check=lambda r: _rv(r,"Timing_Corr_Line_Odds") >= THR["corr_confirm"],
         msg="🔗 Odds Confirm Line Timing"),
    dict(requires_any=["LineOddsMag_Ratio","Hybrid_Line_Odds_Mag_Ratio"],
         check=lambda r: max(_rv(r,"LineOddsMag_Ratio"), _rv(r,"Hybrid_Line_Odds_Mag_Ratio")) >= THR["odds_overmove_ratio"],
         msg="⚖️ Line > Odds Magnitude"),
    dict(requires_any=["Late_Game_Steam_Flag"],
         check=lambda r: _rv(r,"Late_Game_Steam_Flag") > 0.0,
         msg="🌙 Late Game Steam"),

    # Cross‑market alignment / gaps
    dict(requires_any=["CrossMarket_Prob_Gap_Exists","Spread_vs_H2H_ProbGap","Total_vs_Spread_ProbGap"],
         check=lambda r: (_rv(r,"CrossMarket_Prob_Gap_Exists") > 0.0) or
                         (abs(_rv(r,"Spread_vs_H2H_ProbGap")) > 0.05) or
                         (abs(_rv(r,"Total_vs_Spread_ProbGap")) > 0.05),
         msg="🧮 Cross‑Market Probability Dislocation"),
    dict(requires_any=["Total_vs_Spread_Contradiction"],
         check=lambda r: _rv(r,"Total_vs_Spread_Contradiction") > 0.0,
         msg="⚠️ Totals vs Spread Contradiction"),

    # PR / model / market agreement
    dict(requires_any=["model_fav_vs_market_fav_agree"],
         check=lambda r: _rv(r,"model_fav_vs_market_fav_agree") > 0.0,
         msg="🧭 Model & Market Agree"),
    dict(requires_any=["Outcome_Cover_Prob"],
         check=lambda r: _rv(r,"Outcome_Cover_Prob") >= THR["cover_prob_conf"],
         msg="🔮 Strong Cover Probability"),
    dict(requires_any=["PR_Rating_Diff","PR_Abs_Rating_Diff"],
         check=lambda r: max(abs(_rv(r,"PR_Rating_Diff")), _rv(r,"PR_Abs_Rating_Diff")) >= THR["pr_diff_meaningful"],
         msg="📈 Meaningful Power‑Rating Edge"),
    dict(requires_any=["PR_Model_Agree_H2H_Flag"],
         check=lambda r: _rv(r,"PR_Model_Agree_H2H_Flag") > 0.0,
         msg="🧠 Power Ratings Agree with Model"),
    dict(requires_any=["PR_Market_Agree_H2H_Flag"],
         check=lambda r: _rv(r,"PR_Market_Agree_H2H_Flag") > 0.0,
         msg="📊 Power Ratings Agree with Market"),

    # Totals context
    dict(requires_any=["TOT_Mispricing","TOT_Proj_Total_Baseline"],
         check=lambda r: _rv(r,"TOT_Mispricing") > 0.0,
         msg="🧮 Totals Mispricing"),

    # Microstructure / network / consistency
    dict(requires_any=["SmallBook_Heavy_Liquidity_Flag","SmallBook_Limit_Skew_Flag","SmallBook_Limit_Skew"],
         check=lambda r: _rv(r,"SmallBook_Heavy_Liquidity_Flag")
                         + _rv(r,"SmallBook_Limit_Skew_Flag")
                         + max(0.0, _rv(r,"SmallBook_Limit_Skew")) > 0.0,
         msg="💧 Liquidity / Limit Skew Pressure"),
    dict(requires_any=["Book_Reliability_x_Magnitude"],
         check=lambda r: _rv(r,"Book_Reliability_x_Magnitude") > 0.0,
         msg="✅ Reliable Books Driving Magnitude"),
    dict(requires_any=["Book_Reliability_x_PROB_SHIFT"],
         check=lambda r: _rv(r,"Book_Reliability_x_PROB_SHIFT") > 0.0,
         msg="✅ Reliable Books Driving Prob Shift"),
    dict(requires_any=["Sharp_Consensus_Weight"],
         check=lambda r: _rv(r,"Sharp_Consensus_Weight") > 0.0,
         msg="🕸️ Sharp Book Consensus"),
    dict(requires_any=["Spread_ML_Inconsistency","Total_vs_Side_ImpliedDelta"],
         check=lambda r: (_rv(r,"Spread_ML_Inconsistency") > 0.0) or (_rv(r,"Total_vs_Side_ImpliedDelta") > 0.0),
         msg="🧩 Internal Pricing Inconsistency"),

    # ATS trends / priors
    dict(requires_any=["ATS_EB_Rate"],
         check=lambda r: _rv(r,"ATS_EB_Rate") >= THR["ats_rate_strong"],
         msg="🏟️ Strong ATS Trend"),
    dict(requires_any=["ATS_EB_Rate_Home"],
         check=lambda r: _rv(r,"ATS_EB_Rate_Home") >= THR["ats_rate_strong"],
         msg="🏠 Strong Home ATS Trend"),
    dict(requires_any=["ATS_EB_Rate_Away"],
         check=lambda r: _rv(r,"ATS_EB_Rate_Away") >= THR["ats_rate_strong"],
         msg="🧳 Strong Away ATS Trend"),
    dict(requires_any=["ATS_EB_Margin"],
         check=lambda r: _rv(r,"ATS_EB_Margin") >= THR["ats_margin_meaningful"],
         msg="➕ Winning vs Spread (Margin)"),
    dict(requires_any=["ATS_Roll_Margin_Decay"],
         check=lambda r: _rv(r,"ATS_Roll_Margin_Decay") >= THR["ats_roll_decay_hot"],
         msg="🔥 Recent ATS Momentum"),

    # Safety gate: surface rationale when you’ve disabled line-move features
    dict(requires_any=["Disable_Line_Move_Features"],
         check=lambda r: _rv(r,"Disable_Line_Move_Features") == 1.0,
         msg="🧯 Line-move features disabled in this market (e.g., MLB spreads / H2H)"),
]

# add dynamic hybrid bin rules
WHY_RULES_V3.extend(build_hybrid_bin_rules())

# --- feature → labels mapping ---
def build_feature_label_index(why_rules) -> dict[str, list[str]]:
    feat2labels = defaultdict(list)
    for rule in why_rules:
        msg = rule.get("msg", "")
        feats = (rule.get("requires", []) or []) + (rule.get("requires_any", []) or [])
        for f in feats:
            if msg and msg not in feat2labels[f]:
                feat2labels[f].append(msg)
    return {f: sorted(v) for f, v in feat2labels.items()}

# --- main attachment (all-features + metadata) ---
def attach_why_all_features(df_in: pd.DataFrame, bundle, model, why_rules=WHY_RULES_V3,
                            ui_numeric_cols=frozenset((
                                'Outcome_Model_Spread','Outcome_Market_Spread','Outcome_Spread_Edge',
                                'Outcome_Cover_Prob','PR_Team_Rating','PR_Opp_Rating','PR_Rating_Diff','PR_Edge_Pts',
                                'edge_x_k','mu_x_k'
                            ))):
    df = df_in.copy()

    # 1) full active set
    active = _resolve_active_features_union(bundle, model, df, why_rules=why_rules)
    active_set = set(active)

    # 2) coerce the columns rules need
    needed = set()
    for rule in why_rules:
        for req in rule.get("requires", []):
            if (req in df.columns) or (req in active_set):
                needed.add(req)
        for req in rule.get("requires_any", []):
            if (req in df.columns) or (req in active_set):
                needed.add(req)
    _coerce_numeric_inplace(df, list(needed), fill=0.0, ui_cols=ui_numeric_cols)

    # 3) evaluate rules
    msgs = []
    rule_hits = np.zeros(len(why_rules), dtype=np.int64)
    for _, row in df.iterrows():
        local = []
        for i, rule in enumerate(why_rules):
            req_all = rule.get("requires", [])
            req_any = rule.get("requires_any", [])
            # gate by presence
            if req_all and not all((r in df.columns) for r in req_all):
                continue
            if req_any and not any((r in df.columns) for r in req_any):
                continue
            try:
                if rule["check"](row):
                    local.append(rule["msg"])
                    rule_hits[i] += 1
            except Exception:
                continue
        msgs.append(" · ".join(local) if local else "—")

    df["Why Model Likes It"] = msgs
    df["Why_Feature_Count"] = df["Why Model Likes It"].apply(lambda s: 0 if s == "—" else (s.count("·") + 1))

    # 4) metadata
    df.attrs["why_feature_labels"] = build_feature_label_index(why_rules)  # {feature: [labels]}
    df.attrs["active_features_used"] = active                                # ordered superset
    df.attrs["why_rule_hits_counts"] = dict(zip([r.get("msg","") for r in why_rules],
                                                map(int, rule_hits)))
    return df


from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# ============================================================
# FIXED: enrich_power_for_training_lowmem
#   - carries Home/Away_Rating_Method
#   - strict method usage (no silent “all-methods” mixing) when available
#   - removes dead/unreachable duplicated code
#   - as-of gate preserved
# ============================================================

def enrich_power_for_training_lowmem(
    df: pd.DataFrame,
    sport_aliases: dict | None = None,
    bq=None,
    table_history: str = "sharplogger.sharp_data.ratings_history",
    pad_days: int = 21,
    rating_lag_hours: float = 12.0,
    project: str | None = None,
    *,
    debug_asof_cols: bool = True,
    strict_method: bool = True,   # ✅ NEW: prevents mixed-scale ratings in training
) -> pd.DataFrame:
    sport_aliases = sport_aliases or {}

    if df is None or df.empty:
        return df.copy()
    if bq is None:
        raise ValueError("BigQuery client `bq` is None — pass your bigquery.Client (e.g., bq=bq_client).")

    # local import (keeps module light)
    from google.cloud import bigquery

    out = df.copy()

    # --- normalize inputs ---
    out["Sport"] = out["Sport"].astype(str).str.upper().str.strip()
    out["Home_Team_Norm"] = out["Home_Team_Norm"].astype(str).str.lower().str.strip()
    out["Away_Team_Norm"] = out["Away_Team_Norm"].astype(str).str.lower().str.strip()
    out["Game_Start"] = pd.to_datetime(out["Game_Start"], utc=True, errors="coerce")

    # sport alias → canon
    canon: dict[str, str] = {}
    for k, v in (sport_aliases or {}).items():
        if isinstance(v, list):
            for a in v:
                canon[str(a).upper()] = str(k).upper()
        else:
            canon[str(k).upper()] = str(v).upper()
    out["Sport"] = out["Sport"].map(lambda s: canon.get(s, s))

    # assume one sport per call
    sport_canon = str(out["Sport"].iloc[0]).upper()
    out = out[out["Sport"] == sport_canon].copy()
    if out.empty:
        return df.copy()

    # --- helpers ---
    def _norm_team_series(s: pd.Series) -> pd.Series:
        return (s.astype(str).str.lower().str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace(".", "", regex=False)
                .str.replace("&", "and", regex=False)
                .str.replace("-", " ", regex=False))

    def _aliases_for(s: str) -> list[str]:
        s = (s or "").upper()
        al = sport_aliases.get(s, []) if sport_aliases else []
        if not isinstance(al, list):
            al = [al]
        return sorted({s, *[str(x).upper() for x in al]})

    # normalize team keys once
    out["Home_Team_Norm"] = _norm_team_series(out["Home_Team_Norm"])
    out["Away_Team_Norm"] = _norm_team_series(out["Away_Team_Norm"])

    teams = pd.Index(out["Home_Team_Norm"]).union(out["Away_Team_Norm"]).unique().tolist()

    # window for fetch
    gmin, gmax = out["Game_Start"].min(), out["Game_Start"].max()
    pad = pd.Timedelta(days=int(pad_days))
    start_iso = pd.to_datetime(gmin - pad, utc=True).isoformat()
    end_iso   = pd.to_datetime(gmax, utc=True).isoformat()  # do NOT fetch beyond max game start

    # ---------- tiny in-function cache ----------
    if not hasattr(enrich_power_for_training_lowmem, "_ratings_cache"):
        enrich_power_for_training_lowmem._ratings_cache = {}
    _CACHE = enrich_power_for_training_lowmem._ratings_cache

    # Preferred method per sport
    PREFERRED_METHOD = {
        "MLB":   "poisson",
        "NFL":   "elo_kalman",
        "NCAAF": "elo_kalman",
        "NBA":   "nba_bpi_adj_em",   # ✅ keep this single scale for ratings
        "WNBA":  "elo_kalman",
        "CFL":   "elo_kalman",
        "NCAAB": "kp_adj_em",
    }

    ZERO_CENTERED = {"kp_adj_em", "nba_bpi_adj_em"}  # points-ish / not Elo-ish 1500

    def fetch_ratings_window_cached(
        sport: str,
        start_iso: str,
        end_iso: str,
        table_name: str,
    ) -> pd.DataFrame:
        method = PREFERRED_METHOD.get(sport.upper())

        # ✅ IMPORTANT: for training, do NOT pull multiple methods (avoids mixed scale)
        methods = [str(method).lower()] if method else None

        key = (
            sport.upper(),
            start_iso, end_iso,
            str(table_name),
            tuple(sorted(teams)),
            tuple(methods) if methods else None,
            float(rating_lag_hours),
            int(pad_days),
            bool(strict_method),
        )
        if key in _CACHE:
            return _CACHE[key].copy()

        # method-aware query (attempt)
        base_sql = f"""
        SELECT
          UPPER(CAST(Sport AS STRING))        AS Sport,
          LOWER(TRIM(CAST(Team AS STRING)))   AS Team_Norm,
          LOWER(CAST(Method AS STRING))       AS Method,
          CAST(Rating AS FLOAT64)             AS Power_Rating,
          TIMESTAMP(Updated_At)               AS AsOfTS
        FROM `{table_name}`
        WHERE UPPER(CAST(Sport AS STRING)) IN UNNEST(@aliases)
          AND LOWER(TRIM(CAST(Team AS STRING))) IN UNNEST(@teams)
          AND TIMESTAMP(Updated_At) >= @start
          AND TIMESTAMP(Updated_At) <= @end
        """

        params_common = [
            bigquery.ArrayQueryParameter("aliases", "STRING", _aliases_for(sport)),
            bigquery.ArrayQueryParameter("teams", "STRING", teams),
            bigquery.ScalarQueryParameter("start", "TIMESTAMP", pd.Timestamp(start_iso).to_pydatetime()),
            bigquery.ScalarQueryParameter("end",   "TIMESTAMP", pd.Timestamp(end_iso).to_pydatetime()),
        ]

        def _run(sql: str, params: list):
            return bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()

        df_r: pd.DataFrame

        if methods:
            sql1 = base_sql + "\n  AND LOWER(CAST(Method AS STRING)) IN UNNEST(@methods)\n"
            params1 = params_common + [bigquery.ArrayQueryParameter("methods", "STRING", [m.lower() for m in methods])]
            try:
                df_r = _run(sql1, params1)
            except Exception as e:
                msg = str(e).lower()
                method_missing = (("unrecognized name" in msg and "method" in msg) or
                                  ("name method" in msg and "not found" in msg))
                if not method_missing:
                    raise

                # Table has no Method column → fallback query without Method
                base_sql_nomethod = f"""
                SELECT
                  UPPER(CAST(Sport AS STRING))        AS Sport,
                  LOWER(TRIM(CAST(Team AS STRING)))   AS Team_Norm,
                  CAST(Rating AS FLOAT64)             AS Power_Rating,
                  TIMESTAMP(Updated_At)               AS AsOfTS
                FROM `{table_name}`
                WHERE UPPER(CAST(Sport AS STRING)) IN UNNEST(@aliases)
                  AND LOWER(TRIM(CAST(Team AS STRING))) IN UNNEST(@teams)
                  AND TIMESTAMP(Updated_At) >= @start
                  AND TIMESTAMP(Updated_At) <= @end
                """
                df_r = _run(base_sql_nomethod, params_common)
        else:
            # No preferred method defined
            # If strict_method, we can still run and accept (method-less tables)
            df_r = _run(base_sql, params_common)

        # If strict_method and the table supports Method, empty means mismatch → hard error
        if df_r.empty:
            _CACHE[key] = df_r
            if strict_method and methods:
                raise RuntimeError(
                    f"No ratings returned for sport={sport.upper()} methods={methods} "
                    f"from {table_name} in window [{start_iso}, {end_iso}]."
                )
            return df_r

        # normalize + clean
        df_r["Sport"] = df_r["Sport"].astype(str).str.upper()
        df_r["Team_Norm"] = _norm_team_series(df_r["Team_Norm"])
        df_r["Power_Rating"] = pd.to_numeric(df_r["Power_Rating"], errors="coerce")
        df_r["AsOfTS"] = pd.to_datetime(df_r["AsOfTS"], utc=True, errors="coerce")

        keep = ["Sport", "Team_Norm", "Power_Rating", "AsOfTS"]
        if "Method" in df_r.columns:
            df_r["Method"] = df_r["Method"].astype(str).str.lower().str.strip()
            keep.insert(2, "Method")

        df_r = df_r[keep].dropna(subset=["Team_Norm", "Power_Rating", "AsOfTS"])
        df_r = df_r[df_r["Sport"] == sport.upper()]

        _CACHE[key] = df_r
        return df_r.copy()

    # === Pull ratings (cached) ===
    ratings = fetch_ratings_window_cached(
        sport=sport_canon,
        start_iso=start_iso,
        end_iso=end_iso,
        table_name=table_history,
    )

    method_used = str(PREFERRED_METHOD.get(sport_canon, "")).lower()
    base_default = np.float32(0.0) if (method_used in ZERO_CENTERED) else np.float32(1500.0)

    # init defaults
    out["Home_Power_Rating"] = base_default
    out["Away_Power_Rating"] = base_default
    out["Power_Rating_Diff"] = np.float32(0.0)

    # carry methods for gating (even if empty)
    out["Home_Rating_Method"] = method_used
    out["Away_Rating_Method"] = method_used

    if debug_asof_cols:
        out["Home_Rating_AsOfTS"] = pd.NaT
        out["Away_Rating_AsOfTS"] = pd.NaT

    if ratings.empty:
        return out

    ratings = ratings.copy()
    ratings["Team_Norm"] = _norm_team_series(ratings["Team_Norm"])
    ratings = ratings[ratings["Team_Norm"].isin(teams)]
    if ratings.empty:
        return out

    # compact arrays per team: times + values + method (sorted)
    team_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for team, g in ratings.groupby("Team_Norm", sort=False):
        g = g.sort_values("AsOfTS")
        t_arr = g["AsOfTS"].to_numpy(dtype="datetime64[ns]")
        r_arr = g["Power_Rating"].to_numpy(dtype=np.float32)
        if "Method" in g.columns:
            m_arr = g["Method"].astype(str).str.lower().fillna("").to_numpy(dtype=object)
        else:
            m_arr = np.array([""] * len(g), dtype=object)
        team_series[team] = (t_arr, r_arr, m_arr)

    # ============================
    # As-of gate (hard)
    # ============================
    gs_ns = out["Game_Start"].values.astype("datetime64[ns]").astype("int64")
    lag_ns = np.int64(round(float(rating_lag_hours) * 3600.0 * 1e9))
    cutoff_ns = gs_ns - lag_ns

    home_vals = np.full(len(out), base_default, dtype=np.float32)
    away_vals = np.full(len(out), base_default, dtype=np.float32)
    home_methods = np.full(len(out), method_used, dtype=object)
    away_methods = np.full(len(out), method_used, dtype=object)

    if debug_asof_cols:
        home_asof = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
        away_asof = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")

    home_team = out["Home_Team_Norm"].to_numpy()
    away_team = out["Away_Team_Norm"].to_numpy()

    # Home side: pick latest rating with AsOfTS <= cutoff
    for team, (t_arr, r_arr, m_arr) in team_series.items():
        mask = (home_team == team)
        if not mask.any():
            continue
        ts = cutoff_ns[mask].astype("datetime64[ns]")
        idx = np.searchsorted(t_arr, ts, side="right") - 1
        valid = idx >= 0

        vals = np.full(idx.shape, base_default, dtype=np.float32)
        meth = np.full(idx.shape, method_used, dtype=object)

        if valid.any():
            vals[valid] = r_arr[idx[valid]]
            if len(m_arr) == len(r_arr):
                meth[valid] = m_arr[idx[valid]]
            if debug_asof_cols:
                home_asof[mask] = np.where(valid, t_arr[idx], home_asof[mask])

        home_vals[mask] = vals
        home_methods[mask] = meth

    # Away side
    for team, (t_arr, r_arr, m_arr) in team_series.items():
        mask = (away_team == team)
        if not mask.any():
            continue
        ts = cutoff_ns[mask].astype("datetime64[ns]")
        idx = np.searchsorted(t_arr, ts, side="right") - 1
        valid = idx >= 0

        vals = np.full(idx.shape, base_default, dtype=np.float32)
        meth = np.full(idx.shape, method_used, dtype=object)

        if valid.any():
            vals[valid] = r_arr[idx[valid]]
            if len(m_arr) == len(r_arr):
                meth[valid] = m_arr[idx[valid]]
            if debug_asof_cols:
                away_asof[mask] = np.where(valid, t_arr[idx], away_asof[mask])

        away_vals[mask] = vals
        away_methods[mask] = meth

    out["Home_Power_Rating"] = home_vals
    out["Away_Power_Rating"] = away_vals
    out["Power_Rating_Diff"] = (home_vals - away_vals).astype("float32")
    out["Home_Rating_Method"] = pd.Series(home_methods, index=out.index).astype(str).str.lower()
    out["Away_Rating_Method"] = pd.Series(away_methods, index=out.index).astype(str).str.lower()

    if debug_asof_cols:
        out["Home_Rating_AsOfTS"] = pd.to_datetime(home_asof, utc=True, errors="coerce")
        out["Away_Rating_AsOfTS"] = pd.to_datetime(away_asof, utc=True, errors="coerce")

        cutoff_dt = out["Game_Start"] - pd.Timedelta(hours=float(rating_lag_hours))
        bad_home = out["Home_Rating_AsOfTS"].notna() & (out["Home_Rating_AsOfTS"] > cutoff_dt)
        bad_away = out["Away_Rating_AsOfTS"].notna() & (out["Away_Rating_AsOfTS"] > cutoff_dt)
        n_bad = int(bad_home.sum() + bad_away.sum())
        if n_bad:
            ex_cols = [c for c in ["feat_Game_Key","Game_Key","Home_Team_Norm","Away_Team_Norm","Game_Start",
                                   "Home_Rating_AsOfTS","Away_Rating_AsOfTS","Home_Rating_Method","Away_Rating_Method"]
                       if c in out.columns]
            raise RuntimeError(
                f"[LEAK] {n_bad} power-rating rows violate {rating_lag_hours}h gate. Examples:\n"
                f"{out.loc[bad_home | bad_away, ex_cols].head(25)}"
            )

    return out


# ============================================================
# FIXED: enrich_and_grade_for_training
#   - converts ratings to points ONLY when method is Elo-ish
#   - fixes fallback beta direction (25, 35... not 1/25)
#   - preserves *_Units debug cols
# ============================================================

def enrich_and_grade_for_training(
    df_spread_rows: pd.DataFrame,
    bq,
    sport_aliases: dict,
    value_col: str = "Value",
    outcome_col: str = "Outcome_Norm",
    pad_days: int = 30,
    rating_lag_hours: float = 12.0,
    table_history: str = "sharplogger.sharp_data.ratings_history",
    project: str | None = None,
    *,
    beta_project: str = "sharplogger",
    beta_dataset: str = "sharp_data",
    beta_table: str = "sharp_scores_with_features",
    beta_min_rows_per_sport: int = 5000,
) -> pd.DataFrame:
    if df_spread_rows is None or df_spread_rows.empty:
        return df_spread_rows.copy()

    # 1) Ratings (as-of) for the unique games in this batch
    base = enrich_power_for_training_lowmem(
        df=df_spread_rows[['Sport','Home_Team_Norm','Away_Team_Norm','Game_Start']].drop_duplicates(),
        bq=bq,
        sport_aliases=sport_aliases,
        table_history=table_history,
        pad_days=pad_days,
        rating_lag_hours=float(rating_lag_hours),
        project=project,
        debug_asof_cols=True,
        strict_method=True,   # ✅ stop mixed-scale
    )

    # 1b) Convert rating units -> spread points (method-aware)
    ZERO_CENTERED = {"kp_adj_em", "nba_bpi_adj_em"}  # points-ish scales: do NOT rescale

    if base is not None and not base.empty:
        beta_map = resolve_pr_beta_map(
            bq,
            project=beta_project,
            dataset=beta_dataset,
            table=beta_table,
            min_rows_per_sport=int(beta_min_rows_per_sport),
        )

        # ✅ fallback betas MUST be "points per rating point"
        beta_fallback_map = {
            "NFL":   25.0,
            "NCAAF": 25.0,
            "NBA":   35.0,
            "WNBA":  35.0,
            "CFL":   25.0,
            "MLB":   50.0,
            "NCAAB": 1.0,   # kp_adj_em already points-ish; we also gate by method below
        }

        sp = base["Sport"].astype(str).str.upper().values
        beta = np.array([beta_map.get(s, np.nan) for s in sp], dtype="float32")
        beta_fb = np.array([beta_fallback_map.get(s, 30.0) for s in sp], dtype="float32")
        beta = np.where(np.isfinite(beta), beta, beta_fb).astype("float32")

        # ✅ method gate: if method is already points-ish, force beta=1.0
        if "Home_Rating_Method" in base.columns:
            m = base["Home_Rating_Method"].astype(str).str.lower().values
            beta = np.where(np.isin(m, list(ZERO_CENTERED)), 1.0, beta).astype("float32")

        # keep original (units) for debugging
        for c in ["Home_Power_Rating", "Away_Power_Rating", "Power_Rating_Diff"]:
            if c in base.columns and f"{c}_Units" not in base.columns:
                base[f"{c}_Units"] = pd.to_numeric(base[c], errors="coerce").astype("float32")

        base["PR_Pts_Beta_Used"] = beta.astype("float32")

        for c in ["Home_Power_Rating", "Away_Power_Rating", "Power_Rating_Diff"]:
            if c in base.columns:
                base[c] = (pd.to_numeric(base[c], errors="coerce").astype("float32") * beta).astype("float32")

    # 2) Consensus market (favorite + absolute spread “k”)
    g_cons = prep_consensus_market_spread_lowmem(
        df_spread_rows, value_col=value_col, outcome_col=outcome_col
    )

    # 3) Join → game-level, compute model favorite/dog edges + probs
    game_key = ['Sport','Home_Team_Norm','Away_Team_Norm']
    g_full = base.merge(g_cons, on=game_key, how='left')
    g_fc   = favorite_centric_from_powerdiff_lowmem(g_full)

    # Ensure PR cols exist even if base missed
    for c in ['Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff', 'PR_Pts_Beta_Used',
              'Home_Rating_Method','Away_Rating_Method',
              'Home_Power_Rating_Units','Away_Power_Rating_Units','Power_Rating_Diff_Units']:
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
        'Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff',
        'PR_Pts_Beta_Used',
        'Home_Rating_Method','Away_Rating_Method',
        'Home_Power_Rating_Units','Away_Power_Rating_Units','Power_Rating_Diff_Units',
    ]
    for c in keep_cols:
        if c not in g_fc.columns:
            g_fc[c] = np.nan
    g_fc = g_fc[keep_cols].copy()

    out = df_spread_rows.merge(g_fc, on=game_key, how='left')

    # Map game-level numbers to the row’s outcome side
    is_fav = (out[outcome_col].astype(str).values == out['Market_Favorite_Team'].astype(str).values)
    out['Outcome_Model_Spread']  = np.where(is_fav, out['Model_Fav_Spread'].values, out['Model_Dog_Spread'].values).astype('float32')
    out['Outcome_Market_Spread'] = np.where(is_fav, out['Favorite_Market_Spread'].values, out['Underdog_Market_Spread'].values).astype('float32')
    out['Outcome_Spread_Edge']   = np.where(is_fav, out['Fav_Edge_Pts'].values, out['Dog_Edge_Pts'].values).astype('float32')
    out['Outcome_Cover_Prob']    = np.where(is_fav, out['Fav_Cover_Prob'].values, out['Dog_Cover_Prob'].values).astype('float32')

    return out

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def fit_iso_platt_beta(p_oof, y_oof, eps=1e-4, use_quantile_iso=False):
    p = np.clip(np.asarray(p_oof, float), eps, 1-eps)
    y = np.asarray(y_oof, int)

    # Isotonic (regular or quantile-regularized)
    if use_quantile_iso:
        iso = fit_regularized_isotonic(p, y, q=80, eps=eps)
    else:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(p, y)

    # Platt
    platt = LogisticRegression(max_iter=2000)
    platt.fit(p.reshape(-1,1), y)

    # Beta (optional)
    beta = None
    try:
        from betacal import BetaCalibration
        beta = BetaCalibration(parameters="abm")
        beta.fit(p.reshape(-1,1), y)
    except Exception:
        pass

    return {"iso": iso, "platt": platt, "beta": beta}

def apply_calibrator(cal_tuple, p, *, clip=(0.01, 0.99)):
    kind, model = cal_tuple
    p = np.asarray(p, float)
    if kind == "iso":
        out = model.transform(p)
    elif kind == "beta":
        out = model.predict(p.reshape(-1, 1))
    else:  # "platt"
        out = model.predict_proba(p.reshape(-1, 1))[:, 1]
    if clip:
        lo, hi = clip
        out = np.clip(out, lo, hi)
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

PHI = _phi
def select_blend(cals, p_oof, y_oof, eps=1e-4):
    from sklearn.metrics import log_loss
    x = np.clip(np.asarray(p_oof, float), eps, 1-eps)
    y = np.asarray(y_oof,   int)

    kinds = ["iso", "platt"] + (["beta"] if cals["beta"] is not None else [])
    best = None
    # Try convex blends: p = a*iso + (1-a)*base, a∈[0..1]
    for base in kinds:
        p_iso  = _apply_cal("iso",  cals["iso"],  x)
        p_base = _apply_cal(base,   cals[base],   x)
        for a in np.linspace(0.0, 1.0, 11):  # 0.0, 0.1, ..., 1.0
            p_blend = np.clip(a*p_iso + (1-a)*p_base, eps, 1-eps)
            score   = log_loss(y, p_blend, labels=[0,1])
            if (best is None) or (score < best[0]):
                best = (score, base, float(a))
    _, base, alpha = best
    return {"cals": cals, "base": base, "alpha": alpha}


def apply_blend(sel, p, eps=1e-4, clip=(0.001, 0.999)):
    x = np.clip(np.asarray(p, float), eps, 1-eps)
    cals, base, a = sel["cals"], sel["base"], sel["alpha"]
    p_iso  = _apply_cal("iso",  cals["iso"],  x)
    p_base = _apply_cal(base,   cals[base],   x)
    out    = a*p_iso + (1-a)*p_base
    if clip:
        lo, hi = clip
        out = np.clip(out, lo, hi)
    return out

# extra_plumbing.py
import os, numpy as np, pandas as pd

SPORT_SPREAD_CFG = {
    "NFL":   dict(scale=np.float32(1.0),  HFA=np.float32(2.1),  sigma_pts=np.float32(13.2)),
    "NCAAF": dict(scale=np.float32(1.0),  HFA=np.float32(2.6),  sigma_pts=np.float32(16.0)),
    "NBA":   dict(scale=np.float32(1.0),  HFA=np.float32(2.8),  sigma_pts=np.float32(11.5)),
    "WNBA":  dict(scale=np.float32(1.0),  HFA=np.float32(2.0),  sigma_pts=np.float32(10.5)),
    "CFL":   dict(scale=np.float32(1.0),  HFA=np.float32(1.6),  sigma_pts=np.float32(13.5)),
    # MLB ratings are not in run units (1500 + 400*(atk+dfn)), so scale ≈ 89–90.
    "MLB":   dict(scale=np.float32(89.0), HFA=np.float32(0.20), sigma_pts=np.float32(3.1)),
    "NCAAB": dict(scale=np.float32(1.0),  HFA=np.float32(3.2),  sigma_pts=np.float32(11.0)),
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


def enrich_and_grade_for_training(
    df_spread_rows: pd.DataFrame,
    bq,
    sport_aliases: dict,
    value_col: str = "Value",
    outcome_col: str = "Outcome_Norm",
    pad_days: int = 30,
    rating_lag_hours: float = 12.0,
    table_history: str = "sharplogger.sharp_data.ratings_history",
    project: str | None = None,
    *,
    beta_project: str = "sharplogger",
    beta_dataset: str = "sharp_data",
    beta_table: str = "sharp_scores_with_features",
    beta_min_rows_per_sport: int = 5000,
) -> pd.DataFrame:
    if df_spread_rows.empty:
        return df_spread_rows.copy()

    # 1) Ratings (as-of) for the unique games in this batch
    base = enrich_power_for_training_lowmem(
        df=df_spread_rows[['Sport','Home_Team_Norm','Away_Team_Norm','Game_Start']].drop_duplicates(),
        bq=bq,
        sport_aliases=sport_aliases,
        table_history=table_history,
        pad_days=pad_days,
        rating_lag_hours=float(rating_lag_hours),  # ✅ use param
        project=project,
    )

    # ✅ 1b) Convert ratings from "units" -> "spread points" using learned sport betas
    # base must have Sport and rating cols
    if base is not None and not base.empty:
        beta_map = resolve_pr_beta_map(
            bq,
            project=beta_project,
            dataset=beta_dataset,
            table=beta_table,
            min_rows_per_sport=int(beta_min_rows_per_sport),
        )

        # fallback if sport missing / low sample
        beta_fallback_map = {
            "NFL":   25.0,
            "NCAAF": 25.0,
            "NBA":   35.0,
            "WNBA":  35.0,
            "CFL":   25.0,
            "MLB":   50.0,
            "NCAAB": 1.0,   # only if kp_adj_em already in points-like units
        }

        sp = base["Sport"].astype(str).str.upper().values
        beta = np.array([beta_map.get(s, np.nan) for s in sp], dtype="float32")
        beta_fb = np.array([beta_fallback_map.get(s, 1.0/30.0) for s in sp], dtype="float32")
        beta = np.where(np.isfinite(beta), beta, beta_fb).astype("float32")

        # keep original (units) for debugging / optional feature use
        for c in ["Home_Power_Rating", "Away_Power_Rating", "Power_Rating_Diff"]:
            if c in base.columns:
                base[f"{c}_Units"] = pd.to_numeric(base[c], errors="coerce").astype("float32")

        base["PR_Pts_Beta_Used"] = beta
        if "Home_Power_Rating" in base.columns:
            base["Home_Power_Rating"] = (pd.to_numeric(base["Home_Power_Rating"], errors="coerce").astype("float32") * beta).astype("float32")
        if "Away_Power_Rating" in base.columns:
            base["Away_Power_Rating"] = (pd.to_numeric(base["Away_Power_Rating"], errors="coerce").astype("float32") * beta).astype("float32")
        if "Power_Rating_Diff" in base.columns:
            base["Power_Rating_Diff"] = (pd.to_numeric(base["Power_Rating_Diff"], errors="coerce").astype("float32") * beta).astype("float32")

    # 2) Consensus market (favorite + absolute spread “k”)
    g_cons = prep_consensus_market_spread_lowmem(
        df_spread_rows, value_col=value_col, outcome_col=outcome_col
    )

    # 3) Join → game-level, compute model favorite/dog edges + probs
    game_key = ['Sport','Home_Team_Norm','Away_Team_Norm']
    g_full = base.merge(g_cons, on=game_key, how='left')
    g_fc   = favorite_centric_from_powerdiff_lowmem(g_full)

    # Ensure PR cols exist even if base missed
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
        'Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff',
        'PR_Pts_Beta_Used',  # ✅ helpful to propagate
    ]
    for c in keep_cols:
        if c not in g_fc.columns:
            g_fc[c] = np.nan
    g_fc = g_fc[keep_cols].copy()

    out = df_spread_rows.merge(g_fc, on=game_key, how='left')

    # Map game-level numbers to the row’s outcome side
    is_fav = (out[outcome_col].astype(str).values == out['Market_Favorite_Team'].astype(str).values)
    out['Outcome_Model_Spread']  = np.where(is_fav, out['Model_Fav_Spread'].values, out['Model_Dog_Spread'].values).astype('float32')
    out['Outcome_Market_Spread'] = np.where(is_fav, out['Favorite_Market_Spread'].values, out['Underdog_Market_Spread'].values).astype('float32')
    out['Outcome_Spread_Edge']   = np.where(is_fav, out['Fav_Edge_Pts'].values, out['Dog_Edge_Pts'].values).astype('float32')
    out['Outcome_Cover_Prob']    = np.where(is_fav, out['Fav_Cover_Prob'].values, out['Dog_Cover_Prob'].values).astype('float32')

    return out


def favorite_centric_from_powerdiff_lowmem(df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Takes one row per game (home/away) with:
      - Sport, Home_Team_Norm, Away_Team_Norm
      - Power_Rating_Diff (Home - Away)
      - market consensus fields (k, Market_Favorite_Team, etc.)

    FIXES:
      ✅ Handles NCAAB KenPom-style AdjEM correctly:
           Power_Rating_Diff is EM per 100 possessions (not 1500-scale points)
           Convert to expected margin in points via possessions proxy.
      ✅ Keeps existing behavior for Elo-ish sports (1500-scale) + MLB scale.
      ✅ Robust to missing columns (k, totals, etc.) and bad dtypes.
    """

    g = df_games.copy()
    if g.empty:
        return pd.DataFrame(columns=[
            'Sport','Home_Team_Norm','Away_Team_Norm',
            'Model_Expected_Margin','Model_Expected_Margin_Abs','Sigma_Pts',
            'Model_Fav_Spread','Model_Dog_Spread','Model_Favorite_Team','Model_Underdog_Team',
            'Market_Favorite_Team','Market_Underdog_Team','Favorite_Market_Spread','Underdog_Market_Spread',
            'Fav_Edge_Pts','Dog_Edge_Pts','Fav_Cover_Prob','Dog_Cover_Prob'
        ])

    g['Sport'] = g['Sport'].astype(str).str.upper().str.strip()

    n = len(g)

    # rating-units → points (or runs for MLB), default 1.0
    scale = np.full(n, np.float32(1.0), dtype=np.float32)
    # home-field advantage in points/runs, default 0
    hfa   = np.zeros(n, dtype=np.float32)
    # margin SD
    sigma = np.full(n, np.float32(12.0), dtype=np.float32)

    sp = g['Sport'].values
    for s, cfg in SPORT_SPREAD_CFG.items():
        mask = (sp == s)
        if mask.any():
            scale[mask] = np.float32(cfg.get('scale', 1.0))
            hfa[mask]   = np.float32(cfg.get('HFA', 0.0))
            sigma[mask] = np.float32(cfg.get('sigma_pts', 12.0))

    # Power_Rating_Diff must be (Home_Rating - Away_Rating)
    pr_diff = pd.to_numeric(g.get('Power_Rating_Diff', 0.0), errors='coerce').fillna(0.0).astype('float32').to_numpy()

    # ---- NCAAB KenPom AdjEM conversion ----
    # If NCAAB ratings are kp_adj_em: Power_Rating_Diff is "points per 100 possessions".
    # Convert to expected margin in points using possessions proxy:
    #   mu_pts ≈ (AdjEM_diff / 100) * poss
    #
    # Possessions proxy preference:
    #   1) If you have Total_Close or Total (market total points), poss ≈ total / 2.06
    #   2) Else fallback to 68
    #
    # You can feed Total_Close into this function via upstream merge if desired.
    # base expected margin in points for Elo-ish sports
    mu = (pr_diff / np.where(scale == 0, 1.0, scale)) + hfa
    mu = mu.astype("float32")

    # ---- NCAAB KenPom AdjEM conversion (per-100 -> points) ----
    NCAAB_TOTAL_PPP = np.float32(2.06)
    poss_default = np.float32(68.0)

    poss = np.full(n, poss_default, dtype=np.float32)
    for total_col in ("Total_Close", "Market_Total", "Total", "Closing_Total", "Total_Points"):
        if total_col in g.columns:
            tot = pd.to_numeric(g[total_col], errors="coerce").astype("float32").to_numpy()
            poss_est = tot / NCAAB_TOTAL_PPP
            poss_est = np.clip(poss_est, 55.0, 80.0).astype("float32")
            poss = np.where(np.isfinite(poss_est), poss_est, poss).astype("float32")
            break

    is_ncaab = (sp == "NCAAB")
    if is_ncaab.any():
        mu_ncaab = (pr_diff / 100.0) * poss + hfa
        mu = np.where(is_ncaab, mu_ncaab.astype("float32"), mu)

    # ---- NBA BPI AdjEM conversion (per-100 -> points) ----
    # Prefer your tempo columns (nba_bpi_adj_t): possessions per game
    NBA_TOTAL_PPP = np.float32(2.26)
    nba_poss_default = np.float32(100.0)

    is_nba = (sp == "NBA")
    if is_nba.any():
        nba_poss = np.full(n, nba_poss_default, dtype=np.float32)

        # 1) use tempo if present (best)
        if "Home_BPI_Tempo" in g.columns and "Away_BPI_Tempo" in g.columns:
            ht = pd.to_numeric(g["Home_BPI_Tempo"], errors="coerce").astype("float32").to_numpy()
            at = pd.to_numeric(g["Away_BPI_Tempo"], errors="coerce").astype("float32").to_numpy()
            poss_est = 0.5 * (ht + at)
            poss_est = np.clip(poss_est, 90.0, 108.0).astype("float32")
            nba_poss = np.where(np.isfinite(poss_est), poss_est, nba_poss).astype("float32")

        # 2) otherwise derive poss from totals if available
        else:
            for total_col in ("Total_Close", "Market_Total", "Total", "Closing_Total", "Total_Points"):
                if total_col in g.columns:
                    tot = pd.to_numeric(g[total_col], errors="coerce").astype("float32").to_numpy()
                    poss_est = tot / NBA_TOTAL_PPP
                    poss_est = np.clip(poss_est, 90.0, 108.0).astype("float32")
                    nba_poss = np.where(np.isfinite(poss_est), poss_est, nba_poss).astype("float32")
                    break

        # Power_Rating_Diff is nba_bpi_adj_em (net per 100)
        mu_nba = (pr_diff / 100.0) * nba_poss + hfa
        mu = np.where(is_nba, mu_nba.astype("float32"), mu)

    mu_abs = np.abs(mu).astype('float32')

    # market absolute spread (median abs from consensus step)
    k = pd.to_numeric(g.get('k', np.nan), errors='coerce').astype('float32').to_numpy()

    # If k missing, edges/probs should be nan-safe
    fav_edge = (mu_abs - k).astype('float32')
    dog_edge = (k - mu_abs).astype('float32')

    # Cover probs for favorite side under Normal(margin_abs; mu_abs, sigma):
    # P(margin > k) = 1 - Φ((k - mu_abs)/σ)
    denom = sigma.copy()
    denom[denom == 0] = np.nan
    z_cov = (k - mu_abs) / denom

    # _phi should be standard normal CDF; if k is nan -> z_cov nan -> probs nan
    fav_cover = (1.0 - _phi(z_cov)).astype('float32')
    dog_cover = (1.0 - fav_cover).astype('float32')

    # Robust market columns (may be missing)
    def _col(name, default=np.nan):
        if name in g.columns:
            return g[name]
        return pd.Series(default, index=g.index)

    g_out = pd.DataFrame({
        'Sport': g['Sport'].astype(str).to_numpy(),
        'Home_Team_Norm': g['Home_Team_Norm'].astype(str).to_numpy(),
        'Away_Team_Norm': g['Away_Team_Norm'].astype(str).to_numpy(),

        'Model_Expected_Margin': mu.astype('float32'),
        'Model_Expected_Margin_Abs': mu_abs.astype('float32'),
        'Sigma_Pts': sigma.astype('float32'),

        # model spreads (favorite negative, dog positive)
        'Model_Fav_Spread': (-mu_abs).astype('float32'),
        'Model_Dog_Spread': ( mu_abs).astype('float32'),

        # favorite is team with +expected margin (home favored if mu>=0)
        'Model_Favorite_Team': np.where(mu >= 0, g['Home_Team_Norm'].to_numpy(), g['Away_Team_Norm'].to_numpy()),
        'Model_Underdog_Team': np.where(mu >= 0, g['Away_Team_Norm'].to_numpy(), g['Home_Team_Norm'].to_numpy()),

        # market bits (merged upstream)
        'Market_Favorite_Team': _col('Market_Favorite_Team', '').astype(str).to_numpy(),
        'Market_Underdog_Team': _col('Market_Underdog_Team', '').astype(str).to_numpy(),
        'Favorite_Market_Spread': pd.to_numeric(_col('Favorite_Market_Spread', np.nan), errors='coerce').astype('float32').to_numpy(),
        'Underdog_Market_Spread': pd.to_numeric(_col('Underdog_Market_Spread', np.nan), errors='coerce').astype('float32').to_numpy(),

        # edges + cover probs at game level
        'Fav_Edge_Pts': fav_edge.astype('float32'),
        'Dog_Edge_Pts': dog_edge.astype('float32'),
        'Fav_Cover_Prob': fav_cover.astype('float32'),
        'Dog_Cover_Prob': dog_cover.astype('float32'),
    })

    return g_out

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



SMALL_LEAGUES = {"WNBA", "CFL"}
BIG_LEAGUES   = {"MLB", "NBA", "NFL", "NCAAF", "NCAAB"}  # extend as needed

from scipy.stats import randint, uniform, loguniform
# Requires:
#   from scipy.stats import randint, uniform, loguniform

def get_xgb_search_space(
    sport: str,
    *,
    X_rows: int,
    n_jobs: int = 1,
    scale_pos_weight: float = 1.0,   # kept for API compatibility; ignored inside
    features: list[str] | None = None,
    use_monotone: bool = False,
) -> tuple[dict, dict, dict]:
    s = str(sport).upper()
    n_jobs = int(n_jobs)
    SMALL_LEAGUES = {"WNBA", "CFL"}    # keep your set
    SMALL = (s in SMALL_LEAGUES)

    # ---- Base kwargs: conservative + deterministic ----
    base_kwargs = dict(
        objective="binary:logistic",
        tree_method="hist",
        predictor="cpu_predictor",
        grow_policy="lossguide",
        sampling_method="uniform",
        max_bin=256,
        max_delta_step=0.5,        # small clamp helps stability
        reg_lambda=6.0,            # stronger L2 default
        reg_alpha=0.05,            # small L1 default
        min_child_weight=8.0,      # raise split threshold to fight memorization
        n_jobs=n_jobs,
        random_state=42,
        importance_type="total_gain",
    )

    # (optional) monotone scaffolding unchanged
    if use_monotone and features:
        pass
    # ---- Auto-regularize search if data are thin ---------------------------------
    n = int(X_rows)
    p = int(len(features) if features else 0)
    rows_per_feat = (n / max(p, 1)) if p else float("inf")
    
    # Tightness levels based on sample size / dimensionality
    THIN = (n < 600 or rows_per_feat < 25)       # moderate
    VERY_THIN = (n < 300 or rows_per_feat < 12)  # aggressive
    
    def _shrink(space: dict, *, thin=False, very_thin=False) -> dict:
        s = dict(space)
        # Clamp max_leaves / depth (lower variance trees)
        if "max_leaves" in s and hasattr(s["max_leaves"], "high"):
            s["max_leaves"] = randint(32, min(128, s["max_leaves"].high if not very_thin else 96))
        if "max_depth" in s and hasattr(s["max_depth"], "high"):
            s["max_depth"] = randint(2, min(5, s["max_depth"].high))
        # Push LR lower; raise min_child_weight / gamma; tighten sampling
        s["learning_rate"]    = loguniform(0.006, 0.03 if very_thin else 0.035)
        s["min_child_weight"] = loguniform(8.0, 32.0 if thin else 24.0)
        s["gamma"]            = loguniform(2.0, 12.0 if thin else 8.0)
        s["subsample"]        = uniform(0.55, 0.25)      # 0.55–0.80
        s["colsample_bytree"] = uniform(0.50, 0.25)      # 0.50–0.75
        s["colsample_bynode"] = uniform(0.55, 0.25)      # 0.55–0.80
        s["reg_lambda"]       = loguniform(8.0, 24.0)    # ↑ ridge
        s["reg_alpha"]        = loguniform(0.05, 1.5)    # ↑ lasso
        s["max_bin"]          = randint(160, 256)        # keep hist bins tight
        return s

    # ---- Search spaces: tightened + no explosive wideners ----
    if SMALL:
        # small-league => fewer leaves, slower LR
        params_ll = {
            "max_depth":        randint(3, 5),
            "max_leaves":       randint(48, 96),
            "learning_rate":    loguniform(0.010, 0.035),
            "subsample":        uniform(0.65, 0.25),      # 0.65–0.90
            "colsample_bytree": uniform(0.60, 0.30),      # 0.60–0.90
            "colsample_bynode": uniform(0.60, 0.30),
            "min_child_weight": loguniform(4.0, 24.0),
            "gamma":            uniform(1.0, 5.0),
            "reg_lambda":       loguniform(4.0, 12.0),
            "reg_alpha":        loguniform(0.02, 0.8),
            "max_bin":          randint(192, 289),
        }
        params_auc = dict(params_ll)  # same shape; AUC will pick slightly larger leaves via search
    else:
        # bigger leagues can afford a bit more depth, still regularized
        params_ll = {
            "max_depth":        randint(3, 6),
            "max_leaves":       randint(64, 160),
            "learning_rate":    loguniform(0.012, 0.045),
            "subsample":        uniform(0.65, 0.25),      # 0.65–0.90
            "colsample_bytree": uniform(0.60, 0.30),      # 0.60–0.90
            "colsample_bynode": uniform(0.60, 0.30),
            "min_child_weight": loguniform(4.0, 28.0),
            "gamma":            uniform(1.0, 6.0),
            "reg_lambda":       loguniform(4.0, 12.0),
            "reg_alpha":        loguniform(0.02, 1.2),
            "max_bin":          randint(192, 321),
        }
        params_auc = dict(params_ll)

    # ⚠️ remove the old params_common “wideners” (max_leaves=512/1024, LR up to .10, etc.)
    # They were a big overfit lever.
    if VERY_THIN:
        params_ll  = _shrink(params_ll,  very_thin=True)
        params_auc = _shrink(params_auc, very_thin=True)
    elif THIN:
        params_ll  = _shrink(params_ll,  thin=True)
        params_auc = _shrink(params_auc, thin=True)
    # scrub danger keys, as before
    danger = {"objective","_estimator_type","response_method","eval_metric","scale_pos_weight"}
    params_ll  = {k:v for k,v in params_ll.items()  if k not in danger}
    params_auc = {k:v for k,v in params_auc.items() if k not in danger}
    return base_kwargs, params_ll, params_auc

import numpy as np
from sklearn.metrics import roc_auc_score

def compute_overfit_signals(
    y_tr, p_tr, y_va, p_va, w_tr=None, w_va=None, bins: int = 10
) -> dict:
    # AUCs
    auc_tr = float(roc_auc_score(y_tr, p_tr, sample_weight=w_tr)) if np.unique(y_tr).size==2 else np.nan
    auc_va = float(roc_auc_score(y_va, p_va, sample_weight=w_va)) if np.unique(y_va).size==2 else np.nan
    auc_gap = float(auc_tr - auc_va) if np.isfinite(auc_tr) and np.isfinite(auc_va) else np.nan

    # Peakiness (too many extremes)
    extreme_frac = float(((p_va < 0.35) | (p_va > 0.65)).mean())

    # Simple ECE
    def _ece(y_true, p, bins=10, w=None):
        y = np.asarray(y_true, float)
        p = np.asarray(p, float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        idx = np.digitize(p, edges) - 1
        ece = 0.0
        N = len(p)
        w = np.ones_like(p) if w is None else np.asarray(w, float)
        w = w / max(w.sum(), 1e-12)
        for b in range(bins):
            m = (idx == b)
            if m.any():
                conf = float(np.average(p[m], weights=w[m]))
                acc  = float(np.average(y[m], weights=w[m]))
                ece += float(w[m].sum()) * abs(acc - conf)
        return float(ece)

    ece_va = _ece(y_va, p_va, bins=bins, w=w_va)

    # Return all
    return dict(
        auc_tr=auc_tr, auc_va=auc_va, auc_gap=auc_gap,
        extreme_frac=extreme_frac, ece_va=ece_va
    )

def decide_harden_level(sig: dict) -> int:
    """
    0 = no hardening, 1 = moderate, 2 = aggressive
    Thresholds chosen to be conservative; tune if needed.
    """
    auc_gap      = sig.get("auc_gap", np.nan)
    extreme_frac = sig.get("extreme_frac", 0.0)
    ece_va       = sig.get("ece_va", 0.0)

    # Aggressive if any very large issues
    if (np.isfinite(auc_gap) and auc_gap > 0.20) or extreme_frac > 0.70 or ece_va > 0.20:
        return 2
    # Moderate if moderate issues
    if (np.isfinite(auc_gap) and auc_gap > 0.12) or extreme_frac > 0.55 or ece_va > 0.12:
        return 1
    return 0

def harden_params(bp: dict, level: int) -> dict:
    """Return a hardened copy of params depending on level."""
    if level <= 0:
        return dict(bp)

    out = dict(bp)
    # Base clamps
    out["max_leaves"]       = int(min(out.get("max_leaves", 128), 96 if level==1 else 64))
    out["min_child_weight"] = float(max(float(out.get("min_child_weight", 8.0)), 16.0 if level==1 else 24.0))
    out["gamma"]            = float(max(float(out.get("gamma", 2.0)), 8.0 if level==1 else 12.0))
    out["reg_alpha"]        = float(max(float(out.get("reg_alpha", 0.10)), 0.20 if level==1 else 0.40))
    out["reg_lambda"]       = float(max(float(out.get("reg_lambda", 15.0)), 20.0 if level==1 else 32.0))
    out["subsample"]        = float(min(float(out.get("subsample", 0.85)), 0.75 if level==1 else 0.65))
    out["colsample_bytree"] = float(min(float(out.get("colsample_bytree", 0.80)), 0.65 if level==1 else 0.55))
    # bynode/bylevel handled later by your version-safe logic
    out["learning_rate"]    = float(min(float(out.get("learning_rate", 0.02)), 0.015 if level==1 else 0.010))
    out["max_bin"]          = int(min(int(out.get("max_bin", 256)), 256))  # stay tight
    return out




# --- Targets (tune if needed) ---
IDEAL = dict(
    auc_gap_max      = 0.08,   # train - val AUC gap
    extreme_frac_max = 0.50,   # share of p in (<0.35 or >0.65)
    ece_max          = 0.10,   # simple ECE
)

@dataclass
class OverfitSignals:
    auc_tr: float
    auc_va: float
    auc_gap: float
    extreme_frac: float
    ece_va: float

def _signals_ok(sig: dict, ideal=IDEAL) -> bool:
    ok = True
    if np.isfinite(sig.get("auc_gap", np.inf)):
        ok &= (sig["auc_gap"] <= ideal["auc_gap_max"])
    ok &= (sig["extreme_frac"] <= ideal["extreme_frac_max"])
    ok &= (sig["ece_va"] <= ideal["ece_max"])
    return bool(ok)

def _level_for(sig: dict) -> int:
    """0=ok, 1=moderate harden, 2=aggressive harden."""
    auc_gap      = sig.get("auc_gap", np.nan)
    extreme_frac = sig.get("extreme_frac", 0.0)
    ece_va       = sig.get("ece_va", 0.0)
    if (np.isfinite(auc_gap) and auc_gap > 0.20) or extreme_frac > 0.70 or ece_va > 0.20:
        return 2
    if (np.isfinite(auc_gap) and auc_gap > 0.12) or extreme_frac > 0.55 or ece_va > 0.12:
        return 1
    return 0

def _apply_learning_rate_floor(lr: float, level: int) -> float:
    # Nudge LR down as we harden; keep a sane lower bound
    if level == 1:
        return float(np.clip(lr, 0.006, 0.016))
    if level == 2:
        return float(np.clip(lr, 0.005, 0.012))
    return lr

def auto_harden_until_ok(
    *,
    X_tr_es_df, y_tr_es, w_tr_es,
    X_va_es_df, y_va_es, w_va_es,
    base_kwargs: dict,
    start_params: dict,
    start_cap: int,
    start_lr: float,
    early_stop_default: int,
    scale_pos_weight: float,
    max_loops: int = 3,
    verbose: bool = True,
):
    """
    Iteratively fit a probe model on the ES split, measure overfit signals, and
    harden params/cap/learning_rate until signals meet IDEAL thresholds or loops exhausted.
    Returns: final_params, final_cap, final_lr, final_es_rounds, signals_history, probe_model
    """
    params = dict(start_params)
    cap    = int(start_cap)
    lr     = float(start_lr)
    es0    = int(early_stop_default)

    history: list[dict] = []
    model = None

    for loop in range(max_loops + 1):
        # Instantiate fresh probe model
        probe = XGBClassifier(**{**base_kwargs, **params})
        probe.set_params(
            n_estimators=cap,
            learning_rate=lr,
            eval_metric="logloss",
            n_jobs=max(1, int(base_kwargs.get("n_jobs", 1))),
            scale_pos_weight=float(scale_pos_weight),
            random_state=1337 + loop, seed=1337 + loop,
        )

        probe.fit(
            X_tr_es_df, y_tr_es,
            sample_weight=w_tr_es,
            eval_set=[(X_va_es_df, y_va_es)],
            sample_weight_eval_set=[w_va_es],
            verbose=False,
            early_stopping_rounds=es0 if loop == 0 else int(max(50, es0)),
        )

        # Predictions
        p_tr = np.clip(probe.predict_proba(X_tr_es_df)[:, 1], 1e-12, 1-1e-12)
        p_va = np.clip(probe.predict_proba(X_va_es_df)[:, 1], 1e-12, 1-1e-12)

        # Signals (uses your existing utility)
        sig = compute_overfit_signals(
            y_tr_es.astype(int), p_tr, y_va_es.astype(int), p_va, w_tr_es, w_va_es, bins=10
        )

        # Record
        rec = dict(loop=loop, cap=int(cap), lr=float(lr), **sig)
        history.append(rec)
        if verbose:
            try:
                st.json({"auto_harden_probe": rec})
            except Exception:
                pass  # not in Streamlit

        # Check stop condition
        if _signals_ok(sig, IDEAL):
            model = probe
            break

        # Decide hardening level and tighten
        level = _level_for(sig)
        if level == 0 and not _signals_ok(sig, IDEAL):
            # Slightly below thresholds but not ideal: treat as moderate
            level = 1

        # Harden params
        params = harden_params(params, level)

        # Tighten capacity / early stop
        cap, es_tuned = tighten_capacity(cap, level)
        es0 = int(es_tuned)

        # Reduce LR a bit
        lr = _apply_learning_rate_floor(lr, level)

        # Optional: add a mild dropout effect via sampling if still too peaky
        if level >= 1:
            params["subsample"]        = float(min(params.get("subsample", 0.8),        0.70 if level==1 else 0.60))
            params["colsample_bytree"] = float(min(params.get("colsample_bytree", 0.7), 0.60 if level==1 else 0.50))
            # bynode handled by your version-safe stabilizer (keep as-is elsewhere)

        model = probe  # keep latest; loop continues

    return params, int(cap), float(lr), int(es0), history, model

def tighten_capacity(final_estimators_cap: int, level: int) -> tuple[int,int]:
    """Scale n_estimators cap and early stopping given hardening level."""
    cap = int(final_estimators_cap)
    if level == 1:
        cap = int(np.clip(0.85 * cap, 500, 1000))
        es  = int(np.clip(0.10 * cap, 60, 150))
    elif level == 2:
        cap = int(np.clip(0.70 * cap, 400, 900))
        es  = int(np.clip(0.08 * cap, 50, 120))
    else:
        es  = int(np.clip(0.12 * cap, 60, 180))
    return cap, es

def pick_blend_weight_on_oof(
    y_oof, p_oof_auc, p_oof_log=None, grid=None, eps=1e-4,
    metric="logloss", hybrid_alpha=0.9
):


    # --- AUC-only path ---
    if p_oof_log is None:
        (y, b), _ = _clean_probs(y_oof, p_oof_auc, eps=eps)
        # polarity lock
        auc_raw, ok = _auc_safe(y, b)
        auc_flip, _ = _auc_safe(y, 1.0 - b)
        flipped = ok and (np.nan_to_num(auc_flip) > np.nan_to_num(auc_raw))
        if flipped:
            b = 1.0 - b
        return 0.0, b, flipped

    # --- Two-model path ---
    (y, a, b), _ = _clean_probs(y_oof, p_oof_log, p_oof_auc, eps=eps)

    # polarity lock on mean stream
    mix = 0.5 * (a + b)
    auc_raw, ok = _auc_safe(y, mix)
    auc_flip, _ = _auc_safe(y, 1.0 - mix)
    flipped = ok and (np.nan_to_num(auc_flip) > np.nan_to_num(auc_raw))
    if flipped:
        a, b = 1.0 - a, 1.0 - b

    if grid is None:
        grid = np.round(np.linspace(0.20, 0.80, 13), 2)

    best_w, best_score = None, +1e18
    for w in grid:
        p = np.clip(w*a + (1-w)*b, eps, 1-eps)
        if metric == "logloss":
            score = log_loss(y, p, labels=[0, 1])
        else:
            auc, _ = _auc_safe(y, p)
            score = hybrid_alpha*log_loss(y, p, labels=[0,1]) + (1-hybrid_alpha)*(1.0 - np.nan_to_num(auc))
        if score < best_score:
            best_score, best_w = score, float(w)

    p_oof_blend = np.clip(best_w*a + (1-best_w)*b, eps, 1-eps)
    return best_w, p_oof_blend, flipped


def _bake_feature_names_in_(est, cols):
    """
    Best-effort: stamp the exact training feature names & width onto an estimator,
    its inner booster (xgboost), the last step of a Pipeline, and common wrappers.
    """
    if est is None or cols is None:
        return

    # 1) normalize list: strings, de-duped, order-preserved
    cols = [str(c) for c in cols if c is not None]
    cols = list(dict.fromkeys(cols))
    cols_arr = np.asarray(cols, dtype=object)
    n = len(cols)

    def _try_set(obj, attr, value):
        try:
            setattr(obj, attr, value)
            return True
        except Exception:
            return False

    # 2) set on the estimator itself
    _try_set(est, "feature_names_in_", cols_arr)
    # sklearn sometimes reads this; not required but helpful for sanity
    _try_set(est, "n_features_in_", n)

    # 3) xgboost: propagate to Booster if available
    try:
        booster = est.get_booster() if hasattr(est, "get_booster") else None
        if booster is not None:
            # xgboost uses a Python attribute 'feature_names' during predict/plot
            _try_set(booster, "feature_names", list(cols))
    except Exception:
        pass

    # 4) sklearn Pipeline variants
    # prefer named_steps, else look at generic steps
    try:
        if hasattr(est, "named_steps") and est.named_steps:
            last = list(est.named_steps.values())[-1]
            _try_set(last, "feature_names_in_", cols_arr)
            _try_set(last, "n_features_in_", n)
            if hasattr(last, "get_booster"):
                b2 = last.get_booster()
                if b2 is not None:
                    _try_set(b2, "feature_names", list(cols))
        elif hasattr(est, "steps") and est.steps:
            last = est.steps[-1][1]
            _try_set(last, "feature_names_in_", cols_arr)
            _try_set(last, "n_features_in_", n)
            if hasattr(last, "get_booster"):
                b2 = last.get_booster()
                if b2 is not None:
                    _try_set(b2, "feature_names", list(cols))
    except Exception:
        pass

    # 5) common wrappers: CalibratedClassifierCV, custom IsoWrapper, etc.
    for inner_attr in ("base_estimator", "estimator", "classifier", "model"):
        inner = getattr(est, inner_attr, None)
        if inner is not None and inner is not est:
            _try_set(inner, "feature_names_in_", cols_arr)
            _try_set(inner, "n_features_in_", n)
            try:
                b3 = inner.get_booster() if hasattr(inner, "get_booster") else None
                if b3 is not None:
                    _try_set(b3, "feature_names", list(cols))
            except Exception:
                pass

    # 6) LightGBM / CatBoost: expose friendly attributes when present
    # (These are mostly for tooling; many libs don’t *read* them but it helps debugging.)
    for attr in ("feature_name_", "feature_names_"):
        _try_set(est, attr, list(cols))

    # 7) if a bundle dict slipped in here, stamp all known slots
    if isinstance(est, dict):
        for k in ("model_logloss", "model_auc", "model", "calibrator_logloss", "calibrator_auc"):
            _bake_feature_names_in_(est.get(k), cols)


# Compact key levels tuned per sport/market (extend as you like)
TRAIN_KEY_LEVELS = {
    ("nfl","spreads"):  [1.5,2.5,3,3.5,6,6.5,7,7.5,9.5,10,10.5,13.5,14],
    ("ncaaf","spreads"):[1.5,2.5,3,3.5,6,6.5,7,7.5,9.5,10,10.5,13.5,14],
    ("cfl","spreads"):  [1.5,2.5,3,3.5,6,6.5,7,9.5,10,10.5],
    ("nba","spreads"):  [1.5,2.5,3,4.5,5.5,6.5,7.5,9.5],
    ("wnba","spreads"): [1.5,2.5,3,4.5,5.5,6.5,7.5],
    ("mlb","totals"):   [7,7.5,8,8.5,9,9.5,10],
    ("nfl","totals"):   [41,43,44.5,47,49.5,51,52.5],
    ("ncaaf","totals"): [49.5,52.5,55.5,57.5,59.5,61.5],
    ("nba","totals"):   [210,212.5,215,217.5,220,222.5,225],
    ("wnba","totals"):  [158.5,160.5,162.5,164.5,166.5,168.5],
    ("cfl","totals"):   [44.5,46.5,48.5,50.5,52.5],
    ("ncaab","spreads"):  [1.5,2.5,3,4.5,5.5,6.5,7.5,9.5],
}

def _keys_for_training(sport: str, market: str) -> np.ndarray:
    s = (sport or "").strip().lower()
    m = (market or "").strip().lower()
    ks = TRAIN_KEY_LEVELS.get((s, m))
    if ks is None:
        ks = [1.5,2.5,3,3.5,6,6.5,7] if m == "spreads" else ([7,7.5,8,8.5,9] if m=="totals" else [])
    return np.asarray(ks, dtype=np.float32)

def add_resistance_features_training(
    df: pd.DataFrame,
    *,
    sport_col: str = "Sport",
    market_col: str = "Market",
    value_candidates = ("Value","Line_Value","Line"),
    odds_candidates  = ("Odds_Price","Odds","Price"),
    open_value_candidates = ("First_Line_Value","Open_Value","Open_Line","Line_Open","Opening_Line"),
    open_odds_candidates  = ("First_Odds","Open_Odds","Open_Odds_Price","Odds_Open","Opening_Odds"),
    emit_levels_str: bool = False,
) -> pd.DataFrame:
    """
    Training-only, low-memory resistance features.
    • No timestamp sorting or history needed.
    • One row per (Game_Key, Market, Outcome, Bookmaker) expected.
    • Computes:
        - Line_Resistance_Crossed_Count (int16)
        - Was_Line_Resistance_Broken    (uint8)
        - SharpMove_Resistance_Break    (uint8)
      (and optionally Line_Resistance_Crossed_Levels_Str)
    """
    out = df.copy()
    if out.empty:
        out["Line_Resistance_Crossed_Count"] = 0
        out["Was_Line_Resistance_Broken"] = 0
        out["SharpMove_Resistance_Break"] = 0
        if emit_levels_str:
            out["Line_Resistance_Crossed_Levels_Str"] = ""
        return out

    # 0) ensure one row per key (belt & suspenders)
    keys = [c for c in ["Game_Key","Market","Outcome","Bookmaker"] if c in out.columns]
    if keys:
        out = out.drop_duplicates(keys, keep="last")

    # 1) resolve columns by common aliases
    def _resolve(cands, fallback=None):
        for c in cands:
            if c in out.columns:
                return c
        return fallback

    value_col = _resolve(value_candidates, fallback="Value")
    odds_col  = _resolve(odds_candidates,  fallback="Odds_Price")
    open_val  = _resolve(open_value_candidates, fallback=None)
    open_odds = _resolve(open_odds_candidates,  fallback=None)

    # hydrate opens if missing
    if open_val is None:
        out["First_Line_Value"] = out[value_col]
        open_val = "First_Line_Value"
    if open_odds is None:
        out["First_Odds"] = out.get(odds_col, np.nan)
        open_odds = "First_Odds"

    # numeric coercions
    for c in (value_col, open_val, odds_col, open_odds):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # 2) normalized sport/market
    out["_sport_"]  = out.get(sport_col, "").astype(str).str.strip().str.lower()
    out["_market_"] = out.get(market_col, "").astype(str).str.strip().str.lower()

    n = len(out)
    crossed_count = np.zeros(n, dtype=np.int16)
    broken        = np.zeros(n, dtype=np.uint8)
    levels_str    = np.full(n, "", dtype=object) if emit_levels_str else None

    # 3) vectorized per (sport, market)
    for (sp, mk), idx in out.groupby(["_sport_","_market_"]).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        keys_arr = _keys_for_training(sp, mk)
        if keys_arr.size == 0:
            continue

        is_spread = (mk == "spreads")
        o = out.loc[idx, open_val].to_numpy(dtype=float)
        v = out.loc[idx, value_col].to_numpy(dtype=float)
        valid = ~np.isnan(o) & ~np.isnan(v)
        if not valid.any():
            continue

        a = np.abs(o[valid]) if is_spread else o[valid]
        b = np.abs(v[valid]) if is_spread else v[valid]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)

        lo_idx = np.searchsorted(keys_arr, lo, side="right")
        hi_idx = np.searchsorted(keys_arr, hi, side="left")
        cnt = (hi_idx - lo_idx).astype(np.int16)

        rows = idx[valid]
        crossed_count[rows] = cnt
        broken[rows] = (cnt > 0).astype(np.uint8)

        if emit_levels_str and np.any(cnt > 0):
            # Build strings only where we actually crossed keys
            kpos = np.flatnonzero(cnt > 0)
            for k in kpos:
                r = rows[k]
                s_keys = keys_arr[(keys_arr > lo[k]) & (keys_arr < hi[k])]
                if s_keys.size:
                    levels_str[r] = "|".join(str(float(x)).rstrip("0").rstrip(".") for x in s_keys)

    # 4) assign outputs (no merges, so no KeyErrors)
    out["Line_Resistance_Crossed_Count"] = crossed_count
    out["Was_Line_Resistance_Broken"] = broken
    if emit_levels_str:
        out["Line_Resistance_Crossed_Levels_Str"] = levels_str if levels_str is not None else ""

    # 5) tie to sharp signal if available
    if "Sharp_Move_Signal" in out.columns:
        sm = pd.to_numeric(out["Sharp_Move_Signal"], errors="coerce").fillna(0).astype(np.int8)
        out["SharpMove_Resistance_Break"] = ((out["Was_Line_Resistance_Broken"] == 1) & (sm == 1)).astype(np.uint8)
    elif "Sharp_Prob_Shift" in out.columns:
        ps = pd.to_numeric(out["Sharp_Prob_Shift"], errors="coerce").fillna(0.0).astype("float32")
        out["SharpMove_Resistance_Break"] = ((out["Was_Line_Resistance_Broken"] == 1) & (ps > 0)).astype(np.uint8)
    else:
        out["SharpMove_Resistance_Break"] = out["Was_Line_Resistance_Broken"].astype(np.uint8)

    # final cup
    out.drop(columns=["_sport_","_market_"], inplace=True, errors="ignore")
    return out





def _amer_to_prob(o):
    o = pd.to_numeric(o, errors="coerce")
    return np.where(o > 0, 100.0 / (o + 100.0),
           np.where(o < 0, (-o) / ((-o) + 100.0), np.nan))

def _dist_to_next_key_training(x, keys_arr, is_spread: bool):
    if pd.isna(x) or keys_arr.size == 0: return np.nan
    v = abs(float(x)) if is_spread else float(x)
    pos = np.searchsorted(keys_arr, v, side="left")
    if pos >= len(keys_arr): return 0.0
    return float(max(0.0, keys_arr[pos] - v))

def compute_snapshot_micro_features_training(
    df: pd.DataFrame,
    *,
    sport_col="Sport",
    market_col="Market",
    value_col="Value",
    open_col="First_Line_Value",
    prob_col="Implied_Prob",
    price_col="Odds_Price",
) -> pd.DataFrame:
    """
    Emits per-row (BQ-safe) scalars based on the final snapshot only:
      - Implied_Hold_Book, Two_Sided_Offered, Juice_Abs_Delta
      - Dist_To_Next_Key, Key_Corridor_Pressure
      - Book_PctRank_Line, Book_Line_Diff_vs_SharpMedian, Outlier_Flag_SharpBooks
    """
    if df.empty: 
        return df

    out = df.copy()
    out["_sport_"]  = out[sport_col].astype(str).str.lower().str.strip()
    out["_market_"] = out[market_col].astype(str).str.lower().str.strip()
    out["_book_"]   = out["Bookmaker"].astype(str).str.lower().str.strip()

    # per-outcome slice (your training table is already final snapshot)
    cols = ["Game_Key","Market","Bookmaker","Outcome", value_col, open_col, "_sport_","_market_","_book_"]
    if prob_col in out.columns:  cols.append(prob_col)
    if price_col in out.columns: cols.append(price_col)
    per_outcome = out[cols].copy()

    g = per_outcome.groupby(["Game_Key","Market","Bookmaker"], sort=False)

    # Hold & juice
    if prob_col in per_outcome.columns:
        hold = g[prob_col].sum().rename("Implied_Hold_Book")
    elif price_col in per_outcome.columns:
        per_outcome["_p_"] = _amer_to_prob(per_outcome[price_col])
        hold = g["_p_"].sum().rename("Implied_Hold_Book")
    else:
        hold = pd.Series(dtype="float32", name="Implied_Hold_Book")

    if price_col in per_outcome.columns:
        pm = g[price_col].agg(["max","min"])
        juice_abs = (pm["max"] - pm["min"]).astype("float32").rename("Juice_Abs_Delta")
    else:
        juice_abs = pd.Series(dtype="float32", name="Juice_Abs_Delta")

    two_sided = g["Outcome"].nunique().rename("Two_Sided_Offered").astype(np.uint8)

    # 1 row per (G,M,B)
    agg = g.agg(
        sport = ("_sport_", "last"),
        market= ("_market_","last"),
        v_last= (value_col, "last"),
        v_open= (open_col,  "last"),
    ).join([hold, juice_abs, two_sided]).reset_index()

    # Dist / Pressure (uses your training key levels)
    from math import isfinite  # just in case
    v      = pd.to_numeric(agg["v_last"], errors="coerce").astype("float32")
    v_open = pd.to_numeric(agg["v_open"], errors="coerce").astype("float32")
    dist   = np.full(len(agg), np.nan, dtype="float32")

    for (mk, sp), idx in agg.groupby(["market","sport"]).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        keys_arr = _keys_for_training(sp, mk)  # <- your registry
        is_spread = (mk == "spreads")
        for j in idx:
            dist[j] = _dist_to_next_key_training(v.iloc[j], keys_arr, is_spread)

    agg["Dist_To_Next_Key"] = dist
    cur = np.where(agg["market"].values == "spreads", np.abs(v.values), v.values)
    opn = np.where(agg["market"].values == "spreads", np.abs(v_open.values), v_open.values)
    dv  = np.abs(cur - opn)
    den = np.where(np.isnan(dist) | (dist <= 0), 1.0, dist)
    agg["Key_Corridor_Pressure"] = (dv / den).astype("float32")

    # Consensus / outlier (final snapshot)
    val_norm = np.where(agg["market"].values == "spreads", np.abs(v.values), v.values).astype("float32")
    agg["_val_norm_"] = val_norm
    agg["Book_PctRank_Line"] = (
        agg.groupby(["Game_Key","Market"])["_val_norm_"].rank(pct=True, method="average").astype("float32")
    )

    # sharp-median diff
    try:
        from config import SHARP_BOOKS
    except Exception:
        SHARP_BOOKS = set(["pinnacle","betonlineag","lowvig","betfair_ex_uk","betfair_ex_eu","smarkets","betus"])
    mask_sharp = agg["Bookmaker"].astype(str).str.lower().isin(SHARP_BOOKS)
    sharp_median = agg.loc[mask_sharp].groupby(["Game_Key","Market"])["_val_norm_"].median()
    agg = agg.merge(sharp_median.rename("sharp_median"), on=["Game_Key","Market"], how="left")
    agg["Book_Line_Diff_vs_SharpMedian"] = (
        agg["_val_norm_"] - agg["sharp_median"].fillna(agg["_val_norm_"])
    ).astype("float32")

    # Outlier flag (sport-aware thresholds)
    OUTLIER_THRESH_SPREAD, OUTLIER_THRESH_TOTAL = 0.5, 0.5
    thr = np.where(agg["market"].values == "spreads", OUTLIER_THRESH_SPREAD, OUTLIER_THRESH_TOTAL).astype("float32")
    agg["Outlier_Flag_SharpBooks"] = (np.abs(agg["Book_Line_Diff_vs_SharpMedian"].values) >= thr).astype("uint8")

    # Broadcast back to rows
    keep = [
        "Game_Key","Market","Bookmaker",
        "Implied_Hold_Book","Two_Sided_Offered","Juice_Abs_Delta",
        "Dist_To_Next_Key","Key_Corridor_Pressure",
        "Book_PctRank_Line","Book_Line_Diff_vs_SharpMedian","Outlier_Flag_SharpBooks",
    ]
    out = out.merge(agg[keep], on=["Game_Key","Market","Bookmaker"], how="left", copy=False)

    # Fill / downcast
    u8  = ["Two_Sided_Offered","Outlier_Flag_SharpBooks"]
    f32 = ["Implied_Hold_Book","Juice_Abs_Delta","Dist_To_Next_Key","Key_Corridor_Pressure",
           "Book_PctRank_Line","Book_Line_Diff_vs_SharpMedian"]
    for c in u8:  out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0).astype("uint8")
    for c in f32: out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0.0).astype("float32")

    out.drop(columns=["_sport_","_market_","_book_"], errors="ignore", inplace=True)
    return out


def compute_hybrid_timing_derivatives_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds timing magnitude/split/entropy features from your hybrid timing columns.
    Auto-detects columns by prefix.
    """
    if df.empty:
        return df

    out = df.copy()
    n = len(out)

    # Detect column sets
    line_cols = [c for c in out.columns if c.startswith("SharpMove_Magnitude_")]
    odds_cols = [c for c in out.columns if c.startswith("OddsMove_Magnitude_")]

    # Fallbacks (append, don’t overwrite)
    if not line_cols:
        line_cols += [c for c in out.columns
                      if ("line" in c.lower() or "value" in c.lower())
                      and "magnitude" in c.lower()]
    if not odds_cols:
        odds_cols += [c for c in out.columns
                      if ("odds" in c.lower() or "price" in c.lower() or "prob" in c.lower())
                      and "magnitude" in c.lower()]

    def _pick(cols, pats):
        pats = tuple(p.lower() for p in pats)
        return [c for c in cols if any(p in c.lower() for p in pats)]

    early_l = _pick(line_cols, ("overnight","veryearly","early"))
    mid_l   = _pick(line_cols, ("midday","midrange"))
    late_l  = _pick(line_cols, ("late","urgent"))

    early_o = _pick(odds_cols, ("overnight","veryearly","early"))
    mid_o   = _pick(odds_cols, ("midday","midrange"))
    late_o  = _pick(odds_cols, ("late","urgent"))

    def _zeros():
        # vector of zeros aligned to DF rows, float32
        return pd.Series(0.0, index=out.index, dtype="float32")

    # Sums (vectorized) — float32
    out["Hybrid_Line_TotalMag"] = (out[line_cols].sum(axis=1).astype("float32") if line_cols else _zeros())
    out["Hybrid_Line_EarlyMag"] = (out[early_l].sum(axis=1).astype("float32")   if early_l   else _zeros())
    out["Hybrid_Line_MidMag"]   = (out[mid_l].sum(axis=1).astype("float32")     if mid_l     else _zeros())
    out["Hybrid_Line_LateMag"]  = (out[late_l].sum(axis=1).astype("float32")    if late_l    else _zeros())

    out["Hybrid_Odds_TotalMag"] = (out[odds_cols].sum(axis=1).astype("float32") if odds_cols else _zeros())
    out["Hybrid_Odds_EarlyMag"] = (out[early_o].sum(axis=1).astype("float32")   if early_o   else _zeros())
    out["Hybrid_Odds_MidMag"]   = (out[mid_o].sum(axis=1).astype("float32")     if mid_o     else _zeros())
    out["Hybrid_Odds_LateMag"]  = (out[late_o].sum(axis=1).astype("float32")    if late_o    else _zeros())

    # Shares / ratios
    eps = np.float32(1e-6)
    out["Hybrid_Line_LateShare"]  = (out["Hybrid_Line_LateMag"]  / (out["Hybrid_Line_TotalMag"] + eps)).astype("float32")
    out["Hybrid_Line_EarlyShare"] = (out["Hybrid_Line_EarlyMag"] / (out["Hybrid_Line_TotalMag"] + eps)).astype("float32")
    out["Hybrid_Odds_LateShare"]  = (out["Hybrid_Odds_LateMag"]  / (out["Hybrid_Odds_TotalMag"] + eps)).astype("float32")
    out["Hybrid_Odds_EarlyShare"] = (out["Hybrid_Odds_EarlyMag"] / (out["Hybrid_Odds_TotalMag"] + eps)).astype("float32")
    out["Hybrid_Line_Odds_Mag_Ratio"] = (out["Hybrid_Line_TotalMag"] / (out["Hybrid_Odds_TotalMag"] + eps)).astype("float32")

    # Imbalance
    out["Hybrid_Line_Imbalance_LateVsEarly"] = (
        (out["Hybrid_Line_LateMag"] - out["Hybrid_Line_EarlyMag"]) /
        (out["Hybrid_Line_TotalMag"] + eps)
    ).astype("float32")

    # Entropy (only if cols exist)
    _line_cols = [c for c in (line_cols or []) if c in out.columns]
    _odds_cols = [c for c in (odds_cols or []) if c in out.columns]

    if _line_cols:
        W = out[_line_cols].to_numpy(dtype="float32")
        out["Hybrid_Timing_Entropy_Line"] = safe_row_entropy(W).astype("float32")
    else:
        out["Hybrid_Timing_Entropy_Line"] = np.float32(0.0)

    if _odds_cols:
        W2 = out[_odds_cols].to_numpy(dtype="float32")
        out["Hybrid_Timing_Entropy_Odds"] = safe_row_entropy(W2).astype("float32")
    else:
        out["Hybrid_Timing_Entropy_Odds"] = np.float32(0.0)

    # Ensure absolute-from-open columns exist & are float32
    for c in ("Abs_Line_Move_From_Opening", "Abs_Odds_Move_From_Opening"):
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype("float32")

    # Interactions with snapshot microstructure
    if "Key_Corridor_Pressure" in out.columns:
        out["Corridor_x_LateShare_Line"] = (
            out["Key_Corridor_Pressure"].astype("float32") * out["Hybrid_Line_LateShare"]
        ).astype("float32")
    if "Dist_To_Next_Key" in out.columns:
        out["Dist_x_LateShare_Line"] = (
            out["Dist_To_Next_Key"].astype("float32") * out["Hybrid_Line_LateShare"]
        ).astype("float32")
    if "Book_PctRank_Line" in out.columns:
        out["PctRank_x_LateShare_Line"] = (
            out["Book_PctRank_Line"].astype("float32") * out["Hybrid_Line_LateShare"]
        ).astype("float32")

    return out  # ← REQUIRED


# --- Drop-in safety for calibration ------------------------------------------------


from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def _safe_clean_pred_labels(p: np.ndarray, y: np.ndarray):
    """Clean and validate predictions+labels for calibration."""
    p = np.asarray(p, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Hard clamp to [0, 1] and remove non-finite
    p = np.clip(p, 0.0, 1.0)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]

    # Coerce labels to {0,1}; drop anything else
    # (handles cases where y has NaN or floats like 0.0/1.0)
    y_bin_mask = (y == 0) | (y == 1)
    p, y = p[y_bin_mask], y[y_bin_mask]

    return p, y

def fit_regularized_isotonic(p, y, q=80, eps=1e-6):
    """
    Regularized isotonic: bin by quantiles, average, drop degenerate/NaN bins,
    then fit monotone mapping. Falls back to Platt or identity if needed.
    """
    p, y = _safe_clean_pred_labels(p, y)

    # Not enough data after cleaning → identity
    if p.size < 10 or np.unique(p).size < 2:
        return ("iso", _IdentityCal())  # keep API: (kind, model)

    # Build bins by quantiles; protect against duplicate edges
    try:
        q = int(q)
        q = max(10, min(q, max(10, p.size // 5)))
        qs = np.linspace(0, 1, q + 1)
        edges = np.unique(np.quantile(p, qs, method="linear"))
    except Exception:
        # Fallback if quantile fails
        edges = np.unique(np.linspace(p.min(), p.max(), 21))

    # Assign bins; if edges collapse, bail to Platt or identity
    if edges.size < 3:
        # Try Platt (logistic on raw p)
        try:
            pl = LogisticRegression(max_iter=500)
            pl.fit(p.reshape(-1, 1), y)
            return ("platt", pl)
        except Exception:
            return ("iso", _IdentityCal())

    # pd.cut can yield NaN bin for max; include_right=True-like behavior
    # by nudging the last edge
    edges[-1] = np.nextafter(edges[-1], np.float64(np.inf))
    b = pd.cut(p, bins=edges, labels=False, include_lowest=True)

    # Aggregate per bin
    by = (
        pd.DataFrame({"p": p, "y": y, "b": b})
        .groupby("b", dropna=True)
        .agg(p_mean=("p", "mean"), y_mean=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )

    # Drop bins that are NaN/inf or too small
    good = (
        np.isfinite(by["p_mean"]) &
        np.isfinite(by["y_mean"]) &
        (by["n"] >= 2)
    )
    by = by.loc[good].copy()

    # Add tiny regularization to prevent flat ties causing issues
    if not by.empty:
        jitter = eps * np.arange(len(by), dtype=float)
        by["p_mean"] = by["p_mean"].to_numpy(dtype=float) + jitter

    # Need at least 2 ascending x points for isotonic
    if by.shape[0] < 2 or np.unique(by["p_mean"]).size < 2:
        # Try Platt; if even that fails, identity
        try:
            pl = LogisticRegression(max_iter=500)
            pl.fit(p.reshape(-1, 1), y)
            return ("platt", pl)
        except Exception:
            return ("iso", _IdentityCal())

    # Fit isotonic on (p_mean -> y_mean)
    iso = IsotonicRegression(y_min=eps, y_max=1 - eps, increasing=True, out_of_bounds="clip")
    iso.fit(by["p_mean"].to_numpy(), by["y_mean"].to_numpy())
    return ("iso", iso)

class _IdentityCal:
    """Identity calibrator with scikit-learn-like interface."""
    def transform(self, p):
        p = np.asarray(p, dtype=float).reshape(-1)
        p = np.clip(p, 0.0, 1.0)
        return p

# --- Wrap your entry function to clean first --------------------------------------
def fit_iso_platt_beta(p, y, eps=1e-6, use_quantile_iso=True):
    """
    Returns a tuple like ("iso"|... , model) chosen by your logic.
    This wrapper now ensures inputs are clean to avoid NaN errors.
    """
    p, y = _safe_clean_pred_labels(p, y)

    # Guard: no class variety → identity (or a fixed clip)
    if p.size < 2 or np.unique(y).size < 2:
        return ("iso", _IdentityCal())

    # Your prior selection logic can remain; ensure each candidate uses cleaned (p,y)
    if use_quantile_iso:
        return fit_regularized_isotonic(p, y, q=80, eps=eps)

    # Example: try Platt as alternative path
    try:
        pl = LogisticRegression(max_iter=500)
        pl.fit(p.reshape(-1, 1), y)
        return ("platt", pl)
    except Exception:
        return ("iso", _IdentityCal())

def _apply_cal(kind, cal, x):
    if kind == "iso":
        return cal.transform(x)
    if kind == "platt":
        return cal.predict_proba(x.reshape(-1,1))[:,1]
    if kind == "beta":
        return cal.predict(x.reshape(-1,1))
    raise ValueError(kind)

def select_blend(cals, p_oof, y_oof, eps=1e-4):
    from sklearn.metrics import log_loss
    x = np.clip(np.asarray(p_oof, float), eps, 1-eps)
    y = np.asarray(y_oof,   int)

    kinds = ["iso", "platt"] + (["beta"] if cals["beta"] is not None else [])
    best = None
    # Try convex blends: p = a*iso + (1-a)*base, a∈[0..1]
    for base in kinds:
        p_iso  = _apply_cal("iso",  cals["iso"],  x)
        p_base = _apply_cal(base,   cals[base],   x)
        for a in np.linspace(0.0, 1.0, 11):  # 0.0, 0.1, ..., 1.0
            p_blend = np.clip(a*p_iso + (1-a)*p_base, eps, 1-eps)
            score   = log_loss(y, p_blend, labels=[0,1])
            if (best is None) or (score < best[0]):
                best = (score, base, float(a))
    _, base, alpha = best
    return {"cals": cals, "base": base, "alpha": alpha}



# extra_plumbing.py
import os, numpy as np, pandas as pd

# -------------------------
# Shared helpers
# -------------------------
def _amer_to_prob(o):
    o = pd.to_numeric(o, errors="coerce")
    return np.where(o > 0, 100.0 / (o + 100.0),
           np.where(o < 0, (-o) / ((-o) + 100.0), np.nan))

def _prob_to_amer(p):
    p = np.clip(pd.to_numeric(p, errors="coerce"), 1e-6, 1-1e-6)
    pos = 100 * (1 - p) / p
    neg = -100 * p / (1 - p)
    return np.where(p >= 0.5, neg, pos)

# Heuristic scoring-volatility by sport (used for spread↔ML conversions; tune over time)
SIGMA_ML = { "NFL": 10.5, "NCAAF": 12.0, "NBA": 12.0, "WNBA": 10.0, "CFL": 11.5, "MLB": 3.8 }
# keep your imports:


def _norm_cdf(x):
    """Standard normal CDF; accepts scalar or ndarray."""
    x = np.asarray(x, dtype="float64")
    # vectorize math.erf so arrays are OK
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))  # NOTE: np.sqrt, not math.sqrt

def _sigma_lookup(sp):
    """Get sigma (points) for a sport from whichever mapping you have."""
    sp_u = str(sp).upper()
    if "SPORT_SPREAD_CFG" in globals() and sp_u in SPORT_SPREAD_CFG:
        return float(SPORT_SPREAD_CFG[sp_u]["sigma_pts"])
    if "SIGMA_ML" in globals() and sp_u in SIGMA_ML:
        return float(SIGMA_ML[sp_u])
    return 13.5  # sensible default

def _spread_to_winprob(spread_pts, sport="NFL"):
    """
    P(favorite wins) from spread magnitude using Φ(|spread| / σ).
    - spread_pts: scalar/array-like; can be signed or absolute
    - sport: single string or array-like matching spread length
    Returns ndarray.
    """
    spread = np.asarray(pd.to_numeric(spread_pts, errors="coerce"), dtype="float64")
    # sport can be a single string or a vector (same length as spread)
    if np.ndim(sport) == 0:
        sigma = float(_sigma_lookup(sport))
    else:
        sport_arr = np.asarray(sport)
        sigma = np.array([_sigma_lookup(sp) for sp in sport_arr], dtype="float64")

    # use magnitude; negative spread just means "favorite"
    z = np.divide(np.abs(spread), sigma, out=np.full_like(spread, np.nan), where=(np.asarray(sigma) > 0))
    return _norm_cdf(z)

# -------------------------
# 1) Limit Dynamics (uses your existing limit columns)
# -------------------------
def add_limit_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Emits:
      - Limit_Spike_Flag  (uint8): True if Sharp_Limit_Jump >= 1 OR High_Limit_Flag == 1
      - Delta_Limit       (float32): change proxy using Sharp_Limit_Jump if total not available
      - LimitSpike_x_NoMove (float32): interaction with your LimitUp_NoMove_Flag
    """
    if df.empty: return df
    out = df.copy()
    # Coerce inputs if present
    for c in ["Sharp_Limit_Jump","High_Limit_Flag","Sharp_Limit_Total","LimitUp_NoMove_Flag","Line_Magnitude_Abs","Line_Move_Magnitude"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    jump = out.get("Sharp_Limit_Jump")
    high = out.get("High_Limit_Flag")
    out["Limit_Spike_Flag"] = (
        ((jump.fillna(0) >= 1) | (high.fillna(0) == 1)).astype("uint8")
    )

    # ΔLimit proxy: prefer diff of totals if you have both "First_Limit_Total" & "Sharp_Limit_Total"
    if "Sharp_Limit_Total" in out.columns and "First_Limit_Total" in out.columns:
        out["Delta_Limit"] = (out["Sharp_Limit_Total"] - out["First_Limit_Total"]).astype("float32")
    elif "Sharp_Limit_Jump" in out.columns:
        out["Delta_Limit"] = pd.to_numeric(out["Sharp_Limit_Jump"], errors="coerce").fillna(0).astype("float32")
    else:
        out["Delta_Limit"] = np.float32(0.0)

    # Interaction: spike but no move (you already have LimitUp_NoMove_Flag)
    nomove = out.get("LimitUp_NoMove_Flag")
    if nomove is None:
        nomove = 0
    out["LimitSpike_x_NoMove"] = (out["Limit_Spike_Flag"].astype("float32") *
                                  pd.to_numeric(nomove, errors="coerce").fillna(0).astype("float32"))

    return out

# -------------------------
# 2) Bookmaker Network Features
# -------------------------
def add_book_network_features(df: pd.DataFrame, sharp_books=None) -> pd.DataFrame:
    """
    Emits per row:
      - Sharp_Consensus_Weight (float32)  # weighted fraction of sharp books aligned with row's side
      - Sharp_vs_Rec_SpreadGap_Q90_Q10 (float32)
      - Sharp_vs_Rec_SpreadGap_Q50 (float32)
    Requires: Game_Key, Market, Bookmaker, Outcome, Value
    Optional: Book_Reliability_Score, Is_Favorite_Bet
    """
    if df.empty:
        return df

    out = df.copy()


    sharp_books = set(SHARP_BOOKS)
    # --- normalize / safe helpers
    out["Bookmaker_norm"] = out["Bookmaker"].astype(str).str.lower().str.strip()
    out["Market_norm"]    = out["Market"].astype(str).str.lower().str.strip()
    out["Outcome_Norm"]   = out["Outcome"].astype(str).str.lower().str.strip()

    # Safe favorite flag for *this row* (used later only for choosing direction)
    # Prefer explicit Is_Favorite_Bet if present, but coerce numeric + fillna(0)
    fav_num = pd.to_numeric(out.get("Is_Favorite_Bet", np.nan), errors="coerce").fillna(0.0)
    is_fav_row = (fav_num > 0.5)

    # For totals, treat "favorite" concept as N/A; we’ll use OVER/UNDER instead
    is_total_row = out["Market_norm"].eq("totals")
    is_over_row  = is_total_row & out["Outcome_Norm"].eq("over")

    # --- build base once per (G,M,B)
    base_cols = ["Game_Key","Market","Bookmaker","Bookmaker_norm","Market_norm","Outcome_Norm","Value"]
    if "Book_Reliability_Score" in out.columns:
        base_cols.append("Book_Reliability_Score")
    base = out[base_cols].drop_duplicates(["Game_Key","Market","Bookmaker"]).copy()

    # Infer book’s side on that outcome (favorite for spreads/h2h, OVER for totals)
    base_val_num = pd.to_numeric(base["Value"], errors="coerce")
    base["Is_Favorite_Bet_b"] = (base_val_num < 0) & base["Market_norm"].eq("spreads")
    base["Is_Over_Bet_b"]     = base["Outcome_Norm"].eq("over") & base["Market_norm"].eq("totals")
    base["is_sharp_book"]     = base["Bookmaker_norm"].isin(sharp_books).astype("uint8")

    def _weights(gdf):
        return pd.to_numeric(
            gdf.get("Book_Reliability_Score", pd.Series(1.0, index=gdf.index)),
            errors="coerce"
        ).fillna(1.0).astype("float32")

    # Aggregate once per (G,M)
    rows = []
    for (g, m), grp in base.groupby(["Game_Key","Market"], sort=False):
        sharp_grp = grp.loc[grp["is_sharp_book"] == 1]
        rec_grp   = grp.loc[grp["is_sharp_book"] == 0]

        w_sharp = _weights(sharp_grp)

        if str(m).lower() == "totals":
            # alignment among sharp books for OVER
            aligned_mask = sharp_grp["Is_Over_Bet_b"].astype(bool)
            over_w = w_sharp[aligned_mask].sum()
            w_tot  = float(w_sharp.sum())
            consensus_frac = float(over_w / w_tot) if w_tot > 0 else np.nan

            sharp_vals = pd.to_numeric(sharp_grp["Value"], errors="coerce").dropna()
            rec_vals   = pd.to_numeric(rec_grp["Value"], errors="coerce").dropna()
        else:
            # spreads/h2h: alignment among sharp books for FAVORITE
            aligned_mask = sharp_grp["Is_Favorite_Bet_b"].astype(bool)
            fav_w = w_sharp[aligned_mask].sum()
            w_tot = float(w_sharp.sum())
            consensus_frac = float(fav_w / w_tot) if w_tot > 0 else np.nan

            sharp_vals = pd.to_numeric(sharp_grp["Value"], errors="coerce").abs().dropna()
            rec_vals   = pd.to_numeric(rec_grp["Value"], errors="coerce").abs().dropna()

        def q(series, p):
            return float(series.quantile(p)) if len(series) else np.nan

        gap_q90_q10 = q(sharp_vals, 0.90) - q(rec_vals, 0.10) if len(sharp_vals) and len(rec_vals) else np.nan
        gap_q50     = q(sharp_vals, 0.50) - q(rec_vals, 0.50) if len(sharp_vals) and len(rec_vals) else np.nan

        rows.append({
            "Game_Key": g, "Market": m,
            "Sharp_Consensus_Frac_tmp": consensus_frac,   # interpreted per row below
            "Sharp_vs_Rec_SpreadGap_Q90_Q10": gap_q90_q10,
            "Sharp_vs_Rec_SpreadGap_Q50": gap_q50
        })

    net = pd.DataFrame(rows)

    # Merge once, then convert consensus to row-side weight:
    out = out.merge(net, on=["Game_Key","Market"], how="left")

    out["Sharp_Consensus_Weight"] = np.where(
        is_total_row,
        np.where(is_over_row, out["Sharp_Consensus_Frac_tmp"], 1.0 - out["Sharp_Consensus_Frac_tmp"]),
        # spreads/h2h: use favorite fraction; if this row is dog, flip
        np.where(is_fav_row, out["Sharp_Consensus_Frac_tmp"], 1.0 - out["Sharp_Consensus_Frac_tmp"])
    ).astype("float32")

    # Fill and cast the rest
    for c in ["Sharp_Consensus_Weight","Sharp_vs_Rec_SpreadGap_Q90_Q10","Sharp_vs_Rec_SpreadGap_Q50"]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0.0).astype("float32")

    out.drop(columns=["Bookmaker_norm","Market_norm","Sharp_Consensus_Frac_tmp"], errors="ignore", inplace=True)
    return out

# -------------------------
# 3) Internal Consistency Checks (Spread↔ML and Total↔Side)
# -------------------------
def add_internal_consistency_features(df: pd.DataFrame, df_cross_market: pd.DataFrame | None = None, sport_default="NFL") -> pd.DataFrame:
    """
    Emits:
      - Spread_ML_ProbGap (float32): | winprob_from_spread - implied ML prob |
      - Spread_ML_Inconsistency (float32): same as above (alias)
      - Total_vs_Side_ImpliedDelta (float32): | (Total/2 ± Spread/2) difference across teams | (cheap proxy)
    Needs either:
      - H2H odds in df or in df_cross_market to get implied ML prob
      - Spread Value in df or df_cross_market
      - Total value in df or df_cross_market (for Total_vs_Side proxy)
    """
    if df.empty: return df
    out = df.copy()
    out["Sport_norm"]  = out.get("Sport", sport_default).astype(str).str.upper().str.strip()
    out["Market_norm"] = out.get("Market", "").astype(str).str.lower().str.strip()

    # Bring in cross-market pivots if provided (as in your training build)
    if df_cross_market is not None and not df_cross_market.empty:
        out = merge_drop_overlap(
            out,
            df_cross_market,
            on="Game_Key",
            how="left",
            keep_right=True,   # cross-market pivots should win
        )


    # Prefer spread/total from cross-market pivots if present
    spread_val = pd.to_numeric(out.get("Spread_Value", out.get("Value")), errors="coerce")
    total_val  = pd.to_numeric(out.get("Total_Value", np.nan), errors="coerce")
    h2h_odds   = pd.to_numeric(out.get("H2H_Odds", out.get("Odds_Price")), errors="coerce")

    # Implied ML prob from odds
    p_ml = _amer_to_prob(h2h_odds)

    # Win prob from spread (sport-aware sigma)
    p_spread = np.array([
        _spread_to_winprob(s, sport=str(sp)) for s, sp in zip(spread_val, out["Sport_norm"])
    ], dtype="float64")

    # Gap (only meaningful on spreads/h2h rows where both exist)
    gap = np.abs(p_spread - p_ml)
    out["Spread_ML_ProbGap"] = np.nan_to_num(gap, nan=0.0).astype("float32")
    out["Spread_ML_Inconsistency"] = out["Spread_ML_ProbGap"]

    # Simple Total↔Side proxy (no team totals feed): |(T/2 - |S|/2) - (T/2 + |S|/2)| = |S|
    # This collapses to spread magnitude; to add info, scale by how extreme T is vs league mean if available.
    s_abs = np.abs(spread_val)
    out["Total_vs_Side_ImpliedDelta"] = np.nan_to_num(s_abs, nan=0.0).astype("float32")

    out.drop(columns=["Sport_norm","Market_norm"], inplace=True, errors="ignore")
    return out

# -------------------------
# 4) Curvature from Alt Lines (optional data)
# -------------------------

# -------------------------
# 5) CLV-Proxy Features
# -------------------------
def add_clv_proxy_features(df: pd.DataFrame, clv_model_path: str | None = None) -> pd.DataFrame:
    """
    Emits:
      - CLV_Proxy_E_DeltaNext15m (float32)
      - CLV_Proxy_E_CLV          (float32) expected (market move toward close); heuristic if no model.
    If clv_model_path points to a joblib model with .predict, it will be used on a small feature vector.
    Otherwise a light heuristic is applied using timing+microstructure features you already built.
    """
    if df.empty: return df
    out = df.copy()

    # Small feature vector for a side-model (if available)
    clv_feats = [
        "Hybrid_Line_LateMag","Hybrid_Line_EarlyMag","Hybrid_Odds_LateMag","Hybrid_Timing_Entropy_Line",
        "Key_Corridor_Pressure","Dist_To_Next_Key","Outlier_Flag_SharpBooks","Stale_Quote_Flag",
        "Book_Move_Lead_Time_Sec","Implied_Hold_Book"
    ]
    for c in clv_feats:
        if c not in out.columns: out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    E_delta = (
        0.40 * out["Hybrid_Line_LateMag"] -
        0.15 * out["Hybrid_Line_EarlyMag"] +
        0.30 * out["Key_Corridor_Pressure"] -
        0.10 * (out["Stale_Quote_Flag"].astype("float32")) +
        0.10 * (out["Outlier_Flag_SharpBooks"].astype("float32"))
    ).astype("float32")

    out["CLV_Proxy_E_DeltaNext15m"] = E_delta

    # If a tiny side-model exists, refine E_CLV with it; else heuristic = E_delta
    use_model = clv_model_path and os.path.exists(clv_model_path)
    if use_model:
        try:
            from joblib import load
            model = load(clv_model_path)
            X = out[clv_feats].astype("float32")
            y = model.predict(X)
            out["CLV_Proxy_E_CLV"] = pd.to_numeric(y, errors="coerce").fillna(E_delta).astype("float32")
        except Exception:
            out["CLV_Proxy_E_CLV"] = E_delta
    else:
        out["CLV_Proxy_E_CLV"] = E_delta

    return out


PHASES  = ["Overnight","Early","Midday","Late"]
URGENCY = ["VeryEarly","MidRange","LateGame","Urgent"]

def _bins(prefix):
    # returns all 16 column names for a timing family
    return [f"{prefix}{p}_{u}" for p in PHASES for u in URGENCY]

def _sum_cols(df, cols):
    if not cols: return pd.Series(0.0, index=df.index)
    return df[cols].sum(axis=1, numeric_only=True)

def _safe_div(a, b, eps=1e-9):
    return a / (b.abs() + eps)

def _entropy_rowwise(X, eps=1e-12):
    # X: DataFrame of non-negative magnitudes; returns entropy per row
    S = X.sum(axis=1)
    W = X.div(S.replace(0, np.nan), axis=0).clip(lower=eps)  # row-normalize
    H = -(W * np.log(W)).sum(axis=1)
    H = H.replace([np.inf, -np.inf], 0).fillna(0.0)
    return H

def _row_corr(a, b):
    # Coerce to 1-D float arrays robustly
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))

    # Trim to equal length if something went odd
    if a.size != b.size:
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]

    # Handle all-NaN / non-finite cases
    if not np.isfinite(a).any() or not np.isfinite(b).any():
        return 0.0
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    # Zero variance → undefined corr
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0

    # Safe correlation
    c = np.corrcoef(a, b)
    # If numerical issues produce NaN
    val = float(c[0, 1]) if c.shape == (2, 2) else 0.0
    return 0.0 if not np.isfinite(val) else val

def build_timing_aggregates_inplace(df: pd.DataFrame,
                            line_prefix="SharpMove_Magnitude_",
                            odds_prefix="OddsMove_Magnitude_",
                            *,
                            drop_original=False) -> list[str]:
    """
    Enriches the DataFrame in-place with timing aggregates.
    Returns a list of newly created column names.
    """
    out_cols = []
    # --- Gather column groups (only those that exist) ---
    line_bins_all = [c for c in _bins(line_prefix) if c in df.columns]
    odds_bins_all = [c for c in _bins(odds_prefix) if c in df.columns]

    # Phase slices
    line_phase = {p: [f"{line_prefix}{p}_{u}" for u in URGENCY if f"{line_prefix}{p}_{u}" in df.columns] for p in PHASES}
    odds_phase = {p: [f"{odds_prefix}{p}_{u}" for u in URGENCY if f"{odds_prefix}{p}_{u}" in df.columns] for p in PHASES}

    # --- Totals ---
    df["Line_TotalMag"] = _sum_cols(df, line_bins_all)
    df["Odds_TotalMag"] = _sum_cols(df, odds_bins_all)
    out_cols += ["Line_TotalMag","Odds_TotalMag"]

    # --- Per-phase sums ---
    for p in PHASES:
        ln = f"Line_PhaseMag_{p}"
        od = f"Odds_PhaseMag_{p}"
        df[ln] = _sum_cols(df, line_phase[p])
        df[od] = _sum_cols(df, odds_phase[p])
        out_cols += [ln, od]

    # --- Shares & ratios ---
    df["Line_UrgentShare"] = _safe_div(
        df[[c for c in line_bins_all if c.endswith("_Urgent")]].sum(axis=1),
        df["Line_TotalMag"]
    )
    df["Odds_UrgentShare"] = _safe_div(
        df[[c for c in odds_bins_all if c.endswith("_Urgent")]].sum(axis=1),
        df["Odds_TotalMag"]
    )
    out_cols += ["Line_UrgentShare","Odds_UrgentShare"]

    df["Line_LateShare"] = _safe_div(df["Line_PhaseMag_Late"], df["Line_TotalMag"])
    df["Odds_LateShare"] = _safe_div(df["Odds_PhaseMag_Late"], df["Odds_TotalMag"])
    out_cols += ["Line_LateShare","Odds_LateShare"]

    # --- Max bin (spikiness) ---
    df["Line_MaxBinMag"] = df[line_bins_all].max(axis=1) if line_bins_all else 0.0
    df["Odds_MaxBinMag"] = df[odds_bins_all].max(axis=1) if odds_bins_all else 0.0
    out_cols += ["Line_MaxBinMag","Odds_MaxBinMag"]

    # --- Entropy (dispersion of timing) ---
    df["Line_Entropy"] = _entropy_rowwise(df[line_bins_all]) if line_bins_all else 0.0
    df["Odds_Entropy"] = _entropy_rowwise(df[odds_bins_all]) if odds_bins_all else 0.0
    out_cols += ["Line_Entropy","Odds_Entropy"]

    # --- Cross axis confirmations ---
    df["LineOddsMag_Ratio"] = _safe_div(df["Line_TotalMag"], df["Odds_TotalMag"])
    out_cols += ["LineOddsMag_Ratio"]

    # Late vs (Overnight+Early+Midday)
    df["LateVsEarly_Ratio_Line"] = _safe_div(
        df["Line_PhaseMag_Late"],
        (df["Line_PhaseMag_Overnight"] + df["Line_PhaseMag_Early"] + df["Line_PhaseMag_Midday"])
    )
    df["LateVsEarly_Ratio_Odds"] = _safe_div(
        df["Odds_PhaseMag_Late"],
        (df["Odds_PhaseMag_Overnight"] + df["Odds_PhaseMag_Early"] + df["Odds_PhaseMag_Midday"])
    )
    out_cols += ["LateVsEarly_Ratio_Line","LateVsEarly_Ratio_Odds"]

    # Timing correlation across 16 aligned bins
    def _row_corr(a, b):
        # a,b: arrays of equal length
        if np.all(a == 0) and np.all(b == 0): return 0.0
        if np.std(a) == 0 or np.std(b) == 0:  return 0.0
        return float(np.corrcoef(a, b)[0,1])

    if line_bins_all and odds_bins_all and (len(line_bins_all) == len(odds_bins_all)):
    # Ensure we pull 1-D arrays for each row
        df["Timing_Corr_Line_Odds"] = [
            _row_corr(
                df.loc[i, line_bins_all].to_numpy(dtype=float, copy=False),
                df.loc[i, odds_bins_all].to_numpy(dtype=float, copy=False),
            )
            for i in df.index
        ]
    else:
        df["Timing_Corr_Line_Odds"] = 0.0
    out_cols += ["Timing_Corr_Line_Odds"]


    # Optionally drop originals (after creating aggregates!)
    if drop_original:
        df.drop(columns=line_bins_all + odds_bins_all, errors="ignore", inplace=True)

    return out_cols




# ---- sport/period tuning (defaults) ----
SPORT_PRIOR_CFG = {
    "NBA":  {"m_prior": 15, "alpha": 0.5, "min_n": 4},
    "NFL":  {"m_prior": 25, "alpha": 0.6, "min_n": 3},
    "WNBA": {"m_prior": 12, "alpha": 0.5, "min_n": 3},
    "MLB":  {"m_prior": 18, "alpha": 0.4, "min_n": 7},
    "CFL":  {"m_prior": 20, "alpha": 0.5, "min_n": 3},
    "NCAAF":{"m_prior": 22, "alpha": 0.55,"min_n": 3},
    "NCAAB":{"m_prior": 16, "alpha": 0.5, "min_n": 5},
}
SPORT_PERIOD_OVERRIDES = {
    "NBA": {
        "pre":  {"m_prior": 8,  "alpha": 0.4, "min_n": 2},
        "reg":  {"m_prior": 15, "alpha": 0.5, "min_n": 4},
        "post": {"m_prior": 20, "alpha": 0.6, "min_n": 4},
    },
    "NFL": {
        "pre":  {"m_prior": 12, "alpha": 0.4, "min_n": 2},
        "reg":  {"m_prior": 25, "alpha": 0.6, "min_n": 3},
        "post": {"m_prior": 28, "alpha": 0.65,"min_n": 3},
    },
}

def _cfg_for_sport_period(sport: str, period: str | None = None):
    s = (sport or "").upper()
    base = SPORT_PRIOR_CFG.get(s, {"m_prior": 15, "alpha": 0.5, "min_n": 4}).copy()
    if period:
        per = SPORT_PERIOD_OVERRIDES.get(s, {}).get(period, {})
        base.update(per)
    return base

def _eb_shrink(mean_team, n_team, global_mean, m_prior):
    n = np.asarray(n_team, dtype="float64")
    return (n * mean_team + m_prior * global_mean) / np.clip(n + m_prior, 1e-6, None)

def build_team_ats_priors_market_sport(
    df: pd.DataFrame,
    *,
    sport: str,
    market: str,                 # "spreads" | "totals" | "h2h"
    period: str | None = None,
    team_col="Team",
    game_col="Game_Key",
    ts_col="Snapshot_Timestamp",
    is_home_col="Is_Home",
    cover_bool_col="SHARP_HIT_BOOL",     # 0/1: covered/over/won (per market)
    cover_margin_col=None,               # e.g., spread margin or (actual_total - line)
    add_home_away_splits=True,
    suffix=None,
) -> pd.DataFrame:
    """
    Leakage-safe EB priors per (game, team) for the given market.
    Output includes:
        ATS_EB_Rate{suffix}
        ATS_EB_Margin{suffix}              (if cover_margin_col provided)
        ATS_Roll_Margin_Decay{suffix}      (if cover_margin_col provided)
        ATS_EB_Rate_Home{suffix}, ATS_EB_Rate_Away{suffix} (if add_home_away_splits)
    """
    if suffix is None:
        suffix = ""  # or f"_{market.capitalize()}"

    cfg = _cfg_for_sport_period(sport, period)
    m_prior = float(cfg["m_prior"])
    alpha   = float(cfg["alpha"])
    min_n   = int(cfg["min_n"])

    df = df.copy()
    # enforce types
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        sort_cols = [team_col, ts_col]
    else:
        # fallback order by game if no snapshot
        sort_cols = [team_col, game_col]
    df = df.sort_values(sort_cols)

    # targets
    y_hit = pd.to_numeric(df.get(cover_bool_col, 0), errors="coerce").fillna(0.0).astype(float)
    have_margin = bool(cover_margin_col) and (cover_margin_col in df.columns)
    if have_margin:
        y_mrg = pd.to_numeric(df[cover_margin_col], errors="coerce").fillna(0.0).astype(float)
    else:
        y_mrg = pd.Series(np.zeros(len(df), dtype="float64"), index=df.index)

    # global means
    g_mean_hit = float(np.nanmean(y_hit)) if len(y_hit) else 0.5
    g_mean_mrg = float(np.nanmean(y_mrg)) if len(y_mrg) else 0.0

    # cumulative prior-only stats per team (LOO via shift)
    grp = df[team_col]
    df["cum_n"]   = grp.groupby(grp).cumcount()  # number of prior games
    df["cum_hit"] = y_hit.groupby(grp).cumsum().shift(1)
    df["cum_mrg"] = y_mrg.groupby(grp).cumsum().shift(1) if have_margin else np.nan

    n_eff = df["cum_n"].astype(float)
    # smooth min_n by increasing prior weight when n small
    m_adj = m_prior * np.clip(min_n / np.clip(n_eff, 1.0, None), 0.25, 4.0)

    loo_hit = df["cum_hit"] / n_eff.replace(0, np.nan)
    loo_mrg = (df["cum_mrg"] / n_eff.replace(0, np.nan)) if have_margin else np.nan

    df[f"ATS_EB_Rate{suffix}"] = _eb_shrink(loo_hit, n_eff, g_mean_hit, m_adj)
    if have_margin:
        df[f"ATS_EB_Margin{suffix}"] = _eb_shrink(loo_mrg, n_eff, g_mean_mrg, m_adj)

        # EWMA of recent *prior* margins (1-step shift to avoid leakage)
        prev_m = y_mrg.groupby(grp).shift(1)
        df[f"ATS_Roll_Margin_Decay{suffix}"] = (
            prev_m.groupby(df[team_col]).transform(lambda s: s.ewm(alpha=alpha, min_periods=1).mean())
        )

    # optional home/away EB splits
    if add_home_away_splits and is_home_col in df.columns:
        mask_home = df[is_home_col].fillna(False).astype(bool)
        for lab, mask in [("Home", mask_home), ("Away", ~mask_home)]:
            s = y_hit.where(mask, np.nan)
            c = s.groupby(df[team_col]).cumsum().shift(1)
            loo = c / n_eff.replace(0, np.nan)
            df[f"ATS_EB_Rate_{lab}{suffix}"] = _eb_shrink(loo, n_eff, g_mean_hit, m_adj)

    # keep tiny output
    keep = [game_col, team_col, f"ATS_EB_Rate{suffix}"]
    for c in (f"ATS_EB_Margin{suffix}", f"ATS_Roll_Margin_Decay{suffix}",
              f"ATS_EB_Rate_Home{suffix}", f"ATS_EB_Rate_Away{suffix}"):
        if c in df.columns:
            keep.append(c)

    out = df[keep].copy()
    # drop all-NA (e.g., margin fields if not provided)
    all_na = [c for c in out.columns if c not in (game_col, team_col) and out[c].notna().sum() == 0]
    if all_na:
        out = out.drop(columns=all_na)
    return out






@st.cache_resource(show_spinner=False)
def get_bq_clients():
    return bigquery.Client(), bigquery_storage.BigQueryReadClient()

@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_scores_with_features(sport: str, days_back: int):
    bq, bqs = get_bq_clients()

    # === EXACT logic as your original f-string, but parameterized ===
    sql = """
    SELECT *
    FROM `sharplogger.sharp_data.scores_with_features`
    WHERE Sport = @sport
      AND Scored = TRUE
      AND SHARP_HIT_BOOL IS NOT NULL
      AND DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL @days_back DAY)
    """

    job_cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("sport", "STRING", sport.upper()),
            bigquery.ScalarQueryParameter("days_back", "INT64", int(days_back)),
        ],
        use_query_cache=True,
    )

    # Fast path, but stable because we reuse a cached BigQueryReadClient
    return bq.query(sql, job_config=job_cfg).to_dataframe(bqstorage_client=bqs)
def _is_xgb_classifier(m):
    try:
        from xgboost import XGBClassifier
        return isinstance(m, XGBClassifier)
    except Exception:
        return False


def _xgb_params_from_proto(proto):
    p = proto.get_params(deep=False)

    def _f(key, default):
        v = p.get(key, default)
        return default if v is None else v

    params = {
        "objective": _f("objective", "binary:logistic"),
        "eval_metric": _f("eval_metric", ["logloss", "auc"]),
        "tree_method": _f("tree_method", "hist"),
        "grow_policy": _f("grow_policy", "lossguide"),
        "max_bin": int(_f("max_bin", 256)),
        "learning_rate": float(_f("learning_rate", 0.05)),
        "subsample": float(_f("subsample", 0.7)),
        "colsample_bytree": float(_f("colsample_bytree", 0.6)),
        "min_child_weight": float(_f("min_child_weight", 1.0)),
        "reg_lambda": float(_f("reg_lambda", 1.0)),
        "reg_alpha": float(_f("reg_alpha", 0.0)),
        "gamma": float(_f("gamma", 0.0)),
        "max_depth": int(_f("max_depth", 0)),
        "nthread": int(_f("n_jobs", 1) or 1),
        "verbosity": 0,
        "predictor": _f("predictor", "cpu_predictor"),
    }

    num_boost_round = int(_f("n_estimators", 400) or 400)
    return params, num_boost_round




def c_features_inplace(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Sanitize feature columns for modeling + UI. Drop or coerce unsafe dtypes."""
    kept = []
    for c in features:
        if c not in df.columns:
            continue
        s = df[c]

        # --- Drop if it's array, list, dict, or weird object ---
        if not s.map(pd.api.types.is_scalar).all():
            continue

        # --- Handle boos → numeric ---
        if pd.api.types.is_bool_dtype(s):
            df[c] = s.astype("int8")

        # --- Coerce strings/objects/categories to str, then drop ---
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            df[c] = s.astype(str)

        # --- Convert to float64 for model safety ---
        elif pd.api.types.is_numeric_dtype(s):
            df[c] = pd.to_numeric(s, errors="coerce").astype("float64")

        # --- Drop tz-aware or weird datetimes ---
        elif pd.api.types.is_datetime64_any_dtype(s):
            try:
                df[c] = pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
            except Exception:
                continue

        # --- Fill any remaining NaNs ---
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        kept.append(c)

    return kept

def _eval_on_train(estimator, X, y):
    """Simple train-metrics helper."""
    proba = estimator.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y, proba)
    ll    = log_loss(y, proba)
    return auc, ll

def hyperparam_search_until_good(
    base_estimator,
    param_distributions,
    X,
    y,
    cv,
    *,
    min_auc=0.520,
    max_logloss=0.693,
    max_overfit_gap=0.12,
    max_rounds=10,
    n_iter_per_round=25,
    random_state=42,
    logger=None,
):
    """
    Run multiple random-search rounds until a model meets minimum conditions.
    If none do, raise RuntimeError("no optimal solution found") and return
    the best-so-far metrics so you can log them.
    """
    rng = np.random.RandomState(random_state)

    best_global = None
    best_metrics_global = None  # (cv_auc, cv_logloss, train_auc, train_logloss, params)

    for round_idx in range(1, max_rounds + 1):
        seed = int(rng.randint(0, 1e9))
        if logger:
            logger.info(
                "🔁 Hyperparam search round %d/%d (n_iter=%d, seed=%d)",
                round_idx, max_rounds, n_iter_per_round, seed
            )

        search = RandomizedSearchCV(
            estimator=clone(base_estimator),
            param_distributions=param_distributions,
            n_iter=n_iter_per_round,
            scoring={
                "auc": "roc_auc",
                "logloss": "neg_log_loss",
            },
            refit="auc",      # best AUC model kept as best_estimator_
            cv=cv,
            n_jobs=-1,
            random_state=seed,
            verbose=0,
        )
        search.fit(X, y)

        # CV metrics from the search object
        cv_auc = search.best_score_
        cv_logloss = -search.cv_results_["mean_test_logloss"][search.best_index_]

        # Train metrics on whole training fold (for overfit gap)
        train_auc, train_logloss = _eval_on_train(search.best_estimator_, X, y)
        overfit_gap = train_auc - cv_auc

        params = deepcopy(search.best_params_)
        msg = (
            f"Round {round_idx}: cv_auc={cv_auc:.4f}, cv_logloss={cv_logloss:.4f}, "
            f"train_auc={train_auc:.4f}, train_logloss={train_logloss:.4f}, "
            f"overfit_gap={overfit_gap:.4f}"
        )
        if logger:
            logger.info("📊 %s", msg)
        else:
            print(msg)

        # Update global best even if it doesn't pass the gate
        if (
            best_metrics_global is None
            or cv_auc > best_metrics_global[0]
            or (cv_auc == best_metrics_global[0] and cv_logloss < best_metrics_global[1])
        ):
            best_global = deepcopy(search.best_estimator_)
            best_metrics_global = (cv_auc, cv_logloss, train_auc, train_logloss, params)

        # Check minimum conditions
        passes_gate = (
            (cv_auc >= min_auc) and
            (cv_logloss <= max_logloss) and
            (overfit_gap <= max_overfit_gap)
        )
        if passes_gate:
            if logger:
                logger.info(
                    "✅ Model passed minimum conditions in round %d; promoting as optimal.",
                    round_idx,
                )
            return {
                "estimator": search.best_estimator_,
                "params": params,
                "metrics": {
                    "cv_auc": cv_auc,
                    "cv_logloss": cv_logloss,
                    "train_auc": train_auc,
                    "train_logloss": train_logloss,
                    "overfit_gap": overfit_gap,
                },
                "status": "ok",
            }

    # If we get here: nothing passed the gate
    if logger and best_metrics_global is not None:
        cv_auc, cv_logloss, train_auc, train_logloss, params = best_metrics_global
        logger.warning(
            "❌ No optimal solution found after %d rounds. "
            "Best CV AUC=%.4f, CV logloss=%.4f, overfit_gap=%.4f. Params=%s",
            max_rounds, cv_auc, cv_logloss, train_auc - cv_auc, params,
        )

    return {
        "estimator": best_global,
        "params": best_metrics_global[-1] if best_metrics_global else None,
        "metrics": {
            "cv_auc": best_metrics_global[0],
            "cv_logloss": best_metrics_global[1],
            "train_auc": best_metrics_global[2],
            "train_logloss": best_metrics_global[3],
            "overfit_gap": best_metrics_global[2] - best_metrics_global[0],
        } if best_metrics_global else None,
        "status": "no_optimal_solution",
    }

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from google.cloud import storage

CHAMPION_META_VERSION = 1


@dataclass
class ChampionMeta:
    sport: str
    market: str
    model_path: str              # GCS path to pickle
    created_at: str              # ISO8601 timestamp
    metrics: Dict[str, float]    # holdout + CV metrics
    config: Dict[str, Any]       # any training config you want (search space, seeds, etc.)
    version: int = CHAMPION_META_VERSION


def _champion_meta_blob_name(sport: str, market: str) -> str:
    # e.g. "models/NFL/spreads/champion_meta.json"
    return f"models/{sport}/{market}/champion_meta.json"


def load_champion_meta(
    bucket_name: str,
    sport: str,
    market: str,
    client: Optional[storage.Client] = None,
) -> Optional[ChampionMeta]:
    if client is None:
        client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = _champion_meta_blob_name(sport, market)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return None
    data = json.loads(blob.download_as_text())
    return ChampionMeta(**data)


def save_champion_meta(
    bucket_name: str,
    meta: ChampionMeta,
    client: Optional[storage.Client] = None,
) -> None:
    if client is None:
        client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = _champion_meta_blob_name(meta.sport, meta.market)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(json.dumps(asdict(meta), indent=2))


from typing import Tuple


def _score_model_for_promotion(metrics: Dict[str, float]) -> float:
    """
    Build a scalar score that prefers:
      - higher holdout AUC
      - lower holdout LogLoss
      - smaller train–holdout AUC gap (less overfit)

    You can tweak weights per sport/market if needed.
    """
    auc_h  = float(metrics.get("auc_holdout", float("nan")))
    ll_h   = float(metrics.get("logloss_holdout", float("nan")))
    gap_th = float(metrics.get("auc_gap_train_holdout", float("nan")))

    # Soft penalties
    if not np.isfinite(auc_h) or not np.isfinite(ll_h):
        return float("-inf")

    # Normalize logloss to roughly similar scale as AUC (not perfect, but ok)
    # Higher is better for this score.
    score = (
        1.0 * auc_h            # main driver
        - 0.5 * ll_h           # want lower logloss
        - 0.10 * max(gap_th, 0.0)  # penalize big overfit gaps
    )
    return float(score)


def should_promote_challenger(
    challenger_metrics: Dict[str, float],
    champion_metrics: Optional[Dict[str, float]],
    *,
    min_auc_holdout: float = 0.50,
    max_gap_train_holdout: float = 0.50,
    min_auc_improvement: float = 0.00005,
    max_logloss_worsen: float = 0.001,
) -> Tuple[bool, Dict[str, float]]:
    """
    Decide whether to promote challenger vs champion.

    Returns:
      (promote_flag, debug_dict)
    """
    dbg: Dict[str, float] = {}

    auc_h_c = float(challenger_metrics.get("auc_holdout", float("nan")))
    ll_h_c  = float(challenger_metrics.get("logloss_holdout", float("nan")))
    gap_c   = float(challenger_metrics.get("auc_gap_train_holdout", float("nan")))

    dbg["challenger_auc_holdout"] = auc_h_c
    dbg["challenger_logloss_holdout"] = ll_h_c
    dbg["challenger_gap_train_holdout"] = gap_c

    # 1) Basic sanity: challenger must meet absolute thresholds
    if (not np.isfinite(auc_h_c)) or (auc_h_c < min_auc_holdout):
        dbg["reason"] = "challenger_auc_below_min"
        return False, dbg

    if np.isfinite(gap_c) and (gap_c > max_gap_train_holdout):
        dbg["reason"] = "challenger_gap_too_large"
        return False, dbg

    # 2) No existing champion → auto-promote if challenger passes thresholds
    if not champion_metrics:
        dbg["reason"] = "no_champion_auto_promote"
        return True, dbg

    auc_h_champ = float(champion_metrics.get("auc_holdout", float("nan")))
    ll_h_champ  = float(champion_metrics.get("logloss_holdout", float("nan")))
    gap_champ   = float(champion_metrics.get("auc_gap_train_holdout", float("nan")))

    dbg["champ_auc_holdout"] = auc_h_champ
    dbg["champ_logloss_holdout"] = ll_h_champ
    dbg["champ_gap_train_holdout"] = gap_champ

    # 3) Require challenger holdout AUC to beat champion by some epsilon
    if np.isfinite(auc_h_champ):
        auc_improve = auc_h_c - auc_h_champ
        dbg["auc_improvement"] = auc_improve
        if auc_improve < min_auc_improvement:
            dbg["reason"] = "auc_improvement_too_small"
            return False, dbg

    # 4) Ensure logloss is not meaningfully worse
    if np.isfinite(ll_h_champ):
        ll_delta = ll_h_c - ll_h_champ  # challenger - champ (we want <= small)
        dbg["logloss_delta"] = ll_delta
        if ll_delta > max_logloss_worsen:
            dbg["reason"] = "logloss_worse_too_much"
            return False, dbg

    # 5) Optionally, compare scalar scores as a final check
    score_challenger = _score_model_for_promotion(challenger_metrics)
    score_champion   = _score_model_for_promotion(champion_metrics)
    dbg["score_challenger"] = score_challenger
    dbg["score_champion"] = score_champion

    if score_challenger <= score_champion:
        dbg["reason"] = "scalar_score_not_better"
        return False, dbg

    dbg["reason"] = "challenger_better"
    return True, dbg
from datetime import datetime, timezone


def train_with_champion_wrapper(
    sport: str,
    market: str,
    *,
    bucket_name: str,
    # pass whatever config you usually pass into train_sharp_model_from_bq
    **train_kwargs,
) -> None:
    """
    High-level entrypoint: trains a challenger model, compares it to
    the existing champion (if any), and promotes only if better.
    """

    logger = logging.getLogger(__name__)

    # 1) Train challenger with full pipeline (CV, ES, OOF, calibration, etc.)
    logger.info("🏁 Training challenger for %s %s ...", sport, market)
    challenger = train_sharp_model_from_bq(
        sport=sport,
        market=market,
        bucket_name=bucket_name,
        return_artifacts=True,
        **train_kwargs,
    )
    if challenger is None:
        logger.error("Challenger training returned None for %s %s", sport, market)
        return

    challenger_metrics = challenger.get("metrics", {}) or {}
    challenger_model_path = challenger.get("model_path", "")

    # 2) Load current champion metadata (if exists)
    champion_meta = load_champion_meta(bucket_name, sport, market)
    champion_metrics = champion_meta.metrics if champion_meta else None

    # 3) Decide promotion
    promote, dbg = should_promote_challenger(
        challenger_metrics=challenger_metrics,
        champion_metrics=champion_metrics,
    )

    # Streamlit-friendly logging
    try:
        st.subheader(f"Champion vs Challenger — {sport} {market}")
        st.json(
            {
                "promote": promote,
                "decision_debug": dbg,
                "challenger_metrics": challenger_metrics,
                "champion_metrics": champion_metrics,
            }
        )
    except Exception:
        pass

    if not promote:
        logger.info(
            "👑 Champion retained for %s %s. Reason: %s",
            sport,
            market,
            dbg.get("reason"),
        )
        return

    # 4) Promote challenger: mark as champion in metadata
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_meta = ChampionMeta(
        sport=sport,
        market=market,
        model_path=challenger_model_path,
        created_at=now_iso,
        metrics={k: float(v) for k, v in challenger_metrics.items()
                 if np.isfinite(v) or isinstance(v, (int, float))},
        config=challenger.get("config", {}),
    )
    save_champion_meta(bucket_name, new_meta)
    logger.info(
        "✅ Challenger PROMOTED to champion for %s %s. AUC_holdout=%.4f, LogLoss_holdout=%.4f",
        sport,
        market,
        challenger_metrics.get("auc_holdout", float("nan")),
        challenger_metrics.get("logloss_holdout", float("nan")),
    )

def get_quality_thresholds(sport: str, market: str) -> dict:
    """
    Return quality thresholds for hyperparam search based on sport + market.

    sport:  e.g. "NFL", "NBA", "MLB", "NCAAF", "NCAAB"
    market: e.g. "spreads", "totals", "h2h"
    """
    s = (sport or "").upper()
    m = (market or "").lower()

 
    # Defaults (conservative, generic binary) — interpreted as ALIGNED AUC floors
    MIN_AUC           = 0.51     # aligned AUC (>=0.53 means some edge)
    MAX_LOGLOSS       = 0.695    # allow slightly worse than coinflip because class imbalance / priors
    MAX_OVERFIT_GAP   = 0.18     # aligned train - aligned val
    MIN_AUC_THRESHOLD = 0.53
    
    # ---- NFL ----
    if s == "NFL":
        if m == "spreads":
            MIN_AUC         = 0.51
            MAX_OVERFIT_GAP = 0.22
        elif m == "totals":
            MIN_AUC         = 0.51
            MAX_OVERFIT_GAP = 0.20
        else:  # h2h / others
            MIN_AUC         = 0.54
            MAX_OVERFIT_GAP = 0.18
    
    # ---- NBA ----
    elif s == "NBA":
        if m == "spreads":
            MIN_AUC         = 0.51
            MAX_OVERFIT_GAP = 0.20
        elif m == "totals":
            MIN_AUC         = 0.51
            MAX_OVERFIT_GAP = 0.20
        else:
            MIN_AUC         = 0.51
            MAX_OVERFIT_GAP = 0.18
    
    # ---- MLB ----
    elif s == "MLB":
        if m == "totals":
            MIN_AUC         = 0.525
            MAX_OVERFIT_GAP = 0.18
        elif m == "spreads":   # RL
            MIN_AUC         = 0.525
            MAX_OVERFIT_GAP = 0.18
        else:  # moneyline
            MIN_AUC         = 0.53
            MAX_OVERFIT_GAP = 0.18
    
    # ---- NCAAF ----
    elif s in {"NCAAF", "CFB"}:
        # higher variance; allow slightly more gap
        if m == "spreads":
            MIN_AUC         = 0.525
            MAX_OVERFIT_GAP = 0.24
        else:
            MIN_AUC         = 0.525
            MAX_OVERFIT_GAP = 0.24
    
    # ---- NCAAB ----
    elif s in {"NCAAB", "NCAAM"}:
        if m == "spreads":
            MIN_AUC         = 0.51
            MAX_OVERFIT_GAP = 0.22
        else:
            MIN_AUC         = 0.51
            MAX_OVERFIT_GAP = 0.22
    
    return dict(
        MIN_AUC=MIN_AUC,
        MAX_LOGLOSS=MAX_LOGLOSS,
        MAX_OVERFIT_GAP=MAX_OVERFIT_GAP,
        MIN_AUC_THRESHOLD=MIN_AUC_THRESHOLD,
    )


# -----------------------------
# Sport × Market streak config
# -----------------------------
SPORT_MARKET_STREAK_CFG = {
    # Football
    ("NFL",   "spreads"): {"window": 7,  "on_threshold": 2},
    ("NFL",   "h2h"):     {"window": 7,  "on_threshold": 2},
    ("NFL",   "totals"):  {"window": 7,  "on_threshold": 2},

    ("NCAAF", "spreads"): {"window": 7,  "on_threshold": 2},
    ("NCAAF", "h2h"):     {"window": 7,  "on_threshold": 2},
    ("NCAAF", "totals"):  {"window": 7,  "on_threshold": 2},

    ("CFL",   "spreads"): {"window": 5,  "on_threshold": 3},
    ("CFL",   "h2h"):     {"window": 5,  "on_threshold": 3},
    ("CFL",   "totals"):  {"window": 5,  "on_threshold": 3},

    # Basketball
    ("NBA",   "spreads"): {"window": 10, "on_threshold": 3},
    ("NBA",   "h2h"):     {"window": 10, "on_threshold": 3},
    ("NBA",   "totals"):  {"window": 10, "on_threshold": 3},

    ("NCAAB", "spreads"): {"window": 10, "on_threshold": 3},
    ("NCAAB", "h2h"):     {"window": 10, "on_threshold": 3},
    ("NCAAB", "totals"):  {"window": 10, "on_threshold": 3},

    ("WNBA",  "spreads"): {"window": 8,  "on_threshold": 5},
    ("WNBA",  "h2h"):     {"window": 8,  "on_threshold": 5},
    ("WNBA",  "totals"):  {"window": 8,  "on_threshold": 5},

    # Baseball
    ("MLB",   "spreads"): {"window": 20, "on_threshold": 9},
    ("MLB",   "h2h"):     {"window": 20, "on_threshold": 9},
    ("MLB",   "totals"):  {"window": 20, "on_threshold": 9},

    # Hockey if you add it
    ("NHL",   "spreads"): {"window": 10, "on_threshold": 6},
    ("NHL",   "h2h"):     {"window": 10, "on_threshold": 6},
    ("NHL",   "totals"):  {"window": 10, "on_threshold": 6},
}

DEFAULT_CFG = {"window": 8, "on_threshold": 5}


def _norm_market(m: str) -> str:
    m = (m or "").lower().strip()
    if m in ("spread", "spreads", "ats"): return "spreads"
    if m in ("total", "totals", "ou", "overunder"): return "totals"
    if m in ("h2h", "ml", "moneyline", "money_line"): return "h2h"
    return m


def _get_streak_cfg(sport: str, market: str) -> dict:
    s = str(sport).upper().strip()
    m = _norm_market(market)
    return SPORT_MARKET_STREAK_CFG.get((s, m), DEFAULT_CFG)


def build_cover_streaks_game_level(df_bt_prepped: pd.DataFrame, *, sport: str, market: str) -> pd.DataFrame:
    """
    Build rolling-window cover streak / rate features at the GAME grain:
      one row per (Sport, Market, Team, Game_Key), time-ordered by the true game start.

    Key fixes vs prior version:
    - Uses feat_Game_Start as the primary time axis when available (falls back to Game_Start).
    - Adds a deterministic tie-breaker (Game_Key) to ordering.
    - Normalizes Sport/Market/Team keys inside the function to avoid merge misses.
    - Keeps your rolling-window semantics + shift(1) (no leakage).
    """
    cfg = _get_streak_cfg(sport, market)
    window_length = int(cfg["window"])
    on_thresh     = int(cfg["on_threshold"])

    # Prefer true game start if present
    time_col = "feat_Game_Start" if "feat_Game_Start" in df_bt_prepped.columns else "Game_Start"

    need = ["Sport", "Market", "Game_Key", "Team", time_col, "SHARP_HIT_BOOL", "Is_Home", "Is_Favorite_Context"]
    missing = [c for c in need if c not in df_bt_prepped.columns]
    if missing:
        raise ValueError(f"build_cover_streaks_game_level missing cols: {missing}")

    sport_u  = str(sport).upper().strip()
    market_l = _norm_market(market)

    # Slice and normalize keys (prevents silent merge misses)
    d = df_bt_prepped.loc[
        (df_bt_prepped["Sport"].astype(str).str.upper().str.strip() == sport_u) &
        (df_bt_prepped["Market"].astype(str).str.lower().str.strip().map(_norm_market) == market_l),
        need
    ].copy()

    d["Sport"]  = d["Sport"].astype(str).str.upper().str.strip()
    d["Market"] = d["Market"].astype(str).str.lower().str.strip().map(_norm_market)
    d["Team"]   = d["Team"].astype(str).str.lower().str.strip()

    # Parse time axis (true game start preferred)
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)

    # Build one row per team-game (drop book/snapshot noise) ordered by true game time
    g = (
        d.dropna(subset=["Game_Key", "Team", time_col])
         .sort_values(["Sport", "Market", "Team", time_col, "Game_Key"])  # tie-breaker prevents unstable rolls
         .drop_duplicates(["Sport", "Market", "Team", "Game_Key"], keep="last")
         .copy()
    )

    # Ensure numeric hit flag
    g["SHARP_HIT_BOOL"] = pd.to_numeric(g["SHARP_HIT_BOOL"], errors="coerce")

    # Eligible-only series (NaN when not eligible => denom counts eligible games only)
    g["Cover_All"]       = g["SHARP_HIT_BOOL"]
    g["Cover_Home_Only"] = g["SHARP_HIT_BOOL"].where(g["Is_Home"] == 1)
    g["Cover_Away_Only"] = g["SHARP_HIT_BOOL"].where(g["Is_Home"] == 0)
    g["Cover_Fav_Only"]  = g["SHARP_HIT_BOOL"].where(g["Is_Favorite_Context"] == 1)
    g["Cover_Home_Fav"]  = g["SHARP_HIT_BOOL"].where((g["Is_Home"] == 1) & (g["Is_Favorite_Context"] == 1))
    g["Cover_Away_Fav"]  = g["SHARP_HIT_BOOL"].where((g["Is_Home"] == 0) & (g["Is_Favorite_Context"] == 1))

    grp = ["Sport", "Market", "Team"]

    # Robust rolling on shifted series (no leakage)
    def _roll_sum_shift(col: str) -> pd.Series:
        s_prev = g.groupby(grp, sort=False)[col].shift(1)
        out = (
            s_prev.groupby([g[k] for k in grp], sort=False)
                  .rolling(window=window_length, min_periods=1)
                  .sum()
                  .reset_index(level=list(range(len(grp))), drop=True)
        )
        return out

    def _roll_games_shift(col: str) -> pd.Series:
        s_prev = g.groupby(grp, sort=False)[col].shift(1)
        out = (
            s_prev.notna().astype(int)
                  .groupby([g[k] for k in grp], sort=False)
                  .rolling(window=window_length, min_periods=1)
                  .sum()
                  .reset_index(level=list(range(len(grp))), drop=True)
        )
        return out

    def _rate(wins: pd.Series, games: pd.Series) -> pd.Series:
        return wins / games.replace(0, np.nan)

    # ---- sums ("streaks" = wins in last N eligible games) ----
    g["Team_Recent_Cover_Streak"]          = _roll_sum_shift("Cover_All")
    g["Team_Recent_Cover_Streak_Home"]     = _roll_sum_shift("Cover_Home_Only")
    g["Team_Recent_Cover_Streak_Away"]     = _roll_sum_shift("Cover_Away_Only")
    g["Team_Recent_Cover_Streak_Fav"]      = _roll_sum_shift("Cover_Fav_Only")
    g["Team_Recent_Cover_Streak_Home_Fav"] = _roll_sum_shift("Cover_Home_Fav")
    g["Team_Recent_Cover_Streak_Away_Fav"] = _roll_sum_shift("Cover_Away_Fav")

    # ---- denominators (eligible games in last N) ----
    g["Team_Recent_Cover_Games"]           = _roll_games_shift("Cover_All")
    g["Team_Recent_Cover_Games_Home"]      = _roll_games_shift("Cover_Home_Only")
    g["Team_Recent_Cover_Games_Away"]      = _roll_games_shift("Cover_Away_Only")
    g["Team_Recent_Cover_Games_Fav"]       = _roll_games_shift("Cover_Fav_Only")
    g["Team_Recent_Cover_Games_Home_Fav"]  = _roll_games_shift("Cover_Home_Fav")
    g["Team_Recent_Cover_Games_Away_Fav"]  = _roll_games_shift("Cover_Away_Fav")

    # ---- rates (wins/games) ----
    g["Team_Recent_Cover_Rate"]            = _rate(g["Team_Recent_Cover_Streak"],          g["Team_Recent_Cover_Games"])
    g["Team_Recent_Cover_Rate_Home"]       = _rate(g["Team_Recent_Cover_Streak_Home"],     g["Team_Recent_Cover_Games_Home"])
    g["Team_Recent_Cover_Rate_Away"]       = _rate(g["Team_Recent_Cover_Streak_Away"],     g["Team_Recent_Cover_Games_Away"])
    g["Team_Recent_Cover_Rate_Fav"]        = _rate(g["Team_Recent_Cover_Streak_Fav"],      g["Team_Recent_Cover_Games_Fav"])
    g["Team_Recent_Cover_Rate_Home_Fav"]   = _rate(g["Team_Recent_Cover_Streak_Home_Fav"], g["Team_Recent_Cover_Games_Home_Fav"])
    g["Team_Recent_Cover_Rate_Away_Fav"]   = _rate(g["Team_Recent_Cover_Streak_Away_Fav"], g["Team_Recent_Cover_Games_Away_Fav"])

    # ---- "on streak" flags (count-based) ----
    g["On_Cover_Streak"]          = (g["Team_Recent_Cover_Streak"]          >= on_thresh).astype(int)
    g["On_Cover_Streak_Home"]     = (g["Team_Recent_Cover_Streak_Home"]     >= on_thresh).astype(int)
    g["On_Cover_Streak_Away"]     = (g["Team_Recent_Cover_Streak_Away"]     >= on_thresh).astype(int)
    g["On_Cover_Streak_Fav"]      = (g["Team_Recent_Cover_Streak_Fav"]      >= on_thresh).astype(int)
    g["On_Cover_Streak_Home_Fav"] = (g["Team_Recent_Cover_Streak_Home_Fav"] >= on_thresh).astype(int)
    g["On_Cover_Streak_Away_Fav"] = (g["Team_Recent_Cover_Streak_Away_Fav"] >= on_thresh).astype(int)

    streak_cols = [
        # sums
        "Team_Recent_Cover_Streak","Team_Recent_Cover_Streak_Home","Team_Recent_Cover_Streak_Away",
        "Team_Recent_Cover_Streak_Fav","Team_Recent_Cover_Streak_Home_Fav","Team_Recent_Cover_Streak_Away_Fav",
        # denoms
        "Team_Recent_Cover_Games","Team_Recent_Cover_Games_Home","Team_Recent_Cover_Games_Away",
        "Team_Recent_Cover_Games_Fav","Team_Recent_Cover_Games_Home_Fav","Team_Recent_Cover_Games_Away_Fav",
        # rates
        "Team_Recent_Cover_Rate","Team_Recent_Cover_Rate_Home","Team_Recent_Cover_Rate_Away",
        "Team_Recent_Cover_Rate_Fav","Team_Recent_Cover_Rate_Home_Fav","Team_Recent_Cover_Rate_Away_Fav",
        # flags
        "On_Cover_Streak","On_Cover_Streak_Home","On_Cover_Streak_Away",
        "On_Cover_Streak_Fav","On_Cover_Streak_Home_Fav","On_Cover_Streak_Away_Fav"
    ]

    # Always return the canonical "Game_Start" column name, even if we used feat_Game_Start internally
    out = g[["Sport", "Market", "Game_Key", "Team", time_col] + streak_cols].rename(columns={time_col: "Game_Start"})
    return out

def merge_drop_overlap(left, right, on, how="left", *, keep_right=True, validate=None):
    """
    Merge while preventing pandas _x/_y duplicates by dropping overlapping
    non-key columns from the side you DON'T want to keep.
    """
    on = [on] if isinstance(on, str) else list(on)
    overlap = (set(left.columns) & set(right.columns)) - set(on)
    if overlap:
        if keep_right:
            left = left.drop(columns=sorted(overlap), errors="ignore")
        else:
            right = right.drop(columns=sorted(overlap), errors="ignore")
    return left.merge(right, on=on, how=how, validate=validate)

def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1-eps)
    return np.log(p / (1-p))

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def apply_temperature(p, T, eps=1e-6):
    z = _logit(p, eps=eps)
    return np.clip(_sigmoid(z / float(T)), eps, 1-eps)

def fit_temperature_on_oof(y, p, grid=None):
    # T > 1 => soften probabilities (almost always good for spreads)
    if grid is None:
        grid = [1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0]
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    bestT, bestLL = 1.0, 1e18
    for T in grid:
        pp = apply_temperature(p, T)
        ll = log_loss(y, pp, labels=[0,1])
        if ll < bestLL:
            bestLL, bestT = ll, float(T)
    return bestT, bestLL
    
def anchor_to_implied(p_model, p_implied, *, k: float = 0.45, eps: float = 1e-6):
    """
    Anchor model probability toward market-implied probability in logit space.

    k controls how much of the model-vs-market logit gap you keep:
      p_final = sigmoid( logit(p_imp) + k*(logit(p_model) - logit(p_imp)) )

    k=0   -> pure market implied
    k=1   -> pure model
    """
    pm = np.clip(np.asarray(p_model, dtype=float), eps, 1.0 - eps)
    pi = np.clip(np.asarray(p_implied, dtype=float), eps, 1.0 - eps)

    z_m = _logit(pm, eps=eps)
    z_i = _logit(pi, eps=eps)

    z = z_i + float(k) * (z_m - z_i)
    return np.clip(_sigmoid(z), eps, 1.0 - eps)

# ---------------------------
# Leakage guard: snapshot must be <= game start
# ---------------------------
def _audit_and_filter_snapshot_timing(
    df: pd.DataFrame,
    *,
    log=print,
    grace_minutes: float = 500.0,
    return_stats: bool = False,
):
    if df.empty:
        log("[TIME-AUDIT] df is empty; skipping.")
        return (df, None) if return_stats else df

    snap_candidates = [
        "Snapshot_Timestamp", "snapshot_timestamp",
        "Inserted_Timestamp", "Inserted_At",
        "SnapshotTS", "Created_At",
    ]
    snap_col = next((c for c in snap_candidates if c in df.columns), None)

    gs_candidates = ["feat_Game_Start", "Game_Start", "Commence_Hour", "Commence_Time"]
    gs_col = next((c for c in gs_candidates if c in df.columns), None)

    if snap_col is None or gs_col is None:
        log("[TIME-AUDIT] Missing snapshot or game-start column; skipping.")
        return (df, None) if return_stats else df

    snap = pd.to_datetime(df[snap_col], utc=True, errors="coerce")
    gs   = pd.to_datetime(df[gs_col],   utc=True, errors="coerce")

    gs_grace = gs + pd.to_timedelta(float(grace_minutes), unit="m")

    delta = snap - gs
    bad = snap.notna() & gs_grace.notna() & (snap > gs_grace)

    n_bad = int(bad.sum())
    n_all = int(len(df))

    if n_bad:
        log(f"🚨 [LEAK] {n_bad}/{n_all} rows beyond {grace_minutes}m grace")
    else:
        log(f"✅ [TIME-AUDIT] OK: 0/{n_all} rows beyond {grace_minutes}m grace")

    df2 = df.loc[~bad].copy()

    stats = {
        "snap_col": snap_col,
        "gs_col": gs_col,
        "min_delta": delta.min(),
        "max_delta": delta.max(),
    }

    return (df2, stats) if return_stats else df2

# =========================================================
# (A) ADD THIS FUNCTION near build_cover_streaks_game_level()
# =========================================================
def build_schedule_density_game_level(
    df_bt_prepped: pd.DataFrame,
    *,
    sport: str,
    market: str,
    windows_days: tuple[int, ...] = (2, 4, 7),
    b2b_threshold_days: float = 1.0,
) -> pd.DataFrame:
    """
    Leakage-safe schedule density features at GAME grain (one row per team-game).

    Outputs (per Sport/Market/Team/Game_Key):
      - Days_Since_Last_Game
      - Games_Last_{W}_Days for W in windows_days
      - Is_B2B (Days_Since_Last_Game <= b2b_threshold_days)
      - Is_3in4 (Games_Last_4_Days >= 3)  [only if 4 in windows]
    """
    if df_bt_prepped is None or df_bt_prepped.empty:
        return pd.DataFrame(columns=["Sport", "Market", "Game_Key", "Team"])

    sport_u  = str(sport).upper().strip()
    market_l = _norm_market(market)

    time_col = "feat_Game_Start" if "feat_Game_Start" in df_bt_prepped.columns else "Game_Start"
    need = ["Sport", "Market", "Game_Key", "Team", time_col]
    missing = [c for c in need if c not in df_bt_prepped.columns]
    if missing:
        raise ValueError(f"build_schedule_density_game_level missing cols: {missing}")

    d = df_bt_prepped.loc[
        (df_bt_prepped["Sport"].astype(str).str.upper().str.strip() == sport_u) &
        (df_bt_prepped["Market"].astype(str).str.lower().str.strip().map(_norm_market) == market_l),
        need
    ].copy()

    if d.empty:
        return pd.DataFrame(columns=["Sport", "Market", "Game_Key", "Team"])

    # normalize keys
    d["Sport"]  = d["Sport"].astype(str).str.upper().str.strip()
    d["Market"] = d["Market"].astype(str).str.lower().str.strip().map(_norm_market)
    d["Team"]   = d["Team"].astype(str).str.lower().str.strip()

    # parse time axis (UTC)
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce", utc=True)

    # one row per team-game (drop book/snapshot noise)
    g = (
        d.dropna(subset=["Game_Key", "Team", time_col])
         .sort_values(["Sport", "Market", "Team", time_col, "Game_Key"])
         .drop_duplicates(["Sport", "Market", "Team", "Game_Key"], keep="last")
         .copy()
    )

    if g.empty:
        return pd.DataFrame(columns=["Sport", "Market", "Game_Key", "Team"])

    # build per-team arrays for fast searchsorted window counts
    g = g.sort_values(["Sport", "Market", "Team", time_col, "Game_Key"]).reset_index(drop=True)

    # Days since last game (leakage-safe because it's based on prior game time)
    g["Days_Since_Last_Game"] = (
        g.groupby(["Sport", "Market", "Team"])[time_col]
         .diff()
         .dt.total_seconds()
         .div(86400.0)
         .astype("float32")
    )

    # schedule density counts: number of PRIOR games in the last W days
    # For each row i at time t_i, count games with t_j >= t_i - W days AND j < i
    for W in windows_days:
        out_col = f"Games_Last_{int(W)}_Days"
        counts = np.zeros(len(g), dtype=np.int16)

        # group indices for stable vectorized per-team computation
        for _, idx in g.groupby(["Sport", "Market", "Team"], sort=False).indices.items():
            idx = np.asarray(idx, dtype=np.int64)
            t = g.loc[idx, time_col].values.astype("datetime64[ns]")

            # convert to int64 nanos for searchsorted
            t_ns = t.astype("datetime64[ns]").astype("int64")
            # left boundary for each i
            cutoff_ns = t_ns - np.int64(W) * np.int64(86400 * 1_000_000_000)

            # left = first index with time >= cutoff
            left = np.searchsorted(t_ns, cutoff_ns, side="left")
            # position within this group is 0..len-1
            pos = np.arange(len(idx), dtype=np.int64)
            # prior count in window excludes current: pos - left
            c = (pos - left).astype(np.int16)
            counts[idx] = c

        g[out_col] = counts.astype("int16")

    # derived flags
    g["Is_B2B"] = (g["Days_Since_Last_Game"] <= float(b2b_threshold_days)).astype("int8")

    if 4 in windows_days:
        # 3 prior games in last 4 days means "3-in-4 including today" (i.e., current game is the 3rd/4th)
        # If you prefer ">=2 prior games" adjust threshold. Most people use >=2 prior games -> total >=3 in 4 days.
        g["Is_3in4"] = (g["Games_Last_4_Days"] >= 2).astype("int8")  # 2 prior + current = 3-in-4
    else:
        g["Is_3in4"] = 0

    return g[["Sport", "Market", "Game_Key", "Team", "Days_Since_Last_Game", "Is_B2B", "Is_3in4"]
             + [f"Games_Last_{int(W)}_Days" for W in windows_days]]

def add_market_structure_features_training(
    df: pd.DataFrame,
    *,
    sharp_books: list[str],
    rec_books: list[str],
    game_col: str = "Game_Key",
    market_col: str = "Market",
    outcome_col: str = "Outcome",
    book_col: str = "Bookmaker",
) -> pd.DataFrame:
    """
    Training-only, snapshot-safe market structure features.
    Assumes one row per (Game_Key, Market, Outcome, Bookmaker) snapshot.

    Adds:
      - CLV_Prob_Delta, CLV_Line_Pts_Signed
      - Crossed_Key_3/7, Dist_to_3/7 (if spreads)
      - Sharp/Rec consensus mean/std and gaps (prob + line)
      - Pct_Books_Aligned (agreement on direction of belief change)
      - Resistance_Score (intensity, not just flags)
      - Early_Sharp_Flag / Late_Sharp_Flag (regime bins)
    """
    out = df.copy()
    if out.empty:
        return out

    # ---------- helpers ----------
    def _col(*names):
        for n in names:
            if n in out.columns:
                return n
        return None

    gcol = _col(game_col, "feat_Game_Key", "game_key_clean")
    mcol = _col(market_col)
    ocol = _col(outcome_col)
    bcol = _col(book_col, "Book")

    if gcol is None or mcol is None or ocol is None or bcol is None:
        # can't compute group features safely
        return out

    # core numeric columns
    prob_col = _col("Implied_Prob", "Market_Implied_Prob")
    open_prob_col = _col("First_Imp_Prob", "Open_Imp_Prob")
    val_col = _col("Value", "Line_Value", "Line")
    open_val_col = _col("First_Line_Value", "Open_Value", "Opening_Line")
    # fallbacks (keep behavior stable)
    if prob_col is None:
        out["CLV_Prob_Delta"] = 0.0
    else:
        p = pd.to_numeric(out[prob_col], errors="coerce")
        p0 = pd.to_numeric(out[open_prob_col], errors="coerce") if open_prob_col in out.columns else np.nan
        out["CLV_Prob_Delta"] = (p - p0).fillna(0.0).astype("float32")

    # Signed line CLV proxy (best for spreads)
    if (val_col is not None) and (open_val_col is not None) and (open_val_col in out.columns):
        v = pd.to_numeric(out[val_col], errors="coerce")
        v0 = pd.to_numeric(out[open_val_col], errors="coerce")
        raw = (v - v0)

        # If favorite bet: more negative spread = better for favorite; for dog: more positive = better.
        fav_col = _col("Is_Favorite_Bet", "Is_Team_Favorite")
        if fav_col is not None:
            is_fav = pd.to_numeric(out[fav_col], errors="coerce").fillna(0).astype(int)
            sign = np.where(is_fav.values == 1, -1.0, +1.0)
            out["CLV_Line_Pts_Signed"] = (raw.values * sign).astype("float32")
        else:
            out["CLV_Line_Pts_Signed"] = raw.fillna(0.0).astype("float32")
    else:
        out["CLV_Line_Pts_Signed"] = 0.0

    # ---------- key-number logic (spreads only) ----------
    # Works even if you already have Dist_to_3/7; we won't overwrite if present.
    market_lower = out[mcol].astype(str).str.lower()
    is_spreads = market_lower.eq("spreads")

    if "Crossed_Key_3" not in out.columns:
        out["Crossed_Key_3"] = 0
    if "Crossed_Key_7" not in out.columns:
        out["Crossed_Key_7"] = 0

    if (val_col is not None) and (open_val_col is not None) and (open_val_col in out.columns):
        v = pd.to_numeric(out[val_col], errors="coerce")
        v0 = pd.to_numeric(out[open_val_col], errors="coerce")

        # crossing occurs if open and current are on opposite sides of the key OR one equals and the other differs
        for key, name in [(3.0, "Crossed_Key_3"), (7.0, "Crossed_Key_7")]:
            crossed = (
                ((v0 - key) * (v - key) < 0) |
                ((v0 - key) == 0) & ((v - key) != 0) |
                ((v - key) == 0) & ((v0 - key) != 0)
            )
            out.loc[is_spreads, name] = crossed.loc[is_spreads].fillna(False).astype("int8").values

        if "Dist_to_3" not in out.columns:
            out["Dist_to_3"] = np.nan
        if "Dist_to_7" not in out.columns:
            out["Dist_to_7"] = np.nan
        out.loc[is_spreads, "Dist_to_3"] = (v.loc[is_spreads].abs() - 3.0).abs().astype("float32").values
        out.loc[is_spreads, "Dist_to_7"] = (v.loc[is_spreads].abs() - 7.0).abs().astype("float32").values

    # ---------- resistance intensity ----------
    # don't rely on a single flag; build an intensity score.
    lim_up_no_move = pd.to_numeric(out.get("LimitUp_NoMove_Flag", 0), errors="coerce").fillna(0)
    high_limit     = pd.to_numeric(out.get("High_Limit_Flag", 0), errors="coerce").fillna(0)
    # absorption: very narrow traveled range in Value despite movement attempts
    if ("Max_Value" in out.columns) and ("Min_Value" in out.columns):
        vmax = pd.to_numeric(out["Max_Value"], errors="coerce")
        vmin = pd.to_numeric(out["Min_Value"], errors="coerce")
        narrow = ((vmax - vmin).abs() <= 0.25).fillna(False).astype("int8")
    else:
        narrow = 0

    out["Resistance_Score"] = (lim_up_no_move + high_limit + narrow).astype("float32")

    # ---------- timing regime bins ----------
    # If Sharp_Time_Score exists: use it; else fall back to Minutes_To_Game / tier buckets if present.
    if "Sharp_Time_Score" in out.columns:
        s = pd.to_numeric(out["Sharp_Time_Score"], errors="coerce").fillna(0.0)
        out["Early_Sharp_Flag"] = (s >= 0.70).astype("int8")
        out["Late_Sharp_Flag"]  = (s <= 0.30).astype("int8")
    else:
        out["Early_Sharp_Flag"] = 0
        out["Late_Sharp_Flag"] = 0

    # ---------- consensus + dispersion across books ----------
    # Normalize book names to lower for membership tests
    bl = out[bcol].astype(str).str.lower()
    sharp_set = set([str(x).lower() for x in (sharp_books or [])])
    rec_set   = set([str(x).lower() for x in (rec_books or [])])

    out["_is_sharp_book_tmp"] = bl.isin(sharp_set)
    out["_is_rec_book_tmp"]   = bl.isin(rec_set)

    grp_keys = [gcol, mcol, ocol]

    if prob_col is not None and prob_col in out.columns:
        p = pd.to_numeric(out[prob_col], errors="coerce")

        # all books
        g_all = out.groupby(grp_keys, dropna=False)
        out["ImpProb_Mean_AllBooks"] = g_all[p.name].transform("mean").astype("float32")
        out["ImpProb_Std_AllBooks"]  = g_all[p.name].transform("std").fillna(0.0).astype("float32")

        # sharp
        out["_p_tmp"] = p
        sharp_mean = (
            out.loc[out["_is_sharp_book_tmp"]]
               .groupby(grp_keys, dropna=False)["_p_tmp"].mean()
        )
        rec_mean = (
            out.loc[out["_is_rec_book_tmp"]]
               .groupby(grp_keys, dropna=False)["_p_tmp"].mean()
        )
        out["ImpProb_Mean_SharpBooks"] = out.set_index(grp_keys).index.map(sharp_mean).astype("float32")
        out["ImpProb_Mean_RecBooks"]   = out.set_index(grp_keys).index.map(rec_mean).astype("float32")

        out["Sharp_Rec_Prob_Gap"] = (
            (out["ImpProb_Mean_SharpBooks"] - out["ImpProb_Mean_RecBooks"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32")
        )

        # agreement on direction of belief change (book-level) -> percent aligned vs the median direction
        if "CLV_Prob_Delta" in out.columns:
            d = pd.to_numeric(out["CLV_Prob_Delta"], errors="coerce").fillna(0.0)
            dir_med = out.groupby(grp_keys, dropna=False)[d.name].transform("median")
            aligned = (np.sign(d) == np.sign(dir_med)).astype("int8")
            out["Pct_Books_Aligned"] = out.groupby(grp_keys, dropna=False)[aligned.name].transform("mean").fillna(0.0).astype("float32")
        else:
            out["Pct_Books_Aligned"] = 0.0

    else:
        out["ImpProb_Mean_AllBooks"] = 0.0
        out["ImpProb_Std_AllBooks"] = 0.0
        out["ImpProb_Mean_SharpBooks"] = 0.0
        out["ImpProb_Mean_RecBooks"] = 0.0
        out["Sharp_Rec_Prob_Gap"] = 0.0
        out["Pct_Books_Aligned"] = 0.0

    # line consensus gap (optional, but useful)
    if val_col is not None and val_col in out.columns:
        v = pd.to_numeric(out[val_col], errors="coerce")
        out["_v_tmp"] = v
        sharp_v = (
            out.loc[out["_is_sharp_book_tmp"]]
               .groupby(grp_keys, dropna=False)["_v_tmp"].mean()
        )
        rec_v = (
            out.loc[out["_is_rec_book_tmp"]]
               .groupby(grp_keys, dropna=False)["_v_tmp"].mean()
        )
        out["Line_Mean_SharpBooks"] = out.set_index(grp_keys).index.map(sharp_v).astype("float32")
        out["Line_Mean_RecBooks"]   = out.set_index(grp_keys).index.map(rec_v).astype("float32")
        out["Sharp_Rec_Line_Gap"]   = (
            (out["Line_Mean_SharpBooks"] - out["Line_Mean_RecBooks"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32")
        )
        out["Line_Std_AllBooks"] = out.groupby(grp_keys, dropna=False)[v.name].transform("std").fillna(0.0).astype("float32")
    else:
        out["Line_Mean_SharpBooks"] = 0.0
        out["Line_Mean_RecBooks"] = 0.0
        out["Sharp_Rec_Line_Gap"] = 0.0
        out["Line_Std_AllBooks"] = 0.0

    # cleanup temp cols
    for c in ["_is_sharp_book_tmp", "_is_rec_book_tmp", "_p_tmp", "_v_tmp"]:
        if c in out.columns:
            out.drop(columns=[c], inplace=True, errors="ignore")

    return out


def train_sharp_model_from_bq(
    *,
    sport: str = "NBA",
    market: str,
    days_back: int = 35,
    log_func=print,
    bucket_name: str,
    return_artifacts: bool = False,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    SPORT_DAYS_BACK = {"NBA": 365, "NFL": 365, "CFL": 45, "WNBA": 45, "MLB": 60, "NCAAF": 365, "NCAAB": 365}
    days_back = SPORT_DAYS_BACK.get(sport.upper(), days_back)

    st.info(f"🎯 Training sharp model for {sport.upper()} with {days_back} days of historical data...")
    with st.spinner("Pulling training data (cached)…"):
        df = fetch_scores_with_features(sport, days_back)
    if df.empty:
        st.warning("No rows returned for training after filters.")
        return
   
    
    # Work with a single frame going forward
    

    df_bt = df.copy()
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
    # === Columns we expect from the view ===
    history_cols = [
        "After_Win_Flag","Revenge_Flag",
        "Current_Win_Streak_Prior", "Current_Loss_Streak_Prior",
        "H2H_Win_Pct_Prior",  "Opp_WinPct_Prior",
        "Last_Matchup_Result","Last_Matchup_Margin","Days_Since_Last_Matchup",
        "Wins_Last5_Prior",
        "Margin_Last5_Prior",
        "Days_Since_Last_Game",
        "Close_Game_Rate_Prior","Blowout_Game_Rate_Prior",
        "Avg_Home_Margin_Prior","Avg_Away_Margin_Prior",
    ]
    
    # === New team ATS cover / margin stats (prior-only, last-5) ===
    team_cover_cols = [
        # overall cover signal (intentionally off for now)
        "Cover_Rate_Last5",
        "Cover_Rate_After_Win_Last5",
        "Cover_Rate_After_Loss_Last5",
        # situational cover rates (intentionally off for now)
        "Cover_Rate_Home_After_Home_Win_Last5",
        "Cover_Rate_Home_After_Home_Loss_Last5",
        "Cover_Rate_Home_After_Away_Win_Last5",
        "Cover_Rate_Home_After_Away_Loss_Last5",
        "Cover_Rate_Away_After_Home_Win_Last5",
        "Cover_Rate_Away_After_Home_Loss_Last5",
        "Cover_Rate_Away_After_Away_Win_Last5",
        "Cover_Rate_Away_After_Away_Loss_Last5",
        # margin distribution
        "ATS_Cover_Margin_Last5_Prior_Mean",
        "ATS_Cover_Margin_Last5_Prior_Std",
        "Market_Bucket",
        "Market_OddsProb_Bucket",
     ]
    
    # (Opponent mirrors intentionally excluded per your note)
    
    # =============== Schema-safe selection & typing ===============
    all_feature_cols = history_cols + team_cover_cols


    # =============== Schema-safe selection ===============
    all_feature_cols = (
        history_cols
        + team_cover_cols
      
    )
    
    # Keep only columns that actually exist
    history_present = [c for c in history_cols if c in df_bt.columns]
    
    team_cover_present = [c for c in team_cover_cols if c in df_bt.columns]
    #opp_cover_present = [c for c in opp_cover_cols if c in df_bt.columns]
    all_present = history_present + team_cover_present
 
    
    # Handle NaNs if your model can’t
    df_bt[all_present] = df_bt[all_present].fillna(0.0)
    

   

    # Get sport and market
    
        
    # ✅ Make sure helper won't choke if these are missing
    if 'Is_Sharp_Book' not in df_bt.columns:
        df_bt['Is_Sharp_Book'] = df_bt['Bookmaker'].isin(SHARP_BOOKS).astype(int)
    if 'Sharp_Line_Magnitude' not in df_bt.columns:
        df_bt['Sharp_Line_Magnitude'] = pd.to_numeric(df_bt.get('Line_Delta', 0), errors='coerce').abs().fillna(0)
   
    df_bt['Market'] = df_bt['Market'].astype(str).str.lower().str.strip()
    df_bt['Sport']  = df_bt['Sport'].astype(str).str.upper()
    
   
   
    
    # === Get latest snapshot per Game_Key + Market + Outcome (avoid multi-snapshot double counting) ===
    # === Get latest snapshot per Game_Key + Market + Outcome + Book (immutable) ===
    DEDUP_KEY = ['Game_Key','Market','Outcome','Bookmaker']
    df_bt = (
        df_bt.sort_values('Snapshot_Timestamp')
             .drop_duplicates(subset=DEDUP_KEY, keep='last')
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
    #-----do this once right before enrichment) ----
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
                rating_lag_hours=12.0,
            )
        except Exception as e:
            st.exception(e)
            st.stop()


    # === Build hist_df (closers + finals) once, then ρ lookups once ===
    sport_label = sport.upper()
    
    # Ensure df_latest has teams
    if not {"Home_Team_Norm","Away_Team_Norm"}.issubset(df_latest.columns):
        df_latest = df_latest.merge(
            df_bt[["Game_Key","Home_Team_Norm","Away_Team_Norm"]].drop_duplicates("Game_Key"),
            on="Game_Key", how="left"
        )

    # === Build hist_df (closers + finals) once, then ρ lookups once ===
    sport_label = sport.upper()
    
    # Ensure df_latest has teams
    if not {"Home_Team_Norm","Away_Team_Norm"}.issubset(df_latest.columns):
        df_latest = df_latest.merge(
            df_bt[["Game_Key","Home_Team_Norm","Away_Team_Norm"]].drop_duplicates("Game_Key"),
            on="Game_Key", how="left"
        )
    
    # Closers from latest snapshot (per (Game, Market, Outcome, Book))
    closers = df_latest[[
        "Game_Key","Bookmaker","Market","Outcome","Value","Odds_Price",
        "Home_Team_Norm","Away_Team_Norm"
    ]].copy()
    for c in ("Market","Outcome","Home_Team_Norm","Away_Team_Norm"):
        closers[c] = closers[c].astype(str).str.lower().str.strip()
    
    is_spread = closers["Market"].eq("spreads")
    is_total  = closers["Market"].eq("totals")
    is_ml     = closers["Market"].eq("h2h")
    
    # spreads: home/away signed
    sp_home = (closers[is_spread & (closers["Outcome"] == closers["Home_Team_Norm"])]
               .assign(home_spread=pd.to_numeric(
                   closers.loc[is_spread & (closers["Outcome"] == closers["Home_Team_Norm"]), "Value"], errors="coerce"))
               [["Game_Key","Bookmaker","home_spread"]])
    sp_away = (closers[is_spread & (closers["Outcome"] == closers["Away_Team_Norm"])]
               .assign(away_spread=pd.to_numeric(
                   closers.loc[is_spread & (closers["Outcome"] == closers["Away_Team_Norm"]), "Value"], errors="coerce"))
               [["Game_Key","Bookmaker","away_spread"]])
    
    # totals: Over row
    tot_over = (closers[is_total & closers["Outcome"].eq("over")]
                .assign(close_total=pd.to_numeric(
                    closers.loc[is_total & closers["Outcome"].eq("over"), "Value"], errors="coerce"))
                [["Game_Key","Bookmaker","close_total"]])
    
    # ML: both sides → de-vigged fav prob per book
    def _amer_to_prob_vec(x):
        x = pd.to_numeric(x, errors="coerce")
        return np.where(x >= 0, 100.0/(x+100.0), (-x)/((-x)+100.0))
    
    ml_home = closers[is_ml & (closers["Outcome"] == closers["Home_Team_Norm"])][["Game_Key","Bookmaker","Odds_Price"]].rename(columns={"Odds_Price":"ml_home"})
    ml_away = closers[is_ml & (closers["Outcome"] == closers["Away_Team_Norm"])][["Game_Key","Bookmaker","Odds_Price"]].rename(columns={"Odds_Price":"ml_away"})
    ml = ml_home.merge(ml_away, on=["Game_Key","Bookmaker"], how="outer")
    ml["p_home_raw"]     = _amer_to_prob_vec(ml["ml_home"])
    ml["p_away_raw"]     = _amer_to_prob_vec(ml["ml_away"])
    ml["p_ml_fav_book"]  = np.where(
        (ml["p_home_raw"] + ml["p_away_raw"]) > 0,
        np.maximum(ml["p_home_raw"], ml["p_away_raw"]) / (ml["p_home_raw"] + ml["p_away_raw"]),
        np.nan
    )
    
    per_book = (sp_home.merge(sp_away, on=["Game_Key","Bookmaker"], how="outer")
                      .merge(tot_over, on=["Game_Key","Bookmaker"],  how="outer")
                      .merge(ml[["Game_Key","Bookmaker","p_ml_fav_book","p_home_raw","p_away_raw"]],
                             on=["Game_Key","Bookmaker"], how="outer"))
    
    def _collapse_game(g):
        # signed close_spread: favorite negative
        hs = g["home_spread"].dropna()
        as_ = g["away_spread"].dropna()
        if not hs.empty or not as_.empty:
            cand = []
            if not hs.empty: cand.append(hs.min())           # home favorite => negative
            if not as_.empty: cand.append(-abs(as_.min()))   # away favorite => negative
            close_spread = np.nanmin(cand) if cand else np.nan
        else:
            close_spread = np.nan
        return pd.Series(dict(
            close_spread=close_spread,
            close_total = g["close_total"].median(skipna=True),
            p_ml_fav    = g["p_ml_fav_book"].median(skipna=True),
            p_home_raw  = g["p_home_raw"].median(skipna=True),
            p_away_raw  = g["p_away_raw"].median(skipna=True),
        ))
    
    closers_game = per_book.groupby("Game_Key", as_index=False).apply(_collapse_game).reset_index(drop=True)
    
    # finals already loaded as df_results (you did this earlier)
    key_map = (df_bt[["Game_Key","Merge_Key_Short"]].dropna().drop_duplicates("Game_Key"))
    finals_slim = df_results.rename(columns={
        "Score_Home_Score":"Home_Score",
        "Score_Away_Score":"Away_Score"
    })[["Merge_Key_Short","Game_Start","Home_Score","Away_Score","Sport","Home_Team","Away_Team"]]
    
    hist_df = (closers_game
               .merge(key_map, on="Game_Key",  how="left")
               .merge(finals_slim, on="Merge_Key_Short", how="left"))

    with tmr("book path reliability"):
        mkt = df_bt["Market"]  # already normalized
        v   = pd.to_numeric(df_bt["Value"], errors="coerce")
    
        canon_mask = (mkt != "spreads") | (v < 0)   # spreads: fav side only
    
        df_bt = add_book_path_reliability_features(
            df=df_bt,
            closers=closers_game,
            eps_open=0.5,
            prior_strength_open=200.0,
            prior_strength_speed=200.0,
            early_tiers=None,
            assume_normalized=True,     # ✅ NEW
            canon_mask=canon_mask,       # ✅ NEW (big runtime win)
        )

    # Label outcomes
    margin    = pd.to_numeric(hist_df["Home_Score"], errors="coerce") - pd.to_numeric(hist_df["Away_Score"], errors="coerce")
    total_pts = pd.to_numeric(hist_df["Home_Score"], errors="coerce") + pd.to_numeric(hist_df["Away_Score"], errors="coerce")
    home_spread_med = per_book.groupby("Game_Key")["home_spread"].median()
    away_spread_med = per_book.groupby("Game_Key")["away_spread"].median()
    hist_df = hist_df.merge(home_spread_med.rename("home_spread_med"), on="Game_Key", how="left") \
                     .merge(away_spread_med.rename("away_spread_med"), on="Game_Key", how="left")
    
    home_fav = np.where(
        pd.to_numeric(hist_df["home_spread_med"], errors="coerce") < 0, True,
        np.where(pd.to_numeric(hist_df["away_spread_med"], errors="coerce") < 0, False, np.nan)
    )
    
    hist_df["fav_covered"] = np.where(
        np.isnan(home_fav), np.nan,
        np.where(home_fav,
                 margin > -pd.to_numeric(hist_df["home_spread_med"], errors="coerce"),
                 (-margin) > -pd.to_numeric(hist_df["away_spread_med"], errors="coerce"))
    )
    hist_df["fav_won"]   = np.where(hist_df[["p_home_raw","p_away_raw"]].isna().any(axis=1), np.nan,
                                    np.where(hist_df["p_home_raw"] >= hist_df["p_away_raw"], margin > 0, margin < 0))
    hist_df["went_over"] = total_pts > pd.to_numeric(hist_df["close_total"], errors="coerce")
    hist_df["Sport"]     = hist_df["Sport"].astype(str).str.upper()
        # ============================
    # FIX LABEL DIRECTION (SPREADS)
    # ============================
    # hist_df["fav_covered"] is a *favorite-side* label at the game level.
    # Your training rows are team/outcome-level, so underdog rows must get (1 - fav_covered).

    _fav_lbl = hist_df[["Merge_Key_Short", "fav_covered"]].copy()
    _fav_lbl["Merge_Key_Short"] = _fav_lbl["Merge_Key_Short"].astype(str).str.strip().str.lower()

    # df_bt already has Merge_Key_Short normalized earlier; ensure anyway
    df_bt["Merge_Key_Short"] = df_bt["Merge_Key_Short"].astype(str).str.strip().str.lower()

    df_bt = df_bt.merge(_fav_lbl, on="Merge_Key_Short", how="left", suffixes=("", "_favtmp"))

    _m_sp = df_bt["Market"].astype(str).str.lower().str.strip().eq("spreads")
    _y_fav = pd.to_numeric(df_bt["fav_covered"], errors="coerce")
    _val   = pd.to_numeric(df_bt["Value"], errors="coerce")
    _is_fav_row = (_val < 0)

    _mask = _m_sp & _y_fav.notna() & _val.notna()

    # Favorite row keeps fav_covered; underdog row gets 1 - fav_covered
    df_bt.loc[_mask, "SHARP_HIT_BOOL"] = np.where(
        _is_fav_row.loc[_mask],
        _y_fav.loc[_mask].astype(float),
        (1.0 - _y_fav.loc[_mask].astype(float))
    ).astype("float32")

    # cleanup helper column
    df_bt.drop(columns=["fav_covered"], inplace=True, errors="ignore")

    # ρ lookups (Spread↔Total, Spread↔ML, Total↔ML) — do this ONCE
    
    ST_lookup, SM_lookup, TM_lookup = build_corr_lookup_ST_SM_TM(hist_df, sport=sport_label)
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
        pad_days=30,
        rating_lag_hours=12.0,   # strict backward-only
        table_history="sharplogger.sharp_data.ratings_history",
        project="sharplogger",
    )
    
    # Force all three markets to exist in the pivot, even if empty
    MARKETS = ["spreads", "totals", "h2h"]
    
    value_pivot = (
        df_latest.pivot_table(
            index="Game_Key", columns="Market", values="Value",
            aggfunc="last", dropna=False  # keep all-NaN columns
        )
        .reindex(columns=MARKETS)  # ensure columns exist in this order
        .rename(columns={"spreads":"Spread_Value", "totals":"Total_Value", "h2h":"H2H_Value"})
    )
    
    odds_pivot = (
        df_latest.pivot_table(
            index="Game_Key", columns="Market", values="Odds_Price",
            aggfunc="last", dropna=False
        )
        .reindex(columns=MARKETS)
        .rename(columns={"spreads":"Spread_Odds", "totals":"Total_Odds", "h2h":"H2H_Odds"})
    )
    
    df_cross_market = (
        value_pivot
        .join(odds_pivot, how="outer")
        .reset_index()
    )
    
   
    # --- Safe odds/prob scaffolding (post-merge) ---
    
    

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
            df = df[df["Is_Home"] == home_filter]
        if favorite_filter is not None:
            df = df[df["Is_Favorite_Context"] == favorite_filter]
    
        if df.empty:
            return pd.DataFrame(columns=[
                "Sport","Market","Game_Key","Team",
                f"Team_Past_Avg_Model_Prob{col_suffix}",
                f"Team_Past_Hit_Rate{col_suffix}",
            ])
    
        df["Game_Start"] = pd.to_datetime(df.get("Game_Start"), errors="coerce", utc=True)
        df["Snapshot_Timestamp"] = pd.to_datetime(df.get("Snapshot_Timestamp"), errors="coerce", utc=True)
        df["_t"] = df["Game_Start"].fillna(df["Snapshot_Timestamp"])
    
        df["Model_Sharp_Win_Prob"] = pd.to_numeric(df.get("Model_Sharp_Win_Prob"), errors="coerce")
        df["SHARP_HIT_BOOL"] = pd.to_numeric(df.get("SHARP_HIT_BOOL"), errors="coerce")
    
        key_cols = ["Sport","Market","Team","Game_Key"]
    
        df_game = (
            df.dropna(subset=["Game_Key","Team","_t"])
              .sort_values(key_cols + ["_t"])
              .groupby(key_cols, as_index=False)
              .agg(
                  _t=("_t","min"),
                  Model_Sharp_Win_Prob=("Model_Sharp_Win_Prob","mean"),
                  SHARP_HIT_BOOL=("SHARP_HIT_BOOL","max"),
              )
        )
    
        group_keys = ["Sport","Market","Team"]
        df_game = df_game.sort_values(group_keys + ["_t"])
    
        df_game["cum_model_prob"] = df_game.groupby(group_keys)["Model_Sharp_Win_Prob"].cumsum().shift(1)
        df_game["cum_hit"]        = df_game.groupby(group_keys)["SHARP_HIT_BOOL"].cumsum().shift(1)
        df_game["cum_count"]      = df_game.groupby(group_keys).cumcount()
    
        avg_col = f"Team_Past_Avg_Model_Prob{col_suffix}"
        hit_col = f"Team_Past_Hit_Rate{col_suffix}"
        df_game[avg_col] = df_game["cum_model_prob"] / df_game["cum_count"].replace(0, np.nan)
        df_game[hit_col] = df_game["cum_hit"]        / df_game["cum_count"].replace(0, np.nan)
    
        return df_game[["Sport","Market","Game_Key","Team", avg_col, hit_col]]

    
  
    # LOO sets — computed from the FULL base
    overall_stats = compute_loo_stats_by_game(df_bt_prepped)
    home_stats    = compute_loo_stats_by_game(df_bt_prepped, home_filter=1, col_suffix="_Home")
    away_stats    = compute_loo_stats_by_game(df_bt_prepped, home_filter=0, col_suffix="_Away")
    fav_overall   = compute_loo_stats_by_game(df_bt_prepped, favorite_filter=1, col_suffix="_Fav")
    fav_home      = compute_loo_stats_by_game(df_bt_prepped, home_filter=1, favorite_filter=1, col_suffix="_Home_Fav")
    fav_away      = compute_loo_stats_by_game(df_bt_prepped, home_filter=0, favorite_filter=1, col_suffix="_Away_Fav")
    
    # LOO map at (Game_Key, Team)
    df_bt_loostats = (
        df_bt_prepped[["Sport","Market","Game_Key","Team"]].drop_duplicates()
          .merge(overall_stats, on=["Sport","Market","Game_Key","Team"], how="left", validate="1:1")
          .merge(home_stats,    on=["Sport","Market","Game_Key","Team"], how="left", validate="1:1")
          .merge(away_stats,    on=["Sport","Market","Game_Key","Team"], how="left", validate="1:1")
          .merge(fav_overall,   on=["Sport","Market","Game_Key","Team"], how="left", validate="1:1")
          .merge(fav_home,      on=["Sport","Market","Game_Key","Team"], how="left", validate="1:1")
          .merge(fav_away,      on=["Sport","Market","Game_Key","Team"], how="left", validate="1:1")
    )
    
    
   

    before = len(df_bt)
    df_bt = df_bt.drop_duplicates(subset=dedup_cols, keep='last')
    after = len(df_bt)

    # Container for full multi-market training (legacy path)
    trained_models: Dict[str, Any] = {}

    # Containers used only when return_artifacts=True (champion/challenger mode)
    artifact_metrics: Dict[str, float] = {}
    artifact_model_path: Optional[str] = None
    artifact_config: Dict[str, Any] = {}

    # derive markets present (filter to the 3 you care about)
    allowed = {'spreads', 'totals', 'h2h'}
    if return_artifacts:
        # Champion/challenger mode trains exactly one requested market
        requested = str(market).lower()
        if requested not in allowed:
            st.error(f"Unsupported market '{market}'. Must be one of {sorted(allowed)}.")
            return None

        df_bt = df_bt[df_bt['Market'].astype(str).str.lower() == requested].copy()
        if df_bt.empty:
            st.error(f"No training rows found for {sport} {requested}.")
            return None

        markets_present = [requested]
    else:
        markets_present = [
            m for m in df_bt['Market'].astype(str).str.lower().unique()
            if m in allowed
        ]

    n_markets = max(1, len(markets_present))
    pb = st.progress(0)  # 0–100
    status = st.status("🔄 Training in progress...", expanded=True)

    
    for i, market in enumerate(markets_present, 1):
        pct = int(round(i / n_markets * 100))
        status.write(f"🚧 Training model for `{str(market).upper()}`...")
    
        mkt = _norm_market(market)

        df_market = df_bt[df_bt["Market"].astype(str).str.lower().str.strip() == mkt].copy()
    
                # ---------------------------
                # Streamlit-friendly logger
        def _log(msg):
            try:
                status.write(msg)
            except Exception:
                print(msg)
        
        # 0-minute grace by default. If your timestamps are known noisy, try grace_minutes=1.0
        df_market, timing_stats = _audit_and_filter_snapshot_timing(
            df_market,
            log=_log,
            grace_minutes=400.0,
            return_stats=True,
        )
        
        if timing_stats is not None:
            _log(
                f"[SANITY] Δ(snapshot - start): "
                f"min={timing_stats['min_delta']} | "
                f"max={timing_stats['max_delta']}"
            )


        # --- Canonical side filter ---
        # === Canonical side filtering ONLY (training subset) ===
       # ===========================
        # Canonical side filtering ONLY (training subset)
        # ===========================
        if mkt == "totals":
            # keep Over only (fine for totals modeling)
            df_market = df_market[df_market["Outcome"].astype(str).str.lower().str.strip() == "over"].copy()
        
        elif mkt == "spreads":
            # ✅ keep BOTH favorite & dog rows for training
            df_market = df_market.copy()
            df_market["Value"] = pd.to_numeric(df_market["Value"], errors="coerce")
            df_market = df_market[df_market["Value"].notna()].copy()
        
        elif mkt == "h2h":
            df_market = df_market.copy()
        
        if df_market.empty:
            pb.progress(min(100, max(0, pct)))
            continue
        
        # keep latest snapshot per immutable key
        df_market = (
            df_market.sort_values("Snapshot_Timestamp")
                     .drop_duplicates(subset=["Game_Key", "Market", "Outcome", "Bookmaker"], keep="last")
                     .copy()
        )
        
        # ---- PREPPED slice (this market only) ----
        df_prepped_mkt = df_bt_prepped[
            df_bt_prepped["Market"].astype(str).str.lower().str.strip().map(_norm_market) == mkt
        ].copy()
        
        # normalize keys for consistent merges (PREPPED + LOO)
        for _df in (df_prepped_mkt, df_bt_loostats):
            _df["Sport"]  = _df["Sport"].astype(str).str.upper().str.strip()
            _df["Market"] = _df["Market"].astype(str).str.lower().str.strip().map(_norm_market)
            _df["Team"]   = _df["Team"].astype(str).str.lower().str.strip()
        
        # ---------------------------------------------------------
        # Build game-grain streaks (Game_Key + Team)
        # ---------------------------------------------------------
        df_bt_streaks = build_cover_streaks_game_level(df_prepped_mkt, sport=sport, market=mkt)
        df_bt_streaks["Sport"]  = df_bt_streaks["Sport"].astype(str).str.upper().str.strip()
        df_bt_streaks["Market"] = df_bt_streaks["Market"].astype(str).str.lower().str.strip().map(_norm_market)
        df_bt_streaks["Team"]   = df_bt_streaks["Team"].astype(str).str.lower().str.strip()
        
        # ---------------------------------------------------------
        # Build schedule density features (Game_Key + Team)
        # IMPORTANT: for totals, Team key must match df_market Team (Home_Team_Norm)
        # ---------------------------------------------------------
        df_sched_base = df_prepped_mkt.copy()
        df_sched_base["Sport"]  = df_sched_base["Sport"].astype(str).str.upper().str.strip()
        df_sched_base["Market"] = df_sched_base["Market"].astype(str).str.lower().str.strip().map(_norm_market)
        
        # mirror df_market Team logic
        if "Outcome" in df_sched_base.columns:
            df_sched_base["Outcome_Norm"] = df_sched_base["Outcome"].astype(str).str.lower().str.strip()
        else:
            df_sched_base["Outcome_Norm"] = ""
        
        df_sched_base["Home_Team_Norm"] = df_sched_base["Home_Team_Norm"].astype(str).str.lower().str.strip()
        df_sched_base["Away_Team_Norm"] = df_sched_base["Away_Team_Norm"].astype(str).str.lower().str.strip()
        
        is_totals_sched = df_sched_base["Market"].eq("totals")
        df_sched_base["Team"] = (
            df_sched_base["Outcome_Norm"].where(~is_totals_sched, df_sched_base["Home_Team_Norm"])
        ).astype(str).str.lower().str.strip()
        
        df_bt_sched = build_schedule_density_game_level(
            df_sched_base,
            sport=sport,
            market=mkt,
            windows_days=(2, 4, 7),
            b2b_threshold_days=1.0,
        )
        df_bt_sched["Sport"]  = df_bt_sched["Sport"].astype(str).str.upper().str.strip()
        df_bt_sched["Market"] = df_bt_sched["Market"].astype(str).str.lower().str.strip().map(_norm_market)
        df_bt_sched["Team"]   = df_bt_sched["Team"].astype(str).str.lower().str.strip()
        
        # ---------------------------------------------------------
        # Normalize df_market keys + team mapping
        # ---------------------------------------------------------
        df_market["Outcome_Norm"]   = df_market["Outcome"].astype(str).str.lower().str.strip()
        df_market["Home_Team_Norm"] = df_market["Home_Team_Norm"].astype(str).str.lower().str.strip()
        df_market["Away_Team_Norm"] = df_market["Away_Team_Norm"].astype(str).str.lower().str.strip()
        df_market["Sport"]          = df_market["Sport"].astype(str).str.upper().str.strip()
        df_market["Market"]         = df_market["Market"].astype(str).str.lower().str.strip().map(_norm_market)
        
        is_totals = df_market["Market"].eq("totals")
        
        df_market["Team"] = (
            df_market["Outcome_Norm"].where(~is_totals, df_market["Home_Team_Norm"])
        ).astype(str).str.lower().str.strip()
        
        df_market["Is_Home"] = np.where(
            is_totals, 1,
            (df_market["Team"] == df_market["Home_Team_Norm"]).astype(int)
        ).astype(int)
        
        # ---------------------------------------------------------
        # Time-safe per-game merges into df_market  (NOTE validate)
        # ---------------------------------------------------------
        df_market = df_market.merge(
            df_bt_streaks.drop(columns=["Game_Start"], errors="ignore"),
            on=["Sport", "Market", "Game_Key", "Team"],
            how="left",
            validate="many_to_one",
        )
        
        df_market = df_market.merge(
            df_bt_loostats,
            on=["Sport", "Market", "Game_Key", "Team"],
            how="left",
            validate="many_to_one",
        )
        
        # ✅ schedule density features we own at GAME level (avoid _x/_y collisions)

        SCHED_COLS = [
            "Days_Since_Last_Game",
            "Games_Last_2_Days",
            "Games_Last_4_Days",
            "Games_Last_7_Days",
            "Is_B2B",
            "Is_3in4",
        ]
        
        df_market = df_market.drop(
            columns=[c for c in SCHED_COLS if c in df_market.columns],
            errors="ignore",
        )
        
        df_market = df_market.merge(
            df_bt_sched,
            on=["Sport", "Market", "Game_Key", "Team"],
            how="left",
            validate="many_to_one",
        )
                
        # ---------------------------------------------------------
        # Build team_feature_map (state) INCLUDING schedule density (for priors / UI),
        # but NOTE: when merging team_feature_map back into df_market later,
        # exclude SCHED_COLS to avoid x/y collisions.
        # ---------------------------------------------------------
        df_team_base = (
            df_sched_base[["Sport", "Market", "Game_Key", "Team", "Game_Start"]]
              .dropna(subset=["Game_Key", "Team", "Game_Start"])
              .sort_values(["Sport", "Market", "Team", "Game_Start"])
              .drop_duplicates(["Sport", "Market", "Team", "Game_Key"], keep="last")
              .copy()
        )
        
        df_team_base["Sport"]  = df_team_base["Sport"].astype(str).str.upper().str.strip()
        df_team_base["Market"] = df_team_base["Market"].astype(str).str.lower().str.strip().map(_norm_market)
        df_team_base["Team"]   = df_team_base["Team"].astype(str).str.lower().str.strip()
        df_team_base["Game_Start"] = pd.to_datetime(df_team_base["Game_Start"], errors="coerce", utc=True)
        
        df_team_base = df_team_base.merge(
            df_bt_loostats,
            on=["Sport", "Market", "Game_Key", "Team"],
            how="left",
            validate="1:1",
        )
        
        df_team_base = df_team_base.merge(
            df_bt_streaks.drop(columns=["Game_Start"], errors="ignore"),
            on=["Sport", "Market", "Game_Key", "Team"],
            how="left",
            validate="1:1",
        )
       
        df_team_base = df_team_base.merge(
            df_bt_sched,
            on=["Sport", "Market", "Game_Key", "Team"],
            how="left",
            validate="1:1",
        )
        
        df_team_base = df_team_base.sort_values(["Sport", "Market", "Team", "Game_Start"])

        
        STATE_COLS = [c for c in df_bt_streaks.columns if c not in ["Sport","Market","Game_Key","Team","Game_Start"]]
        LOO_PRIOR_COLS = [c for c in df_bt_loostats.columns if c not in ["Sport","Market","Game_Key","Team"]]
        
        # aggregation: use LAST for both state + priors (recommended)
        agg_spec = {c: "last" for c in (STATE_COLS + LOO_PRIOR_COLS) if c in df_team_base.columns}
        
        team_feature_map = (
            df_team_base.groupby(["Sport","Market","Team"], as_index=False).agg(agg_spec)
        )
        
        # numeric coercion for the state cols (optional)
        for c in STATE_COLS + LOO_PRIOR_COLS:
            if c in team_feature_map.columns:
                team_feature_map[c] = pd.to_numeric(team_feature_map[c], errors="coerce")

        def _amer_to_prob_vec(s):
            s = pd.to_numeric(s, errors="coerce")
            return np.where(
                s > 0,
                100.0 / (s + 100.0),
                np.where(s < 0, (-s) / ((-s) + 100.0), np.nan),
            ).astype("float32")

        # === Merge game-level cross-market values (Spread_Value / Total_Value / H2H_Value) ===
        df_market = merge_drop_overlap(
            df_market,
            df_cross_market,
            on="Game_Key",
            how="left",
            keep_right=True,  # keep df_cross_market versions
        )
        
        # === GAME-LEVEL value-aware features: compute once per Game_Key, then broadcast ===
        value_cols = [
            "Game_Key",
            "Spread_Abs_Game", "Spread_Abs_Game_Z", 
            "Total_Game", "Total_Game_Z", 
            "Spread_x_Total", "Spread_over_Total", "Total_over_Spread",
            "Dist_to_3", "Dist_to_7", "Dist_to_10",
        ]
        
        # --- Build game_vals safely (always defined) ---
        base_cols = [c for c in ["Game_Key", "Sport", "Spread_Value", "Total_Value"] if c in df_market.columns]
        if "Game_Key" not in base_cols:
            raise ValueError("Expected 'Game_Key' in df_market before computing game-level features.")
        
        game_vals = (
            df_market[base_cols]
            .drop_duplicates("Game_Key")
            .copy()
        )
        
        # Ensure Sport exists for grouping; if missing, treat all as one sport group
        if "Sport" not in game_vals.columns:
            game_vals["Sport"] = "ALL"
        
        # Pre-create output columns so downstream selection never KeyErrors
        for c in value_cols:
            if c not in game_vals.columns:
                game_vals[c] = np.nan
        
        if not game_vals.empty:
            # Make sure these are numeric (even if missing)
            if "Spread_Value" in game_vals.columns:
                game_vals["Spread_Value"] = pd.to_numeric(game_vals["Spread_Value"], errors="coerce")
            else:
                game_vals["Spread_Value"] = np.nan
        
            if "Total_Value" in game_vals.columns:
                game_vals["Total_Value"] = pd.to_numeric(game_vals["Total_Value"], errors="coerce")
            else:
                game_vals["Total_Value"] = np.nan
        
            # Absolute spread magnitude (fav line) and total points for the game
            game_vals["Spread_Abs_Game"] = game_vals["Spread_Value"].abs()
            game_vals["Total_Game"]      = game_vals["Total_Value"]
        
            # Z-scores per sport (safe even when constant/empty)
            def _zsafe(s: pd.Series) -> pd.Series:
                s = s.astype(float)
                if s.notna().sum() < 2:
                    return pd.Series(np.zeros(len(s)), index=s.index)
                fill = s.median() if s.notna().any() else 0.0
                return pd.Series(zscore(s.fillna(fill), ddof=0), index=s.index)
        
            game_vals["Spread_Abs_Game_Z"] = game_vals.groupby("Sport")["Spread_Abs_Game"].transform(_zsafe)
            game_vals["Total_Game_Z"]      = game_vals.groupby("Sport")["Total_Game"].transform(_zsafe)
        

            # Spread size bucket (MODEL-ONLY; generic but meaningful)
            game_vals["Model_Spread_Size_Bucket"] = pd.cut(
                game_vals["Spread_Abs_Game"],
                bins=[-0.01, 2.5, 6.5, 10.5, np.inf],
                labels=["Close", "Medium", "Large", "Huge"],
                include_lowest=True,
            ).astype(str)
            
            
            # Total size bucket (MODEL-ONLY): per-sport quantiles, robust to duplicate edges
            game_vals["Model_Total_Size_Bucket"] = "NA"
            
            for sport_val, g in game_vals.groupby("Sport"):
                idx = g.index
            
                tg = g["Total_Game"].dropna()
                if tg.shape[0] < 5:
                    continue
            
                q1, q2, q3 = tg.quantile([0.25, 0.50, 0.75]).values.tolist()
                bins = [-np.inf, q1, q2, q3, np.inf]
            
                # de-dup consecutive equal edges (prevents "Bin edges must be unique")
                dedup_bins = []
                for b in bins:
                    if not dedup_bins or b != dedup_bins[-1]:
                        dedup_bins.append(b)
            
                if len(dedup_bins) < 3:
                    game_vals.loc[idx, "Model_Total_Size_Bucket"] = "Flat"
                else:
                    default_labels = ["Low", "Medium", "High", "Very_High"]
                    labels = default_labels[: (len(dedup_bins) - 1)]
            
                    # ✅ assign only where Total_Game is present for this sport
                    mask = idx.intersection(g["Total_Game"].dropna().index)
            
                    game_vals.loc[mask, "Model_Total_Size_Bucket"] = pd.cut(
                        game_vals.loc[mask, "Total_Game"],
                        bins=dedup_bins,
                        labels=labels,
                        include_lowest=True,
                    ).astype(str)

            # Interactions / volatility proxies
            game_vals["Spread_x_Total"]    = game_vals["Spread_Abs_Game"] * game_vals["Total_Game"]
            game_vals["Spread_over_Total"] = game_vals["Spread_Abs_Game"] / game_vals["Total_Game"].replace(0, np.nan)
            game_vals["Total_over_Spread"] = game_vals["Total_Game"] / game_vals["Spread_Abs_Game"].replace(0, np.nan)
        
            # (Optional but 🔥 for NFL/NCAAF): key number distances
            is_fb = game_vals["Sport"].astype(str).isin(["NFL", "NCAAF"])
            if is_fb.any():
                abs_sp = game_vals.loc[is_fb, "Spread_Abs_Game"]
                game_vals.loc[is_fb, "Dist_to_3"]  = (abs_sp - 3).abs()
                game_vals.loc[is_fb, "Dist_to_7"]  = (abs_sp - 7).abs()
                game_vals.loc[is_fb, "Dist_to_10"] = (abs_sp - 10).abs()
        
        # --- Merge back to every training row for the game (single merge, no duplicates) ---
        df_market = merge_drop_overlap(
            df_market,
            game_vals[value_cols],
            on="Game_Key",
            how="left",
            keep_right=True,
        )
        
        # === Guard: ensure value-aware columns exist even if game_vals was empty ===
        for c in value_cols:
            if c not in df_market.columns:
                df_market[c] = np.nan
        
        # ensure buckets are plain strings (not Categorical / NaN)
        for c in ["Model_Spread_Size_Bucket", "Model_Total_Size_Bucket"]:
            df_market[c] = df_market[c].astype(str).replace("nan", "").fillna("")

        # === Implied probabilities directly from Odds_Price by market (unchanged in spirit) ===
        df_market["Market_Implied_Prob"] = _amer_to_prob_vec(df_market["Odds_Price"])
        df_market["Spread_Implied_Prob"] = np.where(
            df_market["Market"].eq("spreads"), df_market["Market_Implied_Prob"], np.nan
        )
        df_market["Total_Implied_Prob"] = np.where(
            df_market["Market"].eq("totals"), df_market["Market_Implied_Prob"], np.nan
        )
        df_market["H2H_Implied_Prob"] = np.where(
            df_market["Market"].eq("h2h"), df_market["Market_Implied_Prob"], np.nan
        )



        # Your existing FE before resistance
        df_market = compute_small_book_liquidity_features(df_market)
        df_market = add_favorite_context_flag(df_market)
        
        # Final-snapshot training slice (no timestamps needed)
        df_market = add_resistance_features_training(
            df_market,
            emit_levels_str=False,  # keep memory tiny for training
        )
        df_market = compute_ev_features_sharp_vs_rec(
            df_market,
            sharp_books=SHARP_BOOKS,
            reliability_col="Book_Reliability_Score",
            limit_col="Sharp_Limit_Total",
            sigma_col="Sigma_Pts",  # your spreads block already tries to create Sigma_Pts
        )
       # === NEW: final-snapshot microstructure + hybrid timing (before normalization/canonical filter) ===
        df_market = compute_snapshot_micro_features_training(
            df_market,
            sport_col="Sport",
            market_col="Market",
            value_col="Value",
            open_col="First_Line_Value" if "First_Line_Value" in df_market.columns else "Open_Value",
            prob_col="Implied_Prob" if "Implied_Prob" in df_market.columns else None,
            price_col="Odds_Price" if "Odds_Price" in df_market.columns else None,
        )

        df_market = add_market_structure_features_training(
            df_market,
            sharp_books=SHARP_BOOKS,
            rec_books=REC_BOOKS,
        )
        # === Compact domain interactions ===
        df_market['ResistBreak_x_Mag']     = df_market.get('Was_Line_Resistance_Broken',0) * df_market.get('Abs_Line_Move_From_Opening',0).fillna(0)
        df_market['LateSteam_x_KeyCount']  = df_market.get('Potential_Overmove_Flag',0)   * df_market.get('Line_Resistance_Crossed_Count',0).fillna(0)
        df_market['Aligned_x_HighLimit']   = df_market.get('Line_Moved_Toward_Team',0)    * (df_market.get('Sharp_Limit_Total',0) >= 7000).astype(int)
        # helpers to keep things aligned & NA-safe
        def _series_or_default(df, col, default=0.0, dtype="float32"):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
            # return an aligned Series, not a scalar
            return pd.Series(default, index=df.index, dtype=dtype)
        
        val_rev  = _series_or_default(df_market, "Value_Reversal_Flag")
        odds_rev = _series_or_default(df_market, "Odds_Reversal_Flag")
        
        rev_flag_bool = (val_rev.fillna(0).eq(1) | odds_rev.fillna(0).eq(1))
        df_market["Rev_x_BookLift"] = rev_flag_bool.astype("int8") * _series_or_default(df_market, "Book_Reliability_Lift").fillna(0)

        df_market['GapExists_x_RecSkew']   = df_market.get('CrossMarket_Prob_Gap_Exists',0) * df_market.get('SmallBook_Limit_Skew_Flag',0)
        # helper that returns an aligned Series (not a scalar) when the col is missing
        def _series_or_default(df, col, default, dtype=None):
            if col in df.columns:
                s = df[col]
            else:
                s = pd.Series(default, index=df.index)
            if dtype is not None:
                # only after alignment
                try:
                    s = s.astype(dtype)
                except Exception:
                    pass
            return s
        
        # 1) model/market (dis)agreement → 1 if misaligned else 0
        # default=1 preserves your previous intent: treat missing as "agree"
        agree = pd.to_numeric(_series_or_default(df_market, 'model_fav_vs_market_fav_agree', 1),
                              errors='coerce').fillna(1)
        misalign_flag = (agree == 0).astype('int8')
        
        # 2) move-away flag as aligned numeric
        move_away = pd.to_numeric(_series_or_default(df_market, 'Line_Moved_Away_From_Team', 0),
                                  errors='coerce').fillna(0).astype('int8')
        
        # 3) interaction
        df_market['Misalign_x_MoveAway'] = (misalign_flag * move_away).astype('int8')


        # If you don’t want line-based metrics on H2H, zero them now
        if market == "h2h":
            for c in ["Dist_To_Next_Key","Key_Corridor_Pressure","Book_PctRank_Line",
                      "Book_Line_Diff_vs_SharpMedian","Outlier_Flag_SharpBooks"]:
                if c in df_market.columns:
                    df_market[c] = 0 if df_market[c].dtype.kind in "iu" else 0.0
        
        # Hybrid timing derivatives from your persisted hybrid columns
        df_market = compute_hybrid_timing_derivatives_training(df_market)
        
        # Tight dtypes for small flags
        for c in ["Two_Sided_Offered","Outlier_Flag_SharpBooks"]:
            if c in df_market.columns:
                df_market[c] = df_market[c].astype("uint8")
        # === END NEW ===

        
        # Optional: zero out for h2h if you don't want resistance there
        if market == "h2h":
            df_market[["Was_Line_Resistance_Broken","Line_Resistance_Crossed_Count","SharpMove_Resistance_Break"]] = 0
        
        # === Make sure these are in your feature set ===
        resist_feats = ["Was_Line_Resistance_Broken","Line_Resistance_Crossed_Count","SharpMove_Resistance_Break"]
       
   
        

        if df_market.empty:
            status.warning(f"⚠️ No data for {market.upper()} — skipping.")
            pb.progress(min(100, max(0, pct)))
            continue
    
        
        
        
        # Normalize...
        df_market['Outcome_Norm']   = df_market['Outcome'].astype(str).str.lower().str.strip()
        df_market['Home_Team_Norm'] = df_market['Home_Team_Norm'].astype(str).str.lower().str.strip()
        df_market['Away_Team_Norm'] = df_market['Away_Team_Norm'].astype(str).str.lower().str.strip()
        df_market['Market']         = df_market['Market'].astype(str).str.lower().str.strip()
        df_market['Game_Key']       = df_market['Game_Key'].astype(str).str.lower().str.strip()
        
        
        # 1) Limit dynamics
        df_market = add_limit_dynamics_features(df_market)
        
        # 2) Bookmaker network features
        df_market = add_book_network_features(df_market)
        
        # 3) Internal consistency (optionally pass your df_cross_market pivot if you have it)
        df_market = add_internal_consistency_features(df_market, df_cross_market=df_cross_market if 'df_cross_market' in locals() else None, sport_default=sport)
        
        
        
        # 5) CLV proxy (heuristic by default; will use a tiny side-model if present)
        df_market = add_clv_proxy_features(df_market, clv_model_path=os.path.join("artifacts","clv_side.joblib"))

        
        # --- Team / Is_Home (totals anchor to home)
        is_totals = df_market['Market'].eq('totals')

        # --- make sure context keys are normalized the same way
        #df_bt_context['Game_Key'] = df_bt_context['Game_Key'].astype(str).str.lower().str.strip()
        #df_bt_context['Team']     = df_bt_context['Team'].astype(str).str.lower().str.strip()
        
        # --- guard selected columns
        #keep_ctx = ['Game_Key','Team'] + [c for c in context_cols if c in df_bt_context.columns]
        
   
        
        
        # === Ensure minimal normalization BEFORE canonical side filter ===
        if "Outcome_Norm" not in df_market.columns:
            df_market["Outcome_Norm"] = (
                df_market.get("Outcome", "")
                .astype(str).str.lower().str.strip()
            )
        else:
            df_market["Outcome_Norm"] = df_market["Outcome_Norm"].astype(str).str.lower().str.strip()
        
        df_market["Market"] = df_market.get("Market", "").astype(str).str.lower().str.strip()
        df_market["Value"]  = pd.to_numeric(df_market.get("Value", np.nan), errors="coerce")
        
        # === Canonical side filtering ONLY (training subset) ===
        # === Canonical side filtering ONLY (training subset) ===
        if market == "totals":
            df_market = df_market[df_market["Outcome_Norm"] == "over"]
        
        elif market == "spreads":
            # ✅ keep BOTH sides (no Value sign filter)
            df_market = df_market[pd.to_numeric(df_market["Value"], errors="coerce").notna()]
        
        elif market == "h2h":
            # ✅ no Value<0 filtering for h2h
            df_market = df_market.copy()

        
        # === Labels ===
        df_market = df_market[df_market["SHARP_HIT_BOOL"].isin([0, 1])]
        if df_market.empty or df_market["SHARP_HIT_BOOL"].nunique() < 2:
            status.warning(f"⚠️ Not enough label variety for {market.upper()} — skipping.")
            pb.progress(min(100, max(0, pct)))
            continue
        
        # ========= Drop-in: normalize, dedup, build & merge market-specific priors =========
        def _norm_lower(s): return s.astype(str).str.lower().str.strip()
        def _norm_upper(s): return s.astype(str).str.upper().str.strip()
        def _assert_no_growth(before_rows, after_df, name):
            after_rows = len(after_df)
            if after_rows > before_rows:
                raise RuntimeError(f"{name}: row explosion {before_rows} → {after_rows}")
        
        # --- 0) Normalize core keys once
        for c in ("Game_Key","Team","Home_Team_Norm","Away_Team_Norm","Outcome","Outcome_Norm","Market","Bookmaker"):
            if c in df_market.columns:
                df_market[c] = _norm_lower(df_market[c])
        if "Sport" in df_market.columns:
            df_market["Sport"] = _norm_upper(df_market["Sport"])
        
        # --- 1) Define Team / Is_Home for THIS market (totals anchor to home)
        is_totals = df_market["Market"].eq("totals")
        df_market["Team"] = df_market["Outcome_Norm"].where(~is_totals, df_market["Home_Team_Norm"])
        df_market["Is_Home"] = np.where(
            is_totals,
            1,
            (df_market["Team"] == df_market["Home_Team_Norm"]).astype(int)
        ).astype(int)
        
        # --- 2) Build a slim, latest-snapshot view ONLY for priors ---
        # Keep df_market intact (no dedup here)
        if "Snapshot_Timestamp" in df_market.columns:
            df_market["Snapshot_Timestamp"] = pd.to_datetime(
                df_market["Snapshot_Timestamp"], errors="coerce", utc=True
            )
            df_prior_input = (
                df_market
                .sort_values(["Game_Key","Team","Snapshot_Timestamp"])
                .groupby(["Game_Key","Team"], as_index=False)
                .tail(1)    # latest snapshot PER (Game_Key, Team) just for priors
                .loc[:, ["Game_Key","Team","Is_Home","Snapshot_Timestamp",
                         "SHARP_HIT_BOOL","Sport","Game_Start"]]  # only what priors need
                .drop_duplicates(subset=["Game_Key","Team"])
            )
        else:
            df_prior_input = (
                df_market
                .loc[:, ["Game_Key","Team","Is_Home","SHARP_HIT_BOOL","Sport","Game_Start"]]
                .drop_duplicates(subset=["Game_Key","Team"])
            )
        
        # --- 3) Build market-specific priors on the slim view (leakage-safe) ---
        sport_cur = df_prior_input["Sport"].iloc[0] if not df_prior_input.empty else "NBA"
        try:
            period_cur = _derive_period(df_prior_input.get("Game_Start", pd.Series(dtype="datetime64[ns]")))
        except NameError:
            period_cur = "reg"
        
        priors = build_team_ats_priors_market_sport(
            df_prior_input,
            sport=sport_cur,
            market=market,   # "spreads" | "totals" | "h2h"
            period=period_cur,
            team_col="Team",
            game_col="Game_Key",
            ts_col="Snapshot_Timestamp",
            is_home_col="Is_Home",
            cover_bool_col="SHARP_HIT_BOOL",
            cover_margin_col=("ATS_Cover_Margin" if "ATS_Cover_Margin" in df_market.columns else None),
            add_home_away_splits=True,
            suffix=None
        )
        
        # --- 4) Normalize keys in priors and force uniqueness there ---
        for c in ("Game_Key","Team"):
            priors[c] = priors[c].astype(str).str.lower().str.strip()
        priors = priors.drop_duplicates(subset=["Game_Key","Team"])
        
        # --- 5) Merge priors back WITHOUT collapsing df_market rows ---
        before_rows = len(df_market)
        df_market = df_market.merge(
            priors, on=["Game_Key","Team"], how="left", validate="many_to_one"
        )
        after_rows = len(df_market)
        if after_rows != before_rows:
            raise RuntimeError(f"merge priors changed row count {before_rows} → {after_rows}")
        
        # --- 6) Safe fill for prior fields (only those present) ---
        prior_fill_cols = [
            c for c in ("ATS_EB_Rate","ATS_EB_Margin","ATS_Roll_Margin_Decay",
                        "ATS_EB_Rate_Home","ATS_EB_Rate_Away")
            if c in df_market.columns
        ]
        if prior_fill_cols:
            df_market[prior_fill_cols] = (
                df_market[prior_fill_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .astype("float32")
            )
       
       
        # ===================== SPREADS branch =====================
        if market == "spreads":
            game_keys = ["Sport","Home_Team_Norm","Away_Team_Norm"]
        
            # Normalize merge keys on BOTH sides
            for c in ("Home_Team_Norm","Away_Team_Norm"):
                df_market[c] = _norm_lower(df_market[c])
            df_market["Sport"] = _norm_upper(df_market["Sport"])
        
            # 1) Skinny slice for this snapshot (only needed cols)
            slice_cols = ["Sport","Game_Start","Home_Team_Norm","Away_Team_Norm","Outcome_Norm","Value"]
            df_slice = (
                df_market[slice_cols]
                .dropna(subset=["Sport","Home_Team_Norm","Away_Team_Norm","Outcome_Norm","Value"])
                .copy()
            )
            df_slice["Value"] = pd.to_numeric(df_slice["Value"], errors="coerce").astype("float32")
        
            # 2) Build maps from df_train (training-safe enrichment)
            pr_map = df_train[game_keys + ["Power_Rating_Diff"]].drop_duplicates(subset=game_keys)
        
            cons_cols = game_keys + [
                "Market_Favorite_Team","Market_Underdog_Team",
                "Favorite_Market_Spread","Underdog_Market_Spread"
            ]
            missing_cons_cols = [c for c in cons_cols if c not in df_train.columns]
            if missing_cons_cols:
                g_cons_fallback = prep_consensus_market_spread_lowmem(
                    df_slice, value_col="Value", outcome_col="Outcome_Norm"
                )
                cons_map = g_cons_fallback[cons_cols].drop_duplicates(subset=game_keys)
            else:
                cons_map = df_train[cons_cols].drop_duplicates(subset=game_keys)
        
            # Reconstruct k = |spread|
            k1 = pd.to_numeric(cons_map["Favorite_Market_Spread"], errors="coerce").abs()
            k2 = pd.to_numeric(cons_map["Underdog_Market_Spread"], errors="coerce").abs()
            cons_map["k"] = k1.fillna(k2).astype("float32")
        
            # 3) Join maps to make game-level frame
            g_full = (
                df_slice[game_keys].drop_duplicates(subset=game_keys)
                .merge(pr_map,  on=game_keys, how="left")
                .merge(cons_map, on=game_keys, how="left")
            )
        
            # 4) Model margin & spreads at game level
            g_fc = favorite_centric_from_powerdiff_lowmem(g_full)
        
            # Ensure market fields & k exist
            need_market = ["Market_Favorite_Team","Market_Underdog_Team",
                           "Favorite_Market_Spread","Underdog_Market_Spread"]
            missing_market = [c for c in need_market if c not in g_fc.columns]
            if missing_market:
                g_fc = g_fc.merge(
                    cons_map[game_keys + need_market].drop_duplicates(subset=game_keys),
                    on=game_keys, how="left", copy=False
                )
        
            if "k" not in g_fc.columns:
                fav_abs = pd.to_numeric(g_fc.get("Favorite_Market_Spread"), errors="coerce").abs()
                dog_abs = pd.to_numeric(g_fc.get("Underdog_Market_Spread"), errors="coerce").abs()
                g_fc["k"] = fav_abs.fillna(dog_abs).astype("float32")
        
            # Guarantee model spreads exist even if helper didn’t attach them
            if "Model_Fav_Spread" not in g_fc.columns or "Model_Dog_Spread" not in g_fc.columns:
                ema = pd.to_numeric(g_fc.get("Model_Expected_Margin_Abs"), errors="coerce").astype("float32")
                g_fc["Model_Fav_Spread"] = (-ema).astype("float32")
                g_fc["Model_Dog_Spread"] = ( ema).astype("float32")
        
            # Only the columns we need from g_fc
           
            proj_cols = [
                "Model_Fav_Spread","Model_Dog_Spread",
                "Market_Favorite_Team","Market_Underdog_Team",
                "Favorite_Market_Spread","Underdog_Market_Spread",
                "Model_Expected_Margin",          # signed home-away expected margin
                "Model_Expected_Margin_Abs",
                "Sigma_Pts","k"
            ]
            
            g_map = g_fc[game_keys + proj_cols].copy()
            g_map = (g_map.sort_values(game_keys)
                          .drop_duplicates(subset=game_keys, keep="last"))
            
            before = len(df_market)
            df_market = df_market.merge(g_map, on=game_keys, how="left", validate="many_to_one")
            _assert_no_growth(before, df_market, "merge game-level projections")
            
            # Compute per-outcome spreads from model/market (fav vs dog)
            for c in ["Outcome_Norm","Market_Favorite_Team"]:
                df_market[c] = df_market[c].astype(str).str.lower().str.strip()
            
            is_fav = (df_market["Outcome_Norm"].values == df_market["Market_Favorite_Team"].values)
            
            df_market["Outcome_Model_Spread"] = np.where(
                is_fav, df_market["Model_Fav_Spread"].values, df_market["Model_Dog_Spread"].values
            ).astype("float32")
            
            df_market["Outcome_Market_Spread"] = np.where(
                is_fav, df_market["Favorite_Market_Spread"].values, df_market["Underdog_Market_Spread"].values
            ).astype("float32")
            
            # TEAM-perspective edge / cover prob
            mu_home = pd.to_numeric(df_market.get("Model_Expected_Margin"), errors="coerce").astype("float32")
            
            out_norm = df_market["Outcome_Norm"].astype(str).str.lower().str.strip().values
            home_t   = df_market["Home_Team_Norm"].astype(str).str.lower().str.strip().values
            away_t   = df_market["Away_Team_Norm"].astype(str).str.lower().str.strip().values
            
            is_home_row = (out_norm == "home") | (out_norm == home_t)
            is_away_row = (out_norm == "away") | (out_norm == away_t)
            
            mu_team = np.where(is_home_row, mu_home, np.where(is_away_row, -mu_home, np.nan)).astype("float32")
            
            val_team = pd.to_numeric(df_market.get("Value"), errors="coerce").astype("float32")
            sigma = pd.to_numeric(df_market.get("Sigma_Pts"), errors="coerce").astype("float32")
            sigma = np.where(sigma > 0, sigma, np.nan).astype("float32")
            
            df_market["Outcome_Spread_Edge"] = (mu_team + val_team).astype("float32")
            
            z = (-val_team - mu_team) / sigma
            df_market["Outcome_Cover_Prob"] = (1.0 - _phi(z)).astype("float32")
            
            df_market["z"] = (df_market["Outcome_Spread_Edge"] / sigma).astype("float32")

        
            # Streamlit KPIs (guarded)
            if "Model_Fav_Spread" in df_market.columns:
                have_spreads = float(df_market["Model_Fav_Spread"].notna().mean())
                have_k = float(df_market["k"].notna().mean()) if "k" in df_market.columns else 0.0
                status.write(
                    f"🧪 SPREADS merge health — rows: {len(df_market):,} | "
                    f"have Model_Fav_Spread: **{have_spreads:.1%}** | "
                    f"have k: **{have_k:.1%}**"
                )
            else:
                status.error("❌ SPREADS: `Model_Fav_Spread` missing after merge — debug below.")
                with st.expander("Debug: spreads merge inputs"):
                    st.write("Expected from g_map: Model_Fav_Spread, Model_Dog_Spread, Favorite/Underdog_Market_Spread, k")
                    st.dataframe(
                        df_market[["Sport","Home_Team_Norm","Away_Team_Norm"]]
                        .drop_duplicates().head(30),
                        use_container_width=True
                    )
        # ===================== end SPREADS branch =====================
        def expected_calibration_error(y_true, p_pred, n_bins: int = 10):
            y = np.asarray(y_true, int)
            p = np.asarray(p_pred, float)
            p = np.clip(p, 1e-9, 1 - 1e-9)
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            idx = np.minimum(np.digitize(p, bins) - 1, n_bins - 1)
            ece = 0.0
            for b in range(n_bins):
                m = (idx == b)
                if not np.any(m):
                    continue
                conf = p[m].mean()
                acc  = y[m].mean()
                ece += (m.mean()) * abs(acc - conf)
            return float(ece)
        
        def population_stability_index(p_ref, p_new, bins: int = 20):
            r = np.asarray(p_ref, float); n = np.asarray(p_new, float)
            r = np.clip(r, 1e-9, 1 - 1e-9); n = np.clip(n, 1e-9, 1 - 1e-9)
            qs = np.quantile(r, np.linspace(0, 1, bins + 1))
            qs[0], qs[-1] = 0.0, 1.0
            R, N = [], []
            for i in range(bins):
                lo, hi = qs[i], qs[i+1]
                R.append(((r >= lo) & (r <= hi)).mean())
                N.append(((n >= lo) & (n <= hi)).mean())
            R = np.clip(np.asarray(R), 1e-6, 1); N = np.clip(np.asarray(N), 1e-6, 1)
            return float(np.sum((N - R) * np.log(N / R)))

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
            pb.progress(min(100, max(0, pct)))
            continue
      
        # -----------------------------
        # Spread-safe direction features
        # -----------------------------
        df_market["Line_Delta"] = pd.to_numeric(df_market["Line_Delta"], errors="coerce")
        df_market["Value"]      = pd.to_numeric(df_market["Value"], errors="coerce")
        
        # (A) RAW: which way did the number move?
        # +1 = increased (e.g., -3 -> -2, or +3 -> +4), -1 = decreased (e.g., -3 -> -4)
        df_market["Line_Move_Raw_Dir"] = np.where(
            df_market["Line_Delta"] > 0, 1,
            np.where(df_market["Line_Delta"] < 0, -1, 0)
        ).astype(int)
        
        # (B) TEAM-RELATIVE: did it move toward THIS team (spreads/h2h only)?
        # Create a team-signed delta: normalizes favorite/dog so "toward team" is consistent.
        sign_val = np.sign(df_market["Value"])
        sign_val = np.where(sign_val == 0, np.nan, sign_val)   # avoid sign(0)
        
        df_market["Line_Delta_Team"] = df_market["Line_Delta"] * sign_val
        
        # Convention (choose ONE and keep it everywhere):
        # Here: Line_Delta_Team < 0  ==> moved TOWARD the team
        df_market["Line_Moved_Toward_Team"] = np.where(
            df_market["Line_Delta_Team"] < 0, 1,
            np.where(df_market["Line_Delta_Team"] > 0, 0, -1)
        ).astype(int)
        
        # (C) This is what you should use for alignment in spreads/h2h
        is_spread_like = df_market["Market"].isin(["spreads", "h2h"])
        
        base = np.where(is_spread_like, df_market["Line_Delta_Team"], df_market["Line_Delta"])
        
        df_market["Direction_Aligned"] = np.where(
            base < 0, 1,               # toward team (per convention above)
            np.where(base > 0, 0, -1)   # away / unknown
        ).astype(int)
        
        
        df_market['Line_Value_Abs'] = df_market['Value'].abs()
        
        
        
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
        df_market["Is_Home_Team_Bet"] = (df_market["Outcome_Norm"] == df_market["Home_Team_Norm"]).astype(int)
        df_market['Is_Favorite_Bet'] = (df_market['Value'] < 0).astype(int)
        
        # Ensure NA-safe boo logic and conversion
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
       

        df_market['Market_Implied_Prob'] = _amer_to_prob_vec(df_market['Odds_Price'])

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
        
        # With this (uses the same vectorizer you already defined):
        df_market['Implied_Prob_Shift'] = (
            _amer_to_prob_vec(df_market['Odds_Price']) - _amer_to_prob_vec(df_market['First_Odds'])
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
        if df_market["Market"].iloc[0] == "spreads":
            st.write(df_market[["Outcome","Value","Line_Delta","Line_Move_Raw_Dir","Line_Delta_Team","Line_Moved_Toward_Team","Direction_Aligned"]].head(12))
        
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
        
        # --- Power rating columns (already in your code) ---
        df_market['PR_Team_Rating'] = np.where(is_home_bet, df_market['Home_Power_Rating'], df_market['Away_Power_Rating'])
        df_market['PR_Opp_Rating']  = np.where(is_home_bet, df_market['Away_Power_Rating'], df_market['Home_Power_Rating'])
        df_market['PR_Rating_Diff']     = df_market['PR_Team_Rating'] - df_market['PR_Opp_Rating']
        df_market['PR_Abs_Rating_Diff'] = df_market['PR_Rating_Diff'].abs()
       
        # ---- H2H-only power rating alignment flags ---------------------------------
        mkt = df_market['Market'].astype(str).str.strip().str.lower()
        is_h2h = mkt.isin(['h2h', 'moneyline', 'ml', 'headtohead'])
        
        # Choose the model prob column
        prob_col = 'Model Prob' if 'Model Prob' in df_market.columns else 'Model_Sharp_Win_Prob'
        if prob_col not in df_market.columns:
            df_market[prob_col] = np.nan
        p_model = pd.to_numeric(df_market[prob_col], errors='coerce')
        
        # --- (A) PR ↔ MODEL agreement (H2H only) ---
        df_market['PR_Model_Agree_H2H'] = np.where(
            is_h2h & p_model.notna(),
            ((df_market['PR_Rating_Diff'] > 0) & (p_model > 0.5)) |
            ((df_market['PR_Rating_Diff'] < 0) & (p_model < 0.5)),
            np.nan
        )
        
        # --- (B) PR ↔ MARKET agreement (H2H only) ---
        if 'Implied_Prob' in df_market.columns:
            p_mkt = pd.to_numeric(df_market['Implied_Prob'], errors='coerce')
        else:
            odds_ = pd.to_numeric(df_market.get('Odds_Price', np.nan), errors='coerce')
            p_mkt = pd.Series(np.where(
                odds_.notna(),
                np.where(odds_ < 0, (-odds_) / ((-odds_) + 100.0), 100.0 / (odds_ + 100.0)),
                np.nan
            ), index=df_market.index)
        
        df_market['PR_Market_Agree_H2H'] = np.where(
            is_h2h & p_mkt.notna(),
            ((df_market['PR_Rating_Diff'] > 0) & (p_mkt > 0.5)) |
            ((df_market['PR_Rating_Diff'] < 0) & (p_mkt < 0.5)),
            np.nan
        )
        
        # Optional UI strings
        df_market['PR_Model_Alignment_H2H'] = np.select(
            [df_market['PR_Model_Agree_H2H'] == True, df_market['PR_Model_Agree_H2H'] == False],
            ["✅ PR ↔ Model Agree", "❌ PR ≠ Model"],
            default="—"
        )
        df_market['PR_Market_Alignment_H2H'] = np.select(
            [df_market['PR_Market_Agree_H2H'] == True, df_market['PR_Market_Agree_H2H'] == False],
            ["✅ PR ↔ Market Agree", "❌ PR ≠ Market"],
            default="—"
        )
        
        # Ensure non-H2H rows show blanks
        df_market.loc[~is_h2h, ['PR_Model_Agree_H2H', 'PR_Market_Agree_H2H']] = np.nan
        df_market.loc[~is_h2h, ['PR_Model_Alignment_H2H', 'PR_Market_Alignment_H2H']] = "—"
        
        # (Optional) numeric flags for modeling
        df_market['PR_Model_Agree_H2H_Flag']  = (df_market['PR_Model_Agree_H2H']  == True).astype('Int8')
        df_market['PR_Market_Agree_H2H_Flag'] = (df_market['PR_Market_Agree_H2H'] == True).astype('Int8')



        
        def extend_unique(base, items):
            for c in items:
                if c not in base:
                    base.append(c)
        # after you finish cleaning/deduping df_market (your canonical training rows)
        cross_pivots = build_cross_market_pivots_for_training(df_market)
        
        df_market = attach_pairwise_correlation_features(
            df_market,
            cross_pivots,
            ST_lookup, SM_lookup, TM_lookup,
            sport_default=sport_label,   # not 'label'
        )

     
        
        if "st" in globals():
            st.write(
                "📋 df_market columns BEFORE feature selection:",
                sorted(df_market.columns.tolist())
            )
        if "st" in globals():
            with st.expander("🔍 df_market sample values", expanded=False):
                st.dataframe(
                    df_market
                    .head(1000)          # or .sample(50, random_state=42)
                    .reset_index(drop=True),
                    use_container_width=True
                )   
        # --- start with your manual core list ---
        features = [
            # 🔹 Core sharp signals
            'Sharp_Move_Signal',
            'Sharp_Limit_Jump','Sharp_Time_Score',#'Book_lift_x_Sharp',
            #'Book_lift_x_Magnitude',
            #'Book_lift_x_PROB_SHIFT',
            'Sharp_Limit_Total',
            'Is_Reinforced_MultiMarket','Market_Leader','LimitUp_NoMove_Flag','Value',
            
        
            # 🔹 Market response
            'Sharp_Line_Magnitude',
            'Is_Home_Team_Bet',
            'Line_Moved_Toward_Team',
            'Team_Implied_Prob_Gap_Home','Team_Implied_Prob_Gap_Away',
        
            # 🔹 Engineered odds shift decomposition
            'SharpMove_Odds_Up','SharpMove_Odds_Down','SharpMove_Odds_Mag',
        
            # 🔹 Engineered interactions
            'MarketLeader_ImpProbShift','LimitProtect_SharpMag','Delta_Sharp_vs_Rec',#'Sharp_Leads',
        
            # 🔁 Reversal logic
            'Value_Reversal_Flag','Odds_Reversal_Flag',
        
            # 🔥 Timing flags
            #'Late_Game_Steam_Flag',
        
            'Abs_Line_Move_From_Opening',
            'Abs_Odds_Move_From_Opening',
            'Market_Mispricing',#'Spread_vs_H2H_Aligned','Total_vs_Spread_Contradiction',
            'Spread_vs_H2H_ProbGap','Total_vs_H2H_ProbGap','Total_vs_Spread_ProbGap',
            'CrossMarket_Prob_Gap_Exists',
            'Line_Moved_Away_From_Team',
            
            
            'Pct_Line_Move_From_Opening',#'Pct_Line_Move_Bin',
            'Potential_Overmove_Flag',
            #'Potential_Overmove_Total_Pct_Flag',#'Mispricing_Flag',
            'Was_Line_Resistance_Broken',
            'Line_Resistance_Crossed_Count','SharpMove_Resistance_Break',
        
            # 🧠 Cross-market alignment
            'Potential_Odds_Overmove_Flag',
            'Abs_Line_Move_Z','Pct_Line_Move_Z',
            'SmallBook_Limit_Skew',
            'SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag',
            #'Book_Reliability_Score',
            #'Book_Reliability_Lift',#'Book_Reliability_x_Sharp',
            #'Book_Reliability_x_Magnitude',
            #'Book_Reliability_x_PROB_SHIFT',
            "Book_Path_Open_Score",
            "Book_Path_Speed_Score",
            "Book_Path_Open_Lift",
            "Book_Path_Speed_Lift",

            # Power ratings / edges
            'PR_Team_Rating','PR_Opp_Rating',
            'PR_Rating_Diff','PR_Abs_Rating_Diff',
            
            'Outcome_Model_Spread','Outcome_Market_Spread',
            'Outcome_Spread_Edge',
            'Outcome_Cover_Prob',
            'model_fav_vs_market_fav_agree',
            'TOT_Proj_Total_Baseline',
            'TOT_Off_H','TOT_Def_H','TOT_Off_A','TOT_Def_A',
            'TOT_GT_H','TOT_GT_A',#'TOT_LgAvg_Total',
            'TOT_Mispricing', 
            'ATS_EB_Rate',
            'ATS_EB_Margin',            # Optional: only if cover_margin_col was set
            'ATS_Roll_Margin_Decay',    # Optional: only if cover_margin_col was set
            'ATS_EB_Rate_Home',
            'ATS_EB_Rate_Away',
            'PR_Model_Agree_H2H_Flag',#'PR_Market_Agree_H2H_Flag',
            "SpreadTotal_Rho","SpreadTotal_Synergy","SpreadTotal_Sign",
            "SpreadML_Rho","SpreadML_Synergy","SpreadML_Sign","Spread_ML_ProbGap",
            "TotalML_Rho","TotalML_Synergy","TotalML_Sign",
            "EV_Sh_vs_Rec_Prob", "EV_Sh_vs_Rec_Dollar", "Kelly_Fraction",
            "Spread_Abs_Game",
            "Spread_Abs_Game_Z",
            
            "Total_Game",
            "Total_Game_Z",
            
            "Spread_x_Total",
            "Spread_over_Total",
            "Total_over_Spread",
            "Dist_to_10",
                    # sums
            "Team_Recent_Cover_Streak","Team_Recent_Cover_Streak_Home","Team_Recent_Cover_Streak_Away",
           
            # denoms
            "Team_Recent_Cover_Games","Team_Recent_Cover_Games_Home","Team_Recent_Cover_Games_Away",
            
            # rates
            "Team_Recent_Cover_Rate","Team_Recent_Cover_Rate_Home","Team_Recent_Cover_Rate_Away",
           
            # flags
            "On_Cover_Streak","On_Cover_Streak_Home","On_Cover_Streak_Away",
            "After_Win_Flag","Revenge_Flag",
            "Current_Win_Streak_Prior", "Current_Loss_Streak_Prior",
            "H2H_Win_Pct_Prior",  "Opp_WinPct_Prior",
            "Last_Matchup_Result","Last_Matchup_Margin","Days_Since_Last_Matchup",
            "Wins_Last5_Prior",
            "Margin_Last5_Prior",
            "Days_Since_Last_Game",
            "Close_Game_Rate_Prior","Blowout_Game_Rate_Prior",
            "Avg_Home_Margin_Prior","Avg_Away_Margin_Prior",
            "Days_Since_Last_Game",
            "Games_Last_2_Days",
            "Games_Last_4_Days",
            "Games_Last_7_Days",
            "Is_B2B",
            "Is_3in4",
            "CLV_Prob_Delta",
            "CLV_Line_Pts_Signed",
            "Resistance_Score",
            "Early_Sharp_Flag",
            "Late_Sharp_Flag",
            "Spread_Size_Bucket",
            "Total_Size_Bucket",
            "H2H_Prob_Bucket",
        
            # Consensus / dispersion
            "ImpProb_Mean_AllBooks",
            "ImpProb_Std_AllBooks",
            "ImpProb_Mean_SharpBooks",
            "ImpProb_Mean_RecBooks",
            "Sharp_Rec_Prob_Gap",
            "Pct_Books_Aligned",
        
            "Line_Mean_SharpBooks",
            "Line_Mean_RecBooks",
            "Sharp_Rec_Line_Gap",
            "Line_Std_AllBooks",
        
            # Key numbers (spreads)
            "Crossed_Key_3",
            "Crossed_Key_7",
            "Dist_to_3",
            "Dist_to_7",
                   
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
        timing_cols = build_timing_aggregates_inplace(df_bt)

        # extend your feature list with timing_cols (and remove the 32 originals)
        extend_unique(features, timing_cols)

        # add historical/streak features you computed earlier (schema-safe list)
        
        extend_unique(features, [
            'ResistBreak_x_Mag','LateSteam_x_KeyCount','Aligned_x_HighLimit',
            'Rev_x_BookLift','GapExists_x_RecSkew','Misalign_x_MoveAway'
        ])

        # add recent team model performance stats
       
        
        # add time-context flags
        extend_unique(features, ['Is_Night_Game','Is_PrimeTime','DOW_Sin'])
        
        extend_unique(features, [
            "Implied_Hold_Book","Two_Sided_Offered","Juice_Abs_Delta",
            "Dist_To_Next_Key","Key_Corridor_Pressure",
            "Book_PctRank_Line",
            "Book_Line_Diff_vs_SharpMedian","Outlier_Flag_SharpBooks",
        ])
        
        # Add hybrid timing derivatives (snapshot-safe, computed above)
        extend_unique(features, [
            "Hybrid_Line_EarlyMag","Hybrid_Line_MidMag","Hybrid_Line_LateMag",
            "Hybrid_Odds_TotalMag","Hybrid_Odds_EarlyMag","Hybrid_Odds_MidMag","Hybrid_Odds_LateMag",
            "Hybrid_Line_LateShare","Hybrid_Line_EarlyShare",
            "Hybrid_Line_Imbalance_LateVsEarly",
            "Hybrid_Odds_LateShare","Hybrid_Odds_EarlyShare",
            "Hybrid_Line_Odds_Mag_Ratio",
            "Hybrid_Timing_Entropy_Line","Hybrid_Timing_Entropy_Odds",
            "Abs_Line_Move_From_Opening","Abs_Odds_Move_From_Opening",
            # interactions (only exist if microstructure ran)
            "Corridor_x_LateShare_Line","Dist_x_LateShare_Line","PctRank_x_LateShare_Line",
        ])

        extend_unique(features, [
            # Limit dynamics
            "Limit_Spike_Flag","Delta_Limit","LimitSpike_x_NoMove",
            # Book network
            "Sharp_Consensus_Weight","Sharp_vs_Rec_SpreadGap_Q90_Q10",#"Sharp_vs_Rec_SpreadGap_Q50",
            # Internal consistency
            #"Spread_ML_ProbGap",
            "Spread_ML_Inconsistency","Total_vs_Side_ImpliedDelta",
            # Alt lines (optional)
            #"AltLine_Slope",#"AltLine_Curv",
            # CLV proxies
            #"CLV_Proxy_E_DeltaNext15m",
            #"CLV_Proxy_E_CLV",
        ])

        # merge view-driven features (all_present) without losing order
        if 'all_present' in locals():
            extend_unique(features, all_present)
        
        # ======= IMPORTANT: work with df_market from here on =======
        
        # Make sure df_market has placeholders for requested feature columns (0 default)
        # Make sure df_market has placeholders for requested feature columns (0 default)
        df_market = ensure_columns(df_market, features, 0)
        
        # Now prune to columns that actually exist (or were just ensured)
        missing_in_market = [c for c in features if c not in df_market.columns]
        if missing_in_market:
            st.write(f"ℹ️ Dropping {len(missing_in_market)} missing feature(s): "
                     f"{sorted(missing_in_market)[:20]}{'...' if len(missing_in_market)>20 else ''}")
        
        # 🔧 C FEATURES SAFELY (inplace mutation)
        features = c_features_inplace(df_market, features)
        
        # Final dataset for modeling
        feature_cols = [str(c) for c in features]
        if str(market).lower() == "spreads":
            drop_like = ('Pct_Line_Move_From_Opening','Pct_Line_Move_Bin','Pct_Line_Move_Z')
            feature_cols = [c for c in feature_cols if not any(x in c for x in drop_like)]

        st.markdown(f"### 📈 Features Used: `{len(features)}`")
           
        
        # ─────────────────────────────────────────────────────────────────────────────
        # Safe feature matrix + high‑corr report (Arrow/Streamlit‑proof)
        # ─────────────────────────────────────────────────────────────────────────────
        # Expect: df_market, feature_cols (list-like), market, st, np, pd in scope
        _fc = list(dict.fromkeys([str(c) for c in (feature_cols or [])]))
        # 1) Build numeric X (all float32), no Inf/NaN, no exotic dtypes
        if not _fc:
            st.info("No features provided.")
            return
        
        X_raw = df_market.reindex(columns=_fc, fill_value=np.nan)
        
        # Coerce everything → numeric
        X = (
            X_raw.apply(pd.to_numeric, errors='coerce')
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0.0)
                 .astype('float32', copy=False)
        )
        
        # 1a) Ensure DataFrame & Arrow‑friendly column names (strings, unique)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=_fc)
        X.columns = [str(c) for c in X.columns]
        
        # 2) Secondary c for correlation input
        Xc = (
            X.apply(pd.to_numeric, errors="coerce")
             .replace([np.inf, -np.inf], np.nan)
        )
        
        # 2a) Drop all‑NA and constant columns (corr degeneracy)
        na_only_cols = [c for c in Xc.columns if Xc[c].isna().all()]
        const_cols   = [c for c in Xc.columns if Xc[c].nunique(dropna=True) <= 1]
        drop_cols    = set(na_only_cols) | set(const_cols)
        keep_cols    = [c for c in Xc.columns if c not in drop_cols]
        Xc = Xc[keep_cols].copy()
        
        # 2b) Cap width for safety (very wide matrices can choke Arrow/corr)
        MAX_CORR_COLS = 400  # tune if needed
        if Xc.shape[1] > MAX_CORR_COLS:
            st.warning(f"Feature count {Xc.shape[1]} too wide for corr; sampling {MAX_CORR_COLS}.")
            # prefer most variable columns (more informative)
            variances = Xc.var(numeric_only=True).sort_values(ascending=False)
            sel = [c for c in variances.index.tolist() if c in Xc.columns][:MAX_CORR_COLS]
            Xc = Xc[sel].copy()
        
        # 2c) If nothing viable, bail gracefully
        if Xc.shape[1] == 0:
            st.info("No valid numeric, non‑constant features available for correlation.")
        else:
            # 3) Compute abs corr with fallback + size cap on rows to avoid memory spikes
            # Downsample rows if extremely tall
            MAX_CORR_ROWS = 20000
            if len(Xc) > MAX_CORR_ROWS:
                Xc = Xc.sample(n=MAX_CORR_ROWS, random_state=13)
        
            try:
                corr_matrix = Xc.corr(method="pearson", min_periods=2).abs()
            except Exception as e:
                st.warning(f"Primary (pearson) corr failed: {e}. Retrying with Spearman…")
                try:
                    corr_matrix = Xc.corr(method="spearman", min_periods=2).abs()
                except Exception as e2:
                    st.error(f"Spearman corr also failed: {e2}. Skipping corr view.")
                    corr_matrix = None
        
            # 4) Extract high‑corr pairs safely
            if corr_matrix is not None and not corr_matrix.empty:
                threshold = 0.85
                cols = corr_matrix.columns.tolist()
                pairs = []
                # Avoid self-pairs; iterate upper triangle only
                for i, ci in enumerate(cols):
                    row = corr_matrix.iloc[i, i+1:]
                    if row is not None and hasattr(row, "dropna"):
                        hits = row.dropna()
                        hits = hits[hits > threshold]
                        if not hits.empty:
                            for cj, val in hits.items():
                                # enforce plain Python scalars (Arrow‑safe)
                                pairs.append((str(ci), str(cj), float(val)))
        
                if not pairs:
                    st.success("✅ No highly correlated feature pairs found")
                else:
                    df_corr = pd.DataFrame(pairs, columns=["Feature_1", "Feature_2", "Correlation"])
                    df_corr = df_corr.sort_values("Correlation", ascending=False)
        
                    # 5) UI safety: cap rows & round; enforce pure Python types
                    MAX_ROWS_SHOW = 500
                    show = df_corr.head(MAX_ROWS_SHOW).copy()
                    show["Feature_1"] = show["Feature_1"].astype(str)
                    show["Feature_2"] = show["Feature_2"].astype(str)
                    show["Correlation"] = pd.to_numeric(show["Correlation"], errors="coerce").round(4)
        
                    # Final sanitization for Arrow / React (#185)
                    # - Replace NaN in object columns with ""
                    # - Ensure pure Python scalars
                    show = show.replace([np.inf, -np.inf], np.nan).dropna(how="all").drop_duplicates().reset_index(drop=True)
                    # Ensure no NaNs in string/object cols
                    for c in ["Feature_1", "Feature_2"]:
                        if c in show.columns:
                            show[c] = show[c].astype(object).where(pd.notna(show[c]), "")
                            # Force Python str
                            show[c] = show[c].map(lambda v: str(v) if v is not None else "")
                    # Force float64 for the numeric col
                    if "Correlation" in show.columns:
                        show["Correlation"] = show["Correlation"].astype("float64")
                    # Arrow prefers no NaN in object cells; use None
                    show = show.where(pd.notna(show), None)
        
                    # 6) Title + Render correlated feature pairs
                    st.markdown("#### 🔗 Highly Correlated Feature Pairs (|r| > 0.85)")
                    try:
                        st.dataframe(show, hide_index=True, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Dataframe render failed, using fallback table. ({e})")
                        st.table(show)
                        # Optional: CSV download fallback
                        try:
                            csv_bytes = show.to_csv(index=False).encode("utf-8")
                            st.download_button("⬇️ Download correlated pairs (CSV)", data=csv_bytes, file_name="high_corr_pairs.csv", mime="text/csv")
                        except Exception:
                            pass
            else:
                st.info("Correlation matrix is empty or unavailable.")
        
        # ── Target checks (Arrow‑ and training‑safe) ──────────────────────────────────
        if 'SHARP_HIT_BOOL' not in df_market.columns:
            st.warning("⚠️ Missing SHARP_HIT_BOOL in df_market — skipping.")
            return  # or `continue` if inside a loop
        
        y = pd.to_numeric(df_market['SHARP_HIT_BOOL'], errors='coerce')
        # Treat any non‑{0,1} as 0; fill NaN to 0 explicitly
        y = y.where(y.isin([0, 1]), 0).fillna(0).astype('int8')
        
        if y.nunique(dropna=True) < 2:
            title_market = str(market).upper() if 'market' in locals() else 'MARKET'
            st.warning(f"⚠️ Skipping {title_market} — only one label class.")
            return  # or `continue`
                
  
      
        # ===============================
        # Purged Group Time-Series CV (PGTSCV) + Embargo
        # ===============================
        # Normalize the sport string for lookup
        # Use the passed-in `sport` exactly as-is (case-sensitive mapping + default)
       
       
        
        # --- sport → embargo (top of file, once) ---
        SPORT_EMBARGO = {
            "MLB":   pd.Timedelta("2 hours"),
            "NBA":   pd.Timedelta("12 hours"),
            "NHL":   pd.Timedelta("2 hours"),
            "NCAAB": pd.Timedelta("11 hours"),
            "NFL":   pd.Timedelta("1 days"),
            "NCAAF": pd.Timedelta("1 days"),
            "WNBA":  pd.Timedelta("8 hours"),
            "MLS":   pd.Timedelta("12 hours"),
            "default": pd.Timedelta("12 hours"),
        }
        def get_embargo_for_sport(sport: str) -> pd.Timedelta:
            return SPORT_EMBARGO.get(str(sport).upper(), SPORT_EMBARGO["default"])
        
        
        class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
            """
            Time-ordered, group-based CV with purge + time embargo.
        
            - Groups (e.g., Game_Key) never straddle train/val.
            - Any group overlapping the validation time window is *purged* from train.
            - Any group overlapping the extended embargo window around validation is embargoed from train.
            - Folds are contiguous in time at the group level.
            """
        
            def __init__(self, n_splits=5, embargo=pd.Timedelta("0 hours"), time_values=None, min_val_size=20):
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
                if len(groups) != len(self.time_values):
                    raise ValueError("groups and time_values must be aligned to X rows")
        
                meta = pd.DataFrame({
                    "group": np.asarray(groups),
                    "time":  pd.to_datetime(self.time_values, errors="coerce", utc=True)
                })
                if meta["time"].isna().any():
                    raise ValueError("time_values contain NaT after to_datetime; check your inputs")
        
                # one row per group, ordered by start time
                gmeta = (meta.groupby("group", as_index=False)["time"]
                            .agg(start="min", end="max")
                            .sort_values("start")
                            .reset_index(drop=True))
        
                n_groups = len(gmeta)
                if n_groups < self.n_splits:
                    self.n_splits = max(2, n_groups)
        
                # contiguous group folds
                fold_sizes = np.full(self.n_splits, n_groups // self.n_splits, dtype=int)
                fold_sizes[: n_groups % self.n_splits] += 1
                edges = np.cumsum(fold_sizes)
        
                start = 0
                for _k, stop in enumerate(edges):
                    val_slice = gmeta.iloc[start:stop]
                    start = stop
                    if val_slice.empty:
                        continue
        
                    val_groups = val_slice["group"].to_numpy()
                    val_start  = val_slice["start"].iloc[0]
                    val_end    = val_slice["end"].iloc[-1]
        
                    # 1) PURGE: overlap with [val_start, val_end]
                    purge_mask = ~((gmeta["end"] < val_start) | (gmeta["start"] > val_end))
                    purged_groups = set(gmeta.loc[purge_mask, "group"])
        
                    # 2) EMBARGO: overlap with [val_start - embargo, val_end + embargo]
                    emb_lo = val_start - self.embargo
                    emb_hi = val_end   + self.embargo
                    embargo_mask = ~((gmeta["end"] < emb_lo) | (gmeta["start"] > emb_hi))
                    embargo_groups = set(gmeta.loc[embargo_mask, "group"])
        
                    bad_groups   = set(val_groups) | purged_groups | embargo_groups
                    train_groups = gmeta.loc[~gmeta["group"].isin(bad_groups), "group"].to_numpy()
        
                    # map back to row indices
                    all_groups = meta["group"].to_numpy()
                    val_idx    = np.flatnonzero(np.isin(all_groups, val_groups))
                    train_idx  = np.flatnonzero(np.isin(all_groups, train_groups))
        
                    # hardening
                    if len(val_idx) == 0 or len(train_idx) == 0:
                        continue
                    if len(val_idx) < self.min_val_size:
                        continue
                    if y is not None:
                        y_arr = np.asarray(y)
                        if np.unique(y_arr[val_idx]).size < 2:
                            continue
        
                    yield train_idx, val_idx
        
        
     
        
        eps = 1e-4  # default probability clip
        
        
        def pos_col_index(est, positive=1):
            cls = getattr(est, "classes_", None)
            if cls is None:
                raise RuntimeError("Estimator has no classes_. Was it fitted?")
            hits = np.where(cls == positive)[0]
            if len(hits) == 0:
                raise RuntimeError(f"Positive class {positive!r} not found in classes_={cls}. Check label encoding.")
            return int(hits[0])
        
        
        def _to_numeric_block(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
            out = df[cols].copy()
            for c in cols:
                col = out[c]
        
                # 1) Boo-like → float (True/False/NA → 1.0/0.0/NaN)
                if is_bool_dtype(col):
                    out[c] = col.astype("float32")
                    continue
        
                # 2) Strings/objects/categories → c then to_numeric
                if is_object_dtype(col) or is_string_dtype(col) or str(col.dtype).startswith("category"):
                    out[c] = col.replace({
                        'True': 1, 'False': 0, 'true': 1, 'false': 0,
                        True: 1, False: 0,
                        '': np.nan, 'none': np.nan, 'None': np.nan,
                        'NA': np.nan, 'NaN': np.nan
                    })
                    out[c] = pd.to_numeric(out[c], errors="coerce")
                    continue
        
                # 3) Everything else → to_numeric safely
                out[c] = pd.to_numeric(col, errors="coerce")
        
            return (out.replace([np.inf, -np.inf], np.nan)
                       .astype("float32")
                       .fillna(0.0))
      
        # ==== Remove PR_* features for SPREADS only ==================================
        # Determine market label (use your local variable if you already have one)
        try:
            market_label = str(market).lower()  # if you have `market` in scope
        except NameError:
            # Fallback: infer from df_market (safe if single market slice)
            market_label = str(df_market["Market"].iloc[0]).lower() if "Market" in df_market.columns and len(df_market) else ""
        
        # Normalize your working feature list name
        feature_cols = list(features) if "features" in locals() else list(feature_cols)
        
  
        features = feature_cols

        # ============================================================================
        def auc_safe(y, p):
            y = np.asarray(y, int)
            if np.unique(y).size < 2:
                return np.nan
            return roc_auc_score(y, p)


        # === Build y and mask FIRST ===
        y_series   = pd.to_numeric(df_market["SHARP_HIT_BOOL"], errors="coerce")
        valid_mask = y_series.notna()
        
        # Work from a masked, reindexed dataframe so .iloc matches X/y
        df_valid = df_market.loc[valid_mask].reset_index(drop=True)
        
        # Final y
        y_full = y_series.loc[valid_mask].astype(int).to_numpy()
        
        # Build X from the masked frame (no extra masking later)
        X_full = _to_numeric_block(df_valid, feature_cols).to_numpy(np.float32)
        # Always work off the masked frame with positional indexing
   
       
        # === Build X / y ===
        
        
         # ---- Cheap feature pruning (before any split/CV) --
      
        check = [
            "Was_Line_Resistance_Broken",
            "Odds_Reversal_Flag",
            "CrossMarket_Prob_Gap_Exists",
            "Sharp_Limit_Jump",
            "Pct_Line_Move_From_Opening",
        ]
        
        for c in check:
            if c in feature_cols:
                j = feature_cols.index(c)
        
                if isinstance(X_full, pd.DataFrame):
                    raw = X_full.iloc[:, j]
                else:
                    raw = X_full[:, j]
        
                col = pd.to_numeric(raw, errors="coerce").to_numpy() if hasattr(raw, "to_numpy") else pd.to_numeric(raw, errors="coerce")
                finite = np.isfinite(col)
        
                st.write(c, {
                    "n_finite": int(finite.sum()),
                    "unique_finite": int(np.unique(col[finite]).size) if finite.any() else 0,
                    "mean_finite": float(np.nanmean(col)) if finite.any() else None,
                })


        st.markdown("### 🧹 Feature Pruning (pre-split)")
        
        Xtmp = X_full
        cols = list(feature_cols)
        
        # ✅ protect these (add more as needed)
        PROTECT = {
            "Sharp_Limit_Jump",
            "Sharp_Time_Score",
            "Sharp_Limit_Total",
            "MarketLeader_ImpProbShift",
            "LimitProtect_SharpMag",
            "Odds_Reversal_Flag",
            "Spread_vs_H2H_ProbGap",
            "Total_vs_H2H_ProbGap",
            "Total_vs_Spread_ProbGap",
            "CrossMarket_Prob_Gap_Exists",
            "Pct_Line_Move_From_Opening",
            "Potential_Overmove_Flag",
            "Was_Line_Resistance_Broken",
            "Line_Resistance_Crossed_Count",
            "SharpMove_Resistance_Break",
            "Pct_Line_Move_Z",
            "Outcome_Model_Spread",
            "Outcome_Market_Spread",
            "Outcome_Spread_Edge",
            "Outcome_Cover_Prob",
        }
        
        def _get_col(X, j):
            return X.iloc[:, j] if isinstance(X, pd.DataFrame) else X[:, j]
        
        MIN_NON_NAN = max(25, int(0.002 * len(Xtmp)))  # looser: 0.2% rows or 25
        MIN_UNIQUE  = 2
        
        keep_idx = []
        removed = []
        removed_stats = []
        
   
        for j, c in enumerate(cols):
            if c in PROTECT:
                keep_idx.append(j)
                continue
        
            col = _get_col(Xtmp, j)
            col = pd.to_numeric(col, errors="coerce")

            finite = np.isfinite(col)
            n_ok = int(finite.sum())
            uniq = int(np.unique(col[finite]).size) if n_ok else 0
        
            # keep rule
            if (n_ok < MIN_NON_NAN) or (uniq < MIN_UNIQUE):
                removed.append(c)
                removed_stats.append((c, n_ok, uniq))
            else:
                keep_idx.append(j)
        
        feature_cols = [cols[j] for j in keep_idx]
        
        if isinstance(X_full, pd.DataFrame):
            X_full = X_full[feature_cols]
        else:
            X_full = X_full[:, keep_idx]
        
        st.write(f"• Removed low-information features: {len(removed)}")
        if removed:
            st.caption(", ".join(removed[:20]) + (" ..." if len(removed) > 20 else ""))
        
        # ✅ show WHY they were removed
        if removed_stats:
            df_removed = pd.DataFrame(removed_stats, columns=["feature", "n_finite", "n_unique"])
            df_removed = df_removed.sort_values(["n_finite", "n_unique"], ascending=True)
            st.dataframe(df_removed.head(80))


        
        # 2) Exact duplicate columns (optional but cheap)
        df_tmp = pd.DataFrame(X_full, columns=feature_cols)
        # Transpose, get unique rows (i.e., unique columns pre-transpose)
        _, uniq_idx = np.unique(df_tmp.T, axis=0, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        
        dup_count = df_tmp.shape[1] - len(uniq_idx)
        if dup_count > 0:
            removed_dups = [feature_cols[i] for i in range(len(feature_cols)) if i not in uniq_idx]
            st.write(f"• Removed duplicate features: {dup_count}")
            if removed_dups:
                st.caption(", ".join(removed_dups[:20]) + (" ..." if len(removed_dups) > 20 else ""))
            df_tmp = df_tmp.iloc[:, uniq_idx]
            feature_cols = list(df_tmp.columns)
        
        # Finalize X_full after pruning
        X_full = df_tmp.to_numpy(dtype=np.float32)
        features_pruned = tuple(feature_cols)   # 🔒 freeze pruned column names
        st.write(f"✅ Final feature count after pruning: {len(feature_cols)}")
        
        # Recompute these from df_valid so everything is aligned
        groups_all = df_valid["Game_Key"].astype(str).to_numpy()
        snap_ts    = pd.to_datetime(df_valid["Snapshot_Timestamp"], errors="coerce", utc=True)
        game_ts    = pd.to_datetime(df_valid["Game_Start"],           errors="coerce", utc=True)
        # ---- Groups & times (snapshot-aware) ----
        
        # If any game has >1 snapshot we’re in snapshot regime (embargo matters)
        by_game_snaps = (
            df_market.loc[valid_mask]
            .groupby("Game_Key")["Snapshot_Timestamp"]
            .nunique(dropna=True)
        )
        has_snapshots = (by_game_snaps.fillna(0).max() > 1)
        
        groups_all = df_valid["Game_Key"].astype(str).to_numpy()
        snap_ts    = pd.to_datetime(df_valid["Snapshot_Timestamp"], errors="coerce", utc=True)
        game_ts    = pd.to_datetime(df_valid["Game_Start"],           errors="coerce", utc=True)
        
        # Use snapshot time if multiple snapshots exist, else game start time
        time_values_all = snap_ts.to_numpy() if has_snapshots else game_ts.to_numpy()
        sport_key  = str(sport).upper()
        embargo_td = SPORT_EMBARGO.get(sport_key, SPORT_EMBARGO["default"]) if has_snapshots else pd.Timedelta(0)

        # Final arrays used downstream
        groups = groups_all
        times  = time_values_all
        
        # ✅ Guard: drop NaT times (rare but fatal for CV) — PLACE THIS BLOCK RIGHT HERE
        if pd.isna(times).any():
            bad  = np.flatnonzero(pd.isna(times))
            keep = np.setdiff1d(np.arange(len(times)), bad)
            X_full  = X_full[keep]
            y_full  = y_full[keep]
            groups  = groups[keep]
            times   = times[keep]
            df_valid = df_valid.iloc[keep].reset_index(drop=True)
            
        # ---- Holdout = last ~N groups (time-forward, group-safe) ----
        meta  = pd.DataFrame({"group": groups, "time": pd.to_datetime(times, utc=True)})
        gmeta = (meta.groupby("group", as_index=False)["time"]
                   .agg(start="min", end="max")
                   .sort_values("start")
                   .reset_index(drop=True))
        # ---- Build time vector once (aligned to df_valid) ----
        t_full = pd.to_datetime(df_valid["Snapshot_Timestamp"], utc=True, errors="coerce")
        
        # Fallback to Commence_Hour where Snapshot_Timestamp is NaT
        if t_full.isna().any() and "Commence_Hour" in df_valid.columns:
            t_full = t_full.fillna(pd.to_datetime(df_valid["Commence_Hour"], utc=True, errors="coerce"))
        
        # Fill any remaining gaps to keep ordering/stability
        t_full = t_full.fillna(method="ffill").fillna(method="bfill")
        
        # Numpy array for downstream use
        times = t_full.to_numpy()

        # --- (7) time‑forward, group‑safe holdout as before ---
        train_all_idx, hold_idx = holdout_by_percent_groups(
            sport=sport,
            groups=groups,
            times=times,
            y=y_full,
            pct_holdout=None,
            min_train_games=25,
            min_hold_games=8,
            ensure_label_diversity=True
        )
        
        # Make sure indices are positional (ints or boolean masks)
        # If your splitter returns numpy arrays of ints, you’re fine.
        # If it returns pandas Index, cast to numpy:
        #train_all_idx = np.asarray(train_all_idx)
        #hold_idx      = np.asarray(hold_idx)
              
     
                            # --- (8) TRAIN subsets (aligned) ---
        X_train = X_full[train_all_idx]
        y_train = y_full[train_all_idx]
        g_train = groups[train_all_idx]
        t_train = times[train_all_idx]
        # --- Filter CV folds to ensure both train and val have class 0 and 1 ---z
        # ---- Quick diagnostics (optional but handy) ----
        y_hold_vec  = y_full[hold_idx]
        y_train_vec = y_full[train_all_idx]
        #st.write(f"✅ Holdout split → Train: {len(y_train_vec)} | Holdout: {len(y_hold_vec)}")
        #if len(y_train_vec): st.write("Train class counts:", np.bincount(y_train_vec))
        #if len(y_hold_vec):  st.write("Holdout class counts:", np.bincount(y_hold_vec))
        # ✅ This line must come AFTER the split:
        train_df = df_valid.iloc[train_all_idx].copy()
       
        # --- Sample weights: build from train_df ONLY (must match X_train/y_train 1:1) ---
        def _build_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
            # Book column present?
            bk_col = "Bookmaker" if "Bookmaker" in train_df.columns else (
                "Bookmaker_Norm" if "Bookmaker_Norm" in train_df.columns else None
            )
        
            # Base group/book weights (no dropping of rows)
            if bk_col is None:
                w_base = np.ones(len(train_df), dtype=np.float32)
            else:
                B_g  = train_df.groupby("Game_Key")[bk_col].nunique()
                n_gb = train_df.groupby(["Game_Key", bk_col]).size()
        
                TAU = 0.7
                def _w_gb(g, b, tau=1.0):
                    Bg  = max(1, int(B_g.get(g, 1)))
                    ngb = max(1, int(n_gb.get((g, b), 1)))
                    return 1.0 / (Bg * (ngb ** tau))
        
                w_base = np.array([_w_gb(g, b, TAU)
                                   for g, b in zip(train_df["Game_Key"], train_df[bk_col])],
                                  dtype=np.float32)
        
                # Sharp-book tilt
                if "Is_Sharp_Book" in train_df.columns:
                    is_sharp = train_df["Is_Sharp_Book"].fillna(False).astype(bool).to_numpy(dtype=np.float32)
                else:
                    is_sharp = train_df[bk_col].isin(SHARP_BOOKS).to_numpy(dtype=np.float32)
        
                w_base *= (1.0 + 0.20 * is_sharp)
        
            # Market/context multiplier (no row drops)
            def _ctx(m: pd.DataFrame) -> np.ndarray:
                w_book = m.get("Book_Reliability_Score", pd.Series(1.0, index=m.index)).clip(0.6, 1.4)
        
                w_mag  = pd.to_numeric(m.get("Abs_Line_Move_From_Opening", 0), errors="coerce") \
                           .fillna(0).clip(0, 2.0) ** 0.7
        
                tier = m.get("Minutes_To_Game_Tier", pd.Series("", index=m.index)).astype(str)
                is_overnight = (tier == "Overnight_VeryEarly").astype(int)
        
                is_too_late  = pd.to_numeric(m.get("Potential_Overmove_Flag", 0), errors="coerce").fillna(0).astype(int)
                w_time = (1.0 - 0.15*is_too_late) * (1.0 - 0.15*is_overnight)
        
                p0 = pd.to_numeric(m.get("Spread_Implied_Prob", np.nan), errors="coerce") \
                        .fillna(pd.to_numeric(m.get("H2H_Implied_Prob", np.nan), errors="coerce")) \
                        .fillna(0.5).clip(0.01, 0.99)
                w_mid = np.where((p0 > 0.45) & (p0 < 0.55), 1.4, 1.0)
        
                rev = ((pd.to_numeric(m.get("Value_Reversal_Flag", 0), errors="coerce").fillna(0) == 1) |
                       (pd.to_numeric(m.get("Odds_Reversal_Flag", 0),  errors="coerce").fillna(0) == 1)).astype(int)
                w_rev = 1.0 - 0.15*rev
        
                out = (w_book.to_numpy("float32")
                       * w_mag.to_numpy("float32")
                       * w_time.astype("float32")
                       * w_mid.astype("float32")
                       * w_rev.astype("float32"))
                return np.asarray(out, dtype=np.float32)
        
            w = (w_base * _ctx(train_df)).astype(np.float32)
            w[~np.isfinite(w)] = 0.0
        
            # Renormalize to mean 1.0
            s = float(w.sum())
            if s > 0:
                w *= (len(w) / s)
            else:
                w[:] = 1.0
            return w
        
        # Build weights aligned to X_train/y_train
        w_train = _build_sample_weights(train_df)
        
        # Hard alignment checks (will raise immediately if anything is off)
        assert len(train_df) == X_train.shape[0] == len(y_train), \
            ("train_df/X_train/y_train length mismatch",
             len(train_df), X_train.shape[0], len(y_train))
        assert len(w_train) == len(y_train) == X_train.shape[0], \
            ("sample_weight misaligned", len(w_train), len(y_train), X_train.shape[0])

           # ---- 5) Simple health checks (Streamlit friendly) ----
        st.markdown("### 🩺 Data Health Checks")
        
        # 0) Basic shape & groups
        n_rows, n_feats = X_train.shape
        n_games = len(pd.unique(g_train))
        st.write(f"🔢 Train shape: {n_rows} rows × {n_feats} features | 🎮 Unique games: {n_games}")
        
        # 1) Class balance
        pos_rate = float(np.mean(y_train))
        st.write(f"✅ Class balance — Positives: `{pos_rate:.3f}` ({int(pos_rate * 100)}%)")
        st.write(f"Counts — 1s: {int((y_train==1).sum())}, 0s: {int((y_train==0).sum())}")
        
        # 2) NaN / Inf checks
        has_nan = np.isnan(X_train).any()
        has_inf = np.isinf(X_train).any()
        if has_nan or has_inf:
            st.error(f"❌ Bad values in X_train — NaN: {bool(has_nan)}, Inf: {bool(has_inf)}")
        else:
            st.success("✅ No NaNs/Inf in X_train")
        
        # 3) Constant/near-constant features (use float64 for stability)
        n_const = (np.std(X_train.astype("float64"), axis=0) < 1e-6).sum()
        if n_const > 0:
            st.warning(f"⚠️ {int(n_const)} features are near-constant in X_train")
        else:
            st.success("✅ No near-constant features")
        
        # 4) Holdout class diversity (if available)
        if 'y_hold_vec' in locals():
            n_pos = int((y_hold_vec == 1).sum())
            n_neg = int((y_hold_vec == 0).sum())
            st.write(f"📊 Holdout — Pos: {n_pos}, Neg: {n_neg}")
        
        # 5) If you’re using sample weights, sanity check alignment
        if 'w_train' in locals():
            assert len(w_train) == len(y_train) == len(X_train), \
                f"sample_weight misaligned: {len(w_train)} vs {len(y_train)} vs {len(X_train)}"
            bad_w = np.sum(~np.isfinite(w_train))
            if bad_w:
                st.warning(f"⚠️ {int(bad_w)} non‑finite weights; they will be zeroed/renormed.")
            st.write(f"🧮 Weight summary — mean: {float(np.mean(w_train)):.3f}, "
                     f"min: {float(np.min(w_train)):.3f}, max: {float(np.max(w_train)):.3f}")
        
        # Optional: quick book‑level diagnostic (only if you want to use the helper)
      
        def _fallback_book_lift(
            df: pd.DataFrame,
            book_col: str,
            label_col: str,
            prior: float = 50.0,            # shrink less than 200
            canon_mask: pd.Series | None = None,
            min_n: int = 25,
        ):
            if (book_col is None) or (book_col not in df.columns) or (label_col not in df.columns):
                return pd.Series(dtype=float)
        
            d = df
            if canon_mask is not None:
                d = d.loc[canon_mask]
        
            dfv = d[[book_col, label_col]].copy()
            y = pd.to_numeric(dfv[label_col], errors="coerce")
            dfv = dfv.loc[y.notna()]
            dfv["y"] = y.loc[y.notna()].astype(float)
        
            if dfv.empty:
                return pd.Series(dtype=float)
        
            m = float(dfv["y"].mean())
            m = min(max(m, 1e-9), 1 - 1e-9)
        
            grp = dfv.groupby(book_col)["y"].agg(hits="sum", n="count")
            grp = grp.loc[grp["n"] >= min_n]  # optional: ignore tiny samples
            if grp.empty:
                return pd.Series(dtype=float)
        
            a = m * prior
            b = (1.0 - m) * prior
            post_mean = (grp["hits"] + a) / (grp["n"] + a + b)
            lift = (post_mean / m) - 1.0
            return lift.sort_values(ascending=False)
                
        # Example usage (diagnostic only):
        mkt = _norm_market(market)  # <- use your existing normalizer

        # canonical mask for diagnostics only
        if mkt == "totals":
            canon_mask = (
                train_df["Market"].astype(str).str.lower().map(_norm_market).eq(mkt)
                & train_df["Outcome"].astype(str).str.lower().str.strip().eq("over")
            )
        else:
            v = pd.to_numeric(train_df["Value"], errors="coerce")
            canon_mask = (
                train_df["Market"].astype(str).str.lower().map(_norm_market).eq(mkt)
                & v.lt(0)
            )
        
        bk_col = (
            "Bookmaker" if "Bookmaker" in train_df.columns
            else ("Bookmaker_Norm" if "Bookmaker_Norm" in train_df.columns else None)
        )
        
        if bk_col:
            lift = _fallback_book_lift(
                train_df,
                bk_col,
                "SHARP_HIT_BOOL",
                prior=50.0,
                canon_mask=canon_mask,
                min_n=25,
            )
            if not lift.empty:
                st.write(f"🏷️ Book reliability (empirical, smoothed) — {mkt}:")
                st.json(lift.head(20).round(3).to_dict())
        bk_col = "Bookmaker" if "Bookmaker" in train_df.columns else ("Bookmaker_Norm" if "Bookmaker_Norm" in train_df.columns else None)
        if bk_col:
            lift = _fallback_book_lift(train_df, bk_col, "SHARP_HIT_BOOL", prior=50.0, canon_mask=canon_mask, min_n=25)
            if not lift.empty:
                st.write("🏷️ Book reliability (empirical, smoothed):")
                st.json(lift.head(20).round(3).to_dict())
  
       
                
        # --- Base weights from book/group structure ------------------------------------
        bk_col = "Bookmaker" if "Bookmaker" in train_df.columns else (
            "Bookmaker_Norm" if "Bookmaker_Norm" in train_df.columns else None
        )

        
        if bk_col is None:
            w_train = np.ones(len(train_df), dtype=np.float32)
        else:
            if bk_col not in train_df.columns:
                train_df[bk_col] = "UNK"
        
            B_g  = train_df.groupby("Game_Key")[bk_col].nunique()
            n_gb = train_df.groupby(["Game_Key", bk_col]).size()
        
            TAU = 0.7
            def _w_gb(g, b, tau=1.0):
                Bg  = max(1, int(B_g.get(g, 1)))
                ngb = max(1, int(n_gb.get((g, b), 1)))
                return 1.0 / (Bg * (ngb ** tau))
        
            w_base = np.array([_w_gb(g, b, TAU) for g, b in zip(train_df["Game_Key"], train_df[bk_col])],
                              dtype=np.float32)
        
            # Sharp-book tilt (no reliability map here)
            if "Is_Sharp_Book" in train_df.columns:
                is_sharp = train_df["Is_Sharp_Book"].fillna(False).astype(bool).to_numpy(dtype=np.float32)
            else:
                is_sharp = train_df[bk_col].isin(SHARP_BOOKS).to_numpy(dtype=np.float32)
        
            ALPHA_SHARP = 0.10
            mult = 1.0 + ALPHA_SHARP * is_sharp
        
            w_train = (w_base * mult).astype(np.float32)
        
        # --- Market/context multiplier (APPLY AFTER base weights) -----------------------
        def market_context_weights(m: pd.DataFrame) -> np.ndarray:
            w_book = m.get("Book_Reliability_Score", pd.Series(1.0, index=m.index)).clip(0.6, 1.4)
            w_mag  = pd.to_numeric(m.get("Abs_Line_Move_From_Opening", 0), errors="coerce") \
                       .fillna(0).clip(0, 2.0) ** 0.7
        
            # time flags (be robust to dtype)
            if "Minutes_To_Game_Tier" in m.columns:
                tier = m["Minutes_To_Game_Tier"].astype(str)
                is_overnight = (tier == "Overnight_VeryEarly").astype(int)
            else:
                is_overnight = 0
        
            is_too_late = pd.to_numeric(m.get("Potential_Overmove_Flag", 0), errors="coerce").fillna(0).astype(int)
            w_time = (1.0 - 0.15 * is_too_late) * (1.0 - 0.15 * is_overnight)
        
            # mid‑probability emphasis
            p0_spread = pd.to_numeric(m.get("Spread_Implied_Prob", np.nan), errors="coerce")
            p0 = p0_spread.fillna(pd.to_numeric(m.get("H2H_Implied_Prob", np.nan), errors="coerce")) \
                         .fillna(0.5).clip(0.01, 0.99)
            w_mid = np.where((p0 > 0.45) & (p0 < 0.55), 1.4, 1.0)
        
            # reversal deprioritization
            rev = ((pd.to_numeric(m.get("Value_Reversal_Flag", 0), errors="coerce").fillna(0) == 1) |
                   (pd.to_numeric(m.get("Odds_Reversal_Flag", 0),  errors="coerce").fillna(0) == 1)).astype(int)
            w_rev = 1.0 - 0.15 * rev
        
            out = (w_book.to_numpy(dtype="float32")
                   * w_mag.to_numpy(dtype="float32")
                   * w_time.astype("float32")
                   * w_mid.astype("float32")
                   * w_rev.astype("float32"))
            return np.asarray(out, dtype=np.float32)
        
        w_ctx = market_context_weights(train_df)
        w_train = (w_train * w_ctx).astype(np.float32)
        
        # sanitize + renormalize
        w_train[~np.isfinite(w_train)] = 0.0
        s = float(w_train.sum())
        if s > 0:
            w_train *= (len(w_train) / s)
        else:
            w_train[:] = 1.0  # degenerate fallback
        
        assert len(w_train) == len(X_train), f"sample_weight misaligned: {len(w_train)} vs {len(X_train)}"

        # ---------------------------------------------------------------------------
      
          #  CV with purge + embargo (snapshot-aware)

       
        
        # Require BOTH classes with a minimum count (train & val)
        def _has_both_classes(idx, y, *, min_pos=5, min_neg=5):
            yy = y[idx]
            pos = int(np.sum(yy == 1))
            neg = int(np.sum(yy == 0))
            return (pos >= min_pos) and (neg >= min_neg)
        
        def filter_cv_splits(cv_obj, X, y, groups=None, *, min_pos=5, min_neg=5):
            """Return only splits whose train AND val have both classes with minimum counts."""
            safe = []
            for tr, va in cv_obj.split(X, y, groups):
                if _has_both_classes(tr, y, min_pos=min_pos, min_neg=min_neg) and \
                   _has_both_classes(va, y, min_pos=min_pos, min_neg=min_neg):
                    safe.append((tr, va))
            return safe
        
        # ============================= CV + SHAP + SEARCH/REFIT =============================
        
        # --- Helpers ------------------------------------------------------------------------
        GLOBAL_SEED = 1337
        
        def _has_min_counts(idx, y, min_pos=5, min_neg=5):
            yy = y[idx]
            return (yy == 1).sum() >= min_pos and (yy == 0).sum() >= min_neg
        
        def _balance_score(idx, y):  # closer to 50/50 is better (higher score)
            p = float((y[idx] == 1).mean())
            return -abs(p - 0.5)
        
  


        def build_deterministic_folds(
            X, y, *, cv=None, groups=None, times=None, n_splits=5, min_pos=5, min_neg=5, seed=GLOBAL_SEED
        ):
            yb = pd.Series(y, copy=False).astype(int).clip(0, 1).to_numpy()
        
            # (1) candidate splits
            if cv is not None:
                # Try passing times= into cv.split; fall back if the splitter doesn't accept it.
                raw = None
                if times is not None:
                    try:
                        raw = [(tr, va) for tr, va in cv.split(X, yb, groups=groups, times=times)]
                    except TypeError:
                        # Splitter doesn't take 'times' kwarg
                        pass
                if raw is None:
                    raw = [(tr, va) for tr, va in cv.split(X, yb, groups=groups)]
                if not raw and hasattr(cv, "n_splits"):
                    n_splits = int(cv.n_splits)
            else:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                raw = [(tr, va) for tr, va in skf.split(X, yb)]
        
            # (2) keep only splits with both classes (train & val)
            folds = [(tr, va) for tr, va in raw
                     if _has_min_counts(tr, yb, min_pos, min_neg)
                     and _has_min_counts(va, yb, min_pos, min_neg)]
        
            # (3) fallback: single stratified split (20/30/40% val)
            if not folds:
                for ts in (0.20, 0.30, 0.40):
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=ts, random_state=seed)
                    for tr, va in sss.split(np.zeros_like(yb), yb):
                        if _has_min_counts(tr, yb, min_pos, min_neg) and _has_min_counts(va, yb, min_pos, min_neg):
                            folds = [(tr, va)]
                            break
                    if folds:
                        break
                if not folds:
                    raise RuntimeError("No class-balanced CV split available.")
        
            # (4) pick ES fold: val closest to 50/50
            folds.sort(key=lambda tv: _balance_score(tv[1], yb), reverse=True)
            tr_es_rel, va_es_rel = folds[0]
            return folds, tr_es_rel, va_es_rel

        
        def _best_rounds(clf):
            br = getattr(clf, "best_iteration", None)
            if br is not None and br >= 0:
                return int(br + 1)
            try:
                booster = clf.get_booster()
                if getattr(booster, "best_iteration", None) is not None:
                    return int(booster.best_iteration + 1)
                if getattr(booster, "best_ntree_limit", None):
                    return int(booster.best_ntree_limit)
            except Exception:
                pass
            return int(getattr(clf, "n_estimators", 200))
        
        # --- Snapshot‑aware CV with purge + embargo -----------------------------------------
        rows_per_game = int(np.ceil(len(X_train) / max(1, pd.unique(g_train).size)))
        if sport_key in SMALL_LEAGUES:
            target_games  = 10
            min_val_size  = max(12, target_games * rows_per_game)
            embargo_td    = pd.Timedelta(hours=24) if 'embargo_td' not in locals() else embargo_td
        else:
            target_games  = 28
            min_val_size  = max(28, target_games * rows_per_game)
            # Slightly stronger embargo for dense-snapshot sports
            _dense = str(sport_key).upper() in {"NBA", "MLB", "WNBA", "CFL"}
            embargo_td    = pd.Timedelta(hours=18 if _dense else 36)
        
        n_groups_train = pd.unique(g_train).size
        target_folds   = 5 if n_groups_train >= 200 else (4 if n_groups_train >= 120 else 3)
        
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=target_folds,
            embargo=embargo_td,
            time_values=t_train,
            min_val_size=min_val_size,
        )
        
        # Enforce per-fold class presence; reuse your build_deterministic_folds
        y_train = pd.Series(y_train, copy=False).astype(int).clip(0, 1).to_numpy()
        folds, tr_es_rel, va_es_rel = build_deterministic_folds(
            X_train, y_train,
            cv=cv,
            groups=g_train,
            times=t_train,   
            n_splits=getattr(cv, 'n_splits', 5),
            min_pos=5, min_neg=5, seed=1337,
        )
        
        # ================== SHAP stability selection (on pruned set) ==================
        # ================== SHAP stability selection (on pruned set) ==================
        # 1) SHAP stability on a pruned base set (safe & fallback-friendly)
        
        # ======== AUTO FEATURE SELECTION (replaces manual SHAP/per-family/global pruning) ========

        # 1) Build a DataFrame view once
        X_df_train = pd.DataFrame(X_train, columns=list(features_pruned))
        
        def _default_proto():
            # Lightweight, single-threaded proto just for SHAP/perm importance
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric=["logloss","auc"],
                tree_method="hist",
                grow_policy="lossguide",
                max_bin=256,
                n_estimators=400,     # modest; SHAP uses this proto only
                learning_rate=0.05,
                subsample=0.7,
                colsample_bytree=0.6,
                n_jobs=1,            # avoid thread storms during CV/SHAP
            )
        
        try:
            _model_proto = est_auc     # if already defined earlier in your script
        except NameError:
            try:
                _model_proto = est_ll  # else try logloss stream if it exists
            except NameError:
                _model_proto = _default_proto()
        # 2) Pick a model prototype for SHAP/perm importance
  
        # 3) Run automatic selection (SHAP-stability with safe fallback to perm AUC),
        #    then family-wise and global correlation pruning, then league-aware cap.
  
        
        # ======== AUTO FEATURE SELECTION (INLINE / ONE-SHOT) ========
 
        feature_cols, shap_summary = select_features_auto(
            model_proto=_model_proto,
            X_df_train=X_df_train,
            y_train=y_train,
            folds=folds,
            sport_key=sport_key,
            must_keep=[],
            auc_min_k=None,            # ✅ seed = len(must_keep) only
            use_auc_auto=True,         # ✅ actually select (earned)
            auc_patience=200,
            auc_min_improve=5e-6,
            accept_metric="auc",
            auc_verbose=True,
            log_func=log_func,
        
            # keep these for backward compat; can be ignored by your new selector
            topk_per_fold=80,
            min_presence=0.60,
            corr_within=0.90,
            corr_global=0.92,
            max_feats_major=100,
            max_feats_small=80,
            sign_flip_max=0.35,
            shap_cv_max=1.00,
        )
        # 4) Rebuild matrices in the final selected order
        X_train = X_df_train.loc[:, feature_cols].to_numpy(np.float32)
        X_full  = pd.DataFrame(X_full, columns=list(features_pruned)).loc[:, feature_cols].to_numpy(np.float32)
        
        # 5) Sanity checks / logs
        assert X_train.shape[1] == len(feature_cols), f"X_train={X_train.shape[1]} vs feature_cols={len(feature_cols)}"
        assert X_full.shape[1]  == len(feature_cols), f"X_full={X_full.shape[1]} vs feature_cols={len(feature_cols)}"
        assert X_train.shape[0] == y_train.shape[0] == len(train_all_idx)
        assert np.isfinite(X_train).all()
        assert set(np.unique(y_train)) <= {0, 1}
        
        st.write(f"🔎 AutoFS kept {len(feature_cols)} features")
        try:
            st.dataframe(shap_summary.head(25))
        except Exception:
            pass
        
        # ================== Search space + base kwargs (REFactored) ==================
        pos_rate = float(np.mean(y_train))
        n_jobs = 1
        base_kwargs, params_ll, params_auc = get_xgb_search_space(
            sport=sport, X_rows=X_train.shape[0], n_jobs=n_jobs, features=feature_cols
        )
        
        base_kwargs["base_score"] = float(np.clip(pos_rate, 1e-4, 1 - 1e-4))
        
        # -------------------- FAST SEARCH → MODERATE/DEEP REFIT ---------------------
   
        # --------- Capacity knobs (tighter, less overfit) ----------
        SEARCH_N_EST    = 600   # was 900 – reduce trees used in CV search
        SEARCH_MAX_BIN  = 256
        DEEP_N_EST      = 1600  # final deep refit is still allowed more trees
        DEEP_MAX_BIN    = 256
        EARLY_STOP      = 100
        HALVING_FACTOR  = 2
        MIN_RESOURCES   = 24
        VCPUS           = get_vcpus()

        # --- Estimators for search (keep n_jobs=1 for parallel CV) ---
        est_ll  = XGBClassifier(
            **{
                **base_kwargs,
                "n_estimators": SEARCH_N_EST,
                "eval_metric": "logloss",
                "max_bin": SEARCH_MAX_BIN,
                "n_jobs": 1,
            }
        )
        est_auc = XGBClassifier(
            **{
                **base_kwargs,
                "n_estimators": SEARCH_N_EST,
                "eval_metric": "auc",
                "max_bin": SEARCH_MAX_BIN,
                "n_jobs": 1,
            }
        )

    
        # ======================= COMMON PARAMETER SPACE (ULTRA-HARDENED) =======================
        param_space_common = dict(
            # Much shallower / fewer leaves
            max_depth        = randint(2, 4),          # was up to 5
            max_leaves       = randint(12, 48),        # was up to ~96

            # Slow learning
            learning_rate    = loguniform(0.006, 0.022),

            # Aggressive subsampling → weaker trees
            subsample        = uniform(0.45, 0.25),    # 0.45–0.70
            colsample_bytree = uniform(0.45, 0.20),    # 0.45–0.65
            colsample_bynode = uniform(0.45, 0.20),

            # Stronger child / split penalties
            min_child_weight = loguniform(16, 256),    # push heavily toward larger leaves
            gamma            = loguniform(5.0, 30.0),

            # Stronger regularization
            reg_alpha        = loguniform(0.5, 15.0),  # L1
            reg_lambda       = loguniform(40.0, 120.0),# L2

            # Conservative histogram / step size
            max_bin          = randint(192, 320),
            max_delta_step   = loguniform(0.5, 1.8),
        )

        # ======================= STREAM-SPECIFIC ADJUSTMENTS =======================
        params_ll  = dict(param_space_common)

        params_auc = dict(param_space_common)
        params_auc.update(
            dict(
                # AUC stream: even slightly smaller trees
                max_depth        = randint(2, 3),
                max_leaves       = randint(12, 40),

                learning_rate    = loguniform(0.007, 0.018),

                subsample        = uniform(0.50, 0.20),   # 0.50–0.70
                colsample_bytree = uniform(0.48, 0.17),   # 0.48–0.65
                colsample_bynode = uniform(0.48, 0.17),

                min_child_weight = loguniform(20, 256),
                gamma            = loguniform(6.0, 30.0),

                reg_alpha        = loguniform(0.8, 12.0),
                reg_lambda       = loguniform(45.0, 100.0),
            )
        )

       
      
                # --------- Quality thresholds / search config ----------
        thr = get_quality_thresholds(sport, market)

        MIN_AUC           = thr["MIN_AUC"]
        MAX_LOGLOSS       = thr["MAX_LOGLOSS"]
        MAX_ROUNDS        = 30  # still global
        MAX_OVERFIT_GAP   = thr["MAX_OVERFIT_GAP"]
        MIN_AUC_THRESHOLD = thr["MIN_AUC_THRESHOLD"]
        
        fit_params_search = dict(sample_weight=w_train, verbose=False)
        n_jobs_search     = max(1, min(VCPUS, 6))
        
        
        def _make_search_objects(seed_ll: int, seed_auc: int):
            """
            Build rs_ll, rs_auc for a given pair of seeds
            (Halving if possible, else Randomized).
        
            Expects HALVING_FACTOR, MIN_RESOURCES, SEARCH_N_EST to be defined above.
            """
            try:
                rs_ll = HalvingRandomSearchCV(
                    estimator=est_ll,
                    param_distributions=params_ll,
                    factor=HALVING_FACTOR,
                    resource="n_estimators",
                    min_resources=MIN_RESOURCES,
                    max_resources=SEARCH_N_EST,
                    aggressive_elimination=True,
                    scoring="neg_log_loss",
                    cv=folds,
                    n_jobs=n_jobs_search,
                    random_state=seed_ll,
                    verbose=1 if st.session_state.get("debug", False) else 0,
                    return_train_score=True,
                )
                rs_auc = HalvingRandomSearchCV(
                    estimator=est_auc,
                    param_distributions=params_auc,
                    factor=HALVING_FACTOR,
                    resource="n_estimators",
                    min_resources=MIN_RESOURCES,
                    max_resources=SEARCH_N_EST,
                    aggressive_elimination=True,
                    scoring="roc_auc",
                    cv=folds,
                    n_jobs=n_jobs_search,
                    random_state=seed_auc,
                    verbose=1 if st.session_state.get("debug", False) else 0,
                    return_train_score=True,
                )
            except Exception:
                # Fallback: pure RandomizedSearchCV with expanded trials
                try:
                    search_trials  # type: ignore
                except NameError:
                    search_trials = _resolve_search_trials(sport, X_train.shape[0])
        
                search_trials = (
                    int(search_trials)
                    if str(search_trials).isdigit()
                    else _resolve_search_trials(sport, X_train.shape[0])
                )
                search_trials = max(50, int(search_trials * 1.2))
        
                rs_ll = RandomizedSearchCV(
                    estimator=est_ll,
                    param_distributions=params_ll,
                    n_iter=search_trials,
                    scoring="neg_log_loss",
                    cv=folds,
                    n_jobs=n_jobs_search,
                    random_state=seed_ll,
                    verbose=1 if st.session_state.get("debug", False) else 0,
                    return_train_score=True,
                )
                rs_auc = RandomizedSearchCV(
                    estimator=est_auc,
                    param_distributions=params_auc,
                    n_iter=search_trials,
                    scoring="roc_auc",
                    cv=folds,
                    n_jobs=n_jobs_search,
                    random_state=seed_auc,
                    verbose=1 if st.session_state.get("debug", False) else 0,
                    return_train_score=True,
                )
            return rs_ll, rs_auc
        

        
    
        # ======= multi-round search with overfit / quality guards =======
        best_ll_params     = None
        best_auc_params    = None
        best_auc_score     = -np.inf
        best_ll_score      = np.inf
        best_round_metrics = None
        
        # NEW: capture best estimators too (more robust than params-only)
        best_auc_estimator = None
        best_ll_estimator  = None
        
        found_good = False
        
        for round_idx in range(MAX_ROUNDS):
            round_no = round_idx + 1
            seed_ll  = 42  + round_idx
            seed_auc = 137 + round_idx
        
            logger.info(
                f"🔎 Hyperparam search round {round_no}/{MAX_ROUNDS} "
                f"(seeds: ll={seed_ll}, auc={seed_auc})"
            )
        
            rs_ll, rs_auc = _make_search_objects(seed_ll, seed_auc)
        
            # Fit with 1 thread inside search to avoid nested parallelism issues
            with threadpool_limits(limits=1):
                rs_ll.fit(X_train, y_train, groups=g_train, **fit_params_search)
                rs_auc.fit(X_train, y_train, groups=g_train, **fit_params_search)
        
          
            # CV metrics
            auc_cv_raw = float(rs_auc.best_score_)       # already roc_auc
            logloss_cv = float(-rs_ll.best_score_)       # neg_log_loss → logloss
            
            # --- aligned AUC so fade models count ---
            auc_cv = float(max(auc_cv_raw, 1.0 - auc_cv_raw))
            
            # Train vs CV AUC gap (overfit measure)
            train_auc_raw = np.nan
            try:
                cv_res_auc = rs_auc.cv_results_
                if "mean_train_score" in cv_res_auc:
                    train_auc_raw = float(cv_res_auc["mean_train_score"][rs_auc.best_index_])
            except Exception as e:
                logger.warning(
                    f"⚠️ Could not compute train AUC for overfit gap in round {round_no}: {e}"
                )
            
            if np.isfinite(train_auc_raw):
                train_auc = float(max(train_auc_raw, 1.0 - train_auc_raw))
                overfit_gap = float(train_auc - auc_cv)          # aligned gap
            else:
                train_auc = np.nan
                overfit_gap = np.nan
            
            # Optional: record whether this round is "fade-ish" on CV
            # (This is ONLY for reporting; do not use this to flip predictions here.)
            cv_suggests_fade = bool(auc_cv_raw < 0.5)
            
            logger.info(
                f"   🧪 Round {round_no} CV: "
                f"AUC_raw={auc_cv_raw:.4f}, AUC_aligned={auc_cv:.4f}, "
                f"LogLoss={logloss_cv:.4f}, "
                f"TrainAUC_raw={train_auc_raw:.4f}, TrainAUC_aligned={train_auc:.4f}, "
                f"OverfitGap(aligned)={overfit_gap:.4f}, "
                f"cv_fade={cv_suggests_fade}"
            )
            
            # Track best seen across all rounds (even if overfit) for reporting
            if np.isfinite(auc_cv):
                better_global = (
                    (auc_cv > best_auc_score + 1e-6) or
                    (
                        abs(auc_cv - best_auc_score) <= 1e-6 and
                        logloss_cv < best_ll_score - 1e-6
                    )
                )
                if better_global:
                    best_auc_score  = auc_cv
                    best_ll_score   = logloss_cv
                    best_auc_params = rs_auc.best_params_.copy()
                    best_ll_params  = rs_ll.best_params_.copy()
            
                    # NEW: store best estimators (already include best params + fitted state)
                    best_auc_estimator = deepcopy(rs_auc.best_estimator_)
                    best_ll_estimator  = deepcopy(rs_ll.best_estimator_)
            
                 
                    best_round_metrics = dict(
                        round_no=round_no,
                    
                        # ✅ keep old names so existing code doesn't break
                        auc_cv=auc_cv,                 # aligned
                        train_auc=train_auc,           # aligned (or nan)
                        overfit_gap=overfit_gap,       # aligned gap
                    
                        # ✅ new explicit fields
                        auc_cv_raw=auc_cv_raw,
                        auc_cv_aligned=auc_cv,
                        train_auc_raw=train_auc_raw,
                        train_auc_aligned=train_auc,
                        logloss_cv=logloss_cv,
                        cv_fade=cv_suggests_fade,
                    )
                    
            
            # Overfit condition: allow NaN gap (no train scores) or small positive gap
            gap_ok = (not np.isfinite(overfit_gap)) or (overfit_gap <= MAX_OVERFIT_GAP)
            
            # IMPORTANT: gate on ALIGNED AUC, not raw AUC
            auc_ok = np.isfinite(auc_cv)     and (auc_cv     >= MIN_AUC)
            ll_ok  = np.isfinite(logloss_cv) and (logloss_cv <= MAX_LOGLOSS)
            
            if auc_ok and ll_ok and gap_ok:
                logger.info(
                    f"✅ Conditions met in round {round_no} for {sport} {market} "
                    f"(AUC_aligned={auc_cv:.4f}, LL={logloss_cv:.4f}, "
                    f"gap(aligned)={overfit_gap:.4f} ≤ {MAX_OVERFIT_GAP:.4f}, "
                    f"cv_fade={cv_suggests_fade})."
                )
                found_good = True
            
                # keep the accepted round's params + estimators (so downstream uses winner)
                best_auc_params = rs_auc.best_params_.copy()
                best_ll_params  = rs_ll.best_params_.copy()
                best_auc_estimator = deepcopy(rs_auc.best_estimator_)
                best_ll_estimator  = deepcopy(rs_ll.best_estimator_)
            
                break
        
        
        # NEW: if nothing met constraints, proceed with best-so-far instead of returning
        if not found_good:
            if best_round_metrics is not None:
                logger.warning(
                    "⚠️ No config met all constraints for %s %s after %d rounds. "
                    "Proceeding with best-so-far. Best round=%d, AUC=%.4f, LogLoss=%.4f, "
                    "TrainAUC=%.4f, OverfitGap=%.4f.",
                    sport, market, MAX_ROUNDS,
                    best_round_metrics["round_no"],
                    best_round_metrics["auc_cv"],
                    best_round_metrics["logloss_cv"],
                    best_round_metrics["train_auc"],
                    best_round_metrics["overfit_gap"],
                )
            else:
                logger.error(
                    "❌ All rounds failed for %s %s (no successful search results). "
                    "Cannot train a model for this market.",
                    sport, market
                )
                return  # only bail if literally nothing ran successfully
        
        
        # NEW: hard guard — must have something to proceed
        if best_auc_params is None or best_ll_params is None:
            logger.error("❌ No params selected for %s %s; cannot continue.", sport, market)
            return
        
        # Optional: warn if you didn't capture estimators (shouldn't happen now)
        if best_auc_estimator is None or best_ll_estimator is None:
            logger.warning("⚠️ Best estimators not captured; downstream should refit from params.")
        
        # At this point:
        # - if found_good: best_* are from the accepted round
        # - else: best_* are best-so-far across rounds
        # You can now either:
        #   (A) use best_auc_estimator / best_ll_estimator directly, OR
        #   (B) refit fresh on full train set using best_*_params.


        
        # At this point best_*_params are the ones from the accepted round
        assert best_auc_params is not None and best_ll_params is not None
        
        
        # ---------------- stabilize best params (regularization-first) ----------------
        STABLE = dict(
            objective="binary:logistic",
            tree_method="hist",
            grow_policy="lossguide",
            max_delta_step=0.5,
        )
        
        def _stabilize(best_params: dict, leaf_cap: int = 128) -> dict:
            from xgboost import XGBClassifier
            bp = dict(best_params or {})
        
            # Version-safe handling for colsample_bynode vs colsample_bylevel
            try:
                _supports_bynode = ("colsample_bynode" in XGBClassifier().get_xgb_params())
            except Exception:
                _supports_bynode = False
            node_key = "colsample_bynode" if _supports_bynode else "colsample_bylevel"
            node_val = float(bp.get("colsample_bynode", bp.get("colsample_bylevel", 0.80)))
        
            updates = {
                **STABLE,
                "max_depth":         0,  # use leaves
                "max_leaves":        int(min(leaf_cap, int(bp.get("max_leaves", leaf_cap)))),
                "min_child_weight":  float(max(12.0, float(bp.get("min_child_weight", 8.0)))),
                "gamma":             float(max(6.0,  float(bp.get("gamma", 2.0)))),
                "reg_alpha":         float(max(0.10, float(bp.get("reg_alpha", 0.05)))),
                "reg_lambda":        float(max(15.0, float(bp.get("reg_lambda", 6.0)))),
                "subsample":         float(min(0.80, float(bp.get("subsample", 0.85)))),
                "colsample_bytree":  float(min(0.70, float(bp.get("colsample_bytree", 0.80)))),
                node_key:            float(min(0.75, node_val)),
                "max_bin":           int(min(256, int(bp.get("max_bin", 256)))),
                "learning_rate":     float(min(0.02, float(bp.get("learning_rate", 0.025)))),
            }
            bp.update(updates)
        
            # Drop only wrapper/managed keys
            for k in ("monotone_constraints","interaction_constraints","predictor",
                      "eval_metric","_estimator_type","response_method","n_estimators",
                      "n_jobs","scale_pos_weight"):
                bp.pop(k, None)
            return bp
        
        best_auc_params = _stabilize(best_auc_params, leaf_cap=128)
        best_ll_params  = _stabilize(best_ll_params,  leaf_cap=128)
        
        
        # ================== ES refit on last fold (deterministic) ===================
        tr_es_rel, va_es_rel = folds[-1]
        tr_es_rel = np.asarray(tr_es_rel); va_es_rel = np.asarray(va_es_rel)
        
        # Use DataFrames so XGBoost keeps real column names
        X_tr_es = X_train[tr_es_rel]
        X_va_es = X_train[va_es_rel]
        X_tr_es_df = pd.DataFrame(X_tr_es, columns=feature_cols)
        X_va_es_df = pd.DataFrame(X_va_es, columns=feature_cols)
        
        y_tr_es = y_train[tr_es_rel]; y_va_es = y_train[va_es_rel]
        w_tr_es = np.maximum(np.nan_to_num(w_train[tr_es_rel], 0.0), 1e-6)
        w_va_es = np.maximum(np.nan_to_num(w_train[va_es_rel], 0.0), 1e-6)
        W_CLIP  = 5.0
        w_tr_es = np.clip(w_tr_es, 0.0, W_CLIP); w_va_es = np.clip(w_va_es, 0.0, W_CLIP)
        
        # Leak/shape checks
        assert set(tr_es_rel).isdisjoint(set(va_es_rel)), "Train/val overlap in ES fold!"
        assert X_tr_es_df.shape[0] == len(y_tr_es) == len(w_tr_es)
        assert X_va_es_df.shape[0] == len(y_va_es) == len(w_va_es)
        u_tr = set(np.unique(y_tr_es)); u_va = set(np.unique(y_va_es))
        assert {0,1}.issubset(u_tr) and {0,1}.issubset(u_va), \
            "ES fold single-class; widen min_val_size or choose different fold."
        
        # -------- Median-impute (from TRAIN) + low-variance drop (critical) --------
        num_cols = [c for c in feature_cols
                    if c in X_tr_es_df.columns and is_numeric_dtype(X_tr_es_df[c])]
        
        if num_cols:
            med = X_tr_es_df[num_cols].median(numeric_only=True)
            X_tr_es_df.loc[:, num_cols] = X_tr_es_df[num_cols].fillna(med)
            X_va_es_df.loc[:, num_cols] = X_va_es_df[num_cols].fillna(med)
        
            var = X_tr_es_df[num_cols].var(numeric_only=True)
            keep_cols = [c for c in num_cols
                         if np.isfinite(var.get(c, 0.0)) and var.get(c, 0.0) > 1e-10]
            drop_cols = [c for c in num_cols if c not in keep_cols]
        else:
            keep_cols, drop_cols = [], []
        
        if drop_cols:
            try:
                st.info({"dropped_low_variance": drop_cols[:25], "n_drop": len(drop_cols)})
            except Exception:
                pass
            X_tr_es_df = X_tr_es_df.drop(columns=drop_cols, errors="ignore")
            X_va_es_df = X_va_es_df.drop(columns=drop_cols, errors="ignore")
        
        # Build an ES-scoped feature list so we don’t mutate global feature_cols
        feature_cols_es = [c for c in feature_cols if c in X_tr_es_df.columns]
        
        # --- Rebuild arrays from the cleaned DF views so shapes match feature list ---
        X_tr_es = X_tr_es_df[feature_cols_es].to_numpy(copy=False)
        X_va_es = X_va_es_df[feature_cols_es].to_numpy(copy=False)
        
        # Sanity: both train/val must now match the cleaned feature set
        assert X_tr_es.shape[1] == X_va_es.shape[1] == len(feature_cols_es), \
            f"Width mismatch: tr={X_tr_es.shape[1]}, va={X_va_es.shape[1]}, feats={len(feature_cols_es)}"
        # --- Align global design matrices to ES feature subset so widths match everywhere ---
        if len(feature_cols_es) < len(feature_cols):
            drop_lowvar_global = [c for c in feature_cols if c not in feature_cols_es]
            try:
                st.info({
                    "global_lowvar_drop": drop_lowvar_global[:25],
                    "n_drop_global": len(drop_lowvar_global),
                })
            except Exception:
                pass
        
            # Rebuild X_train using only the ES-safe feature set
            X_train_df_global = pd.DataFrame(X_train, columns=list(feature_cols))
            X_train = X_train_df_global.loc[:, feature_cols_es].to_numpy(np.float32)
        
            # Rebuild X_full likewise so final scoring uses the same features
            X_full_df_global = pd.DataFrame(X_full, columns=list(feature_cols))
            X_full = X_full_df_global.loc[:, feature_cols_es].to_numpy(np.float32)
        
            # Make ES feature list the new master list
            feature_cols = list(feature_cols_es)

        # threads for refit
        refit_threads = max(1, min(VCPUS, 6))
        pos_tr = float((y_tr_es == 1).sum()); neg_tr = float((y_tr_es == 0).sum())
        scale_pos_weight = max(1.0, neg_tr / max(pos_tr, 1.0))
        
        # AUC stream (early-stop on logloss; compute AUC after)
        deep_auc = XGBClassifier(**{**base_kwargs, **best_auc_params})
        deep_auc.set_params(
            n_estimators=DEEP_N_EST,
            max_bin=DEEP_MAX_BIN,
            n_jobs=refit_threads,
            eval_metric=["logloss","auc"],
            scale_pos_weight=scale_pos_weight,
            random_state=1337, seed=1337,
        )
        deep_ll = XGBClassifier(**{**base_kwargs, **best_ll_params})
        deep_ll.set_params(
            n_estimators=DEEP_N_EST,
            max_bin=DEEP_MAX_BIN,
            n_jobs=refit_threads,
            eval_metric=["logloss","auc"],
            scale_pos_weight=scale_pos_weight,
            random_state=1337, seed=1337,
        )
        
        # === Preliminary deep fits (for diagnostics/cap sense) ===
        deep_auc.fit(
            X_tr_es_df, y_tr_es,
            sample_weight=w_tr_es,
            eval_set=[(X_va_es_df, y_va_es)],
            sample_weight_eval_set=[w_va_es],
            verbose=False,
            early_stopping_rounds=EARLY_STOP,
        )
        planned_cap = int(deep_auc.get_xgb_params().get("n_estimators", DEEP_N_EST))
        p_va_raw    = np.clip(deep_auc.predict_proba(X_va_es_df)[:, 1], 1e-12, 1 - 1e-12)
        auc_va      = float(roc_auc_score(y_va_es.astype(int), p_va_raw, sample_weight=w_va_es))
        spread_std_raw   = float(np.std(p_va_raw))
        extreme_frac_raw = float(((p_va_raw < 0.35) | (p_va_raw > 0.65)).mean())
        best_iter        = getattr(deep_auc, "best_iteration", None)
        cap_hit          = bool(best_iter is not None and best_iter >= 0.7 * DEEP_N_EST)
        learning_rate    = float(np.clip(float(best_auc_params.get("learning_rate", 0.02)),
                                         0.008, 0.04))
        
        # If ES found a peak, set a tight cap around it; else conservative
        if best_iter is not None and best_iter >= 50:
            final_estimators_cap = int(np.clip(int(1.10 * (best_iter + 1)), 500, 1200))
        else:
            final_estimators_cap = 900
        early_stopping_rounds = int(np.clip(int(0.12 * final_estimators_cap), 60, 180))
        
        deep_ll.fit(
            X_tr_es_df, y_tr_es,
            sample_weight=w_tr_es,
            eval_set=[(X_va_es_df, y_va_es)],
            sample_weight_eval_set=[w_va_es],
            verbose=False,
            early_stopping_rounds=EARLY_STOP,
        )
        if getattr(deep_ll, "best_iteration", None) is not None and deep_ll.best_iteration >= 50:
            deep_ll.set_params(n_estimators=deep_ll.best_iteration + 1)
        
        st.session_state.setdefault("calibration", {})
        st.session_state["calibration"]["spread_favorite_offset"] = float(0.0)
        
        st.subheader("Spread AUC diagnostics")
        y_bar = float(np.mean(y_va_es == 1))
        p_bar = float(np.mean(p_va_raw))
        st.json({
            "best_iter": best_iter,
            "n_estimators": int(deep_auc.get_xgb_params().get("n_estimators", 0)),
            "cap_hit": bool(cap_hit),
            "raw": {
                "spread_std": spread_std_raw,
                "extreme_frac": extreme_frac_raw,
                "y_bar": y_bar,
                "p_bar": p_bar,
            },
            "auc_va_es": auc_va,
        })
        
        
        # ----------------- Helpers (once) -----------------
        def _ece(y_true, p, bins=10):
            y = np.asarray(y_true, float); p = np.asarray(p, float)
            edges = np.linspace(0.0, 1.0, bins + 1)
            idx = np.digitize(p, edges) - 1
            e = 0.0
            for b in range(bins):
                m = (idx == b)
                if m.any():
                    e += (m.mean()) * abs(float(np.mean(y[m])) - float(np.mean(p[m])))
            return float(e)
        
        
        def _fast_auc(y, p, w=None):
            try:
                return float(roc_auc_score(y, p, sample_weight=w)) if np.unique(y).size == 2 else np.nan
            except Exception:
                return np.nan
        
        
        def _psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
            a = np.asarray(a, float); b = np.asarray(b, float)
            edges = np.quantile(a, np.linspace(0, 1, bins+1))
            edges[0]  = min(edges[0],  np.min([a.min(), b.min()]) - 1e-9)
            edges[-1] = max(edges[-1], np.max([a.max(), b.max()]) + 1e-9)
            ah, _ = np.histogram(a, bins=edges); bh, _ = np.histogram(b, bins=edges)
            ah = np.clip(ah / max(len(a),1), 1e-6, 1); bh = np.clip(bh / max(len(b),1), 1e-6, 1)
            return float(np.sum((ah - bh) * np.log(ah / bh)))
        
        
        def _drift_report(X_tr_df, X_va_df, top_k: int = 25) -> dict:
            out = {}
            try:
                var = X_tr_df.var(numeric_only=True).abs().sort_values(ascending=False)
                cols = [c for c in var.index
                        if np.issubdtype(X_tr_df[c].dtype, np.number)][:top_k]
                for c in cols:
                    try:
                        out[c] = _psi(X_tr_df[c].to_numpy(),
                                      X_va_df[c].to_numpy(), bins=10)
                    except Exception:
                        pass
            except Exception:
                pass
            return out
        
        
        # --- Overfit/mismatch diagnostics on ES probe ---
        p_tr_es_raw = np.clip(deep_auc.predict_proba(X_tr_es_df)[:, 1], 1e-12, 1-1e-12)
        auc_tr_es   = auc_safe(y_tr_es.astype(int), p_tr_es_raw)
        ece_va_es   = _ece(y_va_es.astype(int), p_va_raw, bins=10)
        
        # Sanity panel (shuffle / flip / PSI)
        rng = np.random.default_rng(123)
        y_va_shuff = rng.permutation(y_va_es)
        report_auc_va         = _fast_auc(y_va_es, p_va_raw, w_va_es)
        report_auc_va_flip    = _fast_auc(y_va_es, 1.0 - p_va_raw, w_va_es)
        report_auc_va_shuffle = _fast_auc(y_va_shuff, p_va_raw, w_va_es)
        psi_map = _drift_report(X_tr_es_df, X_va_es_df, top_k=25)
        psi_max = float(max(psi_map.values()) if psi_map else 0.0)
        
        try:
            st.json({"auto_harden_sanity": {
                "auc_va": report_auc_va,
                "auc_va_flipped": report_auc_va_flip,
                "auc_va_shuffle": report_auc_va_shuffle,
                "psi_max": psi_max,
                "psi_top": dict(sorted(psi_map.items(), key=lambda kv: kv[1], reverse=True)[:8]),
                "ece_va": ece_va_es,
                "extreme_frac": extreme_frac_raw,
                "y_tr_mean": float(np.mean(y_tr_es)),
                "y_va_mean": float(np.mean(y_va_es)),
            }})
        except Exception:
            pass
        
        ES_SUSPECT = (
            (np.isfinite(report_auc_va_shuffle) and
             abs(report_auc_va - report_auc_va_shuffle) < 0.03)     # looks random vs shuffle
            or (report_auc_va < 0.52 and report_auc_va_flip > 0.55) # possible label polarity
            or (psi_max >= 0.25)                                    # heavy drift
        )
        
        if ES_SUSPECT:
            try:
                st.warning({
                    "es_suspect": True,
                    "why": "shuffle≈val or flip better or PSI high",
                    "auc_va": report_auc_va,
                    "auc_va_shuffle": report_auc_va_shuffle,
                    "auc_va_flipped": report_auc_va_flip,
                    "psi_max": psi_max,
                })
            except Exception:
                pass
        
        # If predictions are too flat, allow a small LR bump so the model can move off 0.5
        if float(np.std(p_va_raw)) < 0.02:
            learning_rate = float(np.clip(learning_rate * 1.25, 0.008, 0.03))
        
        # Very light guard against over-peaky ES behaviour (no full auto-hardening)
        MAX_EXTREME_FRAC_ES = 0.30
        if extreme_frac_raw > MAX_EXTREME_FRAC_ES:
            final_estimators_cap = int(np.clip(final_estimators_cap * 0.85, 300, 1200))
        
        
        # ---------------- Build final param dicts (simple, no auto-hardening) -----------------
        VCPUS = max(1, int(VCPUS))
        
        params_auc_final = {
            **base_kwargs,
            **best_auc_params,                  # stabilized AUC params
            "n_estimators": int(final_estimators_cap),
            "learning_rate": float(learning_rate),
            "eval_metric": "logloss",
            "n_jobs": VCPUS,
            "scale_pos_weight": float(scale_pos_weight),
            "max_bin": int(np.clip(DEEP_MAX_BIN, 128, 256)),
        }
        
        # For logloss model, we can use deep_ll.best_iteration as cap if available
        ll_n_estimators = getattr(deep_ll, "best_iteration", None)
        if ll_n_estimators is None or ll_n_estimators < 50:
            ll_n_estimators = final_estimators_cap
        else:
            ll_n_estimators = int(np.clip(ll_n_estimators + 1, 300, 1200))
        
        params_ll_final = {
            **base_kwargs,
            **best_ll_params,                   # stabilized LL params
            "n_estimators": int(ll_n_estimators),
            "learning_rate": float(learning_rate),
            "eval_metric": "logloss",
            "n_jobs": VCPUS,
            "scale_pos_weight": float(scale_pos_weight),
            "max_bin": int(np.clip(DEEP_MAX_BIN, 128, 256)),
        }
        
        
        # ---------------- Apply monotone constraints ONCE --------------------------
        FEATS_FOR_ES = list(feature_cols_es)  # must correspond to X_tr_es / X_va_es matrices
        
        # Define your directional priors only for features that exist
        MONO = {
            'Abs_Line_Move_From_Opening': +1,
            'Implied_Prob_Shift': +1,
            'Was_Line_Resistance_Broken': +1,
            'Line_Resistance_Crossed_Count': +1,
            'Odds_Reversal_Flag': -1,
        }
        
        # Build vector aligned to FEATS_FOR_ES (zeros for everything else)
        mono_vec = [int(MONO.get(c, 0)) for c in FEATS_FOR_ES]
        
        # Only attach if there is at least one non-zero constraint
        if any(m != 0 for m in mono_vec):
            mono_str = "(" + ",".join(map(str, mono_vec)) + ")"
            params_ll_final['monotone_constraints']  = mono_str
            params_auc_final['monotone_constraints'] = mono_str
        else:
            params_ll_final.pop('monotone_constraints',  None)
            params_auc_final.pop('monotone_constraints', None)
        
        # Final safety: if a mismatch still slips through due to any later column changes,
        # drop the constraints rather than crash.
        for _p in (params_ll_final, params_auc_final):
            if isinstance(_p.get('monotone_constraints'), (list, tuple, str)):
                expected = len(FEATS_FOR_ES)
                size = len(mono_vec)
                if size > expected:
                    _p.pop('monotone_constraints', None)
        
        # --- Instantiate & fit finals ---------------------------------------------
        model_logloss = XGBClassifier(**params_ll_final)
        model_auc     = XGBClassifier(**params_auc_final)
        
        
        model_logloss.fit(
            X_tr_es, y_tr_es,
            sample_weight=w_tr_es,
            eval_set=[(X_va_es, y_va_es)],
            sample_weight_eval_set=[w_va_es],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
        model_auc.fit(
            X_tr_es, y_tr_es,
            sample_weight=w_tr_es,
            eval_set=[(X_va_es, y_va_es)],
            sample_weight_eval_set=[w_va_es],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
        
        # Best rounds
        n_trees_ll  = _best_rounds(model_logloss)
        n_trees_auc = _best_rounds(model_auc)
        
        # Optional: refit on ALL training rows at best rounds (deterministic)
        if n_trees_ll > 0:
            model_logloss.set_params(n_estimators=n_trees_ll)
            model_logloss.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        
        if n_trees_auc > 0:
            model_auc.set_params(n_estimators=n_trees_auc)
            model_auc.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        
        # ---- Stamp feature names on the models so we don't get f0,f1,... ----
        try:
            n_model = int(getattr(model_auc, "n_features_in_", X_train.shape[1]))
            if len(feature_cols) == n_model:
                real_names = list(map(str, feature_cols))
                model_auc.feature_names_in_ = np.asarray(real_names, dtype=object)
                model_logloss.feature_names_in_ = np.asarray(real_names, dtype=object)
                try: model_auc.get_booster().feature_names = real_names
                except Exception: pass
                try: model_logloss.get_booster().feature_names = real_names
                except Exception: pass
        except Exception:
            pass
        
        
        def _safe_metric_tail(clf, prefer):
            ev = getattr(clf, "evals_result_", {}) or {}
            ds = next((k for k in ("validation_0","eval","valid_0") if k in ev), None)
            if not ds:
                return None, {"datasets": list(ev.keys())}
            metrics = ev[ds]
            key = prefer if prefer in metrics else next((k for k in metrics if prefer in k), None)
            if not key:
                return None, {"dataset": ds, "metrics_available": list(metrics.keys())}
            arr = metrics[key]
            return (arr[-10:] if len(arr) >= 10 else arr), {
                "dataset": ds, "metric_key": key, "len": len(arr)
            }
        
        val_logloss_last10, info_log = _safe_metric_tail(model_logloss, "logloss")
        val_auc_last10,     info_auc = _safe_metric_tail(model_auc,     "auc")  # may be None
        
        st.write({
            "ES_n_trees_ll": getattr(model_logloss, "best_iteration", None),
            "ES_n_trees_auc": getattr(model_auc, "best_iteration", None),
            "val_logloss_last10": val_logloss_last10,
            "val_auc_last10": val_auc_last10,
        })
        
        # ================== Lightweight interpretation (optional, guarded) ==================
        DEBUG_INTERP = True
        if DEBUG_INTERP:
        
            # --- Permutation AUC importance with 95% CI (guard single-class) ---
            perm_df = None
            if np.unique(y_va_es).size < 2:
                try: st.warning("Permutation importance skipped: ES validation fold has a single class.")
                except Exception: pass
            else:
                assert X_tr_es.shape[1] == X_va_es.shape[1] == len(feature_cols_es), \
                    f"Width mismatch: tr={X_tr_es.shape[1]}, va={X_va_es.shape[1]}, feats={len(feature_cols_es)}"
        
                X_va_es_df_perm = pd.DataFrame(X_va_es, columns=feature_cols_es)
        
                perm_model = None
                for cand in ("deep_auc", "model_auc", "deep_ll", "model_logloss"):
                    if cand in locals() and hasattr(locals()[cand], "predict_proba"):
                        perm_model = locals()[cand]
                        break
        
                if perm_model is None:
                    try: st.warning("Permutation importance skipped: no fitted model available at this point.")
                    except Exception: pass
                else:
                    kwargs = dict(n_repeats=50, random_state=42)
                    try:
                        import inspect
                        if "stratify" in inspect.signature(perm_auc_importance_ci).parameters:
                            kwargs["stratify"] = True
                    except Exception:
                        pass
        
                    base_auc, perm_df = perm_auc_importance_ci(
                        perm_model,
                        X_va_es_df_perm,
                        y_va_es,
                        **kwargs
                    )
                    try: st.write({"perm_base_auc": float(base_auc)})
                    except Exception: pass
        
                    if perm_df is not None and not perm_df.empty:
                        if "feature" not in perm_df.columns:
                            perm_df = perm_df.reset_index().rename(columns={"index": "feature"})
                        perm_df = perm_df.sort_values("perm_auc_drop_mean", ascending=False)
                        perm_df["significant"] = perm_df["ci_lo"] > 0.001
        
            if perm_df is not None and not perm_df.empty:
                st.subheader("Permutation AUC importance with 95% CI")
                st.dataframe(perm_df.head(25))
        
                sig_top = perm_df.loc[perm_df["significant"]].head(20)
                if not sig_top.empty:
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        y_pos = np.arange(len(sig_top))
                        ax.errorbar(
                            sig_top["perm_auc_drop_mean"].to_numpy(),
                            y_pos,
                            xerr=(
                                (sig_top["perm_auc_drop_mean"] - sig_top["ci_lo"]).to_numpy(),
                                (sig_top["ci_hi"] - sig_top["perm_auc_drop_mean"]).to_numpy()
                            ),
                            fmt="o", capsize=3,
                        )
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(sig_top["feature"].tolist())
                        ax.set_xlabel("AUC drop (mean ± 95% CI)")
                        ax.set_title("Significant permutation importances")
                        plt.tight_layout()
                        st.pyplot(fig, clear_figure=True)
                    except Exception as e:
                        st.info(f"(Optional plot skipped: {e})")
        
            # --- SHAP on fixed, time-correct ES slice ---
            ns = int(min(4000, X_va_es.shape[0]))
            if ns >= 10:
                ns = max(1, min(200, X_va_es.shape[0]))
        
                assert X_va_es.shape[1] == len(feature_cols_es), \
                    f"ES val width {X_va_es.shape[1]} != len(feature_cols_es) {len(feature_cols_es)}"
        
                X_shap_df = pd.DataFrame(X_va_es[:ns], columns=list(feature_cols_es))
        
                def _expected_features(m):
                    names = None
                    try:
                        names = getattr(m, "feature_names_in_", None)
                        if names is not None:
                            names = list(map(str, list(names)))
                    except Exception:
                        pass
                    if not names:
                        try:
                            names = list(m.get_booster().feature_names)
                        except Exception:
                            names = None
                    return names
        
                shap_model = None
                for cand in ("model_auc", "deep_auc", "model_logloss", "deep_ll"):
                    if cand in locals() and hasattr(locals()[cand], "predict_proba"):
                        shap_model = locals()[cand]
                        break
        
                if shap_model is None:
                    try: st.warning("SHAP skipped: no fitted model available at this point.")
                    except Exception: pass
                else:
                    exp_feats = _expected_features(shap_model)
        
                    def _align(df, exp):
                        if not exp:
                            return df, df.columns.tolist()
                        missing = [c for c in exp if c not in df.columns]
                        if missing:
                            for c in missing:
                                df[c] = 0.0
                        df = df[exp]
                        return df, exp
        
                    X_shap_df, used_cols = _align(X_shap_df.copy(), exp_feats)
        
                    try:
                        try:
                            expl = shap.TreeExplainer(
                                shap_model, feature_perturbation="tree_path_dependent"
                            )
                        except Exception:
                            expl = shap.Explainer(shap_model)
                        sv = expl.shap_values(X_shap_df)
                    except Exception as e_pred:
                        sv = None
                        if "deep_auc" in locals() and hasattr(deep_auc, "predict_proba"):
                            exp2 = _expected_features(deep_auc)
                            X_shap_df2, used_cols2 = _align(X_shap_df.copy(), exp2)
                            try:
                                try:
                                    expl = shap.TreeExplainer(
                                        deep_auc, feature_perturbation="tree_path_dependent"
                                    )
                                except Exception:
                                    expl = shap.Explainer(deep_auc)
                                sv = expl.shap_values(X_shap_df2)
                                shap_model = deep_auc
                                X_shap_df = X_shap_df2
                                used_cols = used_cols2
                            except Exception as e_pred2:
                                try:
                                    st.warning(f"SHAP skipped (both models failed): {e_pred2}")
                                except Exception:
                                    pass
                                sv = None
                        else:
                            try: st.warning(f"SHAP skipped: {e_pred}")
                            except Exception: pass
                            sv = None
        
                    if sv is not None:
                        if isinstance(sv, list):
                            sv = sv[1]
                        sv = np.asarray(sv)
        
                        shap_mean = np.abs(sv).mean(0)
                        shap_top = (
                            pd.DataFrame({"feature": used_cols, "mean|SHAP|": shap_mean})
                            .sort_values("mean|SHAP|", ascending=False)
                        )
                        st.subheader("SHAP (AUC model, ES fold)")
                        st.dataframe(shap_top.head(25))
        
                        try:
                            from sklearn.inspection import PartialDependenceDisplay
        
                            top_feats = [
                                f for f in shap_top["feature"].head(6).tolist()
                                if f in X_shap_df.columns
                            ]
                            if top_feats:
                                fig = plt.figure(figsize=(10, 8))
                                ax = plt.gca()
                                PartialDependenceDisplay.from_estimator(
                                    estimator=shap_model,
                                    X=X_shap_df,
                                    features=top_feats,
                                    kind="both",
                                    grid_resolution=20,
                                    response_method="predict_proba",
                                    target=1,
                                    ax=ax
                                )
                                st.subheader("PDP/ICE (AUC model, ES fold)")
                                st.pyplot(fig, clear_figure=True)
                                st.caption(f"PDP/ICE for: {', '.join(top_feats)}")
                            else:
                                st.info("PDP/ICE skipped: no top features available in SHAP frame.")
                        except Exception as e:
                            st.warning(f"PDP/ICE rendering skipped: {e}")
        
        
        # -----------------------------------------
        # OOF predictions (train-only) + blending
        # -----------------------------------------
        SMALL = ((sport_key in SMALL_LEAGUES) or
                 (np.unique(g_train).size < 30) or
                 (len(y_train) < 500))
        MIN_OOF = 40 if SMALL else 120
        RUN_LOGLOSS = True  # keep on; you can tie to SMALL if desired
        
        oof_pred_auc = np.full(len(y_train), np.nan, dtype=np.float64)
        oof_pred_logloss = (np.full(len(y_train), np.nan, dtype=np.float64)
                            if RUN_LOGLOSS else None)
        
        def _maybe_flip(p, flip):
            p = np.asarray(p, float)
            return (1.0 - p) if flip else p
        
        def _decide_flip_on_oof(y, p, min_margin=0.01):
            y = np.asarray(y, int); p = np.asarray(p, float)
            if np.unique(y).size < 2:
                return False
            auc0 = roc_auc_score(y, p); auc1 = roc_auc_score(y, 1.0 - p)
            return bool((auc1 - auc0) > min_margin)
        
        # --- Make OOFs with safe proba + per-fold flip only for AUC semantics ----------
        for tr_rel, va_rel in folds:
            m_auc = XGBClassifier(
                **{**base_kwargs, **best_auc_params,
                   "n_estimators": int(n_trees_auc), "n_jobs": 1}
            )
            m_auc.fit(X_train[tr_rel], y_train[tr_rel],
                      sample_weight=w_train[tr_rel], verbose=False)
            pa, _ = pos_proba_safe(m_auc, X_train[va_rel], positive=1)
            _, pa_fixed, _ = auc_with_flip(
                y_train[va_rel].astype(int), pa, w_train[va_rel]
            )
            oof_pred_auc[va_rel] = np.clip(pa_fixed, eps, 1 - eps)
        
            if RUN_LOGLOSS:
                m_ll = XGBClassifier(
                    **{**base_kwargs, **best_ll_params,
                       "n_estimators": int(n_trees_ll), "n_jobs": 1}
                )
                m_ll.fit(X_train[tr_rel], y_train[tr_rel],
                         sample_weight=w_train[tr_rel], verbose=False)
                pl, _ = pos_proba_safe(m_ll, X_train[va_rel], positive=1)
                oof_pred_logloss[va_rel] = np.clip(pl.astype(np.float64), eps, 1 - eps)
        
        mask_auc = np.isfinite(oof_pred_auc)
        mask_log = np.isfinite(oof_pred_logloss) if RUN_LOGLOSS else mask_auc
        mask_oof = mask_auc & mask_log
        n_oof = int(mask_oof.sum())
        
        # --- Small-league gap fill with ES models for rows never validated ------------
        if SMALL and n_oof < MIN_OOF:
            miss_auc = ~mask_auc
            if miss_auc.any():
                pa_fill, _ = pos_proba_safe(model_auc, X_train[miss_auc], positive=1)
                oof_pred_auc[miss_auc] = np.clip(pa_fill, eps, 1 - eps)
                mask_auc = np.isfinite(oof_pred_auc)
        
            if RUN_LOGLOSS and (oof_pred_logloss is not None):
                miss_ll = ~np.isfinite(oof_pred_logloss)
                if miss_ll.any():
                    pl_fill, _ = pos_proba_safe(model_logloss,
                                                X_train[miss_ll], positive=1)
                    oof_pred_logloss[miss_ll] = np.clip(pl_fill, eps, 1 - eps)
        
            mask_log = np.isfinite(oof_pred_logloss) if RUN_LOGLOSS else mask_auc
            mask_oof = mask_auc & mask_log
            n_oof = int(mask_oof.sum())
        
        # --- Assemble OOF sources (handles low/normal coverage cleanly) ----------------
        if n_oof < MIN_OOF:
            if SMALL:
                _, va_es_rel = folds[-1]
                y_oof = y_train[va_es_rel].astype(int)
                pa_es, _ = pos_proba_safe(model_auc, X_train[va_es_rel], positive=1)
                _, p_oof_auc, _ = auc_with_flip(
                    y_oof, pa_es, w_train[va_es_rel]
                )
                p_oof_auc = np.clip(p_oof_auc.astype(np.float64), eps, 1 - eps)
                if RUN_LOGLOSS:
                    pl_es, _ = pos_proba_safe(model_logloss,
                                              X_train[va_es_rel], positive=1)
                    p_oof_log = np.clip(pl_es, eps, 1 - eps).astype(np.float64)
                else:
                    p_oof_log = None
                st.info(
                    f"Small-league fallback: using last-fold validation only "
                    f"(n={len(va_es_rel)}) for blend weight."
                )
            else:
                y_oof = y_train[mask_auc].astype(int)
                p_oof_auc = np.clip(oof_pred_auc[mask_auc], eps, 1 - eps).astype(np.float64)
                p_oof_log = None
                st.warning(
                    f"OOF coverage low ({n_oof}); proceeding with AUC-only blend source."
                )
        else:
            y_oof = y_train[mask_oof].astype(int)
            p_oof_auc = np.clip(oof_pred_auc[mask_oof], eps, 1 - eps).astype(np.float64)
            p_oof_log = (np.clip(oof_pred_logloss[mask_oof], eps, 1 - eps).astype(np.float64)
                         if RUN_LOGLOSS else None)
        
        # --- Choose blend weight ------------------------------------------------------
        use_discrete_grid = (n_oof < MIN_OOF) and RUN_LOGLOSS and (p_oof_log is not None)
        if use_discrete_grid:
            CAND = [0.35, 0.50, 0.65] if not SMALL else [0.35, 0.50]
            best_w, best_ll = 0.50, np.inf
            for w in CAND:
                mix = np.clip(w * p_oof_log + (1 - w) * p_oof_auc, 1e-6, 1 - 1e-6)
                ll = log_loss(y_oof, mix, labels=[0,1])
                if ll < best_ll:
                    best_ll, best_w = ll, w
            p_oof_blend = np.clip(
                best_w * (p_oof_log if p_oof_log is not None else p_oof_auc)
                + (1 - best_w) * p_oof_auc,
                1e-6, 1 - 1e-6
            )
        else:
            best_w, p_oof_blend, _ = pick_blend_weight_on_oof(
                y_oof=y_oof,
                p_oof_auc=p_oof_auc,
                p_oof_log=p_oof_log if RUN_LOGLOSS else None,
                eps=eps,
                metric="hybrid",
                hybrid_alpha=0.92,   # 0.90–0.95 sweet spot for spreads
            )
        
        p_oof_blend = np.asarray(p_oof_blend, dtype=np.float64)
        if not np.isfinite(p_oof_blend).all():
            keep2 = np.isfinite(p_oof_blend)
            y_oof, p_oof_blend = y_oof[keep2], p_oof_blend[keep2]
        
        st.write({
            "SMALL": bool(SMALL),
            "RUN_LOGLOSS": bool(RUN_LOGLOSS),
            "n_oof": int(n_oof),
            "blend_w": float(best_w),
            "oof_minmax": (float(p_oof_blend.min()), float(p_oof_blend.max()))
        })
        
        assert np.isfinite(p_oof_blend).all(), "NaNs in p_oof_blend"
        
        
        def _prior_correct(p, train_pos, hold_pos, clip=1e-6):
            p = np.clip(p, clip, 1-clip)
            def logit(x): return np.log(x/(1-x))
            def sigmoid(z): return 1.0/(1.0+np.exp(-z))
            shift = np.log((hold_pos/(1-hold_pos)) / (train_pos/(1-train_pos)))
            return sigmoid(logit(p) + shift)
        
        # --- Choose priors for calibration context ---
        oof_pos = float(np.mean(y_oof == 1))
        deploy_pos = float(np.mean(y_full[hold_idx] == 1))
        
        # ---------------- Calibration ----------------
        # ---------------- Calibration ----------------
        # ---------------- Calibration ----------------
        
        # 0) Safe defaults (prevents UnboundLocalError on any path)
        CLIP = 0.03 if str(market).lower().strip() == "spreads" else (0.02 if bool(locals().get("SMALL", False)) else 0.01)
        
        cal_name = "iso"
        cal_obj  = _IdentityIsoCal(eps=1e-6)
        cal_blend = (cal_name, cal_obj)
        
        # 1) Decide global flip on OOF
        flip_flag = _decide_flip_on_oof(y_oof, p_oof_blend)
        
        # 2) OOF prior (train) and deployment prior (hold)
        oof_pos    = float(np.mean(y_oof == 1))
        deploy_pos = float(np.mean(y_full[hold_idx] == 1))  # or your rolling/ES estimate
        
        # 3) Apply SAME flip to OOF before prior-correction
        p_oof_for_cal = (1.0 - p_oof_blend) if flip_flag else p_oof_blend
        
        # 4) PRIOR-CORRECT OOF before fitting calibrators
        p_oof_prior = _prior_correct(p_oof_for_cal, train_pos=oof_pos, hold_pos=deploy_pos)
        
        # 5) Fit/normalize available calibrators on prior-corrected OOF
        use_qiso = (len(np.unique(np.round(p_oof_prior, 4))) < 400)
        cals_raw = fit_iso_platt_beta(p_oof_prior, y_oof, eps=1e-6, use_quantile_iso=use_qiso)
        cals = _normalize_cals(cals_raw)
        cals["iso"]   = _ensure_transform_for_iso(cals.get("iso")) or _IdentityIsoCal(eps=1e-6)
        cals["platt"] = _ensure_predict_proba_for_prob_cal(cals.get("platt"), eps=1e-6)
        cals["beta"]  = _ensure_predict_proba_for_prob_cal(cals.get("beta"),  eps=1e-6)
        
        # 6) Evaluate candidates by ECE (lower is better) on prior-corrected OOF
        candidates = []
        if cals.get("beta")  is not None: candidates.append(("beta",  cals["beta"]))
        if cals.get("platt") is not None: candidates.append(("platt", cals["platt"]))
        if cals.get("iso")   is not None: candidates.append(("iso",   cals["iso"]))
        if not candidates:
            candidates = [("iso", _IdentityIsoCal(eps=1e-6))]
        
        scores = []
        for kind, cal in candidates:
            try:
                pp = _apply_cal(kind, cal, p_oof_prior)
                pp = np.asarray(np.clip(pp, CLIP, 1 - CLIP), float)
                ece = expected_calibration_error(y_oof, pp)
                if np.isfinite(ece):
                    scores.append((float(ece), kind, cal))
            except Exception as e:
                st.debug(f"Calibrator {kind} failed: {e}")
        
        # 7) Pick best calibrator (or fallback)
        if scores:
            scores.sort(key=lambda t: t[0])
            ece_best, cal_name, cal_obj = scores[0]
            cal_blend = (cal_name, cal_obj)
            st.write({"calibrator_used": str(cal_name), "flip_on_oof": bool(flip_flag), "ece_best": float(ece_best)})
        else:
            st.warning("No valid calibrator produced finite ECE; using identity isotonic.")
            cal_name, cal_obj = "iso", _IdentityIsoCal(eps=1e-6)
            cal_blend = (cal_name, cal_obj)
        
        # 8) Temperature scaling ON TOP of chosen calibrator (fit on OOF)
        p_cal_oof = _apply_cal(cal_name, cal_obj, p_oof_prior)
        p_cal_oof = np.asarray(np.clip(p_cal_oof, CLIP, 1 - CLIP), float)
        
        T_best, T_ll = fit_temperature_on_oof(y_oof, p_cal_oof)
        st.write({"temp_T": float(T_best), "temp_ll": float(T_ll)})

        scores = []
        for kind, cal in candidates:
            try:
                pp = _apply_cal(kind, cal, p_oof_prior)
                pp = np.asarray(np.clip(pp, CLIP, 1 - CLIP), float)
                ece = expected_calibration_error(y_oof, pp)
                if np.isfinite(ece):
                    scores.append((float(ece), kind, cal))
            except Exception as e:
                st.debug(f"Calibrator {kind} failed: {e}")
        
        if scores:
            scores.sort(key=lambda t: t[0])
            ece_best, cal_name, cal_obj = scores[0]
            cal_blend = (cal_name, cal_obj)
            st.write({"calibrator_used": str(cal_name), "flip_on_oof": bool(flip_flag), "ece_best": float(ece_best)})
        else:
            st.warning("No valid calibrator produced finite ECE; using identity isotonic.")
            # cal_blend remains ('iso', IdentityIsoCal)
        
        # --- Raw model preds (AUC + optional LogLoss model) ---------------------------
        p_tr_auc, _ = pos_proba_safe(model_auc,     X_full[train_all_idx], positive=1)
        p_ho_auc, _ = pos_proba_safe(model_auc,     X_full[hold_idx],      positive=1)
        
        if RUN_LOGLOSS:
            p_tr_log, _ = pos_proba_safe(model_logloss, X_full[train_all_idx], positive=1)
            p_ho_log, _ = pos_proba_safe(model_logloss, X_full[hold_idx],      positive=1)
            p_train_blend_raw = np.clip(best_w * p_tr_log + (1 - best_w) * p_tr_auc, eps, 1 - eps)
            p_hold_blend_raw  = np.clip(best_w * p_ho_log + (1 - best_w) * p_ho_auc, eps, 1 - eps)
        else:
            p_tr_log = np.array([], dtype=float); p_ho_log = np.array([], dtype=float)
            p_train_blend_raw = np.clip(p_tr_auc, eps, 1 - eps)
            p_hold_blend_raw  = np.clip(p_ho_auc,  eps, 1 - eps)
        
        # Apply SAME flip before prior-correction & calibration
        p_train_blend_raw = _maybe_flip(p_train_blend_raw, flip_flag)
        p_hold_blend_raw  = _maybe_flip(p_hold_blend_raw,  flip_flag)
        
        # PRIOR-CORRECT train/hold to deployment prior
        train_pos_for_pc = float(np.mean(y_full[train_all_idx] == 1))
        p_train_prior = _prior_correct(p_train_blend_raw, train_pos=train_pos_for_pc, hold_pos=deploy_pos)
        p_hold_prior  = _prior_correct(p_hold_blend_raw,  train_pos=train_pos_for_pc, hold_pos=deploy_pos)
        

        # Calibrate using the chosen calibrator
        p_cal_tr = _apply_cal(cal_name, cal_obj, p_train_prior)
        p_cal_ho = _apply_cal(cal_name, cal_obj, p_hold_prior)
        
        # Clip (required before logit/temperature)
        p_cal_tr = np.asarray(np.clip(p_cal_tr, CLIP, 1 - CLIP), float)
        p_cal_ho = np.asarray(np.clip(p_cal_ho, CLIP, 1 - CLIP), float)
        
        # Temperature scaling (bet-friendly)
        p_cal_tr = apply_temperature(p_cal_tr, T_best)
        p_cal_ho = apply_temperature(p_cal_ho, T_best)
        
        # Safety clip (optional but fine)
        p_cal_tr = np.asarray(np.clip(p_cal_tr, CLIP, 1 - CLIP), float)
        p_cal_ho = np.asarray(np.clip(p_cal_ho, CLIP, 1 - CLIP), float)
        
        # Gentle shrink of extremes
        p_train_vec = 0.95 * p_cal_tr + 0.05 * 0.5
        p_hold_vec  = 0.95 * p_cal_ho + 0.05 * 0.5


        
        # Diagnostics
        ece_tr = expected_calibration_error(y_full[train_all_idx].astype(int), p_train_vec, n_bins=10)
        ece_ho = expected_calibration_error(y_full[hold_idx].astype(int),      p_hold_vec,  n_bins=10)
        psi    = population_stability_index(p_train_vec, p_hold_vec, bins=20)
        st.write({"cal_used": cal_name, "flip_on_oof": bool(flip_flag),
                  "ece_train": float(ece_tr), "ece_hold": float(ece_ho), "psi": float(psi)})
        
        assert p_train_vec.shape[0] == len(train_all_idx), "p_train_vec length mismatch"
        assert p_hold_vec.shape[0]  == len(hold_idx),      "p_hold_vec length mismatch"

        

        # ---------- Metrics (no flips here) -------------------------------------------
        y_train_vec = y_full[train_all_idx].astype(int)
        y_hold_vec  = y_full[hold_idx].astype(int)
        # ---- Robust time-split check (no NameErrors) ----
      

        # Prevalence shift (can wreck calibration/AUC)
        rate_tr = float(y_full[train_all_idx].mean())
        rate_ho = float(y_full[hold_idx].mean())
        st.write({"label_rate_train": rate_tr, "label_rate_hold": rate_ho})

        auc_train = auc_safe(y_train_vec, p_train_vec)
        auc_val   = auc_safe(y_hold_vec,  p_hold_vec)
        brier_tr  = brier_score_loss(y_train_vec, p_train_vec)
        brier_val = brier_score_loss(y_hold_vec,  p_hold_vec)
        
        st.write(f"🔧 Ensemble weight (logloss vs auc): w={best_w:.2f}")
        st.write(
            f"📉 LogLoss: train={log_loss(y_train_vec, p_train_vec, labels=[0,1]):.5f}, "
            f"val={log_loss(y_hold_vec,  p_hold_vec,  labels=[0,1]):.5f}"
        )
        st.write(
            f"📈 AUC:     train={(auc_train if not np.isnan(auc_train) else '—')}, "
            f"val={(auc_val if not np.isnan(auc_val) else '—')}"
        )
        st.write(f"🎯 Brier:   train={brier_tr:.4f},  val={brier_val:.4f}")
        
        # ==== HOLDOUT: Calibration bins (quantile) + ROI per bin ======================
        # ==== HOLDOUT: Calibration bins (quantile) + ROI per bin ======================
        st.markdown("#### 🎯 Calibration Bins (blended + calibrated)")
        
        # Strong shape/alignment guard
        assert p_hold_vec.shape[0] == y_hold_vec.shape[0] == len(hold_idx)
        _hold_pos = np.asarray(hold_idx, dtype=int)
        
        df_eval = pd.DataFrame({
            "p":    np.clip(np.asarray(p_hold_vec, float), eps, 1 - eps),
            "y":    np.asarray(y_hold_vec, int),
            "odds": pd.to_numeric(df_valid.iloc[_hold_pos]["Odds_Price"], errors="coerce"),
        }).dropna(subset=["p", "y"])
        
        # Degeneracy check: nearly (or exactly) constant probabilities
        uniq_hold = np.unique(np.round(df_eval["p"].to_numpy(), 6)).size
        if uniq_hold < 5:
            st.warning(
                f"⚠️ Holdout predictions are nearly discrete (unique≈{uniq_hold}). "
                "If this persists, check upstream constraints/regularization and calibrator inputs."
            )
        
        # If there's no variation, skip binning to avoid misleading output
        if uniq_hold <= 1 or len(df_eval) == 0:
            st.info("Holdout predictions are effectively constant; skipping calibration bins.")
            st.dataframe(
                pd.DataFrame(columns=["Prob Bin", "N", "Hit Rate", "Avg ROI (unit)", "Avg Pred P"]),
                use_container_width=True
            )
        else:
            # Try quantile bins; fall back to uniform bins
            try:
                cuts = pd.qcut(df_eval["p"], q=10, duplicates="drop")
            except Exception:
                cuts = pd.cut(df_eval["p"], bins=np.linspace(0.0, 1.0, 11), include_lowest=True)
        
            def _roi_mean_inline(sub: pd.DataFrame) -> float:
                if not sub["odds"].notna().any():
                    return float("nan")
                odds = pd.to_numeric(sub["odds"], errors="coerce")
                win  = sub["y"].astype(int)
                # Profit on win (American odds)
                profit_pos = odds.where(odds > 0, np.nan) / 100.0
                profit_neg = 100.0 / odds.abs()
                profit_on_win = np.where(odds > 0, profit_pos, profit_neg)
                profit_on_win = pd.Series(profit_on_win, index=odds.index).fillna(0.0)
                # ROI per bet (unit stake on each example)
                roi = win * profit_on_win - (1 - win) * 1.0
                return float(roi.mean())
        
            gb = df_eval.groupby(cuts, observed=True, sort=False)
        
            if getattr(gb, "ngroups", 0) == 0:
                df_bins = pd.DataFrame(columns=["Prob Bin", "N", "Hit Rate", "Avg ROI (unit)", "Avg Pred P"])
            else:
                # Build columns explicitly as 1-D python lists
                prob_bins = [f"{s.min():.2f}-{s.max():.2f}" for _, s in gb["p"]]
                n_vals    = gb.size().astype(int).to_list()
                hit_rate  = gb["y"].mean().astype(float).to_list()
                avg_pred  = gb["p"].mean().astype(float).to_list()
                avg_roi   = gb.apply(_roi_mean_inline).astype(float).to_list()
        
                df_bins = (pd.DataFrame({
                    "Prob Bin": prob_bins,
                    "N": n_vals,
                    "Hit Rate": hit_rate,
                    "Avg ROI (unit)": avg_roi,
                    "Avg Pred P": avg_pred,
                })
                .sort_values("Avg Pred P")
                .reset_index(drop=True))
        
            st.dataframe(df_bins, use_container_width=True)

        
        # ---- Overfitting check (gaps) ------------------------------------------------
        auc_tr  = auc_train
        auc_ho  = auc_val
        ll_tr   = log_loss(y_train_vec, p_train_vec, labels=[0, 1])
        ll_ho   = log_loss(y_hold_vec,  p_hold_vec,  labels=[0, 1])
        br_tr   = brier_tr
        br_ho   = brier_val
        
        st.markdown("### 📉 Overfitting Check – Gap Analysis")
        st.write(f"- AUC Gap (Train - Holdout): `{(auc_tr - auc_ho):.4f}`")
        st.write(f"- LogLoss Gap (Train - Holdout): `{(ll_tr  - ll_ho):.4f}`")
        st.write(f"- Brier Gap (Train - Holdout): `{(br_tr  - br_ho):.4f}`")
        
        # ---- calibration quality: ECE + Brier decomposition --------------------------
        def ECE(y_true, p, n_bins: int = 10) -> float:
            y_true = np.asarray(y_true, dtype=float)
            p = np.asarray(p, dtype=float)
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            idx = np.digitize(p, bins) - 1
            ece = 0.0
            N = len(p)
            for b in range(n_bins):
                m = (idx == b)
                if np.any(m):
                    conf = float(np.mean(p[m]))
                    acc  = float(np.mean(y_true[m]))
                    ece += (np.sum(m) / N) * abs(acc - conf)
            return float(ece)
        
        def brier_decomp(y_true, p, n_bins: int = 10):
            y_true = np.asarray(y_true, dtype=float)
            p = np.asarray(p, dtype=float)
            base = float(np.mean(y_true))
            unc  = base * (1.0 - base)
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            idx  = np.digitize(p, bins) - 1
            res = 0.0; rel = 0.0
            for b in range(n_bins):
                m = (idx == b)
                if np.any(m):
                    py = float(np.mean(y_true[m]))
                    pp = float(np.mean(p[m]))
                    w  = float(np.mean(m))
                    res += w * (py - base) ** 2
                    rel += w * (pp - py) ** 2
            brier = brier_score_loss(y_true, p)
            return float(brier), float(unc), float(res), float(rel)
        
        ece_tr = ECE(y_train_vec, p_train_vec, 10)
        brier_tr_v, unc_tr, res_tr, rel_tr = brier_decomp(y_train_vec, p_train_vec, 10)
        
        ece_ho = ECE(y_hold_vec, p_hold_vec, 10)
        brier_ho_v, unc_ho, res_ho, rel_ho = brier_decomp(y_hold_vec, p_hold_vec, 10)
        
        st.write(
            f"- ECE(train): `{ece_tr:.4f}` | Brier(train): `{brier_tr_v:.4f}` = "
            f"Unc `{unc_tr:.4f}` - Res `{res_tr:.4f}` + Rel `{rel_tr:.4f}`"
        )
        st.write(
            f"- ECE(holdout): `{ece_ho:.4f}` | Brier(holdout): `{brier_ho_v:.4f}` = "
            f"Unc `{unc_ho:.4f}` - Res `{res_ho:.4f}` + Rel `{rel_ho:.4f}`"
        )
     
        # ---- Feature importance using model-native names only -------------------------
        # ---- Feature importance using model-native names only (small corrections) ----
        # ---- Feature importance using model-native names only (hardened) ----
        def _model_feature_names(clf, n_expected: int | None = None) -> list[str]:
            # 1) sklearn wrapper
            names = getattr(clf, "feature_names_in_", None)
            if names is not None and len(names) > 0:
                return [str(x) for x in list(names)]
            # 2) Booster names
            try:
                booster = clf.get_booster()
                if getattr(booster, "feature_names", None):
                    names = list(booster.feature_names)
                    if names:
                        return [str(x) for x in names]
            except Exception:
                pass
            # 3) Positional fallback
            n = int(getattr(clf, "n_features_in_", 0)) or \
                int(len(getattr(clf, "feature_importances_", []))) or \
                int(n_expected or 0)
            return [f"f{i}" for i in range(max(0, n))]
        
        def _corr_sign_from_matrix(clf, X_mat: np.ndarray, feat_names: list[str]) -> list[str] | None:
            """Return '↑ Increases' / '↓ Decreases' per feature by corr with proba."""
            try:
                p = clf.predict_proba(X_mat)[:, 1]
            except Exception:
                return None
            p = np.asarray(p, float)
            finite_p = np.isfinite(p)
            out = []
            for j in range(X_mat.shape[1]):
                x = X_mat[:, j]
                m = finite_p & np.isfinite(x)
                if m.sum() < 3 or np.std(x[m]) <= 0:
                    out.append("↓ Decreases")  # neutral default
                else:
                    c = np.corrcoef(x[m], p[m])[0, 1]
                    out.append("↑ Increases" if float(c) > 0 else "↓ Decreases")
            return out
        
        importances = np.asarray(getattr(model_auc, "feature_importances_", []), dtype=float)
        if importances.size == 0:
            st.error("❌ model_auc.feature_importances_ is empty. (Was the model fit?)")
        else:
            # Prefer model-stamped names; fallback to Booster; else f0..f{n-1}
            names_all = _model_feature_names(model_auc, n_expected=importances.size)
        
            # Strict alignment for display
            k = min(importances.size, len(names_all))
            names = names_all[:k]
            importances = importances[:k]
        
            importance_df = (
                pd.DataFrame({"Feature": names, "Importance": importances})
                  .sort_values("Importance", ascending=False)
                  .reset_index(drop=True)
            )
            st.markdown("#### 📊 Feature Importance (model-native names)")
            st.dataframe(importance_df, use_container_width=True)
        
            # Try df_market (if columns match); otherwise fall back to X_train matrix.
            did_sign = False
            try:
                if "df_market" in locals():
                    available = [n for n in names if n in df_market.columns]
                    if len(available) >= 2:
                        # keep the same order and length cap as 'names'
                        available = [n for n in names if n in set(available)]
                        X_features = (
                            df_market[available]
                            .apply(pd.to_numeric, errors="coerce")
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0)
                            .astype("float32")
                        )
                        preds = model_auc.predict_proba(X_features[available])[:, 1]
                        finite = np.isfinite(preds)
                        corrs = []
                        for col in available:
                            x = X_features[col].to_numpy()
                            m = finite & np.isfinite(x)
                            if m.sum() < 3 or np.std(x[m]) <= 0:
                                corrs.append(0.0)
                            else:
                                c = np.corrcoef(x[m], preds[m])[0, 1]
                                corrs.append(float(np.nan_to_num(c, nan=0.0)))
                        sign_map = {n: ("↑ Increases" if v > 0 else "↓ Decreases") for n, v in zip(available, corrs)}
                        final_df = (
                            importance_df.merge(pd.DataFrame({"Feature": list(sign_map.keys()),
                                                              "Impact": list(sign_map.values())}),
                                                on="Feature", how="left")
                                        .sort_values("Importance", ascending=False)
                                        .reset_index(drop=True)
                        )
                        st.markdown("##### ➕ Impact sign (from df_market)")
                        st.dataframe(final_df, use_container_width=True)
                        did_sign = True
            except Exception:
                # soft-fail; we’ll try the matrix fallback
                pass
        
            if not did_sign:
                try:
                    # Align with names/importances length to avoid indexing errors
                    X_mat = np.asarray(X_train, dtype=np.float32)
                    X_mat = X_mat[:, :min(X_mat.shape[1], len(names))]  # extra guard
                    if X_mat.shape[1] != len(names):
                        # If this happens, the model was fit with a different view than X_train here.
                        # Keep them aligned by truncation.
                        names = names[:X_mat.shape[1]]
                        importances = importances[:X_mat.shape[1]]
                    signs = _corr_sign_from_matrix(model_auc, X_mat, names)
                    if signs is not None:
                        final_df = (
                            pd.DataFrame({"Feature": names, "Importance": importances, "Impact": signs})
                              .sort_values("Importance", ascending=False)
                              .reset_index(drop=True)
                        )
                        st.markdown("##### ➕ Impact sign (from train matrix)")
                        st.dataframe(final_df, use_container_width=True)
                    else:
                        st.info("Skipped impact sign: could not compute predictions on train matrix.")
                except Exception:
                    st.info("Skipped impact sign: fallback computation failed.")


        # ==== TRAIN: Calibration curve table (quantile bins, robust) ==================
        
        
        p_train_final = np.clip(np.asarray(p_cal_tr, float), eps, 1 - eps)
        y_train_plot  = y_train_vec.astype(int)
        assert len(p_train_final) == len(y_train_plot) == len(train_all_idx)
        
        if not np.isfinite(p_train_final).all():
            st.error("❌ Non-finite values in train probabilities after clipping.")
            p_train_final = np.nan_to_num(p_train_final, nan=0.5, posinf=0.999, neginf=0.001)
        
        uniq_train = np.unique(np.round(p_train_final, 6)).size
        if uniq_train < 5:
            st.warning(
                f"⚠️ Train predictions are nearly discrete (unique≈{uniq_train}). "
                "Ensure you calibrate on probabilities (not logits/labels) and apply any AUC-based flip *before* calibration."
            )
        
        cal_df_src = pd.DataFrame({"p": p_train_final, "y": y_train_plot})
        try:
            qb = pd.qcut(cal_df_src["p"], q=10, duplicates="drop")
        except Exception:
            qb = pd.cut(cal_df_src["p"], bins=np.linspace(0, 1, 11), include_lowest=True)
        
        g = cal_df_src.groupby(qb, observed=True, sort=False)
        calib_df = pd.DataFrame({
            "Predicted Bin Center": g["p"].mean().values,
            "Actual Hit Rate": g["y"].mean().values,
            "N": g.size().astype(int).values,
        }).sort_values("Predicted Bin Center").reset_index(drop=True)
        
        st.markdown("#### 🎯 Calibration Bins – Train")
        st.dataframe(calib_df)
        
        # ---- CV fold health -----------------------------------------------------------
        st.markdown("### 🧩 CV Fold Health")
        cv_rows = []
        for i, (_, va) in enumerate(folds, 1):
            yv = y_train[va].astype(int)
            cv_rows.append({
                "Fold": i,
                "ValN": int(len(va)),
                "ValPosRate": float(np.mean(yv)) if len(yv) else float("nan"),
                "ValBothClasses": bool(np.unique(yv).size == 2)
            })
        st.dataframe(pd.DataFrame(cv_rows))



                
        # --- aliases from the blended/calibrated step ---
       

        ensemble_prob = p_train_vec
        
        if ('X_val' in locals() and isinstance(X_val, pd.DataFrame)) \
           and ('y_val' in locals() and isinstance(y_val, (pd.Series, pd.DataFrame))):
            y_val = y_val.loc[X_val.index]
        
        # Safe metric helpers
        def _safe_auc(yv, pv):
            if len(yv) != len(pv) or np.unique(yv).size < 2:
                return np.nan
            return roc_auc_score(yv, pv)
        

        def _safe_ll(yv, pv):
            if len(yv) != len(pv):
                return np.nan
            return log_loss(yv, np.clip(pv, 1e-6, 1 - 1e-6), labels=[0, 1])

        def _safe_brier(yv, pv):
            if len(yv) != len(pv):
                return np.nan
            return brier_score_loss(yv, np.clip(pv, 1e-6, 1 - 1e-6))

        # targets aligned
        y_train_vec = y_full[train_all_idx].astype(int)
        y_hold_vec  = y_full[hold_idx].astype(int)

        # sanity checks
        assert len(y_train_vec) == len(p_train_vec), f"train len mismatch: y={len(y_train_vec)} p={len(p_train_vec)}"
        assert len(y_hold_vec)  == len(p_hold_vec),  f"holdout len mismatch: y={len(y_hold_vec)} p={len(p_hold_vec)}"

        acc_train = accuracy_score(y_train_vec, (p_train_vec >= 0.5).astype(int)) if np.unique(y_train_vec).size == 2 else np.nan
        acc_hold  = accuracy_score(y_hold_vec,  (p_hold_vec  >= 0.5).astype(int)) if np.unique(y_hold_vec).size  == 2 else np.nan

        auc_train = roc_auc_score(y_train_vec, p_train_vec)
        auc_hold  = roc_auc_score(y_hold_vec,  p_hold_vec)

        logloss_train = log_loss(y_train_vec, np.clip(p_train_vec, 1e-6, 1 - 1e-6), labels=[0, 1])
        logloss_hold  = log_loss(y_hold_vec,  np.clip(p_hold_vec,  1e-6, 1 - 1e-6), labels=[0, 1])

        brier_train = brier_score_loss(y_train_vec, p_train_vec)
        brier_hold  = brier_score_loss(y_hold_vec,  p_hold_vec)

        # For champion/challenger wrapper: capture per-market holdout metrics
        if return_artifacts:
            artifact_metrics = {
                "auc_holdout": float(auc_hold),
                "logloss_holdout": float(logloss_hold),
                "auc_gap_train_holdout": float(auc_train - auc_hold),
            }
            artifact_config = {
                "sport": sport,
                "market": market,
                "feature_count": len(feature_cols),
            }

        # --- Book reliability (build DF exactly as apply expects) ---
        bk_col = 'Bookmaker' if 'Bookmaker' in df_market.columns else 'Bookmaker_Norm'
        need   = ['Sport', 'Market', bk_col, 'SHARP_HIT_BOOL']

        df_rel_in = df_market.loc[:, [c for c in need if c in df_market.columns]].copy()
        if bk_col != 'Bookmaker':  # builder expects 'Bookmaker'
            df_rel_in.rename(columns={bk_col: 'Bookmaker'}, inplace=True)

        try:
            book_reliability_map = build_book_reliability_map(df_rel_in, prior_strength=200.0)
        except Exception as e:
            logger.warning(f"book_reliability_map build failed; defaulting to empty. err={e}")
            book_reliability_map = pd.DataFrame(columns=[
                'Sport', 'Market', 'Bookmaker', 'Book_Reliability_Score', 'Book_Reliability_Lift'
            ])

        # safety check (optional but nice)
        _bake_feature_names_in_(model_logloss, feature_cols)
        _bake_feature_names_in_(model_auc, feature_cols)

        if cal_blend is None:
            cal_blend = ("iso", _IdentityIsoCal(eps=1e-6))
        cal_name, cal_obj = cal_blend
        iso_blend = _CalAdapter(cal_blend, clip=(CLIP, 1 - CLIP))

        # --- Helper to get final calibrated probs (blend → optional flip → calibrate+clip) ---
        def predict_calibrated(models: dict, X):
            # required keys: model_auc, model_logloss, best_w, iso_blend
            pa = models["model_auc"].predict_proba(X)[:, 1]
            pl = models["model_logloss"].predict_proba(X)[:, 1]
            p_blend = float(models.get("best_w", 0.5)) * pl + (1.0 - float(models.get("best_w", 0.5))) * pa
            if models.get("flip_flag", False):
                p_blend = 1.0 - p_blend
            # iso_blend is your _CalAdapter, already clipping (e.g., (CLIP, 1-CLIP))
            return models["iso_blend"].predict(p_blend)

        # === Save ensemble (choose one or both) ===
       
        trained_models[market] = {
            "model_logloss":        model_logloss,
            "model_auc":            model_auc,
            "flip_flag":            bool(flip_flag),
            "iso_blend":            iso_blend,
            "best_w":               float(best_w),
            "team_feature_map":     team_feature_map,
            "book_reliability_map": book_reliability_map,
            "feature_cols":         feature_cols,
        }
        
        save_info = save_model_to_gcs(
            model={  # ✅ THIS IS THE IMPORTANT PART
                "model_logloss": model_logloss,
                "model_auc":     model_auc,
                "best_w":        float(best_w),
                "feature_cols":  feature_cols,
                "flip_flag":     bool(flip_flag),   # ✅ persist polarity
                # optional but useful:
                # "trained_at": datetime.utcnow().isoformat() + "Z",
                # "sport": sport,
                # "market": market,
            },
            calibrator=iso_blend,                 # saved as payload["iso_blend"] by your saver
            sport=sport,
            market=market,
            bucket_name=bucket_name,
            team_feature_map=team_feature_map,
            book_reliability_map=book_reliability_map,
        )
        if return_artifacts and isinstance(save_info, dict):
            artifact_model_path = f"gs://{save_info.get('bucket', bucket_name)}/{save_info.get('path')}"

        # Use the holdout metrics you already computed
        auc    = float(auc_hold)
        acc    = float(acc_hold)
        logloss = float(logloss_hold)
        brier   = float(brier_hold)

        # Keep this in the status box
        status.write(
            f"""✅ Trained + saved ensemble model for {market.upper()}
- AUC: {auc:.4f}
- Accuracy: {acc:.4f}
- Log Loss: {logloss:.4f}
- Brier Score: {brier:.4f}
"""
        )

        pb.progress(min(100, max(0, pct)))

    # ← end of for i, market in enumerate(markets_present, 1)

    status.update(label="✅ All models trained", state="complete", expanded=False)

    if return_artifacts:
        # Champion/challenger mode: return a single artifact dict instead of per-market map
        if not artifact_model_path:
            st.error("❌ Challenger training did not produce a model artifact.")
            return None
        return {
            "model_path": artifact_model_path,
            "metrics": artifact_metrics,
            "config": artifact_config,
        }

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
from sklearn.model_selection import StratifiedKFold

def train_timing_opportunity_model(
    sport: str = "NBA",
    days_back: int = 35,
    table_fq: str = "sharplogger.sharp_data.scores_with_features",
    gcs_bucket: str | None = None,
):
    """
    Train per-market timing models and (UI-only) compute Timing_Opportunity_Score/Timing_Stage
    on the same historical df. Saves a payload with the exact feature_list used.
    """


    st.info(f"🧠 Training timing opportunity models for {sport.upper()}...")

    # ---- Resolve bucket once (param > global > env); bail if missing ----
    if gcs_bucket is None:
        gcs_bucket = globals().get("GCS_BUCKET") or os.getenv("GCS_BUCKET")
    if not gcs_bucket:
        st.error("No GCS bucket configured. Pass gcs_bucket=... or set GCS_BUCKET env/global.")
        return

    # === Load historical scored data (robust to Scored being missing) ===
    query = f"""
        SELECT *
        FROM `{table_fq}`
        WHERE UPPER(Sport) = '{sport.upper()}'
          AND (Scored IS NULL OR Scored = TRUE)
          AND SHARP_HIT_BOOL IS NOT NULL
          AND DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
    """
    df = bq_client.query(query).to_dataframe()
    if df.empty:
        st.warning("⚠️ No historical sharp picks available.")
        return

    # === Normalize casing + types ===
    df = df.copy()
    if 'Market' in df.columns:
        df['Market'] = df['Market'].astype(str).str.lower().str.strip()
    elif 'Market_norm' in df.columns:
        df['Market'] = df['Market_norm'].astype(str).str.lower().str.strip()
    else:
        df['Market'] = ""  # leads to per-market empties and skip

    # canonical-only to avoid inverse duplication
    if 'Was_Canonical' in df.columns:
        df = df[df['Was_Canonical'].fillna(1).astype(int) == 1]

    df['SHARP_HIT_BOOL'] = pd.to_numeric(df['SHARP_HIT_BOOL'], errors='coerce').fillna(0).astype(int)

    # If Model_Sharp_Win_Prob is missing (e.g., spreads train excluded it), keep neutral fallback
    if 'Model_Sharp_Win_Prob' not in df.columns:
        df['Model_Sharp_Win_Prob'] = 0.0
    else:
        df['Model_Sharp_Win_Prob'] = pd.to_numeric(df['Model_Sharp_Win_Prob'], errors='coerce').fillna(0.0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # =========================
    #  TRAIN & SAVE PER MARKET
    # =========================
    for market in MARKETS:
        df_market = df[df['Market'] == market].copy()
        if df_market.empty:
            st.warning(f"⚠️ No rows for market: {market}")
            continue

        # Ensure numeric for movement columns introduced in your pipeline
        for c in ['Abs_Line_Move_From_Opening', 'Abs_Odds_Move_From_Opening']:
            if c not in df_market.columns:
                df_market[c] = 0.0
            df_market[c] = pd.to_numeric(df_market[c], errors='coerce').fillna(0.0)

        # --- Label definition ---
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

        # --- Feature set: movement + hybrid timing (present in your schema) ---
        base_feats = [
            'Abs_Line_Move_From_Opening',
            'Abs_Odds_Move_From_Opening',
            'Late_Game_Steam_Flag',
        ]

        # Use your standardized hybrid magnitude blocks
        blocks = [
            'Overnight_VeryEarly','Overnight_MidRange','Overnight_LateGame','Overnight_Urgent',
            'Early_VeryEarly','Early_MidRange','Early_LateGame','Early_Urgent',
            'Midday_VeryEarly','Midday_MidRange','Midday_LateGame','Midday_Urgent',
            'Late_VeryEarly','Late_MidRange','Late_LateGame','Late_Urgent'
        ]
        timing_feats = [f'SharpMove_Magnitude_{b}' for b in blocks]
        odds_feats   = [f'OddsMove_Magnitude_{b}' for b in blocks]

        feature_cols = [c for c in (base_feats + timing_feats + odds_feats) if c in df_market.columns]

        # Optional: include Minutes_To_Game_Tier if available (numeric or categorical)
        if 'Minutes_To_Game_Tier' in df_market.columns:
            if pd.api.types.is_numeric_dtype(df_market['Minutes_To_Game_Tier']):
                feature_cols.append('Minutes_To_Game_Tier')
            else:
                df_market['Minutes_To_Game_Tier_num'] = (
                    df_market['Minutes_To_Game_Tier'].astype(str).str.strip().factorize()[0].astype(float)
                )
                feature_cols.append('Minutes_To_Game_Tier_num')

        if not feature_cols:
            st.warning(f"⚠️ No usable features for {market} — skipping.")
            continue

        X = df_market[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        y = df_market['TIMING_OPPORTUNITY_LABEL']

        # --- Safety checks for CV ---
        if y.nunique() < 2:
            st.warning(f"⚠️ Not enough variation in label for {market} — skipping.")
            continue
        n_min_class = min((y == 0).sum(), (y == 1).sum())
        if len(y) < 50 or n_min_class < 5:
            st.warning(f"⚠️ Not enough samples for robust CV (n={len(y)}, min_class={n_min_class}) — skipping {market}.")
            continue

        # --- Train calibrated model ---
        base = GradientBoostingClassifier(random_state=42)
        calibrated = CalibratedClassifierCV(base, method='isotonic', cv=cv)
        calibrated.fit(X, y)

        # --- Save a payload with the exact feature list used ---
        payload = {
            "model": calibrated,
            "feature_list": feature_cols,
            "market": market,
            "sport": sport.upper(),
        }
        save_model_to_gcs(
            model=payload,              # okay to pickle a dict
            calibrator=None,
            sport=sport,
            market=f"timing_{market}",
            bucket_name=gcs_bucket,
            team_feature_map=None
        )
        st.success(f"✅ Timing model saved for {sport.upper()} · {market.upper()} (features={len(feature_cols)})")

    
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



from typing import Dict
import numpy as np

try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# module cache works in training jobs and also reduces requery across reruns
_PR_BETA_CACHE: dict[tuple[str, str, str, str, int, float], Dict[str, float]] = {}
# key: (project, dataset, table, ratings_table, min_rows, lag_hours)

from typing import Dict

def compute_pr_points_slopes_from_scores(
    bq,
    project: str = "sharplogger",
    dataset: str = "sharp_data",
    table: str = "sharp_scores_with_features",  
    ratings_table: str = "sharplogger.sharp_data.ratings_history",
    min_rows_per_sport: int = 5000,
    rating_lag_hours: float = 12.0,  # leakage-safe
) -> Dict[str, float]:
    """
    Learns beta per sport where:
        margin ≈ beta * (Team_Rating - Opp_Rating)

    beta is "POINTS per 1 rating point".
    Uses scores_with_features columns:
      Sport, Market, feat_Game_Start, feat_Team, feat_Opponent, Team_Score, Opp_Score
    """

    from google.cloud import bigquery

    sql = f"""
    WITH base AS (
      SELECT
        UPPER(CAST(Sport AS STRING)) AS Sport,
        CAST(Market AS STRING) AS Market,
        TIMESTAMP(feat_Game_Start) AS Game_Start,
        LOWER(TRIM(CAST(feat_Team AS STRING))) AS Team_Norm,
        LOWER(TRIM(CAST(feat_Opponent AS STRING))) AS Opp_Norm,
        SAFE_CAST(Team_Score AS FLOAT64) AS Team_Score,
        SAFE_CAST(Opp_Score  AS FLOAT64) AS Opp_Score,
        (SAFE_CAST(Team_Score AS FLOAT64) - SAFE_CAST(Opp_Score AS FLOAT64)) AS margin
      FROM `{project}.{dataset}.{table}`
      WHERE feat_Game_Start IS NOT NULL
        AND Team_Score IS NOT NULL
        AND Opp_Score IS NOT NULL
        AND feat_Team IS NOT NULL
        AND feat_Opponent IS NOT NULL
        AND Sport IS NOT NULL
        AND Market = 'spreads'
    ),

    team_r AS (
      SELECT
        b.*,
        r.Rating AS Team_Rating
      FROM base b
      JOIN `{ratings_table}` r
        ON UPPER(CAST(r.Sport AS STRING)) = b.Sport
       AND LOWER(TRIM(CAST(r.Team  AS STRING))) = b.Team_Norm
       AND TIMESTAMP(r.Updated_At) <= TIMESTAMP_SUB(b.Game_Start, INTERVAL CAST(@lag_hours AS INT64) HOUR)
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY b.Sport, b.Game_Start, b.Team_Norm, b.Opp_Norm
        ORDER BY TIMESTAMP(r.Updated_At) DESC
      ) = 1
    ),

    both_r AS (
      SELECT
        t.*,
        r.Rating AS Opp_Rating
      FROM team_r t
      JOIN `{ratings_table}` r
        ON UPPER(CAST(r.Sport AS STRING)) = t.Sport
       AND LOWER(TRIM(CAST(r.Team  AS STRING))) = t.Opp_Norm
       AND TIMESTAMP(r.Updated_At) <= TIMESTAMP_SUB(t.Game_Start, INTERVAL CAST(@lag_hours AS INT64) HOUR)
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY t.Sport, t.Game_Start, t.Team_Norm, t.Opp_Norm
        ORDER BY TIMESTAMP(r.Updated_At) DESC
      ) = 1
    ),

    x AS (
      SELECT
        Sport,
        (SAFE_CAST(Team_Rating AS FLOAT64) - SAFE_CAST(Opp_Rating AS FLOAT64)) AS pr_diff,
        margin
      FROM both_r
      WHERE Team_Rating IS NOT NULL
        AND Opp_Rating  IS NOT NULL
        AND ABS(SAFE_CAST(Team_Rating AS FLOAT64) - SAFE_CAST(Opp_Rating AS FLOAT64)) > 1e-9
    ),

    agg AS (
      SELECT
        Sport,
        COUNT(*) AS n,
        SAFE_DIVIDE(SUM(pr_diff * margin), SUM(pr_diff * pr_diff)) AS beta
      FROM x
      GROUP BY Sport
    )
    SELECT Sport, n, beta
    FROM agg
    WHERE n >= @min_rows
      AND beta IS NOT NULL
      AND ABS(beta) < 1000
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("min_rows", "INT64", int(min_rows_per_sport)),
            bigquery.ScalarQueryParameter("lag_hours", "FLOAT64", float(rating_lag_hours)),
        ]
    )

    df = bq.query(sql, job_config=job_config).to_dataframe()
    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        out[str(r["Sport"]).upper()] = float(r["beta"])
    return out


def resolve_pr_beta_map(
    bq,
    project: str = "sharplogger",
    dataset: str = "sharp_data",
    table: str = "sharp_scores_with_features",  
    min_rows_per_sport: int = 5000,
    rating_lag_hours: float = 12.0
) -> Dict[str, float]:
    key = (project, dataset, table, int(min_rows_per_sport), float(rating_lag_hours))

    # 1) module cache always works (training + UI)
    cached = _PR_BETA_CACHE.get(key)
    if cached is not None:
        return cached

    # 2) streamlit session cache only if actually available
    if _HAS_ST:
        try:
            cache_key = f"pr_beta::{project}.{dataset}.{table}::{int(min_rows_per_sport)}"
            if cache_key in st.session_state:
                _PR_BETA_CACHE[key] = st.session_state[cache_key]
                return _PR_BETA_CACHE[key]
        except Exception:
            # not running under `streamlit run`
            pass

    # 3) compute once
    beta_map = compute_pr_points_slopes_from_scores(
        bq,
        project=project,
        dataset=dataset,
        table=table,
        min_rows_per_sport=int(min_rows_per_sport),
        rating_lag_hours=float(rating_lag_hours),
    )

    # store in caches (best effort for streamlit)
    _PR_BETA_CACHE[key] = beta_map
    if _HAS_ST:
        try:
            st.session_state[cache_key] = beta_map
        except Exception:
            pass

    return beta_map

import pandas as pd

def attach_pr_points_edge_cols(
    df: pd.DataFrame,
    *,
    bq,
    project: str = "sharplogger",
    dataset: str = "sharp_data",
    table: str = "sharp_scores_with_features",  
    min_rows_per_sport: int = 5000,
    sport_col: str = "Sport",
    pr_diff_col: str = "PR_Rating_Diff",
) -> pd.DataFrame:
    """
    Adds:
      - PR_Edge_Pts
      - PR_Abs_Edge_Pts
      - PR_Pts_Beta_Used (optional but useful for debugging)

    Requires df[Sport] and df[PR_Rating_Diff].
    """
    out = df.copy()

    # ensure sport strings
    if sport_col not in out.columns or pr_diff_col not in out.columns:
        out["PR_Edge_Pts"] = np.nan
        out["PR_Abs_Edge_Pts"] = np.nan
        out["PR_Pts_Beta_Used"] = np.nan
        return out

    beta_map = resolve_pr_beta_map(
        bq,
        project=project,
        dataset=dataset,
        table=table,
        min_rows_per_sport=int(min_rows_per_sport),
    )

    # last-resort fallback mapping (only used if sport missing / low rows)
    beta_fallback_map = {
        "NFL":   1.0/25.0,
        "NCAAF": 1.0/25.0,
        "NBA":   1.0/35.0,
        "WNBA":  1.0/35.0,
        "CFL":   1.0/25.0,
        "MLB":   1.0/50.0,
        "NCAAB": 1.0,        # only if your NCAAB ratings are already in points
    }

    sp = out[sport_col].astype(str).str.upper().values
    beta = np.array([beta_map.get(s, np.nan) for s in sp], dtype="float32")
    beta_fb = np.array([beta_fallback_map.get(s, 1.0/30.0) for s in sp], dtype="float32")
    beta = np.where(np.isfinite(beta), beta, beta_fb).astype("float32")

    pr_diff = pd.to_numeric(out[pr_diff_col], errors="coerce").astype("float32")
    out["PR_Pts_Beta_Used"] = beta
    out["PR_Edge_Pts"] = (pr_diff * beta).astype("float32")
    out["PR_Abs_Edge_Pts"] = np.abs(out["PR_Edge_Pts"]).astype("float32")
    return out


# =======================
# Main function
# =======================

def attach_ratings_and_edges_for_diagnostics(
    df: pd.DataFrame,
    sport_aliases: dict,
    table_history: str = "sharplogger.sharp_data.ratings_current",
    project: str = "sharplogger",
    pad_days: int = 30,
    allow_forward_hours: float = 0.0,
    bq=None,                        # ✅ pass your BigQuery client in
) -> pd.DataFrame:
    """
    UI diagnostics helper (spreads only):
      - Attaches power ratings (method-aware when using ratings_history)
      - Computes model vs market spreads/edges + cover probs
      - Maps game-level metrics to each outcome row
    """


    UI_EDGE_COLS = [
      'PR_Team_Rating','PR_Opp_Rating','PR_Rating_Diff','PR_Abs_Rating_Diff',
      'PR_Edge_Pts','PR_Abs_Edge_Pts',
      'PR_Pts_Beta_Used',   # ✅ add this (optional)
      'Outcome_Model_Spread','Outcome_Market_Spread','Outcome_Spread_Edge',
      'Outcome_Cover_Prob','model_fav_vs_market_fav_agree','edge_x_k','mu_x_k'
    ]
    if df is None or df.empty:
        out = (df.copy() if df is not None else pd.DataFrame())
        for c in UI_EDGE_COLS:
            out[c] = np.nan
        return out

    if bq is None:
        out = df.copy()
        for c in UI_EDGE_COLS:
            out[c] = np.nan
        out["Ratings_Attach_Error"] = "bq client is None"
        return out

    out = df.copy()

    def _series(col, default=""):
        return out[col] if col in out.columns else pd.Series(default, index=out.index)

    # normalize base fields
    out['Sport']          = _series('Sport').astype(str).str.upper().str.strip()
    out['Market']         = _series('Market').astype(str).str.lower().str.strip()
    out['Home_Team_Norm'] = _series('Home_Team_Norm').astype(str).str.lower().str.strip()
    out['Away_Team_Norm'] = _series('Away_Team_Norm').astype(str).str.lower().str.strip()

    out['Outcome_Norm'] = (
        _series('Outcome_Norm', None)
        .where((_series('Outcome_Norm', None).notna()) if 'Outcome_Norm' in out else False,
               _series('Outcome'))
        .astype(str).str.lower().str.strip()
    )

    if 'Game_Start' not in out.columns or out['Game_Start'].isna().all():
        out['Game_Start'] = pd.to_datetime(_series('Snapshot_Timestamp', pd.NaT),
                                           utc=True, errors='coerce')

    # spreads only
    mask = out['Market'].eq('spreads') & out['Home_Team_Norm'].ne('') & out['Away_Team_Norm'].ne('')
    if not mask.any():
        for c in UI_EDGE_COLS:
            if c not in out.columns:
                out[c] = np.nan
        return out

    need_cols = ['Sport','Game_Start','Home_Team_Norm','Away_Team_Norm','Outcome_Norm','Value']
    d_sp = (
        out.loc[mask, need_cols]
           .dropna(subset=['Sport','Home_Team_Norm','Away_Team_Norm','Outcome_Norm','Value'])
           .copy()
    )
    if d_sp.empty:
        for c in UI_EDGE_COLS:
            out[c] = np.nan
        return out

    d_sp['Value'] = pd.to_numeric(d_sp['Value'], errors='coerce').astype('float32')
    sport0 = str(d_sp['Sport'].iloc[0]).upper()

    # -------------------------
    # Method-aware ratings selection for UI
    # -------------------------
    is_current_table = 'ratings_current' in str(table_history).lower()

    if is_current_table and sport0 == "NCAAB":
        ratings_table_used = "sharplogger.sharp_data.ratings_history"
        ui_lag_hours = 0.0
        pad_days_used = max(int(pad_days), 60)
    else:
        ratings_table_used = table_history
        ui_lag_hours = 0.0 if is_current_table else 12.0
        pad_days_used = (365 if is_current_table else int(pad_days))

    base = enrich_power_for_training_lowmem(
        df=d_sp[['Sport','Home_Team_Norm','Away_Team_Norm','Game_Start']].drop_duplicates(),
        bq=bq,
        sport_aliases=sport_aliases,
        table_history=ratings_table_used,
        pad_days=pad_days_used,
        rating_lag_hours=float(ui_lag_hours),
        project=project,
    )

    # consensus market
    cons = prep_consensus_market_spread_lowmem(d_sp, value_col='Value', outcome_col='Outcome_Norm')

    game_keys = ['Sport','Home_Team_Norm','Away_Team_Norm']
    g_full = base.merge(cons, on=game_keys, how='left')
    g_fc   = favorite_centric_from_powerdiff_lowmem(g_full)

    # ensure PR cols exist
    for c in ['Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff']:
        if c not in g_fc.columns:
            g_fc[c] = g_full[c] if c in g_full.columns else np.nan

    d_map = d_sp.merge(g_fc, on=game_keys, how='left')

    # map game-level to each row’s outcome
    lhs = d_map['Outcome_Norm'].astype(str).str.lower().str.strip()
    rhs = d_map['Market_Favorite_Team'].astype(str).str.lower().str.strip()
    is_fav_row = lhs.eq(rhs)

    d_map['Outcome_Model_Spread']  = np.where(is_fav_row, d_map['Model_Fav_Spread'], d_map['Model_Dog_Spread']).astype('float32')
    d_map['Outcome_Market_Spread'] = np.where(is_fav_row, d_map['Favorite_Market_Spread'], d_map['Underdog_Market_Spread']).astype('float32')
    d_map['Outcome_Spread_Edge']   = np.where(is_fav_row, d_map['Fav_Edge_Pts'], d_map['Dog_Edge_Pts']).astype('float32')
    d_map['Outcome_Cover_Prob']    = np.where(is_fav_row, d_map['Fav_Cover_Prob'], d_map['Dog_Cover_Prob']).astype('float32')

    # PR columns: team/opponent from bet side
    is_home_bet = d_map['Outcome_Norm'].eq(d_map['Home_Team_Norm'])
    d_map['PR_Team_Rating'] = np.where(is_home_bet, d_map['Home_Power_Rating'], d_map['Away_Power_Rating']).astype('float32')
    d_map['PR_Opp_Rating']  = np.where(is_home_bet, d_map['Away_Power_Rating'], d_map['Home_Power_Rating']).astype('float32')
    d_map['PR_Rating_Diff'] = (
        pd.to_numeric(d_map['PR_Team_Rating'], errors='coerce') -
        pd.to_numeric(d_map['PR_Opp_Rating'],  errors='coerce')
    ).astype('float32')
    d_map['PR_Abs_Rating_Diff'] = np.abs(pd.to_numeric(d_map['PR_Rating_Diff'], errors='coerce')).astype('float32')

    # -------------------------
    # ✅ Convert PR diff -> points using learned sport betas (cached)
    # -------------------------
    d_map = attach_pr_points_edge_cols(
        d_map,
        bq=bq,
        project=project,
        dataset="sharp_data",
        table="sharp_scores_with_features",
        min_rows_per_sport=5000,
        sport_col="Sport",
        pr_diff_col="PR_Rating_Diff",
    ) 
   
    # k_abs + agree flags
    k_abs = (
        pd.to_numeric(d_map.get('Favorite_Market_Spread'),  errors='coerce').abs()
          .combine_first(pd.to_numeric(d_map.get('Underdog_Market_Spread'), errors='coerce').abs())
    ).astype('float32').where(lambda s: s > 0, np.nan)

    d_map['model_fav_vs_market_fav_agree'] = (
        (pd.to_numeric(d_map['Outcome_Model_Spread'], errors='coerce') < 0) ==
        lhs.eq(rhs)
    ).astype('int8')

    d_map['edge_x_k'] = (pd.to_numeric(d_map['Outcome_Spread_Edge'], errors='coerce') * k_abs).astype('float32')
    d_map['mu_x_k']   = (pd.to_numeric(d_map['Outcome_Spread_Edge'], errors='coerce').abs() * k_abs).astype('float32')

    # merge back
    out.drop(columns=UI_EDGE_COLS, inplace=True, errors='ignore')
    out = out.merge(d_map[need_cols + UI_EDGE_COLS], on=need_cols, how='left')

    for c in UI_EDGE_COLS:
        if c not in out.columns:
            out[c] = np.nan

    return out


def compute_diagnostics_vectorized(
    df: pd.DataFrame,
    *,
    bundle=None,
    model=None,
    sport: str = None,
    gcs_bucket: str = None,
    timing_models: dict | None = None,
    hybrid_timing_features: list | None = None,
    hybrid_odds_timing_features: list | None = None,
    sport_aliases: dict | None = None,
    ratings_table_fq: str = "sharplogger.sharp_data.ratings_current",
    project: str = "sharplogger",
    bq=None,  # <— NEW
):
    """
    Returns a diagnostics dataframe with:
      - Tier Δ, Confidence Trend, Line/Model Direction
      - Why Model Likes It (aligned to model-active features only)
      - Timing_Opportunity_Score / Timing_Stage (UI-only)

    Notes:
      - Requires attach_why_model_likes_it(...) to be defined (no-ops gracefully if not).
      - If timing_models not given, attempts GCS load for {spreads, totals, h2h} when sport & gcs_bucket are provided.
    """
    df = df.copy()

    # --- 1) De-dupe to latest per (game, market, outcome, bookmaker) ---
    df = (
        df.sort_values('Snapshot_Timestamp')
          .drop_duplicates(subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'], keep='last')
    )

    # --- 2) Optional: attach ratings/edges (safe if function isn't present) ---
    # inside compute_diagnostics_vectorized(..., bq=None)
    try:
        df = attach_ratings_and_edges_for_diagnostics(
            df=df,
            sport_aliases=sport_aliases or {},
            table_history=ratings_table_fq,
            project=project,
            # ✅ allow using the single “current” timestamp for earlier game starts
            allow_forward_hours=(24*365 if "ratings_current" in str(ratings_table_fq).lower() else 0.0),
            bq=bq,  # ✅ pass the client through
        )
    except Exception as e:
        # don't swallow silently — at least surface it once while debugging
        df["Why Model Likes It"] = f"⚠️ Ratings attach failed: {e}"

    # --- 3) Tier Δ ---
    TIER_ORDER = {'🪙 Low Probability':1, '🤏 Lean':2, '🔥 Strong Indication':3, '🌋 Steam':4}
    for col in ['Confidence Tier', 'First_Tier']:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).str.strip()

    tier_now   = df['Confidence Tier'].map(TIER_ORDER).fillna(0).astype(int)
    tier_first = df['First_Tier'].map(TIER_ORDER).fillna(0).astype(int)
    df['Tier_Change'] = np.where(
        df['First_Tier'] != "",
        np.where(tier_now > tier_first, "↑ " + df['First_Tier'] + " → " + df['Confidence Tier'],
        np.where(tier_now < tier_first, "↓ " + df['First_Tier'] + " → " + df['Confidence Tier'],
                 "↔ No Change")),
        "⚠️ Missing"
    )

    # --- 4) Confidence Trend ---
    prob_now   = pd.to_numeric(df.get('Model Prob'), errors='coerce')
    prob_start = pd.to_numeric(df.get('First_Sharp_Prob'), errors='coerce')
    df['Model Prob Snapshot'] = prob_now
    df['First Prob Snapshot'] = prob_start
    delta = prob_now - prob_start
    df['Confidence Trend'] = np.where(
        prob_now.isna() | prob_start.isna(), "⚠️ Missing",
        np.where(delta >= 0.04,
                 "📈 Trending Up: " + (prob_start*100).round(1).astype(str) + "% → " + (prob_now*100).round(1).astype(str) + "%",
        np.where(delta <= -0.04,
                 "📉 Trending Down: " + (prob_start*100).round(1).astype(str) + "% → " + (prob_now*100).round(1).astype(str) + "%",
                 "↔ Stable: "        + (prob_start*100).round(1).astype(str) + "% → " + (prob_now*100).round(1).astype(str) + "%"))
    )

    # --- 5) Line/Model Direction Alignment ---
    df['Line_Delta'] = pd.to_numeric(df.get('Line_Delta'), errors='coerce')
    df['Line_Support_Sign'] = df.apply(
        lambda r: -1 if str(r.get('Market','')).lower() == 'totals' and str(r.get('Outcome','')).lower() == 'under' else 1,
        axis=1
    )
    df['Line_Support_Direction'] = df['Line_Delta'] * df['Line_Support_Sign']
    df['Line/Model Direction'] = np.select(
        [
            (delta > 0) & (df['Line_Support_Direction'] > 0),
            (delta < 0) & (df['Line_Support_Direction'] < 0),
            (delta > 0) & (df['Line_Support_Direction'] < 0),
            (delta < 0) & (df['Line_Support_Direction'] > 0),
        ],
        ["🟢 Aligned ↑","🔻 Aligned ↓","🔴 Model ↑ / Line ↓","🔴 Model ↓ / Line ↑"],
        default="⚪ Mixed"
    )
    # --- 6) Attach explainer aligned to model-active features ---
    PROB_COL_CANDIDATES = ("Model Prob", "Model_Sharp_Win_Prob", "Pred_Prob", "Model_Prob")
    prob_col = next((c for c in PROB_COL_CANDIDATES if c in df.columns), None)
    has_prob_context = prob_col is not None and pd.to_numeric(df[prob_col], errors="coerce").notna().any()
    
    active_feats = []
    try:
        # Prefer the all-features explainer so attrs are populated even if bundle/model are thin
        df = attach_why_all_features(df, bundle=bundle, model=model, why_rules=WHY_RULES_V3)
        active_feats = list(getattr(df, "attrs", {}).get("active_features_used", []))
    except Exception as e:
        # Only show "no context" if we truly have neither a model nor a prob column
        if not has_prob_context:
            df["Why Model Likes It"] = f"⚠️ Explainer failed: {e}"
            df["Why_Feature_Count"] = 0
        active_feats = []
    
    # Robust fallback if the resolver returned nothing but we do have a prob
    if not active_feats and has_prob_context:
        # try exact training cols
        try:
            active_feats = _resolve_feature_cols_like_training(bundle, model, df) or []
        except Exception:
            active_feats = []
        # seed at least the probability (and a few canonical context features) for the UI
        seed = [prob_col] if prob_col else []
        canon = [c for c in ("Outcome_Model_Spread","Outcome_Market_Spread","Outcome_Spread_Edge",
                             "Outcome_Cover_Prob","PR_Rating_Diff") if c in df.columns]
        # Make unique, preserve order
        seen=set(); active_feats = [x for x in (active_feats + seed + canon) if not (x in seen or seen.add(x))]


   
    def _is_active(col: str) -> bool:
        return col in ACTIVE_FEATS
    
    def _num(row, col):
        v = row.get(col)
        if isinstance(v, str):
            v = v.replace('\u2212','-').replace(',', '').strip()
            if v in ('', '—', '–'):
                return np.nan
        return pd.to_numeric(v, errors='coerce')
    
    # --- put this at the SAME INDENT as _num (module level inside the function) ---
    emit_model_market = True  # always append model/market & PR numbers into WHY
    
    if emit_model_market:
        msgs = df["Why Model Likes It"].astype(str).tolist()
        recs = df.to_dict('records')
        for i, row in enumerate(recs):
            parts = [] if msgs[i] in ("—", "nan") else [msgs[i]]
    
            # spreads-only seasonings
            if str(row.get('Market','')).lower() == 'spreads':
                _mod_spread = _num(row, 'Outcome_Model_Spread')
                _mkt_spread = _num(row, 'Outcome_Market_Spread')
                if pd.notna(_mod_spread) and pd.notna(_mkt_spread):
                    parts.append(f"📐 Model Spread {_mod_spread:+.1f} vs Market {_mkt_spread:+.1f}")
    
                _edge = _num(row, 'Outcome_Spread_Edge')
                if pd.notna(_edge):
                    parts.append(f"🎯 Spread Edge {_edge:+.1f}")
    
                _cov = _num(row, 'Outcome_Cover_Prob')
                if pd.notna(_cov):
                    parts.append(f"🛡️ Cover Prob {_cov:.0%}" + (" ✅" if _cov >= THR["cover_prob_conf"] else ""))
    
                agree_val = row.get('model_fav_vs_market_fav_agree', 0)
                agree_num = pd.to_numeric(agree_val, errors='coerce')
                if pd.notna(agree_num) and int(round(float(agree_num))) == 1:
                    parts.append("🤝 Model & Market Favor Same Team")
    
            # always try to print PR numbers (if ratings merged)
            _pr_team = _num(row, 'PR_Team_Rating')
            _pr_opp  = _num(row, 'PR_Opp_Rating')
            _pr_diff = _num(row, 'PR_Edge_Pts')
            if pd.notna(_pr_team) and pd.notna(_pr_opp):
                if pd.notna(_pr_diff):
                    parts.append(f"📊 Power Ratings {int(round(_pr_team))} vs {int(round(_pr_opp))} (Δ {_pr_diff:+.0f})")
                else:
                    parts.append(f"📊 Power Ratings {int(round(_pr_team))} vs {int(round(_pr_opp))}")
            elif pd.notna(_pr_diff):
                parts.append(f"📊 Power Rating Δ {_pr_diff:+.0f}")
    
            msgs[i] = " · ".join([p for p in parts if p and p != "—"]) or "—"
    
        df["Why Model Likes It"] = msgs
        df["Why_Feature_Count"] = df["Why Model Likes It"].apply(lambda s: 0 if s == "—" else (s.count("·") + 1))
          

    # --- 8) Timing Opportunity (UI-only) ---
     # =====================================
    #  8) TIMING OPPORTUNITY (UI-ONLY) RUN
    # =====================================
    # Normalize market labels
    df['Market'] = df['Market'].astype(str).str.strip().str.lower()

    # Discover magnitude columns directly to avoid name drift
    line_mag_cols = sorted([c for c in df.columns if c.startswith('SharpMove_Magnitude_')])
    odds_mag_cols = sorted([c for c in df.columns if c.startswith('OddsMove_Magnitude_')])

    base_feats_infer = ['Abs_Line_Move_From_Opening', 'Abs_Odds_Move_From_Opening', 'Late_Game_Steam_Flag']
    for c in base_feats_infer:
        if c not in df.columns:
            df[c] = 0.0

    # Broad superset to build X; we will align per-model using saved feature_list
    timing_feature_superset = base_feats_infer + line_mag_cols + odds_mag_cols

    # Ensure outputs exist
    if 'Timing_Opportunity_Score' not in df.columns:
        df['Timing_Opportunity_Score'] = np.nan
    if 'Timing_Stage' not in df.columns:
        df['Timing_Stage'] = '—'

    # --- Load timing models (with their feature lists) ---
    # --- Load timing models (with their feature lists) ---
    timing_models = {}
    if sport and gcs_bucket and ('load_model_from_gcs' in globals()):
        for _m in MARKETS:
            try:
                pl = load_model_from_gcs(
                    sport=sport,
                    market=f"timing_{_m}",
                    bucket_name=gcs_bucket
                ) or {}
    
                # timing payloads should have estimator under "model"
                mdl  = pl.get("model", None)
                cols = pl.get("feature_list", None)
    
                # hard guard: NEVER treat dict as model
                if isinstance(mdl, dict) or (
                    mdl is not None and not (hasattr(mdl, "predict_proba") or hasattr(mdl, "predict"))
                ):
                    mdl = None
    
                timing_models[_m] = {"mdl": mdl, "cols": cols}
    
            except Exception as e:
                # if load fails, just set None so inference falls back to heuristic
                st.warning(f"Timing model load failed for '{_m}': {e}")
                timing_models[_m] = {"mdl": None, "cols": None}
    else:
        timing_models = {m: {"mdl": None, "cols": None} for m in MARKETS}


    def _align_X_for_model(X, mdl, cols_from_payload):
        # Prefer the exact saved train columns
        if cols_from_payload:
            return X.reindex(columns=list(cols_from_payload), fill_value=0.0)
        # Otherwise, try sklearn attributes
        names = getattr(mdl, 'feature_names_in_', None)
        if names is not None:
            return X.reindex(columns=list(names), fill_value=0.0)
        n = getattr(mdl, 'n_features_in_', None)
        if n is not None and X.shape[1] != n:
            keep = timing_feature_superset[:n]
            return X.reindex(columns=keep, fill_value=0.0)
        return X

    def _heuristic_score(subdf: pd.DataFrame) -> pd.Series:
        # Fallback if model is missing: invert normalized move size (lower move => better timing)
        L = subdf['Abs_Line_Move_From_Opening'].clip(lower=0).astype(float)
        O = subdf['Abs_Odds_Move_From_Opening'].clip(lower=0).astype(float)
        L90 = float(np.nanquantile(L, 0.90)) if L.notna().any() else 1.0
        O90 = float(np.nanquantile(O, 0.90)) if O.notna().any() else 1.0
        Lnorm = (L / max(L90, 1e-6)).clip(0, 1)
        Onorm = (O / max(O90, 1e-6)).clip(0, 1)
        penalty = 0.6 * Lnorm + 0.4 * Onorm + 0.25 * subdf['Late_Game_Steam_Flag'].fillna(0.0).astype(float)
        p = (1.0 - penalty).clip(0, 1)
        return p

    # Score per market

    for _m, pack in timing_models.items():
        mask = df['Market'].eq(_m)
        if not mask.any():
            continue
    
        mdl, cols = pack.get("mdl"), pack.get("cols")
    
        # ✅ absolute safety: dict is not a model
        if isinstance(mdl, dict):
            st.warning(
                f"Timing model for '{_m}' is a dict (keys={list(mdl.keys())[:12]}). "
                f"Falling back to heuristic."
            )
            df.loc[mask, 'Timing_Opportunity_Score'] = _heuristic_score(df.loc[mask])
            continue
    
        if mdl is None:
            df.loc[mask, 'Timing_Opportunity_Score'] = _heuristic_score(df.loc[mask])
            continue
    
        # Build X only once we know we have a real estimator
        X = (
            df.loc[mask, timing_feature_superset]
              .apply(pd.to_numeric, errors='coerce')
              .fillna(0.0)
        )
    
        X = _align_X_for_model(X, mdl, cols)
    
        try:
            if hasattr(mdl, "predict_proba"):
                p = mdl.predict_proba(X)[:, 1]
            else:
                p = np.clip(mdl.predict(X), 0, 1)
    
            df.loc[mask, 'Timing_Opportunity_Score'] = pd.to_numeric(p, errors='coerce')
        except Exception as e:
            st.warning(f"Timing model inference failed for '{_m}': {e}")
            df.loc[mask, 'Timing_Opportunity_Score'] = _heuristic_score(df.loc[mask])


    df['Timing_Opportunity_Score'] = (
        pd.to_numeric(df['Timing_Opportunity_Score'], errors='coerce')
        .fillna(0.0).clip(0, 1)
    )

    # --- Market-wise quantile staging (adaptive bins) ---
    def _assign_stage_by_market(_df: pd.DataFrame) -> pd.DataFrame:
        labels = ["🔴 Late / Overmoved", "🟡 Developing", "🟢 Smart Timing"]
        _df['Timing_Stage'] = '🔴 Late / Overmoved'  # default
        for _m in MARKETS:
            msk = _df['Market'].eq(_m) & _df['Timing_Opportunity_Score'].notna()
            if msk.sum() < 10:
                bins = [0.0, 0.40, 0.66, 1.01]  # fallback static thresholds
            else:
                q1, q2 = np.quantile(_df.loc[msk, 'Timing_Opportunity_Score'], [0.33, 0.66])
                q1 = float(np.clip(q1, 0.05, 0.85))
                q2 = float(np.clip(max(q2, q1 + 1e-4), 0.10, 0.95))
                bins = [0.0, q1, q2, 1.01]

            idx = np.digitize(_df.loc[msk, 'Timing_Opportunity_Score'].to_numpy(), bins, right=False) - 1
            idx = np.clip(idx, 0, 2)
            _df.loc[msk, 'Timing_Stage'] = pd.Categorical.from_codes(idx, labels, ordered=True)
        return _df

    df = _assign_stage_by_market(df)

    # Optional quick visibility while debugging
    #st.write("⏱️ Timing stage counts:", df['Timing_Stage'].value_counts(dropna=False).to_dict())

    # --- 9) Return only base + ACTIVE feature columns (so UI stays aligned) ---
    base_cols = [
        'Game_Key','Market','Outcome','Bookmaker',
        'Tier_Change','Confidence Trend','Line/Model Direction',
        'Why Model Likes It','Passes_Gate','Confidence Tier',
        'Model Prob Snapshot','First Prob Snapshot',
        'Model_Confidence_Tier',
        'Timing_Opportunity_Score','Timing_Stage',   
    ]
    active_cols_present = [c for c in active_feats if c in df.columns]
    # build keep_cols then de-dupe while preserving order
    seen = set()
    keep_cols = [c for c in ([c for c in base_cols if c in df.columns] + active_cols_present)
                 if not (c in seen or seen.add(c))]
    
    diagnostics_df = df[keep_cols].rename(columns={'Tier_Change':'Tier Δ'})
    diagnostics_df = diagnostics_df.loc[:, ~pd.Index(diagnostics_df.columns).duplicated()]  # belt & suspenders
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
def _debug_picklability(payload):
    import pickle
    for k, v in payload.items():
        try:
            pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"❌ cannot pickle key={k!r} (type={type(v)}): {e}")

def save_model_to_gcs(
    model,
    sport,
    market,
    bucket_name="sharp-models",
    calibrator=None,
    team_feature_map=None,
    book_reliability_map=None,
    **kwargs  # ignore any extras silently
):
    from io import BytesIO
    import logging
    try:
        import cloudpickle as pickler  # more robust than std pickle
    except ImportError:
        import pickle as pickler
    import pickle as std_pickle  # for optional debug
    from google.cloud import storage

    sport_l  = str(sport).lower()
    market_l = str(market).lower()
    filename = f"sharp_win_model_{sport_l}_{market_l}.pkl"

    # --- build payload exactly like before ---
    if isinstance(model, dict):
        payload = dict(model)  # shallow copy
        if calibrator is not None:
            payload["iso_blend"] = calibrator
        if team_feature_map is not None and "team_feature_map" not in payload:
            payload["team_feature_map"] = team_feature_map
        if book_reliability_map is not None and "book_reliability_map" not in payload:
            payload["book_reliability_map"] = book_reliability_map
    else:
        payload = {
            "model": model,
            "iso_blend": calibrator,
            "team_feature_map": team_feature_map,
            "book_reliability_map": book_reliability_map,
        }

    #--- OPTIONAL: quick picklability debug (uncomment if needed) ---
    for k, v in payload.items():
        try:
            std_pickle.dumps(v)
        except Exception as e:
            print(f"❌ cannot pickle key={k!r} (type={type(v)}): {e}")

    try:
        # --- serialize (cloudpickle if available) ---
        buffer = BytesIO()
        pickler.dump(payload, buffer)
        buffer.seek(0)

        # --- upload to GCS ---
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

        print(f"✅ Saved model artifact to GCS: gs://{bucket_name}/{filename}")
        return {"bucket": bucket_name, "path": filename}
    except Exception as e:
        logging.error(f"❌ Failed to save model to GCS: {e}", exc_info=True)
        raise







def _to_df(x):
    if isinstance(x, pd.DataFrame): return x
    if x is None: return pd.DataFrame()
    try: return pd.DataFrame(x)
    except Exception: return pd.DataFrame()

import io, pickle

class _IsoWrapperShim:
    """Minimal shim so old pickles referencing IsoWrapper can load."""
    def __init__(self, model=None, calibrator=None):
        self.model = model
        self.calibrator = calibrator
    # pickle will restore attributes; no other methods required here

class _RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # be permissive: ANY module with class name 'IsoWrapper' -> shim
        if name == "IsoWrapper":
            return _IsoWrapperShim
        return super().find_class(module, name)

def _safe_loads(b: bytes):
    bio = io.BytesIO(b)
    try:
        return _RenamingUnpickler(bio).load()
    except Exception:
        bio.seek(0)
        return pickle.loads(bio.read())

def load_model_from_gcs(sport, market, bucket_name="sharp-models"):
    import io, pickle, logging, pandas as pd
    from google.cloud import storage
    
    sport_l  = str(sport).lower()
    market_l = str(market).lower()
    fname    = f"sharp_win_model_{sport_l}_{market_l}.pkl"

    client = storage.Client()
    blob   = client.bucket(bucket_name).blob(fname)

    try:
        content = blob.download_as_bytes()
    except Exception as e:
        logging.warning(f"⚠️ No artifact: gs://{bucket_name}/{fname} ({e})")
        return None

    try:
        data = _safe_loads(content)  # your saves don’t need IsoWrapper anymore
        logging.info(f"✅ Loaded artifact: gs://{bucket_name}/{fname}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to unpickle {fname}: {e}")
        return None

    # --- Normalize the dict ---
    if not isinstance(data, dict):
        logging.error(f"Unexpected payload type in {fname}: {type(data)}")
        return None
    # ✅ TIMING MODELS: saved as {"model": estimator, "feature_list": [...]}
    # Return them directly instead of trying to normalize into the ensemble bundle.
    if ("model" in data) and ("feature_list" in data):
        return {
            "model": data.get("model"),
            "feature_list": data.get("feature_list") or [],
            "market": data.get("market"),
            "sport": data.get("sport"),
        }

    iso_blend = data.get("iso_blend") or (data.get("calibrator", {}) or {}).get("iso_blend")
    
    bundle = {
        "model_logloss": data.get("model_logloss"),
        "model_auc":     data.get("model_auc"),
        "best_w":        float(data.get("best_w", 1.0)),
        "feature_cols":  data.get("feature_cols") or [],
    
        # ✅ Make it match predict_blended() “NEW path”
        "calibrator": {"iso_blend": iso_blend},
    
        # ✅ Persist global polarity lock so P(win) is always correct
        "flip_flag": bool(
            data.get("flip_flag", False)
            or data.get("global_flip", False)
            or (data.get("polarity", +1) == -1)
        ),
    
        # keep these (your dashboard uses them)
        "team_feature_map":     data.get("team_feature_map") if isinstance(data.get("team_feature_map"), pd.DataFrame) else pd.DataFrame(),
        "book_reliability_map": data.get("book_reliability_map") if isinstance(data.get("book_reliability_map"), pd.DataFrame) else pd.DataFrame(),
    }
    
    return bundle
   



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

import html, re

# Columns that may intentionally contain safe HTML fragments
_ALLOW_HTML_COLS = {"Confidence Spark"}  # add others if you truly need raw HTML

# Strip invalid/invisible chars that can corrupt tags (incl. U+FFFD, ZWJ/ZWNJ, etc.)
_INVALID_XML_RE = re.compile(
    r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F\uFDD0-\uFDEF\uFFFE\uFFFF]"
)
_INVISIBLE_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFE0E\uFE0F]")

def _strip_bad_chars(s: str) -> str:
    s = _INVALID_XML_RE.sub("", s)
    s = _INVISIBLE_RE.sub("", s)
    # guard against the replacement char itself
    s = s.replace("\uFFFD", "")
    return s

def _clean_text(x: str) -> str:
    """Plain text only (for headers and most cells)."""
    if not isinstance(x, str): 
        return x
    return html.escape(_strip_bad_chars(str(x)), quote=True)

def _clean_cell(val, colname: str):
    """Cells: escape by default; allow safe HTML only for whitelisted columns."""
    if isinstance(val, (int, float)) or val is None:
        return val
    s = _strip_bad_chars(str(val))
    if colname not in _ALLOW_HTML_COLS:
        return html.escape(s, quote=True)
    # Very small, conservative allow-list: no table tags, no <script>, no <style>.
    if re.search(r"</?(?:table|thead|tbody|tr|th|td|script|style)\b", s, flags=re.I):
        # If someone tried to sneak table tags in, fall back to escaped text
        return html.escape(s, quote=True)
    return s

def _looks_malformed(html_str: str) -> bool:
    """Heuristics to detect broken tags before injecting."""
    return any(
        bad in html_str
        for bad in ("<td�", "\uFFFD", "<td\uFFFD", "<th\uFFFD", "<tr\uFFFD")
    )


def render_scanner_tab(label, sport_key, container, force_reload=False):

    if st.session_state.get("pause_refresh_user", False) or st.session_state.get("pause_refresh_lock", False):
        st.info("⏸️ Auto-refresh paused")
        return
    with container:
        st.subheader(f"📡 Scanning {label} Sharp Signals")
        # Inject JS patch to log InvalidCharacterError with class/id context (browser console)
        st.markdown("""
        <script>
        (function(){
          function wrap(obj, method){
            const orig = obj && obj[method];
            if (!orig) return;
            obj[method] = function(...args){
              try { return orig.apply(this, args); }
              catch(e){
                if (e && e.name === 'InvalidCharacterError'){
                  console.error('[InvalidCharacterError]', method, 'args=', args);
                  try {
                    const el = (this instanceof Element) ? this : null;
                    if (el) console.error('element=', el, 'outerHTML=', el.outerHTML);
                  } catch(_) {}
                }
                throw e;
              }
            };
          }
          const DTP = DOMTokenList && DOMTokenList.prototype;
          if (DTP){
            ['add','remove','toggle','replace'].forEach(m => wrap(DTP, m));
          }
          const EP = Element && Element.prototype;
          if (EP){
            wrap(EP, 'setAttribute');
          }
          const DP = Document && Document.prototype;
          if (DP){
            ['querySelector','querySelectorAll','getElementById','getElementsByClassName']
              .forEach(m => wrap(DP, m));
          }
        })();
        </script>
        """, unsafe_allow_html=True)
        # === 0) History (used later for trends)
        # --- Build a visible title + safe key EARLY so it's always defined ---
        _summary_title_text = f"📊 Sharp vs Rec Book Summary Table – {label}"  # emoji OK in visible text
        _key_base = re.sub(r'[^a-z0-9_-]+', '-', _summary_title_text.lower()).strip('-')
        if not _key_base or _key_base[0].isdigit():
            _key_base = f"k-{_key_base}"
        _key_hash = hashlib.blake2b(_summary_title_text.encode('utf-8'), digest_size=4).hexdigest()
        _title_key = f"{_key_base}-{_key_hash}"   # ← use everywhere for keys/ids
    
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
            if "Model Prob" not in df_summary_base.columns:
                if "Model_Sharp_Win_Prob" in df_summary_base.columns:
                    df_summary_base["Model Prob"] = pd.to_numeric(
                        df_summary_base["Model_Sharp_Win_Prob"], errors="coerce"
                    )
                else:
                    df_summary_base["Model Prob"] = np.nan
            
            # --- Anchor model prob toward book implied prob (bet-friendly) ---
            p_final = pd.to_numeric(df_summary_base["Model Prob"], errors="coerce")
            
            p_imp = pd.Series(
                calc_implied_prob(df_summary_base.get("Odds_Price", np.nan)),
                index=df_summary_base.index,
            )
            
            mask = p_final.notna() & p_imp.notna()
            
            df_summary_base["Model Prob (Bet)"] = np.nan
            df_summary_base.loc[mask, "Model Prob (Bet)"] = anchor_to_implied(
                p_final.loc[mask],
                p_imp.loc[mask],
                k=0.45,
            )

            
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
            # ✅ Keep per-row probability as the primary “Model Prob”
            df_summary_base['Model Prob'] = base_prob
            
            # ✅ Keep sharp-average as its own column (use this for spark/trend if desired)
            df_summary_base['Model Prob (Sharp Avg)'] = sharp_avg
            
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
            # === 8) STEP 5: per-book diagnostics built on latest-per-book sharp rows from df_pre, then attach ===
            diag_source = (
                df_pre
                  .sort_values('Snapshot_Timestamp')
                  .drop_duplicates(subset=['Game_Key','Market','Outcome','Bookmaker'], keep='last')
            )
            diag_source = diag_source[diag_source['Bookmaker'].str.lower().isin(SHARP_BOOKS)].copy()
            
            if diag_source.empty:
                st.warning("⚠️ No sharp book rows available for diagnostics.")
                for col in ['Tier Δ','Line/Model Direction','Why Model Likes It','Timing_Opportunity_Score','Timing_Stage']:
                    df_summary_base[col] = "⚠️ Missing"
            else:
                # normalize keys before merge
                for df_ in (df_summary_base, diag_source):
                    for k in ['Game_Key','Market','Outcome','Bookmaker']:
                        df_[k] = df_[k].astype(str).str.strip().str.lower()
            
                # Helper: map Market labels to canonical keys used in trained_models
                def _canonical_market(m: str) -> str:
                    m = str(m).lower().strip()
                    if m in ('spread','spreads','ats'): return 'spreads'
                    if m in ('total','totals','o/u','ou'): return 'totals'
                    if m in ('moneyline','ml','h2h','headtohead'): return 'h2h'
                    return m
            
                # Helper: resolve bundle/model for a market key
                def _resolve_bundle_model_for_market(trained_by_market: dict, mkt_norm: str):
                    bundle = (trained_by_market or {}).get(mkt_norm)
                    mdl = None
                    if isinstance(bundle, dict):
                        mdl = bundle.get("model_logloss") or bundle.get("model_auc") or bundle.get("model")
                    return bundle, mdl
            
                # Figure out which markets to loop over:
                # prefer explicit trained_models keys; otherwise derive from diag_source
                if 'trained_models' in locals() and trained_models:
                    markets_present = list(trained_models.keys())
                else:
                    markets_present = (
                        diag_source['Market'].astype(str).str.lower().str.strip().map(_canonical_market).unique().tolist()
                    )
            
                diagnostics_chunks = []
                for market in markets_present:
                    market_norm = _canonical_market(market)
                    bundle, model = _resolve_bundle_model_for_market(trained_models if 'trained_models' in locals() else {}, market_norm)
                    
                    # rows for this market
                    diag_rows = diag_source[diag_source['Market'].astype(str).str.lower().str.strip().map(_canonical_market) == market_norm]
                    if diag_rows.empty:
                        continue
           
                    ratings_current_df = enrich_power_for_training_lowmem(
                        df=diag_rows[['Sport','Home_Team_Norm','Away_Team_Norm','Game_Start']].drop_duplicates(),
                        bq=bq_client,
                        sport_aliases=sport_aliases,
                        table_history="sharplogger.sharp_data.ratings_current",
                        pad_days=365,
                        rating_lag_hours=10_000.0,  # forces always fallback to base (not useful)
                    )
                    diag_rows = diag_rows.merge(
                        ratings_current_df[['Sport','Home_Team_Norm','Away_Team_Norm',
                                    'Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff']],
                        on=['Sport','Home_Team_Norm','Away_Team_Norm'],
                        how='left'
                    )
                    
                    # compute diagnostics for this market with its matching model
                    chunk = compute_diagnostics_vectorized(
                        diag_rows,
                        bundle=bundle,
                        model=model,
                        sport=label,                # you used `label` earlier for loading
                        gcs_bucket=GCS_BUCKET, 
                        bq=bq_client,
                        ratings_table_fq="sharplogger.sharp_data.ratings_current",
                        sport_aliases=sport_aliases,
                        hybrid_timing_features=hybrid_timing_features if 'hybrid_timing_features' in globals() else None,
                        hybrid_odds_timing_features=hybrid_odds_timing_features if 'hybrid_odds_timing_features' in globals() else None,
                    )
                    diagnostics_chunks.append(chunk)
                diagnostics_chunks = [c.loc[:, ~pd.Index(c.columns).duplicated()] for c in diagnostics_chunks]
    
                diagnostics_df = (
                    pd.concat(diagnostics_chunks, ignore_index=True)
                    if diagnostics_chunks else pd.DataFrame(columns=[
                        'Game_Key','Market','Outcome','Bookmaker','Tier Δ',
                        'Line/Model Direction','Why Model Likes It',
                        'Timing_Opportunity_Score','Timing_Stage'
                    ])
                )


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
            # === 9) (Optional) merge team features AFTER the dedupe ===
            if team_feature_map is not None and not team_feature_map.empty:
                # normalize keys
                df_summary_base["Sport"]  = df_summary_base["Sport"].astype(str).str.upper().str.strip()
                df_summary_base["Market"] = df_summary_base["Market"].astype(str).str.lower().str.strip()
                df_summary_base["Team"]   = df_summary_base["Outcome"].astype(str).str.lower().str.strip()
            
                team_feature_map = team_feature_map.copy()
                team_feature_map["Sport"]  = team_feature_map["Sport"].astype(str).str.upper().str.strip()
                team_feature_map["Market"] = team_feature_map["Market"].astype(str).str.lower().str.strip()
                team_feature_map["Team"]   = team_feature_map["Team"].astype(str).str.lower().str.strip()
            
                # merge at the intended grain
                df_summary_base = df_summary_base.merge(
                    team_feature_map,
                    on=["Sport", "Market", "Team"],
                    how="left",
                    validate="many_to_one",
                )

            
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
            # ✅ Always guarantee timing columns exist on df_summary_base (UI-only)
            
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
            
            
            # === 🔍 Diagnostic: Check for duplicate Game × Market × Outcome
            # === 🔍 Diagnostic: Check for duplicate Game × Market × Outcome
            # === 🔍 Diagnostic: Check for duplicate Game × Market × Outcome
            
    
    
            # Optional: final sort if needed
            #summary_df.sort_values(by=['Game_Start', 'Matchup', 'Market'], inplace=True)
            
       
            # === Build Market + Date Filters
            selected_market = st.selectbox(
                f"📊 Filter {label} by Market",
                ["All"] + sorted(summary_df['Market'].dropna().unique()),
                key=f"{_title_key}-market"
            )
            
            selected_date = st.selectbox(
                f"📅 Filter {label} by Date",
                ["All"] + sorted(summary_df['Event_Date_Only'].dropna().unique()),
                key=f"{_title_key}-date"
            )
            
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
            # Round Rec Line and Sharp Line to 1 decimal
            summary_grouped['Rec Line'] = summary_grouped['Rec Line'].round(1)
            summary_grouped['Sharp Line'] = summary_grouped['Sharp Line'].round(1)
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
            # --- Build a visible title + safe key EARLY so it's always defined ---
           
            
            
            # === Final Output
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
            .custom-table tr:nth-child(even) { background-color: #2d3748; }
            .custom-table tr:hover { background-color: #4b5563; }
            </style>
            """, unsafe_allow_html=True)
            
         
            # Keep only columns that actually exist
            cols = [c for c in view_cols if c in summary_grouped.columns]
            missing = sorted(set(view_cols) - set(cols))
            if missing:
                st.warning(f"Missing columns ignored: {missing}")
            
            table_df = summary_grouped[cols].copy()
            
            # Sanitize headers (plain text only)
            table_df.columns = [_clean_text(str(c)) for c in table_df.columns]   # // NEW: shorter, same effect
            
            # Sanitize each cell (escape by default; allow limited HTML in whitelisted cols)
            for c in table_df.columns:
                colname_for_policy = c  # header after cleaning
                table_df[c] = table_df[c].map(lambda v: _clean_cell(v, colname_for_policy))
            
           # Build HTML
            table_html = table_df.to_html(classes="custom-table", index=False, escape=False)
            
            # Drop any stray replacement/invisible chars *from the whole HTML string* (including tags)
            table_html = (
                table_html
                .replace("\uFFFD", "")          # replacement char
                .replace("\u2028", "")          # line sep
                .replace("\u2029", "")          # paragraph sep
            )
            
            # Optional: if your environment ever re-encodes, force a clean UTF-8 pass
            table_html = table_html.encode("utf-8", "ignore").decode("utf-8", "ignore")
            
            # Final guard: refuse to inject if any td/th/tr tag name is corrupted
            if re.search(r"<(?:td|th|tr)\uFFFD", table_html, flags=re.I):
                st.error("Detected malformed <td>/<th>/<tr> tag. Showing safe table instead.")
                st.table(summary_grouped[cols].head(300))
            else:
                safe_id = f"tbl-{_title_key}"
                st.markdown(
                    f"<div id='{html.escape(safe_id)}' class='scrollable-table-container'>{table_html}</div>",
                    unsafe_allow_html=True
                )


              
            st.success("✅ Finished rendering sharp picks table.")
            st.caption(f"Showing {len(table_df)} rows")

            pass
    # === 2. Render Live Odds Snapshot Table
       
    with st.container():  # or a dedicated tab/expander if you want
        st.subheader(f"📊 Live Odds Snapshot – {label} (Value @ Odds + Limit)")
    
 
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
                    mkey = market.get("key")
                    if mkey not in ["h2h", "spreads", "totals"]:
                        continue
    
                    for outcome in market.get("outcomes", []):
                        # odds APIs usually provide:
                        # - 'price' (American odds) for ALL markets
                        # - 'point' ONLY for spreads/totals
                        odds_price = outcome.get("price", None)
                        point_val  = outcome.get("point", None) if mkey != "h2h" else None  # no point for h2h
    
                        odds_rows.append({
                            "Game": game_name,
                            "Market": mkey,
                            "Outcome": outcome.get("name"),
                            "Bookmaker": normalize_book_name(book.get("key", ""), book.get("key", "")),
                            "Value": point_val,                  # spread/totals number; NaN for h2h
                            "Odds_Price": odds_price,            # American odds (e.g., -115, +105)
                            "Limit": outcome.get("bet_limit", 0),
                            "Game_Start": game_start,
                        })
    
        df_odds_raw = pd.DataFrame(odds_rows)
    
        if not df_odds_raw.empty:
            # Build a friendly "Value @ Odds (Limit)" string
            def _fmt(row):
                val  = row.get("Value")
                odds = row.get("Odds_Price")
                lim  = row.get("Limit")
    
                parts = []
                # value only for spreads/totals
                if pd.notnull(val):
                    parts.append(f"{float(val):.1f}")  # -3.5 style
                # odds for all markets
                if pd.notnull(odds):
                    # render odds as integer if possible
                    try:
                        parts.append(f"@ {int(odds)}")
                    except Exception:
                        parts.append(f"@ {odds}")
    
                display = " ".join(parts).strip()
    
                # append limit if present
                if pd.notnull(lim):
                    try:
                        display = f"{display} ({int(lim)})" if display else f"({int(lim)})"
                    except Exception:
                        display = f"{display} ({lim})" if display else f"({lim})"
    
                # fallbacks so we don't show 'nan'
                if not display:
                    # for pure h2h with no limit somehow, show odds or blank
                    if pd.notnull(odds):
                        try:
                            display = f"{int(odds)}"
                        except Exception:
                            display = f"{odds}"
                    else:
                        display = ""
    
                return display
    
            df_odds_raw["Value@Odds(Limit)"] = df_odds_raw.apply(_fmt, axis=1)
    
            # Localize to EST
            eastern = pytz_timezone("US/Eastern")
            df_odds_raw["Date + Time (EST)"] = df_odds_raw["Game_Start"].apply(
                lambda x: x.tz_convert(eastern).strftime("%Y-%m-%d %I:%M %p") if pd.notnull(x) and x.tzinfo
                else pd.to_datetime(x).tz_localize("UTC").tz_convert(eastern).strftime("%Y-%m-%d %I:%M %p") if pd.notnull(x)
                else ""
            )
    
            # Pivot into Bookmaker columns (human-readable)
            df_display = (
                df_odds_raw.pivot_table(
                    index=["Date + Time (EST)", "Game", "Market", "Outcome"],
                    columns="Bookmaker",
                    values="Value@Odds(Limit)",
                    aggfunc="first"
                )
                .rename_axis(columns=None)
                .reset_index()
            )
    
            # (Optional) keep a raw, analytics-friendly table for later use
            # with separate numeric columns Value, Odds_Price, Limit
            # df_odds_raw[['Game','Market','Outcome','Bookmaker','Value','Odds_Price','Limit','Date + Time (EST)']]
    
            # Render as HTML
            table_html_2 = df_display.to_html(classes="custom-table", index=False, escape=False)
            st.markdown(f"<div class='scrollable-table-container'>{table_html_2}</div>", unsafe_allow_html=True)
            st.success(f"✅ Live odds snapshot rendered — {len(df_display)} rows.")
        else:
            st.info("No live odds available right now.")


def fetch_scores_and_backtest(*args, **kwargs):
    print("⚠️ fetch_scores_and_backtest() is deprecated in UI and will be handled by Cloud Scheduler.")
    return pd.DataFrame()

    
def render_power_ranking_tab(tab, sport_label: str, sport_key_api: str, bq_client=None, show_edges: bool = False):
    """
    Streamlit tab to display per-team power ratings with recent trend deltas and (optional) model-vs-market edges.
    Uses the Team column (falls back to Team_Norm ONLY if Team is absent in that table).
    """
    from google.cloud import bigquery
    import pandas as pd
    import numpy as np
    import streamlit as st
    from datetime import date, timedelta

    if bq_client is None:
        bq_client = bigquery.Client(project="sharplogger", location="us")

    # ---------- helpers ----------
    def pick_team(colnames: set[str]) -> str | None:
        return "Team" if "Team" in colnames else ("Team_Norm" if "Team_Norm" in colnames else None)

    def pick_rating(colnames: set[str]) -> str | None:
        for c in ("Rating", "Power_Rating", "Elo", "PR", "PowerIndex"):
            if c in colnames:
                return c
        return None

    def pick_ts(colnames: set[str]) -> str | None:
        for c in ("Updated_At", "Last_Update", "Snapshot_Timestamp"):
            if c in colnames:
                return c
        return None

    def has_method(colnames: set[str]) -> bool:
        return "Method" in colnames

    # -----------------------------

    with tab:
        st.subheader(f"🏆 Power Ratings — {sport_label}")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            days_back = st.slider("History Window (days)", min_value=14, max_value=365, value=90, step=7)
        with col_right:
            trend_win = st.select_slider("Trend Window", options=[7, 14, 30], value=14)

        end_d = date.today()
        start_d = end_d - timedelta(days=days_back)

        cur_fq = "sharplogger.sharp_data.ratings_current"
        his_fq = "sharplogger.sharp_data.ratings_history"

        cur_cols = {f.name for f in bq_client.get_table(cur_fq).schema}
        his_cols = {f.name for f in bq_client.get_table(his_fq).schema}

        team_cur = pick_team(cur_cols)
        team_his = pick_team(his_cols)
        if team_cur is None:
            st.error(f"`{cur_fq}` has neither Team nor Team_Norm.")
            return
        if team_his is None:
            st.error(f"`{his_fq}` has neither Team nor Team_Norm.")
            return

        rating_cur = pick_rating(cur_cols)
        rating_his = pick_rating(his_cols)
        if rating_cur is None:
            st.error(f"`{cur_fq}` has no rating column (tried Rating/Power_Rating/Elo/PR/PowerIndex).")
            return
        if rating_his is None:
            st.error(f"`{his_fq}` has no rating column (tried Rating/Power_Rating/Elo/PR/PowerIndex).")
            return

        ts_cur = pick_ts(cur_cols)
        ts_his = pick_ts(his_cols)

        # ✅ Method support
        cur_has_method = has_method(cur_cols)
        his_has_method = has_method(his_cols)

        # If either table has Method, treat Method as required for correctness.
        use_method = cur_has_method or his_has_method

        chosen_method = None
        if use_method:
            # Pull available methods for this sport from CURRENT (preferred)
            sql_methods = f"""
            SELECT DISTINCT Method
            FROM `{cur_fq}`
            WHERE UPPER(Sport) = @sport
              AND Method IS NOT NULL
            ORDER BY Method
            """
            methods = bq_client.query(
                sql_methods,
                job_config=bigquery.QueryJobConfig(query_parameters=[
                    bigquery.ScalarQueryParameter("sport", "STRING", sport_label.upper())
                ]),
            ).to_dataframe()

            method_list = methods["Method"].dropna().astype(str).tolist()

            if len(method_list) == 0:
                # fallback: try history
                sql_methods_h = f"""
                SELECT DISTINCT Method
                FROM `{his_fq}`
                WHERE UPPER(Sport) = @sport
                  AND Method IS NOT NULL
                ORDER BY Method
                """
                methods_h = bq_client.query(
                    sql_methods_h,
                    job_config=bigquery.QueryJobConfig(query_parameters=[
                        bigquery.ScalarQueryParameter("sport", "STRING", sport_label.upper())
                    ]),
                ).to_dataframe()
                method_list = methods_h["Method"].dropna().astype(str).tolist()

            if len(method_list) == 0:
                # No methods available even though schema has Method — continue without filter,
                # but warn because it can mix.
                st.warning("Ratings tables have `Method` column but no methods found for this sport — results may mix.")
                chosen_method = None
                use_method = False
            elif len(method_list) == 1:
                chosen_method = method_list[0]
                st.caption(f"Method: `{chosen_method}`")
            else:
                # default: pick something stable (prefer elo_kalman if present, else first)
                default = "elo_kalman" if "elo_kalman" in method_list else method_list[0]
                chosen_method = st.selectbox("Rating Method", options=method_list, index=method_list.index(default))

        # ---------- SQL expressions ----------
        team_cur_expr = f"LOWER({team_cur})"
        team_his_expr = f"LOWER({team_his})"
        rating_cur_expr = f"CAST({rating_cur} AS FLOAT64)"
        rating_his_expr = f"CAST({rating_his} AS FLOAT64)"
        ts_cur_expr = ts_cur if ts_cur else "CAST(NULL AS TIMESTAMP)"
        ts_his_expr = ts_his if ts_his else "CAST(NULL AS TIMESTAMP)"

        hist_date_clause = f"AND DATE({ts_his}) BETWEEN @start_d AND @end_d" if ts_his else ""

        method_clause_cur = "AND Method = @method" if (use_method and cur_has_method and chosen_method) else ""
        method_clause_his = "AND Method = @method" if (use_method and his_has_method and chosen_method) else ""

        # --- Current snapshot ---
        sql_current = f"""
        SELECT
          UPPER(Sport) AS Sport,
          {team_cur_expr} AS Team,
          {rating_cur_expr} AS Rating,
          {ts_cur_expr} AS Snapshot_Timestamp
          {", Method" if cur_has_method else ""}
        FROM `{cur_fq}`
        WHERE UPPER(Sport) = @sport
          AND {team_cur} IS NOT NULL
          {method_clause_cur}
        """
        cur_params = [bigquery.ScalarQueryParameter("sport", "STRING", sport_label.upper())]
        if use_method and chosen_method and cur_has_method:
            cur_params.append(bigquery.ScalarQueryParameter("method", "STRING", chosen_method))

        cur_df = bq_client.query(
            sql_current,
            job_config=bigquery.QueryJobConfig(query_parameters=cur_params),
        ).to_dataframe()

        # --- History window ---
        sql_hist = f"""
        SELECT
          UPPER(Sport) AS Sport,
          {team_his_expr} AS Team,
          {rating_his_expr} AS Rating,
          {ts_his_expr} AS Snapshot_Timestamp
          {", Method" if his_has_method else ""}
        FROM `{his_fq}`
        WHERE UPPER(Sport) = @sport
          AND {team_his} IS NOT NULL
          {method_clause_his}
          {(' ' + hist_date_clause) if hist_date_clause else ''}
        """
        hist_params = [bigquery.ScalarQueryParameter("sport", "STRING", sport_label.upper())]
        if ts_his:
            hist_params += [
                bigquery.ScalarQueryParameter("start_d", "DATE", start_d),
                bigquery.ScalarQueryParameter("end_d", "DATE", end_d),
            ]
        if use_method and chosen_method and his_has_method:
            hist_params.append(bigquery.ScalarQueryParameter("method", "STRING", chosen_method))

        hist_df = bq_client.query(
            sql_hist,
            job_config=bigquery.QueryJobConfig(query_parameters=hist_params),
        ).to_dataframe()

        # ---------- types / early exit ----------
        for d in (cur_df, hist_df):
            if not d.empty:
                d["Snapshot_Timestamp"] = pd.to_datetime(d["Snapshot_Timestamp"], utc=True, errors="coerce")
                if "Method" in d.columns:
                    d["Method"] = d["Method"].astype(str)

        if cur_df.empty and hist_df.empty:
            st.warning("No ratings found for this sport in the selected window.")
            return

        # ✅ IMPORTANT: “today” should come from CURRENT table, not “latest of history+current”
        # History is only for deltas.
        cur_df = cur_df.dropna(subset=["Team", "Rating"]).copy()
        cur_df["Team"] = cur_df["Team"].astype(str)
        cur_df = cur_df[~cur_df["Team"].str.contains(r"\bleague\b", case=False, na=False)]
        if cur_df.empty:
            st.warning("No current team-level ratings after filtering.")
            return

        # If multiple rows per team exist in current (possible), take latest timestamp
        cur_df = cur_df.sort_values(["Team", "Snapshot_Timestamp"]).groupby("Team", as_index=False).tail(1)
        cur_df.rename(columns={"Rating": "Rating_Today"}, inplace=True)

        # ---------- deltas from history ----------
        hist_df = hist_df.dropna(subset=["Team", "Rating", "Snapshot_Timestamp"]).copy()
        hist_df["Team"] = hist_df["Team"].astype(str)
        hist_df = hist_df[~hist_df["Team"].str.contains(r"\bleague\b", case=False, na=False)]
        hist_df.sort_values(["Team", "Snapshot_Timestamp"], inplace=True)

        def _value_asof(group: pd.DataFrame, cutoff_days: int):
            g = group.dropna(subset=["Snapshot_Timestamp"]).sort_values("Snapshot_Timestamp")
            if g.empty:
                return np.nan
            ts = g["Snapshot_Timestamp"].values
            vals = g["Rating"].values
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=cutoff_days)
            idx = np.searchsorted(ts, cutoff.to_datetime64(), side="right") - 1
            return float(vals[idx]) if idx >= 0 else np.nan

        deltas = []
        # compute deltas relative to current rating (from cur_df)
        hist_groups = {t: g for t, g in hist_df.groupby("Team")} if not hist_df.empty else {}
        for _, row in cur_df.iterrows():
            team = row["Team"]
            r_today = float(row["Rating_Today"])
            g = hist_groups.get(team)
            if g is None:
                deltas.append((team, np.nan, np.nan, np.nan))
            else:
                deltas.append((
                    team,
                    r_today - _value_asof(g, 7),
                    r_today - _value_asof(g, 14),
                    r_today - _value_asof(g, 30),
                ))

        deltas_df = pd.DataFrame(deltas, columns=["Team", "Delta_7d", "Delta_14d", "Delta_30d"])

        summary = (
            cur_df[["Team", "Rating_Today"]]
            .merge(deltas_df, on="Team", how="left")
            .sort_values("Rating_Today", ascending=False)
            .reset_index(drop=True)
        )
        summary["Rank"] = (np.arange(len(summary)) + 1).astype(int)

        pretty = summary[["Rank", "Team", "Rating_Today", "Delta_7d", "Delta_14d", "Delta_30d"]].copy()
        pretty.rename(columns={
            "Rating_Today": "Rating",
            "Delta_7d": "Delta 7d",
            "Delta_14d": "Delta 14d",
            "Delta_30d": "Delta 30d",
        }, inplace=True)

        for c in ["Rating", "Delta 7d", "Delta 14d", "Delta 30d"]:
            pretty[c] = pd.to_numeric(pretty[c], errors="coerce").round(2)
        pretty["Rating"] = pd.to_numeric(pretty["Rating"], errors="coerce").round(1)

        st.markdown("### Current Ratings & Trend")
        if use_method and chosen_method:
            st.caption(f"Filtered to Method: `{chosen_method}`")
        st.dataframe(pretty, use_container_width=True)

        # ---------- history chart ----------
        with st.expander("📈 Team Rating History"):
            team_sel = st.selectbox("Team", options=list(summary["Team"]), index=0)
            g = hist_df[hist_df["Team"] == team_sel].sort_values("Snapshot_Timestamp").copy()
            if g.empty:
                st.info("No history points in the selected window for this team.")
            else:
                try:
                    g["Date"] = g["Snapshot_Timestamp"].dt.tz_convert("America/New_York").dt.date
                except TypeError:
                    g["Date"] = g["Snapshot_Timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.date
                st.line_chart(g.set_index("Date")["Rating"])

        # ---------- optional edges (unchanged) ----------
        if show_edges:
            st.markdown("---")
            st.markdown("### Model vs Market Edges (Spreads)")
            st.caption("Powered by attach_ratings_and_edges_for_diagnostics(...) using current ratings.")
            try:
                sq = """
                SELECT
                  Sport, Market, Home_Team_Norm, Away_Team_Norm, Outcome, Outcome_Norm,
                  Value, Bookmaker, Snapshot_Timestamp, Game_Start
                FROM `sharplogger.sharp_data.sharp_scores_with_features`
                WHERE UPPER(Sport) = @sport
                  AND Market = 'spreads'
                  AND DATETIME(Game_Start) >= DATETIME(@now_utc)
                LIMIT 3000
                """
                now_utc = pd.Timestamp.now(tz="UTC").to_pydatetime()
                df_scores = bq_client.query(
                    sq,
                    job_config=bigquery.QueryJobConfig(query_parameters=[
                        bigquery.ScalarQueryParameter("sport", "STRING", sport_label.upper()),
                        bigquery.ScalarQueryParameter("now_utc", "DATETIME", now_utc),
                    ]),
                ).to_dataframe()

                edges = attach_ratings_and_edges_for_diagnostics(
                    df_scores,
                    sport_aliases=SPORT_ALIASES,
                    table_history="sharplogger.sharp_data.ratings_current",
                    project="sharplogger",
                    pad_days=30,
                    allow_forward_hours=0.0,
                    bq=bq_client,
                )
                keep_cols = [
                    "Sport","Home_Team_Norm","Away_Team_Norm","Outcome_Norm","Value","Game_Start",
                    "PR_Team_Rating","PR_Opp_Rating","PR_Rating_Diff",
                    "Outcome_Model_Spread","Outcome_Market_Spread","Outcome_Spread_Edge",
                    "Outcome_Cover_Prob","model_fav_vs_market_fav_agree","edge_x_k","mu_x_k"
                ]
                show = edges[[c for c in keep_cols if c in edges.columns]].copy()
                st.dataframe(show, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render edges: {e}")


def render_sharp_signal_analysis_tab(tab, sport_label, sport_key_api, start_date=None, end_date=None):

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
                FROM 'sharplogger.sharp_data.scores_with_features'
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
        prob_bins = [0, 0.50, 0.55, 0.70, 1.0]
        bin_labels = ["✅ Low", "⭐ Lean", "🔥 Strong Indication", "🔥 Steam"]

        
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



# --- SIMPLE RENDER FLOW (no UI sanitization/purge) ---
from streamlit_situations_tab import render_current_situations_tab
# ============================ SIDEBAR + TABS UI ================================
sport = st.sidebar.radio(
    "Select a League",
    ["General", "NFL", "NCAAF", "NBA", "MLB", "CFL", "WNBA","NCAAB"],
    key="sport_radio",
)

st.sidebar.markdown("### ⚙️ Controls")


st.sidebar.checkbox(
    "⏸️ Pause Auto Refresh",
    key="pause_refresh_user",
    disabled=st.session_state.get("is_training", False),
)

force_reload = st.sidebar.button("🔁 Force Reload", key="force_reload_btn")

# --- Optional: Track scanner checkboxes by sport (logical names)
scanner_flags = {
    "NFL": "run_nfl_scanner",
    "NCAAF": "run_ncaaf_scanner",
    "NBA": "run_nba_scanner",
    "MLB": "run_mlb_scanner",
    "CFL": "run_cfl_scanner",
    "WNBA": "run_wnba_scanner",
    "NCAAB": "run_ncaab_scanner"
}
scanner_widget_keys = {k: f"{v}" for k, v in scanner_flags.items()}

# === GENERAL PAGE ===
if sport == "General":
    st.title("🎯 Sharp Scanner Dashboard")
    st.write("Use the sidebar to select a league and begin scanning or training models.")

# === LEAGUE PAGES ===
else:
    st.title(f"🏟️ {sport} Sharp Scanner")

    # Scanner toggle (still visible)
    scanner_key = scanner_widget_keys[sport]
    run_scanner = st.checkbox(
        f"Run {sport} Scanner",
        key=scanner_key,
        value=True,
    )

    label = sport
    sport_key = SPORTS[sport]

    # Market picker
    market_choice = st.sidebar.selectbox(
        "Train which market?",
        ["All", "spreads", "h2h", "totals"],
        key=f"train_market_choice_{sport}",
    )

    # --- If training, pause scanner + stop page from doing anything expensive ---
    if st.session_state.get("is_training", False) or st.session_state.get("pause_refresh_lock", False):
        st.info("⏳ Training in progress… scanner/refresh is paused.")
        st.stop()

    # ✅ Single train button (unique key per sport + choice)
    train_key = f"train::{sport}::{market_choice}"

    if st.button(f"📈 Train {sport} Sharp Model", key=train_key):
        st.session_state["is_training"] = True
        st.session_state["pause_refresh_lock"] = True  # ✅ pause scanner/refresh during training

        with st.spinner(f"Training {sport} model(s)… this may take several minutes"):
            try:
                # If training "All", optionally train timing model once first
                if market_choice == "All":
                    train_timing_opportunity_model(sport=label)
                    markets_to_train = ("h2h", "spreads", "totals")
                else:
                    markets_to_train = (market_choice,)

                for mkt in markets_to_train:
                    st.write(f"— Training market: **{mkt}**")
                    train_with_champion_wrapper(
                        sport=label,
                        market=mkt,
                        bucket_name=GCS_BUCKET,
                        log_func=st.write,
                    )

                st.success("✅ Training complete")

            except Exception as e:
                st.exception(e)

            finally:
                st.session_state["pause_refresh_lock"] = False
                st.session_state["is_training"] = False

        st.stop()  # don’t continue into scanner logic in this run

    # -----------------------------
    # Scanner run (only if not paused)
    # -----------------------------
    if run_scanner and not st.session_state.get("pause_refresh_lock", False):
        # call your scanner logic here
        # e.g., render_scanner_tab(...) or detect_and_render(...)
        pass
    else:
        if st.session_state.get("pause_refresh_lock", False):
            st.info("⏸️ Scanner paused during training.")

            
    # Prevent multiple scanners from running
    conflicting = [
        k for k, v in scanner_widget_keys.items()
        if k != sport and bool(st.session_state.get(v, False)) is True
    ]

    if conflicting:
        st.warning(f"⚠️ Please disable other scanners before running {sport}: {conflicting}")
    elif run_scanner:
        scan_tab, analysis_tab, power_tab, situation_tab = st.tabs(
            ["📡 Live Scanner", "📈 Backtest Analysis", "🏆 Power Ratings", "📚 Situation DB"]
        )       

        with scan_tab:
            render_scanner_tab(label=label, sport_key=sport_key, container=scan_tab)

        with analysis_tab:
            render_sharp_signal_analysis_tab(
                tab=analysis_tab, sport_label=label, sport_key_api=sport_key
            )

        with power_tab:
            client = bigquery.Client(project="sharplogger", location="us")
            render_power_ranking_tab(
                tab=power_tab,
                sport_label=label,
                sport_key_api=sport_key,
                bq_client=client,
                show_edges=False,
            )
        with situation_tab:
            try:
                render_current_situations_tab(selected_sport=sport)
            except Exception as e:
                st.error(f"Situations tab error: {e}")






