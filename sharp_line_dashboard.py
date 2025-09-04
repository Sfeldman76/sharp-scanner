
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

import xgboost as xgb
from xgboost import XGBClassifier

from google.oauth2 import service_account
from google.cloud import storage, bigquery, bigquery_storage_v1
import pandas_gbq
from pandas_gbq import to_gbq
import google.api_core.exceptions
from google.cloud import bigquery, bigquery_storage

from pandas.api.types import (
    is_bool_dtype, is_numeric_dtype, is_categorical_dtype, is_datetime64_any_dtype,
    is_period_dtype, is_interval_dtype, is_object_dtype
)
              
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
  
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier
# put near your imports (only once)
from sklearn.base import is_classifier as sk_is_classifier
import sys, inspect, xgboost, sklearn

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import log_loss, make_scorer

from sklearn.feature_selection import VarianceThreshold
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

import os

def get_vcpus(env_var: str = "XGB_SEARCH_NJOBS") -> int:
    """Prefer ENV override; else fall back to container vCPU count."""
    val = os.getenv(env_var, "")
    try:
        n = int(val) if val != "" else (os.cpu_count() or 1)
    except Exception:
        n = os.cpu_count() or 1
    return max(1, n)

VCPUS = get_vcpus()  # e.g., 8 if XGB_SEARCH_NJOBS=8, else os.cpu_count()


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

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


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
        import numpy as np
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


# --- CACHED CLIENT (resource-level) ---
@st.cache_resource
def get_bq() -> bigquery.Client:
    return bigquery.Client(project="sharplogger")

                     
def _iso(ts) -> str:
    return pd.to_datetime(ts, utc=True).isoformat()


@st.cache_data(show_spinner=False)
# ===========================
# WHY MODEL LIKES IT (v2)
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
    times: np.ndarray,
    y: np.ndarray,
    pct_holdout: float | None = None,
    min_train_games: int = 25,
    min_hold_games: int = 8,
    ensure_label_diversity: bool = True,
):
    """
    Time‑forward holdout by last % of GROUPS (Game_Key).
    If pct_holdout is None, choose per sport via SPORT_HOLDOUT_PCT.
    Returns (train_idx, hold_idx) as row indices.
    """
    SPORT_HOLDOUT_PCT = {
        "NFL": 0.10, "NCAAF": 0.10, "NBA": 0.18, "WNBA": 0.10,
        "NHL": 0.18, "MLB": 0.15, "MLS": 0.18, "CFL": 0.10, "DEFAULT": 0.20,
    }
    if pct_holdout is None:
        key = (sport or "DEFAULT").upper()
        pct_holdout = float(SPORT_HOLDOUT_PCT.get(key, SPORT_HOLDOUT_PCT["DEFAULT"]))

    # Build group-time meta (one row per group)
    meta_rows = pd.DataFrame({
        "row_idx": np.arange(len(groups)),
        "group":   groups.astype(str),
        "time":    pd.to_datetime(times, utc=True, errors="coerce"),
        "y":       y.astype(int),
    }).dropna(subset=["time"])  # drop NaT rows if any

    gmeta = (meta_rows.groupby("group")
             .agg(start=("time", "min"), end=("time", "max"))
             .sort_values("start")
             .reset_index())

    n_groups = len(gmeta)
    if n_groups == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    n_hold_groups = max(min_hold_games, int(np.ceil(n_groups * pct_holdout)))
    n_hold_groups = min(n_groups - max(1, min_train_games), n_hold_groups)
    n_hold_groups = max(min_hold_games, n_hold_groups)

    # Take last n_hold_groups by start time as holdout
    hold_groups = gmeta["group"].iloc[-n_hold_groups:].to_numpy()
    train_groups = gmeta["group"].iloc[: n_groups - n_hold_groups].to_numpy()

    # Map back to row indices
    all_groups = meta_rows["group"].to_numpy()
    hold_idx = np.flatnonzero(np.isin(all_groups, hold_groups))
    train_idx = np.flatnonzero(np.isin(all_groups, train_groups))

    if ensure_label_diversity:
        # If holdout has only one class, expand boundary until it has both or we run out
        def _has_both(idx):
            if idx.size == 0: return False
            return np.unique(meta_rows.loc[idx, "y"]).size >= 2

        k = n_hold_groups
        while not _has_both(hold_idx) and (n_groups - k) >= max(1, min_train_games):
            k += 1
            hold_groups = gmeta["group"].iloc[-k:].to_numpy()
            train_groups = gmeta["group"].iloc[: n_groups - k].to_numpy()
            hold_idx = np.flatnonzero(np.isin(all_groups, hold_groups))
            train_idx = np.flatnonzero(np.isin(all_groups, train_groups))

    return train_idx, hold_idx


def sharp_row_weights(df, a_sharp=0.8, b_limit=0.15, c_liq=0.10, d_steam=0.10):
    import numpy as np, pandas as pd
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


def shap_stability_select(
    model_proto,
    X: pd.DataFrame,
    y: np.ndarray,
    folds,                       # iterable of (train_idx, valid_idx, *rest)
    *,
    topk_per_fold: int = 60,     # K used to count "presence" within each fold
    min_presence: float = 0.6,   # must appear in top-K in ≥60% of folds
    max_keep: int | None = None, # hard cap after sorting by avg_abs_shap (optional)
    sample_per_fold: int = 4000, # SHAP computed on a sample of train fold rows
    random_state: int = 42,
    must_keep: list[str] = None  # always-keep features (unioned at the end)
):
    """
    Returns:
      selected_features: list[str]
      summary_df: pd.DataFrame with avg_rank, presence, avg_abs_shap
    """
    try:
        import shap
    except Exception as e:
        raise RuntimeError("Install shap to use shap_stability_select") from e

    feat = X.columns.tolist()
    nF = len(feat)
    if nF == 0:
        return [], pd.DataFrame()

    rng = np.random.RandomState(random_state)
    counts_in_topk = pd.Series(0, index=feat, dtype=int)
    shap_sum = pd.Series(0.0, index=feat, dtype=float)
    fold_rank_frames = []

    # iterate folds defensively (supports (tr,va) or (tr,va,*,*))
    for fold in folds:
        tr_idx, va_idx = fold[:2]
        Xtr, ytr = X.iloc[tr_idx], y[tr_idx]

        # fit a fresh clone on the fold’s train
        mdl = clone(model_proto)
        mdl.fit(Xtr, ytr)

        # modest sample for speed/memory
        if len(Xtr) > sample_per_fold:
            sample = Xtr.iloc[rng.choice(len(Xtr), size=sample_per_fold, replace=False)]
        else:
            sample = Xtr

        # fast tree explainer; stable for xgboost
        explainer = shap.TreeExplainer(mdl, feature_perturbation="tree_path_dependent")
        sval = explainer.shap_values(sample)

        # binary case: ensure 2D array
        if isinstance(sval, list):
            # some versions return [class0, class1]; take positive class
            sval = sval[1]
        sval = np.asarray(sval)
        shap_abs_mean = pd.Series(np.abs(sval).mean(axis=0), index=feat)

        # accumulate
        shap_sum = shap_sum.add(shap_abs_mean, fill_value=0.0)

        # per-fold ranking (1 = most important)
        r = shap_abs_mean.rank(ascending=False, method="average")
        fold_rank_frames.append(r)

        # count presence in top-K this fold
        topk = shap_abs_mean.sort_values(ascending=False).head(min(topk_per_fold, nF)).index
        counts_in_topk.loc[topk] += 1

    # aggregate across folds
    n_folds = len(fold_rank_frames)
    avg_rank = pd.concat(fold_rank_frames, axis=1).mean(axis=1)          # lower is better
    presence = counts_in_topk / max(n_folds, 1)
    avg_abs_shap = (shap_sum / max(n_folds, 1))

    # selection rule:
    # 1) keep anything with presence ≥ min_presence
    # 2) also keep the best-by-avg_abs_shap up to topk_per_fold (for safety)
    keep_presence = presence.index[presence >= float(min_presence)].tolist()
    keep_top_global = avg_abs_shap.sort_values(ascending=False).head(topk_per_fold).index.tolist()

    # stable order: by descending avg_abs_shap
    selected = list(dict.fromkeys(keep_top_global + keep_presence))
    selected = sorted(selected, key=lambda c: (-avg_abs_shap[c], avg_rank[c]))

    # union must-keep
    if must_keep:
        for c in must_keep:
            if c in feat and c not in selected:
                selected.append(c)

    # optional cap
    if max_keep is not None and len(selected) > max_keep:
        # respect ordering by SHAP, but ensure must_keep are kept
        base = [c for c in selected if (not must_keep or c not in must_keep)]
        head = base[: max_keep - (len(must_keep or []))]
        selected = head + [c for c in (must_keep or []) if c in feat and c not in head]

    summary = (
        pd.DataFrame({
            "avg_rank": avg_rank,
            "presence": presence,
            "avg_abs_shap": avg_abs_shap
        })
        .loc[selected]
        .sort_values(["avg_abs_shap","presence"], ascending=[False, False])
    )
    return selected, summary


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


# --- Active feature resolver (kept from your snippet; safe if already defined) ---
def _resolve_active_features(bundle, model, df_like):
    cand = None
    if isinstance(bundle, dict):
        for k in ("feature_names", "features", "training_features"):
            if k in bundle and bundle[k]:
                cand = list(dict.fromkeys([str(c) for c in bundle[k] if c is not None]))
                break
    if cand is None and hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            if booster is not None and getattr(booster, "feature_names", None):
                cand = list(dict.fromkeys([str(c) for c in booster.feature_names]))
        except Exception:
            pass
    if cand is None:
        try:
            cols = _resolve_feature_cols_like_training(bundle, model, df_like)  # your helper if available
            cand = list(dict.fromkeys([str(c) for c in cols if c is not None]))
        except Exception:
            pass
    if cand is None:
        cand = [c for c in df_like.columns if c not in ("y","label","target")]
    return cand

# --- Numeric coercion (kept from your snippet; safe if already defined) ---
def _coerce_numeric_inplace(df, cols, fill=0.0):
    UI_NUM_COLS = {
        'Outcome_Model_Spread','Outcome_Market_Spread','Outcome_Spread_Edge','Outcome_Cover_Prob',
        'PR_Team_Rating','PR_Opp_Rating','PR_Rating_Diff','edge_x_k','mu_x_k'
    }
    for c in cols:
        # create the column if missing
        if c not in df.columns:
            df[c] = (np.nan if c in UI_NUM_COLS else fill)
            continue

        s = df[c]
        # avoid categorical assignment issues
        if pd.api.types.is_categorical_dtype(s.dtype):
            s = s.astype(object)

        vals = pd.to_numeric(s, errors="coerce")
        # UI columns should preserve NaN (so the UI can show blanks)
        if c in UI_NUM_COLS:
            df[c] = vals
        else:
            df[c] = vals.fillna(fill)
# --- Helper to read first available value across aliases safely ---
def _rv(row, *names, default=0.0):
    for n in names:
        if n in row:
            try:
                return float(row[n])
            except Exception:
                return default
    return default

# --- Thresholds (tune lightly per sport if you want) ---
# --- Thresholds (tune per sport if needed) ---
THR = dict(
    line_mag_big=0.30,
    late_share_high=0.50,
    urgent_share_high=0.20,
    entropy_concentrated=1.20,
    corr_confirm=0.35,
    odds_overmove_ratio=1.10,
    pct_from_open_big=0.10,
    pr_diff_meaningful=4.0,
    cover_prob_conf=0.51,
    # 🆕 ATS thresholds
    ats_rate_strong=0.55,          # e.g., Elo/EB-style ATS rate >= 58% is notable
    ats_margin_meaningful=2,     # avg cover margin (pts) viewed as meaningful
    ats_roll_decay_hot=0.4,        # rolling/decay-weighted margin showing momentum
)

WHY_RULES_V2 = [
    # 🔹 Core sharp/book pressure
    dict(requires_any=["Book_lift_x_Magnitude"],
         check=lambda r: _rv(r,"Book_lift_x_Magnitude") > 0.0,
         msg="🏦 Book Lift Supports Move"),
    dict(requires_any=["Book_lift_x_PROB_SHIFT"],
         check=lambda r: _rv(r,"Book_lift_x_PROB_SHIFT") > 0.0,
         msg="📈 Book Lift Aligned with Prob Shift"),
    dict(requires_any=["Sharp_Limit_Total"],
         check=lambda r: _rv(r,"Sharp_Limit_Total") >= 10000,
         msg="💼 High Sharp Limits"),
    dict(requires_any=["Is_Reinforced_MultiMarket"],
         check=lambda r: _rv(r,"Is_Reinforced_MultiMarket") > 0.0,
         msg="📊 Multi-Market Reinforcement"),
    dict(requires_any=["Market_Leader"],
         check=lambda r: _rv(r,"Market_Leader") > 0.0,
         msg="🏆 Market Leader Led the Move"),

    # 🔹 Market response / mispricing
    dict(requires_any=["Line_Moved_Toward_Team"],
         check=lambda r: _rv(r,"Line_Moved_Toward_Team") > 0.0,
         msg="🧲 Line Moved Toward This Team"),
    dict(requires_any=["Market_Mispricing"],
         check=lambda r: _rv(r,"Market_Mispricing") > 0.0,
         msg="💸 Market Mispricing Detected"),
    dict(requires_any=["Pct_Line_Move_From_Opening"],
         check=lambda r: _rv(r,"Pct_Line_Move_From_Opening") >= THR["pct_from_open_big"],
         msg="📈 Significant Move From Open"),

    # 🔁 Reversal / overmove
    dict(requires_any=["Value_Reversal_Flag"],
         check=lambda r: _rv(r,"Value_Reversal_Flag") > 0.0,
         msg="🔄 Value Reversal"),
    dict(requires_any=["Odds_Reversal_Flag"],
         check=lambda r: _rv(r,"Odds_Reversal_Flag") > 0.0,
         msg="📉 Odds Reversal"),
    dict(requires_any=["Potential_Overmove_Flag"],
         check=lambda r: _rv(r,"Potential_Overmove_Flag") > 0.0,
         msg="📊 Possible Line Overmove"),
    dict(requires_any=["Potential_Odds_Overmove_Flag"],
         check=lambda r: _rv(r,"Potential_Odds_Overmove_Flag") > 0.0,
         msg="🎯 Possible Odds Overmove"),

    # 🚧 Resistance/levels
    dict(requires_any=["Line_Resistance_Crossed_Count"],
         check=lambda r: _rv(r,"Line_Resistance_Crossed_Count") >= 1,
         msg="🪵 Crossed Key Resistance Levels"),
    dict(requires_any=["SharpMove_Resistance_Break"],
         check=lambda r: _rv(r,"SharpMove_Resistance_Break") > 0.0,
         msg="🪓 Resistance Broken by Sharp Move"),

    # 🧠 Book reliability / liquidity microstructure
    dict(requires_any=["Book_Reliability_Lift"],
         check=lambda r: _rv(r,"Book_Reliability_Lift") > 0.0,
         msg="✅ Reliable Book Confirms"),
    dict(requires_any=["SmallBook_Heavy_Liquidity_Flag","SmallBook_Limit_Skew_Flag","SmallBook_Limit_Skew"],
         check=lambda r: _rv(r,"SmallBook_Heavy_Liquidity_Flag")
                         + _rv(r,"SmallBook_Limit_Skew_Flag")
                         + max(0.0, _rv(r,"SmallBook_Limit_Skew")) > 0.0,
         msg="💧 Liquidity/Limit Skew Signals Pressure"),

    # ⏱️ Timing aggregates
    dict(requires_any=["Line_TotalMag","Sharp_Line_Magnitude"],
         check=lambda r: max(_rv(r,"Line_TotalMag"), _rv(r,"Sharp_Line_Magnitude")) >= THR["line_mag_big"],
         msg="📏 Strong Timing Magnitude"),
    dict(requires_any=["Line_LateShare"],
         check=lambda r: _rv(r,"Line_LateShare") >= THR["late_share_high"],
         msg="🌙 Late-Phase Dominant"),
    dict(requires_any=["Line_UrgentShare"],
         check=lambda r: _rv(r,"Line_UrgentShare") >= THR["urgent_share_high"],
         msg="⏱️ Urgent Push Detected"),
    dict(requires_any=["Line_MaxBinMag"],
         check=lambda r: _rv(r,"Line_MaxBinMag") > 0.0,
         msg="💥 Sharp Timing Spike"),
    dict(requires_any=["Line_Entropy","Hybrid_Timing_Entropy_Line"],
         check=lambda r: 0.0 < _rv(r,"Line_Entropy","Hybrid_Timing_Entropy_Line") <= THR["entropy_concentrated"],
         msg="🎯 Concentrated Timing"),
    dict(requires_any=["Timing_Corr_Line_Odds"],
         check=lambda r: _rv(r,"Timing_Corr_Line_Odds") >= THR["corr_confirm"],
         msg="🔗 Odds Confirm Line Timing"),
    dict(requires_any=["LineOddsMag_Ratio"],
         check=lambda r: _rv(r,"LineOddsMag_Ratio") >= THR["odds_overmove_ratio"],
         msg="⚖️ Line > Odds Magnitude"),
    dict(requires_any=["Late_Game_Steam_Flag"],
         check=lambda r: _rv(r,"Late_Game_Steam_Flag") > 0.0,
         msg="🌙 Late Game Steam"),
        # 🧠 Power Rating Agreement (H2H)
    dict(requires_any=["PR_Model_Agree_H2H_Flag"],
         check=lambda r: _rv(r,"PR_Model_Agree_H2H_Flag") > 0.0,
         msg="🧠 Power Ratings Agree with Model"),
    
    dict(requires_any=["PR_Market_Agree_H2H_Flag"],
         check=lambda r: _rv(r,"PR_Market_Agree_H2H_Flag") > 0.0,
         msg="📊 Power Ratings Agree with Market"),

    # 📐 Model & pricing agreement
    dict(requires_any=["model_fav_vs_market_fav_agree"],
         check=lambda r: _rv(r,"model_fav_vs_market_fav_agree") > 0.0,
         msg="🧭 Model & Market Agree"),
    dict(requires_any=["Outcome_Cover_Prob"],
         check=lambda r: _rv(r,"Outcome_Cover_Prob") >= THR["cover_prob_conf"],
         msg="🔮 Strong Cover Probability"),

    # 📊 Power ratings / totals context
    dict(requires_any=["PR_Rating_Diff","PR_Abs_Rating_Diff"],
         check=lambda r: max(abs(_rv(r,"PR_Rating_Diff")), _rv(r,"PR_Abs_Rating_Diff")) >= THR["pr_diff_meaningful"],
         msg="📈 Meaningful Power-Rating Edge"),
    dict(requires_any=["TOT_Mispricing","TOT_Proj_Total_Baseline"],
         check=lambda r: _rv(r,"TOT_Mispricing") > 0.0,
         msg="🧮 Totals Mispricing"),

    # 🔧 Trained cross-terms
    dict(requires_any=["Book_Reliability_x_Magnitude"],
         check=lambda r: _rv(r,"Book_Reliability_x_Magnitude") > 0.0,
         msg="✅ Reliable Books Driving Magnitude"),
    dict(requires_any=["Book_Reliability_x_PROB_SHIFT"],
         check=lambda r: _rv(r,"Book_Reliability_x_PROB_SHIFT") > 0.0,
         msg="✅ Reliable Books Driving Prob Shift"),

    # 🆕 ATS trend signals
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
]
def attach_why_model_likes_it(df_in: pd.DataFrame, bundle, model) -> pd.DataFrame:
    """
    Builds a human-readable 'Why Model Likes It' column using the UPDATED feature set
    (timing aggregates, microstructure/resistance, mispricing, etc.). Only fires rules
    when their required columns are present AND used by the model.
    """
    df = df_in.copy()

    # 1) Resolve which features the model actually used
    active = _resolve_active_features(bundle, model, df)
    active_set = set(map(str, active))

    # 2) Decide which rule columns are available (either active or simply present in df)
    needed = set()
    for rule in WHY_RULES_V2:
        for req in rule.get("requires", []):
            if (req in active_set) or (req in df.columns):
                needed.add(req)
        for req in rule.get("requires_any", []):
            if (req in active_set) or (req in df.columns):
                needed.add(req)

    # 3) Coerce only the needed columns to numeric (prevents categorical setitem errors)
    _coerce_numeric_inplace(df, list(needed), fill=0.0)

    # 4) Evaluate rules row-wise
    msgs = []
    for _, row in df.iterrows():
        reasons = []
        for rule in WHY_RULES_V2:
            req_all  = rule.get("requires", [])
            req_any  = rule.get("requires_any", [])
            # Must have all reqs (if any)
            if req_all and not all(((r in active_set) or (r in df.columns)) for r in req_all):
                continue
            # Must have at least one of req_any (if any)
            if req_any and not any(((r in active_set) or (r in df.columns)) for r in req_any):
                continue
            # Try the check
            try:
                if rule["check"](row):
                    reasons.append(rule["msg"])
            except Exception:
                # swallow row-level issues
                continue
        msgs.append(" · ".join(reasons) if reasons else "—")

    df["Why Model Likes It"] = msgs
    df["Why_Feature_Count"] = df["Why Model Likes It"].apply(lambda s: 0 if s == "—" else (s.count("·") + 1))
    df.attrs["active_features_used"] = active  # expose for downstream gating
    return df


# --- TRAINING ENRICHMENT (LEAK-SAFE) ---


def enrich_power_for_training_lowmem(
    df: pd.DataFrame,
    sport_aliases: dict | None = None,           # optional
    bq=None,                                     # pass your bigquery.Client
    table_history: str = "sharplogger.sharp_data.ratings_history",
    pad_days: int = 10,
    allow_forward_hours: float = 0.0,            # 0 = strict backward-only
    project: str | None = None,                  # kept for signature parity
) -> pd.DataFrame:
    """
    Enrich df with Home_Power_Rating, Away_Power_Rating, Power_Rating_Diff
    by looking up team ratings within a small time window.

    - Works with BOTH ratings_history (filters by Method) and ratings_current (no Method column).
    - Requires: pass a google.cloud.bigquery.Client as `bq`.
    """
    sport_aliases = sport_aliases or {}

    if df.empty:
        return df.copy()
    if bq is None:
        raise ValueError("BigQuery client `bq` is None — pass your bigquery.Client (e.g., bq=bq_client).")

    out = df.copy()

    # --- normalize inputs (strings + time) ---
    out['Sport'] = out['Sport'].astype(str).str.upper()
    out['Home_Team_Norm'] = out['Home_Team_Norm'].astype(str).str.lower().str.strip()
    out['Away_Team_Norm'] = out['Away_Team_Norm'].astype(str).str.lower().str.strip()
    out['Game_Start'] = pd.to_datetime(out['Game_Start'], utc=True, errors='coerce')

    # sport alias → canon
    canon = {}
    for k, v in (sport_aliases or {}).items():
        if isinstance(v, list):
            for a in v:
                canon[str(a).upper()] = str(k).upper()
        else:
            canon[str(k).upper()] = str(v).upper()
    out['Sport'] = out['Sport'].map(lambda s: canon.get(s, s))

    # assume one sport per call
    sport_canon = out['Sport'].iloc[0]
    out = out[out['Sport'] == sport_canon].copy()
    if out.empty:
        return df.copy()

    # teams + window for fetch
    teams = pd.Index(out['Home_Team_Norm']).union(out['Away_Team_Norm']).unique().tolist()
    gmin, gmax = out['Game_Start'].min(), out['Game_Start'].max()
    pad = pd.Timedelta(days=pad_days)
    start_iso = pd.to_datetime(gmin - pad, utc=True).isoformat()
    end_iso   = pd.to_datetime(gmax + pad, utc=True).isoformat()

    # ---------- tiny in-function cache ----------
    if not hasattr(enrich_power_for_training_lowmem, "_ratings_cache"):
        enrich_power_for_training_lowmem._ratings_cache = {}
    _CACHE = enrich_power_for_training_lowmem._ratings_cache

    # Preferred method per sport (history only)
    PREFERRED_METHOD = {
        "MLB":   "poisson",
        "NFL":   "elo_kalman",
        "NCAAF": "elo_kalman",
        "NBA":   "elo_kalman",
        "WNBA":  "elo_kalman",
        "CFL":   "elo_kalman",
        "NCAAB": "ridge_massey",
    }

    def _norm_team_series(s: pd.Series) -> pd.Series:
        return (s.astype(str).str.lower().str.strip()
                .str.replace(r'\s+', ' ', regex=True)
                .str.replace('.', '', regex=False)
                .str.replace('&', 'and', regex=False)
                .str.replace('-', ' ', regex=False))

    def _aliases_for(s: str) -> list[str]:
        s = (s or "").upper()
        al = sport_aliases.get(s, []) if sport_aliases else []
        if not isinstance(al, list):
            al = [al]
        return sorted({s, *[str(x).upper() for x in al]})

    def fetch_training_ratings_window_cached(
        sport: str,
        start_iso: str,
        end_iso: str,
        table_history: str,
        project: str | None = None,
    ) -> pd.DataFrame:
        key = (sport.upper(), start_iso, end_iso, table_history, tuple(sorted(teams)))
        if key in _CACHE:
            return _CACHE[key].copy()

        # ratings_current usually has no Method column → skip that filter
        is_current = 'ratings_current' in str(table_history).lower()
        method = None if is_current else PREFERRED_METHOD.get(sport.upper())

        # Teams are already normalized; bind params
        from google.cloud import bigquery
        sql = f"""
        SELECT
          UPPER(CAST(Sport AS STRING))        AS Sport,
          LOWER(TRIM(CAST(Team AS STRING)))   AS Team_Norm,
          CAST(Rating AS FLOAT64)             AS Power_Rating,
          TIMESTAMP(Updated_At)               AS AsOfTS
        FROM `{table_history}`
        WHERE UPPER(CAST(Sport AS STRING)) IN UNNEST(@aliases)
          AND LOWER(TRIM(CAST(Team AS STRING))) IN UNNEST(@teams)
          AND TIMESTAMP(Updated_At) >= @start
          AND TIMESTAMP(Updated_At) <= @end
          {"AND LOWER(CAST(Method AS STRING)) = @method" if method else ""}
        """

        params = [
            bigquery.ArrayQueryParameter("aliases", "STRING", _aliases_for(sport)),
            bigquery.ArrayQueryParameter("teams", "STRING", teams),
            bigquery.ScalarQueryParameter("start", "TIMESTAMP", pd.Timestamp(start_iso).to_pydatetime()),
            bigquery.ScalarQueryParameter("end",   "TIMESTAMP", pd.Timestamp(end_iso).to_pydatetime()),
        ]
        if method:
            params.append(bigquery.ScalarQueryParameter("method", "STRING", method.lower()))

        df_r = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()

        if df_r.empty:
            _CACHE[key] = df_r
            return df_r

        # Normalize team, ensure types
        df_r['Team_Norm']    = _norm_team_series(df_r['Team_Norm'])
        df_r['Power_Rating'] = pd.to_numeric(df_r['Power_Rating'], errors='coerce')
        df_r['AsOfTS']       = pd.to_datetime(df_r['AsOfTS'], utc=True, errors='coerce')
        df_r = df_r.dropna(subset=['Team_Norm','Power_Rating','AsOfTS'])

        # Keep only this sport (post-alias map)
        df_r = df_r[df_r['Sport'].str.upper() == sport.upper()]
        _CACHE[key] = df_r
        return df_r.copy()

    # === Pull ratings (cached) and filter to these teams ===
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

    ratings = ratings.copy()
    ratings['Team_Norm'] = _norm_team_series(ratings['Team_Norm'])
    ratings = ratings[ratings['Team_Norm'].isin(teams)]
    if ratings.empty:
        base = np.float32(1500.0)
        out['Home_Power_Rating'] = base
        out['Away_Power_Rating'] = base
        out['Power_Rating_Diff'] = np.float32(0.0)
        return out

    # compact arrays per team: times + values (sorted)
    team_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for team, g in ratings.groupby('Team_Norm', sort=False):
        g = g.sort_values('AsOfTS')
        team_series[team] = (
            g['AsOfTS'].to_numpy(dtype='datetime64[ns]'),
            g['Power_Rating'].to_numpy(dtype=np.float32),
        )

    # === Vectorized assignment per team (low-alloc) ===
    base = np.float32(1500.0)
    out['Home_Power_Rating'] = base
    out['Away_Power_Rating'] = base

    allow_ns = np.int64(round(float(allow_forward_hours) * 3600.0 * 1e9))
    gs_ns = out['Game_Start'].values.astype('datetime64[ns]')

    # Home side
    for team, (t_arr, r_arr) in team_series.items():
        mask = (out['Home_Team_Norm'].values == team)
        if mask.any():
            ts = gs_ns[mask].astype('int64') + allow_ns
            ts = ts.astype('datetime64[ns]')
            idx = np.searchsorted(t_arr, ts, side='right') - 1
            valid = idx >= 0
            vals = np.full(idx.shape, base, dtype=np.float32)
            if valid.any():
                vals[valid] = r_arr[idx[valid]]
            out.loc[mask, 'Home_Power_Rating'] = vals

    # Away side
    for team, (t_arr, r_arr) in team_series.items():
        mask = (out['Away_Team_Norm'].values == team)
        if mask.any():
            ts = gs_ns[mask].astype('int64') + allow_ns
            ts = ts.astype('datetime64[ns]')
            idx = np.searchsorted(t_arr, ts, side='right') - 1
            valid = idx >= 0
            vals = np.full(idx.shape, base, dtype=np.float32)
            if valid.any():
                vals[valid] = r_arr[idx[valid]]
            out.loc[mask, 'Away_Power_Rating'] = vals

    out['Power_Rating_Diff'] = (
        pd.to_numeric(out['Home_Power_Rating'], errors='coerce')
        - pd.to_numeric(out['Away_Power_Rating'], errors='coerce')
    ).astype('float32')

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


def enrich_and_grade_for_training(
    df_spread_rows: pd.DataFrame,
    bq,                                   # required BigQuery client
    sport_aliases: dict,
    value_col: str = "Value",
    outcome_col: str = "Outcome_Norm",
    pad_days: int = 30,
    allow_forward_hours: float = 0.0,
    table_history: str = "sharplogger.sharp_data.ratings_history",  # override to ratings_current for live
    project: str | None = None,
) -> pd.DataFrame:
    """
    Attach leakage-safe power ratings + consensus market k, then compute
    model-vs-market spreads/edges and cover probs mapped to each outcome row.
    Expects spreads only.
    """
    if df_spread_rows.empty:
        return df_spread_rows.copy()

    # 1) Ratings (as-of) for the unique games in this batch
    base = enrich_power_for_training_lowmem(
        df=df_spread_rows[['Sport','Home_Team_Norm','Away_Team_Norm','Game_Start']].drop_duplicates(),
        bq=bq,
        sport_aliases=sport_aliases,
        table_history=table_history,
        pad_days=pad_days,
        allow_forward_hours=allow_forward_hours,
        project=project,
    )

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
        'Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff'
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
    use_monotone: bool = False,      # kept for compatibility; unused unless you wire constraints
) -> tuple[dict, dict, dict]:
    """
    Returns (base_kwargs, params_ll, params_auc) for XGBClassifier randomized searches.

    Design:
    - No eval_metric and no scale_pos_weight here.
    - Base kwargs are safe, CPU-friendly, and consistent.
    - Two search spaces:
        * params_ll  → logloss-oriented (probability quality)
        * params_auc → AUC-oriented (ranking lift)
    - "Small leagues" get a slightly hotter learning rate and lighter models.

    Tips:
    - If you want a better prior, set after this call:
          base_kwargs["base_score"] = float(np.clip(np.mean(y_train), 1e-4, 1-1e-4))
    - Set n_estimators on the estimator you pass to CV, not here.
    """
    s = str(sport).upper()
    _ = float(scale_pos_weight)  # kept only for API compatibility
    n_jobs = int(n_jobs)

    # You can tweak this set as you wish
    SMALL_LEAGUES = {"WNBA", "CFL"}  # add/remove: {"NCAAB-W", ...}
    SMALL = (s in SMALL_LEAGUES)

    # ---------------- Base kwargs (single, consistent block) ----------------
    base_kwargs = dict(
        objective="binary:logistic",
        tree_method="hist",
        predictor="cpu_predictor",
        grow_policy="lossguide",
        max_bin=256,                # solid CPU default
        sampling_method="uniform",
        max_delta_step=0.0,         # no update clamp
        reg_lambda=2.0,             # moderate L2
        min_child_weight=1.0,       # allow splits; grids will widen
        n_jobs=n_jobs,
        random_state=42,
        importance_type="total_gain",
        # intentionally: no eval_metric, no scale_pos_weight
    )

    # If you ever want monotone constraints, wire them here.
    # (Kept off by default; supplying a dummy/all-zeros map is usually pointless.)
    if use_monotone and features:
        # Example skeleton (disabled by default):
        # base_kwargs["monotone_constraints"] = "(" + ",".join("0" for _ in features) + ")"
        pass

    # ---------------- Search spaces ----------------
    if SMALL:
        # Small leagues — allow exploration but keep it sane
        params_ll = {
            "max_depth":        randint(3, 7),              # {3,4,5,6}
            "max_leaves":       randint(64, 128),           # {64..127}
            "learning_rate":    loguniform(3e-2, 1.0e-1),   # 0.03–0.10
            "subsample":        uniform(0.70, 0.30),        # 0.70–1.00
            "colsample_bytree": uniform(0.70, 0.30),        # 0.70–1.00
            "colsample_bynode": uniform(0.60, 0.30),        # 0.60–0.90
            "min_child_weight": randint(1, 5),              # {1..4}
            "gamma":            uniform(0.00, 0.30),        # 0.00–0.30
            "reg_lambda":       loguniform(0.5, 6.0),       # 0.5–6
            "reg_alpha":        loguniform(1e-4, 5e-1),     # 0.0001–0.5
        }
        params_auc = {
            "max_depth":        randint(3, 8),              # {3..7}
            "max_leaves":       randint(64, 128),           # {64..127}
            "learning_rate":    loguniform(3e-2, 1.0e-1),   # 0.03–0.10
            "subsample":        uniform(0.75, 0.25),        # 0.75–1.00
            "colsample_bytree": uniform(0.70, 0.30),        # 0.70–1.00
            "colsample_bynode": uniform(0.60, 0.30),        # 0.60–0.90
            "min_child_weight": randint(1, 6),              # {1..5}
            "gamma":            uniform(0.00, 0.30),        # 0.00–0.30
            "reg_lambda":       loguniform(0.5, 6.0),       # 0.5–6
            "reg_alpha":        loguniform(1e-4, 5e-1),     # 0.0001–0.5
        }
    else:
        # Big leagues — more capacity & diversity
        # LogLoss-oriented (probability quality)
        params_ll = {
            "max_depth":        randint(3, 6),              # {3,4,5}
            "max_leaves":       randint(64, 128),           # {64..127}
            "learning_rate":    loguniform(0.03, 0.08),     # 0.03–0.08
            "subsample":        uniform(0.70, 0.25),        # 0.70–0.95
            "colsample_bytree": uniform(0.65, 0.25),        # 0.65–0.90
            "colsample_bynode": uniform(0.70, 0.25),        # 0.70–0.95
            "min_child_weight": randint(2, 8),              # {2..7}
            "gamma":            uniform(0.03, 0.30),        # 0.03–0.30
            "reg_alpha":        loguniform(1e-3, 0.5),      # 0.001–0.5
            "reg_lambda":       loguniform(1.5, 10.0),      # 1.5–10
        }
        # AUC-oriented (rank pickup)
        params_auc = {
            "max_depth":        randint(4, 7),              # {4,5,6}
            "max_leaves":       randint(96, 160),           # {96..159}
            "learning_rate":    loguniform(0.04, 0.10),     # 0.04–0.10
            "subsample":        uniform(0.75, 0.25),        # 0.75–1.00
            "colsample_bytree": uniform(0.70, 0.25),        # 0.70–0.95
            "colsample_bynode": uniform(0.70, 0.25),        # 0.70–0.95
            "min_child_weight": randint(1, 6),              # {1..5}
            "gamma":            uniform(0.02, 0.25),        # 0.02–0.25
            "reg_alpha":        loguniform(1e-4, 0.3),      # 0.0001–0.3
            "reg_lambda":       loguniform(1.5, 8.0),       # 1.5–8
        }

    # -------- Optional common wideners (mix of discrete choices) --------
    params_common = {
        "min_child_weight": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        "gamma":            [0.0, 0.02, 0.05, 0.10],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "max_leaves":       [256, 512, 1024],  # allows bigger capacity if search hits it
        "max_bin":          [128, 256],
        "learning_rate":    [0.05, 0.07, 0.09],
        "reg_lambda":       [0.0, 0.5, 1.0, 2.0],
        "reg_alpha":        [0.0, 0.1, 0.3],
    }
    params_ll.update(params_common)
    params_auc.update(params_common)

    # -------------- scrub params (defensive) --------------
    danger_keys = {
        "objective", "_estimator_type", "response_method",
        "eval_metric", "scale_pos_weight"
    }
    params_ll  = {k: v for k, v in params_ll.items()  if k not in danger_keys}
    params_auc = {k: v for k, v in params_auc.items() if k not in danger_keys}

    return base_kwargs, params_ll, params_auc



def pick_blend_weight_on_oof(
    y_oof, p_oof_auc, p_oof_log=None, grid=None, eps=1e-4,
    metric="logloss", hybrid_alpha=0.9
):
    import numpy as np
    from sklearn.metrics import log_loss

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
import numpy as np
import pandas as pd
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
def _spread_to_winprob(spread_pts, sport="NFL"):
    """Rough favorite win prob from spread using normal CDF with sport-specific sigma.
       Negative spread => favorite. P(win) = Φ(-spread/σ_ml)."""
    from math import erf, sqrt
    s = SIGMA_ML.get(str(sport).upper(), 11.0)
    z = -pd.to_numeric(spread_pts, errors="coerce") / max(s, 1e-6)
    # Φ(z) via erf
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

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
    if df_cross_market is not None:
        out = out.merge(df_cross_market, on="Game_Key", how="left")

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

# Use it in training
def train_sharp_model_from_bq(sport: str = "NBA", days_back: int = 35):
    SPORT_DAYS_BACK = {"NBA": 35, "NFL": 60, "CFL": 45, "WNBA": 45, "MLB": 45, "NCAAF": 45, "NCAAB": 60}
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
        "Current_Win_Streak_Prior","Current_Loss_Streak_Prior",
        "H2H_Win_Pct_Prior",  # "Opp_WinPct_Prior",
        # "Last_Matchup_Result","Last_Matchup_Margin","Days_Since_Last_Matchup",
        # "Wins_Last5_Prior",
        "Margin_Last5_Prior",
        "Days_Since_Last_Game",
        # "Close_Game_Rate_Prior","Blowout_Game_Rate_Prior",
        # "Avg_Home_Margin_Prior","Avg_Away_Margin_Prior",
    ]
    
    # === New team ATS cover / margin stats (prior-only, last-5) ===
    team_cover_cols = [
        # overall cover signal (intentionally off for now)
        # "Cover_Rate_Last5",
        # "Cover_Rate_After_Win_Last5",
        # "Cover_Rate_After_Loss_Last5",
        # situational cover rates (intentionally off for now)
        # "Cover_Rate_Home_After_Home_Win_Last5",
        # "Cover_Rate_Home_After_Home_Loss_Last5",
        # "Cover_Rate_Home_After_Away_Win_Last5",
        # "Cover_Rate_Home_After_Away_Loss_Last5",
        # "Cover_Rate_Away_After_Home_Win_Last5",
        # "Cover_Rate_Away_After_Home_Loss_Last5",
        # "Cover_Rate_Away_After_Away_Win_Last5",
        # "Cover_Rate_Away_After_Away_Loss_Last5",
        # margin distribution
        "ATS_Cover_Margin_Last5_Prior_Mean",
        #"ATS_Cover_Margin_Last5_Prior_Std",
        "Market_Bucket",
        "Market_OddsProb_Bucket"
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
        pad_days=30,
        allow_forward_hours=0.0,  # strict backward-only
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
    
    # derive markets present (filter to the 3 you care about)
    allowed = {'spreads', 'totals', 'h2h'}
    markets_present = [m for m in df_bt['Market'].astype(str).str.lower().unique() if m in allowed]
    
    n_markets = max(1, len(markets_present))
    pb = st.progress(0)  # 0–100
    status = st.status("🔄 Training in progress...", expanded=True)
    
    for i, market in enumerate(markets_present, 1):
        pct = int(round(i / n_markets * 100))
        status.write(f"🚧 Training model for `{market.upper()}`...")
    
        df_market = df_bt[df_bt['Market'].astype(str).str.lower() == market].copy()

       
        if df_market.empty:
            status.warning(f"⚠️ No data for {market.upper()} — skipping.")
            pb.progress(min(100, max(0, pct)))
            continue
        
        # Totals: keep latest-per-outcome snapshot
        if market == "totals":
            df_market = (
                df_market.sort_values(['Snapshot_Timestamp'])
                         .drop_duplicates(subset=['Game_Key','Bookmaker','Market','Outcome'], keep='last')
                         .reset_index(drop=True)
            )
        
        # Defuse categoricals that can raise setitem errors
        for col in ("Sport","Market","Bookmaker"):
            if col in df_market.columns and str(df_market[col].dtype).startswith("category"):
                df_market[col] = df_market[col].astype(str)
        def _amer_to_prob_vec(s):
            s = pd.to_numeric(s, errors="coerce")
            return np.where(s > 0, 100.0/(s+100.0),
                   np.where(s < 0, (-s)/((-s)+100.0), np.nan)).astype("float32")
        df_market = df_market.merge(df_cross_market, on="Game_Key", how="left")
        # Ensure columns exist; if missing, create as NaN
        for c in ["Spread_Odds","Total_Odds","H2H_Odds"]:
            if c not in df_market.columns:
                df_market[c] = np.nan
        
        # Compute implied probs with row-level fallback to Odds_Price (no KeyError, vectorized)
        df_market["Spread_Implied_Prob"] = _amer_to_prob_vec(
            df_market["Spread_Odds"].where(df_market["Spread_Odds"].notna(), df_market.get("Odds_Price"))
        )
        df_market["Total_Implied_Prob"] = _amer_to_prob_vec(
            df_market["Total_Odds"].where(df_market["Total_Odds"].notna(), df_market.get("Odds_Price"))
        )
        df_market["H2H_Implied_Prob"] = _amer_to_prob_vec(
            df_market["H2H_Odds"].where(df_market["H2H_Odds"].notna(), df_market.get("Odds_Price"))
        )

        # Your existing FE before resistance
        df_market = compute_small_book_liquidity_features(df_market)
        df_market = add_favorite_context_flag(df_market)
        
        # Final-snapshot training slice (no timestamps needed)
        df_market = add_resistance_features_training(
            df_market,
            emit_levels_str=False,  # keep memory tiny for training
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
        if market == "totals":
            df_market = df_market[df_market["Outcome_Norm"] == "over"]
        elif market in ("spreads", "h2h"):
            df_market = df_market[df_market["Value"] < 0]
        
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
                "Model_Expected_Margin_Abs","Sigma_Pts","k"
            ]
            g_map = g_fc[game_keys + proj_cols].copy()
            g_map = (g_map.sort_values(game_keys)
                          .drop_duplicates(subset=game_keys, keep="last"))
        
            # Single merge (no duplicates)
            before = len(df_market)
            df_market = df_market.merge(g_map, on=game_keys, how="left", validate="many_to_one")
            _assert_no_growth(before, df_market, "merge game-level projections")
        
            # Compute per-outcome spreads from model/market
            for c in ["Outcome_Norm","Market_Favorite_Team"]:
                df_market[c] = df_market[c].astype(str).str.lower().str.strip()
        
            is_fav = (df_market["Outcome_Norm"].values == df_market["Market_Favorite_Team"].values)
        
            df_market["Outcome_Model_Spread"]  = np.where(
                is_fav, df_market["Model_Fav_Spread"].values, df_market["Model_Dog_Spread"].values
            ).astype("float32")
        
            df_market["Outcome_Market_Spread"] = np.where(
                is_fav, df_market["Favorite_Market_Spread"].values, df_market["Underdog_Market_Spread"].values
            ).astype("float32")
        
            # --- Per-outcome engineered features (requires: k, Sigma_Pts, fav/dog spreads) ---
            df_market["Outcome_Spread_Edge"] = (
                pd.to_numeric(df_market["Outcome_Model_Spread"],  errors="coerce") -
                pd.to_numeric(df_market["Outcome_Market_Spread"], errors="coerce")
            ).astype("float32")
        
            # z = edge / k  (guard against zero/NaN k)
            k = pd.to_numeric(df_market.get("k"), errors="coerce").astype("float32")
            k = k.where(k > 0, np.nan)
            df_market["z"] = (df_market["Outcome_Spread_Edge"] / k).astype("float32")
        
            # cover prob: Φ(z)
            df_market["Outcome_Cover_Prob"] = _phi(df_market["z"]).astype("float32")
        
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
        # === Directional agreement (for spreads/h2h invert line logic)
        df_market['Line_Delta'] = pd.to_numeric(df_market['Line_Delta'], errors='coerce')
       
        
        df_market['Direction_Aligned'] = np.where(
            df_market['Line_Delta'] > 0, 1,
            np.where(df_market['Line_Delta'] < 0, 0, -1)
        ).astype(int)
        df_market['Line_Value_Abs'] = df_market['Value'].abs()
        val_num = pd.to_numeric(df_market['Value'], errors='coerce').fillna(0.0)
        df_market['Line_Delta_Signed'] = df_market['Line_Delta'] * np.sign(val_num)
        
        
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
        
        # --- start with your manual core list ---
        features = [
            # 🔹 Core sharp signals
            #'Sharp_Move_Signal',
            #'Sharp_Limit_Jump',#'Sharp_Time_Score','Book_lift_x_Sharp',
            #'Book_lift_x_Magnitude',
            'Book_lift_x_PROB_SHIFT','Sharp_Limit_Total',
            'Is_Reinforced_MultiMarket','Market_Leader',#'LimitUp_NoMove_Flag',
        
            # 🔹 Market response
            #'Sharp_Line_Magnitude',
            #'Is_Home_Team_Bet',
            'Line_Moved_Toward_Team',
            #'Team_Implied_Prob_Gap_Home','Team_Implied_Prob_Gap_Away',
        
            # 🔹 Engineered odds shift decomposition
            #'SharpMove_Odds_Up','SharpMove_Odds_Down','SharpMove_Odds_Mag',
        
            # 🔹 Engineered interactions
            #'MarketLeader_ImpProbShift','LimitProtect_SharpMag','Delta_Sharp_vs_Rec',#'Sharp_Leads',
        
            # 🔁 Reversal logic
            'Value_Reversal_Flag','Odds_Reversal_Flag',
        
            # 🔥 Timing flags
            'Late_Game_Steam_Flag',
        
            #'Abs_Line_Move_From_Opening',
            #'Abs_Odds_Move_From_Opening',
            #'Market_Mispricing',#'Spread_vs_H2H_Aligned','Total_vs_Spread_Contradiction',
            #'Spread_vs_H2H_ProbGap','Total_vs_H2H_ProbGap','Total_vs_Spread_ProbGap',
            #'CrossMarket_Prob_Gap_Exists',
            'Line_Moved_Away_From_Team',
            
            
            'Pct_Line_Move_From_Opening',#'Pct_Line_Move_Bin',
            'Potential_Overmove_Flag',
            #'Potential_Overmove_Total_Pct_Flag',#'Mispricing_Flag',
            #'Was_Line_Resistance_Broken',
            'Line_Resistance_Crossed_Count','SharpMove_Resistance_Break',
        
            # 🧠 Cross-market alignment
            'Potential_Odds_Overmove_Flag',
            #'Abs_Line_Move_Z','Pct_Line_Move_Z',
            'SmallBook_Limit_Skew',
            'SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag',
            #'Book_Reliability_Score',
            #'Book_Reliability_Lift',#'Book_Reliability_x_Sharp',
            'Book_Reliability_x_Magnitude',
            'Book_Reliability_x_PROB_SHIFT',
        
            # Power ratings / edges
            'PR_Team_Rating','PR_Opp_Rating',
            'PR_Rating_Diff',#'PR_Abs_Rating_Diff',
            #'Outcome_Model_Spread','Outcome_Market_Spread',
            'Outcome_Spread_Edge',
            'Outcome_Cover_Prob','model_fav_vs_market_fav_agree',
            #'TOT_Proj_Total_Baseline',#'TOT_Off_H','TOT_Def_H','TOT_Off_A','TOT_Def_A',
            #'TOT_GT_H','TOT_GT_A',#'TOT_LgAvg_Total',
            #'TOT_Mispricing', 
            'ATS_EB_Rate',
            #'ATS_EB_Margin',            # Optional: only if cover_margin_col was set
            #'ATS_Roll_Margin_Decay',    # Optional: only if cover_margin_col was set
            'ATS_EB_Rate_Home',
            'ATS_EB_Rate_Away',
            'PR_Model_Agree_H2H_Flag','PR_Market_Agree_H2H_Flag'
            
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
        extend_unique(features, [c for c in history_present if c not in features])
        
        # add recent team model performance stats
        extend_unique(features, [
            #'Team_Past_Avg_Model_Prob',
            #'Team_Past_Hit_Rate',
            #'Team_Past_Avg_Model_Prob_Home',
            #'Team_Past_Hit_Rate_Home',
            #'Team_Past_Avg_Model_Prob_Away',
            #'Team_Past_Hit_Rate_Away',
            
            #'Avg_Recent_Cover_Streak','Avg_Recent_Cover_Streak_Home',
            #'Avg_Recent_Cover_Streak_Away'
        ])
        
        # add time-context flags
        extend_unique(features, ['Is_Night_Game','Is_PrimeTime','DOW_Sin'])
        
        extend_unique(features, [
            "Implied_Hold_Book",#"Two_Sided_Offered","Juice_Abs_Delta",
            "Dist_To_Next_Key","Key_Corridor_Pressure",
            #"Book_PctRank_Line",
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
            #"Hybrid_Timing_Entropy_Line","Hybrid_Timing_Entropy_Odds",
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
        
        st.markdown(f"### 📈 Features Used: `{len(features)}`")
        X = (df_market[feature_cols]
             .apply(pd.to_numeric, errors='coerce')
             .replace([np.inf, -np.inf], np.nan)
             .fillna(0.0)
             .astype('float32'))
                
     
        
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
            "MLB":   pd.Timedelta("1 hours"),
            "NBA":   pd.Timedelta("1 hours"),
            "NHL":   pd.Timedelta("1 hours"),
            "NCAAB": pd.Timedelta("11 hours"),
            "NFL":   pd.Timedelta("1 days"),
            "NCAAF": pd.Timedelta("1 days"),
            "WNBA":  pd.Timedelta("1 hours"),
            "MLS":   pd.Timedelta("1 hours"),
            "default": pd.Timedelta("1 hours"),
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
        
        if market_label.startswith("spread") or market_label == "spreads":
            # Explicit PR features you've seen + any columns with PR_ prefix
            PR_EXPLICIT = {
                "PR_Team_Rating",
                "PR_Opp_Rating",
                "PR_Abs_Rating_Diff",
                "PR_Rating_Diff",
            }
            def _is_pr(col: str) -> bool:
                c = str(col)
                return c.startswith("PR_") or (c in PR_EXPLICIT)
        
            before = len(feature_cols)
            feature_cols = [c for c in feature_cols if not _is_pr(c)]
            removed = before - len(feature_cols)
            if removed:
                print(f"🧹 Removed {removed} PR features for SPREADS.")
        
        # Keep a single source of truth for downstream code
        features = feature_cols

        # ============================================================================
        
       
        # === Build X / y ===
        X_full = _to_numeric_block(df_market, feature_cols).to_numpy(np.float32)
        
        y_series   = pd.to_numeric(df_market["SHARP_HIT_BOOL"], errors="coerce")
        valid_mask = ~y_series.isna()
        if not valid_mask.all():
            X_full = X_full[valid_mask.to_numpy()]
        y_full = y_series.loc[valid_mask].astype(int).to_numpy()
         # ---- Cheap feature pruning (before any split/CV) --
        
        
        st.markdown("### 🧹 Feature Pruning (pre-split)")
        
        # 1) Near-constant features
        vt = VarianceThreshold(threshold=1e-5)
        X_full_pruned = vt.fit_transform(X_full)
        
        if hasattr(vt, "get_support"):
            mask_keep = vt.get_support()
            removed_const = [c for c, keep in zip(feature_cols, mask_keep) if not keep]
            feature_cols  = [c for c, keep in zip(feature_cols, mask_keep) if keep]
        else:
            # Fallback (very rare)
            mask_keep = np.ones(X_full.shape[1], dtype=bool)
            removed_const = []
            feature_cols  = feature_cols[:X_full_pruned.shape[1]]
        
        st.write(f"• Removed near-constant features: {len(removed_const)}")
        if removed_const:
            st.caption(", ".join(removed_const[:20]) + (" ..." if len(removed_const) > 20 else ""))
        
        # 2) Exact duplicate columns (optional but cheap)
        df_tmp = pd.DataFrame(X_full_pruned, columns=feature_cols)
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
     
        # ---- Groups & times (snapshot-aware) ----
        groups_all = df_market.loc[valid_mask, "Game_Key"].astype(str).to_numpy()
        
        snap_ts = pd.to_datetime(
            df_market.loc[valid_mask, "Snapshot_Timestamp"], errors="coerce", utc=True
        )
        game_ts = pd.to_datetime(
            df_market.loc[valid_mask, "Game_Start"], errors="coerce", utc=True
        )
        
        # If any game has >1 snapshot we’re in snapshot regime (embargo matters)
        by_game_snaps = (
            df_market.loc[valid_mask]
            .groupby("Game_Key")["Snapshot_Timestamp"]
            .nunique(dropna=True)
        )
        has_snapshots = (by_game_snaps.fillna(0).max() > 1)
        
        time_values_all = snap_ts.to_numpy() if has_snapshots else game_ts.to_numpy()
        
        # Sport-aware embargo only when we truly have multiple snapshots
        sport_key = str(sport).upper()
        embargo_td = SPORT_EMBARGO.get(sport_key, SPORT_EMBARGO["default"]) if has_snapshots else pd.Timedelta(0)
        
        # Replace previous 'groups' and 'times'
        groups = groups_all
        times  = time_values_all
        
        # Guard: drop NaT times (rare but fatal for CV)
        if pd.isna(times).any():
            bad  = np.flatnonzero(pd.isna(times))
            keep = np.setdiff1d(np.arange(len(times)), bad)
            X_full = X_full[keep]; y_full = y_full[keep]
            groups = groups[keep]; times  = times[keep]
        
        # ---- Holdout = last ~N groups (time-forward, group-safe) ----
        meta  = pd.DataFrame({"group": groups, "time": pd.to_datetime(times, utc=True)})
        gmeta = (meta.groupby("group", as_index=False)["time"]
                   .agg(start="min", end="max")
                   .sort_values("start")
                   .reset_index(drop=True))
        
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

        # ---- 5) Simple health checks (Streamlit friendly) ----
        st.markdown("### 🩺 Data Health Checks")
        
        # 1. Class balance
        pos_rate = float(np.mean(y_train))
        st.write(f"✅ Class balance — Positives: `{pos_rate:.3f}` ({int(pos_rate * 100)}%)")
        
        # 2. Shape check
        st.write(f"🔢 Train shape: {X_train.shape[0]} rows × {X_train.shape[1]} features")
        
        # 3. NaN check
        has_nan = np.isnan(X_train).any()
        if has_nan:
            st.error("❌ NaNs found in X_train")
        else:
            st.success("✅ No NaNs in X_train")
        
        # 4. Constant columns (optional redundancy check)
        n_const = (np.std(X_train, axis=0) < 1e-6).sum()
        if n_const > 0:
            st.warning(f"⚠️ {n_const} features are near-constant in X_train")
        
        # 5. Class diversity in holdout (if known)
        if 'y_hold_vec' in locals():
            n_pos = (y_hold_vec == 1).sum()
            n_neg = (y_hold_vec == 0).sum()
            st.write(f"📊 Holdout — Pos: {n_pos}, Neg: {n_neg}")
            
        bk_col = "Bookmaker" if "Bookmaker" in df_market.columns else (
            "Bookmaker_Norm" if "Bookmaker_Norm" in df_market.columns else None
        )
        
        train_df = df_market.loc[valid_mask].iloc[train_all_idx].copy()
        def _fallback_book_lift(df: pd.DataFrame, book_col: str, label_col: str, prior: float = 100.0):
            if (book_col is None) or (book_col not in df.columns) or (label_col not in df.columns):
                return pd.Series(dtype=float)
            dfv = df[[book_col, label_col]].dropna()
            if dfv.empty:
                return pd.Series(dtype=float)
        
            y = pd.to_numeric(dfv[label_col], errors="coerce")
            m = float(np.nanmean(y))
            grp = dfv.groupby(book_col)[label_col].agg(["sum", "count"]).rename(columns={"sum": "hits", "count": "n"})
            a = m * prior; b = (1.0 - m) * prior
            post_mean = (grp["hits"] + a) / (grp["n"] + a + b)
            lift = (post_mean / max(1e-9, m)) - 1.0
            return lift
        
        if bk_col is None:
            w_train = np.ones(len(train_df), dtype=np.float32)
        else:
            if bk_col not in train_df.columns:
                train_df[bk_col] = "UNK"
        
            B_g  = train_df.groupby("Game_Key")[bk_col].nunique()
            n_gb = train_df.groupby(["Game_Key", bk_col]).size()
        
            TAU = 0.6
            def _w_gb(g, b, tau=1.0):
                Bg  = max(1, int(B_g.get(g, 1)))
                ngb = max(1, int(n_gb.get((g, b), 1)))
                return 1.0 / (Bg * (ngb ** tau))
        
            w_base = np.array([_w_gb(g, b, TAU) for g, b in zip(train_df["Game_Key"], train_df[bk_col])], dtype=np.float32)
        
           # --- Sharp-book tilt only (no reliability map) --------------------------------
            # Requires: train_df, w_base, bk_col, SHARP_BOOKS
            
            if bk_col not in train_df.columns:
                raise KeyError(f"bk_col '{bk_col}' not found in train_df columns")
            
            # sharp flag
            if "Is_Sharp_Book" in train_df.columns:
                is_sharp = train_df["Is_Sharp_Book"].fillna(False).astype(bool).astype(np.float32).to_numpy()
            else:
                is_sharp = train_df[bk_col].isin(SHARP_BOOKS).astype(np.float32).to_numpy()
            
            ALPHA_SHARP = 0.80
            
            # base multiplier from sharp tilt
            mult = 1.0 + ALPHA_SHARP * is_sharp
          
            # final weights
            w_train = pd.Series(w_base, index=train_df.index).astype(np.float32).to_numpy()
        
            s = w_train.sum()
            if s > 0:
                w_train *= (len(w_train) / s)
        
        assert len(w_train) == len(X_train), f"sample_weight misaligned: {len(w_train)} vs {len(X_train)}"
        
        # ---------------------------------------------------------------------------
      
          #  CV with purge + embargo (snapshot-aware)
        # ---------------------------------------------------------------------------
        rows_per_game = int(np.ceil(len(X_train) / max(1, pd.unique(g_train).size)))
        target_games  = 8 if sport_key in SMALL_LEAGUES else 20
        min_val_size  = max(10 if sport_key in SMALL_LEAGUES else 20, target_games * rows_per_game)
        
        n_groups_train = pd.unique(g_train).size
        target_folds   = 5 if n_groups_train >= 200 else (4 if n_groups_train >= 120 else 3)
        
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=target_folds,
            embargo=embargo_td,
            time_values=t_train,
            min_val_size=min_val_size,
        )
        
        # --- Ensure binary labels and filter folds so both train & val have {0,1} ----
        def _has_both_classes(idx, y):
            u = set(np.unique(y[idx]))
            return (0 in u) and (1 in u)
        
        def filter_cv_splits(cv_obj, X, y, groups=None):
            safe = []
            for tr, va in cv_obj.split(X, y, groups):
                if _has_both_classes(tr, y) and _has_both_classes(va, y):
                    safe.append((tr, va))
            return safe
        
        y_train = pd.Series(y_train).astype(int).clip(0, 1).to_numpy()
        
        cv_splits = filter_cv_splits(cv, X_train, y_train, groups=g_train)
        # Optional: inspect how many survived
        # print(f"kept {len(cv_splits)} balanced CV folds")
        
        # Fallback if none survive: single 80/20 split, only if both sides are dual-class
        if not cv_splits:
            n = len(y_train)
            cut = max(1, int(0.8 * n))
            tr = np.arange(cut)
            va = np.arange(cut, n)
            if _has_both_classes(tr, y_train) and _has_both_classes(va, y_train):
                cv_splits = [(tr, va)]
            else:
                # Still single-class → consider your 0.50 placeholder path here
                # For now, raise to make the situation explicit
                raise RuntimeError("No class-balanced CV splits available; widen data or use placeholder model.")
        folds = cv_splits

        # ================== << SHAP stability selection >> ==================
        # Use the PRUNED names to build a columned DF for SHAP
        X_train_df = pd.DataFrame(X_train, columns=list(features_pruned))
        
        selected_feats, shap_summary = shap_stability_select(
            model_proto=XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                grow_policy="lossguide",
                max_bin=128,
                max_delta_step=0.5,
                n_jobs=1,
                random_state=42
            ),
            X=X_train_df,
            y=y_train,
            folds=folds,                # reuse the CV folds built earlier (rows unchanged)
            topk_per_fold=35,
            min_presence=0.85,
            max_keep=80,
            sample_per_fold=4000,
            random_state=42,
        )
        
        # 🔒 Final feature list (post-SHAP)
        FEATURE_COLS_FINAL = tuple(selected_feats)
        feature_cols = list(FEATURE_COLS_FINAL)
        features     = feature_cols  # if you use `features` later


       
        # UI: how many SHAP removed
        n_before = len(features_pruned); n_after = len(FEATURE_COLS_FINAL); removed = n_before - n_after
        st.success(f"🧠 SHAP selection kept {n_after} / {n_before} features (−{removed} removed).")
        if removed > 0:
            dropped = [c for c in features_pruned if c not in FEATURE_COLS_FINAL]
            st.caption("Dropped (sample): " + ", ".join(dropped[:20]) + (" ..." if removed > 20 else ""))
        
        # Rebuild BOTH matrices by name to ensure identical columns & order (do this ONCE)
        X_train = (pd.DataFrame(X_train, columns=list(features_pruned))
                     .loc[:, list(FEATURE_COLS_FINAL)]
                     .to_numpy(np.float32))
        X_full  = (pd.DataFrame(X_full,  columns=list(features_pruned))
                     .loc[:, list(FEATURE_COLS_FINAL)]
                     .to_numpy(np.float32))
        
   
        
        # 👉 keep downstream code happy by syncing the legacy var
        feature_cols = list(FEATURE_COLS_FINAL)
        features     = feature_cols  # if you pass `features` elsewhere
        
        # sanity checks (fail fast if anything drifted)
        assert X_train.shape[1] == len(feature_cols), f"X_train={X_train.shape[1]} vs feature_cols={len(feature_cols)}"
        assert X_full.shape[1]  == len(feature_cols),  f"X_full={X_full.shape[1]}  vs feature_cols={len(feature_cols)}"
        
        # Now compute spw & search space (AFTER setting `features`)
        
        # build or update base_kwargs only 
        pos_rate = float(np.mean(y_train))
        n_jobs = 1
        base_kwargs, params_ll, params_auc = get_xgb_search_space(
            sport=sport,
            X_rows=X_train.shape[0],
            n_jobs=n_jobs,
            features=features,
        )
        
        # Optionally set a better prior (recommended):
        base_kwargs["base_score"] = float(np.clip(np.mean(y_train), 1e-4, 1-1e-4))
        
       # ================== FAST SEARCH → DEEP REFIT ==================
        
        # 1) Build *cheap* search estimators (broad exploration)
        SEARCH_N_EST   = 300          # small budget per config during search
        SEARCH_MAX_BIN = 128          # cheaper histograms for speed
        
        # Parallelize across trials (keep estimator n_jobs=1)
        VCPUS = get_vcpus()
        
        est_ll  = XGBClassifier(**{**base_kwargs,
                                   "n_estimators": SEARCH_N_EST,
                                   "eval_metric": "logloss",
                                   "max_bin": SEARCH_MAX_BIN,
                                   "n_jobs": 1})
        est_auc = XGBClassifier(**{**base_kwargs,
                                   "n_estimators": SEARCH_N_EST,
                                   "eval_metric": "auc",
                                   "max_bin": SEARCH_MAX_BIN,
                                   "n_jobs": 1})
        
        # 2) Prefer Successive Halving; fallback to RandomizedSearch
        try:
            
            search_kwargs_ll  = dict(factor=3, resource="n_estimators",
                                     min_resources=64, max_resources=SEARCH_N_EST)
            search_kwargs_auc = dict(factor=3, resource="n_estimators",
                                     min_resources=64, max_resources=SEARCH_N_EST)
        except Exception:
            
            # keep your trial heuristic
            try:
                search_trials  # type: ignore
            except NameError:
                search_trials = _resolve_search_trials(sport, X_train.shape[0])
            search_trials = int(search_trials) if str(search_trials).isdigit() else _resolve_search_trials(sport, X_train.shape[0])
            search_kwargs_ll  = dict(n_iter=search_trials)
            search_kwargs_auc = dict(n_iter=search_trials)
        
        rs_ll = _SearchCV(
            estimator=est_ll,
            param_distributions=params_ll,
            scoring="neg_log_loss",
            cv=folds,                    # your precomputed purge+embargo splits
            n_jobs=VCPUS,                # parallel across trials
            random_state=42,
            verbose=1 if st.session_state.get("debug", False) else 0,
            **search_kwargs_ll
        )
        
        rs_auc = _SearchCV(
            estimator=est_auc,
            param_distributions=params_auc,
            scoring="roc_auc",
            cv=folds,
            n_jobs=VCPUS,
            random_state=137,
            verbose=1 if st.session_state.get("debug", False) else 0,
            **search_kwargs_auc
        )
        
        fit_params_search = dict(sample_weight=w_train, verbose=False)
        
        # Optionally clamp threadpools inside the search fits too
        with threadpool_limits(limits=1):
            rs_ll.fit(X_train, y_train, groups=g_train, **fit_params_search)
            rs_auc.fit(X_train, y_train, groups=g_train, **fit_params_search)
        
        best_ll_params  = rs_ll.best_params_.copy()
        best_auc_params = rs_auc.best_params_.copy()
        for k in ("objective", "eval_metric", "_estimator_type", "response_method"):
            best_ll_params.pop(k, None)
            best_auc_params.pop(k, None)
        
        # 3) Deep refit with early stopping on the last fold
        tr_es_rel, va_es_rel = folds[-1]
        tr_es_rel = np.asarray(tr_es_rel); va_es_rel = np.asarray(va_es_rel)
        
        X_tr_es = X_train[tr_es_rel];  y_tr_es = y_train[tr_es_rel]
        X_va_es = X_train[va_es_rel];  y_va_es = y_train[va_es_rel]
        w_tr_es = np.maximum(w_train[tr_es_rel], 1e-6)
        w_va_es = np.maximum(w_train[va_es_rel], 1e-6)
        
        final_estimators_cap  = 10000   # allow very deep/long growth
        early_stopping_rounds = 500
        
        # (You can keep lossguide for both; if you *want* depthwise for AUC, retain your override.)
        model_logloss = XGBClassifier(
            **{**base_kwargs, **best_ll_params,
               "n_estimators": final_estimators_cap,
               "eval_metric": "logloss",
               "max_bin": 256,            # restore full-quality bins
               "n_jobs": VCPUS}
        )
        
        model_auc = XGBClassifier(
            **{**base_kwargs, **best_auc_params,
               "n_estimators": final_estimators_cap,
               "eval_metric": "auc",
               "max_bin": 256,
               "n_jobs": VCPUS}
               "grow_policy": "depthwise",
        )
        
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
        
        # (Optional) consolidate helpers you already had:
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
        
        n_trees_ll  = _best_rounds(model_logloss)
        n_trees_auc = _best_rounds(model_auc)
        
        # Optional: refit on ALL training rows at the discovered best_rounds
        if n_trees_ll > 0:
            model_logloss.set_params(n_estimators=n_trees_ll)
            model_logloss.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        
        if n_trees_auc > 0:
            model_auc.set_params(n_estimators=n_trees_auc)
            model_auc.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        
        # (Keep your existing metric tail logging if you like)
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
            return (arr[-10:] if len(arr) >= 10 else arr), {"dataset": ds, "metric_key": key, "len": len(arr)}
        
        val_logloss_last10, info_log = _safe_metric_tail(model_logloss, "logloss")
        val_auc_last10,     info_auc = _safe_metric_tail(model_auc,     "auc")
        
        st.write({
            "ES_n_trees_ll": getattr(model_logloss, "best_iteration", None),
            "ES_n_trees_auc": getattr(model_auc, "best_iteration", None),
            "val_logloss_last10": val_logloss_last10,
            "val_auc_last10": val_auc_last10,
        })
        # ================== END FAST SEARCH → DEEP REFIT ==================

        
        # -----------------------------------------
        # OOF predictions (train-only) + blending
        # -----------------------------------------
        # Decide if we actually have a viable logloss stream
        # Force whether to include logloss model in the blend
        USE_LOGLOSS_STREAM = False  # 🔁 set to True if you want logloss+auc blend
        RUN_LOGLOSS = USE_LOGLOSS_STREAM

        
        # Preallocate (float64 for safety)
        oof_pred_auc = np.full(len(y_train), np.nan, dtype=np.float64)
        oof_pred_logloss = np.full(len(y_train), np.nan, dtype=np.float64) if RUN_LOGLOSS else None
        
        for tr_rel, va_rel in folds:
            # AUC stream
            m_auc = XGBClassifier(**{**base_kwargs, **best_auc_params, "n_estimators": int(n_trees_auc)})
            m_auc.fit(X_train[tr_rel], y_train[tr_rel], sample_weight=w_train[tr_rel], verbose=False)
            oof_pred_auc[va_rel] = pos_proba(m_auc, X_train[va_rel], positive=1)
        
            # Logloss stream (only if enabled)
            if RUN_LOGLOSS:
                m_ll = XGBClassifier(**{**base_kwargs, **best_ll_params, "n_estimators": int(n_trees_ll)})
                m_ll.fit(X_train[tr_rel], y_train[tr_rel], sample_weight=w_train[tr_rel], verbose=False)
                oof_pred_logloss[va_rel] = pos_proba(m_ll, X_train[va_rel], positive=1)
        
        # ---- Clean & clip OOF vectors ----
        mask_auc = np.isfinite(oof_pred_auc)
        mask_log = np.isfinite(oof_pred_logloss) if RUN_LOGLOSS else mask_auc
        mask_oof = mask_auc & mask_log
        n_oof = int(mask_oof.sum())
        if n_oof < 50:
            raise RuntimeError(f"Too few finite OOF predictions after masking ({n_oof}). Check folds & models.")
        
        y_oof = y_train[mask_oof].astype(int)
        p_oof_auc = np.clip(oof_pred_auc[mask_oof].astype(np.float64), eps, 1 - eps)
        p_oof_log = np.clip(oof_pred_logloss[mask_oof].astype(np.float64), eps, 1 - eps) if RUN_LOGLOSS else None
        
        # ---- Pick blend weight on OOF (AUC-only if RUN_LOGLOSS=False) ----
        if RUN_LOGLOSS:
            best_w, p_oof_blend, flipped = pick_blend_weight_on_oof(
                y_oof=y_oof, p_oof_auc=p_oof_auc, p_oof_log=p_oof_log, eps=eps, metric="logloss"
            )
        else:
            best_w, p_oof_blend, flipped = pick_blend_weight_on_oof(
                y_oof=y_oof, p_oof_auc=p_oof_auc, p_oof_log=None, eps=eps, metric="logloss"
            )  # best_w will be 0.0 here
        
        # Final safety
        p_oof_blend = np.asarray(p_oof_blend, dtype=np.float64)
        if not np.isfinite(p_oof_blend).all():
            bad = np.flatnonzero(~np.isfinite(p_oof_blend))
            st.error(f"Non-finite OOF blend at {bad[:10]} (showing up to 10). Dropping them.")
            keep2 = np.isfinite(p_oof_blend)
            y_oof, p_oof_blend = y_oof[keep2], p_oof_blend[keep2]
        
        st.write({
            "RUN_LOGLOSS": bool(RUN_LOGLOSS),
            "blend_w": float(best_w),
            "flipped": bool(flipped),
            "oof_len": int(len(p_oof_blend)),
            "oof_minmax": (float(p_oof_blend.min()), float(p_oof_blend.max()))
        })

        # sanity
        assert np.isfinite(p_oof_blend).all(), "NaNs in p_oof_blend"

        # Calibrate (keep your existing calibrator stack)
        CLIP = 0.005 if sport_key not in SMALL_LEAGUES else 0.01
        use_quantile_iso = False
        cals_raw = fit_iso_platt_beta(p_oof_blend, y_oof, eps=eps, use_quantile_iso=use_quantile_iso)
        
        cals = _normalize_cals(cals_raw)
        cals["iso"]   = _ensure_transform_for_iso(cals.get("iso"))
        cals["platt"] = _ensure_predict_proba_for_prob_cal(cals.get("platt"), eps=eps)
        cals["beta"]  = _ensure_predict_proba_for_prob_cal(cals.get("beta"),  eps=eps)
        if cals["iso"] is None: cals["iso"] = _IdentityIsoCal(eps=eps)
        if cals["platt"] is None and cals["beta"] is None: cals["platt"] = _IdentityProbCal(eps=eps)
        sel = select_blend(cals, p_oof_blend, y_oof, eps=eps)
        
        # --- Final blended raw preds (TRAIN/HOLDOUT) ---
        p_tr_auc = pos_proba(model_auc, X_full[train_all_idx], positive=1)
        p_ho_auc = pos_proba(model_auc, X_full[hold_idx],      positive=1)
        
        if RUN_LOGLOSS:
            p_tr_log = pos_proba(model_logloss, X_full[train_all_idx], positive=1)
            p_ho_log = pos_proba(model_logloss, X_full[hold_idx],      positive=1)
            p_train_blend_raw = np.clip(best_w * p_tr_log + (1 - best_w) * p_tr_auc, eps, 1 - eps)
            p_hold_blend_raw  = np.clip(best_w * p_ho_log + (1 - best_w) * p_ho_auc, eps, 1 - eps)
        else:
            best_w = 0.0  # make this explicit for logging
            p_train_blend_raw = np.clip(p_tr_auc, eps, 1 - eps)
            p_hold_blend_raw  = np.clip(p_ho_auc, eps, 1 - eps)
        
        # --- Apply your calibrator blend ---
        p_cal     = apply_blend(sel, p_train_blend_raw, eps=eps, clip=(CLIP, 1 - CLIP))
        p_cal_val = apply_blend(sel, p_hold_blend_raw,  eps=eps, clip=(CLIP, 1 - CLIP))

        
        # Backfill so select_blend never sees None
        if cals["iso"] is None:
            cals["iso"] = _IdentityIsoCal(eps=eps)
        if cals["platt"] is None and cals["beta"] is None:
            cals["platt"] = _IdentityProbCal(eps=eps)
        
        sel = select_blend(cals, p_oof_blend, y_oof, eps=eps)
        print(f"Calibrator blend: base={sel['base']}, alpha_iso={sel['alpha']:.2f}")
        
        # (robust calibrator optional; safe to keep)
        cal_blend = fit_robust_calibrator(p_oof=p_oof_blend, y_oof=y_oof, eps=eps, min_unique=200, prefer_beta=True)
        if cal_blend is not None:
            print(f"Calibration chosen on OOF blend: {cal_blend[0]}")
        
        # ---------- Final blended + calibrated predictions for TRAIN + HOLDOUT ----------
        pos_ll_final  = pos_col_index(model_logloss, positive=1)
        pos_auc_final = pos_col_index(model_auc,     positive=1)
        
        # ✅ Use .iloc for DataFrame row selection
        # ✅ Use regular indexing for NumPy arrays
        p_tr_log = pos_proba(model_logloss, X_full[train_all_idx], positive=1)
        p_tr_auc = pos_proba(model_auc,     X_full[train_all_idx], positive=1)
        p_ho_log = pos_proba(model_logloss, X_full[hold_idx],      positive=1)
        p_ho_auc = pos_proba(model_auc,     X_full[hold_idx],      positive=1)
        p_train_blend_raw = np.clip(best_w*p_tr_log + (1-best_w)*p_tr_auc, eps, 1-eps)
        p_hold_blend_raw  = np.clip(best_w*p_ho_log + (1-best_w)*p_ho_auc, eps, 1-eps)
        
        # Apply the legacy blend calibrator
        p_cal     = apply_blend(sel, p_train_blend_raw, eps=eps, clip=(CLIP, 1-CLIP))
        p_cal_val = apply_blend(sel, p_hold_blend_raw,  eps=eps, clip=(CLIP, 1-CLIP))
        
        # ---------- Metrics (handle single-class holdout safely) ----------
        y_hold_vec  = y_full[hold_idx].astype(int)
        y_train_vec = y_full[train_all_idx].astype(int)
        
        auc_ho = np.nan  # ✅ initialize before branch
        if np.unique(y_hold_vec).size < 2:
            # nothing to do; auc_ho stays NaN
            pass
        else:
            # final safety flip (rare)
            auc_raw, _  = _auc_safe(y_hold_vec, p_cal_val)
            auc_flip, _ = _auc_safe(y_hold_vec, 1.0 - p_cal_val)
            if np.nan_to_num(auc_flip) > np.nan_to_num(auc_raw):
                p_cal_val = 1.0 - p_cal_val
                p_cal     = 1.0 - p_cal
            auc_ho = roc_auc_score(y_hold_vec, p_cal_val)
        
        p_train_vec = np.asarray(p_cal, dtype=float)
        p_hold_vec  = np.asarray(p_cal_val, dtype=float)
        
        auc_train = roc_auc_score(y_train_vec, p_cal) if np.unique(y_train_vec).size > 1 else np.nan
        auc_val   = auc_ho
        brier_tr  = brier_score_loss(y_train_vec, p_cal)
        brier_val = brier_score_loss(y_hold_vec,  p_cal_val)
        
        st.write(f"🔧 Ensemble weight (logloss vs auc): w={best_w:.2f}")
        st.write(f"📉 LogLoss: train={log_loss(y_train_vec, np.clip(p_cal,eps,1-eps), labels=[0,1]):.5f}, "
                 f"val={log_loss(y_hold_vec,  np.clip(p_cal_val,eps,1-eps), labels=[0,1]):.5f}")
        st.write(f"📈 AUC:     train={auc_train if not np.isnan(auc_train) else '—':>}, "
                 f"val={auc_val if not np.isnan(auc_val) else '—':>}")
        st.write(f"🎯 Brier:   train={brier_tr:.4f},  val={brier_val:.4f}")


        
         #--- quick calibration table (for sanity; full plot optional) ---
        
        bins = np.linspace(0,1,11)
        idx  = np.digitize(p_cal_val, bins) - 1
        y_hold_series = pd.Series(y_hold_vec, index=hold_idx)
        
        cal_tbl = []
        for b in range(10):
            mask = (idx == b)
            if np.any(mask):
                avg_p = float(p_cal_val[mask].mean())
                emp   = float(y_hold_vec[mask].mean())
                cal_tbl.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                                "avg_p": avg_p, "emp_rate": emp, "n": int(mask.sum())})
        st.dataframe(pd.DataFrame(cal_tbl))

        


        # ---- headline holdout metrics ----------------------------------------------
        auc_ho = roc_auc_score(y_hold_vec, p_cal_val)
        ll_ho  = log_loss(y_hold_vec, np.clip(p_cal_val, eps, 1-eps), labels=[0,1])
        br_ho  = brier_score_loss(y_hold_vec, p_cal_val)
        
        st.markdown(f"### 🧪 Holdout Validation — `{market.upper()}` (purged-CV tuned, time-forward holdout)")
        st.write(f"- **Optimal Blend Weight (LogLoss model)**: `{best_w:.2f}`  (AUC weight `{1-best_w:.2f}`)")
        st.write(f"- **Blended + Calibrated AUC**: `{auc_ho:.4f}`")
        st.write(f"- **Holdout LogLoss**: `{ll_ho:.4f}`")
        st.write(f"- **Holdout Brier**: `{br_ho:.4f}`")

        
        # ---- sanity sentinels (flatness/entropy) -----------------------------------
        std_ens = float(np.std(p_cal))
        rng_ens = float(np.max(p_cal) - np.min(p_cal))
        entropy = -np.mean(p_cal*np.log(p_cal+1e-10) + (1-p_cal)*np.log(1-p_cal+1e-10))

        
        st.markdown(f"### 🔍 Prediction Confidence Analysis – `{market.upper()}`")
        st.write(f"- Std Dev of Predictions: `{std_ens:.4f}`")
        st.write(f"- Probability Range: `{rng_ens:.4f}`")
        st.write(f"- Avg Prediction Entropy: `{entropy:.4f}`")
        if std_ens < 0.02 or rng_ens < 0.05:
            st.error("⚠️ Predictions look too flat. Loosen min_child_weight/gamma, review constraints & feature variance.")
        
        # ---- model disagreement (diversity) ----------------------------------------
        st.markdown("### 🔀 Model Disagreement")
        corr_train  = float(np.corrcoef(p_tr_log, p_tr_auc)[0,1]) if len(p_tr_log)>1 else np.nan
        corr_hold   = float(np.corrcoef(p_ho_log, p_ho_auc)[0,1]) if len(p_ho_log)>1 else np.nan
        st.write(f"- Corr(train) LogLoss vs AUC: `{corr_train:.3f}`")
        st.write(f"- Corr(holdout) LogLoss vs AUC: `{corr_hold:.3f}`")

        
      
        # ---- calibration bins (holdout) + ROI per bin ------------------------------
        st.markdown("#### 🎯 Calibration Bins (blended + calibrated)")
        bins   = np.linspace(0, 1, 11)
        labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
        
        eval_idx = pd.Index(hold_idx)
        df_eval = pd.DataFrame({
            "p":    pd.Series(p_hold_vec, index=eval_idx),
            "y":    pd.Series(y_hold_vec, index=eval_idx),
            "odds": pd.to_numeric(df_market.loc[eval_idx, "Odds_Price"], errors="coerce"),
        }).dropna(subset=["p","y"])

        
        cuts = pd.cut(df_eval["p"], bins=bins, include_lowest=True, labels=labels)
        
        def _roi_mean_inline(sub: pd.DataFrame) -> float:
            if not sub["odds"].notna().any(): return float("nan")
            odds = pd.to_numeric(sub["odds"], errors="coerce")
            win  = sub["y"].astype(int)
            profit_pos = odds.where(odds > 0, np.nan) / 100.0
            profit_neg = 100.0 / odds.abs()
            profit_on_win = np.where(odds > 0, profit_pos, profit_neg)
            profit_on_win = pd.Series(profit_on_win, index=odds.index).fillna(0.0)
            roi = win * profit_on_win - (1 - win) * 1.0
            return float(roi.mean())
        
        rows=[]
        for lb in labels:
            sub = df_eval[cuts == lb]
            n   = int(len(sub))
            if n == 0:
                rows.append((lb, 0, np.nan, np.nan, np.nan)); continue
            hit   = float(sub["y"].mean())
            roi   = _roi_mean_inline(sub)
            avg_p = float(sub["p"].mean())
            rows.append((lb, n, hit, roi, avg_p))
        
        df_bins = pd.DataFrame(rows, columns=["Prob Bin","N","Hit Rate","Avg ROI (unit)","Avg Pred P"])
        df_bins["N"] = df_bins["N"].astype(int)
        st.dataframe(df_bins)

        # quick extremes snapshot
        hi, lo = df_bins.iloc[-1], df_bins.iloc[0]
        st.write(f"**High bin ({hi['Prob Bin']}):** " + (f"N={hi['N']}, Hit={hi['Hit Rate']:.3f}, ROI={hi['Avg ROI (unit)']:.3f}" if hi["N"]>0 else "N=0"))
        st.write(f"**Low bin  ({lo['Prob Bin']}):** " + (f"N={lo['N']}, Hit={lo['Hit Rate']:.3f}, ROI={lo['Avg ROI (unit)']:.3f}" if lo["N"]>0 else "N=0"))
        
        # ---- overfitting check (train vs holdout) ----------------------------------
        auc_tr = roc_auc_score(y_train_vec, p_cal)
        ll_tr  = log_loss(y_train_vec, np.clip(p_cal, eps, 1-eps), labels=[0,1])
        br_tr  = brier_score_loss(y_train_vec, p_cal)
        
        st.markdown("### 📉 Overfitting Check – Gap Analysis")
        st.write(f"- AUC Gap (Train - Holdout): `{(auc_tr - auc_ho):.4f}`")
        st.write(f"- LogLoss Gap (Train - Holdout): `{(ll_tr  - ll_ho):.4f}`")
        st.write(f"- Brier Gap (Train - Holdout): `{(br_tr  - br_ho):.4f}`")
     
        
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
        
        def ks_stat(y_true, p) -> float:
            fpr, tpr, _ = roc_curve(y_true, p)
            return float(np.max(tpr - fpr))

        
        # ---- calibration quality: ECE + Brier decomposition ------------------------
        # ---- calibration quality: ECE + Brier decomposition ------------------------
        ece_tr = ECE(y_train_vec, p_train_vec, 10)
        brier_tr, unc_tr, res_tr, rel_tr = brier_decomp(y_train_vec, p_train_vec, 10)
        
        ece_ho = ECE(y_hold_vec, p_hold_vec, 10)
        brier_ho, unc_ho, res_ho, rel_ho = brier_decomp(y_hold_vec, p_hold_vec, 10)

        
        st.write(
            f"- ECE(train): `{ece_tr:.4f}` | Brier(train): `{brier_tr:.4f}` = "
            f"Unc `{unc_tr:.4f}` - Res `{res_tr:.4f}` + Rel `{rel_tr:.4f}`"
        )
        st.write(
            f"- ECE(holdout): `{ece_ho:.4f}` | Brier(holdout): `{brier_ho:.4f}` = "
            f"Unc `{unc_ho:.4f}` - Res `{res_ho:.4f}` + Rel `{rel_ho:.4f}`"
        )
        
        

        # ---- Feature importance + directional sign (robust & UI‑safe) -------------------
        try:
            # 1) Derive model feature names (order matters)
            feat_in = getattr(model_auc, "feature_names_in_", None)
            if feat_in is None:
                feat_in = np.array([str(c) for c in features], dtype=object)
        
            # 2) Align to columns that exist in df_market
            available = [c for c in feat_in if c in df_market.columns]
            missing   = [c for c in feat_in if c not in df_market.columns]
        
            if len(available) == 0:
                st.error("❌ Feature-importance skipped: none of the model's features are present in df_market.")
                st.write({"model_features": list(map(str, feat_in))[:30],
                          "df_columns_sample": df_market.columns.tolist()[:30]})
            else:
                # 3) Build c numeric matrix
                X_features = (df_market[available]
                              .apply(pd.to_numeric, errors="coerce")
                              .replace([np.inf, -np.inf], np.nan)
                              .fillna(0.0)
                              .astype("float32"))
        
                if X_features.shape[0] == 0:
                    st.error("❌ Feature-importance skipped: X_features has 0 rows after cing.")
                else:
                    # 4) Importances (length must match model n_features)
                    importances = np.asarray(getattr(model_auc, "feature_importances_", None))
                    if importances is None or importances.size == 0:
                        st.error("❌ model_auc.feature_importances_ is empty. (Was the model fit?)")
                    else:
                        n_model_feats = importances.size
                        if n_model_feats != len(feat_in):
                            st.warning(
                                f"⚠️ model n_features ({n_model_feats}) != feature_names_in_ ({len(feat_in)}). "
                                "Truncating to the smaller size for display."
                            )
                        # Align lengths safely
                        k = min(n_model_feats, len(available))
                        available   = available[:k]
                        X_features  = X_features.iloc[:, :k]
                        importances = importances[:k]
        
                        # 5) Predict proba to get directional sign (safe)
                        try:
                            preds_auc = model_auc.predict_proba(X_features)[:, 1]
                        except Exception as e:
                            st.exception(RuntimeError(f"predict_proba failed on X_features: {e}"))
                            preds_auc = np.full(X_features.shape[0], np.nan, dtype=float)
        
                        # 6) Correlations → impact direction
                        corrs = []
                        finite_mask = np.isfinite(preds_auc)
                        for col in X_features.columns:
                            x = X_features[col].to_numpy()
                            m = finite_mask & np.isfinite(x)
                            if m.sum() < 3 or np.std(x[m]) <= 0:
                                corrs.append(0.0)
                            else:
                                c = np.corrcoef(x[m], preds_auc[m])
                                corrs.append(float(np.nan_to_num(c[0, 1], nan=0.0)))
                        impact = np.where(np.asarray(corrs) > 0, "↑ Increases", "↓ Decreases")
        
                        importance_df = (
                            pd.DataFrame({
                                "Feature": list(map(str, available)),
                                "Importance": importances.astype(float),
                                "Impact": impact,
                            })
                            .sort_values("Importance", ascending=False)
                            .reset_index(drop=True)
                        )
        
                        # 7) ACTIVE rows (Importance > 0)
                        active = importance_df[importance_df["Importance"] > 0].copy().reset_index(drop=True)
        
                        # ---- UI HARDENING (inline, no external helpers) ----
                        def _to_streamlit_scalar(x):
                            if x is None: return None
                            if isinstance(x, (bool, int, float, str)): return x
                            if isinstance(x, (np.bool_, np.integer, np.floating)): return x.item()
                            return str(x)
        
                        # Normalize dtypes & sanitize cells
                        if not active.empty:
                            active["Feature"]    = active["Feature"].astype("string")
                            active["Impact"]     = active["Impact"].astype("string")
                            active["Importance"] = (
                                pd.to_numeric(active["Importance"], errors="coerce")
                                  .replace([np.inf, -np.inf], np.nan)
                                  .fillna(0.0)
                                  .astype(float)
                                  .round(6)
                            )
                            active = active.applymap(_to_streamlit_scalar)
        
                        st.markdown(f"#### 📊 Feature Importance & Impact for `{market.upper()}`")
                        if active.empty:
                            st.info("No non‑zero importances to display (model very regularized or features pruned).")
                        else:
                            # Use modern width API; fall back to table if needed
                            try:
                                st.dataframe(active, width="stretch", hide_index=True)
                            except Exception:
                                st.table(active)
        
                        # Debug panel (kept lightweight)
                        with st.expander("🔧 Debug: feature‑importance inputs"):
                            st.write({
                                "missing_model_features": missing[:40],
                                "used_features": available[:40],
                                "X_features_shape": X_features.shape,
                                "any_nonfinite_preds": (not np.isfinite(preds_auc).all()),
                            })
        
        except Exception as e:
            st.error("Feature‑importance block failed safely.")
            st.exception(e)   
        # ---- calibration curve table (train) ---------------------------------------
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_train_vec, p_cal, n_bins=10)
        calib_df = pd.DataFrame({"Predicted Bin Center": prob_pred, "Actual Hit Rate": prob_true})
        st.markdown(f"#### 🎯 Calibration Bins – {market.upper()} (Train)")
        st.dataframe(calib_df)

        # ---- CV fold health ---------------------------------------------------------
        st.markdown("### 🧩 CV Fold Health")
        cv_rows=[]
        for i,(tr,va) in enumerate(folds,1):
            yv = y_train[va].astype(int)
            cv_rows.append({
                "Fold": i,
                "ValN": int(len(va)),
                "ValPosRate": float(np.mean(yv)),
                "ValBothClasses": bool(np.unique(yv).size==2)
            })
        st.dataframe(pd.DataFrame(cv_rows))


        
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

        
        # --- aliases from the blended/calibrated step ---
        ensemble_prob     = p_cal        # train/OOF blended + calibrated
        ensemble_prob_val = p_cal_val    # holdout/val blended + calibrated
        
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
            return log_loss(yv, np.clip(pv, 1e-6, 1-1e-6), labels=[0, 1])

        def _safe_brier(yv, pv):
            if len(yv) != len(pv):
                return np.nan
            return brier_score_loss(yv, np.clip(pv, 1e-6, 1-1e-6))
        # targets aligned
        y_train_vec = y_full[train_all_idx].astype(int)
        y_hold_vec  = y_full[hold_idx].astype(int)
        
        p_train_vec = np.asarray(p_cal, dtype=float)       # same length as y_train_vec
        p_hold_vec  = np.asarray(p_cal_val, dtype=float)   # same length as y_hold_vec
        
        # sanity checks
        assert len(y_train_vec) == len(p_train_vec), f"train len mismatch: y={len(y_train_vec)} p={len(p_train_vec)}"
        assert len(y_hold_vec)  == len(p_hold_vec),  f"holdout len mismatch: y={len(y_hold_vec)} p={len(p_hold_vec)}"
        acc_train = accuracy_score(y_train_vec, (p_train_vec >= 0.5).astype(int)) if np.unique(y_train_vec).size == 2 else np.nan
        acc_hold  = accuracy_score(y_hold_vec,  (p_hold_vec  >= 0.5).astype(int)) if np.unique(y_hold_vec).size  == 2 else np.nan
        
        auc_train = roc_auc_score(y_train_vec, p_train_vec)
        auc_hold  = roc_auc_score(y_hold_vec,  p_hold_vec)
        
        logloss_train = log_loss(y_train_vec, np.clip(p_train_vec, 1e-6, 1-1e-6), labels=[0,1])
        logloss_hold  = log_loss(y_hold_vec,  np.clip(p_hold_vec,  1e-6, 1-1e-6), labels=[0,1])
        
        brier_train = brier_score_loss(y_train_vec, p_train_vec)
        brier_hold  = brier_score_loss(y_hold_vec,  p_hold_vec)


        # Streamlit summary (choose what you want to display)
        # st.success(f"""✅ Trained + saved ensemble model for {market.upper()}
        # Train — AUC: {auc_train:.4f} | LogLoss: {logloss_train:.4f} | Brier: {brier_train:.4f} | Acc: {acc_train:.4f}
        # Hold  — AUC: {auc_hold:.4f}  | LogLoss: {logloss_hold:.4f}  | Brier: {brier_hold:.4f}  | Acc: {acc_hold:.4f}
        # Ensemble weight (logloss vs auc): w={best_w:.2f}
        # """)


   
        
        # --- Book reliability (build DF exactly as apply expects) ---
        bk_col = 'Bookmaker' if 'Bookmaker' in df_market.columns else 'Bookmaker_Norm'
        need   = ['Sport', 'Market', bk_col, 'SHARP_HIT_BOOL']
        
        df_rel_in = df_market.loc[:, [c for c in need if c in df_market.columns]].copy()
        if bk_col != 'Bookmaker':  # builder expects 'Bookmaker'
            df_rel_in.rename(columns={bk_col: 'Bookmaker'}, inplace=True)
        
        try:
            book_reliability_map = build_book_reliability_map(df_rel_in, prior_strength=200.0)  # ✅ DataFrame
        except Exception as e:
            logger.warning(f"book_reliability_map build failed; defaulting to empty. err={e}")
            book_reliability_map = pd.DataFrame(columns=[
                'Sport','Market','Bookmaker','Book_Reliability_Score','Book_Reliability_Lift'
            ])

   
        
        # safety check (optional but nice)
       

        _bake_feature_names_in_(model_logloss, feature_cols)
        _bake_feature_names_in_(model_auc, feature_cols)
       
        if cal_blend is None:
            cal_blend = ("iso", _IdentityIsoCal(eps=eps))
        
        iso_blend = _CalAdapter(cal_blend, clip=(CLIP, 1-CLIP))
        
        # === Save ensemble (choose one or both)
        trained_models[market] = {
                "model_logloss": model_logloss,
                "model_auc":     model_auc,
                "iso_blend":     iso_blend,
                "best_w":        float(best_w),
                "team_feature_map": team_feature_map,
                "book_reliability_map": book_reliability_map,
                "feature_cols":  feature_cols,
            
        }
        save_model_to_gcs(
            model={"model_logloss": model_logloss, "model_auc": model_auc,
                   "best_w": float(best_w), "feature_cols": feature_cols},
            calibrator=iso_blend,
            sport=sport, market=market, bucket_name=GCS_BUCKET,
            team_feature_map=team_feature_map,
            book_reliability_map=book_reliability_map,
        )
           
        

        auc = auc_hold
        acc = acc_hold
        logloss = logloss_hold
        brier = brier_hold
        
        st.success(
            f"""✅ Trained + saved ensemble model for {market.upper()}
        - AUC: {auc:.4f}
        - Accuracy: {acc:.4f}
        - Log Loss: {logloss:.4f}
        - Brier Score: {brier:.4f}
        """
        )
       
       
        pb.progress(min(100, max(0, pct)))
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
from sklearn.model_selection import StratifiedKFold

def train_timing_opportunity_model(
    sport: str = "NBA",
    days_back: int = 35,
    table_fq: str = "sharplogger.sharp_data.sharp_scores_full",
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
        
def attach_ratings_and_edges_for_diagnostics(
    df: pd.DataFrame,
    sport_aliases: dict,
    table_history: str = "sharplogger.sharp_data.ratings_current",
    project: str = "sharplogger",
    pad_days: int = 30,
    allow_forward_hours: float = 0.0,
    bq=None,                        # ✅ pass your BigQuery client in
) -> pd.DataFrame:                  # << ✅ colon here
    UI_EDGE_COLS = [
        'PR_Team_Rating','PR_Opp_Rating','PR_Rating_Diff',
        'Outcome_Model_Spread','Outcome_Market_Spread','Outcome_Spread_Edge',
        'Outcome_Cover_Prob','model_fav_vs_market_fav_agree','edge_x_k','mu_x_k'
    ]

    if df.empty:
        out = df.copy()
        for c in UI_EDGE_COLS:
            out[c] = np.nan
        return out

    out = df.copy()

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

    if 'Game_Start' not in out.columns or out['Game_Start'].isna().all():
        out['Game_Start'] = pd.to_datetime(_series('Snapshot_Timestamp', pd.NaT),
                                           utc=True, errors='coerce')

    # Only compute edges for spreads; ratings/PRs are needed for spreads logic below
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
    d_sp['Value'] = pd.to_numeric(d_sp['Value'], errors='coerce').astype('float32')

    is_current_table = 'ratings_current' in str(table_history).lower()
    pad_days_eff = (365 if is_current_table else pad_days)

    base = enrich_power_for_training_lowmem(
        df=d_sp[['Sport','Home_Team_Norm','Away_Team_Norm','Game_Start']].drop_duplicates(),
        bq=bq,
        sport_aliases=sport_aliases,
        table_history=table_history,
        pad_days=pad_days_eff,
        allow_forward_hours=allow_forward_hours,
        project=project
    )

    cons = prep_consensus_market_spread_lowmem(d_sp, value_col='Value', outcome_col='Outcome_Norm')

    game_keys = ['Sport','Home_Team_Norm','Away_Team_Norm']
    g_full = base.merge(cons, on=game_keys, how='left')
    g_fc   = favorite_centric_from_powerdiff_lowmem(g_full)

    for c in ['Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff']:
        if c not in g_fc.columns:
            g_fc[c] = g_full[c] if c in g_full.columns else np.nan

    d_map = d_sp.merge(g_fc, on=game_keys, how='left')

    is_fav_row = d_map['Outcome_Norm'].eq(d_map['Market_Favorite_Team'])
    d_map['Outcome_Model_Spread']  = np.where(is_fav_row, d_map['Model_Fav_Spread'], d_map['Model_Dog_Spread']).astype('float32')
    d_map['Outcome_Market_Spread'] = np.where(is_fav_row, d_map['Favorite_Market_Spread'], d_map['Underdog_Market_Spread']).astype('float32')
    d_map['Outcome_Spread_Edge']   = np.where(is_fav_row, d_map['Fav_Edge_Pts'], d_map['Dog_Edge_Pts']).astype('float32')
    d_map['Outcome_Cover_Prob']    = np.where(is_fav_row, d_map['Fav_Cover_Prob'], d_map['Dog_Cover_Prob']).astype('float32')

    is_home_bet = d_map['Outcome_Norm'].eq(d_map['Home_Team_Norm'])
    d_map['PR_Team_Rating'] = np.where(is_home_bet, d_map['Home_Power_Rating'], d_map['Away_Power_Rating']).astype('float32')
    d_map['PR_Opp_Rating']  = np.where(is_home_bet, d_map['Away_Power_Rating'], d_map['Home_Power_Rating']).astype('float32')
    d_map['PR_Rating_Diff'] = (
        pd.to_numeric(d_map['PR_Team_Rating'], errors='coerce') -
        pd.to_numeric(d_map['PR_Opp_Rating'],  errors='coerce')
    ).astype('float32')

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
    active_feats = []
    try:
        if (bundle is not None) and (model is not None) and ('attach_why_model_likes_it' in globals()):
            df = attach_why_model_likes_it(df, bundle, model)
            active_feats = list(getattr(df, "attrs", {}).get("active_features_used", []))
        else:
            df["Why Model Likes It"] = "⚠️ No model context"
            df["Why_Feature_Count"] = 0
            df.attrs["active_features_used"] = []
    except Exception as e:
        df["Why Model Likes It"] = f"⚠️ Explainer failed: {e}"
        df["Why_Feature_Count"] = 0
        df.attrs["active_features_used"] = []

    # --- 7) Model-vs-Market seasoning (only if those cols were active in the model) ---
    ACTIVE_FEATS = set(active_feats)

   
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
            _pr_diff = _num(row, 'PR_Rating_Diff')
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
    timing_models = {}
    if sport and gcs_bucket and ('load_model_from_gcs' in globals()):
        for _m in MARKETS:
            try:
                pl = load_model_from_gcs(sport=sport, market=f"timing_{_m}", bucket_name=gcs_bucket) or {}
                mdl = pl.get("model") or pl.get("calibrator") or None
                cols = pl.get("feature_list")
                timing_models[_m] = {"mdl": mdl, "cols": cols}
            except Exception:
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

        mdl, cols = pack["mdl"], pack["cols"]
        X = df.loc[mask, timing_feature_superset].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        if mdl is None:
            df.loc[mask, 'Timing_Opportunity_Score'] = _heuristic_score(df.loc[mask])
            continue

        X = _align_X_for_model(X, mdl, cols)
        try:
            p = (mdl.predict_proba(X)[:, 1] if hasattr(mdl, 'predict_proba')
                 else np.clip(mdl.predict(X), 0, 1))
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
    keep_cols = [c for c in base_cols if c in df.columns] + active_cols_present

    diagnostics_df = df[keep_cols].rename(columns={'Tier_Change':'Tier Δ'})
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

    bundle = {
        "model_logloss": data.get("model_logloss"),
        "model_auc":     data.get("model_auc"),
        "iso_blend":     data.get("iso_blend"),
        "best_w":        float(data.get("best_w", 1.0)),
        "feature_cols":  data.get("feature_cols") or [],
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



def render_scanner_tab(label, sport_key, container, force_reload=False):
    if st.session_state.get("pause_refresh", False):
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
                    ratings_df = enrich_power_for_training_lowmem(
                        df=diag_rows[['Sport','Home_Team_Norm','Away_Team_Norm','Game_Start']].drop_duplicates(),
                        bq=bq_client,                                  # your BigQuery client
                        sport_aliases=sport_aliases,                   # pass if you have it
                        table_history="sharplogger.sharp_data.ratings_current",
                        pad_days=365,
                        allow_forward_hours=24*365
                    )
                    diag_rows = diag_rows.merge(
                        ratings_df[['Sport','Home_Team_Norm','Away_Team_Norm',
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
            
    
            table_df = summary_grouped[view_cols].copy()
            table_html = table_df.to_html(classes="custom-table", index=False, escape=False)
            
            # ✅ Safe id; visible emoji stays only in text, not in ids/classes
            safe_id = f"tbl-{_title_key}"
            st.markdown(
                f"<div id='{safe_id}' class='scrollable-table-container'>{table_html}</div>",
                unsafe_allow_html=True
            )
            
            st.success("✅ Finished rendering sharp picks table.")
            st.caption(f"Showing {len(table_df)} rows")
            # === Render Sharp Picks Table (HTML Version)
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
        
        
