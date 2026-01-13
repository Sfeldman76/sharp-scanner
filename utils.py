# --- Core Python ---
import os, sys, gc, time, json, math, hashlib, logging, pickle, warnings, traceback
from io import BytesIO
from collections import defaultdict, Counter
from functools import lru_cache
from typing import Optional, Callable, Iterable
from datetime import datetime
import datetime as dt

# --- Data Science ---

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil
import requests
import os, psutil, gc, time
# --- Cloud (BigQuery / GCS) ---
from google.cloud import bigquery, storage, bigquery_storage
from pandas_gbq import to_gbq

# --- Sklearn / ML ---x
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    TimeSeriesSplit,
    BaseCrossValidator,     # <-- needed for custom splitter
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.isotonic import IsotonicRegression  # optional; safe to keep
import math
from typing import Iterable, Optional
# top of file


import numpy as np
try:
    import numpy.special as _nps
    _erf = _nps.erf
except Exception:
    # Slow fallback, but safe and dependency-free
    from math import erf as _math_erf
    _erf = np.vectorize(_math_erf, otypes=[np.float64])

def _phi(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))



import pandas as pd
from google.cloud import bigquery, bigquery_storage
import pandas as pd, datetime as dt, gc, pyarrow as pa
import cloudpickle as cp
import gzip
from google.cloud import storage

# --- XGBoost ---
import xgboost as xgb
from xgboost import XGBClassifier

# --- Streamlit (dashboard) ---
import streamlit as st
# reuse the same helpers you used in training:

# --- Pandas dtype helpers ---
from pandas.api.types import is_categorical_dtype, is_string_dtype

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔇 Suppress scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# 🔇 Suppress XGBoost C++ backend warnings
os.environ["XGBOOST_VERBOSITY"] = "0"

# === Config ===
GCP_PROJECT_ID = "sharplogger"
BQ_DATASET = "sharp_data"
BQ_TABLE = "sharp_moves_master"
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
MARKET_WEIGHTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.market_weights"
#LINE_HISTORY_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.line_history_master"
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
    "NCAAF": "americanfootball_ncaaf",
    "NFL": "americanfootball_nfl", 
    "NCAAB": "basketball_ncaab",  # ✅ was "basketball_NCAAB"
   
}

SHARP_BOOKS_FOR_LIMITS = ['pinnacle']
SHARP_BOOKS = SHARP_BOOKS_FOR_LIMITS + ['betus','mybookieag','smarkets','betfair_ex_eu','betfair_ex_uk','betfair_ex_au','lowvig','betonlineag','matchbook','sport888' ]

REC_BOOKS = [
    'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers',
    'fanatics', 'espnbet', 'hardrockbet','sport888', 'fanatics', 'bovada', 'bet365', 'williamhillus', 'ballybet', 'bet365_au','betopenly']


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
        odds = float(odds)
        if pd.isna(odds):
            return None
        if odds < 0:
            return -odds / (-odds + 100)
        else:
            return 100 / (odds + 100)
    except:
        return None

def calc_implied_prob(odds):
    try:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    except Exception:
        return None


def implied_prob_vec_raw(arr: np.ndarray) -> np.ndarray:
    p = np.full(arr.shape, np.nan, dtype='float64')
    neg = arr < 0
    pos = ~neg & np.isfinite(arr)
    ao = -arr
    p[neg] = ao[neg] / (ao[neg] + 100.0)
    p[pos] = 100.0 / (arr[pos] + 100.0)
    return p


def ensure_columns(df, required_cols, fill_value=None):
    for col in required_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df

def normalize_book_name(bookmaker: str, book: str) -> str:
    book = book.lower().strip() if isinstance(book, str) else ""
    bookmaker = bookmaker.lower().strip() if isinstance(bookmaker, str) else ""

    if bookmaker == "betfair" and book.startswith("betfair_ex_"):
        return book  # e.g., betfair_ex_uk

    return bookmaker


def _levels_to_jsonlike(x):
    # NA/None -> "[]"
    if x is None or (x is pd.NA) or pd.isna(x):
        return "[]"
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, set, np.ndarray)):
        try:
            return json.dumps(list(x))
        except Exception:
            return str(list(x))
    try:
        return json.dumps([x])
    except Exception:
        return str([x])

def _phi(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + nps.erf(x / np.sqrt(2.0)))

BQS = None  # lazy init to avoid global mem

def _bqs():
    global BQS
    if BQS is None:
        BQS = bigquery_storage.BigQueryReadClient()
    return BQS

def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64","int64"]).columns:
        if df[c].dtype.kind == "f":
            df[c] = pd.to_numeric(df[c], downcast="float")
        else:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df

def to_cats(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().astype("category")
    return df



def _bqs_singleton():
    if not hasattr(_bqs_singleton, "_c"):
        _bqs_singleton._c = bigquery_storage.BigQueryReadClient()
    return _bqs_singleton._c


def coerce_for_bq(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Strings (must be bytes/str for Arrow)
    str_cols = [
        'Game_Key','Game','Sport','Market','Outcome','Bookmaker','Book','Event_Date',
        'Home_Team_Norm','Away_Team_Norm','Market_Norm','Outcome_Norm','Merge_Key_Short',
        'Line_Hash','SHARP_HIT_BOOL','SHARP_COVER_RESULT','SupportKey','SharpMove_Timing_Dominant',
        'Team_Key','Scored'  # BQ=STRING but sometimes Int64 in pandas
    ]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype('string')
            df[c] = df[c].where(df[c].notna(), None)

    # 2) Floats (object → float)
    for c in ['Delta_vs_Sharp','Sharp_Time_Score','Open_Book_Value',
              'Final_Confidence_Score','Sharp_Move_Magnitude_Score']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')

    # 3) Integers that might be floats (round then Int64 nullable)
    for c in ['Direction_Aligned','Late_Game_Steam_Flag','SharpMove_Odds_Up','SharpMove_Odds_Down']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').round().astype('Int64')

    # 4) Booleans stored as numbers/ints
    for c in ['Market_Leader','Is_Reinforced_MultiMarket','Scored_By_Model']:
        if c in df.columns:
            ser = pd.to_numeric(df[c], errors='coerce')
            if ser.notna().any():
                df[c] = ser.astype('Int64').astype('boolean')
            else:
                df[c] = df[c].astype('boolean')

    return df
def _to_bq_params(params: dict | None):
    if not params:
        return None
    qps = []
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            if not v:
                qps.append(bigquery.ArrayQueryParameter(k, "STRING", []))
            else:
                x = v[0]
                if isinstance(x, str):
                    qps.append(bigquery.ArrayQueryParameter(k, "STRING", list(v)))
                elif isinstance(x, int):
                    qps.append(bigquery.ArrayQueryParameter(k, "INT64", list(v)))
                elif isinstance(x, float):
                    qps.append(bigquery.ArrayQueryParameter(k, "FLOAT64", list(v)))
                elif isinstance(x, (dt.datetime, pd.Timestamp)):
                    vv = []
                    for t in v:
                        t = pd.to_datetime(t, errors="coerce")
                        t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
                        vv.append(t.to_pydatetime())
                    qps.append(bigquery.ArrayQueryParameter(k, "TIMESTAMP", vv))
                else:
                    qps.append(bigquery.ArrayQueryParameter(k, "STRING", [str(y) for y in v]))
        else:
            if isinstance(v, str):
                qps.append(bigquery.ScalarQueryParameter(k, "STRING", v))
            elif isinstance(v, bool):
                qps.append(bigquery.ScalarQueryParameter(k, "BOOL", v))
            elif isinstance(v, int):
                qps.append(bigquery.ScalarQueryParameter(k, "INT64", v))
            elif isinstance(v, float):
                qps.append(bigquery.ScalarQueryParameter(k, "FLOAT64", v))
            elif isinstance(v, (dt.datetime, pd.Timestamp)):
                t = pd.to_datetime(v, errors="coerce")
                t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
                qps.append(bigquery.ScalarQueryParameter(k, "TIMESTAMP", t.to_pydatetime()))
            elif v is None:
                qps.append(bigquery.ScalarQueryParameter(k, "STRING", None))
            else:
                qps.append(bigquery.ScalarQueryParameter(k, "STRING", str(v)))
    return qps

def stream_query_dfs(
    bq: bigquery.Client,
    sql: str,
    params: dict | None = None,
    *,
    page_rows: int = 200_000,
    select_cols: list[str] | None = None,
    dtypes: dict | None = None,
):
    # Build job config with typed params
    qps = _to_bq_params(params)
    job_config = bigquery.QueryJobConfig(query_parameters=qps) if qps else None

    # Run query
    it = bq.query(sql, job_config=job_config).result(page_size=page_rows)

    # Stream DataFrames using the Storage API
    bqs = _bqs_singleton()
    for df in it.to_dataframe_iterable(bqstorage_client=bqs):
        if select_cols:
            keep = [c for c in select_cols if c in df.columns]
            if keep:
                df = df[keep]
        if dtypes:
            df = df.astype(dtypes, copy=False)
        yield df
        del df
        gc.collect()

def to_utc_ts(x):
    """
    Return a pandas Timestamp in UTC.
    - None/NaT -> NaT
    - Naive -> tz_localize('UTC')
    - Aware -> tz_convert('UTC')
    """
    if x is None or (hasattr(pd, "isna") and pd.isna(x)):
        return pd.NaT
    t = pd.to_datetime(x, errors="coerce")
    if t is None or pd.isna(t):
        return pd.NaT
    # pandas Timestamp has .tz / .tzinfo depending on version; handle both
    if getattr(t, "tzinfo", None) is None and getattr(t, "tz", None) is None:
        return t.tz_localize("UTC")
    else:
        return t.tz_convert("UTC")
# --- add this helper near your other SQL helpers ---
def _table_has_column(bq: bigquery.Client, full_table: str, column: str) -> bool:
    """
    full_table is like 'project.dataset.table' or 'dataset.table'
    """
    parts = full_table.split(".")
    if len(parts) == 3:
        project_id, dataset_id, table_id = parts
    elif len(parts) == 2:
        # assume default project
        project_id = None
        dataset_id, table_id = parts
    else:
        return False

    if project_id:
        from_clause = f"`{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`"
        tbl_filter = f"table_catalog = @proj AND table_schema = @ds AND table_name = @tbl"
        params = [
            bigquery.ScalarQueryParameter("proj", "STRING", project_id),
            bigquery.ScalarQueryParameter("ds",   "STRING", dataset_id),
            bigquery.ScalarQueryParameter("tbl",  "STRING", table_id),
            bigquery.ScalarQueryParameter("col",  "STRING", column),
        ]
    else:
        from_clause = f"`{dataset_id}.INFORMATION_SCHEMA.COLUMNS`"
        tbl_filter = f"table_schema = @ds AND table_name = @tbl"
        params = [
            bigquery.ScalarQueryParameter("ds",   "STRING", dataset_id),
            bigquery.ScalarQueryParameter("tbl",  "STRING", table_id),
            bigquery.ScalarQueryParameter("col",  "STRING", column),
        ]

    sql = f"""
    SELECT COUNT(1) AS n
    FROM {from_clause}
    WHERE {tbl_filter} AND column_name = @col
    """
    df = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
    return (not df.empty) and (int(df.n.iloc[0]) > 0)



# ---------------------------------------------------------------------
# Sport aliases
# ---------------------------------------------------------------------

SPORT_ALIASES = {
    "MLB":   ["MLB", "BASEBALL_MLB", "BASEBALL-MLB", "BASEBALL"],
    "NFL":   ["NFL", "AMERICANFOOTBALL_NFL", "FOOTBALL_NFL"],
    "NCAAF": ["NCAAF", "AMERICANFOOTBALL_NCAAF", "FOOTBALL_NCAAF", "CFB", "COLLEGE_FOOTBALL"],
    "NBA":   ["NBA", "BASKETBALL_NBA"],
    "WNBA":  ["WNBA", "BASKETBALL_WNBA"],
    "CFL":   ["CFL", "CANADIANFOOTBALL", "CANADIAN_FOOTBALL"],
    "NCAAB": ["NCAAB", "BASKETBALL_NCAAB", "BASKETBALL_NCAAM", "NCAAM", "COLLEGE_BASKETBALL"],
}


# ---------------------------------------------------------------------
# Main power-ratings updater
# ---------------------------------------------------------------------

def update_power_ratings(
    bq: bigquery.Client,
    *,
    project_table_scores: str = "sharplogger.sharp_data.game_scores_final",
    table_history: str = "sharplogger.sharp_data.ratings_history",
    table_current: str = "sharplogger.sharp_data.ratings_current",
    default_sport: str = "MLB",
    project_table_market: str | None = "sharplogger.sharp_data.moves_with_features_merged",
) -> pd.DataFrame:
    """
    Streamed, low-memory update of team power ratings per sport.

      - MLB   -> Poisson/Skellam-style attack+defense (scores-only)
      - NFL   -> Kalman/DLM on point margin + market spread sensor (+ SoS)
      - NCAAF -> Kalman/DLM on point margin + market spread sensor (+ SoS)
      - NBA   -> Kalman/DLM on point margin + market spread sensor (+ SoS)
      - WNBA  -> Kalman/DLM on point margin + market spread sensor (+ SoS)
      - CFL   -> Kalman/DLM on point margin + market spread sensor (+ SoS)
      - NCAAB -> Ridge-Massey (scores-only, with sharp-spread calibration)

    Notes:
      • Ratings are stored as 1500 + points_rating so existing consumers keep working.
      • Method values remain: 'poisson', 'elo_kalman', 'ridge_massey'.
      • Downstream logic that relies on rating DIFFERENCES is unchanged.

    Assumes helper functions exist in this module:
      - to_utc_ts(ts) -> pd.Timestamp in UTC
      - stream_query_dfs(bq, sql, params, page_rows, select_cols)
      - downcast_numeric(df)
      - to_cats(df, cols)
    """

    # ---------- config ----------
    SPORT_CFG = {
        "MLB": dict(model="poisson", HFA_pts=0.20, mov_cap=None),

        # --------------------------
        # 🏈 NFL / NCAAF
        # --------------------------
        "NFL": dict(
            model="elo_kalman_offdef",
            HFA_pts=2.1,
            mov_cap=24,
            phi=0.96,
            sigma_eta=6.0,
            sigma_pts=12.0,
            sigma_spread=6.0,
            market_lambda=0.35,
            favored_only_market=True,
            sos_half_life_days=90,
            sos_gamma=0.7,
            season_gap_days=120,
            season_rho=0.75,
            blowout_mov=17,
            late_month_start=11,
            blowout_var_mult=2.25,
        ),

        "NCAAF": dict(
            model="elo_kalman_offdef",
            HFA_pts=2.6,
            mov_cap=28,
            phi=0.96,
            sigma_eta=7.0,
            sigma_pts=13.0,
            sigma_spread=7.0,
            market_lambda=0.30,
            favored_only_market=True,
            sos_half_life_days=90,
            sos_gamma=0.7,
            season_gap_days=150,
            season_rho=0.62,
            blowout_mov=21,
            late_month_start=11,
            blowout_var_mult=2.50,
        ),

        # --------------------------
        # 🏀 NBA / WNBA
        # --------------------------
        "NBA": dict(
            model="elo_kalman_offdef",
            HFA_pts=2.8,
            mov_cap=24,
            phi=0.97,
            sigma_eta=7.5,
            sigma_pts=12.0,
            sigma_spread=10.0,
            market_lambda=0.35,
            favored_only_market=False,
            sos_half_life_days=60,
            sos_gamma=0.6,
            season_gap_days=120,
            season_rho=0.82,
            blowout_mov=22,
            late_month_start=3,
            blowout_var_mult=1.75,
            b2b_sigma_pts_mult=1.20,
        ),

        "WNBA": dict(
            model="elo_kalman_offdef",
            HFA_pts=2.0,
            mov_cap=24,
            phi=0.97,
            sigma_eta=7.0,
            sigma_pts=11.5,
            sigma_spread=9.0,
            market_lambda=0.35,
            favored_only_market=False,
            sos_half_life_days=60,
            sos_gamma=0.6,
            season_gap_days=200,
            season_rho=0.70,
            blowout_mov=18,
            late_month_start=7,
            blowout_var_mult=2.00,
            b2b_sigma_pts_mult=1.20,
        ),

        "CFL": dict(
            model="elo_kalman_offdef",
            HFA_pts=1.6,
            mov_cap=30,
            phi=0.96,
            sigma_eta=6.5,
            sigma_pts=13.5,
            sigma_spread=7.5,
            market_lambda=0.30,
            favored_only_market=True,
            sos_half_life_days=90,
            sos_gamma=0.7,
            season_gap_days=180,
            season_rho=0.65,
            blowout_mov=20,
            late_month_start=9,
            blowout_var_mult=2.10,
        ),

        # --------------------------
        # 🏀 NCAAB Ridge-Massey
        # --------------------------
        "NCAAB": dict(
            model="ridge_massey",
            HFA_pts=3.0,
            mov_cap=25,
            ridge_lambda=65.0,
            window_days=90,
            phase_aware_window=True,
            rm_half_life_days=35.0,
            preseason_prior_games=18,
            use_hfa_term=True,
            spread_calib_clip_abs=15.0,
            spread_calib_floor_abs=1.0,
            sos_gamma=0.25,         # small! prevents double-count
            sos_use_recency=True,   # use same half-life weights
            sos_early_only=False,   # set True if you want only Nov/Dec

        ),
    }

    BACKFILL_DAYS = 365

    PREFERRED_METHOD = {
        "MLB":   "poisson",
        "NFL":   "elo_kalman",
        "NCAAF": "elo_kalman",
        "NBA":   "elo_kalman",
        "WNBA":  "elo_kalman",
        "CFL":   "elo_kalman",
        "NCAAB": "ridge_massey",
    }

    # Use sharp books only for composite close
    SHARP_BOOKS_DEFAULT = ["pinnacle", "betfair_ex"]

    def get_aliases(canon: str) -> list[str]:
        return SPORT_ALIASES.get(canon.upper(), [canon.upper()])

    # ---------------- seed + state helpers ----------------

    def load_seed_ratings(sport: str, asof_ts: dt.datetime | pd.Timestamp | None) -> dict[str, float]:
        if asof_ts is None:
            cur = bq.query(
                f"SELECT Team, Rating FROM `{table_current}` WHERE UPPER(Sport) = @sport",
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport.upper())]
                )
            ).to_dataframe()
            if not cur.empty:
                return dict(zip(cur.Team, cur.Rating))
            hist = bq.query(
                f"""
                SELECT Team, ANY_VALUE(Rating) AS Rating
                FROM (
                  SELECT Team, Rating,
                         ROW_NUMBER() OVER (PARTITION BY Team ORDER BY Updated_At DESC) AS rn
                  FROM `{table_history}` WHERE UPPER(Sport) = @sport
                )
                WHERE rn = 1 GROUP BY Team
                """,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport.upper())]
                )
            ).to_dataframe()
            return dict(zip(hist.Team, hist.Rating)) if not hist.empty else {}

        # normalize tz
        if isinstance(asof_ts, pd.Timestamp):
            asof_ts = (asof_ts.tz_localize("UTC") if asof_ts.tzinfo is None else asof_ts.tz_convert("UTC")).to_pydatetime()
        elif isinstance(asof_ts, dt.datetime) and asof_ts.tzinfo is None:
            asof_ts = asof_ts.replace(tzinfo=dt.timezone.utc)

        df = bq.query(
            f"""
            WITH latest AS (
              SELECT Sport, Team, Rating, Updated_At,
                     ROW_NUMBER() OVER (PARTITION BY Team ORDER BY Updated_At DESC) AS rn
              FROM `{table_history}`
              WHERE UPPER(Sport) = @sport AND Updated_At <= @asof
            )
            SELECT Team, Rating FROM latest WHERE rn = 1
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("sport", "STRING", sport.upper()),
                    bigquery.ScalarQueryParameter("asof",  "TIMESTAMP", asof_ts),
                ]
            )
        ).to_dataframe()
        return dict(zip(df.Team, df.Rating))

    def upsert_current(rows: list[dict]):
        if not rows:
            return
        df_cur = pd.DataFrame(rows)
        stage = table_current.rsplit(".", 1)[0] + "._ratings_current_stage"
        bq.load_table_from_dataframe(df_cur, stage).result()
        bq.query(f"""
        MERGE `{table_current}` T
        USING `{stage}` S
        ON T.Sport = S.Sport AND T.Team = S.Team
        WHEN MATCHED THEN UPDATE SET
          T.Rating     = S.Rating,
          T.Method     = S.Method,
          T.Updated_At = S.Updated_At
        WHEN NOT MATCHED THEN INSERT (Sport, Team, Rating, Method, Updated_At)
          VALUES (S.Sport, S.Team, S.Rating, S.Method, S.Updated_At)
        """).result()
        bq.query(f"DROP TABLE `{stage}`").result()

    def fetch_sports_present() -> list[str]:
        rows = bq.query(f"""
            SELECT DISTINCT UPPER(CAST(Sport AS STRING)) AS s
            FROM `{project_table_scores}`
            WHERE Score_Home_Score IS NOT NULL AND Score_Away_Score IS NOT NULL
        """).to_dataframe()
        present = set()
        if not rows.empty:
            seen = {str(s).upper() for s in rows.s.dropna().tolist()}
            for canon in SPORT_CFG.keys():
                aliases = set(get_aliases(canon))
                if seen & aliases:
                    present.add(canon)
        return sorted(present)

    def get_last_ts(sport: str):
        df = bq.query(
            f"SELECT MAX(Updated_At) AS last_ts FROM `{table_history}` WHERE UPPER(Sport) = @sport",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport.upper())]
            ),
        ).to_dataframe()
        if df.empty:
            return None
        ts = df.last_ts.iloc[0]
        if ts is None or (hasattr(pd, "isna") and pd.isna(ts)):
            return None
        if isinstance(ts, pd.Timestamp):
            return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if isinstance(ts, dt.datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
        return None

    def has_new_finals_since(sport: str, last_ts):
        aliases = get_aliases(sport)
        cutoff_check = None
        if last_ts is not None:
            lt_utc = to_utc_ts(last_ts)
            if not pd.isna(lt_utc):
                cutoff_check = (lt_utc - pd.Timedelta(days=BACKFILL_DAYS)).to_pydatetime()
        sql = f"""
          SELECT COUNT(*) AS n
          FROM `{project_table_scores}`
          WHERE UPPER(CAST(Sport AS STRING)) IN UNNEST(@sport_aliases)
            AND Score_Home_Score IS NOT NULL AND Score_Away_Score IS NOT NULL
            {'' if cutoff_check is None else 'AND TIMESTAMP(Inserted_Timestamp) >= @cutoff'}
          LIMIT 1
        """
        params = [bigquery.ArrayQueryParameter("sport_aliases", "STRING", aliases)]
        if cutoff_check is not None:
            params.append(bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", cutoff_check))
        n = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe().n.iloc[0]
        return n > 0

    def sports_present_in_history() -> list[str]:
        q = f"SELECT DISTINCT UPPER(Sport) AS s FROM `{table_history}`"
        df = bq.query(q).to_dataframe()
        return df.s.dropna().str.upper().tolist()

    def fill_missing_current_from_history(sports: list[str] | None = None):
        if not sports:
            hist = set(sports_present_in_history())
            sports = sorted(hist & set(SPORT_CFG.keys()))
        for sport in sports:
            sql = f"""
            MERGE `{table_current}` T
            USING (
              WITH latest AS (
                SELECT
                  Sport, Team, Method, Rating, Updated_At,
                  ROW_NUMBER() OVER (
                    PARTITION BY Sport, Team, Method
                    ORDER BY Updated_At DESC
                  ) AS rn
                FROM `{table_history}`
                WHERE UPPER(Sport) = @sport
              )
              SELECT Sport, Team, Method, Rating, Updated_At
              FROM latest
              WHERE rn = 1
            ) S
            ON T.Sport = S.Sport AND T.Team = S.Team AND T.Method = S.Method
            WHEN NOT MATCHED THEN
              INSERT (Sport, Team, Method, Rating, Updated_At)
              VALUES (S.Sport, S.Team, S.Method, S.Rating, S.Updated_At)
            """
            bq.query(
                sql,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport.upper())]
                ),
            ).result()

    # ---------------- streaming games ----------------

    def _get_game_time_bounds(sport_aliases: list[str], cutoff_param):
        sql = f"""
        SELECT
          MIN(TIMESTAMP(Game_Start)) AS min_ts,
          MAX(TIMESTAMP(Game_Start)) AS max_ts
        FROM `{project_table_scores}`
        WHERE UPPER(CAST(Sport AS STRING)) IN UNNEST(@sport_aliases)
          AND Score_Home_Score IS NOT NULL AND Score_Away_Score IS NOT NULL
          {"AND TIMESTAMP(Inserted_Timestamp) >= @cutoff" if cutoff_param else ""}
        """
        df = bq.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("sport_aliases", "STRING", sport_aliases),
                    *( [bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", cutoff_param)] if cutoff_param else [] )
                ]
            )
        ).to_dataframe()
        return df.min_ts.iloc[0], df.max_ts.iloc[0]

    def _exp_decay_weight(gs_ts: pd.Timestamp, now_ts: pd.Timestamp, half_life_days: float) -> float:
        age_days = max((now_ts - gs_ts).days, 0)
        return float(np.exp(- age_days / max(half_life_days, 1.0)))

    def _safe_get(d: dict, k, default=0.0):
        v = d.get(k, default)
        return float(v if np.isfinite(v) else default)

    def load_games_stream(
        sport: str,
        aliases: list[str],
        cutoff=None,
        page_rows: int = 200_000,
        *,
        project_table_market: str | None = None,
        sharp_books: list[str] | None = None,
    ):
        cutoff_param = None
        if cutoff is not None and not pd.isna(cutoff):
            cutoff_utc = to_utc_ts(cutoff)
            if not pd.isna(cutoff_utc):
                cutoff_param = cutoff_utc.to_pydatetime()

        base_scores_sql = f"""
        SELECT
          CASE
            WHEN UPPER(CAST(t.Sport AS STRING)) IN ('BASEBALL_MLB','BASEBALL-MLB','BASEBALL','MLB') THEN 'MLB'
            WHEN UPPER(CAST(t.Sport AS STRING)) IN ('AMERICANFOOTBALL_NFL','FOOTBALL_NFL','NFL') THEN 'NFL'
            WHEN UPPER(CAST(t.Sport AS STRING)) IN ('AMERICANFOOTBALL_NCAAF','FOOTBALL_NCAAF','NCAAF','CFB','COLLEGE_FOOTBALL') THEN 'NCAAF'
            WHEN UPPER(CAST(t.Sport AS STRING)) IN ('BASKETBALL_NBA','NBA') THEN 'NBA'
            WHEN UPPER(CAST(t.Sport AS STRING)) IN ('BASKETBALL_WNBA','WNBA') THEN 'WNBA'
            WHEN UPPER(CAST(t.Sport AS STRING)) IN ('BASKETBALL_NCAAB','BASKETBALL_NCAAM','NCAAB','NCAAM','COLLEGE_BASKETBALL') THEN 'NCAAB'
            WHEN UPPER(CAST(t.Sport AS STRING)) IN ('CFL','CANADIANFOOTBALL','CANADIAN_FOOTBALL') THEN 'CFL'
            ELSE COALESCE(CAST(t.Sport AS STRING), @default_sport)
          END AS Sport,
          LOWER(TRIM(t.Home_Team)) AS Home_Team,
          LOWER(TRIM(t.Away_Team)) AS Away_Team,
          TIMESTAMP(t.Game_Start) AS Game_Start,
          SAFE_CAST(t.Score_Home_Score AS FLOAT64) AS Score_Home_Score,
          SAFE_CAST(t.Score_Away_Score AS FLOAT64) AS Score_Away_Score,
          TIMESTAMP(t.Inserted_Timestamp) AS Snapshot_TS
        FROM `{project_table_scores}` t
        WHERE UPPER(CAST(t.Sport AS STRING)) IN UNNEST(@sport_aliases)
          AND t.Score_Home_Score IS NOT NULL AND t.Score_Away_Score IS NOT NULL
          {"AND TIMESTAMP(t.Inserted_Timestamp) >= @cutoff" if cutoff_param else ""}
        """

        params: Dict[str, object] = {"sport_aliases": aliases, "default_sport": default_sport}
        if cutoff_param is not None:
            params["cutoff"] = cutoff_param

        use_market = project_table_market is not None
        sharp_list = (sharp_books or SHARP_BOOKS_DEFAULT)

        if use_market:
            sql_to_run = f"""
            WITH scores AS ({base_scores_sql}),

            latest_per_book AS (
              SELECT
                LOWER(TRIM(m.Home_Team_Norm)) AS Home_Team,
                LOWER(TRIM(m.Away_Team_Norm)) AS Away_Team,
                TIMESTAMP(m.Game_Start)       AS Game_Start,

                REGEXP_REPLACE(
                  LOWER(TRIM(COALESCE(m.Bookmaker, m.Book))),
                  r'^betfair_ex(?:_[a-z]+)?$', 'betfair_ex'
                ) AS BookKey,

                -- Convert team-based Value to HOME-based spread:
                SAFE_CAST(
                  CASE
                    WHEN LOWER(TRIM(m.Team_For_Join)) = LOWER(TRIM(m.Home_Team_Norm))
                      THEN m.Value           -- row is home team → this is home spread
                    ELSE
                      -m.Value               -- row is away team → flip sign
                  END
                  AS FLOAT64
                ) AS SpreadValue,

                m.Snapshot_Timestamp
              FROM `{project_table_market}` m
              WHERE
                UPPER(CAST(m.Sport AS STRING)) IN UNNEST(@sport_aliases)
                AND m.Market IN ('spreads','spread','Spread','SPREADS')
                AND m.Value IS NOT NULL
                AND TIMESTAMP(m.Game_Start) IS NOT NULL
              QUALIFY ROW_NUMBER() OVER (
                PARTITION BY
                  LOWER(TRIM(m.Home_Team_Norm)),
                  LOWER(TRIM(m.Away_Team_Norm)),
                  TIMESTAMP(m.Game_Start),
                  REGEXP_REPLACE(
                    LOWER(TRIM(COALESCE(m.Bookmaker, m.Book))),
                    r'^betfair_ex(?:_[a-z]+)?$', 'betfair_ex'
                  )
                ORDER BY m.Snapshot_Timestamp DESC
              ) = 1
            ),

            market AS (
              SELECT
                Home_Team,
                Away_Team,
                Game_Start,
                APPROX_QUANTILES(SpreadValue, 2)[OFFSET(1)] AS Spread_Close
              FROM latest_per_book
              WHERE BookKey IN UNNEST(@sharp_books)
              GROUP BY Home_Team, Away_Team, Game_Start
            )

            SELECT s.*, m.Spread_Close
            FROM scores s
            LEFT JOIN market m
              USING (Home_Team, Away_Team, Game_Start)
            ORDER BY s.Snapshot_TS, s.Game_Start
            """

            params["sharp_books"] = [b.lower() for b in sharp_list]
        else:
            sql_to_run = base_scores_sql + "\nORDER BY Snapshot_TS, Game_Start"

        for df in stream_query_dfs(bq, sql_to_run, params=params, page_rows=page_rows, select_cols=None):
            if "Spread_Close" not in df.columns:
                df["Spread_Close"] = np.nan
            df = df[[
                "Sport",
                "Home_Team",
                "Away_Team",
                "Game_Start",
                "Snapshot_TS",
                "Score_Home_Score",
                "Score_Away_Score",
                "Spread_Close",
            ]]
            df = downcast_numeric(df)
            df = to_cats(df, ["Sport", "Home_Team", "Away_Team"])
            yield df
            del df
            gc.collect()

    # ---------------- engines ----------------

    def _cap_margin(mov, cap):
        if cap is None:
            return float(mov)
        return float(min(max(mov, -cap), cap))

 
    def run_kalman_offdef(bq, sport: str, aliases: list[str], cfg: dict, window_start):
        """
        Off/Def Kalman-Elo:
          - Maintains Off and Def states per team
          - Uses points sensors (home points, away points)
          - Optional market spread sensor, optionally favored-side only
          - Late-season blowout downweight
          - Explicit season reset (gap-based shrink)
          - SoS correction applied to NET (Off - Def)
        Output stays a single Rating per team: 1500 + (Off - Def)
        """
        phi         = float(cfg.get("phi", 0.96))
        sigma_eta   = float(cfg.get("sigma_eta", 6.0))        # state drift (per component)
        sigma_pts   = float(cfg.get("sigma_pts", 12.0))       # points observation SD
        sigma_s     = float(cfg.get("sigma_spread", 7.0))     # spread sensor SD
        mov_cap     = cfg.get("mov_cap", None)
        HFA_pts     = float(cfg.get("HFA_pts", 2.0))
    
        market_lambda = float(cfg.get("market_lambda", 0.35))
        favored_only  = bool(cfg.get("favored_only_market", False))
    
        sos_hl    = float(cfg.get("sos_half_life_days", 60))
        sos_gamma = float(cfg.get("sos_gamma", 0.6))
    
        season_gap_days = int(cfg.get("season_gap_days", 120))
        season_rho      = float(cfg.get("season_rho", 0.75))
    
        blowout_mov     = float(cfg.get("blowout_mov", 999))
        late_month_start= int(cfg.get("late_month_start", 13))
        blowout_var_mult= float(cfg.get("blowout_var_mult", 1.0))
    
        b2b_sigma_mult  = float(cfg.get("b2b_sigma_pts_mult", 1.0))
    
        # ---- states ----
        off_m: dict[str, float] = {}
        def_m: dict[str, float] = {}
        off_v: dict[str, float] = {}
        def_v: dict[str, float] = {}
    
        def _om(t): return off_m.get(t, 0.0)
        def _dm(t): return def_m.get(t, 0.0)
        def _ov(t): return off_v.get(t, 100.0)
        def _dv(t): return def_v.get(t, 100.0)
    
        # Seed from existing single Rating (net) if present
        seed = load_seed_ratings(sport, window_start)
        for team, rating in seed.items():
            try:
                net = float(rating) - 1500.0
                # split net into Off and Def symmetrically
                off_m[str(team)] =  0.5 * net
                def_m[str(team)] = -0.5 * net
                off_v[str(team)] =  75.0
                def_v[str(team)] =  75.0
            except Exception:
                pass
    
        # SoS accumulators computed on NET
        sos_num: dict[str, float] = {}
        sos_den: dict[str, float] = {}
    
        def _net(t: str) -> float:
            return _om(t) - _dm(t)
    
        def _cap_margin(mov):
            if mov_cap is None:
                return float(mov)
            return float(min(max(mov, -mov_cap), mov_cap))
    
        def _exp_decay_weight(gs_ts: pd.Timestamp, now_ts: pd.Timestamp, half_life_days: float) -> float:
            age_days = max((now_ts - gs_ts).days, 0)
            return float(np.exp(- age_days / max(half_life_days, 1.0)))
    
        def _safe_get(d: dict, k, default=0.0):
            v = d.get(k, default)
            return float(v if np.isfinite(v) else default)
    
        # ---- simple Kalman update for two variables: x1=Off(teamA) and x2=Def(teamB) given points ----
        # obs: pts ≈ mu + Off_A - Def_B + hfa_pts_adj
        def _update_off_def_for_points(team_off, team_def, pts_obs, mu, hfa_adj, obs_var):
            # state means/vars
            m1, v1 = _om(team_off), _ov(team_off)
            m2, v2 = _dm(team_def), _dv(team_def)
    
            # prediction
            y_hat = mu + m1 - m2 + hfa_adj
            e     = float(pts_obs) - float(y_hat)
    
            # Since covariances are ignored (fast/streaming), gain is diagonal-ish:
            S = v1 + v2 + obs_var
            if S <= 0:
                S = 1e-6
            k1 = v1 / S
            k2 = v2 / S
    
            # update: Off increases with e; Def decreases with e (because y_hat uses -Def)
            m1p = m1 + k1 * e
            m2p = m2 - k2 * e
            v1p = (1.0 - k1) * v1
            v2p = (1.0 - k2) * v2
    
            off_m[team_off], off_v[team_off] = m1p, v1p
            def_m[team_def], def_v[team_def] = m2p, v2p
    
        def _apply_net_shift(team: str, delta_net: float):
            # shift net by pushing Off up and Def down equally
            off_m[team] = _om(team) + 0.5 * delta_net
            def_m[team] = _dm(team) - 0.5 * delta_net
    
        # ---- market sensor: apply ONLY to favored team if requested ----
        def _market_anchor_favored_only(home: str, away: str, spread_close_home: float):
            # Spread_Close is HOME spread (negative => home favored) :contentReference[oaicite:5]{index=5}
            sc = float(spread_close_home)
            if not np.isfinite(sc):
                return
    
            # expected MOV from spread
            obs_mov = _cap_margin(-sc)  # home -5.5 -> expected MOV +5.5
    
            # figure out favorite
            home_fav = (sc < 0)
            fav = home if home_fav else away
    
            # compare obs to current expectation
            exp_mov = _cap_margin((_net(home) - _net(away)) + HFA_pts)
            e = obs_mov - exp_mov
    
            lam = max(1e-3, min(1.0, float(market_lambda)))
            # weaker market => bigger variance => smaller effective step
            obs_var = (sigma_s / lam) ** 2
    
            # one-sided gain using fav net variance approx
            v_fav = _ov(fav) + _dv(fav)
            S = v_fav + obs_var
            if S <= 0:
                S = 1e-6
            k = v_fav / S
    
            _apply_net_shift(fav, k * e)
    
            # inflate uncertainty a touch after market adjust (optional)
            off_v[fav] = min(400.0, _ov(fav) + 0.10 * sigma_eta**2)
            def_v[fav] = min(400.0, _dv(fav) + 0.10 * sigma_eta**2)
    
        def _market_anchor_two_sided(home: str, away: str, spread_close_home: float):
            sc = float(spread_close_home)
            if not np.isfinite(sc):
                return
    
            obs_mov = _cap_margin(-sc)
            exp_mov = _cap_margin((_net(home) - _net(away)) + HFA_pts)
            e = obs_mov - exp_mov
    
            lam = max(1e-3, min(1.0, float(market_lambda)))
            obs_var = (sigma_s / lam) ** 2
    
            # split update across both teams' net (fast, stable)
            v_h = _ov(home) + _dv(home)
            v_a = _ov(away) + _dv(away)
            S = v_h + v_a + obs_var
            if S <= 0:
                S = 1e-6
    
            kh = v_h / S
            ka = v_a / S
    
            _apply_net_shift(home,  kh * e)
            _apply_net_shift(away, -ka * e)
    
        history_batch: List[dict] = []
        BATCH_SZ = 50_000
        utc_now = pd.Timestamp.now(tz="UTC")
        last_game_ts = None
    
        # league baseline points per team per game (EWMA)
        mu = None
        mu_alpha = 0.02
    
        for chunk in load_games_stream(
            sport, aliases, cutoff=window_start, page_rows=200_000,
            project_table_market=project_table_market, sharp_books=SHARP_BOOKS_DEFAULT
        ):
            for _, g in chunk.iterrows():
                h, a = g.Home_Team, g.Away_Team
                gs_ts = g.Game_Start if isinstance(g.Game_Start, pd.Timestamp) else pd.Timestamp(g.Game_Start, tz="UTC")
                hs, as_ = float(g.Score_Home_Score), float(g.Score_Away_Score)
    
                # ---- offseason / long-gap reset ----
                if last_game_ts is not None:
                    gap = (gs_ts - last_game_ts).days
                    if gap >= season_gap_days:
                        for t in set(list(off_m.keys()) + list(def_m.keys())):
                            off_m[t] *= season_rho
                            def_m[t] *= season_rho
                            off_v[t]  = min(400.0, _ov(t) + 60.0)
                            def_v[t]  = min(400.0, _dv(t) + 60.0)
                last_game_ts = gs_ts
    
                # Update mu (points per team) from this game
                pts_per_team = 0.5 * (hs + as_)
                if mu is None:
                    mu = pts_per_team
                else:
                    mu = (1.0 - mu_alpha) * mu + mu_alpha * pts_per_team
    
                # ---- downweight late-season blowouts (inflate obs variance) ----
                mov = _cap_margin(hs - as_)
                obs_var_pts = sigma_pts ** 2
                if (abs(mov) >= blowout_mov) and (int(gs_ts.month) >= late_month_start):
                    obs_var_pts *= blowout_var_mult
    
                # ---- optional back-to-back penalty via σ_y (only if rest cols exist later) ----
                # No-op unless your stream includes these columns (currently it doesn’t).
                if hasattr(g, "Home_Rest_Days") and hasattr(g, "Away_Rest_Days"):
                    try:
                        if float(getattr(g, "Home_Rest_Days")) <= 0 or float(getattr(g, "Away_Rest_Days")) <= 0:
                            obs_var_pts *= (b2b_sigma_mult ** 2)
                    except Exception:
                        pass
    
                # HFA split across points sensors
                hfa_adj_home = 0.5 * HFA_pts
                hfa_adj_away = -0.5 * HFA_pts
    
                # ---- score sensors (points) ----
                _update_off_def_for_points(h, a, hs, mu, hfa_adj_home, obs_var_pts)
                _update_off_def_for_points(a, h, as_, mu, hfa_adj_away, obs_var_pts)
    
                # ---- market spread sensor ----
                sc = g.Spread_Close
                if pd.notna(sc):
                    if favored_only:
                        _market_anchor_favored_only(h, a, float(sc))
                    else:
                        _market_anchor_two_sided(h, a, float(sc))
    
                # ---- SoS accumulation on NET ----
                gs_ts_utc = gs_ts if gs_ts.tzinfo else gs_ts.tz_localize("UTC")
                w = _exp_decay_weight(gs_ts_utc, utc_now, sos_hl)
                sos_num[h] = _safe_get(sos_num, h) + w * float(_net(a))
                sos_num[a] = _safe_get(sos_num, a) + w * float(_net(h))
                sos_den[h] = _safe_get(sos_den, h) + w
                sos_den[a] = _safe_get(sos_den, a) + w
    
                # ---- propagate (per component) ----
                for t in (h, a):
                    off_m[t] = phi * _om(t)
                    def_m[t] = phi * _dm(t)
                    off_v[t] = (phi**2) * _ov(t) + sigma_eta**2
                    def_v[t] = (phi**2) * _dv(t) + sigma_eta**2
    
                ts  = g.Game_Start  # ✅ Option A: rating timestamp = game time (pregame)
                if isinstance(ts, pd.Timestamp) and ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                
                tag = "backfill" if window_start is None else "incremental"
                
                history_batch.append({"Sport": sport, "Team": h, "Rating": 1500.0 + float(_net(h)),
                                      "Method": "elo_kalman", "Updated_At": ts, "Source": tag})
                history_batch.append({"Sport": sport, "Team": a, "Rating": 1500.0 + float(_net(a)),
                                      "Method": "elo_kalman", "Updated_At": ts, "Source": tag})

    
                if len(history_batch) >= BATCH_SZ:
                    bq.load_table_from_dataframe(pd.DataFrame(history_batch), table_history).result()
                    history_batch.clear()
    
            del chunk
            gc.collect()
    
        if history_batch:
            bq.load_table_from_dataframe(pd.DataFrame(history_batch), table_history).result()
    
        teams = sorted(set(list(off_m.keys()) + list(def_m.keys())))
        if not teams:
            return []
    
        raw_R = np.array([_net(t) for t in teams], dtype=float)
    
        # SoS correction on NET (same structure as your prior approach) :contentReference[oaicite:6]{index=6}
        sos = np.array([(_safe_get(sos_num, t) / max(_safe_get(sos_den, t), 1e-9)) for t in teams], dtype=float)
        sos = np.where(np.isfinite(sos), sos, np.nan)
    
        league_mean_R = float(np.nanmean(raw_R)) if np.isfinite(np.nanmean(raw_R)) else 0.0
        sos = np.where(np.isnan(sos), league_mean_R, sos)
    
        adj_R = raw_R + sos_gamma * (league_mean_R - sos)
    
        utc_now = pd.Timestamp.now(tz="UTC")
        return [
            {"Sport": sport, "Team": t, "Rating": 1500.0 + float(r), "Method": "elo_kalman", "Updated_At": utc_now}
            for t, r in zip(teams, adj_R)
        ]


    def run_ridge_massey(bq, sport: str, aliases: list[str], cfg: dict):
        """
        Best-practice Ridge-Massey for NCAAB (scores + sharp spreads):
    
        Upgrades vs baseline:
          ✅ Robust MOV (smooth tanh cap, not hard clip)
          ✅ Recency weighting (half-life days) in the normal equations
          ✅ Strong early-season shrink with schedule-strength adjustment
          ✅ Phase-aware window_days (early season longer, late season shorter)
          ✅ Spread calibration with magnitude-aware weights (downweight extreme spreads)
          ✅ Guards for degenerate calibration + NaNs
          ✅ Keeps output schema identical (one Rating per team)
    
        Assumes the following exist in module scope:
          - SHARP_BOOKS_DEFAULT
          - table_history (outer scope in update_power_ratings)
          - bigquery imported, np/pd imported
        """
    
        # ---------------- knobs ----------------
        window_days  = int(cfg.get("window_days", 90))
        mov_cap      = float(cfg.get("mov_cap", 25))
        ridge_lambda = float(cfg.get("ridge_lambda", 65.0))
    
        # NEW knobs (safe defaults)
        rm_half_life_days      = float(cfg.get("rm_half_life_days", 35.0))     # recency weighting
        preseason_prior_games  = float(cfg.get("preseason_prior_games", 18.0)) # stronger w/ all teams
        phase_aware_window     = bool(cfg.get("phase_aware_window", True))
        use_home_field_term    = bool(cfg.get("use_hfa_term", True))           # keep n+1 column
        spread_calib_clip_abs  = float(cfg.get("spread_calib_clip_abs", 15.0)) # cap for weight calc
        spread_calib_floor_abs = float(cfg.get("spread_calib_floor_abs", 1.0)) # avoid /0
    
        # Phase-aware window: stabilize early season, react late season
        if phase_aware_window:
            today = pd.Timestamp.utcnow()
            m = int(today.month)
            if m in (11, 12):          # early season
                window_days = max(window_days, 120)
            elif m in (1, 2):          # conference ramp
                window_days = min(window_days, 100)
            elif m in (3,):            # late season / tournament
                window_days = min(window_days, 75)
    
        # ---------------- query ----------------
        sql = """
        WITH scores AS (
          SELECT
            LOWER(TRIM(Home_Team)) AS home,
            LOWER(TRIM(Away_Team)) AS away,
            SAFE_CAST(Score_Home_Score AS FLOAT64) AS hs,
            SAFE_CAST(Score_Away_Score AS FLOAT64) AS as_,
            TIMESTAMP(Game_Start) AS gs,
            TIMESTAMP(Game_Start) AS game_start_raw
          FROM `sharplogger.sharp_data.game_scores_final`
          WHERE UPPER(CAST(Sport AS STRING)) IN UNNEST(@aliases)
            AND Score_Home_Score IS NOT NULL
            AND Score_Away_Score IS NOT NULL
            AND TIMESTAMP(Game_Start) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @win_days DAY)
        ),
    
        latest_per_book AS (
          SELECT
            LOWER(TRIM(m.Home_Team_Norm)) AS home,
            LOWER(TRIM(m.Away_Team_Norm)) AS away,
            TIMESTAMP(m.Game_Start)       AS game_start_raw,
    
            REGEXP_REPLACE(
              LOWER(TRIM(COALESCE(m.Bookmaker, m.Book))),
              r'^betfair_ex(?:_[a-z]+)?$', 'betfair_ex'
            ) AS BookKey,
    
            SAFE_CAST(m.Value AS FLOAT64) AS spread_value,      -- spread for this team
            m.Snapshot_Timestamp,
            LOWER(TRIM(m.Team_For_Join)) AS fav_team           -- team this row is for
          FROM `sharplogger.sharp_data.moves_with_features_merged` m
          WHERE
            UPPER(CAST(m.Sport AS STRING)) IN UNNEST(@aliases)
            AND m.Market IN ('spreads','spread','Spread','SPREADS')
            AND m.Value IS NOT NULL
            AND TIMESTAMP(m.Game_Start) IS NOT NULL
            AND m.Value < 0    -- favorite side only
          QUALIFY ROW_NUMBER() OVER (
            PARTITION BY
              LOWER(TRIM(m.Home_Team_Norm)),
              LOWER(TRIM(m.Away_Team_Norm)),
              TIMESTAMP(m.Game_Start),
              REGEXP_REPLACE(
                LOWER(TRIM(COALESCE(m.Bookmaker, m.Book))),
                r'^betfair_ex(?:_[a-z]+)?$', 'betfair_ex'
              )
            ORDER BY m.Snapshot_Timestamp DESC
          ) = 1
        ),
    
        market AS (
          SELECT
            home,
            away,
            game_start_raw,
            ANY_VALUE(fav_team) AS favorite_team,
            CASE
              WHEN ANY_VALUE(fav_team) = home THEN away ELSE home
            END AS dog_team,
            APPROX_QUANTILES(spread_value, 2)[OFFSET(1)] AS spread_close
          FROM latest_per_book
          WHERE BookKey IN UNNEST(@sharp_books)
          GROUP BY home, away, game_start_raw
        )
    
        SELECT
          s.home,
          s.away,
          s.hs,
          s.as_,
          s.gs,
          m.spread_close,
          m.favorite_team,
          m.dog_team
        FROM scores s
        LEFT JOIN market m
          ON s.home = m.home
         AND s.away = m.away
         AND s.game_start_raw = m.game_start_raw
        """
    
        df = bq.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("aliases", "STRING", aliases),
                    bigquery.ScalarQueryParameter("win_days", "INT64", int(window_days)),
                    bigquery.ArrayQueryParameter("sharp_books", "STRING", SHARP_BOOKS_DEFAULT),
                ]
            )
        ).to_dataframe()
    
        if df.empty:
            return []
    
        # ---------------- build MOV (robust) ----------------
        # Smooth cap: cap * tanh(mov/cap) reduces blowout distortion vs hard clip
        mov_raw = (df["hs"] - df["as_"]).astype(float)
        cap = float(mov_cap) if np.isfinite(mov_cap) and mov_cap > 0 else 25.0
        df["mov"] = cap * np.tanh(mov_raw / cap)
    
        # Ensure gs is tz-aware
        df["gs"] = pd.to_datetime(df["gs"], utc=True, errors="coerce")
        df = df[df["gs"].notna()].copy()
        if df.empty:
            return []
    
        # ---------------- team index ----------------
        teams = pd.Index(sorted(set(df.home) | set(df.away)))
        n = len(teams)
        if n < 2:
            return []
        idx = {t: i for i, t in enumerate(teams)}
    
        m_rows = len(df)
        X = np.zeros((m_rows, n + (1 if use_home_field_term else 0)), dtype=float)
        y = df["mov"].to_numpy(float)
    
        # Fill X
        # (home +1, away -1, optional HFA indicator +1)
        if use_home_field_term:
            hfa_col = n
        for i, row in enumerate(df.itertuples(index=False)):
            hi, ai = idx[row.home], idx[row.away]
            X[i, hi] = 1.0
            X[i, ai] = -1.0
            if use_home_field_term:
                X[i, hfa_col] = 1.0
    
        # ---------------- recency weights ----------------
        now_ts = df["gs"].max()
        age_days = (now_ts - df["gs"]).dt.total_seconds().to_numpy(float) / 86400.0
        w = np.exp(-age_days / max(rm_half_life_days, 1.0)).astype(float)
    
        # apply weights via sqrt(w)
        sw = np.sqrt(np.clip(w, 1e-9, None))
        Xw = X * sw[:, None]
        yw = y * sw
    
        # ---------------- solve ridge system ----------------
        XtX = Xw.T @ Xw
        # ridge on team coeffs
        XtX[:n, :n] += np.eye(n) * ridge_lambda
        # small ridge on HFA term (if present)
        if use_home_field_term:
            XtX[n, n] += ridge_lambda * 0.01
    
        beta = np.linalg.solve(XtX, Xw.T @ yw)
        ratings_pts = beta[:n].astype(float)
    
        # center ratings around 0
        ratings_pts = ratings_pts - float(np.mean(ratings_pts))
    
        # ---------------- early-season shrink (schedule-aware) ----------------
        # games played in window
        gcounts = (
            pd.concat([df["home"], df["away"]])
            .value_counts()
            .reindex(teams, fill_value=0)
            .to_numpy(float)
        )
    
        # One-pass SoS proxy: avg opponent rating (using current ratings_pts)
        rating_map0 = {t: float(r) for t, r in zip(teams, ratings_pts)}
        opp_sum = pd.Series(0.0, index=teams)
        opp_cnt = pd.Series(0.0, index=teams)
    
        # vectorized-ish loop over pairs (fast enough for NCAAB window sizes)
        for h, a in zip(df["home"].to_numpy(), df["away"].to_numpy()):
            opp_sum[h] += rating_map0.get(a, 0.0); opp_cnt[h] += 1.0
            opp_sum[a] += rating_map0.get(h, 0.0); opp_cnt[a] += 1.0
    
        opp_avg = (opp_sum / opp_cnt.replace(0, np.nan)).fillna(0.0).reindex(teams).to_numpy(float)
    
        # weaker schedules => more shrink early
        # (if opp_avg is very negative, schedule is weak)
        sos_penalty = 1.0 / (1.0 + np.maximum(0.0, -opp_avg) / 6.0)
    
        shrink = (gcounts / (gcounts + preseason_prior_games)) * sos_penalty
        ratings_pts = ratings_pts * shrink
    
        # re-center
        ratings_pts = ratings_pts - float(np.mean(ratings_pts))
    
        # ---------------- market calibration (weighted) ----------------
        df_cal = df[
            df["spread_close"].notna()
            & df["favorite_team"].notna()
            & df["dog_team"].notna()
        ].copy()
    
        if not df_cal.empty:
            # map ratings
            rating_map = {t: float(r) for t, r in zip(teams, ratings_pts)}
            df_cal["rdiff"] = df_cal["favorite_team"].map(rating_map) - df_cal["dog_team"].map(rating_map)
    
            # weights: downweight huge spreads & tiny spreads
            sabs = df_cal["spread_close"].abs().clip(lower=spread_calib_floor_abs, upper=spread_calib_clip_abs)
            wcal = (1.0 / sabs).to_numpy(float)
    
            # Solve k in: spread_close ≈ -k * rdiff  => minimize weighted SSE
            # k = - (Σ w rdiff spread) / (Σ w rdiff^2)
            rdiff = df_cal["rdiff"].to_numpy(float)
            spread = df_cal["spread_close"].to_numpy(float)
    
            num = float(np.sum(wcal * rdiff * spread))
            den = float(np.sum(wcal * (rdiff ** 2)))
    
            if np.isfinite(num) and np.isfinite(den) and den > 1e-9:
                k = - num / den
                # guard against absurd scaling
                if np.isfinite(k) and 0.10 < abs(k) < 50.0:
                    ratings_pts = ratings_pts * float(k)
    
        # final center (optional)
        ratings_pts = ratings_pts - float(np.mean(ratings_pts))
    
        # ---------------- write history + current ----------------
        ts = df["gs"].max()
        if isinstance(ts, pd.Timestamp) and ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        # ---------------- SoS correction (light touch, avoid double-count) ----------------
        sos_gamma = float(cfg.get("sos_gamma", 0.25))
        sos_use_recency = bool(cfg.get("sos_use_recency", True))
        sos_early_only = bool(cfg.get("sos_early_only", False))
        
        if sos_gamma != 0.0:
            # Optional: apply only early season
            if (not sos_early_only) or (pd.Timestamp.utcnow().month in (11, 12)):
                # Map current ratings
                rating_map = {t: float(r) for t, r in zip(teams, ratings_pts)}
        
                # opponent sums
                sos_num = pd.Series(0.0, index=teams)
                sos_den = pd.Series(0.0, index=teams)
        
                # weights per game (reuse your recency weights if desired)
                if sos_use_recency and "gs" in df.columns:
                    now_ts = df["gs"].max()
                    age_days = (now_ts - df["gs"]).dt.total_seconds().to_numpy(float) / 86400.0
                    w_sos = np.exp(-age_days / max(float(cfg.get("rm_half_life_days", 35.0)), 1.0)).astype(float)
                else:
                    w_sos = np.ones(len(df), dtype=float)
        
                homes = df["home"].to_numpy()
                aways = df["away"].to_numpy()
        
                for h, a, w in zip(homes, aways, w_sos):
                    ra = rating_map.get(a, 0.0)
                    rh = rating_map.get(h, 0.0)
        
                    sos_num[h] += w * ra
                    sos_den[h] += w
        
                    sos_num[a] += w * rh
                    sos_den[a] += w
        
                sos = (sos_num / sos_den.replace(0.0, np.nan)).fillna(0.0).to_numpy(float)
        
                league_mean = float(np.mean(ratings_pts))
                # If sos below mean => weak schedule => penalize
                ratings_pts = ratings_pts + sos_gamma * (league_mean - sos)
        
                # re-center (keep mean 0)
                ratings_pts = ratings_pts - float(np.mean(ratings_pts))

        
    
        hist_rows = [
            {
                "Sport": sport,
                "Team": t,
                "Rating": 1500.0 + float(r),
                "Method": "ridge_massey",
                "Updated_At": ts,           # snapshot timestamp for this window fit
                "Source": "window_fit_best",
            }
            for t, r in zip(teams, ratings_pts)
        ]
        if hist_rows:
            # batch load for safety
            for i in range(0, len(hist_rows), 50_000):
                bq.load_table_from_dataframe(pd.DataFrame(hist_rows[i:i+50_000]), table_history).result()
    
        utc_now = pd.Timestamp.now(tz="UTC")
        current_rows = [
            {
                "Sport": sport,
                "Team": t,
                "Rating": 1500.0 + float(r),
                "Method": "ridge_massey",
                "Updated_At": utc_now,
            }
            for t, r in zip(teams, ratings_pts)
        ]
        return current_rows

    # ---------------- MAIN ----------------

    sports_available = fetch_sports_present()
    sports = [s for s in sports_available if s.upper() in SPORT_CFG]
    if not sports:
        return pd.DataFrame(columns=["Sport", "Team", "Method", "Rating", "Updated_At"])

    current_rows_all: list[dict] = []
    updated_sports: list[str] = []

    for sport in sports:
        cfg = SPORT_CFG[sport.upper()]
        last_ts = get_last_ts(sport)

        # Always replay at least BACKFILL_DAYS by snapshot time
        if last_ts is None:
            window_start = None
        else:
            if isinstance(last_ts, pd.Timestamp):
                last_ts_utc = to_utc_ts(last_ts)
                window_start = last_ts_utc - pd.Timedelta(days=BACKFILL_DAYS)
            else:
                lt = last_ts if last_ts.tzinfo else last_ts.replace(tzinfo=dt.timezone.utc)
                window_start = lt - dt.timedelta(days=BACKFILL_DAYS)

        if not has_new_finals_since(sport, window_start):
            continue

        aliases = get_aliases(sport)

        if cfg["model"] == "poisson":
            # ---------- MLB Poisson ----------
            GF: dict[str, float] = {}
            GA: dict[str, float] = {}
            GP: dict[str, int]   = {}

            def add_game(h, a, hs, as_):
                GF[h] = GF.get(h, 0.0) + hs
                GA[h] = GA.get(h, 0.0) + as_
                GP[h] = GP.get(h, 0) + 1
                GF[a] = GF.get(a, 0.0) + as_
                GA[a] = GA.get(a, 0.0) + hs
                GP[a] = GP.get(a, 0) + 1

            def team_rating(team: str) -> float:
                g = int(GP.get(team, 0))
                if g == 0:
                    return 1500.0
                total_team_games = int(sum(GP.values()))
                gf_league = float(sum(GF.values()))
                eps = 1e-6
                league_rate = max(gf_league / max(total_team_games, 1), eps)
                rate_for     = float(GF.get(team, 0.0)) / max(g, 1)
                rate_against = float(GA.get(team, 0.0)) / max(g, 1)
                atk_ratio = max(rate_for / league_rate, eps)
                dfn_ratio = max(rate_against / league_rate, eps)
                atk = np.log(atk_ratio)
                dfn = -np.log(dfn_ratio)
                return float(1500.0 + 400.0 * (atk + dfn))

            history_batch: list[dict] = []
            BATCH_SZ = 50_000

            for chunk in load_games_stream(sport, aliases, cutoff=window_start, page_rows=200_000,
                                           project_table_market=project_table_market):
                for _, g in chunk.iterrows():
                    hs, as_ = float(g.Score_Home_Score), float(g.Score_Away_Score)
                    add_game(g.Home_Team, g.Away_Team, hs, as_)
                    Rh2 = team_rating(g.Home_Team)
                    Ra2 = team_rating(g.Away_Team)
                    ts  = g.Game_Start  # ✅ Option A: timestamp = game time
                    if isinstance(ts, pd.Timestamp) and ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    tag = "backfill" if window_start is None else "incremental"

                    history_batch.append({"Sport": sport, "Team": g.Home_Team, "Rating": Rh2,
                                          "Method": "poisson", "Updated_At": ts, "Source": tag})
                    history_batch.append({"Sport": sport, "Team": g.Away_Team, "Rating": Ra2,
                                          "Method": "poisson", "Updated_At": ts, "Source": tag})

                    if len(history_batch) >= BATCH_SZ:
                        bq.load_table_from_dataframe(pd.DataFrame(history_batch), table_history).result()
                        history_batch.clear()
                del chunk
                gc.collect()

            if history_batch:
                bq.load_table_from_dataframe(pd.DataFrame(history_batch), table_history).result()
                history_batch.clear()

            utc_now = pd.Timestamp.now(tz="UTC")
            all_teams = set(GP.keys())
            for team in all_teams:
                current_rows_all.append({
                    "Sport": sport,
                    "Team": team,
                    "Rating": float(team_rating(team)),
                    "Method": "poisson",
                    "Updated_At": utc_now,
                })
            updated_sports.append(sport)

        
        elif cfg["model"] in ("elo_kalman", "elo_kalman_offdef"):
            cur_rows = run_kalman_offdef(bq=bq, sport=sport, aliases=aliases, cfg=cfg, window_start=window_start)
            if cur_rows:
                current_rows_all.extend(cur_rows)
                updated_sports.append(sport)

        elif cfg["model"] == "ridge_massey":
            cur_rows = run_ridge_massey(bq=bq, sport=sport, aliases=aliases, cfg=cfg)
            if cur_rows:
                current_rows_all.extend(cur_rows)
                updated_sports.append(sport)

        else:
            continue

    # Write current + reconcile
    upsert_current(current_rows_all)

    # Bring current in sync with latest history for updated sports
    if updated_sports:
        pm_values = [{"s": s, "m": PREFERRED_METHOD[s]} for s in updated_sports]

        struct_params = [
            bigquery.StructQueryParameter(
                "",
                bigquery.ScalarQueryParameter("s", "STRING", item["s"]),
                bigquery.ScalarQueryParameter("m", "STRING", item["m"]),
            )
            for item in pm_values
        ]
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("pm", "STRUCT", struct_params)]
        )

        sql = f"""
        MERGE `{table_current}` T
        USING (
          WITH filtered AS (
            SELECT h.Sport, h.Team, h.Method, h.Rating, h.Updated_At
            FROM `{table_history}` h
            JOIN UNNEST(@pm) AS x
              ON UPPER(h.Sport) = UPPER(x.s) AND LOWER(h.Method) = LOWER(x.m)
          ),
          latest AS (
            SELECT Sport, Team, Method, Rating, Updated_At,
                   ROW_NUMBER() OVER (PARTITION BY Sport, Team ORDER BY Updated_At DESC) rn
            FROM filtered
          )
          SELECT Sport, Team, Method, Rating, Updated_At
          FROM latest
          WHERE rn = 1
        ) S
        ON T.Sport = S.Sport AND T.Team = S.Team
        WHEN MATCHED THEN UPDATE SET
          T.Rating = S.Rating,
          T.Method = S.Method,
          T.Updated_At = S.Updated_At
        WHEN NOT MATCHED THEN
          INSERT (Sport, Team, Method, Rating, Updated_At)
          VALUES (S.Sport, S.Team, S.Method, S.Rating, S.Updated_At)
        """
        bq.query(sql, job_config=job_config).result()

    # Safety backfill for any missing current rows
    fill_missing_current_from_history(None)

    if not updated_sports:
        return pd.DataFrame(columns=["Sport", "Team", "Method", "Rating", "Updated_At"])

    q = f"""
    SELECT Sport, Team, Method, Rating, Updated_At
    FROM `{table_current}`
    WHERE UPPER(Sport) IN UNNEST(@sports)
    ORDER BY Sport, Team
    """
    params = [bigquery.ArrayQueryParameter("sports", "STRING", [s.upper() for s in updated_sports])]
    result_df = bq.query(q, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
    return result_df


def normalize_team(t):
    return str(t).strip().lower().replace('.', '').replace('&', 'and')

def build_merge_key(home, away, game_start):
    return f"{normalize_team(home)}_{normalize_team(away)}_{game_start.floor('h').strftime('%Y-%m-%d %H:%M:%S')}"


def compute_line_hash(row, window='1h'):
    try:
        key_fields = [
            str(row.get('Game_Key', '')).strip().lower(),
            str(row.get('Bookmaker', '')).strip().lower(),
            str(row.get('Market', '')).strip().lower(),
            str(row.get('Outcome', '')).strip().lower(),
            str(row.get('Value', '')).strip(),
            str(row.get('Odds_Price', '')).strip(),
            str(row.get('Limit', '')).strip(),
        ]

        # Always include floored snapshot hour
        ts = pd.to_datetime(row.get('Snapshot_Timestamp'), utc=True, errors='coerce')
        if pd.notna(ts):
            ts_bucket = ts.floor(window)  # e.g., '1h', '15min'
            key_fields.append(str(ts_bucket))
        else:
            key_fields.append('')  # keep key length consistent

        key = "|".join(key_fields)
        return hashlib.md5(key.encode()).hexdigest()

    except Exception as e:
        return f"ERROR_HASH_{hashlib.md5(str(e).encode()).hexdigest()[:8]}"
from google.cloud import bigquery
import pandas as pd
def fetch_power_ratings_from_bq(bq, sport: str, lookback_days: int = 400) -> pd.DataFrame:
    """
    Load team power ratings for a given sport from BigQuery (current-only, per your schema).
    Outputs canonical columns:
      ['Sport','Team_Norm','AsOfTS','Power_Rating','PR_Off','PR_Def']
    PR_Off/PR_Def are NULL since not in ratings_current.
    """
   

    sql_current = """
        SELECT
          UPPER(Sport) AS Sport,
          -- normalize team now; we'll still re-normalize in Python before merge
          LOWER(TRIM(CAST(Team AS STRING))) AS Team_Norm,
          TIMESTAMP(IFNULL(Updated_At, CURRENT_TIMESTAMP())) AS AsOfTS,
          CAST(Rating AS FLOAT64) AS Power_Rating,
          CAST(NULL AS FLOAT64) AS PR_Off,
          CAST(NULL AS FLOAT64) AS PR_Def
        FROM `sharplogger.sharp_data.ratings_current`
        WHERE UPPER(Sport) = @sport
          AND Updated_At >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("sport", "STRING", sport.upper()),
            bigquery.ScalarQueryParameter("lookback_days", "INT64", int(lookback_days)),
        ]
    )
    df = bq.query(sql_current, job_config=job_config).to_dataframe(create_bqstorage_client=True)

    # Final normalization / guards
    if not df.empty:
        df["Sport"] = df["Sport"].astype(str).str.upper()
        df["Team_Norm"] = df["Team_Norm"].astype(str).str.strip().str.lower()
        df = df.sort_values(["Team_Norm", "AsOfTS"]).drop_duplicates(["Team_Norm"], keep="last")

    return df


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


model_cache = {}


def get_trained_models(sport_key):
    if sport_key not in model_cache:
        model_cache[sport_key] = {
            market: load_model_from_gcs(sport_key, market)
            for market in ['spreads', 'totals', 'h2h']
        }
    return model_cache[sport_key]
    
    
sharp_moves_cache = {}


# --- Identity calibrators (used as safe fallbacks) ---
class _IdentityIsoCal:
    def __init__(self, eps=1e-4): self.eps = float(eps)
    def transform(self, p):
        import numpy as np
        p = np.asarray(p, float)
        return np.clip(p, self.eps, 1 - self.eps)

class _IdentityProbCal:
    def __init__(self, eps=1e-4): self.eps = float(eps)
    def predict_proba(self, p2d):
        import numpy as np
        p = np.asarray(p2d, float).reshape(-1)
        p = np.clip(p, self.eps, 1 - self.eps)
        return np.column_stack([1 - p, p])

# --- Adapter you can save and later call .predict(p) on ---
class _CalAdapter:
    """
    Wrap a calibrator tuple ('iso'|'beta'|'platt', model) so you can call .predict(p).
    Also applies an output clip to avoid 0/1 extremes.
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
            out = self.model.transform(p)
        elif self.kind == "beta":
            out = self.model.predict(p.reshape(-1, 1))
        else:  # 'platt'
            out = self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        if self.clip:
            lo, hi = self.clip
            out = np.clip(out, lo, hi)
        return out


def read_recent_sharp_master_cached(hours=120):
    cache_key = f"sharp_master_{hours}h"
    
    if cache_key in sharp_moves_cache:
        return sharp_moves_cache[cache_key]
    
    df = read_recent_sharp_moves(hours=hours)  # this fetches from BigQuery
    sharp_moves_cache[cache_key] = df
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
                        'Odds_Price': outcome.get('price'),  # ✅ This adds the odds for spread/total
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

def compute_market_weights(df):
    # Temporary shim to keep new loader working
    return compute_and_write_market_weights(df)

def write_sharp_moves_to_master(df, table='sharp_data.sharp_moves_master'):
    if df is None or df.empty:
        logging.warning("⚠️ No sharp moves to write.")
        return

    df = df.copy()
    df = build_game_key(df)

    allowed_books = SHARP_BOOKS + REC_BOOKS
    # ✅ Normalize book names before any filtering
    if 'Book' in df.columns:
        df['Book'] = df['Book'].astype(str).str.lower()
        # ✅ Now filter using normalized names
        df = df[df['Book'].isin([b.lower() for b in allowed_books])]

    if 'Game_Key' not in df.columns or df['Game_Key'].isnull().all():
        logging.warning("❌ No valid Game_Key present — skipping upload.")
        cols = [c for c in ['Game','Game_Start','Market','Outcome'] if c in df.columns]
        if cols:
            logging.debug(df[cols].head().to_string())
        return
    # --- HARDEN: force Event_Date to STRING for BQ ---
    if 'Event_Date' in df.columns:
        # Coerce any date-like (datetime.date / Timestamp / str) to canonical 'YYYY-MM-DD' string
        ev = pd.to_datetime(df['Event_Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df['Event_Date'] = ev.where(ev.notna(), None)  # keep NULLs as NULLs (not 'NaT')
        # ❌ Don't write any rows that are already past game start
    if 'Post_Game' in df.columns:
        pre_filter = len(df)
        df = df[df['Post_Game'] == False]
        logging.info(f"🧹 Removed {pre_filter - len(df)} post-game rows before writing to sharp_moves_master")

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

    # enforce sport short form before upload
    SPORT_KEY_TO_LABEL = {
        "baseball_mlb": "MLB",
        "basketball_nba": "NBA",
        "americanfootball_nfl": "NFL",   # ✅ was "football_nfl"
        "americanfootball_ncaaf": "NCAAF",  # ✅ was "football_ncaaf"
        "basketball_wnba": "WNBA",
        "canadianfootball_cfl": "CFL",
        "basketball_ncaab": "NCAAB",
        "basketball_ncaam": "NCAAB",     # ✅ new
    }
    if 'Sport' in df.columns:
        df['Sport'] = df['Sport'].astype(str).str.strip()
        df['Sport'] = df['Sport'].replace(SPORT_KEY_TO_LABEL)
        df['Sport'] = df['Sport'].str.upper().replace({
            'BASEBALL_MLB': 'MLB',
            'BASKETBALL_NBA': 'NBA',
            'FOOTBALL_NFL': 'NFL',
            'FOOTBALL_NCAAF': 'NCAAF',
            'BASKETBALL_WNBA': 'WNBA',
            'CANADIANFOOTBALL_CFL': 'CFL',
            'BASKETBALL_NCAAB': 'NCAAB',
        })
        df['Sport'] = df['Sport'].astype('category')

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
        'Sharp_Time_Score', 'Sharp_Limit_Total', 'SharpBetScore',
        'Open_Value', 'Open_Book_Value', 'Opening_Limit', 'Limit_Jump',
        'Sharp_Timing', 'Limit_NonZero', 'Limit_Min', 'Market_Leader',
        'Is_Pinnacle', 'LimitUp_NoMove_Flag', 'SupportKey', 'CrossMarketSharpSupport',
        'Is_Reinforced_MultiMarket','Is_Sharp_Book',

        # Scoring / diagnostics
        'True_Sharp_Confidence_Score', 'Enhanced_Sharp_Confidence_Score',
        'Sharp_Confidence_Tier', 'Model_Sharp_Win_Prob', 'Model_Confidence_Tier',
        'Final_Confidence_Score', 'Model_Confidence', 'Snapshot_Timestamp',
        'Market_Norm', 'Outcome_Norm', 'Merge_Key_Short', 'Line_Hash',
        'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT', 'Scored', 'Pre_Game', 'Post_Game',
        'Unique_Sharp_Books', 'Sharp_Move_Magnitude_Score', 'Was_Canonical',
        'Scored_By_Model', 'Scoring_Market',
        'Line_Move_Magnitude',
        'Is_Home_Team_Bet',
        'Is_Favorite_Bet',
        'High_Limit_Flag',
        'Line_Delta',
        'Line_Magnitude_Abs',
        'Direction_Aligned','Odds_Price', 'Implied_Prob',
        'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds',
        'Value_Reversal_Flag', 'Odds_Reversal_Flag','Open_Odds',
        'Late_Game_Steam_Flag', 'Sharp_Line_Magnitude',
        'Rec_Line_Magnitude',
        'SharpMove_Odds_Up',
        'SharpMove_Odds_Down',
        'SharpMove_Odds_Mag',
        'Active_Signal_Count',
        'SharpMove_Magnitude_Overnight_VeryEarly', 'SharpMove_Magnitude_Overnight_MidRange',
        'SharpMove_Magnitude_Overnight_LateGame', 'SharpMove_Magnitude_Overnight_Urgent',
        'SharpMove_Magnitude_Early_VeryEarly', 'SharpMove_Magnitude_Early_MidRange',
        'SharpMove_Magnitude_Early_LateGame', 'SharpMove_Magnitude_Early_Urgent',
        'SharpMove_Magnitude_Midday_VeryEarly', 'SharpMove_Magnitude_Midday_MidRange',
        'SharpMove_Magnitude_Midday_LateGame', 'SharpMove_Magnitude_Midday_Urgent',
        'SharpMove_Magnitude_Late_VeryEarly', 'SharpMove_Magnitude_Late_MidRange',
        'SharpMove_Magnitude_Late_LateGame', 'SharpMove_Magnitude_Late_Urgent',
        'SharpMove_Timing_Dominant',
        'SharpMove_Timing_Magnitude',
        'Net_Line_Move_From_Opening',
        'Abs_Line_Move_From_Opening',
        'Net_Odds_Move_From_Opening',
        'Abs_Odds_Move_From_Opening',
    ]

    ALLOWED_ODDS_MOVE_COLUMNS = [
        f'OddsMove_Magnitude_{b}' for b in [
            'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
            'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
            'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
            'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
        ]
    ] + ['Odds_Move_Magnitude']
    ALLOWED_COLS += ALLOWED_ODDS_MOVE_COLUMNS

    # Ensure columns exist
    df = ensure_columns(df, ALLOWED_COLS, fill_value=None)
    df['Odds_Price']  = pd.to_numeric(df.get('Odds_Price'), errors='coerce')
    df['Implied_Prob'] = pd.to_numeric(df.get('Implied_Prob'), errors='coerce')

    # Required minimal columns
    required_cols = ['Snapshot_Timestamp', 'Model_Sharp_Win_Prob']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logging.error(f"❌ Missing required columns before upload: {missing}")
        return

    # Check for any ALLOWED_COLS not present (shouldn't happen after ensure_columns)
    missing_cols = [col for col in ALLOWED_COLS if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ Missing required columns for BigQuery upload: {missing_cols}")
        return

    # Deduplicate recent Line_Hash
    try:
        client = bigquery.Client(project=GCP_PROJECT_ID, location="us")
        df_prev = client.query("""
            SELECT DISTINCT Line_Hash
            FROM `sharplogger.sharp_data.sharp_moves_master`
            WHERE DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 DAY)
        """).to_dataframe()
        if not df_prev.empty and 'Line_Hash' in df.columns:
            before = len(df)
            df = df[~df['Line_Hash'].isin(df_prev['Line_Hash'])]
            logging.info(f"🧼 Line_Hash dedup: removed {before - len(df)} duplicate rows")
    except Exception as e:
        logging.warning(f"⚠️ Failed to deduplicate using Line_Hash: {e}")

    if df.empty:
        logging.info("🛑 No new rows after Line_Hash deduplication — exiting.")
        return

    logging.info(f"📦 Final row count to upload after filtering and dedup: {len(df)}")

    # Cast some known numeric columns explicitly (best effort)
    known_float_cols = [
        'SharpMove_Odds_Mag', 'HomeRecLineMag', 'Rec_Line_Delta', 'Sharp_Line_Delta',
        'Odds_Shift', 'Implied_Prob_Shift', 'Line_Delta',
        'Sharp_Line_Magnitude', 'Rec_Line_Magnitude', 'Delta_Sharp_vs_Rec',
        'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds'
    ]
    for col in known_float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    int_cols = [
        'Sharp_Limit_Total', 'Limit_Jump', 'LimitUp_NoMove_Flag',
        'SHARP_SIDE_TO_BET', 'Sharp_Move_Signal', 'Sharp_Limit_Jump',
        'Scored', 'Scored_By_Model', 'Is_Home_Team_Bet',
        'Is_Favorite_Bet', 'High_Limit_Flag', 'CrossMarketSharpSupport',
        'Value_Reversal_Flag', 'Odds_Reversal_Flag'
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')

    # ======= 🔎 BQ schema-aware preflight diagnostics (logs likely culprit) =======
    try:
        client = bigquery.Client(project=GCP_PROJECT_ID, location="us")
        tbl = client.get_table(f"sharplogger.{table}") if not table.startswith("sharplogger.") else client.get_table(table)
        bq_types = {f.name: f.field_type.upper() for f in tbl.schema}
        # summarize sample python types per column
        def _sample_types(s):
            vals = s.dropna().head(8).tolist()
            return list({type(v).__name__ for v in vals}) if vals else []

        # find suspicious pairs
        suspects = []
        for c, t in bq_types.items():
            if c not in df.columns:
                continue
            pytypes = _sample_types(df[c])
            if t == "BYTES" and any(tn in ("date","datetime","Timestamp","datetime64","TimestampTZ") for tn in pytypes):
                suspects.append((c, t, pytypes, "BYTES receiving date/datetime → ArrowTypeError likely"))
            if t in ("TIMESTAMP","DATETIME") and any(tn in ("str","bytes","bytearray") for tn in pytypes):
                suspects.append((c, t, pytypes, "time column receives strings/bytes"))
            if t == "BOOL" and any(tn in ("str","object") for tn in pytypes):
                suspects.append((c, t, pytypes, "BOOL receives strings"))
        if suspects:
            logging.warning("🔎 Preflight schema/type suspects (column, bq_type, sample_py_types, note):")
            for s in suspects:
                logging.warning("   • %s", s)

        # also log all BYTES columns & samples, if any
        bytes_cols = [c for c, t in bq_types.items() if t == "BYTES"]
        if bytes_cols:
            logging.info(f"📦 BYTES columns in BQ: {bytes_cols}")
            for c in bytes_cols:
                if c in df.columns:
                    logging.info(f"   ↳ {c} sample types: {_sample_types(df[c])}")
    except Exception as e:
        logging.warning(f"⚠️ Could not run BQ schema preflight: {e}")

    # Filter to allowed schema/ordering
    df = df[ALLOWED_COLS]

    logging.info("🧪 Preview of model columns being written:")
    try:
        logging.info(df[model_cols].dropna(how='all').head(5).to_string())
    except Exception:
        pass

    if 'Odds_Price' in df.columns and 'Implied_Prob' in df.columns:
        logging.info("🎯 Odds_Price sample:\n" + df['Odds_Price'].dropna().astype(str).head().to_string(index=False))
        logging.info("🎯 Implied_Prob sample:\n" + df['Implied_Prob'].dropna().round(4).astype(str).head().to_string(index=False))
    else:
        logging.warning("⚠️ Odds_Price or Implied_Prob missing from DataFrame before upload")

    logging.info(f"📦 Final row count to upload after filtering and dedup: {len(df)}")
    df = coerce_for_bq(df) 
    # ======= 🚀 Write to BigQuery (with rich error diagnostics) =======
    try:
        logging.info(f"📤 Uploading to `{table}`...")
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='append')
        logging.info(f"✅ Wrote {len(df)} new rows to `{table}`")
    except Exception as e:
        logging.exception(f"❌ Upload to `{table}` failed.")
        # dump dtypes & first rows
        logging.debug("Schema (pandas dtypes):\n" + df.dtypes.to_string())
        logging.debug("Preview (head):\n" + df.head(5).to_string())

        # Deep-dive: show, per BQ field, the pandas dtype and sample python types
        try:
            client = bigquery.Client(project=GCP_PROJECT_ID, location="us")
            tbl = client.get_table(f"sharplogger.{table}") if not table.startswith("sharplogger.") else client.get_table(table)
            bq_types = {f.name: f.field_type.upper() for f in tbl.schema}

            def _sample_types(s):
                vals = s.dropna().head(8).tolist()
                return list({type(v).__name__ for v in vals}) if vals else []

            logging.error("🔬 Column-by-column type map vs BQ:")
            for c in df.columns:
                bq_t = bq_types.get(c, "<not-in-BQ>")
                logging.error(" • %-30s pandas=%-12s   BQ=%-10s   samples=%s",
                              c, str(df[c].dtype), bq_t, _sample_types(df[c]))

            # Special highlight for BYTES columns getting date-likes
            bytes_cols = [c for c, t in bq_types.items() if t == "BYTES"]
            for c in bytes_cols:
                if c in df.columns:
                    samples = df[c].dropna().head(20).tolist()
                    bad = [v for v in samples if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64))]
                    if bad:
                        logging.error(f"🚨 BYTES column '{c}' has date-like sample values (showing up to 5): {bad[:5]}")
        except Exception as ee:
            logging.warning(f"⚠️ Post-failure schema diff also failed: {ee}")

        # re-raise if you want the caller to know it failed
        # raise


def normalize_label(label):
     return str(label).strip().lower().replace('.0', '')

def normalize_book_key(raw_key, sharp_books, rec_books):
    raw_key = raw_key.lower()

    # First: Exact matches for sharp books (preserve unique identifiers like betfair_uk/eu)
    for sharp in sharp_books:
        if raw_key == sharp:
            return sharp

    # Second: Exact matches for rec books
    for rec in rec_books:
        if raw_key == rec:
            return rec

    # Third: Fallback to partial match — but only if no exact match occurred
    for rec in rec_books:
        if rec.replace(" ", "") in raw_key:
            return rec.replace(" ", "")
    for sharp in sharp_books:
        if sharp in raw_key:
            return sharp

    # ✅ Final fallback — return normalized key as-is
    return raw_key

from google.cloud import bigquery
import logging
import time

# Map for sport aliasing in SQL

# Optional per-run cache to avoid repeated BQ hits
_market_weights_cache = {}

def load_market_weights_from_bq(
    sport_label: str,
    days_back: int = 14,
    project: str = "sharplogger",
    dataset_table: str = "sharplogger.sharp_data.sharp_scores_full",
    use_cache: bool = True,
):
    """
    Load recent rows for one sport and compute market weights.
    Returns a dict; never raises (logs and returns {} on error/empty).
    """
    t0 = time.time()
    sport = (sport_label or "").strip().upper()
    key = (sport, days_back)

    if use_cache and key in _market_weights_cache:
        return _market_weights_cache[key]

    aliases = SPORT_ALIASES.get(sport, [sport])

    needed_cols = [
        "Sport", "Market", "Outcome", "Bookmaker", "Value",
        "Model_Sharp_Win_Prob", "SHARP_HIT_BOOL", "Scored", "Snapshot_Timestamp"
    ]
    select_cols = ", ".join(needed_cols)

    query = f"""
      SELECT {select_cols}
      FROM `{dataset_table}`
      WHERE Scored = TRUE
        AND SHARP_HIT_BOOL IS NOT NULL
        AND UPPER(Sport) IN UNNEST(@sports)
        AND Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days_back DAY)
    """

    try:
        client = bigquery.Client(project=project, location="us")
        job = client.query(
            query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("sports", "STRING", aliases),
                    bigquery.ScalarQueryParameter("days_back", "INT64", days_back),
                ]
            ),
        )
        df = job.to_dataframe(create_bqstorage_client=True)
    except Exception as e:
        logging.warning(f"⚠️ Weights query failed (sport={sport}, {days_back}d): {e}")
        return {}

    if df.empty:
        logging.info(f"ℹ️ No rows for weights (sport={sport}, window={days_back}d).")
        return {}

    try:
        # Prefer a pure compute function; avoid writing from a 'load' function.
        weights = compute_market_weights(df)  # <-- extract from your compute_and_write_... if possible
        # If you must keep the current API:
        # weights = compute_and_write_market_weights(df)
    except Exception as e:
        logging.warning(f"⚠️ Failed computing weights (sport={sport}): {e}")
        return {}

    logging.info(
        f"✅ Loaded market weights for {len(weights) if weights else 0} markets "
        f"(sport={sport}, {days_back}d, {time.time()-t0:.1f}s)"
    )

    if use_cache:
        _market_weights_cache[key] = weights
    return weights


def get_market_weights(sport_label, has_models, days_back=14):
    if not has_models:
        return {}
    key = (sport_label.upper(), days_back)
    if key in _market_weights_cache:
        return _market_weights_cache[key]
    mw = load_market_weights_from_bq(sport_label, days_back)
    _market_weights_cache[key] = mw
    return mw

def with_timeout(fn, timeout_s=60, fallback=lambda: {}):
    t0 = time.time()
    try:
        return fn()
    finally:
        if time.time() - t0 > timeout_s:
            logging.error(f"⏰ market weights load exceeded {timeout_s}s; using empty weights.")
            return fallback()


    
def implied_prob_to_point_move(prob_delta, base_odds=-110):
    """
    Convert a probability delta into an approximate odds point movement
    assuming the slope near a base odds value (default -110).
    """
    # Use slope near -110 (standard point spread odds)
    # At -110, implied prob ≈ 0.524, slope ≈ ~0.0045 per point
    approx_slope = 0.0045  # ← 1 point ≈ 0.0045 prob shift near -110
    
    point_equiv = prob_delta / approx_slope
    return point_equiv    


def compute_sharp_metrics(
    entries,
    open_val,
    mtype,
    label,
    gk=None,
    book=None,             # fallback if entries don't include book per row
    open_odds=None,
    opening_limit=None,
    *,
    eps_line=1e-6,         # ignore sub‑micropoint wiggles
    eps_prob=0.001,        # ≈0.1% implied‑prob change = real move
):
    """
    Sharp_Move_Signal logic (UPDATED COMBO):
      • Always requires odds to strengthen the selected side (prob ↑ for the side).
      • For spreads/totals:
          - If the line moves, it must also strengthen the side.
          - If the line is sticky (no meaningful change), odds‑strengthening alone can trigger.
      • For H2H: odds‑strengthening alone can trigger.
    """
    lg_debug = logging.getLogger().isEnabledFor(logging.DEBUG)

    mtype = (mtype or "").strip().lower()
    label = (label or "").strip().lower()

    if entries and not all(entries[i][2] <= entries[i+1][2] for i in range(len(entries)-1)):
        entries = sorted(entries, key=lambda x: x[2])

    move_mag_sum = 0.0
    odds_move_mag_pts = 0.0
    limit_score = 0.0
    total_limit = 0.0

    mag_buckets  = [0.0]*16
    odds_buckets = [0.0]*16

    def _tod_idx(h):  return 0 if 0 <= h <= 5 else 1 if 6 <= h <= 11 else 2 if 12 <= h <= 15 else 3
    def _mtg_idx(m):  return 0 if (m is None or m > 720) else 1 if m > 180 else 2 if m > 60 else 3
    def _bucket_index(ts, game_start):
        try: h = ts.hour
        except Exception: h = pd.to_datetime(ts).hour
        mtg = None
        if game_start is not None:
            try: mtg = (game_start - ts).total_seconds() / 60.0
            except Exception: mtg = (pd.to_datetime(game_start) - pd.to_datetime(ts)).total_seconds() / 60.0
        return _tod_idx(h) * 4 + _mtg_idx(mtg)

    try:
        _SHARP_BOOKS = {str(b).strip().lower() for b in SHARP_BOOKS}  # noqa: F821
    except Exception:
        _SHARP_BOOKS = set()

    def _norm_book(b):
        return (str(b).strip().lower() if b is not None else str(book or "unknown").strip().lower())

    def _is_sharp_book(book_name, lim_val) -> bool:
        nb = _norm_book(book_name)
        if nb in _SHARP_BOOKS:
            return True
        try:
            return float(lim_val) >= 5000 if lim_val is not None else False
        except Exception:
            return False

    # ---- directional rules
    def _line_strengthening(prev_v, curr_v) -> bool:
        if prev_v is None or curr_v is None or pd.isna(prev_v) or pd.isna(curr_v):
            return False
        # spreads
        if mtype in ("spreads", "spread", "ats"):
            if abs(prev_v) < eps_line:     # pick'em → ignore
                return False
            if prev_v < 0:                 # favorite side
                return (curr_v + eps_line) < prev_v    # more negative
            if prev_v > 0:                 # underdog side
                return (curr_v - eps_line) > prev_v    # more positive
            return False
        # totals
        if mtype in ("totals", "total", "o/u", "ou"):
            is_over  = ("over" in label) or (label == "o")
            is_under = ("under" in label) or (label == "u")
            if is_over:
                return (curr_v - prev_v) > eps_line     # total rising
            if is_under:
                return (prev_v - curr_v) > eps_line     # total falling
            return False
        # h2h has no line concept
        return False

    def _odds_strengthening(prev_odds, curr_odds) -> tuple[bool, float]:
        if prev_odds is None or curr_odds is None or pd.isna(prev_odds) or pd.isna(curr_odds):
            return (False, 0.0)
        pp = implied_prob(prev_odds)   # noqa: F821
        cp = implied_prob(curr_odds)   # noqa: F821
        dp = cp - pp                    # prob ↑ means stronger tilt toward this side
        return (dp > eps_prob, dp)

    first_val   = open_val
    first_odds  = open_odds
    first_limit = opening_limit

    prev_val_by_book  = {}
    prev_odds_by_book = {}

    net_line_move = abs_net_line_move = None
    net_odds_move_px = abs_net_odds_move_px = None
    net_odds_move_prob = abs_net_odds_move_prob = None

    sharp_move_seen = False
    line_ever_changed_by_book = {}  # book → bool

    for i, row in enumerate(entries):
        if len(row) >= 6:
            lim, curr_val, ts, game_start, curr_odds, row_book = row[:6]
        else:
            lim, curr_val, ts, game_start, curr_odds = row[:5]
            row_book = book

        # coerce numeric
        try:
            if lim is not None and not isinstance(lim, (int, float)):
                lim = pd.to_numeric(lim, errors="coerce")
            if curr_val is not None and not isinstance(curr_val, (int, float)):
                curr_val = pd.to_numeric(curr_val, errors="coerce")
            if curr_odds is not None and not isinstance(curr_odds, (int, float)):
                curr_odds = pd.to_numeric(curr_odds, errors="coerce")
        except Exception:
            pass

        if first_val is None and pd.notna(curr_val):
            first_val = curr_val
        if first_odds is None and pd.notna(curr_odds):
            first_odds = curr_odds
        if first_limit is None and pd.notna(lim):
            first_limit = lim

        b_idx = _bucket_index(ts, game_start)
        nb = _norm_book(row_book)

        pval = prev_val_by_book.get(nb)
        pods = prev_odds_by_book.get(nb)

        # track if line ever changed meaningfully for this book
        if pd.notna(pval) and pd.notna(curr_val):
            if abs(curr_val - pval) > eps_line:
                line_ever_changed_by_book[nb] = True

        # ----- magnitude buckets (abs)
        if pd.notna(pval) and pd.notna(curr_val):
            delta_pts = float(curr_val - pval)
            mag_buckets[b_idx] += abs(delta_pts)
            move_mag_sum += abs(delta_pts)

        if pd.notna(pods) and pd.notna(curr_odds):
            pp = implied_prob(pods)          # noqa: F821
            cp = implied_prob(curr_odds)     # noqa: F821
            dp = cp - pp
            odds_buckets[b_idx] += abs(dp)
            point_equiv = implied_prob_to_point_move(dp)  # noqa: F821
            odds_move_mag_pts += abs(float(point_equiv))

        # ----- net-from-opening (report using last observed row)
        if pd.notna(first_val) and pd.notna(curr_val):
            net_line_move     = float(curr_val - first_val)
            abs_net_line_move = abs(net_line_move)
        if pd.notna(first_odds) and pd.notna(curr_odds):
            net_odds_move_px      = float(curr_odds - first_odds)
            abs_net_odds_move_px  = abs(net_odds_move_px)
            net_odds_move_prob     = implied_prob(curr_odds) - implied_prob(first_odds)  # noqa: F821
            abs_net_odds_move_prob = abs(net_odds_move_prob)

        # ===== COMBO TRIGGER (core change) =====
        if _is_sharp_book(nb, lim) and pd.notna(pods) and pd.notna(curr_odds):
            odds_ok, _dp = _odds_strengthening(pods, curr_odds)
            if odds_ok:
                if mtype in ("spreads", "spread", "ats", "totals", "total", "o/u", "ou"):
                    # require line to be strengthening IF it moved; else allow odds-only if sticky
                    line_moved = (pd.notna(pval) and pd.notna(curr_val) and abs(curr_val - pval) > eps_line)
                    if line_moved:
                        if _line_strengthening(pval, curr_val):
                            sharp_move_seen = True
                    else:
                        # sticky line: odds-strengthening alone suffices
                        sharp_move_seen = True
                else:
                    # h2h: odds-strengthening alone
                    sharp_move_seen = True

        # advance book-local prevs
        if pd.notna(curr_val):
            prev_val_by_book[nb] = curr_val
        if pd.notna(curr_odds):
            prev_odds_by_book[nb] = curr_odds

        if lg_debug and (i < 3 or i == len(entries)-1):
            logging.debug("…%d/%d val=%s odds=%s lim=%s book=%s",
                          i+1, len(entries), curr_val, curr_odds, lim, nb)

    # dominant timing label
    dom_src = mag_buckets if any(mag_buckets) else (odds_buckets if any(odds_buckets) else None)
    if dom_src is not None:
        dom_idx = max(range(16), key=lambda k: abs(dom_src[k]))
        tods = ["Overnight", "Early", "Midday", "Late"]
        mtgs = ["VeryEarly", "MidRange", "LateGame", "Urgent"]
        dominant_label = f"{tods[dom_idx//4]}_{mtgs[dom_idx%4]}"
    else:
        dominant_label = "unknown"

    first_imp_prob = implied_prob(first_odds) if pd.notna(first_odds) else None  # noqa: F821

    out = {
        "Open_Value": first_val,
        "Open_Odds": first_odds,
        "Opening_Limit": first_limit,
        "First_Imp_Prob": first_imp_prob,

        # combo rule
        "Sharp_Move_Signal": int(bool(sharp_move_seen)),

        # magnitudes
        "Sharp_Line_Magnitude": round(move_mag_sum, 3),
        "Odds_Move_Magnitude": round(odds_move_mag_pts, 3),
        "Sharp_Limit_Jump": int(limit_score >= 10000),
        "Sharp_Limit_Total": round(total_limit, 1),

        # timing
        "SharpMove_Timing_Dominant": dominant_label,

        # net-from-opening (line)
        "Net_Line_Move_From_Opening": round(net_line_move, 3) if net_line_move is not None else None,
        "Abs_Line_Move_From_Opening": round(abs_net_line_move, 3) if abs_net_line_move is not None else None,

        # net-from-opening (odds)
        "Net_Odds_Move_From_Opening": round(net_odds_move_px, 3) if net_odds_move_px is not None else None,
        "Abs_Odds_Move_From_Opening": round(abs_net_odds_move_px, 3) if abs_net_odds_move_px is not None else None,
        "Net_Odds_Prob_Move_From_Opening": round(net_odds_move_prob, 6) if net_odds_move_prob is not None else None,
        "Abs_Odds_Prob_Move_From_Opening": round(abs_net_odds_move_prob, 6) if abs_net_odds_move_prob is not None else None,

        "SharpMove_Timing_Magnitude": round(move_mag_sum, 3),
        "SharpBetScore": 0.0,
    }

    # attach 16×2 timing buckets
    tods = ["Overnight", "Early", "Midday", "Late"]
    mtgs = ["VeryEarly", "MidRange", "LateGame", "Urgent"]
    for ti, tod in enumerate(tods):
        for mi, mtg in enumerate(mtgs):
            idx = ti*4 + mi
            out[f"SharpMove_Magnitude_{tod}_{mtg}"] = round(mag_buckets[idx], 3)
            out[f"OddsMove_Magnitude_{tod}_{mtg}"]  = round(odds_buckets[idx], 6)

    return out
def apply_compute_sharp_metrics_rowwise(
    df: pd.DataFrame,
    df_all_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute sharp/odds timing metrics ONLY when valid snapshot data exists.
    - Does NOT create default timing bucket columns.
    - Does NOT fill with zeros/NaNs for missing groups.
    - Rollups are computed only if their inputs exist on the merged frame.
    """
    if df is None or df.empty or df_all_snapshots is None or df_all_snapshots.empty:
        return df

    # ---- required join keys
    keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in df: {missing}")

    _tod = ['Overnight','Early','Midday','Late']
    _mtg = ['VeryEarly','MidRange','LateGame','Urgent']

    # ── skinny snapshots ────────────────────────────────────────────────────────
    snap_needed = list(dict.fromkeys(keys + [
        'Limit', 'Value', 'Snapshot_Timestamp', 'Game_Start', 'Odds_Price'
    ]))
    snaps = df_all_snapshots.loc[:, [c for c in snap_needed if c in df_all_snapshots.columns]].copy()
    if snaps.empty:
        return df  # nothing to compute, do not add columns

    # normalize key columns (lower/strip, keep string dtype)
    def _to_str_col(s: pd.Series) -> pd.Series:
        if str(s.dtype).startswith('string'):
            out = s
        else:
            try:
                out = s.astype('string[pyarrow]')
            except Exception:
                out = s.astype(str)
        return out.str.strip().str.lower()

    for k in (set(keys) & set(snaps.columns)):
        snaps[k] = _to_str_col(snaps[k])
    for k in (set(keys) & set(df.columns)):
        df[k] = _to_str_col(df[k])

    if 'Snapshot_Timestamp' in snaps.columns:
        snaps['Snapshot_Timestamp'] = pd.to_datetime(snaps['Snapshot_Timestamp'], errors='coerce', utc=True)
    if 'Game_Start' in snaps.columns:
        snaps['Game_Start'] = pd.to_datetime(snaps['Game_Start'], errors='coerce', utc=True)

    for c in ('Limit', 'Value', 'Odds_Price'):
        if c in snaps.columns:
            snaps[c] = pd.to_numeric(snaps[c], errors='coerce').astype('float32', copy=False)

    # keep only snaps for keys present in df (saves work and avoids stray keys)
    snaps = snaps.merge(df[keys].drop_duplicates(), on=keys, how='inner')
    if snaps.empty:
        return df

    sort_cols = [c for c in keys if c in snaps.columns]
    if 'Snapshot_Timestamp' in snaps.columns:
        sort_cols.append('Snapshot_Timestamp')
    if sort_cols:
        snaps.sort_values(sort_cols, inplace=True, kind='mergesort')

    # ── compute metrics per valid group (require >=2 timestamps and some data) ──
    records = []
    append = records.append

    gb = snaps.groupby(keys, sort=False, observed=True, dropna=False)
    for name, g in gb:
        # validity: at least 2 snapshots and at least one signal dimension present
        n_ts = int(g['Snapshot_Timestamp'].notna().sum()) if 'Snapshot_Timestamp' in g else 0
        has_val  = 'Value' in g and g['Value'].notna().any()
        has_odds = 'Odds_Price' in g and g['Odds_Price'].notna().any()
        
        if n_ts < 2 and not (has_val or has_odds):
            # Emit a zeroed record so merge creates the columns
            append({
                keys[0]: name[0], keys[1]: name[1], keys[2]: name[2], keys[3]: name[3],
                'Open_Value': None, 'Open_Odds': None, 'Opening_Limit': None, 'First_Imp_Prob': None,
                'Sharp_Move_Signal': 0, 'Sharp_Line_Magnitude': 0.0, 'Odds_Move_Magnitude': 0.0,
                'Sharp_Limit_Jump': 0, 'Sharp_Limit_Total': 0.0,
                'SharpMove_Timing_Dominant': 'unknown'
            })
            continue

        game_start    = g['Game_Start'].dropna().iloc[0] if 'Game_Start' in g and g['Game_Start'].notna().any() else None
        open_val      = g['Value'].dropna().iloc[0] if has_val else None
        open_odds     = g['Odds_Price'].dropna().iloc[0] if has_odds else None
        opening_limit = g['Limit'].dropna().iloc[0] if 'Limit' in g and g['Limit'].notna().any() else None

        n = len(g)
        lim  = g['Limit'].to_numpy(dtype='float32', copy=False)      if 'Limit' in g else np.empty(n, dtype='float32')
        val  = g['Value'].to_numpy(dtype='float32', copy=False)      if 'Value' in g else np.empty(n, dtype='float32')
        ts   = g['Snapshot_Timestamp'].to_numpy(copy=False)          if 'Snapshot_Timestamp' in g else np.array([pd.NaT]*n, dtype='datetime64[ns]')
        odds = g['Odds_Price'].to_numpy(dtype='float32', copy=False) if 'Odds_Price' in g else np.empty(n, dtype='float32')
        entries = list(zip(lim, val, ts, np.repeat(game_start, n), odds))

        m = compute_sharp_metrics(
            entries=entries,
            open_val=open_val,
            mtype=name[1],
            label=name[2],
            gk=name[0],
            book=name[3],
            open_odds=open_odds,
            opening_limit=opening_limit,
        ) or {}

        # Only attach if compute_sharp_metrics produced anything useful.
        if m:
            rec = {keys[0]: name[0], keys[1]: name[1], keys[2]: name[2], keys[3]: name[3], **m}
            append(rec)

    if not records:
        return df  # nothing valid to add

    metrics_df = pd.DataFrame.from_records(records)

    # never override opener columns coming from upstream
    _opener_cols = ['Open_Value', 'Open_Odds', 'Opening_Limit', 'First_Imp_Prob']
    metrics_df = metrics_df.drop(columns=[c for c in _opener_cols if c in metrics_df.columns], errors='ignore')

    # merge back (left join), only columns that exist in metrics_df are introduced
    out = df.merge(metrics_df, on=keys, how='left', copy=False)

    # ---- optional: rollups ONLY if their inputs were produced
    sharp_cols_present = [c for t in _tod for m in _mtg
                          for c in [f'SharpMove_Magnitude_{t}_{m}']
                          if c in out.columns]
    odds_cols_present  = [c for t in _tod for m in _mtg
                          for c in [f'OddsMove_Magnitude_{t}_{m}']
                          if c in out.columns]

    if sharp_cols_present and 'SharpMove_Timing_Magnitude' not in out.columns:
        out['SharpMove_Timing_Magnitude'] = out[sharp_cols_present].sum(axis=1).astype('float32')
    if odds_cols_present and 'Odds_Move_Magnitude' not in out.columns:
        out['Odds_Move_Magnitude'] = out[odds_cols_present].sum(axis=1).astype('float32')

    if sharp_cols_present and 'SharpMove_Timing_Dominant' not in out.columns:
        max_col = out[sharp_cols_present].idxmax(axis=1)
        dom = (max_col.fillna('')
                      .str.replace('SharpMove_Magnitude_', '', regex=False)
                      .str.replace('_', '/', regex=False))
        has_mass = out[sharp_cols_present].sum(axis=1) > 0
        out.loc[has_mass, 'SharpMove_Timing_Dominant'] = dom[has_mass].astype('string')

    # numeric coercion only for columns that exist (no creation)
    for c in sharp_cols_present + odds_cols_present + [
        'SharpMove_Timing_Magnitude','Odds_Move_Magnitude',
        'Net_Line_Move_From_Opening','Abs_Line_Move_From_Opening',
        'Net_Odds_Move_From_Opening','Abs_Odds_Move_From_Opening',
        'Sharp_Move_Signal','Sharp_Line_Magnitude','Sharp_Limit_Jump','Sharp_Limit_Total'
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').astype('float32', copy=False)

    return out
    

    
    
SPORT_ALIAS = {
    'AMERICANFOOTBALL_CFL': 'CFL',
    'BASEBALL_MLB': 'MLB',
    'BASKETBALL_WNBA': 'WNBA',
    'AMERICANFOOTBALL_NFL': 'NFL',
    'AMERICANFOOTBALL_NCAAF': 'NCAAF',
    'BASKETBALL_NBA': 'NBA',
    'BASKETBALL_NCAAB': 'NCAAB',
    'CFL': 'CFL',
    'MLB': 'MLB',
    'WNBA': 'WNBA',
    'NFL': 'NFL',
    'NCAAF': 'NCAAF',
    'NBA': 'NBA',
    'NCAAB': 'NCAAB'
}



KEY_LEVELS = {
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
    ("ncaab","totals"):  [158.5,160.5,162.5,164.5,166.5,168.5],
}

def _keys_for(sport: str, market: str) -> np.ndarray:
    sport = (sport or "").strip().lower()
    market = (market or "").strip().lower()
    ks = KEY_LEVELS.get((sport, market))
    if ks is None:
        ks = [1.5,2.5,3,3.5,6,6.5,7] if market == "spreads" else ([7,7.5,8,8.5,9] if market=="totals" else [])
    return np.asarray(sorted(ks), dtype=np.float32)

def add_resistance_features_lowmem(
    df: pd.DataFrame,
    *,
    sport_col: str = "Sport",
    market_col: str = "Market",
    value_col: str = "Value",
    odds_col: str = "Odds_Price",
    open_col: str = "Open_Value",
    open_odds_col: str = "Open_Odds",
    sharp_move_col: str | None = "Sharp_Move_Signal",
    sharp_prob_shift_col: str | None = "Sharp_Prob_Shift",
    emit_levels_str: bool = False,
    broadcast: bool = True,
    skip_sort: bool = False,
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        for c in ("Was_Line_Resistance_Broken","Line_Resistance_Crossed_Count","SharpMove_Resistance_Break"):
            out[c] = 0
        if emit_levels_str:
            out["Line_Resistance_Crossed_Levels_Str"] = ""
        return out

    df = df.copy()

    # optional sort only for snapshot-shaped data
    if (not skip_sort) and ("Snapshot_Timestamp" in df.columns):
        if not pd.api.types.is_datetime64_any_dtype(df["Snapshot_Timestamp"]):
            df["Snapshot_Timestamp"] = pd.to_datetime(df["Snapshot_Timestamp"], errors="coerce")
        df = df.sort_values("Snapshot_Timestamp")

    # ensure one row per key
    keys_base = [c for c in ["Game_Key","Market","Outcome","Bookmaker"] if c in df.columns]
    if keys_base:
        df = df.drop_duplicates(keys_base, keep="last")

    # hydrate fallbacks for opens
    if open_col not in df.columns:
        for alt in ["First_Line_Value", value_col]:
            if alt in df.columns:
                df[open_col] = df[alt]; break
        if open_col not in df.columns:
            df[open_col] = np.nan

    if open_odds_col not in df.columns:
        for alt in ["First_Odds", odds_col]:
            if alt in df.columns:
                df[open_odds_col] = df[alt]; break
        if open_odds_col not in df.columns:
            df[open_odds_col] = np.nan

    # numeric coercions
    for c in (value_col, open_col, odds_col, open_odds_col):
        if c in df.columns and df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # working slice
    requested = list({*(keys_base or []), sport_col, market_col, value_col, odds_col,
                      open_col, open_odds_col, sharp_move_col, sharp_prob_shift_col})
    pick_cols = [c for c in requested if c and c in df.columns]
    tmp = df[pick_cols].copy()

    # normalize
    tmp["_sport_"]  = tmp[sport_col].astype(str).str.strip().str.lower()
    tmp["_market_"] = tmp[market_col].astype(str).str.strip().str.lower()
    o = pd.to_numeric(tmp[open_col],  errors="coerce", downcast="float").astype("float32", copy=False)
    v = pd.to_numeric(tmp[value_col], errors="coerce", downcast="float").astype("float32", copy=False)

    n = len(tmp)
    crossed_count = np.zeros(n, dtype=np.int8)
    broken        = np.zeros(n, dtype=bool)
    levels_str    = np.full(n, "", dtype=object) if emit_levels_str else None

    # vectorized per (sport, market)
    for (sp, mk), idx in tmp.groupby(["_sport_","_market_"]).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        keys = _keys_for(sp, mk)
        if keys.size == 0:
            continue

        is_spread = (mk == "spreads")
        o_slice = o[idx]
        v_slice = v[idx]
        valid   = (~np.isnan(o_slice)) & (~np.isnan(v_slice))
        if not valid.any():
            continue

        a = np.where(is_spread, np.abs(o_slice), o_slice)
        b = np.where(is_spread, np.abs(v_slice), v_slice)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)

        # count crossed keys (exclusive bounds)
        lo_v = lo[valid]
        hi_v = hi[valid]
        lo_idx = np.searchsorted(keys, lo_v, side="right")
        hi_idx = np.searchsorted(keys, hi_v, side="left")
        cnt = (hi_idx - lo_idx).astype(np.int8)

        target_rows = idx[valid]
        crossed_count[target_rows] = cnt
        broken[target_rows] = cnt > 0

        if emit_levels_str and np.any(cnt > 0):
            # build strings using relative indices (fixes indexing bug)
            kpos = np.flatnonzero(cnt > 0)
            for k in kpos:
                j = target_rows[k]           # global row index
                s = keys[(keys > lo_v[k]) & (keys < hi_v[k])]
                if s.size:
                    levels_str[j] = "|".join(str(float(x)).rstrip("0").rstrip(".") for x in s)

    tmp["Line_Resistance_Crossed_Count"] = crossed_count
    tmp["Was_Line_Resistance_Broken"]    = broken.astype(np.uint8)
    if emit_levels_str:
        tmp["Line_Resistance_Crossed_Levels_Str"] = levels_str

    if sharp_move_col in tmp.columns:
        sm = pd.to_numeric(tmp[sharp_move_col], errors="coerce").fillna(0).astype(np.int8)
        tmp["SharpMove_Resistance_Break"] = ((tmp["Was_Line_Resistance_Broken"] == 1) & (sm == 1)).astype(np.uint8)
    elif sharp_prob_shift_col in tmp.columns:
        ps = pd.to_numeric(tmp[sharp_prob_shift_col], errors="coerce").fillna(0.0).astype("float32")
        tmp["SharpMove_Resistance_Break"] = ((tmp["Was_Line_Resistance_Broken"] == 1) & (ps > 0)).astype(np.uint8)
    else:
        tmp["SharpMove_Resistance_Break"] = tmp["Was_Line_Resistance_Broken"].astype(np.uint8)

    keep_cols = ["Line_Resistance_Crossed_Count","Was_Line_Resistance_Broken","SharpMove_Resistance_Break"]
    if emit_levels_str:
        keep_cols.append("Line_Resistance_Crossed_Levels_Str")

    # merge back (guard empty key set)
    if keys_base:
        df = df.merge(tmp[keys_base + keep_cols], on=keys_base, how="left", copy=False)
    else:
        # align by index if no keys to merge on
        for c in keep_cols:
            df[c] = tmp[c].values

    df["Line_Resistance_Crossed_Count"] = df["Line_Resistance_Crossed_Count"].fillna(0).astype("int16")
    df["Was_Line_Resistance_Broken"]    = df["Was_Line_Resistance_Broken"].fillna(0).astype("uint8")
    df["SharpMove_Resistance_Break"]    = df["SharpMove_Resistance_Break"].fillna(0).astype("uint8")
    if emit_levels_str:
        df["Line_Resistance_Crossed_Levels_Str"] = df["Line_Resistance_Crossed_Levels_Str"].fillna("")

    df.drop(columns=["_sport_","_market_"], inplace=True, errors="ignore")
    return df



def compute_sharp_magnitude_by_time_bucket(df_all_snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Memory-lean version:
    - Works on a skinny, normalized copy of snapshots
    - Sorts ONCE globally by keys+time
    - Iterates groups with zero-copy numpy views (no per-row iterrows)
    - Avoids 'category' dtypes entirely
    """
    if df_all_snapshots is None or df_all_snapshots.empty:
        return pd.DataFrame()

    keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
    need = list(dict.fromkeys(keys + ['Limit', 'Value', 'Odds_Price', 'Snapshot_Timestamp', 'Game_Start']))


    snaps = df_all_snapshots.loc[:, [c for c in need if c in df_all_snapshots.columns]].copy()

    # ---- normalize keys (string, lower) ----
    def _s(s: pd.Series) -> pd.Series:
        if str(s.dtype).startswith('string'):
            out = s
        else:
            try:
                out = s.astype('string[pyarrow]')
            except Exception:
                out = s.astype(str)
        return out.str.strip().str.lower()

    for k in (set(keys) & set(snaps.columns)):
        snaps[k] = _s(snaps[k])

    # ---- datetimes + downcast numerics ----
    if 'Snapshot_Timestamp' in snaps.columns:
        snaps['Snapshot_Timestamp'] = pd.to_datetime(snaps['Snapshot_Timestamp'], errors='coerce', utc=True)
    if 'Game_Start' in snaps.columns:
        snaps['Game_Start'] = pd.to_datetime(snaps['Game_Start'], errors='coerce', utc=True)
    for c in ('Limit', 'Value'):
        if c in snaps.columns:
            snaps[c] = pd.to_numeric(snaps[c], errors='coerce').astype('float32', copy=False)

    # ---- global stable sort once ----
    sort_cols = [c for c in keys if c in snaps.columns]
    if 'Snapshot_Timestamp' in snaps.columns:
        sort_cols.append('Snapshot_Timestamp')
    if sort_cols:
        snaps.sort_values(sort_cols, inplace=True, kind='mergesort')

    # ---- group & compute with zero-copy views ----
    gb = snaps.groupby(keys, sort=False, observed=True, dropna=False)

    recs = []
    append = recs.append
    dbg = logging.getLogger(__name__).isEnabledFor(logging.DEBUG)

    for name, g in gb:
        if dbg:
            logging.debug("📊 Group: %s", name)

        n = len(g)
        if n == 0:
            continue

        # zero-copy arrays
        val  = g['Value'].to_numpy(dtype='float32', copy=False)               if 'Value' in g else np.empty(0, 'float32')
        lim  = g['Limit'].to_numpy(dtype='float32', copy=False)               if 'Limit' in g else np.empty(n, 'float32')
        ts   = g['Snapshot_Timestamp'].to_numpy(copy=False)                   if 'Snapshot_Timestamp' in g else np.array([pd.NaT]*n, dtype='datetime64[ns]')
        gstart = g['Game_Start'].dropna().iloc[0] if 'Game_Start' in g and g['Game_Start'].notna().any() else None

        # opener (first non-nan after sort)
        if val.size and np.any(~np.isnan(val)):
            first_idx = np.flatnonzero(~np.isnan(val))[0]
            open_val = float(val[first_idx])
        else:
            open_val = None

        # build entries with minimal Python overhead
        entries = list(zip(lim, val, ts, np.repeat(gstart, n)))

        metrics = compute_sharp_metrics(entries, open_val, name[1], name[2])  # (entries, open_val, market, outcome)

        rec = {
            'Game_Key': name[0],
            'Market':   name[1],
            'Outcome':  name[2],
            'Bookmaker':name[3],
        }
        rec.update(metrics)
        append(rec)

    return pd.DataFrame.from_records(recs)


def add_minutes_to_game(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized, low-copy, no 'category'.
    - Computes Minutes_To_Game as int32 (clipped >= 0)
    - Timing_Tier via np.select -> string dtype (not categorical)
    """
    if df is None or df.empty:
        return df

    # Avoid full-frame copy; operate in place (enable pandas CoW globally if desired)
    out = df

    # Ensure timestamps exist
    has_commence = 'Commence_Hour' in out.columns
    has_start    = 'Game_Start' in out.columns
    has_snap     = 'Snapshot_Timestamp' in out.columns

    if not has_snap or (not has_commence and not has_start):
        logging.warning("⏳ Skipping time-to-game calculation — missing Commence_Hour/Game_Start or Snapshot_Timestamp")
        out['Minutes_To_Game'] = np.nan
        out['Timing_Tier'] = pd.Series(pd.array([], dtype="string[pyarrow]")).reindex(out.index)
        return out

    # Datetime normalize
    out['Snapshot_Timestamp'] = pd.to_datetime(out['Snapshot_Timestamp'], errors='coerce', utc=True)
    if has_commence:
        out['Commence_Hour'] = pd.to_datetime(out['Commence_Hour'], errors='coerce', utc=True)
        commence = out['Commence_Hour'].to_numpy(copy=False)
    else:
        out['Game_Start'] = pd.to_datetime(out['Game_Start'], errors='coerce', utc=True)
        # floor to hour to preserve your original intent
        commence = out['Game_Start'].dt.floor('h').to_numpy(copy=False)

    snap = out['Snapshot_Timestamp'].to_numpy(copy=False)

    # Minutes as int32 (clip >= 0)
    mins = ((commence - snap) / np.timedelta64(1, 'm'))
    # convert to float64 numpy -> clip -> nan->keep -> cast
    mins = np.where(mins < 0, 0, mins)
    # keep NaNs (cannot cast NaN to int32 directly)
    minutes_int = mins.astype('float64')
    out['Minutes_To_Game'] = pd.Series(minutes_int, index=out.index)

    # Timing tier via np.select (strings, not category)
    conds = [
        (minutes_int < 60),
        (minutes_int >= 60) & (minutes_int < 360),
        (minutes_int >= 360) & (minutes_int < 1440),
        (minutes_int >= 1440),
    ]
    choices = [
        '🔥 Late (<1h)',
        '⚠️ Mid (1–6h)',
        '⏳ Early (6–24h)',
        '🧊 Very Early (>24h)',
    ]
    tier = np.select(conds, choices, default=None)

    try:
        out['Timing_Tier'] = pd.array(tier, dtype='string[pyarrow]')
    except Exception:
        out['Timing_Tier'] = pd.Series(tier, dtype='string')

    return out


from scipy.stats import zscore

def add_line_and_crossmarket_features(df):
    # === Normalize sport/market
    SPORT_ALIAS = {
        'MLB': 'MLB', 'NFL': 'NFL', 'CFL': 'CFL',
        'WNBA': 'WNBA', 'NBA': 'NBA', 'NCAAF': 'NCAAF', 'NCAAB': 'NCAAB'
    }
    df['Sport_Norm'] = df['Sport'].map(SPORT_ALIAS).fillna(df['Sport'])
    df['Market_Norm'] = df['Market'].str.lower()

    # === Absolute line + odds movement
    df['Abs_Line_Move_From_Opening'] = (df['Value'] - df['Open_Value']).abs()
    df['Odds_Shift'] = df['Odds_Price'] - df['Open_Odds']
    df['Implied_Prob'] = df['Odds_Price'].apply(implied_prob)
    df['First_Imp_Prob'] = df['Open_Odds'].apply(implied_prob)
    df['Implied_Prob_Shift'] = df['Implied_Prob'] - df['First_Imp_Prob']

    # === Directional movement
    df['Line_Moved_Toward_Team'] = np.where(
        ((df['Value'] > df['Open_Value']) & (df['Is_Favorite_Bet'] == 1)) |
        ((df['Value'] < df['Open_Value']) & (df['Is_Favorite_Bet'] == 0)),
        1, 0
    )

    df['Line_Moved_Away_From_Team'] = np.where(
        ((df['Value'] < df['Open_Value']) & (df['Is_Favorite_Bet'] == 1)) |
        ((df['Value'] > df['Open_Value']) & (df['Is_Favorite_Bet'] == 0)),
        1, 0
    )

    # === Percent line move (totals only)
    df['Pct_Line_Move_From_Opening'] = np.where(
        (df['Market_Norm'] == 'total') & (df['Open_Value'].abs() > 0),
        df['Abs_Line_Move_From_Opening'] / df['Open_Value'].abs(),
        np.nan
    )

    df['Pct_Line_Move_Bin'] = pd.cut(
        df['Pct_Line_Move_From_Opening'],
        bins=[-np.inf, 0.0025, 0.005, 0.01, 0.02, np.inf],
        labels=['<0.25%', '0.5%', '1%', '2%', '2%+']
    )

    # === Disable flags for unsupported markets
    df['Disable_Line_Move_Features'] = np.where(
        ((df['Sport_Norm'] == 'MLB') & (df['Market_Norm'] == 'spread')) |
        (df['Market_Norm'].isin(['h2h', 'moneyline'])),
        1, 0
    )

    # === Z-score based overmove detection
    df['Abs_Line_Move_Z'] = (
        df.groupby(['Sport_Norm', 'Market_Norm'])['Abs_Line_Move_From_Opening']
        .transform(lambda x: zscore(x.fillna(0), ddof=0))
    ).clip(-5, 5)

    df['Pct_Line_Move_Z'] = (
        df.groupby(['Sport_Norm'])['Pct_Line_Move_From_Opening']
        .transform(lambda x: zscore(x.fillna(0), ddof=0))
    ).clip(-5, 5)

    df['Implied_Prob_Shift_Z'] = (
        df.groupby(['Sport_Norm', 'Market_Norm'])['Implied_Prob_Shift']
        .transform(lambda x: zscore(x.fillna(0), ddof=0))
    ).clip(-5, 5)

    df['Potential_Overmove_Flag'] = np.where(
        (df['Market_Norm'] == 'spread') &
        (df['Line_Moved_Toward_Team'] == 1) &
        (df['Abs_Line_Move_Z'] >= 2) &
        (df['Disable_Line_Move_Features'] == 0),
        1, 0
    )

    df['Potential_Overmove_Total_Pct_Flag'] = np.where(
        (df['Market_Norm'] == 'total') &
        (df['Line_Moved_Toward_Team'] == 1) &
        (df['Pct_Line_Move_Z'] >= 2) &
        (df['Disable_Line_Move_Features'] == 0),
        1, 0
    )

    df['Potential_Odds_Overmove_Flag'] = np.where(
        (df['Implied_Prob_Shift_Z'] >= 2),
        1, 0
    )

    # === Cross-market alignment
    df['Spread_Implied_Prob'] = df['Spread_Odds'].apply(implied_prob)
    df['H2H_Implied_Prob'] = df['H2H_Odds'].apply(implied_prob)
    df['Total_Implied_Prob'] = df['Total_Odds'].apply(implied_prob)

    df['Spread_vs_H2H_Aligned'] = np.where(
        (df['Market_Norm'] == 'spread') &
        (df['Value'] < 0) &  # favorite
        (df.get('H2H_Implied_Prob', 0) > 0.5),
        1, 0
    )
    df['Total_vs_Spread_Contradiction'] = (
        (df['Spread_Implied_Prob'] > 0.55) &
        (df['Total_Implied_Prob'] < 0.48)
    ).astype(int)

    df['Spread_vs_H2H_ProbGap'] = df['Spread_Implied_Prob'] - df['H2H_Implied_Prob']
    df['Total_vs_H2H_ProbGap'] = df['Total_Implied_Prob'] - df['H2H_Implied_Prob']
    df['Total_vs_Spread_ProbGap'] = df['Total_Implied_Prob'] - df['Spread_Implied_Prob']

    df['CrossMarket_Prob_Gap_Exists'] = (
        (df['Spread_vs_H2H_ProbGap'].abs() > 0.05) |
        (df['Total_vs_Spread_ProbGap'].abs() > 0.05)
    ).astype(int)

    return df


def compute_small_book_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds market-aware small-book liquidity features.
    Used in both background scoring and UI to ensure consistent fields.
    """
    SMALL_LIMIT_BOOKS = [
        'betfair_ex_uk','betfair_ex_eu','betfair_ex_au',
        'matchbook','smarkets'
    ]

    df = df.copy()

    # Normalize bookmaker field
    if 'Bookmaker_Norm' not in df.columns:
        df['Bookmaker_Norm'] = df['Bookmaker'].astype(str).str.lower().str.strip()

    df['Market'] = df['Market'].astype(str).str.lower().str.strip()
    df['Outcome'] = df['Outcome'].astype(str).str.lower().str.strip()

    # Flag small limit books
    df['Is_Small_Limit_Book'] = df['Bookmaker_Norm'].isin(SMALL_LIMIT_BOOKS).astype(int)

    # Ensure numeric limit
    df['Limit'] = pd.to_numeric(df.get('Limit', 0), errors='coerce').fillna(0)

    # Aggregate for small-limit rows
    try:
        agg = (
            df[df['Is_Small_Limit_Book'] == 1]
            .groupby(['Game_Key', 'Market', 'Outcome'])
            .agg(
                SmallBook_Total_Limit=('Limit', 'sum'),
                SmallBook_Max_Limit=('Limit', 'max'),
                SmallBook_Min_Limit=('Limit', 'min'),
                SmallBook_Limit_Count=('Limit', 'count')
            )
            .reset_index()
        )
    except Exception:
        agg = pd.DataFrame(columns=[
            'Game_Key','Market','Outcome',
            'SmallBook_Total_Limit','SmallBook_Max_Limit','SmallBook_Min_Limit','SmallBook_Limit_Count'
        ])

    df = df.merge(agg, on=['Game_Key', 'Market', 'Outcome'], how='left')

    for col in ['SmallBook_Total_Limit', 'SmallBook_Max_Limit', 'SmallBook_Min_Limit']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Skew ratio instead of raw diff
    df['SmallBook_Limit_Skew'] = np.where(
        df['SmallBook_Min_Limit'] > 0,
        df['SmallBook_Max_Limit'] / df['SmallBook_Min_Limit'],
        np.nan
    )

    # Flags — match UI thresholds
    HEAVY_TOTAL = 700
    SKEW_RATIO = 1.5
    df['SmallBook_Heavy_Liquidity_Flag'] = (df['SmallBook_Total_Limit'] >= HEAVY_TOTAL).astype(int)
    df['SmallBook_Limit_Skew_Flag'] = (df['SmallBook_Limit_Skew'] >= SKEW_RATIO).astype(int)

    return df


from scipy.special import ndtr as _phi      # CDF Φ (kept for compatibility elsewhere)
from scipy.special import ndtri as _ppf     # inverse Φ^{-1}

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

def _amer_to_prob(odds_like) -> np.ndarray | float:
    """
    American odds -> implied probability.
    Works for pandas Series / numpy arrays / lists / scalars.
    Returns an ndarray for vector inputs, or a scalar float for scalar input.
    """
    o = pd.to_numeric(odds_like, errors="coerce")

    # Scalar path
    if np.isscalar(o):
        if not np.isfinite(o):
            return np.nan
        return (100.0 / (o + 100.0)) if (o >= 0) else ((-o) / ((-o) + 100.0))

    # Vector path
    o = np.asarray(o, dtype="float64")
    if o.size == 0:
        return o  # empty
    p = np.full(o.shape, np.nan, dtype="float64")
    neg = o < 0
    pos = ~neg & np.isfinite(o)
    oo = -o
    p[neg] = oo[neg] / (oo[neg] + 100.0)
    p[pos] = 100.0 / (o[pos] + 100.0)
    return p

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
    Safe to run even if sharp data/sigma is missing; returns same number of rows.

    Writes/ensures:
      - Truth_Fair_Prob_at_SharpLine, Truth_Margin_Mu, Truth_Sigma
      - Truth_Fair_Prob_at_RecLine, Rec_Implied_Prob
      - EV_Sh_vs_Rec_Prob, EV_Sh_vs_Rec_Dollar, Kelly_Fraction
    """
    # --- local, empty-safe Φ so we don't depend on a global vectorize() variant ---
    def _phi_local(z):
        z = np.asarray(z, dtype=float)
        if z.size == 0:
            return z
        try:
            from scipy.special import ndtr
            return ndtr(z)
        except Exception:
            from numpy import special as nsp
            import math
            return 0.5 * (1.0 + nsp.erf(z / math.sqrt(2.0)))

    out_cols = [
        "Truth_Fair_Prob_at_SharpLine","Truth_Margin_Mu","Truth_Sigma",
        "Truth_Fair_Prob_at_RecLine","Rec_Implied_Prob",
        "EV_Sh_vs_Rec_Prob","EV_Sh_vs_Rec_Dollar","Kelly_Fraction"
    ]

    # empty guard
    if df_market is None or df_market.empty:
        dm = (df_market.copy() if df_market is not None else pd.DataFrame())
        for c in out_cols: dm[c] = np.nan
        return dm

    dm = df_market.copy()

    # normalize minimal keys used here
    dm["Market"] = dm.get("Market", "").astype(str).str.lower().str.strip()
    dm["Outcome_Norm"] = dm.get("Outcome_Norm", dm.get("Outcome", "")).astype(str).str.lower().str.strip()
    dm["Bookmaker"] = dm.get("Bookmaker", dm.get("Book", "")).astype(str).str.lower().str.strip()

    # pre-create outputs (avoids KeyError later)
    for c in out_cols:
        if c not in dm.columns:
            dm[c] = np.nan

    # identify sharp rows
    SHARP_SET = set(sharp_books or SHARP_BOOKS)
    sharp_mask = dm["Bookmaker"].isin(SHARP_SET)

    # if no sharp rows, still compute Rec_Implied_Prob and return
    if not sharp_mask.any():
        dm["Rec_Implied_Prob"] = np.asarray(_amer_to_prob(dm.get("Odds_Price", np.nan)), dtype="float64")
        return dm

    # collect sharp candidates
    keep = ["Game_Key","Market","Outcome_Norm","Bookmaker","Value","Odds_Price"]
    if reliability_col in dm.columns: keep.append(reliability_col)
    if limit_col in dm.columns:       keep.append(limit_col)
    sharp_rows = dm.loc[sharp_mask, keep].copy()

    if sharp_rows.empty:
        dm["Rec_Implied_Prob"] = np.asarray(_amer_to_prob(dm.get("Odds_Price", np.nan)), dtype="float64")
        return dm

    # rank sharp by reliability then limit (both optional)
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

    # build top-2 sharp pair per (Game_Key, Market) for de‑vig
    pairs_base = (sharp_rows
        .sort_values(["Game_Key","Market", reliability_col], ascending=[True,True,False])
        .drop_duplicates(subset=["Game_Key","Market","Outcome_Norm"], keep="first"))
    def _to_pair(df_):
        if len(df_) < 2:
            return pd.Series({"Outcome_A":np.nan,"Line_A":np.nan,"Odds_A":np.nan,
                              "Outcome_B":np.nan,"Line_B":np.nan,"Odds_B":np.nan})
        a, b = df_.iloc[0], df_.iloc[1]
        return pd.Series({"Outcome_A":a["Outcome_Norm"], "Line_A":a["Value"], "Odds_A":a["Odds_Price"],
                          "Outcome_B":b["Outcome_Norm"], "Line_B":b["Value"], "Odds_B":b["Odds_Price"]})
    top2 = (pairs_base.groupby(["Game_Key","Market"], as_index=False)
        .apply(lambda g: g.head(2)).reset_index(drop=True))
    sharp_pairs = top2.groupby(["Game_Key","Market"]).apply(_to_pair).reset_index()

    truth = sharp_ref.merge(sharp_pairs, on=["Game_Key","Market"], how="left")

    # de‑vig fair prob at the sharp line (if pair available)
    same_as_A = truth["Outcome_Norm"].astype(str).eq(truth["Outcome_A"].astype(str))
    odds_this = np.where(same_as_A, truth["Odds_A"], truth["Odds_B"])
    odds_opp  = np.where(same_as_A, truth["Odds_B"], truth["Odds_A"])
    p_this_raw = np.asarray(_amer_to_prob(odds_this), dtype="float64")
    p_opp_raw  = np.asarray(_amer_to_prob(odds_opp),  dtype="float64")
    s = p_this_raw + p_opp_raw
    good = s > 0
    p_fair_this = np.where(good, p_this_raw / s, np.nan)
    truth["Truth_Fair_Prob_at_SharpLine"] = p_fair_this

    # sigma: prefer row sigma; else sport defaults
    if sigma_col not in dm.columns:
        dm[sigma_col] = np.nan
    sig_map = dm[["Game_Key","Market","Outcome_Norm",sigma_col,"Sport"]] \
                .drop_duplicates(["Game_Key","Market","Outcome_Norm"])
    truth = truth.merge(sig_map, on=["Game_Key","Market","Outcome_Norm"], how="left")

    SPORT_SIGMA_DEFAULT = {'NFL':13.0,'NCAAF':14.0,'NBA':12.0,'WNBA':11.0,'MLB':5.5,'CFL':13.5,'NCAAB':11.5}
    def _pick_sigma(row):
        s = pd.to_numeric(row.get(sigma_col), errors="coerce")
        if pd.notna(s) and s > 0: return float(s)
        return float(SPORT_SIGMA_DEFAULT.get(str(row.get("Sport","NFL")).upper(), 12.0))

    s_sharp = pd.to_numeric(truth["Sharp_Line"], errors="coerce")
    p_sharp = pd.to_numeric(truth["Truth_Fair_Prob_at_SharpLine"], errors="coerce")
    sigma   = truth.apply(_pick_sigma, axis=1).astype(float)

    # solve μ where both pieces exist
    mu = np.full(len(truth), np.nan, dtype=float)
    mask_mu = p_sharp.notna() & s_sharp.notna()
    if mask_mu.any():
        mu[mask_mu.values] = (s_sharp[mask_mu] + sigma[mask_mu] * _ppf(p_sharp[mask_mu])).astype(float)

    truth["Truth_Margin_Mu"] = mu
    truth["Truth_Sigma"]     = sigma

    # merge μ,σ back to every book row
    keys = ["Game_Key","Market","Outcome_Norm"]
    dm = dm.merge(truth[keys + ["Truth_Fair_Prob_at_SharpLine","Truth_Margin_Mu","Truth_Sigma"]],
                  on=keys, how="left")

    # price each row at its own line (vectorized, aligned)
    mu_series  = _ensure_series(_col_or_nan(dm, "Truth_Margin_Mu"), dm.index)
    sig_series = _ensure_series(_col_or_nan(dm, "Truth_Sigma"), dm.index)
    line_rec   = _ensure_series(_col_or_nan(dm, "Value"), dm.index)

    ok = mu_series.notna() & sig_series.notna() & (sig_series > 0) & line_rec.notna()
    if ok.any():  # avoid calling any vectorized CDF on size-0 inputs
        z = (mu_series[ok].to_numpy(dtype=float, copy=False)
             - line_rec[ok].to_numpy(dtype=float, copy=False)) \
            / sig_series[ok].to_numpy(dtype=float, copy=False)
        dm.loc[ok, "Truth_Fair_Prob_at_RecLine"] = _phi_local(z)

    # moneyline fallback (no line shift)
    is_ml   = dm["Market"].isin(["h2h","ml","moneyline","headtohead"])
    need_ml = is_ml & dm["Truth_Fair_Prob_at_RecLine"].isna()
    base_sharp = _ensure_series(_col_or_nan(dm, "Truth_Fair_Prob_at_SharpLine"), dm.index)
    dm.loc[need_ml, "Truth_Fair_Prob_at_RecLine"] = base_sharp[need_ml].values

    # implied prob of offered odds (always)
    dm["Rec_Implied_Prob"] = np.asarray(_amer_to_prob(dm.get("Odds_Price", np.nan)), dtype="float64")

    # EV per $1 stake & prob edge
    p_truth = pd.to_numeric(dm["Truth_Fair_Prob_at_RecLine"], errors="coerce")
    odds    = pd.to_numeric(dm.get("Odds_Price", np.nan), errors="coerce")
    payout  = np.where(odds >= 0, odds/100.0, 100.0/(-odds))
    ok_ev   = p_truth.notna() & np.isfinite(payout)
    dm.loc[ok_ev, "EV_Sh_vs_Rec_Dollar"] = (p_truth[ok_ev] * payout[ok_ev]) - (1.0 - p_truth[ok_ev])
    dm.loc[ok_ev, "EV_Sh_vs_Rec_Prob"]   = p_truth[ok_ev] - dm.loc[ok_ev, "Rec_Implied_Prob"]
    dm.loc[ok_ev, "Kelly_Fraction"]      = np.maximum(
        0.0,
        ((p_truth[ok_ev] * payout[ok_ev]) - (1.0 - p_truth[ok_ev])) / payout[ok_ev]
    )

   
        # --- compact dtypes (re‑ensure columns exist first to avoid KeyError) ---
    for c in out_cols:
        if c not in dm.columns:
            dm[c] = np.nan
        dm[c] = pd.to_numeric(dm[c], errors="coerce").astype("float32")

    return dm


def hydrate_inverse_rows_from_snapshot(df_inverse: pd.DataFrame,
                                       df_all_snapshots: pd.DataFrame) -> pd.DataFrame:
    if df_inverse is None or df_inverse.empty or df_all_snapshots is None or df_all_snapshots.empty:
        return df_inverse

    # Work on shallow copies (pandas 2.x CoW friendly)
    df = df_inverse.copy()
    snaps = df_all_snapshots.loc[:, [
        c for c in ['Bookmaker','Book','Home_Team_Norm','Away_Team_Norm',
                    'Game_Start','Market','Outcome','Snapshot_Timestamp',
                    'Value','Odds_Price','Limit']
        if c in df_all_snapshots.columns
    ]].copy()

    # ========= 1) Normalize bookmakers WITHOUT row-wise apply =========
    # Build a unique mapping once, then merge back (drastically fewer function calls).
    cols_for_norm = [c for c in ['Bookmaker','Book'] if c in df.columns]
    cols_for_norm_snaps = [c for c in ['Bookmaker','Book'] if c in snaps.columns]
    # Ensure both frames have both columns for the mapping join
    if 'Book' not in df.columns:    df['Book'] = pd.Series(pd.NA, index=df.index, dtype='string')
    if 'Book' not in snaps.columns: snaps['Book'] = pd.Series(pd.NA, index=snaps.index, dtype='string')

    pairs = (
        pd.concat([
            df[['Bookmaker','Book']].drop_duplicates(),
            snaps[['Bookmaker','Book']].drop_duplicates()
        ], ignore_index=True)
        .drop_duplicates()
    )

    # One call per unique pair (list comprehension is lighter than axis=1 apply)
    pairs['Bookmaker_Norm'] = [
        normalize_book_name(bm, bk)  # re-use your existing function safely
        for bm, bk in zip(pairs['Bookmaker'], pairs['Book'])
    ]

    # Merge mapping back (vectorized)
    df    = df.merge(pairs,   on=['Bookmaker','Book'], how='left')
    snaps = snaps.merge(pairs, on=['Bookmaker','Book'], how='left')
    df['Bookmaker']    = df['Bookmaker_Norm'].fillna(df['Bookmaker'])
    snaps['Bookmaker'] = snaps['Bookmaker_Norm'].fillna(snaps['Bookmaker'])
    df.drop(columns=['Bookmaker_Norm'], inplace=True)
    snaps.drop(columns=['Bookmaker_Norm'], inplace=True)

    # ========= 2) Vectorized key building (no Python '+', no axis=1) =========
    def _s(x: pd.Series) -> pd.Series:
        # string dtype, lower, strip; avoids category/object churn
        if not str(x.dtype).startswith('string'):
            try:
                x = x.astype('string[pyarrow]')
            except Exception:
                x = x.astype('string')
        return x.str.strip().str.lower()

    for c in ['Home_Team_Norm','Away_Team_Norm','Market','Outcome']:
        if c in df.columns:    df[c]    = _s(df[c])
        if c in snaps.columns: snaps[c] = _s(snaps[c])

    # Commence_Hour
    if 'Commence_Hour' in df.columns:
        df['Commence_Hour'] = pd.to_datetime(df['Commence_Hour'], errors='coerce', utc=True).dt.floor('h')
    elif 'Game_Start' in df.columns:
        df['Commence_Hour'] = pd.to_datetime(df['Game_Start'], errors='coerce', utc=True).dt.floor('h')
    else:
        df['Commence_Hour'] = pd.NaT

    snaps['Commence_Hour'] = pd.to_datetime(snaps.get('Game_Start'), errors='coerce', utc=True).dt.floor('h')

    # Team_Key via str.cat (fast & NA-safe)
    def _team_key(frame: pd.DataFrame) -> pd.Series:
        return (
            frame['Home_Team_Norm'].astype('string')
            .str.cat(frame['Away_Team_Norm'].astype('string'), sep='_', na_rep='')
            .str.cat(frame['Commence_Hour'].astype('string'), sep='_', na_rep='')
            .str.cat(frame['Market'].astype('string'), sep='_', na_rep='')
            .str.cat(frame['Outcome'].astype('string'), sep='_', na_rep='')
        )

    df['Team_Key']    = _team_key(df)
    snaps['Team_Key'] = _team_key(snaps)

    # ========= 3) Latest snapshot per (Team_Key, Bookmaker) WITHOUT global sort =========
    snaps['Snapshot_Timestamp'] = pd.to_datetime(snaps['Snapshot_Timestamp'], errors='coerce', utc=True)
    snap_nonnull = snaps.dropna(subset=['Snapshot_Timestamp'])
    if not snap_nonnull.empty:
        idx = snap_nonnull.groupby(['Team_Key','Bookmaker'])['Snapshot_Timestamp'].idxmax()
        df_latest = snap_nonnull.loc[idx, ['Team_Key','Bookmaker','Value','Odds_Price','Limit']].rename(
            columns={'Value': 'Value_opponent', 'Odds_Price': 'Odds_Price_opponent', 'Limit': 'Limit_opponent'}
        )
    else:
        # No timestamps; create empty latest
        df_latest = pd.DataFrame(columns=['Team_Key','Bookmaker','Value_opponent','Odds_Price_opponent','Limit_opponent'])

    # ========= 4) Merge + coalesce to overwrite only when available =========
    df = df.merge(df_latest, on=['Team_Key','Bookmaker'], how='left')
    for col in ['Value','Odds_Price','Limit']:
        opp = f'{col}_opponent'
        if opp in df.columns:
            # prefer snapshot when present
            df[col] = df[opp].combine_first(df[col])
    df.drop(columns=['Value_opponent','Odds_Price_opponent','Limit_opponent'], errors='ignore', inplace=True)

    # Optional: downcast numerics to save memory
    for c in ['Value','Odds_Price','Limit']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32', copy=False)

    return df



def fallback_flip_inverse_rows(df_inverse: pd.DataFrame) -> pd.DataFrame:
    if df_inverse is None or df_inverse.empty:
        return df_inverse

    df = df_inverse  # operate in place (CoW-friendly in pandas 2.x)

    if 'Value' not in df.columns or 'Market' not in df.columns:
        return df

    # normalize just once (no category)
    mkt = df['Market'].astype('string').str.strip().str.lower()
    missing_val = df['Value'].isna()

    # ---- spreads: fill from the opponent value if present, then negate ----
    if 'Value_opponent' in df.columns:
        s_mask = missing_val & mkt.eq('spreads') & df['Value_opponent'].notna()
        if s_mask.any():
            df.loc[s_mask, 'Value'] = -pd.to_numeric(df.loc[s_mask, 'Value_opponent'], errors='coerce')

    # ---- totals: flip labels only where Value is missing (Value may stay NaN) ----
    t_mask = missing_val & mkt.eq('totals')
    if t_mask.any() and 'Outcome_Norm' in df.columns:
        on = df['Outcome_Norm'].astype('string').str.strip().str.lower()
        flipped = on.where(~t_mask, on.map({'over':'under', 'under':'over'}))
        df['Outcome_Norm'] = flipped
        if 'Outcome' in df.columns:
            df.loc[t_mask, 'Outcome'] = flipped

    # optional: tighten dtype
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce').astype('float32', copy=False)
    return df



def add_time_context_flags(df: pd.DataFrame, sport: str, local_tz: str = "America/New_York") -> pd.DataFrame:
    out = df.copy()
    # choose timestamp: prefer Commence_Hour if present, else Game_Start
    ts = None
    if 'Commence_Hour' in out.columns:
        ts = pd.to_datetime(out['Commence_Hour'], errors='coerce', utc=True)
    elif 'Game_Start' in out.columns:
        ts = pd.to_datetime(out['Game_Start'], errors='coerce', utc=True)
    else:
        out['Is_Weekend'] = 0
        out['Is_Night_Game'] = 0
        out['Is_PrimeTime'] = 0
        out['Game_Local_Hour'] = np.nan
        out['Game_DOW'] = np.nan
        return out

    ts_local = ts.dt.tz_convert(local_tz)
    out['Game_Local_Hour'] = ts_local.dt.hour
    out['Game_DOW'] = ts_local.dt.dayofweek  # Mon=0..Sun=6
    out['Is_Weekend'] = out['Game_DOW'].isin([5, 6]).astype(int)

    SPORT_NIGHT_CUTOFF = {'MLB': 18, 'NFL': 18, 'CFL': 18, 'NBA': 18, 'WNBA': 18, 'NCAAF': 18, 'NCAAB': 18}
    cutoff = SPORT_NIGHT_CUTOFF.get(str(sport).upper(), 18)
    out['Is_Night_Game'] = (out['Game_Local_Hour'] >= cutoff).astype(int)

    if str(sport).upper() in {'NFL', 'CFL'}:
        out['Is_PrimeTime'] = ((out['Game_DOW'].isin([3, 6, 0])) & out['Game_Local_Hour'].between(19, 23)).astype(int)
    else:
        out['Is_PrimeTime'] = 0
    return out






# Tiny, fast normalizer (no extra temporaries)
def _norm_team_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace(r'\s+', ' ', regex=True)\
         .str.replace('.', '', regex=False)\
         .str.replace('&', 'and', regex=False)\
         .str.replace('-', ' ', regex=False)
    return s

@lru_cache(maxsize=16)
def fetch_latest_current_ratings_cached(
    sport: str,
    table_current: str = "sharplogger.sharp_data.ratings_current",
    project: str = "sharplogger",
) -> pd.DataFrame:
    """
    Returns minimal, float32-dtyped ratings for one sport:
      columns: ['Team_Norm','Power_Rating']  (float32)
    NOTE: This is memoized per (sport, table_current, project).
    """
    bq = bigquery.Client(project=project)
    sql = f"""
      SELECT
        LOWER(TRIM(Team)) AS Team_Norm,
        CAST(Rating AS FLOAT64) AS Power_Rating
      FROM `{table_current}`
      WHERE UPPER(Sport) = @sport
    """
    df = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            use_query_cache=True,
            query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport.upper())],
        ),
    ).to_dataframe()  # already streamed; small result set

    if df.empty:
        # Return a consistent empty frame with right dtypes
        return pd.DataFrame({"Team_Norm": pd.Series([], dtype="object"),
                             "Power_Rating": pd.Series([], dtype="float32")})

    # Normalize and downcast in-place
    df["Team_Norm"] = _norm_team_series(df["Team_Norm"])
    df["Power_Rating"] = pd.to_numeric(df["Power_Rating"], errors="coerce").astype("float32")
    df = df.dropna(subset=["Team_Norm", "Power_Rating"])

    # Keep only what we need, with tight dtypes
    return df[["Team_Norm", "Power_Rating"]]

SPORT_SPREAD_CFG = {
    "NFL":   dict(scale=np.float32(1.0),  HFA=np.float32(2.1),  sigma_pts=np.float32(13.0)),
    "NCAAF": dict(scale=np.float32(1.0),  HFA=np.float32(2.6),  sigma_pts=np.float32(14.0)),
    "NBA":   dict(scale=np.float32(1.0),  HFA=np.float32(2.8),  sigma_pts=np.float32(12.0)),
    "WNBA":  dict(scale=np.float32(1.0),  HFA=np.float32(2.0),  sigma_pts=np.float32(11.5)),
    "CFL":   dict(scale=np.float32(1.0),  HFA=np.float32(1.6),  sigma_pts=np.float32(13.5)),
    # MLB ratings are 1500 + 400*(atk+dfn) → not run units; scale≈89–90 maps diff → runs
    "MLB":   dict(scale=np.float32(89.0), HFA=np.float32(0.20), sigma_pts=np.float32(3.1)),
    "NCAAB":   dict(scale=np.float32(1.0),  HFA=np.float32(2.8),  sigma_pts=np.float32(12.0)),
}



def enrich_power_from_current_inplace(
    df: pd.DataFrame,
    sport_aliases: dict,
    table_current: str = "sharplogger.sharp_data.ratings_current",
    project: str = "sharplogger",
    baseline: float = 1500.0,
) -> None:
    """
    In-place version: adds/overwrites Home_Power_Rating, Away_Power_Rating, Power_Rating_Diff.
    Requires ['Sport','Home_Team_Norm','Away_Team_Norm'] normalized enough for mapping.
    """
   

    if df.empty:
        # still ensure columns exist
        df["Home_Power_Rating"] = np.float32(baseline)
        df["Away_Power_Rating"] = np.float32(baseline)
        df["Power_Rating_Diff"] = np.float32(0.0)
        return

    # Canon sport (assume single sport batch is typical)
    sport_canon = str(df["Sport"].iloc[0]).upper()

    # Fetch minimal ratings (memoized)
    ratings = fetch_latest_current_ratings_cached(
        sport=sport_canon,
        table_current=table_current,
        project=project,
    )

    # Default fills
    base32 = np.float32(baseline)

    if ratings.empty:
        df["Home_Power_Rating"] = base32
        df["Away_Power_Rating"] = base32
        df["Power_Rating_Diff"] = np.float32(0.0)
        return

    # Build mapping Series (no dict, no copies)
    s_map = pd.Series(
        ratings["Power_Rating"].to_numpy(copy=False),
        index=ratings["Team_Norm"],
        dtype="float32",
    )

    # Ensure team keys are normalized here (cheap and avoids another copy upstream)
    def _norm(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.lower().str.strip()
        s = s.str.replace(r"\s+", " ", regex=True)\
             .str.replace(".", "", regex=False)\
             .str.replace("&", "and", regex=False)\
             .str.replace("-", " ", regex=False)
        return s

    if "Home_Team_Norm" in df.columns:
        home_keys = _norm(df["Home_Team_Norm"])
    else:
        raise KeyError("Home_Team_Norm missing")

    if "Away_Team_Norm" in df.columns:
        away_keys = _norm(df["Away_Team_Norm"])
    else:
        raise KeyError("Away_Team_Norm missing")

    # Vectorized lookups -> float32 arrays
    home = s_map.reindex(home_keys).to_numpy(dtype="float32", copy=False)
    away = s_map.reindex(away_keys).to_numpy(dtype="float32", copy=False)

    # Single sport mean as fallback
    sport_mean = np.float32(s_map.mean(skipna=True))

    # Fill NaNs in place
    np.nan_to_num(home, copy=False, nan=sport_mean)
    np.nan_to_num(away, copy=False, nan=sport_mean)

    # Attach columns without creating intermediates
    df["Home_Power_Rating"] = home
    df["Away_Power_Rating"] = away
    df["Power_Rating_Diff"] = (home - away).astype("float32", copy=False)

    return df

# ---- math utils (Normal CDF; no scipy) ----
def _phi(z: np.ndarray) -> np.ndarray:
    # standard normal CDF
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

def _get_cfg_vec(sport_series: pd.Series):
    s = sport_series.astype(str).str.upper().values
    n = len(s)
    scale = np.full(n, np.float32(1.0), dtype=np.float32)
    hfa   = np.zeros(n, dtype=np.float32)
    sig   = np.full(n, np.float32(12.0), dtype=np.float32)

    for sport, cfg in SPORT_SPREAD_CFG.items():
        m = (s == sport)
        if not m.any():
            continue
        # backward-compat: accept points_per_elo if still present
        sc = cfg.get("scale", cfg.get("points_per_elo", 1.0))
        scale[m] = np.float32(sc)
        hfa[m]   = np.float32(cfg["HFA"])
        sig[m]   = np.float32(cfg["sigma_pts"])

    return scale.astype("float32"), hfa.astype("float32"), sig.astype("float32")

# ---- 1) Build consensus market spread per game (no BQ) ----

def favorite_centric_from_powerdiff_lowmem(g_full: pd.DataFrame) -> pd.DataFrame:
    out = g_full.copy()
    out["Sport"] = out["Sport"].astype(str).str.upper().str.strip()
    for c in ["Home_Team_Norm","Away_Team_Norm"]:
        out[c] = out[c].astype(str).str.lower().str.strip()

    pr_diff = pd.to_numeric(out.get("Power_Rating_Diff"), errors="coerce").fillna(0).astype("float32")

    # ⬇⬇ changed: pull `scale` (not ppe), and compute mu = pr_diff/scale + HFA
    scale, hfa, sig = _get_cfg_vec(out["Sport"])
    mu = (pr_diff / scale) + hfa
    mu = mu.astype("float32")

    out["Model_Expected_Margin"] = mu
    out["Model_Expected_Margin_Abs"] = np.abs(mu, dtype=np.float32)
    out["Sigma_Pts"] = sig

    fav_is_home = (mu >= 0)
    out["Model_Favorite_Team"] = np.where(fav_is_home, out["Home_Team_Norm"], out["Away_Team_Norm"])
    out["Model_Underdog_Team"] = np.where(fav_is_home, out["Away_Team_Norm"], out["Home_Team_Norm"])

    out["Model_Fav_Spread"] = (-np.abs(mu)).astype("float32")
    out["Model_Dog_Spread"] = (+np.abs(mu)).astype("float32")

    fav_mkt = pd.to_numeric(out.get("Favorite_Market_Spread"), errors="coerce").astype("float32")
    dog_mkt = pd.to_numeric(out.get("Underdog_Market_Spread"), errors="coerce").astype("float32")
    k = np.where(np.isnan(fav_mkt), np.abs(dog_mkt), np.abs(fav_mkt)).astype("float32")

    out["Fav_Edge_Pts"] = (fav_mkt - out["Model_Fav_Spread"]).astype("float32")
    out["Dog_Edge_Pts"] = (dog_mkt - out["Model_Dog_Spread"]).astype("float32")

    eps = np.finfo("float32").eps
    z_fav = (np.abs(mu) - k) / np.where(sig == 0, eps, sig)
    out["Fav_Cover_Prob"] = (1.0 - _phi(-z_fav)).astype("float32")  # Φ(z_fav)
    out["Dog_Cover_Prob"] = (1.0 - out["Fav_Cover_Prob"]).astype("float32")
    return out

def prep_consensus_market_spread_lowmem(
    df_spread_rows: pd.DataFrame,
    value_col: str = "Value",
    outcome_col: str = "Outcome_Norm",
) -> pd.DataFrame:
    """
    Input: multiple rows per game/outcome/book with spread in `value_col`.
    Output: one row per game with:
      Market_Favorite_Team, Market_Underdog_Team,
      Favorite_Market_Spread, Underdog_Market_Spread
    Consensus = median spread per outcome; favorite = outcome with more negative median.
    """
    need = ["Sport","Home_Team_Norm","Away_Team_Norm", outcome_col, value_col]
    g = df_spread_rows[need].dropna(subset=[outcome_col]).copy()

    # median spread per game per outcome (robust to outliers/books)
    med = (g.groupby(["Sport","Home_Team_Norm","Away_Team_Norm", outcome_col], dropna=False)[value_col]
             .median()
             .reset_index()
             .rename(columns={value_col: "Outcome_Median_Spread"}))

    # choose favorite = outcome with smallest (most negative) median
    idx_min = (med.groupby(["Sport","Home_Team_Norm","Away_Team_Norm"], dropna=False)["Outcome_Median_Spread"]
                 .idxmin())
    fav_rows = med.loc[idx_min].copy()
    fav_rows = fav_rows.rename(columns={
        outcome_col: "Market_Favorite_Team",
        "Outcome_Median_Spread": "Favorite_Market_Spread"
    })

    # determine underdog = the other team in the game
    # build a two-row frame per game to grab the other team's median
    merged = (fav_rows.merge(
        med,
        on=["Sport","Home_Team_Norm","Away_Team_Norm"],
        suffixes=("","_all"),
        how="left"
    ))
    # pick the "other" outcome row
    mask_other = (merged[outcome_col] != merged["Market_Favorite_Team"])
    other = (merged[mask_other]
             .rename(columns={
                 outcome_col: "Market_Underdog_Team",
                 "Outcome_Median_Spread": "Underdog_Market_Spread"
             })[[
                 "Sport","Home_Team_Norm","Away_Team_Norm",
                 "Market_Underdog_Team","Underdog_Market_Spread"
             ]])

    out = fav_rows.merge(other, on=["Sport","Home_Team_Norm","Away_Team_Norm"], how="left")

    # If signs are inconsistent, fix underdog to be opposite sign of favorite
    # (typical books list dog spread as +L if favorite is -L)
    same_sign = np.sign(out["Favorite_Market_Spread"].fillna(0)) == np.sign(out["Underdog_Market_Spread"].fillna(0))
    out.loc[same_sign, "Underdog_Market_Spread"] = out.loc[same_sign, "Favorite_Market_Spread"] * -1.0

    return out

# ---- 3) The wrapper that ties it all together (minimal memory/BQ) ----
def enrich_and_grade_for_training(
    df_spread_rows: pd.DataFrame,
    sport_aliases: dict,
    value_col: str = "Value",
    outcome_col: str = "Outcome_Norm",
    table_current: str = "sharplogger.sharp_data.ratings_current",
    project: str = "sharplogger",
) -> pd.DataFrame:
    """
    No history pulls. Uses your `enrich_power_from_current_inplace` once on a
    deduped game list, then builds consensus market spreads and grading.
    """
    if df_spread_rows is None or df_spread_rows.empty:
        logging.info("[enrich_and_grade_for_training] empty input")
        return df_spread_rows

    game_key = ["Sport","Home_Team_Norm","Away_Team_Norm"]
    base_cols = game_key + ["Game_Start"]
    df_games = (
        df_spread_rows[base_cols]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    # 3.1) Power enrich (IN-PLACE, single pass, minimal memory)
    enrich_power_from_current_inplace(
        df_games,
        sport_aliases=sport_aliases,
        table_current=table_current,
        project=project,
    )

    # 3.2) Consensus market favorite & spreads (no queries)
    g_cons = prep_consensus_market_spread_lowmem(
        df_spread_rows, value_col=value_col, outcome_col=outcome_col
    )

    # 3.3) Join & compute model grading from power diff
    g_full = df_games.merge(g_cons, on=game_key, how="left")
    g_fc   = favorite_centric_from_powerdiff_lowmem(g_full)

    # ensure required columns exist (avoid KeyErrors downstream)
    needed = game_key + [
        'Market_Favorite_Team','Market_Underdog_Team',
        'Favorite_Market_Spread','Underdog_Market_Spread',
        'Model_Favorite_Team','Model_Underdog_Team',
        'Model_Fav_Spread','Model_Dog_Spread',
        'Fav_Edge_Pts','Dog_Edge_Pts',
        'Fav_Cover_Prob','Dog_Cover_Prob',
        'Model_Expected_Margin','Model_Expected_Margin_Abs','Sigma_Pts',
        'Home_Power_Rating','Away_Power_Rating','Power_Rating_Diff'
    ]
    for c in needed:
        if c not in g_fc.columns:
            g_fc[c] = np.nan
    g_fc = g_fc[needed].copy()

    out = df_spread_rows.merge(g_fc, on=game_key, how="left")

    # 3.4) Per-outcome vectorized mapping
    is_fav = (out[outcome_col].astype(str).values ==
              out['Market_Favorite_Team'].astype(str).values)
    out['Outcome_Model_Spread']  = np.where(is_fav, out['Model_Fav_Spread'].values,      out['Model_Dog_Spread'].values).astype('float32')
    out['Outcome_Market_Spread'] = np.where(is_fav, out['Favorite_Market_Spread'].values, out['Underdog_Market_Spread'].values).astype('float32')
    out['Outcome_Spread_Edge']   = np.where(is_fav, out['Fav_Edge_Pts'].values,           out['Dog_Edge_Pts'].values).astype('float32')
    out['Outcome_Cover_Prob']    = np.where(is_fav, out['Fav_Cover_Prob'].values,         out['Dog_Cover_Prob'].values).astype('float32')

    return out


def implied_prob_vec(odds):
    # expects numpy/pandas array, returns float32 array
    x = pd.to_numeric(odds, errors='coerce').to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(x >= 0, 100.0 / (x + 100.0), (-x) / (-x + 100.0))
    return out.astype('float32', copy=False)
# --- Memory helpers (near imports) ---


_PROC = psutil.Process(os.getpid())

def _fmt_mb(x: int | float) -> str:
    try:
        return f"{(float(x) / (1024**2)):.1f} MB"
    except Exception:
        return f"{x}"

def _rss_bytes() -> int:
    try:
        return _PROC.memory_info().rss
    except Exception:
        return 0

class MemSampler:
    """
    Lightweight periodic memory sampler:
      - logs process RSS,
      - lists top local pandas/numpy objects by size,
      - can auto-gc when a threshold is exceeded.
    """
    def __init__(self, logger, interval_s: float = 5.0, topn: int = 8,
                 deep_columns: bool = False, gc_threshold_mb: float | None = None):
        self.logger = logger
        self.interval_s = interval_s
        self.topn = topn
        self.deep_columns = deep_columns
        self.gc_threshold = (gc_threshold_mb * 1024**2) if gc_threshold_mb else None
        self.start_rss = _rss_bytes()
        self.last_rss = self.start_rss
        self.last_t = time.monotonic()

    def _should_sample(self) -> bool:
        return (time.monotonic() - self.last_t) >= self.interval_s

    def maybe(self, tag: str, frame_locals: dict | None = None):
        """Call at checkpoints; cheap if interval hasn’t elapsed."""
        if not self._should_sample():
            return
        rss = _rss_bytes()
        delta = rss - self.last_rss
        since_start = rss - self.start_rss
        self.logger.info(
            "🧠 MEM[%s] RSS=%s (Δ=%s since last, ΣΔ=%s since start)",
            tag, _fmt_mb(rss), _fmt_mb(delta), _fmt_mb(since_start)
        )
        self.last_rss = rss
        self.last_t = time.monotonic()

        # Top local objects (only pandas/numpy to keep it fast)
        if frame_locals:
            tops = []
            for name, obj in frame_locals.items():
                try:
                    if isinstance(obj, pd.DataFrame):
                        sz = obj.memory_usage(deep=False).sum()
                    elif isinstance(obj, pd.Series):
                        sz = obj.memory_usage(deep=False)
                    elif isinstance(obj, np.ndarray):
                        sz = obj.nbytes
                    else:
                        continue
                    tops.append((int(sz), name, type(obj).__name__))
                except Exception:
                    continue
            if tops:
                tops.sort(reverse=True, key=lambda x: x[0])
                tops = tops[: self.topn]
                desc = ", ".join(f"{n}:{t}={_fmt_mb(sz)}" for sz, n, t in tops)
                self.logger.info("🔎 Top locals (pandas/numpy): %s", desc)

        # Optional emergency GC if we’re climbing
        if self.gc_threshold and rss >= self.gc_threshold:
            freed = gc.collect()
            self.logger.warning("🧹 GC triggered at %s (freed objects=%s)", _fmt_mb(rss), freed)

def log_df_top_columns(df: pd.DataFrame, logger, tag: str, topn: int = 12, deep: bool = False):
    """One-off column-level breakdown for the main df."""
    try:
        mu = df.memory_usage(deep=deep).sort_values(ascending=False)
        head = mu.head(topn)
        total = mu.sum()
        logger.info(
            "📦 DF columns by memory [%s] (top %d, deep=%s): %s | total=%s",
            tag, topn, deep,
            ", ".join(f"{c}={_fmt_mb(int(b))}" for c, b in head.items()),
            _fmt_mb(int(total)),
        )
    except Exception as e:
        logger.warning("⚠️ Column memory breakdown failed [%s]: %s", tag, e)

def _suffix_snapshot(df, tag):
    bad = [c for c in df.columns if c.endswith(('_x','_y')) and
           (c.startswith('Open_') or c.startswith('First_Imp_Prob'))]
    if bad:
        logger.warning("❗ %s introduced %s", tag, bad)

def predict_blended(bundle, X, model=None, iso=None, eps=1e-6):
    """
    Return calibrated/blended probs in [eps, 1-eps] or None.
    Applies bundle["flip_flag"] (global polarity lock) BEFORE calibration,
    so the returned value is always "P(win)" for the positive class.
    """

    def _predict_one(m, cal, X_):
        # Already-calibrated classifier (e.g., CalibratedClassifierCV)
        if m is not None and hasattr(m, "predict_proba") and cal is None and hasattr(m, "base_estimator_"):
            p = m.predict_proba(X_)
            p1 = p[:, 1] if isinstance(p, np.ndarray) and p.ndim > 1 else np.asarray(p, dtype=float)
            return np.asarray(p1, dtype=float).ravel()

        # Raw score/proba
        p_raw = None
        if m is not None:
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X_)
                p_raw = p[:, 1] if isinstance(p, np.ndarray) and p.ndim > 1 else np.asarray(p, dtype=float)
            elif hasattr(m, "decision_function"):
                s = np.asarray(m.decision_function(X_), dtype=float)
                # If we’ll calibrate later, pass score; otherwise squash
                p_raw = s if cal is not None else 1.0 / (1.0 + np.exp(-s))

        # Apply per-model calibrator (OLD path)
        if cal is not None and p_raw is not None:
            if hasattr(cal, "predict"):
                p_cal = cal.predict(p_raw)
            elif hasattr(cal, "transform"):
                p_cal = cal.transform(p_raw)
            elif callable(cal):
                p_cal = cal(p_raw)
            else:
                p_cal = p_raw
            return np.asarray(p_cal, dtype=float).ravel()

        if p_raw is not None:
            return np.asarray(p_raw, dtype=float).ravel()

        # Rare: calibrator acts as classifier
        if cal is not None and hasattr(cal, "predict_proba"):
            p = cal.predict_proba(X_)
            return (p[:, 1] if isinstance(p, np.ndarray) and p.ndim > 1 else np.asarray(p, dtype=float)).ravel()
        if cal is not None and hasattr(cal, "predict"):
            return np.asarray(cal.predict(X_), dtype=float).ravel()

        return None

    def _apply_flip_if_needed(p_vec):
        """Apply global flip_flag to a probability vector."""
        if p_vec is None:
            return None
        p_vec = np.asarray(p_vec, dtype=float).ravel()
        if bool(flip_flag):
            p_vec = 1.0 - p_vec
        return p_vec

    # --- determine global flip flag (supports a couple legacy names) ---
    flip_flag = False
    if isinstance(bundle, dict):
        flip_flag = bool(
            bundle.get("flip_flag", False)
            or bundle.get("global_flip", False)
            or (bundle.get("polarity", +1) == -1)
        )

    # Tuple/list bundle
    if isinstance(bundle, (tuple, list)) and len(bundle) >= 1 and model is None and iso is None:
        model = bundle[0]
        iso   = bundle[1] if len(bundle) > 1 else None

    # Dict bundle
    if isinstance(bundle, dict):
        # NEW: one isotonic over blended prob
        calib_bundle = bundle.get("calibrator")
        if isinstance(calib_bundle, dict) and ("iso_blend" in calib_bundle):
            mL = bundle.get("model_logloss")
            mA = bundle.get("model_auc")
            w  = float(bundle.get("best_w", 0.5))

            pL = _predict_one(mL, None, X) if mL is not None else None
            pA = _predict_one(mA, None, X) if mA is not None else None

            if pL is None and pA is None:
                m_single = bundle.get("model")
                if m_single is None:
                    return None
                p_raw = _predict_one(m_single, None, X)
            else:
                if pL is not None and pA is not None and len(pL) == len(pA):
                    p_raw = (w * pL + (1.0 - w) * pA)
                else:
                    p_raw = pL if pL is not None else pA

            if p_raw is None:
                return None

            # clip → flip (BEFORE iso) → iso
            p_raw = np.clip(np.asarray(p_raw, dtype=float).ravel(), eps, 1 - eps)
            p_raw = _apply_flip_if_needed(p_raw)  # ✅ GLOBAL POLARITY LOCK (P(win))

            iso = calib_bundle["iso_blend"]
            if hasattr(iso, "predict"):
                p_cal = iso.predict(p_raw)
            elif hasattr(iso, "transform"):
                p_cal = iso.transform(p_raw)
            else:
                p_cal = p_raw

            p_cal = np.asarray(p_cal, dtype=float).ravel()
            p_cal = np.nan_to_num(p_cal, nan=0.5, posinf=1 - eps, neginf=eps)
            return np.clip(p_cal, eps, 1 - eps)

        # OLD: per-head calibration, then blend
        if ("model_logloss" in bundle) or ("model_auc" in bundle):
            mL = bundle.get("model_logloss"); cL = bundle.get("calibrator_logloss")
            mA = bundle.get("model_auc");     cA = bundle.get("calibrator_auc")
            w  = float(bundle.get("best_w", 1.0))

            pL = _predict_one(mL, cL, X) if (mL is not None or cL is not None) else None
            pA = _predict_one(mA, cA, X) if (mA is not None or cA is not None) else None

            if pL is not None and pA is not None and len(pL) == len(pA):
                p = (w * np.asarray(pL, float) + (1.0 - w) * np.asarray(pA, float)).ravel()
                p = np.clip(p, eps, 1 - eps)
                p = _apply_flip_if_needed(p)  # ✅ flip after blend (already calibrated heads)
                return np.clip(p, eps, 1 - eps)

            if pL is not None:
                p = np.clip(np.asarray(pL, dtype=float).ravel(), eps, 1 - eps)
                p = _apply_flip_if_needed(p)
                return np.clip(p, eps, 1 - eps)

            if pA is not None:
                p = np.clip(np.asarray(pA, dtype=float).ravel(), eps, 1 - eps)
                p = _apply_flip_if_needed(p)
                return np.clip(p, eps, 1 - eps)

        # Single pair: {"model": est, "calibrator": cal or {"iso_blend": cal}}
        if ("model" in bundle) or ("calibrator" in bundle):
            m = bundle.get("model")
            c = bundle.get("calibrator")

            # single model + iso_blend dict
            if isinstance(c, dict) and "iso_blend" in c:
                p_raw = _predict_one(m, None, X)
                if p_raw is None:
                    return None
                p_raw = np.clip(np.asarray(p_raw, dtype=float).ravel(), eps, 1 - eps)
                p_raw = _apply_flip_if_needed(p_raw)  # ✅ flip BEFORE iso

                iso = c["iso_blend"]
                if hasattr(iso, "predict"):
                    p_cal = iso.predict(p_raw)
                elif hasattr(iso, "transform"):
                    p_cal = iso.transform(p_raw)
                else:
                    p_cal = p_raw

                p_cal = np.asarray(p_cal, dtype=float).ravel()
                p_cal = np.nan_to_num(p_cal, nan=0.5, posinf=1 - eps, neginf=eps)
                return np.clip(p_cal, eps, 1 - eps)

            # generic calibrator
            p = _predict_one(m, c, X)
            if p is None:
                return None
            p = np.clip(np.asarray(p, dtype=float).ravel(), eps, 1 - eps)
            p = _apply_flip_if_needed(p)  # ✅ flip after per-model calibration
            return np.clip(p, eps, 1 - eps)

    # Legacy args path
    p = _predict_one(model, iso, X)
    if p is None:
        return None
    p = np.clip(np.asarray(p, dtype=float).ravel(), eps, 1 - eps)
    # Legacy path has no bundle dict, so no flip applied here
    return np.clip(p, eps, 1 - eps)


def _unwrap_pipeline(est):
    pipe = est if hasattr(est, "named_steps") else None
    last = est
    pre  = None
    if pipe is not None:
        try:
            steps = list(pipe.named_steps.values())
            if steps:
                last = steps[-1]
                for s in steps[:-1]:
                    if hasattr(s, "transform") and hasattr(s, "fit"):
                        pre = s
                if pre is None and hasattr(last, "transform") and hasattr(last, "fit"):
                    pre = last
        except Exception:
            pass
    return pipe, last, pre

def _training_like_feature_selector(df):
    include_prefixes = (
        "line_", "sharp_line_", "rec_line_", "line_move_", "line_magnitude_", "net_line_",
        "odds_", "implied_prob_", "net_odds_", "abs_odds_", "odds_move_", "implied_prob_shift",
        "minutes_", "late_game_", "sharp_timing", "sharpmove_", "oddsmove_", "sharpmove_timing_",
        "spread_", "total_", "h2h_", "spread_vs_", "total_vs_", "crossmarket_",
        "book_reliability_", "book_lift_", "sharp_limit_", "limitprotect_", "high_limit_flag",
        "team_", "market_mispricing", "abs_market_mispricing", "abs_team_implied_prob_gap",
        "small_book_", "liquidity_", "market_leader", "unique_sharp_books", "crossmarketsharpsupport",
        "value_reversal_flag", "odds_reversal_flag",
    )
    exclude_prefixes = (
        "game_", "home_", "away_", "team_key", "game_key",
        "bookmaker", "book", "market_", "outcome",
        "snapshot_", "commence_", "event_", "sport", "time",
        "minutes_to_game_tier",
    )
    num_kinds = set("biuf")
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        kind = getattr(df[c].dtype, "kind", "O")
        if kind not in num_kinds:       # numeric only
            continue
        if any(cl.startswith(p) for p in exclude_prefixes):
            continue
        if any(cl.startswith(p) for p in include_prefixes):
            cols.append(c)
    if not cols:
        cols = [c for c in df.columns
                if getattr(df[c].dtype, "kind", "O") in num_kinds
                and not str(c).lower().startswith(exclude_prefixes)]
    return [str(c) for c in cols]

def _resolve_feature_cols_like_training(bundle, model, df_canon, prefer_saved_only: bool = True):
    """
    Return the exact training feature list if present in the bundle.
    If not present and prefer_saved_only=True, return [] (so we stamp unscored
    instead of guessing). Set prefer_saved_only=False to allow legacy inference.
    """
    # 0) pull from bundle (now persisted in GCS)
    if isinstance(bundle, dict):
        # primary location
        fc = bundle.get("feature_cols")
        if isinstance(fc, (list, tuple)) and fc:
            return [str(c) for c in fc]
        # optional metadata location
        fc = bundle.get("metadata", {}).get("feature_cols") if isinstance(bundle.get("metadata"), dict) else None
        if isinstance(fc, (list, tuple)) and fc:
            return [str(c) for c in fc]

    if prefer_saved_only:
        # Don’t guess; makes apply stable & prevents silent drift
        return []

    # ---- legacy discovery (fallbacks) ----
    _, last, pre = _unwrap_pipeline(model)
    names = getattr(last, "feature_names_in_", None)
    if names is not None and len(names):
        return [str(c) for c in names]

    if hasattr(last, "get_booster"):
        try:
            booster = last.get_booster()
            bnames = getattr(booster, "feature_names", None)
            if bnames:
                return [str(c) for c in bnames]
        except Exception:
            pass

    for attr in ("feature_name_", "feature_names_"):
        names = getattr(last, attr, None)
        if names:
            return [str(c) for c in names]

    if hasattr(last, "booster_"):
        try:
            names = last.booster_.feature_name()
            if names:
                return [str(c) for c in names]
        except Exception:
            pass

    get_names = getattr(last, "get_feature_names", None)
    if callable(get_names):
        try:
            n = get_names()
            if n:
                return [str(c) for c in n]
        except Exception:
            pass

    if pre is not None:
        raw = getattr(pre, "feature_names_in_", None)
        if raw is not None and len(raw):
            return [str(c) for c in raw]

    # final fallback (training-like selector)
    return _training_like_feature_selector(df_canon)


# Only the columns needed for totals features
_TOT_SCORE_COLS = (
    "Sport, Game_Start, Home_Team, Away_Team, "
    "SAFE_CAST(Score_Home_Score AS FLOAT64) AS Score_Home_Score, "
    "SAFE_CAST(Score_Away_Score AS FLOAT64) AS Score_Away_Score, "
    "Merge_Key_Short"
)

def _downcast_scores_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["Game_Start"] = pd.to_datetime(df["Game_Start"], errors="coerce", utc=True)
    for c in ("Score_Home_Score", "Score_Away_Score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in ("Sport", "Home_Team", "Away_Team"):
        if c in df.columns:
            df[c] = df[c].astype("string").astype("category")
    if "Merge_Key_Short" in df.columns:
        df["Merge_Key_Short"] = df["Merge_Key_Short"].astype("string")
    return df

def _bq_pull_scores_noseason(
    table_fq: str,
    *,
    sport: str | None,
    days_back: int | None  # None => no time filter
) -> pd.DataFrame:
    from google.cloud import bigquery
    bq = bigquery.Client()

    filters = []
    params = []

    if days_back is not None:
        filters.append("Game_Start >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days_back DAY)")
        params.append(bigquery.ScalarQueryParameter("days_back", "INT64", int(days_back)))

    if sport:
        filters.append("UPPER(Sport) = UPPER(@sport)")
        params.append(bigquery.ScalarQueryParameter("sport", "STRING", sport))

    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    query = f"""
      SELECT {_TOT_SCORE_COLS}
      FROM `{table_fq}`
      {where_clause}
    """

    job_config = bigquery.QueryJobConfig(query_parameters=params or None)
    df = bq.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=True)
    return _downcast_scores_df(df)

def load_scores_history_cached_backend(
    *,
    sport: str | None = None,
    days_back: int | None = 365,   # ✅ no seasons; set None for all-time if you want
    table_fq: str = "sharplogger.sharp_data.game_scores_final",
    ttl_seconds: int = 3600,
    cache_dir: str = "/tmp"
) -> pd.DataFrame:
    """
    Low-memory, season-free loader for Cloud Run/Functions.
    Caches a Parquet in /tmp keyed by (sport, days_back, table).
    """
    os.makedirs(cache_dir, exist_ok=True)
    sport_key = (sport or "ALL").upper()
    range_key = f"{int(days_back)}d" if days_back is not None else "ALLTIME"
    safe_table = table_fq.replace(".", "__")
    cache_path = os.path.join(cache_dir, f"scores_hist__{safe_table}__{sport_key}__{range_key}.parquet")

    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < ttl_seconds:
            return pd.read_parquet(cache_path)

    df = _bq_pull_scores_noseason(table_fq, sport=sport, days_back=days_back)
    df.to_parquet(cache_path, index=False)
    return df



# =========================
# 1) Build totals baselines (NO Season)
# =========================
def build_totals_training_from_scores(df_scores: pd.DataFrame,
                                      sport: str | None = None,
                                      window_games: int = 30,
                                      shrink: float = 0.30,
                                      key_col: str = "Merge_Key_Short"
                                      ) -> pd.DataFrame:
    """
    Returns one row per historical game with:
      {key_col}, TOT_Proj_Total_Baseline, TOT_Off_H, TOT_Def_H, TOT_Off_A, TOT_Def_A,
      TOT_GT_H, TOT_GT_A, TOT_LgAvg_Total, TOT_Actual_Total

    Leakage-safe via .shift(1). No Season logic; strictly ordered by Game_Start.
    """
    if key_col not in df_scores.columns:
        raise ValueError(f"Expected key_col '{key_col}' in df_scores")

    df = df_scores.copy()
    # normalize join key
    df[key_col] = df[key_col].astype(str).str.strip().str.lower()
    df = df.rename(columns={key_col: "Game_Key"})  # internal

    # filter sport if provided (optional)
    if sport:
        df = df[df["Sport"].astype(str).str.upper() == sport.upper()].copy()

    # ensure time & numeric types
    df["Game_Start"] = pd.to_datetime(df["Game_Start"], errors="coerce", utc=True)
    df["Score_Home_Score"] = pd.to_numeric(df["Score_Home_Score"], errors="coerce")
    df["Score_Away_Score"] = pd.to_numeric(df["Score_Away_Score"], errors="coerce")

    # ---- per-game totals (to compute league avg prior across ALL games chronologically) ---
    g_base = (
        df[["Sport","Game_Key","Game_Start","Home_Team","Away_Team",
            "Score_Home_Score","Score_Away_Score"]]
        .drop_duplicates(subset=["Game_Key"])
        .sort_values("Game_Start")
        .reset_index(drop=True)
    )
    g_base["Actual_Total"] = g_base["Score_Home_Score"] + g_base["Score_Away_Score"]

    # league avg total prior to each game (single value per Game_Key), no seasons
    league_avg_prior = (
        g_base["Actual_Total"]
        .expanding().mean().shift(1)                # prior games only
        .fillna(g_base["Actual_Total"].mean())      # early backfill
    )
    g_base["LgAvg_Total_Prior"] = league_avg_prior

    # ---- long frame (team rows) with rolling stats by team (no seasons) ----
    home = df.assign(team=df["Home_Team"], opp=df["Away_Team"],
                     pts_for=df["Score_Home_Score"], pts_against=df["Score_Away_Score"])
    away = df.assign(team=df["Away_Team"], opp=df["Home_Team"],
                     pts_for=df["Score_Away_Score"], pts_against=df["Score_Home_Score"])
    long = pd.concat([home, away], ignore_index=True)

    long = (
        long[["Sport","Game_Key","Game_Start","team","opp","pts_for","pts_against"]]
        .sort_values(["team","Game_Start"])
        .reset_index(drop=True)
    )
    long["game_total"] = long["pts_for"] + long["pts_against"]

    def _roll_mean_prior(x: pd.Series) -> pd.Series:
        return x.rolling(window_games, min_periods=1).mean().shift(1)  # exclude current

    long["pf_roll"] = long.groupby("team", group_keys=False)["pts_for"].apply(_roll_mean_prior)
    long["pa_roll"] = long.groupby("team", group_keys=False)["pts_against"].apply(_roll_mean_prior)
    long["gt_roll"] = long.groupby("team", group_keys=False)["game_total"].apply(_roll_mean_prior)

    # map league avg prior (per game) onto both rows of that game
    long = long.merge(
        g_base[["Game_Key","LgAvg_Total_Prior"]],
        on="Game_Key", how="left"
    )

    # ratings relative to half the league avg prior (off/def above/below avg)
    half = long["LgAvg_Total_Prior"] / 2.0
    long["Off_Rating"] = (1 - shrink) * (long["pf_roll"] - half)
    long["Def_Rating"] = (1 - shrink) * (long["pa_roll"] - half)

    # reshape to one row per game (H/A)
    ratings = long[["Sport","Game_Key","Game_Start","team","opp",
                    "Off_Rating","Def_Rating","gt_roll","LgAvg_Total_Prior"]]

    home_r = ratings.rename(columns={
        "team":"Home_Team","opp":"Away_Team",
        "Off_Rating":"Off_H","Def_Rating":"Def_H",
        "gt_roll":"GT_H","LgAvg_Total_Prior":"LgAvg_H"
    })
    away_r = ratings.rename(columns={
        "team":"Away_Team","opp":"Home_Team",
        "Off_Rating":"Off_A","Def_Rating":"Def_A",
        "gt_roll":"GT_A","LgAvg_Total_Prior":"LgAvg_A"
    })

    g = g_base.merge(
            home_r[["Game_Key","Home_Team","Away_Team","Off_H","Def_H","GT_H","LgAvg_H"]],
            on=["Game_Key","Home_Team","Away_Team"], how="left"
        ).merge(
            away_r[["Game_Key","Home_Team","Away_Team","Off_A","Def_A","GT_A","LgAvg_A"]],
            on=["Game_Key","Home_Team","Away_Team"], how="left"
        )

    # single league avg per game (prefer whichever is present)
    g["LgAvg_Total"] = g["LgAvg_H"].combine_first(g["LgAvg_A"])

    # baseline total (off+def blend) * pace multiplier (team GT vs league avg)
    base = (
        g["LgAvg_Total"]
        + 0.5*(g["Off_H"].fillna(0) + g["Def_A"].fillna(0))
        + 0.5*(g["Off_A"].fillna(0) + g["Def_H"].fillna(0))
    )
    pace_mult = (
        (g["GT_H"].fillna(g["LgAvg_Total"]) + g["GT_A"].fillna(g["LgAvg_Total"])) /
        (2.0 * g["LgAvg_Total"].replace(0, np.nan))
    ).clip(0.8, 1.2).fillna(1.0)

    g["Proj_Total_Baseline"] = base * pace_mult

    # final outputs
    out = g.rename(columns={"Game_Key": key_col})
    cols = [
        "Proj_Total_Baseline","Off_H","Def_H","Off_A","Def_A",
        "GT_H","GT_A","LgAvg_Total","Actual_Total"
    ]
    out = out[[key_col] + cols].rename(columns={c: f"TOT_{c}" for c in cols})
    return out


# =========================
# 2) Current per-game total line (median across books)
# =========================
def _current_total_line_by_key(df_any_market: pd.DataFrame,
                               key_col: str = "Merge_Key_Short") -> pd.DataFrame:
    """
    From a mixed-market DF, compute one total line per {key_col}:
      - filter to Market == 'totals'
      - median Value by {key_col}
    Returns: [{key_col}, Total_Line_Current]
    """
    if key_col not in df_any_market.columns:
        raise ValueError(f"Expected '{key_col}' in df")

    df_tot = df_any_market.copy()

    # --- HARDEN: Market may be missing depending on upstream slice ---
    if "Market" not in df_tot.columns:
        if "Market_Norm" in df_tot.columns:
            df_tot["Market"] = df_tot["Market_Norm"]
        else:
            # Can't identify totals rows; don't crash the pipeline
            return pd.DataFrame({key_col: [], "Total_Line_Current": []})

    df_tot["Market"] = df_tot["Market"].astype(str).str.strip().str.lower()
    df_tot[key_col]  = df_tot[key_col].astype(str).str.strip().str.lower()

    df_tot = df_tot[df_tot["Market"] == "totals"].copy()
    if df_tot.empty:
        return pd.DataFrame({key_col: [], "Total_Line_Current": []})

    df_tot["Value_num"] = pd.to_numeric(df_tot.get("Value"), errors="coerce")
    per_key = (
        df_tot.groupby(key_col, as_index=False)["Value_num"]
              .median()
              .rename(columns={"Value_num": "Total_Line_Current"})
    )
    return per_key


# =========================
# 3) Features for upcoming (schedule-like DF)
# =========================
def totals_features_for_upcoming(df_scores: pd.DataFrame,
                                 df_schedule_like: pd.DataFrame,
                                 sport: str | None = None,
                                 window_games: int = 30,
                                 shrink: float = 0.30,
                                 key_col: str = "Merge_Key_Short"
                                 ) -> pd.DataFrame:
    """
    Produces TOT_* features for the keys in df_schedule_like[{key_col}],
    excluding TOT_Actual_Total (label).
    """
    if key_col not in df_scores.columns or key_col not in df_schedule_like.columns:
        raise ValueError(f"Expected key_col '{key_col}' in both dataframes")

    want = df_schedule_like[[key_col]].copy()
    want[key_col] = want[key_col].astype(str).str.strip().str.lower()

    hist = build_totals_training_from_scores(
        df_scores, sport=sport, window_games=window_games, shrink=shrink, key_col=key_col
    )
    cols = [c for c in hist.columns if c.startswith("TOT_") and c != "TOT_Actual_Total"]
    return hist.merge(want, on=key_col, how="right")[[key_col] + cols]


# =========================
# 4) Attach to scoring DF (adds TOT_* and TOT_Mispricing)
# =========================
def enrich_df_with_totals_features(
    df_scoring: pd.DataFrame,
    df_scores_history: pd.DataFrame,
    *,
    sport: str | None = None,
    key_col: str = "Merge_Key_Short",
    window_games: int = 30,
    shrink: float = 0.30,
    compute_mispricing: bool = True
) -> pd.DataFrame:

    if key_col not in df_scoring.columns:
        raise ValueError(f"Expected '{key_col}' in df_scoring")

    df_sc = df_scoring.copy()
    df_sc[key_col] = df_sc[key_col].astype(str).str.strip().str.lower()

    tot_feats = totals_features_for_upcoming(
        df_scores=df_scores_history,
        df_schedule_like=df_sc[[key_col]],
        sport=sport,
        window_games=window_games,
        shrink=shrink,
        key_col=key_col,
    )

    df_sc = df_sc.merge(tot_feats, on=key_col, how="left")

    if compute_mispricing:
        if ("Market" in df_sc.columns) or ("Market_Norm" in df_sc.columns):
            cur_line = _current_total_line_by_key(df_sc, key_col=key_col)
            df_sc = df_sc.merge(cur_line, on=key_col, how="left")
            df_sc["TOT_Mispricing"] = (
                df_sc["TOT_Proj_Total_Baseline"] - df_sc["Total_Line_Current"]
            )
        else:
            df_sc["Total_Line_Current"] = np.nan
            df_sc["TOT_Mispricing"] = np.nan

    return df_sc




PHASES  = ["Overnight","Early","Midday","Late"]
URGENCY = ["VeryEarly","MidRange","LateGame","Urgent"]

def _bins(prefix: str) -> list[str]:
    return [f"{prefix}{p}_{u}" for p in PHASES for u in URGENCY]

def _sum_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(0.0, index=df.index)
    # fast, numeric-only; coerce above
    return df[cols].sum(axis=1)

def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    return a / (b.abs() + eps)

def _entropy_rowwise(M: pd.DataFrame | np.ndarray, eps: float = 1e-12) -> pd.Series:
    if isinstance(M, pd.DataFrame):
        A = M.to_numpy(dtype=float, copy=False)
    else:
        A = np.asarray(M, dtype=float)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    S = A.sum(axis=1, keepdims=True)
    W = np.divide(A, np.where(S == 0.0, 1.0, S), where=True)
    W = np.clip(W, eps, None)
    H = -(W * np.log(W)).sum(axis=1)
    return pd.Series(H, index=getattr(M, "index", None))

def _rowwise_corr_matrix(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Vectorized per-row Pearson corr between rows of X and Y (same shape)."""
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(np.asarray(Y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape or X.shape[1] == 0:
        return np.zeros((X.shape[0] if X.ndim == 2 else Y.shape[0],), dtype=float)
    mx = X.mean(axis=1, keepdims=True)
    my = Y.mean(axis=1, keepdims=True)
    Xc = X - mx
    Yc = Y - my
    sx = np.sqrt((Xc * Xc).sum(axis=1))
    sy = np.sqrt((Yc * Yc).sum(axis=1))
    denom = (sx * sy) + eps
    num = (Xc * Yc).sum(axis=1)
    corr = np.where((sx == 0.0) | (sy == 0.0), 0.0, num / denom)
    return corr

def build_timing_aggregates_inplace(
    df: pd.DataFrame,
    *,
    line_prefix: str = "SharpMove_Magnitude_",
    odds_prefix: str = "OddsMove_Magnitude_",
    drop_original: bool = False,
    include_compat_alias: bool = True,  # write legacy alias names used elsewhere
) -> list[str]:
    """
    Adds denoised timing aggregates into `df` (in-place) for current picks.
    Returns the list of newly created column names.
    Safe if some bins are missing; types are coerced to numeric.
    """
    out_cols: list[str] = []

    # Gather present bins
    all_line_bins = [c for c in _bins(line_prefix) if c in df.columns]
    all_odds_bins = [c for c in _bins(odds_prefix) if c in df.columns]

    # Coerce to numeric (fast column-wise)
    if all_line_bins:
        df[all_line_bins] = df[all_line_bins].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if all_odds_bins:
        df[all_odds_bins] = df[all_odds_bins].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Per-phase groupings (only using columns that exist)
    line_phase = {p: [f"{line_prefix}{p}_{u}" for u in URGENCY if f"{line_prefix}{p}_{u}" in df.columns] for p in PHASES}
    odds_phase = {p: [f"{odds_prefix}{p}_{u}" for u in URGENCY if f"{odds_prefix}{p}_{u}" in df.columns] for p in PHASES}

    # Totals
    df["Line_TotalMag"] = _sum_cols(df, all_line_bins)
    df["Odds_TotalMag"] = _sum_cols(df, all_odds_bins)
    out_cols += ["Line_TotalMag", "Odds_TotalMag"]

    # Per-phase sums
    for p in PHASES:
        ln = f"Line_PhaseMag_{p}"
        od = f"Odds_PhaseMag_{p}"
        df[ln] = _sum_cols(df, line_phase[p])
        df[od] = _sum_cols(df, odds_phase[p])
        out_cols += [ln, od]

    # Shares & ratios
    # Urgent share: sum of *_Urgent over TotalMag
    line_urgent_cols = [c for c in all_line_bins if c.endswith("_Urgent")]
    odds_urgent_cols = [c for c in all_odds_bins if c.endswith("_Urgent")]
    df["Line_UrgentShare"] = _safe_div(_sum_cols(df, line_urgent_cols), df["Line_TotalMag"])
    df["Odds_UrgentShare"] = _safe_div(_sum_cols(df, odds_urgent_cols), df["Odds_TotalMag"])
    out_cols += ["Line_UrgentShare", "Odds_UrgentShare"]

    df["Line_LateShare"] = _safe_div(df.get("Line_PhaseMag_Late", pd.Series(0.0, index=df.index)), df["Line_TotalMag"])
    df["Odds_LateShare"] = _safe_div(df.get("Odds_PhaseMag_Late", pd.Series(0.0, index=df.index)), df["Odds_TotalMag"])
    out_cols += ["Line_LateShare", "Odds_LateShare"]

    # Max bin magnitude (spikiness)
    df["Line_MaxBinMag"] = df[all_line_bins].max(axis=1) if all_line_bins else 0.0
    df["Odds_MaxBinMag"] = df[all_odds_bins].max(axis=1) if all_odds_bins else 0.0
    out_cols += ["Line_MaxBinMag", "Odds_MaxBinMag"]

    # Entropy (dispersion of timing)
    df["Line_Entropy"] = _entropy_rowwise(df[all_line_bins]) if all_line_bins else 0.0
    df["Odds_Entropy"] = _entropy_rowwise(df[all_odds_bins]) if all_odds_bins else 0.0
    out_cols += ["Line_Entropy", "Odds_Entropy"]

    # Cross-axis confirmations
    df["LineOddsMag_Ratio"] = _safe_div(df["Line_TotalMag"], df["Odds_TotalMag"])
    out_cols += ["LineOddsMag_Ratio"]

    # Late vs (Overnight+Early+Midday)
    df["LateVsEarly_Ratio_Line"] = _safe_div(
        df.get("Line_PhaseMag_Late", pd.Series(0.0, index=df.index)),
        (df.get("Line_PhaseMag_Overnight", 0.0) + df.get("Line_PhaseMag_Early", 0.0) + df.get("Line_PhaseMag_Midday", 0.0)),
    )
    df["LateVsEarly_Ratio_Odds"] = _safe_div(
        df.get("Odds_PhaseMag_Late", pd.Series(0.0, index=df.index)),
        (df.get("Odds_PhaseMag_Overnight", 0.0) + df.get("Odds_PhaseMag_Early", 0.0) + df.get("Odds_PhaseMag_Midday", 0.0)),
    )
    out_cols += ["LateVsEarly_Ratio_Line", "LateVsEarly_Ratio_Odds"]

    # Timing correlation across aligned bins (by suffix). If none in common → 0.
    suffixes = [f"{p}_{u}" for p in PHASES for u in URGENCY]
    common_line_cols, common_odds_cols = [], []
    for sfx in suffixes:
        lc = f"{line_prefix}{sfx}"
        oc = f"{odds_prefix}{sfx}"
        if lc in df.columns and oc in df.columns:
            common_line_cols.append(lc)
            common_odds_cols.append(oc)

    if common_line_cols and common_odds_cols:
        X = df[common_line_cols].to_numpy(dtype=float, copy=False)
        Y = df[common_odds_cols].to_numpy(dtype=float, copy=False)
        df["Timing_Corr_Line_Odds"] = _rowwise_corr_matrix(X, Y)
    else:
        df["Timing_Corr_Line_Odds"] = 0.0
    out_cols += ["Timing_Corr_Line_Odds"]

    # Optional: legacy/compat aliases some parts of your code expect
    if include_compat_alias:
        alias_map = {
            "Hybrid_Line_Imbalance_LateVsEarly": "LateVsEarly_Ratio_Line",
            "Hybrid_Line_Odds_Mag_Ratio": "LineOddsMag_Ratio",
            "Hybrid_Timing_Entropy_Line": "Line_Entropy",
            "Hybrid_Timing_Entropy_Odds": "Odds_Entropy",
        }
        for alias, src in alias_map.items():
            if alias not in df.columns and src in df.columns:
                df[alias] = df[src]
                out_cols.append(alias)

    # Optionally drop the 32 raw bins after aggregation
    if drop_original:
        df.drop(columns=all_line_bins + all_odds_bins, errors="ignore", inplace=True)

    return out_cols



# Registry for key levels (your list)
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
    ("ncaab","totals"):  [158.5,160.5,162.5,164.5,166.5,168.5],
}

def _keys_for_training(sport: str, market: str) -> np.ndarray:
    s = (sport or "").strip().lower()
    m = (market or "").strip().lower()
    ks = TRAIN_KEY_LEVELS.get((s, m))
    if ks is None:
        ks = [1.5,2.5,3,3.5,6,6.5,7] if m == "spreads" else ([7,7.5,8,8.5,9] if m=="totals" else [])
    return np.asarray(ks, dtype=np.float32)

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
    if df.empty: 
        return df

    out = df.copy()
    out["_sport_"]  = out[sport_col].astype(str).str.lower().str.strip()
    out["_market_"] = out[market_col].astype(str).str.lower().str.strip()
    out["_book_"]   = out["Bookmaker"].astype(str).str.lower().str.strip()

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

    agg = g.agg(
        sport = ("_sport_", "last"),
        market= ("_market_","last"),
        v_last= (value_col, "last"),
        v_open= (open_col,  "last"),
    ).join([hold, juice_abs, two_sided]).reset_index()

    # Dist / Pressure
    v      = pd.to_numeric(agg["v_last"], errors="coerce").astype("float32")
    v_open = pd.to_numeric(agg["v_open"], errors="coerce").astype("float32")
    dist   = np.full(len(agg), np.nan, dtype="float32")

    for (mk, sp), idx in agg.groupby(["market","sport"]).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        keys_arr = _keys_for_training(sp, mk)
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

    try:
        from config import SHARP_BOOKS  # your registry
    except Exception:
        SHARP_BOOKS = set(["pinnacle","betonlineag","lowvig","betfair_ex_uk","betfair_ex_eu","smarkets","betus"])
    mask_sharp = agg["Bookmaker"].astype(str).str.lower().isin(SHARP_BOOKS)
    sharp_median = agg.loc[mask_sharp].groupby(["Game_Key","Market"])["_val_norm_"].median()
    agg = agg.merge(sharp_median.rename("sharp_median"), on=["Game_Key","Market"], how="left")
    agg["Book_Line_Diff_vs_SharpMedian"] = (
        agg["_val_norm_"] - agg["sharp_median"].fillna(agg["_val_norm_"])
    ).astype("float32")

    OUTLIER_THRESH_SPREAD, OUTLIER_THRESH_TOTAL = 0.5, 0.5
    thr = np.where(agg["market"].values == "spreads", OUTLIER_THRESH_SPREAD, OUTLIER_THRESH_TOTAL).astype("float32")
    agg["Outlier_Flag_SharpBooks"] = (np.abs(agg["Book_Line_Diff_vs_SharpMedian"].values) >= thr).astype("uint8")

    keep = [
        "Game_Key","Market","Bookmaker",
        "Implied_Hold_Book","Two_Sided_Offered","Juice_Abs_Delta",
        "Dist_To_Next_Key","Key_Corridor_Pressure",
        "Book_PctRank_Line","Book_Line_Diff_vs_SharpMedian","Outlier_Flag_SharpBooks",
    ]
    out = out.merge(agg[keep], on=["Game_Key","Market","Bookmaker"], how="left", copy=False)
    for c, dtype in [
        ('Sharp_Move_Signal', 'float32'),
        ('Sharp_Line_Magnitude', 'float32'),
        ('Sharp_Limit_Jump', 'float32'),
        ('Sharp_Limit_Total', 'float32'),
    ]:
        if c not in out.columns:
            out[c] = np.float32(0.0)
    # Fill / downcast
    u8  = ["Two_Sided_Offered","Outlier_Flag_SharpBooks"]
    f32 = ["Implied_Hold_Book","Juice_Abs_Delta","Dist_To_Next_Key","Key_Corridor_Pressure",
           "Book_PctRank_Line","Book_Line_Diff_vs_SharpMedian"]
    for c in u8:  out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0).astype("uint8")
    for c in f32: out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0.0).astype("float32")

    out.drop(columns=["_sport_","_market_","_book_"], errors="ignore", inplace=True)
    return out

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
    out = df.copy()
    if out.empty:
        out["Line_Resistance_Crossed_Count"] = 0
        out["Was_Line_Resistance_Broken"] = 0
        out["SharpMove_Resistance_Break"] = 0
        if emit_levels_str:
            out["Line_Resistance_Crossed_Levels_Str"] = ""
        return out

    keys = [c for c in ["Game_Key","Market","Outcome","Bookmaker"] if c in out.columns]
    if keys:
        out = out.drop_duplicates(keys, keep="last")

    def _resolve(cands, fallback=None):
        for c in cands:
            if c in out.columns:
                return c
        return fallback

    value_col = _resolve(value_candidates, fallback="Value")
    odds_col  = _resolve(odds_candidates,  fallback="Odds_Price")
    open_val  = _resolve(open_value_candidates, fallback=None)
    open_odds = _resolve(open_odds_candidates,  fallback=None)

    if open_val is None:
        out["First_Line_Value"] = out[value_col]
        open_val = "First_Line_Value"
    if open_odds is None:
        out["First_Odds"] = out.get(odds_col, np.nan)
        open_odds = "First_Odds"

    for c in (value_col, open_val, odds_col, open_odds):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["_sport_"]  = out.get(sport_col, "").astype(str).str.strip().str.lower()
    out["_market_"] = out.get(market_col, "").astype(str).str.strip().str.lower()

    n = len(out)
    crossed_count = np.zeros(n, dtype=np.int16)
    broken        = np.zeros(n, dtype=np.uint8)
    levels_str    = np.full(n, "", dtype=object) if emit_levels_str else None

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
            kpos = np.flatnonzero(cnt > 0)
            for k in kpos:
                r = rows[k]
                s_keys = keys_arr[(keys_arr > lo[k]) & (keys_arr < hi[k])]
                if s_keys.size:
                    levels_str[r] = "|".join(str(float(x)).rstrip("0").rstrip(".") for x in s_keys)

    out["Line_Resistance_Crossed_Count"] = crossed_count
    out["Was_Line_Resistance_Broken"] = broken
    if emit_levels_str:
        out["Line_Resistance_Crossed_Levels_Str"] = levels_str if levels_str is not None else ""

    if "Sharp_Move_Signal" in out.columns:
        sm = pd.to_numeric(out["Sharp_Move_Signal"], errors="coerce").fillna(0).astype(np.int8)
        out["SharpMove_Resistance_Break"] = ((out["Was_Line_Resistance_Broken"] == 1) & (sm == 1)).astype(np.uint8)
    elif "Sharp_Prob_Shift" in out.columns:
        ps = pd.to_numeric(out["Sharp_Prob_Shift"], errors="coerce").fillna(0.0).astype("float32")
        out["SharpMove_Resistance_Break"] = ((out["Was_Line_Resistance_Broken"] == 1) & (ps > 0)).astype(np.uint8)
    else:
        out["SharpMove_Resistance_Break"] = out["Was_Line_Resistance_Broken"].astype(np.uint8)

    out.drop(columns=["_sport_","_market_"], inplace=True, errors="ignore")
    return out

# --- helper: row-wise entropy with safe math (no warnings, no NaNs) ---
def _row_entropy(df_like: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(0.0, index=df_like.index, dtype="float32")
    # build matrix (float64 for stability)
    M = df_like.reindex(columns=cols, fill_value=0.0).to_numpy(dtype="float64", copy=False)
    # normalize rows to probabilities
    row_sum = M.sum(axis=1, keepdims=True)  # shape (n,1)
    # P = M / row_sum, but 0 when row_sum == 0
    P = np.divide(M, row_sum, out=np.zeros_like(M), where=row_sum > 0)

    # compute log only where P>0, write 0 elsewhere
    logP = np.zeros_like(P)
    np.log(P, out=logP, where=(P > 0))

    ent = -(P * logP).sum(axis=1)  # row-wise entropy
    # Replace any residual non-finite values (paranoia)
    ent[~np.isfinite(ent)] = 0.0
    return pd.Series(ent, index=df_like.index, dtype="float32")

def compute_hybrid_timing_derivatives_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds hybrid timing magnitude/split features for line/odds moves.
    Robust to cases where early/mid/late lists (or *all*) magnitude columns are missing.
    Returns a new DataFrame; never mutates the input.
    """
    if df is None or getattr(df, "empty", True):
        return df

    import numpy as np
    import pandas as pd

    out = df.copy()
    eps = 1e-6

    # ---------- small helpers ----------

    def _zero_series(idx, dtype="float32") -> pd.Series:
        return pd.Series(0.0, index=idx, dtype=dtype)

    def _row_sum(df_: pd.DataFrame, cols: list[str], dtype: str = "float32") -> pd.Series:
        """Rowwise sum of `cols`; if empty, return 0 Series aligned to df_.index."""
        if cols:
            s = df_[cols].sum(axis=1)
            # guard against object dtype sneaking in
            try:
                return s.astype(dtype)
            except Exception:
                return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(dtype)
        return _zero_series(df_.index, dtype)

    def _row_entropy_from_cols(df_: pd.DataFrame, cols: list[str], dtype: str = "float32") -> pd.Series:
        """
        Rowwise entropy of nonnegative weights in `cols`.
        If empty, return zeros. Entropy computed as -sum(p * log p).
        """
        if not cols:
            return _zero_series(df_.index, dtype)
        W = df_[cols].to_numpy(dtype="float32")
        # coerce negatives/NaNs to 0
        np.nan_to_num(W, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        W = np.where(W < 0, 0.0, W)
        tot = W.sum(axis=1, keepdims=True)
        # p = W / max(tot, eps); where tot>0 else 0
        denom = np.maximum(tot, eps)
        P = np.divide(W, denom, out=np.zeros_like(W), where=tot > 0)
        H = -(np.where(P > 0, P * np.log(P), 0.0)).sum(axis=1)
        return pd.Series(H, index=df_.index, dtype=dtype)

    # ---------- identify column pools ----------

    line_cols = [c for c in out.columns if c.startswith("SharpMove_Magnitude_")]
    odds_cols = [c for c in out.columns if c.startswith("OddsMove_Magnitude_")]

    # fallback heuristics if strict prefixes are missing
    if not line_cols:
        line_cols += [
            c for c in out.columns
            if ("line" in c.lower() or "value" in c.lower()) and "magnitude" in c.lower()
        ]
    if not odds_cols:
        odds_cols += [
            c for c in out.columns
            if ("odds" in c.lower() or "price" in c.lower() or "prob" in c.lower())
            and "magnitude" in c.lower()
        ]

    def _pick(cols, pats):
        pats = tuple(p.lower() for p in pats)
        return [c for c in cols if any(p in c.lower() for p in pats)]

    early_l = _pick(line_cols, ("overnight", "veryearly", "early"))
    mid_l   = _pick(line_cols, ("midday", "midrange"))
    late_l  = _pick(line_cols, ("late", "urgent"))

    early_o = _pick(odds_cols, ("overnight", "veryearly", "early"))
    mid_o   = _pick(odds_cols, ("midday", "midrange"))
    late_o  = _pick(odds_cols, ("late", "urgent"))

    # ---------- magnitudes (safe for empty lists) ----------

    out["Hybrid_Line_TotalMag"] = _row_sum(out, line_cols)
    out["Hybrid_Line_EarlyMag"] = _row_sum(out, early_l)
    out["Hybrid_Line_MidMag"]   = _row_sum(out, mid_l)
    out["Hybrid_Line_LateMag"]  = _row_sum(out, late_l)

    out["Hybrid_Odds_TotalMag"] = _row_sum(out, odds_cols)
    out["Hybrid_Odds_EarlyMag"] = _row_sum(out, early_o)
    out["Hybrid_Odds_MidMag"]   = _row_sum(out, mid_o)
    out["Hybrid_Odds_LateMag"]  = _row_sum(out, late_o)

    # ---------- shares / ratios (guarded by eps) ----------

    out["Hybrid_Line_LateShare"]  = (out["Hybrid_Line_LateMag"]  / (out["Hybrid_Line_TotalMag"] + eps)).astype("float32")
    out["Hybrid_Line_EarlyShare"] = (out["Hybrid_Line_EarlyMag"] / (out["Hybrid_Line_TotalMag"] + eps)).astype("float32")
    out["Hybrid_Odds_LateShare"]  = (out["Hybrid_Odds_LateMag"]  / (out["Hybrid_Odds_TotalMag"] + eps)).astype("float32")
    out["Hybrid_Odds_EarlyShare"] = (out["Hybrid_Odds_EarlyMag"] / (out["Hybrid_Odds_TotalMag"] + eps)).astype("float32")
    out["Hybrid_Line_Odds_Mag_Ratio"] = (out["Hybrid_Line_TotalMag"] / (out["Hybrid_Odds_TotalMag"] + eps)).astype("float32")

    out["Hybrid_Line_Imbalance_LateVsEarly"] = (
        (out["Hybrid_Line_LateMag"] - out["Hybrid_Line_EarlyMag"]) / (out["Hybrid_Line_TotalMag"] + eps)
    ).astype("float32")

    # ---------- entropies (rowwise; safe for empty lists) ----------

    out["Hybrid_Timing_Entropy_Line"] = _row_entropy_from_cols(out, line_cols)
    out["Hybrid_Timing_Entropy_Odds"] = _row_entropy_from_cols(out, odds_cols)

    # ---------- base movement fields (ensure present & numeric) ----------

    for c in ("Abs_Line_Move_From_Opening", "Abs_Odds_Move_From_Opening"):
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype("float32")

    # ---------- interaction features (only if source exists) ----------

    if "Key_Corridor_Pressure" in out.columns:
        out["Corridor_x_LateShare_Line"] = (
            pd.to_numeric(out["Key_Corridor_Pressure"], errors="coerce").fillna(0.0).astype("float32")
            * out["Hybrid_Line_LateShare"]
        ).astype("float32")

    if "Dist_To_Next_Key" in out.columns:
        out["Dist_x_LateShare_Line"] = (
            pd.to_numeric(out["Dist_To_Next_Key"], errors="coerce").fillna(0.0).astype("float32")
            * out["Hybrid_Line_LateShare"]
        ).astype("float32")

    if "Book_PctRank_Line" in out.columns:
        out["PctRank_x_LateShare_Line"] = (
            pd.to_numeric(out["Book_PctRank_Line"], errors="coerce").fillna(0.0).astype("float32")
            * out["Hybrid_Line_LateShare"]
        ).astype("float32")

    return out


MICRO_NUMERIC = [
    "Implied_Hold_Book","Two_Sided_Offered","Juice_Abs_Delta",
    "Dist_To_Next_Key","Key_Corridor_Pressure",
    "Book_PctRank_Line","Book_Line_Diff_vs_SharpMedian","Outlier_Flag_SharpBooks",
    "Hybrid_Line_TotalMag","Hybrid_Line_EarlyMag","Hybrid_Line_MidMag","Hybrid_Line_LateMag",
    "Hybrid_Odds_TotalMag","Hybrid_Odds_EarlyMag","Hybrid_Odds_MidMag","Hybrid_Odds_LateMag",
    "Hybrid_Line_LateShare","Hybrid_Line_EarlyShare","Hybrid_Odds_LateShare","Hybrid_Odds_EarlyShare",
    "Hybrid_Line_Odds_Mag_Ratio","Hybrid_Line_Imbalance_LateVsEarly",
    "Hybrid_Timing_Entropy_Line","Hybrid_Timing_Entropy_Odds",
    "Corridor_x_LateShare_Line","Dist_x_LateShare_Line","PctRank_x_LateShare_Line",
]
RESIST_NUMERIC = ["Line_Resistance_Crossed_Count","Was_Line_Resistance_Broken","SharpMove_Resistance_Break"]

def _enrich_snapshot_micro_and_resistance(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in.empty:
        for c in MICRO_NUMERIC + RESIST_NUMERIC:
            if c not in df_in.columns:
                df_in[c] = 0
        return df_in

    # Make sure “open” + implied prob exist
    df_tmp = df_in.copy()
    if "First_Line_Value" not in df_tmp.columns:
        # fall back to any open you keep; keep name First_Line_Value for functions
        if "Open_Value" in df_tmp.columns:
            df_tmp["First_Line_Value"] = pd.to_numeric(df_tmp["Open_Value"], errors="coerce")
        else:
            df_tmp["First_Line_Value"] = pd.to_numeric(df_tmp.get("Value"), errors="coerce")

    if "Implied_Prob" not in df_tmp.columns:
        # compute from American price if needed
        p = pd.to_numeric(df_tmp.get("Odds_Price"), errors="coerce")
        df_tmp["Implied_Prob"] = np.where(
            p > 0, 100.0 / (p + 100.0),
            np.where(p < 0, (-p) / ((-p) + 100.0), np.nan)
        ).astype("float32")

    # Safe de-dup per (G,M,O,B) on latest ts
    if "Snapshot_Timestamp" in df_tmp.columns:
        df_tmp = (df_tmp.sort_values("Snapshot_Timestamp")
                        .drop_duplicates(subset=["Game_Key","Market","Outcome","Bookmaker"], keep="last"))

    # Add snapshot microstructure
    df_tmp = compute_snapshot_micro_features_training(
        df_tmp,
        sport_col="Sport",
        market_col="Market",
        value_col="Value",
        open_col="First_Line_Value",
        prob_col="Implied_Prob",
        price_col="Odds_Price",
    )

    # Add resistance (training-style) features
    df_tmp = add_resistance_features_training(
        df_tmp,
        sport_col="Sport",
        market_col="Market",
        emit_levels_str=False,   # flip to True if you want the string column
    )

    # Hybrid timing derivatives (built from your existing hybrid magnitude cols)
    df_tmp = compute_hybrid_timing_derivatives_training(df_tmp)

    # Type-safety & fill
    for c in MICRO_NUMERIC:
        df_tmp[c] = pd.to_numeric(df_tmp.get(c), errors="coerce").fillna(0).astype("float32")
    for c in RESIST_NUMERIC:
        df_tmp[c] = pd.to_numeric(df_tmp.get(c), errors="coerce").fillna(0).astype("uint8")

    return df_tmp

# --- Prob/De-vig helpers ------------------------------------------------------
def _amer_to_prob_one(odds) -> float:
    import numpy as np, pandas as pd
    if pd.isna(odds): return np.nan
    o = float(odds)
    return 100.0/(o+100.0) if o >= 0 else (-o)/((-o)+100.0)

def _devig_pair(p_a: float, p_b: float) -> float:
    import numpy as np
    if np.isnan(p_a) or np.isnan(p_b): return np.nan
    s = p_a + p_b
    return np.nan if s <= 0 else p_a / s  # prob for "A" leg after de-vig

# --- Binning helpers (NFL defaults; tweak per sport if needed) ----------------
def _bin_spread(x):
    import numpy as np, pandas as pd
    if pd.isna(x): return np.nan
    x = abs(float(x))
    return "≥10" if x >= 10 else ("7–9.5" if x >= 7 else ("3–6.5" if x >= 3 else "<3"))

def _bin_total(x):
    import numpy as np, pandas as pd
    if pd.isna(x): return np.nan
    x = float(x)
    return "high" if x >= 52 else ("mid" if x >= 46 else "low")

def _bin_ml(p):
    import numpy as np, pandas as pd
    if pd.isna(p): return np.nan
    p = float(p)
    if   p >= 0.80: return "≥0.80"
    elif p >= 0.70: return "0.70–0.79"
    elif p >= 0.60: return "0.60–0.69"
    else:           return "0.50–0.59"

def _rho_xy(x_bool, y_bool) -> float:
    import numpy as np, pandas as pd
    x = pd.Series(x_bool).astype(float).to_numpy()
    y = pd.Series(y_bool).astype(float).to_numpy()
    if len(x) < 150 or np.nanstd(x) == 0 or np.nanstd(y) == 0:  # stability guard
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

# --- Build cross-market pivots from (latest) snapshots ------------------------
def build_cross_market_pivots_for_training(df):
    import pandas as pd, numpy as np
    need = [c for c in ["Game_Key","Bookmaker","Market","Outcome","Value","Odds_Price","Snapshot_Timestamp"] if c in df.columns]
    if not need:
        return pd.DataFrame(columns=["Game_Key","Bookmaker","Spread_Value","Total_Value","p_over","p_ml_fav"])

    g = (df.loc[:, need]
           .sort_values("Snapshot_Timestamp")
           .groupby(["Game_Key","Bookmaker","Market","Outcome"], as_index=False)
           .tail(1))

    # Spread magnitude (abs points)
    spread = g[g["Market"].str.lower().eq("spreads")].copy()
    if not spread.empty:
        spread["Spread_Value"] = pd.to_numeric(spread["Value"], errors="coerce").abs()
        spread = spread.groupby(["Game_Key","Bookmaker"], as_index=False)\
                       .agg(Spread_Value=("Spread_Value","max"))
    else:
        spread = pd.DataFrame(columns=["Game_Key","Bookmaker","Spread_Value"])

    # Totals p(Over), de-vig
    tots = g[g["Market"].str.lower().eq("totals")].copy()
    if not tots.empty:
        tots["Outcome_l"]  = tots["Outcome"].astype(str).str.lower()
        tots["Total_Value"] = pd.to_numeric(tots["Value"], errors="coerce")
        tots["p_raw"]       = tots["Odds_Price"].map(_amer_to_prob_one)
        over  = tots[tots["Outcome_l"].eq("over") ][["Game_Key","Bookmaker","Total_Value","p_raw"]].rename(columns={"p_raw":"p_over_raw"})
        under = tots[tots["Outcome_l"].eq("under")][["Game_Key","Bookmaker","p_raw"]].rename(columns={"p_raw":"p_under_raw"})
        tot_p = over.merge(under, on=["Game_Key","Bookmaker"], how="left", validate="m:1")
        tot_p["p_over"] = tot_p.apply(lambda r: _devig_pair(r["p_over_raw"], r["p_under_raw"]), axis=1)
        tot_p = tot_p[["Game_Key","Bookmaker","Total_Value","p_over"]]
    else:
        tot_p = pd.DataFrame(columns=["Game_Key","Bookmaker","Total_Value","p_over"])

    # H2H favorite prob, de-vig
    h2h = g[g["Market"].str.lower().eq("h2h")].copy()
    if not h2h.empty:
        h2h["p_raw"] = h2h["Odds_Price"].map(_amer_to_prob_one)
        favdog = (h2h.sort_values(["Game_Key","Bookmaker","p_raw"], ascending=[True, True, False])
                    .groupby(["Game_Key","Bookmaker"], as_index=False)
                    .agg(p_fav_raw=("p_raw","first"), p_dog_raw=("p_raw","last")))
        favdog["p_ml_fav"] = favdog.apply(lambda r: _devig_pair(r["p_fav_raw"], r["p_dog_raw"]), axis=1)
        favdog = favdog[["Game_Key","Bookmaker","p_ml_fav"]]
    else:
        favdog = pd.DataFrame(columns=["Game_Key","Bookmaker","p_ml_fav"])

    out = (spread.merge(tot_p, on=["Game_Key","Bookmaker"], how="outer")
                 .merge(favdog, on=["Game_Key","Bookmaker"], how="outer"))
    return out

# --- Corr lookup builder (expects hist with required cols) --------------------
def build_corr_lookup_ST_SM_TM(hist, sport: str = "NFL"):
   
    if hist is None or hist.empty:
        # Return empty frames with stable schemas
        return (pd.DataFrame(columns=["spread_bin","total_bin","rho_ST","n"]),
                pd.DataFrame(columns=["spread_bin","ml_bin","rho_SM","n"]),
                pd.DataFrame(columns=["total_bin","ml_bin","rho_TM","n"]))
    H = hist.loc[hist["Sport"].astype(str).str.upper().eq(str(sport).upper())].copy()
    # bins (create if missing)
    if "spread_bin" not in H.columns and "close_spread" in H.columns:
        H["spread_bin"] = H["close_spread"].apply(_bin_spread)
    if "total_bin" not in H.columns and "close_total" in H.columns:
        H["total_bin"]  = H["close_total"].apply(_bin_total)
    if "ml_bin" not in H.columns and "p_ml_fav" in H.columns:
        H["ml_bin"]     = H["p_ml_fav"].apply(_bin_ml)

    # guard for label columns
    if not set(["fav_covered","went_over","fav_won"]).issubset(H.columns):
        return (pd.DataFrame(columns=["spread_bin","total_bin","rho_ST","n"]),
                pd.DataFrame(columns=["spread_bin","ml_bin","rho_SM","n"]),
                pd.DataFrame(columns=["total_bin","ml_bin","rho_TM","n"]))

    ST_rows = [{'spread_bin':sb,'total_bin':tb,'rho_ST':_rho_xy(g['fav_covered'],g['went_over']),'n':len(g)}
               for (sb,tb), g in H.groupby(['spread_bin','total_bin'], dropna=True)]
    SM_rows = [{'spread_bin':sb,'ml_bin':mb,'rho_SM':_rho_xy(g['fav_covered'],g['fav_won']),'n':len(g)}
               for (sb,mb), g in H.groupby(['spread_bin','ml_bin'], dropna=True)]
    TM_rows = [{'total_bin':tb,'ml_bin':mb,'rho_TM':_rho_xy(g['went_over'],g['fav_won']),'n':len(g)}
               for (tb,mb), g in H.groupby(['total_bin','ml_bin'], dropna=True)]

    ST_lookup = pd.DataFrame(ST_rows, columns=["spread_bin","total_bin","rho_ST","n"])
    SM_lookup = pd.DataFrame(SM_rows, columns=["spread_bin","ml_bin","rho_SM","n"])
    TM_lookup = pd.DataFrame(TM_rows, columns=["total_bin","ml_bin","rho_TM","n"])
    return ST_lookup, SM_lookup, TM_lookup

def attach_cross_market_bins_and_corr(
    df: "pd.DataFrame",
    df_all_snapshots: "pd.DataFrame",
    sport: str,
    bq=None,
    hist_table_fq: str | None = None,
):
    """
    - Builds Spread_Value / Total_Value / p_over / p_ml_fav from latest snapshots
    - Adds spread_bin / total_bin / ml_bin
    - Optionally merges rho_ST / rho_SM / rho_TM from historical lookup (if available)
    - Leaves NaNs when data is missing (no defaults).
    """
    
    if df is None or df.empty:
        return df

    # 1) Cross-market pivots from snapshots (no defaults)
    piv = pd.DataFrame(columns=["Game_Key","Bookmaker","Spread_Value","Total_Value","p_over","p_ml_fav"])
    if df_all_snapshots is not None and not df_all_snapshots.empty:
        piv = build_cross_market_pivots_for_training(df_all_snapshots)

    out = df.merge(piv, on=["Game_Key","Bookmaker"], how="left", validate="m:1")

    # 2) Bins (only when source values exist)
    if "Spread_Value" in out.columns:
        out["spread_bin"] = out["Spread_Value"].apply(_bin_spread)
    if "Total_Value" in out.columns:
        out["total_bin"]  = out["Total_Value"].apply(_bin_total)
    if "p_ml_fav" in out.columns:
        out["ml_bin"]     = out["p_ml_fav"].apply(_bin_ml)

    # 3) Optional: historical correlation lookups (BigQuery)
    ST_lookup = SM_lookup = TM_lookup = None
    if bq is not None and hist_table_fq:
        try:
            sql = f"""
            SELECT Sport, close_spread, close_total, p_ml_fav,
                   fav_covered, went_over, fav_won
            FROM `{hist_table_fq}`
            WHERE Sport = '{sport}'
            """
            hist = bq.query(sql).to_dataframe(create_bqstorage_client=False)
            ST_lookup, SM_lookup, TM_lookup = build_corr_lookup_ST_SM_TM(hist, sport=sport)
        except Exception:
            # Silent skip if table/cols unavailable
            ST_lookup = SM_lookup = TM_lookup = None

    # 4) Merge rho_* only if lookups are present and bins exist (no fills)
    if ST_lookup is not None and not ST_lookup.empty \
       and {"spread_bin","total_bin"}.issubset(out.columns):
        out = out.merge(ST_lookup[["spread_bin","total_bin","rho_ST"]],
                        on=["spread_bin","total_bin"], how="left", validate="m:1")

    if SM_lookup is not None and not SM_lookup.empty \
       and {"spread_bin","ml_bin"}.issubset(out.columns):
        out = out.merge(SM_lookup[["spread_bin","ml_bin","rho_SM"]],
                        on=["spread_bin","ml_bin"], how="left", validate="m:1")

    if TM_lookup is not None and not TM_lookup.empty \
       and {"total_bin","ml_bin"}.issubset(out.columns):
        out = out.merge(TM_lookup[["total_bin","ml_bin","rho_TM"]],
                        on=["total_bin","ml_bin"], how="left", validate="m:1")

    return out

# utils.py  — drop-in feature helpers (no Streamlit; no circular imports)


# ----------------------------- small shared helpers -----------------------------

def implied_prob_vec(s: pd.Series | np.ndarray) -> np.ndarray:
    """
    American odds -> implied probability (vectorized).
    +200 -> 100/(200+100) = 0.3333 ;  -150 -> 150/(150+100) = 0.6
    """
    o = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    p = np.full(o.shape, np.nan, dtype="float64")
    neg = o < 0
    pos = ~neg & np.isfinite(o)
    p[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
    p[pos] = 100.0 / (o[pos] + 100.0)
    return p.astype("float32")

def _norm_str(x: pd.Series) -> pd.Series:
    return x.astype(str).str.strip().str.lower()

def _first_nonnull(a: Optional[pd.Series], b: Optional[pd.Series]) -> pd.Series:
    if a is None and b is None:
        return pd.Series([], dtype="float32")
    if a is None:
        return pd.to_numeric(b, errors="coerce")
    if b is None:
        return pd.to_numeric(a, errors="coerce")
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a.where(a.notna(), b)

def _pick_keys(df: pd.DataFrame, prefs: Iterable[Iterable[str]]) -> list[str]:
    for ks in prefs:
        if all(k in df.columns for k in ks):
            return list(ks)
    return []

# ----------------------------- 1) Limit dynamics -----------------------------

def add_limit_dynamics_features(
    df: pd.DataFrame,
    *,
    spike_abs: float = 1000.0,      # absolute step-up that counts as a spike
    spike_rel: float = 0.25,        # or relative step >= 25% of previous
    epsilon_no_move: float = 1e-6,  # "no-line-move" tolerance in absolute line units
) -> pd.DataFrame:
    """
    Emits (adds columns if missing):
      - Delta_Limit (current - prev/open)
      - Limit_Spike_Flag (1/0)   — step-up detected
      - LimitSpike_x_NoMove      — spike AND line didn't move
    Prev limit is taken from:
        1) df['Prev_Limit'] if present,
        2) else df['Opening_Limit'],
        3) else 0.
    'NoMove' is |Value - Open_Value| <= epsilon OR |Line_Delta| <= epsilon.
    Works with either ('Game_Key','Market') or ('Game','Market') if present.
    """
    if df.empty:
        for c in ("Delta_Limit","Limit_Spike_Flag","LimitSpike_x_NoMove"):
            df[c] = df.get(c, pd.Series([], dtype="float32" if c=="Delta_Limit" else "int8"))
        return df

    out = df.copy()
    out["Limit"] = pd.to_numeric(out.get("Limit", np.nan), errors="coerce")
    prev_limit = out.get("Prev_Limit")
    if prev_limit is None:
        prev_limit = out.get("Opening_Limit")
    prev_limit = pd.to_numeric(prev_limit, errors="coerce")
    out["Delta_Limit"] = (out["Limit"] - prev_limit).astype("float32")

    # Spike flag (absolute OR relative to previous)
    prev_safe = prev_limit.fillna(0.0)
    spike_by_abs = out["Delta_Limit"] >= float(spike_abs)
    spike_by_rel = out["Delta_Limit"] >= (prev_safe * float(spike_rel))
    out["Limit_Spike_Flag"] = (spike_by_abs | spike_by_rel).astype("uint8")

    # No-move detection
    v_now  = pd.to_numeric(out.get("Value"), errors="coerce")
    v_open = _first_nonnull(out.get("Open_Value"), out.get("First_Line_Value"))
    line_delta = _first_nonnull(out.get("Line_Delta"), (v_now - v_open))
    no_move = line_delta.abs() <= float(epsilon_no_move)

    out["LimitSpike_x_NoMove"] = (out["Limit_Spike_Flag"].astype(bool) & no_move.fillna(False)).astype("uint8")
    return out

# ----------------------------- 2) Bookmaker network -----------------------------

def add_book_network_features(df: pd.DataFrame, sharp_books: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Emits per row:
      - Sharp_Consensus_Weight (float32)  # weighted fraction of sharp books aligned with row's side
      - Sharp_vs_Rec_SpreadGap_Q90_Q10 (float32)
      - Sharp_vs_Rec_SpreadGap_Q50 (float32)

    Requires minimally: Game_Key, Market, Bookmaker, Outcome, Value
    Optional: Book_Reliability_Score, Is_Favorite_Bet
    Robust to NaNs and missing columns. Totals use OVER/UNDER, spreads use favorite/dog.
    """
    if df.empty:
        for c in ("Sharp_Consensus_Weight","Sharp_vs_Rec_SpreadGap_Q90_Q10","Sharp_vs_Rec_SpreadGap_Q50"):
            df[c] = df.get(c, pd.Series([], dtype="float32"))
        return df

    out = df.copy()
    if sharp_books is None:
        # sane fallback; override in caller if you have a canonical list
        sharp_books = {
            "pinnacle","betonlineag","lowvig","betfair_ex_uk","betfair_ex_eu","smarkets","betus"
        }

    out["Bookmaker_norm"] = _norm_str(out.get("Bookmaker", pd.Series(index=out.index, dtype="object")))
    out["Market_norm"]    = _norm_str(out.get("Market",    pd.Series(index=out.index, dtype="object")))
    out["Outcome_Norm"]   = _norm_str(out.get("Outcome",   pd.Series(index=out.index, dtype="object")))

    # Row side flags
    fav_num = pd.to_numeric(out.get("Is_Favorite_Bet", np.nan), errors="coerce").fillna(0.0)
    is_fav_row  = (fav_num > 0.5)
    is_total    = out["Market_norm"].eq("totals")
    is_over_row = is_total & out["Outcome_Norm"].eq("over")

    # Base per (G,M,B)
    base_cols = [c for c in ["Game_Key","Game","Market","Outcome","Bookmaker","Bookmaker_norm","Market_norm","Outcome_Norm","Value","Book_Reliability_Score"] if c in out.columns]
    base = out[base_cols].drop_duplicates(subset=[c for c in ["Game_Key","Game","Market","Bookmaker"] if c in base_cols]).copy()

    base_val_num = pd.to_numeric(base.get("Value"), errors="coerce")
    base["Is_Favorite_Bet_b"] = (base_val_num < 0) & base["Market_norm"].eq("spreads")
    base["Is_Over_Bet_b"]     = base["Outcome_Norm"].eq("over") & base["Market_norm"].eq("totals")
    base["is_sharp_book"]     = base["Bookmaker_norm"].isin(set(sharp_books)).astype("uint8")

    def _weights(gdf: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(
            gdf.get("Book_Reliability_Score", pd.Series(1.0, index=gdf.index)),
            errors="coerce"
        ).fillna(1.0).astype("float32")

    rows = []
    gkeys = ["Game_Key","Market"] if "Game_Key" in base.columns else ["Game","Market"]
    for (g, m), grp in base.groupby(gkeys, sort=False):
        sharp_grp = grp.loc[grp["is_sharp_book"] == 1]
        rec_grp   = grp.loc[grp["is_sharp_book"] == 0]
        w_sharp   = _weights(sharp_grp)
        m_lc      = str(m).lower()

        if m_lc == "totals":
            aligned_mask = sharp_grp["Is_Over_Bet_b"].astype(bool)
            aligned_w = w_sharp[aligned_mask].sum()
            w_tot = float(w_sharp.sum())
            consensus_frac = float(aligned_w / w_tot) if w_tot > 0 else np.nan

            sharp_vals = pd.to_numeric(sharp_grp.get("Value"), errors="coerce").dropna()
            rec_vals   = pd.to_numeric(rec_grp.get("Value"), errors="coerce").dropna()
        else:
            aligned_mask = sharp_grp["Is_Favorite_Bet_b"].astype(bool)
            aligned_w = w_sharp[aligned_mask].sum()
            w_tot = float(w_sharp.sum())
            consensus_frac = float(aligned_w / w_tot) if w_tot > 0 else np.nan

            sharp_vals = pd.to_numeric(sharp_grp.get("Value"), errors="coerce").abs().dropna()
            rec_vals   = pd.to_numeric(rec_grp.get("Value"), errors="coerce").abs().dropna()

        def q(series, p):
            return float(series.quantile(p)) if len(series) else np.nan

        gap_q90_q10 = (q(sharp_vals, 0.90) - q(rec_vals, 0.10)) if (len(sharp_vals) and len(rec_vals)) else np.nan
        gap_q50     = (q(sharp_vals, 0.50) - q(rec_vals, 0.50)) if (len(sharp_vals) and len(rec_vals)) else np.nan

        rows.append({
            gkeys[0]: g, gkeys[1]: m,
            "Sharp_Consensus_Frac_tmp": consensus_frac,
            "Sharp_vs_Rec_SpreadGap_Q90_Q10": gap_q90_q10,
            "Sharp_vs_Rec_SpreadGap_Q50": gap_q50
        })

    net = pd.DataFrame(rows)
    out = out.merge(net, on=gkeys, how="left")

    out["Sharp_Consensus_Weight"] = np.where(
        is_total,
        np.where(is_over_row, out["Sharp_Consensus_Frac_tmp"], 1.0 - out["Sharp_Consensus_Frac_tmp"]),
        np.where(is_fav_row,  out["Sharp_Consensus_Frac_tmp"], 1.0 - out["Sharp_Consensus_Frac_tmp"])
    ).astype("float32")

    for c in ["Sharp_Consensus_Weight","Sharp_vs_Rec_SpreadGap_Q90_Q10","Sharp_vs_Rec_SpreadGap_Q50"]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0.0).astype("float32")

    out.drop(columns=["Bookmaker_norm","Market_norm","Sharp_Consensus_Frac_tmp"], errors="ignore", inplace=True)
    return out

# ----------------------------- 3) Internal consistency checks -----------------------------

def add_internal_consistency_features(
    df: pd.DataFrame,
    *,
    df_cross_market: Optional[pd.DataFrame] = None,
    sport_default: str = "NFL",
) -> pd.DataFrame:
    """
    Adds consistency diagnostics between spreads and moneyline (ML), and (lightly) totals.
    Emits:
      - Spread_ML_Prob_Strength      (favorite strength implied by spread)
      - ML_Prob_Strength             (favorite strength from H2H odds; symmetric)
      - Spread_ML_ProbGap_Signed     (spread_strength - ml_strength)
      - Spread_ML_ProbGap_Abs
      - Total_Over_Prob              (from Total_Odds or row Odds_Price if Market=='totals')
      - Total_vs_Side_ProbGap        (|Total_Over_Prob - 0.5| - |Spread_ML_Prob_Strength - 0.5|)
    Notes:
      • Uses a normal-CDF mapping from spread magnitude to favorite win prob with sport-specific scale k.
      • Symmetric favorite strength from ML uses max(p, 1-p), since team orientation may be unknown post-pivot.
    """
    if df.empty:
        for c in ["Spread_ML_Prob_Strength","ML_Prob_Strength","Spread_ML_ProbGap_Signed","Spread_ML_ProbGap_Abs","Total_Over_Prob","Total_vs_Side_ProbGap"]:
            df[c] = df.get(c, pd.Series([], dtype="float32"))
        return df

    out = df.copy()
    out["Market"] = _norm_str(out.get("Market", pd.Series(index=out.index, dtype="object")))
    out["Sport"]  = out.get("Sport", sport_default).astype(str).str.upper()

    # sport scale for converting spread -> favorite prob (tunable)
    k_map = {
        "NFL": 6.0, "NCAAF": 7.0, "CFL": 6.5,
        "NBA": 12.0, "WNBA": 10.0, "NCAAB": 10.0,
        "MLB": 2.0,  # runline approximate (weakly used here)
        "NHL": 1.8,
    }
    k = out["Sport"].map(lambda s: k_map.get(str(s).upper(), k_map.get(sport_default, 6.0))).astype("float32")
    spread_mag = pd.to_numeric(out.get("Value"), errors="coerce").abs().astype("float32")

    
    spread_strength = _phi((spread_mag / k).to_numpy(dtype="float64")).astype("float32")
    out["Spread_ML_Prob_Strength"] = spread_strength

    # ML strength: prefer H2H_Odds / H2H_Implied_Prob (pivot or row) -> symmetric favorite prob
    h2h_prob = None
    if df_cross_market is not None and "H2H_Odds" in df_cross_market.columns and "Game_Key" in out.columns:
        out = out.merge(df_cross_market[["Game_Key","H2H_Odds"]], on="Game_Key", how="left", suffixes=("", "_cm"))
        h2h_prob = implied_prob_vec(out["H2H_Odds"])
    if h2h_prob is None or np.all(np.isnan(h2h_prob)):
        # fallback to row-level Odds_Price if the row itself is H2H
        row_h2h = np.where(out["Market"].eq("h2h").to_numpy(), implied_prob_vec(out.get("Odds_Price")), np.nan)
        h2h_prob = row_h2h

    ml_strength = np.maximum(h2h_prob, 1.0 - h2h_prob)  # symmetric "favorite" prob
    out["ML_Prob_Strength"] = pd.to_numeric(ml_strength, errors="coerce").astype("float32")

    gap_signed = (out["Spread_ML_Prob_Strength"] - out["ML_Prob_Strength"]).astype("float32")
    out["Spread_ML_ProbGap_Signed"] = gap_signed
    out["Spread_ML_ProbGap_Abs"]    = gap_signed.abs().astype("float32")

    # Totals: simple over probability from Total_Odds (pivot if present)
    total_odds = None
    if df_cross_market is not None and "Total_Odds" in df_cross_market.columns and "Game_Key" in out.columns:
        out = out.merge(df_cross_market[["Game_Key","Total_Odds"]], on="Game_Key", how="left", suffixes=("", "_cm2"))
        total_odds = out["Total_Odds"]
    if total_odds is None:
        total_odds = out.get("Total_Odds", out.get("Odds_Price"))

    out["Total_Over_Prob"] = implied_prob_vec(total_odds)
    out["Total_vs_Side_ProbGap"] = (out["Total_Over_Prob"] - 0.5).abs() - (out["Spread_ML_Prob_Strength"] - 0.5).abs()
    out["Total_vs_Side_ProbGap"] = pd.to_numeric(out["Total_vs_Side_ProbGap"], errors="coerce").fillna(0.0).astype("float32")

    return out

# ----------------------------- 4) Alt-line slope & curvature (optional) -----------------------------

def add_altline_curvature(
    df: pd.DataFrame,
    df_alt: pd.DataFrame,
    *,
    alt_value_col: str = "Alt_Value",
    alt_odds_col: str  = "Alt_Odds_Price",
) -> pd.DataFrame:
    """
    Fits prob(x) ~ a*x^2 + b*x + c across alt-lines per (Game_Key, Market, Outcome).
    Emits on the base df (merged by Game_Key,Market,Outcome):
      - Implied_Distribution_Slope      (b)
      - Implied_Distribution_Curvature  (a)
    Requires df_alt with columns: Game_Key, Market, Outcome, Alt_Value, Alt_Odds_Price (or pass names).
    """
    if df.empty or df_alt is None or df_alt.empty:
        for c in ("Implied_Distribution_Slope","Implied_Distribution_Curvature"):
            df[c] = df.get(c, pd.Series([], dtype="float32"))
        return df

    alt = df_alt.copy()
    for c in ("Game_Key","Market","Outcome"):
        alt[c] = _norm_str(alt[c])
    alt["p"] = implied_prob_vec(alt[alt_odds_col])
    alt["x"] = pd.to_numeric(alt[alt_value_col], errors="coerce").astype("float32")

    rows = []
    for (g, m, o), gdf in alt.dropna(subset=["x","p"]).groupby(["Game_Key","Market","Outcome"], sort=False):
        if len(gdf) >= 3:
            # center x for numeric stability
            x = gdf["x"].to_numpy(dtype="float64")
            x0 = x.mean()
            x_c = x - x0
            y = gdf["p"].to_numpy(dtype="float64")
            # fit y ≈ a*x^2 + b*x + c
            try:
                coeffs = np.polyfit(x_c, y, deg=2)  # returns [a, b, c]
                a, b, _ = coeffs
            except Exception:
                a, b = np.nan, np.nan
        else:
            a, b = np.nan, np.nan

        rows.append({"Game_Key": g, "Market": m, "Outcome": o,
                     "Implied_Distribution_Slope": b, "Implied_Distribution_Curvature": a})

    curv = pd.DataFrame(rows)
    base = df.copy()
    for c in ("Game_Key","Market","Outcome"):
        if c in base.columns:
            base[c] = _norm_str(base[c])
    base = base.merge(curv, on=["Game_Key","Market","Outcome"], how="left")
    for c in ("Implied_Distribution_Slope","Implied_Distribution_Curvature"):
        base[c] = pd.to_numeric(base.get(c), errors="coerce").fillna(0.0).astype("float32")
    return base

# ----------------------------- 5) CLV proxy (optional) -----------------------------

def add_clv_proxy_features(
    df: pd.DataFrame,
    clv_model_path: Optional[str] = None,
    *,
    horizon_min: int = 15,
) -> pd.DataFrame:
    """
    Adds:
      - E_Delta_Line_next{horizon_min}m  (expected line change)
      - E_CLV                             (expected CLV proxy in line units)
    If clv_model_path provided and loadable via joblib, uses that model on a small
    feature set. Otherwise, uses a lightweight heuristic based on recent shifts.
    """
    if df.empty:
        for c in (f"E_Delta_Line_next{horizon_min}m","E_CLV"):
            df[c] = df.get(c, pd.Series([], dtype="float32"))
        return df

    out = df.copy()

    # Heuristic baseline (if no model)
    odds_shift = pd.to_numeric(out.get("Implied_Prob_Shift"), errors="coerce").fillna(0.0).astype("float32")
    line_delta = pd.to_numeric(out.get("Line_Delta"), errors="coerce").fillna(0.0).astype("float32")
    # small signed expectation: stronger shifts imply more move to come, but damped
    e_dl = (0.75 * line_delta + 3.5 * (odds_shift - odds_shift.mean())) * 0.15
    out[f"E_Delta_Line_next{horizon_min}m"] = e_dl.astype("float32")
    out["E_CLV"] = out[f"E_Delta_Line_next{horizon_min}m"].astype("float32")

    # Try to override with a tiny model if available
    if clv_model_path:
        try:
            from joblib import load  # local import to avoid hard dependency
            model = load(clv_model_path)
            feat_candidates = [
                "Value","Odds_Price","Implied_Prob","Implied_Prob_Shift","Line_Delta",
                "Sharp_Consensus_Weight","Sharp_Limit_Total","Limit","Delta_Limit",
                "Sharp_Move_Signal","Book_Reliability_Score"
            ]
            cols = [c for c in feat_candidates if c in out.columns]
            X = out[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
            pred = None
            if hasattr(model, "predict"):
                pred = model.predict(X)
            elif hasattr(model, "decision_function"):
                pred = model.decision_function(X)
            if pred is not None:
                pred = np.asarray(pred, dtype="float32")
                out[f"E_Delta_Line_next{horizon_min}m"] = pred
                out["E_CLV"] = pred
        except Exception:
            # keep heuristic values
            pass

    return out




def wire_ats_features_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wires ATS features that already exist on df (from the SQL view) into
    market-aware columns used at training/scoring time.

    Expects on df:
      ATS_EB_Rate, ATS_EB_Rate_Home, ATS_EB_Rate_Away,
      ATS_EB_Margin, ATS_Roll_Margin_Decay,
      Home_Team_Norm, Outcome or Outcome_Norm, Market
    Produces:
      MA_ATS_EB_Rate, MA_ATS_EB_Rate_Home, MA_ATS_EB_Rate_Away,
      MA_ATS_EB_Rate_Selected, MA_ATS_EB_Margin, MA_ATS_Roll_Margin_Decay
    """
    if df.empty:
        for c in ["MA_ATS_EB_Rate","MA_ATS_EB_Rate_Home","MA_ATS_EB_Rate_Away",
                  "MA_ATS_EB_Rate_Selected","MA_ATS_EB_Margin","MA_ATS_Roll_Margin_Decay"]:
            if c not in df.columns:
                df[c] = np.float32(np.nan)
        return df

    # --- normalize once ---
    m  = df.get('Market', pd.Series(index=df.index, dtype='object')).astype('string').str.lower().str.strip()
    on = df.get('Outcome_Norm', df.get('Outcome', pd.Series(index=df.index, dtype='object'))).astype('string').str.lower().str.strip()
    ht = df.get('Home_Team_Norm', pd.Series(index=df.index, dtype='object')).astype('string').str.lower().str.strip()

    is_spread = m.isin(['spreads','spread'])
    is_total  = m.isin(['totals','total'])
    is_h2h    = m.isin(['h2h','moneyline','ml'])
    is_st     = (is_spread | is_total)

    is_home_pick = (on == ht)

    # --- ensure sources exist & numeric ---
    for src in ["ATS_EB_Rate","ATS_EB_Rate_Home","ATS_EB_Rate_Away",
                "ATS_EB_Margin","ATS_Roll_Margin_Decay"]:
        if src not in df.columns:
            df[src] = np.float32(np.nan)
        df[src] = pd.to_numeric(df[src], errors='coerce').astype('float32')

    # ---- build full-length outputs, then mask ----
    ma_rate            = df["ATS_EB_Rate"].astype('float32')
    ma_rate_home       = df["ATS_EB_Rate_Home"].astype('float32')
    ma_rate_away       = df["ATS_EB_Rate_Away"].astype('float32')
    ma_rate_selected   = np.where(is_home_pick, ma_rate_home, ma_rate_away).astype('float32')

    # mask out markets that aren't spreads/totals (h2h)
    ma_rate          = ma_rate.where(is_st, np.nan).astype('float32')
    ma_rate_home     = ma_rate_home.where(is_st, np.nan).astype('float32')
    ma_rate_away     = ma_rate_away.where(is_st, np.nan).astype('float32')
    ma_rate_selected = pd.Series(ma_rate_selected, index=df.index).where(is_st, np.nan).astype('float32')

    # margins only for spreads (leave NaN for totals unless you’ve built totals margin)
    ma_margin        = df["ATS_EB_Margin"].where(is_spread, np.nan).astype('float32')
    ma_roll_decay    = df["ATS_Roll_Margin_Decay"].where(is_spread, np.nan).astype('float32')

    # write once (no masked assignment of mismatched arrays)
    df["MA_ATS_EB_Rate"]            = ma_rate
    df["MA_ATS_EB_Rate_Home"]       = ma_rate_home
    df["MA_ATS_EB_Rate_Away"]       = ma_rate_away
    df["MA_ATS_EB_Rate_Selected"]   = ma_rate_selected
    df["MA_ATS_EB_Margin"]          = ma_margin
    df["MA_ATS_Roll_Margin_Decay"]  = ma_roll_decay

    return df

def apply_blended_sharp_score(
    df,
    trained_models,
    df_all_snapshots=None,
    weights=None,                 # ← keep as 4th positional (back-compat)
    *,
    sport: str | None = None,     # ← keyword-only now
    mem_profile: bool = True,
    mem_interval_s: float = 5.0,
    mem_topn: int = 8,
    mem_gc_threshold_mb: float | None = None,
    mem_log_df_columns: bool = False,
):



        # --- init sampler ---
    ms = MemSampler(
        logger,
        interval_s=mem_interval_s,
        topn=mem_topn,
        deep_columns=False,
        gc_threshold_mb=mem_gc_threshold_mb,
    ) if mem_profile else None

    def _mem(tag: str):
        if ms:
            # pass locals() so we can see big frames like df, odds_now, etc.
            ms.maybe(tag, frame_locals=locals())

    _mem("start")

    df_empty = pd.DataFrame()
    total_start = time.time()
    scored_all = []
    # ---------- models presence ----------
    # ---------- models presence ----------

    # ---------- models presence (MUST be before any HAS_MODELS usage) ----------
    trained_models = trained_models or {}
    
    # normalize keys but don’t filter out tuples/objects
    trained_models_norm = {str(k).strip().lower(): v for k, v in trained_models.items()}
    
    # back-compat aliases for older code paths
    trained_models_lc = trained_models_norm
    trained_models_by_market = trained_models_norm
    model_markets_lower = sorted(trained_models_norm.keys())
    
    def _has_any_model(bundle):
        if isinstance(bundle, dict):
            return any(k in bundle for k in (
                "model", "calibrator",
                "model_logloss", "model_auc",
                "calibrator_logloss", "calibrator_auc"
            ))
        return bundle is not None  # accept tuples or single models too
    
    HAS_MODELS = any(_has_any_model(v) for v in trained_models_norm.values())
    logger.info("📦 HAS_MODELS=%s; model markets: %s",
                HAS_MODELS, sorted(trained_models_norm.keys()))
    # ------------------------------------------------------------------------

    # ------------------ HARD BACKFILL: Market ------------------
    def _norm_market(m: str) -> str:
        m = ("" if m is None else str(m)).lower().strip()
        if m in ("spread", "spreads", "ats") or ("spread" in m): return "spreads"
        if m in ("total", "totals", "ou", "overunder") or ("total" in m): return "totals"
        if m in ("h2h", "ml", "moneyline", "money_line") or ("money" in m) or ("h2h" in m): return "h2h"
        return m
    
    if 'Market' not in df.columns:
        df['Market'] = ""
    
    m_raw = df['Market'].astype('string').fillna("").str.lower().str.strip()
    
    # If Market_Norm exists and is usable, use it
    if 'Market_Norm' in df.columns:
        mn = df['Market_Norm'].astype('string').fillna("").str.lower().str.strip()
        m_raw = m_raw.where(m_raw.ne(""), mn)
    
    # Infer for rows still blank
    blank = (m_raw == "")
    if blank.any():
        v = pd.to_numeric(df.loc[blank, 'Value'], errors='coerce')
        o = pd.to_numeric(df.loc[blank, 'Odds_Price'], errors='coerce') if 'Odds_Price' in df.columns else pd.Series(np.nan, index=v.index)
    
        # Heuristics:
        # - H2H: Value often equals Odds_Price (american odds) and magnitude is large
        is_h2h = v.notna() & o.notna() & (v.abs() >= 80) & (o.abs() >= 80) & ((v - o).abs() < 1e-6)
        # - Spreads: point line magnitude usually <= ~40
        is_spreads = v.notna() & (v.abs() <= 40) & (~is_h2h)
        # - Totals: point line magnitude typically > 40 and < 400
        is_totals = v.notna() & (v.abs() > 40) & (v.abs() < 400) & (~is_h2h)
    
        m_raw.loc[is_h2h.index[is_h2h]] = "h2h"
        m_raw.loc[is_spreads.index[is_spreads]] = "spreads"
        m_raw.loc[is_totals.index[is_totals]] = "totals"
    
    df['Market'] = m_raw.map(_norm_market)
    df['Market_norm'] = df['Market'].astype('string').map(_norm_market)
        

    
    # normalize df markets
    if 'Market' in df.columns and not df.empty:
        df['_Market_raw'] = df['Market'].astype('string')
        df['Market_norm'] = df['_Market_raw'].map(_norm_market).astype('string')

        logger.info("🧭 Raw Market uniques: %s", sorted(df['_Market_raw'].dropna().astype(str).str.lower().str.strip().unique().tolist())[:50])
        logger.info("🧭 Norm Market uniques: %s", sorted(df['Market_norm'].dropna().unique().tolist()))
    else:
        df['Market_norm'] = pd.Series(pd.NA, index=df.index, dtype='string')
    
    # intersection of markets present and markets modeled
    if HAS_MODELS and not df.empty:
        df_mkts = set(df['Market_norm'].dropna().unique().tolist())
        model_mkts = set(trained_models_norm.keys())
        markets_present = sorted(df_mkts.intersection(model_mkts))
    else:
        markets_present = []
    
    logger.info("✅ Eligible markets to score: %s", markets_present)
    # ---------- frame guard ----------
    if df is None or len(df) == 0:
        df = pd.DataFrame()
        return df  # nothing to score

    # ⚠️ NO full copy here: rely on pandas Copy-on-Write to avoid surprises.
    # Normalize minimal columns in-place.
    if 'Market' in df.columns:
        df['Market'] = df['Market'].astype('string').str.lower().str.strip()

    # ensure presence/typing (do once, downcast)
    for c in ('Odds_Price', 'Value', 'Limit'):
        if c in df.columns:
            # downcast floats; Limit default 0.0 if missing
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    if 'Limit' not in df.columns:
        df['Limit'] = np.float32(0.0)

    
    if 'Merge_Key_Short' in df.columns:
        df['Merge_Key_Short'] = df['Merge_Key_Short'].astype('string').str.strip().str.lower()
    elif 'Game_Key' in df.columns and df['Game_Key'].notna().any():
        df['Merge_Key_Short'] = df['Game_Key'].astype('string').str.strip().str.lower()
    else:
        logger.warning("⚠️ Merge_Key_Short missing and cannot be derived; downstream totals features may fail")
        df['Merge_Key_Short'] = pd.Series(pd.NA, index=df.index, dtype='string')

    
    # Timestamp: a single scalar; avoid assigning per-row Python object repeatedly
    snap_ts = pd.Timestamp.utcnow()
    df['Snapshot_Timestamp'] = snap_ts

    # 'Bookmaker' fallback + normalization
    if 'Bookmaker' not in df.columns and 'Book' in df.columns:
        df['Bookmaker'] = df['Book']
    elif 'Bookmaker' in df.columns and 'Book' not in df.columns:
        df['Book'] = df['Bookmaker']

    if 'Bookmaker' in df.columns:
        df['Bookmaker'] = df['Bookmaker'].astype('string').str.lower().str.strip()

    # Is_Sharp_Book → int8
    if 'Bookmaker' in df.columns:
        # assume SHARP_BOOKS is lowercased already; otherwise map/normalize once.
        df['Is_Sharp_Book'] = df['Bookmaker'].isin(SHARP_BOOKS).astype('int8')
    else:
        df['Is_Sharp_Book'] = np.int8(0)

    # Event date only if Game_Start exists (downcast to date, not full datetime per row)
    if 'Game_Start' in df.columns:
        gstart = pd.to_datetime(df['Game_Start'], errors='coerce', utc=True)
        df['Event_Date'] = gstart.dt.date  # object dtype but small; skip if unused

    # Normalize merge keys (lower/strip) if present — do NOT rebuild entire cols
    for k in ('Game_Key','Market','Outcome','Bookmaker'):
        if k in df.columns:
            df[k] = df[k].astype('string').str.strip().str.lower()

    # Ensure 'Book' exists (already handled above)

    # ---- Presence log (no heavy ops) ----
    want_cols = ['Open_Value','Open_Odds','Opening_Limit','First_Imp_Prob',
                 'Max_Value','Min_Value','Max_Odds','Min_Odds']
    have_cols = [c for c in want_cols if c in df.columns]
    logger.info("🔎 Pre-scoring columns present: %s", have_cols)

    # ---------- Implied_Prob (vectorized) ----------
    if 'Implied_Prob' not in df.columns and 'Odds_Price' in df.columns:
        df['Implied_Prob'] = implied_prob_vec(df['Odds_Price'])
    elif 'Implied_Prob' in df.columns:
        # ensure float32 to cut memory
        df['Implied_Prob'] = pd.to_numeric(df['Implied_Prob'], errors='coerce').astype('float32')

    # ---------- Openers / Extremes (minimal passes, float32) ----------
    # helper to coerce float32 quickly
    def _f32(s): return pd.to_numeric(s, errors='coerce').astype('float32')

    if 'Open_Value' in df.columns:
        df['Open_Value'] = _f32(df['Open_Value'])
        df['Open_Value'] = df['Open_Value'].where(~df['Open_Value'].isna(), df.get('Value'))
    else:
        df['Open_Value'] = df.get('Value')

    if 'Open_Odds' in df.columns:
        df['Open_Odds'] = _f32(df['Open_Odds'])
        df['Open_Odds'] = df['Open_Odds'].where(~df['Open_Odds'].isna(), df.get('Odds_Price'))
    else:
        df['Open_Odds'] = df.get('Odds_Price')

    if 'Opening_Limit' in df.columns:
        df['Opening_Limit'] = _f32(df['Opening_Limit']).replace(np.float32(0.0), np.float32(np.nan))
    else:
        # keep NaN to avoid pretending we know opener limits
        df['Opening_Limit'] = np.float32(np.nan)

    # First_Imp_Prob: prefer existing, else from Open_Odds, else from Implied_Prob
    if 'First_Imp_Prob' not in df.columns:
        df['First_Imp_Prob'] = np.float32(np.nan)
    else:
        df['First_Imp_Prob'] = _f32(df['First_Imp_Prob'])

    need_imp = df['First_Imp_Prob'].isna() & df['Open_Odds'].notna()
    if need_imp.any():
        df.loc[need_imp, 'First_Imp_Prob'] = implied_prob_vec(df.loc[need_imp, 'Open_Odds'])
    if df['First_Imp_Prob'].isna().any() and 'Implied_Prob' in df.columns:
        df['First_Imp_Prob'] = df['First_Imp_Prob'].fillna(df['Implied_Prob'])

    # Extremes: single pass with where()
    def _ensure_col(col, fb):
        if col in df.columns:
            df[col] = _f32(df[col]).where(~df[col].isna(), df.get(fb))
        else:
            df[col] = df.get(fb)
    _ensure_col('Max_Value',  'Value')
    _ensure_col('Min_Value',  'Value')
    _ensure_col('Max_Odds',   'Odds_Price')
    _ensure_col('Min_Odds',   'Odds_Price')

    # ---------- Quick diagnostics (cheap) ----------
    def _pct_nan(c):
        return float(df[c].isna().mean() * 100.0) if c in df.columns else 100.0
    logger.info("📊 Missing Open_Value: %.2f%%", _pct_nan('Open_Value'))
    logger.info("📊 Missing Open_Odds: %.2f%%", _pct_nan('Open_Odds'))
    logger.info("📊 Missing First_Imp_Prob: %.2f%%", _pct_nan('First_Imp_Prob'))
    logger.info("📊 Missing Opening_Limit: %.2f%%", _pct_nan('Opening_Limit'))
    
    
    
    _mem("post-openers-extremes")
    # ---------- Lightweight preview (no sort/dedup over the whole frame) ----------
    try:
        cols = ['Game_Key','Market','Outcome','Bookmaker','Odds_Price','Value',
                'Open_Odds','Open_Value','First_Imp_Prob','Max_Value','Min_Value','Max_Odds','Min_Odds']
        # Select what exists (avoid KeyError) and slice a few rows
        cols = [c for c in cols if c in df.columns]
        logger.info("🧪 Sample of enriched df after merge:\n%s",
                    df.loc[:10, cols].to_string(index=False))
    except Exception as e:
        logger.warning("⚠️ Failed to print preview: %s", e)
    _suffix_snapshot(df, "before helpers")
  

    # ---------- 0) Cast once up front ----------
    odds_now   = pd.to_numeric(df['Odds_Price'], errors='coerce').astype('float64')
    odds_open  = pd.to_numeric(df['Open_Odds'],  errors='coerce').astype('float64')
    val_now    = pd.to_numeric(df['Value'],      errors='coerce').astype('float64')
    val_open   = pd.to_numeric(df['Open_Value'], errors='coerce').astype('float64')
    df['Limit'] = pd.to_numeric(df['Limit'],     errors='coerce').fillna(0.0).astype('float64')
    _suffix_snapshot(df, "after cast helpers") 
    # ---------- 1) Vectorized implied prob (American odds) ----------
    # Handles NaN; does not allocate extra Series; branch by mask once.
    def implied_prob_vec_raw(o: np.ndarray) -> np.ndarray:
        p = np.full(o.shape, np.nan, dtype='float64')
        neg = o < 0
        pos = ~neg & np.isfinite(o)
        # p = |-o| / (|-o| + 100) for negative odds
        oo = -o
        p[neg] = oo[neg] / (oo[neg] + 100.0)
        # p = 100 / (o + 100) for positive odds
        p[pos] = 100.0 / (o[pos] + 100.0)
        return p
    
    imp_now  = implied_prob_vec_raw(odds_now.values)
    imp_open = implied_prob_vec_raw(odds_open.values)
    
    # Keep as Series aligned to df index (cheap view, no copy of data)
    imp_now_s  = pd.Series(imp_now,  index=df.index)
    imp_open_s = pd.Series(imp_open, index=df.index)
    
    # ---------- 2) Odds & implied-prob shifts ----------
    df['Odds_Shift'] = (imp_now_s - imp_open_s) * 100.0
    
    if 'Implied_Prob' not in df.columns or df['Implied_Prob'].isna().all():
        df['Implied_Prob'] = imp_now_s
    else:
        # Ensure numeric once; no need to re-calc
        df['Implied_Prob'] = pd.to_numeric(df['Implied_Prob'], errors='coerce')
    
    if 'First_Imp_Prob' not in df.columns or df['First_Imp_Prob'].isna().all():
        df['First_Imp_Prob'] = imp_open_s
    else:
        df['First_Imp_Prob'] = pd.to_numeric(df['First_Imp_Prob'], errors='coerce')
    
    df['Implied_Prob_Shift'] = (df['Implied_Prob'] - df['First_Imp_Prob'])
    
    # If you truly don't need First_Imp_Prob beyond this point, drop to save memory
    df.drop(columns=['First_Imp_Prob'], inplace=True, errors='ignore')
    
    # ---------- 3) Line deltas (single-pass) ----------
    line_delta = (val_now - val_open)
    df['Line_Delta']           = line_delta
    df['Delta']                = line_delta           # alias
    abs_delta                  = np.abs(line_delta.values)
    df['Line_Magnitude_Abs']   = abs_delta
    df['Line_Move_Magnitude']  = abs_delta

    # ---------- 3b) Line resistance (binary break + continuous factor) ----------
    # ---------- 3b) Line resistance (binary break + continuous factor) ----------
   
    # ---------- 4) Value reversal (vectorized, single cast) ----------
    def compute_value_reversal(df: pd.DataFrame, market_col: str = 'Market') -> pd.DataFrame:
        m = df[market_col].astype(str).str.lower().str.strip()
        is_spread = m.str.contains('spread', na=False).values
        is_total  = m.str.contains('total',  na=False).values
        is_h2h    = m.str.contains('h2h', na=False).values
    
        v_open = pd.to_numeric(df.get('Open_Value'), errors='coerce').astype('float64').values
        v_now  = pd.to_numeric(df.get('Value'),      errors='coerce').astype('float64').values
        v_max  = pd.to_numeric(df.get('Max_Value'),  errors='coerce').astype('float64').values
        v_min  = pd.to_numeric(df.get('Min_Value'),  errors='coerce').astype('float64').values
        outcome_norm = df.get('Outcome_Norm', '').astype(str).str.lower().str.strip().values
    
        # Spread: reversal if moved toward zero past open and sits at the extreme
        spread_flag = (
            ((v_open < 0) & (v_now >  v_open) & (v_now == v_max)) |
            ((v_open > 0) & (v_now <  v_open) & (v_now == v_min))
        )
    
        # Totals: direction depends on over/under label
        total_flag = (
            ((outcome_norm == 'over')  & (v_now > v_open) & (v_now == v_max)) |
            ((outcome_norm == 'under') & (v_now < v_open) & (v_now == v_min))
        )
    
        # H2H: any move to current extreme opposite open
        h2h_flag = (
            ((v_now > v_open) & (v_now == v_max)) |
            ((v_now < v_open) & (v_now == v_min))
        )
    
        out = np.zeros(len(df), dtype='int8')
        out[is_spread] = spread_flag[is_spread]
        out[is_total]  = total_flag[is_total]
        out[is_h2h]    = h2h_flag[is_h2h]
        df['Value_Reversal_Flag'] = out.astype(int)
        return df
    
    
    def compute_odds_reversal(df: pd.DataFrame, prob_threshold: float = 0.05) -> pd.DataFrame:
        """
        Vectorized odds-reversal flags.
        Writes: Odds_Reversal_Flag (0/1), Abs_Odds_Prob_Move (float)
        """
    
        # ---------- helpers ----------
        def implied_prob_vec_raw(o: np.ndarray) -> np.ndarray:
            # American odds -> implied prob, vectorized
            p = np.full(o.shape, np.nan, dtype='float64')
            neg = o < 0
            pos = ~neg & np.isfinite(o)
            oo = -o
            p[neg] = oo[neg] / (oo[neg] + 100.0)
            p[pos] = 100.0 / (o[pos] + 100.0)
            return p
    
        n = len(df)
        if n == 0:
            df['Odds_Reversal_Flag'] = np.array([], dtype=int)
            df['Abs_Odds_Prob_Move'] = np.array([], dtype='float64')
            return df
    
        # ---------- market masks (once) ----------
        m = df.get('Market')
        if m is None:
            is_spread = np.zeros(n, dtype=bool)
            is_total  = np.zeros(n, dtype=bool)
            is_h2h    = np.zeros(n, dtype=bool)
        else:
            m = m.astype(str).str.lower().str.strip()
            is_spread = m.str.contains('spread', na=False).values
            is_total  = m.str.contains('total',  na=False).values
            is_h2h    = m.str.contains('h2h',    na=False).values
    
        # ---------- ensure key prob columns (cheap) ----------
        # Implied_Prob (now)
        if 'Implied_Prob' in df.columns and not df['Implied_Prob'].isna().all():
            imp_now = pd.to_numeric(df['Implied_Prob'], errors='coerce').astype('float64').values
        else:
            odds_now = pd.to_numeric(df.get('Odds_Price'), errors='coerce').astype('float64').values
            imp_now  = implied_prob_vec_raw(odds_now)
            df['Implied_Prob'] = imp_now
    
        # First_Imp_Prob (open)
        if 'First_Imp_Prob' in df.columns and not df['First_Imp_Prob'].isna().all():
            imp_open = pd.to_numeric(df['First_Imp_Prob'], errors='coerce').astype('float64').values
        else:
            base_open = pd.to_numeric(df.get('Open_Odds', df.get('Odds_Price')), errors='coerce').astype('float64').values
            imp_open  = implied_prob_vec_raw(base_open)
            df['First_Imp_Prob'] = imp_open
    
        # Min/Max probs for H2H extremes
        min_odds = pd.to_numeric(df.get('Min_Odds', df.get('Odds_Price')), errors='coerce').astype('float64').values
        max_odds = pd.to_numeric(df.get('Max_Odds', df.get('Odds_Price')), errors='coerce').astype('float64').values
        min_prob = implied_prob_vec_raw(min_odds)
        max_prob = implied_prob_vec_raw(max_odds)
    
        # ---------- shifts ----------
        if 'Implied_Prob_Shift' in df.columns:
            abs_shift = pd.to_numeric(df['Implied_Prob_Shift'], errors='coerce').abs().astype('float64').values
        else:
            shift = (imp_now - imp_open)
            abs_shift = np.abs(shift)
            df['Implied_Prob_Shift'] = shift
    
        # ---------- flags ----------
        # H2H: reverse if current hits/extremes opposite the open direction (with tiny epsilon)
        eps = 1e-5
        valid = np.isfinite(imp_open) & np.isfinite(imp_now) & np.isfinite(min_prob) & np.isfinite(max_prob)
        h2h_flag = np.zeros(n, dtype=int)
        if valid.any():
            v = valid
            first = imp_open[v]
            curr  = imp_now[v]
            minp  = min_prob[v]
            maxp  = max_prob[v]
            h2h_flag_v = (
                ((first > minp) & (curr <= (minp + eps))) |
                ((first < maxp) & (curr >= (maxp - eps)))
            ).astype(int)
            h2h_flag[v] = h2h_flag_v
    
        # Spread/Totals: absolute prob shift threshold
        st_flag = (abs_shift >= float(prob_threshold)).astype(int)
    
        # Compose final flag
        out = np.zeros(n, dtype=int)
        # Order: assign by masks (last assignment wins if overlapping — they shouldn't)
        out[is_spread | is_total] = st_flag[is_spread | is_total]
        out[is_h2h] = h2h_flag[is_h2h]
    
        df['Odds_Reversal_Flag'] = out
        df['Abs_Odds_Prob_Move'] = abs_shift
    
        return df

    _mem("post-line-reverals")
    # If Time missing, derive from Snapshot_Timestamp for Sharp_Timing
    # earlier, ensure this once:
    df['Time'] = pd.to_datetime(df.get('Time', df['Snapshot_Timestamp']),
                                errors='coerce', utc=True)
    
    # later:
    df['Sharp_Timing'] = df['Time'].dt.hour.map(
        lambda h: 1.0 if 0 <= h <= 5 else 0.9 if 6 <= h <= 11 else 0.5 if 12 <= h <= 15 else 0.2
    )  # this map is fine; it's fast enough

    df['Limit_NonZero'] = df['Limit'].where(df['Limit'] > 0)
    # Group keys must exist; your df includes 'Game' and 'Market'

    # Use keys that actually exist in this frame
    grp_keys = [k for k in ['Game_Key', 'Market'] if k in df.columns]
    if grp_keys:
        df['Limit_Max'] = df.groupby(grp_keys, dropna=False)['Limit_NonZero'].transform('max')
        df['Limit_Min'] = df.groupby(grp_keys, dropna=False)['Limit_NonZero'].transform('min')
    else:
        df['Limit_Max'] = df['Limit_NonZero']
        df['Limit_Min'] = df['Limit_NonZero']
    # === Market leader flags (robust to missing df_all_snapshots) ===
    if df_all_snapshots is not None and isinstance(df_all_snapshots, (pd.DataFrame,)):
        try:
            market_leader_flags = detect_market_leaders(df_all_snapshots, SHARP_BOOKS, REC_BOOKS)
            if not market_leader_flags.empty:
                # Ensure join keys exist & are normalized
                for col in ['Game','Market','Outcome','Book']:
                    if col in market_leader_flags.columns:
                        market_leader_flags[col] = market_leader_flags[col].astype(str).str.lower().str.strip()
                for col in ['Game','Market','Outcome','Book']:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.lower().str.strip()
                if 'Book' not in df.columns and 'Bookmaker' in df.columns:
                    df['Book'] = df['Bookmaker']
                df = df.merge(
                    market_leader_flags[['Game','Market','Outcome','Book','Market_Leader']],
                    on=['Game','Market','Outcome','Book'],
                    how='left',
                    validate='m:1'
                )
        except Exception as e:
            logger.warning(f"⚠️ detect_market_leaders merge skipped: {e}")
    if 'Market_Leader' not in df.columns:
        df['Market_Leader'] = 0
    # Example names; adjust to your variables if different
    df = attach_cross_market_bins_and_corr(
            df=df,
            df_all_snapshots=df_all_snapshots,
            sport=sport,                              # e.g., "NFL"
            bq=bq_client,                             # your bigquery.Client or None
            hist_table_fq="sharplogger.sharp_data.scores_with_features"  # or None to skip rho_*
    )

    # === Flag Pinnacle no-move behavior
    df['Is_Pinnacle'] = (df['Book'].astype(str).str.lower().str.strip() == 'pinnacle')
    df['LimitUp_NoMove_Flag'] = (
        (df['Is_Pinnacle']) &
        (df['Limit'] >= 2500) &
        (df['Value'] == df['Open_Value'])
    ).astype(int)

    # === 3b) Line resistance + snapshot microstructure + hybrid timing ===
    # add timing aggregates for current picks (in-place)
    timing_cols = build_timing_aggregates_inplace(
        df,
        line_prefix="SharpMove_Magnitude_",
        odds_prefix="OddsMove_Magnitude_",
        drop_original=False,           # ← keep raw timing bins so derivatives can see them
        include_compat_alias=True,
    )

    # tiny sanity log (optional)
    _expected = [
        "Line_TotalMag","Line_LateShare","Line_UrgentShare",
        "Line_MaxBinMag","Line_Entropy","LineOddsMag_Ratio","Timing_Corr_Line_Odds"
    ]
    _missing = [c for c in _expected if c not in df.columns]
    if _missing:
        logger.warning("Timing aggregates missing (ok if no bins present): %s", _missing)

    # Build latest-per-(G,M,O,B) view to avoid fanout on merge
    df_pre = (
        df.sort_values('Snapshot_Timestamp')
          .drop_duplicates(['Game_Key','Market','Outcome','Bookmaker'], keep='last')
          .copy()
    )
    
    # Compute microstructure, resistance, and hybrid timing rollups
    df_enrich = _enrich_snapshot_micro_and_resistance(df_pre)
    
    # Merge enriched scalars back (many_to_one prevents accidental row multiplication)
    merge_keys = ['Game_Key','Market','Outcome','Bookmaker']
    cols_to_bring = merge_keys + MICRO_NUMERIC + RESIST_NUMERIC
    df = df.merge(
        df_enrich[cols_to_bring],
        on=merge_keys,
        how='left',
        validate='m:1'
    )
    
    # Type safety (keeps xgb/sklearn happy; compact dtypes)
    for c in MICRO_NUMERIC:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype('float32')
    
    for c in RESIST_NUMERIC:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype('uint8')
    
         
    
    # === Cross-market support (optional)
    df = detect_cross_market_sharp_support(df, SHARP_BOOKS)
    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Unique_Sharp_Books']      = df['Unique_Sharp_Books'].fillna(0).astype(int)
    df['LimitUp_NoMove_Flag']     = df['LimitUp_NoMove_Flag'].fillna(False).astype(int)
    df['Market_Leader']           = df['Market_Leader'].fillna(False).astype(int)
    
    # ---- Power ratings enrich (current) -----------------------------------------
    try:
        enrich_power_from_current_inplace(
            df,
            sport_aliases=SPORT_ALIASES,
            table_current="sharplogger.sharp_data.ratings_current",
            project="sharplogger",
            baseline=1500.0,
        )
    except Exception as e:
        logger.warning("⚠️ Power rating enrichment (current) failed: %s", e, exc_info=True)
        df["Home_Power_Rating"] = np.float32(1500.0)
        df["Away_Power_Rating"] = np.float32(1500.0)
        df["Power_Rating_Diff"] = np.float32(0.0)
    
    # ---- Bet-side team vs opponent PR columns -----------------------------------
    if 'is_home_bet' not in df.columns:
        # Derive: Outcome matches home team
        home_norm    = df.get('Home_Team_Norm', '').astype(str).str.lower().str.strip()
        outcome_norm = df['Outcome'].astype(str).str.lower().str.strip()
        df['is_home_bet'] = (outcome_norm == home_norm)
    
    is_home_bet = df['is_home_bet']
    
    df['PR_Team_Rating'] = np.where(is_home_bet, df['Home_Power_Rating'], df['Away_Power_Rating']).astype('float32')
    df['PR_Opp_Rating']  = np.where(is_home_bet, df['Away_Power_Rating'], df['Home_Power_Rating']).astype('float32')
    df['PR_Rating_Diff']     = (df['PR_Team_Rating'] - df['PR_Opp_Rating']).astype('float32')
    df['PR_Abs_Rating_Diff'] = df['PR_Rating_Diff'].abs().astype('float32')
    
    # ---- H2H-only alignment flags (Model & Market vs PR) ------------------------
    mkt    = df['Market'].astype(str).str.strip().str.lower()
    is_h2h = mkt.isin(['h2h','moneyline','ml','headtohead'])
    
    # Pick the model prob column
    prob_col = 'Model Prob' if 'Model Prob' in df.columns else 'Model_Sharp_Win_Prob'
    if prob_col not in df.columns:
        df[prob_col] = np.nan
    p_model = pd.to_numeric(df[prob_col], errors='coerce')
    
    # Model vs PR
    df['PR_Model_Agree_H2H'] = np.where(
        is_h2h & p_model.notna(),
        ((df['PR_Rating_Diff'] > 0) & (p_model > 0.5)) |
        ((df['PR_Rating_Diff'] < 0) & (p_model < 0.5)),
        np.nan
    )
    
    # Market vs PR (use Implied_Prob if present; else derive from Odds_Price)
    if 'Implied_Prob' in df.columns:
        p_mkt = pd.to_numeric(df['Implied_Prob'], errors='coerce')
    else:
        odds_ = pd.to_numeric(df.get('Odds_Price'), errors='coerce')
        p_mkt = pd.Series(np.where(
            odds_.notna(),
            np.where(odds_ < 0, (-odds_) / ((-odds_) + 100.0), 100.0 / (odds_ + 100.0)),
            np.nan
        ), index=df.index)
    
    df['PR_Market_Agree_H2H'] = np.where(
        is_h2h & p_mkt.notna(),
        ((df['PR_Rating_Diff'] > 0) & (p_mkt > 0.5)) |
        ((df['PR_Rating_Diff'] < 0) & (p_mkt < 0.5)),
        np.nan
    )
    
    # UI-friendly labels (optional)
    df['PR_Model_Alignment_H2H'] = np.select(
        [df['PR_Model_Agree_H2H'] == True, df['PR_Model_Agree_H2H'] == False],
        ["✅ PR ↔ Model Agree", "❌ PR ≠ Model"],
        default="—"
    )
    df['PR_Market_Alignment_H2H'] = np.select(
        [df['PR_Market_Agree_H2H'] == True, df['PR_Market_Agree_H2H'] == False],
        ["✅ PR ↔ Market Agree", "❌ PR ≠ Market"],
        default="—"
    )
    
    # Numeric flags (nullable Int8). Use where() to avoid broadcast shape issues.
    df['PR_Model_Agree_H2H_Flag']  = (df['PR_Model_Agree_H2H']  == True).astype('Int8')
    df['PR_Market_Agree_H2H_Flag'] = (df['PR_Market_Agree_H2H'] == True).astype('Int8')
    
    df['PR_Model_Agree_H2H_Flag']  = df['PR_Model_Agree_H2H_Flag'].where(is_h2h, pd.NA)
    df['PR_Market_Agree_H2H_Flag'] = df['PR_Market_Agree_H2H_Flag'].where(is_h2h, pd.NA)
    
    # Ensure non-H2H rows show blanks/—
    df.loc[~is_h2h, ['PR_Model_Agree_H2H','PR_Market_Agree_H2H']] = np.nan
    df.loc[~is_h2h, ['PR_Model_Alignment_H2H','PR_Market_Alignment_H2H']] = "—"
    
    
    _suffix_snapshot(df, "after detect cross market")
    # === Confidence scores and tiers
    try:
        if weights:
            df = assign_confidence_scores(df, weights)
        else:
            logging.warning("⚠️ Skipping confidence scoring — 'weights' not defined.")
    except Exception as e:
        logging.warning(f"⚠️ Failed to assign confidence scores: {e}")

    # === Patch derived fields before BigQuery write ===
    try:
        # High Limit flag
        df['High_Limit_Flag'] = (pd.to_numeric(df['Sharp_Limit_Total'], errors='coerce') >= 10000).astype(float) \
                                 if 'Sharp_Limit_Total' in df.columns else 0.0

        # Home team indicator
        if 'Home_Team_Norm' in df.columns and 'Outcome' in df.columns:
            df['Is_Home_Team_Bet'] = (df['Outcome'].astype(str).str.lower()
                                      == df['Home_Team_Norm'].astype(str).str.lower()).astype(float)
        else:
            df['Is_Home_Team_Bet'] = 0.0

        # Favorite indicator
        df['Is_Favorite_Bet'] = (pd.to_numeric(df['Value'], errors='coerce') < 0).astype(float)

        # Direction alignment: market-based only
        df['Direction_Aligned'] = np.where(
            df['Line_Delta'] > 0, 1,
            np.where(df['Line_Delta'] < 0, 0, np.nan)
        ).astype(float)
    except Exception as e:
        logging.warning(f"⚠️ Failed to compute sharp move diagnostic columns: {e}")
    # --- Restore: map lookup from bundles (do this BEFORE enrichment log) ---
    def _norm_market(m: str) -> str:
        
        m = (m or "").lower().strip()
        if m in ("spread", "spreads", "ats"): return "spreads"
        if m in ("total", "totals", "ou", "overunder"): return "totals"
        if m in ("h2h", "ml", "moneyline", "money_line"): return "h2h"
        return m
    
  
    
       
  
    # ---- determine which markets are present in df (robust; no iloc[0]) ----
    if "Market_norm" in df.columns and df["Market_norm"].notna().any():
        markets_present = (
            df["Market_norm"].astype(str).str.lower().str.strip().map(_norm_market).dropna().unique().tolist()
        )
    elif "Market" in df.columns and df["Market"].notna().any():
        markets_present = (
            df["Market"].astype(str).str.lower().str.strip().map(_norm_market).dropna().unique().tolist()
        )
    else:
        markets_present = []
    
    # ---- normalize trained_models into a dict keyed by market_norm ----
    trained_models_by_market = {}
    if isinstance(trained_models_norm, dict):
        for k, b in trained_models_norm.items():
            mk = _norm_market(k) if k else None
            if mk is None and isinstance(b, dict):
                meta = b.get("meta") or {}
                mk2 = meta.get("market") or b.get("market")
                mk = _norm_market(mk2) if mk2 else None
            if mk:
                trained_models_by_market[mk] = b
    
    # ---- load maps for ALL markets present (or fallback to ALL bundles) ----
    bundles = []
    if markets_present:
        for mk in markets_present:
            b = trained_models_by_market.get(mk)
            if isinstance(b, dict):
                bundles.append(b)
    else:
        # if df doesn't tell us the market yet, at least don't fail:
        bundles = [b for b in trained_models_by_market.values() if isinstance(b, dict)]
    
    # ---- pull maps, concat, and dedupe ----
    tfm_list = []
    brm_list = []
    
    for b in bundles:
        tfm = b.get("team_feature_map")
        if isinstance(tfm, pd.DataFrame) and not tfm.empty:
            tfm_list.append(tfm)
    
        brm = b.get("book_reliability_map")
        if isinstance(brm, pd.DataFrame) and not brm.empty:
            brm_list.append(brm)
    
    team_feature_map = pd.concat(tfm_list, ignore_index=True) if tfm_list else pd.DataFrame()
    book_reliability_map = pd.concat(brm_list, ignore_index=True) if brm_list else pd.DataFrame()
    
    # ---- normalize keys ONLY after load, then DEDUPE ----
    if not team_feature_map.empty:
        if "Team" in team_feature_map.columns:
            team_feature_map["Team"] = team_feature_map["Team"].astype(str).str.lower().str.strip()
        if "Sport" in team_feature_map.columns:
            team_feature_map["Sport"] = team_feature_map["Sport"].astype(str).str.upper().str.strip()
        if "Market" in team_feature_map.columns:
            team_feature_map["Market"] = (
                team_feature_map["Market"].astype(str).str.lower().str.strip().map(_norm_market)
            )
    
        # ✅ critical: prevent row explosion + prevent Market_x/Market_y cascades
        need_cols = [c for c in ["Sport", "Market", "Team"] if c in team_feature_map.columns]
        if len(need_cols) == 3:
            team_feature_map = team_feature_map.drop_duplicates(["Sport", "Market", "Team"], keep="last").copy()
        elif "Team" in team_feature_map.columns:
            team_feature_map = team_feature_map.drop_duplicates(["Team"], keep="last").copy()
    
    # book map: normalize lightly (dedupe optional, depends on its schema)
    if not book_reliability_map.empty:
        for col in ["Sport", "Market", "Bookmaker", "Bookmaker_Norm", "Book"]:
            if col in book_reliability_map.columns:
                if col == "Sport":
                    book_reliability_map[col] = book_reliability_map[col].astype(str).str.upper().str.strip()
                elif col == "Market":
                    book_reliability_map[col] = book_reliability_map[col].astype(str).str.lower().str.strip().map(_norm_market)
                else:
                    book_reliability_map[col] = book_reliability_map[col].astype(str).str.lower().str.strip()


      # === Cross-Market Pivots (Value + Odds) with guaranteed columns ===
    MARKETS = ["spreads","totals","h2h"]
    
    df_latest = (
        df.sort_values("Snapshot_Timestamp")
          .drop_duplicates(subset=["Game_Key","Market","Outcome"], keep="last")
    )
    
    value_pivot = (
        df_latest.pivot_table(index="Game_Key", columns="Market", values="Value",
                              aggfunc="last", dropna=False)
                 .reindex(columns=MARKETS)
                 .rename(columns={"spreads":"Spread_Value","totals":"Total_Value","h2h":"H2H_Value"})
    )
    
    odds_pivot = (
        df_latest.pivot_table(index="Game_Key", columns="Market", values="Odds_Price",
                              aggfunc="last", dropna=False)
                 .reindex(columns=MARKETS)
                 .rename(columns={"spreads":"Spread_Odds","totals":"Total_Odds","h2h":"H2H_Odds"})
    )
    
    df_cross_market = value_pivot.join(odds_pivot, how="outer").reset_index()
    
    # Ensure the expected columns exist even if a market is absent
    for col in ["Spread_Value","Total_Value","H2H_Value",
                "Spread_Odds","Total_Odds","H2H_Odds"]:
        if col not in df_cross_market.columns:
            df_cross_market[col] = np.nan
    
    # Merge into the working frame (adds *_Value / *_Odds)
    df = df.merge(df_cross_market, on="Game_Key", how="left", validate="m:1")

    # === GAME-LEVEL value-aware features: compute once per Game_Key, then broadcast ===
    try:
        game_base = (
            df[["Game_Key", "Sport"]]
            .drop_duplicates("Game_Key")
            .copy()
        )
    except KeyError:
        game_base = None

    if game_base is not None and not game_base.empty:
        # Attach spread / total value from cross-market pivot
        game_vals = game_base.merge(
            df_cross_market[["Game_Key", "Spread_Value", "Total_Value"]],
            on="Game_Key",
            how="left",
        )

        # Make sure these are numeric
        game_vals["Spread_Value"] = pd.to_numeric(game_vals["Spread_Value"], errors="coerce")
        game_vals["Total_Value"]  = pd.to_numeric(game_vals["Total_Value"],  errors="coerce")

        # Absolute spread magnitude (fav line) and total points for the game
        game_vals["Spread_Abs_Game"] = game_vals["Spread_Value"].abs()
        game_vals["Total_Game"]      = game_vals["Total_Value"]

        # Z-scores per sport (season-relative)
        game_vals["Spread_Abs_Game_Z"] = (
            game_vals.groupby("Sport")["Spread_Abs_Game"]
                     .transform(lambda x: zscore(x.fillna(0), ddof=0))
        )
        game_vals["Total_Game_Z"] = (
            game_vals.groupby("Sport")["Total_Game"]
                     .transform(lambda x: zscore(x.fillna(0), ddof=0))
        )

        # Spread size bucket (generic but meaningful)
        game_vals["Spread_Size_Bucket"] = pd.cut(
            game_vals["Spread_Abs_Game"],
            bins=[-0.01, 2.5, 6.5, 10.5, np.inf],
            labels=["Close", "Medium", "Large", "Huge"],
        )

      
        # Total size bucket: per-sport quantiles (Low / Medium / High / Very High)
        game_vals["Total_Size_Bucket"] = ""
        for sport_val, g in game_vals.groupby("Sport"):
            if g["Total_Game"].notna().sum() < 5:
                continue
        
            qs = g["Total_Game"].quantile([0.25, 0.50, 0.75]).values
            bins_raw = np.array([-np.inf, qs[0], qs[1], qs[2], np.inf], dtype="float64")
        
            # --- HARDEN: quantiles can tie -> duplicate bin edges ---
            bins = np.unique(bins_raw)  # keeps sorted order for numeric arrays
        
            # need at least 3 edges to form >=2 intervals
            if len(bins) < 3:
                # too flat / too many ties; leave as "" for this sport
                continue
        
            # labels must be exactly (#intervals) = (len(bins)-1)
            base_labels = ["Low", "Medium", "High", "Very_High"]
            labels = base_labels[: len(bins) - 1]
        
            mask = game_vals["Sport"].eq(sport_val)
            game_vals.loc[mask, "Total_Size_Bucket"] = pd.cut(
                game_vals.loc[mask, "Total_Game"],
                bins=bins,
                labels=labels,
                include_lowest=True,
            ).astype(str)


        # Interactions / volatility proxies
        game_vals["Spread_x_Total"] = game_vals["Spread_Abs_Game"] * game_vals["Total_Game"]
        game_vals["Spread_over_Total"] = game_vals["Spread_Abs_Game"] / game_vals["Total_Game"].replace(0, np.nan)
        game_vals["Total_over_Spread"] = game_vals["Total_Game"] / game_vals["Spread_Abs_Game"].replace(0, np.nan)

        # (Optional but 🔥 for NFL/NCAAF): key number distances
        def _dist_to_key(abs_spread, key):
            return np.abs(abs_spread - key)

        is_fb = game_vals["Sport"].isin(["NFL", "NCAAF"])
        if is_fb.any():
            g_fb = game_vals[is_fb]
            game_vals.loc[is_fb, "Dist_to_3"]  = _dist_to_key(g_fb["Spread_Abs_Game"], 3)
            game_vals.loc[is_fb, "Dist_to_7"]  = _dist_to_key(g_fb["Spread_Abs_Game"], 7)
            game_vals.loc[is_fb, "Dist_to_10"] = _dist_to_key(g_fb["Spread_Abs_Game"], 10)
        else:
            for k in ["Dist_to_3", "Dist_to_7", "Dist_to_10"]:
                game_vals[k] = np.nan

        # Merge these back to every scoring row for the game
        df = df.merge(
            game_vals[
                [
                    "Game_Key",
                    "Spread_Abs_Game",
                    "Spread_Abs_Game_Z",
                    "Spread_Size_Bucket",
                    "Total_Game",
                    "Total_Game_Z",
                    "Total_Size_Bucket",
                    "Spread_x_Total",
                    "Spread_over_Total",
                    "Total_over_Spread",
                    "Dist_to_3",
                    "Dist_to_7",
                    "Dist_to_10",
                ]
            ],
            on="Game_Key",
            how="left",
            validate="many_to_one",
        )
    else:
        # Ensure columns exist even if we couldn't build game_vals
        for c in [
            "Spread_Abs_Game","Spread_Abs_Game_Z","Spread_Size_Bucket",
            "Total_Game","Total_Game_Z","Total_Size_Bucket",
            "Spread_x_Total","Spread_over_Total","Total_over_Spread",
            "Dist_to_3","Dist_to_7","Dist_to_10",
        ]:
            if c not in df.columns:
                df[c] = np.nan

    # Buckets as plain strings (avoid category setitem headaches)
    for c in ["Spread_Size_Bucket", "Total_Size_Bucket"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("")

 
    # === Implied probabilities per market (vectorized, with row-level fallback) ===
    def _series_or_fallback(primary, fallback):
        return primary.where(primary.notna(), fallback) if fallback is not None else primary
    
    if 'Odds_Price' in df.columns:
        row_odds = pd.to_numeric(df['Odds_Price'], errors='coerce')
    else:
        row_odds = None
    
    for odds_col, ip_col in [
        ("Spread_Odds","Spread_Implied_Prob"),
        ("Total_Odds","Total_Implied_Prob"),
        ("H2H_Odds","H2H_Implied_Prob"),
    ]:
        if odds_col not in df.columns:
            df[odds_col] = np.nan
        ser = pd.to_numeric(df[odds_col], errors='coerce')
        ser = _series_or_fallback(ser, row_odds)
        df[ip_col] = implied_prob_vec(ser).astype("float32")
    
    # === New plumbing features ===
    # 1) Limit dynamics (spikes, Δlimit, spike×no-move)
    df = add_limit_dynamics_features(df)
    
    # 2) Bookmaker network features (weighted sharp consensus & sharp-vs-rec gaps)
    #    Call AFTER you merged book reliability above (you already merged 'book_reliability_map' earlier)
    df = add_book_network_features(df)
    
    # 3) Internal consistency checks (Spread↔ML and Total↔Side) using cross-market pivots
    sport0 = (df['Sport'].dropna().astype(str).str.upper().iloc[0] if df['Sport'].notna().any() else "NFL")
    df = add_internal_consistency_features(df, df_cross_market=df_cross_market, sport_default=sport0)
    # === Add Sharp-vs-Rec EV features (prob edge, $EV, Kelly) ===
    try:
        # Make sure Outcome_Norm exists (you standardize it later too; here is fine)
        if 'Outcome_Norm' not in df.columns:
            df['Outcome_Norm'] = df.get('Outcome', pd.Series(index=df.index, dtype='object')) \
                                     .astype('string').str.lower().str.strip()
    
        # Compute EV features (safe if sharp rows missing; returns same rowcount)
        df = compute_ev_features_sharp_vs_rec(
            df_market=df,
            sharp_books=SHARP_BOOKS,                 # you already define this set
            reliability_col="Book_Reliability_Score",
            limit_col="Sharp_Limit_Total",
            sigma_col="Sigma_Pts",                   # optional; function has defaults/fallbacks
        )
    
        # (Optional) quick dtype tightening of the three main outputs
        for c in ("EV_Sh_vs_Rec_Prob", "EV_Sh_vs_Rec_Dollar", "Kelly_Fraction"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    
        logger.info("✅ EV features added: %s",
                    [c for c in ["Truth_Fair_Prob_at_SharpLine","Truth_Margin_Mu","Truth_Sigma",
                                 "Truth_Fair_Prob_at_RecLine","Rec_Implied_Prob",
                                 "EV_Sh_vs_Rec_Prob","EV_Sh_vs_Rec_Dollar","Kelly_Fraction"]
                     if c in df.columns])
    except Exception as e:
        logger.warning("⚠️ EV feature block skipped: %s", e, exc_info=True)

    # Optional extras if/when you have the inputs handy:
    # if df_alt is available:
    # df = add_altline_curvature(df, df_alt)
    # if you train a tiny CLV side-model:
    # df = add_clv_proxy_features(df, clv_model_path="artifacts/clv_side.joblib")


    # === Normalize keys for joins (safe no-ops if already normalized)
    if 'Bookmaker' not in df.columns and 'Book' in df.columns:
        df['Bookmaker'] = df['Book']
    df['Sport'] = df['Sport'].astype(str).str.upper()
    df['Market'] = df['Market'].astype(str).str.lower().str.strip()
    df['Bookmaker'] = df['Bookmaker'].astype(str).str.lower().str.strip()

    # === Team features (log-only here; merge per-market later)
    if team_feature_map is not None and hasattr(team_feature_map, 'empty') and not team_feature_map.empty:
        logger.info("📊 Team Historical Performance Metrics (Hit Rate and Avg Model Prob):")
        try:
            logger.info(f"\n{team_feature_map.head(40).to_string(index=False)}")
        except Exception:
            logger.info("team_feature_map present (head print failed).")
    else:
        logger.warning("⚠️ team_feature_map is empty or missing.")

    # === Book reliability features (merge on full df as you had)
    if book_reliability_map is not None and hasattr(book_reliability_map, 'empty') and not book_reliability_map.empty:
        bm = book_reliability_map.copy()
        bm['Sport'] = bm['Sport'].astype(str).str.upper()
        bm['Market'] = bm['Market'].astype(str).str.lower().str.strip()
        # Allow maps that used 'Book'
        if 'Bookmaker' not in bm.columns and 'Book' in bm.columns:
            bm = bm.rename(columns={'Book': 'Bookmaker'})
        bm['Bookmaker'] = bm['Bookmaker'].astype(str).str.lower().str.strip()

        logger.info("📊 Bookmaker Reliability Metrics:")
        try:
            logger.info(f"\n{bm.head(40).to_string(index=False)}")
        except Exception:
            logger.info("book_reliability_map present (head print failed).")

        df = df.merge(
            bm[['Sport','Market','Bookmaker','Book_Reliability_Score','Book_Reliability_Lift']],
            on=['Sport','Market','Bookmaker'],
            how='left',
            validate='m:1'
        )
    else:
        logger.warning("⚠️ book_reliability_map is empty or missing.")
        df['Book_Reliability_Score'] = np.nan
        df['Book_Reliability_Lift'] = np.nan

    # === Now loop markets just for scoring (robust to missing models) ===

    # After base enrichment (Implied_Prob, Open_* fills, odds pivot, reliability merge, etc.)
    # === Now loop markets just for scoring (robust to missing models) ===

    # After base enrichment (Implied_Prob, Open_* fills, odds pivot, reliability merge, etc.)
    df = compute_small_book_liquidity_features(df)

    _suffix_snapshot(df, "after compute_small_book_liquidity_features")

    df = add_line_and_crossmarket_features(df)
    _suffix_snapshot(df, "after add_line_and_crossmarket_features")
    # Ensure Outcome_Norm once
    if 'Outcome_Norm' not in df.columns:
        df['Outcome_Norm'] = df.get('Outcome', pd.Series(index=df.index, dtype='object')) \
                                .astype('string').str.lower().str.strip()
    
    # -------- Spreads-only enrichment (skinny slice, parity backend) --------
    mask_spreads = df['Market'].astype(str).str.lower().eq('spreads')
    
    if mask_spreads.any():
        need = ['Sport','Home_Team_Norm','Away_Team_Norm','Outcome_Norm','Value','Game_Start']
        df_sp = df.loc[mask_spreads, need].copy()
    
        # 🔑 keep original row positions so we can realign later
        df_sp['__row__'] = df_sp.index
    
        # Normalize (slice-only)
        df_sp['Sport']           = df_sp['Sport'].astype(str).str.upper().str.strip()
        df_sp['Home_Team_Norm']  = df_sp['Home_Team_Norm'].astype(str).str.lower().str.strip()
        df_sp['Away_Team_Norm']  = df_sp['Away_Team_Norm'].astype(str).str.lower().str.strip()
        df_sp['Outcome_Norm']    = df_sp['Outcome_Norm'].astype(str).str.lower().str.strip()
        df_sp['Value']           = pd.to_numeric(df_sp['Value'], errors='coerce').astype('float32')
    
        # ⬇️ Enrich (produces all spread features)
        df_sp_enriched = enrich_and_grade_for_training(
            df_sp,
            sport_aliases=SPORT_ALIASES,
            value_col="Value",
            outcome_col="Outcome_Norm",
            table_current="sharplogger.sharp_data.ratings_current",
            project="sharplogger",
        )
    
        # Ensure original row id on enriched frame
        if '__row__' not in df_sp_enriched.columns:
            df_sp_enriched = df_sp[['__row__','Sport','Home_Team_Norm','Away_Team_Norm','Outcome_Norm']].merge(
                df_sp_enriched,
                on=['Sport','Home_Team_Norm','Away_Team_Norm','Outcome_Norm'],
                how='left',
                copy=False
            )
    
        # Index to original rows
        df_sp_enriched = (
            df_sp_enriched.drop_duplicates(subset='__row__', keep='last')
                          .set_index('__row__')
        )
    
        # 🚫 Do not write back the base slice cols; only NEW features
        base_cols = set(need) | {'__row__'}
        write_cols = [c for c in df_sp_enriched.columns if c not in base_cols]
    
        # Create any missing columns on df with correct dtypes
        for c in write_cols:
            if c not in df.columns:
                s = df_sp_enriched[c]
                if pd.api.types.is_float_dtype(s):
                    df[c] = pd.Series(np.nan, index=df.index, dtype='float32')
                elif pd.api.types.is_integer_dtype(s):
                    df[c] = pd.Series(pd.NA, index=df.index, dtype='Int32')
                else:
                    df[c] = pd.Series(pd.NA, index=df.index, dtype='string')
    
        # ✅ Assign by exact index (no boolean mask)
        target_idx = df_sp_enriched.index
        # Use update semantics to be extra safe with mixed dtypes
        df.update(df_sp_enriched[write_cols], overwrite=True)
    
        # cleanup
        del df_sp_enriched, df_sp
        # import gc; gc.collect()
    
 
    # ---- Ensure Outcome_Norm first ----
    if 'Outcome_Norm' not in df.columns:
        if 'Outcome' in df.columns and not df.empty:
            df['Outcome_Norm'] = df['Outcome'].astype('string').str.lower().str.strip()
        else:
            df['Outcome_Norm'] = pd.Series(pd.NA, index=df.index, dtype='string')
    
    # ---- Ensure Market_norm (canonical to spreads/totals/h2h) BEFORE team merge ----
    MARKET_KEY_MAP = {
        "spread": "spreads", "spreads": "spreads", "ats": "spreads",
        "total": "totals",   "totals": "totals",   "ou": "totals", "overunder": "totals",
        "moneyline": "h2h",  "ml": "h2h", "h2h": "h2h", "money_line": "h2h",
    }
    if 'Market' in df.columns and not df.empty:
        m_raw = df['Market'].astype('string').fillna("").str.lower().str.strip()
        df['Market_norm'] = m_raw.map(lambda s: MARKET_KEY_MAP.get(s, s)).astype('category')
    else:
        df['Market_norm'] = pd.Series(pd.NA, index=df.index, dtype='category')
    
    # ===== Team features (SAFE merge; cannot wipe Sport/Market) =====
    if team_feature_map is not None and hasattr(team_feature_map, 'empty') and not team_feature_map.empty:
    
        tfm = team_feature_map.copy()
    
        # normalize map keys
        if 'Sport' in tfm.columns:
            tfm['Sport'] = tfm['Sport'].astype(str).str.upper().str.strip()
        if 'Market' in tfm.columns:
            tfm['Market'] = tfm['Market'].astype(str).str.lower().str.strip()
        if 'Team' in tfm.columns:
            tfm['Team'] = tfm['Team'].astype(str).str.lower().str.strip()
    
        # must be unique on join keys to avoid row explosion
        key_cols = [c for c in ['Sport', 'Market', 'Team'] if c in tfm.columns]
        if key_cols:
            tfm = tfm.drop_duplicates(subset=key_cols, keep='last')
        else:
            tfm = tfm.drop_duplicates(subset=['Team'], keep='last')
    
        # only apply to team-like markets (spreads + h2h). totals outcomes are over/under.
        mask_teamlike = df['Market_norm'].astype(str).isin(['spreads', 'h2h'])
    
        if mask_teamlike.any():
            cols = ['Sport', 'Market_norm', 'Outcome_Norm']
            if 'Team_For_Join' in df.columns:
                cols.append('Team_For_Join')
            df_team = df.loc[mask_teamlike, cols].copy()
    
            # build Team join key (prefer Team_For_Join; fallback Outcome_Norm)
            if 'Team_For_Join' in df_team.columns:
                df_team['Team'] = df_team['Team_For_Join'].astype(str).str.lower().str.strip()
            else:
                df_team['Team'] = df_team['Outcome_Norm'].astype(str).str.lower().str.strip()
    
            # normalize df_team join keys
            df_team['Sport'] = df_team['Sport'].astype(str).str.upper().str.strip()
            df_team['Market'] = df_team['Market_norm'].astype(str).str.lower().str.strip()
    
            # choose join keys based on what exists in tfm
            join_left = []
            join_right = []
            if 'Sport' in tfm.columns:
                join_left.append('Sport');  join_right.append('Sport')
            if 'Market' in tfm.columns:
                join_left.append('Market'); join_right.append('Market')
            join_left.append('Team');      join_right.append('Team')
    
            merged = df_team.merge(
                tfm,
                left_on=join_left,
                right_on=join_right,
                how='left',
                validate='m:1'   # 🔥 prevents the blow-up
            )
    
            # write back ONLY non-protected columns
            protected = {'Sport', 'Market', 'Market_norm', 'Outcome_Norm', 'Team', 'Team_For_Join'}
            fill_cols = [c for c in tfm.columns if c not in protected]  # only map-provided cols
    
            for c in fill_cols:
                if c not in df.columns:
                    df[c] = pd.NA
                df.loc[mask_teamlike, c] = merged[c].values
    
    # ---- NOW determine markets_present ----
    if HAS_MODELS and not df.empty:
        markets_present = [
            mk for mk in df['Market_norm'].dropna().astype(str).unique().tolist()
            if mk in model_markets_lower
        ]
    else:
        markets_present = []

        markets_present = []
    _suffix_snapshot(df, "before canon ")
    # ---------- NO-MODEL / NO-ELIGIBLE-MARKETS EARLY EXIT ----------
    # ======================= MARKET RESOLUTION + SCORING LOOP =======================
    # Canonical market mapping
  
    MARKET_KEY_MAP = {
        "spread": "spreads", "spreads": "spreads",
        "total": "totals",   "totals": "totals",
        "moneyline": "h2h",  "ml": "h2h", "h2h": "h2h",
    }

    # Normalize the incoming trained_models without discarding tuples/objects

    # Pull ~3.3 years of scores, or set days_back=None for all-time
    df_scores_hist = load_scores_history_cached_backend(
        
        days_back=365,   # ← no seasons, pure time window
        table_fq="sharplogger.sharp_data.game_scores_final",
        ttl_seconds=3600
    )
    
    # Then enrich your scoring frame (your season-free functions already handle the rest)
    df = enrich_df_with_totals_features(
        df_scoring=df,
        df_scores_history=df_scores_hist,
        
        key_col="Merge_Key_Short",
        window_games=30,
        shrink=0.30,
        compute_mispricing=True
    )

    df = wire_ats_features_inplace(df)      

  
    # Normalize DF markets ONCE (before building markets_present)
    if 'Market' in df.columns and not df.empty:
        m_raw = df['Market'].astype('string').str.lower().str.strip()
        df['Market_norm'] = m_raw.map(lambda s: MARKET_KEY_MAP.get(s, s)).astype('category')
    else:
        df['Market_norm'] = pd.Series(pd.NA, index=df.index, dtype='category')

    # Determine which markets are present AND have models
    if HAS_MODELS and not df.empty:
        markets_present = [
            mk for mk in df['Market_norm'].dropna().unique().tolist()
            if mk in trained_models_norm
        ]
    else:
        markets_present = []

    _suffix_snapshot(df, "before canon")

    # ---------- NO-MODEL / NO-ELIGIBLE-MARKETS EARLY EXIT ----------
    if (not HAS_MODELS) or (len(markets_present) == 0):
        logger.info("ℹ️ No trained models / no eligible markets present; returning minimally-enriched, unscored snapshots.")
        base = df.copy()

        # Stamp prediction columns
        for col, default in [
            ('Model_Sharp_Win_Prob', np.nan),
            ('Model_Confidence',     np.nan),
            ('Scored_By_Model',      False),
            ('Scoring_Market',       base.get('Market_norm', pd.Series(index=base.index, dtype='object'))),
            ('Was_Canonical',        pd.NA),
        ]:
            if col not in base.columns:
                base[col] = default

        # Friendly tier for unscored rows
        if 'Model_Confidence_Tier' not in base.columns:
            base['Model_Confidence_Tier'] = pd.Categorical(
                values=['❔ No Model'] * len(base),
                categories=["❔ No Model", "✅ Low", "⭐ Lean", "🔥 Strong Indication", "🔥 Steam"]
            )
        else:
            base['Model_Confidence_Tier'] = '❔ No Model'

        # Commence hour (needed for Team_Key)
        if 'Commence_Hour' not in base.columns:
            base['Commence_Hour'] = pd.to_datetime(
                base.get('Game_Start', pd.NaT), utc=True, errors='coerce'
            ).dt.floor('h')

        # Latest snapshot per (Game, Market, Outcome, Bookmaker)
        essential = {'Snapshot_Timestamp','Game_Key','Market','Outcome','Bookmaker'}
        if essential.issubset(base.columns):
            base = (
                base.sort_values('Snapshot_Timestamp')
                    .drop_duplicates(subset=['Game_Key','Market','Outcome','Bookmaker'], keep='last')
            )

        # Team_Key (no np.char.*)
        for c, default in [
            ('Home_Team_Norm',''), ('Away_Team_Norm',''), ('Market',''),
            ('Outcome_Norm',''),   ('Outcome','')
        ]:
            if c not in base.columns:
                base[c] = default

        on = base['Outcome_Norm'].astype('string').fillna(base['Outcome'].astype('string').fillna(''))
        ht = base['Home_Team_Norm'].astype('string').fillna('')
        at = base['Away_Team_Norm'].astype('string').fillna('')
        ch = base['Commence_Hour'].astype('string').fillna('')
        mk = base['Market'].astype('string').fillna('')

        base['Team_Key'] = ht.str.cat([at, ch, mk, on], sep='_')

        logger.info("✅ Returning %d unscored rows (no models).", len(base))
        return base

    # ------------------------ 5) SCORING LOOP ------------------------
    for mkt in markets_present:
        mask = (df['Market_norm'] == mkt)
        df_m = df.loc[mask].copy()

        # ----- resolve the bundle for this market (dict / tuple / object) -----
        bundle = trained_models_norm.get(mkt)
        model = iso = None
        if isinstance(bundle, dict):
            # Prefer explicit logloss/auc pair; else generic model/calibrator
            model = bundle.get("model_logloss") or bundle.get("model_auc") or bundle.get("model")
            iso   = bundle.get("calibrator_logloss") or bundle.get("calibrator_auc") or bundle.get("calibrator")
        elif isinstance(bundle, (tuple, list)):
            model = bundle[0] if len(bundle) > 0 else None
            iso   = bundle[1] if len(bundle) > 1 else None
        else:
            model = bundle  # bare model or calibrated estimator

        if not _has_any_model(bundle):
            logger.warning("⚠️ No usable model bundle for %s; available=%s",
                           mkt, list(trained_models_norm.keys()))
            # stamp placeholders on all rows in this market
            for col, default in [
                ('Model_Sharp_Win_Prob', np.nan),
                ('Model_Confidence',     np.nan),
                ('Scored_By_Model',      False),
                ('Scoring_Market',       mkt),
            ]:
                if col not in df.columns:
                    df[col] = default
                else:
                    df.loc[mask, col] = default
            continue

        # ===== Build per-market frame & canonical split =====
        df_m['Outcome'] = df_m['Outcome'].astype(str).str.lower().str.strip()
        df_m['Outcome_Norm'] = df_m['Outcome']
        df_m['Value'] = pd.to_numeric(df_m['Value'], errors='coerce')
        df_m['Commence_Hour'] = pd.to_datetime(df_m['Game_Start'], utc=True, errors='coerce').dt.floor('h')

        if 'Odds_Price' in df_m.columns:
            df_m['Odds_Price'] = pd.to_numeric(df_m['Odds_Price'], errors='coerce')
        else:
            df_m['Odds_Price'] = np.nan

        df_m['Implied_Prob'] = df_m['Odds_Price'].apply(implied_prob)

        sport_str = (str(df_m['Sport'].mode(dropna=True).iloc[0]).upper()
                     if 'Sport' in df_m.columns and not df_m['Sport'].isna().all()
                     else "GENERIC")
        df_m = add_time_context_flags(df_m, sport=sport_str)

        for c in ['Home_Team_Norm','Away_Team_Norm','Market']:
            if c not in df_m.columns:
                df_m[c] = ""

        df_m['Game_Key'] = (
            df_m['Home_Team_Norm'] + "_" +
            df_m['Away_Team_Norm'] + "_" +
            df_m['Commence_Hour'].astype(str) + "_" +
            df_m['Market'] + "_" +
            df_m['Outcome_Norm']
        )
        df_m['Game_Key_Base'] = (
            df_m['Home_Team_Norm'] + "_" +
            df_m['Away_Team_Norm'] + "_" +
            df_m['Commence_Hour'].astype(str) + "_" +
            df_m['Market']
        )

        sided_games_check = (
            df_m.groupby(['Game_Key_Base'])['Outcome']
                .nunique()
                .reset_index(name='Num_Sides')
        )
        valid_games = sided_games_check[sided_games_check['Num_Sides'] >= 2]['Game_Key_Base']
        df_m = df_m[df_m['Game_Key_Base'].isin(valid_games)].copy()

        if df_m.empty:
            logger.info(f"ℹ️ After 2-side filter, no rows for {mkt.upper()} — stamping placeholders.")
            for col, default in [
                ('Model_Sharp_Win_Prob', np.nan),
                ('Model_Confidence',     np.nan),
                ('Scored_By_Model',      False),
                ('Scoring_Market',       mkt),
            ]:
                if col not in df.columns:
                    df[col] = default
                else:
                    df.loc[mask, col] = default
            continue
        _key = ['Game_Key','Market','Outcome','Bookmaker']

        # If Implied_Prob is needed for H2H canon rule, ensure it's present pre-dedup (cheap)
        if 'Implied_Prob' not in df_m.columns and 'Odds_Price' in df_m.columns:
            df_m['Implied_Prob'] = implied_prob_vec(df_m['Odds_Price'])
        
        if 'Snapshot_Timestamp' in df_m.columns:
            df_m = (df_m.sort_values('Snapshot_Timestamp')
                         .drop_duplicates(subset=_key, keep='last'))
        else:
            df_m = df_m.drop_duplicates(subset=_key, keep='last')
        
        # (Optional) quick diagnostics
        logger.info("🧮 %s dedup: rows=%d, unique keys=%d",
            mkt.upper(), len(df_m), df_m[_key].drop_duplicates().shape[0])
        # ---------- canonical selection ----------
        if mkt == "spreads":
            df_m = df_m[df_m['Value'].notna()]
            df_canon = df_m[df_m['Value'] < 0].copy()
            df_full_market_m = df_m.copy()
        elif mkt == "h2h":
            df_m = df_m[df_m['Odds_Price'].notna()]
            df_canon = df_m[(df_m['Odds_Price'] < 0) | (df_m['Implied_Prob'] > 0.5)].copy()
            df_full_market_m = df_m.copy()
        elif mkt == "totals":
            df_canon = df_m[df_m['Outcome_Norm'] == 'over'].copy()
            df_full_market_m = df_m.copy()
        else:
            df_canon = df_m.copy()
            df_full_market_m = df_m.copy()

        df_canon = df_canon.drop_duplicates(subset=['Game_Key','Market','Bookmaker','Outcome'])

        # ---------- feature guards ----------
        if 'Line_Delta' not in df_canon.columns: df_canon['Line_Delta'] = 0.0
        if 'Is_Sharp_Book' not in df_canon.columns: df_canon['Is_Sharp_Book'] = 0

        df_canon['Line_Move_Magnitude'] = pd.to_numeric(df_canon['Line_Delta'], errors='coerce').abs()
        df_canon['Line_Magnitude_Abs'] = df_canon['Line_Move_Magnitude']

        df_canon['Sharp_Line_Delta'] = np.where(df_canon['Is_Sharp_Book'] == 1, df_canon['Line_Delta'], 0.0)
        df_canon['Rec_Line_Delta']   = np.where(df_canon['Is_Sharp_Book'] == 0, df_canon['Line_Delta'], 0.0)
        df_canon['Sharp_Line_Magnitude'] = df_canon['Sharp_Line_Delta'].abs()
        df_canon['Rec_Line_Magnitude']   = df_canon['Rec_Line_Delta'].abs()
        # Odds → implied prob
        if 'Odds_Price' in df_canon.columns:
            df_canon['Odds_Price'] = pd.to_numeric(df_canon['Odds_Price'], errors='coerce')
        else:
            df_canon['Odds_Price'] = np.nan
        df_canon['Implied_Prob'] = df_canon['Odds_Price'].apply(implied_prob)
        
        # --- FIX: use Series default, not scalar ---
        limits = pd.to_numeric(
            df_canon.get('Sharp_Limit_Total', pd.Series(np.nan, index=df_canon.index)),
            errors='coerce'
        )
        df_canon['High_Limit_Flag'] = (limits >= 10000).astype('int8')
        
        home_norm = df_canon.get('Home_Team_Norm', pd.Series('', index=df_canon.index))
        df_canon['Is_Home_Team_Bet'] = (df_canon['Outcome'] == home_norm).astype('int8')

     
        df_canon['Is_Favorite_Bet']  = (pd.to_numeric(df_canon['Value'], errors='coerce') < 0).astype(int)

        if 'Odds_Shift' in df_canon.columns and 'Sharp_Move_Signal' in df_canon.columns:
            df_canon['SharpMove_OddsShift'] = df_canon['Sharp_Move_Signal'] * df_canon['Odds_Shift']
        if 'Implied_Prob_Shift' in df_canon.columns and 'Market_Leader' in df_canon.columns:
            df_canon['MarketLeader_ImpProbShift'] = df_canon['Market_Leader'] * df_canon['Implied_Prob_Shift']

        df_canon['SharpLimit_SharpBook']   = df_canon['Is_Sharp_Book'] * df_canon.get('Sharp_Limit_Total', 0)
        df_canon['LimitProtect_SharpMag']  = df_canon.get('LimitUp_NoMove_Flag', 0) * df_canon['Sharp_Line_Magnitude']
        df_canon['HomeRecLineMag']         = df_canon.get('Is_Home_Team_Bet', 0)   * df_canon['Rec_Line_Magnitude']

        df_canon['Delta_Sharp_vs_Rec'] = df_canon['Sharp_Line_Delta'] - df_canon['Rec_Line_Delta']
        df_canon['Sharp_Leads'] = (df_canon['Sharp_Line_Magnitude'] > df_canon['Rec_Line_Magnitude']).astype(int)
        df_canon['Same_Direction_Move'] = (
            (np.sign(df_canon['Sharp_Line_Delta']) == np.sign(df_canon['Rec_Line_Delta'])) &
            (df_canon['Sharp_Line_Delta'].abs() > 0) & (df_canon['Rec_Line_Delta'].abs() > 0)
        ).astype(int)
        df_canon['Opposite_Direction_Move'] = (
            (np.sign(df_canon['Sharp_Line_Delta']) != np.sign(df_canon['Rec_Line_Delta'])) &
            (df_canon['Sharp_Line_Delta'].abs() > 0) & (df_canon['Rec_Line_Delta'].abs() > 0)
        ).astype(int)
        df_canon['Sharp_Move_No_Rec'] = ((df_canon['Sharp_Line_Delta'].abs() > 0) & (df_canon['Rec_Line_Delta'].abs() == 0)).astype(int)
        df_canon['Rec_Move_No_Sharp'] = ((df_canon['Rec_Line_Delta'].abs() > 0) & (df_canon['Sharp_Line_Delta'].abs() == 0)).astype(int)

        for col in ['Open_Value','Open_Odds']:
            if col not in df_canon.columns:
                df_canon[col] = np.nan

        # --- helpers (vectorized American-odds -> prob) ---
        val_now   = pd.to_numeric(df_canon['Value'],      errors='coerce').astype('float64')
        val_open  = pd.to_numeric(df_canon['Open_Value'], errors='coerce').astype('float64')
        odds_now  = pd.to_numeric(df_canon['Odds_Price'], errors='coerce').astype('float64')
        odds_open = pd.to_numeric(df_canon['Open_Odds'],  errors='coerce').astype('float64')

        # --- line moves ---
        net_line = val_now - val_open
        df_canon['Net_Line_Move_From_Opening'] = net_line
        df_canon['Abs_Line_Move_From_Opening'] = np.abs(net_line)

        # --- odds moves (vectorized; avoid .apply) ---
        imp_now  = implied_prob_vec_raw(odds_now.values)
        imp_open = implied_prob_vec_raw(odds_open.values)
        net_odds = (imp_now - imp_open) * 100.0
        df_canon['Net_Odds_Move_From_Opening'] = net_odds
        df_canon['Abs_Odds_Move_From_Opening'] = np.abs(net_odds)

        # --- timing ---
        ts_game   = pd.to_datetime(df_canon['Game_Start'],           utc=True, errors='coerce')
        ts_snap   = pd.to_datetime(df_canon['Snapshot_Timestamp'],   utc=True, errors='coerce')
        mins_to   = (ts_game - ts_snap).dt.total_seconds() / 60.0
        df_canon['Minutes_To_Game'] = mins_to
        df_canon['Late_Game_Steam_Flag'] = (mins_to <= 60).astype('int8')
        df_canon['Minutes_To_Game_Tier'] = pd.cut(
            mins_to,
            bins=[-1, 30, 60, 180, 360, 720, np.inf],
            labels=['🚨 ≤30m','🔥 ≤1h','⚠️ ≤3h','⏳ ≤6h','📅 ≤12h','🕓 >12h']
        )

        if 'Value_Reversal_Flag' not in df_canon.columns:
            df_canon['Value_Reversal_Flag'] = 0
        else:
            df_canon['Value_Reversal_Flag'] = pd.to_numeric(df_canon['Value_Reversal_Flag'], errors='coerce').fillna(0).astype(int)
        if 'Odds_Reversal_Flag' not in df_canon.columns:
            df_canon['Odds_Reversal_Flag'] = 0
        else:
            df_canon['Odds_Reversal_Flag'] = pd.to_numeric(df_canon['Odds_Reversal_Flag'], errors='coerce').fillna(0).astype(int)

        df_canon['Market_Implied_Prob'] = df_canon['Odds_Price'].apply(implied_prob)

        # Team map fallbacks
        for col in [
            'Team_Past_Avg_Model_Prob',
            'Team_Past_Avg_Model_Prob_Home',
            'Team_Past_Avg_Model_Prob_Away'
        ]:
            if col not in df_canon.columns:
                df_canon[col] = 0.0

        df_canon['Market_Mispricing'] = df_canon['Team_Past_Avg_Model_Prob'] - df_canon['Market_Implied_Prob']
        df_canon['Abs_Market_Mispricing'] = df_canon['Market_Mispricing'].abs()
        df_canon['Team_Implied_Prob_Gap_Home'] = df_canon['Team_Past_Avg_Model_Prob_Home'] - df_canon['Market_Implied_Prob']
        df_canon['Team_Implied_Prob_Gap_Away'] = df_canon['Team_Past_Avg_Model_Prob_Away'] - df_canon['Market_Implied_Prob']
        df_canon['Abs_Team_Implied_Prob_Gap'] = np.where(
            df_canon['Is_Home_Team_Bet'] == 1,
            df_canon['Team_Implied_Prob_Gap_Home'].abs(),
            df_canon['Team_Implied_Prob_Gap_Away'].abs()
        )
        df_canon['Team_Mispriced_Flag'] = (df_canon['Abs_Team_Implied_Prob_Gap'] > 0.05).astype(int)

        # Cross-market alignment (guard for missing pivot cols)
        if 'Spread_Odds' in df_canon.columns:
            df_canon['Spread_Implied_Prob'] = df_canon['Spread_Odds'].apply(implied_prob)
        else:
            df_canon['Spread_Implied_Prob'] = np.nan
        if 'H2H_Odds' in df_canon.columns:
            df_canon['H2H_Implied_Prob']    = df_canon['H2H_Odds'].apply(implied_prob)
        else:
            df_canon['H2H_Implied_Prob'] = np.nan
        if 'Total_Odds' in df_canon.columns:
            df_canon['Total_Implied_Prob']  = df_canon['Total_Odds'].apply(implied_prob)
        else:
            df_canon['Total_Implied_Prob'] = np.nan

        df_canon['Spread_vs_H2H_Aligned'] = ((df_canon['Value'] < 0) & (df_canon['H2H_Implied_Prob'] > 0.5)).astype(int)
        df_canon['Total_vs_Spread_Contradiction'] = (
            (df_canon['Spread_Implied_Prob'] > 0.55) & (df_canon['Total_Implied_Prob'] < 0.48)
        ).astype(int)
        df_canon['Spread_vs_H2H_ProbGap'] = df_canon['Spread_Implied_Prob'] - df_canon['H2H_Implied_Prob']
        df_canon['Total_vs_H2H_ProbGap']  = df_canon['Total_Implied_Prob'] - df_canon['H2H_Implied_Prob']
        df_canon['Total_vs_Spread_ProbGap'] = df_canon['Total_Implied_Prob'] - df_canon['Spread_Implied_Prob']

        # Reliability interactions (guarantee base cols)
        for col in ['Book_Reliability_Score','Book_Reliability_Lift']:
            if col not in df_canon.columns:
                df_canon[col] = 0.0
        df_canon['Book_Reliability_x_Sharp']      = df_canon['Book_Reliability_Score'] * df_canon['Is_Sharp_Book']
        df_canon['Book_Reliability_x_Magnitude']  = df_canon['Book_Reliability_Score'] * df_canon['Sharp_Line_Magnitude']
        df_canon['Book_Reliability_x_PROB_SHIFT'] = df_canon['Book_Reliability_Score'] * df_canon['Is_Sharp_Book'] * df_canon['Implied_Prob_Shift']
        df_canon['Book_lift_x_Sharp']             = df_canon['Book_Reliability_Lift'] * df_canon['Is_Sharp_Book']
        df_canon['Book_lift_x_Magnitude']         = df_canon['Book_Reliability_Lift'] * df_canon['Sharp_Line_Magnitude']
        df_canon['Book_lift_x_PROB_SHIFT']        = df_canon['Book_Reliability_Lift'] * df_canon['Is_Sharp_Book'] * df_canon['Implied_Prob_Shift']
        df_canon['CrossMarket_Prob_Gap_Exists']   = (
            (df_canon['Spread_vs_H2H_ProbGap'].abs() > 0.05) | (df_canon['Total_vs_Spread_ProbGap'].abs() > 0.05)
        ).astype(int)
        # ---- derive timing feature column lists robustly ----
       
   
        try:
            # ===== MODEL SCORING =====
            feature_cols = _resolve_feature_cols_like_training(bundle, model, df_canon)
            # stable de-dup, keep order, drop Nones
            feature_cols = [str(c) for c in dict.fromkeys(feature_cols) if c is not None]
            logger.info("🔧 %s: using %d feature cols", mkt.upper(), len(feature_cols))
        
            # ---- ensure every feature exists & is numeric
            # ---- ensure every feature exists & is numeric (handles category dtypes)
            for c in feature_cols:
                if c not in df_canon.columns:
                    df_canon[c] = 0.0
                    continue
            
                s = df_canon[c]
            
                # If categorical, convert to object/string first (prevents setitem-on-categorical errors)
                if pd.api.types.is_categorical_dtype(s):
                    df_canon[c] = s.astype('string')  # or .astype(object)
                    s = df_canon[c]
            
                # If string/object, normalize booleans/empties then coerce numeric
                if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                    df_canon[c] = s.replace({
                        'True': 1, 'False': 0, True: 1, False: 0,
                        '': np.nan, 'none': np.nan, 'None': np.nan
                    })
            
                # Finally: numeric, finite, fill
                df_canon[c] = (pd.to_numeric(df_canon[c], errors='coerce')
                                 .replace([np.inf, -np.inf], np.nan)
                                 .fillna(0.0)
                                 .astype('float32'))
            
                    
            X_can = (df_canon.reindex(columns=feature_cols)
                 .apply(pd.to_numeric, errors='coerce')
                 .replace([np.inf, -np.inf], np.nan)
                 .fillna(0.0)
                 .astype('float32'))


            # ---- ensure prediction cols exist on BOTH frames before writeback
            cols_to_write_defaults = {
                'Model_Sharp_Win_Prob': np.nan,
                'Model_Confidence':     np.nan,
                'Scored_By_Model':      False,
                'Scoring_Market':       mkt,
            }
            for col, default in cols_to_write_defaults.items():
                if col not in df_canon.columns:
                    df_canon[col] = default
                if col not in df.columns:
                    df[col] = default
        
            # ---- predict or stamp unscored
            if not feature_cols or X_can.shape[1] == 0 or X_can.empty:
                logger.info("ℹ️ %s has no usable features — stamping unscored on canon subset.", mkt.upper())
                df_canon['Model_Sharp_Win_Prob'] = np.nan
                df_canon['Model_Confidence']     = np.nan
                df_canon['Scored_By_Model']      = False
                df_canon['Scoring_Market']       = mkt
            else:
                preds = predict_blended(bundle, X_can, model=model, iso=iso)
                if preds is None or len(preds) != len(df_canon):
                    logger.info("ℹ️ %s no usable predictor or length mismatch — stamping unscored on canon subset.", mkt.upper())
                    df_canon['Model_Sharp_Win_Prob'] = np.nan
                    df_canon['Model_Confidence']     = np.nan
                    df_canon['Scored_By_Model']      = False
                    df_canon['Scoring_Market']       = mkt
                else:
                    preds = np.clip(np.asarray(preds, dtype=float).ravel(), 1e-6, 1-1e-6)
                    df_canon['Model_Sharp_Win_Prob'] = preds
                    df_canon['Model_Confidence']     = preds
                    df_canon['Scored_By_Model']      = True
                    df_canon['Scoring_Market']       = mkt
        
            # ✅ Write back predictions ONLY to the rows we actually scored
            cols_to_write = ['Model_Sharp_Win_Prob','Model_Confidence','Scored_By_Model','Scoring_Market']
            df.loc[df_canon.index, cols_to_write] = df_canon[cols_to_write].values
        
        except Exception as e:
            logger.error("❌ Scoring failed for %s — stamping unscored on canon subset. Error: %s", mkt.upper(), e)
            # Make sure columns exist
            for col in ['Model_Sharp_Win_Prob','Model_Confidence','Scored_By_Model','Scoring_Market']:
                if col not in df_canon.columns:
                    df_canon[col] = np.nan if col != 'Scoring_Market' else mkt
                if col not in df.columns:
                    df[col] = np.nan if col != 'Scoring_Market' else mkt
            df_canon['Model_Sharp_Win_Prob'] = np.nan
            df_canon['Model_Confidence']     = np.nan
            df_canon['Scored_By_Model']      = False
            df_canon['Scoring_Market']       = mkt
            df.loc[df_canon.index, ['Model_Sharp_Win_Prob','Model_Confidence','Scored_By_Model','Scoring_Market']] = \
                df_canon[['Model_Sharp_Win_Prob','Model_Confidence','Scored_By_Model','Scoring_Market']].values

    # ===================== END MARKET RESOLUTION + SCORING LOOP ===================== SCORING LOOP =====================
           
            
        # --- 2) Build INVERSE slice (always outside the try/except) ---
        if 'Was_Canonical' in df_m.columns:
            df_inverse = df_m.loc[df_m['Was_Canonical'] == False].copy()
        else:
            df_inverse = df_m.loc[~df_m.index.isin(df_canon.index)].copy()
        
        if df_inverse.empty or df_canon.empty:
            logger.info(f"ℹ️ No inverse flip possible for {mkt.upper()} (canon={len(df_canon)}, inverse={len(df_inverse)}).")
        else:
            # normalize for joins
            for frame in (df_canon, df_inverse):
                for col in ['Bookmaker','Home_Team_Norm','Away_Team_Norm','Market']:
                    if col in frame.columns:
                        frame[col] = frame[col].astype(str).str.lower().str.strip()
        
            # build join key (safe against NaNs)
            for frame in (df_canon, df_inverse):
                frame['Team_Key_Base'] = (
                    frame['Home_Team_Norm'].fillna('').astype(str).str.lower().str.strip() + "_" +
                    frame['Away_Team_Norm'].fillna('').astype(str).str.lower().str.strip() + "_" +
                    frame['Commence_Hour'].astype(str) + "_" +
                    frame['Market'].fillna('').astype(str).str.lower().str.strip()
                )
        
            df_canon_preds = (
                df_canon[['Team_Key_Base','Bookmaker','Model_Sharp_Win_Prob','Model_Confidence']]
                .drop_duplicates(subset=['Team_Key_Base','Bookmaker'])
                .rename(columns={
                    'Model_Sharp_Win_Prob': 'Model_Sharp_Win_Prob_opponent',
                    'Model_Confidence':     'Model_Confidence_opponent'
                })
            )
        
            df_inverse = df_inverse.merge(df_canon_preds, on=['Team_Key_Base','Bookmaker'], how='left')
            df_inverse['Model_Sharp_Win_Prob'] = 1 - df_inverse['Model_Sharp_Win_Prob_opponent']
            df_inverse['Model_Confidence']     = 1 - df_inverse['Model_Confidence_opponent']
            df_inverse.drop(columns=['Model_Sharp_Win_Prob_opponent','Model_Confidence_opponent'], inplace=True, errors='ignore')
        
            df_inverse['Was_Canonical']   = False
            df_inverse['Scored_By_Model'] = True
            df_inverse['Scoring_Market']  = mkt
                    
            # Inverse feature engineering (kept as in your code)
            df_inverse['Line_Move_Magnitude'] = pd.to_numeric(df_inverse['Line_Delta'], errors='coerce').abs()
            df_inverse['Line_Magnitude_Abs']  = df_inverse['Line_Move_Magnitude']
            df_inverse['Sharp_Line_Delta']    = np.where(df_inverse['Is_Sharp_Book'] == 1, df_inverse['Line_Delta'], 0)
            df_inverse['Rec_Line_Delta']      = np.where(df_inverse['Is_Sharp_Book'] == 0, df_inverse['Line_Delta'], 0)
          
            # magnitudes (keep as-is but numeric-safe)
            df_inverse['Sharp_Line_Magnitude'] = pd.to_numeric(df_inverse.get('Sharp_Line_Delta'), errors='coerce').abs()
            df_inverse['Rec_Line_Magnitude']   = pd.to_numeric(df_inverse.get('Rec_Line_Delta'),   errors='coerce').abs()
            
            # High limit flag — use Series default aligned to index (NOT a scalar)
            limits = pd.to_numeric(
                df_inverse['Sharp_Limit_Total'] if 'Sharp_Limit_Total' in df_inverse.columns
                else pd.Series(np.nan, index=df_inverse.index),
                errors='coerce'
            )
            df_inverse['High_Limit_Flag'] = (limits >= 10_000).astype('int8')
            
            # Sharp move × odds shift (only if Odds_Shift exists)
            sig = (
                pd.to_numeric(
                    df_inverse['Sharp_Move_Signal']
                    if 'Sharp_Move_Signal' in df_inverse.columns
                    else pd.Series(0, index=df_inverse.index),
                    errors='coerce'
                )
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype('int8')
            )
            
            odds = (
                pd.to_numeric(
                    df_inverse['Odds_Shift']
                    if 'Odds_Shift' in df_inverse.columns
                    else pd.Series(0.0, index=df_inverse.index),
                    errors='coerce'
                )
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype('float32')
            )
            
            # products/flags (no integer cast until inputs are clean)
            df_inverse['SharpMove_OddsShift'] = (sig.astype('float32') * odds)
            
            df_inverse['SharpMove_Odds_Up']   = ((sig == 1) & (odds > 0)).astype('int8')
            df_inverse['SharpMove_Odds_Down'] = ((sig == 1) & (odds < 0)).astype('int8')
            # Market leader × implied prob shift (only if Implied_Prob_Shift exists)
            if 'Implied_Prob_Shift' in df_inverse.columns:
                leader = df_inverse['Market_Leader'] if 'Market_Leader' in df_inverse.columns else pd.Series(False, index=df_inverse.index)
                df_inverse['MarketLeader_ImpProbShift'] = leader.astype('int8') * pd.to_numeric(df_inverse['Implied_Prob_Shift'], errors='coerce').fillna(0)

            df_inverse['SharpLimit_SharpBook']  = df_inverse['Is_Sharp_Book'] * df_inverse.get('Sharp_Limit_Total', 0)
            df_inverse['LimitProtect_SharpMag'] = df_inverse['LimitUp_NoMove_Flag'] * df_inverse['Sharp_Line_Magnitude']
            df_inverse['HomeRecLineMag']        = df_inverse['Is_Home_Team_Bet'] * df_inverse['Rec_Line_Magnitude']
        
            df_inverse['Delta_Sharp_vs_Rec'] = df_inverse['Sharp_Line_Delta'] - df_inverse['Rec_Line_Delta']
            df_inverse['Sharp_Leads'] = (df_inverse['Sharp_Line_Magnitude'] > df_inverse['Rec_Line_Magnitude']).astype(int)
            df_inverse['Same_Direction_Move'] = (
                (np.sign(df_inverse['Sharp_Line_Delta']) == np.sign(df_inverse['Rec_Line_Delta'])) &
                (df_inverse['Sharp_Line_Delta'].abs() > 0) & (df_inverse['Rec_Line_Delta'].abs() > 0)
            ).astype(int)
            df_inverse['Opposite_Direction_Move'] = (
                (np.sign(df_inverse['Sharp_Line_Delta']) != np.sign(df_inverse['Rec_Line_Delta'])) &
                (df_inverse['Sharp_Line_Delta'].abs() > 0) & (df_inverse['Rec_Line_Delta'].abs() > 0)
            ).astype(int)
            df_inverse['Sharp_Move_No_Rec'] = ((df_inverse['Sharp_Line_Delta'].abs() > 0) & (df_inverse['Rec_Line_Delta'].abs() == 0)).astype(int)
            df_inverse['Rec_Move_No_Sharp'] = ((df_inverse['Rec_Line_Delta'].abs() > 0) & (df_inverse['Sharp_Line_Delta'].abs() == 0)).astype(int)
        
          
            df_inverse['SharpMove_Odds_Mag'] = (
                df_inverse.get('Odds_Shift', 0).abs() *
                df_inverse.get('Sharp_Move_Signal', 0)
            ).fillna(0).astype(float)
        
            df_inverse['Net_Line_Move_From_Opening'] = df_inverse['Value'] - df_inverse['Open_Value']
            df_inverse['Abs_Line_Move_From_Opening'] = df_inverse['Net_Line_Move_From_Opening'].abs()
        
            # Use implied probability deltas consistently
            df_inverse['Net_Odds_Move_From_Opening'] = (
                df_inverse['Odds_Price'].apply(implied_prob) - df_inverse['Open_Odds'].apply(implied_prob)
            ) * 100.0
            df_inverse['Abs_Odds_Move_From_Opening'] = df_inverse['Net_Odds_Move_From_Opening'].abs()
        
            
            df_inverse['Minutes_To_Game'] = (
                pd.to_datetime(df_inverse['Game_Start'], utc=True) - pd.to_datetime(df_inverse['Snapshot_Timestamp'], utc=True)
            ).dt.total_seconds() / 60.0
            df_inverse['Late_Game_Steam_Flag'] = (df_inverse['Minutes_To_Game'] <= 60).astype(int)
            df_inverse['Market_Implied_Prob'] = df_inverse['Odds_Price'].apply(implied_prob)
             
            # Ensure team columns exist before gaps
            for col in ['Team_Past_Avg_Model_Prob','Team_Past_Avg_Model_Prob_Home','Team_Past_Avg_Model_Prob_Away']:
                if col not in df_inverse.columns:
                    df_inverse[col] = 0.0
         
            df_inverse['Market_Mispricing'] = df_inverse['Team_Past_Avg_Model_Prob'] - df_inverse['Market_Implied_Prob']
            df_inverse['Abs_Mispricing_Gap'] = df_inverse['Market_Mispricing'].abs()
            df_inverse['Mispricing_Flag'] = (df_inverse['Abs_Mispricing_Gap'] > 0.05).astype(int)
            df_inverse['Team_Implied_Prob_Gap_Home'] = df_inverse['Team_Past_Avg_Model_Prob_Home'] - df_inverse['Market_Implied_Prob']
            df_inverse['Team_Implied_Prob_Gap_Away'] = df_inverse['Team_Past_Avg_Model_Prob_Away'] - df_inverse['Market_Implied_Prob']
            df_inverse['Abs_Team_Implied_Prob_Gap'] = np.where(
                df_inverse['Is_Home_Team_Bet'] == 1,
                df_inverse['Team_Implied_Prob_Gap_Home'].abs(),
                df_inverse['Team_Implied_Prob_Gap_Away'].abs()
            )
            df_inverse['Team_Mispriced_Flag'] = (df_inverse['Abs_Team_Implied_Prob_Gap'] > 0.05).astype(int)
            # Reversal diagnostics
            df_inverse = compute_value_reversal(df_inverse)
            df_inverse = compute_odds_reversal(df_inverse)

            df_inverse = add_line_and_crossmarket_features(df_inverse)
            # Guarantee reliability cols
            for col in ['Book_Reliability_Score','Book_Reliability_Lift']:
                if col not in df_inverse.columns: df_inverse[col] = 0.0
            df_inverse['Book_Reliability_x_Sharp']      = df_inverse['Book_Reliability_Score'] * df_inverse['Is_Sharp_Book']
            df_inverse['Book_Reliability_x_Magnitude']  = df_inverse['Book_Reliability_Score'] * df_inverse['Sharp_Line_Magnitude']
            df_inverse['Book_Reliability_x_PROB_SHIFT'] = df_inverse['Book_Reliability_Score'] * df_inverse['Is_Sharp_Book'] * df_inverse['Implied_Prob_Shift']
            df_inverse['Book_lift_x_Sharp']             = df_inverse['Book_Reliability_Lift'] * df_inverse['Is_Sharp_Book']
            df_inverse['Book_lift_x_Magnitude']         = df_inverse['Book_Reliability_Lift'] * df_inverse['Sharp_Line_Magnitude']
            df_inverse['Book_lift_x_PROB_SHIFT']        = df_inverse['Book_Reliability_Lift'] * df_inverse['Is_Sharp_Book'] * df_inverse['Implied_Prob_Shift']
        
            df_inverse['Minutes_To_Game_Tier'] = pd.cut(
                df_inverse['Minutes_To_Game'], bins=[-1, 30, 60, 180, 360, 720, np.inf],
                labels=['🚨 ≤30m','🔥 ≤1h','⚠️ ≤3h','⏳ ≤6h','📅 ≤12h','🕓 >12h']
            )
        
         
        
            if 'Value_Reversal_Flag' not in df_inverse.columns:
                df_inverse['Value_Reversal_Flag'] = 0
            else:
                df_inverse['Value_Reversal_Flag'] = pd.to_numeric(df_inverse['Value_Reversal_Flag'], errors='coerce').fillna(0).astype(int)
            if 'Odds_Reversal_Flag' not in df_inverse.columns:
                df_inverse['Odds_Reversal_Flag'] = 0
            else:
                df_inverse['Odds_Reversal_Flag'] = pd.to_numeric(df_inverse['Odds_Reversal_Flag'], errors='coerce').fillna(0).astype(int)
        
            # ===== WRITEBACK: push canon + inverse features/preds back to base df by index =====
            must_cols = ['Model_Sharp_Win_Prob','Model_Confidence','Scored_By_Model','Scoring_Market','Was_Canonical']
            existing_cols = set(df.columns)
            df_canon['Was_Canonical'] = True
            if 'Was_Canonical' not in df_inverse.columns:
                df_inverse['Was_Canonical'] = False
            
            # Columns in slices but not yet in df
            canon_new = [c for c in df_canon.columns   if c not in existing_cols]
            inv_new   = [c for c in df_inverse.columns if c not in existing_cols]
            new_cols  = sorted(set(canon_new + inv_new))
            
            # Add only missing columns to df with sensible defaults
            for c in new_cols:
                source = df_canon if c in df_canon.columns else df_inverse
                df[c] = 0.0 if pd.api.types.is_numeric_dtype(source[c]) else pd.NA
            
            cols_to_push = sorted(set(must_cols + new_cols))
            
            canon_cols_present = [c for c in cols_to_push if c in df_canon.columns]
            inv_cols_present   = [c for c in cols_to_push if c in df_inverse.columns]
            
            if len(df_canon) > 0 and len(canon_cols_present) > 0:
                df.loc[df_canon.index, canon_cols_present] = df_canon[canon_cols_present].values
            if len(df_inverse) > 0 and len(inv_cols_present) > 0:
                df.loc[df_inverse.index, inv_cols_present] = df_inverse[inv_cols_present].values
            
            logger.info(f"📦 Wrote back canon ({len(df_canon)}) + inverse ({len(df_inverse)}) rows to base df for market {mkt}.")
            
            # === 🔁 Re-merge & refresh inverse rows (PER-MARKET) ===
            try:
                
            
                # Make sure First_Imp_Prob exists for recompute (it was dropped earlier)
                if 'First_Imp_Prob' not in df_inverse.columns:
                    df_inverse['First_Imp_Prob'] = np.where(
                        df_inverse.get('Open_Odds').notna(),
                        df_inverse['Open_Odds'].apply(implied_prob),
                        df_inverse.get('Implied_Prob', np.nan)
                    )
            
                # Drop columns that will be recomputed to avoid _x/_y suffixes
                cols_to_refresh = [
                    'Odds_Shift','Line_Delta','Implied_Prob_Shift',
                 
                    'Is_Home_Team_Bet','Is_Favorite_Bet','Delta','Direction_Aligned',
                    'Line_Move_Magnitude','Line_Magnitude_Abs',
                    'Net_Line_Move_From_Opening','Abs_Line_Move_From_Opening',
                    'Net_Odds_Move_From_Opening','Abs_Odds_Move_From_Opening',
                    'Team_Past_Hit_Rate','Team_Past_Hit_Rate_Home','Team_Past_Hit_Rate_Away',
                    'Team_Past_Avg_Model_Prob','Team_Past_Avg_Model_Prob_Home','Team_Past_Avg_Model_Prob_Away',
                    'Market_Implied_Prob','Mispricing_Flag',
                    'Team_Implied_Prob_Gap_Home','Team_Implied_Prob_Gap_Away',
                    'Avg_Recent_Cover_Streak','Avg_Recent_Cover_Streak_Home','Avg_Recent_Cover_Streak_Away',
                    'Rate_On_Cover_Streak','Rate_On_Cover_Streak_Home','Rate_On_Cover_Streak_Away',
                    'Spread_vs_H2H_Aligned','Total_vs_Spread_Contradiction',
                    'Spread_vs_H2H_ProbGap','Total_vs_H2H_ProbGap','Total_vs_Spread_ProbGap','CrossMarket_Prob_Gap_Exists',
                    'Spread_Implied_Prob','H2H_Implied_Prob','Total_Implied_Prob',
                    'Abs_Line_Move_From_Opening','Pct_Line_Move_From_Opening','Pct_Line_Move_Bin',
                    'Abs_Line_Move_Z','Pct_Line_Move_Z',
                    'Implied_Prob','Implied_Prob_Shift','Implied_Prob_Shift_Z',
                    'Potential_Overmove_Flag','Potential_Overmove_Total_Pct_Flag',
                    'Line_Moved_Away_From_Team',
                    'Disable_Line_Move_Features',
                    'Pct_On_Recent_Cover_Streak_Home','Pct_On_Recent_Cover_Streak_Away','Pct_On_Recent_Cover_Streak',
                    'SmallBook_Total_Limit','SmallBook_Max_Limit','SmallBook_Min_Limit','SmallBook_Limit_Count',
                    'SmallBook_Limit_Skew','SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag',
                    'Team_Recent_Cover_Streak_Away_Fav','Team_Recent_Cover_Streak_Away','Team_Recent_Cover_Streak_Fav',
                    'Team_Recent_Cover_Streak_Home_Fav','Team_Recent_Cover_Streak_Home','Team_Recent_Cover_Streak',
                    'On_Cover_Streak_Away_Fav','On_Cover_Streak_Away','On_Cover_Streak_Fav','On_Cover_Streak_Home_Fav',
                    'On_Cover_Streak_Home','On_Cover_Streak'
                ]
                suffix_cols     = [c for c in df_inverse.columns if c.endswith('_x') or c.endswith('_y')]
                team_stat_cols  = [c for c in df_inverse.columns if c.startswith('Team_Past_')]
                drop_cols       = list(set(cols_to_refresh + suffix_cols + team_stat_cols))
                df_inverse = df_inverse.drop(columns=[c for c in drop_cols if c in df_inverse.columns], errors='ignore')
            
                # Re-attach team features (if available)
                try:
                    if team_feature_map is not None and not team_feature_map.empty:
                        df_inverse['Team'] = df_inverse['Outcome_Norm'].astype(str).str.lower().str.strip()
                        # --- SAFE team_feature_map merge (prevents Sport_x/Market_y, etc.) ---
                        if team_feature_map is not None and hasattr(team_feature_map, "empty") and not team_feature_map.empty:
                            tfm = team_feature_map.copy()
                        
                            # Normalize Team key
                            tfm["Team"] = tfm["Team"].astype(str).str.lower().str.strip()
                            df_inverse["Team"] = df_inverse["Team"].astype(str).str.lower().str.strip()
                        
                            # IMPORTANT: do NOT bring Sport/Market from tfm (df already has them)
                            drop_ctx = [c for c in ["Sport", "Market", "Market_norm", "Market_Norm", "Sport_Norm"] if c in tfm.columns]
                            tfm = tfm.drop(columns=drop_ctx, errors="ignore")
                        
                            # Deduplicate map on Team to avoid row explosion
                            tfm = tfm.drop_duplicates(subset=["Team"], keep="last")
                        
                            # Only merge columns that won't collide
                            tfm_cols = [c for c in tfm.columns if c == "Team" or c not in df_inverse.columns]
                            tfm_use = tfm[tfm_cols]
                        
                            df_inverse = df_inverse.merge(tfm_use, on="Team", how="left", validate="m:1")

                        logger.info("🔁 Re-merged team-level features for %d inverse rows.", len(df_inverse))
                except Exception as e:
                    logger.error("❌ Failed to re-merge team-level features for inverse rows: %s", e)
                df_inverse.drop(columns=['Team'], inplace=True, errors='ignore')
            
                # Recompute outcome-sensitive & movement fields on inverse
                df_inverse['Implied_Prob']       = df_inverse['Odds_Price'].apply(implied_prob)
                df_inverse['Odds_Shift']         = (
                    df_inverse['Odds_Price'].apply(implied_prob) - df_inverse['Open_Odds'].apply(implied_prob)
                ) * 100.0
                df_inverse['Implied_Prob_Shift'] = df_inverse['Implied_Prob'] - pd.to_numeric(df_inverse['First_Imp_Prob'], errors='coerce')
                df_inverse['Line_Delta']         = pd.to_numeric(df_inverse['Value'], errors='coerce') - pd.to_numeric(df_inverse['Open_Value'], errors='coerce')
                df_inverse['Delta']              = df_inverse['Line_Delta']
            
                # Cross-market implied probs (guard for missing)
                if 'Spread_Odds' in df_inverse.columns:
                    df_inverse['Spread_Implied_Prob'] = df_inverse['Spread_Odds'].apply(implied_prob)
                if 'H2H_Odds' in df_inverse.columns:
                    df_inverse['H2H_Implied_Prob']    = df_inverse['H2H_Odds'].apply(implied_prob)
                if 'Total_Odds' in df_inverse.columns:
                    df_inverse['Total_Implied_Prob']  = df_inverse['Total_Odds'].apply(implied_prob)
            
                # Alignment & gaps
                df_inverse['Spread_vs_H2H_Aligned'] = ((df_inverse['Value'] < 0) & (df_inverse.get('H2H_Implied_Prob', 0) > 0.5)).astype(int)
                df_inverse['Total_vs_Spread_Contradiction'] = (
                    (df_inverse.get('Spread_Implied_Prob', 0) > 0.55) & (df_inverse.get('Total_Implied_Prob', 1) < 0.48)
                ).astype(int)
                df_inverse['Spread_vs_H2H_ProbGap']   = df_inverse.get('Spread_Implied_Prob', np.nan) - df_inverse.get('H2H_Implied_Prob', np.nan)
                df_inverse['Total_vs_H2H_ProbGap']    = df_inverse.get('Total_Implied_Prob', np.nan)  - df_inverse.get('H2H_Implied_Prob', np.nan)
                df_inverse['Total_vs_Spread_ProbGap'] = df_inverse.get('Total_Implied_Prob', np.nan)  - df_inverse.get('Spread_Implied_Prob', np.nan)
                df_inverse['CrossMarket_Prob_Gap_Exists'] = (
                    (df_inverse['Spread_vs_H2H_ProbGap'].abs() > 0.05) | (df_inverse['Total_vs_Spread_ProbGap'].abs() > 0.05)
                ).astype(int)
            
                # Directional bets/flags
                df_inverse['Is_Home_Team_Bet'] = (df_inverse['Outcome'].str.lower() == df_inverse['Home_Team_Norm'].str.lower()).astype(float)
                df_inverse['Is_Favorite_Bet']  = (pd.to_numeric(df_inverse['Value'], errors='coerce') < 0).astype(float)
                df_inverse['Direction_Aligned'] = np.where(df_inverse['Line_Delta'] > 0, 1,
                                                    np.where(df_inverse['Line_Delta'] < 0, 0, np.nan)).astype(float)
            
                # Movement from open (line & odds)
                df_inverse['Net_Line_Move_From_Opening'] = df_inverse['Value'] - df_inverse['Open_Value']
                df_inverse['Abs_Line_Move_From_Opening'] = df_inverse['Net_Line_Move_From_Opening'].abs()
                df_inverse['Net_Odds_Move_From_Opening'] = (
                    df_inverse['Odds_Price'].apply(implied_prob) - df_inverse['Open_Odds'].apply(implied_prob)
                ) * 100.0
                df_inverse['Abs_Odds_Move_From_Opening'] = df_inverse['Net_Odds_Move_From_Opening'].abs()
            
                # Magnitudes + market fair prob
                df_inverse['Line_Move_Magnitude'] = df_inverse['Line_Delta'].abs()
                df_inverse['Line_Magnitude_Abs']  = df_inverse['Line_Move_Magnitude']
                df_inverse['Market_Implied_Prob'] = df_inverse['Odds_Price'].apply(implied_prob)
            
                # Mispricing & team gaps
                for col in ['Team_Past_Avg_Model_Prob','Team_Past_Avg_Model_Prob_Home','Team_Past_Avg_Model_Prob_Away']:
                    if col not in df_inverse.columns: df_inverse[col] = 0.0
             
                df_inverse['Mispricing_Flag']      = (df_inverse['Abs_Mispricing_Gap'] > 0.05).astype(int)
                df_inverse['Team_Implied_Prob_Gap_Home'] = df_inverse['Team_Past_Avg_Model_Prob_Home'] - df_inverse['Market_Implied_Prob']
                df_inverse['Team_Implied_Prob_Gap_Away'] = df_inverse['Team_Past_Avg_Model_Prob_Away'] - df_inverse['Market_Implied_Prob']
                df_inverse['Abs_Team_Implied_Prob_Gap'] = np.where(
                    df_inverse['Is_Home_Team_Bet'] == 1,
                    df_inverse['Team_Implied_Prob_Gap_Home'].abs(),
                    df_inverse['Team_Implied_Prob_Gap_Away'].abs()
                )
                df_inverse['Team_Mispriced_Flag'] = (df_inverse['Abs_Team_Implied_Prob_Gap'] > 0.05).astype(int)
            
                # Crossmarket + small book features
                df_inverse = add_line_and_crossmarket_features(df_inverse)
                df_inverse = compute_small_book_liquidity_features(df_inverse)
            
            
                logger.info(f"🔁 Refreshed Open/Extreme alignment for {len(df_inverse)} inverse rows.")
            
                # 🔄 WRITE BACK REFRESHED INVERSE AGAIN
                inv_cols_present = [c for c in df_inverse.columns if c in df.columns]
                df.loc[df_inverse.index, inv_cols_present] = df_inverse[inv_cols_present].values
            
            except Exception as e:
               
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                logger.error("❌ Failed to refresh inverse rows after re-merge.")
                logger.error(f"🛠 Exception Type: {exc_type.__name__}")
                logger.error(f"📍 Exception Message: {e}")
                logger.error(f"🧵 Full Traceback:\n{''.join(tb_lines)}")
            
            # === ✅ Combine canonical and inverse into one scored DataFrame (PER-MARKET) ===
            logger.info(f"📋 Inverse2 row columns after enrichment: {sorted(df_inverse.columns.tolist())}")
            logger.info(f"📋 canon row columns after enrichment: {sorted(df_canon.columns.tolist())}")
            
            def _dedupe_columns(frame):
                return frame.loc[:, ~frame.columns.duplicated()]
            
            df_canon   = _dedupe_columns(df_canon)
            df_inverse = _dedupe_columns(df_inverse)
            
            # Align schemas
            all_cols = sorted(set(df_canon.columns).union(set(df_inverse.columns)))
            for col in all_cols:
                if col not in df_canon.columns:
                    df_canon[col] = np.nan
                if col not in df_inverse.columns:
                    df_inverse[col] = np.nan
            
            df_canon   = df_canon[all_cols].reset_index(drop=True)
            df_inverse = df_inverse[all_cols].reset_index(drop=True)
            df_canon.index.name = None
            df_inverse.index.name = None
            
            # Build df_scored & append
            df_scored = pd.concat([df_canon, df_inverse], ignore_index=True)
            
            # Safe preview logging
            try:
                logger.info(f"📋 scored row columns after enrichment: {sorted(df_scored.columns.tolist())}")
                logger.info("🧩 df_scored — Columns: %s", df_scored.columns.tolist())
                logger.info("🔍 df_scored — Sample Rows:\n%s", df_scored[[
                    'Game_Key','Market','Outcome','Model_Sharp_Win_Prob',
                    'Team_Past_Hit_Rate','Team_Past_Avg_Model_Prob'
                ]].head(5).to_string(index=False))
            except Exception as log_error:
                logger.warning(f"⚠️ Could not log scored row preview: {log_error}")
            
            # Confidence tier (bins are inclusive on right by default)
            df_scored['Model_Confidence_Tier'] = pd.cut(
                df_scored['Model_Sharp_Win_Prob'],
                bins=[0, 0.5, 0.55, 0.7, 1.0],
                labels=["✅ Low","⭐ Lean","🔥 Strong Indication","🔥 Steam"],
                include_lowest=True
            )
            
            scored_all.append(df_scored)   
    try:
        df_final = pd.DataFrame()
    
    
        if scored_all:
            # 1) concat all markets
            df_final = pd.concat(scored_all, ignore_index=True)
    
            logger.info(
                f"🧮 Final scored breakdown — total={len(df_final)}, "
                f"canonical={df_final['Was_Canonical'].sum()}, inverse={(~df_final['Was_Canonical']).sum()}"
            )
    
            # 2) snapshot collapse (latest per book)
            df_final = (
                df_final.sort_values('Snapshot_Timestamp')
                        .drop_duplicates(subset=['Game_Key','Market','Outcome','Bookmaker'], keep='last')
            )
    
            # 3) numeric coercions / default zeros for features used in counts
            cols_for_count = [
                'Sharp_Move_Signal','Sharp_Limit_Jump','Sharp_Limit_Total','LimitUp_NoMove_Flag',
                'Market_Leader','Is_Reinforced_MultiMarket','Sharp_Line_Magnitude','SharpMove_Odds_Up',
                'SharpMove_Odds_Down','SharpMove_Odds_Mag','Late_Game_Steam_Flag',
                'Value_Reversal_Flag','Odds_Reversal_Flag','Abs_Line_Move_From_Opening','Abs_Odds_Move_From_Opening',
                'Team_Past_Hit_Rate','Mispricing_Flag','Team_Implied_Prob_Gap_Home','Team_Implied_Prob_Gap_Away',
                'Avg_Recent_Cover_Streak','Avg_Recent_Cover_Streak_Home','Avg_Recent_Cover_Streak_Away',
                'Spread_vs_H2H_Aligned','Total_vs_Spread_Contradiction','CrossMarket_Prob_Gap_Exists',
                'Potential_Overmove_Flag','Potential_Overmove_Total_Pct_Flag','Potential_Odds_Overmove_Flag',
                'Line_Moved_Toward_Team','Line_Moved_Away_From_Team',
                'Abs_Line_Move_Z','Pct_Line_Move_Z','SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag',
                'SmallBook_Total_Limit'
            ]
    
            # Some columns may not exist depending on earlier paths; create them as 0 for counting
            for c in cols_for_count:
                if c not in df_final.columns:
                    df_final[c] = 0
                df_final[c] = pd.to_numeric(df_final[c], errors='coerce').fillna(0)
    
            # 4) hybrid timing flags

    
            # 5) signal count (kept; matches UI-style “why” tally)
            # Guard for Rec_Line_Magnitude which may not exist on some paths
            if 'Rec_Line_Magnitude' not in df_final.columns:
                df_final['Rec_Line_Magnitude'] = 0
    
            df_final['Active_Signal_Count'] = (
                (df_final['Sharp_Move_Signal'] == 1).astype(int) +
                (df_final['Sharp_Limit_Jump'] == 1).astype(int) +
                (df_final['Sharp_Limit_Total'] > 10000).astype(int) +
                (df_final['LimitUp_NoMove_Flag'] == 1).astype(int) +
                (df_final['Market_Leader'] == 1).astype(int) +
                (df_final['Is_Reinforced_MultiMarket'] == 1).astype(int) +
                (df_final['Sharp_Line_Magnitude'] > 0.5).astype(int) +
                (df_final['Rec_Line_Magnitude'] > 0.5).astype(int) +
                (df_final['SharpMove_Odds_Up'] == 1).astype(int) +
                (df_final['SharpMove_Odds_Down'] == 1).astype(int) +
                (df_final['SharpMove_Odds_Mag'] > 5).astype(int) +
                
                (df_final['Late_Game_Steam_Flag'] == 1).astype(int) +
                (df_final['Value_Reversal_Flag'] == 1).astype(int) +
                (df_final['Odds_Reversal_Flag'] == 1).astype(int) +
                (df_final['Abs_Line_Move_From_Opening'] > 1.0).astype(int) +
                (df_final['Abs_Odds_Move_From_Opening'] > 5).astype(int) +
               
                (df_final['Team_Past_Hit_Rate'] > 0.6).astype(int) +
                df_final['Mispricing_Flag'].astype(int) +
                (df_final['Team_Implied_Prob_Gap_Home'] > 0.05).astype(int) +
                (df_final['Team_Implied_Prob_Gap_Away'] > 0.05).astype(int) +
                (df_final['Avg_Recent_Cover_Streak'] >= 2).astype(int) +
                (df_final['Avg_Recent_Cover_Streak_Home'] >= 2).astype(int) +
                (df_final['Avg_Recent_Cover_Streak_Away'] >= 2).astype(int) +
                df_final['Spread_vs_H2H_Aligned'].astype(int) +
                df_final['Total_vs_Spread_Contradiction'].astype(int) +
                df_final['CrossMarket_Prob_Gap_Exists'].astype(int) +
                (df_final['Potential_Overmove_Flag'] == 1).astype(int) +
                (df_final['Potential_Overmove_Total_Pct_Flag'] == 1).astype(int) +
                (df_final['Potential_Odds_Overmove_Flag'] == 1).astype(int) +
                (df_final['Line_Moved_Toward_Team'] == 1).astype(int) +
                (df_final['Line_Moved_Away_From_Team'] == 1).astype(int) +
                
                (df_final['Abs_Line_Move_Z'] > 1).astype(int) +
                (df_final['Pct_Line_Move_Z'] > 1).astype(int) +
                (df_final['SmallBook_Heavy_Liquidity_Flag'] == 1).astype(int) +
                (df_final['SmallBook_Limit_Skew_Flag'] == 1).astype(int) +
                (df_final['SmallBook_Total_Limit'] > 500).astype(int)
            ).astype(int)
    
            # 6) Diagnostic for unscored rows
            # 6) Diagnostic for unscored rows (cheap)
            try:
                if 'Model_Sharp_Win_Prob' in df_final.columns:
                    unscored_mask = df_final['Model_Sharp_Win_Prob'].isna()
                    if unscored_mask.any():
                        n = int(unscored_mask.sum())
                        logger.warning("⚠️ %d rows were not scored (Model_Sharp_Win_Prob is null).", n)
                        cols_dbg = [c for c in ['Game','Bookmaker','Market','Outcome','Was_Canonical'] if c in df_final.columns]
                        if cols_dbg:
                            logger.warning(df_final.loc[unscored_mask, cols_dbg].head(10).to_string(index=False))
            except Exception as e:
                logger.error("❌ Failed to log unscored rows by market: %s", e)
            
            # 7) Team_Key + placeholders (low-alloc)
            for col, default in [
                ('Home_Team_Norm',''), ('Away_Team_Norm',''),
                ('Commence_Hour',''), ('Market',''), ('Outcome_Norm','')
            ]:
                if col not in df_final.columns:
                    df_final[col] = default
            
            # Build Team_Key with minimal temporaries
            ht = df_final['Home_Team_Norm'].astype('string').fillna('')
            at = df_final['Away_Team_Norm'].astype('string').fillna('')
            ch = df_final['Commence_Hour'].astype('string').fillna('')
            mk = df_final['Market'].astype('string').fillna('')
            on = df_final['Outcome_Norm'].astype('string').fillna('')
            
            df_final['Team_Key'] = (
                ht.str.cat(at, sep='_')
                  .str.cat(ch, sep='_')
                  .str.cat(mk, sep='_')
                  .str.cat(on, sep='_')
            )
   
         
       
            
            # 8) Extra debug (cheap key diff)
            try:
                if 'Game_Key' in df_final.columns and 'Game_Key' in df.columns:
                    final_keys = df_final['Game_Key'].astype('string').unique()
                    orig_keys  = df['Game_Key'].astype('string').unique()
                    missing_keys = np.setdiff1d(orig_keys, final_keys, assume_unique=False)
                    if missing_keys.size:
                        logger.warning("⚠️ %d Game_Keys were not scored by model", missing_keys.size)
                        # sample a few rows safely
                        sample = df[df['Game_Key'].isin(missing_keys)]
                        cols_dbg2 = [c for c in ['Game','Bookmaker','Market','Outcome','Value'] if c in df.columns]
                        if cols_dbg2 and not sample.empty:
                            logger.warning("🧪 Sample unscored rows:")
                            logger.warning(sample.loc[:, cols_dbg2].head(5).to_string(index=False))
            except Exception as debug_error:
                logger.error("❌ Failed to log unscored rows: %s", debug_error)
            
            # 9) Remove unscored UNDER rows with no OVER source (once)
            if {'Market','Outcome_Norm','Was_Canonical','Model_Sharp_Win_Prob'}.issubset(df_final.columns):
                pre = len(df_final)
                df_final = df_final[~(
                    (df_final['Market'] == 'totals') &
                    (df_final['Outcome_Norm'] == 'under') &
                    (~df_final['Was_Canonical'].astype(bool)) &
                    (df_final['Model_Sharp_Win_Prob'].isna())
                )]
                if pre != len(df_final):
                    logger.info("🧹 Removed %d unscored UNDER rows (no OVER available)", pre - len(df_final))
            
         
            logger.info("✅ Scoring completed in %.2f seconds", time.time() - total_start)
            return df_final
                           
        #else:
            #logger.warning("⚠️ No market types scored — returning empty DataFrame.")
        else:
            logger.warning("⚠️ No market types scored — returning minimally-enriched, unscored snapshots.")
            base = df.copy()
        
            # Stamp prediction columns
            for col, default in [
                ('Model_Sharp_Win_Prob', np.nan),
                ('Model_Confidence',     np.nan),
                ('Scored_By_Model',      False),
                ('Scoring_Market',       base['Market_norm'] if 'Market_norm' in base.columns else ''),
                ('Was_Canonical',        pd.NA),
            ]:
                if col not in base.columns:
                    base[col] = default
        
            # Friendly tier for unscored rows
            base['Model_Confidence_Tier'] = '❔ No Model'
        
            # Commence hour (needed for Team_Key)
            if 'Commence_Hour' not in base.columns:
                base['Commence_Hour'] = pd.to_datetime(
                    base.get('Game_Start', pd.NaT), utc=True, errors='coerce'
                ).dt.floor('h')
        
            # Latest snapshot per (Game, Market, Outcome, Bookmaker)
            essential = {'Snapshot_Timestamp','Game_Key','Market','Outcome','Bookmaker'}
            if essential.issubset(base.columns):
                base = (
                    base.sort_values('Snapshot_Timestamp')
                        .drop_duplicates(subset=['Game_Key','Market','Outcome','Bookmaker'], keep='last')
                )
        
            # Ensure key pieces exist, then build Team_Key
            for c, default in [
                ('Home_Team_Norm',''), ('Away_Team_Norm',''),
                ('Market',''), ('Outcome_Norm',''), ('Outcome','')
            ]:
                if c not in base.columns:
                    base[c] = default
        
            on = base['Outcome_Norm'].astype('string').fillna(
                    base['Outcome'].astype('string').fillna('')
                 )
            ht = base['Home_Team_Norm'].astype('string').fillna('')
            at = base['Away_Team_Norm'].astype('string').fillna('')
            ch = base['Commence_Hour'].astype('string').fillna('')
            mk = base['Market'].astype('string').fillna('')
        
            base['Team_Key'] = (
                ht.str.cat(at, sep='_')
                  .str.cat(ch, sep='_')
                  .str.cat(mk, sep='_')
                  .str.cat(on, sep='_')
            )
        
            logger.info("✅ Returning %d unscored rows (no market scored).", len(base))
            return base
    except Exception:
        logger.error("❌ Exception during final aggregation")
        logger.error(traceback.format_exc())
        return pd.DataFrame()
    
def detect_sharp_moves(
    current,
    previous=None,                    # optional/ignored
    sport_key=None,                   # e.g. "basketball_nba"
    SHARP_BOOKS=None,
    REC_BOOKS=None,
    BOOKMAKER_REGIONS=None,
    trained_models=None,
    weights=None,
    has_models: bool | None = None,
    sport_label: str | None = None,   # e.g. "NBA"
    history_hours: int = 120,         # 0/None to skip history
):

    if not current:
        logging.warning("⚠️ No current odds data provided.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ---------- 0) Basics ----------
    df_scored     = pd.DataFrame()
    summary_df    = pd.DataFrame()
    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Normalize sport identifiers (both helpful later)
    _sport_key   = sport_key            # keep as passed (can be None)
    _sport_label = sport_label  

    if has_models is None:
        has_models = bool(trained_models)

    # ---- Assign weights (always at least uniform=1.0)
    if not weights:  # handles None or {}
        weights = {"spreads": 1.0, "totals": 1.0, "h2h": 1.0}
        logging.info(f"✅ Using uniform weights=1.0 for spreads/totals/h2h ({_sport_label or _sport_key})")

    # ---- fast path when no models
    if not has_models:
        skip_snapshot_hydration = True
        skip_df_first = True
        history_hours = 0

      # detect_utils.py
    canon_sport = (_sport_key or _sport_label or None)
    if canon_sport and 'SPORT_ALIASES' in globals():
        s = str(canon_sport).strip().lower()
        for k, alist in SPORT_ALIASES.items():
            if s == k or s in [str(a).strip().lower() for a in (alist or [])]:
                canon_sport = k
                break


    # ---------- 1) Build 'previous' lookup only if provided ----------
    # ---------- 1) Build 'previous' lookup only if provided ----------
    previous_odds_map = {}
    if isinstance(previous, list) and previous:
        for g in previous:
            for book in g.get('bookmakers', []):
                book_key_raw = (book.get('key', '') or '').lower()
                book_key = normalize_book_name(book_key_raw, book_key_raw)
                for market in book.get('markets', []):
                    mtype = (market.get('key') or '').strip().lower()
                    for outcome in market.get('outcomes', []):
                        label = normalize_label(outcome.get('name', ''))
                        price = outcome.get('point') if mtype != 'h2h' else outcome.get('price')
                        previous_odds_map[(g.get('home_team'), g.get('away_team'), mtype, label, book_key)] = price
    elif isinstance(previous, dict) and previous:
        previous_odds_map = previous  # already keyed
       
    # ---------- 2) Pull recent history from BQ (for opens/extremes/reversals) ----------
    df_all_snapshots = pd.DataFrame()
    df_history = pd.DataFrame()
    old_val_map, old_odds_map = {}, {}
    if history_hours and history_hours > 0:
        try:
            df_history = read_recent_sharp_master_cached(hours=history_hours)
            if not df_history.empty:
                # Keep only rows we can key on
                df_history = df_history.dropna(subset=['Game', 'Market', 'Outcome', 'Book', 'Value'])
                df_history = df_history.sort_values('Snapshot_Timestamp')

                # normalize for joins
                for col in ('Book', 'Bookmaker', 'Game', 'Market', 'Outcome'):
                    if col in df_history.columns:
                        df_history[col] = df_history[col].astype(str).str.lower().str.strip()

                # old value/odds maps keyed by (Game, Market, Outcome, Book)
                old_val_map = (
                    df_history
                    .drop_duplicates(subset=['Game', 'Market', 'Outcome', 'Book'], keep='first')
                    .set_index(['Game', 'Market', 'Outcome', 'Book'])['Value']
                    .to_dict()
                )
                old_odds_map = (
                    df_history
                    .dropna(subset=['Odds_Price'])
                    .drop_duplicates(subset=['Game', 'Market', 'Outcome', 'Book'], keep='first')
                    .set_index(['Game', 'Market', 'Outcome', 'Book'])['Odds_Price']
                    .to_dict()
                )
        except Exception as e:
            logging.warning(f"⚠️ History fetch failed: {e}")
        
    else:
        old_val_map, old_odds_map = {}, {}

    # ---------- 3) Flatten CURRENT into rows ----------
    rows = []
    SHARP_REC_SET = set((SHARP_BOOKS or []) + (REC_BOOKS or []))  # assume lowercase
    for game in current:
        home_team = (game.get('home_team', '') or '').strip().lower()
        away_team = (game.get('away_team', '') or '').strip().lower()
        if not home_team or not away_team:
            continue

        # canonical game name used in history keys
        game_name   = f"{home_team.title()} vs {away_team.title()}".strip().lower()
        event_time  = pd.to_datetime(game.get("commence_time"), utc=True, errors='coerce')
        game_hour   = event_time.floor('h') if pd.notnull(event_time) else pd.NaT

        for book in game.get('bookmakers', []):
            book_key_raw = (book.get('key', '') or '').lower()
            book_key     = normalize_book_name(book_key_raw, book_key_raw)  # your util
            if SHARP_REC_SET and book_key not in SHARP_REC_SET:
                continue

            for market in book.get('markets', []):
                mtype = (market.get('key') or '').strip().lower()
                if mtype not in ('spreads', 'totals', 'h2h'):
                    continue

                for o in market.get('outcomes', []):
                    label      = normalize_label(o.get('name', ''))
                    point      = o.get('point')
                    odds_price = o.get('price')
                    value      = odds_price if mtype == 'h2h' else point
                    limit      = o.get('bet_limit')

                    game_key = f"{home_team}_{away_team}_{str(game_hour)}_{mtype}_{label}"
                    team_key = game_key

                    # hydrate opens if available
                    _hist_key = (game_name, mtype, label, book_key)
                    old_val   = old_val_map.get(_hist_key)
                    old_odds  = old_odds_map.get(_hist_key)

                    rows.append({
                        'Sport': _sport_label,          # e.g., "NBA"
                        'Sport_Key': _sport_key,        # e.g., "basketball_nba"
                        'Game_Key': game_key,
                        'Time': snapshot_time,
                        'Game': game_name,              # lower, matches history normalization
                        'Game_Start': event_time,
                        'Market': mtype,
                        'Outcome': label,
                        'Outcome_Norm': label,
                        'Bookmaker': book_key,
                        'Book': book_key,
                        'Value': value,
                        'Odds_Price': odds_price,
                        'Limit': limit,
                        'Old Value': old_val,           # ← from history (open/first seen)
                        'Old_Odds': old_odds,           # ← from history
                        'Home_Team_Norm': home_team,
                        'Away_Team_Norm': away_team,
                        'Commence_Hour': game_hour,
                        'Was_Canonical': None,
                        'Team_Key': team_key,
                    })

  
    # ─────────────────────────────────────────────────────────
    # Build base frame (keep ALL columns)
    # ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    if df.empty:
        logging.warning("⚠️ No sharp rows built.")
        return df, df_history, pd.DataFrame()
    
    # Normalize common string fields (keep dtype='string', no categories here)
    norm_str = lambda s: s.astype('string').str.strip().str.lower()
    for c in ('Market','Outcome','Outcome_Norm','Bookmaker','Game_Key'):
        if c in df.columns:
            df[c] = norm_str(df[c])
    
    # Ensure Outcome_Norm exists
    if 'Outcome_Norm' not in df.columns and 'Outcome' in df.columns:
        df['Outcome_Norm'] = norm_str(df['Outcome'])
    
    # Canonical flag (vectorized; resilient if some cols are missing)
    df['Was_Canonical'] = False
    mkt  = df['Market'] if 'Market' in df.columns else pd.Series(pd.NA, index=df.index, dtype='string')
    outn = df['Outcome_Norm'] if 'Outcome_Norm' in df.columns else pd.Series(pd.NA, index=df.index, dtype='string')
    val  = pd.to_numeric(df.get('Value'), errors='coerce') if 'Value' in df.columns else pd.Series(np.nan, index=df.index)
    
    mask_tot_over = mkt.eq('totals') & outn.eq('over')
    mask_sp_h2h   = mkt.isin(('spreads','h2h')) & (val < 0)
    df.loc[mask_tot_over | mask_sp_h2h, 'Was_Canonical'] = True
    
    # Numeric downcasts (keep columns)
    for c in ('Value','Odds_Price','Limit'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce', downcast='float')
    
    # Timestamps
    if 'Game_Start' in df.columns:
        df['Game_Start'] = pd.to_datetime(df['Game_Start'], errors='coerce', utc=True)
    
    # ─────────────────────────────────────────────────────────
    # 1) Models (lazy load)
    # ─────────────────────────────────────────────────────────
    if trained_models is None:
        trained_models = get_trained_models(sport_key)
    
    # ─────────────────────────────────────────────────────────
    # 2) Read snapshots (KEEP ALL COLUMNS) + normalize hot fields
    # ─────────────────────────────────────────────────────────
    df_all_snapshots = read_recent_sharp_master_cached(hours=120)
    
    # Keep join keys as string for safe merges & concatenation later
    for c in ('Game_Key','Market','Outcome','Bookmaker'):
        if c in df_all_snapshots.columns:
            df_all_snapshots[c] = df_all_snapshots[c].astype('string').str.strip().str.lower()
    
    # Keep team norms as string (we concatenate these later)
    for c in ('Home_Team_Norm','Away_Team_Norm'):
        if c in df_all_snapshots.columns:
            df_all_snapshots[c] = df_all_snapshots[c].astype('string').str.strip().str.lower()
        if c in df.columns:
            df[c] = df[c].astype('string').str.strip().str.lower()
    
    # Numeric downcasts
    for c in ('Value','Odds_Price','Limit'):
        if c in df_all_snapshots.columns:
            df_all_snapshots[c] = pd.to_numeric(df_all_snapshots[c], errors='coerce', downcast='float')
    
    # Timestamps
    for c in ('Snapshot_Timestamp','Game_Start'):
        if c in df_all_snapshots.columns:
            df_all_snapshots[c] = pd.to_datetime(df_all_snapshots[c], errors='coerce', utc=True)
    
    # ─────────────────────────────────────────────────────────
    # 3) Filter snapshot ROWS to keys present in df (not columns)
    # ─────────────────────────────────────────────────────────
    merge_keys = [k for k in ['Game_Key','Market','Outcome','Bookmaker'] if k in df.columns]
    if len(merge_keys) < 4:
        # ensure all keys exist to avoid KeyError in downstream ops
        for k in ['Game_Key','Market','Outcome','Bookmaker']:
            if k not in df.columns:
                df[k] = pd.Series(pd.NA, index=df.index, dtype='string')
            if k not in df_all_snapshots.columns:
                df_all_snapshots[k] = pd.Series(pd.NA, index=df_all_snapshots.index, dtype='string')
        merge_keys = ['Game_Key','Market','Outcome','Bookmaker']
    
    _keys = df[merge_keys].drop_duplicates()
    snaps = df_all_snapshots.merge(_keys, on=merge_keys, how='inner')
    
    # ensure sorted once (stable, low mem)
    if not snaps.empty and 'Snapshot_Timestamp' in snaps.columns:
        snaps = snaps.sort_values('Snapshot_Timestamp', kind='mergesort')
    
    # ─────────────────────────────────────────────────────────
    # 4) Extremes (vectorized)
    # ─────────────────────────────────────────────────────────
    if not snaps.empty:
        # spreads/totals → extremes from Value
        if 'Value' in snaps.columns:
            m_sp_to = snaps['Market'].isin(('spreads','totals'))
            ext_val = (
                snaps.loc[m_sp_to, merge_keys + ['Value']]
                     .dropna(subset=['Value'])
                     .groupby(merge_keys, observed=True)['Value']
                     .agg(Max_Value='max', Min_Value='min')
            )
        else:
            ext_val = pd.DataFrame()
    
        # all markets → extremes from Odds_Price
        if 'Odds_Price' in snaps.columns:
            ext_odds = (
                snaps.loc[:, merge_keys + ['Odds_Price']]
                     .dropna(subset=['Odds_Price'])
                     .groupby(merge_keys, observed=True)['Odds_Price']
                     .agg(Max_Odds='max', Min_Odds='min')
            )
        else:
            ext_odds = pd.DataFrame()
    
        if not ext_val.empty and not ext_odds.empty:
            df_extremes = ext_val.join(ext_odds, how='outer')
        else:
            df_extremes = ext_val if not ext_val.empty else ext_odds
    
        if df_extremes is not None and not df_extremes.empty:
            df = df.merge(df_extremes.reset_index(), on=merge_keys, how='left')
    
    # ─────────────────────────────────────────────────────────
    # 5) Openers (vectorized first after sort)
    # ─────────────────────────────────────────────────────────
    if not snaps.empty:
        gb = snaps.groupby(merge_keys, sort=False, observed=True)
        open_val  = gb['Value'].first()       if 'Value' in snaps.columns else pd.Series(dtype='float32')
        open_odds = gb['Odds_Price'].first()  if 'Odds_Price' in snaps.columns else pd.Series(dtype='float32')
        open_lim  = gb['Limit'].first()       if 'Limit' in snaps.columns else pd.Series(dtype='float32')
    
        opens_df = pd.DataFrame({
            'Open_Value': open_val,
            'Open_Odds':  open_odds,
            'Opening_Limit': open_lim,
        }).reset_index()
    
        df = df.merge(opens_df, on=merge_keys, how='left')
    
    # Fallbacks (fill from current if still missing)
    if 'Opening_Limit' in df.columns and 'Limit' in df.columns:
        need = df['Opening_Limit'].isna() & df['Limit'].notna()
        if need.any():
            df.loc[need, 'Opening_Limit'] = df.loc[need, 'Limit']
    
    # First implied prob from open odds (vectorized)
    if 'Open_Odds' in df.columns:
        if 'First_Imp_Prob' not in df.columns:
            df['First_Imp_Prob'] = np.nan
        need_imp = df['First_Imp_Prob'].isna() & df['Open_Odds'].notna()
        if need_imp.any():
            df.loc[need_imp, 'First_Imp_Prob'] = df.loc[need_imp, 'Open_Odds'].apply(implied_prob)
    
    # ─────────────────────────────────────────────────────────
    # 6) Inverse hydration (only rows needing it; join back by Team_Key)
    # ─────────────────────────────────────────────────────────
    if 'Was_Canonical' in df.columns:
        df_inverse = df.loc[~df['Was_Canonical']]
        need_hydrate_mask = (
            (df_inverse['Value'].isna() if 'Value' in df_inverse.columns else False) |
            (df_inverse['Odds_Price'].isna() if 'Odds_Price' in df_inverse.columns else False)
        )
        if need_hydrate_mask.any():
            needs_hydration = df_inverse.loc[need_hydrate_mask].copy()
            needs_hydration = hydrate_inverse_rows_from_snapshot(needs_hydration, df_all_snapshots)
            needs_hydration = fallback_flip_inverse_rows(needs_hydration)
            if 'Team_Key' in needs_hydration.columns and 'Team_Key' in df.columns:
                tmp = needs_hydration[['Team_Key','Value','Odds_Price','Limit']].drop_duplicates('Team_Key')
                df = df.merge(tmp, on='Team_Key', how='left', suffixes=('','__hyd'))
                for c in ('Value','Odds_Price','Limit'):
                    hc = f'{c}__hyd'
                    if hc in df.columns:
                        need = df[c].isna() & df[hc].notna()
                        if need.any():
                            df.loc[need, c] = df.loc[need, hc]
                        df.drop(columns=[hc], inplace=True)
    
    # ─────────────────────────────────────────────────────────
    # 7) Dedup and rowwise sharp metrics (one pass)
    # ─────────────────────────────────────────────────────────
    df = df.drop_duplicates(subset=merge_keys, keep='last')
    
    # tiny guard: ensure team norm cols exist so metrics won't KeyError
    for c in ('Home_Team_Norm','Away_Team_Norm'):
        if c not in df.columns:
            df[c] = pd.Series(pd.NA, index=df.index, dtype='string')
        if c not in df_all_snapshots.columns:
            df_all_snapshots[c] = pd.Series(pd.NA, index=df_all_snapshots.index, dtype='string')
    
    df = apply_compute_sharp_metrics_rowwise(df, df_all_snapshots)

    # ─────────────────────────────────────────────────────────
    # 8) Score (avoid extra copies) + housekeeping
    # ─────────────────────────────────────────────────────────
    # ---- normalize flags
    if has_models is None:
        has_models = bool(trained_models)
    
    # ---- NEVER self-load weights here; use what the caller passed
    if weights is None:
        # if no models, empty weights; if models but none provided, still proceed with {}
        weights = {}  # keep it simple; caller decides whether to load real weights

    # ✨ MINIMAL ADD: if weights not provided or empty, assign uniform weights=1.0
    if not weights:
        markets_present = (
            df['Market']
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
            .tolist()
        )
        weights = {m: 1.0 for m in markets_present}
        logging.info(
            f"✅ Using uniform weights=1.0 for {len(market_weights)} markets "
            f"({sport_label}): {sorted(market_weights)}"
        )
   
    now = pd.Timestamp.utcnow()
    df['Snapshot_Timestamp'] = now
    if 'Game_Start' in df.columns:
        df['Event_Date'] = df['Game_Start'].dt.date
    
    # Compute hash before scoring
    df['Line_Hash'] = df.apply(compute_line_hash, axis=1)
    
    # If apply_blended_sharp_score mutates, let it copy internally; otherwise pass df directly.
    
    df_scored = apply_blended_sharp_score(
        df, trained_models, df_all_snapshots, weights,  # ← positional
        sport=canon_sport                               # ← keyword-only (after the *)
    )


    
    if not df_scored.empty:
        df_scored['Pre_Game']  = df_scored['Game_Start'] > now
        df_scored['Post_Game'] = ~df_scored['Pre_Game']
        df_scored = df_scored.drop_duplicates(subset=merge_keys, keep='last')
        summary_df = summarize_consensus(df_scored, SHARP_BOOKS, REC_BOOKS)
    else:
        logging.warning("⚠️ apply_blended_sharp_score() returned no rows")
        df_scored = pd.DataFrame()
        summary_df = pd.DataFrame()
    
    return df_scored, df_all_snapshots, summary_df

    
  


def compute_weighted_signal(row, market_weights):
    market = str(row.get('Market', '')).lower()
    total_score = 0
    max_possible = 0

    component_importance = {
        'Sharp_Move_Signal': 2.0,
        'Sharp_Limit_Jump': 2.0,
        'Sharp_Time_Score': 1.5,
        'Sharp_Limit_Total': 0.001  # assumes large values (scale to 0/1 range implicitly)
    }

    for comp, importance in component_importance.items():
        val = row.get(comp)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue

        try:
            if comp == 'Sharp_Limit_Total':
                val_key = str(int(float(val) // 1000 * 1000))  # bucket to nearest 1000
            elif isinstance(val, float) and val.is_integer():
                val_key = str(int(val))
            else:
                val_key = str(val).strip().lower()
            
            weight = market_weights.get(market, {}).get(comp, {}).get(val_key, 0.5)
        except Exception:
            weight = 0.5

        total_score += weight * importance
        max_possible += importance

    return round((total_score / max_possible) * 100, 2) if max_possible else 50.0
    
    
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


    d = df.copy()

    # -------- column resolvers --------
    def _first(colnames, default=None):
        for c in colnames:
            if c in d.columns:
                return c
        return default

    book_col    = _first(['Book', 'Book_Norm', 'Book_Key', 'Bookmaker', 'Bookmaker_Norm'])
    value_col   = _first(['Value', 'Line_Value', 'Spread_Value', 'Total_Value'])
    open_col    = _first(['Open_Value', 'Open_Line_Value', 'OpenValue'])
    event_col   = _first(['Event_Date', 'Snapshot_Date', 'Game_Date', 'feat_Game_Start'], 'Event_Date')
    game_col    = _first(['Game', 'Game_Key', 'feat_Game_Key', 'game_key_clean'], 'Game')
    market_col  = _first(['Market', 'Market_Norm', 'Market_Type'], 'Market')
    outcome_col = _first(['Outcome', 'Outcome_Norm', 'feat_Team'], 'Outcome')

    # score / confidence columns (optional)
    score_col        = _first(['SharpBetScore', 'Blended_Sharp_Score', 'Sharp_Score',
                               'BlendedScore', 'Model_Prob_Blend', 'p_blend',
                               'Calibrated_Prob', 'Sharp_Prob', 'Model_Prob', 'Prob'])
    conf_col         = _first(['Enhanced_Sharp_Confidence_Score', 'Sharp_Confidence_Score',
                               'Confidence_Score'])
    tier_col         = _first(['Sharp_Confidence_Tier', 'Confidence_Tier', 'Tier'])

    # -------- safety guards --------
    # If required columns are missing, create safe defaults
    if book_col is None:
        d['__Book__'] = ''
        book_col = '__Book__'
    if value_col is None:
        d['__Value__'] = np.nan
        value_col = '__Value__'
    if open_col is None:
        d['__Open__'] = np.nan
        open_col = '__Open__'
    for c in (value_col, open_col):
        d[c] = pd.to_numeric(d[c], errors='coerce')

    keys = [event_col, game_col, market_col, outcome_col]

    # -------- aggregate Rec vs Sharp means --------
    rec_mask   = d[book_col].isin(set(REC_BOOKS)) if REC_BOOKS else pd.Series(False, index=d.index)
    sharp_mask = d[book_col].isin(set(SHARP_BOOKS)) if SHARP_BOOKS else pd.Series(False, index=d.index)

    rec_agg = (
        d.loc[rec_mask, keys + [value_col, open_col]]
         .groupby(keys, observed=True)
         .agg(Rec_Book_Consensus=(value_col, 'mean'),
              Rec_Open=(open_col, 'mean'))
         .reset_index()
    )
    sharp_agg = (
        d.loc[sharp_mask, keys + [value_col, open_col]]
         .groupby(keys, observed=True)
         .agg(Sharp_Book_Consensus=(value_col, 'mean'),
              Sharp_Open=(open_col, 'mean'))
         .reset_index()
    )

    summary_df = (
        rec_agg.merge(sharp_agg, on=keys, how='outer')
               .rename(columns={event_col: 'Event_Date',
                                game_col: 'Game',
                                market_col: 'Market',
                                outcome_col: 'Outcome'})
    )

    # preserve your fields
    summary_df['Recommended_Outcome'] = summary_df['Outcome']
    summary_df['Move_From_Open_Rec']   = (summary_df['Rec_Book_Consensus']   - summary_df['Rec_Open']).fillna(0)
    summary_df['Move_From_Open_Sharp'] = (summary_df['Sharp_Book_Consensus'] - summary_df['Sharp_Open']).fillna(0)

    # -------- attach scores / confidence if present --------
    if score_col or conf_col or tier_col:
        cols = ['Event_Date', 'Game', 'Market', 'Outcome']
        extra = d.copy()

        sel = cols.copy()
        if score_col: sel.append(score_col)
        if conf_col:  sel.append(conf_col)
        if tier_col:  sel.append(tier_col)

        sharp_scores = (
            extra[sel]
            .drop_duplicates()
        )

        # rename optional columns to expected output names
        rename_map = {}
        if score_col: rename_map[score_col] = 'SharpBetScore'
        if conf_col:  rename_map[conf_col]  = 'Enhanced_Sharp_Confidence_Score'
        if tier_col:  rename_map[tier_col]  = 'Sharp_Confidence_Tier'
        sharp_scores = sharp_scores.rename(columns=rename_map)

        summary_df = summary_df.merge(sharp_scores, on=cols, how='left')

    # fill if missing (keep your original behavior)
    if 'SharpBetScore' not in summary_df.columns:
        summary_df['SharpBetScore'] = 0.0
    if 'Enhanced_Sharp_Confidence_Score' not in summary_df.columns:
        summary_df['Enhanced_Sharp_Confidence_Score'] = 0.0
    if 'Sharp_Confidence_Tier' not in summary_df.columns:
        summary_df['Sharp_Confidence_Tier'] = '⚠️ Low'

    summary_df[['SharpBetScore', 'Enhanced_Sharp_Confidence_Score']] = (
        summary_df[['SharpBetScore', 'Enhanced_Sharp_Confidence_Score']].fillna(0)
    )
    summary_df['Sharp_Confidence_Tier'] = summary_df['Sharp_Confidence_Tier'].fillna('⚠️ Low')

    return summary_df





        
def detect_market_leaders(df_history, sharp_books, rec_books):
    """
    Memory-lean market-leader detection:
    - Keeps join/group keys as plain strings (not category) to avoid dtype issues
    - Normalizes Book via a tiny unique map (avoids rowwise .apply over full frame)
    - Computes first change times per (Game, Market, Outcome, Book) with a fast numpy helper
    """
    
    if df_history is None or df_history.empty:
        return pd.DataFrame(columns=[
            "Game","Market","Outcome","Book",
            "First_Line_Move_Time","First_Odds_Move_Time",
            "Book_Type","Line_Move_Rank","Odds_Move_Rank",
            "Market_Leader_Line","Market_Leader_Odds","Market_Leader"
        ])

    # ── 0) Skinny slice & types
    need = ["Game","Market","Outcome","Book","Bookmaker","Time","Value"]
    if "Odds_Price" in df_history.columns:
        need.append("Odds_Price")
    df = df_history.loc[:, [c for c in need if c in df_history.columns]].copy()

    # Ensure datetime w/ UTC (NaT ok)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce", utc=True)

    # Normalize string keys (plain string -> safe for merge/groupby)
    for c in ("Game","Market","Outcome","Book","Bookmaker"):
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    if "Book" in df.columns:
        df["Book"] = df["Book"].str.lower()

    # ── 1) Normalize Book using a tiny unique map (no full-frame apply)
    def _norm_book(book, bookmaker):
        # You already have this in your codebase
        return normalize_book_name(book, bookmaker)

    if "Bookmaker" in df.columns:
        uniq = df[["Book","Bookmaker"]].drop_duplicates()
        uniq["Book_norm"] = uniq.apply(lambda r: _norm_book(r["Book"], r["Bookmaker"]), axis=1)
        df = df.merge(uniq, on=["Book","Bookmaker"], how="left")
    else:
        uniq = df[["Book"]].drop_duplicates()
        uniq["Book_norm"] = uniq["Book"].map(lambda b: _norm_book(b, None))
        df = df.merge(uniq, on="Book", how="left")

    df["Book"] = df["Book_norm"].astype("string").str.lower()
    df.drop(columns=["Book_norm"], inplace=True)

    # ── 2) Sort once for deterministic "first-change" logic
    df.sort_values(["Game","Market","Outcome","Book","Time"], inplace=True, kind="mergesort")

    # ── 3) Helper: first change time vs FIRST NON-NULL opener
    def first_change_time_from_open(values: pd.Series, times: pd.Series):
        # values & times are already time-ordered
        v = values.to_numpy()
        t = times.to_numpy()

        if v.size == 0:
            return pd.NaT

        # find first non-null opener
        nn = ~pd.isna(v)
        if not nn.any():
            return pd.NaT
        opener_idx = np.flatnonzero(nn)[0]
        opener = v[opener_idx]

        # find first index AFTER opener where value is non-null and != opener
        # (include opener_idx+1 .. end)
        idx = np.flatnonzero(nn & (v != opener) & (np.arange(v.size) > opener_idx))
        return pd.NaT if idx.size == 0 else pd.Timestamp(t[idx[0]])

    gkeys = ["Game","Market","Outcome","Book"]

    # ── 4) First line-move time (tiny result)
    line_moves = (
        df.groupby(gkeys, observed=True, sort=False)
          .apply(lambda g: first_change_time_from_open(g["Value"], g["Time"]))
          .reset_index(name="First_Line_Move_Time")
    )

    # ── 5) First odds-move time (if present)
    if "Odds_Price" in df.columns:
        odds_moves = (
            df.groupby(gkeys, observed=True, sort=False)
              .apply(lambda g: first_change_time_from_open(g["Odds_Price"], g["Time"]))
              .reset_index(name="First_Odds_Move_Time")
        )
    else:
        odds_moves = pd.DataFrame(columns=gkeys + ["First_Odds_Move_Time"])

    # ── 6) Merge the two tiny tables (outer keeps groups that moved in only one dim)
    first_moves = line_moves.merge(odds_moves, on=gkeys, how="outer")

    # ── 7) Book type mapping (vectorized membership on lowercase)
    sharp_set = set((b or "").lower() for b in (sharp_books or []))
    rec_set   = set((b or "").lower() for b in (rec_books or []))

    first_moves["Book"] = first_moves["Book"].astype("string").str.lower()
    first_moves["Book_Type"] = np.where(
        first_moves["Book"].isin(sharp_set), "Sharp",
        np.where(first_moves["Book"].isin(rec_set), "Rec", "Other")
    )

    # ── 8) Ranks within (Game, Market, Outcome); earlier time -> rank 1
    by_gmo = ["Game","Market","Outcome"]
    first_moves["Line_Move_Rank"] = first_moves.groupby(by_gmo, observed=True)["First_Line_Move_Time"] \
                                               .rank(method="first", ascending=True)
    first_moves["Odds_Move_Rank"] = first_moves.groupby(by_gmo, observed=True)["First_Odds_Move_Time"] \
                                               .rank(method="first", ascending=True)

    # ── 9) Leaders: sharp book that moved first on line and/or odds
    is_sharp = first_moves["Book_Type"].eq("Sharp")
    first_moves["Market_Leader_Line"] = is_sharp & first_moves["Line_Move_Rank"].eq(1.0)
    first_moves["Market_Leader_Odds"] = is_sharp & first_moves["Odds_Move_Rank"].eq(1.0)
    first_moves["Market_Leader"] = is_sharp & (
        first_moves["Line_Move_Rank"].eq(1.0) | first_moves["Odds_Move_Rank"].eq(1.0)
    )

    # ── 10) Trim/Order columns
    cols = [
        "Game","Market","Outcome","Book",
        "First_Line_Move_Time","First_Odds_Move_Time",
        "Book_Type","Line_Move_Rank","Odds_Move_Rank",
        "Market_Leader_Line","Market_Leader_Odds","Market_Leader"
    ]
    out = first_moves.loc[:, [c for c in cols if c in first_moves.columns]]

    # Optional: if you want memory wins after the merge, cast *now* (safe):
    # for c in ("Sport","Market","Outcome","Book"):
    #     if c in out.columns:
    #         out[c] = out[c].astype("category")

    return out


def detect_cross_market_sharp_support(
    df_moves,
    SHARP_BOOKS,
    game_col="Game_Key",
    outcome_col="Outcome_Norm",
    sharp_col=None,         # optional: let caller specify the signal column explicitly
):
    """
    Compute cross-market sharp support with minimal allocations and schema-robustness.
    - Resolves book/sharp/market columns defensively
    - No string-concat keys
    - One groupby + one merge
    """
    import numpy as np
    import pandas as pd

    df = df_moves  # operate on provided frame

    # --- 0) Preconditions on join keys
    if game_col not in df.columns or outcome_col not in df.columns:
        raise KeyError(f"Missing join keys: need '{game_col}' and '{outcome_col}' in df_moves")

    # --- 1) Resolve book column
    def _resolve_book_col(d: pd.DataFrame) -> str:
        for c in ("Book", "Book_Norm", "Book_Key", "Bookmaker", "Bookmaker_Norm"):
            if c in d.columns:
                return c
        raise KeyError("Could not find a book column among ['Book','Book_Norm','Book_Key','Bookmaker','Bookmaker_Norm'].")

    book_col = _resolve_book_col(df)
    sharp_books = set(SHARP_BOOKS)

    # --- 2) Resolve market column
    def _resolve_market_col(d: pd.DataFrame) -> str:
        for c in ("Market", "Market_Norm", "Market_Type"):
            if c in d.columns:
                return c
        # If truly missing, create a dummy single-market column to avoid crash
        d["__MarketDummy__"] = "unknown"
        return "__MarketDummy__"

    market_col = _resolve_market_col(df)

    # --- 3) Resolve sharp signal boolean
    def _as_bool(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s.fillna(False)
        # numeric or string-y truthy
        try:
            return s.fillna(0).astype(float) > 0
        except Exception:
            # last resort: string truthiness
            return s.fillna("").astype(str).str.lower().isin(("true", "t", "yes", "y", "1"))

    def _resolve_sharp_series(d: pd.DataFrame) -> pd.Series:
        # If caller specified a column name and it's present, use it.
        if sharp_col and sharp_col in d.columns:
            return _as_bool(d[sharp_col])

        # Common candidates (old & new names)
        candidates = [
            "Sharp_Move_Signal", "SharpMove_Signal", "Sharp_Move_Flag", "Has_Sharp_Signal",
            "SharpSignal", "Sharp_Signal", "Active_Sharp_Signals", "Active_Sharp_Signal_Count",
        ]
        for c in candidates:
            if c in d.columns:
                return _as_bool(d[c])

        # Heuristic fallback: any positive magnitude implies a sharp move on that row
        mag_cols = [c for c in d.columns if c.startswith("SharpMove_Magnitude_")]
        if not mag_cols:
            # try looser match
            mag_cols = [c for c in d.columns if ("sharp" in c.lower() and "magnitude" in c.lower())]

        if mag_cols:
            # rowwise "any > 0"
            return (d[mag_cols].fillna(0).astype(float) > 0).any(axis=1)

        # If we reach here, no signal available; return all False to keep pipeline alive.
        return pd.Series(False, index=d.index)

    sharp_series = _resolve_sharp_series(df)

    # --- 4) Build mask of "sharp rows" on sharp books
    mask = df[book_col].isin(sharp_books) & sharp_series

    # --- 5) Group & aggregate per (game, outcome)
    cols = [game_col, outcome_col, market_col, book_col]
    gb = (
        df.loc[mask, cols]
          .groupby([game_col, outcome_col], observed=True)
          .agg(
              CrossMarketSharpSupport=(market_col, "nunique"),
              Unique_Sharp_Books=(book_col, "nunique"),
          )
          .reset_index()
    )

    # --- 6) Merge back onto df and finalize types
    if not gb.empty:
        gb["CrossMarketSharpSupport"] = gb["CrossMarketSharpSupport"].astype("int16")
        gb["Unique_Sharp_Books"]      = gb["Unique_Sharp_Books"].astype("int16")

    df = df.merge(gb, on=[game_col, outcome_col], how="left")

    if "CrossMarketSharpSupport" not in df.columns:
        df["CrossMarketSharpSupport"] = 0
    if "Unique_Sharp_Books" not in df.columns:
        df["Unique_Sharp_Books"] = 0

    df["CrossMarketSharpSupport"] = df["CrossMarketSharpSupport"].fillna(0).astype("int16")
    df["Unique_Sharp_Books"]      = df["Unique_Sharp_Books"].fillna(0).astype("int16")

    # --- 7) Final composite flag
    df["Is_Reinforced_MultiMarket"] = (
        (df["CrossMarketSharpSupport"] >= 2) | (df["Unique_Sharp_Books"] >= 2)
    )

    return df
import io, gzip, logging, pickle, re, time

from google.cloud import storage

# Try cloudpickle for robustness
try:
    import cloudpickle as cp
except Exception:
    cp = None

# ---- Shim for old IsoWrapper pickles (kept, single definition) ----
class _IsoWrapperShim:
    def __init__(self, model=None, calibrator=None):
        self.model = model
        self.calibrator = calibrator
    def predict_proba(self, X):
        # old pickles called IsoWrapper.predict_proba(X) directly
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)
            p1 = p[:, 1] if getattr(p, "ndim", 1) == 2 else np.asarray(p).ravel()
        else:
            p1 = np.asarray(self.model.predict(X)).ravel()
        if self.calibrator is not None:
            # many old scikit calibrators expose .transform
            trans = getattr(self.calibrator, "transform", None)
            if callable(trans):
                p1 = trans(p1)
        p1 = np.clip(p1, 1e-9, 1 - 1e-9)
        return np.column_stack([1 - p1, p1])

class _RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map any IsoWrapper reference to our shim
        if name == "IsoWrapper":
            return _IsoWrapperShim
        return super().find_class(module, name)

def _safe_loads(b: bytes):
    bio = io.BytesIO(b)
    # First try with our remapping (for old IsoWrapper pickles)
    try:
        return _RenamingUnpickler(bio).load()
    except Exception:
        pass
    # Then try cloudpickle (if available)
    if cp is not None:
        try:
            return cp.loads(b)
        except Exception:
            pass
    # Finally, vanilla pickle
    bio.seek(0)
    return pickle.loads(bio.read())

def _slug(s: str) -> str:
    s = str(s).strip().lower()
    return re.sub(r"[^\w.-]+", "_", s).strip("_.")

def _maybe_decompress(name: str, b: bytes) -> bytes:
    import gzip, bz2
    if name.endswith(".gz"):  return gzip.decompress(b)
    if name.endswith(".bz2"): return bz2.decompress(b)
    return b

def load_model_from_gcs(
    sport: str,
    market: str,
    bucket_name: str = "sharp-models",
    project: str | None = None,
    use_latest: bool = False,        # set True if you saved timestamped files
) -> dict | None:
    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(bucket_name)

    sport_l, market_l = _slug(sport), _slug(market)
    base_prefix = f"sharp_win_model_{sport_l}_{market_l}"

    if use_latest:
        # find most recent object by prefix
        blobs = list(bucket.list_blobs(prefix=base_prefix))
        if not blobs:
            logging.warning(f"⚠️ No artifacts with prefix {base_prefix} in gs://{bucket_name}")
            return None
        # pick the newest by updated time
        blob = sorted(blobs, key=lambda b: b.updated or b.time_created)[-1]
    else:
        # static filename (legacy)
        blob = bucket.blob(f"{base_prefix}.pkl")

    try:
        content = blob.download_as_bytes()
    except Exception as e:
        logging.warning(f"⚠️ No artifact: gs://{bucket_name}/{blob.name} ({e})")
        return None

    try:
        payload = _safe_loads(_maybe_decompress(blob.name, content))
        logging.info(f"✅ Loaded artifact: gs://{bucket_name}/{blob.name}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to unpickle {blob.name}: {e}")
        return None

    if not isinstance(payload, dict):
        logging.error(f"Unexpected payload type in {blob.name}: {type(payload)}")
        return None

    # ---- Normalize keys across new/old formats ----
    # Models
    model_logloss = payload.get("model_logloss") or payload.get("model")
    model_auc     = payload.get("model_auc")

    # Calibrators:
    # NEW: payload["calibrator"] = {"iso_blend": <IsotonicRegression>}
    # OLD: payload["calibrator_logloss"], payload["calibrator_auc"] (IsoWrapper, etc.)
    calibrator_bundle = payload.get("calibrator")
    # ---- inside utils.py: load_model_from_gcs(...) ----
    # after:
    #   model_logloss = payload.get("model_logloss") or payload.get("model")
    #   model_auc     = payload.get("model_auc")
    
    iso_blend = (
        payload.get("iso_blend")
        or (payload.get("calibrator", {}) or {}).get("iso_blend")
        or (payload.get("calibrator", {}) or {}).get("iso")  # optional legacy
    )
    
    # flip flag: support a few legacy names
    flip_flag = bool(
        payload.get("flip_flag", False)
        or payload.get("global_flip", False)
        or (payload.get("polarity", +1) == -1)
    )
    
    best_w = float(payload.get("best_w", 1.0))
    feature_cols = payload.get("feature_cols") or []
    team_feature_map = payload.get("team_feature_map")
    book_reliability_map = payload.get("book_reliability_map")
    
    bundle = {
        "model_logloss": model_logloss,
        "model_auc": model_auc,
        "best_w": best_w,
        "feature_cols": feature_cols,
    
        # ✅ this is what predict_bundle_proba() expects
        "calibrator": {"iso_blend": iso_blend},
    
        # ✅ carry flip through to inference
        "flip_flag": flip_flag,
    
        # (optional) keep convenience copies too
        "iso_blend": iso_blend,
    
        "team_feature_map": team_feature_map if isinstance(team_feature_map, pd.DataFrame) else pd.DataFrame(),
        "book_reliability_map": book_reliability_map if isinstance(book_reliability_map, pd.DataFrame) else pd.DataFrame(),
    }
    return bundle






DEFAULT_MOVES_VIEW = "sharp_data.moves_with_features_merged"

def read_recent_sharp_moves(
    hours: int = 120,
    table: str = DEFAULT_MOVES_VIEW,
    pregame_only: bool = True
) -> pd.DataFrame:
    """
    Load recent sharp moves (enriched with team features) from BigQuery.

    - `table` should usually be the merged view:
        sharp_data.moves_with_features_merged
      (no totals duplication; has home_/away_ feature columns)

    - Set `pregame_only=False` if you also want in-play/post-game rows.
    """
    try:
        client = bq_client  # reuse your existing client

        where_clauses = [
            "Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @hours HOUR)"
        ]
        if pregame_only:
            where_clauses += [
                "COALESCE(Pre_Game, TRUE)",
                "Game_Start IS NOT NULL",
                "Time IS NOT NULL",
                "TIMESTAMP_DIFF(Game_Start, Time, SECOND) >= 0"
            ]

        query = f"""
            SELECT *
            FROM `{table}`
            WHERE {' AND '.join(where_clauses)}
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("hours", "INT64", hours)]
        )

        df = client.query(query, job_config=job_config).to_dataframe(create_bqstorage_client=True)

        # Normalize timestamps to pandas UTC datetimes
        for col in ("Commence_Hour", "Game_Start", "Time", "Snapshot_Timestamp"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        print(f"✅ Loaded {len(df):,} rows from {table} (last {hours}h, pregame_only={pregame_only})")
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
            'Game_Key', 'Bookmaker','Book', 'Market', 'Outcome', 'Limit', 'Ref_Sharp_Value',
            'Sharp_Move_Signal', 'Sharp_Limit_Jump', 
            'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
            'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
            'Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score',
            'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT', 'Scored', 'Snapshot_Timestamp',
            'Sport', 'Value', 'First_Line_Value', 
            'Line_Delta', 'Model_Prob_Diff', 'Direction_Aligned',
            'Unique_Sharp_Books', 'Merge_Key_Short',
            'Line_Magnitude_Abs', 'Line_Move_Magnitude',
            'Is_Home_Team_Bet', 'Is_Favorite_Bet', 'High_Limit_Flag',
            'Home_Team_Norm', 'Away_Team_Norm', 'Commence_Hour',
            'Model_Sharp_Win_Prob', 'Odds_Price', 'Implied_Prob',
            'First_Odds', 'First_Imp_Prob', 'Odds_Shift', 'Implied_Prob_Shift',
            'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds','Value_Reversal_Flag','Odds_Reversal_Flag', 
    
            # ✅ Resistance logic additions
            'Late_Game_Steam_Flag', 
            'SharpMove_Magnitude_Overnight_VeryEarly', 'SharpMove_Magnitude_Overnight_MidRange',
            'SharpMove_Magnitude_Overnight_LateGame', 'SharpMove_Magnitude_Overnight_Urgent',
            'SharpMove_Magnitude_Early_VeryEarly', 'SharpMove_Magnitude_Early_MidRange',
            'SharpMove_Magnitude_Early_LateGame', 'SharpMove_Magnitude_Early_Urgent',
            'SharpMove_Magnitude_Midday_VeryEarly', 'SharpMove_Magnitude_Midday_MidRange',
            'SharpMove_Magnitude_Midday_LateGame', 'SharpMove_Magnitude_Midday_Urgent',
            'SharpMove_Magnitude_Late_VeryEarly', 'SharpMove_Magnitude_Late_MidRange',
            'SharpMove_Magnitude_Late_LateGame', 'SharpMove_Magnitude_Late_Urgent','SharpMove_Timing_Dominant',
            'SharpMove_Timing_Magnitude', 
                # 🎯 Odds timing magnitude (odds) — ✅ NEW ADDITIONS
            'Odds_Move_Magnitude',
            'OddsMove_Magnitude_Overnight_VeryEarly', 'OddsMove_Magnitude_Overnight_MidRange',
            'OddsMove_Magnitude_Overnight_LateGame', 'OddsMove_Magnitude_Overnight_Urgent',
            'OddsMove_Magnitude_Early_VeryEarly', 'OddsMove_Magnitude_Early_MidRange',
            'OddsMove_Magnitude_Early_LateGame', 'OddsMove_Magnitude_Early_Urgent',
            'OddsMove_Magnitude_Midday_VeryEarly', 'OddsMove_Magnitude_Midday_MidRange',
            'OddsMove_Magnitude_Midday_LateGame', 'OddsMove_Magnitude_Midday_Urgent',
            'OddsMove_Magnitude_Late_VeryEarly', 'OddsMove_Magnitude_Late_MidRange',
            'OddsMove_Magnitude_Late_LateGame', 'OddsMove_Magnitude_Late_Urgent',
            'Net_Line_Move_From_Opening',
            'Abs_Line_Move_From_Opening',
            'Net_Odds_Move_From_Opening',
            'Abs_Odds_Move_From_Opening',
         
    
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
        df['Odds_Price'] = pd.to_numeric(df['Odds_Price'], errors='coerce')
        df['Implied_Prob'] = pd.to_numeric(df['Implied_Prob'], errors='coerce')
        df['Odds_Shift'] = pd.to_numeric(df.get('Odds_Shift'), errors='coerce')
        df['Implied_Prob_Shift'] = pd.to_numeric(df.get('Implied_Prob_Shift'), errors='coerce')

        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='replace' if force_replace else 'append')
        logging.info(f"✅ Uploaded {len(df)} rows to {table}")
    except Exception as e:
        logging.exception(f"❌ Failed to upload to {table}")

def normalize_sport(sport_key: str) -> str:
    s = str(sport_key).strip().lower()
    mapping = {
        "baseball_mlb": "MLB",
        "americanfootball_nfl": "NFL",
        "americanfootball_ncaaf": "NCAAF",
        "basketball_nba": "NBA",
        "basketball_wnba": "WNBA",
        "canadianfootball_cfl": "CFL",
        "basketball_ncaab": "NCAAB",
        "basketball_ncaam": "NCAAB",   # ✅ new
        # Friendly aliases
        "mlb": "MLB", "nfl": "NFL", "ncaaf": "NCAAF",
        "nba": "NBA", "wnba": "WNBA", "cfl": "CFL",
        "ncaab": "NCAAB", "ncaam": "NCAAB",  # ✅ new
    }
    return mapping.get(s, s.upper() or "MLB")


import time, datetime as dt
from typing import Optional

_SCORED_KEYS_CACHE: dict[tuple[str, Optional[str]], tuple[float, set[str]]] = {}
_SCORED_KEYS_TTL_SEC = 600  # 10 minutes

def _bucket_ts(ts: dt.datetime, minutes=10) -> dt.datetime:
    ts = ts.replace(second=0, microsecond=0)
    return ts.replace(minute=(ts.minute // minutes) * minutes)

# at top-level (module scope)
from cachetools import TTLCache
_scored_keys_cache = TTLCache(maxsize=16, ttl=300)  # 5 minutes

def get_scored_keys_cached(bq_client, since_ts, sport_label=None):
    key = (since_ts.replace(microsecond=0), sport_label and sport_label.upper())
    if key in _scored_keys_cache:
        rows = _scored_keys_cache[key]
        logger.info("⏱️ scored_keys BQ fetch skipped (cache hit %d keys)", len(rows))
        return rows

    sql = """
    SELECT DISTINCT Merge_Key_Short
    FROM `sharplogger.sharp_data.sharp_scores_full`
    WHERE Snapshot_Timestamp >= @since_ts
    -- uncomment to scope by sport:
    -- AND UPPER(Sport) = @sport
    """
    params = [bigquery.ScalarQueryParameter("since_ts", "TIMESTAMP", since_ts)]
    #if sport scoping:
    params.append(bigquery.ScalarQueryParameter("sport", "STRING", sport_label.upper()))

    t0 = time.time()
    df = bq_client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=params,
                                           priority=bigquery.QueryPriority.INTERACTIVE)
    ).to_dataframe()
    logger.info("⏱️ scored_keys BQ fetch took %.2fs (fetched %d rows)", time.time() - t0, len(df))
    vals = set(df['Merge_Key_Short'].dropna())
    _scored_keys_cache[key] = vals
    return vals

def _is_sport(df: pd.DataFrame, sport_label: str) -> pd.Series:
    aliases = {s.upper() for s in SPORT_ALIASES.get(sport_label.upper(), [sport_label.upper()])}
    return df["Sport"].astype(str).str.upper().isin(aliases)


def fetch_scores_and_backtest(sport_key, df_moves=None, days_back=3, api_key=API_KEY, trained_models=None):
    expected_label = [k for k, v in SPORTS.items() if v == sport_key]
    sport_label = expected_label[0].upper() if expected_label else "NBA"
    track_feature_evolution = False
    sport = normalize_sport(sport_key)
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
    def is_completed(game: dict) -> bool:
        if not game.get("completed", False):
            return False
        scores = game.get("scores")
        if not isinstance(scores, list) or len(scores) < 2:
            return False
    
        # Normalize names and keep the last entry per team (some feeds repeat updates)
        by_team = {}
        for s in scores:
            if not isinstance(s, dict): 
                continue
            nm = normalize_team(s.get("name",""))
            val = s.get("score")
            if val is None:
                continue
            try:
                by_team[nm] = float(str(val).replace(",","").strip())
            except Exception:
                return False  # non-numeric
    
        # Need exactly 2 distinct teams with numeric scores
        return len(by_team) >= 2 and all(v is not None for v in by_team.values())
    
    completed_games = [g for g in games if is_completed(g)]
    logging.info("✅ Completed games: %d", len(completed_games))

    score_rows = []
    for game in completed_games:
        raw_home = game.get("home_team", "")
        raw_away = game.get("away_team", "")
        # Use the SAME normalization (lower+strip)
        home = normalize_team(raw_home).strip().lower()
        away = normalize_team(raw_away).strip().lower()

        # Floor to hour in UTC; string will include +00:00 when cast to str
        commence_hour = pd.to_datetime(game.get("commence_time"), utc=True).floor("h")
        merge_key = build_merge_key(home, away, commence_hour)
        # Collapse provider scores to numeric
        by_team = {}
        for s in (game.get("scores") or []):
            nm  = normalize_team(s.get("name","")).strip().lower()
            val = s.get("score")
            try:
                by_team[nm] = float(str(val).replace(",","").strip())
            except Exception:
                pass

        if home not in by_team or away not in by_team:
            logging.warning(
                "⚠️ Skip: name mismatch home=%s away=%s present=%s",
                home, away, list(by_team.keys())
            )
            continue

        score_rows.append({
            "Home_Team": home,
            "Away_Team": away,
            "Game_Start": commence_hour,  # keep tz-aware
            "Score_Home_Score": int(by_team[home]),
            "Score_Away_Score": int(by_team[away]),
            "Inserted_Timestamp": pd.Timestamp.utcnow(),
            "Source": "oddsapi",
            "Sport": sport,
            # 🔑 Canonical Merge_Key_Short: teams + str(Commence_Hour) (includes +00:00)
            "Merge_Key_Short": merge_key,

        })

    if not score_rows:
        logging.warning("⚠️ After validation, zero usable completed games.")
        return pd.DataFrame()

    df_scores = pd.DataFrame(score_rows)
    df_scores = df_scores[df_scores['Merge_Key_Short'].notna()].copy()
    
    # Ensure types/casing are stable
    # 0) drop rows with null keys BEFORE string ops (avoid making "nan")
    df_scores = df_scores[df_scores['Merge_Key_Short'].notna()].copy()
    
    # 1) normalize the key (matches master if you used build_merge_key for construction)
    df_scores['Merge_Key_Short'] = df_scores['Merge_Key_Short'].str.strip().str.lower()
    
    # 2) coerce scores to numeric
    df_scores['Score_Home_Score'] = pd.to_numeric(df_scores['Score_Home_Score'], errors='coerce')
    df_scores['Score_Away_Score'] = pd.to_numeric(df_scores['Score_Away_Score'], errors='coerce')
    
    # 3) make sure Inserted_Timestamp is real datetimes for sorting
    df_scores['Inserted_Timestamp'] = pd.to_datetime(df_scores['Inserted_Timestamp'], utc=True, errors='coerce')
    
    # 4) stable latest-per-game dedup, then require non-null scores
    df_scores = (
        df_scores.sort_values('Inserted_Timestamp')
                 .drop_duplicates(subset='Merge_Key_Short', keep='last')
                 .dropna(subset=['Score_Home_Score', 'Score_Away_Score'])
    )


    # === 3. Upload scores to `game_scores_final` ===
    # === 3. Upload scores to `game_scores_final` ===
    # === 3) Write finals to game_scores_final (canonical schema) ===
    # === 3) Write finals to game_scores_final (canonical schema) ===
    try:
        schema_cols = [
            'Merge_Key_Short','Home_Team','Away_Team','Game_Start',
            'Score_Home_Score','Score_Away_Score','Source','Inserted_Timestamp','Sport'
        ]
    
        out = (df_scores if df_scores is not None else pd.DataFrame()).copy()
        if out.empty:
            logging.info("ℹ️ No df_scores rows available — skipping finals upload.")
        else:
            # Normalize core fields
            out['Merge_Key_Short'] = out['Merge_Key_Short'].astype(str).str.strip().str.lower()
            out['Home_Team']       = out['Home_Team'].astype(str)
            out['Away_Team']       = out['Away_Team'].astype(str)
            out['Source']          = out['Source'].astype(str)
            out['Sport']           = out['Sport'].astype(str)
    
            # Timestamps → UTC
            out['Game_Start']         = pd.to_datetime(out['Game_Start'], utc=True, errors='coerce')
            out['Inserted_Timestamp'] = pd.to_datetime(out['Inserted_Timestamp'], utc=True, errors='coerce')
    
            # Scores → float (BQ schema)
            out['Score_Home_Score'] = pd.to_numeric(out['Score_Home_Score'], errors='coerce').astype(float)
            out['Score_Away_Score'] = pd.to_numeric(out['Score_Away_Score'], errors='coerce').astype(float)
    
            # Ensure schema columns exist and order
            for c in schema_cols:
                if c not in out.columns:
                    out[c] = pd.NA
            out = out[schema_cols]
    
            # Drop rows missing critical fields
            out = out.dropna(subset=['Merge_Key_Short','Game_Start','Score_Home_Score','Score_Away_Score'])
            if out.empty:
                logging.info("ℹ️ After cleaning, no complete finals rows to write.")
            else:
                # Normalized key for de-dup
                out['mks_norm'] = out['Merge_Key_Short'].astype(str).str.strip().str.lower()
    
                # Fetch existing keys (normalized the same way)
                existing = bq_client.query("""
                    SELECT DISTINCT LOWER(TRIM(CAST(Merge_Key_Short AS STRING))) AS mks
                    FROM `sharplogger.sharp_data.game_scores_final`
                """).to_dataframe()
                existing_keys = set(existing['mks'].dropna().astype(str))
    
                # Only new rows
                to_write = out.loc[~out['mks_norm'].isin(existing_keys), schema_cols].copy()
    
                logging.info(
                    "⛳ Finals total (clean): %d | Already in table: %d | New to write: %d",
                    len(out), len(out) - len(to_write), len(to_write)
                )
    
                if not to_write.empty:
                    to_gbq(
                        to_write,
                        'sharp_data.game_scores_final',
                        project_id=GCP_PROJECT_ID,
                        if_exists='append'
                    )
                    logging.info("✅ Wrote %d rows to sharp_data.game_scores_final", len(to_write))
                else:
                    logging.info("ℹ️ No new rows to write to sharp_data.game_scores_final")
    
    except Exception:
        logging.exception("❌ Failed to upload finals to game_scores_final")
    
        # Function to optimize and process in chunks


    # Function to process chunks of data (sorting and deduplication)
    # Function to process chunks of data (sorting, deduplication, and memory management)
    def process_chunk(df_chunk):
        # ✅ Normalize string columns first
        for col in ['Game_Key', 'Market', 'Outcome', 'Bookmaker']:
            df_chunk.loc[:, col] = df_chunk[col].astype(str).str.strip().str.lower()
    
        # ✅ Convert to categorical for memory efficiency (after normalization)
        for col in ['Game_Key', 'Market', 'Outcome', 'Bookmaker']:
            df_chunk.loc[:, col] = df_chunk[col].astype('category')
    
        # ✅ Sort by latest snapshot to keep most recent
        df_chunk = df_chunk.sort_values(by='Snapshot_Timestamp', ascending=False)
    
        # ✅ Deduplicate per game/market/outcome/bookmaker
        df_chunk = df_chunk.drop_duplicates(
            subset=['Game_Key', 'Market', 'Outcome', 'Bookmaker'],
            keep='first'
        )
    
        gc.collect()
        return df_chunk
        

    
    def calc_cover(df):
        """
        Calculate SHARP_HIT_BOOL and SHARP_COVER_RESULT for spreads, totals, and h2h.
        Pushes are excluded from scoring.
    
        Returns:
            DataFrame with columns:
                - SHARP_COVER_RESULT: 'Win' or 'Loss'
                - SHARP_HIT_BOOL: 1 or 0
        """
        df = df.copy()
    
        # Defensive normalization
        df['Market_Lower'] = df['Market'].str.lower().fillna('')
        df['Outcome_Lower'] = df['Outcome'].str.lower().fillna('')
    
        # Scores and line
        home = pd.to_numeric(df['Score_Home_Score'], errors='coerce')
        away = pd.to_numeric(df['Score_Away_Score'], errors='coerce')
        value = pd.to_numeric(df['Value'], errors='coerce')
    
        # Booleans
        is_totals = df['Market_Lower'] == 'totals'
        is_spreads = df['Market_Lower'] == 'spreads'
        is_h2h = df['Market_Lower'] == 'h2h'
        is_under = df['Outcome_Lower'] == 'under'
        is_over = df['Outcome_Lower'] == 'over'
        is_home_side = df['Outcome'] == df['Home_Team']
        is_away_side = df['Outcome'] == df['Away_Team']
    
        # Totals logic
        total_score = home + away
        totals_hit = np.where(
            is_under, total_score < value,
            np.where(is_over, total_score > value, False)
        )
        push_totals = total_score == value
    
        # Spreads logic
        spread_margin = np.where(is_home_side, home - away, away - home)
        spread_hit = spread_margin > -value
        push_spreads = spread_margin == -value
    
        # H2H logic
        winner = np.where(home > away, df['Home_Team'],
                 np.where(away > home, df['Away_Team'], 'tie'))
        h2h_hit = df['Outcome'] == winner
    
        # Push filter
        is_push = (is_totals & push_totals) | (is_spreads & push_spreads)
    
        # Final coverage logic
        hit = np.where(
            is_totals, totals_hit,
            np.where(is_spreads, spread_hit,
                     np.where(is_h2h, h2h_hit, False))
        )
    
        # Assign final result (excluding pushes)
        result_df = pd.DataFrame({
            'SHARP_COVER_RESULT': np.where(hit, 'Win', 'Loss'),
            'SHARP_HIT_BOOL': hit.astype(int)
        })
    
        # Remove push rows
        result_df.loc[is_push, 'SHARP_COVER_RESULT'] = None
        result_df.loc[is_push, 'SHARP_HIT_BOOL'] = None
    
        return result_df
    # === 4. Load recent sharp picks
   
    
    # === Filter out games already scored in sharp_scores_full
   
    # === Filter out games already scored in sharp_scores_full (DO THIS FIRST) ===
    two_weeks_ago = dt.datetime.utcnow() - dt.timedelta(days=5)
    
    if not track_feature_evolution:
        already_scored = get_scored_keys_cached(bq_client, two_weeks_ago, sport_label)
        df_scores_needed = df_scores[~df_scores['Merge_Key_Short'].isin(already_scored)]
        logging.info("✅ Remaining unscored completed games: %d", len(df_scores_needed))
        if df_scores_needed.empty:
            logging.info("⏭️ Nothing to score — skipping snapshot loads.")
            return pd.DataFrame()
    else:
        df_scores_needed = df_scores.copy()
        logging.info("📈 Time-series mode enabled: Skipping scored-key filter to allow resnapshots")
    # 🔒 Keep only the target sport (prevents cross-sport joins)
    if "Sport" in df_scores_needed.columns:
        before = len(df_scores_needed)
        df_scores_needed = df_scores_needed[_is_sport(df_scores_needed, sport_label)].copy()
        logging.info("🎯 %s: df_scores_needed filtered by sport: %d → %d",
                     sport_label, before, len(df_scores_needed))
    # Only now load snapshots / df_master / df_first, etc.
    df_master = read_recent_sharp_master_cached(hours=72)
    df_master = build_game_key(df_master)
    # 🎯 sport filter on master
    if "Sport" in df_master.columns:
        m0 = len(df_master)
        df_master = df_master[_is_sport(df_master, sport_label)].copy()
        
    logging.info("🎯 %s: df_master filtered by sport: %d → %d", sport_label, m0, len(df_master))
    # Ensure Merge_Key_Short exists AFTER loading
    if 'Merge_Key_Short' not in df_master.columns:
        df_master = build_game_key(df_master)
    if 'Merge_Key_Short' not in df_scores_needed.columns:
        df_scores_needed = build_game_key(df_scores_needed)
    
    # Debugging: Log the columns of the DataFrames after build_game_key
    # === Log schemas for debug
    logging.info(f"After build_game_key - df_scores_needed columns: {df_scores_needed.columns.tolist()}")
    logging.info(f"After build_game_key - df_master columns: {df_master.columns.tolist()}")
    
    # === Track memory usage
    # === Track memory usage
    process = psutil.Process(os.getpid())
    logging.info(f"Memory before snapshot load: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # === 1) Load recent master history (with openers)
    df_all_snapshots = read_recent_sharp_master_cached(hours=72)
    log_memory("AFTER read_recent_sharp_master_cached")
    # 🎯 sport filter on snapshots (so df_first isn't polluted by CFL/NFL rows)
    if "Sport" in df_all_snapshots.columns:
        s0 = len(df_all_snapshots)
        df_all_snapshots = df_all_snapshots[_is_sport(df_all_snapshots, sport_label)].copy()
        logging.info("🎯 %s: df_all_snapshots filtered by sport: %d → %d",
                     sport_label, s0, len(df_all_snapshots))
    # === (Optional) Latest-line view if you need it downstream.
    df_all_snapshots_filtered = pd.DataFrame()
    try:
        # Only run if process_chunk exists and is callable
        if "process_chunk" in globals() and callable(process_chunk):
            parts = []
            for start in range(0, len(df_all_snapshots), 10_000):
                parts.append(process_chunk(df_all_snapshots.iloc[start:start + 10_000]))
            if parts:
                df_all_snapshots_filtered = pd.concat(parts, ignore_index=True)
            log_memory("AFTER building df_all_snapshots_filtered")
        else:
            logging.info("ℹ️ process_chunk not available; skipping df_all_snapshots_filtered.")
    except Exception as e:
        logging.warning(f"⚠️ Skipping df_all_snapshots_filtered due to error: {e}")
    
   
   
    # === 2) Build df_first directly from Open_* in master
    need_cols = ["Game_Key","Market","Outcome","Bookmaker","Open_Value","Open_Odds","Sport"]  # 👈 keep Sport
    missing = [c for c in need_cols if c not in df_all_snapshots.columns]
    if missing:
        logging.warning(f"⚠️ Cannot build df_first — missing columns: {missing}")
        df_first = pd.DataFrame(columns=[
            "Game_Key","Market","Outcome","Bookmaker",
            "First_Line_Value","First_Odds","First_Imp_Prob","Sport"
        ])
    else:
        df_first = (
            df_all_snapshots.loc[:, need_cols]
            .dropna(subset=["Game_Key","Market","Outcome","Bookmaker"])
            .drop_duplicates(subset=["Game_Key","Market","Outcome","Bookmaker"], keep="first")
            .rename(columns={"Open_Value":"First_Line_Value","Open_Odds":"First_Odds"})
            .reset_index(drop=True)
            .copy()
        )
        df_first["First_Imp_Prob"] = df_first["First_Odds"].map(calc_implied_prob)
        # 🎯 and filter again just in case
        if "Sport" in df_first.columns:
            f0 = len(df_first)
            df_first = df_first[_is_sport(df_first, sport_label)].copy()
            logging.info("🎯 %s: df_first filtered by sport: %d → %d", sport_label, f0, len(df_first))
    
        logging.info("📋 Sample df_first:\n" + df_first[["Game_Key","Market","Outcome","Bookmaker","First_Line_Value"]].head(10).to_string(index=False))
    # === 3) Normalize join keys
    key_cols = ["Game_Key","Market","Outcome","Bookmaker"]
    for _df in (df_master, df_first):
        for c in key_cols:
            if c in _df.columns:
                _df[c] = _df[c].astype(str).str.strip().str.lower()
    for c in key_cols:
        if c in df_first.columns: df_first[c] = df_first[c].astype("category")
        if c in df_master.columns: df_master[c] = df_master[c].astype("category")
    
    # === 4) Prepare scores (dedupe)
    df_scores = df_scores[["Merge_Key_Short","Score_Home_Score","Score_Away_Score"]].copy()
    df_scores["Merge_Key_Short"] = df_scores["Merge_Key_Short"].astype("category")
    if "Inserted_Timestamp" in df_scores.columns:
        df_scores = (df_scores.sort_values("Inserted_Timestamp")
                               .drop_duplicates(subset="Merge_Key_Short", keep="last"))
    else:
        df_scores = df_scores.drop_duplicates(subset="Merge_Key_Short", keep="last")
    
    # === 5) Diagnostics + merge
    overlap = (df_master[key_cols].drop_duplicates()
               .merge(df_first[key_cols].drop_duplicates(), on=key_cols, how="inner"))
    logging.info(f"🧪 Join key overlap: {len(overlap)} / {df_master[key_cols].drop_duplicates().shape[0]}")
    
    df_master = df_master.merge(df_first, on=key_cols, how="left")
    log_memory("AFTER merge with df_first")
    
    # Remove any preexisting score cols to avoid suffix fights, then merge scores
    df_master.drop(columns=["Score_Home_Score","Score_Away_Score"], errors="ignore", inplace=True)
    df_master = df_master.merge(df_scores, on="Merge_Key_Short", how="inner")
    log_memory("AFTER merge with df_scores")
    
    # === 6) Finalize working df
    df = df_master
    logging.info(f"✅ Final df columns before scoring: {list(df.columns)}")
    logging.info(f"df shape after merge: {df.shape}")
    
    if df.empty:
        logging.warning("ℹ️ No rows after merge with scores — skipping backtest scoring.")
        return pd.DataFrame()
    
    # Ensure Merge_Key_Short present via Game_Key map if needed
    if "Merge_Key_Short" in df_master.columns:
        key_map = (df_master.drop_duplicates(subset=["Game_Key"])
                            [["Game_Key","Merge_Key_Short"]]
                            .set_index("Game_Key")["Merge_Key_Short"]
                            .to_dict())
        df["Merge_Key_Short"] = df["Game_Key"].map(key_map)
    
    logging.info(f"🧪 Final Merge_Key_Short nulls: {df['Merge_Key_Short'].isnull().sum()}")
    df["Sport"] = sport_label.upper()
    
    # Memory before next step
    process = psutil.Process(os.getpid())
    logging.info(f"Memory before next operation: {process.memory_info().rss / 1024 / 1024:.2f} MB")


    def process_chunk_logic(df_chunk):
        df_chunk = df_chunk.copy()
    
        # === Clean Line_Delta
        # === Calculate Line_Delta safely
        df_chunk['Line_Delta'] = (
            pd.to_numeric(df_chunk['Value'], errors='coerce') -
            pd.to_numeric(df_chunk['First_Line_Value'], errors='coerce')
        )
        
        # === Direction alignment (nullable-safe)
        # Set Direction_Aligned based on Line_Delta only (market-only)
        df_chunk['Direction_Aligned'] = np.where(
            df_chunk['Line_Delta'] > 0, 1,
            np.where(df_chunk['Line_Delta'] < 0, 0, np.nan)
        ).astype(float)
        df_chunk['High_Limit_Flag'] = (df_chunk['Sharp_Limit_Total'] >= 10000).astype('Int64')
        # === Direction_Aligned: purely market-based logic
        
        return df_chunk
        
    
    def process_in_chunks(df, chunk_size=10000):
        if df.empty:
            logging.info("ℹ️ Empty DataFrame passed to process_in_chunks — returning empty.")
            return df
        chunks = []
        for start in range(0, len(df), chunk_size):
            df_chunk = df.iloc[start:start + chunk_size]
            processed_chunk = process_chunk_logic(df_chunk)
            chunks.append(processed_chunk)
            del df_chunk, processed_chunk
            gc.collect()
        return pd.concat(chunks, ignore_index=True)
    
    df = process_in_chunks(df, chunk_size=10000)
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
    
    logging.info("🔍 df_master columns: %s", df_master.columns.tolist())
    # === Ensure Home_Team and Away_Team are attached before cover calc
    if 'Home_Team' not in df_valid.columns or 'Away_Team' not in df_valid.columns:
        logging.info("🔗 Attaching Home_Team and Away_Team to df_valid from df_master")
        team_cols = df_master[['Game_Key', 'Home_Team_Norm', 'Away_Team_Norm']].drop_duplicates()
        team_cols = team_cols.rename(columns={'Home_Team_Norm': 'Home_Team', 'Away_Team_Norm': 'Away_Team'})
        df_valid = df_valid.merge(team_cols, on='Game_Key', how='left')
    # ✅ Vectorized calculation
    result = calc_cover(df_valid)  # must return a DataFrame with 2 columns
    
    if result.shape[1] != 2:
        logging.error("❌ calc_cover output shape mismatch — expected 2 columns.")
        return pd.DataFrame()
    
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
       
        # === Merge Home_Team_Norm and Away_Team_Norm from df_scores (which has them)
    # Skip fallback if the first merge created normalized team columns
    has_team_norm_clean = 'Home_Team_Norm' in df_master.columns and 'Away_Team_Norm' in df_master.columns
    has_team_norm_merge_x = 'Home_Team_Norm_x' in df_master.columns and 'Away_Team_Norm_x' in df_master.columns
    has_team_norm_merge_y = 'Home_Team_Norm_y' in df_master.columns and 'Away_Team_Norm_y' in df_master.columns
    
    if not (has_team_norm_clean or has_team_norm_merge_x or has_team_norm_merge_y):
        
    
        if (
            'Game_Key' in df_scores.columns and
            'Home_Team' in df_scores.columns and
            'Away_Team' in df_scores.columns
        ):
            logging.info("🧩 Fallback: merging teams via Game_Key from df_scores")
            team_cols = df_scores[['Game_Key', 'Home_Team', 'Away_Team']].drop_duplicates()
            team_cols['Home_Team_Norm'] = team_cols['Home_Team'].astype(str).str.lower().str.strip()
            team_cols['Away_Team_Norm'] = team_cols['Away_Team'].astype(str).str.lower().str.strip()
            df_master = df_master.merge(team_cols, on='Game_Key', how='left')
            logging.info("✅ Fallback team merge complete")
        else:
            logging.warning("⚠️ df_scores is missing Game_Key or team columns — skipping fallback merge")
    else:
        logging.info("⏭️ Skipping fallback — team normalization columns already exist from Merge_Key_Short merge")
    
    logging.info("📦 df_master columns AFTER team merge: %s", df_master.columns.tolist())
    
    # === 🔬 Optional: count missing normalized names
    missing_home_norm = df_master['Home_Team_Norm'].isna().sum()
    missing_away_norm = df_master['Away_Team_Norm'].isna().sum()
    logging.info(f"🔎 Missing Home_Team_Norm: {missing_home_norm}, Away_Team_Norm: {missing_away_norm}")
    # === Compute engineered features
    df_master['Line_Magnitude_Abs'] = df_master['Value'].abs()
    df_master['Is_Home_Team_Bet'] = (df_master['Outcome'].str.lower() == df_master['Home_Team_Norm'].str.lower())
    df_master['Is_Favorite_Bet'] = df_master['Value'] < 0
    
    # Optional: also calculate if not already done
    if 'Line_Move_Magnitude' not in df_master.columns:
        df_master['Line_Move_Magnitude'] = (df_master['Value'] - df_master['First_Line_Value']).abs()
    # === Final Output DataFrame ===
    score_cols = [
        'Game_Key', 'Bookmaker','Book', 'Market', 'Limit', 'Outcome', 'Ref_Sharp_Value',
        'Sharp_Move_Signal', 'Sharp_Limit_Jump', 
        'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
        'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
        'Unique_Sharp_Books', 'Enhanced_Sharp_Confidence_Score',
        'True_Sharp_Confidence_Score', 'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT',
        'Scored', 'Sport', 'Value', 'Merge_Key_Short',
        'First_Line_Value', 
        'Line_Delta', 'Model_Prob_Diff', 'Direction_Aligned',
        'Home_Team_Norm', 'Away_Team_Norm', 'Commence_Hour',
        'Line_Magnitude_Abs', 'High_Limit_Flag',
        'Line_Move_Magnitude', 'Is_Home_Team_Bet', 'Is_Favorite_Bet','Model_Sharp_Win_Prob', 'Odds_Price', 'Implied_Prob','First_Odds', 'First_Imp_Prob','Odds_Shift','Implied_Prob_Shift',
        'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds', 'Value_Reversal_Flag','Odds_Reversal_Flag',      
        'Late_Game_Steam_Flag', 
        'SharpMove_Magnitude_Overnight_VeryEarly', 'SharpMove_Magnitude_Overnight_MidRange',
        'SharpMove_Magnitude_Overnight_LateGame', 'SharpMove_Magnitude_Overnight_Urgent',
        'SharpMove_Magnitude_Early_VeryEarly', 'SharpMove_Magnitude_Early_MidRange',
        'SharpMove_Magnitude_Early_LateGame', 'SharpMove_Magnitude_Early_Urgent',
        'SharpMove_Magnitude_Midday_VeryEarly', 'SharpMove_Magnitude_Midday_MidRange',
        'SharpMove_Magnitude_Midday_LateGame', 'SharpMove_Magnitude_Midday_Urgent',
        'SharpMove_Magnitude_Late_VeryEarly', 'SharpMove_Magnitude_Late_MidRange',
        'SharpMove_Magnitude_Late_LateGame', 'SharpMove_Magnitude_Late_Urgent','SharpMove_Timing_Dominant',
        'SharpMove_Timing_Magnitude', 
            # 🎯 Odds timing magnitude (odds) — ✅ NEW ADDITIONS
        'Odds_Move_Magnitude',
        'OddsMove_Magnitude_Overnight_VeryEarly', 'OddsMove_Magnitude_Overnight_MidRange',
        'OddsMove_Magnitude_Overnight_LateGame', 'OddsMove_Magnitude_Overnight_Urgent',
        'OddsMove_Magnitude_Early_VeryEarly', 'OddsMove_Magnitude_Early_MidRange',
        'OddsMove_Magnitude_Early_LateGame', 'OddsMove_Magnitude_Early_Urgent',
        'OddsMove_Magnitude_Midday_VeryEarly', 'OddsMove_Magnitude_Midday_MidRange',
        'OddsMove_Magnitude_Midday_LateGame', 'OddsMove_Magnitude_Midday_Urgent',
        'OddsMove_Magnitude_Late_VeryEarly', 'OddsMove_Magnitude_Late_MidRange',
        'OddsMove_Magnitude_Late_LateGame', 'OddsMove_Magnitude_Late_Urgent',
        'Net_Line_Move_From_Opening',
        'Abs_Line_Move_From_Opening',
        'Net_Odds_Move_From_Opening',
        'Abs_Odds_Move_From_Opening',

    ]
    
    
    # === Final output
    df_scores_out = ensure_columns(df, score_cols)[score_cols].copy()
  
    logging.info("📦 df Scores out: %s", df_scores_out.columns.tolist())
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
    # === Normalize and unify sport labels
    df_scores_out['Sport'] = (
        df_scores_out['Sport']
          .astype(str).str.strip().str.upper()
          .replace({
              # Baseball
              'BASEBALL_MLB': 'MLB',
              'BASEBALL-MLB': 'MLB',
              'BASEBALL':     'MLB',
              'MLB':          'MLB',
    
              # Basketball
              'BASKETBALL_NBA':  'NBA',
              'NBA':             'NBA',
              'BASKETBALL_WNBA': 'WNBA',
              'WNBA':            'WNBA',
              'BASKETBALL_NCAAB':  'NCAAB',
              'NCAAB':             'NCAAB',
              
              # Football – NFL
              'AMERICANFOOTBALL_NFL': 'NFL',
              'FOOTBALL_NFL':         'NFL',
              'NFL':                  'NFL',
    
              # Football – NCAAF (college)
              'AMERICANFOOTBALL_NCAAF': 'NCAAF',
              'FOOTBALL_NCAAF':         'NCAAF',
              'CFB':                    'NCAAF',
              'NCAAF':                  'NCAAF',
    
              # Football – CFL (Canadian)
              'FOOTBALL_CFL':     'CFL',
              'CANADIANFOOTBALL': 'CFL',
              'CANADIAN_FOOTBALL':'CFL',
              'CFL':              'CFL',
          })
    )
    
    if 'Snapshot_Timestamp' in df.columns:
        df_scores_out['Snapshot_Timestamp'] = df['Snapshot_Timestamp']
    else:
        df_scores_out['Snapshot_Timestamp'] = pd.Timestamp.utcnow()
    
    # === Coerce and clean all fields BEFORE dedup and upload
    df_scores_out['Sharp_Move_Signal'] = pd.to_numeric(df_scores_out['Sharp_Move_Signal'], errors='coerce').astype('Int64')
    df_scores_out['Sharp_Limit_Jump'] = pd.to_numeric(df_scores_out['Sharp_Limit_Jump'], errors='coerce').astype('Int64')
   
    
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
 
    df_scores_out['Line_Delta'] = pd.to_numeric(df_scores_out['Line_Delta'], errors='coerce')
    df_scores_out['Model_Prob_Diff'] = pd.to_numeric(df_scores_out['Model_Prob_Diff'], errors='coerce')
    df_scores_out['Direction_Aligned'] = pd.to_numeric(df_scores_out['Direction_Aligned'], errors='coerce').fillna(0).round().astype('Int64')
    df_scores_out['First_Odds'] = pd.to_numeric(df_scores_out['First_Odds'], errors='coerce')
    df_scores_out['First_Imp_Prob'] = pd.to_numeric(df_scores_out['First_Imp_Prob'], errors='coerce')
    df_scores_out['Odds_Shift'] = df_scores_out['Odds_Price'] - df_scores_out['First_Odds']
    df_scores_out['Implied_Prob_Shift'] = df_scores_out['Implied_Prob'] - df_scores_out['First_Imp_Prob']
   
    df_scores_out['Odds_Reversal_Flag'] = pd.to_numeric(df_scores_out['Odds_Reversal_Flag'], errors='coerce').astype('Int64')

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
        'Game_Key', 'Bookmaker', 'Market', 'Outcome', 'Ref_Sharp_Value',
        'Sharp_Move_Signal', 'Sharp_Limit_Jump',
        'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
        'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
        'Unique_Sharp_Books', 'Enhanced_Sharp_Confidence_Score',
        'True_Sharp_Confidence_Score', 'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT',
        'Scored', 'Sport', 'Value', 'Merge_Key_Short',
        'First_Line_Value',
        'Line_Delta', 'Model_Prob_Diff', 'Direction_Aligned',
        'Home_Team_Norm', 'Away_Team_Norm', 'Commence_Hour',
        'Line_Magnitude_Abs', 'High_Limit_Flag','Model_Sharp_Win_Prob',
        'Line_Move_Magnitude', 'Is_Home_Team_Bet', 'Is_Favorite_Bet','Model_Sharp_Win_Prob']
    
    logging.info(f"🧪 Fingerprint dedup keys: {dedup_fingerprint_cols}")
    float_cols_to_round = [
        'Sharp_Time_Score', 'Sharp_Limit_Total', 'Value',
        'First_Line_Value', 'Line_Delta', 'Model_Prob_Diff'
    ]

  
    for col in float_cols_to_round:
        if col in df_scores_out.columns:
            df_scores_out[col] = pd.to_numeric(df_scores_out[col], errors='coerce').round(4)
    # === 🧹 Canonical row filtering BEFORE fingerprint dedup
    canonical_sort_order = [
       'Scored',  # Prefer scored
       'Snapshot_Timestamp',  # Most recent
       'Model_Sharp_Win_Prob',  # Strongest signal
    ]
    
    df_scores_out = df_scores_out.sort_values(
       by=[col for col in canonical_sort_order if col in df_scores_out.columns],
       ascending=[False] * len(canonical_sort_order)
    )
    
    # Keep best row per game/book/market/outcome
    df_scores_out = df_scores_out.drop_duplicates(
       subset=['Game_Key', 'Bookmaker', 'Market', 'Outcome'], keep='first'
    )
    
    logging.info(f"🧪 Local rows before dedup: {len(df_scores_out)}")

    # === Query BigQuery for existing line-level deduplication
    existing = bq_client.query(f"""
        SELECT DISTINCT {', '.join(dedup_fingerprint_cols)}
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    """).to_dataframe()
    logging.info(f"🧪 Existing fingerprinted rows in BigQuery: {len(existing)}")
    
    # === Fetch all games already scored (to block re-inserts)
    scored_keys_df = bq_client.query("""
        SELECT DISTINCT Merge_Key_Short
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE LOWER(CAST(Scored AS STRING)) = 'true'
    """).to_dataframe()
    already_scored_keys = set(scored_keys_df['Merge_Key_Short'].dropna())
    
    # === Filter out already-scored games
    pre_score_filter = len(df_scores_out)
    # Allow rescore for tracking time-based line evolution
    if track_feature_evolution:
        logging.info("⏱️ Allowing rescore for time-series tracking mode")
    else:
        df_scores_out = df_scores_out[~df_scores_out['Merge_Key_Short'].isin(already_scored_keys)]

    logging.info(f"🧹 Removed {pre_score_filter - len(df_scores_out)} rows from already-scored games")
    
    # === Deduplicate against BigQuery fingerprinted rows
    pre_dedup = len(df_scores_out)
    df_scores_out = df_scores_out.merge(
        existing,
        on=dedup_fingerprint_cols,
        how='left',
        indicator=True
    )
    df_scores_out = df_scores_out[df_scores_out['_merge'] == 'left_only'].drop(columns=['_merge'])
    logging.info(f"🧹 Removed {pre_dedup - len(df_scores_out)} duplicate line-level rows based on fingerprint keys")
    
    # === Final logs and early exit
    if df_scores_out.empty:
        logging.info("ℹ️ No new scores to upload — all rows were already scored or duplicate line states.")
        return pd.DataFrame()
    logging.info("🔎 Sample value/odds extremes:\n" +
        df_scores_out[['Game_Key', 'Value', 'Max_Value', 'Min_Value', 'Odds_Price', 'Max_Odds', 'Min_Odds']].dropna().head(10).to_string(index=False))

    # === Log preview before upload
    logging.info(f"✅ Final rows to upload: {len(df_scores_out)}")
    logging.info("🧪 Sample rows to upload:\n" +
                 df_scores_out[['Game_Key', 'Market', 'Outcome', 'Bookmaker', 'Snapshot_Timestamp']].head(5).to_string(index=False))
    
    # === Parquet validation (debug aid)
    try:
        pa.Table.from_pandas(df_scores_out)
    except Exception as e:
        logging.error("❌ Parquet conversion failure before upload:")
        logging.error(str(e))
        for col in df_scores_out.columns:
            logging.info(f"🔍 {col} → {df_scores_out[col].dtype}, sample: {df_scores_out[col].dropna().unique()[:5].tolist()}")
    
    # === Return final deduplicated and filtered DataFrame
    return df_scores_out


def compute_and_write_market_weights(df):
    component_cols = [
        'Sharp_Move_Signal',
        'Sharp_Limit_Jump',
        
        'Sharp_Time_Score',
        'Sharp_Limit_Total'
    ]

    rows = []

    for comp in component_cols:
        if comp not in df.columns:
            continue

        temp = df[['Market', comp, 'SHARP_HIT_BOOL']].dropna().copy()
        temp['Market'] = temp['Market'].astype(str).str.lower()
        temp['Component'] = comp

        # Convert values to bucketed strings
        if comp == 'Sharp_Limit_Total':
            temp['Value'] = temp[comp].astype(float).apply(lambda x: str(int(x // 1000 * 1000)))
        else:
            temp['Value'] = temp[comp].apply(
                lambda x: str(int(x)) if isinstance(x, float) and x.is_integer() else str(x).strip().lower()
            )

        # Group and average win rates
        grouped = (
            temp.groupby(['Market', 'Component', 'Value'])['SHARP_HIT_BOOL']
            .mean()
            .reset_index()
            .rename(columns={'SHARP_HIT_BOOL': 'Win_Rate'})
        )

        for _, row in grouped.iterrows():
            rows.append({
                'Market': row['Market'],
                'Component': row['Component'],
                'Value': row['Value'],
                'Win_Rate': round(row['Win_Rate'], 4)
            })

    if not rows:
        print("⚠️ No valid market weights to upload.")
        return {}

    df_weights = pd.DataFrame(rows)
    print(f"✅ Prepared {len(df_weights)} market weight rows. Sample:")
    print(df_weights.head(10).to_string(index=False))

    # Return as nested dict for use in scoring
    market_weights = {}
    for _, row in df_weights.iterrows():
        market = row['Market']
        comp = row['Component']
        val = row['Value']
        rate = row['Win_Rate']
        market_weights.setdefault(market, {}).setdefault(comp, {})[val] = rate

    return market_weights


