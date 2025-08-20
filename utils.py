import pandas as pd
from datetime import datetime
from collections import defaultdict
from google.cloud import bigquery, storage
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import requests
import numpy as np
import logging
import hashlib
import time
import json
import psutil
import os
import gc
import psutil
import logging
from pandas_gbq import to_gbq
import traceback
import pickle  # ✅ Add this at the top of your script
import datetime as dt
import sys, traceback, json  # safe local imports used in this block
from typing import Optional, Callable
from functools import lru_cache

import sys
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from google.cloud import bigquery, storage
import logging
logging.basicConfig(level=logging.INFO)  # <- Must be INFO or DEBUG to show .info() logs
logger = logging.getLogger(__name__)

import math

import warnings
from sklearn.exceptions import InconsistentVersionWarning

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
    "NFL": "americanfootball_nfl",     # National Football League
    "NCAAF": "americanfootball_ncaaf", # NCAA College Football
}

SHARP_BOOKS_FOR_LIMITS = ['pinnacle']
SHARP_BOOKS = SHARP_BOOKS_FOR_LIMITS + ['betus','mybookieag','smarkets','betfair_ex_eu','betfair_ex_uk','betfair_ex_au','lowvig','betonlineag','matchbook' ]

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



SPORT_ALIASES = {
        "MLB":   ["MLB", "BASEBALL_MLB", "BASEBALL-MLB", "BASEBALL"],
        "NFL":   ["NFL", "AMERICANFOOTBALL_NFL", "FOOTBALL_NFL"],
        "NCAAF": ["NCAAF", "AMERICANFOOTBALL_NCAAF", "CFB"],
        "NBA":   ["NBA", "BASKETBALL_NBA"],
        "WNBA":  ["WNBA", "BASKETBALL_WNBA"],
        "CFL":   ["CFL", "CANADIANFOOTBALL", "CANADIAN_FOOTBALL"],
    }


def update_power_ratings(
    project_table_scores="sharplogger.sharp_data.game_scores_final",
    table_history="sharplogger.sharp_data.ratings_history",
    table_current="sharplogger.sharp_data.ratings_current",
    default_sport="MLB",
):
    """
    Backfill-or-incremental team power ratings (scores-only), per sport:
      - MLB  -> Poisson/Skellam-style attack+defense rating (no MOV)
      - NFL  -> Elo with MOV boost + K-decay
      - NBA  -> Elo with MOV boost + K-decay
    """
   
    def get_aliases(canon: str) -> list[str]:
        return SPORT_ALIASES.get(canon.upper(), [canon.upper()])

    SPORT_CFG = {
        "MLB":    dict(model="poisson", K_start=None, K_late=None, HFA=18.0, mov_cap=None),
        "NFL":    dict(model="elo", K_start=22.0, K_late=12.0, HFA=22.0, mov_cap=24.0),
        "NCAAF":  dict(model="elo", K_start=22.0, K_late=12.0, HFA=22.0, mov_cap=24.0),
        "NBA":    dict(model="elo", K_start=18.0, K_late=10.0, HFA=55.0, mov_cap=25.0),
        "WNBA":   dict(model="elo", K_start=18.0, K_late=10.0, HFA=45.0, mov_cap=25.0),
        "CFL":    dict(model="elo", K_start=20.0, K_late=12.0, HFA=18.0, mov_cap=30.0),
    }
      

    def elo_expected(delta, hfa=20.0):
        return 1.0 / (1.0 + 10.0 ** (-(delta + hfa) / 400.0))

    def mov_boost(margin, delta, mov_cap=None):
        rd = max(1.0, abs(float(margin)))
        if mov_cap is not None:
            rd = min(rd, mov_cap)
        return np.log1p(rd) * (2.2 / (0.001 * abs(delta) + 2.2))

    def season_progress(ts_series: pd.Series) -> pd.Series:
        if ts_series.empty:
            return pd.Series(dtype=float)
        dmin, dmax = ts_series.min(), ts_series.max()
        span = max((dmax - dmin).total_seconds(), 1.0)
        return (ts_series - dmin).dt.total_seconds() / span

    BACKFILL_DAYS = 7 
    def load_seed_ratings(bq: bigquery.Client, sport: str, asof_ts: dt.datetime | pd.Timestamp | None) -> dict[str, float]:
        """
        Return latest ratings per team with Updated_At <= asof_ts.
        If asof_ts is None, fall back to current -> last hist -> {} (like your old logic).
        """
        if asof_ts is None:
            # original seed fallback
            cur = bq.query(
                f"SELECT Team, Rating FROM `{table_current}` WHERE Sport = @sport",
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport)]
                )
            ).to_dataframe()
            if not cur.empty:
                return dict(zip(cur.Team, cur.Rating))
    
            hist = bq.query(
                f"""
                SELECT Team, ANY_VALUE(Rating) AS Rating
                FROM (
                  SELECT Team, Rating,
                         ROW_NUMBER() OVER (PARTITION BY Team ORDER BY `Updated_At` DESC) AS rn
                  FROM `{table_history}` WHERE Sport = @sport
                )
                WHERE rn = 1 GROUP BY Team
                """,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport)]
                )
            ).to_dataframe()
            return dict(zip(hist.Team, hist.Rating)) if not hist.empty else {}
    
        # normalize tz
        if isinstance(asof_ts, pd.Timestamp):
            asof_ts = (asof_ts.tz_localize("UTC") if asof_ts.tzinfo is None else asof_ts.tz_convert("UTC")).to_pydatetime()
        elif isinstance(asof_ts, dt.datetime) and asof_ts.tzinfo is None:
            asof_ts = asof_ts.replace(tzinfo=dt.timezone.utc)
    
        # seed as-of: pick latest row per team with Updated_At <= asof_ts
        df = bq.query(
            f"""
            WITH latest AS (
              SELECT Sport, Team, Rating, Updated_At,
                     ROW_NUMBER() OVER (PARTITION BY Team ORDER BY Updated_At DESC) AS rn
              FROM `{table_history}`
              WHERE Sport = @sport AND Updated_At <= @asof
            )
            SELECT Team, Rating FROM latest WHERE rn = 1
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("sport", "STRING", sport),
                    bigquery.ScalarQueryParameter("asof",  "TIMESTAMP", asof_ts),
                ]
            )
        ).to_dataframe()
        return dict(zip(df.Team, df.Rating))


    def upsert_current(bq: bigquery.Client, rows: list):
        if not rows:
            return
        df_cur = pd.DataFrame(rows)
        stage = table_current.rsplit(".", 1)[0] + "._ratings_current_stage"
        bq.load_table_from_dataframe(df_cur, stage).result()
        bq.query(f"""
        MERGE `{table_current}` T
        USING `{stage}` S
        ON T.Sport=S.Sport AND T.Team=S.Team
        WHEN MATCHED THEN UPDATE SET
          T.Rating=S.Rating, T.Method=S.Method, T.`Updated_At`=S.`Updated_At`
        WHEN NOT MATCHED THEN INSERT (Sport,Team,Rating,Method,`Updated_At`)
          VALUES (S.Sport,S.Team,S.Rating,S.Method,S.`Updated_At`)
        """).result()
        bq.query(f"DROP TABLE `{stage}`").result()

    def fetch_sports_present(bq):
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

    def get_last_ts(bq, sport: str):
        df = bq.query(
            f"""
            SELECT MAX(`Updated_At`) AS last_ts
            FROM `{table_history}`
            WHERE Sport = @sport
            """,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport)]
            ),
        ).to_dataframe()
    
        if df.empty:
            return None
    
        ts = df.last_ts.iloc[0]
        if ts is None or (hasattr(pd, "isna") and pd.isna(ts)):
            return None
        # Ensure tz-aware UTC and return a pandas Timestamp (we’ll convert later if needed)
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return ts
        # python datetime
        if isinstance(ts, dt.datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)
        return None

    def has_new_finals_since(bq, sport: str, last_ts):
        aliases = get_aliases(sport)
        cutoff_check = None
        if last_ts is not None and not (isinstance(last_ts, pd.Timestamp) and pd.isna(last_ts)):
            # Always look back at least 7 days
            if isinstance(last_ts, pd.Timestamp):
                lt = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts.tz_convert("UTC")
                cutoff_check = (lt - pd.Timedelta(days=BACKFILL_DAYS)).to_pydatetime()
            elif isinstance(last_ts, dt.datetime):
                lt = last_ts if last_ts.tzinfo else last_ts.replace(tzinfo=dt.timezone.utc)
                cutoff_check = lt - dt.timedelta(days=BACKFILL_DAYS)
    
        sql = f"""
          SELECT COUNT(*) AS n
          FROM `{project_table_scores}`
          WHERE UPPER(CAST(Sport AS STRING)) IN UNNEST(@sport_aliases)
            AND Score_Home_Score IS NOT NULL AND Score_Away_Score IS NOT NULL
            {'' if cutoff_check is None else 'AND TIMESTAMP(Inserted_Timestamp) > @cutoff'}
          LIMIT 1
        """
        params = [bigquery.ArrayQueryParameter("sport_aliases", "STRING", aliases)]
        if cutoff_check is not None:
            params.append(bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", cutoff_check))
    
        n = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe().n.iloc[0]
        return n > 0
    


    def sports_present_in_history(bq: bigquery.Client) -> list[str]:
        q = f"SELECT DISTINCT UPPER(Sport) AS s FROM `{table_history}`"
        df = bq.query(q).to_dataframe()
        return df.s.dropna().str.upper().tolist()

    def fill_missing_current_from_history(bq: bigquery.Client, sports: list[str] | None = None):
        """
        Ensure every (Sport, Team, Method) that exists in history also exists in current.
        Only inserts when missing; DOES NOT overwrite existing current rows.
        """
        if not sports:
            # default to all canonical sports we know how to manage AND that exist in history
            hist = set(sports_present_in_history(bq))
            sports = sorted(hist & set(SPORT_CFG.keys()))
    
        for sport in sports:
            sql = f"""
            MERGE `{table_current}` T
            USING (
              WITH latest AS (
                SELECT
                  Sport, Team, Method, Rating, Updated_At,
                  ROW_NUMBER() OVER (PARTITION BY Sport, Team, Method ORDER BY Updated_At DESC) AS rn
                FROM `{table_history}`
                WHERE Sport = @sport
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
                    query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport)]
                ),
            ).result()

    def load_games(bq, sport: str, cutoff=None):
        aliases = get_aliases(sport)
        cutoff_param = None
        if cutoff is not None and not (isinstance(cutoff, pd.Timestamp) and pd.isna(cutoff)):
            if isinstance(cutoff, pd.Timestamp):
                cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
                cutoff_param = cutoff.to_pydatetime()
            elif isinstance(cutoff, dt.datetime):
                cutoff_param = cutoff if cutoff.tzinfo else cutoff.replace(tzinfo=dt.timezone.utc)
    
        base_sql = f"""
        SELECT
          CASE
            WHEN UPPER(CAST(Sport AS STRING)) IN ('BASEBALL_MLB','BASEBALL-MLB','BASEBALL','MLB') THEN 'MLB'
            WHEN UPPER(CAST(Sport AS STRING)) IN ('AMERICANFOOTBALL_NFL','FOOTBALL_NFL','NFL') THEN 'NFL'
            WHEN UPPER(CAST(Sport AS STRING)) IN ('BASKETBALL_NBA','NBA') THEN 'NBA'
            WHEN UPPER(CAST(Sport AS STRING)) IN ('BASKETBALL_WNBA','WNBA') THEN 'WNBA'
            WHEN UPPER(CAST(Sport AS STRING)) IN ('CFL','CANADIANFOOTBALL','CANADIAN_FOOTBALL') THEN 'CFL'
            ELSE COALESCE(CAST(Sport AS STRING), '{default_sport}')
          END AS Sport,
          Home_Team, Away_Team,
          TIMESTAMP(Game_Start) AS Game_Start,
          SAFE_CAST(Score_Home_Score AS FLOAT64) AS Score_Home_Score,
          SAFE_CAST(Score_Away_Score AS FLOAT64) AS Score_Away_Score,
          TIMESTAMP(Inserted_Timestamp) AS Snapshot_TS
        FROM `{project_table_scores}`
        WHERE UPPER(CAST(Sport AS STRING)) IN UNNEST(@sport_aliases)
          AND Score_Home_Score IS NOT NULL AND Score_Away_Score IS NOT NULL
        """
        if cutoff_param is None:
            sql = base_sql + "\nORDER BY Snapshot_TS"
            params = [bigquery.ArrayQueryParameter("sport_aliases", "STRING", aliases)]
        else:
            sql = base_sql + "\n  AND TIMESTAMP(Inserted_Timestamp) > @cutoff\nORDER BY Snapshot_TS"
            params = [
                bigquery.ArrayQueryParameter("sport_aliases", "STRING", aliases),
                bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", cutoff_param),
            ]
        return bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()

    def sync_current_from_history(bq: bigquery.Client, sports: list[str]):
           """
           Upsert ratings_current to the latest snapshot per (Sport, Team, Method)
           from ratings_history for the provided canonical sport names.
           """
           if not sports:
               return
       
           # Use a parameterized MERGE selecting the latest per (Sport, Team, Method)
           # We do it sport-by-sport to keep params simple.
           for sport in sports:
               # canonical sport name (we write canonical keys into history/current)
               sql = f"""
               MERGE `{table_current}` T
               USING (
                 WITH latest AS (
                   SELECT
                     Sport,
                     Team,
                     Method,
                     Rating,
                     Updated_At,
                     ROW_NUMBER() OVER (
                       PARTITION BY Sport, Team, Method
                       ORDER BY Updated_At DESC
                     ) AS rn
                   FROM `{table_history}`
                   WHERE Sport = @sport
                 )
                 SELECT Sport, Team, Method, Rating, Updated_At
                 FROM latest
                 WHERE rn = 1
               ) S
               ON T.Sport = S.Sport AND T.Team = S.Team AND T.Method = S.Method
               WHEN MATCHED THEN UPDATE SET
                 T.Rating = S.Rating,
                 T.Updated_At = S.Updated_At
               WHEN NOT MATCHED THEN
                 INSERT (Sport, Team, Method, Rating, Updated_At)
                 VALUES (S.Sport, S.Team, S.Method, S.Rating, S.Updated_At)
               """
               bq.query(
                   sql,
                   job_config=bigquery.QueryJobConfig(
                       query_parameters=[bigquery.ScalarQueryParameter("sport", "STRING", sport)]
                   ),
               ).result()



    # ---------------- MAIN WORK ----------------
    bq = bigquery.Client()
    sports_available = fetch_sports_present(bq)
    sports = [s for s in sports_available if s.upper() in SPORT_CFG]
    if not sports:
        return {"message": "No finalized games found for supported sports.", "history_rows": 0, "current_rows": 0}
    
    history_rows_all, current_rows_all = [], []
    updated_sports = []
    
    for sport in sports:
        cfg = SPORT_CFG[sport.upper()]
        last_ts = get_last_ts(bq, sport)
    
        # ✅ Always replay at least BACKFILL_DAYS by SNAPSHOT time
        if last_ts is None:
            window_start = None
        else:
            if isinstance(last_ts, pd.Timestamp):
                last_ts_utc = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts.tz_convert("UTC")
                window_start = last_ts_utc - pd.Timedelta(days=BACKFILL_DAYS)
            else:
                lt = last_ts if last_ts.tzinfo else last_ts.replace(tzinfo=dt.timezone.utc)
                window_start = lt - dt.timedelta(days=BACKFILL_DAYS)
    
        # 🚦 Quick existence check should use the 7-day window, not last_ts
        if not has_new_finals_since(bq, sport, window_start):
            continue
    
        if cfg["model"] == "elo":
            # 🧠 Elo: load games since window_start (Inserted_Timestamp based)
            df_games = load_games(bq, sport, cutoff=window_start)
            if df_games.empty:
                continue
    
            # ✅ Seed ratings as of window_start (see earlier guidance to add asof param)
            ratings = load_seed_ratings(bq, sport, asof_ts=window_start)
    
            prog = season_progress(df_games["Game_Start"])
            K_series = cfg["K_late"] + (cfg["K_start"] - cfg["K_late"]) * (1.0 - prog)
    
            history_rows = []
            for (row_idx, g), K in zip(df_games.iterrows(), K_series):
                home, away = g.Home_Team, g.Away_Team
                Rh = ratings.get(home, 1500.0)
                Ra = ratings.get(away, 1500.0)
    
                delta = Rh - Ra
                p_home = elo_expected(delta, hfa=cfg["HFA"])
                result = 1.0 if g.Score_Home_Score > g.Score_Away_Score else 0.0
                mov = g.Score_Home_Score - g.Score_Away_Score
                m = mov_boost(mov, delta, mov_cap=cfg.get("mov_cap"))
    
                adj = K * m * (result - p_home)
                Rh2, Ra2 = Rh + adj, Ra - adj
    
                ts = g.Snapshot_TS
                tag = "backfill" if window_start is None else "incremental"
                history_rows += [
                    dict(Sport=sport, Team=home, Rating=float(Rh2), Method="elo_mov", Updated_At=ts, Source=tag),
                    dict(Sport=sport, Team=away, Rating=float(Ra2), Method="elo_mov", Updated_At=ts, Source=tag),
                ]
                ratings[home], ratings[away] = Rh2, Ra2
    
            if history_rows:
                bq.load_table_from_dataframe(pd.DataFrame(history_rows), table_history).result()
                history_rows_all.extend(history_rows)
                updated_sports.append(sport)
    
            utc_now = pd.Timestamp.now(tz="UTC")
            for team, rating in ratings.items():
                current_rows_all.append({
                    'Sport': sport, 'Team': team, 'Rating': float(rating),
                    'Method': 'elo_mov', 'Updated_At': utc_now,
                })
    
        elif cfg["model"] == "poisson":
            # ⚾️ MLB: base totals up to window_start (snapshot clock)
            if window_start is None:
                df_base = pd.DataFrame(columns=["Home_Team","Away_Team","Score_Home_Score","Score_Away_Score"])
            else:
                df_base = bq.query(f"""
                SELECT Home_Team, Away_Team,
                       SAFE_CAST(Score_Home_Score AS FLOAT64) AS Score_Home_Score,
                       SAFE_CAST(Score_Away_Score AS FLOAT64) AS Score_Away_Score
                FROM `{project_table_scores}`
                WHERE UPPER(CAST(Sport AS STRING)) IN ('MLB','BASEBALL_MLB','BASEBALL-MLB','BASEBALL')
                  AND Score_Home_Score IS NOT NULL AND Score_Away_Score IS NOT NULL
                  AND TIMESTAMP(Inserted_Timestamp) <= @cutoff
                """, job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter(
                        "cutoff", "TIMESTAMP",
                        window_start.to_pydatetime() if isinstance(window_start, pd.Timestamp) else window_start
                    )]
                )).to_dataframe()
    
            GF, GA, GP = {}, {}, {}
            def add_game_to_totals(h, a, hs, as_):
                GF[h] = GF.get(h, 0.0) + hs
                GA[h] = GA.get(h, 0.0) + as_
                GP[h] = GP.get(h, 0) + 1
                GF[a] = GF.get(a, 0.0) + as_
                GA[a] = GA.get(a, 0.0) + hs
                GP[a] = GP.get(a, 0) + 1
    
            for _, r in df_base.iterrows():
                add_game_to_totals(r.Home_Team, r.Away_Team, float(r.Score_Home_Score), float(r.Score_Away_Score))
    
            # 🔁 New games since window_start
            df_new = load_games(bq, sport, cutoff=window_start)   # <-- was last_ts
            if df_new.empty:
                continue
    
            history_rows, ratings = [], {}
            def team_rating(team):
                g = int(GP.get(team, 0))
                if g == 0: return 1500.0
                total_team_games = int(sum(GP.values()))
                gf_league = float(sum(GF.values()))
                eps = 1e-6
                league_rate = max(gf_league / max(total_team_games, 1), eps)
                rate_for   = float(GF.get(team, 0.0)) / max(g, 1)
                rate_against = float(GA.get(team, 0.0)) / max(g, 1)
                atk_ratio = max(rate_for / league_rate, eps)
                dfn_ratio = max(rate_against / league_rate, eps)
                atk = np.log(atk_ratio)
                dfn = -np.log(dfn_ratio)
                return 1500.0 + 400.0 * (atk + dfn)
    
            tag = "backfill" if window_start is None else "incremental"
            for _, g in df_new.iterrows():
                hs, as_ = float(g.Score_Home_Score), float(g.Score_Away_Score)
                add_game_to_totals(g.Home_Team, g.Away_Team, hs, as_)
                Rh2 = float(team_rating(g.Home_Team))
                Ra2 = float(team_rating(g.Away_Team))
                ts = g.Snapshot_TS
                history_rows += [
                    dict(Sport=sport, Team=g.Home_Team, Rating=Rh2, Method="poisson", Updated_At=ts, Source=tag),
                    dict(Sport=sport, Team=g.Away_Team, Rating=Ra2, Method="poisson", Updated_At=ts, Source=tag),
                ]
                ratings[g.Home_Team] = Rh2
                ratings[g.Away_Team] = Ra2
    
            if history_rows:
                bq.load_table_from_dataframe(pd.DataFrame(history_rows), table_history).result()
                history_rows_all.extend(history_rows)
                updated_sports.append(sport)
    
            utc_now = pd.Timestamp.now(tz="UTC")
            for team, rating in ratings.items():
                current_rows_all.append(dict(
                    Sport=sport, Team=team, Rating=float(rating), Method="poisson", Updated_At=utc_now
                ))
    
        else:
            continue


    # Upsert current across all sports (no-op if empty)
    # Upsert current across all sports (no-op if empty)
    upsert_current(bq, current_rows_all)
    
    # 🔁 (optional) keep these if you still want current to reflect latest history for processed sports
    sync_current_from_history(bq, sports)
    
    # ✅ Always ensure *missing* teams in current are backfilled from history for all known sports
    fill_missing_current_from_history(bq)  # no args → uses all canonical sports present in history

    return {
        "message": ("Update complete." if updated_sports else "No new finals; ratings unchanged."),
        "sports_processed": sorted(updated_sports),
        "history_rows": len(history_rows_all),
        "current_rows": len(current_rows_all),
    }



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





def write_sharp_moves_to_master(df, table='sharp_data.sharp_moves_master'):
    if df is None or df.empty:
        logging.warning("⚠️ No sharp moves to write.")
        return

    df = df.copy()
    df = build_game_key(df)

    allowed_books = SHARP_BOOKS + REC_BOOKS
    # Apply normalization early
    # ✅ Normalize book names before any filtering
    df['Book'] = df['Book'].str.lower()
    

    
    # ✅ Now apply the filtering with normalized names
    df = df[df['Book'].isin(allowed_books)]


    if 'Game_Key' not in df.columns or df['Game_Key'].isnull().all():
        logging.warning("❌ No valid Game_Key present — skipping upload.")
        logging.debug(df[['Game', 'Game_Start', 'Market', 'Outcome']].head().to_string())
        return
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
        'Line_Delta',               # ✅ Add this
        'Line_Magnitude_Abs',       # Already present
        'Direction_Aligned','Odds_Price', 'Implied_Prob', 
        'Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds',
        'Value_Reversal_Flag', 'Odds_Reversal_Flag','Open_Odds', 'Was_Line_Resistance_Broken',
        'Line_Resistance_Crossed_Levels',
        'Line_Resistance_Crossed_Count', 'Late_Game_Steam_Flag', 'Sharp_Line_Magnitude',
        'Rec_Line_Magnitude',
        'SharpMove_Odds_Up',
        'SharpMove_Odds_Down',
        'SharpMove_Odds_Mag',
        'SharpMove_Resistance_Break',
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
        'SharpMove_Timing_Magnitude',# === Net movement columns
        'Net_Line_Move_From_Opening',
        'Abs_Line_Move_From_Opening',
        'Net_Odds_Move_From_Opening',
        'Abs_Odds_Move_From_Opening',
    ]
    
    # 🧩 Add schema-consistent consensus fields from summarize_consensus()
    ALLOWED_ODDS_MOVE_COLUMNS = [
        f'OddsMove_Magnitude_{b}' for b in [
            'Overnight_VeryEarly', 'Overnight_MidRange', 'Overnight_LateGame', 'Overnight_Urgent',
            'Early_VeryEarly', 'Early_MidRange', 'Early_LateGame', 'Early_Urgent',
            'Midday_VeryEarly', 'Midday_MidRange', 'Midday_LateGame', 'Midday_Urgent',
            'Late_VeryEarly', 'Late_MidRange', 'Late_LateGame', 'Late_Urgent'
        ]
    ] + ['Odds_Move_Magnitude']  # ✅ Add this total field
    
    ALLOWED_COLS += ALLOWED_ODDS_MOVE_COLUMNS
    # Ensure all required columns exist
    df = ensure_columns(df, ALLOWED_COLS, fill_value=None)
    df['Odds_Price'] = pd.to_numeric(df.get('Odds_Price'), errors='coerce')
    df['Implied_Prob'] = pd.to_numeric(df.get('Implied_Prob'), errors='coerce')

    # 🔍 Add required column check before filtering
    required_cols = ['Snapshot_Timestamp', 'Model_Sharp_Win_Prob']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logging.error(f"❌ Missing required columns before upload: {missing}")
        return

    # Log any remaining mismatches
    missing_cols = [col for col in ALLOWED_COLS if col not in df.columns]
    if missing_cols:
        logging.error(f"❌ Missing required columns for BigQuery upload: {missing_cols}")
        return
    # === Deduplicate using Line_Hash from previous uploads ===
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
    
    # ⛔ Exit early if nothing to write
    if df.empty:
        logging.info("🛑 No new rows after Line_Hash deduplication — exiting.")
        return
    logging.info(f"📦 Final row count to upload after filtering and dedup: {len(df)}")
    # Filter to allowed schema
    # === Force float type on columns that might fail if BigQuery expects INT64
    # === Cast known float columns safely
    known_float_cols = [
        'SharpMove_Odds_Mag', 'HomeRecLineMag',
        'Rec_Line_Delta', 'Sharp_Line_Delta',
        'Odds_Shift', 'Implied_Prob_Shift', 'Line_Delta',
        'Sharp_Line_Magnitude', 'Rec_Line_Magnitude',
        'Delta_Sharp_vs_Rec','Max_Value', 'Min_Value', 'Max_Odds', 'Min_Odds'
    ]
    for col in known_float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    # === Cast known INT64 columns to nullable Int64
    int_cols = [
        'Sharp_Limit_Total', 'Limit_Jump', 'LimitUp_NoMove_Flag',
        'SHARP_SIDE_TO_BET', 'Sharp_Move_Signal', 'Sharp_Limit_Jump',
        'Scored', 'Scored_By_Model', 'Is_Home_Team_Bet',
        'Is_Favorite_Bet', 'High_Limit_Flag', 'CrossMarketSharpSupport','Value_Reversal_Flag', 'Odds_Reversal_Flag'
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
    df = df[ALLOWED_COLS]
    
    logging.info("🧪 Preview of model columns being written:")
    logging.info(df[model_cols].dropna(how='all').head(5).to_string())
    # 🔍 Preview Odds_Price and Implied_Prob distribution
    if 'Odds_Price' in df.columns and 'Implied_Prob' in df.columns:
        logging.info("🎯 Odds_Price sample:\n" + df['Odds_Price'].dropna().astype(str).head().to_string(index=False))
        logging.info("🎯 Implied_Prob sample:\n" + df['Implied_Prob'].dropna().round(4).astype(str).head().to_string(index=False))
    else:
        logging.warning("⚠️ Odds_Price or Implied_Prob missing from DataFrame before upload")
    logging.info(f"📦 Final row count to upload after filtering and dedup: {len(df)}")
    # Write to BigQuery
    try:
        logging.info(f"📤 Uploading to `{table}`...")
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='append')
        logging.info(f"✅ Wrote {len(df)} new rows to `{table}`")
    except Exception as e:
        logging.exception(f"❌ Upload to `{table}` failed.")
        logging.debug("Schema:\n" + df.dtypes.to_string())
        logging.debug("Preview:\n" + df.head(5).to_string())
        


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



def load_market_weights_from_bq():
    
    client = bigquery.Client(project="sharplogger", location="us")

    query = """
        SELECT *
        FROM `sharplogger.sharp_data.sharp_scores_full`
        WHERE SHARP_HIT_BOOL IS NOT NULL
          AND Scored = TRUE
    """
    df = client.query(query).to_dataframe()

    market_weights = compute_and_write_market_weights(df)  # Returns a dict
    logging.info(f"✅ Loaded market weights for {len(market_weights)} markets")
    return market_weights
    
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

def compute_sharp_metrics(entries, open_val, mtype, label, gk=None, book=None, open_odds=None, opening_limit=None):
    logging.debug(f"🔍 Running compute_sharp_metrics for Outcome: {label}, Market: {mtype}")
    logging.debug(f"📥 Open value: {open_val}, Open odds: {open_odds}")
    logging.debug(f"📦 Received {len(entries)} entries")

    move_signal = 0.0
    move_magnitude_score = 0.0
    limit_score = 0.0
    total_limit = 0.0
    hybrid_timing_mags = defaultdict(float)
    odds_move_magnitude_score = 0.0
    hybrid_timing_odds_mags = defaultdict(float)

    # === Net movement from opening
    net_line_move = None
    abs_net_line_move = None
    net_odds_move = None
    abs_net_odds_move = None
    # derive opening trio from entries if missing
    if open_val is None or open_odds is None or opening_limit is None:
        for (lim, val, ts, game_start, odds) in sorted(entries, key=lambda x: x[2]):
            if open_val is None and pd.notna(val):
                open_val = val
            if open_odds is None and pd.notna(odds):
                open_odds = odds
            if opening_limit is None and pd.notna(lim):
                opening_limit = lim
            if (open_val is not None) and (open_odds is not None) and (opening_limit is not None):
                break
    # --- Always sort first
    mtype = (mtype or "").strip().lower()
    label = (label or "").strip().lower()
    entries = sorted(entries, key=lambda x: x[2])  # (limit, value, ts, game_start, odds)

    def get_hybrid_bucket(ts, game_start):
        hour = pd.to_datetime(ts).hour
        minutes_to_game = (
            (pd.to_datetime(game_start) - pd.to_datetime(ts)).total_seconds() / 60
            if pd.notnull(game_start) else None
        )
        tod = ('Overnight' if 0 <= hour <= 5 else
               'Early'     if 6 <= hour <= 11 else
               'Midday'    if 12 <= hour <= 15 else
               'Late')
        mtg = ('VeryEarly' if minutes_to_game is None else
               'VeryEarly' if minutes_to_game > 720 else
               'MidRange'  if 180 < minutes_to_game <= 720 else
               'LateGame'  if 60 < minutes_to_game <= 180 else
               'Urgent')
        return f"{tod}_{mtg}"

    # --- Bootstrap open trio from earliest non-nulls in entries if missing
    if (open_val is None or pd.isna(open_val)) or (open_odds is None or pd.isna(open_odds)) or (opening_limit is None or pd.isna(opening_limit)):
        for (lim, val, ts, game_start, odds) in entries:
            if (open_val is None or pd.isna(open_val)) and pd.notna(val):
                open_val = val
            if (open_odds is None or pd.isna(open_odds)) and pd.notna(odds):
                open_odds = odds
            if (opening_limit is None or pd.isna(opening_limit)) and pd.notna(lim):
                opening_limit = lim
            if pd.notna(open_val) and pd.notna(open_odds) and (opening_limit is not None):
                break

    # === Track previous values for proper delta logic
    prev_val = open_val
    prev_odds = open_odds

    for i, entry in enumerate(entries):
        if len(entry) != 5:
            logging.warning(f"⚠️ Malformed entry {i+1}: {entry}")
            continue

        limit, curr_val, ts, game_start, curr_odds = entry
        logging.debug(f"🧾 Entry {i+1} → Limit={limit}, Value={curr_val}, Time={ts}, Odds={curr_odds}")

        try:
            # ensure numerics
            curr_val = pd.to_numeric(curr_val, errors='coerce')
            curr_odds = pd.to_numeric(curr_odds, errors='coerce')
            limit = pd.to_numeric(limit, errors='coerce')

            # === Line movement
            if pd.notna(prev_val) and pd.notna(curr_val):
                delta = curr_val - prev_val
                sharp_move_delta = delta

                move_magnitude_score += sharp_move_delta

                if mtype == 'totals':
                    if 'under' in label and sharp_move_delta < 0:
                        move_signal += sharp_move_delta
                    elif 'over' in label and sharp_move_delta > 0:
                        move_signal += sharp_move_delta
                elif mtype == 'spreads':
                    if prev_val < 0 and curr_val < prev_val:
                        move_signal += sharp_move_delta
                    elif prev_val > 0 and curr_val > prev_val:
                        move_signal += sharp_move_delta

                timing_label = get_hybrid_bucket(ts, game_start)
                hybrid_timing_mags[timing_label] += sharp_move_delta

                prev_val = curr_val

            # === Odds movement
            if pd.notna(prev_odds) and pd.notna(curr_odds):
                prev_prob = implied_prob(prev_odds)
                curr_prob = implied_prob(curr_odds)

                odds_delta = curr_prob - prev_prob
                point_equiv = implied_prob_to_point_move(odds_delta)

                logging.debug(f"🧾 Odds Δ: {odds_delta:+.3f} (~{point_equiv:+.1f} pts), From {prev_odds} → {curr_odds}")

                odds_move_magnitude_score += point_equiv
                timing_label = get_hybrid_bucket(ts, game_start)
                hybrid_timing_odds_mags[timing_label] += odds_delta

                prev_odds = curr_odds

            # === Net moves from opening
            if pd.notna(open_val) and pd.notna(curr_val):
                net_line_move = curr_val - open_val
                abs_net_line_move = abs(net_line_move)

            if pd.notna(open_odds) and pd.notna(curr_odds):
                net_odds_move = curr_odds - open_odds
                abs_net_odds_move = abs(net_odds_move)

            # === Limit tracking
            if pd.notna(limit):
                total_limit += float(limit)
                if float(limit) >= 100:
                    limit_score += float(limit)

        except Exception as e:
            logging.warning(f"⚠️ Error in entry {i+1}: {e}")

    # === Final bucket flattening
    all_possible_buckets = [
        f"{tod}_{mtg}"
        for tod in ['Overnight', 'Early', 'Midday', 'Late']
        for mtg in ['VeryEarly', 'MidRange', 'LateGame', 'Urgent']
    ]
    flattened_buckets = {f'SharpMove_Magnitude_{b}': round(hybrid_timing_mags.get(b, 0.0), 3) for b in all_possible_buckets}
    flattened_odds_buckets = {f'OddsMove_Magnitude_{b}': round(hybrid_timing_odds_mags.get(b, 0.0), 3) for b in all_possible_buckets}

    dominant_label, dominant_mag = max(hybrid_timing_mags.items(), key=lambda x: x[1], default=("unknown", 0.0))

    # derive First_Imp_Prob from the open odds we ended up with
    first_imp_prob = implied_prob(open_odds) if pd.notna(open_odds) else None

    logging.debug(f"📊 Final hybrid_timing_mags: {dict(hybrid_timing_mags)}")
    logging.debug(f"📊 Final hybrid_timing_odds_mags: {dict(hybrid_timing_odds_mags)}")

    return {
        # ← return the open trio so callers can use them
        'Open_Value': open_val,
        'Open_Odds': open_odds,
        'Opening_Limit': opening_limit,
        'First_Imp_Prob': first_imp_prob,

        'Sharp_Move_Signal': int(move_signal > 0),
        'Sharp_Line_Magnitude': round(move_magnitude_score, 2),
        'Sharp_Limit_Jump': int(limit_score >= 10000),
        'Sharp_Limit_Total': round(total_limit, 1),
        'SharpMove_Timing_Dominant': dominant_label,
        'Net_Line_Move_From_Opening': round(net_line_move, 3) if net_line_move is not None else None,
        'Abs_Line_Move_From_Opening': round(abs_net_line_move, 3) if abs_net_line_move is not None else None,
        'Net_Odds_Move_From_Opening': round(net_odds_move, 3) if net_odds_move is not None else None,
        'Abs_Odds_Move_From_Opening': round(abs_net_odds_move, 3) if abs_net_odds_move is not None else None,
        'SharpMove_Timing_Magnitude': round(move_magnitude_score, 3),
        **flattened_buckets,
        'Odds_Move_Magnitude': round(odds_move_magnitude_score, 2),
        **flattened_odds_buckets,
        'SharpBetScore': 0.0
    }
def apply_compute_sharp_metrics_rowwise(df: pd.DataFrame,
                                        df_all_snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    Applies compute_sharp_metrics() to each row in df using historical snapshots.
    Enriches df with sharp movement features like Sharp_Move_Signal, timing buckets, etc.
    Safe against category dtype by casting to string for key ops.
    """
    if df is None or df.empty or df_all_snapshots is None or df_all_snapshots.empty:
        return df

    df = df.copy()
    snaps = df_all_snapshots.copy()

    # ---- required cols in df ----
    need_df = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
    miss = [c for c in need_df if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required column(s) in df: {miss}")

    # ---- cast keys to string (avoid category + issues) ----
    for c in ('Game_Key','Market','Outcome','Bookmaker'):
        if c in df.columns:
            df[c] = df[c].astype('string')
        if c in snaps.columns:
            snaps[c] = snaps[c].astype('string')

    # ---- team norms as string (avoid numpy op add on category) ----
    for c in ('Home_Team_Norm','Away_Team_Norm'):
        if c in snaps.columns:
            snaps[c] = snaps[c].astype('string')
        if c in df.columns:
            df[c] = df[c].astype('string')

    # ---- timestamps normalized ----
    if 'Snapshot_Timestamp' in snaps.columns:
        snaps['Snapshot_Timestamp'] = pd.to_datetime(snaps['Snapshot_Timestamp'], errors='coerce', utc=True)
    if 'Game_Start' in snaps.columns:
        snaps['Game_Start'] = pd.to_datetime(snaps['Game_Start'], errors='coerce', utc=True)

    # ---- optional hour bucket (only if useful for your metrics) ----
    if 'Game_Start' in snaps.columns:
        snaps['Commence_Hour'] = snaps['Game_Start'].dt.floor('h')
    else:
        snaps['Commence_Hour'] = pd.NaT

    # ---- build a robust Team_Key using str.cat (na safe) ----
    for c in ('Market','Outcome'):
        if c in snaps.columns:
            snaps[c] = snaps[c].astype('string')

    snaps['Team_Key'] = (
        snaps.get('Home_Team_Norm', pd.Series('', index=snaps.index, dtype='string')).str.cat(
        snaps.get('Away_Team_Norm', pd.Series('', index=snaps.index, dtype='string')), sep='_', na_rep='')
        .str.cat(snaps['Commence_Hour'].astype('string'), sep='_', na_rep='')
        .str.cat(snaps.get('Market', ''), sep='_', na_rep='')
        .str.cat(snaps.get('Outcome', ''), sep='_', na_rep='')
    )

    # ---- group snapshots for fast lookup ----
    # (keys are strings now; no category math will be attempted)
    group_keys = ['Game_Key', 'Market', 'Outcome', 'Bookmaker']
    for k in group_keys:
        if k not in snaps.columns:
            snaps[k] = pd.Series(pd.NA, index=snaps.index, dtype='string')
    snapshots_grouped = snaps.groupby(group_keys, sort=False, observed=True)

    enriched_rows = []
    for idx, row in df.iterrows():
        gk = row['Game_Key']
        market = row['Market']
        outcome = row['Outcome']
        book = row['Bookmaker']

        try:
            group = snapshots_grouped.get_group((gk, market, outcome, book))
        except KeyError:
            default_metrics = {
                'Sharp_Move_Signal': 0,
                'Opening_Limit': None,
                'Sharp_Line_Magnitude': 0.0,
                'Sharp_Limit_Jump': 0,
                'Sharp_Limit_Total': 0.0,
                'SharpMove_Timing_Dominant': 'unknown',
                'Net_Line_Move_From_Opening': None,
                'Abs_Line_Move_From_Opening': None,
                'Net_Odds_Move_From_Opening': None,
                'Abs_Odds_Move_From_Opening': None,
                'SharpMove_Timing_Magnitude': 0.0,
                'Odds_Move_Magnitude': 0.0,
                'SharpBetScore': 0.0,
                **{f'SharpMove_Magnitude_{b}': 0.0 for b in [
                    f'{tod}_{mtg}' for tod in ['Overnight','Early','Midday','Late']
                                   for mtg in ['VeryEarly','MidRange','LateGame','Urgent']
                ]},
                **{f'OddsMove_Magnitude_{b}': 0.0 for b in [
                    f'{tod}_{mtg}' for tod in ['Overnight','Early','Midday','Late']
                                   for mtg in ['VeryEarly','MidRange','LateGame','Urgent']
                ]},
            }
            # keep original row columns
            enriched = {**row.to_dict(), **default_metrics}
            enriched_rows.append(enriched)
            continue

        # sorted by time once
        if 'Snapshot_Timestamp' in group.columns:
            group = group.sort_values('Snapshot_Timestamp')

        # openers
        game_start = group['Game_Start'].dropna().iloc[0] if 'Game_Start' in group and not group['Game_Start'].isna().all() else None
        open_val   = group['Value'].dropna().iloc[0]      if 'Value' in group and not group['Value'].isna().all() else None
        open_odds  = group['Odds_Price'].dropna().iloc[0] if 'Odds_Price' in group and not group['Odds_Price'].isna().all() else None
        opening_limit = group['Limit'].dropna().iloc[0]   if 'Limit' in group and not group['Limit'].isna().all() else None

        # entries tuple for metrics
        entries = list(zip(
            group.get('Limit', pd.Series(index=group.index, dtype='float32')),
            group.get('Value', pd.Series(index=group.index, dtype='float32')),
            group.get('Snapshot_Timestamp', pd.Series(index=group.index, dtype='datetime64[ns, UTC]')),
            [game_start] * len(group),
            group.get('Odds_Price', pd.Series(index=group.index, dtype='float32')),
        ))

        metrics = compute_sharp_metrics(
            entries=entries,
            open_val=open_val,
            mtype=market,
            label=outcome,
            gk=gk,
            book=book,
            open_odds=open_odds,
            opening_limit=opening_limit,
        )

        enriched_rows.append({**row.to_dict(), **metrics})

    return pd.DataFrame(enriched_rows)

    
def compute_all_sharp_metrics(df_all_snapshots):
    results = []

    grouped = df_all_snapshots.groupby(['Game_Key', 'Market', 'Outcome', 'Bookmaker'])

    for (gk, market, outcome, book), group in grouped:
        game_start = (
            group['Game_Start'].dropna().iloc[0]
            if 'Game_Start' in group and not group['Game_Start'].isnull().all()
            else None
        )
        open_val = (
            group.sort_values('Snapshot_Timestamp')['Value']
            .dropna().iloc[0]
            if not group['Value'].isnull().all()
            else None
        )
        open_odds = (
            group.sort_values('Snapshot_Timestamp')['Odds_Price']
            .dropna().iloc[0]
            if not group['Odds_Price'].isnull().all()
            else None
        )

        entries = list(zip(
            group['Limit'],
            group['Value'],
            group['Snapshot_Timestamp'],
            [game_start] * len(group),
            group['Odds_Price']
        ))

        metrics = compute_sharp_metrics(
            entries, open_val=open_val, mtype=market, label=outcome,
            gk=gk, book=book, open_odds=open_odds
        )
        metrics.update({
            'Game_Key': gk,
            'Market': market,
            'Outcome': outcome,
            'Bookmaker': book
        })
        results.append(metrics)

    return pd.DataFrame(results)
    
    
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

KEY_LINE_RESISTANCE = {
    'NFL': {'spread': [3, 7, 10, 14], 'total': [41, 44, 47, 51]},
    'NBA': {'spread': [2.5, 5, 7, 10], 'total': [210, 220, 225, 230]},
    'WNBA': {
        'spread': [1.5, 3.5, 6.5, 9.5],     # updated with common clustering points
        'total': [157.5, 162.5, 167.5, 172.5]  # reflects key ranges in WNBA totals
    },
    'CFL': {
        'spread': [2.5, 4.5, 6.5, 9.5],      # narrower CFL games; 6.5 is key for TDs
        'total': [48.5, 52.5, 55.5, 58.5]    # aligns with common key total zones
    },
    'NCAAF': {'spread': [3, 7, 10, 14, 17], 'total': [45, 52, 59, 66]},
    'NCAAB': {'spread': [2, 5, 7, 10], 'total': [125, 135, 145, 150]},
    'MLB': {
        'spread': [],                        # runline is fixed at -1.5 / +1.5
        'total': [6.5, 7, 7.5, 8, 8.5, 9]     # updated with common clustering points
    },
    'NHL': {'spread': [], 'total': [5.5, 6, 6.5, 7]},
}


def was_line_resistance_broken(open_val, close_val, key_levels, market_type):
    if pd.isna(open_val) or pd.isna(close_val):
        return 0, []

    # Use absolute values for spread resistance checks
    if market_type == 'spread':
        open_val, close_val = abs(open_val), abs(close_val)
        key_levels = [abs(k) for k in key_levels]

    crossed = [
        key for key in key_levels
        if (open_val < key < close_val) or (close_val < key < open_val)
    ]

    return int(bool(crossed)), crossed



def compute_line_resistance_flag(df, source='moves'):
    # Normalize sport keys using SPORT_ALIAS
    df['Sport'] = df['Sport'].str.upper().map(SPORT_ALIAS).fillna(df['Sport'].str.upper())

    def get_key_levels(sport, market):
        if not sport or not market:
            return []
        return KEY_LINE_RESISTANCE.get(sport, {}).get(market.lower(), [])

    def get_opening_line(row):
        return row.get('Open_Value') if source == 'moves' else row.get('First_Line_Value')

    def apply_resistance_logic(row):
        open_val = get_opening_line(row)
        close_val = row.get('Value')
        market_type = row.get('Market', '').lower()
        key_levels = get_key_levels(row.get('Sport', ''), market_type)

        flag, levels_crossed = was_line_resistance_broken(open_val, close_val, key_levels, market_type)
        return pd.Series({
            'Was_Line_Resistance_Broken': flag,
            'Line_Resistance_Crossed_Levels': levels_crossed,
            'Line_Resistance_Crossed_Count': len(levels_crossed)
        })

    resistance_flags = df.apply(apply_resistance_logic, axis=1)
    df = pd.concat([df, resistance_flags], axis=1)
    return df


def compute_sharp_magnitude_by_time_bucket(df_all_snapshots):
    results = []

    grouped = df_all_snapshots.groupby(['Game_Key', 'Market', 'Outcome', 'Bookmaker'])

    for (gk, market, outcome, book), group in grouped:
        logging.debug(f"📊 Group: Game={gk}, Market={market}, Outcome={outcome}, Book={book}")
        logging.debug(f"📋 Columns in group: {list(group.columns)}")

        game_start = (
            group['Game_Start'].dropna().iloc[0]
            if 'Game_Start' in group and not group['Game_Start'].isnull().all()
            else None
        )

        entries = list(zip(
            group['Limit'],
            group['Value'],
            group['Snapshot_Timestamp'],
            [game_start] * len(group)  # Apply game_start to all entries
        ))

        open_val = (
            group.sort_values('Snapshot_Timestamp')['Value']
            .dropna().iloc[0]
            if not group['Value'].isnull().all()
            else None
        )

        metrics = compute_sharp_metrics(entries, open_val, market, outcome)
        metrics.update({
            'Game_Key': gk,
            'Market': market,
            'Outcome': outcome,
            'Bookmaker': book
        })
        results.append(metrics)

    return pd.DataFrame(results)
        
def add_minutes_to_game(df):
    df = df.copy()

    if 'Commence_Hour' not in df.columns or 'Snapshot_Timestamp' not in df.columns:
        logger.warning("⏳ Skipping time-to-game calculation — missing 'Commence_Hour' or 'Snapshot_Timestamp'")
        df['Minutes_To_Game'] = None
        df['Timing_Tier'] = None
        return df

    df['Commence_Hour'] = pd.to_datetime(df['Commence_Hour'], utc=True, errors='coerce')
    df['Snapshot_Timestamp'] = pd.to_datetime(df['Snapshot_Timestamp'], utc=True, errors='coerce')

    df['Minutes_To_Game'] = (
        (df['Commence_Hour'] - df['Snapshot_Timestamp'])
        .dt.total_seconds() / 60
    ).clip(lower=0)

    df['Timing_Tier'] = pd.cut(
        df['Minutes_To_Game'],
        bins=[0, 60, 360, 1440, float('inf')],  # <1h, 1–6h, 6–24h, >24h
        labels=[
            '🔥 Late (<1h)',
            '⚠️ Mid (1–6h)',
            '⏳ Early (6–24h)',
            '🧊 Very Early (>24h)',
        ],
        right=False
    )

    return df

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

          
def hydrate_inverse_rows_from_snapshot(df_inverse: pd.DataFrame, df_all_snapshots: pd.DataFrame) -> pd.DataFrame:
    df = df_inverse.copy()

    # Normalize bookmaker names
    df['Bookmaker'] = df.apply(lambda row: normalize_book_name(row.get('Bookmaker'), row.get('Book')), axis=1)
    df_all_snapshots['Bookmaker'] = df_all_snapshots.apply(lambda row: normalize_book_name(row.get('Bookmaker'), row.get('Book')), axis=1)

    # ⛔️ DO NOT FLIP OUTCOMES — inverse rows are already constructed correctly
    # df['Outcome'] = df.apply(get_inverse_label, axis=1)
    # df['Outcome_Norm'] = df['Outcome']

    # Ensure Commence_Hour is normalized
    df['Commence_Hour'] = pd.to_datetime(df['Commence_Hour'], utc=True, errors='coerce').dt.floor('h')
    df['Team_Key'] = (
        df['Home_Team_Norm'] + "_" +
        df['Away_Team_Norm'] + "_" +
        df['Commence_Hour'].astype(str) + "_" +
        df['Market'] + "_" +
        df['Outcome']
    )

    # Build snapshot Team_Key
    df_all_snapshots['Commence_Hour'] = pd.to_datetime(df_all_snapshots['Game_Start'], utc=True, errors='coerce').dt.floor('h')
    df_all_snapshots['Team_Key'] = (
        df_all_snapshots['Home_Team_Norm'] + "_" +
        df_all_snapshots['Away_Team_Norm'] + "_" +
        df_all_snapshots['Commence_Hour'].astype(str) + "_" +
        df_all_snapshots['Market'] + "_" +
        df_all_snapshots['Outcome']
    )

    # Merge latest snapshot per (Team_Key, Bookmaker)
    df_latest = (
        df_all_snapshots
        .sort_values('Snapshot_Timestamp', ascending=False)
        .drop_duplicates(subset=['Team_Key', 'Bookmaker'])
        [['Team_Key', 'Bookmaker', 'Value', 'Odds_Price', 'Limit']]
        .rename(columns={
            'Value': 'Value_opponent',
            'Odds_Price': 'Odds_Price_opponent',
            'Limit': 'Limit_opponent'
        })
    )

    df = df.merge(df_latest, on=['Team_Key', 'Bookmaker'], how='left')

    # Only overwrite if snapshot value is available
    for col in ['Value', 'Odds_Price', 'Limit']:
        opp_col = f"{col}_opponent"
        if opp_col in df.columns:
            df[col] = np.where(df[opp_col].notna(), df[opp_col], df[col])

    df.drop(columns=['Value_opponent', 'Odds_Price_opponent', 'Limit_opponent'], errors='ignore', inplace=True)
    return df


def fallback_flip_inverse_rows(df_inverse: pd.DataFrame) -> pd.DataFrame:
    # Only flip if Value is missing
    missing_value_mask = df_inverse['Value'].isnull()

    df_to_flip = df_inverse[missing_value_mask].copy()
    if df_to_flip.empty:
        return df_inverse  # Nothing to flip

    logger.info(f"🔁 Fallback flipping {len(df_to_flip)} inverse rows missing value...")

    if 'Market' in df_to_flip.columns:
        df_to_flip.loc[df_to_flip['Market'] == 'spreads', 'Value'] *= -1
        df_to_flip.loc[df_to_flip['Market'] == 'totals', 'Outcome_Norm'] = df_to_flip['Outcome_Norm'].map(
            {'over': 'under', 'under': 'over'}
        )
        df_to_flip.loc[df_to_flip['Market'] == 'totals', 'Outcome'] = df_to_flip['Outcome_Norm']
    
    df_inverse.update(df_to_flip)
    return df_inverse


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
    "NFL":   dict(points_per_elo=25.0, HFA=1.6, sigma_pts=13.2),
    "NCAAF": dict(points_per_elo=28.0, HFA=2.4, sigma_pts=16.0),
    "NBA":   dict(points_per_elo=28.0, HFA=2.2, sigma_pts=11.5),
    "WNBA":  dict(points_per_elo=28.0, HFA=2.0, sigma_pts=10.5),
    "CFL":   dict(points_per_elo=26.0, HFA=1.8, sigma_pts=14.0),
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
    ppe = np.full(n, 27.0, dtype=np.float32)
    hfa = np.zeros(n, dtype=np.float32)
    sig = np.full(n, 12.0, dtype=np.float32)
    for name, cfg in SPORT_SPREAD_CFG.items():
        m = (s == name)
        if m.any():
            ppe[m] = np.float32(cfg["points_per_elo"])
            hfa[m] = np.float32(cfg["HFA"])
            sig[m] = np.float32(cfg["sigma_pts"])
    return ppe, hfa, sig

# ---- 1) Build consensus market spread per game (no BQ) ----

def favorite_centric_from_powerdiff_lowmem(g_full: pd.DataFrame) -> pd.DataFrame:
    out = g_full.copy()
    # normalize for safety
    out["Sport"] = out["Sport"].astype(str).str.upper().str.strip()
    for c in ["Home_Team_Norm","Away_Team_Norm"]:
        out[c] = out[c].astype(str).str.lower().str.strip()

    pr_diff = pd.to_numeric(out.get("Power_Rating_Diff"), errors="coerce").fillna(0).astype("float32")
    ppe, hfa, sig = _get_cfg_vec(out["Sport"])

    # model margin (home - away) in points with HFA
    mu = (pr_diff + hfa) / ppe
    mu = mu.astype("float32")
    out["Model_Expected_Margin"] = mu
    out["Model_Expected_Margin_Abs"] = np.abs(mu, dtype=np.float32)
    out["Sigma_Pts"] = sig

    # model favorite/underdog teams
    fav_is_home = (mu >= 0)
    out["Model_Favorite_Team"] = np.where(fav_is_home, out["Home_Team_Norm"], out["Away_Team_Norm"])
    out["Model_Underdog_Team"] = np.where(fav_is_home, out["Away_Team_Norm"], out["Home_Team_Norm"])

    # model spreads: fav negative, dog positive
    out["Model_Fav_Spread"] = (-np.abs(mu)).astype("float32")
    out["Model_Dog_Spread"] = (+np.abs(mu)).astype("float32")

    fav_mkt = pd.to_numeric(out.get("Favorite_Market_Spread"), errors="coerce").astype("float32")
    dog_mkt = pd.to_numeric(out.get("Underdog_Market_Spread"), errors="coerce").astype("float32")
    k = np.where(np.isnan(fav_mkt), np.abs(dog_mkt), np.abs(fav_mkt)).astype("float32")

    # Edges
    out["Fav_Edge_Pts"] = (fav_mkt - out["Model_Fav_Spread"]).astype("float32")   # = mu_abs - k (sign-consistent)
    out["Dog_Edge_Pts"] = (dog_mkt - out["Model_Dog_Spread"]).astype("float32")   # = k - mu_abs

    # Cover probabilities with Normal(mu, sigma)
    eps = np.finfo("float32").eps
    z_fav = (np.abs(mu) - k) / np.where(sig == 0, eps, sig)  # P(margin > k) for favorite
    z_dog = (k - np.abs(mu)) / np.where(sig == 0, eps, sig)  # P(margin < k) for dog
    out["Fav_Cover_Prob"] = (1.0 - _phi(-z_fav)).astype("float32")  # = 1 - Φ((k-μ_abs)/σ)
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

def apply_blended_sharp_score(df, trained_models, df_all_snapshots=None, weights=None):
    logger.info("🛠️ Running `apply_blended_sharp_score()`")
    df_empty = pd.DataFrame()
    total_start = time.time()
    scored_all = []
    # ---------- models presence ----------
    trained_models = trained_models or {}
    trained_models_lc = {
        str(k).strip().lower(): v
        for k, v in trained_models.items()
        if isinstance(v, dict)
    }
    HAS_MODELS = any(
        isinstance(v, dict) and ("model" in v or "calibrator" in v)
        for v in trained_models_lc.values()
    )
    model_markets_lower = {
        mk for mk, bundle in trained_models_lc.items()
        if isinstance(bundle, dict) and ("model" in bundle or "calibrator" in bundle)
    }
    logger.info("📦 HAS_MODELS=%s; model markets: %s", HAS_MODELS, sorted(model_markets_lower))

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

  

    # ---------- 0) Cast once up front ----------
    odds_now   = pd.to_numeric(df['Odds_Price'], errors='coerce').astype('float64')
    odds_open  = pd.to_numeric(df['Open_Odds'],  errors='coerce').astype('float64')
    val_now    = pd.to_numeric(df['Value'],      errors='coerce').astype('float64')
    val_open   = pd.to_numeric(df['Open_Value'], errors='coerce').astype('float64')
    df['Limit'] = pd.to_numeric(df['Limit'],     errors='coerce').fillna(0.0).astype('float64')
    
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
    
    # ---------- 4) Value reversal (vectorized, single cast) ----------
    def compute_value_reversal(df: pd.DataFrame, market_col: str = 'Market') -> pd.DataFrame:
        m = df[market_col].astype(str).str.lower().str.strip()
        is_spread = m.str.contains('spread', na=False).values
        is_total  = m.str.contains('total',  na=False).values
        is_h2h    = m.str_contains('h2h',    na=False).values if hasattr(m, "str_contains") else m.str.contains('h2h', na=False).values
    
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
    df['Limit_Max'] = df.groupby(['Game', 'Market'], dropna=False)['Limit_NonZero'].transform('max')
    df['Limit_Min'] = df.groupby(['Game', 'Market'], dropna=False)['Limit_NonZero'].transform('min')

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

    # === Flag Pinnacle no-move behavior
    df['Is_Pinnacle'] = (df['Book'].astype(str).str.lower().str.strip() == 'pinnacle')
    df['LimitUp_NoMove_Flag'] = (
        (df['Is_Pinnacle']) &
        (df['Limit'] >= 2500) &
        (df['Value'] == df['Open_Value'])
    ).astype(int)
        
    
    # === Cross-market support (optional)
    df = detect_cross_market_sharp_support(df, SHARP_BOOKS)
    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Unique_Sharp_Books'] = df['Unique_Sharp_Books'].fillna(0).astype(int)
    df['LimitUp_NoMove_Flag'] = df['LimitUp_NoMove_Flag'].fillna(False).astype(int)
    df['Market_Leader'] = df['Market_Leader'].fillna(False).astype(int)
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

    # --- Restore: simple map lookup from  bundles (do this BEFORE enrichment log) ---
    team_feature_map = None
    book_reliability_map = None
    if isinstance(trained_models_lc, dict):
        for bundle in trained_models_lc.values():
            if isinstance(bundle, dict):
                if team_feature_map is None:
                    team_feature_map = bundle.get('team_feature_map')
                if book_reliability_map is None:
                    book_reliability_map = bundle.get('book_reliability_map')
                if team_feature_map is not None and book_reliability_map is not None:
                    break


    # === Cross-Market Odds Pivot
    odds_pivot = (
        df.drop_duplicates(subset=['Game_Key', 'Market', 'Outcome'])
          .pivot_table(index='Game_Key', columns='Market', values='Odds_Price')
          .rename(columns={'spreads': 'Spread_Odds', 'totals': 'Total_Odds', 'h2h': 'H2H_Odds'})
          .reset_index()
    )
    df = df.merge(odds_pivot, on='Game_Key', how='left')

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
    df = add_line_and_crossmarket_features(df)
    
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
    
    # ===== Team features (per-market map → merge onto canon) =====
    if team_feature_map is not None and hasattr(team_feature_map, 'empty') and not team_feature_map.empty:
        # make sure the map is 1-row per team to avoid row multiplication
        tfm = team_feature_map.copy()
        tfm['Team'] = tfm['Team'].astype(str).str.lower().str.strip()
        tfm = tfm.drop_duplicates(subset=['Team'], keep='last')
    
        df['Team'] = df['Outcome_Norm'].astype(str).str.strip().str.lower()
        df = df.merge(tfm, on='Team', how='left')
        df.drop(columns=['Team'], inplace=True, errors='ignore')
    
    # ---- Ensure Market_norm (and keep it compact) ----
    if 'Market_norm' not in df.columns:
        if 'Market' in df.columns and not df.empty:
            m = df['Market'].astype('string')
            df['Market_norm'] = m.str.lower().str.strip().astype('category')
        else:
            df['Market_norm'] = pd.Series(pd.NA, index=df.index, dtype='category')
         # ---- Ensure Market_norm (and Outcome_Norm) exist before use ----
   
    
    # Outcome_Norm: safe normalized outcome for downstream keys
    if 'Outcome_Norm' not in df.columns:
        if 'Outcome' in df.columns and not df.empty:
            df['Outcome_Norm'] = (
                df['Outcome'].astype('string')
                              .str.lower()
                              .str.strip()
            )
        else:
            df['Outcome_Norm'] = pd.Series(pd.Categorical([None] * len(df)), index=df.index)
      
    # 4) Determine markets present to score (only those we actually have trained bundles for)
    if HAS_MODELS and not df.empty:
        markets_present = [
            mk for mk in df['Market_norm'].dropna().unique().tolist()
            if mk in model_markets_lower
        ]
    else:
        markets_present = []
    
    # ---------- NO-MODEL / NO-ELIGIBLE-MARKETS EARLY EXIT ----------
    if (not HAS_MODELS) or (len(markets_present) == 0):
        logger.info("ℹ️ No trained models / no eligible markets present; returning minimally-enriched, unscored snapshots.")
    
        base = df.copy()
    
        # Stamp prediction columns (no placeholders helper)
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
                categories=["❔ No Model", "✅ Coinflip", "⭐ Lean", "🔥 Strong Indication", "🔥 Steam"]
            )
        else:
            # if present, set to "No Model" for all
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
    
        # Team_Key without np.char.* (prevents the TypeError you hit)
        for c, default in [
            ('Home_Team_Norm',''), ('Away_Team_Norm',''), ('Market',''),
            ('Outcome_Norm',''),   ('Outcome','')
        ]:
            if c not in base.columns:
                base[c] = default
    
        # Prefer Outcome_Norm when available, else Outcome
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
    
        logger.info("✅ Returning %d unscored rows (no models).", len(base))
        return base

    # 5) Scoring loop (no 'continue' anywhere)
    for mkt in markets_present:
        mask = (df['Market_norm'] == mkt)
        df_m = df.loc[mask].copy()
    
        # Resolve bundle inside loop
        bundle = trained_models_lc.get(mkt) if HAS_MODELS else None
        model = bundle.get('model') if bundle else None
        iso   = bundle.get('calibrator') if bundle else None
    
       
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
        else:
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

            if 'Odds_Price' in df_canon.columns:
                df_canon['Odds_Price'] = pd.to_numeric(df_canon['Odds_Price'], errors='coerce')
            else:
                df_canon['Odds_Price'] = np.nan
            df_canon['Implied_Prob'] = df_canon['Odds_Price'].apply(implied_prob)

            df_canon['High_Limit_Flag'] = (
                (pd.to_numeric(df_canon.get('Sharp_Limit_Total', np.nan), errors='coerce') >= 10000).astype(int)
            )
            df_canon['Is_Home_Team_Bet'] = (
                (df_canon['Outcome'] == df_canon.get('Home_Team_Norm','')).astype(int)
            )
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
            
            # --- crossed levels column (NA-safe, no boolean NA) ---
            col = 'Line_Resistance_Crossed_Levels'
            if col not in df_canon.columns:
                df_canon[col] = "[]"
            else:
                s = df_canon[col].astype('object')
                # vectorized-ish: handle NA, listlikes, scalars without "if x" on pd.NA
                na_mask    = s.isna()
                list_mask  = s.map(lambda v: isinstance(v, (list, tuple, set, np.ndarray)))
                str_mask   = s.map(lambda v: isinstance(v, str))
                other_mask = ~(na_mask | list_mask | str_mask)
            
                s.loc[na_mask]   = "[]"
                s.loc[list_mask] = s.loc[list_mask].map(lambda v: json.dumps(list(v)))
                s.loc[other_mask]= s.loc[other_mask].map(lambda v: json.dumps([v]))
                df_canon[col] = s.astype('string')
            
            # Count column: ensure numeric, fill NA
            cnt_col = 'Line_Resistance_Crossed_Count'
            if cnt_col not in df_canon.columns:
                df_canon[cnt_col] = 0
            else:
                df_canon[cnt_col] = pd.to_numeric(df_canon[cnt_col], errors='coerce').fillna(0).astype('int64')
            
            # --- timing (parse once + compute) ---
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
    
            # These require team map; if absent, create zeros to avoid KeyError
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
    
            # Your existing extras
           
           
            # Numeric timing columns
            hybrid_timing_cols = [
                'SharpMove_Magnitude_Overnight_VeryEarly','SharpMove_Magnitude_Overnight_MidRange',
                'SharpMove_Magnitude_Overnight_LateGame','SharpMove_Magnitude_Overnight_Urgent',
                'SharpMove_Magnitude_Early_VeryEarly','SharpMove_Magnitude_Early_MidRange',
                'SharpMove_Magnitude_Early_LateGame','SharpMove_Magnitude_Early_Urgent',
                'SharpMove_Magnitude_Midday_VeryEarly','SharpMove_Magnitude_Midday_MidRange',
                'SharpMove_Magnitude_Midday_LateGame','SharpMove_Magnitude_Midday_Urgent',
                'SharpMove_Magnitude_Late_VeryEarly','SharpMove_Magnitude_Late_MidRange',
                'SharpMove_Magnitude_Late_LateGame','SharpMove_Magnitude_Late_Urgent',
                'SharpMove_Timing_Magnitude'
            ]
            for col in hybrid_timing_cols:
                df_canon[col] = pd.to_numeric(df_canon[col], errors='coerce').fillna(0.0) if col in df_canon.columns else 0.0
    
            if 'SharpMove_Timing_Dominant' not in df_canon.columns:
                df_canon['SharpMove_Timing_Dominant'] = 'unknown'
    
            hybrid_odds_timing_cols = ['Odds_Move_Magnitude'] + [
                f'OddsMove_Magnitude_{b}' for b in [
                    'Overnight_VeryEarly','Overnight_MidRange','Overnight_LateGame','Overnight_Urgent',
                    'Early_VeryEarly','Early_MidRange','Early_LateGame','Early_Urgent',
                    'Midday_VeryEarly','Midday_MidRange','Midday_LateGame','Midday_Urgent',
                    'Late_VeryEarly','Late_MidRange','Late_LateGame','Late_Urgent'
                ]
            ]
            for col in hybrid_odds_timing_cols:
                df_canon[col] = pd.to_numeric(df_canon[col], errors='coerce').fillna(0.0) if col in df_canon.columns else 0.0
    
            # ===== MODEL SCORING =====
            feature_cols = []
            # ===== MODEL SCORING =====
            feature_cols = []
            if bundle and isinstance(bundle.get('feature_cols'), (list, tuple)) and bundle['feature_cols']:
                feature_cols = list(bundle['feature_cols'])
            elif model is not None:
                feat_in = getattr(model, 'feature_names_in_', None)
                if feat_in is not None:
                    feature_cols = list(feat_in)
                else:
                    booster = getattr(model, 'get_booster', lambda: None)()
                    names = getattr(booster, 'feature_names', None)
                    feature_cols = list(names) if names is not None else []
            
            # ---- scoring (wrapped) ----
            try:
                feature_cols = [str(c) for c in feature_cols]
                feature_cols = list(dict.fromkeys(feature_cols))  # de-dupe, keep order
            
                # coerce only model features
                for c in feature_cols:
                    if c in df_canon.columns:
                        if df_canon[c].dtype == object:
                            df_canon[c] = df_canon[c].replace({
                                'True': 1, 'False': 0, True: 1, False: 0,
                                '': np.nan, 'none': np.nan, 'None': np.nan
                            })
                        df_canon[c] = pd.to_numeric(df_canon[c], errors='coerce')
            
                if feature_cols:
                    # ensure all features exist, then clean
                    df_canon[feature_cols] = (
                        df_canon.reindex(columns=feature_cols)
                                .replace([np.inf, -np.inf], np.nan)
                                .fillna(0.0)
                    )
            
                # ensure pred cols exist
                for col, default in [
                    ('Model_Sharp_Win_Prob', np.nan),
                    ('Model_Confidence',     np.nan),
                    ('Scored_By_Model',      False),
                    ('Scoring_Market',       mkt),
                ]:
                    if col not in df_canon.columns:
                        df_canon[col] = default
            
                # predict or stamp unscored
                if not feature_cols:
                    logger.info(f"ℹ️ {mkt.upper()} has no feature_cols — stamping unscored columns on canon.")
                    for col, default in [
                        ('Model_Sharp_Win_Prob', np.nan),
                        ('Model_Confidence',     np.nan),
                        ('Scored_By_Model',      False),
                        ('Scoring_Market',       mkt),
                    ]:
                        df_canon[col] = default
                else:
                    X_can = df_canon.loc[:, feature_cols]
                    if X_can.empty:
                        logger.info(f"ℹ️ {mkt.upper()} has empty X_can — stamping unscored columns on canon.")
                        for col, default in [
                            ('Model_Sharp_Win_Prob', np.nan),
                            ('Model_Confidence',     np.nan),
                            ('Scored_By_Model',      False),
                            ('Scoring_Market',       mkt),
                        ]:
                            df_canon[col] = default
                    else:
                        preds = None
                        # prefer calibrator
                        if iso is not None:
                            if hasattr(iso, "predict_proba"):
                                p = iso.predict_proba(X_can)
                                preds = p[:, 1] if isinstance(p, np.ndarray) and p.ndim > 1 else np.asarray(p)
                            elif hasattr(iso, "predict"):
                                preds = np.asarray(iso.predict(X_can))
                        # fallback to model
                        if preds is None and model is not None:
                            if hasattr(model, "predict_proba"):
                                p = model.predict_proba(X_can)
                                preds = p[:, 1] if isinstance(p, np.ndarray) and p.ndim > 1 else np.asarray(p)
                            elif hasattr(model, "predict"):
                                preds = np.asarray(model.predict(X_can))
            
                        if preds is None:
                            logger.info(f"ℹ️ {mkt.upper()} has no usable predictor — stamping unscored columns on canon.")
                            for col, default in [
                                ('Model_Sharp_Win_Prob', np.nan),
                                ('Model_Confidence',     np.nan),
                                ('Scored_By_Model',      False),
                                ('Scoring_Market',       mkt),
                            ]:
                                df_canon[col] = default
                        else:
                            df_canon['Model_Sharp_Win_Prob'] = preds
                            df_canon['Model_Confidence']     = preds
                            df_canon['Scored_By_Model']      = True
                            df_canon['Scoring_Market']       = mkt
            
            except Exception as e:
                logger.error(f"❌ Scoring failed for {mkt.upper()} — stamping unscored columns on canon. Error: {e}")
                for col, default in [
                    ('Model_Sharp_Win_Prob', np.nan),
                    ('Model_Confidence',     np.nan),
                    ('Scored_By_Model',      False),
                    ('Scoring_Market',       mkt),
                ]:
                    df_canon[col] = default
            
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
                df_inverse['Sharp_Line_Magnitude'] = df_inverse['Sharp_Line_Delta'].abs()
                df_inverse['Rec_Line_Magnitude']   = df_inverse['Rec_Line_Delta'].abs()
                df_inverse['High_Limit_Flag']      = (df_inverse.get('Sharp_Limit_Total', 0) >= 10000).astype(int)
            
                if 'Odds_Shift' in df_inverse.columns:
                    df_inverse['SharpMove_OddsShift'] = df_inverse['Sharp_Move_Signal'] * df_inverse['Odds_Shift']
                if 'Implied_Prob_Shift' in df_inverse.columns:
                    df_inverse['MarketLeader_ImpProbShift'] = df_inverse['Market_Leader'] * df_inverse['Implied_Prob_Shift']
            
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
            
                df_inverse['SharpMove_Odds_Up']   = ((df_inverse['Sharp_Move_Signal'] == 1) & (df_inverse['Odds_Shift'] > 0)).astype(int)
                df_inverse['SharpMove_Odds_Down'] = ((df_inverse['Sharp_Move_Signal'] == 1) & (df_inverse['Odds_Shift'] < 0)).astype(int)
                df_inverse['SharpMove_Odds_Mag']  = df_inverse['Odds_Shift'].abs() * df_inverse['Sharp_Move_Signal']
            
                df_inverse['Net_Line_Move_From_Opening'] = df_inverse['Value'] - df_inverse['Open_Value']
                df_inverse['Abs_Line_Move_From_Opening'] = df_inverse['Net_Line_Move_From_Opening'].abs()
            
                # Use implied probability deltas consistently
                df_inverse['Net_Odds_Move_From_Opening'] = (
                    df_inverse['Odds_Price'].apply(implied_prob) - df_inverse['Open_Odds'].apply(implied_prob)
                ) * 100.0
                df_inverse['Abs_Odds_Move_From_Opening'] = df_inverse['Net_Odds_Move_From_Opening'].abs()
            
                df_inverse['Line_Resistance_Crossed_Levels'] = df_inverse.get('Line_Resistance_Crossed_Levels', '[]')
                df_inverse['Line_Resistance_Crossed_Count']  = df_inverse.get('Line_Resistance_Crossed_Count', 0)
                col = 'Line_Resistance_Crossed_Levels'
                if col not in df_inverse.columns:
                    df_inverse[col] = "[]"
                else:
                    s = df_inverse[col].astype('object')
                    # optional vectorized-ish speedup; otherwise: s = s.map(_levels_to_jsonlike)
                    na_mask    = s.isna()
                    list_mask  = s.map(lambda v: isinstance(v, (list, tuple, set, np.ndarray)))
                    str_mask   = s.map(lambda v: isinstance(v, str))
                    other_mask = ~(na_mask | list_mask | str_mask)
                
                    s.loc[na_mask]    = "[]"
                    s.loc[list_mask]  = s.loc[list_mask].map(lambda v: json.dumps(list(v)))
                    s.loc[other_mask] = s.loc[other_mask].map(lambda v: json.dumps([v]))
                    df_inverse[col]   = s.astype('string')
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
            
                for col in hybrid_timing_cols:
                    df_inverse[col] = pd.to_numeric(df_inverse[col], errors='coerce').fillna(0.0) if col in df_inverse.columns else 0.0
                if 'SharpMove_Timing_Dominant' not in df_inverse.columns:
                    df_inverse['SharpMove_Timing_Dominant'] = 'unknown'
                else:
                    df_inverse['SharpMove_Timing_Dominant'] = df_inverse['SharpMove_Timing_Dominant'].fillna('unknown').astype(str)
            
                for col in hybrid_odds_timing_cols:
                    df_inverse[col] = pd.to_numeric(df_inverse[col], errors='coerce').fillna(0.0) if col in df_inverse.columns else 0.0
            
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
                    'SmallBook_Total_Limit','SmallBook_Max_Limit','SmallBook_Min_Limit',
                    'SmallBook_Limit_Skew','SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag'
                ]
                suffix_cols     = [c for c in df_inverse.columns if c.endswith('_x') or c.endswith('_y')]
                team_stat_cols  = [c for c in df_inverse.columns if c.startswith('Team_Past_')]
                drop_cols       = list(set(cols_to_refresh + suffix_cols + team_stat_cols))
                df_inverse = df_inverse.drop(columns=[c for c in drop_cols if c in df_inverse.columns], errors='ignore')
            
                # Re-attach team features (if available)
                try:
                    if team_feature_map is not None and not team_feature_map.empty:
                        df_inverse['Team'] = df_inverse['Outcome_Norm'].astype(str).str.lower().str.strip()
                        df_inverse = df_inverse.merge(team_feature_map, on='Team', how='left')
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
                bins=[0, 0.4, 0.6, 0.8, 1.0],
                labels=["✅ Coinflip","⭐ Lean","🔥 Strong Indication","🔥 Steam"],
                include_lowest=True
            )
            
            scored_all.append(df_scored)   
    try:
        df_final = pd.DataFrame()
    
        hybrid_line_cols = [
            f'SharpMove_Magnitude_{b}' for b in [
                'Overnight_VeryEarly','Overnight_MidRange','Overnight_LateGame','Overnight_Urgent',
                'Early_VeryEarly','Early_MidRange','Early_LateGame','Early_Urgent',
                'Midday_VeryEarly','Midday_MidRange','Midday_LateGame','Midday_Urgent',
                'Late_VeryEarly','Late_MidRange','Late_LateGame','Late_Urgent'
            ]
        ]
        hybrid_odds_cols = [
            f'OddsMove_Magnitude_{b}' for b in [
                'Overnight_VeryEarly','Overnight_MidRange','Overnight_LateGame','Overnight_Urgent',
                'Early_VeryEarly','Early_MidRange','Early_LateGame','Early_Urgent',
                'Midday_VeryEarly','Midday_MidRange','Midday_LateGame','Midday_Urgent',
                'Late_VeryEarly','Late_MidRange','Late_LateGame','Late_Urgent'
            ]
        ]
    
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
                'SharpMove_Odds_Down','SharpMove_Odds_Mag','SharpMove_Resistance_Break','Late_Game_Steam_Flag',
                'Value_Reversal_Flag','Odds_Reversal_Flag','Abs_Line_Move_From_Opening','Abs_Odds_Move_From_Opening',
                'Team_Past_Hit_Rate','Mispricing_Flag','Team_Implied_Prob_Gap_Home','Team_Implied_Prob_Gap_Away',
                'Avg_Recent_Cover_Streak','Avg_Recent_Cover_Streak_Home','Avg_Recent_Cover_Streak_Away',
                'Spread_vs_H2H_Aligned','Total_vs_Spread_Contradiction','CrossMarket_Prob_Gap_Exists',
                'Potential_Overmove_Flag','Potential_Overmove_Total_Pct_Flag','Potential_Odds_Overmove_Flag',
                'Line_Moved_Toward_Team','Line_Moved_Away_From_Team','Line_Resistance_Crossed_Count',
                'Abs_Line_Move_Z','Pct_Line_Move_Z','SmallBook_Heavy_Liquidity_Flag','SmallBook_Limit_Skew_Flag',
                'SmallBook_Total_Limit'
            ] + hybrid_line_cols + hybrid_odds_cols
    
            # Some columns may not exist depending on earlier paths; create them as 0 for counting
            for c in cols_for_count:
                if c not in df_final.columns:
                    df_final[c] = 0
                df_final[c] = pd.to_numeric(df_final[c], errors='coerce').fillna(0)
    
            # 4) hybrid timing flags
            df_final['Hybrid_Line_Timing_Flag'] = (df_final[hybrid_line_cols].sum(axis=1) > 0).astype(int)
            df_final['Hybrid_Odds_Timing_Flag'] = (df_final[hybrid_odds_cols].sum(axis=1) > 0).astype(int)
    
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
                (df_final['SharpMove_Resistance_Break'] == 1).astype(int) +
                (df_final['Late_Game_Steam_Flag'] == 1).astype(int) +
                (df_final['Value_Reversal_Flag'] == 1).astype(int) +
                (df_final['Odds_Reversal_Flag'] == 1).astype(int) +
                (df_final['Abs_Line_Move_From_Opening'] > 1.0).astype(int) +
                (df_final['Abs_Odds_Move_From_Opening'] > 5).astype(int) +
                df_final['Hybrid_Line_Timing_Flag'] +
                df_final['Hybrid_Odds_Timing_Flag'] +
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
                (df_final['Line_Resistance_Crossed_Count'] >= 1).astype(int) +
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
    sport_label: str | None = None,   # e.g. "NBA"
    history_hours: int = 120,         # 0/None to skip history
):
    import logging
    from datetime import datetime
    import numpy as np
    import pandas as pd

    if not current:
        logging.warning("⚠️ No current odds data provided.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ---------- 0) Basics ----------
    df_scored     = pd.DataFrame()
    summary_df    = pd.DataFrame()
    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Normalize sport identifiers (both helpful later)
    _sport_key   = (sport_key or "").strip().lower()      # "basketball_nba"
    _sport_label = (sport_label or _sport_key).strip().upper()  # "NBA" (else upper of key)

  

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
    market_weights = load_market_weights_from_bq()
    now = pd.Timestamp.utcnow()
    df['Snapshot_Timestamp'] = now
    if 'Game_Start' in df.columns:
        df['Event_Date'] = df['Game_Start'].dt.date
    
    # Compute hash before scoring
    df['Line_Hash'] = df.apply(compute_line_hash, axis=1)
    
    # If apply_blended_sharp_score mutates, let it copy internally; otherwise pass df directly.
    df_scored = apply_blended_sharp_score(df, trained_models, df_all_snapshots, market_weights)
    
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

    summary_df['Recommended_Outcome'] = summary_df['Outcome']
    summary_df['Move_From_Open_Rec'] = (summary_df['Rec_Book_Consensus'] - summary_df['Rec_Open']).fillna(0)
    summary_df['Move_From_Open_Sharp'] = (summary_df['Sharp_Book_Consensus'] - summary_df['Sharp_Open']).fillna(0)

    sharp_scores = df[df['SharpBetScore'].notnull()][[
        'Event_Date', 'Game', 'Market', 'Outcome',
        'SharpBetScore', 'Enhanced_Sharp_Confidence_Score', 'Sharp_Confidence_Tier'
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

    return summary_df




        
def detect_market_leaders(df_history, sharp_books, rec_books):
    df_history = df_history.copy()
    df_history['Time'] = pd.to_datetime(df_history['Time'])
    df_history['Book'] = df_history['Book'].str.lower()
    df_history['Book'] = df_history.apply(
        lambda row: normalize_book_name(row['Book'], row.get('Bookmaker')), axis=1
    )


    # === LINE MOVE DETECTION ===
    df_open_line = (
        df_history
        .sort_values('Time')
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Line_Value'})
    )

    df_history = df_history.merge(df_open_line, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df_history['Line_Has_Moved'] = (df_history['Value'] != df_history['Open_Line_Value']) & df_history['Value'].notna()

    first_line_moves = (
        df_history[df_history['Line_Has_Moved']]
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Time']
        .min()
        .reset_index()
        .rename(columns={'Time': 'First_Line_Move_Time'})
    )

    # === ODDS MOVE DETECTION ===
    if 'Odds_Price' in df_history.columns:
        df_open_odds = (
            df_history
            .sort_values('Time')
            .groupby(['Game', 'Market', 'Outcome', 'Book'])['Odds_Price']
            .first()
            .reset_index()
            .rename(columns={'Odds_Price': 'Open_Odds_Price'})
        )
        df_history = df_history.merge(df_open_odds, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
        df_history['Odds_Has_Moved'] = (df_history['Odds_Price'] != df_history['Open_Odds_Price']) & df_history['Odds_Price'].notna()

        first_odds_moves = (
            df_history[df_history['Odds_Has_Moved']]
            .groupby(['Game', 'Market', 'Outcome', 'Book'])['Time']
            .min()
            .reset_index()
            .rename(columns={'Time': 'First_Odds_Move_Time'})
        )
    else:
        first_odds_moves = pd.DataFrame(columns=['Game', 'Market', 'Outcome', 'Book', 'First_Odds_Move_Time'])

    # === Merge both move types ===
    first_moves = pd.merge(first_line_moves, first_odds_moves, on=['Game', 'Market', 'Outcome', 'Book'], how='outer')

    # Identify sharp vs. rec books
    first_moves['Book_Type'] = first_moves['Book'].map(
        lambda b: 'Sharp' if b in sharp_books else ('Rec' if b in rec_books else 'Other')
    )

    # Rank order
    first_moves['Line_Move_Rank'] = first_moves.groupby(
        ['Game', 'Market', 'Outcome']
    )['First_Line_Move_Time'].rank(method='first')

    first_moves['Odds_Move_Rank'] = first_moves.groupby(
        ['Game', 'Market', 'Outcome']
    )['First_Odds_Move_Time'].rank(method='first')

    # === Market Leader Flags ===
    first_moves['Market_Leader_Line'] = (
        (first_moves['Book_Type'] == 'Sharp') & (first_moves['Line_Move_Rank'] == 1)
    )

    first_moves['Market_Leader_Odds'] = (
        (first_moves['Book_Type'] == 'Sharp') & (first_moves['Odds_Move_Rank'] == 1)
    )

    # ✅ Final combined flag — fallback-compatible
    first_moves['Market_Leader'] = (
        (first_moves['Book_Type'] == 'Sharp') &
        (
            (first_moves['Line_Move_Rank'] == 1) |
            (first_moves['Odds_Move_Rank'] == 1)
        )
    ) # change logic here if you want a blend

    return first_moves


def detect_cross_market_sharp_support(df_moves, SHARP_BOOKS):
    df = df_moves.copy()
    df['Market'] = df['Market'].astype(str).str.lower().str.strip()
    df['SupportKey'] = df['Game'].astype(str).str.strip() + " | " + df['Outcome'].astype(str).str.strip()

    # ✅ Step 1: Define a "sharp" row: book is sharp and sharp signal exists
    df['Is_Sharp_Row'] = (
        df['Book'].isin(SHARP_BOOKS) & 
        (df['Sharp_Move_Signal'].astype(bool))  # could also include other flags like Limit_Jump or Prob_Shift
    )

    df_sharp = df[df['Is_Sharp_Row']].copy()

    # ✅ Step 2: Count cross-market sharp support per (Game + Outcome)
    market_counts = (
        df_sharp.groupby('SupportKey')['Market']
        .nunique()
        .reset_index()
        .rename(columns={'Market': 'CrossMarketSharpSupport'})
    )

    # ✅ Step 3: Count how many sharp books have that outcome
    sharp_book_counts = (
        df_sharp.groupby('SupportKey')['Book']
        .nunique()
        .reset_index()
        .rename(columns={'Book': 'Unique_Sharp_Books'})
    )

    # ✅ Step 4: Merge back to original dataframe
    df = df.merge(market_counts, on='SupportKey', how='left')
    df = df.merge(sharp_book_counts, on='SupportKey', how='left')

    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Unique_Sharp_Books'] = df['Unique_Sharp_Books'].fillna(0).astype(int)

    # ✅ Step 5: Final flag
    df['Is_Reinforced_MultiMarket'] = (
        (df['CrossMarketSharpSupport'] >= 2) | (df['Unique_Sharp_Books'] >= 2)
    )

    return df


def load_model_from_gcs(sport, market, bucket_name="sharp-models"):
    filename = f"sharp_win_model_{sport.lower()}_{market.lower()}.pkl"
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        content = blob.download_as_bytes()
        data = pickle.loads(content)

        print(f"✅ Loaded model artifact from GCS: gs://{bucket_name}/{filename}")

        # Normalize optional payloads to DataFrames
        tfm = data.get("team_feature_map", pd.DataFrame())
        brm = data.get("book_reliability_map", pd.DataFrame())

        if not isinstance(tfm, pd.DataFrame) and tfm is not None:
            tfm = pd.DataFrame(tfm)
        if not isinstance(brm, pd.DataFrame) and brm is not None:
            brm = pd.DataFrame(brm)

        return {
            "model": data.get("model"),
            "calibrator": data.get("calibrator"),
            "team_feature_map": tfm,                 # ✅ optional
            "book_reliability_map": brm              # ✅ optional
        }
    except Exception as e:
        logging.warning(f"⚠️ Could not load model from GCS for {sport}-{market}: {e}")
        return None


def read_recent_sharp_moves(hours=120, table=BQ_FULL_TABLE):
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
            'Was_Line_Resistance_Broken',
            'SharpMove_Resistance_Break',
            'Line_Resistance_Crossed_Levels',
            'Line_Resistance_Crossed_Count', 'Late_Game_Steam_Flag', 
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
        # Odds API style keys
        "baseball_mlb": "MLB",
        "americanfootball_nfl": "NFL",
        "americanfootball_ncaaf": "NCAAF",
        "basketball_nba": "NBA",
        "basketball_wnba": "WNBA",
        "canadianfootball_cfl": "CFL",
        # Friendly aliases just in case
        "mlb": "MLB", "nfl": "NFL", "ncaaf": "NCAAF",
        "nba": "NBA", "wnba": "WNBA", "cfl": "CFL",
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
    
    # Only now load snapshots / df_master / df_first, etc.
    df_master = read_recent_sharp_master_cached(hours=24)
    df_master = build_game_key(df_master)

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
    df_all_snapshots = read_recent_sharp_master_cached(hours=120)
    log_memory("AFTER read_recent_sharp_master_cached")
    
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
    need_cols = ["Game_Key","Market","Outcome","Bookmaker","Open_Value","Open_Odds"]
    missing = [c for c in need_cols if c not in df_all_snapshots.columns]
    if missing:
        logging.warning(f"⚠️ Cannot build df_first — missing columns: {missing}")
        df_first = pd.DataFrame(columns=[
            "Game_Key","Market","Outcome","Bookmaker",
            "First_Line_Value","First_Odds","First_Imp_Prob"
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
        'Was_Line_Resistance_Broken',
        'SharpMove_Resistance_Break',
        'Line_Resistance_Crossed_Levels',
        'Line_Resistance_Crossed_Count', 'Late_Game_Steam_Flag', 
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
    df_scores_out['Sport'] = df_scores_out['Sport'].replace({
        'BASEBALL_MLB': 'MLB',
        'BASKETBALL_NBA': 'NBA',
        'BASKETBALL_WNBA': 'WNBA',
        'FOOTBALL_CFL': 'CFL',
        'MLB': 'MLB',  # handles redundancy safely
        'NBA': 'NBA',
        'WNBA': 'WNBA',
        'CFL': 'CFL',
        'NFL': 'NFL',
        'NCAAF': 'NCAAF',
    }).str.upper()
    
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
    df_scores_out['Was_Line_Resistance_Broken'] = pd.to_numeric(df_scores_out['Was_Line_Resistance_Broken'], errors='coerce').astype('Int64')
    df_scores_out['SharpMove_Resistance_Break'] = pd.to_numeric(df_scores_out['SharpMove_Resistance_Break'], errors='coerce').astype('Int64')
    df_scores_out['Line_Resistance_Crossed_Count'] = pd.to_numeric(df_scores_out['Line_Resistance_Crossed_Count'], errors='coerce').astype('Int64')
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


