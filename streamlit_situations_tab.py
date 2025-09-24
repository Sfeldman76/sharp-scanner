# ================== Situations Tab (league-wide by sport; roles from MOVES spreads only) ==================
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, date, time, timezone
from google.cloud import bigquery

# ---------- config ----------
PROJECT = "sharplogger"
DATASET = "sharp_data"
MOVES_FQ  = f"{PROJECT}.{DATASET}.moves_with_features_merged"
SCORES_FQ = f"{PROJECT}.{DATASET}.scores_with_features"

MOVES  = f"`{MOVES_FQ}`"
SCORES = f"`{SCORES_FQ}`"

# sportsbook columns + default
BOOK_COL_MOVES  = "Book"
BOOK_COL_SCORES = "Book"
DEFAULT_BOOK    = "Pinnacle"

CLIENT = bigquery.Client(project=PROJECT)
CACHE_TTL_SEC = 120

# ---------- tiny utils ----------
def _to_utc_dt(x) -> dt.datetime:
    """Coerce anything into a tz‑aware UTC datetime; fallback to now() if bad."""
    if isinstance(x, dt.datetime):
        return x.astimezone(timezone.utc) if x.tzinfo else x.replace(tzinfo=timezone.utc)
    if isinstance(x, dt.date):
        return dt.datetime.combine(x, time(0, 0, 0, tzinfo=timezone.utc))
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
        return ts.to_pydatetime()
    return datetime.now(timezone.utc)

def _sport_is_football(s: str) -> bool:
    return (s or "").upper() in {"NFL", "NCAAF"}

def _sport_is_basketball(s: str) -> bool:
    return (s or "").upper() in {"NBA", "NCAAB", "WNBA"}

def _fb_spread_bucket(v: float | None) -> str:
    if v is None: return ""
    try: v = float(v)
    except Exception: return ""
    if v <= -10.5: return "Fav ≤ -10.5"
    if v <=  -7.5: return "Fav -8 to -10.5"
    if v <=  -6.5: return "Fav -7 to -6.5"
    if v <=  -3.5: return "Fav -4 to -6.5"
    if v <=  -0.5: return "Fav -0.5 to -3.5"
    if v ==   0.0: return "Pick (0)"
    if v <=   3.5: return "Dog +0.5 to +3.5"
    if v <=   6.5: return "Dog +4 to +6.5"
    if v <=  10.5: return "Dog +7 to +10.5"
    return "Dog ≥ +11"

def _bb_spread_bucket(v: float | None) -> str:
    if v is None: return ""
    try: v = float(v)
    except Exception: return ""
    if v <= -12.5: return "Fav ≤ -12.5"
    if v <=  -9.5: return "Fav -10 to -12.5"
    if v <=  -6.5: return "Fav -7 to -9.5"
    if v <=  -3.5: return "Fav -4 to -6.5"
    if v <=  -0.5: return "Fav -0.5 to -3.5"
    if v ==   0.0: return "Pick (0)"
    if v <=   3.5: return "Dog +0.5 to +3.5"
    if v <=   6.5: return "Dog +4 to +6.5"
    if v <=   9.5: return "Dog +7 to +9.5"
    if v <=  12.5: return "Dog +10 to +12.5"
    return "Dog ≥ +13"

def _fb_total_bucket(v: float | None) -> str:
    if v is None: return ""
    try: v = float(v)
    except Exception: return ""
    if v <= 37.5: return "OU ≤ 37.5"
    if v <= 41.5: return "OU 38–41.5"
    if v <= 44.5: return "OU 42–44.5"
    if v <= 47.5: return "OU 45–47.5"
    if v <= 50.5: return "OU 48–50.5"
    if v <= 53.5: return "OU 51–53.5"
    if v <= 56.5: return "OU 54–56.5"
    if v <= 59.5: return "OU 57–59.5"
    return "OU ≥ 60"

def _bb_total_bucket(v: float | None) -> str:
    if v is None: return ""
    try: v = float(v)
    except Exception: return ""
    if v <= 205.5: return "OU ≤ 205.5"
    if v <= 209.5: return "OU 206–209.5"
    if v <= 213.5: return "OU 210–213.5"
    if v <= 217.5: return "OU 214–217.5"
    if v <= 221.5: return "OU 218–221.5"
    if v <= 225.5: return "OU 222–225.5"
    if v <= 229.5: return "OU 226–229.5"
    if v <= 233.5: return "OU 230–233.5"
    return "OU ≥ 234"

def _derive_spread_bucket_for_ui(sport: str, closing_spread: float | None) -> str:
    if closing_spread is None:
        return ""
    if _sport_is_basketball(sport):
        return _bb_spread_bucket(closing_spread)
    return _fb_spread_bucket(closing_spread)

def _wanted_labels(is_home: bool | None, is_favorite: bool | None, bucket: str | None):
    out = []
    if is_home is True:  out.append("Home")
    if is_home is False: out.append("Road")
    if (is_home is not None) and (is_favorite is not None):
        out.append(f"{'Home' if is_home else 'Road'} {'Favorite' if is_favorite else 'Underdog'}")
    if bucket:
        out.append(bucket)
        if is_home is True:  out.append(f"Home · {bucket}")
        if is_home is False: out.append(f"Road · {bucket}")
    # de‑dupe keep order
    seen, keep = set(), []
    for s in out:
        if s and s not in seen:
            keep.append(s); seen.add(s)
    return keep

def _compose_four_views_spreads(league_spreads: pd.DataFrame,
                                is_home: bool | None, is_favorite: bool | None, bucket: str | None) -> pd.DataFrame:
    """
    Build a compact table with up to 4 rows, in this order:
      1) Fav/Dog (Role4)
      2) Home/Road (Venue)
      3) Bucket (overall)
      4) Bucket × Location (Bucket·Venue)
    """
    if league_spreads is None or league_spreads.empty:
        return pd.DataFrame(columns=["Section","GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"])

    out_parts = []

    if is_home is not None and is_favorite is not None:
        role_lbl = f"{'Home' if is_home else 'Road'} {'Favorite' if is_favorite else 'Underdog'}"
        df = league_spreads[(league_spreads["GroupLabel"] == "Role4") &
                            (league_spreads["Situation"]  == role_lbl)]
        if not df.empty:
            df = df.copy(); df.insert(0, "Section", "Role (Fav/Dog)")
            out_parts.append(df)

    if is_home is not None:
        v_lbl = "Home" if is_home else "Road"
        df = league_spreads[(league_spreads["GroupLabel"] == "Venue") &
                            (league_spreads["Situation"]  == v_lbl)]
        if not df.empty:
            df = df.copy(); df.insert(0, "Section", "Venue (Home/Road)")
            out_parts.append(df)

    if bucket:
        df = league_spreads[(league_spreads["GroupLabel"] == "Bucket") &
                            (league_spreads["Situation"]  == bucket)]
        if not df.empty:
            df = df.copy(); df.insert(0, "Section", "Bucket (overall)")
            out_parts.append(df)

        if is_home is not None:
            bvl = f"{'Home' if is_home else 'Road'} · {bucket}"
            df = league_spreads[(league_spreads["GroupLabel"] == "Bucket·Venue") &
                                (league_spreads["Situation"]  == bvl)]
            if not df.empty:
                df = df.copy(); df.insert(0, "Section", "Bucket × Venue")
                out_parts.append(df)

    if not out_parts:
        return pd.DataFrame(columns=["Section","GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"])

    out = pd.concat(out_parts, ignore_index=True)
    for c in ("WinPct","ROI_Pct"):
        if c in out.columns:
            out[c] = out[c].map(lambda x: None if pd.isna(x) else round(float(x), 1))
    order = {"Role (Fav/Dog)":0, "Venue (Home/Road)":1, "Bucket (overall)":2, "Bucket × Venue":3}
    out["__o"] = out["Section"].map(order).fillna(99)
    out = out.sort_values(["__o"]).drop(columns="__o")
    return out[["Section","GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"]]

# ---------- MOVES: current games (spreads only) & roles ----------
@st.cache_data(ttl=90, show_spinner=False)
def list_current_games_from_moves(sport: str, hard_cap: int = 200, book: str = DEFAULT_BOOK) -> pd.DataFrame:
    sql = f"""
    WITH src AS (
      SELECT
        UPPER(Sport) AS Sport_Upper,
        UPPER(Market) AS Market_U,
        TIMESTAMP(COALESCE(Game_Start, Commence_Hour, feat_Game_Start)) AS gs,
        COALESCE(Home_Team_Norm, home_l) AS home_n,
        COALESCE(Away_Team_Norm, away_l) AS away_n,
        COALESCE(game_key_clean, feat_Game_Key, Game_Key) AS stable_key
      FROM {MOVES}
      WHERE UPPER(Sport) = @sport_u
        AND UPPER(Market) = 'SPREADS'
        AND (@book_u = '' OR UPPER(`{BOOK_COL_MOVES}`) = @book_u)   -- 🔒 book filter
        AND COALESCE(Game_Start, Commence_Hour, feat_Game_Start) IS NOT NULL
        AND TIMESTAMP(COALESCE(Game_Start, Commence_Hour, feat_Game_Start)) >= CURRENT_TIMESTAMP()
    ),
    canon AS (
      SELECT
        COALESCE(
          stable_key,
          CONCAT(
            IFNULL(LEAST(LOWER(home_n), LOWER(away_n)), 'tbd'), "_",
            IFNULL(GREATEST(LOWER(home_n), LOWER(away_n)), 'tbd'), "_",
            FORMAT_TIMESTAMP('%Y-%m-%d %H:%MZ', gs)
          )
        ) AS Game_Id,
        gs AS Game_Start,
        team
      FROM src,
      UNNEST([home_n, away_n]) AS team
      WHERE team IS NOT NULL
    ),
    grouped AS (
      SELECT
        Game_Id,
        ANY_VALUE(Game_Start) AS Game_Start,
        ARRAY_AGG(DISTINCT team ORDER BY team LIMIT 2) AS Teams
      FROM canon
      GROUP BY Game_Id
    )
    SELECT Game_Id, Game_Start, Teams
    FROM grouped
    WHERE ARRAY_LENGTH(Teams) = 2
    ORDER BY Game_Start
    LIMIT @hard_cap
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sport_u", "STRING", (sport or "").upper()),
                bigquery.ScalarQueryParameter("book_u", "STRING", (book or "").upper()),
                bigquery.ScalarQueryParameter("hard_cap", "INT64", int(hard_cap)),
            ],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()

@st.cache_data(ttl=120, show_spinner=False)
def team_context_from_moves(game_id: str, teams: list[str], book: str = DEFAULT_BOOK) -> dict:
    """
    Pick latest SPREADS snapshot row per team (by Market_Leader, Limit, Snapshot_Timestamp).
    Extract Is_Home + Value (team POV spread; fav < 0) to infer role/bucket.
    If Is_Home is NULL in source, derive it by matching team to home/away names.
    """
    if not teams:
        return {}
    teams_norm = [t.lower() for t in teams if t]

    sql = f"""
    WITH src AS (
      SELECT
        UPPER(Market) AS Market_U,
        TIMESTAMP(COALESCE(Game_Start, Commence_Hour, feat_Game_Start)) AS gs,
        COALESCE(Home_Team_Norm, home_l) AS home_n,
        COALESCE(Away_Team_Norm, away_l) AS away_n,
        COALESCE(feat_Team, Team_For_Join, Home_Team_Norm, home_l, Away_Team_Norm, away_l) AS team_any,
        COALESCE(game_key_clean, feat_Game_Key, Game_Key) AS stable_key,
        Is_Home,
        Value,
        Snapshot_Timestamp,
        Market_Leader,
        `Limit`,
        `{BOOK_COL_MOVES}` AS BookName
      FROM {MOVES}
      WHERE COALESCE(Game_Start, Commence_Hour, feat_Game_Start) IS NOT NULL
        AND UPPER(Market) = 'SPREADS'
        AND (@book_u = '' OR UPPER(`{BOOK_COL_MOVES}`) = @book_u)   -- 🔒 book filter
    ),
    canon AS (
      SELECT
        CONCAT(
          IFNULL(LEAST(LOWER(home_n), LOWER(away_n)), 'tbd'), "_",
          IFNULL(GREATEST(LOWER(home_n), LOWER(away_n)), 'tbd'), "_",
          FORMAT_TIMESTAMP('%Y-%m-%d %H:%MZ', gs)
        ) AS concat_id,
        LOWER(IFNULL(stable_key, ''))      AS stable_id,
        LOWER(team_any)                    AS team_norm,
        LOWER(home_n)                      AS home_norm,
        LOWER(away_n)                      AS away_norm,
        Is_Home,
        Value,
        gs AS cutoff,
        Market_Leader,
        `Limit`,
        Snapshot_Timestamp
      FROM src
      WHERE home_n IS NOT NULL AND away_n IS NOT NULL
    ),
    with_derived_home AS (
      SELECT
        team_norm,
        Value,
        cutoff,
        COALESCE(
          Is_Home,
          CASE
            WHEN team_norm = home_norm THEN TRUE
            WHEN team_norm = away_norm THEN FALSE
            ELSE NULL
          END
        ) AS is_home_final
      FROM canon
    ),
    picked AS (
      SELECT
        team_norm,
        ARRAY_AGG(
          STRUCT(is_home_final AS is_home, Value, cutoff)
          ORDER BY Value IS NULL, Market_Leader DESC, `Limit` DESC, Snapshot_Timestamp DESC
          LIMIT 1
        )[OFFSET(0)] AS s
      FROM (
        SELECT c.*, w.is_home_final
        FROM canon c
        JOIN with_derived_home w
        USING (team_norm, Value, cutoff)
      )
      WHERE (concat_id = LOWER(@gid) OR stable_id = LOWER(@gid))
        AND team_norm IN UNNEST(@teams_norm)
      GROUP BY team_norm
    )
    SELECT
      team_norm AS Team,
      s.is_home      AS is_home,
      s.Value        AS closing_spread,
      s.cutoff       AS cutoff
    FROM picked
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("gid", "STRING", game_id),
                bigquery.ArrayQueryParameter("teams_norm", "STRING", teams_norm),
                bigquery.ScalarQueryParameter("book_u", "STRING", (book or "").upper()),
            ],
            use_query_cache=True,
        ),
    )
    df = job.result().to_dataframe()

    out = {}
    for _, r in df.iterrows():
        sp  = r.get("closing_spread")
        spf = float(sp) if pd.notnull(sp) else None
        out[str(r["Team"])] = {
            "is_home": None if pd.isna(r.get("is_home")) else bool(r.get("is_home")),
            "is_favorite": (spf is not None and spf < 0),
            "closing_spread": spf,
            "cutoff": _to_utc_dt(r.get("cutoff")),
        }
    return out

def _enforce_role_coherence(ctxs: dict, teams: list[str]) -> dict:
    """Ensure exactly one favorite and one underdog iff any spread exists."""
    a, b = (teams + [None, None])[:2]
    if not a or not b:
        return ctxs
    sa = ctxs.get(a, {}).get("closing_spread")
    sb = ctxs.get(b, {}).get("closing_spread")
    if sa is None and sb is None:
        return ctxs
    if sa is not None and sb is not None:
        a_fav = sa < sb
        b_fav = sb < sa
    else:
        if sa is not None:
            a_fav = sa < 0; b_fav = not a_fav
        else:
            b_fav = sb < 0; a_fav = not b_fav
    for t, fav in [(a, a_fav), (b, b_fav)]:
        if t in ctxs:
            ctxs[t]["is_favorite"] = bool(fav)
    return ctxs

# ---------- SCORES: league-wide situation totals ----------

@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def league_totals_spreads(sport: str, cutoff_date: date, min_n: int = 0, book: str = DEFAULT_BOOK) -> pd.DataFrame:
    sql = f"""
    WITH enriched AS (
      SELECT
        Is_Home,
        Value AS Closing_Spread,             -- team POV spread; fav < 0
        Spread_Cover_Flag,
        ATS_Cover_Margin,
        CASE WHEN Value IS NULL THEN NULL
             WHEN Value < 0 THEN TRUE ELSE FALSE END AS Is_Favorite,
        CASE
          WHEN UPPER(@sport_upper) IN ('NBA','NCAAB','WNBA') THEN
            CASE
              WHEN Value IS NULL THEN ''
              WHEN Value <= -12.5 THEN 'Fav ≤ -12.5'
              WHEN Value <=  -9.5 THEN 'Fav -10 to -12.5'
              WHEN Value <=  -6.5 THEN 'Fav -7 to -9.5'
              WHEN Value <=  -3.5 THEN 'Fav -4 to -6.5'
              WHEN Value <=  -0.5 THEN 'Fav -0.5 to -3.5'
              WHEN Value =    0.0 THEN 'Pick (0)'
              WHEN Value <=   3.5 THEN 'Dog +0.5 to +3.5'
              WHEN Value <=   6.5 THEN 'Dog +4 to +6.5'
              WHEN Value <=   9.5 THEN 'Dog +7 to +9.5'
              WHEN Value <=  12.5 THEN 'Dog +10 to +12.5'
              ELSE 'Dog ≥ +13'
            END
          ELSE
            CASE
              WHEN Value IS NULL THEN ''
              WHEN Value <= -10.5 THEN 'Fav ≤ -10.5'
              WHEN Value <=  -7.5 THEN 'Fav -8 to -10.5'
              WHEN Value <=  -6.5 THEN 'Fav -7 to -6.5'
              WHEN Value <=  -3.5 THEN 'Fav -4 to -6.5'
              WHEN Value <=  -0.5 THEN 'Fav -0.5 to -3.5'
              WHEN Value =    0.0 THEN 'Pick (0)'
              WHEN Value <=   3.5 THEN 'Dog +0.5 to +3.5'
              WHEN Value <=   6.5 THEN 'Dog +4 to +6.5'
              WHEN Value <=  10.5 THEN 'Dog +7 to +10.5'
              ELSE 'Dog ≥ +11'
            END
        END AS Spread_Bucket
      FROM {SCORES}
      WHERE UPPER(Sport)  = @sport_upper
        AND UPPER(Market) = 'SPREADS'
        AND (@book_u = '' OR UPPER(`{BOOK_COL_SCORES}`) = @book_u)
        AND DATE(feat_Game_Start) <= @cutoff
    )

    -- 1) Venue
    SELECT 'Venue' AS GroupLabel,
           CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END AS Situation,
           COUNT(*) AS N,
           COUNTIF(Spread_Cover_Flag = 1) AS W,
           COUNTIF(Spread_Cover_Flag = 0) AS L,
           COUNTIF(ATS_Cover_Margin = 0)  AS P,
           SAFE_MULTIPLY(SAFE_DIVIDE(COUNTIF(Spread_Cover_Flag=1), NULLIF(COUNTIF(Spread_Cover_Flag IN (0,1)),0)),100.0) AS WinPct,
           CASE WHEN COUNTIF(Spread_Cover_Flag IN (0,1))>0
                THEN ((COUNTIF(Spread_Cover_Flag=1)*(100.0/110.0) + COUNTIF(Spread_Cover_Flag=0)*(-1.0))
                      / COUNTIF(Spread_Cover_Flag IN (0,1))) * 100.0
                ELSE NULL END AS ROI_Pct
    FROM enriched
    WHERE Is_Home IS NOT NULL
    GROUP BY Situation
    HAVING N >= @min_n

    UNION ALL

    -- 2) Role4
    SELECT 'Role4',
           CONCAT(CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END, ' ',
                  CASE WHEN Is_Favorite THEN 'Favorite' ELSE 'Underdog' END),
           COUNT(*),
           COUNTIF(Spread_Cover_Flag = 1),
           COUNTIF(Spread_Cover_Flag = 0),
           COUNTIF(ATS_Cover_Margin = 0),
           SAFE_MULTIPLY(SAFE_DIVIDE(COUNTIF(Spread_Cover_Flag=1), NULLIF(COUNTIF(Spread_Cover_Flag IN (0,1)),0)),100.0),
           CASE WHEN COUNTIF(Spread_Cover_Flag IN (0,1))>0
                THEN ((COUNTIF(Spread_Cover_Flag=1)*(100.0/110.0) + COUNTIF(Spread_Cover_Flag=0)*(-1.0))
                      / COUNTIF(Spread_Cover_Flag IN (0,1))) * 100.0
                ELSE NULL END
    FROM enriched
    WHERE Is_Home IS NOT NULL AND Is_Favorite IS NOT NULL
    GROUP BY 2
    HAVING COUNT(*) >= @min_n

    UNION ALL

    -- 3) Bucket (overall)
    SELECT 'Bucket',
           Spread_Bucket,
           COUNT(*),
           COUNTIF(Spread_Cover_Flag = 1),
           COUNTIF(Spread_Cover_Flag = 0),
           COUNTIF(ATS_Cover_Margin = 0),
           SAFE_MULTIPLY(SAFE_DIVIDE(COUNTIF(Spread_Cover_Flag=1), NULLIF(COUNTIF(Spread_Cover_Flag IN (0,1)),0)),100.0),
           CASE WHEN COUNTIF(Spread_Cover_Flag IN (0,1))>0
                THEN ((COUNTIF(Spread_Cover_Flag=1)*(100.0/110.0) + COUNTIF(Spread_Cover_Flag=0)*(-1.0))
                      / COUNTIF(Spread_Cover_Flag IN (0,1))) * 100.0
                ELSE NULL END
    FROM enriched
    WHERE Spread_Bucket <> ''
    GROUP BY Spread_Bucket
    HAVING COUNT(*) >= @min_n

    UNION ALL

    -- 4) Bucket × Venue
    SELECT 'Bucket·Venue',
           CONCAT(CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END, ' · ', Spread_Bucket),
           COUNT(*),
           COUNTIF(Spread_Cover_Flag = 1),
           COUNTIF(Spread_Cover_Flag = 0),
           COUNTIF(ATS_Cover_Margin = 0),
           SAFE_MULTIPLY(SAFE_DIVIDE(COUNTIF(Spread_Cover_Flag=1), NULLIF(COUNTIF(Spread_Cover_Flag IN (0,1)),0)),100.0),
           CASE WHEN COUNTIF(Spread_Cover_Flag IN (0,1))>0
                THEN ((COUNTIF(Spread_Cover_Flag=1)*(100.0/110.0) + COUNTIF(Spread_Cover_Flag=0)*(-1.0))
                      / COUNTIF(Spread_Cover_Flag IN (0,1))) * 100.0
                ELSE NULL END
    FROM enriched
    WHERE Spread_Bucket <> '' AND Is_Home IS NOT NULL
    GROUP BY 2
    HAVING COUNT(*) >= @min_n
    ORDER BY 1, 2
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sport_upper","STRING",(sport or "").upper()),
                bigquery.ScalarQueryParameter("book_u","STRING",(book or "").upper()),
                bigquery.ScalarQueryParameter("cutoff","DATE", cutoff_date),
                bigquery.ScalarQueryParameter("min_n","INT64", int(min_n)),
            ],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def league_totals_overunder(sport: str, cutoff_date: date, min_n: int = 0, book: str = DEFAULT_BOOK) -> pd.DataFrame:
    sql = f"""
    -- One home row per game for totals using ROW_NUMBER (avoids extra CTEs)
    WITH per_game_totals AS (
      SELECT *
      FROM (
        SELECT
          COALESCE(feat_Game_Key, Game_Key) AS GKey,
          feat_Game_Start,
          Value AS Total_Line,
          SAFE_CAST(Team_Score AS FLOAT64) + SAFE_CAST(Opp_Score AS FLOAT64) AS Total_Points,
          Is_Home,
          ROW_NUMBER() OVER (PARTITION BY COALESCE(feat_Game_Key, Game_Key)
                             ORDER BY Is_Home DESC) AS rn
        FROM {SCORES}
        WHERE UPPER(Sport)  = @sport_upper
          AND UPPER(Market) = 'TOTALS'
          AND (@book_u = '' OR UPPER(`{BOOK_COL_SCORES}`) = @book_u)
          AND DATE(feat_Game_Start) <= @cutoff
          AND Value IS NOT NULL
          AND Team_Score IS NOT NULL AND Opp_Score IS NOT NULL
      )
      WHERE rn = 1   -- pick the home row
    ),
    home_spreads AS (
      SELECT
        COALESCE(feat_Game_Key, Game_Key) AS GKey,
        ANY_VALUE(Value) AS Home_Closing_Spread
      FROM {SCORES}
      WHERE UPPER(Sport)  = @sport_upper
        AND UPPER(Market) = 'SPREADS'
        AND (@book_u = '' OR UPPER(`{BOOK_COL_SCORES}`) = @book_u)
        AND DATE(feat_Game_Start) <= @cutoff
        AND Is_Home = TRUE
      GROUP BY GKey
    ),
    enriched AS (
      SELECT
        t.GKey,
        t.Total_Line,
        t.Total_Points,
        (t.Total_Points > t.Total_Line) AS Over_Win,
        (t.Total_Points < t.Total_Line) AS Under_Win,
        (t.Total_Points = t.Total_Line) AS Push,
        CASE
          WHEN UPPER(@sport_upper) IN ('NBA','NCAAB','WNBA') THEN
            CASE
              WHEN t.Total_Line <= 205.5 THEN 'OU ≤ 205.5'
              WHEN t.Total_Line <= 209.5 THEN 'OU 206–209.5'
              WHEN t.Total_Line <= 213.5 THEN 'OU 210–213.5'
              WHEN t.Total_Line <= 217.5 THEN 'OU 214–217.5'
              WHEN t.Total_Line <= 221.5 THEN 'OU 218–221.5'
              WHEN t.Total_Line <= 225.5 THEN 'OU 222–225.5'
              WHEN t.Total_Line <= 229.5 THEN 'OU 226–229.5'
              WHEN t.Total_Line <= 233.5 THEN 'OU 230–233.5'
              ELSE 'OU ≥ 234'
            END
          ELSE
            CASE
              WHEN t.Total_Line <= 37.5 THEN 'OU ≤ 37.5'
              WHEN t.Total_Line <= 41.5 THEN 'OU 38–41.5'
              WHEN t.Total_Line <= 44.5 THEN 'OU 42–44.5'
              WHEN t.Total_Line <= 47.5 THEN 'OU 45–47.5'
              WHEN t.Total_Line <= 50.5 THEN 'OU 48–50.5'
              WHEN t.Total_Line <= 53.5 THEN 'OU 51–53.5'
              WHEN t.Total_Line <= 56.5 THEN 'OU 54–56.5'
              WHEN t.Total_Line <= 59.5 THEN 'OU 57–59.5'
              ELSE 'OU ≥ 60'
            END
        END AS Total_Bucket,
        -- Role labels from the home spread (pick -> treat as dog)
        CASE
          WHEN s.Home_Closing_Spread IS NULL THEN 'Home Underdog'
          WHEN s.Home_Closing_Spread < 0 THEN 'Home Favorite'
          WHEN s.Home_Closing_Spread = 0 THEN 'Home Underdog'
          ELSE 'Home Underdog'
        END AS Role4_HomeSide,
        CASE
          WHEN s.Home_Closing_Spread IS NULL THEN 'Road Favorite'
          WHEN s.Home_Closing_Spread < 0 THEN 'Road Underdog'
          WHEN s.Home_Closing_Spread = 0 THEN 'Road Favorite'
          ELSE 'Road Favorite'
        END AS Role4_RoadSide
      FROM per_game_totals t
      LEFT JOIN home_spreads s USING (GKey)
    )

    -- VENUE
    SELECT Side, 'Venue' AS GroupLabel, Situation, N, W, L, P,
           SAFE_MULTIPLY(SAFE_DIVIDE(W, NULLIF(W+L,0)),100.0) AS WinPct,
           CASE WHEN (W+L)>0 THEN ((W*(100.0/110.0)+L*(-1.0))/(W+L))*100.0 ELSE NULL END AS ROI_Pct
    FROM (
      SELECT 'Over' AS Side, 'Home' AS Situation,
             COUNT(*) AS N, COUNTIF(Over_Win) AS W, COUNTIF(Under_Win) AS L, COUNTIF(Push) AS P FROM enriched
      UNION ALL
      SELECT 'Under', 'Home', COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push) FROM enriched
      UNION ALL
      SELECT 'Over', 'Road', COUNT(*), COUNTIF(Over_Win), COUNTIF(Under_Win), COUNTIF(Push) FROM enriched
      UNION ALL
      SELECT 'Under','Road', COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push) FROM enriched
    )
    WHERE N >= @min_n

    UNION ALL

    -- ROLE4 (home+road sides)
    SELECT Side, 'Role4', Situation, N, W, L, P,
           SAFE_MULTIPLY(SAFE_DIVIDE(W, NULLIF(W+L,0)),100.0),
           CASE WHEN (W+L)>0 THEN ((W*(100.0/110.0)+L*(-1.0))/(W+L))*100.0 ELSE NULL END
    FROM (
      SELECT 'Over' AS Side, Role4_HomeSide AS Situation,
             COUNT(*) AS N, COUNTIF(Over_Win) AS W, COUNTIF(Under_Win) AS L, COUNTIF(Push) AS P
      FROM enriched GROUP BY Role4_HomeSide
      UNION ALL
      SELECT 'Under', Role4_HomeSide, COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push)
      FROM enriched GROUP BY Role4_HomeSide
      UNION ALL
      SELECT 'Over', Role4_RoadSide, COUNT(*), COUNTIF(Over_Win), COUNTIF(Under_Win), COUNTIF(Push)
      FROM enriched GROUP BY Role4_RoadSide
      UNION ALL
      SELECT 'Under', Role4_RoadSide, COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push)
      FROM enriched GROUP BY Role4_RoadSide
    )
    WHERE N >= @min_n

    UNION ALL

    -- BUCKET (overall)
    SELECT Side, 'Bucket', Situation, N, W, L, P,
           SAFE_MULTIPLY(SAFE_DIVIDE(W, NULLIF(W+L,0)),100.0),
           CASE WHEN (W+L)>0 THEN ((W*(100.0/110.0)+L*(-1.0))/(W+L))*100.0 ELSE NULL END
    FROM (
      SELECT 'Over' AS Side, Total_Bucket AS Situation,
             COUNT(*) AS N, COUNTIF(Over_Win) AS W, COUNTIF(Under_Win) AS L, COUNTIF(Push) AS P
      FROM enriched GROUP BY Total_Bucket
      UNION ALL
      SELECT 'Under', Total_Bucket, COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push)
      FROM enriched GROUP BY Total_Bucket
    )
    WHERE N >= @min_n

    UNION ALL

    -- BUCKET × VENUE
    SELECT Side, 'Bucket·Venue', Situation, N, W, L, P,
           SAFE_MULTIPLY(SAFE_DIVIDE(W, NULLIF(W+L,0)),100.0),
           CASE WHEN (W+L)>0 THEN ((W*(100.0/110.0)+L*(-1.0))/(W+L))*100.0 ELSE NULL END
    FROM (
      SELECT 'Over' AS Side, CONCAT('Home · ', Total_Bucket) AS Situation,
             COUNT(*) AS N, COUNTIF(Over_Win) AS W, COUNTIF(Under_Win) AS L, COUNTIF(Push) AS P
      FROM enriched GROUP BY Total_Bucket
      UNION ALL
      SELECT 'Under', CONCAT('Home · ', Total_Bucket),
             COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push)
      FROM enriched GROUP BY Total_Bucket
      UNION ALL
      SELECT 'Over', CONCAT('Road · ', Total_Bucket),
             COUNT(*), COUNTIF(Over_Win), COUNTIF(Under_Win), COUNTIF(Push)
      FROM enriched GROUP BY Total_Bucket
      UNION ALL
      SELECT 'Under', CONCAT('Road · ', Total_Bucket),
             COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push)
      FROM enriched GROUP BY Total_Bucket
    )
    WHERE N >= @min_n
    ORDER BY GroupLabel, Situation, Side
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sport_upper","STRING",(sport or "").upper()),
                bigquery.ScalarQueryParameter("book_u","STRING",(book or "").upper()),
                bigquery.ScalarQueryParameter("cutoff","DATE", cutoff_date),
                bigquery.ScalarQueryParameter("min_n","INT64", int(min_n)),
            ],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()

# ---------- helpers for rendering ----------
def _labels_for_team(sport: str, ctx: dict) -> list[str]:
    is_home = ctx.get("is_home")
    is_favorite = ctx.get("is_favorite")
    spread = ctx.get("closing_spread")
    bucket = _derive_spread_bucket_for_ui(sport, spread)
    return _wanted_labels(is_home, is_favorite, bucket)

def _filter_rows(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    if df is None or df.empty or not labels:
        return pd.DataFrame(columns=["GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"])
    return df[df["Situation"].isin(labels)].copy()

# ---------- main UI ----------
def render_current_situations_tab(selected_sport: str):
    st.subheader("📚 Situation DB — League Totals (by Sport)")

    if not selected_sport:
        st.warning("Pick a sport.")
        st.stop()

    # choose book – keep it simple (hidden control or expose as input if you like)
    book_name = DEFAULT_BOOK

    games = list_current_games_from_moves(selected_sport, book=book_name)
    if games.empty:
        st.info("No upcoming games found for this sport (from MOVES; spreads only).")
        return

    games = games.copy()
    games["Teams"] = games["Teams"].apply(lambda xs: list(xs)[:2])
    games["label"] = games.apply(
        lambda r: f"{r['Game_Start']} — {', '.join(r['Teams'])}" if r["Teams"] else f"{r['Game_Start']} — (teams TBD)",
        axis=1
    )

    row = st.selectbox("Select game", games.to_dict("records"), format_func=lambda r: r["label"])
    if not row:
        st.stop()

    game_id = row["Game_Id"]
    game_start = _to_utc_dt(row["Game_Start"])
    teams = row["Teams"]

    min_n = st.slider("Min graded games per situation", 10, 200, 30, step=5)
    cutoff_date_for_stats: date = st.date_input("Cutoff for historical stats (DATE)", value=date.today())

    # League-wide tables (one query each) — filtered to book
    league_spreads = league_totals_spreads(selected_sport, cutoff_date_for_stats, min_n, book=book_name)
    league_totals  = league_totals_overunder(selected_sport, cutoff_date_for_stats, min_n, book=book_name)

    # Roles from MOVES (SPREADS-ONLY) — filtered to book
    ctxs = team_context_from_moves(game_id, teams, book=book_name)
    ctxs = _enforce_role_coherence(ctxs, teams)

    # Per-team panels
    cols = st.columns(2)
    for i, team in enumerate(teams):
        with cols[i]:
            st.markdown(f"### {team}")
            ctx = ctxs.get(team, {})

            is_home = ctx.get("is_home")
            is_favorite = ctx.get("is_favorite")
            spread = ctx.get("closing_spread")
            bucket = _derive_spread_bucket_for_ui(selected_sport, spread)

            bits = []
            bits.append("🏠 Home" if is_home is True else ("🚗 Road" if is_home is False else ""))
            bits.append("⭐ Favorite" if is_favorite is True else ("🐶 Underdog" if is_favorite is False else ""))
            if bucket: bits.append(bucket)
            st.caption(" / ".join([b for b in bits if b]) or "Role: Unknown")

            st.markdown("**ATS (Spreads Only) — Fav/Dog + Venue + Bucket + Bucket × Venue**")
            table4 = _compose_four_views_spreads(
                league_spreads=league_spreads,
                is_home=is_home,
                is_favorite=is_favorite,
                bucket=bucket
            )
            if table4.empty:
                st.write("_No league rows meet N threshold for this role/bucket._")
            else:
                st.dataframe(table4, use_container_width=True)

    # Spreads: full league table (helpful for verifying rows)
    with st.expander("🔎 League — Full table for this sport (Spreads Only)"):
        if league_spreads.empty:
            st.write("_No rows for this sport/cutoff._")
        else:
            show = league_spreads.copy()
            show = show[show["GroupLabel"].isin(["Role4","Bucket","Bucket·Venue","Venue"])]
            for c in ["WinPct","ROI_Pct"]:
                show[c] = show[c].map(lambda x: None if pd.isna(x) else round(float(x), 1))
            st.dataframe(show, use_container_width=True)

    # OVER/UNDER league table (totals-only; side split)
    st.markdown("### 📈 League Totals — Over/Under (Totals Only)")
    if league_totals.empty:
        st.write("_No totals rows meet N threshold for this sport/cutoff._")
    else:
        show = league_totals.copy()
        for c in ["WinPct","ROI_Pct"]:
            if c in show.columns:
                show[c] = show[c].map(lambda x: None if pd.isna(x) else round(float(x), 1))
        st.dataframe(
            show[["Side","GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"]],
            use_container_width=True
        )

# ---------- optional quick debug ----------
def render_current_games_section(selected_sport: str):
    st.subheader("📡 Current/Upcoming Games (from MOVES — spreads only)")
    games = list_current_games_from_moves(selected_sport, book=DEFAULT_BOOK)
    if games.empty:
        st.info("No upcoming games found for this sport (from MOVES).")
        with st.expander("Debug this sport in MOVES"):
            dbg_sql = f"""
            SELECT
              UPPER(Sport) AS Sport_Upper,
              UPPER(Market) AS Market_U,
              COALESCE(game_key_clean, feat_Game_Key, Game_Key) AS Game_Id,
              COALESCE(Game_Start, Commence_Hour, feat_Game_Start) AS Game_Start,
              Home_Team_Norm, Away_Team_Norm, Team_For_Join, feat_Team, home_l, away_l,
              Value, Is_Home, Snapshot_Timestamp, `{BOOK_COL_MOVES}` AS Book
            FROM {MOVES}
            WHERE UPPER(Sport) = @sport_upper AND UPPER(Market)='SPREADS'
            ORDER BY Game_Start DESC
            LIMIT 100
            """
            job = CLIENT.query(
                dbg_sql,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("sport_upper", "STRING", (selected_sport or "").upper())],
                    use_query_cache=True,
                ),
            )
            st.dataframe(job.result().to_dataframe())
        return

    df = games.copy()
    df["label"] = df.apply(lambda r: f"{r['Game_Start']} — {', '.join(r['Teams'])}", axis=1)
    row = st.selectbox("Select game", df.to_dict("records"), format_func=lambda r: r["label"])
    if not row:
        st.stop()
    st.write(f"**Game Id:** {row['Game_Id']}")
    st.write(f"**Teams:** {', '.join(row['Teams'])}")
    st.write(f"**Start:** {row['Game_Start']}")
