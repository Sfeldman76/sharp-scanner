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

# backticked column-name tokens (safe to inject into SQL)
BOOK_MOVES_COL_SQL  = f"`{BOOK_COL_MOVES}`"
BOOK_SCORES_COL_SQL = f"`{BOOK_COL_SCORES}`"

CLIENT = bigquery.Client(project=PROJECT)
CACHE_TTL_SEC = 120

# ---------- tiny utils ----------
def _to_utc_dt(x) -> dt.datetime:
    if isinstance(x, dt.datetime):
        return x.astimezone(timezone.utc) if x.tzinfo else x.replace(tzinfo=timezone.utc)
    if isinstance(x, dt.date):
        return dt.datetime.combine(x, time(0, 0, 0, tzinfo=timezone.utc))
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
        return ts.to_pydatetime()
    return datetime.now(timezone.utc)

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

def _derive_spread_bucket_for_ui(sport: str, closing_spread: float | None) -> str:
    if closing_spread is None:
        return ""
    return _bb_spread_bucket(closing_spread) if _sport_is_basketball(sport) else _fb_spread_bucket(closing_spread)

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
    seen, keep = set(), []
    for s in out:
        if s and s not in seen:
            keep.append(s); seen.add(s)
    return keep

def _compose_four_views_spreads(league_spreads: pd.DataFrame,
                                is_home: bool | None, is_favorite: bool | None, bucket: str | None) -> pd.DataFrame:
    if league_spreads is None or league_spreads.empty:
        return pd.DataFrame(columns=["Section","GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"])
    parts = []
    if is_home is not None and is_favorite is not None:
        role = f"{'Home' if is_home else 'Road'} {'Favorite' if is_favorite else 'Underdog'}"
        df = league_spreads[(league_spreads["GroupLabel"]=="Role4") & (league_spreads["Situation"]==role)]
        if not df.empty: df = df.copy(); df.insert(0,"Section","Role (Fav/Dog)"); parts.append(df)
    if is_home is not None:
        v = "Home" if is_home else "Road"
        df = league_spreads[(league_spreads["GroupLabel"]=="Venue") & (league_spreads["Situation"]==v)]
        if not df.empty: df = df.copy(); df.insert(0,"Section","Venue (Home/Road)"); parts.append(df)
    if bucket:
        df = league_spreads[(league_spreads["GroupLabel"]=="Bucket") & (league_spreads["Situation"]==bucket)]
        if not df.empty: df = df.copy(); df.insert(0,"Section","Bucket (overall)"); parts.append(df)
        if is_home is not None:
            bvl = f"{'Home' if is_home else 'Road'} · {bucket}"
            df = league_spreads[(league_spreads["GroupLabel"]=="Bucket·Venue") & (league_spreads["Situation"]==bvl)]
            if not df.empty: df = df.copy(); df.insert(0,"Section","Bucket × Venue"); parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["Section","GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"])
    out = pd.concat(parts, ignore_index=True)
    # assure numeric before rounding
    for c in ("N","W","L","P","WinPct","ROI_Pct"):
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("WinPct","ROI_Pct"):
        if c in out.columns: out[c] = out[c].round(1)
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
        AND (@book_u = '' OR UPPER({BOOK_MOVES_COL_SQL}) = @book_u)
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
      FROM src, UNNEST([home_n, away_n]) AS team
      WHERE team IS NOT NULL
    ),
    grouped AS (
      SELECT Game_Id, ANY_VALUE(Game_Start) AS Game_Start,
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
        `Limit`
      FROM {MOVES}
      WHERE COALESCE(Game_Start, Commence_Hour, feat_Game_Start) IS NOT NULL
        AND UPPER(Market) = 'SPREADS'
        AND (@book_u = '' OR UPPER({BOOK_MOVES_COL_SQL}) = @book_u)
    ),
    canon AS (
      SELECT
        CONCAT(
          IFNULL(LEAST(LOWER(home_n), LOWER(away_n)), 'tbd'), "_",
          IFNULL(GREATEST(LOWER(home_n), LOWER(away_n)), 'tbd'), "_",
          FORMAT_TIMESTAMP('%Y-%m-%d %H:%MZ', gs)
        ) AS concat_id,
        LOWER(IFNULL(stable_key, '')) AS stable_id,
        LOWER(team_any)               AS team_norm,
        LOWER(home_n)                 AS home_norm,
        LOWER(away_n)                 AS away_norm,
        Is_Home, Value, gs AS cutoff,
        Market_Leader, `Limit`, Snapshot_Timestamp
      FROM src
      WHERE home_n IS NOT NULL AND away_n IS NOT NULL
    ),
    with_home AS (
      SELECT
        team_norm, Value, cutoff,
        COALESCE(Is_Home,
          CASE WHEN team_norm = home_norm THEN TRUE
               WHEN team_norm = away_norm THEN FALSE
               ELSE NULL END) AS is_home_final
      FROM canon
    ),
    picked AS (
      SELECT
        team_norm,
        ARRAY_AGG(STRUCT(is_home_final AS is_home, Value, cutoff)
                  ORDER BY Value IS NULL, Market_Leader DESC, `Limit` DESC, Snapshot_Timestamp DESC
                  LIMIT 1)[OFFSET(0)] AS s
      FROM (SELECT c.*, w.is_home_final FROM canon c JOIN with_home w USING (team_norm, Value, cutoff))
      WHERE (concat_id = LOWER(@gid) OR stable_id = LOWER(@gid))
        AND team_norm IN UNNEST(@teams_norm)
      GROUP BY team_norm
    )
    SELECT team_norm AS Team, s.is_home AS is_home, s.Value AS closing_spread, s.cutoff AS cutoff
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
    a, b = (teams + [None, None])[:2]
    if not a or not b: return ctxs
    sa = ctxs.get(a, {}).get("closing_spread")
    sb = ctxs.get(b, {}).get("closing_spread")
    if sa is None and sb is None: return ctxs
    if sa is not None and sb is not None:
        a_fav = sa < sb; b_fav = sb < sa
    else:
        if sa is not None: a_fav = sa < 0; b_fav = not a_fav
        else:              b_fav = sb < 0; a_fav = not b_fav
    for t, fav in [(a, a_fav), (b, b_fav)]:
        if t in ctxs: ctxs[t]["is_favorite"] = bool(fav)
    return ctxs

# ---------- SCORES: league-wide situation totals ----------
@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def league_totals_spreads(sport: str, cutoff_date: date, min_n: int = 0, years_back: int = 10, book: str = DEFAULT_BOOK) -> pd.DataFrame:
    # sport-aware buckets (as SQL fragment)
    bucket_case = """
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
      END
    """
    base_filter = f"""
      FROM {SCORES}
      WHERE UPPER(Sport)  = @sport_upper
        AND UPPER(Market) = 'SPREADS'
        AND (@book_u = '' OR UPPER({BOOK_SCORES_COL_SQL}) = @book_u)
        AND DATE(feat_Game_Start) <= @cutoff
        AND feat_Game_Start >= TIMESTAMP(DATETIME_SUB(DATETIME(@cutoff), INTERVAL @years_back YEAR))
    """
    def _run(q: str):
        return CLIENT.query(
            q,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("sport_upper","STRING",(sport or "").upper()),
                    bigquery.ScalarQueryParameter("book_u","STRING",(book or "").upper()),
                    bigquery.ScalarQueryParameter("cutoff","DATE", cutoff_date),
                    bigquery.ScalarQueryParameter("years_back","INT64", int(years_back)),
                    bigquery.ScalarQueryParameter("min_n","INT64", int(min_n)),
                ],
                use_query_cache=True,
            ),
        ).result().to_dataframe()

    q_venue = f"""
    SELECT 'Venue' AS GroupLabel,
           CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END AS Situation,
           COUNT(*) AS N,
           COUNTIF(Spread_Cover_Flag=1) AS W,
           COUNTIF(Spread_Cover_Flag=0) AS L,
           COUNTIF(ATS_Cover_Margin=0)  AS P
    {base_filter}
    GROUP BY Situation
    HAVING N >= @min_n
    """
    df1 = _run(q_venue)

    q_role4 = f"""
    SELECT 'Role4' AS GroupLabel,
           CONCAT(CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END, ' ',
                  CASE WHEN Value < 0 THEN 'Favorite' ELSE 'Underdog' END) AS Situation,
           COUNT(*) AS N,
           COUNTIF(Spread_Cover_Flag=1) AS W,
           COUNTIF(Spread_Cover_Flag=0) AS L,
           COUNTIF(ATS_Cover_Margin=0)  AS P
    {base_filter}
      AND Value IS NOT NULL
      AND Is_Home IS NOT NULL
    GROUP BY Situation
    HAVING N >= @min_n
    """
    df2 = _run(q_role4)

    q_bucket = f"""
    SELECT 'Bucket' AS GroupLabel,
           ({bucket_case}) AS Situation,
           COUNT(*) AS N,
           COUNTIF(Spread_Cover_Flag=1) AS W,
           COUNTIF(Spread_Cover_Flag=0) AS L,
           COUNTIF(ATS_Cover_Margin=0)  AS P
    {base_filter}
      AND Value IS NOT NULL
    GROUP BY Situation
    HAVING Situation <> '' AND N >= @min_n
    """
    df3 = _run(q_bucket)

    q_bv = f"""
    SELECT 'Bucket·Venue' AS GroupLabel,
           CONCAT(CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END, ' · ', ({bucket_case})) AS Situation,
           COUNT(*) AS N,
           COUNTIF(Spread_Cover_Flag=1) AS W,
           COUNTIF(Spread_Cover_Flag=0) AS L,
           COUNTIF(ATS_Cover_Margin=0)  AS P
    {base_filter}
      AND Value IS NOT NULL
      AND Is_Home IS NOT NULL
    GROUP BY Situation
    HAVING SPLIT(Situation,' · ')[OFFSET(1)] <> '' AND N >= @min_n
    """
    df4 = _run(q_bv)

    out = pd.concat([df1, df2, df3, df4], ignore_index=True)
    if out.empty:
        return out
    for c in ("N","W","L","P"): out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
    den = (out["W"] + out["L"]).astype("float64")
    out["WinPct"] = np.where(den>0, (out["W"].astype("float64")/den)*100.0, np.nan)
    out["ROI_Pct"] = np.where(den>0,
                              ((out["W"].astype("float64")*(100.0/110.0) + out["L"].astype("float64")*(-1.0))/den)*100.0,
                              np.nan)
    out["WinPct"] = pd.to_numeric(out["WinPct"], errors="coerce").round(1)
    out["ROI_Pct"] = pd.to_numeric(out["ROI_Pct"], errors="coerce").round(1)
    return out.sort_values(["GroupLabel","Situation"]).reset_index(drop=True)

@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def league_totals_overunder(sport: str, cutoff_date: date, min_n: int = 0, years_back: int = 10, book: str = DEFAULT_BOOK) -> pd.DataFrame:
    # One CTE block + parameterized filters (no braces in final SQL)
    totals_cte = f"""
      WITH per_game AS (
        SELECT *
        FROM (
          SELECT
            COALESCE(feat_Game_Key, Game_Key) AS GKey,
            feat_Game_Start,
            Value AS Total_Line,
            SAFE_CAST(Team_Score AS FLOAT64) + SAFE_CAST(Opp_Score AS FLOAT64) AS Total_Points,
            Is_Home,
            ROW_NUMBER() OVER (PARTITION BY COALESCE(feat_Game_Key, Game_Key) ORDER BY Is_Home DESC) AS rn
          FROM {SCORES}
          WHERE UPPER(Sport) = @sport_upper
            AND UPPER(Market) = 'TOTALS'
            AND (@book_u = '' OR UPPER({BOOK_SCORES_COL_SQL}) = @book_u)
            AND DATE(feat_Game_Start) <= @cutoff
            AND feat_Game_Start >= TIMESTAMP(DATETIME_SUB(DATETIME(@cutoff), INTERVAL @years_back YEAR))
            AND Value IS NOT NULL
            AND Team_Score IS NOT NULL AND Opp_Score IS NOT NULL
        )
        WHERE rn = 1
      ),
      spreads AS (
        SELECT COALESCE(feat_Game_Key, Game_Key) AS GKey,
               ANY_VALUE(Value) AS Home_Closing_Spread
        FROM {SCORES}
        WHERE UPPER(Sport) = @sport_upper
          AND UPPER(Market) = 'SPREADS'
          AND (@book_u = '' OR UPPER({BOOK_SCORES_COL_SQL}) = @book_u)
          AND DATE(feat_Game_Start) <= @cutoff
          AND feat_Game_Start >= TIMESTAMP(DATETIME_SUB(DATETIME(@cutoff), INTERVAL @years_back YEAR))
          AND Is_Home = TRUE
        GROUP BY GKey
      ),
      e AS (
        SELECT
          p.GKey,
          p.Total_Line,
          p.Total_Points,
          (p.Total_Points > p.Total_Line) AS Over_Win,
          (p.Total_Points < p.Total_Line) AS Under_Win,
          (p.Total_Points = p.Total_Line) AS Push,
          CASE
            WHEN UPPER(@sport_upper) IN ('NBA','NCAAB','WNBA') THEN
              CASE
                WHEN p.Total_Line <= 205.5 THEN 'OU ≤ 205.5'
                WHEN p.Total_Line <= 209.5 THEN 'OU 206–209.5'
                WHEN p.Total_Line <= 213.5 THEN 'OU 210–213.5'
                WHEN p.Total_Line <= 217.5 THEN 'OU 214–217.5'
                WHEN p.Total_Line <= 221.5 THEN 'OU 218–221.5'
                WHEN p.Total_Line <= 225.5 THEN 'OU 222–225.5'
                WHEN p.Total_Line <= 229.5 THEN 'OU 226–229.5'
                WHEN p.Total_Line <= 233.5 THEN 'OU 230–233.5'
                ELSE 'OU ≥ 234'
              END
            ELSE
              CASE
                WHEN p.Total_Line <= 37.5 THEN 'OU ≤ 37.5'
                WHEN p.Total_Line <= 41.5 THEN 'OU 38–41.5'
                WHEN p.Total_Line <= 44.5 THEN 'OU 42–44.5'
                WHEN p.Total_Line <= 47.5 THEN 'OU 45–47.5'
                WHEN p.Total_Line <= 50.5 THEN 'OU 48–50.5'
                WHEN p.Total_Line <= 53.5 THEN 'OU 51–53.5'
                WHEN p.Total_Line <= 56.5 THEN 'OU 54–56.5'
                WHEN p.Total_Line <= 59.5 THEN 'OU 57–59.5'
                ELSE 'OU ≥ 60'
              END
          END AS Total_Bucket,
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
        FROM per_game p
        LEFT JOIN spreads s USING (GKey)
      )
    """
    def _run(q: str):
        return CLIENT.query(
            totals_cte + q,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("sport_upper","STRING",(sport or "").upper()),
                    bigquery.ScalarQueryParameter("book_u","STRING",(book or "").upper()),
                    bigquery.ScalarQueryParameter("cutoff","DATE", cutoff_date),
                    bigquery.ScalarQueryParameter("years_back","INT64", int(years_back)),
                    bigquery.ScalarQueryParameter("min_n","INT64", int(min_n)),
                ],
                use_query_cache=True,
            ),
        ).result().to_dataframe()

    q_venue = """
    SELECT 'Over' AS Side, 'Venue' AS GroupLabel, 'Home' AS Situation,
           COUNT(*) AS N, COUNTIF(Over_Win) AS W, COUNTIF(Under_Win) AS L, COUNTIF(Push) AS P FROM e
    UNION ALL
    SELECT 'Under','Venue','Home', COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push) FROM e
    UNION ALL
    SELECT 'Over','Venue','Road', COUNT(*), COUNTIF(Over_Win), COUNTIF(Under_Win), COUNTIF(Push) FROM e
    UNION ALL
    SELECT 'Under','Venue','Road', COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push) FROM e
    """
    df1 = _run(q_venue)

    q_role4 = """
    SELECT 'Over' AS Side, 'Role4' AS GroupLabel, Role4_HomeSide AS Situation,
           COUNT(*) AS N, COUNTIF(Over_Win) AS W, COUNTIF(Under_Win) AS L, COUNTIF(Push) AS P
    FROM e GROUP BY Role4_HomeSide
    UNION ALL
    SELECT 'Under','Role4', Role4_HomeSide, COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push)
    FROM e GROUP BY Role4_HomeSide
    UNION ALL
    SELECT 'Over','Role4', Role4_RoadSide, COUNT(*), COUNTIF(Over_Win), COUNTIF(Under_Win), COUNTIF(Push)
    FROM e GROUP BY Role4_RoadSide
    UNION ALL
    SELECT 'Under','Role4', Role4_RoadSide, COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push)
    FROM e GROUP BY Role4_RoadSide
    """
    df2 = _run(q_role4)

    q_bucket = """
    SELECT 'Over' AS Side, 'Bucket' AS GroupLabel, Total_Bucket AS Situation,
           COUNT(*) AS N, COUNTIF(Over_Win) AS W, COUNTIF(Under_Win) AS L, COUNTIF(Push) AS P
    FROM e GROUP BY Total_Bucket
    UNION ALL
    SELECT 'Under','Bucket', Total_Bucket, COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push)
    FROM e GROUP BY Total_Bucket
    """
    df3 = _run(q_bucket)

    q_bv = """
    SELECT 'Over' AS Side, 'Bucket·Venue' AS GroupLabel, CONCAT('Home · ', Total_Bucket) AS Situation,
           COUNT(*) AS N, COUNTIF(Over_Win) AS W, COUNTIF(Under_Win) AS L, COUNTIF(Push) AS P
    FROM e GROUP BY Total_Bucket
    UNION ALL
    SELECT 'Under','Bucket·Venue', CONCAT('Home · ', Total_Bucket), COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push)
    FROM e GROUP BY Total_Bucket
    UNION ALL
    SELECT 'Over','Bucket·Venue', CONCAT('Road · ', Total_Bucket), COUNT(*), COUNTIF(Over_Win), COUNTIF(Under_Win), COUNTIF(Push)
    FROM e GROUP BY Total_Bucket
    UNION ALL
    SELECT 'Under','Bucket·Venue', CONCAT('Road · ', Total_Bucket), COUNT(*), COUNTIF(Under_Win), COUNTIF(Over_Win), COUNTIF(Push)
    FROM e GROUP BY Total_Bucket
    """
    df4 = _run(q_bv)

    out = pd.concat([df1, df2, df3, df4], ignore_index=True)
    if out.empty:
        return out
    out = out.groupby(["Side","GroupLabel","Situation"], as_index=False)[["N","W","L","P"]].sum()
    for c in ("N","W","L","P"): out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
    out = out[out["N"] >= min_n].copy()
    den = (out["W"] + out["L"]).astype("float64")
    out["WinPct"] = np.where(den>0, (out["W"].astype("float64")/den)*100.0, np.nan)
    out["ROI_Pct"] = np.where(den>0,
                              ((out["W"].astype("float64")*(100.0/110.0) + out["L"].astype("float64")*(-1.0))/den)*100.0,
                              np.nan)
    out["WinPct"] = pd.to_numeric(out["WinPct"], errors="coerce").round(1)
    out["ROI_Pct"] = pd.to_numeric(out["ROI_Pct"], errors="coerce").round(1)
    return out.sort_values(["GroupLabel","Situation","Side"]).reset_index(drop=True)

# ---------- main UI ----------
def render_current_situations_tab(selected_sport: str):
    st.subheader("📚 Situation DB — League Totals (by Sport)")

    if not selected_sport:
        st.warning("Pick a sport.")
        st.stop()

    book_name = DEFAULT_BOOK  # or expose as a control

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

    league_spreads = league_totals_spreads(selected_sport, cutoff_date_for_stats, min_n, book=book_name)
    league_totals  = league_totals_overunder(selected_sport, cutoff_date_for_stats, min_n, book=book_name)

    ctxs = team_context_from_moves(game_id, teams, book=book_name)
    ctxs = _enforce_role_coherence(ctxs, teams)

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

    # Spreads: full league table
    with st.expander("🔎 League — Full table for this sport (Spreads Only)"):
        if league_spreads.empty:
            st.write("_No rows for this sport/cutoff._")
        else:
            show = league_spreads.copy()
            show = show[show["GroupLabel"].isin(["Role4","Bucket","Bucket·Venue","Venue"])]
            for c in ("N","W","L","P","WinPct","ROI_Pct"):
                if c in show.columns: show[c] = pd.to_numeric(show[c], errors="coerce")
            for c in ("WinPct","ROI_Pct"):
                if c in show.columns: show[c] = show[c].round(1)
            show = show.sort_values(["GroupLabel","Situation"]).reset_index(drop=True)
            st.dataframe(show, use_container_width=True)

    # OVER/UNDER (totals only)
    st.markdown("### 📈 League Totals — Over/Under (Totals Only)")
    if league_totals.empty:
        st.write("_No totals rows meet N threshold for this sport/cutoff._")
    else:
        show = league_totals.copy()
        for c in ("N","W","L","P","WinPct","ROI_Pct"):
            if c in show.columns: show[c] = pd.to_numeric(show[c], errors="coerce")
        for c in ("WinPct","ROI_Pct"):
            if c in show.columns: show[c] = show[c].round(1)
        show = show.sort_values(["GroupLabel","Situation","Side"]).reset_index(drop=True)
        st.dataframe(show[["Side","GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"]],
                     use_container_width=True)

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
              Value, Is_Home, Snapshot_Timestamp, {BOOK_MOVES_COL_SQL} AS Book
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
