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
    # football total line buckets (tweak as desired)
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
    # basketball total line buckets (NBA-ish)
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
    # Venue
    if is_home is True:  out.append("Home")
    if is_home is False: out.append("Road")
    # Role4
    if (is_home is not None) and (is_favorite is not None):
        out.append(f"{'Home' if is_home else 'Road'} {'Favorite' if is_favorite else 'Underdog'}")
    # Buckets (overall + split by venue)
    if bucket:
        out.append(bucket)  # overall bucket
        if is_home is True:  out.append(f"Home · {bucket}")
        if is_home is False: out.append(f"Road · {bucket}")
    # de‑dupe keep order
    seen, keep = set(), []
    for s in out:
        if s and s not in seen:
            keep.append(s); seen.add(s)
    return keep

def _role_label(is_home: bool | None, is_favorite: bool | None) -> str | None:
    if is_home is None or is_favorite is None:
        return None
    return f"{'Home' if is_home else 'Road'} {'Favorite' if is_favorite else 'Underdog'}"

def _compose_three_views(league_spreads: pd.DataFrame,
                         is_home: bool | None, is_favorite: bool | None, bucket: str | None) -> pd.DataFrame:
    """
    Return a single table (in order) with 3 sections:
      1) Overall role (Role4)
      2) Overall by bucket (Bucket)
      3) Bucket × Location (Bucket·Venue)
    Only includes rows that exist / pass N filter.
    """
    if league_spreads is None or league_spreads.empty:
        return pd.DataFrame(columns=["Section","GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"])

    rows = []

    # 1) Overall role
    rlabel = _role_label(is_home, is_favorite)
    if rlabel:
        df1 = league_spreads[
            (league_spreads["GroupLabel"] == "Role4") &
            (league_spreads["Situation"]  == rlabel)
        ]
        if not df1.empty:
            df1 = df1.copy(); df1.insert(0, "Section", "Role")
            rows.append(df1)

    # 2) Overall by bucket
    if bucket:
        df2 = league_spreads[
            (league_spreads["GroupLabel"] == "Bucket") &
            (league_spreads["Situation"]  == bucket)
        ]
        if not df2.empty:
            df2 = df2.copy(); df2.insert(0, "Section", "Bucket")
            rows.append(df2)

        # 3) Bucket × Location
        if is_home is not None:
            bv = f"{'Home' if is_home else 'Road'} · {bucket}"
            df3 = league_spreads[
                (league_spreads["GroupLabel"] == "Bucket·Venue") &
                (league_spreads["Situation"]  == bv)
            ]
            if not df3.empty:
                df3 = df3.copy(); df3.insert(0, "Section", "Bucket × Location")
                rows.append(df3)

    if not rows:
        return pd.DataFrame(columns=["Section","GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"])

    out = pd.concat(rows, ignore_index=True)

    # pretty: round pct columns
    for c in ("WinPct", "ROI_Pct"):
        if c in out.columns:
            out[c] = out[c].map(lambda x: None if pd.isna(x) else round(float(x), 1))

    # enforce order Role -> Bucket -> Bucket × Location
    order = {"Role": 0, "Bucket": 1, "Bucket × Location": 2}
    out["__ord"] = out["Section"].map(order).fillna(99)
    out = out.sort_values(["__ord"]).drop(columns="__ord")
    return out[["Section","GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"]]


# ---------- MOVES: current games (spreads only) & roles ----------
@st.cache_data(ttl=90, show_spinner=False)
def list_current_games_from_moves(sport: str, hard_cap: int = 200) -> pd.DataFrame:
    """
    Returns (Game_Id, Game_Start, Teams[2]) for upcoming games in MOVES for the sport.
    """
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
        AND UPPER(Market) = 'SPREADS'                  -- 🔒 spreads only for listing
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
                bigquery.ScalarQueryParameter("hard_cap", "INT64", int(hard_cap)),
            ],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()

@st.cache_data(ttl=120, show_spinner=False)
def team_context_from_moves(game_id: str, teams: list[str]) -> dict:
    """
    Pick latest SPREADS snapshot row per team (by Market_Leader, Limit, Snapshot_Timestamp).
    Extract Is_Home + Value (team POV spread; fav < 0) to infer role/bucket.
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
        Value,                          -- team POV spread; fav < 0
        Snapshot_Timestamp,
        Market_Leader,
        `Limit`
      FROM {MOVES}
      WHERE COALESCE(Game_Start, Commence_Hour, feat_Game_Start) IS NOT NULL
        AND UPPER(Market) = 'SPREADS'                  -- 🔒 SPREADS ONLY for role
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
        Is_Home,
        Value,
        gs AS cutoff,
        Market_Leader,
        `Limit`,
        Snapshot_Timestamp
      FROM src
      WHERE home_n IS NOT NULL AND away_n IS NOT NULL
    ),
    picked AS (
      SELECT
        team_norm,
        ARRAY_AGG(
          STRUCT(Is_Home, Value, cutoff)
          ORDER BY Market_Leader DESC, `Limit` DESC, Snapshot_Timestamp DESC
          LIMIT 1
        )[OFFSET(0)] AS s
      FROM canon
      WHERE (concat_id = LOWER(@gid) OR stable_id = LOWER(@gid))
        AND team_norm IN UNNEST(@teams_norm)
      GROUP BY team_norm
    )
    SELECT
      team_norm AS Team,
      s.Is_Home      AS is_home,
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
            "is_home": bool(r["is_home"]) if pd.notnull(r["is_home"]) else None,
            "is_favorite": (spf is not None and spf < 0),
            "closing_spread": spf,
            "cutoff": _to_utc_dt(r.get("cutoff")),
        }
    return out

def _enforce_role_coherence(ctxs: dict, teams: list[str]) -> dict:
    """
    Ensure exactly one favorite and one underdog iff any spread exists.
    If both spreads present, lower (more negative) spread is the favorite.
    """
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
        present = sa if sa is not None else sb
        if sa is not None:
            a_fav = sa < 0
            b_fav = not a_fav
        else:
            b_fav = sb < 0
            a_fav = not b_fav

    for t, fav in [(a, a_fav), (b, b_fav)]:
        if t in ctxs:
            ctxs[t]["is_favorite"] = bool(fav)
    return ctxs


# ---------- SCORES: league-wide situation totals ----------
@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def league_totals_spreads(sport: str, cutoff_date: date, min_n: int = 0) -> pd.DataFrame:
    sql = f"""
    WITH base AS (
      SELECT
        UPPER(Sport) AS Sport_U,
        UPPER(Market) AS Market_U,
        Is_Home,
        Value AS Closing_Spread,             -- team POV spread; fav < 0
        Spread_Cover_Flag,
        ATS_Cover_Margin,
        feat_Game_Start
      FROM {SCORES}
      WHERE UPPER(Sport) = @sport_upper
        AND UPPER(Market) = 'SPREADS'        -- 🔒 spreads only for ATS
        AND DATE(feat_Game_Start) <= @cutoff
    ),
    enriched AS (
      SELECT
        Is_Home,
        Closing_Spread,
        CASE WHEN Closing_Spread IS NULL THEN NULL
             WHEN Closing_Spread < 0 THEN TRUE ELSE FALSE END AS Is_Favorite,
        CASE
          WHEN UPPER(@sport_upper) IN ('NBA','NCAAB','WNBA') THEN
            CASE
              WHEN Closing_Spread IS NULL THEN ''
              WHEN Closing_Spread <= -12.5 THEN 'Fav ≤ -12.5'
              WHEN Closing_Spread <=  -9.5 THEN 'Fav -10 to -12.5'
              WHEN Closing_Spread <=  -6.5 THEN 'Fav -7 to -9.5'
              WHEN Closing_Spread <=  -3.5 THEN 'Fav -4 to -6.5'
              WHEN Closing_Spread <=  -0.5 THEN 'Fav -0.5 to -3.5'
              WHEN Closing_Spread =    0.0 THEN 'Pick (0)'
              WHEN Closing_Spread <=   3.5 THEN 'Dog +0.5 to +3.5'
              WHEN Closing_Spread <=   6.5 THEN 'Dog +4 to +6.5'
              WHEN Closing_Spread <=   9.5 THEN 'Dog +7 to +9.5'
              WHEN Closing_Spread <=  12.5 THEN 'Dog +10 to +12.5'
              ELSE 'Dog ≥ +13'
            END
          ELSE
            CASE
              WHEN Closing_Spread IS NULL THEN ''
              WHEN Closing_Spread <= -10.5 THEN 'Fav ≤ -10.5'
              WHEN Closing_Spread <=  -7.5 THEN 'Fav -8 to -10.5'
              WHEN Closing_Spread <=  -6.5 THEN 'Fav -7 to -6.5'
              WHEN Closing_Spread <=  -3.5 THEN 'Fav -4 to -6.5'
              WHEN Closing_Spread <=  -0.5 THEN 'Fav -0.5 to -3.5'
              WHEN Closing_Spread =    0.0 THEN 'Pick (0)'
              WHEN Closing_Spread <=   3.5 THEN 'Dog +0.5 to +3.5'
              WHEN Closing_Spread <=   6.5 THEN 'Dog +4 to +6.5'
              WHEN Closing_Spread <=  10.5 THEN 'Dog +7 to +10.5'
              ELSE 'Dog ≥ +11'
            END
        END AS Spread_Bucket,
        Spread_Cover_Flag,
        ATS_Cover_Margin
      FROM base
    ),

    -- Home / Road
    venue AS (
      SELECT
        'Venue' AS GroupLabel,
        CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END AS Situation,
        COUNT(*) AS N,
        COUNTIF(Spread_Cover_Flag = 1) AS W,
        COUNTIF(Spread_Cover_Flag = 0) AS L,
        COUNTIF(ATS_Cover_Margin = 0)  AS P
      FROM enriched
      WHERE Is_Home IS NOT NULL
      GROUP BY Situation
    ),

    -- Home Favorite, Home Underdog, Road Favorite, Road Underdog
    role4 AS (
      SELECT
        'Role4' AS GroupLabel,
        CONCAT(CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END, ' ',
               CASE WHEN Is_Favorite THEN 'Favorite' ELSE 'Underdog' END) AS Situation,
        COUNT(*) AS N,
        COUNTIF(Spread_Cover_Flag = 1) AS W,
        COUNTIF(Spread_Cover_Flag = 0) AS L,
        COUNTIF(ATS_Cover_Margin = 0)  AS P
      FROM enriched
      WHERE Is_Home IS NOT NULL AND Is_Favorite IS NOT NULL
      GROUP BY Situation
    ),

    -- Spread buckets (overall)
    buckets AS (
      SELECT
        'Bucket' AS GroupLabel,
        Spread_Bucket AS Situation,
        COUNT(*) AS N,
        COUNTIF(Spread_Cover_Flag = 1) AS W,
        COUNTIF(Spread_Cover_Flag = 0) AS L,
        COUNTIF(ATS_Cover_Margin = 0)  AS P
      FROM enriched
      WHERE Spread_Bucket <> ''
      GROUP BY Situation
    ),

    -- 🔥 NEW: Spread buckets split by Home/Road
    buckets_by_venue AS (
      SELECT
        'Bucket·Venue' AS GroupLabel,
        CONCAT(CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END, ' · ', Spread_Bucket) AS Situation,
        COUNT(*) AS N,
        COUNTIF(Spread_Cover_Flag = 1) AS W,
        COUNTIF(Spread_Cover_Flag = 0) AS L,
        COUNTIF(ATS_Cover_Margin = 0)  AS P
      FROM enriched
      WHERE Spread_Bucket <> '' AND Is_Home IS NOT NULL
      GROUP BY Situation
    ),

    unioned AS (
      SELECT * FROM venue
      UNION ALL SELECT * FROM role4
      UNION ALL SELECT * FROM buckets
      UNION ALL SELECT * FROM buckets_by_venue   -- 👈 include new split
    )

    SELECT
      GroupLabel, Situation, N, W, L, P,
      SAFE_MULTIPLY(SAFE_DIVIDE(W, NULLIF(W + L, 0)), 100.0) AS WinPct,
      CASE
        WHEN (W + L) > 0 THEN ((W * (100.0/110.0) + L * (-1.0)) / (W + L)) * 100.0
        ELSE NULL END AS ROI_Pct
    FROM unioned
    WHERE N >= @min_n
    ORDER BY GroupLabel, Situation
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sport_upper","STRING",(sport or "").upper()),
                bigquery.ScalarQueryParameter("cutoff","DATE", cutoff_date),
                bigquery.ScalarQueryParameter("min_n","INT64", int(min_n)),
            ],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def league_totals_overunder(sport: str, cutoff_date: date, min_n: int = 0) -> pd.DataFrame:
    """
    League-wide (sport-wide) Over/Under results **totals only**:
      - Side = Over / Under
      - Total line buckets (football vs basketball buckets)
      - Uses per-game de-duplication via feat_Game_Key
      - ROI computed at -110 for the chosen side (pushes excluded)
    """
    sql = f"""
    -- 1) Totals rows only; compute per-game total points
    WITH base AS (
      SELECT
        UPPER(Sport) AS Sport_U,
        UPPER(Market) AS Market_U,
        COALESCE(feat_Game_Key, Game_Key) AS GKey,
        feat_Game_Start,
        Value AS Total_Line,                                -- totals line (same both teams)
        SAFE_CAST(Team_Score AS FLOAT64) AS Team_Score,
        SAFE_CAST(Opp_Score  AS FLOAT64) AS Opp_Score
      FROM {SCORES}
      WHERE UPPER(Sport) = @sport_upper
        AND UPPER(Market) = 'TOTALS'                         -- 🔒 totals only
        AND DATE(feat_Game_Start) <= @cutoff
        AND Value IS NOT NULL
        AND Team_Score IS NOT NULL AND Opp_Score IS NOT NULL
    ),
    per_game AS (
      -- de-duplicate to one row per game
      SELECT
        GKey,
        ANY_VALUE(feat_Game_Start) AS gs,
        ANY_VALUE(Total_Line)      AS Total_Line,
        ANY_VALUE(Team_Score + Opp_Score) AS Total_Points
      FROM base
      GROUP BY GKey
    ),
    enriched AS (
      SELECT
        Total_Line,
        Total_Points,
        -- per-side win flags
        (Total_Points > Total_Line) AS Over_Win,
        (Total_Points < Total_Line) AS Under_Win,
        (Total_Points = Total_Line) AS Push,
        -- sport-aware total buckets
        CASE
          WHEN UPPER(@sport_upper) IN ('NBA','NCAAB','WNBA') THEN
            CASE
              WHEN Total_Line <= 205.5 THEN 'OU ≤ 205.5'
              WHEN Total_Line <= 209.5 THEN 'OU 206–209.5'
              WHEN Total_Line <= 213.5 THEN 'OU 210–213.5'
              WHEN Total_Line <= 217.5 THEN 'OU 214–217.5'
              WHEN Total_Line <= 221.5 THEN 'OU 218–221.5'
              WHEN Total_Line <= 225.5 THEN 'OU 222–225.5'
              WHEN Total_Line <= 229.5 THEN 'OU 226–229.5'
              WHEN Total_Line <= 233.5 THEN 'OU 230–233.5'
              ELSE 'OU ≥ 234'
            END
          ELSE
            CASE
              WHEN Total_Line <= 37.5 THEN 'OU ≤ 37.5'
              WHEN Total_Line <= 41.5 THEN 'OU 38–41.5'
              WHEN Total_Line <= 44.5 THEN 'OU 42–44.5'
              WHEN Total_Line <= 47.5 THEN 'OU 45–47.5'
              WHEN Total_Line <= 50.5 THEN 'OU 48–50.5'
              WHEN Total_Line <= 53.5 THEN 'OU 51–53.5'
              WHEN Total_Line <= 56.5 THEN 'OU 54–56.5'
              WHEN Total_Line <= 59.5 THEN 'OU 57–59.5'
              ELSE 'OU ≥ 60'
            END
        END AS Total_Bucket
      FROM per_game
    ),
    over_rows AS (
      SELECT
        'Over' AS Side,
        Total_Bucket AS Situation,
        COUNT(*) AS N,
        COUNTIF(Over_Win) AS W,
        COUNTIF(Under_Win) AS L,
        COUNTIF(Push) AS P
      FROM enriched
      GROUP BY Situation
    ),
    under_rows AS (
      SELECT
        'Under' AS Side,
        Total_Bucket AS Situation,
        COUNT(*) AS N,
        COUNTIF(Under_Win) AS W,
        COUNTIF(Over_Win) AS L,
        COUNTIF(Push) AS P
      FROM enriched
      GROUP BY Situation
    ),
    unioned AS (
      SELECT * FROM over_rows
      UNION ALL
      SELECT * FROM under_rows
    )
    SELECT
      Side, Situation, N, W, L, P,
      SAFE_MULTIPLY(SAFE_DIVIDE(W, NULLIF(W + L, 0)), 100.0) AS WinPct,
      CASE
        WHEN (W + L) > 0 THEN ((W * (100.0/110.0) + L * (-1.0)) / (W + L)) * 100.0
        ELSE NULL END AS ROI_Pct
    FROM unioned
    WHERE N >= @min_n
    ORDER BY Situation, Side
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sport_upper","STRING",(sport or "").upper()),
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

    games = list_current_games_from_moves(selected_sport)
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

    # League-wide tables (one query each)
    league_spreads = league_totals_spreads(selected_sport, cutoff_date_for_stats, min_n)
    league_totals  = league_totals_overunder(selected_sport, cutoff_date_for_stats, min_n)

    # Roles from MOVES (SPREADS-ONLY)
    ctxs = team_context_from_moves(game_id, teams)
    ctxs = _enforce_role_coherence(ctxs, teams)

    # Show per-team ROLE → league rows (spreads / ATS)
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

            st.markdown("**ATS (Spreads Only) — Role + Bucket + Bucket × Location**")

            table3 = _compose_three_views(
                league_spreads=league_spreads,
                is_home=is_home,
                is_favorite=is_favorite,
                bucket=bucket
            )
            
            if table3.empty:
                st.write("_No league rows meet N threshold for this role/bucket._")
            else:
                st.dataframe(
                    table3,
                    use_container_width=True
                )
    with st.expander("🔎 League — Full table for this sport"):
        show = league_df.copy()
        show = show[show["GroupLabel"].isin(["Role4","Bucket","Bucket·Venue"])]
        for c in ["WinPct","ROI_Pct"]:
            show[c] = show[c].map(lambda x: None if pd.isna(x) else round(float(x), 1))
        st.dataframe(show, use_container_width=True)


    # League-wide OVER/UNDER tables (separate; totals-only)
    st.markdown("### 📈 League Totals — Over/Under (Totals Only)")
    if league_totals.empty:
        st.write("_No totals rows meet N threshold for this sport/cutoff._")
    else:
        show = league_totals.copy()
        for c in ["WinPct","ROI_Pct"]:
            if c in show.columns:
                show[c] = show[c].map(lambda x: None if pd.isna(x) else round(float(x), 1))
        st.dataframe(show[["Situation","Side","N","W","L","P","WinPct","ROI_Pct"]],
                     use_container_width=True)

# ---------- optional quick debug ----------
def render_current_games_section(selected_sport: str):
    st.subheader("📡 Current/Upcoming Games (from MOVES — spreads only)")
    games = list_current_games_from_moves(selected_sport)
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
              Value, Is_Home, Snapshot_Timestamp
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
