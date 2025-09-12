# ================== Situations Tab (current games from MOVES; history from SCORES) ==================
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, date, time, timezone
from google.cloud import bigquery

PROJECT = "sharplogger"
DATASET = "sharp_data"
MOVES_FQ  = f"{PROJECT}.{DATASET}.moves_with_features_merged"
SCORES_FQ = f"{PROJECT}.{DATASET}.scores_with_features"

MOVES  = f"`{MOVES_FQ}`"
SCORES = f"`{SCORES_FQ}`"

CLIENT = bigquery.Client(project=PROJECT)
CACHE_TTL_SEC = 120

# ---------- helpers ----------
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

def _spread_bucket(v: float | None) -> str:
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

def _wanted_situations(is_home: bool | None, is_favorite: bool | None, spread_bucket: str | None):
    out = []
    if is_home is True  and is_favorite is True:   out.append("Home Favorite")
    if is_home is True  and is_favorite is False:  out.append("Home Underdog")
    if is_home is False and is_favorite is True:   out.append("Road Favorite")
    if is_home is False and is_favorite is False:  out.append("Road Underdog")
    if is_home is True:  out.append("Home")
    if is_home is False: out.append("Road")
    if is_favorite is True: out.append("Favorite")
    if is_favorite is False: out.append("Underdog")
    if spread_bucket: out.append(spread_bucket)
    # de‑dupe, keep order
    seen, keep = set(), []
    for s in out:
        if s and s not in seen:
            keep.append(s); seen.add(s)
    return keep

# ---------- 1) CURRENT GAMES from MOVES (Sport filtered, start > NOW) ----------
@st.cache_data(ttl=CACHE_TTL_SEC)
def list_current_games_from_moves(sport: str) -> pd.DataFrame:
    """
    Build the current/upcoming games list from the live moves table.
    Uses COALESCE(feat_Game_Start, Commence_Hour) to tolerate missing values.
    Filters: Sport = @sport AND start > NOW().
    """
    sql = f"""
    WITH src AS (
      SELECT
        feat_Game_Key,
        COALESCE(feat_Game_Start, Commence_Hour) AS gs,
        feat_Team
      FROM {MOVES}
      WHERE Sport = @sport
        AND COALESCE(feat_Game_Start, Commence_Hour) > CURRENT_TIMESTAMP()
    )
    SELECT
      feat_Game_Key AS Game_Id,
      ANY_VALUE(gs) AS Game_Start,
      ARRAY_AGG(DISTINCT feat_Team ORDER BY feat_Team) AS Teams
    FROM src
    GROUP BY feat_Game_Key
    HAVING ARRAY_LENGTH(Teams) >= 2
    ORDER BY Game_Start ASC
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("sport","STRING", sport)],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()

# ---------- 2) TEAM ROLE CONTEXT from MOVES (home/fav/spread bucket) ----------
@st.cache_data(ttl=CACHE_TTL_SEC)
def team_context_from_moves(game_id: str, teams: list[str]) -> dict:
    """
    Returns {team: {is_home, is_favorite, spread_bucket, cutoff}} for the given game.
    Reads from MOVES only (consistent, fast). Adjust Value→Closing_Spread_For_Team if desired.
    """
    if not teams:
        return {}
    sql = f"""
    WITH t AS (SELECT team FROM UNNEST(@teams) AS team)
    SELECT
      feat_Team,
      ANY_VALUE(Is_Home)         AS is_home,
      ANY_VALUE(Value)           AS closing_spread,  -- or Closing_Spread_For_Team
      ANY_VALUE(COALESCE(feat_Game_Start, Commence_Hour)) AS cutoff
    FROM {MOVES}
    WHERE feat_Game_Key = @gid
      AND feat_Team IN (SELECT team FROM t)
    GROUP BY feat_Team
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("gid","STRING", game_id),
                bigquery.ArrayQueryParameter("teams","STRING", teams),
            ],
            use_query_cache=True,
        ),
    )
    df = job.result().to_dataframe()

    out = {}
    for _, r in df.iterrows():
        sp  = r.get("closing_spread")
        spf = float(sp) if pd.notnull(sp) else None
        out[str(r["feat_Team"])] = {
            "is_home": bool(r["is_home"]) if pd.notnull(r["is_home"]) else None,
            "is_favorite": (spf is not None and spf < 0),
            "spread_bucket": _spread_bucket(spf),
            "cutoff": _to_utc_dt(r.get("cutoff")),
        }
    return out

# ---------- 3) SITUATIONS from SCORES (history up to a DATE cutoff) ----------
@st.cache_data(ttl=CACHE_TTL_SEC)
def situations_ats_by_team(sport: str, cutoff_date: date, min_n: int) -> pd.DataFrame:
    """
    ATS situations per team from scores_with_features (history).
    Uses Spread_Cover_Flag and Value columns you confirmed.
    """
    sql = f"""
    WITH base AS (
      SELECT
        feat_Team AS Team,
        Is_Home,
        Value AS Closing_Spread,
        Spread_Cover_Flag,
        ATS_Cover_Margin
      FROM {SCORES}
      WHERE Sport = @sport
        AND DATE(feat_Game_Start) <= @cutoff
    ),
    enriched AS (
      SELECT
        Team, Is_Home, Closing_Spread, Spread_Cover_Flag, ATS_Cover_Margin,
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
        END AS Spread_Bucket,
        CASE WHEN Closing_Spread IS NULL THEN NULL
             WHEN Closing_Spread < 0 THEN TRUE ELSE FALSE END AS Is_Favorite
      FROM base
    ),
    role4 AS (
      SELECT
        Team,
        CONCAT(CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END, ' ',
               CASE WHEN Is_Favorite THEN 'Favorite' ELSE 'Underdog' END) AS Situation,
        COUNT(*) AS N,
        COUNTIF(Spread_Cover_Flag = 1) AS W,
        COUNTIF(Spread_Cover_Flag = 0) AS L,
        COUNTIF(ATS_Cover_Margin = 0)  AS P
      FROM enriched
      GROUP BY Team, Situation
    ),
    home_road AS (
      SELECT Team,
             CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END AS Situation,
             COUNT(*) AS N,
             COUNTIF(Spread_Cover_Flag = 1) AS W,
             COUNTIF(Spread_Cover_Flag = 0) AS L,
             COUNTIF(ATS_Cover_Margin = 0)  AS P
      FROM enriched
      GROUP BY Team, Situation
    ),
    fav_dog AS (
      SELECT Team,
             CASE WHEN Is_Favorite THEN 'Favorite' ELSE 'Underdog' END AS Situation,
             COUNT(*) AS N,
             COUNTIF(Spread_Cover_Flag = 1) AS W,
             COUNTIF(Spread_Cover_Flag = 0) AS L,
             COUNTIF(ATS_Cover_Margin = 0)  AS P
      FROM enriched
      GROUP BY Team, Situation
    ),
    buckets AS (
      SELECT Team, Spread_Bucket AS Situation,
             COUNT(*) AS N,
             COUNTIF(Spread_Cover_Flag = 1) AS W,
             COUNTIF(Spread_Cover_Flag = 0) AS L,
             COUNTIF(ATS_Cover_Margin = 0)  AS P
      FROM enriched
      WHERE Spread_Bucket <> ''
      GROUP BY Team, Situation
    ),
    unioned AS (
      SELECT * FROM role4
      UNION ALL SELECT * FROM home_road
      UNION ALL SELECT * FROM fav_dog
      UNION ALL SELECT * FROM buckets
    )
    SELECT
      Team, Situation, N, W, L, P,
      SAFE_MULTIPLY(SAFE_DIVIDE(W, NULLIF(W + L, 0)), 100.0) AS WinPct,
      CASE WHEN (W + L) > 0
           THEN ((W * (100.0/110.0) + L * (-1.0)) / (W + L)) * 100.0
           ELSE NULL END AS ROI_Pct
    FROM unioned
    WHERE N >= @min_n
    ORDER BY Team, WinPct DESC, N DESC
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sport","STRING", sport),
                bigquery.ScalarQueryParameter("cutoff","DATE", cutoff_date),
                bigquery.ScalarQueryParameter("min_n","INT64", int(min_n)),
            ],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()

@st.cache_data(ttl=CACHE_TTL_SEC)
def situations_ml_by_team(sport: str, cutoff_date: date, min_n: int) -> pd.DataFrame:
    sql = f"""
    WITH base AS (
      SELECT
        feat_Team AS Team,
        Is_Home,
        Value AS Closing_Spread,
        Team_Score, Opp_Score
      FROM {SCORES}
      WHERE Sport = @sport
        AND DATE(feat_Game_Start) <= @cutoff
    ),
    enriched AS (
      SELECT
        Team, Is_Home, Closing_Spread, Team_Score, Opp_Score,
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
        END AS Spread_Bucket,
        CASE WHEN Closing_Spread IS NULL THEN NULL
             WHEN Closing_Spread < 0 THEN TRUE ELSE FALSE END AS Is_Favorite,
        CASE WHEN Team_Score > Opp_Score THEN 1 ELSE 0 END AS ML_Win
      FROM base
    ),
    role4 AS (
      SELECT Team,
             CONCAT(CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END, ' ',
                    CASE WHEN Is_Favorite THEN 'Favorite' ELSE 'Underdog' END) AS Situation,
             COUNT(*) AS N,
             SUM(ML_Win) AS W,
             COUNT(*) - SUM(ML_Win) AS L,
             0 AS P
      FROM enriched
      GROUP BY Team, Situation
    ),
    home_road AS (
      SELECT Team,
             CASE WHEN Is_Home THEN 'Home' ELSE 'Road' END AS Situation,
             COUNT(*) AS N,
             SUM(ML_Win) AS W,
             COUNT(*) - SUM(ML_Win) AS L,
             0 AS P
      FROM enriched
      GROUP BY Team, Situation
    ),
    fav_dog AS (
      SELECT Team,
             CASE WHEN Is_Favorite THEN 'Favorite' ELSE 'Underdog' END AS Situation,
             COUNT(*) AS N,
             SUM(ML_Win) AS W,
             COUNT(*) - SUM(ML_Win) AS L,
             0 AS P
      FROM enriched
      GROUP BY Team, Situation
    ),
    buckets AS (
      SELECT Team, Spread_Bucket AS Situation,
             COUNT(*) AS N,
             SUM(ML_Win) AS W,
             COUNT(*) - SUM(ML_Win) AS L,
             0 AS P
      FROM enriched
      WHERE Spread_Bucket <> ''
      GROUP BY Team, Situation
    ),
    unioned AS (
      SELECT * FROM role4
      UNION ALL SELECT * FROM home_road
      UNION ALL SELECT * FROM fav_dog
      UNION ALL SELECT * FROM buckets
    )
    SELECT
      Team, Situation, N, W, L, P,
      SAFE_MULTIPLY(SAFE_DIVIDE(W, NULLIF(W + L, 0)), 100.0) AS WinPct
    FROM unioned
    WHERE N >= @min_n
    ORDER BY Team, WinPct DESC, N DESC
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sport","STRING", sport),
                bigquery.ScalarQueryParameter("cutoff","DATE", cutoff_date),
                bigquery.ScalarQueryParameter("min_n","INT64", int(min_n)),
            ],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()

@st.cache_data(ttl=CACHE_TTL_SEC)
def league_role_leaderboard_spreads(sport: str, cutoff_ts: dt.datetime,
                                    is_home: bool | None,
                                    spread_bucket: str | None,
                                    is_favorite: bool | None,
                                    min_n: int) -> pd.DataFrame:
    cutoff_ts = _to_utc_dt(cutoff_ts)
    sql = f"""
    WITH e AS (
      SELECT
        feat_Team AS Team,
        Is_Home,
        Value AS Closing_Spread,
        Spread_Cover_Flag,
        ATS_Cover_Margin,
        feat_Game_Start
      FROM {SCORES}
      WHERE Sport = @sport
        AND feat_Game_Start < @cutoff
        AND feat_Game_Start >= TIMESTAMP(DATETIME_SUB(DATETIME(@cutoff), INTERVAL 10 YEAR))
    ),
    with_bucket AS (
      SELECT
        Team,
        Is_Home,
        Closing_Spread,
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
        END AS Spread_Bucket,
        Spread_Cover_Flag,
        ATS_Cover_Margin
      FROM e
    ),
    filtered AS (
      SELECT *
      FROM with_bucket
      WHERE (@is_home IS NULL OR Is_Home = @is_home)
        AND (@spread_bucket = '' OR Spread_Bucket = @spread_bucket)
        AND (@is_favorite IS NULL OR (Closing_Spread IS NOT NULL AND (Closing_Spread < 0) = @is_favorite))
    ),
    by_team AS (
      SELECT
        Team,
        COUNT(*) AS N,
        COUNTIF(Spread_Cover_Flag = 1) AS W,
        COUNTIF(Spread_Cover_Flag = 0) AS L,
        COUNTIF(ATS_Cover_Margin = 0)  AS P
      FROM filtered
      GROUP BY Team
      HAVING N >= @min_n
    )
    SELECT
      Team, N, W, L, P,
      SAFE_MULTIPLY(SAFE_DIVIDE(W, NULLIF(W + L, 0)), 100.0) AS WinPct,
      CASE
        WHEN (W + L) > 0 THEN ((W * (100.0/110.0) + L * (-1.0)) / (W + L)) * 100.0
        ELSE NULL END AS ROI_Pct
    FROM by_team
    ORDER BY IFNULL(ROI_Pct, 0.0) DESC, WinPct DESC, N DESC
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sport","STRING", sport),
                bigquery.ScalarQueryParameter("cutoff","TIMESTAMP", cutoff_ts),
                bigquery.ScalarQueryParameter("is_home","BOOL", is_home),
                bigquery.ScalarQueryParameter("spread_bucket","STRING", spread_bucket or ""),
                bigquery.ScalarQueryParameter("is_favorite","BOOL", is_favorite),
                bigquery.ScalarQueryParameter("min_n","INT64", int(min_n)),
            ],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()

# ---------- RENDER ----------
def render_current_situations_tab(selected_sport: str):
    st.subheader("📚 Situation DB (Current Games)")

    if not selected_sport:
        st.warning("Pick a sport.")
        st.stop()

    games = list_current_games_from_moves(selected_sport)
    if games.empty:
        st.info("No upcoming games found for this sport (from moves).")
        return

    games = games.copy()
    games["Teams"] = games["Teams"].apply(lambda xs: list(xs)[:2])
    games["label"] = games.apply(
        lambda r: f"{r['Game_Start']} — {', '.join(r['Teams'])}" if r["Teams"]
                  else f"{r['Game_Start']} — (teams TBD)",
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

    # Role context straight from moves
    ctxs = team_context_from_moves(game_id, teams)

    cols = st.columns(2)
    for i, team in enumerate(teams):
        with cols[i]:
            st.markdown(f"### {team}")

            ctx = ctxs.get(team, {})
            is_home = ctx.get("is_home")
            is_favorite = ctx.get("is_favorite")
            bucket = ctx.get("spread_bucket")
            cutoff_ts = _to_utc_dt(ctx.get("cutoff") or game_start)  # for leaderboard

            bits = []
            bits.append("🏠 Home" if is_home is True else ("🚗 Road" if is_home is False else ""))
            bits.append("⭐ Favorite" if is_favorite is True else ("🐶 Underdog" if is_favorite is False else ""))
            if bucket: bits.append(bucket)
            st.caption(" / ".join([b for b in bits if b]) or "Role: Unknown")

            # ATS situations (history)
            st.markdown("**Spreads — current role**")
            ats_all = situations_ats_by_team(selected_sport, cutoff_date_for_stats, min_n)
            _print_matching_situations(team, "spreads", ats_all[ats_all["Team"] == team], _wanted_situations(is_home, is_favorite, bucket))

            # ML situations (history)
            st.markdown("**Moneyline — current role**")
            ml_all = situations_ml_by_team(selected_sport, cutoff_date_for_stats, min_n)
            _print_matching_situations(team, "moneyline", ml_all[ml_all["Team"] == team], _wanted_situations(is_home, is_favorite, bucket))

            # League leaderboard for this exact role (spreads)
            st.markdown("**League — this exact role (Spreads)**")
            lb = league_role_leaderboard_spreads(
                selected_sport,
                cutoff_ts,
                is_home,
                bucket,
                is_favorite,
                min_n
            )
            if lb.empty:
                st.write("_No teams meet N threshold in this role._")
            else:
                st.dataframe(lb.head(20))

def _print_matching_situations(team, market, df, wanted_order):
    if df is None or df.empty:
        st.write("_No situations meeting N threshold._")
        return
    # df contains multiple situations for the team; index by name
    by_name = {row.Situation: row for _, row in df.iterrows()}
    printed = 0
    for name in wanted_order:
        r = by_name.get(name)
        if r is None:
            continue
        W = int(r.W) if pd.notnull(r.W) else 0
        L = int(r.L) if pd.notnull(r.L) else 0
        P = int(r.P) if ("P" in r.index and pd.notnull(r.P)) else 0
        N = int(r.N) if pd.notnull(r.N) else 0
        winpct = float(r.WinPct) if pd.notnull(r.WinPct) else 0.0
        roi = r.get("ROI_Pct")
        roi_txt = f", ROI {float(roi):+,.1f}%" if roi is not None and pd.notnull(roi) else ""
        st.markdown(f"• **{team} — {market.capitalize()}** · {name}: {W}-{L}{('-'+str(P)+'P') if P else ''} ({winpct:.1f}%) over {N}{roi_txt}.")
        printed += 1
        if printed >= 3:
            break
    if printed == 0:
        for _, r in df.head(3).iterrows():
            name = r.Situation
            W = int(r.W) if pd.notnull(r.W) else 0
            L = int(r.L) if pd.notnull(r.L) else 0
            P = int(r.P) if ("P" in r.index and pd.notnull(r.P)) else 0
            N = int(r.N) if pd.notnull(r.N) else 0
            winpct = float(r.WinPct) if pd.notnull(r.WinPct) else 0.0
            roi = r.get("ROI_Pct")
            roi_txt = f", ROI {float(roi):+,.1f}%" if roi is not None and pd.notnull(roi) else ""
            st.markdown(f"• **{team} — {market.capitalize()}** · {name}: {W}-{L}{('-'+str(P)+'P') if P else ''} ({winpct:.1f}%) over {N}{roi_txt}.")
