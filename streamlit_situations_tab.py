# streamlit_situations_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timezone
from google.cloud import bigquery

# --- constants / config ----------------------------------------------------
PROJECT = "sharplogger"
DATASET = "sharp_data"

VIEW_GAMES       = f"`{PROJECT}.{DATASET}.scores_games_list`"
SCORES_ROLE_VIEW = f"`{PROJECT}.{DATASET}.scores_role_view`"
MOVES_TABLE      = f"`{PROJECT}.{DATASET}.moves_with_features_merged`"

# Schema-aligned columns
HOME_COL   = "Is_Home"   # BOOL in moves/scores_role_view
SPREAD_COL = "Value"     # closing spread for the team (fav < 0)

CLIENT = bigquery.Client(project=PROJECT)
CACHE_TTL_SEC = 90

# --- small utils -----------------------------------------------------------

# --- small utils -----------------------------------------------------------
import datetime as dt
import pandas as pd

def _coerce_timestamp(x) -> dt.datetime | None:
    """
    Return a timezone-aware UTC datetime or None.
    - Returns None for None/NaT/empty.
    - Accepts datetime, pandas.Timestamp, or date (assumed 00:00 local).
    - Localizes naive inputs to UTC.
    """
    if x is None:
        return None

    # Pandas Timestamp
    if isinstance(x, pd.Timestamp):
        if pd.isna(x):                 # catches NaT
            return None
        ts = x
    # Python datetime
    elif isinstance(x, dt.datetime):
        ts = pd.Timestamp(x)
    # Python date (no time) → assume midnight
    elif isinstance(x, dt.date):
        ts = pd.Timestamp(dt.datetime.combine(x, dt.time(0, 0, 0)))
    else:
        # Try to parse anything else with pandas, then re-run checks
        try:
            ts = pd.to_datetime(x)
            if pd.isna(ts):
                return None
        except Exception:
            return None

    # Ensure tz-aware UTC
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.to_pydatetime()

def _round_cutoff(ts: dt.datetime | None) -> dt.datetime | None:
    if ts is None:
        return None
    return ts.replace(minute=0, second=0, microsecond=0)

def _fallback_now_utc() -> dt.datetime:
    return datetime.now(timezone.utc)

def _safe_cutoff(primary: dt.datetime | None, fallback: dt.datetime | None) -> dt.datetime | None:
    """Pick a valid cutoff; prefer primary, else fallback, else None."""
    a = _coerce_timestamp(primary)
    b = _coerce_timestamp(fallback)
    return a or b

def _coerce_teams_list(x):
    """Normalize Teams to a list[str]. Accepts list, array, 'A, B' string, etc."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple)):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, np.ndarray):
        return [str(v).strip() for v in x.tolist() if str(v).strip()]
    if isinstance(x, pd.Series):
        return [str(v).strip() for v in x.astype(str).tolist() if str(v).strip()]
    if isinstance(x, str):
        parts = [p.strip() for p in x.split(",")] if "," in x else [x.strip()]
        return [p for p in parts if p]
    if hasattr(x, "tolist"):
        try:
            out = x.tolist()
            if isinstance(out, (list, tuple, np.ndarray, pd.Series)):
                return [str(v).strip() for v in list(out) if str(v).strip()]
            return [str(out).strip()]
        except Exception:
            pass
    try:
        return [str(v).strip() for v in list(x) if str(v).strip()]
    except Exception:
        return [str(x).strip()]

def _spread_bucket(v: float | None) -> str:
    if v is None:
        return ""
    try:
        v = float(v)
    except Exception:
        return ""
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

# --- DB helpers (cached) ---------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SEC)
def list_games_cached(sport_label: str) -> pd.DataFrame:
    SPORT_MAP = {
        "NFL": "NFL", "NCAAF": "NCAAF", "NBA": "NBA",
        "NCAAB": "NCAAB", "MLB": "MLB", "WNBA": "WNBA", "CFL": "CFL",
    }
    db_sport = SPORT_MAP.get(sport_label, sport_label)
    sql = f"""
      SELECT Game_Id, Game_Start, Teams
      FROM {VIEW_GAMES}
      WHERE Sport = @sport
        AND Game_Start > CURRENT_TIMESTAMP()
      ORDER BY Game_Start ASC
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("sport","STRING", db_sport)],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()

@st.cache_data(ttl=CACHE_TTL_SEC)
def get_contexts_for_game(game_id: str, teams: tuple[str, ...]) -> dict:
    """Return {team: {is_home, is_favorite, spread_bucket, cutoff, source}} for up to 2 teams."""
    if not teams:
        return {}

    sql = f"""
    WITH teams_param AS (
      SELECT team FROM UNNEST(@teams) AS team
    ),
    moves_pre AS (
      SELECT
        feat_Team,
        ANY_VALUE({HOME_COL})      AS is_home,
        ANY_VALUE({SPREAD_COL})    AS closing_spread,
        ANY_VALUE(feat_Game_Start) AS game_start
      FROM {MOVES_TABLE}
      WHERE feat_Game_Key = @gid
        AND feat_Team IN (SELECT team FROM teams_param)
      GROUP BY feat_Team
    ),
    scores_pre AS (
      SELECT
        feat_Team,
        ANY_VALUE({HOME_COL})      AS is_home,
        ANY_VALUE({SPREAD_COL})    AS closing_spread,
        ANY_VALUE(feat_Game_Start) AS cutoff
      FROM {SCORES_ROLE_VIEW}
      WHERE feat_Game_Key = @gid
        AND feat_Team IN (SELECT team FROM teams_param)
      GROUP BY feat_Team
    ),
    joined AS (
      SELECT
        t.team AS feat_Team,
        m.is_home         AS m_is_home,
        m.closing_spread  AS m_spread,
        m.game_start      AS m_cutoff,
        s.is_home         AS s_is_home,
        s.closing_spread  AS s_spread,
        s.cutoff          AS s_cutoff
      FROM teams_param t
      LEFT JOIN moves_pre  m ON m.feat_Team = t.team
      LEFT JOIN scores_pre s ON s.feat_Team = t.team
    )
    SELECT
      feat_Team,
      COALESCE(m_is_home, s_is_home)  AS is_home,
      COALESCE(m_spread,  s_spread)   AS closing_spread,
      COALESCE(m_cutoff,  s_cutoff)   AS cutoff,
      CASE WHEN m_cutoff IS NOT NULL THEN 'moves' ELSE 'scores' END AS source
    FROM joined
    """

    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("gid", "STRING", game_id),
                bigquery.ArrayQueryParameter("teams", "STRING", list(teams)),
            ],
            use_query_cache=True,
        ),
    )
    df = job.result().to_dataframe()

    out = {}
    for _, r in df.iterrows():
        sp = r.get("closing_spread")
        spf = float(sp) if pd.notnull(sp) else None
        out[r["feat_Team"]] = {
            "is_home": bool(r["is_home"]) if pd.notnull(r["is_home"]) else None,
            "is_favorite": (spf is not None and spf < 0),
            "spread_bucket": _spread_bucket(spf),
            "cutoff": r.get("cutoff"),
            "source": r.get("source"),
        }
    return out

def get_team_context(game_id: str, team: str) -> dict:
    """Single-team wrapper reusing the cached batch function."""
    ctxs = get_contexts_for_game(game_id, (team,))
    return ctxs.get(team, {"is_home": None, "is_favorite": None, "spread_bucket": "", "cutoff": None, "source": "none"})


@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def situation_stats_cached(sport_str: str, team: str, cutoff, market: str, min_n: int):
    # normalize cutoff
    cutoff_ts = _coerce_timestamp(cutoff)  # -> datetime|None

    # base SQL with optional cutoff filter injected
    cutoff_filter = "AND Event_Timestamp < @cutoff_ts" if cutoff_ts is not None else ""

    sql = f"""
    -- situation stats
    WITH base AS (
      SELECT *
      FROM {SCORES_ROLE_VIEW}
      WHERE Sport = @sport
        AND Team  = @team
        AND Market = @market
        {cutoff_filter}
    )
    SELECT
      Team,
      COUNT(*) AS N,
      AVG(CAST(Win_Flag AS INT64)) AS WinRate,
      SAFE_DIVIDE(SUM(Profit_USD), NULLIF(COUNT(*), 0)) AS AvgProfit
    FROM base
    GROUP BY Team
    HAVING N >= @min_n
    """

    params = [
        bigquery.ScalarQueryParameter("sport",  "STRING", sport_str),
        bigquery.ScalarQueryParameter("team",   "STRING", team),
        bigquery.ScalarQueryParameter("market", "STRING", market),
        bigquery.ScalarQueryParameter("min_n",  "INT64",  int(min_n)),
    ]
    if cutoff_ts is not None:
        params.append(bigquery.ScalarQueryParameter("cutoff_ts", "TIMESTAMP", cutoff_ts))

    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=params),
    )
    return job.result().to_dataframe()


@st.cache_data(ttl=CACHE_TTL_SEC)
def league_baseline_filtered_cached(sport: str, cutoff: dt.datetime, market: str,
                                    is_home: bool | None, spread_bucket: str | None, min_n: int):
    cutoff = _round_cutoff(_coerce_timestamp(cutoff))
    if cutoff is None:
        return None
    sql = """
      SELECT * FROM `sharplogger.sharp_data.league_situation_stats_from_scores`
        (@sport, @cutoff, @market, @is_home, @spread_bucket, @min_n)
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sport","STRING", sport),
                bigquery.ScalarQueryParameter("cutoff","TIMESTAMP", cutoff),
                bigquery.ScalarQueryParameter("market","STRING", market),
                bigquery.ScalarQueryParameter("is_home","BOOL", is_home),
                bigquery.ScalarQueryParameter("spread_bucket","STRING", spread_bucket or ""),
                bigquery.ScalarQueryParameter("min_n","INT64", min_n),
            ],
            use_query_cache=True,
        ),
    )
    df = job.result().to_dataframe()
    return None if df.empty else df.iloc[0]

@st.cache_data(ttl=CACHE_TTL_SEC)
def role_leaderboard_cached(sport: str, cutoff: dt.datetime, market: str,
                            is_home: bool | None, spread_bucket: str | None,
                            is_favorite: bool | None, min_n: int = 30) -> pd.DataFrame:
    """Leaderboard of historical performance for this exact role (uses scores_role_view)."""
    cutoff = _round_cutoff(_coerce_timestamp(cutoff))
    if cutoff is None:
        return pd.DataFrame()
    sql = f"""
    WITH src AS (
      SELECT
        feat_Team AS Team,
        {HOME_COL}   AS Is_Home,
        {SPREAD_COL} AS Closing_Spread,
        Spread_Cover_Flag, ATS_Cover_Margin,
        Team_Score, Opp_Score, feat_Game_Start, Sport
      FROM {SCORES_ROLE_VIEW}
      WHERE Sport = @sport
        AND feat_Game_Start < @cutoff
        AND feat_Game_Start >= TIMESTAMP(DATETIME_SUB(DATETIME(@cutoff), INTERVAL 10 YEAR))
    ),
    e AS (
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
        CASE WHEN Spread_Cover_Flag = 1 THEN 'W'
             WHEN Spread_Cover_Flag = 0 THEN 'L'
             WHEN ATS_Cover_Margin  = 0 THEN 'P'
             ELSE NULL END AS ATS_Result,
        CASE WHEN Team_Score > Opp_Score THEN 'W' ELSE 'L' END AS SU_Result
      FROM src
    ),
    filtered AS (
      SELECT *
      FROM e
      WHERE (@is_home IS NULL OR Is_Home = @is_home)
        AND (@spread_bucket = '' OR Spread_Bucket = @spread_bucket)
        AND (@is_favorite IS NULL OR (Closing_Spread IS NOT NULL AND (Closing_Spread < 0) = @is_favorite))
    ),
    by_team AS (
      SELECT
        Team,
        COUNT(*) AS N,
        COUNTIF( (@market='spreads' AND ATS_Result='W') OR
                 (@market='moneyline' AND SU_Result='W') ) AS W,
        COUNTIF( (@market='spreads' AND ATS_Result='L') OR
                 (@market='moneyline' AND SU_Result='L') ) AS L,
        COUNTIF( @market='spreads' AND ATS_Result='P' ) AS P
      FROM filtered
      GROUP BY Team
      HAVING N >= @min_n
    )
    SELECT
      Team, N, W, L, P,
      SAFE_MULTIPLY(SAFE_DIVIDE(W, NULLIF(W + L, 0)), 100.0) AS WinPct,
      CASE
        WHEN @market = 'spreads' AND (W + L) > 0
          THEN ((W * (100.0/110.0) + L * (-1.0)) / (W + L)) * 100.0
        ELSE NULL
      END AS ROI_Pct
    FROM by_team
    ORDER BY IFNULL(ROI_Pct, 0.0) DESC, WinPct DESC, N DESC
    """
    job = CLIENT.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("sport","STRING", sport),
                bigquery.ScalarQueryParameter("cutoff","TIMESTAMP", cutoff),
                bigquery.ScalarQueryParameter("market","STRING", market),
                bigquery.ScalarQueryParameter("is_home","BOOL", is_home),
                bigquery.ScalarQueryParameter("spread_bucket","STRING", spread_bucket or ""),
                bigquery.ScalarQueryParameter("is_favorite","BOOL", is_favorite),
                bigquery.ScalarQueryParameter("min_n","INT64", min_n),
            ],
            use_query_cache=True,
        ),
    )
    return job.result().to_dataframe()

# --- display helpers -------------------------------------------------------

def bullet(team: str, market: str, r: pd.Series) -> str:
    W = int(r.W) if pd.notnull(r.W) else 0
    L = int(r.L) if pd.notnull(r.L) else 0
    P = int(r.P) if ('P' in r.index and pd.notnull(r.P)) else 0
    N = int(r.N) if pd.notnull(r.N) else 0
    winpct = float(r.WinPct) if pd.notnull(r.WinPct) else 0.0
    roi_val = r.get("ROI_Pct")
    roi_txt = f", ROI {float(roi_val):+,.1f}%" if pd.notnull(roi_val) else ""
    push_txt = f"-{P}P" if P else ""
    situation = r.get("Situation", "Situation")
    return (f"**{team} — {market.capitalize()}** · {situation}: "
            f"{W}-{L}{push_txt} ({winpct:.1f}%) over {N}{roi_txt}.")

def bullet_with_baseline(team, market, r, base):
    line = bullet(team, market, r)
    if base is None:
        return "• " + line
    base_WP = float(base.WinPct) if pd.notnull(base.WinPct) else 0.0
    base_N  = int(base.N) if pd.notnull(base.N) else 0
    return "• " + line + f" — League baseline: {base_WP:.1f}% (N={base_N})"

def _wanted_situations(is_home: bool | None, is_favorite: bool | None, spread_bucket: str | None):
    """Ordered list of situation names to display for the *current* role."""
    out = []
    # Specific combos first
    if is_home is True  and is_favorite is True:   out.append("Home Favorite")
    if is_home is True  and is_favorite is False:  out.append("Home Underdog")
    if is_home is False and is_favorite is True:   out.append("Road Favorite")
    if is_home is False and is_favorite is False:  out.append("Road Underdog")
    # Individual facets
    if is_home is True:      out.append("Home")
    if is_home is False:     out.append("Road")
    if is_favorite is True:  out.append("Favorite")
    if is_favorite is False: out.append("Underdog")
    # Bucket
    if spread_bucket: out.append(spread_bucket)
    # De-dupe preserve order
    seen, keep = set(), []
    for s in out:
        if s and s not in seen:
            keep.append(s); seen.add(s)
    return keep

def _print_matching_situations(team, market, df, wanted_order, baseline_row):
    """Print only the situations we care about; fallback to top 3 if none match."""
    if df.empty:
        st.write("_No situations meeting N threshold._")
        return
    by_name = {row.Situation: row for _, row in df.iterrows() if "Situation" in row.index}
    printed = 0
    for name in wanted_order:
        r = by_name.get(name)
        if r is None:
            continue
        st.markdown(bullet_with_baseline(team, market, r, baseline_row))
        printed += 1
        if printed >= 3:
            break
    if printed == 0:
        for _, r in df.head(3).iterrows():
            st.markdown(bullet_with_baseline(team, market, r, baseline_row))

# --- main UI ---------------------------------------------------------------

def render_situation_db_tab(selected_sport: str | None = None):
    st.header("📊 Situation Database — Best % by Team")

    if not selected_sport:
        st.warning("Please pick a sport in the sidebar.")
        st.stop()

    df_games = list_games_cached(selected_sport)
    if df_games.empty:
        st.info("No upcoming games found for this sport.")
        return

    df_games = df_games.copy()
    df_games["TeamsList"] = df_games["Teams"].apply(lambda x: tuple(_coerce_teams_list(x)))
    df_games["label"] = df_games.apply(
        lambda r: f"{r['Game_Start']} — {', '.join(map(str, r['TeamsList']))}"
                  if r["TeamsList"] else f"{r['Game_Start']} — (teams TBD)",
        axis=1
    )

    row = st.selectbox("Select game", df_games.to_dict("records"), format_func=lambda r: r["label"])
    if not row:
        st.stop()

    game_id    = row["Game_Id"]
    game_start = _coerce_timestamp(row["Game_Start"])  # may be None; we’ll handle
    teams      = row["TeamsList"]

    min_n = st.slider("Min graded games per situation", 10, 200, 30, step=5)
    baseline_min_n = st.number_input("League baseline min N", 50, 1000, 150, step=10)

    # Prefetch contexts for both teams
    contexts = get_contexts_for_game(game_id, teams[:2])

    cols = st.columns(2)
    for i, team in enumerate(teams[:2]):
        with cols[i]:
            st.subheader(team)

            ctx = contexts.get(team) or get_team_context(game_id, team)
            role_bits = []
            role_bits.append("🏠 Home" if ctx.get("is_home") is True else ("🚗 Road" if ctx.get("is_home") is False else ""))
            role_bits.append("⭐ Favorite" if ctx.get("is_favorite") is True else ("🐶 Underdog" if ctx.get("is_favorite") is False else ""))
            if ctx.get("spread_bucket"): role_bits.append(ctx["spread_bucket"])
            st.caption(" / ".join([b for b in role_bits if b]) or "Role: Unknown")

            sport_str     = selected_sport
            is_home       = ctx.get("is_home")
            spread_bucket = ctx.get("spread_bucket")

            # ✅ SAFE CUTOFF: prefer context cutoff; else game_start; else now (never None)
            cutoff_input = st.date_input("Cutoff (optional)", value=None)
            #cutoff = _safe_cutoff(ctx.get("cutoff"), game_start) or _fallback_now_utc()

            # Spreads (current role)
            st.markdown("**Spreads — current role**")
            s_ats = situation_stats_cached(sport_str, team, cutoff_input, "spreads", min_n)
            if s_ats.empty:
                st.write("_No ATS situations meeting N threshold._")
            else:
                base = league_baseline_filtered_cached(sport_str, cutoff, "spreads", is_home, spread_bucket, baseline_min_n)
                wanted = _wanted_situations(is_home, ctx.get("is_favorite"), spread_bucket)
                _print_matching_situations(team, "spreads", s_ats, wanted, base)

            # Moneyline (current role)
            st.markdown("**Moneyline — current role**")
            s_ml = situation_stats_cached(sport_str, team, cutoff, "moneyline", min_n)
            if s_ml.empty:
                st.write("_No ML situations meeting N threshold._")
            else:
                base_ml = league_baseline_filtered_cached(sport_str, cutoff, "moneyline", is_home, spread_bucket, baseline_min_n)
                wanted_ml = _wanted_situations(is_home, ctx.get("is_favorite"), spread_bucket)
                _print_matching_situations(team, "moneyline", s_ml, wanted_ml, base_ml)

            # League leaderboard for this exact role (spreads)
            st.markdown("**League — this exact role**")
            rb = role_leaderboard_cached(sport_str, cutoff, "spreads", is_home, spread_bucket, ctx.get("is_favorite"), baseline_min_n)
            if rb.empty:
                st.write("_No teams meet N threshold in this role._")
            else:
                st.dataframe(rb.head(20))
