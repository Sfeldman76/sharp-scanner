import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, date, time, timezone, timedelta
from google.cloud import bigquery

# --- constants / config ----------------------------------------------------
PROJECT = "sharplogger"
DATASET = "sharp_data"

VIEW_GAMES_FQ       = f"{PROJECT}.{DATASET}.scores_games_list"
SCORES_ROLE_VIEW_FQ = f"{PROJECT}.{DATASET}.scores_role_view"
MOVES_TABLE_FQ      = f"{PROJECT}.{DATASET}.moves_with_features_merged"

# Backticked for SQL
VIEW_GAMES       = f"`{VIEW_GAMES_FQ}`"
SCORES_ROLE_VIEW = f"`{SCORES_ROLE_VIEW_FQ}`"
MOVES_TABLE      = f"`{MOVES_TABLE_FQ}`"

# Schema-aligned columns
HOME_COL   = "Is_Home"   # BOOL in moves/scores_role_view
SPREAD_COL = "Value"     # closing spread for the team (fav < 0)

CLIENT = bigquery.Client(project=PROJECT)
CACHE_TTL_SEC = 90

# --- small utils -----------------------------------------------------------
def _coerce_timestamp(x) -> dt.datetime | None:
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        if pd.isna(x):
            return None
        ts = x
    elif isinstance(x, dt.datetime):
        ts = pd.Timestamp(x)
    elif isinstance(x, dt.date):
        ts = pd.Timestamp(dt.datetime.combine(x, dt.time(0, 0, 0)))
    else:
        try:
            ts = pd.to_datetime(x)
            if pd.isna(ts):
                return None
        except Exception:
            return None

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
    a = _coerce_timestamp(primary)
    b = _coerce_timestamp(fallback)
    return a or b

def _coerce_teams_list(x) -> list[str]:
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
def _get_view_columns_via_api(bq: bigquery.Client, fq_table: str) -> set[str]:
    tbl = bq.get_table(fq_table)
    return {field.name for field in tbl.schema}

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
def get_contexts_for_game(game_id: str, teams: tuple[str, ...] | list[str]) -> dict:
    teams = list(teams or [])
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
                bigquery.ArrayQueryParameter("teams", "STRING", teams),
            ],
            use_query_cache=True,
        ),
    )
    df = job.result().to_dataframe()

    out = {}
    for _, r in df.iterrows():
        sp = r.get("closing_spread")
        spf = float(sp) if pd.notnull(sp) else None
        out[str(r["feat_Team"])] = {
            "is_home": bool(r["is_home"]) if pd.notnull(r["is_home"]) else None,
            "is_favorite": (spf is not None and spf < 0),
            "spread_bucket": _spread_bucket(spf),
            "cutoff": _coerce_timestamp(r.get("cutoff")),
            "source": r.get("source"),
        }
    return out

def get_team_context(game_id: str, team: str) -> dict:
    ctxs = get_contexts_for_game(game_id, (team,))
    return ctxs.get(team, {"is_home": None, "is_favorite": None, "spread_bucket": "", "cutoff": None, "source": "none"})

# ---------- Situations queries that match your schema ----------
@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def situation_stats_cached(sport_str: str, team: str, cutoff_date: date | None,
                           market: str, min_n: int) -> pd.DataFrame:
    """
    Per-team situation stats (DATE-based cutoff), grouped by (Team, Situation).
    - Spreads: uses Spread_Cover_Flag, excludes pushes from denom
    - Moneyline: uses (Team_Score > Opp_Score)
    """
    cutoff_d = cutoff_date or date.today()
    params = [
        bigquery.ScalarQueryParameter("sport", "STRING", (sport_str or "").upper()),
        bigquery.ScalarQueryParameter("cutoff", "DATE", cutoff_d),
        bigquery.ScalarQueryParameter("min_n", "INT64", int(min_n)),
    ]

    if market.lower() == "spreads":
        sql = f"""
        WITH base AS (
          SELECT
            feat_Team AS Team,
            Is_Home,
            Value     AS Closing_Spread,
            Spread_Cover_Flag,
            ATS_Cover_Margin,
            feat_Game_Start
          FROM {SCORES_ROLE_VIEW}
          WHERE Sport = @sport
            AND DATE(feat_Game_Start) <= @cutoff
        ),
        enriched AS (
          SELECT
            *,
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
            END AS Spread_Bucket
          FROM base
        ),
        situations AS (
          SELECT
            Team,
            s AS Situation,
            Spread_Cover_Flag,
            ATS_Cover_Margin,
            Closing_Spread
          FROM enriched,
          UNNEST(ARRAY(
            SELECT x FROM UNNEST([
              CASE WHEN Is_Home AND Closing_Spread < 0 THEN 'Home Favorite' END,
              CASE WHEN Is_Home AND (Closing_Spread >= 0 OR Closing_Spread IS NULL) THEN 'Home Underdog' END,
              CASE WHEN NOT Is_Home AND Closing_Spread < 0 THEN 'Road Favorite' END,
              CASE WHEN NOT Is_Home AND (Closing_Spread >= 0 OR Closing_Spread IS NULL) THEN 'Road Underdog' END,
              CASE WHEN Is_Home THEN 'Home' END,
              CASE WHEN NOT Is_Home THEN 'Road' END,
              CASE WHEN Closing_Spread < 0 THEN 'Favorite' END,
              CASE WHEN Closing_Spread >= 0 THEN 'Underdog' END,
              Spread_Bucket
            ]) AS x
            WHERE x IS NOT NULL AND x != ''
          )) AS s
        )
        SELECT
          Team,
          Situation,
          COUNT(*) AS N,
          COUNTIF(Spread_Cover_Flag IN (0,1)) AS Decisions,
          COUNTIF(Spread_Cover_Flag = 1) AS W,
          COUNTIF(Spread_Cover_Flag = 0) AS L,
          COUNTIF(ATS_Cover_Margin = 0) AS P,
          SAFE_DIVIDE(COUNTIF(Spread_Cover_Flag = 1),
                      NULLIF(COUNTIF(Spread_Cover_Flag IN (0,1)), 0)) AS Win_Rate,
          AVG(Closing_Spread) AS Avg_Spread
        FROM situations
        GROUP BY Team, Situation
        HAVING N >= @min_n
        ORDER BY N DESC, Win_Rate DESC
        """
    else:
        sql = f"""
        WITH base AS (
          SELECT
            feat_Team AS Team,
            Is_Home,
            Value     AS Closing_Spread,
            Team_Score,
            Opp_Score,
            feat_Game_Start
          FROM {SCORES_ROLE_VIEW}
          WHERE Sport = @sport
            AND DATE(feat_Game_Start) <= @cutoff
        ),
        enriched AS (
          SELECT
            *,
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
            CAST(Team_Score > Opp_Score AS INT64) AS ML_Win
          FROM base
        ),
        situations AS (
          SELECT
            Team,
            s AS Situation,
            ML_Win
          FROM enriched,
          UNNEST(ARRAY(
            SELECT x FROM UNNEST([
              CASE WHEN Is_Home AND Closing_Spread < 0 THEN 'Home Favorite' END,
              CASE WHEN Is_Home AND (Closing_Spread >= 0 OR Closing_Spread IS NULL) THEN 'Home Underdog' END,
              CASE WHEN NOT Is_Home AND Closing_Spread < 0 THEN 'Road Favorite' END,
              CASE WHEN NOT Is_Home AND (Closing_Spread >= 0 OR Closing_Spread IS NULL) THEN 'Road Underdog' END,
              CASE WHEN Is_Home THEN 'Home' END,
              CASE WHEN NOT Is_Home THEN 'Road' END,
              CASE WHEN Closing_Spread < 0 THEN 'Favorite' END,
              CASE WHEN Closing_Spread >= 0 THEN 'Underdog' END,
              Spread_Bucket
            ]) AS x
            WHERE x IS NOT NULL AND x != ''
          )) AS s
        )
        SELECT
          Team,
          Situation,
          COUNT(*) AS N,
          AVG(ML_Win) AS Win_Rate
        FROM situations
        GROUP BY Team, Situation
        HAVING N >= @min_n
        ORDER BY N DESC, Win_Rate DESC
        """
    try:
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        return CLIENT.query(sql, job_config=job_config).to_dataframe()
    except Exception as e:
        st.error(f"❌ situation_stats_cached failed: {e}")
        cols = ["Team", "Situation", "N", "Win_Rate"]
        if market.lower() == "spreads":
            cols += ["Decisions", "W", "L", "P", "Avg_Spread"]
        return pd.DataFrame(columns=cols)

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
    cutoff = _coerce_timestamp(cutoff)
    if cutoff is None or (isinstance(cutoff, pd.Timestamp) and pd.isna(cutoff)):
        cutoff = _fallback_now_utc()
    cutoff = _round_cutoff(cutoff) or _fallback_now_utc()
    if isinstance(cutoff, pd.Timestamp):
        cutoff = cutoff.to_pydatetime()

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
    W = int(r.W) if 'W' in r.index and pd.notnull(r.W) else 0
    L = int(r.L) if 'L' in r.index and pd.notnull(r.L) else 0
    P = int(r.P) if 'P' in r.index and pd.notnull(r.P) else 0
    N = int(r.N) if 'N' in r.index and pd.notnull(r.N) else 0
    winpct = float(r.Win_Rate if 'Win_Rate' in r.index else r.WinPct) if pd.notnull(r.get('Win_Rate', r.get('WinPct', np.nan))) else 0.0
    roi_val = r.get("ROI_Pct")
    roi_txt = f", ROI {float(roi_val):+,.1f}%" if pd.notnull(roi_val) else ""
    push_txt = f"-{P}P" if P else ""
    situation = r.get("Situation", "Situation")
    return (f"**{team} — {market.capitalize()}** · {situation}: "
            f"{W}-{L}{push_txt} ({winpct*100:.1f}%) over {N}{roi_txt}.")

def bullet_with_baseline(team, market, r, base):
    line = bullet(team, market, r)
    if base is None:
        return "• " + line
    base_WP = float(base.WinPct) if pd.notnull(base.WinPct) else 0.0
    base_N  = int(base.N) if pd.notnull(base.N) else 0
    return "• " + line + f" — League baseline: {base_WP:.1f}% (N={base_N})"

def _wanted_situations(is_home: bool | None, is_favorite: bool | None, spread_bucket: str | None):
    out = []
    if is_home is True  and is_favorite is True:   out.append("Home Favorite")
    if is_home is True  and is_favorite is False:  out.append("Home Underdog")
    if is_home is False and is_favorite is True:   out.append("Road Favorite")
    if is_home is False and is_favorite is False:  out.append("Road Underdog")
    if is_home is True:      out.append("Home")
    if is_home is False:     out.append("Road")
    if is_favorite is True:  out.append("Favorite")
    if is_favorite is False: out.append("Underdog")
    if spread_bucket: out.append(spread_bucket)
    seen, keep = set(), []
    for s in out:
        if s and s not in seen:
            keep.append(s); seen.add(s)
    return keep

def _print_matching_situations(team, market, df, wanted_order, baseline_row):
    if df.empty:
        st.write("_No situations meeting N threshold._")
        return
    # df here is ALREADY filtered to the selected team
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
    df_games["TeamsList"] = df_games["Teams"].apply(_coerce_teams_list)
    df_games["TeamsList"] = df_games["TeamsList"].apply(lambda xs: tuple(xs)[:2])  # max two
    df_games["label"] = df_games.apply(
        lambda r: f"{r['Game_Start']} — {', '.join(map(str, r['TeamsList']))}" if r["TeamsList"]
                  else f"{r['Game_Start']} — (teams TBD)",
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
    contexts = get_contexts_for_game(game_id, teams)

    cols = st.columns(2)
    for i, team in enumerate(teams):
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

            # Cutoffs: DATE for per-team stats; TIMESTAMP for league/leaderboard
            cutoff_input: date = st.date_input("Cutoff (DATE for stats)", value=date.today(), key=f"cutoff_{team}")
            cutoff_date_for_stats = cutoff_input or date.today()

            cutoff_ts_context = _safe_cutoff(ctx.get("cutoff"), game_start) or _fallback_now_utc()
            cutoff_ts = _round_cutoff(cutoff_ts_context) or _fallback_now_utc()
            if isinstance(cutoff_ts, pd.Timestamp):
                cutoff_ts = cutoff_ts.to_pydatetime()

            # Spreads (current role) — DATE cutoff
            st.markdown("**Spreads — current role**")
            s_ats_all = situation_stats_cached(sport_str, team, cutoff_date_for_stats, "spreads", min_n)
            s_ats = s_ats_all[s_ats_all["Team"] == team] if not s_ats_all.empty else s_ats_all
            if s_ats.empty:
                st.write("_No ATS situations meeting N threshold._")
            else:
                base = league_baseline_filtered_cached(sport_str, cutoff_ts, "spreads", is_home, spread_bucket, baseline_min_n)
                wanted = _wanted_situations(is_home, ctx.get("is_favorite"), spread_bucket)
                _print_matching_situations(team, "spreads", s_ats, wanted, base)

            # Moneyline (current role) — DATE cutoff
            st.markdown("**Moneyline — current role**")
            s_ml_all = situation_stats_cached(sport_str, team, cutoff_date_for_stats, "moneyline", min_n)
            s_ml = s_ml_all[s_ml_all["Team"] == team] if not s_ml_all.empty else s_ml_all
            if s_ml.empty:
                st.write("_No ML situations meeting N threshold._")
            else:
                base_ml = league_baseline_filtered_cached(sport_str, cutoff_ts, "moneyline", is_home, spread_bucket, baseline_min_n)
                wanted_ml = _wanted_situations(is_home, ctx.get("is_favorite"), spread_bucket)
                _print_matching_situations(team, "moneyline", s_ml, wanted_ml, base_ml)

            # League leaderboard for this exact role (spreads) — TIMESTAMP cutoff
            st.markdown("**League — this exact role**")
            rb = role_leaderboard_cached(sport_str, cutoff_ts, "spreads", is_home, spread_bucket, ctx.get("is_favorite"), baseline_min_n)
            if rb.empty:
                st.write("_No teams meet N threshold in this role._")
            else:
                st.dataframe(rb.head(20))

