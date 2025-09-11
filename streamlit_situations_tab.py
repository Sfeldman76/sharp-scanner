# streamlit_situations_tab.py
import streamlit as st
import pandas as pd
from google.cloud import bigquery
import numpy as np
PROJECT = "sharplogger"
DATASET = "sharp_data"
VIEW_GAMES = f"`{PROJECT}.{DATASET}.scores_games_list`"
VIEW_FEAT  = f"`{PROJECT}.{DATASET}.scores_with_features`"
CLIENT = bigquery.Client(project=PROJECT)
HOME_COL   = "Is_Home"
SPREAD_COL = "Value"  # used by moves and scores_role_view
SCORES_ROLE_VIEW = "`sharplogger.sharp_data.scores_role_view`"
MOVES_TABLE      = "`sharplogger.sharp_data.moves_with_features_merged`"

# --- helpers ---------------------------------------------------------------

def list_games(sport_label: str) -> pd.DataFrame:
    # Map sidebar label to DB value if needed
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
            query_parameters=[bigquery.ScalarQueryParameter("sport","STRING", db_sport)]
        ),
    )
    return job.result().to_dataframe()

def team_rows(game_id: str) -> pd.DataFrame:
    sql = f"""
    SELECT feat_Team AS Team, ANY_VALUE(feat_Game_Start) AS Cutoff
    FROM {VIEW_FEAT}
    WHERE feat_Game_Key = @gid
    GROUP BY feat_Team
    """
    job = CLIENT.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("gid","STRING",game_id)]
    ))
    return job.result().to_dataframe()

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

def get_team_context(game_id: str, team: str) -> dict:
    # 1) moves (preferred)
    sql_moves = f"""
    SELECT
      ANY_VALUE({HOME_COL})         AS is_home,
      ANY_VALUE({SPREAD_COL})       AS closing_spread,
      ANY_VALUE(feat_Game_Start)    AS game_start
    FROM {MOVES_TABLE}
    WHERE feat_Game_Key = @gid AND feat_Team = @team
    """
    job = CLIENT.query(sql_moves, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("gid","STRING", game_id),
            bigquery.ScalarQueryParameter("team","STRING", team),
        ]))
    df = job.result().to_dataframe()
    if not df.empty:
        r = df.iloc[0]
        sp = r.get("closing_spread")
        spf = float(sp) if pd.notnull(sp) else None
        return {
            "is_home": bool(r["is_home"]) if pd.notnull(r["is_home"]) else None,
            "is_favorite": (spf is not None and spf < 0),
            "spread_bucket": _spread_bucket(spf),
            "cutoff": r.get("game_start"),
            "source": "moves",
        }

    # 2) fallback: scores_role_view
    sql_scores = f"""
    SELECT
      ANY_VALUE({HOME_COL})         AS is_home,
      ANY_VALUE({SPREAD_COL})       AS closing_spread,
      ANY_VALUE(feat_Game_Start)    AS cutoff
    FROM {SCORES_ROLE_VIEW}
    WHERE feat_Game_Key = @gid AND feat_Team = @team
    """
    job2 = CLIENT.query(sql_scores, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("gid","STRING", game_id),
            bigquery.ScalarQueryParameter("team","STRING", team),
        ]))
    df2 = job2.result().to_dataframe()
    if df2.empty:
        return {"is_home": None, "is_favorite": None, "spread_bucket": "", "cutoff": None, "source": "none"}

    r2 = df2.iloc[0]
    sp2 = r2.get("closing_spread")
    sp2f = float(sp2) if pd.notnull(sp2) else None
    return {
        "is_home": bool(r2["is_home"]) if pd.notnull(r2["is_home"]) else None,
        "is_favorite": (sp2f is not None and sp2f < 0),
        "spread_bucket": _spread_bucket(sp2f),
        "cutoff": r2.get("cutoff"),
        "source": "scores",
    }



def _coerce_teams_list(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, pd.Series):
        return x.astype(str).tolist()
    if isinstance(x, str):
        # split on comma if present; otherwise treat as one team
        parts = [p.strip() for p in x.split(",")] if "," in x else [x.strip()]
        # filter out accidental empties
        return [p for p in parts if p]
    if hasattr(x, "tolist"):
        try:
            out = x.tolist()
            if isinstance(out, (list, tuple, np.ndarray, pd.Series)):
                return [str(v) for v in list(out)]
            return [str(out)]
        except Exception:
            pass
    try:
        return [str(v) for v in list(x)]
    except Exception:
        return [str(x)]


# --- table-function wrappers ----------------------------------------------

def situation_stats(sport: str, team: str, cutoff, market: str, min_n: int) -> pd.DataFrame:
    # matches: (p_sport, p_team, p_cutoff, p_market, p_min_n)
    sql = f"""
    SELECT * FROM `{PROJECT}.{DATASET}.situation_stats_from_scores`
      (@sport, @team, @cutoff, @market, @min_n)
    """
    job = CLIENT.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("sport","STRING",sport),
            bigquery.ScalarQueryParameter("team","STRING",team),
            bigquery.ScalarQueryParameter("cutoff","TIMESTAMP",cutoff),
            bigquery.ScalarQueryParameter("market","STRING",market),
            bigquery.ScalarQueryParameter("min_n","INT64",min_n),
        ]
    ))
    return job.result().to_dataframe()

def league_baseline_filtered(sport: str, cutoff, market: str,
                             is_home: bool | None,
                             spread_bucket: str | None,
                             min_n: int = 100):
    sql = """
    SELECT * FROM `sharplogger.sharp_data.league_situation_stats_from_scores`
      (@sport, @cutoff, @market, @is_home, @spread_bucket, @min_n)
    """
    job = CLIENT.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("sport","STRING", sport),
        bigquery.ScalarQueryParameter("cutoff","TIMESTAMP", cutoff),
        bigquery.ScalarQueryParameter("market","STRING", market),
        bigquery.ScalarQueryParameter("is_home","BOOL", is_home),
        bigquery.ScalarQueryParameter("spread_bucket","STRING", spread_bucket or ""),
        bigquery.ScalarQueryParameter("min_n","INT64", min_n),
    ]))
    df = job.result().to_dataframe()
    return None if df.empty else df.iloc[0]

def role_leaderboard(sport: str, cutoff, market: str,
                     is_home: bool | None,
                     spread_bucket: str | None,
                     min_n: int = 30) -> pd.DataFrame:
    sql = f"""
    WITH src AS (
      SELECT
        feat_Team AS Team,
        {HOME_COL}   AS Is_Home,
        {SPREAD_COL} AS Closing_Spread,
        Spread_Cover_Flag, ATS_Cover_Margin,
        Team_Score, Opp_Score, feat_Game_Start
      FROM {SCORES_ROLE_VIEW}
      WHERE Sport = @sport
        AND feat_Game_Start < @cutoff
        AND feat_Game_Start >= TIMESTAMP(DATETIME_SUB(DATETIME(@cutoff), INTERVAL 10 YEAR))
    ),
    e AS (
      SELECT
        Team,
        Is_Home,
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
    job = CLIENT.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("sport","STRING", sport),
        bigquery.ScalarQueryParameter("cutoff","TIMESTAMP", cutoff),
        bigquery.ScalarQueryParameter("market","STRING", market),
        bigquery.ScalarQueryParameter("is_home","BOOL", is_home),
        bigquery.ScalarQueryParameter("spread_bucket","STRING", spread_bucket or ""),
        bigquery.ScalarQueryParameter("min_n","INT64", min_n),
    ]))
    return job.result().to_dataframe()


# --- formatting ------------------------------------------------------------

def bullet(team: str, market: str, r: pd.Series) -> str:
    W = int(r.W) if pd.notnull(r.W) else 0
    L = int(r.L) if pd.notnull(r.L) else 0
    P = int(r.P) if ('P' in r.index and pd.notnull(r.P)) else 0
    N = int(r.N) if pd.notnull(r.N) else 0
    winpct = float(r.WinPct) if pd.notnull(r.WinPct) else 0.0
    roi_val = r.get('ROI_Pct')
    roi_txt = f", ROI {float(roi_val):+,.1f}%" if pd.notnull(roi_val) else ""
    push_txt = f"-{P}P" if P else ""
    return (f"**{team} — {market.capitalize()}** · {r.Situation}: "
            f"{W}-{L}{push_txt} ({winpct:.1f}%) over {N}{roi_txt}.")


def bullet_with_baseline(team, market, r, base):
    line = bullet(team, market, r)
    if base is None:
        return "• " + line
    base_WP = float(base.WinPct) if pd.notnull(base.WinPct) else 0.0
    base_N  = int(base.N) if pd.notnull(base.N) else 0
    return "• " + line + f" — League baseline: {base_WP:.1f}% (N={base_N})"

# --- UI --------------------------------------------------------------------

def render_situation_db_tab(selected_sport: str | None = None):
    st.header("📊 Situation Database — Best % by Team (from scores_with_features)")

    if not selected_sport:
        st.warning("Please pick a sport in the sidebar.")
        st.stop()


    # ✅ Create df_games first
    df_games = list_games(selected_sport)
    if df_games.empty:
        st.info("No upcoming games found for this sport.")
        return

    df_games = df_games.copy()
    df_games["TeamsList"] = df_games["Teams"].apply(_coerce_teams_list)
    df_games["label"] = df_games.apply(
        lambda r: f"{r['Game_Start']} — {', '.join(map(str, r['TeamsList']))}" if r["TeamsList"] else f"{r['Game_Start']} — (teams TBD)",
        axis=1
    )

    # ✅ Let the user pick a game
    row = st.selectbox(
        "Select game",
        df_games.to_dict("records"),
        format_func=lambda r: r["label"]
    )
    if not row:
        st.stop()

    # ✅ Use TeamsList from the normalized column
    game_id, game_start, teams = row["Game_Id"], row["Game_Start"], row["TeamsList"]

    min_n = st.slider("Min graded games per situation", 10, 200, 30, step=5)
    baseline_min_n = st.number_input("League baseline min N", 50, 1000, 150, step=10)

    cols = st.columns(2)
    for i, team in enumerate(teams[:2]):  # guard if Teams has >2
        with cols[i]:
            st.subheader(team)

            # 🧠 Get context from moves/scores
            ctx = get_team_context(game_id, team)
            role_bits = []
            if ctx["is_home"] is True: role_bits.append("🏠 Home")
            elif ctx["is_home"] is False: role_bits.append("🚗 Road")
            if ctx["is_favorite"] is True: role_bits.append("⭐ Favorite")
            elif ctx["is_favorite"] is False: role_bits.append("🐶 Underdog")
            if ctx["spread_bucket"]: role_bits.append(ctx["spread_bucket"])
            st.caption(" / ".join(role_bits) or "Role: Unknown")

            # Inputs for table functions
            sport_str = selected_sport or ctx.get("sport") or ""
            is_home = ctx["is_home"]
            spread_bucket = ctx["spread_bucket"]
            cutoff = ctx["cutoff"] or game_start  # fallback

            # 🧾 Spreads
            st.markdown("**Spreads — top 3**")
            s_ats = situation_stats(sport_str, team, cutoff, "spreads", min_n)
            if s_ats.empty:
                st.write("_No ATS situations meeting N threshold._")
            else:
                base = league_baseline_filtered(
                    sport=sport_str,
                    cutoff=cutoff,
                    market="spreads",
                    is_home=is_home,
                    spread_bucket=spread_bucket,
                    min_n=baseline_min_n
                )
                for _, r in s_ats.head(3).iterrows():
                    st.markdown(bullet_with_baseline(team, "spreads", r, base))

            # 💰 Moneyline
            st.markdown("**Moneyline — top 3**")
            s_ml = situation_stats(sport_str, team, cutoff, "moneyline", min_n)
            if s_ml.empty:
                st.write("_No ML situations meeting N threshold._")
            else:
                base_ml = league_baseline_filtered(
                    sport=sport_str,
                    cutoff=cutoff,
                    market="moneyline",
                    is_home=is_home,
                    spread_bucket=spread_bucket,
                    min_n=baseline_min_n
                )
                for _, r in s_ml.head(3).iterrows():
                    st.markdown(bullet_with_baseline(team, "moneyline", r, base_ml))

            # 🏆 League leaderboard for this exact role (spreads)
            rb = role_leaderboard(
                sport_str, cutoff, "spreads",
                ctx["is_home"], ctx["spread_bucket"], baseline_min_n
            )
            st.markdown("**League — this exact role**")
            if rb.empty:
                st.write("_No teams meet N threshold in this role._")
            else:
                st.dataframe(rb.head(20))

