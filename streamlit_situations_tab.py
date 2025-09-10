# streamlit_situations_tab.py
import streamlit as st
import pandas as pd
from google.cloud import bigquery

PROJECT = "sharplogger"
DATASET = "sharp_data"
VIEW_GAMES = f"`{PROJECT}.{DATASET}.scores_games_list`"
VIEW_FEAT  = f"`{PROJECT}.{DATASET}.scores_with_features`"
CLIENT = bigquery.Client(project=PROJECT)

# --- helpers ---------------------------------------------------------------

def list_games() -> pd.DataFrame:
    sql = f"""
    SELECT Game_Id, Game_Start, Teams
    FROM {VIEW_GAMES}
    WHERE Game_Start BETWEEN TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
                        AND TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    ORDER BY Game_Start DESC
    """
    return CLIENT.query(sql).result().to_dataframe()

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

def _spread_bucket(v: float|None) -> str:
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
    if v ==   0.0: return "Pick"
    if v <=   3.5: return "Dog +0.5 to +3.5"
    if v <=   6.5: return "Dog +4 to +6.5"
    if v <=  10.5: return "Dog +7 to +10.5"
    return "Dog ≥ +11"

def game_team_context(game_id: str, team: str) -> dict:
    """
    Pull one row for this (game, team) and derive:
      - is_home (bool via your Home_/Away_ flags)
      - off_loss (bool via your *_After_*_Loss_Flag OR previous SU from scores)
      - spread_bucket (string via Closing_Spread_For_Team)
      - sport (if present; otherwise return None)
    """
    sql = f"""
    SELECT
      ANY_VALUE(Home_After_Home_Win_Flag) AS H_H_W,
      ANY_VALUE(Home_After_Home_Loss_Flag) AS H_H_L,
      ANY_VALUE(Home_After_Away_Win_Flag) AS H_A_W,
      ANY_VALUE(Home_After_Away_Loss_Flag) AS H_A_L,
      ANY_VALUE(Away_After_Home_Win_Flag) AS A_H_W,
      ANY_VALUE(Away_After_Home_Loss_Flag) AS A_H_L,
      ANY_VALUE(Away_After_Away_Win_Flag) AS A_A_W,
      ANY_VALUE(Away_After_Away_Loss_Flag) AS A_A_L,
      ANY_VALUE(Closing_Spread_For_Team) AS closing_spread,
      ANY_VALUE(feat_Game_Start) AS cutoff
      -- If your view has Sport, add: , ANY_VALUE(Sport) AS Sport
    FROM {VIEW_FEAT}
    WHERE feat_Game_Key = @gid AND feat_Team = @team
    """
    job = CLIENT.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("gid","STRING",game_id),
            bigquery.ScalarQueryParameter("team","STRING",team),
        ]
    ))
    df = job.result().to_dataframe()
    if df.empty:
        return {"is_home": None, "off_loss": None, "spread_bucket": "", "sport": None}

    r = df.iloc[0]
    # infer home
    is_home = any(pd.notnull(r[x]) for x in ["H_H_W","H_H_L","H_A_W","H_A_L"])
    away_markers = any(pd.notnull(r[x]) for x in ["A_H_W","A_H_L","A_A_W","A_A_L"])
    if is_home:
        site = True
    elif away_markers:
        site = False
    else:
        site = None

    # infer off_loss (if any “*_Loss_Flag” is present)
    off_loss = any((r.get(col) == 1) for col in ["H_H_L","H_A_L","A_H_L","A_A_L"])

    bucket = _spread_bucket(r.get("closing_spread"))

    return {
        "is_home": site,
        "off_loss": bool(off_loss) if off_loss is not None else None,
        "spread_bucket": bucket,
        "sport": None,  # fill if you add Sport column in scores_with_features
        "cutoff": r.get("cutoff"),
    }

def situation_stats(team: str, cutoff, market: str, min_n: int) -> pd.DataFrame:
    sql = f"""
    SELECT * FROM `{PROJECT}.{DATASET}.situation_stats_from_scores`(@team, @cutoff, @market, @min_n)
    """
    job = CLIENT.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("team","STRING",team),
            bigquery.ScalarQueryParameter("cutoff","TIMESTAMP",cutoff),
            bigquery.ScalarQueryParameter("market","STRING",market),
            bigquery.ScalarQueryParameter("min_n","INT64",min_n),
        ]
    ))
    return job.result().to_dataframe()

def league_baseline(sport, cutoff, market, is_home, off_loss, spread_bucket, min_n=100):
    sql = f"""
    SELECT * FROM `sharplogger.sharp_data.league_situation_stats_from_scores`
      (@sport, @cutoff, @market, @is_home, @off_loss, @spread_bucket, @min_n)
    """
    job = CLIENT.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("sport","STRING", sport),
        bigquery.ScalarQueryParameter("cutoff","TIMESTAMP", cutoff),
        bigquery.ScalarQueryParameter("market","STRING", market),
        bigquery.ScalarQueryParameter("is_home","BOOL", is_home),
        bigquery.ScalarQueryParameter("off_loss","BOOL", off_loss),
        bigquery.ScalarQueryParameter("spread_bucket","STRING", spread_bucket or ""),
        bigquery.ScalarQueryParameter("min_n","INT64", min_n),
    ]))
    df = job.result().to_dataframe()
    return None if df.empty else df.iloc[0]

def bullet(team: str, market: str, r: pd.Series) -> str:
    roi = (f", ROI {r.ROI_Pct:+.1f}%" if pd.notnull(r.ROI_Pct) else "")
    return (f"**{team} — {market.capitalize()}** · {r.Situation}: "
            f"{int(r.W)}-{int(r.L)}{('-'+str(int(r.P))+'P' if r.P else '')} "
            f"({float(r.WinPct):.1f}%) over {int(r.N)}{roi}.")

def bullet_with_baseline(team, market, r, base):
    team_line = f"**{team} — {market.capitalize()}** · {r.Situation}: {int(r.W)}-{int(r.L)}{('-'+str(int(r.P))+'P' if r.P else '')} ({float(r.WinPct):.1f}%) over {int(r.N)}"
    if base is None: 
        return "• " + team_line
    base_line = f" — League baseline: {float(base.WinPct):.1f}% (N={int(base.N)})"
    return "• " + team_line + base_line

# --- UI --------------------------------------------------------------------

def render_situation_db_tab(selected_sport: str|None = None):
    st.header("Situation Database — Best % by Team (from scores_with_features)")

    df_games = list_games()
    if df_games.empty:
        st.info("No games found in window.")
        return

    df_games["label"] = df_games.apply(
        lambda r: f"{r['Game_Start']} — {', '.join(r['Teams'])}", axis=1
    )
    row = st.selectbox(
        "Select game",
        df_games.to_dict("records"),
        format_func=lambda r: r["label"]
    )
    game_id, game_start, teams = row["Game_Id"], row["Game_Start"], row["Teams"]

    min_n = st.slider("Min graded games per situation", 10, 200, 30, step=5)
    baseline_min_n = st.number_input("League baseline min N", 50, 1000, 150, step=10)

    cols = st.columns(2)
    for i, team in enumerate(teams):
        with cols[i]:
            st.subheader(team)

            # pull current context for the baseline
            ctx = game_team_context(game_id, team)
            # if your scores_with_features has Sport, set sport_str = ctx["sport"]; else use selected_sport or None
            sport_str = selected_sport  # or ctx.get("sport")
            is_home = ctx["is_home"]
            off_loss = ctx["off_loss"]
            spread_bucket = ctx["spread_bucket"]

            # Spreads
            s_ats = situation_stats(team, game_start, "spreads", min_n)
            st.markdown("**Spreads — top 3**")
            if s_ats.empty:
                st.write("_No ATS situations meeting N threshold._")
            else:
                base = league_baseline(sport_str, game_start, "spreads", is_home, off_loss, spread_bucket, baseline_min_n)
                for _, r in s_ats.head(3).iterrows():
                    st.markdown(bullet_with_baseline(team, "spreads", r, base))

            # Moneyline (percent only unless ML odds added to view)
            s_ml = situation_stats(team, game_start, "moneyline", min_n)
            st.markdown("**Moneyline — top 3**")
            if s_ml.empty:
                st.write("_No ML situations meeting N threshold._")
            else:
                base_ml = league_baseline(sport_str, game_start, "moneyline", is_home, off_loss, spread_bucket, baseline_min_n)
                for _, r in s_ml.head(3).iterrows():
                    st.markdown(bullet_with_baseline(team, "moneyline", r, base_ml))

