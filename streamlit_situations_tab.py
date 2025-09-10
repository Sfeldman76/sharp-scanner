# streamlit_situations_tab.py
import streamlit as st
import pandas as pd
from google.cloud import bigquery

PROJECT = "sharplogger"
DATASET = "sharp_data"
VIEW_GAMES = f"`{PROJECT}.{DATASET}.scores_games_list`"
VIEW_FEAT  = f"`{PROJECT}.{DATASET}.scores_with_features`"
CLIENT = bigquery.Client(project=PROJECT)

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

def bullet(team: str, market: str, r: pd.Series) -> str:
    roi = (f", ROI {r.ROI_Pct:+.1f}%" if pd.notnull(r.ROI_Pct) else "")
    return (f"**{team} — {market.capitalize()}** · {r.Situation}: "
            f"{int(r.W)}-{int(r.L)}{('-'+str(int(r.P))+'P' if r.P else '')} "
            f"({float(r.WinPct):.1f}%) over {int(r.N)}{roi}.")

def render_situation_db_tab():
    st.header("Situation Database — Best % by Team (from scores_with_features)")

    df_games = list_games()
    if df_games.empty:
        st.info("No games found in window.")
        return

    df_games["label"] = df_games.apply(
        lambda r: f"{r['Game_Start']} — {','.join(r['Teams'])}", axis=1)
    row = st.selectbox("Select game", df_games.to_dict("records"),
                       format_func=lambda r: r["label"])
    game_id, game_start, teams = row["Game_Id"], row["Game_Start"], row["Teams"]

    min_n = st.slider("Min graded games per situation", 10, 200, 30, step=5)

    # For each team, pull ATS & ML top-3
    cols = st.columns(2)
    for i, team in enumerate(teams):
        with cols[i]:
            st.subheader(team)
            # ATS (spreads)
            s_ats = situation_stats(team, game_start, "spreads", min_n)
            st.markdown("**Spreads — top 3**")
            if s_ats.empty:
                st.write("_No ATS situations meeting N threshold._")
            else:
                for _, r in s_ats.head(3).iterrows():
                    st.markdown("• " + bullet(team, "spreads", r))

            # ML (percent only unless you add ML odds)
            s_ml = situation_stats(team, game_start, "moneyline", min_n)
            st.markdown("**Moneyline — top 3**")
            if s_ml.empty:
                st.write("_No ML situations meeting N threshold._")
            else:
                for _, r in s_ml.head(3).iterrows():
                    st.markdown("• " + bullet(team, "moneyline", r))

# In your app:
# render_situation_db_tab()
