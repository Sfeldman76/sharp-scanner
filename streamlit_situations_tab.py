# ================== Situations Tab (league-wide by sport; roles from MOVES) ==================
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


def _fb_bucket(v: float | None) -> str:
    """
    Local helper used only for UI labeling when we already know we're in a football sport.
    SQL uses its own CASE; this mirrors it.
    """
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


def _bb_bucket(v: float | None) -> str:
    """
    Basketball-ish buckets: slightly tighter near zero, wider tails.
    """
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


def _sport_is_football(s: str) -> bool:
    return (s or "").upper() in {"NFL", "NCAAF"}


def _sport_is_basketball(s: str) -> bool:
    return (s or "").upper() in {"NBA", "NCAAB", "WNBA"}


def _wanted_labels(is_home: bool | None, is_favorite: bool | None, bucket: str | None):
    """
    Build the ordered list of league labels we want to show for this team's current role.
    """
    out = []
    # venue only
    if is_home is True:  out.append("Home")
    if is_home is False: out.append("Road")

    # 4-way role
    if (is_home is not None) and (is_favorite is not None):
        if is_home and is_favorite:   out.append("Home Favorite")
        if is_home and not is_favorite:  out.append("Home Underdog")
        if (not is_home) and is_favorite: out.append("Road Favorite")
        if (not is_home) and (not is_favorite): out.append("Road Underdog")

    # spread bucket
    if bucket: out.append(bucket)

    # de-dupe, keep order
    seen, keep = set(), []
    for s in out:
        if s and s not in seen:
            keep.append(s); seen.add(s)
    return keep


# ---------- MOVES: find current games; infer roles ----------
@st.cache_data(ttl=90, show_spinner=False)
def list_current_games_from_moves(sport: str, hard_cap: int = 200) -> pd.DataFrame:
    """
    Returns (Game_Id, Game_Start, Teams[2]) for upcoming games in MOVES for the sport.
    """
    sql = f"""
    WITH src AS (
      SELECT
        UPPER(Sport) AS Sport_Upper,
        TIMESTAMP(COALESCE(Game_Start, Commence_Hour, feat_Game_Start)) AS gs,
        COALESCE(Home_Team_Norm, home_l) AS home_n,
        COALESCE(Away_Team_Norm, away_l) AS away_n,
        COALESCE(game_key_clean, feat_Game_Key, Game_Key) AS stable_key
      FROM {MOVES}
      WHERE UPPER(Sport) = @sport_u
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
    For the two teams in the selected game, pick one latest snapshot row (by Market_Leader, Limit, Snapshot_Timestamp)
    and extract Is_Home + Value (closing_spread proxy) to infer role/bucket.
    """
    if not teams:
        return {}

    teams_norm = [t.lower() for t in teams if t]

    sql = f"""
    WITH src AS (
      SELECT
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


# ---------- SCORES: league-wide situation totals (ATS only) ----------
@st.cache_data(ttl=CACHE_TTL_SEC, show_spinner=False)
def league_totals_spreads(sport: str, cutoff_date: date, min_n: int = 0) -> pd.DataFrame:
    """
    League-wide (sport-wide) ATS results by:
      - Home / Road
      - Home Favorite, Home Underdog, Road Favorite, Road Underdog
      - Spread buckets (football vs basketball buckets)
    """
    sql = f"""
    WITH base AS (
      SELECT
        UPPER(Sport) AS Sport_U,
        Is_Home,
        Value AS Closing_Spread,     -- spread for the team (fav < 0)
        Spread_Cover_Flag,           -- 1=cover, 0=no cover, NULL missing
        ATS_Cover_Margin,            -- 0 = push
        feat_Game_Start
      FROM {SCORES}
      WHERE UPPER(Sport) = @sport_upper
        AND DATE(feat_Game_Start) <= @cutoff
    ),

    enriched AS (
      SELECT
        Is_Home,
        Closing_Spread,
        CASE
          WHEN Closing_Spread IS NULL THEN NULL
          WHEN Closing_Spread < 0 THEN TRUE ELSE FALSE
        END AS Is_Favorite,

        -- sport-aware spread buckets
        CASE
          WHEN UPPER(@sport_upper) IN ('NFL','NCAAF') THEN
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
            -- default to football buckets
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

    -- Spread buckets
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

    unioned AS (
      SELECT * FROM venue
      UNION ALL SELECT * FROM role4
      UNION ALL SELECT * FROM buckets
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


# ---------- render helpers ----------
def _derive_bucket_for_ui(sport: str, closing_spread: float | None) -> str:
    if closing_spread is None:
        return ""
    if _sport_is_basketball(sport):
        return _bb_bucket(closing_spread)
    # default to football buckets for all others
    return _fb_bucket(closing_spread)


def _pick_labels_for_team(sport: str, ctx: dict) -> list[str]:
    is_home = ctx.get("is_home")
    is_favorite = ctx.get("is_favorite")
    spread = ctx.get("closing_spread")
    bucket = _derive_bucket_for_ui(sport, spread)
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
        st.info("No upcoming games found for this sport (from MOVES).")
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

    # Fetch league-wide totals once
    league_df = league_totals_spreads(selected_sport, cutoff_date_for_stats, min_n)

    # Roles from MOVES (to decide which rows to show)
    ctxs = team_context_from_moves(game_id, teams)
    ctxs = _enforce_role_coherence(ctxs, teams)

    # Build UI: one column per team, each shows league totals for that team's current role(s)
    cols = st.columns(2)
    for i, team in enumerate(teams):
        with cols[i]:
            st.markdown(f"### {team}")

            ctx = ctxs.get(team, {})
            is_home = ctx.get("is_home")
            is_favorite = ctx.get("is_favorite")
            spread = ctx.get("closing_spread")
            bucket = _derive_bucket_for_ui(selected_sport, spread)

            bits = []
            bits.append("🏠 Home" if is_home is True else ("🚗 Road" if is_home is False else ""))
            bits.append("⭐ Favorite" if is_favorite is True else ("🐶 Underdog" if is_favorite is False else ""))
            if bucket: bits.append(bucket)
            st.caption(" / ".join([b for b in bits if b]) or "Role: Unknown")

            labels = _pick_labels_for_team(selected_sport, ctx)
            view = _filter_rows(league_df, labels)

            if view.empty:
                st.write("_No league rows meet N threshold for this role/bucket._")
            else:
                # Friendly formatting
                show = view.copy()
                for c in ["WinPct","ROI_Pct"]:
                    if c in show.columns:
                        show[c] = show[c].map(lambda x: None if pd.isna(x) else round(float(x), 1))
                st.dataframe(
                    show[["GroupLabel","Situation","N","W","L","P","WinPct","ROI_Pct"]],
                    use_container_width=True
                )

    with st.expander("🔎 League — Full table for this sport"):
        if league_df.empty:
            st.write("_Empty for this sport and cutoff / N filter._")
        else:
            show = league_df.copy()
            for c in ["WinPct","ROI_Pct"]:
                if c in show.columns:
                    show[c] = show[c].map(lambda x: None if pd.isna(x) else round(float(x), 1))
            st.dataframe(show, use_container_width=True)


# ---------- optional small section for quick game debug ----------
def render_current_games_section(selected_sport: str):
    st.subheader("📡 Current/Upcoming Games (from MOVES)")
    games = list_current_games_from_moves(selected_sport)
    if games.empty:
        st.info("No upcoming games found for this sport (from MOVES).")
        with st.expander("Debug this sport in MOVES"):
            dbg_sql = f"""
            SELECT
              UPPER(Sport) AS Sport_Upper,
              COALESCE(game_key_clean, feat_Game_Key, Game_Key) AS Game_Id,
              COALESCE(Game_Start, Commence_Hour, feat_Game_Start) AS Game_Start,
              Home_Team_Norm, Away_Team_Norm, Team_For_Join, feat_Team, home_l, away_l
            FROM {MOVES}
            WHERE UPPER(Sport) = @sport_upper
            ORDER BY Game_Start DESC
            LIMIT 50
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
