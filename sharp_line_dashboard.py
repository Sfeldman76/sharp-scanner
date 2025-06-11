import streamlit as st
from streamlit_autorefresh import st_autorefresh

# === Page Config ===
st.set_page_config(layout="wide")
st.title("Scott's Sharp Edge Scanner")

# === Auto-refresh every 180 seconds ===
st_autorefresh(interval=180 * 1000, key="data_refresh")

# === Standard Imports ===
import os
import json
import pickle
import pytz
import time
import requests
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pytz import timezone as pytz_timezone
from google.oauth2 import service_account
import pandas_gbq
import pandas as pd
import pandas_gbq  # ✅ Required for setting .context.project / .context.credentials
from google.cloud import bigquery
from google.cloud import storage


from google.cloud import bigquery
from pandas_gbq import to_gbq
import pandas as pd


GCP_PROJECT_ID = "sharplogger"  # ✅ confirmed project ID
BQ_DATASET = "sharp_data"       # ✅ your dataset name
BQ_TABLE = "sharp_moves_master" # ✅ your table name
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
MARKET_WEIGHTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.market_weights"
LINE_HISTORY_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.line_history_master"
SNAPSHOTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.odds_snapshot_log"
GCS_BUCKET = "sharp-models"
import os, json
if os.path.exists(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]):
    with open(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]) as f:
        info = json.load(f)
        st.write("🔍 Credential file type:", info.get("type"))



pandas_gbq.context.project = GCP_PROJECT_ID  # credentials will be inferred

bq_client = bigquery.Client(project=GCP_PROJECT_ID)  # uses env var
gcs_client = storage.Client(project=GCP_PROJECT_ID)
st.success(f"✅ Using GCP Project: {bq_client.project}")


# === Constants and Config ===
API_KEY = "3879659fe861d68dfa2866c211294684"
#FOLDER_ID = "1v6WB0jRX_yJT2JSdXRvQOLQNfOZ97iGA"
REDIRECT_URI = "https://sharp-scanner-723770381669.us-east4.run.app/"  # no longer used for login, just metadata

SPORTS = {
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb"
}

SHARP_BOOKS_FOR_LIMITS = ['pinnacle', 'bookmaker', 'betonlineag']
SHARP_BOOKS = SHARP_BOOKS_FOR_LIMITS + ['betfair_ex_eu', 'smarkets', 'matchbook']

REC_BOOKS = [
    'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers',
    'fanatics', 'espnbet', 'hard rock bet'
]

BOOKMAKER_REGIONS = {
    'pinnacle': 'us', 'bookmaker': 'us', 'betonlineag': 'us',
    'bovada': 'us', 'heritagesports': 'us', 'betus': 'us',
    'betmgm': 'us', 'draftkings': 'us', 'fanduel': 'us', 'betrivers': 'us', 'pointsbetus': 'us2',
    'bet365': 'uk', 'williamhill': 'uk', 'ladbrokes': 'uk', 'unibet': 'eu', 'bwin': 'eu',
    'sportsbet': 'au', 'ladbrokesau': 'au', 'neds': 'au'
}

MARKETS = ['spreads', 'totals', 'h2h']



# === Component fields used in sharp scoring ===
component_fields = OrderedDict({
    'Sharp_Move_Signal': 'Win Rate by Move Signal',
    'Sharp_Time_Score': 'Win Rate by Time Score',
    'Sharp_Limit_Jump': 'Win Rate by Limit Jump',
    'Sharp_Prob_Shift': 'Win Rate by Prob Shift',
    'Is_Reinforced_MultiMarket': 'Win Rate by Cross-Market Reinforcement',
    'Market_Leader': 'Win Rate by Market Leader',
    'LimitUp_NoMove_Flag': 'Win Rate by Limit↑ No Move'
})


# 🔁 Force one test row into sharp_moves_master
test_df = pd.DataFrame([{
    'Game': 'test_game',
    'Market': 'spread',
    'Outcome': 'team_a',
    'Bookmaker': 'testbook',
    'Value': -3.5,
    'Limit': 2500,
    'SHARP_HIT_BOOL': 1,
    'SHARP_COVER_RESULT': 'Win',
    'Snapshot_Timestamp': pd.Timestamp.utcnow(),
    'Sport': 'NBA'
}])

to_gbq(
    test_df,
    destination_table="sharp_data.sharp_moves_master",
    project_id="sharplogger",
    if_exists="append"
)
print("✅ Test row written to BigQuery.")



def implied_prob(odds):
    try:
        if odds < 0:
            return -odds / (-odds + 100)
        else:
            return 100 / (odds + 100)
    except:
        return None

def ensure_columns(df, required_cols, fill_value=None):
    for col in required_cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


@st.cache_data(ttl=180)
def fetch_live_odds(sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us,us2,uk,eu,au',
        'markets': ','.join(MARKETS),
        'oddsFormat': 'american',
        'includeBetLimits': 'true'
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Odds API Error: {e}")

        return []

def write_snapshot_to_bigquery(snapshot_list):
    rows = []
    snapshot_time = pd.Timestamp.utcnow()

    for game in snapshot_list:
        gid = game.get('id')
        if not gid:
            continue
        for book in game.get('bookmakers', []):
            book_key = book.get('key')
            for market in book.get('markets', []):
                market_key = market.get('key')
                for outcome in market.get('outcomes', []):
                    rows.append({
                        'Game_ID': gid,
                        'Bookmaker': book_key,
                        'Market': market_key,
                        'Outcome': outcome.get('name'),
                        'Value': outcome.get('point') if market_key != 'h2h' else outcome.get('price'),
                        'Limit': outcome.get('bet_limit'),
                        'Snapshot_Timestamp': snapshot_time
                    })

    df_snap = pd.DataFrame(rows)

    if df_snap.empty:
        print("⚠️ No snapshot data to upload.")
        return

    print("🧪 Snapshot dtypes:\n", df_snap.dtypes)

    try:
        to_gbq(df_snap, SNAPSHOTS_TABLE, project_id=GCP_PROJECT_ID, if_exists="append")
        print(f"✅ Uploaded {len(df_snap)} odds snapshot rows to BigQuery.")
    except Exception as e:
        print(f"❌ Failed to upload odds snapshot: {e}")



def write_to_bigquery(df, table=BQ_FULL_TABLE):
    if df.empty:
        print(f"⚠️ Skipping BigQuery write to {table} — DataFrame is empty.")
        return

    df = df.copy()  # avoid mutating the original
    df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()

    print(f"🟢 Attempting to write to BigQuery table: {table}")
    print("🧪 DataFrame shape:", df.shape)
    print("🧪 Columns:", df.columns.tolist())
    print("🧪 Dtypes:\n", df.dtypes)

    try:
        # First, try to CREATE the table if it doesn't exist
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='fail')
        print(f"✅ Created new table and wrote {len(df)} rows to {table}")
    except Exception as e:
        print(f"🔁 Table exists or failed to create: {e}")
        try:
            # Then fall back to append
            to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='append')
            print(f"✅ Appended {len(df)} rows to existing BigQuery table: {table}")
        except Exception as e2:
            print(f"❌ Final BigQuery write failed: {e2}")




        
def build_game_key(df):
    required = ['Game', 'Game_Start', 'Market', 'Outcome']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"⚠️ Skipping build_game_key — missing columns: {missing}")
        return df

    df = df.copy()
    df['Home_Team_Norm'] = df['Game'].str.extract(r'^(.*?) vs')[0].str.strip().str.lower()
    df['Away_Team_Norm'] = df['Game'].str.extract(r'vs (.*)$')[0].str.strip().str.lower()
    df['Commence_Hour'] = pd.to_datetime(df['Game_Start'], errors='coerce', utc=True).dt.floor('h')
    df['Market_Norm'] = df['Market'].str.strip().str.lower()
    df['Outcome_Norm'] = df['Outcome'].str.strip().str.lower()
    
    df['Game_Key'] = (
        df['Home_Team_Norm'] + "_" +
        df['Away_Team_Norm'] + "_" +
        df['Commence_Hour'].astype(str) + "_" +
        df['Market_Norm'] + "_" +
        df['Outcome_Norm']
    )

    # 🔁 ADD THIS: Create Merge_Key_Short for scoring merge
    df['Merge_Key_Short'] = (
        df['Home_Team_Norm'] + "_" +
        df['Away_Team_Norm'] + "_" +
        df['Commence_Hour'].astype(str)
    )

    return df

def read_recent_sharp_moves(hours=72, table=BQ_FULL_TABLE):
    try:
        client = bq_client
        query = f"""
            SELECT * FROM `{table}`
            WHERE Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
        """
        df = client.query(query).to_dataframe()
        print(f"✅ Loaded {len(df)} rows from BigQuery (last {hours}h)")
        return df
    except Exception as e:
        print(f"❌ Failed to read from BigQuery: {e}")
        return pd.DataFrame()


       
def read_latest_snapshot_from_bigquery(hours=2):
    try:
        client = bq_client
        query = f"""
            SELECT * FROM `{SNAPSHOTS_TABLE}`
            WHERE Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
        """
        df = client.query(query).to_dataframe()
        # Group back into the same format as before if needed
        grouped = defaultdict(lambda: {"bookmakers": []})
        for _, row in df.iterrows():
            gid = row["Game_ID"]
            entry = grouped[gid]
            found_book = next((b for b in entry["bookmakers"] if b["key"] == row["Bookmaker"]), None)
            if not found_book:
                found_book = {"key": row["Bookmaker"], "markets": []}
                entry["bookmakers"].append(found_book)
            found_market = next((m for m in found_book["markets"] if m["key"] == row["Market"]), None)
            if not found_market:
                found_market = {"key": row["Market"], "outcomes": []}
                found_book["markets"].append(found_market)
            found_market["outcomes"].append({
                "name": row["Outcome"],
                "point": row["Value"],
                "price": row["Value"] if row["Market"] == "h2h" else None,
                "bet_limit": row["Limit"]
            })
        print(f"✅ Reconstructed {len(grouped)} snapshot games from BigQuery")
        return dict(grouped)
    except Exception as e:
        print(f"❌ Failed to load snapshot from BigQuery: {e}")
        return {}


def write_market_weights_to_bigquery(weights_dict):
    try:
        rows = []
        for market, components in weights_dict.items():
            for component, values in components.items():
                for val_key, win_rate in values.items():
                    rows.append({
                        'Market': market,
                        'Component': component,
                        'Value': val_key,
                        'Win_Rate': float(win_rate)
                    })
        df = pd.DataFrame(rows)
        if not df.empty:
            to_gbq(df, MARKET_WEIGHTS_TABLE, project_id=GCP_PROJECT_ID, if_exists='replace')
            print(f"✅ Overwrote {len(df)} rows in market_weights BigQuery table.")
    except Exception as e:
        print(f"❌ Failed to write market weights to BigQuery: {e}")


def write_line_history_to_bigquery(df):
    if df is None or df.empty:
        print("⚠️ No line history data to upload.")
        return
    try:
        df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()
        to_gbq(df, LINE_HISTORY_TABLE, project_id=GCP_PROJECT_ID, if_exists="append")
        print(f"✅ Uploaded {len(df)} line history rows to BigQuery.")
    except Exception as e:
        print(f"❌ Failed to upload line history: {e}")

def fetch_scores_and_backtest(sport_key, df_moves, days_back=3, api_key=API_KEY):
    import requests
    import pandas as pd
    from datetime import datetime
    import streamlit as st

    def normalize_team(t):
        return str(t).strip().lower()

    # Determine label from SPORTS dict
    expected_label = [k for k, v in SPORTS.items() if v == sport_key]
    sport_label = expected_label[0].upper() if expected_label else "NBA"

    # ✅ Load master sharp picks
    df = read_recent_sharp_moves(hours=72)
    if df.empty or 'Game' not in df.columns:
        st.warning(f"⚠️ No sharp picks available to score for {sport_label}.")
        df_moves[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL', 'Scored']] = None
        return df_moves

    if 'Sport' not in df.columns:
        df['Sport'] = sport_label
    df = df[df['Sport'] == sport_label]

    if df.empty:
        st.warning(f"⚠️ No historical picks found for {sport_label}.")
        df_moves[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL', 'Scored']] = None
        return df_moves

    df = build_game_key(df)

    # Ensure necessary columns exist
    required = ['Game', 'Game_Start']
    for col in required:
        if col not in df.columns:
            st.warning(f"⚠️ Missing required column in historical data: {col}")
            df_moves[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL', 'Scored']] = None
            return df_moves

    df['Game_Start'] = pd.to_datetime(df['Game_Start'], utc=True, errors='coerce')
    df['Home_Team_Norm'] = df['Game'].str.extract(r'^(.*?) vs')[0].apply(normalize_team)
    df['Away_Team_Norm'] = df['Game'].str.extract(r'vs (.*)$')[0].apply(normalize_team)
    df['Commence_Hour'] = df['Game_Start'].dt.floor('h').dt.strftime("%Y-%m-%d %H:%M:%S")
    df['Merge_Key_Short'] = df['Home_Team_Norm'] + "_" + df['Away_Team_Norm'] + "_" + df['Commence_Hour']
    df = df[df['Game_Start'] < pd.Timestamp.utcnow()]
    if 'SHARP_HIT_BOOL' in df.columns:
        df = df[df['SHARP_HIT_BOOL'].isna()]

    # ✅ Fetch completed games with scores
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
        response = requests.get(url, params={'apiKey': api_key, 'daysFrom': days_back}, timeout=10)
        response.raise_for_status()
        games = response.json()
        completed_games = [g for g in games if g.get("completed")]
    except Exception as e:
        st.error(f"❌ Failed to fetch scores from Odds API: {e}")
        df_moves[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL', 'Scored']] = None
        return df_moves

    # Build score rows
    score_rows = []
    for game in completed_games:
        home = normalize_team(game.get("home_team", ""))
        away = normalize_team(game.get("away_team", ""))
        game_start = pd.to_datetime(game.get("commence_time"), utc=True)
        if pd.isna(game_start):
            continue
        merge_key = f"{home}_{away}_{game_start.floor('h').strftime('%Y-%m-%d %H:%M:%S')}"
        scores = {s.get("name", "").strip().lower(): s.get("score") for s in game.get("scores", [])}
        if home in scores and away in scores:
            score_rows.append({
                'Merge_Key_Short': merge_key,
                'Score_Home_Score': scores[home],
                'Score_Away_Score': scores[away]
            })

    df_scores = pd.DataFrame(score_rows)
    if df_scores.empty:
        st.warning("ℹ️ No valid score rows found from completed games.")
        df_moves[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL', 'Scored']] = None
        return df_moves

    if 'Merge_Key_Short' not in df.columns:
        st.warning("⚠️ Missing merge keys in sharp picks.")
        df_moves[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL', 'Scored']] = None
        return df_moves

    # ✅ Merge scores
    df_scores.rename(columns={
        "Score_Home_Score": "Score_Home_Score_api",
        "Score_Away_Score": "Score_Away_Score_api"
    }, inplace=True)

    df = df.merge(df_scores, on='Merge_Key_Short', how='left')
    for col in ['Score_Home_Score', 'Score_Away_Score']:
        api_col = f"{col}_api"
        df[col] = pd.to_numeric(df.get(col), errors='coerce')
        df[api_col] = pd.to_numeric(df.get(api_col), errors='coerce')
        df[col] = df[col].combine_first(df[api_col])
        df.drop(columns=[api_col], inplace=True, errors='ignore')

    # ✅ Prepare to score
    df_valid = df.dropna(subset=['Score_Home_Score', 'Score_Away_Score', 'Ref Sharp Value']).copy()
    if df_valid.empty:
        st.warning("ℹ️ No valid sharp picks with both scores and line values to score.")
        df_moves[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL', 'Scored']] = None
        return df_moves

    st.success(f"✅ Scores merged. {len(df_valid)} valid rows ready for scoring.")
    st.dataframe(df_valid[['Game', 'Market', 'Outcome', 'Ref Sharp Value', 'Score_Home_Score', 'Score_Away_Score']].head())

    # ✅ Score logic
    def calc_cover(row):
        try:
            h, a = float(row['Score_Home_Score']), float(row['Score_Away_Score'])
            market, outcome = str(row.get('Market', '')).lower(), str(row.get('Outcome', '')).lower()
            val = float(row.get('Ref Sharp Value', 0))

            if market == 'totals':
                total = h + a
                return ['Win', 1] if ('under' in outcome and total < val) or ('over' in outcome and total > val) else ['Loss', 0]

            margin = h - a if row['Home_Team_Norm'] in outcome else a - h
            if market == 'spreads':
                hit = (margin > abs(val)) if val < 0 else (margin + val > 0)
                return ['Win', 1] if hit else ['Loss', 0]

            if market == 'h2h':
                return ['Win', 1] if ((row['Home_Team_Norm'] in outcome and h > a) or
                                      (row['Away_Team_Norm'] in outcome and a > h)) else ['Loss', 0]

            return [None, 0]
        except:
            return [None, 0]

    # ✅ Apply scoring
    df[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']] = None
    df['Scored'] = False
    result = df_valid.apply(calc_cover, axis=1, result_type='expand')
    result.columns = ['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']
    df.loc[df_valid.index, ['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']] = result
    df.loc[df_valid.index, 'Scored'] = result['SHARP_COVER_RESULT'].notna()

    return df


            
def detect_market_leaders(df_history, sharp_books, rec_books):
    df_history = df_history.copy()
    df_history['Time'] = pd.to_datetime(df_history['Time'])
    df_history['Book'] = df_history['Book'].str.lower()

    # Detect open value per Book
    df_open = (
        df_history
        .sort_values('Time')
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Value'})
    )

    df_history = df_history.merge(df_open, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df_history['Has_Moved'] = (df_history['Value'] != df_history['Open_Value']) & df_history['Value'].notna()

    first_moves = (
        df_history[df_history['Has_Moved']]
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Time']
        .min()
        .reset_index()
        .rename(columns={'Time': 'First_Move_Time'})
    )

    first_moves['Book_Type'] = first_moves['Book'].map(
        lambda b: 'Sharp' if b in sharp_books else ('Rec' if b in rec_books else 'Other')
    )
    first_moves['Move_Rank'] = first_moves.groupby(
        ['Game', 'Market', 'Outcome']
    )['First_Move_Time'].rank(method='first')

    first_moves['Market_Leader'] = (
        (first_moves['Book_Type'] == 'Sharp') & (first_moves['Move_Rank'] == 1)
    )

    return first_moves



def read_market_weights_from_bigquery():
    try:
        client = bq_client
        query = f"SELECT * FROM `{MARKET_WEIGHTS_TABLE}`"
        df = client.query(query).to_dataframe()
        weights = defaultdict(lambda: defaultdict(dict))
        for _, row in df.iterrows():
            market = row['Market']
            component = row['Component']
            value = str(row['Value']).lower()
            win_rate = float(row['Win_Rate'])
            weights[market][component][value] = win_rate
        print(f"✅ Loaded {len(df)} market weight rows from BigQuery.")
        return dict(weights)
    except Exception as e:
        print(f"❌ Failed to load market weights from BigQuery: {e}")
        return {}



def detect_cross_market_sharp_support(df_moves):
    df = df_moves.copy()
    df['SupportKey'] = df['Game'].astype(str) + " | " + df['Outcome'].astype(str)

    df_sharp = df[df['SharpBetScore'] >= 25].copy()

    market_counts = (
        df_sharp.groupby('SupportKey')['Market']
        .nunique()
        .reset_index()
        .rename(columns={'Market': 'CrossMarketSharpSupport'})
    )

    df = df.merge(market_counts, on='SupportKey', how='left')
    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Is_Reinforced_MultiMarket'] = df['CrossMarketSharpSupport'] >= 2

    return df


def detect_sharp_moves(current, previous, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, weights={}):
    from collections import defaultdict
    import pandas as pd
    from datetime import datetime

    def normalize_label(label):
        return str(label).strip().lower().replace('.0', '')

    if not current:
        print("⚠️ No current odds data provided.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}

    rows = []
    sharp_limit_map = defaultdict(lambda: defaultdict(list))
    sharp_total_limit_map = defaultdict(int)
    sharp_lines = {}
    line_history_log = {}
    line_open_map = {}
    sharp_side_flags = {}
    sharp_metrics_map = {}

    sport_scope_key = {
        'MLB': 'baseball_mlb',
        'NBA': 'basketball_nba'
    }.get(sport_key.upper(), sport_key.lower())
    confidence_weights = weights.get(sport_scope_key, {})

    # === Safely flatten previous odds
    previous_odds_map = {}
    for g in previous_map.values():
        for book in g.get('bookmakers', []):
            book_key = book.get('key')
            for market in book.get('markets', []):
                mtype = market.get('key')
                for outcome in market.get('outcomes', []):
                    label = normalize_label(outcome.get('name', ''))
                    price = outcome.get('point') if mtype != 'h2h' else outcome.get('price')
                    previous_odds_map[(g.get('home_team'), g.get('away_team'), mtype, label, book_key)] = price

    # === Parse current odds
    for game in current:
        home_team = game.get('home_team', '').strip().lower()
        away_team = game.get('away_team', '').strip().lower()
        if not home_team or not away_team:
            continue

        game_name = f"{home_team.title()} vs {away_team.title()}"
        event_time = pd.to_datetime(game.get("commence_time"), utc=True, errors='coerce')
        event_date = event_time.strftime("%Y-%m-%d") if pd.notnull(event_time) else ""
        game_hour = event_time.floor('h') if pd.notnull(event_time) else pd.NaT
        gid = game.get('id')
        prev_game = previous_map.get(gid, {})
        
        for book in game.get('bookmakers', []):
            book_key_raw = book.get('key', '').lower()
            book_key = book_key_raw  # default
        
            # 🔄 Normalize to known sharp/rec books
            for rec in REC_BOOKS:
                if rec.replace(" ", "") in book_key_raw:
                    book_key = rec.replace(" ", "")
                    break
            for sharp in SHARP_BOOKS:
                if sharp in book_key_raw:
                    book_key = sharp
                    break
        
            book_title = book.get('title', book_key)
        
            for market in book.get('markets', []):
                mtype = market.get('key', '').strip().lower()
                if mtype not in ['spreads', 'totals', 'h2h']:
                    continue

                for o in market.get('outcomes', []):
                    label = normalize_label(o.get('name', ''))
                    val = o.get('point') if mtype != 'h2h' else o.get('price')
                    limit = o.get('bet_limit')
                    prev_key = (game.get('home_team'), game.get('away_team'), mtype, label, book_key)
                    old_val = previous_odds_map.get(prev_key)

                    game_key = f"{home_team}_{away_team}_{str(game_hour)}_{mtype}_{label}"

                    entry = {
                        'Sport': sport_key.upper(),
                        'Game_Key': game_key,
                        'Time': snapshot_time,
                        'Game': game_name,
                        'Game_Start': event_time,
                        'Event_Date': event_date,
                        'Market': mtype,
                        'Outcome': label,
                        'Bookmaker': book_title,
                        'Book': book_key,
                        'Value': val,
                        'Limit': limit,
                        'Old Value': old_val,
                        'Delta': round(val - old_val, 2) if old_val is not None and val is not None else None,
                        'Home_Team_Norm': home_team,
                        'Away_Team_Norm': away_team,
                        'Commence_Hour': game_hour
                    }

                    # Fallback to previous game override if available
                    if prev_game:
                        for prev_b in prev_game.get('bookmakers', []):
                            if prev_b.get('key', '').lower() == book_key_raw:
                                for prev_m in prev_b.get('markets', []):
                                    if prev_m.get('key') == mtype:
                                        for prev_o in prev_m.get('outcomes', []):
                                            if normalize_label(prev_o.get('name')) == label:
                                                prev_val = prev_o.get('point') if mtype != 'h2h' else prev_o.get('price')
                                                if prev_val is not None:
                                                    entry['Old Value'] = prev_val
                                                    entry['Delta'] = round(val - prev_val, 2) if val is not None else None

                    rows.append(entry)
                    line_history_log.setdefault(gid, []).append(entry.copy())

                    if val is not None:
                        sharp_lines[(game_name, mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit, val, old_val))
                        if book_key in SHARP_BOOKS:
                            sharp_total_limit_map[(game_name, mtype, label)] += limit or 0
                        if (game_name, mtype, label) not in line_open_map:
                            line_open_map[(game_name, mtype, label)] = (val, snapshot_time)

    df_moves = pd.DataFrame(rows)
    df_audit = pd.DataFrame([item for sublist in line_history_log.values() for item in sublist])
    df_sharp_lines = pd.DataFrame(sharp_lines.values())




    
    # === Sharp scoring logic
    # === Sharp scoring logic (safe version)
    for (game_name, mtype), label_map in sharp_limit_map.items():
        scores = {}
        label_signals = {}
        for label, entries in label_map.items():
            move_signal = limit_jump = prob_shift = time_score = 0
            move_magnitude_score = 0.0
    
            for limit, curr, _ in entries:
                open_val, _ = line_open_map.get((game_name, mtype, label), (None, None))
                if open_val is not None and curr is not None:
                    try:
                        sharp_move_delta = abs(curr - open_val)
                        if sharp_move_delta >= 0.01:
                            move_signal += 1
                            move_magnitude_score += sharp_move_delta
    
                        if mtype == 'totals':
                            if 'under' in label and curr < open_val: move_signal += 1
                            elif 'over' in label and curr > open_val: move_signal += 1
                        elif mtype == 'spreads' and abs(curr) > abs(open_val): move_signal += 1
                        elif mtype == 'h2h':
                            imp_now, imp_open = implied_prob(curr), implied_prob(open_val)
                            if imp_now and imp_open and imp_now > imp_open:
                                prob_shift += 1
                    except:
                        continue
    
                if limit is not None and limit >= 100:
                    limit_jump += 1
    
                try:
                    hour = datetime.now().hour
                    time_score += 1.0 if 6 <= hour <= 11 else 0.5 if hour <= 15 else 0.2
                except:
                    time_score += 0.5  # fallback
    
            move_magnitude_score = min(move_magnitude_score, 5.0)
            total_limit = sharp_total_limit_map.get((game_name, mtype, label), 0)
    
            scores[label] = (
                2 * move_signal +
                2 * limit_jump +
                1.5 * time_score +
                1.0 * prob_shift +
                0.001 * total_limit +
                3.0 * move_magnitude_score
            )
    
            label_signals[label] = {
                'Sharp_Move_Signal': move_signal,
                'Sharp_Limit_Jump': limit_jump,
                'Sharp_Time_Score': time_score,
                'Sharp_Prob_Shift': prob_shift,
                'Sharp_Limit_Total': total_limit,
                'Sharp_Move_Magnitude_Score': round(move_magnitude_score, 2)
            }
    
        if scores:
            best_label = max(scores, key=scores.get)
            sharp_side_flags[(game_name, mtype, best_label)] = 1
            sharp_metrics_map[(game_name, mtype, best_label)] = label_signals.get(best_label, {})

    # === Assign sharp-side logic + metric scores to all rows
    # === Assign sharp-side logic + metric scores
    for row in rows:
        game_name = row.get('Game', '')
        mtype = row.get('Market', '')
        label = row.get('Outcome', '')
        metrics = sharp_metrics_map.get((game_name, mtype, label), {})
        is_sharp_side = int(sharp_side_flags.get((game_name, mtype, label), 0))
    
        row.update({
            'Ref Sharp Value': row.get('Value'),
            'Ref Sharp Old Value': row.get('Old Value'),
            'Delta vs Sharp': 0.0,
            'SHARP_SIDE_TO_BET': is_sharp_side,
            'Sharp_Move_Signal': metrics.get('Sharp_Move_Signal', 0),
            'Sharp_Limit_Jump': metrics.get('Sharp_Limit_Jump', 0),
            'Sharp_Time_Score': metrics.get('Sharp_Time_Score', 0),
            'Sharp_Prob_Shift': metrics.get('Sharp_Prob_Shift', 0),
            'Sharp_Limit_Total': metrics.get('Sharp_Limit_Total', 0)
        })
    
        try:
            row['SharpBetScore'] = round(
                2.0 * row['Sharp_Move_Signal'] +
                2.0 * row['Sharp_Limit_Jump'] +
                1.5 * row['Sharp_Time_Score'] +
                1.0 * row['Sharp_Prob_Shift'] +
                0.001 * row['Sharp_Limit_Total'], 2
            )
        except:
            row['SharpBetScore'] = 0.0

    
    df = pd.DataFrame(rows)
    df_history = df.copy()  # needed for downstream sort on 'Time'

    # === Intelligence scoring
    def compute_weighted_signal(row, market_weights):
        market = str(row.get('Market', '')).lower()
        total_score = 0
        max_possible = 0
    
        component_importance = {
            'Sharp_Move_Signal': 2.0,
            'Sharp_Limit_Jump': 2.0,
            'Sharp_Time_Score': 1.5,
            'Sharp_Prob_Shift': 1.0,
            'Sharp_Limit_Total': 0.001
        }
    
        for comp, importance in component_importance.items():
            val = row.get(comp)
            if val is None:
                continue
    
            try:
                val_key = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val).lower()
                weight = market_weights.get(market, {}).get(comp, {}).get(val_key, 0.5)
            except:
                weight = 0.5
    
            total_score += weight * importance
            max_possible += importance
    
        return round((total_score / max_possible) * 100 if max_possible else 50, 2)
    
    
    def compute_confidence(row, market_weights):
        try:
            # Base scaled sharp signal
            base_score = min(row.get('SharpBetScore', 0) / 50, 1.0) * 50
    
            # Empirical weight signal
            weight_score = compute_weighted_signal(row, market_weights)
    
            # Limit positioning bonus
            limit_position_bonus = 0
            if row.get('LimitUp_NoMove_Flag') == 1:
                limit_position_bonus = 15
            elif row.get('Limit_Jump') == 1 and abs(row.get('Delta vs Sharp', 0)) > 0.25:
                limit_position_bonus = 5
    
            # Market leadership bonus
            market_lead_bonus = 5 if row.get('Market_Leader') else 0
    
            # Final blended score
            final_conf = base_score + weight_score + limit_position_bonus + market_lead_bonus
            return round(min(final_conf, 100), 2)
    
        except Exception as e:
            print(f"⚠️ Confidence scoring error: {e}")
            return 50.0  # fallback neutral score
    
 
    
    # === Sort by timestamp and extract open lines
    df_history_sorted = df_history.sort_values('Time')
    
    # Global open per market/outcome (first ever seen)
    line_open_df = (
        df_history_sorted.dropna(subset=['Value'])
        .groupby(['Game', 'Market', 'Outcome'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Value'})
    )
    
    # Per-book open value (first seen per book)
    line_open_per_book = (
        df_history_sorted.dropna(subset=['Value'])
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Book_Value'})
    )
    # === Extract opening limit for each (Game, Market, Outcome, Book)
    open_limit_df = (
        df_history_sorted
        .dropna(subset=['Limit'])
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Limit']
        .first()
        .reset_index()
        .rename(columns={'Limit': 'Opening_Limit'})
    )

    # === Early exit if empty
    if df.empty:
        return df, df_history
    
    # === Merge opening lines into live odds
    df = df.merge(line_open_df, on=['Game', 'Market', 'Outcome'], how='left')
    df = df.merge(line_open_per_book, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df = df.merge(open_limit_df, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df['Delta vs Sharp'] = df['Value'] - df['Open_Value']
    df['Delta'] = pd.to_numeric(df['Delta vs Sharp'], errors='coerce')
    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)
    df['Limit_Jump'] = (df['Limit'] >= 2500).astype(int)
    df['Sharp_Timing'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour.apply(
        lambda h: 1.0 if 6 <= h <= 11 else 0.5 if h <= 15 else 0.2
    )
    df['Limit_NonZero'] = df['Limit'].where(df['Limit'] > 0)
    df['Limit_Max'] = df.groupby(['Game', 'Market'])['Limit_NonZero'].transform('max')
    df['Limit_Min'] = df.groupby(['Game', 'Market'])['Limit_NonZero'].transform('min')
    
  
    df['Book'] = df['Book'].str.lower()
    
    market_leader_flags = detect_market_leaders(df_history, SHARP_BOOKS, REC_BOOKS)
    df = df.merge(
        market_leader_flags[['Game', 'Market', 'Outcome', 'Book', 'Market_Leader']],
        on=['Game', 'Market', 'Outcome', 'Book'],
        how='left'
    )
    
    df['Is_Pinnacle'] = df['Book'] == 'pinnacle'
    df['LimitUp_NoMove_Flag'] = (
        (df['Is_Pinnacle']) &
        (df['Limit'] >= 2500) &
        (df['Value'] == df['Open_Value'])
    ).astype(int)
    
    df = detect_cross_market_sharp_support(df)
    
    # ✅ Compute dynamic, market-calibrated confidence
    df['True_Sharp_Confidence_Score'] = df.apply(
        lambda r: compute_weighted_signal(r, confidence_weights), axis=1
    )
    df['Enhanced_Sharp_Confidence_Score'] = df.apply(
        lambda r: compute_confidence(r, confidence_weights), axis=1
    )
    
    # ✅ Tiering based on enhanced score
    df['Sharp_Confidence_Tier'] = pd.cut(
        df['Enhanced_Sharp_Confidence_Score'],
        bins=[-1, 25, 50, 75, float('inf')],
        labels=['⚠️ Low', '✅ Medium', '⭐ High', '🔥 Steam']
    )
    
    # === Sharp vs Rec Book Consensus Summary
    rec_df = df[df['Book'].isin(REC_BOOKS)].copy()
    sharp_df = df[df['Book'].isin(SHARP_BOOKS)].copy()
    
    def summarize_group(g):
        return pd.Series({
            'Rec_Book_Consensus': g[g['Book'].isin(REC_BOOKS)]['Value'].mean(),
            'Sharp_Book_Consensus': g[g['Book'].isin(SHARP_BOOKS)]['Value'].mean(),
            'Rec_Open': g[g['Book'].isin(REC_BOOKS)]['Open_Value'].mean(),
            'Sharp_Open': g[g['Book'].isin(SHARP_BOOKS)]['Open_Value'].mean()
        })
    
    summary_df = (
        df.groupby(['Event_Date', 'Game', 'Market', 'Outcome'])
        .apply(summarize_group)
        .reset_index()
    )
    
    # Restore 'Recommended_Outcome'
    summary_df['Recommended_Outcome'] = summary_df['Outcome']
    
    # Compute move deltas
    summary_df['Move_From_Open_Rec'] = (
        summary_df['Rec_Book_Consensus'] - summary_df['Rec_Open']
    ).fillna(0)
    
    summary_df['Move_From_Open_Sharp'] = (
        summary_df['Sharp_Book_Consensus'] - summary_df['Sharp_Open']
    ).fillna(0)
    
    # Merge sharp scoring values
    sharp_scores = df[df['SharpBetScore'].notnull()][[
        'Event_Date', 'Game', 'Market', 'Outcome',
        'SharpBetScore',
        'Enhanced_Sharp_Confidence_Score',
        'Sharp_Confidence_Tier'
    ]].drop_duplicates()
    
    summary_df = summary_df.merge(
        sharp_scores,
        on=['Event_Date', 'Game', 'Market', 'Outcome'],
        how='left'
    )
    
    summary_df[['SharpBetScore', 'Enhanced_Sharp_Confidence_Score']] = summary_df[[
        'SharpBetScore', 'Enhanced_Sharp_Confidence_Score'
    ]].fillna(0)
    
    summary_df['Sharp_Confidence_Tier'] = summary_df['Sharp_Confidence_Tier'].fillna('⚠️ Low')


    required_sharp_cols = [
        'SharpBetScore', 'Enhanced_Sharp_Confidence_Score',
        'Sharp_Confidence_Tier', 'True_Sharp_Confidence_Score'
    ]
    for col in required_sharp_cols:
        if col not in df.columns:
            df[col] = None
    
    return df, df_history, summary_df

def train_sharp_win_model(df):
    st.subheader("🔍 Sharp Model Training Debug")
    st.write(f"Total rows: {len(df)}")

    st.write("With SHARP_HIT_BOOL:", len(df[df['SHARP_HIT_BOOL'].notna()]))
    st.write("With Enhanced_Sharp_Confidence_Score:", len(df[df['Enhanced_Sharp_Confidence_Score'].notna()]))
    st.write("With True_Sharp_Confidence_Score:", len(df[df['True_Sharp_Confidence_Score'].notna()]))
    
    

    # === Fallback logic: use Enhanced if available, else True_Sharp_Confidence_Score
    df['Final_Confidence_Score'] = df['Enhanced_Sharp_Confidence_Score']
    if 'True_Sharp_Confidence_Score' in df.columns:
        df['Final_Confidence_Score'] = df['Final_Confidence_Score'].fillna(df['True_Sharp_Confidence_Score'])

    df_filtered = df[
        df['SHARP_HIT_BOOL'].notna() &
        df['Final_Confidence_Score'].notna() 
            
    ]
    st.write("Rows passing all filters:", len(df_filtered))

    df_labeled = df_filtered.copy()
    if df_labeled.empty:
        raise ValueError("❌ No data available for sharp model training — df_labeled is empty.")

    df_labeled['target'] = df_labeled['SHARP_HIT_BOOL'].astype(int)

    # Normalize score to 0–1 range
    df_labeled['Final_Confidence_Score'] = df_labeled['Final_Confidence_Score'] / 100

    feature_cols = ['Final_Confidence_Score']
    if 'CrossMarketSharpSupport' in df_labeled.columns:
        feature_cols.append('CrossMarketSharpSupport')

    df_labeled = df_labeled.dropna(subset=feature_cols)
    if len(df_labeled) < 5:
        raise ValueError(f"❌ Not enough samples to train model — only {len(df_labeled)} rows.")

    X = df_labeled[feature_cols].astype(float)
    y = df_labeled['target'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"✅ Trained Sharp Win Model — AUC: {auc:.3f} on {len(df_labeled)} samples")

    return model





def apply_blended_sharp_score(df, model):
    import numpy as np
    df = df.copy()

    # === Step 1: Clean up any _x / _y duplicates to avoid confusion
    df = df.drop(columns=[col for col in df.columns if col.endswith(('_x', '_y'))], errors='ignore')

    # === Step 2: Confirm confidence column exists
    if 'Enhanced_Sharp_Confidence_Score' not in df.columns:
        raise ValueError("❌ Missing Enhanced_Sharp_Confidence_Score in df")

    # === Step 3: Build final confidence column
    df['Final_Confidence_Score'] = df['Enhanced_Sharp_Confidence_Score']
    if 'True_Sharp_Confidence_Score' in df.columns:
        df['Final_Confidence_Score'] = df['Final_Confidence_Score'].fillna(df['True_Sharp_Confidence_Score'])

    # Clamp or convert safely
    df['Final_Confidence_Score'] = pd.to_numeric(df['Final_Confidence_Score'], errors='coerce')
    df['Final_Confidence_Score'] = df['Final_Confidence_Score'] / 100
    df['Final_Confidence_Score'] = df['Final_Confidence_Score'].clip(0, 1)

    # === Step 4: Validate model features
    model_features = model.get_booster().feature_names
    feature_cols = [col for col in model_features if col in df.columns]
    missing = set(model_features) - set(feature_cols)
    if missing:
        raise ValueError(f"❌ Missing model feature columns: {missing}")

    # === Step 5: Predict
    X = df[feature_cols].astype(float)
    df['Model_Sharp_Win_Prob'] = model.predict_proba(X)[:, 1]

    # === Step 6: Blend into final sharp score
    df['Blended_Sharp_Score'] = (
        0.5 * df['Model_Sharp_Win_Prob'] +
        0.5 * df['Final_Confidence_Score']
    )

    return df

from google.cloud import storage

def save_model_to_gcs(model, bucket_name="sharp-models", filename="sharp_win_model.pkl"):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        buffer = BytesIO()
        pickle.dump(model, buffer)
        blob.upload_from_string(buffer.getvalue(), content_type='application/octet-stream')
        print(f"✅ Model saved to GCS: gs://{bucket_name}/{filename}")
    except Exception as e:
        print(f"❌ Failed to save model to GCS: {e}")


def load_model_from_gcs(bucket_name="sharp-models", filename="sharp_win_model.pkl"):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        content = blob.download_as_bytes()
        model = pickle.loads(content)
        print(f"✅ Loaded model from GCS: gs://{bucket_name}/{filename}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model from GCS: {e}")
        return None



def train_sharp_win_model(df):
    st.subheader("🔍 Sharp Model Training Debug")
    st.write(f"Total rows in df: {len(df)}")

    # === Build final confidence score from available columns
    if 'Enhanced_Sharp_Confidence_Score' not in df.columns and 'True_Sharp_Confidence_Score' not in df.columns:
        st.error("❌ No confidence columns available for training.")
        return None

    df['Final_Confidence_Score'] = df.get('Enhanced_Sharp_Confidence_Score')
    if 'True_Sharp_Confidence_Score' in df.columns:
        df['Final_Confidence_Score'] = df['Final_Confidence_Score'].fillna(df['True_Sharp_Confidence_Score'])

    # === Filter training set: only rows with score and result
    df_filtered = df[
        df['Final_Confidence_Score'].notna() &
        df['SHARP_HIT_BOOL'].notna()
    ].copy()

    st.write("📊 Rows with SHARP_HIT_BOOL:", df_filtered['SHARP_HIT_BOOL'].notna().sum())
    st.write("📊 Rows with confidence:", df_filtered['Final_Confidence_Score'].notna().sum())
    st.write("📊 Rows passing both filters:", len(df_filtered))

    if df_filtered.empty or len(df_filtered) < 5:
        st.warning("⚠️ Not enough rows to train model.")
        return None

    # === Prepare features and labels
    df_filtered['Final_Confidence_Score'] = df_filtered['Final_Confidence_Score'] / 100
    df_filtered['Final_Confidence_Score'] = df_filtered['Final_Confidence_Score'].clip(0, 1)
    df_filtered['target'] = df_filtered['SHARP_HIT_BOOL'].astype(int)

    feature_cols = ['Final_Confidence_Score']
    if 'CrossMarketSharpSupport' in df_filtered.columns:
        feature_cols.append('CrossMarketSharpSupport')

    df_filtered = df_filtered.dropna(subset=feature_cols)

    X = df_filtered[feature_cols].astype(float)
    y = df_filtered['target']

    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    st.success(f"✅ Trained Sharp Win Model — AUC: {auc:.3f} on {len(df_filtered)} samples")

    return model


def render_scanner_tab(label, sport_key, container):
    market_component_win_rates = read_market_weights_from_bigquery()
    timestamp = pd.Timestamp.utcnow()
    sport_key_lower = sport_key.lower()

    with container:
        st.subheader(f"📡 Scanning {label} Sharp Signals")

        # === 1. Fetch Live Odds ===
        live = fetch_live_odds(sport_key)
        if not live:
            st.warning("⚠️ No live odds returned.")
            return pd.DataFrame()

        write_snapshot_to_bigquery(live)

        # === 2. Load Previous Snapshot from BigQuery ===
        prev = read_latest_snapshot_from_bigquery(hours=2)
        if not prev:
            st.info("🟡 First run — no previous snapshot. Continuing with empty prev.")
            prev = {}

        # === 3. Run Sharp Detection ===
        df_moves_raw, df_audit, summary_df = detect_sharp_moves(
            live, prev, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS,
            weights=market_component_win_rates
        )

        if df_moves_raw.empty or 'Enhanced_Sharp_Confidence_Score' not in df_moves_raw.columns:
            st.warning("⚠️ No sharp signals detected.")
            return pd.DataFrame()

        df_moves_raw['Snapshot_Timestamp'] = timestamp
        df_moves_raw['Game_Start'] = pd.to_datetime(df_moves_raw['Game_Start'], errors='coerce', utc=True)
        df_moves_raw['Sport'] = label.upper()
        df_moves_raw = df_moves_raw[df_moves_raw['Sport'] == label.upper()]
        df_moves_raw = build_game_key(df_moves_raw)

        df_moves = df_moves_raw.drop_duplicates(subset=['Game_Key', 'Bookmaker'], keep='first').copy()

        # === 4. Restore Game_Start if missing
        if 'Game_Start' not in df_moves.columns or df_moves['Game_Start'].isna().all():
            df_moves = df_moves.merge(
                df_moves_raw[['Game_Key', 'Bookmaker', 'Game_Start']].drop_duplicates(),
                on=['Game_Key', 'Bookmaker'],
                how='left'
            )
        
        model = load_model_from_gcs(bucket_name=GCS_BUCKET)
        
        if model is not None:
            try:
                df_moves_raw = apply_blended_sharp_score(df_moves_raw, model)
                st.success("✅ Applied model scoring to raw sharp data")
            except Exception as e:
                st.warning(f"⚠️ Could not apply model scoring early: {e}")
        
        # ✅ Always upload today's sharp picks (raw) — even if not yet scored
        if not df_moves_raw.empty:
            df_moves_raw['Sport'] = label.upper()
            write_to_bigquery(df_moves_raw)
            st.info(f"✅ Uploaded {len(df_moves_raw)} unscored sharp picks to BigQuery.")
        
        # === 5. Score Historical Games
        df_bt = fetch_scores_and_backtest(sport_key, df_moves, api_key=API_KEY)


        if not df_bt.empty:
            # Ensure merge-safe columns exist
            merge_cols = ['Game_Key', 'Market', 'Bookmaker']
            confidence_cols = ['Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score', 'Sharp_Confidence_Tier']
            df_bt = ensure_columns(df_bt, merge_cols)
            df_moves_raw = ensure_columns(df_moves_raw, merge_cols + confidence_cols)

            available = [col for col in confidence_cols if col in df_moves_raw.columns]
            df_bt = df_bt.merge(
                df_moves_raw[merge_cols + available].drop_duplicates(),
                on=merge_cols,
                how='left'
            )

            if model is not None:
                try:
                    df_bt = apply_blended_sharp_score(df_bt, model)
                except Exception as e:
                    st.warning(f"⚠️ Could not apply model scoring to backtested data: {e}")

            if 'Enhanced_Sharp_Confidence_Score' in df_bt.columns:
                trainable = df_bt[
                    df_bt['SHARP_HIT_BOOL'].notna() &
                    df_bt['Enhanced_Sharp_Confidence_Score'].notna()
                ]
                if len(trainable) >= 5:
                    model = train_sharp_win_model(trainable)
                    save_model_to_gcs(model, bucket_name=GCS_BUCKET)
                else:
                    st.info("ℹ️ Not enough completed sharp picks to retrain model.")
            else:
                st.warning("⚠️ Skipping training — confidence score column missing.")

            df_bt['Sport'] = label.upper()
            write_to_bigquery(df_bt)
            df_moves = df_bt.copy()
            
            # ✅ Confirm score upload
            count_scored = len(df_bt[df_bt['SHARP_HIT_BOOL'].notna()])
            st.success(f"✅ Confirmed {count_scored} scored sharp picks uploaded to BigQuery.")
            def preview_sharp_master(label, limit=25):
                client = bq_client
                query = f"""
                    SELECT Game, Market, Outcome, SHARP_HIT_BOOL, SHARP_COVER_RESULT, Snapshot_Timestamp
                    FROM `{BQ_FULL_TABLE}`
                    WHERE Sport = '{label.upper()}'
                      AND SHARP_HIT_BOOL IS NULL
                    ORDER BY Snapshot_Timestamp DESC
                    LIMIT {limit}
                """
                df_preview = client.query(query).to_dataframe()
                return df_preview
            
            # Display table preview in the app
            st.subheader(f"📋 Latest Scored Picks in Sharp Master – {label}")
            df_preview = preview_sharp_master(label)
            st.dataframe(df_preview)
            st.success(f"✅ Uploaded {len(df_bt)} scored picks to BigQuery")
        else:
            st.info("ℹ️ No backtest results to score.")

        if not df_audit.empty:
            df_audit['Snapshot_Timestamp'] = timestamp
            write_line_history_to_bigquery(df_audit)

        # === 6. Summary Table ===
        if summary_df.empty:
            st.info("ℹ️ No summary data available.")
            return df_moves

        st.subheader(f"📊 Sharp vs Rec Book Consensus Summary – {label}")
        for col in ['Game', 'Outcome']:
            summary_df[col] = summary_df[col].str.strip().str.lower()
            df_moves[col] = df_moves[col].str.strip().str.lower()

        if 'Blended_Sharp_Score' in df_moves.columns:
            df_merge_scores = df_moves[['Game', 'Market', 'Outcome', 'Blended_Sharp_Score', 'Model_Sharp_Win_Prob']].drop_duplicates()
            summary_df = summary_df.merge(
                df_merge_scores,
                on=['Game', 'Market', 'Outcome'],
                how='left'
            )

        if {'Event_Date', 'Market', 'Game'}.issubset(df_moves_raw.columns):
            df_game_start = df_moves_raw[['Game', 'Market', 'Event_Date', 'Game_Start']].dropna().drop_duplicates()
            df_game_start['MergeKey'] = (
                df_game_start['Game'].str.strip().str.lower() + "_" +
                df_game_start['Market'].str.strip().str.lower() + "_" +
                df_game_start['Event_Date'].astype(str)
            )
            summary_df['MergeKey'] = (
                summary_df['Game'].str.strip().str.lower() + "_" +
                summary_df['Market'].str.strip().str.lower() + "_" +
                summary_df['Event_Date'].astype(str)
            )
            if 'MergeKey' in df_game_start.columns:
                summary_df = summary_df.merge(
                    df_game_start[['MergeKey', 'Game_Start']],
                    on='MergeKey',
                    how='left'
                )

        def safe_to_est(dt):
            if pd.isna(dt):
                return ""
            try:
                dt = pd.to_datetime(dt, errors='coerce')
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                return dt.tz_convert('US/Eastern').strftime('%Y-%m-%d %I:%M %p')
            except:
                return ""

        summary_df['Game_Start'] = pd.to_datetime(summary_df['Game_Start'], errors='coerce', utc=True)
        summary_df['Date + Time (EST)'] = summary_df['Game_Start'].apply(safe_to_est)
        summary_df = summary_df[summary_df['Date + Time (EST)'] != ""]

        summary_df.rename(columns={
            'Date + Time (EST)': 'Date\n+ Time (EST)',
            'Game': 'Matchup',
            'Recommended_Outcome': 'Pick\nSide',
            'Rec_Book_Consensus': 'Rec\nConsensus',
            'Sharp_Book_Consensus': 'Sharp\nConsensus',
            'Move_From_Open_Rec': 'Rec\nMove',
            'Move_From_Open_Sharp': 'Sharp\nMove',
            'SharpBetScore': 'Sharp\nBet\nScore',
            'Enhanced_Sharp_Confidence_Score': 'Enhanced\nConf.\nScore',
        }, inplace=True)

        summary_df = summary_df.drop_duplicates(subset=["Matchup", "Market", "Pick\nSide", "Date\n+ Time (EST)"])

        market_options = ["All"] + sorted(summary_df['Market'].dropna().unique())
        market = st.selectbox(f"📊 Filter {label} by Market", market_options, key=f"{label}_market_summary")
        filtered_df = summary_df if market == "All" else summary_df[summary_df['Market'] == market]

        view_cols = ['Date\n+ Time (EST)', 'Matchup', 'Market', 'Pick\nSide',
                     'Rec\nConsensus', 'Sharp\nConsensus', 'Rec\nMove', 'Sharp\nMove',
                     'Sharp\nBet\nScore', 'Enhanced\nConf.\nScore']

        st.dataframe(
            filtered_df[[col for col in view_cols if col in filtered_df.columns]]
            .sort_values(by='Date\n+ Time (EST)', na_position='last'),
            use_container_width=True
        )

        # === Live Odds Pivot Table
        st.subheader(f"📋 Live Odds Snapshot – {label} (Odds + Limit)")
        odds_rows = []
        for game in live:
            game_name = f"{game['home_team']} vs {game['away_team']}"
            game_start = pd.to_datetime(game.get("commence_time")) if game.get("commence_time") else pd.NaT
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    for o in market.get("outcomes", []):
                        price = o.get('point') if market['key'] != 'h2h' else o.get('price')
                        odds_rows.append({
                            "Game": game_name,
                            "Market": market['key'],
                            "Outcome": o["name"],
                            "Bookmaker": book["title"],
                            "Value": price,
                            "Limit": o.get("bet_limit", 0),
                            "Game_Start": game_start
                        })

        df_odds_raw = pd.DataFrame(odds_rows)
        if not df_odds_raw.empty:
            df_odds_raw['Value_Limit'] = df_odds_raw.apply(
                lambda r: f"{round(r['Value'], 1)} ({int(r['Limit'])})" if pd.notnull(r['Limit']) and pd.notnull(r['Value'])
                else "" if pd.isnull(r['Value']) else f"{round(r['Value'], 1)}",
                axis=1
            )

         
            
            eastern = pytz_timezone('US/Eastern')
            
            df_odds_raw['Date + Time (EST)'] = pd.to_datetime(df_odds_raw['Game_Start'], errors='coerce').apply(
                lambda x: x.tz_convert(eastern).strftime('%Y-%m-%d %I:%M %p') if pd.notnull(x) and x.tzinfo
                else pd.to_datetime(x).tz_localize('UTC').tz_convert(eastern).strftime('%Y-%m-%d %I:%M %p') if pd.notnull(x)
                else ""
            )

            df_display = df_odds_raw.pivot_table(
                index=["Date + Time (EST)", "Game", "Market", "Outcome"],
                columns="Bookmaker",
                values="Value_Limit",
                aggfunc="first"
            ).reset_index()

            sharp_books = ['Pinnacle', 'Bookmaker', 'BetOnline']
            def highlight_sharp(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                for col in df.columns:
                    if col in sharp_books:
                        styles[col] = 'background-color: #d0f0c0; color: black'
                return styles

            st.dataframe(df_display.style.apply(highlight_sharp, axis=None), use_container_width=True)

        return df_moves




# Safe predefinition
df_nba_bt = pd.DataFrame()
df_mlb_bt = pd.DataFrame()



def render_sharp_signal_analysis_tab(tab, sport_label, sport_key_api):
    with tab:
        st.subheader(f"📈 Backtest Performance – {sport_label}")
        sport_key_lower = sport_key_api

        # ✅ 1. Load recent sharp picks from BigQuery
        df_master = read_recent_sharp_moves(hours=168)
        if df_master.empty:
            st.warning(f"⚠️ No sharp picks found in BigQuery.")
            return

        # ✅ 2. Filter for the current sport
        if 'Sport' not in df_master.columns:
            st.warning("⚠️ Missing 'Sport' column in sharp picks. Skipping.")
            return

        df_master = df_master[df_master['Sport'] == sport_label.upper()]
        if df_master.empty:
            st.warning(f"⚠️ No data for {sport_label}.")
            return

        # ✅ 3. Fetch updated scores (avoids stale SHARP_HIT_BOOLs)
        df_bt = fetch_scores_and_backtest(sport_key_api, df_master.copy(), api_key=API_KEY)
        if df_bt.empty:
            st.warning("⚠️ No backtest data to evaluate.")
            return

        if 'SHARP_HIT_BOOL' not in df_bt.columns:
            st.warning("⚠️ Missing SHARP_HIT_BOOL. Skipping backtest summary.")
            return

        # === 4. Attach derived confidence tiers
        df_bt['SharpConfidenceTier'] = pd.cut(
            df_bt['SharpBetScore'],
            bins=[0, 15, 25, 40, 100],
            labels=["⚠️ Low", "✅ Moderate", "⭐ High", "🔥 Steam"]
        )

        # ✅ 5. Filter to only completed (scored) picks
        scored = df_bt[df_bt['SHARP_HIT_BOOL'].notna()]
        if scored.empty:
            st.warning("ℹ️ No completed sharp picks available for analysis.")
            return

        # === 6. Signal Leaderboard
        st.subheader(f"🏆 Top Sharp Signal Performers by Market ({sport_label})")
        leaderboard_rows = []
        for comp in component_fields:
            if comp in scored.columns:
                group = (
                    scored.groupby(['Market', comp])['SHARP_HIT_BOOL']
                    .agg(['count', 'mean'])
                    .reset_index()
                    .rename(columns={
                        'count': 'Signal_Count',
                        'mean': 'Win_Rate',
                        comp: 'Component_Value'
                    })
                )
                group['Component'] = comp
                leaderboard_rows.append(group)

        if leaderboard_rows:
            leaderboard_df = pd.concat(leaderboard_rows, ignore_index=True)
            leaderboard_df = leaderboard_df[[
                'Market', 'Component', 'Component_Value', 'Signal_Count', 'Win_Rate'
            ]].sort_values(by='Win_Rate', ascending=False)

            st.dataframe(leaderboard_df.head(50))
        else:
            st.info("ℹ️ No signal components available to summarize.")

        # === 7. Confidence Tier Summary
        st.subheader(f"📊 Confidence Tier Performance by Market")
        df_market_tier_summary = (
            scored
            .groupby(['Market', 'SharpConfidenceTier'])
            .agg(
                Total_Picks=('SHARP_HIT_BOOL', 'count'),
                Hits=('SHARP_HIT_BOOL', 'sum'),
                Win_Rate=('SHARP_HIT_BOOL', 'mean')
            )
            .reset_index()
            .round(3)
        )
        if not df_market_tier_summary.empty:
            st.dataframe(df_market_tier_summary)
        else:
            st.info("ℹ️ No market-tier breakdown available.")

        # === 8. Learn Market Weights (to be saved to BigQuery)
        st.subheader("🧠 Sharp Component Learning by Market")
        market_component_win_rates_sport = {}
        for comp in component_fields:
            if comp in scored.columns:
                result = (
                    scored.groupby(['Market', comp])['SHARP_HIT_BOOL']
                    .mean()
                    .reset_index()
                    .rename(columns={'SHARP_HIT_BOOL': 'Win_Rate'})
                    .sort_values(by=['Market', comp])
                )
                st.markdown(f"**📊 {comp} by Market**")
                st.dataframe(result)

                for _, row in result.iterrows():
                    market = str(row['Market']).lower()
                    val = row[comp]
                    win_rate = max(0.5, row['Win_Rate'])  # clamp floor

                    # Normalize key
                    if pd.isna(val):
                        continue
                    elif isinstance(val, bool):
                        val_key = str(val).lower()
                    elif isinstance(val, float) and val.is_integer():
                        val_key = str(int(val))
                    else:
                        val_key = str(val).lower()

                    market_component_win_rates_sport \
                        .setdefault(market, {}) \
                        .setdefault(comp, {})[val_key] = win_rate

        # === 9. Save learned weights
        if market_component_win_rates_sport:
            try:
                all_weights = globals().get("market_component_win_rates", {})
                all_weights[sport_key_lower] = market_component_win_rates_sport
                globals()["market_component_win_rates"] = all_weights

                write_market_weights_to_bigquery(all_weights)
                st.success(f"✅ Saved learned weights for {sport_label} to BigQuery")
            except Exception as e:
                st.error(f"❌ Failed to save weights: {e}")
        else:
            st.info("ℹ️ No weight data generated.")

        # === 10. Debug View
        st.subheader(f"📥 Current Learned Weights ({sport_label})")
        st.json(globals().get("market_component_win_rates", {}).get(sport_key_lower, {}))

        st.subheader(f"🧪 Sample {sport_label} Confidence Inputs")
        sample_cols = ['Market', 'Sharp_Move_Signal', 'Sharp_Time_Score', 'True_Sharp_Confidence_Score']
        st.dataframe(scored[sample_cols].head(10) if all(c in scored.columns for c in sample_cols) else scored.head(10))


tab_nba, tab_mlb = st.tabs(["🏀 NBA", "⚾ MLB"])

with tab_nba:


    with st.expander("📊 Real-Time Sharp Scanner", expanded=True):
        df_nba_live = render_scanner_tab("NBA", SPORTS["NBA"], tab_nba)


with tab_mlb:
   
    with st.expander("📊 Real-Time Sharp Scanner", expanded=True):
        df_mlb_live = render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb)
