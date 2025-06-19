import pandas as pd
from datetime import datetime
from collections import defaultdict
from google.cloud import bigquery, storage
from pandas_gbq import to_gbq
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import requests
import numpy as np
import logging
import hashlib

import pickle  # ✅ Add this at the top of your script
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from google.cloud import bigquery, storage

# === Config ===
GCP_PROJECT_ID = "sharplogger"
BQ_DATASET = "sharp_data"
BQ_TABLE = "sharp_moves_master"
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
MARKET_WEIGHTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.market_weights"
LINE_HISTORY_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.line_history_master"
SNAPSHOTS_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET}.odds_snapshot_log"
GCS_BUCKET = "sharp-models"
API_KEY = "3879659fe861d68dfa2866c211294684"
bq_client = bigquery.Client(project=GCP_PROJECT_ID)
gcs_client = storage.Client(project=GCP_PROJECT_ID)

SPORTS = {
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "CFL": "americanfootball_cfl",
    "WNBA": "basketball_wnba",
}

SHARP_BOOKS_FOR_LIMITS = ['pinnacle']
SHARP_BOOKS = SHARP_BOOKS_FOR_LIMITS + ['betfair_ex_eu','betfair_ex_uk', 'smarkets','betonlineag','lowvig']

REC_BOOKS = [
    'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers',
    'fanatics', 'espnbet', 'hardrockbet', 'bovada', 'betus'
]

BOOKMAKER_REGIONS = {
    'pinnacle': 'eu', 'betfair_ex_eu': 'eu', 'betfair_ex_uk': 'uk', 'smarkets': 'uk',
    'matchbook': 'uk', 'betonlineag': 'us', 'lowvig': 'us', 'betanysports': 'us2', 'betus': 'us',
    'betmgm': 'us', 'draftkings': 'us', 'fanduel': 'us', 'betrivers': 'us', 'espnbet': 'us2',
    'hardrockbet': 'us2', 'fanatics': 'us', 'mybookieag': 'us', 'bovada': 'us', 'rebet': 'us2',
    'windcreek': 'us2', 'bet365': 'uk', 'williamhill': 'uk', 'ladbrokes': 'uk', 'unibet': 'eu',
    'bwin': 'eu', 'sportsbet': 'au', 'ladbrokesau': 'au', 'neds': 'au'
}

MARKETS = ['spreads', 'totals', 'h2h']

# === Utility Functions ===

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

def normalize_team(t):
    return str(t).strip().lower().replace('.', '').replace('&', 'and')

def build_merge_key(home, away, game_start):
    return f"{normalize_team(home)}_{normalize_team(away)}_{game_start.floor('h').strftime('%Y-%m-%d %H:%M:%S')}"


def compute_line_hash(row):
    key = f"{row['Game_Key']}|{row['Bookmaker']}|{row['Market']}|{row['Outcome']}|{row['Value']}|{row.get('Limit', '')}|{row.get('Sharp_Move_Signal', '')}"
    return hashlib.md5(key.encode()).hexdigest()
def build_game_key(df):
    required = ['Game', 'Game_Start', 'Market', 'Outcome']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"⚠️ Missing columns in build_game_key: {missing}")
        return df

    df = df.copy()
    df['Home_Team_Norm'] = df['Game'].str.extract(r'^(.*?) vs')[0].str.strip().str.lower()
    df['Away_Team_Norm'] = df['Game'].str.extract(r'vs (.*)$')[0].str.strip().str.lower()
    df['Commence_Hour'] = pd.to_datetime(df['Game_Start'], errors='coerce', utc=True).dt.floor('h')
    df['Market_Norm'] = df['Market'].str.strip().str.lower()
    df['Outcome_Norm'] = df['Outcome'].str.strip().str.lower()
    df['Game_Key'] = (
        df['Home_Team_Norm'] + "_" + df['Away_Team_Norm'] + "_" +
        df['Commence_Hour'].astype(str) + "_" + df['Market_Norm'] + "_" + df['Outcome_Norm']
    )
    df['Merge_Key_Short'] = df.apply(
        lambda row: build_merge_key(row['Home_Team_Norm'], row['Away_Team_Norm'], row['Commence_Hour']),
        axis=1
    )
    return df

def fetch_live_odds(sport_key, api_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us,us2,uk,eu,au',
        'markets': ','.join(MARKETS),
        'oddsFormat': 'american',
        'includeBetLimits': 'true'
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def safe_to_gbq(df, table, replace=False):
    mode = 'replace' if replace else 'append'
    try:
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists=mode)
        return True
    except Exception as e:
        print(f"❌ Failed to upload to {table}: {e}")
        return False

def write_parquet_to_gcs(df, filename, bucket_name=GCS_BUCKET, folder="snapshots/"):
    if df.empty:
        print("⚠️ No data to write.")
        return
    table = pa.Table.from_pandas(df)
    buffer = BytesIO()
    pq.write_table(table, buffer, compression='snappy')
    blob_path = f"{folder}{filename}"
    blob = gcs_client.bucket(bucket_name).blob(blob_path)
    blob.upload_from_string(buffer.getvalue(), content_type="application/octet-stream")
    print(f"✅ Uploaded Parquet to gs://{bucket_name}/{blob_path}")


def write_snapshot_to_gcs_parquet(snapshot_list, bucket_name="sharp-models", folder="snapshots/"):
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

 

    # Build Game_Key in df_snap using the same function as df_moves_raw
    df_snap = pd.DataFrame(rows)

    # ✅ Only run build_game_key if required fields exist
    required_fields = {'Game', 'Game_Start', 'Market', 'Outcome'}
    if required_fields.issubset(df_snap.columns):
        df_snap = build_game_key(df_snap)
    else:
        missing = required_fields - set(df_snap.columns)
        logging.warning(f"⚠️ Skipping build_game_key — missing columns: {missing}")

    if df_snap.empty:
        logging.warning("⚠️ No snapshot data to upload to GCS.")
        return

    filename = f"{folder}{snapshot_time.strftime('%Y%m%d_%H%M%S')}_snapshot.parquet"
    buffer = BytesIO()

    try:
        table = pa.Table.from_pandas(df_snap)
        pq.write_table(table, buffer, compression='snappy')
    except Exception as e:
        logging.exception("❌ Failed to write snapshot DataFrame to Parquet.")
        return

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_string(buffer.getvalue(), content_type='application/octet-stream')
        logging.info(f"✅ Snapshot uploaded to GCS: gs://{bucket_name}/{filename}")
    except Exception as e:
        logging.exception("❌ Failed to upload snapshot to GCS.")

def read_latest_snapshot_from_bigquery(hours=2):
    try:
        query = f"""
            SELECT * FROM `{SNAPSHOTS_TABLE}`
            WHERE Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
        """
        df = bq_client.query(query).to_dataframe()

        # Restructure into grouped dict format expected by detect_sharp_moves
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

        return dict(grouped)
    except Exception as e:
        print(f"❌ Failed to load snapshot from BigQuery: {e}")
        return {}



def write_sharp_moves_to_master(df, table='sharp_data.sharp_moves_master'):
    if df is None or df.empty:
        logging.warning("⚠️ No sharp moves to write.")
        return

    df = df.copy()
	
    # ✅ Only keep rows from sharp and rec books
    allowed_books = SHARP_BOOKS + REC_BOOKS
    df = df[df['Book'].isin(allowed_books)]
    
    # Safety: Ensure Game_Key exists
    if 'Game_Key' not in df.columns or df['Game_Key'].isnull().all():
        logging.warning("❌ No valid Game_Key present — skipping upload.")
        logging.debug(df[['Game', 'Game_Start', 'Market', 'Outcome']].head().to_string())
        return

    df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()
    logging.info(f"🧪 Sharp moves ready to write: {len(df)}")

    # Clean column names
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]
    df = df.drop(columns=[col for col in df.columns if col.endswith('_x') or col.endswith('_y')], errors='ignore')

    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce', utc=True)

    # Convert object columns safely
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].where(df[col].notna(), None)
    # === Ensure model scoring fields are preserved
    model_cols = [
        'Model_Sharp_Win_Prob', 'Model_Confidence_Tier', 'SharpBetScore',
        'Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score',
        'Final_Confidence_Score', 'Model_Confidence'
    ]
    for col in model_cols:
        if col not in df.columns:
            df[col] = None  # Fill with nulls if missing
        
    logging.info("🧪 Preview of model columns being written:")
    logging.info(df[model_cols].dropna(how='all').head(5).to_string())


    try:
        logging.info(f"📤 Uploading to `{table}`...")
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='append')
        logging.info(f"✅ Wrote {len(df)} new rows to `{table}`")
    except Exception as e:
        logging.exception(f"❌ Upload to `{table}` failed.")
        logging.debug("Schema:\n" + df.dtypes.to_string())
        logging.debug("Preview:\n" + df.head(5).to_string())


def read_market_weights_from_bigquery():
    try:
        query = f"SELECT * FROM `{MARKET_WEIGHTS_TABLE}`"
        df = bq_client.query(query).to_dataframe()
        weights = {row["Feature"]: row["Weight"] for _, row in df.iterrows()}
        print(f"✅ Loaded {len(weights)} market weights from BigQuery")
        return weights
    except Exception as e:
        print(f"❌ Failed to load market weights from BigQuery: {e}")
        return {}

def detect_sharp_moves(current, previous, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS, weights={}):
    from collections import defaultdict
    import pandas as pd
    from datetime import datetime

    def normalize_label(label):
        return str(label).strip().lower().replace('.0', '')

    def normalize_book_key(raw_key, sharp_books, rec_books):
        for rec in rec_books:
            if rec.replace(" ", "") in raw_key:
                return rec.replace(" ", "")
        for sharp in sharp_books:
            if sharp in raw_key:
                return sharp
        return raw_key

    def implied_prob(odds):
        try:
            odds = float(odds)
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return abs(odds) / (abs(odds) + 100)
        except:
            return None

    def compute_sharp_metrics(entries, open_val, mtype, label):

        move_signal = limit_jump = prob_shift = time_score = 0
        move_magnitude_score = 0.0
        for limit, curr, _ in entries:
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
                time_score += 0.5

        move_magnitude_score = min(move_magnitude_score, 5.0)
        total_limit = sum([l or 0 for l, _, _ in entries])

        return {
            'Sharp_Move_Signal': move_signal,
            'Sharp_Limit_Jump': limit_jump,
            'Sharp_Prob_Shift': prob_shift,
            'Sharp_Time_Score': time_score,
            'Sharp_Limit_Total': total_limit,
            'Sharp_Move_Magnitude_Score': round(move_magnitude_score, 2),
            'SharpBetScore': round(
                2 * move_signal +
                2 * limit_jump +
                1.5 * time_score +
                1.0 * prob_shift +
                0.001 * total_limit +
                3.0 * move_magnitude_score, 2
            )
        }

    def apply_sharp_scoring(rows, sharp_limit_map, line_open_map, sharp_total_limit_map):
        sharp_side_flags = {}
        sharp_metrics_map = {}

        for (game_name, mtype), label_map in sharp_limit_map.items():
            scores = {}
            label_signals = {}
            for label, entries in label_map.items():
                open_val, _ = line_open_map.get((game_name, mtype, label), (None, None))
                metrics = compute_sharp_metrics(entries, open_val, mtype, label)

                scores[label] = metrics['SharpBetScore']
                label_signals[label] = metrics

            if scores:
                best_label = max(scores, key=scores.get)
                sharp_side_flags[(game_name, mtype, best_label)] = 1
                sharp_metrics_map[(game_name, mtype, best_label)] = label_signals.get(best_label, {})

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
                'Sharp_Limit_Total': metrics.get('Sharp_Limit_Total', 0),
                'SharpBetScore': metrics.get('SharpBetScore', 0.0)
            })

        return rows

    # === Begin main parsing
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

    for game in current:
        home_team = game.get('home_team', '').strip().lower()
        away_team = game.get('away_team', '').strip().lower()
        if not home_team or not away_team:
            continue

        game_name = f"{home_team.title()} vs {away_team.title()}"
        event_time = pd.to_datetime(game.get("commence_time"), utc=True, errors='coerce')
        game_hour = event_time.floor('h') if pd.notnull(event_time) else pd.NaT
        gid = game.get('id')
        prev_game = previous_map.get(gid, {})

        for book in game.get('bookmakers', []):
            book_key_raw = book.get('key', '').lower()
            book_key = normalize_book_key(book_key_raw, SHARP_BOOKS, REC_BOOKS)
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

                    rows.append(entry)
                    line_history_log.setdefault(gid, []).append(entry.copy())

                    if val is not None:
                        sharp_lines[(game_name, mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit, val, old_val))
                        if book_key in SHARP_BOOKS:
                            sharp_total_limit_map[(game_name, mtype, label)] += limit or 0
                        if (game_name, mtype, label) not in line_open_map:
                            line_open_map[(game_name, mtype, label)] = (val, snapshot_time)

    # 🔁 REFACTORED SHARP SCORING
    rows = apply_sharp_scoring(rows, sharp_limit_map, line_open_map, sharp_total_limit_map)


    df_sharp_lines = pd.DataFrame(sharp_lines.values())



    
       # === Build main DataFrame
    df = pd.DataFrame(rows)
    df['Book'] = df['Book'].str.lower()
    df['Event_Date'] = pd.to_datetime(df['Game_Start'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # === Historical sorting for open-line extraction
    df_history = df.copy()
    df_history_sorted = df_history.sort_values('Time')
    
    # === Extract open lines globally and per book
    line_open_df = (
        df_history_sorted.dropna(subset=['Value'])
        .groupby(['Game', 'Market', 'Outcome'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Value'})
    )
    
    line_open_per_book = (
        df_history_sorted.dropna(subset=['Value'])
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Book_Value'})
    )
    
    open_limit_df = (
        df_history_sorted
        .dropna(subset=['Limit'])
        .groupby(['Game', 'Market', 'Outcome', 'Book'])['Limit']
        .first()
        .reset_index()
        .rename(columns={'Limit': 'Opening_Limit'})
    )
    
    # === Merge open lines into df
    df = df.merge(line_open_df, on=['Game', 'Market', 'Outcome'], how='left')
    df = df.merge(line_open_per_book, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df = df.merge(open_limit_df, on=['Game', 'Market', 'Outcome', 'Book'], how='left')
    df['Delta vs Sharp'] = df['Value'] - df['Open_Value']
    df['Delta'] = pd.to_numeric(df['Delta vs Sharp'], errors='coerce')
    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)
    
    # === Additional sharp flags
    df['Limit_Jump'] = (df['Limit'] >= 2500).astype(int)
    df['Sharp_Timing'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour.apply(
        lambda h: 1.0 if 6 <= h <= 11 else 0.5 if h <= 15 else 0.2
    )
    df['Limit_NonZero'] = df['Limit'].where(df['Limit'] > 0)
    df['Limit_Max'] = df.groupby(['Game', 'Market'])['Limit_NonZero'].transform('max')
    df['Limit_Min'] = df.groupby(['Game', 'Market'])['Limit_NonZero'].transform('min')

  

    
   # === Detect market leaders
    market_leader_flags = detect_market_leaders(df_history, SHARP_BOOKS, REC_BOOKS)
    df = df.merge(
        market_leader_flags[['Game', 'Market', 'Outcome', 'Book', 'Market_Leader']],
        on=['Game', 'Market', 'Outcome', 'Book'],
        how='left'
    )
    
    # === Flag Pinnacle no-move behavior
    df['Is_Pinnacle'] = df['Book'] == 'pinnacle'
    df['LimitUp_NoMove_Flag'] = (
        (df['Is_Pinnacle']) &
        (df['Limit'] >= 2500) &
        (df['Value'] == df['Open_Value'])
    ).astype(int)
    
    # === Cross-market support (optional)
    df = detect_cross_market_sharp_support(df)
    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Unique_Sharp_Books'] = df['Unique_Sharp_Books'].fillna(0).astype(int)
    df['LimitUp_NoMove_Flag'] = df['LimitUp_NoMove_Flag'].fillna(False).astype(int)
    df['Market_Leader'] = df['Market_Leader'].fillna(False).astype(int)
    
    # === Confidence scores and tiers
    df = assign_confidence_scores(df, weights)
    
    # === Summary consensus metrics
    summary_df = summarize_consensus(df, SHARP_BOOKS, REC_BOOKS)
    
    # ✅ Final return (no field names changed)
    return df, df_history, summary_df

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
    base_score = min(row.get('SharpBetScore', 0) / 50, 1.0) * 50
    weight_score = compute_weighted_signal(row, market_weights)

    limit_position_bonus = 0
    if row.get('LimitUp_NoMove_Flag') == 1:
        limit_position_bonus = 15
    elif row.get('Limit_Jump') == 1 and abs(row.get('Delta vs Sharp', 0)) > 0.25:
        limit_position_bonus = 5

    market_lead_bonus = 5 if row.get('Market_Leader') else 0

    final_conf = base_score + weight_score + limit_position_bonus + market_lead_bonus
    return round(min(final_conf, 100), 2)


# === Outside of detect_sharp_moves ===
def assign_confidence_scores(df, market_weights):
    df['True_Sharp_Confidence_Score'] = df.apply(
        lambda r: compute_weighted_signal(r, market_weights), axis=1
    )
    df['Enhanced_Sharp_Confidence_Score'] = df.apply(
        lambda r: compute_confidence(r, market_weights), axis=1
    )
    df['Sharp_Confidence_Tier'] = pd.cut(
        df['Enhanced_Sharp_Confidence_Score'],
        bins=[-1, 25, 50, 75, float('inf')],
        labels=['⚠️ Low', '✅ Medium', '⭐ High', '🔥 Steam']
    )
    return df

def summarize_consensus(df, SHARP_BOOKS, REC_BOOKS):
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

    summary_df['Recommended_Outcome'] = summary_df['Outcome']
    summary_df['Move_From_Open_Rec'] = (summary_df['Rec_Book_Consensus'] - summary_df['Rec_Open']).fillna(0)
    summary_df['Move_From_Open_Sharp'] = (summary_df['Sharp_Book_Consensus'] - summary_df['Sharp_Open']).fillna(0)

    sharp_scores = df[df['SharpBetScore'].notnull()][[
        'Event_Date', 'Game', 'Market', 'Outcome',
        'SharpBetScore', 'Enhanced_Sharp_Confidence_Score', 'Sharp_Confidence_Tier'
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

    return summary_df


def write_line_history_to_bigquery(df):
    if df is None or df.empty:
        logging.warning("⚠️ No line history data to upload.")
        return

    df = df.copy()

    # Convert Time to UTC datetime
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce', utc=True)

    df['Snapshot_Timestamp'] = pd.Timestamp.utcnow()

    # Clean merge artifacts
    df = df.rename(columns=lambda x: x.rstrip('_x'))
    df = df.drop(columns=[col for col in df.columns if col.endswith('_y')], errors='ignore')

    # Define the allowed schema columns
    LINE_HISTORY_ALLOWED_COLS = [
        "Sport", "Game_Key", "Time", "Game", "Game_Start", "Event_Date",
        "Market", "Outcome", "Bookmaker", "Book", "Value", "Limit",
        "Old Value", "Delta", "Home_Team_Norm", "Away_Team_Norm", "Commence_Hour",
        "Ref Sharp Value", "Ref Sharp Old Value", "Delta vs Sharp",
        "SHARP_SIDE_TO_BET", "Sharp_Move_Signal", "Sharp_Limit_Jump",
        "Sharp_Time_Score", "Sharp_Prob_Shift", "Sharp_Limit_Total",
        "SharpBetScore", "Snapshot_Timestamp"
    ]

    # ✅ Remove any unexpected columns before upload
    allowed = set(LINE_HISTORY_ALLOWED_COLS)
    actual = set(df.columns)
    extra = actual - allowed
    if extra:
        logging.warning(f"⚠️ Dropping unexpected columns before upload: {extra}")
    df = df[[col for col in LINE_HISTORY_ALLOWED_COLS if col in df.columns]]

    # Log preview
    logging.debug("🧪 Line history dtypes:\n" + str(df.dtypes.to_dict()))
    logging.debug("Sample rows:\n" + df.head(2).to_string())

    # Upload
    if not safe_to_gbq(df, LINE_HISTORY_TABLE):
        logging.error(f"❌ Failed to upload line history to {LINE_HISTORY_TABLE}")
    else:
        logging.info(f"✅ Uploaded {len(df)} line history rows to {LINE_HISTORY_TABLE}.")



        
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

def detect_cross_market_sharp_support(df_moves, score_threshold=25):
    df = df_moves.copy()
    df['SupportKey'] = df['Game'].astype(str) + " | " + df['Outcome'].astype(str)

    df_sharp = df[df['SharpBetScore'] >= score_threshold].copy()

    # Count unique markets
    market_counts = (
        df_sharp.groupby('SupportKey')['Market']
        .nunique()
        .reset_index()
        .rename(columns={'Market': 'CrossMarketSharpSupport'})
    )

    # Count unique sharp bookmakers
    sharp_book_counts = (
        df_sharp[df_sharp['Book'].isin(SHARP_BOOKS)]
        .groupby('SupportKey')['Book']
        .nunique()
        .reset_index()
        .rename(columns={'Book': 'Unique_Sharp_Books'})
    )

    df = df.merge(market_counts, on='SupportKey', how='left')
    df = df.merge(sharp_book_counts, on='SupportKey', how='left')

    df['CrossMarketSharpSupport'] = df['CrossMarketSharpSupport'].fillna(0).astype(int)
    df['Unique_Sharp_Books'] = df['Unique_Sharp_Books'].fillna(0).astype(int)
    df['Is_Reinforced_MultiMarket'] = (
        (df['CrossMarketSharpSupport'] >= 2) | (df['Unique_Sharp_Books'] >= 2)
    )

    return df


def apply_blended_sharp_score(df, trained_models):
    import numpy as np
    import pandas as pd


    logging.info("🔍 Entered apply_blended_sharp_score()")
    df = df.copy()

    try:
        df = df.drop(columns=[col for col in df.columns if col.endswith(('_x', '_y'))], errors='ignore')
    except Exception as e:
        st.error(f"❌ Cleanup failed: {e}")
        return pd.DataFrame()

    try:
        if 'Enhanced_Sharp_Confidence_Score' in df.columns:
            df['Final_Confidence_Score'] = pd.to_numeric(df['Enhanced_Sharp_Confidence_Score'], errors='coerce') / 100
            df['Final_Confidence_Score'] = df['Final_Confidence_Score'].clip(0, 1)
    except Exception as e:
        logging.warning(f"⚠️ Could not compute fallback confidence score: {e}")

    df['Market'] = df['Market'].astype(str).str.lower()

    for market_type, bundle in trained_models.items():
        model = bundle['model']
        iso = bundle['calibrator']
        df_market = df[df['Market'] == market_type].copy()

        if df_market.empty:
            continue

        try:
            model_features = model.get_booster().feature_names

            # Ensure all required features exist and are numeric
            for col in model_features:
                if col not in df_market.columns:
                    df_market[col] = 0
                df_market[col] = (
                    df_market[col]
                    .astype(str)
                    .replace({'True': 1, 'False': 0, 'true': 1, 'false': 0})
                )
                df_market[col] = pd.to_numeric(df_market[col], errors='coerce').fillna(0)

            df_market = df_market[model_features].astype(float)

            # Predict with calibration
            raw_probs = model.predict_proba(df_market)[:, 1]
            calibrated_probs = iso.predict(raw_probs)

            df.loc[df['Market'] == market_type, 'Model_Sharp_Win_Prob'] = raw_probs
            df.loc[df['Market'] == market_type, 'Model_Confidence'] = calibrated_probs

        except Exception as e:
            logging.error(f"❌ Failed to apply model for {market_type}: {e}")

    # Apply tiering only if Model_Sharp_Win_Prob exists
    if 'Model_Sharp_Win_Prob' in df.columns:
        df['Model_Confidence'] = df['Model_Confidence'].fillna(0).clip(0, 1)
        df['Model_Confidence_Tier'] = pd.cut(
            df['Model_Sharp_Win_Prob'],
            bins=[0.0, 0.4, 0.5, 0.6, 1.0],
            labels=["⚠️ Weak Indication", "✅ Coinflip", "⭐ Lean", "🔥 Strong Indication"]
        )

    #st.success("✅ Model scoring complete (per-market)")
    return df
 


def load_model_from_gcs(sport, market, bucket_name="sharp-models"):
    filename = f"sharp_win_model_{sport.lower()}_{market.lower()}.pkl"
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        content = blob.download_as_bytes()
        data = pickle.loads(content)

        print(f"✅ Loaded model + calibrator from GCS: gs://{bucket_name}/{filename}")
        return {
            "model": data["model"],
            "calibrator": data["calibrator"]
        }
    except Exception as e:
        logging.warning(f"⚠️ Could not load model from GCS for {sport}-{market}: {e}")
        return None

def read_recent_sharp_moves(hours=500, table=BQ_FULL_TABLE):
    try:
        client = bq_client
        query = f"""
            SELECT * FROM `{table}`
            WHERE Snapshot_Timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
        """
        df = client.query(query).to_dataframe()
        df['Commence_Hour'] = pd.to_datetime(df['Commence_Hour'], errors='coerce', utc=True)

        print(f"✅ Loaded {len(df)} rows from BigQuery (last {hours}h)")
        return df
    except Exception as e:
        print(f"❌ Failed to read from BigQuery: {e}")
        return pd.DataFrame()
       

def write_to_bigquery(df, table='sharp_data.sharp_scores_full', force_replace=False):
    from pandas_gbq import to_gbq

    if df.empty:
        logging.info("ℹ️ No data to write to BigQuery.")
        return

    df = df.copy()
    df.columns = [col.replace(" ", "_") for col in df.columns]

    # Drop unapproved fields (BigQuery strict schema match)
    allowed_cols = {
        'sharp_data.sharp_scores_full': [
            'Game_Key', 'Bookmaker', 'Market', 'Outcome', 'Ref_Sharp_Value',
            'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
            'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
            'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
            'Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score',
            'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT', 'Scored', 'Snapshot_Timestamp'
        ],
        'sharp_data.sharp_moves_master': None  # Add allowed list here if needed
    }
    if table in allowed_cols and allowed_cols[table] is not None:
        df = df[[col for col in df.columns if col in allowed_cols[table]]]

    try:
        to_gbq(df, table, project_id=GCP_PROJECT_ID, if_exists='replace' if force_replace else 'append')
        logging.info(f"✅ Uploaded {len(df)} rows to {table}")
    except Exception as e:
        logging.exception(f"❌ Failed to upload to {table}")
  
def fetch_scores_and_backtest(sport_key, df_moves=None, days_back=3, api_key=API_KEY, trained_models=None):

    expected_label = [k for k, v in SPORTS.items() if v == sport_key]
    sport_label = expected_label[0].upper() if expected_label else "NBA"

    # === 1. Fetch completed games ===
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
        response = requests.get(url, params={'apiKey': api_key, 'daysFrom': int(days_back)}, timeout=10)
        response.raise_for_status()
        games = response.json()
    except Exception as e:
        logging.error(f"❌ Failed to fetch scores: {e}")
        return pd.DataFrame()

    completed_games = [g for g in games if g.get("completed")]
    logging.info(f"✅ Completed games: {len(completed_games)}")

    # === 2. Extract valid score rows ===
    score_rows = []
    for game in completed_games:
        home = normalize_team(game.get("home_team", ""))
        away = normalize_team(game.get("away_team", ""))
        game_start = pd.to_datetime(game.get("commence_time"), utc=True)
        merge_key = build_merge_key(home, away, game_start)
        scores = {s.get("name", "").strip().lower(): s.get("score") for s in game.get("scores", [])}
        if home in scores and away in scores:
            score_rows.append({
                'Merge_Key_Short': merge_key,
                'Home_Team': home,
                'Away_Team': away,
                'Game_Start': game_start,
                'Score_Home_Score': scores[home],
                'Score_Away_Score': scores[away],
                'Source': 'oddsapi',
                'Inserted_Timestamp': pd.Timestamp.utcnow()
            })

    df_scores = pd.DataFrame(score_rows).dropna(subset=['Merge_Key_Short', 'Game_Start'])
    df_scores = df_scores.drop_duplicates(subset=['Merge_Key_Short'])
    df_scores['Score_Home_Score'] = pd.to_numeric(df_scores['Score_Home_Score'], errors='coerce')
    df_scores['Score_Away_Score'] = pd.to_numeric(df_scores['Score_Away_Score'], errors='coerce')
    df_scores = df_scores.dropna(subset=['Score_Home_Score', 'Score_Away_Score'])

    # === 3. Upload scores to `game_scores_final` ===
    try:
        existing_keys = bq_client.query("""
            SELECT DISTINCT Merge_Key_Short FROM `sharp_data.game_scores_final`
        """).to_dataframe()
        existing_keys = set(existing_keys['Merge_Key_Short'].dropna())
        new_scores = df_scores[~df_scores['Merge_Key_Short'].isin(existing_keys)].copy()
       
 

        pass#to_gbq(new_scores, 'sharp_data.game_scores_final', project_id=GCP_PROJECT_ID, if_exists='append')
        pass#st.success(f"✅ Uploaded {len(new_scores)} new game scores")
    except Exception as e:
        pass#st.error(f"❌ Failed to upload game scores: {e}")
        pass#st.code(new_scores.dtypes.to_string())

    # === 4. Load recent sharp picks
    df_master = read_recent_sharp_moves(hours=days_back * 72)
    df_master = build_game_key(df_master)
    df_master = ensure_columns(df_master, ['Game_Start'])
    df_master = df_master[df_master['Merge_Key_Short'].isin(df_scores['Merge_Key_Short'])]

    if df_master.empty:
        logging.warning("⚠️ No sharp picks to backtest")
        return pd.DataFrame()

    # === 5. Merge scores and filter
    df = df_master.merge(
        df_scores[['Merge_Key_Short', 'Score_Home_Score', 'Score_Away_Score']],
        on='Merge_Key_Short', how='inner'
    )
    df = df[df['Book'].isin(SHARP_BOOKS + REC_BOOKS)]
    df = df[pd.to_datetime(df['Game_Start'], utc=True, errors='coerce') < pd.Timestamp.utcnow()]

    if 'Ref_Sharp_Value' not in df.columns:
        df['Ref_Sharp_Value'] = df.get('Value')
    else:
        df['Ref_Sharp_Value'] = df['Ref_Sharp_Value'].combine_first(df.get('Value'))

    # === 6. Calculate result
    df_valid = df.dropna(subset=['Score_Home_Score', 'Score_Away_Score', 'Ref_Sharp_Value'])
    if df_valid.empty:
        logging.warning("ℹ️ No valid sharp picks with scores to evaluate")
        return pd.DataFrame()

    # === 6. Calculate result
    def calc_cover(row):
        try:
            h = float(row['Score_Home_Score'])
            a = float(row['Score_Away_Score'])
            val = float(row['Ref_Sharp_Value'])
            market = str(row.get('Market', '')).lower()
            outcome = str(row.get('Outcome', '')).lower()
    
            if market == 'totals':
                if 'under' in outcome and h + a < val:
                    return ['Win', 1]
                elif 'over' in outcome and h + a > val:
                    return ['Win', 1]
                else:
                    return ['Loss', 0]
    
            if market == 'spreads':
                margin = h - a if row['Home_Team_Norm'] in outcome else a - h
                hit = (margin > abs(val)) if val < 0 else (margin + val > 0)
                return ['Win', 1] if hit else ['Loss', 0]
    
            if market == 'h2h':
                home_win = row['Home_Team_Norm'] in outcome and h > a
                away_win = row['Away_Team_Norm'] in outcome and a > h
                return ['Win', 1] if home_win or away_win else ['Loss', 0]
    
            return [None, 0]
        except Exception:
            return [None, 0]
    
    result = df_valid.apply(calc_cover, axis=1, result_type='expand')
    result.columns = ['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']
    df['SHARP_COVER_RESULT'] = None
    df['SHARP_HIT_BOOL'] = None
    df['Scored'] = False
    df.loc[df_valid.index, ['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']] = result
    df.loc[df_valid.index, 'Scored'] = result['SHARP_COVER_RESULT'].notna()
    # Ensure 'Unique_Sharp_Books' is present and numeric
    if 'Unique_Sharp_Books' not in df.columns:
        df['Unique_Sharp_Books'] = 0
    df['Unique_Sharp_Books'] = pd.to_numeric(df['Unique_Sharp_Books'], errors='coerce').fillna(0).astype(int)

    
   # === 8. Final output DataFrame ===
    score_cols = [
        'Game_Key', 'Bookmaker', 'Market', 'Outcome', 'Ref_Sharp_Value',
        'Sharp_Move_Signal', 'Sharp_Limit_Jump', 'Sharp_Prob_Shift',
        'Sharp_Time_Score', 'Sharp_Limit_Total', 'Is_Reinforced_MultiMarket',
        'Market_Leader', 'LimitUp_NoMove_Flag', 'SharpBetScore',
        'Unique_Sharp_Books',
        'Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score',
        'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT', 'Scored', 'Sport'
    ]

    
    # Build full output
    df_scores_out = ensure_columns(df, score_cols)[score_cols].copy()
    df_scores_out['Sport'] = sport_label.upper()
    df_scores_out['Snapshot_Timestamp'] = pd.Timestamp.utcnow()  # ✅ Only do this once here
    # === Coerce and clean all fields BEFORE dedup and upload
    df_scores_out['Sharp_Move_Signal'] = pd.to_numeric(df_scores_out['Sharp_Move_Signal'], errors='coerce').astype('Int64')
    df_scores_out['Sharp_Limit_Jump'] = pd.to_numeric(df_scores_out['Sharp_Limit_Jump'], errors='coerce').astype('Int64')
    df_scores_out['Sharp_Prob_Shift'] = pd.to_numeric(df_scores_out['Sharp_Prob_Shift'], errors='coerce').astype('Int64')
    df_scores_out['Sharp_Time_Score'] = pd.to_numeric(df_scores_out['Sharp_Time_Score'], errors='coerce')
    df_scores_out['Sharp_Limit_Total'] = pd.to_numeric(df_scores_out['Sharp_Limit_Total'], errors='coerce')
    df_scores_out['Is_Reinforced_MultiMarket'] = df_scores_out['Is_Reinforced_MultiMarket'].fillna(False).astype(bool)
    df_scores_out['Market_Leader'] = df_scores_out['Market_Leader'].fillna(False).astype(bool)
    df_scores_out['LimitUp_NoMove_Flag'] = df_scores_out['LimitUp_NoMove_Flag'].fillna(False).astype(bool)
    df_scores_out['SharpBetScore'] = pd.to_numeric(df_scores_out['SharpBetScore'], errors='coerce')
    df_scores_out['Enhanced_Sharp_Confidence_Score'] = pd.to_numeric(df_scores_out['Enhanced_Sharp_Confidence_Score'], errors='coerce')
    df_scores_out['True_Sharp_Confidence_Score'] = pd.to_numeric(df_scores_out['True_Sharp_Confidence_Score'], errors='coerce')
    df_scores_out['SHARP_HIT_BOOL'] = pd.to_numeric(df_scores_out['SHARP_HIT_BOOL'], errors='coerce').astype('Int64')
    df_scores_out['SHARP_COVER_RESULT'] = df_scores_out['SHARP_COVER_RESULT'].fillna('').astype(str)
    df_scores_out['Scored'] = df_scores_out['Scored'].fillna(False).astype(bool)
    df_scores_out['Sport'] = df_scores_out['Sport'].fillna('').astype(str)
    df_scores_out['Unique_Sharp_Books'] = pd.to_numeric(df_scores_out['Unique_Sharp_Books'], errors='coerce').fillna(0).astype(int)

  
 
        # Debug: ensure schema matches
    try:
        import pyarrow as pa
        pa.Table.from_pandas(df_scores_out)
    except Exception as e:
        logging.error("❌ Parquet validation failed before upload")
        logging.code(str(e))
        logging.write(df_scores_out.dtypes)
        logging.stop()
    # ✅ Define full deduplication fingerprint (ignore timestamp)
    dedup_fingerprint_cols = score_cols.copy()  # includes all except timestamp
    
    # ✅ Remove local exact duplicates before querying BigQuery
    df_scores_out = df_scores_out.drop_duplicates(subset=dedup_fingerprint_cols)
    
    # ✅ Query BigQuery for existing fingerprints (same line state, any time)
    existing = bq_client.query(f"""
        SELECT DISTINCT {', '.join(dedup_fingerprint_cols)}
        FROM `sharp_data.sharp_scores_full`
        WHERE DATE(Snapshot_Timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
    """).to_dataframe()
    
    # ✅ Remove already-existing line states (not new even if timestamp is different)
    df_scores_out = df_scores_out.merge(
        existing,
        on=dedup_fingerprint_cols,
        how='left',
        indicator=True
    )
    df_scores_out = df_scores_out[df_scores_out['_merge'] == 'left_only'].drop(columns=['_merge'])
    
    if df_scores_out.empty:
        logging.info("ℹ️ No new scored picks to upload — all identical line states already in BigQuery.")
        return df, pd.DataFrame()

    from pandas_gbq import to_gbq
    to_gbq(
        df_scores_out,
        destination_table='sharp_data.sharp_scores_full',
        project_id=GCP_PROJECT_ID,
        if_exists='append'
    )
    logging.info(f"✅ Uploaded {len(df_scores_out)} new scored picks to `sharp_data.sharp_scores_full`")

    return df

          

