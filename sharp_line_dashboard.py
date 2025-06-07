import streamlit as st
import pandas as pd
import requests
import os
import json
import pickle
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from io import StringIO
from collections import defaultdict
import pytz
from collections import OrderedDict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from io import BytesIO  # ✅ Use BytesIO for binary models
import pickle
from datetime import datetime, timedelta, timezone as dt_timezone
from pytz import timezone as pytz_timezone


API_KEY = "3879659fe861d68dfa2866c211294684"

SPORTS = {"NBA": "basketball_nba", "MLB": "baseball_mlb"}

SHARP_BOOKS_FOR_LIMITS = ['pinnacle', 'bookmaker', 'betonlineag']  # limit-based signals
SHARP_BOOKS = SHARP_BOOKS_FOR_LIMITS + ['betfair_ex_eu', 'smarkets', 'matchbook']  # price-based sharp list

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


FOLDER_ID = "1v6WB0jRX_yJT2JSdXRvQOLQNfOZ97iGA"

# 🔁 Shared list of components used for scoring, learning, and tiering
component_fields = OrderedDict({
    'Sharp_Move_Signal': 'Win Rate by Move Signal',
    'Sharp_Time_Score': 'Win Rate by Time Score',
    'Sharp_Limit_Jump': 'Win Rate by Limit Jump',
    'Sharp_Prob_Shift': 'Win Rate by Prob Shift',
    'Is_Reinforced_MultiMarket': 'Win Rate by Cross-Market Reinforcement',
    'Market_Leader': 'Win Rate by Market Leader',
    'LimitUp_NoMove_Flag': 'Win Rate by Limit↑ No Move'
})

market_component_win_rates = {}

def get_snapshot(data):
    return {g['id']: g for g in data}

def init_gdrive():
    try:
        creds_path = "/tmp/service_creds.json"
        with open(creds_path, "w") as f:
            json.dump(dict(st.secrets["gdrive"]), f)

        scope = ['https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)

        gauth = GoogleAuth()
        gauth.auth_method = 'service'
        gauth.credentials = credentials

        return GoogleDrive(gauth)
    except Exception as e:
        st.error(f"❌ Google Drive auth failed: {e}")
        return None
drive = init_gdrive()
def implied_prob(odds):
    try:
        if odds < 0:
            return -odds / (-odds + 100)
        else:
            return 100 / (odds + 100)
    except:
        return None



@st.cache_data(ttl=60)


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


def append_to_master_csv_on_drive(df_new, filename, drive, folder_id):
    from io import StringIO
    from datetime import datetime
    import pandas as pd

    try:
        if df_new.empty:
            print(f"⚠️ Skipping append — {filename} input is empty.")
            return

        # Step 1: Load existing master
        file_list = drive.ListFile({
            'q': f"title='{filename}' and '{folder_id}' in parents and trashed=false"
        }).GetList()

        df_existing = pd.DataFrame()
        if file_list:
            file_drive = file_list[0]
            existing_data = StringIO(file_drive.GetContentString())
            df_existing = pd.read_csv(existing_data)
            file_drive.Delete()
            print(f"📚 Loaded existing {filename} with {len(df_existing)} rows")

        # Step 2: Add batch ID and timestamp to new data
        snapshot_ts = pd.Timestamp.utcnow()
        df_new['Snapshot_Timestamp'] = snapshot_ts
        df_new['Snapshot_ID'] = f"{filename}_{snapshot_ts.strftime('%Y%m%d_%H%M%S')}"

        # Step 3: Combine all rows (no deduplication)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # Step 4: Sort by time for clarity
        df_combined.sort_values(by='Snapshot_Timestamp', inplace=True)
        df_new = build_game_key(df_new)
        # Step 5: Upload to Drive
        csv_buffer = StringIO()
        df_combined.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        new_file = drive.CreateFile({'title': filename, "parents": [{"id": folder_id}]})
        new_file.SetContentString(csv_buffer.getvalue())
        new_file.Upload()

        print(f"✅ Uploaded updated {filename} to Drive — total rows: {len(df_combined)}")

    except Exception as e:
        print(f"❌ Failed to append to {filename}: {e}")

def build_game_key(df):
    """
    Builds a fully unique Game_Key from Game, Game_Start, Market, and Outcome.
    Adds Game_Key, Home_Team_Norm, Away_Team_Norm, Commence_Hour columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing columns: Game, Game_Start, Market, Outcome

    Returns:
        pd.DataFrame: The same DataFrame with a new 'Game_Key' column and supporting normalized fields.
    """
    import pandas as pd

    if not all(col in df.columns for col in ['Game', 'Game_Start', 'Market', 'Outcome']):
        raise ValueError("Missing one or more required columns: ['Game', 'Game_Start', 'Market', 'Outcome']")

    df = df.copy()

    # Normalize and extract components
    df['Home_Team_Norm'] = df['Game'].str.extract(r'^(.*?) vs')[0].str.strip().str.lower()
    df['Away_Team_Norm'] = df['Game'].str.extract(r'vs (.*)$')[0].str.strip().str.lower()
    df['Commence_Hour'] = pd.to_datetime(df['Game_Start'], errors='coerce', utc=True).dt.floor('H')
    df['Market_Norm'] = df['Market'].str.strip().str.lower()
    df['Outcome_Norm'] = df['Outcome'].str.strip().str.lower()

    # Construct full Game_Key
    df['Game_Key'] = (
        df['Home_Team_Norm'] + "_" +
        df['Away_Team_Norm'] + "_" +
        df['Commence_Hour'].astype(str) + "_" +
        df['Market_Norm'] + "_" +
        df['Outcome_Norm']
    )

    return df

def load_master_sharp_moves(drive, filename="sharp_moves_master.csv", folder_id=None):
    import pandas as pd
    from io import StringIO

    try:
        file_list = drive.ListFile({
            'q': f"title='{filename}' and '{folder_id}' in parents and trashed=false"
        }).GetList()

        if not file_list:
            print("⚠️ No master file found.")
            return pd.DataFrame()

        file_drive = file_list[0]
        csv_buffer = StringIO(file_drive.GetContentString())
        df_master = pd.read_csv(csv_buffer)

        # === Patch: Recover missing Game_Key if possible
        try:
            df_master = build_game_key(df_master)
        except ValueError as e:
            st.warning(f"⚠️ Could not build Game_Key: {e}")
        # ✅ Ensure Game_Start is datetime UTC
        df_master['Game_Start'] = pd.to_datetime(df_master['Game_Start'], errors='coerce', utc=True)

        return df_master

    except Exception as e:
        print(f"❌ Failed to load master file: {e}")
        return pd.DataFrame()
        
        
def upload_snapshot_to_drive(sport_key, snapshot, drive, folder_id):
    from io import StringIO
    import json

    filename = f"{sport_key}_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    buffer = StringIO()
    json.dump(snapshot, buffer)
    buffer.seek(0)

    try:
        file_drive = drive.CreateFile({'title': filename, 'parents': [{'id': folder_id}]})
        file_drive.SetContentString(buffer.getvalue())
        file_drive.Upload()
        print(f"✅ Snapshot uploaded to Google Drive: {filename}")
    except Exception as e:
        print(f"❌ Failed to upload snapshot: {e}")


        
def load_latest_snapshot_from_drive(sport_key, drive, folder_id):
    try:
        file_list = drive.ListFile({
            'q': f"title contains '{sport_key}_snapshot_' and '{folder_id}' in parents and trashed=false"
        }).GetList()

        if not file_list:
            print(f"⚠️ No previous snapshot found for {sport_key}")
            return {}

        latest_file = sorted(file_list, key=lambda f: f['title'], reverse=True)[0]
        content = latest_file.GetContentString()
        return json.loads(content)
    except Exception as e:
        print(f"❌ Failed to load snapshot from Drive: {e}")
        return {}

def fetch_scores_and_backtest(sport_key, df_moves, days_back=3, api_key="REPLACE_WITH_KEY"):
    import requests
    import pytz
    import pandas as pd
    from datetime import datetime

    def normalize_team(t):
        return str(t).strip().lower()

    df_moves = df_moves.copy()

    # Rebuild Game_Start if needed
    if 'Game_Start' not in df_moves.columns:
        if 'Event_Date' in df_moves.columns and 'Commence_Hour' in df_moves.columns:
            print("🔄 Rebuilding Game_Start from Event_Date + Commence_Hour...")
            df_moves['Game_Start'] = pd.to_datetime(
                df_moves['Event_Date'].astype(str) + ' ' + df_moves['Commence_Hour'].astype(str),
                errors='coerce',
                utc=True
            )
        else:
            print("⚠️ 'Game_Start' not found and cannot be rebuilt — skipping scoring.")
            df_moves['Scored'] = False
            df_moves['SHARP_COVER_RESULT'] = None
            df_moves['SHARP_HIT_BOOL'] = None
            return df_moves

    df_moves['Game_Start'] = pd.to_datetime(df_moves['Game_Start'], utc=True, errors='coerce')
    now_utc = datetime.now(pytz.utc)
    cutoff = now_utc - pd.Timedelta(days=days_back)
    df_moves = df_moves[
        (df_moves['Game_Start'] < now_utc) & (df_moves['Game_Start'] > cutoff)
    ]

    # Normalize and create Merge_Key_Short
    df_moves['Home_Team_Norm'] = df_moves['Game'].str.extract(r'^(.*?) vs')[0].apply(normalize_team)
    df_moves['Away_Team_Norm'] = df_moves['Game'].str.extract(r'vs (.*)$')[0].apply(normalize_team)
    df_moves['Commence_Hour'] = pd.to_datetime(df_moves['Game_Start'], utc=True).dt.floor('H')
    df_moves['Merge_Key_Short'] = (
        df_moves['Home_Team_Norm'] + "_" +
        df_moves['Away_Team_Norm'] + "_" +
        df_moves['Commence_Hour'].astype(str)
    )

    # Fetch completed scores
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
    params = {'apiKey': api_key, 'daysFrom': days_back}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        games = response.json()
    except Exception as e:
        print(f"⚠️ Failed to fetch scores: {e}")
        return df_moves

    score_rows = []
    for game in games:
        if not game.get("completed"):
            continue

        home = normalize_team(game.get("home_team", ""))
        away = normalize_team(game.get("away_team", ""))
        game_start = pd.to_datetime(game.get("commence_time"), utc=True)
        game_hour = game_start.floor('H')

        scores = game.get("scores", [])
        score_dict = {
            s["name"].strip().lower(): s["score"]
            for s in scores if "name" in s and "score" in s
        }

        home_score = score_dict.get(home)
        away_score = score_dict.get(away)
        if home_score is None or away_score is None:
            continue

        score_rows.append({
            'Merge_Key_Short': f"{home}_{away}_{game_hour}",
            'Score_Home_Score': home_score,
            'Score_Away_Score': away_score
        })

    df_scores = pd.DataFrame(score_rows)
    if df_scores.empty:
        print("🕒 No completed games returned by the Odds API.")
        df_moves['Scored'] = False
        df_moves['SHARP_COVER_RESULT'] = None
        df_moves['SHARP_HIT_BOOL'] = None
        return df_moves

    # === Merge scores into subset
    df_scored_subset = df_moves.merge(df_scores, on='Merge_Key_Short', how='inner')

    # === Apply cover logic to scored rows
    def calc_cover(row):
        from pandas import Series
        try:
            hscore = float(row['Score_Home_Score'])
            ascore = float(row['Score_Away_Score'])
        except:
            return Series([None, None])

        market = str(row.get('Market', '')).lower()
        outcome = str(row.get('Outcome', '')).lower()
        ref_val = row.get('Ref Sharp Value', 0)

        if market == 'totals':
            try:
                total = float(ref_val)
                total_points = hscore + ascore
                if 'under' in outcome:
                    return Series(['Win', 1]) if total_points < total else Series(['Loss', 0])
                elif 'over' in outcome:
                    return Series(['Win', 1]) if total_points > total else Series(['Loss', 0])
            except:
                return Series([None, None])

        team = outcome
        home = str(row.get('Home_Team', '')).lower()
        away = str(row.get('Away_Team', '')).lower()

        if team in home:
            team_score, opp_score = hscore, ascore
        elif team in away:
            team_score, opp_score = ascore, hscore
        else:
            return Series([None, None])

        margin = team_score - opp_score

        if market == 'h2h':
            return Series(['Win', 1]) if margin > 0 else Series(['Loss', 0])

        if market == 'spreads':
            try:
                spread = float(ref_val)
                hit = (margin > abs(spread)) if spread < 0 else (margin + spread > 0)
                return Series(['Win', 1]) if hit else Series(['Loss', 0])
            except:
                return Series([None, None])

        return Series([None, None])

    # Apply scoring
    df_scored_subset[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']] = df_scored_subset.apply(
        calc_cover, axis=1, result_type="expand"
    )

    # Mark those with scores
    df_scored_subset['Scored'] = True

    # === Propagate scores back to full df_moves (with full Game_Key)
    df_moves.update(
        df_scored_subset.set_index(['Game_Key', 'Market', 'Outcome', 'Bookmaker'])[
            ['Score_Home_Score', 'Score_Away_Score', 'SHARP_COVER_RESULT', 'SHARP_HIT_BOOL', 'Scored']
        ]
    )

    return df_moves
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

def save_weights_to_drive(weights, drive, folder_id=FOLDER_ID):
    try:
        # Delete old file
        file_list = drive.ListFile({
            'q': f"title='market_weights.json' and '{folder_id}' in parents and trashed=false"
        }).GetList()
        for file in file_list:
            file.Delete()

        buffer = StringIO()
        json.dump(weights, buffer, indent=2)
        buffer.seek(0)

        new_file = drive.CreateFile({'title': "market_weights.json", "parents": [{"id": folder_id}]})
        new_file.SetContentString(buffer.getvalue())
        new_file.Upload()
        print("✅ Saved market weights to Google Drive.")
    except Exception as e:
        print(f"❌ Failed to save market weights to Drive: {e}")


def load_weights_from_drive(drive, folder_id=FOLDER_ID):
    try:
        file_list = drive.ListFile({
            'q': f"title='market_weights.json' and '{folder_id}' in parents and trashed=false"
        }).GetList()

        if not file_list:
            print("⚠️ No saved market weights found on Google Drive.")
            return {}

        file_drive = file_list[0]
        content = file_drive.GetContentString()
        print("✅ Loaded market weights from Google Drive.")
        return json.loads(content)

    except Exception as e:
        print(f"❌ Failed to load weights from Drive: {e}")
        return {}

market_component_win_rates = load_weights_from_drive(drive)



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
    import json

    def normalize_label(label):
        return str(label).strip().lower().replace('.0', '')

    rows = []
    sharp_limit_map = defaultdict(lambda: defaultdict(list))
    sharp_total_limit_map = defaultdict(int)
    sharp_lines, sharp_side_flags, sharp_metrics_map = {}, {}, {}
    line_history_log = {}
    line_open_map = {}

    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}

    sport_scope_key = {
        'MLB': 'baseball_mlb',
        'NBA': 'basketball_nba'
    }.get(sport_key.upper(), sport_key.lower())

    confidence_weights = weights.get(sport_scope_key, {})

    previous_odds_map = {}
    for g in previous_map.values():
        for book in g.get('bookmakers', []):
            book_key = book['key']
            for market in book.get('markets', []):
                mtype = market.get('key')
                for outcome in market.get('outcomes', []):
                    label = normalize_label(outcome['name'])
                    price = outcome.get('point') if mtype != 'h2h' else outcome.get('price')
                    previous_odds_map[(g['home_team'], g['away_team'], mtype, label, book_key)] = price

    for game in current:
        home_team = game['home_team'].strip().lower()
        away_team = game['away_team'].strip().lower()
        game_name = f"{game['home_team']} vs {game['away_team']}"
        event_time = pd.to_datetime(game.get("commence_time"), utc=True, errors='coerce')
        event_date = event_time.strftime("%Y-%m-%d") if pd.notnull(event_time) else ""
        game_hour = event_time.floor('H') if pd.notnull(event_time) else pd.NaT
        gid = game.get('id')
        prev_game = previous_map.get(gid, {})

        for book in game.get('bookmakers', []):
            book_key_raw = book['key'].lower()
            book_key = book_key_raw

            for rec in REC_BOOKS:
                if rec.replace(" ", "") in book_key_raw:
                    book_key = rec.replace(" ", "")
                    break
            for sharp in SHARP_BOOKS:
                if sharp in book_key_raw:
                    book_key = sharp
                    break

            if book_key not in SHARP_BOOKS and book_key not in REC_BOOKS:
                continue

            book_title = book.get('title')

            for market in book.get('markets', []):
                mtype = market.get('key').strip().lower()
                for o in market.get('outcomes', []):
                    label = normalize_label(o['name'])
                    val = o.get('point') if mtype != 'h2h' else o.get('price')
                    limit = o.get('bet_limit') if o.get('bet_limit') is not None else None
                    prev_key = (game['home_team'], game['away_team'], mtype, label, book_key)
                    old_val = previous_odds_map.get(prev_key)

                    # ✅ Build full Game_Key
                    game_key = f"{home_team}_{away_team}_{str(game_hour)}_{mtype}_{label}"

                    entry = {
                        'Sport': sport_key,
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

                    if prev_game:
                        for prev_b in prev_game.get('bookmakers', []):
                            if prev_b['key'].lower() == book_key_raw:
                                for prev_m in prev_b.get('markets', []):
                                    if prev_m['key'] == mtype:
                                        for prev_o in prev_m.get('outcomes', []):
                                            if normalize_label(prev_o['name']) == label:
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
                            sharp_total_limit_map[(game_name, mtype, label)] += limit if limit is not None else 0

                    if (game_name, mtype, label) not in line_open_map and val is not None:
                        line_open_map[(game_name, mtype, label)] = (val, snapshot_time)



    # === Sharp scoring logic
    for (game_name, mtype), label_map in sharp_limit_map.items():
        scores = {}
        label_signals = {}
        for label, entries in label_map.items():
            move_signal = limit_jump = prob_shift = time_score = 0
            move_magnitude_score = 0
    
            for limit, curr, _ in entries:
                open_val, _ = line_open_map.get((game_name, mtype, label), (None, None))
                if open_val is not None and curr is not None:
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
                        if imp_now and imp_open and imp_now > imp_open: prob_shift += 1
    
                if limit and limit >= 100:
                    limit_jump += 1
    
                hour = datetime.now().hour
                time_score += 1.0 if 6 <= hour <= 11 else 0.5 if hour <= 15 else 0.2
    
            # Optional: cap magnitude boost
            move_magnitude_score = min(move_magnitude_score, 5.0)
    
            total_limit = sharp_total_limit_map.get((game_name, mtype, label), 0)
            scores[label] = (
                2 * move_signal +
                2 * limit_jump +
                1.5 * time_score +
                1.0 * prob_shift +
                0.001 * total_limit +
                3.0 * move_magnitude_score  # ← weighted boost
            )
    
            label_signals[label] = {
                'Sharp_Move_Signal': move_signal,
                'Sharp_Limit_Jump': limit_jump,
                'Sharp_Time_Score': time_score,
                'Sharp_Prob_Shift': prob_shift,
                'Sharp_Limit_Total': total_limit,
                'Sharp_Move_Magnitude_Score': round(move_magnitude_score, 2)  # ← optional tracking
            }

     

        if scores:
            best_label = max(scores, key=scores.get)
            sharp_side_flags[(game_name, mtype, best_label)] = 1
            

            sharp_metrics_map[(game_name, mtype, best_label)] = label_signals[best_label]

    # === Assign sharp-side logic + metric scores to all rows
    # === Assign sharp-side logic + metric scores to all rows
    for row in rows:
        game_name = row['Game']
        mtype = row['Market']
        label = row['Outcome']
    
        metrics = sharp_metrics_map.get((game_name, mtype, label), {})
        is_sharp_side = int(sharp_side_flags.get((game_name, mtype, label), 0))
    
        row.update({
            'Ref Sharp Value': row['Value'],
            'Ref Sharp Old Value': row.get('Old Value'),
            'Delta vs Sharp': 0.0,
            'SHARP_SIDE_TO_BET': is_sharp_side,
            'Sharp_Move_Signal': metrics.get('Sharp_Move_Signal', 0),
            'Sharp_Limit_Jump': metrics.get('Sharp_Limit_Jump', 0),
            'Sharp_Time_Score': metrics.get('Sharp_Time_Score', 0),
            'Sharp_Prob_Shift': metrics.get('Sharp_Prob_Shift', 0),
            'Sharp_Limit_Total': metrics.get('Sharp_Limit_Total', 0)
        })
    
        row['SharpBetScore'] = round(
            2.0 * row['Sharp_Move_Signal'] +
            2.0 * row['Sharp_Limit_Jump'] +
            1.5 * row['Sharp_Time_Score'] +
            1.0 * row['Sharp_Prob_Shift'] +
            0.001 * row['Sharp_Limit_Total'], 2
        )
    
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



st.set_page_config(layout="wide")


# === Initialize Google Drive once ===




st.title("📊 Sharp Edge Scanner")

def log_rec_snapshot(df_moves, sport_key, drive=None):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{sport_key}_snapshot_{now}.csv"
    path = f"/tmp/rec_snapshots/{file_name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean_old_snapshots()

    df_snapshot = df_moves[
        ['Game', 'Market', 'Bookmaker', 'Value', 'Time']
    ].copy()
    df_snapshot.to_csv(path, index=False)
    print(f"📦 Rec snapshot saved to: {path}")

    # Upload to Google Drive if available
    if drive:
        try:
            file_drive = drive.CreateFile({'title': file_name, "parents": [{"id": FOLDER_ID}]})
            file_drive.SetContentFile(path)
            file_drive.Upload()
            print(f"☁️ Rec snapshot uploaded to Google Drive as: {file_name}")
        except Exception as e:
            print(f"❌ Failed to upload rec snapshot to Drive: {e}")
import time

def clean_old_snapshots(snapshot_dir="/tmp/rec_snapshots", max_age_hours=120):
    now = time.time()
    cutoff = now - (max_age_hours * 3600)

    deleted = 0
    for fname in os.listdir(snapshot_dir):
        path = os.path.join(snapshot_dir, fname)
        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
            try:
                os.remove(path)
                deleted += 1
            except Exception as e:
                print(f"⚠️ Failed to delete {path}: {e}")
    print(f"🧹 Pruned {deleted} old snapshots.")


def track_rec_drift(game_key, outcome_key, snapshot_dir="/tmp/rec_snapshots", minutes=30):
    import glob
    from datetime import timedelta

    files = sorted(glob.glob(os.path.join(snapshot_dir, "*.csv")))
    drift_rows = []

    for f in files:
        df = pd.read_csv(f)
        df = df[df['Game'] == game_key]
        df = df[df['Outcome'].str.lower() == outcome_key.lower()]

        if df.empty:
            continue

        for _, row in df.iterrows():
            drift_rows.append({
                'Snapshot_Time': f.split('_')[-1].replace('.csv', ''),
                'Value': row['Value'],
                'Bookmaker': row['Bookmaker'],
                'Market': row['Market']
            })

    return pd.DataFrame(drift_rows).sort_values(by='Snapshot_Time')

def save_model_to_drive(model, drive, filename='sharp_win_model.pkl', folder_id=FOLDER_ID):
    buffer = BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    
    try:
        file_list = drive.ListFile({
            'q': f"title='{filename}' and '{folder_id}' in parents and trashed=false"
        }).GetList()
        for file in file_list:
            file.Delete()  # delete old model

        file_drive = drive.CreateFile({'title': filename, 'parents': [{"id": folder_id}]})
        file_drive.SetContentString(buffer.read().decode('latin1'))  # Save as encoded string
        file_drive.Upload()
        print(f"✅ Model saved to Google Drive as {filename}")
    except Exception as e:
        print(f"❌ Failed to save model to Google Drive: {e}")

def load_model_from_drive(drive, filename='sharp_win_model.pkl', folder_id=FOLDER_ID):
    try:
        file_list = drive.ListFile({
            'q': f"title='{filename}' and '{folder_id}' in parents and trashed=false"
        }).GetList()
        if not file_list:
            return None

        content = file_list[0].GetContentString()
        buffer = BytesIO(content.encode('latin1'))
        model = pickle.load(buffer)
        print(f"✅ Loaded model from Google Drive: {filename}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def should_retrain_model(drive, filename='model_last_updated.txt', folder_id=FOLDER_ID):
    try:
        file_list = drive.ListFile({
            'q': f"title='{filename}' and '{folder_id}' in parents and trashed=false"
        }).GetList()
        
        if not file_list:
            return True  # No record yet = retrain

        file = file_list[0]
        last_updated_str = file.GetContentString().strip()
        last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d")
        return (datetime.utcnow() - last_updated) > timedelta(days=14)

    except Exception as e:
        print(f"⚠️ Failed to check model timestamp: {e}")
        return True

def save_model_timestamp(drive, filename='model_last_updated.txt', folder_id=FOLDER_ID):
    timestamp_str = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        file_list = drive.ListFile({
            'q': f"title='{filename}' and '{folder_id}' in parents and trashed=false"
        }).GetList()
        for file in file_list:
            file.Delete()

        file_drive = drive.CreateFile({'title': filename, 'parents': [{"id": folder_id}]})
        file_drive.SetContentString(timestamp_str)
        file_drive.Upload()
        print(f"✅ Model timestamp updated: {timestamp_str}")
    except Exception as e:
        print(f"❌ Failed to save timestamp: {e}")
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


def render_scanner_tab(label, sport_key, container, drive):
    global market_component_win_rates
    timestamp = pd.Timestamp.utcnow()
    sport_key_lower = sport_key.lower()

    with container:
        st.subheader(f"📡 Scanning {label} Sharp Signals")

        live = fetch_live_odds(sport_key)
        prev = load_latest_snapshot_from_drive(sport_key, drive, FOLDER_ID)

        if not live:
            st.warning("⚠️ No live odds returned.")
            return pd.DataFrame()
        if not prev:
            st.info("🟡 First run — no previous snapshot. Continuing with empty prev.")
            prev = {}

        upload_snapshot_to_drive(sport_key, get_snapshot(live), drive, FOLDER_ID)

        confidence_weights = market_component_win_rates.get(sport_key_lower, {})
        df_moves_raw, df_audit, summary_df = detect_sharp_moves(
            live, prev, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS,
            weights=market_component_win_rates
        )

        if 'Enhanced_Sharp_Confidence_Score' not in df_moves_raw.columns:
            st.error("❌ detect_sharp_moves() missing Enhanced_Sharp_Confidence_Score")
            return pd.DataFrame()

        df_moves_raw['Snapshot_Timestamp'] = timestamp
        df_moves_raw['Sport'] = label
        # After df_moves_raw is created:
        df_moves_raw = build_game_key(df_moves_raw)
        df_moves = df_moves_raw.drop_duplicates(subset=['Game_Key', 'Bookmaker'])

        model = load_model_from_drive(drive)
        if model is not None:
            try:
                df_moves_raw = apply_blended_sharp_score(df_moves_raw, model)
                st.success("✅ Applied model scoring to raw sharp data")
            except Exception as e:
                st.warning(f"⚠️ Could not apply model scoring early: {e}")

        if 'Enhanced_Sharp_Confidence_Score' not in df_moves.columns or df_moves['Enhanced_Sharp_Confidence_Score'].isna().all():
            st.warning("⚠️ Enhanced_Sharp_Confidence_Score missing — attempting to recover from df_moves_raw")
            try:
                df_moves = df_moves.merge(
                    df_moves_raw[['Game_Key', 'Market', 'Bookmaker', 'Enhanced_Sharp_Confidence_Score']],
                    on=['Game_Key', 'Market', 'Bookmaker'],
                    how='left'
                )
                st.success("✅ Recovered confidence score from df_moves_raw")
            except Exception as e:
                st.error(f"❌ Recovery failed: {e}")

        df_bt = fetch_scores_and_backtest(sport_key, df_moves, api_key=API_KEY)
        if not df_bt.empty:
            merge_cols = ['Game_Key', 'Market', 'Bookmaker']
            confidence_cols = ['Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score', 'Sharp_Confidence_Tier']
            available = [col for col in confidence_cols if col in df_moves_raw.columns]

            df_bt_merged = df_bt.merge(
                df_moves_raw[merge_cols + available].drop_duplicates(),
                on=merge_cols,
                how='left',
                indicator=True
            )
            st.write(f"🔍 {label} merge result:", df_bt_merged['_merge'].value_counts())

            for col in available:
                if col in df_bt_merged.columns and col in df_moves_raw.columns:
                    df_bt_merged[col] = df_bt_merged[col].fillna(df_moves_raw[col])

            df_moves = df_bt_merged.drop(columns=['_merge'])
        else:
            st.info("ℹ️ No backtest results found — skipped.")

        if 'SHARP_HIT_BOOL' in df_moves.columns and 'Enhanced_Sharp_Confidence_Score' in df_moves.columns:
            trainable = df_moves[
                df_moves['SHARP_HIT_BOOL'].notna() &
                df_moves['Enhanced_Sharp_Confidence_Score'].notna()
            ]
            if len(trainable) >= 5:
                model = train_sharp_win_model(trainable)
                save_model_to_drive(model, drive)
                save_model_timestamp(drive)
            else:
                st.warning("⚠️ Not enough completed sharp picks to train model.")
        else:
            st.warning("⚠️ Required columns missing for model training.")

        if model is not None:
            if 'Enhanced_Sharp_Confidence_Score' not in df_moves.columns or df_moves['Enhanced_Sharp_Confidence_Score'].isna().all():
                try:
                    df_moves = df_moves.drop(
                        columns=[c for c in ['Enhanced_Sharp_Confidence_Score', 'True_Sharp_Confidence_Score']],
                        errors='ignore'
                    )
                    df_moves = df_moves.merge(
                        df_moves_raw[['Game_Key', 'Market', 'Bookmaker', 'Enhanced_Sharp_Confidence_Score']],
                        on=['Game_Key', 'Market', 'Bookmaker'],
                        how='left'
                    )
                    st.success("✅ Final confidence recovery successful")
                except Exception as e:
                    st.error(f"❌ Final recovery failed: {e}")

            try:
                df_moves = apply_blended_sharp_score(df_moves, model)
            except Exception as e:
                st.warning(f"⚠️ Final model scoring failed: {e}")

        if not df_moves.empty:
            append_to_master_csv_on_drive(df_moves, "sharp_moves_master.csv", drive, FOLDER_ID)

        if not df_audit.empty:
            df_audit['Snapshot_Timestamp'] = timestamp
            try:
                file_list = drive.ListFile({
                    'q': f"title='line_history_master.csv' and '{FOLDER_ID}' in parents and trashed=false"
                }).GetList()
                df_existing = pd.DataFrame()
                if file_list:
                    file_drive = file_list[0]
                    existing_data = StringIO(file_drive.GetContentString())
                    df_existing = pd.read_csv(existing_data)
                    file_drive.Delete()
                df_combined = pd.concat([df_existing, df_audit], ignore_index=True)
                buffer = StringIO()
                df_combined.to_csv(buffer, index=False)
                buffer.seek(0)
                new_file = drive.CreateFile({'title': "line_history_master.csv", 'parents': [{"id": FOLDER_ID}]})
                if new_file is not None:
                    new_file.SetContentString(buffer.getvalue())
                    new_file.Upload()
                else:
                    st.error("❌ drive.CreateFile returned None for line_history_master.csv — skipping upload.")

            except Exception as e:
                st.error(f"❌ Failed to update line history: {e}")

        # === Sharp Summary Table
        # === Sharp Summary Table
        st.subheader(f"📊 Sharp vs Rec Book Consensus Summary – {label}")
        
        # === Normalize for clean merge
        summary_df['Game'] = summary_df['Game'].str.strip().str.lower()
        df_moves['Game'] = df_moves['Game'].str.strip().str.lower()
        df_moves['Outcome'] = df_moves['Outcome'].str.strip().str.lower()
        summary_df['Outcome'] = summary_df['Outcome'].str.strip().str.lower()
        
        # === Merge Blended Score + Win Prob
        if 'Blended_Sharp_Score' in df_moves.columns:
            df_merge_scores = df_moves[['Game', 'Market', 'Outcome', 'Blended_Sharp_Score', 'Model_Sharp_Win_Prob']].drop_duplicates()
            summary_df = summary_df.merge(
                df_merge_scores,
                on=['Game', 'Market', 'Outcome'],
                how='left'
            )
          
        df_game_start = df_moves_raw[['Game', 'Market', 'Game_Start']].dropna().drop_duplicates()
        # ✅ Add Game_Start to summary_df using (Game + Market + Event_Date)
        df_game_start = df_moves_raw[['Game', 'Market', 'Event_Date', 'Game_Start']].dropna().drop_duplicates()
        df_master = build_game_key(df_master)
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
        
        summary_df = summary_df.merge(
            df_game_start[['MergeKey', 'Game_Start']],
            on='MergeKey',
            how='left'
        )

        # === Convert Game_Start to EST safely
        def safe_to_est(dt):
            if pd.isna(dt):
                return ""
            try:
                dt = pd.to_datetime(dt, errors='coerce')
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
                return dt.tz_convert('US/Eastern').strftime('%Y-%m-%d %I:%M %p')
            except Exception as e:
                print(f"⚠️ Date conversion error: {e}")
                return ""
        
        summary_df['Game_Start'] = pd.to_datetime(summary_df['Game_Start'], errors='coerce', utc=True)
        summary_df['Date + Time (EST)'] = summary_df['Game_Start'].apply(safe_to_est)
        
        # === Optional: filter out rows with no date for clarity
        summary_df = summary_df[summary_df['Date + Time (EST)'] != ""]
        
        # === Rename and clean
        summary_df.drop(columns=[col for col in ['Date', 'Time\n(EST)'] if col in summary_df.columns], inplace=True)
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
        
        # === Display filter
        market_options = ["All"] + sorted(summary_df['Market'].dropna().unique())
        market = st.selectbox(f"📊 Filter {label} by Market", market_options, key=f"{label}_market_summary")
        filtered_df = summary_df if market == "All" else summary_df[summary_df['Market'] == market]
        
        # === Final render
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



def render_sharp_signal_analysis_tab(tab, sport_label, sport_key_api, drive):

    with tab:
       # Now load master AFTER optional patch
        sport_key_lower = sport_key_api
        df_master = load_master_sharp_moves(drive)
        df_master = load_master_sharp_moves(drive)
        df_master = build_game_key(df_master)
        
        # Load scores from the past N days via Odds API
        df_scored = fetch_scores_and_backtest(sport_key_api, df_master.copy(), api_key=API_KEY)
        
        
        # Merge back into master — only overwrite scores where Game_Key matches
        # ✅ Safe merge of updated score columns
        required_cols = [
            'Game_Key', 'Market', 'Outcome', 'Bookmaker',
            'Score_Home_Score', 'Score_Away_Score',
            'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT'
        ]
        


        merge_keys = [col for col in ['Game_Key', 'Market', 'Bookmaker']
              if col in df_scored.columns and col in df_master.columns]
        value_cols = [col for col in ['Score_Home_Score', 'Score_Away_Score', 'SHARP_HIT_BOOL', 'SHARP_COVER_RESULT']
                      if col in df_scored.columns]
        # === 🛡️ FORCE 'Game_Key' CREATION IN df_master if missing
        if 'Game_Key' not in df_master.columns:
            if all(col in df_master.columns for col in ['Game', 'Game_Start']):
                df_master['Home_Team_Norm'] = df_master['Game'].str.extract(r'^(.*?) vs')[0].str.strip().str.lower()
                df_master['Away_Team_Norm'] = df_master['Game'].str.extract(r'vs (.*)$')[0].str.strip().str.lower()
                df_master['Commence_Hour'] = pd.to_datetime(df_master['Game_Start'], errors='coerce', utc=True).dt.floor('H')
                df_master['Game_Key'] = (
                    df_master['Home_Team_Norm'] + "_" +
                    df_master['Away_Team_Norm'] + "_" +
                    df_master['Commence_Hour'].astype(str)
                )
            else:
                st.error("❌ 'Game_Key' missing and cannot be created (requires 'Game' and 'Game_Start')")
                return
                
        # 🛡️ Defensive check for missing keys
        missing_keys_master = [col for col in merge_keys if col not in df_master.columns]
        missing_keys_scored = [col for col in merge_keys if col not in df_scored.columns]

        # 🛡️ Ensure all merge keys exist
        missing_keys = [col for col in merge_keys if col not in df_master.columns]
        if missing_keys:
            st.error(f"❌ Cannot merge — df_master is missing columns: {missing_keys}")
        else:
            df_master = df_master.drop(columns=value_cols, errors='ignore')
            df_master = df_master.merge(
                df_scored[merge_keys + value_cols],
                on=merge_keys,
                how='left'
            )
        # 🛡️ Defensive check for missing keys
        missing_keys_master = [col for col in merge_keys if col not in df_master.columns]
        missing_keys_scored = [col for col in merge_keys if col not in df_scored.columns]

      

        
        if merge_keys and value_cols:
            df_master = df_master.drop(columns=value_cols, errors='ignore')
            df_master = df_master.merge(
                df_scored[merge_keys + value_cols],
                on=merge_keys,
                how='left'
            )
        else:
            st.warning("⚠️ No valid merge keys or value columns for backtest score update.")
        

        if not df_master.empty:
            df_bt = fetch_scores_and_backtest(sport_key_api, df_master, api_key=API_KEY)

            if not df_bt.empty and 'SHARP_HIT_BOOL' in df_bt.columns:
                df_bt['SharpConfidenceTier'] = pd.cut(
                    df_bt['SharpBetScore'],
                    bins=[0, 15, 25, 40, 100],
                    labels=["⚠️ Low", "✅ Moderate", "⭐ High", "🔥 Steam"]
                )

                scored = df_bt[df_bt['SHARP_HIT_BOOL'].notna()]
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

                leaderboard_df = pd.concat(leaderboard_rows, ignore_index=True)
                leaderboard_df = leaderboard_df[[
                    'Market', 'Component', 'Component_Value', 'Signal_Count', 'Win_Rate'
                ]].sort_values(by='Win_Rate', ascending=False)

                st.dataframe(leaderboard_df.head(50))
           

                st.subheader(f"📊 {sport_label} Sharp Signal Performance by Market + Confidence Tier")
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
                st.dataframe(df_market_tier_summary)

                # === Component Learning & Weight Storage ===
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
                            win_rate = max(0.5, row['Win_Rate'])  # ✅ clamp at 0.5

                            # ✅ normalize value keys
                            if pd.isna(val):
                                continue  # skip NaNs
                            elif isinstance(val, bool):
                                val_key = str(val).lower()  # 'true'/'false'
                            elif isinstance(val, float) and val.is_integer():
                                val_key = str(int(val))  # 2.0 → '2'
                            else:
                                val_key = str(val).lower()

                            market_component_win_rates_sport \
                                .setdefault(market, {}) \
                                .setdefault(comp, {})[val_key] = win_rate

                # 🔁 Update and save to global + Drive
                market_component_win_rates = globals().get("market_component_win_rates", {})
                market_component_win_rates[sport_key_lower] = market_component_win_rates_sport
                globals()["market_component_win_rates"] = market_component_win_rates

                try:
                    save_weights_to_drive(market_component_win_rates, drive)
                    print(f"✅ Saved weights for {sport_key_lower} to Google Drive.")
                except Exception as e:
                    print(f"❌ Failed to save weights for {sport_key_lower}: {e}")

                # 🔍 Debug View
                st.subheader(f"📥 Learned Weights for {sport_label}")
                st.json(market_component_win_rates.get(sport_key_lower, {}))

                st.subheader(f"🧪 Sample {sport_label} Confidence Inputs")
                st.dataframe(df_bt[[
                    'Market', 'Sharp_Move_Signal', 'Sharp_Time_Score', 'True_Sharp_Confidence_Score'
                ]].head())
            else:
                st.warning(f"⚠️ {sport_label} backtest missing 'SHARP_HIT_BOOL'. No results to summarize.")
        else:
            st.warning(f"⚠️ No historical sharp picks found for {sport_label}.")




tab_nba, tab_mlb = st.tabs(["🏀 NBA", "⚾ MLB"])

with tab_nba:
    with st.expander("📊 Real-Time Sharp Scanner", expanded=True):
        df_nba_live = render_scanner_tab("NBA", SPORTS["NBA"], tab_nba, drive)
    with st.expander("📈 Backtest Performance", expanded=False):
        render_sharp_signal_analysis_tab(tab_nba, "NBA", SPORTS["NBA"], drive)

with tab_mlb:
    with st.expander("📊 Real-Time Sharp Scanner", expanded=True):
        df_mlb_live = render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb, drive)
    with st.expander("📈 Backtest Performance", expanded=False):
        render_sharp_signal_analysis_tab(tab_mlb, "MLB", SPORTS["MLB"], drive)


