import streamlit as st
import pandas as pd
import requests
import os
import json
import pickle
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from io import StringIO
from collections import defaultdict
import pytz
from collections import OrderedDict

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
    try:
        # 🔍 Step 1: Find all files with the same name
        file_list = drive.ListFile({
            'q': f"title='{filename}' and '{folder_id}' in parents and trashed=false"
        }).GetList()

        # 🔁 Step 2: Load and merge with existing file(s)
        df_combined = df_new
        if file_list:
            print(f"📂 Found {len(file_list)} existing file(s) for {filename}. Merging contents.")
            for file_drive in file_list:
                try:
                    existing_data = StringIO(file_drive.GetContentString())
                    df_existing = pd.read_csv(existing_data)
                    df_combined = pd.concat([df_existing, df_combined], ignore_index=True)
                except Exception as read_error:
                    print(f"⚠️ Could not read one file: {read_error}")
                # Always delete old files to prevent duplicates
                file_drive.Delete()
                print(f"🗑️ Deleted old {filename} from Drive")

        # 🧼 Step 3: Remove duplicates
        df_combined.drop_duplicates(
            subset=["Event_Date", "Game", "Market", "Outcome", "Bookmaker"],
            keep='last',
            inplace=True
        )

        # 💾 Step 4: Save to buffer and upload to Drive
        csv_buffer = StringIO()
        df_combined.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        new_file = drive.CreateFile({'title': filename, "parents": [{"id": folder_id}]})
        new_file.SetContentString(csv_buffer.getvalue())
        new_file.Upload()
        print(f"✅ {filename} uploaded to Drive with {len(df_combined)} total rows.")

    except Exception as e:
        print(f"❌ Error appending to {filename}: {e}")


def load_master_sharp_moves(drive, filename="sharp_moves_master.csv"):
    try:
        file_list = drive.ListFile({
            'q': f"title='{filename}' and '{FOLDER_ID}' in parents and trashed=false"
        }).GetList()
        if not file_list:
            print("⚠️ No master file found.")
            return pd.DataFrame()

        file_drive = file_list[0]
        csv_buffer = StringIO(file_drive.GetContentString())
        return pd.read_csv(csv_buffer)
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

def fetch_scores_and_backtest(df_moves, sport_key='baseball_mlb', days_back=3, api_key=API_KEY):

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
    params = {'daysFrom': days_back, 'apiKey': api_key}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        score_data = response.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch scores: {e}")
        return pd.DataFrame()

    result_rows = []
    completed_games = 0
    
    for game in score_data:
        if not game.get("completed"):
            continue
    
        completed_games += 1
    
        home = game.get("home_team", "").strip().lower()
        away = game.get("away_team", "").strip().lower()
        scores = game.get("scores", [])
        team_scores = {
            s["name"].strip().lower(): s["score"]
            for s in scores if "name" in s and "score" in s
        }
    
        if home not in team_scores or away not in team_scores:
            print(f"❌ Skipped — Missing score: {home=} {away=} vs {team_scores}")
            continue
    
        game_name = f"{game['home_team']} vs {game['away_team']}"  # keep display casing
        result_rows.append({
            'Game': game_name,
            'Home_Team': game['home_team'],
            'Away_Team': game['away_team'],
            'Home_Score': team_scores[home],
            'Away_Score': team_scores[away]
        })
    
    df_results = pd.DataFrame(result_rows)
    st.write(f"✅ Completed games in API: {completed_games}, Parsed: {len(df_results)}")
    
    if df_results.empty:
        st.warning("⚠️ No parsed scores matched your games, even though API reported completed games.")
        return pd.DataFrame()


    # Prepare sharp data
    df_moves = df_moves.drop_duplicates(subset=['Game', 'Market', 'Outcome']).copy()
    if 'Home_Team' not in df_moves.columns or 'Away_Team' not in df_moves.columns:
        df_moves[['Home_Team', 'Away_Team']] = df_moves['Game'].str.extract(r'^(.*?) vs (.*?)$')

    import pytz
    df_moves['Snapshot_Timestamp'] = pd.to_datetime(df_moves['Snapshot_Timestamp'], errors='coerce')
    if df_moves['Snapshot_Timestamp'].dt.tz is None:
        df_moves['Snapshot_Timestamp'] = df_moves['Snapshot_Timestamp'].dt.tz_localize('US/Eastern')
    df_moves['Snapshot_Timestamp'] = df_moves['Snapshot_Timestamp'].dt.tz_convert('UTC')

    df_moves['Game_Start'] = pd.to_datetime(df_moves['Game_Start'], errors='coerce')
    if df_moves['Game_Start'].dt.tz is None:
        df_moves['Game_Start'] = df_moves['Game_Start'].dt.tz_localize('UTC')

    # Filter for pregame entries only
    df_moves = df_moves[df_moves['Snapshot_Timestamp'] < df_moves['Game_Start']]

    # Rename scores to avoid conflict
    df_results.rename(columns={
        'Home_Team': 'Score_Home_Team',
        'Away_Team': 'Score_Away_Team',
        'Home_Score': 'Score_Home_Score',
        'Away_Score': 'Score_Away_Score'
    }, inplace=True)

    # Merge and score
    df = df_moves.merge(df_results, on='Game', how='left')
    df[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']] = df.apply(lambda r: pd.Series(calc_cover(r)), axis=1)

    return df

    # === Refactored cover logic
def calc_cover(row):
    market = str(row['Market']).strip().lower()

    hscore = row['Score_Home_Score']
    ascore = row['Score_Away_Score']
    if pd.isna(hscore) or pd.isna(ascore):
        return None, None

    try:
        hscore = float(hscore)
        ascore = float(ascore)
    except ValueError:
        return None, None

    if market == 'totals':
        try:
            total = float(row.get('Ref Sharp Value'))
        except (TypeError, ValueError):
            print(f"❌ Invalid total value: {row.get('Ref Sharp Value')}")
            return None, None

        total_points = hscore + ascore
        outcome = str(row['Outcome']).strip().lower()

        if 'under' in outcome:
            hit = int(total_points < total)
            return ('Win', hit) if hit else ('Loss', hit)
        elif 'over' in outcome:
            hit = int(total_points > total)
            return ('Win', hit) if hit else ('Loss', hit)
        else:
            print(f"❓ Unknown totals outcome: '{outcome}'")
            return None, None

    # === H2H / Spreads logic only applies if not totals
    team = str(row['Outcome']).strip().lower()
    home = str(row['Home_Team']).strip().lower()
    away = str(row['Away_Team']).strip().lower()

    if team in home:
        team_score, opp_score = hscore, ascore
    elif team in away:
        team_score, opp_score = ascore, hscore
    else:
        if market in ['spreads', 'h2h']:
            print(f"❌ Could not match team: '{team}' with Home: '{home}' or Away: '{away}'")
        return None, None

    margin = team_score - opp_score

    if market == 'h2h':
        hit = int(team_score > opp_score)
        return ('Win', hit) if hit else ('Loss', hit)

    if market == 'spreads':
        try:
            spread = float(row.get('Ref Sharp Value'))
        except (TypeError, ValueError):
            return None, None

        hit = int((margin > abs(spread)) if spread < 0 else (margin + spread > 0))
        return ('Win', hit) if hit else ('Loss', hit)

    # If market is unrecognized
    return None, None



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

    def normalize_label(label):
        return str(label).strip().lower().replace('.0', '')

    rows = []
    line_open_map = {}  # {(game, market, label): (open_val, open_time)}

    sharp_limit_map = defaultdict(lambda: defaultdict(list))
    sharp_total_limit_map = defaultdict(int)  # NEW: Track summed limits across SHARP_BOOKS
    sharp_lines, sharp_side_flags, sharp_metrics_map = {}, {}, {}
    line_history_log = []

    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}
    # Use learned weights if available, else fallback to default neutral confidence
    # Normalize to consistent keys used in weights
    sport_scope_key = {
        'MLB': 'baseball_mlb',
        'NBA': 'basketball_nba'
    }.get(sport_key.upper(), sport_key.lower())  # fallback safe
    
    
    print("🔍 Weights structure preview:", json.dumps(weights, indent=2))
    print("🧠 Extracting weights for:", sport_scope_key)
    confidence_weights = weights.get(sport_scope_key, {})
    print(f"✅ Using weights for {sport_scope_key} — Available markets: {list(confidence_weights.keys())}")

    # === Component fields for confidence scoring
    component_fields = [
        'Sharp_Move_Signal',
        'Sharp_Limit_Jump',
        'Sharp_Time_Score',
        'Sharp_Prob_Shift',
        'Sharp_Limit_Total'
    ]
    
    # Incorrect: local_components = list(component_fields.keys())
    local_components = component_fields  # ✅ Fix


    
    previous_odds_map = {}
    for g in previous.values():
        for book in g.get('bookmakers', []):
            book_key = book['key']
            for market in book.get('markets', []):
                mtype = market.get('key')
                for outcome in market.get('outcomes', []):
                    label = str(outcome['name']).strip().lower()
                    price = outcome.get('point') if mtype != 'h2h' else outcome.get('price')
                    previous_odds_map[(g['home_team'], g['away_team'], mtype, label, book_key)] = price


    for game in current:
        game_name = f"{game['home_team']} vs {game['away_team']}"
        event_date = pd.to_datetime(game.get("commence_time")).strftime("%Y-%m-%d") if game.get("commence_time") else ""
        event_time = pd.to_datetime(game.get("commence_time"))
        gid = game['id']
        prev_game = previous_map.get(gid, {})

        for book in game.get('bookmakers', []):
            book_key = book['key']
            for market in book.get('markets', []):
                mtype = market['key']
                for o in market.get('outcomes', []):
                    label = normalize_label(o['name'])
                    val = o.get('point') if mtype != 'h2h' else o.get('price')
                    limit = o.get('bet_limit') if 'bet_limit' in o and o.get('bet_limit') is not None else None
                    key = (game_name, mtype, label)
                    prev_key = (game['home_team'], game['away_team'], mtype, label, book_key)
                    old_val = previous_odds_map.get(prev_key)
                    
                    if key not in line_open_map and val is not None:
                        line_open_map[key] = (val, snapshot_time)
                    
                    entry = {
                        'Sport': sport_key,
                        'Time': snapshot_time,
                        'Game': game_name,
                        'Game_Start': event_time,
                        'Market': mtype,
                        'Outcome': label,
                        'Bookmaker': book['title'],
                        'Book': book_key,
                        'Value': val,
                        'Limit': limit,
                        'Old Value': old_val,
                        'Delta': round(val - old_val, 2) if old_val is not None and val is not None else None,
                        'Event_Date': event_date,
                        'Region': BOOKMAKER_REGIONS.get(book_key, 'unknown'),
                    }

                    # Add historical audit entry
                    line_history_log.append(entry.copy())

                    if val is None:
                        continue

                    # Previous value lookup
                    if prev_game:
                        for prev_b in prev_game.get('bookmakers', []):
                            if prev_b['key'] == book_key:
                                for prev_m in prev_b.get('markets', []):
                                    if prev_m['key'] == mtype:
                                        for prev_o in prev_m.get('outcomes', []):
                                            if normalize_label(prev_o['name']) == label:
                                                prev_val = prev_o.get('point') if mtype != 'h2h' else prev_o.get('price')
                                                if prev_val is not None:
                                                    entry['Old Value'] = prev_val
                                                    entry['Delta'] = round(val - prev_val, 2) if prev_val is not None and val is not None else None



                    rows.append(entry)

                    # Track sharp data if limit is present
                    if limit is not None:
                        sharp_lines[(game_name, mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit, val, entry.get('Old Value')))
                        if book_key in SHARP_BOOKS_FOR_LIMITS:
                            sharp_total_limit_map[(game_name, mtype, label)] += limit

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

    # === Append sharp-sided bets with enriched component scores
    for (game_name, mtype, label), entry in sharp_lines.items():
        if sharp_side_flags.get((game_name, mtype, label), 0):
            metrics = sharp_metrics_map.get((game_name, mtype, label), {})
            enriched = entry.copy()
            enriched.update({
                'Ref Sharp Value': entry['Value'],
                'Ref Sharp Old Value': entry.get('Old Value'),
                'Delta vs Sharp': 0.0,
                'SHARP_SIDE_TO_BET': 1,
                'Sharp_Move_Signal': metrics.get('Sharp_Move_Signal', 0),
                'Sharp_Limit_Jump': metrics.get('Sharp_Limit_Jump', 0),
                'Sharp_Time_Score': metrics.get('Sharp_Time_Score', 0),
                'Sharp_Prob_Shift': metrics.get('Sharp_Prob_Shift', 0),
                'Sharp_Limit_Total': metrics.get('Sharp_Limit_Total', 0)
            })
            # Calculate basic SharpBetScore (used for legacy tiering if needed)
            enriched['SharpBetScore'] = round(
                2.0 * enriched['Sharp_Move_Signal'] +
                2.0 * enriched['Sharp_Limit_Jump'] +
                1.5 * enriched['Sharp_Time_Score'] +
                1.0 * enriched['Sharp_Prob_Shift'] +
                0.001 * enriched['Sharp_Limit_Total'], 2
            )
            rows.append(enriched)
    
    # === Intelligence scoring
    def compute_intelligence_score(row):
        score = 0
        reasons = []
        if row.get('LimitUp_NoMove_Flag', 0) == 1:
            score += 15
            reasons.append("🤫 Limit ↑, price ↔ (Position signal)")
        if abs(row.get('Delta vs Sharp', 0)) >= 0.5:
            score += 10
            reasons.append("📈 Price moved from sharp baseline")
        if row.get('Limit_Jump', 0) == 1 and abs(row.get('Delta vs Sharp', 0)) == 0:
            score += 15
            reasons.append("🤫 Limit ↑, price ↔")
        return pd.Series({
            'SharpIntelligenceScore': min(score, 100),
            'SharpIntelReasons': ", ".join(reasons) if reasons else "No clear signal"
        })
    
    # === Confidence scoring using learned weights
    def compute_confidence(row, market_weights):
        market = str(row.get('Market', '')).lower()
        score = 0
        max_score = 0
    
        for comp in component_fields:
            val = row.get(comp)
    
            try:
                val_key = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val).lower()
                weight = market_weights.get(market, {}).get(comp, {}).get(val_key, 0.5)
            except:
                weight = 0.5
    
            score += weight * 10
            max_score += 10
    
    
        return round(score, 2) if max_score > 0 else None

    
    # === Create base DataFrame
    df = pd.DataFrame(rows)
    df_history = pd.DataFrame(line_history_log)
    
    # Open line merge
    df_history_sorted = df_history.sort_values('Time')
    line_open_df = (
        df_history_sorted.dropna(subset=['Value'])
        .groupby(['Game', 'Market', 'Outcome'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Value'})
    )
    
    if df.empty:
        return df, df_history
    
    df = df.merge(line_open_df, on=['Game', 'Market', 'Outcome'], how='left')
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
    
    df[['SharpIntelligenceScore', 'SharpIntelReasons']] = df.apply(compute_intelligence_score, axis=1)
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
        lambda r: compute_confidence(r, confidence_weights), axis=1
    )

    
    # Confidence tiering
    df['Sharp_Confidence_Tier'] = pd.cut(
        df['True_Sharp_Confidence_Score'],
        bins=[-1, 25, 50, 75, float('inf')],
        labels=['⚠️ Low', '✅ Medium', '⭐ High', '🔥 Steam']
    )
    # === Sharp vs Rec Book Consensus Summary ===
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
        'SharpBetScore', 'True_Sharp_Confidence_Score', 'Sharp_Confidence_Tier'
    ]].drop_duplicates()
    
    summary_df = summary_df.merge(
        sharp_scores,
        on=['Event_Date', 'Game', 'Market', 'Outcome'],
        how='left'
    )
    
    summary_df[['SharpBetScore', 'True_Sharp_Confidence_Score']] = summary_df[[
        'SharpBetScore', 'True_Sharp_Confidence_Score'
    ]].fillna(0)
    
    summary_df['Sharp_Confidence_Tier'] = summary_df['Sharp_Confidence_Tier'].fillna('⚠️ Low')
    
    return df, df_history, summary_df
        

st.set_page_config(layout="wide")
# === Initialize Google Drive once ===




st.title("📊 Sharp Edge Scanner")
auto_mode = st.sidebar.radio("🕹️ Refresh Mode", ["Auto Refresh", "Manual"], index=0)
if auto_mode == "Auto Refresh":
    st_autorefresh(interval=320000, key="autorefresh")

def log_rec_snapshot(df_moves, sport_key, drive=None):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{sport_key}_snapshot_{now}.csv"
    path = f"/tmp/rec_snapshots/{file_name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean_old_snapshots()

    df_snapshot = df_moves[
        ['Game', 'Market', 'Outcome', 'Bookmaker', 'Value', 'Time']
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

def render_scanner_tab(label, sport_key, container, drive):
    global market_component_win_rates  # ✅ FIXED HERE
    with container:
        live = fetch_live_odds(sport_key)
        prev = load_latest_snapshot_from_drive(sport_key, drive, FOLDER_ID)

        if not prev:
            st.info("🟡 First run detected — saving snapshot and skipping detection.")
            upload_snapshot_to_drive(sport_key, get_snapshot(live), drive, FOLDER_ID)
            return pd.DataFrame()

        if not live or not isinstance(live, list) or len(live) == 0:
            st.warning(f"⚠️ No live odds returned for {label}.")
            return pd.DataFrame()

        # ✅ Defer sharp move detection until learned weights are available
        # This placeholder is just for structure; no detection yet
        df_moves = pd.DataFrame()
        df_audit = pd.DataFrame()
        summary_df = pd.DataFrame()

        # === Snapshot live odds (before sharp detection)
        upload_snapshot_to_drive(sport_key, get_snapshot(live), drive, FOLDER_ID)

        # ✅ CALCULATE SPORT KEY LOWER
        sport_key_lower = sport_key.lower()

        # ✅ USE LEARNED WEIGHTS FROM GLOBAL DICT
        confidence_weights = market_component_win_rates.get(sport_key_lower, {})
        if not confidence_weights:
            st.warning(f"⚠️ No learned weights found for {sport_key_lower} — fallback weights will apply.")
        # ✅ Re-load weights if not already in memory
        
        # ✅ DETECT SHARP MOVES with latest weights
        df_moves, df_audit, summary_df = detect_sharp_moves(
            live, prev, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS,
            weights=market_component_win_rates
        )

        # ✅ Add timestamps and append if needed
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not df_moves.empty:
            df_moves['Snapshot_Timestamp'] = timestamp
            df_moves['Sport'] = label
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
                    try:
                        existing_data = StringIO(file_drive.GetContentString())
                        df_existing = pd.read_csv(existing_data)
                    except Exception as read_error:
                        print(f"⚠️ Could not read existing file: {read_error}")
                        df_existing = pd.DataFrame()

                    # Delete old file before uploading new
                    try:
                        file_drive.Delete()
                        print("🗑️ Deleted old line_history_master.csv")
                    except Exception as delete_error:
                        print(f"⚠️ Could not delete old file: {delete_error}")

                df_combined = pd.concat([df_existing, df_audit], ignore_index=True)
                csv_buffer = StringIO()
                df_combined.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                new_file = drive.CreateFile({'title': "line_history_master.csv", "parents": [{"id": FOLDER_ID}]})
                new_file.SetContentString(csv_buffer.getvalue())
                new_file.Upload()
                print(f"✅ line_history_master.csv uploaded with {len(df_combined)} total rows.")

            except Exception as e:
                st.error(f"❌ Failed to append to line history: {e}")

        # === Sharp Summary
        if summary_df is None or summary_df.empty:
            st.info(f"⚠️ No sharp consensus movements detected for {label}.")
        else:
            st.subheader(f"📊 Sharp vs Rec Book Consensus Summary – {label}")
            market_options = ["All"] + sorted(summary_df['Market'].dropna().unique())
            market = st.selectbox(f"📊 Filter {label} by Market", market_options, key=f"{label}_market_summary")
            filtered_df = summary_df if market == "All" else summary_df[summary_df['Market'] == market]
            st.dataframe(filtered_df[[
                'Event_Date', 'Game', 'Market', 'Recommended_Outcome',
                'Rec_Book_Consensus', 'Sharp_Book_Consensus',
                'Move_From_Open_Rec', 'Move_From_Open_Sharp',
                'SharpBetScore', 'True_Sharp_Confidence_Score'
            ]], use_container_width=True)

      
       

        # === Odds snapshot (pivoted, with limits, highlighted best lines)
        raw_odds_table = []
        for game in live:
            game_name = f"{game['home_team']} vs {game['away_team']}"
            for book in game.get("bookmakers", []):
                book_title = book["title"]
                for market in book.get("markets", []):
                    mtype = market.get("key")
                    for outcome in market.get("outcomes", []):
                        price = outcome.get("point") if mtype != "h2h" else outcome.get("price")
                        raw_odds_table.append({
                            'Event_Date': pd.to_datetime(game.get("commence_time")).strftime("%Y-%m-%d") if game.get("commence_time") else "",
                            "Game": game_name,
                            "Market": mtype,
                            "Outcome": outcome["name"],
                            "Bookmaker": book_title,
                            "Value": price,
                            "Limit": outcome.get("bet_limit", 0)
                        })

        df_odds_raw = pd.DataFrame(raw_odds_table)

        if not df_odds_raw.empty:
            st.subheader(f"📋 Live Odds Snapshot – {label} (Odds + Limit)")

            # Safely format odds + limit into one string like "-110.0 (20000)"
            import math

            def safe_format_value_limit(row):
                try:
                    val = float(row['Value'])
                    lim = int(row['Limit']) if pd.notnull(row['Limit']) and not math.isnan(row['Limit']) else 0
                    return f"{round(val, 1)} ({lim})"
                except:
                    return ""

            df_odds_raw['Value_Limit'] = df_odds_raw.apply(safe_format_value_limit, axis=1)


            # Pivot to wide format by bookmaker
            df_combined_display = df_odds_raw.pivot_table(
                index=["Event_Date", "Game", "Market", "Outcome"],
                columns="Bookmaker",
                values="Value_Limit",
                aggfunc="first"
            ).reset_index()

            # List of sharp books to highlight
            sharp_books = ['Pinnacle', 'Bookmaker', 'BetOnline']

            # Highlight sharp book columns with a light green background
            # Highlight columns if their name is in sharp_books
            def highlight_sharp_columns(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                for col in df.columns:
                    if col in sharp_books:
                        styles[col] = 'background-color: #d0f0c0; color: black'
                return styles
            st.dataframe(
                df_combined_display.style.apply(highlight_sharp_columns, axis=None),
                use_container_width=True
            )

        return df_moves






tab_nba, tab_mlb = st.tabs(["🏀 NBA", "⚾ MLB"])

with tab_nba:
    df_nba = render_scanner_tab("NBA", SPORTS["NBA"], tab_nba, drive)

with tab_mlb:
    df_mlb = render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb, drive)


# Load sharp picks from Drive
df_master = load_master_sharp_moves(drive)

# Safe predefinition
df_nba_bt = pd.DataFrame()
df_mlb_bt = pd.DataFrame()


# === NBA Sharp Signal Performance
# === NBA Sharp Signal Performance
# === NBA Sharp Signal Performance
with tab_nba:
    if not df_master.empty:
        df_nba_bt = fetch_scores_and_backtest(
            df_master[
                (df_master['Sport'] == 'NBA') &
                (df_master['SHARP_SIDE_TO_BET'] == 1)
            ],
            sport_key='basketball_nba'
        )

        if not df_nba_bt.empty and 'SHARP_HIT_BOOL' in df_nba_bt.columns:
            df_nba_bt['SharpConfidenceTier'] = pd.cut(
                df_nba_bt['SharpBetScore'],
                bins=[0, 15, 25, 40, 100],
                labels=["⚠️ Low", "✅ Moderate", "⭐ High", "🔥 Steam"]
            )

            scored = df_nba_bt[df_nba_bt['SHARP_HIT_BOOL'].notna()]
            st.subheader("🏆 Top Sharp Signal Performers by Market")

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

            # Tier performance
            st.subheader("📊 NBA Sharp Signal Performance by Market + Confidence Tier")
            if 'Market' not in df_nba_bt.columns:
                st.warning("⚠️ 'Market' column missing from df_nba_bt!")
                st.write("📋 Available columns:", df_nba_bt.columns.tolist())
            else:
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

            # Totals check
            st.subheader("🔍 Totals Presence Check")
            totals_rows = scored[scored['Market'].str.lower() == 'totals']
            st.write("✅ Totals backtest rows found:", len(totals_rows))
            if not totals_rows.empty:
                st.dataframe(totals_rows[[
                    'Game', 'Outcome', 'Market', 'Ref Sharp Value', 'SHARP_HIT_BOOL', 'SharpBetScore'
                ]].head(10))
            else:
                st.warning("❌ No scored totals rows found.")

            # Component Learning
            sport_key_lower = 'basketball_nba'
            market_component_win_rates_sport = {}

            st.subheader("🧠 Sharp Component Learning by Market")
            for comp, label in component_fields.items():
                if comp in scored.columns:
                    result = (
                        scored.groupby(['Market', comp])['SHARP_HIT_BOOL']
                        .mean().reset_index()
                        .rename(columns={'SHARP_HIT_BOOL': 'Win_Rate'})
                        .sort_values(by=['Market', comp])
                    )
                    st.markdown(f"**📊 {label} by Market**")
                    st.dataframe(result)

                    for _, row in result.iterrows():
                        market = row['Market'].lower()
                        val = row[comp]
                        win_rate = row['Win_Rate']
                        market_component_win_rates_sport.setdefault(market, {}).setdefault(comp, {})[val] = win_rate

            # Store globally and persist
            
            globals()["market_component_win_rates"] = market_component_win_rates

            market_component_win_rates = globals().get("market_component_win_rates", {})
            market_component_win_rates[sport_key_lower] = market_component_win_rates_sport
         

            
            try:
                save_weights_to_drive(market_component_win_rates, drive)
                print(f"✅ Saved weights for {sport_key_lower} to Google Drive.")
            except Exception as e:
                print(f"❌ Failed to save weights for {sport_key_lower}: {e}")


            
            # 📥 Show learned weights in UI
            st.subheader(f"📥 Learned Weights for {sport_key} (Debug)")
            st.json(market_component_win_rates.get(sport_key_lower, {}))
            
            # 🧪 Sample confidence scores for inspection
            st.subheader(f"🧪 Sample {sport_key} Confidence Inputs")
            st.dataframe(df_nba_bt[[
                'Market', 'Sharp_Move_Signal', 'Sharp_Time_Score', 'True_Sharp_Confidence_Score'
            ]].head())


        else:
            st.warning("⚠️ NBA backtest missing 'SHARP_HIT_BOOL'. No results to summarize.")
    else:
        st.warning("⚠️ No historical sharp picks found in Google Drive yet.")

# === MLB Sharp Signal Performance
# === MLB Sharp Signal Performance
# === MLB Sharp Signal Performance
with tab_mlb:
    if not df_master.empty:
        df_mlb_bt = fetch_scores_and_backtest(
            df_master[
                (df_master['Sport'] == 'MLB') &
                (df_master['SHARP_SIDE_TO_BET'] == 1)
            ],
            sport_key='baseball_mlb'
        )

        if not df_mlb_bt.empty and 'SHARP_HIT_BOOL' in df_mlb_bt.columns:
            # Create tier
            df_mlb_bt['SharpConfidenceTier'] = pd.cut(
                df_mlb_bt['SharpBetScore'],
                bins=[0, 15, 25, 40, 100],
                labels=["⚠️ Low", "✅ Moderate", "⭐ High", "🔥 Steam"]
            )

            # Filter scored rows once for reuse
            scored = df_mlb_bt[df_mlb_bt['SHARP_HIT_BOOL'].notna()]
            st.subheader("🏆 Top Sharp Signal Performers by Market")

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

            # Tier performance
            st.subheader("📊 MLB Sharp Signal Performance by Market + Confidence Tier")
            if 'Market' not in df_mlb_bt.columns:
                st.warning("⚠️ 'Market' column missing from df_mlb_bt!")
                st.write("📋 Available columns:", df_mlb_bt.columns.tolist())
            else:
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

            # Totals check
            st.subheader("🔍 Totals Presence Check")
            totals_rows = scored[
                scored['Market'].str.lower() == 'totals'
            ]
            st.write("✅ Totals backtest rows found:", len(totals_rows))
            if not totals_rows.empty:
                st.dataframe(totals_rows[[
                    'Game', 'Outcome', 'Market', 'Ref Sharp Value', 'SHARP_HIT_BOOL', 'SharpBetScore'
                ]].head(10))
            else:
                st.warning("❌ No scored totals rows found.")

            # Learning blocks
            # Store weights scoped by sport
            sport_key_lower = 'baseball_mlb'  
            market_component_win_rates_sport = {}
            
            st.subheader("🧠 Sharp Component Learning by Market")
            for comp, label in component_fields.items():
                if comp in scored.columns:
                    result = (
                        scored.groupby(['Market', comp])['SHARP_HIT_BOOL']
                        .mean().reset_index()
                        .rename(columns={'SHARP_HIT_BOOL': 'Win_Rate'})
                        .sort_values(by=['Market', comp])
                    )
                    st.markdown(f"**📊 {label} by Market**")
                    st.dataframe(result)
            
                    for _, row in result.iterrows():
                        market = row['Market'].lower()
                        val = row[comp]
                        win_rate = row['Win_Rate']
                        market_component_win_rates_sport.setdefault(market, {}).setdefault(comp, {})[val] = win_rate
            
            # 🔁 Update global scoped object
            # 🔁 Update global scoped object
            # 🔁 Update global scoped object
            # 🔁 Update global scoped object
           
            globals()["market_component_win_rates"] = market_component_win_rates
            market_component_win_rates = globals().get("market_component_win_rates", {})
            market_component_win_rates[sport_key_lower] = market_component_win_rates_sport
          

            
            try:
                save_weights_to_drive(market_component_win_rates, drive)
                print(f"✅ Saved weights for {sport_key_lower} to Google Drive.")
            except Exception as e:
                print(f"❌ Failed to save weights for {sport_key_lower}: {e}")

            
            # 📥 Show learned weights in UI
            st.subheader(f"📥 Learned Weights for {sport_key} (Debug)")
            st.json(market_component_win_rates.get(sport_key_lower, {}))
            
            # 🧪 Sample confidence scores for inspection
            st.subheader(f"🧪 Sample {sport_key} Confidence Inputs")
            st.dataframe(df_mlb_bt[[
                'Market', 'Sharp_Move_Signal', 'Sharp_Time_Score', 'True_Sharp_Confidence_Score'
            ]].head())
