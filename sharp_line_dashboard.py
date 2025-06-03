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

API_KEY = "3879659fe861d68dfa2866c211294684"

SPORTS = {"NBA": "basketball_nba", "MLB": "baseball_mlb"}
SHARP_BOOKS = ['pinnacle', 'bookmaker', 'betonlineag','betfair_ex_eu','smarkets','matchbook']
REC_BOOKS = ['bovada', 'heritagesports', 'betus', 'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers', 'pointsbetus']

BOOKMAKER_REGIONS = {
    'pinnacle': 'us', 'bookmaker': 'us', 'betonlineag': 'us',
    'bovada': 'us', 'heritagesports': 'us', 'betus': 'us',
    'betmgm': 'us', 'draftkings': 'us', 'fanduel': 'us', 'betrivers': 'us', 'pointsbetus': 'us2',
    'bet365': 'uk', 'williamhill': 'uk', 'ladbrokes': 'uk', 'unibet': 'eu', 'bwin': 'eu',
    'sportsbet': 'au', 'ladbrokesau': 'au', 'neds': 'au'
}

MARKETS = ['spreads', 'totals', 'h2h']


FOLDER_ID = "1v6WB0jRX_yJT2JSdXRvQOLQNfOZ97iGA"

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

def fetch_scores_and_backtest(df_moves, sport_key='baseball_mlb', days_back=3, api_key='3879659fe861d68dfa2866c211294684'):
    print(f"🔁 Fetching scores for {sport_key} (last {days_back} days)...")

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
    params = {'daysFrom': days_back, 'apiKey': api_key}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        score_data = response.json()
    except Exception as e:
        print(f"❌ Failed to fetch scores: {e}")
        return pd.DataFrame()

    # Build results table
    result_rows = []
    for game in score_data:
        if not game.get("completed"):
            continue
        home = game.get("home_team")
        away = game.get("away_team")
        scores = game.get("scores", [])
        team_scores = {s["name"]: s["score"] for s in scores if "name" in s and "score" in s}

        if home not in team_scores or away not in team_scores:
            continue

        game_name = f"{home} vs {away}"
        result_rows.append({
            'Game': game_name,
            'Home_Team': home,
            'Away_Team': away,
            'Home_Score': team_scores[home],
            'Away_Score': team_scores[away]
        })

    df_results = pd.DataFrame(result_rows)
    if df_results.empty:
        print("⚠️ No completed games found.")
        return pd.DataFrame()

    # Deduplicate and merge
    df_moves = df_moves.drop_duplicates(subset=['Game', 'Market', 'Outcome'])
    df = df_moves.merge(df_results, on='Game', how='left')


    return df  # ✅ THIS LINE IS REQUIRED
    # === Refactored cover logic
def calc_cover(row):
    team = str(row['Outcome']).strip().lower()
    home = str(row['Home_Team']).strip().lower()
    away = str(row['Away_Team']).strip().lower()
    hscore = row['Home_Score']
    ascore = row['Away_Score']
    market = str(row['Market']).strip().lower()

    if pd.isna(hscore) or pd.isna(ascore):
        return None, None

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
        return 'Win' if hit else 'Loss', hit

    if market == 'spreads':
        spread = row.get('Ref Sharp Value')
        if spread is None or not isinstance(spread, (int, float)):
            return None, None
        hit = int((margin > abs(spread)) if spread < 0 else (margin + spread > 0))
        return 'Win' if hit else 'Loss', hit

    if market == 'totals':
        total = row.get('Ref Sharp Value')
        if total is None or not isinstance(total, (int, float)):
            return None, None
        total_points = hscore + ascore
        if 'under' in team:
            hit = int(total_points < total)
        elif 'over' in team:
            hit = int(total_points > total)
        else:
            return None, None
        return 'Win' if hit else 'Loss', hit

    return None, None


    # Apply result scoring
    df[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']] = df.apply(lambda r: pd.Series(calc_cover(r)), axis=1)

    print(f"✅ Backtested {df['SHARP_HIT_BOOL'].notna().sum()} sharp edges with game results.")
    return df



from collections import defaultdict
import pandas as pd
from datetime import datetime

def detect_sharp_moves(current, previous, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS):
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

    for game in current:
        game_name = f"{game['home_team']} vs {game['away_team']}"
        event_date = pd.to_datetime(game.get("commence_time")).strftime("%Y-%m-%d") if game.get("commence_time") else ""
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
                    if key not in line_open_map and val is not None:
                        line_open_map[key] = (val, snapshot_time)

                    entry = {
                        'Sport': sport_key, 'Time': snapshot_time, 'Game': game_name,
                        'Market': mtype, 'Outcome': label, 'Bookmaker': book['title'],
                        'Book': book_key, 'Value': val, 'Limit': limit,
                        'Old Value': None, 'Delta': None, 'Event_Date': event_date,
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
                                                    open_val, _ = line_open_map.get((game_name, mtype, label), (None, None))
                                                    entry['Old Value'] = open_val
                                                    entry['Delta'] = round(val - open_val, 2) if open_val is not None else None


                    rows.append(entry)

                    # Track sharp data if limit is present
                    if limit is not None:
                        sharp_lines[(game_name, mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit, val, entry.get('Old Value')))
                        if book_key in SHARP_BOOKS:
                            sharp_total_limit_map[(game_name, mtype, label)] += limit

    # === Sharp scoring logic
    for (game_name, mtype), label_map in sharp_limit_map.items():
        scores = {}
        label_signals = {}
        for label, entries in label_map.items():
            move_signal = limit_jump = prob_shift = time_score = 0
            for limit, curr, _ in entries:
                open_val, _ = line_open_map.get((game_name, mtype, label), (None, None))
                if open_val is not None and curr is not None:
                    if mtype == 'totals':
                        if 'under' in label and curr < open_val: move_signal += 1
                        elif 'over' in label and curr > open_val: move_signal += 1
                    elif mtype == 'spreads' and abs(curr) > abs(open_val): move_signal += 1
                    elif mtype == 'h2h':
                        imp_now, imp_open = implied_prob(curr), implied_prob(open_val)
                        if imp_now and imp_open and imp_now > imp_open: prob_shift += 1

                if limit and limit >= 100: limit_jump += 1
                hour = datetime.now().hour
                time_score += 1.0 if 6 <= hour <= 11 else 0.5 if hour <= 15 else 0.2

            total_limit = sharp_total_limit_map.get((game_name, mtype, label), 0)
            scores[label] = (
                2 * move_signal +
                2 * limit_jump +
                1.5 * time_score +
                1.0 * prob_shift +
                0.001 * total_limit  # adjust this weight if needed
            )
            
            label_signals[label] = {
                'Sharp_Move_Signal': move_signal,
                'Sharp_Limit_Jump': limit_jump,
                'Sharp_Time_Score': time_score,
                'Sharp_Prob_Shift': prob_shift,
                'Sharp_Limit_Total': total_limit
            }


        if scores:
            best_label = max(scores, key=scores.get)
            sharp_side_flags[(game_name, mtype, best_label)] = 1
            sharp_metrics_map[(game_name, mtype, best_label)] = label_signals[best_label]

    # === Append sharp-sided bets with scores
    for (game_name, mtype, label), entry in sharp_lines.items():
        if sharp_side_flags.get((game_name, mtype, label), 0):
            metrics = sharp_metrics_map.get((game_name, mtype, label), {})
            enriched = entry.copy()
            enriched.update({
                'Ref Sharp Value': entry['Value'],
                'Ref Sharp Old Value': entry.get('Old Value'),
                'Delta vs Sharp': 0.0,
                'SHARP_SIDE_TO_BET': 1,
                'SharpBetScore': round(
                    2.0 * metrics.get('Sharp_Move_Signal', 0) +
                    2.0 * metrics.get('Sharp_Limit_Jump', 0) +
                    1.5 * metrics.get('Sharp_Time_Score', 0) +
                    1.0 * metrics.get('Sharp_Prob_Shift', 0) +
                    0.001 * metrics.get('Sharp_Limit_Total', 0), 2
                ),

                'Sharp_Move_Signal': metrics.get('Sharp_Move_Signal', 0),
                'Sharp_Limit_Jump': metrics.get('Sharp_Limit_Jump', 0),
                'Sharp_Time_Score': metrics.get('Sharp_Time_Score', 0),
                'Sharp_Prob_Shift': metrics.get('Sharp_Prob_Shift', 0),
                'Sharp_Limit_Total': metrics.get('Sharp_Limit_Total', 0),

            })
            rows.append(enriched)

    # === Intelligence scoring
    def compute_intelligence_score(row):
        score = 0
        reasons = []
        if row.get('Limit_Imbalance', 0) >= 2500:
            score += 15
            reasons.append("💰 High limit spread")
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

    # === Final DataFrame output
    df = pd.DataFrame(rows)
    df_history = pd.DataFrame(line_history_log)
    
    # Sort history and extract true opening line per Game × Market × Outcome
    df_history_sorted = df_history.sort_values('Time')
    line_open_df = (
        df_history_sorted
        .dropna(subset=['Value'])
        .groupby(['Game', 'Market', 'Outcome'])['Value']
        .first()
        .reset_index()
        .rename(columns={'Value': 'Open_Value'})
    )
    
    # Don't return early before merging!
    if df.empty:
        return df, df_history
    
    # Merge in opening line value and compute true market delta
    df = df.merge(line_open_df, on=['Game', 'Market', 'Outcome'], how='left')
    df['Delta vs Sharp'] = df['Value'] - df['Open_Value']
    
    # Additional computed fields
    df['Delta'] = pd.to_numeric(df['Delta vs Sharp'], errors='coerce')
    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)
    df['Limit_Jump'] = (df['Limit'] >= 2500).astype(int)
    df['Sharp_Timing'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour.apply(
        lambda h: 1.0 if 6 <= h <= 11 else 0.5 if h <= 15 else 0.2
    )
    df['Limit_Max'] = df.groupby(['Game', 'Market'])['Limit'].transform('max')
    df['Limit_Min'] = df.groupby(['Game', 'Market'])['Limit'].transform('min')
    df['Limit_Imbalance'] = df['Limit_Max'] - df['Limit_Min']
    df['Asymmetry_Flag'] = (df['Limit_Imbalance'] >= 2500).astype(int)
    df[['SharpIntelligenceScore', 'SharpIntelReasons']] = df.apply(compute_intelligence_score, axis=1)
    
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
    
    # Compute consensus deltas from open
    summary_df['Move_From_Open_Rec'] = summary_df['Rec_Book_Consensus'] - summary_df['Rec_Open']
    summary_df['Move_From_Open_Sharp'] = summary_df['Sharp_Book_Consensus'] - summary_df['Sharp_Open']
    
    # Round for clarity
    summary_df = summary_df.round({
        'Rec_Book_Consensus': 2,
        'Sharp_Book_Consensus': 2,
        'Move_From_Open_Rec': 2,
        'Move_From_Open_Sharp': 2
    })
    
    summary_df['Recommended_Outcome'] = summary_df['Outcome']
    
    # ✅ Merge SharpBetScore only for sharp-backed outcomes
    sharp_scores = df_moves[['Game', 'Market', 'Outcome', 'SharpBetScore']].drop_duplicates()
    summary_df = summary_df.merge(
        sharp_scores,
        on=['Game', 'Market', 'Outcome'],
        how='left'
    )
    
    # (Optional) Fill empty scores with 0 or drop them
    summary_df['SharpBetScore'] = summary_df['SharpBetScore'].fillna(0)
    
        
    # ✅ Return all three
    print(f"✅ Final sharp-backed rows: {len(df)}")
    return df, df_history, summary_df

st.set_page_config(layout="wide")
# === Initialize Google Drive once ===
drive = init_gdrive()

st.title("📊 Sharp Edge Scanner")
auto_mode = st.sidebar.radio("🕹️ Refresh Mode", ["Auto Refresh", "Manual"], index=0)
if auto_mode == "Auto Refresh":
    st_autorefresh(interval=220000, key="autorefresh")

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
    with container:

        df_moves = pd.DataFrame()
        live = fetch_live_odds(sport_key)
        prev = load_latest_snapshot_from_drive(sport_key, drive, FOLDER_ID)

        if not prev:
            st.info("🟡 First run detected — saving snapshot and skipping detection.")
            upload_snapshot_to_drive(sport_key, get_snapshot(live), drive, FOLDER_ID)
            return pd.DataFrame()

        if not live or not isinstance(live, list) or len(live) == 0:
            st.warning(f"⚠️ No live odds returned for {label}.")
            return pd.DataFrame()

        try:
            df_moves, df_audit, summary_df = detect_sharp_moves(live, prev, label, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS)


            # ✅ Add timestamp to both dataframes
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not df_moves.empty:
                df_moves['Snapshot_Timestamp'] = timestamp
                df_moves['Sport'] = label
                append_to_master_csv_on_drive(df_moves, "sharp_moves_master.csv", drive, FOLDER_ID)
            if not df_audit.empty:
                df_audit['Snapshot_Timestamp'] = timestamp
                append_to_master_csv_on_drive(df_audit, "line_history_master.csv", drive, FOLDER_ID)

        except Exception as e:
            st.error(f"❌ Error in detect_sharp_moves: {e}")
            return pd.DataFrame()

        upload_snapshot_to_drive(sport_key, get_snapshot(live), drive, FOLDER_ID)



        # === Show sharp moves first
        # === Show summarized sharp consensus movement table instead
        if summary_df is None or summary_df.empty:
            st.info(f"⚠️ No sharp consensus movements detected for {label}.")
        else:
            st.subheader(f"📊 Sharp vs Rec Book Consensus Summary – {label}")
            
            market_options = ["All"] + sorted(summary_df['Market'].dropna().unique())
            market = st.selectbox(f"📊 Filter {label} by Market", market_options, key=f"{label}_market_summary")
        
            filtered_df = summary_df.copy()
            if market != "All":
                filtered_df = filtered_df[filtered_df['Market'] == market]
        
            st.dataframe(filtered_df[
                [
                    'Event_Date', 'Game', 'Market', 'Recommended_Outcome',
                    'Rec_Book_Consensus', 'Sharp_Book_Consensus',
                    'Move_From_Open_Rec', 'Move_From_Open_Sharp'
                ]
            ], use_container_width=True)



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

df_nba = render_scanner_tab("NBA", SPORTS["NBA"], tab_nba, drive)
df_mlb = render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb, drive)

# Upload sharp moves to master file
if df_nba is not None and not df_nba.empty:
    df_nba['Sport'] = 'NBA'
    append_to_master_csv_on_drive(df_nba, "sharp_moves_master.csv", drive, FOLDER_ID)
if df_mlb is not None and not df_mlb.empty:
    df_mlb['Sport'] = 'MLB'
    append_to_master_csv_on_drive(df_mlb, "sharp_moves_master.csv", drive, FOLDER_ID)

# Safe predefinition
df_nba_bt = pd.DataFrame()
df_mlb_bt = pd.DataFrame()

# Load and evaluate full history
df_master = load_master_sharp_moves(drive)

if df_master.empty:
    st.warning("⚠️ No historical sharp picks found in Google Drive yet.")
else:
    df_nba_bt = fetch_scores_and_backtest(df_master[df_master['Sport'] == 'NBA'], sport_key='basketball_nba')
    df_mlb_bt = fetch_scores_and_backtest(df_master[df_master['Sport'] == 'MLB'], sport_key='baseball_mlb')

    # === NBA Sharp Signal Performance
    if not df_nba_bt.empty and 'SHARP_HIT_BOOL' in df_nba_bt.columns:
        df_nba_bt['SharpConfidenceTier'] = pd.cut(
            df_nba_bt['SharpBetScore'],
            bins=[0, 15, 25, 40, 100],
            labels=["⚠️ Low", "✅ Moderate", "⭐ High", "🔥 Steam"]
        )

        st.subheader("📊 NBA Sharp Signal Performance")
        st.dataframe(
            df_nba_bt.groupby('SharpConfidenceTier').agg(
                Total_Picks=('SHARP_HIT_BOOL', 'count'),
                Hits=('SHARP_HIT_BOOL', 'sum'),
                Win_Rate=('SHARP_HIT_BOOL', 'mean')
            ).round(3).reset_index()
        )

        st.subheader("🧠 Sharp Component Learning – NBA")
        st.dataframe(
            df_nba_bt.groupby('Sharp_Move_Signal')['SHARP_HIT_BOOL']
            .mean().reset_index()
            .rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Move_Signal'})
        )
        st.dataframe(
            df_nba_bt.groupby('Sharp_Time_Score')['SHARP_HIT_BOOL']
            .mean().reset_index()
            .rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Time_Score'})
        )
    else:
        st.warning("⚠️ NBA backtest missing 'SHARP_HIT_BOOL'. No results to summarize.")

    # === MLB Sharp Signal Performance
    if not df_mlb_bt.empty and 'SHARP_HIT_BOOL' in df_mlb_bt.columns:
        df_mlb_bt['SharpConfidenceTier'] = pd.cut(
            df_mlb_bt['SharpBetScore'],
            bins=[0, 15, 25, 40, 100],
            labels=["⚠️ Low", "✅ Moderate", "⭐ High", "🔥 Steam"]
        )

        st.subheader("📊 MLB Sharp Signal Performance")
        st.dataframe(
            df_mlb_bt.groupby('SharpConfidenceTier').agg(
                Total_Picks=('SHARP_HIT_BOOL', 'count'),
                Hits=('SHARP_HIT_BOOL', 'sum'),
                Win_Rate=('SHARP_HIT_BOOL', 'mean')
            ).round(3).reset_index()
        )

        st.subheader("🧠 Sharp Component Learning – MLB")
        st.dataframe(
            df_mlb_bt.groupby('Sharp_Move_Signal')['SHARP_HIT_BOOL']
            .mean().reset_index()
            .rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Move_Signal'})
        )
        st.dataframe(
            df_mlb_bt.groupby('Sharp_Time_Score')['SHARP_HIT_BOOL']
            .mean().reset_index()
            .rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Time_Score'})
        )
    else:
        st.warning("⚠️ MLB backtest missing 'SHARP_HIT_BOOL'. No results to summarize.")


