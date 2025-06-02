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


API_KEY = "3879659fe861d68dfa2866c211294684"

SPORTS = {"NBA": "basketball_nba", "MLB": "baseball_mlb"}
SHARP_BOOKS = ['pinnacle', 'bookmaker', 'betonlineag']
REC_BOOKS = ['bovada', 'heritagesports', 'betus', 'betmgm', 'bet365', 'draftkings', 'fanduel', 'betrivers', 'pointsbetus']

BOOKMAKER_REGIONS = {
    'pinnacle': 'us', 'bookmaker': 'us', 'betonlineag': 'us',
    'bovada': 'us', 'heritagesports': 'us', 'betus': 'us',
    'betmgm': 'us', 'draftkings': 'us', 'fanduel': 'us', 'betrivers': 'us', 'pointsbetus': 'us2',
    'bet365': 'uk', 'williamhill': 'uk', 'ladbrokes': 'uk', 'unibet': 'eu', 'bwin': 'eu',
    'sportsbet': 'au', 'ladbrokesau': 'au', 'neds': 'au'
}

MARKETS = ['spreads', 'totals', 'h2h']
LOG_FOLDER = "/tmp/sharp_logs"
SNAPSHOT_DIR = "/tmp/sharp_snapshots"
FOLDER_ID = "1v6WB0jRX_yJT2JSdXRvQOLQNfOZ97iGA"


os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

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
def get_snapshot(data):
    return {g['id']: g for g in data}

def save_snapshot(sport_key, snapshot):
    path = os.path.join(SNAPSHOT_DIR, f"{sport_key}_snapshot.pkl")
    with open(path, "wb") as f:
        pickle.dump(snapshot, f)

def load_snapshot(sport_key):
    path = os.path.join(SNAPSHOT_DIR, f"{sport_key}_snapshot.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    return data
                else:
                    st.warning(f"⚠️ Snapshot file for {sport_key} is invalid (not a dict). Resetting...")
                    return {}
        except Exception as e:
            st.error(f"❌ Failed to load snapshot for {sport_key}: {e}")
            return {}
    return {}

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





def fetch_scores_and_backtest(df_moves, sport_key='baseball_mlb', days_back=3, api_key='3879659fe861d68dfa2866c211294684'):
    print(f"🔁 Fetching scores for {sport_key} (last {days_back} days)...")
    
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/scores"
    params = {
        'daysFrom': days_back,
        'apiKey': api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        score_data = response.json()
    except Exception as e:
        print(f"❌ Failed to fetch scores: {e}")
        return df_moves.copy()

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
            'Home': home,
            'Away': away,
            'Home_Score': team_scores[home],
            'Away_Score': team_scores[away]
        })

    df_results = pd.DataFrame(result_rows)
    if df_results.empty:
        print("⚠️ No completed games found.")
        return df_moves.copy()

    # Merge with detected sharp moves
    df = df_moves.merge(df_results, on='Game', how='left')

    def calc_cover(row):
        team = row['Outcome'].lower()
        mkt = row['Market']
        hscore = row['Home_Score']
        ascore = row['Away_Score']
        if pd.isna(hscore) or pd.isna(ascore):
            return None, None

        team_score = hscore if team in row['Home'].lower() else ascore
        opp_score = ascore if team in row['Home'].lower() else hscore
        margin = team_score - opp_score

        # === Determine cover result
        if mkt == 'h2h':
            hit = int(team_score > opp_score)
            return 'Win' if hit else 'Loss', hit
        elif mkt == 'spreads':
            spread = row.get('Ref Sharp Value')
            if spread is None:
                return None, None
            if spread < 0:  # favorite
                hit = int(margin > abs(spread))
            else:  # underdog
                hit = int(margin + spread > 0)
            return 'Win' if hit else 'Loss', hit
        elif mkt == 'totals':
            total = row.get('Ref Sharp Value')
            if total is None:
                return None, None
            total_points = hscore + ascore
            if 'under' in row['Outcome'].lower():
                hit = int(total_points < total)
            elif 'over' in row['Outcome'].lower():
                hit = int(total_points > total)
            else:
                return None, None
            return 'Win' if hit else 'Loss', hit
        return None, None

    # Apply backtest
    df[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']] = df.apply(lambda r: pd.Series(calc_cover(r)), axis=1)

    print(f"✅ Backtested {df['SHARP_HIT_BOOL'].notna().sum()} sharp edges with game results.")
    return df
    
def detect_sharp_moves(current, previous, sport_key):
    from collections import defaultdict

    def normalize_label(label):
        return str(label).strip().lower().replace('.0', '')

    rows = []
    sharp_limit_map = defaultdict(lambda: defaultdict(list))
    snapshot_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}
    sharp_side_flags = {}
    sharp_lines = {}
    sharp_audit_rows = []
    rec_lines = []

    for game in current:
        game_name = f"{game['home_team']} vs {game['away_team']}"
        event_date = pd.to_datetime(game.get("commence_time")).strftime("%Y-%m-%d") if "commence_time" in game else ""
        gid = game['id']
        prev_game = previous_map.get(gid, {})

        for book in game.get('bookmakers', []):
            book_key = book['key']
            for market in book.get('markets', []):
                mtype = market['key']
                for o in market.get('outcomes', []):
                    label = normalize_label(o['name'])
                    val = o.get('point') if mtype != 'h2h' else o.get('price')
                    if val is None:
                        continue
                    limit = o.get('bet_limit') if book_key in SHARP_BOOKS else None

                    entry = {
                        'Sport': sport_key, 'Time': snapshot_time, 'Game': game_name,
                        'Market': mtype, 'Outcome': label, 'Bookmaker': book['title'],
                        'Book': book_key, 'Value': val, 'Limit': limit,
                        'Old Value': None, 'Delta': None, 'Event_Date': event_date, 'Region': BOOKMAKER_REGIONS.get(book_key, 'unknown'),

                    }

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
                                                    entry['Delta'] = round(val - prev_val, 2)

                    if book_key in SHARP_BOOKS:
                        sharp_lines[(game_name, mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit or 0, val, entry.get("Old Value")))
                    elif book_key in REC_BOOKS:
                        rec_lines.append(entry)

    # === Determine sharp side
    for (game_name, mtype), label_map in sharp_limit_map.items():
        scores = {}
        move_signal, limit_jump, prob_shift, time_score = 0, 0, 0, 0
        for label, entries in label_map.items():
            move_signal = limit_jump = prob_shift = time_score = 0
            for limit, curr, old in entries:
                if old is not None and curr is not None:
                    if mtype == 'totals':
                        if 'under' in label and curr < old:
                            move_signal += 1
                        elif 'over' in label and curr > old:
                            move_signal += 1
                    elif mtype == 'spreads' and abs(curr) > abs(old):
                        move_signal += 1
                    elif mtype == 'h2h':
                        imp_now = implied_prob(curr)
                        imp_old = implied_prob(old)
                        if imp_now and imp_old and imp_now > imp_old:
                            prob_shift += 1
                if limit and limit >= 5000:
                    limit_jump += 1
                hour = datetime.now().hour
                time_score += 1.0 if 6 <= hour <= 11 else 0.5 if hour <= 15 else 0.2

            score = 2 * move_signal + 2 * limit_jump + 1.5 * time_score + 1.0 * prob_shift
            scores[label] = score

        if scores:
            best_label = max(scores, key=scores.get)
            sharp_side_flags[(game_name, mtype, best_label)] = 1

    # === Rec Book Evaluation
    for rec in rec_lines:
        rec_label = normalize_label(rec['Outcome'])
        market_type = rec['Market']
        rec_key = (rec['Game'], market_type, rec_label)

        # Smart sharp-side comparison (exact outcome match only)
        is_sharp_side = 0
        sharp_outcome = None
        rec_key = (rec['Game'], market_type, normalize_label(rec['Outcome']))

        for key in sharp_side_flags:
            if rec_key == key:
                is_sharp_side = 1
                sharp_outcome = key[2]
                break

        if not is_sharp_side:
            continue  # skip all non-sharp-side outcomes

        sharp = sharp_lines.get(rec_key)
        if not sharp:
            continue

        implied_rec = implied_prob(rec['Value']) if market_type == 'h2h' else None

        # === No-Vig Sharp Implied Probability (H2H only)
        implied_sharp = None
        if market_type == 'h2h':
            this_price = sharp['Value']
            other_label = None
            for (g, m, lbl) in sharp_lines:
                if g == rec['Game'] and m == market_type and lbl != rec_label:
                    other_label = lbl
                    break
            other_price = sharp_lines.get((rec['Game'], market_type, other_label), {}).get('Value')
            if this_price and other_price:
                p1 = implied_prob(this_price)
                p2 = implied_prob(other_price)
                total = p1 + p2
                implied_sharp = p1 / total if total > 0 else None

        elif market_type != 'h2h':
            implied_sharp = implied_prob(sharp['Value'])

        delta_vs_sharp = rec['Value'] - sharp['Value'] if sharp and rec['Value'] is not None and sharp['Value'] is not None else None

        # === Bias Match
        bias_match = 0
        if market_type == 'spreads' and abs(rec['Value']) < abs(sharp['Value']):
            bias_match = 1
        elif market_type == 'totals':
            if 'under' in rec['Outcome'] and rec['Value'] > sharp['Value']:
                bias_match = 1
            elif 'over' in rec['Outcome'] and rec['Value'] < sharp['Value']:
                bias_match = 1
        elif market_type == 'h2h' and implied_rec and implied_sharp and implied_rec < implied_sharp:
            bias_match = 1

        # === Alignment & Reason
        delta = round(delta_vs_sharp, 2) if delta_vs_sharp is not None else None
        if delta is None:
            alignment = "❓ Unknown or Incomplete"
            reason = "Insufficient pricing data"
        elif abs(delta) >= 0.01:
            alignment = "Sharp move, Rec books not reponded"
            reason = "Sharp move, Rec books not reponded"
        else:
            alignment = "Aligned with Sharps"
            reason = "Rec book has adjusted to sharp line"

        # === Final output row
        row = rec.copy()
        row.update({
            'Event_Date': rec.get('Event_Date', ""),
            'Ref Sharp Value': sharp['Value'] if sharp else None,
            'Delta vs Sharp': delta,
            'Bias Match': bias_match,
            'Implied_Prob_Rec': implied_rec,
            'Implied_Prob_Sharp': implied_sharp,
            'Implied_Prob_Diff': (implied_sharp - implied_rec) if implied_rec and implied_sharp else None,
            'Limit': sharp.get('Limit') if sharp and sharp.get('Limit') is not None else 0,
            'SHARP_SIDE_TO_BET': 1,
            'SharpAlignment': alignment,
            'SHARP_REASON': reason,
            'Sharp_Move_Signal': move_signal,
            'Sharp_Limit_Jump': limit_jump,
            'Sharp_Time_Score': time_score,
            'Sharp_Prob_Shift': prob_shift,
            'Sharp_Outcome_Key': sharp_outcome,
            'SharpBetScore': round(
                2.0 * move_signal +
                2.0 * limit_jump +
                1.5 * time_score +
                1.0 * prob_shift, 2
            )
        })

        rows.append(row)

    df = pd.DataFrame(rows)
    df_audit = pd.DataFrame(sharp_audit_rows)
    print(f"✅ Final sharp-backed rows: {len(df)}")
    return df, df_audit


st.set_page_config(layout="wide")
# === Initialize Google Drive once ===
drive = init_gdrive()

st.title("📊 Sharp Edge Scanner")
auto_mode = st.sidebar.radio("🕹️ Refresh Mode", ["Auto Refresh", "Manual"], index=0)
if auto_mode == "Auto Refresh":
    st_autorefresh(interval=220000, key="autorefresh")

def log_rec_snapshot(df_moves, sport_key):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = f"/tmp/rec_snapshots/{sport_key}_snapshot_{now}.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df_snapshot = df_moves[
        ['Game', 'Market', 'Outcome', 'Bookmaker', 'Value', 'Time']
    ].copy()
    df_snapshot.to_csv(path, index=False)
    print(f"📦 Snapshot saved to: {path}")

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
        live = fetch_live_odds(sport_key)
        prev = load_snapshot(sport_key)

        if not live:
            st.warning(f"No odds returned for {label}.")
            return pd.DataFrame()  # safe fallback

        df_moves, df_audit = detect_sharp_moves(live, prev, label)
        save_snapshot(sport_key, get_snapshot(live))

        if df_moves is None or df_moves.empty:
            st.info(f"⚠️ No sharp moves detected for {label}.")
            return pd.DataFrame()

        # === Validate necessary columns
        if 'SmartSharpScore' not in df_moves.columns:
            df_moves['SmartSharpScore'] = 0
        if 'SharpAlignment' not in df_moves.columns:
            df_moves['SharpAlignment'] = "Unknown"
        if 'SHARP_REASON' not in df_moves.columns:
            df_moves['SHARP_REASON'] = "Reason not available"

        df_display = df_moves.sort_values(by='SmartSharpScore', ascending=False)
        df_display = df_display.drop_duplicates(subset=['Game', 'Market', 'Outcome'], keep='first')

        st.subheader(f"🚨 Detected Sharp Moves – {label}")

        # === Filters
        region_options = ["All"]
        if 'Region' in df_display.columns:
            region_options += sorted(df_display['Region'].dropna().unique())

        region = st.selectbox(
            f"🌍 Filter {label} by Region",
            region_options,
            key=f"{label}_region_main"
        )
        if region != "All":
            df_display = df_display[df_display['Region'] == region]

        market = st.selectbox(f"📊 Filter {label} by Market", ["All"] + sorted(df_display['Market'].dropna().unique()), key=f"{label}_market_main")
        alignment_filter = st.selectbox(
            f"🧭 Sharp Alignment Filter ({label})",
            ["All", "Sharp move, Rec books not reponded", "Aligned with Sharps", "❓ Unknown or Incomplete"],
            key=f"{label}_alignment_main"
        )

        if region != "All":
            df_display = df_display[df_display['Region'] == region]
        if market != "All":
            df_display = df_display[df_display['Market'] == market]
        if alignment_filter != "All":
            df_display = df_display[df_display['SharpAlignment'] == alignment_filter]

        # === Display Table
        display_cols = [
            'Event_Date', 'Game', 'Market', 'Outcome', 'Bookmaker',
            'Value', 'Ref Sharp Value', 'LineMove',
            'Delta vs Sharp', 'Limit', 'SharpConfidenceTier', 'SharpAlignment', 'SHARP_REASON'
        ]
        safe_cols = [col for col in display_cols if col in df_display.columns]

        if not df_display.empty:
            def highlight_edge(row):
                align = row.get('SharpAlignment')
                if align == "🚨 Edge (better than sharps)":
                    return ['background-color: #d4edda; color: black'] * len(row)
                elif align == "⚠️ Worse than sharps":
                    return ['background-color: #ffcccc; color: black'] * len(row)
                elif align == "✅ Matched with sharps":
                    return ['background-color: #fff3cd; color: black'] * len(row)
                return ['background-color: white; color: black'] * len(row)

            styled_df = df_display[safe_cols].style.apply(highlight_edge, axis=1)
            st.dataframe(styled_df, use_container_width=True)

            # === Save & Upload
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f"{label}_sharp_moves_{timestamp}.csv"
            audit_fname = f"{label}_sharp_audit_{timestamp}.csv"

            csv_path = os.path.join(LOG_FOLDER, fname)
            audit_path = os.path.join(LOG_FOLDER, audit_fname)

            df_display[safe_cols].to_csv(csv_path, index=False)
            df_audit.to_csv(audit_path, index=False)

            try:
                gfile_audit = drive.CreateFile({'title': audit_fname, 'parents': [{'id': FOLDER_ID}]})
                gfile_audit.SetContentFile(audit_path)
                gfile_audit.Upload()
                st.success(f"📄 Audit uploaded: {audit_fname}")
            except Exception as e:
                st.error(f"❌ Audit upload failed: {e}")

            try:
                gfile = drive.CreateFile({'title': fname, 'parents': [{'id': FOLDER_ID}]})
                gfile.SetContentFile(csv_path)
                gfile.Upload()
                st.success(f"✅ Uploaded to Google Drive: {fname}")
                st.caption(f"📁 [Sharp Logs Folder](https://drive.google.com/drive/folders/{FOLDER_ID})")
            except Exception as e:
                st.error(f"❌ Sharp CSV upload failed: {e}")
        else:
            st.info(f"⚠️ No results match the selected filters.")

        # === Drift Viewer
        with st.expander(f"🔍 Drift Tracker (Rec Book Lag) – {label}"):
            if not df_moves.empty and 'Game' in df_moves.columns:
                game_opts = sorted(df_moves['Game'].dropna().unique())
                selected_game = st.selectbox("Select game", game_opts, key=f"{label}_drift_game")
                outcomes = df_moves[df_moves['Game'] == selected_game]['Outcome'].dropna().unique()
                selected_outcome = st.selectbox("Select outcome", sorted(outcomes), key=f"{label}_drift_outcome")
                if st.button("Show Drift", key=f"{label}_drift_btn"):
                    drift_df = track_rec_drift(selected_game, selected_outcome)
                    st.dataframe(drift_df)

        # === Current Odds
        st.subheader(f"📋 Current Odds – {label}")
        odds = []
        for game in live:
            game_name = f"{game['home_team']} vs {game['away_team']}"
            event_date = pd.to_datetime(game.get("commence_time")).strftime("%Y-%m-%d") if "commence_time" in game else ""
            for book in game.get('bookmakers', []):
                for market in book.get('markets', []):
                    for o in market.get('outcomes', []):
                        val = o.get('point') if market['key'] != 'h2h' else o.get('price')
                        odds.append({
                            'Event_Date': event_date,
                            'Game': game_name,
                            'Market': market['key'],
                            'Outcome': o['name'],
                            'Bookmaker': book['title'],
                            'Value': val
                        })

        df_odds = pd.DataFrame(odds)
        if not df_odds.empty:
            pivot = df_odds.pivot_table(index=['Event_Date', 'Game', 'Market', 'Outcome'], columns='Bookmaker', values='Value')
            st.dataframe(pivot.reset_index(), use_container_width=True)

        # === Manual Refresh Button
        if auto_mode == "Manual":
            if st.button(f"🔄 Refresh {label}", key=f"{label}_manual_refresh"):
                st.rerun()

        # === Save snapshot
        log_rec_snapshot(df_moves, sport_key)

    return df_moves


tab_nba, tab_mlb = st.tabs(["🏀 NBA", "⚾ MLB"])

# Get sharp edges
df_nba = render_scanner_tab("NBA", SPORTS["NBA"], tab_nba, drive)
df_mlb = render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb, drive)

# Backtest and show performance
if df_nba is not None and not df_nba.empty:
    df_nba_bt = fetch_scores_and_backtest(df_nba, sport_key='basketball_nba')
    st.subheader("📊 NBA Sharp Signal Performance")
    st.dataframe(
        df_nba_bt.groupby('SharpConfidenceTier').agg(
            Total_Picks=('SHARP_HIT_BOOL', 'count'),
            Hits=('SHARP_HIT_BOOL', 'sum'),
            Win_Rate=('SHARP_HIT_BOOL', 'mean')
        ).round(3).reset_index()
    )

if df_mlb is not None and not df_mlb.empty:
    df_mlb_bt = fetch_scores_and_backtest(df_mlb, sport_key='baseball_mlb')
    st.subheader("📊 MLB Sharp Signal Performance")
    st.dataframe(
        df_mlb_bt.groupby('SharpConfidenceTier').agg(
            Total_Picks=('SHARP_HIT_BOOL', 'count'),
            Hits=('SHARP_HIT_BOOL', 'sum'),
            Win_Rate=('SHARP_HIT_BOOL', 'mean')
        ).round(3).reset_index()
    )
# 🧠 Sharp Signal Learning (Component Breakdown)
if df_nba_bt is not None and not df_nba_bt.empty:
    st.subheader("🧠 Sharp Component Learning – NBA")
    st.dataframe(df_nba_bt.groupby('Sharp_Move_Signal')['SHARP_HIT_BOOL'].mean().reset_index().rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Move_Signal'}))
    st.dataframe(df_nba_bt.groupby('Sharp_Time_Score')['SHARP_HIT_BOOL'].mean().reset_index().rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Time_Score'}))

if df_mlb_bt is not None and not df_mlb_bt.empty:
    st.subheader("🧠 Sharp Component Learning – MLB")
    st.dataframe(df_mlb_bt.groupby('Sharp_Move_Signal')['SHARP_HIT_BOOL'].mean().reset_index().rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Move_Signal'}))
    st.dataframe(df_mlb_bt.groupby('Sharp_Time_Score')['SHARP_HIT_BOOL'].mean().reset_index().rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Time_Score'}))
