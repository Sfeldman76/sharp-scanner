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

API_KEY = "4f95ea43cc1c29cd44c40fe59b6c14ce"

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
def detect_sharp_moves(current, previous, sport_key):
    from collections import defaultdict

    def normalize_label(label):
        return str(label).strip().lower().replace('.0', '')

    rows = []
    sharp_limit_map = defaultdict(lambda: defaultdict(list))
    snapshot_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}

    for game in current:
        game_name = f"{game['home_team']} vs {game['away_team']}"
        gid = game['id']
        prev_game = previous_map.get(gid, {})
        sharp_lines = {}
        rec_lines = []

        for book in game.get('bookmakers', []):
            book_key = book['key']
            region = BOOKMAKER_REGIONS.get(book_key, 'unknown')
            for market in book.get('markets', []):
                mtype = market['key']
                for o in market.get('outcomes', []):
                    val = o.get('point') if mtype != 'h2h' else o.get('price')
                    label = normalize_label(o['name'])
                    limit = o.get('bet_limit') if book_key in SHARP_BOOKS else None
                    entry = {
                        'Sport': sport_key, 'Time': snapshot_time, 'Game': game_name,
                        'Market': mtype, 'Outcome': label, 'Bookmaker': book['title'],
                        'Book': book_key, 'Region': region, 'Value': val, 'Limit': limit,
                        'Old Value': None, 'Delta': None
                    }

                    if prev_game:
                        for prev_b in prev_game.get('bookmakers', []):
                            if prev_b['key'] == book_key:
                                for prev_m in prev_b.get('markets', []):
                                    if prev_m['key'] == mtype:
                                        for prev_o in prev_m.get('outcomes', []):
                                            if normalize_label(prev_o['name']) == label:
                                                prev_val = prev_o.get('point') if mtype != 'h2h' else prev_o.get('price')
                                                if prev_val is not None and val is not None:
                                                    entry['Old Value'] = prev_val
                                                    entry['Delta'] = round(val - prev_val, 2)

                    if book_key in SHARP_BOOKS:
                        sharp_lines[(mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit or 0, val))
                    elif book_key in REC_BOOKS:
                        rec_lines.append(entry)

        # === Sharp side assignment logic
        sharp_side_flags = {}
        for (game_mkt, label_map) in sharp_limit_map.items():
            labels = list(label_map.keys())
            if len(labels) == 1:
                sharp_side_flags[(game_mkt[0], game_mkt[1], labels[0])] = 1
                continue

            l0, l1 = labels[0], labels[1]
            prices0 = [x[1] for x in label_map[l0] if x[1] is not None]
            prices1 = [x[1] for x in label_map[l1] if x[1] is not None]
            mtype = game_mkt[1]

            if not prices0 or not prices1:
                fallback_label = l0 if prices0 else l1
                sharp_side_flags[(game_mkt[0], mtype, fallback_label)] = 1
                continue

            avg0 = round(sum(prices0) / len(prices0), 2)
            avg1 = round(sum(prices1) / len(prices1), 2)

            if mtype == 'totals' and avg0 == avg1:
                continue

            if mtype == 'totals':
                l0_under = 'under' in l0
                l1_under = 'under' in l1
                if l0_under and avg0 < avg1:
                    sharp_side_label = l0
                elif l1_under and avg1 < avg0:
                    sharp_side_label = l1
                elif not l0_under and avg0 > avg1:
                    sharp_side_label = l0
                else:
                    sharp_side_label = l1
            else:
                score0 = sum([x[0] for x in label_map[l0]]) + avg0 * 2
                score1 = sum([x[0] for x in label_map[l1]]) + avg1 * 2
                sharp_side_label = l0 if score0 > score1 else l1

            sharp_side_flags[(game_mkt[0], mtype, sharp_side_label)] = 1

        # === Final build
        for rec in rec_lines:
            key = (rec['Market'], normalize_label(rec['Outcome']))
            sharp = sharp_lines.get(key)

            if (rec['Game'], rec['Market'], normalize_label(rec['Outcome'])) in sharp_side_flags:
                row = rec.copy()

                implied_rec = implied_prob(rec['Value']) if rec['Market'] == 'h2h' else None
                implied_sharp = implied_prob(sharp['Value']) if sharp and sharp['Market'] == 'h2h' else None

                delta_vs_sharp = rec['Value'] - sharp['Value'] if sharp and rec['Value'] is not None and sharp['Value'] is not None else None
                bias_match = 0

                if rec['Market'] == 'spreads':
                    if sharp and abs(rec['Value']) < abs(sharp['Value']):
                        bias_match = 1
                elif rec['Market'] == 'totals':
                    if "under" in rec['Outcome'] and sharp and rec['Value'] > sharp['Value']:
                        bias_match = 1
                    elif "over" in rec['Outcome'] and sharp and rec['Value'] < sharp['Value']:
                        bias_match = 1
                elif rec['Market'] == 'h2h' and implied_rec is not None and implied_sharp is not None:
                    if implied_rec < implied_sharp:
                        bias_match = 1

                # === Direction-aware sharp alignment logic
                alignment = "🚨 Edge vs sharps"  # default
                if rec['Market'] == 'h2h' and implied_rec is not None and implied_sharp is not None:
                    alignment = "✅ Aligned with sharps" if implied_rec <= implied_sharp else "🚨 Edge vs sharps"
                elif rec['Market'] == 'spreads' and sharp and rec['Value'] is not None and sharp['Value'] is not None:
                    alignment = "✅ Aligned with sharps" if abs(rec['Value']) <= abs(sharp['Value']) else "🚨 Edge vs sharps"
                elif rec['Market'] == 'totals' and sharp and rec['Value'] is not None and sharp['Value'] is not None:
                    if "under" in rec['Outcome']:
                        alignment = "✅ Aligned with sharps" if rec['Value'] >= sharp['Value'] else "🚨 Edge vs sharps"
                    elif "over" in rec['Outcome']:
                        alignment = "✅ Aligned with sharps" if rec['Value'] <= sharp['Value'] else "🚨 Edge vs sharps"

                row.update({
                    'Ref Sharp Value': sharp['Value'] if sharp else None,
                    'Delta vs Sharp': round(delta_vs_sharp, 2) if delta_vs_sharp is not None else None,
                    'Bias Match': bias_match,
                    'Implied_Prob_Rec': implied_rec,
                    'Implied_Prob_Sharp': implied_sharp,
                    'Implied_Prob_Diff': (implied_sharp - implied_rec) if implied_rec and implied_sharp else None,
                    'Limit': sharp.get('Limit') if sharp and sharp.get('Limit') is not None else 0,
                    'SHARP_SIDE_TO_BET': 1,
                    'SharpAlignment': alignment,
                    'SHARP_REASON': "📈 Sharp side backed by limit bias, price delta, and alignment"
                })

                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ No sharp-backed rec lines made it through.")
        return df

    df['LineMove'] = df.apply(
        lambda r: round(r['Value'] - r['Old Value'], 2) if pd.notnull(r['Old Value']) else None,
        axis=1
    )

    df['Delta'] = pd.to_numeric(df['Delta vs Sharp'], errors='coerce')
    df['Delta_Abs'] = df['Delta'].abs()
    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)
    df['Limit_Jump'] = (df['Limit'] >= 5000).astype(int)
    df['Sharp_Timing'] = pd.to_datetime(df['Time']).dt.hour.apply(lambda h: 1.0 if 6 <= h <= 11 else 0.5 if h <= 15 else 0.2)
    df['Limit_Max'] = df.groupby(['Game', 'Market'])['Limit'].transform('max')
    df['Limit_Min'] = df.groupby(['Game', 'Market'])['Limit'].transform('min')
    df['Limit_Imbalance'] = df['Limit_Max'] - df['Limit_Min']
    df['Asymmetry_Flag'] = (df['Limit_Imbalance'] >= 5000).astype(int)

    df['SmartSharpScore'] = (
        5 * df['Bias Match'] +
        4 * df['SHARP_SIDE_TO_BET'] +
        2 * df['Limit_Jump'] +
        2 * df['Delta_Abs'] +
        2 * df['Sharp_Timing'] +
        1 * df['Asymmetry_Flag']
    ).round(2)

    def assign_confidence_tier(score):
        if score >= 40:
            return "🔥 Steam"
        elif score >= 25:
            return "⭐ High"
        elif score >= 15:
            return "✅ Moderate"
        else:
            return "⚠️ Low"

    df['SharpConfidenceTier'] = df['SmartSharpScore'].apply(assign_confidence_tier)

    print("✅ Final sharp-backed rows:", len(df))
    return df


st.set_page_config(layout="wide")
# === Initialize Google Drive once ===
drive = init_gdrive()

st.title("📊 Sharp Edge Scanner")
auto_mode = st.sidebar.radio("🕹️ Refresh Mode", ["Auto Refresh", "Manual"], index=0)
if auto_mode == "Auto Refresh":
    st_autorefresh(interval=120000, key="autorefresh")

def render_scanner_tab(label, sport_key, container, drive):
    with container:
        live = fetch_live_odds(sport_key)
        prev = load_snapshot(sport_key)

        if not live:
            st.warning(f"No odds returned for {label}.")
            return

        # === Process sharp moves
        df_moves = detect_sharp_moves(live, prev, label)
        save_snapshot(sport_key, get_snapshot(live))  # persist new snapshot

        print(f"📊 {label} rows returned:", len(df_moves) if df_moves is not None else "None")

        if df_moves is not None and not df_moves.empty:
            df_display = df_moves.sort_values(by='SmartSharpScore', ascending=False)
            df_display = df_display.drop_duplicates(subset=['Game', 'Market', 'Outcome'], keep='first')

            st.subheader("🚨 Detected Sharp Moves")

            # === Filters
           
            region = st.selectbox(f"🌍 Filter {label} by Region", ["All"] + sorted(df_display['Region'].unique()), key=f"{label}_region")
            market = st.selectbox(f"📊 Filter {label} by Market", ["All"] + sorted(df_display['Market'].unique()), key=f"{label}_market")
            alignment_filter = st.selectbox("🧭 Sharp Alignment Filter", ["All", "✅ Aligned with sharps", "🚨 Edge vs sharps"], key=f"{label}_alignment")

            if region != "All":
                df_display = df_display[df_display['Region'] == region]
            if market != "All":
                df_display = df_display[df_display['Market'] == market]
            if alignment_filter != "All" and 'SharpAlignment' in df_display.columns:
                df_display = df_display[df_display['SharpAlignment'] == alignment_filter]

            # === Final displayed columns (safe fallback)
            available_cols = df_display.columns.tolist()
            safe_cols = [col for col in [
                'Game', 'Market', 'Outcome', 'Bookmaker',
                'Value', 'Ref Sharp Value', 'LineMove',
                'Delta vs Sharp', 'Limit', 'SharpConfidenceTier', 'SharpAlignment', 'SHARP_REASON'
            ] if col in available_cols]

            print("🧪 Displaying columns:", safe_cols)

            # === Display logic
            if not df_display.empty:
                def highlight_edge(row):
                    delta = row.get('Delta vs Sharp', 0)
                    if abs(delta) >= 2:
                        return ['background-color: #ffcccc; color: black'] * len(row)
                    elif abs(delta) >= 1:
                        return ['background-color: #fff3cd; color: black'] * len(row)
                    else:
                        return ['background-color: #d4edda; color: black'] * len(row)

                styled_df = df_display[safe_cols].style.apply(highlight_edge, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                # === Save + Upload CSV
                fname = f"{label}_sharp_moves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                csv_path = os.path.join(LOG_FOLDER, fname)
                df_display[safe_cols].to_csv(csv_path, index=False)

                try:
                    gfile = drive.CreateFile({
                        'title': fname,
                        'parents': [{'id': FOLDER_ID}]
                    })
                    gfile.SetContentFile(csv_path)
                    gfile.Upload()
                    st.success(f"✅ Uploaded to Google Drive: {fname}")
                    st.caption(f"📁 [Sharp Logs Folder](https://drive.google.com/drive/folders/{FOLDER_ID})")
                except Exception as e:
                    st.error(f"❌ Google Drive Upload Failed: {e}")
            else:
                st.info(f"⚠️ No results match the selected filters.")
        else:
            st.info(f"⚠️ No sharp moves detected for {label}.")

        # === Current Odds View
        st.subheader("📋 Current Odds")
        odds = []
        for game in live:
            game_name = f"{game['home_team']} vs {game['away_team']}"
            for book in game.get('bookmakers', []):
                for market in book.get('markets', []):
                    for o in market.get('outcomes', []):
                        val = o.get('point') if market['key'] != 'h2h' else o.get('price')
                        odds.append({
                            'Game': game_name,
                            'Market': market['key'],
                            'Outcome': o['name'],
                            'Bookmaker': book['title'],
                            'Value': val
                        })
        df_odds = pd.DataFrame(odds)
        if not df_odds.empty:
            pivot = df_odds.pivot_table(index=['Game', 'Market', 'Outcome'], columns='Bookmaker', values='Value')
            st.dataframe(pivot.reset_index(), use_container_width=True)

        # === Manual Refresh Button
        if auto_mode == "Manual":
            if st.button(f"🔄 Refresh {label}"):
                st.rerun()


tab_nba, tab_mlb = st.tabs(["🏀 NBA", "⚾ MLB"])
render_scanner_tab("NBA", SPORTS["NBA"], tab_nba, drive)
render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb, drive)
