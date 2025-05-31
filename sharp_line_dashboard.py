import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import json

# === CONFIG ===
API_KEY = '4f95ea43cc1c29cd44c40fe59b6c14ce'
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

def init_gdrive():
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive
    import json

    try:
        creds_path = "/tmp/service_creds.json"

        # Write Streamlit secrets to file
        with open(creds_path, "w") as f:
            json.dump(dict(st.secrets["gdrive"]), f)

        # Set up authentication
        gauth = GoogleAuth()
        gauth.settings["client_config_backend"] = "service"
        gauth.settings["service_config"] = {
            "client_json_file_path": creds_path
        }

        gauth.ServiceAuth()
        drive = GoogleDrive(gauth)
        return drive
    except Exception as e:
        st.error(f"❌ Google Drive auth failed: {e}")
        return None


# === LIVE ODDS FETCH ===
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
        data = r.json()
        return data
    except Exception as e:
        st.error(f"❌ Odds API Error: {e}")
        return []

def get_snapshot(data):
    return {g['id']: g for g in data}

# === SHARP MOVE SCORING ===
def score_sharp_moves(df):
    df['Delta'] = pd.to_numeric(df['Delta'], errors='coerce')
    df['Delta_Abs'] = df['Delta'].abs()
    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)
    df['Limit_Jump'] = (df['Limit'] >= 10000).astype(int)
    df['Sharp_Timing'] = pd.to_datetime(df['Time']).dt.hour.apply(lambda h: 1.0 if 6 <= h <= 11 else 0.5 if h <= 15 else 0.2)
    df['Asymmetric_Limit'] = df.groupby(['Game', 'Market'])['Limit'].transform(lambda x: x.max() - x.min())
    df['Asymmetry_Flag'] = (df['Asymmetric_Limit'] >= 5000).astype(int)
    df['MillerSharpScore'] = (
        5 * df['Bias Match'] +
        3 * df['Limit_Jump'] +
        2 * df['Delta_Abs'] +
        2 * df['Sharp_Timing'] +
        2 * df['Asymmetry_Flag']
    ).round(2)
    return df

# === SHARP DETECTION ===
def detect_sharp_moves(current, previous, sport_key):
    opportunities = []
    snapshot_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    previous_map = {g['id']: g for g in previous} if previous else {}

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
                    limit = o.get('bet_limit')
                    label = o['name']

                    entry = {
                        'Sport': sport_key,
                        'Time': snapshot_time,
                        'Game': game_name,
                        'Market': mtype,
                        'Outcome': label,
                        'Bookmaker': book['title'],
                        'Book': book_key,
                        'Region': region,
                        'Value': val,
                        'Limit': limit,
                        'Old Value': None,
                        'Delta': None
                    }

                    if prev_game:
                        for prev_b in prev_game.get('bookmakers', []):
                            if prev_b['key'] == book_key:
                                for prev_m in prev_b.get('markets', []):
                                    if prev_m['key'] == mtype:
                                        for prev_o in prev_m.get('outcomes', []):
                                            if prev_o['name'] == label:
                                                prev_val = prev_o.get('point') if mtype != 'h2h' else prev_o.get('price')
                                                if prev_val is not None and val is not None:
                                                    entry['Old Value'] = prev_val
                                                    entry['Delta'] = round(val - prev_val, 2)

                    if book_key in SHARP_BOOKS:
                        sharp_lines[(mtype, label)] = entry
                    elif book_key in REC_BOOKS:
                        rec_lines.append(entry)

        for rec_entry in rec_lines:
            key = (rec_entry['Market'], rec_entry['Outcome'])
            sharp_entry = sharp_lines.get(key)
            if sharp_entry and sharp_entry['Value'] is not None and rec_entry['Value'] is not None:
                sharp_val = sharp_entry['Value']
                rec_val = rec_entry['Value']
                delta_vs_sharp = round(rec_val - sharp_val, 2)

                bias_match = 0
                if rec_entry['Market'] == 'spreads':
                    if sharp_val < 0 and rec_val > sharp_val:
                        bias_match = 1
                    elif sharp_val > 0 and rec_val > sharp_val:
                        bias_match = 1
                elif rec_entry['Market'] == 'totals':
                    if "under" in rec_entry['Outcome'].lower() and rec_val > sharp_val:
                        bias_match = 1
                    elif "over" in rec_entry['Outcome'].lower() and rec_val < sharp_val:
                        bias_match = 1
                elif rec_entry['Market'] == 'h2h':
                    if sharp_val < 0 and rec_val > sharp_val:
                        bias_match = 1
                    elif sharp_val > 0 and rec_val > sharp_val:
                        bias_match = 1

                combined = rec_entry.copy()
                combined.update({
                    'Ref Sharp Value': sharp_val,
                    'Delta vs Sharp': delta_vs_sharp,
                    'Bias Match': bias_match
                })
                opportunities.append(combined)

    return pd.DataFrame(opportunities)

# === PAGE SETUP ===
st.set_page_config(layout="wide")
st.title("📊 Sharp Edge Scanner")
auto_mode = st.sidebar.radio("🕹️ Refresh Mode", ["Auto Refresh", "Manual"], index=0)
if auto_mode == "Auto Refresh":
    st_autorefresh(interval=120000, key="autorefresh")

if 'previous_snapshots' not in st.session_state:
    st.session_state.previous_snapshots = {}

st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === SCANNER TAB FUNCTION ===
def render_scanner_tab(label, sport_key, container):
    with container:
        os.makedirs(LOG_FOLDER, exist_ok=True)
        live = fetch_live_odds(sport_key)
        prev = st.session_state.previous_snapshots.get(sport_key, {})
        snapshot = get_snapshot(live)

        if not live:
            st.warning(f"No odds returned for {label}.")
            return

        df_moves = detect_sharp_moves(live, prev, label)
        if not df_moves.empty:
            df_moves = score_sharp_moves(df_moves)
            df_display = df_moves.sort_values(by='MillerSharpScore', ascending=False)
            df_display = df_display.drop_duplicates(subset=['Game', 'Market'], keep='first')

            st.subheader("🚨 Detected Sharp Moves")
            region = st.selectbox(f"🌍 Filter {label} by Region", ["All"] + sorted(df_display['Region'].unique()))
            if region != "All":
                df_display = df_display[df_display['Region'] == region]

            st.dataframe(df_display, use_container_width=True)

            fname = f"{label}_sharp_moves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = os.path.join(LOG_FOLDER, fname)
            df_display.to_csv(csv_path, index=False)

            drive = init_gdrive()
            if drive:
                gfile = drive.CreateFile({'title': fname})
                gfile.SetContentFile(csv_path)
                gfile.Upload()
                st.success(f"✅ Uploaded to Google Drive: {fname}")

            st.download_button("📥 Download CSV", df_display.to_csv(index=False).encode('utf-8'), fname, "text/csv")
        else:
            st.info("No sharp moves detected this cycle.")

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

        if auto_mode == "Manual":
            if st.button(f"🔄 Refresh {label}"):
                st.session_state.previous_snapshots[sport_key] = snapshot
                st.rerun()

# === TABS ===
tab_nba, tab_mlb = st.tabs(["🏀 NBA", "⚾ MLB"])
render_scanner_tab("NBA", SPORTS["NBA"], tab_nba)
render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb)

