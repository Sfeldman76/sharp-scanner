import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import altair as alt
import glob
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import json

# === CONFIG ===
API_KEY = '4f95ea43cc1c29cd44c40fe59b6c14ce'
SPORTS = {"NBA": "basketball_nba", "MLB": "baseball_mlb"}
SHARP_BOOKS = ['pinnacle', 'bookmaker', 'betonlineag']
REC_BOOKS = [
    'bovada', 'heritagesports', 'betus', 'betmgm', 'bet365',
    'draftkings', 'fanduel', 'betrivers', 'pointsbetus'
]
BOOKMAKER_REGIONS = {
    'pinnacle': 'us', 'bookmaker': 'us', 'betonlineag': 'us',
    'bovada': 'us', 'heritagesports': 'us', 'betus': 'us',
    'betmgm': 'us', 'draftkings': 'us', 'fanduel': 'us',
    'betrivers': 'us', 'pointsbetus': 'us2',
    'bet365': 'uk', 'williamhill': 'uk', 'ladbrokes': 'uk',
    'unibet': 'eu', 'bwin': 'eu', 'sportsbet': 'au',
    'ladbrokesau': 'au', 'neds': 'au'
}
MARKETS = ['spreads', 'totals', 'h2h']
LOG_FOLDER = "/tmp/sharp_logs"


def init_gdrive():
    import json
    from pydrive2.auth import GoogleAuth
    from pydrive2.drive import GoogleDrive

    creds_path = "/tmp/service_creds.json"
    creds_dict = dict(st.secrets["gdrive"])

    # Debug: show what's going into the file
    st.write("🔒 GDrive Secret Keys:", list(creds_dict.keys()))

    with open(creds_path, "w") as f:
        json.dump(creds_dict, f)

    # Show the file contents back just to confirm
    with open(creds_path) as f:
        st.code(f.read(), language="json")

    gauth = GoogleAuth()
    gauth.LoadServiceConfigFile(creds_path)
    gauth.ServiceAuth()

    return GoogleDrive(gauth)


# === PAGE ===
st.set_page_config(layout="wide")
st.title("📊 Sharp Edge Scanner with Region Tagging & Movement Graphs")

auto_mode = st.sidebar.radio("🕹️ Refresh Mode", ["Auto Refresh", "Manual"], index=0)
if auto_mode == "Auto Refresh":
    st_autorefresh(interval=150000, key="autorefresh")
st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.header("📏 Sharp Move Detection Thresholds")
spread_thresh = st.sidebar.slider("Line Move (spread/total)", 0.1, 2.0, 0.5, 0.1)
ml_thresh = st.sidebar.slider("Price Move (moneyline)", 1, 50, 10, 1)

if 'previous_snapshots' not in st.session_state:
    st.session_state.previous_snapshots = {}

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
        st.error(f"Error fetching odds: {e}")
        return []

def get_snapshot(data):
    return {g['id']: g for g in data}


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

                    # Optional delta from previous snapshot
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

        # Compare rec books to sharp lines
        for rec_entry in rec_lines:
            key = (rec_entry['Market'], rec_entry['Outcome'])
            sharp_entry = sharp_lines.get(key)

            if sharp_entry and sharp_entry['Value'] is not None and rec_entry['Value'] is not None:
                sharp_val = sharp_entry['Value']
                rec_val = rec_entry['Value']
                delta_vs_sharp = round(rec_val - sharp_val, 2)

                # === Compute components ===
                # Bias match by meaningful advantage
                bias_match = 0
                if rec_entry['Market'] == 'spreads' and abs(delta_vs_sharp) >= 0.5:
                    bias_match = 1
                elif rec_entry['Market'] == 'totals' and abs(delta_vs_sharp) >= 1.0:
                    bias_match = 1
                elif rec_entry['Market'] == 'h2h' and abs(delta_vs_sharp) >= 10:
                    bias_match = 1

                # Line moved from previous
                delta = rec_entry.get("Delta")
                if isinstance(delta, (int, float)):
                    line_moved = 1 if abs(delta) >= (1 if rec_entry['Market'] != 'h2h' else 10) else 0
                else:
                    line_moved = 0

                # Limit asymmetry
                sharp_limit = sharp_entry.get('Limit') or 0
                rec_limit = rec_entry.get('Limit') or 0
                asymmetry = 1 if abs(sharp_limit - rec_limit) >= 5000 else 0

                # Sharp confidence
                limit_score = 1 if sharp_limit >= 10000 else 0.5 if sharp_limit > 0 else 0

                # Final score
                miller_score = (
                    5 * bias_match +
                    4 * limit_score +
                    3 * line_moved +
                    2 * asymmetry
                )

                # === Output debug trace ===
                st.write(f"🧠 Scoring: {rec_entry['Game']} | {rec_entry['Outcome']}")
                st.write(f"  - Delta vs Sharp: {delta_vs_sharp}")
                st.write(f"  - Bias Match: {bias_match}")
                st.write(f"  - Line Moved: {line_moved}")
                st.write(f"  - Limit Asymmetry: {asymmetry} (Sharp: {sharp_limit}, Rec: {rec_limit})")
                st.write(f"  - Limit Score: {limit_score}")
                st.write(f"  => Final MillerSharpScore: {miller_score}")

                combined = rec_entry.copy()
                combined.update({
                    'Ref Sharp Value': sharp_val,
                    'Sharp Limit': sharp_limit,
                    'Delta vs Sharp': delta_vs_sharp,
                    'Bias Match': bias_match,
                    'Line Moved': line_moved,
                    'Asymmetry Flag': asymmetry,
                    'Limit Score': limit_score,
                    'MillerSharpScore': miller_score
                })

                opportunities.append(combined)

    return pd.DataFrame(opportunities)

# === TAB SETUP ===
tab_nba, tab_mlb, tab_graphs = st.tabs(["🏀 NBA Scanner", "⚾ MLB Scanner", "📈 Sharp Movement Graphs"])

# === SCANNER TAB DISPLAY FUNCTION ===
def render_scanner_tab(label, sport_key, container):
    with container:
        os.makedirs(LOG_FOLDER, exist_ok=True)

        live = fetch_live_odds(sport_key)
        prev = st.session_state.previous_snapshots.get(sport_key, {})
        snapshot = get_snapshot(live)

        df_moves = detect_sharp_moves(live, prev, label)
        if not df_moves.empty:
            df_scored = score_sharp_moves(df_moves)
            df_display = df_scored.sort_values(by='MillerSharpScore', ascending=False)

            region = st.selectbox(f"🌍 Filter {label} by Region", ["All"] + sorted(df_display['Region'].unique()))
            if region != "All":
                df_display = df_display[df_display['Region'] == region]

            st.subheader("🚨 Detected Sharp Moves")
            st.dataframe(df_display, use_container_width=True)

            fname = f"{label}_sharp_moves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = os.path.join(LOG_FOLDER, fname)
            df_display.to_csv(csv_path, index=False)

            drive = init_gdrive()
            gfile = drive.CreateFile({'title': fname})
            gfile.SetContentFile(csv_path)
            gfile.Upload()
            st.success(f"✅ Uploaded to Google Drive: {fname}")

            st.download_button("📥 Download CSV", df_display.to_csv(index=False).encode('utf-8'), fname, "text/csv")

        else:
            st.info("No sharp moves this cycle.")

        # LIVE ODDS
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

# === RENDER NBA / MLB TABS ===
render_scanner_tab("NBA", SPORTS["NBA"], tab_nba)
render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb)

# === GRAPH TAB ===
with tab_graphs:
    st.header("📈 Sharp Line Movement Over Time")

    os.makedirs(LOG_FOLDER, exist_ok=True)
    all_files = glob.glob(os.path.join(LOG_FOLDER, "*.csv"))

    # If no logs yet, create starter CSV
    if not all_files:
        df_start = pd.DataFrame([{
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Game': 'Demo Team A vs Demo Team B',
            'Market': 'spreads',
            'Outcome': 'Demo',
            'Bookmaker': 'Pinnacle',
            'Book': 'pinnacle',
            'Region': 'us',
            'Old Value': -3.5,
            'New Value': -4.0,
            'Delta': 0.5,
            'Limit': 10000,
            'MillerSharpScore': 12,
            'Sharp Reason': 'Limit ≥ 10K, Big Move'
        }])
        fname = "starter_demo_sharp_log.csv"
        csv_path = os.path.join(LOG_FOLDER, fname)
        df_start.to_csv(csv_path, index=False)

        drive = init_gdrive()
        gfile = drive.CreateFile({'title': fname})
        gfile.SetContentFile(csv_path)
        gfile.Upload()
        all_files = [csv_path]
        st.success("✅ Created and uploaded starter demo log.")

    df_all = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    df_all['Time'] = pd.to_datetime(df_all['Time'], errors='coerce')
    df_all = df_all.dropna(subset=['Game', 'Market', 'Outcome', 'Time', 'New Value'])

    game = st.selectbox("Game", sorted(df_all['Game'].unique()))
    df_game = df_all[df_all['Game'] == game]

    if not df_game.empty:
        market = st.selectbox("Market", sorted(df_game['Market'].unique()))
        df_market = df_game[df_game['Market'] == market]

        outcome = st.selectbox("Outcome", sorted(df_market['Outcome'].unique()))
        df_plot = df_market[df_market['Outcome'] == outcome]

        chart = alt.Chart(df_plot).mark_line(point=True).encode(
            x='Time:T',
            y=alt.Y('New Value:Q', title='Line / Price'),
            color='Bookmaker:N',
            tooltip=['Bookmaker', 'Region', 'New Value', 'Limit', 'Sharp Reason']
        ).properties(
            width=1000,
            height=400,
            title=f"{game} — {market.upper()} — {outcome}"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No historical data found for selected game.")
