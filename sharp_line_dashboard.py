import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import altair as alt
import glob

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
LOG_FOLDER = r"C:\Users\sfeldman\OneDrive\Betting files\NBA\Sharp Money tracker\SharpScannerLogs"

# === PAGE ===
st.set_page_config(layout="wide")
st.title("📊 Sharp Edge Scanner with Region Tagging & Movement Graphs")

# === MODE TOGGLE ===
auto_mode = st.sidebar.radio("🕹️ Refresh Mode", ["Auto Refresh", "Manual"], index=0)
if auto_mode == "Auto Refresh":
    st_autorefresh(interval=150000, key="autorefresh")
st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === THRESHOLD CONTROLS ===
st.sidebar.header("📏 Sharp Move Detection Thresholds")
spread_thresh = st.sidebar.slider("Line Move (spread/total)", 0.1, 2.0, 0.5, 0.1)
ml_thresh = st.sidebar.slider("Price Move (moneyline)", 1, 50, 10, 1)

if 'previous_snapshots' not in st.session_state:
    st.session_state.previous_snapshots = {}

# === DATA HELPERS ===
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
    moves = []
    for game in current:
        gid = game['id']
        game_name = f"{game['home_team']} vs {game['away_team']}"
        prev_game = previous.get(gid, {}) if previous else {}

        for book in game.get('bookmakers', []):
            book_key = book['key']
            if book_key not in SHARP_BOOKS + REC_BOOKS:
                continue
            region = BOOKMAKER_REGIONS.get(book_key, 'unknown')
            for market in book.get('markets', []):
                mtype = market['key']
                for o in market.get('outcomes', []):
                    val = o.get('point') if mtype != 'h2h' else o.get('price')
                    limit = o.get('bet_limit')
                    label = o.get('name')
                    prev_val = None
                    if prev_game:
                        for b in prev_game.get('bookmakers', []):
                            if b['key'] == book_key:
                                for m in b.get('markets', []):
                                    if m['key'] == mtype:
                                        for po in m.get('outcomes', []):
                                            if po['name'] == label:
                                                prev_val = po.get('point') if mtype != 'h2h' else po.get('price')
                    if prev_val is not None and val is not None:
                        delta = round(val - prev_val, 2)
                        is_valid_move = (
                            (mtype == 'h2h' and abs(delta) >= ml_thresh) or
                            (mtype in ['spreads', 'totals'] and abs(delta) >= spread_thresh)
                        )
                        if is_valid_move:
                            moves.append({
                                'Sport': sport_key,
                                'Game': game_name,
                                'Market': mtype,
                                'Outcome': label,
                                'Bookmaker': book['title'],
                                'Book': book_key,
                                'Region': region,
                                'Old Value': prev_val,
                                'New Value': val,
                                'Delta': delta,
                                'Limit': limit,
                                'Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
    return pd.DataFrame(moves)

def score_sharp_moves(df):
    df['Delta_Abs'] = df['Delta'].abs()
    df['Limit_Jump'] = (df['Limit'].fillna(0) >= 10000).astype(int)
    df['Sharp_Timing'] = pd.to_datetime(df['Time']).dt.hour.apply(lambda h: 1 if 8 <= h <= 11 else 0.5 if h <= 6 else 0)
    df['Asymmetric_Limit'] = df.groupby(['Game', 'Market'])['Limit'].transform(lambda x: x.max() - x.min())
    df['Asymmetric_Flag'] = (df['Asymmetric_Limit'] >= 5000).astype(int)
    df['MillerSharpScore'] = (
        5 * df['Limit_Jump'] +
        3 * df['Delta_Abs'] +
        2 * df['Sharp_Timing'] +
        2 * df['Asymmetric_Flag']
    )
    df['Sharp Reason'] = df.apply(lambda r: ", ".join([
        "Limit ≥ 10K" if r['Limit_Jump'] else "",
        "Big Move" if r['Delta_Abs'] >= 1 else "",
        "Morning Shaping" if r['Sharp_Timing'] >= 1 else "",
        "Asymmetric Limit" if r['Asymmetric_Flag'] else ""
    ]).strip(', '), axis=1)
    return df

# === TABS ===
tab_nba, tab_mlb, tab_graphs = st.tabs(["🏀 NBA Scanner", "⚾ MLB Scanner", "📈 Sharp Movement Graphs"])

# === SCANNER TAB DISPLAY FUNCTION ===
def render_scanner_tab(label, sport_key, container):
    with container:
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
            os.makedirs(LOG_FOLDER, exist_ok=True)
            fname = f"{label}_sharp_moves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_display.to_csv(os.path.join(LOG_FOLDER, fname), index=False)
            st.download_button("📥 Download CSV", df_display.to_csv(index=False).encode('utf-8'), f"{label}_sharp.csv", "text/csv")
        else:
            st.info("No sharp moves this cycle.")

        # Live odds
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

# === RENDER TABS ===
render_scanner_tab("NBA", SPORTS["NBA"], tab_nba)
render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb)

# === GRAPH TAB ===
with tab_graphs:
    st.header("📈 Sharp Line Movement Over Time")
    all_files = glob.glob(os.path.join(LOG_FOLDER, "*.csv"))
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
