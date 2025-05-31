import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

# === CONFIG ===
API_KEY = '4f95ea43cc1c29cd44c40fe59b6c14ce'
SPORTS = {"NBA": "basketball_nba", "MLB": "baseball_mlb"}
SHARP_BOOKS = ['pinnacle', 'bookmaker', 'betonlineag']
REC_BOOKS = [
    'bovada', 'heritagesports', 'betus', 'betmgm', 'bet365',
    'draftkings', 'fanduel', 'betrivers', 'pointsbetus'
]
BOOKMAKER_REGIONS = {
    'pinnacle': 'us',
    'bookmaker': 'us',
    'betonlineag': 'us',
    'bovada': 'us',
    'heritagesports': 'us',
    'betus': 'us',
    'betmgm': 'us',
    'draftkings': 'us',
    'fanduel': 'us',
    'betrivers': 'us',
    'pointsbetus': 'us2',
    'bet365': 'uk',
    'williamhill': 'uk',
    'ladbrokes': 'uk',
    'unibet': 'eu',
    'bwin': 'eu',
    'sportsbet': 'au',
    'ladbrokesau': 'au',
    'neds': 'au'
}
MARKETS = ['spreads', 'totals', 'h2h']

st.set_page_config(layout="wide")
st.title("📊 Sharp Edge Scanner + Regional Tagging")

auto_mode = st.sidebar.radio("🕹️ Refresh Mode", ["Auto Refresh", "Manual"], index=0)
if auto_mode == "Auto Refresh":
    st_autorefresh(interval=150000, key="autorefresh")

st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

def detect_sharp_moves(current, previous, sport_key, spread_thresh, ml_thresh):
    moves = []
    for game in current:
        gid = game['id']
        game_name = f"{game['home_team']} vs {game['away_team']}"
        prev_game = previous.get(gid, {}) if previous else {}

        for book in game.get('bookmakers', []):
            book_key = book['key']
            if book_key not in SHARP_BOOKS + REC_BOOKS + list(BOOKMAKER_REGIONS.keys()):
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

    def explain(row):
        r = []
        if row['Limit_Jump']: r.append("Limit ≥ 10K")
        if row['Delta_Abs'] >= 1: r.append("Big Move")
        if row['Sharp_Timing'] >= 1: r.append("Morning Shaping")
        if row['Asymmetric_Flag']: r.append("Asymmetric Limit")
        return ", ".join(r)
    df['Sharp Reason'] = df.apply(explain, axis=1)

    return df

# === SIDEBAR FILTERS ===
st.sidebar.header("📏 Detection Thresholds")
spread_thresh = st.sidebar.slider("Line Move Threshold (spread/total)", 0.1, 2.0, 0.5, 0.1)
ml_thresh = st.sidebar.slider("Price Move Threshold (moneyline)", 1, 50, 10, 1)

# === MAIN LOOP ===
for label, sport_key in SPORTS.items():
    st.header(f"{label} Sharp Scanner")
    live = fetch_live_odds(sport_key)
    prev = st.session_state.previous_snapshots.get(sport_key, {})
    snapshot = get_snapshot(live)

    df_moves = detect_sharp_moves(live, prev, label, spread_thresh, ml_thresh)
    if not df_moves.empty:
        df_scored = score_sharp_moves(df_moves)
        df_display = df_scored.sort_values(by='MillerSharpScore', ascending=False)

        # 🌍 Region Filter
        selected_region = st.selectbox(f"🌍 Filter {label} by Region", options=["All"] + sorted(df_display['Region'].unique()))
        if selected_region != "All":
            df_display = df_display[df_display['Region'] == selected_region]

        st.subheader("🚨 Detected Sharp Moves")
        st.dataframe(df_display[[
            'Time', 'Region', 'Game', 'Market', 'Bookmaker', 'Outcome',
            'Old Value', 'New Value', 'Delta',
            'Limit', 'MillerSharpScore', 'Sharp Reason'
        ]], use_container_width=True)

        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(f"📥 Download {label} Sharp Moves", csv, f"{label.lower()}_sharp_moves.csv", "text/csv")

        folder = r"C:\Users\sfeldman\OneDrive\Betting files\NBA\Sharp Money tracker\SharpScannerLogs"
        os.makedirs(folder, exist_ok=True)
        fname = f"{label}_sharp_moves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_display.to_csv(os.path.join(folder, fname), index=False)

    else:
        st.info("No sharp edges detected.")

    if auto_mode == "Manual":
        if st.button(f"🔄 Refresh {label}"):
            st.session_state.previous_snapshots[sport_key] = snapshot
            st.rerun()

# === Init Baseline Snapshots ===
for sport_key in SPORTS.values():
    if sport_key not in st.session_state.previous_snapshots:
        st.session_state.previous_snapshots[sport_key] = get_snapshot(fetch_live_odds(sport_key))

# === Live Odds Table ===
st.subheader(f"📋 {label} Current Odds")
odds_rows = []
for game in live:
    game_name = f"{game['home_team']} vs {game['away_team']}"
    for book in game.get('bookmakers', []):
        for market in book.get('markets', []):
            for o in market.get('outcomes', []):
                val = o.get('point') if market['key'] != 'h2h' else o.get('price')
                odds_rows.append({
                    'Game': game_name,
                    'Market': market['key'],
                    'Outcome': o['name'],
                    'Bookmaker': book['title'],
                    'Value': val
                })

df_odds = pd.DataFrame(odds_rows)
if not df_odds.empty:
    pivot = df_odds.pivot_table(index=['Game', 'Market', 'Outcome'], columns='Bookmaker', values='Value')
    st.dataframe(pivot.reset_index(), use_container_width=True)




