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
        except Exception as e:
            st.error(f"❌ Failed to load snapshot: {e}")
    return {}  # return empty dict safely


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
            'Home_Team': home,
            'Away_Team': away,
            'Home_Score': team_scores[home],
            'Away_Score': team_scores[away]
        })

     # Step 1: Build results table
    df_results = pd.DataFrame(result_rows)

    # Step 2: If empty, exit early
    if df_results.empty:
        print("⚠️ No completed games found.")
        return df_moves.copy()

    # ✅ Step 3: Rename only now that df_results exists
    df_results = df_results.rename(columns={
        'Home': 'Home_Team',
        'Away': 'Away_Team'
    })

    # Step 4: Deduplicate sharp picks and merge
    df_moves = df_moves.drop_duplicates(subset=['Game', 'Market', 'Outcome'])
    df = df_moves.merge(df_results, on='Game', how='left')


def calc_cover(row):
    team = row['Outcome'].strip().lower()
    home = row['Home_Team'].strip().lower()
    away = row['Away_Team'].strip().lower()
    hscore = row['Home_Score']
    ascore = row['Away_Score']
    market = row['Market'].strip().lower()

    if pd.isna(hscore) or pd.isna(ascore):
        return None, None

    # === Match sharp side to home or away with partial match tolerance
    if team in home:
        team_score, opp_score = hscore, ascore
    elif team in away:
        team_score, opp_score = ascore, hscore
    else:
        if market in ['spreads', 'h2h']:
            print(f"❌ Could not match team: '{team}' with Home: '{home}' or Away: '{away}'")
        return None, None

    margin = team_score - opp_score

    # === h2h result
    if market == 'h2h':
        hit = int(team_score > opp_score)
        return 'Win' if hit else 'Loss', hit

    # === spread result
    if market == 'spreads':
        spread = row.get('Ref Sharp Value')
        if spread is None or not isinstance(spread, (int, float)):
            return None, None
        hit = int((margin > abs(spread)) if spread < 0 else (margin + spread > 0))
        return 'Win' if hit else 'Loss', hit

    # === total result
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


    df[['SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']] = df.apply(lambda r: pd.Series(calc_cover(r)), axis=1)

    print(f"✅ Backtested {df['SHARP_HIT_BOOL'].notna().sum()} sharp edges with game results.")
    return df

    
import pandas as pd
from datetime import datetime
from collections import defaultdict

def implied_prob(price):
    try:
        price = float(price)
        if price > 0:
            return 100 / (price + 100)
        elif price < 0:
            return -price / (-price + 100)
        return None
    except:
        return None

def detect_sharp_moves(current, previous, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS):
    def normalize_label(label):
        return str(label).strip().lower().replace('.0', '')

    rows, sharp_audit_rows, rec_lines = [], [], []
    sharp_limit_map = defaultdict(lambda: defaultdict(list))
    sharp_lines, sharp_side_flags, sharp_metrics_map = {}, {}, {}
    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}

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
                        'Old Value': None, 'Delta': None, 'Event_Date': event_date,
                        'Region': BOOKMAKER_REGIONS.get(book_key, 'unknown'),
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

    for (game_name, mtype), label_map in sharp_limit_map.items():
        scores = {}
        label_signals = {}
        for label, entries in label_map.items():
            move_signal = limit_jump = prob_shift = time_score = 0
            for limit, curr, old in entries:
                if old is not None and curr is not None:
                    if mtype == 'totals':
                        if 'under' in label and curr < old: move_signal += 1
                        elif 'over' in label and curr > old: move_signal += 1
                    elif mtype == 'spreads' and abs(curr) > abs(old): move_signal += 1
                    elif mtype == 'h2h':
                        imp_now, imp_old = implied_prob(curr), implied_prob(old)
                        if imp_now and imp_old and imp_now > imp_old: prob_shift += 1
                if limit and limit >= 5000: limit_jump += 1
                hour = datetime.now().hour
                time_score += 1.0 if 6 <= hour <= 11 else 0.5 if hour <= 15 else 0.2

            score = 2 * move_signal + 2 * limit_jump + 1.5 * time_score + 1.0 * prob_shift
            scores[label] = score
            label_signals[label] = {
                'Sharp_Move_Signal': move_signal,
                'Sharp_Limit_Jump': limit_jump,
                'Sharp_Time_Score': time_score,
                'Sharp_Prob_Shift': prob_shift
            }

        if scores:
            best_label = max(scores, key=scores.get)
            sharp_side_flags[(game_name, mtype, best_label)] = 1
            sharp_metrics_map[(game_name, mtype, best_label)] = label_signals[best_label]

    for rec in rec_lines:
        rec_label = normalize_label(rec['Outcome'])
        market_type = rec['Market']
        rec_key = (rec['Game'], market_type, rec_label)

        if not sharp_side_flags.get(rec_key, 0):
            continue

        sharp = sharp_lines.get(rec_key)
        if not sharp:
            continue

        metrics = sharp_metrics_map.get(rec_key, {})
        row = rec.copy()
        row.update({
            'Ref Sharp Value': sharp['Value'],
            'Delta vs Sharp': rec['Value'] - sharp['Value'] if sharp and rec['Value'] is not None and sharp['Value'] is not None else None,
            'SHARP_SIDE_TO_BET': 1,
            'SharpBetScore': round(
                2.0 * metrics.get('Sharp_Move_Signal', 0) +
                2.0 * metrics.get('Sharp_Limit_Jump', 0) +
                1.5 * metrics.get('Sharp_Time_Score', 0) +
                1.0 * metrics.get('Sharp_Prob_Shift', 0), 2
            ),
            'Sharp_Move_Signal': metrics.get('Sharp_Move_Signal', 0),
            'Sharp_Limit_Jump': metrics.get('Sharp_Limit_Jump', 0),
            'Sharp_Time_Score': metrics.get('Sharp_Time_Score', 0),
            'Sharp_Prob_Shift': metrics.get('Sharp_Prob_Shift', 0)
        })
        rows.append(row)
         
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

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

    print(f"✅ Final sharp-backed rows: {len(df)}")
    return df, pd.DataFrame()





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
        df_bt = pd.DataFrame()  # ✅ declare safely at top

        if not prev:
            st.info("🟡 First run detected — saving snapshot and skipping detection.")
            save_snapshot(sport_key, get_snapshot(live))
            return pd.DataFrame()

        if not live or not isinstance(live, list) or len(live) == 0:
            st.warning(f"⚠️ No live odds returned for {label}.")
            return pd.DataFrame()

        try:
            df_moves, df_audit = detect_sharp_moves(live, prev, label, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS)
        except Exception as e:
            st.error(f"❌ Error in detect_sharp_moves: {e}")
            return pd.DataFrame()

        save_snapshot(sport_key, get_snapshot(live))

        # ✅ Run backtest only if we got sharp picks
        if not df_moves.empty:
            df_bt = fetch_scores_and_backtest(df_moves, sport_key=sport_key)

        # ✅ Safe condition check to display results
        if not df_bt.empty and 'SHARP_HIT_BOOL' in df_bt.columns:
            st.subheader(f"📊 Backtest Results – {label}")
            st.dataframe(
                df_bt[['Game', 'Market', 'Outcome', 'SharpBetScore', 'Ref Sharp Value',
                       'SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']]
            )


        # === Show sharp moves first
        if df_moves is None or df_moves.empty:
            st.info(f"⚠️ No sharp moves detected for {label}.")
        else:
            df_moves['SharpBetScore'] = df_moves.get('SharpBetScore', 0)
            df_moves['SharpAlignment'] = df_moves.get('SharpAlignment', "Unknown")
            df_moves['SHARP_REASON'] = df_moves.get('SHARP_REASON', "Reason not available")
            df_moves['Region'] = df_moves.get('Region', "unknown")

            st.subheader(f"🚨 Detected Sharp Moves – {label}")
            try:
                df_display = df_moves.sort_values(by='SharpBetScore', ascending=False)
            except Exception as e:
                st.error(f"❌ Failed to sort by SharpBetScore: {e}")
                df_display = df_moves

            df_display = df_display.drop_duplicates(subset=['Game', 'Market', 'Outcome'], keep='first')

            # === Filters
            region_options = ["All"] + sorted(df_display['Region'].dropna().unique())
            region = st.selectbox(f"🌍 Filter {label} by Region", region_options, key=f"{label}_region_main")
            if region != "All":
                df_display = df_display[df_display['Region'] == region]

            market_options = ["All"] + sorted(df_display['Market'].dropna().unique())
            market = st.selectbox(f"📊 Filter {label} by Market", market_options, key=f"{label}_market_main")
            if market != "All":
                df_display = df_display[df_display['Market'] == market]

            alignment_filter = st.selectbox(
                f"🧭 Sharp Alignment Filter ({label})",
                ["All", "Sharp move, Rec books not reponded", "Aligned with Sharps", "❓ Unknown or Incomplete"],
                key=f"{label}_alignment_main"
            )
            if alignment_filter != "All":
                df_display = df_display[df_display['SharpAlignment'] == alignment_filter]

            # === Display columns
            display_cols = [
                'Event_Date', 'Game', 'Market', 'Outcome', 'Bookmaker',
                'Value', 'Ref Sharp Value', 'LineMove',
                'Delta vs Sharp', 'Limit', 'SharpAlignment', 'SHARP_REASON', 'SharpBetScore'
            ]
            safe_cols = [col for col in display_cols if col in df_display.columns]

            if not df_display.empty:
                st.dataframe(df_display[safe_cols], use_container_width=True)
            else:
                st.warning(f"⚠️ No sharp edges match your filters for {label}.")

            log_rec_snapshot(df_moves, sport_key)

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
                            "Game": game_name,
                            "Market": mtype,
                            "Outcome": outcome["name"],
                            "Bookmaker": book_title,
                            "Value": price,
                            "Limit": outcome.get("bet_limit", 0)
                        })

        df_odds_raw = pd.DataFrame(raw_odds_table)

        if not df_odds_raw.empty:
            st.subheader(f"📋 Live Odds Snapshot – {label}")

            # Pivot values and limits
            df_pivot_value = df_odds_raw.pivot_table(
                index=["Game", "Market", "Outcome"],
                columns="Bookmaker",
                values="Value",
                aggfunc="first"
            )

            df_pivot_limit = df_odds_raw.pivot_table(
                index=["Game", "Market", "Outcome"],
                columns="Bookmaker",
                values="Limit",
                aggfunc="first"
            ).add_suffix(" (Limit)")

            # Combine both
            df_combined = pd.concat([df_pivot_value, df_pivot_limit], axis=1).reset_index()

            # Highlight best value per row (min for spreads/h2h, max for totals)
            def highlight_best(row):
                all_cols = df_combined.columns.tolist()
                price_cols = df_pivot_value.columns.tolist()

                try:
                    values = row[price_cols].astype(float)
                    if row["Market"] in ["spreads", "h2h"]:
                        best_col = values.idxmin()
                    else:  # totals
                        best_col = values.idxmax()

                    return [
                        'background-color: #d0f0c0' if col == best_col else ''
                        if col in price_cols else '' for col in all_cols
                    ]
                except:
                    return [''] * len(all_cols)

                
            st.dataframe(
                df_combined.style.apply(highlight_best, axis=1),
                use_container_width=True
            )
        else:
            st.warning("⚠️ No odds to display for this sport.")
        # === Automatically backtest sharp picks (no button)
        df_bt = fetch_scores_and_backtest(df_moves, sport_key=sport_key)

        if not df_bt.empty and 'SHARP_HIT_BOOL' in df_bt.columns:
            st.subheader(f"📊 Backtest Results – {label}")
            st.dataframe(
                df_bt[['Game', 'Market', 'Outcome', 'SharpBetScore', 'Ref Sharp Value',
                       'SHARP_COVER_RESULT', 'SHARP_HIT_BOOL']]
            )
        else:
            st.warning("⚠️ No results available for backtest yet.")
           

        
        return df_moves




tab_nba, tab_mlb = st.tabs(["🏀 NBA", "⚾ MLB"])

# Get sharp edges
df_nba = render_scanner_tab("NBA", SPORTS["NBA"], tab_nba, drive)
df_mlb = render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb, drive)


# Backtest and show performance
if df_nba is not None and not df_nba.empty:
    df_nba_bt = fetch_scores_and_backtest(df_nba, sport_key='basketball_nba')

    # 🔧 Create SharpConfidenceTier from SharpBetScore
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

if df_mlb is not None and not df_mlb.empty:
    df_mlb_bt = fetch_scores_and_backtest(df_mlb, sport_key='baseball_mlb')

    # 🔧 Create SharpConfidenceTier from SharpBetScore
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

# 🧠 Sharp Signal Learning (Component Breakdown)
if df_nba_bt is not None and not df_nba_bt.empty:
    st.subheader("🧠 Sharp Component Learning – NBA")
    st.dataframe(df_nba_bt.groupby('Sharp_Move_Signal')['SHARP_HIT_BOOL'].mean().reset_index().rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Move_Signal'}))
    st.dataframe(df_nba_bt.groupby('Sharp_Time_Score')['SHARP_HIT_BOOL'].mean().reset_index().rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Time_Score'}))

if df_mlb_bt is not None and not df_mlb_bt.empty:
    st.subheader("🧠 Sharp Component Learning – MLB")
    st.dataframe(df_mlb_bt.groupby('Sharp_Move_Signal')['SHARP_HIT_BOOL'].mean().reset_index().rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Move_Signal'}))
    st.dataframe(df_mlb_bt.groupby('Sharp_Time_Score')['SHARP_HIT_BOOL'].mean().reset_index().rename(columns={'SHARP_HIT_BOOL': 'Win_Rate_By_Time_Score'}))
