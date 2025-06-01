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
                    label = o['name']
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
                                            if prev_o['name'] == label:
                                                prev_val = prev_o.get('point') if mtype != 'h2h' else prev_o.get('price')
                                                if prev_val is not None and val is not None:
                                                    entry['Old Value'] = prev_val
                                                    entry['Delta'] = round(val - prev_val, 2)

                    if book_key in SHARP_BOOKS:
                        sharp_lines[(mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit or 0, val))
                    elif book_key in REC_BOOKS:
                        rec_lines.append(entry)

    # === Determine SHARP_SIDE_TO_BET per market
    sharp_side_flags = {}
    for (game_mkt, label_map) in sharp_limit_map.items():
        if len(label_map) != 2:
            continue  # need both sides to compare

        labels = list(label_map.keys())
        l0, l1 = labels[0], labels[1]
        prices0 = [x[1] for x in label_map[l0] if x[1] is not None]
        prices1 = [x[1] for x in label_map[l1] if x[1] is not None]
        mtype = game_mkt[1]

        if not prices0 or not prices1:
            continue

        avg0 = round(sum(prices0) / len(prices0), 2)
        avg1 = round(sum(prices1) / len(prices1), 2)

        # Special logic for totals: only assign sharp side if lines are different
        if mtype == 'totals' and avg0 == avg1:
            # Still store a SHARP_REASON later, but do NOT assign SHARP_SIDE
            continue

        # General logic (totals, spreads, h2h)
        if mtype == 'totals':
            l0_under = 'under' in l0.lower()
            l1_under = 'under' in l1.lower()

            if l0_under and avg0 < avg1:
                sharp_side_label = l0
            elif l1_under and avg1 < avg0:
                sharp_side_label = l1
            elif not l0_under and avg0 > avg1:
                sharp_side_label = l0
            else:
                sharp_side_label = l1
        else:
            # Use combined price & sharp consensus logic for spreads and h2h
            score0 = sum([x[0] for x in label_map[l0]]) + avg0 * 2
            score1 = sum([x[0] for x in label_map[l1]]) + avg1 * 2
            sharp_side_label = l0 if score0 > score1 else l1

        sharp_side_flags[(game_mkt[0], game_mkt[1], sharp_side_label)] = 1

    # === Build final rows
    for rec in rec_lines:
        key = (rec['Market'], rec['Outcome'])
        sharp = sharp_lines.get(key)

        # fallback to alt outcome
        if not sharp:
            alt_outcome = None
            if rec['Market'] == 'totals':
                if "over" in rec['Outcome'].lower():
                    alt_outcome = "Under"
                elif "under" in rec['Outcome'].lower():
                    alt_outcome = "Over"
            elif rec['Market'] == 'h2h':
                game_outcomes = [k[1] for k in sharp_lines.keys() if k[0] == rec['Market']]
                alt_outcome = next((o for o in game_outcomes if o != rec['Outcome']), None)
            if alt_outcome:
                sharp = sharp_lines.get((rec['Market'], alt_outcome))

        if not sharp or sharp['Outcome'] != rec['Outcome']:
            continue

        delta_vs_sharp = rec['Value'] - sharp['Value']

        implied_rec = implied_prob(rec['Value']) if rec['Market'] == 'h2h' else None
        implied_sharp = implied_prob(sharp['Value']) if sharp['Market'] == 'h2h' else None
        bias_match = 0

        if rec['Market'] == 'spreads':
            if abs(rec['Value']) < abs(sharp['Value']):
                bias_match = 1
        elif rec['Market'] == 'totals':
            if "under" in rec['Outcome'].lower() and rec['Value'] > sharp['Value']:
                bias_match = 1
            elif "over" in rec['Outcome'].lower() and rec['Value'] < sharp['Value']:
                bias_match = 1
        elif rec['Market'] == 'h2h' and implied_rec is not None and implied_sharp is not None:
            if implied_rec < implied_sharp:
                bias_match = 1

        sharp_side_flag = sharp_side_flags.get((rec['Game'], rec['Market'], rec['Outcome']), 0)

        if sharp_side_flag == 1:
            row = rec.copy()
            row.update({
                'Ref Sharp Value': sharp['Value'],
                'Delta vs Sharp': round(delta_vs_sharp, 2),
                'Bias Match': bias_match,
                'Implied_Prob_Rec': implied_rec,
                'Implied_Prob_Sharp': implied_sharp,
                'Implied_Prob_Diff': (implied_sharp - implied_rec) if implied_rec and implied_sharp else None,
                'Limit': sharp.get('Limit') or 0,
                'SHARP_SIDE_TO_BET': sharp_side_flag,
                'Time': snapshot_time
            })

            # ✅ Sharp Alignment Label
            if abs(delta_vs_sharp) < 0.01:
                alignment = "✅ Aligned with sharps"
            else:
                alignment = "🚨 Edge vs sharps"

            row['SharpAlignment'] = alignment

            # === SHARP REASON GENERATOR
            side_label = rec['Outcome']
            game_key = (rec['Game'], rec['Market'])
            sharp_side_metrics = sharp_limit_map.get(game_key, {})
            side_limits = [x[0] for x in sharp_side_metrics.get(side_label, []) if x[0]]
            side_prices = [x[1] for x in sharp_side_metrics.get(side_label, []) if x[1]]
            side_count = len(side_limits)
            limit_tag = f"{max(side_limits):,.0f}" if side_limits else "unknown"
            implied = implied_prob(rec['Value'])
            price_tag = f"{round(implied * 100, 1)}%" if implied else "N/A"

            anchor_tag = ""
            fade_tag = ""
            matchup_tag = ""

            labels = list(sharp_side_metrics.keys())
            opp_label = next((l for l in labels if l != side_label), None)
            opp_vals = [x[1] for x in sharp_side_metrics.get(opp_label, []) if x[1]] if opp_label else []

            if len(side_prices) > 1 and max(side_prices) - min(side_prices) < 0.01:
                anchor_tag = "💰 Anchor price at all sharp books"
            if not opp_vals:
                fade_tag = "📉 Sharp fade: opponent has no limit"

            if rec['Market'] == 'totals' and opp_label:
                if side_prices and opp_vals and round(sum(side_prices)/len(side_prices), 2) == round(sum(opp_vals)/len(opp_vals), 2):
                    matchup_tag = "⛔ Equal market line – no edge"
            elif rec['Market'] == 'spreads' and opp_label:
                matchup_tag = f"🛡️ Sharps protecting {side_label} vs {opp_label}"
            elif rec['Market'] == 'h2h' and opp_label:
                side_avg = round(sum(side_prices) / len(side_prices))
                opp_avg = round(sum(opp_vals) / len(opp_vals)) if opp_vals else None
                if opp_avg:
                    matchup_tag = f"🏆 Sharps favoring {side_label} ({side_avg}) vs {opp_label} ({opp_avg})"

            reason_parts = [
                f"📈 Sharp limit ${limit_tag}",
                f"🧮 implied prob {price_tag}",
                f"🧠 {side_count} sharp book(s)",
                anchor_tag,
                fade_tag,
                matchup_tag
            ]
            row['SHARP_REASON'] = ", ".join([r for r in reason_parts if r])

            rows.append(row)  # ✅ Make sure this happens inside the if-block


    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df['LineMove'] = df.apply(
        lambda r: round(r['Value'] - r['Old Value'], 2) if pd.notnull(r['Old Value']) else None,
        axis=1
    )

    df['Delta'] = pd.to_numeric(df['Delta vs Sharp'], errors='coerce')
    df = df[df['Delta'].abs() > 0.01]
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

    # Final filter: only show smart sharp-backed moves
    df = df[df['SHARP_SIDE_TO_BET'] == 1]

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
            df_display = df_display.drop_duplicates(subset=['Game', 'Market'], keep='first')

            st.subheader("🚨 Detected Sharp Moves")

            region = st.selectbox(f"🌍 Filter {label} by Region", ["All"] + sorted(df_display['Region'].unique()))
            market = st.selectbox(f"📊 Filter {label} by Market", ["All"] + sorted(df_display['Market'].unique()))
            alignment_filter = st.selectbox("🧭 Sharp Alignment Filter", ["All", "✅ Aligned with sharps", "🚨 Edge vs sharps"])

            if region != "All":
                df_display = df_display[df_display['Region'] == region]
            if market != "All":
                df_display = df_display[df_display['Market'] == market]
            if alignment_filter != "All" and 'SharpAlignment' in df_display.columns:
                df_display = df_display[df_display['SharpAlignment'] == alignment_filter]

            # === Final displayed columns
            cols_to_display = [
                'Game', 'Market', 'Outcome', 'Bookmaker',
                'Value', 'Ref Sharp Value', 'LineMove',
                'Delta vs Sharp', 'Limit', 'SharpConfidenceTier', 'SharpAlignment', 'SHARP_REASON'
            ]

            # === Display
            if not df_display.empty:
                def highlight_edge(row):
                    delta = row['Delta vs Sharp']
                    if abs(delta) >= 2:
                        return ['background-color: #ffcccc'] * len(row)  # red for large edge
                    elif abs(delta) >= 1:
                        return ['background-color: #fff3cd'] * len(row)  # yellow for moderate
                    else:
                        return ['background-color: #d4edda'] * len(row)  # green for aligned

                st.dataframe(
                    df_display[cols_to_display].style.apply(highlight_edge, axis=1),
                    use_container_width=True
                )
            else:
                st.info("⚠️ No results match the selected filters.")


            if not df_display.empty:
                st.dataframe(df_display[cols_to_display], use_container_width=True)

                # === Save + Upload CSV ===
                fname = f"{label}_sharp_moves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                csv_path = os.path.join(LOG_FOLDER, fname)
                df_display.to_csv(csv_path, index=False)

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
                st.info(f"⚠️ No qualifying sharp moves for {label} after filters.")
        else:
            st.info(f"⚠️ No sharp moves detected for {label}.")


        # === Current Odds View ===
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

        # === Manual Refresh Button ===
        if auto_mode == "Manual":
            if st.button(f"🔄 Refresh {label}"):
                st.rerun()

tab_nba, tab_mlb = st.tabs(["🏀 NBA", "⚾ MLB"])
render_scanner_tab("NBA", SPORTS["NBA"], tab_nba, drive)
render_scanner_tab("MLB", SPORTS["MLB"], tab_mlb, drive)
