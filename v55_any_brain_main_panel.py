"""Read-only, lazy Any-Brain panel from the latest committed V5.5 BigQuery board."""
import hashlib
import json

TABLE = 'sharplogger.sharp_data.ncaaf_v55_live_board'


def classify(origins):
    families = {str(x).upper().strip() for x in (origins or [])}
    if 'CORE' in families and len(families) > 1:
        return 'CORE + INDEPENDENT'
    if 'CORE' in families:
        return 'CORE ONLY'
    return 'INDEPENDENT ONLY'


def flatten(rows):
    result = []
    for game in rows:
        try:
            candidates = json.loads(game.get('V55_Any_Brain_Candidates_JSON') or '[]')
        except (TypeError, ValueError):
            continue
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            origins = candidate.get('origins') or []
            if isinstance(origins, str):
                origins = [origins]
            core = str(game.get('Selected_Team') or '').strip().casefold()
            team = str(candidate.get('team') or '').strip()
            is_independent = any(str(f).upper() != 'CORE' for f in origins)
            record = {
                'Game ID': str(candidate.get('game_id') or game.get('Game_ID') or ''),
                'Team': team,
                'Side': candidate.get('side'),
                'Category': classify(origins),
                'Opposes CORE': bool(is_independent and core and team.casefold() != core),
                'Origin families': ', '.join(map(str, origins)),
                'Research status': candidate.get('status'),
                'Book': candidate.get('price_book'),
                'Spread': candidate.get('price_spread'),
                'Odds': candidate.get('price_odds'),
                'Quote observed UTC': candidate.get('price_observed_utc'),
                'Independent qualified families': candidate.get('qualified_independent_families', 0),
                'System IDs': ', '.join(map(str, candidate.get('system_ids') or [])),
                'Bet authority': 0,
                '_raw': candidate,
            }
            result.append(record)
    return result


def fetch_latest():
    from google.cloud import bigquery
    client = bigquery.Client(project='sharplogger')
    sql = f'''WITH committed AS (
        SELECT snapshot_id, published_at, row_count, payload_sha256, audit_json, publication_type
        FROM `{TABLE}`
        WHERE record_kind='COMPLETE'
        ORDER BY published_at DESC LIMIT 1
    )
    SELECT c.snapshot_id, c.published_at, c.row_count,
           c.payload_sha256, c.audit_json, c.publication_type,
           d.payload_json
    FROM committed c
    JOIN `{TABLE}` d ON d.snapshot_id=c.snapshot_id AND d.record_kind='DATA'
    '''
    found = list(client.query(sql).result())
    if len(found) != 1:
        raise RuntimeError('LATEST_COMMITTED_V55_SNAPSHOT_UNAVAILABLE')
    row = found[0]
    payload = row.payload_json
    if hashlib.sha256(payload.encode('utf-8')).hexdigest() != row.payload_sha256:
        raise RuntimeError('LATEST_V55_PAYLOAD_SHA256_MISMATCH')
    games = json.loads(payload)
    if not isinstance(games, list) or len(games) != int(row.row_count):
        raise RuntimeError('LATEST_V55_ROW_COUNT_MISMATCH')
    audit = json.loads(row.audit_json or '{}')
    if not isinstance(audit, dict):
        raise RuntimeError('LATEST_V55_AUDIT_INVALID')
    return games, audit, {
        'snapshot_id': row.snapshot_id,
        'published_at': str(row.published_at),
        'publication_type': str(row.publication_type),
        'rows': len(games),
    }


def render_any_brain_panel():
    import streamlit as st
    st.markdown('#### V5.5 Any-Brain Candidates — Research Only')
    st.caption('Read-only candidates from the latest completed BigQuery publication. No bets are authorized here.')
    if not st.checkbox('Show Any-Brain candidates and evidence', value=False, key='v55_any_brain_main_open'):
        return
    @st.cache_data(ttl=120, show_spinner=False)
    def _cached_latest():
        return fetch_latest()
    try:
        games, audit, meta = _cached_latest()
    except Exception as exc:
        st.error(f'Any-Brain publication unavailable: {type(exc).__name__}: {exc}')
        return
    st.caption(f"Snapshot {meta['snapshot_id']} | {meta['published_at']} UTC | {meta['publication_type']} | {meta['rows']} rows")
    if not games or not all('V55_Any_Brain_Candidates_JSON' in x for x in games):
        st.warning('Latest completed snapshot has no Any-Brain schema. Background publisher may still use older code.')
        return
    records = flatten(games)
    discovery = audit.get('v55_any_brain_candidate_discovery', {})
    if not records:
        st.info('No candidates in latest published board.')
        st.json(discovery)
        return
    from collections import Counter
    group_counts = Counter(x['Category'] for x in records)
    st.caption(f"{len(records)} game-side candidates | " + ' | '.join(f'{k}: {v}' for k, v in sorted(group_counts.items())) +
               f" | Opposes CORE: {sum(x['Opposes CORE'] for x in records)}")
    options = ['ALL', 'INDEPENDENT ONLY', 'CORE + INDEPENDENT', 'CORE ONLY', 'OPPOSES CORE']
    chosen = st.selectbox('Candidate filter', options, key='v55_any_brain_main_filter')
    subset = records if chosen == 'ALL' else [x for x in records if (x['Opposes CORE'] if chosen == 'OPPOSES CORE' else x['Category'] == chosen)]
    columns = ['Game ID', 'Team', 'Side', 'Category', 'Opposes CORE', 'Origin families',
               'Research status', 'Book', 'Spread', 'Odds', 'Quote observed UTC',
               'Independent qualified families', 'System IDs']
    st.dataframe([{k: record.get(k) for k in columns} for record in subset], use_container_width=True, hide_index=True)
    if subset:
        choices = [f"{i+1}. {r['Game ID']} | {r['Team']} | {r['Category']}" for i, r in enumerate(subset)]
        picked = st.selectbox('Inspect published candidate evidence', choices, key='v55_any_brain_main_detail')
        pos = choices.index(picked)
        st.json(subset[pos]['_raw'])
    with st.expander('Backend discovery audit'):
        st.json(discovery)
        st.json(audit.get('v55_any_brain_handoff_diagnostics', {}))
