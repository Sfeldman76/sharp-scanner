"""Display-only V5.5 Any-Brain candidate extraction and Streamlit panel.
No recalculation, recommendations, probabilities, or betting authority.
"""
import json
import pandas as pd

FIELDS = ['Matchup','Kickoff (ET)','Team','Side','Origins','Class','System IDs',
          'Book','Spread','Odds','Quote observed (UTC)','CORE side','Opposes CORE','Game ID']


def extract_candidates(frame):
    output=[]; problems=[]
    if not isinstance(frame,pd.DataFrame):
        return pd.DataFrame(columns=FIELDS), ['NOT_A_DATAFRAME']
    if 'V55_Any_Brain_Candidates_JSON' not in frame.columns:
        return pd.DataFrame(columns=FIELDS), ['ANY_BRAIN_FIELD_MISSING_IN_SNAPSHOT']
    for ix,row in frame.iterrows():
        value=row.get('V55_Any_Brain_Candidates_JSON')
        if value is None or (isinstance(value,float) and pd.isna(value)): value='[]'
        try:
            items=json.loads(value) if isinstance(value,str) else value
        except (TypeError,ValueError):
            problems.append(f'INVALID_JSON_ROW_{ix}');continue
        if not isinstance(items,list):
            problems.append(f'NONLIST_CANDIDATES_ROW_{ix}');continue
        expected=pd.to_numeric(row.get('V55_Any_Brain_Candidate_Count'),errors='coerce')
        if pd.notna(expected) and int(expected)!=len(items):
            problems.append(f'COUNT_MISMATCH_ROW_{ix}')
        core=str(row.get('Selected_Team') or '').strip()
        kickoff=pd.to_datetime(row.get('Game_Start_UTC'),utc=True,errors='coerce')
        date=kickoff.tz_convert('America/New_York').strftime('%Y-%m-%d %I:%M %p') if pd.notna(kickoff) else ''
        for item in items:
            if not isinstance(item,dict):
                problems.append(f'INVALID_ITEM_ROW_{ix}');continue
            origins=item.get('origins') or []
            origins=sorted({str(x).upper() for x in origins}) if isinstance(origins,list) else []
            team=str(item.get('team') or '')
            independent=any(x!='CORE' for x in origins)
            core_here='CORE' in origins
            category=('COMBINED' if core_here and independent else
                      'INDEPENDENT ONLY' if independent else 'CORE ONLY')
            oppose=bool(independent and core and core.upper()!='NO CORE PICK' and team.casefold()!=core.casefold())
            sysids=item.get('system_ids') or []
            output.append({
                'Matchup':row.get('Matchup'),'Kickoff (ET)':date,'Team':team,
                'Side':item.get('side'),'Origins':' | '.join(origins),'Class':category,
                'System IDs':' | '.join(str(s) for s in sysids) if isinstance(sysids,list) else '',
                'Book':item.get('price_book'),'Spread':item.get('price_spread'),
                'Odds':item.get('price_odds'),'Quote observed (UTC)':item.get('price_observed_utc'),
                'CORE side':core,'Opposes CORE':oppose,'Game ID':str(item.get('game_id') or ''),
                '_candidate':item,'_row_index':ix,
            })
    return pd.DataFrame(output,columns=FIELDS+['_candidate','_row_index']),problems


def render_any_brain_panel(frame,audit,*,st=None):
    if st is None:
        import streamlit as st
    st.subheader('V5.5 Any-Brain Candidates — Research Shadow')
    st.caption('Read directly from the COMPLETE backend BigQuery publication. '
               'No UI scoring; research candidates are not authorized bets.')
    discovery=(audit or {}).get('v55_any_brain_candidate_discovery') or {}
    bridge=(audit or {}).get('v55_any_brain_shadow_bridge') or {}
    data,problems=extract_candidates(frame)
    if problems:
        st.warning('Candidate publication incomplete: '+', '.join(problems[:5]))
    if not discovery:
        st.warning('Any-Brain audit missing; this may be an older background snapshot.')
    if not isinstance(bridge,dict) or not bridge:
        st.caption('Research signal bridge audit was not included in this snapshot.')
    if not data.empty:
        a,b,c,d=st.columns(4)
        a.metric('Candidate game-sides',len(data))
        b.metric('Independent-origin sides',int(data['Class'].isin(['INDEPENDENT ONLY','COMBINED']).sum()))
        c.metric('Independent-only sides',int(data['Class'].eq('INDEPENDENT ONLY').sum()))
        d.metric('Opposite CORE sides',int(data['Opposes CORE'].sum()))
        # Show only games with future kickoff; use exact source timestamps, no inferred freshness.
        types=st.multiselect('Candidate type', ['CORE ONLY','INDEPENDENT ONLY','COMBINED'],
                             default=['CORE ONLY','INDEPENDENT ONLY','COMBINED'],key='v55_ab_types')
        visible=data.loc[data['Class'].isin(types)].reset_index(drop=True)
        choices=visible[FIELDS].copy()
        st.dataframe(choices,use_container_width=True,hide_index=True)
        if not visible.empty:
            labels=[f"{r['Matchup']} — {r['Team']} — {r['Origins']} ({r['Game ID']})" for _,r in visible.iterrows()]
            selected=st.selectbox('Inspect candidate system evidence',range(len(labels)),
                                  format_func=lambda j:labels[j],key='v55_ab_evidence_row')
            candidate=visible.iloc[selected]['_candidate']
            st.json(candidate)
            st.caption('System IDs are producer-supplied and remain shadow-only unless a frozen OOS registry independently validates them.')
    else:
        st.info('No Any-Brain candidate records in this published snapshot.')
    st.caption(f"Published discovery: {discovery.get('status','NOT_PUBLISHED')} · "
               f"Historically qualified sides: {discovery.get('qualified_candidate_game_sides','UNAVAILABLE')} · "
               f"Bet authority changed: {discovery.get('bet_authority_changed','UNAVAILABLE')}")
