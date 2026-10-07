#!/usr/bin/env python3
from pathlib import Path
import ast
import re
import shutil
import sys
import time
import py_compile

TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('sharp_line_dashboard.py')
if not TARGET.exists():
    raise SystemExit(f'ERROR: {TARGET} not found')

text = TARGET.read_text(encoding='utf-8')
ORIGINAL = text
MARKER = 'PT_MARGIN_VS_MARKET_BOTH_V1_20261007'
if MARKER in text:
    print('Dashboard already contains PT Margin vs Market patch for NFL + NCAAF; no changes made.')
    py_compile.compile(str(TARGET), doraise=True)
    raise SystemExit(0)

# -----------------------------------------------------------------------------
# Shared NFL live PT helper.
# Reads only current nflpredictions.csv, uses source-native line* predictors,
# collapses known same-provider variants, and returns a cluster-balanced median
# home spread. No model or Bet Authority consumes this value.
# -----------------------------------------------------------------------------
helper = r'''

# PT_MARGIN_VS_MARKET_BOTH_V1_20261007
_NFL_PT_MARGIN_GCS_BLOB = 'research/nfl/external/prediction_tracker/raw/nflpredictions.csv'
_NFL_PT_MARGIN_MIN_CLUSTERS = 8
_NFL_PT_MARGIN_EXCLUDE = {
    'line','lineopen','lineavg','linemed','linemedian','linestd','linemidweek',
    'phcover','phwin','hscore','rscore','vscore','actual','total','week','date',
}
_NFL_PT_MARGIN_CLUSTERS = {
    'SAGARIN': {'linesag','linesaggm','linesagr','linesagp'},
    'PI_RATINGS': {'linepi','linepim','linepib'},
    'REGRESSION_VARIANTS': {'linel1','linel2','linel2h','linel2to','linelog'},
}
_NFL_PT_MARGIN_TEAM_ALIASES = {
    'ARI': ('ari','arizona','arizona cardinals','cardinals'), 'ATL': ('atl','atlanta','atlanta falcons','falcons'),
    'BAL': ('bal','baltimore','baltimore ravens','ravens'), 'BUF': ('buf','buffalo','buffalo bills','bills'),
    'CAR': ('car','carolina','carolina panthers','panthers'), 'CHI': ('chi','chicago','chicago bears','bears'),
    'CIN': ('cin','cincinnati','cincinnati bengals','bengals'), 'CLE': ('cle','cleveland','cleveland browns','browns'),
    'DAL': ('dal','dallas','dallas cowboys','cowboys'), 'DEN': ('den','denver','denver broncos','broncos'),
    'DET': ('det','detroit','detroit lions','lions'), 'GB': ('gb','green bay','green bay packers','packers'),
    'HOU': ('hou','houston','houston texans','texans'), 'IND': ('ind','indianapolis','indianapolis colts','colts'),
    'JAX': ('jax','jac','jacksonville','jacksonville jaguars','jaguars'), 'KC': ('kc','kansas city','kansas city chiefs','chiefs'),
    'LV': ('lv','las vegas','las vegas raiders','oakland','oakland raiders','raiders'),
    'LAC': ('lac','la chargers','los angeles chargers','san diego','san diego chargers','chargers'),
    'LAR': ('lar','la rams','los angeles rams','st louis','st louis rams','rams'),
    'MIA': ('mia','miami','miami dolphins','dolphins'), 'MIN': ('min','minnesota','minnesota vikings','vikings'),
    'NE': ('ne','new england','new england patriots','patriots'), 'NO': ('no','new orleans','new orleans saints','saints'),
    'NYG': ('nyg','new york giants','giants'), 'NYJ': ('nyj','new york jets','jets'),
    'PHI': ('phi','philadelphia','philadelphia eagles','eagles'), 'PIT': ('pit','pittsburgh','pittsburgh steelers','steelers'),
    'SEA': ('sea','seattle','seattle seahawks','seahawks'), 'SF': ('sf','sfo','san francisco','san francisco 49ers','49ers'),
    'TB': ('tb','tampa bay','tampa bay buccaneers','buccaneers','bucs'), 'TEN': ('ten','tennessee','tennessee titans','titans'),
    'WAS': ('was','wsh','washington','washington commanders','washington football team','washington redskins','commanders'),
}

def _nfl_pt_margin_norm(v):
    s=str(v or '').lower().strip().replace('&',' and ')
    s=re.sub(r'[^a-z0-9]+',' ',s)
    return re.sub(r'\s+',' ',s).strip()

_NFL_PT_MARGIN_ALIAS_TO_CODE={}
for _c,_aa in _NFL_PT_MARGIN_TEAM_ALIASES.items():
    for _a in _aa:
        _NFL_PT_MARGIN_ALIAS_TO_CODE[_nfl_pt_margin_norm(_a)] = _c

def _nfl_pt_margin_team_code(v):
    s=_nfl_pt_margin_norm(v)
    if not s: return ''
    if s in _NFL_PT_MARGIN_ALIAS_TO_CODE: return _NFL_PT_MARGIN_ALIAS_TO_CODE[s]
    hits=[]
    for a,c in _NFL_PT_MARGIN_ALIAS_TO_CODE.items():
        if len(a)>=4 and (s==a or s.startswith(a+' ') or s.endswith(' '+a)):
            hits.append((len(a),c))
    if not hits: return ''
    hits.sort(reverse=True)
    if len(hits)>1 and hits[0][0]==hits[1][0] and hits[0][1]!=hits[1][1]: return ''
    return hits[0][1]

def _nfl_pt_margin_cluster(canon):
    k=str(canon).strip().lower()
    for fam,members in _NFL_PT_MARGIN_CLUSTERS.items():
        if k in members: return fam
    return k.upper()

@st.cache_data(ttl=300,show_spinner=False)
def _nfl_pt_margin_current_map():
    try:
        raw=storage.Client().bucket(GCS_BUCKET).blob(_NFL_PT_MARGIN_GCS_BLOB).download_as_bytes()
        q=pd.read_csv(io.BytesIO(raw))
        lower={str(c).strip().lower():c for c in q.columns}
        hc=lower.get('home'); rc=lower.get('road') or lower.get('away') or lower.get('visitor')
        lc=lower.get('line')
        if hc is None or rc is None:
            return {}
        preds=[]
        for c in q.columns:
            k=str(c).strip().lower()
            if k.startswith('line') and k not in _NFL_PT_MARGIN_EXCLUDE:
                preds.append((k,c))
        out={}
        for _,r in q.iterrows():
            h=_nfl_pt_margin_team_code(r.get(hc)); a=_nfl_pt_margin_team_code(r.get(rc))
            if not h or not a: continue
            groups={}
            for canon,orig in preds:
                v=pd.to_numeric(pd.Series([r.get(orig)]),errors='coerce').iloc[0]
                if pd.notna(v): groups.setdefault(_nfl_pt_margin_cluster(canon),[]).append(float(v))
            cluster_vals=[]
            for vv in groups.values():
                if vv: cluster_vals.append(float(np.median(np.asarray(vv,float))))
            if len(cluster_vals) < _NFL_PT_MARGIN_MIN_CLUSTERS: continue
            pt_home_spread=float(np.median(np.asarray(cluster_vals,float)))
            pt_market_line=pd.to_numeric(pd.Series([r.get(lc)]),errors='coerce').iloc[0] if lc is not None else np.nan
            out[(h,a)]={'pt_home_spread':pt_home_spread,'pt_market_home_spread':float(pt_market_line) if pd.notna(pt_market_line) else np.nan,'cluster_count':len(cluster_vals)}
        return out
    except Exception:
        return {}

def _nfl_pt_margin_vs_market(home,away,spread_row=None):
    pt=_nfl_pt_margin_current_map().get((_nfl_pt_margin_team_code(home),_nfl_pt_margin_team_code(away)))
    if not pt: return np.nan
    # Prediction Tracker line predictors are home-team spreads. Convert the current
    # production quote back to a home-team spread when possible. Positive display
    # means PT is more favorable to the HOME team than the current market.
    market_home=np.nan
    if isinstance(spread_row,dict):
        mv=pd.to_numeric(pd.Series([spread_row.get('market_value')]),errors='coerce').iloc[0]
        sel=_nfl_pt_margin_team_code(spread_row.get('selected'))
        home_code=_nfl_pt_margin_team_code(home); away_code=_nfl_pt_margin_team_code(away)
        if pd.notna(mv):
            if sel==home_code: market_home=float(mv)
            elif sel==away_code: market_home=-float(mv)
    if pd.isna(market_home):
        market_home=pt.get('pt_market_home_spread',np.nan)
    if pd.isna(market_home): return np.nan
    # home-margin differential = market_home_spread - projected_home_spread
    return float(market_home)-float(pt['pt_home_spread'])
'''

nfl_anchor='def _render_nfl_betting_engine_v1_ui(df_moves_raw,label):'
if nfl_anchor not in text:
    raise SystemExit('ERROR: NFL production board function not found; file left unchanged')
text=text.replace(nfl_anchor,helper+'\n\n'+nfl_anchor,1)

# NFL: add PT Margin vs Market after the market rows are indexed.
nfl_bym="        bym={str(r.get(\"market\") or \"\").upper():r for r in grows}"
if nfl_bym not in text:
    raise SystemExit('ERROR: NFL market-row anchor not found; file left unchanged')
nfl_insert=nfl_bym+"\n        rec['PT Margin vs Market']=_nfl_pt_margin_vs_market(r0.get('home_team',''),r0.get('away_team',''),bym.get('SPREADS'))"
text=text.replace(nfl_bym,nfl_insert,1)

# Utility: update board list after a subheader/function region.
def replace_board_columns(src, start_pos, required_add, remove_cols):
    m=re.search(r"main\s*=\s*view\[\[(?P<inner>.*?)\]\]\.copy\(\)",src[start_pos:],flags=re.S)
    if not m: raise RuntimeError('board column list not found')
    a=start_pos+m.start(); b=start_pos+m.end(); inner=m.group('inner')
    cols=ast.literal_eval('['+inner+']')
    cols=[c for c in cols if c not in set(remove_cols+[required_add])]
    if 'Matchup' not in cols: raise RuntimeError('Matchup column not found')
    cols.insert(cols.index('Matchup')+1,required_add)
    rep='main=view[['+','.join(repr(c) for c in cols)+']].copy()'
    return src[:a]+rep+src[b:],a

old_pt_cols=['PT Meta Home Margin','PT Meta vs Market','PT External Consensus Home Margin','PT External Consensus vs Market']
nfl_pos=text.find('st.subheader("NFL Production — Spread, H2H & Totals")')
if nfl_pos<0: raise SystemExit('ERROR: NFL production subheader missing; file left unchanged')
try: text,nfl_board_pos=replace_board_columns(text,nfl_pos,'PT Margin vs Market',old_pt_cols)
except Exception as exc: raise SystemExit(f'ERROR: NFL board patch failed: {exc}')

# NFL formatting just before its dataframe.
nfl_df=text.find('st.dataframe(main',nfl_board_pos)
if nfl_df<0: raise SystemExit('ERROR: NFL dataframe anchor missing; file left unchanged')
nfl_line=text.rfind('\n',nfl_board_pos,nfl_df)+1
fmt_nfl="""    if 'PT Margin vs Market' in main.columns:\n        _ptv=pd.to_numeric(main['PT Margin vs Market'],errors='coerce')\n        main['PT Margin vs Market']=_ptv.map(lambda x:f'{x:+.2f} pts' if pd.notna(x) else '—')\n"""
text=text[:nfl_line]+fmt_nfl+text[nfl_line:]

# NCAAF: expose only the already-calculated external consensus edge.
ncaaf_anchor="        rec={'_game_key':gk,'_game_start':r0.get('Game_Start'),'ET Date':r0.get('ET Date',''),'Game Time':r0.get('Game Time',''),'Matchup':r0.get('Matchup','')}"
if ncaaf_anchor not in text:
    raise SystemExit('ERROR: NCAAF record anchor missing; file left unchanged')
ncaaf_insert=ncaaf_anchor+"\n        rec['PT Margin vs Market']=pd.to_numeric(pd.Series([r0.get('NCAAF_PT_External_Consensus_Edge_Home',np.nan)]),errors='coerce').iloc[0]"
text=text.replace(ncaaf_anchor,ncaaf_insert,1)

ncaaf_pos=text.find("st.subheader('NCAAF Production — Spread, H2H & Totals')")
if ncaaf_pos<0: ncaaf_pos=text.find('st.subheader("NCAAF Production — Spread, H2H & Totals")')
if ncaaf_pos<0: raise SystemExit('ERROR: NCAAF production subheader missing; file left unchanged')
try: text,ncaaf_board_pos=replace_board_columns(text,ncaaf_pos,'PT Margin vs Market',old_pt_cols)
except Exception as exc: raise SystemExit(f'ERROR: NCAAF board patch failed: {exc}')

ncaaf_df=text.find('st.dataframe(main',ncaaf_board_pos)
if ncaaf_df<0: raise SystemExit('ERROR: NCAAF dataframe anchor missing; file left unchanged')
ncaaf_line=text.rfind('\n',ncaaf_board_pos,ncaaf_df)+1
fmt_ncaaf="""    if 'PT Margin vs Market' in main.columns:\n        _ptv=pd.to_numeric(main['PT Margin vs Market'],errors='coerce')\n        main['PT Margin vs Market']=_ptv.map(lambda x:f'{x:+.2f} pts' if pd.notna(x) else '—')\n"""
text=text[:ncaaf_line]+fmt_ncaaf+text[ncaaf_line:]


# Remove the old V2.18.3 explanatory caption if that patch was already applied;
# the board now intentionally exposes one PT display field only.
_old_caption = (
    "    st.caption('PT Meta = frozen five-system benchmark and remains N/A unless all five exact inputs are present. '\n"
    "               'PT External Consensus = cluster-balanced median across available non-META Prediction Tracker source clusters '\n"
    "               '(minimum 8 clusters); display/research context only, zero production authority.')\n"
)
text=text.replace(_old_caption,'')

# Remove obsolete formatting blocks from the old four-column UI patch. The source
# fields may remain in the record for compatibility, but they are no longer shown.
_old_fmt = r"""    # V2.18.3: make fixed META absence explicit instead of showing blank cells.
    if 'PT Meta Home Margin' in main.columns:
        _v=pd.to_numeric(main['PT Meta Home Margin'],errors='coerce')
        main['PT Meta Home Margin']=_v.map(lambda x:f'{x:+.2f}' if pd.notna(x) else 'N/A (5/5 unavailable)')
    if 'PT Meta vs Market' in main.columns:
        _v=pd.to_numeric(main['PT Meta vs Market'],errors='coerce')
        main['PT Meta vs Market']=_v.map(lambda x:f'{x:+.2f} pts' if pd.notna(x) else 'N/A')
    if 'PT External Consensus Home Margin' in main.columns:
        _v=pd.to_numeric(main['PT External Consensus Home Margin'],errors='coerce')
        main['PT External Consensus Home Margin']=_v.map(lambda x:f'{x:+.2f}' if pd.notna(x) else '—')
    if 'PT External Consensus vs Market' in main.columns:
        _v=pd.to_numeric(main['PT External Consensus vs Market'],errors='coerce')
        main['PT External Consensus vs Market']=_v.map(lambda x:f'{x:+.2f} pts' if pd.notna(x) else '—')
"""
text=text.replace(_old_fmt,'')

# Hide/remove any prior V2.18.3 PT display columns from board is handled above. Old
# record fields may remain internally, which is harmless and keeps backward compatibility.
for token in (MARKER,"rec['PT Margin vs Market']","_nfl_pt_margin_vs_market",'PT Margin vs Market'):
    if token not in text:
        raise SystemExit(f'ERROR: validation failed; missing {token!r}; file left unchanged')

backup=TARGET.with_name(TARGET.name+f'.backup-pt-margin-both-{time.strftime("%Y%m%d-%H%M%S")}')
shutil.copy2(TARGET,backup)
TARGET.write_text(text,encoding='utf-8')
try:
    py_compile.compile(str(TARGET),doraise=True)
except Exception:
    shutil.copy2(backup,TARGET)
    raise

print(f'PASS: patched {TARGET}')
print(f'Backup: {backup}')
print('NFL: PT Margin vs Market = cluster-balanced current PT home-spread consensus vs current production market')
print('NCAAF: PT Margin vs Market = existing V2.18.3 external-consensus home-margin edge')
print('Authority: display only / zero')
