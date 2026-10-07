#!/usr/bin/env python3
from pathlib import Path
import ast
import re
import shutil
import sys
import time
import py_compile

TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sharp_line_dashboard.py")
if not TARGET.exists():
    raise SystemExit(f"ERROR: {TARGET} not found")

text = TARGET.read_text(encoding="utf-8")

if "PT External Consensus Home Margin" in text and "NCAAF_PT_External_Consensus_Margin_Home" in text:
    print("Dashboard already contains PT External Consensus UI patch; no changes made.")
    py_compile.compile(str(TARGET), doraise=True)
    raise SystemExit(0)

anchor = "NCAAF production game identity unavailable; predictions withheld."
pos = text.find(anchor)
if pos < 0:
    raise SystemExit("ERROR: could not locate NCAAF production-board anchor; file left unchanged")

rec_match = re.search(r"(?m)^(?P<indent>\s*)rec=\{'_game_key':gk,.*$", text[pos:])
if not rec_match:
    raise SystemExit("ERROR: could not locate NCAAF rec constructor; file left unchanged")
rec_end = pos + rec_match.end()
indent = rec_match.group("indent")

snippet = f"""
{indent}# V2.18.3 PT display context: fixed 5-system META stays fail-closed.
{indent}# External Consensus is the independent cluster-balanced median across
{indent}# non-META Prediction Tracker source clusters and has zero production authority.
{indent}_pt_meta_margin=pd.to_numeric(pd.Series([r0.get('NCAAF_PT_Meta_Margin_Home',np.nan)]),errors='coerce').iloc[0]
{indent}_pt_meta_edge=pd.to_numeric(pd.Series([r0.get('NCAAF_PT_Meta_Edge_Home',np.nan)]),errors='coerce').iloc[0]
{indent}_pt_ext_margin=pd.to_numeric(pd.Series([r0.get('NCAAF_PT_External_Consensus_Margin_Home',np.nan)]),errors='coerce').iloc[0]
{indent}_pt_ext_edge=pd.to_numeric(pd.Series([r0.get('NCAAF_PT_External_Consensus_Edge_Home',np.nan)]),errors='coerce').iloc[0]
{indent}_pt_ext_clusters=pd.to_numeric(pd.Series([r0.get('NCAAF_PT_External_Consensus_Cluster_Count',np.nan)]),errors='coerce').iloc[0]
{indent}rec['PT Meta Home Margin']=_pt_meta_margin
{indent}rec['PT Meta vs Market']=_pt_meta_edge
{indent}rec['PT External Consensus Home Margin']=_pt_ext_margin
{indent}rec['PT External Consensus vs Market']=_pt_ext_edge
{indent}rec['_PT External Cluster Count']=_pt_ext_clusters
"""
text = text[:rec_end] + snippet + text[rec_end:]

sub_anchor = "st.subheader('NCAAF Production — Spread, H2H & Totals')"
sub_pos = text.find(sub_anchor, pos)
if sub_pos < 0:
    sub_anchor = 'st.subheader("NCAAF Production — Spread, H2H & Totals")'
    sub_pos = text.find(sub_anchor, pos)
if sub_pos < 0:
    raise SystemExit("ERROR: could not locate NCAAF production subheader; file left unchanged")

caption = (
    "\n    st.caption('PT Meta = frozen five-system benchmark and remains N/A unless all five exact inputs are present. '\n"
    "               'PT External Consensus = cluster-balanced median across available non-META Prediction Tracker source clusters '\n"
    "               '(minimum 8 clusters); display/research context only, zero production authority.')\n"
)
sub_line_end = text.find("\n", sub_pos)
text = text[:sub_line_end+1] + caption + text[sub_line_end+1:]

m = re.search(r"main\s*=\s*view\[\[(?P<inner>.*?)\]\]\.copy\(\)", text[sub_pos:], flags=re.S)
if not m:
    raise SystemExit("ERROR: could not locate NCAAF main board column list; file left unchanged")
abs_start = sub_pos + m.start()
abs_end = sub_pos + m.end()
inner = m.group("inner")
try:
    cols = ast.literal_eval("[" + inner + "]")
except Exception as exc:
    raise SystemExit(f"ERROR: could not parse NCAAF board columns: {exc}")

wanted = [
    "PT Meta Home Margin",
    "PT Meta vs Market",
    "PT External Consensus Home Margin",
    "PT External Consensus vs Market",
]
cols = [c for c in cols if c not in wanted]
try:
    insert_at = cols.index("Matchup") + 1
except ValueError:
    raise SystemExit("ERROR: Matchup column missing from NCAAF board; file left unchanged")
for c in reversed(wanted):
    cols.insert(insert_at, c)
replacement = "main=view[[" + ",".join(repr(c) for c in cols) + "]].copy()"
text = text[:abs_start] + replacement + text[abs_end:]

df_pos = text.find("st.dataframe(main", abs_start)
if df_pos < 0:
    raise SystemExit("ERROR: could not locate NCAAF st.dataframe(main...); file left unchanged")
df_line_start = text.rfind("\n", abs_start, df_pos) + 1

format_block = r"""
    # V2.18.3: make fixed META absence explicit instead of showing blank cells.
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
text = text[:df_line_start] + format_block + text[df_line_start:]

for token in (
    "PT External Consensus Home Margin",
    "PT External Consensus vs Market",
    "NCAAF_PT_External_Consensus_Margin_Home",
    "N/A (5/5 unavailable)",
):
    if token not in text:
        raise SystemExit(f"ERROR: patch validation failed; missing {token!r}; file left unchanged")

backup = TARGET.with_name(TARGET.name + f".backup-v2183-{time.strftime('%Y%m%d-%H%M%S')}")
shutil.copy2(TARGET, backup)
TARGET.write_text(text, encoding="utf-8")
try:
    py_compile.compile(str(TARGET), doraise=True)
except Exception:
    shutil.copy2(backup, TARGET)
    raise

print(f"PASS: patched {TARGET}")
print(f"Backup: {backup}")
print("Added: PT External Consensus Home Margin / PT External Consensus vs Market")
print("Fixed display: missing fixed-five PT META now shows N/A instead of blank")
