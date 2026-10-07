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
original = text
DISPLAY_COLS = {
    'PT Margin vs Market',
    'PT Meta Home Margin',
    'PT Meta vs Market',
    'PT External Consensus Home Margin',
    'PT External Consensus vs Market',
}

lines=text.splitlines(True)
out=[]
i=0
while i < len(lines):
    line=lines[i]

    # New combined UI helper: remove from marker through the line before NFL renderer.
    if line.strip() == '# PT_MARGIN_VS_MARKET_BOTH_V1_20261007':
        i += 1
        while i < len(lines) and not lines[i].startswith('def _render_nfl_betting_engine_v1_ui(df_moves_raw,label):'):
            i += 1
        continue

    # Older NCAAF display-only record block.
    if line.strip() == '# V2.18.3 PT display context: fixed 5-system META stays fail-closed.':
        i += 1
        while i < len(lines):
            if "rec['_PT External Cluster Count']" in lines[i]:
                i += 1
                break
            i += 1
        continue

    # Old caption: three physical lines beginning with this caption.
    if "st.caption('PT Meta = frozen five-system benchmark" in line:
        i += 1
        while i < len(lines) and "zero production authority.')" not in lines[i]:
            i += 1
        if i < len(lines): i += 1
        continue

    # Direct PT UI record assignments.
    if any((f"rec['{c}']" in line or f'rec["{c}"]' in line) for c in DISPLAY_COLS):
        i += 1
        continue

    # Formatter blocks: each generated variant is `if` + two child lines.
    fmt_col=None
    for c in DISPLAY_COLS:
        if re.match(rf"^\s*if ['\"]{re.escape(c)}['\"] in main\.columns:\s*$", line.rstrip('\n')):
            fmt_col=c; break
    if fmt_col is not None:
        base_indent=len(line)-len(line.lstrip())
        i += 1
        while i < len(lines):
            nxt=lines[i]
            if not nxt.strip():
                i += 1
                continue
            nindent=len(nxt)-len(nxt.lstrip())
            if nindent <= base_indent:
                break
            i += 1
        continue

    if line.strip() == '# V2.18.3: make fixed META absence explicit instead of showing blank cells.':
        i += 1
        continue

    # Board column lists are single lines in the production dashboard.
    if 'main=view[[' in line and ']].copy()' in line:
        m=re.search(r"main\s*=\s*view\[\[(.*?)\]\]\.copy\(\)", line)
        if m:
            try:
                cols=ast.literal_eval('['+m.group(1)+']')
                newcols=[c for c in cols if c not in DISPLAY_COLS]
                if newcols != cols:
                    repl='main=view[['+','.join(repr(c) for c in newcols)+']].copy()'
                    line=line[:m.start()]+repl+line[m.end():]
            except Exception:
                pass

    out.append(line)
    i += 1

text=''.join(out)

remaining=[c for c in DISPLAY_COLS if c in text]
if remaining:
    raise SystemExit(f'ERROR: PT UI labels still remain: {remaining}; file left unchanged')

if text == original:
    py_compile.compile(str(TARGET), doraise=True)
    print('PASS: no PT margin/UI display fields were present; no changes needed.')
    raise SystemExit(0)

backup = TARGET.with_name(TARGET.name+f'.backup-remove-pt-ui-{time.strftime("%Y%m%d-%H%M%S")}')
shutil.copy2(TARGET, backup)
TARGET.write_text(text, encoding='utf-8')
try:
    py_compile.compile(str(TARGET), doraise=True)
except Exception:
    shutil.copy2(backup, TARGET)
    raise

print(f'PASS: removed PT margin/UI fields from {TARGET}')
print(f'Backup: {backup}')
print('Backend PT research/miner/residual code is unchanged.')
