#!/usr/bin/env python3
from __future__ import annotations

import py_compile
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

INSERT_MARKER = "[NFL-V1.9.3-TRAIN-JOB]"

ANCHOR_TOKENS = (
    "NFL-V1.9.2",
    "v1_9_2",
    "v1.9.2",
    "v192",
    "NFL_V1_9_2_RESEARCH_FOUNDATION_READY",
    "nfl_research",
    "research_foundation",
)

BLOCK_LINES = [
    "from nfl_research_execution_v1_9_3 import run_after_v192_foundation",
    "",
    "log_func(",
    '    "[NFL-V1.9.3-TRAIN-JOB] status=START "',
    '    "after_v1_9_2=TRUE production_authority=0 "',
    '    "legacy_nfl=UNCHANGED ncaaf=UNCHANGED"',
    ")",
    "",
    "_nfl_v193 = run_after_v192_foundation(",
    "    dashboard_module=_sld,",
    "    log_func=log_func,",
    ")",
    "",
    "log_func(",
    '    "[NFL-V1.9.3-TRAIN-JOB] "',
    '    f"status={_nfl_v193.get(\'status\')} "',
    '    "production_authority=0 publication=FALSE "',
    '    "legacy_nfl=UNCHANGED ncaaf=UNCHANGED"',
    ")",
]

def score_return(lines, idx):
    if not re.match(r"^\s*return\s*(?:#.*)?$", lines[idx]):
        return -999, ""

    start = max(0, idx - 140)
    window = "\n".join(lines[start:idx + 1]).lower()

    score = 0
    why = []

    hits = [t for t in ANCHOR_TOKENS if t.lower() in window]
    if hits:
        score += min(8, 2 * len(hits))
        why.append("anchors=" + ",".join(hits[:5]))

    if "sport" in window and "nfl" in window:
        score += 3
        why.append("nfl_sport_context")

    if "production_authority" in window:
        score += 1
        why.append("authority_context")

    if "contract" in window and "ledger" in window:
        score += 2
        why.append("contract_ledger_context")

    tail = "\n".join(lines[max(0, idx - 35):idx + 1]).lower()
    if "ncaaf" in tail and "nfl" not in tail:
        score -= 8
        why.append("ncaaf_only_penalty")

    return score, ";".join(why)

def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "train_job.py").resolve()
    if not target.exists():
        print(f"ERROR: {target} does not exist.")
        return 1

    text = target.read_text(encoding="utf-8")
    if INSERT_MARKER in text:
        print("NO-OP: V1.9.3 hook already present.")
        return 0

    lines = text.splitlines()
    candidates = []

    for i in range(len(lines)):
        score, why = score_return(lines, i)
        if score > -999:
            candidates.append((score, i, why))

    candidates.sort(reverse=True)
    viable = [c for c in candidates if c[0] >= 6]

    if not viable:
        print("ERROR: Could not identify the NFL V1.9.2 early return.")
        print("No file was changed.")
        for score, i, why in candidates[:8]:
            print(f"  line {i+1}: score={score} {why}")
        return 2

    best = viable[0]
    if len(viable) > 1 and viable[1][0] == best[0]:
        print("ERROR: Ambiguous candidate returns; refusing to patch.")
        print("No file was changed.")
        for score, i, why in viable[:8]:
            print(f"  line {i+1}: score={score} {why}")
        return 3

    score, idx, why = best
    indent = re.match(r"^(\s*)", lines[idx]).group(1)

    block = []
    for ln in BLOCK_LINES:
        block.append(indent + ln if ln else "")

    new_lines = lines[:idx] + block + [""] + lines[idx:]
    new_text = "\n".join(new_lines)
    if text.endswith("\n"):
        new_text += "\n"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = target.with_name(f"{target.stem}.pre_nfl_v1_9_3_{stamp}{target.suffix}")
    shutil.copy2(target, backup)

    target.write_text(new_text, encoding="utf-8")

    try:
        py_compile.compile(str(target), doraise=True)
    except Exception as exc:
        shutil.copy2(backup, target)
        print(f"ERROR: patched train_job.py failed syntax compile: {exc}")
        print(f"RESTORED: {backup}")
        return 4

    verify = target.read_text(encoding="utf-8")
    if INSERT_MARKER not in verify:
        shutil.copy2(backup, target)
        print("ERROR: insertion marker missing; restored backup.")
        return 5

    print("PATCH PASS")
    print(f"target: {target}")
    print(f"backup: {backup}")
    print(f"inserted before original return line: {idx+1}")
    print(f"candidate score: {score}")
    print(f"reason: {why}")
    print("syntax compile: PASS")
    print("")
    print("Expected next-run order:")
    print("  [NFL-V1.9.2-CONTRACT] ... HEALTHY")
    print("  [NFL-V1.9.3-TRAIN-JOB] status=START")
    print("  [NFL-V1.9.3-ORCHESTRATION] status=ENTER")
    print("  [NFL-V1.9.3-PREFLIGHT] status=START")
    print("  ... NFL research modules ...")
    print("  [NFL-V1.9.3-FINAL] status=RESEARCH_EXECUTION_COMPLETE")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
