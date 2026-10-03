#!/usr/bin/env python3
"""NFL V2.6.1 active-root cleanup.

The current operator surface is intentionally only:
  * NFL Production — Weekly Update
  * NFL Model Authority — Historical Validation
  * NFL Challenger Research
  * NFL System Research / Miner

This script removes exact retired root files after checking that no remaining
root Python module imports them. Git/history or a deployment snapshot should be
used for old-release archaeology; retired runtime files are not kept in the
active repository.

Usage:
  python cleanup_nfl_repository.py          # dry run
  python cleanup_nfl_repository.py --apply  # delete exact retired files/docs
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RETIRED_CODE = [
    # Retired standalone routes / isolated dependency clusters after V2.6.1.
    "nfl_edge_attribution_v1.py",
    "nfl_frozen_confirmation_v1.py",
    "nfl_market_shadow_v1.py",
    "nfl_prospective_ledger_v4.py",
    "nfl_prospective_settlement_v1.py",
    "nfl_prospective_shadow_v2.py",
    "nfl_research_engine_v1.py",
    "nfl_structured_research_v1.py",
    "nfl_system_shadow_v1.py",
    "nfl_system_shadow_v2.py",
    # Earlier already-superseded files that may still exist in an older checkout.
    "nfl_production_recommendations_v1.py",
    "nfl_system_lab_v2.py",
    "nfl_prospective_shadow_v1.py",
    "patch_train_job_for_nfl_v1_9_3.py",
]

RETIRED_EXACT_DOCS = [
    "NFL_ACTIVE_ARCHITECTURE_20261002.json",
    "NFL_REPOSITORY_CLEANUP_20261002.txt",
    "README_NFL_BETTING_ENGINE_V1.txt",
    "NFL_BETTING_ENGINE_V1_DEPLOY.txt",
    "SHA256_NFL_BETTING_ENGINE_V1.txt",
    "README_NFL_EDGE_AUTHORITY_V2.txt",
    "NFL_EDGE_AUTHORITY_V2_DEPLOY.txt",
    "SHA256_NFL_EDGE_AUTHORITY_V2.txt",
    "README_NFL_EDGE_AUTHORITY_V2_1_FULL_RESEARCH.txt",
    "NFL_EDGE_AUTHORITY_V2_1_FULL_DIAGNOSTICS_DEPLOY.txt",
    "SHA256_NFL_EDGE_AUTHORITY_V2_1_FULL_RESEARCH.txt",
    "README_NFL_EDGE_AUTHORITY_V2_3_STAT_SELECTOR_DEPENDENCY.txt",
    "NFL_EDGE_AUTHORITY_V2_3_STAT_SELECTOR_DEPENDENCY_DEPLOY.txt",
    "SHA256_NFL_EDGE_AUTHORITY_V2_3_STAT_SELECTOR_DEPENDENCY.txt",
    "README_NFL_EDGE_AUTHORITY_V2_4_ADVANCED_STAT_RESEARCH.txt",
    "NFL_EDGE_AUTHORITY_V2_4_ADVANCED_STAT_RESEARCH_DEPLOY.txt",
    "SHA256_NFL_EDGE_AUTHORITY_V2_4_ADVANCED_STAT_RESEARCH.txt",
    "README_NFL_EDGE_AUTHORITY_V2_5_FOUR_LANE_MULTIDIMENSIONAL.txt",
    "NFL_EDGE_AUTHORITY_V2_5_FOUR_LANE_MULTIDIMENSIONAL_DEPLOY.txt",
    "SHA256_NFL_EDGE_AUTHORITY_V2_5_FOUR_LANE_MULTIDIMENSIONAL.txt",
    "README_NFL_MODEL_AUTHORITY_V2_6_FINAL_UI.txt",
    "NFL_MODEL_AUTHORITY_V2_6_DEPLOY.txt",
    "SHA256_NFL_MODEL_AUTHORITY_V2_6.txt",
]

RETIRED_DOC_PATTERNS = [
    "README_FIRST.txt",
    "README_NFL_PRODUCTION_V1_LIVE_PARITY.txt",
    "NFL_PRODUCTION_V1_LIVE_PARITY_DEPLOY.txt",
    "NFL_PRODUCTION_V1_GOVERNANCE_20261001.json",
    "NFL_RESEARCH_V2_DECISION_HISTORY.json",
    "SHA256SUMS.txt",
    "NFL_PRODUCTION_V1_0*", "NFL_PRODUCTION_V1_1*", "NFL_PRODUCTION_V1_2*",
    "NFL_PRODUCTION_V1_3*", "NFL_PRODUCTION_V1_4*",
    "NFL_RESEARCH_V2_0*_DEPLOY*", "NFL_RESEARCH_V2_1*_DEPLOY*", "NFL_RESEARCH_V2_2*_DEPLOY*",
    "README_NFL_PROD_V1_0*", "README_NFL_PRODUCTION_V1_2*", "README_NFL_PRODUCTION_V1_3*",
    "README_NFL_PRODUCTION_V1_4*", "README_SYSTEM_MINER_V2*", "README_SYSTEM_MINER_V3*",
    "README_V2_2_1*", "README_V2_2_3*", "README_HOTFIX*", "README_UI_FIX*",
    "SHA256_NFL_PROD_V1_0*", "SHA256_NFL_PRODUCTION_V1_1*", "SHA256_NFL_PRODUCTION_V1_2*",
    "SHA256_NFL_PRODUCTION_V1_3*", "SHA256_NFL_PRODUCTION_V1_4*",
]

KEEP = {
    "NFL_ACTIVE_ARCHITECTURE_20261003.json",
    "SPORTS_EDGE_AUTHORITY_STANDARD_V1.md",
    "README_NFL_MODEL_AUTHORITY_V2_6_1_UTILS_BACKEND.txt",
    "NFL_MODEL_AUTHORITY_V2_6_1_DEPLOY.txt",
    "SHA256_NFL_MODEL_AUTHORITY_V2_6_1.txt",
    "NFL_REPOSITORY_CLEANUP_20261003.txt",
    "cleanup_nfl_repository.py",
}


def _static_importers(root: Path, module: str):
    pat = re.compile(r"(^|\\s)(?:from\\s+" + re.escape(module) + r"\\s+import|import\\s+" + re.escape(module) + r"\\b)", re.M)
    hits = []
    for p in root.glob("*.py"):
        if p.name in {module + ".py", Path(__file__).name}:
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        if pat.search(txt) or f'"{module}"' in txt or f"'{module}'" in txt:
            hits.append(p.name)
    return sorted(set(hits))


def collect(root: Path):
    delete = []
    blocked = {}
    for name in RETIRED_CODE:
        p = root / name
        if not p.is_file():
            continue
        importers = _static_importers(root, p.stem)
        if importers:
            blocked[name] = importers
        else:
            delete.append(p)
    for name in RETIRED_EXACT_DOCS:
        p = root / name
        if p.is_file() and p.name not in KEEP:
            delete.append(p)
    for pat in RETIRED_DOC_PATTERNS:
        for p in root.glob(pat):
            if p.is_file() and p.name not in KEEP:
                delete.append(p)
    out = []
    seen = set()
    for p in delete:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp); out.append(p)
    return out, blocked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="delete exact retired files; default is dry-run")
    args = ap.parse_args()
    root = Path.cwd()
    files, blocked = collect(root)
    report = {
        "mode": "APPLY_DELETE" if args.apply else "DRY_RUN",
        "files": [p.name for p in files],
        "blocked_by_active_dependency": blocked,
        "active_operator_routes": [
            "nfl_production_weekly", "nfl_production_replay", "nfl_research_engine", "nfl_system_lab"
        ],
        "deletes": bool(args.apply),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.apply:
        return
    if blocked:
        raise SystemExit("Refusing cleanup: a retired-code candidate still has an active importer")
    for p in files:
        p.unlink()
        print(f"DELETED {p.name}")


if __name__ == "__main__":
    main()
