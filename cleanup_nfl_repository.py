#!/usr/bin/env python3
"""Conservative NFL repository cleanup: archive only files proven inactive.

Usage from repository root:
  python cleanup_nfl_repository.py            # dry run
  python cleanup_nfl_repository.py --apply    # move safe files into archive/

The cleanup is intentionally conservative. Some old-looking V1 modules remain
runtime dependencies of still-supported research/diagnostic routes. Those are
left in place until those routes are explicitly retired in a later cleanup.
Nothing is deleted; moved files are preserved under archive/nfl_legacy_20261002/.
"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path

ARCHIVE_DIR = Path("archive/nfl_legacy_20261002")

# These are superseded in the active production architecture and have no active
# dependency in the Edge Authority V2 / current dashboard workflow.
SAFE_SUPERSEDED_CODE = [
    "nfl_production_recommendations_v1.py",  # replaced by nfl_betting_engine_v1.py
    "nfl_system_lab_v2.py",                 # superseded by V3; V3 is self-contained
    "nfl_prospective_shadow_v1.py",         # superseded by V2 compatibility alias
    "patch_train_job_for_nfl_v1_9_3.py",    # one-time patch artifact
]

# Keep these despite old-looking names: current research/diagnostic routes still
# import them directly or via dependency chains.
PRESERVE_NOTE = {
    "nfl_betting_engine_v1.py": "retained benchmark/shadow documenting the failed generic second-stage architecture",
    "nfl_research_engine_v1.py": "legacy research route in train_job still references it",
    "nfl_structured_research_v1.py": "dependency of legacy V1.9.3 research route",
    "nfl_system_lab_v1.py": "still imported by current PBP attribution diagnostic",
    "nfl_prospective_ledger_v1.py": "dependency of current ledger v3/v4 and settlement diagnostics",
    "nfl_prospective_ledger_v2.py": "dependency of ledger v3 and legacy research route",
    "nfl_challenger_v1.py": "still imported by current research/intelligence",
    "nfl_score_engine_v1.py": "still imported by current intelligence research",
    "nfl_specialized_v1.py": "still imported by current intelligence research",
    "nfl_system_shadow_v1.py": "still owns the Role-Flip prospective clock",
    "nfl_system_shadow_v2.py": "current mechanism-family prospective clock",
    "nfl_prospective_ledger_v3.py": "current research health dependency",
    "nfl_prospective_ledger_v4.py": "current market/prospective research dependency",
    "nfl_pbp_foundation_v1.py": "current PBP research dependency",
    "nfl_pbp_attribution_v1.py": "current PBP diagnostic",
}

DOC_PATTERNS = [
    "README_FIRST.txt", "README_NFL_PRODUCTION_V1_LIVE_PARITY.txt", "NFL_PRODUCTION_V1_LIVE_PARITY_DEPLOY.txt",
    "NFL_PRODUCTION_V1_GOVERNANCE_20261001.json", "NFL_RESEARCH_V2_DECISION_HISTORY.json", "SHA256SUMS.txt",
    "NFL_PRODUCTION_V1_0*", "NFL_PRODUCTION_V1_1*", "NFL_PRODUCTION_V1_2*",
    "NFL_PRODUCTION_V1_3*", "NFL_PRODUCTION_V1_4*",
    "NFL_RESEARCH_V2_0*_DEPLOY*", "NFL_RESEARCH_V2_1*_DEPLOY*", "NFL_RESEARCH_V2_2*_DEPLOY*",
    "README_NFL_PROD_V1_0*", "README_NFL_PRODUCTION_V1_2*", "README_NFL_PRODUCTION_V1_3*",
    "README_NFL_PRODUCTION_V1_4*", "README_SYSTEM_MINER_V2*", "README_SYSTEM_MINER_V3*",
    "README_V2_2_1*", "README_V2_2_3*", "README_HOTFIX*", "README_UI_FIX*",
    "SHA256_NFL_PROD_V1_0*", "SHA256_NFL_PRODUCTION_V1_1*", "SHA256_NFL_PRODUCTION_V1_2*",
    "SHA256_NFL_PRODUCTION_V1_3*", "SHA256_NFL_PRODUCTION_V1_4*",
]

KEEP_DOCS = {
    "README_NFL_BETTING_ENGINE_V1.txt",
    "README_NFL_EDGE_AUTHORITY_V2.txt",
    "NFL_EDGE_AUTHORITY_V2_DEPLOY.txt",
    "SPORTS_EDGE_AUTHORITY_STANDARD_V1.md",
    "NFL_BETTING_ENGINE_V1_DEPLOY.txt",
    "NFL_REPOSITORY_CLEANUP_20261002.txt",
    "NFL_ACTIVE_ARCHITECTURE_20261002.json",
    "SHA256_NFL_BETTING_ENGINE_V1.txt",
}


def _static_importers(root: Path, module: str):
    """Best-effort safety check for literal Python imports outside archived code."""
    pat = re.compile(r"(^|\\s)(?:from\\s+" + re.escape(module) + r"\\s+import|import\\s+" + re.escape(module) + r"\\b)", re.M)
    hits=[]
    for p in root.glob("*.py"):
        if p.name in {module+".py", Path(__file__).name}: continue
        try: txt=p.read_text(errors="ignore")
        except Exception: continue
        if pat.search(txt): hits.append(p.name)
    return sorted(hits)


def collect(root: Path):
    found=[]; blocked={}
    for name in SAFE_SUPERSEDED_CODE:
        p=root/name
        if not (p.exists() and p.is_file()): continue
        importers=_static_importers(root,p.stem)
        if importers:
            blocked[name]=importers
            continue
        found.append(p)
    for pat in DOC_PATTERNS:
        for p in root.glob(pat):
            if p.is_file() and p.name not in KEEP_DOCS:
                found.append(p)
    seen=set(); out=[]
    for p in found:
        rp=p.resolve()
        if rp in seen: continue
        seen.add(rp); out.append(p)
    return out, blocked


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--apply",action="store_true",help="actually move files; default is dry-run")
    args=ap.parse_args()
    root=Path.cwd(); files,blocked=collect(root)
    report={
        "mode":"APPLY" if args.apply else "DRY_RUN",
        "archive":str(ARCHIVE_DIR),
        "files":[p.name for p in files],
        "blocked_by_static_dependency":blocked,
        "preserved":PRESERVE_NOTE,
        "deletes":False,
    }
    print(json.dumps(report,indent=2,sort_keys=True))
    if not args.apply:return
    (root/ARCHIVE_DIR).mkdir(parents=True,exist_ok=True)
    for src in files:
        dst=root/ARCHIVE_DIR/src.name
        if dst.exists():
            i=2
            while (root/ARCHIVE_DIR/f"{src.stem}__{i}{src.suffix}").exists(): i+=1
            dst=root/ARCHIVE_DIR/f"{src.stem}__{i}{src.suffix}"
        shutil.move(str(src),str(dst)); print(f"ARCHIVED {src.name} -> {dst}")

if __name__=="__main__":main()
