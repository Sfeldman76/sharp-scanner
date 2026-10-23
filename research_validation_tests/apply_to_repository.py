"""Guarded, idempotent overlay for existing NCAAF V5.5 Heavy Research.

Usage: python apply_to_repository.py /path/to/full/sharp-scanner-repository
Refuses unknown/older entrypoints. Writes a backup and unified diff.
"""
from __future__ import annotations
import argparse
import ast
import difflib
from pathlib import Path
import re
import shutil

MODULES = ("ncaaf_brain_full_history.py", "ncaaf_heavy_brain_bridge.py",
           "audit_bigquery_brain_sources.py")
MARKER = "# NCAAF_FULL_HISTORY_COMBINED_BRAIN_SHADOW_HOOK_V1"


def patch_training_source(source: str) -> str:
    if MARKER in source:
        if source.count(MARKER) != 1:
            raise ValueError("Multiple existing brain shadow hooks")
        return source
    # Never patch pre-V55 or an unrecognized training route.
    if "import ncaaf_v55_primary_research" not in source or "_v55_primary" not in source:
        raise ValueError("REFUSE: V5.5 primary Heavy Research code not present")
    match=re.search(r"(?m)^(?P<ws>[ \t]+)if _ncaaf_research_heavy_run:\s*$",source)
    if not match:
        raise ValueError("REFUSE: Heavy Research route anchor absent")
    ws=match.group("ws")
    next_head=re.search(r"(?m)^"+re.escape(ws)+r"if _ncaaf_core_challenger_run:\s*$",source[match.end():])
    if not next_head:
        raise ValueError("REFUSE: Could not identify protected end of Heavy Research route")
    end=match.end()+next_head.start()
    section=source[match.end():end]
    anchor=re.search(r"(?m)^(?P<indent>[ \t]+)_miners=_research\.get\(\"system_miner_v3\"\) or \{\}\s*$",section)
    if not anchor:
        raise ValueError("REFUSE: Expected legacy brain results anchor missing")
    if '"NCAAF_RESEARCH_V2_COMPLETE"' not in section or "_v55_primary_pass" not in section:
        raise ValueError("REFUSE: V5.5 and secondary research completion guards missing")
    indent=anchor.group("indent")
    hook='\n'.join([
        indent+MARKER,
        indent+'# Research-only: 2013-2025 all-brain history, never live bet authority.',
        indent+'# Missing archival events/eligibility remain HOLD, not inferred non-triggers.',
        indent+'try:',
        indent+'    from ncaaf_heavy_brain_bridge import run_heavy_brain_shadow as _run_full_history_brains',
        indent+'    _historical_brains = _run_full_history_brains(',
        indent+'        storage_client=gcs, bucket_name=bucket, log_func=log_func,',
        indent+'        legacy_cache=_cache, legacy_research=_research,',
        indent+'        v55_primary_result=_v55_primary,',
        indent+'    )',
        indent+'    log_func("[NCAAF-HEAVY-COMBINED-BRAINS] status="',
        indent+'             + str(_historical_brains.get("status", "HOLD"))',
        indent+'             + " reason=" + str(_historical_brains.get("reason", "UNKNOWN"))',
        indent+'             + " bet_authority=0 production_mutation=FALSE")',
        indent+'except Exception as _brain_shadow_exc:',
        indent+'    log_func("[NCAAF-HEAVY-COMBINED-BRAINS] status=HOLD type="',
        indent+'             + type(_brain_shadow_exc).__name__',
        indent+'             + " reason=" + str(_brain_shadow_exc)[:250]',
        indent+'             + " bet_authority=0 production_mutation=FALSE")',
        ''
    ])
    source=source[:match.end()+anchor.start()] + hook + source[match.end()+anchor.start():]
    # Resolve UI route mismatch without changing any other market.
    alias='str(market).lower().strip() == "ncaaf_research_heavy"'
    expanded='str(market).lower().strip() in ("ncaaf_research_heavy", "ncaaf_v55_heavy")'
    # Current consolidated scanner already uses a set-literal route alias.
    # Never rewrite that valid route or reject it for being newer than the
    # original patch fixture. Accept either proven implementation.
    alias_set = re.search(
        r'str\(market\)\.lower\(\)\.strip\(\)\s+in\s+\{[^}]*'
        r'["\']ncaaf_research_heavy["\'][^}]*'
        r'["\']ncaaf_v55_heavy["\'][^}]*\}', source
    )
    if expanded not in source and not alias_set:
        if source.count(alias) != 1:
            raise ValueError("REFUSE: heavy route market alias absent or ambiguous")
        source=source.replace(alias,expanded,1)
    ast.parse(source)
    return source


def apply(repo: Path, dry_run=False) -> dict:
    folder=Path(repo).expanduser().resolve()
    entry=folder/"train_job.py"
    if not entry.is_file():
        raise FileNotFoundError(f"REFUSE: full repository missing train_job.py: {folder}")
    original=entry.read_text(encoding="utf-8")
    patched=patch_training_source(original)
    package=Path(__file__).resolve().parent
    for filename in MODULES:
        ast.parse((package/filename).read_text(encoding="utf-8"))
    changed=patched!=original
    # Never overwrite a different/newer installed research module implicitly.
    conflicts=[filename for filename in MODULES
               if (folder/filename).exists() and
               (folder/filename).read_bytes()!=(package/filename).read_bytes()]
    if conflicts:
        raise FileExistsError("REFUSE: existing modules differ; review before replacing: "+", ".join(conflicts))
    files_to_copy=[x for x in MODULES if not (folder/x).exists() or (folder/x).read_bytes()!=(package/x).read_bytes()]
    if dry_run:
        return {"path":str(folder),"dry_run":True,"entrypoint_patch":changed,
                "modules_copy":files_to_copy,"production_v1_files_changed":False}
    backup=entry.with_name("train_job.py.pre_brain_shadow.bak")
    if changed:
        if backup.exists() and backup.read_bytes()!=entry.read_bytes():
            raise FileExistsError("REFUSE: existing train_job backup belongs to another version")
        if not backup.exists():
            shutil.copy2(entry,backup)
    for filename in files_to_copy:
        dest=folder/filename
        shutil.copy2(package/filename,dest)
    if changed:
        entry.write_text(patched,encoding="utf-8")
        diff=''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),
                                         fromfile="train_job.py.before",tofile="train_job.py.after"))
        (folder/"NCAAF_FULL_HISTORY_BRAIN_INTEGRATION.diff").write_text(diff,encoding="utf-8")
    return {"path":str(folder),"dry_run":False,"entrypoint_patch":changed,
            "modules_copy":files_to_copy,"production_v1_files_changed":False,
            "backup":str(backup) if backup.exists() else None}


def main():
    p=argparse.ArgumentParser(description="Integrate full-history shadow brain research into existing V5.5 Heavy Research")
    p.add_argument("repository",type=Path)
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    import json
    print(json.dumps(apply(args.repository,args.dry_run),indent=2))


if __name__=="__main__":
    main()
