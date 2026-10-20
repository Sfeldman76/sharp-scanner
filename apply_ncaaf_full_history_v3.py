"""Install the isolated full-history challenger without touching Production V1."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import py_compile
import shutil
import tempfile
import ast

MARKER='NCAAF_FULL_HISTORY_V3_ISOLATED_ROUTE_20261019'
BRIDGE_NAME='ncaaf_historical_core_bridge.py'
BRIDGE_FUNCTION='run_historical_to_core_bridge'
ANCHOR='    # NFL Heavy Research / Engine V3.17 — persistent Miner library + Big Al current-card fingerprint + protected PT lanes.'
ROUTE='''    # NCAAF_FULL_HISTORY_V3_ISOLATED_ROUTE_20261019 — separate research job mode.
    # No Production V1 or shared heavy-research path is executed by this branch.
    if str(sport).upper().strip() == "NCAAF" and str(market).lower().strip() == "ncaaf_full_history_v3":
        if str(os.getenv("NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED", "0")).strip().lower() not in {"1","true","yes","on"}:
            raise RuntimeError("[NCAAF-FULL-HISTORY-V3] archive opt-in required; set NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED=1")
        import sharp_line_dashboard as _v3_dashboard
        import ncaaf_full_history_challenger_v3 as _v3
        log_func("[NCAAF-FULL-HISTORY-V3] phase=HISTORICAL_CACHE source=ARCHIVE_2013_2025 year_2026_queried=FALSE production_mutation=FALSE")
        pw.emit("cache", "NCAAF Full-History V3: prepare historical game cache", pct=0.15)
        _v3_dashboard.fit_historical_ncaaf_core_expert("spreads", log_func=log_func)
        _v3_dashboard.fit_ncaaf_statistical_brain(log_func=log_func)
        pw.emit("v3", "NCAAF Full-History V3: fit expanding-origin football models", pct=0.55)
        _v3_result=_v3.run_full_history_challenger_v3(
            dashboard_module=_v3_dashboard,storage_client=gcs,
            bucket_name=bucket,log_func=log_func)
        if _v3_result.get("status") != "RESEARCH_COMPLETE_NO_PROMOTION":
            raise RuntimeError("[NCAAF-FULL-HISTORY-V3] discovery incomplete: "+str(_v3_result.get("status")))
        pw.emit("done", "NCAAF Full-History V3 complete (research only)", pct=1.0)
        return

'''


def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()

def bridge_status(repo, bridge_source):
    path=repo/BRIDGE_NAME
    # Preserve a user's existing authentic implementation. A different valid
    # version is allowed; the identity of its function is syntax-verified.
    if not path.is_file():
        compile(bridge_source,str(path),'exec')
        return 'RESTORE_MISSING_BRIDGE'
    existing=path.read_text(encoding='utf-8')
    try:
        tree=ast.parse(existing,filename=str(path))
        functions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==BRIDGE_FUNCTION]
        if len(functions)!=1:
            raise RuntimeError('existing historical CORE bridge lacks exactly one callable '+BRIDGE_FUNCTION)
        compile(existing,str(path),'exec')
    except (SyntaxError,UnicodeDecodeError) as ex:
        raise RuntimeError('existing historical CORE bridge is invalid; refusing overwrite') from ex
    return 'PRESERVE_EXISTING_BRIDGE'


def check(repo, source, bridge_source):
    target=repo/'train_job.py'
    if not target.is_file():raise RuntimeError('train_job.py missing')
    original=target.read_text(encoding='utf-8')
    if MARKER in original:
        if original.count(MARKER)!=1:raise RuntimeError('duplicate V3 marker')
        target_text=original
        status='ALREADY_INSTALLED'
    else:
        if original.count(ANCHOR)!=1:raise RuntimeError('train_job source mismatch: expected one routing anchor')
        target_text=original.replace(ANCHOR,ROUTE+ANCHOR,1)
        status='READY'
    core=repo/'ncaaf_core_challenger_v2.py'
    if not core.is_file():raise RuntimeError('CORE V2 reader missing')
    core_text=core.read_text(encoding='utf-8')
    for sentinel in ('NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED','ncaaf_cfbd_archive_research_v1','def _attach_cfbd_core_context'):
        if sentinel not in core_text:raise RuntimeError('archive V2 reader prerequisite missing: '+sentinel)
    bridge_state=bridge_status(repo,bridge_source)
    new_module=repo/'ncaaf_full_history_challenger_v3.py'
    if new_module.exists() and sha(new_module.read_text(encoding='utf-8'))!=sha(source):
        raise RuntimeError('V3 module exists with unexpected contents; refusing overwrite')
    if target_text==original and not new_module.exists():status='READY_MODULE_ONLY'
    compile(target_text,str(target),'exec'); compile(source,str(new_module),'exec')
    return status,target_text,original,bridge_state


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',required=True,type=Path)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true')
    g.add_argument('--apply',action='store_true')
    a=p.parse_args()
    repo=a.repo.expanduser().resolve()
    src=(Path(__file__).resolve().parent/'ncaaf_full_history_challenger_v3.py').read_text(encoding='utf-8')
    bridge_src=(Path(__file__).resolve().parent/BRIDGE_NAME).read_text(encoding='utf-8')
    status,patched,old,bridge_state=check(repo,src,bridge_src)
    print(f'[NCAAF-V3-INSTALL] check={status} bridge={bridge_state} train_job_current_sha={sha(old)[:16]} v3_sha={sha(src)[:16]}')
    if a.check:
        print('[NCAAF-V3-INSTALL] CHECK_PASS; files_modified=0');return
    if bridge_state=='RESTORE_MISSING_BRIDGE':
        bridge_dest=repo/BRIDGE_NAME
        # Exclusive create: never overwrite a file created since preflight.
        with bridge_dest.open('x',encoding='utf-8') as fh:
            fh.write(bridge_src)
        print('[NCAAF-V3-INSTALL] RESTORED_AUTHENTIC_BRIDGE path='+str(bridge_dest))
    if status not in ('ALREADY_INSTALLED','READY_MODULE_ONLY'):
        now=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        backup=repo/f'train_job.py.before_full_history_v3_{now}.bak'
        shutil.copy2(repo/'train_job.py',backup)
        (repo/'train_job.py').write_text(patched,encoding='utf-8')
        print('[NCAAF-V3-INSTALL] backup='+str(backup))
    if not (repo/'ncaaf_full_history_challenger_v3.py').exists():
        (repo/'ncaaf_full_history_challenger_v3.py').write_text(src,encoding='utf-8')
    print('[NCAAF-V3-INSTALL] APPLY_PASS production_files_modified=0 bigquery_writes=0 deployment_performed=FALSE')

if __name__=='__main__':main()
