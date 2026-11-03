#!/usr/bin/env python3
"""Guarded in-repo installation of the Heavy Research bridge research-only hook.

Mutates only ncaaf_heavy_brain_bridge.py with an additive 4-line import wrapper.
Never touches research engine functions, train_job, CORE, or betting authority.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

START = '# NCAAF-V55-RESEARCH-STATUS-HOOK-BEGIN'
END = '# NCAAF-V55-RESEARCH-STATUS-HOOK-END'
BLOCK = '''\n# NCAAF-V55-RESEARCH-STATUS-HOOK-BEGIN
# Keep the original bridge's return value and certification HOLD unchanged.
from ncaaf_v55_heavy_research_hook import wrap_heavy_brain_shadow as _v55_wrap_research
run_heavy_brain_shadow = _v55_wrap_research(run_heavy_brain_shadow)
# NCAAF-V55-RESEARCH-STATUS-HOOK-END
'''


def plan(source: str) -> tuple[str,str]:
    tree = ast.parse(source)
    defs = [node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
            and node.name=='run_heavy_brain_shadow']
    if len(defs)!=1:
        raise ValueError('EXPECTED_ONE_TOP_LEVEL_RUN_HEAVY_BRAIN_SHADOW_FOUND:'+str(len(defs)))
    if START in source or END in source:
        if source.count(START)==1 and source.count(END)==1 and BLOCK.strip() in source:
            return source,'ALREADY_INSTALLED'
        raise ValueError('EXISTING_HOOK_MARKERS_DO_NOT_MATCH_EXPECTED_BLOCK')
    modified = source.rstrip()+'\n'+BLOCK
    ast.parse(modified)
    return modified,'READY'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=Path.home()/'sharp-scanner')
    ap.add_argument('--apply',action='store_true')
    args=ap.parse_args()
    repo=args.repo.resolve()
    p=repo/'ncaaf_heavy_brain_bridge.py'
    required=(repo/'ncaaf_v55_research_status.py',repo/'ncaaf_v55_heavy_research_hook.py')
    for entry in (p,*required):
        if not entry.is_file():
            raise SystemExit('REQUIRED_REPO_FILE_MISSING:'+str(entry))
    original=p.read_text(encoding='utf-8')
    result,state=plan(original)
    print('SOURCE:',p)
    print('STATUS:',state)
    if not args.apply or state=='ALREADY_INSTALLED':
        print('ACTION:', 'NO_CHANGE' if state=='ALREADY_INSTALLED' else 'DRY_RUN_ADD_HOOK')
        return
    backup=repo/'ncaaf_heavy_brain_bridge.py.before_v55_research_status'
    if backup.exists():
        raise SystemExit('BACKUP_ALREADY_EXISTS_DO_NOT_OVERWRITE:'+str(backup))
    backup.write_text(original,encoding='utf-8')
    p.write_text(result,encoding='utf-8')
    print('BACKUP:',backup)
    print('ACTION: HOOK_INSTALLED_SOURCE_ONLY_NOT_DEPLOYED')

if __name__=='__main__':
    main()
