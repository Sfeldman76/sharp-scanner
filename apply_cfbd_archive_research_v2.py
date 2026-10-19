#!/usr/bin/env python3
"""Hash-locked local installer for NCAAF CFBD archive research v2.

No API, BigQuery, Cloud Run or Git operations. Default is CHECK-ONLY.

    python3 apply_cfbd_archive_research_v2.py --repo ~/sharp-scanner --check
    python3 apply_cfbd_archive_research_v2.py --repo ~/sharp-scanner --apply

The patch is bound to the three exact source files uploaded 2026-10-19.
If any have changed since uploading, refuse installation and request current files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.home()/'sharp-scanner')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--check',action='store_true',help='Dry run; default behavior')
    g.add_argument('--apply',action='store_true',help='Back up and install research files')
    args=p.parse_args()
    pkg=Path(__file__).resolve().parent
    manifest=json.loads((pkg/'sources/manifest.json').read_text())
    repo=args.repo.expanduser().resolve()
    if not repo.is_dir():
        raise SystemExit(f'[ARCHIVE-V2] REFUSE: repo does not exist: {repo}')
    # The production paths are never in the manifest.
    prohibited={'ncaaf_production_v1.py','ncaaf_production_ledger_v1.py','utils.py','train_job.py','sharp_line_dashboard.py'}
    if set(manifest)&prohibited:
        raise SystemExit('[ARCHIVE-V2] REFUSE: manifest references a production module')
    planned=[]
    for name,conf in manifest.items():
        if name!=Path(name).name or not name.endswith('.py'):
            raise SystemExit(f'[ARCHIVE-V2] REFUSE: invalid source filename {name}')
        src=pkg/'sources'/name
        dest=repo/name
        if not src.is_file() or sha(src)!=conf['patched_sha256']:
            raise SystemExit(f'[ARCHIVE-V2] REFUSE: package integrity mismatch for {name}')
        compile(src.read_text(encoding='utf-8'),str(src),'exec')
        if dest.is_file():
            actual=sha(dest)
            if actual==conf['patched_sha256']:
                planned.append((name,'ALREADY_INSTALLED',src,dest))
            elif conf['expected_before_sha256'] is not None and actual==conf['expected_before_sha256']:
                planned.append((name,'REPLACE_WITH_BACKUP',src,dest))
            else:
                raise SystemExit(f'[ARCHIVE-V2] REFUSE: {name} differs from uploaded source. No changes made. Current SHA256={actual}')
        elif conf['expected_before_sha256'] is None:
            planned.append((name,'ADD',src,dest))
        else:
            raise SystemExit(f'[ARCHIVE-V2] REFUSE: required existing source missing: {dest}')
    if not (repo/'ncaaf_production_v1.py').is_file():
        print('[ARCHIVE-V2] Note: frozen production file not found in repo; it will not be created/modified.')
    for name,action,_,_ in planned:
        print(f'[ARCHIVE-V2] {action}: {name}')
    if not args.apply:
        print('[ARCHIVE-V2] CHECK_PASS; 0 files modified; run again with --apply when ready')
        return
    # Backup every existing file before any source is modified, so a partial
    # failed copy never overwrites the only remaining version.
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    backups={}
    try:
        for name,action,_,dest in planned:
            if action!='REPLACE_WITH_BACKUP': continue
            backup=dest.with_name(dest.name+f'.before_archive_research_v2_{stamp}.bak')
            if backup.exists():
                raise RuntimeError(f'Backup path unexpectedly exists: {backup}')
            shutil.copy2(dest,backup)
            if sha(backup)!=sha(dest):
                raise RuntimeError(f'Backup checksum mismatch for {name}')
            backups[dest]=backup
            print(f'[ARCHIVE-V2] backup={backup}')
        installed=[]
        for name,action,src,dest in planned:
            if action=='ALREADY_INSTALLED':continue
            temp=dest.with_name(dest.name+f'.archive_v2_{stamp}.tmp')
            try:
                shutil.copy2(src,temp)
                if sha(temp)!=sha(src):
                    raise RuntimeError(f'Stage checksum mismatch for {name}')
                os.replace(temp,dest)
            finally:
                temp.unlink(missing_ok=True)
            installed.append((dest,action))
            print(f'[ARCHIVE-V2] installed={name}')
        for name,_,src,dest in planned:
            if sha(dest)!=sha(src):
                raise RuntimeError(f'Post-install checksum mismatch: {name}')
    except Exception:
        # Backups provide recovery in case of an unexpected partial failure.
        # Do not auto-overwrite the current repo when an uncertain write occurs.
        print('[ARCHIVE-V2] FAILED: preserved backups; inspect before retry',file=sys.stderr)
        raise
    print('[ARCHIVE-V2] APPLY_PASS; BigQuery writes=0; production files modified=0; no cloud deployment performed')
    print('[ARCHIVE-V2] IMPORTANT: new CORE archive source does NOT yet make CORE discovery train on 2013-2021; a separate challenger redesign is required.')


if __name__=='__main__':
    main()
