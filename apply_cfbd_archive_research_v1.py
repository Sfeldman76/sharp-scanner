#!/usr/bin/env python3
"""Safely patch existing NCAAF archive-loader, CORE, Miner; install one coordinator.

Usage: python3 ~/apply_cfbd_archive_research_v1.py --repo ~/sharp-scanner
This updates LOCAL repo files ONLY; it never calls BigQuery, CFBD API, Cloud Run,
or changes Production V1. Backs up all changed originals.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

MARK='CFBD_ARCHIVE_RESEARCH_V1_OPT_IN_20261019'

def once(t, old, new, name):
    found=t.count(old)
    if found!=1:
        raise RuntimeError(f'{name}: expected exactly one source anchor, found {found}: {old[:120]!r}')
    return t.replace(old,new,1)

def patch_loader(s):
    if MARK in s:return s
    if '--archive-preview' not in s or 'run_archive_research_preview' not in s:
        raise RuntimeError('Existing archive-preview loader required. Apply the previous archive-preview patch first.')
    s=once(s,'if not years or min(years)<2013 or max(years)>2025:',
           'if not years or min(years)<2013 or max(years)>2026:  # '+MARK,
           'archive preview 2026 support')
    s=s.replace('CFBD_ARCHIVE_PREVIEW_SEASONS_MUST_BE_2013_TO_2025',
                'CFBD_ARCHIVE_PREVIEW_SEASONS_MUST_BE_2013_TO_2026')
    return s

def function_region(s,start):
    anchor=s.find('\ndef '+start+'(')
    if anchor<0:raise RuntimeError(f'Could not find function {start}')
    next_def=s.find('\ndef ',anchor+4)
    if next_def<0:next_def=len(s)
    return s[:anchor],s[anchor:next_def],s[next_def:]

def patch_core(s):
    if MARK in s:return s
    beginning,fn,ending=function_region(s,'_attach_cfbd_core_context')
    fn=fn.replace('NCAAF_CFBD_MINER_FEATURE_TABLE','_source_table')
    insertion=(
        "    # "+MARK+": separate historical research from existing live feature source.\n"
        "    _archive_research = str(os.getenv('NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED','0')).strip().lower() in ('1','true','yes','on')\n"
        "    _source_table = ('sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1'\n"
        "                     if _archive_research else NCAAF_CFBD_MINER_FEATURE_TABLE)\n")
    fn=once(fn,'    bq=getattr(dashboard_module,"bq_client",None) if dashboard_module is not None else None\n',
            insertion+'    bq=getattr(dashboard_module,"bq_client",None) if dashboard_module is not None else None\n','CORE source selection')
    fn=once(fn,
        "    out=g.drop(columns=[c for c in g.columns if _core_raw_cfbd(c)],errors='ignore').copy()",
        "    out=g.drop(columns=[c for c in g.columns if _core_raw_cfbd(c)],errors='ignore').copy()\n"
        "    if _archive_research:\n"
        "        out=out.drop(columns=[c for c in out.columns if str(c).startswith('CFBD_')],errors='ignore')",
        'CORE prevent mixing legacy CFBD features with archive features')
    fn=once(fn,'    lo=max(2022,int(finite.min())); hi=min(max(CONFIRMATION_SEASONS),int(finite.max()))',
            '    lo=(2013 if _archive_research else max(2022,int(finite.min()))); hi=min(max(CONFIRMATION_SEASONS),int(finite.max()))',
            'CORE full archive prior-history window')
    fn=once(fn,'    ck=f"{lo}:{hi}"','    ck=f"{_source_table}:{lo}:{hi}"','CORE cache isolation')
    fn=once(fn,'    feature_cols=_core_safe_cfbd_columns(list(right.columns))',
        "    feature_cols=_core_safe_cfbd_columns(list(right.columns))\n"
        "    if _archive_research:\n"
        "        # Archived personnel/talent/ratings release-vintage is not certified.\n"
        "        # Expose ONLY strict historical-game priors rebuilt immediately above.\n"
        "        _strict_prefix=('CFBD_Rebuilt_', 'CFBD_Season_CFBD_Game_',\n"
        "            'CFBD_Recent3_CFBD_Game_', 'CFBD_Recent5_CFBD_Game_',\n"
        "            'CFBD_PrevSeason_CFBD_Game_', 'CFBD_Opp_Season_CFBD_Game_',\n"
        "            'CFBD_Opp_Recent3_CFBD_Game_', 'CFBD_Opp_Recent5_CFBD_Game_',\n"
        "            'CFBD_Opp_PrevSeason_CFBD_Game_', 'CFBD_Pregame_Matchup_')\n"
        "        feature_cols=[c for c in feature_cols if c.startswith(_strict_prefix)]",
        'CORE allow only rebuilt pregame features')
    return beginning+fn+ending

def patch_miner(s):
    if MARK in s:return s
    beginning,fn,ending=function_region(s,'_attach_cfbd_context_to_miner')
    fn=fn.replace('NCAAF_CFBD_MINER_FEATURE_TABLE','_source_table')
    insertion=(
        "    # "+MARK+": archive is research-only, never a live trigger source.\n"
        "    _archive_research = (not for_live) and str(os.getenv('NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED','0')).strip().lower() in ('1','true','yes','on')\n"
        "    _source_table = ('sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1'\n"
        "                     if _archive_research else NCAAF_CFBD_MINER_FEATURE_TABLE)\n")
    fn=once(fn,'    out=miner_games.copy()\n',insertion+
            "    out=miner_games.copy()\n"
            "    if _archive_research:\n"
            "        out=out.drop(columns=[c for c in out.columns if str(c).startswith('CFBD_')],errors='ignore')\n",
            'Miner historical-only source selection and legacy-data isolation')
    fn=once(fn,'    lo=int(finite.min()); hi=int(finite.max()); ck=f"{lo}:{hi}"',
            '    lo=int(finite.min()); hi=int(finite.max()); ck=f"{_source_table}:{lo}:{hi}"',
            'Miner cache isolation')
    fn=once(fn,'    feature_cols=[c for c in right.columns if str(c).startswith("CFBD_")]',
        "    feature_cols=[c for c in right.columns if str(c).startswith('CFBD_')]\n"
        "    if _archive_research:\n"
        "        # Crucial: never supply contemporaneous CFBD_Game_* / Opp_Game_*\n"
        "        # outcome observations or unverified preseason/static releases\n"
        "        # to the historical System Miner search.\n"
        "        def _archive_safe_pregame(c):\n"
        "            x=c[5:]\n"
        "            if x.startswith('Opp_'): x=x[4:]\n"
        "            return (x.startswith(('Season_', 'Recent3_', 'Recent5_',\n"
        "                'Games_Prior','Conference_Games_Prior', 'Conference_Game_Streak_Prior'))\n"
        "                or x in {'Matchup_Net_PPA_Adv','Matchup_Net_Success_Adv',\n"
        "                    'Matchup_Net_Explosiveness_Adv','Matchup_EarlyDown_PPA_Adv',\n"
        "                    'Matchup_PPA_Per_Drive_Adv','Matchup_Scoring_Opp_Rate_Adv'})\n"
        "        feature_cols=[c for c in feature_cols if _archive_safe_pregame(c)]",
        'Miner no-current-game, no-uncertified-static feature gate')
    return beginning+fn+ending

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.home()/'sharp-scanner')
    args=p.parse_args();repo=args.repo.expanduser().resolve()
    if not repo.is_dir():raise RuntimeError(f'Missing repo: {repo}')
    specs={
        'ncaaf_cfbd_warehouse_loader_v2.py':patch_loader,
        'ncaaf_core_challenger_v2.py':patch_core,
        'ncaaf_research_v2.py':patch_miner,
    }
    updates={}
    for filename,patch in specs.items():
        f=repo/filename
        if not f.is_file():raise RuntimeError(f'Missing required repo file: {f}')
        old=f.read_text(encoding='utf-8');new=patch(old)
        compile(new,str(f),'exec')
        updates[f]=(old,new)
    script=Path(__file__).resolve().with_name('ncaaf_cfbd_archive_research_build_v1.py')
    if not script.is_file():raise RuntimeError(f'Please upload accompanying coordinator: {script}')
    compile(script.read_text(),str(script),'exec')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    changed=[]
    for f,(old,new) in updates.items():
        if old==new:
            print(f'[PATCH] already_applied={f.name}')
            continue
        backup=f.with_name(f.name+f'.before_archive_research_{stamp}.bak')
        shutil.copy2(f,backup)
        f.write_text(new,encoding='utf-8')
        changed.append(f)
        print(f'[PATCH] modified={f} backup={backup}')
    target=repo/script.name
    if target.is_file():
        if target.read_bytes()!=script.read_bytes():
            raise RuntimeError(f'Existing coordinator differs: {target} (leave existing untouched)')
    else:
        shutil.copy2(script,target)
        print(f'[PATCH] added={target}')
    print(f'[PATCH] status=PASS code_files_changed={len(changed)} bigquery_mutations=0 production_model_mutations=0')
    print('[PATCH] Archive selection only when NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED=1; live Miner continues on original table.')

if __name__=='__main__':main()
