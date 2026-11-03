#!/usr/bin/env python3
"""Audit exact repository path for the existing fresh full-history Miner chain.

This program does not claim a new Miner rebuild; it exposes legacy coupling
before changes could affect live authority. It requires manual source review.
"""
import argparse
import json
import re
from pathlib import Path


def check(repo: Path) -> dict:
    research = repo/'ncaaf_research_v2.py'
    unified = repo/'ncaaf_unified_history_miner_v5.py'
    report={'source_files':{},'history_contract':'2013-2025','bet_authority':0,
            'full_history_discovery_ready':False,'legacy_rebuild_separation_verified':False}
    for name,p in [('ncaaf_research_v2',research),('ncaaf_unified_history_miner_v5',unified)]:
        report['source_files'][name]={'exists':p.is_file()}
    if not research.is_file() or not unified.is_file():
        report['status']='SOURCE_FILES_MISSING'
        return report
    text=research.read_text(encoding='utf-8')
    ut=unified.read_text(encoding='utf-8')
    report['existing_discovery_invocation']=bool(re.search(r'\b_run_unified_full_history\s*\(',text))
    report['existing_discovery_module']=('ncaaf_unified_history_miner_v5' in text and
                                         'def run_full_history(' in ut)
    report['incumbent_reconciliation_call_present']=bool(re.search(
        r'(?m)^\s*miners\s*,\s*incumbent_system_library_audit\s*=\s*_reconcile_incumbent_system_library\s*\(',text))
    report['legacy_catalog_passed_to_unified_miner']=bool(re.search(r'existing_catalog\s*=\s*miners\b',text))
    report['source_contains_legacy_hold']=('LEGACY_CONTRACT_HOLD' in text or 'incumbent_missing_atoms' in text)
    report['fresh_history_2013_2025_confirmed']=False  # requires actual run provenance
    report['full_history_discovery_ready']=(report['existing_discovery_invocation'] and
                                            report['existing_discovery_module'])
    report['status']='EXISTING_FULL_HISTORY_RESEARCH_PATH_PRESENT' if report['full_history_discovery_ready'] else 'DISCOVERY_ENTRY_POINT_MISSING'
    report['required_followup']='Verify a fresh full-history research audit; retire incumbent carry before live qualification.'
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path.home()/'sharp-scanner')
    args=p.parse_args()
    print(json.dumps(check(args.repo),indent=2,sort_keys=True))

if __name__=='__main__': main()
