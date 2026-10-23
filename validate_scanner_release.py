"""Quick release-integrity check for the complete Sharp Scanner V5.5 source tree.

This validates code shape and preserved source hashes only. It never makes a
GCS/BigQuery call and cannot establish real-world historical profitability.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
BASELINES = {
    'ncaaf_production_v1.py': '58e234a523001fcceb030ce42ef5af5e3082fbe29679c65031681a79d4086bca',
    'nfl_engine.py': 'd9215e0c81fd6b92d22c946e36e3f13869f6f2468ecec7156080cd597bc63e31',
    'utils.py': '04297ef4dc15612384d0b9e99b3ca47c1f65de59094582ac24401a7bdece62c1',
    'sharp_line_dashboard.py': 'fdc490557fa0da60292ed2303122bc9c0452d14583918658b21a82206c744c24',
}
MODEL_FILES = ('train_job.py','ncaaf_v55_primary_research.py','ncaaf_v552_live.py',
               'ncaaf_v55_combined_brain.py','ncaaf_brain_full_history.py',
               'ncaaf_heavy_brain_bridge.py','audit_bigquery_brain_sources.py')

def main():
    failures=[]
    for name, expected in BASELINES.items():
        p=ROOT/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            failures.append('changed protected source: '+name)
    for name in MODEL_FILES:
        p=ROOT/name
        if not p.exists():
            failures.append('missing model module: '+name)
        else:
            try: ast.parse(p.read_text())
            except SyntaxError as ex: failures.append(f'python syntax invalid: {name}: {ex}')
    entry=(ROOT/'train_job.py').read_text()
    if entry.count('NCAAF_FULL_HISTORY_COMBINED_BRAIN_SHADOW_HOOK_V1')!=1:
        failures.append('combined-brain hook missing or duplicated')
    if not ('"ncaaf_research_heavy", "ncaaf_v55_heavy"' in entry
            or '"ncaaf_research_heavy", "ncaaf_v55_heavy"' in entry):
        failures.append('NCAAF heavy route aliases absent')
    if entry.index('run_ncaaf_research_v2(') > entry.index('run_heavy_brain_shadow'):
        failures.append('historical validation hook precedes legacy historical replay')
    if not (ROOT/'research_validation_tests/tests').is_dir():
        failures.append('historical validation tests missing')
    out={'status':'LOCAL_REPOSITORY_CHECKS_PASS' if not failures else 'FAIL',
         'protected_files_unchanged':not any('protected' in f for f in failures),
         'full_history_shadow_hook_present':'NCAAF_FULL_HISTORY_COMBINED_BRAIN_SHADOW_HOOK_V1' in entry,
         'test_suite':'python -m unittest discover -s research_validation_tests/tests -v',
         'cloud_run_deployed':False,'real_historical_bigquery_validation':False,
         'bet_authority_added':0,'failures':failures}
    print(json.dumps(out,indent=2))
    return 1 if failures else 0

if __name__=='__main__':sys.exit(main())
