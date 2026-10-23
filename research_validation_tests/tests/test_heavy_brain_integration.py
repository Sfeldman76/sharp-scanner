from __future__ import annotations
from pathlib import Path
import ast
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(1,str(Path(__file__).resolve().parents[1]))
from apply_to_repository import apply,patch_training_source
from ncaaf_heavy_brain_bridge import run_heavy_brain_shadow, INPUTS

TRAIN_JOB_SAMPLE = '''def training(sport, market, gcs, bucket, log_func, _cache):
    _ncaaf_research_heavy_run = bool(str(sport).upper() == "NCAAF" and str(market).lower().strip() == "ncaaf_research_heavy")
    _ncaaf_core_challenger_run = False
    if _ncaaf_research_heavy_run:
        import ncaaf_v55_primary_research as _v55_ats_primary
        _v55_primary = _v55_ats_primary.run_cloud(storage_client=gcs, bucket_name=bucket, log_func=log_func)
        _v55_primary_pass = bool(_v55_primary.get("status") == "COMPLETE_V55_ATS_PRIMARY_RESEARCH")
        _research = {"status": "NCAAF_RESEARCH_V2_COMPLETE"}
        if _research.get("status") != "NCAAF_RESEARCH_V2_COMPLETE":
            raise RuntimeError("failed")
        _miners=_research.get("system_miner_v3") or {}
        return
    if _ncaaf_core_challenger_run:
        return
'''

class FakeBlob:
    def __init__(self,store,name):
        self.store,self.name=store,name
        self.generation=3
    def exists(self): return self.name in self.store
    def reload(self):
        if not self.exists():raise FileNotFoundError(self.name)
    def download_as_bytes(self): return self.store[self.name]
    def upload_from_string(self,data,content_type=None,if_generation_match=None):
        if if_generation_match==0 and self.name in self.store:
            raise FileExistsError(self.name)
        self.store[self.name]=data.encode() if isinstance(data,str) else data

class FakeBucket:
    def __init__(self,store):self.store=store
    def blob(self,name):return FakeBlob(self.store,name)
class FakeGCS:
    def __init__(self,store):self.store=store
    def bucket(self,name):return FakeBucket(self.store)

class TestHeavyIntegration(unittest.TestCase):
    def test_patch_ast_idempotent_and_alias(self):
        s=patch_training_source(TRAIN_JOB_SAMPLE)
        ast.parse(s)
        self.assertIn('"ncaaf_v55_heavy"',s)
        self.assertIn('run_heavy_brain_shadow',s)
        self.assertIn('v55_primary_result=_v55_primary',s)
        self.assertEqual(s,patch_training_source(s))
        self.assertEqual(s.count('NCAAF_FULL_HISTORY_COMBINED_BRAIN_SHADOW_HOOK_V1'),1)

    def test_refuses_old_nonv55(self):
        with self.assertRaisesRegex(ValueError,'REFUSE'):
            patch_training_source(TRAIN_JOB_SAMPLE.replace('import ncaaf_v55_primary_research as _v55_ats_primary','pass'))

    def test_repo_guarded_dry_run_and_files(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);(repo/'train_job.py').write_text(TRAIN_JOB_SAMPLE)
            before=(repo/'train_job.py').read_text()
            preview=apply(repo,True)
            self.assertTrue(preview['entrypoint_patch'])
            self.assertEqual(before,(repo/'train_job.py').read_text())
            self.assertFalse((repo/'ncaaf_heavy_brain_bridge.py').exists())
            output=apply(repo)
            self.assertTrue(output['entrypoint_patch'])
            self.assertTrue((repo/'train_job.py.pre_brain_shadow.bak').exists())
            self.assertTrue((repo/'NCAAF_FULL_HISTORY_BRAIN_INTEGRATION.diff').exists())
            self.assertTrue((repo/'ncaaf_heavy_brain_bridge.py').exists())
            self.assertFalse(apply(repo)['entrypoint_patch'])

    def test_conflicting_module_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d);(repo/'train_job.py').write_text(TRAIN_JOB_SAMPLE)
            (repo/'ncaaf_heavy_brain_bridge.py').write_text('# newer existing module\n')
            with self.assertRaisesRegex(FileExistsError, 'REFUSE'):
                apply(repo)
            self.assertEqual((repo/'train_job.py').read_text(),TRAIN_JOB_SAMPLE)

    def test_missing_archival_exports_fails_closed(self):
        store={}
        out=run_heavy_brain_shadow(storage_client=FakeGCS(store),input_prefix='immutable/test',env={},publish=True)
        self.assertEqual(out['status'],'HOLD')
        self.assertEqual(set(out['missing_inputs']),set(INPUTS))
        self.assertEqual(out['bet_authority'],0)
        self.assertTrue(any(x.endswith('/status.json') for x in store))
        self.assertFalse(any('production_v1' in x for x in store))

    def test_complete_archival_fixture_runs_full_13_years(self):
        input_dir=Path(__file__).resolve().parent/'fixtures'/'SYNTHETIC_ONLY'
        if not input_dir.is_dir():self.skipTest('session smoke data not mounted')
        prefix='immutable/archive_batch'
        store={f'{prefix}/{key}.csv':(input_dir/(key+'.csv')).read_bytes() for key in INPUTS}
        out=run_heavy_brain_shadow(storage_client=FakeGCS(store),input_prefix=prefix,env={},publish=True,
                                   v55_primary_result={'status':'RETROSPECTIVE_MARGINS_READY'})
        self.assertEqual(out['universe_years'],13)
        self.assertTrue(out['all_brains_full_coverage'])
        self.assertEqual(out['bet_authority'],0)
        self.assertEqual(out['2026_selection_influence'],0)
        self.assertEqual(out['status'],'HISTORICAL_RESEARCH_COMPLETE_SHADOW_ONLY')
        self.assertEqual(len(out['input_receipts']),7)
        self.assertTrue(any(x.endswith('/systems_by_period.csv') for x in store))
        self.assertFalse(any(x.startswith('production/') for x in store))

if __name__=='__main__':unittest.main()
