import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]));sys.path.insert(1,str(Path(__file__).resolve().parents[1]))
from preflight_live_integration import inspect_repo,inspect_gcs,inspect_bigquery,evaluate
from test_heavy_brain_integration import TRAIN_JOB_SAMPLE

class TestLiveReadiness(unittest.TestCase):
    def test_missing_repo_blocks_live_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=evaluate(Path(tmp),'none','sharp-models','')
            self.assertEqual(r['status'],'HOLD')
            self.assertFalse(r['live_cloud_run_deployment_verified'])
            self.assertFalse(r['real_historical_system_validation_verified'])

    def test_compatible_entry_needs_full_v55_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'train_job.py').write_text(TRAIN_JOB_SAMPLE)
            r=inspect_repo(p)
            self.assertEqual(r['patch_status'],'DRY_RUN_READY')
            self.assertFalse(r['v55_primary_present'])

    def test_gcs_empty_blocks(self):
        blob=MagicMock();blob.exists.return_value=False
        bucket=MagicMock();bucket.blob.return_value=blob
        client=MagicMock();client.bucket.return_value=bucket
        r=inspect_gcs('sharp-models','archive/v1',client)
        self.assertEqual(r['status'],'BLOCKED_MISSING_INPUT_OBJECTS')
        self.assertEqual(len(r['missing_inputs']),7)

    def test_gcs_all_present_still_unverified(self):
        blob=MagicMock();blob.exists.return_value=True
        blob.generation=3;blob.size=123;blob.crc32c='crc'
        bucket=MagicMock();bucket.blob.return_value=blob
        client=MagicMock();client.bucket.return_value=bucket
        r=inspect_gcs('sharp-models','archive/v1',client)
        self.assertEqual(r['status'],'ALL_INPUT_OBJECTS_PRESENT_UNVERIFIED')
        self.assertFalse(r['point_in_time_certified'])
        self.assertFalse(r['history_validation_completed'])

    def test_bq_missing_bindings_blocks(self):
        r=inspect_bigquery({},None)
        self.assertEqual(r['status'],'BLOCKED_MISSING_TABLE_BINDINGS')
        self.assertEqual(len(r['missing_inputs']),7)

    def test_bq_existing_tables_still_unverified(self):
        keys=('games','core','signals','coverage','eligibility','rule_evaluations','universe_manifest')
        env={'NCAAF_BRAIN_BQ_'+k.upper()+'_TABLE':'sharplogger.sharp_data.'+k for k in keys}
        client=MagicMock()
        fake=MagicMock(); fake.schema=[MagicMock(name='foo')]
        fake.schema[0].name='season';fake.num_rows=300
        client.get_table.return_value=fake
        r=inspect_bigquery(env,client)
        self.assertEqual(r['status'],'ALL_TABLES_FOUND_UNVERIFIED')
        self.assertFalse(r['source_schema_certified'])
        self.assertFalse(r['history_validation_completed'])

if __name__=='__main__':unittest.main()
