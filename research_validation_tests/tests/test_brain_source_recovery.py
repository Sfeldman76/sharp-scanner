"""Source reconstruction is descriptive only and cannot fabricate missing votes."""
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ncaaf_brain_source_recovery import (
    historical_games_from_legacy_cache, retrospective_core_from_immutable_replay,
    recover_existing_sources)
from ncaaf_heavy_brain_bridge import run_heavy_brain_shadow

class Blob:
    def __init__(self, store, name): self.store,self.name=store,name;self.generation=1
    def exists(self):return self.name in self.store
    def reload(self):pass
    def download_as_bytes(self):return self.store[self.name]
    def upload_from_string(self,value,content_type=None,if_generation_match=None):
        if if_generation_match==0 and self.name in self.store:raise FileExistsError(self.name)
        self.store[self.name]=value if isinstance(value,bytes) else value.encode()
class Bucket:
    def __init__(self,store):self.store=store
    def blob(self,key):return Blob(self.store,key)
class Client:
    def __init__(self,store):self.store=store
    def bucket(self,name):return Bucket(self.store)

class SourceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.storage={}; self.client=Client(self.storage)
        self.cache={"games":pd.DataFrame({
            "Season":[2013,2013,2025,2025],"Source_Game_ID":["g1","g1","g2","g2"],
            "Game_Date":["2013-09-01","2013-09-01","2025-09-10","2025-09-10"],
            "Actual_Margin":[5,-5,-7,7],"Is_Home":[1,0,1,0],
            "Consensus_Open_Spread":[-2,2,3,-3]})}
        self.prefix='research/ncaaf/v55/ats_primary/replay_runs/20261023T233213575183Z'
        self.replay=pd.DataFrame({"season":[2025],"game_id":["g2"],"model_home_margin":[-10.],
                                  "prediction_at_utc":["2025-09-08T00:00:00Z"],"model_id":["NONLINEAR_HGB"]})
        payload=self.replay.to_csv(index=False).encode()
        self.storage[self.prefix+'/game_predictions.csv']=payload
        self.storage[self.prefix+'/audit.json']=json.dumps({
            "status":"COMPLETE_V55_EXACT_RECIPE_RETROSPECTIVE_MARGINS",
            "predictions_sha256":hashlib.sha256(payload).hexdigest(),
            "retrospective_recipe_selected_after_historical_predictions":True}).encode()
        self.primary={'status':'COMPLETE_V55_RETROSPECTIVE_MARGINS__ATS_SOURCE_VINTAGE_HOLD',
                      'model_replay_artifacts':'gs://sharp-models/'+self.prefix}

    def test_game_side_identity_only_home_and_sealed_years(self):
        g,err=historical_games_from_legacy_cache(self.cache)
        self.assertIsNone(err);self.assertEqual(len(g),2)
        self.assertFalse(g.quote_vintage_certified.any())
        self.assertEqual(g.season.tolist(),[2013,2025])
        self.assertEqual(g.actual_home_margin.tolist(),[5,-7])

    def test_malformed_or_duplicate_games_fail_closed(self):
        x=dict(self.cache)
        x['games']=pd.concat([self.cache['games'],self.cache['games'].iloc[[0]]],ignore_index=True)
        g,err=historical_games_from_legacy_cache(x)
        self.assertIsNone(g);self.assertIn('DUPLICATE',err)

    def test_hash_verified_retro_predictions_stay_uncertified(self):
        f,err=retrospective_core_from_immutable_replay(self.client,'sharp-models',self.primary)
        self.assertIsNone(err);self.assertEqual(len(f),1)
        self.assertFalse(f.true_oos.any())
        self.assertFalse(f.pregame_features_verified.any())
        self.assertEqual(f.model_recipe_sha256.tolist(),[''])

    def test_replay_hash_mismatch_rejected(self):
        self.storage[self.prefix+'/game_predictions.csv']=b'garbled'
        out,error=retrospective_core_from_immutable_replay(self.client,'sharp-models',self.primary)
        self.assertIsNone(out);self.assertEqual(error,'V55_REPLAY_SHA256_MISMATCH')

    def test_no_fake_sixth_or_seventh_input(self):
        frames,rec,audit=recover_existing_sources(storage_client=self.client,
            bucket_name='sharp-models',legacy_cache=self.cache,v55_primary_result=self.primary)
        self.assertEqual(set(frames),{'games','core'})
        self.assertEqual(len(rec),2)
        self.assertTrue(all(not r['source_vintage_certified'] for r in rec))
        self.assertIn('rule_evaluations',audit['not_reconstructed'])

    def test_shadow_status_reports_real_remaining_gaps(self):
        out=run_heavy_brain_shadow(storage_client=self.client,bucket_name='sharp-models',
             env={},legacy_cache=self.cache,v55_primary_result=self.primary)
        self.assertEqual(out['status'],'HOLD')
        self.assertEqual(set(out['missing_inputs']),{
             'signals','coverage','eligibility','rule_evaluations','universe_manifest'})
        self.assertEqual(out['bet_authority'],0)
        self.assertEqual(out['reason'],'HISTORICAL_BRAIN_EVENT_AND_COVERAGE_ARCHIVES_MISSING')
        self.assertTrue(any('/recovered_uncertified/games.csv' in x for x in self.storage))
        self.assertFalse(any('production_v1' in x for x in self.storage))

    def test_missing_primary_stays_hold_and_no_events_created(self):
        out=run_heavy_brain_shadow(storage_client=self.client,bucket_name='sharp-models',
             env={},legacy_cache=self.cache,v55_primary_result={})
        self.assertEqual(out['status'],'HOLD');self.assertIn('core',out['missing_inputs'])
        self.assertEqual(out['source_recovery']['recovered_inputs'],['games'])

    def test_explicit_prefix_does_not_blend_uncertified_cache(self):
        out=run_heavy_brain_shadow(storage_client=self.client,bucket_name='sharp-models',
             env={},input_prefix='immutable/certified_archive',legacy_cache=self.cache,
             v55_primary_result=self.primary)
        self.assertEqual(set(out['missing_inputs']),set(['games','core','signals','coverage',
                                 'eligibility','rule_evaluations','universe_manifest']))
        self.assertNotIn('source_recovery',out)

if __name__=='__main__': unittest.main()
