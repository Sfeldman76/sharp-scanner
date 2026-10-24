"""Read-only historical brain replay regression/identity safety tests."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ncaaf_historical_brain_replay_v1 import (
 CFBD_COLUMNS, CONTEXT_COLUMNS, _stage_cfbd, _stage_context, _attach_context,
 reconstruct, replay_rules, attach_pt, pt_archive_inputs, compare_to_core)
from ncaaf_heavy_brain_bridge import run_heavy_brain_shadow


def sample():
    c=pd.DataFrame([{**dict.fromkeys(CFBD_COLUMNS,np.nan),"Season":y,"Game_ID":gid,
            "Game_Date":d,"Team":h,"Opponent":a,"Is_Home":1,
            "CFBD_Games_Prior":6,"CFBD_Matchup_Net_PPA_Adv":ppa,
            "CFBD_Matchup_Net_Success_Adv":succ} for y,gid,d,h,a,ppa,succ in [
                (2013,"a","2013-09-07T18:00:00Z","HOME 1","AWAY 1",.12,.06),
                (2022,"b","2022-10-01T18:00:00Z","HOME 2","AWAY 2",-.12,-.07),
                (2025,"c","2025-10-01T18:00:00Z","HOME 3","AWAY 3",.12,.06),
                (2025,"d","2025-10-08T18:00:00Z","HOME 4","AWAY 4",np.nan,np.nan)]])
    x=pd.DataFrame([{**dict.fromkeys(CONTEXT_COLUMNS,np.nan),"Season":y,"Source_Game_ID":gid,
            "Game_Date":d,"Game_Start":d,"Team":h,"Opponent":a,"Is_Home":1,
            "SU_Margin":margin,"Opening_Spread":spread,"Prev_SU_Loss":loss,
            "Prev_ATS_Loss":0,"WinPct_Prior_System":0.4,"Revenge_Flag_Current":1}
            for y,gid,d,h,a,margin,spread,loss in [
                (2013,"a","2013-09-07T18:00:00Z","HOME 1","AWAY 1",6,-3,1),
                (2022,"b","2022-10-01T18:00:00Z","HOME 2","AWAY 2",-10,3,0),
                (2025,"c","2025-10-01T18:00:00Z","HOME 3","AWAY 3",8,-3,1)]])
    core=pd.DataFrame({"season":[2013,2022,2025],"game_id":["a","b","c"],
          "model_home_margin":[7,-7,10]})
    return c,x,core

class Blob:
 def __init__(self,s,k):self.s=s;self.k=k
 def exists(self):return self.k in self.s
 def download_as_bytes(self):return self.s[self.k]
 def upload_from_string(self,payload,**kwargs):self.s[self.k]=payload if isinstance(payload,bytes) else payload.encode()
 def reload(self):pass
class Bucket:
 def __init__(self,s):self.s=s
 def blob(self,k):return Blob(self.s,k)
class Storage:
 def __init__(self):self.s={}
 def bucket(self,b):return Bucket(self.s)

class HistoricalReplayTests(unittest.TestCase):
 def test_full_universe_survives_unmatched_context(self):
  cfbd,ctx,core=sample()
  result=reconstruct(cfbd=cfbd,context=ctx,core=core)
  self.assertEqual(len(result['game_join_coverage']),4)
  self.assertEqual(result['audit']['source_coverage']['join']['matched_context_games'],3)
  self.assertEqual(len(result['rule_evaluations_retrospective']),4*10)
  self.assertFalse(result['audit']['independent_2024_2025_holdout_certified'])
  self.assertEqual(result['audit']['bet_authority'],0)
  missing=result['rule_evaluations_retrospective'].query('game_id == "d"')
  self.assertEqual(set(missing.query('brain == "STAT"').evaluation_status),{'UNKNOWN'})
  self.assertEqual(set(missing.query('brain == "SYSTEM_MINER"').evaluation_status),{'UNKNOWN'})

 def test_cfbd_rule_fires_and_no_fires_explicit(self):
  cfbd,ctx,_=sample()
  u,features,audit=_stage_cfbd(cfbd)
  games,_=_attach_context(u,_stage_context(ctx)[0])
  evaluations,_=replay_rules(games,features)
  p=evaluations.query('system_id == "HYP_CFBD_HOME_NET_PPA_ADV"').set_index('game_id')
  self.assertEqual(p.loc['a','evaluation_status'],'FIRE')
  self.assertEqual(p.loc['b','evaluation_status'],'NO_FIRE')
  self.assertEqual(p.loc['d','evaluation_status'],'UNKNOWN')
  self.assertEqual(p.loc['a','direction'],'HOME')
  self.assertFalse(p.rule_frozen_before_game.any())

 def test_key_number_is_non_directional(self):
  cfbd,ctx,_=sample(); x=reconstruct(cfbd=cfbd,context=ctx)
  z=x['rule_evaluations_retrospective'].query('system_id == "HYP_OPEN_HOME_KEY_3"')
  self.assertEqual(set(z.direction),{'NONE'})
  self.assertTrue(z.query('game_id == "a"').evaluation_status.eq('FIRE').all())

 def test_duplicate_physical_id_rejected(self):
  cfbd,ctx,_=sample();cfbd=pd.concat([cfbd,cfbd.iloc[[0]]])
  with self.assertRaisesRegex(ValueError,'DUPLICATE'):reconstruct(cfbd=cfbd,context=ctx)

 def test_2026_labels_sealed(self):
  cfbd,ctx,_=sample();row=cfbd.iloc[[0]].copy();row['Season']=2026;row['Game_ID']='future'
  out=reconstruct(cfbd=pd.concat([cfbd,row]),context=ctx)
  self.assertEqual(len(out['game_join_coverage']),4)
  self.assertNotIn(2026,out['cfbd_universe_manifest'].season.tolist())

 def test_core_must_be_matched_and_retrospective(self):
  cfbd,ctx,core=sample();r=reconstruct(cfbd=cfbd,context=ctx,core=core)
  x=r['core_matched_game_edges'];self.assertEqual(len(x),3)
  self.assertTrue(x.retrospective_only.all())
  self.assertFalse(x.original_price_vintage_certified.any())
  self.assertEqual(set(x.cohort).difference({'ANY_BRAIN_CONFLICT__VETO_HYPOTHESIS',
   'TWO_PLUS_FAMILIES__STRONG_HYPOTHESIS','ONE_FAMILY_SUPPORT','CORE_ONLY_NO_BRAIN_SUPPORT'}),set())

 def test_pt_sparse_exact_identity_and_timestamp_not_fabricated(self):
  cfbd,ctx,_=sample();u,features,_=_stage_cfbd(cfbd)
  games,_=_attach_context(u,_stage_context(ctx)[0])
  pt=pd.DataFrame({'season':[2025,2025], 'home_key':['home3','home4'],
      'away_key':['away3','away4'],'game_date':['2025-10-01','2025-10-08'],
      'prediction_avg_home':[8,-3]})
  signals,meta=attach_pt(games,pt)
  self.assertEqual(len(signals),2)
  self.assertEqual(meta['matched_games'],2)
  self.assertFalse(signals.source_vintage_certified.any())
  self.assertFalse(signals.rule_frozen_before_game.any())

 def test_pt_with_missing_date_is_not_turned_into_signal(self):
  cfbd,ctx,_=sample();u,features,_=_stage_cfbd(cfbd)
  games,_=_attach_context(u,_stage_context(ctx)[0])
  pt=pd.DataFrame({'season':[2025],'home_key':['home3'],'away_key':['away3'],
       'game_date':[''], 'prediction_avg_home':[5]})
  signals,meta=attach_pt(games,pt)
  self.assertTrue(signals.empty)

 def test_no_big_al_context_fabricated(self):
  cfbd,ctx,_=sample();r=reconstruct(cfbd=cfbd,context=ctx)
  self.assertNotIn('BIG_AL',r['rule_evaluations_retrospective'].brain.unique())
  self.assertIn('UNAVAILABLE',r['audit']['big_al_pick_history'])

 def test_actual_bridge_publishes_retrospective_sidecar_even_while_strict_hold(self):
  cfbd,ctx,core=sample();store=Storage()
  out=run_heavy_brain_shadow(storage_client=store,env={},legacy_cache={},
       v55_primary_result={},replay_source_frames=(cfbd,ctx))
  self.assertEqual(out['status'],'HOLD')
  self.assertEqual(out['retrospective_replay']['status'],
      'RETROSPECTIVE_DIAGNOSTICS_COMPLETE_NO_CERTIFIED_BET_AUTHORITY')
  self.assertEqual(out['bet_authority'],0)
  self.assertTrue(any('/retrospective_replay/rule_evaluations_retrospective.csv' in k for k in store.s))
  self.assertTrue(any('/retrospective_replay/cfbd_universe_manifest.csv' in k for k in store.s))
  self.assertFalse(any('/production_v1/' in k for k in store.s))

if __name__=='__main__':unittest.main()
