"""Regression: separate legacy and CFBD IDs, aliases, UTC/local rollover, ambiguous games."""
import unittest
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ncaaf_replay_identity_v2 import reconcile,make_aliases
from ncaaf_historical_brain_replay_v1 import (_stage_cfbd,_stage_context,_attach_context,
                                              compare_to_core)
from test_historical_brain_replay_v1 import sample

class RealIdentityTests(unittest.TestCase):
 def _different_ids(self):
  cfbd,ctx,core=sample()
  ctx=ctx.copy();ctx['Source_Game_ID']=['lg2013','lg2022','lg2025']
  ctx.loc[ctx.Season.eq(2022),'Game_Date']='2022-10-02T01:00:00Z'
  ctx.loc[ctx.Season.eq(2022),'Source_Team_Name']='HOME 2'
  ctx.loc[ctx.Season.eq(2022),'Source_Opponent_Name']='AWAY 2'
  core=core.copy();core['game_id']=['lg2013','lg2022','lg2025']
  return cfbd,ctx,core
 def test_legacy_ids_date_shift_and_core_mapping(self):
  cfbd,ctx,core=self._different_ids()
  u,_,_=_stage_cfbd(cfbd);c,_=_stage_context(ctx)
  g,meta=_attach_context(u,c)
  self.assertEqual(meta['matched_context_games'],3)
  self.assertEqual(meta['matched_by_source_team_and_date'],3)
  comparison,summary,core_meta=compare_to_core(g,pd.DataFrame(columns=['evaluation_status','direction','brain']),core)
  self.assertEqual(core_meta['matched_core_games'],3)
  self.assertTrue(comparison.identity_route.eq('CONTEXT_SOURCE_GAME_ID').all())
  self.assertFalse(comparison.original_price_vintage_certified.any())
 def test_alias_name_propagates_when_source_team_differs(self):
  cfbd,ctx,_=sample()
  ctx.loc[0,'Team']='CRIMSON TIDE';ctx.loc[0,'Source_Team_Name']='CRIMSON TIDE'
  ctx.loc[0,'Opponent']='TIGERS';ctx.loc[0,'Source_Opponent_Name']='TIGERS'
  aliases=pd.DataFrame({'Source_Team_Name':['HOME 1','CRIMSON TIDE','AWAY 1','TIGERS'],
                        'Team_Norm':['university one','university one','university two','university two']})
  u,_,_=_stage_cfbd(cfbd);c,_=_stage_context(ctx)
  g,meta=_attach_context(u,c,aliases=aliases)
  self.assertEqual(meta['matched_context_games'],3)
  self.assertEqual(meta['explicit_alias_mappings'],4)
 def test_conflicting_alias_is_not_accepted(self):
  aliases=pd.DataFrame({'Source_Team_Name':['HOME 1','HOME 1'],
                        'Team_Norm':['another school','different school']})
  self.assertNotIn('home1',make_aliases(aliases))
 def test_no_ambiguous_date_rematch_forced(self):
  cfbd,ctx,_=sample()
  duplicate=ctx.iloc[[0]].copy();duplicate['Source_Game_ID']='other';duplicate['Game_Date']='2013-09-08T01:00:00Z'
  context=pd.concat([ctx,duplicate],ignore_index=True)
  u,_,_=_stage_cfbd(cfbd);c,_=_stage_context(context)
  g,meta=_attach_context(u,c)
  self.assertEqual(meta['matched_context_games'],3) # exact ID first wins
  # Take away first legacy direct ID -> cannot disambiguate same pair within +/- day
  context.loc[0,'Source_Game_ID']='changed';c,_=_stage_context(context)
  g,meta=_attach_context(u,c)
  self.assertEqual(meta['matched_context_games'],2)
  self.assertFalse(g.loc[g.game_id.eq('a'),'has_context'].any())

if __name__=='__main__':unittest.main()
