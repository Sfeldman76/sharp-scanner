"""Regression for actual PT archives lacking Date and no-date matchup matching."""
import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ncaaf_pt_history_identity_v3 import match_pt_to_cfbd
from ncaaf_historical_brain_replay_v1 import (_stage_cfbd, _stage_context, _attach_context, attach_pt,
    compare_to_core)
from test_historical_brain_replay_v1 import sample

class PTRealHistoryIdentityTests(unittest.TestCase):
 def test_no_date_unique_pair_matches_without_fabricating_timestamp(self):
  cfbd,ctx,core=sample();u,_,_=_stage_cfbd(cfbd);g,_=_attach_context(u,_stage_context(ctx)[0])
  pt=pd.DataFrame({'season':[2025],'home_key':['home3'],'away_key':['away3'],
       'game_date':[''],'prediction_avg_home':[5]})
  out,audit=attach_pt(g,pt)
  self.assertEqual(audit['matched_games'],1)
  self.assertEqual(audit['source_date_missing'],1)
  self.assertEqual(audit['matched_pt_with_directional_edge'],1)
  self.assertEqual(out.iloc[0].identity_method,'SEASON_UNIQUE_ORDERED_PAIR_NO_SOURCE_DATE')
  self.assertFalse(out.source_vintage_certified.any())

 def test_missing_date_rematch_is_not_forced(self):
  games=pd.DataFrame({'season':[2024,2024], 'game_id':['game1','game2'],
       'game_date_key':['2024-10-02','2024-12-30'],'home_team_key':['alabama','alabama'],
       'away_team_key':['georgia','georgia'],'open_home_spread':[-3,-2]})
  pt=pd.DataFrame({'season':[2024], 'home_key':['Alabama'],'away_key':['Georgia'],
       'game_date':[''],'prediction_avg_home':[5]})
  mapping,audit=match_pt_to_cfbd(games,pt)
  self.assertTrue(mapping.empty)
  self.assertEqual(audit['reasons']['AMBIGUOUS_REMATCH_NO_DATE'],1)

 def test_source_date_makes_rematch_unique(self):
  games=pd.DataFrame({'season':[2024,2024], 'game_id':['game1','game2'],
       'game_date_key':['2024-10-02','2024-12-30'],'home_team_key':['alabama','alabama'],
       'away_team_key':['georgia','georgia'],'open_home_spread':[-3,-2]})
  pt=pd.DataFrame({'season':[2024], 'home_key':['Alabama'],'away_key':['Georgia'],
       'game_date':['2024-12-30'],'prediction_avg_home':[5]})
  mapping,audit=match_pt_to_cfbd(games,pt)
  self.assertEqual(mapping.iloc[0].game_id,'game2')

 def test_unknown_alias_is_not_matched_or_fuzzy_guessed(self):
  games=pd.DataFrame({'season':[2025], 'game_id':['game1'],
       'game_date_key':['2025-10-01'],'home_team_key':['miami'],'away_team_key':['floridastate']})
  pt=pd.DataFrame({'season':[2025], 'home_key':['Magic School'],'away_key':['Florida State'],
       'game_date':['']})
  mapping,audit=match_pt_to_cfbd(games,pt)
  self.assertTrue(mapping.empty)
  self.assertEqual(audit['reasons']['TEAM_NOT_RESOLVED'],1)

 def test_pt_legacy_known_alias_and_explicit_table(self):
  games=pd.DataFrame({'season':[2025], 'game_id':['g'], 'game_date_key':['2025-10-01'],
      'home_team_key':['olemiss'],'away_team_key':['ucf']})
  pt=pd.DataFrame({'season':[2025], 'home_key':['Mississippi'],
      'away_key':['Central Florida'], 'game_date':['']})
  mapping,audit=match_pt_to_cfbd(games,pt)
  self.assertEqual(mapping.iloc[0].game_id,'g')
  games.loc[0,'home_team_key']='mississippi university rebels'
  aliases=pd.DataFrame({'Source_Team_Name':['Mississippi'],'Team_Norm':['mississippi university rebels']})
  mapping,audit=match_pt_to_cfbd(games,pt,aliases=aliases)
  self.assertEqual(len(mapping),1)

 def test_duplicate_pt_source_same_game_rejected(self):
  games=pd.DataFrame({'season':[2025], 'game_id':['g'], 'game_date_key':['2025-10-01'],
       'home_team_key':['alabama'],'away_team_key':['georgia']})
  pt=pd.DataFrame({'season':[2025,2025], 'home_key':['Alabama','Alabama'],
       'away_key':['Georgia','Georgia'],'game_date':['','']})
  mapping,audit=match_pt_to_cfbd(games,pt)
  self.assertTrue(mapping.empty)
  self.assertEqual(audit['reasons']['DUPLICATE_PT_ROWS_SAME_GAME'],2)

 def test_core_before_filter_versus_eligible_bets_separated(self):
  cfbd,ctx,core=sample();u,_,_=_stage_cfbd(cfbd);g,_=_attach_context(u,_stage_context(ctx)[0])
  comparisons, _,meta=compare_to_core(g,pd.DataFrame(columns=['evaluation_status','direction','brain']),core)
  self.assertEqual(meta['core_identity_matched_games'],3)
  self.assertEqual(meta['eligible_core_comparisons'],3)
  core.loc[0,'model_home_margin']=0
  comps,_,meta=compare_to_core(g,pd.DataFrame(columns=['evaluation_status','direction','brain']),core)
  self.assertEqual(meta['core_identity_matched_games'],3)
  self.assertGreaterEqual(meta['core_identity_matched_games'],meta['eligible_core_comparisons'])

if __name__=='__main__':unittest.main()

# V4 regression cases are a separate class so they are discoverable without
# changing the frozen behavior assertions above.
class PTIdentityNormalizationV4Tests(unittest.TestCase):
 def test_parser_does_not_duplicate_school_names(self):
  from ncaaf_research_v2 import _pt_team_key
  for raw, normalized in (
       ('Northern Illinois','northern illinois'),('N Illinois','northern illinois'),
       ('Southern Miss','southern mississippi'),('Southern Mississippi','southern mississippi'),
       ('S Miss','southern mississippi'),('North Carolina State','north carolina state')):
   self.assertEqual(_pt_team_key(raw),normalized)
   self.assertEqual(_pt_team_key(normalized),normalized)

 def test_actual_pt_aliases_are_school_and_season_scoped(self):
  games=pd.DataFrame({'season':[2022,2022,2022,2022],
      'game_id':['uconn','kentstate','ulmonroe','northernillinois'],
      'game_date_key':['2022-09-01']*4,
      'home_team_key':['uconn','kentstate','ulmonroe','northernillinois'],
      'away_team_key':['syracuse','georgia','texas','tulsa']})
  pt=pd.DataFrame({'season':[2022]*4,
      'home_key':['Connecticut','Kent','Louisiana Monroe','Northernorthern Illinois'],
      'away_key':['Syracuse','Georgia','Texas','Tulsa'],
      'game_date':['']*4})
  m,a=match_pt_to_cfbd(games,pt)
  self.assertEqual(len(m),4)
  self.assertEqual(a['reasons']['MATCHED'],4)

 def test_2023_sam_houston_state_alias_is_exact_and_year_scoped(self):
  # The 36 unresolved real PT rows all use the source school spelling.
  games=pd.DataFrame({'season':[2023,2024], 'game_id':['2023g','2024g'],
     'game_date_key':['2023-09-02','2024-09-02'],
     'home_team_key':['samhouston','samhouston'],
     'away_team_key':['byu','byu']})
  pt=pd.DataFrame({'season':[2023], 'home_key':['Sam Houston State'],
     'away_key':['BYU'], 'game_date':['']})
  matched,audit=match_pt_to_cfbd(games,pt)
  self.assertEqual(len(matched),1)
  self.assertEqual(matched.iloc[0].game_id,'2023g')
  self.assertEqual(audit['reasons']['MATCHED'],1)
  other=games.copy();other.loc[other.season.eq(2023),'home_team_key']='totallydifferent'
  matched,audit=match_pt_to_cfbd(other,pt)
  self.assertTrue(matched.empty)
  self.assertEqual(audit['reasons']['TEAM_NOT_RESOLVED'],1)

 def test_neutral_site_order_reversal_is_flagged_not_guessed(self):
  games=pd.DataFrame({'season':[2024],'game_id':['neutral'],'game_date_key':['2024-09-01'],
      'home_team_key':['alabama'],'away_team_key':['georgia'],
      'open_home_spread':[-3.0]})
  pt=pd.DataFrame({'season':[2024],'home_key':['Georgia'],'away_key':['Alabama'],
      'game_date':[''],'prediction_avg_home':[7.0]})
  m,a=match_pt_to_cfbd(games,pt)
  self.assertEqual(len(m),1)
  self.assertEqual(m.iloc[0].identity_orientation,'REVERSED')
  out,audit=attach_pt(games,pt)
  self.assertEqual(out.iloc[0].direction,'AWAY')
  self.assertEqual(float(out.iloc[0].pt_forecast_numeric),-7.0)
  self.assertEqual(float(out.iloc[0].pt_forecast_original_home_numeric),7.0)
  self.assertEqual(audit['pt_reversed_orientation_directional_rows'],1)

 def test_undated_reverse_orientation_rematches_remain_ambiguous(self):
  games=pd.DataFrame({'season':[2024,2024],'game_id':['regular','title'],
      'game_date_key':['2024-09-01','2024-12-01'],
      'home_team_key':['alabama','georgia'],'away_team_key':['georgia','alabama']})
  pt=pd.DataFrame({'season':[2024],'home_key':['Georgia'],'away_key':['Alabama'],
      'game_date':['']})
  m,a=match_pt_to_cfbd(games,pt)
  self.assertTrue(m.empty)
  self.assertEqual(a['reasons']['AMBIGUOUS_REMATCH_NO_DATE'],1)

 def test_dated_reverse_orientation_rematch_can_match_one(self):
  games=pd.DataFrame({'season':[2024,2024],'game_id':['regular','title'],
      'game_date_key':['2024-09-01','2024-12-01'],
      'home_team_key':['alabama','georgia'],'away_team_key':['georgia','alabama']})
  pt=pd.DataFrame({'season':[2024],'home_key':['Georgia'],'away_key':['Alabama'],
      'game_date':['2024-09-01']})
  m,a=match_pt_to_cfbd(games,pt)
  self.assertEqual(m.iloc[0].game_id,'regular')
  self.assertEqual(m.iloc[0].identity_orientation,'REVERSED')

class MarketCoverageV4Tests(unittest.TestCase):
 def test_fallback_is_only_diagnostic_and_keeps_open_count(self):
  from preflight_ncaaf_pt_core_market_v4 import market_alternatives
  games=pd.DataFrame({'season':[2025,2025,2025],'game_id':['1','2','3'],
      'open_home_spread':[-3.0,None,None],
      'Spread_Value':[-2.5,-7,None],
      'actual_home_margin':[14,8,7]})
  core=pd.DataFrame({'season':[2025]*3,'game_id':['1','2','3'],
      'model_home_margin':[7.0,10.0,2.0]})
  rep=market_alternatives(games,core)
  self.assertEqual(rep['per_season'][0]['open_present'],1)
  self.assertEqual(rep['per_season'][0]['open_missing_fallback_Spread_Value_present'],1)
  self.assertEqual(rep['per_season'][0]['market_either_present'],2)
  self.assertEqual(rep['per_season'][0]['eligible_open_edge_ge2'],1)
  self.assertTrue(rep['fallback_is_not_enabled_in_comparisons'])
