import sys
from pathlib import Path
import unittest
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]));sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from ncaaf_brain_full_history import report, RunConfig


def fixture():
    years=range(2013,2026)
    games=pd.DataFrame([dict(season=y,game_id=str(300000+y),kickoff_utc=f'{y}-10-07T20:00:00Z',
        actual_home_margin=8,home_spread=-3,quote_at_utc=f'{y}-10-06T09:00:00Z',
        quote_source='verified_archive_provider',quote_snapshot_id=f'fixture-quote-{y}',quote_vintage_certified=True) for y in years])
    coverage=pd.DataFrame([dict(season=y,brain=b,status='COMPLETE',eligible_games=1,feature_rows=1,
        source_hash=hashlib.sha256(f'{y}:{b}'.encode()).hexdigest(),point_in_time_certified=True)
        for y in years for b in ('STAT','BIG_AL','PATHI','SYSTEM_MINER','PT')])
    signals=pd.DataFrame([dict(season=y,game_id=str(300000+y),brain=b,system_id=b+'_A',
        side='HOME',feature_asof_utc=f'{y}-10-06T10:00:00Z',
        evidence_family=family,rule_fit_max_season=y-1,
        rule_version_sha256=hashlib.sha256(f'{y}-{b}-{b}_A'.encode()).hexdigest())
        for y in years for b,family in [('STAT','SCHEDULE'),('BIG_AL','EXPERT'),('PATHI','TREND'),
                                          ('SYSTEM_MINER','SCHEDULE'),('PT','CONSENSUS')]])
    core=pd.DataFrame([dict(season=y,game_id=str(300000+y),model_home_margin=10,
                            prediction_at_utc=f'{y}-10-06T11:00:00Z',
                            training_end_utc=f'{y-1}-12-31T00:00:00Z',
                            recipe_selected_at_utc='2016-01-01T00:00:00Z',
                            model_recipe_sha256=hashlib.sha256(b'fixture-V55-HGB-frozen').hexdigest(),
                            pregame_features_verified=True,true_oos=True,model_id='NONLINEAR_HGB')
                        for y in range(2017,2026)])
    return games,core,signals,coverage

def eligibility(g):
    return pd.DataFrame([dict(season=int(row.season),game_id=row.game_id,brain=brain,
                              evaluation_status='EVALUATED',
                              feature_asof_utc=f'{int(row.season)}-10-06T09:00:00Z')
        for row in g.itertuples() if 2013 <= row.season <= 2025
        for brain in ('STAT','BIG_AL','PATHI','SYSTEM_MINER','PT')])

def universe_manifest(g):
    return pd.DataFrame([dict(season=int(y),expected_games=len(t),
            game_ids_sha256=hashlib.sha256('\n'.join(sorted(t.game_id.astype(str))).encode()).hexdigest(),
            source='INDEPENDENT_CFBD_ARCHIVE_TEST_FIXTURE',
            source_snapshot_utc=f'{int(y)+1}-01-10T00:00:00Z')
            for y,t in g.groupby('season') if y<=2025])

def rule_evaluations(g,s):
    systems=(s[['brain','system_id']].drop_duplicates().values.tolist())
    return pd.DataFrame([dict(season=int(row.season),game_id=row.game_id,brain=brain,system_id=system,
             evaluation_status='EVALUATED',feature_asof_utc=f'{int(row.season)}-10-06T10:00:00Z',
             rule_fit_max_season=int(row.season)-1,
             rule_version_sha256=hashlib.sha256(f'{row.season}-{brain}-{system}'.encode()).hexdigest())
             for row in g.itertuples() if row.season<=2025 for brain,system in systems])

def run_fixture(g,c,s,v):
    return report(g,s,v,c,RunConfig(),eligibility(g),universe_manifest(g),rule_evaluations(g,s))

class TestFullHistory(unittest.TestCase):
    def test_all_13_years_brains_independent_of_core(self):
        g,c,s,v=fixture()
        r=run_fixture(g,c,s,v)
        self.assertEqual(r['status']['universe_years'],13)
        self.assertEqual(r['status']['brain_events'],65)
        self.assertEqual(r['status']['core_predictions'],9)
        self.assertEqual(r['status']['core_candidate_games'],9)
        self.assertEqual(r['status']['all_brains_full_coverage'],True)
        self.assertEqual(set(r['systems_by_period'].query("phase=='full_history'").brain),set(v.brain))
        self.assertEqual(r['status']['bet_authority'],0)
        # STAT and Miner deliberately share SCHEDULE and count as one family.
        self.assertEqual(set(r['core_matched'].support_families),{4})

    def test_missing_brain_year_never_means_no_trigger(self):
        g,c,s,v=fixture()
        v=v.loc[~((v.season==2019)&(v.brain=='PT'))]
        r=run_fixture(g,c,s,v)
        self.assertFalse(r['status']['all_brains_full_coverage'])
        self.assertTrue((r['coverage_matrix'].status=='UNKNOWN').any())
        missing=r['core_matched'].query('season==2019')
        self.assertEqual(missing.iloc[0].classification,'COVERAGE_UNKNOWN')
        self.assertFalse(missing.iloc[0].research_certification)

    def test_hindsight_signal_not_certified(self):
        g,c,s,v=fixture()
        s.loc[s.season==2024,'feature_asof_utc']='2024-10-08T00:00:00Z'
        r=run_fixture(g,c,s,v)
        self.assertTrue((~r['signals_ledger'].query('season==2024').system_verified).all())
        self.assertFalse(r['core_matched'].query('season==2024').research_certification.iloc[0])

    def test_refuse_2026_final_labels(self):
        g,c,s,v=fixture()
        g.loc[len(g)]=dict(season=2026,game_id='2026',kickoff_utc='2026-10-01T12:00:00Z',
                           actual_home_margin=4,home_spread=-2)
        with self.assertRaisesRegex(ValueError,'outside sealed'):
            run_fixture(g,c,s,v)

    def test_missing_rule_level_eval_holds_inference(self):
        g,c,s,v=fixture()
        r=report(g,s,v,c,RunConfig(),eligibility(g),universe_manifest(g))
        self.assertEqual(r['status']['brain_events_system_vintage_verified'],0)
        self.assertEqual(r['status']['brain_events_ats_qualifying'],0)

    def test_rule_present_only_some_years_blocks_full_history(self):
        g,c,s,v=fixture()
        re=rule_evaluations(g,s)
        # A brain-year can be covered while a particular rule-year is absent.
        re=re.loc[~((re.season==2023)&(re.system_id=='BIG_AL_A'))]
        out=report(g,s,v,c,RunConfig(),eligibility(g),universe_manifest(g),re)
        self.assertEqual(out['status']['universe_years'],13)
        self.assertGreater(out['status']['rule_year_cells_missing_or_partial'],0)
        self.assertTrue(any('SYSTEM_YEAR_CELLS' in x for x in out['status']['warnings']))

    def test_full_history_su_retained_without_market_quotes(self):
        g,c,s,v=fixture()
        g=g.drop(columns=['home_spread', 'quote_at_utc', 'quote_source',
                          'quote_snapshot_id', 'quote_vintage_certified'])
        r=run_fixture(g,c,s,v)
        self.assertEqual(r['status']['universe_years'],13)
        self.assertEqual(r['status']['brain_events_su_qualifying'],65)
        self.assertEqual(r['status']['brain_events_ats_qualifying'],0)
        sys=r['systems_by_period'].query("phase=='full_history'")
        self.assertEqual(set(sys.su_graded),{13})
        self.assertTrue((sys.graded==0).all())
        self.assertEqual(r['status']['bet_authority'],0)

    def test_price_unverified_descriptive_only(self):
        g,c,s,v=fixture()
        g['quote_vintage_certified']=False
        r=run_fixture(g,c,s,v)
        self.assertEqual(r['status']['brain_events_ats_qualifying'],0)
        self.assertTrue(r['status']['brain_events_system_vintage_verified']>0)
        self.assertFalse(r['core_matched'].research_certification.any())
        self.assertTrue(all(x==0 for x in r['systems_by_period'].graded))

    def test_absent_game_universe_manifest_prevents_certification(self):
        g,c,s,v=fixture()
        r=report(g,s,v,c,RunConfig(),eligibility(g))
        self.assertEqual(r['status']['universe_archive_years_verified'],0)
        self.assertFalse(r['status']['all_brains_full_coverage'])

    def test_brain_no_fire_known_only_with_full_game_evaluation(self):
        g,c,s,v=fixture()
        s=s.loc[s.season.ne(2020)]
        r=run_fixture(g,c,s,v)
        self.assertEqual(r['core_matched'].query('season==2020').classification.iloc[0],'NO_SIGNAL')
        self.assertTrue(r['systems_by_year'].query('season==2020').no_fire_count_known.all())
        self.assertTrue((r['systems_by_year'].query('season==2020').observed_no_fire_count==1).all())

    def test_quote_must_be_available_before_core_prediction(self):
        g,c,s,v=fixture()
        g.loc[g.season==2023,'quote_at_utc']='2023-10-06T15:00:00Z'
        r=run_fixture(g,c,s,v)
        self.assertTrue(r['core_matched'].query('season==2023').research_certification.iloc[0])

    def test_close_fallback_is_descriptive_only(self):
        g,c,s,v=fixture()
        g.loc[g.season==2025,'home_spread']=float('nan')
        g['close_home_spread']=-4.0
        r=run_fixture(g,c,s,v)
        self.assertEqual(r['status']['close_fallback_descriptive_games'],1)
        self.assertFalse(r['games'].query('season==2025').price_vintage_verified.iloc[0])

    def test_veto(self):
        g,c,s,v=fixture()
        s.loc[(s.brain=='PATHI') & (s.season==2020),'side']='AWAY'
        r=run_fixture(g,c,s,v)
        self.assertEqual(r['core_matched'].query('season==2020').classification.iloc[0],'VETO_CONFLICT')

    def test_rule_hash_mismatch_never_certified(self):
        g,c,s,v=fixture()
        s.loc[s.season.eq(2024)&s.brain.eq("BIG_AL"),"rule_version_sha256"]="a"*64
        r=run_fixture(g,c,s,v)
        row=r['signals_ledger'].query('season==2024 and brain=="BIG_AL"')
        self.assertFalse(row.system_verified.any())

    def test_unattested_core_replay_cannot_certify_combination(self):
        g,c,s,v=fixture()
        c=c.drop(columns=['model_recipe_sha256'])
        r=run_fixture(g,c,s,v)
        self.assertFalse(r['core_matched'].research_certification.any())
        self.assertIn('CORE_PREDECISION_MODEL_RECIPE_UNVERIFIED__NO_CERTIFIED_COMBINED_BETS',r['status']['warnings'])
        self.assertGreater(r['status']['brain_events_system_vintage_verified'],0)

    def test_quoted_later_than_forecast_but_pregame_is_eligible(self):
        g,c,s,v=fixture()
        g.loc[g.season.eq(2023),'quote_at_utc']='2023-10-06T15:00:00Z'
        r=run_fixture(g,c,s,v)
        row=r['core_matched'].query('season==2023').iloc[0]
        self.assertTrue(row.quote_available_at_decision)
        self.assertTrue(row.research_certification)
        self.assertFalse(row.quote_known_at_prediction)

    def test_never_firing_rule_remains_in_denominator(self):
        g,c,s,v=fixture()
        r=rule_evaluations(g,s)
        new=r.loc[r.brain.eq('STAT')].copy()
        new['system_id']='STAT_NEVER_FIRES'
        new['rule_version_sha256']=hashlib.sha256(b'neverfires').hexdigest()
        r=pd.concat([r,new],ignore_index=True)
        result=report(g,s,v,c,RunConfig(),eligibility(g),universe_manifest(g),r)
        rows=result['systems_by_year'].query('system_id=="STAT_NEVER_FIRES"')
        self.assertEqual(len(rows),13)
        self.assertTrue(rows.rule_replay_verified.all())
        self.assertTrue((rows.fired_events_observed==0).all())
        self.assertTrue((rows.observed_no_fire_count==1).all())

    def test_mixed_rule_versions_in_same_year_hold(self):
        g,c,s,v=fixture()
        re=rule_evaluations(g,s)
        # Two games needed to demonstrate incompatible versions within one year.
        extra_game=g.query('season==2023').copy()
        extra_game['game_id']='SECOND_2023'
        g=pd.concat([g,extra_game],ignore_index=True)
        # Other brain coverage no longer complete; rule audit itself must fail.
        row=re.query('season==2023 and brain=="STAT"').iloc[[0]].copy()
        row['game_id']='SECOND_2023'
        row['rule_version_sha256']='b'*64
        re=pd.concat([re,row],ignore_index=True)
        e=eligibility(g)
        # Evaluate only STAT on both games; other statuses still okay.
        replays,_=__import__('ncaaf_brain_full_history').prepare_rule_evaluations(
            re,e.assign(kickoff_utc=e.season.map({y:pd.Timestamp(f'{y}-10-07T20:00:00Z') for y in range(2013,2026)}),
            game_brain_verified=True),RunConfig())
        stat=replays.query('season==2023 and system_id=="STAT_A"')
        self.assertFalse(stat.rule_game_verified.any())

    def test_duplicate_events_fail_closed(self):
        g,c,s,v=fixture()
        s=pd.concat([s,s.iloc[[0]]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'duplicate identities'):
            run_fixture(g,c,s,v)

    def test_unverified_rule_fitted_on_future(self):
        g,c,s,v=fixture()
        s.loc[s.season==2024,'rule_fit_max_season']=2025
        r=run_fixture(g,c,s,v)
        self.assertFalse(r['signals_ledger'].query('season==2024').ats_qualifies.any())

if __name__ == '__main__':
    unittest.main()
