"""No BigQuery, GCS credentials, or production files needed."""
import json
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import ncaaf_v41_integrity as v
import ncaaf_v41_prospective as lock

rng = np.random.default_rng(91)
rows = []
for yr in range(2017, 2026):
    for j in range(300):
        actual = float(rng.normal(0, 17))
        rows.append(dict(Season=yr, Game_Date=f'{yr}-09-{j%28+1:02d}',
            Game_ID=f'{yr}-{j:04}', Team=f'Team_{j%75}', Opponent=f'Opp_{(j+2)%75}',
            Actual_Margin=actual, NONLINEAR_STATE_HGB=actual+rng.normal(0,16),
            V3_PROGRAM_STRENGTH_RIDGE=actual+rng.normal(0,20),
            DYNAMIC_MODERATE=actual+rng.normal(0,17),
            NONLINEAR_HGB=actual+rng.normal(0,16.1),
            Verified_Open_Margin_Diagnostic=actual+rng.normal(0,15) if j%3 else np.nan))
frame = pd.DataFrame(rows)
folds = {}
for yr, grp in frame.groupby('Season'):
    folds[str(yr)] = dict(status='PASS', train_end=int(yr)-1, valid=len(grp),
             models={m:v.score(grp.Actual_Margin.to_numpy(float),grp[m].to_numpy(float))
                    for m in ('NONLINEAR_STATE_HGB','V3_PROGRAM_STRENGTH_RIDGE',
                              'DYNAMIC_MODERATE','NONLINEAR_HGB')})
report = dict(status='RESEARCH_COMPLETE_NO_PROMOTION',
    selected_pre2024='NONLINEAR_STATE_HGB',
    source='sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1',
    audit={'2026_labels_accessed':False,
           'pregame_point_in_time_certification':'NOT_COMPLETE_SOURCE_VINTAGE_AUDIT_REQUIRED'},
    current_2026_outcomes_queried=False, automatic_promotion=False,
    production_mutated=False, training_dependency_on_opening_line=False,
    selection_uses_2024_2025=False, folds=folds,
    discovery_rank=[{'status':'ELIGIBLE','model':'NONLINEAR_STATE_HGB','pooled_rmse':17.1},
                    {'status':'ELIGIBLE','model':'V3_PROGRAM_STRENGTH_RIDGE','pooled_rmse':18.7}])
result,errors = v.analyze(report,frame,bootstrap=50)
assert result['status']=='RETROSPECTIVE_AUDITED__SOURCE_VINTAGE_HOLD'
assert result['integrity']['status']=='PASS_REPRODUCED_LEDGER_METRICS'
assert result['source_point_in_time_gate']['status']=='HOLD_SOURCE_VINTAGE_NOT_CERTIFIED'
assert result['state_filter_canaries']['status']=='PASS_SYNTHETIC_DYNAMIC_STATE_CANARIES'
assert result['prospective_2026_prediction_count']==0
assert len(errors)==len(frame)
assert result['retrospective_paired_v3']['rmse_gain_ci95']
assert result['opening_line_matched_n']>100
# Perturbed archive results must not silently pass with original report metrics.
dirty=frame.copy();dirty.loc[0,'NONLINEAR_STATE_HGB']+=10
try:v.analyze(report,dirty,bootstrap=10)
except ValueError as ex:assert 'RMSE_MISMATCH' in str(ex)
else:raise AssertionError('V41 failed closed on corrupted ledger')
# Any 2026 retrospective row is a hard violation.
dirty=pd.concat([frame,frame.iloc[:1].assign(Season=2026,Game_ID='2026-TEST')],ignore_index=True)
try:v.analyze(report,dirty,bootstrap=10)
except ValueError as ex:assert 'UNEXPECTED_SEASON_OR_2026' in str(ex)
else:raise AssertionError('V41 failed closed on 2026 result')
# Same generated UTC mock to establish atomic GCS lock semantics.
now=datetime(2026,10,20,17,0,tzinfo=timezone.utc)
good={
    'season':2026,'game_id':'2026-test-001','model':'NONLINEAR_STATE_HGB',
    'team':'A','opponent':'B','predicted_team_margin':5.2,
    'kickoff_utc':(now+timedelta(days=2)).isoformat(),
    'feature_as_of_utc':(now-timedelta(hours=1)).isoformat(),
    'immutable_model_artifact_uri':'gs://sharp-models/research/example/model.joblib',
    'model_artifact_sha256':'a'*64,
    'immutable_pregame_feature_snapshot_uri':'gs://sharp-models/research/example/features.json',
    'feature_snapshot_sha256':'b'*64}
assert lock.validate_and_prepare(good,clock=now)['status']=='LOCKED_PREGAME_SOURCE_VINTAGE_PENDING'
for mutation in [dict(actual_margin=3),dict(kickoff_utc=(now-timedelta(minutes=1)).isoformat()),
                 dict(feature_as_of_utc=(now+timedelta(days=3)).isoformat()),
                 dict(immutable_model_artifact_uri=''),dict(model_artifact_sha256='invalid')]:
    try:lock.validate_and_prepare({**good,**mutation},clock=now)
    except (ValueError,KeyError):pass
    else:raise AssertionError('Dangerous pregame lock accepted:'+str(mutation))
class FakeBlob:
    def __init__(self):self.text=None;self.count=0
    def upload_from_string(self,text,*,if_generation_match=None,content_type=None):
        assert if_generation_match==0
        if self.text is not None:raise RuntimeError('atomic existing lock')
        self.text=text;self.count+=1
class FakeBucket:
    def __init__(self):self.items={}
    def blob(self,p):
        if p not in self.items:self.items[p]=FakeBlob()
        return self.items[p]
class FakeClient:
    def __init__(self):self.buckets={}
    def bucket(self,name):
        if name not in self.buckets:self.buckets[name]=FakeBucket()
        return self.buckets[name]
fake=FakeClient()
saved=lock.lock_to_gcs(fake,good,clock=now)
assert saved['source_vintage_certified'] is False
try:lock.lock_to_gcs(fake,good,clock=now)
except RuntimeError as ex:assert 'atomic existing lock' in str(ex)
else:raise AssertionError('prospective lock overwritten')
print('V41_SELFTEST_PASS', 'rows',len(frame),'retro',result['retrospective_game_count'],
      'source_vintage',result['source_point_in_time_gate']['status'],
      'retrospective_rmse_gain',round(result['retrospective_paired_v3']['rmse_gain'],3))
