"""Offline historical-only promotion regression; no cloud or live service access."""
import json
import pt_promotion_v1 as gov
class FakeBlob:
    def __init__(self, db, name):self.db,self.name=db,name
    def exists(self):return self.name in self.db
    def download_as_bytes(self):return self.db[self.name].encode()
    def upload_from_string(self,val,content_type=None,if_generation_match=None):
        if if_generation_match==0 and self.exists():
            class PreconditionFailed(Exception):pass
            raise PreconditionFailed()
        self.db[self.name]=val.decode() if isinstance(val,bytes) else str(val)
class FakeBucket:
    def __init__(self,db):self.db=db
    def blob(self,name):return FakeBlob(self.db,name)
    def list_blobs(self,prefix):return [FakeBlob(self.db,k) for k in sorted(self.db) if k.startswith(prefix)]
class FakeStorage:
    def __init__(self):self.db={}
    def bucket(self,bucket):return FakeBucket(self.db)

sc=FakeStorage(); bucket='test'
selection={'status':'FIVE_SELECTED_DEVELOPMENT_ONLY','sources':['a','b','c','d','e'],'source_providers':{x:x for x in 'abcde'},'selection_sha256':'LOCKED','development_years':[2017,2018,2019,2020,2021,2022]}
assert gov.lock_selection(sc,bucket,'NFL','SPREADS',selection)['status']=='FIVE_LOCKED'
evidence={'periods_disjoint':True,'asof_max_year':2025,'holdout_years':[2024,2025],'holdout':{'n':562,'incumbent_mae':10.2,'challenger_mae':9.9584},'holdout_by_year':{'2024':{'n':284,'gain_points':.18},'2025':{'n':278,'gain_points':.30}},'paired_bootstrap_ci95':[.056, .429],'selection_sha256':'LOCKED'}
assert gov.record_historical(sc,bucket,'NFL','SPREADS',evidence)['pass'] is True
# No pending or settled games, no waiting 4 weeks.
assert not any('/settled/' in k for k in sc.db)
decision=gov.review(sc,bucket,'NFL','SPREADS');assert decision['status']=='PROMOTED',decision
assert decision['qualification_basis']=='PAIRED_CHRONOLOGICAL_HISTORICAL'
state=gov.read_state(sc,bucket,'NFL','SPREADS')
assert state['historical']['n']==562 and state['prospective']['status']=='MONITOR_ONLY_NOT_A_REQUIREMENT'
assert state['pt_family_vote_cap']==1 and state['core_model_unchanged']
assert gov.promoted_vote(state,5.0,-2.0,5,'LOCKED')==1
assert gov.promoted_vote(state,-5.0,-2.0,5,'LOCKED')==-1
assert gov.promoted_vote(state,5.0,-2.0,2,'LOCKED')==0
assert gov.promoted_vote(state,5.0,-2.0,5,'DIFFERENT')==0
assert gov.review(sc,bucket,'NFL','SPREADS')['status']=='ALREADY_PROMOTED'
# Missing/bad identities block promotion.
s2=FakeStorage();gov.lock_selection(s2,bucket,'NCAAF','SPREADS',selection)
assert gov.record_historical(s2,bucket,'NCAAF','SPREADS',{**evidence,'selection_sha256':'BAD'})['pass'] is False
assert gov.review(s2,bucket,'NCAAF','SPREADS')['status']=='KEEP_INCUMBENT'
# Better on overall MAE, worse in one year => hold.
s3=FakeStorage();gov.lock_selection(s3,bucket,'NCAAF','SPREADS',selection)
bad={**evidence,'holdout_by_year':{'2024':{'n':284,'gain_points':-.02},'2025':{'n':278,'gain_points':.50}}}
assert gov.record_historical(s3,bucket,'NCAAF','SPREADS',bad)['pass'] is False
assert gov.review(s3,bucket,'NCAAF','SPREADS')['status']=='KEEP_INCUMBENT'
# Too weak even if both years positive.
s4=FakeStorage();gov.lock_selection(s4,bucket,'NFL','TOTALS',selection)
weak={**evidence,'holdout':{'n':562,'incumbent_mae':10.25,'challenger_mae':10.32},'paired_bootstrap_ci95':[-.23,.08]}
assert not gov.record_historical(s4,bucket,'NFL','TOTALS',weak)['pass']
assert gov.review(s4,bucket,'NFL','TOTALS')['status']=='KEEP_INCUMBENT'
# Explicit tampering with current_historical prevents promotion.
s5=FakeStorage();gov.lock_selection(s5,bucket,'NFL','SPREADS',selection);gov.record_historical(s5,bucket,'NFL','SPREADS',evidence)
key='research/pt_promotion_v1/nfl/spreads/current_historical.json'
corrupted=json.loads(s5.db[key]);corrupted['historical']['holdout']['challenger_mae']=1
s5.db[key]=json.dumps(corrupted)
assert gov.review(s5,bucket,'NFL','SPREADS')['reason']=='HISTORICAL_EVIDENCE_HASH_MISMATCH'
print('PASS historical-only promotion, missing 2026 games, incumbent retention, tamper detection, one PT vote')
