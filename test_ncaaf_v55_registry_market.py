"""Standalone unit tests for guarded V55 integrity repair."""
from __future__ import annotations
import ast
import datetime as dt
from pathlib import Path
import sys
import unittest
import pandas as pd

sys.path.insert(0,str(Path(__file__).parent))
import repair_ncaaf_v55_registry_market as fix


def minimal_source():
    return f'''\
V55_RESEARCH_MINER_POLICY = "{fix.MINER_OLD}"

def reader(report):
        if (report.get('miner_live_authority_policy') or {{}}).get('policy')!=V55_RESEARCH_MINER_POLICY:return None,'RESEARCH_POLICY_MISMATCH'
        mechanisms={{}}
        created=None
        for rec in report.get('items',[]):
            if rec.get('confirmation_pass') is not True or rec.get('live_authority_eligible') is not True:continue
        return {{'mechanisms':mechanisms,'created_utc':created,}}

def vote(registry, votes):
        for v in votes:
                if v.get('live_authority_policy')!=V55_RESEARCH_MINER_POLICY:reject('POLICY_VERSION_MISMATCH');continue

def _v55_read_enriched_pregame_overlays(bq_client, other=None):
    optional=list(dict.fromkeys(x for x in (*V55_RAW_OVERLAY_FIELDS,'Bookmaker','Book',
                         *sorted(qualified_fields)) if x in schema and x not in base_fields))

def display(out, idx, richrow, same_book, anchor, r):
    if True:
            out.at[idx,'V55_Raw_Market_Rich']=_v55_raw_rich(richrow)
'''


class TestPatch(unittest.TestCase):
    def test_transform(self):
        src=fix.transform(minimal_source())
        ast.parse(src)
        self.assertIn('if rec.get(\'live_authority_policy\')!=frozen_policy:continue',src)
        self.assertIn("if v.get('live_authority_policy')!=registry.get('policy')",src)
        self.assertIn(fix.MINER_NEW,src)
        self.assertIn("def _v55_rich_from_market_snapshots",src)
        self.assertIn("'Book','Open_Value'", src)
        self.assertEqual(src.count('V55_Raw_Market_Rich'),1)

    def test_abort_on_schema_change(self):
        with self.assertRaisesRegex(ValueError,'UNIQUE_ANCHOR_REQUIRED'):
            fix.transform('V55_RESEARCH_MINER_POLICY = "x"')

    def test_idempotence(self):
        with self.assertRaisesRegex(ValueError,'ALREADY_APPLIED'):
            fix.transform(fix.transform(minimal_source()))


class TestMarketRich(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope={'pd':pd,
               'V55_RAW_OVERLAY_TABLE':'sharplogger.sharp_data.moves_with_features_merged',
               '_price': lambda v: float(v) if v is not None else None,
               '_timestamp': lambda t: pd.Timestamp(t) if t is not None else pd.NaT}
        exec(fix.MARKET_HELPER,scope)
        cls.helper=staticmethod(scope['_v55_rich_from_market_snapshots'])
        cls.at=pd.Timestamp('2026-11-02 01:00:00+00:00')
        cls.kick=pd.Timestamp('2026-11-06 22:00:00+00:00')

    def history(self):
        vals=[]
        for m,spread in ((120,3.0),(60,5.0),(30,6.0),(0,7.5)):
            vals.append({'_snap':self.at-pd.Timedelta(minutes=m),
                         '_kick':self.kick,'Value':spread,'Open_Value':4.0,
                         'V55_Overlay_Actual_Source':'sharplogger.sharp_data.moves_with_features_merged'})
        return pd.DataFrame(vals)

    def test_real_lookback_deltas(self):
        x=self.helper(self.history(),self.at,7.5)
        for val in ('Open→Now +3.5','30m +1.5','60m +2.5','120m +4.5'):
            self.assertIn(val,x)

    def test_no_future_reuse(self):
        h=self.history()
        h=pd.concat([h,pd.DataFrame([{'_snap':self.at+pd.Timedelta(minutes=10),
                         '_kick':self.kick,'Value':-900,'Open_Value':4,
                         'V55_Overlay_Actual_Source':'sharplogger.sharp_data.moves_with_features_merged'}])],ignore_index=True)
        self.assertNotIn('-900',self.helper(h,self.at,7.5))

    def test_no_cross_source(self):
        h=self.history()
        h['V55_Overlay_Actual_Source']='unapproved.other_table'
        self.assertEqual(self.helper(h,self.at,7.5),'NOT_PUBLISHED')

    def test_no_stale_quote(self):
        h=self.history()
        self.assertEqual(self.helper(h,self.at+pd.Timedelta(minutes=16),7.5),'NOT_PUBLISHED')

    def test_no_wrong_execution_spread(self):
        self.assertEqual(self.helper(self.history(),self.at,-7.5),'NOT_PUBLISHED')

    def test_no_ambiguous_same_time(self):
        h=self.history()
        altered=h.iloc[-1].copy();altered['Value']=9.5
        h=pd.concat([h,pd.DataFrame([altered])],ignore_index=True)
        self.assertEqual(self.helper(h,self.at,7.5),'NOT_PUBLISHED')

    def test_no_interpolated_windows(self):
        h=self.history().iloc[[-1]].copy()
        self.assertEqual(self.helper(h,self.at,7.5),'Open→Now +3.5')


if __name__=='__main__':
    unittest.main(verbosity=2)
