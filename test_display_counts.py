import ast
import pathlib
import pandas as pd

S=pathlib.Path(__file__).with_name('sharp_line_dashboard.py').read_text()
node=next(n for n in ast.parse(S).body if isinstance(n,ast.FunctionDef) and n.name=='_ncaaf_v55_quote_coverage')
ns={}; exec(compile(ast.Module(body=[node],type_ignores=[]),'<helper>','exec'),ns)
coverage=ns['_ncaaf_v55_quote_coverage']


def test_exact_uploaded_metrics_and_stale_legacy_audit():
    actual=pd.read_csv('/mnt/data/2026-10-22T16-28_export.csv',keep_default_na=False)
    d=coverage(actual,{'upcoming_forecast_rows':113,'priced_games':39,'no_market_match':374})
    assert d=={'eligible':113,'priced':39,'unmatched':74,'outside':261,'uncertain':False}


def test_audit_claims_prices_but_no_verified_quote():
    d=coverage(pd.DataFrame({'Quote_State':['NO_VERIFIED_BACKEND_QUOTE'], 'Game_Start_UTC':['2026-10-24T12:00:00Z']}),
        {'upcoming_forecast_rows':1,'priced_games':5})
    assert d['priced']==0
    assert d['unmatched']==1


def test_fallback_without_eligible_audit():
    d=coverage(pd.DataFrame({'Quote_State':['FRESH_TWO_SIDED_BACKEND_PRICE_NO_SELECTED_TEAM'],
       'Game_Start_UTC':['2026-10-24T12:00:00Z']}),
       {'published_at_utc':'2026-10-22T15:00:00+00:00','priced_games':1})
    assert d['eligible']==1 and d['unmatched']==0


def test_backend_shadow_side_counts_as_verified_price():
    d=coverage(pd.DataFrame({'Quote_State':['FRESH_TWO_SIDED_BACKEND_PRICE_RESEARCH_SIDE'],
       'Game_Start_UTC':['2026-10-24T12:00:00Z']}),
       {'upcoming_forecast_rows':1,'priced_games':1})
    assert d['priced']==1 and d['unmatched']==0
