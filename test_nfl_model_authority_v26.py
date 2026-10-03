"""Offline contract tests for NFL Model Authority V2.6."""
import copy
import pandas as pd
import nfl_model_authority_v26 as ma


def replay_fixture(confirm_spread_win_rate=0.60):
    rows=[]
    for sy in (2021,2022,2023,2024,2025):
        n=30 if sy <= 2023 else 20
        spread_rate=0.60 if sy <= 2023 else confirm_spread_win_rate
        sw=int(round(n*spread_rate))
        tw=int(round(n*0.40))
        hw=int(round(n*0.50))
        for i in range(n):
            rows.append({
                "season":sy,
                "frozen_fair_margin":4.0,
                "close_spread":0.0,
                "actual_margin":1.0 if i < sw else -1.0,
                "frozen_fair_total":50.0,
                "close_total":46.0,
                "actual_total":47.0 if i < tw else 45.0,
                "frozen_home_win_probability":0.65,
                "close_novig_home_probability":0.50,
                "home_win_label":1.0 if i < hw else 0.0,
                "home_close_ml":-110.0,
                "away_close_ml":-110.0,
            })
    return pd.DataFrame(rows)


def test_discovery_freeze_then_confirmation():
    c=ma.build_contract(replay_fixture(), log_func=lambda *_:None)
    # 2/3/4-point spread thresholds all clear this synthetic sample, but the
    # contract must freeze the first/smallest discovery pass rather than choose
    # a threshold using confirmation performance.
    assert c["markets"]["SPREADS"]["threshold"] == 2.0
    assert c["markets"]["SPREADS"]["production_authority"] is True
    assert c["markets"]["TOTALS"]["production_authority"] is False
    assert c["markets"]["H2H"]["production_authority"] is False
    assert c["year_2026_queried"] is False


def test_confirmation_can_close_authority_without_reselection():
    c=ma.build_contract(replay_fixture(confirm_spread_win_rate=0.40), log_func=lambda *_:None)
    assert c["markets"]["SPREADS"]["threshold"] == 2.0
    assert c["markets"]["SPREADS"]["production_authority"] is False
    assert c["markets"]["SPREADS"]["status"] == "MODEL_ONLY_CONFIRMATION_NOT_PASSED"


def test_2026_is_sealed():
    d=replay_fixture()
    x=d.iloc[[0]].copy(); x["season"]=2026
    d=pd.concat([d,x],ignore_index=True)
    try:
        ma.build_contract(d, log_func=lambda *_:None)
    except RuntimeError as exc:
        assert "2026_MUST_REMAIN_SEALED" in str(exc)
    else:
        raise AssertionError("2026 row must hard-fail contract freeze")


def test_shadow_conflict_cannot_change_model_bet():
    contract={"markets":{
        "SPREADS":{"production_authority":True,"threshold":3.0,"status":"MODEL_BET_AUTHORITY_FROZEN"},
        "H2H":{"production_authority":False,"threshold":None,"status":"MODEL_ONLY"},
        "TOTALS":{"production_authority":False,"threshold":None,"status":"MODEL_ONLY"},
    }}
    row={
        "prediction_pair_id":"p1","market":"SPREADS","action":"MODEL ONLY",
        "raw_model_edge":4.5,"selected_price":-110.0,"model_direction":1,
        "selected":"HOME","market_value":-3.0,"model_value":7.5,
        "market_move_toward_model":-0.5,
        "edge_votes":[{"label":"SYSTEM_X","direction":-1}],
        "system_labels":["SYSTEM_X"],"stat_selector_support":[],
        "shadow_edge_action":"PASS — CONFLICT","shadow_edge_decision":"CONFLICT",
    }
    r=ma.apply_model_authority_rows([row],contract)[0]
    assert r["action"] == "BET"
    assert r["betting_decision_authority"] is True
    assert r["system_state"] == "CONFLICT"
    assert r["market_state"] == "CONFLICT"
    assert r["shadow_edge_action"] == "PASS — CONFLICT"


def test_authoritative_quote_rows_do_not_require_shadow_engine():
    old=ma._authoritative_consensus_market
    try:
        ma._authoritative_consensus_market=lambda client,now:{
            (pd.Timestamp("2026-10-04T17:00:00Z").round("s").isoformat(),ma._norm("Home Team"),ma._norm("Away Team")):{
                "current_home_spread":-3.0,"open_home_spread":-2.5,
                "current_home_spread_odds":-110.0,"current_away_spread_odds":-110.0,
                "current_total":44.0,"open_total":44.5,"current_over_odds":-105.0,"current_under_odds":-115.0,
                "current_home_novig_probability":0.55,"open_home_novig_probability":0.53,
                "current_home_ml":-130.0,"current_away_ml":115.0,
            }
        }
        pred=[{"prediction_pair_id":"p1","game_start":"2026-10-04T17:00:00Z","home_team":"Home Team","away_team":"Away Team",
               "champion_fair_margin":7.0,"champion_fair_total":47.0,"champion_home_win_probability":0.62}]
        base=ma.build_authoritative_market_rows(bq_client=None,prediction_rows=pred,now=pd.Timestamp("2026-10-03T16:00:00Z"))
        assert len(base)==3
        sp=next(x for x in base if x["market"]=="SPREADS")
        assert sp["selected"]=="Home Team" and sp["raw_model_edge"]==4.0
        merged=ma.merge_shadow_evidence(base,[])
        assert all(x["shadow_evidence_status"]=="UNAVAILABLE" for x in merged)
    finally:
        ma._authoritative_consensus_market=old



def test_authoritative_consensus_never_fabricates_minus110():
    old=ma.benchmark._fetch_market_rows
    try:
        ma.benchmark._fetch_market_rows=lambda client,now:pd.DataFrame([
            {"market":"spreads","outcome":"Home Team","value":-3.0,"odds":float("nan"),"bookmaker":"Book A",
             "game_start":pd.Timestamp("2026-10-04T17:00:00Z"),"snapshot_ts":pd.Timestamp("2026-10-03T15:00:00Z"),
             "home_team":"Home Team","away_team":"Away Team","home_key":ma._norm("Home Team"),"away_key":ma._norm("Away Team"),
             "outcome_key":ma._norm("Home Team"),"market_norm":"spreads"},
            {"market":"spreads","outcome":"Away Team","value":3.0,"odds":float("nan"),"bookmaker":"Book A",
             "game_start":pd.Timestamp("2026-10-04T17:00:00Z"),"snapshot_ts":pd.Timestamp("2026-10-03T15:00:00Z"),
             "home_team":"Home Team","away_team":"Away Team","home_key":ma._norm("Home Team"),"away_key":ma._norm("Away Team"),
             "outcome_key":ma._norm("Away Team"),"market_norm":"spreads"},
        ])
        q=ma._authoritative_consensus_market(None,pd.Timestamp("2026-10-03T16:00:00Z"))
        rec=next(iter(q.values()))
        assert pd.isna(rec["current_home_spread_odds"])
        assert pd.isna(rec["current_away_spread_odds"])
    finally:
        ma.benchmark._fetch_market_rows=old


def test_execution_price_gate_fails_closed():
    contract={"markets":{"SPREADS":{"production_authority":True,"threshold":3.0,"status":"MODEL_BET_AUTHORITY_FROZEN"}}}
    base={"market":"SPREADS","action":"MODEL ONLY","raw_model_edge":4.0,"model_direction":1,"edge_votes":[],"stat_selector_support":[],"market_move_toward_model":0.0}
    missing=ma.apply_model_authority_rows([{**base,"selected_price":float("nan")}],contract)[0]
    expensive=ma.apply_model_authority_rows([{**base,"selected_price":-115.0}],contract)[0]
    assert missing["action"] == "EDGE — NO EXEC QUOTE"
    assert expensive["action"] == "EDGE — NO EXEC QUOTE"

def run_all():
    tests=[v for k,v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in tests: fn()
    print({"status":"PASS","tests":len(tests),"source_tag":ma.SOURCE_TAG})


if __name__ == "__main__":
    run_all()
