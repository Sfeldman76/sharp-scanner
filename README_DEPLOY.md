# NFL Live PT System Evaluation Fix V3.14.1

Replace **only `nfl_engine.py`** for this fix, then redeploy the NFL dashboard/service. The weekly/heavy job can use the same engine file on its next deployment; no retraining is required just to activate the live trigger fix.

## What was wrong
The six qualified PT-derived spread systems were valid in the frozen research registry, but the live evaluator supported only a small subset of their non-PT condition atoms. They therefore appeared as `fail-closed / not evaluable` before their PT signals could vote.

## Fixed live atoms
- `LAST_GAME_FAVORITE`
- `OPP_LAST_GAME_DOG`
- `LAST_GAME_DIVISION`
- `DEFENSE_IMPROVING_2`
- `TURNOVER_POS_LAST3`

Existing atoms such as `OFF_ATS_COVER_7_PLUS` and `OPP_OFF_SU_LOSS` remain unchanged.

The live history query now also carries division context and exact turnover context. If `Postgame_Turnovers` is unavailable in the historical-core view, the engine merges it from the canonical `nfl_historical_game_side_raw` table and reconstructs turnover margin from the two game-side rows.

## Safety / authority unchanged
- CORE remains the only prediction authority.
- PT raw model weight remains `0.0`.
- PT-derived systems can only support/conflict after qualification.
- All PT-derived systems still collapse to **one** `NFL_SPREAD_EXTERNAL_RATINGS_FAMILY` vote.
- 2026 is trigger context only; it is not used to qualify/reselect systems.

## Validation
- Consolidated engine preflight: PASS, 33/33 components compiled.
- Model Authority self-test: PASS.
- Synthetic live-parity test: all six previously fail-closed PT systems are evaluable and fire when their exact frozen conditions are satisfied.
