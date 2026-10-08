# NFL + NCAAF Source-Neutral PT System Authority V1

This package implements one policy in both sports: a Prediction Tracker-derived Miner system is judged by the same system qualification framework as the other system brains. PT gets no blanket model weight. A qualified PT-derived system may contribute one bounded confirmation/conflict family vote after CORE has already created a candidate. Correlated PT variants collapse to one external-ratings family; disagreement inside that family causes abstention.

## Replace these files

- `nfl_engine.py` — NFL Engine V3.14.0
- `ncaaf_research_v2.py` — NCAAF Research V2.20.0
- `train_job.py` — merged launcher with exact updated source tags
- `sharp_line_dashboard.py` — current dashboard with PT margin columns removed and policy text updated

Do not replace `ncaaf_production_v1.py`; this change uses the existing generic system-vote consumer and avoids rolling back newer production logic.

## Authority behavior

NFL Spread remains CORE-first. CORE must clear the frozen probability/price gate before any system matters. A qualified Miner, Pathi, Big Al, or PT-derived system family can then confirm or conflict. Two PT rules do not count as two votes: all PT spread rules collapse to `NFL_SPREAD_EXTERNAL_RATINGS_FAMILY`. NFL H2H and Totals remain model-only until those markets earn their own production betting gates.

NCAAF keeps its existing strong-validation gate: confirmation must pass the frozen 2024/2025 validation structure, with confirmation N >= 60 and hit rate >= 56%. PT-derived Miner systems now pass through that same source-neutral gate. All PT variants collapse to one `EXTERNAL_RATINGS_FAMILY` vote. Raw PT margin/model fields remain zero-authority and hidden from the production board.

2026 outcomes are not used to qualify or select systems in either sport. Current 2026 PT data may only determine whether an already-qualified PT-derived rule fires now.

## Run after deployment

1. Run `NFL Research — Heavy Challenger Search` once. This republishes the NFL system registry with qualified PT-derived systems included.
2. Run `NCAAF Research — Heavy Challenger Search` once. This republishes the NCAAF source-neutral Miner authority state.
3. Run the normal NFL/NCAAF production or weekly refresh so the current slate consumes the newly published qualified-system registries.

## Tests included

- `nfl_engine_self_test.json`: 33/33 bundled components compile/check, zero failures.
- `component_self_tests.json`: NFL System Lab and NFL Model Authority source-neutral/family-cap tests pass.
- `ncaaf_self_test.json`: strong PT-qualified system passes, weak PT system fails, CORE OOF bridge remains blocked, PT vote cap = 1.
- All four deployment `.py` files compile successfully.
