# NCAAF V2.3 — Protected CORE Challenger Search

This release adds a **research-only Spread CORE challenger lane** on top of the current NCAAF V2.2.1 production stack. It does not change the frozen Production V1 model, the V2.2.1 Bet Authority rules, the Miner authority gate, the prospective ledger, or any live bet.

## Why this exists

The current NCAAF Spread CORE is intentionally tiny and stable. The new lane asks a narrower question: can a different **compact fair-line recipe** beat that frozen CORE on untouched historical confirmation seasons?

The search is protected:

- 2022 = initial model fit
- 2023 = discovery / feature and recipe selection
- 2024 = untouched confirmation
- 2025 = untouched confirmation
- 2026+ = sealed; never used for feature selection, model choice, tuning, confirmation, ranking, promotion, or threshold choice

No production artifact can be written by this workflow.

## What it tests

The current frozen Spread CORE recipe is replayed as the benchmark:

- `Context_Intercept`
- `Diff_RawRecent3_Off_YPP`
- `B_RawSeason_GameAdj_Def_Rush_YPA`
- market-residual target
- Ridge + shallow HGB blend, 75% Ridge

The challenger searches the existing leakage-safe pregame feature universe and builds only compact recipes:

1. **FAIR_MARGIN_COMPACT** — market-blind model of actual game margin.
2. **RESIDUAL_COMPACT** — model of market opening-line error, converted back to a fair margin.
3. **HYBRID_COMPACT** — frozen discovery-selected blend of the market-blind and residual fair margins.

Search limits are deliberately small:

- max 6 non-intercept features
- max 2 features from one feature family
- discovery screen limited before greedy admission
- Ridge alpha grid: 12 / 24 / 48
- Ridge/HGB blend grid: 50% / 75% / 100% Ridge
- hybrid weights: 25% / 50% / 75% market-blind fair margin

2024 and 2025 do not reselect any feature, model, alpha, blend, or hybrid weight.

## Confirmation output

For the incumbent and each challenger the report publishes:

- RMSE / MAE versus actual margin
- improvement versus the opening market
- rate the model is closer to the result than the opening market
- ATS-direction diagnostics at 1 / 2 / 3 / 4 / 5 point fair-line disagreement bands
- 2024 and 2025 results separately
- pooled paired-bootstrap RMSE and MAE gain versus the incumbent

Possible research states:

- `NO_IMPROVEMENT`
- `MIXED`
- `PROMOTION_ELIGIBLE_RESEARCH`
- `STRONG_CHALLENGER`

Even `STRONG_CHALLENGER` receives **zero production authority**. The next step would be to freeze that exact recipe into a prospective 2026 paired shadow. Production promotion would remain a separate explicit decision.

## Deploy

Starting from NCAAF V2.2.1, replace/add only these files:

1. **ADD** `ncaaf_core_challenger_v1.py`
2. **REPLACE** `train_job.py`
3. **REPLACE** `sharp_line_dashboard.py`

The included `ncaaf_production_v1.py`, `ncaaf_production_ledger_v1.py`, `ncaaf_research_v2.py`, and `utils.py` are unchanged from V2.2.1 and are included only to keep this archive self-contained.

After redeploying, choose:

**NCAAF CORE — Challenger Search**

Run it once and review the Cloud Run log or open **CORE Challenger — 2024/25 Confirmation Results** in the NCAAF dashboard.

## Do not run

Do **not** run `NCAAF Production — Publish Approved Contract` for this research test. No Heavy/legacy route is required.
