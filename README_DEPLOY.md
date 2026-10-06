# NCAAF Engine V2.13 — Prediction Tracker Name-Safe Header Contract

This is a narrow patch on top of V2.12. Production V1 remains frozen.

## Why this patch exists
Prediction Tracker CSV exports can use cryptic headers such as `linedokter` and `lineespn`, while the website displays human system names such as Dokter and ESPN FPI. V2.13 removes any possibility that a system prediction can be assigned by column position or fuzzy substring guessing.

## Name-safe contract
- Home/Road identities resolve only by exact normalized header names. There is no positional fallback.
- The five metamodel systems resolve only by exact human-readable names, exact self-identifying aliases, or an exact cryptic header that was previously verified.
- No system column is selected by position.
- No system column is selected by substring/fuzzy match.
- If a required system is missing or ambiguous, META_MARGIN fails closed for that source/row.
- A single source column cannot be assigned to two canonical systems.

## Automatic cryptic-header verification
When both the live CSV and live HTML table are available, V2.13 compares the actual prediction vectors game-by-game. A cryptic CSV header is associated with a named system only when one unique column reproduces the human-named HTML values across at least 8 games with >=98% agreement to 0.01 points. The verified mapping is cached at:

`gs://sharp-models/research/ncaaf/external/prediction_tracker/header_manifest.json`

That exact mapping can then be reused by historical archive CSVs when the same cryptic header exists. Current live HTML remains the preferred fallback when CSV identity is incomplete.

## New audit lines
The run now prints:

- `[NCAAF-PT-HEADER]` — complete raw header list for the source.
- `[NCAAF-PT-HEADER-VERIFY]` — value-validated cryptic-to-human mapping and any ambiguity.
- `[NCAAF-PT-COLUMN-MAP]` — exact canonical mapping for DOKTER, PI_RATE_BIAS, KEEPER, ESPN_FPI, and PIGSKIN_INDEX, plus missing/ambiguous fields.
- `[NCAAF-PT-LIVE] ... header_contract=FULL_FIVE_VERIFIED` when the current source is safe to use.

## Fetch order
On Heavy Research, the current live source is checked first so the verified header manifest can be refreshed before 2022–2025 archive CSVs are parsed.

## Authority
No change. Prediction Tracker remains research-only, has zero Production/Bet Authority, cannot mutate Production V1, and 2026 remains sealed from retrospective selection.

## Deployment from V2.12
Replace only:

`ncaaf_research_v2.py`

Redeploy and run **NCAAF Research — Heavy Challenger Search**.

## Tests
- All Python files compile.
- `ncaaf_core_challenger_v2.py` self-test PASS.
- `ncaaf_research_v2.py` self-test PASS.
- Header-order regression PASS: shuffling columns does not change system identity or META_MARGIN.
- Fuzzy near-miss regression PASS: `lineespn_extra` is rejected rather than treated as ESPN FPI.
- Synthetic live HTML↔CSV vector verification PASS for all five canonical systems.
