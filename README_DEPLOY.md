# NCAAF V2.5.2 — Physical-Game Dedupe + Authority Diagnostics Fix

This is a narrow runtime/UI patch on top of V2.5.1. It fixes the two issues visible in the current NCAAF Weekly tables without changing CORE, model probabilities, research qualification, Miner thresholds, or Bet Authority.

## Fix 1 — duplicate physical games

The live feed can occasionally carry the same matchup with two kickoff-time variants. The previous production identity used home + away + UTC kickoff hour, so a stale schedule time could make one real game appear twice.

V2.5.2 changes only physical-game identity handling:

- groups the same directional home/away matchup when kickoff variants are within 18 hours;
- chooses one canonical kickoff using the freshest snapshot first, then row coverage, then the later kickoff as a deterministic tie-breaker;
- preserves the original kickoff in `_prod_source_game_start`;
- stores the canonical kickoff in `_prod_game_start`;
- assigns all variants to one `_prod_game_id`;
- uses the canonical kickoff for the selected production row and prospective-ledger lock timing;
- leaves normal one-kickoff games on the same legacy `home_away_UTC-hour` production ID, minimizing ledger identity churn;
- exposes a dashboard caption when schedule-time conflicts were normalized.

The current exported board had 61 rows but only 59 unique matchups. The duplicated matchups were Arizona State–Hawaii and Indiana–Ohio State. This patch collapses those schedule variants before the board is built.

## Fix 2 — blank Miner / Pathi / Big Al detail fields

The compact board was correctly using the authoritative support/conflict fields, but the expanded "Why Bet Authority made each decision" table relied mostly on optional presentation columns (`NCAAF_RV2_System_Summary`, `Pathi_Active_Text`, `BigAl_Active_Text`). That made the detail fields look blank even when Bet Authority had actually consumed a frozen Miner or Pathi vote.

V2.5.2 makes the detail table derive diagnostics from the same authoritative fields used by Bet Authority:

- `_system_support_sources`
- `_system_support_families`
- `_system_conflict_sources`
- `_system_conflict_families`

It still shows the richer raw RV2 / Pathi / Big Al text when available, but now falls back to explicit normalized states such as:

- `SUPPORT: MINER_DOG_3_7`
- `CONFLICT: MINER_DOG_3_7`
- `SUPPORT: PATHI_CROSSED_KEY_AWAY_FADE`
- `CONFLICT: PATHI_CROSSED_KEY_AWAY_FADE`

The current exported detail table contained 17 rows with actual Miner support/conflict and 19 rows with actual Pathi support/conflict, while all of the old display-only diagnostic columns were blank. The new helper surfaces those real authority states.

## What did NOT change

- NCAAF Spread CORE: unchanged
- H2H CORE: unchanged
- Totals CORE: unchanged
- CORE 2% edge + 2% EV candidate gates: unchanged
- STAT authority: unchanged
- Pathi rules: unchanged
- Big Al rules: unchanged
- Miner qualification: unchanged
- `STRONG_VALIDATED` live-authority gate: unchanged
- Bet Authority: unchanged
- Heavy Research: unchanged
- 2026 remains sealed from research selection
- Production contract source tag: unchanged
- Automatic model/policy promotion: disabled

## Deploy

From V2.5.1, only two runtime files need to change:

1. `ncaaf_production_v1.py`
2. `sharp_line_dashboard.py`

The full ZIP includes the synchronized unchanged files as well.

After deployment, run **NCAAF Production — Weekly Update** once so the current board is rebuilt with canonical physical-game identity. You do **not** need to rerun Heavy Research for this patch.

The normal background scanner will then continue using the same dedupe logic for prospective captures.
