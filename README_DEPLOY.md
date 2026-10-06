# NCAAF V2.6 — Conditional Specialist Attribution + Advanced-Data Readiness

This release builds on V2.5.2. Production V1 remains frozen. The change is on the protected Heavy Research path: instead of continuing to search only for a globally larger CORE, the research layer now asks where an independent specialist adds incremental value to the frozen CORE.

## What changed

### 1. Conditional specialist attribution

Heavy Research now evaluates these independent specialist branches:

- POWER_SOS_CONTEXT
- STRUCTURED_STATS
- MATCHUP_CONTEXT (new standalone matchup/trend specialist)
- ADVANCED_STATS (only when real historical advanced fields exist)
- EXPERT_COMBINED
- PROGRAM_HIERARCHY

The global challenger comparison is retained, but a specialist no longer has to beat CORE on every game to be useful. A separate attribution layer tests pregame regimes such as:

- cross-conference vs same-conference
- early season vs mature season
- large strength-of-schedule gap
- large power-vs-market disagreement
- large conference-strength gap
- big spreads
- CORE and specialist strong agreement
- CORE and specialist strong conflict
- large specialist-vs-CORE divergence

Regime numeric cut points are frozen from 2023 discovery only. At most two regimes per specialist advance to confirmation. They are then evaluated independently in 2024 and 2025.

A regime can become `QUALIFIED_PROSPECTIVE_SHADOW` only when:

- each confirmation season has at least 40 games;
- a fixed 25% specialist / 75% incumbent blend improves RMSE in BOTH 2024 and 2025;
- pooled RMSE improves;
- pooled MAE is non-inferior;
- the paired-bootstrap 95% lower bound for RMSE improvement is above zero.

Even then it has:

- production_authority = 0
- bet_authority_vote = false
- automatic_promotion = false

So this release cannot change a live wager or rewrite CORE.

### 2. Standalone matchup specialist

V2.5 created matchup/trend interactions, but they were primarily consumed inside the broad expert model. V2.6 now screens and tunes a separate `MATCHUP_CONTEXT` specialist so passing/rushing mismatches, recent-vs-season trends, and matchup interactions can be judged independently.

### 3. Advanced-data readiness contract

Heavy Research now publishes explicit historical coverage for genuinely new data families rather than silently substituting box-score proxies:

- EPA/play
- success rate
- explosiveness
- havoc
- line yards
- stuff rate
- power success
- field position
- drive efficiency
- sack / pressure rate
- early-down efficiency
- passing-down efficiency
- QB efficiency
- returning production / roster continuity
- recruiting / talent

Each family is `READY`, `PARTIAL`, or `MISSING`. Missing sources are never fabricated. Once real leakage-safe historical columns are populated, the existing Heavy Research path can test them automatically.

### 4. Dashboard research UI

The CORE Challenger panel now adds:

- conditional specialist attribution table;
- 2024 and 2025 regime-specific gains;
- pooled gain and bootstrap lower bound;
- SUPPORT_SHADOW / CAUTION_SHADOW / CONTEXT_SHADOW role;
- count of qualified prospective-shadow regimes;
- advanced-data readiness table.

## What did NOT change

- frozen NCAAF Spread CORE
- frozen H2H CORE
- frozen Totals CORE
- current production probability artifact
- 2% CORE edge + 2% live EV candidate gates
- frozen STAT selector
- Pathi
- Big Al
- Miner qualification / STRONG_VALIDATED gate
- Bet Authority
- physical-game dedupe
- prospective ledger identity
- 2026 outcomes remain sealed from research selection
- no automatic promotion

## Deploy

From V2.5.2 replace:

1. `ncaaf_core_challenger_v2.py`
2. `train_job.py`
3. `sharp_line_dashboard.py`

The ZIP also contains the synchronized unchanged NCAAF files for a full deployment snapshot.

After deployment run **NCAAF Research — Heavy Challenger Search** once. That creates the new conditional-attribution and advanced-data-readiness report. You do not need to run Production Publish, and you do not need to rerun Weekly solely because of this research patch.

Normal weekly operation remains:

1. update 2026 YTD statistics;
2. validate the uploader;
3. run **NCAAF Production — Weekly Update**.
