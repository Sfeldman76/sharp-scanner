# NCAAF Engine V2.7 — Existing-Feed Coverage Completion + System Attribution

This release extends the protected NCAAF research path while leaving **Production V1 frozen**. It does not create another global model or another parallel market backend. The purpose is to finish representation of useful data that already exists, test where it adds independent value, and make system evidence easier to audit.

## What changed

### 1. Existing-feed coverage completion (CORE Challenger V2.2)

Heavy Research now audits every compatible historical feed against a standard representation contract:

- season-to-date;
- recent 3;
- recent 5 when the protected source actually contains it;
- opponent-adjusted where the source and football meaning support it;
- team vs opponent;
- offense-vs-opponent-defense matchup differential where semantically valid;
- recent-vs-season trend.

The implementation is deliberately conservative. It may derive algebraic differences/trends only from already leakage-safe pregame inputs. It **does not manufacture Recent-5 history, opponent-adjusted history, or missing advanced data from postgame box scores**.

A coverage registry is published with COMPLETE_CORE / COMPLETE_R5 / PARTIAL status for each feed. The research UI shows the audit and the safely derived fields.

### 2. Feed-coverage specialist instead of feature dumping

Coverage-completed fields enter the existing specialist/conditional-attribution framework. They are not automatically dumped into Production CORE.

The FEED_COVERAGE specialist is discovery-selected and then evaluated through the same protected 2024/2025 confirmation framework as the other research specialists.

### 3. Incremental-value and ablation diagnostics

For selected coverage features Heavy Research now publishes:

- one-feature-at-a-time incremental RMSE/MAE value vs the frozen incumbent;
- separate 2023 discovery, 2024 confirmation, and 2025 confirmation results;
- remove-one-feature ablation of the frozen FEED_COVERAGE recipe.

These diagnostics have zero production authority.

### 4. Conditional-attribution neighborhood diagnostics

Conditional specialist regimes retain the existing 2023 selection -> 2024/2025 confirmation contract, but now also publish nearby diagnostics:

- 15%, 25%, and 35% specialist blend weights;
- nearby 1 / 2 / 3 point edge floors;
- whether all nearby blend weights retain positive RMSE value.

This is a robustness check only. It does not retune the selected regime after seeing confirmation data.

### 5. Miner parent/child lineage + incremental attribution (Research V2.3)

Miner systems now explicitly identify nested rule lineage. For a child system that adds conditions to a broader parent, the report includes:

- parent system ID;
- lineage depth;
- added condition(s);
- child discovery performance vs parent-only performance;
- child 2024/2025 confirmation performance vs parent-only performance;
- nested-total-variant flag.

This prevents a tighter version of the same mechanism from being mistaken for an independent discovery merely because it has a new system ID.

### 6. Published Pathi / Big Al NCAAF W-L table

Research V2.3 directly grades directional NCAAF Pathi and Big Al flags that already exist in the historical frame. The dashboard now shows discovery and 2024-2025 confirmation W-L, hit rate, flat -110 ROI reference, and a shrunk hit-rate diagnostic.

Included Pathi families cover the directional football spread bands/key-number systems already in the codebase, including dog below 3, dog on/above 7, dog 10+, hook bands, line moves across key numbers, and the total-spread-gap <= 10 condition. Big Al CF1/CF2/CF3 and the CF2 away tightener are included when historical fields are available.

Only directional recommended-side flags are graded. Symmetric key events and ambiguous screens remain context and are not assigned artificial system W-L records.

### 7. Evidence tiers and shrinkage diagnostics

The Miner report adds research-only evidence labels:

- STRONG_VALIDATED — unchanged live-authority gate;
- PROMISING_SHADOW;
- VALIDATED_SHADOW;
- WATCHLIST;
- RESEARCH_SHADOW.

The new labels do **not** weaken the existing live authority requirement. The frozen live gate remains both-year confirmation plus pooled confirmation N >= 60 and hit rate >= 56%.

Direct system-result tables also publish Beta(15,15) shrunk hit rates and Wilson lower bounds so small samples are not read at face value.

### 8. Miner authority threshold-neighborhood diagnostic

The report shows how many confirmed mechanism families would clear neighboring combinations of:

- confirmation N = 50 / 60 / 70;
- hit-rate floor = 55% / 56% / 57%.

The existing 60 / 56% gate remains frozen. This table is diagnostic only and cannot retune the threshold.

### 9. Directional market-residual attribution

Spread and totals mechanism attribution now records the magnitude by which the recommended side beat or missed the opening market number, separately for discovery and confirmation. This supplements W-L with directional market error rather than treating every cover by the same amount.

## Existing research preserved

This release keeps the useful V2.6/V2.2 machinery rather than rebuilding it under new names:

- PLAY_ON vs FADE evaluation;
- discovery FDR controls;
- 2022-2023 discovery only;
- both 2024 and 2025 confirmation;
- dependency/mechanism collapse;
- conference and conference-pair context;
- rivalry/revenge/H2H context;
- 1/2/3-game sequence and team-memory atoms;
- role changes and timing/rest context;
- market movement, sharp/soft divergence, key crossing and persistence inputs when historically available;
- price-aware H2H ROI / market-residual / price-band validation;
- 2026+ sealed prospective tracking;
- no opposing automatic production side selection.

## What intentionally did NOT change

- frozen NCAAF Spread Production V1 model;
- frozen H2H Production V1 model;
- frozen Totals Production V1 model;
- Production V1 probability artifact;
- frozen STAT selector;
- 2% CORE edge + 2% live EV candidate gates;
- STRONG_VALIDATED Miner live-authority threshold;
- Bet Authority architecture;
- Utils / Move Master as the live market backend;
- production ledger identity / physical-game dedupe;
- 2026 outcomes remain excluded from research selection;
- no automatic promotion.

The following production/backend files are byte-for-byte unchanged from V2.6 in this package:

- `ncaaf_production_v1.py`
- `ncaaf_production_ledger_v1.py`
- `utils.py`

## Files to replace

Replace these four files from V2.6:

1. `ncaaf_core_challenger_v2.py`
2. `ncaaf_research_v2.py`
3. `sharp_line_dashboard.py`
4. `train_job.py`

The ZIP also includes the synchronized unchanged production/backend files so it can be used as a full repository snapshot.

## After deployment

Run **NCAAF Research — Heavy Challenger Search** once.

That run will regenerate:

- the coverage registry;
- coverage incremental/ablation results;
- conditional attribution + neighborhood diagnostics;
- Miner lineage diagnostics;
- published Pathi / Big Al W-L tables;
- Miner threshold-neighborhood diagnostics;
- the normal V2 research artifacts.

You do **not** need to run Production Publish, and this research release should not promote or mutate Production V1.

Normal weekly operation remains:

1. update 2026 YTD statistics;
2. validate the uploader;
3. run **NCAAF Production — Weekly Update**.
