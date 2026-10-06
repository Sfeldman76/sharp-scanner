# NCAAF Engine V2.11.1 — Miner Bridge Boolean-Mask Hotfix


## V2.11.1 hotfix

V2.11.1 fixes a Heavy Research crash in `ncaaf_core_challenger_v2.py` inside `_publish_miner_intelligence_bridge`. When `Season` is stored as pandas nullable `Int64`, `.eq(year).to_numpy()` can produce an object array containing nullable booleans. NumPy rejects that object array as an index.

The bridge now uses `fillna(False).to_numpy(dtype=bool)` for both the year-validation mask and the 2023 discovery mask. This does not change any model logic, thresholds, features, authority, or chronology; it only makes the existing season-forward OOF bridge robust to nullable pandas dtypes. A regression guard for nullable `Int64` season masks was added to the built-in self-test.

V2.11 keeps **Production V1 frozen** and fixes the current-season ingestion contract: Prediction Tracker uses one season-to-date archive and a separate live/current-week feed. Both are now fetched and merged automatically.

## What changed

The NCAAF research pipeline now automatically retrieves the Prediction Tracker NCAA season CSVs from:

- `https://www.thepredictiontracker.com/ncaa2022.csv`
- `https://www.thepredictiontracker.com/ncaa2023.csv`
- `https://www.thepredictiontracker.com/ncaa2024.csv`
- `https://www.thepredictiontracker.com/ncaa2025.csv`
- 2026 season-to-date archive: `https://www.thepredictiontracker.com/ncaa2026.csv`
- 2026 active/current week: `https://www.thepredictiontracker.com/ncaapredictions.csv`
- live reference page: `https://www.thepredictiontracker.com/predncaa.php`

Completed seasons are cached in GCS after the first successful fetch. For 2026, **both** the season archive and the separate live/current-week CSV are fetched. The merged snapshot retains all older 2026 archive rows, then overlays the live row only when the same home/away matchup appears in both sources. The live feed is web-first with its own GCS fallback.

GCS keeps all three current-season representations:

- `current_archive.csv` — older/completed 2026 archive rows;
- `current_live.csv` — current-week ratings only;
- `current.csv` — merged archive + current week used by live dashboard attachment.

Every current-season row carries `source_kind` and `source_priority`. `LIVE_CURRENT` wins an overlap; prior archive rows are never discarded globally.

## Published META_MARGIN benchmark

V2.11 reproduces the published no-intercept five-system benchmark exactly:

- Dokter Entropy: `0.242406`
- Pi-Rate Bias: `0.281205`
- Keeper: `0.135398`
- ESPN Football Power Index: `0.163639`
- Pigskin Index: `0.114519`

`META_MARGIN` is emitted only when **all five** system predictions are finite for a game. Missing systems fail closed; the code does not replace a missing component with the site's overall prediction average and does not re-fit the published coefficients on our outcomes.

## Historical matching

The loader:

1. parses each season with flexible header matching;
2. normalizes external school names;
3. conservatively maps short tracker names to our historical canonical team names;
4. matches by season + home/away pair;
5. uses calendar date for duplicate same-season matchups when both sources expose a date;
6. leaves ambiguous/unresolved matches unmatched rather than guessing.

Historical META_MARGIN is converted to each team-row orientation and attached as:

- `_V210_PT_META_MARGIN_TEAM`
- `_V210_PT_META_EDGE_POINTS`
- `_V210_PT_META_SYSTEM_COUNT`
- `_V210_PT_PREDICTION_AVG_TEAM` (diagnostic only)
- `_V210_PT_ARCHIVE_OPEN_MARGIN_TEAM` (diagnostic only)
- `_V210_PT_META_MINUS_CORE_EDGE` when the OOF CORE bridge exists

## New Miner atoms

The **same existing Miner** may now test:

- `META_PT_EDGE_TEAM_2PLUS`
- `META_PT_EDGE_TEAM_3PLUS`
- `META_PT_EDGE_OPP_2PLUS`
- `META_PT_EDGE_OPP_3PLUS`
- `META_PT_EDGE_ABS_4PLUS`
- `META_PT_CORE_STRONG_AGREE`
- `META_PT_CORE_STRONG_CONFLICT`
- `META_PT_CORE_GAP_4PLUS`

These can combine with the existing Pathi, Big Al, conference, rivalry, H2H, timing, market, team-memory, OOF CORE, and specialist atoms.

Examples:

- `PATHI dog hook >3 + META_PT_EDGE_TEAM_3PLUS`
- `BIG AL CF2 + META_PT_CORE_STRONG_AGREE`
- `META_PT_CORE_STRONG_CONFLICT + same conference`

The existing discovery → FDR → 2024/2025 confirmation → dependency collapse → parent/child lineage → prospective workflow remains unchanged.

## External metamodel scorecard

Heavy Research now records per-season and pooled diagnostics for META_MARGIN versus:

- actual margin;
- opening market margin;
- season-forward OOF CORE when available;
- ATS results when META_MARGIN differs from the opening market by more than 3 points.

These diagnostics are research-only and do not alter CORE.

## Automatic current-season refresh

`NCAAF Production — Weekly Update` now refreshes the current Prediction Tracker season file automatically and stores a normalized snapshot in:

- `gs://sharp-models/research/ncaaf/external/prediction_tracker/current.csv`
- `gs://sharp-models/research/ncaaf/external/prediction_tracker/current_meta.json`

The NCAAF dashboard reads that cached current file and displays:

- **PT Meta Home Margin**
- **PT Meta vs Market**

Current META_MARGIN can also make already-confirmed `META_PT_*` mechanisms evaluable for research display, but it has **zero Bet Authority**.

## Authority guard

Any Miner mechanism containing:

- `CORE_OOF_*`
- `SPEC_*`
- `META_PT_*`

is research-only and cannot become a live Bet Authority family in V2.11.

Prediction Tracker failure is non-fatal. If the site is unavailable and no cached copy exists, the external layer simply reports unavailable and the existing NCAAF engine continues normally.

## What did NOT change

- frozen NCAAF Spread Production V1;
- frozen H2H Production V1;
- frozen Totals Production V1;
- CORE probability artifacts;
- 2% CORE edge + 2% live EV candidate gates;
- STRONG_VALIDATED Miner authority policy for eligible non-external mechanisms;
- Pathi / Big Al semantics;
- Utils / Move Master market backend;
- production ledger / physical-game dedupe;
- 2026 outcomes remain excluded from research selection;
- no automatic promotion.

The production/backend files remain unchanged from V2.9:

- `ncaaf_production_v1.py`
- `ncaaf_production_ledger_v1.py`
- `utils.py`

## Files to replace

If V2.11 is already deployed, replace only:

1. `ncaaf_core_challenger_v2.py`

For a clean V2.11.1 deployment from an older version, replace:

1. `ncaaf_core_challenger_v2.py`
2. `ncaaf_research_v2.py`
3. `sharp_line_dashboard.py`
4. `train_job.py`


The ZIP contains the unchanged production/backend files as a complete repository snapshot.

## After deployment

Run **NCAAF Research — Heavy Challenger Search** once.

Watch for:

- `[NCAAF-PT-SEASON]` — fetch/cache + five-system coverage by season;
- `[NCAAF-PT-MATCH]` — historical game/team matching coverage;
- `[NCAAF-PT-CONTRACT]` — final external bridge status;
- `[NCAAF-RV25-ATOM-BRIDGE]` — expanded Miner atom inventory;
- normal Miner mechanism + lineage diagnostics;
- `[NCAAF-RV25-CONTRACT]` — final research contract.

After that, normal **NCAAF Production — Weekly Update** keeps the current external file refreshed automatically. Do **not** run Production Publish for this change.

## V2.11 current-season safety contract

2026 archive rows are retained for prospective/intelligence history, while the separate live feed supplies the current week's latest system ratings. 2026 remains sealed from retrospective discovery/confirmation and does not change Production CORE or Bet Authority.

Expected refresh logs now include:

- `[NCAAF-PT-SEASON] season=2026 ...` — season archive
- `[NCAAF-PT-LIVE] season=2026 ...` — active-week CSV
- `[NCAAF-PT-CURRENT-MERGE] ...` — archive/live merge and overlap count
- `[NCAAF-WEEKLY-PT-REFRESH] ... archive_rows=... live_rows=... live_full_five=...`
