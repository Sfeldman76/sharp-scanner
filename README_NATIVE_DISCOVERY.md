# NCAAF CFBD Native Discovery V1 — Independent Research Foundation

**Status:** Research-only implementation + offline synthetic verification; no live CFBD access, no live warehouse audit, no credible new predictive discoveries yet.

This is **NOT** another way to reproduce frozen Core or Big Al rules. CollegeFootballData's original football data is a separate scientific source, and the first question is what its source fields can teach us independently about expected points, variance, and matchup interactions.

## Distinguish what has been proved

**Established from October 18 Cloud Run logs:** 2013–25 research game outcomes span 10,396 scored games. The strictly reconstructed richer CFBD feature path covered only 2022–25 and 7,320 game-side observations. The existing 2/5/full window ablation used only 36 shared hierarchical predictors, so it did not measure the predictive value of all available CFBD record families. The earlier football-only Ridge was worse than the hierarchical football-only model, and neither consistently beat the opener. This is evidence of a *data architecture / representation* limitation, not proof that longer history must mathematically win.

**Established from official CFBD documentation:** /games/teams and /games/players return game-specific team/player box statistics; /drives returns real possessions; /plays returns plays; /stats/game/advanced contains PPA, explosiveness, success and offense/defense splits. Historical coverage extends back earlier than 2013 for most of these but varies by field/game/period.

**Not established:** That the user's warehouse actually has complete 2013–21 box/drives/play payloads; their historical quality, identity alignment, and as-of metric vintage; that the native research improves genuine out-of-sample RMSE or prices; that Core's historical predictors are leakage free. Run the attached **read-only BigQuery** audit and fetch/pilot source families before drawing these conclusions.

## Architecture: source-first, NOT a legacy feature adapter

1. **Bronze — immutable source family responses.** Never discard drives, plays, nested player statistics, individual team-box categories, or provenance before feature learning. For each response preserve original JSON, endpoint, request parameters, retrieval timestamp, payload hash, CFBD availability range, and `vintage_verified=False` unless independently demonstrated.
2. **Silver — canonical CFBD game IDs, two known teams, separate game/team/drive/play granularity.** Resolve identity only within a known game; never silently fuzzy match other teams. Reject duplicate game/team rows, broken scores, invalid chronology, and contradictory snapshots.
3. **Gold — previous-game-only team and matchup features.** Fold each team's completed prior-game performances into offense/defense, field position, efficiency, finishing, explosive and possession histories. Use only observations whose information is available before kickoff; absent a verified end time, this V1 introduces a conservative 24-hour embargo after game kickoff. Missing isn't zero. Target score is kept in its own dataset.
4. **Research-only year-forward OOF.** Separate margin, totals and H2H. Compare predeclared source-family ablations and 2-,5-,full-history fitting windows; evaluate on the same games. No feature selection or global imputation using evaluation outcomes. Nonlinear modeling is explicitly optional and slow.
5. **Independent secondary market comparison.** Only certified and timestamped pre-kickoff opening prices can be compared; no inferred openers, no backward reconstruction from closing markets, no actual betting authority.

### V1 is a foundation, not a complete 26-endpoint learner

The catalog `cfbd_native_all_source_families.csv` inventories **26** relevant endpoints. The V1 archiver has explicit fetch profiles for **7** core data families: games, team box, player box, drives, plays, advanced game stats, and havoc. The implemented Gold pregame features consume 5: games, team box, drives, plays and advanced game stats. Player box, havoc and other families are preserved as independent unprocessed raw information until player identity, definition stability, and historical availability audits pass. The broader 26-endpoint program (roster, returners, personnel, recruiting, ratings, play-level player attribution, opponent context) is **planned, not yet implemented**. Do not confuse this V1 coverage with 'CFBD in its entirety.'

## Commands / GitHub placement

Add the Python files at **repository root**, with `tests/test_native_v1.py` under `tests/`. This research path has **no imports from any production or Big Al modules**. It does not edit or overwrite production files.

Requires Python 3.11+, pandas, numpy, scikit-learn, threadpoolctl, pytest; `requests` only for optional network retrieval. The API key remains in environment variable `CFBD_API_KEY`, never saved to the raw JSON archive.

### 1) Query actual warehouse coverage first

`sql/01_existing_warehouse_read_only_audit.sql` contains read-only schema / field-coverage / identity probes for known BigQuery objects. Inspect table schema before running season-specific queries: the example columns must match the *live* warehouse before running. Export season-by-source-family counts and the schema results. **No authenticated BigQuery work was performed in building this package.**

### 2) Inventory available raw CFBD source archives

```
python cfbd_native_cli.py inventory --root native_cfbd_archive --years 2013:2025
```

No downloads or mutations. Lists response files, seasons, source counts, integrity issues, and missing families. Existing `ncaaf_cfbd_api_archive` is a different BigQuery object; a separate export/import adapter is needed to ingest its payloads if available. **Don't download CFBD calls that are already archived elsewhere.**

### 3) Plan API retrieval, optionally for a small pilot only

```
python cfbd_native_cli.py plan --root native_cfbd_archive --years 2013,2015,2022,2025 --families games,team_box,advanced,drives,plays
```

Week-constrained endpoints require an archived `/games` season response before they can plan only actual regular/postseason weeks. The `--years` argument accepts `2013:2025` or comma-separated years, not a mixed list of ranges.

**No HTTP requests are made by plan, inventory, build, or research.** A small explicit call (requires your existing key and authorisation):

```
export CFBD_API_KEY='<provided-through-secure-env>'
python cfbd_native_cli.py fetch --root native_cfbd_archive --years 2013 --families games --max-requests 1
```

`fetch` fails **before any network requests** if more calls are pending than `--max-requests`; optional quota may apply per account. Rate limiting/backoff and quota-aware resumability are not built yet; do not attempt 13 seasons of all-play requests in one unsupervised action. The archive is versioned by SHA-256 and never silently overwritten.

### 4) Build source-native pregame team states

```
python cfbd_native_cli.py build --root native_cfbd_archive --years 2013:2025 --output native_cfbd_gold
```

Exports:
- `cfbd_native_pregame_features.csv` — only pregame allowable fields and identifiers, no outcome/line target.
- `cfbd_native_outcomes_RESEARCH_ONLY.csv` — separately stored score/win labels.
- `cfbd_native_season_coverage.csv` — actual family-by-season raw and pregame computability.
- `cfbd_native_quality_report.json` — unresolved mapping, coverage, missingness, effective feature list.

The first yearly pregame observations can naturally have no prior games; these are missing/low-confidence, not filled by final season aggregates. The source-derived PPA definitions themselves may be revised retrospectively and need a separate methodology-vintage audit; 24h lag alone does **not** certify them.

### 5) Test which raw source families really matter

```
python cfbd_native_cli.py research --input native_cfbd_gold --output native_cfbd_research --earliest-year 2016 --latest-year 2025
```

The default lightweight study compares 2/5/all years using:

- Scores history only
- + native team box
- + drives
- + play measurements
- + advanced game efficiency

Each is evaluated on **margin, total and H2H**, year-forward with training-only preprocessing. Optional slower nonlinear research pass: add `--nonlinear` (only native full-source margin/totals). No model promotion is enabled.

Exports:
- `cfbd_native_oof_ledger_RESEARCH_ONLY.csv` — game-specific OOF evidence and model identity.
- `cfbd_native_research_report.json` — per-season actual metrics, usable predictor count, models, provenance status.

For a certified opening-price diagnostic, supply a CSV with `game_id, season, home_open_margin, quote_timestamp_utc, kickoff_timestamp_utc, opener_vintage_verified` using `--market-quotes FILE`. The quoted target sign must be the **expected home margin**; point spreads in home-team wager orientation must be converted correctly before upload. It will return `NO_CERTIFIED_PAIRED_OPENS` if there aren't verified quotes, not fabricate a market result.

## Scientific agenda beyond software pass/fail

### Immediate forensic priorities

1. **True archive coverage:** What exists 2013–21 for drives and plays, not what CFBD theoretically offers? Coverage by conference, FBS/FCS, game date, score, field, and missingness patterns. Capture not only row counts but observation *distribution and variance* and low-level source quality.
2. **Football concepts from native grain:** Drives (true possessions; starting field position; three-and-outs; points and completion); plays (down, distance, explosive plays, PPA calibration and play sequencing); team boxes (turnovers, sacks, efficiency not captured by rollups); identity-linked player exposure/continuity when validated. Compare original calculations to reconstructed calculations; don't silently treat derived success rates as CFBD's original definitions.
3. **Opponent-/schedule-adjusted latent offense, defense and pace:** A strong game model predicts points for and against, not only last-three YPP. Test hierarchical partial pooling, paired team strength updates, seasonal drift, home field and schedule corrections. The initial V1 includes a simple opponent-adjusted prior margin; it is a research control, **not** a completed hierarchical latent-state estimator.
4. **Unseen relationship discovery:** Use interaction discovery on *out-of-fold residuals* with independent-family evidence: defensive explosiveness vs opponent's pass depth; tempo x turnover risk; starting field position x finishing efficiency; opponent-strength acceleration vs current opponent. Record effect direction, confidence, eras, team exposure and independent number of games. Treat as hypotheses, not betting systems, until repeated forward confirmation.
5. **Information-over-time experiments:** Use 2, 5, 13 year fits *and* a distinct state-reset experiment, era breaks, recency weighting, and measurement-quality-aware weights. The V1 code tests **training-window only** (team-history state retains earlier games), not full state reset. This distinction must be labeled.
6. **Systematic leakage falsification:** Perturb a single game's postgame metrics, outcome, opponent stats and every future score; rerun upstream reconstruction and assert earlier features unchanged. Proven pregame quote snapshots must precede kickoff; do not infer quotes from 2026 retrieval.
7. **Statistical selection control:** Nested chronological selection, fixed comparison population, paired bootstrap by game/team/week, season-stability and multiple-hypothesis corrections. 2024–25 are revisited developmental seasons and cannot be portrayed as untouched. Use a genuinely unseen prospective 2026+ ledger for promotion decisions.

### Acceptance criteria to call a CFBD finding legitimate

- Attributable to an identified raw API family and stable definition; no outcome/current-game leakage.
- Replicable on independently held-out chronological blocks with explicit candidate-selection timestamps.
- Has nontrivial support across teams, opponents and seasons; not merely many correlated plays from the same game.
- Beats a score-history-only benchmark and separately tests incremental value beyond market on **identical certified-opener games**.
- Improvement magnitude exceeds sampling error and survives research search/multiple comparisons.
- Reproducible immutable input digest / model parameters / fold identifier.
- No production authority until separate prospective confirmation and explicit governance change.

## Tests and verified scope

`python -m pytest -q tests/test_native_v1.py`

16 focused tests pass, including archive conflict detection, team identity, same-game canary, future-game outcome invariance, same-day embargo, missingness, train-only median imputation, price provenance, and no target leakage. Additionally a **390-game generated synthetic archive across 2013–2018** completed source-native features and a **135-fold chronological Ridge/logistic integration experiment** locally. Synthetic metrics establish software behavior only; **no authentic CFBD model performance is reported**.

No Core source files or tables changed, no Big Al/Pathi/Miners integrated, no auto-betting capability, no GCP job executed, no new measured RMSE improvement.
