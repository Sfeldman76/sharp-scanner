# NCAAF Heavy Research — October 18 Three-Fix Release

**Research only. Do not alter the frozen `ncaaf_production_v1.py`.**

This release was constructed from the October 17 `ncaaf_research_v2.py` provided by the user. That file's **pre-update SHA-256 prefix was `ba4032b50789bd1f`**, matching the October 18 Cloud Run `[NCAAF-RV2330-DEPLOY-PREFLIGHT]` fingerprint. The foundation/hierarchical/discovery source comes from the October 17 GitHub-ready research packages. **If you have committed later changes to any replacement target, merge rather than blindly overwrite.**

## GitHub files (put all in the same directory as `ncaaf_research_v2.py`)

| File | GitHub action | Function |
|---|---|---|
| `ncaaf_research_v2.py` | Replace matching Oct 17/18 version | Merge Big Al card with existing cloud observations by ID and exact selection fingerprint, preserve existing remote data; add NFL/NCAAF descriptive layer comparisons and merge logs |
| `big_al_observations.jsonl` | Add / merge | Exactly **23** rated Oct 17–18 observations (13 NCAAF, 10 NFL). No outcome data; lines are the user-observed release prices, not inferred closes |
| `big_al_multilayer_attribution.py` | Add | Match comparable high-vs-low ratings by market, role and nearby price; detect same-game side/total pairs, group duplicate-looking atoms. Zero system qualification or voting authority |
| `ncaaf_historical_signal_discovery.py` | Replace Oct 17 file | Modern CFBD specialist trains only from 2022+ verified pregame data; select on 2023, evaluate 2024–25 retrospectively. Full-history lane unchanged |
| `ncaaf_combined_signal_challenger.py` | Add | Tests promising historical family pairs, child vs both parents, paired 2018–23 selection, blocked 2024–25 review; Holm correction; no promotion |
| `ncaaf_full_history_foundation.py` | Replace Oct 17 file | Integrates combined-signal challenger after existing hierarchical/discovery research |
| `ncaaf_hierarchical_research.py` | Replace Oct 17 file | Adds frozen-feature observed-population audit marker by history era and training-fold usage |

**Do not change:** `ncaaf_production_v1.py`, `ncaaf_core_challenger_v2.py`, `ncaaf_pregame_cfbd.py`, `ncaaf_core_feature_parity.py` (already deployed), `nfl_engine.py`, `utils.py`, or `train_job.py`.

`BIG_AL_23_PICKS.csv` and `BIG_AL_READABLE_PICKS.txt` are for human review and should **not** be uploaded as model code. Tests are in `tests/` and are optional.

## Cloud Run deployment

Rebuild/redeploy your existing `sharp-train-job` from GitHub using your normal Google Cloud workflow. Then run **NCAAF Research — Heavy Challenger Search**. Merely changing GitHub does not update an existing Cloud Run image unless your pipeline automatically rebuilds.

## Required log markers

- `[BIGAL-CARD-MERGE]` — `local_rows=23`, `new_ids=...`, GCS status `REMOTE_SYNCED` (or `REMOTE_READ` on subsequent runs). Distinct count may not be 23 more than previous count when picks already exist in GCS.
- `[BIGAL-OBSERVATION-LEDGER]` — includes both 13 new NCAAF and 10 new NFL entries; preserves old entries. **NFL-specific production/Miner runs are not altered by this NCAAF package.**
- `[BIGAL-MULTILAYER-RESEARCH]` — descriptive rated observations, matched pregame fingerprints, comparable 6*/2* or 10*/2* hypotheses, no 2026 results used.
- `[NCAAF-HISTORICAL-SIGNAL-FAMILY]` for `modern_cfbd_pregame` — `modern_selection_train_years=[2022]` and 2023 `selection_n` should be greater than zero if verified modern data remains available. Remains unqualified if source or sample is insufficient.
- `[NCAAF-COMBINED-SIGNAL]` — pair counts and pre-2024 research shadow nominations per head; does **not** promote features to Core.
- `[NCAAF-FROZEN-FEATURE-COVERAGE]` — observed raw coverage in 2013–21 and 2022–25, plus folds using each frozen feature; this is **not** a numerical equality claim against the production model's training data.
- `[NCAAF-FULL-HISTORY-CONTRACT]` — still PASS, full 2013–2025 score-based margin, H2H, totals OOF; Production V1 frozen.

## Limitations and safety

- The **11 observations in the last live log** were stored in remote GCS; the 23-file card has not yet been run in Cloud Run. We therefore cannot claim the live ledger has updated until a new Heavy log confirms it.
- NCAAF Heavy now reads and reports **NFL** observations in the shared expert ledger, but actual **NFL Miner historical hypothesis integration** still requires a separately verified version of the current `nfl_engine.py`. This ZIP does not replace that engine, nor claim that NFL bet decisions changed.
- Some recently observed college games may not match the historical-context game frame. `matched_fingerprints` will report that honestly. The Big Al star rating does not grant extra votes.
- 2024–25 have been analyzed repeatedly; positive retrospective results are exploratory, not independent prospective confirmation.
- No 2026 outcomes are used for research selection, no live betting authority is introduced, and no source values are filled with invented historical observations.
- Cloud ledger uploads use GCS generation preconditions to avoid clobbering concurrent writers. If concurrent update is detected or the generation is unavailable, a safe sync error appears; retry on next Heavy job.

## Offline tests

Tests included in the ZIP: `test_oct18_three_fixes.py`, `test_historical_signal_discovery.py`, `test_hierarchical_dynamic.py`. They verify genuine eligible modern 2022→2023 training, full history, parent-child ablation, metadata-only ratings, GCS deduplication/idempotence, protection of remote entries, no 2026 labels, same-day leakage guards and other preexisting research contracts.

You can inspect the CSV or `README_GITHUB.md` directly on mobile; unlike the earlier ZIP, this package includes full replacement Python files and a human-readable index.
