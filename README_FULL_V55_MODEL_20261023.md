# Sharp Scanner — Complete Repository, V5.5 Historical-Brain Integration

**Base:** the user-supplied `scanner.zip` (October 23, 2026). This archive contains the **entire retained repository**, not a replacement overlay or a shortened nine-file consolidation.

## Integrated model

The existing production models, V5.5 replay, independent weekly CORE scorer, V5.5 market board, legacy System Miner/STAT/Big Al/Pathi/PT pipelines, NFL engine, dashboard, deployment scripts and 2022–2026 CFBD research/archive utilities are retained.

**Updated:** `train_job.py` now invokes the full-history system evaluation after the legacy Heavy Research brains finish, using `ncaaf_heavy_brain_bridge.py`. The run is triggered by the existing `SPORT=NCAAF`, `MARKET=ncaaf_v55_heavy` route. It publishes to a distinct research-only GCS prefix, never a production-bet table.

**Added:** `ncaaf_brain_full_history.py` validates 2013–2025 independent system history, eligible/no-fire games, exact historical rule versions, separate straight-up and authentic ATS evidence, CORE-conditioned support/oppose/veto cohorts, season/role/edge splits, multiple-tested brain pairs and independent confirmation years. Rule histories are **never reduced to the CORE forecast game subset**. 2026 is held out of historical selection.

**Added:** `audit_bigquery_brain_sources.py` (read-only source inventory), and `research_validation_tests/` with source/schema requirements, test fixtures, and local checks. Fixed the previous installer incompatibility with this repository's already-supported `ncaaf_v55_heavy` alias.

## Data contract and execution

This is a **deployable source merge**, not proof of a Cloud Run deployment or a historical validation run. The new lane needs *actual source evidence* for the five historical brains. It does NOT infer historical Big Al/PT/Pathi selections from 2026 feeds or assume missing events mean no trigger.

You must provide seven historical exports under an immutable GCS prefix (CSV or Parquet):

- `games`, `core`, `signals`, `coverage`, `eligibility`, `rule_evaluations`, `universe_manifest`.

Set `NCAAF_BRAIN_HISTORY_INPUT_PREFIX=research/ncaaf/brain_inputs/<archive_batch>` on the **Heavy Research job**, and upload each `<name>.parquet` or `<name>.csv` to that prefix inside `gs://sharp-models`. An alternative seven-table BigQuery binding is supported, but the read-only tables and point-in-time provenance must be independently audited. See `research_validation_tests/DATA_SCHEMA.md` for every field. A schema-discovery script does not generate or certify historical brain events.

Deploy from this full repo using your existing Dockerfile/Cloud Build mechanism and then run `SPORT=NCAAF MARKET=ncaaf_v55_heavy`. Shadow reports go to `gs://sharp-models/research/ncaaf/v55/combined_brains_shadow/`.

If those input archives are absent, the full Heavy Research workflow logs and publishes **HOLD**, while existing V5.5/legacy paths continue. It does not mark missing PT/Big Al history as losing or ineligible. No new betting authority or automatic system promotion is permitted. The currently existing V5.5 live policy is **unchanged** by this work.

## How to verify locally

```bash
python -m compileall -q .
python -m unittest discover -s research_validation_tests/tests -p 'test_*.py' -v
python validate_scanner_release.py
```

The tests include synthetic histories (not actual game validations). This environment does not have authenticated project-level GCS or BigQuery access and **has not tested 2013–2025 source provenance against live data or deployed Cloud Run**. Historical system performance and financial returns remain unknown.

## Preserved integrity and source history

`release_notes/RELEASE_FILE_SHA256.json` lists each shipped file digest. Historical manifests such as `PACKAGE_MANIFEST.json` and `LEGACY_MODULE_REFERENCE_SHA256.json` are preserved as legacy documents, not overwritten with invented current-version assertions. `release_notes/train_job_combined_brain_merge.diff` records the exact entry-point modification. The unmodified `train_job.py` backup is in release notes with a non-`.py` extension. Revert through source control if required.
