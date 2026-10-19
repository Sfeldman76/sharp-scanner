# CFBD Archive Research Upgrade (October 2026)

This upgrade uses the existing `ncaaf_cfbd_warehouse_loader_v2.py` archive-preview feature builder, not a competing feature engine. No historical feature is copied from the old live BigQuery data; only the live table's schema is reused when creating a new immutable research dataset.

## Installation in Google Cloud Shell

Upload `cfbd_archive_research_upgrade.zip` via Cloud Shell's Upload File menu, then:

```bash
cd ~
unzip -o cfbd_archive_research_upgrade.zip
python3 apply_cfbd_archive_research_v1.py --repo ~/sharp-scanner
python3 ~/sharp-scanner/ncaaf_cfbd_warehouse_loader_v2.py --self-test
```

The installer backups the three updated Python source files in `~/sharp-scanner` before writing. It does not alter `ncaaf_production_v1.py`, `utils.py`, `train_job.py` or any BigQuery dataset.

## Rebuild the **full** 2013–2026 candidate archive features

```bash
cd ~/sharp-scanner
python3 ncaaf_cfbd_archive_research_build_v1.py --resume \
  2>&1 | tee ~/cfbd_archive_research_build.log
```

This runs the existing verified archive-preview feature builder once per season using the SAME feature definitions throughout. It reads only `~/cfbd_rebuild/cfbd_archive_v2/cfbd_archive`, saves a separate quality report and preview CSV for each year, and does not call CFBD API or BigQuery.

`--resume` skips a completed local season only if its archive manifest and patched loader SHA256 remain unchanged. It never reuses unchecked earlier outputs.

To publish **only after all 14 seasons pass local checks**:

```bash
python3 ncaaf_cfbd_archive_research_build_v1.py --resume --publish \
  2>&1 | tee ~/cfbd_archive_research_publish.log
```

Creates a new immutable BigQuery table `ncaaf_cfbd_archive_research_build_<timestamp>` and two new views:

- `sharplogger.sharp_data.ncaaf_cfbd_archive_research_v1` — 2013–2025 historical research; **no 2026 outcomes**.
- `sharplogger.sharp_data.ncaaf_cfbd_archive_shadow_2026_v1` — 2026 prospective/shadow only; never for retrospective selection.

If either view already exists, the publisher **refuses to overwrite** it. It never calls the live warehouse `WRITE_TRUNCATE` writer.

## Turn on the historical research source

The existing CORE Challenger and System Miner remain unchanged by default. To use the new research source in a research-only process/job, opt in:

```bash
export NCAAF_CFBD_ARCHIVE_RESEARCH_ENABLED=1
```

Enable only in the **heavy NCAAF research/challenger job**, not in a shared production/weekly or live prediction environment. In CORE, the research bridge reconstructs strict prior-only CFBD metrics from historical game records; in Miner, the bridge excludes same-game and uncertified static columns. All existing live Miner calls (`for_live=True`) continue to read the preexisting live table regardless of the environment flag. The 2026 shadow view is never used by historical model selection.

NOTE: This does **not** automatically deploy the source changes to Cloud Run. A repo patch in Cloud Shell is not a deployed job update; build/deploy the changed source through the existing Cloud Build/Git pipeline only when ready.

## Open cautions

- New archive is the **candidate** authoritative source. Different from old BQ != proven correct; raw API definitions/vintages still require independent checks.
- 2013–2020 historical opening odds do not exist; no fabricated ATS/line evaluation.
- Current-game `CFBD_Game_*` numbers are retrospective observations and MUST NOT be used as predictors. CORE reconstructs strict date-earlier prior statistics, and Miner restricts its archive input list to prior features.
- Archived roster/returning/talent release dates are not validated. These static feature families are excluded from the new CORE and Miner opt-in bridge until a release-vintage audit.
- The feature builder currently drops missing source families rather than borrowing the old warehouse's values. The archive may contain endpoint-definition differences; record sources and collect the intended param variant before claiming production-ready accuracy.
- The existing 2013–2021 research staging table is retained but NOT mixed with the new unified archive research view.
