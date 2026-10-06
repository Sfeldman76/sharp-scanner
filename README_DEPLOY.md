# NCAAF Engine V2.14 — PT Relay + Exact Expert-Side Miner Bridge

## What this fixes

V2.13.1 proved the Heavy job and CORE/specialist bridge were healthy, but two intended intelligence sources were absent:

1. Prediction Tracker returned HTTP 403 to Cloud Run for both archive and live endpoints, leaving `external=0` and no META margin.
2. Pathi/Big Al were present in the historical W/L cache but disappeared before Miner atom construction, leaving `pathi=0 bigal=0`.

V2.14 fixes only those integration boundaries. Production V1, CORE challenger, thresholds and authority policy remain unchanged.

## Prediction Tracker fetch path

The loader remains source-faithful to Prediction Tracker. It now tries:

1. direct same-origin browser session;
2. read-only relay transport (`PT_RELAY_PREFIX`, default `https://r.jina.ai/http://www.thepredictiontracker.com`);
3. previously validated GCS cache.

The relay is transport only. It does not provide or rename ratings. CSV columns are still subject to the V2.13 name-safe header contract. The live named table and live CSV must uniquely validate the five systems before `META_MARGIN` is created. Archive files then reuse that verified manifest. Unknown/ambiguous mapping fails closed.

Completed historical seasons remain cache-once in GCS. The 2026 archive and current-week source refresh normally and are merged with live records winning only exact overlaps.

## Exact Pathi / Big Al side bridge

V2.14 no longer depends on the lean `miner_games` frame to recreate published-system flags. Heavy Research reuses the same validated historical source/builders used by the Pathi/Big Al W/L engine, then maps those team-side flags to the Miner's HOME-oriented physical-game frame.

- HOME-side triggers retain the canonical flag name.
- ROAD-side triggers receive `__ROAD_SIDE`.
- Miner still chooses PLAY_ON vs FADE from discovery and must pass the existing confirmation/FDR/dependency/lineage gates.
- No Pathi or Big Al flag receives direct authority.

This preserves road-side systems instead of forcing every expert trigger into the home orientation.

## Expected diagnostic lines

A successful run should include:

```text
[NCAAF-RV214-DEPLOY-PREFLIGHT] PASS
[NCAAF-PT-HEADER]
[NCAAF-PT-HEADER-VERIFY]
[NCAAF-PT-COLUMN-MAP]
[NCAAF-PT-LIVE] ... source=WEB_RELAY ...   # if direct remains blocked
[NCAAF-PT-SEASON] ... source=WEB_RELAY ... # first uncached archive fetches
[NCAAF-PT-MATCH] ... matched_rows=>0
[NCAAF-PT-CONTRACT] status=PASS ... matched_rows=>0
[NCAAF-RV214-EXPERT-SIDE-BRIDGE] status=PASS ... pathi_cols=>0 bigal_cols=>0 ...
[NCAAF-RV25-ATOM-BRIDGE] market=spreads ... pathi=>0 bigal=>0 external=>0
[NCAAF-RV25-CONTRACT] status=PASS ...
```

`bridge_mechanisms` may still be zero if none of the added atoms survives the research gates; that would then be a research result rather than an integration failure.

## Deploy

From V2.13.1 replace only:

- `ncaaf_research_v2.py`
- `train_job.py`

Redeploy, then run **NCAAF Research — Heavy Challenger Search** once. Do not run Production Publish.

## Safety / authority

- Production V1 unchanged.
- CORE challenger unchanged.
- 2026 remains excluded from discovery/confirmation/threshold selection.
- Prediction Tracker, Pathi and Big Al bridge atoms are research evidence only unless they independently satisfy the existing Miner authority policy.
- External ratings remain one correlated evidence family; five external models are not five authority votes.
