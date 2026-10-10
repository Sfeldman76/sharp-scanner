# NCAAF Production V1 Ledger Contract-Bind Repair

## Problem confirmed from the 2026-10-09 Weekly Update log
The Weekly Update successfully loaded a valid Production V1 artifact for settlement, but Utils independently re-loaded the contract and returned `CONTRACT_NOT_PUBLISHED`. As a result, the current slate had thousands of market rows but `scored_rows=0`, `attempted=0`, and `inserted=0`.

This was a scoring/ledger plumbing failure, not a missing Production V1 artifact and not an NCAAF Research V2.29 failure.

## Files to replace
Replace together:
1. `utils.py`
2. `train_job.py`

No changes are required to:
- `ncaaf_research_v2.py`
- `ncaaf_production_v1.py`
- `ncaaf_production_ledger_v1.py`
- the Production V1 artifact

Do **not** republish Production V1 solely for this repair.

## Repair 1 — one contract identity for Weekly Update
`train_job.py` already validates and loads the Production V1 contract before settlement. Weekly Update now passes that exact validated contract into:

`score_and_record_ncaaf_production_v1(..., contract=_contract)`

This removes the unnecessary second contract load for the weekly path and guarantees settlement and current-slate scoring use the same artifact identity.

Expected weekly log after deployment:

`[NCAAF-WEEKLY-REFRESH] ... contract_source=CALLER_VALIDATED artifact=<same artifact shown by NCAAF-WEEKLY-SETTLE> ...`

## Repair 2 — background scanner fallback preserved
Background scanner callers still call `score_and_record_ncaaf_production_v1()` without a contract. They continue to load Production V1 from GCS through the short-TTL cache.

New audit marker:

`[NCAAF-PROD-V1-CONTRACT-LOAD]`

PASS output identifies the module, source tag, bucket and artifact hash. If the loader fails closed, the diagnostic reports the artifact path, whether the blob exists, generation and size so a stale-module/source-tag issue can be distinguished from a genuinely missing object.

## Repair 3 — no negative caching of failed contract loads
Previously a `None` contract result could remain cached for up to five minutes. A failed load now resets the cache timestamp to zero, so the next scanner pass retries immediately.

Valid contracts still use the existing five-minute positive cache.

## Safety / architecture preserved
- Frozen Production V1 models are unchanged.
- No probability refit.
- No production publish or promotion.
- Research V2.29 is unchanged.
- CORE/Bet Authority logic is unchanged.
- Existing immutable FIRST/T24/T6/T1 ledger behavior is unchanged.
- Background scoring remains additive and fail-closed.

## Tests completed
- Python compilation: PASS for `utils.py` and `train_job.py`.
- Caller-validated contract bypasses fallback loader: PASS.
- Background fallback contract loader remains supported: PASS.
- Failed fallback load resets negative cache instead of holding `None` for five minutes: PASS.
- Diff audit confirms only the Production V1 contract/ledger plumbing and Weekly Update call/log were modified.

## Run after deployment
Run **NCAAF Production — Weekly Update** once.

Review:
1. `[NCAAF-WEEKLY-SETTLE]`
2. `[NCAAF-PROD-V1-CONTRACT-BIND]`
3. `[NCAAF-WEEKLY-REFRESH]`
4. `[NCAAF-PROD-V1-BACKGROUND]` if present
5. `[NCAAF-PROD-V1-CONTRACT-LOAD]` on ordinary background scanner runs

Success criteria for the Weekly Update:
- `contract_source=CALLER_VALIDATED`
- settlement artifact and refresh artifact are identical
- `scored_rows > 0` when current pregame rows exist
- `status` is no longer `CONTRACT_NOT_PUBLISHED`

`selected_markets`, `attempted`, and `inserted` may legitimately be zero if no current market passes Bet Authority or no new lock snapshot is due. `scored_rows=0` with thousands of current pregame rows is the condition this repair fixes.
