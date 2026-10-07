# NFL Engine V3.11.0 — Expert/System Atom Expansion + Exact Occurrence Bridge

Base: `NFL_ENGINE_V3_10_8_LIVE_MINER_EVALUATOR.zip`

## Replace only

- `nfl_engine.py`
- `train_job.py`

Do **not** replace `utils.py`, `sports_edge_authority_v1.py`, or `sharp_line_dashboard.py` for this patch.

## Run

Run **NFL Research — Heavy Challenger Search** (`MARKET=nfl_research_heavy`).

## What changed

The existing NFL System Miner remains the single Miner. V3.11.0 adds Pathi and Big Al as first-class research atoms by projecting the exact occurrence ledgers already used to grade those systems onto the same side-state used by the Miner.

No Pathi or Big Al rule is reconstructed a second time. Every `(physical_game_id, bet_team, system_id)` occurrence must match an exact Miner side row or the run fails closed.

Pathi mirror/nested variants share normalized Pathi family categories. Big Al nested variants share their existing independence families. The Miner can also test bounded confluence atoms such as 2+ Pathi families, 3+ Pathi families, 2+ independent Big Al families, and Pathi + Big Al on the same side.

The existing horizon-symmetric 1/2/3-game grammar, magnitude, opponent symmetry, team memory, FDR/max-stat controls, chronological folds, LOSO, remove-best-season, frozen 2023/2024/2025 validation, and 2026 seal remain intact.

## Authority policy

New Miner mechanisms containing any `EXPERT_PATHI_*` or `EXPERT_BIGAL_*` atom are **research/shadow only at introduction**. They cannot immediately become a production confirmation family from retrospective evidence alone. Existing qualified Miner/Pathi/Big Al production behavior is unchanged.

CORE remains the prediction authority. H2H and Totals remain model-only. Production V1 and the live market backend are unchanged.

## Expected log markers

Look for:

- `[NFL-SYSTEM-V311-EXPERT-OCCURRENCE-BRIDGE]`
- `[NFL-RESEARCH-V2-SYSTEM-EXPERT-ATOM-BRIDGE]`
- `[NFL-RESEARCH-V2-SYSTEM-RETAINED]` rows whose conditions contain `EXPERT_PATHI_` or `EXPERT_BIGAL_`
- `[NFL-RESEARCH-V2-SYSTEM-MECHANISM-FAMILY]` with `expert_bridge=true`
- `[NFL-RESEARCH-V2-SYSTEM-CONTRACT]` with nonzero `expert_atom_count`; `expert_bridge_mechanisms` may legitimately be zero if no expert interaction survives the research gates.

The occurrence bridge should report `pathi_occurrences == pathi_matched` and `bigal_occurrences == bigal_matched`. Any mismatch is a hard failure.
