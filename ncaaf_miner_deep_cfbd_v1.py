"""Bounded deep CFBD x situational Miner with gated historical promotion.

Search <=2023; freeze rules; evaluate 2024 and 2025 with unique-game
holdout inference and multiplicity adjustment. Only eligible if independently
certified as-of pregame features, fixed exact thresholds and live atom evaluation.
Retrospective CFBD_PG_ values can never earn historical authority by name.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

VERSION = 'NCAAF-DEEP-CFBD-AUTHORITY-V2-20261028'
CONFIRM_YEARS = (2024, 2025)
DISC_MAX = 2023

# Conservative source-class allowlist. A field is NOT PIT-certified merely
# because its name says Prior; full source lineage/prospective verification is
# required before any future authority proposal.
_PRIOR_PATTERN = re.compile(
    r'^CFBD_(?:Opp_)?(?:Season_.*_Prior|Recent3_.*_Prior|PriorYear_.*)$'
)
_STATIC_PATTERN = re.compile(
    r'^CFBD_(?:Opp_)?(?:Roster_|Returning_|Talent_|Recruit_|Portal_|Prior_(?:PPA_|Usage_|Top|High|QB|Returning))'
)
_BAD_ANY = re.compile(
    r'(?:Actual|Final|Outcome|Target|Against_Spread|Game_Winner|Result|Live_|Postgame|'
    r'_Observed|_Weather|WindSpeed|Precipitation|Closing|Snapshot|_At$)', re.I
)
_CONTEXT_FAMILIES = {
    'ATS_FORM','ATS_SEQUENCE','SEASON_RECORD_STATE','ROLE_TRANSITION_BOUNCEBACK',
    'MATCHUP_HISTORY','RIVALRY','CONFERENCE_PAIR','SCHEDULE_RESUME_QUALITY',
    'SU_FORM','SU_SEQUENCE','MATCHUP_DIFFERENTIAL','TEAM_STATE','OPP_STATE',
    'VENUE','MARKET_ROLE','REST','TRAVEL_CONTEXT','ROAD_SEQUENCE',
    'CFBD_CONFERENCE_CONTEXT','CFBD_RATINGS_CONTEXT',
}


def _cfbd_status(column: str) -> str:
    if not str(column).startswith('CFBD_'):
        return 'NOT_CFBD'
    if column.startswith('CFBD_PG_'):
        return 'POSTGAME_FIRST_SEEN_NOT_HISTORICALLY_CERTIFIED'
    if _BAD_ANY.search(column):
        return 'POTENTIAL_OUTCOME_OR_NONPREGAME'
    if not (_PRIOR_PATTERN.match(column) or _STATIC_PATTERN.match(column)):
        return 'PREGAME_LINEAGE_UNVERIFIED'
    return 'NAME_SCREENED_PRIOR_OR_PRESEASON'


def _fam(column: str) -> str:
    s = column.upper()
    # Separate families permit OFF+DEF+ATS+RIVALRY interactions while
    # discouraging redundant overlapping variants within the same family.
    if any(x in s for x in ('PORTAL','ROSTER','RETURN','RECRUIT','TALENT')):
        return 'AUTO_PERSONNEL'
    if 'PRIORYEAR' in s or 'FPI' in s or 'SRS' in s or 'ELO' in s:
        return 'AUTO_PRIORYEAR_RATING'
    if 'RECENT3' in s:
        return 'AUTO_RECENT_FORM'
    if 'MATCHUP' in s:
        return 'AUTO_MATCHUP_ADV'
    if 'OFF_' in s or '_OFF_' in s:
        return 'AUTO_OPP_OFF' if 'OPP_' in s else 'AUTO_TEAM_OFF'
    if 'DEF_' in s or '_DEF_' in s:
        return 'AUTO_OPP_DEF' if 'OPP_' in s else 'AUTO_TEAM_DEF'
    if any(x in s for x in ('PACE', 'PLAY', 'TEMPO')):
        return 'AUTO_TEMPO'
    return 'AUTO_OTHER'


def _two_sided_p(successes: float, n_games: int) -> float:
    # n is unique physical games, not mirrored team-side rows. Conservative
    # normal approximation; exact low-n inference is intentionally not used.
    if n_games <= 0:
        return 1.0
    z = abs(successes - 0.5) / math.sqrt(0.25 / n_games)
    return float(math.erfc(z / math.sqrt(2)))


def _bh(pvalues: list[float]) -> np.ndarray:
    p = np.array(pvalues, dtype=float)
    if not len(p):
        return p
    ranks = np.argsort(p)
    qs = np.empty(len(p), dtype=float)
    qs[ranks] = np.minimum.accumulate(
        (p[ranks] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1].clip(0, 1)
    return qs


def _get_sides(g: pd.DataFrame, seasons: np.ndarray):
    sy = np.asarray(seasons, dtype=float)
    game_id = g.get('Game_ID', pd.Series(np.arange(len(g)), index=g.index)).astype(str).to_numpy()
    # Distinct key when Game_ID is missing; never collapse every null into one.
    for i, gid in enumerate(game_id):
        if gid in ('', 'nan', 'None', '<NA>'):
            game_id[i] = f'ROW_{i}'
    return sy, game_id


# 53% is a minimum screen, NOT a complete statistical qualification.
MIN_CONFIRM_N = 60
MIN_CONFIRM_RATE = 0.53
REFERENCE_MINUS110_BREAKEVEN = 110.0 / 210.0
HOLDOUT_FDR_Q = 0.10


def _exact_binomial_upper_tail(wins: int, n: int, p0: float) -> float:
    """P[X >= wins], exact binomial using log probabilities (no SciPy needed)."""
    if not 0 <= wins <= n or not 0 < p0 < 1:
        return 1.0
    if wins == 0:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k*math.log(p0) + (n-k)*math.log1p(-p0) for k in range(wins,n + 1)]
    mx=max(logs)
    return min(1.0,math.exp(mx)*sum(math.exp(t-mx) for t in logs))


def _certified_fields():
    """Load separately audited feature lineages, not a name-based assertion.

    JSON format: {"CFBD_Season_Off_PPA_Prior":
      {"pregame_asof_verified": true, "verified_2024_2025": true,
       "source": "...", "audit_id": "..."}}
    Source/audit_id must identify independent reproducible lineage evidence.
    """
    path=os.getenv('NCAAF_MINER_PIT_CERTIFICATION_FILE','').strip()
    if not path:
        path=str(Path(__file__).with_name('ncaaf_deep_pit_certified_fields.json'))
    if not os.path.isfile(path):
        return {}
    try:
        with open(path,encoding='utf-8') as f:
            data=json.load(f)
        return {str(k):v for k,v in data.items() if isinstance(v,dict)
                and v.get('pregame_asof_verified') is True
                and v.get('verified_2024_2025') is True
                and str(v.get('source') or '').strip()
                and str(v.get('audit_id') or '').strip()}
    except (OSError,ValueError,TypeError):
        return {}


def frozen_live_auto_atoms(g: pd.DataFrame, mechanisms: list[dict]) -> dict[str,np.ndarray]:
    """Evaluate ONLY exact frozen CFBD atom definitions; no new quantiles."""
    masks={}
    for mech in mechanisms:
        if mech.get('miner_lane')!='DEEP_CFBD_AUTO':
            continue
        for atom in mech.get('auto_atom_specs') or []:
            col=str(atom.get('source_field') or '')
            name=str(atom.get('name') or '')
            if _cfbd_status(col)!='NAME_SCREENED_PRIOR_OR_PRESEASON' or not col in g:
                continue
            try:
                op=str(atom['operator']); t=float(atom['threshold'])
                if not math.isfinite(t) or op not in ('EQ','LE','GE') or name != _identity(col,op,t):
                    continue
                vec=pd.to_numeric(g[col],errors='coerce').to_numpy(dtype=float)
                masks[name]=np.asarray(np.isfinite(vec)&((vec==t) if op=='EQ' else ((vec<=t) if op=='LE' else (vec>=t))),dtype=bool)
            except (KeyError,TypeError,ValueError):
                continue
    return masks


def _rate(mask: np.ndarray, y: np.ndarray, valid: np.ndarray, gid: np.ndarray, scope: np.ndarray):
    ix = np.flatnonzero(mask & valid & scope)
    if not len(ix):
        return {'rows': 0, 'games': 0, 'rate': None}
    # A physical game contributes one effective observation. Two team sides
    # qualifying in opposite directions are averaged within the game.
    _, inv = np.unique(gid[ix], return_inverse=True)
    cnt=np.bincount(inv)
    vals = np.bincount(inv, weights=y[ix]) / cnt
    # Mirrored qualifying team-sides are deliberately NOT counted as
    # independent betting opportunities in inferential promotion tests.
    unambiguous=(cnt==1)
    return {'rows': int(len(ix)), 'games': int(len(vals)), 'rate': float(vals.mean()),
            'unambiguous_games':int(unambiguous.sum()),
            'unambiguous_wins':int(np.rint(vals[unambiguous].sum())),
            'ambiguous_games':int((~unambiguous).sum())}


def _support(metric: dict, years: np.ndarray, mask: np.ndarray, valid: np.ndarray,
             gids: np.ndarray, y: np.ndarray, min_game_n: int):
    if metric['games'] < min_game_n:
        return False
    byyear = 0
    for yr in sorted(set(int(t) for t in years if np.isfinite(t) and t <= DISC_MAX)):
        q = _rate(mask, y, valid, gids, years == yr)
        if q['games'] >= 12:
            byyear += 1
    return byyear >= 2


def _identity(column: str, op: str, cut: float) -> str:
    return f'AUTO::{column}::{op}::{cut:.8g}'


def _numeric_catalog(g: pd.DataFrame, sy: np.ndarray, y: np.ndarray,
                     valid: np.ndarray, gids: np.ndarray, *, min_discovery_games: int = 80):
    disc = np.isfinite(sy) & (sy <= DISC_MAX)
    columns = [c for c in g.columns if str(c).startswith('CFBD_')]
    audit = []
    atoms = []
    for col in columns:
        status = _cfbd_status(str(col))
        item = {'field': str(col), 'source_class': status}
        if status != 'NAME_SCREENED_PRIOR_OR_PRESEASON':
            item['admission'] = 'REJECTED'; audit.append(item); continue
        vec = pd.to_numeric(g[col], errors='coerce').to_numpy(dtype=float)
        ok = np.isfinite(vec) & valid & disc
        n_games = len(set(gids[ok]))
        item['discovery_game_coverage'] = n_games
        if n_games < 2 * min_discovery_games:
            item['admission'] = 'LOW_DISCOVERY_COVERAGE'; audit.append(item); continue
        if len(set(int(x) for x in sy[ok])) < 2:
            item['admission'] = 'ONE_DISCOVERY_SEASON'; audit.append(item); continue
        vals = vec[ok]
        if len(np.unique(vals)) <= 1:
            item['admission'] = 'CONSTANT'; audit.append(item); continue
        if not np.all(np.isfinite(vals)):
            item['admission'] = 'NONFINITE'; audit.append(item); continue
        # Thresholds selected using <=2023 ONLY. Test broad tails per feature;
        # include binary variables as a single equality test.
        unique = np.unique(vals)
        defs = [('EQ', float(unique[-1]))] if len(unique) == 2 else [
            ('LE', float(np.quantile(vals, 0.25))),
            ('GE', float(np.quantile(vals, 0.75)))
        ]
        admitted = 0
        for op, cut in defs:
            mask = (vec == cut) if op == 'EQ' else ((vec <= cut) if op == 'LE' else (vec >= cut))
            mask = np.asarray(mask & np.isfinite(vec), dtype=bool)
            metric = _rate(mask, y, valid, gids, disc)
            if not _support(metric, sy, mask, valid, gids, y, min_discovery_games):
                continue
            atoms.append({'name': _identity(col, op, cut), 'source_field': col,
                          'family': _fam(col), 'mask': mask,
                          'kind': 'CFBD', 'threshold': cut, 'operator': op,
                          'discovery_games': metric['games']})
            admitted += 1
        item['admission'] = 'ATOMS_CREATED' if admitted else 'NO_THRESHOLD_SUPPORT'
        item['atom_count'] = admitted
        audit.append(item)
    return atoms, audit


def _context_catalog(existing: list[dict], y: np.ndarray, sy: np.ndarray,
                     valid: np.ndarray, gids: np.ndarray, min_discovery_games: int):
    disc = np.isfinite(sy) & (sy <= DISC_MAX)
    out = []
    seen = set()
    for atom in existing:
        fam = str(atom.get('family') or '')
        name = str(atom.get('name') or '')
        if fam not in _CONTEXT_FAMILIES or name in seen:
            continue
        mask = np.asarray(atom.get('mask'), dtype=bool)
        if mask.size != len(y):
            continue
        metric = _rate(mask, y, valid, gids, disc)
        if not _support(metric, sy, mask, valid, gids, y, min_discovery_games):
            continue
        out.append({'name': name, 'family': fam, 'mask': mask, 'kind': 'CONTEXT',
                    'discovery_games': metric['games']})
        seen.add(name)
    return out


def _leaderboard(g: pd.DataFrame, sy: np.ndarray, y: np.ndarray,
                 valid: np.ndarray, gids: np.ndarray, existing_atoms: list[dict], *,
                 max_depth: int = 6, max_features: int = 120,
                 beam_width: int = 64, support_each: int = 100,
                 pool_context_per_family: int = 8):
    disc = np.isfinite(sy) & (sy <= DISC_MAX)
    cfbd, inventory = _numeric_catalog(g, sy, y, valid, gids)
    context = _context_catalog(existing_atoms, y, sy, valid, gids, 45)
    all_tests = []
    tested_signatures = set()

    def evaluate(atoms: tuple[dict, ...]):
        keys = tuple(sorted(a['name'] for a in atoms))
        if keys in tested_signatures:
            return None
        tested_signatures.add(keys)
        mask = np.logical_and.reduce([a['mask'] for a in atoms])
        # A cheap row-count guard prevents high-cost grouping for tiny deep intersections.
        if int(np.count_nonzero(mask & valid & disc)) < support_each:
            return None
        met = _rate(mask, y, valid, gids, disc)
        if not _support(met, sy, mask, valid, gids, y, support_each):
            return None
        rate = met['rate']
        direction = 'PLAY_ON' if rate >= .5 else 'FADE'
        chosen = max(rate, 1-rate)
        p = _two_sided_p(rate, met['games'])
        yrs = {}
        for year in sorted(set(int(z) for z in sy[disc])):
            m = _rate(mask, y, valid, gids, sy == year)
            if m['games']:
                yrs[year] = {'n': m['games'], 'rate':m['rate'] if direction == 'PLAY_ON' else 1-m['rate']}
        stable = sum(v['rate'] > 0.5 for v in yrs.values() if v['n'] >= 12)
        stable_n = sum(v['n'] >= 12 for v in yrs.values())
        # Prefer a reproducible edge with broader history and less complexity.
        score = (chosen - 0.5) * math.sqrt(met['games']) + 0.12 * stable - 0.17 * (len(atoms)-1)
        item = {'atoms':atoms, 'mask':mask, 'name_key':keys,
                'families': tuple(sorted(a['family'] for a in atoms)),
                'depth':len(atoms), 'n':met['games'], 'rate':chosen,
                'direction':direction, 'p':p, 'score':score,
                'stable_years':stable,'years_with_sample':stable_n,
                'discovery_by_season':yrs}
        all_tests.append(item)
        return item

    # Every eligible CFBD variable is screened before a bounded deep beam.
    cfbd_single = []
    for a in cfbd:
        item = evaluate((a,))
        if item:
            cfbd_single.append(item)
    ctx_single = []
    for a in context:
        item = evaluate((a,))
        if item:
            ctx_single.append(item)
    # One CFBD family per atom, but multiple independent CFBD families can
    # coexist in deep hypotheses (e.g. prior offense + prior defense).
    by_fam = defaultdict(list)
    for item in sorted(cfbd_single, key=lambda v: v['score'], reverse=True):
        a = item['atoms'][0]
        by_fam[a['family']].append(a)
    selected_cfbd = []
    while len(selected_cfbd) < max_features and any(by_fam.values()):
        for fam in sorted(by_fam):
            if by_fam[fam] and len(selected_cfbd) < max_features:
                selected_cfbd.append(by_fam[fam].pop(0))
    contexts_by_fam = defaultdict(list)
    for item in sorted(ctx_single, key=lambda v: v['score'], reverse=True):
        contexts_by_fam[item['atoms'][0]['family']].append(item['atoms'][0])
    selected_ctx = [x for fam, arr in sorted(contexts_by_fam.items()) for x in arr[:pool_context_per_family]]
    # Search is anchored in one eligible CFBD atom AND one contextual state.
    # Starting with a context seed guarantees ATS/rivalry aren't buried by CFBD.
    pairs = []
    for a in selected_cfbd:
        for b in selected_ctx:
            st = evaluate((a,b))
            if st is not None:
                pairs.append(st)
    beam = sorted(pairs, key=lambda z:z['score'], reverse=True)[:beam_width]
    support_pool = selected_cfbd + selected_ctx
    depth_count = {1: len(cfbd_single)+len(ctx_single), 2: len(pairs)}
    for depth in range(3, max_depth+1):
        nxt = []
        for state in beam:
            used_fam = set(state['families']); used_names = set(state['name_key'])
            for atom in support_pool:
                if atom['family'] in used_fam or atom['name'] in used_names:
                    continue
                v = evaluate(state['atoms']+(atom,))
                if v is not None:
                    nxt.append(v)
        # Family-diverse beam: prevent a single statistical family dominating.
        ranked = sorted(nxt, key=lambda z: z['score'], reverse=True)
        beam = []
        buckets = Counter()
        for st in ranked:
            grp = tuple(sorted(x for x in st['families'] if x.startswith('AUTO_')))
            if buckets[grp] >= max(4, beam_width // 8):
                continue
            beam.append(st); buckets[grp] += 1
            if len(beam) >= beam_width:
                break
        depth_count[depth] = len(nxt)
        if not beam:
            break
    # FDR covers all evaluated discovery hypotheses, including beam rejects.
    q = _bh([z['p'] for z in all_tests])
    for z, val in zip(all_tests, q):
        z['q'] = float(val)
    shortlist = [z for z in all_tests if z['depth'] >= 2 and z['q'] <= .10 and
                 z['rate'] >= .53 and z['stable_years'] >= 2]
    shortlist = sorted(shortlist, key=lambda v:(v['score'],v['n']), reverse=True)[:75]
    return shortlist, inventory, {'tested_hypotheses':len(all_tests),
                                  'depth_evaluated':depth_count,
                                  'cfbd_atoms_discovery_supported':len(cfbd),
                                  'context_atoms_supported':len(context),
                                  'cfbd_atoms_in_deep_beam':len(selected_cfbd),
                                  'context_atoms_in_deep_beam':len(selected_ctx)}


def run_deep_cfbd_challenger(games: pd.DataFrame, seasons: np.ndarray,
                             existing_atoms: list[dict], *, log_func=print,
                             max_depth: int = 6) -> dict[str,Any]:
    """Return shadow-only deep ATS/context/CFBD discovery report.

    Call from Heavy Research on a fully enriched historical frame. Never pass
    this report to mechanism families or production authority.
    """
    g = games.reset_index(drop=True)
    sy, gids = _get_sides(g, seasons)
    gids = pd.factorize(gids, sort=False)[0]  # fast unique physical-game scoring
    if len(sy) != len(g):
        raise ValueError('season/frame alignment mismatch')
    # Requirement: 2026 outcomes are NEVER read, even for dynamic threshold
    # discovery. Explicitly discard 2026+ rows before constructing all metrics.
    eligible = np.isfinite(sy) & (sy <= max(CONFIRM_YEARS))
    g = g.loc[eligible].reset_index(drop=True)
    sy = sy[eligible]; gids = gids[eligible]
    sub_atoms = []
    for atom in existing_atoms:
        mask = np.asarray(atom['mask'], dtype=bool)
        if len(mask) == len(eligible):
            sub_atoms.append({**atom, 'mask':mask[eligible]})
        elif len(mask) == len(g):
            sub_atoms.append(atom)
    # A push (ATS margin == 0) is not a win or loss.
    actual = pd.to_numeric(g.get('Actual_Margin'), errors='coerce').to_numpy(dtype=float)
    spread = pd.to_numeric(g.get('Consensus_Open_Spread'), errors='coerce').to_numpy(dtype=float)
    net = actual + spread
    valid = np.isfinite(net) & ~np.isclose(net,0)
    y = (net > 0).astype(float)
    audit_stats = {'historical_sides':len(g), 'valid_ats_sides':int(valid.sum()),
                   'discovery_max_year':DISC_MAX, 'holdout_years':list(CONFIRM_YEARS)}
    if int(np.sum(valid & (sy <= DISC_MAX))) < 250:
        return {'status':'INSUFFICIENT_DISCOVERY_ROWS','version':VERSION,
                **audit_stats,'production_authority':0}
    shortlist, inventory, counts = _leaderboard(g, sy, y, valid, gids,
                                                 sub_atoms, max_depth=max_depth)
    # The 2024/25 holdout is read for the FIRST TIME after discovery thresholds,
    # ranks and shortlisted combinations are frozen.
    core_edge = pd.to_numeric(g.get('_V29_CORE_INCUMBENT_EDGE_POINTS',
                                 pd.Series(np.nan, index=g.index)), errors='coerce').to_numpy(dtype=float)
    core_candidate = np.isfinite(core_edge) & (core_edge >= 2.0)
    report = []
    for st in shortlist:
        direction = st['direction']; m = st['mask']
        ho = {}
        good = True
        for year in CONFIRM_YEARS:
            res = _rate(m, y if direction=='PLAY_ON' else 1-y, valid, gids, sy == year)
            ho[str(year)] = res
            if res['games'] < 10 or res['rate'] is None or res['rate'] < .50:
                good = False
        all_conf = _rate(m, y if direction=='PLAY_ON' else 1-y, valid, gids, np.isin(sy, CONFIRM_YEARS))
        if all_conf['games'] < 40 or all_conf['rate'] is None or all_conf['rate'] < .53:
            good = False
        matched_core = _rate(m & core_candidate, y, valid, gids, np.isin(sy, CONFIRM_YEARS))
        all_core = _rate(core_candidate, y, valid, gids, np.isin(sy, CONFIRM_YEARS))
        core_delta = (matched_core['rate'] - all_core['rate']) if (
            matched_core['rate'] is not None and all_core['rate'] is not None) else None
        _atom_specs=[{'name':a['name'],'source_field':a['source_field'],
                      'operator':a['operator'],'threshold':float(a['threshold'])}
                     for a in st['atoms'] if a.get('kind')=='CFBD']
        _pid='NCAAF-DEEP-'+hashlib.sha256((direction+'|'+','.join(st['name_key'])).encode()).hexdigest()[:16].upper()
        report.append({'conditions':list(st['name_key']), 'families':list(st['families']),
                       'mechanism_id':_pid,'auto_atom_specs':_atom_specs,
                       'depth':st['depth'], 'direction':direction,
                       'discovery_games':st['n'], 'discovery_rate':round(st['rate'],5),
                       'discovery_p':round(st['p'],9), 'discovery_fdr_q':round(st['q'],7),
                       'discovery_by_season':st['discovery_by_season'],
                       'confirmation':ho, 'confirmation_all':all_conf,
                       'confirmation_screen_pass':good,
                       'core_oof_candidate_definition':'CORE_OOF_EDGE_TEAM_2PLUS_ONLY_IF_PRESENT',
                       'core_oof_matched':matched_core,
                       'core_oof_unfiltered_candidates':all_core,
                       'core_oof_conditional_ats_lift':core_delta,
                       'state':'RESEARCH_SHADOW_NO_BET_AUTHORITY'})
    # Holdout is examined ONLY after discovery thresholds, atom directions,
    # full tested-hypothesis FDR and finalist list have been frozen. Use one
    # independent physical game per exact rule, and Holm/BH correction across
    # every held-out finalist (including those that fail the first screen).
    cert=_certified_fields()
    for h in report:
        ho=h['confirmation_all']
        n=int(ho.get('unambiguous_games',0)); wins=int(ho.get('unambiguous_wins',0))
        h['confirmation_unambiguous_n']=n
        h['confirmation_unambiguous_wins']=wins
        h['confirmation_p_above_minus110']=_exact_binomial_upper_tail(wins,n,REFERENCE_MINUS110_BREAKEVEN)
    qvals=_bh([h['confirmation_p_above_minus110'] for h in report])
    promotions=[]; blocked_pit=0; evidence_pass=0
    for h,q in zip(report,qvals):
        h['confirmation_fdr_q']=float(q)
        h['pit_field_attestations']={str(x['source_field']):cert.get(str(x['source_field']))
                                     for x in h['auto_atom_specs']}
        h['pit_certified']=bool(h['auto_atom_specs']) and all(
            x['source_field'] in cert for x in h['auto_atom_specs'])
        h['confirmation_unique_games']=bool(
            h['confirmation_all'].get('ambiguous_games',0)==0)
        h['holdout_fdr_pass']=bool(float(q)<=HOLDOUT_FDR_Q)
        h['statistical_pass']=bool(
            h['confirmation_screen_pass'] and h['confirmation_unique_games']
            and h['confirmation_unambiguous_n']>=MIN_CONFIRM_N
            and h['confirmation_all'].get('rate') is not None
            and h['confirmation_all']['rate']>=MIN_CONFIRM_RATE
            and h['holdout_fdr_pass'])
        if h['statistical_pass']:
            evidence_pass+=1
            if not h['pit_certified']:
                blocked_pit+=1
        h['authority_ready']=bool(h['statistical_pass'] and h['pit_certified'])
        h['authority_block_reason']=(
            'STATISTICAL_OR_HOLDOUT_EVIDENCE_INSUFFICIENT' if not h['statistical_pass'] else
            'PIT_LINEAGE_NOT_CERTIFIED' if not h['pit_certified'] else 'READY_FOR_EXACT_LIVE_EVALUATION')
        if h['authority_ready']:
            promotions.append({
                'mechanism_id':h['mechanism_id'], 'market':'spreads',
                'representative_system_id':h['mechanism_id'],
                'member_system_ids':[h['mechanism_id']], 'member_count':1,
                'miner_lane':'DEEP_CFBD_AUTO','representative_conditions':h['conditions'],
                'direction':h['direction'], 'families':h['families'],
                # All deep CFBD patterns count as at most one independent family
                # when supporting a CORE candidate, avoiding false multiple votes.
                'evidence_family_key':'NCAAF_CFBD_DEEP_DISCOVERY_FAMILY',
                'auto_atom_specs':h['auto_atom_specs'],
                'auto_atom_fields':[a['source_field'] for a in h['auto_atom_specs']],
                'confirmation_n':h['confirmation_unambiguous_n'],
                'confirmation_rate':h['confirmation_all']['rate'],
                'confirmation_pass':True,
                'deep_cfbd_validation_pass':True,
                'deep_cfbd_pit_certified':True,
                'deep_cfbd_holdout_fdr_pass':True,
                'deep_cfbd_exact_live_atoms_available':True, # checked again in live eval
                'deep_cfbd_unique_games_confirmed':True,
                'discovery_n':h['discovery_games'], 'discovery_rate':h['discovery_rate'],
                'deep_cfbd_holdout_q':float(q),
                'authority_state':'STRONG_VALIDATED',
                'live_authority_eligible':True,
                'current_qualified':True,
                'production_authority':0,
            })
    count_by_reject = Counter(z['admission'] for z in inventory)
    output = {'status':'PASS', 'version':VERSION, 'market':'spreads',
              'source':'ALL_NAME_SCREENED_PRE2024_CFBD_FIELDS__SHADOW',
              'numeric_cfbd_columns_inventoried':len(inventory),
              'feature_admission':dict(count_by_reject),
              'features_admitted':[z['field'] for z in inventory if z['admission']=='ATOMS_CREATED'],
              'excluded_postgame_pit':[z['field'] for z in inventory if z['admission']=='REJECTED' and
                                       z['source_class']=='POSTGAME_FIRST_SEEN_NOT_HISTORICALLY_CERTIFIED'],
              'feature_inventory':inventory, **counts, **audit_stats,
              'shortlisted_systems':len(shortlist),
              'holdout_screen_pass':sum(x['confirmation_screen_pass'] for x in report),
              'statistical_authority_pass':evidence_pass,
              'blocked_pit_count':blocked_pit,
              'authority_ready_count':len(promotions),
              'auto_qualified_mechanisms':promotions,
              'top_hypotheses':report,
              'search_depth_cap':max_depth,
              'pvalue_policy':'TWO_SIDED_DISCOVERY_BH_FDR_ALL_EVALUATED__UNIQUE_PHYSICAL_GAMES',
              'validation_policy':'2024_2025_SEALED__HOLDOUT_FDR__PIT_CERTIFICATION_REQUIRED',
              'core_policy':'OOF_CORE_EDGE_IF_PRESENT__DESCRIPTIVE_CONDITIONAL_LIFT_NOT_CAUSAL',
              'production_authority':0,'automatic_promotion':True,
              'promotion_policy':'60_UNAMBIGUOUS_GAMES__53PCT_MIN__HOLDOUT_BH_FDR_Q10__PIT_CERTIFICATE__EXACT_LIVE_ATOMS',
              'selection_influence':0}
    log_func('[NCAAF-DEEP-CFBD] status=PASS columns=%d admit=%d tested=%d max_depth=%d shortlisted=%d holdout_screen=%d evidence_pass=%d blocked_pit=%d auto_ready=%d' %
             (len(inventory), len(output['features_admitted']), counts['tested_hypotheses'],
              max_depth, len(shortlist), output['holdout_screen_pass'], evidence_pass,
              blocked_pit,len(promotions)))
    return output
