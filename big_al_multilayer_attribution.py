"""Read-only Big Al observation pattern diagnostics for NFL and NCAAF.

Ratings suggest hypotheses only. No same-game outcomes, no automatic mining
qualification, no betting votes. Incomplete cards do NOT establish negative
examples from unselected games.
"""
from __future__ import annotations
from collections import Counter,defaultdict

VERSION='BIG_AL_LAYER_FINGERPRINT_20261018_V1'


def _atom_family(atom):
    """Coarse conceptual grouping, not a guarantee of causal independence."""
    s=str(atom).upper()
    for family,terms in (
        ('MARKET',('SPREAD_','TOTAL_','KEY_','LINE_','DOG_','FAVORITE_')),
        ('BOUNCEBACK',('BOUNCE','UPSET','ATS','SHUTOUT','PRIOR_LOSS','SU_LOSS')),
        ('MATCHUP',('DEF_','OFF_','MATCHUP','YPP','PPA')),
        ('REVENGE_RIVALRY',('REVENGE','RIVALRY','H2H_')),
        ('REST_TRAVEL',('REST_','TRAVEL_','BYE_','TIME_ZONE')),
        ('SEASON_CONTEXT',('RECORD','WIN_PCT','SEASON','WEEK','GAME_NO')),
        ('CONSENSUS',('PT_','CONSENSUS','EXTERNAL_')),
    ):
        if any(t in s for t in terms):return family
    return 'OTHER'


def _obs_key(x):
    sport=str(x.get('sport') or '').upper()
    market=str(x.get('market') or '').lower()
    date=str(x.get('game_date') or '')
    line=x.get('line')
    try: line=round(abs(float(line)),1)
    except (ValueError,TypeError):line=None
    if market=='spreads':
        role='DOG' if x.get('line') is not None and float(x['line'])>0 else 'FAV'
        if line==0:role='PICK'
    else:role=str(x.get('selection_direction') or '').upper()
    return sport,date,market,role,line


def analyze_observed_ratings(observations, fingerprints=None):
    fingerprints=fingerprints or []
    by_id={str(x.get('observation_id')):x for x in fingerprints if isinstance(x,dict)}
    rows=[]
    for obs in observations:
        if not isinstance(obs,dict):continue
        fp=by_id.get(str(obs.get('observation_id')),{})
        rating=obs.get('rating')
        if rating not in (2,6,10):continue
        atoms=set(fp.get('active_atoms') or [])
        layers=sorted({_atom_family(a) for a in atoms})
        rows.append({'observation_id':obs.get('observation_id'),
                     'sport':obs.get('sport'),'game_date':obs.get('game_date'),
                     'market':obs.get('market'),'line':obs.get('line'),'rating':rating,
                     'team':obs.get('selection_team'),
                     'match_status':fp.get('context_match','NOT_EVALUATED'),
                     'atoms':sorted(atoms),'conceptual_layers':layers,
                     'layer_count':len(layers) if atoms else None})
    # Matched comparisons suggest which *independent families* to test with
    # historical Miner logic. They DO NOT fit the 2026 game result or use the
    # expert rating to move the current betting decision.
    contrasts=[]
    for sport in ('NCAAF','NFL'):
        group=[x for x in rows if x['sport']==sport]
        higher=[x for x in group if x['rating']>=6]
        lower=[x for x in group if x['rating']==2]
        for high in higher:
            candidates=[]
            hk=_obs_key(high)
            for low in lower:
                lk=_obs_key(low)
                if hk[2]!=lk[2] or hk[3]!=lk[3]:continue
                if hk[4] is None or lk[4] is None:continue
                distance=abs(hk[4]-lk[4])
                threshold=5 if hk[2]=='spreads' else 8
                if distance>threshold:continue
                candidates.append((distance,low))
            for dist,low in sorted(candidates,key=lambda z:z[0])[:2]:
                hfam=set(high['conceptual_layers']);lfam=set(low['conceptual_layers'])
                comparable=high['layer_count'] is not None and low['layer_count'] is not None
                contrasts.append({'sport':sport,'higher':high['observation_id'],
                    'lower':low['observation_id'],'ratings':[high['rating'],low['rating']],
                    'same_market':True,'same_side_role':True,'line_gap':dist,
                    'matched_pregame_atoms':bool(comparable),
                    'added_conceptual_layers':sorted(hfam-lfam) if comparable else [],
                    'lower_only_conceptual_layers':sorted(lfam-hfam) if comparable else [],
                    'hypothesis_status':'FINGERPRINT_READY_NEEDS_HISTORICAL_ABLATION'
                                         if comparable else 'AWAITING_PRE_GAME_FINGERPRINT'})
    counts=Counter((x['sport'],x['rating']) for x in rows)
    same_game=defaultdict(list)
    for o in observations:
        pair=sorted([str(o.get('team_a') or '').strip(),str(o.get('team_b') or '').strip()])
        if all(pair):
            key=(str(o.get('sport')),str(o.get('game_date')),*pair)
            same_game[key].append({'observation_id':o.get('observation_id'),
                                   'market':o.get('market'),'rating':o.get('rating')})
    paired_cards=[{'game':list(k),'picks':v} for k,v in same_game.items() if len(v)>1]
    return {'status':'RESEARCH_DESCRIPTIVE_ONLY','version':VERSION,'rated_observations':len(rows),
        'counts_by_sport_rating':{f'{sp}_{rt}':n for (sp,rt),n in counts.items()},
        'fingerprints_matched':sum(x['layer_count'] is not None for x in rows),
        'matched_rating_comparisons':contrasts[:60],
        'same_game_spread_total_stacks':paired_cards,
        'comparisons_are_causal_or_confirmed':False,
        'unselected_games_treated_as_negative_examples':False,
        'card_outcomes_used':False,'rating_qualification_weight':0,
        'miner_votes':0,'production_mutation':False,
        'next_step':'Existing Miner tests independent parent-child conditions across historical games; stars seed hypotheses only.'}
