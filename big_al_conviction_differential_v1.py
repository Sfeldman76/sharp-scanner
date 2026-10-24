"""Big Al selected-card conviction research, NCAAF only. Never grants betting authority.

This module deliberately cannot grade outcomes, turn nonselections into negative examples,
or infer archived release/market quotes. CFBD prior fields are reconstructed prior-game
features and require source-vintage inspection before they become certified inputs.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import math
import re
from typing import Mapping

VERSION = 'BIG_AL_CONVICTION_DIFFERENTIAL_SHADOW_V1_20261024'
# Immutable, explicit alias identities; never fuzzy-match or use results to disambiguate.
_ALIASES = {
    'centralflorida': 'ucf', 'southcarolina': 'southcarolina',
    'louisianamonroe': 'ulmonroe', 'louisianalafayette': 'louisiana',
    'san diegostate': 'sandiegostate', 'sandiegost': 'sandiegostate',
    'coloradost': 'coloradostate', 'utahst': 'utahstate',
    'texassanantonio': 'utsa', 'utsa': 'utsa',
    'samhoustonstate': 'samhouston', 'connecticut': 'uconn',
    'appalachianstate': 'appstate', 'kent': 'kentstate',
    'northernillinois': 'northernillinois', 'southernmiss': 'southernmississippi',
    'southernmississippi': 'southernmississippi',
    'lsu': 'lsu', 'olemiss': 'olemiss', 'mississippi': 'olemiss',
    'miamiohio': 'miamioh',
}
# Only named lagged inputs. Do not read final scores, outcomes, current-game CFBD_Game_*
# columns, current quote/close, or completed-game efficiency in the fingerprint.
PRIOR_FIELDS = (
    'CFBD_Games_Prior', 'CFBD_Elo_Prior',
    'CFBD_Season_Net_PPA_Prior', 'CFBD_Recent3_Net_PPA_Prior',
    'CFBD_Season_Off_PPA_Prior', 'CFBD_Season_Def_PPA_Prior',
    'CFBD_Season_Off_Success_Prior', 'CFBD_Season_Def_Success_Prior',
    'CFBD_Recent3_Off_PPA_Prior', 'CFBD_Recent3_Def_PPA_Prior',
    'CFBD_Season_Off_Plays_Prior', 'CFBD_Season_Def_Plays_Prior',
)


def school(x):
    s = re.sub(r'[^a-z0-9]+', '', str(x or '').lower())
    return _ALIASES.get(s, s)


def day(x):
    if not x: return None
    try: return date.fromisoformat(str(x)[:10])
    except (ValueError, TypeError): return None


def number(x):
    try:
        a = float(x)
        return a if math.isfinite(a) else None
    except (ValueError, TypeError): return None


def boolean(x):
    if x is True or x == 1: return True
    if x is False or x == 0: return False
    if isinstance(x, str):
        if x.strip().lower() in {'true', 't', '1'}: return True
        if x.strip().lower() in {'false', 'f', '0'}: return False
    return None


def _timestamp(x):
    if not x: return None
    if isinstance(x, datetime): d=x
    else:
        try: d = datetime.fromisoformat(str(x).replace('Z', '+00:00'))
        except (ValueError, TypeError): return None
    if d.tzinfo is None: return None
    return d.astimezone(timezone.utc)


def _side_groups(side_rows):
    groups=defaultdict(list)
    for row in side_rows:
        season=number(row.get('Season'))
        gid=str(row.get('Game_ID') or '').strip()
        if season is None or not gid: continue
        groups[(int(season),gid)].append(row)
    games=[]; rejected=Counter()
    for (season,gid),rows in groups.items():
        if len(rows)!=2 or {boolean(x.get('Is_Home')) for x in rows}!={True,False}:
            rejected['NOT_TWO_OPPOSITE_SIDES']+=1;continue
        h=next(x for x in rows if boolean(x.get('Is_Home')) is True)
        a=next(x for x in rows if boolean(x.get('Is_Home')) is False)
        if not school(h.get('Team')) or not school(a.get('Team')) or school(h.get('Team'))==school(a.get('Team')):
            rejected['INVALID_SCHOOL']+=1;continue
        if school(h.get('Opponent'))!=school(a.get('Team')) or school(a.get('Opponent'))!=school(h.get('Team')):
            rejected['SIDE_OPPONENT_CONFLICT']+=1;continue
        dates={day(x.get('Game_Date')) for x in rows}
        if len(dates)!=1 or None in dates:
            rejected['SIDE_DATE_CONFLICT']+=1;continue
        games.append({'season':season,'game_id':gid,'day':dates.pop(),'home':h,'away':a})
    return games,dict(rejected)


def _identity(obs, games):
    od=day(obs.get('game_date'))
    if od is None: return None,'INVALID_CARD_DATE'
    market=str(obs.get('market') or '').lower()
    if market=='spreads':
        sel=school(obs.get('selection_team'))
        if not sel: return None,'MISSING_SELECTION_TEAM'
        candidate=lambda g: sel in (school(g['home']['Team']), school(g['away']['Team']))
    elif market=='totals':
        pair={school(obs.get('team_a')),school(obs.get('team_b'))}
        if len(pair)!=2 or '' in pair:return None,'INCOMPLETE_TOTAL_TEAMS'
        candidate=lambda g: pair=={school(g['home']['Team']),school(g['away']['Team'])}
    else:return None,'UNSUPPORTED_MARKET'
    # Exact date first; UTC/local +/-1 day only if the candidate is unique.
    possibles=[g for g in games if g['season']==od.year and candidate(g)]
    exact=[g for g in possibles if g['day']==od]
    if len(exact)==1: return exact[0],'EXACT_CARD_DATE_AND_SCHOOL'
    if len(exact)>1: return None,'AMBIGUOUS_EXACT_MATCH'
    nearby=[g for g in possibles if abs((g['day']-od).days)<=1]
    if len(nearby)==1:return nearby[0],'UNIQUE_UTC_DATE_SHIFT_AND_SCHOOL'
    if len(nearby)>1:return None,'AMBIGUOUS_DATE_SHIFT'
    return None,'NO_MATCHING_CFBD_GAME'


def _lagged_features(row):
    return {key:number(row.get(key)) for key in PRIOR_FIELDS}


def _delta(a,b,key):
    x,y=a.get(key),b.get(key)
    return None if x is None or y is None else round(x-y,5)


def _fingerprint(obs, game, route):
    market=str(obs.get('market') or '')
    selected=None
    if market=='spreads':
        selected=game['home'] if school(obs.get('selection_team'))==school(game['home']['Team']) else game['away']
    other=None if selected is None else (game['away'] if selected is game['home'] else game['home'])
    hs,as_=_lagged_features(game['home']),_lagged_features(game['away'])
    ss,os_=(hs,as_) if selected is game['home'] else (as_,hs) if selected is game['away'] else ({},{})
    line=number(obs.get('line'))
    flat={
        'market':market,
        'observed_line':line,
        'selected_is_home':None if selected is None else (selected is game['home']),
        'spread_role':('FAVORITE' if line < 0 else 'DOG' if line > 0 else 'PICK') if market=='spreads' and line is not None else None,
        'favorite_key_3_or_7':market=='spreads' and line is not None and line<0 and (abs(abs(line)-3)<=0.5 or abs(abs(line)-7)<=0.5),
        'dog_seven_plus':market=='spreads' and line is not None and line>=7,
        'under_total_59_plus':market=='totals' and str(obs.get('selection_direction') or '').upper()=='UNDER' and line is not None and line>=59,
        'selected_vs_opp_season_net_ppa':_delta(ss,os_,'CFBD_Season_Net_PPA_Prior') if selected is not None else None,
        'selected_vs_opp_elo':_delta(ss,os_,'CFBD_Elo_Prior') if selected is not None else None,
        'selected_vs_opp_off_success':_delta(ss,os_,'CFBD_Season_Off_Success_Prior') if selected is not None else None,
        'selected_off_ppa_vs_opp_def_ppa':(round(ss['CFBD_Season_Off_PPA_Prior']-os_['CFBD_Season_Def_PPA_Prior'],5)
             if selected is not None and ss.get('CFBD_Season_Off_PPA_Prior') is not None
                  and os_.get('CFBD_Season_Def_PPA_Prior') is not None else None),
        'selected_recent3_minus_season_net_ppa':_delta(ss,{'CFBD_Recent3_Net_PPA_Prior':ss.get('CFBD_Season_Net_PPA_Prior')},'CFBD_Recent3_Net_PPA_Prior') if selected is not None else None,
        'home_vs_away_prior_net_ppa':_delta(hs,as_,'CFBD_Season_Net_PPA_Prior'),
        'home_vs_away_prior_elo':_delta(hs,as_,'CFBD_Elo_Prior'),
        'home_vs_away_prior_play_volume':_delta(hs,as_,'CFBD_Season_Off_Plays_Prior'),
    }
    # We only know the outcome-free reconstruction by its data contents; not the
    # date at which a bookmaker or user could actually have received it.
    return {'observation_id':obs.get('observation_id'),'rating':obs.get('rating'),
            'game_date':obs.get('game_date'),'game_id':game['game_id'],
            'home':game['home']['Team'],'away':game['away']['Team'],
            'selected_side':obs.get('selection_team') or obs.get('selection_direction'),
            'identity_method':route,'match_status':'MATCHED_RETROSPECTIVE_PRIOR_FEATURES',
            'feature_source':'NCAAF_CFBD_MINER_FEATURES_PRIOR_FIELDS',
            'original_feature_vintage_certified':False,
            'original_big_al_release_time_certified':False,
            'features':flat,'selected_prior':ss or None,'opponent_prior':os_ or None,
            'home_prior':hs,'away_prior':as_}


def _comparisons(matched):
    high=[r for r in matched if number(r.get('rating')) is not None and number(r['rating'])>=6]
    low=[r for r in matched if number(r.get('rating'))==2]
    out=[];seen=set()
    for hi in high:
        a=hi['features']; ap=a.get('observed_line')
        if ap is None:continue
        eligible=[]
        for lo in low:
            b=lo['features']; bp=b.get('observed_line')
            if hi['game_id']==lo['game_id'] or bp is None: continue
            if a['market']!=b['market']:continue
            if a['market']=='spreads' and a['spread_role']!=b['spread_role']:continue
            if a['market']=='totals' and hi['selected_side']!=lo['selected_side']:continue
            gap=abs(abs(ap)-abs(bp))
            if gap>(1.0 if a['market']=='spreads' else 2.0):continue
            same_day=hi['game_date']==lo['game_date']
            eligible.append((not same_day,gap,lo))
        for _not_same,gap,lo in sorted(eligible,key=lambda x:(x[0],x[1],str(x[2]['observation_id'])))[:2]:
            pair=(hi['observation_id'],lo['observation_id'])
            if pair in seen:continue
            seen.add(pair)
            keys=('selected_vs_opp_season_net_ppa','selected_vs_opp_elo','selected_vs_opp_off_success',
                  'selected_off_ppa_vs_opp_def_ppa', 'selected_recent3_minus_season_net_ppa','home_vs_away_prior_net_ppa')
            diffs={k:_delta(hi['features'],lo['features'],k) for k in keys}
            out.append({'higher_observation_id':hi['observation_id'],'higher_rating':hi['rating'],
                        'lower_observation_id':lo['observation_id'],'lower_rating':lo['rating'],
                        'market':hi['features']['market'],'price_distance':gap,
                        'same_card_date':hi['game_date']==lo['game_date'],
                        'prior_feature_deltas_high_minus_low':diffs,
                        'interpretation':'DESCRIPTIVE_SELECTED_ONLY_NOT_VALIDATED'})
    return out


def shadow_confirmation(obs, evidence):
    """Fail-closed shadow gate. Timestamped, independent validated support required.

    Evidence is an externally supplied, immutable game/market side-aligned event.
    Never infer it from the presence of a prior feature, a 2026 outcome, or a Big Al star.
    """
    base={'rule':'BIG_AL_HIGH_CONVICTION_CONFIRMATION','observation_id':obs.get('observation_id'),
          'rating':obs.get('rating'),'bet_authority':0,'production_mutation':False,
          'is_executable':False,'status':'UNKNOWN','reason':None}
    if (number(obs.get('rating')) or 0)<6:
        return dict(base,status='NOT_ELIGIBLE',reason='BELOW_6_STARS')
    if str(obs.get('market'))!='spreads':
        return dict(base,status='NOT_ELIGIBLE',reason='SPREAD_RESEARCH_ONLY')
    if not evidence: return dict(base,reason='NO_DATED_INDEPENDENT_EVIDENCE')
    kickoff=_timestamp(evidence.get('game_start_utc'))
    release=_timestamp(evidence.get('big_al_published_at_utc'))
    if not kickoff or not release or release>=kickoff or evidence.get('big_al_release_verified') is not True:
        return dict(base,reason='ORIGINAL_BIG_AL_RELEASE_OR_GAME_START_UNVERIFIED')
    if not evidence.get('game_identity_verified') or not evidence.get('selection_side_verified'):
        return dict(base,reason='PHYSICAL_GAME_OR_SIDE_UNVERIFIED')
    core=evidence.get('core') or {}
    if core.get('supports') is not True or core.get('opposes') is True:
        return dict(base,reason='CORE_NOT_SUPPORTING')
    if not core.get('valid_asof') or not (_timestamp(core.get('recorded_at_utc')) and _timestamp(core.get('recorded_at_utc'))<kickoff):
        return dict(base,reason='CORE_ASOF_NOT_VERIFIED')
    independent=[]; opposite=[]
    for e in evidence.get('independent') or []:
        if e.get('family') not in {'STAT','SYSTEM_MINER'}:continue
        stamp=_timestamp(e.get('recorded_at_utc'))
        if not stamp or stamp>=kickoff or not e.get('source_asof_verified') or not e.get('oos_validated'):
            continue
        if e.get('qualifies') is not True:continue
        if e.get('conflicts') is True: opposite.append(e['family'])
        elif e.get('supports') is True: independent.append(e['family'])
    if opposite:return dict(base,status='VETOED_SHADOW',reason='INDEPENDENT_CONFLICT',opposing_families=sorted(set(opposite)))
    if not independent:return dict(base,reason='NO_QUALIFIED_INDEPENDENT_CONFIRMATION')
    return dict(base,status='CONFIRMED_RESEARCH_SHADOW_ONLY',reason='CORE_PLUS_INDEPENDENT_QUALIFIED_FAMILY',
                confirming_families=sorted(set(independent)),original_big_al_release_utc=release.isoformat())


def study(observations, cfbd_side_rows, *, shadow_evidence=None):
    """Pure analysis for complete selected Big Al ledger; no scores allowed in output."""
    obs=[x for x in observations if x.get('sport')=='NCAAF']
    if len({o.get('observation_id') for o in obs})!=len(obs):
        raise ValueError('DUPLICATE_BIG_AL_OBSERVATION_ID')
    games,invalid=_side_groups(cfbd_side_rows)
    matched=[];unmatched=[]
    for o in obs:
        g,route=_identity(o,games)
        if g:matched.append(_fingerprint(o,g,route))
        else:unmatched.append({'observation_id':o.get('observation_id'), 'game_date':o.get('game_date'),
                               'selection':o.get('selection_team') or [o.get('team_a'),o.get('team_b')],
                               'reason':route})
    cmp=_comparisons(matched)
    evidence=shadow_evidence or {}
    gates=[shadow_confirmation(o,evidence.get(str(o.get('observation_id')))) for o in obs]
    return {'version':VERSION,'status':'RETROSPECTIVE_PREGAME_FEATURE_FINGERPRINT_RESEARCH_ONLY',
            'ncaaf_observations':len(obs),'cfbd_games_in_snapshot':len(games),
            'matched_fingerprints':len(matched),'unmatched_observations':unmatched,
            'rejected_cfbd_game_rows':invalid,'match_routes':dict(Counter(x['identity_method'] for x in matched)),
            'ratings':dict(Counter(str(x.get('rating')) for x in obs)),
            'paired_conviction_contrasts':cmp,'contrasts_computed':len(cmp),
            'fingerprints':matched,'shadow_rule_evaluations':gates,
            'shadow_status_counts':dict(Counter(x['status'] for x in gates)),
            'rule_registry':[{'id':'BIG_AL_CONVICTION_DIFFERENTIAL','role':'MINER_HYPOTHESIS_DISCOVERY',
                'trained_on_2026_results':False,'development_seasons':list(range(2013,2024)),
                'holdout_seasons':[2024,2025],'rule_selection_via_2026_outcomes':False,'authority':0},
              {'id':'BIG_AL_HIGH_CONVICTION_CONFIRMATION','role':'INDEPENDENT_SUPPORT_RESEARCH_SHADOW',
               'min_rating':6,'independent_allowed_families':['STAT','SYSTEM_MINER'],
               'separate_core_requirement':True,'correlated_families_deduplicated':True,'authority':0}],
            'source_vintage_certified':False,'big_al_release_timestamps_certified':False,
            'selected_only_no_negative_controls':True,'outcomes_used':False,
            'no_2026_result_selection':True,'bet_authority':0,'production_mutation':False}
