"""Permanent NCAAF research catalog integrity and independent scenario engine.

Invoked by the NCAAF Heavy research pipeline AFTER incumbent reconciliation and
BEFORE any research artifact or downstream system summary is generated.

The historic research library is preserved; only recycled IDs receive stable,
unique rule-definition identities. Research results have ZERO betting authority.
Retrospective rules, original quote times, actual Big Al selections, and 2024/25
model-selection independence are NOT certified by this module.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json

import pandas as pd

VERSION = 'NCAAF_MINER_PERMANENT_INTEGRITY_V4_20261024'
PERIODS = {'DISCOVERY': range(2013, 2022), 'SELECTION': range(2022, 2024),
           'RETROSPECTIVE_CONFIRMATION': range(2024, 2026)}


def canonical_conditions(value):
    if isinstance(value, str):
        raw = value.split(' AND ')
    else:
        raw = value or []
    return tuple(sorted({str(a).strip().upper() for a in raw if str(a).strip()}))


def definition_key(market, direction, conditions):
    normalized = '|'.join((str(market).strip().lower(), str(direction).strip().upper(),
                           '&'.join(canonical_conditions(conditions))))
    return sha256(normalized.encode('utf-8')).hexdigest()


def information_sources(conditions):
    """Source membership; unknown atoms fail closed to one conservative bucket.

    A rule depending on multiple providers bridges all providers into one
    correlated component. Sharing the same atom also creates dependency.
    """
    tags = set()
    for a in canonical_conditions(conditions):
        if a.startswith(('EXPERT_PATHI', 'PATHI_')):
            tags.add('PATHI_CONTEXT')
        elif a.startswith(('EXPERT_BIGAL', 'OBS_BIGAL', 'BIG_AL_')):
            tags.add('BIG_AL_STYLE_NOT_ORIGINAL_PICKS')
        elif a.startswith(('PTIDX_', 'META_PT_', 'PT_', 'EXTERNAL_')):
            tags.add('PT_EXTERNAL_RATINGS')
        elif a.startswith(('CFBD_PORTAL_', 'CFBD_RETURNING_', 'CFBD_QB_RETURNING',
                           'CFBD_HIGH_USAGE', 'CFBD_TOP_PPA_PLAYER', 'CFBD_TRANSFER_',
                           'CFBD_ROSTER_', 'CFBD_TALENT_')):
            tags.add('CFBD_PERSONNEL')
        elif a.startswith(('CFBD_HAVOC_', 'CFBD_NET_', 'CFBD_QB_PPA_',
                           'CFBD_SACK_RATE_', 'CFBD_SUCCESS_', 'CFBD_EPA_')):
            tags.add('CFBD_ADVANCED')
        elif 'MATCHUP' in a:
            tags.add('MATCHUP_DIFFERENTIAL')
        elif 'TRAVEL' in a or 'TIME_ZONE' in a or a.startswith(('TEAM_SOS', 'OPP_SOS')):
            tags.add('SCHEDULE_SOS')
        elif 'H2H' in a or 'REVENGE' in a:
            tags.add('H2H_REVENGE')
        elif a.startswith('STAT_'):
            tags.add('STAT')
    return frozenset(tags or {'UNKNOWN_SOURCE'})


def repair_recycled_ids(miners):
    """Assign immutable distinct definition IDs and repair recycled legacy IDs.

    The same exact definition always maps to the same ID, regardless of input
    order. Existing unique legacy IDs are retained for historical compatibility.
    Recycled IDs are remapped to a SHA-256-based ID with a legacy alias; both
    conflicting definitions are quarantined for new live-system promotion.
    """
    grouped = defaultdict(list)
    all_mechs = []
    for market, registry in sorted((miners or {}).items()):
        for m in (registry or {}).get('mechanism_families', []):
            mid = str(m.get('mechanism_id') or '').strip()
            if not mid or not m.get('representative_conditions'):
                raise ValueError('INVALID_MINER_MECHANISM_DEFINITION')
            digest = definition_key(market, m.get('direction'), m.get('representative_conditions'))
            m['immutable_definition_id'] = 'NCAAF-DEF-' + digest[:24].upper()
            m['definition_sha256'] = digest
            m['source_dependencies'] = sorted(information_sources(m['representative_conditions']))
            m['legacy_mechanism_id'] = str(m.get('legacy_mechanism_id') or mid)
            grouped[(str(market), mid)].append(m)
            all_mechs.append((str(market), m))
    # Recycled legacy IDs always fail closed, even if the two records have
    # identical content: duplicate entries cannot manufacture extra evidence.
    legacy_collisions = {k for k, mm in grouped.items() if len({z['definition_sha256'] for z in mm}) > 1}
    records = []
    used = set()
    for market, m in all_mechs:
        orig = str(m['mechanism_id'])
        is_collision = (market, orig) in legacy_collisions
        new_id = m['immutable_definition_id'] if is_collision else orig
        k = (market, new_id)
        if k in used:
            raise ValueError('DUPLICATE_EXACT_DEFINITION_IN_CATALOG:' + market + ':' + new_id)
        used.add(k)
        m['mechanism_id'] = new_id
        m['id_integrity_state'] = ('REPAIRED_LEGACY_COLLISION_RESEARCH_HOLD' if is_collision else 'UNIQUE_VERIFIED')
        m['legacy_id_collision'] = bool(is_collision)
        # Recycled identifiers have ambiguous historical attribution. They
        # remain in the catalog and shadow analyses, but must not inherit
        # grandfathered live authority on the basis of that ambiguous ID.
        if is_collision:
            m['live_authority_eligible'] = False
            m['current_qualified'] = False
            m['live_authority_collision_hold'] = True
        if str(m.get('evidence_family_key') or '') == orig and is_collision:
            m['evidence_family_key'] = new_id
        records.append({'market':market,'legacy_mechanism_id':orig,
                        'mechanism_id':new_id,'immutable_definition_id':m['immutable_definition_id'],
                        'definition_sha256':m['definition_sha256'],
                        'id_collision_repaired':is_collision,
                        'new_live_authority_blocked':is_collision,
                        'source_dependencies':'|'.join(m['source_dependencies']),
                        'conditions':' AND '.join(canonical_conditions(m['representative_conditions']))})
    return {'version':VERSION, 'status':'PASS', 'rules':len(records),
            'recycled_legacy_id_groups':len(legacy_collisions),
            'canonical_ids_created':sum(z['id_collision_repaired'] for z in records),
            'legacy_alias_map':records, 'production_model_mutation':False,
            'automatic_promotion':False}


def _normalized_events(firings, catalog):
    needed = {'season','source_game_id','market','target_team','mechanism_id','direction','conditions'}
    if firings is None or firings.empty:
        return pd.DataFrame(columns=sorted(needed | {'evidence_family_key','definition_sha256','source_dependencies'}))
    if not needed.issubset(firings.columns):
        raise ValueError('EXACT_FIRINGS_REQUIRED_COLUMNS_MISSING:' + str(sorted(needed-set(firings.columns))))
    f = firings.copy()
    f['season'] = pd.to_numeric(f['season'], errors='raise').astype(int)
    f = f[f.season.between(2013, 2025)].copy()
    f['source_game_id'] = f.source_game_id.astype(str).str.strip()
    f['target_team'] = f.target_team.astype(str).str.strip().str.lower()
    f = f[f.source_game_id.ne('') & f.target_team.ne('') & f.source_game_id.ne('nan')]
    f['definition_sha256'] = [definition_key(m,d,c) for m,d,c in f[['market','direction','conditions']].itertuples(index=False,name=None)]
    f['source_dependencies'] = f.conditions.map(information_sources)
    f['atoms_v4'] = f.conditions.map(canonical_conditions)
    if catalog is not None and not catalog.empty and 'definition_sha256' in catalog.columns:
        # Fail-closed join: never accept a reused ID without identical definition.
        c = catalog[['market','mechanism_id','definition_sha256']].drop_duplicates()
        if c.duplicated(['market','mechanism_id','definition_sha256']).any():
            raise ValueError('REPEATED_CATALOG_DEFINITION')
        f = f.merge(c, on=['market','mechanism_id','definition_sha256'], how='inner',validate='many_to_one')
    return f.drop_duplicates(['season','source_game_id','market','target_team','definition_sha256'])


def collapse_source_components(group):
    """Return independent direction blocks within one physical source-game.

    Dependence is transitive. A Pathi+PT joint rule joins Pathi and PT-only
    rules into one source component, not two independent supporters.
    """
    rows=group.to_dict('records')
    parent=list(range(len(rows)))
    def root(i):
        while i != parent[i]:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def link(i,j):
        a,b=root(i),root(j)
        if a!=b: parent[b]=a
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            a,b=rows[i],rows[j]
            if (str(a.get('evidence_family_key','')) == str(b.get('evidence_family_key',''))
                or bool(set(a['atoms_v4']) & set(b['atoms_v4']))
                or bool(a['source_dependencies'] & b['source_dependencies'])):
                link(i,j)
    comp=defaultdict(list)
    for i,r in enumerate(rows): comp[root(i)].append(r)
    out=[]
    for members in comp.values():
        targets=sorted({r['target_team'] for r in members})
        out.append({'target_team': targets[0] if len(targets)==1 else 'CONFLICT',
                    'sources':tuple(sorted(set().union(*(r['source_dependencies'] for r in members)))),
                    'mechanism_ids':tuple(sorted({r['mechanism_id'] for r in members})),
                    'definition_ids':tuple(sorted({r['definition_sha256'][:24] for r in members})),
                    'member_count':len(members),
                    'representative_conditions':tuple(sorted(members,key=lambda r:r['definition_sha256'])[0]['atoms_v4']),
                    'representative_direction':str(sorted(members,key=lambda r:r['definition_sha256'])[0]['direction'])})
    return out


def collapse_live_source_votes(votes):
    """Conservative permanent vote repair: can REMOVE, never add, live votes.

    Unavailable evidence is not reconstructed. A shared source, identical atom,
    or reported evidence family makes votes dependent. Opposing sides of the
    same dependent component abstain. This only guards Miner confirmation;
    it does NOT independently qualify any rule for live betting.
    """
    groups=defaultdict(list)
    for v in votes or []:
        groups[str(v.get('market') or '').lower()].append(dict(v))
    kept=[]
    for market, items in groups.items():
        records=[]
        for item in items:
            rule=str(item.get('rule') or '')
            family=str(item.get('evidence_family_key') or item.get('family_id') or '')
            src=set(information_sources(rule))
            if family in ('EXTERNAL_RATINGS_FAMILY','PT_EXTERNAL_RATINGS') or item.get('uses_external_ratings_family'):
                src.discard('UNKNOWN_SOURCE');src.add('PT_EXTERNAL_RATINGS')
            item['_source_tags']=frozenset(src)
            item['_atoms']=canonical_conditions(rule)
            records.append(item)
        parent=list(range(len(records)))
        def root(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i in range(len(records)):
            for j in range(i+1,len(records)):
                a,b=records[i],records[j]
                if (str(a.get('evidence_family_key') or a.get('family_id') or '')==
                        str(b.get('evidence_family_key') or b.get('family_id') or '')
                    or bool(set(a['_atoms'])&set(b['_atoms']))
                    or bool(a['_source_tags']&b['_source_tags'])):
                    x,y=root(i),root(j)
                    if x!=y: parent[y]=x
        components=defaultdict(list)
        for i,v in enumerate(records):components[root(i)].append(v)
        for members in components.values():
            targets={str(z.get('target') or '').lower().strip() for z in members}
            if len(targets)!=1 or '' in targets:
                continue
            def rank(v):
                try: rate=float(v.get('confirmation_rate') or 0)
                except (TypeError,ValueError):rate=0
                try: n=int(v.get('confirmation_n') or 0)
                except (TypeError,ValueError):n=0
                return (rate,n,str(v.get('family_id') or ''))
            best=max(members,key=rank).copy()
            best.pop('_atoms',None);best.pop('_source_tags',None)
            best['source_dependency_v4']='SOURCE_ATOM_CONNECTED_COMPONENT'
            best['source_dependency_component_size']=len(members)
            best['source_dependencies']=sorted(set().union(*(z['_source_tags'] for z in members)))
            best['correlated_family_candidates']=len(members)
            best['family_vote_collapsed']=len(members)>1
            kept.append(best)
    return kept


def _period(year):
    for name, years in PERIODS.items():
        if year in years:return name
    return 'EXCLUDED'


def research_scenarios(firings, catalog):
    """Produce exact-rule scenario interactions without posthoc bet promotion.

    Pair discovery uses only 2013-2021 historical co-fire *counts*, not later
    results. 2022-25 counts are descriptive, and 2026 is sealed. These rules
    themselves were created in hindsight, so no authentic holdout is asserted.
    """
    f = _normalized_events(firings,catalog)
    summaries,components=[] ,[]
    # Strict source-game id + season scoped; this is not a CFBD identity claim.
    pair_seasons=defaultdict(lambda:defaultdict(set))
    rule_pair_seasons=defaultdict(lambda:defaultdict(set))
    for (y,gid,market), group in f.groupby(['season','source_game_id','market'],sort=False):
        blocks=collapse_source_components(group)
        support=defaultdict(list)
        for b in blocks:
            components.append({'season':int(y),'source_game_id':str(gid),'market':market,
                               'target_team':b['target_team'],
                               'sources':'|'.join(b['sources']),
                               'mechanism_ids':'|'.join(b['mechanism_ids']),
                               'definition_ids':'|'.join(b['definition_ids']),
                               'member_count':b['member_count'],
                               'source_game_identity_verified':False,
                               'bet_authority':0})
            if b['target_team']!='CONFLICT':support[b['target_team']].append(b)
        targets={b['target_team'] for b in blocks if b['target_team']!='CONFLICT'}
        has_conflict=any(b['target_team']=='CONFLICT' for b in blocks)
        for target, ss in support.items():
            summaries.append({'season':int(y),'source_game_id':str(gid),'market':market,
                              'target_team':target,'independent_components':len(ss),
                              'internally_conflicted_components':sum(b['target_team']=='CONFLICT' for b in blocks),
                              'opposing_targets':int(len(targets - {target})),
                              'unanimous_two_plus':len(ss)>=2 and len(targets)==1 and not has_conflict,
                              'period':_period(int(y)), 'source_game_identity_verified':False,
                              'bet_authority':0})
            if len(ss)<2 or len(targets)>1 or has_conflict:
                continue
            for a,b in combinations(ss,2):
                # Discover source pathways, not fake independent named rules.
                sa, sb=('+'.join(a['sources']),'+'.join(b['sources']))
                key=(market,*sorted((sa,sb)))
                pair_seasons[key][int(y)].add(str(gid))
                # A true new pair of exact predicates on two firing source rows.
                # Not assumed to be a single-row AND or a certified pregame path.
                ra=('+'.join(a['representative_conditions']),a['representative_direction'])
                rb=('+'.join(b['representative_conditions']),b['representative_direction'])
                rk=(market,*sorted((ra,rb)))
                rule_pair_seasons[rk][int(y)].add(str(gid))
    candidates, season_rows=[],[]
    for key,byyear in sorted(pair_seasons.items()):
        disc=sum(len(byyear.get(y,set())) for y in PERIODS['DISCOVERY'])
        select=sum(len(byyear.get(y,set())) for y in PERIODS['SELECTION'])
        conf=sum(len(byyear.get(y,set())) for y in PERIODS['RETROSPECTIVE_CONFIRMATION'])
        keytext='|'.join(key)
        pathid='NCAAF-PATH-V4-'+sha256(keytext.encode()).hexdigest()[:16].upper()
        # Do not select a path on 2022-25 data. Low-frequency paths are
        # retained as descriptive candidates with explicit non-promotion.
        candidates.append({'path_id':pathid,'market':key[0],'source_path_a':key[1],
                           'source_path_b':key[2],'discovery_2013_2021_games':disc,
                           'selection_2022_2023_games':select,
                           'retrospective_2024_2025_games':conf,
                           'discovery_frequency_qualified':disc>=20,
                           'genuinely_oos_validated':False,
                           'original_big_al_picks_replayed':False,
                           'bet_authority':0})
        for year,count in sorted((y,len(ids)) for y,ids in byyear.items()):
            season_rows.append({'path_id':pathid,'season':year,'games_cofired':count,
                                'period':_period(year),'bet_authority':0})
    joint_rules=[]
    for key,byyear in sorted(rule_pair_seasons.items()):
        d=sum(len(byyear.get(y,set())) for y in PERIODS['DISCOVERY'])
        s=sum(len(byyear.get(y,set())) for y in PERIODS['SELECTION'])
        c=sum(len(byyear.get(y,set())) for y in PERIODS['RETROSPECTIVE_CONFIRMATION'])
        sig=json.dumps(key,sort_keys=True)
        joint_rules.append({'candidate_id':'NCAAF-RULE-PAIR-'+sha256(sig.encode()).hexdigest()[:16].upper(),
                            'market':key[0], 'first_rule_atoms':key[1][0],
                            'first_direction':key[1][1],
                            'second_rule_atoms':key[2][0],
                            'second_direction':key[2][1],
                            'discovery_2013_2021_games':d,
                            'selection_2022_2023_games':s,
                            'retrospective_2024_2025_games':c,
                            'discovery_frequency_qualified':d>=20,
                            'pair_definition':'TWO_SOURCE_ROWS_MUST_COFIRE_SAME_PHYSICAL_GAME_AND_TARGET',
                            'new_rule_tested_on_independent_holdout':False,
                            'bet_authority':0})
    hybrid=[]
    special=[]
    if catalog is not None and not catalog.empty:
        for row in catalog.itertuples():
            src=information_sources(row.conditions)
            origin=str(getattr(row,'path_origin','') or '')
            bigal_style=origin.startswith('BIG_AL')
            pathi=('PATHI_CONTEXT' in src or origin=='BIG_AL_PATHI_INTERACTION')
            if len(src)>=2:
                hybrid.append({'market':row.market,'mechanism_id':row.mechanism_id,
                               'sources':'|'.join(sorted(src)),'conditions':row.conditions,
                               'path_origin':origin,
                               'big_al_real_pick_evidence':False,'bet_authority':0})
            if bigal_style or pathi:
                special.append({'market':row.market,'mechanism_id':row.mechanism_id,
                                'hypothesis_kind':('BIG_AL_STYLE_AND_PATHI' if bigal_style and pathi else
                                    'BIG_AL_STYLE_ONLY' if bigal_style else 'PATHI_CONTEXT_ONLY'),
                                'sources':'|'.join(sorted(src)),'conditions':row.conditions,
                                'big_al_actual_pick_archive_matched':False,
                                'independent_historical_holdout':False,'bet_authority':0})
    report={'version':VERSION,'status':'RETROSPECTIVE_RESEARCH_ONLY',
            'catalog_exact_firings':len(f), 'catalog_mechanisms':len(catalog) if catalog is not None else None,
            'source_game_target_rows':len(summaries),'dependency_components':len(components),
            'interaction_paths':len(candidates),
            'exact_rule_cofire_pairs':len(joint_rules),
            'cross_source_hybrid_catalog_rules':len(hybrid),
            'big_al_pathi_style_hypotheses':len(special),
            'discovery_qualified_paths':sum(z['discovery_frequency_qualified'] for z in candidates),
            'market_quote_vintage_verified':False,'rule_vintage_verified':False,
            'independent_historical_holdout':False,
            'production_model_mutation':False,'automatic_promotion':False,'bet_authority':0}
    return {'integrity_v4_audit':report,
            'source_dependency_components_v4':pd.DataFrame(components),
            'source_dependency_game_states_v4':pd.DataFrame(summaries),
            'miner_interaction_paths_v4':pd.DataFrame(candidates),
            'miner_interaction_path_seasons_v4':pd.DataFrame(season_rows),
            'miner_exact_rule_cofire_candidates_v4':pd.DataFrame(joint_rules),
            'miner_cross_source_hybrid_rules_v4':pd.DataFrame(hybrid),
            'miner_bigal_pathi_style_hypotheses_v4':pd.DataFrame(special)}


def run_regression_from_csv(mechanisms_csv, firings_csv, output_dir):
    """Re-run the V4 research outputs from prior exports, without Heavy."""
    from pathlib import Path
    directory=Path(output_dir); directory.mkdir(parents=True,exist_ok=True)
    catalog=pd.read_csv(mechanisms_csv)
    firings=pd.read_csv(firings_csv)
    catalog['definition_sha256']=[definition_key(*r) for r in catalog[['market','direction','conditions']].itertuples(index=False,name=None)]
    # Prior exported colliding IDs must be rebuilt with market + full definition.
    colliding=(catalog.groupby(['market','mechanism_id']).definition_sha256.nunique()>1)
    aliases=[]
    for r in catalog.itertuples():
        old=str(r.mechanism_id)
        collision=bool(colliding.get((str(r.market),old),False))
        new='NCAAF-DEF-'+r.definition_sha256[:24].upper() if collision else old
        aliases.append({'market':r.market,'old_id':old,'new_id':new,
                        'definition_sha256':r.definition_sha256,'collision':collision})
    alias=pd.DataFrame(aliases)
    for frame in (firings,catalog):
        frame['definition_sha256']=[definition_key(*r) for r in frame[['market','direction','conditions']].itertuples(index=False,name=None)]
        frame['legacy_mechanism_id']=frame.mechanism_id
        frame.drop(columns=['mechanism_id'],inplace=True)
    # Explicit assignment, not implicit in-place pandas merge.
    for frame in (firings,catalog):
        frame['mechanism_id']=['NCAAF-DEF-'+h[:24].upper() if colliding.get((str(m),str(mid)),False) else str(mid)
            for m,mid,h in frame[['market','legacy_mechanism_id','definition_sha256']].itertuples(index=False,name=None)]
    output=research_scenarios(firings,catalog)
    for k,v in output.items():
        if isinstance(v,pd.DataFrame):v.to_csv(directory/(k+'.csv'),index=False)
        else:(directory/(k+'.json')).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
    alias.to_csv(directory/'mechanism_legacy_id_crosswalk_v4.csv',index=False)
    return output['integrity_v4_audit']


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--mechanisms',required=True)
    p.add_argument('--firings',required=True)
    p.add_argument('--out-dir',required=True)
    a=p.parse_args()
    print(json.dumps(run_regression_from_csv(a.mechanisms,a.firings,a.out_dir),sort_keys=True))
