"""Display-only, NFL-shaped, market-aware NCAAF V5.5 ROI betting board.

Exactly one research authority module and one live authority module exist elsewhere.
This UI imports NEITHER, never scores bets, and does not use legacy V1/CORE
probabilities. It reads the complete SHA-verified BigQuery publication as-is.
"""
from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime, timezone
from collections import Counter

BOARD_TABLE = 'sharplogger.sharp_data.ncaaf_v55_live_board'
REGISTRY_URI = 'gs://sharp-models/research/ncaaf/v55/two_engine_roi/current_registry.json'
POLICY = 'V55-UNIFIED-SEVEN-BRAIN-VALIDATED-BETTING-EDGE-20261106-R1'
REGISTRY_SCHEMA = 'NCAAF-V55-TWO-ENGINE-ROI-REGISTRY-V1'
ACTIONS = frozenset({'BET','STRONG BET'})
MARKETS = ('spreads', 'h2h', 'totals')
BR = {'spreads':'Spr','h2h':'H2H','totals':'Tot'}
# Retains the NFL-style 19-column board, replacing unrelated predicted
# probabilities / model point edges with the requested OOS ROI evidence.
NFL_COLUMNS = [
    'Game Time','Matchup',
    'Spr Action','Spr Pick','Spr OOS ROI','Spr ROI Floor',
    'H2H Action','H2H Pick','H2H OOS ROI','H2H ROI Floor',
    'Tot Action','Tot Pick','Tot OOS ROI','Tot ROI Floor',
    'Production Plays','Pathi State','Pathi Overlay','Edge Sources','System Trigger',
]

def _txt(v):
    if v is None:return ''
    a=str(v).strip()
    return '' if a.lower() in ('','nan','none','nat','null','<na>','—') else a

def _num(v):
    try:
        n=float(v)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError):return None

def _dt(x):
    if isinstance(x,datetime):return x.astimezone(timezone.utc) if x.tzinfo else None
    try:
        d=datetime.fromisoformat(_txt(x).replace('Z','+00:00'))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except (ValueError,TypeError,OverflowError):return None

def _auth(v):return v is True or _txt(v).lower() in ('1','1.0','true')

def _norm_market(v):
    t=_txt(v).lower()
    return {'spread':'spreads','ats':'spreads','moneyline':'h2h','ml':'h2h','total':'totals'}.get(t,t)

def _json_candidates(v):
    if isinstance(v,list):d=v
    elif isinstance(v,str) and v.strip():
        try:d=json.loads(v)
        except (ValueError,TypeError):return [],'INVALID_CANDIDATE_JSON'
    else:return [],None
    if not isinstance(d,list):return [],'CANDIDATES_NOT_LIST'
    return [x for x in d if isinstance(x,dict)],None

def _pct(v):
    n=_num(v)
    return f'{100*n:+.1f}%' if n is not None else '—'

def _team_label(row,c):
    side=_txt(c.get('side')).upper();m=_norm_market(c.get('market'))
    if m=='totals':name=side.title()
    else:
        name=_txt(c.get('team'))
        if not name:
            match=_txt(row.get('Matchup')).split(' @ ',1)
            if len(match)==2:name=match[1] if side=='HOME' else match[0]
    line=_num(c.get('price_spread'));odds=_num(c.get('price_odds'))
    if m=='h2h':return f'{name} ({odds:+g})' if odds is not None else name or '—'
    if line is not None and odds is not None:
        return f'{name} {line:g} ({odds:+g})' if m=='totals' else f'{name} {line:+g} ({odds:+g})'
    return name or '—'

def _quote_fresh(c,now,kickoff,limit_minutes=15):
    q=_dt(c.get('quote_utc') or c.get('price_observed_utc'))
    return bool(q and kickoff and q<=now<kickoff and (now-q).total_seconds()<=limit_minutes*60)

def _valid_action(row,c,audit,now,kick):
    """Fail-closed UI depiction of a backend-authorized wager, not promotion.

    Require exact match of published market, side, sportsbook, current line,
    current odds, certificate and backend ROI policy; quoted price is recent.
    """
    if not (_auth(row.get('V55_Bet_Authority')) and _auth(c.get('bet_authority')) and
            _txt(row.get('V55_Bet_Decision')).upper() in ACTIONS and
            _txt(c.get('status')).upper()==_txt(row.get('V55_Bet_Decision')).upper() and
            _norm_market(row.get('V55_Selected_Market'))==_norm_market(c.get('market')) and
            _txt(row.get('V55_Authority_Policy_Id'))==POLICY and
            _txt(audit.get('v55_authority_policy_id'))==POLICY and
            audit.get('production_v1_probabilities_used') is False and
            _txt(row.get('V55_Validated_Edge_Certificate_ID'))==_txt(c.get('certificate_id')) and
            _txt(c.get('certificate_id')) and
            _txt(row.get('V55_Research_Selected_Side')).upper()==_txt(c.get('side')).upper() and
            _txt(row.get('Backend_Quote_Book')).lower()==_txt(c.get('price_book')).lower() and
            _txt(row.get('Backend_Quote_Book')) and
            _quote_fresh(c,now,kick)):
        return False
    a,b=_num(row.get('Odds_American')),_num(c.get('price_odds'))
    if a is None or b is None or abs(a-b)>1e-7 or (-100<a<100):return False
    if _norm_market(c.get('market'))!='h2h':
        a,b=_num(row.get('Current_Spread')),_num(c.get('price_spread'))
        if a is None or b is None or abs(a-b)>1e-7:return False
    return _num(c.get('validated_roi')) is not None and _num(c.get('validated_roi_lower')) is not None

def read_published_latest(*,client=None,now=None,max_age_minutes=180):
    """Read the latest COMPLETE publication without importing legacy authority.

    Do not infer missing source fields, repair incomplete books, or re-authorize
    expired bets. A partial/modified snapshot is rejected entirely.
    """
    from google.cloud import bigquery
    import pandas as pd
    bq=client or bigquery.Client(project='sharplogger')
    sql=f'''SELECT snapshot_id,published_at,record_kind,payload_json,audit_json,
                   row_count,payload_sha256,publication_type
             FROM `{BOARD_TABLE}`
             WHERE snapshot_id=(SELECT snapshot_id FROM `{BOARD_TABLE}`
                                 WHERE record_kind='COMPLETE'
                                 ORDER BY published_at DESC LIMIT 1)'''
    records=list(bq.query(sql).result())
    if len(records)!=2 or {str(r.record_kind) for r in records}!={'DATA','COMPLETE'}:
        raise RuntimeError('INCOMPLETE_NCAAF_SNAPSHOT')
    data=next(r for r in records if r.record_kind=='DATA')
    done=next(r for r in records if r.record_kind=='COMPLETE')
    if (data.snapshot_id!=done.snapshot_id or data.payload_sha256!=done.payload_sha256 or
            data.row_count!=done.row_count or not data.payload_json or not done.audit_json or
            hashlib.sha256(data.payload_json.encode('utf-8')).hexdigest()!=done.payload_sha256):
        raise RuntimeError('CORRUPTED_OR_PARTIAL_NCAAF_SNAPSHOT')
    audit=json.loads(done.audit_json)
    values=json.loads(data.payload_json)
    if not isinstance(values,list) or len(values)!=int(done.row_count):
        raise RuntimeError('BAD_NCAAF_ROW_COUNT')
    if audit.get('status')!='COMPLETE_V55_INDEPENDENT_WEEKLY_SHADOW':
        raise RuntimeError('UNVERIFIED_NCAAF_PUBLICATION_STATUS')
    at=_dt(now) if now else datetime.now(timezone.utc)
    published=_dt(done.published_at)
    if not at or not published:raise RuntimeError('INVALID_NCAAF_SNAPSHOT_TIME')
    age=(at-published).total_seconds()/60
    audit={**audit,'snapshot_id':str(done.snapshot_id),'published_at_utc':published.isoformat(),
           'published_age_minutes':round(age,1),
           'publication_status':'PASS' if -5<=age<=max_age_minutes else 'STALE_MARKET'}
    frame=pd.DataFrame(values)
    if 'v55_authorized_bet_rows' in audit:
        claimed=int(audit.get('v55_authorized_bet_rows') or 0)
        observed=sum(_auth(r.get('V55_Bet_Authority')) for r in values)
        if claimed!=observed:raise RuntimeError('AUTHORIZED_BET_COUNT_DISAGREES_WITH_PUBLICATION_AUDIT')
        if bool(claimed)!=bool(audit.get('v55_bet_authority')):
            raise RuntimeError('AUTHORIZED_BET_FLAG_DISAGREES_WITH_PUBLICATION_AUDIT')
    return frame,audit

def nfl_shaped_game_board(weekly, audit=None, *, now=None):
    """One NFL-style row PER GAME + source-direct candidate/system history ledger.

    No synthesized candidate, probability, inferred CORE pick, or implied bet.
    """
    import pandas as pd
    if not isinstance(weekly,pd.DataFrame):raise TypeError('WEEKLY_MUST_BE_DATAFRAME')
    audit=audit or {}
    at=_dt(now) if now else datetime.now(timezone.utc)
    if not at:raise ValueError('INVALID_NOW')
    stale=audit.get('publication_status')=='STALE_MARKET'
    main=[];detail=[];errors=[];seen_games=set();metrics=Counter()
    for _,srow in weekly.iterrows():
        row=srow.to_dict();gid=_txt(row.get('Game_ID') or row.get('game_id'))
        kick=_dt(row.get('Game_Start_UTC') or row.get('kickoff_utc'))
        if kick and kick<=at:continue
        if gid in seen_games and gid:
            errors.append({'game_id':gid,'reason':'DUPLICATE_GAME_IN_PUBLISHED_DATA'})
            continue
        if gid:seen_games.add(gid)
        match=_txt(row.get('Matchup')) or '—'
        candidates,err=_json_candidates(row.get('V55_Unified_Candidates_JSON'))
        if err:errors.append({'game_id':gid,'reason':err})
        cand_by_market={m:[] for m in MARKETS};identity=set()
        for c in candidates:
            m=_norm_market(c.get('market'));side=_txt(c.get('side')).upper()
            book=_txt(c.get('price_book')).lower();line=_num(c.get('price_spread'))
            if m not in MARKETS or side not in (('OVER','UNDER') if m=='totals' else ('HOME','AWAY')):
                errors.append({'game_id':gid,'reason':'BAD_CANDIDATE_MARKET_SIDE'});continue
            ident=(gid,m,side,book,line)
            if ident in identity:
                errors.append({'game_id':gid,'reason':'DUPLICATE_CANDIDATE_IDENTITY'});continue
            identity.add(ident)
            c=dict(c)
            c['origins'] = c.get('origins') if isinstance(c.get('origins'),list) else []
            cand_by_market[m].append(c)
            authorized=bool(not stale and _valid_action(row,c,audit,at,kick))
            if _auth(c.get('bet_authority')) and not authorized:
                errors.append({'game_id':gid,'reason':'BACKEND_AUTHORITY_UNVERIFIED_IN_DISPLAY','market':m,'side':side})
            # Preserve backend history/certificate keys but do not infer them.
            detail.append({'Game ID':gid,'Game Time':kick.astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%a %b %d %I:%M %p') if kick else '—',
                'Matchup':match,'Market':m.upper(),'Side':side,'Team/Selection':_txt(c.get('team')) or side,
                'Book':book,'Line':line,'Odds':_num(c.get('price_odds')),
                'Backend Status':_txt(c.get('status')) or '—',
                'Published Action':_txt(c.get('status')) if authorized else 'RESEARCH — NOT A BET',
                'Bet Authority':int(authorized),'Originating Brains':' | '.join(c['origins']) or '—',
                'System IDs':' | '.join(c.get('system_ids') or [] ) or '—',
                'Qualified Components':' | '.join(c.get('qualified_family_keys') or []) or '—',
                'OOS ROI':_num(c.get('validated_roi')) if authorized else None,
                'OOS ROI Floor':_num(c.get('validated_roi_lower')) if authorized else None,
                'OOS Certificate':_txt(c.get('certificate_id')),
                'Hold': '—' if authorized else _txt(c.get('hold') or c.get('status') or row.get('V55_Authority_Hold_Reason')),
                'Quote UTC':_txt(c.get('quote_utc')),'Core Reference':_txt(c.get('core_reference_team')),
                'Opposes CORE':bool(c.get('opposes_core',False))})
            metrics['candidate_sides']+=1
            if authorized:metrics['authorized_candidates']+=1
            for f in c['origins']:metrics['brain_'+_txt(f).upper()]+=1
            if 'CORE' in c['origins'] and c.get('opposes_core'):
                errors.append({'game_id':gid,'reason':'CORE_ORIGIN_OPPOSES_CORE_CONTRADICTION'})
        approved=next((c for group in cand_by_market.values() for c in group
                       if not stale and _valid_action(row,c,audit,at,kick)),None)
        if _auth(row.get('V55_Bet_Authority')) and not approved:
            errors.append({'game_id':gid,'reason':'AUTHORIZED_GAME_HAS_NO_EXACT_CANDIDATE_MATCH'})
        market_cells={}
        for m in MARKETS:
            prefix=BR[m]
            cand=cand_by_market[m]
            valid=next((x for x in cand if not stale and _valid_action(row,x,audit,at,kick)),None)
            if valid:
                action=_txt(valid['status']).upper();picked=_team_label(row,valid)
                roi=_pct(valid.get('validated_roi'));floor=_pct(valid.get('validated_roi_lower'))
            elif cand:
                # Show backend research signals without promoting them to bets.
                action='RESEARCH';picked='—';roi='—';floor='—'
            else:
                action=('PASS' if m=='spreads' and _txt(row.get('V55_Bet_Decision')).upper()=='PASS' else '—')
                picked='—';roi='—';floor='—'
            market_cells.update({f'{prefix} Action':action,f'{prefix} Pick':picked,
                                 f'{prefix} OOS ROI':roi,f'{prefix} ROI Floor':floor})
        origins=sorted({f for group in cand_by_market.values() for c in group for f in c.get('origins',[])})
        systems=sorted({t for group in cand_by_market.values() for c in group for t in (c.get('system_ids') or [])})
        pathi=_txt(row.get('V55_Raw_Pathi'))
        date=kick.astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')).strftime('%a %b %d %I:%M %p') if kick else '—'
        act=(BR[_norm_market(approved['market'])]+': '+_team_label(row,approved)) if approved else '—'
        main.append({'Game Time':date,'Matchup':match,**market_cells,
             'Production Plays':act,'Pathi State':'TRACK' if pathi else '—',
             'Pathi Overlay':pathi or '—','Edge Sources':' | '.join(origins) or '—',
             'System Trigger':' | '.join(systems) or '—','_kick':kick,'_gid':gid})
        metrics['upcoming_games']+=1
        metrics['authorized_games']+=bool(approved)
        metrics['strong_bets']+=bool(approved and approved.get('status')=='STRONG BET')
        metrics['games_with_candidates']+=bool(candidates)
    mainframe=pd.DataFrame(main)
    if not mainframe.empty:
        mainframe['_sortable']=mainframe['_kick'].map(lambda d:d or datetime.max.replace(tzinfo=timezone.utc))
        mainframe=mainframe.sort_values(['_sortable','Matchup'],kind='stable').drop(columns=['_sortable','_kick','_gid'])
        mainframe=mainframe[NFL_COLUMNS].reset_index(drop=True)
    else:mainframe=pd.DataFrame(columns=NFL_COLUMNS)
    return mainframe,pd.DataFrame(detail),dict(metrics),errors

def _read_history_registry(uri=REGISTRY_URI):
    """Read-only certificate summaries; no automatic recovery or source fabrication."""
    from google.cloud import storage
    import hashlib
    b,p=uri[5:].split('/',1)
    doc=json.loads(storage.Client().bucket(b).blob(p).download_as_text())
    if doc.get('schema')!=REGISTRY_SCHEMA or doc.get('status')!='PASS_CERTIFIED_ROI_REGISTRY':
        raise ValueError('NO_PUBLISHED_VALIDATED_ROI_CERTIFICATES')
    cert=doc.get('certificates',[])
    calculated=hashlib.sha256(json.dumps(cert,sort_keys=True,separators=(',',':'),allow_nan=False,default=str).encode()).hexdigest()
    if doc.get('certificates_sha256')!=calculated:raise ValueError('CORRUPT_ROI_CERTIFICATE_REGISTRY')
    return doc

def render_published_v55_board(st):
    """Existing sharp_line_dashboard.py calls this exact function unchanged."""
    import pandas as pd
    st.subheader('NCAAF V5.5 — Validated Betting Edges')
    try:weekly,audit=read_published_latest()
    except Exception as exc:
        st.error('NCAAF verified publication unavailable. No V1 fallback.')
        st.code(f'{type(exc).__name__}: {str(exc)[:350]}')
        return
    if audit.get('v55_authority_policy_id')!=POLICY:
        st.warning('This board was published by a different/legacy authority policy. It cannot show ROI-authorized bets.')
    if audit.get('publication_status')=='STALE_MARKET':
        st.error('Published snapshot is stale. All betting actions are hidden until the backend refreshes.')
    try:main,details,stats,issues=nfl_shaped_game_board(weekly,audit)
    except Exception as exc:
        st.error('Unable to format backend publication. No bet calculation attempted.')
        st.code(f'{type(exc).__name__}: {str(exc)[:350]}')
        return
    # All date filters operate on backend kickoff, never displayed text.
    dates=['All']
    if not weekly.empty and 'Game_Start_UTC' in weekly:
        kick=pd.to_datetime(weekly['Game_Start_UTC'],utc=True,errors='coerce')
        dates += sorted({str(x) for x in kick.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d').dropna().unique()})
    select=st.selectbox('Game date',dates,key='ncaaf-v55-roi-date')
    if select!='All' and 'Game_Start_UTC' in weekly:
        target=weekly.copy()
        target['_et']=pd.to_datetime(target['Game_Start_UTC'],utc=True,errors='coerce').dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
        main,details,stats,issues=nfl_shaped_game_board(target[target['_et']==select].drop(columns=['_et']),audit)
    st.caption(f"Published: {audit.get('published_at_utc','—')} · snapshot: {audit.get('snapshot_id','—')} · ROI certificate count: {audit.get('v55_validated_edge_certificates',0)}")
    st.caption('Spread / H2H / Totals; backend ROI and conservative ROI floor only. Research signals are not bets. No predicted probabilities.')
    cs=st.columns(5)
    for col,label,value in zip(cs,('Upcoming games','Games with signals','Authorized bets','Strong bets','ROI certificates'),
                               (stats.get('upcoming_games',0),stats.get('games_with_candidates',0),stats.get('authorized_games',0),stats.get('strong_bets',0),audit.get('v55_validated_edge_certificates',0))):
        col.metric(label,int(value or 0))
    st.dataframe(main,hide_index=True,use_container_width=True)
    extra=st.selectbox('Additional NCAAF tables — load only when selected',
      ['Betting Board only','Research Candidate Sides','System Details & History','Publication / Price Diagnostics'],
      key='ncaaf-v55-roi-extra')
    if extra=='Research Candidate Sides':
        if details.empty:st.info('No backend-originated candidates in this publication.')
        else:st.dataframe(details,hide_index=True,use_container_width=True)
    if extra=='System Details & History':
        if details.empty:st.info('No published system events.')
        else:
            options=details.index.tolist()
            # Click any row to inspect the named historical edge; fall back to
            # a selectbox on Streamlit versions without dataframe selection.
            idx=None
            if hasattr(st,'dataframe'):
                try:
                    event=st.dataframe(details[['Matchup','Market','Side','Book','System IDs','OOS Certificate','Published Action']],
                        use_container_width=True,hide_index=True,on_select='rerun',
                        selection_mode='single-row',key='ncaaf-v55-roi-system-select')
                    chosen=event.selection.rows if event is not None else []
                    if chosen:idx=options[chosen[0]]
                except (TypeError,AttributeError):pass
            if idx is None:
                def label(idx):
                    r=details.loc[idx]
                    return f"{r['Matchup']} — {r['Market']} {r['Side']} / {r['Book']} / {r['System IDs'][:70]}"
                idx=st.selectbox('Select a system firing to inspect',options,format_func=label,key='ncaaf-v55-roi-history')
            selected=details.loc[idx]
            st.json({k:selected[k] for k in ['Game ID','Market','Side','Originating Brains','System IDs','Qualified Components','Backend Status','Published Action','Hold','OOS Certificate']})
            certificate=_txt(selected['OOS Certificate'])
            if not certificate:
                st.info('No validated OOS betting record for this live research signal.')
            else:
                try:
                    registry=_read_history_registry()
                    found=next((r for r in registry.get('certificates',[]) if r.get('id')==certificate),None)
                    if not found:st.error('Published live certificate is absent from the research registry.')
                    else:
                        st.dataframe(pd.DataFrame([{'Season':season,**result} for season,result in sorted(found.get('by_season',{}).items())]),hide_index=True,use_container_width=True)
                        st.json({k:found.get(k) for k in ('components','market','side','book','line','wins','losses','pushes','oos_unique_games','realized_roi','oos_roi_lower_adjusted','confidence_method','source_sha256')})
                except Exception as exc:st.warning(f'Certificate history unavailable: {type(exc).__name__}: {str(exc)[:140]}')
    if extra=='Publication / Price Diagnostics':
        st.json({k:audit.get(k) for k in ('status','publication_status','v55_authority_policy_id','v55_authority_probability_basis','v55_authorized_bet_rows','v55_validated_edge_certificates','v55_live_signal_rejections','v55_rejected_quote_pairs','v55_conflict_veto_rows')})
        if issues:st.dataframe(pd.DataFrame(issues),hide_index=True,use_container_width=True)
        elif not stats.get('authorized_games'):st.info('No published ROI-authorized bet; inspect backend research certificate status.')
