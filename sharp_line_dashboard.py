def detect_sharp_moves(current, previous, sport_key, SHARP_BOOKS, REC_BOOKS, BOOKMAKER_REGIONS):
    def normalize_label(label):
        return str(label).strip().lower().replace('.0', '')

    rows, sharp_audit_rows, rec_lines = [], [], []
    sharp_limit_map = defaultdict(lambda: defaultdict(list))
    sharp_lines, sharp_side_flags, sharp_metrics_map = {}, {}, {}
    snapshot_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    previous_map = {g['id']: g for g in previous} if isinstance(previous, list) else previous or {}

    for game in current:
        game_name = f"{game['home_team']} vs {game['away_team']}"
        event_date = pd.to_datetime(game.get("commence_time")).strftime("%Y-%m-%d") if "commence_time" in game else ""
        gid = game['id']
        prev_game = previous_map.get(gid, {})

        for book in game.get('bookmakers', []):
            book_key = book['key']
            for market in book.get('markets', []):
                mtype = market['key']
                for o in market.get('outcomes', []):
                    label = normalize_label(o['name'])
                    val = o.get('point') if mtype != 'h2h' else o.get('price')
                    if val is None:
                        continue
                    limit = o.get('bet_limit') if book_key in SHARP_BOOKS else None

                    entry = {
                        'Sport': sport_key, 'Time': snapshot_time, 'Game': game_name,
                        'Market': mtype, 'Outcome': label, 'Bookmaker': book['title'],
                        'Book': book_key, 'Value': val, 'Limit': limit,
                        'Old Value': None, 'Delta': None, 'Event_Date': event_date,
                        'Region': BOOKMAKER_REGIONS.get(book_key, 'unknown'),
                    }

                    if prev_game:
                        for prev_b in prev_game.get('bookmakers', []):
                            if prev_b['key'] == book_key:
                                for prev_m in prev_b.get('markets', []):
                                    if prev_m['key'] == mtype:
                                        for prev_o in prev_m.get('outcomes', []):
                                            if normalize_label(prev_o['name']) == label:
                                                prev_val = prev_o.get('point') if mtype != 'h2h' else prev_o.get('price')
                                                if prev_val is not None:
                                                    entry['Old Value'] = prev_val
                                                    entry['Delta'] = round(val - prev_val, 2)

                    if book_key in SHARP_BOOKS:
                        sharp_lines[(game_name, mtype, label)] = entry
                        sharp_limit_map[(game_name, mtype)][label].append((limit or 0, val, entry.get("Old Value")))
                    elif book_key in REC_BOOKS:
                        rec_lines.append(entry)

    for (game_name, mtype), label_map in sharp_limit_map.items():
        scores = {}
        label_signals = {}
        for label, entries in label_map.items():
            move_signal = limit_jump = prob_shift = time_score = 0
            for limit, curr, old in entries:
                if old is not None and curr is not None:
                    if mtype == 'totals':
                        if 'under' in label and curr < old: move_signal += 1
                        elif 'over' in label and curr > old: move_signal += 1
                    elif mtype == 'spreads' and abs(curr) > abs(old): move_signal += 1
                    elif mtype == 'h2h':
                        imp_now, imp_old = implied_prob(curr), implied_prob(old)
                        if imp_now and imp_old and imp_now > imp_old: prob_shift += 1
                if limit and limit >= 5000: limit_jump += 1
                hour = datetime.now().hour
                time_score += 1.0 if 6 <= hour <= 11 else 0.5 if hour <= 15 else 0.2

            score = 2 * move_signal + 2 * limit_jump + 1.5 * time_score + 1.0 * prob_shift
            scores[label] = score
            label_signals[label] = {
                'Sharp_Move_Signal': move_signal,
                'Sharp_Limit_Jump': limit_jump,
                'Sharp_Time_Score': time_score,
                'Sharp_Prob_Shift': prob_shift
            }

        if scores:
            best_label = max(scores, key=scores.get)
            sharp_side_flags[(game_name, mtype, best_label)] = 1
            sharp_metrics_map[(game_name, mtype, best_label)] = label_signals[best_label]

       for rec in rec_lines:
        rec_label = normalize_label(rec['Outcome'])
        market_type = rec['Market']
        rec_key = (rec['Game'], market_type, rec_label)

        if not sharp_side_flags.get(rec_key, 0):
            continue

        sharp = sharp_lines.get(rec_key)
        if not sharp:
            continue

        metrics = sharp_metrics_map.get(rec_key, {})
        row = rec.copy()
        row.update({
            'Ref Sharp Value': sharp['Value'],
            'SHARP_SIDE_TO_BET': 1,
            'SharpBetScore': round(
                2.0 * metrics.get('Sharp_Move_Signal', 0) +
                2.0 * metrics.get('Sharp_Limit_Jump', 0) +
                1.5 * metrics.get('Sharp_Time_Score', 0) +
                1.0 * metrics.get('Sharp_Prob_Shift', 0), 2
            ),
            'Sharp_Move_Signal': metrics.get('Sharp_Move_Signal', 0),
            'Sharp_Limit_Jump': metrics.get('Sharp_Limit_Jump', 0),
            'Sharp_Time_Score': metrics.get('Sharp_Time_Score', 0),
            'Sharp_Prob_Shift': metrics.get('Sharp_Prob_Shift', 0)
        })
        rows.append(row)
         
    df = pd.DataFrame(rows)
    if df.empty:
        return df, pd.DataFrame()

    df['Delta'] = pd.to_numeric(df['Delta vs Sharp'], errors='coerce')
    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce').fillna(0)
    df['Limit_Jump'] = (df['Limit'] >= 2500).astype(int)
    df['Sharp_Timing'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour.apply(
        lambda h: 1.0 if 6 <= h <= 11 else 0.5 if h <= 15 else 0.2
    )
    df['Limit_Max'] = df.groupby(['Game', 'Market'])['Limit'].transform('max')
    df['Limit_Min'] = df.groupby(['Game', 'Market'])['Limit'].transform('min')
    df['Limit_Imbalance'] = df['Limit_Max'] - df['Limit_Min']
    df['Asymmetry_Flag'] = (df['Limit_Imbalance'] >= 2500).astype(int)

    print(f"✅ Final sharp-backed rows: {len(df)}")
    return df, pd.DataFrame(sharp_audit_rows)
