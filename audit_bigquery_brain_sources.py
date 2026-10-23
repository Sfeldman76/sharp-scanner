"""Read-only BigQuery schema discovery; prints possible historical brain sources.

Does NOT infer system signals, fill missing history, or assume a table contains
pregame-vintage data. Useful before wiring the real V5.5 Heavy Research adapter.
"""
from __future__ import annotations
import argparse
import json
import re

PATTERNS={
 "game_id":re.compile(r"^(game_?id|source_?game_?id)$",re.I),
 "season":re.compile(r"^(season|year)$",re.I),
 "pregame_vintage":re.compile(r"(asof|as_of|created_at|event_at|prediction_at|quote_at|snapshot_at|source_updated_at|frozen_at|cutoff)",re.I),
 "signal_or_side":re.compile(r"(rule|system|trigger|signal|side|pick|recommend|brain|pathi|big_al|stat|miner|pt_)",re.I),
 "coverage":re.compile(r"(eligible|evaluation|coverage|source_hash|vintage|input_hash|status)",re.I),
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',default='sharplogger')
    p.add_argument('--dataset',default='sharp_data')
    p.add_argument('--out',default='')
    args=p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+",args.project) or not re.fullmatch(r"[A-Za-z0-9_]+",args.dataset):
        raise ValueError("Invalid BigQuery project/dataset identifier")
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise SystemExit("Install google-cloud-bigquery in the authorized Cloud Run research environment") from exc
    client=bigquery.Client(project=args.project)
    q=f"""SELECT table_name, column_name, data_type
            FROM `{args.project}.{args.dataset}.INFORMATION_SCHEMA.COLUMNS`
            WHERE REGEXP_CONTAINS(LOWER(table_name),
                   r'ncaaf|cfbd|miner|pathi|big_al|bigal|prediction|system|signal|market|lines|stat|consensus')
            ORDER BY table_name, ordinal_position"""
    tables={}
    for row in client.query(q).result():
        obj=tables.setdefault(row.table_name,{"columns":[],"matches":{k:[] for k in PATTERNS}})
        obj['columns'].append({"name":row.column_name,"type":row.data_type})
        for kind,pattern in PATTERNS.items():
            if pattern.search(row.column_name):obj['matches'][kind].append(row.column_name)
    report={"status":"READ_ONLY_SCHEMA_INSPECTION_NOT_VALIDATED_HISTORY",
            "location":f"{args.project}.{args.dataset}","tables":tables,
            "blocked_shortcuts":["Never synthesize unknown Big Al/Pathi historical non-picks",
              "Never call retrospective CFBD opening prices original-timestamp certified",
              "Never infer event source time from replay run timestamp",
              "Never treat a rule that did not exist at game time as a live historical pick",
              "Do not include 2026 outcomes in historical system or pair selection"]}
    txt=json.dumps(report,indent=2)
    if args.out:
        with open(args.out,'w') as f:f.write(txt+'\n')
    else:print(txt)
if __name__=='__main__':main()
