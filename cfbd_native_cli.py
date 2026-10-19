"""CFBD Native Discovery V1, CLI; no coupling to Production V1 or Core.

Examples:
 python cfbd_native_cli.py inventory --root path/to/archive
 python cfbd_native_cli.py plan --root path/to/archive --years 2013:2025 --families games,advanced,drives
 python cfbd_native_cli.py fetch --root path/to/archive --years 2013:2013 --families games --max-requests 1
 python cfbd_native_cli.py build --root path/to/archive --output research_output
 python cfbd_native_cli.py research --input research_output --output research_report
"""
import argparse
import json
import os
from pathlib import Path

from cfbd_native_archive import request_plan, inventory_archive, load_family,fetch_plan,RAW_FAMILIES
from cfbd_native_builder import write_research_exports
from cfbd_native_research import save_research


def year_range(spec):
    if ':' in spec:
        lo,hi=map(int,spec.split(':'))
        if lo>hi or hi-lo>30:raise ValueError('Expected start<=end, span<=30 years')
        return list(range(lo,hi+1))
    return sorted({int(v) for v in spec.split(',')})


def main(argv=None):
    p=argparse.ArgumentParser(description='Independent, source-grain CFBD research')
    sub=p.add_subparsers(dest='task',required=True)
    for name in ('inventory','plan','fetch','build'):
        s=sub.add_parser(name);s.add_argument('--root',required=True)
        s.add_argument('--years',default='2013:2025')
        if name in ('plan','fetch'):
            s.add_argument('--families',default='games,team_box,advanced,drives,plays')
        if name=='fetch':
            s.add_argument('--max-requests',type=int,default=1)
        if name=='build':s.add_argument('--output',required=True)
    s=sub.add_parser('research')
    s.add_argument('--input',required=True);s.add_argument('--output',required=True)
    s.add_argument('--earliest-year',type=int,default=2016)
    s.add_argument('--market-quotes',default=None)
    s.add_argument('--latest-year',type=int,default=2025)
    s.add_argument('--nonlinear',action='store_true',help='Slower nonlinear challenger, research only')
    args=p.parse_args(argv)
    if args.task in ('inventory','plan','fetch','build'):
        yrs=year_range(args.years)
        if args.task=='inventory':result=inventory_archive(args.root,yrs)
        if args.task=='build':result=write_research_exports(args.root,args.output,yrs)
        if args.task in ('plan','fetch'):
            families=[k.strip() for k in args.families.split(',') if k.strip()]
            games={y:load_family(args.root,'games',[y]) for y in yrs} if 'games' not in families else {}
            # Request plans for week sources additionally use any already archived games.
            if not games:
                games={y:load_family(args.root,'games',[y]) for y in yrs}
            result=request_plan(args.root,yrs,families,games)
            if args.task=='fetch':
                result=fetch_plan(args.root,result,api_key=os.getenv('CFBD_API_KEY',''),
                                  max_requests=args.max_requests)
    else:
        result=save_research(args.input,args.output,min_year=args.earliest_year,
                             market_file=args.market_quotes,latest_year=args.latest_year,
                             include_nonlinear=args.nonlinear)
    print(json.dumps(result,indent=2,default=str))

if __name__=='__main__':main()
