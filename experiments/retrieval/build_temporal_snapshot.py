"""Create a conservative dated evidence corpus before indexing."""
import argparse
import json
from pathlib import Path
from expert_search.metadata.temporal_snapshot import build_temporal_snapshot


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--as-of',required=True,help='Inclusive ISO date, YYYY-MM-DD')
    args=parser.parse_args()
    print(json.dumps(build_temporal_snapshot(args.corpus,args.output,args.as_of),indent=2))
