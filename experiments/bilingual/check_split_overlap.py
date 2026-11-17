"""Read-only cross-split character-shingle similarity diagnostic."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


def check(path):
    payload=path.read_bytes();rows=[json.loads(line) for line in payload.splitlines()]
    def shingles(text):
        text=re.sub(r'\s+','',text)
        return {text[i:i+4] for i in range(max(0,len(text)-3))}
    construction=[shingles(r['zh']) for r in rows if r['split']=='construction']
    heldout=[shingles(r['zh']) for r in rows if r['split'] in ('validation','sealed')]
    index=collections.defaultdict(list)
    for i,items in enumerate(construction):
        for item in items:index[item].append(i)
    maxima=[]
    for items in heldout:
        counts=collections.Counter(i for item in items for i in index[item])
        maxima.append(max((count/(len(items)+len(construction[i])-count) for i,count in counts.items()),default=0))
    return {'input_sha256':hashlib.sha256(payload).hexdigest(),'construction':len(construction),'heldout':len(heldout),
            'max_jaccard':max(maxima,default=0),'at_least_0_5':sum(x>=0.5 for x in maxima),
            'at_least_0_8':sum(x>=0.8 for x in maxima),
            'scope':'Whitespace-stripped character four-gram Jaccard; lexical screening only, not semantic deduplication.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);args=parser.parse_args()
    print(json.dumps(check(args.input),indent=2))
