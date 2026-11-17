"""Measure approximate title overlap without declaring semantic duplicates."""
import hashlib
import argparse
import json
import statistics
from pathlib import Path

from .build_author_recovery import key
from .verify_author_recovery import verify


def grams(value):
    text=key(value).replace(' ','')
    return {text[i:i+4] for i in range(len(text)-3)} if len(text)>=4 else set()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('artifacts/openalex-title-overlap-v2'));args=parser.parse_args()
    root=Path('benchmarks/openalex-author-recovery-v1');verify(root)
    read=lambda name:[json.loads(line) for line in (root/name).read_text().splitlines()]
    history=[r for r in read('source-records.jsonl') if r['period']=='history'];queries=read('queries.jsonl')
    labels={r['query_id']:set(r['known_author_ids']) for r in read('qrels.jsonl')}
    candidates=[(r,grams(r['title'])) for r in history];scores=[];shared_author=0;empty=0
    for q in queries:
        a=grams(q['text'])
        if not a:empty+=1;continue
        best=(-1.,None)
        for row,b in candidates:
            score=len(a&b)/len(a|b) if a|b else 0.
            if score>best[0] or (score==best[0] and (best[1] is None or row['id']<best[1]['id'])):best=(score,row)
        scores.append(best[0]);shared_author+=bool(set(best[1]['authors'])&labels[q['id']])
    report={'queries':len(queries),'scored_queries':len(scores),'empty_ngram_queries':empty,'max_jaccard':max(scores) if scores else None,
            'mean_max_jaccard':statistics.mean(scores) if scores else None,
            'queries_at_or_above_0_5':sum(x>=.5 for x in scores),'queries_at_or_above_0_8':sum(x>=.8 for x in scores),
            'best_overlap_source_shares_known_author':shared_author,
            'dataset_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
            'scope':'Whitespace-stripped normalized character 4-gram Jaccard; best of selected history titles per query. Not a semantic duplicate adjudication or proof of absence of leakage. A shared author and topical overlap are expected in this task.'}
    out=args.output;out.mkdir(parents=True,exist_ok=False);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
